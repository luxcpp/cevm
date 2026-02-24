// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file block_stm.cu
/// CUDA port of metal/block_stm.metal — full implementation.
///
/// Implements the entire Block-STM optimistic concurrency control loop on GPU:
///   - MvMemory as a GPU hash table with version chains
///   - Scheduler with atomic counters (execution_idx, validation_idx)
///   - Execute -> validate -> re-execute cycle with zero CPU intervention
///
/// Each GPU thread is an independent worker that:
///   1. Atomically grabs a tx from execution_idx or validation_idx
///   2. Executes the tx, writing to MvMemory
///   3. Validates the tx's read-set against MvMemory
///   4. On conflict: increments incarnation, re-queues for execution
///
/// CUDA mapping notes (vs Metal source):
///   - `device atomic_uint*` -> raw `uint32_t*`, accessed via atomicCAS/atomicAdd.
///   - Metal `atomic_compare_exchange_weak_explicit(ptr, &expected, desired)` returns
///     true on success and writes the observed value to `expected` on failure.
///     The CUDA equivalent is `old = atomicCAS(ptr, expected, desired)`; success when
///     `old == expected`. We expose a small helper `cas_weak` to keep semantics
///     equivalent and reuse the Metal control flow verbatim.
///   - Metal `memory_order_relaxed` -> default CUDA atomic ordering. We add
///     `__threadfence()` only where the Metal kernel relies on cross-warp
///     visibility (none here — Block-STM ordering is established through atomics).
///   - One thread per worker; same dispatch shape as Metal.

#include <cstdint>
#include <cuda_runtime.h>

namespace evm::gpu::cuda
{

// =============================================================================
// Constants (must match metal/block_stm.metal)
// =============================================================================

__device__ static constexpr uint32_t MAX_READS_PER_TX    = 64;
__device__ static constexpr uint32_t MAX_WRITES_PER_TX   = 64;
__device__ static constexpr uint32_t MV_TABLE_SIZE       = 65536;
__device__ static constexpr uint32_t MV_TABLE_MASK       = MV_TABLE_SIZE - 1;
__device__ static constexpr uint32_t MAX_INCARNATIONS    = 16;
__device__ static constexpr uint32_t MAX_SCHEDULER_LOOPS = 65536;
__device__ static constexpr uint32_t VERSION_BASE_STATE  = 0xFFFFFFFFu;
__device__ static constexpr uint32_t MV_EMPTY            = 0xFFFFFFFFu;

// =============================================================================
// Data structures (must match host-side layouts in block_stm_host.hpp exactly)
// =============================================================================

struct DeviceTransaction
{
    uint8_t  from[20];
    uint8_t  to[20];
    uint64_t gas_limit;
    uint64_t value;
    uint64_t nonce;
    uint64_t gas_price;
    uint32_t calldata_offset;
    uint32_t calldata_size;
};

struct DeviceAccountState
{
    uint8_t  address[20];
    uint32_t _pad;
    uint64_t nonce;
    uint64_t balance;
    uint8_t  code_hash[32];
    uint32_t code_size;
    uint32_t _pad2;
};

struct DeviceMvEntry
{
    uint32_t tx_index;     // MV_EMPTY (0xFFFFFFFF) means slot is empty
    uint32_t incarnation;
    uint8_t  address[20];
    uint32_t _pad;
    uint8_t  slot[32];
    uint8_t  value[32];
    uint32_t is_estimate;  // 1 = speculative, may be invalid
    uint32_t _pad2;
};

struct DeviceTxState
{
    uint32_t incarnation;
    uint32_t validated;
    uint32_t executed;
    uint32_t status;
    uint64_t gas_used;
    uint32_t read_count;
    uint32_t write_count;
};

struct DeviceReadSetEntry
{
    uint8_t  address[20];
    uint32_t _pad;
    uint8_t  slot[32];
    uint32_t read_tx_index;
    uint32_t read_incarnation;
};

struct DeviceWriteSetEntry
{
    uint8_t  address[20];
    uint32_t _pad;
    uint8_t  slot[32];
    uint8_t  value[32];
};

struct DeviceBlockStmResult
{
    uint64_t gas_used;
    uint32_t status;
    uint32_t incarnation;
};

struct DeviceBlockStmParams
{
    uint32_t num_txs;
    uint32_t max_iterations;
};

// =============================================================================
// Atomic helpers — mirror Metal's CAS-with-expected semantics on top of
// CUDA's atomicCAS-which-returns-old-value primitive.
// =============================================================================

/// Mimic Metal's atomic_compare_exchange_weak_explicit:
///   on success: *ptr was 'expected', is now 'desired', returns true.
///   on failure: *ptr was something else, *expected updated to observed, returns false.
__device__ __forceinline__ bool cas_weak(uint32_t* ptr, uint32_t& expected, uint32_t desired)
{
    uint32_t old = atomicCAS(ptr, expected, desired);
    if (old == expected)
        return true;
    expected = old;
    return false;
}

// =============================================================================
// Hash function for MvMemory table lookup (FNV-1a)
// =============================================================================

__device__ __forceinline__ uint32_t mv_hash(
    uint32_t tx_index, const uint8_t* address, const uint8_t* slot)
{
    uint32_t hash = 2166136261u;
    hash ^= (tx_index & 0xFF);          hash *= 16777619u;
    hash ^= ((tx_index >> 8) & 0xFF);   hash *= 16777619u;
    hash ^= ((tx_index >> 16) & 0xFF);  hash *= 16777619u;
    hash ^= ((tx_index >> 24) & 0xFF);  hash *= 16777619u;
    #pragma unroll
    for (int i = 0; i < 20; ++i)
    {
        hash ^= address[i];
        hash *= 16777619u;
    }
    #pragma unroll
    for (int i = 0; i < 32; ++i)
    {
        hash ^= slot[i];
        hash *= 16777619u;
    }
    return hash & MV_TABLE_MASK;
}

// =============================================================================
// Byte comparators
// =============================================================================

__device__ __forceinline__ bool addr_eq(const uint8_t* a, const uint8_t* b)
{
    #pragma unroll
    for (int i = 0; i < 20; ++i)
        if (a[i] != b[i]) return false;
    return true;
}

__device__ __forceinline__ bool slot_eq(const uint8_t* a, const uint8_t* b)
{
    #pragma unroll
    for (int i = 0; i < 32; ++i)
        if (a[i] != b[i]) return false;
    return true;
}

// =============================================================================
// MvMemory operations (lock-free via atomics)
// =============================================================================

/// Write a version entry to MvMemory. Open-addressed linear probing with CAS
/// on tx_index to claim an empty slot; updates in place if (tx, addr, slot)
/// already exists in the table.
__device__ void mv_write(
    DeviceMvEntry* table,
    uint32_t       tx_index,
    uint32_t       incarnation,
    const uint8_t* address,
    const uint8_t* slot,
    const uint8_t* value)
{
    uint32_t h = mv_hash(tx_index, address, slot);

    for (uint32_t probe = 0; probe < 256; ++probe)
    {
        uint32_t idx = (h + probe) & MV_TABLE_MASK;
        DeviceMvEntry& entry = table[idx];

        // Snapshot current tx_index using atomicAdd-of-zero (CUDA's "atomic load").
        uint32_t current = atomicAdd(&entry.tx_index, 0u);

        // Existing entry for our (tx, addr, slot) -> update in place.
        if (current == tx_index && addr_eq(entry.address, address) && slot_eq(entry.slot, slot))
        {
            entry.incarnation = incarnation;
            #pragma unroll
            for (int i = 0; i < 32; ++i)
                entry.value[i] = value[i];
            // Clear is_estimate atomically.
            atomicExch(&entry.is_estimate, 0u);
            return;
        }

        // Empty slot -> try to claim via CAS.
        if (current == MV_EMPTY)
        {
            uint32_t expected = MV_EMPTY;
            if (cas_weak(&entry.tx_index, expected, tx_index))
            {
                entry.incarnation = incarnation;
                #pragma unroll
                for (int i = 0; i < 20; ++i)
                    entry.address[i] = address[i];
                entry._pad = 0;
                #pragma unroll
                for (int i = 0; i < 32; ++i)
                    entry.slot[i] = slot[i];
                #pragma unroll
                for (int i = 0; i < 32; ++i)
                    entry.value[i] = value[i];
                atomicExch(&entry.is_estimate, 0u);
                return;
            }
            // CAS failed: another thread won this slot, fall through to next probe.
        }
        // Otherwise: occupied by another (tx, addr, slot); keep probing.
    }
    // Table full -- should not happen with proper sizing.
}

/// Read the latest valid value written by a tx with index < reader_tx_index.
/// Returns true if a valid version was found; out_value/out_tx_index/out_incarnation
/// describe the version. We scan candidates [0, reader_tx_index) and probe the
/// hash table for each — matching the Metal kernel semantics 1:1.
__device__ bool mv_read(
    const DeviceMvEntry* table,
    uint32_t             reader_tx_index,
    const uint8_t*       address,
    const uint8_t*       slot,
    uint8_t*             out_value,
    uint32_t&            out_tx_index,
    uint32_t&            out_incarnation)
{
    uint32_t best_tx  = VERSION_BASE_STATE;
    uint32_t best_inc = 0;
    bool found = false;

    for (uint32_t candidate = 0; candidate < reader_tx_index; ++candidate)
    {
        uint32_t h = mv_hash(candidate, address, slot);

        for (uint32_t probe = 0; probe < 64; ++probe)
        {
            uint32_t idx = (h + probe) & MV_TABLE_MASK;
            const DeviceMvEntry& entry = table[idx];

            // Atomic load of tx_index (relaxed).
            uint32_t etx = atomicAdd(
                const_cast<uint32_t*>(&entry.tx_index), 0u);
            if (etx == MV_EMPTY)
                break;  // empty slot stops probing

            if (etx == candidate &&
                addr_eq(entry.address, address) &&
                slot_eq(entry.slot, slot))
            {
                uint32_t est = atomicAdd(
                    const_cast<uint32_t*>(&entry.is_estimate), 0u);
                if (est == 0 && candidate > best_tx)
                {
                    best_tx  = candidate;
                    best_inc = entry.incarnation;
                    #pragma unroll
                    for (int i = 0; i < 32; ++i)
                        out_value[i] = entry.value[i];
                    found = true;
                }
                break;  // entry for this candidate processed, advance candidate
            }
        }
    }

    out_tx_index    = best_tx;
    out_incarnation = best_inc;
    return found;
}

/// Mark all entries written by tx_index as estimates (speculative, pending re-exec).
__device__ void mv_mark_estimate(DeviceMvEntry* table, uint32_t tx_index)
{
    for (uint32_t i = 0; i < MV_TABLE_SIZE; ++i)
    {
        uint32_t etx = atomicAdd(&table[i].tx_index, 0u);
        if (etx == tx_index)
            atomicExch(&table[i].is_estimate, 1u);
    }
}

/// Validate a read: check if the version at (address, slot) for reader_tx_index
/// still matches the version originally read.
__device__ bool mv_validate_read(
    const DeviceMvEntry* table,
    uint32_t             reader_tx_index,
    const uint8_t*       address,
    const uint8_t*       slot,
    uint32_t             expected_tx_index,
    uint32_t             expected_incarnation)
{
    uint8_t  dummy[32];
    uint32_t found_tx, found_inc;
    bool found = mv_read(table, reader_tx_index, address, slot,
                         dummy, found_tx, found_inc);

    if (!found)
        return expected_tx_index == VERSION_BASE_STATE;

    return found_tx == expected_tx_index && found_inc == expected_incarnation;
}

// =============================================================================
// Block-STM scheduler kernel
// =============================================================================

/// Main Block-STM kernel. Each GPU thread is an independent worker.
///
/// sched_state layout:
///   [0] execution_idx   — atomic counter for next tx to execute
///   [1] validation_idx  — atomic counter for next tx to validate
///   [2] done_count      — number of validated txs
///   [3] abort_flag      — set to 1 to terminate all workers
extern "C" __global__ void block_stm_execute_kernel(
    const DeviceTransaction*  __restrict__ txs,
    DeviceMvEntry*            __restrict__ mv_memory,
    uint32_t*                 __restrict__ sched_state,
    DeviceTxState*            __restrict__ tx_states,
    DeviceReadSetEntry*       __restrict__ read_sets,
    DeviceWriteSetEntry*      __restrict__ write_sets,
    const DeviceAccountState* __restrict__ /*base_state*/,
    DeviceBlockStmResult*     __restrict__ results,
    DeviceBlockStmParams                   params)
{
    const uint32_t num_txs = params.num_txs;
    uint32_t loops = 0;

    while (loops < MAX_SCHEDULER_LOOPS)
    {
        ++loops;

        // ---- Fast-path completion check ------------------------------------
        uint32_t done = atomicAdd(&sched_state[2], 0u);
        if (done >= num_txs)
        {
            bool all_valid = true;
            for (uint32_t i = 0; i < num_txs; ++i)
                if (tx_states[i].validated != 1) { all_valid = false; break; }
            if (all_valid) break;
        }

        uint32_t aborted = atomicAdd(&sched_state[3], 0u);
        if (aborted) break;

        // ---- Try EXECUTION task --------------------------------------------
        uint32_t exec_idx = atomicAdd(&sched_state[0], 1u);
        if (exec_idx < num_txs)
        {
            uint32_t cur_incarnation = tx_states[exec_idx].incarnation;

            if (cur_incarnation >= MAX_INCARNATIONS)
            {
                results[exec_idx].status = 3;  // error
                results[exec_idx].gas_used = 0;
                tx_states[exec_idx].validated = 1;
                atomicAdd(&sched_state[2], 1u);
                continue;
            }

            const DeviceTransaction& tx = txs[exec_idx];
            DeviceTxState& ts = tx_states[exec_idx];

            // Slot 0 = balance slot — same simplified balance-transfer model
            // as metal/block_stm.metal; full EVM execution is dispatched via
            // a separate kernel that feeds back into MvMemory.
            uint8_t sender_slot[32] = {};
            uint8_t value_bytes[32] = {};
            // Big-endian encode tx.value into low 8 bytes of 32.
            value_bytes[24] = (uint8_t)((tx.value >> 56) & 0xFF);
            value_bytes[25] = (uint8_t)((tx.value >> 48) & 0xFF);
            value_bytes[26] = (uint8_t)((tx.value >> 40) & 0xFF);
            value_bytes[27] = (uint8_t)((tx.value >> 32) & 0xFF);
            value_bytes[28] = (uint8_t)((tx.value >> 24) & 0xFF);
            value_bytes[29] = (uint8_t)((tx.value >> 16) & 0xFF);
            value_bytes[30] = (uint8_t)((tx.value >> 8)  & 0xFF);
            value_bytes[31] = (uint8_t)(tx.value         & 0xFF);

            // Write sender balance.
            uint8_t from_addr[20];
            #pragma unroll
            for (int i = 0; i < 20; ++i) from_addr[i] = tx.from[i];
            mv_write(mv_memory, exec_idx, cur_incarnation,
                     from_addr, sender_slot, value_bytes);

            uint32_t wi = exec_idx * MAX_WRITES_PER_TX;
            #pragma unroll
            for (int i = 0; i < 20; ++i) write_sets[wi].address[i] = tx.from[i];
            #pragma unroll
            for (int i = 0; i < 32; ++i) write_sets[wi].slot[i]    = sender_slot[i];
            #pragma unroll
            for (int i = 0; i < 32; ++i) write_sets[wi].value[i]   = value_bytes[i];

            // Read sender prior state.
            uint8_t  read_val[32];
            uint32_t read_tx, read_inc;
            bool has_prior = mv_read(mv_memory, exec_idx, from_addr, sender_slot,
                                     read_val, read_tx, read_inc);

            uint32_t ri = exec_idx * MAX_READS_PER_TX;
            #pragma unroll
            for (int i = 0; i < 20; ++i) read_sets[ri].address[i] = tx.from[i];
            #pragma unroll
            for (int i = 0; i < 32; ++i) read_sets[ri].slot[i]    = sender_slot[i];
            read_sets[ri].read_tx_index    = has_prior ? read_tx  : VERSION_BASE_STATE;
            read_sets[ri].read_incarnation = has_prior ? read_inc : 0u;

            // Receiver?
            bool has_to = false;
            #pragma unroll
            for (int i = 0; i < 20; ++i)
                if (tx.to[i] != 0) { has_to = true; break; }

            uint32_t wcount = 1;
            uint32_t rcount = 1;

            if (has_to)
            {
                uint8_t to_addr[20];
                #pragma unroll
                for (int i = 0; i < 20; ++i) to_addr[i] = tx.to[i];
                mv_write(mv_memory, exec_idx, cur_incarnation,
                         to_addr, sender_slot, value_bytes);

                #pragma unroll
                for (int i = 0; i < 20; ++i) write_sets[wi + 1].address[i] = tx.to[i];
                #pragma unroll
                for (int i = 0; i < 32; ++i) write_sets[wi + 1].slot[i]    = sender_slot[i];
                #pragma unroll
                for (int i = 0; i < 32; ++i) write_sets[wi + 1].value[i]   = value_bytes[i];
                wcount = 2;

                uint8_t  recv_val[32];
                uint32_t recv_tx, recv_inc;
                bool recv_prior = mv_read(mv_memory, exec_idx, to_addr, sender_slot,
                                          recv_val, recv_tx, recv_inc);
                #pragma unroll
                for (int i = 0; i < 20; ++i) read_sets[ri + 1].address[i] = tx.to[i];
                #pragma unroll
                for (int i = 0; i < 32; ++i) read_sets[ri + 1].slot[i]    = sender_slot[i];
                read_sets[ri + 1].read_tx_index    = recv_prior ? recv_tx  : VERSION_BASE_STATE;
                read_sets[ri + 1].read_incarnation = recv_prior ? recv_inc : 0u;
                rcount = 2;
            }

            // Intrinsic gas: 21000 baseline + 16/byte calldata (Metal simplification).
            uint64_t intrinsic_gas = 21000ULL;
            uint64_t calldata_gas  = (uint64_t)tx.calldata_size * 16ULL;
            uint64_t total_gas     = intrinsic_gas + calldata_gas;

            ts.gas_used    = total_gas;
            ts.status      = (total_gas <= tx.gas_limit) ? 0u : 2u;
            ts.read_count  = rcount;
            ts.write_count = wcount;
            ts.executed    = 1u;
            ts.validated   = 0u;

            results[exec_idx].gas_used    = total_gas;
            results[exec_idx].status      = ts.status;
            results[exec_idx].incarnation = cur_incarnation;

            continue;
        }

        // ---- Try VALIDATION task -------------------------------------------
        uint32_t val_idx = atomicAdd(&sched_state[1], 1u);
        if (val_idx < num_txs)
        {
            // Wait until target tx has been executed.
            if (tx_states[val_idx].executed == 0)
            {
                // Roll back our atomic claim if validation_idx hasn't moved further.
                uint32_t expected = val_idx + 1;
                cas_weak(&sched_state[1], expected, val_idx);
                continue;
            }

            uint32_t rcount   = tx_states[val_idx].read_count;
            uint32_t ri_base  = val_idx * MAX_READS_PER_TX;
            bool valid = true;

            for (uint32_t r = 0; r < rcount && r < MAX_READS_PER_TX; ++r)
            {
                const DeviceReadSetEntry& re = read_sets[ri_base + r];
                uint8_t addr[20], sl[32];
                #pragma unroll
                for (int i = 0; i < 20; ++i) addr[i] = re.address[i];
                #pragma unroll
                for (int i = 0; i < 32; ++i) sl[i]   = re.slot[i];

                if (!mv_validate_read(mv_memory, val_idx, addr, sl,
                                      re.read_tx_index, re.read_incarnation))
                {
                    valid = false;
                    break;
                }
            }

            if (valid)
            {
                tx_states[val_idx].validated = 1u;
                atomicAdd(&sched_state[2], 1u);
            }
            else
            {
                // Conflict: re-execute this tx.
                mv_mark_estimate(mv_memory, val_idx);

                tx_states[val_idx].incarnation += 1u;
                tx_states[val_idx].executed     = 0u;
                tx_states[val_idx].validated    = 0u;

                // Reset execution_idx down to val_idx.
                uint32_t expected_exec = atomicAdd(&sched_state[0], 0u);
                while (expected_exec > val_idx)
                {
                    if (cas_weak(&sched_state[0], expected_exec, val_idx))
                        break;
                }

                // Invalidate later validated txs and decrement done_count.
                for (uint32_t i = val_idx + 1; i < num_txs; ++i)
                {
                    if (tx_states[i].validated == 1u)
                    {
                        tx_states[i].validated = 0u;
                        atomicSub(&sched_state[2], 1u);
                    }
                }

                // Reset validation_idx down to val_idx.
                uint32_t expected_val = atomicAdd(&sched_state[1], 0u);
                while (expected_val > val_idx)
                {
                    if (cas_weak(&sched_state[1], expected_val, val_idx))
                        break;
                }
            }

            continue;
        }

        // No work available — re-check completion before spinning again.
        done = atomicAdd(&sched_state[2], 0u);
        if (done >= num_txs)
        {
            bool all_valid = true;
            for (uint32_t i = 0; i < num_txs; ++i)
                if (tx_states[i].validated != 1) { all_valid = false; break; }
            if (all_valid) break;
        }
        // Otherwise: spin and retry.
    }
}

// =============================================================================
// Host-callable launcher
// =============================================================================

extern "C" cudaError_t evm_cuda_block_stm_launch(
    const void*  d_txs,
    void*        d_mv_memory,
    void*        d_sched_state,
    void*        d_tx_states,
    void*        d_read_sets,
    void*        d_write_sets,
    const void*  d_base_state,
    void*        d_results,
    uint32_t     num_txs,
    uint32_t     max_iterations,
    cudaStream_t stream)
{
    if (num_txs == 0) return cudaSuccess;

    DeviceBlockStmParams params{};
    params.num_txs        = num_txs;
    params.max_iterations = max_iterations;

    // Worker count: at least 64 for occupancy, otherwise one worker per tx
    // (matches metal/block_stm_host.mm). One thread per worker.
    uint32_t num_workers = num_txs;
    if (num_workers < 64) num_workers = 64;

    constexpr uint32_t threads_per_block = 128;
    const uint32_t blocks =
        (num_workers + threads_per_block - 1) / threads_per_block;

    block_stm_execute_kernel<<<blocks, threads_per_block, 0, stream>>>(
        static_cast<const DeviceTransaction*>(d_txs),
        static_cast<DeviceMvEntry*>(d_mv_memory),
        static_cast<uint32_t*>(d_sched_state),
        static_cast<DeviceTxState*>(d_tx_states),
        static_cast<DeviceReadSetEntry*>(d_read_sets),
        static_cast<DeviceWriteSetEntry*>(d_write_sets),
        static_cast<const DeviceAccountState*>(d_base_state),
        static_cast<DeviceBlockStmResult*>(d_results),
        params);

    return cudaGetLastError();
}

}  // namespace evm::gpu::cuda
