// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file block_stm.metal
/// Metal compute shader for GPU-native Block-STM parallel execution.
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
/// Key: (tx_index, address, storage_slot)
/// Value: (incarnation, data, is_estimate)
///
/// Buffer layout:
///   [0] Transaction*     -- transaction descriptors
///   [1] MvEntry*         -- multi-version memory hash table
///   [2] atomic_uint*     -- scheduler state: [0]=execution_idx, [1]=validation_idx
///   [3] TxState*         -- per-tx state: incarnation, validated, read/write counts
///   [4] ReadSetEntry*    -- per-tx read sets (MAX_READS_PER_TX * num_txs)
///   [5] WriteSetEntry*   -- per-tx write sets (MAX_WRITES_PER_TX * num_txs)
///   [6] AccountState*    -- base account state (GPU-resident)
///   [7] BlockStmResult*  -- per-tx results (gas_used, status)
///   [8] BlockStmParams   -- constant params

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Constants
// =============================================================================

constant uint MAX_READS_PER_TX = 64;
constant uint MAX_WRITES_PER_TX = 64;
constant uint MV_TABLE_SIZE    = 65536;  // Power of 2 for hash table
constant uint MV_TABLE_MASK    = MV_TABLE_SIZE - 1;
constant uint MAX_INCARNATIONS = 16;     // Safety bound on re-executions
constant uint MAX_SCHEDULER_LOOPS = 65536;  // Prevent infinite loops

// Version sentinel for "read from base state" (no prior write in block)
constant uint VERSION_BASE_STATE = 0xFFFFFFFF;

// =============================================================================
// Data structures (must match host-side layouts exactly)
// =============================================================================

struct uint256 {
    ulong w[4];  // w[0] = low, w[3] = high
};

struct Transaction {
    uchar from[20];
    uchar to[20];
    ulong gas_limit;
    ulong value;
    ulong nonce;
    ulong gas_price;
    uint  calldata_offset;  // Into shared calldata buffer
    uint  calldata_size;
};

struct AccountState {
    uchar address[20];
    uint  _pad;
    ulong nonce;
    ulong balance;       // Simplified to uint64 for GPU (wei units)
    uchar code_hash[32]; // Keccak256 of code
    uint  code_size;
    uint  _pad2;
};

/// Multi-version memory entry in the GPU hash table.
/// Open addressing with linear probing.
struct MvEntry {
    uint  tx_index;      // Which tx wrote this (0xFFFFFFFF = empty)
    uint  incarnation;
    uchar address[20];
    uint  _pad;
    uchar slot[32];
    uchar value[32];     // Storage value
    uint  is_estimate;   // 1 = speculative, may be invalid
    uint  _pad2;
};

/// Per-transaction scheduler state.
struct TxState {
    uint incarnation;
    uint validated;       // 0 = not validated, 1 = validated
    uint executed;        // 0 = not executed, 1 = executed
    uint status;          // 0 = success, 1 = revert, 2 = oog, 3 = error
    ulong gas_used;
    uint read_count;
    uint write_count;
};

/// Read-set entry: records what version was read from MvMemory.
struct ReadSetEntry {
    uchar address[20];
    uint  _pad;
    uchar slot[32];
    uint  read_tx_index;    // tx_index of the version read
    uint  read_incarnation; // incarnation of the version read
};

/// Write-set entry: records what was written.
struct WriteSetEntry {
    uchar address[20];
    uint  _pad;
    uchar slot[32];
    uchar value[32];
};

struct BlockStmResult {
    ulong gas_used;
    uint  status;       // 0=success, 1=revert, 2=oog, 3=error
    uint  incarnation;  // Final incarnation
};

struct BlockStmParams {
    uint num_txs;
    uint max_iterations;  // Safety bound
};

// =============================================================================
// Hash function for MvMemory table lookup
// =============================================================================

/// FNV-1a hash over (tx_index, address[20], slot[32]) = 56 bytes.
/// Returns index into the MvEntry table.
inline uint mv_hash(uint tx_index, thread const uchar* address, thread const uchar* slot) {
    uint hash = 2166136261u;
    // Hash tx_index bytes
    hash ^= (tx_index & 0xFF);       hash *= 16777619u;
    hash ^= ((tx_index >> 8) & 0xFF);  hash *= 16777619u;
    hash ^= ((tx_index >> 16) & 0xFF); hash *= 16777619u;
    hash ^= ((tx_index >> 24) & 0xFF); hash *= 16777619u;
    // Hash address
    for (int i = 0; i < 20; i++) {
        hash ^= address[i];
        hash *= 16777619u;
    }
    // Hash slot
    for (int i = 0; i < 32; i++) {
        hash ^= slot[i];
        hash *= 16777619u;
    }
    return hash & MV_TABLE_MASK;
}

/// Location-only hash (for reads that scan all tx_index values).
inline uint loc_hash(thread const uchar* address, thread const uchar* slot) {
    uint hash = 2166136261u;
    for (int i = 0; i < 20; i++) {
        hash ^= address[i];
        hash *= 16777619u;
    }
    for (int i = 0; i < 32; i++) {
        hash ^= slot[i];
        hash *= 16777619u;
    }
    return hash;
}

/// Device-memory overloads for MvEntry fields.
inline uint mv_hash_device(uint tx_index, device const uchar* address, device const uchar* slot) {
    uint hash = 2166136261u;
    hash ^= (tx_index & 0xFF);       hash *= 16777619u;
    hash ^= ((tx_index >> 8) & 0xFF);  hash *= 16777619u;
    hash ^= ((tx_index >> 16) & 0xFF); hash *= 16777619u;
    hash ^= ((tx_index >> 24) & 0xFF); hash *= 16777619u;
    for (int i = 0; i < 20; i++) {
        hash ^= address[i];
        hash *= 16777619u;
    }
    for (int i = 0; i < 32; i++) {
        hash ^= slot[i];
        hash *= 16777619u;
    }
    return hash & MV_TABLE_MASK;
}

// =============================================================================
// MvMemory operations (device-memory, lock-free via atomics)
// =============================================================================

/// Compare 20-byte addresses in device memory.
inline bool addr_eq(device const uchar* a, thread const uchar* b) {
    for (int i = 0; i < 20; i++) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

/// Compare 32-byte slots in device memory.
inline bool slot_eq(device const uchar* a, thread const uchar* b) {
    for (int i = 0; i < 32; i++) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

/// Write a version entry to MvMemory.
/// Uses atomic CAS on tx_index to claim a slot (open addressing).
inline void mv_write(
    device MvEntry* table,
    uint tx_index,
    uint incarnation,
    thread const uchar* address,
    thread const uchar* slot,
    thread const uchar* value)
{
    uint h = mv_hash(tx_index, address, slot);

    for (uint probe = 0; probe < 256; probe++) {
        uint idx = (h + probe) & MV_TABLE_MASK;
        device MvEntry& entry = table[idx];

        // Try to claim empty slot or find existing entry for this (tx, addr, slot)
        uint expected = 0xFFFFFFFF;  // Empty marker
        device atomic_uint* tx_atomic = (device atomic_uint*)&entry.tx_index;

        // Check if this slot already has our entry
        uint current = atomic_load_explicit(tx_atomic, memory_order_relaxed);
        if (current == tx_index && addr_eq(entry.address, address) && slot_eq(entry.slot, slot)) {
            // Update existing entry
            entry.incarnation = incarnation;
            for (int i = 0; i < 32; i++) entry.value[i] = value[i];
            atomic_store_explicit((device atomic_uint*)&entry.is_estimate, 0u, memory_order_relaxed);
            return;
        }

        // Try to claim empty slot
        if (current == 0xFFFFFFFF) {
            if (atomic_compare_exchange_weak_explicit(tx_atomic, &expected, tx_index,
                    memory_order_relaxed, memory_order_relaxed)) {
                // Claimed. Fill in the entry.
                entry.incarnation = incarnation;
                for (int i = 0; i < 20; i++) entry.address[i] = address[i];
                entry._pad = 0;
                for (int i = 0; i < 32; i++) entry.slot[i] = slot[i];
                for (int i = 0; i < 32; i++) entry.value[i] = value[i];
                atomic_store_explicit((device atomic_uint*)&entry.is_estimate, 0u, memory_order_relaxed);
                return;
            }
            // CAS failed -- another thread took this slot. Re-read and retry.
        }
    }
    // Table full -- should not happen with proper sizing.
}

/// Read the latest valid value written by a tx with index < reader_tx_index.
/// Returns true if a valid version was found. Writes the version info to out_*.
inline bool mv_read(
    device const MvEntry* table,
    uint reader_tx_index,
    thread const uchar* address,
    thread const uchar* slot,
    thread uchar* out_value,
    thread uint& out_tx_index,
    thread uint& out_incarnation)
{
    // Scan the hash table for all entries matching (address, slot) with tx_index < reader_tx_index.
    // Find the one with the largest tx_index (latest write before us).
    uint best_tx = VERSION_BASE_STATE;
    uint best_inc = 0;
    bool found = false;

    // We need to scan because entries for different tx_index values hash to different slots.
    // Optimization: we search from tx_index = reader_tx_index - 1 down to 0.
    // For each candidate tx_index, probe the hash table directly.
    for (uint candidate = 0; candidate < reader_tx_index; candidate++) {
        // Compute hash for (candidate, address, slot)
        uint h = mv_hash(candidate, address, slot);

        for (uint probe = 0; probe < 64; probe++) {
            uint idx = (h + probe) & MV_TABLE_MASK;
            device const MvEntry& entry = table[idx];

            uint etx = atomic_load_explicit((device atomic_uint*)&entry.tx_index, memory_order_relaxed);
            if (etx == 0xFFFFFFFF) break;  // Empty slot, stop probing

            if (etx == candidate && addr_eq(entry.address, address) && slot_eq(entry.slot, slot)) {
                uint est = atomic_load_explicit((device atomic_uint*)&entry.is_estimate, memory_order_relaxed);
                if (est == 0 && candidate > best_tx) {
                    best_tx = candidate;
                    best_inc = entry.incarnation;
                    for (int i = 0; i < 32; i++) out_value[i] = entry.value[i];
                    found = true;
                }
                break;  // Found entry for this candidate, move to next
            }
        }
    }

    out_tx_index = best_tx;
    out_incarnation = best_inc;
    return found;
}

/// Mark all entries from tx_index as estimates (speculative, pending re-execution).
inline void mv_mark_estimate(device MvEntry* table, uint tx_index) {
    for (uint i = 0; i < MV_TABLE_SIZE; i++) {
        uint etx = atomic_load_explicit((device atomic_uint*)&table[i].tx_index, memory_order_relaxed);
        if (etx == tx_index) {
            atomic_store_explicit((device atomic_uint*)&table[i].is_estimate, 1u, memory_order_relaxed);
        }
    }
}

/// Validate a read: check if the version at (address, slot) for reader_tx_index
/// still matches the version that was originally read.
inline bool mv_validate_read(
    device const MvEntry* table,
    uint reader_tx_index,
    thread const uchar* address,
    thread const uchar* slot,
    uint expected_tx_index,
    uint expected_incarnation)
{
    uchar dummy_value[32];
    uint found_tx, found_inc;
    bool found = mv_read(table, reader_tx_index, address, slot, dummy_value, found_tx, found_inc);

    if (!found) {
        // No write exists now -- valid only if we also read from base state
        return expected_tx_index == VERSION_BASE_STATE;
    }

    return found_tx == expected_tx_index && found_inc == expected_incarnation;
}

// =============================================================================
// Block-STM GPU kernel
// =============================================================================

/// Main Block-STM kernel. Each GPU thread is an independent worker.
///
/// The algorithm:
///   loop {
///     1. Try to grab an execution task (atomic increment of execution_idx)
///     2. If no execution task, try validation (atomic increment of validation_idx)
///     3. If no work, spin-check for completion
///     4. Execute: run simplified tx logic, record reads/writes
///     5. Validate: check read-set against MvMemory
///     6. On conflict: mark estimates, re-queue for execution
///   }
///
/// This kernel handles the scheduling and MvMemory operations.
/// The actual EVM execution is simplified here (balance transfer only);
/// for full EVM, the host dispatches to the evm_kernel.metal per-tx
/// and feeds results back through MvMemory.
kernel void block_stm_execute(
    device const Transaction*  txs         [[buffer(0)]],
    device MvEntry*            mv_memory   [[buffer(1)]],
    device atomic_uint*        sched_state [[buffer(2)]],
    device TxState*            tx_states   [[buffer(3)]],
    device ReadSetEntry*       read_sets   [[buffer(4)]],
    device WriteSetEntry*      write_sets  [[buffer(5)]],
    device AccountState*       base_state  [[buffer(6)]],
    device BlockStmResult*     results     [[buffer(7)]],
    constant BlockStmParams&   params      [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    // Scheduler atomic indices:
    //   sched_state[0] = execution_idx (next tx to execute)
    //   sched_state[1] = validation_idx (next tx to validate)
    //   sched_state[2] = done_count (number of validated txs)
    //   sched_state[3] = abort_flag (set to 1 to stop all workers)

    const uint num_txs = params.num_txs;
    uint loops = 0;

    while (loops < MAX_SCHEDULER_LOOPS) {
        loops++;

        // Check if all done. Use done_count as a fast path, but verify
        // all tx_states[i].validated == 1 to guard against stale counts
        // from invalidation races.
        uint done = atomic_load_explicit(&sched_state[2], memory_order_relaxed);
        if (done >= num_txs) {
            bool all_valid = true;
            for (uint i = 0; i < num_txs; i++) {
                if (tx_states[i].validated != 1) { all_valid = false; break; }
            }
            if (all_valid) break;
        }

        // Check abort flag
        uint aborted = atomic_load_explicit(&sched_state[3], memory_order_relaxed);
        if (aborted) break;

        // ---- Try to get an EXECUTION task ----
        uint exec_idx = atomic_fetch_add_explicit(&sched_state[0], 1u, memory_order_relaxed);
        if (exec_idx < num_txs) {
            // Check if this tx actually needs (re-)execution
            uint cur_incarnation = tx_states[exec_idx].incarnation;

            if (cur_incarnation >= MAX_INCARNATIONS) {
                // Too many re-executions. Flag error and skip.
                results[exec_idx].status = 3;  // error
                results[exec_idx].gas_used = 0;
                tx_states[exec_idx].validated = 1;
                atomic_fetch_add_explicit(&sched_state[2], 1u, memory_order_relaxed);
                continue;
            }

            // === EXECUTE TRANSACTION ===
            // Simplified execution: balance transfer logic.
            // Full EVM execution would invoke the EVM interpreter here or
            // coordinate with evm_kernel.metal via a separate dispatch.

            device const Transaction& tx = txs[exec_idx];
            device TxState& ts = tx_states[exec_idx];

            // Record a write to sender balance (simplified: decrement by value + gas)
            uchar sender_slot[32] = {};  // slot 0 = balance
            uchar value_bytes[32] = {};
            // Encode the transfer value into 32 bytes (big-endian, simplified)
            value_bytes[24] = uchar((tx.value >> 56) & 0xFF);
            value_bytes[25] = uchar((tx.value >> 48) & 0xFF);
            value_bytes[26] = uchar((tx.value >> 40) & 0xFF);
            value_bytes[27] = uchar((tx.value >> 32) & 0xFF);
            value_bytes[28] = uchar((tx.value >> 24) & 0xFF);
            value_bytes[29] = uchar((tx.value >> 16) & 0xFF);
            value_bytes[30] = uchar((tx.value >> 8) & 0xFF);
            value_bytes[31] = uchar(tx.value & 0xFF);

            // Write to MvMemory: sender balance change
            uchar from_addr[20];
            for (int i = 0; i < 20; i++) from_addr[i] = tx.from[i];
            mv_write(mv_memory, exec_idx, cur_incarnation, from_addr, sender_slot, value_bytes);

            // Record in write set
            uint wi = exec_idx * MAX_WRITES_PER_TX;
            for (int i = 0; i < 20; i++) write_sets[wi].address[i] = tx.from[i];
            for (int i = 0; i < 32; i++) write_sets[wi].slot[i] = sender_slot[i];
            for (int i = 0; i < 32; i++) write_sets[wi].value[i] = value_bytes[i];

            // Read sender's previous state (for nonce/balance check in validation)
            uchar read_val[32];
            uint read_tx, read_inc;
            bool has_prior = mv_read(mv_memory, exec_idx, from_addr, sender_slot,
                                     read_val, read_tx, read_inc);

            // Record in read set
            uint ri = exec_idx * MAX_READS_PER_TX;
            for (int i = 0; i < 20; i++) read_sets[ri].address[i] = tx.from[i];
            for (int i = 0; i < 32; i++) read_sets[ri].slot[i] = sender_slot[i];
            read_sets[ri].read_tx_index = has_prior ? read_tx : VERSION_BASE_STATE;
            read_sets[ri].read_incarnation = has_prior ? read_inc : 0;

            // If tx has a recipient, write to receiver balance
            bool has_to = false;
            for (int i = 0; i < 20; i++) {
                if (tx.to[i] != 0) { has_to = true; break; }
            }

            uint wcount = 1;
            uint rcount = 1;

            if (has_to) {
                uchar to_addr[20];
                for (int i = 0; i < 20; i++) to_addr[i] = tx.to[i];
                mv_write(mv_memory, exec_idx, cur_incarnation, to_addr, sender_slot, value_bytes);

                // Record write
                for (int i = 0; i < 20; i++) write_sets[wi + 1].address[i] = tx.to[i];
                for (int i = 0; i < 32; i++) write_sets[wi + 1].slot[i] = sender_slot[i];
                for (int i = 0; i < 32; i++) write_sets[wi + 1].value[i] = value_bytes[i];
                wcount = 2;

                // Read receiver prior state
                uchar recv_val[32];
                uint recv_tx, recv_inc;
                bool recv_prior = mv_read(mv_memory, exec_idx, to_addr, sender_slot,
                                          recv_val, recv_tx, recv_inc);
                for (int i = 0; i < 20; i++) read_sets[ri + 1].address[i] = tx.to[i];
                for (int i = 0; i < 32; i++) read_sets[ri + 1].slot[i] = sender_slot[i];
                read_sets[ri + 1].read_tx_index = recv_prior ? recv_tx : VERSION_BASE_STATE;
                read_sets[ri + 1].read_incarnation = recv_prior ? recv_inc : 0;
                rcount = 2;
            }

            // Intrinsic gas: 21000 for simple transfer
            ulong intrinsic_gas = 21000;
            ulong calldata_gas = 0;
            // 4 gas per zero byte, 16 per non-zero byte (simplified: assume all non-zero)
            calldata_gas = (ulong)tx.calldata_size * 16;
            ulong total_gas = intrinsic_gas + calldata_gas;

            ts.gas_used = total_gas;
            ts.status = (total_gas <= tx.gas_limit) ? 0 : 2;  // 0=ok, 2=oog
            ts.read_count = rcount;
            ts.write_count = wcount;
            ts.executed = 1;
            ts.validated = 0;

            results[exec_idx].gas_used = total_gas;
            results[exec_idx].status = ts.status;
            results[exec_idx].incarnation = cur_incarnation;

            continue;
        }

        // ---- Try to get a VALIDATION task ----
        uint val_idx = atomic_fetch_add_explicit(&sched_state[1], 1u, memory_order_relaxed);
        if (val_idx < num_txs) {
            // Wait until this tx has been executed
            if (tx_states[val_idx].executed == 0) {
                // Put it back (decrement validation_idx) and try again
                // Use CAS to avoid underflow race
                uint expected = val_idx + 1;
                atomic_compare_exchange_weak_explicit(&sched_state[1], &expected, val_idx,
                    memory_order_relaxed, memory_order_relaxed);
                continue;
            }

            // === VALIDATE TRANSACTION ===
            uint rcount = tx_states[val_idx].read_count;
            uint ri_base = val_idx * MAX_READS_PER_TX;
            bool valid = true;

            for (uint r = 0; r < rcount && r < MAX_READS_PER_TX; r++) {
                device const ReadSetEntry& re = read_sets[ri_base + r];
                uchar addr[20], sl[32];
                for (int i = 0; i < 20; i++) addr[i] = re.address[i];
                for (int i = 0; i < 32; i++) sl[i] = re.slot[i];

                if (!mv_validate_read(mv_memory, val_idx, addr, sl,
                                      re.read_tx_index, re.read_incarnation)) {
                    valid = false;
                    break;
                }
            }

            if (valid) {
                // Validation passed
                tx_states[val_idx].validated = 1;
                atomic_fetch_add_explicit(&sched_state[2], 1u, memory_order_relaxed);
            } else {
                // Conflict detected. Re-execute this tx.
                // 1. Mark all writes from this tx as estimates
                mv_mark_estimate(mv_memory, val_idx);

                // 2. Increment incarnation
                tx_states[val_idx].incarnation += 1;
                tx_states[val_idx].executed = 0;
                tx_states[val_idx].validated = 0;

                // 3. Reset execution_idx to re-execute from this tx
                uint expected_exec = atomic_load_explicit(&sched_state[0], memory_order_relaxed);
                while (expected_exec > val_idx) {
                    if (atomic_compare_exchange_weak_explicit(&sched_state[0],
                            &expected_exec, val_idx,
                            memory_order_relaxed, memory_order_relaxed))
                        break;
                }

                // 4. Invalidate all later transactions that were previously
                //    validated. Decrement done_count for each so it stays
                //    consistent -- prevents premature completion.
                for (uint i = val_idx + 1; i < num_txs; i++) {
                    if (tx_states[i].validated == 1) {
                        tx_states[i].validated = 0;
                        atomic_fetch_sub_explicit(&sched_state[2], 1u, memory_order_relaxed);
                    }
                }

                // 5. Reset validation_idx
                uint expected_val = atomic_load_explicit(&sched_state[1], memory_order_relaxed);
                while (expected_val > val_idx) {
                    if (atomic_compare_exchange_weak_explicit(&sched_state[1],
                            &expected_val, val_idx,
                            memory_order_relaxed, memory_order_relaxed))
                        break;
                }
            }

            continue;
        }

        // No work available. Check if done (same verified check).
        done = atomic_load_explicit(&sched_state[2], memory_order_relaxed);
        if (done >= num_txs) {
            bool all_valid = true;
            for (uint i = 0; i < num_txs; i++) {
                if (tx_states[i].validated != 1) { all_valid = false; break; }
            }
            if (all_valid) break;
        }

        // Spin-wait briefly. On GPU, threads are cheap -- just retry.
    }
}
