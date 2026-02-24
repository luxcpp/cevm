// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file state_table.cu
/// CUDA port of metal/state_table.metal — full implementation.
///
/// GPU-resident open-addressing hash table for Ethereum state.
///   Account table: keyed by 20-byte address, value is AccountData.
///   Storage table: keyed by 20-byte address + 32-byte slot, value is 32-byte data.
///
/// Plus a state-root reduction pipeline:
///   - state_root_compact:    pack occupied account entries
///   - state_root_sort:       one bitonic-sort step
///   - state_root_hash_entries: keccak256(RLP(account)) per entry
///   - state_root_reduce:     pairwise keccak combine to one digest
///
/// CUDA mapping notes (vs Metal source):
///   - Metal `atomic_uint key_valid` -> raw `uint32_t key_valid`, accessed
///     via `atomicCAS` for slot reservation.
///   - Inline keccak-f[1600] permutation duplicated from keccak256.cu so this
///     translation unit is self-contained (matches Metal's design).

#include <cstdint>
#include <cuda_runtime.h>

namespace evm::gpu::cuda
{

// =============================================================================
// Public structs (must match metal/state_table.metal byte-for-byte)
// =============================================================================

struct DeviceAccountData
{
    uint64_t nonce;
    uint64_t balance[4];   // uint256 little-endian limbs (w[0] = low)
    uint8_t  code_hash[32];
    uint8_t  storage_root[32];
};
static_assert(sizeof(DeviceAccountData) == 8 + 32 + 32 + 32,
              "DeviceAccountData layout mismatch");

struct DeviceAccountEntry
{
    uint8_t            key[20];
    uint32_t           key_valid;   // 0 = empty, 1 = occupied
    uint32_t           _pad;
    DeviceAccountData  data;
};

struct DeviceStorageKey
{
    uint8_t addr[20];
    uint8_t slot[32];
};

struct DeviceStorageEntry
{
    uint8_t  key_addr[20];
    uint8_t  key_slot[32];
    uint32_t key_valid;
    uint32_t _pad;
    uint8_t  value[32];
};

struct DeviceBatchParams
{
    uint32_t count;
    uint32_t capacity;
};

// =============================================================================
// Inline Keccak-f[1600] (mirrors metal/state_table.metal local copy)
// =============================================================================

__device__ static const uint64_t KC_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL,
};

__device__ static const int KPI_LANE[24] = {
    10,  7, 11, 17, 18,  3,  5, 16,  8, 21, 24,  4,
    15, 23, 19, 13, 12,  2, 20, 14, 22,  9,  6,  1
};

__device__ static const int KRHO[24] = {
     1,  3,  6, 10, 15, 21, 28, 36, 45, 55,  2, 14,
    27, 41, 56,  8, 25, 43, 62, 18, 39, 61, 20, 44
};

__device__ __forceinline__ uint64_t krotl64(uint64_t x, int n)
{
    return (x << n) | (x >> (64 - n));
}

__device__ void keccak_f_state(uint64_t st[25])
{
    #pragma unroll 1
    for (int round = 0; round < 24; ++round)
    {
        uint64_t C[5];
        #pragma unroll
        for (int x = 0; x < 5; ++x)
            C[x] = st[x] ^ st[x + 5] ^ st[x + 10] ^ st[x + 15] ^ st[x + 20];
        #pragma unroll
        for (int x = 0; x < 5; ++x)
        {
            uint64_t d = C[(x + 4) % 5] ^ krotl64(C[(x + 1) % 5], 1);
            #pragma unroll
            for (int y = 0; y < 5; ++y)
                st[x + 5 * y] ^= d;
        }
        uint64_t t = st[1];
        #pragma unroll
        for (int i = 0; i < 24; ++i)
        {
            uint64_t tmp = st[KPI_LANE[i]];
            st[KPI_LANE[i]] = krotl64(t, KRHO[i]);
            t = tmp;
        }
        #pragma unroll
        for (int y = 0; y < 5; ++y)
        {
            uint64_t row[5];
            #pragma unroll
            for (int x = 0; x < 5; ++x)
                row[x] = st[x + 5 * y];
            #pragma unroll
            for (int x = 0; x < 5; ++x)
                st[x + 5 * y] = row[x] ^ ((~row[(x + 1) % 5]) & row[(x + 2) % 5]);
        }
        st[0] ^= KC_RC[round];
    }
}

/// Keccak-256 of a byte array stored in a thread-private buffer.
/// Output goes into a 32-byte thread-private buffer.
__device__ void keccak256_local(const uint8_t* data, uint32_t len, uint8_t out[32])
{
    constexpr uint32_t rate = 136;
    uint64_t state[25] = {0};

    uint32_t absorbed = 0;
    while (absorbed + rate <= len)
    {
        #pragma unroll
        for (uint32_t w = 0; w < rate / 8; ++w)
        {
            uint64_t lane = 0;
            #pragma unroll
            for (uint32_t b = 0; b < 8; ++b)
                lane |= (uint64_t)data[absorbed + w * 8 + b] << (b * 8);
            state[w] ^= lane;
        }
        keccak_f_state(state);
        absorbed += rate;
    }

    uint8_t padded[136] = {0};
    uint32_t remaining = len - absorbed;
    for (uint32_t i = 0; i < remaining; ++i)
        padded[i] = data[absorbed + i];
    padded[remaining]   = 0x01;
    padded[rate - 1]   |= 0x80;

    #pragma unroll
    for (uint32_t w = 0; w < rate / 8; ++w)
    {
        uint64_t lane = 0;
        #pragma unroll
        for (uint32_t b = 0; b < 8; ++b)
            lane |= (uint64_t)padded[w * 8 + b] << (b * 8);
        state[w] ^= lane;
    }
    keccak_f_state(state);

    #pragma unroll
    for (uint32_t w = 0; w < 4; ++w)
    {
        uint64_t lane = state[w];
        #pragma unroll
        for (uint32_t b = 0; b < 8; ++b)
            out[w * 8 + b] = (uint8_t)(lane >> (b * 8));
    }
}

// =============================================================================
// Hash functions for table indexing (FNV-1a, capacity is power of 2)
// =============================================================================

__device__ __forceinline__ uint32_t hash_address(const uint8_t* addr, uint32_t capacity)
{
    uint32_t h = 0x811c9dc5u;
    #pragma unroll
    for (uint32_t i = 0; i < 20; ++i)
    {
        h ^= (uint32_t)addr[i];
        h *= 0x01000193u;
    }
    return h & (capacity - 1);
}

__device__ __forceinline__ uint32_t hash_storage_key(
    const uint8_t* addr, const uint8_t* slot, uint32_t capacity)
{
    uint32_t h = 0x811c9dc5u;
    #pragma unroll
    for (uint32_t i = 0; i < 20; ++i)
    {
        h ^= (uint32_t)addr[i];
        h *= 0x01000193u;
    }
    #pragma unroll
    for (uint32_t i = 0; i < 32; ++i)
    {
        h ^= (uint32_t)slot[i];
        h *= 0x01000193u;
    }
    return h & (capacity - 1);
}

__device__ __forceinline__ bool addr_eq_dev(const uint8_t* a, const uint8_t* b)
{
    #pragma unroll
    for (uint32_t i = 0; i < 20; ++i)
        if (a[i] != b[i]) return false;
    return true;
}

__device__ __forceinline__ bool storage_key_eq_dev(
    const uint8_t* a_addr, const uint8_t* a_slot,
    const uint8_t* b_addr, const uint8_t* b_slot)
{
    #pragma unroll
    for (uint32_t i = 0; i < 20; ++i)
        if (a_addr[i] != b_addr[i]) return false;
    #pragma unroll
    for (uint32_t i = 0; i < 32; ++i)
        if (a_slot[i] != b_slot[i]) return false;
    return true;
}

// =============================================================================
// Account table kernels
// =============================================================================

extern "C" __global__ void account_lookup_batch_kernel(
    const uint8_t*            __restrict__ keys_buf,    // count * 20
    const DeviceAccountEntry* __restrict__ table,
    DeviceAccountData*        __restrict__ results_buf, // count results
    uint32_t*                 __restrict__ found_buf,   // count flags
    DeviceBatchParams                      params)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= params.count) return;

    const uint8_t* key = keys_buf + tid * 20;
    uint32_t idx = hash_address(key, params.capacity);

    for (uint32_t probe = 0; probe < params.capacity; ++probe)
    {
        uint32_t s = (idx + probe) & (params.capacity - 1);
        if (table[s].key_valid == 0)
        {
            found_buf[tid] = 0;
            return;
        }
        if (addr_eq_dev(table[s].key, key))
        {
            results_buf[tid] = table[s].data;
            found_buf[tid] = 1;
            return;
        }
    }
    found_buf[tid] = 0;
}

extern "C" __global__ void account_insert_batch_kernel(
    const uint8_t*           __restrict__ keys_buf,  // count * 20
    const DeviceAccountData* __restrict__ data_buf,  // count entries
    DeviceAccountEntry*      __restrict__ table,
    DeviceBatchParams                      params)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= params.count) return;

    const uint8_t* key = keys_buf + tid * 20;
    uint32_t idx = hash_address(key, params.capacity);

    for (uint32_t probe = 0; probe < params.capacity; ++probe)
    {
        uint32_t s = (idx + probe) & (params.capacity - 1);

        // Try to claim empty slot. atomicCAS returns the OLD value; success
        // when old == 0 (slot was empty).
        uint32_t old = atomicCAS(&table[s].key_valid, 0u, 1u);
        if (old == 0)
        {
            // Claimed: write key + data.
            #pragma unroll
            for (uint32_t i = 0; i < 20; ++i)
                table[s].key[i] = key[i];
            table[s].data = data_buf[tid];
            return;
        }

        // Slot occupied: if it's our key, update in place.
        if (addr_eq_dev(table[s].key, key))
        {
            table[s].data = data_buf[tid];
            return;
        }
        // Otherwise probe further.
    }
}

// =============================================================================
// Storage table kernels
// =============================================================================

extern "C" __global__ void storage_lookup_batch_kernel(
    const DeviceStorageKey*   __restrict__ keys_buf,
    const DeviceStorageEntry* __restrict__ table,
    uint8_t*                  __restrict__ results_buf, // count * 32
    uint32_t*                 __restrict__ found_buf,
    DeviceBatchParams                      params)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= params.count) return;

    const uint8_t* addr = keys_buf[tid].addr;
    const uint8_t* slot = keys_buf[tid].slot;
    uint32_t idx = hash_storage_key(addr, slot, params.capacity);

    for (uint32_t probe = 0; probe < params.capacity; ++probe)
    {
        uint32_t s = (idx + probe) & (params.capacity - 1);
        if (table[s].key_valid == 0)
        {
            found_buf[tid] = 0;
            return;
        }
        if (storage_key_eq_dev(table[s].key_addr, table[s].key_slot, addr, slot))
        {
            uint8_t* out = results_buf + tid * 32;
            #pragma unroll
            for (uint32_t i = 0; i < 32; ++i)
                out[i] = table[s].value[i];
            found_buf[tid] = 1;
            return;
        }
    }
    found_buf[tid] = 0;
}

extern "C" __global__ void storage_insert_batch_kernel(
    const DeviceStorageKey* __restrict__ keys_buf,
    const uint8_t*          __restrict__ values_buf, // count * 32
    DeviceStorageEntry*     __restrict__ table,
    DeviceBatchParams                    params)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= params.count) return;

    const uint8_t* addr = keys_buf[tid].addr;
    const uint8_t* slot = keys_buf[tid].slot;
    uint32_t idx = hash_storage_key(addr, slot, params.capacity);

    for (uint32_t probe = 0; probe < params.capacity; ++probe)
    {
        uint32_t s = (idx + probe) & (params.capacity - 1);

        uint32_t old = atomicCAS(&table[s].key_valid, 0u, 1u);
        if (old == 0)
        {
            #pragma unroll
            for (uint32_t i = 0; i < 20; ++i)
                table[s].key_addr[i] = addr[i];
            #pragma unroll
            for (uint32_t i = 0; i < 32; ++i)
                table[s].key_slot[i] = slot[i];
            const uint8_t* val = values_buf + tid * 32;
            #pragma unroll
            for (uint32_t i = 0; i < 32; ++i)
                table[s].value[i] = val[i];
            return;
        }

        if (storage_key_eq_dev(table[s].key_addr, table[s].key_slot, addr, slot))
        {
            const uint8_t* val = values_buf + tid * 32;
            #pragma unroll
            for (uint32_t i = 0; i < 32; ++i)
                table[s].value[i] = val[i];
            return;
        }
    }
}

// =============================================================================
// State root pipeline
// =============================================================================

__device__ __forceinline__ bool addr_less_than(const uint8_t* a, const uint8_t* b)
{
    #pragma unroll
    for (uint32_t i = 0; i < 20; ++i)
    {
        if (a[i] < b[i]) return true;
        if (a[i] > b[i]) return false;
    }
    return false;
}

/// Compact occupied entries to the front of out_buf. Atomically increments
/// counter[0] for each occupied entry copied.
extern "C" __global__ void state_root_compact_kernel(
    const DeviceAccountEntry* __restrict__ table,
    DeviceAccountEntry*       __restrict__ out_buf,
    uint32_t*                 __restrict__ counter,
    DeviceBatchParams                      params)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= params.capacity) return;

    if (table[tid].key_valid != 0)
    {
        uint32_t idx = atomicAdd(counter, 1u);
        out_buf[idx] = table[tid];
    }
}

/// One step of bitonic sort. params.capacity packs:
///   bits [15:0]  = substep (partner XOR mask)
///   bits [31:16] = direction mask (step << 1)
extern "C" __global__ void state_root_sort_kernel(
    DeviceAccountEntry*       __restrict__ entries,
    DeviceBatchParams                      params)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= params.count) return;

    uint32_t substep  = params.capacity & 0xFFFFu;
    uint32_t dir_mask = (params.capacity >> 16) & 0xFFFFu;

    uint32_t partner = tid ^ substep;
    if (partner <= tid || partner >= params.count) return;

    bool ascending = ((tid & dir_mask) == 0);
    bool a_lt_b    = addr_less_than(entries[tid].key, entries[partner].key);

    if (ascending != a_lt_b)
    {
        DeviceAccountEntry tmp = entries[tid];
        entries[tid]     = entries[partner];
        entries[partner] = tmp;
    }
}

/// Phase 1 of state-root reduce: hash one occupied account entry
/// into a 32-byte digest using keccak256(RLP(account)).
extern "C" __global__ void state_root_hash_entries_kernel(
    const DeviceAccountEntry* __restrict__ table,
    uint8_t*                  __restrict__ hash_buf, // capacity * 32
    DeviceBatchParams                      params)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= params.capacity) return;

    uint8_t* out = hash_buf + tid * 32;

    if (table[tid].key_valid == 0)
    {
        #pragma unroll
        for (uint32_t i = 0; i < 32; ++i) out[i] = 0;
        return;
    }

    // Build RLP encoding into a thread-private buffer.
    uint8_t  rlp_buf[256];
    uint32_t pos = 0;
    uint8_t  payload[200];
    uint32_t ppos = 0;

    // Encode nonce (uint64).
    {
        uint64_t nonce = table[tid].data.nonce;
        if (nonce == 0)
        {
            payload[ppos++] = 0x80;
        }
        else
        {
            uint8_t nbuf[8];
            int nlen = 0;
            uint64_t tmp = nonce;
            while (tmp > 0)
            {
                nbuf[7 - nlen] = (uint8_t)(tmp & 0xFF);
                tmp >>= 8;
                ++nlen;
            }
            if (nlen == 1 && nbuf[7] < 0x80)
            {
                payload[ppos++] = nbuf[7];
            }
            else
            {
                payload[ppos++] = (uint8_t)(0x80 + nlen);
                for (int i = 8 - nlen; i < 8; ++i)
                    payload[ppos++] = nbuf[i];
            }
        }
    }

    // Encode balance (uint256). balance[3] is the high word.
    {
        uint8_t bal_be[32];
        #pragma unroll
        for (int w = 0; w < 4; ++w)
        {
            uint64_t word = table[tid].data.balance[w];
            int base_idx = (3 - w) * 8;
            #pragma unroll
            for (int b = 0; b < 8; ++b)
                bal_be[base_idx + 7 - b] = (uint8_t)((word >> (b * 8)) & 0xFF);
        }
        uint32_t bal_start = 0;
        while (bal_start < 32 && bal_be[bal_start] == 0) ++bal_start;
        uint32_t bal_len = 32 - bal_start;
        if (bal_len == 0)
        {
            payload[ppos++] = 0x80;
        }
        else if (bal_len == 1 && bal_be[bal_start] < 0x80)
        {
            payload[ppos++] = bal_be[bal_start];
        }
        else
        {
            payload[ppos++] = (uint8_t)(0x80 + bal_len);
            for (uint32_t i = bal_start; i < 32; ++i)
                payload[ppos++] = bal_be[i];
        }
    }

    // Encode storage_root (32 bytes).
    payload[ppos++] = 0x80 + 32;
    #pragma unroll
    for (uint32_t i = 0; i < 32; ++i)
        payload[ppos++] = table[tid].data.storage_root[i];

    // Encode code_hash (32 bytes).
    payload[ppos++] = 0x80 + 32;
    #pragma unroll
    for (uint32_t i = 0; i < 32; ++i)
        payload[ppos++] = table[tid].data.code_hash[i];

    // Wrap in RLP list header.
    if (ppos < 56)
    {
        rlp_buf[0] = (uint8_t)(0xC0 + ppos);
        for (uint32_t i = 0; i < ppos; ++i)
            rlp_buf[1 + i] = payload[i];
        pos = 1 + ppos;
    }
    else
    {
        uint8_t lbuf[4];
        int llen = 0;
        uint32_t tmp2 = ppos;
        while (tmp2 > 0)
        {
            lbuf[3 - llen] = (uint8_t)(tmp2 & 0xFF);
            tmp2 >>= 8;
            ++llen;
        }
        rlp_buf[0] = (uint8_t)(0xF7 + llen);
        for (int i = 4 - llen; i < 4; ++i)
            rlp_buf[1 + i - (4 - llen)] = lbuf[i];
        uint32_t hdr_len = 1 + (uint32_t)llen;
        for (uint32_t i = 0; i < ppos; ++i)
            rlp_buf[hdr_len + i] = payload[i];
        pos = hdr_len + ppos;
    }

    uint8_t digest[32];
    keccak256_local(rlp_buf, pos, digest);
    #pragma unroll
    for (uint32_t i = 0; i < 32; ++i)
        out[i] = digest[i];
}

/// Phase 2: parallel reduce. Replace pair (hash[2i], hash[2i+1]) with
/// keccak256(hash[2i] || hash[2i+1]) at index i. params.count = active
/// hash count for this pass.
extern "C" __global__ void state_root_reduce_kernel(
    uint8_t*                  __restrict__ hash_buf,
    DeviceBatchParams                      params)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= params.count / 2) return;

    uint8_t* dst = hash_buf + tid * 32;
    const uint8_t* a = hash_buf + (2 * tid)     * 32;
    const uint8_t* b = hash_buf + (2 * tid + 1) * 32;

    uint8_t combined[64];
    #pragma unroll
    for (uint32_t i = 0; i < 32; ++i) combined[i]      = a[i];
    #pragma unroll
    for (uint32_t i = 0; i < 32; ++i) combined[32 + i] = b[i];

    uint8_t digest[32];
    keccak256_local(combined, 64, digest);
    #pragma unroll
    for (uint32_t i = 0; i < 32; ++i)
        dst[i] = digest[i];
}

// =============================================================================
// Host-callable launchers
// =============================================================================

namespace
{
    constexpr uint32_t TPB = 128;

    inline uint32_t blocks_for(uint32_t n)
    {
        return (n + TPB - 1) / TPB;
    }
}  // namespace

extern "C" cudaError_t evm_cuda_account_lookup_launch(
    const void*  d_keys, const void* d_table,
    void*        d_results, void* d_found,
    uint32_t     count, uint32_t capacity, cudaStream_t stream)
{
    if (count == 0) return cudaSuccess;
    DeviceBatchParams p{count, capacity};
    account_lookup_batch_kernel<<<blocks_for(count), TPB, 0, stream>>>(
        static_cast<const uint8_t*>(d_keys),
        static_cast<const DeviceAccountEntry*>(d_table),
        static_cast<DeviceAccountData*>(d_results),
        static_cast<uint32_t*>(d_found),
        p);
    return cudaGetLastError();
}

extern "C" cudaError_t evm_cuda_account_insert_launch(
    const void*  d_keys, const void* d_data, void* d_table,
    uint32_t     count, uint32_t capacity, cudaStream_t stream)
{
    if (count == 0) return cudaSuccess;
    DeviceBatchParams p{count, capacity};
    account_insert_batch_kernel<<<blocks_for(count), TPB, 0, stream>>>(
        static_cast<const uint8_t*>(d_keys),
        static_cast<const DeviceAccountData*>(d_data),
        static_cast<DeviceAccountEntry*>(d_table),
        p);
    return cudaGetLastError();
}

extern "C" cudaError_t evm_cuda_storage_lookup_launch(
    const void*  d_keys, const void* d_table,
    void*        d_results, void* d_found,
    uint32_t     count, uint32_t capacity, cudaStream_t stream)
{
    if (count == 0) return cudaSuccess;
    DeviceBatchParams p{count, capacity};
    storage_lookup_batch_kernel<<<blocks_for(count), TPB, 0, stream>>>(
        static_cast<const DeviceStorageKey*>(d_keys),
        static_cast<const DeviceStorageEntry*>(d_table),
        static_cast<uint8_t*>(d_results),
        static_cast<uint32_t*>(d_found),
        p);
    return cudaGetLastError();
}

extern "C" cudaError_t evm_cuda_storage_insert_launch(
    const void*  d_keys, const void* d_values, void* d_table,
    uint32_t     count, uint32_t capacity, cudaStream_t stream)
{
    if (count == 0) return cudaSuccess;
    DeviceBatchParams p{count, capacity};
    storage_insert_batch_kernel<<<blocks_for(count), TPB, 0, stream>>>(
        static_cast<const DeviceStorageKey*>(d_keys),
        static_cast<const uint8_t*>(d_values),
        static_cast<DeviceStorageEntry*>(d_table),
        p);
    return cudaGetLastError();
}

extern "C" cudaError_t evm_cuda_state_root_compact_launch(
    const void*  d_table, void* d_out, void* d_counter,
    uint32_t     capacity, cudaStream_t stream)
{
    if (capacity == 0) return cudaSuccess;
    DeviceBatchParams p{0, capacity};  // count unused for compact
    state_root_compact_kernel<<<blocks_for(capacity), TPB, 0, stream>>>(
        static_cast<const DeviceAccountEntry*>(d_table),
        static_cast<DeviceAccountEntry*>(d_out),
        static_cast<uint32_t*>(d_counter),
        p);
    return cudaGetLastError();
}

extern "C" cudaError_t evm_cuda_state_root_sort_launch(
    void*        d_entries,
    uint32_t     count,
    uint32_t     packed_step,   // (dir_mask << 16) | substep
    cudaStream_t stream)
{
    if (count == 0) return cudaSuccess;
    DeviceBatchParams p{count, packed_step};
    state_root_sort_kernel<<<blocks_for(count), TPB, 0, stream>>>(
        static_cast<DeviceAccountEntry*>(d_entries),
        p);
    return cudaGetLastError();
}

extern "C" cudaError_t evm_cuda_state_root_hash_launch(
    const void*  d_table, void* d_hash_buf,
    uint32_t     capacity, cudaStream_t stream)
{
    if (capacity == 0) return cudaSuccess;
    DeviceBatchParams p{0, capacity};
    state_root_hash_entries_kernel<<<blocks_for(capacity), TPB, 0, stream>>>(
        static_cast<const DeviceAccountEntry*>(d_table),
        static_cast<uint8_t*>(d_hash_buf),
        p);
    return cudaGetLastError();
}

extern "C" cudaError_t evm_cuda_state_root_reduce_launch(
    void*        d_hash_buf,
    uint32_t     active_count,
    cudaStream_t stream)
{
    if (active_count < 2) return cudaSuccess;
    DeviceBatchParams p{active_count, 0};
    uint32_t threads = active_count / 2;
    state_root_reduce_kernel<<<blocks_for(threads), TPB, 0, stream>>>(
        static_cast<uint8_t*>(d_hash_buf),
        p);
    return cudaGetLastError();
}

}  // namespace evm::gpu::cuda
