// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file state_table.metal
/// GPU-resident open-addressing hash table for Ethereum state.
///
/// Keys are 20-byte addresses (accounts) or 20+32 bytes (storage slots).
/// Values are account data or 32-byte storage values.
/// All data lives in device memory persistently across blocks.
///
/// Table layout: power-of-2 slots with linear probing.
/// Empty slots are marked by key = all zeros.
/// Concurrent inserts use atomic_compare_exchange_weak_explicit on a
/// per-slot lock word.

#include <metal_stdlib>
using namespace metal;

// -- Types matching the C++ host ------------------------------------------------

/// 20-byte Ethereum address stored in a 32-byte field (right-aligned, like EVMC).
struct Address {
    uchar bytes[20];
};

/// Account data stored per address in the hash table.
struct AccountData {
    ulong  nonce;          // 8 bytes
    ulong  balance[4];     // uint256 as 4x uint64 (w[0]=low, w[3]=high)
    uchar  code_hash[32];  // Keccak-256 of deployed code
    uchar  storage_root[32]; // Placeholder storage root
};

/// Entry in the account hash table.
/// key_valid == 0 means the slot is empty.
struct AccountEntry {
    uchar       key[20];    // address
    uint        key_valid;  // 0 = empty, 1 = occupied
    uint        _pad;
    AccountData data;
};

/// Storage key: 20-byte address + 32-byte slot.
struct StorageKey {
    uchar addr[20];
    uchar slot[32];
};

/// Entry in the storage hash table.
struct StorageEntry {
    uchar key_addr[20];   // address
    uchar key_slot[32];   // storage slot
    uint  key_valid;      // 0 = empty, 1 = occupied
    uint  _pad;
    uchar value[32];      // 32-byte storage value
};

/// Parameters for batch operations.
struct BatchParams {
    uint count;       // number of keys in this batch
    uint capacity;    // hash table capacity (must be power of 2)
};

// -- Keccak-256 inline (for state root computation) ----------------------------
// Duplicated from keccak256.metal to keep this shader self-contained.

constant ulong KC[24] = {
    0x0000000000000001UL, 0x0000000000008082UL,
    0x800000000000808AUL, 0x8000000080008000UL,
    0x000000000000808BUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL,
    0x000000000000008AUL, 0x0000000000000088UL,
    0x0000000080008009UL, 0x000000008000000AUL,
    0x000000008000808BUL, 0x800000000000008BUL,
    0x8000000000008089UL, 0x8000000000008003UL,
    0x8000000000008002UL, 0x8000000000000080UL,
    0x000000000000800AUL, 0x800000008000000AUL,
    0x8000000080008081UL, 0x8000000000008080UL,
    0x0000000080000001UL, 0x8000000080008008UL,
};

constant int KPI_LANE[24] = {
    10,  7, 11, 17, 18,  3,  5, 16,  8, 21, 24,  4,
    15, 23, 19, 13, 12,  2, 20, 14, 22,  9,  6,  1
};

constant int KRHO[24] = {
     1,  3,  6, 10, 15, 21, 28, 36, 45, 55,  2, 14,
    27, 41, 56,  8, 25, 43, 62, 18, 39, 61, 20, 44
};

inline ulong krotl64(ulong x, int n) {
    return (x << n) | (x >> (64 - n));
}

void keccak_f_state(thread ulong st[25]) {
    for (int round = 0; round < 24; ++round) {
        ulong C[5];
        for (int x = 0; x < 5; ++x)
            C[x] = st[x] ^ st[x + 5] ^ st[x + 10] ^ st[x + 15] ^ st[x + 20];
        for (int x = 0; x < 5; ++x) {
            ulong d = C[(x + 4) % 5] ^ krotl64(C[(x + 1) % 5], 1);
            for (int y = 0; y < 5; ++y)
                st[x + 5 * y] ^= d;
        }
        ulong t = st[1];
        for (int i = 0; i < 24; ++i) {
            ulong tmp = st[KPI_LANE[i]];
            st[KPI_LANE[i]] = krotl64(t, KRHO[i]);
            t = tmp;
        }
        for (int y = 0; y < 5; ++y) {
            ulong row[5];
            for (int x = 0; x < 5; ++x)
                row[x] = st[x + 5 * y];
            for (int x = 0; x < 5; ++x)
                st[x + 5 * y] = row[x] ^ ((~row[(x + 1) % 5]) & row[(x + 2) % 5]);
        }
        st[0] ^= KC[round];
    }
}

/// Keccak-256 of a byte array in thread-private memory.
void keccak256_local(const thread uchar* data, uint len, thread uchar out[32]) {
    constexpr uint rate = 136;
    ulong state[25] = {};

    uint absorbed = 0;
    while (absorbed + rate <= len) {
        for (uint w = 0; w < rate / 8; ++w) {
            ulong lane = 0;
            for (uint b = 0; b < 8; ++b)
                lane |= ulong(data[absorbed + w * 8 + b]) << (b * 8);
            state[w] ^= lane;
        }
        keccak_f_state(state);
        absorbed += rate;
    }

    uchar padded[136] = {};
    uint remaining = len - absorbed;
    for (uint i = 0; i < remaining; ++i)
        padded[i] = data[absorbed + i];
    padded[remaining] = 0x01;
    padded[rate - 1] |= 0x80;

    for (uint w = 0; w < rate / 8; ++w) {
        ulong lane = 0;
        for (uint b = 0; b < 8; ++b)
            lane |= ulong(padded[w * 8 + b]) << (b * 8);
        state[w] ^= lane;
    }
    keccak_f_state(state);

    for (uint w = 0; w < 4; ++w) {
        ulong lane = state[w];
        for (uint b = 0; b < 8; ++b)
            out[w * 8 + b] = uchar(lane >> (b * 8));
    }
}

// -- Hash functions for table indexing ------------------------------------------

/// FNV-1a hash of 20 bytes, returns a 32-bit index.
inline uint hash_address(const device uchar* addr, uint capacity) {
    uint h = 0x811c9dc5u;
    for (uint i = 0; i < 20; ++i) {
        h ^= uint(addr[i]);
        h *= 0x01000193u;
    }
    return h & (capacity - 1);  // capacity is power of 2
}

/// FNV-1a hash of 52 bytes (20 addr + 32 slot).
inline uint hash_storage_key(const device uchar* addr, const device uchar* slot, uint capacity) {
    uint h = 0x811c9dc5u;
    for (uint i = 0; i < 20; ++i) {
        h ^= uint(addr[i]);
        h *= 0x01000193u;
    }
    for (uint i = 0; i < 32; ++i) {
        h ^= uint(slot[i]);
        h *= 0x01000193u;
    }
    return h & (capacity - 1);
}

/// Compare 20 bytes.
inline bool addr_eq(const device uchar* a, const device uchar* b) {
    for (uint i = 0; i < 20; ++i)
        if (a[i] != b[i]) return false;
    return true;
}

/// Compare 20 + 32 bytes.
inline bool storage_key_eq(
    const device uchar* a_addr, const device uchar* a_slot,
    const device uchar* b_addr, const device uchar* b_slot)
{
    for (uint i = 0; i < 20; ++i)
        if (a_addr[i] != b_addr[i]) return false;
    for (uint i = 0; i < 32; ++i)
        if (a_slot[i] != b_slot[i]) return false;
    return true;
}

// -- Account hash table kernels -------------------------------------------------

/// Batch lookup: for each address in keys_buf, find the account data.
/// results_buf[i].key_valid = 0 if not found, 1 if found (with data filled).
kernel void account_lookup_batch(
    device const uchar*       keys_buf    [[buffer(0)]],  // N * 20 bytes
    device const AccountEntry* table      [[buffer(1)]],
    device AccountData*       results_buf [[buffer(2)]],  // N results
    device uint*              found_buf   [[buffer(3)]],  // N found flags
    device const BatchParams* params      [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    uint count = params->count;
    uint capacity = params->capacity;
    if (tid >= count) return;

    const device uchar* key = keys_buf + tid * 20;
    uint idx = hash_address(key, capacity);

    // Linear probe.
    for (uint probe = 0; probe < capacity; ++probe) {
        uint slot = (idx + probe) & (capacity - 1);
        if (table[slot].key_valid == 0) {
            // Empty slot: key not in table.
            found_buf[tid] = 0;
            return;
        }
        if (addr_eq(table[slot].key, key)) {
            results_buf[tid] = table[slot].data;
            found_buf[tid] = 1;
            return;
        }
    }
    // Table full, not found.
    found_buf[tid] = 0;
}

/// Batch insert: for each (address, AccountData) pair, insert into the table.
/// Uses atomic CAS on the key_valid field for concurrent safety.
kernel void account_insert_batch(
    device const uchar*       keys_buf    [[buffer(0)]],  // N * 20 bytes
    device const AccountData* data_buf    [[buffer(1)]],  // N AccountData
    device AccountEntry*      table       [[buffer(2)]],
    device const BatchParams* params      [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint count = params->count;
    uint capacity = params->capacity;
    if (tid >= count) return;

    const device uchar* key = keys_buf + tid * 20;
    uint idx = hash_address(key, capacity);

    for (uint probe = 0; probe < capacity; ++probe) {
        uint slot = (idx + probe) & (capacity - 1);

        // Try to claim an empty slot via atomic CAS.
        device atomic_uint* valid_ptr =
            reinterpret_cast<device atomic_uint*>(&table[slot].key_valid);

        uint expected = 0;
        if (atomic_compare_exchange_weak_explicit(
                valid_ptr, &expected, 1u,
                memory_order_relaxed, memory_order_relaxed))
        {
            // Claimed empty slot. Write key and data.
            for (uint i = 0; i < 20; ++i)
                table[slot].key[i] = key[i];
            table[slot].data = data_buf[tid];
            return;
        }

        // Slot is occupied. Check if it is our key (update in place).
        if (addr_eq(table[slot].key, key)) {
            table[slot].data = data_buf[tid];
            return;
        }
        // Otherwise continue probing.
    }
    // Table full -- should not happen if load factor < 0.75.
}

// -- Storage hash table kernels -------------------------------------------------

/// Batch lookup: for each StorageKey, find the 32-byte value.
kernel void storage_lookup_batch(
    device const StorageKey*   keys_buf    [[buffer(0)]],  // N keys
    device const StorageEntry* table       [[buffer(1)]],
    device uchar*              results_buf [[buffer(2)]],  // N * 32 bytes
    device uint*               found_buf   [[buffer(3)]],  // N flags
    device const BatchParams*  params      [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    uint count = params->count;
    uint capacity = params->capacity;
    if (tid >= count) return;

    const device uchar* addr = keys_buf[tid].addr;
    const device uchar* slot = keys_buf[tid].slot;
    uint idx = hash_storage_key(addr, slot, capacity);

    for (uint probe = 0; probe < capacity; ++probe) {
        uint s = (idx + probe) & (capacity - 1);
        if (table[s].key_valid == 0) {
            found_buf[tid] = 0;
            return;
        }
        if (storage_key_eq(table[s].key_addr, table[s].key_slot, addr, slot)) {
            device uchar* out = results_buf + tid * 32;
            for (uint i = 0; i < 32; ++i)
                out[i] = table[s].value[i];
            found_buf[tid] = 1;
            return;
        }
    }
    found_buf[tid] = 0;
}

/// Batch insert: for each (StorageKey, value) pair, insert into storage table.
kernel void storage_insert_batch(
    device const StorageKey*  keys_buf    [[buffer(0)]],  // N keys
    device const uchar*       values_buf  [[buffer(1)]],  // N * 32 bytes
    device StorageEntry*      table       [[buffer(2)]],
    device const BatchParams* params      [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    uint count = params->count;
    uint capacity = params->capacity;
    if (tid >= count) return;

    const device uchar* addr = keys_buf[tid].addr;
    const device uchar* slot = keys_buf[tid].slot;
    uint idx = hash_storage_key(addr, slot, capacity);

    for (uint probe = 0; probe < capacity; ++probe) {
        uint s = (idx + probe) & (capacity - 1);

        device atomic_uint* valid_ptr =
            reinterpret_cast<device atomic_uint*>(&table[s].key_valid);

        uint expected = 0;
        if (atomic_compare_exchange_weak_explicit(
                valid_ptr, &expected, 1u,
                memory_order_relaxed, memory_order_relaxed))
        {
            for (uint i = 0; i < 20; ++i)
                table[s].key_addr[i] = addr[i];
            for (uint i = 0; i < 32; ++i)
                table[s].key_slot[i] = slot[i];
            const device uchar* val = values_buf + tid * 32;
            for (uint i = 0; i < 32; ++i)
                table[s].value[i] = val[i];
            return;
        }

        if (storage_key_eq(table[s].key_addr, table[s].key_slot, addr, slot)) {
            const device uchar* val = values_buf + tid * 32;
            for (uint i = 0; i < 32; ++i)
                table[s].value[i] = val[i];
            return;
        }
    }
}

// -- State root computation (parallel reduce + keccak) --------------------------

/// Phase 1: Each thread hashes one occupied account entry into a 32-byte digest.
/// The digest is: keccak256(RLP(nonce, balance, storage_root, code_hash)).
///
/// RLP encoding is done inline. For accounts, the RLP is:
///   list(encode_uint64(nonce), encode_uint256(balance),
///        encode_bytes32(storage_root), encode_bytes32(code_hash))
///
/// Output: hash_buf[tid] = 32-byte hash for each occupied slot.
///         count_buf[0] = number of occupied entries (set by host before dispatch).
kernel void state_root_hash_entries(
    device const AccountEntry* table    [[buffer(0)]],
    device uchar*              hash_buf [[buffer(1)]],  // capacity * 32
    device const BatchParams*  params   [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    uint capacity = params->capacity;
    if (tid >= capacity) return;

    device uchar* out = hash_buf + tid * 32;

    if (table[tid].key_valid == 0) {
        // Empty slot: zero hash.
        for (uint i = 0; i < 32; ++i)
            out[i] = 0;
        return;
    }

    // Build RLP encoding in thread-private memory.
    // Max RLP size for an account: 1 (list header) + 9 (nonce) + 33 (balance) + 33 (storage_root) + 33 (code_hash) = ~110
    uchar rlp_buf[256];
    uint pos = 0;

    // We write payload first, then wrap in list header.
    uchar payload[200];
    uint ppos = 0;

    // Encode nonce (uint64).
    ulong nonce = table[tid].data.nonce;
    if (nonce == 0) {
        payload[ppos++] = 0x80;
    } else {
        uchar nbuf[8];
        int nlen = 0;
        ulong tmp = nonce;
        while (tmp > 0) {
            nbuf[7 - nlen] = uchar(tmp & 0xFF);
            tmp >>= 8;
            nlen++;
        }
        if (nlen == 1 && nbuf[7] < 0x80) {
            payload[ppos++] = nbuf[7];
        } else {
            payload[ppos++] = uchar(0x80 + nlen);
            for (int i = 8 - nlen; i < 8; ++i)
                payload[ppos++] = nbuf[i];
        }
    }

    // Encode balance (uint256 as big-endian, strip leading zeros).
    uchar bal_be[32];
    for (int w = 0; w < 4; ++w) {
        ulong word = table[tid].data.balance[w];
        // Store in big-endian: balance[3] is high word -> bytes [0..7]
        int base_idx = (3 - w) * 8;
        for (int b = 0; b < 8; ++b)
            bal_be[base_idx + 7 - b] = uchar((word >> (b * 8)) & 0xFF);
    }
    uint bal_start = 0;
    while (bal_start < 32 && bal_be[bal_start] == 0) bal_start++;
    uint bal_len = 32 - bal_start;
    if (bal_len == 0) {
        payload[ppos++] = 0x80;
    } else if (bal_len == 1 && bal_be[bal_start] < 0x80) {
        payload[ppos++] = bal_be[bal_start];
    } else {
        payload[ppos++] = uchar(0x80 + bal_len);
        for (uint i = bal_start; i < 32; ++i)
            payload[ppos++] = bal_be[i];
    }

    // Encode storage_root (always 32 bytes).
    payload[ppos++] = 0x80 + 32;
    for (uint i = 0; i < 32; ++i)
        payload[ppos++] = table[tid].data.storage_root[i];

    // Encode code_hash (always 32 bytes).
    payload[ppos++] = 0x80 + 32;
    for (uint i = 0; i < 32; ++i)
        payload[ppos++] = table[tid].data.code_hash[i];

    // Wrap in RLP list.
    if (ppos < 56) {
        rlp_buf[0] = uchar(0xC0 + ppos);
        for (uint i = 0; i < ppos; ++i)
            rlp_buf[1 + i] = payload[i];
        pos = 1 + ppos;
    } else {
        // Long list (unlikely for account RLP, but handle it).
        uchar lbuf[4];
        int llen = 0;
        uint tmp2 = ppos;
        while (tmp2 > 0) {
            lbuf[3 - llen] = uchar(tmp2 & 0xFF);
            tmp2 >>= 8;
            llen++;
        }
        rlp_buf[0] = uchar(0xF7 + llen);
        for (int i = 4 - llen; i < 4; ++i)
            rlp_buf[1 + i - (4 - llen)] = lbuf[i];
        uint hdr_len = 1 + uint(llen);
        for (uint i = 0; i < ppos; ++i)
            rlp_buf[hdr_len + i] = payload[i];
        pos = hdr_len + ppos;
    }

    // Hash the RLP-encoded account.
    uchar digest[32];
    keccak256_local(rlp_buf, pos, digest);
    for (uint i = 0; i < 32; ++i)
        out[i] = digest[i];
}

/// Phase 2: Parallel reduction -- combine N hashes into one root.
/// Each thread XORs pairs of hashes: hash_buf[2*tid] ^= hash_buf[2*tid+1].
/// Host dispatches log2(N) passes, halving the working set each time.
/// Final hash: keccak256 of the reduced concatenation.
kernel void state_root_reduce(
    device uchar*              hash_buf [[buffer(0)]],
    device const BatchParams*  params   [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    uint count = params->count;  // number of active hashes this pass
    if (tid >= count / 2) return;

    // XOR hash_buf[2*tid] with hash_buf[2*tid+1] into hash_buf[tid].
    device uchar* dst = hash_buf + tid * 32;
    device uchar* a   = hash_buf + (2 * tid) * 32;
    device uchar* b   = hash_buf + (2 * tid + 1) * 32;

    uchar combined[64];
    for (uint i = 0; i < 32; ++i) combined[i] = a[i];
    for (uint i = 0; i < 32; ++i) combined[32 + i] = b[i];

    uchar digest[32];
    keccak256_local(combined, 64, digest);
    for (uint i = 0; i < 32; ++i)
        dst[i] = digest[i];
}
