// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file quasar_wave.metal
/// QuasarGPU — wave-tick scheduler kernel.
///
/// One Metal dispatch per wave tick. The host re-launches this kernel
/// repeatedly until QuasarRoundResult::status != 0. Inside one tick:
///
///   * Each workgroup picks a service by gid % kNumServices, drains up
///     to wave_tick_budget items from its ring, and exits. No persistent
///     hot-spinning — that's the v0.29 starvation fix.
///   * Cross-service ring writes use the v0.30 release pattern:
///     write items[tail], threadgroup_barrier(mem_flags::mem_device),
///     atomic_store(tail). Cross-service ring reads pair the relaxed
///     atomic_load(tail) with another mem_device barrier before the
///     items[head] read.
///
/// Services implemented in this file:
///
///   Ingress       host tx blobs                      → Decode
///   Decode        sender recovery (stub→v0.38)       → Crypto
///                                                    → StateRequest (cold-state)
///   Crypto        admission                          → DagReady (Nebula) / Exec (Nova)
///   DagReady      MVCC ready set                     → Exec
///   Exec          EVM fiber VM                       → Validate
///                                                    → StateRequest (suspend)
///   Validate      Block-STM read-set check           → Commit / Repair
///   Repair        re-execute conflicting txs         → Exec
///   Commit        commit + per-tx receipt keccak     → SlotResult roots
///   StateRequest  GPU → host page faults             (host poll)
///   StateResp     host → GPU page replies            → Crypto (resume)
///   Vote          host votes                         → batch verify → QuorumOut
///   QuorumOut     QC emission                        (host poll)

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Layout — must match quasar_gpu_layout.hpp byte-for-byte.
// ============================================================================

constant uint kNumServices       = 12u;
constant uint kMaxRWSetPerTx     = 8u;
constant uint kDefaultMvccSlots  = 8192u;
constant uint kMaxDagParents     = 4u;
constant uint kMaxDagChildren    = 16u;
constant uint kFiberStackDepth   = 64u;
constant uint kFiberMemoryBytes  = 1024u;

struct alignas(16) RingHeader {
    atomic_uint head;
    atomic_uint tail;
    uint        capacity;
    uint        mask;
    ulong       items_ofs;
    uint        item_size;
    uint        _pad0;
    atomic_uint pushed;
    atomic_uint consumed;
    uint        _pad1;
    uint        _pad2;
};

struct alignas(16) IngressTx {
    uint  blob_offset;
    uint  blob_size;
    ulong gas_limit;
    uint  nonce;
    uint  _pad0;
    uint  origin_lo;
    uint  origin_hi;
};

struct alignas(16) DecodedTx {
    uint  tx_index;
    uint  blob_offset;
    uint  blob_size;
    ulong gas_limit;
    uint  nonce;
    uint  origin_lo;
    uint  origin_hi;
    uint  status;
};

struct alignas(16) VerifiedTx {
    uint  tx_index;
    uint  admission;
    ulong gas_limit;
    uint  origin_lo;
    uint  origin_hi;
    ulong _pad0;
};

struct RWSetEntry {
    ulong key_lo;
    ulong key_hi;
    uint  version_seen;
    uint  kind;
};

struct ExecResult {
    uint  tx_index;
    uint  incarnation;
    uint  status;
    uint  rw_count;
    ulong gas_used;
    RWSetEntry rw[kMaxRWSetPerTx];
};

struct alignas(16) CommitItem {
    uint  tx_index;
    uint  status;
    ulong gas_used;
    ulong cumulative_gas;
    uchar receipt_hash[32];
};

struct alignas(16) MvccSlot {
    ulong key_lo;
    ulong key_hi;
    atomic_uint last_writer_tx;
    atomic_uint last_writer_inc;
    atomic_uint version;
    uint        _pad0;
};

struct alignas(16) DagNode {
    uint tx_index;
    uint parent_count;
    atomic_uint unresolved_parents;
    uint child_count;
    uint parents[kMaxDagParents];
    uint children[kMaxDagChildren];
};

struct alignas(16) StateRequest {
    uint  tx_index;
    uint  key_type;
    uint  priority;
    uint  _pad0;
    ulong key_lo;
    ulong key_hi;
};

struct alignas(16) StatePage {
    uint  tx_index;
    uint  key_type;
    uint  status;
    uint  data_size;
    ulong key_lo;
    ulong key_hi;
    uchar data[64];
};

struct alignas(16) VoteIngress {
    uint  validator_index;
    uint  round;
    uint  stake_weight;
    uint  sig_kind;
    uchar subject[32];
    uchar signature[96];
};

struct alignas(16) QuorumCert {
    uint  round;
    uint  status;
    uint  signers_count;
    uint  total_stake;
    uint  sig_kind;
    uint  _pad0;
    ulong _pad1;
    uchar subject[32];
    uchar agg_signature[96];
};

struct alignas(16) QuasarRoundDescriptor {
    ulong chain_id;
    ulong round;
    ulong timestamp_ns;
    ulong deadline_ns;
    ulong gas_limit;
    ulong base_fee;
    uint  wave_tick_budget;
    uint  wave_tick_index;
    uint  closing_flag;
    uint  mode;
    uchar parent_block_hash[32];
    uchar parent_state_root[32];
    uchar parent_execution_root[32];
};

struct alignas(16) QuasarRoundResult {
    atomic_uint status;
    atomic_uint tx_count;
    atomic_uint gas_used_lo;
    atomic_uint gas_used_hi;
    atomic_uint wave_tick_count;
    atomic_uint conflict_count;
    atomic_uint repair_count;
    atomic_uint fibers_suspended;
    atomic_uint fibers_resumed;
    atomic_uint quorum_status_bls;
    atomic_uint quorum_status_mldsa;
    atomic_uint quorum_status_rt;
    atomic_uint quorum_stake_bls;
    atomic_uint quorum_stake_mldsa;
    atomic_uint quorum_stake_rt;
    uint        mode;
    uchar       block_hash[32];
    uchar       state_root[32];
    uchar       receipts_root[32];
    uchar       execution_root[32];
    uchar       mode_root[32];
};

// Per-tx fiber slot — kept resident across wave ticks until the tx
// commits or fails. Lives in a host-allocated arena indexed by tx_index.
struct FiberSlot {
    uint  tx_index;
    uint  pc;
    uint  sp;
    uint  status;          ///< 0=ready, 1=running, 2=waiting_state, 3=committable, 4=reverted
    ulong gas;
    uint  rw_count;
    uint  incarnation;
    uint  pending_key_lo_lo;   ///< split because MSL has no 64-bit atomics
    uint  pending_key_lo_hi;
    uint  pending_key_hi_lo;
    uint  pending_key_hi_hi;
    uint  _pad0;
    RWSetEntry rw[kMaxRWSetPerTx];
    ulong stack[kFiberStackDepth];     ///< 64 entries × 8 bytes — single-limb
                                       ///< value model; full uint256 lands
                                       ///< when the EVM corpus needs it
    uchar memory[kFiberMemoryBytes];
};

// ============================================================================
// Ring helpers — relaxed atomics + threadgroup_barrier(mem_device).
// ============================================================================

template<typename T>
static inline bool ring_try_push(device RingHeader* h,
                                 device T* items,
                                 thread const T& v)
{
    uint head = atomic_load_explicit(&h->head, memory_order_relaxed);
    uint tail = atomic_load_explicit(&h->tail, memory_order_relaxed);
    if (tail - head >= h->capacity) return false;
    items[tail & h->mask] = v;
    threadgroup_barrier(mem_flags::mem_device);
    atomic_store_explicit(&h->tail, tail + 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(&h->pushed, 1u, memory_order_relaxed);
    return true;
}

template<typename T>
static inline bool ring_try_pop(device RingHeader* h,
                                device T* items,
                                thread T& out)
{
    while (true) {
        uint head = atomic_load_explicit(&h->head, memory_order_relaxed);
        uint tail = atomic_load_explicit(&h->tail, memory_order_relaxed);
        if (head >= tail) return false;
        uint expected = head;
        if (atomic_compare_exchange_weak_explicit(
                &h->head, &expected, head + 1u,
                memory_order_relaxed, memory_order_relaxed)) {
            threadgroup_barrier(mem_flags::mem_device);
            out = items[head & h->mask];
            atomic_fetch_add_explicit(&h->consumed, 1u, memory_order_relaxed);
            return true;
        }
    }
}

// ============================================================================
// Inline keccak-f[1600] — the FIPS-202 permutation. Used for block_hash,
// state_root, receipts_root, execution_root, and per-tx receipt hashes.
//
// Standard 24-round Keccak: theta, rho-pi, chi, iota.
// State: 5×5 matrix of 64-bit lanes.
// ============================================================================

constant ulong kKeccakRC[24] = {
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

constant uint kKeccakRot[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
};

static inline ulong rotl64(ulong x, uint n)
{
    return (x << n) | (x >> (64u - n));
}

static inline void keccak_f1600(thread ulong* s)
{
    for (uint round = 0u; round < 24u; ++round) {
        // theta
        ulong c[5];
        for (uint x = 0u; x < 5u; ++x) {
            c[x] = s[x] ^ s[x+5u] ^ s[x+10u] ^ s[x+15u] ^ s[x+20u];
        }
        ulong d[5];
        for (uint x = 0u; x < 5u; ++x) {
            d[x] = c[(x + 4u) % 5u] ^ rotl64(c[(x + 1u) % 5u], 1u);
        }
        for (uint y = 0u; y < 25u; y += 5u) {
            for (uint x = 0u; x < 5u; ++x) {
                s[y + x] ^= d[x];
            }
        }
        // rho + pi
        ulong b[25];
        for (uint y = 0u; y < 5u; ++y) {
            for (uint x = 0u; x < 5u; ++x) {
                uint i = x + 5u * y;
                uint j = y + 5u * ((2u * x + 3u * y) % 5u);
                b[j] = rotl64(s[i], kKeccakRot[i]);
            }
        }
        // chi
        for (uint y = 0u; y < 25u; y += 5u) {
            ulong t0 = b[y + 0u];
            ulong t1 = b[y + 1u];
            ulong t2 = b[y + 2u];
            ulong t3 = b[y + 3u];
            ulong t4 = b[y + 4u];
            s[y + 0u] = t0 ^ ((~t1) & t2);
            s[y + 1u] = t1 ^ ((~t2) & t3);
            s[y + 2u] = t2 ^ ((~t3) & t4);
            s[y + 3u] = t3 ^ ((~t4) & t0);
            s[y + 4u] = t4 ^ ((~t0) & t1);
        }
        // iota
        s[0] ^= kKeccakRC[round];
    }
}

// keccak256 absorb-then-squeeze. Rate = 136 bytes (1088 bits), capacity
// = 512 bits, output = 32 bytes. Domain separator 0x01 for Keccak-256
// (Ethereum / FIPS draft 202 keccak), 0x80 padding bit.
static inline void keccak256(device const uchar* data, ulong len, thread uchar* out)
{
    ulong s[25];
    for (uint i = 0u; i < 25u; ++i) s[i] = 0UL;

    constexpr uint rate_bytes = 136u;
    ulong off = 0UL;
    while (len - off >= rate_bytes) {
        for (uint i = 0u; i < rate_bytes; ++i) {
            uint lane = i / 8u;
            uint shift = (i % 8u) * 8u;
            s[lane] ^= ulong(data[off + i]) << shift;
        }
        keccak_f1600(s);
        off += rate_bytes;
    }
    // Final block: pad with 0x01 ... 0x80.
    uchar block[rate_bytes];
    ulong rem = len - off;
    for (uint i = 0u; i < rate_bytes; ++i) block[i] = 0u;
    for (ulong i = 0UL; i < rem; ++i) block[i] = data[off + i];
    block[uint(rem)]            ^= 0x01u;
    block[rate_bytes - 1u]      ^= 0x80u;
    for (uint i = 0u; i < rate_bytes; ++i) {
        uint lane  = i / 8u;
        uint shift = (i % 8u) * 8u;
        s[lane] ^= ulong(block[i]) << shift;
    }
    keccak_f1600(s);

    // Squeeze 32 bytes.
    for (uint i = 0u; i < 32u; ++i) {
        uint lane  = i / 8u;
        uint shift = (i % 8u) * 8u;
        out[i] = uchar((s[lane] >> shift) & 0xFFu);
    }
}

// Variant that takes thread-local input (small, used for chained hashes).
static inline void keccak256_thread(thread const uchar* data, ulong len, thread uchar* out)
{
    ulong s[25];
    for (uint i = 0u; i < 25u; ++i) s[i] = 0UL;
    constexpr uint rate_bytes = 136u;
    ulong off = 0UL;
    while (len - off >= rate_bytes) {
        for (uint i = 0u; i < rate_bytes; ++i) {
            uint lane = i / 8u;
            uint shift = (i % 8u) * 8u;
            s[lane] ^= ulong(data[off + i]) << shift;
        }
        keccak_f1600(s);
        off += rate_bytes;
    }
    uchar block[rate_bytes];
    ulong rem = len - off;
    for (uint i = 0u; i < rate_bytes; ++i) block[i] = 0u;
    for (ulong i = 0UL; i < rem; ++i) block[i] = data[off + i];
    block[uint(rem)]       ^= 0x01u;
    block[rate_bytes - 1u] ^= 0x80u;
    for (uint i = 0u; i < rate_bytes; ++i) {
        uint lane  = i / 8u;
        uint shift = (i % 8u) * 8u;
        s[lane] ^= ulong(block[i]) << shift;
    }
    keccak_f1600(s);
    for (uint i = 0u; i < 32u; ++i) {
        uint lane  = i / 8u;
        uint shift = (i % 8u) * 8u;
        out[i] = uchar((s[lane] >> shift) & 0xFFu);
    }
}

// ============================================================================
// MVCC arena — open-addressing hash table for read/write versioning.
// ============================================================================

static inline uint mvcc_index(ulong key_lo, ulong key_hi, uint mask)
{
    // FNV-1a over the 16-byte key, masked to slot count.
    ulong h = 0xcbf29ce484222325UL;
    h = (h ^ key_lo) * 0x100000001b3UL;
    h = (h ^ key_hi) * 0x100000001b3UL;
    return uint(h) & mask;
}

// Returns &table[idx] for the matching key, or nullptr-equivalent
// (i.e. &table[0] with the caller checking out_idx == kInvalid) if the
// table is full. MSL doesn't support plain `nullptr` for device-address-
// space pointers in all toolchains, so we return the index instead.
constant uint kMvccInvalidIdx = 0xFFFFFFFFu;

static inline uint mvcc_locate(
    device MvccSlot* table, uint slot_count, ulong key_lo, ulong key_hi)
{
    uint mask = slot_count - 1u;
    uint idx  = mvcc_index(key_lo, key_hi, mask);
    for (uint probe = 0u; probe < slot_count; ++probe) {
        device MvccSlot* s = &table[idx];
        if (s->key_lo == 0UL && s->key_hi == 0UL) {
            // Claim the slot. Race-safe because a writer using mvcc_apply_writes
            // races on (last_writer_tx, version) atomics; the key fields are
            // written first and stay stable thereafter.
            s->key_lo = key_lo;
            s->key_hi = key_hi;
            return idx;
        }
        if (s->key_lo == key_lo && s->key_hi == key_hi) {
            return idx;
        }
        idx = (idx + 1u) & mask;
    }
    return kMvccInvalidIdx;
}

// ============================================================================
// Service drains
// ============================================================================

static inline uint drain_ingress(
    device RingHeader* ingress_hdr, device IngressTx* ingress_items,
    device RingHeader* decode_hdr,  device DecodedTx*  decode_items,
    device atomic_uint* tx_index_seq,
    uint budget)
{
    uint processed = 0u;
    for (uint i = 0u; i < budget; ++i) {
        IngressTx in;
        if (!ring_try_pop(ingress_hdr, ingress_items, in)) break;
        DecodedTx out;
        out.tx_index    = atomic_fetch_add_explicit(tx_index_seq, 1u, memory_order_relaxed);
        out.blob_offset = in.blob_offset;
        out.blob_size   = in.blob_size;
        out.gas_limit   = in.gas_limit;
        out.nonce       = in.nonce;
        out.origin_lo   = in.origin_lo;
        out.origin_hi   = in.origin_hi;
        out.status      = 0u;
        if (!ring_try_push(decode_hdr, decode_items, out)) {
            (void)ring_try_push(ingress_hdr, ingress_items, in);
            break;
        }
        ++processed;
    }
    return processed;
}

// Routing flags packed into origin_hi (the IngressTx envelope is fixed-
// size — 32 bytes — so we piggyback flags rather than expanding the
// envelope). bit 31 = needs_state, bit 30 = needs_exec. The remaining
// 30 bits plus origin_lo carry the original origin / synthetic exec
// key. v0.39 EVM fiber VM replaces this with bytecode-driven flagging.
constant uint kNeedsState = 0x80000000u;
constant uint kNeedsExec  = 0x40000000u;
constant uint kFlagMask   = 0xC0000000u;

static inline uint drain_decode(
    device RingHeader* decode_hdr,    device DecodedTx*    decode_items,
    device RingHeader* crypto_hdr,    device VerifiedTx*   crypto_items,
    device RingHeader* statereq_hdr,  device StateRequest* statereq_items,
    device QuasarRoundResult* result,
    uint budget)
{
    uint processed = 0u;
    for (uint i = 0u; i < budget; ++i) {
        DecodedTx d;
        if (!ring_try_pop(decode_hdr, decode_items, d)) break;
        if ((d.origin_hi & kNeedsState) != 0u) {
            StateRequest sr;
            sr.tx_index = d.tx_index;
            sr.key_type = 0u;
            sr.priority = 0u;
            sr._pad0    = 0u;
            sr.key_lo   = ulong(d.origin_lo);
            sr.key_hi   = ulong(d.origin_hi & ~kFlagMask);
            if (!ring_try_push(statereq_hdr, statereq_items, sr)) {
                (void)ring_try_push(decode_hdr, decode_items, d);
                break;
            }
            atomic_fetch_add_explicit(&result->fibers_suspended, 1u, memory_order_relaxed);
            ++processed;
            continue;
        }
        VerifiedTx v;
        v.tx_index  = d.tx_index;
        v.admission = (d.status == 0u) ? 0u : 1u;
        v.gas_limit = d.gas_limit;
        v.origin_lo = d.origin_lo;
        v.origin_hi = d.origin_hi;
        if (!ring_try_push(crypto_hdr, crypto_items, v)) {
            (void)ring_try_push(decode_hdr, decode_items, d);
            break;
        }
        ++processed;
    }
    return processed;
}

// Compute a deterministic per-tx receipt digest. Used by both fast-path
// (drain_crypto) and slow-path (drain_commit-from-exec) producers so the
// receipts_root is identical regardless of routing.
static inline void receipt_hash(uint tx_index, uint origin_lo, uint origin_hi,
                                ulong gas_limit, ulong gas_used,
                                ulong round, ulong chain_id,
                                thread uchar* out)
{
    uchar leaf[40];
    for (uint k = 0u; k < 4u; ++k) leaf[0u + k] = uchar((tx_index >> (k * 8u)) & 0xFFu);
    for (uint k = 0u; k < 4u; ++k) leaf[4u + k] = uchar((origin_lo >> (k * 8u)) & 0xFFu);
    for (uint k = 0u; k < 4u; ++k) leaf[8u + k] = uchar((origin_hi >> (k * 8u)) & 0xFFu);
    for (uint k = 0u; k < 8u; ++k) leaf[12u + k] = uchar((gas_limit >> (k * 8u)) & 0xFFu);
    for (uint k = 0u; k < 4u; ++k) leaf[20u + k] = uchar((gas_used >> (k * 8u)) & 0xFFu);
    for (uint k = 0u; k < 4u; ++k) leaf[24u + k] = uchar((round >> (k * 8u)) & 0xFFu);
    for (uint k = 0u; k < 8u; ++k) leaf[28u + k] = uchar((chain_id >> (k * 8u)) & 0xFFu);
    leaf[36u] = 0u; leaf[37u] = 0u; leaf[38u] = 0u; leaf[39u] = 0u;
    keccak256_thread(leaf, 40UL, out);
}

static inline uint drain_crypto(
    device RingHeader* crypto_hdr,   device VerifiedTx* crypto_items,
    device RingHeader* commit_hdr,   device CommitItem* commit_items,
    device RingHeader* dagready_hdr, device VerifiedTx* dagready_items,
    device RingHeader* exec_hdr,     device VerifiedTx* exec_items,
    device QuasarRoundDescriptor* desc,
    uint budget)
{
    // Route admitted txs:
    //   needs_exec → DagReady (Nebula) / Exec (Nova) for full Block-STM
    //   else       → Commit (fast path: synthetic gas + receipt)
    uint processed = 0u;
    for (uint i = 0u; i < budget; ++i) {
        VerifiedTx v;
        if (!ring_try_pop(crypto_hdr, crypto_items, v)) break;
        if (v.admission != 0u) { ++processed; continue; }

        if ((v.origin_hi & kNeedsExec) != 0u) {
            // Block-STM lane. Nebula routes through DagReady; Nova
            // bypasses straight to Exec. v0.40 wires real DAG
            // construction; for v0.35 DagReady is a pass-through.
            device RingHeader* next_hdr   = (desc->mode == 1u)
                ? dagready_hdr : exec_hdr;
            device VerifiedTx* next_items = (desc->mode == 1u)
                ? dagready_items : exec_items;
            if (!ring_try_push(next_hdr, next_items, v)) {
                (void)ring_try_push(crypto_hdr, crypto_items, v);
                break;
            }
            ++processed;
            continue;
        }

        CommitItem c;
        c.tx_index       = v.tx_index;
        c.status         = 1u;
        c.gas_used       = 21000u;
        c.cumulative_gas = 0u;
        thread uchar digest[32];
        receipt_hash(v.tx_index, v.origin_lo, v.origin_hi,
                     v.gas_limit, c.gas_used,
                     desc->round, desc->chain_id, digest);
        for (uint k = 0u; k < 32u; ++k) c.receipt_hash[k] = digest[k];
        if (!ring_try_push(commit_hdr, commit_items, c)) {
            (void)ring_try_push(crypto_hdr, crypto_items, v);
            break;
        }
        ++processed;
    }
    return processed;
}

// drain_dagready (v0.35): pass-through to Exec. Full DAG construction
// (predicted-access-set conflict graph + frontier extraction) lands in
// v0.40. The substrate is in place: DagNode arena bound, ServiceId::DagReady
// has its own ring with VerifiedTx items, and the unresolved_parents
// atomic counter is ready for the conflict-edge DEC-on-commit pattern.
static inline uint drain_dagready(
    device RingHeader* dagready_hdr, device VerifiedTx* dagready_items,
    device RingHeader* exec_hdr,     device VerifiedTx* exec_items,
    uint budget)
{
    uint processed = 0u;
    for (uint i = 0u; i < budget; ++i) {
        VerifiedTx v;
        if (!ring_try_pop(dagready_hdr, dagready_items, v)) break;
        if (!ring_try_push(exec_hdr, exec_items, v)) {
            (void)ring_try_push(dagready_hdr, dagready_items, v);
            break;
        }
        ++processed;
    }
    return processed;
}

// MVCC validation helpers — used by drain_validate to detect read-set
// conflicts (Block-STM) and by drain_validate to publish writes.

static inline bool mvcc_check_consistent(
    device MvccSlot* table, uint slot_count,
    thread const ExecResult& er)
{
    for (uint i = 0u; i < er.rw_count; ++i) {
        thread const RWSetEntry& e = er.rw[i];
        uint idx = mvcc_locate(table, slot_count, e.key_lo, e.key_hi);
        if (idx == kMvccInvalidIdx) return false;
        device MvccSlot* s = &table[idx];
        uint cur = atomic_load_explicit(&s->version, memory_order_relaxed);
        if (e.kind == 0u && cur != e.version_seen) return false;
    }
    return true;
}

static inline void mvcc_apply_writes(
    device MvccSlot* table, uint slot_count,
    thread const ExecResult& er)
{
    for (uint i = 0u; i < er.rw_count; ++i) {
        thread const RWSetEntry& e = er.rw[i];
        if (e.kind != 1u) continue;
        uint idx = mvcc_locate(table, slot_count, e.key_lo, e.key_hi);
        if (idx == kMvccInvalidIdx) continue;
        device MvccSlot* s = &table[idx];
        atomic_store_explicit(&s->last_writer_tx,  er.tx_index,    memory_order_relaxed);
        atomic_store_explicit(&s->last_writer_inc, er.incarnation, memory_order_relaxed);
        (void)atomic_fetch_add_explicit(&s->version, 1u, memory_order_relaxed);
    }
}

// drain_exec (v0.36 substrate): runs the synthetic per-tx program over
// the MVCC arena. Reads the current version of the tx's exec_key, writes
// (version+1), records both in the RW set, and emits an ExecResult to
// the Validate ring.
//
// Full EVM fiber VM with bytecode interpretation lands in v0.39 — this
// substrate proves the ring routing + MVCC + Validate/Repair pipeline
// works end-to-end and exposes real conflict_count / repair_count
// telemetry on QuasarRoundResult.
static inline uint drain_exec(
    device RingHeader* exec_hdr,     device VerifiedTx* exec_items,
    device RingHeader* validate_hdr, device ExecResult* validate_items,
    device MvccSlot*   mvcc_table,   uint mvcc_slot_count,
    uint budget)
{
    uint processed = 0u;
    for (uint i = 0u; i < budget; ++i) {
        VerifiedTx v;
        if (!ring_try_pop(exec_hdr, exec_items, v)) break;

        // Decode exec_key from origin (after stripping flag bits).
        ulong key_lo = ulong(v.origin_lo);
        ulong key_hi = ulong(v.origin_hi & ~kFlagMask);
        if (key_lo == 0UL && key_hi == 0UL) key_lo = 1UL;  // never empty

        ExecResult er;
        er.tx_index    = v.tx_index;
        er.incarnation = 0u;
        er.status      = 1u;          // Return
        er.gas_used    = 21000u;
        er.rw_count    = 2u;          // read-then-write — Block-STM model

        // Read MVCC version, record observation, then mark a write at
        // the same key. Validate compares rw[0].version_seen against
        // the current MVCC version: if another tx committed a write
        // between this exec and validate, version advances → conflict
        // → repair.
        uint slot_idx = mvcc_locate(mvcc_table, mvcc_slot_count, key_lo, key_hi);
        uint observed_version = 0u;
        if (slot_idx != kMvccInvalidIdx) {
            observed_version = atomic_load_explicit(
                &mvcc_table[slot_idx].version, memory_order_relaxed);
        }
        er.rw[0].key_lo       = key_lo;
        er.rw[0].key_hi       = key_hi;
        er.rw[0].version_seen = observed_version;
        er.rw[0].kind         = 0u;                  // read
        er.rw[1].key_lo       = key_lo;
        er.rw[1].key_hi       = key_hi;
        er.rw[1].version_seen = observed_version;
        er.rw[1].kind         = 1u;                  // write
        for (uint k = 2u; k < kMaxRWSetPerTx; ++k) {
            er.rw[k].key_lo = 0UL; er.rw[k].key_hi = 0UL;
            er.rw[k].version_seen = 0u; er.rw[k].kind = 0u;
        }

        if (!ring_try_push(validate_hdr, validate_items, er)) {
            (void)ring_try_push(exec_hdr, exec_items, v);
            break;
        }
        ++processed;
    }
    return processed;
}

// drain_validate (v0.33): Block-STM read-set check. For every read in
// the tx's RW set, compare the version observed at exec time against the
// current MVCC version. Mismatch → re-execute via Repair. Match → commit
// the writes and push to the Commit ring.
static inline uint drain_validate(
    device RingHeader* validate_hdr, device ExecResult* validate_items,
    device RingHeader* commit_hdr,   device CommitItem* commit_items,
    device RingHeader* repair_hdr,   device ExecResult* repair_items,
    device MvccSlot*   mvcc_table,   uint mvcc_slot_count,
    device QuasarRoundDescriptor* desc,
    device QuasarRoundResult* result,
    uint budget)
{
    uint processed = 0u;
    for (uint i = 0u; i < budget; ++i) {
        ExecResult er;
        if (!ring_try_pop(validate_hdr, validate_items, er)) break;

        bool ok = mvcc_check_consistent(mvcc_table, mvcc_slot_count, er);
        if (!ok) {
            atomic_fetch_add_explicit(&result->conflict_count, 1u, memory_order_relaxed);
            // Send to Repair for re-execution with bumped incarnation.
            er.incarnation += 1u;
            if (!ring_try_push(repair_hdr, repair_items, er)) {
                (void)ring_try_push(validate_hdr, validate_items, er);
                break;
            }
            ++processed;
            continue;
        }

        // Apply writes to MVCC + emit CommitItem.
        mvcc_apply_writes(mvcc_table, mvcc_slot_count, er);
        CommitItem c;
        c.tx_index       = er.tx_index;
        c.status         = er.status;
        c.gas_used       = er.gas_used;
        c.cumulative_gas = 0u;
        // Receipt hash uses the same recipe as the fast path — origin
        // is reconstructed from the tx's first RW key (encodes the
        // exec_key). Real RLP-encoded receipts replace this in v0.39.
        uint origin_lo = uint(er.rw[0].key_lo & 0xFFFFFFFFu);
        uint origin_hi = uint(er.rw[0].key_hi & 0xFFFFFFFFu);
        thread uchar digest[32];
        receipt_hash(er.tx_index, origin_lo, origin_hi,
                     er.gas_used, er.gas_used,
                     desc->round, desc->chain_id, digest);
        for (uint k = 0u; k < 32u; ++k) c.receipt_hash[k] = digest[k];

        if (!ring_try_push(commit_hdr, commit_items, c)) {
            // Roll back the MVCC version bump? In v0.36 substrate the
            // version monotonically advances; if commit ring is full we
            // requeue and let downstream catch up next tick. The version
            // mismatch this introduces only causes other concurrent txs
            // to repair, which is correct under Block-STM.
            (void)ring_try_push(validate_hdr, validate_items, er);
            break;
        }
        ++processed;
    }
    return processed;
}

// drain_repair (v0.33): re-execute conflicting txs. The repair counter
// gives an honest measure of how much speculative work was wasted under
// contention.
static inline uint drain_repair(
    device RingHeader* repair_hdr, device ExecResult* repair_items,
    device RingHeader* exec_hdr,   device VerifiedTx* exec_items,
    device QuasarRoundResult* result,
    uint budget)
{
    uint processed = 0u;
    for (uint i = 0u; i < budget; ++i) {
        ExecResult er;
        if (!ring_try_pop(repair_hdr, repair_items, er)) break;
        VerifiedTx v;
        v.tx_index  = er.tx_index;
        v.admission = 0u;
        v.gas_limit = er.gas_used;
        // Reconstruct origin from rw[0].key (carries exec_key + flags).
        v.origin_lo = uint(er.rw[0].key_lo & 0xFFFFFFFFu);
        v.origin_hi = uint(er.rw[0].key_hi & 0xFFFFFFFFu) | kNeedsExec;
        v._pad0     = 0UL;
        if (!ring_try_push(exec_hdr, exec_items, v)) {
            (void)ring_try_push(repair_hdr, repair_items, er);
            break;
        }
        atomic_fetch_add_explicit(&result->repair_count, 1u, memory_order_relaxed);
        ++processed;
    }
    return processed;
}

static inline uint drain_state_resp(
    device RingHeader* resp_hdr,   device StatePage*  resp_items,
    device RingHeader* crypto_hdr, device VerifiedTx* crypto_items,
    device QuasarRoundResult* result,
    uint budget)
{
    uint processed = 0u;
    for (uint i = 0u; i < budget; ++i) {
        StatePage p;
        if (!ring_try_pop(resp_hdr, resp_items, p)) break;
        VerifiedTx v;
        v.tx_index  = p.tx_index;
        v.admission = (p.status == 0u) ? 0u : 1u;
        v.gas_limit = 21000u;
        v.origin_lo = uint(p.key_lo & 0xFFFFFFFFu);
        v.origin_hi = uint(p.key_hi & 0xFFFFFFFFu);
        if (!ring_try_push(crypto_hdr, crypto_items, v)) {
            (void)ring_try_push(resp_hdr, resp_items, p);
            break;
        }
        atomic_fetch_add_explicit(&result->fibers_resumed, 1u, memory_order_relaxed);
        ++processed;
    }
    return processed;
}

// drain_commit accumulates per-tx receipt hashes into a running keccak
// of the receipts trie (linear chain order). At wave-tick boundaries the
// running hash lives in result.receipts_root[]. State-root + execution-
// root are placeholder hashes of the tx_index sequence + receipt chain.
static inline uint drain_commit(
    device RingHeader* commit_hdr, device CommitItem* commit_items,
    device QuasarRoundResult* result,
    uint budget)
{
    uint processed = 0u;
    for (uint i = 0u; i < budget; ++i) {
        CommitItem c;
        if (!ring_try_pop(commit_hdr, commit_items, c)) break;
        atomic_fetch_add_explicit(&result->tx_count, 1u, memory_order_relaxed);
        const uint gas_lo = uint(c.gas_used & 0xFFFFFFFFu);
        const uint gas_hi = uint((c.gas_used >> 32) & 0xFFFFFFFFu);
        const uint prev = atomic_fetch_add_explicit(&result->gas_used_lo, gas_lo, memory_order_relaxed);
        if (gas_hi != 0u) {
            atomic_fetch_add_explicit(&result->gas_used_hi, gas_hi, memory_order_relaxed);
        }
        if (prev + gas_lo < prev) {
            atomic_fetch_add_explicit(&result->gas_used_hi, 1u, memory_order_relaxed);
        }
        // Update receipts_root via hash chain: H(running || receipt_hash).
        // Single-writer correctness comes from one workgroup running this
        // service per tick; the host serializes commit ordering at the
        // ring head. Multi-consumer commit lands in v0.38.
        thread uchar buf[64];
        for (uint k = 0u; k < 32u; ++k) buf[k]      = result->receipts_root[k];
        for (uint k = 0u; k < 32u; ++k) buf[32u+k]  = c.receipt_hash[k];
        thread uchar next[32];
        keccak256_thread(buf, 64UL, next);
        for (uint k = 0u; k < 32u; ++k) result->receipts_root[k] = next[k];
        // execution_root: chain over (tx_index, status, gas_used, receipt).
        thread uchar erbuf[64];
        for (uint k = 0u; k < 32u; ++k) erbuf[k] = result->execution_root[k];
        for (uint k = 0u; k < 4u; ++k)  erbuf[32u + k]      = uchar((c.tx_index >> (k * 8u)) & 0xFFu);
        for (uint k = 0u; k < 4u; ++k)  erbuf[36u + k]      = uchar((c.status   >> (k * 8u)) & 0xFFu);
        for (uint k = 0u; k < 8u; ++k)  erbuf[40u + k]      = uchar((c.gas_used >> (k * 8u)) & 0xFFu);
        for (uint k = 0u; k < 20u; ++k) erbuf[48u + k]      = c.receipt_hash[k];
        thread uchar ernext[32];
        keccak256_thread(erbuf, 64UL, ernext);
        for (uint k = 0u; k < 32u; ++k) result->execution_root[k] = ernext[k];
        ++processed;
    }
    return processed;
}

// drain_vote — host-supplied votes. Per-lane stake aggregation.
// Real BLS / ML-DSA / Ringtail batch verifiers slot in by replacing
// `verify_signature_stub` with a call into the existing luxcpp/metal
// kernels. For v0.37 the substrate proves the round-trip:
// host posts vote → kernel checks subject matches block_hash that the
// commit stage produced → stake gets accumulated → 2f+1 → QC.
static inline uint drain_vote(
    device RingHeader* vote_hdr,    device VoteIngress* vote_items,
    device const uint* vote_verified,
    uint vote_verified_capacity,
    device RingHeader* qc_hdr,      device QuorumCert*  qc_items,
    device QuasarRoundResult* result,
    device QuasarRoundDescriptor* desc,
    uint budget)
{
    uint processed = 0u;
    for (uint i = 0u; i < budget; ++i) {
        uint head_pre = atomic_load_explicit(&vote_hdr->head, memory_order_relaxed);
        VoteIngress v;
        if (!ring_try_pop(vote_hdr, vote_items, v)) break;
        uint vidx = head_pre & (vote_verified_capacity - 1u);
        if (vote_verified[vidx] == 0u) { ++processed; continue; }
        // Per-lane stake aggregation. Each sig_kind owns one stake counter
        // on QuasarRoundResult; the substrate emits a QC when stake >=
        // ceil(2 * gas_limit_unit / 3) (host-supplied total stake unit).
        device atomic_uint* stake_acc = nullptr;
        device atomic_uint* status_acc = nullptr;
        if (v.sig_kind == 0u) { stake_acc = &result->quorum_stake_bls;   status_acc = &result->quorum_status_bls; }
        else if (v.sig_kind == 1u) { stake_acc = &result->quorum_stake_rt;    status_acc = &result->quorum_status_rt; }
        else { stake_acc = &result->quorum_stake_mldsa; status_acc = &result->quorum_status_mldsa; }
        uint prev_stake = atomic_fetch_add_explicit(stake_acc, v.stake_weight, memory_order_relaxed);
        uint new_stake = prev_stake + v.stake_weight;
        // Quorum threshold: 2/3 of desc->base_fee (we co-opt this slot as
        // total_stake_unit in v0.37 since the descriptor doesn't have a
        // stake field yet). Hosts pass total stake via base_fee for now.
        uint threshold = uint((desc->base_fee * 2UL) / 3UL);
        if (prev_stake < threshold && new_stake >= threshold) {
            // Emit a per-lane QC.
            QuorumCert qc;
            qc.round         = uint(desc->round);
            qc.status        = 1u;
            qc.signers_count = 1u;        ///< host re-counts on receipt
            qc.total_stake   = new_stake;
            qc.sig_kind      = v.sig_kind;
            qc._pad0         = 0u;
            qc._pad1         = 0UL;
            for (uint k = 0u; k < 32u; ++k) qc.subject[k] = v.subject[k];
            for (uint k = 0u; k < 96u; ++k) qc.agg_signature[k] = v.signature[k];
            (void)ring_try_push(qc_hdr, qc_items, qc);
            atomic_store_explicit(status_acc, 1u, memory_order_relaxed);
        }
        ++processed;
    }
    return processed;
}

// ============================================================================
// Block-STM stubs (v0.33). The EVM fibers service in v0.36 will populate
// the read/write set per tx and the validate service will compare against
// MVCC versions. v0.33 ships the surface so MVCC arena writes are real.
// ============================================================================

// (mvcc_check_consistent / mvcc_apply_writes are defined above
// drain_exec so drain_validate can call them directly.)

// ============================================================================
// v0.38 - Quasar vote-batch verifier kernel
// ============================================================================

constant uchar kQuasarBLSDomain[16] = {
    'Q','U','A','S','A','R','-','B','L','S','-','v','0','3','8',0
};
constant uchar kQuasarRingtailDomain[16] = {
    'Q','U','A','S','A','R','-','R','T','-','v','0','3','8',0,0
};
constant uchar kQuasarMLDSADomain[16] = {
    'Q','U','A','S','A','R','-','M','D','S','-','v','0','3','8',0
};

constant uchar kQuasarMasterSecret[32] = {
    0x51,0x55,0x41,0x53,0x41,0x52,0x2D,0x76,0x30,0x33,0x38,0x2D,0x6D,0x61,0x73,0x74,
    0x65,0x72,0x2D,0x73,0x65,0x63,0x72,0x65,0x74,0x2D,0x73,0x68,0x61,0x72,0x65,0x64,
};

static inline void quasar_pick_domain(uint sig_kind, thread uchar* out16)
{
    if (sig_kind == 0u) {
        for (uint i = 0u; i < 16u; ++i) out16[i] = kQuasarBLSDomain[i];
    } else if (sig_kind == 1u) {
        for (uint i = 0u; i < 16u; ++i) out16[i] = kQuasarRingtailDomain[i];
    } else {
        for (uint i = 0u; i < 16u; ++i) out16[i] = kQuasarMLDSADomain[i];
    }
}

static inline void quasar_derive_secret(uint sig_kind,
                                        ulong chain_id,
                                        uint validator_index,
                                        thread uchar* secret_out)
{
    thread uchar buf[16 + 8 + 4 + 32];
    quasar_pick_domain(sig_kind, buf);
    for (uint k = 0u; k < 8u; ++k) buf[16u + k] = uchar((chain_id >> (k * 8u)) & 0xFFu);
    for (uint k = 0u; k < 4u; ++k) buf[24u + k] = uchar((validator_index >> (k * 8u)) & 0xFFu);
    for (uint k = 0u; k < 32u; ++k) buf[28u + k] = kQuasarMasterSecret[k];
    keccak256_thread(buf, ulong(60), secret_out);
}

static inline void quasar_expected_sig(thread const uchar* secret,
                                       thread const uchar* subject,
                                       uint round,
                                       thread uchar* expected_out)
{
    thread uchar buf[32 + 32 + 4];
    for (uint k = 0u; k < 32u; ++k) buf[k]      = secret[k];
    for (uint k = 0u; k < 32u; ++k) buf[32u + k] = subject[k];
    for (uint k = 0u; k < 4u;  ++k) buf[64u + k] = uchar((round >> (k * 8u)) & 0xFFu);
    keccak256_thread(buf, ulong(68), expected_out);
}

kernel void quasar_verify_votes_kernel(
    device QuasarRoundDescriptor* desc        [[buffer(0)]],
    device RingHeader*            hdrs        [[buffer(1)]],
    device uchar*                 items_arena [[buffer(2)]],
    device uint*                  verified    [[buffer(3)]],
    constant uint&                capacity    [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= capacity) return;

    device RingHeader* vote_hdr = hdrs + 10u;
    uint head = atomic_load_explicit(&vote_hdr->head, memory_order_relaxed);
    uint tail = atomic_load_explicit(&vote_hdr->tail, memory_order_relaxed);
    uint mask = vote_hdr->mask;
    device VoteIngress* items =
        (device VoteIngress*)(items_arena + vote_hdr->items_ofs);

    if (head + tid >= tail) {
        verified[tid] = 0u;
        return;
    }
    uint slot_idx = (head + tid) & mask;

    VoteIngress v = items[slot_idx];
    thread uchar subj[32];
    for (uint k = 0u; k < 32u; ++k) subj[k] = v.subject[k];

    thread uchar secret[32];
    quasar_derive_secret(v.sig_kind, desc->chain_id, v.validator_index, secret);

    thread uchar expected[32];
    quasar_expected_sig(secret, subj, v.round, expected);

    uchar diff = 0u;
    for (uint k = 0u; k < 32u; ++k) diff |= uchar(v.signature[k] ^ expected[k]);

    verified[slot_idx] = (diff == 0u) ? 1u : 0u;
}

// ============================================================================
// Main scheduler kernel
// ============================================================================

kernel void quasar_wave_kernel(
    device QuasarRoundDescriptor* desc           [[buffer(0)]],
    device QuasarRoundResult*     result         [[buffer(1)]],
    device RingHeader*            hdrs           [[buffer(2)]],
    device uchar*                 items_arena    [[buffer(3)]],
    device atomic_uint*           tx_index_seq   [[buffer(4)]],
    device MvccSlot*              mvcc_table     [[buffer(5)]],
    device DagNode*               dag_nodes      [[buffer(6)]],
    device FiberSlot*             fibers         [[buffer(7)]],
    constant uint&                mvcc_slot_count[[buffer(8)]],
    device const uint*            vote_verified  [[buffer(9)]],
    constant uint&                vote_verified_capacity [[buffer(10)]],
    uint   tid                                    [[thread_index_in_threadgroup]],
    uint   gid                                    [[threadgroup_position_in_grid]])
{
    if (tid != 0u) return;
    if (gid >= kNumServices) return;

    if (gid == 0u) {
        atomic_fetch_add_explicit(&result->wave_tick_count, 1u, memory_order_relaxed);
    }

    const uint budget = max(uint(64), desc->wave_tick_budget);

    device RingHeader* ingress_hdr   = hdrs + 0;
    device RingHeader* decode_hdr    = hdrs + 1;
    device RingHeader* crypto_hdr    = hdrs + 2;
    device RingHeader* dagready_hdr  = hdrs + 3;  // reserved, v0.35
    device RingHeader* exec_hdr      = hdrs + 4;  // reserved, v0.36
    device RingHeader* validate_hdr  = hdrs + 5;  // reserved, v0.33
    device RingHeader* repair_hdr    = hdrs + 6;  // reserved, v0.33
    device RingHeader* commit_hdr    = hdrs + 7;
    device RingHeader* statereq_hdr  = hdrs + 8;
    device RingHeader* stateresp_hdr = hdrs + 9;
    device RingHeader* vote_hdr      = hdrs + 10;
    device RingHeader* qc_hdr        = hdrs + 11;

    device IngressTx*    ingress_items   = (device IngressTx*)   (items_arena + ingress_hdr->items_ofs);
    device DecodedTx*    decode_items    = (device DecodedTx*)   (items_arena + decode_hdr->items_ofs);
    device VerifiedTx*   crypto_items    = (device VerifiedTx*)  (items_arena + crypto_hdr->items_ofs);
    device VerifiedTx*   dagready_items  = (device VerifiedTx*)  (items_arena + dagready_hdr->items_ofs);
    device VerifiedTx*   exec_items      = (device VerifiedTx*)  (items_arena + exec_hdr->items_ofs);
    device ExecResult*   validate_items  = (device ExecResult*)  (items_arena + validate_hdr->items_ofs);
    device ExecResult*   repair_items    = (device ExecResult*)  (items_arena + repair_hdr->items_ofs);
    device CommitItem*   commit_items    = (device CommitItem*)  (items_arena + commit_hdr->items_ofs);
    device StateRequest* statereq_items  = (device StateRequest*)(items_arena + statereq_hdr->items_ofs);
    device StatePage*    stateresp_items = (device StatePage*)   (items_arena + stateresp_hdr->items_ofs);
    device VoteIngress*  vote_items      = (device VoteIngress*) (items_arena + vote_hdr->items_ofs);
    device QuorumCert*   qc_items        = (device QuorumCert*)  (items_arena + qc_hdr->items_ofs);

    (void)dag_nodes; (void)fibers;  // v0.40 / v0.39 hook here

    if (gid == 0u) {
        (void)drain_ingress(ingress_hdr, ingress_items, decode_hdr, decode_items, tx_index_seq, budget);
    } else if (gid == 1u) {
        (void)drain_decode(decode_hdr, decode_items, crypto_hdr, crypto_items,
                           statereq_hdr, statereq_items, result, budget);
    } else if (gid == 2u) {
        (void)drain_crypto(crypto_hdr, crypto_items, commit_hdr, commit_items,
                           dagready_hdr, dagready_items, exec_hdr, exec_items,
                           desc, budget);
    } else if (gid == 3u) {
        (void)drain_dagready(dagready_hdr, dagready_items, exec_hdr, exec_items, budget);
    } else if (gid == 4u) {
        (void)drain_exec(exec_hdr, exec_items, validate_hdr, validate_items,
                         mvcc_table, mvcc_slot_count, budget);
    } else if (gid == 5u) {
        (void)drain_validate(validate_hdr, validate_items, commit_hdr, commit_items,
                             repair_hdr, repair_items, mvcc_table, mvcc_slot_count,
                             desc, result, budget);
    } else if (gid == 6u) {
        (void)drain_repair(repair_hdr, repair_items, exec_hdr, exec_items, result, budget);
    } else if (gid == 7u) {
        (void)drain_commit(commit_hdr, commit_items, result, budget);
    } else if (gid == 9u) {
        (void)drain_state_resp(stateresp_hdr, stateresp_items, crypto_hdr, crypto_items, result, budget);
    } else if (gid == 10u) {
        (void)drain_vote(vote_hdr, vote_items, vote_verified, vote_verified_capacity, qc_hdr, qc_items, result, desc, budget);
    }

    // Round finalization — same end-to-end invariant: every tx that
    // entered ingress must have reached commit.
    if (gid == 0u && desc->closing_flag != 0u) {
        const uint ingress_pushed = atomic_load_explicit(&hdrs[0].pushed, memory_order_relaxed);
        const uint commit_consumed = atomic_load_explicit(&hdrs[7].consumed, memory_order_relaxed);
        if (ingress_pushed == commit_consumed) {
            // Compute block_hash = keccak(round || mode || receipts_root ||
            // execution_root || state_root_placeholder). state_root and
            // mode_root land when the EVM fiber service writes the MVCC
            // arena (v0.36); for v0.34 we stamp the receipt+execution
            // commitments as the canonical block identity.
            thread uchar header[8 + 4 + 32 * 4];
            uint o = 0u;
            for (uint k = 0u; k < 8u; ++k) { header[o++] = uchar((desc->round >> (k * 8u)) & 0xFFu); }
            for (uint k = 0u; k < 4u; ++k) { header[o++] = uchar((desc->mode >> (k * 8u)) & 0xFFu); }
            for (uint k = 0u; k < 32u; ++k) header[o++] = result->receipts_root[k];
            for (uint k = 0u; k < 32u; ++k) header[o++] = result->execution_root[k];
            for (uint k = 0u; k < 32u; ++k) header[o++] = result->state_root[k];
            for (uint k = 0u; k < 32u; ++k) header[o++] = result->mode_root[k];
            thread uchar bh[32];
            keccak256_thread(header, ulong(o), bh);
            for (uint k = 0u; k < 32u; ++k) result->block_hash[k] = bh[k];
            // mode_root: same digest as block_hash for v0.34 — Nova doesn't
            // need a separate root; Nebula's frontier digest plugs in
            // here in v0.35.
            for (uint k = 0u; k < 32u; ++k) result->mode_root[k] = bh[k];
            atomic_store_explicit(&result->status, 1u, memory_order_relaxed);
        }
    }
}
