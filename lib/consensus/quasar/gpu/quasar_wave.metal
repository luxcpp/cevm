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

// v0.44 — kNumServices grew to 17 with the addition of five per-chain
// transition services (PlatformVMTransition .. MPCVMTransition). The new
// rings are wave-tick addresses for future per-VM ingest; the substrate
// passes through them with no work to do today (host writes the descriptor
// roots directly), but they reserve the slot ordering so v0.45+ wiring is
// purely additive.
constant uint kNumServices       = 17u;
constant uint kMaxRWSetPerTx     = 8u;
constant uint kDefaultMvccSlots  = 8192u;
constant uint kMaxDagParents     = 4u;
constant uint kMaxDagChildren    = 16u;
constant uint kMaxPredictedKeys  = 4u;
constant uint kFiberStackDepth   = 64u;
constant uint kFiberStackLimbs   = 4u;       ///< v0.41 — 256-bit
constant uint kFiberMemoryBytes  = 1024u;
constant uint kFiberInstrBudget  = 100000u;  ///< per-fiber dispatch loop cap

// FiberSlot status — substrate semantics, reused by the v0.41 EVM VM.
constant uint kFiberReady        = 0u;
constant uint kFiberRunning      = 1u;
constant uint kFiberWaitingState = 2u;
constant uint kFiberCommittable  = 3u;
constant uint kFiberReverted     = 4u;

// ExecResult status — terminal disposition reported by drain_exec.
constant uint kExecStatusReturn  = 1u;
constant uint kExecStatusRevert  = 2u;
constant uint kExecStatusOOG     = 3u;
constant uint kExecStatusError   = 4u;
constant uint kExecStatusSuspend = 5u;

// Berlin-ish gas costs — must match quasar_gpu_layout.hpp.
constant ulong kGasDefault     = 3UL;
constant ulong kGasJumpdest    = 1UL;
constant ulong kGasSloadWarm   = 100UL;
constant ulong kGasSstore      = 5000UL;
constant ulong kGasKeccakBase  = 30UL;
constant ulong kGasKeccakWord  = 6UL;
constant ulong kGasExpByte     = 50UL;

// v0.40 — DagNode lifecycle states. drain_dagready re-checks state
// after publishing the parent→child edge; if it sees Committed, it
// backs out its unresolved_parents claim. drain_commit sets
// state=Committed AFTER walking children.
constant uint kDagNodeUnset      = 0u;
constant uint kDagNodeRegistered = 1u;
constant uint kDagNodeEmitted    = 2u;
constant uint kDagNodeCommitted  = 3u;
constant uint kInvalidWriter     = 0xFFFFFFFFu;

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
    uint  blob_offset;        ///< v0.41 — bytecode slice in code arena
    uint  blob_size;
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

// STM-002 (v0.42.2): atomic claim flag eliminates the torn-key window.
//   claim_state = 0  →  free
//   claim_state = 1  →  claim in progress (key fields not yet stable)
//   claim_state = 2  →  ready (key_lo/key_hi committed, safe to compare)
// mvcc_locate uses atomic_compare_exchange on claim_state to win the
// claim; the winner writes key bytes, fences, then publishes via a
// store of state=2.
struct alignas(16) MvccSlot {
    ulong key_lo;
    ulong key_hi;
    atomic_uint last_writer_tx;
    atomic_uint last_writer_inc;
    atomic_uint version;
    atomic_uint claim_state;            ///< STM-002 atomic claim
};

struct alignas(16) DagNode {
    uint tx_index;
    uint parent_count;
    atomic_uint unresolved_parents;
    atomic_uint child_count;
    uint parents[kMaxDagParents];
    uint children[kMaxDagChildren];
    // v0.40: cached envelope so drain_commit can re-emit a freed child
    // straight to Exec without re-popping from DagReady.
    ulong pending_gas_limit;
    uint  pending_origin_lo;
    uint  pending_origin_hi;
    uint  pending_admission;
    atomic_uint state;
    // v0.41: bytecode slice — drain_exec needs it on every re-emit.
    uint pending_blob_offset;
    uint pending_blob_size;
};

// v0.40 — per-key writer table. drain_dagready is single-threaded
// (gid=3,tid=0), so last_writer_tx is plain (not atomic).
struct alignas(16) DagWriterSlot {
    ulong key_lo;
    ulong key_hi;
    uint  last_writer_tx;
    uint  _pad0;
};

// v0.40 — predicted access entry, indexed by tx_index.
struct PredictedKey {
    ulong key_lo;
    ulong key_hi;
    uint  is_write;
    uint  valid;
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

// CERT-021/006/007: round + stake widened to uint64; stake_weight is now
// MAC-bound. Layout MUST match VoteIngress in quasar_gpu_layout.hpp.
struct alignas(16) VoteIngress {
    uint  validator_index;
    uint  sig_kind;
    ulong round;             ///< CERT-021
    ulong stake_weight;      ///< CERT-006/007
    ulong _pad0;
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

// CERT-003 (v0.42) + v0.44: epoch + P/Q/Z roots + total_stake + validator_count
// + host-precomputed certificate_subject + 5 new per-chain transition roots
// (X, A, B, M, F). Canonical 9-chain order in cert subject is P, C, X, Q, Z,
// A, B, M, F where C reuses parent_block_hash. Verifier rejects vote with
// subject != desc->certificate_subject (CERT-022). MUST match
// quasar_gpu_layout.hpp byte-for-byte.
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
    ulong epoch;                       ///< CERT-010
    ulong total_stake;                 ///< CERT-020
    uint  validator_count;             ///< CERT-023
    uint  _pad0;
    uchar pchain_validator_root[32];
    uchar qchain_ceremony_root[32];
    uchar zchain_vk_root[32];
    uchar certificate_subject[32];
    // v0.44 — five new per-chain transition roots.
    uchar xchain_execution_root[32];
    uchar achain_state_root[32];
    uchar bchain_state_root[32];
    uchar mchain_state_root[32];
    uchar fchain_state_root[32];
};

// CERT-007/004/022 (v0.42): uint64-split stake counters, per-validator
// dedup bitmaps (BLS / Ringtail / MLDSA). MUST match QuasarRoundResult in
// quasar_gpu_layout.hpp byte-for-byte.
#define QUASAR_VALIDATOR_BITMAP_BITS  256
#define QUASAR_VALIDATOR_BITMAP_WORDS 8
constant uint kValidatorBitmapBits  = QUASAR_VALIDATOR_BITMAP_BITS;
constant uint kValidatorBitmapWords = QUASAR_VALIDATOR_BITMAP_WORDS;

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
    atomic_uint quorum_stake_bls_lo;
    atomic_uint quorum_stake_bls_hi;
    atomic_uint quorum_stake_mldsa_lo;
    atomic_uint quorum_stake_mldsa_hi;
    atomic_uint quorum_stake_rt_lo;
    atomic_uint quorum_stake_rt_hi;
    uint        mode;
    atomic_uint subject_mismatch_count;
    atomic_uint dedup_skipped_count;
    atomic_uint repair_capped_count;       ///< STM-003 telemetry
    uchar       block_hash[32];
    uchar       state_root[32];
    uchar       receipts_root[32];
    uchar       execution_root[32];
    uchar       mode_root[32];
    // v0.44 — 9-chain root echoes + cert subject echo. Canonical order
    // matches compute_certificate_subject (P, C, X, Q, Z, A, B, M, F).
    uchar       pchain_root_echo[32];
    uchar       cchain_root_echo[32];
    uchar       xchain_root_echo[32];
    uchar       qchain_root_echo[32];
    uchar       zchain_root_echo[32];
    uchar       achain_root_echo[32];
    uchar       bchain_root_echo[32];
    uchar       mchain_root_echo[32];
    uchar       fchain_root_echo[32];
    uchar       certificate_subject_echo[32];
    atomic_uint validator_voted_bitmap[3][QUASAR_VALIDATOR_BITMAP_WORDS];
};

// Per-tx fiber slot — kept resident across wave ticks until the tx
// commits or fails. Lives in a host-allocated arena indexed by tx_index.
//
// v0.41: stack widened to 256 bits per entry (4 ulong limbs, LE). The EVM
// bytecode interpreter in drain_exec uses this directly. `gas_limit` is
// captured on entry so we can emit gas_used = gas_limit - gas at the
// terminal point. `msize` tracks the high-water mark of memory touches.
// `code_offset/code_size` snapshot the tx's bytecode slice in the code
// arena so the dispatch loop reads from a stable pointer across ticks.
struct FiberSlot {
    uint  tx_index;
    uint  pc;
    uint  sp;                  ///< stack depth in 256-bit entries
    uint  status;              ///< 0=ready, 1=running, 2=waiting_state, 3=committable, 4=reverted
    ulong gas;
    uint  rw_count;
    uint  incarnation;
    uint  pending_key_lo_lo;   ///< split because MSL has no 64-bit atomics
    uint  pending_key_lo_hi;
    uint  pending_key_hi_lo;
    uint  pending_key_hi_hi;
    uint  origin_lo;           ///< v0.41 — captured for ADDRESS/CALLER/ORIGIN
    uint  origin_hi;
    ulong gas_limit;           ///< v0.41 — entry gas, for gas_used computation
    uint  msize;               ///< v0.41 — memory high-water mark (bytes)
    uint  code_offset;         ///< v0.41 — bytecode slice in code arena
    uint  code_size;
    uint  _pad0;
    RWSetEntry rw[kMaxRWSetPerTx];
    ulong stack[kFiberStackDepth * kFiberStackLimbs];   ///< 64 × 4 limbs (256-bit)
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

// STM-002 (v0.42.2): atomic CAS-based slot claim eliminates torn-key
// reads. claim_state transitions 0→1 (winner) or 0→stay-zero (loser
// re-reads). Winner writes key bytes, then publishes claim_state=2 with
// a device-memory fence. Other threads can match the slot only when
// claim_state==2.
static inline uint mvcc_locate(
    device MvccSlot* table, uint slot_count, ulong key_lo, ulong key_hi)
{
    uint mask = slot_count - 1u;
    uint idx  = mvcc_index(key_lo, key_hi, mask);
    for (uint probe = 0u; probe < slot_count; ++probe) {
        device MvccSlot* s = &table[idx];
        uint state = atomic_load_explicit(&s->claim_state, memory_order_relaxed);
        if (state == 0u) {
            // Try to claim: CAS 0 → 1. Winner installs the key; loser
            // re-checks the slot against the (now claimed) keys.
            uint expected = 0u;
            bool won = atomic_compare_exchange_weak_explicit(
                &s->claim_state, &expected, 1u,
                memory_order_relaxed, memory_order_relaxed);
            if (won) {
                s->key_lo = key_lo;
                s->key_hi = key_hi;
                threadgroup_barrier(mem_flags::mem_device);
                atomic_store_explicit(&s->claim_state, 2u, memory_order_relaxed);
                return idx;
            }
            // Lost the CAS — fall through and re-read state.
            state = atomic_load_explicit(&s->claim_state, memory_order_relaxed);
        }
        // Spin until publication completes; bounded by physical thread count.
        for (uint w = 0u; w < 64u && state == 1u; ++w) {
            state = atomic_load_explicit(&s->claim_state, memory_order_relaxed);
        }
        if (state == 2u && s->key_lo == key_lo && s->key_hi == key_hi) {
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
        // STM-005/006 (v0.42.2): peek tx_index_seq to fill the envelope,
        // but do NOT increment until decode push succeeds. Backpressure
        // re-queues the IngressTx and on next tick the same tx claims the
        // same tx_index — predicted_keys[tx_index] indexing stays stable
        // across requeues.
        uint candidate_index = atomic_load_explicit(tx_index_seq, memory_order_relaxed);
        DecodedTx out;
        out.tx_index    = candidate_index;
        out.blob_offset = in.blob_offset;
        out.blob_size   = in.blob_size;
        out.gas_limit   = in.gas_limit;
        out.nonce       = in.nonce;
        out.origin_lo   = in.origin_lo;
        out.origin_hi   = in.origin_hi;
        out.status      = 0u;
        if (!ring_try_push(decode_hdr, decode_items, out)) {
            // Decode ring full — requeue WITHOUT incrementing tx_index_seq.
            (void)ring_try_push(ingress_hdr, ingress_items, in);
            break;
        }
        // Push succeeded — only now consume the tx_index slot. CAS guards
        // against another drain_ingress thread racing here (single-thread
        // dispatch in v0.42.2 but defensive for forward-compat).
        uint expected = candidate_index;
        bool consumed = atomic_compare_exchange_weak_explicit(
            tx_index_seq, &expected, candidate_index + 1u,
            memory_order_relaxed, memory_order_relaxed);
        (void)consumed;
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
        v.tx_index    = d.tx_index;
        v.admission   = (d.status == 0u) ? 0u : 1u;
        v.gas_limit   = d.gas_limit;
        v.origin_lo   = d.origin_lo;
        v.origin_hi   = d.origin_hi;
        v.blob_offset = d.blob_offset;
        v.blob_size   = d.blob_size;
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

// =============================================================================
// v0.40 — DAG construction (predicted-access-set conflict graph)
// =============================================================================
//
// Data flow (Nebula mode):
//   Host push_txs(): for tx at host-position k, copy predicted_access into
//     predicted_keys[k * kMaxPredictedKeys .. (k+1) * kMaxPredictedKeys].
//   drain_ingress() pops Ingress in FIFO order and assigns
//     tx_index = atomic_fetch_add(tx_index_seq, 1). Both host and GPU
//     process in insertion order within one round, so host slot k maps
//     to GPU tx_index k.
//   drain_dagready() ingests VerifiedTx, walks predicted_keys[tx_index],
//     locates a DagWriterSlot per key (open-addressing on (key_lo,key_hi)),
//     reads the most recent prior writer, registers a parent→child edge,
//     and (if write) updates the slot's last_writer_tx to this tx.
//   drain_commit(), on committing tx P, walks dag_nodes[P].children and
//     atomically decrements each child's unresolved_parents. When a child
//     hits zero, drain_commit re-emits it to Exec from the cached envelope
//     in dag_nodes[child].pending_*.
//
// Antichain (Prism frontier): the set of DagNodes whose unresolved_parents
// == 0 is the maximal antichain. drain_dagready emits exactly this set
// each tick. The frontier evolves as drain_commit decrements children.

constant uint kDagWriterInvalidIdx = 0xFFFFFFFFu;

static inline uint dag_writer_index(ulong key_lo, ulong key_hi, uint mask)
{
    ulong h = 0xcbf29ce484222325UL;
    h = (h ^ key_lo) * 0x100000001b3UL;
    h = (h ^ key_hi) * 0x100000001b3UL;
    return uint(h) & mask;
}

static inline uint dag_writer_locate(
    device DagWriterSlot* table, uint slot_count, ulong key_lo, ulong key_hi)
{
    uint mask = slot_count - 1u;
    uint idx  = dag_writer_index(key_lo, key_hi, mask);
    for (uint probe = 0u; probe < slot_count; ++probe) {
        device DagWriterSlot* s = &table[idx];
        if (s->key_lo == 0UL && s->key_hi == 0UL) {
            s->key_lo = key_lo;
            s->key_hi = key_hi;
            s->last_writer_tx = kInvalidWriter;
            return idx;
        }
        if (s->key_lo == key_lo && s->key_hi == key_hi) {
            return idx;
        }
        idx = (idx + 1u) & mask;
    }
    return kDagWriterInvalidIdx;
}

static inline VerifiedTx dag_envelope_to_verified(device const DagNode* n)
{
    VerifiedTx v;
    v.tx_index    = n->tx_index;
    v.admission   = n->pending_admission;
    v.gas_limit   = n->pending_gas_limit;
    v.origin_lo   = n->pending_origin_lo;
    v.origin_hi   = n->pending_origin_hi;
    v.blob_offset = n->pending_blob_offset;
    v.blob_size   = n->pending_blob_size;
    return v;
}

// drain_dagready (v0.40): real predicted-access-set DAG construction +
// antichain emission to Exec.
static inline uint drain_dagready(
    device RingHeader* dagready_hdr, device VerifiedTx* dagready_items,
    device RingHeader* exec_hdr,     device VerifiedTx* exec_items,
    device DagNode*    dag_nodes,    uint   dag_node_capacity,
    device DagWriterSlot* writer_table, uint writer_slot_count,
    device const PredictedKey* predicted_keys, uint predicted_capacity,
    uint budget)
{
    uint processed = 0u;
    for (uint i = 0u; i < budget; ++i) {
        VerifiedTx v;
        if (!ring_try_pop(dagready_hdr, dagready_items, v)) break;

        const uint tidx = v.tx_index;
        if (tidx >= dag_node_capacity) {
            // No DAG slot — fall back to direct Exec.
            if (!ring_try_push(exec_hdr, exec_items, v)) {
                (void)ring_try_push(dagready_hdr, dagready_items, v);
                break;
            }
            ++processed;
            continue;
        }

        device DagNode* T = &dag_nodes[tidx];
        T->tx_index           = tidx;
        T->pending_gas_limit  = v.gas_limit;
        T->pending_origin_lo  = v.origin_lo;
        T->pending_origin_hi  = v.origin_hi;
        T->pending_admission  = v.admission;
        T->pending_blob_offset = v.blob_offset;
        T->pending_blob_size   = v.blob_size;
        T->parent_count       = 0u;
        atomic_store_explicit(&T->child_count,        0u, memory_order_relaxed);
        atomic_store_explicit(&T->unresolved_parents, 0u, memory_order_relaxed);
        atomic_store_explicit(&T->state,              kDagNodeRegistered, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_device);

        const uint base = (tidx < predicted_capacity)
            ? (tidx * kMaxPredictedKeys) : 0u;
        const bool has_predicted = (tidx < predicted_capacity);
        for (uint k = 0u; k < kMaxPredictedKeys && has_predicted; ++k) {
            device const PredictedKey* pk = &predicted_keys[base + k];
            if (pk->valid == 0u) continue;
            ulong key_lo = pk->key_lo;
            ulong key_hi = pk->key_hi;
            if (key_lo == 0UL && key_hi == 0UL) continue;

            uint widx = dag_writer_locate(writer_table, writer_slot_count, key_lo, key_hi);
            if (widx == kDagWriterInvalidIdx) continue;

            uint prev = writer_table[widx].last_writer_tx;
            if (prev != kInvalidWriter && prev != tidx && prev < dag_node_capacity) {
                device DagNode* P = &dag_nodes[prev];
                uint p_state = atomic_load_explicit(&P->state, memory_order_relaxed);
                if (p_state != kDagNodeCommitted) {
                    (void)atomic_fetch_add_explicit(&T->unresolved_parents, 1u, memory_order_relaxed);
                    threadgroup_barrier(mem_flags::mem_device);
                    uint cidx = atomic_fetch_add_explicit(&P->child_count, 1u, memory_order_relaxed);
                    if (cidx < kMaxDagChildren) {
                        P->children[cidx] = tidx;
                    }
                    threadgroup_barrier(mem_flags::mem_device);
                    uint p_state2 = atomic_load_explicit(&P->state, memory_order_relaxed);
                    if (p_state2 == kDagNodeCommitted) {
                        (void)atomic_fetch_sub_explicit(&T->unresolved_parents, 1u, memory_order_relaxed);
                    } else {
                        if (T->parent_count < kMaxDagParents) {
                            T->parents[T->parent_count] = prev;
                            T->parent_count += 1u;
                        }
                    }
                }
            }

            if (pk->is_write != 0u) {
                writer_table[widx].last_writer_tx = tidx;
            }
        }

        uint unresolved = atomic_load_explicit(&T->unresolved_parents, memory_order_relaxed);
        if (unresolved == 0u) {
            VerifiedTx out = dag_envelope_to_verified(T);
            if (!ring_try_push(exec_hdr, exec_items, out)) {
                break;
            }
            atomic_store_explicit(&T->state, kDagNodeEmitted, memory_order_relaxed);
        }
        ++processed;
    }
    return processed;
}

// (Old pass-through drain_dagready replaced by v0.40 predicted-access-set
// version above. The signature changed; the kernel dispatch below was
// updated accordingly.)
static inline uint drain_dagready_passthrough(
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

// =============================================================================
// v0.41 — U256 arithmetic library (4 × 64-bit limbs, little-endian)
// =============================================================================
//
// Strategy:
//   add/sub  : limb-level carry chain
//   mul      : 4×4 schoolbook, low 256 bits
//   div/mod  : bit-at-a-time long division (256 iterations); correct on
//              every input including divide-by-zero (returns 0/x per EVM)
//   shl/shr  : limb-shift + bit-shift split
//   sar      : sign-extends from bit 255
//   sdiv/smod: two's-complement wrappers around divmod
//
// All operations are pure functions on `thread const U256&`; they don't
// touch device memory and don't allocate. The EVM dispatch loop pulls
// values onto the thread-local register file and writes results back
// to the device fiber stack.

struct U256 { ulong v[4]; };

static inline U256 u256_zero()      { U256 z; z.v[0]=0UL; z.v[1]=0UL; z.v[2]=0UL; z.v[3]=0UL; return z; }
static inline U256 u256_one()       { U256 z = u256_zero(); z.v[0] = 1UL; return z; }
static inline U256 u256_u64(ulong x){ U256 z = u256_zero(); z.v[0] = x; return z; }

static inline bool u256_iszero(thread const U256& a) {
    return (a.v[0] | a.v[1] | a.v[2] | a.v[3]) == 0UL;
}
static inline bool u256_eq(thread const U256& a, thread const U256& b) {
    return a.v[0]==b.v[0] && a.v[1]==b.v[1] && a.v[2]==b.v[2] && a.v[3]==b.v[3];
}
static inline bool u256_lt(thread const U256& a, thread const U256& b) {
    if (a.v[3] != b.v[3]) return a.v[3] < b.v[3];
    if (a.v[2] != b.v[2]) return a.v[2] < b.v[2];
    if (a.v[1] != b.v[1]) return a.v[1] < b.v[1];
    return a.v[0] < b.v[0];
}
static inline bool u256_slt(thread const U256& a, thread const U256& b) {
    bool an = (a.v[3] >> 63) != 0UL;
    bool bn = (b.v[3] >> 63) != 0UL;
    if (an != bn) return an;        ///< neg < non-neg
    return u256_lt(a, b);
}

static inline U256 u256_add(thread const U256& a, thread const U256& b) {
    U256 r;
    ulong carry = 0UL;
    for (uint i = 0u; i < 4u; ++i) {
        ulong sum = a.v[i] + b.v[i];
        ulong c1  = (sum < a.v[i]) ? 1UL : 0UL;
        ulong sum2 = sum + carry;
        ulong c2  = (sum2 < sum) ? 1UL : 0UL;
        r.v[i] = sum2;
        carry = c1 + c2;
    }
    return r;
}

static inline U256 u256_sub(thread const U256& a, thread const U256& b) {
    U256 r;
    ulong borrow = 0UL;
    for (uint i = 0u; i < 4u; ++i) {
        ulong bi  = b.v[i];
        ulong diff = a.v[i] - bi;
        ulong b1   = (a.v[i] < bi) ? 1UL : 0UL;
        ulong diff2 = diff - borrow;
        ulong b2    = (diff < borrow) ? 1UL : 0UL;
        r.v[i] = diff2;
        borrow = b1 + b2;
    }
    return r;
}

static inline U256 u256_neg(thread const U256& a) {
    U256 z = u256_zero();
    return u256_sub(z, a);
}

// Multiply two 64-bit values into a 128-bit pair (lo, hi).
static inline void mul64(ulong a, ulong b, thread ulong& lo, thread ulong& hi) {
    ulong a_lo = a & 0xFFFFFFFFUL;
    ulong a_hi = a >> 32;
    ulong b_lo = b & 0xFFFFFFFFUL;
    ulong b_hi = b >> 32;
    ulong ll = a_lo * b_lo;
    ulong lh = a_lo * b_hi;
    ulong hl = a_hi * b_lo;
    ulong hh = a_hi * b_hi;
    ulong mid = (ll >> 32) + (lh & 0xFFFFFFFFUL) + (hl & 0xFFFFFFFFUL);
    lo = (ll & 0xFFFFFFFFUL) | (mid << 32);
    hi = hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
}

static inline U256 u256_mul(thread const U256& a, thread const U256& b) {
    ulong r[4]; r[0]=0UL; r[1]=0UL; r[2]=0UL; r[3]=0UL;
    for (uint i = 0u; i < 4u; ++i) {
        ulong carry = 0UL;
        for (uint j = 0u; j + i < 4u; ++j) {
            ulong lo, hi;
            mul64(a.v[i], b.v[j], lo, hi);
            ulong sum1 = r[i+j] + lo;
            ulong c1   = (sum1 < r[i+j]) ? 1UL : 0UL;
            ulong sum2 = sum1 + carry;
            ulong c2   = (sum2 < sum1) ? 1UL : 0UL;
            r[i+j] = sum2;
            carry = hi + c1 + c2;
        }
    }
    U256 out; out.v[0]=r[0]; out.v[1]=r[1]; out.v[2]=r[2]; out.v[3]=r[3];
    return out;
}

static inline U256 u256_and(thread const U256& a, thread const U256& b) {
    U256 r; for (uint i=0u;i<4u;++i) r.v[i] = a.v[i] & b.v[i]; return r;
}
static inline U256 u256_or(thread const U256& a, thread const U256& b) {
    U256 r; for (uint i=0u;i<4u;++i) r.v[i] = a.v[i] | b.v[i]; return r;
}
static inline U256 u256_xor(thread const U256& a, thread const U256& b) {
    U256 r; for (uint i=0u;i<4u;++i) r.v[i] = a.v[i] ^ b.v[i]; return r;
}
static inline U256 u256_not(thread const U256& a) {
    U256 r; for (uint i=0u;i<4u;++i) r.v[i] = ~a.v[i]; return r;
}

static inline U256 u256_shl(thread const U256& a, uint n) {
    if (n >= 256u) return u256_zero();
    U256 r = u256_zero();
    uint limb_shift = n / 64u;
    uint bit_shift  = n % 64u;
    for (uint i = 0u; i < 4u; ++i) {
        uint dst = i + limb_shift;
        if (dst >= 4u) break;
        r.v[dst] |= (a.v[i] << bit_shift);
        if (bit_shift != 0u && dst + 1u < 4u) {
            r.v[dst + 1u] |= (a.v[i] >> (64u - bit_shift));
        }
    }
    return r;
}

static inline U256 u256_shr(thread const U256& a, uint n) {
    if (n >= 256u) return u256_zero();
    U256 r = u256_zero();
    uint limb_shift = n / 64u;
    uint bit_shift  = n % 64u;
    for (uint i = 0u; i < 4u; ++i) {
        if (i < limb_shift) continue;
        uint src = i;
        uint dst = i - limb_shift;
        r.v[dst] |= (a.v[src] >> bit_shift);
        if (bit_shift != 0u && src + 1u < 4u) {
            r.v[dst] |= (a.v[src + 1u] << (64u - bit_shift));
        }
    }
    return r;
}

static inline U256 u256_sar(thread const U256& a, uint n) {
    bool neg = (a.v[3] >> 63) != 0UL;
    if (n >= 256u) {
        if (!neg) return u256_zero();
        U256 ones; for (uint i=0u;i<4u;++i) ones.v[i] = 0xFFFFFFFFFFFFFFFFUL;
        return ones;
    }
    U256 r = u256_shr(a, n);
    if (neg) {
        // OR in the sign extension over the top n bits.
        uint keep = 256u - n;
        // Build a mask with bits [256-n .. 255] = 1.
        for (uint i = 0u; i < 4u; ++i) {
            uint base = i * 64u;
            for (uint b = 0u; b < 64u; ++b) {
                uint pos = base + b;
                if (pos >= keep) r.v[i] |= (1UL << b);
            }
        }
    }
    return r;
}

// 256-bit unsigned long division. Bit-at-a-time; 256 iterations, exact.
// On divide-by-zero returns (q=0, r=0) per EVM semantics.
static inline void u256_divmod(thread const U256& a, thread const U256& b,
                               thread U256& q, thread U256& r)
{
    if (u256_iszero(b)) { q = u256_zero(); r = u256_zero(); return; }
    q = u256_zero();
    r = u256_zero();
    for (int i = 255; i >= 0; --i) {
        // r = (r << 1) | bit_i(a)
        r = u256_shl(r, 1u);
        uint limb = uint(i) / 64u;
        uint bit  = uint(i) % 64u;
        ulong abit = (a.v[limb] >> bit) & 1UL;
        r.v[0] |= abit;
        if (!u256_lt(r, b)) {
            r = u256_sub(r, b);
            // q |= (1 << i)
            q.v[limb] |= (1UL << bit);
        }
    }
}

static inline U256 u256_sdiv(thread const U256& a, thread const U256& b) {
    if (u256_iszero(b)) return u256_zero();
    bool an = (a.v[3] >> 63) != 0UL;
    bool bn = (b.v[3] >> 63) != 0UL;
    U256 ua = an ? u256_neg(a) : a;
    U256 ub = bn ? u256_neg(b) : b;
    U256 q, r; u256_divmod(ua, ub, q, r);
    if (an != bn) q = u256_neg(q);
    return q;
}

static inline U256 u256_smod(thread const U256& a, thread const U256& b) {
    if (u256_iszero(b)) return u256_zero();
    bool an = (a.v[3] >> 63) != 0UL;
    bool bn = (b.v[3] >> 63) != 0UL;
    U256 ua = an ? u256_neg(a) : a;
    U256 ub = bn ? u256_neg(b) : b;
    U256 q, r; u256_divmod(ua, ub, q, r);
    if (an) r = u256_neg(r);
    return r;
}

// EVM BYTE: extract the i-th most-significant byte of x as a U256.
// Convention: i=0 returns the highest byte, i=31 the lowest. i>=32 → 0.
static inline U256 u256_byte(thread const U256& i_, thread const U256& x) {
    if (!u256_lt(i_, u256_u64(32UL))) return u256_zero();
    uint i = uint(i_.v[0]);
    uint pos = 31u - i;            ///< least-significant-byte index
    uint limb = pos / 8u;
    uint shift = (pos % 8u) * 8u;
    ulong b = (x.v[limb] >> shift) & 0xFFUL;
    return u256_u64(b);
}

// SIGNEXTEND: treat x as a (i+1)-byte value (i in [0..31]), sign-extend to
// 256 bits. i>=31 → x unchanged.
static inline U256 u256_signextend(thread const U256& i_, thread const U256& x) {
    if (!u256_lt(i_, u256_u64(31UL))) return x;
    uint i = uint(i_.v[0]);
    uint sign_bit_pos = (i + 1u) * 8u - 1u;
    uint limb = sign_bit_pos / 64u;
    uint bit  = sign_bit_pos % 64u;
    bool neg = ((x.v[limb] >> bit) & 1UL) != 0UL;
    // Build mask of bits [sign_bit_pos+1 .. 255].
    U256 r = x;
    for (uint k = sign_bit_pos + 1u; k < 256u; ++k) {
        uint kl = k / 64u;
        uint kb = k % 64u;
        if (neg) r.v[kl] |=  (1UL << kb);
        else     r.v[kl] &= ~(1UL << kb);
    }
    return r;
}

// EXP: a^b. Square-and-multiply over U256.
static inline U256 u256_exp(thread const U256& a, thread const U256& b) {
    U256 r = u256_one();
    U256 base = a;
    U256 e = b;
    for (int bit = 0; bit < 256; ++bit) {
        if ((e.v[0] & 1UL) != 0UL) r = u256_mul(r, base);
        e = u256_shr(e, 1u);
        if (u256_iszero(e)) break;
        base = u256_mul(base, base);
    }
    return r;
}

// Number of significant bytes in `e` (0..32). Used for EXP gas costing.
static inline uint u256_byte_len(thread const U256& e) {
    for (int i = 31; i >= 0; --i) {
        uint limb = uint(i) / 8u;
        uint shift = (uint(i) % 8u) * 8u;
        if (((e.v[limb] >> shift) & 0xFFUL) != 0UL) return uint(i) + 1u;
    }
    return 0u;
}

// =============================================================================
// Fiber stack / memory helpers
// =============================================================================

static inline void fiber_push(device FiberSlot* f, thread const U256& x)
{
    if (f->sp >= kFiberStackDepth) return;
    uint base = f->sp * kFiberStackLimbs;
    f->stack[base + 0u] = x.v[0];
    f->stack[base + 1u] = x.v[1];
    f->stack[base + 2u] = x.v[2];
    f->stack[base + 3u] = x.v[3];
    f->sp += 1u;
}

static inline U256 fiber_pop(device FiberSlot* f)
{
    U256 r = u256_zero();
    if (f->sp == 0u) return r;
    f->sp -= 1u;
    uint base = f->sp * kFiberStackLimbs;
    r.v[0] = f->stack[base + 0u];
    r.v[1] = f->stack[base + 1u];
    r.v[2] = f->stack[base + 2u];
    r.v[3] = f->stack[base + 3u];
    return r;
}

static inline U256 fiber_peek(device FiberSlot* f, uint depth)
{
    U256 r = u256_zero();
    if (depth >= f->sp) return r;
    uint idx = f->sp - 1u - depth;
    uint base = idx * kFiberStackLimbs;
    r.v[0] = f->stack[base + 0u];
    r.v[1] = f->stack[base + 1u];
    r.v[2] = f->stack[base + 2u];
    r.v[3] = f->stack[base + 3u];
    return r;
}

static inline void fiber_poke(device FiberSlot* f, uint depth, thread const U256& x)
{
    if (depth >= f->sp) return;
    uint idx = f->sp - 1u - depth;
    uint base = idx * kFiberStackLimbs;
    f->stack[base + 0u] = x.v[0];
    f->stack[base + 1u] = x.v[1];
    f->stack[base + 2u] = x.v[2];
    f->stack[base + 3u] = x.v[3];
}

// Memory MSTORE: write 32-byte big-endian U256 at byte offset.
// MSTORE8: write low byte. MLOAD: read 32-byte big-endian U256.
// All clamp to kFiberMemoryBytes; out-of-range is a no-op (still updates msize
// for accurate MSIZE reporting).
static inline void fiber_mstore(device FiberSlot* f, uint off, thread const U256& x)
{
    uint end = off + 32u;
    if (end > f->msize) f->msize = end;
    if (end > kFiberMemoryBytes) return;
    for (uint i = 0u; i < 32u; ++i) {
        // Big-endian: most-significant byte first.
        uint pos = 31u - i;             ///< least-significant-byte index
        uint limb = pos / 8u;
        uint shift = (pos % 8u) * 8u;
        f->memory[off + i] = uchar((x.v[limb] >> shift) & 0xFFUL);
    }
}

static inline void fiber_mstore8(device FiberSlot* f, uint off, thread const U256& x)
{
    uint end = off + 1u;
    if (end > f->msize) f->msize = end;
    if (end > kFiberMemoryBytes) return;
    f->memory[off] = uchar(x.v[0] & 0xFFUL);
}

static inline U256 fiber_mload(device FiberSlot* f, uint off)
{
    uint end = off + 32u;
    if (end > f->msize) f->msize = end;
    U256 r = u256_zero();
    if (end > kFiberMemoryBytes) return r;
    for (uint i = 0u; i < 32u; ++i) {
        uint pos = 31u - i;
        uint limb = pos / 8u;
        uint shift = (pos % 8u) * 8u;
        r.v[limb] |= (ulong(f->memory[off + i]) << shift);
    }
    return r;
}

// Append an entry to the fiber's RW set; saturates at kMaxRWSetPerTx
// (further entries are dropped — same conservative semantics as Block-STM
// CPU reference under read-set capacity pressure).
static inline void fiber_rw_add(device FiberSlot* f,
                                ulong key_lo, ulong key_hi,
                                uint version_seen, uint kind)
{
    if (f->rw_count >= kMaxRWSetPerTx) return;
    f->rw[f->rw_count].key_lo       = key_lo;
    f->rw[f->rw_count].key_hi       = key_hi;
    f->rw[f->rw_count].version_seen = version_seen;
    f->rw[f->rw_count].kind         = kind;
    f->rw_count += 1u;
}

// =============================================================================
// EVM opcode constants (subset implemented in v0.41)
// =============================================================================

constant uchar OP_STOP        = 0x00;
constant uchar OP_ADD         = 0x01;
constant uchar OP_MUL         = 0x02;
constant uchar OP_SUB         = 0x03;
constant uchar OP_DIV         = 0x04;
constant uchar OP_SDIV        = 0x05;
constant uchar OP_MOD         = 0x06;
constant uchar OP_SMOD        = 0x07;
constant uchar OP_ADDMOD      = 0x08;
constant uchar OP_MULMOD      = 0x09;
constant uchar OP_EXP         = 0x0a;
constant uchar OP_SIGNEXTEND  = 0x0b;
constant uchar OP_LT          = 0x10;
constant uchar OP_GT          = 0x11;
constant uchar OP_SLT         = 0x12;
constant uchar OP_SGT         = 0x13;
constant uchar OP_EQ          = 0x14;
constant uchar OP_ISZERO      = 0x15;
constant uchar OP_AND         = 0x16;
constant uchar OP_OR          = 0x17;
constant uchar OP_XOR         = 0x18;
constant uchar OP_NOT         = 0x19;
constant uchar OP_BYTE        = 0x1a;
constant uchar OP_SHL         = 0x1b;
constant uchar OP_SHR         = 0x1c;
constant uchar OP_SAR         = 0x1d;
constant uchar OP_KECCAK256   = 0x20;
constant uchar OP_ADDRESS     = 0x30;
constant uchar OP_ORIGIN      = 0x32;
constant uchar OP_CALLER      = 0x33;
constant uchar OP_CALLVALUE   = 0x34;
constant uchar OP_CALLDATALOAD = 0x35;
constant uchar OP_CALLDATASIZE = 0x36;
constant uchar OP_CHAINID     = 0x46;
constant uchar OP_GASLIMIT    = 0x45;
constant uchar OP_POP         = 0x50;
constant uchar OP_MLOAD       = 0x51;
constant uchar OP_MSTORE      = 0x52;
constant uchar OP_MSTORE8     = 0x53;
constant uchar OP_SLOAD       = 0x54;
constant uchar OP_SSTORE      = 0x55;
constant uchar OP_JUMP        = 0x56;
constant uchar OP_JUMPI       = 0x57;
constant uchar OP_PC          = 0x58;
constant uchar OP_MSIZE       = 0x59;
constant uchar OP_GAS         = 0x5a;
constant uchar OP_JUMPDEST    = 0x5b;
constant uchar OP_PUSH0       = 0x5f;
constant uchar OP_PUSH1       = 0x60;
constant uchar OP_PUSH32      = 0x7f;
constant uchar OP_DUP1        = 0x80;
constant uchar OP_DUP16       = 0x8f;
constant uchar OP_SWAP1       = 0x90;
constant uchar OP_SWAP16      = 0x9f;
constant uchar OP_RETURN      = 0xf3;
constant uchar OP_REVERT      = 0xfd;
constant uchar OP_INVALID     = 0xfe;

// Hash a slice of fiber memory with keccak256, writing 32 bytes to `out`.
// Mirrors keccak256() but reads from the device fiber memory pointer.
static inline void keccak256_fiber_mem(device const uchar* data, ulong len,
                                       thread uchar* out)
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

// =============================================================================
// drain_exec (v0.41) — real EVM bytecode interpreter
// =============================================================================
//
// Per-tx state lives in FiberSlot indexed by tx_index. Each call to
// drain_exec pops a VerifiedTx from the Exec ring, initializes (or
// re-initializes) the fiber from the tx envelope, and runs the dispatch
// loop until one of:
//
//   * STOP / RETURN              → status = Return
//   * REVERT                     → status = Revert
//   * INVALID / unknown opcode   → status = Error (consume all gas)
//   * gas == 0                   → status = OOG
//   * pc >= code_size            → status = Return  (implicit STOP)
//   * SLOAD on never-loaded slot → status = Suspend (StateRequest emitted)
//   * instruction budget hit     → status = OOG     (runaway guard)
//
// On any terminal status, the substrate emits an ExecResult to Validate.
// On suspend, the fiber stays resident in the slot and the tx is *not*
// pushed back to Exec; the cold-state response will resume it via
// drain_state_resp → Crypto → Exec (re-running drain_exec on the same
// fiber, which sees `status == kFiberWaitingState` and resumes from `pc`).
static inline uint drain_exec(
    device RingHeader* exec_hdr,     device VerifiedTx* exec_items,
    device RingHeader* validate_hdr, device ExecResult* validate_items,
    device RingHeader* statereq_hdr, device StateRequest* statereq_items,
    device MvccSlot*   mvcc_table,   uint mvcc_slot_count,
    device FiberSlot*  fibers,       uint fiber_capacity,
    device const uchar* code_arena,  uint code_arena_size,
    device QuasarRoundDescriptor* desc,
    device QuasarRoundResult* result,
    uint budget)
{
    uint processed = 0u;
    for (uint i = 0u; i < budget; ++i) {
        VerifiedTx v;
        if (!ring_try_pop(exec_hdr, exec_items, v)) break;

        // Bounds-check tx_index and clamp to fiber arena.
        if (v.tx_index >= fiber_capacity) {
            // No fiber slot — emit synthetic Error so drain_validate moves on.
            ExecResult er;
            er.tx_index    = v.tx_index;
            er.incarnation = 0u;
            er.status      = kExecStatusError;
            er.gas_used    = v.gas_limit;
            er.rw_count    = 0u;
            for (uint k = 0u; k < kMaxRWSetPerTx; ++k) {
                er.rw[k].key_lo = 0UL; er.rw[k].key_hi = 0UL;
                er.rw[k].version_seen = 0u; er.rw[k].kind = 0u;
            }
            if (!ring_try_push(validate_hdr, validate_items, er)) {
                (void)ring_try_push(exec_hdr, exec_items, v);
                break;
            }
            ++processed;
            continue;
        }

        device FiberSlot* f = &fibers[v.tx_index];

        // Initialize the fiber if it's fresh OR if we're re-running after
        // Repair (incarnation bump). Suspend resume reuses the existing
        // fiber state; it enters with status == kFiberWaitingState.
        bool is_resume = (f->status == kFiberWaitingState);
        if (!is_resume) {
            f->tx_index    = v.tx_index;
            f->pc          = 0u;
            f->sp          = 0u;
            f->status      = kFiberRunning;
            f->gas         = v.gas_limit;
            f->gas_limit   = v.gas_limit;
            f->rw_count    = 0u;
            f->incarnation = (v.admission == 0u) ? 0u : v.admission;
            f->origin_lo   = v.origin_lo;
            f->origin_hi   = v.origin_hi;
            f->msize       = 0u;
            f->code_offset = v.blob_offset;
            f->code_size   = v.blob_size;
        } else {
            f->status = kFiberRunning;
        }

        // Bytecode pointer + bounds. If blob_size==0 (legacy substrate
        // path: needs_exec without blob) we emit a synthetic R+W on the
        // origin key so existing v0.36 / Block-STM tests still see
        // contention behaviour, then return.
        device const uchar* code = code_arena + f->code_offset;
        uint code_size = f->code_size;
        bool legacy = (code_size == 0u || (f->code_offset + code_size) > code_arena_size);

        if (legacy) {
            // Synthesize one R+W at the origin-derived key — preserves
            // v0.36 substrate behaviour for tests that don't push real
            // bytecode (e.g. block_stm_independent_txs).
            ulong key_lo = ulong(f->origin_lo);
            ulong key_hi = ulong(f->origin_hi & ~kFlagMask);
            if (key_lo == 0UL && key_hi == 0UL) key_lo = 1UL;
            uint slot_idx = mvcc_locate(mvcc_table, mvcc_slot_count, key_lo, key_hi);
            uint observed_version = 0u;
            if (slot_idx != kMvccInvalidIdx) {
                observed_version = atomic_load_explicit(
                    &mvcc_table[slot_idx].version, memory_order_relaxed);
            }
            fiber_rw_add(f, key_lo, key_hi, observed_version, 0u);
            fiber_rw_add(f, key_lo, key_hi, observed_version, 1u);

            ExecResult er;
            er.tx_index    = v.tx_index;
            er.incarnation = f->incarnation;
            er.status      = kExecStatusReturn;
            er.gas_used    = 21000UL;
            er.rw_count    = f->rw_count;
            for (uint k = 0u; k < kMaxRWSetPerTx; ++k) {
                if (k < f->rw_count) er.rw[k] = f->rw[k];
                else { er.rw[k].key_lo = 0UL; er.rw[k].key_hi = 0UL;
                       er.rw[k].version_seen = 0u; er.rw[k].kind = 0u; }
            }
            if (!ring_try_push(validate_hdr, validate_items, er)) {
                (void)ring_try_push(exec_hdr, exec_items, v);
                break;
            }
            f->status = kFiberCommittable;
            ++processed;
            continue;
        }

        // ====================================================================
        // Dispatch loop
        // ====================================================================
        uint final_status = 0u;        ///< 0 = still running
        bool suspended = false;
        for (uint step = 0u; step < kFiberInstrBudget; ++step) {
            if (f->pc >= code_size) { final_status = kExecStatusReturn; break; }
            uchar op = code[f->pc];

            // Per-opcode gas. Default is 3 (kGasDefault); special cases below.
            ulong cost = kGasDefault;
            if      (op == OP_SLOAD)     cost = kGasSloadWarm;
            else if (op == OP_SSTORE)    cost = kGasSstore;
            else if (op == OP_KECCAK256) cost = kGasKeccakBase;
            else if (op == OP_JUMPDEST)  cost = kGasJumpdest;
            else if (op == OP_POP)       cost = 2UL;
            else if (op == OP_PC || op == OP_GAS || op == OP_MSIZE
                  || op == OP_ADDRESS || op == OP_CALLER || op == OP_ORIGIN
                  || op == OP_CALLVALUE || op == OP_CALLDATASIZE
                  || op == OP_CHAINID || op == OP_GASLIMIT) cost = 2UL;
            else if (op == OP_EXP) {
                // base + 50 * byte_len(exp); peek the exponent (top of stack
                // after `base`), so it's stack[sp-2].
                if (f->sp >= 2u) {
                    U256 e = fiber_peek(f, 0u);
                    cost = 10UL + kGasExpByte * ulong(u256_byte_len(e));
                } else cost = 10UL;
            }

            if (f->gas < cost) { final_status = kExecStatusOOG; break; }
            f->gas -= cost;

            if (op == OP_STOP) { final_status = kExecStatusReturn; break; }
            else if (op == OP_ADD) {
                U256 b = fiber_pop(f); U256 a = fiber_pop(f);
                fiber_push(f, u256_add(a, b));
                f->pc += 1u;
            }
            else if (op == OP_MUL) {
                U256 b = fiber_pop(f); U256 a = fiber_pop(f);
                fiber_push(f, u256_mul(a, b));
                f->pc += 1u;
            }
            else if (op == OP_SUB) {
                U256 b = fiber_pop(f); U256 a = fiber_pop(f);
                fiber_push(f, u256_sub(a, b));
                f->pc += 1u;
            }
            else if (op == OP_DIV) {
                U256 b = fiber_pop(f); U256 a = fiber_pop(f);
                U256 q, r; u256_divmod(a, b, q, r);
                fiber_push(f, q);
                f->pc += 1u;
            }
            else if (op == OP_SDIV) {
                U256 b = fiber_pop(f); U256 a = fiber_pop(f);
                fiber_push(f, u256_sdiv(a, b));
                f->pc += 1u;
            }
            else if (op == OP_MOD) {
                U256 b = fiber_pop(f); U256 a = fiber_pop(f);
                U256 q, r;
                if (u256_iszero(b)) r = u256_zero(); else u256_divmod(a, b, q, r);
                fiber_push(f, r);
                f->pc += 1u;
            }
            else if (op == OP_SMOD) {
                U256 b = fiber_pop(f); U256 a = fiber_pop(f);
                fiber_push(f, u256_smod(a, b));
                f->pc += 1u;
            }
            else if (op == OP_ADDMOD) {
                U256 m = fiber_pop(f); U256 b = fiber_pop(f); U256 a = fiber_pop(f);
                U256 s = u256_add(a, b);
                U256 q, r;
                if (u256_iszero(m)) r = u256_zero(); else u256_divmod(s, m, q, r);
                fiber_push(f, r);
                f->pc += 1u;
            }
            else if (op == OP_MULMOD) {
                U256 m = fiber_pop(f); U256 b = fiber_pop(f); U256 a = fiber_pop(f);
                U256 p = u256_mul(a, b);
                U256 q, r;
                if (u256_iszero(m)) r = u256_zero(); else u256_divmod(p, m, q, r);
                fiber_push(f, r);
                f->pc += 1u;
            }
            else if (op == OP_EXP) {
                U256 e = fiber_pop(f); U256 a = fiber_pop(f);
                fiber_push(f, u256_exp(a, e));
                f->pc += 1u;
            }
            else if (op == OP_SIGNEXTEND) {
                U256 x = fiber_pop(f); U256 i_ = fiber_pop(f);
                fiber_push(f, u256_signextend(i_, x));
                f->pc += 1u;
            }
            else if (op == OP_LT) {
                U256 b = fiber_pop(f); U256 a = fiber_pop(f);
                fiber_push(f, u256_lt(a, b) ? u256_one() : u256_zero());
                f->pc += 1u;
            }
            else if (op == OP_GT) {
                U256 b = fiber_pop(f); U256 a = fiber_pop(f);
                fiber_push(f, u256_lt(b, a) ? u256_one() : u256_zero());
                f->pc += 1u;
            }
            else if (op == OP_SLT) {
                U256 b = fiber_pop(f); U256 a = fiber_pop(f);
                fiber_push(f, u256_slt(a, b) ? u256_one() : u256_zero());
                f->pc += 1u;
            }
            else if (op == OP_SGT) {
                U256 b = fiber_pop(f); U256 a = fiber_pop(f);
                fiber_push(f, u256_slt(b, a) ? u256_one() : u256_zero());
                f->pc += 1u;
            }
            else if (op == OP_EQ) {
                U256 b = fiber_pop(f); U256 a = fiber_pop(f);
                fiber_push(f, u256_eq(a, b) ? u256_one() : u256_zero());
                f->pc += 1u;
            }
            else if (op == OP_ISZERO) {
                U256 a = fiber_pop(f);
                fiber_push(f, u256_iszero(a) ? u256_one() : u256_zero());
                f->pc += 1u;
            }
            else if (op == OP_AND) {
                U256 b = fiber_pop(f); U256 a = fiber_pop(f);
                fiber_push(f, u256_and(a, b));
                f->pc += 1u;
            }
            else if (op == OP_OR) {
                U256 b = fiber_pop(f); U256 a = fiber_pop(f);
                fiber_push(f, u256_or(a, b));
                f->pc += 1u;
            }
            else if (op == OP_XOR) {
                U256 b = fiber_pop(f); U256 a = fiber_pop(f);
                fiber_push(f, u256_xor(a, b));
                f->pc += 1u;
            }
            else if (op == OP_NOT) {
                U256 a = fiber_pop(f);
                fiber_push(f, u256_not(a));
                f->pc += 1u;
            }
            else if (op == OP_BYTE) {
                U256 x = fiber_pop(f); U256 i_ = fiber_pop(f);
                fiber_push(f, u256_byte(i_, x));
                f->pc += 1u;
            }
            else if (op == OP_SHL) {
                U256 v_ = fiber_pop(f); U256 sh = fiber_pop(f);
                uint n = (sh.v[1] | sh.v[2] | sh.v[3]) != 0UL ? 256u : uint(sh.v[0]);
                fiber_push(f, u256_shl(v_, n));
                f->pc += 1u;
            }
            else if (op == OP_SHR) {
                U256 v_ = fiber_pop(f); U256 sh = fiber_pop(f);
                uint n = (sh.v[1] | sh.v[2] | sh.v[3]) != 0UL ? 256u : uint(sh.v[0]);
                fiber_push(f, u256_shr(v_, n));
                f->pc += 1u;
            }
            else if (op == OP_SAR) {
                U256 v_ = fiber_pop(f); U256 sh = fiber_pop(f);
                uint n = (sh.v[1] | sh.v[2] | sh.v[3]) != 0UL ? 256u : uint(sh.v[0]);
                fiber_push(f, u256_sar(v_, n));
                f->pc += 1u;
            }
            else if (op == OP_KECCAK256) {
                U256 len = fiber_pop(f); U256 off = fiber_pop(f);
                uint o = uint(off.v[0]); uint l = uint(len.v[0]);
                ulong word_cost = kGasKeccakWord * ulong((l + 31u) / 32u);
                if (f->gas < word_cost) { final_status = kExecStatusOOG; break; }
                f->gas -= word_cost;
                if (o + l > kFiberMemoryBytes) { final_status = kExecStatusError; break; }
                if (o + l > f->msize) f->msize = o + l;
                uchar digest[32];
                keccak256_fiber_mem(&f->memory[o], ulong(l), digest);
                U256 r = u256_zero();
                for (uint k = 0u; k < 32u; ++k) {
                    uint pos = 31u - k;
                    uint limb = pos / 8u;
                    uint shift = (pos % 8u) * 8u;
                    r.v[limb] |= (ulong(digest[k]) << shift);
                }
                fiber_push(f, r);
                f->pc += 1u;
            }
            else if (op == OP_ADDRESS || op == OP_CALLER || op == OP_ORIGIN) {
                U256 r = u256_zero();
                r.v[0] = ulong(f->origin_lo) | (ulong(f->origin_hi & ~kFlagMask) << 32);
                fiber_push(f, r);
                f->pc += 1u;
            }
            else if (op == OP_CALLVALUE) {
                fiber_push(f, u256_zero());
                f->pc += 1u;
            }
            else if (op == OP_CALLDATALOAD) {
                (void)fiber_pop(f);     // offset (unused — calldata not wired in v0.41)
                fiber_push(f, u256_zero());
                f->pc += 1u;
            }
            else if (op == OP_CALLDATASIZE) {
                fiber_push(f, u256_zero());
                f->pc += 1u;
            }
            else if (op == OP_CHAINID) {
                fiber_push(f, u256_u64(desc->chain_id));
                f->pc += 1u;
            }
            else if (op == OP_GASLIMIT) {
                fiber_push(f, u256_u64(desc->gas_limit));
                f->pc += 1u;
            }
            else if (op == OP_POP) {
                (void)fiber_pop(f);
                f->pc += 1u;
            }
            else if (op == OP_MLOAD) {
                U256 off = fiber_pop(f);
                fiber_push(f, fiber_mload(f, uint(off.v[0])));
                f->pc += 1u;
            }
            else if (op == OP_MSTORE) {
                U256 v_ = fiber_pop(f); U256 off = fiber_pop(f);
                fiber_mstore(f, uint(off.v[0]), v_);
                f->pc += 1u;
            }
            else if (op == OP_MSTORE8) {
                U256 v_ = fiber_pop(f); U256 off = fiber_pop(f);
                fiber_mstore8(f, uint(off.v[0]), v_);
                f->pc += 1u;
            }
            else if (op == OP_MSIZE) {
                uint m = (f->msize + 31u) & ~uint(31);
                fiber_push(f, u256_u64(ulong(m)));
                f->pc += 1u;
            }
            else if (op == OP_SLOAD) {
                U256 key = fiber_pop(f);
                ulong key_lo = key.v[0];
                ulong key_hi = key.v[1] | key.v[2] | key.v[3];
                if (key_lo == 0UL && key_hi == 0UL) key_lo = 1UL;
                uint slot_idx = mvcc_locate(mvcc_table, mvcc_slot_count, key_lo, key_hi);

                // Cold-miss detection: a slot is cold when no tx has ever
                // observed it (version == 0 AND last_writer_tx still at the
                // initial 0). Cold reads suspend the fiber and emit a
                // StateRequest; the cold-state loop re-routes the tx
                // through Crypto → Exec, where we resume from the same pc.
                bool is_cold = false;
                if (slot_idx == kMvccInvalidIdx) {
                    is_cold = true;
                } else {
                    device MvccSlot* s = &mvcc_table[slot_idx];
                    uint cur_ver  = atomic_load_explicit(&s->version, memory_order_relaxed);
                    uint cur_lwtx = atomic_load_explicit(&s->last_writer_tx, memory_order_relaxed);
                    is_cold = (cur_ver == 0u && cur_lwtx == 0u);
                }

                if (is_cold) {
                    StateRequest sr;
                    sr.tx_index = v.tx_index;
                    sr.key_type = 1u;       // Storage
                    sr.priority = 0u;
                    sr._pad0    = 0u;
                    sr.key_lo   = key_lo;
                    sr.key_hi   = key_hi;
                    if (!ring_try_push(statereq_hdr, statereq_items, sr)) {
                        // Backpressure on StateRequest: re-push the tx,
                        // abort drain pass, retry next wave tick.
                        (void)ring_try_push(exec_hdr, exec_items, v);
                        return processed;
                    }
                    // Restore the popped key so resume re-runs SLOAD.
                    fiber_push(f, key);
                    f->status = kFiberWaitingState;
                    f->pending_key_lo_lo = uint(key_lo & 0xFFFFFFFFu);
                    f->pending_key_lo_hi = uint(key_lo >> 32);
                    f->pending_key_hi_lo = uint(key_hi & 0xFFFFFFFFu);
                    f->pending_key_hi_hi = uint(key_hi >> 32);
                    atomic_fetch_add_explicit(&result->fibers_suspended, 1u,
                                              memory_order_relaxed);
                    suspended = true;
                    break;
                }

                // Warm path: record the read in the RW set and push 0
                // (the slot value body persists into v0.45's value journal —
                // for now Block-STM only checks version-conflict shape).
                uint observed_version = (slot_idx == kMvccInvalidIdx) ? 0u
                    : atomic_load_explicit(&mvcc_table[slot_idx].version,
                                           memory_order_relaxed);
                fiber_rw_add(f, key_lo, key_hi, observed_version, 0u);
                fiber_push(f, u256_zero());
                f->pc += 1u;
            }
            else if (op == OP_SSTORE) {
                U256 val = fiber_pop(f); U256 key = fiber_pop(f);
                ulong key_lo = key.v[0];
                ulong key_hi = key.v[1] | key.v[2] | key.v[3];
                if (key_lo == 0UL && key_hi == 0UL) key_lo = 1UL;
                uint slot_idx = mvcc_locate(mvcc_table, mvcc_slot_count, key_lo, key_hi);
                uint observed_version = 0u;
                if (slot_idx != kMvccInvalidIdx) {
                    observed_version = atomic_load_explicit(
                        &mvcc_table[slot_idx].version, memory_order_relaxed);
                }
                fiber_rw_add(f, key_lo, key_hi, observed_version, 1u);
                (void)val;       // value body lands in v0.45 journal
                f->pc += 1u;
            }
            else if (op == OP_JUMP) {
                U256 dst = fiber_pop(f);
                uint d = uint(dst.v[0]);
                if (d >= code_size || code[d] != OP_JUMPDEST) {
                    final_status = kExecStatusError; break;
                }
                f->pc = d;
            }
            else if (op == OP_JUMPI) {
                U256 cond = fiber_pop(f); U256 dst = fiber_pop(f);
                if (!u256_iszero(cond)) {
                    uint d = uint(dst.v[0]);
                    if (d >= code_size || code[d] != OP_JUMPDEST) {
                        final_status = kExecStatusError; break;
                    }
                    f->pc = d;
                } else {
                    f->pc += 1u;
                }
            }
            else if (op == OP_PC) {
                fiber_push(f, u256_u64(ulong(f->pc)));
                f->pc += 1u;
            }
            else if (op == OP_GAS) {
                fiber_push(f, u256_u64(f->gas));
                f->pc += 1u;
            }
            else if (op == OP_JUMPDEST) {
                f->pc += 1u;
            }
            else if (op == OP_PUSH0) {
                fiber_push(f, u256_zero());
                f->pc += 1u;
            }
            else if (op >= OP_PUSH1 && op <= OP_PUSH32) {
                uint n = uint(op - OP_PUSH1) + 1u;
                U256 r = u256_zero();
                for (uint k = 0u; k < n; ++k) {
                    uint pos = f->pc + 1u + k;
                    uchar b = (pos < code_size) ? code[pos] : uchar(0);
                    uint bit_pos = (n - 1u - k) * 8u;
                    uint limb = bit_pos / 64u;
                    uint shift = bit_pos % 64u;
                    r.v[limb] |= ulong(b) << shift;
                }
                fiber_push(f, r);
                f->pc += 1u + n;
            }
            else if (op >= OP_DUP1 && op <= OP_DUP16) {
                uint depth = uint(op - OP_DUP1);
                U256 x = fiber_peek(f, depth);
                fiber_push(f, x);
                f->pc += 1u;
            }
            else if (op >= OP_SWAP1 && op <= OP_SWAP16) {
                uint depth = uint(op - OP_SWAP1) + 1u;
                U256 top = fiber_peek(f, 0u);
                U256 oth = fiber_peek(f, depth);
                fiber_poke(f, 0u, oth);
                fiber_poke(f, depth, top);
                f->pc += 1u;
            }
            else if (op == OP_RETURN) {
                (void)fiber_pop(f); (void)fiber_pop(f);
                final_status = kExecStatusReturn; break;
            }
            else if (op == OP_REVERT) {
                (void)fiber_pop(f); (void)fiber_pop(f);
                final_status = kExecStatusRevert; break;
            }
            else if (op == OP_INVALID) {
                final_status = kExecStatusError; break;
            }
            else {
                // Unimplemented opcode → INVALID semantics (consume all gas).
                // CALL/CREATE/EXTCODE/LOG/MCOPY/TLOAD/TSTORE/RETURNDATA*/
                // BLOCKHASH/BASEFEE/BLOBHASH land in v0.42–v0.45.
                final_status = kExecStatusError; break;
            }
        }

        if (suspended) {
            ++processed;
            continue;
        }

        if (final_status == 0u) {
            // Hit instruction budget — treat as OOG so Validate sees a
            // concrete status and the tx counts as terminal.
            final_status = kExecStatusOOG;
        }

        // INVALID / Error consume all gas; OOG also consumes all gas; Return
        // and Revert report what was actually used.
        ulong gas_used;
        if (final_status == kExecStatusError || final_status == kExecStatusOOG) {
            gas_used = f->gas_limit;
        } else {
            gas_used = (f->gas_limit > f->gas) ? (f->gas_limit - f->gas) : 0UL;
        }
        if (gas_used < 21000UL) gas_used = 21000UL;     // intrinsic floor

        ExecResult er;
        er.tx_index    = v.tx_index;
        er.incarnation = f->incarnation;
        er.status      = final_status;
        er.gas_used    = gas_used;
        er.rw_count    = f->rw_count;
        for (uint k = 0u; k < kMaxRWSetPerTx; ++k) {
            if (k < f->rw_count) {
                er.rw[k] = f->rw[k];
            } else {
                er.rw[k].key_lo = 0UL; er.rw[k].key_hi = 0UL;
                er.rw[k].version_seen = 0u; er.rw[k].kind = 0u;
            }
        }

        // Block-STM substrate guarantee: every tx that reaches Validate
        // touches at least one RW key so MVCC validation has something to
        // compare. Pure-compute programs (no SLOAD/SSTORE) get a
        // synthetic R+W on the origin-derived key — same shape as the
        // v0.36 substrate, so existing tests still see contention.
        if (er.rw_count == 0u) {
            ulong key_lo = ulong(f->origin_lo);
            ulong key_hi = ulong(f->origin_hi & ~kFlagMask);
            if (key_lo == 0UL && key_hi == 0UL) key_lo = 1UL;
            uint slot_idx = mvcc_locate(mvcc_table, mvcc_slot_count, key_lo, key_hi);
            uint observed_version = 0u;
            if (slot_idx != kMvccInvalidIdx) {
                observed_version = atomic_load_explicit(
                    &mvcc_table[slot_idx].version, memory_order_relaxed);
            }
            er.rw[0].key_lo = key_lo; er.rw[0].key_hi = key_hi;
            er.rw[0].version_seen = observed_version; er.rw[0].kind = 0u;
            er.rw[1].key_lo = key_lo; er.rw[1].key_hi = key_hi;
            er.rw[1].version_seen = observed_version; er.rw[1].kind = 1u;
            er.rw_count = 2u;
        }

        if (!ring_try_push(validate_hdr, validate_items, er)) {
            (void)ring_try_push(exec_hdr, exec_items, v);
            break;
        }
        f->status = kFiberCommittable;
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

        // STM-001 (v0.42.2): defer mvcc_apply_writes until AFTER successful
        // commit-ring push. Try the commit push first; on failure, requeue
        // the original ExecResult to Validate WITHOUT mutating MVCC. This
        // makes the commit order deterministic against host backpressure
        // (the commit-ring-full case no longer leaks a version bump).
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
            // Commit ring full — requeue to Validate WITHOUT mutating MVCC.
            (void)ring_try_push(validate_hdr, validate_items, er);
            break;
        }
        // Push succeeded; now safely apply writes to MVCC.
        mvcc_apply_writes(mvcc_table, mvcc_slot_count, er);
        ++processed;
    }
    return processed;
}

// STM-003 (v0.42.2): MAX_TOTAL_REPAIRS hard cap. LP-010 §4.0 mandates a
// deterministic abort once a tx's incarnation reaches MAX_TOTAL_REPAIRS;
// the abort is a CommitItem with status=Error so all validators agree on
// the same outcome. repair_capped_count telemetry tracks how often this
// fires.
#define QUASAR_MAX_TOTAL_REPAIRS 8u

static inline uint drain_repair(
    device RingHeader* repair_hdr, device ExecResult* repair_items,
    device RingHeader* exec_hdr,   device VerifiedTx* exec_items,
    device RingHeader* commit_hdr, device CommitItem* commit_items,
    device QuasarRoundDescriptor* desc,
    device QuasarRoundResult* result,
    uint budget)
{
    uint processed = 0u;
    for (uint i = 0u; i < budget; ++i) {
        ExecResult er;
        if (!ring_try_pop(repair_hdr, repair_items, er)) break;
        // STM-003 hard cap: once incarnation reaches MAX_TOTAL_REPAIRS,
        // emit a deterministic Error commit item instead of looping back
        // to Exec.
        if (er.incarnation >= QUASAR_MAX_TOTAL_REPAIRS) {
            CommitItem c;
            c.tx_index       = er.tx_index;
            c.status         = kExecStatusError;
            c.gas_used       = 0u;
            c.cumulative_gas = 0u;
            uint origin_lo = uint(er.rw[0].key_lo & 0xFFFFFFFFu);
            uint origin_hi = uint(er.rw[0].key_hi & 0xFFFFFFFFu);
            thread uchar digest[32];
            receipt_hash(er.tx_index, origin_lo, origin_hi,
                         0u, 0u,
                         desc->round, desc->chain_id, digest);
            for (uint k = 0u; k < 32u; ++k) c.receipt_hash[k] = digest[k];
            if (!ring_try_push(commit_hdr, commit_items, c)) {
                (void)ring_try_push(repair_hdr, repair_items, er);
                break;
            }
            atomic_fetch_add_explicit(&result->repair_capped_count, 1u,
                                      memory_order_relaxed);
            ++processed;
            continue;
        }
        VerifiedTx v;
        v.tx_index    = er.tx_index;
        v.admission   = 0u;
        v.gas_limit   = er.gas_used;
        // Reconstruct origin from rw[0].key (carries exec_key + flags).
        v.origin_lo   = uint(er.rw[0].key_lo & 0xFFFFFFFFu);
        v.origin_hi   = uint(er.rw[0].key_hi & 0xFFFFFFFFu) | kNeedsExec;
        // Repair re-emits don't carry bytecode (no fiber persistence
        // across the Validate→Repair→Exec hop yet); v0.42 wires this when
        // the fiber slot becomes durable.
        v.blob_offset = 0u;
        v.blob_size   = 0u;
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
        v.tx_index    = p.tx_index;
        v.admission   = (p.status == 0u) ? 0u : 1u;
        v.gas_limit   = 21000u;
        v.origin_lo   = uint(p.key_lo & 0xFFFFFFFFu);
        v.origin_hi   = uint(p.key_hi & 0xFFFFFFFFu);
        v.blob_offset = 0u;
        v.blob_size   = 0u;
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
// of the receipts trie. v0.40: also walks dag_nodes[c.tx_index].children
// and decrements unresolved_parents; re-emits any child that reaches zero.
static inline uint drain_commit(
    device RingHeader* commit_hdr, device CommitItem* commit_items,
    device DagNode*    dag_nodes,  uint dag_node_capacity,
    device RingHeader* exec_hdr,   device VerifiedTx* exec_items,
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
        thread uchar buf[64];
        for (uint k = 0u; k < 32u; ++k) buf[k]      = result->receipts_root[k];
        for (uint k = 0u; k < 32u; ++k) buf[32u+k]  = c.receipt_hash[k];
        thread uchar next[32];
        keccak256_thread(buf, 64UL, next);
        for (uint k = 0u; k < 32u; ++k) result->receipts_root[k] = next[k];
        thread uchar erbuf[64];
        for (uint k = 0u; k < 32u; ++k) erbuf[k] = result->execution_root[k];
        for (uint k = 0u; k < 4u; ++k)  erbuf[32u + k]      = uchar((c.tx_index >> (k * 8u)) & 0xFFu);
        for (uint k = 0u; k < 4u; ++k)  erbuf[36u + k]      = uchar((c.status   >> (k * 8u)) & 0xFFu);
        for (uint k = 0u; k < 8u; ++k)  erbuf[40u + k]      = uchar((c.gas_used >> (k * 8u)) & 0xFFu);
        for (uint k = 0u; k < 20u; ++k) erbuf[48u + k]      = c.receipt_hash[k];
        thread uchar ernext[32];
        keccak256_thread(erbuf, 64UL, ernext);
        for (uint k = 0u; k < 32u; ++k) result->execution_root[k] = ernext[k];

        // v0.40: walk DAG children of just-committed tx and decrement.
        if (c.tx_index < dag_node_capacity) {
            device DagNode* P = &dag_nodes[c.tx_index];
            threadgroup_barrier(mem_flags::mem_device);
            uint child_n = atomic_load_explicit(&P->child_count, memory_order_relaxed);
            if (child_n > kMaxDagChildren) child_n = kMaxDagChildren;
            for (uint k = 0u; k < child_n; ++k) {
                uint child_tidx = P->children[k];
                if (child_tidx >= dag_node_capacity) continue;
                device DagNode* C = &dag_nodes[child_tidx];
                uint old = atomic_fetch_sub_explicit(&C->unresolved_parents, 1u, memory_order_relaxed);
                if (old == 1u) {
                    VerifiedTx out = dag_envelope_to_verified(C);
                    if (ring_try_push(exec_hdr, exec_items, out)) {
                        atomic_store_explicit(&C->state, kDagNodeEmitted, memory_order_relaxed);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_device);
            atomic_store_explicit(&P->state, kDagNodeCommitted, memory_order_relaxed);
        }
        ++processed;
    }
    return processed;
}

// drain_vote — host-supplied votes. Per-lane stake aggregation.
//
// v0.42 (CERT-001..CERT-023):
//   - CERT-022: subject must equal desc->certificate_subject — reject mismatch.
//   - CERT-023: validator_index must be < desc->validator_count
//     and < kValidatorBitmapBits.
//   - CERT-004: per-validator dedup bitmap (per lane). atomic_or test-and-set.
//   - CERT-007: uint64-split stake accumulator with carry from lo to hi
//     (matches the gas_used pattern; MSL has no native 64-bit atomics).
//   - CERT-020: threshold = (desc->total_stake * 2) / 3, NOT base_fee.
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

        // CERT-022 — subject must match descriptor's certificate_subject.
        uchar diff = 0u;
        for (uint k = 0u; k < 32u; ++k) {
            diff |= uchar(v.subject[k] ^ desc->certificate_subject[k]);
        }
        if (diff != 0u) {
            atomic_fetch_add_explicit(&result->subject_mismatch_count, 1u,
                                      memory_order_relaxed);
            ++processed;
            continue;
        }

        // CERT-023 — bound validator_index against descriptor + bitmap width.
        uint vi = v.validator_index;
        if (vi >= desc->validator_count || vi >= kValidatorBitmapBits) {
            ++processed;
            continue;
        }

        // CERT-005/018 — bound sig_kind to the 3 active lanes; default reject.
        uint lane = v.sig_kind;
        if (lane > 2u) { ++processed; continue; }

        // CERT-004 — per-validator dedup. atomic_fetch_or test-and-set on
        // a flat [3*WORDS] bitmap (linear lane*WORDS + word_idx).
        uint word_idx = lane * uint(QUASAR_VALIDATOR_BITMAP_WORDS) + (vi >> 5u);
        uint bit_pos  = vi & 31u;
        uint bit_mask = 1u << bit_pos;
        device atomic_uint* bm_base = &result->validator_voted_bitmap[0][0];
        uint prev_bits = atomic_fetch_or_explicit(bm_base + word_idx, bit_mask,
                                                  memory_order_relaxed);
        if ((prev_bits & bit_mask) != 0u) {
            atomic_fetch_add_explicit(&result->dedup_skipped_count, 1u,
                                      memory_order_relaxed);
            ++processed;
            continue;
        }

        // CERT-007 — uint64-split stake accumulator with carry.
        device atomic_uint* stake_lo = &result->quorum_stake_bls_lo;
        device atomic_uint* stake_hi = &result->quorum_stake_bls_hi;
        device atomic_uint* status_acc = &result->quorum_status_bls;
        if (lane == 1u) {
            stake_lo = &result->quorum_stake_rt_lo;
            stake_hi = &result->quorum_stake_rt_hi;
            status_acc = &result->quorum_status_rt;
        } else if (lane == 2u) {
            stake_lo = &result->quorum_stake_mldsa_lo;
            stake_hi = &result->quorum_stake_mldsa_hi;
            status_acc = &result->quorum_status_mldsa;
        }
        uint sw_lo = uint(v.stake_weight & 0xFFFFFFFFul);
        uint sw_hi = uint((v.stake_weight >> 32u) & 0xFFFFFFFFul);
        uint prev_lo = atomic_fetch_add_explicit(stake_lo, sw_lo,
                                                 memory_order_relaxed);
        uint new_lo = prev_lo + sw_lo;
        if (new_lo < prev_lo) {
            atomic_fetch_add_explicit(stake_hi, 1u, memory_order_relaxed);
        }
        if (sw_hi != 0u) {
            atomic_fetch_add_explicit(stake_hi, sw_hi, memory_order_relaxed);
        }
        ulong threshold_u = (desc->total_stake * 2ul) / 3ul;
        uint threshold = (threshold_u >= ulong(0xFFFFFFFFu)) ?
                         0xFFFFFFFFu : uint(threshold_u);
        if (prev_lo < threshold && new_lo >= threshold) {
            QuorumCert qc;
            qc.round         = uint(desc->round);
            qc.status        = 1u;
            qc.signers_count = 1u;
            qc.total_stake   = new_lo;
            qc.sig_kind      = lane;
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

// CERT-001 (v0.42): the master secret is no longer in source. Hosts populate
// `cert_master_secret` (buffer(5) of quasar_verify_votes_kernel) from the
// QUASAR_MASTER_SECRET environment variable (placeholder until KMS lands at
// v0.43). The kernel reads from that buffer; nothing about the master secret
// is hard-coded. (CERT-018: also gates lane membership.)

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
                                        device const uchar* master_secret,
                                        thread uchar* secret_out)
{
    thread uchar buf[16 + 8 + 4 + 32];
    quasar_pick_domain(sig_kind, buf);
    for (uint k = 0u; k < 8u; ++k) buf[16u + k] = uchar((chain_id >> (k * 8u)) & 0xFFu);
    for (uint k = 0u; k < 4u; ++k) buf[24u + k] = uchar((validator_index >> (k * 8u)) & 0xFFu);
    for (uint k = 0u; k < 32u; ++k) buf[28u + k] = master_secret[k];
    keccak256_thread(buf, ulong(60), secret_out);
}

// CERT-006/021 (v0.42): MAC binds full uint64 round + uint64 stake_weight +
// validator_index. Build subject_with_vote = keccak(subject || stake_weight ||
// validator_index || round_full_uint64), then sign that.
static inline void quasar_expected_sig(thread const uchar* secret,
                                       thread const uchar* subject,
                                       ulong round,
                                       ulong stake_weight,
                                       uint validator_index,
                                       thread uchar* expected_out)
{
    // Step 1: subject_with_vote = keccak(subject || stake_weight_le8
    //                                    || validator_le4 || round_le8).
    thread uchar buf1[32 + 8 + 4 + 8];
    for (uint k = 0u; k < 32u; ++k) buf1[k] = subject[k];
    for (uint k = 0u; k < 8u; ++k) buf1[32u + k] = uchar((stake_weight >> (k * 8u)) & 0xFFul);
    for (uint k = 0u; k < 4u; ++k) buf1[40u + k] = uchar((validator_index >> (k * 8u)) & 0xFFu);
    for (uint k = 0u; k < 8u; ++k) buf1[44u + k] = uchar((round >> (k * 8u)) & 0xFFul);
    thread uchar subject_with_vote[32];
    keccak256_thread(buf1, ulong(52), subject_with_vote);

    // Step 2: expected = keccak(secret || subject_with_vote).
    thread uchar buf2[32 + 32];
    for (uint k = 0u; k < 32u; ++k) buf2[k]      = secret[k];
    for (uint k = 0u; k < 32u; ++k) buf2[32u + k] = subject_with_vote[k];
    keccak256_thread(buf2, ulong(64), expected_out);
}

kernel void quasar_verify_votes_kernel(
    device QuasarRoundDescriptor* desc                [[buffer(0)]],
    device RingHeader*            hdrs                [[buffer(1)]],
    device uchar*                 items_arena         [[buffer(2)]],
    device uint*                  verified            [[buffer(3)]],
    constant uint&                capacity            [[buffer(4)]],
    device const uchar*           cert_master_secret  [[buffer(5)]],
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

    // CERT-018/005 — only 3 lanes valid.
    if (v.sig_kind > 2u) { verified[slot_idx] = 0u; return; }

    thread uchar subj[32];
    for (uint k = 0u; k < 32u; ++k) subj[k] = v.subject[k];

    thread uchar secret[32];
    quasar_derive_secret(v.sig_kind, desc->chain_id, v.validator_index,
                         cert_master_secret, secret);

    thread uchar expected[32];
    quasar_expected_sig(secret, subj, v.round, v.stake_weight,
                        v.validator_index, expected);

    uchar diff = 0u;
    for (uint k = 0u; k < 32u; ++k) diff |= uchar(v.signature[k] ^ expected[k]);

    verified[slot_idx] = (diff == 0u) ? 1u : 0u;
}

// ============================================================================
// Main scheduler kernel
// ============================================================================

kernel void quasar_wave_kernel(
    device QuasarRoundDescriptor* desc                  [[buffer(0)]],
    device QuasarRoundResult*     result                [[buffer(1)]],
    device RingHeader*            hdrs                  [[buffer(2)]],
    device uchar*                 items_arena           [[buffer(3)]],
    device atomic_uint*           tx_index_seq          [[buffer(4)]],
    device MvccSlot*              mvcc_table            [[buffer(5)]],
    device DagNode*               dag_nodes             [[buffer(6)]],
    device FiberSlot*             fibers                [[buffer(7)]],
    constant uint&                mvcc_slot_count       [[buffer(8)]],
    device const uint*            vote_verified         [[buffer(9)]],
    constant uint&                vote_verified_capacity [[buffer(10)]],
    device DagWriterSlot*         dag_writer_table      [[buffer(11)]],
    constant uint&                dag_writer_slot_count [[buffer(12)]],
    device const PredictedKey*    predicted_keys        [[buffer(13)]],
    constant uint&                predicted_capacity    [[buffer(14)]],
    constant uint&                dag_node_capacity     [[buffer(15)]],
    // v0.41 — EVM bytecode interpreter inputs.
    device const uchar*           code_arena            [[buffer(16)]],
    constant uint&                code_arena_size       [[buffer(17)]],
    constant uint&                fiber_capacity        [[buffer(18)]],
    uint   tid                                          [[thread_index_in_threadgroup]],
    uint   gid                                          [[threadgroup_position_in_grid]])
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
        (void)drain_dagready(dagready_hdr, dagready_items, exec_hdr, exec_items,
                             dag_nodes, dag_node_capacity,
                             dag_writer_table, dag_writer_slot_count,
                             predicted_keys, predicted_capacity, budget);
    } else if (gid == 4u) {
        // v0.41 — real EVM bytecode interpreter
        (void)drain_exec(exec_hdr, exec_items, validate_hdr, validate_items,
                         statereq_hdr, statereq_items,
                         mvcc_table, mvcc_slot_count,
                         fibers, fiber_capacity,
                         code_arena, code_arena_size,
                         desc, result, budget);
    } else if (gid == 5u) {
        (void)drain_validate(validate_hdr, validate_items, commit_hdr, commit_items,
                             repair_hdr, repair_items, mvcc_table, mvcc_slot_count,
                             desc, result, budget);
    } else if (gid == 6u) {
        (void)drain_repair(repair_hdr, repair_items, exec_hdr, exec_items,
                           commit_hdr, commit_items, desc, result, budget);
    } else if (gid == 7u) {
        (void)drain_commit(commit_hdr, commit_items,
                           dag_nodes, dag_node_capacity,
                           exec_hdr, exec_items,
                           result, budget);
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
