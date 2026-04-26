// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file quasar_wave.cu
/// QuasarGPU — wave-tick scheduler kernel (CUDA mirror of quasar_wave.metal).
///
/// Persistent-CTA model: one CTA per ServiceId, 32 threads each, only
/// thread 0 does the drain (matching MSL's `tid != 0u → return`). Lanes
/// 1..31 are reserved for SIMT EVM ops in v0.42+. The host launches this
/// kernel once per wave tick — no persistent hot-spinning, same v0.29
/// starvation fix as the Metal path.
///
/// MSL → CUDA mapping:
///   threadgroup_barrier(mem_device) → __threadfence()
///   atomic_*_explicit(memory_order_relaxed) → atomicAdd / atomicCAS / atomicOr
///   `device T*` → raw `T*` (already in __device__ address space at the kernel entry)
///   `constant uint&` → kernel arg (uint32_t)
///   `thread T` → register / local var (default in CUDA)
///
/// Layout structs MUST match quasar_gpu_layout.hpp byte-for-byte, same as
/// Metal — alignas(16) is honoured by nvcc identically to the MSL alignas
/// directive.

#include <cstdint>
#include <cuda_runtime.h>

namespace quasar::gpu::cuda {

// ============================================================================
// Layout constants (must match quasar_gpu_layout.hpp)
// ============================================================================

__device__ static constexpr uint32_t kNumServices       = 17u;  // v0.44
__device__ static constexpr uint32_t kMaxRWSetPerTx     = 8u;
__device__ static constexpr uint32_t kMaxDagParents     = 4u;
__device__ static constexpr uint32_t kMaxDagChildren    = 16u;
__device__ static constexpr uint32_t kFiberStackDepth   = 64u;
__device__ static constexpr uint32_t kFiberStackLimbs   = 4u;     ///< 256-bit
__device__ static constexpr uint32_t kFiberMemoryBytes  = 1024u;
__device__ static constexpr uint32_t kFiberInstrBudget  = 100000u;

__device__ static constexpr uint32_t kFiberReady        = 0u;
__device__ static constexpr uint32_t kFiberRunning      = 1u;
__device__ static constexpr uint32_t kFiberWaitingState = 2u;
__device__ static constexpr uint32_t kFiberCommittable  = 3u;
__device__ static constexpr uint32_t kFiberReverted     = 4u;

__device__ static constexpr uint64_t kGasDefault     = 3ull;
__device__ static constexpr uint64_t kGasJumpdest    = 1ull;
__device__ static constexpr uint64_t kGasSloadWarm   = 100ull;
__device__ static constexpr uint64_t kGasSloadCold   = 2100ull;
__device__ static constexpr uint64_t kGasSstore      = 5000ull;
__device__ static constexpr uint64_t kGasKeccakBase  = 30ull;
__device__ static constexpr uint64_t kGasKeccakWord  = 6ull;
__device__ static constexpr uint64_t kGasExpByte     = 50ull;

__device__ static constexpr uint32_t kNeedsState = 0x80000000u;
__device__ static constexpr uint32_t kNeedsExec  = 0x40000000u;
__device__ static constexpr uint32_t kFlagMask   = 0xC0000000u;

__device__ static constexpr uint32_t kMvccInvalidIdx = 0xFFFFFFFFu;

// ============================================================================
// Layout types — byte-for-byte mirrors of MSL structs.
// CUDA reuses the C++ alignas(16) from the shared header semantics.
// ============================================================================

struct __align__(16) RingHeader {
    uint32_t head;          // atomic via raw ops
    uint32_t tail;
    uint32_t capacity;
    uint32_t mask;
    uint64_t items_ofs;
    uint32_t item_size;
    uint32_t _pad0;
    uint32_t pushed;
    uint32_t consumed;
    uint32_t _pad1;
    uint32_t _pad2;
};

struct __align__(16) IngressTx {
    uint32_t blob_offset;
    uint32_t blob_size;
    uint64_t gas_limit;
    uint32_t nonce;
    uint32_t _pad0;
    uint32_t origin_lo;
    uint32_t origin_hi;
};

struct __align__(16) DecodedTx {
    uint32_t tx_index;
    uint32_t blob_offset;
    uint32_t blob_size;
    uint64_t gas_limit;
    uint32_t nonce;
    uint32_t origin_lo;
    uint32_t origin_hi;
    uint32_t status;
};

struct __align__(16) VerifiedTx {
    uint32_t tx_index;
    uint32_t admission;
    uint64_t gas_limit;
    uint32_t origin_lo;
    uint32_t origin_hi;
    uint32_t blob_offset;
    uint32_t blob_size;
};

struct RWSetEntry {
    uint64_t key_lo;
    uint64_t key_hi;
    uint32_t version_seen;
    uint32_t kind;
};

struct ExecResult {
    uint32_t tx_index;
    uint32_t incarnation;
    uint32_t status;
    uint32_t rw_count;
    uint64_t gas_used;
    uint64_t gas_refund;       ///< v0.45 — pre-cap refund accumulator
    uint32_t journal_count;    ///< v0.45 — # journal entries to apply on commit
    uint32_t _pad0;
    RWSetEntry rw[kMaxRWSetPerTx];
};

struct __align__(16) CommitItem {
    uint32_t tx_index;
    uint32_t status;
    uint64_t gas_used;
    uint64_t cumulative_gas;
    uint64_t gas_refund;       ///< v0.45 — capped refund (gas_used / 5)
    uint64_t _pad0;
    uint8_t  receipt_hash[32];
};

struct __align__(16) MvccSlot {
    uint64_t key_lo;
    uint64_t key_hi;
    uint32_t last_writer_tx;
    uint32_t last_writer_inc;
    uint32_t version;
    uint32_t present;          ///< v0.45 — 0 = never written, 1 = has value
    uint8_t  value[32];        ///< v0.45 — persisted storage value (BE)
};

struct __align__(16) DagNode {
    uint32_t tx_index;
    uint32_t parent_count;
    uint32_t unresolved_parents;
    uint32_t child_count;
    uint32_t parents[kMaxDagParents];
    uint32_t children[kMaxDagChildren];
    // v0.40 / v0.41: cached envelope so drain_commit can re-emit a freed
    // child straight to Exec without re-popping from DagReady.
    uint64_t pending_gas_limit;
    uint32_t pending_origin_lo;
    uint32_t pending_origin_hi;
    uint32_t pending_admission;
    uint32_t state;
    uint32_t pending_blob_offset;       ///< v0.41 — bytecode slice
    uint32_t pending_blob_size;
};

struct __align__(16) StateRequest {
    uint32_t tx_index;
    uint32_t key_type;
    uint32_t priority;
    uint32_t _pad0;
    uint64_t key_lo;
    uint64_t key_hi;
};

struct __align__(16) StatePage {
    uint32_t tx_index;
    uint32_t key_type;
    uint32_t status;
    uint32_t data_size;
    uint64_t key_lo;
    uint64_t key_hi;
    uint8_t  data[64];
};

struct __align__(16) VoteIngress {
    uint32_t validator_index;
    uint32_t round;
    uint32_t stake_weight;
    uint32_t sig_kind;
    uint8_t  subject[32];
    uint8_t  signature[96];
};

struct __align__(16) QuorumCert {
    uint32_t round;
    uint32_t status;
    uint32_t signers_count;
    uint32_t total_stake;
    uint32_t sig_kind;
    uint32_t _pad0;
    uint64_t _pad1;
    uint8_t  subject[32];
    uint8_t  agg_signature[96];
};

// v0.44: descriptor must match quasar_gpu_layout.hpp byte-for-byte. v0.42
// added epoch / P-Q-Z roots / cert subject; v0.44 adds X/A/B/M/F roots so
// the cert subject covers all 9 LP-134 chains.
struct __align__(16) QuasarRoundDescriptor {
    uint64_t chain_id;
    uint64_t round;
    uint64_t timestamp_ns;
    uint64_t deadline_ns;
    uint64_t gas_limit;
    uint64_t base_fee;
    uint32_t wave_tick_budget;
    uint32_t wave_tick_index;
    uint32_t closing_flag;
    uint32_t mode;
    uint8_t  parent_block_hash[32];
    uint8_t  parent_state_root[32];
    uint8_t  parent_execution_root[32];
    uint64_t epoch;
    uint64_t total_stake;
    uint32_t validator_count;
    uint32_t _pad0;
    uint8_t  pchain_validator_root[32];
    uint8_t  qchain_ceremony_root[32];
    uint8_t  zchain_vk_root[32];
    uint8_t  certificate_subject[32];
    uint8_t  xchain_execution_root[32];
    uint8_t  achain_state_root[32];
    uint8_t  bchain_state_root[32];
    uint8_t  mchain_state_root[32];
    uint8_t  fchain_state_root[32];
};

struct __align__(16) QuasarRoundResult {
    uint32_t status;
    uint32_t tx_count;
    uint32_t gas_used_lo;
    uint32_t gas_used_hi;
    uint32_t wave_tick_count;
    uint32_t conflict_count;
    uint32_t repair_count;
    uint32_t fibers_suspended;
    uint32_t fibers_resumed;
    uint32_t quorum_status_bls;
    uint32_t quorum_status_mldsa;
    uint32_t quorum_status_rt;
    uint32_t quorum_stake_bls;
    uint32_t quorum_stake_mldsa;
    uint32_t quorum_stake_rt;
    uint32_t mode;
    uint8_t  block_hash[32];
    uint8_t  state_root[32];
    uint8_t  receipts_root[32];
    uint8_t  execution_root[32];
    uint8_t  mode_root[32];
};

// FiberSlot mirrors quasar_wave.metal byte-for-byte. v0.41 stack widened
// to 256 bits (4 limbs × 64), and per-fiber EVM execution context
// (origin / gas_limit / msize / code slice) added.
struct FiberSlot {
    uint32_t tx_index;
    uint32_t pc;
    uint32_t sp;
    uint32_t status;
    uint64_t gas;
    uint32_t rw_count;
    uint32_t incarnation;
    uint32_t pending_key_lo_lo;
    uint32_t pending_key_lo_hi;
    uint32_t pending_key_hi_lo;
    uint32_t pending_key_hi_hi;
    uint32_t origin_lo;        ///< v0.41 — captured for ADDRESS/CALLER/ORIGIN
    uint32_t origin_hi;
    uint64_t gas_limit;        ///< v0.41 — entry gas, for gas_used computation
    uint32_t msize;            ///< v0.41 — memory high-water mark (bytes)
    uint32_t code_offset;      ///< v0.41 — bytecode slice in code arena
    uint32_t code_size;
    uint32_t _pad0;
    RWSetEntry rw[kMaxRWSetPerTx];
    uint64_t   stack[kFiberStackDepth * kFiberStackLimbs];   ///< v0.41 — 256-bit
    uint8_t    memory[kFiberMemoryBytes];
};

// ============================================================================
// Atomic + ring helpers — CUDA equivalents of the MSL ring_try_push/pop.
//
// MSL ring_try_pop uses atomic_compare_exchange_weak_explicit. The CUDA
// equivalent is atomicCAS, which returns the *old* value: success when
// (old == expected). The retry loop is the same shape as MSL.
//
// Cross-CTA visibility comes from __threadfence() before tail-store
// (push) and after head-CAS (pop). Same pattern as the Metal mem_device
// barrier.
// ============================================================================

template<typename T>
__device__ __forceinline__ bool ring_try_push(RingHeader* h, T* items, const T& v)
{
    uint32_t head = atomicAdd(&h->head, 0u);   // atomic load (idiomatic CUDA)
    uint32_t tail = atomicAdd(&h->tail, 0u);
    if (tail - head >= h->capacity) return false;
    items[tail & h->mask] = v;
    __threadfence();                            // publish item before tail
    atomicExch(&h->tail, tail + 1u);
    atomicAdd(&h->pushed, 1u);
    return true;
}

template<typename T>
__device__ __forceinline__ bool ring_try_pop(RingHeader* h, T* items, T& out)
{
    while (true) {
        uint32_t head = atomicAdd(&h->head, 0u);
        uint32_t tail = atomicAdd(&h->tail, 0u);
        if (head >= tail) return false;
        uint32_t observed = atomicCAS(&h->head, head, head + 1u);
        if (observed == head) {
            __threadfence();                    // see published item bytes
            out = items[head & h->mask];
            atomicAdd(&h->consumed, 1u);
            return true;
        }
        // CAS lost — retry. Mirrors MSL's atomic_compare_exchange_weak loop.
    }
}

// ============================================================================
// Inline keccak-f[1600] — same FIPS-202 24-round permutation as MSL.
// Constants and rotation table are identical to MSL kKeccakRC / kKeccakRot.
// ============================================================================

__device__ static const uint64_t kKeccakRC[24] = {
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

__device__ static const uint32_t kKeccakRot[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
};

__device__ __forceinline__ uint64_t rotl64(uint64_t x, uint32_t n)
{
    return (x << n) | (x >> (64u - n));
}

__device__ static void keccak_f1600(uint64_t* s)
{
    for (uint32_t round = 0u; round < 24u; ++round) {
        // theta
        uint64_t c[5];
        for (uint32_t x = 0u; x < 5u; ++x) {
            c[x] = s[x] ^ s[x+5u] ^ s[x+10u] ^ s[x+15u] ^ s[x+20u];
        }
        uint64_t d[5];
        for (uint32_t x = 0u; x < 5u; ++x) {
            d[x] = c[(x + 4u) % 5u] ^ rotl64(c[(x + 1u) % 5u], 1u);
        }
        for (uint32_t y = 0u; y < 25u; y += 5u) {
            for (uint32_t x = 0u; x < 5u; ++x) {
                s[y + x] ^= d[x];
            }
        }
        // rho + pi
        uint64_t b[25];
        for (uint32_t y = 0u; y < 5u; ++y) {
            for (uint32_t x = 0u; x < 5u; ++x) {
                uint32_t i = x + 5u * y;
                uint32_t j = y + 5u * ((2u * x + 3u * y) % 5u);
                b[j] = rotl64(s[i], kKeccakRot[i]);
            }
        }
        // chi
        for (uint32_t y = 0u; y < 25u; y += 5u) {
            uint64_t t0 = b[y + 0u];
            uint64_t t1 = b[y + 1u];
            uint64_t t2 = b[y + 2u];
            uint64_t t3 = b[y + 3u];
            uint64_t t4 = b[y + 4u];
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

// keccak256: rate=136, capacity=512, output=32. Domain 0x01 ... 0x80
// (Ethereum keccak — NOT NIST SHA-3). Local-buffer variant only —
// drain_commit only feeds 64-byte buffers, drain_state path is unused on
// the device; the kernel never touches device-typed input here.
__device__ static void keccak256_local(const uint8_t* data, uint64_t len, uint8_t* out)
{
    uint64_t s[25];
    for (uint32_t i = 0u; i < 25u; ++i) s[i] = 0ull;

    constexpr uint32_t rate_bytes = 136u;
    uint64_t off = 0ull;
    while (len - off >= rate_bytes) {
        for (uint32_t i = 0u; i < rate_bytes; ++i) {
            uint32_t lane  = i / 8u;
            uint32_t shift = (i % 8u) * 8u;
            s[lane] ^= uint64_t(data[off + i]) << shift;
        }
        keccak_f1600(s);
        off += rate_bytes;
    }
    uint8_t block[rate_bytes];
    uint64_t rem = len - off;
    for (uint32_t i = 0u; i < rate_bytes; ++i) block[i] = 0u;
    for (uint64_t i = 0ull; i < rem; ++i) block[i] = data[off + i];
    block[uint32_t(rem)]       ^= 0x01u;
    block[rate_bytes - 1u]     ^= 0x80u;
    for (uint32_t i = 0u; i < rate_bytes; ++i) {
        uint32_t lane  = i / 8u;
        uint32_t shift = (i % 8u) * 8u;
        s[lane] ^= uint64_t(block[i]) << shift;
    }
    keccak_f1600(s);
    for (uint32_t i = 0u; i < 32u; ++i) {
        uint32_t lane  = i / 8u;
        uint32_t shift = (i % 8u) * 8u;
        out[i] = uint8_t((s[lane] >> shift) & 0xFFu);
    }
}

// ============================================================================
// MVCC arena — open-addressing hash table.
// ============================================================================

__device__ __forceinline__ uint32_t mvcc_index(uint64_t key_lo, uint64_t key_hi, uint32_t mask)
{
    uint64_t h = 0xcbf29ce484222325ull;
    h = (h ^ key_lo) * 0x100000001b3ull;
    h = (h ^ key_hi) * 0x100000001b3ull;
    return uint32_t(h) & mask;
}

__device__ static uint32_t mvcc_locate(MvccSlot* table, uint32_t slot_count,
                                       uint64_t key_lo, uint64_t key_hi)
{
    uint32_t mask = slot_count - 1u;
    uint32_t idx  = mvcc_index(key_lo, key_hi, mask);
    for (uint32_t probe = 0u; probe < slot_count; ++probe) {
        MvccSlot* s = &table[idx];
        if (s->key_lo == 0ull && s->key_hi == 0ull) {
            // Claim: write keys first, atomics race on (last_writer_tx, version).
            s->key_lo = key_lo;
            s->key_hi = key_hi;
            return idx;
        }
        if (s->key_lo == key_lo && s->key_hi == key_hi) return idx;
        idx = (idx + 1u) & mask;
    }
    return kMvccInvalidIdx;
}

// ============================================================================
// Receipt hash helper — same recipe as MSL.
// ============================================================================

__device__ static void receipt_hash(uint32_t tx_index, uint32_t origin_lo, uint32_t origin_hi,
                                    uint64_t gas_limit, uint64_t gas_used,
                                    uint64_t round, uint64_t chain_id,
                                    uint8_t* out)
{
    uint8_t leaf[40];
    for (uint32_t k = 0u; k < 4u; ++k) leaf[0u + k]  = uint8_t((tx_index  >> (k * 8u)) & 0xFFu);
    for (uint32_t k = 0u; k < 4u; ++k) leaf[4u + k]  = uint8_t((origin_lo >> (k * 8u)) & 0xFFu);
    for (uint32_t k = 0u; k < 4u; ++k) leaf[8u + k]  = uint8_t((origin_hi >> (k * 8u)) & 0xFFu);
    for (uint32_t k = 0u; k < 8u; ++k) leaf[12u + k] = uint8_t((gas_limit >> (k * 8u)) & 0xFFu);
    for (uint32_t k = 0u; k < 4u; ++k) leaf[20u + k] = uint8_t((gas_used  >> (k * 8u)) & 0xFFu);
    for (uint32_t k = 0u; k < 4u; ++k) leaf[24u + k] = uint8_t((round     >> (k * 8u)) & 0xFFu);
    for (uint32_t k = 0u; k < 8u; ++k) leaf[28u + k] = uint8_t((chain_id  >> (k * 8u)) & 0xFFu);
    leaf[36u] = 0u; leaf[37u] = 0u; leaf[38u] = 0u; leaf[39u] = 0u;
    keccak256_local(leaf, 40ull, out);
}

// ============================================================================
// Service drains — direct ports of MSL drain_* functions.
// ============================================================================

__device__ static uint32_t drain_ingress(
    RingHeader* ingress_hdr, IngressTx* ingress_items,
    RingHeader* decode_hdr,  DecodedTx*  decode_items,
    uint32_t*   tx_index_seq,
    uint32_t budget)
{
    uint32_t processed = 0u;
    for (uint32_t i = 0u; i < budget; ++i) {
        IngressTx in;
        if (!ring_try_pop(ingress_hdr, ingress_items, in)) break;
        DecodedTx out;
        out.tx_index    = atomicAdd(tx_index_seq, 1u);
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

__device__ static uint32_t drain_decode(
    RingHeader* decode_hdr,    DecodedTx*    decode_items,
    RingHeader* crypto_hdr,    VerifiedTx*   crypto_items,
    RingHeader* statereq_hdr,  StateRequest* statereq_items,
    QuasarRoundResult* result,
    uint32_t budget)
{
    uint32_t processed = 0u;
    for (uint32_t i = 0u; i < budget; ++i) {
        DecodedTx d;
        if (!ring_try_pop(decode_hdr, decode_items, d)) break;
        if ((d.origin_hi & kNeedsState) != 0u) {
            StateRequest sr;
            sr.tx_index = d.tx_index;
            sr.key_type = 0u;
            sr.priority = 0u;
            sr._pad0    = 0u;
            sr.key_lo   = uint64_t(d.origin_lo);
            sr.key_hi   = uint64_t(d.origin_hi & ~kFlagMask);
            if (!ring_try_push(statereq_hdr, statereq_items, sr)) {
                (void)ring_try_push(decode_hdr, decode_items, d);
                break;
            }
            atomicAdd(&result->fibers_suspended, 1u);
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

__device__ static uint32_t drain_crypto(
    RingHeader* crypto_hdr,   VerifiedTx* crypto_items,
    RingHeader* commit_hdr,   CommitItem* commit_items,
    RingHeader* dagready_hdr, VerifiedTx* dagready_items,
    RingHeader* exec_hdr,     VerifiedTx* exec_items,
    QuasarRoundDescriptor* desc,
    uint32_t budget)
{
    uint32_t processed = 0u;
    for (uint32_t i = 0u; i < budget; ++i) {
        VerifiedTx v;
        if (!ring_try_pop(crypto_hdr, crypto_items, v)) break;
        if (v.admission != 0u) { ++processed; continue; }

        if ((v.origin_hi & kNeedsExec) != 0u) {
            RingHeader* next_hdr   = (desc->mode == 1u) ? dagready_hdr : exec_hdr;
            VerifiedTx* next_items = (desc->mode == 1u) ? dagready_items : exec_items;
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
        uint8_t digest[32];
        receipt_hash(v.tx_index, v.origin_lo, v.origin_hi,
                     v.gas_limit, c.gas_used,
                     desc->round, desc->chain_id, digest);
        for (uint32_t k = 0u; k < 32u; ++k) c.receipt_hash[k] = digest[k];
        if (!ring_try_push(commit_hdr, commit_items, c)) {
            (void)ring_try_push(crypto_hdr, crypto_items, v);
            break;
        }
        ++processed;
    }
    return processed;
}

__device__ static uint32_t drain_dagready(
    RingHeader* dagready_hdr, VerifiedTx* dagready_items,
    RingHeader* exec_hdr,     VerifiedTx* exec_items,
    uint32_t budget)
{
    uint32_t processed = 0u;
    for (uint32_t i = 0u; i < budget; ++i) {
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

__device__ static bool mvcc_check_consistent(MvccSlot* table, uint32_t slot_count,
                                             const ExecResult& er)
{
    for (uint32_t i = 0u; i < er.rw_count; ++i) {
        const RWSetEntry& e = er.rw[i];
        uint32_t idx = mvcc_locate(table, slot_count, e.key_lo, e.key_hi);
        if (idx == kMvccInvalidIdx) return false;
        uint32_t cur = atomicAdd(&table[idx].version, 0u);
        if (e.kind == 0u && cur != e.version_seen) return false;
    }
    return true;
}

__device__ static void mvcc_apply_writes(MvccSlot* table, uint32_t slot_count,
                                         const ExecResult& er)
{
    for (uint32_t i = 0u; i < er.rw_count; ++i) {
        const RWSetEntry& e = er.rw[i];
        if (e.kind != 1u) continue;
        uint32_t idx = mvcc_locate(table, slot_count, e.key_lo, e.key_hi);
        if (idx == kMvccInvalidIdx) continue;
        atomicExch(&table[idx].last_writer_tx,  er.tx_index);
        atomicExch(&table[idx].last_writer_inc, er.incarnation);
        atomicAdd(&table[idx].version, 1u);
    }
}

__device__ static uint32_t drain_exec(
    RingHeader* exec_hdr,     VerifiedTx* exec_items,
    RingHeader* validate_hdr, ExecResult* validate_items,
    MvccSlot*   mvcc_table,   uint32_t mvcc_slot_count,
    uint32_t budget)
{
    uint32_t processed = 0u;
    for (uint32_t i = 0u; i < budget; ++i) {
        VerifiedTx v;
        if (!ring_try_pop(exec_hdr, exec_items, v)) break;

        uint64_t key_lo = uint64_t(v.origin_lo);
        uint64_t key_hi = uint64_t(v.origin_hi & ~kFlagMask);
        if (key_lo == 0ull && key_hi == 0ull) key_lo = 1ull;

        ExecResult er;
        er.tx_index    = v.tx_index;
        er.incarnation = 0u;
        er.status      = 1u;
        er.gas_used    = 21000u;
        er.rw_count    = 2u;

        uint32_t slot_idx = mvcc_locate(mvcc_table, mvcc_slot_count, key_lo, key_hi);
        uint32_t observed_version = 0u;
        if (slot_idx != kMvccInvalidIdx) {
            observed_version = atomicAdd(&mvcc_table[slot_idx].version, 0u);
        }
        er.rw[0].key_lo       = key_lo;
        er.rw[0].key_hi       = key_hi;
        er.rw[0].version_seen = observed_version;
        er.rw[0].kind         = 0u;
        er.rw[1].key_lo       = key_lo;
        er.rw[1].key_hi       = key_hi;
        er.rw[1].version_seen = observed_version;
        er.rw[1].kind         = 1u;
        for (uint32_t k = 2u; k < kMaxRWSetPerTx; ++k) {
            er.rw[k].key_lo = 0ull; er.rw[k].key_hi = 0ull;
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

__device__ static uint32_t drain_validate(
    RingHeader* validate_hdr, ExecResult* validate_items,
    RingHeader* commit_hdr,   CommitItem* commit_items,
    RingHeader* repair_hdr,   ExecResult* repair_items,
    MvccSlot*   mvcc_table,   uint32_t mvcc_slot_count,
    QuasarRoundDescriptor* desc,
    QuasarRoundResult* result,
    uint32_t budget)
{
    uint32_t processed = 0u;
    for (uint32_t i = 0u; i < budget; ++i) {
        ExecResult er;
        if (!ring_try_pop(validate_hdr, validate_items, er)) break;

        bool ok = mvcc_check_consistent(mvcc_table, mvcc_slot_count, er);
        if (!ok) {
            atomicAdd(&result->conflict_count, 1u);
            er.incarnation += 1u;
            if (!ring_try_push(repair_hdr, repair_items, er)) {
                (void)ring_try_push(validate_hdr, validate_items, er);
                break;
            }
            ++processed;
            continue;
        }

        mvcc_apply_writes(mvcc_table, mvcc_slot_count, er);
        CommitItem c;
        c.tx_index       = er.tx_index;
        c.status         = er.status;
        c.gas_used       = er.gas_used;
        c.cumulative_gas = 0u;
        uint32_t origin_lo = uint32_t(er.rw[0].key_lo & 0xFFFFFFFFu);
        uint32_t origin_hi = uint32_t(er.rw[0].key_hi & 0xFFFFFFFFu);
        uint8_t digest[32];
        receipt_hash(er.tx_index, origin_lo, origin_hi,
                     er.gas_used, er.gas_used,
                     desc->round, desc->chain_id, digest);
        for (uint32_t k = 0u; k < 32u; ++k) c.receipt_hash[k] = digest[k];

        if (!ring_try_push(commit_hdr, commit_items, c)) {
            (void)ring_try_push(validate_hdr, validate_items, er);
            break;
        }
        ++processed;
    }
    return processed;
}

__device__ static uint32_t drain_repair(
    RingHeader* repair_hdr, ExecResult* repair_items,
    RingHeader* exec_hdr,   VerifiedTx* exec_items,
    QuasarRoundResult* result,
    uint32_t budget)
{
    uint32_t processed = 0u;
    for (uint32_t i = 0u; i < budget; ++i) {
        ExecResult er;
        if (!ring_try_pop(repair_hdr, repair_items, er)) break;
        VerifiedTx v;
        v.tx_index    = er.tx_index;
        v.admission   = 0u;
        v.gas_limit   = er.gas_used;
        v.origin_lo   = uint32_t(er.rw[0].key_lo & 0xFFFFFFFFu);
        v.origin_hi   = uint32_t(er.rw[0].key_hi & 0xFFFFFFFFu) | kNeedsExec;
        v.blob_offset = 0u;
        v.blob_size   = 0u;
        if (!ring_try_push(exec_hdr, exec_items, v)) {
            (void)ring_try_push(repair_hdr, repair_items, er);
            break;
        }
        atomicAdd(&result->repair_count, 1u);
        ++processed;
    }
    return processed;
}

__device__ static uint32_t drain_state_resp(
    RingHeader* resp_hdr,   StatePage*  resp_items,
    RingHeader* crypto_hdr, VerifiedTx* crypto_items,
    QuasarRoundResult* result,
    uint32_t budget)
{
    uint32_t processed = 0u;
    for (uint32_t i = 0u; i < budget; ++i) {
        StatePage p;
        if (!ring_try_pop(resp_hdr, resp_items, p)) break;
        VerifiedTx v;
        v.tx_index    = p.tx_index;
        v.admission   = (p.status == 0u) ? 0u : 1u;
        v.gas_limit   = 21000u;
        v.origin_lo   = uint32_t(p.key_lo & 0xFFFFFFFFu);
        v.origin_hi   = uint32_t(p.key_hi & 0xFFFFFFFFu);
        v.blob_offset = 0u;
        v.blob_size   = 0u;
        if (!ring_try_push(crypto_hdr, crypto_items, v)) {
            (void)ring_try_push(resp_hdr, resp_items, p);
            break;
        }
        atomicAdd(&result->fibers_resumed, 1u);
        ++processed;
    }
    return processed;
}

__device__ static uint32_t drain_commit(
    RingHeader* commit_hdr, CommitItem* commit_items,
    QuasarRoundResult* result,
    uint32_t budget)
{
    uint32_t processed = 0u;
    for (uint32_t i = 0u; i < budget; ++i) {
        CommitItem c;
        if (!ring_try_pop(commit_hdr, commit_items, c)) break;
        atomicAdd(&result->tx_count, 1u);
        const uint32_t gas_lo = uint32_t(c.gas_used & 0xFFFFFFFFu);
        const uint32_t gas_hi = uint32_t((c.gas_used >> 32) & 0xFFFFFFFFu);
        const uint32_t prev = atomicAdd(&result->gas_used_lo, gas_lo);
        if (gas_hi != 0u) {
            atomicAdd(&result->gas_used_hi, gas_hi);
        }
        if (prev + gas_lo < prev) {
            atomicAdd(&result->gas_used_hi, 1u);
        }
        // receipts_root chain: H(running || receipt_hash).
        uint8_t buf[64];
        for (uint32_t k = 0u; k < 32u; ++k) buf[k]      = result->receipts_root[k];
        for (uint32_t k = 0u; k < 32u; ++k) buf[32u+k]  = c.receipt_hash[k];
        uint8_t next[32];
        keccak256_local(buf, 64ull, next);
        for (uint32_t k = 0u; k < 32u; ++k) result->receipts_root[k] = next[k];
        // execution_root chain.
        uint8_t erbuf[64];
        for (uint32_t k = 0u; k < 32u; ++k) erbuf[k] = result->execution_root[k];
        for (uint32_t k = 0u; k < 4u; ++k)  erbuf[32u + k] = uint8_t((c.tx_index >> (k * 8u)) & 0xFFu);
        for (uint32_t k = 0u; k < 4u; ++k)  erbuf[36u + k] = uint8_t((c.status   >> (k * 8u)) & 0xFFu);
        for (uint32_t k = 0u; k < 8u; ++k)  erbuf[40u + k] = uint8_t((c.gas_used >> (k * 8u)) & 0xFFu);
        for (uint32_t k = 0u; k < 20u; ++k) erbuf[48u + k] = c.receipt_hash[k];
        uint8_t ernext[32];
        keccak256_local(erbuf, 64ull, ernext);
        for (uint32_t k = 0u; k < 32u; ++k) result->execution_root[k] = ernext[k];
        ++processed;
    }
    return processed;
}

__device__ __forceinline__ bool verify_signature_stub(const uint8_t* subject, const uint8_t* sig)
{
    for (uint32_t i = 0u; i < 32u; ++i) {
        if (sig[i] != subject[i]) return false;
    }
    return true;
}

__device__ static uint32_t drain_vote(
    RingHeader* vote_hdr,    VoteIngress* vote_items,
    RingHeader* qc_hdr,      QuorumCert*  qc_items,
    QuasarRoundResult* result,
    QuasarRoundDescriptor* desc,
    uint32_t budget)
{
    uint32_t processed = 0u;
    for (uint32_t i = 0u; i < budget; ++i) {
        VoteIngress v;
        if (!ring_try_pop(vote_hdr, vote_items, v)) break;
        uint8_t subj[32];
        for (uint32_t k = 0u; k < 32u; ++k) subj[k] = v.subject[k];
        uint8_t sig[96];
        for (uint32_t k = 0u; k < 96u; ++k) sig[k] = v.signature[k];
        if (!verify_signature_stub(subj, sig)) { ++processed; continue; }
        uint32_t* stake_acc  = nullptr;
        uint32_t* status_acc = nullptr;
        if (v.sig_kind == 0u)      { stake_acc = &result->quorum_stake_bls;    status_acc = &result->quorum_status_bls; }
        else if (v.sig_kind == 1u) { stake_acc = &result->quorum_stake_rt;     status_acc = &result->quorum_status_rt; }
        else                       { stake_acc = &result->quorum_stake_mldsa;  status_acc = &result->quorum_status_mldsa; }
        uint32_t prev_stake = atomicAdd(stake_acc, v.stake_weight);
        uint32_t new_stake  = prev_stake + v.stake_weight;
        uint32_t threshold  = uint32_t((desc->base_fee * 2ull) / 3ull);
        if (prev_stake < threshold && new_stake >= threshold) {
            QuorumCert qc;
            qc.round         = uint32_t(desc->round);
            qc.status        = 1u;
            qc.signers_count = 1u;
            qc.total_stake   = new_stake;
            qc.sig_kind      = v.sig_kind;
            qc._pad0         = 0u;
            qc._pad1         = 0ull;
            for (uint32_t k = 0u; k < 32u; ++k) qc.subject[k]       = v.subject[k];
            for (uint32_t k = 0u; k < 96u; ++k) qc.agg_signature[k] = v.signature[k];
            (void)ring_try_push(qc_hdr, qc_items, qc);
            atomicExch(status_acc, 1u);
        }
        ++processed;
    }
    return processed;
}

// ============================================================================
// Main scheduler kernel — persistent CTAs.
//
// One CTA per service (gridDim.x = kNumServices = 12), 32 threads each.
// Only thread 0 of each block does the drain — lanes 1..31 reserved for
// SIMT EVM ops in v0.42+. Same shape as the Metal dispatch.
// ============================================================================

__global__ void quasar_wave_kernel(
    QuasarRoundDescriptor* desc,
    QuasarRoundResult*     result,
    RingHeader*            hdrs,
    uint8_t*               items_arena,
    uint32_t*              tx_index_seq,
    MvccSlot*              mvcc_table,
    DagNode*               dag_nodes,
    FiberSlot*             fibers,
    uint32_t               mvcc_slot_count)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t gid = blockIdx.x;

    if (tid != 0u) return;
    if (gid >= kNumServices) return;

    if (gid == 0u) {
        atomicAdd(&result->wave_tick_count, 1u);
    }

    const uint32_t budget = max(uint32_t(64), desc->wave_tick_budget);

    RingHeader* ingress_hdr   = hdrs + 0;
    RingHeader* decode_hdr    = hdrs + 1;
    RingHeader* crypto_hdr    = hdrs + 2;
    RingHeader* dagready_hdr  = hdrs + 3;
    RingHeader* exec_hdr      = hdrs + 4;
    RingHeader* validate_hdr  = hdrs + 5;
    RingHeader* repair_hdr    = hdrs + 6;
    RingHeader* commit_hdr    = hdrs + 7;
    RingHeader* statereq_hdr  = hdrs + 8;
    RingHeader* stateresp_hdr = hdrs + 9;
    RingHeader* vote_hdr      = hdrs + 10;
    RingHeader* qc_hdr        = hdrs + 11;

    IngressTx*    ingress_items   = (IngressTx*)   (items_arena + ingress_hdr->items_ofs);
    DecodedTx*    decode_items    = (DecodedTx*)   (items_arena + decode_hdr->items_ofs);
    VerifiedTx*   crypto_items    = (VerifiedTx*)  (items_arena + crypto_hdr->items_ofs);
    VerifiedTx*   dagready_items  = (VerifiedTx*)  (items_arena + dagready_hdr->items_ofs);
    VerifiedTx*   exec_items      = (VerifiedTx*)  (items_arena + exec_hdr->items_ofs);
    ExecResult*   validate_items  = (ExecResult*)  (items_arena + validate_hdr->items_ofs);
    ExecResult*   repair_items    = (ExecResult*)  (items_arena + repair_hdr->items_ofs);
    CommitItem*   commit_items    = (CommitItem*)  (items_arena + commit_hdr->items_ofs);
    StateRequest* statereq_items  = (StateRequest*)(items_arena + statereq_hdr->items_ofs);
    StatePage*    stateresp_items = (StatePage*)   (items_arena + stateresp_hdr->items_ofs);
    VoteIngress*  vote_items      = (VoteIngress*) (items_arena + vote_hdr->items_ofs);
    QuorumCert*   qc_items        = (QuorumCert*)  (items_arena + qc_hdr->items_ofs);

    (void)dag_nodes; (void)fibers;  // v0.40 / v0.39 hooks.

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
        (void)drain_vote(vote_hdr, vote_items, qc_hdr, qc_items, result, desc, budget);
    }
    // gid 8 (StateRequest) and gid 11 (QuorumOut) are host-poll outputs; no drain.

    // Round finalization — same invariant: every ingress tx reached commit.
    if (gid == 0u && desc->closing_flag != 0u) {
        const uint32_t ingress_pushed  = atomicAdd(&hdrs[0].pushed,   0u);
        const uint32_t commit_consumed = atomicAdd(&hdrs[7].consumed, 0u);
        if (ingress_pushed == commit_consumed) {
            uint8_t header[8 + 4 + 32 * 4];
            uint32_t o = 0u;
            for (uint32_t k = 0u; k < 8u; ++k) { header[o++] = uint8_t((desc->round >> (k * 8u)) & 0xFFu); }
            for (uint32_t k = 0u; k < 4u; ++k) { header[o++] = uint8_t((desc->mode  >> (k * 8u)) & 0xFFu); }
            for (uint32_t k = 0u; k < 32u; ++k) header[o++] = result->receipts_root[k];
            for (uint32_t k = 0u; k < 32u; ++k) header[o++] = result->execution_root[k];
            for (uint32_t k = 0u; k < 32u; ++k) header[o++] = result->state_root[k];
            for (uint32_t k = 0u; k < 32u; ++k) header[o++] = result->mode_root[k];
            uint8_t bh[32];
            keccak256_local(header, uint64_t(o), bh);
            for (uint32_t k = 0u; k < 32u; ++k) result->block_hash[k] = bh[k];
            for (uint32_t k = 0u; k < 32u; ++k) result->mode_root[k]  = bh[k];
            atomicExch(&result->status, 1u);
        }
    }
}

}  // namespace quasar::gpu::cuda

// ============================================================================
// Host-callable C launcher — used by quasar_gpu_engine_cuda.cpp.
//
// The .cpp host driver can't use the <<<>>> launch syntax because it isn't
// compiled by nvcc. This C-linkage wrapper packages the launch and lives
// in the same translation unit as the kernel.
// ============================================================================

extern "C" cudaError_t quasar_wave_launch(
    void*        d_desc,
    void*        d_result,
    void*        d_hdrs,
    void*        d_items_arena,
    void*        d_tx_index_seq,
    void*        d_mvcc_table,
    void*        d_dag_nodes,
    void*        d_fibers,
    uint32_t     mvcc_slot_count,
    cudaStream_t stream)
{
    using namespace quasar::gpu::cuda;
    // Service count is fixed at 12 — keep the constant local to the host
    // launcher so we don't reach into `__device__ constexpr` from host.
    constexpr uint32_t num_services = 12u;
    dim3 grid(num_services, 1, 1);
    dim3 block(32, 1, 1);
    quasar_wave_kernel<<<grid, block, 0, stream>>>(
        static_cast<QuasarRoundDescriptor*>(d_desc),
        static_cast<QuasarRoundResult*>(d_result),
        static_cast<RingHeader*>(d_hdrs),
        static_cast<uint8_t*>(d_items_arena),
        static_cast<uint32_t*>(d_tx_index_seq),
        static_cast<MvccSlot*>(d_mvcc_table),
        static_cast<DagNode*>(d_dag_nodes),
        static_cast<FiberSlot*>(d_fibers),
        mvcc_slot_count);
    return cudaGetLastError();
}
