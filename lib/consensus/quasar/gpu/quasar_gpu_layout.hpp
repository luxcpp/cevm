// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file quasar_gpu_layout.hpp
/// Shared host/GPU memory layouts for the QuasarGPU execution adapter.
///
/// **Scope** — this module is **CEVM's GPU execution adapter for
/// Quasar-certified rounds**. It does NOT replace luxfi/consensus.
///
///   lux/consensus   orders, votes, certifies, finalizes
///   cevm            executes, validates, roots, receipts
///   quasar/gpu/     GPU-side execution adapter for Quasar rounds
///
/// One statement: consensus decides what must be executed; cevm proves
/// what execution produced; Quasar certifies the resulting commitment.
///
/// QuasarGPU runs Block-STM, EVM fibers, batched crypto, and root
/// construction to produce the artifacts the consensus engine needs:
/// block_hash, state_root, receipts_root, execution_root. It optionally
/// participates in Quasar quorum aggregation when the host posts votes
/// to the Vote ingress ring. Both Nova (linear, protocol/nova) and
/// Nebula (DAG, protocol/nebula) modes share this substrate; the round
/// descriptor's `mode` field selects between them.
///
/// All offsets here MUST match quasar_wave.metal byte-for-byte.

#pragma once

#include <cstdint>

namespace quasar::gpu {

// =============================================================================
// Service identifiers (work-queue addresses)
// =============================================================================
//
// Each ServiceId owns one device ring. Workgroups in the scheduler kernel
// pick a service via gid % kNumServices and drain up to epoch_budget_items
// items per epoch. Add new services at the end — never reorder or recycle.

enum class ServiceId : uint32_t {
    Ingress              = 0,   ///< raw tx blobs from host
    Decode               = 1,   ///< decoded txs awaiting sender recovery
    Crypto               = 2,   ///< sig-verified txs awaiting Block-STM scheduling
    DagReady             = 3,   ///< MVCC ready set — txs whose parents committed
    Exec                 = 4,   ///< txs being executed (EVM fibers running)
    Validate             = 5,   ///< executed txs awaiting Block-STM validation
    Repair               = 6,   ///< validation-failed txs awaiting re-execution
    Commit               = 7,   ///< validated txs ready to commit + hash
    StateRequest         = 8,   ///< GPU-emitted cold-state requests (out)
    StateResp            = 9,   ///< host-emitted cold-state responses (in)
    Vote                 = 10,  ///< raw consensus votes from host
    QuorumOut            = 11,  ///< GPU-emitted quorum certs (out)
    // v0.44 — wave-tick services that compose per-chain transition roots into
    // the round descriptor. Host posts a ChainTransitionItem per chain; the
    // GPU substrate copies it into the descriptor's <chain>_root field. Order
    // is canonical (P, X, A, B, M); C/Q/Z/F either already exist
    // (parent_block_hash, qchain_ceremony_root, zchain_vk_root) or carry their
    // own field (fchain_state_root). Subject keccak in compute_certificate_subject
    // covers all 9 in canonical P, C, X, Q, Z, A, B, M, F order.
    PlatformVMTransition = 12,  ///< PlatformVM v0.53.x → pchain_validator_root
    XVMTransition        = 13,  ///< XVM v0.55.x        → xchain_execution_root
    AIVMTransition       = 14,  ///< AIVM v0.58.x       → achain_state_root
    BridgeVMTransition   = 15,  ///< BridgeVM v0.59.x   → bchain_state_root
    MPCVMTransition      = 16,  ///< MPCVM v0.60.x      → mchain_state_root
    Count                = 17
};

inline constexpr uint32_t kNumServices = static_cast<uint32_t>(ServiceId::Count);

// =============================================================================
// Device-resident ring buffer
// =============================================================================
//
// Capacity is power-of-two; mask trick makes wrap a single AND. Producers
// bump tail with relaxed atomic; consumers CAS head. Cross-workgroup item
// visibility uses threadgroup_barrier(mem_flags::mem_device) on both
// sides — same pattern proven in v0.30 (V3 wave-dispatch).

struct alignas(16) RingHeader {
    uint32_t head;
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
static_assert(sizeof(RingHeader) == 48, "RingHeader layout drift");

// =============================================================================
// Per-service ring records
// =============================================================================

struct alignas(16) IngressTx {
    uint32_t blob_offset;
    uint32_t blob_size;
    uint64_t gas_limit;
    uint32_t nonce;
    uint32_t _pad0;
    uint32_t origin_lo;
    uint32_t origin_hi;
};
static_assert(sizeof(IngressTx) == 32, "IngressTx layout drift");

struct alignas(16) DecodedTx {
    uint32_t tx_index;
    uint32_t blob_offset;
    uint32_t blob_size;
    uint64_t gas_limit;
    uint32_t nonce;
    uint32_t origin_lo;
    uint32_t origin_hi;
    uint32_t status;
};
static_assert(sizeof(DecodedTx) == 48, "DecodedTx layout drift");

// v0.41 — VerifiedTx now carries the bytecode slice (blob_offset/blob_size)
// so drain_exec can index into the per-round code arena. Replaces the
// previous _pad0; the 32-byte envelope is unchanged.
struct alignas(16) VerifiedTx {
    uint32_t tx_index;
    uint32_t admission;
    uint64_t gas_limit;
    uint32_t origin_lo;
    uint32_t origin_hi;
    uint32_t blob_offset;
    uint32_t blob_size;
};
static_assert(sizeof(VerifiedTx) == 32, "VerifiedTx layout drift");

// Block-STM read/write set per tx. Natural 8-byte alignment keeps the
// struct compact (24 B) so ExecResult arrays pack tightly. v0.36 fiber
// VM streams entries into a per-tx journal arena.
struct RWSetEntry {
    uint64_t key_lo;
    uint64_t key_hi;
    uint32_t version_seen;   ///< version observed at read time (validate later)
    uint32_t kind;           ///< 0=read, 1=write
};
static_assert(sizeof(RWSetEntry) == 24, "RWSetEntry layout drift");

inline constexpr uint32_t kMaxRWSetPerTx = 8u;

struct ExecResult {
    uint32_t tx_index;
    uint32_t incarnation;    ///< Block-STM re-execution counter
    uint32_t status;         ///< 1=Return,2=Revert,3=OOG,4=Error,5=Suspend
    uint32_t rw_count;
    uint64_t gas_used;
    RWSetEntry rw[kMaxRWSetPerTx];
};
static_assert(sizeof(ExecResult) == 24 + 24u * kMaxRWSetPerTx, "ExecResult layout drift");

struct alignas(16) CommitItem {
    uint32_t tx_index;
    uint32_t status;
    uint64_t gas_used;
    uint64_t cumulative_gas;
    uint8_t  receipt_hash[32];   ///< per-tx keccak digest, accumulated into receipts_root
};
static_assert(sizeof(CommitItem) == 64, "CommitItem layout drift");

// =============================================================================
// MVCC arena (Block-STM)
// =============================================================================
//
// Open-addressing hash table keyed by (key_lo, key_hi). Each slot tracks
// the last writer's tx_index and incarnation, plus a monotonic version
// counter. Validate compares observed versions against current versions
// to detect read-after-write conflicts.

struct alignas(16) MvccSlot {
    uint64_t key_lo;            ///< 0,0 means empty (gated by claim_state)
    uint64_t key_hi;
    uint32_t last_writer_tx;    ///< 0xFFFFFFFF = no writer
    uint32_t last_writer_inc;   ///< incarnation of last writer
    uint32_t version;           ///< monotonic; bumped on every write
    uint32_t claim_state;       ///< STM-002: 0=free, 1=claiming, 2=ready
};
static_assert(sizeof(MvccSlot) == 32, "MvccSlot layout drift");

inline constexpr uint32_t kDefaultMvccSlots = 8192u;

// =============================================================================
// DAG node arena (Nebula mode)
// =============================================================================
//
// v0.40 — predicted-access-set DAG construction.
//
// drain_dagready ingests each VerifiedTx, walks its predicted access set,
// finds the most-recent prior writer per key (DagWriterSlot below), and
// emits parent→child edges. unresolved_parents tracks how many parents
// have not yet committed; once zero, the tx becomes part of the antichain
// (Prism frontier) and is emitted to Exec.

inline constexpr uint32_t kMaxDagParents    = 4u;
inline constexpr uint32_t kMaxDagChildren   = 16u;
inline constexpr uint32_t kMaxPredictedKeys = 4u;

/// DagNode lifecycle states. drain_dagready re-checks state after
/// publishing the parent→child edge; if Committed, it backs out its
/// unresolved_parents claim. drain_commit sets state=Committed AFTER
/// walking children.
enum DagNodeState : uint32_t {
    kDagNodeUnset      = 0u,
    kDagNodeRegistered = 1u,
    kDagNodeEmitted    = 2u,
    kDagNodeCommitted  = 3u,
};

struct alignas(16) DagNode {
    uint32_t tx_index;
    uint32_t parent_count;
    uint32_t unresolved_parents;   ///< atomically decremented on parent commit
    uint32_t child_count;
    uint32_t parents[kMaxDagParents];
    uint32_t children[kMaxDagChildren];
    // v0.40: cached envelope so drain_commit can re-emit a freed child
    // straight to Exec without re-popping from DagReady.
    uint64_t pending_gas_limit;
    uint32_t pending_origin_lo;
    uint32_t pending_origin_hi;
    uint32_t pending_admission;
    uint32_t state;                ///< DagNodeState
    // v0.41: bytecode slice — drain_exec needs it on every re-emit.
    uint32_t pending_blob_offset;
    uint32_t pending_blob_size;
};
// 16 (header) + 16 (parents[4]) + 64 (children[16]) + 32 (envelope+state+blob)
// = 128 bytes; alignas(16) is identity at this size.
static_assert(sizeof(DagNode) == 128, "DagNode layout drift");

/// DagWriterSlot — open-addressing table for most recent prior writers.
struct alignas(16) DagWriterSlot {
    uint64_t key_lo;
    uint64_t key_hi;
    uint32_t last_writer_tx;
    uint32_t _pad0;
};
// 24 bytes meaningful; alignas(16) bumps to 32.
static_assert(sizeof(DagWriterSlot) == 32, "DagWriterSlot layout drift");

inline constexpr uint32_t kDefaultDagWriterSlots = 8192u;

/// PredictedKey — host-supplied predicted read/write entry, indexed by
/// tx_index in a per-round arena.
struct PredictedKey {
    uint64_t key_lo;
    uint64_t key_hi;
    uint32_t is_write;
    uint32_t valid;
};
static_assert(sizeof(PredictedKey) == 24, "PredictedKey layout drift");

// =============================================================================
// EVM fiber VM (v0.41) — bytecode interpreter constants
// =============================================================================
//
// drain_exec walks each tx's bytecode through a switch-based opcode dispatch
// driven by FiberSlot. The fiber holds a 256-bit-wide stack of
// kFiberStackDepth entries (each a 4-limb little-endian U256), 1 KB of
// zero-initialized scratch memory, and a per-fiber instruction budget that
// bounds the dispatch loop so a malformed program can't run forever.
inline constexpr uint32_t kFiberStackDepth   = 64u;
inline constexpr uint32_t kFiberStackLimbs   = 4u;        ///< U256 = 4 × 64-bit
inline constexpr uint32_t kFiberMemoryBytes  = 1024u;
inline constexpr uint32_t kFiberInstrBudget  = 100000u;   ///< runaway guard

// FiberSlot::status — substrate semantics, reused by the EVM fiber VM.
inline constexpr uint32_t kFiberReady        = 0u;
inline constexpr uint32_t kFiberRunning      = 1u;
inline constexpr uint32_t kFiberWaitingState = 2u;
inline constexpr uint32_t kFiberCommittable  = 3u;
inline constexpr uint32_t kFiberReverted     = 4u;

// ExecResult::status — terminal disposition reported by drain_exec.
inline constexpr uint32_t kExecStatusReturn  = 1u;
inline constexpr uint32_t kExecStatusRevert  = 2u;
inline constexpr uint32_t kExecStatusOOG     = 3u;
inline constexpr uint32_t kExecStatusError   = 4u;
inline constexpr uint32_t kExecStatusSuspend = 5u;

// Berlin-ish gas costs (subset; v0.43 layers full EIP-2929 cold/warm and
// per-byte calldata accounting).
inline constexpr uint64_t kGasDefault    = 3u;
inline constexpr uint64_t kGasJumpdest   = 1u;
inline constexpr uint64_t kGasSloadWarm  = 100u;
inline constexpr uint64_t kGasSstore     = 5000u;
inline constexpr uint64_t kGasKeccakBase = 30u;
inline constexpr uint64_t kGasKeccakWord = 6u;
inline constexpr uint64_t kGasExpByte    = 50u;          ///< per byte of exponent

// =============================================================================
// Cold-state page-fault rings
// =============================================================================

enum class StateKeyType : uint32_t {
    Account = 0,
    Storage = 1,
    Code    = 2,
};

struct alignas(16) StateRequest {
    uint32_t tx_index;
    uint32_t key_type;
    uint32_t priority;
    uint32_t _pad0;
    uint64_t key_lo;
    uint64_t key_hi;
};
static_assert(sizeof(StateRequest) == 32, "StateRequest layout drift");

struct alignas(16) StatePage {
    uint32_t tx_index;
    uint32_t key_type;
    uint32_t status;
    uint32_t data_size;
    uint64_t key_lo;
    uint64_t key_hi;
    uint8_t  data[64];
};
static_assert(sizeof(StatePage) == 96, "StatePage layout drift");

// =============================================================================
// Quasar quorum (votes + QC) — Nova/Nebula both feed this
// =============================================================================

// Quasar consensus-mode tag. Nova=linear (protocol/nova/ray), Nebula=DAG
// (protocol/nebula/field). Host writes this once per round descriptor.
enum class QuasarMode : uint8_t {
    Nova   = 0,
    Nebula = 1,
};

// Three independently-toggleable signing layers from luxfi/consensus
// QuasarCert (protocol/quasar/types.go:40). VoteIngress::sig_kind picks
// the verifier for that lane; the substrate aggregates each lane
// independently.
//
// `MLDSA` here is the raw FIPS 204 ML-DSA-65 vote-signing kind. The third
// CERT LANE that aggregates these votes is a Groth16 SNARK
// (QuasarCertLane::MLDSAGroth16 in quasar_sig.hpp). Never call the cert
// lane raw "MLDSA"; the lane verifies a Groth16 proof, not the underlying
// ML-DSA-65 signatures directly.
enum class QuasarSigKind : uint8_t {
    BLS12_381 = 0,   ///< BLS12-381 aggregate (classical fast path)
    Ringtail  = 1,   ///< Ring-LWE 2-round threshold (PQ threshold)
    MLDSA     = 2,   ///< ML-DSA-65 (FIPS 204) — raw signing kind that
                     ///< feeds the MLDSAGroth16 cert lane
};

// CERT-021/006/007 (v0.42): round + stake widened to uint64; stake_weight
// is now bound into the MAC (via subject_with_vote — see drain_vote /
// quasar_verify_votes_kernel).
struct alignas(16) VoteIngress {
    uint32_t validator_index;
    uint32_t sig_kind;           ///< QuasarSigKind
    uint64_t round;              ///< CERT-021: full uint64 round
    uint64_t stake_weight;       ///< CERT-006/007: uint64, MAC-bound
    uint64_t _pad0;
    uint8_t  subject[32];        ///< MUST equal desc->certificate_subject
                                 ///< after CERT-003 — verifier rejects mismatch
    uint8_t  signature[96];      ///< BLS-G1 width covers BLS aggregates;
                                 ///< Ringtail / ML-DSA shares go out-of-band
                                 ///< via a host-managed signature arena
                                 ///< (this slot holds the digest only)
};
static_assert(sizeof(VoteIngress) == 32 + 32 + 96, "VoteIngress layout drift");

// Per-lane Quasar certificate aggregator. Three independent QuorumCerts
// land on QuorumOut — one per signing layer. The host composes them into
// a single QuasarCert (protocol/quasar/types.go) before forwarding to
// the consensus engine.
struct alignas(16) QuorumCert {
    uint32_t round;
    uint32_t status;             ///< 0=incomplete, 1=quorum, 2=conflict
    uint32_t signers_count;
    uint32_t total_stake;
    uint32_t sig_kind;           ///< QuasarSigKind — which lane this cert is for
    uint32_t _pad0;
    uint64_t _pad1;
    uint8_t  subject[32];        ///< matches the VoteIngress::subject
    uint8_t  agg_signature[96];  ///< BLS aggregate (classical) /
                                 ///< Ringtail aggregator state (PQ) /
                                 ///< Z-Chain Groth16 reference (ML-DSA)
};
static_assert(sizeof(QuorumCert) == 32 + 32 + 96, "QuorumCert layout drift");

// =============================================================================
// Round descriptor (host -> GPU, written once per round)
// =============================================================================
//
// "Round" here = one consensus decision boundary = one block in Nova mode,
// one DAG cut/frontier in Nebula mode.

struct alignas(16) QuasarRoundDescriptor {
    uint64_t chain_id;
    uint64_t round;
    uint64_t timestamp_ns;
    uint64_t deadline_ns;
    uint64_t gas_limit;
    uint64_t base_fee;
    uint32_t wave_tick_budget;   ///< per-service drain budget per wave tick.
                                 ///< Bounds workgroup work and yields back
                                 ///< to the GPU scheduler — the v0.29 fix.
    uint32_t wave_tick_index;    ///< monotonic across re-launches
    uint32_t closing_flag;
    uint32_t mode;               ///< QuasarMode — Nova=0, Nebula=1
    uint8_t  parent_block_hash[32];
    uint8_t  parent_state_root[32];
    uint8_t  parent_execution_root[32];
    // CERT-003 / CERT-010 / CERT-020 (v0.42): bind validator-set, ceremony,
    // VK roots, epoch, total_stake, validator_count.
    uint64_t epoch;                       ///< CERT-010: validator-set epoch
    uint64_t total_stake;                 ///< CERT-020: total stake
                                          ///< (replaces base_fee re-use)
    uint32_t validator_count;             ///< CERT-023: bound on validator_index
    uint32_t _pad0;
    uint8_t  pchain_validator_root[32];   ///< CERT-003: validator-set commitment
    uint8_t  qchain_ceremony_root[32];    ///< CERT-003: Ringtail ceremony
    uint8_t  zchain_vk_root[32];          ///< CERT-003: Z-Chain VK root
    uint8_t  certificate_subject[32];     ///< CERT-003 / v0.44: host-precomputed
                                          ///< keccak(chain_id||epoch||round||
                                          ///< mode||P||C||X||Q||Z||A||B||M||F||
                                          ///< parent_state||parent_execution||
                                          ///< gas_limit||base_fee). Canonical
                                          ///< 9-chain order: P, C, X, Q, Z,
                                          ///< A, B, M, F. C uses
                                          ///< parent_block_hash (this chain
                                          ///< IS C); the rest are dedicated
                                          ///< fields. Verifier rejects
                                          ///< v.subject != this.
    // v0.44 — five new per-chain transition roots. Each is the prior epoch's
    // canonical commitment from the corresponding VM. Zero means "no
    // contribution this round" (e.g. BridgeVM may lag by one epoch); the
    // certificate_subject still binds the zero so a tampered descriptor
    // produces a different cert.
    uint8_t  xchain_execution_root[32];   ///< XVM v0.55.x       (X-Chain)
    uint8_t  achain_state_root[32];       ///< AIVM v0.58.x      (A-Chain)
    uint8_t  bchain_state_root[32];       ///< BridgeVM v0.59.x  (B-Chain)
    uint8_t  mchain_state_root[32];       ///< MPCVM v0.60.x     (M-Chain)
    uint8_t  fchain_state_root[32];       ///< FHEVM             (F-Chain)
    // v0.42 cert ABI hardening — attestation_root + cert_mode.
    //
    //  attestation_root: keccak commitment over the round's CPU TEE +
    //   GPU TEE measurement + any other attestation witnesses. Bound into
    //   certificate_subject in canonical position (see
    //   compute_certificate_subject in quasar_sig.hpp). A tampered TEE
    //   measurement → different cert subject → verifier rejects.
    //
    //  cert_mode: QuasarCertMode discriminant — chooses whether this round
    //   runs the BLS-only fast path, BLS-now-PQ-async, or full PQ-blocking
    //   policy. Bound into certificate_subject so flipping the mode forces
    //   a fresh subject (cannot replay a fast-classical cert as a full-PQ
    //   cert and vice versa).
    uint8_t  attestation_root[32];
    uint8_t  cert_mode;                    ///< QuasarCertMode
    uint8_t  _pad_cert_mode[23];           ///< explicit pad to next 16B boundary
};
static_assert(sizeof(QuasarRoundDescriptor) == 528,
              "QuasarRoundDescriptor layout drift");

// =============================================================================
// Round result (GPU -> host)
// =============================================================================

// Round result — what the consensus engine certifies. Mirrors the
// commitments QuasarCert is computed over (luxfi/consensus
// protocol/quasar/types.go: QuasarCert is signed over a digest of
// these four roots plus mode and round).
// CERT-004 (v0.42): per-validator dedup bitmap on QuasarRoundResult.
// Bound by VALIDATOR_BITMAP_BITS — covers up to that many validator indices
// per lane; CERT-023 enforces validator_index < desc->validator_count, which
// itself must be <= VALIDATOR_BITMAP_BITS. Three cert lanes
// (BLSAggregate / RingtailThreshold / MLDSAGroth16) get independent bitmaps
// so a validator may sign each lane once.
inline constexpr uint32_t kValidatorBitmapBits = 256u;        ///< 256 validators
inline constexpr uint32_t kValidatorBitmapWords = kValidatorBitmapBits / 32u;

struct alignas(16) QuasarRoundResult {
    uint32_t status;             ///< 0=in-progress, 1=finalized, 2=needs_state, 3=failed
    uint32_t tx_count;
    uint32_t gas_used_lo;
    uint32_t gas_used_hi;
    uint32_t wave_tick_count;
    uint32_t conflict_count;     ///< Block-STM read-set conflicts detected
    uint32_t repair_count;       ///< re-executions performed
    uint32_t fibers_suspended;
    uint32_t fibers_resumed;
    // Cert-lane aggregator state. Names follow QuasarCertLane in
    // quasar_sig.hpp — third lane is `mldsa_groth16`, never raw `mldsa`.
    uint32_t quorum_status_bls;          ///< 0=incomplete, 1=quorum, 2=conflict
    uint32_t quorum_status_mldsa_groth16;///< MLDSAGroth16 lane (third cert lane)
    uint32_t quorum_status_rt;
    // CERT-007 (v0.42): uint64 stake split lo/hi (matches gas_used).
    uint32_t quorum_stake_bls_lo;
    uint32_t quorum_stake_bls_hi;
    uint32_t quorum_stake_mldsa_groth16_lo;
    uint32_t quorum_stake_mldsa_groth16_hi;
    uint32_t quorum_stake_rt_lo;
    uint32_t quorum_stake_rt_hi;
    uint32_t mode;               ///< QuasarMode
    uint32_t subject_mismatch_count;  ///< CERT-022: rejected votes
    uint32_t dedup_skipped_count;     ///< CERT-004: replays skipped
    uint32_t repair_capped_count;     ///< STM-003: txs hitting MAX_TOTAL_REPAIRS
    uint8_t  block_hash[32];     ///< keccak — populated by HashService
    uint8_t  state_root[32];
    uint8_t  receipts_root[32];
    uint8_t  execution_root[32]; ///< Block-STM trace root (independent of
                                 ///< the underlying state-trie shape)
    uint8_t  mode_root[32];      ///< nova_root (Nova) or nebula_root
                                 ///< (Nebula DAG/Horizon prefix) — the
                                 ///< mode-specific commitment
    // v0.44 — echo the 9 canonical chain roots used in certificate_subject.
    // The cert binds these; the result returns them so downstream consumers
    // (lux/consensus QuasarCert composer, audit trail) can reconstruct the
    // exact subject without re-parsing the descriptor.
    uint8_t  pchain_root_echo[32];
    uint8_t  cchain_root_echo[32];   ///< == desc.parent_block_hash
    uint8_t  xchain_root_echo[32];
    uint8_t  qchain_root_echo[32];
    uint8_t  zchain_root_echo[32];
    uint8_t  achain_root_echo[32];
    uint8_t  bchain_root_echo[32];
    uint8_t  mchain_root_echo[32];
    uint8_t  fchain_root_echo[32];
    uint8_t  certificate_subject_echo[32];   ///< == desc.certificate_subject
    // CERT-004 — three lane bitmaps, each kValidatorBitmapBits bits wide.
    uint32_t validator_voted_bitmap[3][kValidatorBitmapWords];

    uint64_t gas_used() const {
        return (uint64_t(gas_used_hi) << 32) | uint64_t(gas_used_lo);
    }
    uint64_t quorum_stake_bls() const {
        return (uint64_t(quorum_stake_bls_hi) << 32) | uint64_t(quorum_stake_bls_lo);
    }
    uint64_t quorum_stake_rt() const {
        return (uint64_t(quorum_stake_rt_hi) << 32) | uint64_t(quorum_stake_rt_lo);
    }
    uint64_t quorum_stake_mldsa_groth16() const {
        return (uint64_t(quorum_stake_mldsa_groth16_hi) << 32)
             | uint64_t(quorum_stake_mldsa_groth16_lo);
    }
};
// v0.44 — layout: 64 atomic counters + 32 (mode/subject/dedup/repair counters
// already counted) + 32*5 mode/state/receipts/execution/block roots + 32*10
// nine canonical chain echoes + cert_subject_echo + 3-lane bitmap.
static_assert(sizeof(QuasarRoundResult) ==
              64 + 32 + 32 * 5 + 32 * 10 + 3 * kValidatorBitmapWords * 4,
              "QuasarRoundResult layout drift");

}  // namespace quasar::gpu
