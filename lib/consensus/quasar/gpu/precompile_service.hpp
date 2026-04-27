// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file precompile_service.hpp
/// QuasarGPU v0.43 — precompile-as-service ABI.
///
/// User directive: "Precompiles should not be opaque calls. They should be GPU
/// services that produce rootable artifacts."
///
/// Each precompile is one device-resident queue + one batched drain stage +
/// one rootable artifact. Fibers calling a precompile suspend on the queue
/// and resume when the matching result lands. At round end, every active
/// precompile_id emits a PrecompileArtifact (input_root, output_root,
/// gas_root, transcript_root) that rolls into QuasarRoundResult::execution_root.
///
/// Layout invariants:
///   * All structs trivially-copyable, alignas(16).
///   * Per-id queues live in unified memory; producers atomic-bump tail,
///     consumers atomic-bump head. Wrap is single-mask AND.
///   * FiberState transitions are wave-tick-driven; the scheduler kernel in
///     fiber_suspend_resume.metal moves a fiber Ready → WaitingPrecompile on
///     yield, WaitingPrecompile → Ready on result match.
///   * PrecompileArtifact is keccak-rolled host-side from the call records the
///     service drained — the same recipe runs on GPU in v0.44.

#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace quasar::gpu::precompile {

/// 32-byte rooted artifact hash type. Aliased here so the cert ABI sibling
/// (lib/consensus/quasar/gpu/quasar_sig.hpp) and the precompile lane share
/// one canonical hash representation.
using Hash = std::array<uint8_t, 32>;

// =============================================================================
// PrecompileCall — one fiber's request envelope into a per-id queue.
// =============================================================================
//
// Bytes 0..15 layout (atomic-friendly, hot path):
//   tx_id (4) | fiber_id (4) | precompile_id (2) | flags (2) |
//   input_offset (4) | input_len (4)
// Bytes 16..31:
//   output_offset (4) | output_capacity (4) | gas_budget (8)

struct PrecompileCall {
    uint32_t tx_id;
    uint32_t fiber_id;
    uint16_t precompile_id;       ///< 0x01..0x11 (ETH std), 0x0A01..0x0A08 (AIVM),
                                  ///< 0x0400..0x04FF (DEX), etc.
    uint16_t flags;
    uint32_t input_offset;        ///< into per-round call_data arena
    uint32_t input_len;
    uint32_t output_offset;       ///< into per-round result_data arena
    uint32_t output_capacity;
    uint64_t gas_budget;
};
// Natural layout with 8-byte alignment for gas_budget:
//   tx_id(4) | fiber_id(4) | precompile_id(2) | flags(2) | input_offset(4) |
//   input_len(4) | output_offset(4) | output_capacity(4) | _pad(4) |
//   gas_budget(8)
// = 40 bytes. The GPU-side mirror in precompile_dispatch.metal uses the same
// field ordering; size differences across host/device must be caught by the
// matching static_assert there (the .metal file relies on host-staged
// buffers, so a host-side change here is the canonical source of truth).
static_assert(sizeof(PrecompileCall) == 40, "PrecompileCall layout drift");

// =============================================================================
// PrecompileResult — one per call, written back to the result ring.
// =============================================================================

enum : uint16_t {
    kStatusOk            = 0,
    kStatusOOG           = 1,
    kStatusInvalidInput  = 2,
    kStatusInternalError = 3,
};

struct PrecompileResult {
    uint32_t tx_id;
    uint32_t fiber_id;
    uint16_t status;              ///< kStatusOk/.../kStatusInternalError
    uint16_t flags;
    uint32_t output_len;
    uint64_t gas_used;
};
// Natural layout: 4+4+2+2+4+8 = 24 bytes. Caller is free to pack into a
// larger transport record; this is the on-the-wire shape.
static_assert(sizeof(PrecompileResult) == 24, "PrecompileResult layout drift");

// =============================================================================
// PrecompileArtifact — per-precompile_id rooted commitment.
// =============================================================================
//
// Computed at round end by the service. Each root is a binary keccak Merkle
// over the per-call records drained for this precompile_id, in queue order:
//
//   input_root      = root of keccak256(input_bytes_i)
//   output_root     = root of keccak256(output_bytes_i)
//   gas_root        = root of keccak256_le(gas_used_i)
//   transcript_root = root of keccak256(input_bytes_i || output_bytes_i ||
//                                       gas_used_le8 || status_le2)
//
// transcript_root is the canonical commitment — it binds inputs, outputs, gas,
// and status atomically, so a single hash collision attempt would have to
// match all four lanes. The QuasarRoundResult::execution_root rolls every
// active precompile_id's transcript_root into the per-round commitment.

struct PrecompileArtifact {
    uint16_t precompile_id;
    uint16_t _pad0;
    uint32_t call_count;
    uint32_t failed_count;
    uint32_t _pad1;
    Hash input_root;
    Hash output_root;
    Hash gas_root;
    Hash transcript_root;
};
// Header (2+2+4+4+4 = 16) + 4 * 32-byte hashes = 144 bytes natural.
static_assert(sizeof(PrecompileArtifact) == 16 + 4 * 32,
              "PrecompileArtifact layout drift");

// =============================================================================
// FiberState — substrate slot used by fiber_suspend_resume.metal.
// =============================================================================
//
// The wave-tick scheduler tests `status == kFiberWaitingPrecompile` and the
// pair (waiting_precompile_id, request_id). When the matching result lands,
// status flips back to kFiberReady and the result_index is set so the fiber
// can read its result without scanning.

enum : uint32_t {
    kFiberReady              = 0,   ///< runnable; scheduler will dispatch
    kFiberRunning            = 1,
    kFiberWaitingPrecompile  = 2,   ///< blocked on (waiting_precompile_id, request_id)
    kFiberCompleted          = 3,
    kFiberReverted           = 4,
};

struct FiberState {
    uint32_t fiber_id;
    uint32_t tx_id;
    uint32_t status;                ///< kFiberReady/.../kFiberReverted
    uint32_t waiting_precompile_id; ///< meaningful iff status == kFiberWaitingPrecompile
    uint32_t request_id;            ///< unique within a precompile_id
    uint32_t result_index;          ///< populated on wake; index into result_ring
    uint64_t resume_pc;             ///< host-supplied resume point (opaque to scheduler)
};
// 6 * 4 + 8 = 32 bytes — naturally 8-aligned, matches the GPU-side struct.
static_assert(sizeof(FiberState) == 32, "FiberState layout drift");

// =============================================================================
// Per-precompile queue identity.
// =============================================================================
//
// Range buckets are defined here so callers can sanity-check the precompile_id
// without depending on PrecompileService internals.
inline constexpr uint16_t kEthStdMin     = 0x01;
inline constexpr uint16_t kEthStdMax     = 0x11;
inline constexpr uint16_t kLuxDexMin     = 0x0400;
inline constexpr uint16_t kLuxDexMax     = 0x04FF;
inline constexpr uint16_t kAivmMin       = 0x0A01;
inline constexpr uint16_t kAivmMax       = 0x0A08;

inline bool is_known_precompile_id(uint16_t id) {
    if (id >= kEthStdMin && id <= kEthStdMax) return true;
    if (id == 0x100) return true;                                ///< DEX_MATCH legacy
    if (id >= kLuxDexMin && id <= kLuxDexMax) return true;
    if (id >= kAivmMin && id <= kAivmMax) return true;
    return false;
}

// =============================================================================
// PrecompileService — host-side facade for the on-device service stages.
// =============================================================================
//
// Lifetime is per-round (pairs with QuasarGPUEngine::begin_round). Internally
// owns:
//   * One per-id MTLBuffer queue (head/tail/items) with item type PrecompileCall
//   * One result ring (PrecompileResult[])
//   * One scratch arena for input/output bytes referenced by offsets
//   * One FiberState[] view (provided by caller — same buffer the EVM fiber
//     interpreter writes)
//
// The service drains queues by precompile_id, executes the per-id batch via
// the real GPU lane (ecrecover, bls, point_eval already exist; keccak runs
// through metal::KeccakHasher; everything else falls through to the existing
// PrecompileDispatcher which preserves consensus-equivalent output bytes), and
// emits results back to the ring. wake_waiting_fibers transitions every
// fiber blocked on a now-ready (precompile_id, request_id) back to Ready.

class PrecompileService {
public:
    virtual ~PrecompileService() = default;

    /// Construct on the system default Metal device. Returns nullptr if no
    /// device is available.
    static std::unique_ptr<PrecompileService> create();

    /// Reset for a new round. Clears every per-id queue, the result ring, and
    /// the per-call arenas. Must be called exactly once before push_call.
    virtual void begin_round(uint64_t round, uint64_t chain_id) = 0;

    /// Enqueue one fiber's call. Caller is responsible for pre-staging input
    /// bytes into the input arena (see input_arena()) and reserving output
    /// space in the output arena (output_arena()). The (input_offset,
    /// input_len, output_offset, output_capacity) on the PrecompileCall index
    /// into those arenas. Returns the request_id assigned to this call (used
    /// by fiber_yield to bind the fiber to the right slot).
    virtual uint32_t push_call(const PrecompileCall& call) = 0;

    /// Yield: transition fiber `fiber_id` from Ready → WaitingPrecompile,
    /// bound to (precompile_id, request_id). Idempotent if already waiting on
    /// the same pair. Returns false if fiber_id is out of range.
    virtual bool fiber_yield(uint32_t fiber_id,
                             uint16_t precompile_id,
                             uint32_t request_id,
                             uint64_t resume_pc) = 0;

    /// Drain every per-id queue: batch-execute, write results, run wake. One
    /// wave-tick of progress. Returns the number of calls processed across
    /// all ids (sum). Safe to call repeatedly until it returns 0.
    virtual uint32_t drain_one_tick() = 0;

    /// Fully drain. Convenience wrapper over drain_one_tick that loops until
    /// every queue is empty. Returns total calls processed.
    virtual uint32_t drain_all() = 0;

    /// Look up a result by (precompile_id, request_id). Returns nullptr if
    /// the call has not yet been processed. Pointer is valid until end_round.
    virtual const PrecompileResult* result_for(uint16_t precompile_id,
                                               uint32_t request_id) const = 0;

    /// View of the output bytes for a finalized call. Returns empty span if
    /// status != kStatusOk or the call has not been drained.
    virtual std::span<const uint8_t> result_bytes(
        uint16_t precompile_id, uint32_t request_id) const = 0;

    /// Compute every active precompile_id's artifact. Must be called after
    /// drain_all (live calls would corrupt the roots otherwise). Returns one
    /// artifact per id with at least one call this round.
    virtual std::vector<PrecompileArtifact> emit_artifacts() = 0;

    /// Bind the FiberState[] this service should drive. Caller owns the
    /// memory; the service writes to (status, result_index) on wake.
    virtual void bind_fibers(FiberState* fibers, size_t fiber_count) = 0;

    // -- arenas ---------------------------------------------------------------
    /// Input data arena — caller writes input bytes here, sets PrecompileCall
    /// input_offset to the byte index. Span lifetime is the round.
    virtual std::span<uint8_t> input_arena(size_t bytes) = 0;
    virtual std::span<uint8_t> output_arena(size_t bytes) = 0;

    /// Tear down state for the current round. Idempotent.
    virtual void end_round() = 0;

    /// Diagnostic: number of unique precompile ids seen this round.
    virtual uint32_t active_id_count() const = 0;

    /// Diagnostic: device name (e.g. "Apple M3 Max"). "cpu" if no GPU.
    virtual const char* device_name() const = 0;

protected:
    PrecompileService() = default;
    PrecompileService(const PrecompileService&) = delete;
    PrecompileService& operator=(const PrecompileService&) = delete;
};

}  // namespace quasar::gpu::precompile
