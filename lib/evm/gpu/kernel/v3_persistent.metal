// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file v3_persistent.metal
/// V3 wave-dispatch kernel — one Metal dispatch per host wave.
///
/// v0.29 attempted persistent kernels with cross-workgroup queues. Apple
/// Silicon's compute scheduler does not pre-empt hot-spinning workgroups,
/// so the exec workgroup starved validate/commit no matter how the grid
/// was structured. v0.30 reverts to the same one-shot pattern every other
/// passing GPU test in this tree uses (evm_kernel.metal, block_stm,
/// metal_host): the host encodes one MTLComputeCommandEncoder per wave
/// with N workgroups (one per tx). Each workgroup runs the full
/// exec → validate → commit pipeline for its tx in a straight line.
///
/// The Counters block keeps the same shape v0.29 exposed (executed,
/// validated, committed, *_alive) so the public C++ API does not change.
/// alive counters are no longer meaningful on the wave model — there is
/// no long-lived workgroup to count — and stay at zero between waves.

#include <metal_stdlib>
using namespace metal;

// -----------------------------------------------------------------------------
// Layout — matches lib/evm/gpu/kernel/v3_queue.hpp byte-for-byte.
// -----------------------------------------------------------------------------

struct alignas(16) V3Control {
    atomic_uint shutdown_flag;
    atomic_uint exec_alive;
    atomic_uint validate_alive;
    atomic_uint commit_alive;

    atomic_uint exec_done;
    atomic_uint validate_done;
    atomic_uint commit_done;
    uint        _pad0;
};

struct V3TxInput {
    uint  code_size;
    uint  code_offset;
    uint  calldata_size;
    uint  calldata_offset;
    ulong gas_limit;
    uint  read_set_size;
    uint  read_set_offset;
    uint  _pad0;
    uint  _pad1;
};

struct V3TxResult {
    uint  status;
    ulong gas_used;
    uint  output_size;
    uint  _pad0;
};

// -----------------------------------------------------------------------------
// Per-tx executor. Deterministic synthetic results keyed off code_size — same
// contract v3_persistent_test.mm asserts on. The full EVM interpreter wires
// in via a separate v0.31 task; this kernel exists to prove the wave-dispatch
// pipeline + counter monotonicity that the V3 design doc calls for.
// -----------------------------------------------------------------------------

static inline void exec_one_tx(device const V3TxInput& in,
                               device V3TxResult&      r)
{
    r.status      = 1u;  // TxStatus::Return
    r.gas_used    = ulong(in.code_size) * 21u + 21000u;
    r.output_size = (in.code_size < 32u) ? in.code_size : 32u;
}

// -----------------------------------------------------------------------------
// v3_wave_kernel — one workgroup per tx.
//
// Each workgroup uses tid==0 to run the tx through exec → validate → commit
// in sequence. atomic counters give the host a monotonic view of pipeline
// progress; per-tx commit flag tells WaveFuture::ready() the work is done.
// -----------------------------------------------------------------------------

kernel void v3_wave_kernel(
    device const V3TxInput* inputs    [[buffer(0)]],
    device V3TxResult*      results   [[buffer(1)]],
    device atomic_uint*     committed [[buffer(2)]],
    device V3Control*       ctl       [[buffer(3)]],
    constant uint&          base      [[buffer(4)]],
    constant uint&          count     [[buffer(5)]],
    uint tid                          [[thread_index_in_threadgroup]],
    uint gid                          [[threadgroup_position_in_grid]])
{
    if (tid != 0u) return;
    if (gid >= count) return;

    const uint idx = base + gid;

    // exec
    exec_one_tx(inputs[idx], results[idx]);
    atomic_fetch_add_explicit(&ctl->exec_done, 1u, memory_order_relaxed);

    // validate (empty read-set ⇒ pass; non-empty ⇒ flag as Error per v0.29
    // contract). Real MVCC re-execution is a v0.32+ concern.
    if (inputs[idx].read_set_size != 0u) {
        results[idx].status = 4u;  // TxStatus::Error
    }
    atomic_fetch_add_explicit(&ctl->validate_done, 1u, memory_order_relaxed);

    // commit — flip the per-tx flag and bump the counter. WaveFuture polls
    // the MTLCommandBuffer completion handler to know when results are
    // ready; the per-tx flag is kept for the V2 streaming consumers.
    atomic_store_explicit(&committed[idx], 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(&ctl->commit_done, 1u, memory_order_relaxed);
}
