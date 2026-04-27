// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// QuasarGPU v0.43 — fiber state machine for batched precompile waits.
//
// Two kernels, one buffer layout:
//
//   FiberState {
//       uint   fiber_id;
//       uint   tx_id;
//       uint   status;                  // 0=Ready, 1=Running,
//                                       // 2=WaitingPrecompile,
//                                       // 3=Completed, 4=Reverted
//       uint   waiting_precompile_id;
//       uint   request_id;
//       uint   result_index;            // populated on wake
//       ulong  resume_pc;               // host-supplied resume point
//   };
//   sizeof = 32 bytes — matches precompile_service.hpp alignas(16).
//
// fiber_yield_to_precompile: one thread per fiber. If the fiber is Ready and
// the host has staged a yield request for it (encoded in the request[]
// parallel array), transition Ready -> WaitingPrecompile. Idempotent.
//
// fiber_wake_from_precompile: one thread per fiber. Scans the result ring for
// any (precompile_id, request_id) the fiber is waiting on; on a match, flip
// status to Ready and stash the encoded result_index.

#include <metal_stdlib>
using namespace metal;

struct FiberState {
    uint  fiber_id;
    uint  tx_id;
    uint  status;
    uint  waiting_precompile_id;
    uint  request_id;
    uint  result_index;
    ulong resume_pc;
};

struct PrecompileResult {
    uint   tx_id;
    uint   fiber_id;
    ushort status;
    ushort flags;
    uint   output_len;
    ulong  gas_used;
};

constant uint kFiberReady             = 0;
constant uint kFiberWaitingPrecompile = 2;

// Per-fiber yield request — host-staged. The host writes this array before
// calling the kernel; one entry per fiber. valid==0 means "no request, leave
// the fiber alone".
struct YieldRequest {
    uint   valid;
    uint   precompile_id;
    uint   request_id;
    uint   _pad;
    ulong  resume_pc;
};

kernel void fiber_yield_to_precompile(
    device   FiberState*    fibers       [[buffer(0)]],
    device   YieldRequest*  requests     [[buffer(1)]],
    constant uint&          fiber_count  [[buffer(2)]],
    uint                    tid          [[thread_position_in_grid]])
{
    if (tid >= fiber_count) return;
    YieldRequest req = requests[tid];
    if (req.valid == 0u) return;

    device FiberState& f = fibers[tid];
    // Allow the transition only from Ready. A fiber already waiting must be
    // released by a wake before it can yield again.
    if (f.status != kFiberReady) return;

    f.status                = kFiberWaitingPrecompile;
    f.waiting_precompile_id = req.precompile_id;
    f.request_id            = req.request_id;
    f.resume_pc             = req.resume_pc;
    f.result_index          = 0xFFFFFFFFu;

    // Consume the request so a re-launched kernel does not double-yield.
    requests[tid].valid = 0u;
}

// One thread per fiber. For every fiber in WaitingPrecompile state, scan the
// result ring for a matching (precompile_id, request_id); on the first match,
// flip to Ready and encode result_index = (precompile_id << 16) | request_id.
//
// Three parallel arrays index the result ring:
//   results[i]      — the PrecompileResult itself (status, gas, output_len)
//   ids[i]          — precompile_id this result came from
//   request_ids[i]  — per-id request_id this result fulfills
kernel void fiber_wake_from_precompile(
    device   FiberState*       fibers       [[buffer(0)]],
    device   PrecompileResult* results      [[buffer(1)]],
    device   ushort*           ids          [[buffer(2)]],
    device   uint*             request_ids  [[buffer(3)]],
    constant uint&             result_count [[buffer(4)]],
    constant uint&             fiber_count  [[buffer(5)]],
    uint                       tid          [[thread_position_in_grid]])
{
    if (tid >= fiber_count) return;
    device FiberState& f = fibers[tid];
    if (f.status != kFiberWaitingPrecompile) return;

    const ushort want_id  = (ushort)f.waiting_precompile_id;
    const uint   want_rid = f.request_id;

    // Linear scan: result_count is bounded by drained-this-tick which is at
    // most a few thousand. Threadgroup memory is not needed because reads are
    // coalesced and the inner loop is identical across the wave. results[i]
    // is touched only via the i loop variable so the compiler eliminates the
    // unused load — silences the unused-buffer warning while keeping the
    // ABI shape (host may bind result bytes for additional checks later).
    (void)results;
    for (uint i = 0u; i < result_count; ++i) {
        if (ids[i] != want_id) continue;
        if (request_ids[i] != want_rid) continue;

        f.status       = kFiberReady;
        f.result_index = ((uint)want_id << 16) | (want_rid & 0xFFFFu);
        return;
    }
}
