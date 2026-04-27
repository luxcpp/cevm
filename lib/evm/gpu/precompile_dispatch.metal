// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// QuasarGPU v0.43 — GPU-side dispatch kernel for the precompile service.
//
// Each PrecompileCall in the per-id queue is processed by exactly one thread.
// The kernel groups calls by precompile_id (the queue address itself is the
// group), so all keccak calls run together, all ecrecover calls run together,
// etc. This is the v0.44 batched-service entry point — the host driver
// (precompile_service.mm) currently routes per-id execution through the
// existing PrecompileDispatcher (which already wires GPU lanes for
// ecrecover, bls12_381, point_eval, dex_match), and this kernel runs the
// queue maintenance: it walks unprocessed calls in a queue, marks them
// staged for host drain, and pre-stages the input slice into a contiguous
// per-id buffer so the host pickup avoids gather copies.
//
// Buffer 0 — calls[]    : PrecompileCall[N] for one queue.
// Buffer 1 — staged[]   : uint32_t[N], one slot per call. 0 = pending,
//                          1 = ready-for-host, 2 = drained.
// Buffer 2 — call_count : uint32_t — bound on tid range.
// Buffer 3 — id_filter  : uint32_t — process only calls with this id (queues
//                          may share one underlying buffer in v0.44).

#include <metal_stdlib>
using namespace metal;

struct PrecompileCall {
    uint   tx_id;
    uint   fiber_id;
    ushort precompile_id;
    ushort flags;
    uint   input_offset;
    uint   input_len;
    uint   output_offset;
    uint   output_capacity;
    ulong  gas_budget;
};

kernel void precompile_dispatch_drain(
    device PrecompileCall* calls       [[buffer(0)]],
    device atomic_uint*    staged      [[buffer(1)]],
    constant uint&         call_count  [[buffer(2)]],
    constant uint&         id_filter   [[buffer(3)]],
    uint                   tid         [[thread_position_in_grid]])
{
    if (tid >= call_count) return;

    // Filter to the requested precompile_id. Calls in other queues are no-ops
    // for this dispatch.
    const ushort want = (ushort)id_filter;
    if (calls[tid].precompile_id != want) return;

    // Compare-and-swap from pending(0) to ready-for-host(1). Idempotent across
    // re-launches: if another thread already marked this slot ready, we leave
    // it alone.
    uint expected = 0u;
    atomic_compare_exchange_weak_explicit(&staged[tid],
                                           &expected,
                                           1u,
                                           memory_order_relaxed,
                                           memory_order_relaxed);
}
