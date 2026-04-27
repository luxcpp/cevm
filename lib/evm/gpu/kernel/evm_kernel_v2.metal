// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file evm_kernel_v2.metal
/// V2 EVM kernel — 32 threads/tx threadgroup, leader-follower SIMD.
///
/// Dispatch shape (set by `EvmKernelHostMetal::execute_v2`):
///   grid       = (num_txs * 32, 1, 1)
///   threadgroup = (32, 1, 1)
///
/// Each threadgroup represents one transaction. Lane 0 is the leader; lanes
/// 1..31 currently idle and participate only in the closing barrier so the
/// host completion handler observes a fully committed write set. The
/// dispatch shape is the substrate the v0.45.x patches will fan out to —
/// memory expansion, keccak rounds, log emit and per-call gas metering all
/// have parallel structure that maps to lanes 1..31 and lands in subsequent
/// patches against this same kernel entry point.
///
/// The output bytes (status / gas_used / gas_refund / output / storage /
/// logs) are byte-identical to V1 because the V2 entry forwards to the V1
/// implementation: the host sets the V2 buffer status to 255 on every tx
/// not handled by the V2 fast-path, and the host retries those txs on the
/// V1 pipeline. CPU oracle parity is preserved on the same workloads as
/// v0.44.

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// V2 fast-path constants. The values must match the V1 host header:
// HOST_MAX_OUTPUT_PER_TX = 1024, HOST_MAX_STORAGE_PER_TX = 64,
// HOST_MAX_LOGS_PER_TX = 16. Replicated here so V2 stays self-contained
// and does not transitively include the V1 metal source (which would
// re-define `evm_execute`).
// =============================================================================

constant uint kV2MaxStoragePerTx = 64;
constant uint kV2MaxLogsPerTx    = 16;

// V1 header layouts. Buffer indices and field order MUST match
// evm_kernel_host.hpp byte-for-byte.

struct uint256_v2 {
    ulong w[4];
};

struct V2TxInput {
    uint code_offset;
    uint code_size;
    uint calldata_offset;
    uint calldata_size;
    ulong gas_limit;
    uint256_v2 caller;
    uint256_v2 address;
    uint256_v2 value;
    uint warm_addr_offset;
    uint warm_addr_count;
    uint warm_slot_offset;
    uint warm_slot_count;
};

struct V2TxOutput {
    uint   status;
    ulong  gas_used;
    long   gas_refund;
    uint   output_size;
};

struct V2StorageEntry {
    uint256_v2 key;
    uint256_v2 value;
};

struct V2LogEntry {
    uint256_v2 topics[4];
    uint   num_topics;
    uint   data_offset;
    uint   data_size;
    uint   _pad0;
};

// =============================================================================
// V2 fast-path entry point.
//
// We mark every tx with `status = 255` (NEEDS_V1_RETRY). The host's
// `execute_v2()` recognises this status and re-dispatches the affected
// txs through the V1 pipeline. The dispatch infrastructure (32-lane
// threadgroups, lane 0 leader, mem_device barrier) is the durable v0.45
// surface; the per-opcode SIMD parallelisation lands in v0.45.x.
//
// This design preserves byte-deterministic outputs (the V1 fallback path
// is the same code that produced the v0.44 numbers in BENCHMARKS.md) and
// gives the host a stable launch shape to amortise over.
// =============================================================================
kernel void evm_execute_v2(
    device const V2TxInput*     inputs           [[buffer(0)]],
    device const uchar*         blob             [[buffer(1)]],
    device V2TxOutput*          outputs          [[buffer(2)]],
    device uchar*               out_data         [[buffer(3)]],
    device uchar*               mem_pool         [[buffer(4)]],
    device V2StorageEntry*      storage_pool     [[buffer(5)]],
    device uint*                storage_counts   [[buffer(6)]],
    device const uint*          params           [[buffer(7)]],
    device V2StorageEntry*      transient_pool   [[buffer(8)]],
    device uint*                transient_counts [[buffer(9)]],
    device V2LogEntry*          log_pool         [[buffer(10)]],
    device uint*                log_counts       [[buffer(11)]],
    device const uchar*         block_ctx        [[buffer(12)]],
    uint tg_id                                   [[threadgroup_position_in_grid]],
    uint lane                                    [[thread_position_in_threadgroup]])
{
    uint num_txs = params[0];
    uint tx = tg_id;

    if (tx >= num_txs) return;

    // Lanes 1..31 idle today. They reach the closing barrier so that any
    // device-memory writes by lane 0 above are committed before the
    // threadgroup retires and the command buffer signals completion.
    if (lane != 0) {
        threadgroup_barrier(mem_flags::mem_device);
        return;
    }

    // Lane 0: write the V2 sentinel status. The host's execute_v2 recognises
    // this and runs V1 on the same batch. Per-tx storage/log counters are
    // reset to 0 so the V1 pass starts from a clean slate (the host also
    // memsets these, this is belt + suspenders).
    device V2TxOutput& out = outputs[tx];
    out.status = 255u;        // V2 fast-path: needs V1 retry
    out.gas_used = 0;
    out.gas_refund = 0;
    out.output_size = 0;

    storage_counts[tx] = 0;
    transient_counts[tx] = 0;
    log_counts[tx] = 0;

    // Touch one byte of each per-tx buffer so the threadgroup's device-mem
    // visibility ordering with mem_pool / out_data is well-defined for the
    // host completion handler. Cheap; one cache line per tx.
    if (block_ctx != nullptr) {
        // No-op read; ensures buffer 12 is bound (Metal validates this).
        volatile uchar v = block_ctx[0];
        (void)v;
    }
    if (mem_pool != nullptr) mem_pool[tx * 1u] = 0;
    if (out_data != nullptr) out_data[tx * 1u] = 0;

    threadgroup_barrier(mem_flags::mem_device);
}
