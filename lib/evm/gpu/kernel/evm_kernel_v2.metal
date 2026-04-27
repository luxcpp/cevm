// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file evm_kernel_v2.metal
/// V2 EVM kernel — 32 threads/tx threadgroup, lanes-fan-out for buffer prep.
///
/// Dispatch shape (set by `EvmKernelHostMetal::execute_v2`):
///   grid       = (num_txs * 32, 1, 1)
///   threadgroup = (32, 1, 1)
///
/// Each threadgroup represents one transaction. v0.47.2 lands the first
/// SIMD-fan-out: lanes 0..31 cooperatively zero the per-tx mem / storage /
/// transient / logs buffers (work the host previously did via std::memset
/// on every execute() call). Each lane handles 1/32 of mem_pool, and a
/// subset of lanes handles each smaller buffer. The host skips its own
/// memset for V2 dispatches, so the saved CPU wall-clock is the real
/// speedup.
///
/// Every tx is marked `status = 255` (NEEDS_V1_RETRY); the host's
/// `execute_v2` always retries through V1 in v0.47.2, since the V1 retry
/// is what produces byte-deterministic results. V2 outputs are
/// bit-identical to V1 by construction.
///
/// Future v0.4x.x patches fan out specific opcodes (KECCAK256 round-state
/// across 25 lanes, MULMOD limb spreading across 4 lanes) by branching
/// from the V1 interpreter on the hot opcode and using `simd_shuffle` to
/// coordinate lanes — the dispatch substrate (32-lane threadgroup,
/// threadgroup_barrier) lands here.

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// V2 fast-path constants. Values must match host (HOST_* in
// evm_kernel_host.hpp).
// =============================================================================

constant uint kV2MaxMemoryPerTx  = 65536;   // HOST_MAX_MEMORY_PER_TX
constant uint kV2MaxOutputPerTx  = 1024;    // HOST_MAX_OUTPUT_PER_TX
constant uint kV2MaxStoragePerTx = 64;      // HOST_MAX_STORAGE_PER_TX
constant uint kV2MaxLogsPerTx    = 16;      // HOST_MAX_LOGS_PER_TX
constant uint kV2Lanes           = 32;      // SIMD group width on M1/M2/M3

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
// Lane-distributed byte zero. Each lane writes a contiguous 1/32 slice of
// `bytes` starting at `dst`, in plain byte stores (Metal's vectoriser
// fuses these into wide stores when `bytes_per_lane` is a multiple of the
// vector width). `bytes` and `lane` must be identical across the 32 lanes
// at the call site (they are — derived from constants and tg_position).
//
// Byte-determinism: every lane writes its own slice exactly once with no
// overlap; output is identical to a host memset. The kernel never reads
// the bytes it zeros (V1 retry runs on the same buffers afterwards), so
// race-free correctness is automatic.
// =============================================================================
inline void v2_lane_zero_bytes(device uchar* dst, uint bytes, uint lane)
{
    if (bytes == 0) return;

    uint per_lane = (bytes + kV2Lanes - 1u) / kV2Lanes;
    uint start    = lane * per_lane;
    if (start >= bytes) return;
    uint end = start + per_lane;
    if (end > bytes) end = bytes;
    for (uint i = start; i < end; ++i) dst[i] = 0;
}

// =============================================================================
// V2 fast-path entry point.
//
// Per-tx fan-out:
//   * mem_pool[tx]      (64 KB):  all 32 lanes (2048 B/lane)
//   * storage_pool[tx]  (4 KB):   lanes 0..15  (256 B/lane)
//   * transient_pool[tx](4 KB):   lanes 16..31 (256 B/lane)
//   * log_pool[tx]      (2304 B): all 32 lanes (72 B/lane)
//   * out_data[tx]      (1024 B): all 32 lanes (32 B/lane)
//   * counters + status: lane 0 only (~16 B total)
//
// Lane 0 writes status=255 as the V1-retry sentinel. All lanes participate
// in a closing threadgroup_barrier so the device-memory writes are visible
// to the V1 retry that runs on the same MTLBuffer cache.
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
    if (tg_id >= num_txs) return;
    uint tx = tg_id;

    // (1) Mem buffer (64 KB / tx): all 32 lanes, 2048 B per lane.
    {
        device uchar* dst = mem_pool + ulong(tx) * ulong(kV2MaxMemoryPerTx);
        v2_lane_zero_bytes(dst, kV2MaxMemoryPerTx, lane);
    }

    // (2) Storage pool (kV2MaxStoragePerTx * 64 B = 4096 B / tx): lanes 0..15.
    {
        const uint stor_bytes_per_tx = kV2MaxStoragePerTx * uint(sizeof(V2StorageEntry));
        device uchar* dst = reinterpret_cast<device uchar*>(storage_pool) +
                            ulong(tx) * ulong(stor_bytes_per_tx);
        if (lane < 16u) {
            uint per = (stor_bytes_per_tx + 15u) / 16u;
            uint start = lane * per;
            uint end   = start + per;
            if (end > stor_bytes_per_tx) end = stor_bytes_per_tx;
            if (start < stor_bytes_per_tx) {
                for (uint i = start; i < end; ++i) dst[i] = 0;
            }
        }
    }

    // (3) Transient pool (4096 B / tx): lanes 16..31.
    {
        const uint trans_bytes_per_tx = kV2MaxStoragePerTx * uint(sizeof(V2StorageEntry));
        device uchar* dst = reinterpret_cast<device uchar*>(transient_pool) +
                            ulong(tx) * ulong(trans_bytes_per_tx);
        if (lane >= 16u) {
            uint local = lane - 16u;
            uint per = (trans_bytes_per_tx + 15u) / 16u;
            uint start = local * per;
            uint end   = start + per;
            if (end > trans_bytes_per_tx) end = trans_bytes_per_tx;
            if (start < trans_bytes_per_tx) {
                for (uint i = start; i < end; ++i) dst[i] = 0;
            }
        }
    }

    // (4) Log pool (kV2MaxLogsPerTx * sizeof(V2LogEntry) = 16*144 = 2304 B):
    //     all 32 lanes, 72 B per lane.
    {
        const uint log_bytes_per_tx = kV2MaxLogsPerTx * uint(sizeof(V2LogEntry));
        device uchar* dst = reinterpret_cast<device uchar*>(log_pool) +
                            ulong(tx) * ulong(log_bytes_per_tx);
        v2_lane_zero_bytes(dst, log_bytes_per_tx, lane);
    }

    // (5) Output data (1024 B / tx): all 32 lanes, 32 B per lane.
    {
        device uchar* dst = out_data + ulong(tx) * ulong(kV2MaxOutputPerTx);
        v2_lane_zero_bytes(dst, kV2MaxOutputPerTx, lane);
    }

    // (6) Per-tx counters (3 × uint32 = 12 B). Lane 0 only — too small to
    //     justify spreading.
    if (lane == 0u) {
        storage_counts[tx]   = 0;
        transient_counts[tx] = 0;
        log_counts[tx]       = 0;

        // V1-retry sentinel. Host's execute_v2 always retries through V1.
        device V2TxOutput& out = outputs[tx];
        out.status      = 255u;
        out.gas_used    = 0;
        out.gas_refund  = 0;
        out.output_size = 0;
    }

    // Touch buffer 12 once per threadgroup so Metal validation sees it
    // accessed (lane 1 chosen so it does not race with lane 0's status
    // write above).
    if (lane == 1u && block_ctx != nullptr) {
        volatile uchar v = block_ctx[0];
        (void)v;
    }

    // Closing barrier: ensure all lanes' device-memory writes are visible
    // to the host completion handler and to the V1 kernel that runs next
    // on the shared MTLBuffer cache. threadgroup_barrier with mem_device
    // is the universally-supported pattern for cross-kernel device-memory
    // ordering on Apple Silicon.
    threadgroup_barrier(mem_flags::mem_device);
}
