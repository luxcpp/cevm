// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file v3_queue.hpp
/// Shared layout for the V3 persistent-kernel work queues.
///
/// Three queues drive the V3 pipeline:
///   * exec_q     — host pushes (tx_index, incarnation); exec_worker drains
///   * validate_q — exec_worker pushes; validate_worker drains
///   * commit_q   — validate_worker pushes; commit_worker drains
///
/// Each queue is a fixed-capacity ring buffer in unified memory. Producers
/// bump `tail` with `atomic_fetch_add`; consumers bump `head` likewise.
/// Capacity is `1 << log2_capacity` so wrap is a single mask. Backpressure
/// is the producer's job: if `tail - head` >= capacity, retry.
///
/// Layout MUST match the device-side counterpart in v3_persistent.metal
/// byte-for-byte. See V3_QUEUE_CAPACITY for the agreed capacity.
///
/// One way to do queues: this header is the single source of truth. Both
/// the host (.mm) and the kernel (.metal) include it.

#pragma once

#include <cstdint>

namespace evm::gpu::kernel::v3 {

/// Power-of-two capacity for every queue. 16384 covers a typical wave (a
/// few thousand txs) with headroom; producers spin if full, which the
/// backpressure test exercises.
inline constexpr uint32_t Q_LOG2_CAPACITY = 14;
inline constexpr uint32_t Q_CAPACITY      = 1u << Q_LOG2_CAPACITY;
inline constexpr uint32_t Q_MASK          = Q_CAPACITY - 1u;

/// One work item flowing through the persistent pipeline.
/// Layout MUST match struct WorkItem in v3_persistent.metal.
struct WorkItem
{
    uint32_t tx_index;     ///< Index into the per-batch arrays (inputs[], results[])
    uint32_t incarnation;  ///< Block-STM incarnation (0 on first try, +1 per replay)
    uint32_t wave_id;      ///< Wave number (for pipelining / FPC tagging later)
    uint32_t flags;        ///< Reserved (FPC-certified, etc.); 0 in v0.29
};

/// Per-queue control block. The kernel reads/writes head and tail with
/// atomic ops; capacity stays constant for the lifetime of one V3 session.
///
/// The "items" array follows the QueueHeader contiguously in the buffer;
/// host reserves one MTLBuffer holding `sizeof(QueueHeader) + sizeof(WorkItem) * Q_CAPACITY`.
/// Layout MUST match struct QueueHeader in v3_persistent.metal.
struct alignas(16) QueueHeader
{
    uint32_t head;       ///< Next index to consume.
    uint32_t tail;       ///< Next index to produce.
    uint32_t mask;       ///< Q_MASK; passed via header so kernel can read it once.
    uint32_t _pad0;
};

static_assert(sizeof(QueueHeader) == 16, "QueueHeader must be 16 bytes");
static_assert(sizeof(WorkItem)    == 16, "WorkItem must be 16 bytes");

/// Global control block shared by all three persistent kernels. Lives in
/// its own MTLBuffer so the host can flip `shutdown_flag` to 1 to signal
/// orderly exit.
///
/// Layout MUST match struct V3Control in v3_persistent.metal.
struct alignas(16) V3Control
{
    uint32_t shutdown_flag;   ///< 0 = run; 1 = drain & exit
    uint32_t exec_alive;      ///< Workgroups still executing (debug)
    uint32_t validate_alive;  ///< Workgroups still validating (debug)
    uint32_t commit_alive;    ///< Workgroups still committing (debug)

    uint32_t exec_done;       ///< Total tx successfully executed (lifetime)
    uint32_t validate_done;   ///< Total tx successfully validated
    uint32_t commit_done;     ///< Total tx committed (lifetime)
    uint32_t _pad0;
};

static_assert(sizeof(V3Control) == 32, "V3Control must be 32 bytes");

}  // namespace evm::gpu::kernel::v3
