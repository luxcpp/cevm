// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file slot_engine.hpp
/// Public C++ API for the GPU Slot Engine.
///
/// The host's job is reduced to:
///   * begin_slot(SlotDescriptor)   — once per slot
///   * push_txs(...)                — as packets arrive
///   * push_votes(...)              — as votes arrive
///   * push_state_pages(...)        — service GPU-emitted state requests
///   * poll_state_requests(...)     — drain GPU-emitted requests
///   * poll_result()                — eventually yields a finalized SlotResult
///   * end_slot()                   — close the kernel and release buffers
///
/// All scheduling, decode, recovery, admission, validation, repair, commit,
/// receipts, root computation, vote verification, and quorum formation
/// happens on the GPU inside slot_kernel.metal. The driver below is the
/// "doorbell + mailbox" boundary the architecture calls for.

#pragma once

#include "slot_layout.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace evm::gpu::slot {

/// Opaque handle returned by begin_slot. Tied to the engine that issued it.
struct SlotHandle {
    uint64_t opaque = 0;
    bool valid() const { return opaque != 0; }
};

/// One transaction blob plus the host-side metadata the slot kernel needs
/// before EVM execution lands inside the kernel (v0.36+).
struct HostTxBlob {
    std::vector<uint8_t> bytes;     ///< raw tx payload (RLP-encoded eventually)
    uint64_t gas_limit = 0;
    uint32_t nonce = 0;
    uint64_t origin = 0;            ///< first 8 bytes of sender; full address
                                    ///< recovery moves into the kernel later
};

/// Engine — one per device. Owns the slot kernel pipeline state, the
/// per-slot ring arena, and the in-flight Metal command buffers.
class SlotEngine {
public:
    virtual ~SlotEngine() = default;

    /// Construct on the system default Metal device. Returns nullptr if
    /// Metal or the slot kernel are unavailable.
    static std::unique_ptr<SlotEngine> create();

    /// Begin a slot. The descriptor's slot, timestamp, deadline, gas_limit,
    /// base_fee, and epoch_budget_items must be set. Returns an invalid
    /// handle if a slot is already active.
    virtual SlotHandle begin_slot(const SlotDescriptor& desc) = 0;

    /// Push a batch of txs into the ingress ring. Blocks the caller only if
    /// the ring is full (in which case the kernel must drain it first).
    virtual void push_txs(SlotHandle h,
                          std::span<const HostTxBlob> txs) = 0;

    /// Drive the slot scheduler kernel for one bounded epoch. Returns the
    /// current SlotResult snapshot (status=0 while in progress, 1 once
    /// finalized). Cheap on Apple Silicon — internally encodes a single
    /// MTLComputeCommandEncoder + 1 dispatch and waits for completion.
    /// Re-callable until poll_result reports status != 0.
    virtual SlotResult run_epoch(SlotHandle h) = 0;

    /// Convenience: run epochs until status != 0 or `max` epochs have
    /// elapsed. Returns the final result snapshot.
    virtual SlotResult run_until_done(SlotHandle h,
                                      std::size_t max_epochs = 1024) = 0;

    /// Non-blocking: read the latest published SlotResult. Mostly useful
    /// from a separate poller thread.
    virtual SlotResult poll_result(SlotHandle h) const = 0;

    /// Tell the kernel to drain and finalize at the end of the next epoch.
    virtual void request_close(SlotHandle h) = 0;

    /// Close the slot. Joins any in-flight command buffers and releases
    /// per-slot buffers. Idempotent.
    virtual void end_slot(SlotHandle h) = 0;

    /// True iff a slot is currently active.
    virtual bool slot_active() const = 0;

    /// Device name (debugging).
    virtual const char* device_name() const = 0;

    /// Per-service ring stats: total pushed and total consumed lifetime
    /// counters straight from the device-side RingHeader. Useful for
    /// asserting forward progress in tests.
    struct RingStats {
        uint32_t pushed = 0;
        uint32_t consumed = 0;
        uint32_t head = 0;
        uint32_t tail = 0;
        uint32_t capacity = 0;
    };
    virtual RingStats ring_stats(SlotHandle h, ServiceId s) const = 0;

protected:
    SlotEngine() = default;
    SlotEngine(const SlotEngine&) = delete;
    SlotEngine& operator=(const SlotEngine&) = delete;
};

}  // namespace evm::gpu::slot
