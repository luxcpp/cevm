// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file v3_persistent_host.hpp
/// V3 persistent-kernel host driver.
///
/// One-way contract: a `V3PersistentRunner` owns a single Metal device and
/// holds three persistent kernel launches alive (exec_worker,
/// validate_worker, commit_worker). The runner exposes one async API:
///
///     enqueue_wave(transactions) -> WaveFuture
///
/// Each call pushes the txs onto the exec queue. The kernels drain
/// continuously without re-dispatch — this is the difference vs. v0.28's
/// dispatch-per-call model. `await()` on the returned future blocks until
/// every tx in the wave has reached `committed`.
///
/// Tear-down is explicit: `shutdown()` flips the device-side
/// `V3Control::shutdown_flag` and joins the kernel command buffers. Calling
/// the destructor without shutdown() asserts.
///
/// Limits in v0.29:
///   * Single device (no multi-GPU)
///   * No on-device MVCC yet — validate_worker is a structural pass-through
///     that confirms the read-set is empty (correct for the v0.29 corpus,
///     which intentionally exercises the queue mechanics, not MVCC). v0.30
///     wires the SlotEntry/Version arena from the design doc.
///   * No FPC integration; the wave_id field is plumbed but unused.

#pragma once

#include "evm_kernel_host.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace evm::gpu::kernel::v3 {

/// Future returned by V3PersistentRunner::enqueue_wave. Resolves once
/// every tx in the wave has been committed.
///
/// `await()` blocks the calling thread; `ready()` is a non-blocking poll.
class WaveFuture
{
public:
    virtual ~WaveFuture() = default;

    /// Block until the wave completes; return per-tx results.
    /// May be called only once.
    virtual std::vector<TxResult> await() = 0;

    /// Block until either the wave completes or the timeout elapses.
    /// Returns true if completed, false on timeout (results stay pending).
    virtual bool await_for(std::chrono::milliseconds timeout) = 0;

    /// Non-blocking: true once every tx in the wave is committed.
    virtual bool ready() const = 0;

    /// True once the GPU exec_worker has finished this wave (pre-validate).
    /// Useful for the pipeline test to assert ordering.
    virtual bool exec_done() const = 0;
};

/// V3 persistent-kernel runner. One instance owns one Metal device.
class V3PersistentRunner
{
public:
    virtual ~V3PersistentRunner() = default;

    /// Construct on the system default Metal device. Returns nullptr if
    /// Metal is unavailable or the persistent kernels fail to compile /
    /// dispatch.
    static std::unique_ptr<V3PersistentRunner> create();

    /// Push a wave of transactions onto the exec queue. Blocks the caller
    /// only if the queue lacks capacity (backpressure).
    ///
    /// The returned future resolves when the wave's last tx has reached
    /// the commit_q. The caller may submit further waves before awaiting
    /// — that is the pipelining the design doc calls out.
    virtual std::unique_ptr<WaveFuture> enqueue_wave(
        std::span<const HostTransaction> txs,
        const BlockContext& ctx) = 0;

    /// Convenience: empty BlockContext.
    virtual std::unique_ptr<WaveFuture> enqueue_wave(
        std::span<const HostTransaction> txs) = 0;

    /// Flip the shutdown flag and join all three persistent kernels.
    /// Idempotent. Subsequent enqueue_wave calls throw.
    virtual void shutdown() = 0;

    /// Lifetime counters (post-shutdown they freeze; useful for tests).
    struct Counters
    {
        uint64_t executed;
        uint64_t validated;
        uint64_t committed;
        uint32_t exec_alive;
        uint32_t validate_alive;
        uint32_t commit_alive;
    };
    virtual Counters counters() const = 0;

    /// True iff shutdown() has been called.
    virtual bool is_shut_down() const = 0;

    /// Device name for logging.
    virtual const char* device_name() const = 0;

protected:
    V3PersistentRunner() = default;
    V3PersistentRunner(const V3PersistentRunner&) = delete;
    V3PersistentRunner& operator=(const V3PersistentRunner&) = delete;
};

}  // namespace evm::gpu::kernel::v3
