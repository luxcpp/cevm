// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file evmc_bridge.hpp
/// EVMC Host bridge for the GPU EVM kernel.
///
/// The GPU EVM kernel (Metal + CUDA) cannot make host calls mid-execution:
/// a kernel cannot fetch external code/balance/nonce while running. The bridge
/// makes the kernel "complete" by:
///
///   1. Statically analysing each transaction's bytecode.
///   2. For every reachable CALL/CREATE target with a constant address,
///      pre-fetching the called account's code/balance/nonce from the host
///      (`evmc::Host`).
///   3. Driving the kernel interpreter as a stack of call frames, with
///      orchestration in C++ between frames.
///
/// Anything that requires a runtime-computed address (CALLDATALOAD then CALL,
/// SELFBALANCE then CREATE, etc.) is reported as "needs CPU fallback" via
/// `std::nullopt`. The dispatcher then routes those transactions to evmone
/// with a proper `evmc::Host*`.
///
/// The bridge is designed to keep the existing GPU fast-path entirely intact
/// — the leaf-frame interpreter is unchanged — and to add a thin frame
/// manager around it.
///
/// Limits (derived from existing kernel buffers):
///   * 8 frames per transaction (vs the 1024 EVM allows).
///   * 65 536 bytes of memory per frame (HOST_MAX_MEMORY_PER_TX).
///   * 1 024 bytes of return data per frame (HOST_MAX_OUTPUT_PER_TX).
///   * 64 storage slots per (frame, account) pair (HOST_MAX_STORAGE_PER_TX).
///
/// Hitting any of these → `std::nullopt`, fall back to evmone.

#pragma once

#include "../kernel/evm_kernel_host.hpp"

#include <evmc/evmc.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace evm::gpu::host {

/// One transaction the bridge will try to execute.
///
/// Mirrors the dispatcher's `evm::gpu::Transaction` shape but using
/// EVMC types so the bridge can resolve receiver addresses against
/// `evmc::Host` directly.
struct Transaction
{
    evmc::address sender{};
    evmc::address recipient{};       ///< Zero address for contract creation.
    evmc::uint256be value{};
    std::vector<uint8_t> input;       ///< Calldata (or initcode for CREATE).
    int64_t gas = 0;
    bool is_create = false;
};

/// Result of executing one transaction through the bridge.
struct TxResult
{
    evmc_status_code status = EVMC_FAILURE;
    int64_t gas_used = 0;
    int64_t gas_refund = 0;
    std::vector<uint8_t> output;
    /// True when the GPU kernel executed at least one frame.
    /// False means everything ran in the bridge frame manager (no GPU work).
    /// Either way, the result matches what evmone would return.
    bool used_gpu = false;
};

/// Hard limits — see file header.
constexpr unsigned MAX_FRAME_DEPTH = 8;

/// Bridge between an `evmc::Host` and the GPU EVM kernel.
class GpuHostBridge
{
public:
    /// Build a bridge attached to `host`. The bridge keeps a reference,
    /// so the host must outlive the bridge.
    static std::unique_ptr<GpuHostBridge> create(evmc::Host& host);

    /// Try to run `tx` end-to-end. Returns `std::nullopt` whenever the
    /// transaction needs effects the bridge cannot satisfy without a real
    /// host (dynamic CALL targets, depth > MAX_FRAME_DEPTH, contracts
    /// holding more storage than the kernel's per-frame budget, etc.).
    ///
    /// On success the result is byte-equivalent to running the same tx
    /// on evmone with the same host and revision.
    std::optional<TxResult> try_execute(const Transaction& tx, evmc_revision rev);

    /// Batch helper. The output vector has the same length as `txs`;
    /// each entry is either a successful result or `std::nullopt`.
    std::vector<std::optional<TxResult>> try_execute_batch(
        std::span<const Transaction> txs, evmc_revision rev);

    GpuHostBridge(const GpuHostBridge&) = delete;
    GpuHostBridge& operator=(const GpuHostBridge&) = delete;

    ~GpuHostBridge();

private:
    explicit GpuHostBridge(evmc::Host& host);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace evm::gpu::host
