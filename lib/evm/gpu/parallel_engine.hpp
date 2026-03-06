// Copyright (C) 2026, The cevm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file parallel_engine.hpp
/// Block-STM parallel execution engine wired to cevm's interpreter.
///
/// Takes a vector of transactions, creates MvMemory + Scheduler,
/// spawns N worker threads. Each worker runs cevm's baseline execute()
/// with a ParallelHost that routes state access through MvMemory.

#pragma once

#include "gpu_dispatch.hpp"
#include <evmc/evmc.hpp>
#include <cstdint>
#include <vector>

namespace evm::gpu
{

/// A transaction prepared for EVM execution.
/// Carries the EVMC message and bytecode needed by cevm.
struct EvmTransaction
{
    evmc_message msg{};
    std::vector<uint8_t> code;  ///< EVM bytecode to execute (empty for simple transfers)
};

/// Execute a block of transactions sequentially using cevm.
///
/// @param txs         Prepared EVM transactions.
/// @param base_host   The underlying state database host.
/// @param rev         EVM revision (e.g. EVMC_SHANGHAI).
/// @return            Block execution result.
BlockResult execute_sequential_cevm(
    const std::vector<EvmTransaction>& txs,
    evmc::Host& base_host,
    evmc_revision rev);

/// Execute a block of transactions in parallel using Block-STM + cevm.
///
/// @param txs         Prepared EVM transactions.
/// @param base_host   The underlying state database host.
/// @param rev         EVM revision (e.g. EVMC_SHANGHAI).
/// @param num_threads Number of worker threads (0 = auto-detect).
/// @return            Block execution result with conflict/re-execution stats.
BlockResult execute_parallel_cevm(
    const std::vector<EvmTransaction>& txs,
    evmc::Host& base_host,
    evmc_revision rev,
    uint32_t num_threads);

/// Convert gpu::Transaction (the dispatch-layer type) to EvmTransaction.
EvmTransaction to_evm_transaction(const Transaction& tx);

}  // namespace evm::gpu
