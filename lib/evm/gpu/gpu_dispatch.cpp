// Copyright (C) 2026, The evmone Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "gpu_dispatch.hpp"
#include "parallel_engine.hpp"

#include <chrono>
#include <thread>

namespace evm::gpu
{

const char* backend_name(Backend b)
{
    switch (b)
    {
    case Backend::CPU_Sequential: return "cpu-sequential";
    case Backend::CPU_Parallel:   return "cpu-parallel (Block-STM)";
    case Backend::GPU_Metal:      return "gpu-metal";
    case Backend::GPU_CUDA:       return "gpu-cuda";
    }
    return "unknown";
}

std::vector<Backend> available_backends()
{
    std::vector<Backend> backends;
    backends.push_back(Backend::CPU_Sequential);
    backends.push_back(Backend::CPU_Parallel);

#ifdef __APPLE__
    // Metal is available on macOS/iOS
    backends.push_back(Backend::GPU_Metal);
#endif

#ifdef EVM_CUDA
    backends.push_back(Backend::GPU_CUDA);
#endif

    return backends;
}

/// Execute a block using the dispatch-layer Transaction type.
/// Converts to EvmTransaction and delegates to the real engine.
/// The state pointer is expected to be an evmc::Host* when non-null.
static BlockResult execute_via_engine(const Config& config,
                                      const std::vector<Transaction>& txs,
                                      void* state,
                                      bool parallel)
{
    // Convert dispatch-layer transactions to EVM transactions
    std::vector<EvmTransaction> evm_txs;
    evm_txs.reserve(txs.size());
    for (const auto& tx : txs)
        evm_txs.push_back(to_evm_transaction(tx));

    // If caller provided a Host, use it; otherwise fall back to gas-only mode
    if (state != nullptr)
    {
        auto* host = static_cast<evmc::Host*>(state);
        if (parallel)
        {
            return execute_parallel_evmone(evm_txs, *host, EVMC_SHANGHAI,
                config.num_threads);
        }
        return execute_sequential_evmone(evm_txs, *host, EVMC_SHANGHAI);
    }

    // No host provided: run gas-estimation-only mode (no state access).
    // This preserves backward compatibility with callers that pass nullptr.
    BlockResult result;
    result.gas_used.resize(txs.size());

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < txs.size(); ++i)
    {
        result.gas_used[i] = txs[i].gas_limit;
        result.total_gas += txs[i].gas_limit;
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.execution_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    return result;
}

BlockResult execute_block(const Config& config,
                          const std::vector<Transaction>& txs,
                          void* state)
{
    switch (config.backend)
    {
    case Backend::CPU_Sequential:
        return execute_via_engine(config, txs, state, false);

    case Backend::CPU_Parallel:
        return execute_via_engine(config, txs, state, true);

    case Backend::GPU_Metal:
        // Metal compute shaders not yet implemented.
        // Fall back to CPU parallel (Block-STM) which uses all cores.
        return execute_via_engine(config, txs, state, true);

    case Backend::GPU_CUDA:
        // CUDA kernels not yet implemented.
        // Fall back to CPU parallel (Block-STM) which uses all cores.
        return execute_via_engine(config, txs, state, true);
    }

    return execute_via_engine(config, txs, state, false);
}

}  // namespace evm::gpu
