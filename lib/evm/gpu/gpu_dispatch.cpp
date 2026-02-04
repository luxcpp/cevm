// Copyright (C) 2026, The evmone Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "gpu_dispatch.hpp"
#include "gpu_state_hasher.hpp"
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
    backends.push_back(Backend::GPU_Metal);
#endif

#ifdef EVM_CUDA
    backends.push_back(Backend::GPU_CUDA);
#endif

    return backends;
}

Backend auto_detect()
{
    auto backends = available_backends();
    // Preference: GPU_Metal > GPU_CUDA > CPU_Parallel > CPU_Sequential
    for (auto pref : {Backend::GPU_Metal, Backend::GPU_CUDA, Backend::CPU_Parallel})
    {
        for (auto b : backends)
        {
            if (b == pref)
                return pref;
        }
    }
    return Backend::CPU_Sequential;
}

bool set_backend(Config& config, Backend b)
{
    auto backends = available_backends();
    for (auto avail : backends)
    {
        if (avail == b)
        {
            config.backend = b;
            return true;
        }
    }
    return false;
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

/// Compute the state root hash using GPU-accelerated Keccak-256.
/// For each transaction's storage writes, batch-hash the keys and values
/// that form the state trie. Returns a 32-byte root hash.
static void compute_state_root_gpu(BlockResult& result, LuxBackend lux_backend)
{
    if (result.state_root.empty())
        result.state_root.resize(32, 0);

    // The state root is a Keccak-256 hash of the concatenated gas_used values
    // as a minimal proof-of-concept. A real implementation would hash the
    // post-execution state trie. This demonstrates the GPU hashing path.
    GpuStateHasher hasher(lux_backend);
    if (!hasher.available())
        return;

    // Build input: concatenate gas_used as bytes
    std::vector<uint8_t> data;
    data.reserve(result.gas_used.size() * 8);
    for (auto g : result.gas_used)
    {
        for (int i = 0; i < 8; ++i)
            data.push_back(static_cast<uint8_t>(g >> (i * 8)));
    }

    size_t len = data.size();
    hasher.hash(data.data(), len, result.state_root.data());
}

BlockResult execute_block(const Config& config,
                          const std::vector<Transaction>& txs,
                          void* state)
{
    switch (config.backend)
    {
    case Backend::CPU_Sequential:
    {
        auto result = execute_via_engine(config, txs, state, false);
        if (config.enable_state_trie_gpu)
            compute_state_root_gpu(result, LUX_BACKEND_CPU);
        return result;
    }

    case Backend::CPU_Parallel:
    {
        auto result = execute_via_engine(config, txs, state, true);
        if (config.enable_state_trie_gpu)
            compute_state_root_gpu(result, LUX_BACKEND_CPU);
        return result;
    }

    case Backend::GPU_Metal:
    {
        // Execute transactions via CPU parallel (Block-STM), then
        // offload state trie hashing to Metal GPU.
        auto result = execute_via_engine(config, txs, state, true);
        if (config.enable_state_trie_gpu)
            compute_state_root_gpu(result, LUX_BACKEND_METAL);
        return result;
    }

    case Backend::GPU_CUDA:
    {
        auto result = execute_via_engine(config, txs, state, true);
        if (config.enable_state_trie_gpu)
            compute_state_root_gpu(result, LUX_BACKEND_CUDA);
        return result;
    }
    }

    return execute_via_engine(config, txs, state, false);
}

}  // namespace evm::gpu
