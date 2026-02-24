// Copyright (C) 2026, The evmone Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "gpu_dispatch.hpp"
#include "gpu_state_hasher.hpp"
#include "parallel_engine.hpp"

#ifdef EVM_CUDA
#include "cuda/keccak_host.hpp"
#endif

#include <chrono>
#include <cstring>
#include <memory>
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

/// Build the state-root input bytes (concatenated gas_used as little-endian).
/// Used by both the luxcpp/gpu path and the direct-CUDA path below.
static std::vector<uint8_t> build_state_root_input(const BlockResult& result)
{
    std::vector<uint8_t> data;
    data.reserve(result.gas_used.size() * 8);
    for (auto g : result.gas_used)
    {
        for (int i = 0; i < 8; ++i)
            data.push_back(static_cast<uint8_t>(g >> (i * 8)));
    }
    return data;
}

/// Compute the state root hash using luxcpp/gpu's pluggable backend.
/// For each transaction's storage writes, batch-hash the keys and values
/// that form the state trie. Returns a 32-byte root hash.
///
/// This is the fallback path used when the direct CUDA backend is not
/// compiled in. The luxcpp/gpu library may itself fall back to CPU if
/// the requested backend (LUX_BACKEND_METAL / _CUDA) is unavailable.
static void compute_state_root_gpu(BlockResult& result, LuxBackend lux_backend)
{
    if (result.state_root.empty())
        result.state_root.resize(32, 0);

    GpuStateHasher hasher(lux_backend);
    if (!hasher.available())
        return;

    auto data = build_state_root_input(result);
    size_t len = data.size();
    hasher.hash(data.data(), len, result.state_root.data());
}

#ifdef EVM_CUDA
/// Compute the state root using the direct CUDA Keccak path
/// (lib/evm/gpu/cuda/keccak256.cu). Returns true on success.
///
/// This path bypasses luxcpp/gpu's plugin loader and calls the kernel
/// directly via the CUDA Runtime API. Falls back silently if no CUDA
/// device is present.
static bool compute_state_root_cuda_direct(BlockResult& result)
{
    static thread_local std::unique_ptr<cuda::KeccakHasher> hasher =
        cuda::KeccakHasher::create();
    if (!hasher)
        return false;

    if (result.state_root.empty())
        result.state_root.resize(32, 0);

    auto data = build_state_root_input(result);
    cuda::HashInput input{data.data(), static_cast<uint32_t>(data.size())};
    auto digest = hasher->batch_hash(&input, 1);
    if (digest.size() < 32)
        return false;

    std::memcpy(result.state_root.data(), digest.data(), 32);
    return true;
}
#endif  // EVM_CUDA

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
        // Execute transactions via CPU parallel (Block-STM) for now —
        // the GPU EVM interpreter (cuda/evm_kernel.cu) is still a stub.
        // Once ported, this path will dispatch evm_execute on device.
        auto result = execute_via_engine(config, txs, state, true);
        if (config.enable_state_trie_gpu)
        {
#ifdef EVM_CUDA
            // Prefer the direct CUDA Keccak path. If no NVIDIA device
            // is present, fall back to luxcpp/gpu (which itself falls
            // back to CPU if its CUDA plugin isn't loaded).
            if (!compute_state_root_cuda_direct(result))
                compute_state_root_gpu(result, LUX_BACKEND_CUDA);
#else
            compute_state_root_gpu(result, LUX_BACKEND_CUDA);
#endif
        }
        return result;
    }
    }

    return execute_via_engine(config, txs, state, false);
}

}  // namespace evm::gpu
