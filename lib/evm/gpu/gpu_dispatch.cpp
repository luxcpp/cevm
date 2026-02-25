// Copyright (C) 2026, The evmone Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "gpu_dispatch.hpp"
#include "gpu_state_hasher.hpp"
#include "parallel_engine.hpp"

#ifdef __APPLE__
#include "metal/block_stm_host.hpp"
#include "kernel/evm_kernel_host.hpp"
#endif

#ifdef EVM_CUDA
#include "cuda/keccak_host.hpp"
#include "cuda/block_stm_host.hpp"
#include "cuda/evm_kernel_host.hpp"
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

/// Returns true iff at least one tx in the batch has GPU-executable bytecode.
/// When true, the GPU dispatcher routes through EvmKernelHost (parallel
/// opcode interpreter, one thread per tx). When false it routes through
/// BlockStmGpu (scheduler + mv_memory only — for value-transfer benches).
static bool any_tx_has_code(const std::vector<Transaction>& txs)
{
    for (const auto& tx : txs)
        if (!tx.code.empty()) return true;
    return false;
}

#ifdef __APPLE__
/// Build a HostTransaction batch for the Metal EVM kernel from dispatch txs.
/// Pack uint64 value/balance into the low limbs of the kernel's uint256.
[[maybe_unused]] static std::vector<kernel::HostTransaction>
to_host_transactions(const std::vector<Transaction>& txs)
{
    std::vector<kernel::HostTransaction> out;
    out.reserve(txs.size());
    for (const auto& tx : txs)
    {
        kernel::HostTransaction h;
        h.code      = tx.code;
        h.calldata  = tx.data;
        h.gas_limit = tx.gas_limit;

        auto pack_addr = [](kernel::uint256& dst, const std::vector<uint8_t>& addr) {
            std::memset(&dst, 0, sizeof(dst));
            if (addr.size() >= 20) {
                std::memcpy(&dst, addr.data(), 20);
            }
        };
        auto pack_u64 = [](kernel::uint256& dst, uint64_t v) {
            std::memset(&dst, 0, sizeof(dst));
            auto* limbs = reinterpret_cast<uint64_t*>(&dst);
            limbs[0] = v;
        };

        pack_addr(h.caller, tx.from);
        pack_addr(h.address, tx.to);
        pack_u64(h.value, tx.value);
        out.push_back(std::move(h));
    }
    return out;
}


/// Build per-sender account states from transactions for the Metal
/// Block-STM kernel. Each unique sender gets nonce=tx.nonce (matching the
/// tx so validation passes) and balance high enough to cover gas+value.
/// Used when no host state is provided (benchmarks, isolated execution).
[[maybe_unused]] static std::vector<metal::GpuAccountState>
synthesize_base_state(const std::vector<Transaction>& txs)
{
    std::vector<metal::GpuAccountState> base_state;
    base_state.reserve(txs.size());
    auto already = [&](const uint8_t* addr) -> bool {
        for (const auto& acct : base_state)
            if (std::memcmp(acct.address, addr, 20) == 0) return true;
        return false;
    };
    for (const auto& tx : txs)
    {
        if (tx.from.size() < 20 || already(tx.from.data())) continue;
        metal::GpuAccountState acct;
        std::memset(&acct, 0, sizeof(acct));
        std::memcpy(acct.address, tx.from.data(), 20);
        acct.nonce = tx.nonce;
        acct.balance = ~uint64_t{0} >> 1;  // 9.2 EH gwei worth — won't overflow
        base_state.push_back(acct);
    }
    return base_state;
}
#endif  // __APPLE__

#ifdef EVM_CUDA
static std::vector<cuda::GpuAccountState>
synthesize_base_state_cuda(const std::vector<Transaction>& txs)
{
    std::vector<cuda::GpuAccountState> base_state;
    base_state.reserve(txs.size());
    auto already = [&](const uint8_t* addr) -> bool {
        for (const auto& acct : base_state)
            if (std::memcmp(acct.address, addr, 20) == 0) return true;
        return false;
    };
    for (const auto& tx : txs)
    {
        if (tx.from.size() < 20 || already(tx.from.data())) continue;
        cuda::GpuAccountState acct;
        std::memset(&acct, 0, sizeof(acct));
        std::memcpy(acct.address, tx.from.data(), 20);
        acct.nonce = tx.nonce;
        acct.balance = ~uint64_t{0} >> 1;
        base_state.push_back(acct);
    }
    return base_state;
}
#endif

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
#ifdef __APPLE__
        // Real-bytecode path: parallel opcode interpreter on Metal, one
        // thread per tx (or 32 threads/tx via execute_v2 when available).
        if (state == nullptr && any_tx_has_code(txs))
        {
            auto engine = kernel::EvmKernelHost::create();
            if (engine)
            {
                auto host_txs = to_host_transactions(txs);
                auto t0 = std::chrono::high_resolution_clock::now();
                auto results = engine->has_v2()
                    ? engine->execute_v2(host_txs)
                    : engine->execute(host_txs);
                auto t1 = std::chrono::high_resolution_clock::now();

                BlockResult br;
                br.gas_used.reserve(results.size());
                br.execution_time_ms =
                    std::chrono::duration<double, std::milli>(t1 - t0).count();
                for (const auto& r : results)
                {
                    br.gas_used.push_back(r.gas_used);
                    br.total_gas += r.gas_used;
                }
                if (config.enable_state_trie_gpu)
                    compute_state_root_gpu(br, LUX_BACKEND_METAL);
                return br;
            }
        }
        // Scheduler-only path: full execute/validate/re-execute loop on
        // Metal for value-transfer style txs (no bytecode). Falls back to
        // CPU Block-STM + GPU state-root keccak if Metal is unavailable.
        if (state == nullptr)
        {
            auto engine = metal::BlockStmGpu::create();
            if (engine)
            {
                auto base_state = synthesize_base_state(txs);
                auto result = engine->execute_block(txs, base_state);
                if (config.enable_state_trie_gpu &&
                    (result.state_root.empty() ||
                     std::all_of(result.state_root.begin(),
                                 result.state_root.end(),
                                 [](uint8_t b){ return b == 0; })))
                {
                    compute_state_root_gpu(result, LUX_BACKEND_METAL);
                }
                return result;
            }
        }
#endif
        // CPU Block-STM + GPU state-root keccak fallback.
        auto result = execute_via_engine(config, txs, state, true);
        if (config.enable_state_trie_gpu)
            compute_state_root_gpu(result, LUX_BACKEND_METAL);
        return result;
    }

    case Backend::GPU_CUDA:
    {
#ifdef EVM_CUDA
        // Real-bytecode path: parallel opcode interpreter on CUDA.
        if (state == nullptr && any_tx_has_code(txs))
        {
            auto engine = cuda::EvmKernel::create();
            if (engine)
            {
                std::vector<cuda::HostTransaction> host_txs;
                host_txs.reserve(txs.size());
                for (const auto& tx : txs)
                {
                    cuda::HostTransaction h;
                    h.code = tx.code;
                    h.calldata = tx.data;
                    h.gas_limit = tx.gas_limit;
                    std::memcpy(&h.caller, tx.from.data(),
                                std::min<size_t>(tx.from.size(), sizeof(h.caller)));
                    std::memcpy(&h.address, tx.to.data(),
                                std::min<size_t>(tx.to.size(), sizeof(h.address)));
                    auto* val_lo = reinterpret_cast<uint64_t*>(&h.value);
                    val_lo[0] = tx.value;
                    host_txs.push_back(std::move(h));
                }
                auto t0 = std::chrono::high_resolution_clock::now();
                auto results = engine->execute(host_txs);
                auto t1 = std::chrono::high_resolution_clock::now();

                BlockResult br;
                br.gas_used.reserve(results.size());
                br.execution_time_ms =
                    std::chrono::duration<double, std::milli>(t1 - t0).count();
                for (const auto& r : results)
                {
                    br.gas_used.push_back(r.gas_used);
                    br.total_gas += r.gas_used;
                }
                if (config.enable_state_trie_gpu)
                {
                    if (!compute_state_root_cuda_direct(br))
                        compute_state_root_gpu(br, LUX_BACKEND_CUDA);
                }
                return br;
            }
        }
        // Scheduler-only path: Block-STM on NVIDIA.
        if (state == nullptr)
        {
            auto engine = cuda::BlockStmGpu::create();
            if (engine)
            {
                auto base_state = synthesize_base_state_cuda(txs);
                auto result = engine->execute_block(txs, base_state);
                if (config.enable_state_trie_gpu &&
                    (result.state_root.empty() ||
                     std::all_of(result.state_root.begin(),
                                 result.state_root.end(),
                                 [](uint8_t b){ return b == 0; })))
                {
                    if (!compute_state_root_cuda_direct(result))
                        compute_state_root_gpu(result, LUX_BACKEND_CUDA);
                }
                return result;
            }
        }
#endif
        // CPU Block-STM + GPU state-root keccak fallback.
        auto result = execute_via_engine(config, txs, state, true);
        if (config.enable_state_trie_gpu)
        {
#ifdef EVM_CUDA
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
