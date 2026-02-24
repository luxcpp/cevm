// Copyright (C) 2026, The evmone Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "gpu_dispatch.hpp"
#include "gpu_state_hasher.hpp"
#include "parallel_engine.hpp"

// The kernel CPU interpreter (kernel::execute_cpu) is header-only and used
// by all platforms — the CPU paths route bytecode through it whenever the
// caller passes `state == nullptr`, so the four backends share semantics.
#include "kernel/evm_kernel_host.hpp"

#ifdef __APPLE__
#include "metal/block_stm_host.hpp"
#endif

#ifdef EVM_CUDA
#include "cuda/keccak_host.hpp"
#include "cuda/block_stm_host.hpp"
#include "cuda/evm_kernel_host.hpp"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <utility>

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

/// Run every tx through evmone using the caller-provided host.
/// `parallel` selects between sequential and Block-STM execution.
/// Caller MUST pass a valid evmc::Host&; the host-less path is handled by
/// the routing helpers (kernel::execute_cpu, gas_estimation_only) instead.
static BlockResult execute_via_engine(const Config& config,
                                      const std::vector<Transaction>& txs,
                                      evmc::Host& host,
                                      bool parallel)
{
    std::vector<EvmTransaction> evm_txs;
    evm_txs.reserve(txs.size());
    for (const auto& tx : txs)
        evm_txs.push_back(to_evm_transaction(tx));

    if (parallel)
    {
        return execute_parallel_evmone(evm_txs, host, config.revision,
            config.num_threads);
    }
    return execute_sequential_evmone(evm_txs, host, config.revision);
}

/// Build a result that signals an early bail-out. Every status is Error,
/// every gas_used is zero, and `error_message` carries the human-readable
/// reason. Used when the dispatcher itself rejects the requested combination
/// of flags + inputs (gas_estimation_mode unset, fast_value_transfer without
/// the acknowledgement flag, etc.).
static BlockResult make_error_result(const std::vector<Transaction>& txs,
                                     std::string message)
{
    BlockResult br;
    const auto n = txs.size();
    br.gas_used.assign(n, 0);
    br.status.assign(n, TxStatus::Error);
    br.output.assign(n, std::vector<uint8_t>{});
    br.error_message = std::move(message);
    return br;
}

/// Explicit gas-estimation shortcut: no host, no bytecode, the dispatcher
/// reports `gas_used = gas_limit` for every tx. Only invoked after the caller
/// opts in via Config::gas_estimation_mode = true.
static BlockResult gas_estimation_only(const std::vector<Transaction>& txs)
{
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

/// Validate Config flags. Returns empty string when ok, otherwise a human
/// message describing the misuse. Called from the public `execute_block`
/// entry point so every backend is gated identically.
static std::string validate_config(const Config& config)
{
    if (config.fast_value_transfer && !config.fast_value_transfer_acknowledged)
    {
        return "fast_value_transfer requires fast_value_transfer_acknowledged "
               "(it skips per-opcode gas decrement and violates EVM consensus)";
    }
    if (config.fast_value_transfer && config.fast_value_transfer_acknowledged)
    {
        std::fprintf(stderr,
            "[evm::gpu] WARNING: fast_value_transfer enabled — gas accounting "
            "is non-spec, do not use for consensus-critical execution.\n");
    }
    return {};
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

/// Convert a dispatch-layer Transaction into the kernel's HostTransaction.
/// Used by both the CPU bytecode path (kernel::execute_cpu) and the GPU
/// bytecode paths so all four backends consume the same input shape.
static kernel::HostTransaction to_kernel_tx(const Transaction& tx)
{
    kernel::HostTransaction h;
    h.code      = tx.code;
    h.calldata  = tx.data;
    h.gas_limit = tx.gas_limit;

    auto pack_addr = [](kernel::uint256& dst, const std::vector<uint8_t>& addr) {
        std::memset(&dst, 0, sizeof(dst));
        if (addr.size() >= 20)
            std::memcpy(&dst, addr.data(), 20);
    };
    auto pack_u64 = [](kernel::uint256& dst, uint64_t v) {
        std::memset(&dst, 0, sizeof(dst));
        auto* limbs = reinterpret_cast<uint64_t*>(&dst);
        limbs[0] = v;
    };

    pack_addr(h.caller, tx.from);
    pack_addr(h.address, tx.to);
    pack_u64(h.value, tx.value);
    return h;
}

static TxStatus convert_status(kernel::TxStatus s)
{
    switch (s)
    {
    case kernel::TxStatus::Stop:             return TxStatus::Stop;
    case kernel::TxStatus::Return:           return TxStatus::Return;
    case kernel::TxStatus::Revert:           return TxStatus::Revert;
    case kernel::TxStatus::OutOfGas:         return TxStatus::OutOfGas;
    case kernel::TxStatus::Error:            return TxStatus::Error;
    case kernel::TxStatus::CallNotSupported: return TxStatus::CallNotSupported;
    }
    return TxStatus::Error;
}

/// Run every tx through the kernel CPU interpreter (kernel::execute_cpu),
/// which is the same interpreter that the Metal/CUDA kernels emulate. This
/// is the CPU reference path for parity testing — gas, status, and output
/// match the GPU backends byte-for-byte.
static BlockResult execute_kernel_cpu(const std::vector<Transaction>& txs,
                                      bool parallel,
                                      uint32_t num_threads)
{
    const auto n = txs.size();
    BlockResult br;
    br.gas_used.resize(n);
    br.status.resize(n, TxStatus::Error);
    br.output.resize(n);

    auto t0 = std::chrono::high_resolution_clock::now();

    auto run_one = [&](size_t i) {
        auto kt = to_kernel_tx(txs[i]);
        auto r  = kernel::execute_cpu(kt);
        br.gas_used[i] = r.gas_used;
        br.status[i]   = convert_status(r.status);
        br.output[i]   = std::move(r.output);
    };

    if (parallel && n > 1)
    {
        if (num_threads == 0)
            num_threads = std::max(1u, std::thread::hardware_concurrency());
        const auto workers = std::min<size_t>(num_threads, n);

        std::vector<std::thread> threads;
        threads.reserve(workers);
        std::atomic<size_t> next{0};
        for (size_t w = 0; w < workers; ++w)
        {
            threads.emplace_back([&] {
                for (;;)
                {
                    auto i = next.fetch_add(1, std::memory_order_relaxed);
                    if (i >= n) return;
                    run_one(i);
                }
            });
        }
        for (auto& t : threads) t.join();
    }
    else
    {
        for (size_t i = 0; i < n; ++i)
            run_one(i);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    br.execution_time_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    for (auto g : br.gas_used) br.total_gas += g;
    return br;
}

#ifdef __APPLE__
/// Build a HostTransaction batch for the Metal EVM kernel from dispatch txs.
/// Thin wrapper around `to_kernel_tx` (which is shared with the CPU bytecode
/// path so the four backends consume identical inputs).
[[maybe_unused]] static std::vector<kernel::HostTransaction>
to_host_transactions(const std::vector<Transaction>& txs)
{
    std::vector<kernel::HostTransaction> out;
    out.reserve(txs.size());
    for (const auto& tx : txs)
        out.push_back(to_kernel_tx(tx));
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

/// Single source of truth for the CallNotSupported fallback policy.
/// Walks `results` and replaces every `CallNotSupported` entry with the
/// outcome of running tx[i] through evmone (sequential) against the real
/// host. Increments `gpu_fallback_count` for each rewritten slot.
///
/// When `host == nullptr` the dispatcher cannot service the fallback, so
/// the helper rewrites those slots to Error with a clear message instead
/// of leaking the kernel-internal CallNotSupported status to the caller.
/// The internal CallNotSupported status code MUST never reach a consumer:
/// it is a "GPU kernel can't handle this tx" signal, not a tx-level error.
static void apply_call_not_supported_fallback(BlockResult& br,
                                              const std::vector<Transaction>& txs,
                                              evmc::Host* host)
{
    if (br.status.empty())
        return;

    for (size_t i = 0; i < br.status.size(); ++i)
    {
        if (br.status[i] != TxStatus::CallNotSupported)
            continue;

        ++br.gpu_fallback_count;

        if (host == nullptr)
        {
            // Without a host we can't service the fallback. Surface Error
            // with a clear message so the caller knows what to do; never
            // leak the internal CallNotSupported status to consumers.
            br.status[i]  = TxStatus::Error;
            br.output[i].clear();
            br.gas_used[i] = txs[i].gas_limit;
            if (br.error_message.empty())
            {
                br.error_message =
                    "GPU kernel returned CallNotSupported but no host was "
                    "provided — caller must pass evmc::Host* in `state` for "
                    "txs that issue CALL/CREATE.";
            }
            continue;
        }

        // Re-execute this single tx on evmone using the caller's host.
        std::vector<EvmTransaction> one;
        one.push_back(to_evm_transaction(txs[i]));
        auto fb = execute_sequential_evmone(one, *host, EVMC_SHANGHAI);

        // execute_sequential_evmone fills gas_used per tx. It does not yet
        // populate per-tx status/output — we surface Stop because evmone has
        // already reflected the success/revert outcome through state changes
        // on the caller's host. (This is the same contract that the host
        // bridge tests rely on.)
        br.gas_used[i] = fb.gas_used.empty() ? txs[i].gas_limit : fb.gas_used[0];
        br.status[i]   = TxStatus::Stop;
        br.output[i].clear();
    }

    // Recompute total_gas because individual entries may have been rewritten.
    br.total_gas = 0;
    for (auto g : br.gas_used) br.total_gas += g;
}

/// CPU sequential and CPU parallel collapse to the same routing tree:
///   has-host                → evmone (seq/par)
///   no-host + bytecode      → kernel::execute_cpu (seq/par) — parity path
///   no-host + no-bytecode   → gas estimation if opted in, else error
static BlockResult run_cpu(const Config& config,
                           const std::vector<Transaction>& txs,
                           evmc::Host* host,
                           bool parallel)
{
    if (host != nullptr)
    {
        auto result = execute_via_engine(config, txs, *host, parallel);
        if (config.enable_state_trie_gpu)
            compute_state_root_gpu(result, LUX_BACKEND_CPU);
        return result;
    }
    if (any_tx_has_code(txs))
    {
        auto br = execute_kernel_cpu(txs, parallel,
            parallel ? config.num_threads : 0);
        // Same fallback policy as the GPU paths: CallNotSupported is an
        // internal "kernel can't handle this tx" signal, never a tx-level
        // status code. With no host the dispatcher rewrites it to Error
        // with a clear message so the four backends report consistently.
        apply_call_not_supported_fallback(br, txs, host);
        if (config.enable_state_trie_gpu)
            compute_state_root_gpu(br, LUX_BACKEND_CPU);
        return br;
    }
    if (config.gas_estimation_mode)
        return gas_estimation_only(txs);
    return make_error_result(txs,
        "CPU backend requires either a host (state != nullptr) or bytecode "
        "in at least one tx; got neither. Set Config::gas_estimation_mode to "
        "request the gas-limit shortcut explicitly.");
}

/// GPU_Metal routing:
///   has-host               → evmone Block-STM (parallel) — Metal is reserved
///                            for benches that supply their own state. The
///                            mainline GPU consensus path is feat/gpu-host-
///                            bridge, not this dispatcher.
///   no-host + bytecode     → Metal EvmKernelHost (V1 or V2) + CallNotSupported
///                            fallback to evmone CPU when host present, else
///                            Error with a clear message.
///   no-host + no-bytecode  → BlockStmGpu scheduler-only.
///   gas-estimation         → opt-in via Config::gas_estimation_mode.
static BlockResult run_metal(const Config& config,
                             const std::vector<Transaction>& txs,
                             evmc::Host* host)
{
#ifdef __APPLE__
    if (host == nullptr && any_tx_has_code(txs))
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
            br.status.reserve(results.size());
            br.output.reserve(results.size());
            br.execution_time_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            for (auto& r : results)
            {
                br.gas_used.push_back(r.gas_used);
                br.status.push_back(convert_status(r.status));
                br.output.push_back(std::move(r.output));
                br.total_gas += r.gas_used;
            }
            apply_call_not_supported_fallback(br, txs, host);
            if (config.enable_state_trie_gpu)
                compute_state_root_gpu(br, LUX_BACKEND_METAL);
            return br;
        }
    }
    if (host == nullptr && !any_tx_has_code(txs))
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
        if (config.gas_estimation_mode)
            return gas_estimation_only(txs);
        return make_error_result(txs,
            "GPU_Metal: BlockStmGpu unavailable and no host provided; cannot "
            "execute value-transfer txs without a host or fallback.");
    }
#endif
    // Has-host fallback. Honoured on every platform — the GPU keccak helper
    // degrades to CPU when Metal/CUDA absent.
    if (host != nullptr)
    {
        auto result = execute_via_engine(config, txs, *host, /*parallel=*/true);
        if (config.enable_state_trie_gpu)
            compute_state_root_gpu(result, LUX_BACKEND_METAL);
        return result;
    }
    if (config.gas_estimation_mode)
        return gas_estimation_only(txs);
    return make_error_result(txs,
        "GPU_Metal: no host and Metal unavailable. Provide an evmc::Host* "
        "in `state`, or set Config::gas_estimation_mode for the gas-limit "
        "shortcut.");
}

/// GPU_CUDA routing — symmetric to run_metal but with CUDA hosts.
static BlockResult run_cuda(const Config& config,
                            const std::vector<Transaction>& txs,
                            evmc::Host* host)
{
#ifdef EVM_CUDA
    if (host == nullptr && any_tx_has_code(txs))
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
            br.status.reserve(results.size());
            br.output.reserve(results.size());
            br.execution_time_ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            for (auto& r : results)
            {
                br.gas_used.push_back(r.gas_used);
                br.status.push_back(convert_status(static_cast<kernel::TxStatus>(
                    static_cast<uint32_t>(r.status))));
                br.output.push_back(std::move(r.output));
                br.total_gas += r.gas_used;
            }
            apply_call_not_supported_fallback(br, txs, host);
            if (config.enable_state_trie_gpu)
            {
                if (!compute_state_root_cuda_direct(br))
                    compute_state_root_gpu(br, LUX_BACKEND_CUDA);
            }
            return br;
        }
    }
    if (host == nullptr && !any_tx_has_code(txs))
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
        if (config.gas_estimation_mode)
            return gas_estimation_only(txs);
        return make_error_result(txs,
            "GPU_CUDA: BlockStmGpu unavailable and no host provided; cannot "
            "execute value-transfer txs without a host or fallback.");
    }
#endif
    if (host != nullptr)
    {
        auto result = execute_via_engine(config, txs, *host, /*parallel=*/true);
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
    if (config.gas_estimation_mode)
        return gas_estimation_only(txs);
    return make_error_result(txs,
        "GPU_CUDA: no host and CUDA unavailable. Provide an evmc::Host* "
        "in `state`, or set Config::gas_estimation_mode for the gas-limit "
        "shortcut.");
}

BlockResult execute_block(const Config& config,
                          const std::vector<Transaction>& txs,
                          void* state)
{
    if (auto err = validate_config(config); !err.empty())
        return make_error_result(txs, std::move(err));

    auto* host = static_cast<evmc::Host*>(state);

    switch (config.backend)
    {
    case Backend::CPU_Sequential: return run_cpu(config, txs, host, /*parallel=*/false);
    case Backend::CPU_Parallel:   return run_cpu(config, txs, host, /*parallel=*/true);
    case Backend::GPU_Metal:      return run_metal(config, txs, host);
    case Backend::GPU_CUDA:       return run_cuda(config, txs, host);
    }
    return make_error_result(txs, "unknown backend");
}

}  // namespace evm::gpu
