// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file unified_pipeline.cpp
/// Implementation of UnifiedPipeline: composes BlockStmGpu, GpuStateHasher,
/// BlsVerifier, and the post-quantum kernels into a single block-processing
/// pipeline that shares a LuxGPU handle.
///
/// Pipelining strategy on Apple Metal:
///   - All keccak/state-root/consensus work targets the same LuxGPU handle.
///   - The Metal driver multiplexes overlapping command buffers from
///     BlockStmGpu, GpuStateHasher, and BlsVerifier on the same MTLDevice.
///   - For multi-block backlogs, std::async is used to run host-side prep +
///     synchronous GPU calls for K blocks concurrently. Pipeline depth is
///     bounded by UnifiedConfig::max_concurrent_blocks to keep memory bounded.
///
/// EVM execution path selection:
///   - APPLE: Metal Block-STM if available, else kernel CPU interpreter.
///   - EVMONE_CUDA: CUDA EvmKernel if available, else kernel CPU interpreter.
///   - Otherwise: kernel CPU interpreter (the same reference interpreter the
///     Metal/CUDA kernels emulate). Gas, status, and output match byte-for-byte
///     for opcodes the CPU interpreter implements; opcodes not yet implemented
///     surface as TxStatus::Error (see feat/v0.25-cpu-interpreter-opcodes for
///     the in-flight coverage work).
///
/// Note on post-quantum signatures (ML-DSA-65, lattice round of Quasar):
/// `lux_gpu_mldsa_verify_batch` is declared in lux/gpu.h but its C entry-point
/// is not yet linked into luxgpu_core_static — only the vtable hook exists,
/// and the vtable signature does not match the public declaration. Calling
/// the public symbol therefore fails to link. Until upstream lux-gpu lands a
/// working C entry-point, the unified pipeline reports pq_verifies = 0. The
/// BlockContext::pq_signatures field is preserved in the API so callers can
/// wire ML-DSA today and a future lux-gpu patch will turn it on without
/// changing UnifiedPipeline's interface.

#include "unified_pipeline.hpp"

#include "gpu_state_hasher.hpp"
#include "kernel/evm_kernel_host.hpp"

#if defined(__APPLE__)
#include "metal/block_stm_host.hpp"
#include "metal/bls_host.hpp"
#endif

#if defined(EVMONE_CUDA)
#include "cuda/evm_kernel_host.hpp"
#endif

#include <lux/gpu.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <vector>

namespace evm::gpu::unified {

namespace {

// ---- Layout asserts: keep crash-on-mismatch loud ---------------------------

#if defined(__APPLE__)
static_assert(sizeof(metal::GpuAccountState) == 80,
              "GpuAccountState layout must match block_stm.metal");
#endif
static_assert(sizeof(LuxEcrecoverInput) == 128,
              "LuxEcrecoverInput layout from lux/gpu.h");
static_assert(sizeof(LuxEcrecoverOutput) == 32,
              "LuxEcrecoverOutput layout from lux/gpu.h");

using Clock = std::chrono::steady_clock;

inline double ms_since(const Clock::time_point& t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ---- Transaction translation ----------------------------------------------

/// Convert a dispatch-layer Transaction into the kernel CPU interpreter's
/// HostTransaction. Mirrors the converter in gpu_dispatch.cpp so all backends
/// consume the same shape.
inline kernel::HostTransaction to_kernel_tx(const Transaction& tx) {
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

inline TxStatus convert_kernel_status(kernel::TxStatus s) {
    switch (s) {
    case kernel::TxStatus::Stop:             return TxStatus::Stop;
    case kernel::TxStatus::Return:           return TxStatus::Return;
    case kernel::TxStatus::Revert:           return TxStatus::Revert;
    case kernel::TxStatus::OutOfGas:         return TxStatus::OutOfGas;
    case kernel::TxStatus::Error:            return TxStatus::Error;
    case kernel::TxStatus::CallNotSupported: return TxStatus::CallNotSupported;
    }
    return TxStatus::Error;
}

#if defined(EVMONE_CUDA)
/// Convert a dispatch-layer Transaction into the CUDA host's HostTransaction.
/// The CUDA backend uses uint256_host (different layout from kernel::uint256),
/// so this is a separate converter.
inline cuda::HostTransaction to_cuda_tx(const Transaction& tx) {
    cuda::HostTransaction h;
    h.code      = tx.code;
    h.calldata  = tx.data;
    h.gas_limit = tx.gas_limit;

    auto pack_addr = [](cuda::uint256_host& dst, const std::vector<uint8_t>& addr) {
        std::memset(&dst, 0, sizeof(dst));
        if (addr.size() >= 20)
            std::memcpy(&dst, addr.data(), 20);
    };

    pack_addr(h.caller, tx.from);
    pack_addr(h.address, tx.to);
    h.value = cuda::uint256_host(tx.value);
    return h;
}

inline TxStatus convert_cuda_status(cuda::TxStatus s) {
    switch (s) {
    case cuda::TxStatus::Stop:             return TxStatus::Stop;
    case cuda::TxStatus::Return:           return TxStatus::Return;
    case cuda::TxStatus::Revert:           return TxStatus::Revert;
    case cuda::TxStatus::OutOfGas:         return TxStatus::OutOfGas;
    case cuda::TxStatus::Error:            return TxStatus::Error;
    case cuda::TxStatus::CallNotSupported: return TxStatus::CallNotSupported;
    }
    return TxStatus::Error;
}
#endif  // EVMONE_CUDA

// ---- Concrete implementation ----------------------------------------------

class UnifiedPipelineImpl final : public UnifiedPipeline {
public:
    explicit UnifiedPipelineImpl(const UnifiedConfig& cfg) : cfg_(cfg) {}

    bool init() {
        // Single LuxGPU handle for all keccak/state-root/consensus paths.
        gpu_ = lux_gpu_create();
        if (gpu_) {
            backend_name_ = lux_gpu_backend_name(gpu_);
            LuxDeviceInfo info{};
            if (lux_gpu_device_info(gpu_, &info) == LUX_OK && info.name) {
                device_name_ = info.name;
            }
        } else {
            backend_name_ = "cpu";
            device_name_ = "cpu";
        }

#if defined(__APPLE__)
        // EVM Block-STM engine (Apple Metal only at present).
        block_stm_ = metal::BlockStmGpu::create();

        // BLS12-381 batch verifier (Apple Metal only at present).
        if (cfg_.enable_signature_verify) {
            bls_ = metal::BlsVerifier::create();
        }
#endif
        return true;
    }

    ~UnifiedPipelineImpl() override {
        if (gpu_) lux_gpu_destroy(gpu_);
    }

    PipelineResult process_block(
        std::span<const Transaction> txs,
        std::span<const AccountInfo> accounts,
        const BlockContext& ctx) override {
        return run_one(txs, accounts, ctx);
    }

    std::vector<PipelineResult> process_blocks(
        std::span<const std::span<const Transaction>> blocks,
        std::span<const AccountInfo> base_accounts,
        std::span<const BlockContext> ctxs) override {
        const size_t n = blocks.size();
        std::vector<PipelineResult> results(n);
        if (n == 0) return results;
        if (ctxs.size() != n) {
            return results;
        }

        const uint32_t depth = std::max<uint32_t>(1, cfg_.max_concurrent_blocks);

        // Window of in-flight futures. Bounded by `depth` to keep memory and
        // GPU command-buffer pressure in check.
        std::vector<std::future<PipelineResult>> in_flight;
        in_flight.reserve(depth);
        std::vector<size_t> in_flight_idx;
        in_flight_idx.reserve(depth);

        size_t next = 0;
        while (next < n || !in_flight.empty()) {
            while (next < n && in_flight.size() < depth) {
                const size_t idx = next++;
                auto fut = std::async(std::launch::async,
                    [this, blocks, base_accounts, &ctxs, idx]() {
                        return run_one(blocks[idx], base_accounts, ctxs[idx]);
                    });
                in_flight.push_back(std::move(fut));
                in_flight_idx.push_back(idx);
            }

            if (!in_flight.empty()) {
                results[in_flight_idx.front()] = in_flight.front().get();
                in_flight.erase(in_flight.begin());
                in_flight_idx.erase(in_flight_idx.begin());
            }
        }

        return results;
    }

    const char* device_name() const override { return device_name_; }
    const char* backend_name() const override { return backend_name_; }

private:
    /// Execute all stages for a single block. May be invoked concurrently from
    /// multiple threads — every stateful subsystem accessed here is protected.
    PipelineResult run_one(
        std::span<const Transaction> txs,
        std::span<const AccountInfo> accounts,
        const BlockContext& ctx) {
        const auto t_start = Clock::now();
        PipelineResult r;
        r.state_root.assign(32, 0);
        r.consensus_hash.assign(32, 0);

        // Stage 1: Signature verification (BLS + ML-DSA).
        // Done first so that on multi-block backlogs, block N+1's verify runs
        // while block N's EVM execution proceeds (when threads overlap).
        const auto t_sigs = Clock::now();
        verify_signatures(ctx, r);
        r.sig_verify_ms = ms_since(t_sigs);

        // Stage 2: EVM execution (Block-STM on GPU, CUDA EVM kernel, or kernel CPU).
        const auto t_evm = Clock::now();
        run_evm(txs, accounts, r);
        r.evm_ms = ms_since(t_evm);

        // Stage 3: State root (Keccak-256 batch on modified accounts).
        const auto t_root = Clock::now();
        if (cfg_.enable_state_root) {
            compute_state_root(accounts, r);
        }
        r.state_root_ms = ms_since(t_root);

        // Stage 4: Consensus block hash (Keccak-256 of canonical header).
        const auto t_cons = Clock::now();
        if (cfg_.enable_consensus_hash) {
            compute_consensus_hash(ctx, r);
        }
        r.consensus_ms = ms_since(t_cons);

        r.total_ms = ms_since(t_start);
        return r;
    }

    void run_evm(
        std::span<const Transaction> txs,
        std::span<const AccountInfo> accounts,
        PipelineResult& r) {
        if (txs.empty()) {
            r.evm.gas_used.clear();
            r.evm.status.clear();
            r.evm.output.clear();
            r.evm.total_gas = 0;
            r.evm.state_root.assign(32, 0);
            return;
        }

#if defined(__APPLE__)
        if (block_stm_) {
            std::vector<metal::GpuAccountState> gpu_accounts(accounts.size());
            for (size_t i = 0; i < accounts.size(); i++) {
                auto& ga = gpu_accounts[i];
                std::memset(&ga, 0, sizeof(ga));
                std::memcpy(ga.address, accounts[i].address, 20);
                ga.nonce = accounts[i].nonce;
                ga.balance = accounts[i].balance;
            }

            // BlockStmGpu owns its own Metal command queue. Serialize across
            // worker threads to keep the queue's command buffer state sane.
            std::lock_guard<std::mutex> lock(block_stm_mu_);
            r.evm = block_stm_->execute_block(txs, gpu_accounts);
            return;
        }
        run_evm_cpu(txs, r);
        return;
#elif defined(EVMONE_CUDA)
        (void)accounts;
        {
            std::lock_guard<std::mutex> lock(cuda_mu_);
            if (!cuda_kernel_init_attempted_) {
                cuda_kernel_init_attempted_ = true;
                if (cuda::evm_kernel_cuda_available()) {
                    cuda_kernel_ = cuda::EvmKernel::create();
                }
            }
            if (cuda_kernel_) {
                std::vector<cuda::HostTransaction> ctxs;
                ctxs.reserve(txs.size());
                for (const auto& tx : txs)
                    ctxs.push_back(to_cuda_tx(tx));

                auto results = cuda_kernel_->execute(ctxs);
                r.evm.gas_used.resize(results.size());
                r.evm.status.resize(results.size(), TxStatus::Error);
                r.evm.output.resize(results.size());
                r.evm.total_gas = 0;
                for (size_t i = 0; i < results.size(); ++i) {
                    r.evm.gas_used[i] = results[i].gas_used;
                    r.evm.status[i]   = convert_cuda_status(results[i].status);
                    r.evm.output[i]   = std::move(results[i].output);
                    r.evm.total_gas  += results[i].gas_used;
                }
                r.evm.state_root.assign(32, 0);
                return;
            }
        }
        run_evm_cpu(txs, r);
        return;
#else
        (void)accounts;
        run_evm_cpu(txs, r);
        return;
#endif
    }

    /// Execute every tx through the kernel CPU interpreter — the same
    /// reference interpreter the GPU kernels emulate. Gas, status, and
    /// output match the GPU backends byte-for-byte for opcodes the CPU
    /// interpreter implements; opcodes not yet implemented surface as
    /// TxStatus::Error (see feat/v0.25-cpu-interpreter-opcodes for the
    /// in-flight coverage work).
    void run_evm_cpu(std::span<const Transaction> txs, PipelineResult& r) {
        const size_t n = txs.size();
        r.evm.gas_used.resize(n);
        r.evm.status.resize(n, TxStatus::Error);
        r.evm.output.resize(n);
        r.evm.total_gas = 0;
        for (size_t i = 0; i < n; ++i) {
            auto kt = to_kernel_tx(txs[i]);
            auto kr = kernel::execute_cpu(kt);
            r.evm.gas_used[i] = kr.gas_used;
            r.evm.status[i]   = convert_kernel_status(kr.status);
            r.evm.output[i]   = std::move(kr.output);
            r.evm.total_gas  += kr.gas_used;
        }
        r.evm.state_root.assign(32, 0);
    }

    void compute_state_root(
        std::span<const AccountInfo> accounts,
        PipelineResult& r) {
        if (accounts.empty()) {
            r.state_root.assign(32, 0);
            return;
        }

        // Hash one 36-byte preimage per account: address(20) || nonce(8) ||
        // balance(8). Then reduce by hashing the concatenation. This matches
        // the GPU state hasher's two-stage tree.
        const size_t n = accounts.size();
        std::vector<uint8_t> leaves(n * 36);
        std::vector<size_t> lens(n, 36);
        for (size_t i = 0; i < n; i++) {
            uint8_t* p = leaves.data() + i * 36;
            std::memcpy(p, accounts[i].address, 20);
            uint64_t nonce_le = accounts[i].nonce;
            uint64_t balance_le = accounts[i].balance;
            std::memcpy(p + 20, &nonce_le, 8);
            std::memcpy(p + 28, &balance_le, 8);
        }

        std::vector<uint8_t> leaf_digests(n * 32);
        bool ok = false;
        if (gpu_) {
            std::lock_guard<std::mutex> lock(gpu_mu_);
            LuxError err = lux_gpu_keccak256_batch(
                gpu_, leaves.data(), leaf_digests.data(), lens.data(), n);
            ok = (err == LUX_OK);
        }
        if (!ok) {
            r.state_root.assign(32, 0);
            return;
        }

        // Reduce: hash the concatenation of leaf digests.
        size_t reduce_len = leaf_digests.size();
        std::vector<uint8_t> root(32, 0);
        if (gpu_) {
            std::lock_guard<std::mutex> lock(gpu_mu_);
            LuxError err = lux_gpu_keccak256_batch(
                gpu_, leaf_digests.data(), root.data(), &reduce_len, 1);
            if (err != LUX_OK) {
                r.state_root.assign(32, 0);
                return;
            }
        }
        r.state_root = std::move(root);
    }

    void compute_consensus_hash(
        const BlockContext& ctx,
        PipelineResult& r) {
        if (ctx.header_bytes.empty() || !gpu_) {
            r.consensus_hash.assign(32, 0);
            return;
        }
        size_t len = ctx.header_bytes.size();
        std::vector<uint8_t> out(32, 0);
        {
            std::lock_guard<std::mutex> lock(gpu_mu_);
            LuxError err = lux_gpu_keccak256_batch(
                gpu_, ctx.header_bytes.data(), out.data(), &len, 1);
            if (err != LUX_OK) {
                r.consensus_hash.assign(32, 0);
                return;
            }
        }
        r.consensus_hash = std::move(out);
    }

    void verify_signatures(const BlockContext& ctx, PipelineResult& r) {
        if (!cfg_.enable_signature_verify) return;

        // BLS12-381 batch verify (Quasar finality round 1).
#if defined(__APPLE__)
        if (bls_ && !ctx.bls_signatures.empty()) {
            const size_t count = ctx.bls_signatures.size() / 48;
            if (count > 0
                && ctx.bls_pubkeys.size() == count * 96
                && ctx.bls_messages.size() == count * 32) {
                std::lock_guard<std::mutex> lock(bls_mu_);
                auto results = bls_->verify_batch(
                    ctx.bls_signatures.data(),
                    ctx.bls_pubkeys.data(),
                    ctx.bls_messages.data(),
                    count);
                for (bool ok : results) if (ok) r.bls_verifies++;
            }
        }
#else
        (void)ctx;
#endif

        // ML-DSA-65 batch verify is not yet wired (see top-of-file note). The
        // BlockContext fields are preserved so callers can already provide
        // data; a future lux-gpu patch will activate verification here.
        (void)ctx.pq_signatures;
        (void)ctx.pq_pubkeys;
        (void)ctx.pq_messages;
    }

    UnifiedConfig cfg_{};
    LuxGPU* gpu_ = nullptr;
#if defined(__APPLE__)
    std::unique_ptr<metal::BlockStmGpu> block_stm_;
    std::unique_ptr<metal::BlsVerifier> bls_;
#endif
#if defined(EVMONE_CUDA)
    std::unique_ptr<cuda::EvmKernel> cuda_kernel_;
    bool cuda_kernel_init_attempted_ = false;
#endif

    // Subsystem mutexes — each Metal/CUDA host owns its own command queue
    // and is not safe to call from multiple worker threads concurrently.
    std::mutex gpu_mu_;
    std::mutex block_stm_mu_;
    std::mutex bls_mu_;
#if defined(EVMONE_CUDA)
    std::mutex cuda_mu_;
#endif

    const char* backend_name_ = "uninit";
    const char* device_name_ = "uninit";
};

}  // namespace

std::unique_ptr<UnifiedPipeline> UnifiedPipeline::create(const UnifiedConfig& cfg) {
    auto p = std::make_unique<UnifiedPipelineImpl>(cfg);
    if (!p->init()) {
        return nullptr;
    }
    return p;
}

}  // namespace evm::gpu::unified
