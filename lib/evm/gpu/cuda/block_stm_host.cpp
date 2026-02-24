// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file block_stm_host.cpp
/// CUDA Runtime API host launcher for GPU Block-STM execution.
///
/// Allocates persistent device buffers, packs the input transactions, then
/// dispatches the block_stm_execute kernel. After completion, reads back
/// per-tx gas and conflict statistics into a BlockResult mirroring the
/// Metal host implementation.

#include "block_stm_host.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace evm::gpu::cuda
{

// Defined in block_stm.cu.
extern "C" cudaError_t evm_cuda_block_stm_launch(
    const void*  d_txs,
    void*        d_mv_memory,
    void*        d_sched_state,
    void*        d_tx_states,
    void*        d_read_sets,
    void*        d_write_sets,
    const void*  d_base_state,
    void*        d_results,
    uint32_t     num_txs,
    uint32_t     max_iterations,
    cudaStream_t stream);

namespace
{

inline void cuda_check(cudaError_t err, const char* what)
{
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("CUDA error in ") + what +
                                 ": " + cudaGetErrorString(err));
    }
}

}  // namespace

bool block_stm_cuda_available()
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return false;
    return count > 0;
}

class BlockStmGpuCuda final : public BlockStmGpu
{
public:
    BlockStmGpuCuda(int device, cudaStream_t stream, std::string name)
        : device_(device), stream_(stream), name_(std::move(name))
    {}

    ~BlockStmGpuCuda() override
    {
        if (d_txs_)         cudaFree(d_txs_);
        if (d_mv_memory_)   cudaFree(d_mv_memory_);
        if (d_sched_state_) cudaFree(d_sched_state_);
        if (d_tx_states_)   cudaFree(d_tx_states_);
        if (d_read_sets_)   cudaFree(d_read_sets_);
        if (d_write_sets_)  cudaFree(d_write_sets_);
        if (d_base_state_)  cudaFree(d_base_state_);
        if (d_results_)     cudaFree(d_results_);
        if (stream_)        cudaStreamDestroy(stream_);
    }

    const char* device_name() const override { return name_.c_str(); }

    BlockResult execute_block(
        std::span<const Transaction> txs,
        std::span<const GpuAccountState> base_state) override
    {
        const auto t0 = std::chrono::steady_clock::now();

        BlockResult result;
        const uint32_t num_txs = static_cast<uint32_t>(txs.size());

        if (num_txs == 0 || num_txs > MAX_TXS)
        {
            result.state_root.resize(32, 0);
            return result;
        }

        cuda_check(cudaSetDevice(device_), "cudaSetDevice");

        // -- Pack transactions into GPU layout -------------------------------
        std::vector<GpuTransaction> gpu_txs(num_txs);
        std::vector<uint8_t>        calldata_blob;

        for (uint32_t i = 0; i < num_txs; ++i)
        {
            const auto& tx = txs[i];
            auto& gt = gpu_txs[i];

            std::memset(&gt, 0, sizeof(gt));
            if (tx.from.size() >= 20) std::memcpy(gt.from, tx.from.data(), 20);
            if (tx.to.size()   >= 20) std::memcpy(gt.to,   tx.to.data(),   20);

            gt.gas_limit       = tx.gas_limit;
            gt.value           = tx.value;
            gt.nonce           = tx.nonce;
            gt.gas_price       = tx.gas_price;
            gt.calldata_offset = static_cast<uint32_t>(calldata_blob.size());
            gt.calldata_size   = static_cast<uint32_t>(tx.data.size());
            calldata_blob.insert(calldata_blob.end(), tx.data.begin(), tx.data.end());
        }

        // -- Sizing ----------------------------------------------------------
        const size_t tx_bytes        = num_txs * sizeof(GpuTransaction);
        const size_t mv_bytes        = MV_TABLE_SIZE * sizeof(GpuMvEntry);
        const size_t sched_bytes     = 4 * sizeof(uint32_t);
        const size_t txstate_bytes   = num_txs * sizeof(GpuTxState);
        const size_t readset_bytes   = num_txs * MAX_READS_PER_TX  * sizeof(GpuReadSetEntry);
        const size_t writeset_bytes  = num_txs * MAX_WRITES_PER_TX * sizeof(GpuWriteSetEntry);
        const size_t state_bytes     = base_state.size() * sizeof(GpuAccountState);
        const size_t result_bytes    = num_txs * sizeof(GpuBlockStmResult);

        // -- Allocate / grow device buffers ----------------------------------
        ensure_capacity(d_txs_,         cap_txs_,        tx_bytes);
        ensure_capacity(d_mv_memory_,   cap_mv_,         mv_bytes);
        ensure_capacity(d_sched_state_, cap_sched_,      sched_bytes);
        ensure_capacity(d_tx_states_,   cap_txstate_,    txstate_bytes);
        ensure_capacity(d_read_sets_,   cap_readset_,    readset_bytes);
        ensure_capacity(d_write_sets_,  cap_writeset_,   writeset_bytes);
        ensure_capacity(d_base_state_,  cap_base_,       state_bytes > 0 ? state_bytes : 1);
        ensure_capacity(d_results_,     cap_results_,    result_bytes);

        // -- Initialize MvMemory (all entries empty: tx_index = 0xFFFFFFFF) --
        // Build the empty entry pattern host-side, then upload once. We only
        // need tx_index = 0xFFFFFFFF; the rest may be zero.
        std::vector<GpuMvEntry> empty_mv(MV_TABLE_SIZE);
        std::memset(empty_mv.data(), 0, mv_bytes);
        for (uint32_t i = 0; i < MV_TABLE_SIZE; ++i)
            empty_mv[i].tx_index = VERSION_BASE_STATE;
        cuda_check(cudaMemcpyAsync(d_mv_memory_, empty_mv.data(), mv_bytes,
                                   cudaMemcpyHostToDevice, stream_),
                   "H2D mv_memory");

        // Scheduler / per-tx state / read+write sets / results: zero-init.
        cuda_check(cudaMemsetAsync(d_sched_state_, 0, sched_bytes,    stream_), "memset sched");
        cuda_check(cudaMemsetAsync(d_tx_states_,   0, txstate_bytes,  stream_), "memset tx_states");
        cuda_check(cudaMemsetAsync(d_read_sets_,   0, readset_bytes,  stream_), "memset read_sets");
        cuda_check(cudaMemsetAsync(d_write_sets_,  0, writeset_bytes, stream_), "memset write_sets");
        cuda_check(cudaMemsetAsync(d_results_,     0, result_bytes,   stream_), "memset results");

        // Upload txs and base_state.
        cuda_check(cudaMemcpyAsync(d_txs_, gpu_txs.data(), tx_bytes,
                                   cudaMemcpyHostToDevice, stream_),
                   "H2D txs");
        if (state_bytes > 0)
        {
            cuda_check(cudaMemcpyAsync(d_base_state_, base_state.data(), state_bytes,
                                       cudaMemcpyHostToDevice, stream_),
                       "H2D base_state");
        }

        // -- Launch kernel ---------------------------------------------------
        cuda_check(evm_cuda_block_stm_launch(
                       d_txs_, d_mv_memory_, d_sched_state_,
                       d_tx_states_, d_read_sets_, d_write_sets_,
                       d_base_state_, d_results_,
                       num_txs, /*max_iterations=*/65536, stream_),
                   "block_stm launch");

        // -- Read results back -----------------------------------------------
        std::vector<GpuBlockStmResult> gpu_results(num_txs);
        cuda_check(cudaMemcpyAsync(gpu_results.data(), d_results_, result_bytes,
                                   cudaMemcpyDeviceToHost, stream_),
                   "D2H results");
        cuda_check(cudaStreamSynchronize(stream_), "stream sync");

        result.gas_used.resize(num_txs);
        result.total_gas     = 0;
        result.conflicts     = 0;
        result.re_executions = 0;

        for (uint32_t i = 0; i < num_txs; ++i)
        {
            result.gas_used[i] = gpu_results[i].gas_used;
            result.total_gas  += gpu_results[i].gas_used;
            if (gpu_results[i].incarnation > 0)
            {
                ++result.conflicts;
                result.re_executions += gpu_results[i].incarnation;
            }
        }

        result.state_root.resize(32, 0);  // computed via state_table::compute_state_root

        const auto t1 = std::chrono::steady_clock::now();
        result.execution_time_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        return result;
    }

private:
    void ensure_capacity(void*& ptr, size_t& cap, size_t needed)
    {
        if (cap >= needed) return;
        if (ptr) cudaFree(ptr);
        cuda_check(cudaMalloc(&ptr, needed), "cudaMalloc");
        cap = needed;
    }

    int          device_ = 0;
    cudaStream_t stream_ = nullptr;
    std::string  name_;

    void*  d_txs_         = nullptr; size_t cap_txs_      = 0;
    void*  d_mv_memory_   = nullptr; size_t cap_mv_       = 0;
    void*  d_sched_state_ = nullptr; size_t cap_sched_    = 0;
    void*  d_tx_states_   = nullptr; size_t cap_txstate_  = 0;
    void*  d_read_sets_   = nullptr; size_t cap_readset_  = 0;
    void*  d_write_sets_  = nullptr; size_t cap_writeset_ = 0;
    void*  d_base_state_  = nullptr; size_t cap_base_     = 0;
    void*  d_results_     = nullptr; size_t cap_results_  = 0;
};

std::unique_ptr<BlockStmGpu> BlockStmGpu::create()
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0)
        return nullptr;

    int device = 0;
    if (cudaSetDevice(device) != cudaSuccess)
        return nullptr;

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess)
        return nullptr;

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess)
        return nullptr;

    return std::make_unique<BlockStmGpuCuda>(
        device, stream, std::string(prop.name));
}

}  // namespace evm::gpu::cuda
