// Copyright (C) 2026, The evmone Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file evm_kernel_host.cpp
/// CUDA Runtime API host launcher for the GPU EVM interpreter.
///
/// Companion to evm_kernel.cu. Compiled by the host C++ compiler (NOT nvcc):
/// the kernel itself lives in evm_kernel.cu and is launched through
/// evm_cuda_evm_execute_launch() which has C linkage so it can be called
/// from a non-nvcc translation unit.

#include "evm_kernel_host.hpp"

#include <cuda_runtime.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace evm::gpu::cuda
{

// -- Implemented in evm_kernel.cu --------------------------------------------

extern "C" cudaError_t evm_cuda_evm_execute_launch(
    const void*  d_inputs,
    const void*  d_blob,
    void*        d_outputs,
    void*        d_out_data,
    void*        d_mem_pool,
    void*        d_storage_pool,
    void*        d_storage_counts,
    const void*  d_params,
    unsigned int num_txs,
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

// -- Layout sanity --------------------------------------------------------
//
// Anything misaligned here will silently miscompute uint256s on the GPU.
// uint256_host is 32 bytes (4 uint64s); TxInput is 4*4 + 8 + 3*32 = 120 bytes.

static_assert(sizeof(uint256_host)    == 32,  "uint256_host wire layout");
static_assert(sizeof(TxInput)         == 136, "TxInput wire layout");
static_assert(sizeof(TxOutput)        == 32,  "TxOutput wire layout");  // 4 + 8 + 8 + 4 + padding
static_assert(sizeof(StorageEntry)    == 64,  "StorageEntry wire layout");

// -- Implementation ----------------------------------------------------------

class EvmKernelCuda final : public EvmKernel
{
public:
    EvmKernelCuda(int device, cudaStream_t stream, std::string name)
        : device_(device), stream_(stream), name_(std::move(name))
    {}

    ~EvmKernelCuda() override
    {
        if (d_inputs_)         cudaFree(d_inputs_);
        if (d_blob_)           cudaFree(d_blob_);
        if (d_outputs_)        cudaFree(d_outputs_);
        if (d_out_data_)       cudaFree(d_out_data_);
        if (d_mem_pool_)       cudaFree(d_mem_pool_);
        if (d_storage_pool_)   cudaFree(d_storage_pool_);
        if (d_storage_counts_) cudaFree(d_storage_counts_);
        if (d_params_)         cudaFree(d_params_);
        if (stream_)           cudaStreamDestroy(stream_);
    }

    const char* device_name() const override { return name_.c_str(); }

    std::vector<TxResult> execute(std::span<const HostTransaction> txs) override
    {
        const size_t num_txs = txs.size();
        if (num_txs == 0) return {};

        cuda_check(cudaSetDevice(device_), "cudaSetDevice");

        // -- Build the contiguous code+calldata blob --------------------------
        size_t total_blob = 0;
        for (const auto& tx : txs)
        {
            total_blob += tx.code.size() + tx.calldata.size();
            // EIP-2929 warm sets share the blob: 20 bytes/addr,
            // 52 bytes/(addr,slot) pair.
            total_blob += tx.warm_addresses.size();
            total_blob += tx.warm_storage_keys.size();
        }
        if (total_blob == 0) total_blob = 1;  // CUDA can't malloc 0 bytes

        // -- Build TxInput descriptors ---------------------------------------
        std::vector<TxInput> inputs(num_txs);
        std::vector<uint8_t> blob(total_blob, 0);
        uint32_t offset = 0;

        for (size_t i = 0; i < num_txs; ++i)
        {
            const auto& tx = txs[i];

            inputs[i].code_offset = offset;
            inputs[i].code_size   = static_cast<uint32_t>(tx.code.size());
            if (!tx.code.empty())
                std::memcpy(blob.data() + offset, tx.code.data(), tx.code.size());
            offset += static_cast<uint32_t>(tx.code.size());

            inputs[i].calldata_offset = offset;
            inputs[i].calldata_size   = static_cast<uint32_t>(tx.calldata.size());
            if (!tx.calldata.empty())
                std::memcpy(blob.data() + offset, tx.calldata.data(),
                            tx.calldata.size());
            offset += static_cast<uint32_t>(tx.calldata.size());

            // EIP-2929 caller-supplied warm sets — packed into the same
            // blob, 20 bytes/addr and 52 bytes/(addr,slot) pair.
            inputs[i].warm_addr_offset = offset;
            inputs[i].warm_addr_count = static_cast<uint32_t>(tx.warm_addresses.size() / 20);
            if (!tx.warm_addresses.empty())
                std::memcpy(blob.data() + offset, tx.warm_addresses.data(), tx.warm_addresses.size());
            offset += static_cast<uint32_t>(tx.warm_addresses.size());

            inputs[i].warm_slot_offset = offset;
            inputs[i].warm_slot_count = static_cast<uint32_t>(tx.warm_storage_keys.size() / 52);
            if (!tx.warm_storage_keys.empty())
                std::memcpy(blob.data() + offset, tx.warm_storage_keys.data(), tx.warm_storage_keys.size());
            offset += static_cast<uint32_t>(tx.warm_storage_keys.size());

            inputs[i].gas_limit = tx.gas_limit;
            inputs[i].caller    = tx.caller;
            inputs[i].address   = tx.address;
            inputs[i].value     = tx.value;
        }

        // -- Allocate device buffers (reused across calls when possible) -----
        const size_t input_size    = num_txs * sizeof(TxInput);
        const size_t output_size   = num_txs * sizeof(TxOutput);
        const size_t outdata_size  = num_txs * HOST_MAX_OUTPUT_PER_TX;
        const size_t mem_size      = num_txs * HOST_MAX_MEMORY_PER_TX;
        const size_t storage_size  = num_txs * HOST_MAX_STORAGE_PER_TX
                                     * sizeof(StorageEntry);
        const size_t storage_count_size = num_txs * sizeof(uint32_t);
        const size_t params_size   = sizeof(uint32_t);

        ensure_capacity(d_inputs_,         cap_inputs_,         input_size);
        ensure_capacity(d_blob_,           cap_blob_,           total_blob);
        ensure_capacity(d_outputs_,        cap_outputs_,        output_size);
        ensure_capacity(d_out_data_,       cap_out_data_,       outdata_size);
        ensure_capacity(d_mem_pool_,       cap_mem_pool_,       mem_size);
        ensure_capacity(d_storage_pool_,   cap_storage_pool_,   storage_size);
        ensure_capacity(d_storage_counts_, cap_storage_counts_, storage_count_size);
        ensure_capacity(d_params_,         cap_params_,         params_size);

        // -- H2D copies -------------------------------------------------------
        cuda_check(cudaMemcpyAsync(d_inputs_, inputs.data(), input_size,
                                   cudaMemcpyHostToDevice, stream_),
                   "H2D inputs");
        cuda_check(cudaMemcpyAsync(d_blob_, blob.data(), total_blob,
                                   cudaMemcpyHostToDevice, stream_),
                   "H2D blob");
        cuda_check(cudaMemsetAsync(d_storage_counts_, 0, storage_count_size, stream_),
                   "memset storage_counts");
        // Kernel writes everything in d_outputs_, but zero-init for safety.
        cuda_check(cudaMemsetAsync(d_outputs_, 0, output_size, stream_),
                   "memset outputs");

        const uint32_t num_txs_u32 = static_cast<uint32_t>(num_txs);
        cuda_check(cudaMemcpyAsync(d_params_, &num_txs_u32, params_size,
                                   cudaMemcpyHostToDevice, stream_),
                   "H2D params");

        // -- Launch -----------------------------------------------------------
        cuda_check(evm_cuda_evm_execute_launch(
                       d_inputs_, d_blob_, d_outputs_, d_out_data_,
                       d_mem_pool_, d_storage_pool_, d_storage_counts_,
                       d_params_, num_txs_u32, stream_),
                   "evm_execute launch");

        // -- D2H copies + sync ------------------------------------------------
        std::vector<TxOutput> gpu_outputs(num_txs);
        std::vector<uint8_t>  gpu_outdata(outdata_size);

        cuda_check(cudaMemcpyAsync(gpu_outputs.data(), d_outputs_, output_size,
                                   cudaMemcpyDeviceToHost, stream_),
                   "D2H outputs");
        cuda_check(cudaMemcpyAsync(gpu_outdata.data(), d_out_data_, outdata_size,
                                   cudaMemcpyDeviceToHost, stream_),
                   "D2H outdata");
        cuda_check(cudaStreamSynchronize(stream_), "stream sync");

        // -- Build TxResult vector -------------------------------------------
        std::vector<TxResult> results(num_txs);
        for (size_t i = 0; i < num_txs; ++i)
        {
            const auto& go = gpu_outputs[i];
            auto& r = results[i];

            switch (go.status)
            {
            case 0:  r.status = TxStatus::Stop; break;
            case 1:  r.status = TxStatus::Return; break;
            case 2:  r.status = TxStatus::Revert; break;
            case 3:  r.status = TxStatus::OutOfGas; break;
            case 5:  r.status = TxStatus::CallNotSupported; break;
            default: r.status = TxStatus::Error; break;
            }
            r.gas_used   = go.gas_used;
            r.gas_refund = go.gas_refund;

            if (go.output_size > 0)
            {
                const uint8_t* p = gpu_outdata.data() + i * HOST_MAX_OUTPUT_PER_TX;
                const uint32_t n = (go.output_size > HOST_MAX_OUTPUT_PER_TX)
                                       ? HOST_MAX_OUTPUT_PER_TX
                                       : go.output_size;
                r.output.assign(p, p + n);
            }
        }
        return results;
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

    // Device buffers (reused across calls).
    void* d_inputs_         = nullptr; size_t cap_inputs_         = 0;
    void* d_blob_           = nullptr; size_t cap_blob_           = 0;
    void* d_outputs_        = nullptr; size_t cap_outputs_        = 0;
    void* d_out_data_       = nullptr; size_t cap_out_data_       = 0;
    void* d_mem_pool_       = nullptr; size_t cap_mem_pool_       = 0;
    void* d_storage_pool_   = nullptr; size_t cap_storage_pool_   = 0;
    void* d_storage_counts_ = nullptr; size_t cap_storage_counts_ = 0;
    void* d_params_         = nullptr; size_t cap_params_         = 0;
};

// -- Factory + availability ---------------------------------------------------

bool evm_kernel_cuda_available()
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return false;
    return count > 0;
}

std::unique_ptr<EvmKernel> EvmKernel::create()
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

    return std::make_unique<EvmKernelCuda>(
        device, stream, std::string(prop.name));
}

}  // namespace evm::gpu::cuda
