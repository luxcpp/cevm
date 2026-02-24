// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file tx_validate_host.cpp
/// CUDA Runtime API host launcher for transaction validation.

#include "tx_validate_host.hpp"

#include <cuda_runtime.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace evm::gpu::cuda
{

// Implemented in tx_validate.cu.
extern "C" cudaError_t evm_cuda_tx_validate_launch(
    const void*  d_txs,
    const void*  d_state,
    void*        d_valid_flags,
    void*        d_error_codes,
    uint32_t     num_txs,
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

class TxValidatorCuda final : public TxValidator
{
public:
    TxValidatorCuda(int device, cudaStream_t stream, std::string name)
        : device_(device), stream_(stream), name_(std::move(name))
    {}

    ~TxValidatorCuda() override
    {
        if (d_txs_)    cudaFree(d_txs_);
        if (d_state_)  cudaFree(d_state_);
        if (d_flags_)  cudaFree(d_flags_);
        if (d_errors_) cudaFree(d_errors_);
        if (stream_)   cudaStreamDestroy(stream_);
    }

    const char* device_name() const override { return name_.c_str(); }

    std::vector<TxValidationResult> validate(
        const TxValidateInput* txs, size_t num_txs,
        const AccountLookup* state_table, size_t table_size) override
    {
        std::vector<TxValidationResult> results(num_txs);
        if (num_txs == 0) return results;

        cuda_check(cudaSetDevice(device_), "cudaSetDevice");

        const size_t tx_bytes    = num_txs * sizeof(TxValidateInput);
        const size_t state_bytes = table_size * sizeof(AccountLookup);
        const size_t flag_bytes  = num_txs * sizeof(uint32_t);
        const size_t err_bytes   = num_txs * sizeof(uint32_t);

        ensure_capacity(d_txs_,    cap_txs_,    tx_bytes);
        ensure_capacity(d_state_,  cap_state_,  state_bytes);
        ensure_capacity(d_flags_,  cap_flags_,  flag_bytes);
        ensure_capacity(d_errors_, cap_errors_, err_bytes);

        cuda_check(cudaMemcpyAsync(d_txs_, txs, tx_bytes,
                                   cudaMemcpyHostToDevice, stream_),
                   "H2D txs");
        cuda_check(cudaMemcpyAsync(d_state_, state_table, state_bytes,
                                   cudaMemcpyHostToDevice, stream_),
                   "H2D state");
        cuda_check(cudaMemsetAsync(d_flags_, 0, flag_bytes, stream_),
                   "memset flags");
        cuda_check(cudaMemsetAsync(d_errors_, 0, err_bytes, stream_),
                   "memset errors");

        cuda_check(evm_cuda_tx_validate_launch(
                       d_txs_, d_state_, d_flags_, d_errors_,
                       (uint32_t)num_txs, stream_),
                   "tx_validate launch");

        std::vector<uint32_t> flags(num_txs);
        std::vector<uint32_t> errors(num_txs);
        cuda_check(cudaMemcpyAsync(flags.data(), d_flags_, flag_bytes,
                                   cudaMemcpyDeviceToHost, stream_),
                   "D2H flags");
        cuda_check(cudaMemcpyAsync(errors.data(), d_errors_, err_bytes,
                                   cudaMemcpyDeviceToHost, stream_),
                   "D2H errors");
        cuda_check(cudaStreamSynchronize(stream_), "stream sync");

        for (size_t i = 0; i < num_txs; ++i)
        {
            results[i].valid      = (flags[i] != 0);
            results[i].error_code = errors[i];
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

    int device_ = 0;
    cudaStream_t stream_ = nullptr;
    std::string name_;

    void*  d_txs_    = nullptr; size_t cap_txs_    = 0;
    void*  d_state_  = nullptr; size_t cap_state_  = 0;
    void*  d_flags_  = nullptr; size_t cap_flags_  = 0;
    void*  d_errors_ = nullptr; size_t cap_errors_ = 0;
};

std::unique_ptr<TxValidator> TxValidator::create()
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
    return std::make_unique<TxValidatorCuda>(device, stream, std::string(prop.name));
}

}  // namespace evm::gpu::cuda
