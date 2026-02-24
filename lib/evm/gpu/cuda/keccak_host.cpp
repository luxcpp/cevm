// Copyright (C) 2026, The evmone Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file keccak_host.cpp
/// CUDA Runtime API host-side launcher for Keccak-256 batch hashing.
///
/// Compiled by the host C++ compiler (NOT nvcc). The kernel itself
/// (keccak256.cu) is compiled by nvcc into a relocatable object and
/// linked in. The kernel launch is wrapped in evm_cuda_keccak256_launch,
/// declared with C linkage, so we can call it without needing the
/// triple-bracket <<<...>>> syntax which only nvcc understands.

#include "keccak_host.hpp"

#include <cuda_runtime.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace evm::gpu::cuda
{

// Device-side input descriptor (must match the layout in keccak256.cu).
struct DeviceHashInput
{
    uint32_t offset;
    uint32_t length;
};
static_assert(sizeof(DeviceHashInput) == 8);

// Implemented in keccak256.cu.
extern "C" cudaError_t evm_cuda_keccak256_launch(
    const void* d_inputs,
    const void* d_data,
    void*       d_outputs,
    uint32_t    num_inputs,
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

// =============================================================================
// CUDA hasher implementation
// =============================================================================

class KeccakHasherCuda final : public KeccakHasher
{
public:
    KeccakHasherCuda(int device_index, cudaStream_t stream, std::string name)
        : device_(device_index), stream_(stream), name_(std::move(name))
    {}

    ~KeccakHasherCuda() override
    {
        if (d_inputs_)  cudaFree(d_inputs_);
        if (d_data_)    cudaFree(d_data_);
        if (d_outputs_) cudaFree(d_outputs_);
        if (stream_)    cudaStreamDestroy(stream_);
    }

    const char* device_name() const override { return name_.c_str(); }

    std::vector<uint8_t> batch_hash(
        const HashInput* inputs, size_t num_inputs) override
    {
        if (num_inputs == 0)
            return {};

        cuda_check(cudaSetDevice(device_), "cudaSetDevice");

        // Pack inputs into a contiguous buffer + descriptor table.
        size_t total_data = 0;
        for (size_t i = 0; i < num_inputs; ++i)
            total_data += inputs[i].length;

        std::vector<DeviceHashInput> descs(num_inputs);
        std::vector<uint8_t> packed(total_data > 0 ? total_data : 1, 0);

        uint32_t offset = 0;
        for (size_t i = 0; i < num_inputs; ++i)
        {
            descs[i].offset = offset;
            descs[i].length = inputs[i].length;
            if (inputs[i].length > 0)
                std::memcpy(packed.data() + offset, inputs[i].data,
                            inputs[i].length);
            offset += inputs[i].length;
        }

        const size_t desc_bytes = num_inputs * sizeof(DeviceHashInput);
        const size_t out_bytes  = num_inputs * 32;

        ensure_capacity(d_inputs_,  cap_inputs_,  desc_bytes);
        ensure_capacity(d_data_,    cap_data_,    packed.size());
        ensure_capacity(d_outputs_, cap_outputs_, out_bytes);

        // H2D
        cuda_check(cudaMemcpyAsync(d_inputs_, descs.data(), desc_bytes,
                                   cudaMemcpyHostToDevice, stream_),
                   "H2D inputs");
        cuda_check(cudaMemcpyAsync(d_data_, packed.data(), packed.size(),
                                   cudaMemcpyHostToDevice, stream_),
                   "H2D data");

        // Launch
        cuda_check(evm_cuda_keccak256_launch(
                       d_inputs_, d_data_, d_outputs_,
                       (uint32_t)num_inputs, stream_),
                   "keccak256 launch");

        // D2H + sync
        std::vector<uint8_t> result(out_bytes);
        cuda_check(cudaMemcpyAsync(result.data(), d_outputs_, out_bytes,
                                   cudaMemcpyDeviceToHost, stream_),
                   "D2H outputs");
        cuda_check(cudaStreamSynchronize(stream_), "stream sync");

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

    int device_ = 0;
    cudaStream_t stream_ = nullptr;
    std::string name_;

    void*  d_inputs_  = nullptr;  size_t cap_inputs_  = 0;
    void*  d_data_    = nullptr;  size_t cap_data_    = 0;
    void*  d_outputs_ = nullptr;  size_t cap_outputs_ = 0;
};

// =============================================================================
// Factory
// =============================================================================

std::unique_ptr<KeccakHasher> KeccakHasher::create()
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

    return std::make_unique<KeccakHasherCuda>(
        device, stream, std::string(prop.name));
}

// =============================================================================
// CPU reference implementation (mirrors metal::keccak256_cpu byte-for-byte)
// =============================================================================

namespace
{

constexpr uint64_t CPU_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL,
};

inline uint64_t rotl_cpu(uint64_t x, int n)
{
    return (x << n) | (x >> (64 - n));
}

void keccak_f_cpu(uint64_t st[25])
{
    static constexpr int PI_LANE[24] = {
        10,  7, 11, 17, 18,  3,  5, 16,  8, 21, 24,  4,
        15, 23, 19, 13, 12,  2, 20, 14, 22,  9,  6,  1
    };
    static constexpr int RHO[24] = {
         1,  3,  6, 10, 15, 21, 28, 36, 45, 55,  2, 14,
        27, 41, 56,  8, 25, 43, 62, 18, 39, 61, 20, 44
    };

    for (int round = 0; round < 24; ++round)
    {
        uint64_t C[5];
        for (int x = 0; x < 5; ++x)
            C[x] = st[x] ^ st[x + 5] ^ st[x + 10] ^ st[x + 15] ^ st[x + 20];
        for (int x = 0; x < 5; ++x)
        {
            uint64_t d = C[(x + 4) % 5] ^ rotl_cpu(C[(x + 1) % 5], 1);
            for (int y = 0; y < 5; ++y)
                st[x + 5 * y] ^= d;
        }

        uint64_t t = st[1];
        for (int i = 0; i < 24; ++i)
        {
            uint64_t tmp = st[PI_LANE[i]];
            st[PI_LANE[i]] = rotl_cpu(t, RHO[i]);
            t = tmp;
        }

        for (int y = 0; y < 5; ++y)
        {
            uint64_t row[5];
            for (int x = 0; x < 5; ++x)
                row[x] = st[x + 5 * y];
            for (int x = 0; x < 5; ++x)
                st[x + 5 * y] =
                    row[x] ^ ((~row[(x + 1) % 5]) & row[(x + 2) % 5]);
        }

        st[0] ^= CPU_RC[round];
    }
}

}  // namespace

void keccak256_cpu(const uint8_t* data, size_t length, uint8_t out[32])
{
    constexpr size_t rate = 136;

    uint64_t state[25] = {};

    size_t absorbed = 0;
    while (absorbed + rate <= length)
    {
        for (size_t w = 0; w < rate / 8; ++w)
        {
            uint64_t lane = 0;
            for (size_t b = 0; b < 8; ++b)
                lane |= (uint64_t)data[absorbed + w * 8 + b] << (b * 8);
            state[w] ^= lane;
        }
        keccak_f_cpu(state);
        absorbed += rate;
    }

    uint8_t padded[136] = {};
    size_t remaining = length - absorbed;
    std::memcpy(padded, data + absorbed, remaining);
    padded[remaining] = 0x01;
    padded[rate - 1] |= 0x80;

    for (size_t w = 0; w < rate / 8; ++w)
    {
        uint64_t lane = 0;
        for (size_t b = 0; b < 8; ++b)
            lane |= (uint64_t)padded[w * 8 + b] << (b * 8);
        state[w] ^= lane;
    }
    keccak_f_cpu(state);

    for (size_t w = 0; w < 4; ++w)
    {
        uint64_t lane = state[w];
        for (size_t b = 0; b < 8; ++b)
            out[w * 8 + b] = static_cast<uint8_t>(lane >> (b * 8));
    }
}

}  // namespace evm::gpu::cuda
