// Copyright (C) 2026, The evmone Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file keccak256.cu
/// CUDA compute kernel + launcher for parallel Keccak-256 hashing.
///
/// One thread per hash. Mirrors metal/keccak256.metal byte-for-byte.
/// Buffer layout matches metal/keccak_host.mm:
///   - inputs:  array of (offset, length) pairs (HashInput)
///   - data:    contiguous byte buffer
///   - outputs: 32 * num_hashes bytes
///
/// Algorithm: Keccak-256 (Ethereum variant, NOT NIST SHA-3)
///   - State: 5x5 x 64-bit = 1600-bit sponge
///   - Rate:  1088 bits (136 bytes)
///   - Capacity: 512 bits
///   - Rounds: 24
///   - Padding: 0x01 || 0x00...0x00 || 0x80 (Keccak, not SHA-3's 0x06)

#include <cstdint>
#include <cuda_runtime.h>

namespace evm::gpu::cuda
{

// =============================================================================
// Round constants
// =============================================================================

__device__ static const uint64_t KECCAK_RC[24] = {
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

__device__ static const int KECCAK_PI[24] = {
    10,  7, 11, 17, 18,  3,  5, 16,  8, 21, 24,  4,
    15, 23, 19, 13, 12,  2, 20, 14, 22,  9,  6,  1
};

__device__ static const int KECCAK_RHO[24] = {
     1,  3,  6, 10, 15, 21, 28, 36, 45, 55,  2, 14,
    27, 41, 56,  8, 25, 43, 62, 18, 39, 61, 20, 44
};

// =============================================================================
// Helpers
// =============================================================================

__device__ __forceinline__ uint64_t rotl64(uint64_t x, int n)
{
    return (x << n) | (x >> (64 - n));
}

// =============================================================================
// Keccak-f[1600] permutation
// =============================================================================

__device__ void keccak_f1600(uint64_t st[25])
{
    #pragma unroll 1
    for (int round = 0; round < 24; ++round)
    {
        // Theta
        uint64_t C[5];
        #pragma unroll
        for (int x = 0; x < 5; ++x)
            C[x] = st[x] ^ st[x + 5] ^ st[x + 10] ^ st[x + 15] ^ st[x + 20];

        #pragma unroll
        for (int x = 0; x < 5; ++x)
        {
            uint64_t d = C[(x + 4) % 5] ^ rotl64(C[(x + 1) % 5], 1);
            #pragma unroll
            for (int y = 0; y < 5; ++y)
                st[x + 5 * y] ^= d;
        }

        // Rho + Pi (moving-lane sequence)
        uint64_t t = st[1];
        #pragma unroll
        for (int i = 0; i < 24; ++i)
        {
            uint64_t tmp = st[KECCAK_PI[i]];
            st[KECCAK_PI[i]] = rotl64(t, KECCAK_RHO[i]);
            t = tmp;
        }

        // Chi
        #pragma unroll
        for (int y = 0; y < 5; ++y)
        {
            uint64_t row[5];
            #pragma unroll
            for (int x = 0; x < 5; ++x)
                row[x] = st[x + 5 * y];
            #pragma unroll
            for (int x = 0; x < 5; ++x)
                st[x + 5 * y] = row[x] ^ ((~row[(x + 1) % 5]) & row[(x + 2) % 5]);
        }

        // Iota
        st[0] ^= KECCAK_RC[round];
    }
}

// =============================================================================
// Input descriptor (must match metal HashInput layout)
// =============================================================================

struct DeviceHashInput
{
    uint32_t offset;
    uint32_t length;
};
static_assert(sizeof(DeviceHashInput) == 8, "DeviceHashInput layout mismatch");

// =============================================================================
// Kernel: one thread per hash
// =============================================================================

__global__ void keccak256_batch_kernel(
    const DeviceHashInput* __restrict__ inputs,
    const uint8_t*         __restrict__ data,
    uint8_t*               __restrict__ outputs,
    uint32_t                            num_inputs)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_inputs) return;

    const uint32_t offset = inputs[tid].offset;
    const uint32_t len    = inputs[tid].length;
    constexpr uint32_t rate = 136;  // 1088 bits / 8

    uint64_t state[25] = {0};

    // Absorb full rate-sized blocks
    uint32_t absorbed = 0;
    while (absorbed + rate <= len)
    {
        #pragma unroll
        for (uint32_t w = 0; w < rate / 8; ++w)
        {
            uint64_t lane = 0;
            #pragma unroll
            for (uint32_t b = 0; b < 8; ++b)
                lane |= (uint64_t)data[offset + absorbed + w * 8 + b] << (b * 8);
            state[w] ^= lane;
        }
        keccak_f1600(state);
        absorbed += rate;
    }

    // Final block with Keccak padding (0x01...0x80, NOT SHA-3's 0x06)
    uint8_t padded[136] = {0};
    uint32_t remaining = len - absorbed;
    for (uint32_t i = 0; i < remaining; ++i)
        padded[i] = data[offset + absorbed + i];
    padded[remaining] = 0x01;
    padded[rate - 1] |= 0x80;

    #pragma unroll
    for (uint32_t w = 0; w < rate / 8; ++w)
    {
        uint64_t lane = 0;
        #pragma unroll
        for (uint32_t b = 0; b < 8; ++b)
            lane |= (uint64_t)padded[w * 8 + b] << (b * 8);
        state[w] ^= lane;
    }
    keccak_f1600(state);

    // Squeeze first 32 bytes
    uint8_t* out = outputs + tid * 32;
    #pragma unroll
    for (uint32_t w = 0; w < 4; ++w)
    {
        uint64_t lane = state[w];
        #pragma unroll
        for (uint32_t b = 0; b < 8; ++b)
            out[w * 8 + b] = (uint8_t)(lane >> (b * 8));
    }
}

// =============================================================================
// Host-callable launcher (exported with C linkage so the .cpp host can call it
// without needing nvcc's <<<...>>> syntax)
// =============================================================================

extern "C" cudaError_t evm_cuda_keccak256_launch(
    const void*  d_inputs,    // DeviceHashInput*
    const void*  d_data,      // uint8_t*
    void*        d_outputs,   // uint8_t*
    uint32_t     num_inputs,
    cudaStream_t stream)
{
    if (num_inputs == 0) return cudaSuccess;

    constexpr uint32_t threads_per_block = 128;
    const uint32_t blocks =
        (num_inputs + threads_per_block - 1) / threads_per_block;

    keccak256_batch_kernel<<<blocks, threads_per_block, 0, stream>>>(
        static_cast<const DeviceHashInput*>(d_inputs),
        static_cast<const uint8_t*>(d_data),
        static_cast<uint8_t*>(d_outputs),
        num_inputs);

    return cudaGetLastError();
}

}  // namespace evm::gpu::cuda
