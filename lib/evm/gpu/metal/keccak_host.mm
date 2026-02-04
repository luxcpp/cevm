// Copyright (C) 2026, The evmone Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file keccak_host.mm
/// Objective-C++ implementation of Metal-accelerated Keccak-256 hashing.
///
/// Compile with:
///   clang++ -std=c++20 -framework Metal -framework Foundation keccak_host.mm

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "keccak_host.hpp"

#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace evm::gpu::metal
{

// -- GPU descriptor matching the Metal shader struct --------------------------

struct alignas(4) GPUHashInput
{
    uint32_t offset;
    uint32_t length;
};

static_assert(sizeof(GPUHashInput) == 8);

// -- Metal implementation -----------------------------------------------------

class KeccakHasherMetal final : public KeccakHasher
{
public:
    KeccakHasherMetal(id<MTLDevice> device,
                      id<MTLCommandQueue> queue,
                      id<MTLComputePipelineState> pipeline,
                      NSString* name)
        : device_(device)
        , queue_(queue)
        , pipeline_(pipeline)
        , device_name_str_([name UTF8String])
    {}

    ~KeccakHasherMetal() override = default;

    const char* device_name() const override { return device_name_str_.c_str(); }

    std::vector<uint8_t> batch_hash(const HashInput* inputs, size_t num_inputs) override
    {
        if (num_inputs == 0)
            return {};

        // Compute total data size and build GPU input descriptors.
        size_t total_data = 0;
        for (size_t i = 0; i < num_inputs; ++i)
            total_data += inputs[i].length;

        // Allocate Metal buffers.
        const size_t desc_size = num_inputs * sizeof(GPUHashInput);
        const size_t out_size  = num_inputs * 32;

        id<MTLBuffer> desc_buf = [device_ newBufferWithLength:desc_size
                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> data_buf = [device_ newBufferWithLength:(total_data > 0 ? total_data : 1)
                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> out_buf  = [device_ newBufferWithLength:out_size
                                          options:MTLResourceStorageModeShared];

        if (!desc_buf || !data_buf || !out_buf)
            throw std::runtime_error("Metal buffer allocation failed");

        // Fill descriptor and data buffers.
        auto* gpu_descs = static_cast<GPUHashInput*>([desc_buf contents]);
        auto* gpu_data  = static_cast<uint8_t*>([data_buf contents]);

        uint32_t offset = 0;
        for (size_t i = 0; i < num_inputs; ++i)
        {
            gpu_descs[i].offset = offset;
            gpu_descs[i].length = inputs[i].length;
            if (inputs[i].length > 0)
                std::memcpy(gpu_data + offset, inputs[i].data, inputs[i].length);
            offset += inputs[i].length;
        }

        // Encode and dispatch.
        id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:pipeline_];
        [enc setBuffer:desc_buf offset:0 atIndex:0];
        [enc setBuffer:data_buf offset:0 atIndex:1];
        [enc setBuffer:out_buf  offset:0 atIndex:2];

        // One thread per hash. Threads per threadgroup = min(pipeline max, num_inputs).
        NSUInteger tpg = pipeline_.maxTotalThreadsPerThreadgroup;
        if (tpg > num_inputs)
            tpg = num_inputs;

        MTLSize grid = MTLSizeMake(num_inputs, 1, 1);
        MTLSize group = MTLSizeMake(tpg, 1, 1);

        [enc dispatchThreads:grid threadsPerThreadgroup:group];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if ([cmd error])
        {
            NSString* desc = [[cmd error] localizedDescription];
            throw std::runtime_error(std::string("Metal command failed: ") + [desc UTF8String]);
        }

        // Copy results.
        std::vector<uint8_t> result(out_size);
        std::memcpy(result.data(), [out_buf contents], out_size);

        return result;
    }

private:
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLComputePipelineState> pipeline_;
    std::string device_name_str_;
};

// -- Factory ------------------------------------------------------------------

/// Try to load a pre-compiled .metallib next to the .metal source, falling back
/// to runtime compilation of the .metal source.
static id<MTLLibrary> load_library(id<MTLDevice> device)
{
    NSError* error = nil;

    // Try pre-compiled metallib first (same directory as this source file).
    // In a deployment scenario, the metallib would be embedded in the app bundle.
    NSBundle* bundle = [NSBundle mainBundle];
    NSString* libPath = [bundle pathForResource:@"keccak256" ofType:@"metallib"];
    if (libPath)
    {
        NSURL* url = [NSURL fileURLWithPath:libPath];
        id<MTLLibrary> lib = [device newLibraryWithURL:url error:&error];
        if (lib)
            return lib;
    }

    // Runtime compilation: find the .metal source relative to this file.
    // __FILE__ gives the compile-time path of this source.
    std::filesystem::path src(__FILE__);
    std::filesystem::path metal_path = src.parent_path() / "keccak256.metal";

    NSString* path = [NSString stringWithUTF8String:metal_path.c_str()];
    NSString* source = [NSString stringWithContentsOfFile:path
                                 encoding:NSUTF8StringEncoding
                                 error:&error];
    if (!source)
        throw std::runtime_error("Cannot read keccak256.metal: " +
                                 std::string([[error localizedDescription] UTF8String]));

    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.mathMode = MTLMathModeFast;
    opts.languageVersion = MTLLanguageVersion3_0;

    id<MTLLibrary> lib = [device newLibraryWithSource:source options:opts error:&error];
    if (!lib)
        throw std::runtime_error("Metal shader compilation failed: " +
                                 std::string([[error localizedDescription] UTF8String]));

    return lib;
}

std::unique_ptr<KeccakHasher> KeccakHasher::create()
{
    @autoreleasepool
    {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device)
            return nullptr;

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue)
            return nullptr;

        id<MTLLibrary> lib = load_library(device);

        id<MTLFunction> func = [lib newFunctionWithName:@"keccak256_batch"];
        if (!func)
            throw std::runtime_error("Kernel function 'keccak256_batch' not found in library");

        NSError* error = nil;
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:func error:&error];
        if (!pipeline)
            throw std::runtime_error("Pipeline creation failed: " +
                                     std::string([[error localizedDescription] UTF8String]));

        return std::make_unique<KeccakHasherMetal>(device, queue, pipeline, [device name]);
    }
}

// -- CPU reference implementation ---------------------------------------------

static constexpr uint64_t CPU_RC[24] = {
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

static inline uint64_t rotl(uint64_t x, int n)
{
    return (x << n) | (x >> (64 - n));
}

/// Keccak-f[1600] permutation — unrolled rho/pi using the standard lane mapping.
/// State layout: state[x + 5*y] = lane(x,y), x=0..4, y=0..4.
static void keccak_f_cpu(uint64_t st[25])
{
    for (int round = 0; round < 24; ++round)
    {
        // Theta
        uint64_t C[5];
        for (int x = 0; x < 5; ++x)
            C[x] = st[x] ^ st[x + 5] ^ st[x + 10] ^ st[x + 15] ^ st[x + 20];

        for (int x = 0; x < 5; ++x)
        {
            uint64_t d = C[(x + 4) % 5] ^ rotl(C[(x + 1) % 5], 1);
            for (int y = 0; y < 5; ++y)
                st[x + 5 * y] ^= d;
        }

        // Rho + Pi (unrolled).
        // Pi moves lane(x,y) -> position(y, 2x+3y mod 5).
        // We build B[y + 5*(2x+3y mod 5)] = rot(st[x + 5y], rho_offset(x,y)).
        // Using the precomputed destination index and rotation for each of 25 lanes.
        uint64_t t = st[1];
        // The 24-step "moving lane" sequence from the Keccak reference.
        // Each step: new_t = st[PI_LANE[i]], st[PI_LANE[i]] = rot(old_t, RHO[i])
        static constexpr int PI_LANE[24] = {
            10,  7, 11, 17, 18,  3,  5, 16,  8, 21, 24,  4,
            15, 23, 19, 13, 12,  2, 20, 14, 22,  9,  6,  1
        };
        static constexpr int RHO[24] = {
             1,  3,  6, 10, 15, 21, 28, 36, 45, 55,  2, 14,
            27, 41, 56,  8, 25, 43, 62, 18, 39, 61, 20, 44
        };
        for (int i = 0; i < 24; ++i)
        {
            uint64_t tmp = st[PI_LANE[i]];
            st[PI_LANE[i]] = rotl(t, RHO[i]);
            t = tmp;
        }

        // Chi
        for (int y = 0; y < 5; ++y)
        {
            uint64_t row[5];
            for (int x = 0; x < 5; ++x)
                row[x] = st[x + 5 * y];
            for (int x = 0; x < 5; ++x)
                st[x + 5 * y] = row[x] ^ ((~row[(x + 1) % 5]) & row[(x + 2) % 5]);
        }

        // Iota
        st[0] ^= CPU_RC[round];
    }
}

void keccak256_cpu(const uint8_t* data, size_t length, uint8_t out[32])
{
    constexpr size_t rate = 136;

    uint64_t state[25] = {};

    // Absorb full blocks.
    size_t absorbed = 0;
    while (absorbed + rate <= length)
    {
        for (size_t w = 0; w < rate / 8; ++w)
        {
            uint64_t lane = 0;
            for (size_t b = 0; b < 8; ++b)
                lane |= uint64_t(data[absorbed + w * 8 + b]) << (b * 8);
            state[w] ^= lane;
        }
        keccak_f_cpu(state);
        absorbed += rate;
    }

    // Final block with padding.
    uint8_t padded[136] = {};
    size_t remaining = length - absorbed;
    std::memcpy(padded, data + absorbed, remaining);
    padded[remaining] = 0x01;
    padded[rate - 1] |= 0x80;

    for (size_t w = 0; w < rate / 8; ++w)
    {
        uint64_t lane = 0;
        for (size_t b = 0; b < 8; ++b)
            lane |= uint64_t(padded[w * 8 + b]) << (b * 8);
        state[w] ^= lane;
    }
    keccak_f_cpu(state);

    // Squeeze: first 32 bytes.
    for (size_t w = 0; w < 4; ++w)
    {
        uint64_t lane = state[w];
        for (size_t b = 0; b < 8; ++b)
            out[w * 8 + b] = static_cast<uint8_t>(lane >> (b * 8));
    }
}

}  // namespace evm::gpu::metal
