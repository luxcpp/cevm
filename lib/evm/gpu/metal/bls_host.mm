// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file bls_host.mm
/// Metal host dispatch for BLS12-381 batch signature verification.
/// Compiles and dispatches bls12_381.metal compute kernels.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "bls_host.hpp"

#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace evm::gpu::metal
{

// -- Metal implementation -----------------------------------------------------

class BlsVerifierMetal final : public BlsVerifier
{
public:
    BlsVerifierMetal(id<MTLDevice> device,
                     id<MTLCommandQueue> queue,
                     id<MTLComputePipelineState> pipeline,
                     NSString* name)
        : device_(device)
        , queue_(queue)
        , pipeline_(pipeline)
        , device_name_str_([name UTF8String])
    {}

    ~BlsVerifierMetal() override = default;

    const char* device_name() const override { return device_name_str_.c_str(); }

    std::vector<bool> verify_batch(
        const uint8_t* sigs,
        const uint8_t* pubkeys,
        const uint8_t* messages,
        size_t count) override
    {
        std::vector<bool> results(count, false);
        if (count == 0) return results;

        @autoreleasepool {
            // BLSSignature = 48, BLSPublicKey = 96, BLSMessage = 32
            id<MTLBuffer> sig_buf = [device_ newBufferWithBytes:sigs
                                             length:count * 48
                                             options:MTLResourceStorageModeShared];
            id<MTLBuffer> pk_buf  = [device_ newBufferWithBytes:pubkeys
                                             length:count * 96
                                             options:MTLResourceStorageModeShared];
            id<MTLBuffer> msg_buf = [device_ newBufferWithBytes:messages
                                             length:count * 32
                                             options:MTLResourceStorageModeShared];
            id<MTLBuffer> res_buf = [device_ newBufferWithLength:count * sizeof(uint32_t)
                                             options:MTLResourceStorageModeShared];
            uint32_t n = static_cast<uint32_t>(count);
            id<MTLBuffer> cnt_buf = [device_ newBufferWithBytes:&n
                                             length:sizeof(uint32_t)
                                             options:MTLResourceStorageModeShared];

            if (!sig_buf || !pk_buf || !msg_buf || !res_buf || !cnt_buf)
                throw std::runtime_error("Metal buffer allocation failed");

            std::memset([res_buf contents], 0, count * sizeof(uint32_t));

            // Encode and dispatch
            id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline_];
            [enc setBuffer:sig_buf offset:0 atIndex:0];
            [enc setBuffer:pk_buf  offset:0 atIndex:1];
            [enc setBuffer:msg_buf offset:0 atIndex:2];
            [enc setBuffer:res_buf offset:0 atIndex:3];
            [enc setBuffer:cnt_buf offset:0 atIndex:4];

            NSUInteger tpg = pipeline_.maxTotalThreadsPerThreadgroup;
            if (tpg > count) tpg = count;

            MTLSize grid = MTLSizeMake(count, 1, 1);
            MTLSize group = MTLSizeMake(tpg, 1, 1);

            [enc dispatchThreads:grid threadsPerThreadgroup:group];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];

            if ([cmd error])
            {
                NSString* desc = [[cmd error] localizedDescription];
                throw std::runtime_error(
                    std::string("Metal bls_verify failed: ") + [desc UTF8String]);
            }

            const auto* gpu_results = static_cast<const uint32_t*>([res_buf contents]);
            for (size_t i = 0; i < count; i++)
                results[i] = (gpu_results[i] != 0);
        }
        return results;
    }

private:
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLComputePipelineState> pipeline_;
    std::string device_name_str_;
};

// -- Factory ------------------------------------------------------------------

static id<MTLLibrary> load_bls_library(id<MTLDevice> device)
{
    NSError* error = nil;

    // Search for bls12_381.metal relative to this source file.
    std::filesystem::path src(__FILE__);
    // bls12_381.metal lives in gpu/kernels/ (the shared crypto kernel dir)
    std::filesystem::path metal_path =
        src.parent_path().parent_path().parent_path().parent_path().parent_path()
        / ".." / ".." / "gpu" / "kernels" / "bls12_381.metal";

    // Normalize
    if (std::filesystem::exists(metal_path))
        metal_path = std::filesystem::canonical(metal_path);

    // Fallback: search relative to CWD
    if (!std::filesystem::exists(metal_path))
        metal_path = std::filesystem::current_path() / "gpu" / "kernels" / "bls12_381.metal";

    if (!std::filesystem::exists(metal_path))
        return nil;

    NSString* path = [NSString stringWithUTF8String:metal_path.c_str()];
    NSString* source = [NSString stringWithContentsOfFile:path
                                 encoding:NSUTF8StringEncoding
                                 error:&error];
    if (!source) return nil;

    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.mathMode = MTLMathModeFast;
    opts.languageVersion = MTLLanguageVersion3_0;

    return [device newLibraryWithSource:source options:opts error:&error];
}

std::unique_ptr<BlsVerifier> BlsVerifier::create()
{
    @autoreleasepool
    {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return nullptr;

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) return nullptr;

        id<MTLLibrary> lib = load_bls_library(device);
        if (!lib) return nullptr;

        id<MTLFunction> func = [lib newFunctionWithName:@"bls_verify_batch"];
        if (!func) return nullptr;

        NSError* error = nil;
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:func error:&error];
        if (!pipeline) return nullptr;

        return std::make_unique<BlsVerifierMetal>(device, queue, pipeline, [device name]);
    }
}

}  // namespace evm::gpu::metal
