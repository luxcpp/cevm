// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file tx_validate_host.mm
/// Metal host dispatch for GPU transaction validation.
/// Compiles and dispatches tx_validate.metal compute kernels.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "tx_validate_host.hpp"

#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace evm::gpu::metal
{

// -- Metal implementation -----------------------------------------------------

class TxValidatorMetal final : public TxValidator
{
public:
    TxValidatorMetal(id<MTLDevice> device,
                     id<MTLCommandQueue> queue,
                     id<MTLComputePipelineState> validate_pipeline,
                     NSString* name)
        : device_(device)
        , queue_(queue)
        , validate_pipeline_(validate_pipeline)
        , device_name_str_([name UTF8String])
    {}

    ~TxValidatorMetal() override = default;

    const char* device_name() const override { return device_name_str_.c_str(); }

    std::vector<TxValidationResult> validate(
        const TxValidateInput* txs, size_t num_txs,
        const AccountLookup* state_table, size_t table_size) override
    {
        std::vector<TxValidationResult> results(num_txs);
        if (num_txs == 0) return results;

        @autoreleasepool {
            const size_t tx_buf_size    = num_txs * sizeof(TxValidateInput);
            const size_t state_buf_size = table_size * sizeof(AccountLookup);
            const size_t flags_size     = num_txs * sizeof(uint32_t);
            const size_t errors_size    = num_txs * sizeof(uint32_t);

            id<MTLBuffer> tx_buf = [device_ newBufferWithBytes:txs
                                            length:tx_buf_size
                                            options:MTLResourceStorageModeShared];
            id<MTLBuffer> state_buf = [device_ newBufferWithBytes:state_table
                                               length:state_buf_size
                                               options:MTLResourceStorageModeShared];
            id<MTLBuffer> flags_buf = [device_ newBufferWithLength:flags_size
                                               options:MTLResourceStorageModeShared];
            id<MTLBuffer> errors_buf = [device_ newBufferWithLength:errors_size
                                                options:MTLResourceStorageModeShared];

            uint32_t num_txs_u32 = static_cast<uint32_t>(num_txs);
            id<MTLBuffer> ntx_buf = [device_ newBufferWithBytes:&num_txs_u32
                                             length:sizeof(uint32_t)
                                             options:MTLResourceStorageModeShared];
            uint32_t num_accts = static_cast<uint32_t>(table_size);
            id<MTLBuffer> nacct_buf = [device_ newBufferWithBytes:&num_accts
                                               length:sizeof(uint32_t)
                                               options:MTLResourceStorageModeShared];

            if (!tx_buf || !state_buf || !flags_buf || !errors_buf || !ntx_buf || !nacct_buf)
                throw std::runtime_error("Metal buffer allocation failed");

            std::memset([flags_buf contents], 0, flags_size);
            std::memset([errors_buf contents], 0, errors_size);

            // Encode and dispatch
            id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:validate_pipeline_];
            [enc setBuffer:tx_buf     offset:0 atIndex:0];
            [enc setBuffer:state_buf  offset:0 atIndex:1];
            [enc setBuffer:flags_buf  offset:0 atIndex:2];
            [enc setBuffer:errors_buf offset:0 atIndex:3];
            [enc setBuffer:ntx_buf    offset:0 atIndex:4];
            [enc setBuffer:nacct_buf  offset:0 atIndex:5];

            NSUInteger tpg = validate_pipeline_.maxTotalThreadsPerThreadgroup;
            if (tpg > num_txs) tpg = num_txs;

            MTLSize grid = MTLSizeMake(num_txs, 1, 1);
            MTLSize group = MTLSizeMake(tpg, 1, 1);

            [enc dispatchThreads:grid threadsPerThreadgroup:group];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];

            if ([cmd error])
            {
                NSString* desc = [[cmd error] localizedDescription];
                throw std::runtime_error(
                    std::string("Metal tx_validate failed: ") + [desc UTF8String]);
            }

            // Read results
            const auto* gpu_flags = static_cast<const uint32_t*>([flags_buf contents]);
            const auto* gpu_errors = static_cast<const uint32_t*>([errors_buf contents]);

            for (size_t i = 0; i < num_txs; i++)
            {
                results[i].valid = (gpu_flags[i] != 0);
                results[i].error_code = gpu_errors[i];
            }
        }
        return results;
    }

private:
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLComputePipelineState> validate_pipeline_;
    std::string device_name_str_;
};

// -- Factory ------------------------------------------------------------------

static id<MTLLibrary> load_tx_validate_library(id<MTLDevice> device)
{
    NSError* error = nil;

    // Runtime compilation from source.
    std::filesystem::path src(__FILE__);
    std::filesystem::path metal_path = src.parent_path() / "tx_validate.metal";

    if (!std::filesystem::exists(metal_path))
        metal_path = std::filesystem::current_path() / "lib" / "evm" / "gpu" / "metal" / "tx_validate.metal";

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

std::unique_ptr<TxValidator> TxValidator::create()
{
    @autoreleasepool
    {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return nullptr;

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) return nullptr;

        id<MTLLibrary> lib = load_tx_validate_library(device);
        if (!lib) return nullptr;

        id<MTLFunction> func = [lib newFunctionWithName:@"validate_transactions"];
        if (!func) return nullptr;

        NSError* error = nil;
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:func error:&error];
        if (!pipeline) return nullptr;

        return std::make_unique<TxValidatorMetal>(device, queue, pipeline, [device name]);
    }
}

}  // namespace evm::gpu::metal
