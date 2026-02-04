// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file evm_kernel_host.mm
/// Objective-C++ implementation of the Metal EVM kernel host.
///
/// Compile with:
///   clang++ -std=c++20 -framework Metal -framework Foundation evm_kernel_host.mm

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "evm_kernel_host.hpp"

#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace evm::gpu::kernel {

// -- Metal implementation -----------------------------------------------------

class EvmKernelHostMetal final : public EvmKernelHost
{
public:
    EvmKernelHostMetal(id<MTLDevice> device,
                       id<MTLCommandQueue> queue,
                       id<MTLComputePipelineState> pipeline,
                       NSString* name)
        : device_(device)
        , queue_(queue)
        , pipeline_(pipeline)
        , device_name_str_([name UTF8String])
    {}

    ~EvmKernelHostMetal() override = default;

    const char* device_name() const override { return device_name_str_.c_str(); }

    std::vector<TxResult> execute(std::span<const HostTransaction> txs) override
    {
        if (txs.empty())
            return {};

        const size_t num_txs = txs.size();

        // -- Build the contiguous blob (code + calldata for all txs) ----------
        size_t total_blob = 0;
        for (const auto& tx : txs)
            total_blob += tx.code.size() + tx.calldata.size();
        // Ensure at least 1 byte for the blob buffer.
        if (total_blob == 0)
            total_blob = 1;

        // -- Build TxInput descriptors ----------------------------------------
        std::vector<TxInput> inputs(num_txs);
        std::vector<uint8_t> blob(total_blob, 0);
        uint32_t offset = 0;

        for (size_t i = 0; i < num_txs; ++i)
        {
            const auto& tx = txs[i];
            inputs[i].code_offset = offset;
            inputs[i].code_size = static_cast<uint32_t>(tx.code.size());
            if (!tx.code.empty())
                std::memcpy(blob.data() + offset, tx.code.data(), tx.code.size());
            offset += static_cast<uint32_t>(tx.code.size());

            inputs[i].calldata_offset = offset;
            inputs[i].calldata_size = static_cast<uint32_t>(tx.calldata.size());
            if (!tx.calldata.empty())
                std::memcpy(blob.data() + offset, tx.calldata.data(), tx.calldata.size());
            offset += static_cast<uint32_t>(tx.calldata.size());

            inputs[i].gas_limit = tx.gas_limit;
            inputs[i].caller = tx.caller;
            inputs[i].address = tx.address;
            inputs[i].value = tx.value;
        }

        // -- Allocate Metal buffers -------------------------------------------
        const size_t input_size   = num_txs * sizeof(TxInput);
        const size_t output_size  = num_txs * sizeof(TxOutput);
        const size_t outdata_size = num_txs * HOST_MAX_OUTPUT_PER_TX;
        const size_t mem_size     = num_txs * HOST_MAX_MEMORY_PER_TX;
        const size_t stor_size    = num_txs * HOST_MAX_STORAGE_PER_TX * sizeof(StorageEntry);
        const size_t stor_cnt_size = num_txs * sizeof(uint32_t);
        const size_t params_size  = sizeof(uint32_t);

        id<MTLBuffer> buf_inputs   = [device_ newBufferWithBytes:inputs.data()
                                               length:input_size
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_blob     = [device_ newBufferWithBytes:blob.data()
                                               length:total_blob
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_outputs  = [device_ newBufferWithLength:output_size
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_outdata  = [device_ newBufferWithLength:outdata_size
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_mem      = [device_ newBufferWithLength:mem_size
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_storage  = [device_ newBufferWithLength:stor_size
                                               options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_stor_cnt = [device_ newBufferWithLength:stor_cnt_size
                                               options:MTLResourceStorageModeShared];
        uint32_t num_txs_u32 = static_cast<uint32_t>(num_txs);
        id<MTLBuffer> buf_params   = [device_ newBufferWithBytes:&num_txs_u32
                                               length:params_size
                                               options:MTLResourceStorageModeShared];

        if (!buf_inputs || !buf_blob || !buf_outputs || !buf_outdata ||
            !buf_mem || !buf_storage || !buf_stor_cnt || !buf_params)
            throw std::runtime_error("Metal buffer allocation failed");

        // Zero storage counts.
        std::memset([buf_stor_cnt contents], 0, stor_cnt_size);

        // -- Encode and dispatch ----------------------------------------------
        id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:pipeline_];
        [enc setBuffer:buf_inputs   offset:0 atIndex:0];
        [enc setBuffer:buf_blob     offset:0 atIndex:1];
        [enc setBuffer:buf_outputs  offset:0 atIndex:2];
        [enc setBuffer:buf_outdata  offset:0 atIndex:3];
        [enc setBuffer:buf_mem      offset:0 atIndex:4];
        [enc setBuffer:buf_storage  offset:0 atIndex:5];
        [enc setBuffer:buf_stor_cnt offset:0 atIndex:6];
        [enc setBuffer:buf_params   offset:0 atIndex:7];

        NSUInteger tpg = pipeline_.maxTotalThreadsPerThreadgroup;
        if (tpg > num_txs)
            tpg = num_txs;

        MTLSize grid = MTLSizeMake(num_txs, 1, 1);
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

        // -- Collect results --------------------------------------------------
        const auto* gpu_outputs = static_cast<const TxOutput*>([buf_outputs contents]);
        const auto* gpu_outdata = static_cast<const uint8_t*>([buf_outdata contents]);

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
            r.gas_used = go.gas_used;

            if (go.output_size > 0)
            {
                const uint8_t* data = gpu_outdata + i * HOST_MAX_OUTPUT_PER_TX;
                r.output.assign(data, data + go.output_size);
            }
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

/// Search paths for the Metal shader source file.
/// Returns the compiled library, or nil on failure.
static id<MTLLibrary> load_evm_library(id<MTLDevice> device)
{
    NSError* error = nil;

    // Try metallib from app bundle.
    NSBundle* bundle = [NSBundle mainBundle];
    NSString* libPath = [bundle pathForResource:@"evm_kernel" ofType:@"metallib"];
    if (libPath)
    {
        NSURL* url = [NSURL fileURLWithPath:libPath];
        id<MTLLibrary> lib = [device newLibraryWithURL:url error:&error];
        if (lib) return lib;
    }

    // Search for evm_kernel.metal in several locations.
    std::filesystem::path candidates[] = {
        // Relative to this source file (works when building in-tree).
        std::filesystem::path(__FILE__).parent_path() / "evm_kernel.metal",
        // Current working directory.
        std::filesystem::current_path() / "evm_kernel.metal",
        // Relative to CWD in the typical CMake build layout.
        std::filesystem::current_path() / "lib" / "evm" / "gpu" / "kernel" / "evm_kernel.metal",
    };

    for (const auto& metal_path : candidates)
    {
        if (!std::filesystem::exists(metal_path))
            continue;

        NSString* path = [NSString stringWithUTF8String:metal_path.c_str()];
        NSString* source = [NSString stringWithContentsOfFile:path
                                     encoding:NSUTF8StringEncoding
                                     error:&error];
        if (!source)
            continue;

        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        opts.mathMode = MTLMathModeFast;
        opts.languageVersion = MTLLanguageVersion3_0;

        id<MTLLibrary> lib = [device newLibraryWithSource:source options:opts error:&error];
        if (lib)
            return lib;
    }

    return nil;  // Shader not found — return nil, caller handles gracefully.
}

std::unique_ptr<EvmKernelHost> EvmKernelHost::create()
{
    @autoreleasepool
    {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device)
            return nullptr;

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue)
            return nullptr;

        id<MTLLibrary> lib = load_evm_library(device);
        if (!lib)
            return nullptr;  // Shader not found — GPU path unavailable.

        id<MTLFunction> func = [lib newFunctionWithName:@"evm_execute"];
        if (!func)
            return nullptr;  // Kernel function not found.

        NSError* error = nil;
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:func error:&error];
        if (!pipeline)
            return nullptr;  // Pipeline creation failed.

        return std::make_unique<EvmKernelHostMetal>(device, queue, pipeline, [device name]);
    }
}

}  // namespace evm::gpu::kernel
