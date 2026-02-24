// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file evm_kernel_host.mm
/// Objective-C++ implementation of the Metal EVM kernel host.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "evm_kernel_host.hpp"

#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace evm::gpu::kernel {

class EvmKernelHostMetal final : public EvmKernelHost
{
public:
    EvmKernelHostMetal(id<MTLDevice> device,
                       id<MTLCommandQueue> queue,
                       id<MTLComputePipelineState> pipeline_v1,
                       id<MTLComputePipelineState> pipeline_v2,
                       NSString* name)
        : device_(device)
        , queue_(queue)
        , pipeline_v1_(pipeline_v1)
        , pipeline_v2_(pipeline_v2)
        , device_name_str_([name UTF8String])
    {}

    ~EvmKernelHostMetal() override = default;

    const char* device_name() const override { return device_name_str_.c_str(); }

    std::vector<TxResult> execute(std::span<const HostTransaction> txs) override
    {
        BlockContext ctx{};
        return execute_impl(txs, ctx, pipeline_v1_);
    }

    std::vector<TxResult> execute(std::span<const HostTransaction> txs,
                                  const BlockContext& ctx) override
    {
        return execute_impl(txs, ctx, pipeline_v1_);
    }

    std::vector<TxResult> execute_v2(std::span<const HostTransaction> txs) override
    {
        BlockContext ctx{};
        if (!pipeline_v2_)
            return execute_impl(txs, ctx, pipeline_v1_);
        return execute_impl(txs, ctx, pipeline_v2_);
    }

    bool has_v2() const override { return pipeline_v2_ != nil; }

private:
    std::vector<TxResult> execute_impl(std::span<const HostTransaction> txs,
                                       const BlockContext& ctx,
                                       id<MTLComputePipelineState> pipeline)
    {
        if (txs.empty())
            return {};

        const size_t num_txs = txs.size();

        size_t total_blob = 0;
        for (const auto& tx : txs)
            total_blob += tx.code.size() + tx.calldata.size();
        if (total_blob == 0)
            total_blob = 1;

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

        const size_t input_size      = num_txs * sizeof(TxInput);
        const size_t output_size     = num_txs * sizeof(TxOutput);
        const size_t outdata_size    = num_txs * HOST_MAX_OUTPUT_PER_TX;
        const size_t mem_size        = num_txs * HOST_MAX_MEMORY_PER_TX;
        const size_t stor_size       = num_txs * HOST_MAX_STORAGE_PER_TX * sizeof(StorageEntry);
        const size_t stor_cnt_size   = num_txs * sizeof(uint32_t);
        const size_t params_size     = sizeof(uint32_t);
        const size_t trans_size      = num_txs * HOST_MAX_STORAGE_PER_TX * sizeof(StorageEntry);
        const size_t trans_cnt_size  = num_txs * sizeof(uint32_t);
        const size_t log_size        = num_txs * HOST_MAX_LOGS_PER_TX * sizeof(GpuLogEntry);
        const size_t log_cnt_size    = num_txs * sizeof(uint32_t);
        const size_t ctx_size        = sizeof(BlockContext);

        id<MTLBuffer> buf_inputs    = [device_ newBufferWithBytes:inputs.data()
                                                length:input_size
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_blob      = [device_ newBufferWithBytes:blob.data()
                                                length:total_blob
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_outputs   = [device_ newBufferWithLength:output_size
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_outdata   = [device_ newBufferWithLength:outdata_size
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_mem       = [device_ newBufferWithLength:mem_size
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_storage   = [device_ newBufferWithLength:stor_size
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_stor_cnt  = [device_ newBufferWithLength:stor_cnt_size
                                                options:MTLResourceStorageModeShared];
        uint32_t num_txs_u32 = static_cast<uint32_t>(num_txs);
        id<MTLBuffer> buf_params    = [device_ newBufferWithBytes:&num_txs_u32
                                                length:params_size
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_trans     = [device_ newBufferWithLength:trans_size
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_trans_cnt = [device_ newBufferWithLength:trans_cnt_size
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_logs      = [device_ newBufferWithLength:log_size
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_log_cnt   = [device_ newBufferWithLength:log_cnt_size
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_ctx       = [device_ newBufferWithBytes:&ctx
                                                length:ctx_size
                                                options:MTLResourceStorageModeShared];

        if (!buf_inputs || !buf_blob || !buf_outputs || !buf_outdata ||
            !buf_mem || !buf_storage || !buf_stor_cnt || !buf_params ||
            !buf_trans || !buf_trans_cnt || !buf_logs || !buf_log_cnt || !buf_ctx)
            throw std::runtime_error("Metal buffer allocation failed");

        std::memset([buf_stor_cnt contents],  0, stor_cnt_size);
        std::memset([buf_trans_cnt contents], 0, trans_cnt_size);
        std::memset([buf_log_cnt contents],   0, log_cnt_size);

        id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buf_inputs    offset:0 atIndex:0];
        [enc setBuffer:buf_blob      offset:0 atIndex:1];
        [enc setBuffer:buf_outputs   offset:0 atIndex:2];
        [enc setBuffer:buf_outdata   offset:0 atIndex:3];
        [enc setBuffer:buf_mem       offset:0 atIndex:4];
        [enc setBuffer:buf_storage   offset:0 atIndex:5];
        [enc setBuffer:buf_stor_cnt  offset:0 atIndex:6];
        [enc setBuffer:buf_params    offset:0 atIndex:7];
        [enc setBuffer:buf_trans     offset:0 atIndex:8];
        [enc setBuffer:buf_trans_cnt offset:0 atIndex:9];
        [enc setBuffer:buf_logs      offset:0 atIndex:10];
        [enc setBuffer:buf_log_cnt   offset:0 atIndex:11];
        [enc setBuffer:buf_ctx       offset:0 atIndex:12];

        NSUInteger tpg = pipeline.maxTotalThreadsPerThreadgroup;
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
            throw std::runtime_error(std::string("Metal command failed: ") + [desc UTF8String]);
        }

        const auto* gpu_outputs = static_cast<const TxOutput*>([buf_outputs contents]);
        const auto* gpu_outdata = static_cast<const uint8_t*>([buf_outdata contents]);
        const auto* gpu_mem     = static_cast<const uint8_t*>([buf_mem contents]);
        const auto* gpu_logs    = static_cast<const GpuLogEntry*>([buf_logs contents]);
        const auto* gpu_log_cnt = static_cast<const uint32_t*>([buf_log_cnt contents]);

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

            uint32_t lc = gpu_log_cnt[i];
            if (lc > HOST_MAX_LOGS_PER_TX) lc = HOST_MAX_LOGS_PER_TX;
            r.logs.reserve(lc);
            const GpuLogEntry* base = gpu_logs + i * HOST_MAX_LOGS_PER_TX;
            const uint8_t*     mem  = gpu_mem  + i * HOST_MAX_MEMORY_PER_TX;
            for (uint32_t k = 0; k < lc; ++k)
            {
                HostLog hl;
                hl.topics.assign(base[k].topics, base[k].topics + base[k].num_topics);
                if (base[k].data_size > 0 &&
                    base[k].data_offset + base[k].data_size <= HOST_MAX_MEMORY_PER_TX)
                {
                    hl.data.assign(mem + base[k].data_offset,
                                   mem + base[k].data_offset + base[k].data_size);
                }
                r.logs.push_back(std::move(hl));
            }
        }

        return results;
    }

    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLComputePipelineState> pipeline_v1_;
    id<MTLComputePipelineState> pipeline_v2_;
    std::string device_name_str_;
};

static id<MTLLibrary> compile_metal_source(id<MTLDevice> device, const char* filename)
{
    NSError* error = nil;

    std::filesystem::path candidates[] = {
        std::filesystem::path(__FILE__).parent_path() / filename,
        std::filesystem::current_path() / filename,
        std::filesystem::current_path() / "lib" / "evm" / "gpu" / "kernel" / filename,
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

        if (error) {
            NSString* desc = [error localizedDescription];
            fprintf(stderr, "Metal compile error for %s: %s\n",
                    metal_path.c_str(), [desc UTF8String]);
        }
    }

    return nil;
}

static id<MTLLibrary> load_evm_library(id<MTLDevice> device)
{
    NSError* error = nil;

    NSBundle* bundle = [NSBundle mainBundle];
    NSString* libPath = [bundle pathForResource:@"evm_kernel" ofType:@"metallib"];
    if (libPath)
    {
        NSURL* url = [NSURL fileURLWithPath:libPath];
        id<MTLLibrary> lib = [device newLibraryWithURL:url error:&error];
        if (lib) return lib;
    }

    return compile_metal_source(device, "evm_kernel.metal");
}

static id<MTLLibrary> load_evm_v2_library(id<MTLDevice> device)
{
    NSError* error = nil;

    NSBundle* bundle = [NSBundle mainBundle];
    NSString* libPath = [bundle pathForResource:@"evm_kernel_v2" ofType:@"metallib"];
    if (libPath)
    {
        NSURL* url = [NSURL fileURLWithPath:libPath];
        id<MTLLibrary> lib = [device newLibraryWithURL:url error:&error];
        if (lib) return lib;
    }

    return compile_metal_source(device, "evm_kernel_v2.metal");
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

        id<MTLLibrary> lib_v1 = load_evm_library(device);
        if (!lib_v1)
            return nullptr;

        id<MTLFunction> func_v1 = [lib_v1 newFunctionWithName:@"evm_execute"];
        if (!func_v1)
            return nullptr;

        NSError* error = nil;
        id<MTLComputePipelineState> pipeline_v1 =
            [device newComputePipelineStateWithFunction:func_v1 error:&error];
        if (!pipeline_v1)
            return nullptr;

        id<MTLComputePipelineState> pipeline_v2 = nil;
        id<MTLLibrary> lib_v2 = load_evm_v2_library(device);
        if (lib_v2) {
            id<MTLFunction> func_v2 = [lib_v2 newFunctionWithName:@"evm_execute_v2"];
            if (func_v2) {
                pipeline_v2 = [device newComputePipelineStateWithFunction:func_v2 error:&error];
            }
        }

        return std::make_unique<EvmKernelHostMetal>(
            device, queue, pipeline_v1, pipeline_v2, [device name]);
    }
}

}  // namespace evm::gpu::kernel
