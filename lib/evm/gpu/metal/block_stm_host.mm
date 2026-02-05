// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file block_stm_host.mm
/// Objective-C++ implementation of Metal-accelerated Block-STM execution.
///
/// Compile with:
///   clang++ -std=c++20 -framework Metal -framework Foundation block_stm_host.mm

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "block_stm_host.hpp"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace evm::gpu::metal
{

// -- Metal implementation -----------------------------------------------------

class BlockStmGpuMetal final : public BlockStmGpu
{
public:
    BlockStmGpuMetal(id<MTLDevice> device,
                     id<MTLCommandQueue> queue,
                     id<MTLComputePipelineState> pipeline,
                     NSString* name)
        : device_(device)
        , queue_(queue)
        , pipeline_(pipeline)
        , device_name_str_([name UTF8String])
    {}

    ~BlockStmGpuMetal() override = default;

    const char* device_name() const override { return device_name_str_.c_str(); }

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

        // -- Pack transactions into GPU format --------------------------------
        std::vector<GpuTransaction> gpu_txs(num_txs);
        std::vector<uint8_t> calldata_blob;

        for (uint32_t i = 0; i < num_txs; i++)
        {
            const auto& tx = txs[i];
            auto& gt = gpu_txs[i];

            std::memset(&gt, 0, sizeof(gt));
            if (tx.from.size() >= 20)
                std::memcpy(gt.from, tx.from.data(), 20);
            if (tx.to.size() >= 20)
                std::memcpy(gt.to, tx.to.data(), 20);

            gt.gas_limit = tx.gas_limit;
            gt.value = tx.value;
            gt.nonce = tx.nonce;
            gt.gas_price = tx.gas_price;
            gt.calldata_offset = static_cast<uint32_t>(calldata_blob.size());
            gt.calldata_size = static_cast<uint32_t>(tx.data.size());
            calldata_blob.insert(calldata_blob.end(), tx.data.begin(), tx.data.end());
        }

        // -- Allocate Metal buffers -------------------------------------------
        const size_t tx_buf_size     = num_txs * sizeof(GpuTransaction);
        const size_t mv_buf_size     = MV_TABLE_SIZE * sizeof(GpuMvEntry);
        const size_t sched_buf_size  = 4 * sizeof(uint32_t);
        const size_t txstate_size    = num_txs * sizeof(GpuTxState);
        const size_t readset_size    = num_txs * MAX_READS_PER_TX * sizeof(GpuReadSetEntry);
        const size_t writeset_size   = num_txs * MAX_WRITES_PER_TX * sizeof(GpuWriteSetEntry);
        const size_t state_buf_size  = base_state.size() * sizeof(GpuAccountState);
        const size_t result_buf_size = num_txs * sizeof(GpuBlockStmResult);
        const size_t params_size     = sizeof(GpuBlockStmParams);

        id<MTLBuffer> tx_buf = [device_ newBufferWithBytes:gpu_txs.data()
                                        length:tx_buf_size
                                        options:MTLResourceStorageModeShared];

        // Initialize MvMemory: all entries empty (tx_index = 0xFFFFFFFF)
        id<MTLBuffer> mv_buf = [device_ newBufferWithLength:mv_buf_size
                                        options:MTLResourceStorageModeShared];
        std::memset([mv_buf contents], 0xFF, mv_buf_size);  // 0xFF sets tx_index to 0xFFFFFFFF
        // But is_estimate and other fields also get 0xFF. Fix: zero-init then mark tx_index.
        auto* mv_entries = static_cast<GpuMvEntry*>([mv_buf contents]);
        for (uint32_t i = 0; i < MV_TABLE_SIZE; i++)
        {
            std::memset(&mv_entries[i], 0, sizeof(GpuMvEntry));
            mv_entries[i].tx_index = 0xFFFFFFFF;
        }

        // Scheduler state: all zeros (execution starts at tx 0)
        id<MTLBuffer> sched_buf = [device_ newBufferWithLength:sched_buf_size
                                            options:MTLResourceStorageModeShared];
        std::memset([sched_buf contents], 0, sched_buf_size);

        // Per-tx state: all zeros
        id<MTLBuffer> txstate_buf = [device_ newBufferWithLength:txstate_size
                                             options:MTLResourceStorageModeShared];
        std::memset([txstate_buf contents], 0, txstate_size);

        // Read/write sets: zero-init
        id<MTLBuffer> readset_buf = [device_ newBufferWithLength:readset_size
                                             options:MTLResourceStorageModeShared];
        std::memset([readset_buf contents], 0, readset_size);

        id<MTLBuffer> writeset_buf = [device_ newBufferWithLength:writeset_size
                                              options:MTLResourceStorageModeShared];
        std::memset([writeset_buf contents], 0, writeset_size);

        // Base account state
        id<MTLBuffer> state_buf = [device_ newBufferWithBytes:base_state.data()
                                           length:(state_buf_size > 0 ? state_buf_size : 1)
                                           options:MTLResourceStorageModeShared];

        // Results
        id<MTLBuffer> result_buf = [device_ newBufferWithLength:result_buf_size
                                            options:MTLResourceStorageModeShared];
        std::memset([result_buf contents], 0, result_buf_size);

        // Params
        GpuBlockStmParams params{};
        params.num_txs = num_txs;
        params.max_iterations = 65536;
        id<MTLBuffer> params_buf = [device_ newBufferWithBytes:&params
                                            length:params_size
                                            options:MTLResourceStorageModeShared];

        // -- Dispatch ---------------------------------------------------------
        id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:pipeline_];
        [enc setBuffer:tx_buf       offset:0 atIndex:0];
        [enc setBuffer:mv_buf       offset:0 atIndex:1];
        [enc setBuffer:sched_buf    offset:0 atIndex:2];
        [enc setBuffer:txstate_buf  offset:0 atIndex:3];
        [enc setBuffer:readset_buf  offset:0 atIndex:4];
        [enc setBuffer:writeset_buf offset:0 atIndex:5];
        [enc setBuffer:state_buf    offset:0 atIndex:6];
        [enc setBuffer:result_buf   offset:0 atIndex:7];
        [enc setBuffer:params_buf   offset:0 atIndex:8];

        // Launch one thread per potential worker. Use at least num_txs workers.
        // More workers = more parallelism for the scheduler. Cap at pipeline max.
        NSUInteger num_workers = num_txs;
        if (num_workers < 64) num_workers = 64;  // Minimum occupancy
        NSUInteger tpg = pipeline_.maxTotalThreadsPerThreadgroup;
        if (tpg > num_workers) tpg = num_workers;

        MTLSize grid = MTLSizeMake(num_workers, 1, 1);
        MTLSize group = MTLSizeMake(tpg, 1, 1);

        [enc dispatchThreads:grid threadsPerThreadgroup:group];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if ([cmd error])
        {
            NSString* desc = [[cmd error] localizedDescription];
            throw std::runtime_error(std::string("Block-STM Metal dispatch failed: ") +
                                     [desc UTF8String]);
        }

        // -- Read results back ------------------------------------------------
        const auto* gpu_results = static_cast<const GpuBlockStmResult*>([result_buf contents]);
        const auto* gpu_txstates = static_cast<const GpuTxState*>([txstate_buf contents]);

        result.gas_used.resize(num_txs);
        result.total_gas = 0;
        result.conflicts = 0;
        result.re_executions = 0;

        for (uint32_t i = 0; i < num_txs; i++)
        {
            result.gas_used[i] = gpu_results[i].gas_used;
            result.total_gas += gpu_results[i].gas_used;
            if (gpu_results[i].incarnation > 0)
            {
                result.conflicts++;
                result.re_executions += gpu_results[i].incarnation;
            }
        }

        result.state_root.resize(32, 0);  // Placeholder; compute via GPU keccak

        const auto t1 = std::chrono::steady_clock::now();
        result.execution_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        return result;
    }

private:
    // GPU read/write set entries must match Metal structs.
    // Defined here as private since they are only used for buffer sizing.
    struct GpuReadSetEntry
    {
        uint8_t  address[20];
        uint32_t _pad;
        uint8_t  slot[32];
        uint32_t read_tx_index;
        uint32_t read_incarnation;
    };

    struct GpuWriteSetEntry
    {
        uint8_t  address[20];
        uint32_t _pad;
        uint8_t  slot[32];
        uint8_t  value[32];
    };

    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLComputePipelineState> pipeline_;
    std::string device_name_str_;
};

// -- Factory ------------------------------------------------------------------

static id<MTLLibrary> load_block_stm_library(id<MTLDevice> device)
{
    NSError* error = nil;

    // Try pre-compiled metallib first
    NSBundle* bundle = [NSBundle mainBundle];
    NSString* libPath = [bundle pathForResource:@"block_stm" ofType:@"metallib"];
    if (libPath)
    {
        NSURL* url = [NSURL fileURLWithPath:libPath];
        id<MTLLibrary> lib = [device newLibraryWithURL:url error:&error];
        if (lib)
            return lib;
    }

    // Runtime compilation from .metal source
    std::filesystem::path src(__FILE__);
    std::filesystem::path metal_path = src.parent_path() / "block_stm.metal";

    NSString* path = [NSString stringWithUTF8String:metal_path.c_str()];
    NSString* source = [NSString stringWithContentsOfFile:path
                                 encoding:NSUTF8StringEncoding
                                 error:&error];
    if (!source)
        throw std::runtime_error("Cannot read block_stm.metal: " +
                                 std::string([[error localizedDescription] UTF8String]));

    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.mathMode = MTLMathModeFast;
    opts.languageVersion = MTLLanguageVersion3_0;

    id<MTLLibrary> lib = [device newLibraryWithSource:source options:opts error:&error];
    if (!lib)
        throw std::runtime_error("Block-STM shader compilation failed: " +
                                 std::string([[error localizedDescription] UTF8String]));

    return lib;
}

std::unique_ptr<BlockStmGpu> BlockStmGpu::create()
{
    @autoreleasepool
    {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device)
            return nullptr;

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue)
            return nullptr;

        id<MTLLibrary> lib = load_block_stm_library(device);

        id<MTLFunction> func = [lib newFunctionWithName:@"block_stm_execute"];
        if (!func)
            throw std::runtime_error("Kernel function 'block_stm_execute' not found");

        NSError* error = nil;
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:func error:&error];
        if (!pipeline)
            throw std::runtime_error("Pipeline creation failed: " +
                                     std::string([[error localizedDescription] UTF8String]));

        return std::make_unique<BlockStmGpuMetal>(device, queue, pipeline, [device name]);
    }
}

}  // namespace evm::gpu::metal
