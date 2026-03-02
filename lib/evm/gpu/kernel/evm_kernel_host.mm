// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file evm_kernel_host.mm
/// Objective-C++ implementation of the Metal EVM kernel host.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "evm_kernel_host.hpp"

#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
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
    // Cached MTLBuffers reused across execute() calls. Each entry holds the
    // current allocation; we grow on demand and never shrink. Eliminates
    // 13× newBufferWithBytes/Length allocations per call (~2-3ms on M1 Max).
    //
    // THREAD SAFETY: the cache is mutated under exec_mutex_. Concurrent calls
    // to execute()/execute_v2() on the same instance serialize through the
    // mutex. Different EvmKernelHost instances are independent.
    struct CachedBuf
    {
        id<MTLBuffer> buf = nil;
        size_t size = 0;          // bytes allocated in `buf`
        size_t valid = 0;         // bytes considered valid for the current call
    };
    CachedBuf cached_inputs_, cached_blob_, cached_outputs_, cached_outdata_;
    CachedBuf cached_mem_, cached_storage_, cached_stor_cnt_, cached_params_;
    CachedBuf cached_trans_, cached_trans_cnt_, cached_logs_, cached_log_cnt_, cached_ctx_;

    // Serializes access to the cached buffers and the command queue submission
    // for this instance. Different instances are independent. We hold the lock
    // for the duration of execute_impl so the GPU has exclusive use of the
    // cache for one call at a time.
    std::mutex exec_mutex_;

    /// Get or grow a cached buffer to fit `needed`. On grow, releases the old
    /// buffer (ARC) and allocates a fresh one. Returns the buffer ready for use.
    /// `valid_bytes` records how many bytes the caller will populate; bytes
    /// beyond that are zeroed BEFORE the GPU sees them so we never leak stale
    /// content from a prior call.
    id<MTLBuffer> ensure_buf(CachedBuf& slot, size_t needed, size_t valid_bytes)
    {
        if (slot.buf == nil || slot.size < needed)
        {
            // Round up to next 64KB to amortize regrow cost across small fluctuations.
            const size_t alloc = (needed + 65535) & ~size_t{65535};
            slot.buf = [device_ newBufferWithLength:alloc
                                            options:MTLResourceStorageModeShared];
            slot.size = alloc;
            // Fresh allocation is already zero-filled by Metal; no leak source.
            slot.valid = 0;
        }
        // If a previous call wrote past the active range, scrub the tail to
        // prevent the GPU (or the host on read-back) from observing stale tx
        // data from a prior, unrelated batch. Cheap: only the dirty extent.
        if (slot.valid > valid_bytes)
        {
            uint8_t* base = static_cast<uint8_t*>([slot.buf contents]);
            std::memset(base + valid_bytes, 0, slot.valid - valid_bytes);
        }
        slot.valid = valid_bytes;
        return slot.buf;
    }

    /// Variant: ensure buffer + memcpy `src` of `len` bytes into it. Bytes in
    /// [len, valid_bytes) are zeroed; bytes beyond valid_bytes from a prior
    /// call (if any) are zeroed by ensure_buf().
    id<MTLBuffer> ensure_buf_with(CachedBuf& slot, size_t needed,
                                  const void* src, size_t len)
    {
        id<MTLBuffer> buf = ensure_buf(slot, needed, needed);
        if (src != nullptr && len > 0)
            std::memcpy([buf contents], src, len);
        if (len < needed)
        {
            uint8_t* base = static_cast<uint8_t*>([buf contents]);
            std::memset(base + len, 0, needed - len);
        }
        return buf;
    }

    /// Per-tx host-side validation. Returns true if the transaction is well
    /// formed enough to submit to the kernel. Invalid tx are flagged in
    /// `invalid` and reported as TxStatus::Error in the result vector — we
    /// never feed garbage to the GPU.
    ///
    /// Bounds we enforce here (kept narrow; the kernel enforces gas/opcode
    /// semantics):
    ///   * code.size() <= MAX_CODE_PER_TX  — bytecode must fit in uint32_t
    ///     offsets and avoid pathological compile time
    ///   * calldata.size() <= MAX_CALLDATA_PER_TX
    ///   * code.size() + calldata.size() must not overflow the running offset
    ///     accumulator used for blob packing
    static constexpr uint32_t MAX_CODE_PER_TX     = 24576 * 2;   // 2× EIP-170
    static constexpr uint32_t MAX_CALLDATA_PER_TX = 1u << 24;    // 16 MiB

    static bool validate_tx(const HostTransaction& tx)
    {
        if (tx.code.size() > MAX_CODE_PER_TX)
            return false;
        if (tx.calldata.size() > MAX_CALLDATA_PER_TX)
            return false;
        return true;
    }

    std::vector<TxResult> execute_impl(std::span<const HostTransaction> txs,
                                       const BlockContext& ctx,
                                       id<MTLComputePipelineState> pipeline)
    {
        if (txs.empty())
            return {};

        // Serialize buffer cache + submission on this instance. The mutex is
        // held for the entire call (including [cmd waitUntilCompleted]) so we
        // never overlap two GPU runs on the same shared cache. Multiple
        // EvmKernelHost instances remain independent for higher concurrency.
        std::lock_guard<std::mutex> guard(exec_mutex_);

        const size_t num_txs = txs.size();

        // Boundary validation: any tx that fails host-side checks is marked
        // invalid and replaced with a zero-cost no-op (empty code) for the
        // GPU. We still have to dispatch num_txs threads because kernel
        // arrays are indexed by gid; we just give the bad ones nothing to
        // execute and overwrite their result with Error after dispatch.
        std::vector<uint8_t> invalid(num_txs, 0);
        bool any_invalid = false;
        for (size_t i = 0; i < num_txs; ++i)
        {
            if (!validate_tx(txs[i]))
            {
                invalid[i] = 1;
                any_invalid = true;
            }
        }

        size_t total_blob = 0;
        for (size_t i = 0; i < num_txs; ++i)
        {
            if (invalid[i])
                continue;
            total_blob += txs[i].code.size() + txs[i].calldata.size();
        }
        if (total_blob == 0)
            total_blob = 1;

        // Overflow guard: the running offset is uint32_t; an attacker-supplied
        // batch with total_blob > 4 GiB would wrap. We've already capped
        // per-tx sizes so this is reachable only with millions of large tx;
        // the explicit check is defensive.
        if (total_blob > std::numeric_limits<uint32_t>::max())
            throw std::runtime_error("Metal host: total tx blob exceeds 4 GiB");

        std::vector<TxInput> inputs(num_txs);
        std::vector<uint8_t> blob(total_blob, 0);
        uint32_t offset = 0;

        for (size_t i = 0; i < num_txs; ++i)
        {
            const auto& tx = txs[i];
            if (invalid[i])
            {
                // Empty code+calldata, gas_limit=0 → kernel will return OOG
                // immediately. We overwrite with TxStatus::Error post-run.
                inputs[i].code_offset = offset;
                inputs[i].code_size = 0;
                inputs[i].calldata_offset = offset;
                inputs[i].calldata_size = 0;
                inputs[i].gas_limit = 0;
                inputs[i].caller = uint256{};
                inputs[i].address = uint256{};
                inputs[i].value = uint256{};
                continue;
            }
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

        // OPTIMIZATION: reuse cached MTLBuffers across calls. Eliminates the
        // ~13 fresh allocations/call that dominate small-batch latency.
        // SECURITY: ensure_buf scrubs any tail bytes from prior larger calls
        // so a smaller follow-up batch cannot observe stale tx data via the
        // GPU (writes/reads outside its valid range) or the host (read-back
        // of count fields the kernel didn't touch this call).
        uint32_t num_txs_u32 = static_cast<uint32_t>(num_txs);

        id<MTLBuffer> buf_inputs    = ensure_buf_with(cached_inputs_, input_size, inputs.data(), input_size);
        id<MTLBuffer> buf_blob      = ensure_buf_with(cached_blob_, total_blob, blob.data(), total_blob);
        id<MTLBuffer> buf_outputs   = ensure_buf(cached_outputs_, output_size, output_size);
        id<MTLBuffer> buf_outdata   = ensure_buf(cached_outdata_, outdata_size, outdata_size);
        id<MTLBuffer> buf_mem       = ensure_buf(cached_mem_, mem_size, mem_size);
        id<MTLBuffer> buf_storage   = ensure_buf(cached_storage_, stor_size, stor_size);
        id<MTLBuffer> buf_stor_cnt  = ensure_buf(cached_stor_cnt_, stor_cnt_size, stor_cnt_size);
        id<MTLBuffer> buf_params    = ensure_buf_with(cached_params_, params_size, &num_txs_u32, params_size);
        id<MTLBuffer> buf_trans     = ensure_buf(cached_trans_, trans_size, trans_size);
        id<MTLBuffer> buf_trans_cnt = ensure_buf(cached_trans_cnt_, trans_cnt_size, trans_cnt_size);
        id<MTLBuffer> buf_logs      = ensure_buf(cached_logs_, log_size, log_size);
        id<MTLBuffer> buf_log_cnt   = ensure_buf(cached_log_cnt_, log_cnt_size, log_cnt_size);
        id<MTLBuffer> buf_ctx       = ensure_buf_with(cached_ctx_, ctx_size, &ctx, ctx_size);

        if (!buf_inputs || !buf_blob || !buf_outputs || !buf_outdata ||
            !buf_mem || !buf_storage || !buf_stor_cnt || !buf_params ||
            !buf_trans || !buf_trans_cnt || !buf_logs || !buf_log_cnt || !buf_ctx)
            throw std::runtime_error("Metal buffer allocation failed");

        // Zero counter buffers at the start of each call (we don't zero the full
        // buf_outputs / buf_outdata since the kernel writes them; zeroing _cnt is
        // cheap because they're per-tx 4-byte counters).
        std::memset([buf_stor_cnt contents],  0, stor_cnt_size);
        std::memset([buf_trans_cnt contents], 0, trans_cnt_size);
        std::memset([buf_log_cnt contents],   0, log_cnt_size);

        // SECURITY: zero the active region of mem/storage/transient/logs at the
        // start of every call. The kernel's expand_mem_range zeros up to the
        // per-tx high-water mark, but bytes beyond that mark retain prior-call
        // content. A LOG that references offsets past the high-water mark
        // (kernel bug, future opcode, or attacker-influenced data_offset)
        // would expose stale tx data from an earlier batch — non-deterministic
        // LOG output and a side channel between unrelated batches.
        //
        // Zero-on-entry makes every byte the GPU reads either fresh-allocated
        // zero (Metal newBufferWithLength is zero-initialized) or explicitly
        // written by this call's kernel.
        //
        // Cost on M1 unified memory: ~75KB × num_txs for mem (dominant). At
        // N=2048 that's 150MB memset — sub-millisecond and dwarfed by full
        // kernel runtime. Storage/transient/logs are smaller per-tx.
        std::memset([buf_mem contents],     0, mem_size);
        std::memset([buf_storage contents], 0, stor_size);
        std::memset([buf_trans contents],   0, trans_size);
        std::memset([buf_logs contents],    0, log_size);

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
            auto& r = results[i];

            // Tx that failed host-side validation: report Error and skip the
            // GPU output for this slot. The kernel ran with empty code and
            // gas_limit=0 → returned OOG, but the caller asked for an
            // explicit Error so they can distinguish "ran-out-of-gas" from
            // "host rejected the input".
            if (any_invalid && invalid[i])
            {
                r.status = TxStatus::Error;
                r.gas_used = 0;
                continue;
            }

            const auto& go = gpu_outputs[i];

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
