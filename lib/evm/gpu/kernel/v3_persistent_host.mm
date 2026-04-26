// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file v3_persistent_host.mm
/// V3 wave-dispatch host driver.
///
/// One MTLCommandBuffer per host wave. Each enqueue_wave call:
///   1. Reserves a contiguous range in the cumulative inputs/results arena
///      (so counters and committed[] are monotonic across waves).
///   2. Encodes one MTLComputeCommandEncoder dispatching v3_wave_kernel
///      with N workgroups (one per tx). The kernel runs each tx through
///      exec → validate → commit in a straight line.
///   3. Returns a WaveFuture that wraps the cmd buffer and signals
///      completion via Metal's addCompletedHandler.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "v3_persistent_host.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace evm::gpu::kernel::v3 {

namespace {

// -----------------------------------------------------------------------------
// Device-side per-tx layout. MUST match v3_persistent.metal struct V3TxInput.
// -----------------------------------------------------------------------------
struct V3TxInputDev
{
    uint32_t code_size;
    uint32_t code_offset;
    uint32_t calldata_size;
    uint32_t calldata_offset;
    uint64_t gas_limit;
    uint32_t read_set_size;
    uint32_t read_set_offset;
    uint32_t _pad0;
    uint32_t _pad1;
};

struct V3TxResultDev
{
    uint32_t status;
    uint64_t gas_used;
    uint32_t output_size;
    uint32_t _pad0;
};

static_assert(sizeof(V3TxInputDev) == 40, "V3TxInputDev layout drift");
static_assert(sizeof(V3TxResultDev) == 24, "V3TxResultDev layout drift");

// MUST match V3Control in v3_persistent.metal byte-for-byte. Mirrors the
// device-side struct used by the kernel for atomic counter writes.
struct V3ControlDev
{
    uint32_t shutdown_flag;
    uint32_t exec_alive;
    uint32_t validate_alive;
    uint32_t commit_alive;
    uint32_t exec_done;
    uint32_t validate_done;
    uint32_t commit_done;
    uint32_t _pad0;
};

static_assert(sizeof(V3ControlDev) == 32, "V3ControlDev layout drift");

// -----------------------------------------------------------------------------
// Metal source loader — same search policy as evm_kernel_host.mm.
// -----------------------------------------------------------------------------
id<MTLLibrary> compile_v3_source(id<MTLDevice> device)
{
    NSError* error = nil;
    std::filesystem::path candidates[] = {
        std::filesystem::path(__FILE__).parent_path() / "v3_persistent.metal",
        std::filesystem::current_path() / "v3_persistent.metal",
        std::filesystem::current_path() / "lib" / "evm" / "gpu" / "kernel" / "v3_persistent.metal",
    };
    for (const auto& p : candidates)
    {
        if (!std::filesystem::exists(p))
            continue;
        NSString* path = [NSString stringWithUTF8String:p.c_str()];
        NSString* source = [NSString stringWithContentsOfFile:path
                                     encoding:NSUTF8StringEncoding
                                     error:&error];
        if (!source)
            continue;
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        opts.languageVersion = MTLLanguageVersion3_0;
        id<MTLLibrary> lib = [device newLibraryWithSource:source options:opts error:&error];
        if (lib)
            return lib;
        if (error)
            std::fprintf(stderr, "v3 metal compile error for %s: %s\n",
                         p.c_str(), [[error localizedDescription] UTF8String]);
    }
    return nil;
}

// -----------------------------------------------------------------------------
// Per-runner buffer arena. Buffers grow on demand to fit the cumulative
// per-tx slot index across waves; results/committed[] therefore stay
// addressable for any past wave's WaveFuture::collect().
// -----------------------------------------------------------------------------
struct BatchArena
{
    id<MTLBuffer> inputs    = nil;
    id<MTLBuffer> results   = nil;
    id<MTLBuffer> blob      = nil;
    id<MTLBuffer> committed = nil;

    size_t inputs_cap_txs    = 0;
    size_t results_cap_txs   = 0;
    size_t blob_cap_bytes    = 0;
    size_t committed_cap_txs = 0;

    void grow(id<MTLDevice> device, size_t txs, size_t blob_bytes)
    {
        if (txs > inputs_cap_txs)
        {
            size_t cap = std::max<size_t>(64, txs * 2);
            id<MTLBuffer> next = [device newBufferWithLength:cap * sizeof(V3TxInputDev)
                                                     options:MTLResourceStorageModeShared];
            if (inputs)
                std::memcpy([next contents], [inputs contents],
                            inputs_cap_txs * sizeof(V3TxInputDev));
            inputs = next;
            inputs_cap_txs = cap;
        }
        if (txs > results_cap_txs)
        {
            size_t cap = std::max<size_t>(64, txs * 2);
            id<MTLBuffer> next = [device newBufferWithLength:cap * sizeof(V3TxResultDev)
                                                     options:MTLResourceStorageModeShared];
            std::memset([next contents], 0, cap * sizeof(V3TxResultDev));
            if (results)
                std::memcpy([next contents], [results contents],
                            results_cap_txs * sizeof(V3TxResultDev));
            results = next;
            results_cap_txs = cap;
        }
        if (txs > committed_cap_txs)
        {
            size_t cap = std::max<size_t>(64, txs * 2);
            id<MTLBuffer> next = [device newBufferWithLength:cap * sizeof(std::uint32_t)
                                                     options:MTLResourceStorageModeShared];
            std::memset([next contents], 0, cap * sizeof(std::uint32_t));
            if (committed)
                std::memcpy([next contents], [committed contents],
                            committed_cap_txs * sizeof(std::uint32_t));
            committed = next;
            committed_cap_txs = cap;
        }
        if (blob_bytes > blob_cap_bytes)
        {
            size_t cap = std::max<size_t>(4096, blob_bytes * 2);
            id<MTLBuffer> next = [device newBufferWithLength:cap
                                                     options:MTLResourceStorageModeShared];
            if (blob)
                std::memcpy([next contents], [blob contents], blob_cap_bytes);
            blob = next;
            blob_cap_bytes = cap;
        }
    }
};

// -----------------------------------------------------------------------------
// Concrete WaveFuture. Wraps the cmd buffer; resolves on its
// addCompletedHandler.
// -----------------------------------------------------------------------------
class V3WaveFuture final : public WaveFuture
{
public:
    V3WaveFuture(id<MTLBuffer> results,
                 size_t base,
                 size_t count)
        : results_(results)
        , base_(base)
        , count_(count) {}

    /// Mark the wave complete. Called from the cmd buffer's completion
    /// handler. Safe to call exactly once.
    void mark_complete()
    {
        {
            std::lock_guard<std::mutex> g(mu_);
            completed_ = true;
        }
        cv_.notify_all();
    }

    bool ready() const override
    {
        std::lock_guard<std::mutex> g(mu_);
        return completed_;
    }

    bool exec_done() const override
    {
        // Wave model: exec finishes at the same time as commit.
        return ready();
    }

    std::vector<TxResult> await() override
    {
        {
            std::unique_lock<std::mutex> g(mu_);
            cv_.wait(g, [&]{ return completed_; });
        }
        return collect();
    }

    bool await_for(std::chrono::milliseconds timeout) override
    {
        std::unique_lock<std::mutex> g(mu_);
        return cv_.wait_for(g, timeout, [&]{ return completed_; });
    }

private:
    std::vector<TxResult> collect() const
    {
        const auto* res = static_cast<const V3TxResultDev*>([results_ contents]);
        std::vector<TxResult> out(count_);
        for (size_t i = 0; i < count_; ++i)
        {
            const auto& d = res[base_ + i];
            TxResult r;
            switch (d.status)
            {
            case 0: r.status = TxStatus::Stop;             break;
            case 1: r.status = TxStatus::Return;           break;
            case 2: r.status = TxStatus::Revert;           break;
            case 3: r.status = TxStatus::OutOfGas;         break;
            case 5: r.status = TxStatus::CallNotSupported; break;
            default: r.status = TxStatus::Error;           break;
            }
            r.gas_used   = d.gas_used;
            r.gas_refund = 0;
            r.output.clear();
            out[i] = std::move(r);
        }
        return out;
    }

    id<MTLBuffer> results_;
    size_t        base_;
    size_t        count_;

    mutable std::mutex      mu_;
    std::condition_variable cv_;
    bool                    completed_ = false;
};

// -----------------------------------------------------------------------------
// V3PersistentRunner — wave-dispatch implementation.
// -----------------------------------------------------------------------------
class V3PersistentRunnerImpl final : public V3PersistentRunner
{
public:
    V3PersistentRunnerImpl(id<MTLDevice> device,
                           id<MTLCommandQueue> queue,
                           id<MTLComputePipelineState> wave_pso,
                           NSString* device_name)
        : device_(device)
        , queue_(queue)
        , wave_pso_(wave_pso)
        , device_name_str_([device_name UTF8String])
    {
        ctl_buf_ = [device_ newBufferWithLength:sizeof(V3ControlDev)
                                        options:MTLResourceStorageModeShared];
        std::memset([ctl_buf_ contents], 0, sizeof(V3ControlDev));
    }

    ~V3PersistentRunnerImpl() override
    {
        if (!shut_down_.load())
            shutdown();
    }

    const char* device_name() const override { return device_name_str_.c_str(); }

    bool is_shut_down() const override { return shut_down_.load(); }

    Counters counters() const override
    {
        const auto* c = static_cast<const V3ControlDev*>([ctl_buf_ contents]);
        Counters out;
        out.executed       = c->exec_done;
        out.validated      = c->validate_done;
        out.committed      = c->commit_done;
        out.exec_alive     = c->exec_alive;
        out.validate_alive = c->validate_alive;
        out.commit_alive   = c->commit_alive;
        return out;
    }

    std::unique_ptr<WaveFuture> enqueue_wave(
        std::span<const HostTransaction> txs,
        const BlockContext&) override
    {
        if (shut_down_.load())
            throw std::runtime_error("V3PersistentRunner: enqueue_wave after shutdown");

        if (txs.empty())
        {
            class Empty : public WaveFuture {
            public:
                std::vector<TxResult> await() override { return {}; }
                bool await_for(std::chrono::milliseconds) override { return true; }
                bool ready() const override { return true; }
                bool exec_done() const override { return true; }
            };
            return std::make_unique<Empty>();
        }

        // Serialize wave admission so the per-runner cumulative `base` is
        // monotonic and arena growth doesn't race with another encoder.
        std::lock_guard<std::mutex> g(admission_mu_);

        const size_t base  = next_tx_base_;
        const size_t count = txs.size();
        const size_t end   = base + count;

        size_t blob_extra = 0;
        for (const auto& t : txs)
            blob_extra += t.code.size() + t.calldata.size();

        const size_t blob_base = next_blob_offset_;
        const size_t blob_end  = blob_base + blob_extra;

        arena_.grow(device_, end, blob_end);

        // Populate inputs[] and blob[].
        auto* inputs = static_cast<V3TxInputDev*>([arena_.inputs contents]);
        auto* blob   = static_cast<uint8_t*>([arena_.blob contents]);
        size_t off = blob_base;
        for (size_t i = 0; i < count; ++i)
        {
            const auto& tx = txs[i];
            V3TxInputDev d{};
            d.code_offset = static_cast<uint32_t>(off);
            d.code_size   = static_cast<uint32_t>(tx.code.size());
            if (!tx.code.empty())
                std::memcpy(blob + off, tx.code.data(), tx.code.size());
            off += tx.code.size();

            d.calldata_offset = static_cast<uint32_t>(off);
            d.calldata_size   = static_cast<uint32_t>(tx.calldata.size());
            if (!tx.calldata.empty())
                std::memcpy(blob + off, tx.calldata.data(), tx.calldata.size());
            off += tx.calldata.size();

            d.gas_limit       = tx.gas_limit;
            d.read_set_size   = 0u;
            d.read_set_offset = 0u;
            inputs[base + i]  = d;
        }
        next_blob_offset_ = off;
        next_tx_base_ = end;

        // Reset committed flags for this range.
        auto* flags = static_cast<std::uint32_t*>([arena_.committed contents]);
        std::memset(flags + base, 0, count * sizeof(std::uint32_t));

        // Encode dispatch.
        auto future = std::make_shared<V3WaveFuture>(arena_.results, base, count);

        const uint32_t base_u32  = static_cast<uint32_t>(base);
        const uint32_t count_u32 = static_cast<uint32_t>(count);

        id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:wave_pso_];
        [enc setBuffer:arena_.inputs    offset:0 atIndex:0];
        [enc setBuffer:arena_.results   offset:0 atIndex:1];
        [enc setBuffer:arena_.committed offset:0 atIndex:2];
        [enc setBuffer:ctl_buf_         offset:0 atIndex:3];
        [enc setBytes:&base_u32  length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&count_u32 length:sizeof(uint32_t) atIndex:5];

        // One workgroup per tx; tid==0 of each does the work. 32 lanes per
        // group keeps the dispatch on a SIMD-aligned boundary even though
        // the kernel only uses lane 0 today — leaves room for fanning the
        // EVM interpreter across lanes when v0.31 wires it in.
        [enc dispatchThreadgroups:MTLSizeMake(count, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];

        std::weak_ptr<V3WaveFuture> weak_future = future;
        [cmd addCompletedHandler:^(id<MTLCommandBuffer>) {
            if (auto f = weak_future.lock())
                f->mark_complete();
        }];

        // Track in-flight buffers so shutdown() can join them.
        {
            std::lock_guard<std::mutex> ig(inflight_mu_);
            inflight_.push_back(cmd);
        }
        // Drop already-completed buffers from the in-flight list.
        prune_inflight();

        [cmd commit];
        return std::unique_ptr<WaveFuture>(new SharedFutureHolder(future));
    }

    std::unique_ptr<WaveFuture> enqueue_wave(
        std::span<const HostTransaction> txs) override
    {
        BlockContext ctx{};
        return enqueue_wave(txs, ctx);
    }

    void shutdown() override
    {
        bool expected = false;
        if (!shut_down_.compare_exchange_strong(expected, true))
            return;  // already shut down — idempotent

        // Wait for every in-flight cmd buffer. The completion handlers
        // mark their futures complete first, so by the time we return
        // every WaveFuture is resolved.
        std::vector<id<MTLCommandBuffer>> snapshot;
        {
            std::lock_guard<std::mutex> ig(inflight_mu_);
            snapshot = inflight_;
            inflight_.clear();
        }
        for (id<MTLCommandBuffer> c : snapshot)
            [c waitUntilCompleted];
    }

private:
    /// Wraps a shared_ptr<V3WaveFuture> so the public unique_ptr<WaveFuture>
    /// API works while the cmd-buffer completion handler holds a separate
    /// strong reference for the lifetime of the GPU work.
    class SharedFutureHolder final : public WaveFuture
    {
    public:
        explicit SharedFutureHolder(std::shared_ptr<V3WaveFuture> f)
            : f_(std::move(f)) {}

        std::vector<TxResult> await() override            { return f_->await(); }
        bool await_for(std::chrono::milliseconds t) override { return f_->await_for(t); }
        bool ready() const override                       { return f_->ready(); }
        bool exec_done() const override                   { return f_->exec_done(); }

    private:
        std::shared_ptr<V3WaveFuture> f_;
    };

    void prune_inflight()
    {
        std::lock_guard<std::mutex> ig(inflight_mu_);
        inflight_.erase(
            std::remove_if(inflight_.begin(), inflight_.end(),
                [](id<MTLCommandBuffer> c) {
                    auto s = [c status];
                    return s == MTLCommandBufferStatusCompleted
                        || s == MTLCommandBufferStatusError;
                }),
            inflight_.end());
    }

    id<MTLDevice>               device_;
    id<MTLCommandQueue>         queue_;
    id<MTLComputePipelineState> wave_pso_;
    std::string                 device_name_str_;

    id<MTLBuffer> ctl_buf_;
    BatchArena    arena_;

    std::mutex admission_mu_;
    size_t   next_tx_base_     = 0;
    size_t   next_blob_offset_ = 0;

    mutable std::mutex                       inflight_mu_;
    std::vector<id<MTLCommandBuffer>>        inflight_;

    std::atomic<bool> shut_down_{false};
};

}  // namespace

std::unique_ptr<V3PersistentRunner> V3PersistentRunner::create()
{
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return nullptr;

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) return nullptr;

        id<MTLLibrary> lib = compile_v3_source(device);
        if (!lib) return nullptr;

        id<MTLFunction> f_wave = [lib newFunctionWithName:@"v3_wave_kernel"];
        if (!f_wave) return nullptr;

        NSError* err = nil;
        id<MTLComputePipelineState> pso_wave =
            [device newComputePipelineStateWithFunction:f_wave error:&err];
        if (!pso_wave) return nullptr;

        return std::make_unique<V3PersistentRunnerImpl>(
            device, queue, pso_wave, [device name]);
    }
}

}  // namespace evm::gpu::kernel::v3
