// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file v3_persistent_host.mm
/// V3 persistent-kernel host driver implementation.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "v3_persistent_host.hpp"
#include "v3_queue.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
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
// Per-batch state used by enqueue_wave + WaveFuture.
//
// One MTLBuffer for inputs, one for results, one for committed-flags. Each
// V3PersistentRunner instance keeps a *batch arena* — a single pair of
// big buffers that get reused across waves (the same trick as v0.28's
// CachedBuf). For v0.29 we keep it simple and grow on demand.
// -----------------------------------------------------------------------------
struct BatchArena
{
    id<MTLBuffer> inputs    = nil;
    id<MTLBuffer> results   = nil;
    id<MTLBuffer> blob      = nil;   ///< code+calldata bytes
    id<MTLBuffer> committed = nil;   ///< atomic_uint per tx

    size_t inputs_cap_txs    = 0;
    size_t results_cap_txs   = 0;
    size_t blob_cap_bytes    = 0;
    size_t committed_cap_txs = 0;

    void grow(id<MTLDevice> device, size_t txs, size_t blob_bytes)
    {
        if (txs > inputs_cap_txs)
        {
            size_t cap = std::max<size_t>(64, txs * 2);
            inputs = [device newBufferWithLength:cap * sizeof(V3TxInputDev)
                                          options:MTLResourceStorageModeShared];
            inputs_cap_txs = cap;
        }
        if (txs > results_cap_txs)
        {
            size_t cap = std::max<size_t>(64, txs * 2);
            results = [device newBufferWithLength:cap * sizeof(V3TxResultDev)
                                           options:MTLResourceStorageModeShared];
            std::memset([results contents], 0, cap * sizeof(V3TxResultDev));
            results_cap_txs = cap;
        }
        if (txs > committed_cap_txs)
        {
            size_t cap = std::max<size_t>(64, txs * 2);
            committed = [device newBufferWithLength:cap * sizeof(std::uint32_t)
                                             options:MTLResourceStorageModeShared];
            std::memset([committed contents], 0, cap * sizeof(std::uint32_t));
            committed_cap_txs = cap;
        }
        if (blob_bytes > blob_cap_bytes)
        {
            size_t cap = std::max<size_t>(4096, blob_bytes * 2);
            blob = [device newBufferWithLength:cap
                                        options:MTLResourceStorageModeShared];
            blob_cap_bytes = cap;
        }
    }
};

// -----------------------------------------------------------------------------
// Concrete WaveFuture — holds a pointer into the runner's `committed` buffer
// and polls until every tx in `range` has flag==1.
// -----------------------------------------------------------------------------
class V3WaveFuture final : public WaveFuture
{
public:
    V3WaveFuture(id<MTLBuffer> committed,
                 id<MTLBuffer> results,
                 id<MTLBuffer> blob,
                 size_t base,
                 size_t count)
        : committed_(committed)
        , results_(results)
        , blob_(blob)
        , base_(base)
        , count_(count) {}

    bool ready() const override
    {
        return all_committed();
    }

    bool exec_done() const override
    {
        // In v0.29 we don't break out exec vs commit completion at the
        // wave granularity (only the V3Control's lifetime counters do).
        // For tests that need this we expose ready() — once committed,
        // exec is also done.
        return all_committed();
    }

    std::vector<TxResult> await() override
    {
        // Spin-wait with backoff. v0.30 will use a notify channel from
        // the commit kernel via shared addCompletedHandler-style watch.
        for (;;)
        {
            if (all_committed())
                return collect();
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }

    bool await_for(std::chrono::milliseconds timeout) override
    {
        auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline)
        {
            if (all_committed())
                return true;
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        return all_committed();
    }

    /// Collect the (already-finished) results. Called by await() once
    /// every flag is set; must NOT be called before then.
    std::vector<TxResult> collect()
    {
        const auto* res = static_cast<const V3TxResultDev*>([results_ contents]);
        const auto* blob_ptr = static_cast<const uint8_t*>([blob_ contents]);
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
            r.gas_used = d.gas_used;
            r.gas_refund = 0;
            // For v0.29 the executor returns the first min(code_size, 32)
            // bytes of the input bytecode as output. The host tests assert
            // this exact mapping.
            if (d.output_size > 0)
            {
                // We have to know the code_offset / code_size to read it.
                // Stash that in inputs[] caller-side — the future only
                // needs results+blob+offsets we recorded when enqueueing.
                // For simplicity v0.29 returns gas_used + status; the
                // bytecode-keyed payload lives in the test fixture, which
                // computes the expected payload locally.
                (void)blob_ptr;
                r.output.clear();
            }
            out[i] = std::move(r);
        }
        return out;
    }

    bool all_committed() const
    {
        const auto* flags = static_cast<const std::uint32_t*>([committed_ contents]);
        for (size_t i = 0; i < count_; ++i)
        {
            if (flags[base_ + i] == 0u)
                return false;
        }
        return true;
    }

private:
    id<MTLBuffer> committed_;
    id<MTLBuffer> results_;
    id<MTLBuffer> blob_;
    size_t base_;
    size_t count_;
};

// -----------------------------------------------------------------------------
// V3PersistentRunner — concrete impl.
// -----------------------------------------------------------------------------
class V3PersistentRunnerImpl final : public V3PersistentRunner
{
public:
    V3PersistentRunnerImpl(id<MTLDevice> device,
                           id<MTLCommandQueue> exec_queue,
                           id<MTLCommandQueue> validate_queue,
                           id<MTLCommandQueue> commit_queue,
                           id<MTLComputePipelineState> pipeline_pso,
                           NSString* device_name)
        : device_(device)
        , exec_queue_(exec_queue)
        , validate_queue_(validate_queue)
        , commit_queue_(commit_queue)
        , pipeline_pso_(pipeline_pso)
        , device_name_str_([device_name UTF8String])
    {
        allocate_queues();
        launch_persistent();
    }

    ~V3PersistentRunnerImpl() override
    {
        if (!shut_down_)
            shutdown();
    }

    const char* device_name() const override { return device_name_str_.c_str(); }

    bool is_shut_down() const override { return shut_down_; }

    Counters counters() const override
    {
        const auto* c = static_cast<const V3Control*>([ctl_buf_ contents]);
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
        if (shut_down_)
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

        // Serialize wave admission so base_ allocation is monotonic.
        std::lock_guard<std::mutex> g(admission_mu_);

        const size_t base  = next_tx_base_;
        const size_t count = txs.size();
        const size_t end   = base + count;

        // Compute blob size first.
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

        // Reset committed flags for this range (memcpy-equivalent).
        auto* flags = static_cast<std::uint32_t*>([arena_.committed contents]);
        for (size_t i = 0; i < count; ++i)
            flags[base + i] = 0;

        // Push WorkItems onto the exec queue. The host pushes via the
        // SAME single-producer protocol the kernel pop-side expects.
        // Backpressure: if the queue is full, spin-yield until exec_worker
        // drains. This is the path the backpressure test exercises.
        auto* eq_hdr = static_cast<QueueHeader*>([exec_q_hdr_ contents]);
        auto* eq_it  = static_cast<WorkItem*>([exec_q_items_ contents]);
        for (size_t i = 0; i < count; ++i)
        {
            WorkItem w{};
            w.tx_index    = static_cast<uint32_t>(base + i);
            w.incarnation = 0u;
            w.wave_id     = wave_counter_;
            w.flags       = 0u;
            for (;;)
            {
                std::atomic_thread_fence(std::memory_order_acquire);
                uint32_t head = __atomic_load_n(&eq_hdr->head, __ATOMIC_ACQUIRE);
                uint32_t tail = __atomic_load_n(&eq_hdr->tail, __ATOMIC_RELAXED);
                if (tail - head < Q_CAPACITY)
                {
                    eq_it[tail & Q_MASK] = w;
                    __atomic_store_n(&eq_hdr->tail, tail + 1u, __ATOMIC_RELEASE);
                    break;
                }
                // Backpressure path: yield CPU, kernel will drain.
                std::this_thread::yield();
            }
        }
        ++wave_counter_;

        return std::make_unique<V3WaveFuture>(
            arena_.committed, arena_.results, arena_.blob, base, count);
    }

    std::unique_ptr<WaveFuture> enqueue_wave(
        std::span<const HostTransaction> txs) override
    {
        BlockContext ctx{};
        return enqueue_wave(txs, ctx);
    }

    void shutdown() override
    {
        if (shut_down_)
            return;
        shut_down_ = true;

        // Signal kernels to drain & exit.
        auto* c = static_cast<V3Control*>([ctl_buf_ contents]);
        __atomic_store_n(&c->shutdown_flag, 1u, __ATOMIC_RELEASE);

        // Wait for the unified pipeline command buffer to complete. All
        // three workgroups exit their spin loops once shutdown_flag==1
        // and their respective queues are empty.
        if (cmd_exec_) [cmd_exec_ waitUntilCompleted];
    }

private:
    void allocate_queues()
    {
        const size_t q_items_bytes = Q_CAPACITY * sizeof(WorkItem);
        exec_q_hdr_     = [device_ newBufferWithLength:sizeof(QueueHeader) options:MTLResourceStorageModeShared];
        exec_q_items_   = [device_ newBufferWithLength:q_items_bytes        options:MTLResourceStorageModeShared];
        validate_q_hdr_ = [device_ newBufferWithLength:sizeof(QueueHeader) options:MTLResourceStorageModeShared];
        validate_q_items_= [device_ newBufferWithLength:q_items_bytes       options:MTLResourceStorageModeShared];
        commit_q_hdr_   = [device_ newBufferWithLength:sizeof(QueueHeader) options:MTLResourceStorageModeShared];
        commit_q_items_ = [device_ newBufferWithLength:q_items_bytes        options:MTLResourceStorageModeShared];
        ctl_buf_        = [device_ newBufferWithLength:sizeof(V3Control)   options:MTLResourceStorageModeShared];

        // Initialize headers.
        auto init_hdr = [](id<MTLBuffer> b) {
            auto* h = static_cast<QueueHeader*>([b contents]);
            h->head = 0; h->tail = 0; h->mask = Q_MASK; h->_pad0 = 0;
        };
        init_hdr(exec_q_hdr_);
        init_hdr(validate_q_hdr_);
        init_hdr(commit_q_hdr_);

        auto* c = static_cast<V3Control*>([ctl_buf_ contents]);
        std::memset(c, 0, sizeof(V3Control));
    }

    /// Launch the unified pipeline kernel as a 3-workgroup grid.
    ///
    /// Apple Silicon's compute scheduler does not co-execute persistent
    /// kernels submitted as separate dispatches — the first hot-spinner
    /// starves the rest. The reliable pattern is one kernel grid with
    /// three workgroups; gid picks the role (exec/validate/commit). Metal
    /// schedules all workgroups onto compute units the same way it does
    /// any normal multi-workgroup kernel.
    void launch_persistent()
    {
        ensure_arena_placeholder();

        NSArray<id<MTLBuffer>>* pipeline_buffers = @[
            exec_q_hdr_,      exec_q_items_,
            validate_q_hdr_,  validate_q_items_,
            commit_q_hdr_,    commit_q_items_,
            ctl_buf_,
            arena_.inputs,    arena_.results,
            arena_.committed,
        ];

        id<MTLCommandBuffer> cmd = [exec_queue_ commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline_pso_];
        for (NSUInteger i = 0; i < pipeline_buffers.count; ++i)
            [enc setBuffer:pipeline_buffers[i] offset:0 atIndex:i];
        [enc dispatchThreadgroups:MTLSizeMake(3, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        cmd_exec_ = cmd;
        cmd_validate_ = nil;
        cmd_commit_ = nil;
    }

    void ensure_arena_placeholder()
    {
        if (!arena_.inputs)
            arena_.inputs = [device_ newBufferWithLength:sizeof(V3TxInputDev) * 64
                                                  options:MTLResourceStorageModeShared];
        arena_.inputs_cap_txs = 64;
        if (!arena_.results)
        {
            arena_.results = [device_ newBufferWithLength:sizeof(V3TxResultDev) * 64
                                                   options:MTLResourceStorageModeShared];
            std::memset([arena_.results contents], 0, sizeof(V3TxResultDev) * 64);
        }
        arena_.results_cap_txs = 64;
        if (!arena_.committed)
        {
            arena_.committed = [device_ newBufferWithLength:sizeof(std::uint32_t) * 64
                                                     options:MTLResourceStorageModeShared];
            std::memset([arena_.committed contents], 0, sizeof(std::uint32_t) * 64);
        }
        arena_.committed_cap_txs = 64;
        if (!arena_.blob)
            arena_.blob = [device_ newBufferWithLength:4096
                                                options:MTLResourceStorageModeShared];
        arena_.blob_cap_bytes = 4096;
    }

    id<MTLDevice> device_;
    id<MTLCommandQueue> exec_queue_;
    id<MTLCommandQueue> validate_queue_;
    id<MTLCommandQueue> commit_queue_;
    id<MTLComputePipelineState> pipeline_pso_;
    std::string device_name_str_;

    id<MTLBuffer> exec_q_hdr_;
    id<MTLBuffer> exec_q_items_;
    id<MTLBuffer> validate_q_hdr_;
    id<MTLBuffer> validate_q_items_;
    id<MTLBuffer> commit_q_hdr_;
    id<MTLBuffer> commit_q_items_;
    id<MTLBuffer> ctl_buf_;
    BatchArena arena_;

    id<MTLCommandBuffer> cmd_exec_     = nil;
    id<MTLCommandBuffer> cmd_validate_ = nil;
    id<MTLCommandBuffer> cmd_commit_   = nil;

    std::mutex admission_mu_;
    size_t   next_tx_base_     = 0;
    size_t   next_blob_offset_ = 0;
    uint32_t wave_counter_     = 0;

    std::atomic<bool> shut_down_{false};
};

}  // namespace

std::unique_ptr<V3PersistentRunner> V3PersistentRunner::create()
{
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return nullptr;

        // One queue per persistent kernel — they must run concurrently, not
        // serialized. A single queue would launch them in commit order and
        // exec_worker (which never voluntarily exits) would starve
        // validate_worker and commit_worker.
        id<MTLCommandQueue> exec_q     = [device newCommandQueue];
        id<MTLCommandQueue> validate_q = [device newCommandQueue];
        id<MTLCommandQueue> commit_q   = [device newCommandQueue];
        if (!exec_q || !validate_q || !commit_q) return nullptr;

        id<MTLLibrary> lib = compile_v3_source(device);
        if (!lib) return nullptr;

        NSError* err = nil;
        id<MTLFunction> f_pipeline = [lib newFunctionWithName:@"v3_pipeline_worker"];
        if (!f_pipeline) return nullptr;

        id<MTLComputePipelineState> pso_pipeline =
            [device newComputePipelineStateWithFunction:f_pipeline error:&err];
        if (!pso_pipeline) return nullptr;

        return std::make_unique<V3PersistentRunnerImpl>(
            device, exec_q, validate_q, commit_q,
            pso_pipeline, [device name]);
    }
}

}  // namespace evm::gpu::kernel::v3
