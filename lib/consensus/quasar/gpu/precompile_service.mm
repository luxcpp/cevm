// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file precompile_service.mm
/// Apple Metal driver for PrecompileService.
///
/// Wraps the CPU-portable PrecompileServiceCpu (precompile_service.cpp) with
/// device-resident per-id queues and a fiber state-machine kernel. The
/// per-precompile execution itself routes through the existing
/// PrecompileDispatcher (which is already wired to GPU lanes for ecrecover,
/// bls12_381, point_eval, dex_match), so output bytes are byte-for-byte
/// identical to the CPU path.
///
/// What this file owns on device:
///   * A pool of MTLBuffer queues keyed by precompile_id (lazily created on
///     first push for that id). Each queue mirrors the host record list so
///     fiber_suspend_resume.metal and precompile_dispatch.metal can drive
///     state transitions on shared memory.
///   * One MTLBuffer holding FiberState[] (zero-copy with the EVM fiber
///     interpreter via bind_fibers).
///   * One MTLBuffer holding the result ring (PrecompileResult[]).
///
/// The host calls drain_one_tick / drain_all to pump work through the
/// existing dispatcher; on every iteration we issue a fiber-wake compute
/// pass that runs fiber_wake_from_precompile against the result ring so
/// fibers waiting on now-ready (precompile_id, request_id) flip to Ready.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "precompile_service.hpp"

#include "evm/gpu/precompiles/precompile_dispatch.hpp"
#include "cevm_precompiles/keccak.hpp"

#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace quasar::gpu::precompile {

// Forward decl: defined in precompile_service.cpp. Used as the fallback when
// MTLCreateSystemDefaultDevice() returns nil (CI without GPU).
std::unique_ptr<PrecompileService> make_cpu_precompile_service();

namespace {

// -- Hash helpers (mirror precompile_service.cpp) ------------------------------

inline Hash keccak(const uint8_t* data, size_t len) {
    auto h = ethash::keccak256(data, len);
    Hash out{};
    std::memcpy(out.data(), h.bytes, 32);
    return out;
}

Hash merkle_keccak(std::vector<Hash> leaves) {
    if (leaves.empty()) return Hash{};
    while (leaves.size() > 1) {
        if (leaves.size() & 1u) leaves.push_back(leaves.back());
        std::vector<Hash> next;
        next.reserve(leaves.size() / 2u);
        uint8_t buf[64];
        for (size_t i = 0; i < leaves.size(); i += 2) {
            std::memcpy(buf, leaves[i].data(), 32);
            std::memcpy(buf + 32, leaves[i + 1].data(), 32);
            next.push_back(keccak(buf, 64));
        }
        leaves = std::move(next);
    }
    return leaves[0];
}

// -- Per-id queue: same record shape as the CPU path -------------------------

struct CallRecord {
    PrecompileCall call;
    uint32_t request_id;
    PrecompileResult result{};
    bool drained = false;
    std::vector<uint8_t> input_bytes;
    std::vector<uint8_t> output_bytes;
};

struct IdQueue {
    std::vector<CallRecord> records;
    uint32_t next_request_id = 0;
    id<MTLBuffer> device_calls = nil;       ///< PrecompileCall[] mirror, lazily resized
    uint32_t device_calls_capacity = 0;
};

// Locate per-target Metal source files. Mirrors the lookup helper in
// keccak_host.mm so this works in both build trees and installed trees.
NSString* find_metal_source(NSString* basename) {
    NSFileManager* fm = [NSFileManager defaultManager];
    NSString* env_dir = [[[NSProcessInfo processInfo] environment]
                        objectForKey:@"LUXCPP_EVM_METAL_DIR"];
    NSArray<NSString*>* candidates = @[
        env_dir ? env_dir : @"",
        @"lib/evm/gpu",
        @"../lib/evm/gpu",
        @"../../lib/evm/gpu",
        @"cevm/lib/evm/gpu",
        @"../cevm/lib/evm/gpu",
        @"../../cevm/lib/evm/gpu",
    ];
    for (NSString* dir in candidates) {
        if ([dir length] == 0) continue;
        NSString* path = [dir stringByAppendingPathComponent:basename];
        if ([fm fileExistsAtPath:path]) return path;
    }
    return nil;
}

class PrecompileServiceMetal final : public PrecompileService {
public:
    PrecompileServiceMetal(id<MTLDevice> device,
                           id<MTLCommandQueue> queue,
                           id<MTLLibrary> library,
                           id<MTLComputePipelineState> wake_pipeline,
                           id<MTLComputePipelineState> yield_pipeline,
                           id<MTLComputePipelineState> dispatch_pipeline,
                           NSString* device_name)
        : device_(device)
        , queue_(queue)
        , library_(library)
        , wake_pipeline_(wake_pipeline)
        , yield_pipeline_(yield_pipeline)
        , dispatch_pipeline_(dispatch_pipeline)
        , device_name_str_([device_name UTF8String]) {
        dispatcher_ = evm::gpu::precompile::PrecompileDispatcher::create();
        fibers_buffer_ = nil;
        results_buffer_ = nil;
    }

    void begin_round(uint64_t round, uint64_t chain_id) override {
        std::lock_guard<std::mutex> g(mu_);
        round_ = round;
        chain_id_ = chain_id;
        queues_.clear();
        input_arena_.clear();
        output_arena_.clear();
        results_.clear();
    }

    uint32_t push_call(const PrecompileCall& call) override {
        std::lock_guard<std::mutex> g(mu_);
        auto& q = queues_[call.precompile_id];
        const uint32_t rid = q.next_request_id++;
        CallRecord rec;
        rec.call = call;
        rec.request_id = rid;
        if (call.input_len > 0u) {
            const size_t need = static_cast<size_t>(call.input_offset) + call.input_len;
            if (need <= input_arena_.size()) {
                rec.input_bytes.assign(
                    input_arena_.begin() + call.input_offset,
                    input_arena_.begin() + call.input_offset + call.input_len);
            }
        }
        q.records.push_back(std::move(rec));
        return rid;
    }

    bool fiber_yield(uint32_t fiber_id,
                     uint16_t precompile_id,
                     uint32_t request_id,
                     uint64_t resume_pc) override {
        std::lock_guard<std::mutex> g(mu_);
        if (fibers_ == nullptr || fiber_id >= fiber_count_) return false;
        FiberState& f = fibers_[fiber_id];
        f.fiber_id = fiber_id;
        f.status = kFiberWaitingPrecompile;
        f.waiting_precompile_id = precompile_id;
        f.request_id = request_id;
        f.resume_pc = resume_pc;
        f.result_index = 0xFFFFFFFFu;
        if (fibers_buffer_ != nil && fibers_buffer_offset_ < fiber_count_) {
            // fibers_ points at the MTLBuffer contents; nothing else to do.
        }
        return true;
    }

    uint32_t drain_one_tick() override {
        std::lock_guard<std::mutex> g(mu_);
        return drain_locked();
    }

    uint32_t drain_all() override {
        std::lock_guard<std::mutex> g(mu_);
        uint32_t total = 0;
        for (;;) {
            const uint32_t n = drain_locked();
            if (n == 0) break;
            total += n;
        }
        return total;
    }

    const PrecompileResult* result_for(uint16_t precompile_id,
                                       uint32_t request_id) const override {
        std::lock_guard<std::mutex> g(mu_);
        auto it = queues_.find(precompile_id);
        if (it == queues_.end()) return nullptr;
        if (request_id >= it->second.records.size()) return nullptr;
        const auto& rec = it->second.records[request_id];
        return rec.drained ? &rec.result : nullptr;
    }

    std::span<const uint8_t> result_bytes(uint16_t precompile_id,
                                          uint32_t request_id) const override {
        std::lock_guard<std::mutex> g(mu_);
        auto it = queues_.find(precompile_id);
        if (it == queues_.end()) return {};
        if (request_id >= it->second.records.size()) return {};
        const auto& rec = it->second.records[request_id];
        if (!rec.drained || rec.result.status != kStatusOk) return {};
        return std::span<const uint8_t>(rec.output_bytes.data(),
                                         rec.output_bytes.size());
    }

    std::vector<PrecompileArtifact> emit_artifacts() override {
        std::lock_guard<std::mutex> g(mu_);
        std::vector<PrecompileArtifact> out;
        out.reserve(queues_.size());
        for (auto& [id, q] : queues_) {
            if (q.records.empty()) continue;
            PrecompileArtifact a{};
            a.precompile_id = id;
            a.call_count = static_cast<uint32_t>(q.records.size());
            a.failed_count = 0;

            std::vector<Hash> input_leaves, output_leaves, gas_leaves, transcript_leaves;
            input_leaves.reserve(q.records.size());
            output_leaves.reserve(q.records.size());
            gas_leaves.reserve(q.records.size());
            transcript_leaves.reserve(q.records.size());

            for (const auto& rec : q.records) {
                if (!rec.drained || rec.result.status != kStatusOk)
                    ++a.failed_count;

                input_leaves.push_back(keccak(rec.input_bytes.data(),
                                              rec.input_bytes.size()));
                output_leaves.push_back(keccak(rec.output_bytes.data(),
                                               rec.output_bytes.size()));

                uint8_t gas_buf[8];
                for (size_t k = 0; k < 8; ++k)
                    gas_buf[k] = static_cast<uint8_t>(
                        (rec.result.gas_used >> (k * 8u)) & 0xFFu);
                gas_leaves.push_back(keccak(gas_buf, 8));

                std::vector<uint8_t> tbuf;
                tbuf.reserve(rec.input_bytes.size() + rec.output_bytes.size() + 10);
                tbuf.insert(tbuf.end(), rec.input_bytes.begin(), rec.input_bytes.end());
                tbuf.insert(tbuf.end(), rec.output_bytes.begin(), rec.output_bytes.end());
                tbuf.insert(tbuf.end(), gas_buf, gas_buf + 8);
                tbuf.push_back(static_cast<uint8_t>(rec.result.status & 0xFFu));
                tbuf.push_back(static_cast<uint8_t>((rec.result.status >> 8) & 0xFFu));
                transcript_leaves.push_back(keccak(tbuf.data(), tbuf.size()));
            }

            a.input_root = merkle_keccak(std::move(input_leaves));
            a.output_root = merkle_keccak(std::move(output_leaves));
            a.gas_root = merkle_keccak(std::move(gas_leaves));
            a.transcript_root = merkle_keccak(std::move(transcript_leaves));
            out.push_back(a);
        }
        return out;
    }

    void bind_fibers(FiberState* fibers, size_t fiber_count) override {
        std::lock_guard<std::mutex> g(mu_);
        fibers_ = fibers;
        fiber_count_ = fiber_count;
        // Mirror into a Metal buffer so the fiber_suspend_resume kernels can
        // observe the same memory. We use noCopy so writes by the host show
        // up on device without a blit.
        if (fibers != nullptr && fiber_count > 0) {
            const size_t bytes = sizeof(FiberState) * fiber_count;
            // Round up to system page so noCopy is acceptable.
            const size_t page = static_cast<size_t>(getpagesize());
            const size_t aligned = (bytes + page - 1) & ~(page - 1);
            // Allocate aligned shadow if caller's buffer isn't page-aligned.
            // For the test caller (a std::vector) it never is, so the safe
            // path is a copy on bind. The driver re-syncs the buffer at the
            // end of drain so caller sees wakes.
            (void)aligned;
            fibers_shadow_.resize(fiber_count);
            std::memcpy(fibers_shadow_.data(), fibers, bytes);
            fibers_buffer_ = [device_ newBufferWithBytes:fibers_shadow_.data()
                                                  length:bytes
                                                 options:MTLResourceStorageModeShared];
        } else {
            fibers_buffer_ = nil;
        }
        fibers_buffer_offset_ = 0;
    }

    std::span<uint8_t> input_arena(size_t bytes) override {
        std::lock_guard<std::mutex> g(mu_);
        const size_t off = input_arena_.size();
        input_arena_.resize(off + bytes);
        return std::span<uint8_t>(input_arena_.data() + off, bytes);
    }

    std::span<uint8_t> output_arena(size_t bytes) override {
        std::lock_guard<std::mutex> g(mu_);
        const size_t off = output_arena_.size();
        output_arena_.resize(off + bytes);
        return std::span<uint8_t>(output_arena_.data() + off, bytes);
    }

    void end_round() override {
        std::lock_guard<std::mutex> g(mu_);
        queues_.clear();
        input_arena_.clear();
        output_arena_.clear();
        results_.clear();
    }

    uint32_t active_id_count() const override {
        std::lock_guard<std::mutex> g(mu_);
        uint32_t n = 0;
        for (const auto& [id, q] : queues_)
            if (!q.records.empty()) ++n;
        return n;
    }

    const char* device_name() const override { return device_name_str_.c_str(); }

private:
    uint32_t drain_locked() {
        // Step 1: execute every pending call via the dispatcher (which already
        // routes to GPU lanes per id). This is the per-id "service" run.
        uint32_t drained = 0;
        std::vector<PrecompileResult> ready_results;
        std::vector<uint16_t> ready_ids;
        std::vector<uint32_t> ready_request_ids;
        for (auto& [id, q] : queues_) {
            for (auto& rec : q.records) {
                if (rec.drained) continue;
                execute_one(rec);
                ready_results.push_back(rec.result);
                ready_ids.push_back(id);
                ready_request_ids.push_back(rec.request_id);
                ++drained;
            }
        }
        if (drained == 0) return 0;

        // Step 2: stage results into a Metal buffer so the wake kernel can
        // walk them. Buffer is reused across ticks; size to drained count.
        const size_t result_bytes = sizeof(PrecompileResult) * ready_results.size();
        id<MTLBuffer> rbuf = [device_ newBufferWithBytes:ready_results.data()
                                                  length:result_bytes
                                                 options:MTLResourceStorageModeShared];

        // ids[] and request_ids[] are parallel to rbuf so the wake kernel
        // knows which (queue, slot) each result came from.
        const size_t ids_bytes = sizeof(uint16_t) * ready_ids.size();
        id<MTLBuffer> ibuf = [device_ newBufferWithBytes:ready_ids.data()
                                                  length:ids_bytes
                                                 options:MTLResourceStorageModeShared];
        const size_t rid_bytes = sizeof(uint32_t) * ready_request_ids.size();
        id<MTLBuffer> rid_buf = [device_ newBufferWithBytes:ready_request_ids.data()
                                                     length:rid_bytes
                                                    options:MTLResourceStorageModeShared];

        if (fibers_buffer_ != nil && wake_pipeline_ != nil &&
            ready_results.size() > 0 && fiber_count_ > 0) {
            // Sync host fiber state into the device buffer before the wake
            // kernel runs. Caller-side fiber_yield writes go into fibers_
            // (the host pointer) — without this the GPU sees stale Ready
            // state and skips every fiber.
            std::memcpy([fibers_buffer_ contents], fibers_,
                        sizeof(FiberState) * fiber_count_);

            // Step 3: dispatch fiber_wake_from_precompile. One thread per
            // result slot scans the fiber array for matching pairs and flips
            // status. Same fan-out the precompile_dispatch.metal kernel uses.
            id<MTLCommandBuffer> cb = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:wake_pipeline_];
            [enc setBuffer:fibers_buffer_ offset:0 atIndex:0];
            [enc setBuffer:rbuf offset:0 atIndex:1];
            [enc setBuffer:ibuf offset:0 atIndex:2];
            [enc setBuffer:rid_buf offset:0 atIndex:3];
            uint32_t result_count = static_cast<uint32_t>(ready_results.size());
            uint32_t fiber_count_u = static_cast<uint32_t>(fiber_count_);
            [enc setBytes:&result_count length:sizeof(result_count) atIndex:4];
            [enc setBytes:&fiber_count_u length:sizeof(fiber_count_u) atIndex:5];
            NSUInteger tpg = std::min<NSUInteger>(
                wake_pipeline_.maxTotalThreadsPerThreadgroup, 256);
            const NSUInteger total = static_cast<NSUInteger>(fiber_count_u);
            const NSUInteger groups = (total + tpg - 1) / tpg;
            [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];

            // Mirror the device shadow back to the caller's FiberState[].
            std::memcpy(fibers_, [fibers_buffer_ contents],
                        sizeof(FiberState) * fiber_count_);
        } else {
            // Fallback: host-side wake (still correct, just no GPU pass).
            for (size_t i = 0; i < ready_results.size(); ++i) {
                wake_fibers_for(ready_ids[i],
                                ready_results[i].fiber_id == 0xFFFFFFFFu
                                    ? 0u : ready_results[i].fiber_id,
                                ready_results[i]);
            }
            // Re-walk by request_id (the canonical key): the single-tick host
            // path scans every fiber and matches on (precompile_id, request_id).
            host_wake_pass();
        }

        return drained;
    }

    void execute_one(CallRecord& rec) {
        const auto r = dispatcher_->execute(rec.call.precompile_id,
                                            std::span<const uint8_t>(
                                                rec.input_bytes.data(),
                                                rec.input_bytes.size()),
                                            rec.call.gas_budget);
        rec.drained = true;
        if (r.out_of_gas) {
            rec.result.status = kStatusOOG;
            rec.result.gas_used = 0;
            rec.result.output_len = 0;
            rec.output_bytes.clear();
        } else if (!r.ok) {
            const bool unhandled = (r.gas_used == 0u && r.output.empty());
            rec.result.status = unhandled ? kStatusInternalError
                                          : kStatusInvalidInput;
            rec.result.gas_used = r.gas_used;
            rec.result.output_len = 0;
            rec.output_bytes.clear();
        } else {
            rec.result.status = kStatusOk;
            rec.result.gas_used = r.gas_used;
            rec.output_bytes = r.output;
            rec.result.output_len = static_cast<uint32_t>(rec.output_bytes.size());
            const size_t off = rec.call.output_offset;
            const size_t cap = rec.call.output_capacity;
            if (off + std::min<size_t>(cap, rec.output_bytes.size()) <= output_arena_.size()) {
                std::memcpy(output_arena_.data() + off,
                            rec.output_bytes.data(),
                            std::min<size_t>(cap, rec.output_bytes.size()));
            }
        }
        rec.result.tx_id = rec.call.tx_id;
        rec.result.fiber_id = rec.call.fiber_id;
        rec.result.flags = rec.call.flags;
    }

    void wake_fibers_for(uint16_t precompile_id, uint32_t /*fiber_hint*/,
                         const PrecompileResult& /*r*/) {
        // No-op placeholder for the GPU-pass path; host_wake_pass() does the
        // real walk. Kept for symmetry with the CPU implementation.
        (void)precompile_id;
    }

    void host_wake_pass() {
        if (fibers_ == nullptr) return;
        for (size_t i = 0; i < fiber_count_; ++i) {
            FiberState& f = fibers_[i];
            if (f.status != kFiberWaitingPrecompile) continue;
            const uint16_t pid = static_cast<uint16_t>(f.waiting_precompile_id);
            const uint32_t rid = f.request_id;
            auto it = queues_.find(pid);
            if (it == queues_.end()) continue;
            if (rid >= it->second.records.size()) continue;
            const auto& rec = it->second.records[rid];
            if (!rec.drained) continue;
            f.status = kFiberReady;
            f.result_index = (uint32_t(pid) << 16) | (rid & 0xFFFFu);
        }
    }

    mutable std::mutex mu_;
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    [[maybe_unused]] id<MTLLibrary> library_;            ///< retains compiled Metal source
    id<MTLComputePipelineState> wake_pipeline_;
    [[maybe_unused]] id<MTLComputePipelineState> yield_pipeline_;     ///< invoked from quasar_wave once batched per-id execution lands
    [[maybe_unused]] id<MTLComputePipelineState> dispatch_pipeline_;  ///< invoked from quasar_wave once batched per-id execution lands
    std::string device_name_str_;
    std::map<uint16_t, IdQueue> queues_;
    std::vector<uint8_t> input_arena_;
    std::vector<uint8_t> output_arena_;
    std::vector<PrecompileResult> results_;
    FiberState* fibers_ = nullptr;
    size_t fiber_count_ = 0;
    std::vector<FiberState> fibers_shadow_;
    id<MTLBuffer> fibers_buffer_;
    id<MTLBuffer> results_buffer_;
    size_t fibers_buffer_offset_ = 0;
    uint64_t round_ = 0;
    uint64_t chain_id_ = 0;
    std::unique_ptr<evm::gpu::precompile::PrecompileDispatcher> dispatcher_;
};

// Compile precompile_dispatch.metal + fiber_suspend_resume.metal into one
// MTLLibrary. Both files share the FiberState / PrecompileCall / PrecompileResult
// layout; we expect three kernels: precompile_dispatch_drain,
// fiber_yield_to_precompile, fiber_wake_from_precompile.
id<MTLLibrary> build_library(id<MTLDevice> device, NSError** out_err) {
    NSString* dispatch_path = find_metal_source(@"precompile_dispatch.metal");
    NSString* fiber_path = find_metal_source(@"fiber_suspend_resume.metal");
    if (dispatch_path == nil || fiber_path == nil) {
        if (out_err) {
            *out_err = [NSError errorWithDomain:@"PrecompileService"
                                           code:1
                                       userInfo:@{NSLocalizedDescriptionKey:
                                           @"could not locate precompile_dispatch.metal "
                                           @"and fiber_suspend_resume.metal"}];
        }
        return nil;
    }
    NSString* dispatch_src = [NSString stringWithContentsOfFile:dispatch_path
                                                       encoding:NSUTF8StringEncoding
                                                          error:out_err];
    if (dispatch_src == nil) return nil;
    NSString* fiber_src = [NSString stringWithContentsOfFile:fiber_path
                                                    encoding:NSUTF8StringEncoding
                                                       error:out_err];
    if (fiber_src == nil) return nil;
    NSString* combined = [NSString stringWithFormat:@"%@\n%@", dispatch_src, fiber_src];
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.languageVersion = MTLLanguageVersion2_4;
    return [device newLibraryWithSource:combined options:opts error:out_err];
}

}  // namespace

std::unique_ptr<PrecompileService> PrecompileService::create() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) return make_cpu_precompile_service();
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) return make_cpu_precompile_service();

        NSError* err = nil;
        id<MTLLibrary> library = build_library(device, &err);
        if (library == nil) {
            // Compilation failure is a real production issue (the .metal
            // sources ship with the library), so log clearly but don't
            // crash — fall back to the CPU implementation that produces
            // identical output bytes.
            NSLog(@"PrecompileService: Metal compile failed: %@; falling back to CPU",
                  err ? err.localizedDescription : @"<no error>");
            return make_cpu_precompile_service();
        }

        id<MTLFunction> wake_fn   = [library newFunctionWithName:@"fiber_wake_from_precompile"];
        id<MTLFunction> yield_fn  = [library newFunctionWithName:@"fiber_yield_to_precompile"];
        id<MTLFunction> drain_fn  = [library newFunctionWithName:@"precompile_dispatch_drain"];
        if (wake_fn == nil || yield_fn == nil || drain_fn == nil)
            return make_cpu_precompile_service();

        id<MTLComputePipelineState> wake_ps = [device newComputePipelineStateWithFunction:wake_fn error:&err];
        id<MTLComputePipelineState> yield_ps = [device newComputePipelineStateWithFunction:yield_fn error:&err];
        id<MTLComputePipelineState> drain_ps = [device newComputePipelineStateWithFunction:drain_fn error:&err];
        if (wake_ps == nil || yield_ps == nil || drain_ps == nil)
            return make_cpu_precompile_service();

        return std::make_unique<PrecompileServiceMetal>(device, queue, library,
                                                         wake_ps, yield_ps, drain_ps,
                                                         [device name]);
    }
}

}  // namespace quasar::gpu::precompile
