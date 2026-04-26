// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file slot_engine.mm
/// Metal-backed GPU Slot Engine driver. See slot_engine.hpp for design.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "slot_engine.hpp"

#include <atomic>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace evm::gpu::slot {

namespace {

// Per-ring capacity defaults. Power-of-two so the kernel's mask trick
// works. v0.31 keeps these modest; the slot arena grows in v0.34 once
// real wave sizes are known.
constexpr uint32_t kDefaultRingCapacity = 4096u;

// Per-ring item sizes (bytes). MUST match slot_layout.hpp.
constexpr uint32_t kItemSizes[] = {
    sizeof(IngressTx),     // ServiceId::Ingress
    sizeof(DecodedTx),     // Decode
    sizeof(VerifiedTx),    // Crypto
    sizeof(CommitItem),    // Commit
    sizeof(uint32_t) * 4,  // StateRequest (placeholder for v0.31)
    sizeof(uint32_t) * 4,  // StateResp     (placeholder)
    sizeof(uint32_t) * 4,  // Vote          (placeholder)
    sizeof(uint32_t) * 4,  // QuorumOut     (placeholder)
};
static_assert(sizeof(kItemSizes) / sizeof(kItemSizes[0]) == kNumServices,
              "kItemSizes must cover every ServiceId");

// =============================================================================
// Source loader — same path-search policy used by evm_kernel_host.mm and
// v3_persistent_host.mm.
// =============================================================================

id<MTLLibrary> compile_slot_source(id<MTLDevice> device)
{
    NSError* error = nil;
    std::filesystem::path candidates[] = {
        std::filesystem::path(__FILE__).parent_path() / "slot_kernel.metal",
        std::filesystem::current_path() / "slot_kernel.metal",
        std::filesystem::current_path() / "lib" / "evm" / "gpu" / "slot" / "slot_kernel.metal",
    };
    for (const auto& p : candidates)
    {
        if (!std::filesystem::exists(p))
            continue;
        NSString* path = [NSString stringWithUTF8String:p.c_str()];
        NSString* src  = [NSString stringWithContentsOfFile:path
                                   encoding:NSUTF8StringEncoding error:&error];
        if (!src) continue;
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        opts.languageVersion = MTLLanguageVersion3_0;
        id<MTLLibrary> lib = [device newLibraryWithSource:src
                                                  options:opts
                                                    error:&error];
        if (lib) return lib;
        if (error)
            std::fprintf(stderr, "slot_kernel.metal compile error for %s: %s\n",
                         p.c_str(), [[error localizedDescription] UTF8String]);
    }
    return nil;
}

// =============================================================================
// Slot — per-active-slot state.
// =============================================================================

struct Slot {
    SlotHandle handle{};
    SlotDescriptor desc{};

    id<MTLBuffer> desc_buf      = nil;
    id<MTLBuffer> result_buf    = nil;
    id<MTLBuffer> hdrs_buf      = nil;
    id<MTLBuffer> items_buf     = nil;
    id<MTLBuffer> tx_index_buf  = nil;
};

class SlotEngineImpl final : public SlotEngine {
public:
    SlotEngineImpl(id<MTLDevice> device,
                   id<MTLCommandQueue> queue,
                   id<MTLComputePipelineState> pso,
                   NSString* device_name)
        : device_(device)
        , queue_(queue)
        , pso_(pso)
        , device_name_str_([device_name UTF8String]) {}

    ~SlotEngineImpl() override {
        if (slot_active())
            end_slot(slot_.handle);
    }

    const char* device_name() const override { return device_name_str_.c_str(); }
    bool slot_active() const override { return slot_.handle.valid(); }

    SlotHandle begin_slot(const SlotDescriptor& desc) override {
        std::lock_guard<std::mutex> g(mu_);
        if (slot_.handle.valid())
            return SlotHandle{0};

        slot_ = Slot{};
        slot_.desc = desc;
        slot_.desc.epoch_index = 0;
        slot_.desc.closing_flag = 0;

        // Per-service item arena: lay each service's items[] back-to-back
        // in one MTLBuffer. items_ofs in the RingHeader points into this
        // arena; the kernel reads the offset and reinterprets the typed
        // view at dispatch time.
        std::vector<uint64_t> per_service_offset(kNumServices, 0);
        uint64_t arena_bytes = 0;
        for (uint32_t s = 0; s < kNumServices; ++s) {
            per_service_offset[s] = arena_bytes;
            arena_bytes += static_cast<uint64_t>(kDefaultRingCapacity) * kItemSizes[s];
        }

        slot_.desc_buf = [device_ newBufferWithLength:sizeof(SlotDescriptor)
                                              options:MTLResourceStorageModeShared];
        slot_.result_buf = [device_ newBufferWithLength:sizeof(SlotResult)
                                                options:MTLResourceStorageModeShared];
        slot_.hdrs_buf = [device_ newBufferWithLength:sizeof(RingHeader) * kNumServices
                                              options:MTLResourceStorageModeShared];
        slot_.items_buf = [device_ newBufferWithLength:arena_bytes
                                               options:MTLResourceStorageModeShared];
        slot_.tx_index_buf = [device_ newBufferWithLength:sizeof(uint32_t)
                                                  options:MTLResourceStorageModeShared];
        if (!slot_.desc_buf || !slot_.result_buf || !slot_.hdrs_buf
            || !slot_.items_buf || !slot_.tx_index_buf)
            return SlotHandle{0};

        std::memcpy([slot_.desc_buf contents], &slot_.desc, sizeof(SlotDescriptor));
        std::memset([slot_.result_buf contents], 0, sizeof(SlotResult));
        std::memset([slot_.items_buf contents], 0, arena_bytes);
        *static_cast<uint32_t*>([slot_.tx_index_buf contents]) = 0;

        // Initialize RingHeaders.
        auto* hdrs = static_cast<RingHeader*>([slot_.hdrs_buf contents]);
        for (uint32_t s = 0; s < kNumServices; ++s) {
            RingHeader h{};
            h.head      = 0;
            h.tail      = 0;
            h.capacity  = kDefaultRingCapacity;
            h.mask      = kDefaultRingCapacity - 1;
            h.items_ofs = per_service_offset[s];
            h.item_size = kItemSizes[s];
            h.pushed    = 0;
            h.consumed  = 0;
            hdrs[s] = h;
        }

        slot_.handle = SlotHandle{++next_handle_};
        return slot_.handle;
    }

    void push_txs(SlotHandle h, std::span<const HostTxBlob> txs) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return;
        if (txs.empty()) return;

        auto* hdrs = static_cast<RingHeader*>([slot_.hdrs_buf contents]);
        RingHeader& ingress = hdrs[static_cast<uint32_t>(ServiceId::Ingress)];
        auto* items = reinterpret_cast<IngressTx*>(
            static_cast<uint8_t*>([slot_.items_buf contents]) + ingress.items_ofs);

        // Host pushes via the same SPSC protocol the kernel pop side uses.
        // Backpressure: if the ring is full, the caller has to retry next
        // epoch. We stop early rather than spin so the policy stays in the
        // host (which can choose to drop, queue, or re-launch the kernel).
        // The slot tx_index sequence stays kernel-owned — host only stages
        // blob metadata.
        for (const auto& tx : txs) {
            uint32_t head = ingress.head;
            uint32_t tail = ingress.tail;
            if (tail - head >= ingress.capacity)
                break;

            IngressTx in{};
            in.blob_offset = static_cast<uint32_t>(blob_arena_.size());
            in.blob_size   = static_cast<uint32_t>(tx.bytes.size());
            in.gas_limit   = tx.gas_limit;
            in.nonce       = tx.nonce;
            in.origin_lo   = static_cast<uint32_t>(tx.origin & 0xFFFFFFFFu);
            in.origin_hi   = static_cast<uint32_t>(tx.origin >> 32);
            blob_arena_.insert(blob_arena_.end(), tx.bytes.begin(), tx.bytes.end());

            items[tail & ingress.mask] = in;
            std::atomic_thread_fence(std::memory_order_release);
            ingress.tail = tail + 1u;
            ingress.pushed += 1u;
        }
    }

    SlotResult run_epoch(SlotHandle h) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return SlotResult{};

        auto* desc_dev = static_cast<SlotDescriptor*>([slot_.desc_buf contents]);
        desc_dev->epoch_index = slot_.desc.epoch_index;
        desc_dev->closing_flag = slot_.desc.closing_flag;

        @autoreleasepool {
            id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso_];
            [enc setBuffer:slot_.desc_buf     offset:0 atIndex:0];
            [enc setBuffer:slot_.result_buf   offset:0 atIndex:1];
            [enc setBuffer:slot_.hdrs_buf     offset:0 atIndex:2];
            [enc setBuffer:slot_.items_buf    offset:0 atIndex:3];
            [enc setBuffer:slot_.tx_index_buf offset:0 atIndex:4];

            // One workgroup per service, 32 threads each. Only tid==0
            // does work; the SIMD-wide dispatch keeps room for the v0.36
            // fiber EVM to fan instruction streams across lanes.
            [enc dispatchThreadgroups:MTLSizeMake(kNumServices, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        slot_.desc.epoch_index += 1;
        return read_result_locked();
    }

    SlotResult run_until_done(SlotHandle h, std::size_t max_epochs) override {
        SlotResult last{};
        for (std::size_t i = 0; i < max_epochs; ++i) {
            last = run_epoch(h);
            if (last.status != 0) break;
        }
        return last;
    }

    SlotResult poll_result(SlotHandle h) const override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return SlotResult{};
        return read_result_locked();
    }

    void request_close(SlotHandle h) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return;
        slot_.desc.closing_flag = 1u;
    }

    void end_slot(SlotHandle h) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return;
        slot_ = Slot{};
        blob_arena_.clear();
    }

    RingStats ring_stats(SlotHandle h, ServiceId s) const override {
        std::lock_guard<std::mutex> g(mu_);
        RingStats out{};
        if (!check_handle(h)) return out;
        const auto* hdrs = static_cast<const RingHeader*>([slot_.hdrs_buf contents]);
        const auto& r = hdrs[static_cast<uint32_t>(s)];
        out.pushed   = r.pushed;
        out.consumed = r.consumed;
        out.head     = r.head;
        out.tail     = r.tail;
        out.capacity = r.capacity;
        return out;
    }

private:
    bool check_handle(SlotHandle h) const {
        return slot_.handle.valid() && slot_.handle.opaque == h.opaque;
    }

    SlotResult read_result_locked() const {
        SlotResult out{};
        std::memcpy(&out, [slot_.result_buf contents], sizeof(SlotResult));
        return out;
    }

    id<MTLDevice>               device_;
    id<MTLCommandQueue>         queue_;
    id<MTLComputePipelineState> pso_;
    std::string                 device_name_str_;

    mutable std::mutex   mu_;
    Slot                 slot_;
    uint64_t             next_handle_ = 0;
    std::vector<uint8_t> blob_arena_;
};

}  // namespace

std::unique_ptr<SlotEngine> SlotEngine::create()
{
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return nullptr;

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) return nullptr;

        id<MTLLibrary> lib = compile_slot_source(device);
        if (!lib) return nullptr;

        id<MTLFunction> fn = [lib newFunctionWithName:@"slot_scheduler_kernel"];
        if (!fn) return nullptr;

        NSError* err = nil;
        id<MTLComputePipelineState> pso =
            [device newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) return nullptr;

        return std::make_unique<SlotEngineImpl>(device, queue, pso, [device name]);
    }
}

}  // namespace evm::gpu::slot
