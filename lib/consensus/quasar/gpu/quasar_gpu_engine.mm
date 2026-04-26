// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file quasar_gpu_engine.mm
/// Metal-backed driver for QuasarGPUEngine. See quasar_gpu_engine.hpp.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "quasar_gpu_engine.hpp"

#include <atomic>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace quasar::gpu {

namespace {

// MUST match constants in quasar_wave.metal.
constexpr uint32_t kDefaultRingCapacity = 4096u;
constexpr uint32_t kMaxDagParents       = 4u;
constexpr uint32_t kMaxDagChildren      = 16u;
constexpr uint32_t kFiberStackDepth     = 64u;
constexpr uint32_t kFiberMemoryBytes    = 1024u;
constexpr uint32_t kDefaultMvccSlots    = 8192u;
constexpr uint32_t kMaxFibers           = 4096u;

// Per-ring item sizes — MUST match quasar_gpu_layout.hpp / quasar_wave.metal.
constexpr uint32_t kItemSizes[] = {
    sizeof(IngressTx),         // Ingress
    sizeof(DecodedTx),         // Decode
    sizeof(VerifiedTx),        // Crypto
    sizeof(VerifiedTx),        // DagReady (same item shape — v0.35 ready set)
    sizeof(ExecResult),        // Exec
    sizeof(ExecResult),        // Validate
    sizeof(ExecResult),        // Repair
    sizeof(CommitItem),        // Commit
    sizeof(StateRequest),      // StateRequest
    sizeof(StatePage),         // StateResp
    sizeof(VoteIngress),       // Vote
    sizeof(QuorumCert),        // QuorumOut
};
static_assert(sizeof(kItemSizes) / sizeof(kItemSizes[0]) == kNumServices,
              "kItemSizes must cover every ServiceId");

// Mirrors FiberSlot in quasar_wave.metal so the host can size the fiber
// arena. Fields are not directly used from C++ — only sizeof matters.
struct alignas(16) FiberSlot {
    uint32_t tx_index;
    uint32_t pc;
    uint32_t sp;
    uint32_t status;
    uint64_t gas;
    uint32_t rw_count;
    uint32_t incarnation;
    uint32_t pending_key_lo_lo;
    uint32_t pending_key_lo_hi;
    uint32_t pending_key_hi_lo;
    uint32_t pending_key_hi_hi;
    uint32_t _pad0;
    RWSetEntry rw[kMaxRWSetPerTx];
    uint64_t  stack[kFiberStackDepth];
    uint8_t   memory[kFiberMemoryBytes];
};

struct alignas(16) DagNodeHost {
    uint32_t tx_index;
    uint32_t parent_count;
    uint32_t unresolved_parents;
    uint32_t child_count;
    uint32_t parents[kMaxDagParents];
    uint32_t children[kMaxDagChildren];
};

// =============================================================================
// Source loader
// =============================================================================

id<MTLLibrary> compile_quasar_source(id<MTLDevice> device)
{
    NSError* error = nil;
    std::filesystem::path candidates[] = {
        std::filesystem::path(__FILE__).parent_path() / "quasar_wave.metal",
        std::filesystem::current_path() / "quasar_wave.metal",
        std::filesystem::current_path() / "lib" / "consensus" / "quasar" / "gpu" / "quasar_wave.metal",
    };
    for (const auto& p : candidates) {
        if (!std::filesystem::exists(p)) continue;
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
            std::fprintf(stderr, "quasar_wave.metal compile error for %s: %s\n",
                         p.c_str(), [[error localizedDescription] UTF8String]);
    }
    return nil;
}

// =============================================================================
// Round — per-active-round state
// =============================================================================

struct Round {
    QuasarRoundHandle handle{};
    QuasarRoundDescriptor desc{};

    id<MTLBuffer> desc_buf       = nil;
    id<MTLBuffer> result_buf     = nil;
    id<MTLBuffer> hdrs_buf       = nil;
    id<MTLBuffer> items_buf      = nil;
    id<MTLBuffer> tx_index_buf   = nil;
    id<MTLBuffer> mvcc_buf       = nil;
    id<MTLBuffer> dag_buf        = nil;
    id<MTLBuffer> fibers_buf     = nil;
    id<MTLBuffer> mvcc_count_buf = nil;
};

class QuasarGPUEngineImpl final : public QuasarGPUEngine {
public:
    QuasarGPUEngineImpl(id<MTLDevice> device,
                        id<MTLCommandQueue> queue,
                        id<MTLComputePipelineState> pso,
                        NSString* device_name)
        : device_(device)
        , queue_(queue)
        , pso_(pso)
        , device_name_str_([device_name UTF8String]) {}

    ~QuasarGPUEngineImpl() override {
        if (round_active())
            end_round(round_.handle);
    }

    const char* device_name() const override { return device_name_str_.c_str(); }
    bool round_active() const override { return round_.handle.valid(); }

    QuasarRoundHandle begin_round(const QuasarRoundDescriptor& desc) override {
        std::lock_guard<std::mutex> g(mu_);
        if (round_.handle.valid()) return QuasarRoundHandle{0};

        round_ = Round{};
        round_.desc = desc;
        round_.desc.wave_tick_index = 0;
        round_.desc.closing_flag = 0;

        std::vector<uint64_t> per_service_offset(kNumServices, 0);
        uint64_t arena_bytes = 0;
        for (uint32_t s = 0; s < kNumServices; ++s) {
            per_service_offset[s] = arena_bytes;
            arena_bytes += static_cast<uint64_t>(kDefaultRingCapacity) * kItemSizes[s];
        }

        round_.desc_buf       = [device_ newBufferWithLength:sizeof(QuasarRoundDescriptor)        options:MTLResourceStorageModeShared];
        round_.result_buf     = [device_ newBufferWithLength:sizeof(QuasarRoundResult)            options:MTLResourceStorageModeShared];
        round_.hdrs_buf       = [device_ newBufferWithLength:sizeof(RingHeader) * kNumServices    options:MTLResourceStorageModeShared];
        round_.items_buf      = [device_ newBufferWithLength:arena_bytes                          options:MTLResourceStorageModeShared];
        round_.tx_index_buf   = [device_ newBufferWithLength:sizeof(uint32_t)                     options:MTLResourceStorageModeShared];
        round_.mvcc_buf       = [device_ newBufferWithLength:sizeof(MvccSlot) * kDefaultMvccSlots options:MTLResourceStorageModeShared];
        round_.dag_buf        = [device_ newBufferWithLength:sizeof(DagNodeHost) * kMaxFibers     options:MTLResourceStorageModeShared];
        round_.fibers_buf     = [device_ newBufferWithLength:sizeof(FiberSlot) * kMaxFibers       options:MTLResourceStorageModeShared];
        round_.mvcc_count_buf = [device_ newBufferWithLength:sizeof(uint32_t)                     options:MTLResourceStorageModeShared];

        if (!round_.desc_buf || !round_.result_buf || !round_.hdrs_buf
            || !round_.items_buf || !round_.tx_index_buf || !round_.mvcc_buf
            || !round_.dag_buf  || !round_.fibers_buf || !round_.mvcc_count_buf)
            return QuasarRoundHandle{0};

        std::memcpy([round_.desc_buf contents], &round_.desc, sizeof(QuasarRoundDescriptor));
        std::memset([round_.result_buf contents], 0, sizeof(QuasarRoundResult));
        std::memset([round_.items_buf contents], 0, arena_bytes);
        std::memset([round_.mvcc_buf contents], 0, sizeof(MvccSlot) * kDefaultMvccSlots);
        std::memset([round_.dag_buf contents], 0, sizeof(DagNodeHost) * kMaxFibers);
        std::memset([round_.fibers_buf contents], 0, sizeof(FiberSlot) * kMaxFibers);
        *static_cast<uint32_t*>([round_.tx_index_buf contents]) = 0;
        *static_cast<uint32_t*>([round_.mvcc_count_buf contents]) = kDefaultMvccSlots;

        // Stamp the result.mode so the host doesn't have to read it back
        // separately.
        auto* result = static_cast<QuasarRoundResult*>([round_.result_buf contents]);
        result->mode = desc.mode;

        // Initialize RingHeaders.
        auto* hdrs = static_cast<RingHeader*>([round_.hdrs_buf contents]);
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

        round_.handle = QuasarRoundHandle{++next_handle_};
        return round_.handle;
    }

    void push_txs(QuasarRoundHandle h, std::span<const HostTxBlob> txs) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return;
        if (txs.empty()) return;

        auto* hdrs = static_cast<RingHeader*>([round_.hdrs_buf contents]);
        RingHeader& ingress = hdrs[static_cast<uint32_t>(ServiceId::Ingress)];
        auto* items = reinterpret_cast<IngressTx*>(
            static_cast<uint8_t*>([round_.items_buf contents]) + ingress.items_ofs);

        for (const auto& tx : txs) {
            uint32_t head = ingress.head;
            uint32_t tail = ingress.tail;
            if (tail - head >= ingress.capacity) break;

            IngressTx in{};
            in.blob_offset = static_cast<uint32_t>(blob_arena_.size());
            in.blob_size   = static_cast<uint32_t>(tx.bytes.size());
            in.gas_limit   = tx.gas_limit;
            in.nonce       = tx.nonce;
            uint32_t origin_lo = static_cast<uint32_t>(tx.origin & 0xFFFFFFFFu);
            uint32_t origin_hi = static_cast<uint32_t>(tx.origin >> 32);
            if (tx.needs_state) origin_hi |= 0x80000000u;
            in.origin_lo = origin_lo;
            in.origin_hi = origin_hi;
            blob_arena_.insert(blob_arena_.end(), tx.bytes.begin(), tx.bytes.end());

            items[tail & ingress.mask] = in;
            std::atomic_thread_fence(std::memory_order_release);
            ingress.tail = tail + 1u;
            ingress.pushed += 1u;
        }
    }

    void push_votes(QuasarRoundHandle h, std::span<const HostVote> votes) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return;
        if (votes.empty()) return;

        auto* hdrs = static_cast<RingHeader*>([round_.hdrs_buf contents]);
        RingHeader& vote = hdrs[static_cast<uint32_t>(ServiceId::Vote)];
        auto* items = reinterpret_cast<VoteIngress*>(
            static_cast<uint8_t*>([round_.items_buf contents]) + vote.items_ofs);

        for (const auto& v : votes) {
            uint32_t head = vote.head;
            uint32_t tail = vote.tail;
            if (tail - head >= vote.capacity) break;

            VoteIngress out{};
            out.validator_index = v.validator_index;
            out.round           = v.round;
            out.stake_weight    = v.stake_weight;
            out.sig_kind        = v.sig_kind;
            std::memcpy(out.subject, v.block_hash, 32);
            const size_t copy = std::min<size_t>(v.signature.size(), sizeof(out.signature));
            if (copy) std::memcpy(out.signature, v.signature.data(), copy);

            items[tail & vote.mask] = out;
            std::atomic_thread_fence(std::memory_order_release);
            vote.tail   = tail + 1u;
            vote.pushed += 1u;
        }
    }

    std::vector<HostStateRequest> poll_state_requests(QuasarRoundHandle h) override {
        std::lock_guard<std::mutex> g(mu_);
        std::vector<HostStateRequest> out;
        if (!check_handle(h)) return out;

        auto* hdrs = static_cast<RingHeader*>([round_.hdrs_buf contents]);
        RingHeader& req = hdrs[static_cast<uint32_t>(ServiceId::StateRequest)];
        auto* items = reinterpret_cast<StateRequest*>(
            static_cast<uint8_t*>([round_.items_buf contents]) + req.items_ofs);

        while (true) {
            uint32_t head = req.head;
            uint32_t tail = req.tail;
            if (head >= tail) break;
            const auto& d = items[head & req.mask];
            HostStateRequest hr{};
            hr.tx_index = d.tx_index;
            hr.key_type = d.key_type;
            hr.priority = d.priority;
            hr.key_lo   = d.key_lo;
            hr.key_hi   = d.key_hi;
            out.push_back(hr);
            req.head     = head + 1u;
            req.consumed += 1u;
        }
        return out;
    }

    void push_state_pages(QuasarRoundHandle h,
                          std::span<const HostStatePage> pages) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return;
        if (pages.empty()) return;

        auto* hdrs = static_cast<RingHeader*>([round_.hdrs_buf contents]);
        RingHeader& resp = hdrs[static_cast<uint32_t>(ServiceId::StateResp)];
        auto* items = reinterpret_cast<StatePage*>(
            static_cast<uint8_t*>([round_.items_buf contents]) + resp.items_ofs);

        for (const auto& p : pages) {
            uint32_t head = resp.head;
            uint32_t tail = resp.tail;
            if (tail - head >= resp.capacity) break;

            StatePage out{};
            out.tx_index  = p.tx_index;
            out.key_type  = p.key_type;
            out.status    = p.status;
            out.key_lo    = p.key_lo;
            out.key_hi    = p.key_hi;
            const size_t copy = std::min<size_t>(p.data.size(), sizeof(out.data));
            out.data_size = static_cast<uint32_t>(copy);
            if (copy) std::memcpy(out.data, p.data.data(), copy);

            items[tail & resp.mask] = out;
            std::atomic_thread_fence(std::memory_order_release);
            resp.tail   = tail + 1u;
            resp.pushed += 1u;
        }
    }

    std::vector<HostQuorumCert> poll_quorum_certs(QuasarRoundHandle h) override {
        std::lock_guard<std::mutex> g(mu_);
        std::vector<HostQuorumCert> out;
        if (!check_handle(h)) return out;

        auto* hdrs = static_cast<RingHeader*>([round_.hdrs_buf contents]);
        RingHeader& qc = hdrs[static_cast<uint32_t>(ServiceId::QuorumOut)];
        auto* items = reinterpret_cast<QuorumCert*>(
            static_cast<uint8_t*>([round_.items_buf contents]) + qc.items_ofs);

        while (true) {
            uint32_t head = qc.head;
            uint32_t tail = qc.tail;
            if (head >= tail) break;
            const auto& d = items[head & qc.mask];
            HostQuorumCert hc{};
            hc.round         = d.round;
            hc.status        = d.status;
            hc.signers_count = d.signers_count;
            hc.total_stake   = d.total_stake;
            std::memcpy(hc.block_hash, d.subject, 32);
            hc.agg_signature.assign(d.agg_signature, d.agg_signature + 96);
            out.push_back(std::move(hc));
            qc.head     = head + 1u;
            qc.consumed += 1u;
        }
        return out;
    }

    QuasarRoundResult run_epoch(QuasarRoundHandle h) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return QuasarRoundResult{};

        auto* desc_dev = static_cast<QuasarRoundDescriptor*>([round_.desc_buf contents]);
        desc_dev->wave_tick_index = round_.desc.wave_tick_index;
        desc_dev->closing_flag    = round_.desc.closing_flag;

        @autoreleasepool {
            id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso_];
            [enc setBuffer:round_.desc_buf       offset:0 atIndex:0];
            [enc setBuffer:round_.result_buf     offset:0 atIndex:1];
            [enc setBuffer:round_.hdrs_buf       offset:0 atIndex:2];
            [enc setBuffer:round_.items_buf      offset:0 atIndex:3];
            [enc setBuffer:round_.tx_index_buf   offset:0 atIndex:4];
            [enc setBuffer:round_.mvcc_buf       offset:0 atIndex:5];
            [enc setBuffer:round_.dag_buf        offset:0 atIndex:6];
            [enc setBuffer:round_.fibers_buf     offset:0 atIndex:7];
            [enc setBuffer:round_.mvcc_count_buf offset:0 atIndex:8];

            [enc dispatchThreadgroups:MTLSizeMake(kNumServices, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        round_.desc.wave_tick_index += 1;
        return read_result_locked();
    }

    QuasarRoundResult run_until_done(QuasarRoundHandle h, std::size_t max_epochs) override {
        QuasarRoundResult last{};
        for (std::size_t i = 0; i < max_epochs; ++i) {
            last = run_epoch(h);
            if (last.status != 0) break;
        }
        return last;
    }

    QuasarRoundResult poll_round_result(QuasarRoundHandle h) const override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return QuasarRoundResult{};
        return read_result_locked();
    }

    void request_close(QuasarRoundHandle h) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return;
        round_.desc.closing_flag = 1u;
    }

    void end_round(QuasarRoundHandle h) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return;
        round_ = Round{};
        blob_arena_.clear();
    }

    RingStats ring_stats(QuasarRoundHandle h, ServiceId s) const override {
        std::lock_guard<std::mutex> g(mu_);
        RingStats out{};
        if (!check_handle(h)) return out;
        const auto* hdrs = static_cast<const RingHeader*>([round_.hdrs_buf contents]);
        const auto& r = hdrs[static_cast<uint32_t>(s)];
        out.pushed   = r.pushed;
        out.consumed = r.consumed;
        out.head     = r.head;
        out.tail     = r.tail;
        out.capacity = r.capacity;
        return out;
    }

private:
    bool check_handle(QuasarRoundHandle h) const {
        return round_.handle.valid() && round_.handle.opaque == h.opaque;
    }

    QuasarRoundResult read_result_locked() const {
        QuasarRoundResult out{};
        std::memcpy(&out, [round_.result_buf contents], sizeof(QuasarRoundResult));
        return out;
    }

    id<MTLDevice>               device_;
    id<MTLCommandQueue>         queue_;
    id<MTLComputePipelineState> pso_;
    std::string                 device_name_str_;

    mutable std::mutex   mu_;
    Round                round_;
    uint64_t             next_handle_ = 0;
    std::vector<uint8_t> blob_arena_;
};

}  // namespace

std::unique_ptr<QuasarGPUEngine> QuasarGPUEngine::create()
{
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return nullptr;

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) return nullptr;

        id<MTLLibrary> lib = compile_quasar_source(device);
        if (!lib) return nullptr;

        id<MTLFunction> fn = [lib newFunctionWithName:@"quasar_wave_kernel"];
        if (!fn) return nullptr;

        NSError* err = nil;
        id<MTLComputePipelineState> pso =
            [device newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) return nullptr;

        return std::make_unique<QuasarGPUEngineImpl>(device, queue, pso, [device name]);
    }
}

}  // namespace quasar::gpu
