// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file quasar_gpu_engine_cuda.cpp
/// CUDA-backed driver for QuasarGPUEngine. CUDA mirror of
/// quasar_gpu_engine.mm.
///
/// Same lifecycle, same handle/round model, same ring layout as the Metal
/// path. The kernel itself is in quasar_wave.cu (extern "C"
/// quasar_wave_kernel). The host:
///
///   begin_round  — cudaMalloc the descriptor / result / hdrs / items
///                  arenas + zero-init via cudaMemset
///   push_*       — host-side ring write into a host-shadow buffer, then
///                  cudaMemcpyAsync the diff block to device
///   run_epoch    — cudaMemcpy(desc) ; launch kernel (12 blocks, 32 threads) ;
///                  cudaStreamSynchronize ; cudaMemcpy(result, hdrs)
///   poll_*       — cudaMemcpy(items[range]) into host vector
///   end_round    — cudaFree everything
///
/// The host carries shadow copies of RingHeader[] and items_arena to keep
/// the push/poll APIs cheap; the device side is the source of truth for
/// head/tail counters via atomicCAS in the kernel.

#include "quasar_gpu_engine.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace quasar::gpu {

// Host launcher exported by quasar_wave.cu. The .cu file owns the
// <<<grid,block>>> launch syntax (only nvcc parses it); this declaration
// gives the .cpp host driver a normal C-linkage entry point.
extern "C" cudaError_t quasar_wave_launch(
    void*        d_desc,
    void*        d_result,
    void*        d_hdrs,
    void*        d_items_arena,
    void*        d_tx_index_seq,
    void*        d_mvcc_table,
    void*        d_dag_nodes,
    void*        d_fibers,
    uint32_t     mvcc_slot_count,
    cudaStream_t stream);

namespace {

// MUST match constants in quasar_wave.cu / quasar_gpu_layout.hpp.
constexpr uint32_t kDefaultRingCapacity = 4096u;
constexpr uint32_t kMaxDagParents       = 4u;
constexpr uint32_t kMaxDagChildren      = 16u;
constexpr uint32_t kFiberStackDepth     = 64u;
constexpr uint32_t kFiberMemoryBytes    = 1024u;
constexpr uint32_t kDefaultMvccSlots    = 8192u;
constexpr uint32_t kMaxFibers           = 4096u;

// Per-ring item sizes — same table as the Metal driver. Layout structs
// come from quasar_gpu_layout.hpp.
constexpr uint32_t kItemSizes[] = {
    sizeof(IngressTx),
    sizeof(DecodedTx),
    sizeof(VerifiedTx),
    sizeof(VerifiedTx),
    sizeof(ExecResult),
    sizeof(ExecResult),
    sizeof(ExecResult),
    sizeof(CommitItem),
    sizeof(StateRequest),
    sizeof(StatePage),
    sizeof(VoteIngress),
    sizeof(QuorumCert),
};
static_assert(sizeof(kItemSizes) / sizeof(kItemSizes[0]) == kNumServices,
              "kItemSizes must cover every ServiceId");

// Mirrors FiberSlot in quasar_wave.cu / quasar_wave.metal so the host can
// size the fiber arena. Fields are not directly used from C++ — only
// sizeof matters for arena layout.
struct alignas(16) FiberSlotHost {
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
    uint64_t   stack[kFiberStackDepth];
    uint8_t    memory[kFiberMemoryBytes];
};

struct alignas(16) DagNodeHost {
    uint32_t tx_index;
    uint32_t parent_count;
    uint32_t unresolved_parents;
    uint32_t child_count;
    uint32_t parents[kMaxDagParents];
    uint32_t children[kMaxDagChildren];
};

inline void cuda_must(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cuda ") + what + ": " +
                                 cudaGetErrorString(err));
    }
}

// =============================================================================
// Round — per-active-round device + host shadow state.
// =============================================================================

struct Round {
    QuasarRoundHandle      handle{};
    QuasarRoundDescriptor  desc{};

    // Device-side allocations.
    void*    d_desc        = nullptr;
    void*    d_result      = nullptr;
    void*    d_hdrs        = nullptr;
    void*    d_items       = nullptr;
    void*    d_tx_index    = nullptr;
    void*    d_mvcc        = nullptr;
    void*    d_dag         = nullptr;
    void*    d_fibers      = nullptr;
    uint64_t arena_bytes   = 0;

    // Host shadows — push/poll mutate these; sync to device on launch /
    // pull from device after launch. A simple, deterministic pattern that
    // mirrors what the Metal MTLResourceStorageModeShared buffer gives us
    // for free on Apple Silicon.
    QuasarRoundResult           shadow_result{};
    std::vector<RingHeader>     shadow_hdrs;
    std::vector<uint8_t>        shadow_items;
    uint32_t                    shadow_tx_index = 0;
    std::vector<uint8_t>        blob_arena;
};

// =============================================================================
// Engine
// =============================================================================

class QuasarGPUEngineCuda final : public QuasarGPUEngine {
public:
    QuasarGPUEngineCuda(int device, std::string device_name, cudaStream_t stream)
        : device_(device)
        , stream_(stream)
        , device_name_str_(std::move(device_name)) {}

    ~QuasarGPUEngineCuda() override {
        if (round_active())
            end_round(round_.handle);
        if (stream_) cudaStreamDestroy(stream_);
    }

    const char* device_name() const override { return device_name_str_.c_str(); }
    bool round_active() const override { return round_.handle.valid(); }

    QuasarRoundHandle begin_round(const QuasarRoundDescriptor& desc) override {
        std::lock_guard<std::mutex> g(mu_);
        if (round_.handle.valid()) return QuasarRoundHandle{0};

        round_ = Round{};
        round_.desc = desc;
        round_.desc.wave_tick_index = 0;
        round_.desc.closing_flag    = 0;

        std::vector<uint64_t> per_service_offset(kNumServices, 0);
        uint64_t arena_bytes = 0;
        for (uint32_t s = 0; s < kNumServices; ++s) {
            per_service_offset[s] = arena_bytes;
            arena_bytes += static_cast<uint64_t>(kDefaultRingCapacity) * kItemSizes[s];
        }
        round_.arena_bytes = arena_bytes;

        try {
            cuda_must(cudaMalloc(&round_.d_desc,     sizeof(QuasarRoundDescriptor)),       "malloc desc");
            cuda_must(cudaMalloc(&round_.d_result,   sizeof(QuasarRoundResult)),           "malloc result");
            cuda_must(cudaMalloc(&round_.d_hdrs,     sizeof(RingHeader) * kNumServices),   "malloc hdrs");
            cuda_must(cudaMalloc(&round_.d_items,    arena_bytes),                         "malloc items");
            cuda_must(cudaMalloc(&round_.d_tx_index, sizeof(uint32_t)),                    "malloc tx_index");
            cuda_must(cudaMalloc(&round_.d_mvcc,     sizeof(MvccSlot) * kDefaultMvccSlots),"malloc mvcc");
            cuda_must(cudaMalloc(&round_.d_dag,      sizeof(DagNodeHost) * kMaxFibers),    "malloc dag");
            cuda_must(cudaMalloc(&round_.d_fibers,   sizeof(FiberSlotHost) * kMaxFibers),  "malloc fibers");

            cuda_must(cudaMemsetAsync(round_.d_result, 0, sizeof(QuasarRoundResult), stream_),               "memset result");
            cuda_must(cudaMemsetAsync(round_.d_items,  0, arena_bytes,               stream_),               "memset items");
            cuda_must(cudaMemsetAsync(round_.d_mvcc,   0, sizeof(MvccSlot) * kDefaultMvccSlots, stream_),    "memset mvcc");
            cuda_must(cudaMemsetAsync(round_.d_dag,    0, sizeof(DagNodeHost) * kMaxFibers,     stream_),    "memset dag");
            cuda_must(cudaMemsetAsync(round_.d_fibers, 0, sizeof(FiberSlotHost) * kMaxFibers,   stream_),    "memset fibers");
            cuda_must(cudaMemsetAsync(round_.d_tx_index, 0, sizeof(uint32_t), stream_),                      "memset tx_index");
        } catch (const std::exception& e) {
            std::fprintf(stderr, "quasar(cuda) begin_round: %s\n", e.what());
            free_round_locked();
            return QuasarRoundHandle{0};
        }

        // Stamp the result mode so the host can read it back without an
        // extra read of QuasarRoundDescriptor.
        QuasarRoundResult initial_result{};
        initial_result.mode = desc.mode;
        cuda_must(cudaMemcpyAsync(round_.d_result, &initial_result,
                                  sizeof(QuasarRoundResult),
                                  cudaMemcpyHostToDevice, stream_), "cpy result.mode");
        round_.shadow_result = initial_result;

        // Initialize RingHeaders and host shadow.
        round_.shadow_hdrs.assign(kNumServices, RingHeader{});
        round_.shadow_items.assign(arena_bytes, 0);
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
            round_.shadow_hdrs[s] = h;
        }
        cuda_must(cudaMemcpyAsync(round_.d_hdrs, round_.shadow_hdrs.data(),
                                  sizeof(RingHeader) * kNumServices,
                                  cudaMemcpyHostToDevice, stream_), "cpy hdrs");

        // Stage the descriptor.
        cuda_must(cudaMemcpyAsync(round_.d_desc, &round_.desc,
                                  sizeof(QuasarRoundDescriptor),
                                  cudaMemcpyHostToDevice, stream_), "cpy desc");

        cuda_must(cudaStreamSynchronize(stream_), "sync begin");

        round_.handle = QuasarRoundHandle{++next_handle_};
        return round_.handle;
    }

    void push_txs(QuasarRoundHandle h, std::span<const HostTxBlob> txs) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h) || txs.empty()) return;

        RingHeader& ingress = round_.shadow_hdrs[static_cast<uint32_t>(ServiceId::Ingress)];
        auto* items = reinterpret_cast<IngressTx*>(round_.shadow_items.data() + ingress.items_ofs);

        const uint32_t first_tail = ingress.tail;
        const uint32_t pre_pushed = ingress.pushed;

        for (const auto& tx : txs) {
            uint32_t head = ingress.head;
            uint32_t tail = ingress.tail;
            if (tail - head >= ingress.capacity) break;

            IngressTx in{};
            in.blob_offset = static_cast<uint32_t>(round_.blob_arena.size());
            in.blob_size   = static_cast<uint32_t>(tx.bytes.size());
            in.gas_limit   = tx.gas_limit;
            in.nonce       = tx.nonce;
            uint32_t origin_lo = static_cast<uint32_t>(tx.origin & 0xFFFFFFFFu);
            uint32_t origin_hi = static_cast<uint32_t>(tx.origin >> 32);
            origin_hi &= 0x3FFFFFFFu;
            if (tx.needs_state) origin_hi |= 0x80000000u;
            if (tx.needs_exec)  origin_hi |= 0x40000000u;
            in.origin_lo = origin_lo;
            in.origin_hi = origin_hi;
            round_.blob_arena.insert(round_.blob_arena.end(), tx.bytes.begin(), tx.bytes.end());

            items[tail & ingress.mask] = in;
            ingress.tail   = tail + 1u;
            ingress.pushed += 1u;
        }

        // Push the shadow diffs to device. Items first, then header.
        const uint32_t pushed_now = ingress.pushed - pre_pushed;
        if (pushed_now != 0) {
            // Items may wrap — handle the two-segment case explicitly.
            const uint32_t cap  = ingress.capacity;
            const uint32_t mask = ingress.mask;
            const uint32_t a    = first_tail & mask;
            const uint32_t b    = ingress.tail & mask;
            const uint64_t base = ingress.items_ofs;
            uint8_t* d_arena = static_cast<uint8_t*>(round_.d_items);
            uint8_t* h_arena = round_.shadow_items.data();
            if (b > a) {
                cuda_must(cudaMemcpyAsync(d_arena + base + a * sizeof(IngressTx),
                                          h_arena + base + a * sizeof(IngressTx),
                                          pushed_now * sizeof(IngressTx),
                                          cudaMemcpyHostToDevice, stream_), "cpy ingress items");
            } else {
                const uint32_t first_seg = cap - a;
                cuda_must(cudaMemcpyAsync(d_arena + base + a * sizeof(IngressTx),
                                          h_arena + base + a * sizeof(IngressTx),
                                          first_seg * sizeof(IngressTx),
                                          cudaMemcpyHostToDevice, stream_), "cpy ingress wrap.A");
                cuda_must(cudaMemcpyAsync(d_arena + base,
                                          h_arena + base,
                                          b * sizeof(IngressTx),
                                          cudaMemcpyHostToDevice, stream_), "cpy ingress wrap.B");
            }
            // Header (tail / pushed advanced).
            cuda_must(cudaMemcpyAsync(static_cast<uint8_t*>(round_.d_hdrs) +
                                      sizeof(RingHeader) * static_cast<uint32_t>(ServiceId::Ingress),
                                      &ingress, sizeof(RingHeader),
                                      cudaMemcpyHostToDevice, stream_), "cpy ingress hdr");
        }
    }

    void push_votes(QuasarRoundHandle h, std::span<const HostVote> votes) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h) || votes.empty()) return;

        RingHeader& vote = round_.shadow_hdrs[static_cast<uint32_t>(ServiceId::Vote)];
        auto* items = reinterpret_cast<VoteIngress*>(round_.shadow_items.data() + vote.items_ofs);

        const uint32_t first_tail = vote.tail;
        const uint32_t pre_pushed = vote.pushed;

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
            vote.tail   = tail + 1u;
            vote.pushed += 1u;
        }

        const uint32_t pushed_now = vote.pushed - pre_pushed;
        if (pushed_now != 0) {
            push_ring_segment_locked(ServiceId::Vote, first_tail, vote.tail);
            cuda_must(cudaMemcpyAsync(static_cast<uint8_t*>(round_.d_hdrs) +
                                      sizeof(RingHeader) * static_cast<uint32_t>(ServiceId::Vote),
                                      &vote, sizeof(RingHeader),
                                      cudaMemcpyHostToDevice, stream_), "cpy vote hdr");
        }
    }

    std::vector<HostStateRequest> poll_state_requests(QuasarRoundHandle h) override {
        std::lock_guard<std::mutex> g(mu_);
        std::vector<HostStateRequest> out;
        if (!check_handle(h)) return out;

        // Pull the latest StateRequest ring header + items from device.
        pull_ring_locked(ServiceId::StateRequest);

        RingHeader& req = round_.shadow_hdrs[static_cast<uint32_t>(ServiceId::StateRequest)];
        auto* items = reinterpret_cast<StateRequest*>(round_.shadow_items.data() + req.items_ofs);
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
        // Mirror the consumer-advanced head/consumed back to device.
        cuda_must(cudaMemcpyAsync(static_cast<uint8_t*>(round_.d_hdrs) +
                                  sizeof(RingHeader) * static_cast<uint32_t>(ServiceId::StateRequest),
                                  &req, sizeof(RingHeader),
                                  cudaMemcpyHostToDevice, stream_), "cpy statereq hdr");
        return out;
    }

    void push_state_pages(QuasarRoundHandle h, std::span<const HostStatePage> pages) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h) || pages.empty()) return;

        RingHeader& resp = round_.shadow_hdrs[static_cast<uint32_t>(ServiceId::StateResp)];
        auto* items = reinterpret_cast<StatePage*>(round_.shadow_items.data() + resp.items_ofs);

        const uint32_t first_tail = resp.tail;
        const uint32_t pre_pushed = resp.pushed;

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
            resp.tail   = tail + 1u;
            resp.pushed += 1u;
        }

        const uint32_t pushed_now = resp.pushed - pre_pushed;
        if (pushed_now != 0) {
            push_ring_segment_locked(ServiceId::StateResp, first_tail, resp.tail);
            cuda_must(cudaMemcpyAsync(static_cast<uint8_t*>(round_.d_hdrs) +
                                      sizeof(RingHeader) * static_cast<uint32_t>(ServiceId::StateResp),
                                      &resp, sizeof(RingHeader),
                                      cudaMemcpyHostToDevice, stream_), "cpy stateresp hdr");
        }
    }

    std::vector<HostQuorumCert> poll_quorum_certs(QuasarRoundHandle h) override {
        std::lock_guard<std::mutex> g(mu_);
        std::vector<HostQuorumCert> out;
        if (!check_handle(h)) return out;

        pull_ring_locked(ServiceId::QuorumOut);

        RingHeader& qc = round_.shadow_hdrs[static_cast<uint32_t>(ServiceId::QuorumOut)];
        auto* items = reinterpret_cast<QuorumCert*>(round_.shadow_items.data() + qc.items_ofs);
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
        cuda_must(cudaMemcpyAsync(static_cast<uint8_t*>(round_.d_hdrs) +
                                  sizeof(RingHeader) * static_cast<uint32_t>(ServiceId::QuorumOut),
                                  &qc, sizeof(RingHeader),
                                  cudaMemcpyHostToDevice, stream_), "cpy qc hdr");
        return out;
    }

    QuasarRoundResult run_epoch(QuasarRoundHandle h) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return QuasarRoundResult{};

        // Stage descriptor (tick index + closing flag may have changed).
        cuda_must(cudaMemcpyAsync(round_.d_desc, &round_.desc,
                                  sizeof(QuasarRoundDescriptor),
                                  cudaMemcpyHostToDevice, stream_), "cpy desc");

        // Launch: 12 blocks (one per service), 32 threads (lane 0 drains).
        // The actual <<<grid,block>>> dispatch lives in quasar_wave.cu; we
        // call its C-linkage launcher.
        cudaError_t lerr = quasar_wave_launch(
            round_.d_desc, round_.d_result, round_.d_hdrs, round_.d_items,
            round_.d_tx_index, round_.d_mvcc, round_.d_dag, round_.d_fibers,
            kDefaultMvccSlots, stream_);
        if (lerr != cudaSuccess) {
            std::fprintf(stderr, "quasar(cuda) launch: %s\n", cudaGetErrorString(lerr));
        }
        cuda_must(cudaStreamSynchronize(stream_), "sync run_epoch");

        // Pull result + headers back. Items for output rings are pulled
        // lazily on poll_*.
        cuda_must(cudaMemcpy(&round_.shadow_result, round_.d_result,
                             sizeof(QuasarRoundResult),
                             cudaMemcpyDeviceToHost), "pull result");
        cuda_must(cudaMemcpy(round_.shadow_hdrs.data(), round_.d_hdrs,
                             sizeof(RingHeader) * kNumServices,
                             cudaMemcpyDeviceToHost), "pull hdrs");

        round_.desc.wave_tick_index += 1;
        return round_.shadow_result;
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
        return round_.shadow_result;
    }

    void request_close(QuasarRoundHandle h) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return;
        round_.desc.closing_flag = 1u;
    }

    void end_round(QuasarRoundHandle h) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return;
        free_round_locked();
    }

    RingStats ring_stats(QuasarRoundHandle h, ServiceId s) const override {
        std::lock_guard<std::mutex> g(mu_);
        RingStats out{};
        if (!check_handle(h)) return out;
        const auto& r = round_.shadow_hdrs[static_cast<uint32_t>(s)];
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

    void free_round_locked() {
        auto safe_free = [](void*& p) { if (p) { cudaFree(p); p = nullptr; } };
        safe_free(round_.d_desc);
        safe_free(round_.d_result);
        safe_free(round_.d_hdrs);
        safe_free(round_.d_items);
        safe_free(round_.d_tx_index);
        safe_free(round_.d_mvcc);
        safe_free(round_.d_dag);
        safe_free(round_.d_fibers);
        round_ = Round{};
    }

    void push_ring_segment_locked(ServiceId s, uint32_t old_tail, uint32_t new_tail) {
        const uint32_t sid  = static_cast<uint32_t>(s);
        const RingHeader& h = round_.shadow_hdrs[sid];
        const uint32_t cap  = h.capacity;
        const uint32_t mask = h.mask;
        const uint32_t isz  = h.item_size;
        const uint64_t base = h.items_ofs;
        const uint32_t a    = old_tail & mask;
        const uint32_t b    = new_tail & mask;
        const uint32_t pushed_now = new_tail - old_tail;
        uint8_t* d_arena = static_cast<uint8_t*>(round_.d_items);
        uint8_t* h_arena = round_.shadow_items.data();
        if (b > a) {
            cuda_must(cudaMemcpyAsync(d_arena + base + a * isz,
                                      h_arena + base + a * isz,
                                      pushed_now * isz,
                                      cudaMemcpyHostToDevice, stream_), "cpy ring seg");
        } else {
            const uint32_t first_seg = cap - a;
            cuda_must(cudaMemcpyAsync(d_arena + base + a * isz,
                                      h_arena + base + a * isz,
                                      first_seg * isz,
                                      cudaMemcpyHostToDevice, stream_), "cpy ring seg.A");
            cuda_must(cudaMemcpyAsync(d_arena + base,
                                      h_arena + base,
                                      b * isz,
                                      cudaMemcpyHostToDevice, stream_), "cpy ring seg.B");
        }
    }

    void pull_ring_locked(ServiceId s) {
        const uint32_t sid = static_cast<uint32_t>(s);
        // Pull the header first so head/tail reflect what the kernel just produced.
        cuda_must(cudaMemcpy(&round_.shadow_hdrs[sid],
                             static_cast<const uint8_t*>(round_.d_hdrs) + sid * sizeof(RingHeader),
                             sizeof(RingHeader),
                             cudaMemcpyDeviceToHost), "pull hdr");

        const RingHeader& h = round_.shadow_hdrs[sid];
        const uint32_t mask = h.mask;
        const uint32_t isz  = h.item_size;
        const uint64_t base = h.items_ofs;
        if (h.head >= h.tail) return;
        const uint32_t a = h.head & mask;
        const uint32_t b = h.tail & mask;
        uint8_t* d_arena = static_cast<uint8_t*>(round_.d_items);
        uint8_t* h_arena = round_.shadow_items.data();
        if (b > a) {
            cuda_must(cudaMemcpy(h_arena + base + a * isz,
                                 d_arena + base + a * isz,
                                 (b - a) * isz,
                                 cudaMemcpyDeviceToHost), "pull ring");
        } else {
            const uint32_t first_seg = h.capacity - a;
            cuda_must(cudaMemcpy(h_arena + base + a * isz,
                                 d_arena + base + a * isz,
                                 first_seg * isz,
                                 cudaMemcpyDeviceToHost), "pull ring.A");
            if (b > 0) {
                cuda_must(cudaMemcpy(h_arena + base,
                                     d_arena + base,
                                     b * isz,
                                     cudaMemcpyDeviceToHost), "pull ring.B");
            }
        }
    }

    int                  device_;
    cudaStream_t         stream_;
    std::string          device_name_str_;

    mutable std::mutex   mu_;
    Round                round_;
    uint64_t             next_handle_ = 0;
};

}  // namespace

// =============================================================================
// Factory entry point — used by the runtime dispatcher when CEVM_CUDA is on.
// =============================================================================

std::unique_ptr<QuasarGPUEngine> create_quasar_gpu_engine_cuda()
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) return nullptr;

    int device = 0;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) return nullptr;
    if (cudaSetDevice(device) != cudaSuccess) return nullptr;

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) return nullptr;

    return std::make_unique<QuasarGPUEngineCuda>(device, std::string(prop.name), stream);
}

}  // namespace quasar::gpu
