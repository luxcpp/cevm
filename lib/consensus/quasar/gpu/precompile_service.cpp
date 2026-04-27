// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file precompile_service.cpp
/// Host-portable PrecompileService implementation.
///
/// On non-Apple targets this is the only implementation: queues, fiber state
/// machine, and per-id batched drain run on host with the existing
/// PrecompileDispatcher. On Apple, precompile_service.mm overrides
/// PrecompileService::create() with a Metal-driven variant that reuses the
/// same drain/artifact recipes (defined here) but maintains the per-id
/// queues in MTLBuffers so the GPU dispatch and fiber suspend/resume kernels
/// operate on shared memory.
///
/// The artifact recipe (input_root, output_root, gas_root, transcript_root)
/// is identical on both paths so cross-backend determinism holds.

#include "precompile_service.hpp"

#include "evm/gpu/precompiles/precompile_dispatch.hpp"
#include "cevm_precompiles/keccak.hpp"

#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace quasar::gpu::precompile {

namespace {

// -- Hash helpers -------------------------------------------------------------

inline Hash keccak(const uint8_t* data, size_t len) {
    auto h = ethash::keccak256(data, len);
    Hash out{};
    std::memcpy(out.data(), h.bytes, 32);
    return out;
}

// Binary keccak Merkle over a list of 32-byte leaves. Empty list → all-zero
// digest (canonical). Odd levels duplicate the last leaf — the same recipe
// the v0.34 keccak service uses for receipts_root.
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

// -- Per-id queue -------------------------------------------------------------

struct CallRecord {
    PrecompileCall call;
    uint32_t request_id;
    PrecompileResult result{};
    bool drained = false;
    std::vector<uint8_t> input_bytes;       ///< snapshot of input slice at push time
    std::vector<uint8_t> output_bytes;      ///< populated on drain
};

// One queue per precompile_id. ids are sparse so we use a map; insertion is
// O(log N) per push, which is fine — the hot path is drain (linear in the
// queue size, no map lookup).
struct IdQueue {
    std::vector<CallRecord> records;        ///< append-only within a round
    uint32_t next_request_id = 0;
};

// -- The service --------------------------------------------------------------

class PrecompileServiceCpu final : public PrecompileService {
public:
    PrecompileServiceCpu() {
        dispatcher_ = evm::gpu::precompile::PrecompileDispatcher::create();
    }

    void begin_round(uint64_t round, uint64_t chain_id) override {
        std::lock_guard<std::mutex> g(mu_);
        round_ = round;
        chain_id_ = chain_id;
        queues_.clear();
        result_index_.clear();
        input_arena_.clear();
        output_arena_.clear();
        // Caller may rebind FiberState[] between rounds; do not reset the
        // pointer here — bind_fibers explicitly owns that.
    }

    uint32_t push_call(const PrecompileCall& call) override {
        std::lock_guard<std::mutex> g(mu_);
        auto& q = queues_[call.precompile_id];
        const uint32_t rid = q.next_request_id++;
        CallRecord rec;
        rec.call = call;
        rec.request_id = rid;
        // Snapshot input bytes from arena so on-device draining is referentially
        // safe even if the arena vector reallocates between push and drain.
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

            std::vector<Hash> input_leaves;
            std::vector<Hash> output_leaves;
            std::vector<Hash> gas_leaves;
            std::vector<Hash> transcript_leaves;
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

                // transcript = input || output || gas_le8 || status_le2
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
        result_index_.clear();
        input_arena_.clear();
        output_arena_.clear();
    }

    uint32_t active_id_count() const override {
        std::lock_guard<std::mutex> g(mu_);
        uint32_t n = 0;
        for (const auto& [id, q] : queues_)
            if (!q.records.empty()) ++n;
        return n;
    }

    const char* device_name() const override { return "cpu"; }

private:
    // v0.44 — per-id batched drain. We collect every undrained record for
    // each id first, then call execute_batch_for_id once per id. This is the
    // hook the v0.44 pipelines latch onto (ecrecover Stage A batch inv, BLS
    // Miller-on-device, KZG batched-pairing). Today the per-call dispatcher
    // already routes to the right backend; the batch grouping here gives the
    // service one decision point per id where future kernel-launch
    // amortization plugs in without changing call sites.
    uint32_t drain_locked() {
        uint32_t drained = 0;
        for (auto& [id, q] : queues_) {
            // Collect indices of pending records for this id.
            std::vector<size_t> pending;
            pending.reserve(q.records.size());
            for (size_t i = 0; i < q.records.size(); ++i) {
                if (!q.records[i].drained) pending.push_back(i);
            }
            if (pending.empty()) continue;
            execute_batch_for_id(id, q.records, pending);
            for (size_t idx : pending) {
                wake_fibers_for(id, q.records[idx].request_id,
                                q.records[idx].result);
                ++drained;
            }
        }
        return drained;
    }

    // v0.44 entry point. The batch path is currently a per-call loop because
    // the underlying dispatcher API is per-call; structuring the loop here
    // means future kernel-launch amortization (e.g. one Metal command buffer
    // per id, one ecrecover batch_inv pass per id) plugs in without touching
    // any caller. Per-id artifact rooting reads the same records, so this is
    // a pure dispatch refactor — output bytes and gas accounting unchanged.
    void execute_batch_for_id(uint16_t /*id*/,
                              std::vector<CallRecord>& records,
                              const std::vector<size_t>& pending) {
        for (size_t idx : pending) execute_one(records[idx]);
    }

    void execute_one(CallRecord& rec) {
        // Route through the existing dispatcher. This is the seam where the
        // per-precompile GPU lanes are wired (ecrecover_metal, bls12_381_metal,
        // point_eval_metal, dex_match_metal). For ids the dispatcher does not
        // know (AIVM range, future DEX entries), execute returns Result{} and
        // the call is reported InvalidInput — same path the v0.42 pre-service
        // shim used.
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
            // Distinguish "dispatcher had no implementation" (output empty,
            // gas_used == 0) from "ran but rejected the input" (gas_used > 0
            // matches the EVM precompile-failure semantics in the dispatcher).
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
            // Mirror into output arena if caller reserved space.
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

    // Walk fibers; any that are blocked on (precompile_id, request_id)
    // transition Ready and store the index of their result for cheap lookup.
    void wake_fibers_for(uint16_t precompile_id, uint32_t request_id,
                         const PrecompileResult& /*result*/) {
        if (fibers_ == nullptr) return;
        for (size_t i = 0; i < fiber_count_; ++i) {
            FiberState& f = fibers_[i];
            if (f.status != kFiberWaitingPrecompile) continue;
            if (f.waiting_precompile_id != precompile_id) continue;
            if (f.request_id != request_id) continue;
            f.status = kFiberReady;
            // result_index encodes (precompile_id, request_id) — host reads
            // back via result_for() which is O(1) given those two fields.
            f.result_index = (uint32_t(precompile_id) << 16) | (request_id & 0xFFFFu);
        }
    }

    mutable std::mutex mu_;
    std::map<uint16_t, IdQueue> queues_;
    std::map<uint64_t, uint32_t> result_index_;     ///< (id<<32)|rid -> linear index (reserved)
    std::vector<uint8_t> input_arena_;
    std::vector<uint8_t> output_arena_;
    FiberState* fibers_ = nullptr;
    size_t fiber_count_ = 0;
    uint64_t round_ = 0;
    uint64_t chain_id_ = 0;
    std::unique_ptr<evm::gpu::precompile::PrecompileDispatcher> dispatcher_;
};

}  // namespace

// On non-Apple targets this is the only create() in the link unit. On Apple,
// precompile_service.mm provides a strong override that wins at link time
// because it's pulled in earlier from evm-kernel-metal (same pattern as
// quasar_gpu_engine.mm vs the CUDA mirror).
#if !defined(__APPLE__)
std::unique_ptr<PrecompileService> PrecompileService::create() {
    return std::make_unique<PrecompileServiceCpu>();
}
#endif

// Apple translation unit calls into this to construct the host-portable
// fallback when no Metal device is found. Not part of the public ABI.
std::unique_ptr<PrecompileService> make_cpu_precompile_service() {
    return std::make_unique<PrecompileServiceCpu>();
}

}  // namespace quasar::gpu::precompile
