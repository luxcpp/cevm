// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file quasar_gpu_engine.mm
/// Metal-backed driver for QuasarGPUEngine. See quasar_gpu_engine.hpp.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "quasar_cpu_reference.hpp"
#include "quasar_gpu_engine.hpp"
#include "quasar_sig.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace quasar::gpu {

namespace {

// MUST match constants in quasar_wave.metal.
constexpr uint32_t kDefaultRingCapacity   = 4096u;
constexpr uint32_t kMaxDagParents         = 4u;
constexpr uint32_t kMaxDagChildren        = 16u;
constexpr uint32_t kMaxPredictedKeys      = 4u;
constexpr uint32_t kFiberStackDepth       = 64u;
constexpr uint32_t kFiberStackLimbs       = 4u;       ///< v0.41 — 256-bit
constexpr uint32_t kFiberMemoryBytes      = 1024u;
constexpr uint32_t kDefaultMvccSlots      = 8192u;
constexpr uint32_t kDefaultDagWriterSlots = 8192u;
constexpr uint32_t kMaxFibers             = 4096u;
constexpr uint32_t kDagNodeCapacity       = kMaxFibers;
constexpr uint32_t kCodeArenaBytes        = 256u * 1024u;   ///< v0.41 — bytecode pool

constexpr uint32_t kItemSizes[] = {
    sizeof(IngressTx), sizeof(DecodedTx), sizeof(VerifiedTx), sizeof(VerifiedTx),
    sizeof(ExecResult), sizeof(ExecResult), sizeof(ExecResult), sizeof(CommitItem),
    sizeof(StateRequest), sizeof(StatePage), sizeof(VoteIngress), sizeof(QuorumCert),
    // v0.44 — five new chain-transition services. Each ring carries
    // ChainTransitionItem-sized records (32-byte root). The substrate doesn't
    // process them yet (host writes the descriptor field directly); the rings
    // exist as work-queue addresses for the per-VM ingress path landing in
    // v0.45+. Use the smallest stable record (IngressTx envelope) so the
    // arena layout is conservative.
    sizeof(IngressTx), sizeof(IngressTx), sizeof(IngressTx),
    sizeof(IngressTx), sizeof(IngressTx),
};
static_assert(sizeof(kItemSizes) / sizeof(kItemSizes[0]) == kNumServices,
              "kItemSizes must cover every ServiceId");

// MUST match struct FiberSlot in quasar_wave.metal byte-for-byte.
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
    uint32_t origin_lo;
    uint32_t origin_hi;
    uint64_t gas_limit;
    uint32_t msize;
    uint32_t code_offset;
    uint32_t code_size;
    uint32_t _pad0;
    RWSetEntry rw[kMaxRWSetPerTx];
    uint64_t  stack[kFiberStackDepth * kFiberStackLimbs];   ///< v0.41 — 256-bit
    uint8_t   memory[kFiberMemoryBytes];
};

struct alignas(16) DagNodeHost {
    uint32_t tx_index;
    uint32_t parent_count;
    uint32_t unresolved_parents;
    uint32_t child_count;
    uint32_t parents[kMaxDagParents];
    uint32_t children[kMaxDagChildren];
    uint64_t pending_gas_limit;
    uint32_t pending_origin_lo;
    uint32_t pending_origin_hi;
    uint32_t pending_admission;
    uint32_t state;
    uint32_t pending_blob_offset;       ///< v0.41
    uint32_t pending_blob_size;
};

struct PredictedKeyHost {
    uint64_t key_lo;
    uint64_t key_hi;
    uint32_t is_write;
    uint32_t valid;
};
static_assert(sizeof(PredictedKeyHost) == 24, "PredictedKeyHost layout drift");

constexpr uint32_t kPredictedSlotsPerRound = kDagNodeCapacity * kMaxPredictedKeys;

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

struct Round {
    QuasarRoundHandle handle{};
    QuasarRoundDescriptor desc{};

    id<MTLBuffer> desc_buf            = nil;
    id<MTLBuffer> result_buf          = nil;
    id<MTLBuffer> hdrs_buf            = nil;
    id<MTLBuffer> items_buf           = nil;
    id<MTLBuffer> tx_index_buf        = nil;
    id<MTLBuffer> mvcc_buf            = nil;
    id<MTLBuffer> dag_buf             = nil;
    id<MTLBuffer> fibers_buf          = nil;
    id<MTLBuffer> mvcc_count_buf      = nil;
    // v0.38 — vote-verifier output bitmap.
    id<MTLBuffer> vote_verified_buf   = nil;
    id<MTLBuffer> vote_capacity_buf   = nil;
    // v0.40 — DAG construction buffers.
    id<MTLBuffer> dag_writer_buf      = nil;
    id<MTLBuffer> dag_writer_count_buf = nil;
    id<MTLBuffer> predicted_buf       = nil;
    id<MTLBuffer> predicted_cap_buf   = nil;
    id<MTLBuffer> dag_node_cap_buf    = nil;
    // v0.41 — EVM bytecode interpreter buffers.
    id<MTLBuffer> code_buf            = nil;
    id<MTLBuffer> code_size_buf       = nil;
    id<MTLBuffer> fiber_cap_buf       = nil;
    uint32_t      next_code_offset    = 0u;
    // v0.42 — CERT-001: master secret out of source. Host populates from env.
    id<MTLBuffer> cert_master_secret_buf = nil;

    uint32_t      next_predicted_idx  = 0u;

    // v0.46.1 — host-side mirror of accepted ingress txs. Used by the
    // substrate-threshold gate in run_until_done to dispatch small N to the
    // CPU reference. Output is byte-equal across both branches by the
    // existing cross-backend determinism contract (quasar_stm_red_review_test).
    std::vector<quasar::gpu::ref::HostInputTx> host_input_mirror;
    // Mirror of host-side blob-related copies of HostTxBlob for replay into
    // Metal buffers if and only if the gate routes to Metal. Holds the
    // original gas/nonce/origin/needs/predicted_access/bytes per accepted tx.
    std::vector<HostTxBlob> tx_replay;
    // Index into tx_replay of the next tx to mirror into Metal buffers.
    // push_txs_to_metal_locked drains [tx_replay_metal_cursor .. tx_replay.end()).
    std::size_t tx_replay_metal_cursor = 0;

    // v0.46.1 — Metal initialization is deferred until run_until_done knows
    // whether the gate routes to CPU (no Metal needed) or Metal (allocate
    // + replay tx_replay into device buffers). begin_round only sets desc +
    // handle; push_txs only mirrors. metal_initialized stays false on the
    // CPU branch, eliminating ~20 Metal buffer allocations + ~12 memsets at
    // small N.
    bool metal_initialized = false;
    // Host-side cached result, populated by run_cpu_reference_locked when
    // the gate routes to CPU. poll_round_result returns this on the CPU
    // branch (Metal result_buf isn't allocated until init_metal_locked).
    QuasarRoundResult cached_cpu_result{};
    bool              cached_cpu_result_valid = false;
    // Set by push_votes / push_state_pages / push_state_requests. When
    // any of these has been called, the round must dispatch to Metal —
    // the CPU reference doesn't model votes / cold-state.
    bool requires_metal = false;
};

class QuasarGPUEngineImpl final : public QuasarGPUEngine {
public:
    QuasarGPUEngineImpl(id<MTLDevice> device,
                        id<MTLCommandQueue> queue,
                        id<MTLComputePipelineState> pso,
                        id<MTLComputePipelineState> verify_pso,
                        NSString* device_name)
        : device_(device)
        , queue_(queue)
        , pso_(pso)
        , verify_pso_(verify_pso)
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
        round_.host_input_mirror.clear();
        round_.tx_replay.clear();
        round_.metal_initialized = false;

        // v0.46.1 — Metal init is deferred to run_until_done so the gate can
        // skip ~20 Metal buffer allocations + memsets at small N. The CERT-003
        // certificate_subject keccak is host-side and required by both
        // branches (Metal kernel echoes it; CPU branch echoes it via the
        // descriptor); compute it here once.
        const auto cert_mode_enum =
            static_cast<quasar::gpu::sig::QuasarCertMode>(round_.desc.cert_mode);
        auto subj = quasar::gpu::sig::compute_certificate_subject(
            round_.desc.chain_id, round_.desc.epoch, round_.desc.round,
            round_.desc.mode, cert_mode_enum,
            round_.desc.pchain_validator_root,
            round_.desc.parent_block_hash,
            round_.desc.xchain_execution_root,
            round_.desc.qchain_ceremony_root,
            round_.desc.zchain_vk_root,
            round_.desc.achain_state_root,
            round_.desc.bchain_state_root,
            round_.desc.mchain_state_root,
            round_.desc.fchain_state_root,
            round_.desc.attestation_root,
            round_.desc.parent_state_root,
            round_.desc.parent_execution_root,
            round_.desc.gas_limit, round_.desc.base_fee);
        std::memcpy(round_.desc.certificate_subject, subj.data(), 32);

        round_.handle = QuasarRoundHandle{++next_handle_};
        return round_.handle;
    }

    // v0.46.1 — late Metal init. Called from run_until_done when the
    // gate routes the round to the Metal substrate. Allocates 17 Metal
    // buffers, computes certificate_subject, populates descriptor, and
    // replays host_input_mirror / tx_replay into the Ingress ring.
    void init_metal_locked() {
        if (round_.metal_initialized) return;

        std::vector<uint64_t> per_service_offset(kNumServices, 0);
        uint64_t arena_bytes = 0;
        for (uint32_t s = 0; s < kNumServices; ++s) {
            per_service_offset[s] = arena_bytes;
            arena_bytes += static_cast<uint64_t>(kDefaultRingCapacity) * kItemSizes[s];
        }

        round_.desc_buf            = [device_ newBufferWithLength:sizeof(QuasarRoundDescriptor)        options:MTLResourceStorageModeShared];
        round_.result_buf          = [device_ newBufferWithLength:sizeof(QuasarRoundResult)            options:MTLResourceStorageModeShared];
        round_.hdrs_buf            = [device_ newBufferWithLength:sizeof(RingHeader) * kNumServices    options:MTLResourceStorageModeShared];
        round_.items_buf           = [device_ newBufferWithLength:arena_bytes                          options:MTLResourceStorageModeShared];
        round_.tx_index_buf        = [device_ newBufferWithLength:sizeof(uint32_t)                     options:MTLResourceStorageModeShared];
        round_.mvcc_buf            = [device_ newBufferWithLength:sizeof(MvccSlot) * kDefaultMvccSlots options:MTLResourceStorageModeShared];
        round_.dag_buf             = [device_ newBufferWithLength:sizeof(DagNodeHost) * kMaxFibers     options:MTLResourceStorageModeShared];
        round_.fibers_buf          = [device_ newBufferWithLength:sizeof(FiberSlot) * kMaxFibers       options:MTLResourceStorageModeShared];
        round_.mvcc_count_buf      = [device_ newBufferWithLength:sizeof(uint32_t)                     options:MTLResourceStorageModeShared];
        // v0.38 — vote-verifier output bitmap (one uint per vote ring slot).
        round_.vote_verified_buf   = [device_ newBufferWithLength:sizeof(uint32_t) * kDefaultRingCapacity options:MTLResourceStorageModeShared];
        round_.vote_capacity_buf   = [device_ newBufferWithLength:sizeof(uint32_t)                     options:MTLResourceStorageModeShared];
        // v0.40 — DAG construction buffers.
        round_.dag_writer_buf      = [device_ newBufferWithLength:32u * kDefaultDagWriterSlots         options:MTLResourceStorageModeShared];
        round_.dag_writer_count_buf= [device_ newBufferWithLength:sizeof(uint32_t)                     options:MTLResourceStorageModeShared];
        round_.predicted_buf       = [device_ newBufferWithLength:sizeof(PredictedKeyHost) * kPredictedSlotsPerRound options:MTLResourceStorageModeShared];
        round_.predicted_cap_buf   = [device_ newBufferWithLength:sizeof(uint32_t)                     options:MTLResourceStorageModeShared];
        round_.dag_node_cap_buf    = [device_ newBufferWithLength:sizeof(uint32_t)                     options:MTLResourceStorageModeShared];
        // v0.41 — EVM bytecode arena.
        round_.code_buf            = [device_ newBufferWithLength:kCodeArenaBytes                      options:MTLResourceStorageModeShared];
        round_.code_size_buf       = [device_ newBufferWithLength:sizeof(uint32_t)                     options:MTLResourceStorageModeShared];
        round_.fiber_cap_buf       = [device_ newBufferWithLength:sizeof(uint32_t)                     options:MTLResourceStorageModeShared];
        // v0.42 — CERT-001: 32-byte master secret derived from env.
        round_.cert_master_secret_buf =
            [device_ newBufferWithLength:32 options:MTLResourceStorageModeShared];

        if (!round_.desc_buf || !round_.result_buf || !round_.hdrs_buf
            || !round_.items_buf || !round_.tx_index_buf || !round_.mvcc_buf
            || !round_.dag_buf  || !round_.fibers_buf || !round_.mvcc_count_buf
            || !round_.vote_verified_buf || !round_.vote_capacity_buf
            || !round_.dag_writer_buf || !round_.dag_writer_count_buf
            || !round_.predicted_buf || !round_.predicted_cap_buf
            || !round_.dag_node_cap_buf
            || !round_.code_buf || !round_.code_size_buf || !round_.fiber_cap_buf
            || !round_.cert_master_secret_buf)
            return;  // v0.46.1: caller still holds a valid handle; Metal kernel
                     // dispatch will fail downstream if buffers couldn't allocate.

        // CERT-003 / v0.42 cert ABI: compute certificate_subject host-side and
        // write into desc. Both kernel and host MUST agree byte-for-byte; the
        // verifier compares v.subject == desc->certificate_subject and
        // rejects mismatch. v0.44 binds all 9 LP-134 chain roots in canonical
        // P, C, X, Q, Z, A, B, M, F order; v0.42 cert ABI further binds
        // attestation_root + cert_mode so a tampered TEE measurement or a
        // forged cert-mode forces a different cert subject.
        const auto cert_mode_enum =
            static_cast<quasar::gpu::sig::QuasarCertMode>(round_.desc.cert_mode);
        auto subj = quasar::gpu::sig::compute_certificate_subject(
            round_.desc.chain_id, round_.desc.epoch, round_.desc.round,
            round_.desc.mode, cert_mode_enum,
            round_.desc.pchain_validator_root,    // P
            round_.desc.parent_block_hash,        // C — this round's parent
            round_.desc.xchain_execution_root,    // X
            round_.desc.qchain_ceremony_root,     // Q
            round_.desc.zchain_vk_root,           // Z
            round_.desc.achain_state_root,        // A
            round_.desc.bchain_state_root,        // B
            round_.desc.mchain_state_root,        // M
            round_.desc.fchain_state_root,        // F
            round_.desc.attestation_root,
            round_.desc.parent_state_root,
            round_.desc.parent_execution_root,
            round_.desc.gas_limit, round_.desc.base_fee);
        std::memcpy(round_.desc.certificate_subject, subj.data(), 32);

        // CERT-001: load master secret from env (placeholder; KMS in v0.43).
        auto ms = quasar::gpu::sig::load_master_secret();
        std::memcpy([round_.cert_master_secret_buf contents], ms.data(), 32);

        std::memcpy([round_.desc_buf contents], &round_.desc, sizeof(QuasarRoundDescriptor));
        std::memset([round_.result_buf contents], 0, sizeof(QuasarRoundResult));
        std::memset([round_.items_buf contents], 0, arena_bytes);
        std::memset([round_.mvcc_buf contents], 0, sizeof(MvccSlot) * kDefaultMvccSlots);
        std::memset([round_.dag_buf contents], 0, sizeof(DagNodeHost) * kMaxFibers);
        std::memset([round_.fibers_buf contents], 0, sizeof(FiberSlot) * kMaxFibers);
        std::memset([round_.dag_writer_buf contents], 0, 32u * kDefaultDagWriterSlots);
        std::memset([round_.predicted_buf contents], 0, sizeof(PredictedKeyHost) * kPredictedSlotsPerRound);
        std::memset([round_.vote_verified_buf contents], 0, sizeof(uint32_t) * kDefaultRingCapacity);
        std::memset([round_.code_buf contents], 0, kCodeArenaBytes);
        *static_cast<uint32_t*>([round_.tx_index_buf contents]) = 0;
        *static_cast<uint32_t*>([round_.mvcc_count_buf contents]) = kDefaultMvccSlots;
        *static_cast<uint32_t*>([round_.vote_capacity_buf contents]) = kDefaultRingCapacity;
        *static_cast<uint32_t*>([round_.dag_writer_count_buf contents]) = kDefaultDagWriterSlots;
        *static_cast<uint32_t*>([round_.predicted_cap_buf contents]) = kDagNodeCapacity;
        *static_cast<uint32_t*>([round_.dag_node_cap_buf contents]) = kDagNodeCapacity;
        *static_cast<uint32_t*>([round_.code_size_buf contents]) = kCodeArenaBytes;
        *static_cast<uint32_t*>([round_.fiber_cap_buf contents]) = kMaxFibers;
        round_.next_predicted_idx = 0u;
        round_.next_code_offset = 0u;

        auto* result = static_cast<QuasarRoundResult*>([round_.result_buf contents]);
        result->mode = round_.desc.mode;
        // v0.44 — echo the 9 canonical chain roots + cert subject so consumers
        // can reconstruct the cert subject without re-parsing the descriptor.
        // Order matches compute_certificate_subject (P, C, X, Q, Z, A, B, M, F).
        std::memcpy(result->pchain_root_echo, round_.desc.pchain_validator_root, 32);
        std::memcpy(result->cchain_root_echo, round_.desc.parent_block_hash,     32);
        std::memcpy(result->xchain_root_echo, round_.desc.xchain_execution_root, 32);
        std::memcpy(result->qchain_root_echo, round_.desc.qchain_ceremony_root,  32);
        std::memcpy(result->zchain_root_echo, round_.desc.zchain_vk_root,        32);
        std::memcpy(result->achain_root_echo, round_.desc.achain_state_root,     32);
        std::memcpy(result->bchain_root_echo, round_.desc.bchain_state_root,     32);
        std::memcpy(result->mchain_root_echo, round_.desc.mchain_state_root,     32);
        std::memcpy(result->fchain_root_echo, round_.desc.fchain_state_root,     32);
        std::memcpy(result->certificate_subject_echo,
                    round_.desc.certificate_subject, 32);

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

        round_.metal_initialized = true;

        // Replay the host-side tx mirror into the now-initialized Metal
        // buffers. push_txs deferred all device writes until this point.
        push_txs_to_metal_locked();
    }

    // Drains round_.tx_replay into the Metal-backed Ingress ring. Called
    // from init_metal_locked once the Metal buffers exist. Mirrors the
    // original push_txs body byte-for-byte. Drains
    // [tx_replay_metal_cursor .. tx_replay.end()) so subsequent push_txs
    // calls after Metal init only mirror new txs.
    void push_txs_to_metal_locked() {
        if (!round_.metal_initialized) return;
        if (round_.tx_replay_metal_cursor >= round_.tx_replay.size()) return;

        auto* hdrs = static_cast<RingHeader*>([round_.hdrs_buf contents]);
        RingHeader& ingress = hdrs[static_cast<uint32_t>(ServiceId::Ingress)];
        auto* items = reinterpret_cast<IngressTx*>(
            static_cast<uint8_t*>([round_.items_buf contents]) + ingress.items_ofs);
        auto* predicted_arena = static_cast<PredictedKeyHost*>([round_.predicted_buf contents]);
        auto* code_arena      = static_cast<uint8_t*>([round_.code_buf contents]);

        const std::size_t start = round_.tx_replay_metal_cursor;
        for (std::size_t i = start; i < round_.tx_replay.size(); ++i) {
            const auto& tx = round_.tx_replay[i];
            round_.tx_replay_metal_cursor = i + 1;
            uint32_t head = ingress.head;
            uint32_t tail = ingress.tail;
            if (tail - head >= ingress.capacity) break;

            uint32_t code_off  = round_.next_code_offset;
            uint32_t code_size = static_cast<uint32_t>(tx.bytes.size());
            if (code_off + code_size > kCodeArenaBytes) {
                code_size = 0u;
                code_off  = 0u;
            } else if (code_size > 0u) {
                std::memcpy(code_arena + code_off, tx.bytes.data(), code_size);
                round_.next_code_offset += code_size;
            }

            IngressTx in{};
            in.blob_offset = code_off;
            in.blob_size   = code_size;
            in.gas_limit   = tx.gas_limit;
            in.nonce       = tx.nonce;
            uint32_t origin_lo = static_cast<uint32_t>(tx.origin & 0xFFFFFFFFu);
            uint32_t origin_hi = static_cast<uint32_t>(tx.origin >> 32);
            origin_hi &= 0x3FFFFFFFu;
            if (tx.needs_state) origin_hi |= 0x80000000u;
            if (tx.needs_exec)  origin_hi |= 0x40000000u;
            in.origin_lo = origin_lo;
            in.origin_hi = origin_hi;
            blob_arena_.insert(blob_arena_.end(), tx.bytes.begin(), tx.bytes.end());

            const uint32_t tidx = round_.next_predicted_idx;
            if (tidx < kDagNodeCapacity) {
                PredictedKeyHost* slot = &predicted_arena[tidx * kMaxPredictedKeys];
                for (uint32_t k = 0; k < kMaxPredictedKeys; ++k) {
                    slot[k] = PredictedKeyHost{};
                }
                const size_t n = std::min<size_t>(tx.predicted_access.size(), kMaxPredictedKeys);
                for (size_t k = 0; k < n; ++k) {
                    const auto& pa = tx.predicted_access[k];
                    slot[k].key_lo   = pa.key_lo;
                    slot[k].key_hi   = pa.key_hi;
                    slot[k].is_write = pa.is_write ? 1u : 0u;
                    slot[k].valid    = 1u;
                }
            }
            round_.next_predicted_idx += 1u;

            items[tail & ingress.mask] = in;
            std::atomic_thread_fence(std::memory_order_release);
            ingress.tail = tail + 1u;
            ingress.pushed += 1u;
        }
    }

    void push_txs(QuasarRoundHandle h, std::span<const HostTxBlob> txs) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return;
        if (txs.empty()) return;

        // v0.46.1 — buffer host-side. Cap at the substrate's per-round ingress
        // capacity (matches Metal-path behaviour). When Metal is initialized
        // the kernel sees these via push_txs_to_metal_locked; otherwise the
        // CPU branch in run_until_done consumes host_input_mirror directly.
        const std::size_t cap = static_cast<std::size_t>(kDefaultRingCapacity);
        for (const auto& tx : txs) {
            if (round_.host_input_mirror.size() >= cap) break;

            quasar::gpu::ref::HostInputTx mirror;
            mirror.gas_limit   = tx.gas_limit;
            mirror.origin      = tx.origin;
            mirror.needs_state = tx.needs_state;
            mirror.needs_exec  = tx.needs_exec;
            round_.host_input_mirror.push_back(mirror);
            round_.tx_replay.push_back(tx);
        }

        // If Metal has already been initialized (force-Metal env was set
        // before begin_round, or a previous run_until_done lazy-init'd it),
        // mirror the freshly accepted txs into device buffers immediately.
        if (round_.metal_initialized)
            push_txs_to_metal_locked();
    }

    void push_votes(QuasarRoundHandle h, std::span<const HostVote> votes) override {
        std::lock_guard<std::mutex> g(mu_);
        if (!check_handle(h)) return;
        if (votes.empty()) return;
        // v0.46.1 — vote ingestion needs Metal buffers; force lazy init.
        round_.requires_metal = true;
        init_metal_locked();

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
            out.sig_kind        = v.sig_kind;
            out.round           = v.round;          ///< CERT-021: uint64
            out.stake_weight    = v.stake_weight;   ///< CERT-006/007: uint64
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
        // No state requests can exist without a Metal kernel having run.
        if (!round_.metal_initialized) return out;

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
        // v0.46.1 — state-page ingestion presumes a Metal kernel will consume.
        round_.requires_metal = true;
        init_metal_locked();

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
        // No quorum certs without a Metal kernel having run.
        if (!round_.metal_initialized) return out;

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
        // v0.46.1 — Metal kernel dispatch requires initialized buffers.
        // Direct callers (skipping run_until_done's gate) get the Metal
        // path unconditionally; we lazy-init here.
        init_metal_locked();

        auto* desc_dev = static_cast<QuasarRoundDescriptor*>([round_.desc_buf contents]);
        desc_dev->wave_tick_index = round_.desc.wave_tick_index;
        desc_dev->closing_flag    = round_.desc.closing_flag;

        @autoreleasepool {
            // v0.38 — single command buffer, two encoders. Encoder 1 runs
            // the vote-batch verifier; encoder 2 runs the wave-tick
            // scheduler. Implicit ordering inside one MTLCommandBuffer
            // ensures the wave kernel sees the verifier's writes.
            id<MTLCommandBuffer> cmd = [queue_ commandBuffer];

            // Encoder 1: vote-batch verifier.
            id<MTLComputeCommandEncoder> venc = [cmd computeCommandEncoder];
            [venc setComputePipelineState:verify_pso_];
            [venc setBuffer:round_.desc_buf            offset:0 atIndex:0];
            [venc setBuffer:round_.hdrs_buf            offset:0 atIndex:1];
            [venc setBuffer:round_.items_buf           offset:0 atIndex:2];
            [venc setBuffer:round_.vote_verified_buf   offset:0 atIndex:3];
            [venc setBuffer:round_.vote_capacity_buf   offset:0 atIndex:4];
            // CERT-001 (v0.42) — master secret out of source.
            [venc setBuffer:round_.cert_master_secret_buf offset:0 atIndex:5];
            const NSUInteger vtpg = std::min<NSUInteger>(
                verify_pso_.maxTotalThreadsPerThreadgroup, 256);
            [venc dispatchThreads:MTLSizeMake(kDefaultRingCapacity, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(vtpg, 1, 1)];
            [venc endEncoding];

            // Encoder 2: wave-tick scheduler.
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pso_];
            [enc setBuffer:round_.desc_buf             offset:0 atIndex:0];
            [enc setBuffer:round_.result_buf           offset:0 atIndex:1];
            [enc setBuffer:round_.hdrs_buf             offset:0 atIndex:2];
            [enc setBuffer:round_.items_buf            offset:0 atIndex:3];
            [enc setBuffer:round_.tx_index_buf         offset:0 atIndex:4];
            [enc setBuffer:round_.mvcc_buf             offset:0 atIndex:5];
            [enc setBuffer:round_.dag_buf              offset:0 atIndex:6];
            [enc setBuffer:round_.fibers_buf           offset:0 atIndex:7];
            [enc setBuffer:round_.mvcc_count_buf       offset:0 atIndex:8];
            [enc setBuffer:round_.vote_verified_buf    offset:0 atIndex:9];
            [enc setBuffer:round_.vote_capacity_buf    offset:0 atIndex:10];
            [enc setBuffer:round_.dag_writer_buf       offset:0 atIndex:11];
            [enc setBuffer:round_.dag_writer_count_buf offset:0 atIndex:12];
            [enc setBuffer:round_.predicted_buf        offset:0 atIndex:13];
            [enc setBuffer:round_.predicted_cap_buf    offset:0 atIndex:14];
            [enc setBuffer:round_.dag_node_cap_buf     offset:0 atIndex:15];
            // v0.41 — EVM bytecode interpreter inputs.
            [enc setBuffer:round_.code_buf             offset:0 atIndex:16];
            [enc setBuffer:round_.code_size_buf        offset:0 atIndex:17];
            [enc setBuffer:round_.fiber_cap_buf        offset:0 atIndex:18];

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
        // v0.46.1 — substrate-threshold gate. At small N the Metal substrate's
        // fixed per-round setup cost dominates; the CPU reference produces
        // byte-equal output (asserted by quasar_stm_red_review_test) at a
        // fraction of the wall-clock. Above the threshold, fall through to
        // the Metal path. LUX_QUASAR_FORCE_METAL=1 bypasses the gate for
        // bench reproducibility.
        {
            std::lock_guard<std::mutex> g(mu_);
            if (!check_handle(h)) return QuasarRoundResult{};
            if (substrate_should_use_cpu_locked()) {
                QuasarRoundResult packed{};
                run_cpu_reference_locked(packed);
                return packed;
            }
            // Above threshold: lazy-init Metal buffers + replay mirror.
            init_metal_locked();
        }

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
        // v0.46.1 — when the gate took the CPU branch, ring stats reflect
        // the host-side mirror: ingress was pushed N times, consumed N
        // times by the CPU reference, capacity is the substrate's nominal
        // ring capacity.
        if (!round_.metal_initialized) {
            const uint32_t n = static_cast<uint32_t>(round_.host_input_mirror.size());
            if (s == ServiceId::Ingress) {
                out.pushed   = n;
                out.consumed = n;
                out.head     = n;
                out.tail     = n;
            }
            out.capacity = kDefaultRingCapacity;
            return out;
        }
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
        // v0.46.1 — CPU branch: return cached result. Metal branch: read
        // from device-shared result_buf.
        if (!round_.metal_initialized) {
            return round_.cached_cpu_result_valid
                ? round_.cached_cpu_result : QuasarRoundResult{};
        }
        QuasarRoundResult out{};
        std::memcpy(&out, [round_.result_buf contents], sizeof(QuasarRoundResult));
        return out;
    }

    // v0.46.1 — substrate-threshold gate. Returns true when the round
    // should fall through to the CPU reference instead of Metal. Conditions:
    //   1. count < kQuasarSubstrateMetalThreshold (perf — Metal is slower
    //      below this); AND
    //   2. no tx has EVM bytecode AND no votes have been pushed — the CPU
    //      reference only models the v0.36 synthetic substrate (one R+W
    //      pair per tx + Block-STM + receipts/state/execution roots). EVM
    //      bytecode interpretation (v0.41) and on-device vote verification
    //      (v0.38) are Metal-only. Routing those workloads to CPU would
    //      diverge from Metal output and fail STM-004 byte-equality.
    //   3. LUX_QUASAR_FORCE_METAL=1 unset (bench reproducibility override).
    bool substrate_should_use_cpu_locked() const {
        if (const char* force = std::getenv("LUX_QUASAR_FORCE_METAL")) {
            if (force[0] == '1') return false;
        }
        // Vote / state-page ingestion poisons the CPU branch: the CPU
        // reference doesn't model BLS / Ringtail vote verification, QC
        // composition, or cold-state requests. push_votes / push_state_pages
        // set requires_metal so the gate falls through to Metal.
        if (round_.requires_metal) return false;
        const std::size_t n = round_.host_input_mirror.size();
        if (n >= kQuasarSubstrateMetalThreshold) return false;
        // Any tx with bytecode forces the Metal path: the CPU reference
        // doesn't interpret EVM bytecode (substrate-only — synthetic R+W
        // per tx + receipts / state / execution roots).
        for (const auto& tx : round_.tx_replay) {
            if (!tx.bytes.empty()) return false;
        }
        return true;
    }

    // Pack the CPU reference result into a QuasarRoundResult and write it
    // to round_.result_buf so poll_round_result and ring_stats remain
    // consistent. The CPU reference fills the four substrate roots
    // (block, state, receipts, execution, mode) plus tx_count, gas, and
    // status; the chain-echo fields are taken from the descriptor (same
    // path as begin_round populates them on the Metal side).
    void run_cpu_reference_locked(QuasarRoundResult& out) {
        auto cpu = quasar::gpu::ref::run_reference(
            round_.desc, std::span<const quasar::gpu::ref::HostInputTx>(
                round_.host_input_mirror));

        std::memset(&out, 0, sizeof(out));
        out.status         = cpu.status;
        out.tx_count       = cpu.tx_count;
        out.gas_used_lo    = static_cast<uint32_t>(cpu.gas_used & 0xFFFFFFFFu);
        out.gas_used_hi    = static_cast<uint32_t>(cpu.gas_used >> 32);
        out.conflict_count = cpu.conflict_count;
        out.repair_count   = cpu.repair_count;
        out.mode           = cpu.mode;
        std::memcpy(out.block_hash,     cpu.block_hash,     32);
        std::memcpy(out.state_root,     cpu.state_root,     32);
        std::memcpy(out.receipts_root,  cpu.receipts_root,  32);
        std::memcpy(out.execution_root, cpu.execution_root, 32);
        std::memcpy(out.mode_root,      cpu.mode_root,      32);

        // Echo the 9 canonical chain roots + cert subject from the descriptor
        // (Metal path does the same in begin_round; keep both branches in lockstep).
        std::memcpy(out.pchain_root_echo,           round_.desc.pchain_validator_root,    32);
        std::memcpy(out.cchain_root_echo,           round_.desc.parent_block_hash,        32);
        std::memcpy(out.xchain_root_echo,           round_.desc.xchain_execution_root,    32);
        std::memcpy(out.qchain_root_echo,           round_.desc.qchain_ceremony_root,     32);
        std::memcpy(out.zchain_root_echo,           round_.desc.zchain_vk_root,           32);
        std::memcpy(out.achain_root_echo,           round_.desc.achain_state_root,        32);
        std::memcpy(out.bchain_root_echo,           round_.desc.bchain_state_root,        32);
        std::memcpy(out.mchain_root_echo,           round_.desc.mchain_state_root,        32);
        std::memcpy(out.fchain_root_echo,           round_.desc.fchain_state_root,        32);
        std::memcpy(out.certificate_subject_echo,   round_.desc.certificate_subject,      32);

        // Cache for poll_round_result on the CPU branch. Metal result_buf is
        // not allocated; cached_cpu_result is the only source of truth.
        round_.cached_cpu_result = out;
        round_.cached_cpu_result_valid = true;
    }

    id<MTLDevice>               device_;
    id<MTLCommandQueue>         queue_;
    id<MTLComputePipelineState> pso_;
    id<MTLComputePipelineState> verify_pso_;
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
        if (!pso) {
            std::fprintf(stderr, "quasar.create: wave_kernel PSO failed: %s\n",
                         err ? [[err localizedDescription] UTF8String] : "(null)");
            return nullptr;
        }

        // v0.38 — vote-batch verifier pipeline (separate compute encoder
        // in the same MTLCommandBuffer as the wave-tick kernel).
        id<MTLFunction> vfn = [lib newFunctionWithName:@"quasar_verify_votes_kernel"];
        if (!vfn) return nullptr;
        id<MTLComputePipelineState> verify_pso =
            [device newComputePipelineStateWithFunction:vfn error:&err];
        if (!verify_pso) {
            std::fprintf(stderr, "quasar.create: verify PSO failed: %s\n",
                         err ? [[err localizedDescription] UTF8String] : "(null)");
            return nullptr;
        }

        return std::make_unique<QuasarGPUEngineImpl>(device, queue, pso, verify_pso, [device name]);
    }
}

}  // namespace quasar::gpu
