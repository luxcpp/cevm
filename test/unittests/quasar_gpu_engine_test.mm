// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file quasar_gpu_engine_test.mm
/// Tests for QuasarGPUEngine — CEVM's GPU execution adapter for Quasar
/// rounds. Coverage:
///
///   1. Empty round
///   2. Single-tx pipeline + real keccak (block_hash, receipts_root,
///      execution_root all non-zero and deterministic)
///   3. Multi-tx counter monotonicity
///   4. Bounded backpressure across wave ticks
///   5. End-to-end stress (1024 txs)
///   6. Cold-state page-fault round trip
///   7. Real receipts_root chain — verify deterministic across runs
///   8. Per-lane Quasar quorum aggregation (BLS / ML-DSA / Ringtail
///      stubs accumulate stake; emit QC at 2/3 threshold; host drains
///      QuorumOut)

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "consensus/quasar/gpu/quasar_gpu_engine.hpp"
#include "consensus/quasar/gpu/quasar_sig.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

using quasar::gpu::HostQuorumCert;
using quasar::gpu::HostStatePage;
using quasar::gpu::HostStateRequest;
using quasar::gpu::HostTxBlob;
using quasar::gpu::HostVote;
using quasar::gpu::QuasarGPUEngine;
using quasar::gpu::QuasarRoundDescriptor;
using quasar::gpu::QuasarRoundResult;
using quasar::gpu::ServiceId;

namespace {

int g_passed = 0;
int g_failed = 0;

#define EXPECT(name, cond)                                                  \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::printf("  FAIL[%s]: %s\n", (name), #cond);                 \
            std::fflush(stdout);                                            \
            ++g_failed;                                                     \
            return;                                                         \
        }                                                                   \
    } while (0)

#define PASS(name)                                                          \
    do {                                                                    \
        std::printf("  ok  : %s\n", (name));                                \
        std::fflush(stdout);                                                \
        ++g_passed;                                                         \
    } while (0)

QuasarRoundDescriptor make_desc(uint64_t round, uint32_t mode = 0)
{
    QuasarRoundDescriptor d{};
    d.chain_id = 1u;
    d.round = round;
    d.timestamp_ns = 0u;
    d.deadline_ns = 0u;
    d.gas_limit = 30'000'000u;
    d.base_fee = 100u;       // total stake for v0.37 quorum tests
    d.wave_tick_budget = 256u;
    d.mode = mode;
    return d;
}

HostTxBlob make_tx(uint64_t origin, uint32_t nonce)
{
    HostTxBlob t;
    t.gas_limit = 21'000u;
    t.nonce = nonce;
    t.origin = origin;
    return t;
}

bool is_zero32(const uint8_t* h)
{
    for (int i = 0; i < 32; ++i) if (h[i] != 0) return false;
    return true;
}

// 1. Empty round
void test_empty_round()
{
    auto e = QuasarGPUEngine::create();
    EXPECT("empty.create", e != nullptr);
    auto h = e->begin_round(make_desc(1));
    EXPECT("empty.handle", h.valid());
    e->request_close(h);
    auto r = e->run_until_done(h, 8);
    EXPECT("empty.finalized", r.status == 1u);
    EXPECT("empty.tx_count_zero", r.tx_count == 0u);
    e->end_round(h);
    PASS("empty_round");
}

// 2. Single-tx flow with real keccak roots
void test_single_tx_real_roots()
{
    auto e = QuasarGPUEngine::create();
    auto h = e->begin_round(make_desc(2));
    EXPECT("one.handle", h.valid());

    std::vector<HostTxBlob> txs = { make_tx(0xCAFEu, 0u) };
    e->push_txs(h, txs);
    e->request_close(h);
    auto r = e->run_until_done(h, 16);
    EXPECT("one.finalized", r.status == 1u);
    EXPECT("one.tx_count", r.tx_count == 1u);
    EXPECT("one.gas",     r.gas_used() == 21'000u);

    EXPECT("one.block_hash_nonzero",     !is_zero32(r.block_hash));
    EXPECT("one.receipts_root_nonzero",  !is_zero32(r.receipts_root));
    EXPECT("one.execution_root_nonzero", !is_zero32(r.execution_root));

    e->end_round(h);
    PASS("single_tx_real_roots");
}

// 3. Multi-tx counter monotonicity
void test_multi_tx_counters()
{
    auto e = QuasarGPUEngine::create();
    auto h = e->begin_round(make_desc(3));
    EXPECT("multi.handle", h.valid());

    constexpr size_t N = 128;
    std::vector<HostTxBlob> txs;
    for (size_t i = 0; i < N; ++i) txs.push_back(make_tx(i + 1, uint32_t(i)));
    e->push_txs(h, txs);
    e->request_close(h);
    auto r = e->run_until_done(h, 32);
    EXPECT("multi.finalized", r.status == 1u);
    EXPECT("multi.tx_count", r.tx_count == N);
    EXPECT("multi.gas_total", r.gas_used() == N * 21'000u);

    for (auto sid : {ServiceId::Ingress, ServiceId::Decode,
                     ServiceId::Crypto, ServiceId::Commit}) {
        auto s = e->ring_stats(h, sid);
        if (s.consumed != s.pushed) {
            std::printf("  multi mismatch id=%u pushed=%u consumed=%u\n",
                        unsigned(sid), s.pushed, s.consumed);
            ++g_failed; return;
        }
    }
    e->end_round(h);
    PASS("multi_tx_counters");
}

// 4. Bounded backpressure
void test_bounded_backpressure()
{
    auto e = QuasarGPUEngine::create();
    auto d = make_desc(4);
    d.wave_tick_budget = 64;
    auto h = e->begin_round(d);

    constexpr size_t N = 1024;
    std::vector<HostTxBlob> txs;
    for (size_t i = 0; i < N; ++i) txs.push_back(make_tx(i + 1, uint32_t(i)));

    e->push_txs(h, std::span<const HostTxBlob>(txs.data(), N/2));
    for (int i = 0; i < 4; ++i) e->run_epoch(h);
    auto mid = e->ring_stats(h, ServiceId::Commit);
    EXPECT("bp.partial_drain", mid.consumed > 0u);

    e->push_txs(h, std::span<const HostTxBlob>(txs.data() + N/2, N/2));
    e->request_close(h);
    auto r = e->run_until_done(h, 64);
    EXPECT("bp.finalized", r.status == 1u);
    EXPECT("bp.tx_count", r.tx_count == N);
    EXPECT("bp.gas_total", r.gas_used() == N * 21'000u);

    e->end_round(h);
    PASS("bounded_backpressure");
}

// 5. End-to-end stress
void test_end_to_end_stress()
{
    auto e = QuasarGPUEngine::create();
    auto h = e->begin_round(make_desc(5));

    constexpr size_t N = 1024;
    std::vector<HostTxBlob> txs;
    for (size_t i = 0; i < N; ++i) txs.push_back(make_tx(i + 1, uint32_t(i)));

    auto t0 = std::chrono::steady_clock::now();
    e->push_txs(h, txs);
    e->request_close(h);
    auto r = e->run_until_done(h, 64);
    auto t1 = std::chrono::steady_clock::now();

    EXPECT("e2e.finalized", r.status == 1u);
    EXPECT("e2e.tx_count", r.tx_count == N);
    EXPECT("e2e.gas_total", r.gas_used() == N * 21'000u);

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::printf("  e2e: %zu txs across %u wave ticks in %lld ms\n",
                N, r.wave_tick_count, (long long)ms);
    std::fflush(stdout);

    e->end_round(h);
    PASS("end_to_end_stress");
}

// 6. Cold-state page-fault round trip
void test_state_page_fault()
{
    auto e = QuasarGPUEngine::create();
    auto h = e->begin_round(make_desc(6));

    constexpr size_t N_FAST = 64;
    constexpr size_t N_SLOW = 16;
    std::vector<HostTxBlob> txs;
    for (size_t i = 0; i < N_FAST; ++i) txs.push_back(make_tx(i + 1, uint32_t(i)));
    for (size_t i = 0; i < N_SLOW; ++i) {
        auto t = make_tx(0xA0000 + i + 1, uint32_t(N_FAST + i));
        t.needs_state = true;
        txs.push_back(t);
    }
    e->push_txs(h, txs);

    e->request_close(h);
    QuasarRoundResult r{};
    size_t serviced = 0;
    for (int epoch = 0; epoch < 64; ++epoch) {
        r = e->run_epoch(h);
        auto reqs = e->poll_state_requests(h);
        if (!reqs.empty()) {
            std::vector<HostStatePage> pages;
            pages.reserve(reqs.size());
            for (const auto& q : reqs) {
                HostStatePage p{};
                p.tx_index = q.tx_index;
                p.key_type = q.key_type;
                p.status   = 0;
                p.key_lo   = q.key_lo;
                p.key_hi   = q.key_hi;
                p.data.assign({0xDE, 0xAD, 0xBE, 0xEF});
                pages.push_back(std::move(p));
            }
            e->push_state_pages(h, pages);
            serviced += pages.size();
        }
        if (r.status != 0u) break;
    }
    EXPECT("sf.finalized", r.status == 1u);
    EXPECT("sf.tx_total",  r.tx_count == N_FAST + N_SLOW);
    EXPECT("sf.serviced",  serviced == N_SLOW);
    auto rs = e->ring_stats(h, ServiceId::StateRequest);
    EXPECT("sf.req_consumed", rs.consumed == N_SLOW);
    e->end_round(h);
    PASS("state_page_fault");
}

// 7. Determinism — same input yields same roots across two runs.
void test_root_determinism()
{
    QuasarRoundResult r1{}, r2{};
    for (int run = 0; run < 2; ++run) {
        auto e = QuasarGPUEngine::create();
        auto h = e->begin_round(make_desc(7));
        std::vector<HostTxBlob> txs;
        for (uint32_t i = 0; i < 16; ++i) txs.push_back(make_tx(i + 1, i));
        e->push_txs(h, txs);
        e->request_close(h);
        auto r = e->run_until_done(h, 16);
        if (run == 0) r1 = r; else r2 = r;
        e->end_round(h);
    }
    EXPECT("det.bh",  std::memcmp(r1.block_hash,     r2.block_hash,     32) == 0);
    EXPECT("det.rr",  std::memcmp(r1.receipts_root,  r2.receipts_root,  32) == 0);
    EXPECT("det.er",  std::memcmp(r1.execution_root, r2.execution_root, 32) == 0);
    EXPECT("det.tx_count", r1.tx_count == r2.tx_count);
    EXPECT("det.gas",      r1.gas_used() == r2.gas_used());
    PASS("root_determinism");
}

// v0.35/v0.36: Block-STM full round-trip (Crypto → DagReady →
// Exec → Validate → Commit). Independent txs commit in parallel without
// conflicts; concurrent same-key txs trigger conflict_count + repair_count
// telemetry, and all eventually commit.
void test_block_stm_independent_txs()
{
    auto e = QuasarGPUEngine::create();
    auto desc = make_desc(20);
    desc.mode = 0;        // Nova; Crypto routes straight to Exec
    auto h = e->begin_round(desc);

    constexpr size_t N = 64;
    std::vector<HostTxBlob> txs;
    for (size_t i = 0; i < N; ++i) {
        auto t = make_tx(i + 0x10000, uint32_t(i));
        t.needs_exec = true;       // independent keys → no conflicts
        txs.push_back(t);
    }
    e->push_txs(h, txs);
    e->request_close(h);
    auto r = e->run_until_done(h, 32);

    EXPECT("bstm_indep.finalized", r.status == 1u);
    EXPECT("bstm_indep.tx_count",  r.tx_count == N);
    EXPECT("bstm_indep.no_conflicts", r.conflict_count == 0u);
    EXPECT("bstm_indep.no_repairs",   r.repair_count   == 0u);
    e->end_round(h);
    PASS("block_stm_independent_txs");
}

void test_block_stm_conflict_repair()
{
    auto e = QuasarGPUEngine::create();
    auto desc = make_desc(21);
    desc.mode = 1;        // Nebula path: Crypto → DagReady → Exec → ...
    auto h = e->begin_round(desc);

    // All txs target the SAME exec_key — every tx after the first sees
    // a stale version_seen at validate time → repair → re-exec.
    constexpr size_t N = 16;
    std::vector<HostTxBlob> txs;
    for (size_t i = 0; i < N; ++i) {
        auto t = make_tx(0xCAFEDEAD, uint32_t(i));
        t.needs_exec = true;
        txs.push_back(t);
    }
    e->push_txs(h, txs);
    e->request_close(h);
    auto r = e->run_until_done(h, 256);

    EXPECT("bstm_conflict.finalized", r.status == 1u);
    EXPECT("bstm_conflict.tx_count",  r.tx_count == N);
    // Under contention there must be at least one conflict and one
    // repair — the substrate proves Block-STM is real on GPU.
    if (r.conflict_count == 0u || r.repair_count == 0u) {
        std::printf("  bstm_conflict: expected conflicts/repairs > 0; "
                    "got conflicts=%u repairs=%u\n",
                    r.conflict_count, r.repair_count);
        ++g_failed; return;
    }
    std::printf("  bstm_conflict: %zu txs, conflicts=%u repairs=%u\n",
                N, r.conflict_count, r.repair_count);
    e->end_round(h);
    PASS("block_stm_conflict_repair");
}

// 8. Quasar quorum aggregation per lane - v0.38 real verifier.
void test_quasar_quorum_round_trip()
{
    auto e = QuasarGPUEngine::create();
    auto desc = make_desc(8);
    desc.chain_id = 1u;
    desc.base_fee = 90u;
    auto h = e->begin_round(desc);

    std::vector<HostTxBlob> txs = { make_tx(0xBABEu, 0u) };
    e->push_txs(h, txs);
    e->request_close(h);
    auto r = e->run_until_done(h, 8);
    EXPECT("q.finalized", r.status == 1u);

    uint8_t subject[32];
    std::memcpy(subject, r.block_hash, 32);
    const uint32_t round = 8u;

    auto mk = [&](uint32_t vi, uint32_t weight, uint32_t kind) {
        HostVote v{};
        v.validator_index = vi;
        v.round           = round;
        v.stake_weight    = weight;
        v.sig_kind        = kind;
        std::memcpy(v.block_hash, subject, 32);
        v.signature = quasar::gpu::sig::sign(kind, desc.chain_id, vi, round, subject);
        return v;
    };

    std::vector<HostVote> votes;
    votes.push_back(mk(0, 30, 0));
    votes.push_back(mk(1, 30, 0));
    votes.push_back(mk(2, 30, 0));
    HostVote bad_bls = mk(3, 60, 0);
    bad_bls.signature[0] ^= 0xFFu;
    votes.push_back(bad_bls);

    votes.push_back(mk(0, 10, 2));
    HostVote replay = mk(4, 80, 0);
    replay.sig_kind = 2;
    votes.push_back(replay);

    votes.push_back(mk(0, 30, 1));
    votes.push_back(mk(1, 30, 1));
    e->push_votes(h, votes);

    auto r2 = e->run_epoch(h);
    auto certs = e->poll_quorum_certs(h);

    std::printf("  q.stake bls=%u rt=%u mldsa=%u certs=%zu\n",
                r2.quorum_stake_bls, r2.quorum_stake_rt,
                r2.quorum_stake_mldsa, certs.size());

    EXPECT("q.bls_quorum",   r2.quorum_status_bls   == 1u);
    EXPECT("q.bls_stake_excludes_tamper", r2.quorum_stake_bls    == 90u);
    EXPECT("q.rt_quorum",    r2.quorum_status_rt    == 1u);
    EXPECT("q.rt_stake",     r2.quorum_stake_rt     == 60u);
    EXPECT("q.mldsa_no_q",   r2.quorum_status_mldsa == 0u);
    EXPECT("q.mldsa_stake_excludes_replay", r2.quorum_stake_mldsa == 10u);
    EXPECT("q.certs_count",  certs.size() == 2u);
    e->end_round(h);
    PASS("quasar_quorum_round_trip");
}

}  // namespace

int main(int /*argc*/, char** /*argv*/)
{
    setvbuf(stdout, nullptr, _IOLBF, 0);
    @autoreleasepool {
        std::printf("[quasar_gpu_engine_test] starting\n");
        std::fflush(stdout);
        test_empty_round();
        test_single_tx_real_roots();
        test_multi_tx_counters();
        test_bounded_backpressure();
        test_end_to_end_stress();
        test_state_page_fault();
        test_root_determinism();
        test_block_stm_independent_txs();
        test_block_stm_conflict_repair();
        test_quasar_quorum_round_trip();
        std::printf("[quasar_gpu_engine_test] passed=%d failed=%d\n",
                    g_passed, g_failed);
        return g_failed == 0 ? 0 : 1;
    }
}
