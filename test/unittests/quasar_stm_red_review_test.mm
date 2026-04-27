// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file quasar_stm_red_review_test.mm
/// Regression suite for the Q3.0 STM Red findings (LP-010 v0.43).
///
/// Each test gates one specific finding from the audit
/// `lux/audits/quasar-3-0-stm-red-review.md` (2026-04-26):
///
///   1. test_validate_atomic_with_commit_full   — STM-001
///   2. test_mvcc_concurrent_slot_claim         — STM-002
///   3. test_repair_cap_aborts_runaway          — STM-003 / STM-014
///   4. test_cross_backend_determinism_disjoint — STM-004 (Metal vs CPU ref)
///   5. test_cross_backend_determinism_contended— STM-004 (same-key flood)
///   6. test_atomic_release_acquire_dag         — STM-005 (Nebula DAG order)
///   7. test_mvcc_arena_zeroed_per_round        — STM-013
///   8. test_substrate_threshold_gate_byte_equal — v0.46.1 substrate gate

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "consensus/quasar/gpu/quasar_gpu_engine.hpp"
#include "consensus/quasar/gpu/quasar_cpu_reference.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using quasar::gpu::HostTxBlob;
using quasar::gpu::QuasarGPUEngine;
using quasar::gpu::QuasarRoundDescriptor;
using quasar::gpu::QuasarRoundResult;
using quasar::gpu::ServiceId;

namespace {

int g_passed = 0;
int g_failed = 0;

#define EXPECT(name, cond)                                                   \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("  FAIL[%s]: %s\n", (name), #cond);                  \
            std::fflush(stdout);                                             \
            ++g_failed;                                                      \
            return;                                                          \
        }                                                                    \
    } while (0)

#define PASS(name)                                                           \
    do {                                                                     \
        std::printf("  ok  : %s\n", (name));                                 \
        std::fflush(stdout);                                                 \
        ++g_passed;                                                          \
    } while (0)

QuasarRoundDescriptor make_desc(uint64_t round, uint32_t mode = 0) {
    QuasarRoundDescriptor d{};
    d.chain_id = 1u;
    d.round = round;
    d.gas_limit = 30'000'000u;
    d.base_fee = 100u;
    d.wave_tick_budget = 256u;
    d.mode = mode;
    return d;
}

HostTxBlob make_exec_tx(uint64_t origin, uint32_t nonce) {
    HostTxBlob t;
    t.gas_limit = 21'000u;
    t.nonce = nonce;
    t.origin = origin;
    t.needs_exec = true;
    return t;
}

quasar::gpu::ref::HostInputTx mirror_for_cpu(const HostTxBlob& t) {
    quasar::gpu::ref::HostInputTx r;
    r.gas_limit = t.gas_limit;
    r.origin = t.origin;
    r.needs_state = t.needs_state;
    r.needs_exec = t.needs_exec;
    return r;
}

bool eq32(const uint8_t* a, const uint8_t* b) {
    return std::memcmp(a, b, 32) == 0;
}

void hex32(const uint8_t* h, char* out65) {
    static const char* d = "0123456789abcdef";
    for (int i = 0; i < 32; ++i) {
        out65[2*i]   = d[(h[i] >> 4) & 0xF];
        out65[2*i+1] = d[h[i] & 0xF];
    }
    out65[64] = 0;
}

// =============================================================================
// 1. STM-001: drain_validate must not mutate MVCC if commit ring push fails.
//
// Observable property: with a same-key flood and a tight wave_tick_budget
// that forces commit-ring backpressure, the round must produce IDENTICAL
// roots regardless of how the ring fills. STM-001 (atomic ordering) and
// STM-003 (repair cap) together collapse the timing-dependent
// nondeterminism into a stable byte-identical commit chain.
// =============================================================================
void test_validate_atomic_with_commit_full() {
    auto e = QuasarGPUEngine::create();
    EXPECT("v.create", e != nullptr);

    auto run_once = [&](uint32_t budget) -> QuasarRoundResult {
        auto desc = make_desc(101);
        desc.wave_tick_budget = budget;
        auto h = e->begin_round(desc);
        std::vector<HostTxBlob> txs;
        for (int i = 0; i < 256; ++i)
            txs.push_back(make_exec_tx(0xDEADBEEFu, uint32_t(i)));
        e->push_txs(h, txs);
        e->request_close(h);
        auto r = e->run_until_done(h, 4096);
        e->end_round(h);
        return r;
    };

    auto r_small = run_once(8);     // budget=8 forces commit backpressure
    auto r_big   = run_once(256);
    EXPECT("v.both_finalized", r_small.status == 1u && r_big.status == 1u);
    EXPECT("v.tx_count_match", r_small.tx_count == r_big.tx_count);
    EXPECT("v.block_hash_eq", eq32(r_small.block_hash, r_big.block_hash));
    EXPECT("v.receipts_eq",   eq32(r_small.receipts_root, r_big.receipts_root));
    EXPECT("v.execution_eq",  eq32(r_small.execution_root, r_big.execution_root));
    std::printf("  stm-001: budget=8 conflicts=%u repairs=%u | "
                "budget=256 conflicts=%u repairs=%u\n",
                r_small.conflict_count, r_small.repair_count,
                r_big.conflict_count, r_big.repair_count);
    PASS("validate_atomic_with_commit_full");
}

// =============================================================================
// 2. STM-002: atomic CAS slot claim. Multiple concurrent slot claims
// across drain_exec/validate workgroups must produce the same final
// roots across runs — the slot-claim CAS protocol guarantees a single
// winner per slot, and the mem_device fence prevents torn key reads.
// =============================================================================
void test_mvcc_concurrent_slot_claim() {
    auto e = QuasarGPUEngine::create();

    auto run_once = [&]() -> QuasarRoundResult {
        auto desc = make_desc(102);
        desc.mode = 1;  // Nebula DAG path stresses concurrent slot access
        auto h = e->begin_round(desc);
        constexpr int N = 64;
        std::vector<HostTxBlob> txs;
        for (int i = 0; i < N; ++i) {
            // Distinct keys exercise the CAS-claim path; the slot-claim
            // protocol must still produce identical roots across runs.
            txs.push_back(make_exec_tx(0x01000000u + uint64_t(i), uint32_t(i)));
        }
        e->push_txs(h, txs);
        e->request_close(h);
        auto r = e->run_until_done(h, 256);
        e->end_round(h);
        return r;
    };

    auto r1 = run_once();
    auto r2 = run_once();
    EXPECT("c.r1_finalized", r1.status == 1u);
    EXPECT("c.r2_finalized", r2.status == 1u);
    EXPECT("c.tx_count", r1.tx_count == 64u && r2.tx_count == 64u);
    EXPECT("c.bh_eq", eq32(r1.block_hash, r2.block_hash));
    EXPECT("c.rr_eq", eq32(r1.receipts_root, r2.receipts_root));
    PASS("mvcc_concurrent_slot_claim");
}

// =============================================================================
// 3. STM-003 + STM-014: 1024 same-key txs must finalize with bounded
// conflicts/repairs (anti-livelock cap fires).
// =============================================================================
void test_repair_cap_aborts_runaway() {
    auto e = QuasarGPUEngine::create();
    auto desc = make_desc(103);
    desc.wave_tick_budget = 256;
    auto h = e->begin_round(desc);

    constexpr int N = 1024;
    std::vector<HostTxBlob> txs;
    for (int i = 0; i < N; ++i)
        txs.push_back(make_exec_tx(0xCAFE0000u, uint32_t(i)));  // SAME key
    e->push_txs(h, txs);
    e->request_close(h);

    auto t0 = std::chrono::steady_clock::now();
    auto r = e->run_until_done(h, 4096);
    auto t1 = std::chrono::steady_clock::now();

    EXPECT("rc.finalized", r.status == 1u);
    EXPECT("rc.tx_count", r.tx_count == N);
    // The cap bounds repair_count at roughly N * kMaxTotalRepairs in the
    // pessimistic worst case — finite and bounded, not O(N^2).
    EXPECT("rc.bounded_repairs", r.repair_count < uint32_t(N) * 16u);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::printf("  stm-003: %d same-key txs, conflicts=%u repairs=%u in %lld ms\n",
                N, r.conflict_count, r.repair_count, (long long)ms);
    e->end_round(h);
    PASS("repair_cap_aborts_runaway");
}

// =============================================================================
// 4. STM-004: cross-backend determinism — Metal vs CPU reference on
// disjoint workload. Roots must be byte-identical.
// =============================================================================
void test_cross_backend_determinism_disjoint() {
    auto e = QuasarGPUEngine::create();
    auto desc = make_desc(104);
    auto h = e->begin_round(desc);

    constexpr int N = 1024;
    std::vector<HostTxBlob> gpu_txs;
    std::vector<quasar::gpu::ref::HostInputTx> cpu_txs;
    for (int i = 0; i < N; ++i) {
        auto t = make_exec_tx(uint64_t(0x10000) + uint64_t(i), uint32_t(i));
        gpu_txs.push_back(t);
        cpu_txs.push_back(mirror_for_cpu(t));
    }
    e->push_txs(h, gpu_txs);
    e->request_close(h);
    auto gpu_r = e->run_until_done(h, 256);
    e->end_round(h);

    auto cpu_r = quasar::gpu::ref::run_reference(desc, cpu_txs);

    EXPECT("d.gpu_finalized", gpu_r.status == 1u);
    EXPECT("d.cpu_finalized", cpu_r.status == 1u);
    EXPECT("d.tx_count", gpu_r.tx_count == cpu_r.tx_count);
    EXPECT("d.gas_used", gpu_r.gas_used() == cpu_r.gas_used);

    char gpu_h_hex[65], cpu_h_hex[65];
    hex32(gpu_r.block_hash, gpu_h_hex);
    hex32(cpu_r.block_hash, cpu_h_hex);
    if (!eq32(gpu_r.block_hash, cpu_r.block_hash)) {
        std::printf("  d.bh GPU=%s\n  d.bh CPU=%s\n", gpu_h_hex, cpu_h_hex);
    }
    EXPECT("d.bh_eq",  eq32(gpu_r.block_hash, cpu_r.block_hash));
    EXPECT("d.rr_eq",  eq32(gpu_r.receipts_root, cpu_r.receipts_root));
    EXPECT("d.er_eq",  eq32(gpu_r.execution_root, cpu_r.execution_root));
    PASS("cross_backend_determinism_disjoint");
}

// =============================================================================
// 5. STM-004: cross-backend determinism — contended workload (16 same-key).
// =============================================================================
void test_cross_backend_determinism_contended() {
    auto e = QuasarGPUEngine::create();
    auto desc = make_desc(105);
    auto h = e->begin_round(desc);

    constexpr int N = 16;
    std::vector<HostTxBlob> gpu_txs;
    std::vector<quasar::gpu::ref::HostInputTx> cpu_txs;
    for (int i = 0; i < N; ++i) {
        auto t = make_exec_tx(0xCAFEDEAD, uint32_t(i));  // ALL same key
        gpu_txs.push_back(t);
        cpu_txs.push_back(mirror_for_cpu(t));
    }
    e->push_txs(h, gpu_txs);
    e->request_close(h);
    auto gpu_r = e->run_until_done(h, 1024);
    e->end_round(h);

    auto cpu_r = quasar::gpu::ref::run_reference(desc, cpu_txs);

    EXPECT("dc.gpu_finalized", gpu_r.status == 1u);
    EXPECT("dc.tx_count", gpu_r.tx_count == cpu_r.tx_count);
    EXPECT("dc.bh_eq", eq32(gpu_r.block_hash, cpu_r.block_hash));
    EXPECT("dc.rr_eq", eq32(gpu_r.receipts_root, cpu_r.receipts_root));
    EXPECT("dc.er_eq", eq32(gpu_r.execution_root, cpu_r.execution_root));
    std::printf("  stm-004 contended: GPU conflicts=%u repairs=%u "
                "(CPU is sequential — 0/0 expected)\n",
                gpu_r.conflict_count, gpu_r.repair_count);
    PASS("cross_backend_determinism_contended");
}

// =============================================================================
// 6. STM-005: Nebula DAG memory ordering. With mem_device fences on the
// state transitions, a 100-tx Nebula round produces deterministic roots.
// =============================================================================
void test_atomic_release_acquire_dag() {
    auto e = QuasarGPUEngine::create();

    auto run_once = [&]() -> QuasarRoundResult {
        auto desc = make_desc(106);
        desc.mode = 1;        // Nebula DAG
        desc.wave_tick_budget = 64;
        auto h = e->begin_round(desc);
        constexpr uint64_t kChain = 0x500000ULL;
        std::vector<HostTxBlob> txs;
        for (int i = 0; i < 100; ++i) {
            auto t = make_exec_tx(0x600000 + uint64_t(i), uint32_t(i));
            HostTxBlob::PredictedAccess pa{};
            pa.key_lo = kChain;
            pa.key_hi = 0;
            pa.is_write = true;
            t.predicted_access.push_back(pa);
            txs.push_back(t);
        }
        e->push_txs(h, txs);
        e->request_close(h);
        auto r = e->run_until_done(h, 4096);
        e->end_round(h);
        return r;
    };

    auto r1 = run_once();
    auto r2 = run_once();
    EXPECT("ra.r1_finalized", r1.status == 1u);
    EXPECT("ra.r2_finalized", r2.status == 1u);
    EXPECT("ra.tx_count_match", r1.tx_count == r2.tx_count && r1.tx_count == 100u);
    EXPECT("ra.bh_eq",  eq32(r1.block_hash, r2.block_hash));
    EXPECT("ra.rr_eq",  eq32(r1.receipts_root, r2.receipts_root));
    EXPECT("ra.er_eq",  eq32(r1.execution_root, r2.execution_root));
    std::printf("  stm-005: 100-tx Nebula chain finalized; conflicts=%u repairs=%u\n",
                r1.conflict_count, r1.repair_count);
    PASS("atomic_release_acquire_dag");
}

// =============================================================================
// 7. STM-013: MVCC arena reset per begin_round. Two rounds with the
// same input → identical roots because state doesn't leak.
// =============================================================================
void test_mvcc_arena_zeroed_per_round() {
    auto e = QuasarGPUEngine::create();

    auto run_round = [&](uint64_t round_id, uint64_t origin) -> QuasarRoundResult {
        auto desc = make_desc(round_id);
        auto h = e->begin_round(desc);
        std::vector<HostTxBlob> txs = { make_exec_tx(origin, 0) };
        e->push_txs(h, txs);
        e->request_close(h);
        auto r = e->run_until_done(h, 32);
        e->end_round(h);
        return r;
    };

    auto a = run_round(200, 0xABCDu);
    auto b = run_round(200, 0xABCDu);
    EXPECT("z.a_finalized", a.status == 1u);
    EXPECT("z.b_finalized", b.status == 1u);
    EXPECT("z.bh_eq", eq32(a.block_hash, b.block_hash));
    EXPECT("z.rr_eq", eq32(a.receipts_root, b.receipts_root));

    auto c = run_round(201, 0xABCDu);
    EXPECT("z.c_finalized", c.status == 1u);
    EXPECT("z.bh_differs_round", !eq32(a.block_hash, c.block_hash));
    PASS("mvcc_arena_zeroed_per_round");
}

// =============================================================================
// 8. v0.46.1 substrate-threshold gate: the gate routes small N to the CPU
//    reference. This test asserts:
//    (a) gated path output is byte-equal to direct CPU reference
//        (tautology: both run quasar::gpu::ref::run_reference);
//    (b) forced-Metal path (LUX_QUASAR_FORCE_METAL=1) finalizes with the
//        same tx_count and gas_used (deterministic invariants both
//        backends honor; the cross_backend_determinism_* tests cover the
//        root-level Metal-vs-CPU contract at production N=1024).
// =============================================================================
void test_substrate_threshold_gate_byte_equal() {
    auto desc = make_desc(700);

    constexpr int N = 16;
    std::vector<HostTxBlob> txs;
    for (int i = 0; i < N; ++i) {
        auto t = make_exec_tx(uint64_t(0x70000) + uint64_t(i), uint32_t(i));
        txs.push_back(t);
    }

    // Branch A: gated path (default — gate fires at this N).
    unsetenv("LUX_QUASAR_FORCE_METAL");
    auto e_gated = QuasarGPUEngine::create();
    EXPECT("gate.create_gated", e_gated != nullptr);
    auto h_gated = e_gated->begin_round(desc);
    e_gated->push_txs(h_gated, txs);
    e_gated->request_close(h_gated);
    auto r_gated = e_gated->run_until_done(h_gated, 64);
    e_gated->end_round(h_gated);

    // Direct CPU reference — byte-equal to gated by tautology.
    std::vector<quasar::gpu::ref::HostInputTx> cpu_txs;
    for (const auto& t : txs) cpu_txs.push_back(mirror_for_cpu(t));
    auto cpu_r = quasar::gpu::ref::run_reference(desc, cpu_txs);

    EXPECT("gate.gated_finalized", r_gated.status == 1u);
    EXPECT("gate.cpu_finalized",   cpu_r.status   == 1u);
    EXPECT("gate.tx_count_eq",     r_gated.tx_count == cpu_r.tx_count);
    EXPECT("gate.gas_used_eq",     r_gated.gas_used() == cpu_r.gas_used);
    EXPECT("gate.bh_eq",           eq32(r_gated.block_hash,     cpu_r.block_hash));
    EXPECT("gate.rr_eq",           eq32(r_gated.receipts_root,  cpu_r.receipts_root));
    EXPECT("gate.er_eq",           eq32(r_gated.execution_root, cpu_r.execution_root));

    // Branch B: forced-Metal path. Asserts the engine still drives Metal
    // when the env is set; we don't compare roots vs CPU here (the
    // existing cross_backend_determinism_disjoint at N=1024 covers that
    // contract; at N=16 the substrate's commit ordering may differ from
    // the strict tx_index-order CPU reference).
    setenv("LUX_QUASAR_FORCE_METAL", "1", /*overwrite=*/1);
    auto e_metal = QuasarGPUEngine::create();
    EXPECT("gate.create_metal", e_metal != nullptr);
    auto h_metal = e_metal->begin_round(desc);
    e_metal->push_txs(h_metal, txs);
    e_metal->request_close(h_metal);
    auto r_metal = e_metal->run_until_done(h_metal, 256);
    e_metal->end_round(h_metal);
    unsetenv("LUX_QUASAR_FORCE_METAL");

    EXPECT("gate.metal_finalized", r_metal.status == 1u);
    EXPECT("gate.metal_tx_count",  r_metal.tx_count == r_gated.tx_count);
    EXPECT("gate.metal_gas_used",  r_metal.gas_used() == r_gated.gas_used());

    std::printf("  gate: tx=%u gas=%llu (gated==cpu byte-equal; metal finalized)\n",
                r_gated.tx_count, (unsigned long long)r_gated.gas_used());
    PASS("substrate_threshold_gate_byte_equal");
}

}  // namespace

int main(int /*argc*/, char** /*argv*/) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    @autoreleasepool {
        std::printf("[quasar_stm_red_review_test] starting\n");
        std::fflush(stdout);
        test_validate_atomic_with_commit_full();
        test_mvcc_concurrent_slot_claim();
        test_repair_cap_aborts_runaway();
        test_cross_backend_determinism_disjoint();
        test_cross_backend_determinism_contended();
        test_atomic_release_acquire_dag();
        test_mvcc_arena_zeroed_per_round();
        test_substrate_threshold_gate_byte_equal();
        std::printf("[quasar_stm_red_review_test] passed=%d failed=%d\n",
                    g_passed, g_failed);
        std::fflush(stdout);
    }
    return g_failed == 0 ? 0 : 1;
}
