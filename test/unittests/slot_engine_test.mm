// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file slot_engine_test.mm
/// Tests for the GPU Slot Engine (v0.31 foundation).
///
/// Coverage:
///   1. Empty slot — begin/run/close cycle without txs
///   2. Single-tx pipeline — host pushes 1 tx, GPU drains all 4 stages,
///      slot finalizes after closing_flag
///   3. Multi-tx counters — pushed/consumed monotonic per-service
///   4. Bounded backpressure — push more txs than the per-epoch budget
///      and verify the kernel still drains everything across epochs
///   5. End-to-end — 1024 txs across many epochs, total gas matches
///
/// No GoogleTest dep — same bare-Apple harness as v3_persistent_test.mm.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "gpu/slot/slot_engine.hpp"

#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

using evm::gpu::slot::HostTxBlob;
using evm::gpu::slot::ServiceId;
using evm::gpu::slot::SlotDescriptor;
using evm::gpu::slot::SlotEngine;
using evm::gpu::slot::SlotResult;

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

SlotDescriptor make_desc(uint64_t slot)
{
    SlotDescriptor d{};
    d.chain_id = 1u;
    d.slot = slot;
    d.timestamp_ns = 0u;
    d.deadline_ns = 0u;
    d.gas_limit = 30'000'000u;
    d.base_fee = 1u;
    d.epoch_budget_items = 256u;
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

// ---------------------------------------------------------------------------
// 1. Empty slot
// ---------------------------------------------------------------------------
void test_empty_slot()
{
    auto engine = SlotEngine::create();
    EXPECT("empty.create", engine != nullptr);
    auto h = engine->begin_slot(make_desc(1));
    EXPECT("empty.handle", h.valid());

    // No txs pushed; closing should finalize on the next epoch.
    engine->request_close(h);
    auto r = engine->run_until_done(h, 8);
    EXPECT("empty.finalized", r.status == 1u);
    EXPECT("empty.tx_count_zero", r.tx_count == 0u);
    EXPECT("empty.gas_zero", r.gas_used() == 0u);

    engine->end_slot(h);
    PASS("empty_slot");
}

// ---------------------------------------------------------------------------
// 2. Single-tx pipeline
// ---------------------------------------------------------------------------
void test_single_tx_pipeline()
{
    auto engine = SlotEngine::create();
    EXPECT("one.create", engine != nullptr);
    auto h = engine->begin_slot(make_desc(2));
    EXPECT("one.handle", h.valid());

    std::vector<HostTxBlob> txs = { make_tx(0xCAFEu, 0u) };
    engine->push_txs(h, txs);

    auto stats_in = engine->ring_stats(h, ServiceId::Ingress);
    EXPECT("one.ingress_pushed", stats_in.pushed == 1u);

    engine->request_close(h);
    auto r = engine->run_until_done(h, 16);
    EXPECT("one.finalized", r.status == 1u);
    EXPECT("one.tx_count_one", r.tx_count == 1u);
    EXPECT("one.gas_21000", r.gas_used() == 21'000u);

    auto in_after  = engine->ring_stats(h, ServiceId::Ingress);
    auto cmt_after = engine->ring_stats(h, ServiceId::Commit);
    EXPECT("one.ingress_consumed", in_after.consumed == 1u);
    EXPECT("one.commit_consumed",  cmt_after.consumed == 1u);

    engine->end_slot(h);
    PASS("single_tx_pipeline");
}

// ---------------------------------------------------------------------------
// 3. Multi-tx counter monotonicity
// ---------------------------------------------------------------------------
void test_multi_tx_counters()
{
    auto engine = SlotEngine::create();
    auto h = engine->begin_slot(make_desc(3));
    EXPECT("multi.handle", h.valid());

    constexpr size_t N = 128;
    std::vector<HostTxBlob> txs;
    for (size_t i = 0; i < N; ++i) txs.push_back(make_tx(i + 1, uint32_t(i)));
    engine->push_txs(h, txs);

    engine->request_close(h);
    auto r = engine->run_until_done(h, 32);
    EXPECT("multi.finalized", r.status == 1u);
    EXPECT("multi.tx_count", r.tx_count == N);
    EXPECT("multi.gas_total", r.gas_used() == N * 21'000u);

    // Every stage's consumed counter must equal pushed (and N for the
    // first three rings since no admission rejections happen in v0.31).
    for (auto sid : {ServiceId::Ingress, ServiceId::Decode,
                     ServiceId::Crypto, ServiceId::Commit}) {
        auto s = engine->ring_stats(h, sid);
        if (s.consumed != s.pushed) {
            std::printf("  multi.stage_drain mismatch: id=%u pushed=%u consumed=%u\n",
                        unsigned(sid), s.pushed, s.consumed);
            ++g_failed; return;
        }
    }

    engine->end_slot(h);
    PASS("multi_tx_counters");
}

// ---------------------------------------------------------------------------
// 4. Bounded backpressure across epochs
// ---------------------------------------------------------------------------
void test_bounded_backpressure()
{
    auto engine = SlotEngine::create();
    SlotDescriptor d = make_desc(4);
    d.epoch_budget_items = 64;  // small budget forces multi-epoch drain
    auto h = engine->begin_slot(d);

    constexpr size_t N = 1024;
    std::vector<HostTxBlob> txs;
    for (size_t i = 0; i < N; ++i) txs.push_back(make_tx(i + 1, uint32_t(i)));

    // Push half before any epoch...
    engine->push_txs(h, std::span<const HostTxBlob>(txs.data(), N/2));
    // Run a few epochs to drain partially.
    for (int i = 0; i < 4; ++i) engine->run_epoch(h);

    auto mid = engine->ring_stats(h, ServiceId::Commit);
    EXPECT("bp.partial_drain", mid.consumed > 0u);

    // ...push the rest, then close.
    engine->push_txs(h, std::span<const HostTxBlob>(txs.data() + N/2, N/2));
    engine->request_close(h);
    auto r = engine->run_until_done(h, 64);

    EXPECT("bp.finalized", r.status == 1u);
    EXPECT("bp.tx_count", r.tx_count == N);
    EXPECT("bp.gas_total", r.gas_used() == N * 21'000u);

    engine->end_slot(h);
    PASS("bounded_backpressure");
}

// ---------------------------------------------------------------------------
// 5. End-to-end stress
// ---------------------------------------------------------------------------
void test_end_to_end_stress()
{
    auto engine = SlotEngine::create();
    auto h = engine->begin_slot(make_desc(5));

    constexpr size_t N = 1024;
    std::vector<HostTxBlob> txs;
    for (size_t i = 0; i < N; ++i) txs.push_back(make_tx(i + 1, uint32_t(i)));

    auto t0 = std::chrono::steady_clock::now();
    engine->push_txs(h, txs);
    engine->request_close(h);
    auto r = engine->run_until_done(h, 64);
    auto t1 = std::chrono::steady_clock::now();

    EXPECT("e2e.finalized", r.status == 1u);
    EXPECT("e2e.tx_count", r.tx_count == N);
    EXPECT("e2e.gas_total", r.gas_used() == N * 21'000u);

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    std::printf("  e2e: %zu txs across %u epochs in %lld ms\n",
                N, r.epoch_count, (long long)ms);
    std::fflush(stdout);

    engine->end_slot(h);
    PASS("end_to_end_stress");
}

}  // namespace

int main(int /*argc*/, char** /*argv*/)
{
    setvbuf(stdout, nullptr, _IOLBF, 0);
    @autoreleasepool {
        std::printf("[slot_engine_test] starting (device=%s)\n",
                    SlotEngine::create() ? "metal" : "missing");
        std::fflush(stdout);
        test_empty_slot();
        test_single_tx_pipeline();
        test_multi_tx_counters();
        test_bounded_backpressure();
        test_end_to_end_stress();
        std::printf("[slot_engine_test] passed=%d failed=%d\n",
                    g_passed, g_failed);
        return g_failed == 0 ? 0 : 1;
    }
}
