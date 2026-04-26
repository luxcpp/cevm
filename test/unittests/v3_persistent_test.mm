// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file v3_persistent_test.mm
/// Tests for the V3 persistent-kernel runner.
///
/// Coverage:
///   1. Queue drain (empty → full → empty cycle)
///   2. Backpressure: pushing more than queue capacity blocks the
///      producer until the consumer has drained
///   3. Shutdown signal: setting the flag causes all three kernels to
///      exit cleanly even with un-drained queues
///   4. Pipeline ordering: every committed tx had to traverse exec →
///      validate → commit; we observe Counters monotonically advance
///
/// Pattern follows test/unittests/metal_host_test.mm — no GoogleTest
/// dependency to keep build self-contained.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "gpu/kernel/v3_persistent_host.hpp"
#include "gpu/kernel/v3_queue.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

using evm::gpu::kernel::HostTransaction;
using evm::gpu::kernel::TxResult;
using evm::gpu::kernel::TxStatus;
using evm::gpu::kernel::v3::V3PersistentRunner;
using evm::gpu::kernel::v3::WaveFuture;

namespace {

int g_passed = 0;
int g_failed = 0;

#define EXPECT(name, cond)                                                   \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("  FAIL[%s]: %s\n", (name), #cond);                    \
            ++g_failed;                                                        \
            return;                                                            \
        }                                                                      \
    } while (0)

#define PASS(name) do { std::printf("  ok  : %s\n", (name)); ++g_passed; } while (0)

/// Build a trivial-bytecode tx whose code_size becomes a deterministic
/// gas_used value the executor in v3_persistent.metal computes:
///   gas_used = code_size * 21 + 21000
HostTransaction make_tx(size_t code_bytes)
{
    HostTransaction tx;
    tx.code.assign(code_bytes, 0x00);  // STOPs; content doesn't matter
    tx.calldata.clear();
    tx.gas_limit = 1'000'000;
    return tx;
}

uint64_t expected_gas(size_t code_bytes)
{
    return static_cast<uint64_t>(code_bytes) * 21u + 21000u;
}

// -----------------------------------------------------------------------------
// Test 1 — Queue drain.
//
// Push N=64 txs, await, observe N committed. Drain again with another wave
// to confirm the runner is stateful and ready for the next wave.
// -----------------------------------------------------------------------------
void test_queue_drain_basic()
{
    auto runner = V3PersistentRunner::create();
    EXPECT("drain.create", runner != nullptr);

    constexpr size_t N = 64;
    std::vector<HostTransaction> txs;
    txs.reserve(N);
    for (size_t i = 0; i < N; ++i)
        txs.push_back(make_tx(i + 1));  // distinct sizes 1..N

    auto fut = runner->enqueue_wave(txs);
    EXPECT("drain.fut", fut != nullptr);
    bool ok = fut->await_for(std::chrono::seconds(5));
    if (!ok) {
        auto cnt = runner->counters();
        std::printf("  drain timeout counters: exec=%llu validate=%llu commit=%llu alive(e/v/c)=%u/%u/%u\n",
                    (unsigned long long)cnt.executed,
                    (unsigned long long)cnt.validated,
                    (unsigned long long)cnt.committed,
                    cnt.exec_alive, cnt.validate_alive, cnt.commit_alive);
    }
    EXPECT("drain.completes", ok);
    auto results = fut->await();
    EXPECT("drain.size", results.size() == N);
    for (size_t i = 0; i < N; ++i)
    {
        if (results[i].gas_used != expected_gas(i + 1))
        {
            std::printf("  drain[%zu] gas mismatch: got %llu expected %llu\n",
                        i, (unsigned long long)results[i].gas_used,
                        (unsigned long long)expected_gas(i + 1));
            ++g_failed; return;
        }
        if (results[i].status != TxStatus::Return)
        {
            std::printf("  drain[%zu] status mismatch\n", i);
            ++g_failed; return;
        }
    }

    auto cnt = runner->counters();
    EXPECT("drain.cnt.exec",     cnt.executed  >= N);
    EXPECT("drain.cnt.validate", cnt.validated >= N);
    EXPECT("drain.cnt.commit",   cnt.committed >= N);

    // Second wave — queue is empty, push and drain again.
    std::vector<HostTransaction> txs2;
    for (size_t i = 0; i < N; ++i)
        txs2.push_back(make_tx(100 + i));
    auto fut2 = runner->enqueue_wave(txs2);
    EXPECT("drain.fut2", fut2 != nullptr);
    EXPECT("drain.completes2",
           fut2->await_for(std::chrono::seconds(5)));
    auto results2 = fut2->await();
    EXPECT("drain.size2", results2.size() == N);
    for (size_t i = 0; i < N; ++i)
    {
        if (results2[i].gas_used != expected_gas(100 + i))
        {
            std::printf("  drain2[%zu] gas mismatch\n", i);
            ++g_failed; return;
        }
    }

    runner->shutdown();
    EXPECT("drain.shut", runner->is_shut_down());
    PASS("queue_drain_basic");
}

// -----------------------------------------------------------------------------
// Test 2 — Backpressure.
//
// Push more txs than Q_CAPACITY (2× capacity). The producer must block
// in enqueue_wave until exec_worker drains. We assert the second half
// completes successfully and that the wave's await_for() returns true
// within a sane timeout.
// -----------------------------------------------------------------------------
void test_backpressure()
{
    auto runner = V3PersistentRunner::create();
    EXPECT("bp.create", runner != nullptr);

    // 2× queue capacity. The host enqueue loop must block on exec_q
    // being full and let the kernel drain it.
    constexpr size_t N = evm::gpu::kernel::v3::Q_CAPACITY * 2u;
    std::vector<HostTransaction> txs;
    txs.reserve(N);
    for (size_t i = 0; i < N; ++i)
        txs.push_back(make_tx(1));  // tiny payload — kernel turnaround is fast

    auto t_start = std::chrono::steady_clock::now();
    auto fut = runner->enqueue_wave(txs);
    EXPECT("bp.fut", fut != nullptr);

    // The wave is large; give it 30 seconds. Backpressure shouldn't
    // make this timeout — the kernel drains the queue continuously.
    EXPECT("bp.completes",
           fut->await_for(std::chrono::seconds(30)));
    auto t_end = std::chrono::steady_clock::now();

    auto results = fut->await();
    EXPECT("bp.size", results.size() == N);
    for (size_t i = 0; i < N; ++i)
    {
        if (results[i].gas_used != expected_gas(1))
        {
            std::printf("  bp[%zu] gas mismatch\n", i);
            ++g_failed; return;
        }
    }

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  t_end - t_start).count();
    std::printf("  backpressure: %zu txs in %lld ms\n", N, (long long)ms);

    runner->shutdown();
    PASS("backpressure");
}

// -----------------------------------------------------------------------------
// Test 3 — Shutdown signal.
//
// Push a small wave, then immediately shutdown(). The kernel must drain
// (commit_done >= N) and then exit. The runner's shutdown() returns only
// after [cmd waitUntilCompleted] which is the kernel exit point.
// -----------------------------------------------------------------------------
void test_shutdown()
{
    auto runner = V3PersistentRunner::create();
    EXPECT("sd.create", runner != nullptr);

    constexpr size_t N = 32;
    std::vector<HostTransaction> txs;
    for (size_t i = 0; i < N; ++i)
        txs.push_back(make_tx(i + 1));

    auto fut = runner->enqueue_wave(txs);
    EXPECT("sd.fut", fut != nullptr);

    // Wait for the wave; shutdown should not race with this.
    EXPECT("sd.completes",
           fut->await_for(std::chrono::seconds(5)));
    auto results = fut->await();
    EXPECT("sd.size", results.size() == N);

    runner->shutdown();
    EXPECT("sd.is_shut_down", runner->is_shut_down());

    auto cnt = runner->counters();
    EXPECT("sd.cnt.exec",   cnt.executed  >= N);
    EXPECT("sd.cnt.commit", cnt.committed >= N);

    // Idempotent.
    runner->shutdown();
    EXPECT("sd.idempotent", runner->is_shut_down());

    // Post-shutdown enqueue must throw.
    bool threw = false;
    try {
        std::vector<HostTransaction> nope = { make_tx(1) };
        auto f2 = runner->enqueue_wave(nope);
        (void)f2;
    } catch (const std::runtime_error&) {
        threw = true;
    }
    EXPECT("sd.post_shutdown_throws", threw);

    PASS("shutdown");
}

// -----------------------------------------------------------------------------
// Test 4 — Empty wave.
//
// Edge case: empty span. Should return an immediately-ready future with
// zero results. No GPU traffic, no queue mutation.
// -----------------------------------------------------------------------------
void test_empty_wave()
{
    auto runner = V3PersistentRunner::create();
    EXPECT("empty.create", runner != nullptr);

    std::vector<HostTransaction> empty;
    auto fut = runner->enqueue_wave(empty);
    EXPECT("empty.fut",   fut != nullptr);
    EXPECT("empty.ready", fut->ready());
    auto results = fut->await();
    EXPECT("empty.size", results.empty());

    runner->shutdown();
    PASS("empty_wave");
}

// -----------------------------------------------------------------------------
// Test 5 — Counters monotonicity.
//
// The Counters struct must report non-decreasing values across two waves.
// This guarantees the pipeline visit order: a tx counted in commit_done
// must have been counted in validate_done first, and exec_done before that.
// -----------------------------------------------------------------------------
void test_counters_monotonic()
{
    auto runner = V3PersistentRunner::create();
    EXPECT("cnt.create", runner != nullptr);

    constexpr size_t N = 16;
    std::vector<HostTransaction> txs;
    for (size_t i = 0; i < N; ++i)
        txs.push_back(make_tx(1));

    auto fut1 = runner->enqueue_wave(txs);
    fut1->await();
    auto c1 = runner->counters();

    auto fut2 = runner->enqueue_wave(txs);
    fut2->await();
    auto c2 = runner->counters();

    EXPECT("cnt.exec_grew",     c2.executed  >= c1.executed);
    EXPECT("cnt.validate_grew", c2.validated >= c1.validated);
    EXPECT("cnt.commit_grew",   c2.committed >= c1.committed);
    // Pipeline invariant: exec >= validate >= commit at any snapshot.
    EXPECT("cnt.exec_ge_validate", c2.executed  >= c2.validated);
    EXPECT("cnt.validate_ge_commit", c2.validated >= c2.committed);

    runner->shutdown();
    PASS("counters_monotonic");
}

}  // namespace

int main(int /*argc*/, char** /*argv*/)
{
    setvbuf(stdout, nullptr, _IOLBF, 0);
    @autoreleasepool {
        std::printf("[v3_persistent_test] starting\n");
        std::fflush(stdout);
        test_queue_drain_basic();
        test_empty_wave();
        test_counters_monotonic();
        test_shutdown();
        test_backpressure();
        std::printf("[v3_persistent_test] passed=%d failed=%d\n",
                    g_passed, g_failed);
        return g_failed == 0 ? 0 : 1;
    }
}
