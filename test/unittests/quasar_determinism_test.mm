// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file quasar_determinism_test.mm
/// Cross-backend determinism harness for QuasarGPUEngine (Metal).
///
/// Pinned for Quasar 4.0 activation (2026-02-14): every byte that lands
/// in QuasarRoundResult must be a pure function of the round descriptor
/// and the input transactions — across:
///   (a) two runs on the *same* engine instance,
///   (b) two runs on *separate* engine instances.
///
/// The CUDA equivalent will run on a self-hosted runner once H100 hardware
/// is wired (.github/workflows/quasar-cuda-build.yml). Until then this
/// suite proves the Metal backend is deterministic; cross-backend (Metal
/// vs CUDA) parity will be a `quasar_xbackend_determinism_test` that
/// shells out to the CUDA test binary's stdout digest.
///
/// Workloads:
///   1. 1024-tx straight Crypto→Commit fast lane
///   2. 16-tx all-same-exec-key (Block-STM conflict + repair)
///   3. 100-tx Nebula DAG with predicted access set
///
/// Captured fields (all part of QuasarCert subject):
///   block_hash, state_root, receipts_root, execution_root, mode_root,
///   tx_count, gas_used, conflict_count, repair_count

// Same source feeds both:
//   * lib/evm/CMakeLists.txt -> quasar-determinism-test (Apple, .mm)
//   * .github/workflows/quasar-cuda-build.yml -> quasar-determinism-test (Linux, .cpp wrapper)
// On non-Apple builds the file is included from a tiny .cpp shim that
// pre-defines QUASAR_DET_NO_OBJC.
#if defined(__APPLE__) && !defined(QUASAR_DET_NO_OBJC)
#  import <Metal/Metal.h>
#  import <Foundation/Foundation.h>
#  define QUASAR_AUTORELEASEPOOL_BEGIN @autoreleasepool {
#  define QUASAR_AUTORELEASEPOOL_END   }
#else
#  define QUASAR_AUTORELEASEPOOL_BEGIN /* nothing */
#  define QUASAR_AUTORELEASEPOOL_END   /* nothing */
#endif

#include "consensus/quasar/gpu/quasar_gpu_engine.hpp"

#include <cstdio>
#include <cstring>
#include <vector>

using quasar::gpu::HostTxBlob;
using quasar::gpu::QuasarGPUEngine;
using quasar::gpu::QuasarRoundDescriptor;
using quasar::gpu::QuasarRoundResult;

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
    d.gas_limit = 30'000'000u;
    d.base_fee = 100u;
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

// Build the three deterministic workloads.
std::vector<HostTxBlob> workload_1024_fast()
{
    std::vector<HostTxBlob> txs;
    txs.reserve(1024);
    for (uint32_t i = 0; i < 1024; ++i) {
        txs.push_back(make_tx(0x80000000ULL + i, i));
    }
    return txs;
}

std::vector<HostTxBlob> workload_16_same_key()
{
    std::vector<HostTxBlob> txs;
    txs.reserve(16);
    for (uint32_t i = 0; i < 16; ++i) {
        auto t = make_tx(0xCAFE'BABEULL, i);
        t.needs_exec = true;
        t.exec_key_lo = 0xDEAD'BEEF'C0FF'EEULL;
        t.exec_key_hi = 0u;
        txs.push_back(t);
    }
    return txs;
}

std::vector<HostTxBlob> workload_100_nebula_dag()
{
    std::vector<HostTxBlob> txs;
    txs.reserve(100);
    for (uint32_t i = 0; i < 100; ++i) {
        auto t = make_tx(0x9000'0000ULL + i, i);
        t.needs_exec = true;
        // Disjoint key per tx — DAG with no edges, full antichain.
        HostTxBlob::PredictedAccess pa{};
        pa.key_lo  = 0xA000'0000ULL + i;
        pa.key_hi  = 0u;
        pa.is_write = true;
        t.predicted_access.push_back(pa);
        txs.push_back(t);
    }
    return txs;
}

struct Snapshot {
    uint8_t  block_hash[32];
    uint8_t  state_root[32];
    uint8_t  receipts_root[32];
    uint8_t  execution_root[32];
    uint8_t  mode_root[32];
    uint32_t tx_count;
    uint64_t gas_used;
    uint32_t conflict_count;
    uint32_t repair_count;

    static Snapshot from(const QuasarRoundResult& r)
    {
        Snapshot s{};
        std::memcpy(s.block_hash,     r.block_hash,     32);
        std::memcpy(s.state_root,     r.state_root,     32);
        std::memcpy(s.receipts_root,  r.receipts_root,  32);
        std::memcpy(s.execution_root, r.execution_root, 32);
        std::memcpy(s.mode_root,      r.mode_root,      32);
        s.tx_count       = r.tx_count;
        s.gas_used       = r.gas_used();
        s.conflict_count = r.conflict_count;
        s.repair_count   = r.repair_count;
        return s;
    }

    bool equals(const Snapshot& o) const
    {
        return std::memcmp(block_hash,     o.block_hash,     32) == 0 &&
               std::memcmp(state_root,     o.state_root,     32) == 0 &&
               std::memcmp(receipts_root,  o.receipts_root,  32) == 0 &&
               std::memcmp(execution_root, o.execution_root, 32) == 0 &&
               std::memcmp(mode_root,      o.mode_root,      32) == 0 &&
               tx_count       == o.tx_count &&
               gas_used       == o.gas_used &&
               conflict_count == o.conflict_count &&
               repair_count   == o.repair_count;
    }
};

Snapshot run_one(QuasarGPUEngine* e,
                 uint64_t round,
                 uint32_t mode,
                 const std::vector<HostTxBlob>& txs,
                 std::size_t max_epochs)
{
    auto h = e->begin_round(make_desc(round, mode));
    e->push_txs(h, txs);
    e->request_close(h);
    auto r = e->run_until_done(h, max_epochs);
    auto s = Snapshot::from(r);
    e->end_round(h);
    return s;
}

// 1. Same engine, two rounds — bytes must match.
void test_same_engine_repeats(const char* name,
                              uint64_t round,
                              uint32_t mode,
                              const std::vector<HostTxBlob>& txs,
                              std::size_t max_epochs)
{
    auto e = QuasarGPUEngine::create();
    EXPECT(name, e != nullptr);
    auto s1 = run_one(e.get(), round, mode, txs, max_epochs);
    auto s2 = run_one(e.get(), round, mode, txs, max_epochs);
    EXPECT(name, s1.equals(s2));
    std::printf("  same-engine: %s tx=%u gas=%llu conf=%u repair=%u\n",
                name, s1.tx_count, (unsigned long long)s1.gas_used,
                s1.conflict_count, s1.repair_count);
    PASS(name);
}

// 2. Two separate engines — bytes must match.
void test_two_engines(const char* name,
                      uint64_t round,
                      uint32_t mode,
                      const std::vector<HostTxBlob>& txs,
                      std::size_t max_epochs)
{
    auto a = QuasarGPUEngine::create();
    auto b = QuasarGPUEngine::create();
    EXPECT(name, a != nullptr);
    EXPECT(name, b != nullptr);
    auto sa = run_one(a.get(), round, mode, txs, max_epochs);
    auto sb = run_one(b.get(), round, mode, txs, max_epochs);
    EXPECT(name, sa.equals(sb));
    std::printf("  two-engines: %s tx=%u gas=%llu conf=%u repair=%u\n",
                name, sa.tx_count, (unsigned long long)sa.gas_used,
                sa.conflict_count, sa.repair_count);
    PASS(name);
}

}  // namespace

int main(int /*argc*/, char** /*argv*/)
{
    setvbuf(stdout, nullptr, _IOLBF, 0);
    QUASAR_AUTORELEASEPOOL_BEGIN
        std::printf("[quasar_determinism_test] starting\n");
        std::fflush(stdout);

        auto w_fast   = workload_1024_fast();
        auto w_same   = workload_16_same_key();
        auto w_nebula = workload_100_nebula_dag();

        // Same engine, two rounds.
        test_same_engine_repeats("same.fast.1024",   100, 0, w_fast,   64);
        test_same_engine_repeats("same.same_key.16", 200, 1, w_same,   256);
        test_same_engine_repeats("same.nebula.100",  300, 1, w_nebula, 64);

        // Two engines, one round each.
        test_two_engines("two.fast.1024",   400, 0, w_fast,   64);
        test_two_engines("two.same_key.16", 500, 1, w_same,   256);
        test_two_engines("two.nebula.100",  600, 1, w_nebula, 64);

        std::printf("[quasar_determinism_test] passed=%d failed=%d\n",
                    g_passed, g_failed);
        return g_failed == 0 ? 0 : 1;
    QUASAR_AUTORELEASEPOOL_END
}
