// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// quasar_round_bench.mm — wall-clock bench: Metal QuasarGPUEngine vs CPU
// reference (quasar_cpu_reference) on the same three deterministic
// substrate workloads used by the determinism harness, plus per-call
// timings for the BLS / Groth16 / Ringtail host verifiers.
//
// Output is plain stdout; consumed by BENCHMARKS.md. Self-contained — no
// GoogleTest, no google-benchmark, just a small timing harness that runs
// each backend N times with a warm-up round and reports min / mean.
//
// Exit code is always 0; this is a measurement tool, not a pass/fail
// test.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "consensus/quasar/gpu/quasar_gpu_engine.hpp"
#include "consensus/quasar/gpu/quasar_cpu_reference.hpp"
#include "consensus/quasar/gpu/quasar_bls_verifier.hpp"
#include "consensus/quasar/gpu/quasar_groth16_verifier.hpp"
#include "consensus/quasar/gpu/quasar_ringtail_verifier.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

using quasar::gpu::HostTxBlob;
using quasar::gpu::QuasarGPUEngine;
using quasar::gpu::QuasarRoundDescriptor;
using quasar::gpu::QuasarRoundResult;

namespace {

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

std::vector<HostTxBlob> workload_1024_fast()
{
    std::vector<HostTxBlob> txs;
    txs.reserve(1024);
    for (uint32_t i = 0; i < 1024; ++i)
        txs.push_back(make_tx(0x80000000ULL + i, i));
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
        HostTxBlob::PredictedAccess pa{};
        pa.key_lo  = 0xA000'0000ULL + i;
        pa.is_write = true;
        t.predicted_access.push_back(pa);
        txs.push_back(t);
    }
    return txs;
}

std::vector<quasar::gpu::ref::HostInputTx> to_ref(const std::vector<HostTxBlob>& txs)
{
    std::vector<quasar::gpu::ref::HostInputTx> out;
    out.reserve(txs.size());
    for (const auto& t : txs) {
        quasar::gpu::ref::HostInputTx r;
        r.gas_limit   = t.gas_limit;
        r.origin      = t.origin;
        r.needs_state = t.needs_state;
        r.needs_exec  = t.needs_exec;
        out.push_back(r);
    }
    return out;
}

struct Stats {
    double mean_ms;
    double min_ms;
    double max_ms;
};

Stats compute_stats(const std::vector<double>& v)
{
    Stats s{};
    s.mean_ms = std::accumulate(v.begin(), v.end(), 0.0) / double(v.size());
    s.min_ms  = *std::min_element(v.begin(), v.end());
    s.max_ms  = *std::max_element(v.begin(), v.end());
    return s;
}

double run_metal_round(QuasarGPUEngine* e,
                       const QuasarRoundDescriptor& desc,
                       const std::vector<HostTxBlob>& txs,
                       std::size_t max_epochs)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    auto h = e->begin_round(desc);
    e->push_txs(h, txs);
    e->request_close(h);
    auto r = e->run_until_done(h, max_epochs);
    e->end_round(h);
    auto t1 = std::chrono::high_resolution_clock::now();
    (void)r;
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

double run_cpu_round(const QuasarRoundDescriptor& desc,
                     const std::vector<quasar::gpu::ref::HostInputTx>& txs)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    auto r = quasar::gpu::ref::run_reference(desc, txs);
    auto t1 = std::chrono::high_resolution_clock::now();
    (void)r;
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

void bench_workload(const char* name,
                    const std::vector<HostTxBlob>& gpu_txs,
                    std::size_t max_epochs,
                    std::size_t runs,
                    std::size_t warmup)
{
    auto cpu_txs = to_ref(gpu_txs);
    auto desc = make_desc(1u);

    // Metal.
    auto e = QuasarGPUEngine::create();
    if (!e) {
        std::printf("workload=%s metal=unavailable\n", name);
        return;
    }
    for (std::size_t i = 0; i < warmup; ++i)
        run_metal_round(e.get(), desc, gpu_txs, max_epochs);
    std::vector<double> metal_times;
    metal_times.reserve(runs);
    for (std::size_t i = 0; i < runs; ++i)
        metal_times.push_back(run_metal_round(e.get(), desc, gpu_txs, max_epochs));
    auto m = compute_stats(metal_times);

    // CPU.
    for (std::size_t i = 0; i < warmup; ++i)
        run_cpu_round(desc, cpu_txs);
    std::vector<double> cpu_times;
    cpu_times.reserve(runs);
    for (std::size_t i = 0; i < runs; ++i)
        cpu_times.push_back(run_cpu_round(desc, cpu_txs));
    auto c = compute_stats(cpu_times);

    double speedup_min  = c.min_ms  / m.min_ms;
    double speedup_mean = c.mean_ms / m.mean_ms;
    double rounds_per_sec_metal = 1000.0 / m.min_ms;
    double rounds_per_sec_cpu   = 1000.0 / c.min_ms;

    std::printf("workload=%-18s tx=%-5zu cpu_min=%8.3fms cpu_mean=%8.3fms"
                " metal_min=%8.3fms metal_mean=%8.3fms"
                " speedup_min=%5.2fx speedup_mean=%5.2fx"
                " cpu_rounds_per_sec=%8.1f metal_rounds_per_sec=%8.1f\n",
                name, gpu_txs.size(),
                c.min_ms, c.mean_ms,
                m.min_ms, m.mean_ms,
                speedup_min, speedup_mean,
                rounds_per_sec_cpu, rounds_per_sec_metal);
}

void bench_bls_aggregate_verify(std::size_t batch_size, std::size_t runs)
{
    using namespace quasar::gpu;

    std::vector<BLSPublicKey> pks(batch_size);
    std::vector<std::array<uint8_t, 96>> sigs(batch_size);
    uint8_t subject[32]{};
    for (uint8_t k = 0; k < 32; ++k) subject[k] = uint8_t(0x42 ^ k);

    for (std::size_t i = 0; i < batch_size; ++i) {
        std::array<uint8_t, 32> ikm{};
        for (uint8_t k = 0; k < 32; ++k) ikm[k] = uint8_t(0xA0 ^ uint8_t(i) ^ k);
        BLSSecretKey sk;
        if (!keygen_bls(ikm.data(), ikm.size(), sk, pks[i])) {
            std::printf("bench_bls keygen failed\n");
            return;
        }
        if (!sign_subject(sk, subject, sigs[i].data())) {
            std::printf("bench_bls sign failed\n");
            return;
        }
    }

    // Warm-up.
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t b = 0; b < batch_size; ++b)
            (void)verify_bls_aggregate(subject, sigs[b].data(), pks[b].bytes.data());
    }

    std::vector<double> times_us;
    times_us.reserve(runs);
    for (std::size_t r = 0; r < runs; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (std::size_t b = 0; b < batch_size; ++b) {
            volatile bool ok = verify_bls_aggregate(subject, sigs[b].data(), pks[b].bytes.data());
            (void)ok;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        times_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    std::sort(times_us.begin(), times_us.end());
    double min_us = times_us.front();
    double per_call_us = min_us / double(batch_size);
    double per_sec = 1.0e6 / per_call_us;
    std::printf("precompile=bls_aggregate_verify_single batch=%-4zu batch_min_us=%9.1f"
                " per_call_us=%7.2f calls_per_sec=%8.0f\n",
                batch_size, min_us, per_call_us, per_sec);
}

void bench_groth16_vk_root(std::size_t runs)
{
    using namespace quasar::gpu;

    Groth16VerifyingKey vk{};
    for (auto& b : vk.alpha_g1) b = 0x11;
    for (auto& b : vk.beta_g2)  b = 0x22;
    for (auto& b : vk.gamma_g2) b = 0x33;
    for (auto& b : vk.delta_g2) b = 0x44;
    vk.ic.resize(8);
    for (size_t i = 0; i < vk.ic.size(); ++i)
        for (auto& b : vk.ic[i]) b = uint8_t(0x50 + i);

    // Warm-up.
    for (std::size_t i = 0; i < 2; ++i) (void)compute_vk_root(vk);

    std::vector<double> times_us;
    times_us.reserve(runs);
    for (std::size_t r = 0; r < runs; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto root = compute_vk_root(vk);
        auto t1 = std::chrono::high_resolution_clock::now();
        times_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        (void)root;
    }
    std::sort(times_us.begin(), times_us.end());
    double min_us = times_us.front();
    double per_sec = 1.0e6 / min_us;
    std::printf("precompile=groth16_vk_root           ic_size=%-4zu min_us=%9.2f"
                " calls_per_sec=%8.0f\n",
                vk.ic.size(), min_us, per_sec);
}

void bench_ringtail_share(std::size_t runs)
{
    uint8_t subject[32]{};
    uint8_t ceremony_root[32]{};
    for (uint8_t k = 0; k < 32; ++k) {
        subject[k]       = uint8_t(0x55 ^ k);
        ceremony_root[k] = uint8_t(0x77 ^ k);
    }
    std::vector<uint8_t> share(64);
    for (size_t i = 0; i < share.size(); ++i)
        share[i] = uint8_t(i * 7 + 1);

    // Warm-up.
    for (std::size_t i = 0; i < 2; ++i)
        (void)quasar::gpu::verify_ringtail_share(subject, share.data(), uint32_t(share.size()),
                                                 0u, 1u, ceremony_root);

    std::vector<double> times_us;
    times_us.reserve(runs);
    for (std::size_t r = 0; r < runs; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        volatile bool ok = quasar::gpu::verify_ringtail_share(
            subject, share.data(), uint32_t(share.size()), 0u, 1u, ceremony_root);
        auto t1 = std::chrono::high_resolution_clock::now();
        times_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        (void)ok;
    }
    std::sort(times_us.begin(), times_us.end());
    double min_us = times_us.front();
    double per_sec = 1.0e6 / min_us;
    std::printf("precompile=ringtail_share_verify     share_len=%-4zu min_us=%9.2f"
                " calls_per_sec=%8.0f\n",
                share.size(), min_us, per_sec);
}

}  // namespace

int main(int /*argc*/, char** /*argv*/)
{
    setvbuf(stdout, nullptr, _IOLBF, 0);
    @autoreleasepool {
        std::printf("[quasar_round_bench] starting\n");
        std::printf("device: ");
        if (auto e = QuasarGPUEngine::create()) std::printf("%s\n", e->device_name());
        else std::printf("metal-unavailable\n");

        // Substrate round wall clock — the headline number.
        std::printf("\n# Quasar full-round wall clock (Metal vs CPU reference)\n");
        bench_workload("fast.1024",   workload_1024_fast(),    1024, /*runs=*/10, /*warmup=*/2);
        bench_workload("same_key.16", workload_16_same_key(),    64, /*runs=*/10, /*warmup=*/2);
        bench_workload("nebula.100",  workload_100_nebula_dag(), 256, /*runs=*/10, /*warmup=*/2);

        // Per-precompile micro-timings (host blst / keccak — same hot
        // path the GPU substrate calls into for vote verification).
        std::printf("\n# Precompile per-call timings (host CPU)\n");
        bench_bls_aggregate_verify(   1, /*runs=*/200);
        bench_bls_aggregate_verify(  16, /*runs=*/100);
        bench_bls_aggregate_verify( 128, /*runs=*/ 50);
        bench_bls_aggregate_verify(1024, /*runs=*/ 10);
        bench_groth16_vk_root(/*runs=*/200);
        bench_ringtail_share(/*runs=*/2000);

        std::printf("\n[quasar_round_bench] done\n");
    }
    return 0;
}
