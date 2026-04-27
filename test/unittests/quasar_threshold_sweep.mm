// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// quasar_threshold_sweep.mm — measures the Metal-vs-CPU cutover point for
// the Quasar substrate. Sweeps N over {16, 64, 256, 1024, 4096, 16384,
// 65536, 262144} and reports min wall-clock for each backend, the smallest
// N where Metal min <= CPU min, and a markdown-friendly table.
//
// Self-contained measurement tool. Sets LUX_QUASAR_FORCE_METAL=1 in-process
// before each Metal run so that any compiled-in threshold gate falls
// through to Metal (i.e. measures the raw Metal cost, not the gate).
//
// Output is stdout, consumed by BENCHMARKS.md. Exit code is always 0.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "consensus/quasar/gpu/quasar_gpu_engine.hpp"
#include "consensus/quasar/gpu/quasar_cpu_reference.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using quasar::gpu::HostTxBlob;
using quasar::gpu::QuasarGPUEngine;
using quasar::gpu::QuasarRoundDescriptor;
using quasar::gpu::QuasarRoundResult;

namespace {

QuasarRoundDescriptor make_desc(uint64_t round)
{
    QuasarRoundDescriptor d{};
    d.chain_id = 1u;
    d.round = round;
    d.gas_limit = 30'000'000u;
    d.base_fee = 100u;
    d.wave_tick_budget = 256u;
    d.mode = 0;
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

std::vector<HostTxBlob> workload_fast(uint32_t n)
{
    std::vector<HostTxBlob> txs;
    txs.reserve(n);
    for (uint32_t i = 0; i < n; ++i)
        txs.push_back(make_tx(0x80000000ULL + uint64_t(i), i));
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

double median(std::vector<double>& v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

double min_of(const std::vector<double>& v)
{
    return *std::min_element(v.begin(), v.end());
}

double run_metal(QuasarGPUEngine* e,
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

double run_cpu(const QuasarRoundDescriptor& desc,
               const std::vector<quasar::gpu::ref::HostInputTx>& txs)
{
    auto t0 = std::chrono::high_resolution_clock::now();
    auto r = quasar::gpu::ref::run_reference(desc, txs);
    auto t1 = std::chrono::high_resolution_clock::now();
    (void)r;
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

struct Row {
    uint32_t n;
    double cpu_min_ms;
    double cpu_med_ms;
    double metal_min_ms;
    double metal_med_ms;
    double speedup_min;
};

Row sweep_one(uint32_t n, std::size_t runs, std::size_t warmup)
{
    auto gpu_txs = workload_fast(n);
    auto cpu_txs = to_ref(gpu_txs);
    // wave_tick_budget >= n / 4 for 32-thread groups across Ingress; pick a
    // generous max_epochs so the substrate completes regardless of N.
    std::size_t max_epochs = std::max<std::size_t>(64u, n / 4u + 8u);

    auto desc = make_desc(uint64_t(n));

    auto e = QuasarGPUEngine::create();
    if (!e) {
        Row r{n, 0, 0, 0, 0, 0};
        return r;
    }

    // Force the Metal path even after the threshold gate lands in
    // run_until_done — sweep needs raw Metal numbers to find the cutover.
    setenv("LUX_QUASAR_FORCE_METAL", "1", /*overwrite=*/1);

    for (std::size_t i = 0; i < warmup; ++i)
        run_metal(e.get(), desc, gpu_txs, max_epochs);
    std::vector<double> metal;
    metal.reserve(runs);
    for (std::size_t i = 0; i < runs; ++i)
        metal.push_back(run_metal(e.get(), desc, gpu_txs, max_epochs));

    for (std::size_t i = 0; i < warmup; ++i)
        run_cpu(desc, cpu_txs);
    std::vector<double> cpu;
    cpu.reserve(runs);
    for (std::size_t i = 0; i < runs; ++i)
        cpu.push_back(run_cpu(desc, cpu_txs));

    Row r{};
    r.n            = n;
    r.cpu_min_ms   = min_of(cpu);
    r.cpu_med_ms   = median(cpu);
    r.metal_min_ms = min_of(metal);
    r.metal_med_ms = median(metal);
    r.speedup_min  = r.cpu_min_ms / r.metal_min_ms;
    return r;
}

}  // namespace

int main(int argc, char** argv)
{
    setvbuf(stdout, nullptr, _IOLBF, 0);
    @autoreleasepool {
        std::printf("[quasar_threshold_sweep] starting\n");
        if (auto e = QuasarGPUEngine::create())
            std::printf("device: %s\n", e->device_name());
        else {
            std::printf("device: metal-unavailable\n");
            return 0;
        }

        // Smaller N gets more runs; larger N gets fewer (each Metal round
        // is expensive at small N because of per-round setup, and CPU is
        // very fast — we want enough samples for stable medians).
        // The Metal substrate has a per-round ingress capacity of 4096
        // (kDefaultRingCapacity in quasar_gpu_engine.mm). Beyond that the
        // engine's push_txs path silently drops txs, so any N>4096 sweep
        // measures 4096-tx Metal vs N-tx CPU — meaningless. Sweep within
        // the substrate's own capacity envelope.
        const uint32_t Ns[]        = {16u, 64u, 256u, 1024u, 2048u, 4096u};
        const std::size_t runs[]   = {  9u,  9u,  9u,    9u,    7u,    7u};
        const std::size_t warmup[] = {  3u,  3u,  3u,    2u,    2u,    2u};

        std::vector<Row> rows;
        for (size_t i = 0; i < sizeof(Ns) / sizeof(Ns[0]); ++i) {
            auto r = sweep_one(Ns[i], runs[i], warmup[i]);
            rows.push_back(r);
            std::printf("N=%-7u cpu_min=%9.3fms cpu_med=%9.3fms"
                        " metal_min=%9.3fms metal_med=%9.3fms speedup_min=%6.3fx\n",
                        r.n, r.cpu_min_ms, r.cpu_med_ms,
                        r.metal_min_ms, r.metal_med_ms, r.speedup_min);
        }

        // Find smallest N where Metal min <= CPU min.
        uint32_t threshold = 0;
        for (const auto& r : rows) {
            if (r.metal_min_ms <= r.cpu_min_ms) { threshold = r.n; break; }
        }
        if (threshold == 0)
            std::printf("\nNo cutover within sweep range — Metal slower than CPU at all sampled N.\n");
        else
            std::printf("\nN_threshold_substrate (smallest N where Metal beats CPU): %u\n", threshold);

        std::printf("\n# Markdown table\n");
        std::printf("| N | CPU min | Metal min | Speedup vs CPU |\n");
        std::printf("|---:|--------:|----------:|---------------:|\n");
        for (const auto& r : rows)
            std::printf("| %u | %.3f ms | %.3f ms | %.3fx |\n",
                        r.n, r.cpu_min_ms, r.metal_min_ms, r.speedup_min);

        std::printf("\n[quasar_threshold_sweep] done\n");
    }
    return 0;
}
