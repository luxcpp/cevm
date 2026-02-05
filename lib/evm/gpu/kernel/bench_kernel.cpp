// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file bench_kernel.cpp
/// Benchmark: GPU EVM kernel V1 vs V2 vs CPU interpreter.
///
/// Generates N transactions with compute loops (ADD/MUL/PUSH/POP)
/// and compares GPU V1 (1 thread/tx), GPU V2 (32 threads/tx SIMD),
/// and CPU sequential.
///
/// Build: cmake --build build --target evm-bench-kernel
/// Usage: evm-bench-kernel [num_txs] [num_runs] [iterations]

#include "evm_kernel_host.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace {

using namespace evm::gpu::kernel;

/// Build bytecode for a simple compute loop:
///   PUSH1 0        ; counter = 0
///   JUMPDEST       ; loop:
///   PUSH1 1
///   ADD            ; counter += 1
///   DUP1
///   PUSH2 <N>      ; push iteration count
///   SWAP1          ; stack: [counter, N, counter]
///   LT             ; pops counter(top), N -> counter < N?
///   PUSH1 2        ; jump target (JUMPDEST at offset 2)
///   JUMPI          ; if so, loop
///   STOP
///
/// This exercises PUSH, ADD, DUP, SWAP, LT, JUMPI -- the bread and butter
/// of real EVM execution. Each iteration costs ~35 gas.
std::vector<uint8_t> build_loop_bytecode(uint16_t iterations)
{
    std::vector<uint8_t> code;

    // PUSH1 0
    code.push_back(0x60);
    code.push_back(0x00);

    // JUMPDEST (offset 2)
    code.push_back(0x5b);

    // PUSH1 1
    code.push_back(0x60);
    code.push_back(0x01);

    // ADD
    code.push_back(0x01);

    // DUP1
    code.push_back(0x80);

    // PUSH2 <iterations>
    code.push_back(0x61);
    code.push_back(static_cast<uint8_t>(iterations >> 8));
    code.push_back(static_cast<uint8_t>(iterations & 0xFF));

    // SWAP1: stack [counter, counter, N] -> [counter, N, counter]
    code.push_back(0x90);

    // LT: pops counter(top), N(second) -> counter < N
    code.push_back(0x10);

    // PUSH1 2 (JUMPDEST offset)
    code.push_back(0x60);
    code.push_back(0x02);

    // JUMPI
    code.push_back(0x57);

    // POP (clean up counter)
    code.push_back(0x50);

    // STOP
    code.push_back(0x00);

    return code;
}

/// Build bytecode for a multiply-accumulate loop:
///   PUSH1 1    ; accumulator = 1
///   PUSH1 0    ; counter = 0
///   JUMPDEST   ; loop:
///   PUSH1 1; ADD (increment counter)
///   SWAP1; PUSH1 3; MUL (accumulate *= 3)
///   SWAP1
///   DUP1; PUSH2 <N>; SWAP1; LT
///   PUSH1 4; JUMPI
///   POP; POP; STOP
std::vector<uint8_t> build_mul_loop_bytecode(uint16_t iterations)
{
    std::vector<uint8_t> code;

    // PUSH1 1
    code.push_back(0x60);
    code.push_back(0x01);

    // PUSH1 0 (loop counter)
    code.push_back(0x60);
    code.push_back(0x00);

    // JUMPDEST (offset 4)
    code.push_back(0x5b);

    // PUSH1 1; ADD (increment counter)
    code.push_back(0x60);
    code.push_back(0x01);
    code.push_back(0x01);

    // SWAP1 (bring accumulator to top)
    code.push_back(0x90);

    // PUSH1 3; MUL (accumulate)
    code.push_back(0x60);
    code.push_back(0x03);
    code.push_back(0x02);

    // SWAP1 (bring counter back to top)
    code.push_back(0x90);

    // DUP1; PUSH2 <iterations>; SWAP1; LT
    code.push_back(0x80);
    code.push_back(0x61);
    code.push_back(static_cast<uint8_t>(iterations >> 8));
    code.push_back(static_cast<uint8_t>(iterations & 0xFF));
    code.push_back(0x90);  // SWAP1: [counter, N, counter] for LT
    code.push_back(0x10);  // LT: counter < N

    // PUSH1 4 (JUMPDEST offset); JUMPI
    code.push_back(0x60);
    code.push_back(0x04);
    code.push_back(0x57);

    // POP; POP; STOP
    code.push_back(0x50);
    code.push_back(0x50);
    code.push_back(0x00);

    return code;
}

/// Count the number of opcodes executed per loop iteration.
/// add_loop: JUMPDEST(1) + PUSH1(1) + ADD(1) + DUP1(1) + PUSH2(1) + SWAP1(1) + LT(1) + PUSH1(1) + JUMPI(1) = 9 ops/iter
/// + PUSH1(initial) + POP + STOP = 3 ops overhead
/// mul_loop: JUMPDEST(1) + PUSH1(1) + ADD(1) + SWAP1(1) + PUSH1(1) + MUL(1) + SWAP1(1) + DUP1(1) + PUSH2(1) + SWAP1(1) + LT(1) + PUSH1(1) + JUMPI(1) = 13 ops/iter
/// + PUSH1 + PUSH1 + POP + POP + STOP = 5 ops overhead
uint64_t count_opcodes(uint32_t num_txs, uint16_t iterations)
{
    // Half add_loop, half mul_loop.
    uint64_t add_txs = num_txs / 2 + (num_txs % 2);
    uint64_t mul_txs = num_txs / 2;
    uint64_t add_ops = add_txs * (9ULL * iterations + 3);
    uint64_t mul_ops = mul_txs * (13ULL * iterations + 5);
    return add_ops + mul_ops;
}

/// Generate N transactions with loop bytecodes.
std::vector<HostTransaction> generate_transactions(uint32_t num_txs, uint16_t iterations)
{
    std::vector<HostTransaction> txs;
    txs.reserve(num_txs);

    auto add_loop = build_loop_bytecode(iterations);
    auto mul_loop = build_mul_loop_bytecode(iterations);

    for (uint32_t i = 0; i < num_txs; ++i)
    {
        HostTransaction tx;
        tx.code = (i % 2 == 0) ? add_loop : mul_loop;
        tx.gas_limit = static_cast<uint64_t>(iterations) * 100 + 10000;
        tx.caller = uint256{static_cast<uint64_t>(i + 1)};
        tx.address = uint256{static_cast<uint64_t>(i + num_txs + 1)};
        tx.value = uint256::zero();
        txs.push_back(std::move(tx));
    }

    return txs;
}

struct RunStats
{
    double mean_ms;
    double stddev_ms;
    double min_ms;
};

RunStats compute_stats(const std::vector<double>& times_ms)
{
    RunStats s{};
    double n = static_cast<double>(times_ms.size());
    s.mean_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / n;
    double var = 0;
    for (auto t : times_ms)
        var += (t - s.mean_ms) * (t - s.mean_ms);
    s.stddev_ms = std::sqrt(var / n);
    s.min_ms = *std::min_element(times_ms.begin(), times_ms.end());
    return s;
}

/// Verify that V2 results match V1 results exactly.
bool verify_results(const std::vector<TxResult>& v1, const std::vector<TxResult>& v2, uint32_t num_txs)
{
    if (v1.size() != v2.size()) return false;
    uint32_t mismatches = 0;
    for (size_t i = 0; i < v1.size(); ++i) {
        if (v1[i].status != v2[i].status || v1[i].gas_used != v2[i].gas_used) {
            if (mismatches < 5) {
                std::printf("  MISMATCH tx %zu: V1 status=%d gas=%llu, V2 status=%d gas=%llu\n",
                    i, (int)v1[i].status, (unsigned long long)v1[i].gas_used,
                    (int)v2[i].status, (unsigned long long)v2[i].gas_used);
            }
            mismatches++;
        }
    }
    if (mismatches > 0)
        std::printf("  TOTAL MISMATCHES: %u / %u\n", mismatches, num_txs);
    return mismatches == 0;
}

}  // anonymous namespace

int main(int argc, char* argv[])
{
    uint32_t num_txs = 10000;
    uint32_t num_runs = 5;
    uint16_t iterations = 100;

    if (argc > 1) num_txs = static_cast<uint32_t>(std::atoi(argv[1]));
    if (argc > 2) num_runs = static_cast<uint32_t>(std::atoi(argv[2]));
    if (argc > 3) iterations = static_cast<uint16_t>(std::atoi(argv[3]));

    uint64_t total_ops = count_opcodes(num_txs, iterations);

    std::printf("GPU EVM Kernel Benchmark (V1 vs V2)\n");
    std::printf("====================================\n");
    std::printf("Transactions:   %u\n", num_txs);
    std::printf("Loop iters:     %u per tx\n", iterations);
    std::printf("Total opcodes:  %llu\n", (unsigned long long)total_ops);
    std::printf("Runs:           %u\n", num_runs);
    std::printf("\n");

    auto txs = generate_transactions(num_txs, iterations);

    // -- CPU Reference (sequential) -------------------------------------------
    std::printf("Running CPU sequential...\n");
    std::vector<double> cpu_times;
    cpu_times.reserve(num_runs);

    std::vector<TxResult> cpu_results;
    uint32_t cpu_ok_count = 0;
    for (uint32_t r = 0; r < num_runs; ++r)
    {
        cpu_results.clear();
        cpu_results.reserve(num_txs);
        auto t0 = std::chrono::high_resolution_clock::now();
        cpu_ok_count = 0;
        for (const auto& tx : txs)
        {
            auto result = evm::gpu::kernel::execute_cpu(tx);
            if (result.status == TxStatus::Stop || result.status == TxStatus::Return)
                ++cpu_ok_count;
            cpu_results.push_back(std::move(result));
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        cpu_times.push_back(ms);
    }
    auto cpu_stats = compute_stats(cpu_times);
    double cpu_ops_sec = total_ops / (cpu_stats.min_ms / 1000.0);

    std::printf("  CPU: %.1f +/- %.1f ms (min %.1f ms) | %u/%u OK | %.2f M ops/sec\n",
        cpu_stats.mean_ms, cpu_stats.stddev_ms, cpu_stats.min_ms,
        cpu_ok_count, num_txs, cpu_ops_sec / 1e6);

    // -- GPU Kernel -----------------------------------------------------------
    auto gpu_host = evm::gpu::kernel::EvmKernelHost::create();
    if (!gpu_host)
    {
        std::printf("Metal GPU not available. Skipping GPU benchmarks.\n");
        return 0;
    }

    std::printf("GPU device: %s\n", gpu_host->device_name());

    // -- GPU V1 ---------------------------------------------------------------
    {
        std::printf("\nRunning GPU V1 (1 thread/tx)...\n");
        std::vector<double> gpu_times;
        gpu_times.reserve(num_runs);

        std::vector<TxResult> v1_results;
        uint32_t gpu_ok = 0, gpu_fallback = 0;

        for (uint32_t r = 0; r < num_runs; ++r)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            v1_results = gpu_host->execute(txs);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            gpu_times.push_back(ms);

            gpu_ok = 0; gpu_fallback = 0;
            for (const auto& res : v1_results) {
                if (res.status == TxStatus::Stop || res.status == TxStatus::Return) ++gpu_ok;
                if (res.status == TxStatus::CallNotSupported) ++gpu_fallback;
            }
        }
        auto gpu_stats = compute_stats(gpu_times);
        double gpu_ops_sec = total_ops / (gpu_stats.min_ms / 1000.0);

        std::printf("  V1:  %.1f +/- %.1f ms (min %.1f ms) | %u/%u OK | %.2f M ops/sec\n",
            gpu_stats.mean_ms, gpu_stats.stddev_ms, gpu_stats.min_ms,
            gpu_ok, num_txs, gpu_ops_sec / 1e6);

        // Verify V1 matches CPU.
        std::printf("  Verifying V1 vs CPU...\n");
        if (verify_results(cpu_results, v1_results, num_txs))
            std::printf("  V1 vs CPU: PASS (gas_used matches)\n");
        else
            std::printf("  V1 vs CPU: FAIL\n");
    }

    // -- GPU V2 ---------------------------------------------------------------
    if (gpu_host->has_v2())
    {
        std::printf("\nRunning GPU V2 (32 threads/tx SIMD)...\n");
        std::vector<double> gpu_times;
        gpu_times.reserve(num_runs);

        std::vector<TxResult> v2_results;
        uint32_t gpu_ok = 0, gpu_fallback = 0;

        for (uint32_t r = 0; r < num_runs; ++r)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            v2_results = gpu_host->execute_v2(txs);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            gpu_times.push_back(ms);

            gpu_ok = 0; gpu_fallback = 0;
            for (const auto& res : v2_results) {
                if (res.status == TxStatus::Stop || res.status == TxStatus::Return) ++gpu_ok;
                if (res.status == TxStatus::CallNotSupported) ++gpu_fallback;
            }
        }
        auto gpu_stats = compute_stats(gpu_times);
        double gpu_ops_sec = total_ops / (gpu_stats.min_ms / 1000.0);

        std::printf("  V2:  %.1f +/- %.1f ms (min %.1f ms) | %u/%u OK | %.2f M ops/sec\n",
            gpu_stats.mean_ms, gpu_stats.stddev_ms, gpu_stats.min_ms,
            gpu_ok, num_txs, gpu_ops_sec / 1e6);

        // Verify V2 matches V1.
        std::printf("  Verifying V2 vs CPU...\n");
        if (verify_results(cpu_results, v2_results, num_txs))
            std::printf("  V2 vs CPU: PASS (gas_used matches)\n");
        else
            std::printf("  V2 vs CPU: FAIL\n");

        // -- Summary ----------------------------------------------------------
        std::printf("\n");
        std::printf("%-20s| %-22s| %-16s| %s\n", "Mode", "Time (ms)", "Ops/sec", "Speedup");
        std::printf("%-20s| %-22s| %-16s| %s\n", "--------------------", "----------------------", "----------------", "-------");

        auto print_row = [&](const char* label, const RunStats& s, double ops, double baseline_ms) {
            double speedup = (baseline_ms > 0) ? baseline_ms / s.min_ms : 1.0;
            char time_buf[40];
            std::snprintf(time_buf, sizeof(time_buf), "%.1f +/- %.1f (min %.1f)", s.mean_ms, s.stddev_ms, s.min_ms);
            char ops_buf[32];
            if (ops >= 1e9)
                std::snprintf(ops_buf, sizeof(ops_buf), "%.2f B", ops / 1e9);
            else
                std::snprintf(ops_buf, sizeof(ops_buf), "%.2f M", ops / 1e6);
            std::printf("%-20s| %-22s| %-16s| %.1fx\n", label, time_buf, ops_buf, speedup);
        };

        print_row("CPU Sequential", cpu_stats, cpu_ops_sec, cpu_stats.min_ms);

        auto v1_stats = compute_stats([&]{
            std::vector<double> t; t.reserve(num_runs);
            for (uint32_t r = 0; r < num_runs; ++r) {
                auto t0 = std::chrono::high_resolution_clock::now();
                gpu_host->execute(txs);
                auto t1 = std::chrono::high_resolution_clock::now();
                t.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            return t;
        }());
        double v1_ops = total_ops / (v1_stats.min_ms / 1000.0);
        print_row("GPU V1 (1 thr/tx)", v1_stats, v1_ops, cpu_stats.min_ms);
        print_row("GPU V2 (32 thr/tx)", gpu_stats, gpu_ops_sec, cpu_stats.min_ms);
    }
    else
    {
        std::printf("\nGPU V2 kernel not available. Skipping V2 benchmark.\n");
    }

    std::printf("\n");
    return 0;
}
