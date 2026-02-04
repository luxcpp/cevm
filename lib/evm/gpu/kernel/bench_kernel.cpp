// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file bench_kernel.cpp
/// Benchmark: GPU EVM kernel vs CPU interpreter.
///
/// Generates 10k simple transactions (ADD/MUL/PUSH/POP loops, no storage)
/// and compares GPU kernel execution time against the CPU reference
/// interpreter from evm_kernel_host.hpp.
///
/// Build: cmake --build build --target evm-bench-kernel
/// Usage: evm-bench-kernel [num_txs] [num_runs]

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
///   JUMPDEST   ; loop:
///   PUSH1 3
///   MUL        ; accumulator *= 3
///   DUP1
///   PUSH1 0
///   ISZERO     ; check if zero (overflow wraps)
///   PUSH1 <end>
///   JUMPI      ; if zero, exit
///   DUP1
///   PUSH2 <N>
///   LT
///   PUSH1 2
///   JUMPI
///   end: STOP
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
        // Alternate between add and mul loops for variety.
        tx.code = (i % 2 == 0) ? add_loop : mul_loop;
        // Gas: ~30 gas per iteration + overhead. Be generous.
        tx.gas_limit = static_cast<uint64_t>(iterations) * 100 + 10000;
        // Distinct addresses.
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
    return s;
}

}  // anonymous namespace

int main(int argc, char* argv[])
{
    uint32_t num_txs = 10000;
    uint32_t num_runs = 3;
    uint16_t iterations = 100;  // loop iterations per transaction

    if (argc > 1) num_txs = static_cast<uint32_t>(std::atoi(argv[1]));
    if (argc > 2) num_runs = static_cast<uint32_t>(std::atoi(argv[2]));
    if (argc > 3) iterations = static_cast<uint16_t>(std::atoi(argv[3]));

    std::printf("GPU EVM Kernel Benchmark\n");
    std::printf("========================\n");
    std::printf("Transactions: %u\n", num_txs);
    std::printf("Loop iters:   %u per tx\n", iterations);
    std::printf("Runs:         %u\n", num_runs);
    std::printf("\n");

    auto txs = generate_transactions(num_txs, iterations);

    // -- CPU Reference (sequential) -------------------------------------------
    std::printf("Running CPU sequential...\n");
    std::vector<double> cpu_times;
    cpu_times.reserve(num_runs);

    uint32_t cpu_ok_count = 0;
    for (uint32_t r = 0; r < num_runs; ++r)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        cpu_ok_count = 0;
        for (const auto& tx : txs)
        {
            auto result = evm::gpu::kernel::execute_cpu(tx);
            if (result.status == TxStatus::Stop || result.status == TxStatus::Return)
                ++cpu_ok_count;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        cpu_times.push_back(ms);
    }
    auto cpu_stats = compute_stats(cpu_times);

    std::printf("  CPU: %.1f +/- %.1f ms  (%u/%u OK)\n",
        cpu_stats.mean_ms, cpu_stats.stddev_ms, cpu_ok_count, num_txs);

    // -- GPU Kernel -----------------------------------------------------------
    auto gpu_host = evm::gpu::kernel::EvmKernelHost::create();
    if (gpu_host)
    {
        std::printf("GPU device: %s\n", gpu_host->device_name());
        std::printf("Running GPU kernel...\n");

        std::vector<double> gpu_times;
        gpu_times.reserve(num_runs);

        uint32_t gpu_ok_count = 0;
        uint32_t gpu_fallback_count = 0;

        for (uint32_t r = 0; r < num_runs; ++r)
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto results = gpu_host->execute(txs);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            gpu_times.push_back(ms);

            gpu_ok_count = 0;
            gpu_fallback_count = 0;
            for (const auto& res : results)
            {
                if (res.status == TxStatus::Stop || res.status == TxStatus::Return)
                    ++gpu_ok_count;
                if (res.status == TxStatus::CallNotSupported)
                    ++gpu_fallback_count;
            }
        }
        auto gpu_stats = compute_stats(gpu_times);

        std::printf("  GPU: %.1f +/- %.1f ms  (%u/%u OK, %u fallback)\n",
            gpu_stats.mean_ms, gpu_stats.stddev_ms, gpu_ok_count, num_txs, gpu_fallback_count);

        // -- Results table ----------------------------------------------------
        std::printf("\n");
        std::printf("%-18s| %-16s| %s\n", "Mode", "Time (ms)", "Speedup");
        std::printf("%-18s| %-16s| %s\n", "------------------", "----------------", "-------");

        auto print_row = [&](const char* label, const RunStats& s, double baseline_ms) {
            double speedup = (baseline_ms > 0) ? baseline_ms / s.mean_ms : 1.0;
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.1f +/- %.1f", s.mean_ms, s.stddev_ms);
            std::printf("%-18s| %-16s| %.1fx\n", label, buf, speedup);
        };

        print_row("CPU Sequential", cpu_stats, cpu_stats.mean_ms);
        print_row("GPU Metal", gpu_stats, cpu_stats.mean_ms);
    }
    else
    {
        std::printf("Metal GPU not available. Skipping GPU benchmark.\n");
        std::printf("\n");
        std::printf("%-18s| %-16s| %s\n", "Mode", "Time (ms)", "Speedup");
        std::printf("%-18s| %-16s| %s\n", "------------------", "----------------", "-------");

        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.1f +/- %.1f", cpu_stats.mean_ms, cpu_stats.stddev_ms);
        std::printf("%-18s| %-16s| %.1fx\n", "CPU Sequential", buf, 1.0);
        std::printf("%-18s| %-16s| %s\n", "GPU Metal", "--", "--");
    }

    std::printf("\n");
    return 0;
}
