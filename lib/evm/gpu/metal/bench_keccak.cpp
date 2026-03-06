// Copyright (C) 2026, The cevm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file bench_keccak.cpp
/// Benchmark: Keccak-256 CPU (single-thread) vs Metal GPU.
///
/// Hashes 1M random 32-byte inputs and reports throughput and speedup.
///
/// Build:
///   clang++ -std=c++20 -O2 -framework Metal -framework Foundation \
///     bench_keccak.cpp keccak_host.mm -o bench_keccak

#include "keccak_host.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace evm::gpu::metal;

static constexpr size_t NUM_HASHES    = 1'000'000;
static constexpr size_t INPUT_SIZE    = 32;  // bytes per input (state trie node)

int main()
{
    std::printf("Keccak-256 benchmark: %zu inputs x %zu bytes\n\n", NUM_HASHES, INPUT_SIZE);

    // Generate random input data.
    std::mt19937_64 rng(42);  // deterministic seed for reproducibility
    std::vector<uint8_t> data(NUM_HASHES * INPUT_SIZE);
    {
        auto* p = reinterpret_cast<uint64_t*>(data.data());
        for (size_t i = 0; i < data.size() / 8; ++i)
            p[i] = rng();
    }

    // Build input descriptors.
    std::vector<HashInput> inputs(NUM_HASHES);
    for (size_t i = 0; i < NUM_HASHES; ++i)
    {
        inputs[i].data = data.data() + i * INPUT_SIZE;
        inputs[i].length = INPUT_SIZE;
    }

    // -- CPU benchmark --------------------------------------------------------
    std::vector<uint8_t> cpu_results(NUM_HASHES * 32);
    auto cpu_start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < NUM_HASHES; ++i)
        keccak256_cpu(inputs[i].data, inputs[i].length, cpu_results.data() + i * 32);

    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    std::printf("CPU (single-thread):\n");
    std::printf("  Time:       %.1f ms\n", cpu_ms);
    std::printf("  Throughput: %.1f Mhash/s\n", NUM_HASHES / cpu_ms / 1000.0);

    // -- GPU benchmark --------------------------------------------------------
    auto hasher = KeccakHasher::create();
    if (!hasher)
    {
        std::fprintf(stderr, "\nMetal device not available. Skipping GPU benchmark.\n");
        return 1;
    }

    std::printf("\nMetal device: %s\n", hasher->device_name());

    // Warm-up: one small batch to trigger shader compilation.
    {
        HashInput warm = {data.data(), INPUT_SIZE};
        hasher->batch_hash(&warm, 1);
    }

    auto gpu_start = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> gpu_results = hasher->batch_hash(inputs.data(), NUM_HASHES);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();

    std::printf("\nGPU (Metal):\n");
    std::printf("  Time:       %.1f ms\n", gpu_ms);
    std::printf("  Throughput: %.1f Mhash/s\n", NUM_HASHES / gpu_ms / 1000.0);

    // -- Correctness check ----------------------------------------------------
    size_t mismatches = 0;
    for (size_t i = 0; i < NUM_HASHES; ++i)
    {
        if (std::memcmp(cpu_results.data() + i * 32, gpu_results.data() + i * 32, 32) != 0)
        {
            ++mismatches;
            if (mismatches <= 3)
            {
                std::printf("\nMISMATCH at index %zu:\n  CPU: ", i);
                for (int b = 0; b < 32; ++b)
                    std::printf("%02x", cpu_results[i * 32 + b]);
                std::printf("\n  GPU: ");
                for (int b = 0; b < 32; ++b)
                    std::printf("%02x", gpu_results[i * 32 + b]);
                std::printf("\n");
            }
        }
    }

    // -- Summary --------------------------------------------------------------
    std::printf("\n--- Summary ---\n");
    if (mismatches > 0)
    {
        std::printf("FAIL: %zu/%zu hash mismatches\n", mismatches, NUM_HASHES);
        return 1;
    }

    std::printf("Correctness: PASS (%zu hashes verified)\n", NUM_HASHES);
    std::printf("Speedup:     %.2fx (GPU vs single-thread CPU)\n", cpu_ms / gpu_ms);

    return 0;
}
