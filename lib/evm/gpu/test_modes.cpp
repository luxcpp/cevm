// Copyright (C) 2026, The cevm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file test_modes.cpp
/// Tests that all execution modes (sequential, parallel, GPU) produce
/// identical results for the same block of transactions.
///
/// Also verifies that the GPU Keccak-256 batch hasher matches CPU output.
///
/// Build:
///   Linked via evm-gpu target; see lib/evm/CMakeLists.txt.

#include "gpu_dispatch.hpp"
#include "gpu_state_hasher.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// =============================================================================
// Test Harness
// =============================================================================

static int g_passed = 0;
static int g_failed = 0;

#define CHECK(cond, msg)                                                       \
    do                                                                         \
    {                                                                          \
        if (cond)                                                              \
        {                                                                      \
            ++g_passed;                                                        \
            std::printf("  [PASS] %s\n", msg);                                 \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            ++g_failed;                                                        \
            std::printf("  [FAIL] %s (line %d)\n", msg, __LINE__);             \
        }                                                                      \
    } while (false)

// =============================================================================
// Helpers
// =============================================================================

static evm::gpu::Transaction make_transfer(uint64_t nonce, uint64_t value, uint64_t gas)
{
    evm::gpu::Transaction tx;
    tx.from.resize(20, 0);
    tx.from[19] = 0x01;
    tx.to.resize(20, 0);
    tx.to[19] = 0x02;
    tx.nonce = nonce;
    tx.value = value;
    tx.gas_limit = gas;
    tx.gas_price = 1;
    return tx;
}

static std::vector<evm::gpu::Transaction> make_block(size_t num_txs)
{
    std::vector<evm::gpu::Transaction> txs;
    txs.reserve(num_txs);
    for (size_t i = 0; i < num_txs; ++i)
        txs.push_back(make_transfer(i, 1000 + i, 21000));
    return txs;
}

// =============================================================================
// Test: Sequential vs Parallel produce same gas results
// =============================================================================

static void test_sequential_vs_parallel()
{
    std::printf("\n=== Sequential vs Parallel ===\n");

    auto txs = make_block(16);

    evm::gpu::Config seq_cfg;
    seq_cfg.backend = evm::gpu::Backend::CPU_Sequential;

    evm::gpu::Config par_cfg;
    par_cfg.backend = evm::gpu::Backend::CPU_Parallel;
    par_cfg.num_threads = 4;

    auto seq_result = evm::gpu::execute_block(seq_cfg, txs, nullptr);
    auto par_result = evm::gpu::execute_block(par_cfg, txs, nullptr);

    CHECK(seq_result.gas_used.size() == par_result.gas_used.size(),
          "Same number of gas entries");

    bool gas_match = true;
    for (size_t i = 0; i < seq_result.gas_used.size(); ++i)
    {
        if (seq_result.gas_used[i] != par_result.gas_used[i])
        {
            gas_match = false;
            break;
        }
    }
    CHECK(gas_match, "Gas used matches between sequential and parallel");
    CHECK(seq_result.total_gas == par_result.total_gas, "Total gas matches");
}

// =============================================================================
// Test: GPU backend (falls back to CPU parallel) produces same results
// =============================================================================

static void test_gpu_fallback_same_results()
{
    std::printf("\n=== GPU Backend vs Sequential ===\n");

    auto txs = make_block(8);

    evm::gpu::Config seq_cfg;
    seq_cfg.backend = evm::gpu::Backend::CPU_Sequential;
    // GPU_Metal value-transfer-only execution returns the deterministic
    // gas-limit per tx via BlockStmGpu; opt the CPU comparator into the
    // matching gas_estimation shortcut so the two paths converge.
    seq_cfg.gas_estimation_mode = true;

    evm::gpu::Config gpu_cfg;
    gpu_cfg.backend = evm::gpu::Backend::GPU_Metal;
    gpu_cfg.gas_estimation_mode = true;

    auto seq_result = evm::gpu::execute_block(seq_cfg, txs, nullptr);
    auto gpu_result = evm::gpu::execute_block(gpu_cfg, txs, nullptr);

    CHECK(seq_result.total_gas == gpu_result.total_gas,
          "GPU fallback total gas matches sequential");

    bool gas_match = (seq_result.gas_used == gpu_result.gas_used);
    CHECK(gas_match, "GPU fallback per-tx gas matches sequential");
}

// =============================================================================
// Test: Keccak-256 GPU hasher produces correct results
// =============================================================================

static void test_keccak256_known_vectors()
{
    std::printf("\n=== Keccak-256 Known Vectors ===\n");

    evm::gpu::GpuStateHasher hasher(LUX_BACKEND_CPU);
    CHECK(hasher.available(), "CPU Keccak hasher available");

    // Empty input: Keccak-256("") = c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470
    {
        uint8_t out[32] = {};
        uint8_t empty_data = 0;
        bool ok = hasher.hash(&empty_data, 0, out);

        static const uint8_t expected[32] = {
            0xc5, 0xd2, 0x46, 0x01, 0x86, 0xf7, 0x23, 0x3c,
            0x92, 0x7e, 0x7d, 0xb2, 0xdc, 0xc7, 0x03, 0xc0,
            0xe5, 0x00, 0xb6, 0x53, 0xca, 0x82, 0x27, 0x3b,
            0x7b, 0xfa, 0xd8, 0x04, 0x5d, 0x85, 0xa4, 0x70,
        };
        CHECK(ok && std::memcmp(out, expected, 32) == 0,
              "Keccak-256('') matches known vector");
    }

    // "abc": Keccak-256("abc") = 4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45
    {
        const uint8_t abc[] = {'a', 'b', 'c'};
        uint8_t out[32] = {};
        bool ok = hasher.hash(abc, 3, out);

        static const uint8_t expected[32] = {
            0x4e, 0x03, 0x65, 0x7a, 0xea, 0x45, 0xa9, 0x4f,
            0xc7, 0xd4, 0x7b, 0xa8, 0x26, 0xc8, 0xd6, 0x67,
            0xc0, 0xd1, 0xe6, 0xe3, 0x3a, 0x64, 0xa0, 0x36,
            0xec, 0x44, 0xf5, 0x8f, 0xa1, 0x2d, 0x6c, 0x45,
        };
        CHECK(ok && std::memcmp(out, expected, 32) == 0,
              "Keccak-256('abc') matches known vector");
    }
}

// =============================================================================
// Test: Keccak-256 batch consistency
// =============================================================================

static void test_keccak256_batch_consistency()
{
    std::printf("\n=== Keccak-256 Batch Consistency ===\n");

    evm::gpu::GpuStateHasher hasher(LUX_BACKEND_CPU);
    CHECK(hasher.available(), "Hasher available for batch test");

    // Hash 4 inputs individually, then as a batch, compare.
    const char* inputs[] = {"hello", "world", "ethereum", ""};
    const size_t num = 4;
    size_t lens[4];
    for (size_t i = 0; i < num; ++i)
        lens[i] = std::strlen(inputs[i]);

    // Individual hashes
    uint8_t individual[4][32];
    for (size_t i = 0; i < num; ++i)
    {
        hasher.hash(reinterpret_cast<const uint8_t*>(inputs[i]), lens[i],
                    individual[i]);
    }

    // Batch hash via scattered API
    const uint8_t* ptrs[4];
    for (size_t i = 0; i < num; ++i)
        ptrs[i] = reinterpret_cast<const uint8_t*>(inputs[i]);

    uint8_t batch[4 * 32];
    bool ok = hasher.batch_hash_scattered(ptrs, lens, num, batch);
    CHECK(ok, "Batch hash succeeded");

    bool match = true;
    for (size_t i = 0; i < num; ++i)
    {
        if (std::memcmp(individual[i], batch + i * 32, 32) != 0)
        {
            match = false;
            std::printf("    Mismatch at input %zu\n", i);
            break;
        }
    }
    CHECK(match, "Batch results match individual results");
}

// =============================================================================
// Test: State root with enable_state_trie_gpu
// =============================================================================

static void test_state_root_hashing()
{
    std::printf("\n=== State Root GPU Hashing ===\n");

    auto txs = make_block(4);

    // Sequential with GPU hashing
    evm::gpu::Config cfg_seq;
    cfg_seq.backend = evm::gpu::Backend::CPU_Sequential;
    cfg_seq.enable_state_trie_gpu = true;

    // Parallel with GPU hashing
    evm::gpu::Config cfg_par;
    cfg_par.backend = evm::gpu::Backend::CPU_Parallel;
    cfg_par.enable_state_trie_gpu = true;

    auto r_seq = evm::gpu::execute_block(cfg_seq, txs, nullptr);
    auto r_par = evm::gpu::execute_block(cfg_par, txs, nullptr);

    CHECK(r_seq.state_root.size() == 32, "Sequential state root is 32 bytes");
    CHECK(r_par.state_root.size() == 32, "Parallel state root is 32 bytes");

    bool root_nonzero = false;
    for (auto b : r_seq.state_root)
    {
        if (b != 0)
        {
            root_nonzero = true;
            break;
        }
    }
    CHECK(root_nonzero, "State root is non-zero");

    CHECK(r_seq.state_root == r_par.state_root,
          "Sequential and parallel produce same state root");
}

// =============================================================================
// Test: auto_detect and set_backend
// =============================================================================

static void test_backend_selection()
{
    std::printf("\n=== Backend Selection ===\n");

    auto detected = evm::gpu::auto_detect();
    CHECK(static_cast<uint8_t>(detected) <= 3, "auto_detect returns valid backend");
    std::printf("    auto_detect chose: %s\n", evm::gpu::backend_name(detected));

    evm::gpu::Config cfg;
    bool ok = evm::gpu::set_backend(cfg, evm::gpu::Backend::CPU_Sequential);
    CHECK(ok, "set_backend(CPU_Sequential) succeeds");
    CHECK(cfg.backend == evm::gpu::Backend::CPU_Sequential,
          "Config backend updated to CPU_Sequential");

    ok = evm::gpu::set_backend(cfg, evm::gpu::Backend::CPU_Parallel);
    CHECK(ok, "set_backend(CPU_Parallel) succeeds");
    CHECK(cfg.backend == evm::gpu::Backend::CPU_Parallel,
          "Config backend updated to CPU_Parallel");

    auto backends = evm::gpu::available_backends();
    CHECK(backends.size() >= 2, "At least 2 backends available (CPU seq + par)");
}

// =============================================================================
// Main
// =============================================================================

int main()
{
    std::printf("=== EVM GPU Mode Consistency Tests ===\n");

    test_sequential_vs_parallel();
    test_gpu_fallback_same_results();
    test_keccak256_known_vectors();
    test_keccak256_batch_consistency();
    test_state_root_hashing();
    test_backend_selection();

    std::printf("\n========================================\n");
    std::printf("Passed: %d  Failed: %d\n", g_passed, g_failed);
    std::printf("========================================\n");

    return g_failed > 0 ? 1 : 0;
}
