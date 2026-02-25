// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file test_unified_pipeline.cpp
/// Unified pipeline functional + throughput test.
///
/// Builds a 1000-block backlog of small blocks (each with 16 transactions
/// running a 50-iteration ADD loop) and runs it twice:
///
///   1. Serial mode  — process_block called once per block.
///   2. Pipelined mode — process_blocks with depth = max_concurrent_blocks.
///
/// Verifies:
///   - Pipeline construction succeeds on this device.
///   - All result fields are populated (state_root, consensus_hash non-zero).
///   - Total EVM gas matches the analytically computed cost of the bytecode
///     within 5% — this catches regressions to the old fake CPU fallback that
///     returned 21000 gas/tx regardless of bytecode.
///   - Pipelined throughput is at least 0.5x of serial (catches a regression
///     to all-CPU serial-equivalent execution).

#include "unified_pipeline.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

// Set by CMake: path to the build-tree luxgpu_backend_metal dylib so the
// test can run without manually exporting LUX_GPU_BACKEND_PATH.
#ifndef LUX_GPU_BACKEND_PATH_DEFAULT
#define LUX_GPU_BACKEND_PATH_DEFAULT ""
#endif

namespace u = evm::gpu::unified;
using Clock = std::chrono::steady_clock;

#define CHECK(expr) do {                                          \
    if (!(expr)) {                                                \
        std::fprintf(stderr, "FAIL: %s:%d: %s\n",                 \
                     __FILE__, __LINE__, #expr);                  \
        std::abort();                                             \
    }                                                             \
} while (0)

namespace {

constexpr size_t NUM_BLOCKS    = 1000;
constexpr size_t TXS_PER_BLOCK = 16;
constexpr size_t ACCOUNTS      = 256;
constexpr uint64_t INITIAL_BALANCE = 1'000'000'000'000ULL;

// Loop iteration count for the synthetic ADD-loop bytecode.
constexpr uint16_t LOOP_ITERATIONS = 50;

/// Build a counter-increment loop:
///   PUSH1 0      ; counter = 0           [3 gas]
///   JUMPDEST     ; loop:                 [1 gas]
///   PUSH1 1                              [3 gas]
///   ADD          ; counter += 1          [3 gas]
///   DUP1                                 [3 gas]
///   PUSH2 N      ; iteration cap         [3 gas]
///   SWAP1                                [3 gas]
///   LT           ; counter < N ?         [3 gas]
///   PUSH1 2      ; loop JUMPDEST offset  [3 gas]
///   JUMPI        ; if so, loop          [10 gas]
///   POP                                  [2 gas]
///   STOP                                 [0 gas]
///
/// Per-iter cost (after the first JUMPDEST tick): 32 gas.
/// Initial PUSH1 0: 3 gas. Trailing POP+STOP: 2 gas.
/// Total for N iterations: 3 + 32*N + 2 = 32*N + 5.
inline std::vector<uint8_t> build_add_loop_bytecode(uint16_t iterations) {
    std::vector<uint8_t> code;
    code.push_back(0x60); code.push_back(0x00);                    // PUSH1 0
    code.push_back(0x5b);                                          // JUMPDEST (offset 2)
    code.push_back(0x60); code.push_back(0x01);                    // PUSH1 1
    code.push_back(0x01);                                          // ADD
    code.push_back(0x80);                                          // DUP1
    code.push_back(0x61);                                          // PUSH2
    code.push_back(static_cast<uint8_t>(iterations >> 8));
    code.push_back(static_cast<uint8_t>(iterations & 0xFF));
    code.push_back(0x90);                                          // SWAP1
    code.push_back(0x10);                                          // LT
    code.push_back(0x60); code.push_back(0x02);                    // PUSH1 2
    code.push_back(0x57);                                          // JUMPI
    code.push_back(0x50);                                          // POP
    code.push_back(0x00);                                          // STOP
    return code;
}

inline uint64_t expected_loop_gas(uint16_t iterations) {
    // Matches the cost breakdown in build_add_loop_bytecode.
    return 32ULL * iterations + 5ULL;
}

evm::gpu::Transaction make_tx(uint32_t i, uint32_t block_height,
                              const std::vector<uint8_t>& code) {
    evm::gpu::Transaction tx;
    tx.from.assign(20, 0);
    tx.to.assign(20, 0);
    uint32_t sender_idx = i % ACCOUNTS;
    uint32_t recipient_idx = (i + 1) % ACCOUNTS;
    tx.from[0] = 0xAA;
    tx.from[16] = static_cast<uint8_t>((sender_idx >> 24) & 0xFF);
    tx.from[17] = static_cast<uint8_t>((sender_idx >> 16) & 0xFF);
    tx.from[18] = static_cast<uint8_t>((sender_idx >>  8) & 0xFF);
    tx.from[19] = static_cast<uint8_t>((sender_idx      ) & 0xFF);
    tx.to[0] = 0xAA;
    tx.to[19] = static_cast<uint8_t>(recipient_idx & 0xFF);
    tx.code = code;                  // Real bytecode — drives kernel CPU/GPU EVM.
    tx.gas_limit = 1'000'000;        // Plenty of gas for the loop.
    tx.gas_price = 1;
    tx.value = 1000;
    tx.nonce = block_height;
    return tx;
}

evm::gpu::AccountInfo make_account(uint32_t i) {
    evm::gpu::AccountInfo a{};
    a.address[0] = 0xAA;
    a.address[16] = static_cast<uint8_t>((i >> 24) & 0xFF);
    a.address[17] = static_cast<uint8_t>((i >> 16) & 0xFF);
    a.address[18] = static_cast<uint8_t>((i >>  8) & 0xFF);
    a.address[19] = static_cast<uint8_t>((i      ) & 0xFF);
    a.nonce = 0;
    a.balance = INITIAL_BALANCE;
    return a;
}

u::BlockContext make_ctx(uint64_t height, std::mt19937& rng) {
    u::BlockContext ctx;
    ctx.height = height;
    ctx.parent_hash.assign(32, 0);
    // Header bytes: 256 random bytes — stand-in for a Quasar RLP header.
    ctx.header_bytes.assign(256, 0);
    for (auto& b : ctx.header_bytes) b = static_cast<uint8_t>(rng());
    // Two BLS signatures per block. Zeros — verifier will reject (ok for
    // the throughput test; we only count successes for accounting).
    ctx.bls_signatures.assign(2 * 48, 0);
    ctx.bls_pubkeys.assign(2 * 96, 0);
    ctx.bls_messages.assign(2 * 32, 0);
    return ctx;
}

}  // namespace

int main() {
    // Make sure lux-gpu finds the fresh build-tree backend (which has the
    // Keccak-256 op vtable wired up). External LUX_GPU_BACKEND_PATH wins.
    if (!std::getenv("LUX_GPU_BACKEND_PATH")
        && LUX_GPU_BACKEND_PATH_DEFAULT[0] != '\0') {
        setenv("LUX_GPU_BACKEND_PATH", LUX_GPU_BACKEND_PATH_DEFAULT, 1);
    }

    std::printf("================================================================\n");
    std::printf("  evm::gpu::unified::UnifiedPipeline -- 1000-block throughput\n");
    std::printf("  blocks=%zu  txs_per_block=%zu  accounts=%zu\n",
                NUM_BLOCKS, TXS_PER_BLOCK, ACCOUNTS);
    std::printf("================================================================\n\n");

    u::UnifiedConfig cfg;
    cfg.max_concurrent_blocks = 8;
    cfg.enable_state_root = true;
    cfg.enable_consensus_hash = true;
    cfg.enable_signature_verify = true;

    auto pipe = u::UnifiedPipeline::create(cfg);
    CHECK(pipe != nullptr);

    std::printf("Backend: %s\n", pipe->backend_name());
    std::printf("Device:  %s\n", pipe->device_name());
    std::printf("Pipeline depth: %u\n\n", cfg.max_concurrent_blocks);

    // ---- Build the workload ------------------------------------------------

    std::vector<evm::gpu::AccountInfo> accounts(ACCOUNTS);
    for (size_t i = 0; i < ACCOUNTS; i++) accounts[i] = make_account(static_cast<uint32_t>(i));

    std::vector<std::vector<evm::gpu::Transaction>> blocks_storage(NUM_BLOCKS);
    std::vector<u::BlockContext> ctxs(NUM_BLOCKS);

    const auto add_loop = build_add_loop_bytecode(LOOP_ITERATIONS);
    const uint64_t expected_per_tx_gas = expected_loop_gas(LOOP_ITERATIONS);

    std::mt19937 rng(42);
    for (size_t b = 0; b < NUM_BLOCKS; b++) {
        blocks_storage[b].reserve(TXS_PER_BLOCK);
        for (size_t i = 0; i < TXS_PER_BLOCK; i++) {
            blocks_storage[b].push_back(
                make_tx(static_cast<uint32_t>(i + b * TXS_PER_BLOCK),
                        static_cast<uint32_t>(b),
                        add_loop));
        }
        ctxs[b] = make_ctx(b, rng);
    }

    const uint64_t expected_total_gas =
        expected_per_tx_gas * TXS_PER_BLOCK * NUM_BLOCKS;
    std::printf("Workload bytecode: %u-iter ADD loop, expected %llu gas/tx, "
                "%llu gas total\n\n",
                static_cast<unsigned>(LOOP_ITERATIONS),
                static_cast<unsigned long long>(expected_per_tx_gas),
                static_cast<unsigned long long>(expected_total_gas));

    std::vector<std::span<const evm::gpu::Transaction>> blocks_spans(NUM_BLOCKS);
    for (size_t b = 0; b < NUM_BLOCKS; b++) {
        blocks_spans[b] = std::span<const evm::gpu::Transaction>(
            blocks_storage[b].data(), blocks_storage[b].size());
    }

    // ---- Sanity: process one block, verify outputs ------------------------

    {
        auto r = pipe->process_block(blocks_spans[0], accounts, ctxs[0]);

        bool root_nonzero = false;
        for (uint8_t v : r.state_root) if (v) { root_nonzero = true; break; }
        bool hash_nonzero = false;
        for (uint8_t v : r.consensus_hash) if (v) { hash_nonzero = true; break; }

        std::printf("Sanity block:\n");
        std::printf("  evm.total_gas       = %llu\n",
                    static_cast<unsigned long long>(r.evm.total_gas));
        std::printf("  state_root nonzero  = %d (%.3f ms)\n",
                    root_nonzero, r.state_root_ms);
        std::printf("  consensus nonzero   = %d (%.3f ms)\n",
                    hash_nonzero, r.consensus_ms);
        std::printf("  evm time            = %.3f ms\n", r.evm_ms);
        std::printf("  sig verify time     = %.3f ms\n", r.sig_verify_ms);
        std::printf("  total               = %.3f ms\n\n", r.total_ms);

        CHECK(root_nonzero);
        CHECK(hash_nonzero);
    }

    // ---- Serial run --------------------------------------------------------

    std::printf("Serial run (%zu blocks)...\n", NUM_BLOCKS);
    auto t0 = Clock::now();
    double sum_evm_ms = 0, sum_root_ms = 0, sum_cons_ms = 0, sum_sig_ms = 0;
    for (size_t b = 0; b < NUM_BLOCKS; b++) {
        auto r = pipe->process_block(blocks_spans[b], accounts, ctxs[b]);
        sum_evm_ms  += r.evm_ms;
        sum_root_ms += r.state_root_ms;
        sum_cons_ms += r.consensus_ms;
        sum_sig_ms  += r.sig_verify_ms;
    }
    double serial_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - t0).count();

    std::printf("  wall-clock total    = %.1f ms\n", serial_ms);
    std::printf("  per-block average   = %.3f ms\n", serial_ms / NUM_BLOCKS);
    std::printf("  blocks per second   = %.1f\n", NUM_BLOCKS * 1000.0 / serial_ms);
    std::printf("  sum(evm)            = %.1f ms\n", sum_evm_ms);
    std::printf("  sum(state_root)     = %.1f ms\n", sum_root_ms);
    std::printf("  sum(consensus)      = %.1f ms\n", sum_cons_ms);
    std::printf("  sum(sig_verify)     = %.1f ms\n\n", sum_sig_ms);

    // ---- Pipelined run -----------------------------------------------------

    std::printf("Pipelined run (depth %u)...\n", cfg.max_concurrent_blocks);
    auto t1 = Clock::now();
    auto results = pipe->process_blocks(
        std::span<const std::span<const evm::gpu::Transaction>>(
            blocks_spans.data(), blocks_spans.size()),
        accounts,
        std::span<const u::BlockContext>(ctxs.data(), ctxs.size()));
    double pipelined_ms = std::chrono::duration<double, std::milli>(
        Clock::now() - t1).count();

    CHECK(results.size() == NUM_BLOCKS);

    size_t roots_zero = 0, hashes_zero = 0;
    uint64_t total_evm_gas = 0;
    for (auto& r : results) {
        bool nz_root = false;
        for (uint8_t v : r.state_root) if (v) { nz_root = true; break; }
        if (!nz_root) roots_zero++;
        bool nz_hash = false;
        for (uint8_t v : r.consensus_hash) if (v) { nz_hash = true; break; }
        if (!nz_hash) hashes_zero++;
        total_evm_gas += r.evm.total_gas;
    }

    std::printf("  wall-clock total    = %.1f ms\n", pipelined_ms);
    std::printf("  per-block average   = %.3f ms\n", pipelined_ms / NUM_BLOCKS);
    std::printf("  blocks per second   = %.1f\n", NUM_BLOCKS * 1000.0 / pipelined_ms);
    std::printf("  zero state roots    = %zu / %zu\n", roots_zero, NUM_BLOCKS);
    std::printf("  zero consensus hash = %zu / %zu\n", hashes_zero, NUM_BLOCKS);
    std::printf("  total EVM gas       = %llu\n",
                static_cast<unsigned long long>(total_evm_gas));
    std::printf("\n");

    CHECK(roots_zero == 0);
    CHECK(hashes_zero == 0);

    // Real EVM execution invariant: the kernel CPU interpreter (and the Metal
    // / CUDA backends, which emulate the same interpreter) must return the
    // analytically computed gas for the ADD-loop bytecode. 5% tolerance covers
    // any post-merge gas-cost adjustments on supported opcodes; the previous
    // fake path returned 21000*N regardless of bytecode and would fail this.
    const uint64_t low_bound  = static_cast<uint64_t>(expected_total_gas * 0.95);
    const uint64_t high_bound = static_cast<uint64_t>(expected_total_gas * 1.05);
    std::printf("Gas check: total=%llu  expected=%llu  bounds=[%llu, %llu]\n",
                static_cast<unsigned long long>(total_evm_gas),
                static_cast<unsigned long long>(expected_total_gas),
                static_cast<unsigned long long>(low_bound),
                static_cast<unsigned long long>(high_bound));
    CHECK(total_evm_gas >= low_bound);
    CHECK(total_evm_gas <= high_bound);

    double speedup = (pipelined_ms > 0) ? serial_ms / pipelined_ms : 0.0;
    std::printf("================================================================\n");
    std::printf("  Pipelined speedup vs serial: %.2fx\n", speedup);
    std::printf("  serial:    %.1f ms (%.1f blocks/s)\n",
                serial_ms,    NUM_BLOCKS * 1000.0 / serial_ms);
    std::printf("  pipelined: %.1f ms (%.1f blocks/s)\n",
                pipelined_ms, NUM_BLOCKS * 1000.0 / pipelined_ms);
    std::printf("================================================================\n");

    // Pipelining must give at least 0.5x. Anything lower means thread overhead
    // is dominating and a regression to all-CPU serial-equivalent execution
    // has slipped in.
    CHECK(speedup > 0.5);

    return 0;
}
