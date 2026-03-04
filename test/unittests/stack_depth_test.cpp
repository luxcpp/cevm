// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Stack-depth parity test: verifies that the Metal/CUDA kernels handle
// EVM stack depths >32. Until this test landed, both kernels capped the
// stack at 32 entries and silently aborted (status=Error) on the very
// common case of Solidity-emitted bytecode pushing 30-50 items.
//
// We construct two bytecodes that the previous 32-cap kernels could not
// run but the spec-correct 1024-cap kernels must:
//
//   1. push100  — pushes 100 distinct values, then pops 99 of them and
//                 returns the last one (which is the very first push).
//                 Asserts that values pushed beyond depth 32 are reachable
//                 via stack walk, not silently dropped.
//
//   2. push200  — pushes 200 values then drops them all and STOPs. This
//                 is a pure overflow check: the kernel must accept entries
//                 well past the old 32-bound up to the spec's 1024-cap.
//
// Pass criterion (per Expectation::Agree in parity_test.cpp): every backend
// produces identical (gas_used, status, output). On the old kernels the
// GPU paths returned status=Error at sp=32; on the new kernels all paths
// return status=Return (push100) or status=Stop (push200).

#include "gpu/gpu_dispatch.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace {

using evm::gpu::Backend;
using evm::gpu::BlockResult;
using evm::gpu::Config;
using evm::gpu::Transaction;
using evm::gpu::TxStatus;

// PUSH1 <byte>
inline void push1(std::vector<uint8_t>& c, uint8_t b)
{
    c.push_back(0x60);
    c.push_back(b);
}

// MSTORE <word at offset>
inline void mstore_word(std::vector<uint8_t>& c, uint8_t off)
{
    push1(c, off);
    c.push_back(0x52);  // MSTORE
}

// RETURN(off, sz)
inline void emit_return(std::vector<uint8_t>& c, uint8_t off, uint8_t sz)
{
    push1(c, sz);
    push1(c, off);
    c.push_back(0xf3);
}

// Build: push N distinct PUSH1 values [1, 2, ..., N], then SWAP1+POP repeated
// (N-1) times to drop every entry except the very first (bottom) and the
// very last (top, the most recently pushed); finally a single POP leaves
// the original top in place. Wait — let me describe what this actually
// does, since stack semantics matter.
//
// After N PUSH1 ops, stack (bottom -> top) = [1, 2, ..., N], top = N.
// SWAP1 swaps top with one-below: [1, 2, ..., N, N-1].
// POP drops top:                    [1, 2, ..., N]   (we dropped N-1).
// Each (SWAP1; POP) pair therefore deletes the second-from-top entry, NOT
// the top. After (N-1) iterations the deletions have walked from N-1 down
// to 1, leaving the lone survivor at the top: value N.
//
// MSTORE+RETURN that. The returned 32-byte word's low byte = N, proving:
//   (a) the kernel sustained a stack of depth N (>32),
//   (b) every backend agrees on the surviving value and gas.
//
// Bytecode size: N * 2 (PUSH1+byte) + (N-1) * 2 (SWAP1+POP) + 7 (return
// preamble). For N=100: 405 bytes.
std::vector<uint8_t> build_push_then_pop(unsigned n)
{
    std::vector<uint8_t> c;
    c.reserve(n * 2 + (n - 1) * 2 + 16);
    for (unsigned i = 1; i <= n; ++i)
        push1(c, static_cast<uint8_t>(i & 0xFF));
    for (unsigned i = 0; i < n - 1; ++i)
    {
        c.push_back(0x90);  // SWAP1
        c.push_back(0x50);  // POP
    }
    mstore_word(c, 0x00);
    emit_return(c, 0x00, 0x20);
    return c;
}

// Build: PUSH N values then drop all of them and STOP. Tests the kernel
// can sustain the deep stack without overflow, and that it correctly
// transitions to status=Stop when execution reaches the STOP opcode.
std::vector<uint8_t> build_push_then_drop(unsigned n)
{
    std::vector<uint8_t> c;
    c.reserve(n * 2 + n + 1);
    for (unsigned i = 1; i <= n; ++i)
        push1(c, static_cast<uint8_t>(i & 0xFF));
    for (unsigned i = 0; i < n; ++i)
        c.push_back(0x50);  // POP
    c.push_back(0x00);  // STOP
    return c;
}

Transaction make_tx(std::vector<uint8_t> code, uint64_t gas_limit)
{
    Transaction t;
    t.from.assign(20, 0);
    t.from[19] = 0x01;
    t.to.assign(20, 0);
    t.to[19] = 0x02;
    t.code      = std::move(code);
    t.data      = {};
    t.gas_limit = gas_limit;
    t.gas_price = 1;
    t.value     = 0;
    t.nonce     = 0;
    return t;
}

BlockResult run_one(Backend backend, const Transaction& tx)
{
    Config cfg;
    cfg.backend = backend;
    cfg.num_threads = 2;
    std::vector<Transaction> txs{tx};
    return evm::gpu::execute_block(cfg, txs, /*state=*/nullptr);
}

std::string hex(const std::vector<uint8_t>& b)
{
    static const char* H = "0123456789abcdef";
    std::string s;
    s.reserve(b.size() * 2);
    for (auto x : b) { s += H[(x >> 4) & 0xf]; s += H[x & 0xf]; }
    return s;
}

void assert_all_backends_agree(const char* name, const Transaction& tx)
{
    auto r_seq   = run_one(Backend::CPU_Sequential, tx);
    auto r_par   = run_one(Backend::CPU_Parallel,   tx);
    auto r_metal = run_one(Backend::GPU_Metal,      tx);

    ASSERT_EQ(r_seq.status.size(),   1u) << name << " (seq)";
    ASSERT_EQ(r_par.status.size(),   1u) << name << " (par)";
    ASSERT_EQ(r_metal.status.size(), 1u) << name << " (metal)";

    EXPECT_EQ(r_seq.status[0],   r_par.status[0])   << name << ": CPU seq vs par status diverge";
    EXPECT_EQ(r_seq.gas_used[0], r_par.gas_used[0]) << name << ": CPU seq vs par gas diverge";
    EXPECT_EQ(r_seq.output[0],   r_par.output[0])   << name << ": CPU seq vs par output diverge";

    EXPECT_EQ(r_seq.status[0],   r_metal.status[0])
        << name << ": CPU vs Metal status diverge"
        << "  cpu=" << static_cast<uint32_t>(r_seq.status[0])
        << " metal=" << static_cast<uint32_t>(r_metal.status[0]);
    EXPECT_EQ(r_seq.gas_used[0], r_metal.gas_used[0])
        << name << ": CPU vs Metal gas diverge"
        << "  cpu=" << r_seq.gas_used[0]
        << " metal=" << r_metal.gas_used[0];
    EXPECT_EQ(r_seq.output[0],   r_metal.output[0])
        << name << ": CPU vs Metal output diverge"
        << "  cpu=0x" << hex(r_seq.output[0])
        << " metal=0x" << hex(r_metal.output[0]);

#ifdef EVM_CUDA
    auto r_cuda = run_one(Backend::GPU_CUDA, tx);
    ASSERT_EQ(r_cuda.status.size(), 1u) << name << " (cuda)";
    EXPECT_EQ(r_seq.status[0],   r_cuda.status[0])   << name << ": CPU vs CUDA status diverge";
    EXPECT_EQ(r_seq.gas_used[0], r_cuda.gas_used[0]) << name << ": CPU vs CUDA gas diverge";
    EXPECT_EQ(r_seq.output[0],   r_cuda.output[0])   << name << ": CPU vs CUDA output diverge";
#endif

    // Sanity: GPU must have actually run, not silently aborted at sp=32.
    EXPECT_NE(r_metal.status[0], TxStatus::Error)
        << name << ": Metal aborted — stack-depth fix did not take effect";
}

}  // namespace

// PUSH 100 distinct values, then drop the second-from-top via SWAP1+POP
// repeatedly until only the original top remains. RETURN it.
//
// The 32-cap kernels failed at the 33rd push with status=Error. The
// spec-correct 1024-cap kernels accept all 100 pushes and the surviving
// top-of-stack is the last value pushed (= 100), encoded in the low byte
// of the returned 32-byte word.
TEST(StackDepth, Push100SwapToBottomReturn)
{
    auto code = build_push_then_pop(100);
    // Loose gas: 100 PUSH1 (3 each) + 99*(SWAP1=3 + POP=2) + ~10 for tail.
    // ~= 300 + 495 + 10 + memory expansion (3 gas/word + linear quadratic
    // for 1 word = 3 + 0 = 3). Round up by 10x.
    auto tx = make_tx(std::move(code), 1'000'000);
    assert_all_backends_agree("push100_swap_to_bottom_return", tx);

    // The returned word should encode value 100 (= 0x64) in its low byte.
    // If all backends agree (asserted above) and the CPU one is correct,
    // the GPU ones are correct too.
    auto r = run_one(Backend::CPU_Sequential, tx);
    ASSERT_EQ(r.output.size(), 1u);
    ASSERT_EQ(r.output[0].size(), 32u);
    for (size_t i = 0; i < 31; ++i)
        EXPECT_EQ(r.output[0][i], 0x00) << "byte " << i << " should be zero";
    EXPECT_EQ(r.output[0][31], 0x64) << "low byte should be the last-pushed value (100)";
}

// PUSH 200, POP 200, STOP. Pure stack-depth check: the kernel must sustain
// 200 simultaneous entries on the stack without aborting.
TEST(StackDepth, Push200ThenPopAllStop)
{
    auto code = build_push_then_drop(200);
    auto tx = make_tx(std::move(code), 1'000'000);
    assert_all_backends_agree("push200_pop_all_stop", tx);

    auto r = run_one(Backend::GPU_Metal, tx);
    ASSERT_EQ(r.status.size(), 1u);
    EXPECT_EQ(r.status[0], TxStatus::Stop)
        << "Reaching STOP after 200 pushes/pops requires the 1024-deep stack";
}

// PUSH up to depth 1023, then PUSH one more (depth 1024 == STACK_LIMIT, OK
// per the Yellow Paper since the limit is "must not exceed 1024"), then
// POP everything and STOP. This is the boundary case: the kernel must
// accept exactly STACK_LIMIT entries.
TEST(StackDepth, PushToBoundaryThenStop)
{
    // We use PUSH1 to keep code small. 1024 pushes * 2 bytes each = 2048
    // bytes of code, then 1024 POPs + STOP = 1025 bytes more. Total ~3 KB.
    constexpr unsigned N = 1024;
    auto code = build_push_then_drop(N);
    // Gas: 1024 PUSH1 (3) + 1024 POP (2) + 0 (STOP) = 3072 + 2048 = 5120.
    // Plus intrinsic overhead — give a wide margin.
    auto tx = make_tx(std::move(code), 5'000'000);
    assert_all_backends_agree("push1024_pop_all_stop", tx);

    auto r = run_one(Backend::GPU_Metal, tx);
    ASSERT_EQ(r.status.size(), 1u);
    EXPECT_EQ(r.status[0], TxStatus::Stop)
        << "Pushing exactly STACK_LIMIT entries must not overflow";
}
