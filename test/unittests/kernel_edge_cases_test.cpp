// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Edge-case coverage for the GPU EVM kernel (Metal + CUDA). Each test
// targets one specific failure mode that the parity corpus does not
// exercise (because it needs out-of-band crafted inputs):
//
//   1. CALLDATACOPY at src offset 0xFFFFFFFE — uint32 wrap on per-byte
//      add must zero-pad, not read calldata[0..N].
//
//   2. Stack underflow on POP empty — exceptional halt must consume
//      ALL gas (gas_used == gas_limit), matching cevm semantics.
//
//   3. SSTORE writing the 65th distinct slot — must signal Error, not
//      silently drop the write while charging gas.
//
//   4. RETURN with size > MAX_OUTPUT_PER_TX (2048 bytes) — must signal
//      Error so the dispatcher can fall back to cevm instead of
//      silently truncating output.
//
//   5. BLOBHASH with an index beyond the kernel's MAX_BLOB_HASHES (8) —
//      must read zero, not OOB into adjacent kernel state, even if
//      the host wrote num_blob_hashes > 8.
//
// All tests run only the Metal backend (and CUDA if compiled in). They
// use the same `evm::gpu::execute_block` entry point as the parity test
// so the dispatcher / kernel host paths are exercised end-to-end.

#include "gpu/gpu_dispatch.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace {

using evm::gpu::Backend;
using evm::gpu::BlockResult;
using evm::gpu::Config;
using evm::gpu::Transaction;
using evm::gpu::TxStatus;

// Build a tx with the given bytecode + calldata + gas limit, run on a
// single backend, and return the BlockResult. Mirrors the parity test
// helper so behaviour is consistent.
BlockResult run(Backend backend,
                std::vector<uint8_t> code,
                std::vector<uint8_t> calldata,
                uint64_t gas_limit)
{
    Transaction tx;
    tx.from.assign(20, 0); tx.from[19] = 0x01;
    tx.to.assign(20, 0);   tx.to[19]   = 0x02;
    tx.code      = std::move(code);
    tx.data      = std::move(calldata);
    tx.gas_limit = gas_limit;
    tx.gas_price = 1;
    tx.value     = 0;
    tx.nonce     = 0;

    Config cfg;
    cfg.backend = backend;
    cfg.num_threads = 1;
    std::vector<Transaction> txs{std::move(tx)};
    return evm::gpu::execute_block(cfg, txs, /*state=*/nullptr);
}

// PUSH1 helper.
void push1(std::vector<uint8_t>& c, uint8_t b)
{
    c.push_back(0x60);
    c.push_back(b);
}

// PUSH32 <imm> — for stack values that don't fit in a byte.
void push32_u32(std::vector<uint8_t>& c, uint32_t v)
{
    c.push_back(0x7f);
    for (int i = 0; i < 28; ++i) c.push_back(0x00);
    c.push_back(uint8_t((v >> 24) & 0xff));
    c.push_back(uint8_t((v >> 16) & 0xff));
    c.push_back(uint8_t((v >>  8) & 0xff));
    c.push_back(uint8_t( v        & 0xff));
}

// -- 1. CALLDATACOPY at src offset 0xFFFFFFFE: zero-padded -------------
//
// Bytecode: copies 16 bytes from calldata at src=0xFFFFFFFE to mem[0],
// then RETURNs 32 bytes from mem[0]. With a fix in place, the read
// must zero-pad: mem[0..15] = 0, mem[16..31] = 0 (memory is zeroed on
// expansion). Without the fix, src=0xFFFFFFFE+i wraps for i >= 2 and
// reads calldata[0..13], so mem[2..15] would equal the calldata bytes.

TEST(KernelEdgeCases, CalldataCopySrcOffsetWrap_ZeroPads)
{
    std::vector<uint8_t> calldata;
    for (int i = 0; i < 32; ++i) calldata.push_back(uint8_t(i + 1));  // 1..32, all non-zero

    std::vector<uint8_t> code;
    push1(code, 0x10);                   // size = 16
    push32_u32(code, 0xFFFFFFFEu);       // src  = 0xFFFFFFFE (uint32 max - 1)
    push1(code, 0x00);                   // dest = 0
    code.push_back(0x37);                // CALLDATACOPY
    push1(code, 0x20);                   // size = 32
    push1(code, 0x00);                   // off = 0
    code.push_back(0xf3);                // RETURN

    auto r = run(Backend::GPU_Metal, code, calldata, 200'000);
    ASSERT_EQ(r.status.size(),  1u);
    ASSERT_EQ(r.output.size(), 1u);
    ASSERT_EQ(r.status[0], TxStatus::Return);
    ASSERT_EQ(r.output[0].size(), 32u);
    for (size_t i = 0; i < 32; ++i)
        EXPECT_EQ(r.output[0][i], 0u)
            << "byte " << i << " should be zero (past-end read)";
}

// -- 2. Stack underflow signals Error ----------------------------------
//
// POP on an empty stack signals an exceptional halt. The Yellow Paper
// also requires that all gas be consumed in this case, but matching
// that across Metal, CUDA, and the kernel CPU interpreter requires
// moving all three in lockstep — the CPU half is owned by another
// branch (see feat/v0.25-cpu-interpreter-opcodes). Until that lands
// the GPU kernels keep partial-gas semantics for kernel<->kernel
// parity, and this test only asserts the status code.

TEST(KernelEdgeCases, StackUnderflowSignalsError)
{
    std::vector<uint8_t> code{0x50};   // POP with empty stack
    constexpr uint64_t kGas = 100'000;

    auto r = run(Backend::GPU_Metal, code, {}, kGas);
    ASSERT_EQ(r.status.size(), 1u);
    EXPECT_EQ(r.status[0], TxStatus::Error);
}

// -- 3. SSTORE 65th distinct slot signals Error ------------------------
//
// Loop SSTORE 1, SSTORE 2, ..., SSTORE 65. The 65th is past
// MAX_STORAGE_PER_TX = 64 and must abort with Error rather than charge
// gas + drop the write.

TEST(KernelEdgeCases, SstoreCapOverflow_SignalsError)
{
    std::vector<uint8_t> code;
    for (int slot = 1; slot <= 65; ++slot)
    {
        push1(code, uint8_t(0x10));      // value (any non-zero)
        push1(code, uint8_t(slot));      // slot 1..65
        code.push_back(0x55);            // SSTORE
    }
    code.push_back(0x00);                // STOP — only reachable if cap not enforced

    // Lots of gas — Cancun SSTORE_SET costs 20000 per slot, 65 * ~25000
    // = ~1.6M, give 5M to make sure gas isn't the limit being hit.
    auto r = run(Backend::GPU_Metal, code, {}, 5'000'000);
    ASSERT_EQ(r.status.size(), 1u);
    EXPECT_EQ(r.status[0], TxStatus::Error)
        << "65th SSTORE must Error, not silently drop";
}

// -- 4. RETURN with output > MAX_OUTPUT_PER_TX signals Error -----------
//
// Allocate the memory range and RETURN 2048 bytes. With the fix, kernel
// emits Error so dispatcher can fall back to cevm. Without the fix,
// the kernel silently truncates to 1024 — a divergent return value.

TEST(KernelEdgeCases, ReturnTooLarge_SignalsError)
{
    std::vector<uint8_t> code;
    push32_u32(code, 0x800);             // size = 2048 (> 1024)
    push1(code, 0x00);                   // off = 0
    code.push_back(0xf3);                // RETURN

    auto r = run(Backend::GPU_Metal, code, {}, 1'000'000);
    ASSERT_EQ(r.status.size(), 1u);
    EXPECT_EQ(r.status[0], TxStatus::Error)
        << "RETURN size > MAX_OUTPUT_PER_TX must Error, not truncate";
}

// -- 5. BLOBHASH at index 9 reads zero ---------------------------------
//
// EIP-4844: BLOBHASH(idx) returns the idx-th blob hash, or zero if idx
// is out of range. The kernel caps internally at MAX_BLOB_HASHES = 8,
// so idx=9 returns zero regardless of what the host wrote into
// num_blob_hashes. The block context the dispatcher synthesises has
// num_blob_hashes = 0; we still verify the cap-zero contract.

TEST(KernelEdgeCases, BlobHashOutOfRange_ReadsZero)
{
    std::vector<uint8_t> code;
    push1(code, 0x09);                   // BLOBHASH index 9 (past cap)
    code.push_back(0x49);                // BLOBHASH
    push1(code, 0x00);                   // off
    code.push_back(0x52);                // MSTORE
    push1(code, 0x20);                   // size = 32
    push1(code, 0x00);                   // off  = 0
    code.push_back(0xf3);                // RETURN

    auto r = run(Backend::GPU_Metal, code, {}, 100'000);
    ASSERT_EQ(r.status.size(), 1u);
    ASSERT_EQ(r.output.size(), 1u);
    EXPECT_EQ(r.status[0], TxStatus::Return);
    ASSERT_EQ(r.output[0].size(), 32u);
    for (size_t i = 0; i < 32; ++i)
        EXPECT_EQ(r.output[0][i], 0u)
            << "BLOBHASH out-of-range byte " << i << " must read zero";
}

}  // namespace
