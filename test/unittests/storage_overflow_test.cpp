// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Storage-overflow detection: a transaction that issues more SSTORE / TSTORE
// writes than the per-tx kernel cap (MAX_STORAGE_PER_TX = 64) used to
// silently drop the 65th and beyond. Silent drops corrupt state — the EVM's
// SLOAD/TLOAD reads back an old value while the rest of the transaction
// continues as if the write succeeded. The kernels now signal an exceptional
// halt (status=Error, all gas consumed) on the offending opcode so the host
// dispatcher can route the tx to evmone CPU (which has no cap), or — when
// no host is available (benchmark mode) — surface an honest "GPU can't
// process this tx" instead of a corrupt result.
//
// Test scenarios:
//
//   1. SSTORE within cap (64 distinct slots) — every backend succeeds.
//   2. SSTORE one over cap (65 distinct slots), no host — every backend
//      returns status=Error with gas_used == gas_limit.
//   3. SSTORE one over cap, MockedHost provided — the dispatcher routes
//      through evmone (no cap), execution succeeds.
//   4. TSTORE within cap (64 slots) — every GPU backend succeeds; the
//      kernel CPU path doesn't implement TSTORE so it returns Error
//      (KernelCpuMissing pattern, mirroring parity_test.cpp).
//   5. TSTORE one over cap, no host — every GPU backend returns
//      status=Error with gas_used == gas_limit.
//   6. TSTORE one over cap, MockedHost provided — evmone executes it.

#include "gpu/gpu_dispatch.hpp"

#include <evmc/mocked_host.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

extern "C" struct evmc_vm* evmc_create_evmone(void) noexcept;

namespace {

using evm::gpu::Backend;
using evm::gpu::BlockResult;
using evm::gpu::Config;
using evm::gpu::Transaction;
using evm::gpu::TxStatus;

constexpr uint32_t KERNEL_STORAGE_CAP = 64;

// -- Bytecode builders --------------------------------------------------------

// Append: PUSH1 <imm>
inline void push1(std::vector<uint8_t>& c, uint8_t b)
{
    c.push_back(0x60);
    c.push_back(b);
}

// Build a program that issues `n` distinct SSTORE writes (slot = 1..n,
// value = slot ^ 0xAA so the value is non-zero and slot-dependent), then
// STOPs. PUSH1 value; PUSH1 slot; SSTORE  →  5 bytes per write.
//
// `n` is bounded by 0xFF because we use PUSH1 for the slot. The kernel cap
// is 64 — well below the limit.
std::vector<uint8_t> build_sstore_program(uint32_t n)
{
    std::vector<uint8_t> code;
    code.reserve(static_cast<size_t>(n) * 5 + 1);
    for (uint32_t i = 1; i <= n; ++i)
    {
        const auto slot = static_cast<uint8_t>(i);
        const auto val  = static_cast<uint8_t>(i ^ 0xAA);
        push1(code, val);    // value (second pop)
        push1(code, slot);   // slot  (first pop / top)
        code.push_back(0x55);  // SSTORE
    }
    code.push_back(0x00);  // STOP
    return code;
}

// Same shape as build_sstore_program but with TSTORE (0x5d) instead of
// SSTORE (0x55). EIP-1153 transient storage shares the same per-tx cap in
// the GPU kernels.
std::vector<uint8_t> build_tstore_program(uint32_t n)
{
    std::vector<uint8_t> code;
    code.reserve(static_cast<size_t>(n) * 5 + 1);
    for (uint32_t i = 1; i <= n; ++i)
    {
        const auto slot = static_cast<uint8_t>(i);
        const auto val  = static_cast<uint8_t>(i ^ 0xAA);
        push1(code, val);
        push1(code, slot);
        code.push_back(0x5d);  // TSTORE
    }
    code.push_back(0x00);  // STOP
    return code;
}

// -- Transaction + host helpers ----------------------------------------------

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

// Mocked host that lets evmone run any bytecode without external services.
// The block context fields are populated so opcodes that read them (none of
// the ones used here, but the dispatcher may touch tx_context regardless)
// return deterministic values.
evmc::MockedHost make_host()
{
    evmc::MockedHost host;
    host.tx_context.block_gas_limit = 30'000'000;
    host.tx_context.block_number    = 1;
    host.tx_context.block_timestamp = 1700000000;
    host.tx_context.tx_gas_price    = {};
    host.tx_context.chain_id        = {};
    host.call_result.status_code = EVMC_SUCCESS;
    host.call_result.gas_left    = 0;
    host.call_result.gas_refund  = 0;
    host.call_result.output_data = nullptr;
    host.call_result.output_size = 0;
    host.call_result.release     = nullptr;
    return host;
}

BlockResult run(Backend backend, const Transaction& tx, evmc::Host* host = nullptr)
{
    Config cfg;
    cfg.backend = backend;
    cfg.num_threads = 2;
    std::vector<Transaction> txs{tx};
    return evm::gpu::execute_block(cfg, txs, host);
}

// -- Per-backend assertions ---------------------------------------------------

// All GPU/kernel-CPU backends must report status=Error AND gas_used ==
// gas_limit on the over-cap tx. The kernel CPU and Metal paths are always
// checked. CUDA is only built into this binary on EVM_CUDA platforms.
void assert_overflow_signaled(const char* name,
                              const Transaction& tx)
{
    SCOPED_TRACE(name);

    auto r_seq   = run(Backend::CPU_Sequential, tx);
    auto r_par   = run(Backend::CPU_Parallel,   tx);
    auto r_metal = run(Backend::GPU_Metal,      tx);

    ASSERT_EQ(r_seq.status.size(),   1u);
    ASSERT_EQ(r_par.status.size(),   1u);
    ASSERT_EQ(r_metal.status.size(), 1u);

    EXPECT_EQ(r_seq.status[0],   TxStatus::Error)
        << "kernel CPU silently dropped over-cap write";
    EXPECT_EQ(r_par.status[0],   TxStatus::Error);
    EXPECT_EQ(r_metal.status[0], TxStatus::Error);

    // Yellow Paper INVALID-style: exceptional halt consumes ALL gas.
    EXPECT_EQ(r_seq.gas_used[0],   tx.gas_limit);
    EXPECT_EQ(r_par.gas_used[0],   tx.gas_limit);
    EXPECT_EQ(r_metal.gas_used[0], tx.gas_limit);

#ifdef EVM_CUDA
    auto r_cuda = run(Backend::GPU_CUDA, tx);
    ASSERT_EQ(r_cuda.status.size(), 1u);
    EXPECT_EQ(r_cuda.status[0],   TxStatus::Error);
    EXPECT_EQ(r_cuda.gas_used[0], tx.gas_limit);
#endif
}

// Evmone (host-backed) executes the same bytecode and succeeds because it
// has no per-tx storage cap. The host-routed dispatcher path reads bytecode
// from `tx.data` (parallel_engine.cpp:to_evm_transaction sets etx.code from
// the data field), so we mirror the bytecode into both fields here. gas_used
// is the only signal evmone exposes through the dispatcher's host path
// (status/output are not populated for evmone-routed txs).
void assert_evmone_succeeds(const char* name,
                            std::vector<uint8_t> bytecode,
                            uint64_t gas_limit)
{
    SCOPED_TRACE(name);

    Transaction tx;
    tx.from.assign(20, 0);  tx.from[19] = 0x01;
    tx.to.assign(20, 0);    tx.to[19]   = 0x02;
    tx.code      = bytecode;
    tx.data      = std::move(bytecode);  // host path consumes data as code
    tx.gas_limit = gas_limit;
    tx.gas_price = 1;
    tx.value     = 0;
    tx.nonce     = 0;

    auto host = make_host();
    auto r = run(Backend::CPU_Sequential, tx, &host);

    ASSERT_EQ(r.gas_used.size(), 1u);
    EXPECT_GT(r.gas_used[0], 0u);
    EXPECT_LT(r.gas_used[0], tx.gas_limit)
        << "evmone consumed all gas — bytecode failed even on the unbounded "
           "CPU path, which means the test is not exercising what we think";
    EXPECT_TRUE(r.error_message.empty()) << r.error_message;
}

}  // namespace

// -- 1. SSTORE within cap (64 distinct slots) --------------------------------
//
// 64 SSTOREs produce 64 entries in the kernel's per-tx storage table.
// Every backend must succeed: status=Stop (the program ends with STOP),
// gas_used reflects 64 × SSTORE_SET (20000 each) + push gas, and the
// per-tx storage buffer is exactly full but no overflow has occurred.
TEST(StorageOverflow, SStoreAtCap_AllBackendsSucceed)
{
    auto tx = make_tx(build_sstore_program(KERNEL_STORAGE_CAP), 5'000'000);

    auto r_seq   = run(Backend::CPU_Sequential, tx);
    auto r_par   = run(Backend::CPU_Parallel,   tx);
    auto r_metal = run(Backend::GPU_Metal,      tx);

    ASSERT_EQ(r_seq.status.size(),   1u);
    ASSERT_EQ(r_par.status.size(),   1u);
    ASSERT_EQ(r_metal.status.size(), 1u);

    EXPECT_EQ(r_seq.status[0],   TxStatus::Stop) << "CPU kernel hit cap early";
    EXPECT_EQ(r_par.status[0],   TxStatus::Stop);
    EXPECT_EQ(r_metal.status[0], TxStatus::Stop);

    // 64 SSTOREs × 20000 gas (cold writes from 0 to non-zero) is the
    // dominant cost. Anything below 1.2M means SSTORE bookkeeping is
    // wrong. Anything at gas_limit means a silent OOG.
    EXPECT_GT(r_seq.gas_used[0],   1'200'000u);
    EXPECT_LT(r_seq.gas_used[0],   tx.gas_limit);

#ifdef EVM_CUDA
    auto r_cuda = run(Backend::GPU_CUDA, tx);
    ASSERT_EQ(r_cuda.status.size(), 1u);
    EXPECT_EQ(r_cuda.status[0], TxStatus::Stop);
#endif
}

// -- 2. SSTORE one over cap, no host -----------------------------------------
//
// The 65th SSTORE attempts to append a new slot when the per-tx buffer is
// full. Pre-fix: silently dropped, status reported as Stop with the write
// missing. Post-fix: kernel emits status=Error and consumes all remaining
// gas (INVALID-style halt).
TEST(StorageOverflow, SStoreOverCap_NoHost_SignalsError)
{
    auto tx = make_tx(build_sstore_program(KERNEL_STORAGE_CAP + 1), 5'000'000);
    assert_overflow_signaled("sstore_65_no_host", tx);
}

// -- 3. SSTORE one over cap, with host ---------------------------------------
//
// With a MockedHost, the dispatcher routes the tx through evmone — which
// has no per-tx storage cap and therefore executes all 65 writes.
TEST(StorageOverflow, SStoreOverCap_WithHost_EvmoneSucceeds)
{
    assert_evmone_succeeds("sstore_65_with_host",
        build_sstore_program(KERNEL_STORAGE_CAP + 1), 5'000'000);
}

// -- 4. TSTORE within cap ----------------------------------------------------
//
// 64 distinct TSTOREs must succeed on every backend that implements
// EIP-1153. The Metal and CUDA kernels do; the kernel CPU interpreter
// (lib/evm/gpu/kernel/evm_interpreter.hpp) does not. CPU paths therefore
// CPU and GPU both implement TSTORE as of v0.26. Filling exactly to capacity
// (64 slots) reaches STOP on every backend.
TEST(StorageOverflow, TStoreAtCap_AllBackendsSucceed)
{
    auto tx = make_tx(build_tstore_program(KERNEL_STORAGE_CAP), 5'000'000);

    auto r_seq   = run(Backend::CPU_Sequential, tx);
    auto r_metal = run(Backend::GPU_Metal,      tx);

    ASSERT_EQ(r_seq.status.size(),   1u);
    ASSERT_EQ(r_metal.status.size(), 1u);

    EXPECT_EQ(r_seq.status[0],   TxStatus::Stop);
    EXPECT_EQ(r_metal.status[0], TxStatus::Stop);

#ifdef EVM_CUDA
    auto r_cuda = run(Backend::GPU_CUDA, tx);
    ASSERT_EQ(r_cuda.status.size(), 1u);
    EXPECT_EQ(r_cuda.status[0], TxStatus::Stop);
#endif
}

// -- 5. TSTORE one over cap, no host -----------------------------------------
//
// 65 TSTOREs: GPU kernels signal Error + all gas. CPU kernel still doesn't
// implement TSTORE (Error on the first 0x5d), but its gas accounting is
// (gas_start - gas) i.e. partial — only the GPU paths exercise the new
// cap-overflow code. This test asserts that whichever path executed the
// 65th TSTORE returned Error.
TEST(StorageOverflow, TStoreOverCap_NoHost_SignalsError)
{
    auto tx = make_tx(build_tstore_program(KERNEL_STORAGE_CAP + 1), 5'000'000);

    auto r_metal = run(Backend::GPU_Metal, tx);

    ASSERT_EQ(r_metal.status.size(), 1u);
    EXPECT_EQ(r_metal.status[0],   TxStatus::Error);
    EXPECT_EQ(r_metal.gas_used[0], tx.gas_limit)
        << "Metal failed to consume all gas on TSTORE overflow";

#ifdef EVM_CUDA
    auto r_cuda = run(Backend::GPU_CUDA, tx);
    ASSERT_EQ(r_cuda.status.size(), 1u);
    EXPECT_EQ(r_cuda.status[0],   TxStatus::Error);
    EXPECT_EQ(r_cuda.gas_used[0], tx.gas_limit);
#endif
}

// -- 6. TSTORE one over cap, with host ---------------------------------------
//
// MockedHost + evmone: TSTORE is fully supported, no cap, executes all 65.
TEST(StorageOverflow, TStoreOverCap_WithHost_EvmoneSucceeds)
{
    assert_evmone_succeeds("tstore_65_with_host",
        build_tstore_program(KERNEL_STORAGE_CAP + 1), 5'000'000);
}
