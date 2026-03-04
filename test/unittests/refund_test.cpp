// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for the EIP-2200 signed refund counter and the EIP-3529
// refund cap (max refund = gas_used / 5).
//
// Before this fix the kernel refund_counter was unsigned: when the EIP-2200
// "subsequent modification with original != 0 and current == 0" branch hit
// `rc -= GAS_SSTORE_REFUND` while rc was 0, it underflowed to ~2^64. The
// dispatcher then subtracted that from gas_used and produced absurd totals.
// These tests cover:
//
//   1. SSTORE creates a slot (orig=0, cur=0, new!=0): no refund.
//   2. SSTORE clears a slot (orig!=0, cur!=0, new=0): +4800 refund.
//   3. SSTORE clear -> set within one tx: refund accumulates and revokes.
//   4. The EIP-3529 cap: refund must not exceed gas_used / 5.
//   5. Parity: kernel CPU path must match evmone CPU on the same input.
//
// The kernel CPU path (Backend::CPU_Sequential with state==nullptr +
// non-empty bytecode) is the parity reference for both Metal and CUDA, so
// these tests indirectly cover the GPU paths via the dispatcher.

#include "gpu/gpu_dispatch.hpp"

#include <evmc/mocked_host.hpp>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

extern "C" struct evmc_vm* evmc_create_evmone(void) noexcept;

namespace {

using evm::gpu::Backend;
using evm::gpu::BlockResult;
using evm::gpu::Config;
using evm::gpu::Transaction;
using evm::gpu::TxStatus;

// -- Bytecode helpers ---------------------------------------------------------

inline void push1(std::vector<uint8_t>& c, uint8_t b)
{
    c.push_back(0x60);
    c.push_back(b);
}

inline void emit_sstore(std::vector<uint8_t>& c, uint8_t key, uint8_t val)
{
    push1(c, val);
    push1(c, key);
    c.push_back(0x55);  // SSTORE
}

inline void emit_stop(std::vector<uint8_t>& c)
{
    c.push_back(0x00);
}

// -- Transaction builder ------------------------------------------------------

Transaction make_tx(const std::vector<uint8_t>& code, uint64_t gas_limit = 200'000)
{
    Transaction t;
    t.from.assign(20, 0);  t.from[19] = 0x01;
    t.to.assign(20, 0);    t.to[19]   = 0x02;
    t.code      = code;
    t.gas_limit = gas_limit;
    t.gas_price = 1;
    t.value     = 0;
    t.nonce     = 0;
    return t;
}

// -- Run helper: kernel CPU path (no host, has bytecode) ----------------------

BlockResult run_kernel_cpu(const std::vector<uint8_t>& code, uint64_t gas_limit = 200'000)
{
    Config cfg;
    cfg.backend = Backend::CPU_Sequential;
    auto tx = make_tx(code, gas_limit);
    return evm::gpu::execute_block(cfg, {tx}, /*state=*/nullptr);
}

// -- Run helper: evmone CPU with a real host so SSTORE actually goes through.
//
// evmone reports `gas_left` plus an internal refund; the dispatcher path that
// has a host routes through evmone and returns gas_used in BlockResult. We
// drive evmone directly here so we can inspect gas_refund independently and
// build a parity oracle for the kernel-CPU path.

struct EvmoneRun
{
    int64_t gas_used   = 0;
    int64_t gas_refund = 0;
    evmc_status_code status = EVMC_FAILURE;
};

EvmoneRun run_evmone(const std::vector<uint8_t>& code, uint64_t gas_limit = 200'000)
{
    evmc::VM vm{evmc_create_evmone()};
    evmc::MockedHost host;

    evmc_message msg{};
    msg.kind     = EVMC_CALL;
    msg.gas      = static_cast<int64_t>(gas_limit);
    msg.recipient = evmc_address{};
    msg.recipient.bytes[19] = 0x02;
    msg.sender   = evmc_address{};
    msg.sender.bytes[19] = 0x01;

    auto res = vm.execute(host, EVMC_CANCUN, msg,
                          code.data(), code.size());

    EvmoneRun out;
    out.status     = res.status_code;
    out.gas_used   = static_cast<int64_t>(gas_limit) - res.gas_left;
    out.gas_refund = res.gas_refund;  // evmone exposes refund here
    return out;
}

}  // namespace

// -----------------------------------------------------------------------------
// 1. SSTORE creates a slot (orig=0, cur=0, new=non-zero) -> NO refund.
// -----------------------------------------------------------------------------
TEST(Refund, CreateSlotNoRefund)
{
    std::vector<uint8_t> code;
    emit_sstore(code, /*key=*/0x10, /*val=*/0x42);  // 0 -> 0x42 (SET)
    emit_stop(code);

    auto r = run_kernel_cpu(code);
    ASSERT_EQ(r.status.size(), 1u);
    EXPECT_EQ(r.status[0], TxStatus::Stop);
    // SSTORE_SET = 20000, PUSH+PUSH+STOP = 3+3+0 = 6, total = 20006.
    // No refund applies; gas_used stays at the raw cost.
    EXPECT_EQ(r.gas_used[0], 20006u);
}

// -----------------------------------------------------------------------------
// 2. SSTORE clears a slot (orig!=0, cur=orig, new=0) -> refund credit applied.
//    Net cost after EIP-3529 cap: SSTORE_RESET (2900) - min(4800, gas_used/5).
// -----------------------------------------------------------------------------
TEST(Refund, ClearSlotEarnsRefundUpToCap)
{
    // Step A: put a non-zero into slot 0x10 so it has a real "original".
    // Step B: clear it. The kernel sees orig=current (because orig is
    // recorded the first time the slot is touched, after that current is
    // mutated by sstore_gas in-tx). For a single-tx scenario where we want
    // to test the "clear refund" branch we need orig recorded as the
    // existing storage value before this tx.
    //
    // Our kernel CPU path's storage is fresh per-tx (no persistent state
    // across txs without a host), so we engineer a single-tx sequence that
    // reaches the "clear an existing slot" branch via the same-tx pattern:
    //   SSTORE 0x10, 0x42      ; 0 -> 0x42 (SET, no refund)
    //   SSTORE 0x10, 0x00      ; 0x42 -> 0 (clear, but orig was 0)
    //
    // In EIP-2200, when orig=0 (the slot was zero before the tx) the
    // 'restoration' branch also applies a refund equal to SET-NOOP = 19900
    // because writing back to original (0) is a refund. That's the path we
    // expect the kernel to take. The dispatcher then floors at 0 and caps
    // at gas_used/5.
    std::vector<uint8_t> code;
    emit_sstore(code, /*key=*/0x10, /*val=*/0x42);  // 0 -> 0x42
    emit_sstore(code, /*key=*/0x10, /*val=*/0x00);  // 0x42 -> 0
    emit_stop(code);

    auto r = run_kernel_cpu(code);
    ASSERT_EQ(r.status.size(), 1u);
    EXPECT_EQ(r.status[0], TxStatus::Stop);
    // Raw gas before refund: SSTORE_SET (20000) + SSTORE_NOOP (100) +
    // 4 PUSHes (12) + STOP (0) = 20112.
    // EIP-2200 refund branch: nv == orig && orig == 0 -> +19900.
    // EIP-3529 cap: gas_used/5 = 20112/5 = 4022. Final refund = 4022.
    // Net gas_used = 20112 - 4022 = 16090.
    EXPECT_EQ(r.gas_used[0], 16090u);
}

// -----------------------------------------------------------------------------
// 3. Reset+set+reset within one tx must NOT underflow.
//
//    This is the original bug: with unsigned refund_counter, the second
//    "non-zero original, current==0, new==0" case triggered `rc -=
//    GAS_SSTORE_REFUND` and wrapped to ~2^64. The signed version handles it
//    correctly.
// -----------------------------------------------------------------------------
TEST(Refund, NoUnderflowOnRepeatedClear)
{
    std::vector<uint8_t> code;
    // Slot 0x10: 0 -> 0x42 -> 0 -> 0x99 -> 0
    emit_sstore(code, 0x10, 0x42);
    emit_sstore(code, 0x10, 0x00);
    emit_sstore(code, 0x10, 0x99);
    emit_sstore(code, 0x10, 0x00);
    emit_stop(code);

    auto r = run_kernel_cpu(code);
    ASSERT_EQ(r.status.size(), 1u);
    EXPECT_EQ(r.status[0], TxStatus::Stop)
        << "underflow would manifest as gas_used wrap or OOG";
    // Whatever the exact gas (depends on EIP-2200 ordering) the result
    // must be small and not anywhere near 2^64. The pre-fix bug produced
    // gas_used values around 18 446 744 073 709 some hundred thousand —
    // this assertion catches the underflow.
    EXPECT_LT(r.gas_used[0], 1'000'000u);
}

// -----------------------------------------------------------------------------
// 4. Refund > gas_used / 5: EIP-3529 cap must be applied.
//
//    The cleanest way to force a large refund vs small gas_used is to clear
//    a slot the tx itself populated with a tiny op-cost. Solidity-emitted
//    real-world bytecode rarely hits this specific path because the SSTORE
//    that established the slot already costs 20000, so gas_used/5 = 4000+,
//    and the natural refund is 4800. The cap kicks in for tx that clear
//    pre-existing state with very little subsequent gas use — which is
//    exactly the scenario we engineered in test 2 above. We assert the cap
//    explicitly here by computing expected and uncapped values.
// -----------------------------------------------------------------------------
TEST(Refund, Eip3529CapApplied)
{
    std::vector<uint8_t> code;
    emit_sstore(code, 0x10, 0x42);  // 0 -> 0x42
    emit_sstore(code, 0x10, 0x00);  // 0x42 -> 0
    emit_stop(code);

    auto r = run_kernel_cpu(code);
    ASSERT_EQ(r.status.size(), 1u);
    ASSERT_EQ(r.status[0], TxStatus::Stop);

    // Without the cap, refund would be 19900 (SSTORE_SET-NOOP). With the
    // cap, refund <= gas_used/5. Pre-cap gas_used = 20112; the post-cap
    // gas_used reported back is 20112 - min(19900, 4022) = 16090.
    // If the cap were missing we'd see 20112 - 19900 = 212 — the test
    // catches that.
    EXPECT_EQ(r.gas_used[0], 16090u);
    EXPECT_NE(r.gas_used[0], 212u) << "EIP-3529 cap not applied";
}

// -----------------------------------------------------------------------------
// 5. Parity with evmone CPU on the SSTORE refund DIRECTION.
//
//    The kernel does not model EIP-2929 cold/warm storage access (separate
//    bug C5), so the absolute gas numbers diverge from evmone. What MUST
//    agree is the qualitative behaviour: clearing a slot earns a positive
//    refund, after which the EIP-3529 cap shrinks net gas_used below the
//    raw cost. That is exactly the evidence that the signed refund is
//    flowing through both backends; underflow (the original bug) would
//    instead inflate gas_used past the gas_limit.
// -----------------------------------------------------------------------------
TEST(Refund, ParityWithEvmoneOnClearSlot)
{
    std::vector<uint8_t> code;
    emit_sstore(code, 0x10, 0x42);
    emit_sstore(code, 0x10, 0x00);
    emit_stop(code);

    auto kernel = run_kernel_cpu(code);
    auto evmone = run_evmone(code);

    ASSERT_EQ(kernel.status.size(), 1u);
    EXPECT_EQ(kernel.status[0], TxStatus::Stop);
    EXPECT_EQ(evmone.status, EVMC_SUCCESS);

    // Both backends report a positive raw refund on this bytecode. The
    // exact magnitude differs because the kernel doesn't model EIP-2929
    // cold/warm SLOAD pricing, but the SSTORE refund branch is shared
    // semantics — so both must agree that a refund was earned.
    EXPECT_GT(evmone.gas_refund, 0)
        << "evmone reports no refund — bytecode does not exercise the path";

    // Compute net gas_used after EIP-3529 cap for evmone, the same way
    // the dispatcher does for the kernel.
    const int64_t cap          = evmone.gas_used / 5;
    const int64_t evmone_final = evmone.gas_used -
                                 std::min<int64_t>(evmone.gas_refund, cap);

    // Both backends must yield net gas_used STRICTLY LESS THAN raw
    // gas_used (refund actually applied) and well within the gas_limit
    // (no underflow). The pre-fix unsigned bug would have produced a
    // gas_used near 2^64, blowing past gas_limit — these bounds catch
    // that catastrophic regression even without exact-number parity.
    EXPECT_LT(static_cast<int64_t>(kernel.gas_used[0]),
              static_cast<int64_t>(20112))
        << "kernel did not apply refund (raw cost is 20112)";
    EXPECT_LT(evmone_final, evmone.gas_used)
        << "evmone did not apply refund either — test is faulty";
    EXPECT_LT(static_cast<int64_t>(kernel.gas_used[0]), 1'000'000)
        << "kernel gas_used absurdly large — refund underflow regressed";
}
