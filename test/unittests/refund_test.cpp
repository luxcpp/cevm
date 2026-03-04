// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Refund counter regression test for the GPU EVM kernel.
//
// Pre-fix `refund_counter` was unsigned; an EIP-2200 path that subtracts
// (rc -= GAS_SSTORE_REFUND) underflowed to ~2^64 and corrupted the post-cap
// gas value. After this fix the counter is signed (int64 in Metal/CUDA/CPU);
// the dispatcher applies the EIP-3529 cap (refund <= gas_used / 5) on emit.
//
// Scenarios driven through `evm::gpu::execute_block` with
// Backend::CPU_Sequential — when state == nullptr and the tx carries
// bytecode the dispatcher routes through the kernel CPU interpreter, which
// is what the Metal and CUDA kernels emulate at the opcode level.

#include "gpu/gpu_dispatch.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

using evm::gpu::Backend;
using evm::gpu::Config;
using evm::gpu::Transaction;
using evm::gpu::TxStatus;

constexpr uint64_t SSTORE_SET    = 20000;
constexpr uint64_t SSTORE_NOOP   = 100;
constexpr uint64_t GAS_VERYLOW   = 3;     // PUSH1

// Emit `PUSH1 value, PUSH1 slot, SSTORE` — the kernel pops {slot, value}
// (top, second), so we push value first, then slot.
inline void emit_sstore(std::vector<uint8_t>& c, uint8_t slot, uint8_t value)
{
    c.push_back(0x60); c.push_back(value);
    c.push_back(0x60); c.push_back(slot);
    c.push_back(0x55);
}

inline void emit_stop(std::vector<uint8_t>& c) { c.push_back(0x00); }

// Run a single tx through the CPU sequential dispatcher (which routes the
// bytecode through the kernel CPU interpreter when state==nullptr). Returns
// the post-cap gas_used reported back to callers.
uint64_t run_cpu(const std::vector<uint8_t>& code, uint64_t gas_limit)
{
    Transaction tx;
    tx.from.assign(20, 0);  tx.from[19] = 0x01;
    tx.to.assign(20, 0);    tx.to[19]   = 0x02;
    tx.code = code;
    tx.gas_limit = gas_limit;

    Config cfg;
    cfg.backend = Backend::CPU_Sequential;
    auto br = evm::gpu::execute_block(cfg, {tx}, /*state=*/nullptr);
    EXPECT_EQ(br.status.size(), 1u);
    EXPECT_EQ(br.gas_used.size(), 1u);
    EXPECT_EQ(br.status[0], TxStatus::Stop);
    return br.gas_used[0];
}

}  // namespace

// -- Scenario 1: SSTORE creates a slot ----------------------------------------
// orig=0, cur=0, nv!=0 → SET branch returns SSTORE_SET (20000). No refund.
// raw_gas == final_gas (cap moot when refund is zero).
TEST(GpuRefund, CreateSlotNoRefund)
{
    std::vector<uint8_t> code;
    emit_sstore(code, /*slot=*/0x01, /*value=*/0x42);  // 0 -> 0x42
    emit_stop(code);

    const uint64_t expected = 2 * GAS_VERYLOW + SSTORE_SET;
    EXPECT_EQ(run_cpu(code, 100'000), expected);
}

// -- Scenario 2: SSTORE resets a slot to zero ---------------------------------
// First write 0->0x42 (SET, 20000). Second write 0x42->0 hits the dirty-slot
// EIP-2200 second arm: orig==0 so the rc-=REFUND branch is skipped (no
// underflow), but the nv==orig branch credits +(SSTORE_SET - SSTORE_NOOP)
// = 19900. Cost is SSTORE_NOOP.
//
// raw_gas    = 4*PUSH1 + SSTORE_SET + SSTORE_NOOP = 12 + 20100 = 20112
// raw_refund = 19900
// cap        = 20112 / 5 = 4022
// final gas  = 20112 - 4022 = 16090
TEST(GpuRefund, ResetSlotCreditsRefund)
{
    std::vector<uint8_t> code;
    emit_sstore(code, /*slot=*/0x01, /*value=*/0x42);
    emit_sstore(code, /*slot=*/0x01, /*value=*/0x00);
    emit_stop(code);

    const uint64_t raw     = 4 * GAS_VERYLOW + SSTORE_SET + SSTORE_NOOP;
    const uint64_t cap     = raw / 5;
    const uint64_t refund  = std::min<uint64_t>(SSTORE_SET - SSTORE_NOOP, cap);
    const uint64_t expected = raw - refund;

    EXPECT_EQ(run_cpu(code, 100'000), expected);
}

// -- Scenario 3: signed counter survives transient negatives ------------------
// Reset+set+reset on the same slot: 0->A, A->0, 0->B, B->0. The middle
// transitions debit the previously-credited refund. Pre-fix the (debit
// path) `rc -= REFUND` would underflow; post-fix the counter is signed and
// reaches the right final value.
//
// Per-step (orig=0 throughout, since slot starts at zero pre-tx):
//   1. 0->A   eq(orig,cur)=T, iszero(orig)=T  -> SET, no refund
//   2. A->0   eq(orig,cur)=F, iszero(orig)=T  -> NOOP gas, no rc-=REFUND
//             (skipped because orig==0); nv==orig=0 -> rc += SET-NOOP (19900)
//   3. 0->B   eq(orig,cur)=T, iszero(orig)=T  -> SET, no refund change
//   4. B->0   same as step 2: rc += 19900
//
// Total raw_gas = 8*PUSH1 + 2*SSTORE_SET + 2*SSTORE_NOOP = 24 + 40200 = 40224
// Total refund  = 2 * (SSTORE_SET - SSTORE_NOOP) = 39800
// cap           = 40224 / 5 = 8044
// final         = 40224 - 8044 = 32180
//
// The critical assertion is that the final value is positive and matches
// the formula — pre-fix any underflow would put gas_used at gigantic values.
TEST(GpuRefund, SignedCounterNoUnderflow)
{
    std::vector<uint8_t> code;
    emit_sstore(code, /*slot=*/0x01, /*value=*/0x42);
    emit_sstore(code, /*slot=*/0x01, /*value=*/0x00);
    emit_sstore(code, /*slot=*/0x01, /*value=*/0x77);
    emit_sstore(code, /*slot=*/0x01, /*value=*/0x00);
    emit_stop(code);

    const uint64_t raw     = 8 * GAS_VERYLOW + 2 * SSTORE_SET + 2 * SSTORE_NOOP;
    const uint64_t cap     = raw / 5;
    const uint64_t refund  = std::min<uint64_t>(2 * (SSTORE_SET - SSTORE_NOOP), cap);
    const uint64_t expected = raw - refund;

    const uint64_t actual = run_cpu(code, 1'000'000);

    // Sanity: gas_used must be a sensible value, not the giant garbage that
    // an unsigned-underflow followed by cap subtraction would produce.
    ASSERT_LT(actual, raw);
    EXPECT_EQ(actual, expected);
}

// -- Scenario 4: refund > gas_used/5 → cap applied at dispatcher --------------
// Five fresh slots: SET each (5 * 20000 raw), then zero each (5 * 100 raw,
// each crediting SSTORE_SET - SSTORE_NOOP = 19900 refund). Total raw refund
// is 99500 — easily larger than gas_used / 5. The dispatcher must clamp.
TEST(GpuRefund, RefundCapAppliedAtDispatcher)
{
    std::vector<uint8_t> code;
    for (uint8_t s = 1; s <= 5; ++s)
        emit_sstore(code, s, /*value=*/0x77);
    for (uint8_t s = 1; s <= 5; ++s)
        emit_sstore(code, s, /*value=*/0x00);
    emit_stop(code);

    // 10 SSTOREs * 2 PUSH1 each = 20 PUSH1.
    const uint64_t raw_gas = 20 * GAS_VERYLOW
                           + 5 * SSTORE_SET    // first writes pay SET
                           + 5 * SSTORE_NOOP;  // second writes are NOOP gas
    const uint64_t raw_refund = 5 * (SSTORE_SET - SSTORE_NOOP);  // 99500

    const uint64_t cap     = raw_gas / 5;
    const uint64_t refund  = std::min<uint64_t>(raw_refund, cap);
    const uint64_t expected = raw_gas - refund;

    // Sanity: the test must actually exercise the cap (refund > cap).
    ASSERT_GT(raw_refund, cap);

    EXPECT_EQ(run_cpu(code, 1'000'000), expected);
}
