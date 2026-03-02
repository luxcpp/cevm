// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for the GPU execution dispatcher (lib/evm/gpu/gpu_dispatch.*).
//
// The dispatcher routes EVM block execution between four backends
// (CPU_Sequential, CPU_Parallel, GPU_Metal, GPU_CUDA) based on whether the
// caller supplied an evmc::Host*, whether the txs carry bytecode, and which
// optional Config flags were enabled. These tests exercise:
//
//   1. CallNotSupported fallback: GPU kernel rejects CALL/CREATE; the
//      dispatcher must intercept and re-execute on evmone using the real
//      host. The internal CallNotSupported status MUST never leak to the
//      consumer.
//
//   2. fast_value_transfer guard: opting in without the explicit
//      acknowledgement flag is rejected at config-validate time.
//
//   3. The 4 × 4 routing matrix: every combination of (backend, has-host,
//      has-bytecode, gas-estimation) produces a sensible result — never a
//      crash, never a CallNotSupported leak, never a silent fake-gas.
//
//   4. Status translation: every kernel::TxStatus value (Stop, Return,
//      Revert, OutOfGas, Error, CallNotSupported) maps to the correct
//      dispatch::TxStatus when surfaced through execute_block().
//
//   5. Gas-estimation mode: explicit Config::gas_estimation_mode toggles the
//      "gas_used = gas_limit" shortcut. Default-off path is an Error.

#include "gpu/gpu_dispatch.hpp"

#include <evmc/mocked_host.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

extern "C" struct evmc_vm* evmc_create_evmone(void) noexcept;

namespace {

using evm::gpu::Backend;
using evm::gpu::BlockResult;
using evm::gpu::Config;
using evm::gpu::Transaction;
using evm::gpu::TxStatus;

// -- bytecode helpers ---------------------------------------------------------

inline void push1(std::vector<uint8_t>& c, uint8_t b)
{
    c.push_back(0x60);
    c.push_back(b);
}

/// PUSH1 size; PUSH1 offset; RETURN — emits memory[offset .. offset+size]
inline void emit_return(std::vector<uint8_t>& c, uint8_t off, uint8_t sz)
{
    push1(c, sz);
    push1(c, off);
    c.push_back(0xf3);
}

// 0xF1 — CALL. The GPU/CPU kernel surfaces CallNotSupported here. With a
// host the dispatcher must fall back to evmone (which actually services
// CALL via host->call()).
const uint8_t OP_CALL = 0xf1;

// -- transaction builders -----------------------------------------------------

Transaction make_tx_with_code(const std::vector<uint8_t>& code,
                              uint64_t gas_limit = 100'000)
{
    Transaction t;
    t.from.assign(20, 0);  t.from[19]  = 0x01;
    t.to.assign(20, 0);    t.to[19]    = 0x02;
    t.code      = code;
    t.gas_limit = gas_limit;
    t.gas_price = 1;
    t.value     = 0;
    t.nonce     = 0;
    return t;
}

Transaction make_tx_no_code(uint64_t gas_limit = 21'000)
{
    Transaction t;
    t.from.assign(20, 0);  t.from[19]  = 0x01;
    t.to.assign(20, 0);    t.to[19]    = 0x02;
    t.gas_limit = gas_limit;
    t.gas_price = 1;
    t.value     = 100;
    t.nonce     = 0;
    return t;
}

// -- shared host --------------------------------------------------------------

/// Set up a MockedHost where calling any address succeeds with a small
/// pre-baked result. Lets the CallNotSupported fallback path through evmone
/// reach a deterministic outcome.
evmc::MockedHost make_call_friendly_host()
{
    evmc::MockedHost host;
    host.tx_context.block_gas_limit = 30'000'000;
    host.tx_context.block_number    = 1;
    host.tx_context.block_timestamp = 1700000000;
    host.tx_context.tx_gas_price    = {};
    host.tx_context.chain_id        = {};

    // Leave call_result default — release==nullptr, gas_left==0,
    // status_code==EVMC_SUCCESS by default.
    host.call_result.status_code = EVMC_SUCCESS;
    host.call_result.gas_left = 0;
    host.call_result.gas_refund = 0;
    host.call_result.output_data = nullptr;
    host.call_result.output_size = 0;
    host.call_result.release = nullptr;

    return host;
}

}  // namespace

// -- 1. CallNotSupported fallback ---------------------------------------------

TEST(Dispatch, CallNotSupported_HostProvided_RoutesThroughEvmone)
{
    // When the caller passes a host, the dispatcher routes CALL bytecode
    // through evmone — bypassing the GPU kernel entirely (and therefore
    // never hitting the kernel's CallNotSupported branch). The result has
    // no per-tx status array (evmone path doesn't populate it) but the
    // gas accounting is the canonical evmone reference. Both CPU_Sequential
    // and GPU_Metal must agree on gas_used.
    std::vector<uint8_t> code;
    code.push_back(OP_CALL);
    auto tx = make_tx_with_code(code);

    auto host_gpu = make_call_friendly_host();
    Config cfg;
    cfg.backend = Backend::GPU_Metal;
    auto gpu_result = evm::gpu::execute_block(cfg, {tx}, &host_gpu);

    // Must not surface CallNotSupported under any circumstance.
    for (auto s : gpu_result.status)
        EXPECT_NE(s, TxStatus::CallNotSupported);

    EXPECT_TRUE(gpu_result.error_message.empty());
    ASSERT_EQ(gpu_result.gas_used.size(), 1u);

    auto host_cpu = make_call_friendly_host();
    Config cfg_cpu;
    cfg_cpu.backend = Backend::CPU_Sequential;
    auto cpu_result = evm::gpu::execute_block(cfg_cpu, {tx}, &host_cpu);

    ASSERT_EQ(cpu_result.gas_used.size(), 1u);
    EXPECT_EQ(cpu_result.gas_used[0], gpu_result.gas_used[0])
        << "CPU and GPU must agree on gas when both route through evmone";
    for (auto s : cpu_result.status)
        EXPECT_NE(s, TxStatus::CallNotSupported);
    EXPECT_TRUE(cpu_result.error_message.empty());
}

TEST(Dispatch, CallNotSupported_NoHost_RewritesToErrorWithMessage)
{
    // Same CALL bytecode, but no host provided. The dispatcher cannot
    // service the fallback, so CallNotSupported is rewritten to Error
    // with a clear error_message. CallNotSupported MUST NOT leak.
    std::vector<uint8_t> code;
    code.push_back(OP_CALL);
    auto tx = make_tx_with_code(code);

    Config cfg;
    cfg.backend = Backend::GPU_Metal;
    auto result = evm::gpu::execute_block(cfg, {tx}, /*state=*/nullptr);

    ASSERT_EQ(result.status.size(), 1u);
    EXPECT_NE(result.status[0], TxStatus::CallNotSupported)
        << "internal CallNotSupported leaked to consumer";
    EXPECT_EQ(result.status[0], TxStatus::Error);
    EXPECT_EQ(result.gpu_fallback_count, 1u);
    EXPECT_FALSE(result.error_message.empty());
    EXPECT_NE(result.error_message.find("CallNotSupported"), std::string::npos);
}

TEST(Dispatch, CallNotSupported_CpuPath_AlsoFiltered)
{
    // The CPU bytecode path (kernel::execute_cpu) also surfaces
    // CallNotSupported on 0xF1. The dispatcher applies the same filter.
    std::vector<uint8_t> code;
    code.push_back(OP_CALL);
    auto tx = make_tx_with_code(code);

    Config cfg;
    cfg.backend = Backend::CPU_Sequential;
    auto result = evm::gpu::execute_block(cfg, {tx}, /*state=*/nullptr);

    ASSERT_EQ(result.status.size(), 1u);
    EXPECT_NE(result.status[0], TxStatus::CallNotSupported);
    EXPECT_EQ(result.status[0], TxStatus::Error);
    EXPECT_EQ(result.gpu_fallback_count, 1u);
}

// -- 2. fast_value_transfer guard --------------------------------------------

TEST(Dispatch, FastValueTransfer_RequiresAcknowledgement)
{
    auto tx = make_tx_no_code();

    Config cfg;
    cfg.backend = Backend::CPU_Sequential;
    cfg.fast_value_transfer = true;
    cfg.fast_value_transfer_acknowledged = false;

    auto result = evm::gpu::execute_block(cfg, {tx}, /*state=*/nullptr);

    ASSERT_EQ(result.status.size(), 1u);
    EXPECT_EQ(result.status[0], TxStatus::Error);
    EXPECT_FALSE(result.error_message.empty());
    EXPECT_NE(result.error_message.find("fast_value_transfer"),
              std::string::npos);
    EXPECT_NE(result.error_message.find("acknowledged"), std::string::npos);
}

TEST(Dispatch, FastValueTransfer_AcknowledgedPasses)
{
    auto tx = make_tx_no_code();

    Config cfg;
    cfg.backend = Backend::CPU_Sequential;
    cfg.fast_value_transfer = true;
    cfg.fast_value_transfer_acknowledged = true;
    cfg.gas_estimation_mode = true;  // no host + no code path

    auto result = evm::gpu::execute_block(cfg, {tx}, /*state=*/nullptr);

    // Validation passes; no error_message.
    EXPECT_TRUE(result.error_message.empty());
}

// -- 3. backend routing matrix ------------------------------------------------
//
// 16 cells: 4 backends × {has-code, no-code} × {has-host, no-host}.
// Each cell asserts: doesn't crash, status doesn't leak CallNotSupported,
// gas accounting is sensible (matches the documented routing tree).

namespace {

struct MatrixCell
{
    Backend     backend;
    bool        has_code;
    bool        has_host;
};

void run_cell(const MatrixCell& cell)
{
    SCOPED_TRACE(::testing::Message()
        << "backend=" << evm::gpu::backend_name(cell.backend)
        << " has_code=" << cell.has_code
        << " has_host=" << cell.has_host);

    std::vector<uint8_t> code;
    if (cell.has_code)
    {
        // Simple program: PUSH1 0x42, PUSH1 0x00, MSTORE; PUSH1 0x20,
        // PUSH1 0x00, RETURN. Returns 32 bytes, leading byte 0x42.
        push1(code, 0x42);
        push1(code, 0x00);
        code.push_back(0x52);  // MSTORE
        emit_return(code, 0x00, 0x20);
    }

    auto tx = cell.has_code ? make_tx_with_code(code) : make_tx_no_code();

    Config cfg;
    cfg.backend = cell.backend;
    // Allow the gas-estimation path so the no-host/no-code cells produce
    // a non-error result (we want to test routing, not the gating gate).
    cfg.gas_estimation_mode = true;

    evmc::MockedHost host;
    auto result = evm::gpu::execute_block(cfg, {tx},
        cell.has_host ? &host : nullptr);

    // Universal assertions for every cell.
    ASSERT_EQ(result.gas_used.size(), 1u);
    if (!result.status.empty())
    {
        EXPECT_NE(result.status[0], TxStatus::CallNotSupported)
            << "internal CallNotSupported leaked";
    }
    if (!cell.has_host && cell.has_code)
    {
        // The bytecode path runs through the kernel/GPU interpreter and
        // populates per-tx status + output. Our small program returns
        // 32 bytes successfully, so status must be Return.
        ASSERT_EQ(result.status.size(), 1u);
        EXPECT_EQ(result.status[0], TxStatus::Return);
        EXPECT_LE(result.gas_used[0], tx.gas_limit);
    }
    if (cell.has_host && !cell.has_code)
    {
        // Value-transfer through evmone: gas_used should be small (no
        // bytecode means evmone runs an empty program).
        EXPECT_LE(result.gas_used[0], tx.gas_limit);
    }
    if (!cell.has_host && !cell.has_code)
    {
        // gas-estimation shortcut.
        EXPECT_EQ(result.gas_used[0], tx.gas_limit);
    }
}

}  // namespace

TEST(Dispatch, RoutingMatrix_AllSixteenCells)
{
    auto avail = evm::gpu::available_backends();
    auto has = [&](Backend b) {
        for (auto a : avail) if (a == b) return true;
        return false;
    };

    for (auto backend : {Backend::CPU_Sequential, Backend::CPU_Parallel,
                          Backend::GPU_Metal,     Backend::GPU_CUDA})
    {
        // Skip backends not built into this binary — set_backend would
        // refuse anyway, and there's no interpreter to exercise.
        if (!has(backend)) continue;

        for (bool has_code : {true, false})
            for (bool has_host : {true, false})
                run_cell({backend, has_code, has_host});
    }
}

// -- 4. status code translation -----------------------------------------------

TEST(Dispatch, StatusTranslation_Stop)
{
    // STOP opcode (0x00) → status::Stop.
    std::vector<uint8_t> code{0x00};
    auto tx = make_tx_with_code(code);
    Config cfg;
    cfg.backend = Backend::CPU_Sequential;
    auto r = evm::gpu::execute_block(cfg, {tx}, /*state=*/nullptr);
    ASSERT_EQ(r.status.size(), 1u);
    EXPECT_EQ(r.status[0], TxStatus::Stop);
}

TEST(Dispatch, StatusTranslation_Return)
{
    // PUSH1 0x42, PUSH1 0x00, MSTORE; PUSH1 0x20, PUSH1 0x00, RETURN.
    std::vector<uint8_t> code;
    push1(code, 0x42);
    push1(code, 0x00);
    code.push_back(0x52);  // MSTORE
    emit_return(code, 0x00, 0x20);
    auto tx = make_tx_with_code(code);

    Config cfg;
    cfg.backend = Backend::CPU_Sequential;
    auto r = evm::gpu::execute_block(cfg, {tx}, /*state=*/nullptr);
    ASSERT_EQ(r.status.size(), 1u);
    EXPECT_EQ(r.status[0], TxStatus::Return);
    ASSERT_EQ(r.output[0].size(), 32u);
    EXPECT_EQ(r.output[0][31], 0x42);
}

TEST(Dispatch, StatusTranslation_Revert)
{
    // PUSH1 0x00, PUSH1 0x00, REVERT — empty revert message.
    std::vector<uint8_t> code;
    push1(code, 0x00);
    push1(code, 0x00);
    code.push_back(0xfd);  // REVERT
    auto tx = make_tx_with_code(code);
    Config cfg;
    cfg.backend = Backend::CPU_Sequential;
    auto r = evm::gpu::execute_block(cfg, {tx}, /*state=*/nullptr);
    ASSERT_EQ(r.status.size(), 1u);
    EXPECT_EQ(r.status[0], TxStatus::Revert);
}

TEST(Dispatch, StatusTranslation_OutOfGas)
{
    // Same return program but only 5 gas — runs out partway through.
    std::vector<uint8_t> code;
    push1(code, 0x42);
    push1(code, 0x00);
    code.push_back(0x52);
    emit_return(code, 0x00, 0x20);
    auto tx = make_tx_with_code(code, /*gas_limit=*/5);

    Config cfg;
    cfg.backend = Backend::CPU_Sequential;
    auto r = evm::gpu::execute_block(cfg, {tx}, /*state=*/nullptr);
    ASSERT_EQ(r.status.size(), 1u);
    EXPECT_EQ(r.status[0], TxStatus::OutOfGas);
}

TEST(Dispatch, StatusTranslation_Error_OnInvalidOpcode)
{
    // 0xFE = INVALID — universal "abort with error" instruction.
    std::vector<uint8_t> code{0xfe};
    auto tx = make_tx_with_code(code);
    Config cfg;
    cfg.backend = Backend::CPU_Sequential;
    auto r = evm::gpu::execute_block(cfg, {tx}, /*state=*/nullptr);
    ASSERT_EQ(r.status.size(), 1u);
    EXPECT_EQ(r.status[0], TxStatus::Error);
}

TEST(Dispatch, StatusTranslation_CallNotSupported_NeverLeaks)
{
    // Verifies the public-facing contract: no caller path can observe
    // CallNotSupported. We try every backend combination that triggers
    // the kernel interpreter (no host, with code).
    std::vector<uint8_t> code{OP_CALL};
    auto tx = make_tx_with_code(code);

    for (auto backend : {Backend::CPU_Sequential, Backend::CPU_Parallel,
                          Backend::GPU_Metal})
    {
        SCOPED_TRACE(::testing::Message()
            << "backend=" << evm::gpu::backend_name(backend));
        Config cfg;
        cfg.backend = backend;
        auto r = evm::gpu::execute_block(cfg, {tx}, /*state=*/nullptr);
        ASSERT_EQ(r.status.size(), 1u);
        EXPECT_NE(r.status[0], TxStatus::CallNotSupported);
    }
}

// -- 5. gas-estimation mode ---------------------------------------------------

TEST(Dispatch, GasEstimation_DefaultOff_NoHost_NoCode_IsError)
{
    auto tx = make_tx_no_code();
    Config cfg;
    cfg.backend = Backend::CPU_Sequential;
    // gas_estimation_mode = false (default)
    auto r = evm::gpu::execute_block(cfg, {tx}, /*state=*/nullptr);

    ASSERT_EQ(r.status.size(), 1u);
    EXPECT_EQ(r.status[0], TxStatus::Error);
    EXPECT_FALSE(r.error_message.empty());
    EXPECT_NE(r.error_message.find("gas_estimation_mode"),
              std::string::npos);
}

TEST(Dispatch, GasEstimation_OptIn_ReturnsGasLimit)
{
    auto tx = make_tx_no_code();
    tx.gas_limit = 21'000;

    Config cfg;
    cfg.backend = Backend::CPU_Sequential;
    cfg.gas_estimation_mode = true;
    auto r = evm::gpu::execute_block(cfg, {tx}, /*state=*/nullptr);

    EXPECT_TRUE(r.error_message.empty());
    ASSERT_EQ(r.gas_used.size(), 1u);
    EXPECT_EQ(r.gas_used[0], 21'000u);
}

TEST(Dispatch, GasEstimation_NotInvoked_WhenHostPresent)
{
    auto tx = make_tx_no_code();
    tx.gas_limit = 21'000;

    Config cfg;
    cfg.backend = Backend::CPU_Sequential;
    cfg.gas_estimation_mode = true;  // ignored when host present

    evmc::MockedHost host;
    auto r = evm::gpu::execute_block(cfg, {tx}, &host);

    // gas_used should NOT be the full gas_limit — evmone ran an empty
    // program (no code installed) and returned without consuming gas.
    ASSERT_EQ(r.gas_used.size(), 1u);
    EXPECT_LT(r.gas_used[0], tx.gas_limit);
}
