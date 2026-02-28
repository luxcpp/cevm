// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file host_bridge_test.cpp
/// Unit tests for the GPU EVMC host bridge.
///
/// These tests verify the bridge's accept/reject decisions:
///   * Pure leaf      → accepted (eligible for GPU kernel)
///   * Static CALL    → accepted (with code-dictionary pre-resolution)
///   * Dynamic CALL   → rejected (CPU fallback required)
///   * Deep call chain (≤ MAX_FRAME_DEPTH) → accepted
///   * Deeper than MAX_FRAME_DEPTH → rejected
///   * DELEGATECALL preserves the caller's apparent state
///
/// On accept, the result is compared against running the same transaction
/// through evmone with the same evmc::Host. The two must match byte-for-
/// byte (same status, same gas_used, same output).

#include "../../lib/evm/gpu/host/code_resolver.hpp"
#include "../../lib/evm/gpu/host/evmc_bridge.hpp"

#include <evmc/mocked_host.hpp>
#include <gtest/gtest.h>

#include <cstring>
#include <vector>

extern "C" struct evmc_vm* evmc_create_evmone(void) noexcept;

namespace evm::gpu::host::test {

using namespace evmc::literals;

// -- Helpers ------------------------------------------------------------------

namespace {

evmc::address addr_from_byte(uint8_t b)
{
    evmc::address a{};
    a.bytes[19] = b;
    return a;
}

void install_code(evmc::MockedHost& host, const evmc::address& addr,
                  const std::vector<uint8_t>& code)
{
    auto& acct = host.accounts[addr];
    acct.code.assign(code.begin(), code.end());
}

// Run the same tx through evmone (the "ground truth" reference).
TxResult evmone_reference(
    evmc::Host& host, const Transaction& tx, evmc_revision rev)
{
    auto* vm = evmc_create_evmone();

    std::vector<uint8_t> code;
    if (tx.is_create)
    {
        code = tx.input;
    }
    else
    {
        const auto sz = host.get_code_size(tx.recipient);
        code.resize(sz);
        if (sz > 0)
            host.copy_code(tx.recipient, 0, code.data(), sz);
    }

    evmc_message msg{};
    msg.kind = tx.is_create ? EVMC_CREATE : EVMC_CALL;
    msg.depth = 0;
    msg.gas = tx.gas;
    msg.recipient = tx.recipient;
    msg.sender = tx.sender;
    msg.input_data = tx.input.empty() ? nullptr : tx.input.data();
    msg.input_size = tx.input.size();
    msg.value = tx.value;
    msg.code_address = tx.recipient;

    const auto& iface = evmc::Host::get_interface();
    auto* ctx = host.to_context();
    auto r = vm->execute(vm, &iface, ctx, rev, &msg,
                         code.empty() ? nullptr : code.data(), code.size());

    TxResult out;
    out.status = r.status_code;
    out.gas_used = tx.gas - r.gas_left;
    out.gas_refund = r.gas_refund;
    if (r.output_size > 0 && r.output_data != nullptr)
        out.output.assign(r.output_data, r.output_data + r.output_size);
    if (r.release != nullptr)
        r.release(&r);
    vm->destroy(vm);
    return out;
}

void expect_results_equal(const TxResult& a, const TxResult& b)
{
    EXPECT_EQ(a.status, b.status);
    EXPECT_EQ(a.gas_used, b.gas_used);
    EXPECT_EQ(a.output, b.output);
}

// Build a "PUSH20 <addr> ; rest" prefix.
void emit_push20(std::vector<uint8_t>& out, const evmc::address& addr)
{
    out.push_back(0x73);  // PUSH20
    out.insert(out.end(), std::begin(addr.bytes), std::end(addr.bytes));
}

// PUSH1 <byte>
void emit_push1(std::vector<uint8_t>& out, uint8_t b)
{
    out.push_back(0x60);
    out.push_back(b);
}

}  // namespace

// -- Tests --------------------------------------------------------------------

TEST(host_bridge, pure_leaf_no_calls_runs_on_gpu)
{
    // Bytecode: PUSH1 0x42 ; PUSH1 0 ; MSTORE ; PUSH1 32 ; PUSH1 0 ; RETURN
    std::vector<uint8_t> code;
    emit_push1(code, 0x42);
    emit_push1(code, 0);
    code.push_back(0x52);          // MSTORE
    emit_push1(code, 32);
    emit_push1(code, 0);
    code.push_back(0xf3);          // RETURN

    evmc::MockedHost host;
    const auto recipient = addr_from_byte(0xaa);
    install_code(host, recipient, code);

    Transaction tx;
    tx.recipient = recipient;
    tx.sender = addr_from_byte(0x01);
    tx.gas = 100'000;

    auto bridge = GpuHostBridge::create(host);
    auto result = bridge->try_execute(tx, EVMC_SHANGHAI);

    ASSERT_TRUE(result.has_value()) << "pure leaf should be accepted";
    EXPECT_TRUE(result->used_gpu);
    EXPECT_EQ(result->status, EVMC_SUCCESS);

    auto reference = evmone_reference(host, tx, EVMC_SHANGHAI);
    expect_results_equal(*result, reference);
}

TEST(host_bridge, static_call_with_constant_target_is_accepted)
{
    // Outer caller contract: STATICCALL to a known address.
    //   PUSH1 0    ; out_size
    //   PUSH1 0    ; out_off
    //   PUSH1 0    ; in_size
    //   PUSH1 0    ; in_off
    //   PUSH20 <callee>
    //   PUSH3 fffff ; gas
    //   STATICCALL
    //   STOP
    const auto callee = addr_from_byte(0xcc);

    std::vector<uint8_t> caller_code;
    emit_push1(caller_code, 0);
    emit_push1(caller_code, 0);
    emit_push1(caller_code, 0);
    emit_push1(caller_code, 0);
    emit_push20(caller_code, callee);
    caller_code.push_back(0x62);          // PUSH3
    caller_code.push_back(0x0f);
    caller_code.push_back(0xff);
    caller_code.push_back(0xff);
    caller_code.push_back(0xfa);          // STATICCALL
    caller_code.push_back(0x00);          // STOP

    // Pure callee: STOP.
    std::vector<uint8_t> callee_code = {0x00};

    evmc::MockedHost host;
    const auto recipient = addr_from_byte(0xaa);
    install_code(host, recipient, caller_code);
    install_code(host, callee, callee_code);

    Transaction tx;
    tx.recipient = recipient;
    tx.sender = addr_from_byte(0x01);
    tx.gas = 200'000;

    // Sanity-check the static analyzer agrees the caller is fully static.
    const auto info = analyze(std::span<const uint8_t>{
        host.accounts[recipient].code.data(), host.accounts[recipient].code.size()});
    ASSERT_FALSE(info.pure);
    ASSERT_TRUE(info.all_static);

    auto bridge = GpuHostBridge::create(host);
    auto result = bridge->try_execute(tx, EVMC_SHANGHAI);

    ASSERT_TRUE(result.has_value()) << "static-target CALL should be accepted";
    auto reference = evmone_reference(host, tx, EVMC_SHANGHAI);
    expect_results_equal(*result, reference);
}

TEST(host_bridge, dynamic_call_target_is_rejected)
{
    // Outer code uses CALLDATALOAD to compute the target address.
    //   PUSH1 0   ; out_size
    //   PUSH1 0   ; out_off
    //   PUSH1 0   ; in_size
    //   PUSH1 0   ; in_off
    //   PUSH1 0   ; value
    //   PUSH1 0   ; offset for CALLDATALOAD
    //   CALLDATALOAD
    //   PUSH3 fffff ; gas
    //   CALL
    //   STOP
    std::vector<uint8_t> code;
    emit_push1(code, 0);
    emit_push1(code, 0);
    emit_push1(code, 0);
    emit_push1(code, 0);
    emit_push1(code, 0);
    emit_push1(code, 0);
    code.push_back(0x35);          // CALLDATALOAD
    code.push_back(0x62);
    code.push_back(0x0f);
    code.push_back(0xff);
    code.push_back(0xff);
    code.push_back(0xf1);          // CALL
    code.push_back(0x00);

    evmc::MockedHost host;
    const auto recipient = addr_from_byte(0xaa);
    install_code(host, recipient, code);

    Transaction tx;
    tx.recipient = recipient;
    tx.sender = addr_from_byte(0x01);
    tx.gas = 200'000;
    tx.input.resize(32);
    tx.input[31] = 0xcc;  // dynamic target

    auto bridge = GpuHostBridge::create(host);
    auto result = bridge->try_execute(tx, EVMC_SHANGHAI);

    EXPECT_FALSE(result.has_value())
        << "dynamic CALL target must drive CPU fallback";
}

TEST(host_bridge, deterministic_create_runs_on_gpu)
{
    // Initcode: PUSH1 1 ; PUSH1 0 ; MSTORE ; PUSH1 32 ; PUSH1 0 ; RETURN
    // (Returns a single non-zero byte — semantically a no-op contract body.)
    std::vector<uint8_t> initcode;
    emit_push1(initcode, 1);
    emit_push1(initcode, 0);
    initcode.push_back(0x52);     // MSTORE
    emit_push1(initcode, 32);
    emit_push1(initcode, 0);
    initcode.push_back(0xf3);     // RETURN

    evmc::MockedHost host;

    Transaction tx;
    tx.is_create = true;
    tx.sender = addr_from_byte(0x01);
    tx.gas = 200'000;
    tx.input = initcode;

    auto bridge = GpuHostBridge::create(host);
    auto result = bridge->try_execute(tx, EVMC_SHANGHAI);

    ASSERT_TRUE(result.has_value()) << "deterministic CREATE should be accepted";
    EXPECT_TRUE(result->used_gpu);
    auto reference = evmone_reference(host, tx, EVMC_SHANGHAI);
    expect_results_equal(*result, reference);
}

TEST(host_bridge, four_deep_static_call_chain_runs_on_gpu)
{
    // Each depth-N contract calls a depth-(N+1) contract. All targets are
    // PUSH20 constants. Final contract is pure (STOP).

    evmc::MockedHost host;
    const auto a0 = addr_from_byte(0x10);
    const auto a1 = addr_from_byte(0x11);
    const auto a2 = addr_from_byte(0x12);
    const auto a3 = addr_from_byte(0x13);

    auto build_caller = [](const evmc::address& target) {
        std::vector<uint8_t> c;
        emit_push1(c, 0);
        emit_push1(c, 0);
        emit_push1(c, 0);
        emit_push1(c, 0);
        emit_push20(c, target);
        c.push_back(0x62);
        c.push_back(0x0f);
        c.push_back(0xff);
        c.push_back(0xff);
        c.push_back(0xfa);   // STATICCALL
        c.push_back(0x00);
        return c;
    };

    install_code(host, a0, build_caller(a1));
    install_code(host, a1, build_caller(a2));
    install_code(host, a2, build_caller(a3));
    install_code(host, a3, std::vector<uint8_t>{0x00});  // pure STOP

    Transaction tx;
    tx.recipient = a0;
    tx.sender = addr_from_byte(0x01);
    tx.gas = 500'000;

    auto bridge = GpuHostBridge::create(host);
    auto result = bridge->try_execute(tx, EVMC_SHANGHAI);

    ASSERT_TRUE(result.has_value()) << "depth-4 chain must be accepted";
    auto reference = evmone_reference(host, tx, EVMC_SHANGHAI);
    expect_results_equal(*result, reference);
}

TEST(host_bridge, nine_deep_static_call_chain_is_rejected)
{
    // depth 9 > MAX_FRAME_DEPTH (8): bridge must bail and return nullopt.
    evmc::MockedHost host;
    std::vector<evmc::address> chain;
    for (uint8_t i = 0; i < 10; ++i)
        chain.push_back(addr_from_byte(static_cast<uint8_t>(0x20 + i)));

    auto build_caller = [](const evmc::address& target) {
        std::vector<uint8_t> c;
        emit_push1(c, 0);
        emit_push1(c, 0);
        emit_push1(c, 0);
        emit_push1(c, 0);
        emit_push20(c, target);
        c.push_back(0x62);
        c.push_back(0x0f);
        c.push_back(0xff);
        c.push_back(0xff);
        c.push_back(0xfa);
        c.push_back(0x00);
        return c;
    };

    for (size_t i = 0; i + 1 < chain.size(); ++i)
        install_code(host, chain[i], build_caller(chain[i + 1]));
    install_code(host, chain.back(), std::vector<uint8_t>{0x00});

    Transaction tx;
    tx.recipient = chain[0];
    tx.sender = addr_from_byte(0x01);
    tx.gas = 1'000'000;

    auto bridge = GpuHostBridge::create(host);
    auto result = bridge->try_execute(tx, EVMC_SHANGHAI);

    EXPECT_FALSE(result.has_value())
        << "depth > MAX_FRAME_DEPTH must trigger CPU fallback";
}

TEST(host_bridge, delegatecall_preserves_caller_context)
{
    // DELEGATECALL: callee runs with the caller's recipient/value/sender.
    // Verify the bridge's result equals evmone's: that's what makes the
    // bridge "correct" for DELEGATECALL — it doesn't have to implement
    // the semantics itself, just confirm the analysis is happy with the
    // static call site, then defer to the host-driven execution.
    const auto callee = addr_from_byte(0xcc);

    // Outer: DELEGATECALL <callee> ; STOP
    std::vector<uint8_t> caller_code;
    emit_push1(caller_code, 0);  // out_size
    emit_push1(caller_code, 0);  // out_off
    emit_push1(caller_code, 0);  // in_size
    emit_push1(caller_code, 0);  // in_off
    emit_push20(caller_code, callee);
    caller_code.push_back(0x62);
    caller_code.push_back(0x0f);
    caller_code.push_back(0xff);
    caller_code.push_back(0xff);
    caller_code.push_back(0xf4);   // DELEGATECALL
    caller_code.push_back(0x00);

    // Callee: ADDRESS ; PUSH1 0 ; MSTORE ; PUSH1 32 ; PUSH1 0 ; RETURN
    // It returns the executing address — under DELEGATECALL this should be
    // the *caller's* address.
    std::vector<uint8_t> callee_code;
    callee_code.push_back(0x30);    // ADDRESS
    emit_push1(callee_code, 0);
    callee_code.push_back(0x52);
    emit_push1(callee_code, 32);
    emit_push1(callee_code, 0);
    callee_code.push_back(0xf3);

    evmc::MockedHost host;
    const auto recipient = addr_from_byte(0xaa);
    install_code(host, recipient, caller_code);
    install_code(host, callee, callee_code);

    Transaction tx;
    tx.recipient = recipient;
    tx.sender = addr_from_byte(0x01);
    tx.gas = 200'000;

    auto bridge = GpuHostBridge::create(host);
    auto result = bridge->try_execute(tx, EVMC_SHANGHAI);

    ASSERT_TRUE(result.has_value());
    auto reference = evmone_reference(host, tx, EVMC_SHANGHAI);
    expect_results_equal(*result, reference);
}

}  // namespace evm::gpu::host::test
