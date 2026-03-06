// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// EIP-2929 access-list test for the GPU EVM kernels.
//
// EIP-2929 (Berlin, 2021) introduced cold/warm pricing for state-access
// opcodes. The kernels (CPU header-only interpreter, Metal, CUDA) must
// charge:
//
//   SLOAD       : cold 2100 / warm 100 — keyed on (contract, slot)
//   BALANCE     : cold 2600 / warm 100 — keyed on address
//   EXTCODE*    : cold 2600 / warm 100
//   SSTORE      : EIP-2200 base + 2100 cold surcharge on first slot access
//
// At tx start the kernels pre-warm:
//   - the tx's `from` address (caller)
//   - the tx's `to` address (recipient / contract under execution)
//   - precompiles 0x01..0x11 (Cancun + Prague superset)
//   - any caller-supplied entries from `Config::warm_addresses` and
//     `Config::warm_storage_keys` (EIP-2930 access list shape)
//
// The kernels also mark slots/addresses warm as they're touched during
// execution, so the second SLOAD of the same slot pays the warm price.

#include "gpu/gpu_dispatch.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace {

using evm::gpu::Backend;
using evm::gpu::BlockResult;
using evm::gpu::Config;
using evm::gpu::Transaction;
using evm::gpu::TxStatus;

// EIP-2929 constants. These are the ground-truth values cevm (and every
// other Ethereum client) charges; the kernels MUST match.
constexpr uint64_t SLOAD_COLD   = 2100;
constexpr uint64_t SLOAD_WARM   = 100;
constexpr uint64_t ACCOUNT_COLD = 2600;
constexpr uint64_t ACCOUNT_WARM = 100;

// -- bytecode helpers ---------------------------------------------------------

inline void push1(std::vector<uint8_t>& c, uint8_t b)
{
    c.push_back(0x60);
    c.push_back(b);
}

/// Emit: PUSH1 size; PUSH1 offset; RETURN. Reads memory[offset, offset+size).
inline void emit_return(std::vector<uint8_t>& c, uint8_t off, uint8_t sz)
{
    push1(c, sz);
    push1(c, off);
    c.push_back(0xf3);
}

// -- transaction builders -----------------------------------------------------

Transaction make_tx_with_code(const std::vector<uint8_t>& code,
                              uint64_t gas_limit = 1'000'000)
{
    Transaction t;
    t.from.assign(20, 0); t.from[19] = 0x01;
    t.to.assign(20, 0);   t.to[19]   = 0x02;
    t.code      = code;
    t.gas_limit = gas_limit;
    t.gas_price = 1;
    return t;
}

BlockResult run_one(Backend backend, const Transaction& tx, const Config& cfg_in = {})
{
    Config cfg = cfg_in;
    cfg.backend = backend;
    cfg.num_threads = 2;
    std::vector<Transaction> txs{tx};
    return evm::gpu::execute_block(cfg, txs, /*state=*/nullptr);
}

// Compute the framework cost of a tx that wraps a state-access opcode in
// PUSH/MSTORE/RETURN scaffolding. The scaffold's gas is shared across every
// vector so we can compute the expected gas as: scaffold + opcode_cost.
//
// Layout used in the tests below:
//   PUSH1 <arg>      -- 3
//   <op>             -- variable (the opcode we're testing)
//   PUSH1 0   MSTORE -- 3 + 3 + memory expand 3 (1 word) = 9
//   PUSH1 32  PUSH1 0  RETURN -- 3 + 3 + 0 = 6
// = 3 (PUSH1 arg) + 9 (MSTORE @0) + 6 (RETURN) = 18.
constexpr uint64_t SCAFFOLD_PUSH_MSTORE_RETURN = 3 + 3 + 3 + 3 + 3 + 3;

/// Run the same scenario through CPU_Sequential (the kernel CPU
/// interpreter) and assert the gas matches `expected`.
void expect_cpu_gas(const Transaction& tx, uint64_t expected,
                    const Config& cfg = {})
{
    auto r = run_one(Backend::CPU_Sequential, tx, cfg);
    ASSERT_EQ(r.status.size(), 1u);
    EXPECT_EQ(r.status[0], TxStatus::Return) << "tx must succeed";
    EXPECT_EQ(r.gas_used[0], expected)
        << "CPU gas mismatch (got " << r.gas_used[0]
        << ", expected " << expected << ")";
}

/// Run through GPU_Metal and assert gas matches.
void expect_metal_gas(const Transaction& tx, uint64_t expected,
                      const Config& cfg = {})
{
    auto r = run_one(Backend::GPU_Metal, tx, cfg);
    ASSERT_EQ(r.status.size(), 1u);
    EXPECT_EQ(r.status[0], TxStatus::Return) << "tx must succeed";
    EXPECT_EQ(r.gas_used[0], expected)
        << "Metal gas mismatch (got " << r.gas_used[0]
        << ", expected " << expected << ")";
}

}  // namespace

// =============================================================================
// Scenario 1: two SLOADs of the same slot. First pays cold, second pays warm.
// Run on CPU_Sequential (the kernel CPU interpreter implements SLOAD) and
// on GPU_Metal. CUDA mirrors Metal.
// =============================================================================

TEST(AccessList, SloadTwiceSameSlot_ColdThenWarm_CPU)
{
    // PUSH1 7  SLOAD POP    -- first SLOAD: cold (2100)
    // PUSH1 7  SLOAD        -- second SLOAD: warm (100)
    // PUSH1 0  MSTORE       -- store top of stack at mem[0]
    // PUSH1 32 PUSH1 0 RETURN
    std::vector<uint8_t> code = {
        0x60, 0x07, 0x54, 0x50,            // PUSH1 7 SLOAD POP
        0x60, 0x07, 0x54,                  // PUSH1 7 SLOAD
        0x60, 0x00, 0x52,                  // PUSH1 0 MSTORE
        0x60, 0x20, 0x60, 0x00, 0xf3,      // PUSH1 32 PUSH1 0 RETURN
    };

    // Expected gas:
    //   PUSH1 7        : 3
    //   SLOAD (cold)   : 2100
    //   POP            : 2
    //   PUSH1 7        : 3
    //   SLOAD (warm)   : 100   <- second access to slot 7
    //   PUSH1 0        : 3
    //   MSTORE         : 3 + 3 (memory expansion: 1 word)
    //   PUSH1 32       : 3
    //   PUSH1 0        : 3
    //   RETURN         : 0
    // Total: 3+2100+2+3+100+3+3+3+3+3 = 2223
    const uint64_t expected = 3 + SLOAD_COLD + 2 + 3 + SLOAD_WARM
                            + 3 + 3 + 3 + 3 + 3;
    expect_cpu_gas(make_tx_with_code(code), expected);
}

TEST(AccessList, SloadTwiceSameSlot_ColdThenWarm_Metal)
{
    std::vector<uint8_t> code = {
        0x60, 0x07, 0x54, 0x50,
        0x60, 0x07, 0x54,
        0x60, 0x00, 0x52,
        0x60, 0x20, 0x60, 0x00, 0xf3,
    };
    const uint64_t expected = 3 + SLOAD_COLD + 2 + 3 + SLOAD_WARM
                            + 3 + 3 + 3 + 3 + 3;
    expect_metal_gas(make_tx_with_code(code), expected);
}

// =============================================================================
// Scenario 2: BALANCE of the contract's own address — pre-warmed, pays warm.
// Only Metal/CUDA implement BALANCE; the kernel CPU interpreter does not.
// =============================================================================

TEST(AccessList, BalanceOwnAddress_Warm_Metal)
{
    // ADDRESS BALANCE PUSH1 0 MSTORE PUSH1 32 PUSH1 0 RETURN
    // ADDRESS pushes the contract's own address (pre-warmed).
    std::vector<uint8_t> code = {
        0x30,                              // ADDRESS (2 gas)
        0x31,                              // BALANCE (warm: 100)
        0x60, 0x00, 0x52,                  // PUSH1 0 MSTORE
        0x60, 0x20, 0x60, 0x00, 0xf3,      // PUSH1 32 PUSH1 0 RETURN
    };
    // Gas: ADDRESS 2 + BALANCE warm 100 + PUSH1 3 + MSTORE 3 + memexp 3
    //    + PUSH1 3 + PUSH1 3 + RETURN 0 = 117
    const uint64_t expected = 2 + ACCOUNT_WARM + 3 + 3 + 3 + 3 + 3;
    expect_metal_gas(make_tx_with_code(code), expected);
}

// =============================================================================
// Scenario 3: BALANCE of an unrelated cold address — pays cold (2600).
// =============================================================================

TEST(AccessList, BalanceColdAddress_Cold_Metal)
{
    // PUSH1 0xAA  BALANCE  PUSH1 0 MSTORE  PUSH1 32 PUSH1 0 RETURN
    std::vector<uint8_t> code = {
        0x60, 0xAA,                        // PUSH1 0xAA (3)
        0x31,                              // BALANCE (cold: 2600)
        0x60, 0x00, 0x52,                  // PUSH1 0 MSTORE
        0x60, 0x20, 0x60, 0x00, 0xf3,      // PUSH1 32 PUSH1 0 RETURN
    };
    const uint64_t expected = 3 + ACCOUNT_COLD + 3 + 3 + 3 + 3 + 3;
    expect_metal_gas(make_tx_with_code(code), expected);
}

// =============================================================================
// Scenario 4: pre-warm a slot via Config::warm_storage_keys; SLOAD pays warm.
// Tests the dispatcher → HostTransaction → kernel pre-warm path. Run on CPU
// where the kernel interpreter consumes the warm sets directly through
// EvmInterpreter.warm_slot_*.
// =============================================================================

TEST(AccessList, PreWarmedSlot_Config_PaysWarm_CPU)
{
    // PUSH1 7 SLOAD PUSH1 0 MSTORE PUSH1 32 PUSH1 0 RETURN
    std::vector<uint8_t> code = {
        0x60, 0x07, 0x54,                  // PUSH1 7 SLOAD (warm because Config says so)
        0x60, 0x00, 0x52,                  // PUSH1 0 MSTORE
        0x60, 0x20, 0x60, 0x00, 0xf3,      // PUSH1 32 PUSH1 0 RETURN
    };
    Transaction tx = make_tx_with_code(code);

    // Build a Config that pre-warms (contract_addr, slot=7).
    Config cfg;
    cfg.warm_storage_keys.resize(52);
    // contract_addr: tx.to (right-aligned 20 bytes — last byte is 0x02).
    std::memcpy(cfg.warm_storage_keys.data(), tx.to.data(), 20);
    // slot key (big-endian 32 bytes ending in 0x07).
    cfg.warm_storage_keys[20 + 31] = 0x07;

    // Gas: PUSH1 3 + SLOAD warm 100 + PUSH1 3 + MSTORE 3 + memexp 3
    //    + PUSH1 3 + PUSH1 3 + RETURN 0 = 118
    const uint64_t expected = 3 + SLOAD_WARM + 3 + 3 + 3 + 3 + 3;
    expect_cpu_gas(tx, expected, cfg);
}

// =============================================================================
// Scenario 5: pre-warm an address via Config::warm_addresses; CPU interp
// doesn't implement BALANCE, so we exercise the Metal path only. This test
// also proves the Config → blob → kernel pipeline works for addresses.
// =============================================================================

TEST(AccessList, PreWarmedAddress_Config_PaysWarm_Metal)
{
    std::vector<uint8_t> code = {
        0x60, 0xAA,                        // PUSH1 0xAA
        0x31,                              // BALANCE (warm because Config says so)
        0x60, 0x00, 0x52,                  // PUSH1 0 MSTORE
        0x60, 0x20, 0x60, 0x00, 0xf3,      // PUSH1 32 PUSH1 0 RETURN
    };
    Transaction tx = make_tx_with_code(code);

    Config cfg;
    cfg.warm_addresses.resize(20, 0);
    cfg.warm_addresses[19] = 0xAA;        // 20-byte address ending in 0xAA

    const uint64_t expected = 3 + ACCOUNT_WARM + 3 + 3 + 3 + 3 + 3;
    expect_metal_gas(tx, expected, cfg);
}

// =============================================================================
// Scenario 6: parity. The same SLOAD-twice tx run through CPU_Sequential
// and GPU_Metal MUST report identical gas. (CUDA mirrors Metal byte-for-byte
// — this assertion plus the parity test corpus covers it.)
// =============================================================================

TEST(AccessList, CpuVsMetal_Parity_SloadTwice)
{
    std::vector<uint8_t> code = {
        0x60, 0x07, 0x54, 0x50,
        0x60, 0x07, 0x54,
        0x60, 0x00, 0x52,
        0x60, 0x20, 0x60, 0x00, 0xf3,
    };
    Transaction tx = make_tx_with_code(code);

    auto r_cpu   = run_one(Backend::CPU_Sequential, tx);
    auto r_metal = run_one(Backend::GPU_Metal,      tx);
    ASSERT_EQ(r_cpu.status.size(),   1u);
    ASSERT_EQ(r_metal.status.size(), 1u);
    EXPECT_EQ(r_cpu.gas_used[0], r_metal.gas_used[0])
        << "CPU/Metal must agree on gas for SLOAD-twice";
    EXPECT_EQ(r_cpu.status[0], r_metal.status[0]);
    EXPECT_EQ(r_cpu.output[0], r_metal.output[0]);
}

// =============================================================================
// Scenario 7: parity for the pre-warmed-slot path. CPU + Metal must agree.
// =============================================================================

TEST(AccessList, CpuVsMetal_Parity_PreWarmedSlot)
{
    std::vector<uint8_t> code = {
        0x60, 0x07, 0x54,
        0x60, 0x00, 0x52,
        0x60, 0x20, 0x60, 0x00, 0xf3,
    };
    Transaction tx = make_tx_with_code(code);

    Config cfg;
    cfg.warm_storage_keys.resize(52);
    std::memcpy(cfg.warm_storage_keys.data(), tx.to.data(), 20);
    cfg.warm_storage_keys[20 + 31] = 0x07;

    auto r_cpu   = run_one(Backend::CPU_Sequential, tx, cfg);
    auto r_metal = run_one(Backend::GPU_Metal,      tx, cfg);
    ASSERT_EQ(r_cpu.status.size(),   1u);
    ASSERT_EQ(r_metal.status.size(), 1u);
    EXPECT_EQ(r_cpu.gas_used[0], r_metal.gas_used[0])
        << "CPU/Metal must agree on gas for pre-warmed slot";
}
