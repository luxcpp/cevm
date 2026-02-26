// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file test_opcodes.cpp
/// Per-opcode coverage test for the GPU EVM kernel.
///
/// For every Cancun-era opcode that the kernel implements, this test runs a
/// minimal program exercising that opcode and verifies (a) status, (b) gas
/// used, (c) returned bytes (when applicable). Expected values are
/// hand-computed and match the EVM specification.

#include "gpu/kernel/evm_kernel_host.hpp"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

using evm::gpu::kernel::BlockContext;
using evm::gpu::kernel::EvmKernelHost;
using evm::gpu::kernel::HostTransaction;
using evm::gpu::kernel::TxResult;
using evm::gpu::kernel::TxStatus;
using evm::gpu::kernel::uint256;

static int g_failed = 0;
static int g_passed = 0;

#define EXPECT(name, cond)                                                     \
    do                                                                         \
    {                                                                          \
        if (!(cond))                                                           \
        {                                                                      \
            std::printf("  FAIL[%s]: %s\n", (name), #cond);                    \
            ++g_failed;                                                        \
            return;                                                            \
        }                                                                     \
    } while (0)

#define PASS(name)                                                             \
    do                                                                         \
    {                                                                          \
        std::printf("  ok  : %s\n", (name));                                   \
        ++g_passed;                                                            \
    } while (0)

namespace
{

HostTransaction make_tx(std::vector<uint8_t> code, uint64_t gas_limit = 1'000'000)
{
    HostTransaction tx;
    tx.code = std::move(code);
    tx.gas_limit = gas_limit;
    return tx;
}

TxResult run(EvmKernelHost& host, std::vector<uint8_t> code,
             uint64_t gas_limit = 1'000'000)
{
    auto txs = std::vector<HostTransaction>{make_tx(std::move(code), gas_limit)};
    auto results = host.execute(txs);
    return std::move(results[0]);
}

TxResult run_ctx(EvmKernelHost& host, std::vector<uint8_t> code,
                 const BlockContext& ctx, uint64_t gas_limit = 1'000'000)
{
    auto txs = std::vector<HostTransaction>{make_tx(std::move(code), gas_limit)};
    auto results = host.execute(txs, ctx);
    return std::move(results[0]);
}

void emit_push1(std::vector<uint8_t>& code, uint8_t byte)
{
    code.push_back(0x60);
    code.push_back(byte);
}

void emit_return(std::vector<uint8_t>& code, uint8_t offset, uint8_t size)
{
    emit_push1(code, size);
    emit_push1(code, offset);
    code.push_back(0xf3);
}

bool out_eq_u8(const std::vector<uint8_t>& out, uint8_t v)
{
    return out.size() == 1 && out[0] == v;
}

bool out_eq_word(const std::vector<uint8_t>& out, uint64_t lo,
                 uint64_t hi = 0, uint64_t hh = 0, uint64_t hhh = 0)
{
    if (out.size() != 32) return false;
    auto rd = [&](size_t off)
    {
        uint64_t v = 0;
        for (size_t i = 0; i < 8; ++i) v = (v << 8) | out[off + i];
        return v;
    };
    return rd(24) == lo && rd(16) == hi && rd(8) == hh && rd(0) == hhh;
}

}  // namespace

static void test_arith(EvmKernelHost& host)
{
    {
        std::vector<uint8_t> code;
        emit_push1(code, 0x05); emit_push1(code, 0x03);
        code.push_back(0x01);
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code);
        EXPECT("ADD", out_eq_word(r.output, 8));
        PASS("0x01 ADD");
    }
    {
        std::vector<uint8_t> code;
        emit_push1(code, 0x06); emit_push1(code, 0x07);
        code.push_back(0x02);
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code);
        EXPECT("MUL", out_eq_word(r.output, 42));
        PASS("0x02 MUL");
    }
    {
        std::vector<uint8_t> code;
        emit_push1(code, 0x03); emit_push1(code, 0x0A);
        code.push_back(0x03);
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code);
        EXPECT("SUB", out_eq_word(r.output, 7));
        PASS("0x03 SUB");
    }
    {
        std::vector<uint8_t> code;
        emit_push1(code, 0x04); emit_push1(code, 0x14);
        code.push_back(0x04);
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code);
        EXPECT("DIV", out_eq_word(r.output, 5));
        PASS("0x04 DIV");
    }
    {
        std::vector<uint8_t> code;
        emit_push1(code, 0x07); emit_push1(code, 0x19);
        code.push_back(0x06);
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code);
        EXPECT("MOD", out_eq_word(r.output, 4));
        PASS("0x06 MOD");
    }
    {
        std::vector<uint8_t> code;
        emit_push1(code, 0x03); emit_push1(code, 0x05); emit_push1(code, 0x04);
        code.push_back(0x08);
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code);
        EXPECT("ADDMOD", out_eq_word(r.output, 0));
        PASS("0x08 ADDMOD");
    }
    {
        std::vector<uint8_t> code;
        emit_push1(code, 0x0d); emit_push1(code, 0x0b); emit_push1(code, 0x07);
        code.push_back(0x09);
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code);
        EXPECT("MULMOD", out_eq_word(r.output, 12));
        PASS("0x09 MULMOD");
    }
    {
        std::vector<uint8_t> code;
        emit_push1(code, 0x04); emit_push1(code, 0x03);
        code.push_back(0x0a);
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code);
        EXPECT("EXP", out_eq_word(r.output, 81));
        PASS("0x0a EXP");
    }
    {
        std::vector<uint8_t> code;
        code.push_back(0x5f);
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code);
        EXPECT("PUSH0", out_eq_word(r.output, 0));
        PASS("0x5f PUSH0");
    }
}

static void test_compare_bitwise(EvmKernelHost& host)
{
    auto sample = [&](const char* name, uint8_t opcode, uint8_t a, uint8_t b, uint64_t expected)
    {
        std::vector<uint8_t> code;
        emit_push1(code, b); emit_push1(code, a);
        code.push_back(opcode);
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code);
        EXPECT(name, out_eq_word(r.output, expected));
        PASS(name);
    };
    sample("0x10 LT", 0x10, 0x05, 0x07, 1);
    sample("0x11 GT", 0x11, 0x07, 0x05, 1);
    sample("0x14 EQ", 0x14, 0x2a, 0x2a, 1);
    sample("0x16 AND", 0x16, 0xff, 0x0f, 0x0f);
    sample("0x17 OR",  0x17, 0xf0, 0x0f, 0xff);
    sample("0x18 XOR", 0x18, 0xff, 0x0f, 0xf0);
    {
        std::vector<uint8_t> code;
        emit_push1(code, 0); code.push_back(0x15);
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code);
        EXPECT("ISZERO", out_eq_word(r.output, 1));
        PASS("0x15 ISZERO");
    }
    // BYTE: top = position (i), second = value. Pop top is `i`, replace second with byte_at(val, i).
    // sample(name, opcode, a=top_after, b=second_after, expected). With "push b; push a", a is top.
    // We want i=31, val=0x56 → kernel: a (popped) = 31, b (replaced) = byte_at(0x56, 31) = 0x56.
    sample("0x1a BYTE", 0x1a, 0x1f, 0x56, 0x56);
    // SHL: top = shift, second = value. a=4 (shift), b=1 (value) → 1<<4 = 16.
    sample("0x1b SHL",  0x1b, 0x04, 0x01, 16);
    // SHR: top = shift, second = value. a=4 (shift), b=16 (value) → 16>>4 = 1.
    sample("0x1c SHR",  0x1c, 0x04, 0x10, 1);
}

static void test_keccak(EvmKernelHost& host)
{
    {
        std::vector<uint8_t> code;
        emit_push1(code, 0x00); emit_push1(code, 0x00);
        code.push_back(0x20);
        emit_push1(code, 0x00); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code);
        EXPECT("KECCAK status", r.status == TxStatus::Return);
        const uint8_t expect[32] = {
            0xc5,0xd2,0x46,0x01,0x86,0xf7,0x23,0x3c,
            0x92,0x7e,0x7d,0xb2,0xdc,0xc7,0x03,0xc0,
            0xe5,0x00,0xb6,0x53,0xca,0x82,0x27,0x3b,
            0x7b,0xfa,0xd8,0x04,0x5d,0x85,0xa4,0x70,
        };
        EXPECT("KECCAK(empty)", std::memcmp(r.output.data(), expect, 32) == 0);
        PASS("0x20 KECCAK256 (empty)");
    }
    {
        std::vector<uint8_t> code;
        emit_push1(code, 0x61); emit_push1(code, 0x00); code.push_back(0x53);
        emit_push1(code, 0x62); emit_push1(code, 0x01); code.push_back(0x53);
        emit_push1(code, 0x63); emit_push1(code, 0x02); code.push_back(0x53);
        emit_push1(code, 0x03); emit_push1(code, 0x00);
        code.push_back(0x20);
        emit_push1(code, 0x00); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code);
        const uint8_t expect[32] = {
            0x4e,0x03,0x65,0x7a,0xea,0x45,0xa9,0x4f,
            0xc7,0xd4,0x7b,0xa8,0x26,0xc8,0xd6,0x67,
            0xc0,0xd1,0xe6,0xe3,0x3a,0x64,0xa0,0x36,
            0xec,0x44,0xf5,0x8f,0xa1,0x2d,0x6c,0x45,
        };
        EXPECT("KECCAK(abc)", std::memcmp(r.output.data(), expect, 32) == 0);
        PASS("0x20 KECCAK256 (abc)");
    }
}

static void test_block_ctx(EvmKernelHost& host)
{
    BlockContext ctx{};
    ctx.gas_price     = 0x12345678;
    ctx.timestamp     = 0xDEADBEEF;
    ctx.number        = 0x7777;
    ctx.gas_limit     = 0x9999999;
    ctx.chain_id      = 96369;
    ctx.base_fee      = 0x100;
    ctx.blob_base_fee = 0x200;
    ctx.origin.w[0]   = 0xDEADCAFEBABEBEEFULL;
    ctx.coinbase.w[0] = 0x1111222233334444ULL;

    auto check = [&](const char* name, uint8_t opcode, uint64_t expected)
    {
        std::vector<uint8_t> code;
        code.push_back(opcode);
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run_ctx(host, code, ctx);
        EXPECT(name, out_eq_word(r.output, expected));
        PASS(name);
    };
    check("0x32 ORIGIN",      0x32, 0xDEADCAFEBABEBEEFULL);
    check("0x3a GASPRICE",    0x3a, ctx.gas_price);
    check("0x41 COINBASE",    0x41, 0x1111222233334444ULL);
    check("0x42 TIMESTAMP",   0x42, ctx.timestamp);
    check("0x43 NUMBER",      0x43, ctx.number);
    check("0x45 GASLIMIT",    0x45, ctx.gas_limit);
    check("0x46 CHAINID",     0x46, ctx.chain_id);
    check("0x48 BASEFEE",     0x48, ctx.base_fee);
    check("0x4a BLOBBASEFEE", 0x4a, ctx.blob_base_fee);

    {
        std::vector<uint8_t> code;
        emit_push1(code, 0x05); code.push_back(0x40);
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run_ctx(host, code, ctx);
        EXPECT("BLOCKHASH", out_eq_word(r.output, 0));
        PASS("0x40 BLOCKHASH");
    }
    {
        std::vector<uint8_t> code;
        emit_push1(code, 0x00); code.push_back(0x49);
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run_ctx(host, code, ctx);
        EXPECT("BLOBHASH(empty)", out_eq_word(r.output, 0));
        PASS("0x49 BLOBHASH (empty)");
    }
    {
        BlockContext c2 = ctx;
        c2.num_blob_hashes = 1;
        for (int i = 0; i < 32; ++i) c2.blob_hashes[0][i] = (uint8_t)(i + 1);
        std::vector<uint8_t> code;
        emit_push1(code, 0x00); code.push_back(0x49);
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run_ctx(host, code, c2);
        EXPECT("BLOBHASH(set)",
               r.output.size() == 32 && r.output[0] == 1 && r.output[31] == 32);
        PASS("0x49 BLOBHASH (set)");
    }
}

static void test_memory(EvmKernelHost& host)
{
    {
        std::vector<uint8_t> code;
        emit_push1(code, 0x42); emit_push1(code, 0x00); code.push_back(0x52);
        emit_push1(code, 0x00); code.push_back(0x51);
        emit_push1(code, 0x20); code.push_back(0x52);
        emit_return(code, 0x20, 32);
        auto r = run(host, code);
        EXPECT("MLOAD/MSTORE", out_eq_word(r.output, 0x42));
        PASS("0x51 MLOAD + 0x52 MSTORE");
    }
    {
        std::vector<uint8_t> code;
        emit_push1(code, 0xAB); emit_push1(code, 0x00); code.push_back(0x53);
        emit_push1(code, 0x00); code.push_back(0x51);
        emit_push1(code, 0x20); code.push_back(0x52);
        emit_return(code, 0x20, 32);
        auto r = run(host, code);
        EXPECT("MSTORE8", r.output.size() == 32 && r.output[0] == 0xAB);
        PASS("0x53 MSTORE8");
    }
    {
        std::vector<uint8_t> code;
        emit_push1(code, 0x00); emit_push1(code, 0x00); code.push_back(0x52);
        code.push_back(0x59);
        emit_push1(code, 0x20); code.push_back(0x52);
        emit_return(code, 0x20, 32);
        auto r = run(host, code);
        EXPECT("MSIZE", out_eq_word(r.output, 32));
        PASS("0x59 MSIZE");
    }
    {
        std::vector<uint8_t> code;
        emit_push1(code, 0xAA); emit_push1(code, 0x00); code.push_back(0x53);
        emit_push1(code, 0x01); emit_push1(code, 0x00); emit_push1(code, 0x10);
        code.push_back(0x5e);
        emit_return(code, 0x10, 1);
        auto r = run(host, code);
        EXPECT("MCOPY", r.status == TxStatus::Return && out_eq_u8(r.output, 0xAA));
        PASS("0x5e MCOPY");
    }
}

static void test_calldata_code(EvmKernelHost& host)
{
    {
        HostTransaction tx;
        tx.code = {0x36, 0x60, 0x00, 0x52, 0x60, 0x20, 0x60, 0x00, 0xf3};
        tx.calldata = {1, 2, 3, 4, 5};
        tx.gas_limit = 1'000'000;
        auto rs = host.execute(std::vector<HostTransaction>{tx});
        EXPECT("CALLDATASIZE", out_eq_word(rs[0].output, 5));
        PASS("0x36 CALLDATASIZE");
    }
    {
        HostTransaction tx;
        tx.code = {0x60, 0x00, 0x35, 0x60, 0x00, 0x52, 0x60, 0x20, 0x60, 0x00, 0xf3};
        tx.calldata = std::vector<uint8_t>(32, 0xAB);
        tx.gas_limit = 1'000'000;
        auto rs = host.execute(std::vector<HostTransaction>{tx});
        EXPECT("CALLDATALOAD",
               rs[0].output.size() == 32 && rs[0].output[0] == 0xAB && rs[0].output[31] == 0xAB);
        PASS("0x35 CALLDATALOAD");
    }
    {
        HostTransaction tx;
        tx.code = {0x60, 0x05, 0x60, 0x00, 0x60, 0x00, 0x37,
                   0x60, 0x05, 0x60, 0x00, 0xf3};
        tx.calldata = {0x11, 0x22, 0x33, 0x44, 0x55};
        tx.gas_limit = 1'000'000;
        auto rs = host.execute(std::vector<HostTransaction>{tx});
        EXPECT("CALLDATACOPY", rs[0].output.size() == 5);
        EXPECT("CALLDATACOPY[0]", rs[0].output[0] == 0x11);
        EXPECT("CALLDATACOPY[4]", rs[0].output[4] == 0x55);
        PASS("0x37 CALLDATACOPY");
    }
    {
        std::vector<uint8_t> code = {
            0x38, 0x60, 0x00, 0x52, 0x60, 0x20, 0x60, 0x00, 0xf3
        };
        auto r = run(host, code);
        EXPECT("CODESIZE", out_eq_word(r.output, code.size()));
        PASS("0x38 CODESIZE");
    }
    {
        std::vector<uint8_t> code = {
            0x60, 0x04, 0x60, 0x00, 0x60, 0x00, 0x39,
            0x60, 0x04, 0x60, 0x00, 0xf3
        };
        auto r = run(host, code);
        EXPECT("CODECOPY", r.output.size() == 4);
        EXPECT("CODECOPY[0..1]", r.output[0] == 0x60 && r.output[1] == 0x04);
        PASS("0x39 CODECOPY");
    }
    {
        std::vector<uint8_t> code;
        code.push_back(0x3d);
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code);
        EXPECT("RETURNDATASIZE", out_eq_word(r.output, 0));
        PASS("0x3d RETURNDATASIZE");
    }
    {
        std::vector<uint8_t> code;
        emit_push1(code, 0x01); emit_push1(code, 0x00); emit_push1(code, 0x00);
        code.push_back(0x3e);
        auto r = run(host, code);
        EXPECT("RETURNDATACOPY", r.status == TxStatus::Error);
        EXPECT("RETURNDATACOPY-gas", r.gas_used == 1'000'000);
        PASS("0x3e RETURNDATACOPY (invalid)");
    }
}

static void test_state_defaults(EvmKernelHost& host)
{
    {
        std::vector<uint8_t> code;
        emit_push1(code, 0xAA); code.push_back(0x31);
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code);
        EXPECT("BALANCE", out_eq_word(r.output, 0));
        PASS("0x31 BALANCE");
    }
    {
        std::vector<uint8_t> code;
        code.push_back(0x47);
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code);
        EXPECT("SELFBALANCE", out_eq_word(r.output, 0));
        PASS("0x47 SELFBALANCE");
    }
    {
        std::vector<uint8_t> code;
        emit_push1(code, 0xAA); code.push_back(0x3b);
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code);
        EXPECT("EXTCODESIZE", out_eq_word(r.output, 0));
        PASS("0x3b EXTCODESIZE");
    }
    {
        std::vector<uint8_t> code;
        emit_push1(code, 0xAA); code.push_back(0x3f);
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code);
        const uint8_t expect[32] = {
            0xc5,0xd2,0x46,0x01,0x86,0xf7,0x23,0x3c,
            0x92,0x7e,0x7d,0xb2,0xdc,0xc7,0x03,0xc0,
            0xe5,0x00,0xb6,0x53,0xca,0x82,0x27,0x3b,
            0x7b,0xfa,0xd8,0x04,0x5d,0x85,0xa4,0x70,
        };
        EXPECT("EXTCODEHASH", r.output.size() == 32 && std::memcmp(r.output.data(), expect, 32) == 0);
        PASS("0x3f EXTCODEHASH");
    }
    {
        std::vector<uint8_t> code;
        emit_push1(code, 0x04); emit_push1(code, 0x00);
        emit_push1(code, 0x00); emit_push1(code, 0xAA);
        code.push_back(0x3c);
        emit_return(code, 0, 4);
        auto r = run(host, code);
        EXPECT("EXTCODECOPY", r.output.size() == 4 && r.output[0] == 0 && r.output[3] == 0);
        PASS("0x3c EXTCODECOPY");
    }
}

static void test_storage(EvmKernelHost& host)
{
    {
        std::vector<uint8_t> code;
        emit_push1(code, 0x07); emit_push1(code, 0x01);
        code.push_back(0x55);
        emit_push1(code, 0x01); code.push_back(0x54);
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code);
        EXPECT("SSTORE/SLOAD", out_eq_word(r.output, 7));
        PASS("0x54 SLOAD + 0x55 SSTORE");
    }
    {
        std::vector<uint8_t> code;
        emit_push1(code, 0x09); emit_push1(code, 0x02);
        code.push_back(0x5d);
        emit_push1(code, 0x02); code.push_back(0x5c);
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code);
        EXPECT("TSTORE/TLOAD", out_eq_word(r.output, 9));
        PASS("0x5c TLOAD + 0x5d TSTORE");
    }
}

static void test_logs(EvmKernelHost& host)
{
    {
        std::vector<uint8_t> code;
        emit_push1(code, 0xDE); emit_push1(code, 0x00); code.push_back(0x53);
        emit_push1(code, 0xAD); emit_push1(code, 0x01); code.push_back(0x53);
        emit_push1(code, 0xBE); emit_push1(code, 0x02); code.push_back(0x53);
        emit_push1(code, 0xEF); emit_push1(code, 0x03); code.push_back(0x53);
        emit_push1(code, 0x04); emit_push1(code, 0x00);
        code.push_back(0xa0);
        code.push_back(0x00);
        auto r = run(host, code);
        EXPECT("LOG0 status", r.status == TxStatus::Stop);
        EXPECT("LOG0 count",  r.logs.size() == 1);
        EXPECT("LOG0 size",   r.logs[0].data.size() == 4);
        EXPECT("LOG0 data",   r.logs[0].data[0] == 0xDE && r.logs[0].data[3] == 0xEF);
        EXPECT("LOG0 topics", r.logs[0].topics.empty());
        PASS("0xa0 LOG0");
    }
    {
        std::vector<uint8_t> code;
        emit_push1(code, 0x22); emit_push1(code, 0x11);
        emit_push1(code, 0x00); emit_push1(code, 0x00);
        code.push_back(0xa2);
        code.push_back(0x00);
        auto r = run(host, code);
        EXPECT("LOG2 status",  r.status == TxStatus::Stop);
        EXPECT("LOG2 count",   r.logs.size() == 1);
        EXPECT("LOG2 topics",  r.logs[0].topics.size() == 2);
        EXPECT("LOG2 t1",      r.logs[0].topics[0].w[0] == 0x11);
        EXPECT("LOG2 t2",      r.logs[0].topics[1].w[0] == 0x22);
        PASS("0xa2 LOG2");
    }
}

static void test_jumps(EvmKernelHost& host)
{
    {
        std::vector<uint8_t> code = {
            0x60, 0x05, 0x56, 0x60, 0x63,
            0x5b,
            0x60, 0x01, 0x60, 0x00, 0x52,
            0x60, 0x20, 0x60, 0x00, 0xf3
        };
        auto r = run(host, code);
        EXPECT("JUMP", r.status == TxStatus::Return && out_eq_word(r.output, 1));
        PASS("0x56 JUMP + 0x5b JUMPDEST");
    }
    {
        std::vector<uint8_t> code;
        emit_push1(code, 0x2A); code.push_back(0x80);
        code.push_back(0x01);
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code);
        EXPECT("DUP1", out_eq_word(r.output, 84));
        PASS("0x80 DUP1");
    }
    {
        std::vector<uint8_t> code;
        emit_push1(code, 0x05); emit_push1(code, 0x03);
        code.push_back(0x90); code.push_back(0x03);
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code);
        EXPECT("SWAP1", out_eq_word(r.output, 2));
        PASS("0x90 SWAP1");
    }
}

static void test_invalid(EvmKernelHost& host)
{
    {
        std::vector<uint8_t> code = {0xfe};
        auto r = run(host, code, 50000);
        EXPECT("INVALID", r.status == TxStatus::Error && r.gas_used == 50000);
        PASS("0xfe INVALID");
    }
    {
        std::vector<uint8_t> code = {0x0c};
        auto r = run(host, code, 50000);
        EXPECT("undef", r.status == TxStatus::Error && r.gas_used == 50000);
        PASS("undefined opcode 0x0c (Bug 2 fix)");
    }
    for (uint8_t op : {0xf0, 0xf1, 0xf2, 0xf4, 0xf5, 0xfa, 0xff})
    {
        std::vector<uint8_t> code = {op};
        auto r = run(host, code);
        char name[64];
        std::snprintf(name, sizeof name, "0x%02x CALL-family -> CallNotSupported", op);
        EXPECT(name, r.status == TxStatus::CallNotSupported);
        PASS(name);
    }
}

static void test_signed(EvmKernelHost& host)
{
    {
        std::vector<uint8_t> code;
        code.push_back(0x60); code.push_back(0x04);
        code.push_back(0x7f);
        for (int i = 0; i < 31; ++i) code.push_back(0xFF);
        code.push_back(0xF8);
        code.push_back(0x05);
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code);
        EXPECT("SDIV", r.output.size() == 32 && r.output[31] == 0xFE && r.output[0] == 0xFF);
        PASS("0x05 SDIV");
    }
    {
        std::vector<uint8_t> code;
        code.push_back(0x60); code.push_back(0x03);
        code.push_back(0x7f);
        for (int i = 0; i < 31; ++i) code.push_back(0xFF);
        code.push_back(0xF9);
        code.push_back(0x07);
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code);
        EXPECT("SMOD", r.output.size() == 32 && r.output[31] == 0xFF && r.output[0] == 0xFF);
        PASS("0x07 SMOD");
    }
    {
        std::vector<uint8_t> code;
        code.push_back(0x60); code.push_back(0x01);
        code.push_back(0x7f);
        for (int i = 0; i < 32; ++i) code.push_back(0xFF);
        code.push_back(0x12);
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code);
        EXPECT("SLT", out_eq_word(r.output, 1));
        PASS("0x12 SLT");
    }
    {
        std::vector<uint8_t> code;
        code.push_back(0x60); code.push_back(0x01);
        code.push_back(0x7f);
        for (int i = 0; i < 32; ++i) code.push_back(0xFF);
        code.push_back(0x13);
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code);
        EXPECT("SGT", out_eq_word(r.output, 0));
        PASS("0x13 SGT");
    }
    {
        std::vector<uint8_t> code;
        emit_push1(code, 0xFF); emit_push1(code, 0x00);
        code.push_back(0x0b);
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code);
        EXPECT("SIGNEXTEND", r.output.size() == 32 && r.output[0] == 0xFF && r.output[31] == 0xFF);
        PASS("0x0b SIGNEXTEND");
    }
    {
        std::vector<uint8_t> code;
        code.push_back(0x7f);
        for (int i = 0; i < 32; ++i) code.push_back(0xFF);
        code.push_back(0x60); code.push_back(0x04);
        code.push_back(0x1d);
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code);
        EXPECT("SAR", r.output.size() == 32 && r.output[0] == 0xFF && r.output[31] == 0xFF);
        PASS("0x1d SAR");
    }
    {
        std::vector<uint8_t> code;
        emit_push1(code, 0x07); emit_push1(code, 0x05);
        code.push_back(0x19);
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code);
        EXPECT("NOT", r.output.size() == 32 && r.output[31] == 0xFA);
        PASS("0x19 NOT");
    }
}

static void test_pc_gas(EvmKernelHost& host)
{
    {
        std::vector<uint8_t> code = {0x60, 0x00, 0x50, 0x60, 0x00, 0x50, 0x58};
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code);
        EXPECT("PC", out_eq_word(r.output, 6));
        PASS("0x58 PC");
    }
    {
        std::vector<uint8_t> code;
        code.push_back(0x5a);
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code, 12345);
        EXPECT("GAS", r.status == TxStatus::Return && r.output.size() == 32);
        PASS("0x5a GAS");
    }
    {
        std::vector<uint8_t> code;
        code.push_back(0x50);
        auto r = run(host, code, 1000);
        EXPECT("POP-empty", r.status == TxStatus::Error);
        PASS("0x50 POP (empty stack -> Error)");
    }
}

static void test_revert_stop(EvmKernelHost& host)
{
    {
        std::vector<uint8_t> code = {0x00};
        auto r = run(host, code);
        EXPECT("STOP", r.status == TxStatus::Stop && r.gas_used == 0);
        PASS("0x00 STOP");
    }
    {
        std::vector<uint8_t> code;
        emit_push1(code, 0xAB); emit_push1(code, 0x00); code.push_back(0x53);
        emit_push1(code, 0x01); emit_push1(code, 0x00); code.push_back(0xfd);
        auto r = run(host, code);
        EXPECT("REVERT", r.status == TxStatus::Revert && out_eq_u8(r.output, 0xAB));
        PASS("0xfd REVERT");
    }
    {
        std::vector<uint8_t> code;
        emit_push1(code, 0xCD); emit_push1(code, 0x00); code.push_back(0x53);
        emit_return(code, 0, 1);
        auto r = run(host, code);
        EXPECT("RETURN", r.status == TxStatus::Return && out_eq_u8(r.output, 0xCD));
        PASS("0xf3 RETURN");
    }
    {
        std::vector<uint8_t> code;
        code.push_back(0x30);
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code);
        EXPECT("ADDRESS", r.status == TxStatus::Return);
        PASS("0x30 ADDRESS");
    }
    {
        std::vector<uint8_t> code;
        code.push_back(0x33);
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code);
        EXPECT("CALLER", r.status == TxStatus::Return);
        PASS("0x33 CALLER");
    }
    {
        std::vector<uint8_t> code;
        code.push_back(0x34);
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code);
        EXPECT("CALLVALUE", r.status == TxStatus::Return);
        PASS("0x34 CALLVALUE");
    }
}

static void test_pushes(EvmKernelHost& host)
{
    for (int n = 1; n <= 32; ++n)
    {
        std::vector<uint8_t> code;
        code.push_back(0x60 + (uint8_t)(n - 1));
        for (int i = 0; i < n; ++i) code.push_back(0xAB);
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code);
        char name[32]; std::snprintf(name, sizeof name, "PUSH%d", n);
        bool ok = r.status == TxStatus::Return && r.output.size() == 32;
        if (ok)
        {
            for (int i = 0; i < 32; ++i)
            {
                bool should_be_ab = (i >= 32 - n);
                if (should_be_ab && r.output[(size_t)i] != 0xAB) { ok = false; break; }
                if (!should_be_ab && r.output[(size_t)i] != 0x00) { ok = false; break; }
            }
        }
        if (!ok) { std::printf("  FAIL[%s]\n", name); ++g_failed; return; }
        std::printf("  ok  : 0x%02x %s\n", 0x60 + (n - 1), name);
        ++g_passed;
    }
}

static void test_dup_swap(EvmKernelHost& host)
{
    for (int n = 1; n <= 16; ++n)
    {
        std::vector<uint8_t> code;
        for (int v = n; v >= 1; --v) emit_push1(code, (uint8_t)v);
        code.push_back(0x80 + (uint8_t)(n - 1));
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code);
        char name[32]; std::snprintf(name, sizeof name, "DUP%d", n);
        if (r.status != TxStatus::Return || !out_eq_word(r.output, (uint64_t)n))
        { std::printf("  FAIL[%s]\n", name); ++g_failed; return; }
        std::printf("  ok  : 0x%02x %s\n", 0x80 + (n - 1), name);
        ++g_passed;
    }
    for (int n = 1; n <= 16; ++n)
    {
        std::vector<uint8_t> code;
        emit_push1(code, 0xCC);
        for (int v = 0; v < n; ++v) emit_push1(code, 0x11);
        code.push_back(0x90 + (uint8_t)(n - 1));
        emit_push1(code, 0); code.push_back(0x52);
        emit_return(code, 0, 32);
        auto r = run(host, code);
        char name[32]; std::snprintf(name, sizeof name, "SWAP%d", n);
        if (r.status != TxStatus::Return || !out_eq_word(r.output, 0xCC))
        { std::printf("  FAIL[%s]\n", name); ++g_failed; return; }
        std::printf("  ok  : 0x%02x %s\n", 0x90 + (n - 1), name);
        ++g_passed;
    }
}

static void test_bug3(EvmKernelHost& host)
{
    {
        std::vector<uint8_t> code;
        code.push_back(0x7f);
        for (int i = 0; i < 31; ++i) code.push_back(0xFF);
        code.push_back(0xE0);
        code.push_back(0x51);
        auto r = run(host, code, 50000);
        EXPECT("Bug3 MLOAD", r.status == TxStatus::Error);
        PASS("Bug 3 (mem overflow) — MLOAD with offset = 2^256-32 errors");
    }
    {
        std::vector<uint8_t> code;
        emit_push1(code, 0x42);
        code.push_back(0x7f);
        for (int i = 0; i < 31; ++i) code.push_back(0xFF);
        code.push_back(0xE0);
        code.push_back(0x52);
        auto r = run(host, code, 50000);
        EXPECT("Bug3 MSTORE", r.status == TxStatus::Error);
        PASS("Bug 3 (mem overflow) — MSTORE with offset = 2^256-32 errors");
    }
}

int main()
{
    std::printf("============================================================\n");
    std::printf("  GPU EVM kernel — full opcode coverage test\n");
    std::printf("============================================================\n\n");

    auto host = EvmKernelHost::create();
    if (!host)
    {
        std::printf("Metal kernel host unavailable — SKIPPING ALL TESTS.\n");
        return 0;
    }
    std::printf("Device: %s\n\n", host->device_name());

    std::printf("--- arithmetic / stack ---\n");
    test_arith(*host);
    std::printf("--- comparison / bitwise ---\n");
    test_compare_bitwise(*host);
    std::printf("--- signed ops ---\n");
    test_signed(*host);
    std::printf("--- KECCAK256 ---\n");
    test_keccak(*host);
    std::printf("--- block context ---\n");
    test_block_ctx(*host);
    std::printf("--- memory ops ---\n");
    test_memory(*host);
    std::printf("--- calldata / code ---\n");
    test_calldata_code(*host);
    std::printf("--- account state defaults ---\n");
    test_state_defaults(*host);
    std::printf("--- storage / transient ---\n");
    test_storage(*host);
    std::printf("--- logs ---\n");
    test_logs(*host);
    std::printf("--- jumps / DUP / SWAP ---\n");
    test_jumps(*host);
    std::printf("--- PC / GAS / POP ---\n");
    test_pc_gas(*host);
    std::printf("--- REVERT / STOP / RETURN / ADDRESS / CALLER / CALLVALUE ---\n");
    test_revert_stop(*host);
    std::printf("--- INVALID / unknown / CALL family ---\n");
    test_invalid(*host);
    std::printf("--- PUSH1..PUSH32 (all 32) ---\n");
    test_pushes(*host);
    std::printf("--- DUP1..DUP16, SWAP1..SWAP16 (all 32) ---\n");
    test_dup_swap(*host);
    std::printf("--- Bug 3 (memory offset overflow) ---\n");
    test_bug3(*host);

    std::printf("\n============================================================\n");
    std::printf(" passed: %d   failed: %d\n", g_passed, g_failed);
    std::printf("============================================================\n");
    return g_failed == 0 ? 0 : 1;
}
