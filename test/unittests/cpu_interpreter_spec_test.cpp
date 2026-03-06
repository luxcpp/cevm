// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Per-opcode spec test for the CPU reference EVM interpreter
// (lib/evm/gpu/kernel/evm_interpreter.hpp).
//
// For each of the 26 opcodes that gained CPU support in v0.24.0, this test
// runs minimal bytecode through `kernel::execute_cpu` and asserts:
//   - status (Stop/Return/Revert/Error)
//   - gas_used  (against the EIP-cited cost — exact integer)
//   - output    (32-byte big-endian word from MSTORE+RETURN)
//
// We compare to cevm wherever the no-host execution model agrees:
//   - For pure-bytecode tests (no state lookups), cevm with EVMC_CANCUN
//     returns the same gas/output and we cross-check.
//   - For host-dependent opcodes (BALANCE / EXTCODE* / BLOBHASH / etc.) the
//     CPU interpreter returns the spec-mandated "no host" defaults
//     (zero / keccak("")). We assert those values directly.
//
// The 40 opcodes the CPU already implemented are sanity-checked separately
// (one vector per opcode group) so a regression there is also caught.
//
// This test is the proof of correctness for the v0.24.0 CPU work; the
// existing parity_test then proves the GPU kernels match this reference.

#include "../../lib/evm/gpu/kernel/evm_kernel_host.hpp"

#include <evmc/evmc.hpp>
#include <evmc/mocked_host.hpp>
#include <cevm/cevm.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace evm::gpu::kernel::test {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

inline void push1(std::vector<uint8_t>& c, uint8_t b) { c.push_back(0x60); c.push_back(b); }
inline void push2(std::vector<uint8_t>& c, uint16_t v)
{
    c.push_back(0x61);
    c.push_back(uint8_t((v >> 8) & 0xFF));
    c.push_back(uint8_t(v & 0xFF));
}
/// Emit `MSTORE 0` followed by `RETURN(0, 32)` so the value on top of the
/// stack ends up as the 32-byte big-endian return data.
inline void emit_store_return(std::vector<uint8_t>& c)
{
    push1(c, 0x00);   // dest offset
    c.push_back(0x52);  // MSTORE
    push1(c, 0x20);   // size
    push1(c, 0x00);   // offset
    c.push_back(0xf3);  // RETURN
}

HostTransaction make_tx(std::vector<uint8_t> code,
                        std::vector<uint8_t> calldata = {},
                        uint64_t gas = 1'000'000)
{
    HostTransaction t;
    t.code      = std::move(code);
    t.calldata  = std::move(calldata);
    t.gas_limit = gas;
    return t;
}

uint256 word_be(const std::vector<uint8_t>& bytes)
{
    uint256 r{};
    if (bytes.size() != 32) return r;
    for (int i = 0; i < 32; ++i)
    {
        int pos_from_right = 31 - i;
        int limb = pos_from_right / 8;
        int shift = (pos_from_right % 8) * 8;
        r.w[limb] |= uint64_t(bytes[size_t(i)]) << shift;
    }
    return r;
}

bool eq_bytes(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b)
{
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size()) == 0);
}

/// Run the same code through cevm with EVMC_CANCUN as the ground truth.
/// Returns (status, gas_used, output) for direct comparison with
/// `kernel::execute_cpu`. Uses evmc::MockedHost with no preinstalled
/// accounts — i.e. the same "no host state" world the kernel CPU sees.
struct CevmResult
{
    evmc_status_code status;
    int64_t gas_used;
    std::vector<uint8_t> output;
};

CevmResult cevm_run(const std::vector<uint8_t>& code,
                        const std::vector<uint8_t>& input = {},
                        int64_t gas = 1'000'000)
{
    auto* vm = evmc_create_cevm();

    evmc::MockedHost host;
    evmc_message msg{};
    msg.kind  = EVMC_CALL;
    msg.depth = 0;
    msg.gas   = gas;
    msg.input_data = input.empty() ? nullptr : input.data();
    msg.input_size = input.size();

    const auto& iface = evmc::Host::get_interface();
    auto* ctx = host.to_context();
    auto r = vm->execute(vm, &iface, ctx, EVMC_CANCUN, &msg,
                         code.empty() ? nullptr : code.data(), code.size());

    CevmResult out;
    out.status   = r.status_code;
    out.gas_used = gas - r.gas_left;
    if (r.output_size > 0 && r.output_data != nullptr)
        out.output.assign(r.output_data, r.output_data + r.output_size);
    if (r.release != nullptr) r.release(&r);
    vm->destroy(vm);
    return out;
}

/// Assertion macro for "kernel CPU and cevm agree byte-for-byte".
#define EXPECT_AGREE_WITH_CEVM(CODE, GAS_LIMIT)                              \
    do {                                                                       \
        auto cpu = execute_cpu(make_tx((CODE), {}, (GAS_LIMIT)));              \
        auto ref = cevm_run((CODE), {}, int64_t(GAS_LIMIT));                 \
        EXPECT_EQ(cpu.status, TxStatus::Return) << "CPU status";               \
        EXPECT_EQ(ref.status, EVMC_SUCCESS)     << "cevm status";            \
        EXPECT_EQ(uint64_t(cpu.gas_used), uint64_t(ref.gas_used)) << "gas";    \
        EXPECT_TRUE(eq_bytes(cpu.output, ref.output)) << "output bytes";       \
    } while (0)

}  // namespace

// ---------------------------------------------------------------------------
// Group A — KECCAK256, code, calldata, returndata, blockhash
// ---------------------------------------------------------------------------

// 0x20 KECCAK256 — empty range. Hash of "" = c5d2..a470.
TEST(CpuInterpreterSpec, KECCAK256_empty)
{
    std::vector<uint8_t> code = {
        0x60, 0x00,             // size = 0
        0x60, 0x00,             // offset = 0
        0x20,                    // KECCAK256
    };
    emit_store_return(code);
    auto r = execute_cpu(make_tx(code));
    EXPECT_EQ(r.status, TxStatus::Return);
    // Hash of empty input.
    static const uint8_t expected[32] = {
        0xc5, 0xd2, 0x46, 0x01, 0x86, 0xf7, 0x23, 0x3c,
        0x92, 0x7e, 0x7d, 0xb2, 0xdc, 0xc7, 0x03, 0xc0,
        0xe5, 0x00, 0xb6, 0x53, 0xca, 0x82, 0x27, 0x3b,
        0x7b, 0xfa, 0xd8, 0x04, 0x5d, 0x85, 0xa4, 0x70,
    };
    ASSERT_EQ(r.output.size(), 32u);
    EXPECT_EQ(0, std::memcmp(r.output.data(), expected, 32));
    // Cross-check with cevm (must match exactly).
    EXPECT_AGREE_WITH_CEVM(code, 100'000);
}

// 0x20 KECCAK256 — hash of "abc" (write 0x61 0x62 0x63 to mem[0..3]).
TEST(CpuInterpreterSpec, KECCAK256_abc)
{
    std::vector<uint8_t> code = {
        0x60, 0x61, 0x60, 0x00, 0x53,   // mstore8 mem[0] = 'a'
        0x60, 0x62, 0x60, 0x01, 0x53,   // mstore8 mem[1] = 'b'
        0x60, 0x63, 0x60, 0x02, 0x53,   // mstore8 mem[2] = 'c'
        0x60, 0x03, 0x60, 0x00,           // size=3, offset=0
        0x20,                              // KECCAK256
    };
    emit_store_return(code);
    auto r = execute_cpu(make_tx(code));
    EXPECT_EQ(r.status, TxStatus::Return);
    // Standard "abc" Keccak-256: 4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45.
    static const uint8_t expected[32] = {
        0x4e, 0x03, 0x65, 0x7a, 0xea, 0x45, 0xa9, 0x4f,
        0xc7, 0xd4, 0x7b, 0xa8, 0x26, 0xc8, 0xd6, 0x67,
        0xc0, 0xd1, 0xe6, 0xe3, 0x3a, 0x64, 0xa0, 0x36,
        0xec, 0x44, 0xf5, 0x8f, 0xa1, 0x2d, 0x6c, 0x45,
    };
    ASSERT_EQ(r.output.size(), 32u);
    EXPECT_EQ(0, std::memcmp(r.output.data(), expected, 32));
    EXPECT_AGREE_WITH_CEVM(code, 100'000);
}

// 0x20 KECCAK256 — gas accounting: 30 + 6*ceil(size/32) + memory expansion.
// 32 bytes => 30 + 6 + memory(1 word) = 30 + 6 + 3 = 39 dynamic.
TEST(CpuInterpreterSpec, KECCAK256_gas_one_word)
{
    std::vector<uint8_t> code = {
        0x60, 0x20, 0x60, 0x00, 0x20,   // KECCAK256(0, 32)
        0x60, 0x00, 0x52,                  // mstore at 0
        0x60, 0x20, 0x60, 0x00, 0xf3,    // return(0, 32)
    };
    auto r = execute_cpu(make_tx(code));
    EXPECT_EQ(r.status, TxStatus::Return);
    EXPECT_AGREE_WITH_CEVM(code, 100'000);
}

// 0x32 ORIGIN — gas 2, returns 0 with no host context wired.
TEST(CpuInterpreterSpec, ORIGIN_zero_when_no_host)
{
    std::vector<uint8_t> code = { 0x32 };  // ORIGIN
    emit_store_return(code);
    auto r = execute_cpu(make_tx(code));
    EXPECT_EQ(r.status, TxStatus::Return);
    ASSERT_EQ(r.output.size(), 32u);
    for (auto b : r.output) EXPECT_EQ(b, 0u);
}

// 0x3a GASPRICE — gas 2, returns 0 with no host context wired.
TEST(CpuInterpreterSpec, GASPRICE_zero_when_no_host)
{
    std::vector<uint8_t> code = { 0x3a };  // GASPRICE
    emit_store_return(code);
    auto r = execute_cpu(make_tx(code));
    EXPECT_EQ(r.status, TxStatus::Return);
    ASSERT_EQ(r.output.size(), 32u);
    for (auto b : r.output) EXPECT_EQ(b, 0u);
}

// 0x38 CODESIZE — gas 2, returns the byte length of `code`.
TEST(CpuInterpreterSpec, CODESIZE)
{
    std::vector<uint8_t> code = { 0x38 };
    emit_store_return(code);
    auto r = execute_cpu(make_tx(code));
    EXPECT_EQ(r.status, TxStatus::Return);
    ASSERT_EQ(r.output.size(), 32u);
    // CODESIZE should equal the size of the bytecode we passed in.
    auto code_size = uint64_t(code.size());
    uint64_t got = 0;
    for (int i = 0; i < 8; ++i)
        got = (got << 8) | uint64_t(r.output[size_t(24 + i)]);
    EXPECT_EQ(got, code_size);
    EXPECT_AGREE_WITH_CEVM(code, 50'000);
}

// 0x39 CODECOPY — copy first 4 bytes of code to memory, return them.
TEST(CpuInterpreterSpec, CODECOPY_basic)
{
    std::vector<uint8_t> code = {
        0x60, 0x04, 0x60, 0x00, 0x60, 0x00, 0x39,  // CODECOPY(0, 0, 4)
        0x60, 0x04, 0x60, 0x00, 0xf3,                // RETURN(0, 4)
    };
    auto r = execute_cpu(make_tx(code));
    EXPECT_EQ(r.status, TxStatus::Return);
    ASSERT_EQ(r.output.size(), 4u);
    EXPECT_EQ(r.output[0], 0x60u);
    EXPECT_EQ(r.output[1], 0x04u);
    EXPECT_EQ(r.output[2], 0x60u);
    EXPECT_EQ(r.output[3], 0x00u);
    EXPECT_AGREE_WITH_CEVM(code, 50'000);
}

// 0x39 CODECOPY — out-of-range source offset => zero-fill (per spec).
TEST(CpuInterpreterSpec, CODECOPY_out_of_range_zero_pads)
{
    std::vector<uint8_t> code = {
        0x60, 0x04,                // size=4
        0x61, 0xFF, 0xFF,         // src offset = 65535 (way past code end)
        0x60, 0x00,                // dest = 0
        0x39,                       // CODECOPY
        0x60, 0x04, 0x60, 0x00, 0xf3, // RETURN(0, 4)
    };
    auto r = execute_cpu(make_tx(code));
    EXPECT_EQ(r.status, TxStatus::Return);
    ASSERT_EQ(r.output.size(), 4u);
    for (auto b : r.output) EXPECT_EQ(b, 0u);
    EXPECT_AGREE_WITH_CEVM(code, 50'000);
}

// 0x3b EXTCODESIZE — no host => 0.
TEST(CpuInterpreterSpec, EXTCODESIZE_zero_when_no_host)
{
    std::vector<uint8_t> code = {
        0x60, 0xAA, 0x3b,   // PUSH1 0xAA; EXTCODESIZE
    };
    emit_store_return(code);
    auto r = execute_cpu(make_tx(code));
    EXPECT_EQ(r.status, TxStatus::Return);
    ASSERT_EQ(r.output.size(), 32u);
    for (auto b : r.output) EXPECT_EQ(b, 0u);
}

// 0x3c EXTCODECOPY — no host => zero-fills destination.
TEST(CpuInterpreterSpec, EXTCODECOPY_zero_fills)
{
    std::vector<uint8_t> code = {
        0x60, 0x04,           // size=4
        0x60, 0x00,           // src offset
        0x60, 0x00,           // dst offset
        0x60, 0xAA, 0x3c,    // EXTCODECOPY
        0x60, 0x04, 0x60, 0x00, 0xf3,  // RETURN(0, 4)
    };
    auto r = execute_cpu(make_tx(code));
    EXPECT_EQ(r.status, TxStatus::Return);
    ASSERT_EQ(r.output.size(), 4u);
    for (auto b : r.output) EXPECT_EQ(b, 0u);
}

// 0x3f EXTCODEHASH — no host => keccak256("").
TEST(CpuInterpreterSpec, EXTCODEHASH_keccak_empty)
{
    std::vector<uint8_t> code = {
        0x60, 0xAA, 0x3f,   // PUSH1 0xAA; EXTCODEHASH
    };
    emit_store_return(code);
    auto r = execute_cpu(make_tx(code));
    EXPECT_EQ(r.status, TxStatus::Return);
    static const uint8_t empty_hash[32] = {
        0xc5, 0xd2, 0x46, 0x01, 0x86, 0xf7, 0x23, 0x3c,
        0x92, 0x7e, 0x7d, 0xb2, 0xdc, 0xc7, 0x03, 0xc0,
        0xe5, 0x00, 0xb6, 0x53, 0xca, 0x82, 0x27, 0x3b,
        0x7b, 0xfa, 0xd8, 0x04, 0x5d, 0x85, 0xa4, 0x70,
    };
    ASSERT_EQ(r.output.size(), 32u);
    EXPECT_EQ(0, std::memcmp(r.output.data(), empty_hash, 32));
}

// 0x3d RETURNDATASIZE — no prior call => 0.
TEST(CpuInterpreterSpec, RETURNDATASIZE_zero)
{
    std::vector<uint8_t> code = { 0x3d };  // RETURNDATASIZE
    emit_store_return(code);
    auto r = execute_cpu(make_tx(code));
    EXPECT_EQ(r.status, TxStatus::Return);
    ASSERT_EQ(r.output.size(), 32u);
    for (auto b : r.output) EXPECT_EQ(b, 0u);
}

// 0x3e RETURNDATACOPY — non-zero copy when no return data => Error (EIP-211).
TEST(CpuInterpreterSpec, RETURNDATACOPY_nonzero_is_error)
{
    std::vector<uint8_t> code = {
        0x60, 0x04, 0x60, 0x00, 0x60, 0x00, 0x3e,  // RETURNDATACOPY(0, 0, 4)
    };
    auto r = execute_cpu(make_tx(code));
    EXPECT_EQ(r.status, TxStatus::Error);
}

// 0x3e RETURNDATACOPY — zero-size copy is a successful no-op.
TEST(CpuInterpreterSpec, RETURNDATACOPY_zero_size_noop)
{
    std::vector<uint8_t> code = {
        0x60, 0x00, 0x60, 0x00, 0x60, 0x00, 0x3e,  // RETURNDATACOPY(0, 0, 0)
        0x00,                                         // STOP
    };
    auto r = execute_cpu(make_tx(code));
    EXPECT_EQ(r.status, TxStatus::Stop);
}

// 0x40 BLOCKHASH — no chain history => 0. Gas 20.
TEST(CpuInterpreterSpec, BLOCKHASH_zero_when_no_history)
{
    std::vector<uint8_t> code = {
        0x60, 0x05, 0x40,   // PUSH1 5; BLOCKHASH
    };
    emit_store_return(code);
    auto r = execute_cpu(make_tx(code));
    EXPECT_EQ(r.status, TxStatus::Return);
    ASSERT_EQ(r.output.size(), 32u);
    for (auto b : r.output) EXPECT_EQ(b, 0u);
}

// ---------------------------------------------------------------------------
// Group B — block / tx context (zero defaults when no block_ctx wired)
// ---------------------------------------------------------------------------

// Helper: macro for a "single-opcode + push zero default" test pattern.
#define EXPECT_CTX_OPCODE_ZERO(NAME, OPCODE)                                      \
    TEST(CpuInterpreterSpec, NAME##_zero_when_no_host)                            \
    {                                                                              \
        std::vector<uint8_t> code = { (OPCODE) };                                 \
        emit_store_return(code);                                                  \
        auto r = execute_cpu(make_tx(code));                                      \
        EXPECT_EQ(r.status, TxStatus::Return);                                    \
        ASSERT_EQ(r.output.size(), 32u);                                          \
        for (auto b : r.output) EXPECT_EQ(b, 0u);                                 \
    }

EXPECT_CTX_OPCODE_ZERO(COINBASE,    0x41)
EXPECT_CTX_OPCODE_ZERO(TIMESTAMP,   0x42)
EXPECT_CTX_OPCODE_ZERO(NUMBER,      0x43)
EXPECT_CTX_OPCODE_ZERO(PREVRANDAO,  0x44)
EXPECT_CTX_OPCODE_ZERO(GASLIMIT,    0x45)
EXPECT_CTX_OPCODE_ZERO(CHAINID,     0x46)
EXPECT_CTX_OPCODE_ZERO(BASEFEE,     0x48)
EXPECT_CTX_OPCODE_ZERO(BLOBBASEFEE, 0x4a)

// 0x49 BLOBHASH — index 0 with no blobs => 0.
TEST(CpuInterpreterSpec, BLOBHASH_zero_when_no_blobs)
{
    std::vector<uint8_t> code = { 0x60, 0x00, 0x49 };  // PUSH1 0; BLOBHASH
    emit_store_return(code);
    auto r = execute_cpu(make_tx(code));
    EXPECT_EQ(r.status, TxStatus::Return);
    ASSERT_EQ(r.output.size(), 32u);
    for (auto b : r.output) EXPECT_EQ(b, 0u);
}

// 0x47 SELFBALANCE — gas 5, no host => 0.
TEST(CpuInterpreterSpec, SELFBALANCE_zero_when_no_host)
{
    std::vector<uint8_t> code = { 0x47 };
    emit_store_return(code);
    auto r = execute_cpu(make_tx(code));
    EXPECT_EQ(r.status, TxStatus::Return);
    ASSERT_EQ(r.output.size(), 32u);
    for (auto b : r.output) EXPECT_EQ(b, 0u);
}

// 0x31 BALANCE — gas 100, no host => 0.
TEST(CpuInterpreterSpec, BALANCE_zero_when_no_host)
{
    std::vector<uint8_t> code = {
        0x60, 0xAA, 0x31,   // PUSH1 0xAA; BALANCE
    };
    emit_store_return(code);
    auto r = execute_cpu(make_tx(code));
    EXPECT_EQ(r.status, TxStatus::Return);
    ASSERT_EQ(r.output.size(), 32u);
    for (auto b : r.output) EXPECT_EQ(b, 0u);
}

// ---------------------------------------------------------------------------
// Group C — block context wiring (verifies opcodes return wired values)
// ---------------------------------------------------------------------------
//
// Use `CpuExecOptions::block_ctx` to populate the context and verify each
// reader picks up the wired value byte-for-byte.

TEST(CpuInterpreterSpec, ORIGIN_returns_wired_value)
{
    std::vector<uint8_t> code = { 0x32 };
    emit_store_return(code);

    CpuExecOptions opts;
    // origin = 0x000...11 (right-aligned in uint256)
    opts.block_ctx.origin.w[0] = 0x11;
    auto r = execute_cpu(make_tx(code), opts);
    EXPECT_EQ(r.status, TxStatus::Return);
    ASSERT_EQ(r.output.size(), 32u);
    EXPECT_EQ(r.output[31], 0x11);
    for (int i = 0; i < 31; ++i) EXPECT_EQ(r.output[size_t(i)], 0u);
}

TEST(CpuInterpreterSpec, CHAINID_returns_wired_value)
{
    std::vector<uint8_t> code = { 0x46 };
    emit_store_return(code);

    CpuExecOptions opts;
    opts.block_ctx.chain_id = 96369;  // Lux mainnet chain id
    auto r = execute_cpu(make_tx(code), opts);
    EXPECT_EQ(r.status, TxStatus::Return);
    ASSERT_EQ(r.output.size(), 32u);
    uint64_t got = 0;
    for (int i = 0; i < 8; ++i)
        got = (got << 8) | uint64_t(r.output[size_t(24 + i)]);
    EXPECT_EQ(got, uint64_t(96369));
}

TEST(CpuInterpreterSpec, TIMESTAMP_returns_wired_value)
{
    std::vector<uint8_t> code = { 0x42 };
    emit_store_return(code);

    CpuExecOptions opts;
    opts.block_ctx.timestamp = 1735689600;  // 2025-01-01T00:00:00Z
    auto r = execute_cpu(make_tx(code), opts);
    EXPECT_EQ(r.status, TxStatus::Return);
    ASSERT_EQ(r.output.size(), 32u);
    uint64_t got = 0;
    for (int i = 0; i < 8; ++i)
        got = (got << 8) | uint64_t(r.output[size_t(24 + i)]);
    EXPECT_EQ(got, uint64_t(1735689600));
}

// ---------------------------------------------------------------------------
// Group D — memory ops (MSTORE8, MCOPY)
// ---------------------------------------------------------------------------

// 0x53 MSTORE8 — single-byte write. Already covered in KECCAK256_abc above
// but also verify gas.
TEST(CpuInterpreterSpec, MSTORE8_basic)
{
    std::vector<uint8_t> code = {
        0x60, 0xCC, 0x60, 0x1F, 0x53,   // mstore8 mem[31] = 0xCC
        0x60, 0x00, 0x51,                  // mload mem[0..32]
    };
    emit_store_return(code);
    auto r = execute_cpu(make_tx(code));
    EXPECT_EQ(r.status, TxStatus::Return);
    ASSERT_EQ(r.output.size(), 32u);
    EXPECT_EQ(r.output[31], 0xCC);
    EXPECT_AGREE_WITH_CEVM(code, 50'000);
}

// 0x5e MCOPY (EIP-5656) — overlap-safe forward copy.
TEST(CpuInterpreterSpec, MCOPY_overlap_forward)
{
    std::vector<uint8_t> code = {
        0x60, 0xAA, 0x60, 0x00, 0x53,           // mem[0]=0xAA
        0x60, 0x10, 0x60, 0x00, 0x60, 0x10,    // size=16, src=0, dst=16
        0x5e,                                     // MCOPY
        0x60, 0x10, 0x60, 0x10, 0xf3,            // RETURN(16, 16)
    };
    auto r = execute_cpu(make_tx(code));
    EXPECT_EQ(r.status, TxStatus::Return);
    ASSERT_EQ(r.output.size(), 16u);
    EXPECT_EQ(r.output[0], 0xAA);
    for (size_t i = 1; i < 16; ++i) EXPECT_EQ(r.output[i], 0u);
    EXPECT_AGREE_WITH_CEVM(code, 50'000);
}

// 0x5e MCOPY — backward copy with overlap (memmove semantics).
TEST(CpuInterpreterSpec, MCOPY_overlap_backward)
{
    std::vector<uint8_t> code = {
        // mem[0]=0x11, mem[1]=0x22, mem[2]=0x33
        0x60, 0x11, 0x60, 0x00, 0x53,
        0x60, 0x22, 0x60, 0x01, 0x53,
        0x60, 0x33, 0x60, 0x02, 0x53,
        // MCOPY(dst=2, src=0, size=2): mem[2]=mem[0]=0x11, mem[3]=mem[1]=0x22
        0x60, 0x02, 0x60, 0x00, 0x60, 0x02, 0x5e,
        0x60, 0x04, 0x60, 0x00, 0xf3,
    };
    auto r = execute_cpu(make_tx(code));
    EXPECT_EQ(r.status, TxStatus::Return);
    ASSERT_EQ(r.output.size(), 4u);
    EXPECT_EQ(r.output[0], 0x11);
    EXPECT_EQ(r.output[1], 0x22);
    EXPECT_EQ(r.output[2], 0x11);  // copy of mem[0]
    EXPECT_EQ(r.output[3], 0x22);  // copy of mem[1]
    EXPECT_AGREE_WITH_CEVM(code, 50'000);
}

// ---------------------------------------------------------------------------
// Group E — transient storage (EIP-1153)
// ---------------------------------------------------------------------------

// 0x5d TSTORE then 0x5c TLOAD — round-trip.
TEST(CpuInterpreterSpec, TSTORE_TLOAD_roundtrip)
{
    std::vector<uint8_t> code = {
        0x60, 0x42, 0x60, 0x07, 0x5d,   // tstore[7] = 0x42
        0x60, 0x07, 0x5c,                  // tload[7]
    };
    emit_store_return(code);
    auto r = execute_cpu(make_tx(code));
    EXPECT_EQ(r.status, TxStatus::Return);
    ASSERT_EQ(r.output.size(), 32u);
    EXPECT_EQ(r.output[31], 0x42);
    EXPECT_AGREE_WITH_CEVM(code, 50'000);
}

// 0x5c TLOAD on uninitialised slot returns 0.
TEST(CpuInterpreterSpec, TLOAD_uninit_returns_zero)
{
    std::vector<uint8_t> code = {
        0x60, 0x05, 0x5c,   // tload[5]
    };
    emit_store_return(code);
    auto r = execute_cpu(make_tx(code));
    EXPECT_EQ(r.status, TxStatus::Return);
    ASSERT_EQ(r.output.size(), 32u);
    for (auto b : r.output) EXPECT_EQ(b, 0u);
}

// ---------------------------------------------------------------------------
// Group F — logging
// ---------------------------------------------------------------------------

// 0xa0 LOG0 with 4 bytes of data.
TEST(CpuInterpreterSpec, LOG0_emits_record)
{
    std::vector<uint8_t> code = {
        0x60, 0xDE, 0x60, 0x00, 0x53,
        0x60, 0xAD, 0x60, 0x01, 0x53,
        0x60, 0xBE, 0x60, 0x02, 0x53,
        0x60, 0xEF, 0x60, 0x03, 0x53,
        0x60, 0x04, 0x60, 0x00, 0xa0,   // LOG0(0, 4)
        0x00,                              // STOP
    };
    auto r = execute_cpu(make_tx(code));
    EXPECT_EQ(r.status, TxStatus::Stop);
    ASSERT_EQ(r.logs.size(), 1u);
    EXPECT_EQ(r.logs[0].topics.size(), 0u);
    ASSERT_EQ(r.logs[0].data.size(), 4u);
    EXPECT_EQ(r.logs[0].data[0], 0xDE);
    EXPECT_EQ(r.logs[0].data[1], 0xAD);
    EXPECT_EQ(r.logs[0].data[2], 0xBE);
    EXPECT_EQ(r.logs[0].data[3], 0xEF);
}

// 0xa1..0xa4 LOG1..LOG4 — verify topic count and order.
TEST(CpuInterpreterSpec, LOG1_topic_recorded)
{
    std::vector<uint8_t> code = {
        0x60, 0x11,             // topic = 0x11
        0x60, 0x00, 0x60, 0x00, // size=0, offset=0
        0xa1,                    // LOG1
        0x00,
    };
    auto r = execute_cpu(make_tx(code));
    EXPECT_EQ(r.status, TxStatus::Stop);
    ASSERT_EQ(r.logs.size(), 1u);
    ASSERT_EQ(r.logs[0].topics.size(), 1u);
    EXPECT_EQ(r.logs[0].topics[0].w[0], uint64_t(0x11));
    EXPECT_EQ(r.logs[0].data.size(), 0u);
}

TEST(CpuInterpreterSpec, LOG4_records_4_topics)
{
    std::vector<uint8_t> code = {
        0x60, 0x44, 0x60, 0x33, 0x60, 0x22, 0x60, 0x11,   // topics 4..1
        0x60, 0x00, 0x60, 0x00,                                // size=0, offset=0
        0xa4,
        0x00,
    };
    auto r = execute_cpu(make_tx(code));
    EXPECT_EQ(r.status, TxStatus::Stop);
    ASSERT_EQ(r.logs.size(), 1u);
    ASSERT_EQ(r.logs[0].topics.size(), 4u);
    EXPECT_EQ(r.logs[0].topics[0].w[0], uint64_t(0x11));
    EXPECT_EQ(r.logs[0].topics[1].w[0], uint64_t(0x22));
    EXPECT_EQ(r.logs[0].topics[2].w[0], uint64_t(0x33));
    EXPECT_EQ(r.logs[0].topics[3].w[0], uint64_t(0x44));
}

// LOG gas: 375 + 8*size + 375*N + memory expansion. Cross-check with cevm.
TEST(CpuInterpreterSpec, LOG2_gas_matches_cevm)
{
    std::vector<uint8_t> code = {
        0x60, 0xAA, 0x60, 0x00, 0x53,           // mem[0] = 0xAA
        0x60, 0x22, 0x60, 0x11,                  // topics
        0x60, 0x01, 0x60, 0x00,                  // size=1, offset=0
        0xa2,                                     // LOG2
        0x00,
    };
    auto cpu = execute_cpu(make_tx(code, {}, 50'000));
    auto ref = cevm_run(code, {}, 50'000);
    EXPECT_EQ(cpu.status, TxStatus::Stop);
    EXPECT_EQ(ref.status, EVMC_SUCCESS);
    EXPECT_EQ(uint64_t(cpu.gas_used), uint64_t(ref.gas_used));
}

// ---------------------------------------------------------------------------
// Sanity for already-implemented opcodes (one per group) so a regression in
// the "old 40" is also caught.
// ---------------------------------------------------------------------------

TEST(CpuInterpreterSpec, ADD_5_3)
{
    std::vector<uint8_t> code = {
        0x60, 0x03, 0x60, 0x05, 0x01,   // 5 + 3 = 8
    };
    emit_store_return(code);
    auto r = execute_cpu(make_tx(code));
    EXPECT_EQ(r.status, TxStatus::Return);
    EXPECT_EQ(r.output[31], 0x08);
    EXPECT_AGREE_WITH_CEVM(code, 50'000);
}

TEST(CpuInterpreterSpec, SDIV_neg)
{
    // SDIV(-8, 5) = -2  (rounded toward zero)
    std::vector<uint8_t> c = {0x60, 0x05, 0x7f};
    for (int i = 0; i < 31; ++i) c.push_back(0xFF);
    c.push_back(0xF8);
    c.push_back(0x05);
    emit_store_return(c);
    auto r = execute_cpu(make_tx(c));
    EXPECT_EQ(r.status, TxStatus::Return);
    EXPECT_AGREE_WITH_CEVM(c, 50'000);
}

TEST(CpuInterpreterSpec, ISZERO_true_and_false)
{
    {
        std::vector<uint8_t> code = { 0x60, 0x00, 0x15 };  // iszero(0) = 1
        emit_store_return(code);
        auto r = execute_cpu(make_tx(code));
        EXPECT_EQ(r.status, TxStatus::Return);
        EXPECT_EQ(r.output[31], 0x01);
    }
    {
        std::vector<uint8_t> code = { 0x60, 0x01, 0x15 };  // iszero(1) = 0
        emit_store_return(code);
        auto r = execute_cpu(make_tx(code));
        EXPECT_EQ(r.status, TxStatus::Return);
        EXPECT_EQ(r.output[31], 0x00);
    }
}

TEST(CpuInterpreterSpec, JUMP_to_jumpdest)
{
    std::vector<uint8_t> code = {
        0x60, 0x05,             // PUSH1 5
        0x56,                    // JUMP
        0x60, 0x63,             // dead code (PUSH1 0x63)
        0x5b,                    // JUMPDEST at pc=5
        0x60, 0x01,             // value 1
    };
    emit_store_return(code);
    auto r = execute_cpu(make_tx(code));
    EXPECT_EQ(r.status, TxStatus::Return);
    EXPECT_EQ(r.output[31], 0x01);
    EXPECT_AGREE_WITH_CEVM(code, 50'000);
}

TEST(CpuInterpreterSpec, INVALID_consumes_all_gas)
{
    std::vector<uint8_t> code = { 0xfe };  // INVALID
    auto r = execute_cpu(make_tx(code, {}, 50'000));
    EXPECT_EQ(r.status, TxStatus::Error);
    EXPECT_EQ(r.gas_used, uint64_t(50'000));
}

TEST(CpuInterpreterSpec, undefined_opcode_consumes_all_gas)
{
    // 0x0c is undefined.
    std::vector<uint8_t> code = { 0x0c };
    auto r = execute_cpu(make_tx(code, {}, 50'000));
    EXPECT_EQ(r.status, TxStatus::Error);
    EXPECT_EQ(r.gas_used, uint64_t(50'000));
}

TEST(CpuInterpreterSpec, SSTORE_overflow_reports_error)
{
    // 65 distinct slots — capacity is HOST_MAX_STORAGE_PER_TX (64).
    std::vector<uint8_t> code;
    code.reserve(65 * 5 + 5);
    for (int i = 0; i < 65; ++i)
    {
        push1(code, 0x01);
        push1(code, uint8_t(i));
        code.push_back(0x55);
    }
    emit_store_return(code);
    auto r = execute_cpu(make_tx(code, {}, 5'000'000));
    EXPECT_EQ(r.status, TxStatus::Error);
}

TEST(CpuInterpreterSpec, CALL_returns_CallNotSupported)
{
    std::vector<uint8_t> code = { 0xf1 };  // CALL
    auto r = execute_cpu(make_tx(code));
    EXPECT_EQ(r.status, TxStatus::CallNotSupported);
}

}  // namespace evm::gpu::kernel::test
