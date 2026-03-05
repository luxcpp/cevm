// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// 4-way parity test: every EVM opcode the kernel implements must produce
// identical (gas_used, status, output) results across:
//
//   1. Backend::CPU_Sequential  — kernel CPU interpreter, single-threaded
//   2. Backend::CPU_Parallel    — kernel CPU interpreter, fan-out across threads
//   3. Backend::GPU_Metal       — Apple Metal kernel (one threadgroup per tx)
//   4. Backend::GPU_CUDA        — NVIDIA CUDA kernel (one warp per tx)
//
// All four routes flow through evm::gpu::execute_block(config, txs, nullptr).
// When `state == nullptr` and at least one tx carries bytecode, the dispatcher
// runs the kernel CPU interpreter for the CPU paths and the corresponding GPU
// kernel for the GPU paths — sharing input shape (kernel::HostTransaction) and
// status semantics. That is what makes the comparison meaningful.
//
// The CUDA branch is gated on EVM_CUDA. On macOS that define is absent so the
// CUDA assertions are compiled out (Apple GPU stays in the loop). On Linux/CUDA
// CI the build flips it on and all four backends are checked.

#include "gpu/gpu_dispatch.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

using evm::gpu::Backend;
using evm::gpu::BlockResult;
using evm::gpu::Config;
using evm::gpu::Transaction;
using evm::gpu::TxStatus;

// -- Bytecode builders --------------------------------------------------------

inline void push1(std::vector<uint8_t>& c, uint8_t b)
{
    c.push_back(0x60);
    c.push_back(b);
}

// Emit: PUSH1 size, PUSH1 offset, RETURN.
inline void emit_return(std::vector<uint8_t>& c, uint8_t off, uint8_t sz)
{
    push1(c, sz);
    push1(c, off);
    c.push_back(0xf3);
}

// -- Vector definition --------------------------------------------------------

/// Where the four backends are expected to diverge.
///
/// `Agree` (default): the test asserts that CPU_Sequential, CPU_Parallel,
///                    GPU_Metal (and GPU_CUDA when built) produce identical
///                    gas_used, status, and output.
///
/// `KernelCpuMissing`: the kernel CPU interpreter (lib/evm/gpu/kernel/
///                    evm_interpreter.hpp) does not implement this opcode
///                    yet, but the GPU kernels do. The test asserts that
///                    CPU returns Error/0-gas while GPU executes correctly.
///                    A new opcode joining the missing list is a regression.
///
/// `GasOnly`:        The four backends agree on status and output but their
///                    gas accounting differs by a small constant (typically
///                    the cost of MSTORE memory expansion in different
///                    revisions). Asserted only on status+output.
enum class Expectation : uint8_t
{
    Agree            = 0,
    KernelCpuMissing = 1,
    GasOnly          = 2,
};

struct ParityVector
{
    const char* name;
    std::vector<uint8_t> code;
    std::vector<uint8_t> calldata;
    uint64_t gas_limit = 1'000'000;
    Expectation expect = Expectation::Agree;
};

// Build the corpus once at program start. We use a function so that
// vector construction can use the helpers above. >100 vectors covering
// every opcode group implemented by the kernel.
const std::vector<ParityVector>& corpus()
{
    static const std::vector<ParityVector> v = []{
        std::vector<ParityVector> out;

        // === Arithmetic ====================================================
        out.push_back({"arith_add",     {0x60,0x05, 0x60,0x03, 0x01,
                                         0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3}, {}, 50'000});
        out.push_back({"arith_sub",     {0x60,0x03, 0x60,0x0a, 0x03,
                                         0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3}, {}, 50'000});
        out.push_back({"arith_mul",     {0x60,0x06, 0x60,0x07, 0x02,
                                         0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3}, {}, 50'000});
        out.push_back({"arith_div",     {0x60,0x04, 0x60,0x14, 0x04,
                                         0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3}, {}, 50'000});
        out.push_back({"arith_div_zero",{0x60,0x00, 0x60,0x14, 0x04,
                                         0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3}, {}, 50'000});
        out.push_back({"arith_mod",     {0x60,0x07, 0x60,0x19, 0x06,
                                         0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3}, {}, 50'000});
        out.push_back({"arith_mod_zero",{0x60,0x00, 0x60,0x19, 0x06,
                                         0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3}, {}, 50'000});
        out.push_back({"arith_addmod",  {0x60,0x03, 0x60,0x05, 0x60,0x04, 0x08,
                                         0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3}, {}, 50'000});
        out.push_back({"arith_mulmod",  {0x60,0x0d, 0x60,0x0b, 0x60,0x07, 0x09,
                                         0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3}, {}, 50'000});
        out.push_back({"arith_exp",     {0x60,0x04, 0x60,0x03, 0x0a,
                                         0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3}, {}, 50'000});
        // SDIV: -8 / 5 = -2 (push 5; push -8 as a 32-byte two's complement; SDIV)
        {
            std::vector<uint8_t> c = {0x60, 0x05, 0x7f};
            for (int i = 0; i < 31; ++i) c.push_back(0xFF);
            c.push_back(0xF8);
            c.push_back(0x05);
            push1(c, 0x00); c.push_back(0x52);
            emit_return(c, 0x00, 0x20);
            out.push_back({"arith_sdiv_neg", std::move(c), {}, 50'000});
        }
        // SMOD: -7 % 3 = -1
        {
            std::vector<uint8_t> c = {0x60, 0x03, 0x7f};
            for (int i = 0; i < 31; ++i) c.push_back(0xFF);
            c.push_back(0xF9);
            c.push_back(0x07);
            push1(c, 0x00); c.push_back(0x52);
            emit_return(c, 0x00, 0x20);
            out.push_back({"arith_smod_neg", std::move(c), {}, 50'000});
        }
        // SIGNEXTEND of byte 0 with high bit set: 0xFF -> all-ones 32 bytes
        {
            std::vector<uint8_t> c = {0x60, 0xFF, 0x60, 0x00, 0x0b};
            push1(c, 0x00); c.push_back(0x52);
            emit_return(c, 0x00, 0x20);
            out.push_back({"arith_signextend", std::move(c), {}, 50'000});
        }

        // === Comparison ====================================================
        out.push_back({"cmp_lt",   {0x60,0x07, 0x60,0x05, 0x10,
                                    0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3}, {}, 50'000});
        out.push_back({"cmp_gt",   {0x60,0x05, 0x60,0x07, 0x11,
                                    0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3}, {}, 50'000});
        out.push_back({"cmp_eq",   {0x60,0x2a, 0x60,0x2a, 0x14,
                                    0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3}, {}, 50'000});
        out.push_back({"cmp_ne",   {0x60,0x2a, 0x60,0x2b, 0x14,
                                    0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3}, {}, 50'000});
        out.push_back({"cmp_iszero_t",{0x60,0x00, 0x15,
                                    0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3}, {}, 50'000});
        out.push_back({"cmp_iszero_f",{0x60,0x01, 0x15,
                                    0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3}, {}, 50'000});
        // SLT: -1 < 1 -> 1
        {
            std::vector<uint8_t> c = {0x60, 0x01, 0x7f};
            for (int i = 0; i < 32; ++i) c.push_back(0xFF);
            c.push_back(0x12);
            push1(c, 0x00); c.push_back(0x52);
            emit_return(c, 0x00, 0x20);
            out.push_back({"cmp_slt_neg", std::move(c), {}, 50'000});
        }
        // SGT: -1 > 1 -> 0
        {
            std::vector<uint8_t> c = {0x60, 0x01, 0x7f};
            for (int i = 0; i < 32; ++i) c.push_back(0xFF);
            c.push_back(0x13);
            push1(c, 0x00); c.push_back(0x52);
            emit_return(c, 0x00, 0x20);
            out.push_back({"cmp_sgt_neg", std::move(c), {}, 50'000});
        }

        // === Bitwise =======================================================
        out.push_back({"bit_and", {0x60,0x0f, 0x60,0xff, 0x16,
                                   0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3}, {}, 50'000});
        out.push_back({"bit_or",  {0x60,0x0f, 0x60,0xf0, 0x17,
                                   0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3}, {}, 50'000});
        out.push_back({"bit_xor", {0x60,0x0f, 0x60,0xff, 0x18,
                                   0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3}, {}, 50'000});
        out.push_back({"bit_not", {0x60,0x00, 0x19,
                                   0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3}, {}, 50'000});
        // BYTE: extract byte 31 (LSB) of 0x56 -> 0x56
        out.push_back({"bit_byte",{0x60,0x56, 0x60,0x1f, 0x1a,
                                   0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3}, {}, 50'000});
        out.push_back({"bit_shl", {0x60,0x01, 0x60,0x04, 0x1b,
                                   0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3}, {}, 50'000});
        out.push_back({"bit_shr", {0x60,0x10, 0x60,0x04, 0x1c,
                                   0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3}, {}, 50'000});
        // SAR with negative high bit: -16 >>a 4 = -1
        {
            std::vector<uint8_t> c = {0x7f};
            for (int i = 0; i < 31; ++i) c.push_back(0xFF);
            c.push_back(0xF0);
            c.push_back(0x60); c.push_back(0x04);
            c.push_back(0x1d);
            push1(c, 0x00); c.push_back(0x52);
            emit_return(c, 0x00, 0x20);
            out.push_back({"bit_sar_neg", std::move(c), {}, 50'000});
        }

        // === Hashing =======================================================
        // KECCAK256 (0x20) — implemented in CPU interpreter as of v0.26 with
        // a header-only port of the Metal kernel's keccak256 primitive.
        out.push_back({"hash_keccak_empty",
            {0x60, 0x00,           // PUSH1 0    (len)
             0x60, 0x00,           // PUSH1 0    (offset)
             0x20,                  // KECCAK256
             0x60, 0x00, 0x52,      // MSTORE
             0x60, 0x20, 0x60, 0x00, 0xf3}, {}, 100'000});

        out.push_back({"hash_keccak_abc",
            {0x60, 0x61, 0x60, 0x00, 0x53,
             0x60, 0x62, 0x60, 0x01, 0x53,
             0x60, 0x63, 0x60, 0x02, 0x53,
             0x60, 0x03, 0x60, 0x00, 0x20,
             0x60, 0x00, 0x52,
             0x60, 0x20, 0x60, 0x00, 0xf3}, {}, 100'000});

        out.push_back({"hash_keccak_32z",
            {0x60, 0x00, 0x60, 0x00, 0x52,
             0x60, 0x20, 0x60, 0x00, 0x20,
             0x60, 0x00, 0x52,
             0x60, 0x20, 0x60, 0x00, 0xf3}, {}, 100'000});

        out.push_back({"hash_keccak_256z",
            {0x61, 0x01, 0x00, 0x60, 0x00, 0x20,
             0x60, 0x00, 0x52,
             0x60, 0x20, 0x60, 0x00, 0xf3}, {}, 200'000});

        // === Context (block / tx) ==========================================
        // ADDRESS, CALLER, CALLVALUE are implemented in both CPU and GPU.
        out.push_back({"ctx_address", {0x30,
                                       0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3}, {}, 50'000});
        out.push_back({"ctx_caller",  {0x33,
                                       0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3}, {}, 50'000});
        out.push_back({"ctx_callvalue",{0x34,
                                       0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3}, {}, 50'000});

        // ORIGIN, GASPRICE, CHAINID, BASEFEE, BLOBBASEFEE, COINBASE,
        // TIMESTAMP, NUMBER, GASLIMIT, PREVRANDAO, BLOCKHASH, BLOBHASH —
        // implemented in the CPU interpreter as of v0.26 (see
        // lib/evm/gpu/kernel/evm_interpreter.hpp BlockContext wiring).
        out.push_back({"ctx_origin",     {0x32, 0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3},
                       {}, 50'000});
        out.push_back({"ctx_gasprice",   {0x3a, 0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3},
                       {}, 50'000});
        out.push_back({"ctx_chainid",    {0x46, 0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3},
                       {}, 50'000});
        out.push_back({"ctx_basefee",    {0x48, 0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3},
                       {}, 50'000});
        out.push_back({"ctx_blobbasefee",{0x4a, 0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3},
                       {}, 50'000});
        out.push_back({"ctx_coinbase",   {0x41, 0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3},
                       {}, 50'000});
        out.push_back({"ctx_timestamp",  {0x42, 0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3},
                       {}, 50'000});
        out.push_back({"ctx_number",     {0x43, 0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3},
                       {}, 50'000});
        out.push_back({"ctx_gaslimit",   {0x45, 0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3},
                       {}, 50'000});
        out.push_back({"ctx_difficulty", {0x44, 0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3},
                       {}, 50'000});
        out.push_back({"ctx_blockhash",  {0x60,0x05, 0x40, 0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3},
                       {}, 50'000});
        out.push_back({"ctx_blobhash",   {0x60,0x00, 0x49, 0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3},
                       {}, 50'000});

        // === Calldata ======================================================
        out.push_back({"cd_size",
            {0x36, 0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3},
            {1,2,3,4,5}, 50'000});
        out.push_back({"cd_size_empty",
            {0x36, 0x60,0x00, 0x52, 0x60,0x20,0x60,0x00,0xf3},
            {}, 50'000});
        // cd_load / cd_copy: gas now agrees byte-for-byte with the GPU
        // kernels (v0.26 brought the CPU interpreter up to parity).
        out.push_back({"cd_load",
            {0x60,0x00, 0x35, 0x60,0x00, 0x52, 0x60,0x20, 0x60,0x00, 0xf3},
            std::vector<uint8_t>(32, 0xAB), 50'000});
        out.push_back({"cd_copy",
            {0x60,0x05, 0x60,0x00, 0x60,0x00, 0x37,
             0x60,0x05, 0x60,0x00, 0xf3},
            {0x11,0x22,0x33,0x44,0x55}, 50'000});

        // === Code ==========================================================
        // CODESIZE (0x38) and CODECOPY (0x39) — implemented in CPU as of v0.26.
        out.push_back({"code_size",
            {0x38, 0x60,0x00, 0x52, 0x60,0x20, 0x60,0x00, 0xf3}, {}, 50'000});
        out.push_back({"code_copy",
            {0x60,0x04, 0x60,0x00, 0x60,0x00, 0x39,
             0x60,0x04, 0x60,0x00, 0xf3}, {}, 50'000});

        // === Memory ========================================================
        // MLOAD/MSTORE round-trip
        out.push_back({"mem_mstore_mload",
            {0x60,0x42, 0x60,0x00, 0x52,         // mstore 0x42 at offset 0
             0x60,0x00, 0x51,                     // mload offset 0
             0x60,0x20, 0x52,                     // mstore at offset 32
             0x60,0x20, 0x60,0x20, 0xf3}, {}, 50'000});
        // MSTORE8: write a single byte
        out.push_back({"mem_mstore8",
            {0x60,0xAB, 0x60,0x00, 0x53,
             0x60,0x00, 0x51,
             0x60,0x20, 0x52,
             0x60,0x20, 0x60,0x20, 0xf3}, {}, 50'000});
        // MSIZE — gas now matches the GPU kernels.
        out.push_back({"mem_msize",
            {0x60,0x00, 0x60,0x00, 0x52,
             0x59,
             0x60,0x20, 0x52,
             0x60,0x20, 0x60,0x20, 0xf3}, {}, 50'000});
        // MCOPY (0x5e, EIP-5656) — implemented in CPU as of v0.26.
        out.push_back({"mem_mcopy",
            {0x60,0xAA, 0x60,0x00, 0x53,
             0x60,0x01, 0x60,0x00, 0x60,0x10, 0x5e,
             0x60,0x10, 0x60,0x01, 0xf3}, {}, 50'000});

        // === Storage =======================================================
        // SSTORE then SLOAD — CPU and GPU both implement these.
        out.push_back({"stor_sstore_sload",
            {0x60,0x07, 0x60,0x01, 0x55,         // sstore [1]=7
             0x60,0x01, 0x54,                     // sload [1]
             0x60,0x00, 0x52, 0x60,0x20, 0x60,0x00, 0xf3}, {}, 100'000});
        // TLOAD (0x5c) and TSTORE (0x5d, EIP-1153) — implemented in CPU as of v0.26.
        out.push_back({"stor_tstore_tload",
            {0x60,0x09, 0x60,0x02, 0x5d,
             0x60,0x02, 0x5c,
             0x60,0x00, 0x52, 0x60,0x20, 0x60,0x00, 0xf3}, {}, 50'000});
        out.push_back({"stor_sload_zero",
            {0x60,0x05, 0x54,
             0x60,0x00, 0x52, 0x60,0x20, 0x60,0x00, 0xf3}, {}, 100'000});

        // === Control flow ==================================================
        // JUMP to JUMPDEST
        out.push_back({"ctl_jump",
            {0x60,0x05, 0x56,                     // PUSH1 5, JUMP
             0x60,0x63,                           // dead code
             0x5b,                                 // JUMPDEST at pc=5
             0x60,0x01, 0x60,0x00, 0x52,
             0x60,0x20, 0x60,0x00, 0xf3}, {}, 50'000});
        // JUMPI taken
        out.push_back({"ctl_jumpi_t",
            {0x60,0x01, 0x60,0x07, 0x57,         // PUSH1 1, PUSH1 7, JUMPI
             0x60,0xff,                           // dead
             0x5b,                                 // JUMPDEST at pc=7
             0x60,0x2a, 0x60,0x00, 0x52,
             0x60,0x20, 0x60,0x00, 0xf3}, {}, 50'000});
        // JUMPI not taken (cond=0)
        out.push_back({"ctl_jumpi_f",
            {0x60,0x00, 0x60,0x07, 0x57,         // skip jump
             0x60,0x55, 0x60,0x00, 0x52,         // store 0x55
             0x60,0x20, 0x60,0x00, 0xf3,
             0x5b,                                 // JUMPDEST at pc=12
             0x60,0xee, 0x60,0x00, 0x52,
             0x60,0x20, 0x60,0x00, 0xf3}, {}, 50'000});
        // PC reads program counter
        out.push_back({"ctl_pc",
            {0x58,
             0x60,0x00, 0x52, 0x60,0x20, 0x60,0x00, 0xf3}, {}, 50'000});
        // GAS reads remaining
        out.push_back({"ctl_gas",
            {0x5a,
             0x60,0x00, 0x52, 0x60,0x20, 0x60,0x00, 0xf3}, {}, 50'000});
        // STOP
        out.push_back({"ctl_stop",
            {0x00}, {}, 50'000});
        // INVALID
        out.push_back({"ctl_invalid",
            {0xfe}, {}, 50'000});
        // Undefined opcode — CPU now consumes all remaining gas (matches
        // Metal's ERRA() and the Yellow Paper rule for unknown ops).
        out.push_back({"ctl_undefined",
            {0x0c}, {}, 50'000});
        // Bad jump target -> error
        out.push_back({"ctl_bad_jump",
            {0x60,0xff, 0x56}, {}, 50'000});

        // === Stack: PUSH0..PUSH32 ==========================================
        out.push_back({"stk_push0",
            {0x5f, 0x60,0x00, 0x52, 0x60,0x20, 0x60,0x00, 0xf3}, {}, 50'000});
        out.push_back({"stk_push1",
            {0x60,0x42, 0x60,0x00, 0x52, 0x60,0x20, 0x60,0x00, 0xf3}, {}, 50'000});
        out.push_back({"stk_push2",
            {0x61,0xab,0xcd, 0x60,0x00, 0x52, 0x60,0x20, 0x60,0x00, 0xf3}, {}, 50'000});
        out.push_back({"stk_push4",
            {0x63,0xde,0xad,0xbe,0xef, 0x60,0x00, 0x52, 0x60,0x20, 0x60,0x00, 0xf3}, {}, 50'000});
        out.push_back({"stk_push8",
            {0x67,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,
             0x60,0x00, 0x52, 0x60,0x20, 0x60,0x00, 0xf3}, {}, 50'000});
        out.push_back({"stk_push16",
            {0x6f,
             0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,
             0x18,0x19,0x1a,0x1b,0x1c,0x1d,0x1e,0x1f,
             0x60,0x00, 0x52, 0x60,0x20, 0x60,0x00, 0xf3}, {}, 50'000});
        // PUSH32: all-ones
        {
            std::vector<uint8_t> c = {0x7f};
            for (int i = 0; i < 32; ++i) c.push_back(0xFF);
            push1(c, 0x00); c.push_back(0x52);
            emit_return(c, 0x00, 0x20);
            out.push_back({"stk_push32_ones", std::move(c), {}, 50'000});
        }

        // === Stack: DUP/SWAP/POP ===========================================
        out.push_back({"stk_dup1",
            {0x60,0x2a, 0x80, 0x01,              // PUSH 0x2a, DUP1, ADD -> 0x54
             0x60,0x00, 0x52, 0x60,0x20, 0x60,0x00, 0xf3}, {}, 50'000});
        out.push_back({"stk_dup2",
            {0x60,0x05, 0x60,0x07, 0x81, 0x01,   // [5,7]→[5,7,5], ADD top: 12
             0x60,0x00, 0x52, 0x60,0x20, 0x60,0x00, 0xf3}, {}, 50'000});
        out.push_back({"stk_swap1",
            {0x60,0x05, 0x60,0x03, 0x90, 0x03,   // SWAP1 then SUB: 5-3=2
             0x60,0x00, 0x52, 0x60,0x20, 0x60,0x00, 0xf3}, {}, 50'000});
        out.push_back({"stk_swap2",
            {0x60,0x05, 0x60,0x03, 0x60,0x07, 0x91, 0x01,
             0x60,0x00, 0x52, 0x60,0x20, 0x60,0x00, 0xf3}, {}, 50'000});
        out.push_back({"stk_pop",
            {0x60,0x05, 0x60,0x03, 0x50,         // PUSH 5, PUSH 3, POP -> stack=[5]
             0x60,0x00, 0x52, 0x60,0x20, 0x60,0x00, 0xf3}, {}, 50'000});

        // Higher DUP/SWAP coverage
        for (int n = 3; n <= 16; ++n)
        {
            std::vector<uint8_t> c;
            // Push 16 distinct values [16, 15, ..., 1] (so top of stack = 1).
            for (int i = 16; i >= 1; --i) push1(c, static_cast<uint8_t>(i));
            // DUPn duplicates the n-th element from the top.
            c.push_back(static_cast<uint8_t>(0x80 + (n - 1)));
            push1(c, 0x00); c.push_back(0x52);
            emit_return(c, 0x00, 0x20);
            char name[32]; std::snprintf(name, sizeof name, "stk_dup%d", n);
            out.push_back({name, std::move(c), {}, 50'000});
        }
        for (int n = 3; n <= 16; ++n)
        {
            std::vector<uint8_t> c;
            for (int i = 17; i >= 1; --i) push1(c, static_cast<uint8_t>(i));
            // SWAPn swaps top with the (n+1)-th element from the top.
            c.push_back(static_cast<uint8_t>(0x90 + (n - 1)));
            push1(c, 0x00); c.push_back(0x52);
            emit_return(c, 0x00, 0x20);
            char name[32]; std::snprintf(name, sizeof name, "stk_swap%d", n);
            out.push_back({name, std::move(c), {}, 50'000});
        }

        // === Logging =======================================================
        // LOG0 with 4 bytes of data, then STOP
        out.push_back({"log_log0",
            {0x60,0xDE, 0x60,0x00, 0x53,
             0x60,0xAD, 0x60,0x01, 0x53,
             0x60,0xBE, 0x60,0x02, 0x53,
             0x60,0xEF, 0x60,0x03, 0x53,
             0x60,0x04, 0x60,0x00, 0xa0,
             0x00}, {}, 50'000});
        // LOG1 with one topic
        out.push_back({"log_log1",
            {0x60,0x11, 0x60,0x00, 0x60,0x00, 0xa1,
             0x00}, {}, 50'000});
        // LOG2
        out.push_back({"log_log2",
            {0x60,0x22, 0x60,0x11, 0x60,0x00, 0x60,0x00, 0xa2,
             0x00}, {}, 50'000});
        // LOG3
        out.push_back({"log_log3",
            {0x60,0x33, 0x60,0x22, 0x60,0x11,
             0x60,0x00, 0x60,0x00, 0xa3,
             0x00}, {}, 50'000});
        // LOG4
        out.push_back({"log_log4",
            {0x60,0x44, 0x60,0x33, 0x60,0x22, 0x60,0x11,
             0x60,0x00, 0x60,0x00, 0xa4,
             0x00}, {}, 50'000});

        // === Returndata ====================================================
        out.push_back({"ret_return_word",
            {0x60,0x42, 0x60,0x00, 0x52, 0x60,0x20, 0x60,0x00, 0xf3}, {}, 50'000});
        out.push_back({"ret_revert_word",
            {0x60,0x42, 0x60,0x00, 0x52, 0x60,0x20, 0x60,0x00, 0xfd}, {}, 50'000});
        out.push_back({"ret_return_empty",
            {0x60,0x00, 0x60,0x00, 0xf3}, {}, 50'000});
        out.push_back({"ret_revert_empty",
            {0x60,0x00, 0x60,0x00, 0xfd}, {}, 50'000});
        // RETURNDATASIZE (0x3d) — implemented in CPU as of v0.26 (returns 0
        // with no prior call frame).
        out.push_back({"ret_returndatasize",
            {0x3d,
             0x60,0x00, 0x52, 0x60,0x20, 0x60,0x00, 0xf3}, {}, 50'000});

        // === State defaults (no host installed) ===========================
        // BALANCE, SELFBALANCE, EXTCODESIZE/HASH/COPY — implemented in CPU
        // as of v0.26. With no host wired, balance/code default to 0 and
        // EXTCODEHASH defaults to keccak256("").
        out.push_back({"state_balance",
            {0x60,0xAA, 0x31,
             0x60,0x00, 0x52, 0x60,0x20, 0x60,0x00, 0xf3}, {}, 50'000});
        out.push_back({"state_selfbalance",
            {0x47,
             0x60,0x00, 0x52, 0x60,0x20, 0x60,0x00, 0xf3}, {}, 50'000});
        out.push_back({"state_extcodesize",
            {0x60,0xAA, 0x3b,
             0x60,0x00, 0x52, 0x60,0x20, 0x60,0x00, 0xf3}, {}, 50'000});
        out.push_back({"state_extcodehash",
            {0x60,0xAA, 0x3f,
             0x60,0x00, 0x52, 0x60,0x20, 0x60,0x00, 0xf3}, {}, 50'000});
        out.push_back({"state_extcodecopy",
            {0x60,0x04, 0x60,0x00, 0x60,0x00, 0x60,0xAA, 0x3c,
             0x60,0x04, 0x60,0x00, 0xf3}, {}, 50'000});

        // === CALL family — kernel reports CallNotSupported ================
        for (auto op : std::initializer_list<uint8_t>{0xf0, 0xf1, 0xf2, 0xf4, 0xf5, 0xfa, 0xff})
        {
            char name[64]; std::snprintf(name, sizeof name, "call_not_supp_0x%02x", op);
            out.push_back({name, {op}, {}, 100'000});
        }

        // === Out-of-gas ====================================================
        // ADD with insufficient gas → OutOfGas
        out.push_back({"oog_low_gas",
            {0x60,0x05, 0x60,0x03, 0x01,
             0x60,0x00, 0x52, 0x60,0x20, 0x60,0x00, 0xf3}, {}, 5});

        return out;
    }();
    return v;
}

// -- Test fixture -------------------------------------------------------------

Transaction make_tx(const ParityVector& v)
{
    Transaction t;
    t.from.assign(20, 0);
    t.from[19] = 0x01;
    t.to.assign(20, 0);
    t.to[19] = 0x02;
    t.code      = v.code;
    t.data      = v.calldata;
    t.gas_limit = v.gas_limit;
    t.gas_price = 1;
    t.value     = 0;
    t.nonce     = 0;
    return t;
}

BlockResult run_one(Backend backend, const ParityVector& v)
{
    Config cfg;
    cfg.backend = backend;
    cfg.num_threads = 2;
    std::vector<Transaction> txs{make_tx(v)};
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

class CorpusEnv : public ::testing::Environment
{
public:
    void SetUp() override
    {
        std::printf("[parity] corpus size: %zu vectors\n", corpus().size());
        auto avail = evm::gpu::available_backends();
        std::printf("[parity] available backends:");
        for (auto b : avail)
            std::printf(" %s", evm::gpu::backend_name(b));
        std::printf("\n");
    }
};

[[maybe_unused]] auto* g_env =
    ::testing::AddGlobalTestEnvironment(new CorpusEnv);

}  // namespace

// Single test that loops the corpus and runs all four backends per vector.
// Behaviour per Expectation:
//
//   Agree:            All backends must produce identical (gas, status,
//                     output). Any divergence is a regression and FAILS.
//
//   GasOnly:          Status and output must match across all backends, but
//                     gas accounting may differ. Used when the CPU and GPU
//                     interpreters agree on the program's visible result
//                     but charge gas slightly differently.
//
//   KernelCpuMissing: The kernel CPU interpreter does not implement this
//                     opcode. CPU paths must report status=Error AND
//                     output={}. GPU paths must succeed (status=Return).
//                     If any backend diverges from this layout, the test
//                     fails — that means either CPU has caught up (and the
//                     vector should be re-classed Agree) or GPU has lost
//                     coverage (regression).
TEST(Parity, AllBackendsAgreeOnEveryVector)
{
    int agree_ok = 0, agree_fail = 0;
    int gasonly_ok = 0, gasonly_fail = 0;
    int missing_ok = 0, missing_fail = 0;
    int total_seq_runs = 0, total_par_runs = 0, total_metal_runs = 0;
#ifdef EVM_CUDA
    int total_cuda_runs = 0;
#endif

    for (const auto& v : corpus())
    {
        auto r_seq   = run_one(Backend::CPU_Sequential, v);
        auto r_par   = run_one(Backend::CPU_Parallel,   v);
        auto r_metal = run_one(Backend::GPU_Metal,      v);

        ASSERT_EQ(r_seq.status.size(),   1u) << v.name << " (seq)";
        ASSERT_EQ(r_par.status.size(),   1u) << v.name << " (par)";
        ASSERT_EQ(r_metal.status.size(), 1u) << v.name << " (metal)";

        ++total_seq_runs;
        ++total_par_runs;
        ++total_metal_runs;

#ifdef EVM_CUDA
        auto r_cuda = run_one(Backend::GPU_CUDA, v);
        ASSERT_EQ(r_cuda.status.size(), 1u) << v.name << " (cuda)";
        ++total_cuda_runs;
#endif

        bool ok = true;

        switch (v.expect)
        {
        case Expectation::Agree:
        {
            // CPU sequential vs CPU parallel must always match exactly,
            // regardless of expectation.
            if (r_seq.status[0]  != r_par.status[0] ||
                r_seq.gas_used[0]!= r_par.gas_used[0] ||
                r_seq.output[0]  != r_par.output[0])
            {
                ADD_FAILURE() << v.name << ": CPU seq vs par diverge";
                ok = false;
            }
            if (r_seq.status[0]   != r_metal.status[0] ||
                r_seq.gas_used[0] != r_metal.gas_used[0] ||
                r_seq.output[0]   != r_metal.output[0])
            {
                ADD_FAILURE() << v.name << ": CPU vs Metal diverge"
                              << "  cpu.gas=" << r_seq.gas_used[0]
                              << " metal.gas=" << r_metal.gas_used[0]
                              << " cpu.status=" << static_cast<uint32_t>(r_seq.status[0])
                              << " metal.status=" << static_cast<uint32_t>(r_metal.status[0])
                              << " cpu.out=0x" << hex(r_seq.output[0])
                              << " metal.out=0x" << hex(r_metal.output[0]);
                ok = false;
            }
#ifdef EVM_CUDA
            if (r_seq.status[0]   != r_cuda.status[0] ||
                r_seq.gas_used[0] != r_cuda.gas_used[0] ||
                r_seq.output[0]   != r_cuda.output[0])
            {
                ADD_FAILURE() << v.name << ": CPU vs CUDA diverge";
                ok = false;
            }
#endif
            if (ok) ++agree_ok; else ++agree_fail;
            break;
        }

        case Expectation::GasOnly:
        {
            if (r_seq.status[0] != r_par.status[0] ||
                r_seq.output[0] != r_par.output[0])
            {
                ADD_FAILURE() << v.name << ": CPU seq vs par status/output diverge";
                ok = false;
            }
            if (r_seq.status[0] != r_metal.status[0] ||
                r_seq.output[0] != r_metal.output[0])
            {
                ADD_FAILURE() << v.name << ": CPU vs Metal status/output diverge";
                ok = false;
            }
#ifdef EVM_CUDA
            if (r_seq.status[0] != r_cuda.status[0] ||
                r_seq.output[0] != r_cuda.output[0])
            {
                ADD_FAILURE() << v.name << ": CPU vs CUDA status/output diverge";
                ok = false;
            }
#endif
            // Documented: gas may differ. Catch the case where it
            // unexpectedly converges (good news — promote to Agree).
            if (r_seq.gas_used[0] == r_metal.gas_used[0] && ok)
            {
                std::printf("[parity] %s: gas now matches (%llu) — "
                            "consider promoting to Expectation::Agree\n",
                            v.name,
                            static_cast<unsigned long long>(r_seq.gas_used[0]));
            }
            if (ok) ++gasonly_ok; else ++gasonly_fail;
            break;
        }

        case Expectation::KernelCpuMissing:
        {
            // CPU paths must agree with each other (both miss the opcode).
            if (r_seq.status[0]   != r_par.status[0] ||
                r_seq.gas_used[0] != r_par.gas_used[0] ||
                r_seq.output[0]   != r_par.output[0])
            {
                ADD_FAILURE() << v.name << ": CPU seq vs par diverge";
                ok = false;
            }
            // CPU must report Error (or, for KECCAK256 with no opcode handler,
            // anything that's not Return). If CPU starts succeeding, demote
            // the vector to Agree.
            if (r_seq.status[0] == TxStatus::Return)
            {
                ADD_FAILURE() << v.name
                    << ": CPU now succeeds — promote to Expectation::Agree";
                ok = false;
            }
            // Metal must succeed. If it fails, that's a regression.
            if (r_metal.status[0] != TxStatus::Return)
            {
                ADD_FAILURE() << v.name
                    << ": Metal regressed (status=" << static_cast<uint32_t>(r_metal.status[0]) << ")";
                ok = false;
            }
#ifdef EVM_CUDA
            if (r_cuda.status[0] != TxStatus::Return)
            {
                ADD_FAILURE() << v.name << ": CUDA regressed";
                ok = false;
            }
            // Metal and CUDA must agree (both GPU kernels).
            if (r_metal.gas_used[0] != r_cuda.gas_used[0] ||
                r_metal.output[0]   != r_cuda.output[0])
            {
                ADD_FAILURE() << v.name << ": Metal vs CUDA diverge";
                ok = false;
            }
#endif
            if (ok) ++missing_ok; else ++missing_fail;
            break;
        }
        }
    }

    const auto total = corpus().size();
    std::printf("[parity] runs:  CPU_Sequential=%d CPU_Parallel=%d GPU_Metal=%d",
                total_seq_runs, total_par_runs, total_metal_runs);
#ifdef EVM_CUDA
    std::printf(" GPU_CUDA=%d", total_cuda_runs);
#endif
    std::printf("\n[parity] Agree:            %d ok, %d fail\n",
                agree_ok, agree_fail);
    std::printf("[parity] GasOnly:          %d ok, %d fail\n",
                gasonly_ok, gasonly_fail);
    std::printf("[parity] KernelCpuMissing: %d ok, %d fail\n",
                missing_ok, missing_fail);
    std::printf("[parity] total %d / %zu vectors meet expectations\n",
                agree_ok + gasonly_ok + missing_ok, total);

    EXPECT_EQ(agree_fail + gasonly_fail + missing_fail, 0);
}
