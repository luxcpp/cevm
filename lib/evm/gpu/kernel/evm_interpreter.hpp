// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file evm_interpreter.hpp
/// GPU EVM bytecode interpreter.
///
/// Executes EVM bytecode in a single thread (one transaction per thread).
/// Storage access (SLOAD/SSTORE) goes through a callback interface so the
/// host can wire it to MvMemory for Block-STM parallel execution.
///
/// Opcodes NOT implemented (require cross-transaction state or nested calls):
///   CALL, STATICCALL, DELEGATECALL, CALLCODE, CREATE, CREATE2,
///   SELFDESTRUCT, EXTCODESIZE, EXTCODECOPY, EXTCODEHASH, BALANCE,
///   BLOCKHASH, COINBASE, TIMESTAMP, NUMBER, PREVRANDAO, GASLIMIT,
///   CHAINID, SELFBALANCE, BASEFEE, BLOBHASH, BLOBBASEFEE,
///   GASPRICE, ORIGIN, KECCAK256, CODESIZE, CODECOPY,
///   RETURNDATASIZE, RETURNDATACOPY
///
/// These are handled by falling back to CPU evmone in the host dispatcher.

#pragma once

#include "evm_stack.hpp"
#include "uint256_gpu.hpp"

namespace evm::gpu::kernel {

// -- Gas costs (Shanghai) -----------------------------------------------------

struct GasCost
{
    static constexpr gpu_u64 ZERO       = 0;
    static constexpr gpu_u64 BASE       = 2;
    static constexpr gpu_u64 VERYLOW    = 3;
    static constexpr gpu_u64 LOW        = 5;
    static constexpr gpu_u64 MID        = 8;
    static constexpr gpu_u64 HIGH       = 10;
    static constexpr gpu_u64 JUMPDEST   = 1;
    static constexpr gpu_u64 SLOAD      = 2100;  // cold
    static constexpr gpu_u64 SSTORE_SET = 20000;
    static constexpr gpu_u64 SSTORE_RESET = 2900;
    static constexpr gpu_u64 EXP_BASE   = 10;
    static constexpr gpu_u64 EXP_BYTE   = 50;
    static constexpr gpu_u64 MEMORY     = 3;
    static constexpr gpu_u64 LOG_BASE   = 375;
    static constexpr gpu_u64 LOG_DATA   = 8;
    static constexpr gpu_u64 LOG_TOPIC  = 375;
    static constexpr gpu_u64 COPY       = 3;
};

// -- EVM Memory ---------------------------------------------------------------

/// Flat byte-addressable memory for EVM execution.
/// On GPU, we cap this at a fixed size to avoid unbounded allocation.
static constexpr gpu_u32 MAX_MEMORY = 1024 * 1024;  // 1 MB per transaction

struct EvmMemory
{
    gpu_u32 size;  // current logical size in bytes (always multiple of 32)

    // On GPU: memory is passed as a pointer into a device buffer.
    // On CPU: this points to a heap allocation.
    // The interpreter receives this pointer externally.

    GPU_INLINE static gpu_u64 memory_cost(gpu_u32 word_count)
    {
        gpu_u64 w = word_count;
        return GasCost::MEMORY * w + (w * w) / 512;
    }
};

// -- Storage callback ---------------------------------------------------------

/// Storage access callback. On GPU, these are function pointers or indices
/// into a storage buffer. On CPU, they can be virtual calls.
///
/// For the GPU kernel, storage is a flat buffer of (address, slot) -> value
/// mappings provided by the host.

struct StorageSlot
{
    uint256 value;
    bool found;
};

// -- Log entry ----------------------------------------------------------------

/// A log entry recorded during execution.
static constexpr gpu_u32 MAX_LOG_TOPICS = 4;
static constexpr gpu_u32 MAX_LOG_DATA = 256;
static constexpr gpu_u32 MAX_LOGS = 64;

struct LogEntry
{
    uint256 topics[MAX_LOG_TOPICS];
    gpu_u32 num_topics;
    gpu_u32 data_offset;  // offset into output data area
    gpu_u32 data_size;
};

// -- Interpreter result -------------------------------------------------------

static constexpr gpu_u32 MAX_OUTPUT = 1024;

struct InterpreterResult
{
    ExecStatus status;
    gpu_u64 gas_used;
    gpu_u64 gas_remaining;
    gpu_u32 output_size;
    // Output data is written to a separate buffer provided by the caller.
};

// -- The Interpreter ----------------------------------------------------------

/// EVM bytecode interpreter for GPU execution.
///
/// Usage (pseudocode):
///   EvmInterpreter interp;
///   interp.code = bytecode_ptr;
///   interp.code_size = len;
///   interp.gas = gas_limit;
///   interp.caller = caller_addr;
///   ...
///   auto result = interp.execute(memory_buf, output_buf, storage_buf, ...);
///
/// All pointers are to device memory on GPU, heap memory on CPU.
struct EvmInterpreter
{
    // -- Input parameters (set by host before execute) ------------------------
    const gpu_u8_t* code;
    gpu_u32 code_size;
    const gpu_u8_t* calldata;
    gpu_u32 calldata_size;
    gpu_u64 gas;
    uint256 caller;    // 20 bytes right-aligned in uint256
    uint256 address;   // contract address, 20 bytes right-aligned
    uint256 value;     // msg.value in wei

    // -- Execution state ------------------------------------------------------
    EvmStack stack;
    gpu_u32 pc;        // program counter
    gpu_u32 mem_size;  // current memory size in bytes

    // -- Storage interface (set by host) --------------------------------------
    // storage_keys[i] = slot, storage_values[i] = value, storage_count = number of entries
    // For SLOAD: linear scan (fine for GPU with small storage sets per tx).
    // For SSTORE: append new entry or update existing.
    uint256*  storage_keys;
    uint256*  storage_values;
    gpu_u32*  storage_count;
    gpu_u32   storage_capacity;

    // -- Log output -----------------------------------------------------------
    LogEntry* logs;
    gpu_u32*  log_count;
    gpu_u32   log_capacity;

    // -- Helpers --------------------------------------------------------------

    GPU_INLINE bool consume_gas(gpu_u64 cost)
    {
        if (gas < cost)
            return false;
        gas -= cost;
        return true;
    }

    /// Expand memory to cover [offset, offset+size). Returns gas cost or max
    /// on failure. Updates mem_size.
    GPU_INLINE gpu_u64 expand_memory(gpu_u32 offset, gpu_u32 size, gpu_u8_t* mem)
    {
        if (size == 0)
            return 0;
        gpu_u32 end = offset + size;
        if (end < offset)  // overflow
            return ~gpu_u64(0);
        if (end > MAX_MEMORY)
            return ~gpu_u64(0);

        gpu_u32 new_words = (end + 31) / 32;
        gpu_u32 old_words = mem_size / 32;
        if (new_words <= old_words)
            return 0;

        gpu_u64 cost = EvmMemory::memory_cost(new_words) - EvmMemory::memory_cost(old_words);

        // Zero-fill the new region.
        gpu_u32 new_size = new_words * 32;
        for (gpu_u32 i = mem_size; i < new_size; ++i)
            mem[i] = 0;
        mem_size = new_size;
        return cost;
    }

    /// Read a uint256 from memory at byte offset (big-endian, 32 bytes).
    GPU_INLINE uint256 mload(const gpu_u8_t* mem, gpu_u32 offset) const
    {
        uint256 r;
        // Big-endian: byte at offset is most significant.
        for (int limb = 3; limb >= 0; --limb)
        {
            gpu_u64 v = 0;
            int start = (3 - limb) * 8;
            for (int b = 0; b < 8; ++b)
                v = (v << 8) | gpu_u64(mem[offset + start + b]);
            r.w[limb] = v;
        }
        return r;
    }

    /// Write a uint256 to memory at byte offset (big-endian, 32 bytes).
    GPU_INLINE void mstore(gpu_u8_t* mem, gpu_u32 offset, const uint256& val) const
    {
        for (int limb = 3; limb >= 0; --limb)
        {
            gpu_u64 v = val.w[limb];
            int start = (3 - limb) * 8;
            for (int b = 7; b >= 0; --b)
            {
                mem[offset + start + b] = gpu_u8_t(v & 0xFF);
                v >>= 8;
            }
        }
    }

    /// Extract uint256 from a byte at position for PUSH operations.
    GPU_INLINE uint256 read_push_data(gpu_u32 num_bytes) const
    {
        uint256 r = uint256::zero();
        gpu_u32 start = pc + 1;  // bytes start after the opcode
        // Push data is big-endian: first byte is most significant.
        for (gpu_u32 i = 0; i < num_bytes && (start + i) < code_size; ++i)
        {
            gpu_u32 byte_pos = num_bytes - 1 - i;  // position from LSB
            gpu_u32 limb = byte_pos / 8;
            gpu_u32 shift = (byte_pos % 8) * 8;
            r.w[limb] |= gpu_u64(code[start + i]) << shift;
        }
        return r;
    }

    /// Check if a given PC is a valid JUMPDEST.
    GPU_INLINE bool is_jumpdest(gpu_u32 target) const
    {
        if (target >= code_size)
            return false;
        if (code[target] != 0x5b)  // JUMPDEST opcode
            return false;

        // Verify this isn't inside PUSH data by scanning from the beginning.
        // This is expensive but correct. On GPU with small bytecodes, it's acceptable.
        // For production, we'd precompute a JUMPDEST bitmap on the host.
        gpu_u32 i = 0;
        while (i < target)
        {
            gpu_u8_t op = code[i];
            if (op >= 0x60 && op <= 0x7f)  // PUSH1..PUSH32
                i += 1 + (op - 0x60 + 1);  // skip push data
            else
                i += 1;
        }
        return i == target;
    }

    /// SLOAD: find value for slot in the storage buffer.
    GPU_INLINE uint256 sload(const uint256& slot) const
    {
        gpu_u32 count = *storage_count;
        // Search backwards to find the latest write.
        for (gpu_u32 i = count; i > 0; --i)
        {
            if (eq(storage_keys[i - 1], slot))
                return storage_values[i - 1];
        }
        return uint256::zero();
    }

    /// SSTORE: write value for slot. Appends or updates.
    GPU_INLINE void sstore(const uint256& slot, const uint256& val)
    {
        gpu_u32 count = *storage_count;
        // Check if slot already exists (update in place for latest entry).
        for (gpu_u32 i = count; i > 0; --i)
        {
            if (eq(storage_keys[i - 1], slot))
            {
                storage_values[i - 1] = val;
                return;
            }
        }
        // Append new entry.
        if (count < storage_capacity)
        {
            storage_keys[count] = slot;
            storage_values[count] = val;
            *storage_count = count + 1;
        }
    }

    // -- Main execution loop --------------------------------------------------

    GPU_INLINE InterpreterResult execute(gpu_u8_t* mem, gpu_u8_t* output)
    {
        pc = 0;
        mem_size = 0;
        gpu_u64 gas_start = gas;

        while (pc < code_size)
        {
            gpu_u8_t op = code[pc];

            // -- STOP (0x00) --------------------------------------------------
            if (op == 0x00)
            {
                return {ExecStatus::Stop, gas_start - gas, gas, 0};
            }

            // -- Arithmetic (0x01 - 0x0b) -------------------------------------
            if (op >= 0x01 && op <= 0x0b)
            {
                if (!consume_gas(op <= 0x09 ? GasCost::VERYLOW :
                                 op == 0x0a ? GasCost::EXP_BASE : GasCost::LOW))
                    return {ExecStatus::OutOfGas, gas_start, 0, 0};

                uint256 a, b;
                ExecStatus s;

                // EVM convention: a = first pop (top of stack), b = second pop.
                // Operations use (a, b) in Yellow Paper order.
                switch (op)
                {
                case 0x01: // ADD: a + b
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.push(add(a, b));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    break;
                case 0x02: // MUL: a * b
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.push(mul(a, b));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    break;
                case 0x03: // SUB: a - b
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.push(sub(a, b));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    break;
                case 0x04: // DIV: a / b
                    if (!consume_gas(GasCost::LOW - GasCost::VERYLOW)) // adjust to LOW total
                        return {ExecStatus::OutOfGas, gas_start, 0, 0};
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.push(div(a, b));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    break;
                case 0x05: // SDIV: a / b (signed)
                    if (!consume_gas(GasCost::LOW - GasCost::VERYLOW))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0};
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.push(sdiv(a, b));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    break;
                case 0x06: // MOD: a % b
                    if (!consume_gas(GasCost::LOW - GasCost::VERYLOW))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0};
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.push(mod(a, b));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    break;
                case 0x07: // SMOD: a % b (signed)
                    if (!consume_gas(GasCost::LOW - GasCost::VERYLOW))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0};
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.push(smod(a, b));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    break;
                case 0x08: // ADDMOD: (a + b) % N
                {
                    uint256 n;
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.pop(n); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    if (!consume_gas(GasCost::MID - GasCost::VERYLOW))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0};
                    s = stack.push(addmod(a, b, n));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    break;
                }
                case 0x09: // MULMOD: (a * b) % N
                {
                    uint256 n;
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.pop(n); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    if (!consume_gas(GasCost::MID - GasCost::VERYLOW))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0};
                    s = stack.push(mulmod(a, b, n));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    break;
                }
                case 0x0a: // EXP: a ** b
                {
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    // Dynamic gas: 50 * (number of bytes in exponent)
                    gpu_u32 exp_bytes = 0;
                    if (!iszero(b))
                        exp_bytes = (256 - clz256(b) + 7) / 8;
                    if (!consume_gas(GasCost::EXP_BYTE * exp_bytes))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0};
                    s = stack.push(exp(a, b));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    break;
                }
                case 0x0b: // SIGNEXTEND: signextend(b, x) where b=byte_pos, x=value
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.push(signextend(a, b));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    break;
                }
                ++pc;
                continue;
            }

            // -- Comparison (0x10 - 0x15) -------------------------------------
            if (op >= 0x10 && op <= 0x15)
            {
                if (!consume_gas(GasCost::VERYLOW))
                    return {ExecStatus::OutOfGas, gas_start, 0, 0};

                uint256 a, b;
                ExecStatus s;

                // a = first pop (top), b = second pop
                switch (op)
                {
                case 0x10: // LT: a < b
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.push(lt(a, b) ? uint256::one() : uint256::zero());
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    break;
                case 0x11: // GT: a > b
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.push(gt(a, b) ? uint256::one() : uint256::zero());
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    break;
                case 0x12: // SLT: a < b (signed)
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.push(slt(a, b) ? uint256::one() : uint256::zero());
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    break;
                case 0x13: // SGT: a > b (signed)
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.push(sgt(a, b) ? uint256::one() : uint256::zero());
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    break;
                case 0x14: // EQ: a == b
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.push(eq(a, b) ? uint256::one() : uint256::zero());
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    break;
                case 0x15: // ISZERO: a == 0
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.push(iszero(a) ? uint256::one() : uint256::zero());
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    break;
                }
                ++pc;
                continue;
            }

            // -- Bitwise (0x16 - 0x1d) ----------------------------------------
            if (op >= 0x16 && op <= 0x1d)
            {
                if (!consume_gas(GasCost::VERYLOW))
                    return {ExecStatus::OutOfGas, gas_start, 0, 0};

                uint256 a, b;
                ExecStatus s;

                // a = first pop (top), b = second pop
                switch (op)
                {
                case 0x16: // AND: a & b
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.push(bitwise_and(a, b));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    break;
                case 0x17: // OR: a | b
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.push(bitwise_or(a, b));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    break;
                case 0x18: // XOR: a ^ b
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.push(bitwise_xor(a, b));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    break;
                case 0x19: // NOT: ~a
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.push(bitwise_not(a));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    break;
                case 0x1a: // BYTE: byte_at(x, i) — i=top, x=second
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.push(byte_at(b, a));  // byte_at(value, position)
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    break;
                case 0x1b: // SHL: shift=a (top), value=b (second) -> b << a
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.push(shl(a, b));  // shl(shift_amount, value)
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    break;
                case 0x1c: // SHR: shift=a (top), value=b (second) -> b >> a
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.push(shr(a, b));  // shr(shift_amount, value)
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    break;
                case 0x1d: // SAR: shift=a (top), value=b (second) -> b >>> a
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.push(sar(a, b));  // sar(shift_amount, value)
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    break;
                }
                ++pc;
                continue;
            }

            // -- Environment (0x30 - 0x37) ------------------------------------
            if (op >= 0x30 && op <= 0x37)
            {
                if (!consume_gas(GasCost::BASE))
                    return {ExecStatus::OutOfGas, gas_start, 0, 0};

                ExecStatus s;
                uint256 a, b, c;

                switch (op)
                {
                case 0x30: // ADDRESS
                    s = stack.push(address);
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    break;
                case 0x33: // CALLER
                    s = stack.push(caller);
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    break;
                case 0x34: // CALLVALUE
                    s = stack.push(value);
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    break;
                case 0x35: // CALLDATALOAD
                {
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    uint256 result = uint256::zero();
                    // If offset > calldata_size, result is zero.
                    if (!a.w[1] && !a.w[2] && !a.w[3] && a.w[0] < calldata_size)
                    {
                        gpu_u32 off = gpu_u32(a.w[0]);
                        // Read up to 32 bytes from calldata, zero-padded.
                        for (gpu_u32 i = 0; i < 32; ++i)
                        {
                            gpu_u32 src = off + i;
                            gpu_u8_t byte_val = (src < calldata_size) ? calldata[src] : 0;
                            // Big-endian: byte 0 at most significant position
                            gpu_u32 pos_from_right = 31 - i;
                            gpu_u32 limb = pos_from_right / 8;
                            gpu_u32 shift = (pos_from_right % 8) * 8;
                            result.w[limb] |= gpu_u64(byte_val) << shift;
                        }
                    }
                    s = stack.push(result);
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    break;
                }
                case 0x36: // CALLDATASIZE
                    s = stack.push(uint256{gpu_u64(calldata_size)});
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    break;
                case 0x37: // CALLDATACOPY
                {
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.pop(c); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    // a = destOffset (top), b = offset, c = size
                    if (c.w[1] | c.w[2] | c.w[3] || a.w[1] | a.w[2] | a.w[3])
                        return {ExecStatus::InvalidMemAccess, gas_start - gas, gas, 0};
                    gpu_u32 dest = gpu_u32(a.w[0]);
                    gpu_u32 src_off = (b.w[1] | b.w[2] | b.w[3]) ? calldata_size : gpu_u32(b.w[0]);
                    gpu_u32 sz = gpu_u32(c.w[0]);
                    if (sz > 0)
                    {
                        // Gas: copy cost
                        gpu_u32 words = (sz + 31) / 32;
                        if (!consume_gas(gpu_u64(words) * GasCost::COPY))
                            return {ExecStatus::OutOfGas, gas_start, 0, 0};
                        gpu_u64 mem_cost = expand_memory(dest, sz, mem);
                        if (mem_cost == ~gpu_u64(0) || !consume_gas(mem_cost))
                            return {ExecStatus::OutOfGas, gas_start, 0, 0};
                        for (gpu_u32 i = 0; i < sz; ++i)
                            mem[dest + i] = (src_off + i < calldata_size) ? calldata[src_off + i] : 0;
                    }
                    break;
                }
                default:
                    // Other environment opcodes not supported on GPU.
                    return {ExecStatus::InvalidOpcode, gas_start - gas, gas, 0};
                }
                ++pc;
                continue;
            }

            // -- POP (0x50) ---------------------------------------------------
            if (op == 0x50)
            {
                if (!consume_gas(GasCost::BASE))
                    return {ExecStatus::OutOfGas, gas_start, 0, 0};
                ExecStatus s = stack.drop();
                if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                ++pc;
                continue;
            }

            // -- Memory (0x51 - 0x53, 0x59) -----------------------------------
            if (op == 0x51 || op == 0x52 || op == 0x53 || op == 0x59)
            {
                if (!consume_gas(GasCost::VERYLOW))
                    return {ExecStatus::OutOfGas, gas_start, 0, 0};

                ExecStatus s;
                uint256 a, b;

                switch (op)
                {
                case 0x51: // MLOAD
                {
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    if (a.w[1] | a.w[2] | a.w[3])
                        return {ExecStatus::InvalidMemAccess, gas_start - gas, gas, 0};
                    gpu_u32 off = gpu_u32(a.w[0]);
                    gpu_u64 mem_cost = expand_memory(off, 32, mem);
                    if (mem_cost == ~gpu_u64(0) || !consume_gas(mem_cost))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0};
                    s = stack.push(mload(mem, off));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    break;
                }
                case 0x52: // MSTORE: offset=top, value=second
                {
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    if (a.w[1] | a.w[2] | a.w[3])
                        return {ExecStatus::InvalidMemAccess, gas_start - gas, gas, 0};
                    gpu_u32 off = gpu_u32(a.w[0]);
                    gpu_u64 mem_cost = expand_memory(off, 32, mem);
                    if (mem_cost == ~gpu_u64(0) || !consume_gas(mem_cost))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0};
                    mstore(mem, off, b);
                    break;
                }
                case 0x53: // MSTORE8: offset=top, value=second
                {
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    if (a.w[1] | a.w[2] | a.w[3])
                        return {ExecStatus::InvalidMemAccess, gas_start - gas, gas, 0};
                    gpu_u32 off = gpu_u32(a.w[0]);
                    gpu_u64 mem_cost = expand_memory(off, 1, mem);
                    if (mem_cost == ~gpu_u64(0) || !consume_gas(mem_cost))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0};
                    mem[off] = gpu_u8_t(b.w[0] & 0xFF);
                    break;
                }
                case 0x59: // MSIZE
                    s = stack.push(uint256{gpu_u64(mem_size)});
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    break;
                }
                ++pc;
                continue;
            }

            // -- Storage (0x54 - 0x55) ----------------------------------------
            if (op == 0x54 || op == 0x55)
            {
                ExecStatus s;
                uint256 a, b;

                if (op == 0x54) // SLOAD
                {
                    if (!consume_gas(GasCost::SLOAD))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0};
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.push(sload(a));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                }
                else // SSTORE: key=top, value=second
                {
                    if (!consume_gas(GasCost::SSTORE_SET))  // pessimistic: charge SET cost
                        return {ExecStatus::OutOfGas, gas_start, 0, 0};
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    sstore(a, b);  // sstore(key, value)
                }
                ++pc;
                continue;
            }

            // -- Control flow (0x56 - 0x5b) -----------------------------------
            if (op >= 0x56 && op <= 0x5b)
            {
                ExecStatus s;
                uint256 a, b;

                switch (op)
                {
                case 0x56: // JUMP
                    if (!consume_gas(GasCost::MID))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0};
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    if (a.w[1] | a.w[2] | a.w[3])
                        return {ExecStatus::InvalidJump, gas_start - gas, gas, 0};
                    {
                        gpu_u32 dest = gpu_u32(a.w[0]);
                        if (!is_jumpdest(dest))
                            return {ExecStatus::InvalidJump, gas_start - gas, gas, 0};
                        pc = dest;
                    }
                    continue;  // don't increment pc

                case 0x57: // JUMPI: dest=top, cond=second
                    if (!consume_gas(GasCost::HIGH))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0};
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    if (!iszero(b))  // b is the condition
                    {
                        if (a.w[1] | a.w[2] | a.w[3])
                            return {ExecStatus::InvalidJump, gas_start - gas, gas, 0};
                        gpu_u32 dest = gpu_u32(a.w[0]);
                        if (!is_jumpdest(dest))
                            return {ExecStatus::InvalidJump, gas_start - gas, gas, 0};
                        pc = dest;
                        continue;
                    }
                    ++pc;
                    continue;

                case 0x58: // PC
                    if (!consume_gas(GasCost::BASE))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0};
                    s = stack.push(uint256{gpu_u64(pc)});
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    ++pc;
                    continue;

                case 0x5a: // GAS
                    if (!consume_gas(GasCost::BASE))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0};
                    s = stack.push(uint256{gas});
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    ++pc;
                    continue;

                case 0x5b: // JUMPDEST
                    if (!consume_gas(GasCost::JUMPDEST))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0};
                    ++pc;
                    continue;
                }
            }

            // -- PUSH0 (0x5f) -------------------------------------------------
            if (op == 0x5f)
            {
                if (!consume_gas(GasCost::BASE))
                    return {ExecStatus::OutOfGas, gas_start, 0, 0};
                ExecStatus s = stack.push(uint256::zero());
                if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                ++pc;
                continue;
            }

            // -- PUSH1..PUSH32 (0x60 - 0x7f) ---------------------------------
            if (op >= 0x60 && op <= 0x7f)
            {
                if (!consume_gas(GasCost::VERYLOW))
                    return {ExecStatus::OutOfGas, gas_start, 0, 0};
                gpu_u32 num_bytes = op - 0x60 + 1;
                uint256 val = read_push_data(num_bytes);
                ExecStatus s = stack.push(val);
                if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                pc += 1 + num_bytes;
                continue;
            }

            // -- DUP1..DUP16 (0x80 - 0x8f) -----------------------------------
            if (op >= 0x80 && op <= 0x8f)
            {
                if (!consume_gas(GasCost::VERYLOW))
                    return {ExecStatus::OutOfGas, gas_start, 0, 0};
                gpu_u32 n = op - 0x80 + 1;
                ExecStatus s = stack.dup(n);
                if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                ++pc;
                continue;
            }

            // -- SWAP1..SWAP16 (0x90 - 0x9f) ---------------------------------
            if (op >= 0x90 && op <= 0x9f)
            {
                if (!consume_gas(GasCost::VERYLOW))
                    return {ExecStatus::OutOfGas, gas_start, 0, 0};
                gpu_u32 n = op - 0x90 + 1;
                ExecStatus s = stack.swap(n);
                if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                ++pc;
                continue;
            }

            // -- LOG0..LOG4 (0xa0 - 0xa4) ------------------------------------
            if (op >= 0xa0 && op <= 0xa4)
            {
                gpu_u32 num_topics = op - 0xa0;

                ExecStatus s;
                uint256 offset_val, size_val;
                s = stack.pop(offset_val); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                s = stack.pop(size_val); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};

                if (offset_val.w[1] | offset_val.w[2] | offset_val.w[3] ||
                    size_val.w[1] | size_val.w[2] | size_val.w[3])
                    return {ExecStatus::InvalidMemAccess, gas_start - gas, gas, 0};

                gpu_u32 data_off = gpu_u32(offset_val.w[0]);
                gpu_u32 data_sz  = gpu_u32(size_val.w[0]);

                // Gas: LOG_BASE + LOG_TOPIC * num_topics + LOG_DATA * data_size + memory expansion
                gpu_u64 log_gas = GasCost::LOG_BASE + GasCost::LOG_TOPIC * num_topics +
                                  GasCost::LOG_DATA * data_sz;
                if (!consume_gas(log_gas))
                    return {ExecStatus::OutOfGas, gas_start, 0, 0};

                if (data_sz > 0)
                {
                    gpu_u64 mem_cost = expand_memory(data_off, data_sz, mem);
                    if (mem_cost == ~gpu_u64(0) || !consume_gas(mem_cost))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0};
                }

                // Record the log entry.
                if (logs && log_count && *log_count < log_capacity)
                {
                    LogEntry& entry = logs[*log_count];
                    entry.num_topics = num_topics;
                    entry.data_offset = data_off;
                    entry.data_size = data_sz;
                    for (gpu_u32 t = 0; t < num_topics; ++t)
                    {
                        s = stack.pop(entry.topics[t]);
                        if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    }
                    ++(*log_count);
                }
                else
                {
                    // Pop topics even if we can't record.
                    uint256 dummy;
                    for (gpu_u32 t = 0; t < num_topics; ++t)
                    {
                        s = stack.pop(dummy);
                        if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                    }
                }

                ++pc;
                continue;
            }

            // -- RETURN (0xf3) ------------------------------------------------
            if (op == 0xf3)
            {
                ExecStatus s;
                uint256 offset_val, size_val;
                s = stack.pop(offset_val); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                s = stack.pop(size_val); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};

                if (offset_val.w[1] | offset_val.w[2] | offset_val.w[3] ||
                    size_val.w[1] | size_val.w[2] | size_val.w[3])
                    return {ExecStatus::InvalidMemAccess, gas_start - gas, gas, 0};

                gpu_u32 off = gpu_u32(offset_val.w[0]);
                gpu_u32 sz  = gpu_u32(size_val.w[0]);

                if (sz > 0)
                {
                    gpu_u64 mem_cost = expand_memory(off, sz, mem);
                    if (mem_cost == ~gpu_u64(0) || !consume_gas(mem_cost))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0};

                    gpu_u32 copy_sz = (sz > MAX_OUTPUT) ? MAX_OUTPUT : sz;
                    for (gpu_u32 i = 0; i < copy_sz; ++i)
                        output[i] = mem[off + i];
                    return {ExecStatus::Return, gas_start - gas, gas, copy_sz};
                }
                return {ExecStatus::Return, gas_start - gas, gas, 0};
            }

            // -- REVERT (0xfd) ------------------------------------------------
            if (op == 0xfd)
            {
                ExecStatus s;
                uint256 offset_val, size_val;
                s = stack.pop(offset_val); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};
                s = stack.pop(size_val); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0};

                if (offset_val.w[1] | offset_val.w[2] | offset_val.w[3] ||
                    size_val.w[1] | size_val.w[2] | size_val.w[3])
                    return {ExecStatus::InvalidMemAccess, gas_start - gas, gas, 0};

                gpu_u32 off = gpu_u32(offset_val.w[0]);
                gpu_u32 sz  = gpu_u32(size_val.w[0]);

                if (sz > 0)
                {
                    gpu_u64 mem_cost = expand_memory(off, sz, mem);
                    if (mem_cost == ~gpu_u64(0) || !consume_gas(mem_cost))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0};

                    gpu_u32 copy_sz = (sz > MAX_OUTPUT) ? MAX_OUTPUT : sz;
                    for (gpu_u32 i = 0; i < copy_sz; ++i)
                        output[i] = mem[off + i];
                    return {ExecStatus::Revert, gas_start - gas, gas, copy_sz};
                }
                return {ExecStatus::Revert, gas_start - gas, gas, 0};
            }

            // -- INVALID (0xfe) -----------------------------------------------
            if (op == 0xfe)
            {
                // Consume all remaining gas.
                gas = 0;
                return {ExecStatus::InvalidOpcode, gas_start, 0, 0};
            }

            // -- Unimplemented / unsupported opcode ---------------------------
            // CALL, CREATE, DELEGATECALL, STATICCALL, SELFDESTRUCT, etc.
            // These are caught by the host dispatcher and routed to CPU evmone.
            if (op == 0xf0 || op == 0xf1 || op == 0xf2 || op == 0xf4 ||
                op == 0xf5 || op == 0xfa || op == 0xff)
            {
                return {ExecStatus::CallNotSupported, gas_start - gas, gas, 0};
            }

            // Any other opcode is invalid.
            return {ExecStatus::InvalidOpcode, gas_start - gas, gas, 0};

        }  // while (pc < code_size)

        // Fell off the end of code -> implicit STOP.
        return {ExecStatus::Stop, gas_start - gas, gas, 0};
    }
};

}  // namespace evm::gpu::kernel
