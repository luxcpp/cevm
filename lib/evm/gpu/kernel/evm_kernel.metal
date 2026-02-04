// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file evm_kernel.metal
/// Metal compute shader for GPU EVM execution.
///
/// Each thread executes one transaction's EVM bytecode independently.
/// The interpreter runs entirely in thread-private memory (stack, locals).
/// Storage is accessed via device-memory buffers shared with the host.
///
/// Buffer layout:
///   [0] TxInput*    — per-transaction input descriptors
///   [1] uchar*      — contiguous bytecode + calldata blob
///   [2] TxOutput*   — per-transaction output descriptors
///   [3] uchar*      — per-transaction output data (MAX_OUTPUT_PER_TX each)
///   [4] uchar*      — per-transaction memory (MAX_MEMORY_PER_TX each)
///   [5] StorageEntry* — per-transaction storage (MAX_STORAGE_PER_TX entries each)
///   [6] uint*       — per-transaction storage counts
///   [7] uint*       — params: [0] = num_txs

#include <metal_stdlib>
using namespace metal;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-const-variable"

// -- Include the cross-platform headers inline --------------------------------
// Metal does not support #include of non-system headers at runtime, so we
// replicate the essential types and logic from the C++ headers directly.
// The host compiles these headers as C++; the Metal compiler gets this file.
//
// In practice, we'd use a build step to concatenate the headers. For now,
// the types are defined here to match uint256_gpu.hpp / evm_stack.hpp /
// evm_interpreter.hpp exactly.

// -- Platform types -----------------------------------------------------------

typedef ulong  gpu_u64;
typedef uint   gpu_u32;
typedef uchar  gpu_u8_t;
typedef long   gpu_i64;

// -- uint256 ------------------------------------------------------------------

struct uint256
{
    gpu_u64 w[4];  // w[0] = low, w[3] = high
};

static inline uint256 u256_zero()
{
    uint256 r;
    r.w[0] = 0; r.w[1] = 0; r.w[2] = 0; r.w[3] = 0;
    return r;
}

static inline uint256 u256_from(gpu_u64 lo)
{
    uint256 r;
    r.w[0] = lo; r.w[1] = 0; r.w[2] = 0; r.w[3] = 0;
    return r;
}

static inline uint256 u256_one() { return u256_from(1); }

static inline uint256 u256_max()
{
    uint256 r;
    gpu_u64 m = ~gpu_u64(0);
    r.w[0] = m; r.w[1] = m; r.w[2] = m; r.w[3] = m;
    return r;
}

static inline bool u256_iszero(uint256 a)
{
    return (a.w[0] | a.w[1] | a.w[2] | a.w[3]) == 0;
}

static inline bool u256_eq(uint256 a, uint256 b)
{
    return a.w[0] == b.w[0] && a.w[1] == b.w[1] && a.w[2] == b.w[2] && a.w[3] == b.w[3];
}

static inline bool u256_lt(uint256 a, uint256 b)
{
    if (a.w[3] != b.w[3]) return a.w[3] < b.w[3];
    if (a.w[2] != b.w[2]) return a.w[2] < b.w[2];
    if (a.w[1] != b.w[1]) return a.w[1] < b.w[1];
    return a.w[0] < b.w[0];
}

static inline bool u256_gt(uint256 a, uint256 b) { return u256_lt(b, a); }

static inline uint256 u256_add(uint256 a, uint256 b)
{
    uint256 r;
    gpu_u64 s0 = a.w[0] + b.w[0];
    gpu_u64 c0 = (s0 < a.w[0]) ? 1UL : 0UL;
    gpu_u64 s1 = a.w[1] + b.w[1] + c0;
    gpu_u64 c1 = (s1 < a.w[1] || (c0 && s1 == a.w[1])) ? 1UL : 0UL;
    gpu_u64 s2 = a.w[2] + b.w[2] + c1;
    gpu_u64 c2 = (s2 < a.w[2] || (c1 && s2 == a.w[2])) ? 1UL : 0UL;
    r.w[0] = s0;
    r.w[1] = s1;
    r.w[2] = s2;
    r.w[3] = a.w[3] + b.w[3] + c2;
    return r;
}

static inline uint256 u256_sub(uint256 a, uint256 b)
{
    uint256 r;
    gpu_u64 d0 = a.w[0] - b.w[0];
    gpu_u64 bw0 = (d0 > a.w[0]) ? 1UL : 0UL;
    gpu_u64 d1 = a.w[1] - b.w[1] - bw0;
    gpu_u64 bw1 = (a.w[1] < b.w[1] + bw0 || (bw0 && b.w[1] == ~0UL)) ? 1UL : 0UL;
    gpu_u64 d2 = a.w[2] - b.w[2] - bw1;
    gpu_u64 bw2 = (a.w[2] < b.w[2] + bw1 || (bw1 && b.w[2] == ~0UL)) ? 1UL : 0UL;
    r.w[0] = d0;
    r.w[1] = d1;
    r.w[2] = d2;
    r.w[3] = a.w[3] - b.w[3] - bw2;
    return r;
}

static inline uint256 u256_bitwise_and(uint256 a, uint256 b)
{
    uint256 r;
    r.w[0] = a.w[0] & b.w[0]; r.w[1] = a.w[1] & b.w[1];
    r.w[2] = a.w[2] & b.w[2]; r.w[3] = a.w[3] & b.w[3];
    return r;
}

static inline uint256 u256_bitwise_or(uint256 a, uint256 b)
{
    uint256 r;
    r.w[0] = a.w[0] | b.w[0]; r.w[1] = a.w[1] | b.w[1];
    r.w[2] = a.w[2] | b.w[2]; r.w[3] = a.w[3] | b.w[3];
    return r;
}

static inline uint256 u256_bitwise_xor(uint256 a, uint256 b)
{
    uint256 r;
    r.w[0] = a.w[0] ^ b.w[0]; r.w[1] = a.w[1] ^ b.w[1];
    r.w[2] = a.w[2] ^ b.w[2]; r.w[3] = a.w[3] ^ b.w[3];
    return r;
}

static inline uint256 u256_bitwise_not(uint256 a)
{
    uint256 r;
    r.w[0] = ~a.w[0]; r.w[1] = ~a.w[1]; r.w[2] = ~a.w[2]; r.w[3] = ~a.w[3];
    return r;
}

static inline uint256 u256_shl(gpu_u64 n, uint256 val)
{
    if (n >= 256) return u256_zero();
    if (n == 0) return val;
    uint256 r = u256_zero();
    uint ls = uint(n / 64);
    uint bs = uint(n % 64);
    for (uint i = ls; i < 4; ++i)
    {
        r.w[i] = val.w[i - ls] << bs;
        if (bs > 0 && i > ls)
            r.w[i] |= val.w[i - ls - 1] >> (64 - bs);
    }
    return r;
}

static inline uint256 u256_shr(gpu_u64 n, uint256 val)
{
    if (n >= 256) return u256_zero();
    if (n == 0) return val;
    uint256 r = u256_zero();
    uint ls = uint(n / 64);
    uint bs = uint(n % 64);
    for (uint i = 0; i + ls < 4; ++i)
    {
        r.w[i] = val.w[i + ls] >> bs;
        if (bs > 0 && i + ls + 1 < 4)
            r.w[i] |= val.w[i + ls + 1] << (64 - bs);
    }
    return r;
}

// 64x64 -> 128 multiplication (schoolbook half-word)
struct pair64 { gpu_u64 lo; gpu_u64 hi; };

static inline pair64 mul_wide(gpu_u64 a, gpu_u64 b)
{
    gpu_u64 a_lo = a & 0xFFFFFFFFUL;
    gpu_u64 a_hi = a >> 32;
    gpu_u64 b_lo = b & 0xFFFFFFFFUL;
    gpu_u64 b_hi = b >> 32;
    gpu_u64 p0 = a_lo * b_lo;
    gpu_u64 p1 = a_lo * b_hi;
    gpu_u64 p2 = a_hi * b_lo;
    gpu_u64 p3 = a_hi * b_hi;
    gpu_u64 mid = (p0 >> 32) + (p1 & 0xFFFFFFFFUL) + (p2 & 0xFFFFFFFFUL);
    gpu_u64 hi = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);
    gpu_u64 lo = (p0 & 0xFFFFFFFFUL) | ((mid & 0xFFFFFFFFUL) << 32);
    pair64 result;
    result.lo = lo;
    result.hi = hi;
    return result;
}

static inline uint256 u256_mul(uint256 a, uint256 b)
{
    uint256 r = u256_zero();
    gpu_u64 carry = 0;

    pair64 p00 = mul_wide(a.w[0], b.w[0]);
    r.w[0] = p00.lo;
    carry = p00.hi;

    pair64 p01 = mul_wide(a.w[0], b.w[1]);
    pair64 p10 = mul_wide(a.w[1], b.w[0]);
    gpu_u64 s1 = p01.lo + p10.lo + carry;
    gpu_u64 c1 = p01.hi + p10.hi;
    c1 += (s1 < p01.lo) ? 1UL : 0UL;
    gpu_u64 tmp = p10.lo + carry;
    c1 += (tmp < p10.lo) ? 1UL : 0UL;
    r.w[1] = s1;

    pair64 p02 = mul_wide(a.w[0], b.w[2]);
    pair64 p11 = mul_wide(a.w[1], b.w[1]);
    pair64 p20 = mul_wide(a.w[2], b.w[0]);
    gpu_u64 s2 = p02.lo + p11.lo;
    gpu_u64 c2 = (s2 < p02.lo) ? 1UL : 0UL;
    gpu_u64 s2b = s2 + p20.lo;
    c2 += (s2b < s2) ? 1UL : 0UL;
    gpu_u64 s2c = s2b + c1;
    c2 += (s2c < s2b) ? 1UL : 0UL;
    c2 += p02.hi + p11.hi + p20.hi;
    r.w[2] = s2c;

    r.w[3] = a.w[0] * b.w[3] + a.w[1] * b.w[2] + a.w[2] * b.w[1] + a.w[3] * b.w[0] + c2;
    return r;
}

// -- GPU buffer descriptors ---------------------------------------------------

struct TxInput
{
    uint code_offset;      // offset into blob buffer
    uint code_size;
    uint calldata_offset;  // offset into blob buffer
    uint calldata_size;
    ulong gas_limit;
    uint256 caller;
    uint256 address;
    uint256 value;
};

struct TxOutput
{
    uint status;       // 0=stop, 1=return, 2=revert, 3=oog, 4=error, 5=call_not_supported
    ulong gas_used;
    uint output_size;
};

// Storage entry for per-tx storage buffer.
struct StorageEntry
{
    uint256 key;
    uint256 value;
};

// -- Constants ----------------------------------------------------------------

constant uint MAX_MEMORY_PER_TX  = 65536;   // 64 KB per thread (conservative for GPU)
constant uint MAX_OUTPUT_PER_TX  = 1024;
constant uint MAX_STORAGE_PER_TX = 64;
constant uint STACK_LIMIT        = 1024;

// -- Gas costs ----------------------------------------------------------------

constant ulong GAS_VERYLOW  = 3;
constant ulong GAS_LOW      = 5;
constant ulong GAS_MID      = 8;
constant ulong GAS_HIGH     = 10;
constant ulong GAS_BASE     = 2;
constant ulong GAS_JUMPDEST = 1;
constant ulong GAS_SLOAD    = 2100;
constant ulong GAS_SSTORE   = 20000;
constant ulong GAS_MEMORY   = 3;
constant ulong GAS_EXP_BASE = 10;
constant ulong GAS_EXP_BYTE = 50;
constant ulong GAS_LOG_BASE = 375;
constant ulong GAS_LOG_DATA = 8;
constant ulong GAS_LOG_TOPIC = 375;
constant ulong GAS_COPY     = 3;

// -- Minimal interpreter inline -----------------------------------------------
// The full interpreter is in evm_interpreter.hpp (C++ host).
// This Metal version covers the core opcode subset for compute workloads.

kernel void evm_execute(
    device const TxInput*       inputs      [[buffer(0)]],
    device const uchar*         blob        [[buffer(1)]],
    device TxOutput*            outputs     [[buffer(2)]],
    device uchar*               out_data    [[buffer(3)]],
    device uchar*               mem_pool    [[buffer(4)]],
    device StorageEntry*        storage_pool [[buffer(5)]],
    device uint*                storage_counts [[buffer(6)]],
    device const uint*          params      [[buffer(7)]],
    uint tid                                [[thread_position_in_grid]])
{
    uint num_txs = params[0];
    if (tid >= num_txs)
        return;

    // Per-transaction buffers.
    device const TxInput& inp = inputs[tid];
    device TxOutput& out = outputs[tid];
    device uchar* mem = mem_pool + uint(tid) * MAX_MEMORY_PER_TX;
    device uchar* output = out_data + uint(tid) * MAX_OUTPUT_PER_TX;
    device StorageEntry* storage = storage_pool + uint(tid) * MAX_STORAGE_PER_TX;
    device uint& stor_count = storage_counts[tid];

    device const uchar* code = blob + inp.code_offset;
    uint code_size = inp.code_size;
    device const uchar* calldata = blob + inp.calldata_offset;
    uint calldata_size = inp.calldata_size;

    // Thread-local stack (lives in registers + stack memory).
    uint256 stack[STACK_LIMIT];
    uint sp = 0;  // stack pointer (number of items)

    ulong gas = inp.gas_limit;
    uint pc = 0;
    uint mem_size = 0;
    ulong gas_start = gas;

    // Main interpreter loop. Each opcode is handled inline with early returns
    // on error. Status codes match ExecStatus: 0=stop, 1=return, 2=revert,
    // 3=oog, 4=error, 5=call_not_supported.

    while (pc < code_size)
    {
        uchar op = code[pc];

        // STOP
        if (op == 0x00)
        {
            out.status = 0;  // Stop
            out.gas_used = gas_start - gas;
            out.output_size = 0;
            return;
        }

        // PUSH1..PUSH32
        if (op >= 0x60 && op <= 0x7f)
        {
            if (gas < GAS_VERYLOW) { out.status = 3; out.gas_used = gas_start; out.output_size = 0; return; }
            gas -= GAS_VERYLOW;
            if (sp >= STACK_LIMIT) { out.status = 4; out.gas_used = gas_start - gas; out.output_size = 0; return; }

            uint n = op - 0x60 + 1;
            uint256 val = u256_zero();
            uint start = pc + 1;
            for (uint i = 0; i < n && (start + i) < code_size; ++i)
            {
                uint byte_pos = n - 1 - i;
                uint limb = byte_pos / 8;
                uint shift = (byte_pos % 8) * 8;
                val.w[limb] |= gpu_u64(code[start + i]) << shift;
            }
            stack[sp++] = val;
            pc += 1 + n;
            continue;
        }

        // PUSH0
        if (op == 0x5f)
        {
            if (gas < GAS_BASE) { out.status = 3; out.gas_used = gas_start; out.output_size = 0; return; }
            gas -= GAS_BASE;
            if (sp >= STACK_LIMIT) { out.status = 4; out.gas_used = gas_start - gas; out.output_size = 0; return; }
            stack[sp++] = u256_zero();
            ++pc;
            continue;
        }

        // POP
        if (op == 0x50)
        {
            if (gas < GAS_BASE) { out.status = 3; out.gas_used = gas_start; out.output_size = 0; return; }
            gas -= GAS_BASE;
            if (sp == 0) { out.status = 4; out.gas_used = gas_start - gas; out.output_size = 0; return; }
            --sp;
            ++pc;
            continue;
        }

        // DUP1..DUP16
        if (op >= 0x80 && op <= 0x8f)
        {
            if (gas < GAS_VERYLOW) { out.status = 3; out.gas_used = gas_start; out.output_size = 0; return; }
            gas -= GAS_VERYLOW;
            uint n = op - 0x80 + 1;
            if (n > sp || sp >= STACK_LIMIT) { out.status = 4; out.gas_used = gas_start - gas; out.output_size = 0; return; }
            stack[sp] = stack[sp - n];
            ++sp;
            ++pc;
            continue;
        }

        // SWAP1..SWAP16
        if (op >= 0x90 && op <= 0x9f)
        {
            if (gas < GAS_VERYLOW) { out.status = 3; out.gas_used = gas_start; out.output_size = 0; return; }
            gas -= GAS_VERYLOW;
            uint n = op - 0x90 + 1;
            if (n >= sp) { out.status = 4; out.gas_used = gas_start - gas; out.output_size = 0; return; }
            uint idx = sp - 1 - n;
            uint256 tmp = stack[sp - 1];
            stack[sp - 1] = stack[idx];
            stack[idx] = tmp;
            ++pc;
            continue;
        }

        // ADD
        if (op == 0x01)
        {
            if (gas < GAS_VERYLOW) { out.status = 3; out.gas_used = gas_start; out.output_size = 0; return; }
            gas -= GAS_VERYLOW;
            if (sp < 2) { out.status = 4; out.gas_used = gas_start - gas; out.output_size = 0; return; }
            uint256 a = stack[--sp];
            uint256 b = stack[--sp];
            stack[sp++] = u256_add(a, b);
            ++pc;
            continue;
        }

        // MUL
        if (op == 0x02)
        {
            if (gas < GAS_LOW) { out.status = 3; out.gas_used = gas_start; out.output_size = 0; return; }
            gas -= GAS_LOW;
            if (sp < 2) { out.status = 4; out.gas_used = gas_start - gas; out.output_size = 0; return; }
            uint256 a = stack[--sp];
            uint256 b = stack[--sp];
            stack[sp++] = u256_mul(a, b);
            ++pc;
            continue;
        }

        // SUB
        if (op == 0x03)
        {
            if (gas < GAS_VERYLOW) { out.status = 3; out.gas_used = gas_start; out.output_size = 0; return; }
            gas -= GAS_VERYLOW;
            if (sp < 2) { out.status = 4; out.gas_used = gas_start - gas; out.output_size = 0; return; }
            uint256 a = stack[--sp];
            uint256 b = stack[--sp];
            stack[sp++] = u256_sub(a, b);
            ++pc;
            continue;
        }

        // ISZERO
        if (op == 0x15)
        {
            if (gas < GAS_VERYLOW) { out.status = 3; out.gas_used = gas_start; out.output_size = 0; return; }
            gas -= GAS_VERYLOW;
            if (sp < 1) { out.status = 4; out.gas_used = gas_start - gas; out.output_size = 0; return; }
            uint256 a = stack[--sp];
            stack[sp++] = u256_iszero(a) ? u256_one() : u256_zero();
            ++pc;
            continue;
        }

        // AND, OR, XOR
        if (op >= 0x16 && op <= 0x18)
        {
            if (gas < GAS_VERYLOW) { out.status = 3; out.gas_used = gas_start; out.output_size = 0; return; }
            gas -= GAS_VERYLOW;
            if (sp < 2) { out.status = 4; out.gas_used = gas_start - gas; out.output_size = 0; return; }
            uint256 a = stack[--sp];
            uint256 b = stack[--sp];
            if (op == 0x16) stack[sp++] = u256_bitwise_and(a, b);
            else if (op == 0x17) stack[sp++] = u256_bitwise_or(a, b);
            else stack[sp++] = u256_bitwise_xor(a, b);
            ++pc;
            continue;
        }

        // NOT
        if (op == 0x19)
        {
            if (gas < GAS_VERYLOW) { out.status = 3; out.gas_used = gas_start; out.output_size = 0; return; }
            gas -= GAS_VERYLOW;
            if (sp < 1) { out.status = 4; out.gas_used = gas_start - gas; out.output_size = 0; return; }
            uint256 a = stack[--sp];
            stack[sp++] = u256_bitwise_not(a);
            ++pc;
            continue;
        }

        // LT, GT, EQ
        if (op == 0x10 || op == 0x11 || op == 0x14)
        {
            if (gas < GAS_VERYLOW) { out.status = 3; out.gas_used = gas_start; out.output_size = 0; return; }
            gas -= GAS_VERYLOW;
            if (sp < 2) { out.status = 4; out.gas_used = gas_start - gas; out.output_size = 0; return; }
            uint256 a = stack[--sp];
            uint256 b = stack[--sp];
            bool result = false;
            if (op == 0x10) result = u256_lt(a, b);
            else if (op == 0x11) result = u256_gt(a, b);
            else result = u256_eq(a, b);
            stack[sp++] = result ? u256_one() : u256_zero();
            ++pc;
            continue;
        }

        // MLOAD
        if (op == 0x51)
        {
            if (gas < GAS_VERYLOW) { out.status = 3; out.gas_used = gas_start; out.output_size = 0; return; }
            gas -= GAS_VERYLOW;
            if (sp < 1) { out.status = 4; out.gas_used = gas_start - gas; out.output_size = 0; return; }
            uint256 off_val = stack[--sp];
            if (off_val.w[1] | off_val.w[2] | off_val.w[3] || off_val.w[0] + 32 > MAX_MEMORY_PER_TX)
            {
                out.status = 4; out.gas_used = gas_start - gas; out.output_size = 0; return;
            }
            uint off = uint(off_val.w[0]);
            // Expand memory if needed.
            uint end = off + 32;
            uint new_words = (end + 31) / 32;
            if (new_words * 32 > mem_size)
            {
                uint old_words = mem_size / 32;
                ulong cost = GAS_MEMORY * new_words + (ulong(new_words) * new_words) / 512
                           - GAS_MEMORY * old_words - (ulong(old_words) * old_words) / 512;
                if (gas < cost) { out.status = 3; out.gas_used = gas_start; out.output_size = 0; return; }
                gas -= cost;
                for (uint i = mem_size; i < new_words * 32; ++i) mem[i] = 0;
                mem_size = new_words * 32;
            }
            // Read big-endian uint256 from memory.
            uint256 r = u256_zero();
            for (int limb = 3; limb >= 0; --limb)
            {
                gpu_u64 v = 0;
                int start = (3 - limb) * 8;
                for (int byte_i = 0; byte_i < 8; ++byte_i)
                    v = (v << 8) | gpu_u64(mem[off + start + byte_i]);
                r.w[limb] = v;
            }
            stack[sp++] = r;
            ++pc;
            continue;
        }

        // MSTORE
        if (op == 0x52)
        {
            if (gas < GAS_VERYLOW) { out.status = 3; out.gas_used = gas_start; out.output_size = 0; return; }
            gas -= GAS_VERYLOW;
            if (sp < 2) { out.status = 4; out.gas_used = gas_start - gas; out.output_size = 0; return; }
            uint256 off_val = stack[--sp];
            uint256 val = stack[--sp];
            if (off_val.w[1] | off_val.w[2] | off_val.w[3] || off_val.w[0] + 32 > MAX_MEMORY_PER_TX)
            {
                out.status = 4; out.gas_used = gas_start - gas; out.output_size = 0; return;
            }
            uint off = uint(off_val.w[0]);
            uint end = off + 32;
            uint new_words = (end + 31) / 32;
            if (new_words * 32 > mem_size)
            {
                uint old_words = mem_size / 32;
                ulong cost = GAS_MEMORY * new_words + (ulong(new_words) * new_words) / 512
                           - GAS_MEMORY * old_words - (ulong(old_words) * old_words) / 512;
                if (gas < cost) { out.status = 3; out.gas_used = gas_start; out.output_size = 0; return; }
                gas -= cost;
                for (uint i = mem_size; i < new_words * 32; ++i) mem[i] = 0;
                mem_size = new_words * 32;
            }
            // Write big-endian uint256 to memory.
            for (int limb = 3; limb >= 0; --limb)
            {
                gpu_u64 v = val.w[limb];
                int start = (3 - limb) * 8;
                for (int byte_i = 7; byte_i >= 0; --byte_i)
                {
                    mem[off + start + byte_i] = uchar(v & 0xFF);
                    v >>= 8;
                }
            }
            ++pc;
            continue;
        }

        // JUMP
        if (op == 0x56)
        {
            if (gas < GAS_MID) { out.status = 3; out.gas_used = gas_start; out.output_size = 0; return; }
            gas -= GAS_MID;
            if (sp < 1) { out.status = 4; out.gas_used = gas_start - gas; out.output_size = 0; return; }
            uint256 dest_val = stack[--sp];
            if (dest_val.w[1] | dest_val.w[2] | dest_val.w[3] || dest_val.w[0] >= code_size)
            {
                out.status = 4; out.gas_used = gas_start - gas; out.output_size = 0; return;
            }
            uint dest = uint(dest_val.w[0]);
            if (code[dest] != 0x5b)
            {
                out.status = 4; out.gas_used = gas_start - gas; out.output_size = 0; return;
            }
            pc = dest;
            continue;
        }

        // JUMPI
        if (op == 0x57)
        {
            if (gas < GAS_HIGH) { out.status = 3; out.gas_used = gas_start; out.output_size = 0; return; }
            gas -= GAS_HIGH;
            if (sp < 2) { out.status = 4; out.gas_used = gas_start - gas; out.output_size = 0; return; }
            uint256 dest_val = stack[--sp];
            uint256 cond = stack[--sp];
            if (!u256_iszero(cond))
            {
                if (dest_val.w[1] | dest_val.w[2] | dest_val.w[3] || dest_val.w[0] >= code_size)
                {
                    out.status = 4; out.gas_used = gas_start - gas; out.output_size = 0; return;
                }
                uint dest = uint(dest_val.w[0]);
                if (code[dest] != 0x5b)
                {
                    out.status = 4; out.gas_used = gas_start - gas; out.output_size = 0; return;
                }
                pc = dest;
                continue;
            }
            ++pc;
            continue;
        }

        // JUMPDEST
        if (op == 0x5b)
        {
            if (gas < GAS_JUMPDEST) { out.status = 3; out.gas_used = gas_start; out.output_size = 0; return; }
            gas -= GAS_JUMPDEST;
            ++pc;
            continue;
        }

        // PC
        if (op == 0x58)
        {
            if (gas < GAS_BASE) { out.status = 3; out.gas_used = gas_start; out.output_size = 0; return; }
            gas -= GAS_BASE;
            if (sp >= STACK_LIMIT) { out.status = 4; out.gas_used = gas_start - gas; out.output_size = 0; return; }
            stack[sp++] = u256_from(gpu_u64(pc));
            ++pc;
            continue;
        }

        // GAS
        if (op == 0x5a)
        {
            if (gas < GAS_BASE) { out.status = 3; out.gas_used = gas_start; out.output_size = 0; return; }
            gas -= GAS_BASE;
            if (sp >= STACK_LIMIT) { out.status = 4; out.gas_used = gas_start - gas; out.output_size = 0; return; }
            stack[sp++] = u256_from(gas);
            ++pc;
            continue;
        }

        // SLOAD
        if (op == 0x54)
        {
            if (gas < GAS_SLOAD) { out.status = 3; out.gas_used = gas_start; out.output_size = 0; return; }
            gas -= GAS_SLOAD;
            if (sp < 1) { out.status = 4; out.gas_used = gas_start - gas; out.output_size = 0; return; }
            uint256 slot = stack[--sp];
            uint256 val = u256_zero();
            for (uint i = stor_count; i > 0; --i)
            {
                if (u256_eq(storage[i - 1].key, slot))
                {
                    val = storage[i - 1].value;
                    break;
                }
            }
            stack[sp++] = val;
            ++pc;
            continue;
        }

        // SSTORE
        if (op == 0x55)
        {
            if (gas < GAS_SSTORE) { out.status = 3; out.gas_used = gas_start; out.output_size = 0; return; }
            gas -= GAS_SSTORE;
            if (sp < 2) { out.status = 4; out.gas_used = gas_start - gas; out.output_size = 0; return; }
            uint256 slot = stack[--sp];
            uint256 val = stack[--sp];
            bool found = false;
            for (uint i = stor_count; i > 0; --i)
            {
                if (u256_eq(storage[i - 1].key, slot))
                {
                    storage[i - 1].value = val;
                    found = true;
                    break;
                }
            }
            if (!found && stor_count < MAX_STORAGE_PER_TX)
            {
                storage[stor_count].key = slot;
                storage[stor_count].value = val;
                stor_count++;
            }
            ++pc;
            continue;
        }

        // RETURN
        if (op == 0xf3)
        {
            if (sp < 2) { out.status = 4; out.gas_used = gas_start - gas; out.output_size = 0; return; }
            uint256 off_val = stack[--sp];
            uint256 sz_val = stack[--sp];
            uint off = uint(off_val.w[0]);
            uint sz = uint(sz_val.w[0]);
            uint copy_sz = (sz > MAX_OUTPUT_PER_TX) ? MAX_OUTPUT_PER_TX : sz;
            for (uint i = 0; i < copy_sz && off + i < mem_size; ++i)
                output[i] = mem[off + i];
            out.status = 1;  // Return
            out.gas_used = gas_start - gas;
            out.output_size = copy_sz;
            return;
        }

        // REVERT
        if (op == 0xfd)
        {
            if (sp < 2) { out.status = 4; out.gas_used = gas_start - gas; out.output_size = 0; return; }
            uint256 off_val = stack[--sp];
            uint256 sz_val = stack[--sp];
            uint off = uint(off_val.w[0]);
            uint sz = uint(sz_val.w[0]);
            uint copy_sz = (sz > MAX_OUTPUT_PER_TX) ? MAX_OUTPUT_PER_TX : sz;
            for (uint i = 0; i < copy_sz && off + i < mem_size; ++i)
                output[i] = mem[off + i];
            out.status = 2;  // Revert
            out.gas_used = gas_start - gas;
            out.output_size = copy_sz;
            return;
        }

        // INVALID
        if (op == 0xfe)
        {
            out.status = 4;
            out.gas_used = gas_start;
            out.output_size = 0;
            return;
        }

        // CALL/CREATE family — signal CPU fallback
        if (op == 0xf0 || op == 0xf1 || op == 0xf2 || op == 0xf4 ||
            op == 0xf5 || op == 0xfa || op == 0xff)
        {
            out.status = 5;  // CallNotSupported
            out.gas_used = gas_start - gas;
            out.output_size = 0;
            return;
        }

        // ADDRESS (0x30)
        if (op == 0x30)
        {
            if (gas < GAS_BASE) { out.status = 3; out.gas_used = gas_start; out.output_size = 0; return; }
            gas -= GAS_BASE;
            if (sp >= STACK_LIMIT) { out.status = 4; out.gas_used = gas_start - gas; out.output_size = 0; return; }
            stack[sp++] = inp.address;
            ++pc;
            continue;
        }

        // CALLER (0x33)
        if (op == 0x33)
        {
            if (gas < GAS_BASE) { out.status = 3; out.gas_used = gas_start; out.output_size = 0; return; }
            gas -= GAS_BASE;
            if (sp >= STACK_LIMIT) { out.status = 4; out.gas_used = gas_start - gas; out.output_size = 0; return; }
            stack[sp++] = inp.caller;
            ++pc;
            continue;
        }

        // CALLVALUE (0x34)
        if (op == 0x34)
        {
            if (gas < GAS_BASE) { out.status = 3; out.gas_used = gas_start; out.output_size = 0; return; }
            gas -= GAS_BASE;
            if (sp >= STACK_LIMIT) { out.status = 4; out.gas_used = gas_start - gas; out.output_size = 0; return; }
            stack[sp++] = inp.value;
            ++pc;
            continue;
        }

        // CALLDATASIZE (0x36)
        if (op == 0x36)
        {
            if (gas < GAS_BASE) { out.status = 3; out.gas_used = gas_start; out.output_size = 0; return; }
            gas -= GAS_BASE;
            if (sp >= STACK_LIMIT) { out.status = 4; out.gas_used = gas_start - gas; out.output_size = 0; return; }
            stack[sp++] = u256_from(gpu_u64(calldata_size));
            ++pc;
            continue;
        }

        // CALLDATALOAD (0x35)
        if (op == 0x35)
        {
            if (gas < GAS_VERYLOW) { out.status = 3; out.gas_used = gas_start; out.output_size = 0; return; }
            gas -= GAS_VERYLOW;
            if (sp < 1) { out.status = 4; out.gas_used = gas_start - gas; out.output_size = 0; return; }
            uint256 off_val = stack[--sp];
            uint256 result = u256_zero();
            if (!off_val.w[1] && !off_val.w[2] && !off_val.w[3] && off_val.w[0] < calldata_size)
            {
                uint off = uint(off_val.w[0]);
                for (uint i = 0; i < 32; ++i)
                {
                    uint src = off + i;
                    uchar byte_val = (src < calldata_size) ? calldata[src] : 0;
                    uint pos_from_right = 31 - i;
                    uint limb = pos_from_right / 8;
                    uint shift = (pos_from_right % 8) * 8;
                    result.w[limb] |= gpu_u64(byte_val) << shift;
                }
            }
            stack[sp++] = result;
            ++pc;
            continue;
        }

        // MSIZE (0x59)
        if (op == 0x59)
        {
            if (gas < GAS_BASE) { out.status = 3; out.gas_used = gas_start; out.output_size = 0; return; }
            gas -= GAS_BASE;
            if (sp >= STACK_LIMIT) { out.status = 4; out.gas_used = gas_start - gas; out.output_size = 0; return; }
            stack[sp++] = u256_from(gpu_u64(mem_size));
            ++pc;
            continue;
        }

        // Unrecognized opcode — error.
        out.status = 4;
        out.gas_used = gas_start - gas;
        out.output_size = 0;
        return;
    }

    // Fell off end of code -> implicit STOP.
    out.status = 0;
    out.gas_used = gas_start - gas;
    out.output_size = 0;
}

#pragma clang diagnostic pop
