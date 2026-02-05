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

/// 256-bit multiplication (low 256 bits of 512-bit product).
/// Uses the same proven accumulator loop as mulmod's 512-bit product,
/// but only computes the low 4 limbs (i+j < 4). Products where i+j >= 4
/// overflow past 256 bits and are discarded per EVM spec.
static inline uint256 u256_mul(uint256 a, uint256 b)
{
    gpu_u64 r[4] = {0, 0, 0, 0};

    for (uint i = 0; i < 4; ++i)
    {
        gpu_u64 carry = 0;
        for (uint j = 0; j < 4; ++j)
        {
            if (i + j >= 4)
                break;  // overflow past 256 bits, discard
            pair64 p = mul_wide(a.w[i], b.w[j]);
            gpu_u64 s = r[i + j] + p.lo;
            gpu_u64 c = (s < r[i + j]) ? 1UL : 0UL;
            s += carry;
            c += (s < carry) ? 1UL : 0UL;
            r[i + j] = s;
            carry = p.hi + c;
        }
        // carry propagates into r[i + j] where j = 4-i or beyond,
        // but those limbs are past 256 bits and discarded.
    }

    uint256 result;
    result.w[0] = r[0]; result.w[1] = r[1]; result.w[2] = r[2]; result.w[3] = r[3];
    return result;
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
    ulong gas_refund;  // EIP-2200/3529 refund counter
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
constant ulong GAS_SLOAD       = 2100;
constant ulong GAS_SSTORE_SET    = 20000;
constant ulong GAS_SSTORE_RESET  = 2900;
constant ulong GAS_SSTORE_NOOP   = 100;   // EIP-2200: no-op write (same value)
constant ulong GAS_SSTORE_REFUND = 4800;
constant ulong GAS_MEMORY   = 3;
constant ulong GAS_EXP_BASE = 10;
constant ulong GAS_EXP_BYTE = 50;
constant ulong GAS_LOG_BASE = 375;
constant ulong GAS_LOG_DATA = 8;
constant ulong GAS_LOG_TOPIC = 375;
constant ulong GAS_COPY     = 3;

// -- JUMPDEST validation ------------------------------------------------------
// Must scan bytecode from offset 0, skipping PUSH data bytes, to confirm the
// target is an actual JUMPDEST opcode and not embedded inside PUSH data.

static inline bool is_valid_jumpdest(device const uchar* code, uint code_size, uint target)
{
    if (target >= code_size || code[target] != 0x5b)
        return false;
    uint i = 0;
    while (i < target)
    {
        uchar op = code[i];
        if (op >= 0x60 && op <= 0x7f)  // PUSH1..PUSH32
            i += (op - 0x60 + 2);      // skip opcode + data bytes
        else
            i++;
    }
    return i == target;  // must land exactly on the target
}

// -- uint256 division (binary long division) ----------------------------------

struct divmod_result { uint256 quot; uint256 rem; };

static inline uint clz64_metal(gpu_u64 x)
{
    if (x == 0) return 64;
    uint n = 0;
    if ((x & 0xFFFFFFFF00000000UL) == 0) { n += 32; x <<= 32; }
    if ((x & 0xFFFF000000000000UL) == 0) { n += 16; x <<= 16; }
    if ((x & 0xFF00000000000000UL) == 0) { n +=  8; x <<=  8; }
    if ((x & 0xF000000000000000UL) == 0) { n +=  4; x <<=  4; }
    if ((x & 0xC000000000000000UL) == 0) { n +=  2; x <<=  2; }
    if ((x & 0x8000000000000000UL) == 0) { n +=  1; }
    return n;
}

static inline uint clz256_metal(uint256 x)
{
    if (x.w[3]) return clz64_metal(x.w[3]);
    if (x.w[2]) return 64 + clz64_metal(x.w[2]);
    if (x.w[1]) return 128 + clz64_metal(x.w[1]);
    return 192 + clz64_metal(x.w[0]);
}

static inline divmod_result u256_divmod(uint256 a, uint256 b)
{
    if (u256_iszero(b))
        return {u256_zero(), u256_zero()};
    if (u256_lt(a, b))
        return {u256_zero(), a};
    if (u256_eq(a, b))
        return {u256_one(), u256_zero()};

    uint shift = clz256_metal(b) - clz256_metal(a);
    uint256 divisor = u256_shl(gpu_u64(shift), b);
    uint256 quotient = u256_zero();
    uint256 remainder = a;

    for (uint i = 0; i <= shift; ++i)
    {
        quotient = u256_shl(1, quotient);
        if (!u256_lt(remainder, divisor))
        {
            remainder = u256_sub(remainder, divisor);
            quotient.w[0] |= 1;
        }
        divisor = u256_shr(1, divisor);
    }
    return {quotient, remainder};
}

static inline uint256 u256_div(uint256 a, uint256 b)
{
    return u256_divmod(a, b).quot;
}

static inline uint256 u256_mod(uint256 a, uint256 b)
{
    return u256_divmod(a, b).rem;
}

// -- Signed division/modulo ---------------------------------------------------

static inline uint256 u256_negate(uint256 x)
{
    return u256_add(u256_bitwise_not(x), u256_one());
}

static inline uint256 u256_sdiv(uint256 a, uint256 b)
{
    if (u256_iszero(b)) return u256_zero();
    bool a_neg = (a.w[3] >> 63) != 0;
    bool b_neg = (b.w[3] >> 63) != 0;
    uint256 abs_a = a_neg ? u256_negate(a) : a;
    uint256 abs_b = b_neg ? u256_negate(b) : b;
    uint256 q = u256_div(abs_a, abs_b);
    if (a_neg != b_neg) q = u256_negate(q);
    return q;
}

static inline uint256 u256_smod(uint256 a, uint256 b)
{
    if (u256_iszero(b)) return u256_zero();
    bool a_neg = (a.w[3] >> 63) != 0;
    bool b_neg = (b.w[3] >> 63) != 0;
    uint256 abs_a = a_neg ? u256_negate(a) : a;
    uint256 abs_b = b_neg ? u256_negate(b) : b;
    uint256 r = u256_mod(abs_a, abs_b);
    if (a_neg && !u256_iszero(r)) r = u256_negate(r);
    return r;
}

// -- ADDMOD: (a + b) % m with 320-bit intermediate ---------------------------
//
// Computes the full 320-bit sum (5 limbs), then reduces mod m using the same
// bit-by-bit shift-subtract approach that MULMOD uses for 512 bits. This avoids
// the double-carry bug where subtracting m via u256_add(sum, negate(m)) can
// overflow a second time, losing the carry.

static inline uint256 u256_addmod(uint256 a, uint256 b, uint256 m)
{
    if (u256_iszero(m)) return u256_zero();

    // Full 320-bit addition into 5 limbs.
    gpu_u64 s0 = a.w[0] + b.w[0];
    gpu_u64 c0 = (s0 < a.w[0]) ? 1UL : 0UL;
    gpu_u64 s1 = a.w[1] + b.w[1] + c0;
    gpu_u64 c1 = (s1 < a.w[1] || (c0 && s1 == a.w[1])) ? 1UL : 0UL;
    gpu_u64 s2 = a.w[2] + b.w[2] + c1;
    gpu_u64 c2 = (s2 < a.w[2] || (c1 && s2 == a.w[2])) ? 1UL : 0UL;
    gpu_u64 s3 = a.w[3] + b.w[3] + c2;
    gpu_u64 c3 = (s3 < a.w[3] || (c2 && s3 == a.w[3])) ? 1UL : 0UL;

    // No overflow: standard 256-bit mod.
    if (c3 == 0)
    {
        uint256 sum; sum.w[0] = s0; sum.w[1] = s1; sum.w[2] = s2; sum.w[3] = s3;
        return u256_mod(sum, m);
    }

    // Overflow: reduce the 320-bit value {c3, s3, s2, s1, s0} mod m.
    // Use bit-by-bit reduction: for each bit from MSB (bit 256) to LSB (bit 0),
    // shift the accumulator left by 1, add the bit, subtract m if >= m.
    // Since c3 is 0 or 1, the 320-bit value has at most 257 significant bits.
    gpu_u64 r[5] = {s0, s1, s2, s3, c3};
    uint256 result = u256_zero();
    for (int bit = 256; bit >= 0; --bit)
    {
        // Shift result left by 1.
        result = u256_shl(1, result);
        // Add the current bit of the 320-bit sum.
        uint limb = uint(bit) / 64;
        uint pos  = uint(bit) % 64;
        if ((r[limb] >> pos) & 1)
            result.w[0] |= 1;
        // Reduce: if result >= m, subtract m.
        if (!u256_lt(result, m))
            result = u256_sub(result, m);
    }
    return result;
}

// -- MULMOD: (a * b) % m with 512-bit intermediate ---------------------------

static inline uint256 u256_mulmod(uint256 a, uint256 b, uint256 m)
{
    if (u256_iszero(m)) return u256_zero();

    gpu_u64 r[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint i = 0; i < 4; ++i)
    {
        gpu_u64 carry = 0;
        for (uint j = 0; j < 4; ++j)
        {
            pair64 p = mul_wide(a.w[i], b.w[j]);
            gpu_u64 s = r[i + j] + p.lo;
            gpu_u64 c = (s < r[i + j]) ? 1UL : 0UL;
            s += carry;
            c += (s < carry) ? 1UL : 0UL;
            r[i + j] = s;
            carry = p.hi + c;
        }
        if (i + 4 < 8) r[i + 4] += carry;
    }

    uint256 result = u256_zero();
    for (int bit = 511; bit >= 0; --bit)
    {
        result = u256_shl(1, result);
        uint limb = uint(bit) / 64;
        uint pos  = uint(bit) % 64;
        if ((r[limb] >> pos) & 1)
            result.w[0] |= 1;
        if (!u256_lt(result, m))
            result = u256_sub(result, m);
    }
    return result;
}

// -- EXP: base^exponent mod 2^256 (square-and-multiply) ----------------------

static inline uint256 u256_exp(uint256 base, uint256 exponent)
{
    if (u256_iszero(exponent)) return u256_one();
    uint256 result = u256_one();
    uint256 b = base;
    uint256 e = exponent;
    while (!u256_iszero(e))
    {
        if (e.w[0] & 1)
            result = u256_mul(result, b);
        e = u256_shr(1, e);
        if (!u256_iszero(e))
            b = u256_mul(b, b);
    }
    return result;
}

// -- SIGNEXTEND ---------------------------------------------------------------

static inline uint256 u256_signextend(uint256 b_val, uint256 x)
{
    if (b_val.w[1] | b_val.w[2] | b_val.w[3]) return x;
    gpu_u64 b = b_val.w[0];
    if (b >= 31) return x;

    gpu_u64 sign_bit = b * 8 + 7;
    uint limb = uint(sign_bit / 64);
    uint pos  = uint(sign_bit % 64);
    bool negative = ((x.w[limb] >> pos) & 1) != 0;

    uint256 one_shifted = u256_shl(sign_bit + 1, u256_one());
    uint256 mask = u256_bitwise_not(u256_sub(one_shifted, u256_one()));

    if (negative)
        return u256_bitwise_or(x, mask);
    else
        return u256_bitwise_and(x, u256_bitwise_not(mask));
}

// -- Signed comparison --------------------------------------------------------

static inline bool u256_slt(uint256 a, uint256 b)
{
    bool a_neg = (a.w[3] >> 63) != 0;
    bool b_neg = (b.w[3] >> 63) != 0;
    if (a_neg != b_neg) return a_neg;
    return u256_lt(a, b);
}

static inline bool u256_sgt(uint256 a, uint256 b)
{
    return u256_slt(b, a);
}

// -- BYTE opcode: extract byte at position i (0 = MSB) -----------------------

static inline uint256 u256_byte_at(uint256 val, uint256 pos)
{
    if (pos.w[1] | pos.w[2] | pos.w[3]) return u256_zero();
    gpu_u64 i = pos.w[0];
    if (i >= 32) return u256_zero();
    uint byte_from_right = uint(31 - i);
    uint limb = byte_from_right / 8;
    uint shift = (byte_from_right % 8) * 8;
    gpu_u64 b = (val.w[limb] >> shift) & 0xFFUL;
    return u256_from(b);
}

// -- SAR: arithmetic shift right (sign-extending) ----------------------------

static inline uint256 u256_sar(gpu_u64 n, uint256 val)
{
    bool negative = (val.w[3] >> 63) != 0;
    if (n >= 256)
        return negative ? u256_max() : u256_zero();
    uint256 r = u256_shr(n, val);
    if (negative && n > 0)
    {
        uint256 mask = u256_bitwise_not(u256_shr(n, u256_max()));
        r = u256_bitwise_or(r, mask);
    }
    return r;
}

// -- EXP byte count helper ---------------------------------------------------

static inline uint u256_byte_length(uint256 x)
{
    if (u256_iszero(x)) return 0;
    uint bits = 256 - clz256_metal(x);
    return (bits + 7) / 8;
}

// -- EIP-2200 original-value tracking -----------------------------------------
// Tracks the value each storage slot had at the start of the transaction.
// Used by SSTORE to determine correct gas cost.
// Thread-local: MAX_STORAGE_PER_TX entries per tx, linear scan (small N on GPU).

struct OriginalEntry
{
    uint256 key;
    uint256 value;
    bool    valid;
};

/// Look up original value for a slot. Returns {value, found}.
static inline bool original_value_lookup(
    thread OriginalEntry* originals, uint orig_count, uint256 slot, thread uint256& out_value)
{
    for (uint i = 0; i < orig_count; ++i)
    {
        if (originals[i].valid && u256_eq(originals[i].key, slot))
        {
            out_value = originals[i].value;
            return true;
        }
    }
    out_value = u256_zero();
    return false;
}

/// Record original value for a slot (only if not already recorded).
static inline void original_value_record(
    thread OriginalEntry* originals, thread uint& orig_count, uint256 slot, uint256 value)
{
    for (uint i = 0; i < orig_count; ++i)
    {
        if (originals[i].valid && u256_eq(originals[i].key, slot))
            return;  // already recorded
    }
    if (orig_count < MAX_STORAGE_PER_TX)
    {
        originals[orig_count].key   = slot;
        originals[orig_count].value = value;
        originals[orig_count].valid = true;
        orig_count++;
    }
}

/// Compute EIP-2200 SSTORE gas cost.
/// original = value at start of tx, current = value in storage now, new_val = value being written.
/// SSTORE gas cost + refund per EIP-2200/EIP-3529.
/// Returns the gas to charge. Adds to refund_counter for clearing/restoring storage.
static inline ulong sstore_gas_eip2200(uint256 original, uint256 current, uint256 new_val,
                                        thread ulong& refund_counter)
{
    // No-op: writing the same value that's already there.
    if (u256_eq(new_val, current))
        return GAS_SSTORE_NOOP;  // 100 (warm access cost)

    // First modification to this slot in the tx (current == original).
    if (u256_eq(original, current))
    {
        if (u256_iszero(original))
            return GAS_SSTORE_SET;    // 0 -> non-zero: 20000

        // non-zero -> different non-zero or zero
        if (u256_iszero(new_val))
            refund_counter += GAS_SSTORE_REFUND;  // 4800: clearing storage
        return GAS_SSTORE_RESET;      // 2900
    }

    // Subsequent modification (current != original): cheap.
    // EIP-2200 refund adjustments for restore/re-clear scenarios:
    if (!u256_iszero(original))
    {
        if (u256_iszero(current))
            refund_counter -= GAS_SSTORE_REFUND;  // undo clear refund (was going to 0, now changing again)
        else if (u256_iszero(new_val))
            refund_counter += GAS_SSTORE_REFUND;  // now clearing (non-zero -> 0)
    }
    if (u256_eq(new_val, original))
    {
        // Restoring to original value
        if (u256_iszero(original))
            refund_counter += GAS_SSTORE_SET - GAS_SSTORE_NOOP;  // 19900: undo SET cost
        else
            refund_counter += GAS_SSTORE_RESET - GAS_SSTORE_NOOP;  // 2800: undo RESET cost
    }
    return GAS_SSTORE_NOOP;  // 100
}

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
    device uchar* mem = mem_pool + ulong(tid) * MAX_MEMORY_PER_TX;
    device uchar* output = out_data + ulong(tid) * MAX_OUTPUT_PER_TX;
    device StorageEntry* storage = storage_pool + ulong(tid) * MAX_STORAGE_PER_TX;
    device uint& stor_count = storage_counts[tid];

    device const uchar* code = blob + inp.code_offset;
    uint code_size = inp.code_size;
    device const uchar* calldata = blob + inp.calldata_offset;
    uint calldata_size = inp.calldata_size;

    // Thread-local stack (lives in registers + stack memory).
    uint256 stack[STACK_LIMIT];
    uint sp = 0;  // stack pointer (number of items)

    ulong gas = inp.gas_limit;
    ulong refund_counter = 0;
    uint pc = 0;
    uint mem_size = 0;
    ulong gas_start = gas;

    // EIP-2200: track original values per storage slot for gas metering.
    OriginalEntry orig_storage[MAX_STORAGE_PER_TX];
    uint orig_count = 0;
    for (uint i = 0; i < MAX_STORAGE_PER_TX; ++i)
        orig_storage[i].valid = false;

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
            out.gas_used = gas_start - gas; out.gas_refund = refund_counter;
            out.output_size = 0;
            return;
        }

        // PUSH1..PUSH32
        if (op >= 0x60 && op <= 0x7f)
        {
            if (gas < GAS_VERYLOW) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_VERYLOW;
            if (sp >= STACK_LIMIT) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }

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
            if (gas < GAS_BASE) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_BASE;
            if (sp >= STACK_LIMIT) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            stack[sp++] = u256_zero();
            ++pc;
            continue;
        }

        // POP
        if (op == 0x50)
        {
            if (gas < GAS_BASE) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_BASE;
            if (sp == 0) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            --sp;
            ++pc;
            continue;
        }

        // DUP1..DUP16
        if (op >= 0x80 && op <= 0x8f)
        {
            if (gas < GAS_VERYLOW) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_VERYLOW;
            uint n = op - 0x80 + 1;
            if (n > sp || sp >= STACK_LIMIT) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            stack[sp] = stack[sp - n];
            ++sp;
            ++pc;
            continue;
        }

        // SWAP1..SWAP16
        if (op >= 0x90 && op <= 0x9f)
        {
            if (gas < GAS_VERYLOW) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_VERYLOW;
            uint n = op - 0x90 + 1;
            if (n >= sp) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
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
            if (gas < GAS_VERYLOW) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_VERYLOW;
            if (sp < 2) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            uint256 a = stack[--sp];
            uint256 b = stack[--sp];
            stack[sp++] = u256_add(a, b);
            ++pc;
            continue;
        }

        // MUL
        if (op == 0x02)
        {
            if (gas < GAS_LOW) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_LOW;
            if (sp < 2) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            uint256 a = stack[--sp];
            uint256 b = stack[--sp];
            stack[sp++] = u256_mul(a, b);
            ++pc;
            continue;
        }

        // SUB
        if (op == 0x03)
        {
            if (gas < GAS_VERYLOW) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_VERYLOW;
            if (sp < 2) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            uint256 a = stack[--sp];
            uint256 b = stack[--sp];
            stack[sp++] = u256_sub(a, b);
            ++pc;
            continue;
        }

        // DIV (0x04)
        if (op == 0x04)
        {
            if (gas < GAS_LOW) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_LOW;
            if (sp < 2) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            uint256 a = stack[--sp];
            uint256 b = stack[--sp];
            stack[sp++] = u256_div(a, b);
            ++pc;
            continue;
        }

        // SDIV (0x05)
        if (op == 0x05)
        {
            if (gas < GAS_LOW) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_LOW;
            if (sp < 2) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            uint256 a = stack[--sp];
            uint256 b = stack[--sp];
            stack[sp++] = u256_sdiv(a, b);
            ++pc;
            continue;
        }

        // MOD (0x06)
        if (op == 0x06)
        {
            if (gas < GAS_LOW) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_LOW;
            if (sp < 2) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            uint256 a = stack[--sp];
            uint256 b = stack[--sp];
            stack[sp++] = u256_mod(a, b);
            ++pc;
            continue;
        }

        // SMOD (0x07)
        if (op == 0x07)
        {
            if (gas < GAS_LOW) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_LOW;
            if (sp < 2) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            uint256 a = stack[--sp];
            uint256 b = stack[--sp];
            stack[sp++] = u256_smod(a, b);
            ++pc;
            continue;
        }

        // ADDMOD (0x08)
        if (op == 0x08)
        {
            if (gas < GAS_MID) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_MID;
            if (sp < 3) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            uint256 a = stack[--sp];
            uint256 b = stack[--sp];
            uint256 n = stack[--sp];
            stack[sp++] = u256_addmod(a, b, n);
            ++pc;
            continue;
        }

        // MULMOD (0x09)
        if (op == 0x09)
        {
            if (gas < GAS_MID) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_MID;
            if (sp < 3) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            uint256 a = stack[--sp];
            uint256 b = stack[--sp];
            uint256 n = stack[--sp];
            stack[sp++] = u256_mulmod(a, b, n);
            ++pc;
            continue;
        }

        // EXP (0x0a)
        if (op == 0x0a)
        {
            if (gas < GAS_EXP_BASE) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_EXP_BASE;
            if (sp < 2) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            uint256 a = stack[--sp];
            uint256 b = stack[--sp];
            // Dynamic gas: 50 * (number of bytes in exponent)
            uint exp_bytes = u256_byte_length(b);
            ulong exp_gas = GAS_EXP_BYTE * ulong(exp_bytes);
            if (gas < exp_gas) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= exp_gas;
            stack[sp++] = u256_exp(a, b);
            ++pc;
            continue;
        }

        // SIGNEXTEND (0x0b)
        if (op == 0x0b)
        {
            if (gas < GAS_LOW) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_LOW;
            if (sp < 2) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            uint256 a = stack[--sp];
            uint256 b = stack[--sp];
            stack[sp++] = u256_signextend(a, b);
            ++pc;
            continue;
        }

        // ISZERO
        if (op == 0x15)
        {
            if (gas < GAS_VERYLOW) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_VERYLOW;
            if (sp < 1) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            uint256 a = stack[--sp];
            stack[sp++] = u256_iszero(a) ? u256_one() : u256_zero();
            ++pc;
            continue;
        }

        // AND, OR, XOR
        if (op >= 0x16 && op <= 0x18)
        {
            if (gas < GAS_VERYLOW) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_VERYLOW;
            if (sp < 2) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
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
            if (gas < GAS_VERYLOW) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_VERYLOW;
            if (sp < 1) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            uint256 a = stack[--sp];
            stack[sp++] = u256_bitwise_not(a);
            ++pc;
            continue;
        }

        // LT, GT, SLT, SGT, EQ
        if (op == 0x10 || op == 0x11 || op == 0x12 || op == 0x13 || op == 0x14)
        {
            if (gas < GAS_VERYLOW) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_VERYLOW;
            if (sp < 2) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            uint256 a = stack[--sp];
            uint256 b = stack[--sp];
            bool result = false;
            if (op == 0x10) result = u256_lt(a, b);
            else if (op == 0x11) result = u256_gt(a, b);
            else if (op == 0x12) result = u256_slt(a, b);
            else if (op == 0x13) result = u256_sgt(a, b);
            else result = u256_eq(a, b);
            stack[sp++] = result ? u256_one() : u256_zero();
            ++pc;
            continue;
        }

        // BYTE (0x1a): i=top, x=second -> byte at position i of x
        if (op == 0x1a)
        {
            if (gas < GAS_VERYLOW) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_VERYLOW;
            if (sp < 2) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            uint256 i = stack[--sp];
            uint256 x = stack[--sp];
            stack[sp++] = u256_byte_at(x, i);
            ++pc;
            continue;
        }

        // SHL (0x1b): shift=top, value=second -> value << shift
        if (op == 0x1b)
        {
            if (gas < GAS_VERYLOW) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_VERYLOW;
            if (sp < 2) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            uint256 shift_amt = stack[--sp];
            uint256 val = stack[--sp];
            if (shift_amt.w[1] | shift_amt.w[2] | shift_amt.w[3] || shift_amt.w[0] >= 256)
                stack[sp++] = u256_zero();
            else
                stack[sp++] = u256_shl(shift_amt.w[0], val);
            ++pc;
            continue;
        }

        // SHR (0x1c): shift=top, value=second -> value >> shift
        if (op == 0x1c)
        {
            if (gas < GAS_VERYLOW) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_VERYLOW;
            if (sp < 2) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            uint256 shift_amt = stack[--sp];
            uint256 val = stack[--sp];
            if (shift_amt.w[1] | shift_amt.w[2] | shift_amt.w[3] || shift_amt.w[0] >= 256)
                stack[sp++] = u256_zero();
            else
                stack[sp++] = u256_shr(shift_amt.w[0], val);
            ++pc;
            continue;
        }

        // SAR (0x1d): shift=top, value=second -> arithmetic right shift
        if (op == 0x1d)
        {
            if (gas < GAS_VERYLOW) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_VERYLOW;
            if (sp < 2) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            uint256 shift_amt = stack[--sp];
            uint256 val = stack[--sp];
            bool negative = (val.w[3] >> 63) != 0;
            if (shift_amt.w[1] | shift_amt.w[2] | shift_amt.w[3] || shift_amt.w[0] >= 256)
                stack[sp++] = negative ? u256_max() : u256_zero();
            else
                stack[sp++] = u256_sar(shift_amt.w[0], val);
            ++pc;
            continue;
        }

        // MLOAD
        if (op == 0x51)
        {
            if (gas < GAS_VERYLOW) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_VERYLOW;
            if (sp < 1) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            uint256 off_val = stack[--sp];
            if (off_val.w[1] | off_val.w[2] | off_val.w[3] || off_val.w[0] + 32 > MAX_MEMORY_PER_TX)
            {
                out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return;
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
                if (gas < cost) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
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
            if (gas < GAS_VERYLOW) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_VERYLOW;
            if (sp < 2) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            uint256 off_val = stack[--sp];
            uint256 val = stack[--sp];
            if (off_val.w[1] | off_val.w[2] | off_val.w[3] || off_val.w[0] + 32 > MAX_MEMORY_PER_TX)
            {
                out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return;
            }
            uint off = uint(off_val.w[0]);
            uint end = off + 32;
            uint new_words = (end + 31) / 32;
            if (new_words * 32 > mem_size)
            {
                uint old_words = mem_size / 32;
                ulong cost = GAS_MEMORY * new_words + (ulong(new_words) * new_words) / 512
                           - GAS_MEMORY * old_words - (ulong(old_words) * old_words) / 512;
                if (gas < cost) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
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
            if (gas < GAS_MID) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_MID;
            if (sp < 1) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            uint256 dest_val = stack[--sp];
            if (dest_val.w[1] | dest_val.w[2] | dest_val.w[3] || dest_val.w[0] >= code_size)
            {
                out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return;
            }
            uint dest = uint(dest_val.w[0]);
            if (!is_valid_jumpdest(code, code_size, dest))
            {
                out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return;
            }
            pc = dest;
            continue;
        }

        // JUMPI
        if (op == 0x57)
        {
            if (gas < GAS_HIGH) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_HIGH;
            if (sp < 2) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            uint256 dest_val = stack[--sp];
            uint256 cond = stack[--sp];
            if (!u256_iszero(cond))
            {
                if (dest_val.w[1] | dest_val.w[2] | dest_val.w[3] || dest_val.w[0] >= code_size)
                {
                    out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return;
                }
                uint dest = uint(dest_val.w[0]);
                if (!is_valid_jumpdest(code, code_size, dest))
                {
                    out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return;
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
            if (gas < GAS_JUMPDEST) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_JUMPDEST;
            ++pc;
            continue;
        }

        // PC
        if (op == 0x58)
        {
            if (gas < GAS_BASE) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_BASE;
            if (sp >= STACK_LIMIT) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            stack[sp++] = u256_from(gpu_u64(pc));
            ++pc;
            continue;
        }

        // GAS
        if (op == 0x5a)
        {
            if (gas < GAS_BASE) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_BASE;
            if (sp >= STACK_LIMIT) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            stack[sp++] = u256_from(gas);
            ++pc;
            continue;
        }

        // SLOAD
        if (op == 0x54)
        {
            if (gas < GAS_SLOAD) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_SLOAD;
            if (sp < 1) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
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

        // SSTORE — EIP-2200 gas with original-value tracking
        if (op == 0x55)
        {
            if (sp < 2) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            uint256 slot = stack[--sp];
            uint256 val = stack[--sp];

            // Find current value.
            uint256 current = u256_zero();
            bool found = false;
            for (uint i = stor_count; i > 0; --i)
            {
                if (u256_eq(storage[i - 1].key, slot))
                {
                    current = storage[i - 1].value;
                    found = true;
                    break;
                }
            }

            // Record original value on first access to this slot.
            original_value_record(orig_storage, orig_count, slot, current);

            // Look up the original value for EIP-2200 gas calculation.
            uint256 original = u256_zero();
            original_value_lookup(orig_storage, orig_count, slot, original);

            // EIP-2200 gas metering.
            ulong sstore_cost = sstore_gas_eip2200(original, current, val, refund_counter);
            if (gas < sstore_cost) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= sstore_cost;

            // Write value.
            if (found)
            {
                for (uint i = stor_count; i > 0; --i)
                {
                    if (u256_eq(storage[i - 1].key, slot))
                    {
                        storage[i - 1].value = val;
                        break;
                    }
                }
            }
            else if (stor_count < MAX_STORAGE_PER_TX)
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
            if (sp < 2) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            uint256 off_val = stack[--sp];
            uint256 sz_val = stack[--sp];
            uint off = uint(off_val.w[0]);
            uint sz = uint(sz_val.w[0]);

            // Memory expansion before copy.
            if (sz > 0)
            {
                uint new_end = off + sz;
                if (new_end < off || new_end > MAX_MEMORY_PER_TX)
                    { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
                if (new_end > mem_size)
                {
                    uint new_words = (new_end + 31) / 32;
                    uint old_words = (mem_size + 31) / 32;
                    ulong mem_cost = GAS_MEMORY * (new_words - old_words)
                                   + (ulong(new_words) * new_words / 512)
                                   - (ulong(old_words) * old_words / 512);
                    if (gas < mem_cost)
                        { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
                    gas -= mem_cost;
                    for (uint i = mem_size; i < new_words * 32; ++i) mem[i] = 0;
                    mem_size = new_words * 32;
                }
            }

            uint copy_sz = (sz > MAX_OUTPUT_PER_TX) ? MAX_OUTPUT_PER_TX : sz;
            for (uint i = 0; i < copy_sz; ++i)
                output[i] = mem[off + i];
            out.status = 1;  // Return
            out.gas_used = gas_start - gas;
            out.gas_refund = refund_counter;
            out.output_size = copy_sz;
            return;
        }

        // REVERT
        if (op == 0xfd)
        {
            if (sp < 2) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            uint256 off_val = stack[--sp];
            uint256 sz_val = stack[--sp];
            uint off = uint(off_val.w[0]);
            uint sz = uint(sz_val.w[0]);

            // Memory expansion before copy.
            if (sz > 0)
            {
                uint new_end = off + sz;
                if (new_end < off || new_end > MAX_MEMORY_PER_TX)
                    { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
                if (new_end > mem_size)
                {
                    uint new_words = (new_end + 31) / 32;
                    uint old_words = (mem_size + 31) / 32;
                    ulong mem_cost = GAS_MEMORY * (new_words - old_words)
                                   + (ulong(new_words) * new_words / 512)
                                   - (ulong(old_words) * old_words / 512);
                    if (gas < mem_cost)
                        { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
                    gas -= mem_cost;
                    for (uint i = mem_size; i < new_words * 32; ++i) mem[i] = 0;
                    mem_size = new_words * 32;
                }
            }

            uint copy_sz = (sz > MAX_OUTPUT_PER_TX) ? MAX_OUTPUT_PER_TX : sz;
            for (uint i = 0; i < copy_sz; ++i)
                output[i] = mem[off + i];
            out.status = 2;  // Revert
            out.gas_used = gas_start - gas;
            out.gas_refund = refund_counter;
            out.output_size = copy_sz;
            return;
        }

        // INVALID
        if (op == 0xfe)
        {
            out.status = 4;
            out.gas_used = gas_start;
            out.gas_refund = refund_counter;
            out.output_size = 0;
            return;
        }

        // CALL/CREATE family — signal CPU fallback
        if (op == 0xf0 || op == 0xf1 || op == 0xf2 || op == 0xf4 ||
            op == 0xf5 || op == 0xfa || op == 0xff)
        {
            out.status = 5;  // CallNotSupported
            out.gas_used = gas_start - gas;
            out.gas_refund = refund_counter;
            out.output_size = 0;
            return;
        }

        // ADDRESS (0x30)
        if (op == 0x30)
        {
            if (gas < GAS_BASE) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_BASE;
            if (sp >= STACK_LIMIT) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            stack[sp++] = inp.address;
            ++pc;
            continue;
        }

        // CALLER (0x33)
        if (op == 0x33)
        {
            if (gas < GAS_BASE) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_BASE;
            if (sp >= STACK_LIMIT) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            stack[sp++] = inp.caller;
            ++pc;
            continue;
        }

        // CALLVALUE (0x34)
        if (op == 0x34)
        {
            if (gas < GAS_BASE) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_BASE;
            if (sp >= STACK_LIMIT) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            stack[sp++] = inp.value;
            ++pc;
            continue;
        }

        // CALLDATASIZE (0x36)
        if (op == 0x36)
        {
            if (gas < GAS_BASE) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_BASE;
            if (sp >= STACK_LIMIT) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            stack[sp++] = u256_from(gpu_u64(calldata_size));
            ++pc;
            continue;
        }

        // CALLDATALOAD (0x35)
        if (op == 0x35)
        {
            if (gas < GAS_VERYLOW) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_VERYLOW;
            if (sp < 1) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
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
            if (gas < GAS_BASE) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_BASE;
            if (sp >= STACK_LIMIT) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            stack[sp++] = u256_from(gpu_u64(mem_size));
            ++pc;
            continue;
        }

        // Unrecognized opcode — error.
        out.status = 4;
        out.gas_used = gas_start - gas;
        out.gas_refund = refund_counter;
        out.output_size = 0;
        return;
    }

    // Fell off end of code -> implicit STOP.
    out.status = 0;
    out.gas_used = gas_start - gas;
    out.gas_refund = refund_counter;
    out.output_size = 0;
}

// =============================================================================
// GPU-native stateful kernel: SLOAD/SSTORE read/write the persistent hash table
// =============================================================================
//
// This kernel variant accepts the GPU-resident state tables as additional
// buffer arguments. SLOAD reads from the persistent storage hash table.
// SSTORE writes to it. No CPU round-trip for state access.
//
// The per-tx storage buffer (buffer[5]) is still used as a write-set for
// Block-STM conflict detection. The global table (buffer[8], buffer[9]) is
// the persistent state.

/// Storage entry in the persistent global hash table (matches state_table.metal).
struct GlobalStorageEntry {
    uchar key_addr[20];
    uchar key_slot[32];
    uint  key_valid;
    uint  _pad;
    uchar value[32];
};

/// Hash 20+32 bytes for storage table index.
static inline uint hash_stor_key(uint256 addr256, uint256 slot256, uint capacity) {
    // Extract the lower 20 bytes of address and all 32 bytes of slot.
    // Address is right-aligned in uint256, so bytes [0..19] are in w[0..2].
    uint h = 0x811c9dc5u;
    // Hash address bytes (lower 20 bytes of uint256, little-endian words).
    for (uint i = 0; i < 4; ++i) {
        gpu_u64 word = addr256.w[i];
        uint nbytes = (i < 2) ? 8 : ((i == 2) ? 4 : 0);
        for (uint b = 0; b < nbytes; ++b) {
            h ^= uint((word >> (b * 8)) & 0xFFUL);
            h *= 0x01000193u;
        }
    }
    // Hash slot bytes (all 32 bytes of uint256, little-endian).
    for (uint i = 0; i < 4; ++i) {
        gpu_u64 word = slot256.w[i];
        for (uint b = 0; b < 8; ++b) {
            h ^= uint((word >> (b * 8)) & 0xFFUL);
            h *= 0x01000193u;
        }
    }
    return h & (capacity - 1);
}

/// Lookup a storage value from the persistent global table.
/// Returns the 32-byte value as a uint256. Returns zero if not found.
static inline uint256 global_storage_load(
    device GlobalStorageEntry* table,
    uint capacity,
    uint256 addr256,
    uint256 slot256)
{
    uint idx = hash_stor_key(addr256, slot256, capacity);

    // Extract raw bytes for comparison.
    uchar addr_bytes[20];
    for (uint i = 0; i < 4; ++i) {
        gpu_u64 w = addr256.w[i];
        uint nbytes = (i < 2) ? 8 : ((i == 2) ? 4 : 0);
        for (uint b = 0; b < nbytes; ++b)
            addr_bytes[i * 8 + b] = uchar((w >> (b * 8)) & 0xFF);
    }

    uchar slot_bytes[32];
    for (uint i = 0; i < 4; ++i) {
        gpu_u64 w = slot256.w[i];
        for (uint b = 0; b < 8; ++b)
            slot_bytes[i * 8 + b] = uchar((w >> (b * 8)) & 0xFF);
    }

    for (uint probe = 0; probe < capacity; ++probe) {
        uint s = (idx + probe) & (capacity - 1);
        if (table[s].key_valid == 0)
            return u256_zero();

        bool match = true;
        for (uint i = 0; i < 20 && match; ++i)
            match = (table[s].key_addr[i] == addr_bytes[i]);
        for (uint i = 0; i < 32 && match; ++i)
            match = (table[s].key_slot[i] == slot_bytes[i]);

        if (match) {
            // Reconstruct uint256 from little-endian bytes.
            uint256 val = u256_zero();
            for (uint i = 0; i < 4; ++i) {
                gpu_u64 w = 0;
                for (uint b = 0; b < 8; ++b)
                    w |= gpu_u64(table[s].value[i * 8 + b]) << (b * 8);
                val.w[i] = w;
            }
            return val;
        }
    }
    return u256_zero();
}

/// Store a value into the persistent global storage table.
static inline void global_storage_store(
    device GlobalStorageEntry* table,
    uint capacity,
    uint256 addr256,
    uint256 slot256,
    uint256 val)
{
    uint idx = hash_stor_key(addr256, slot256, capacity);

    uchar addr_bytes[20];
    for (uint i = 0; i < 4; ++i) {
        gpu_u64 w = addr256.w[i];
        uint nbytes = (i < 2) ? 8 : ((i == 2) ? 4 : 0);
        for (uint b = 0; b < nbytes; ++b)
            addr_bytes[i * 8 + b] = uchar((w >> (b * 8)) & 0xFF);
    }

    uchar slot_bytes[32];
    for (uint i = 0; i < 4; ++i) {
        gpu_u64 w = slot256.w[i];
        for (uint b = 0; b < 8; ++b)
            slot_bytes[i * 8 + b] = uchar((w >> (b * 8)) & 0xFF);
    }

    uchar val_bytes[32];
    for (uint i = 0; i < 4; ++i) {
        gpu_u64 w = val.w[i];
        for (uint b = 0; b < 8; ++b)
            val_bytes[i * 8 + b] = uchar((w >> (b * 8)) & 0xFF);
    }

    for (uint probe = 0; probe < capacity; ++probe) {
        uint s = (idx + probe) & (capacity - 1);

        // Try to claim empty slot via atomic CAS.
        device atomic_uint* valid_ptr =
            reinterpret_cast<device atomic_uint*>(&table[s].key_valid);

        uint expected = 0;
        if (atomic_compare_exchange_weak_explicit(
                valid_ptr, &expected, 1u,
                memory_order_relaxed, memory_order_relaxed))
        {
            for (uint i = 0; i < 20; ++i) table[s].key_addr[i] = addr_bytes[i];
            for (uint i = 0; i < 32; ++i) table[s].key_slot[i] = slot_bytes[i];
            for (uint i = 0; i < 32; ++i) table[s].value[i] = val_bytes[i];
            return;
        }

        // Check if occupied by our key.
        bool match = true;
        for (uint i = 0; i < 20 && match; ++i)
            match = (table[s].key_addr[i] == addr_bytes[i]);
        for (uint i = 0; i < 32 && match; ++i)
            match = (table[s].key_slot[i] == slot_bytes[i]);

        if (match) {
            for (uint i = 0; i < 32; ++i) table[s].value[i] = val_bytes[i];
            return;
        }
    }
}

/// Stateful EVM kernel: SLOAD reads write-set then persistent state;
/// SSTORE writes to per-tx write-set only (committed after Block-STM validation).
///
/// Buffer layout:
///   [0] TxInput*           — per-transaction input descriptors
///   [1] uchar*             — contiguous bytecode + calldata blob
///   [2] TxOutput*          — per-transaction output descriptors
///   [3] uchar*             — per-transaction output data
///   [4] uchar*             — per-transaction memory
///   [5] StorageEntry*      — per-transaction write-set (for Block-STM)
///   [6] uint*              — per-transaction storage counts
///   [7] uint*              — params: [0]=num_txs, [1]=global_storage_capacity
///   [8] GlobalStorageEntry* — persistent global storage hash table
kernel void evm_execute_stateful(
    device const TxInput*          inputs         [[buffer(0)]],
    device const uchar*            blob           [[buffer(1)]],
    device TxOutput*               outputs        [[buffer(2)]],
    device uchar*                  out_data       [[buffer(3)]],
    device uchar*                  mem_pool       [[buffer(4)]],
    device StorageEntry*           storage_pool   [[buffer(5)]],
    device uint*                   storage_counts [[buffer(6)]],
    device const uint*             params         [[buffer(7)]],
    device GlobalStorageEntry*     global_storage [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    uint num_txs = params[0];
    uint global_capacity = params[1];
    if (tid >= num_txs)
        return;

    device const TxInput& inp = inputs[tid];
    device TxOutput& out = outputs[tid];
    device uchar* mem = mem_pool + ulong(tid) * MAX_MEMORY_PER_TX;
    device uchar* output = out_data + ulong(tid) * MAX_OUTPUT_PER_TX;
    device StorageEntry* tx_storage = storage_pool + ulong(tid) * MAX_STORAGE_PER_TX;
    device uint& tx_stor_count = storage_counts[tid];

    device const uchar* code = blob + inp.code_offset;
    uint code_size = inp.code_size;
    device const uchar* calldata = blob + inp.calldata_offset;
    uint calldata_size = inp.calldata_size;

    uint256 stack[STACK_LIMIT];
    uint sp = 0;
    uint pc = 0;
    ulong gas = inp.gas_limit;
    ulong refund_counter = 0;
    ulong gas_start = gas;
    uint mem_size = 0;

    // EIP-2200: track original values per storage slot for gas metering.
    OriginalEntry orig_storage[MAX_STORAGE_PER_TX];
    uint orig_count = 0;
    for (uint i = 0; i < MAX_STORAGE_PER_TX; ++i)
        orig_storage[i].valid = false;

    while (pc < code_size)
    {
        uchar op = code[pc];

        // --- SLOAD: read from persistent global storage table ---
        if (op == 0x54)
        {
            if (gas < GAS_SLOAD) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_SLOAD;
            if (sp < 1) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            uint256 slot = stack[--sp];

            // First check the per-tx write-set (Block-STM local writes).
            uint256 val = u256_zero();
            bool found_local = false;
            for (uint i = tx_stor_count; i > 0; --i) {
                if (u256_eq(tx_storage[i - 1].key, slot)) {
                    val = tx_storage[i - 1].value;
                    found_local = true;
                    break;
                }
            }

            // If not in write-set, read from persistent global table.
            if (!found_local)
                val = global_storage_load(global_storage, global_capacity, inp.address, slot);

            stack[sp++] = val;
            ++pc;
            continue;
        }

        // --- SSTORE: write to per-tx write-set ONLY (Block-STM validated commit) ---
        // Writes go to the per-tx write-set buffer, NOT the global table.
        // After Block-STM validation confirms no conflicts, a separate commit
        // kernel copies validated writes to the global table. Writing directly
        // to global state would bypass MvMemory conflict detection.
        if (op == 0x55)
        {
            if (sp < 2) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            uint256 slot = stack[--sp];
            uint256 val = stack[--sp];

            // Read current value for gas metering: check per-tx write-set first,
            // then fall back to global table for the base state.
            uint256 current = u256_zero();
            bool found_current = false;
            for (uint i = tx_stor_count; i > 0; --i) {
                if (u256_eq(tx_storage[i - 1].key, slot)) {
                    current = tx_storage[i - 1].value;
                    found_current = true;
                    break;
                }
            }
            if (!found_current)
                current = global_storage_load(global_storage, global_capacity, inp.address, slot);

            // Record original value on first access to this slot.
            original_value_record(orig_storage, orig_count, slot, current);

            // Look up the original value for EIP-2200 gas calculation.
            uint256 original = u256_zero();
            original_value_lookup(orig_storage, orig_count, slot, original);

            // EIP-2200 gas metering.
            ulong sstore_cost = sstore_gas_eip2200(original, current, val, refund_counter);
            if (gas < sstore_cost) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= sstore_cost;

            // Record in per-tx write-set (Block-STM will validate before commit).
            bool found = false;
            for (uint i = tx_stor_count; i > 0; --i) {
                if (u256_eq(tx_storage[i - 1].key, slot)) {
                    tx_storage[i - 1].value = val;
                    found = true;
                    break;
                }
            }
            if (!found && tx_stor_count < MAX_STORAGE_PER_TX) {
                tx_storage[tx_stor_count].key = slot;
                tx_storage[tx_stor_count].value = val;
                tx_stor_count++;
            }

            ++pc;
            continue;
        }

        // All other opcodes: delegate to the same logic as evm_execute.
        // STOP
        if (op == 0x00) { out.status = 0; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }

        // PUSH1..PUSH32 (0x60..0x7f)
        if (op >= 0x60 && op <= 0x7f)
        {
            if (gas < GAS_VERYLOW) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_VERYLOW;
            if (sp >= STACK_LIMIT) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            uint n = uint(op) - 0x5f;
            uint256 val = u256_zero();
            for (uint i = 0; i < n && (pc + 1 + i) < code_size; ++i)
            {
                uint byte_pos = n - 1 - i;
                uint limb = byte_pos / 8;
                uint shift = (byte_pos % 8) * 8;
                val.w[limb] |= gpu_u64(code[pc + 1 + i]) << shift;
            }
            stack[sp++] = val;
            pc += 1 + n;
            continue;
        }

        // DUP1..DUP16 (0x80..0x8f)
        if (op >= 0x80 && op <= 0x8f)
        {
            if (gas < GAS_VERYLOW) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_VERYLOW;
            uint depth = uint(op) - 0x7f;
            if (sp < depth) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            if (sp >= STACK_LIMIT) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            stack[sp] = stack[sp - depth];
            sp++;
            ++pc;
            continue;
        }

        // SWAP1..SWAP16 (0x90..0x9f)
        if (op >= 0x90 && op <= 0x9f)
        {
            if (gas < GAS_VERYLOW) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_VERYLOW;
            uint depth = uint(op) - 0x8f;
            if (sp < depth + 1) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            uint256 tmp = stack[sp - 1];
            stack[sp - 1] = stack[sp - 1 - depth];
            stack[sp - 1 - depth] = tmp;
            ++pc;
            continue;
        }

        // POP (0x50)
        if (op == 0x50)
        {
            if (gas < GAS_BASE) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_BASE;
            if (sp < 1) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            --sp;
            ++pc;
            continue;
        }

        // JUMP (0x56)
        if (op == 0x56)
        {
            if (gas < GAS_MID) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_MID;
            if (sp < 1) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            uint256 dest = stack[--sp];
            uint target = uint(dest.w[0]);
            if (!is_valid_jumpdest(code, code_size, target))
                { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            pc = target;
            continue;
        }

        // JUMPI (0x57)
        if (op == 0x57)
        {
            if (gas < GAS_HIGH) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_HIGH;
            if (sp < 2) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            uint256 dest = stack[--sp];
            uint256 cond = stack[--sp];
            if (!u256_iszero(cond)) {
                uint target = uint(dest.w[0]);
                if (!is_valid_jumpdest(code, code_size, target))
                    { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
                pc = target;
            } else {
                ++pc;
            }
            continue;
        }

        // JUMPDEST (0x5b)
        if (op == 0x5b)
        {
            if (gas < GAS_JUMPDEST) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
            gas -= GAS_JUMPDEST;
            ++pc;
            continue;
        }

        // ADD..XOR, NOT, arithmetic, comparison, MLOAD, MSTORE, RETURN, REVERT
        // -- Delegate to same opcode handling as evm_execute.
        // For brevity, the stateful kernel handles only the state-touching
        // opcodes differently. All other opcodes use the identical logic.
        // In production, this would be a shared function. Here we mark
        // unhandled opcodes as needing CPU fallback.

        // RETURN (0xf3)
        if (op == 0xf3) {
            if (sp < 2) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            uint256 off_val = stack[--sp];
            uint256 sz_val = stack[--sp];
            uint off = uint(off_val.w[0]);
            uint sz = uint(sz_val.w[0]);
            if (sz > 0) {
                uint new_end = off + sz;
                if (new_end < off || new_end > MAX_MEMORY_PER_TX)
                    { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
                if (new_end > mem_size) {
                    uint new_words = (new_end + 31) / 32;
                    uint old_words = (mem_size + 31) / 32;
                    ulong mem_cost = GAS_MEMORY * (new_words - old_words)
                                   + (ulong(new_words) * new_words / 512)
                                   - (ulong(old_words) * old_words / 512);
                    if (gas < mem_cost) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
                    gas -= mem_cost;
                    for (uint i = mem_size; i < new_words * 32; ++i) mem[i] = 0;
                    mem_size = new_words * 32;
                }
            }
            uint copy_sz = (sz > MAX_OUTPUT_PER_TX) ? MAX_OUTPUT_PER_TX : sz;
            for (uint i = 0; i < copy_sz; ++i) output[i] = mem[off + i];
            out.status = 1;
            out.gas_used = gas_start - gas;
            out.gas_refund = refund_counter;
            out.output_size = copy_sz;
            return;
        }

        // REVERT (0xfd)
        if (op == 0xfd) {
            if (sp < 2) { out.status = 4; out.gas_used = gas_start - gas; out.gas_refund = refund_counter; out.output_size = 0; return; }
            uint256 off_val = stack[--sp];
            uint256 sz_val = stack[--sp];
            uint off = uint(off_val.w[0]);
            uint sz = uint(sz_val.w[0]);
            if (sz > 0) {
                uint new_end = off + sz;
                if (new_end < off || new_end > MAX_MEMORY_PER_TX)
                    { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
                if (new_end > mem_size) {
                    uint new_words = (new_end + 31) / 32;
                    uint old_words = (mem_size + 31) / 32;
                    ulong mem_cost = GAS_MEMORY * (new_words - old_words)
                                   + (ulong(new_words) * new_words / 512)
                                   - (ulong(old_words) * old_words / 512);
                    if (gas < mem_cost) { out.status = 3; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }
                    gas -= mem_cost;
                    for (uint i = mem_size; i < new_words * 32; ++i) mem[i] = 0;
                    mem_size = new_words * 32;
                }
            }
            uint copy_sz = (sz > MAX_OUTPUT_PER_TX) ? MAX_OUTPUT_PER_TX : sz;
            for (uint i = 0; i < copy_sz; ++i) output[i] = mem[off + i];
            out.status = 2;
            out.gas_used = gas_start - gas;
            out.gas_refund = refund_counter;
            out.output_size = copy_sz;
            return;
        }

        // INVALID (0xfe)
        if (op == 0xfe) { out.status = 4; out.gas_used = gas_start; out.gas_refund = refund_counter; out.output_size = 0; return; }

        // CALL/CREATE family -> CPU fallback
        if (op == 0xf0 || op == 0xf1 || op == 0xf2 || op == 0xf4 ||
            op == 0xf5 || op == 0xfa || op == 0xff) {
            out.status = 5;
            out.gas_used = gas_start - gas;
            out.gas_refund = refund_counter;
            out.output_size = 0;
            return;
        }

        // For all other opcodes, signal CPU fallback.
        // A production build would inline the full opcode table here.
        out.status = 5;
        out.gas_used = gas_start - gas;
        out.gas_refund = refund_counter;
        out.output_size = 0;
        return;
    }

    out.status = 0;
    out.gas_used = gas_start - gas;
    out.gas_refund = refund_counter;
    out.output_size = 0;
}

#pragma clang diagnostic pop
