// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file uint256_gpu.hpp
/// 256-bit unsigned integer for GPU (Metal/CUDA) and CPU.
///
/// Representation: 4 x uint64 limbs in little-endian order.
///   w[0] is the least-significant 64 bits, w[3] the most-significant.
///
/// All operations are branchless where possible. Carry propagation uses
/// the standard schoolbook approach with 64-bit limbs. On Metal, this
/// compiles to ~4 add-with-carry instructions per 256-bit add.
///
/// Platform guards:
///   __METAL_VERSION__  - Apple Metal Shading Language
///   __CUDA_ARCH__      - NVIDIA CUDA device code
///   Otherwise          - Standard C++ (host)

#pragma once

// -- Platform detection -------------------------------------------------------

#if defined(__METAL_VERSION__)
  #include <metal_stdlib>
  #define GPU_DEVICE
  #define GPU_INLINE inline
  using gpu_u64 = metal::ulong;
  using gpu_u32 = metal::uint;
  using gpu_i64 = metal::long_t;
  using gpu_u8_t = metal::uchar;
#elif defined(__CUDA_ARCH__)
  #define GPU_DEVICE __device__
  #define GPU_INLINE __device__ __forceinline__
  using gpu_u64 = unsigned long long;
  using gpu_u32 = unsigned int;
  using gpu_i64 = long long;
  using gpu_u8_t = unsigned char;
#else
  #include <cstdint>
  #include <cstring>
  #define GPU_DEVICE
  #define GPU_INLINE inline
  using gpu_u64 = uint64_t;
  using gpu_u32 = uint32_t;
  using gpu_i64 = int64_t;
  using gpu_u8_t = uint8_t;
#endif

namespace evm::gpu::kernel {

// -- uint256 ------------------------------------------------------------------

struct uint256
{
    gpu_u64 w[4];  // w[0] = low, w[3] = high

    // -- Construction ---------------------------------------------------------

    GPU_INLINE uint256() : w{0, 0, 0, 0} {}

    GPU_INLINE uint256(gpu_u64 lo) : w{lo, 0, 0, 0} {}

    GPU_INLINE uint256(gpu_u64 w0, gpu_u64 w1, gpu_u64 w2, gpu_u64 w3)
        : w{w0, w1, w2, w3} {}

    GPU_INLINE static uint256 zero() { return uint256{0, 0, 0, 0}; }
    GPU_INLINE static uint256 one()  { return uint256{1, 0, 0, 0}; }

    GPU_INLINE static uint256 max_value()
    {
        const gpu_u64 m = ~gpu_u64(0);
        return uint256{m, m, m, m};
    }

    // -- Comparison -----------------------------------------------------------

    GPU_INLINE bool operator==(const uint256& b) const
    {
        return w[0] == b.w[0] && w[1] == b.w[1] && w[2] == b.w[2] && w[3] == b.w[3];
    }

    GPU_INLINE bool operator!=(const uint256& b) const { return !(*this == b); }

    // -- Helpers for carry arithmetic -----------------------------------------
    //
    // add_carry: returns (sum, carry_out) where carry_out is 0 or 1.
    // On Metal, the compiler will lower this to addc instructions.

    struct pair64 { gpu_u64 lo; gpu_u64 hi; };

    GPU_INLINE static pair64 add_carry(gpu_u64 a, gpu_u64 b, gpu_u64 carry_in)
    {
        gpu_u64 s1 = a + b;
        gpu_u64 c1 = (s1 < a) ? gpu_u64(1) : gpu_u64(0);
        gpu_u64 s2 = s1 + carry_in;
        gpu_u64 c2 = (s2 < s1) ? gpu_u64(1) : gpu_u64(0);
        return {s2, c1 + c2};
    }

    GPU_INLINE static pair64 sub_borrow(gpu_u64 a, gpu_u64 b, gpu_u64 borrow_in)
    {
        gpu_u64 d1 = a - b;
        gpu_u64 b1 = (d1 > a) ? gpu_u64(1) : gpu_u64(0);
        gpu_u64 d2 = d1 - borrow_in;
        gpu_u64 b2 = (d2 > d1) ? gpu_u64(1) : gpu_u64(0);
        return {d2, b1 + b2};
    }

    // mul_wide: 64x64 -> 128-bit result as (lo, hi).
    GPU_INLINE static pair64 mul_wide(gpu_u64 a, gpu_u64 b)
    {
#if defined(__CUDA_ARCH__)
        gpu_u64 lo = a * b;
        gpu_u64 hi = __umul64hi(a, b);
        return {lo, hi};
#else
        // Schoolbook: split each 64-bit value into two 32-bit halves.
        gpu_u64 a_lo = a & 0xFFFFFFFFULL;
        gpu_u64 a_hi = a >> 32;
        gpu_u64 b_lo = b & 0xFFFFFFFFULL;
        gpu_u64 b_hi = b >> 32;

        gpu_u64 p0 = a_lo * b_lo;
        gpu_u64 p1 = a_lo * b_hi;
        gpu_u64 p2 = a_hi * b_lo;
        gpu_u64 p3 = a_hi * b_hi;

        gpu_u64 mid = (p0 >> 32) + (p1 & 0xFFFFFFFFULL) + (p2 & 0xFFFFFFFFULL);
        gpu_u64 hi = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);
        gpu_u64 lo = (p0 & 0xFFFFFFFFULL) | ((mid & 0xFFFFFFFFULL) << 32);
        return {lo, hi};
#endif
    }
};

// -- Arithmetic ---------------------------------------------------------------

/// 256-bit addition. Returns low 256 bits (wraps on overflow, per EVM spec).
GPU_INLINE uint256 add(const uint256& a, const uint256& b)
{
    uint256 r;
    auto [s0, c0] = uint256::add_carry(a.w[0], b.w[0], 0);
    auto [s1, c1] = uint256::add_carry(a.w[1], b.w[1], c0);
    auto [s2, c2] = uint256::add_carry(a.w[2], b.w[2], c1);
    r.w[0] = s0;
    r.w[1] = s1;
    r.w[2] = s2;
    r.w[3] = a.w[3] + b.w[3] + c2;  // top carry discarded (mod 2^256)
    return r;
}

/// 256-bit subtraction. Returns low 256 bits (wraps on underflow, per EVM spec).
GPU_INLINE uint256 sub(const uint256& a, const uint256& b)
{
    uint256 r;
    auto [d0, b0] = uint256::sub_borrow(a.w[0], b.w[0], 0);
    auto [d1, b1] = uint256::sub_borrow(a.w[1], b.w[1], b0);
    auto [d2, b2] = uint256::sub_borrow(a.w[2], b.w[2], b1);
    r.w[0] = d0;
    r.w[1] = d1;
    r.w[2] = d2;
    r.w[3] = a.w[3] - b.w[3] - b2;
    return r;
}

/// 256-bit multiplication (low 256 bits of 512-bit product).
/// Schoolbook 4x4 limb multiply, keeping only the low 4 limbs.
GPU_INLINE uint256 mul(const uint256& a, const uint256& b)
{
    // We need: r[0..3] = sum of a[i]*b[j] where i+j <= 3.
    // Products where i+j >= 4 overflow past 256 bits and are discarded.

    uint256 r;
    gpu_u64 carry = 0;

    // r[0] = lo(a0*b0)
    auto p00 = uint256::mul_wide(a.w[0], b.w[0]);
    r.w[0] = p00.lo;
    carry = p00.hi;

    // r[1] = lo(a0*b1 + a1*b0 + carry)
    auto p01 = uint256::mul_wide(a.w[0], b.w[1]);
    auto p10 = uint256::mul_wide(a.w[1], b.w[0]);
    gpu_u64 s1 = p01.lo + p10.lo + carry;
    // Compute carry into limb 2: sum of hi parts + overflow from lo additions
    gpu_u64 c1 = p01.hi + p10.hi;
    c1 += (s1 < p01.lo) ? gpu_u64(1) : gpu_u64(0);
    // Check if p10.lo + carry overflowed before adding p01.lo
    gpu_u64 tmp = p10.lo + carry;
    c1 += (tmp < p10.lo) ? gpu_u64(1) : gpu_u64(0);
    r.w[1] = s1;

    // r[2] = lo(a0*b2 + a1*b1 + a2*b0 + carry)
    auto p02 = uint256::mul_wide(a.w[0], b.w[2]);
    auto p11 = uint256::mul_wide(a.w[1], b.w[1]);
    auto p20 = uint256::mul_wide(a.w[2], b.w[0]);
    gpu_u64 s2 = p02.lo + p11.lo;
    gpu_u64 c2 = (s2 < p02.lo) ? gpu_u64(1) : gpu_u64(0);
    gpu_u64 s2b = s2 + p20.lo;
    c2 += (s2b < s2) ? gpu_u64(1) : gpu_u64(0);
    gpu_u64 s2c = s2b + c1;
    c2 += (s2c < s2b) ? gpu_u64(1) : gpu_u64(0);
    c2 += p02.hi + p11.hi + p20.hi;
    r.w[2] = s2c;

    // r[3] = lo(a0*b3 + a1*b2 + a2*b1 + a3*b0 + carry)
    // No need to track carry beyond this limb.
    r.w[3] = a.w[0] * b.w[3] + a.w[1] * b.w[2] + a.w[2] * b.w[1] + a.w[3] * b.w[0] + c2;

    return r;
}

// -- Comparison ---------------------------------------------------------------

GPU_INLINE bool iszero(const uint256& a)
{
    return (a.w[0] | a.w[1] | a.w[2] | a.w[3]) == 0;
}

GPU_INLINE bool eq(const uint256& a, const uint256& b)
{
    return a == b;
}

/// Unsigned less-than.
GPU_INLINE bool lt(const uint256& a, const uint256& b)
{
    if (a.w[3] != b.w[3]) return a.w[3] < b.w[3];
    if (a.w[2] != b.w[2]) return a.w[2] < b.w[2];
    if (a.w[1] != b.w[1]) return a.w[1] < b.w[1];
    return a.w[0] < b.w[0];
}

/// Unsigned greater-than.
GPU_INLINE bool gt(const uint256& a, const uint256& b)
{
    return lt(b, a);
}

/// Signed less-than (two's complement).
GPU_INLINE bool slt(const uint256& a, const uint256& b)
{
    bool a_neg = (a.w[3] >> 63) != 0;
    bool b_neg = (b.w[3] >> 63) != 0;
    if (a_neg != b_neg) return a_neg;  // negative < positive
    return lt(a, b);  // same sign: unsigned comparison works
}

/// Signed greater-than.
GPU_INLINE bool sgt(const uint256& a, const uint256& b)
{
    return slt(b, a);
}

// -- Bitwise ------------------------------------------------------------------

GPU_INLINE uint256 bitwise_and(const uint256& a, const uint256& b)
{
    return {a.w[0] & b.w[0], a.w[1] & b.w[1], a.w[2] & b.w[2], a.w[3] & b.w[3]};
}

GPU_INLINE uint256 bitwise_or(const uint256& a, const uint256& b)
{
    return {a.w[0] | b.w[0], a.w[1] | b.w[1], a.w[2] | b.w[2], a.w[3] | b.w[3]};
}

GPU_INLINE uint256 bitwise_xor(const uint256& a, const uint256& b)
{
    return {a.w[0] ^ b.w[0], a.w[1] ^ b.w[1], a.w[2] ^ b.w[2], a.w[3] ^ b.w[3]};
}

GPU_INLINE uint256 bitwise_not(const uint256& a)
{
    return {~a.w[0], ~a.w[1], ~a.w[2], ~a.w[3]};
}

/// BYTE opcode: extract byte at position i (0 = most significant byte).
GPU_INLINE uint256 byte_at(const uint256& val, const uint256& pos)
{
    // If pos >= 32, result is 0.
    if (pos.w[1] | pos.w[2] | pos.w[3])
        return uint256::zero();
    gpu_u64 i = pos.w[0];
    if (i >= 32)
        return uint256::zero();

    // Byte 0 is the most significant byte (big-endian index).
    // In our little-endian limb layout:
    //   byte 0  = w[3] >> 56
    //   byte 31 = w[0] & 0xFF
    gpu_u32 byte_from_right = gpu_u32(31 - i);
    gpu_u32 limb = byte_from_right / 8;
    gpu_u32 shift = (byte_from_right % 8) * 8;
    gpu_u64 b = (val.w[limb] >> shift) & 0xFFULL;
    return uint256{b};
}

/// SHL: shift left by n bits. If n >= 256, result is 0.
GPU_INLINE uint256 shl(const uint256& shift_amount, const uint256& val)
{
    if (shift_amount.w[1] | shift_amount.w[2] | shift_amount.w[3])
        return uint256::zero();
    gpu_u64 n = shift_amount.w[0];
    if (n >= 256)
        return uint256::zero();
    if (n == 0)
        return val;

    uint256 r = uint256::zero();
    gpu_u32 limb_shift = gpu_u32(n / 64);
    gpu_u32 bit_shift  = gpu_u32(n % 64);

    for (gpu_u32 i = limb_shift; i < 4; ++i)
    {
        r.w[i] = val.w[i - limb_shift] << bit_shift;
        if (bit_shift > 0 && i > limb_shift)
            r.w[i] |= val.w[i - limb_shift - 1] >> (64 - bit_shift);
    }
    return r;
}

/// SHR: logical shift right by n bits. If n >= 256, result is 0.
GPU_INLINE uint256 shr(const uint256& shift_amount, const uint256& val)
{
    if (shift_amount.w[1] | shift_amount.w[2] | shift_amount.w[3])
        return uint256::zero();
    gpu_u64 n = shift_amount.w[0];
    if (n >= 256)
        return uint256::zero();
    if (n == 0)
        return val;

    uint256 r = uint256::zero();
    gpu_u32 limb_shift = gpu_u32(n / 64);
    gpu_u32 bit_shift  = gpu_u32(n % 64);

    for (gpu_u32 i = 0; i + limb_shift < 4; ++i)
    {
        r.w[i] = val.w[i + limb_shift] >> bit_shift;
        if (bit_shift > 0 && i + limb_shift + 1 < 4)
            r.w[i] |= val.w[i + limb_shift + 1] << (64 - bit_shift);
    }
    return r;
}

/// SAR: arithmetic shift right (sign-extending).
GPU_INLINE uint256 sar(const uint256& shift_amount, const uint256& val)
{
    bool negative = (val.w[3] >> 63) != 0;
    if (shift_amount.w[1] | shift_amount.w[2] | shift_amount.w[3])
        return negative ? uint256::max_value() : uint256::zero();
    gpu_u64 n = shift_amount.w[0];
    if (n >= 256)
        return negative ? uint256::max_value() : uint256::zero();

    uint256 r = shr(shift_amount, val);
    if (negative && n > 0)
    {
        // Fill top n bits with 1s.
        // Build a mask with the top n bits set.
        uint256 mask = bitwise_not(shr(uint256{n}, uint256::max_value()));
        r = bitwise_or(r, mask);
    }
    return r;
}

// -- Division (long division, Knuth Algorithm D simplified) -------------------
//
// EVM division: a / b. If b == 0, result is 0 (not an error).

/// Helper: count leading zero bits in a 64-bit word.
GPU_INLINE gpu_u32 clz64(gpu_u64 x)
{
    if (x == 0) return 64;
    gpu_u32 n = 0;
    if ((x & 0xFFFFFFFF00000000ULL) == 0) { n += 32; x <<= 32; }
    if ((x & 0xFFFF000000000000ULL) == 0) { n += 16; x <<= 16; }
    if ((x & 0xFF00000000000000ULL) == 0) { n +=  8; x <<=  8; }
    if ((x & 0xF000000000000000ULL) == 0) { n +=  4; x <<=  4; }
    if ((x & 0xC000000000000000ULL) == 0) { n +=  2; x <<=  2; }
    if ((x & 0x8000000000000000ULL) == 0) { n +=  1; }
    return n;
}

/// Count leading zeros in a uint256.
GPU_INLINE gpu_u32 clz256(const uint256& x)
{
    if (x.w[3]) return clz64(x.w[3]);
    if (x.w[2]) return 64 + clz64(x.w[2]);
    if (x.w[1]) return 128 + clz64(x.w[1]);
    return 192 + clz64(x.w[0]);
}

/// Division and modulo via binary long division (shift-subtract).
/// Returns {quotient, remainder}.
struct divmod_result { uint256 quot; uint256 rem; };

GPU_INLINE divmod_result divmod(const uint256& a, const uint256& b)
{
    if (iszero(b))
        return {uint256::zero(), uint256::zero()};

    if (lt(a, b))
        return {uint256::zero(), a};

    if (eq(a, b))
        return {uint256::one(), uint256::zero()};

    // Binary long division.
    gpu_u32 shift = clz256(b) - clz256(a);
    uint256 divisor = shl(uint256{shift}, b);
    uint256 quotient = uint256::zero();
    uint256 remainder = a;

    for (gpu_u32 i = 0; i <= shift; ++i)
    {
        quotient = shl(uint256{1}, quotient);
        if (!lt(remainder, divisor))
        {
            remainder = sub(remainder, divisor);
            quotient.w[0] |= 1;
        }
        divisor = shr(uint256{1}, divisor);
    }

    return {quotient, remainder};
}

GPU_INLINE uint256 div(const uint256& a, const uint256& b)
{
    return divmod(a, b).quot;
}

GPU_INLINE uint256 mod(const uint256& a, const uint256& b)
{
    return divmod(a, b).rem;
}

/// Signed division (EVM SDIV).
GPU_INLINE uint256 sdiv(const uint256& a, const uint256& b)
{
    if (iszero(b))
        return uint256::zero();

    bool a_neg = (a.w[3] >> 63) != 0;
    bool b_neg = (b.w[3] >> 63) != 0;

    // Negate if negative: -x = ~x + 1
    uint256 abs_a = a_neg ? add(bitwise_not(a), uint256::one()) : a;
    uint256 abs_b = b_neg ? add(bitwise_not(b), uint256::one()) : b;

    uint256 q = div(abs_a, abs_b);

    // Special case: min_int256 / -1 = min_int256 (overflow, per EVM spec).
    // This naturally works because negate(min_int256) = min_int256 in two's complement.

    if (a_neg != b_neg)
        q = add(bitwise_not(q), uint256::one());  // negate
    return q;
}

/// Signed modulo (EVM SMOD).
GPU_INLINE uint256 smod(const uint256& a, const uint256& b)
{
    if (iszero(b))
        return uint256::zero();

    bool a_neg = (a.w[3] >> 63) != 0;
    bool b_neg = (b.w[3] >> 63) != 0;

    uint256 abs_a = a_neg ? add(bitwise_not(a), uint256::one()) : a;
    uint256 abs_b = b_neg ? add(bitwise_not(b), uint256::one()) : b;

    uint256 r = mod(abs_a, abs_b);

    // Result takes the sign of the dividend (a), per EVM spec.
    if (a_neg && !iszero(r))
        r = add(bitwise_not(r), uint256::one());
    return r;
}

/// ADDMOD: (a + b) % m. Uses 320-bit intermediate to avoid overflow.
GPU_INLINE uint256 addmod(const uint256& a, const uint256& b, const uint256& m)
{
    if (iszero(m))
        return uint256::zero();

    // Compute 320-bit sum (5 limbs).
    auto [s0, c0] = uint256::add_carry(a.w[0], b.w[0], 0);
    auto [s1, c1] = uint256::add_carry(a.w[1], b.w[1], c0);
    auto [s2, c2] = uint256::add_carry(a.w[2], b.w[2], c1);
    auto [s3, c3] = uint256::add_carry(a.w[3], b.w[3], c2);
    // c3 is the 5th limb (0 or 1).

    // If no overflow (c3 == 0), just do regular mod.
    if (c3 == 0)
    {
        uint256 sum{s0, s1, s2, s3};
        return mod(sum, m);
    }

    // Overflow case: the sum is 2^256 + {s0,s1,s2,s3}.
    // We reduce by subtracting m until we fit in 256 bits.
    // Since a < 2^256 and b < 2^256, sum < 2^257, so sum - m < 2^256 + m.
    // At most 2 subtractions needed.
    uint256 sum{s0, s1, s2, s3};
    // First subtract m from the conceptual 257-bit value.
    // (2^256 + sum) - m = sum + (2^256 - m) = sum + (~m + 1)
    uint256 neg_m = add(bitwise_not(m), uint256::one());
    sum = add(sum, neg_m);
    // Now the value fits in 256 bits. Do final mod.
    return mod(sum, m);
}

/// MULMOD: (a * b) % m. Uses 512-bit intermediate.
GPU_INLINE uint256 mulmod(const uint256& a, const uint256& b, const uint256& m)
{
    if (iszero(m))
        return uint256::zero();

    // Full 512-bit product via schoolbook multiplication.
    // We need all 8 limbs of a[0..3] * b[0..3].
    gpu_u64 r[8] = {};

    for (gpu_u32 i = 0; i < 4; ++i)
    {
        gpu_u64 carry = 0;
        for (gpu_u32 j = 0; j < 4; ++j)
        {
            auto [lo, hi] = uint256::mul_wide(a.w[i], b.w[j]);
            gpu_u64 s = r[i + j] + lo;
            gpu_u64 c = (s < r[i + j]) ? gpu_u64(1) : gpu_u64(0);
            s += carry;
            c += (s < carry) ? gpu_u64(1) : gpu_u64(0);
            r[i + j] = s;
            carry = hi + c;
        }
        if (i + 4 < 8)
            r[i + 4] += carry;
    }

    // Reduce 512-bit result mod m using repeated shift-subtract.
    // Build the 512-bit value and reduce limb by limb from the top.
    // This is equivalent to: for each bit from MSB to LSB of the product,
    // shift the intermediate remainder left by 1 and subtract m if >= m.
    uint256 result = uint256::zero();
    for (int bit = 511; bit >= 0; --bit)
    {
        // Shift result left by 1.
        result = shl(uint256{1}, result);
        // Add the current bit of the product.
        gpu_u32 limb = gpu_u32(bit) / 64;
        gpu_u32 pos  = gpu_u32(bit) % 64;
        if ((r[limb] >> pos) & 1)
            result.w[0] |= 1;
        // Reduce: if result >= m, subtract m.
        if (!lt(result, m))
            result = sub(result, m);
    }

    return result;
}

/// EXP: base^exponent mod 2^256. Square-and-multiply.
GPU_INLINE uint256 exp(const uint256& base, const uint256& exponent)
{
    if (iszero(exponent))
        return uint256::one();

    uint256 result = uint256::one();
    uint256 b = base;
    uint256 e = exponent;

    // Find highest set bit to avoid unnecessary multiplications.
    while (!iszero(e))
    {
        if (e.w[0] & 1)
            result = mul(result, b);
        e = shr(uint256{1}, e);
        if (!iszero(e))
            b = mul(b, b);
    }
    return result;
}

/// SIGNEXTEND: sign-extend byte position b of value x.
/// If b >= 31, no-op. Otherwise, extends the sign bit of byte b through
/// the higher bytes.
GPU_INLINE uint256 signextend(const uint256& b_val, const uint256& x)
{
    if (b_val.w[1] | b_val.w[2] | b_val.w[3])
        return x;  // b >= 2^64 > 31, no-op
    gpu_u64 b = b_val.w[0];
    if (b >= 31)
        return x;

    // Bit position of the sign bit: b*8 + 7
    gpu_u64 sign_bit = b * 8 + 7;

    // Extract the sign bit.
    gpu_u32 limb = gpu_u32(sign_bit / 64);
    gpu_u32 pos  = gpu_u32(sign_bit % 64);
    bool negative = ((x.w[limb] >> pos) & 1) != 0;

    // Build a mask: bits [sign_bit+1 .. 255] all set.
    // mask = ~((1 << (sign_bit + 1)) - 1)
    uint256 one_shifted = shl(uint256{sign_bit + 1}, uint256::one());
    uint256 mask = bitwise_not(sub(one_shifted, uint256::one()));

    if (negative)
        return bitwise_or(x, mask);   // set high bits to 1
    else
        return bitwise_and(x, bitwise_not(mask));  // clear high bits
}

// -- Conversion helpers (host only) -------------------------------------------

#if !defined(__METAL_VERSION__) && !defined(__CUDA_ARCH__)

/// Load uint256 from 32-byte big-endian buffer.
inline uint256 from_be_bytes(const uint8_t* data)
{
    uint256 r;
    for (int limb = 3; limb >= 0; --limb)
    {
        gpu_u64 v = 0;
        int byte_start = (3 - limb) * 8;
        for (int b = 0; b < 8; ++b)
            v = (v << 8) | data[byte_start + b];
        r.w[limb] = v;
    }
    return r;
}

/// Store uint256 to 32-byte big-endian buffer.
inline void to_be_bytes(const uint256& val, uint8_t* out)
{
    for (int limb = 3; limb >= 0; --limb)
    {
        gpu_u64 v = val.w[limb];
        int byte_start = (3 - limb) * 8;
        for (int b = 7; b >= 0; --b)
        {
            out[byte_start + b] = static_cast<uint8_t>(v & 0xFF);
            v >>= 8;
        }
    }
}

#endif  // host-only

}  // namespace evm::gpu::kernel
