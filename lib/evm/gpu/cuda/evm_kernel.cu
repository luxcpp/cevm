// Copyright (C) 2026, The cevm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file evm_kernel.cu
/// CUDA port of kernel/evm_kernel.metal — single-thread-per-tx EVM interpreter.
///
/// Mirrors the Metal V1 kernel byte-for-byte:
///   - Same uint256 layout (4x uint64 limbs, w[0] = low)
///   - Same TxInput / TxOutput / StorageEntry / BlockContext struct sizes
///     (matched on the host via static_assert)
///   - Same gas costs and dispatch order
///   - Same status codes: 0=stop, 1=return, 2=revert, 3=oog, 4=error,
///     5=call_not_supported
///
/// Spec coverage: every Cancun-era opcode in mainnet today is implemented
/// here EXCEPT the CALL/CREATE family (0xf0/f1/f2/f4/f5/fa/ff). Those return
/// status=CallNotSupported so the host falls back to CPU cevm.
///
/// Account-state opcodes (BALANCE, EXTCODESIZE, EXTCODEHASH, EXTCODECOPY,
/// SELFBALANCE) operate as if the address has no code and zero balance — this
/// matches Cancun semantics for an unset account and is the safe-default for
/// a kernel that has no Host plumbing. See report for caveats.

#include <cstdint>
#include <cuda_runtime.h>

namespace evm::gpu::cuda
{

// =============================================================================
// uint256 — 4 little-endian 64-bit limbs (matches Metal struct uint256 { ulong w[4]; })
// =============================================================================

struct uint256
{
    unsigned long long w[4];
};

__device__ __forceinline__ uint256 u256_zero()
{
    uint256 r; r.w[0]=0; r.w[1]=0; r.w[2]=0; r.w[3]=0; return r;
}
__device__ __forceinline__ uint256 u256_from(unsigned long long lo)
{
    uint256 r; r.w[0]=lo; r.w[1]=0; r.w[2]=0; r.w[3]=0; return r;
}
__device__ __forceinline__ uint256 u256_one() { return u256_from(1); }
__device__ __forceinline__ uint256 u256_max()
{
    uint256 r; unsigned long long m=~0ULL;
    r.w[0]=m; r.w[1]=m; r.w[2]=m; r.w[3]=m; return r;
}
__device__ __forceinline__ bool u256_iszero(const uint256& a)
{
    return (a.w[0] | a.w[1] | a.w[2] | a.w[3]) == 0ULL;
}
__device__ __forceinline__ bool u256_eq(const uint256& a, const uint256& b)
{
    return a.w[0]==b.w[0] && a.w[1]==b.w[1] && a.w[2]==b.w[2] && a.w[3]==b.w[3];
}
__device__ __forceinline__ bool u256_lt(const uint256& a, const uint256& b)
{
    if (a.w[3] != b.w[3]) return a.w[3] < b.w[3];
    if (a.w[2] != b.w[2]) return a.w[2] < b.w[2];
    if (a.w[1] != b.w[1]) return a.w[1] < b.w[1];
    return a.w[0] < b.w[0];
}
__device__ __forceinline__ bool u256_gt(const uint256& a, const uint256& b)
{
    return u256_lt(b, a);
}

__device__ __forceinline__ uint256 u256_add(const uint256& a, const uint256& b)
{
    uint256 r;
    unsigned long long s0 = a.w[0] + b.w[0];
    unsigned long long c0 = (s0 < a.w[0]) ? 1ULL : 0ULL;
    unsigned long long s1 = a.w[1] + b.w[1] + c0;
    unsigned long long c1 = (s1 < a.w[1] || (c0 && s1 == a.w[1])) ? 1ULL : 0ULL;
    unsigned long long s2 = a.w[2] + b.w[2] + c1;
    unsigned long long c2 = (s2 < a.w[2] || (c1 && s2 == a.w[2])) ? 1ULL : 0ULL;
    r.w[0]=s0; r.w[1]=s1; r.w[2]=s2;
    r.w[3] = a.w[3] + b.w[3] + c2;
    return r;
}
__device__ __forceinline__ uint256 u256_sub(const uint256& a, const uint256& b)
{
    uint256 r;
    unsigned long long d0 = a.w[0] - b.w[0];
    unsigned long long bw0 = (d0 > a.w[0]) ? 1ULL : 0ULL;
    unsigned long long d1 = a.w[1] - b.w[1] - bw0;
    unsigned long long bw1 = (a.w[1] < b.w[1] + bw0 || (bw0 && b.w[1] == ~0ULL)) ? 1ULL : 0ULL;
    unsigned long long d2 = a.w[2] - b.w[2] - bw1;
    unsigned long long bw2 = (a.w[2] < b.w[2] + bw1 || (bw1 && b.w[2] == ~0ULL)) ? 1ULL : 0ULL;
    r.w[0]=d0; r.w[1]=d1; r.w[2]=d2;
    r.w[3] = a.w[3] - b.w[3] - bw2;
    return r;
}
__device__ __forceinline__ uint256 u256_and(const uint256& a, const uint256& b)
{
    uint256 r;
    r.w[0]=a.w[0]&b.w[0]; r.w[1]=a.w[1]&b.w[1];
    r.w[2]=a.w[2]&b.w[2]; r.w[3]=a.w[3]&b.w[3];
    return r;
}
__device__ __forceinline__ uint256 u256_or(const uint256& a, const uint256& b)
{
    uint256 r;
    r.w[0]=a.w[0]|b.w[0]; r.w[1]=a.w[1]|b.w[1];
    r.w[2]=a.w[2]|b.w[2]; r.w[3]=a.w[3]|b.w[3];
    return r;
}
__device__ __forceinline__ uint256 u256_xor(const uint256& a, const uint256& b)
{
    uint256 r;
    r.w[0]=a.w[0]^b.w[0]; r.w[1]=a.w[1]^b.w[1];
    r.w[2]=a.w[2]^b.w[2]; r.w[3]=a.w[3]^b.w[3];
    return r;
}
__device__ __forceinline__ uint256 u256_not(const uint256& a)
{
    uint256 r;
    r.w[0]=~a.w[0]; r.w[1]=~a.w[1];
    r.w[2]=~a.w[2]; r.w[3]=~a.w[3];
    return r;
}

__device__ __forceinline__ uint256 u256_shl(unsigned long long n, const uint256& val)
{
    if (n >= 256) return u256_zero();
    if (n == 0) return val;
    uint256 r = u256_zero();
    unsigned int ls = (unsigned int)(n / 64);
    unsigned int bs = (unsigned int)(n % 64);
    for (unsigned int i = ls; i < 4; ++i)
    {
        r.w[i] = val.w[i - ls] << bs;
        if (bs > 0 && i > ls)
            r.w[i] |= val.w[i - ls - 1] >> (64 - bs);
    }
    return r;
}
__device__ __forceinline__ uint256 u256_shr(unsigned long long n, const uint256& val)
{
    if (n >= 256) return u256_zero();
    if (n == 0) return val;
    uint256 r = u256_zero();
    unsigned int ls = (unsigned int)(n / 64);
    unsigned int bs = (unsigned int)(n % 64);
    for (unsigned int i = 0; i + ls < 4; ++i)
    {
        r.w[i] = val.w[i + ls] >> bs;
        if (bs > 0 && i + ls + 1 < 4)
            r.w[i] |= val.w[i + ls + 1] << (64 - bs);
    }
    return r;
}

struct pair64 { unsigned long long lo; unsigned long long hi; };

__device__ __forceinline__ pair64 mul_wide(unsigned long long a, unsigned long long b)
{
    // CUDA has __umul64hi for the high 64 bits; combined with a*b for low.
    pair64 r;
    r.lo = a * b;
    r.hi = __umul64hi(a, b);
    return r;
}

__device__ __forceinline__ uint256 u256_mul(const uint256& a, const uint256& b)
{
    unsigned long long r[4] = {0, 0, 0, 0};
    for (unsigned int i = 0; i < 4; ++i)
    {
        unsigned long long carry = 0;
        for (unsigned int j = 0; j < 4; ++j)
        {
            if (i + j >= 4) break;
            pair64 p = mul_wide(a.w[i], b.w[j]);
            unsigned long long s = r[i + j] + p.lo;
            unsigned long long c = (s < r[i + j]) ? 1ULL : 0ULL;
            s += carry;
            c += (s < carry) ? 1ULL : 0ULL;
            r[i + j] = s;
            carry = p.hi + c;
        }
    }
    uint256 result;
    result.w[0]=r[0]; result.w[1]=r[1]; result.w[2]=r[2]; result.w[3]=r[3];
    return result;
}

__device__ __forceinline__ unsigned int clz64_dev(unsigned long long x)
{
    if (x == 0) return 64;
    // CUDA __clzll returns 64 for input 0, otherwise count leading zeros.
    return (unsigned int)__clzll((long long)x);
}
__device__ __forceinline__ unsigned int clz256_dev(const uint256& x)
{
    if (x.w[3]) return clz64_dev(x.w[3]);
    if (x.w[2]) return 64 + clz64_dev(x.w[2]);
    if (x.w[1]) return 128 + clz64_dev(x.w[1]);
    return 192 + clz64_dev(x.w[0]);
}

struct divmod_result { uint256 quot; uint256 rem; };

__device__ divmod_result u256_divmod(const uint256& a, const uint256& b)
{
    divmod_result r;
    if (u256_iszero(b)) { r.quot = u256_zero(); r.rem = u256_zero(); return r; }
    if (u256_lt(a, b))   { r.quot = u256_zero(); r.rem = a;           return r; }
    if (u256_eq(a, b))   { r.quot = u256_one();  r.rem = u256_zero(); return r; }

    unsigned int shift = clz256_dev(b) - clz256_dev(a);
    uint256 divisor   = u256_shl((unsigned long long)shift, b);
    uint256 quotient  = u256_zero();
    uint256 remainder = a;

    for (unsigned int i = 0; i <= shift; ++i)
    {
        quotient = u256_shl(1, quotient);
        if (!u256_lt(remainder, divisor))
        {
            remainder = u256_sub(remainder, divisor);
            quotient.w[0] |= 1;
        }
        divisor = u256_shr(1, divisor);
    }
    r.quot = quotient; r.rem = remainder; return r;
}
__device__ __forceinline__ uint256 u256_div(const uint256& a, const uint256& b)
{
    return u256_divmod(a, b).quot;
}
__device__ __forceinline__ uint256 u256_mod(const uint256& a, const uint256& b)
{
    return u256_divmod(a, b).rem;
}
__device__ __forceinline__ uint256 u256_negate(const uint256& x)
{
    return u256_add(u256_not(x), u256_one());
}
__device__ uint256 u256_sdiv(const uint256& a, const uint256& b)
{
    if (u256_iszero(b)) return u256_zero();
    bool an = (a.w[3] >> 63) != 0;
    bool bn = (b.w[3] >> 63) != 0;
    uint256 absa = an ? u256_negate(a) : a;
    uint256 absb = bn ? u256_negate(b) : b;
    uint256 q = u256_div(absa, absb);
    if (an != bn) q = u256_negate(q);
    return q;
}
__device__ uint256 u256_smod(const uint256& a, const uint256& b)
{
    if (u256_iszero(b)) return u256_zero();
    bool an = (a.w[3] >> 63) != 0;
    bool bn = (b.w[3] >> 63) != 0;
    uint256 absa = an ? u256_negate(a) : a;
    uint256 absb = bn ? u256_negate(b) : b;
    uint256 r = u256_mod(absa, absb);
    if (an && !u256_iszero(r)) r = u256_negate(r);
    return r;
}

__device__ uint256 u256_addmod(const uint256& a, const uint256& b, const uint256& m)
{
    if (u256_iszero(m)) return u256_zero();
    unsigned long long s0 = a.w[0] + b.w[0]; unsigned long long c0 = (s0 < a.w[0]) ? 1ULL : 0ULL;
    unsigned long long s1 = a.w[1] + b.w[1] + c0; unsigned long long c1 = (s1 < a.w[1] || (c0 && s1 == a.w[1])) ? 1ULL : 0ULL;
    unsigned long long s2 = a.w[2] + b.w[2] + c1; unsigned long long c2 = (s2 < a.w[2] || (c1 && s2 == a.w[2])) ? 1ULL : 0ULL;
    unsigned long long s3 = a.w[3] + b.w[3] + c2; unsigned long long c3 = (s3 < a.w[3] || (c2 && s3 == a.w[3])) ? 1ULL : 0ULL;
    if (c3 == 0)
    {
        uint256 s; s.w[0]=s0; s.w[1]=s1; s.w[2]=s2; s.w[3]=s3;
        return u256_mod(s, m);
    }
    unsigned long long r5[5] = { s0, s1, s2, s3, c3 };
    uint256 result = u256_zero();
    for (int bit = 256; bit >= 0; --bit)
    {
        result = u256_shl(1, result);
        unsigned int l = (unsigned int)bit / 64;
        unsigned int p = (unsigned int)bit % 64;
        if ((r5[l] >> p) & 1ULL) result.w[0] |= 1ULL;
        if (!u256_lt(result, m)) result = u256_sub(result, m);
    }
    return result;
}

// FIXED (Bug 1): the old loop dropped the i=3 carry (because the guard
// `if (i + 4 < 8)` is false for i=3) and also failed to propagate further
// carries from r8[i+4] += if that wraparound. Use 128-bit arithmetic to
// accumulate, then propagate the carry chain through ALL remaining limbs.
__device__ uint256 u256_mulmod(const uint256& a, const uint256& b, const uint256& m)
{
    if (u256_iszero(m)) return u256_zero();
    unsigned long long r8[8] = { 0,0,0,0,0,0,0,0 };
    for (unsigned int i = 0; i < 4; ++i)
    {
        unsigned long long carry = 0;
        for (unsigned int j = 0; j < 4; ++j)
        {
            pair64 p = mul_wide(a.w[i], b.w[j]);
            unsigned __int128 t = (unsigned __int128)r8[i + j] + p.lo + carry;
            r8[i + j] = (unsigned long long)t;
            carry     = (unsigned long long)(t >> 64) + p.hi;
        }
        // Propagate the remaining carry through ALL higher limbs (not just one).
        unsigned int k = i + 4;
        while (carry && k < 8)
        {
            unsigned __int128 t = (unsigned __int128)r8[k] + carry;
            r8[k] = (unsigned long long)t;
            carry = (unsigned long long)(t >> 64);
            ++k;
        }
    }
    uint256 result = u256_zero();
    for (int bit = 511; bit >= 0; --bit)
    {
        result = u256_shl(1, result);
        unsigned int l = (unsigned int)bit / 64;
        unsigned int p = (unsigned int)bit % 64;
        if ((r8[l] >> p) & 1ULL) result.w[0] |= 1ULL;
        if (!u256_lt(result, m)) result = u256_sub(result, m);
    }
    return result;
}

__device__ uint256 u256_exp(const uint256& base, const uint256& exponent)
{
    if (u256_iszero(exponent)) return u256_one();
    uint256 result = u256_one();
    uint256 b = base;
    uint256 e = exponent;
    while (!u256_iszero(e))
    {
        if (e.w[0] & 1ULL) result = u256_mul(result, b);
        e = u256_shr(1, e);
        if (!u256_iszero(e)) b = u256_mul(b, b);
    }
    return result;
}

__device__ uint256 u256_signextend(const uint256& b_val, const uint256& x)
{
    if (b_val.w[1] | b_val.w[2] | b_val.w[3]) return x;
    unsigned long long b = b_val.w[0];
    if (b >= 31) return x;
    unsigned long long sb = b * 8 + 7;
    unsigned int l = (unsigned int)(sb / 64);
    unsigned int p = (unsigned int)(sb % 64);
    bool neg = ((x.w[l] >> p) & 1ULL) != 0;
    uint256 mask = u256_not(u256_sub(u256_shl(sb + 1, u256_one()), u256_one()));
    return neg ? u256_or(x, mask) : u256_and(x, u256_not(mask));
}

__device__ __forceinline__ bool u256_slt(const uint256& a, const uint256& b)
{
    bool an = (a.w[3] >> 63) != 0;
    bool bn = (b.w[3] >> 63) != 0;
    if (an != bn) return an;
    return u256_lt(a, b);
}
__device__ __forceinline__ bool u256_sgt(const uint256& a, const uint256& b)
{
    return u256_slt(b, a);
}

__device__ uint256 u256_byte_at(const uint256& val, const uint256& pos)
{
    if (pos.w[1] | pos.w[2] | pos.w[3]) return u256_zero();
    unsigned long long i = pos.w[0];
    if (i >= 32) return u256_zero();
    unsigned int bfr = (unsigned int)(31 - i);
    return u256_from((val.w[bfr / 8] >> ((bfr % 8) * 8)) & 0xFFULL);
}

__device__ uint256 u256_sar(unsigned long long n, const uint256& val)
{
    bool neg = (val.w[3] >> 63) != 0;
    if (n >= 256) return neg ? u256_max() : u256_zero();
    uint256 r = u256_shr(n, val);
    if (neg && n > 0)
        r = u256_or(r, u256_not(u256_shr(n, u256_max())));
    return r;
}

__device__ unsigned int u256_byte_length(const uint256& x)
{
    if (u256_iszero(x)) return 0;
    return (256 - clz256_dev(x) + 7) / 8;
}

// =============================================================================
// Keccak-256 (Ethereum, NOT NIST SHA-3) — used by KECCAK256 opcode (0x20).
// Standalone single-thread implementation. Mirrors gpu/kernels/keccak256.cu
// but inline so we don't have to link a separate translation unit into the
// kernel module.
// =============================================================================

__device__ static const unsigned long long KECCAK_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};
__device__ static const int KECCAK_ROTC[24] = {
    1,3,6,10,15,21,28,36,45,55,2,14,27,41,56,8,25,43,62,18,39,61,20,44
};
__device__ static const int KECCAK_PI[24] = {
    10,7,11,17,18,3,5,16,8,21,24,4,15,23,19,13,12,2,20,14,22,9,6,1
};

__device__ void keccak_f1600(unsigned long long* state)
{
    for (int round = 0; round < 24; ++round)
    {
        unsigned long long C[5], D[5];
        for (int x = 0; x < 5; ++x)
            C[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        for (int x = 0; x < 5; ++x)
        {
            D[x] = C[(x + 4) % 5]
                 ^ ((C[(x + 1) % 5] << 1) | (C[(x + 1) % 5] >> 63));
            for (int y = 0; y < 25; y += 5)
                state[y + x] ^= D[x];
        }
        unsigned long long t = state[1];
        for (int i = 0; i < 24; ++i)
        {
            int j = KECCAK_PI[i];
            unsigned long long tmp = state[j];
            state[j] = (t << KECCAK_ROTC[i]) | (t >> (64 - KECCAK_ROTC[i]));
            t = tmp;
        }
        for (int y = 0; y < 25; y += 5)
        {
            unsigned long long t0 = state[y], t1 = state[y+1], t2 = state[y+2],
                               t3 = state[y+3], t4 = state[y+4];
            state[y]   = t0 ^ (~t1 & t2);
            state[y+1] = t1 ^ (~t2 & t3);
            state[y+2] = t2 ^ (~t3 & t4);
            state[y+3] = t3 ^ (~t4 & t0);
            state[y+4] = t4 ^ (~t0 & t1);
        }
        state[0] ^= KECCAK_RC[round];
    }
}

/// Keccak-256 over `[input, input+len)`. Writes 32 bytes to `output`.
/// Output is the digest in big-endian byte order (Ethereum convention).
__device__ void keccak256_dev(const unsigned char* input, unsigned int len,
                              unsigned char output[32])
{
    unsigned long long state[25];
    for (int i = 0; i < 25; ++i) state[i] = 0;

    constexpr unsigned int RATE = 136;  // 1088 bits
    unsigned int pos = 0;

    while (pos + RATE <= len)
    {
        for (unsigned int i = 0; i < 17; ++i)
        {
            unsigned long long word = 0;
            for (unsigned int b = 0; b < 8; ++b)
                word |= (unsigned long long)input[pos + i * 8 + b] << (b * 8);
            state[i] ^= word;
        }
        keccak_f1600(state);
        pos += RATE;
    }

    unsigned char block[RATE];
    for (unsigned int i = 0; i < RATE; ++i) block[i] = 0;
    unsigned int rem = len - pos;
    for (unsigned int i = 0; i < rem; ++i) block[i] = input[pos + i];
    block[rem]      |= 0x01;
    block[RATE - 1] |= 0x80;

    for (unsigned int i = 0; i < 17; ++i)
    {
        unsigned long long word = 0;
        for (unsigned int b = 0; b < 8; ++b)
            word |= (unsigned long long)block[i * 8 + b] << (b * 8);
        state[i] ^= word;
    }
    keccak_f1600(state);

    // Squeeze 32 bytes (big-endian by convention here: byte 0 is the most
    // significant byte of the 256-bit digest. We write each lane LE because
    // the EVM later reads the 32-byte output through MLOAD which is BE — but
    // KECCAK256 *result* in EVM is the big-endian word value pushed onto
    // the stack. Convention: we return raw little-endian bytes here and
    // load them into uint256 limbs preserving order.)
    for (int i = 0; i < 4; ++i)
        for (int b = 0; b < 8; ++b)
            output[i * 8 + b] = (unsigned char)((state[i] >> (b * 8)) & 0xFFULL);
}

// =============================================================================
// GPU buffer descriptors — must match Metal evm_kernel.metal layouts EXACTLY.
// Host file evm_kernel_host.hpp also defines a TxInput/TxOutput; we keep
// the on-device layout identical so the host can memcpy into our buffers.
// =============================================================================

struct TxInput
{
    unsigned int        code_offset;
    unsigned int        code_size;
    unsigned int        calldata_offset;
    unsigned int        calldata_size;
    unsigned long long  gas_limit;
    uint256             caller;
    uint256             address;
    uint256             value;
    // EIP-2929 caller-supplied warm sets. Offsets index into the same
    // `blob` buffer that holds code+calldata. Layouts:
    //   warm_addrs blob: 20-byte addresses, packed.
    //   warm_slots blob: (20-byte addr | 32-byte slot key) pairs, packed.
    unsigned int        warm_addr_offset;
    unsigned int        warm_addr_count;
    unsigned int        warm_slot_offset;
    unsigned int        warm_slot_count;
};

// gas_refund is signed: SSTORE may transiently subtract refund credit
// (clear-then-set-then-clear within a tx). The kernel emits the raw
// signed value; the host dispatcher floors at 0 and applies the EIP-3529
// cap of gas_used/5.
struct TxOutput
{
    unsigned int        status;
    unsigned long long  gas_used;
    long long           gas_refund;
    unsigned int        output_size;
};

struct StorageEntry
{
    uint256 key;
    uint256 value;
};

// Block-level context — same for every tx in a block. ABI v3 addition.
// Layout MUST match Metal kernel BlockContext struct.
struct BlockContext
{
    uint256             origin;            // 0x32 ORIGIN
    unsigned long long  gas_price;         // 0x3a GASPRICE (top 64 bits, fits Cancun)
    unsigned long long  timestamp;         // 0x42 TIMESTAMP
    unsigned long long  number;            // 0x43 NUMBER
    uint256             prevrandao;        // 0x44 PREVRANDAO (full 256)
    unsigned long long  gas_limit;         // 0x45 GASLIMIT
    unsigned long long  chain_id;          // 0x46 CHAINID
    unsigned long long  base_fee;          // 0x48 BASEFEE
    unsigned long long  blob_base_fee;     // 0x4a BLOBBASEFEE
    uint256             coinbase;          // 0x41 COINBASE (20 bytes right-aligned)
    unsigned char       blob_hashes[8][32];// 0x49 BLOBHASH (max 8, BE bytes)
    unsigned int        num_blob_hashes;
    unsigned int        _pad0;             // align to 8
};

// Log entry — written into per-tx log pool. Topics are uint256, data is held
// inside the per-tx memory buffer (we record offset+size, not a copy, since
// memory for the tx is preserved in the device buffer until D2H).
struct LogEntry
{
    uint256       topics[4];
    unsigned int  num_topics;
    unsigned int  data_offset;   // offset into tx's memory pool
    unsigned int  data_size;
    unsigned int  _pad0;
};

// Validate that the device-side wire layout matches the host expectations.
static_assert(sizeof(uint256)      == 32,  "device uint256 size");
static_assert(sizeof(TxInput)      == 136, "device TxInput size");
static_assert(sizeof(TxOutput)     == 32,  "device TxOutput size");
static_assert(sizeof(StorageEntry) == 64,  "device StorageEntry size");
static_assert(sizeof(BlockContext) == 32 + 8 + 8 + 8 + 32 + 8 + 8 + 8 + 8 + 32 + 8*32 + 4 + 4,
              "device BlockContext size");
static_assert(sizeof(LogEntry)     == 4*32 + 4 + 4 + 4 + 4, "device LogEntry size");

// =============================================================================
// Constants — must match Metal evm_kernel.metal.
// =============================================================================

__device__ static constexpr unsigned int MAX_MEMORY_PER_TX  = 65536;
__device__ static constexpr unsigned int MAX_OUTPUT_PER_TX  = 1024;
__device__ static constexpr unsigned int MAX_STORAGE_PER_TX = 64;
__device__ static constexpr unsigned int MAX_LOGS_PER_TX    = 16;
// Mirrors blob_hashes[8] in BlockContext. The kernel must never index
// past this even if the host writes a larger num_blob_hashes.
__device__ static constexpr unsigned int MAX_BLOB_HASHES    = 8;

__device__ static constexpr unsigned long long GAS_VERYLOW    = 3;
__device__ static constexpr unsigned long long GAS_LOW        = 5;
__device__ static constexpr unsigned long long GAS_MID        = 8;
__device__ static constexpr unsigned long long GAS_HIGH       = 10;
__device__ static constexpr unsigned long long GAS_BASE       = 2;
__device__ static constexpr unsigned long long GAS_JUMPDEST   = 1;
// EIP-2929 (Berlin) cold/warm pricing for state-access opcodes.
__device__ static constexpr unsigned long long GAS_SLOAD_COLD = 2100;
__device__ static constexpr unsigned long long GAS_SLOAD_WARM = 100;
__device__ static constexpr unsigned long long GAS_ACCOUNT_COLD = 2600;
__device__ static constexpr unsigned long long GAS_ACCOUNT_WARM = 100;
__device__ static constexpr unsigned long long GAS_SLOAD = GAS_SLOAD_COLD;  // legacy
__device__ static constexpr unsigned long long GAS_SSTORE_SET   = 20000;
__device__ static constexpr unsigned long long GAS_SSTORE_RESET = 2900;
__device__ static constexpr unsigned long long GAS_SSTORE_NOOP   = 100;
__device__ static constexpr unsigned long long GAS_SSTORE_REFUND = 4800;
__device__ static constexpr unsigned long long GAS_MEMORY     = 3;
__device__ static constexpr unsigned long long GAS_EXP_BASE   = 10;
__device__ static constexpr unsigned long long GAS_EXP_BYTE   = 50;
__device__ static constexpr unsigned long long GAS_KECCAK_BASE  = 30;
__device__ static constexpr unsigned long long GAS_KECCAK_WORD  = 6;
__device__ static constexpr unsigned long long GAS_COPY         = 3;
__device__ static constexpr unsigned long long GAS_LOG_BASE     = 375;
__device__ static constexpr unsigned long long GAS_LOG_TOPIC    = 375;
__device__ static constexpr unsigned long long GAS_LOG_DATA     = 8;
__device__ static constexpr unsigned long long GAS_SELFBALANCE  = 5;
__device__ static constexpr unsigned long long GAS_TLOAD        = 100;
__device__ static constexpr unsigned long long GAS_TSTORE       = 100;
__device__ static constexpr unsigned long long GAS_BLOCKHASH    = 20;
// EIP-2929 per-tx warm-set caps. Saturation falls back to cold pricing.
__device__ static constexpr unsigned int MAX_WARM_ADDRS_2929 = 64;
__device__ static constexpr unsigned int MAX_WARM_SLOTS_2929 = 128;
// Pre-warmed precompile range (Cancun→Prague superset).
__device__ static constexpr unsigned int PRECOMPILE_FIRST = 1;
__device__ static constexpr unsigned int PRECOMPILE_LAST  = 0x11;

// keccak256 of empty input (hex). EXTCODEHASH of an account with no code.
__device__ static const unsigned char KECCAK_EMPTY[32] = {
    0xc5, 0xd2, 0x46, 0x01, 0x86, 0xf7, 0x23, 0x3c,
    0x92, 0x7e, 0x7d, 0xb2, 0xdc, 0xc7, 0x03, 0xc0,
    0xe5, 0x00, 0xb6, 0x53, 0xca, 0x82, 0x27, 0x3b,
    0x7b, 0xfa, 0xd8, 0x04, 0x5d, 0x85, 0xa4, 0x70,
};

// =============================================================================
// JUMPDEST validation — same algorithm as Metal: scan from byte 0 skipping
// PUSHN immediates; valid only if the linear cursor lands exactly on target
// AND code[target] == 0x5b.
// =============================================================================

__device__ bool is_valid_jumpdest(const unsigned char* __restrict__ code,
                                  unsigned int code_size,
                                  unsigned int target)
{
    if (target >= code_size || code[target] != 0x5b) return false;
    unsigned int i = 0;
    while (i < target)
    {
        unsigned char op = code[i];
        if (op >= 0x60 && op <= 0x7f) i += (op - 0x60 + 2);
        else                          i += 1;
    }
    return i == target;
}

// =============================================================================
// EIP-2200 SSTORE gas + refund. Mirrors Metal's helpers.
// =============================================================================

struct OriginalEntry
{
    uint256 key;
    uint256 value;
    bool    valid;
};

__device__ bool original_value_lookup(OriginalEntry* o, unsigned int n,
                                      const uint256& slot, uint256& v)
{
    for (unsigned int i = 0; i < n; ++i)
    {
        if (o[i].valid && u256_eq(o[i].key, slot))
        {
            v = o[i].value;
            return true;
        }
    }
    v = u256_zero();
    return false;
}

__device__ void original_value_record(OriginalEntry* o, unsigned int& n,
                                      const uint256& slot, const uint256& v)
{
    for (unsigned int i = 0; i < n; ++i)
        if (o[i].valid && u256_eq(o[i].key, slot)) return;
    if (n < MAX_STORAGE_PER_TX)
    {
        o[n].key = slot;
        o[n].value = v;
        o[n].valid = true;
        n++;
    }
}

// EIP-2929 warm-set helpers. Returns true iff the key was already warm.
__device__ __forceinline__ bool warm_addr_mark_2929(uint256* set, unsigned int& n,
                                                    const uint256& a)
{
    for (unsigned int i = 0; i < n; ++i) if (u256_eq(set[i], a)) return true;
    if (n < MAX_WARM_ADDRS_2929) { set[n++] = a; }
    return false;
}
__device__ __forceinline__ bool warm_slot_mark_2929(uint256* addrs, uint256* keys,
                                                    unsigned int& n,
                                                    const uint256& addr,
                                                    const uint256& slot)
{
    for (unsigned int i = 0; i < n; ++i)
        if (u256_eq(addrs[i], addr) && u256_eq(keys[i], slot)) return true;
    if (n < MAX_WARM_SLOTS_2929) { addrs[n] = addr; keys[n] = slot; n++; }
    return false;
}

// EIP-2200 net gas SSTORE accounting. `rc` accumulates the *signed* refund
// delta — clear-then-set-then-clear within a single tx produces a transient
// negative value. The dispatcher applies EIP-3529 (max refund = gas_used/5)
// and floors at 0 after execution.
__device__ unsigned long long sstore_gas_eip2200(const uint256& orig,
                                                 const uint256& cur,
                                                 const uint256& nv,
                                                 long long& rc)
{
    if (u256_eq(nv, cur)) return GAS_SSTORE_NOOP;
    if (u256_eq(orig, cur))
    {
        if (u256_iszero(orig))      return GAS_SSTORE_SET;
        if (u256_iszero(nv))        rc += static_cast<long long>(GAS_SSTORE_REFUND);
        return GAS_SSTORE_RESET;
    }
    if (!u256_iszero(orig))
    {
        if (u256_iszero(cur))       rc -= static_cast<long long>(GAS_SSTORE_REFUND);
        else if (u256_iszero(nv))   rc += static_cast<long long>(GAS_SSTORE_REFUND);
    }
    if (u256_eq(nv, orig))
    {
        if (u256_iszero(orig))      rc += static_cast<long long>(GAS_SSTORE_SET   - GAS_SSTORE_NOOP);
        else                        rc += static_cast<long long>(GAS_SSTORE_RESET - GAS_SSTORE_NOOP);
    }
    return GAS_SSTORE_NOOP;
}

// =============================================================================
// Memory expansion helper. Charges incremental gas; returns false on OOG.
// FIXED (Bug 3): the old test ov.w[0] + 32 > MAX_MEMORY_PER_TX wraps when
// ov.w[0] is near 2^64. We now check the high limbs and the raw value first.
// =============================================================================

__device__ __forceinline__ bool offset_in_bounds(const uint256& v,
                                                 unsigned long long extra)
{
    // Reject if any high limb is non-zero, OR if w[0] exceeds the cap, OR if
    // w[0] + extra exceeds the cap (and the addition itself doesn't wrap).
    if (v.w[1] | v.w[2] | v.w[3]) return false;
    if (v.w[0] > MAX_MEMORY_PER_TX) return false;
    unsigned long long sum = v.w[0] + extra;
    if (sum < v.w[0]) return false;            // wrap
    if (sum > MAX_MEMORY_PER_TX) return false;
    return true;
}

__device__ __forceinline__ bool expand_mem_words(unsigned char* mem,
                                                 unsigned int& mem_size,
                                                 unsigned long long& gas,
                                                 unsigned int new_words)
{
    unsigned int old_words = mem_size / 32;
    if (new_words <= old_words) return true;
    unsigned long long cost =
        GAS_MEMORY * new_words + ((unsigned long long)new_words * new_words) / 512
        - GAS_MEMORY * old_words - ((unsigned long long)old_words * old_words) / 512;
    if (gas < cost) return false;
    gas -= cost;
    unsigned int new_size = new_words * 32;
    for (unsigned int i = mem_size; i < new_size; ++i) mem[i] = 0;
    mem_size = new_size;
    return true;
}

__device__ __forceinline__ bool expand_mem_range(unsigned char* mem,
                                                 unsigned int& mem_size,
                                                 unsigned long long& gas,
                                                 unsigned int offset,
                                                 unsigned int size)
{
    if (size == 0) return true;
    unsigned int end = offset + size;
    if (end < offset || end > MAX_MEMORY_PER_TX) return false;
    unsigned int new_words = (end + 31) / 32;
    return expand_mem_words(mem, mem_size, gas, new_words);
}

// =============================================================================
// Main kernel — single-thread-per-tx interpreter.
// =============================================================================

__global__ void evm_execute_kernel(
    const TxInput*       __restrict__ inputs,
    const unsigned char* __restrict__ blob,
    TxOutput*            __restrict__ outputs,
    unsigned char*       __restrict__ out_data,
    unsigned char*       __restrict__ mem_pool,
    StorageEntry*        __restrict__ storage_pool,
    unsigned int*        __restrict__ storage_counts,
    StorageEntry*        __restrict__ transient_pool,
    unsigned int*        __restrict__ transient_counts,
    LogEntry*            __restrict__ log_pool,
    unsigned int*        __restrict__ log_counts,
    const BlockContext*  __restrict__ ctx,
    const unsigned int*  __restrict__ params)
{
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int num_txs = params[0];
    if (tid >= num_txs) return;

    const TxInput inp = inputs[tid];
    TxOutput& out = outputs[tid];

    unsigned char* mem    = mem_pool + (unsigned long long)tid * MAX_MEMORY_PER_TX;
    unsigned char* output = out_data + (unsigned long long)tid * MAX_OUTPUT_PER_TX;
    StorageEntry*  storage = storage_pool + (unsigned long long)tid * MAX_STORAGE_PER_TX;
    unsigned int&  stor_count = storage_counts[tid];
    StorageEntry*  transient = transient_pool
                              ? transient_pool + (unsigned long long)tid * MAX_STORAGE_PER_TX
                              : nullptr;
    unsigned int*  trans_count_ptr = transient_counts ? &transient_counts[tid] : nullptr;
    LogEntry*      logs = log_pool ? log_pool + (unsigned long long)tid * MAX_LOGS_PER_TX
                                   : nullptr;
    unsigned int*  log_count_ptr = log_counts ? &log_counts[tid] : nullptr;

    const unsigned char* code_dev = blob + inp.code_offset;
    const unsigned int   code_size = inp.code_size;
    const unsigned char* calldata = blob + inp.calldata_offset;
    const unsigned int   calldata_size = inp.calldata_size;

    // Cache up to first 256 bytes of bytecode in registers/local for quick fetch.
    unsigned char code_cache[256];
    const unsigned int cached_size = (code_size < 256) ? code_size : 256u;
    for (unsigned int i = 0; i < cached_size; ++i) code_cache[i] = code_dev[i];

    #define CODE_BYTE(idx) ((idx) < cached_size ? code_cache[(idx)] : code_dev[(idx)])

    // EVM Yellow Paper: 1024-deep stack. Solidity-emitted bytecode regularly
    // hits 30-50 entries; the previous 32-cap silently truncated real
    // contracts and diverged from CPU cevm. nvcc places the array in
    // per-thread local memory, backed by global memory with L1/L2 caching
    // (32 KB/thread).
    uint256 stack[1024];
    unsigned int       sp  = 0;
    unsigned long long gas = inp.gas_limit;
    long long          refund_counter = 0;
    unsigned int       pc = 0;
    unsigned int       mem_size = 0;
    const unsigned long long gas_start = gas;

    OriginalEntry orig_storage[MAX_STORAGE_PER_TX];
    unsigned int  orig_count = 0;
    for (unsigned int i = 0; i < MAX_STORAGE_PER_TX; ++i) orig_storage[i].valid = false;

    // EIP-2929 per-tx warm sets. Seed with caller, recipient,
    // precompiles 0x01..0x11, and any caller-supplied entries.
    uint256 warm_addrs[MAX_WARM_ADDRS_2929];
    unsigned int warm_addr_count = 0;
    uint256 warm_slot_addrs[MAX_WARM_SLOTS_2929];
    uint256 warm_slot_keys[MAX_WARM_SLOTS_2929];
    unsigned int warm_slot_count = 0;
    warm_addr_mark_2929(warm_addrs, warm_addr_count, inp.caller);
    warm_addr_mark_2929(warm_addrs, warm_addr_count, inp.address);
    for (unsigned int p = PRECOMPILE_FIRST; p <= PRECOMPILE_LAST; ++p) {
        uint256 a = u256_zero(); a.w[0] = (unsigned long long)p;
        warm_addr_mark_2929(warm_addrs, warm_addr_count, a);
    }
    {
        const unsigned char* blob_warm_a = blob + inp.warm_addr_offset;
        for (unsigned int i = 0; i < inp.warm_addr_count; ++i) {
            // 20 BE address bytes → big-int uint256 (matches PUSH).
            uint256 a = u256_zero();
            for (unsigned int b = 0; b < 20; ++b) {
                unsigned int pfr = 19 - b;
                a.w[pfr / 8] |= (unsigned long long)blob_warm_a[i * 20 + b] << ((pfr % 8) * 8);
            }
            warm_addr_mark_2929(warm_addrs, warm_addr_count, a);
        }
    }
    {
        const unsigned char* blob_warm_s = blob + inp.warm_slot_offset;
        for (unsigned int i = 0; i < inp.warm_slot_count; ++i) {
            // Address is 20 BE bytes (matches PUSH-derived addresses).
            uint256 a = u256_zero();
            for (unsigned int b = 0; b < 20; ++b) {
                unsigned int pfr = 19 - b;
                a.w[pfr / 8] |= (unsigned long long)blob_warm_s[i * 52 + b] << ((pfr % 8) * 8);
            }
            uint256 k = u256_zero();
            for (unsigned int b = 0; b < 32; ++b) {
                unsigned int pfr = 31 - b;
                k.w[pfr / 8] |= (unsigned long long)blob_warm_s[i * 52 + 20 + b] << ((pfr % 8) * 8);
            }
            warm_slot_mark_2929(warm_slot_addrs, warm_slot_keys, warm_slot_count, a, k);
        }
    }

    #define EMIT(st, gu, gr, os) do { \
        out.status     = (st); \
        out.gas_used   = (gu); \
        out.gas_refund = (gr); \
        out.output_size = (os); \
        return; \
    } while (0)
    #define OOG()  EMIT(3, gas_start, refund_counter, 0)
    #define ERR()  EMIT(4, gas_start - gas, refund_counter, 0)
    // INVALID-style: consume all gas. Used for malformed bytecode that the
    // EVM treats like the explicit 0xFE INVALID opcode.
    #define ERRA() EMIT(4, gas_start, refund_counter, 0)

    while (pc < code_size)
    {
        const unsigned char op = CODE_BYTE(pc);

        switch (op)
        {
        case 0x00: // STOP
            EMIT(0, gas_start - gas, refund_counter, 0);

        case 0x01: { // ADD
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 2) ERR();
            uint256 a = stack[--sp];
            stack[sp - 1] = u256_add(a, stack[sp - 1]);
            ++pc; continue;
        }
        case 0x02: { // MUL
            if (gas < GAS_LOW) OOG(); gas -= GAS_LOW;
            if (sp < 2) ERR();
            uint256 a = stack[--sp];
            stack[sp - 1] = u256_mul(a, stack[sp - 1]);
            ++pc; continue;
        }
        case 0x03: { // SUB
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 2) ERR();
            uint256 a = stack[--sp];
            stack[sp - 1] = u256_sub(a, stack[sp - 1]);
            ++pc; continue;
        }
        case 0x04: { // DIV
            if (gas < GAS_LOW) OOG(); gas -= GAS_LOW;
            if (sp < 2) ERR();
            uint256 a = stack[--sp];
            stack[sp - 1] = u256_div(a, stack[sp - 1]);
            ++pc; continue;
        }
        case 0x05: { // SDIV
            if (gas < GAS_LOW) OOG(); gas -= GAS_LOW;
            if (sp < 2) ERR();
            uint256 a = stack[--sp];
            stack[sp - 1] = u256_sdiv(a, stack[sp - 1]);
            ++pc; continue;
        }
        case 0x06: { // MOD
            if (gas < GAS_LOW) OOG(); gas -= GAS_LOW;
            if (sp < 2) ERR();
            uint256 a = stack[--sp];
            stack[sp - 1] = u256_mod(a, stack[sp - 1]);
            ++pc; continue;
        }
        case 0x07: { // SMOD
            if (gas < GAS_LOW) OOG(); gas -= GAS_LOW;
            if (sp < 2) ERR();
            uint256 a = stack[--sp];
            stack[sp - 1] = u256_smod(a, stack[sp - 1]);
            ++pc; continue;
        }
        case 0x08: { // ADDMOD
            if (gas < GAS_MID) OOG(); gas -= GAS_MID;
            if (sp < 3) ERR();
            uint256 a = stack[--sp];
            uint256 b = stack[--sp];
            stack[sp - 1] = u256_addmod(a, b, stack[sp - 1]);
            ++pc; continue;
        }
        case 0x09: { // MULMOD
            if (gas < GAS_MID) OOG(); gas -= GAS_MID;
            if (sp < 3) ERR();
            uint256 a = stack[--sp];
            uint256 b = stack[--sp];
            stack[sp - 1] = u256_mulmod(a, b, stack[sp - 1]);
            ++pc; continue;
        }
        case 0x0a: { // EXP
            if (gas < GAS_EXP_BASE) OOG(); gas -= GAS_EXP_BASE;
            if (sp < 2) ERR();
            uint256 a = stack[--sp];
            uint256 b = stack[sp - 1];
            unsigned long long eg = GAS_EXP_BYTE * (unsigned long long)u256_byte_length(b);
            if (gas < eg) OOG(); gas -= eg;
            stack[sp - 1] = u256_exp(a, b);
            ++pc; continue;
        }
        case 0x0b: { // SIGNEXTEND
            if (gas < GAS_LOW) OOG(); gas -= GAS_LOW;
            if (sp < 2) ERR();
            uint256 a = stack[--sp];
            stack[sp - 1] = u256_signextend(a, stack[sp - 1]);
            ++pc; continue;
        }

        case 0x10: { // LT
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 2) ERR();
            uint256 a = stack[--sp];
            stack[sp - 1] = u256_lt(a, stack[sp - 1]) ? u256_one() : u256_zero();
            ++pc; continue;
        }
        case 0x11: { // GT
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 2) ERR();
            uint256 a = stack[--sp];
            stack[sp - 1] = u256_gt(a, stack[sp - 1]) ? u256_one() : u256_zero();
            ++pc; continue;
        }
        case 0x12: { // SLT
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 2) ERR();
            uint256 a = stack[--sp];
            stack[sp - 1] = u256_slt(a, stack[sp - 1]) ? u256_one() : u256_zero();
            ++pc; continue;
        }
        case 0x13: { // SGT
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 2) ERR();
            uint256 a = stack[--sp];
            stack[sp - 1] = u256_sgt(a, stack[sp - 1]) ? u256_one() : u256_zero();
            ++pc; continue;
        }
        case 0x14: { // EQ
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 2) ERR();
            uint256 a = stack[--sp];
            stack[sp - 1] = u256_eq(a, stack[sp - 1]) ? u256_one() : u256_zero();
            ++pc; continue;
        }
        case 0x15: { // ISZERO
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 1) ERR();
            stack[sp - 1] = u256_iszero(stack[sp - 1]) ? u256_one() : u256_zero();
            ++pc; continue;
        }
        case 0x16: { // AND
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 2) ERR();
            uint256 a = stack[--sp];
            stack[sp - 1] = u256_and(a, stack[sp - 1]);
            ++pc; continue;
        }
        case 0x17: { // OR
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 2) ERR();
            uint256 a = stack[--sp];
            stack[sp - 1] = u256_or(a, stack[sp - 1]);
            ++pc; continue;
        }
        case 0x18: { // XOR
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 2) ERR();
            uint256 a = stack[--sp];
            stack[sp - 1] = u256_xor(a, stack[sp - 1]);
            ++pc; continue;
        }
        case 0x19: { // NOT
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 1) ERR();
            stack[sp - 1] = u256_not(stack[sp - 1]);
            ++pc; continue;
        }
        case 0x1a: { // BYTE
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 2) ERR();
            uint256 i = stack[--sp];
            stack[sp - 1] = u256_byte_at(stack[sp - 1], i);
            ++pc; continue;
        }
        case 0x1b: { // SHL
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 2) ERR();
            uint256 s = stack[--sp];
            uint256 v = stack[sp - 1];
            stack[sp - 1] = (s.w[1] | s.w[2] | s.w[3] || s.w[0] >= 256)
                                ? u256_zero()
                                : u256_shl(s.w[0], v);
            ++pc; continue;
        }
        case 0x1c: { // SHR
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 2) ERR();
            uint256 s = stack[--sp];
            uint256 v = stack[sp - 1];
            stack[sp - 1] = (s.w[1] | s.w[2] | s.w[3] || s.w[0] >= 256)
                                ? u256_zero()
                                : u256_shr(s.w[0], v);
            ++pc; continue;
        }
        case 0x1d: { // SAR
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 2) ERR();
            uint256 s = stack[--sp];
            uint256 v = stack[sp - 1];
            bool neg = (v.w[3] >> 63) != 0;
            stack[sp - 1] = (s.w[1] | s.w[2] | s.w[3] || s.w[0] >= 256)
                                ? (neg ? u256_max() : u256_zero())
                                : u256_sar(s.w[0], v);
            ++pc; continue;
        }

        case 0x20: { // KECCAK256
            if (gas < GAS_KECCAK_BASE) OOG(); gas -= GAS_KECCAK_BASE;
            if (sp < 2) ERR();
            uint256 ov = stack[--sp];
            uint256 sv = stack[--sp];
            // size must fit in 32 bits AND offset+size must fit in memory.
            if (sv.w[1] | sv.w[2] | sv.w[3]) ERR();
            unsigned long long size = sv.w[0];
            if (!offset_in_bounds(ov, size)) ERR();
            unsigned int sz = (unsigned int)size;
            unsigned int off = (unsigned int)ov.w[0];
            unsigned long long words = (size + 31) / 32;
            unsigned long long word_gas = GAS_KECCAK_WORD * words;
            if (gas < word_gas) OOG(); gas -= word_gas;
            if (!expand_mem_range(mem, mem_size, gas, off, sz)) OOG();
            unsigned char digest[32];
            keccak256_dev(mem + off, sz, digest);
            // Load digest as a uint256: bytes are big-endian (byte 0 = MSB).
            uint256 r = u256_zero();
            for (unsigned int i = 0; i < 32; ++i)
            {
                unsigned int pfr = 31 - i;
                r.w[pfr / 8] |= (unsigned long long)digest[i] << ((pfr % 8) * 8);
            }
            stack[sp++] = r;
            ++pc; continue;
        }

        case 0x30: // ADDRESS
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = inp.address; ++pc; continue;
        case 0x31: { // BALANCE — EIP-2929 cold 2600 / warm 100; no Host → 0.
            if (sp < 1) ERR();
            uint256 a = stack[sp - 1];
            unsigned long long cost = warm_addr_mark_2929(warm_addrs, warm_addr_count, a)
                                       ? GAS_ACCOUNT_WARM : GAS_ACCOUNT_COLD;
            if (gas < cost) OOG(); gas -= cost;
            stack[sp - 1] = u256_zero();
            ++pc; continue;
        }
        case 0x32: // ORIGIN
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = ctx->origin; ++pc; continue;
        case 0x33: // CALLER
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = inp.caller; ++pc; continue;
        case 0x34: // CALLVALUE
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = inp.value; ++pc; continue;
        case 0x35: { // CALLDATALOAD
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 1) ERR();
            uint256 ov = stack[sp - 1];
            uint256 result = u256_zero();
            if (!ov.w[1] && !ov.w[2] && !ov.w[3] && ov.w[0] < calldata_size)
            {
                unsigned int off = (unsigned int)ov.w[0];
                for (unsigned int i = 0; i < 32; ++i)
                {
                    unsigned int src = off + i;
                    unsigned char bv = (src < calldata_size) ? calldata[src] : 0;
                    unsigned int pfr = 31 - i;
                    result.w[pfr / 8] |= (unsigned long long)bv << ((pfr % 8) * 8);
                }
            }
            stack[sp - 1] = result;
            ++pc; continue;
        }
        case 0x36: // CALLDATASIZE
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = u256_from((unsigned long long)calldata_size);
            ++pc; continue;
        case 0x37: { // CALLDATACOPY
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 3) ERR();
            uint256 dv = stack[--sp];
            uint256 sv = stack[--sp];
            uint256 lv = stack[--sp];
            if (lv.w[1] | lv.w[2] | lv.w[3]) ERR();
            unsigned long long sz = lv.w[0];
            if (!offset_in_bounds(dv, sz)) ERR();
            unsigned int dest = (unsigned int)dv.w[0];
            unsigned int size = (unsigned int)sz;
            unsigned long long words = (sz + 31) / 32;
            unsigned long long copy_gas = GAS_COPY * words;
            if (gas < copy_gas) OOG(); gas -= copy_gas;
            if (!expand_mem_range(mem, mem_size, gas, dest, size)) OOG();
            // src_off treated as past-end (zero-pad) when:
            //   - any high limb of sv non-zero,
            //   - sv.w[0] already >= calldata_size,
            //   - sv.w[0] + size overflows uint32 (per-byte add would wrap).
            // Old check `sv.w[0] >= 0xFFFFFFFFULL` only caught exactly
            // 0xFFFFFFFF; src=0xFFFFFFFE size=5 would wrap and read
            // calldata[0..2] instead of returning zeros.
            unsigned long long src_lo = sv.w[0];
            bool src_past_end = (sv.w[1] | sv.w[2] | sv.w[3]) != 0 ||
                                src_lo >= calldata_size ||
                                (src_lo + sz) > 0xFFFFFFFFULL;
            unsigned int src_off = src_past_end ? calldata_size
                                                : (unsigned int)src_lo;
            for (unsigned int i = 0; i < size; ++i)
            {
                unsigned long long s = (unsigned long long)src_off
                                     + (unsigned long long)i;
                mem[dest + i] = (s < calldata_size) ? calldata[s] : 0;
            }
            ++pc; continue;
        }
        case 0x38: // CODESIZE
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = u256_from((unsigned long long)code_size);
            ++pc; continue;
        case 0x39: { // CODECOPY
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 3) ERR();
            uint256 dv = stack[--sp];
            uint256 sv = stack[--sp];
            uint256 lv = stack[--sp];
            if (lv.w[1] | lv.w[2] | lv.w[3]) ERR();
            unsigned long long sz = lv.w[0];
            if (!offset_in_bounds(dv, sz)) ERR();
            unsigned int dest = (unsigned int)dv.w[0];
            unsigned int size = (unsigned int)sz;
            unsigned long long words = (sz + 31) / 32;
            unsigned long long copy_gas = GAS_COPY * words;
            if (gas < copy_gas) OOG(); gas -= copy_gas;
            if (!expand_mem_range(mem, mem_size, gas, dest, size)) OOG();
            // Same zero-pad rule as CALLDATACOPY: avoid uint32 wrap on
            // src+i by working in 64-bit and detecting (src+size) > 2^32.
            unsigned long long src_lo = sv.w[0];
            bool src_past_end = (sv.w[1] | sv.w[2] | sv.w[3]) != 0 ||
                                src_lo >= code_size ||
                                (src_lo + sz) > 0xFFFFFFFFULL;
            unsigned int src_off = src_past_end ? code_size
                                                : (unsigned int)src_lo;
            for (unsigned int i = 0; i < size; ++i)
            {
                unsigned long long s = (unsigned long long)src_off
                                     + (unsigned long long)i;
                mem[dest + i] = (s < code_size) ? code_dev[s] : 0;
            }
            ++pc; continue;
        }
        case 0x3a: // GASPRICE
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = u256_from(ctx->gas_price);
            ++pc; continue;
        case 0x3b: { // EXTCODESIZE — EIP-2929 cold 2600 / warm 100; no Host → 0.
            if (sp < 1) ERR();
            uint256 a = stack[sp - 1];
            unsigned long long cost = warm_addr_mark_2929(warm_addrs, warm_addr_count, a)
                                       ? GAS_ACCOUNT_WARM : GAS_ACCOUNT_COLD;
            if (gas < cost) OOG(); gas -= cost;
            stack[sp - 1] = u256_zero();
            ++pc; continue;
        }
        case 0x3c: { // EXTCODECOPY — EIP-2929 cold 2600 / warm 100 + 3*ceil(size/32).
            if (sp < 4) ERR();
            uint256 addr_v = stack[--sp];      // addr — EIP-2929 access subject
            uint256 dv = stack[--sp];
            (void)stack[--sp];                 // src offset (unused — pads 0)
            uint256 lv = stack[--sp];
            if (lv.w[1] | lv.w[2] | lv.w[3]) ERR();
            unsigned long long sz = lv.w[0];
            if (!offset_in_bounds(dv, sz)) ERR();
            unsigned int dest = (unsigned int)dv.w[0];
            unsigned int size = (unsigned int)sz;
            unsigned long long words = (sz + 31) / 32;
            unsigned long long access_cost = warm_addr_mark_2929(warm_addrs, warm_addr_count, addr_v)
                                              ? GAS_ACCOUNT_WARM : GAS_ACCOUNT_COLD;
            unsigned long long total_gas = access_cost + GAS_COPY * words;
            if (gas < total_gas) OOG(); gas -= total_gas;
            if (!expand_mem_range(mem, mem_size, gas, dest, size)) OOG();
            for (unsigned int i = 0; i < size; ++i) mem[dest + i] = 0;
            ++pc; continue;
        }
        case 0x3d: // RETURNDATASIZE
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = u256_zero();          // no prior call: empty
            ++pc; continue;
        case 0x3e: { // RETURNDATACOPY — any nonzero size is invalid (returndata empty).
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 3) ERR();
            (void)stack[--sp];                  // dest
            (void)stack[--sp];                  // src
            uint256 lv = stack[--sp];
            if (lv.w[1] | lv.w[2] | lv.w[3] || lv.w[0] != 0) ERRA();
            ++pc; continue;
        }
        case 0x3f: { // EXTCODEHASH — EIP-2929 cold 2600 / warm 100; no Host → keccak256("").
            if (sp < 1) ERR();
            uint256 a = stack[sp - 1];
            unsigned long long cost = warm_addr_mark_2929(warm_addrs, warm_addr_count, a)
                                       ? GAS_ACCOUNT_WARM : GAS_ACCOUNT_COLD;
            if (gas < cost) OOG(); gas -= cost;
            uint256 r = u256_zero();
            for (unsigned int i = 0; i < 32; ++i)
            {
                unsigned int pfr = 31 - i;
                r.w[pfr / 8] |= (unsigned long long)KECCAK_EMPTY[i] << ((pfr % 8) * 8);
            }
            stack[sp - 1] = r;
            ++pc; continue;
        }
        case 0x40: { // BLOCKHASH — return 0 (no history table in kernel).
            if (gas < GAS_BLOCKHASH) OOG(); gas -= GAS_BLOCKHASH;
            if (sp < 1) ERR();
            stack[sp - 1] = u256_zero();
            ++pc; continue;
        }
        case 0x41: // COINBASE
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = ctx->coinbase; ++pc; continue;
        case 0x42: // TIMESTAMP
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = u256_from(ctx->timestamp); ++pc; continue;
        case 0x43: // NUMBER
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = u256_from(ctx->number); ++pc; continue;
        case 0x44: // PREVRANDAO (formerly DIFFICULTY)
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = ctx->prevrandao; ++pc; continue;
        case 0x45: // GASLIMIT (block)
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = u256_from(ctx->gas_limit); ++pc; continue;
        case 0x46: // CHAINID
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = u256_from(ctx->chain_id); ++pc; continue;
        case 0x47: // SELFBALANCE — no Host: zero
            if (gas < GAS_SELFBALANCE) OOG(); gas -= GAS_SELFBALANCE;
            if (sp >= 1024) ERR();
            stack[sp++] = u256_zero(); ++pc; continue;
        case 0x48: // BASEFEE
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = u256_from(ctx->base_fee); ++pc; continue;
        case 0x49: { // BLOBHASH
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 1) ERR();
            uint256 iv = stack[sp - 1];
            uint256 r = u256_zero();
            // Cap at MAX_BLOB_HASHES even if the host wrote num_blob_hashes
            // > 8: blob_hashes is a fixed-size array. Reading past index 7
            // would be OOB into adjacent kernel state. EIP-4844 caps blobs
            // per tx well below 8 in practice; belt-and-suspenders.
            unsigned int nhashes = (ctx->num_blob_hashes > MAX_BLOB_HASHES)
                                       ? MAX_BLOB_HASHES
                                       : ctx->num_blob_hashes;
            if (!iv.w[1] && !iv.w[2] && !iv.w[3] && iv.w[0] < nhashes)
            {
                const unsigned char* h = ctx->blob_hashes[(unsigned int)iv.w[0]];
                for (unsigned int i = 0; i < 32; ++i)
                {
                    unsigned int pfr = 31 - i;
                    r.w[pfr / 8] |= (unsigned long long)h[i] << ((pfr % 8) * 8);
                }
            }
            stack[sp - 1] = r;
            ++pc; continue;
        }
        case 0x4a: // BLOBBASEFEE
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = u256_from(ctx->blob_base_fee); ++pc; continue;

        case 0x50: // POP
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp == 0) ERR();
            --sp; ++pc; continue;

        case 0x51: { // MLOAD
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 1) ERR();
            uint256 ov = stack[sp - 1];
            if (!offset_in_bounds(ov, 32)) ERR();
            unsigned int off = (unsigned int)ov.w[0];
            if (!expand_mem_range(mem, mem_size, gas, off, 32)) OOG();
            uint256 r = u256_zero();
            for (int lmb = 3; lmb >= 0; --lmb)
            {
                unsigned long long v = 0;
                int st = (3 - lmb) * 8;
                for (int bi = 0; bi < 8; ++bi)
                    v = (v << 8) | (unsigned long long)mem[off + st + bi];
                r.w[lmb] = v;
            }
            stack[sp - 1] = r;
            ++pc; continue;
        }
        case 0x52: { // MSTORE
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 2) ERR();
            uint256 ov = stack[--sp];
            uint256 val = stack[--sp];
            if (!offset_in_bounds(ov, 32)) ERR();
            unsigned int off = (unsigned int)ov.w[0];
            if (!expand_mem_range(mem, mem_size, gas, off, 32)) OOG();
            for (int lmb = 3; lmb >= 0; --lmb)
            {
                unsigned long long v = val.w[lmb];
                int st = (3 - lmb) * 8;
                for (int bi = 7; bi >= 0; --bi)
                {
                    mem[off + st + bi] = (unsigned char)(v & 0xFFULL);
                    v >>= 8;
                }
            }
            ++pc; continue;
        }
        case 0x53: { // MSTORE8
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 2) ERR();
            uint256 ov = stack[--sp];
            uint256 val = stack[--sp];
            if (!offset_in_bounds(ov, 1)) ERR();
            unsigned int off = (unsigned int)ov.w[0];
            if (!expand_mem_range(mem, mem_size, gas, off, 1)) OOG();
            mem[off] = (unsigned char)(val.w[0] & 0xFFULL);
            ++pc; continue;
        }
        case 0x54: { // SLOAD — EIP-2929: cold 2100 / warm 100, keyed on (contract, slot).
            if (sp < 1) ERR();
            uint256 slot = stack[sp - 1];
            unsigned long long cost = warm_slot_mark_2929(warm_slot_addrs, warm_slot_keys,
                                                          warm_slot_count, inp.address, slot)
                                       ? GAS_SLOAD_WARM : GAS_SLOAD_COLD;
            if (gas < cost) OOG(); gas -= cost;
            uint256 val = u256_zero();
            for (unsigned int i = stor_count; i > 0; --i)
            {
                if (u256_eq(storage[i - 1].key, slot))
                {
                    val = storage[i - 1].value;
                    break;
                }
            }
            stack[sp - 1] = val;
            ++pc; continue;
        }
        case 0x55: { // SSTORE — EIP-2200 base + EIP-2929 cold surcharge on first slot access.
            if (sp < 2) ERR();
            uint256 slot = stack[--sp];
            uint256 val  = stack[--sp];
            unsigned long long access_surcharge = warm_slot_mark_2929(warm_slot_addrs, warm_slot_keys,
                                                                       warm_slot_count, inp.address, slot)
                                                    ? 0ULL : GAS_SLOAD_COLD;
            uint256 current = u256_zero();
            bool found = false;
            for (unsigned int i = stor_count; i > 0; --i)
            {
                if (u256_eq(storage[i - 1].key, slot))
                {
                    current = storage[i - 1].value;
                    found = true;
                    break;
                }
            }
            // Cap check before any state mutation: appending a new slot when
            // the per-tx buffer is full would silently corrupt state. Signal
            // INVALID-style (status=Error, all gas consumed). With a host the
            // dispatcher routes this tx to cevm CPU (which has no cap);
            // without one the Error is honest — the GPU can't process this tx.
            if (!found && stor_count >= MAX_STORAGE_PER_TX) ERRA();
            original_value_record(orig_storage, orig_count, slot, current);
            uint256 original = u256_zero();
            original_value_lookup(orig_storage, orig_count, slot, original);
            unsigned long long sc = sstore_gas_eip2200(original, current, val, refund_counter);
            unsigned long long total_cost = sc + access_surcharge;
            if (gas < total_cost) OOG(); gas -= total_cost;
            if (found)
            {
                for (unsigned int i = stor_count; i > 0; --i)
                {
                    if (u256_eq(storage[i - 1].key, slot))
                    {
                        storage[i - 1].value = val;
                        break;
                    }
                }
            }
            else
            {
                // Capacity overflow: silently dropping a charged write
                // would diverge from CPU cevm (no cap there) and corrupt
                // consensus. Status=Error, all gas — the dispatcher routes
                // a tx that errs here to cevm CPU as the unbounded
                // fallback so legitimate large-storage txs still execute.
                if (stor_count >= MAX_STORAGE_PER_TX) ERR();
                storage[stor_count].key = slot;
                storage[stor_count].value = val;
                stor_count++;
            }
            ++pc; continue;
        }

        case 0x56: { // JUMP
            if (gas < GAS_MID) OOG(); gas -= GAS_MID;
            if (sp < 1) ERR();
            uint256 dv = stack[--sp];
            if (dv.w[1] | dv.w[2] | dv.w[3] || dv.w[0] >= code_size) ERR();
            unsigned int dest = (unsigned int)dv.w[0];
            if (!is_valid_jumpdest(code_dev, code_size, dest)) ERR();
            pc = dest; continue;
        }
        case 0x57: { // JUMPI
            if (gas < GAS_HIGH) OOG(); gas -= GAS_HIGH;
            if (sp < 2) ERR();
            uint256 dv   = stack[--sp];
            uint256 cond = stack[--sp];
            if (!u256_iszero(cond))
            {
                if (dv.w[1] | dv.w[2] | dv.w[3] || dv.w[0] >= code_size) ERR();
                unsigned int dest = (unsigned int)dv.w[0];
                if (!is_valid_jumpdest(code_dev, code_size, dest)) ERR();
                pc = dest; continue;
            }
            ++pc; continue;
        }

        case 0x58: // PC
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = u256_from((unsigned long long)pc);
            ++pc; continue;
        case 0x59: // MSIZE
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = u256_from((unsigned long long)mem_size);
            ++pc; continue;
        case 0x5a: // GAS
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = u256_from(gas);
            ++pc; continue;
        case 0x5b: // JUMPDEST
            if (gas < GAS_JUMPDEST) OOG(); gas -= GAS_JUMPDEST;
            ++pc; continue;
        case 0x5c: { // TLOAD (EIP-1153)
            if (gas < GAS_TLOAD) OOG(); gas -= GAS_TLOAD;
            if (sp < 1) ERR();
            if (!transient || !trans_count_ptr) ERR();
            uint256 slot = stack[sp - 1];
            uint256 v = u256_zero();
            unsigned int tc = *trans_count_ptr;
            for (unsigned int i = tc; i > 0; --i)
                if (u256_eq(transient[i - 1].key, slot))
                {
                    v = transient[i - 1].value;
                    break;
                }
            stack[sp - 1] = v;
            ++pc; continue;
        }
        case 0x5d: { // TSTORE (EIP-1153)
            if (gas < GAS_TSTORE) OOG(); gas -= GAS_TSTORE;
            if (sp < 2) ERR();
            if (!transient || !trans_count_ptr) ERR();
            uint256 slot = stack[--sp];
            uint256 val  = stack[--sp];
            unsigned int tc = *trans_count_ptr;
            bool found = false;
            for (unsigned int i = tc; i > 0; --i)
                if (u256_eq(transient[i - 1].key, slot))
                {
                    transient[i - 1].value = val;
                    found = true;
                    break;
                }
            if (!found && tc >= MAX_STORAGE_PER_TX) ERRA();
            if (!found)
            {
                transient[tc].key = slot;
                transient[tc].value = val;
                *trans_count_ptr = tc + 1;
            }
            ++pc; continue;
        }
        case 0x5e: { // MCOPY (EIP-5656)
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 3) ERR();
            uint256 dv = stack[--sp];
            uint256 sv = stack[--sp];
            uint256 lv = stack[--sp];
            if (lv.w[1] | lv.w[2] | lv.w[3]) ERR();
            unsigned long long sz = lv.w[0];
            if (!offset_in_bounds(dv, sz)) ERR();
            if (!offset_in_bounds(sv, sz)) ERR();
            unsigned int dest = (unsigned int)dv.w[0];
            unsigned int src  = (unsigned int)sv.w[0];
            unsigned int size = (unsigned int)sz;
            unsigned long long words = (sz + 31) / 32;
            unsigned long long copy_gas = GAS_COPY * words;
            if (gas < copy_gas) OOG(); gas -= copy_gas;
            // Touch BOTH ranges to expand memory correctly.
            unsigned int touch = (dest > src ? dest + size : src + size);
            unsigned int near  = (dest > src ? src : dest);
            if (!expand_mem_range(mem, mem_size, gas, near, touch - near)) OOG();
            // Handle overlap with memmove semantics.
            if (size > 0)
            {
                if (dest < src)
                    for (unsigned int i = 0; i < size; ++i)
                        mem[dest + i] = mem[src + i];
                else
                    for (unsigned int i = size; i > 0; --i)
                        mem[dest + i - 1] = mem[src + i - 1];
            }
            ++pc; continue;
        }
        case 0x5f: // PUSH0
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = u256_zero();
            ++pc; continue;

        // PUSH1 with opcode fusion (PUSH1 + ADD).
        case 0x60: {
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp >= 1024) ERR();
            unsigned long long push_val =
                (unsigned long long)((pc + 1 < code_size) ? CODE_BYTE(pc + 1) : 0);

            if (pc + 2 < code_size)
            {
                unsigned char next_op = CODE_BYTE(pc + 2);
                if (next_op == 0x01 && sp >= 1 && gas >= GAS_VERYLOW)
                {
                    gas -= GAS_VERYLOW;
                    stack[sp - 1] = u256_add(stack[sp - 1], u256_from(push_val));
                    pc += 3; continue;
                }
            }

            stack[sp++] = u256_from(push_val);
            pc += 2; continue;
        }

        // DUP1..DUP16
        case 0x80: case 0x81: case 0x82: case 0x83:
        case 0x84: case 0x85: case 0x86: case 0x87:
        case 0x88: case 0x89: case 0x8a: case 0x8b:
        case 0x8c: case 0x8d: case 0x8e: case 0x8f: {
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            unsigned int n = op - 0x80 + 1;
            if (n > sp || sp >= 1024) ERR();
            stack[sp] = stack[sp - n];
            ++sp; ++pc; continue;
        }

        // SWAP1..SWAP16
        case 0x90: case 0x91: case 0x92: case 0x93:
        case 0x94: case 0x95: case 0x96: case 0x97:
        case 0x98: case 0x99: case 0x9a: case 0x9b:
        case 0x9c: case 0x9d: case 0x9e: case 0x9f: {
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            unsigned int n = op - 0x90 + 1;
            if (n >= sp) ERR();
            unsigned int idx = sp - 1 - n;
            uint256 tmp = stack[sp - 1];
            stack[sp - 1] = stack[idx];
            stack[idx] = tmp;
            ++pc; continue;
        }

        // LOG0..LOG4
        case 0xa0: case 0xa1: case 0xa2: case 0xa3: case 0xa4: {
            unsigned int num_topics = op - 0xa0;
            if (sp < 2u + num_topics) ERR();
            uint256 ov = stack[--sp];
            uint256 sv = stack[--sp];
            if (sv.w[1] | sv.w[2] | sv.w[3]) ERR();
            unsigned long long size = sv.w[0];
            if (!offset_in_bounds(ov, size)) ERR();
            unsigned int off = (unsigned int)ov.w[0];
            unsigned int sz  = (unsigned int)size;
            unsigned long long log_gas =
                GAS_LOG_BASE + GAS_LOG_TOPIC * num_topics + GAS_LOG_DATA * size;
            if (gas < log_gas) OOG(); gas -= log_gas;
            if (!expand_mem_range(mem, mem_size, gas, off, sz)) OOG();
            // Topics come off the stack in order topic1, topic2, ... topic_n.
            uint256 topics[4];
            for (unsigned int t = 0; t < num_topics; ++t) topics[t] = stack[--sp];
            if (logs && log_count_ptr)
            {
                unsigned int lc = *log_count_ptr;
                if (lc >= MAX_LOGS_PER_TX) ERR();
                LogEntry& e = logs[lc];
                e.num_topics = num_topics;
                e.data_offset = off;
                e.data_size = sz;
                for (unsigned int t = 0; t < num_topics; ++t) e.topics[t] = topics[t];
                *log_count_ptr = lc + 1;
            }
            ++pc; continue;
        }

        case 0xf3: { // RETURN
            if (sp < 2) ERR();
            uint256 ov = stack[--sp];
            uint256 sv = stack[--sp];
            if (sv.w[1] | sv.w[2] | sv.w[3]) ERR();
            unsigned long long size = sv.w[0];
            if (!offset_in_bounds(ov, size)) ERR();
            // Output buffer is fixed at MAX_OUTPUT_PER_TX. Silently
            // truncating would diverge from CPU cevm (no cap there);
            // the dispatcher routes Error txs to cevm CPU as the
            // unbounded fallback so legitimate large RETURN payloads
            // still execute correctly.
            if (size > MAX_OUTPUT_PER_TX) ERR();
            unsigned int off = (unsigned int)ov.w[0];
            unsigned int sz  = (unsigned int)size;
            if (!expand_mem_range(mem, mem_size, gas, off, sz)) OOG();
            for (unsigned int i = 0; i < sz; ++i) output[i] = mem[off + i];
            EMIT(1, gas_start - gas, refund_counter, sz);
        }
        case 0xfd: { // REVERT
            if (sp < 2) ERR();
            uint256 ov = stack[--sp];
            uint256 sv = stack[--sp];
            if (sv.w[1] | sv.w[2] | sv.w[3]) ERR();
            unsigned long long size = sv.w[0];
            if (!offset_in_bounds(ov, size)) ERR();
            // Same cap as RETURN: fail-loud on output > MAX_OUTPUT_PER_TX
            // so the dispatcher falls back to cevm (no cap) instead of
            // silently truncating REVERT data.
            if (size > MAX_OUTPUT_PER_TX) ERR();
            unsigned int off = (unsigned int)ov.w[0];
            unsigned int sz  = (unsigned int)size;
            if (!expand_mem_range(mem, mem_size, gas, off, sz)) OOG();
            for (unsigned int i = 0; i < sz; ++i) output[i] = mem[off + i];
            EMIT(2, gas_start - gas, refund_counter, sz);
        }
        case 0xfe: // INVALID — consume all gas
            EMIT(4, gas_start, refund_counter, 0);

        // CREATE / CALL / CALLCODE / DELEGATECALL / CREATE2 / STATICCALL / SELFDESTRUCT
        case 0xf0: case 0xf1: case 0xf2: case 0xf4:
        case 0xf5: case 0xfa: case 0xff:
            EMIT(5, gas_start - gas, refund_counter, 0);

        default: break;
        } // end switch

        // PUSH2..PUSH32 (0x61..0x7f) — outside switch like Metal.
        if (op >= 0x61 && op <= 0x7f)
        {
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp >= 1024) ERR();
            unsigned int n = op - 0x60 + 1;
            uint256 val = u256_zero();
            unsigned int start = pc + 1;
            for (unsigned int i = 0; i < n && (start + i) < code_size; ++i)
            {
                unsigned int bp = n - 1 - i;
                val.w[bp / 8] |= (unsigned long long)CODE_BYTE(start + i)
                                 << ((bp % 8) * 8);
            }
            stack[sp++] = val;
            pc += 1 + n;
            continue;
        }

        // FIXED (Bug 2): unrecognized opcodes now consume ALL gas, matching
        // the explicit 0xFE INVALID. Before, `ERR()` reported the partial
        // consumed-so-far amount which is consensus-incorrect.
        ERRA();
    }

    EMIT(0, gas_start - gas, refund_counter, 0);

    #undef EMIT
    #undef OOG
    #undef ERR
    #undef ERRA
    #undef CODE_BYTE
}

// =============================================================================
// Host-callable launcher. Marked extern "C" to avoid C++ name mangling so the
// host .cpp file can declare and call it without nvcc's <<<...>>> syntax.
// =============================================================================

extern "C" cudaError_t evm_cuda_evm_execute_launch(
    const void*  d_inputs,            // TxInput*
    const void*  d_blob,              // unsigned char*
    void*        d_outputs,           // TxOutput*
    void*        d_out_data,          // unsigned char*
    void*        d_mem_pool,          // unsigned char*
    void*        d_storage_pool,      // StorageEntry*
    void*        d_storage_counts,    // unsigned int*
    void*        d_transient_pool,    // StorageEntry* (may be null)
    void*        d_transient_counts,  // unsigned int* (may be null)
    void*        d_log_pool,          // LogEntry*    (may be null)
    void*        d_log_counts,        // unsigned int* (may be null)
    const void*  d_block_ctx,         // const BlockContext*
    const void*  d_params,            // const unsigned int*
    unsigned int num_txs,
    cudaStream_t stream)
{
    if (num_txs == 0) return cudaSuccess;

    constexpr unsigned int threads_per_block = 64;
    const unsigned int blocks =
        (num_txs + threads_per_block - 1) / threads_per_block;

    evm_execute_kernel<<<blocks, threads_per_block, 0, stream>>>(
        static_cast<const TxInput*>(d_inputs),
        static_cast<const unsigned char*>(d_blob),
        static_cast<TxOutput*>(d_outputs),
        static_cast<unsigned char*>(d_out_data),
        static_cast<unsigned char*>(d_mem_pool),
        static_cast<StorageEntry*>(d_storage_pool),
        static_cast<unsigned int*>(d_storage_counts),
        static_cast<StorageEntry*>(d_transient_pool),
        static_cast<unsigned int*>(d_transient_counts),
        static_cast<LogEntry*>(d_log_pool),
        static_cast<unsigned int*>(d_log_counts),
        static_cast<const BlockContext*>(d_block_ctx),
        static_cast<const unsigned int*>(d_params));

    return cudaGetLastError();
}

}  // namespace evm::gpu::cuda
