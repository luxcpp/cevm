// Copyright (C) 2026, The evmone Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file evm_kernel.cu
/// CUDA port of kernel/evm_kernel.metal — single-thread-per-tx EVM interpreter.
///
/// Mirrors the Metal V1 kernel byte-for-byte:
///   - Same uint256 layout (4x uint64 limbs, w[0] = low)
///   - Same TxInput / TxOutput / StorageEntry struct sizes (matched on the host
///     via static_assert)
///   - Same gas costs and dispatch order
///   - Same status codes: 0=stop, 1=return, 2=revert, 3=oog, 4=error,
///     5=call_not_supported
///
/// The Metal V1 source uses a 32-entry per-thread stack (line 343 of
/// evm_kernel.metal). This port keeps that to preserve register pressure
/// behaviour and gas/error parity. Programs that need >32 stack depth get
/// status=Error and the dispatcher falls back to CPU evmone.

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
            unsigned long long s = r8[i + j] + p.lo;
            unsigned long long c = (s < r8[i + j]) ? 1ULL : 0ULL;
            s += carry;
            c += (s < carry) ? 1ULL : 0ULL;
            r8[i + j] = s;
            carry = p.hi + c;
        }
        if (i + 4 < 8) r8[i + 4] += carry;
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
};

struct TxOutput
{
    unsigned int        status;
    unsigned long long  gas_used;
    unsigned long long  gas_refund;
    unsigned int        output_size;
};

struct StorageEntry
{
    uint256 key;
    uint256 value;
};

// Validate that the device-side wire layout matches the host expectations.
static_assert(sizeof(uint256)      == 32,  "device uint256 size");
static_assert(sizeof(TxInput)      == 120, "device TxInput size");
static_assert(sizeof(TxOutput)     == 32,  "device TxOutput size");
static_assert(sizeof(StorageEntry) == 64,  "device StorageEntry size");

// =============================================================================
// Constants — must match Metal evm_kernel.metal.
// =============================================================================

__device__ static constexpr unsigned int MAX_MEMORY_PER_TX  = 65536;
__device__ static constexpr unsigned int MAX_OUTPUT_PER_TX  = 1024;
__device__ static constexpr unsigned int MAX_STORAGE_PER_TX = 64;

__device__ static constexpr unsigned long long GAS_VERYLOW    = 3;
__device__ static constexpr unsigned long long GAS_LOW        = 5;
__device__ static constexpr unsigned long long GAS_MID        = 8;
__device__ static constexpr unsigned long long GAS_HIGH       = 10;
__device__ static constexpr unsigned long long GAS_BASE       = 2;
__device__ static constexpr unsigned long long GAS_JUMPDEST   = 1;
__device__ static constexpr unsigned long long GAS_SLOAD      = 2100;
__device__ static constexpr unsigned long long GAS_SSTORE_SET   = 20000;
__device__ static constexpr unsigned long long GAS_SSTORE_RESET = 2900;
__device__ static constexpr unsigned long long GAS_SSTORE_NOOP   = 100;
__device__ static constexpr unsigned long long GAS_SSTORE_REFUND = 4800;
__device__ static constexpr unsigned long long GAS_MEMORY     = 3;
__device__ static constexpr unsigned long long GAS_EXP_BASE   = 10;
__device__ static constexpr unsigned long long GAS_EXP_BYTE   = 50;

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

__device__ unsigned long long sstore_gas_eip2200(const uint256& orig,
                                                 const uint256& cur,
                                                 const uint256& nv,
                                                 unsigned long long& rc)
{
    if (u256_eq(nv, cur)) return GAS_SSTORE_NOOP;
    if (u256_eq(orig, cur))
    {
        if (u256_iszero(orig))      return GAS_SSTORE_SET;
        if (u256_iszero(nv))        rc += GAS_SSTORE_REFUND;
        return GAS_SSTORE_RESET;
    }
    if (!u256_iszero(orig))
    {
        if (u256_iszero(cur))       rc -= GAS_SSTORE_REFUND;
        else if (u256_iszero(nv))   rc += GAS_SSTORE_REFUND;
    }
    if (u256_eq(nv, orig))
    {
        if (u256_iszero(orig))      rc += GAS_SSTORE_SET   - GAS_SSTORE_NOOP;
        else                        rc += GAS_SSTORE_RESET - GAS_SSTORE_NOOP;
    }
    return GAS_SSTORE_NOOP;
}

// =============================================================================
// V1 kernel: optimized single-thread-per-tx interpreter.
// One thread per transaction. Mirrors Metal's switch dispatch verbatim.
// =============================================================================

__global__ void evm_execute_kernel(
    const TxInput*       __restrict__ inputs,
    const unsigned char* __restrict__ blob,
    TxOutput*            __restrict__ outputs,
    unsigned char*       __restrict__ out_data,
    unsigned char*       __restrict__ mem_pool,
    StorageEntry*        __restrict__ storage_pool,
    unsigned int*        __restrict__ storage_counts,
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

    const unsigned char* code_dev = blob + inp.code_offset;
    const unsigned int   code_size = inp.code_size;
    const unsigned char* calldata = blob + inp.calldata_offset;
    const unsigned int   calldata_size = inp.calldata_size;

    // Cache up to first 256 bytes of bytecode in registers/local for quick fetch.
    unsigned char code_cache[256];
    const unsigned int cached_size = (code_size < 256) ? code_size : 256u;
    for (unsigned int i = 0; i < cached_size; ++i) code_cache[i] = code_dev[i];

    #define CODE_BYTE(idx) ((idx) < cached_size ? code_cache[(idx)] : code_dev[(idx)])

    // 32-entry stack — same as Metal V1 (line 343 of evm_kernel.metal).
    // Programs needing >32 stack entries return Error; the host falls back to CPU.
    uint256 stack[32];
    unsigned int       sp  = 0;
    unsigned long long gas = inp.gas_limit;
    unsigned long long refund_counter = 0;
    unsigned int       pc = 0;
    unsigned int       mem_size = 0;
    const unsigned long long gas_start = gas;

    OriginalEntry orig_storage[MAX_STORAGE_PER_TX];
    unsigned int  orig_count = 0;
    for (unsigned int i = 0; i < MAX_STORAGE_PER_TX; ++i) orig_storage[i].valid = false;

    #define EMIT(st, gu, gr, os) do { \
        out.status     = (st); \
        out.gas_used   = (gu); \
        out.gas_refund = (gr); \
        out.output_size = (os); \
        return; \
    } while (0)
    #define OOG() EMIT(3, gas_start, refund_counter, 0)
    #define ERR() EMIT(4, gas_start - gas, refund_counter, 0)

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

        case 0x30: // ADDRESS
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 32) ERR();
            stack[sp++] = inp.address; ++pc; continue;
        case 0x33: // CALLER
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 32) ERR();
            stack[sp++] = inp.caller; ++pc; continue;
        case 0x34: // CALLVALUE
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 32) ERR();
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
            if (sp >= 32) ERR();
            stack[sp++] = u256_from((unsigned long long)calldata_size);
            ++pc; continue;

        case 0x50: // POP
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp == 0) ERR();
            --sp; ++pc; continue;

        case 0x51: { // MLOAD
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 1) ERR();
            uint256 ov = stack[sp - 1];
            if (ov.w[1] | ov.w[2] | ov.w[3] || ov.w[0] + 32 > MAX_MEMORY_PER_TX) ERR();
            unsigned int off = (unsigned int)ov.w[0];
            unsigned int end = off + 32;
            unsigned int nw  = (end + 31) / 32;
            if (nw * 32 > mem_size)
            {
                unsigned int ow = mem_size / 32;
                unsigned long long cost =
                    GAS_MEMORY * nw + ((unsigned long long)nw * nw) / 512
                    - GAS_MEMORY * ow - ((unsigned long long)ow * ow) / 512;
                if (gas < cost) OOG(); gas -= cost;
                for (unsigned int i = mem_size; i < nw * 32; ++i) mem[i] = 0;
                mem_size = nw * 32;
            }
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
            if (ov.w[1] | ov.w[2] | ov.w[3] || ov.w[0] + 32 > MAX_MEMORY_PER_TX) ERR();
            unsigned int off = (unsigned int)ov.w[0];
            unsigned int end = off + 32;
            unsigned int nw  = (end + 31) / 32;
            if (nw * 32 > mem_size)
            {
                unsigned int ow = mem_size / 32;
                unsigned long long cost =
                    GAS_MEMORY * nw + ((unsigned long long)nw * nw) / 512
                    - GAS_MEMORY * ow - ((unsigned long long)ow * ow) / 512;
                if (gas < cost) OOG(); gas -= cost;
                for (unsigned int i = mem_size; i < nw * 32; ++i) mem[i] = 0;
                mem_size = nw * 32;
            }
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
        case 0x54: { // SLOAD
            if (gas < GAS_SLOAD) OOG(); gas -= GAS_SLOAD;
            if (sp < 1) ERR();
            uint256 slot = stack[sp - 1];
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
        case 0x55: { // SSTORE
            if (sp < 2) ERR();
            uint256 slot = stack[--sp];
            uint256 val  = stack[--sp];
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
            original_value_record(orig_storage, orig_count, slot, current);
            uint256 original = u256_zero();
            original_value_lookup(orig_storage, orig_count, slot, original);
            unsigned long long sc = sstore_gas_eip2200(original, current, val, refund_counter);
            if (gas < sc) OOG(); gas -= sc;
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
            else if (stor_count < MAX_STORAGE_PER_TX)
            {
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
            if (sp >= 32) ERR();
            stack[sp++] = u256_from((unsigned long long)pc);
            ++pc; continue;
        case 0x59: // MSIZE
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 32) ERR();
            stack[sp++] = u256_from((unsigned long long)mem_size);
            ++pc; continue;
        case 0x5a: // GAS
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 32) ERR();
            stack[sp++] = u256_from(gas);
            ++pc; continue;
        case 0x5b: // JUMPDEST
            if (gas < GAS_JUMPDEST) OOG(); gas -= GAS_JUMPDEST;
            ++pc; continue;
        case 0x5f: // PUSH0
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 32) ERR();
            stack[sp++] = u256_zero();
            ++pc; continue;

        // PUSH1 with opcode fusion (PUSH1 + ADD).
        case 0x60: {
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp >= 32) ERR();
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
            if (n > sp || sp >= 32) ERR();
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

        case 0xf3: { // RETURN
            if (sp < 2) ERR();
            uint256 ov = stack[--sp];
            uint256 sv = stack[--sp];
            unsigned int off = (unsigned int)ov.w[0];
            unsigned int sz  = (unsigned int)sv.w[0];
            if (sz > 0)
            {
                unsigned int ne = off + sz;
                if (ne < off || ne > MAX_MEMORY_PER_TX) OOG();
                if (ne > mem_size)
                {
                    unsigned int nw = (ne + 31) / 32;
                    unsigned int ow = (mem_size + 31) / 32;
                    unsigned long long mc =
                        GAS_MEMORY * (nw - ow)
                        + ((unsigned long long)nw * nw / 512)
                        - ((unsigned long long)ow * ow / 512);
                    if (gas < mc) OOG(); gas -= mc;
                    for (unsigned int i = mem_size; i < nw * 32; ++i) mem[i] = 0;
                    mem_size = nw * 32;
                }
            }
            unsigned int csz = (sz > MAX_OUTPUT_PER_TX) ? MAX_OUTPUT_PER_TX : sz;
            for (unsigned int i = 0; i < csz; ++i) output[i] = mem[off + i];
            EMIT(1, gas_start - gas, refund_counter, csz);
        }
        case 0xfd: { // REVERT
            if (sp < 2) ERR();
            uint256 ov = stack[--sp];
            uint256 sv = stack[--sp];
            unsigned int off = (unsigned int)ov.w[0];
            unsigned int sz  = (unsigned int)sv.w[0];
            if (sz > 0)
            {
                unsigned int ne = off + sz;
                if (ne < off || ne > MAX_MEMORY_PER_TX) OOG();
                if (ne > mem_size)
                {
                    unsigned int nw = (ne + 31) / 32;
                    unsigned int ow = (mem_size + 31) / 32;
                    unsigned long long mc =
                        GAS_MEMORY * (nw - ow)
                        + ((unsigned long long)nw * nw / 512)
                        - ((unsigned long long)ow * ow / 512);
                    if (gas < mc) OOG(); gas -= mc;
                    for (unsigned int i = mem_size; i < nw * 32; ++i) mem[i] = 0;
                    mem_size = nw * 32;
                }
            }
            unsigned int csz = (sz > MAX_OUTPUT_PER_TX) ? MAX_OUTPUT_PER_TX : sz;
            for (unsigned int i = 0; i < csz; ++i) output[i] = mem[off + i];
            EMIT(2, gas_start - gas, refund_counter, csz);
        }
        case 0xfe: // INVALID
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
            if (sp >= 32) ERR();
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

        // Unrecognized opcode.
        ERR();
    }

    EMIT(0, gas_start - gas, refund_counter, 0);

    #undef EMIT
    #undef OOG
    #undef ERR
    #undef CODE_BYTE
}

// =============================================================================
// Host-callable launcher. Marked extern "C" to avoid C++ name mangling so the
// host .cpp file can declare and call it without nvcc's <<<...>>> syntax.
// =============================================================================

extern "C" cudaError_t evm_cuda_evm_execute_launch(
    const void*  d_inputs,         // TxInput*
    const void*  d_blob,           // unsigned char*
    void*        d_outputs,        // TxOutput*
    void*        d_out_data,       // unsigned char*
    void*        d_mem_pool,       // unsigned char*
    void*        d_storage_pool,   // StorageEntry*
    void*        d_storage_counts, // unsigned int*
    const void*  d_params,         // const unsigned int*
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
        static_cast<const unsigned int*>(d_params));

    return cudaGetLastError();
}

}  // namespace evm::gpu::cuda
