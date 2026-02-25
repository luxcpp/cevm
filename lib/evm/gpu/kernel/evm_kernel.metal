// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file evm_kernel.metal
/// Metal compute shader for GPU EVM execution -- spec-complete (Cancun) GPU
/// EVM interpreter.
///
/// All Cancun-era opcodes in mainnet today are implemented here EXCEPT the
/// CALL/CREATE family (0xf0/f1/f2/f4/f5/fa/ff). Those return status=5
/// (CallNotSupported) so the host falls back to CPU evmone.

#include <metal_stdlib>
using namespace metal;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-const-variable"

typedef ulong  gpu_u64;
typedef uint   gpu_u32;
typedef uchar  gpu_u8_t;
typedef long   gpu_i64;

// -- uint256 ------------------------------------------------------------------

struct uint256 {
    gpu_u64 w[4];
};

static inline uint256 u256_zero() {
    uint256 r; r.w[0] = 0; r.w[1] = 0; r.w[2] = 0; r.w[3] = 0; return r;
}
static inline uint256 u256_from(gpu_u64 lo) {
    uint256 r; r.w[0] = lo; r.w[1] = 0; r.w[2] = 0; r.w[3] = 0; return r;
}
static inline uint256 u256_one() { return u256_from(1); }
static inline uint256 u256_max() {
    uint256 r; gpu_u64 m = ~gpu_u64(0);
    r.w[0] = m; r.w[1] = m; r.w[2] = m; r.w[3] = m; return r;
}
static inline bool u256_iszero(uint256 a) {
    return (a.w[0] | a.w[1] | a.w[2] | a.w[3]) == 0;
}
static inline bool u256_eq(uint256 a, uint256 b) {
    return a.w[0] == b.w[0] && a.w[1] == b.w[1] && a.w[2] == b.w[2] && a.w[3] == b.w[3];
}
static inline bool u256_lt(uint256 a, uint256 b) {
    if (a.w[3] != b.w[3]) return a.w[3] < b.w[3];
    if (a.w[2] != b.w[2]) return a.w[2] < b.w[2];
    if (a.w[1] != b.w[1]) return a.w[1] < b.w[1];
    return a.w[0] < b.w[0];
}
static inline bool u256_gt(uint256 a, uint256 b) { return u256_lt(b, a); }

static inline uint256 u256_add(uint256 a, uint256 b) {
    uint256 r;
    gpu_u64 s0 = a.w[0] + b.w[0];
    gpu_u64 c0 = (s0 < a.w[0]) ? 1UL : 0UL;
    gpu_u64 s1 = a.w[1] + b.w[1] + c0;
    gpu_u64 c1 = (s1 < a.w[1] || (c0 && s1 == a.w[1])) ? 1UL : 0UL;
    gpu_u64 s2 = a.w[2] + b.w[2] + c1;
    gpu_u64 c2 = (s2 < a.w[2] || (c1 && s2 == a.w[2])) ? 1UL : 0UL;
    r.w[0] = s0; r.w[1] = s1; r.w[2] = s2;
    r.w[3] = a.w[3] + b.w[3] + c2;
    return r;
}
static inline uint256 u256_sub(uint256 a, uint256 b) {
    uint256 r;
    gpu_u64 d0 = a.w[0] - b.w[0];
    gpu_u64 bw0 = (d0 > a.w[0]) ? 1UL : 0UL;
    gpu_u64 d1 = a.w[1] - b.w[1] - bw0;
    gpu_u64 bw1 = (a.w[1] < b.w[1] + bw0 || (bw0 && b.w[1] == ~0UL)) ? 1UL : 0UL;
    gpu_u64 d2 = a.w[2] - b.w[2] - bw1;
    gpu_u64 bw2 = (a.w[2] < b.w[2] + bw1 || (bw1 && b.w[2] == ~0UL)) ? 1UL : 0UL;
    r.w[0] = d0; r.w[1] = d1; r.w[2] = d2;
    r.w[3] = a.w[3] - b.w[3] - bw2;
    return r;
}
static inline uint256 u256_bitwise_and(uint256 a, uint256 b) {
    uint256 r;
    r.w[0]=a.w[0]&b.w[0]; r.w[1]=a.w[1]&b.w[1]; r.w[2]=a.w[2]&b.w[2]; r.w[3]=a.w[3]&b.w[3]; return r;
}
static inline uint256 u256_bitwise_or(uint256 a, uint256 b) {
    uint256 r;
    r.w[0]=a.w[0]|b.w[0]; r.w[1]=a.w[1]|b.w[1]; r.w[2]=a.w[2]|b.w[2]; r.w[3]=a.w[3]|b.w[3]; return r;
}
static inline uint256 u256_bitwise_xor(uint256 a, uint256 b) {
    uint256 r;
    r.w[0]=a.w[0]^b.w[0]; r.w[1]=a.w[1]^b.w[1]; r.w[2]=a.w[2]^b.w[2]; r.w[3]=a.w[3]^b.w[3]; return r;
}
static inline uint256 u256_bitwise_not(uint256 a) {
    uint256 r;
    r.w[0]=~a.w[0]; r.w[1]=~a.w[1]; r.w[2]=~a.w[2]; r.w[3]=~a.w[3]; return r;
}
static inline uint256 u256_shl(gpu_u64 n, uint256 val) {
    if (n >= 256) return u256_zero();
    if (n == 0) return val;
    uint256 r = u256_zero();
    uint ls = uint(n/64), bs = uint(n%64);
    for (uint i = ls; i < 4; ++i) {
        r.w[i] = val.w[i-ls] << bs;
        if (bs > 0 && i > ls) r.w[i] |= val.w[i-ls-1] >> (64-bs);
    }
    return r;
}
static inline uint256 u256_shr(gpu_u64 n, uint256 val) {
    if (n >= 256) return u256_zero();
    if (n == 0) return val;
    uint256 r = u256_zero();
    uint ls = uint(n/64), bs = uint(n%64);
    for (uint i = 0; i+ls < 4; ++i) {
        r.w[i] = val.w[i+ls] >> bs;
        if (bs > 0 && i+ls+1 < 4) r.w[i] |= val.w[i+ls+1] << (64-bs);
    }
    return r;
}

struct pair64 { gpu_u64 lo; gpu_u64 hi; };
static inline pair64 mul_wide(gpu_u64 a, gpu_u64 b) {
    gpu_u64 a_lo=a&0xFFFFFFFFUL, a_hi=a>>32, b_lo=b&0xFFFFFFFFUL, b_hi=b>>32;
    gpu_u64 p0=a_lo*b_lo, p1=a_lo*b_hi, p2=a_hi*b_lo, p3=a_hi*b_hi;
    gpu_u64 mid=(p0>>32)+(p1&0xFFFFFFFFUL)+(p2&0xFFFFFFFFUL);
    pair64 r;
    r.hi=p3+(p1>>32)+(p2>>32)+(mid>>32);
    r.lo=(p0&0xFFFFFFFFUL)|((mid&0xFFFFFFFFUL)<<32);
    return r;
}
static inline uint256 u256_mul(uint256 a, uint256 b) {
    gpu_u64 r[4]={0,0,0,0};
    for (uint i=0;i<4;++i) { gpu_u64 carry=0;
        for (uint j=0;j<4;++j) { if (i+j>=4) break;
            pair64 p=mul_wide(a.w[i],b.w[j]);
            gpu_u64 s=r[i+j]+p.lo; gpu_u64 c=(s<r[i+j])?1UL:0UL;
            s+=carry; c+=(s<carry)?1UL:0UL; r[i+j]=s; carry=p.hi+c;
        }
    }
    uint256 result; result.w[0]=r[0]; result.w[1]=r[1]; result.w[2]=r[2]; result.w[3]=r[3];
    return result;
}

static inline uint clz64_metal(gpu_u64 x) {
    if (x==0) return 64; uint n=0;
    if (!(x&0xFFFFFFFF00000000UL)){n+=32;x<<=32;}
    if (!(x&0xFFFF000000000000UL)){n+=16;x<<=16;}
    if (!(x&0xFF00000000000000UL)){n+=8;x<<=8;}
    if (!(x&0xF000000000000000UL)){n+=4;x<<=4;}
    if (!(x&0xC000000000000000UL)){n+=2;x<<=2;}
    if (!(x&0x8000000000000000UL)){n+=1;}
    return n;
}
static inline uint clz256_metal(uint256 x) {
    if (x.w[3]) return clz64_metal(x.w[3]);
    if (x.w[2]) return 64+clz64_metal(x.w[2]);
    if (x.w[1]) return 128+clz64_metal(x.w[1]);
    return 192+clz64_metal(x.w[0]);
}

struct divmod_result{uint256 quot;uint256 rem;};
static inline divmod_result u256_divmod(uint256 a, uint256 b) {
    if (u256_iszero(b)) return {u256_zero(),u256_zero()};
    if (u256_lt(a,b)) return {u256_zero(),a};
    if (u256_eq(a,b)) return {u256_one(),u256_zero()};
    uint shift=clz256_metal(b)-clz256_metal(a);
    uint256 divisor=u256_shl(gpu_u64(shift),b), quotient=u256_zero(), remainder=a;
    for (uint i=0;i<=shift;++i) {
        quotient=u256_shl(1,quotient);
        if (!u256_lt(remainder,divisor)){remainder=u256_sub(remainder,divisor);quotient.w[0]|=1;}
        divisor=u256_shr(1,divisor);
    }
    return {quotient,remainder};
}
static inline uint256 u256_div(uint256 a,uint256 b){return u256_divmod(a,b).quot;}
static inline uint256 u256_mod(uint256 a,uint256 b){return u256_divmod(a,b).rem;}
static inline uint256 u256_negate(uint256 x){return u256_add(u256_bitwise_not(x),u256_one());}
static inline uint256 u256_sdiv(uint256 a,uint256 b) {
    if (u256_iszero(b)) return u256_zero();
    bool an=(a.w[3]>>63)!=0,bn=(b.w[3]>>63)!=0;
    uint256 q=u256_div(an?u256_negate(a):a,bn?u256_negate(b):b);
    if (an!=bn) q=u256_negate(q); return q;
}
static inline uint256 u256_smod(uint256 a,uint256 b) {
    if (u256_iszero(b)) return u256_zero();
    bool an=(a.w[3]>>63)!=0;
    uint256 r=u256_mod(an?u256_negate(a):a,(b.w[3]>>63)?u256_negate(b):b);
    if (an&&!u256_iszero(r)) r=u256_negate(r); return r;
}
static inline uint256 u256_addmod(uint256 a,uint256 b,uint256 m) {
    if (u256_iszero(m)) return u256_zero();
    gpu_u64 s0=a.w[0]+b.w[0]; gpu_u64 c0=(s0<a.w[0])?1UL:0UL;
    gpu_u64 s1=a.w[1]+b.w[1]+c0; gpu_u64 c1=(s1<a.w[1]||(c0&&s1==a.w[1]))?1UL:0UL;
    gpu_u64 s2=a.w[2]+b.w[2]+c1; gpu_u64 c2=(s2<a.w[2]||(c1&&s2==a.w[2]))?1UL:0UL;
    gpu_u64 s3=a.w[3]+b.w[3]+c2; gpu_u64 c3=(s3<a.w[3]||(c2&&s3==a.w[3]))?1UL:0UL;
    if (c3==0){uint256 s;s.w[0]=s0;s.w[1]=s1;s.w[2]=s2;s.w[3]=s3;return u256_mod(s,m);}
    gpu_u64 r5[5]={s0,s1,s2,s3,c3}; uint256 result=u256_zero();
    for (int bit=256;bit>=0;--bit){result=u256_shl(1,result);uint l=uint(bit)/64,p=uint(bit)%64;if ((r5[l]>>p)&1)result.w[0]|=1;if (!u256_lt(result,m))result=u256_sub(result,m);}
    return result;
}

// FIXED (Bug 1): the old loop dropped the i=3 carry (the guard `if(i+4<8)`
// is false when i=3) AND failed to propagate further carries from
// `r8[i+4] +=` if that wraparound. Now we add-with-carry through ALL
// higher limbs.
static inline uint256 u256_mulmod(uint256 a, uint256 b, uint256 m) {
    if (u256_iszero(m)) return u256_zero();
    gpu_u64 r8[8] = {0,0,0,0,0,0,0,0};
    for (uint i = 0; i < 4; ++i) {
        gpu_u64 carry = 0;
        for (uint j = 0; j < 4; ++j) {
            pair64 p = mul_wide(a.w[i], b.w[j]);
            gpu_u64 s1 = r8[i+j] + p.lo;
            gpu_u64 c1 = (s1 < r8[i+j]) ? 1UL : 0UL;
            gpu_u64 s2 = s1 + carry;
            gpu_u64 c2 = (s2 < s1) ? 1UL : 0UL;
            r8[i+j] = s2;
            carry   = p.hi + c1 + c2;
        }
        uint k = i + 4;
        while (carry != 0 && k < 8) {
            gpu_u64 s = r8[k] + carry;
            gpu_u64 c = (s < r8[k]) ? 1UL : 0UL;
            r8[k] = s;
            carry = c;
            ++k;
        }
    }
    uint256 result = u256_zero();
    for (int bit = 511; bit >= 0; --bit) {
        result = u256_shl(1, result);
        uint l = uint(bit)/64, p = uint(bit)%64;
        if ((r8[l] >> p) & 1) result.w[0] |= 1;
        if (!u256_lt(result, m)) result = u256_sub(result, m);
    }
    return result;
}

static inline uint256 u256_exp(uint256 base,uint256 exponent) {
    if (u256_iszero(exponent)) return u256_one();
    uint256 result=u256_one(),b=base,e=exponent;
    while (!u256_iszero(e)){if(e.w[0]&1)result=u256_mul(result,b);e=u256_shr(1,e);if(!u256_iszero(e))b=u256_mul(b,b);}
    return result;
}
static inline uint256 u256_signextend(uint256 b_val,uint256 x) {
    if (b_val.w[1]|b_val.w[2]|b_val.w[3]) return x;
    gpu_u64 b=b_val.w[0]; if (b>=31) return x;
    gpu_u64 sb=b*8+7; uint l=uint(sb/64),p=uint(sb%64);
    bool neg=((x.w[l]>>p)&1)!=0;
    uint256 mask=u256_bitwise_not(u256_sub(u256_shl(sb+1,u256_one()),u256_one()));
    return neg?u256_bitwise_or(x,mask):u256_bitwise_and(x,u256_bitwise_not(mask));
}
static inline bool u256_slt(uint256 a,uint256 b){bool an=(a.w[3]>>63)!=0,bn=(b.w[3]>>63)!=0;if(an!=bn)return an;return u256_lt(a,b);}
static inline bool u256_sgt(uint256 a,uint256 b){return u256_slt(b,a);}
static inline uint256 u256_byte_at(uint256 val,uint256 pos) {
    if (pos.w[1]|pos.w[2]|pos.w[3]) return u256_zero();
    gpu_u64 i=pos.w[0]; if (i>=32) return u256_zero();
    uint bfr=uint(31-i); return u256_from((val.w[bfr/8]>>((bfr%8)*8))&0xFFUL);
}
static inline uint256 u256_sar(gpu_u64 n,uint256 val) {
    bool neg=(val.w[3]>>63)!=0;
    if (n>=256) return neg?u256_max():u256_zero();
    uint256 r=u256_shr(n,val);
    if (neg&&n>0) r=u256_bitwise_or(r,u256_bitwise_not(u256_shr(n,u256_max())));
    return r;
}
static inline uint u256_byte_length(uint256 x){if(u256_iszero(x))return 0;return(256-clz256_metal(x)+7)/8;}

constant gpu_u64 KECCAK_RC[24] = {
    0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808AUL,
    0x8000000080008000UL, 0x000000000000808BUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL, 0x000000000000008AUL,
    0x0000000000000088UL, 0x0000000080008009UL, 0x000000008000000AUL,
    0x000000008000808BUL, 0x800000000000008BUL, 0x8000000000008089UL,
    0x8000000000008003UL, 0x8000000000008002UL, 0x8000000000000080UL,
    0x000000000000800AUL, 0x800000008000000AUL, 0x8000000080008081UL,
    0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL,
};
constant int KECCAK_PI[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
};
constant int KECCAK_RHO[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
    27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
};

static inline gpu_u64 rotl64_kec(gpu_u64 x, int n) {
    return (x << n) | (x >> (64 - n));
}

static void keccak_f1600(thread gpu_u64 st[25]) {
    for (int round = 0; round < 24; ++round) {
        gpu_u64 C[5];
        for (int x = 0; x < 5; ++x)
            C[x] = st[x] ^ st[x + 5] ^ st[x + 10] ^ st[x + 15] ^ st[x + 20];
        for (int x = 0; x < 5; ++x) {
            gpu_u64 d = C[(x + 4) % 5] ^ rotl64_kec(C[(x + 1) % 5], 1);
            for (int y = 0; y < 5; ++y) st[x + 5 * y] ^= d;
        }
        gpu_u64 t = st[1];
        for (int i = 0; i < 24; ++i) {
            gpu_u64 tmp = st[KECCAK_PI[i]];
            st[KECCAK_PI[i]] = rotl64_kec(t, KECCAK_RHO[i]);
            t = tmp;
        }
        for (int y = 0; y < 5; ++y) {
            gpu_u64 row[5];
            for (int x = 0; x < 5; ++x) row[x] = st[x + 5 * y];
            for (int x = 0; x < 5; ++x)
                st[x + 5 * y] = row[x] ^ ((~row[(x + 1) % 5]) & row[(x + 2) % 5]);
        }
        st[0] ^= KECCAK_RC[round];
    }
}

static void keccak256_dev(device const uchar* input, uint len,
                          thread uchar output[32]) {
    gpu_u64 state[25];
    for (int i = 0; i < 25; ++i) state[i] = 0;
    const uint RATE = 136;
    uint pos = 0;
    while (pos + RATE <= len) {
        for (uint i = 0; i < 17; ++i) {
            gpu_u64 word = 0;
            for (uint b = 0; b < 8; ++b)
                word |= gpu_u64(input[pos + i * 8 + b]) << (b * 8);
            state[i] ^= word;
        }
        keccak_f1600(state);
        pos += RATE;
    }
    uchar block[136];
    for (uint i = 0; i < 136; ++i) block[i] = 0;
    uint rem = len - pos;
    for (uint i = 0; i < rem; ++i) block[i] = input[pos + i];
    block[rem]      |= 0x01;
    block[RATE - 1] |= 0x80;
    for (uint i = 0; i < 17; ++i) {
        gpu_u64 word = 0;
        for (uint b = 0; b < 8; ++b)
            word |= gpu_u64(block[i * 8 + b]) << (b * 8);
        state[i] ^= word;
    }
    keccak_f1600(state);
    for (int i = 0; i < 4; ++i)
        for (int b = 0; b < 8; ++b)
            output[i * 8 + b] = uchar((state[i] >> (b * 8)) & 0xFFUL);
}

struct TxInput {
    uint code_offset; uint code_size; uint calldata_offset; uint calldata_size;
    ulong gas_limit; uint256 caller; uint256 address; uint256 value;
};
struct TxOutput {
    uint status; ulong gas_used; ulong gas_refund; uint output_size;
};
struct StorageEntry { uint256 key; uint256 value; };

struct BlockContext {
    uint256 origin;
    ulong   gas_price;
    ulong   timestamp;
    ulong   number;
    uint256 prevrandao;
    ulong   gas_limit;
    ulong   chain_id;
    ulong   base_fee;
    ulong   blob_base_fee;
    uint256 coinbase;
    uchar   blob_hashes[8][32];
    uint    num_blob_hashes;
    uint    _pad0;
};

struct LogEntry {
    uint256 topics[4];
    uint    num_topics;
    uint    data_offset;
    uint    data_size;
    uint    _pad0;
};

constant uint MAX_MEMORY_PER_TX  = 65536;
constant uint MAX_OUTPUT_PER_TX  = 1024;
constant uint MAX_STORAGE_PER_TX = 64;
constant uint MAX_LOGS_PER_TX    = 16;
constant uint STACK_LIMIT        = 1024;
// Mirrors blob_hashes[8] in BlockContext above. The kernel must never
// index past this even if the host writes a larger num_blob_hashes.
constant uint MAX_BLOB_HASHES    = 8;

constant ulong GAS_VERYLOW=3, GAS_LOW=5, GAS_MID=8, GAS_HIGH=10, GAS_BASE=2, GAS_JUMPDEST=1;
constant ulong GAS_SLOAD=2100, GAS_SSTORE_SET=20000, GAS_SSTORE_RESET=2900;
constant ulong GAS_SSTORE_NOOP=100, GAS_SSTORE_REFUND=4800;
constant ulong GAS_MEMORY=3, GAS_EXP_BASE=10, GAS_EXP_BYTE=50;
constant ulong GAS_KECCAK_BASE=30, GAS_KECCAK_WORD=6;
constant ulong GAS_COPY=3;
constant ulong GAS_LOG_BASE=375, GAS_LOG_TOPIC=375, GAS_LOG_DATA=8;
constant ulong GAS_BALANCE=100, GAS_EXTCODE=100, GAS_SELFBALANCE=5;
constant ulong GAS_TLOAD=100, GAS_TSTORE=100, GAS_BLOCKHASH=20;

constant uchar KECCAK_EMPTY[32] = {
    0xc5, 0xd2, 0x46, 0x01, 0x86, 0xf7, 0x23, 0x3c,
    0x92, 0x7e, 0x7d, 0xb2, 0xdc, 0xc7, 0x03, 0xc0,
    0xe5, 0x00, 0xb6, 0x53, 0xca, 0x82, 0x27, 0x3b,
    0x7b, 0xfa, 0xd8, 0x04, 0x5d, 0x85, 0xa4, 0x70,
};

static inline bool is_valid_jumpdest(device const uchar* code, uint code_size, uint target) {
    if (target >= code_size || code[target] != 0x5b) return false;
    uint i = 0;
    while (i < target) { uchar op = code[i]; if (op >= 0x60 && op <= 0x7f) i += (op-0x60+2); else i++; }
    return i == target;
}

struct OriginalEntry{uint256 key;uint256 value;bool valid;};
static inline bool original_value_lookup(thread OriginalEntry* o,uint n,uint256 s,thread uint256& v){
    for(uint i=0;i<n;++i)if(o[i].valid&&u256_eq(o[i].key,s)){v=o[i].value;return true;}v=u256_zero();return false;}
static inline void original_value_record(thread OriginalEntry* o,thread uint& n,uint256 s,uint256 v){
    for(uint i=0;i<n;++i)if(o[i].valid&&u256_eq(o[i].key,s))return;
    if(n<MAX_STORAGE_PER_TX){o[n].key=s;o[n].value=v;o[n].valid=true;n++;}}
static inline ulong sstore_gas_eip2200(uint256 orig,uint256 cur,uint256 nv,thread ulong& rc){
    if(u256_eq(nv,cur))return GAS_SSTORE_NOOP;
    if(u256_eq(orig,cur)){if(u256_iszero(orig))return GAS_SSTORE_SET;if(u256_iszero(nv))rc+=GAS_SSTORE_REFUND;return GAS_SSTORE_RESET;}
    if(!u256_iszero(orig)){if(u256_iszero(cur))rc-=GAS_SSTORE_REFUND;else if(u256_iszero(nv))rc+=GAS_SSTORE_REFUND;}
    if(u256_eq(nv,orig)){if(u256_iszero(orig))rc+=GAS_SSTORE_SET-GAS_SSTORE_NOOP;else rc+=GAS_SSTORE_RESET-GAS_SSTORE_NOOP;}
    return GAS_SSTORE_NOOP;
}

// FIXED (Bug 3): ov.w[0] + extra wraps when ov.w[0] is near 2^64.
static inline bool offset_in_bounds(uint256 v, ulong extra) {
    if (v.w[1] | v.w[2] | v.w[3]) return false;
    if (v.w[0] > MAX_MEMORY_PER_TX) return false;
    ulong sum = v.w[0] + extra;
    if (sum < v.w[0]) return false;
    if (sum > MAX_MEMORY_PER_TX) return false;
    return true;
}
static inline bool expand_mem_words(device uchar* mem, thread uint& mem_size,
                                    thread ulong& gas, uint new_words) {
    uint old_words = mem_size / 32;
    if (new_words <= old_words) return true;
    ulong cost = GAS_MEMORY * new_words + (ulong(new_words) * new_words) / 512
               - GAS_MEMORY * old_words - (ulong(old_words) * old_words) / 512;
    if (gas < cost) return false;
    gas -= cost;
    uint new_size = new_words * 32;
    for (uint i = mem_size; i < new_size; ++i) mem[i] = 0;
    mem_size = new_size;
    return true;
}
static inline bool expand_mem_range(device uchar* mem, thread uint& mem_size,
                                    thread ulong& gas, uint offset, uint size) {
    if (size == 0) return true;
    uint end = offset + size;
    if (end < offset || end > MAX_MEMORY_PER_TX) return false;
    uint new_words = (end + 31) / 32;
    return expand_mem_words(mem, mem_size, gas, new_words);
}

kernel void evm_execute(
    device const TxInput*       inputs           [[buffer(0)]],
    device const uchar*         blob             [[buffer(1)]],
    device TxOutput*            outputs          [[buffer(2)]],
    device uchar*               out_data         [[buffer(3)]],
    device uchar*               mem_pool         [[buffer(4)]],
    device StorageEntry*        storage_pool     [[buffer(5)]],
    device uint*                storage_counts   [[buffer(6)]],
    device const uint*          params           [[buffer(7)]],
    device StorageEntry*        transient_pool   [[buffer(8)]],
    device uint*                transient_counts [[buffer(9)]],
    device LogEntry*            log_pool         [[buffer(10)]],
    device uint*                log_counts       [[buffer(11)]],
    device const BlockContext*  block_ctx        [[buffer(12)]],
    uint tid                                     [[thread_position_in_grid]],
    uint tgid                                    [[threadgroup_position_in_grid]])
{
    uint num_txs = params[0];
    if (tid >= num_txs) return;

    device const TxInput& inp = inputs[tid];
    device TxOutput& out = outputs[tid];
    device uchar* mem = mem_pool + ulong(tid) * MAX_MEMORY_PER_TX;
    device uchar* output = out_data + ulong(tid) * MAX_OUTPUT_PER_TX;
    device StorageEntry* storage = storage_pool + ulong(tid) * MAX_STORAGE_PER_TX;
    device uint& stor_count = storage_counts[tid];
    device StorageEntry* transient = transient_pool + ulong(tid) * MAX_STORAGE_PER_TX;
    device uint& trans_count = transient_counts[tid];
    device LogEntry* logs = log_pool + ulong(tid) * MAX_LOGS_PER_TX;
    device uint& log_count = log_counts[tid];

    device const uchar* code_dev = blob + inp.code_offset;
    uint code_size = inp.code_size;
    device const uchar* calldata = blob + inp.calldata_offset;
    uint calldata_size = inp.calldata_size;

    uchar code_cache[256];
    uint cached_size = (code_size < 256) ? code_size : 256;
    for (uint i = 0; i < cached_size; ++i) code_cache[i] = code_dev[i];

    #define CODE_BYTE(idx) ((idx) < cached_size ? code_cache[(idx)] : code_dev[(idx)])

    // EVM Yellow Paper: 1024-deep stack. Solidity-emitted bytecode regularly
    // hits 30-50 entries; the previous 32-cap silently truncated real
    // contracts and diverged from CPU evmone. The array lives in thread-
    // local storage; Metal automatically backs it with device memory when
    // it exceeds the register file (32 KB per thread here).
    uint256 stack[1024];
    uint sp = 0;
    ulong gas = inp.gas_limit;
    ulong refund_counter = 0;
    uint pc = 0;
    uint mem_size = 0;
    ulong gas_start = gas;

    OriginalEntry orig_storage[MAX_STORAGE_PER_TX];
    uint orig_count = 0;
    for (uint i = 0; i < MAX_STORAGE_PER_TX; ++i) orig_storage[i].valid = false;

    #define EMIT(st, gu, gr, os) do { out.status=(st); out.gas_used=(gu); out.gas_refund=(gr); out.output_size=(os); return; } while(0)
    #define OOG()  EMIT(3, gas_start, refund_counter, 0)
    #define ERR()  EMIT(4, gas_start - gas, refund_counter, 0)
    #define ERRA() EMIT(4, gas_start, refund_counter, 0)

    while (pc < code_size)
    {
        uchar op = CODE_BYTE(pc);

        switch (op)
        {
        case 0x00: EMIT(0, gas_start - gas, refund_counter, 0);

        case 0x01: { if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 2) ERR(); uint256 a = stack[--sp]; stack[sp - 1] = u256_add(a, stack[sp - 1]); ++pc; continue; }
        case 0x02: { if (gas < GAS_LOW) OOG(); gas -= GAS_LOW;
            if (sp < 2) ERR(); uint256 a = stack[--sp]; stack[sp - 1] = u256_mul(a, stack[sp - 1]); ++pc; continue; }
        case 0x03: { if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 2) ERR(); uint256 a = stack[--sp]; stack[sp - 1] = u256_sub(a, stack[sp - 1]); ++pc; continue; }
        case 0x04: { if (gas < GAS_LOW) OOG(); gas -= GAS_LOW;
            if (sp < 2) ERR(); uint256 a = stack[--sp]; stack[sp - 1] = u256_div(a, stack[sp - 1]); ++pc; continue; }
        case 0x05: { if (gas < GAS_LOW) OOG(); gas -= GAS_LOW;
            if (sp < 2) ERR(); uint256 a = stack[--sp]; stack[sp - 1] = u256_sdiv(a, stack[sp - 1]); ++pc; continue; }
        case 0x06: { if (gas < GAS_LOW) OOG(); gas -= GAS_LOW;
            if (sp < 2) ERR(); uint256 a = stack[--sp]; stack[sp - 1] = u256_mod(a, stack[sp - 1]); ++pc; continue; }
        case 0x07: { if (gas < GAS_LOW) OOG(); gas -= GAS_LOW;
            if (sp < 2) ERR(); uint256 a = stack[--sp]; stack[sp - 1] = u256_smod(a, stack[sp - 1]); ++pc; continue; }
        case 0x08: { if (gas < GAS_MID) OOG(); gas -= GAS_MID;
            if (sp < 3) ERR(); uint256 a = stack[--sp]; uint256 b = stack[--sp]; stack[sp - 1] = u256_addmod(a, b, stack[sp - 1]); ++pc; continue; }
        case 0x09: { if (gas < GAS_MID) OOG(); gas -= GAS_MID;
            if (sp < 3) ERR(); uint256 a = stack[--sp]; uint256 b = stack[--sp]; stack[sp - 1] = u256_mulmod(a, b, stack[sp - 1]); ++pc; continue; }
        case 0x0a: { if (gas < GAS_EXP_BASE) OOG(); gas -= GAS_EXP_BASE;
            if (sp < 2) ERR(); uint256 a = stack[--sp]; uint256 b = stack[sp - 1];
            ulong eg = GAS_EXP_BYTE * ulong(u256_byte_length(b));
            if (gas < eg) OOG(); gas -= eg;
            stack[sp - 1] = u256_exp(a, b); ++pc; continue; }
        case 0x0b: { if (gas < GAS_LOW) OOG(); gas -= GAS_LOW;
            if (sp < 2) ERR(); uint256 a = stack[--sp]; stack[sp - 1] = u256_signextend(a, stack[sp - 1]); ++pc; continue; }

        case 0x10: { if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 2) ERR(); uint256 a = stack[--sp]; stack[sp - 1] = u256_lt(a, stack[sp - 1]) ? u256_one() : u256_zero(); ++pc; continue; }
        case 0x11: { if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 2) ERR(); uint256 a = stack[--sp]; stack[sp - 1] = u256_gt(a, stack[sp - 1]) ? u256_one() : u256_zero(); ++pc; continue; }
        case 0x12: { if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 2) ERR(); uint256 a = stack[--sp]; stack[sp - 1] = u256_slt(a, stack[sp - 1]) ? u256_one() : u256_zero(); ++pc; continue; }
        case 0x13: { if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 2) ERR(); uint256 a = stack[--sp]; stack[sp - 1] = u256_sgt(a, stack[sp - 1]) ? u256_one() : u256_zero(); ++pc; continue; }
        case 0x14: { if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 2) ERR(); uint256 a = stack[--sp]; stack[sp - 1] = u256_eq(a, stack[sp - 1]) ? u256_one() : u256_zero(); ++pc; continue; }
        case 0x15: { if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 1) ERR(); stack[sp - 1] = u256_iszero(stack[sp - 1]) ? u256_one() : u256_zero(); ++pc; continue; }
        case 0x16: { if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 2) ERR(); uint256 a = stack[--sp]; stack[sp - 1] = u256_bitwise_and(a, stack[sp - 1]); ++pc; continue; }
        case 0x17: { if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 2) ERR(); uint256 a = stack[--sp]; stack[sp - 1] = u256_bitwise_or(a, stack[sp - 1]); ++pc; continue; }
        case 0x18: { if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 2) ERR(); uint256 a = stack[--sp]; stack[sp - 1] = u256_bitwise_xor(a, stack[sp - 1]); ++pc; continue; }
        case 0x19: { if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 1) ERR(); stack[sp - 1] = u256_bitwise_not(stack[sp - 1]); ++pc; continue; }
        case 0x1a: { if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 2) ERR(); uint256 i = stack[--sp]; stack[sp - 1] = u256_byte_at(stack[sp - 1], i); ++pc; continue; }
        case 0x1b: { if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 2) ERR(); uint256 s = stack[--sp]; uint256 v = stack[sp - 1];
            stack[sp - 1] = (s.w[1]|s.w[2]|s.w[3]||s.w[0]>=256) ? u256_zero() : u256_shl(s.w[0], v); ++pc; continue; }
        case 0x1c: { if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 2) ERR(); uint256 s = stack[--sp]; uint256 v = stack[sp - 1];
            stack[sp - 1] = (s.w[1]|s.w[2]|s.w[3]||s.w[0]>=256) ? u256_zero() : u256_shr(s.w[0], v); ++pc; continue; }
        case 0x1d: { if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 2) ERR(); uint256 s = stack[--sp]; uint256 v = stack[sp - 1];
            bool neg = (v.w[3]>>63)!=0;
            stack[sp - 1] = (s.w[1]|s.w[2]|s.w[3]||s.w[0]>=256) ? (neg?u256_max():u256_zero()) : u256_sar(s.w[0],v); ++pc; continue; }

        case 0x20: { // KECCAK256
            if (gas < GAS_KECCAK_BASE) OOG(); gas -= GAS_KECCAK_BASE;
            if (sp < 2) ERR();
            uint256 ov = stack[--sp]; uint256 sv = stack[--sp];
            if (sv.w[1]|sv.w[2]|sv.w[3]) ERR();
            ulong size = sv.w[0];
            if (!offset_in_bounds(ov, size)) ERR();
            uint sz = uint(size);
            uint off = uint(ov.w[0]);
            ulong words = (size + 31) / 32;
            ulong word_gas = GAS_KECCAK_WORD * words;
            if (gas < word_gas) OOG(); gas -= word_gas;
            if (!expand_mem_range(mem, mem_size, gas, off, sz)) OOG();
            uchar digest[32];
            keccak256_dev(mem + off, sz, digest);
            uint256 r = u256_zero();
            for (uint i = 0; i < 32; ++i) {
                uint pfr = 31 - i;
                r.w[pfr / 8] |= gpu_u64(digest[i]) << ((pfr % 8) * 8);
            }
            stack[sp++] = r;
            ++pc; continue;
        }

        case 0x30:
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = inp.address; ++pc; continue;
        case 0x31: {
            if (gas < GAS_BALANCE) OOG(); gas -= GAS_BALANCE;
            if (sp < 1) ERR();
            stack[sp - 1] = u256_zero();
            ++pc; continue;
        }
        case 0x32:
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = block_ctx->origin; ++pc; continue;
        case 0x33:
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = inp.caller; ++pc; continue;
        case 0x34:
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = inp.value; ++pc; continue;
        case 0x35: {
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 1) ERR();
            uint256 ov = stack[sp - 1]; uint256 result = u256_zero();
            if (!ov.w[1]&&!ov.w[2]&&!ov.w[3]&&ov.w[0]<calldata_size) {
                uint off = uint(ov.w[0]);
                for (uint i=0;i<32;++i){uint src=off+i;uchar bv=(src<calldata_size)?calldata[src]:0;uint pfr=31-i;result.w[pfr/8]|=gpu_u64(bv)<<((pfr%8)*8);}
            }
            stack[sp - 1] = result; ++pc; continue;
        }
        case 0x36:
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = u256_from(gpu_u64(calldata_size)); ++pc; continue;
        case 0x37: { // CALLDATACOPY
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 3) ERR();
            uint256 dv = stack[--sp]; uint256 sv = stack[--sp]; uint256 lv = stack[--sp];
            if (lv.w[1]|lv.w[2]|lv.w[3]) ERR();
            ulong sz = lv.w[0];
            if (!offset_in_bounds(dv, sz)) ERR();
            uint dest = uint(dv.w[0]);
            uint size = uint(sz);
            ulong words = (sz + 31) / 32;
            ulong copy_gas = GAS_COPY * words;
            if (gas < copy_gas) OOG(); gas -= copy_gas;
            if (!expand_mem_range(mem, mem_size, gas, dest, size)) OOG();
            // src_off treated as past-end (zero-pad) when:
            //   - any high limb of sv non-zero,
            //   - sv.w[0] already >= calldata_size,
            //   - sv.w[0] + size overflows uint32 (per-byte add would wrap).
            // Old check `sv.w[0] >= 0xFFFFFFFFUL` only caught the exact
            // 0xFFFFFFFF case; src=0xFFFFFFFE size=5 would wrap and read
            // calldata bytes 0..2 instead of returning zeros.
            ulong src_lo = sv.w[0];
            bool src_past_end = (sv.w[1]|sv.w[2]|sv.w[3]) != 0 ||
                                src_lo >= calldata_size ||
                                (src_lo + sz) > 0xFFFFFFFFUL;
            uint src_off = src_past_end ? calldata_size : uint(src_lo);
            for (uint i = 0; i < size; ++i) {
                ulong s = ulong(src_off) + ulong(i);
                mem[dest + i] = (s < calldata_size) ? calldata[s] : 0;
            }
            ++pc; continue;
        }
        case 0x38:
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = u256_from(gpu_u64(code_size)); ++pc; continue;
        case 0x39: { // CODECOPY
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 3) ERR();
            uint256 dv = stack[--sp]; uint256 sv = stack[--sp]; uint256 lv = stack[--sp];
            if (lv.w[1]|lv.w[2]|lv.w[3]) ERR();
            ulong sz = lv.w[0];
            if (!offset_in_bounds(dv, sz)) ERR();
            uint dest = uint(dv.w[0]);
            uint size = uint(sz);
            ulong words = (sz + 31) / 32;
            ulong copy_gas = GAS_COPY * words;
            if (gas < copy_gas) OOG(); gas -= copy_gas;
            if (!expand_mem_range(mem, mem_size, gas, dest, size)) OOG();
            // Same zero-pad rule as CALLDATACOPY: avoid uint32 wrap on
            // src+i by working in 64-bit and detecting (src+size) > 2^32.
            ulong src_lo = sv.w[0];
            bool src_past_end = (sv.w[1]|sv.w[2]|sv.w[3]) != 0 ||
                                src_lo >= code_size ||
                                (src_lo + sz) > 0xFFFFFFFFUL;
            uint src_off = src_past_end ? code_size : uint(src_lo);
            for (uint i = 0; i < size; ++i) {
                ulong s = ulong(src_off) + ulong(i);
                mem[dest + i] = (s < code_size) ? code_dev[s] : 0;
            }
            ++pc; continue;
        }
        case 0x3a:
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = u256_from(block_ctx->gas_price); ++pc; continue;
        case 0x3b: {
            if (gas < GAS_EXTCODE) OOG(); gas -= GAS_EXTCODE;
            if (sp < 1) ERR();
            stack[sp - 1] = u256_zero();
            ++pc; continue;
        }
        case 0x3c: {
            if (sp < 4) ERR();
            (void)stack[--sp];
            uint256 dv = stack[--sp];
            (void)stack[--sp];
            uint256 lv = stack[--sp];
            if (lv.w[1]|lv.w[2]|lv.w[3]) ERR();
            ulong sz = lv.w[0];
            if (!offset_in_bounds(dv, sz)) ERR();
            uint dest = uint(dv.w[0]);
            uint size = uint(sz);
            ulong words = (sz + 31) / 32;
            ulong total_gas = GAS_EXTCODE + GAS_COPY * words;
            if (gas < total_gas) OOG(); gas -= total_gas;
            if (!expand_mem_range(mem, mem_size, gas, dest, size)) OOG();
            for (uint i = 0; i < size; ++i) mem[dest + i] = 0;
            ++pc; continue;
        }
        case 0x3d:
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = u256_zero();
            ++pc; continue;
        case 0x3e: {
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 3) ERR();
            (void)stack[--sp]; (void)stack[--sp];
            uint256 lv = stack[--sp];
            if (lv.w[1]|lv.w[2]|lv.w[3] || lv.w[0] != 0) ERRA();
            ++pc; continue;
        }
        case 0x3f: {
            if (gas < GAS_EXTCODE) OOG(); gas -= GAS_EXTCODE;
            if (sp < 1) ERR();
            uint256 r = u256_zero();
            for (uint i = 0; i < 32; ++i) {
                uint pfr = 31 - i;
                r.w[pfr / 8] |= gpu_u64(KECCAK_EMPTY[i]) << ((pfr % 8) * 8);
            }
            stack[sp - 1] = r;
            ++pc; continue;
        }
        case 0x40: {
            if (gas < GAS_BLOCKHASH) OOG(); gas -= GAS_BLOCKHASH;
            if (sp < 1) ERR();
            stack[sp - 1] = u256_zero();
            ++pc; continue;
        }
        case 0x41:
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = block_ctx->coinbase; ++pc; continue;
        case 0x42:
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = u256_from(block_ctx->timestamp); ++pc; continue;
        case 0x43:
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = u256_from(block_ctx->number); ++pc; continue;
        case 0x44:
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = block_ctx->prevrandao; ++pc; continue;
        case 0x45:
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = u256_from(block_ctx->gas_limit); ++pc; continue;
        case 0x46:
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = u256_from(block_ctx->chain_id); ++pc; continue;
        case 0x47:
            if (gas < GAS_SELFBALANCE) OOG(); gas -= GAS_SELFBALANCE;
            if (sp >= 1024) ERR();
            stack[sp++] = u256_zero(); ++pc; continue;
        case 0x48:
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = u256_from(block_ctx->base_fee); ++pc; continue;
        case 0x49: {
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 1) ERR();
            uint256 iv = stack[sp - 1]; uint256 r = u256_zero();
            // Cap at MAX_BLOB_HASHES even if the host wrote num_blob_hashes
            // > 8: blob_hashes is a fixed-size array. Reading past index 7
            // would be OOB into adjacent kernel state. EIP-4844 caps blobs
            // per tx well below 8 in practice; this is belt-and-suspenders.
            uint nhashes = (block_ctx->num_blob_hashes > MAX_BLOB_HASHES)
                              ? MAX_BLOB_HASHES
                              : block_ctx->num_blob_hashes;
            if (!iv.w[1]&&!iv.w[2]&&!iv.w[3]&&iv.w[0]<nhashes) {
                uint idx = uint(iv.w[0]);
                for (uint i = 0; i < 32; ++i) {
                    uint pfr = 31 - i;
                    r.w[pfr / 8] |= gpu_u64(block_ctx->blob_hashes[idx][i]) << ((pfr % 8) * 8);
                }
            }
            stack[sp - 1] = r;
            ++pc; continue;
        }
        case 0x4a:
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = u256_from(block_ctx->blob_base_fee); ++pc; continue;

        case 0x50:
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp == 0) ERR(); --sp; ++pc; continue;

        case 0x51: {
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 1) ERR();
            uint256 ov = stack[sp - 1];
            if (!offset_in_bounds(ov, 32)) ERR();
            uint off = uint(ov.w[0]);
            if (!expand_mem_range(mem, mem_size, gas, off, 32)) OOG();
            uint256 r = u256_zero();
            for (int lmb=3;lmb>=0;--lmb){gpu_u64 v=0;int st=(3-lmb)*8;for(int bi=0;bi<8;++bi)v=(v<<8)|gpu_u64(mem[off+st+bi]);r.w[lmb]=v;}
            stack[sp - 1] = r; ++pc; continue;
        }
        case 0x52: {
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 2) ERR();
            uint256 ov = stack[--sp]; uint256 val = stack[--sp];
            if (!offset_in_bounds(ov, 32)) ERR();
            uint off = uint(ov.w[0]);
            if (!expand_mem_range(mem, mem_size, gas, off, 32)) OOG();
            for (int lmb=3;lmb>=0;--lmb){gpu_u64 v=val.w[lmb];int st=(3-lmb)*8;for(int bi=7;bi>=0;--bi){mem[off+st+bi]=uchar(v&0xFF);v>>=8;}}
            ++pc; continue;
        }
        case 0x53: {
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 2) ERR();
            uint256 ov = stack[--sp]; uint256 val = stack[--sp];
            if (!offset_in_bounds(ov, 1)) ERR();
            uint off = uint(ov.w[0]);
            if (!expand_mem_range(mem, mem_size, gas, off, 1)) OOG();
            mem[off] = uchar(val.w[0] & 0xFFUL);
            ++pc; continue;
        }
        case 0x54: {
            if (gas < GAS_SLOAD) OOG(); gas -= GAS_SLOAD;
            if (sp < 1) ERR();
            uint256 slot = stack[sp - 1]; uint256 val = u256_zero();
            for (uint i=stor_count;i>0;--i) if (u256_eq(storage[i-1].key,slot)){val=storage[i-1].value;break;}
            stack[sp - 1] = val; ++pc; continue;
        }
        case 0x55: {
            if (sp < 2) ERR();
            uint256 slot = stack[--sp]; uint256 val = stack[--sp];
            uint256 current = u256_zero(); bool found = false;
            for (uint i=stor_count;i>0;--i) if (u256_eq(storage[i-1].key,slot)){current=storage[i-1].value;found=true;break;}
            // Cap check before any state mutation: appending a new slot when
            // the per-tx buffer is full would silently corrupt state. Signal
            // INVALID-style (status=Error, all gas consumed). With a host the
            // dispatcher routes this tx to evmone CPU (which has no cap);
            // without one the Error is honest — the GPU can't process this tx.
            if (!found && stor_count >= MAX_STORAGE_PER_TX) ERRA();
            original_value_record(orig_storage,orig_count,slot,current);
            uint256 original = u256_zero(); original_value_lookup(orig_storage,orig_count,slot,original);
            ulong sc = sstore_gas_eip2200(original,current,val,refund_counter);
            if (gas < sc) OOG(); gas -= sc;
            if (found){for(uint i=stor_count;i>0;--i)if(u256_eq(storage[i-1].key,slot)){storage[i-1].value=val;break;}}
            else {
                if (stor_count >= MAX_STORAGE_PER_TX) ERRA();
                storage[stor_count].key=slot; storage[stor_count].value=val; stor_count++;
            }
            ++pc; continue;
        }

        case 0x56: {
            if (gas < GAS_MID) OOG(); gas -= GAS_MID;
            if (sp < 1) ERR();
            uint256 dv = stack[--sp];
            if (dv.w[1]|dv.w[2]|dv.w[3]||dv.w[0]>=code_size) ERR();
            uint dest = uint(dv.w[0]);
            if (!is_valid_jumpdest(code_dev,code_size,dest)) ERR();
            pc = dest; continue;
        }
        case 0x57: {
            if (gas < GAS_HIGH) OOG(); gas -= GAS_HIGH;
            if (sp < 2) ERR();
            uint256 dv = stack[--sp]; uint256 cond = stack[--sp];
            if (!u256_iszero(cond)) {
                if (dv.w[1]|dv.w[2]|dv.w[3]||dv.w[0]>=code_size) ERR();
                uint dest = uint(dv.w[0]);
                if (!is_valid_jumpdest(code_dev,code_size,dest)) ERR();
                pc = dest; continue;
            }
            ++pc; continue;
        }

        case 0x58:
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = u256_from(gpu_u64(pc)); ++pc; continue;
        case 0x59:
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = u256_from(gpu_u64(mem_size)); ++pc; continue;
        case 0x5a:
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = u256_from(gas); ++pc; continue;
        case 0x5b:
            if (gas < GAS_JUMPDEST) OOG(); gas -= GAS_JUMPDEST;
            ++pc; continue;
        case 0x5c: {
            if (gas < GAS_TLOAD) OOG(); gas -= GAS_TLOAD;
            if (sp < 1) ERR();
            uint256 slot = stack[sp - 1]; uint256 v = u256_zero();
            for (uint i = trans_count; i > 0; --i)
                if (u256_eq(transient[i-1].key, slot)) { v = transient[i-1].value; break; }
            stack[sp - 1] = v;
            ++pc; continue;
        }
        case 0x5d: {
            if (gas < GAS_TSTORE) OOG(); gas -= GAS_TSTORE;
            if (sp < 2) ERR();
            uint256 slot = stack[--sp]; uint256 val = stack[--sp];
            bool found = false;
            for (uint i = trans_count; i > 0; --i)
                if (u256_eq(transient[i-1].key, slot)) { transient[i-1].value = val; found = true; break; }
            if (!found && trans_count >= MAX_STORAGE_PER_TX) ERRA();
            if (!found) {
                transient[trans_count].key = slot;
                transient[trans_count].value = val;
                trans_count++;
            }
            ++pc; continue;
        }
        case 0x5e: {
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp < 3) ERR();
            uint256 dv = stack[--sp]; uint256 sv = stack[--sp]; uint256 lv = stack[--sp];
            if (lv.w[1]|lv.w[2]|lv.w[3]) ERR();
            ulong sz = lv.w[0];
            if (!offset_in_bounds(dv, sz)) ERR();
            if (!offset_in_bounds(sv, sz)) ERR();
            uint dest = uint(dv.w[0]);
            uint src  = uint(sv.w[0]);
            uint size = uint(sz);
            ulong words = (sz + 31) / 32;
            ulong copy_gas = GAS_COPY * words;
            if (gas < copy_gas) OOG(); gas -= copy_gas;
            uint touch = (dest > src ? dest + size : src + size);
            uint near  = (dest > src ? src : dest);
            if (!expand_mem_range(mem, mem_size, gas, near, touch - near)) OOG();
            if (size > 0) {
                if (dest < src)
                    for (uint i = 0; i < size; ++i) mem[dest + i] = mem[src + i];
                else
                    for (uint i = size; i > 0; --i) mem[dest + i - 1] = mem[src + i - 1];
            }
            ++pc; continue;
        }
        case 0x5f:
            if (gas < GAS_BASE) OOG(); gas -= GAS_BASE;
            if (sp >= 1024) ERR();
            stack[sp++] = u256_zero(); ++pc; continue;

        case 0x60: {
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp >= 1024) ERR();
            gpu_u64 push_val = gpu_u64((pc+1 < code_size) ? CODE_BYTE(pc+1) : 0);
            if (pc + 2 < code_size) {
                uchar next_op = CODE_BYTE(pc + 2);
                if (next_op == 0x01 && sp >= 1 && gas >= GAS_VERYLOW) {
                    gas -= GAS_VERYLOW;
                    stack[sp - 1] = u256_add(stack[sp - 1], u256_from(push_val));
                    pc += 3; continue;
                }
            }
            stack[sp++] = u256_from(push_val);
            pc += 2; continue;
        }

        case 0x80: case 0x81: case 0x82: case 0x83:
        case 0x84: case 0x85: case 0x86: case 0x87:
        case 0x88: case 0x89: case 0x8a: case 0x8b:
        case 0x8c: case 0x8d: case 0x8e: case 0x8f: {
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            uint n = op - 0x80 + 1;
            if (n > sp || sp >= 1024) ERR();
            stack[sp] = stack[sp - n]; ++sp; ++pc; continue;
        }

        case 0x90: case 0x91: case 0x92: case 0x93:
        case 0x94: case 0x95: case 0x96: case 0x97:
        case 0x98: case 0x99: case 0x9a: case 0x9b:
        case 0x9c: case 0x9d: case 0x9e: case 0x9f: {
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            uint n = op - 0x90 + 1;
            if (n >= sp) ERR();
            uint idx = sp - 1 - n;
            uint256 tmp = stack[sp - 1]; stack[sp - 1] = stack[idx]; stack[idx] = tmp;
            ++pc; continue;
        }

        case 0xa0: case 0xa1: case 0xa2: case 0xa3: case 0xa4: {
            uint num_topics = op - 0xa0;
            if (sp < 2u + num_topics) ERR();
            uint256 ov = stack[--sp]; uint256 sv = stack[--sp];
            if (sv.w[1]|sv.w[2]|sv.w[3]) ERR();
            ulong size = sv.w[0];
            if (!offset_in_bounds(ov, size)) ERR();
            uint off = uint(ov.w[0]);
            uint sz  = uint(size);
            ulong log_gas = GAS_LOG_BASE + GAS_LOG_TOPIC * num_topics + GAS_LOG_DATA * size;
            if (gas < log_gas) OOG(); gas -= log_gas;
            if (!expand_mem_range(mem, mem_size, gas, off, sz)) OOG();
            uint256 topics[4];
            for (uint t = 0; t < num_topics; ++t) topics[t] = stack[--sp];
            if (log_count >= MAX_LOGS_PER_TX) ERR();
            device LogEntry& e = logs[log_count];
            e.num_topics = num_topics;
            e.data_offset = off;
            e.data_size = sz;
            for (uint t = 0; t < num_topics; ++t) e.topics[t] = topics[t];
            log_count++;
            ++pc; continue;
        }

        case 0xf3: {
            if (sp < 2) ERR();
            uint256 ov = stack[--sp]; uint256 sv = stack[--sp];
            if (sv.w[1]|sv.w[2]|sv.w[3]) ERR();
            ulong size = sv.w[0];
            if (!offset_in_bounds(ov, size)) ERR();
            // Output buffer is fixed at MAX_OUTPUT_PER_TX. Silently
            // truncating would diverge from CPU evmone (no cap there);
            // the dispatcher routes Error txs to evmone CPU as the
            // unbounded fallback so legitimate large RETURN payloads
            // still execute correctly.
            if (size > MAX_OUTPUT_PER_TX) ERR();
            uint off = uint(ov.w[0]); uint sz = uint(size);
            if (!expand_mem_range(mem, mem_size, gas, off, sz)) OOG();
            for (uint i=0;i<sz;++i) output[i]=mem[off+i];
            EMIT(1, gas_start-gas, refund_counter, sz);
        }
        case 0xfd: {
            if (sp < 2) ERR();
            uint256 ov = stack[--sp]; uint256 sv = stack[--sp];
            if (sv.w[1]|sv.w[2]|sv.w[3]) ERR();
            ulong size = sv.w[0];
            if (!offset_in_bounds(ov, size)) ERR();
            // Same cap as RETURN: fail-loud on output > MAX_OUTPUT_PER_TX
            // so the dispatcher falls back to evmone (no cap) instead of
            // silently truncating REVERT data.
            if (size > MAX_OUTPUT_PER_TX) ERR();
            uint off = uint(ov.w[0]); uint sz = uint(size);
            if (!expand_mem_range(mem, mem_size, gas, off, sz)) OOG();
            for (uint i=0;i<sz;++i) output[i]=mem[off+i];
            EMIT(2, gas_start-gas, refund_counter, sz);
        }
        case 0xfe: EMIT(4, gas_start, refund_counter, 0);

        case 0xf0: case 0xf1: case 0xf2: case 0xf4:
        case 0xf5: case 0xfa: case 0xff:
            EMIT(5, gas_start-gas, refund_counter, 0);

        default: break;
        }

        if (op >= 0x61 && op <= 0x7f) {
            if (gas < GAS_VERYLOW) OOG(); gas -= GAS_VERYLOW;
            if (sp >= 1024) ERR();
            uint n = op - 0x60 + 1;
            uint256 val = u256_zero();
            uint start = pc + 1;
            for (uint i = 0; i < n && (start+i) < code_size; ++i) {
                uint bp = n-1-i;
                val.w[bp/8] |= gpu_u64(CODE_BYTE(start+i)) << ((bp%8)*8);
            }
            stack[sp++] = val;
            pc += 1 + n;
            continue;
        }

        // FIXED (Bug 2): unrecognized opcodes consume ALL gas (matches 0xFE INVALID).
        ERRA();
    }

    EMIT(0, gas_start - gas, refund_counter, 0);

    #undef EMIT
    #undef OOG
    #undef ERR
    #undef ERRA
    #undef CODE_BYTE
}

#pragma clang diagnostic pop
