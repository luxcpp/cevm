// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file evm_interpreter.hpp
/// CPU/GPU EVM bytecode interpreter (parity reference for the Metal/CUDA
/// kernels under lib/evm/gpu/kernel/evm_kernel.{metal,cu}).
///
/// Cancun-complete in pure-bytecode mode: every opcode that doesn't require
/// host state or nested call frames is implemented and produces the exact
/// gas/output the GPU kernels do. Host-dependent opcodes (BALANCE,
/// EXTCODE*, BLOCKHASH, etc.) return the spec-mandated "no host" defaults
/// (zero / keccak("")) to match Metal's behaviour with no host wired.
///
/// Opcodes that REQUIRE a host (so cannot be made parity here):
///   CALL, STATICCALL, DELEGATECALL, CALLCODE, CREATE, CREATE2,
///   SELFDESTRUCT, GAS-COST-2929 cold/warm bookkeeping. These yield
///   ExecStatus::CallNotSupported so the dispatcher routes the tx to
///   evmone CPU.

#pragma once

#include "evm_stack.hpp"
#include "uint256_gpu.hpp"

namespace evm::gpu::kernel {

// -- Gas costs (Cancun, EIP-3529) ---------------------------------------------

struct GasCost
{
    static constexpr gpu_u64 ZERO       = 0;
    static constexpr gpu_u64 BASE       = 2;
    static constexpr gpu_u64 VERYLOW    = 3;
    static constexpr gpu_u64 LOW        = 5;
    static constexpr gpu_u64 MID        = 8;
    static constexpr gpu_u64 HIGH       = 10;
    static constexpr gpu_u64 JUMPDEST   = 1;
    // EIP-2929 (Berlin) cold/warm pricing. Cold = first access in the tx.
    static constexpr gpu_u64 SLOAD_COLD = 2100;
    static constexpr gpu_u64 SLOAD_WARM = 100;
    static constexpr gpu_u64 ACCOUNT_COLD = 2600;
    static constexpr gpu_u64 ACCOUNT_WARM = 100;
    /// Legacy alias — pre-2929 SLOAD constant. Retained so older callers
    /// that spell `GasCost::SLOAD` keep compiling; the dispatcher uses the
    /// cold/warm split.
    static constexpr gpu_u64 SLOAD      = SLOAD_COLD;
    static constexpr gpu_u64 SSTORE_SET   = 20000;
    static constexpr gpu_u64 SSTORE_RESET = 2900;
    static constexpr gpu_u64 SSTORE_NOOP  = 100;   // EIP-2200: no-op write
    static constexpr gpu_u64 EXP_BASE   = 10;
    static constexpr gpu_u64 EXP_BYTE   = 50;
    static constexpr gpu_u64 MEMORY     = 3;
    static constexpr gpu_u64 LOG_BASE   = 375;
    static constexpr gpu_u64 LOG_DATA   = 8;
    static constexpr gpu_u64 LOG_TOPIC  = 375;
    static constexpr gpu_u64 COPY       = 3;
    static constexpr gpu_u64 KECCAK_BASE = 30;
    static constexpr gpu_u64 KECCAK_WORD = 6;
    static constexpr gpu_u64 BALANCE    = 100;
    static constexpr gpu_u64 EXTCODE    = 100;
    static constexpr gpu_u64 SELFBALANCE= 5;
    static constexpr gpu_u64 TLOAD      = 100;
    static constexpr gpu_u64 TSTORE     = 100;
    static constexpr gpu_u64 BLOCKHASH  = 20;
};

// -- EIP-2929 warm-set caps --------------------------------------------------
static constexpr gpu_u32 MAX_WARM_ADDRESSES = 64;
static constexpr gpu_u32 MAX_WARM_SLOTS     = 128;

// Pre-warmed precompile range. EIP-2929 mandates pre-warming the
// precompile addresses at tx start. Cancun mainnet exposes 0x01..0x0a;
// Prague extends to 0x11. Pre-warming the strict superset 0x01..0x11
// keeps the kernel forward-compatible without branching on revision.
static constexpr gpu_u32 PRECOMPILE_FIRST = 1;
static constexpr gpu_u32 PRECOMPILE_LAST  = 0x11;

// -- EVM Memory ---------------------------------------------------------------

/// Flat byte-addressable memory for EVM execution. Capped per-tx so the
/// dispatcher can size buffers up front; over-cap returns OOG just like a
/// gas exhaustion would.
static constexpr gpu_u32 MAX_MEMORY = 65536;  // matches Metal MAX_MEMORY_PER_TX

struct EvmMemory
{
    gpu_u32 size;  // current logical size in bytes (always multiple of 32)

    GPU_INLINE static gpu_u64 memory_cost(gpu_u32 word_count)
    {
        gpu_u64 w = word_count;
        return GasCost::MEMORY * w + (w * w) / 512;
    }
};

// -- Storage ------------------------------------------------------------------

struct StorageSlot
{
    uint256 value;
    bool found;
};

// -- Logs ---------------------------------------------------------------------

static constexpr gpu_u32 MAX_LOG_TOPICS = 4;
static constexpr gpu_u32 MAX_LOG_DATA = 256;
static constexpr gpu_u32 MAX_LOGS = 64;

struct LogEntry
{
    uint256 topics[MAX_LOG_TOPICS];
    gpu_u32 num_topics;
    gpu_u32 data_offset;  // offset into the EVM memory buffer at emit time
    gpu_u32 data_size;
};

// -- Block context (Cancun) ---------------------------------------------------

/// Per-block context wired by the host before execute(). Mirrors the
/// `BlockContext` struct laid out in evm_kernel_host.hpp / evm_kernel.metal
/// so the same wire bytes flow through every backend.
struct BlockContext
{
    uint256  origin;
    gpu_u64  gas_price;
    gpu_u64  timestamp;
    gpu_u64  number;
    uint256  prevrandao;
    gpu_u64  gas_limit;
    gpu_u64  chain_id;
    gpu_u64  base_fee;
    gpu_u64  blob_base_fee;
    uint256  coinbase;
    gpu_u8_t blob_hashes[8][32];
    gpu_u32  num_blob_hashes;
    gpu_u32  _pad0;
};

static constexpr gpu_u32 MAX_BLOB_HASHES = 8;

// -- Interpreter result -------------------------------------------------------

static constexpr gpu_u32 MAX_OUTPUT = 1024;

struct InterpreterResult
{
    ExecStatus status;
    gpu_u64 gas_used;
    gpu_u64 gas_remaining;
    gpu_u32 output_size;
    /// Signed EIP-2200/3529 refund counter. SSTORE may transiently subtract
    /// refund credit (clear-then-set within a tx), so the accumulator is
    /// signed. The dispatcher floors at 0 and applies the EIP-3529 cap
    /// (max refund = gas_used / 5).
    int64_t gas_refund = 0;
};

// -- Keccak-256 ---------------------------------------------------------------
//
// Header-only port of the same primitive Metal/CUDA use. Keeps evm-kernel a
// pure interface library — no link dependency on evmone::precompiles.

namespace keccak_internal {

static constexpr gpu_u64 RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL,
    0x8000000080008000ULL, 0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008AULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL,
};
static constexpr int PI[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1,
};
static constexpr int RHO[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
    27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44,
};

GPU_INLINE gpu_u64 rotl64(gpu_u64 x, int n)
{
    return (x << n) | (x >> (64 - n));
}

GPU_INLINE void f1600(gpu_u64 st[25])
{
    for (int round = 0; round < 24; ++round)
    {
        gpu_u64 C[5];
        for (int x = 0; x < 5; ++x)
            C[x] = st[x] ^ st[x + 5] ^ st[x + 10] ^ st[x + 15] ^ st[x + 20];
        for (int x = 0; x < 5; ++x)
        {
            gpu_u64 d = C[(x + 4) % 5] ^ rotl64(C[(x + 1) % 5], 1);
            for (int y = 0; y < 5; ++y) st[x + 5 * y] ^= d;
        }
        gpu_u64 t = st[1];
        for (int i = 0; i < 24; ++i)
        {
            gpu_u64 tmp = st[PI[i]];
            st[PI[i]] = rotl64(t, RHO[i]);
            t = tmp;
        }
        for (int y = 0; y < 5; ++y)
        {
            gpu_u64 row[5];
            for (int x = 0; x < 5; ++x) row[x] = st[x + 5 * y];
            for (int x = 0; x < 5; ++x)
                st[x + 5 * y] = row[x] ^ ((~row[(x + 1) % 5]) & row[(x + 2) % 5]);
        }
        st[0] ^= RC[round];
    }
}

GPU_INLINE void keccak256(const gpu_u8_t* input, gpu_u32 len, gpu_u8_t out[32])
{
    gpu_u64 state[25];
    for (int i = 0; i < 25; ++i) state[i] = 0;
    constexpr gpu_u32 RATE = 136;
    gpu_u32 pos = 0;
    while (pos + RATE <= len)
    {
        for (gpu_u32 i = 0; i < 17; ++i)
        {
            gpu_u64 word = 0;
            for (gpu_u32 b = 0; b < 8; ++b)
                word |= gpu_u64(input[pos + i * 8 + b]) << (b * 8);
            state[i] ^= word;
        }
        f1600(state);
        pos += RATE;
    }
    gpu_u8_t block[136];
    for (gpu_u32 i = 0; i < 136; ++i) block[i] = 0;
    gpu_u32 rem = len - pos;
    for (gpu_u32 i = 0; i < rem; ++i) block[i] = input[pos + i];
    block[rem]      |= 0x01;
    block[RATE - 1] |= 0x80;
    for (gpu_u32 i = 0; i < 17; ++i)
    {
        gpu_u64 word = 0;
        for (gpu_u32 b = 0; b < 8; ++b)
            word |= gpu_u64(block[i * 8 + b]) << (b * 8);
        state[i] ^= word;
    }
    f1600(state);
    for (int i = 0; i < 4; ++i)
        for (int b = 0; b < 8; ++b)
            out[i * 8 + b] = gpu_u8_t((state[i] >> (b * 8)) & 0xFF);
}

}  // namespace keccak_internal

/// Hash of the empty input: keccak256(""). Used as the EXTCODEHASH default
/// when no host is wired (matches the GPU kernels and EIP-1052).
GPU_INLINE uint256 keccak_empty()
{
    static constexpr gpu_u8_t bytes[32] = {
        0xc5, 0xd2, 0x46, 0x01, 0x86, 0xf7, 0x23, 0x3c,
        0x92, 0x7e, 0x7d, 0xb2, 0xdc, 0xc7, 0x03, 0xc0,
        0xe5, 0x00, 0xb6, 0x53, 0xca, 0x82, 0x27, 0x3b,
        0x7b, 0xfa, 0xd8, 0x04, 0x5d, 0x85, 0xa4, 0x70,
    };
    uint256 r{};
    for (int i = 0; i < 32; ++i)
    {
        int pfr = 31 - i;
        r.w[pfr / 8] |= gpu_u64(bytes[i]) << ((pfr % 8) * 8);
    }
    return r;
}

// -- The Interpreter ----------------------------------------------------------

/// EVM bytecode interpreter for GPU/CPU execution.
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

    // -- Optional block context (nullptr → all-zero defaults) -----------------
    const BlockContext* block_ctx = nullptr;

    // -- Execution state ------------------------------------------------------
    EvmStack stack;
    gpu_u32 pc;        // program counter
    gpu_u32 mem_size;  // current memory size in bytes

    // -- Storage interface (set by host) --------------------------------------
    uint256*  storage_keys;
    uint256*  storage_values;
    gpu_u32*  storage_count;
    gpu_u32   storage_capacity;

    // -- EIP-2200 original-value tracking -------------------------------------
    uint256*  orig_keys;
    uint256*  orig_values;
    gpu_u32*  orig_count;

    // -- Transient storage (EIP-1153) -----------------------------------------
    // Same shape as the persistent storage buffers; capacity equals
    // storage_capacity. Cleared per-tx by the host (the CPU dispatcher
    // calls execute() with a fresh zero-counted set).
    uint256*  transient_keys   = nullptr;
    uint256*  transient_values = nullptr;
    gpu_u32*  transient_count  = nullptr;

    // -- Log output -----------------------------------------------------------
    LogEntry* logs;
    gpu_u32*  log_count;
    gpu_u32   log_capacity;

    // -- EIP-2929 warm sets ---------------------------------------------------
    // The host seeds these at tx start with the caller, recipient, the
    // standard precompile range, and any EIP-2930 / Config-level entries.
    // SLOAD / SSTORE consult `warm_slot_*` for (address, slot); BALANCE /
    // EXTCODE family (where implemented) consult `warm_addrs`. Append-on-
    // cold so subsequent accesses pay warm. Saturation = cold (over-charge,
    // never under-charge).
    uint256*  warm_addrs;
    gpu_u32*  warm_addr_count;
    gpu_u32   warm_addr_capacity;
    uint256*  warm_slot_addrs;
    uint256*  warm_slot_keys;
    gpu_u32*  warm_slot_count;
    gpu_u32   warm_slot_capacity;

    // -- Helpers --------------------------------------------------------------

    /// EIP-2929 access-list mark for an address. Returns true iff already warm.
    /// Append-on-cold; cap saturates as cold.
    GPU_INLINE bool mark_warm_addr(const uint256& addr)
    {
        gpu_u32 n = *warm_addr_count;
        for (gpu_u32 i = 0; i < n; ++i)
            if (eq(warm_addrs[i], addr))
                return true;
        if (n < warm_addr_capacity)
        {
            warm_addrs[n] = addr;
            *warm_addr_count = n + 1;
        }
        return false;
    }

    /// EIP-2929 access-list mark for (contract, slot). Returns true iff
    /// already warm. Append-on-cold; cap saturates as cold.
    GPU_INLINE bool mark_warm_slot(const uint256& addr, const uint256& slot)
    {
        gpu_u32 n = *warm_slot_count;
        for (gpu_u32 i = 0; i < n; ++i)
            if (eq(warm_slot_addrs[i], addr) && eq(warm_slot_keys[i], slot))
                return true;
        if (n < warm_slot_capacity)
        {
            warm_slot_addrs[n] = addr;
            warm_slot_keys[n]  = slot;
            *warm_slot_count = n + 1;
        }
        return false;
    }

    /// Yellow-paper §6.1.4 + EIP-2929 §"Specification": the tx's caller,
    /// recipient, and the precompile range are warm at tx start.
    GPU_INLINE void seed_warm_sets()
    {
        mark_warm_addr(caller);
        mark_warm_addr(address);
        for (gpu_u32 i = PRECOMPILE_FIRST; i <= PRECOMPILE_LAST; ++i)
        {
            uint256 a = uint256::zero();
            a.w[0] = gpu_u64(i);
            mark_warm_addr(a);
        }
    }

    GPU_INLINE bool consume_gas(gpu_u64 cost)
    {
        if (gas < cost)
            return false;
        gas -= cost;
        return true;
    }

    /// Expand memory to cover [offset, offset+size). Returns gas cost or
    /// the all-ones sentinel on failure (overflow / past MAX_MEMORY).
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
        gpu_u32 start = pc + 1;
        for (gpu_u32 i = 0; i < num_bytes && (start + i) < code_size; ++i)
        {
            gpu_u32 byte_pos = num_bytes - 1 - i;
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
        if (code[target] != 0x5b)
            return false;
        gpu_u32 i = 0;
        while (i < target)
        {
            gpu_u8_t op = code[i];
            if (op >= 0x60 && op <= 0x7f)
                i += 1 + (op - 0x60 + 1);
            else
                i += 1;
        }
        return i == target;
    }

    /// SLOAD: linear scan, latest write wins.
    GPU_INLINE uint256 sload(const uint256& slot) const
    {
        gpu_u32 count = *storage_count;
        for (gpu_u32 i = count; i > 0; --i)
        {
            if (eq(storage_keys[i - 1], slot))
                return storage_values[i - 1];
        }
        return uint256::zero();
    }

    GPU_INLINE uint256 sload_original(const uint256& slot) const
    {
        gpu_u32 count = *orig_count;
        for (gpu_u32 i = count; i > 0; --i)
        {
            if (eq(orig_keys[i - 1], slot))
                return orig_values[i - 1];
        }
        return uint256::zero();
    }

    GPU_INLINE void record_original(const uint256& slot, const uint256& value)
    {
        gpu_u32 count = *orig_count;
        for (gpu_u32 i = count; i > 0; --i)
        {
            if (eq(orig_keys[i - 1], slot))
                return;
        }
        if (count < storage_capacity)
        {
            orig_keys[count] = slot;
            orig_values[count] = value;
            *orig_count = count + 1;
        }
    }

    /// Transient SLOAD (EIP-1153). Latest write wins, default 0.
    GPU_INLINE uint256 tload(const uint256& slot) const
    {
        if (!transient_count)
            return uint256::zero();
        gpu_u32 count = *transient_count;
        for (gpu_u32 i = count; i > 0; --i)
        {
            if (eq(transient_keys[i - 1], slot))
                return transient_values[i - 1];
        }
        return uint256::zero();
    }

    /// EIP-2200 SSTORE gas calculation, mirroring evm_kernel.metal byte-for-byte.
    GPU_INLINE gpu_u64 sstore_gas_eip2200(const uint256& original,
                                           const uint256& current,
                                           const uint256& new_val,
                                           int64_t& refund_counter) const
    {
        constexpr int64_t SSTORE_REFUND = 4800;  // EIP-3529 (Cancun)

        if (eq(new_val, current))
            return GasCost::SSTORE_NOOP;

        if (eq(original, current))
        {
            if (iszero(original))
                return GasCost::SSTORE_SET;
            if (iszero(new_val))
                refund_counter += SSTORE_REFUND;
            return GasCost::SSTORE_RESET;
        }

        if (!iszero(original))
        {
            if (iszero(current))
                refund_counter -= SSTORE_REFUND;
            else if (iszero(new_val))
                refund_counter += SSTORE_REFUND;
        }

        if (eq(new_val, original))
        {
            if (iszero(original))
                refund_counter += static_cast<int64_t>(GasCost::SSTORE_SET   - GasCost::SSTORE_NOOP);
            else
                refund_counter += static_cast<int64_t>(GasCost::SSTORE_RESET - GasCost::SSTORE_NOOP);
        }
        return GasCost::SSTORE_NOOP;
    }

    /// SSTORE: write value for slot. Appends or updates.
    GPU_INLINE void sstore(const uint256& slot, const uint256& val)
    {
        gpu_u32 count = *storage_count;
        for (gpu_u32 i = count; i > 0; --i)
        {
            if (eq(storage_keys[i - 1], slot))
            {
                storage_values[i - 1] = val;
                return;
            }
        }
        if (count < storage_capacity)
        {
            storage_keys[count] = slot;
            storage_values[count] = val;
            *storage_count = count + 1;
        }
    }

    /// TSTORE: write value for slot in transient storage.
    GPU_INLINE void tstore(const uint256& slot, const uint256& val)
    {
        gpu_u32 count = *transient_count;
        for (gpu_u32 i = count; i > 0; --i)
        {
            if (eq(transient_keys[i - 1], slot))
            {
                transient_values[i - 1] = val;
                return;
            }
        }
        if (count < storage_capacity)
        {
            transient_keys[count] = slot;
            transient_values[count] = val;
            *transient_count = count + 1;
        }
    }

    // -- Main execution loop --------------------------------------------------

    GPU_INLINE InterpreterResult execute(gpu_u8_t* mem, gpu_u8_t* output)
    {
        pc = 0;
        mem_size = 0;
        gpu_u64 gas_start = gas;
        // Signed accumulator for the EIP-2200 refund counter.
        int64_t refund_counter = 0;

        // EIP-2929: pre-warm caller, recipient, precompiles before the
        // first opcode runs. Caller-supplied entries are appended to the
        // sets by the host before execute() is invoked.
        seed_warm_sets();

        // Sentinel block context used when the host doesn't wire one. All
        // fields zero matches the GPU "no host" defaults the parity tests
        // assert on.
        BlockContext zero_ctx{};
        const BlockContext& bctx = block_ctx ? *block_ctx : zero_ctx;

        while (pc < code_size)
        {
            gpu_u8_t op = code[pc];

            // -- STOP (0x00) --------------------------------------------------
            if (op == 0x00)
            {
                return {ExecStatus::Stop, gas_start - gas, gas, 0, refund_counter};
            }

            // -- Arithmetic (0x01 - 0x0b) -------------------------------------
            if (op >= 0x01 && op <= 0x0b)
            {
                gpu_u64 base_gas = (op == 0x01 || op == 0x03) ? GasCost::VERYLOW :
                                   (op == 0x08 || op == 0x09) ? GasCost::MID :
                                   (op == 0x0a) ? GasCost::EXP_BASE : GasCost::LOW;
                if (!consume_gas(base_gas))
                    return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};

                uint256 a, b;
                ExecStatus s;

                switch (op)
                {
                case 0x01: // ADD
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.push(add(a, b));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                case 0x02: // MUL
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.push(mul(a, b));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                case 0x03: // SUB
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.push(sub(a, b));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                case 0x04: // DIV
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.push(div(a, b));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                case 0x05: // SDIV
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.push(sdiv(a, b));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                case 0x06: // MOD
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.push(mod(a, b));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                case 0x07: // SMOD
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.push(smod(a, b));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                case 0x08: // ADDMOD
                {
                    uint256 n;
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(n); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.push(addmod(a, b, n));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                }
                case 0x09: // MULMOD
                {
                    uint256 n;
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(n); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.push(mulmod(a, b, n));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                }
                case 0x0a: // EXP
                {
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    gpu_u32 exp_bytes = 0;
                    if (!iszero(b))
                        exp_bytes = (256 - clz256(b) + 7) / 8;
                    if (!consume_gas(GasCost::EXP_BYTE * exp_bytes))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.push(exp(a, b));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                }
                case 0x0b: // SIGNEXTEND
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.push(signextend(a, b));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                }
                ++pc;
                continue;
            }

            // -- Comparison (0x10 - 0x15) -------------------------------------
            if (op >= 0x10 && op <= 0x15)
            {
                if (!consume_gas(GasCost::VERYLOW))
                    return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};

                uint256 a, b;
                ExecStatus s;

                switch (op)
                {
                case 0x10: // LT
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.push(lt(a, b) ? uint256::one() : uint256::zero());
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                case 0x11: // GT
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.push(gt(a, b) ? uint256::one() : uint256::zero());
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                case 0x12: // SLT
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.push(slt(a, b) ? uint256::one() : uint256::zero());
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                case 0x13: // SGT
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.push(sgt(a, b) ? uint256::one() : uint256::zero());
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                case 0x14: // EQ
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.push(eq(a, b) ? uint256::one() : uint256::zero());
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                case 0x15: // ISZERO
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.push(iszero(a) ? uint256::one() : uint256::zero());
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                }
                ++pc;
                continue;
            }

            // -- Bitwise (0x16 - 0x1d) ----------------------------------------
            if (op >= 0x16 && op <= 0x1d)
            {
                if (!consume_gas(GasCost::VERYLOW))
                    return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};

                uint256 a, b;
                ExecStatus s;

                switch (op)
                {
                case 0x16: // AND
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.push(bitwise_and(a, b));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                case 0x17: // OR
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.push(bitwise_or(a, b));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                case 0x18: // XOR
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.push(bitwise_xor(a, b));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                case 0x19: // NOT
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.push(bitwise_not(a));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                case 0x1a: // BYTE
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.push(byte_at(b, a));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                case 0x1b: // SHL
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.push(shl(a, b));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                case 0x1c: // SHR
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.push(shr(a, b));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                case 0x1d: // SAR
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.push(sar(a, b));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                }
                ++pc;
                continue;
            }

            // -- KECCAK256 (0x20) ---------------------------------------------
            if (op == 0x20)
            {
                if (!consume_gas(GasCost::KECCAK_BASE))
                    return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                ExecStatus s;
                uint256 ov, sv;
                s = stack.pop(ov); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                s = stack.pop(sv); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                if (sv.w[1] | sv.w[2] | sv.w[3])
                    return {ExecStatus::InvalidMemAccess, gas_start - gas, gas, 0, refund_counter};
                if (ov.w[1] | ov.w[2] | ov.w[3])
                    return {ExecStatus::InvalidMemAccess, gas_start - gas, gas, 0, refund_counter};
                gpu_u64 size64 = sv.w[0];
                gpu_u64 off64  = ov.w[0];
                if (off64 + size64 < off64 || off64 + size64 > MAX_MEMORY)
                    return {ExecStatus::InvalidMemAccess, gas_start - gas, gas, 0, refund_counter};
                gpu_u32 sz  = gpu_u32(size64);
                gpu_u32 off = gpu_u32(off64);
                gpu_u64 words = (size64 + 31) / 32;
                if (!consume_gas(GasCost::KECCAK_WORD * words))
                    return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                gpu_u64 mem_cost = expand_memory(off, sz, mem);
                if (mem_cost == ~gpu_u64(0) || !consume_gas(mem_cost))
                    return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                gpu_u8_t digest[32];
                keccak_internal::keccak256(mem + off, sz, digest);
                uint256 r{};
                for (int i = 0; i < 32; ++i)
                {
                    int pfr = 31 - i;
                    r.w[pfr / 8] |= gpu_u64(digest[i]) << ((pfr % 8) * 8);
                }
                s = stack.push(r);
                if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                ++pc;
                continue;
            }

            // -- Environment (0x30 - 0x3f) ------------------------------------
            if (op >= 0x30 && op <= 0x3f)
            {
                ExecStatus s;
                uint256 a, b, c, d;

                switch (op)
                {
                case 0x30: // ADDRESS — gas BASE
                    if (!consume_gas(GasCost::BASE))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.push(address);
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                case 0x31: // BALANCE — EIP-2929 cold 2600 / warm 100. No host => 0.
                {
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    gpu_u64 bal_cost = mark_warm_addr(a) ? GasCost::ACCOUNT_WARM : GasCost::ACCOUNT_COLD;
                    if (!consume_gas(bal_cost))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.push(uint256::zero());
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                }
                case 0x32: // ORIGIN
                    if (!consume_gas(GasCost::BASE))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.push(bctx.origin);
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                case 0x33: // CALLER
                    if (!consume_gas(GasCost::BASE))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.push(caller);
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                case 0x34: // CALLVALUE
                    if (!consume_gas(GasCost::BASE))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.push(value);
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                case 0x35: // CALLDATALOAD
                {
                    if (!consume_gas(GasCost::VERYLOW))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    uint256 result = uint256::zero();
                    if (!a.w[1] && !a.w[2] && !a.w[3] && a.w[0] < calldata_size)
                    {
                        gpu_u32 off = gpu_u32(a.w[0]);
                        for (gpu_u32 i = 0; i < 32; ++i)
                        {
                            gpu_u32 src = off + i;
                            gpu_u8_t byte_val = (src < calldata_size) ? calldata[src] : 0;
                            gpu_u32 pfr = 31 - i;
                            result.w[pfr / 8] |= gpu_u64(byte_val) << ((pfr % 8) * 8);
                        }
                    }
                    s = stack.push(result);
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                }
                case 0x36: // CALLDATASIZE
                    if (!consume_gas(GasCost::BASE))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.push(uint256{gpu_u64(calldata_size)});
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                case 0x37: // CALLDATACOPY: dest=top, src=second, size=third
                {
                    if (!consume_gas(GasCost::VERYLOW))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(c); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    if (c.w[1] | c.w[2] | c.w[3] || a.w[1] | a.w[2] | a.w[3])
                        return {ExecStatus::InvalidMemAccess, gas_start - gas, gas, 0, refund_counter};
                    gpu_u32 dest = gpu_u32(a.w[0]);
                    gpu_u64 src_lo = b.w[0];
                    gpu_u64 sz64   = c.w[0];
                    gpu_u32 sz     = gpu_u32(sz64);
                    if (sz > 0)
                    {
                        gpu_u64 words = (sz64 + 31) / 32;
                        if (!consume_gas(words * GasCost::COPY))
                            return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                        gpu_u64 mem_cost = expand_memory(dest, sz, mem);
                        if (mem_cost == ~gpu_u64(0) || !consume_gas(mem_cost))
                            return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                        // Match Metal's zero-pad rule: src_off = calldata_size
                        // when any high limb non-zero, src_lo >= calldata_size,
                        // or src_lo + sz overflows uint32.
                        bool past_end = (b.w[1] | b.w[2] | b.w[3]) != 0 ||
                                        src_lo >= calldata_size ||
                                        (src_lo + sz64) > 0xFFFFFFFFULL;
                        gpu_u32 src_off = past_end ? calldata_size : gpu_u32(src_lo);
                        for (gpu_u32 i = 0; i < sz; ++i)
                        {
                            gpu_u64 ss = gpu_u64(src_off) + gpu_u64(i);
                            mem[dest + i] = (ss < calldata_size) ? calldata[ss] : 0;
                        }
                    }
                    break;
                }
                case 0x38: // CODESIZE
                    if (!consume_gas(GasCost::BASE))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.push(uint256{gpu_u64(code_size)});
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                case 0x39: // CODECOPY: dest=top, src=second, size=third
                {
                    if (!consume_gas(GasCost::VERYLOW))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(c); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    if (c.w[1] | c.w[2] | c.w[3] || a.w[1] | a.w[2] | a.w[3])
                        return {ExecStatus::InvalidMemAccess, gas_start - gas, gas, 0, refund_counter};
                    gpu_u32 dest = gpu_u32(a.w[0]);
                    gpu_u64 src_lo = b.w[0];
                    gpu_u64 sz64   = c.w[0];
                    gpu_u32 sz     = gpu_u32(sz64);
                    if (sz > 0)
                    {
                        gpu_u64 words = (sz64 + 31) / 32;
                        if (!consume_gas(words * GasCost::COPY))
                            return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                        gpu_u64 mem_cost = expand_memory(dest, sz, mem);
                        if (mem_cost == ~gpu_u64(0) || !consume_gas(mem_cost))
                            return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                        bool past_end = (b.w[1] | b.w[2] | b.w[3]) != 0 ||
                                        src_lo >= code_size ||
                                        (src_lo + sz64) > 0xFFFFFFFFULL;
                        gpu_u32 src_off = past_end ? code_size : gpu_u32(src_lo);
                        for (gpu_u32 i = 0; i < sz; ++i)
                        {
                            gpu_u64 ss = gpu_u64(src_off) + gpu_u64(i);
                            mem[dest + i] = (ss < code_size) ? code[ss] : 0;
                        }
                    }
                    break;
                }
                case 0x3a: // GASPRICE
                    if (!consume_gas(GasCost::BASE))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.push(uint256{bctx.gas_price});
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                case 0x3b: // EXTCODESIZE — EIP-2929 cold/warm. No host => 0.
                {
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    gpu_u64 ext_cost = mark_warm_addr(a) ? GasCost::ACCOUNT_WARM : GasCost::ACCOUNT_COLD;
                    if (!consume_gas(ext_cost))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.push(uint256::zero());
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                }
                case 0x3c: // EXTCODECOPY: addr=top, dest=2nd, src=3rd, size=4th
                {
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(c); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(d); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    // a=addr, b=dest, c=src (ignored without host), d=size.
                    if (d.w[1] | d.w[2] | d.w[3] || b.w[1] | b.w[2] | b.w[3])
                        return {ExecStatus::InvalidMemAccess, gas_start - gas, gas, 0, refund_counter};
                    gpu_u32 dest = gpu_u32(b.w[0]);
                    gpu_u64 sz64 = d.w[0];
                    gpu_u32 sz   = gpu_u32(sz64);
                    gpu_u64 words = (sz64 + 31) / 32;
                    // EIP-2929: cold/warm address surcharge + per-word copy.
                    gpu_u64 access_cost = mark_warm_addr(a) ? GasCost::ACCOUNT_WARM : GasCost::ACCOUNT_COLD;
                    gpu_u64 total_gas = access_cost + GasCost::COPY * words;
                    if (!consume_gas(total_gas))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    if (sz > 0)
                    {
                        gpu_u64 mem_cost = expand_memory(dest, sz, mem);
                        if (mem_cost == ~gpu_u64(0) || !consume_gas(mem_cost))
                            return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                        for (gpu_u32 i = 0; i < sz; ++i)
                            mem[dest + i] = 0;  // no host: zero-fill
                    }
                    break;
                }
                case 0x3d: // RETURNDATASIZE — no prior call => 0
                    if (!consume_gas(GasCost::BASE))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.push(uint256::zero());
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                case 0x3e: // RETURNDATACOPY: dest=top, src=2nd, size=3rd
                {
                    if (!consume_gas(GasCost::VERYLOW))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(c); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    // size > 0 with no return data is an EIP-211 error (all gas).
                    if (c.w[1] | c.w[2] | c.w[3] || c.w[0] != 0)
                    {
                        gas = 0;
                        return {ExecStatus::InvalidOpcode, gas_start, 0, 0, refund_counter};
                    }
                    break;
                }
                case 0x3f: // EXTCODEHASH — EIP-2929 cold/warm. No host => keccak256("").
                {
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    gpu_u64 hash_cost = mark_warm_addr(a) ? GasCost::ACCOUNT_WARM : GasCost::ACCOUNT_COLD;
                    if (!consume_gas(hash_cost))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.push(keccak_empty());
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                }
                }
                ++pc;
                continue;
            }

            // -- Block context (0x40 - 0x4a) ----------------------------------
            if (op >= 0x40 && op <= 0x4a)
            {
                ExecStatus s;
                uint256 a;

                switch (op)
                {
                case 0x40: // BLOCKHASH — no chain history => 0, gas 20
                    if (!consume_gas(GasCost::BLOCKHASH))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.push(uint256::zero());
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                case 0x41: // COINBASE
                    if (!consume_gas(GasCost::BASE))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.push(bctx.coinbase);
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                case 0x42: // TIMESTAMP
                    if (!consume_gas(GasCost::BASE))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.push(uint256{bctx.timestamp});
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                case 0x43: // NUMBER
                    if (!consume_gas(GasCost::BASE))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.push(uint256{bctx.number});
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                case 0x44: // PREVRANDAO (ex-DIFFICULTY)
                    if (!consume_gas(GasCost::BASE))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.push(bctx.prevrandao);
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                case 0x45: // GASLIMIT
                    if (!consume_gas(GasCost::BASE))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.push(uint256{bctx.gas_limit});
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                case 0x46: // CHAINID
                    if (!consume_gas(GasCost::BASE))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.push(uint256{bctx.chain_id});
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                case 0x47: // SELFBALANCE — gas 5, no host => 0
                    if (!consume_gas(GasCost::SELFBALANCE))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.push(uint256::zero());
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                case 0x48: // BASEFEE
                    if (!consume_gas(GasCost::BASE))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.push(uint256{bctx.base_fee});
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                case 0x49: // BLOBHASH — gas VERYLOW
                {
                    if (!consume_gas(GasCost::VERYLOW))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    uint256 r = uint256::zero();
                    gpu_u32 nh = (bctx.num_blob_hashes > MAX_BLOB_HASHES)
                                    ? MAX_BLOB_HASHES : bctx.num_blob_hashes;
                    if (!a.w[1] && !a.w[2] && !a.w[3] && a.w[0] < nh)
                    {
                        gpu_u32 idx = gpu_u32(a.w[0]);
                        for (int i = 0; i < 32; ++i)
                        {
                            int pfr = 31 - i;
                            r.w[pfr / 8] |= gpu_u64(bctx.blob_hashes[idx][i]) << ((pfr % 8) * 8);
                        }
                    }
                    s = stack.push(r);
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                }
                case 0x4a: // BLOBBASEFEE
                    if (!consume_gas(GasCost::BASE))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.push(uint256{bctx.blob_base_fee});
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                }
                ++pc;
                continue;
            }

            // -- POP (0x50) ---------------------------------------------------
            if (op == 0x50)
            {
                if (!consume_gas(GasCost::BASE))
                    return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                ExecStatus s = stack.drop();
                if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                ++pc;
                continue;
            }

            // -- Memory (0x51 - 0x53) -----------------------------------------
            // MSIZE (0x59) is BASE-priced and lives in the control-flow band
            // alongside PC/GAS/JUMPDEST; it's handled in that branch.
            if (op == 0x51 || op == 0x52 || op == 0x53)
            {
                if (!consume_gas(GasCost::VERYLOW))
                    return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};

                ExecStatus s;
                uint256 a, b;

                switch (op)
                {
                case 0x51: // MLOAD
                {
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    if (a.w[1] | a.w[2] | a.w[3])
                        return {ExecStatus::InvalidMemAccess, gas_start - gas, gas, 0, refund_counter};
                    gpu_u32 off = gpu_u32(a.w[0]);
                    gpu_u64 mem_cost = expand_memory(off, 32, mem);
                    if (mem_cost == ~gpu_u64(0) || !consume_gas(mem_cost))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.push(mload(mem, off));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    break;
                }
                case 0x52: // MSTORE: offset=top, value=second
                {
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    if (a.w[1] | a.w[2] | a.w[3])
                        return {ExecStatus::InvalidMemAccess, gas_start - gas, gas, 0, refund_counter};
                    gpu_u32 off = gpu_u32(a.w[0]);
                    gpu_u64 mem_cost = expand_memory(off, 32, mem);
                    if (mem_cost == ~gpu_u64(0) || !consume_gas(mem_cost))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    mstore(mem, off, b);
                    break;
                }
                case 0x53: // MSTORE8: offset=top, value=second
                {
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    if (a.w[1] | a.w[2] | a.w[3])
                        return {ExecStatus::InvalidMemAccess, gas_start - gas, gas, 0, refund_counter};
                    gpu_u32 off = gpu_u32(a.w[0]);
                    gpu_u64 mem_cost = expand_memory(off, 1, mem);
                    if (mem_cost == ~gpu_u64(0) || !consume_gas(mem_cost))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    mem[off] = gpu_u8_t(b.w[0] & 0xFF);
                    break;
                }
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
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    // EIP-2929: cold 2100 / warm 100 keyed on (contract, slot).
                    bool warm = mark_warm_slot(address, a);
                    gpu_u64 cost = warm ? GasCost::SLOAD_WARM : GasCost::SLOAD_COLD;
                    if (!consume_gas(cost))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.push(sload(a));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                }
                else // SSTORE: key=top, value=second (EIP-2200 + EIP-2929)
                {
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    bool slot_present = false;
                    {
                        gpu_u32 sc = *storage_count;
                        for (gpu_u32 i = sc; i > 0; --i)
                            if (eq(storage_keys[i - 1], a)) { slot_present = true; break; }
                    }
                    if (!slot_present && *storage_count >= storage_capacity)
                    {
                        gas = 0;
                        return {ExecStatus::InvalidOpcode, gas_start, 0, 0, refund_counter};
                    }
                    // EIP-2929 surcharge on first access to the slot (cold).
                    bool warm = mark_warm_slot(address, a);
                    gpu_u64 access_surcharge = warm ? 0 : GasCost::SLOAD_COLD;
                    // EIP-2200: base cost depends on original, current, and new value.
                    // sstore_gas_eip2200 mutates refund_counter (signed delta).
                    uint256 current = sload(a);
                    record_original(a, current);
                    uint256 original = sload_original(a);
                    gpu_u64 base = sstore_gas_eip2200(original, current, b, refund_counter);
                    if (!consume_gas(base + access_surcharge))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    sstore(a, b);
                }
                ++pc;
                continue;
            }

            // -- Control flow / misc (0x56 - 0x5e) ----------------------------
            if (op >= 0x56 && op <= 0x5e)
            {
                ExecStatus s;
                uint256 a, b, c;

                switch (op)
                {
                case 0x56: // JUMP
                    if (!consume_gas(GasCost::MID))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    if (a.w[1] | a.w[2] | a.w[3])
                        return {ExecStatus::InvalidJump, gas_start - gas, gas, 0, refund_counter};
                    {
                        gpu_u32 dest = gpu_u32(a.w[0]);
                        if (!is_jumpdest(dest))
                            return {ExecStatus::InvalidJump, gas_start - gas, gas, 0, refund_counter};
                        pc = dest;
                    }
                    continue;

                case 0x57: // JUMPI: dest=top, cond=second
                    if (!consume_gas(GasCost::HIGH))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    if (!iszero(b))
                    {
                        if (a.w[1] | a.w[2] | a.w[3])
                            return {ExecStatus::InvalidJump, gas_start - gas, gas, 0, refund_counter};
                        gpu_u32 dest = gpu_u32(a.w[0]);
                        if (!is_jumpdest(dest))
                            return {ExecStatus::InvalidJump, gas_start - gas, gas, 0, refund_counter};
                        pc = dest;
                        continue;
                    }
                    ++pc;
                    continue;

                case 0x58: // PC
                    if (!consume_gas(GasCost::BASE))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.push(uint256{gpu_u64(pc)});
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    ++pc;
                    continue;

                case 0x59: // MSIZE — gas BASE (2)
                    if (!consume_gas(GasCost::BASE))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.push(uint256{gpu_u64(mem_size)});
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    ++pc;
                    continue;

                case 0x5a: // GAS
                    if (!consume_gas(GasCost::BASE))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.push(uint256{gas});
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    ++pc;
                    continue;

                case 0x5b: // JUMPDEST
                    if (!consume_gas(GasCost::JUMPDEST))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    ++pc;
                    continue;

                case 0x5c: // TLOAD (EIP-1153)
                    if (!consume_gas(GasCost::TLOAD))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.push(tload(a));
                    if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    ++pc;
                    continue;

                case 0x5d: // TSTORE (EIP-1153): key=top, value=second
                {
                    if (!consume_gas(GasCost::TSTORE))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    // Cap check (mirrors SSTORE): appending a new slot when
                    // the per-tx transient buffer is full would silently drop
                    // the write. Fail loud so the dispatcher can route to
                    // evmone CPU.
                    if (transient_count && transient_keys && transient_values)
                    {
                        bool slot_present = false;
                        gpu_u32 tc = *transient_count;
                        for (gpu_u32 i = tc; i > 0; --i)
                            if (eq(transient_keys[i - 1], a)) { slot_present = true; break; }
                        if (!slot_present && tc >= storage_capacity)
                        {
                            gas = 0;
                            return {ExecStatus::InvalidOpcode, gas_start, 0, 0, refund_counter};
                        }
                        tstore(a, b);
                    }
                    ++pc;
                    continue;
                }

                case 0x5e: // MCOPY (EIP-5656): dest=top, src=2nd, size=3rd
                {
                    if (!consume_gas(GasCost::VERYLOW))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    s = stack.pop(a); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(b); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    s = stack.pop(c); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    if (c.w[1] | c.w[2] | c.w[3] ||
                        a.w[1] | a.w[2] | a.w[3] ||
                        b.w[1] | b.w[2] | b.w[3])
                        return {ExecStatus::InvalidMemAccess, gas_start - gas, gas, 0, refund_counter};
                    gpu_u32 dest = gpu_u32(a.w[0]);
                    gpu_u32 src  = gpu_u32(b.w[0]);
                    gpu_u64 sz64 = c.w[0];
                    gpu_u32 sz   = gpu_u32(sz64);
                    gpu_u64 words = (sz64 + 31) / 32;
                    if (!consume_gas(GasCost::COPY * words))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                    // Expand memory to cover both source and destination ranges.
                    gpu_u32 hi_end = (dest + sz > src + sz) ? dest + sz : src + sz;
                    gpu_u32 lo_off = (dest < src) ? dest : src;
                    if (hi_end < lo_off || hi_end > MAX_MEMORY)
                        return {ExecStatus::InvalidMemAccess, gas_start - gas, gas, 0, refund_counter};
                    if (sz > 0)
                    {
                        gpu_u64 mem_cost = expand_memory(lo_off, hi_end - lo_off, mem);
                        if (mem_cost == ~gpu_u64(0) || !consume_gas(mem_cost))
                            return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                        // memmove with overlap.
                        if (dest < src)
                            for (gpu_u32 i = 0; i < sz; ++i) mem[dest + i] = mem[src + i];
                        else
                            for (gpu_u32 i = sz; i > 0; --i) mem[dest + i - 1] = mem[src + i - 1];
                    }
                    ++pc;
                    continue;
                }
                }
            }

            // -- PUSH0 (0x5f) -------------------------------------------------
            if (op == 0x5f)
            {
                if (!consume_gas(GasCost::BASE))
                    return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                ExecStatus s = stack.push(uint256::zero());
                if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                ++pc;
                continue;
            }

            // -- PUSH1..PUSH32 (0x60 - 0x7f) ---------------------------------
            if (op >= 0x60 && op <= 0x7f)
            {
                if (!consume_gas(GasCost::VERYLOW))
                    return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                gpu_u32 num_bytes = op - 0x60 + 1;
                uint256 val = read_push_data(num_bytes);
                ExecStatus s = stack.push(val);
                if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                pc += 1 + num_bytes;
                continue;
            }

            // -- DUP1..DUP16 (0x80 - 0x8f) -----------------------------------
            if (op >= 0x80 && op <= 0x8f)
            {
                if (!consume_gas(GasCost::VERYLOW))
                    return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                gpu_u32 n = op - 0x80 + 1;
                ExecStatus s = stack.dup(n);
                if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                ++pc;
                continue;
            }

            // -- SWAP1..SWAP16 (0x90 - 0x9f) ---------------------------------
            if (op >= 0x90 && op <= 0x9f)
            {
                if (!consume_gas(GasCost::VERYLOW))
                    return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                gpu_u32 n = op - 0x90 + 1;
                ExecStatus s = stack.swap(n);
                if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                ++pc;
                continue;
            }

            // -- LOG0..LOG4 (0xa0 - 0xa4) ------------------------------------
            if (op >= 0xa0 && op <= 0xa4)
            {
                gpu_u32 num_topics = op - 0xa0;

                ExecStatus s;
                uint256 offset_val, size_val;
                s = stack.pop(offset_val); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                s = stack.pop(size_val); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};

                if (offset_val.w[1] | offset_val.w[2] | offset_val.w[3] ||
                    size_val.w[1] | size_val.w[2] | size_val.w[3])
                    return {ExecStatus::InvalidMemAccess, gas_start - gas, gas, 0, refund_counter};

                gpu_u32 data_off = gpu_u32(offset_val.w[0]);
                gpu_u64 data_sz64 = size_val.w[0];
                gpu_u32 data_sz   = gpu_u32(data_sz64);

                gpu_u64 log_gas = GasCost::LOG_BASE + GasCost::LOG_TOPIC * num_topics +
                                  GasCost::LOG_DATA * data_sz64;
                if (!consume_gas(log_gas))
                    return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};

                if (data_sz > 0)
                {
                    gpu_u64 mem_cost = expand_memory(data_off, data_sz, mem);
                    if (mem_cost == ~gpu_u64(0) || !consume_gas(mem_cost))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};
                }

                if (logs && log_count && *log_count < log_capacity)
                {
                    LogEntry& entry = logs[*log_count];
                    entry.num_topics = num_topics;
                    entry.data_offset = data_off;
                    entry.data_size = data_sz;
                    for (gpu_u32 t = 0; t < num_topics; ++t)
                    {
                        s = stack.pop(entry.topics[t]);
                        if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                    }
                    ++(*log_count);
                }
                else
                {
                    uint256 dummy;
                    for (gpu_u32 t = 0; t < num_topics; ++t)
                    {
                        s = stack.pop(dummy);
                        if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
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
                s = stack.pop(offset_val); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                s = stack.pop(size_val); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};

                if (offset_val.w[1] | offset_val.w[2] | offset_val.w[3] ||
                    size_val.w[1] | size_val.w[2] | size_val.w[3])
                    return {ExecStatus::InvalidMemAccess, gas_start - gas, gas, 0, refund_counter};

                gpu_u32 off = gpu_u32(offset_val.w[0]);
                gpu_u32 sz  = gpu_u32(size_val.w[0]);

                if (sz > MAX_OUTPUT)
                    return {ExecStatus::InvalidMemAccess, gas_start - gas, gas, 0, refund_counter};

                if (sz > 0)
                {
                    gpu_u64 mem_cost = expand_memory(off, sz, mem);
                    if (mem_cost == ~gpu_u64(0) || !consume_gas(mem_cost))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};

                    for (gpu_u32 i = 0; i < sz; ++i)
                        output[i] = mem[off + i];
                    return {ExecStatus::Return, gas_start - gas, gas, sz, refund_counter};
                }
                return {ExecStatus::Return, gas_start - gas, gas, 0, refund_counter};
            }

            // -- REVERT (0xfd) ------------------------------------------------
            if (op == 0xfd)
            {
                ExecStatus s;
                uint256 offset_val, size_val;
                s = stack.pop(offset_val); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};
                s = stack.pop(size_val); if (s != ExecStatus::Ok) return {s, gas_start - gas, gas, 0, refund_counter};

                if (offset_val.w[1] | offset_val.w[2] | offset_val.w[3] ||
                    size_val.w[1] | size_val.w[2] | size_val.w[3])
                    return {ExecStatus::InvalidMemAccess, gas_start - gas, gas, 0, refund_counter};

                gpu_u32 off = gpu_u32(offset_val.w[0]);
                gpu_u32 sz  = gpu_u32(size_val.w[0]);

                if (sz > MAX_OUTPUT)
                    return {ExecStatus::InvalidMemAccess, gas_start - gas, gas, 0, refund_counter};

                if (sz > 0)
                {
                    gpu_u64 mem_cost = expand_memory(off, sz, mem);
                    if (mem_cost == ~gpu_u64(0) || !consume_gas(mem_cost))
                        return {ExecStatus::OutOfGas, gas_start, 0, 0, refund_counter};

                    for (gpu_u32 i = 0; i < sz; ++i)
                        output[i] = mem[off + i];
                    return {ExecStatus::Revert, gas_start - gas, gas, sz, refund_counter};
                }
                return {ExecStatus::Revert, gas_start - gas, gas, 0, refund_counter};
            }

            // -- INVALID (0xfe) -----------------------------------------------
            if (op == 0xfe)
            {
                gas = 0;
                return {ExecStatus::InvalidOpcode, gas_start, 0, 0, refund_counter};
            }

            // -- CALL family -- routed to evmone CPU --------------------------
            if (op == 0xf0 || op == 0xf1 || op == 0xf2 || op == 0xf4 ||
                op == 0xf5 || op == 0xfa || op == 0xff)
            {
                return {ExecStatus::CallNotSupported, gas_start - gas, gas, 0, refund_counter};
            }

            // -- Undefined opcode ---------------------------------------------
            // Matches Metal's ERRA(): consume all remaining gas. Same
            // behaviour as 0xFE INVALID — per Yellow Paper, undefined
            // opcodes halt with all gas consumed.
            gas = 0;
            return {ExecStatus::InvalidOpcode, gas_start, 0, 0, refund_counter};

        }  // while (pc < code_size)

        // Fell off the end of code -> implicit STOP.
        return {ExecStatus::Stop, gas_start - gas, gas, 0, refund_counter};
    }
};

}  // namespace evm::gpu::kernel
