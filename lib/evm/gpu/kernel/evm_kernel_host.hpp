// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file evm_kernel_host.hpp
/// Host-side dispatcher for GPU EVM kernel execution.

#pragma once

#include "evm_interpreter.hpp"
#include "uint256_gpu.hpp"

#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <vector>

namespace evm::gpu::kernel {

static constexpr uint32_t HOST_MAX_MEMORY_PER_TX  = 65536;
static constexpr uint32_t HOST_MAX_OUTPUT_PER_TX  = 1024;
static constexpr uint32_t HOST_MAX_STORAGE_PER_TX = 64;
static constexpr uint32_t HOST_MAX_LOGS_PER_TX    = 16;

/// On-wire transaction input for the kernel. Layout MUST match the device
/// structs in evm_kernel.metal and evm_kernel.cu byte-for-byte.
///
/// EIP-2929 caller-supplied warm sets are packed into the same shared
/// `blob` buffer that holds code+calldata. The kernel reads them once at
/// tx startup and seeds its per-thread warm sets. Layout in the blob:
///   warm_addr_offset .. +20*warm_addr_count  : 20-byte addresses
///   warm_slot_offset .. +52*warm_slot_count  : (20-byte addr | 32-byte slot) pairs
struct TxInput
{
    uint32_t code_offset;
    uint32_t code_size;
    uint32_t calldata_offset;
    uint32_t calldata_size;
    uint64_t gas_limit;
    uint256  caller;
    uint256  address;
    uint256  value;
    uint32_t warm_addr_offset;
    uint32_t warm_addr_count;
    uint32_t warm_slot_offset;
    uint32_t warm_slot_count;
};

struct TxOutput
{
    uint32_t status;
    uint64_t gas_used;
    // Signed: SSTORE may transiently subtract refund credit. The host
    // dispatcher floors at 0 and applies the EIP-3529 cap (gas_used / 5).
    int64_t  gas_refund;
    uint32_t output_size;
};

struct StorageEntry
{
    uint256 key;
    uint256 value;
};

// BlockContext is declared in evm_interpreter.hpp (already #included above)
// so callers of evm_kernel_host.hpp pick it up via this same namespace.

/// Per-tx log entry written by LOG0..LOG4.
struct GpuLogEntry
{
    uint256  topics[4];
    uint32_t num_topics;
    uint32_t data_offset;
    uint32_t data_size;
    uint32_t _pad0;
};

struct HostTransaction
{
    std::vector<uint8_t> code;
    std::vector<uint8_t> calldata;
    uint64_t gas_limit = 0;
    uint256  caller;
    uint256  address;
    uint256  value;

    /// EIP-2929 / EIP-2930 caller-supplied warm sets. Flat layouts:
    ///   warm_addresses:    [20-byte addr][20-byte addr]...
    ///   warm_storage_keys: [20-byte addr | 32-byte slot]...
    /// Empty by default; `gpu_dispatch.cpp` fills them from `Config`.
    std::vector<uint8_t> warm_addresses;
    std::vector<uint8_t> warm_storage_keys;
};

enum class TxStatus : uint32_t
{
    Stop             = 0,
    Return           = 1,
    Revert           = 2,
    OutOfGas         = 3,
    Error            = 4,
    CallNotSupported = 5,
};

struct HostLog
{
    std::vector<uint256> topics;
    std::vector<uint8_t> data;
};

struct TxResult
{
    TxStatus status;
    uint64_t gas_used;
    // Signed EIP-2200 refund counter. The kernel emits the raw signed
    // value; the dispatcher floors at 0 and applies the EIP-3529 cap
    // (max refund = gas_used / 5).
    int64_t  gas_refund = 0;
    std::vector<uint8_t> output;
    std::vector<HostLog> logs;
};

/// Future returned by EvmKernelHost::execute_async(). The kernel host has
/// already enqueued the GPU work and registered a completion handler; the
/// caller can do other work and call await() when ready for the results.
///
/// The future blocks the calling thread inside await() until the GPU
/// completion handler fires. ready() is a non-blocking poll. Each future
/// owns its own staging buffers and command buffer so the dispatcher can
/// pipeline batch N+1's host prep with batch N's GPU run.
class TxResultFuture
{
public:
    virtual ~TxResultFuture() = default;

    /// Block until the GPU completes this batch and return per-tx results.
    /// May be called only once; calling twice throws.
    virtual std::vector<TxResult> await() = 0;

    /// Non-blocking check: true if the GPU has signalled completion. The
    /// caller must still call await() to retrieve results.
    virtual bool ready() const = 0;
};

class EvmKernelHost
{
public:
    virtual ~EvmKernelHost() = default;

    static std::unique_ptr<EvmKernelHost> create();

    /// Synchronous execute: enqueues work, blocks until GPU completes, then
    /// returns the per-tx results. Implemented as a thin wrapper around
    /// execute_async()→await() for backward compatibility.
    virtual std::vector<TxResult> execute(std::span<const HostTransaction> txs) = 0;

    /// Execute with explicit block context.
    virtual std::vector<TxResult> execute(std::span<const HostTransaction> txs,
                                          const BlockContext& ctx) = 0;

    virtual std::vector<TxResult> execute_v2(std::span<const HostTransaction> txs) = 0;

    /// Asynchronous execute. Enqueues the kernel on the GPU and returns a
    /// future immediately so the caller can interleave host work (preparing
    /// the next batch, building results from the previous batch, etc.) with
    /// the in-flight GPU run. The dispatcher MUST call await() on the
    /// returned future before the EvmKernelHost is destroyed.
    ///
    /// Concurrency: the host's internal buffer cache is locked from
    /// enqueue-time through await(), so multiple in-flight futures from the
    /// same EvmKernelHost serialize through the cache. Use multiple
    /// EvmKernelHost instances for true overlap on the same device.
    virtual std::unique_ptr<TxResultFuture> execute_async(
        std::span<const HostTransaction> txs,
        const BlockContext& ctx) = 0;

    /// Convenience overload with an empty BlockContext.
    virtual std::unique_ptr<TxResultFuture> execute_async(
        std::span<const HostTransaction> txs) = 0;

    virtual bool has_v2() const = 0;

    virtual const char* device_name() const = 0;

protected:
    EvmKernelHost() = default;
    EvmKernelHost(const EvmKernelHost&) = delete;
    EvmKernelHost& operator=(const EvmKernelHost&) = delete;
};

/// Optional per-execution context: block fields (chain id, timestamp, etc.)
/// the interpreter reads via ORIGIN/COINBASE/TIMESTAMP/.../BLOBHASH. Default-
/// constructed = all zeros, which matches the GPU "no host" behaviour the
/// parity tests assert on.
struct CpuExecOptions
{
    BlockContext block_ctx{};
};

inline TxResult execute_cpu(const HostTransaction& tx,
                            const CpuExecOptions& opts = {})
{
    EvmInterpreter interp{};
    interp.code = tx.code.data();
    interp.code_size = static_cast<uint32_t>(tx.code.size());
    interp.calldata = tx.calldata.data();
    interp.calldata_size = static_cast<uint32_t>(tx.calldata.size());
    interp.gas = tx.gas_limit;
    interp.caller = tx.caller;
    interp.address = tx.address;
    interp.value = tx.value;
    interp.block_ctx = &opts.block_ctx;

    std::vector<uint8_t> memory(HOST_MAX_MEMORY_PER_TX, 0);
    std::vector<uint8_t> output(HOST_MAX_OUTPUT_PER_TX, 0);
    std::vector<uint256> storage_keys(HOST_MAX_STORAGE_PER_TX);
    std::vector<uint256> storage_values(HOST_MAX_STORAGE_PER_TX);
    std::vector<uint256> orig_keys(HOST_MAX_STORAGE_PER_TX);
    std::vector<uint256> orig_values(HOST_MAX_STORAGE_PER_TX);
    std::vector<uint256> trans_keys(HOST_MAX_STORAGE_PER_TX);
    std::vector<uint256> trans_values(HOST_MAX_STORAGE_PER_TX);
    uint32_t storage_count = 0;
    uint32_t orig_count = 0;
    uint32_t trans_count = 0;
    std::vector<LogEntry> log_entries(MAX_LOGS);
    uint32_t log_count = 0;

    interp.storage_keys = storage_keys.data();
    interp.storage_values = storage_values.data();
    interp.storage_count = &storage_count;
    interp.storage_capacity = HOST_MAX_STORAGE_PER_TX;
    interp.orig_keys = orig_keys.data();
    interp.orig_values = orig_values.data();
    interp.orig_count = &orig_count;
    interp.transient_keys = trans_keys.data();
    interp.transient_values = trans_values.data();
    interp.transient_count = &trans_count;
    interp.logs = log_entries.data();
    interp.log_count = &log_count;
    interp.log_capacity = MAX_LOGS;

    // EIP-2929: per-tx warm sets. Pre-warming for caller / recipient /
    // precompiles happens inside interp.execute(). Here we just append any
    // caller-supplied entries (EIP-2930 / Config-level) so subsequent
    // accesses pay warm.
    std::vector<uint256> warm_addrs(MAX_WARM_ADDRESSES);
    std::vector<uint256> warm_slot_addrs(MAX_WARM_SLOTS);
    std::vector<uint256> warm_slot_keys(MAX_WARM_SLOTS);
    uint32_t warm_addr_count = 0;
    uint32_t warm_slot_count = 0;
    interp.warm_addrs = warm_addrs.data();
    interp.warm_addr_count = &warm_addr_count;
    interp.warm_addr_capacity = MAX_WARM_ADDRESSES;
    interp.warm_slot_addrs = warm_slot_addrs.data();
    interp.warm_slot_keys = warm_slot_keys.data();
    interp.warm_slot_count = &warm_slot_count;
    interp.warm_slot_capacity = MAX_WARM_SLOTS;

    // 20 BE address bytes → big-int uint256 matching the canonical
    // Ethereum representation. PUSH-derived addresses on the stack use
    // this same layout, so warm-set comparisons work.
    auto unpack_addr_be = [](const uint8_t* src) {
        uint256 v = uint256::zero();
        for (int b = 0; b < 20; ++b)
        {
            int pos_from_right = 19 - b;
            v.w[pos_from_right / 8] |= static_cast<uint64_t>(src[b]) << ((pos_from_right % 8) * 8);
        }
        return v;
    };
    {
        const uint8_t* a = tx.warm_addresses.data();
        size_t n = tx.warm_addresses.size() / 20;
        for (size_t i = 0; i < n && warm_addr_count < MAX_WARM_ADDRESSES; ++i)
            warm_addrs[warm_addr_count++] = unpack_addr_be(a + i * 20);
    }
    {
        const uint8_t* p = tx.warm_storage_keys.data();
        size_t n = tx.warm_storage_keys.size() / 52;
        for (size_t i = 0; i < n && warm_slot_count < MAX_WARM_SLOTS; ++i)
        {
            uint256 a = unpack_addr_be(p + i * 52);
            uint256 k = uint256::zero();
            // Slot key is stored big-endian on the wire; load into the
            // limb layout used by the kernel uint256 (w[0]=low) by
            // reversing the byte order.
            const uint8_t* kb = p + i * 52 + 20;
            for (int b = 0; b < 32; ++b)
            {
                int pos_from_right = 31 - b;
                int limb = pos_from_right / 8;
                int shift = (pos_from_right % 8) * 8;
                k.w[limb] |= static_cast<uint64_t>(kb[b]) << shift;
            }
            warm_slot_addrs[warm_slot_count] = a;
            warm_slot_keys[warm_slot_count] = k;
            ++warm_slot_count;
        }
    }

    auto result = interp.execute(memory.data(), output.data());

    TxResult r;
    switch (result.status)
    {
    case ExecStatus::Stop:          r.status = TxStatus::Stop; break;
    case ExecStatus::Return:        r.status = TxStatus::Return; break;
    case ExecStatus::Revert:        r.status = TxStatus::Revert; break;
    case ExecStatus::OutOfGas:      r.status = TxStatus::OutOfGas; break;
    case ExecStatus::CallNotSupported: r.status = TxStatus::CallNotSupported; break;
    default:                        r.status = TxStatus::Error; break;
    }
    r.gas_used   = result.gas_used;
    r.gas_refund = result.gas_refund;
    if (result.output_size > 0)
        r.output.assign(output.data(), output.data() + result.output_size);

    // Resolve LOG entries against the memory image *after* execution. The
    // kernel records data_offset+data_size only; we copy the bytes here
    // (the GPU dispatcher does the analogous step out of mem_pool).
    for (uint32_t i = 0; i < log_count && i < log_entries.size(); ++i)
    {
        const auto& e = log_entries[i];
        HostLog hl;
        hl.topics.assign(e.topics, e.topics + e.num_topics);
        if (e.data_size > 0 && e.data_offset + e.data_size <= memory.size())
            hl.data.assign(memory.data() + e.data_offset,
                           memory.data() + e.data_offset + e.data_size);
        r.logs.push_back(std::move(hl));
    }

    return r;
}

}  // namespace evm::gpu::kernel
