// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file evm_kernel_host.hpp
/// Host-side dispatcher for GPU EVM kernel execution.

#pragma once

#include "evm_interpreter.hpp"
#include "uint256_gpu.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace evm::gpu::kernel {

static constexpr uint32_t HOST_MAX_MEMORY_PER_TX  = 65536;
static constexpr uint32_t HOST_MAX_OUTPUT_PER_TX  = 1024;
static constexpr uint32_t HOST_MAX_STORAGE_PER_TX = 64;
static constexpr uint32_t HOST_MAX_LOGS_PER_TX    = 16;

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
};

struct TxOutput
{
    uint32_t status;
    uint64_t gas_used;
    int64_t  gas_refund;  // signed: EIP-2200 allows refund subtraction
    uint32_t output_size;
};

struct StorageEntry
{
    uint256 key;
    uint256 value;
};

/// Block-level context — same for every tx in a block. ABI v3 addition.
struct BlockContext
{
    uint256  origin;
    uint64_t gas_price;
    uint64_t timestamp;
    uint64_t number;
    uint256  prevrandao;
    uint64_t gas_limit;
    uint64_t chain_id;
    uint64_t base_fee;
    uint64_t blob_base_fee;
    uint256  coinbase;
    uint8_t  blob_hashes[8][32];
    uint32_t num_blob_hashes;
    uint32_t _pad0;
};

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
    int64_t  gas_refund = 0;  // EIP-2200 raw refund counter; dispatcher applies EIP-3529 cap
    std::vector<uint8_t> output;
    std::vector<HostLog> logs;
};

class EvmKernelHost
{
public:
    virtual ~EvmKernelHost() = default;

    static std::unique_ptr<EvmKernelHost> create();

    virtual std::vector<TxResult> execute(std::span<const HostTransaction> txs) = 0;

    /// Execute with explicit block context.
    virtual std::vector<TxResult> execute(std::span<const HostTransaction> txs,
                                          const BlockContext& ctx) = 0;

    virtual std::vector<TxResult> execute_v2(std::span<const HostTransaction> txs) = 0;

    virtual bool has_v2() const = 0;

    virtual const char* device_name() const = 0;

protected:
    EvmKernelHost() = default;
    EvmKernelHost(const EvmKernelHost&) = delete;
    EvmKernelHost& operator=(const EvmKernelHost&) = delete;
};

inline TxResult execute_cpu(const HostTransaction& tx)
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

    std::vector<uint8_t> memory(HOST_MAX_MEMORY_PER_TX, 0);
    std::vector<uint8_t> output(HOST_MAX_OUTPUT_PER_TX, 0);
    std::vector<uint256> storage_keys(HOST_MAX_STORAGE_PER_TX);
    std::vector<uint256> storage_values(HOST_MAX_STORAGE_PER_TX);
    std::vector<uint256> orig_keys(HOST_MAX_STORAGE_PER_TX);
    std::vector<uint256> orig_values(HOST_MAX_STORAGE_PER_TX);
    uint32_t storage_count = 0;
    uint32_t orig_count = 0;
    std::vector<LogEntry> log_entries(MAX_LOGS);
    uint32_t log_count = 0;

    interp.storage_keys = storage_keys.data();
    interp.storage_values = storage_values.data();
    interp.storage_count = &storage_count;
    interp.storage_capacity = HOST_MAX_STORAGE_PER_TX;
    interp.orig_keys = orig_keys.data();
    interp.orig_values = orig_values.data();
    interp.orig_count = &orig_count;
    interp.logs = log_entries.data();
    interp.log_count = &log_count;
    interp.log_capacity = MAX_LOGS;

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

    return r;
}

}  // namespace evm::gpu::kernel
