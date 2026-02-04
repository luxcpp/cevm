// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file evm_kernel_host.hpp
/// Host-side dispatcher for GPU EVM kernel execution.
///
/// Dispatches N transactions to the Metal GPU. Transactions containing
/// CALL/CREATE opcodes are detected by the GPU kernel (returns status
/// CallNotSupported) and fall back to CPU evmone execution.
///
/// Usage:
///   auto engine = EvmKernelHost::create();
///   auto results = engine->execute(transactions);

#pragma once

#include "evm_interpreter.hpp"
#include "uint256_gpu.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace evm::gpu::kernel {

// -- Constants matching the Metal kernel --------------------------------------

static constexpr uint32_t HOST_MAX_MEMORY_PER_TX  = 65536;
static constexpr uint32_t HOST_MAX_OUTPUT_PER_TX   = 1024;
static constexpr uint32_t HOST_MAX_STORAGE_PER_TX  = 64;

// -- GPU buffer descriptors (must match Metal structs exactly) ----------------

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
    uint32_t status;      // 0=stop, 1=return, 2=revert, 3=oog, 4=error, 5=call_not_supported
    uint64_t gas_used;
    uint32_t output_size;
};

struct StorageEntry
{
    uint256 key;
    uint256 value;
};

// -- Transaction input for the host API ---------------------------------------

struct HostTransaction
{
    std::vector<uint8_t> code;      // EVM bytecode
    std::vector<uint8_t> calldata;  // Transaction calldata
    uint64_t gas_limit = 0;
    uint256  caller;
    uint256  address;
    uint256  value;
};

// -- Execution result per transaction -----------------------------------------

enum class TxStatus : uint32_t
{
    Stop             = 0,
    Return           = 1,
    Revert           = 2,
    OutOfGas         = 3,
    Error            = 4,
    CallNotSupported = 5,  // needs CPU fallback
};

struct TxResult
{
    TxStatus status;
    uint64_t gas_used;
    std::vector<uint8_t> output;
};

// -- Kernel host interface ----------------------------------------------------

class EvmKernelHost
{
public:
    virtual ~EvmKernelHost() = default;

    /// Create a Metal-backed kernel host. Returns nullptr if Metal is unavailable.
    static std::unique_ptr<EvmKernelHost> create();

    /// Execute a batch of transactions on the GPU.
    ///
    /// Transactions that use CALL/CREATE will have status == CallNotSupported.
    /// The caller is responsible for re-executing those on the CPU.
    ///
    /// @param txs  Transactions to execute.
    /// @return     Per-transaction results.
    virtual std::vector<TxResult> execute(std::span<const HostTransaction> txs) = 0;

    /// Get the GPU device name.
    virtual const char* device_name() const = 0;

protected:
    EvmKernelHost() = default;
    EvmKernelHost(const EvmKernelHost&) = delete;
    EvmKernelHost& operator=(const EvmKernelHost&) = delete;
};

// -- CPU reference interpreter ------------------------------------------------
//
// Runs the same interpreter as the GPU kernel, but on CPU. Useful for
// verification and fallback.

inline TxResult execute_cpu(const HostTransaction& tx)
{
    // Allocate interpreter state.
    EvmInterpreter interp{};
    interp.code = tx.code.data();
    interp.code_size = static_cast<uint32_t>(tx.code.size());
    interp.calldata = tx.calldata.data();
    interp.calldata_size = static_cast<uint32_t>(tx.calldata.size());
    interp.gas = tx.gas_limit;
    interp.caller = tx.caller;
    interp.address = tx.address;
    interp.value = tx.value;

    // Allocate memory, storage, logs.
    std::vector<uint8_t> memory(HOST_MAX_MEMORY_PER_TX, 0);
    std::vector<uint8_t> output(HOST_MAX_OUTPUT_PER_TX, 0);
    std::vector<uint256> storage_keys(HOST_MAX_STORAGE_PER_TX);
    std::vector<uint256> storage_values(HOST_MAX_STORAGE_PER_TX);
    uint32_t storage_count = 0;
    std::vector<LogEntry> log_entries(MAX_LOGS);
    uint32_t log_count = 0;

    interp.storage_keys = storage_keys.data();
    interp.storage_values = storage_values.data();
    interp.storage_count = &storage_count;
    interp.storage_capacity = HOST_MAX_STORAGE_PER_TX;
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
    r.gas_used = result.gas_used;
    if (result.output_size > 0)
        r.output.assign(output.data(), output.data() + result.output_size);

    return r;
}

}  // namespace evm::gpu::kernel
