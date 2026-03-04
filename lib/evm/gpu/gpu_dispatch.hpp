// Copyright (C) 2026, The evmone Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file gpu_dispatch.hpp
/// GPU execution dispatcher for evmone.
///
/// Routes EVM block execution to either:
/// - CPU sequential (baseline evmone)
/// - CPU parallel (Block-STM scheduler with evmone workers)
/// - GPU parallel (CUDA/Metal kernels for opcode dispatch)
///
/// The GPU path offloads three categories of work:
/// 1. Opcode interpretation (the EVM interpreter loop)
/// 2. State trie hashing (Keccak-256 on Merkle paths)
/// 3. Precompile operations (ecrecover, bn256, blake2f)

#pragma once

#include <evmc/evmc.h>

#include <cstdint>
#include <vector>
#include <memory>

namespace evm::gpu
{

/// Execution backend selection.
enum class Backend : uint8_t
{
    CPU_Sequential = 0,  ///< Single-threaded evmone (baseline)
    CPU_Parallel = 1,    ///< Block-STM with N worker threads
    GPU_Metal = 2,       ///< Apple Metal compute shaders
    GPU_CUDA = 3,        ///< NVIDIA CUDA kernels
};

/// Configuration for the GPU execution engine.
struct Config
{
    Backend backend = Backend::CPU_Sequential;
    uint32_t num_threads = 0;  ///< 0 = auto-detect (std::thread::hardware_concurrency)
    uint32_t gpu_device = 0;   ///< GPU device index
    bool enable_state_trie_gpu = false;  ///< Offload Keccak-256 trie hashing to GPU
    bool enable_precompile_gpu = false;  ///< Offload precompiles to GPU
    /// EVM revision for the evmone fallback path. The kernel CPU/GPU paths
    /// implement the Cancun opcode set unconditionally — this field only
    /// governs the evmone code path, which must agree with the kernel set
    /// or the same transaction routes to a different result depending on
    /// backend. Default = Cancun, matching Lux mainnet.
    evmc_revision revision = EVMC_CANCUN;
};

/// Per-tx execution status. Mirrors kernel::TxStatus so all backends
/// (CPU sequential, CPU parallel, Metal, CUDA) report the same coding.
///   0 = Stop, 1 = Return, 2 = Revert, 3 = OutOfGas, 4 = Error,
///   5 = CallNotSupported (GPU kernel rejects external CALLs).
enum class TxStatus : uint32_t
{
    Stop             = 0,
    Return           = 1,
    Revert           = 2,
    OutOfGas         = 3,
    Error            = 4,
    CallNotSupported = 5,
};

/// Result of executing a block of transactions.
struct BlockResult
{
    std::vector<uint8_t> state_root;   ///< Post-execution state root (32 bytes)
    std::vector<uint64_t> gas_used;    ///< Gas used per transaction
    /// Per-tx execution status. Populated when `txs[i].code` is non-empty AND
    /// `state == nullptr` (i.e. the dispatch routes through the kernel CPU/GPU
    /// path). Empty otherwise.
    std::vector<TxStatus> status;
    /// Per-tx return-data bytes. Populated alongside `status`.
    std::vector<std::vector<uint8_t>> output;
    uint64_t total_gas = 0;
    double execution_time_ms = 0.0;    ///< Wall-clock execution time
    uint32_t conflicts = 0;            ///< Number of Block-STM conflicts (parallel only)
    uint32_t re_executions = 0;        ///< Number of re-executed transactions
};

/// A transaction in a block (pre-signed, ready for execution).
struct Transaction
{
    std::vector<uint8_t> from;     ///< 20 bytes
    std::vector<uint8_t> to;       ///< 20 bytes (empty for contract creation)
    std::vector<uint8_t> data;     ///< Calldata
    std::vector<uint8_t> code;     ///< EVM bytecode for direct execution on the GPU.
                                    ///< When empty, GPU paths use the scheduler-only
                                    ///< (Block-STM) kernel. When non-empty, GPU paths
                                    ///< dispatch through the parallel opcode interpreter
                                    ///< (kernel::EvmKernelHost on Metal, cuda::EvmKernel
                                    ///< on NVIDIA).
    uint64_t gas_limit = 0;
    uint64_t value = 0;            ///< Value in wei (simplified to uint64 for now)
    uint64_t nonce = 0;
    uint64_t gas_price = 0;
};

/// Execute a block of transactions.
///
/// @param config    Execution configuration (backend, threads, GPU settings)
/// @param txs       Pre-signed transactions to execute
/// @param state     Opaque pointer to the state database
/// @return          Block execution result
BlockResult execute_block(
    const Config& config,
    const std::vector<Transaction>& txs,
    void* state
);

/// Query available backends on this system.
std::vector<Backend> available_backends();

/// Get a human-readable name for a backend.
const char* backend_name(Backend b);

/// Auto-detect the best available backend.
/// Preference order: GPU_Metal > GPU_CUDA > CPU_Parallel > CPU_Sequential.
Backend auto_detect();

/// Set the backend on an existing config. Returns false if the backend
/// is not available on this system.
bool set_backend(Config& config, Backend b);

}  // namespace evm::gpu
