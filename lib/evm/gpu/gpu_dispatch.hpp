// Copyright (C) 2026, The cevm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file gpu_dispatch.hpp
/// GPU execution dispatcher for cevm.
///
/// Routes EVM block execution to either:
/// - CPU sequential (baseline cevm)
/// - CPU parallel (Block-STM scheduler with cevm workers)
/// - GPU parallel (CUDA/Metal kernels for opcode dispatch)
///
/// The GPU path offloads three categories of work:
/// 1. Opcode interpretation (the EVM interpreter loop)
/// 2. State trie hashing (Keccak-256 on Merkle paths)
/// 3. Precompile operations (ecrecover, bn256, blake2f)

#pragma once

#include <evmc/evmc.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace evm::gpu
{

/// Execution backend selection.
enum class Backend : uint8_t
{
    CPU_Sequential = 0,  ///< Single-threaded cevm (baseline)
    CPU_Parallel = 1,    ///< Block-STM with N worker threads
    GPU_Metal = 2,       ///< Apple Metal compute shaders
    GPU_CUDA = 3,        ///< NVIDIA CUDA kernels
};

/// Block-level context shared by every tx in a block. Field layout matches
/// kernel::BlockContext (lib/evm/gpu/kernel/evm_kernel_host.hpp) and the
/// Metal kernel's BlockContext (evm_kernel.metal) byte-for-byte so the
/// dispatcher can forward this struct to either backend without translation.
///
/// All-zero is the documented "no context provided" default — every backend
/// treats a zero ctx as it did before BlockContext existed (TIMESTAMP, NUMBER,
/// etc. resolve to zero, CHAINID resolves to zero, GASPRICE resolves to the
/// per-tx gas_price). Production callers MUST populate this from the block
/// header before invoking execute_block.
struct BlockContext
{
    uint8_t  origin[20];                ///< tx originator (msg.sender at top frame)
    uint64_t gas_price;                 ///< effective gas price for the tx
    uint64_t timestamp;                 ///< block timestamp (TIMESTAMP)
    uint64_t number;                    ///< block number (NUMBER)
    uint8_t  prevrandao[32];            ///< PREVRANDAO / DIFFICULTY (post-Merge)
    uint64_t gas_limit;                 ///< block gas limit (GASLIMIT)
    uint64_t chain_id;                  ///< chain ID (CHAINID)
    uint64_t base_fee;                  ///< EIP-1559 base fee (BASEFEE)
    uint64_t blob_base_fee;             ///< EIP-4844 blob base fee (BLOBBASEFEE)
    uint8_t  coinbase[20];              ///< miner address (COINBASE)
    uint8_t  blob_hashes[8][32];        ///< EIP-4844 versioned hashes
    uint32_t num_blob_hashes;           ///< 0..8
};

/// Configuration for the GPU execution engine.
///
/// New fields are placed at the bottom and default to values that preserve
/// the dispatcher's pre-existing behaviour: callers that compile against an
/// older Config layout still get the same routing decisions they did before
/// these flags were introduced. Anything that changes EVM consensus (gas
/// accounting, parity guarantees, gas-estimation shortcuts) is opt-in.
struct Config
{
    Backend backend = Backend::CPU_Sequential;
    uint32_t num_threads = 0;  ///< 0 = auto-detect (std::thread::hardware_concurrency)
    uint32_t gpu_device = 0;   ///< GPU device index
    bool enable_state_trie_gpu = false;  ///< Offload Keccak-256 trie hashing to GPU
    bool enable_precompile_gpu = false;  ///< Offload precompiles to GPU

    /// Gas-estimation only path: if set, the "no host + no bytecode" short
    /// circuit returns gas_used=gas_limit per tx. Off by default — wrong
    /// for consensus, only useful for callers explicitly estimating gas.
    bool gas_estimation_mode = false;

    /// UNSAFE — skip per-opcode gas decrement in the GPU kernel. Violates
    /// EVM consensus. Refused unless fast_value_transfer_acknowledged is
    /// also set. Logs to stderr when enabled.
    bool fast_value_transfer = false;
    bool fast_value_transfer_acknowledged = false;

    /// Disable 4-way parity assertions when running raw-speed benchmarks.
    bool skip_parity_check = false;

    /// Pre-warmed access list (EIP-2929). Flat byte vectors:
    ///   warm_addresses:    [20-byte addr][20-byte addr]...
    ///   warm_storage_keys: [20-byte addr | 32-byte slot]...
    std::vector<uint8_t> warm_addresses;
    std::vector<uint8_t> warm_storage_keys;

    /// EVM revision for the cevm fallback path. Kernel paths implement
    /// Cancun unconditionally; this field aligns the cevm path so the
    /// same tx produces the same result regardless of backend.
    evmc_revision revision = EVMC_CANCUN;

    /// Block-level context (chain id, timestamp, base fee, etc.) forwarded
    /// to every backend. Zero-initialised by default — that matches the
    /// dispatcher's pre-v0.26 behaviour where chain-id/timestamp/etc. all
    /// resolved to zero. Populated by callers that own a real block header.
    BlockContext block_context = {};
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

    /// Number of txs that the GPU kernel rejected with CallNotSupported and
    /// the dispatcher re-ran on CPU cevm. Zero means the GPU executed every
    /// tx end-to-end. Non-zero is informational, not an error.
    uint32_t gpu_fallback_count = 0;

    /// Set when the dispatcher itself bails out (no backend can run the
    /// requested combination of flags + inputs). When non-empty, every entry
    /// of `status` is `Error` and `gas_used` is unset / zeroed.
    std::string error_message;
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
