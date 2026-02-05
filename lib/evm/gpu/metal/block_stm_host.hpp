// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file block_stm_host.hpp
/// C++ host-side interface for Metal-accelerated Block-STM execution.
///
/// Dispatches the entire Block-STM optimistic concurrency loop to GPU.
/// The GPU handles scheduling, execution, validation, and conflict resolution
/// with zero CPU round-trips during the main loop.
///
/// Usage:
///   auto engine = BlockStmGpu::create();
///   auto result = engine->execute_block(txs, base_state);

#pragma once

#include "../gpu_dispatch.hpp"  // Transaction, BlockResult
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace evm::gpu::metal
{

// -- GPU-side struct mirrors (must match block_stm.metal exactly) -------------

static constexpr uint32_t MAX_TXS           = 4096;
static constexpr uint32_t MAX_READS_PER_TX  = 64;
static constexpr uint32_t MAX_WRITES_PER_TX = 64;
static constexpr uint32_t MV_TABLE_SIZE     = 65536;
static constexpr uint32_t VERSION_BASE_STATE = 0xFFFFFFFF;

struct GpuTransaction
{
    uint8_t from[20];
    uint8_t to[20];
    uint64_t gas_limit;
    uint64_t value;
    uint64_t nonce;
    uint64_t gas_price;
    uint32_t calldata_offset;
    uint32_t calldata_size;
};

struct GpuAccountState
{
    uint8_t  address[20];
    uint32_t _pad;
    uint64_t nonce;
    uint64_t balance;
    uint8_t  code_hash[32];
    uint32_t code_size;
    uint32_t _pad2;
};

struct GpuMvEntry
{
    uint32_t tx_index;
    uint32_t incarnation;
    uint8_t  address[20];
    uint32_t _pad;
    uint8_t  slot[32];
    uint8_t  value[32];
    uint32_t is_estimate;
    uint32_t _pad2;
};

struct GpuTxState
{
    uint32_t incarnation;
    uint32_t validated;
    uint32_t executed;
    uint32_t status;
    uint64_t gas_used;
    uint32_t read_count;
    uint32_t write_count;
};

struct GpuBlockStmResult
{
    uint64_t gas_used;
    uint32_t status;
    uint32_t incarnation;
};

struct GpuBlockStmParams
{
    uint32_t num_txs;
    uint32_t max_iterations;
};

// -- Public interface ---------------------------------------------------------

/// GPU-accelerated Block-STM execution engine using Apple Metal.
///
/// Creates Metal buffers, compiles the block_stm kernel, and dispatches
/// the full execute/validate/re-execute loop on GPU.
class BlockStmGpu
{
public:
    virtual ~BlockStmGpu() = default;

    /// Create a BlockStmGpu engine. Returns nullptr if Metal is unavailable.
    static std::unique_ptr<BlockStmGpu> create();

    /// Execute a block of transactions using GPU Block-STM.
    ///
    /// @param txs          Transactions to execute.
    /// @param base_state   Account states (nonce, balance) for referenced accounts.
    /// @return              Block execution result with gas, conflicts, timing.
    virtual BlockResult execute_block(
        std::span<const Transaction> txs,
        std::span<const GpuAccountState> base_state) = 0;

    /// Get the Metal device name (diagnostics).
    virtual const char* device_name() const = 0;

    /// Get max transactions per block for this engine.
    uint32_t max_txs() const { return MAX_TXS; }

protected:
    BlockStmGpu() = default;
    BlockStmGpu(const BlockStmGpu&) = delete;
    BlockStmGpu& operator=(const BlockStmGpu&) = delete;
};

}  // namespace evm::gpu::metal
