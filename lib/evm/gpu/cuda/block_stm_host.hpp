// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file block_stm_host.hpp
/// CUDA host interface for GPU Block-STM execution.
///
/// Mirrors the API of metal/block_stm_host.hpp so the dispatcher can swap
/// backends transparently. The CUDA backend dispatches the same
/// optimistic-concurrency-control loop on an NVIDIA GPU via block_stm.cu.

#pragma once

#include "../gpu_dispatch.hpp"  // Transaction, BlockResult

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace evm::gpu::cuda
{

// -- GPU-side struct mirrors (must match block_stm.cu byte-for-byte) ----------

static constexpr uint32_t MAX_TXS            = 4096;
static constexpr uint32_t MAX_READS_PER_TX   = 64;
static constexpr uint32_t MAX_WRITES_PER_TX  = 64;
static constexpr uint32_t MV_TABLE_SIZE      = 65536;
static constexpr uint32_t VERSION_BASE_STATE = 0xFFFFFFFF;

struct GpuTransaction
{
    uint8_t  from[20];
    uint8_t  to[20];
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

struct GpuReadSetEntry
{
    uint8_t  address[20];
    uint32_t _pad;
    uint8_t  slot[32];
    uint32_t read_tx_index;
    uint32_t read_incarnation;
};

struct GpuWriteSetEntry
{
    uint8_t  address[20];
    uint32_t _pad;
    uint8_t  slot[32];
    uint8_t  value[32];
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

/// Returns true when the CUDA backend is available at runtime.
bool block_stm_cuda_available();

/// CUDA-accelerated Block-STM execution engine.
class BlockStmGpu
{
public:
    virtual ~BlockStmGpu() = default;

    /// Returns nullptr if CUDA is unavailable (no driver / no device).
    static std::unique_ptr<BlockStmGpu> create();

    virtual BlockResult execute_block(
        std::span<const Transaction> txs,
        std::span<const GpuAccountState> base_state) = 0;

    virtual const char* device_name() const = 0;

    uint32_t max_txs() const { return MAX_TXS; }

protected:
    BlockStmGpu() = default;
    BlockStmGpu(const BlockStmGpu&) = delete;
    BlockStmGpu& operator=(const BlockStmGpu&) = delete;
};

}  // namespace evm::gpu::cuda
