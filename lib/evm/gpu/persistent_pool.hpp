// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file persistent_pool.hpp
/// Pre-allocated Metal buffer pool that persists across blocks.
///
/// On Apple Silicon (M1/M2/M3/M4), MTLResourceStorageModeShared means
/// the buffer lives in unified memory, accessible by both CPU and GPU
/// with zero copy. The GPU hash table IS the state database.

#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __OBJC__
#import <Metal/Metal.h>
#else
// Forward-declare for pure C++ headers. The .mm implementation sees the real types.
using MTLBuffer_id = void*;
using MTLDevice_id = void*;
#endif

namespace evm::gpu
{

/// Default buffer sizes for the persistent pool.
struct PoolSizes
{
    size_t account_table  = 1ULL << 30;  // 1 GB for account state
    size_t storage_table  = 4ULL << 30;  // 4 GB for storage slots
    size_t tx_input       = 256ULL << 20; // 256 MB for current block txs
    size_t tx_output      = 256ULL << 20; // 256 MB for execution results
    size_t evm_memory     = 1ULL << 30;  // 1 GB for EVM memory per-tx
    size_t mv_memory      = 512ULL << 20; // 512 MB for Block-STM versioned state
};

/// Opaque handle to a persistent Metal buffer.
/// The actual MTLBuffer pointer is stored here. On unified memory architectures,
/// the CPU can read/write the contents pointer directly.
struct PersistentBuffer
{
    void* metal_buffer = nullptr;  // id<MTLBuffer> stored as void*
    void* contents = nullptr;      // mapped CPU pointer (shared mode)
    size_t size = 0;
};

/// Pre-allocated buffer pool for GPU state.
///
/// All buffers use MTLResourceStorageModeShared for zero-copy on Apple Silicon.
/// Created once at node startup, reused across all block executions.
class PersistentBufferPool
{
public:
    PersistentBufferPool() = default;
    ~PersistentBufferPool();

    PersistentBufferPool(const PersistentBufferPool&) = delete;
    PersistentBufferPool& operator=(const PersistentBufferPool&) = delete;

    PersistentBufferPool(PersistentBufferPool&& other) noexcept;
    PersistentBufferPool& operator=(PersistentBufferPool&& other) noexcept;

    /// Initialize the pool with a Metal device. Returns false on failure.
    /// @param device  MTLDevice pointer (as void* for C++ header compatibility).
    /// @param sizes   Buffer sizes to allocate.
    bool init(void* device, const PoolSizes& sizes = {});

    /// Check if the pool is initialized.
    [[nodiscard]] bool valid() const noexcept { return device_ != nullptr; }

    // -- Buffer accessors (return the Metal buffer as void*) ---

    [[nodiscard]] PersistentBuffer& account_table() noexcept { return account_table_; }
    [[nodiscard]] PersistentBuffer& storage_table() noexcept { return storage_table_; }
    [[nodiscard]] PersistentBuffer& tx_input() noexcept { return tx_input_; }
    [[nodiscard]] PersistentBuffer& tx_output() noexcept { return tx_output_; }
    [[nodiscard]] PersistentBuffer& evm_memory() noexcept { return evm_memory_; }
    [[nodiscard]] PersistentBuffer& mv_memory() noexcept { return mv_memory_; }

    [[nodiscard]] const PersistentBuffer& account_table() const noexcept { return account_table_; }
    [[nodiscard]] const PersistentBuffer& storage_table() const noexcept { return storage_table_; }

private:
    void destroy() noexcept;

    void* device_ = nullptr;  // id<MTLDevice>
    PersistentBuffer account_table_{};
    PersistentBuffer storage_table_{};
    PersistentBuffer tx_input_{};
    PersistentBuffer tx_output_{};
    PersistentBuffer evm_memory_{};
    PersistentBuffer mv_memory_{};
};

}  // namespace evm::gpu
