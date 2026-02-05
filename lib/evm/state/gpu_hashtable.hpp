// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file gpu_hashtable.hpp
/// GPU-resident open-addressing hash table for Ethereum state.
///
/// All state lives in GPU device memory (Metal buffers) persistently across
/// blocks. On Apple Silicon unified memory, these buffers are also directly
/// readable by the CPU (zero-copy via MTLResourceStorageModeShared).
///
/// The C++ methods do NOT copy data to/from GPU. They dispatch Metal compute
/// kernels that read/write the table directly on the GPU.

#pragma once

#include "account.hpp"

#include <evmc/evmc.hpp>
#include <cstdint>
#include <memory>

namespace evm::state
{

/// Account data as stored in the GPU hash table.
/// Must match the Metal shader's AccountData struct exactly.
struct alignas(8) GpuAccountData
{
    uint64_t nonce;            //  8 bytes
    uint64_t balance[4];       // 32 bytes
    uint8_t  code_hash[32];    // 32 bytes
    uint8_t  storage_root[32]; // 32 bytes
};                             // total: 104 bytes
static_assert(sizeof(GpuAccountData) == 104);

/// Storage key: address + slot.
/// Must match the Metal shader's StorageKey struct exactly.
struct GpuStorageKey
{
    uint8_t addr[20];
    uint8_t slot[32];
};
static_assert(sizeof(GpuStorageKey) == 52);

/// GPU-resident hash table.
///
/// Wraps a persistent Metal buffer containing an open-addressing hash table.
/// All operations dispatch Metal compute kernels -- no CPU-side hash map.
///
/// On Apple Silicon, the CPU can also read the table contents directly through
/// the shared memory pointer for RPC queries or debugging.
class GpuHashTable
{
public:
    virtual ~GpuHashTable() = default;

    /// Create a GPU hash table backed by Metal.
    /// @param capacity  Number of slots (must be power of 2).
    /// @param device    MTLDevice pointer (as void*). If null, uses system default.
    /// @return          nullptr if Metal is unavailable.
    static std::unique_ptr<GpuHashTable> create(uint32_t capacity, void* device = nullptr);

    /// Create with a pre-allocated buffer (from PersistentBufferPool).
    /// The buffer must be large enough for `capacity` entries.
    /// @param capacity   Number of slots (must be power of 2).
    /// @param buffer     Metal buffer (id<MTLBuffer> as void*). Ownership NOT transferred.
    /// @param device     MTLDevice pointer (as void*).
    /// @param queue      MTLCommandQueue pointer (as void*).
    /// @return           nullptr on failure.
    static std::unique_ptr<GpuHashTable> create_with_buffer(
        uint32_t capacity, void* buffer, void* device, void* queue);

    // -- Account operations (dispatch GPU kernels) ---

    /// Look up N accounts by address. Results written to `results`.
    /// `found[i]` is set to 1 if the account exists, 0 otherwise.
    virtual void lookup_accounts(
        const evmc::address* addrs, uint32_t n,
        GpuAccountData* results, uint32_t* found) = 0;

    /// Insert or update N accounts.
    virtual void insert_accounts(
        const evmc::address* addrs, const GpuAccountData* data, uint32_t n) = 0;

    // -- Storage operations (dispatch GPU kernels) ---

    /// Look up N storage slots.
    virtual void lookup_storage(
        const GpuStorageKey* keys, uint32_t n,
        evmc::bytes32* values, uint32_t* found) = 0;

    /// Insert or update N storage slots.
    virtual void insert_storage(
        const GpuStorageKey* keys, const evmc::bytes32* values, uint32_t n) = 0;

    // -- State root computation (entirely on GPU) ---

    /// Compute the state root by hashing all occupied entries.
    /// Phase 1: parallel RLP-encode + keccak256 each account.
    /// Phase 2: parallel reduce via pairwise keccak256.
    virtual evmc::bytes32 compute_state_root() = 0;

    // -- Direct CPU access (zero-copy on unified memory) ---

    /// Get the raw table buffer pointer for CPU-side reads.
    /// On Apple Silicon, this is the same physical memory the GPU writes to.
    [[nodiscard]] virtual void* table_contents() const noexcept = 0;

    /// Get the Metal buffer handle (id<MTLBuffer> as void*).
    [[nodiscard]] virtual void* metal_buffer() const noexcept = 0;

    [[nodiscard]] virtual uint32_t capacity() const noexcept = 0;
    [[nodiscard]] virtual uint32_t count() const noexcept = 0;

protected:
    GpuHashTable() = default;
    GpuHashTable(const GpuHashTable&) = delete;
    GpuHashTable& operator=(const GpuHashTable&) = delete;
};

}  // namespace evm::state
