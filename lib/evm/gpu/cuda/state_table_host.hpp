// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file state_table_host.hpp
/// CUDA host interface for GPU-resident Ethereum state hash table.
///
/// Wraps the kernels in state_table.cu:
///   - account_lookup_batch / account_insert_batch (key = 20-byte address)
///   - storage_lookup_batch / storage_insert_batch (key = 20-byte addr + 32-byte slot)
///   - state_root pipeline   (compact + bitonic sort + keccak(RLP) + reduce)
///
/// The table is a power-of-two open-addressing hash table that lives in
/// device memory across calls.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace evm::gpu::cuda
{

// -- GPU-side struct mirrors (must match state_table.cu byte-for-byte) -------

struct GpuAccountData
{
    uint64_t nonce;
    uint64_t balance[4];        // uint256 little-endian limbs
    uint8_t  code_hash[32];
    uint8_t  storage_root[32];
};
static_assert(sizeof(GpuAccountData) == 8 + 32 + 32 + 32,
              "GpuAccountData layout mismatch");

struct GpuAccountEntry
{
    uint8_t        key[20];
    uint32_t       key_valid;
    uint32_t       _pad;
    GpuAccountData data;
};

struct GpuStorageKey
{
    uint8_t addr[20];
    uint8_t slot[32];
};

struct GpuStorageEntry
{
    uint8_t  key_addr[20];
    uint8_t  key_slot[32];
    uint32_t key_valid;
    uint32_t _pad;
    uint8_t  value[32];
};

/// Capacity of the on-device tables (power of 2). Matches metal default.
static constexpr uint32_t DEFAULT_ACCOUNT_CAPACITY = 16384;
static constexpr uint32_t DEFAULT_STORAGE_CAPACITY = 65536;

/// Returns true when the CUDA backend is available at runtime.
bool state_table_cuda_available();

/// GPU-resident Ethereum state hash table.
class StateTable
{
public:
    virtual ~StateTable() = default;

    /// Returns nullptr if CUDA is unavailable.
    static std::unique_ptr<StateTable> create(
        uint32_t account_capacity = DEFAULT_ACCOUNT_CAPACITY,
        uint32_t storage_capacity = DEFAULT_STORAGE_CAPACITY);

    virtual const char* device_name() const = 0;

    /// Reset both tables to empty.
    virtual void clear() = 0;

    // -- Account ops ----------------------------------------------------------

    virtual void account_insert(
        const uint8_t*        keys_20bytes,    // count * 20
        const GpuAccountData* values,          // count entries
        uint32_t              count) = 0;

    virtual void account_lookup(
        const uint8_t*        keys_20bytes,    // count * 20
        GpuAccountData*       out_values,      // count entries
        uint32_t*             out_found_flags, // count flags
        uint32_t              count) = 0;

    // -- Storage ops ----------------------------------------------------------

    virtual void storage_insert(
        const GpuStorageKey* keys,
        const uint8_t*       values_32bytes,   // count * 32
        uint32_t             count) = 0;

    virtual void storage_lookup(
        const GpuStorageKey* keys,
        uint8_t*             out_values_32bytes,  // count * 32
        uint32_t*            out_found_flags,
        uint32_t             count) = 0;

    // -- State root -----------------------------------------------------------

    /// Compute the state root over the current account table.
    /// Pipeline: keccak256(RLP(account)) per entry, then pairwise reduce.
    /// Output: 32-byte big-endian digest.
    virtual std::vector<uint8_t> compute_state_root() = 0;

protected:
    StateTable() = default;
    StateTable(const StateTable&) = delete;
    StateTable& operator=(const StateTable&) = delete;
};

}  // namespace evm::gpu::cuda
