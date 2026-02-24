// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file tx_validate_host.hpp
/// C++ host-side interface for CUDA-accelerated transaction validation.
///
/// API mirrors metal/tx_validate_host.hpp so the dispatcher can swap
/// backends transparently.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace evm::gpu::cuda
{

/// Transaction input for GPU validation
/// (must match TxValidateInput in tx_validate.cu).
struct TxValidateInput
{
    uint8_t  from[20];
    uint8_t  to[20];
    uint64_t gas_limit;
    uint64_t value;
    uint64_t nonce;
    uint64_t gas_price;
    uint32_t calldata_size;
    uint32_t is_create;
};
static_assert(sizeof(TxValidateInput) == 80);

/// Account state entry for the GPU hash table
/// (must match AccountLookup in tx_validate.cu).
struct AccountLookup
{
    uint8_t  address[20];
    uint32_t occupied;
    uint64_t nonce;
    uint64_t balance;
};
static_assert(sizeof(AccountLookup) == 40);

/// Per-transaction validation result.
struct TxValidationResult
{
    bool valid;
    uint32_t error_code;
};

/// GPU-accelerated transaction validator using NVIDIA CUDA.
class TxValidator
{
public:
    virtual ~TxValidator() = default;

    /// Create a TxValidator. Returns nullptr if CUDA is unavailable
    /// (no driver, no device, or build-time CUDA support disabled).
    static std::unique_ptr<TxValidator> create();

    /// Validate a batch of transactions against GPU-resident account state.
    ///
    /// @param txs          Transaction inputs.
    /// @param num_txs      Number of transactions.
    /// @param state_table  Account state hash table (open-addressed,
    ///                     16384 entries).
    /// @param table_size   Number of entries in the hash table.
    /// @return             Per-transaction validation results.
    virtual std::vector<TxValidationResult> validate(
        const TxValidateInput* txs, size_t num_txs,
        const AccountLookup* state_table, size_t table_size) = 0;

    /// Get the CUDA device name.
    virtual const char* device_name() const = 0;

protected:
    TxValidator() = default;
    TxValidator(const TxValidator&) = delete;
    TxValidator& operator=(const TxValidator&) = delete;
};

}  // namespace evm::gpu::cuda
