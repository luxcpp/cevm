// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file tx_validate.cu
/// CUDA port of metal/tx_validate.metal.
///
/// Validates transactions in parallel before EVM execution:
///   - Sender non-zero
///   - Nonce check: account.nonce == tx.nonce
///   - Intrinsic gas: gas_limit >= base + calldata + initcode cost
///   - Balance: balance >= gas_limit * gas_price + value
///   - Overflow guards on the gas-price multiplication
///
/// One thread per transaction. The account state is a GPU-resident
/// open-addressing hash table keyed by 20-byte address.

#include <cstdint>
#include <cuda_runtime.h>

namespace evm::gpu::cuda
{

// =============================================================================
// Constants (must match metal)
// =============================================================================

__device__ static constexpr uint32_t ACCOUNT_TABLE_SIZE = 16384;
__device__ static constexpr uint32_t ACCOUNT_TABLE_MASK = ACCOUNT_TABLE_SIZE - 1;

__device__ static constexpr uint32_t ERR_NONE         = 0;
__device__ static constexpr uint32_t ERR_NONCE_LOW    = 1u << 0;
__device__ static constexpr uint32_t ERR_NONCE_HIGH   = 1u << 1;
__device__ static constexpr uint32_t ERR_BALANCE_LOW  = 1u << 2;
__device__ static constexpr uint32_t ERR_GAS_LOW      = 1u << 3;
__device__ static constexpr uint32_t ERR_SENDER_ZERO  = 1u << 4;
__device__ static constexpr uint32_t ERR_GAS_OVERFLOW = 1u << 5;

__device__ static constexpr uint64_t MAX_U64           = 0xFFFFFFFFFFFFFFFFULL;
__device__ static constexpr uint32_t MAX_INITCODE_SIZE = 49152;  // EIP-3860

// =============================================================================
// Data structures (must match metal exactly + tx_validate_host.hpp)
// =============================================================================

struct DeviceTxInput
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
static_assert(sizeof(DeviceTxInput) == 80, "DeviceTxInput layout mismatch");

struct DeviceAccount
{
    uint8_t  address[20];
    uint32_t occupied;
    uint64_t nonce;
    uint64_t balance;
};
static_assert(sizeof(DeviceAccount) == 40, "DeviceAccount layout mismatch");

// =============================================================================
// Account lookup (FNV-1a hash of 20-byte address + linear probing)
// =============================================================================

__device__ __forceinline__ uint32_t addr_hash(const uint8_t* addr)
{
    uint32_t h = 2166136261u;
    #pragma unroll
    for (int i = 0; i < 20; ++i)
    {
        h ^= addr[i];
        h *= 16777619u;
    }
    return h & ACCOUNT_TABLE_MASK;
}

__device__ uint32_t find_account(const DeviceAccount* table, const uint8_t* addr)
{
    uint32_t h = addr_hash(addr);
    for (uint32_t probe = 0; probe < 256; ++probe)
    {
        uint32_t idx = (h + probe) & ACCOUNT_TABLE_MASK;
        if (table[idx].occupied == 0)
            return ACCOUNT_TABLE_SIZE;  // empty slot -> not found
        bool match = true;
        #pragma unroll
        for (int i = 0; i < 20; ++i)
        {
            if (table[idx].address[i] != addr[i])
            {
                match = false;
                break;
            }
        }
        if (match) return idx;
    }
    return ACCOUNT_TABLE_SIZE;
}

// =============================================================================
// Validation kernel
// =============================================================================

__global__ void validate_transactions_kernel(
    const DeviceTxInput*  __restrict__ txs,
    const DeviceAccount*  __restrict__ state,
    uint32_t*             __restrict__ valid_flags,
    uint32_t*             __restrict__ error_codes,
    uint32_t                            num_txs)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_txs) return;

    const DeviceTxInput tx = txs[tid];
    uint32_t errors = ERR_NONE;

    // -- 1. Sender non-zero ---------------------------------------------------
    bool sender_zero = true;
    uint8_t from_addr[20];
    #pragma unroll
    for (int i = 0; i < 20; ++i)
    {
        from_addr[i] = tx.from[i];
        if (tx.from[i] != 0) sender_zero = false;
    }
    if (sender_zero)
    {
        valid_flags[tid] = 0;
        error_codes[tid] = ERR_SENDER_ZERO;
        return;
    }

    // -- 2. Look up sender account -------------------------------------------
    uint32_t acct_idx = find_account(state, from_addr);
    uint64_t acct_nonce   = 0;
    uint64_t acct_balance = 0;
    if (acct_idx < ACCOUNT_TABLE_SIZE)
    {
        acct_nonce   = state[acct_idx].nonce;
        acct_balance = state[acct_idx].balance;
    }

    // -- 3. Nonce -------------------------------------------------------------
    if (tx.nonce < acct_nonce)
        errors |= ERR_NONCE_LOW;
    else if (tx.nonce > acct_nonce)
        errors |= ERR_NONCE_HIGH;

    // -- 4. Intrinsic gas -----------------------------------------------------
    uint64_t base_gas    = tx.is_create ? 53000 : 21000;
    uint64_t calldata_gas = (uint64_t)tx.calldata_size * 16;
    uint64_t initcode_gas = 0;
    if (tx.is_create)
    {
        if (tx.calldata_size > MAX_INITCODE_SIZE)
            errors |= ERR_GAS_LOW;
        initcode_gas = ((uint64_t)tx.calldata_size + 31) / 32 * 2;
    }
    uint64_t intrinsic_gas = base_gas + calldata_gas + initcode_gas;
    if (tx.gas_limit < intrinsic_gas)
        errors |= ERR_GAS_LOW;

    // -- 5. Balance with overflow guards -------------------------------------
    if (tx.gas_price > 0 && tx.gas_limit > MAX_U64 / tx.gas_price)
    {
        errors |= ERR_GAS_OVERFLOW;
        errors |= ERR_BALANCE_LOW;
    }
    else
    {
        uint64_t gas_cost = tx.gas_limit * tx.gas_price;
        if (gas_cost > MAX_U64 - tx.value)
        {
            errors |= ERR_BALANCE_LOW;
        }
        else
        {
            uint64_t total_cost = gas_cost + tx.value;
            if (acct_balance < total_cost)
                errors |= ERR_BALANCE_LOW;
        }
    }

    valid_flags[tid] = (errors == ERR_NONE) ? 1u : 0u;
    error_codes[tid] = errors;
}

// =============================================================================
// Nonce-ordering kernel: detect same-sender txs out of order in the block.
// =============================================================================

__global__ void validate_nonce_ordering_kernel(
    const DeviceTxInput*  __restrict__ txs,
    uint32_t*             __restrict__ valid_flags,
    uint32_t                            num_txs)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_txs) return;
    if (valid_flags[tid] == 0) return;

    const DeviceTxInput my_tx = txs[tid];

    for (uint32_t i = 0; i < tid; ++i)
    {
        if (valid_flags[i] == 0) continue;
        const DeviceTxInput other = txs[i];

        bool same_sender = true;
        #pragma unroll
        for (int j = 0; j < 20; ++j)
        {
            if (my_tx.from[j] != other.from[j])
            {
                same_sender = false;
                break;
            }
        }
        if (same_sender && other.nonce >= my_tx.nonce)
        {
            valid_flags[tid] = 0;
            return;
        }
    }
}

// =============================================================================
// Host-callable launchers
// =============================================================================

extern "C" cudaError_t evm_cuda_tx_validate_launch(
    const void*  d_txs,
    const void*  d_state,
    void*        d_valid_flags,
    void*        d_error_codes,
    uint32_t     num_txs,
    cudaStream_t stream)
{
    if (num_txs == 0) return cudaSuccess;
    constexpr uint32_t threads_per_block = 128;
    const uint32_t blocks =
        (num_txs + threads_per_block - 1) / threads_per_block;

    validate_transactions_kernel<<<blocks, threads_per_block, 0, stream>>>(
        static_cast<const DeviceTxInput*>(d_txs),
        static_cast<const DeviceAccount*>(d_state),
        static_cast<uint32_t*>(d_valid_flags),
        static_cast<uint32_t*>(d_error_codes),
        num_txs);

    return cudaGetLastError();
}

extern "C" cudaError_t evm_cuda_tx_nonce_order_launch(
    const void*  d_txs,
    void*        d_valid_flags,
    uint32_t     num_txs,
    cudaStream_t stream)
{
    if (num_txs == 0) return cudaSuccess;
    constexpr uint32_t threads_per_block = 128;
    const uint32_t blocks =
        (num_txs + threads_per_block - 1) / threads_per_block;

    validate_nonce_ordering_kernel<<<blocks, threads_per_block, 0, stream>>>(
        static_cast<const DeviceTxInput*>(d_txs),
        static_cast<uint32_t*>(d_valid_flags),
        num_txs);

    return cudaGetLastError();
}

}  // namespace evm::gpu::cuda
