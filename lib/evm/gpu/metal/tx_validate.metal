// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file tx_validate.metal
/// Metal compute shader for GPU-accelerated transaction validation.
///
/// Validates transactions in parallel before EVM execution:
///   - Nonce check: account nonce == tx nonce
///   - Balance check: account balance >= gas_limit * gas_price + value
///   - Intrinsic gas check: gas_limit >= 21000 + calldata_cost
///   - Signature presence (non-zero from address)
///
/// All reads are from GPU-resident state -- no CPU round-trip.
/// One thread per transaction. Each thread reads from a shared account
/// state table (read-only) and writes validation results.
///
/// Buffer layout:
///   [0] TxValidateInput*  -- transaction data
///   [1] AccountLookup*    -- account state hash table (GPU-resident)
///   [2] uint*             -- valid_flags: 1 = valid, 0 = invalid
///   [3] uint*             -- error_codes per tx
///   [4] uint              -- constant: num_txs
///   [5] uint              -- constant: num_accounts (hash table size)

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Constants
// =============================================================================

constant uint ACCOUNT_TABLE_SIZE = 16384;
constant uint ACCOUNT_TABLE_MASK = ACCOUNT_TABLE_SIZE - 1;

// Error codes (bitmask -- a tx can have multiple errors)
constant uint ERR_NONE           = 0;
constant uint ERR_NONCE_LOW      = (1u << 0);
constant uint ERR_NONCE_HIGH     = (1u << 1);
constant uint ERR_BALANCE_LOW    = (1u << 2);
constant uint ERR_GAS_LOW        = (1u << 3);
constant uint ERR_SENDER_ZERO    = (1u << 4);
constant uint ERR_GAS_OVERFLOW   = (1u << 5);

// EIP-2681: nonce must fit in uint64
constant ulong MAX_NONCE = 0xFFFFFFFFFFFFFFFF;

// EIP-3860: initcode max size
constant uint MAX_INITCODE_SIZE = 49152;  // 48 KB

// =============================================================================
// Data structures
// =============================================================================

struct TxValidateInput {
    uchar  from[20];       // Sender address (recovered from signature)
    uchar  to[20];         // Recipient (empty = contract creation)
    ulong  gas_limit;
    ulong  value;          // Value in wei (simplified to uint64)
    ulong  nonce;
    ulong  gas_price;      // Legacy or max_fee_per_gas
    uint   calldata_size;
    uint   is_create;      // 1 if contract creation (to == 0)
};

/// Account state entry in the GPU hash table.
/// Open addressing with linear probing, keyed by address.
struct AccountLookup {
    uchar address[20];
    uint  occupied;        // 1 = occupied, 0 = empty
    ulong nonce;
    ulong balance;         // Simplified to uint64 (wei)
};

// =============================================================================
// Account lookup
// =============================================================================

/// FNV-1a hash of a 20-byte address.
inline uint addr_hash(thread const uchar* addr) {
    uint hash = 2166136261u;
    for (int i = 0; i < 20; i++) {
        hash ^= addr[i];
        hash *= 16777619u;
    }
    return hash & ACCOUNT_TABLE_MASK;
}

/// Look up an account in the GPU hash table.
/// Returns the index if found, or ACCOUNT_TABLE_SIZE if not found.
inline uint find_account(
    device const AccountLookup* table,
    thread const uchar* addr)
{
    uint h = addr_hash(addr);
    for (uint probe = 0; probe < 256; probe++) {
        uint idx = (h + probe) & ACCOUNT_TABLE_MASK;
        if (table[idx].occupied == 0) return ACCOUNT_TABLE_SIZE;  // Not found

        bool match = true;
        for (int i = 0; i < 20; i++) {
            if (table[idx].address[i] != addr[i]) {
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

kernel void validate_transactions(
    device const TxValidateInput* txs    [[buffer(0)]],
    device const AccountLookup*   state  [[buffer(1)]],
    device uint*                  valid_flags [[buffer(2)]],
    device uint*                  error_codes [[buffer(3)]],
    constant uint&                num_txs     [[buffer(4)]],
    constant uint&                num_accounts [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_txs) return;

    device const TxValidateInput& tx = txs[tid];
    uint errors = ERR_NONE;

    // -- Check 1: sender is non-zero ------------------------------------------
    bool sender_zero = true;
    uchar from_addr[20];
    for (int i = 0; i < 20; i++) {
        from_addr[i] = tx.from[i];
        if (tx.from[i] != 0) sender_zero = false;
    }
    if (sender_zero) {
        errors |= ERR_SENDER_ZERO;
        valid_flags[tid] = 0;
        error_codes[tid] = errors;
        return;
    }

    // -- Look up sender account -----------------------------------------------
    uint acct_idx = find_account(state, from_addr);
    ulong acct_nonce = 0;
    ulong acct_balance = 0;

    if (acct_idx < ACCOUNT_TABLE_SIZE) {
        acct_nonce = state[acct_idx].nonce;
        acct_balance = state[acct_idx].balance;
    }
    // If account not found, nonce=0 and balance=0 (new account)

    // -- Check 2: nonce -------------------------------------------------------
    if (tx.nonce < acct_nonce) {
        errors |= ERR_NONCE_LOW;
    } else if (tx.nonce > acct_nonce) {
        errors |= ERR_NONCE_HIGH;
    }

    // -- Check 3: intrinsic gas -----------------------------------------------
    // Base cost: 21000 for call, 53000 for create
    ulong base_gas = tx.is_create ? 53000 : 21000;

    // Calldata cost: 4 per zero byte, 16 per non-zero byte.
    // Since we don't have access to the actual calldata bytes here,
    // we use the conservative estimate: 16 * calldata_size (all non-zero).
    // The host can pass pre-computed calldata gas if needed.
    ulong calldata_gas = (ulong)tx.calldata_size * 16;

    // EIP-3860: initcode cost for contract creation
    ulong initcode_gas = 0;
    if (tx.is_create) {
        if (tx.calldata_size > MAX_INITCODE_SIZE) {
            errors |= ERR_GAS_LOW;  // Initcode too large
        }
        // 2 gas per 32-byte word of initcode (rounded up)
        initcode_gas = ((ulong)tx.calldata_size + 31) / 32 * 2;
    }

    ulong intrinsic_gas = base_gas + calldata_gas + initcode_gas;

    if (tx.gas_limit < intrinsic_gas) {
        errors |= ERR_GAS_LOW;
    }

    // -- Check 4: balance -----------------------------------------------------
    // Required: balance >= gas_limit * gas_price + value
    // Check for overflow in gas_limit * gas_price
    ulong gas_cost;
    if (tx.gas_price > 0 && tx.gas_limit > MAX_NONCE / tx.gas_price) {
        // Overflow
        errors |= ERR_GAS_OVERFLOW;
        errors |= ERR_BALANCE_LOW;
    } else {
        gas_cost = tx.gas_limit * tx.gas_price;
        // Check for overflow in gas_cost + value
        if (gas_cost > MAX_NONCE - tx.value) {
            errors |= ERR_BALANCE_LOW;
        } else {
            ulong total_cost = gas_cost + tx.value;
            if (acct_balance < total_cost) {
                errors |= ERR_BALANCE_LOW;
            }
        }
    }

    // -- Write results --------------------------------------------------------
    valid_flags[tid] = (errors == ERR_NONE) ? 1 : 0;
    error_codes[tid] = errors;
}

// =============================================================================
// Batch nonce-sort validation
// =============================================================================

/// Validate that transactions from the same sender are ordered by nonce.
/// One thread per transaction. Checks if any earlier tx in the block
/// from the same sender has a higher nonce (ordering violation).
kernel void validate_nonce_ordering(
    device const TxValidateInput* txs         [[buffer(0)]],
    device uint*                  valid_flags  [[buffer(1)]],
    constant uint&                num_txs      [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_txs) return;
    if (valid_flags[tid] == 0) return;  // Already invalid

    device const TxValidateInput& my_tx = txs[tid];

    // Check all earlier transactions for same sender with higher nonce
    for (uint i = 0; i < tid; i++) {
        if (valid_flags[i] == 0) continue;

        device const TxValidateInput& other = txs[i];

        // Compare sender addresses
        bool same_sender = true;
        for (int j = 0; j < 20; j++) {
            if (my_tx.from[j] != other.from[j]) {
                same_sender = false;
                break;
            }
        }

        if (same_sender && other.nonce >= my_tx.nonce) {
            // Duplicate or misordered nonce from same sender
            valid_flags[tid] = 0;
            return;
        }
    }
}
