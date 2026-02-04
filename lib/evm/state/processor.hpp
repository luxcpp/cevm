// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file processor.hpp
/// Block transaction processor wiring StateDB to evmone.
/// Supports GPU-accelerated batch ecrecover for sender recovery.

#pragma once

#include "evmc_host.hpp"
#include "state_db.hpp"

#include <evmc/evmc.hpp>
#include <intx/intx.hpp>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace evm::state
{

/// A signed transaction ready for processing.
struct Transaction
{
    evmc::address sender{};
    evmc::address recipient{};       ///< Empty for contract creation.
    intx::uint256 value = 0;
    intx::uint256 gas_price = 0;
    uint64_t gas_limit = 21000;
    uint64_t nonce = 0;
    std::vector<uint8_t> data;       ///< Calldata or init code.
    bool is_create = false;

    /// Raw ECDSA signature fields (populated from RLP decode).
    /// When present, GPU batch ecrecover uses these instead of per-tx recovery.
    uint8_t sig_r[32] = {};
    uint8_t sig_s[32] = {};
    uint8_t sig_v = 0;
    uint8_t msg_hash[32] = {};
    bool has_signature = false;       ///< True if sig fields are populated.
};

/// Cache of pre-recovered transaction senders.
/// Key: index into the transaction vector.
using SenderCache = std::unordered_map<size_t, evmc::address>;

/// Result of processing a single transaction.
struct TxResult
{
    uint64_t gas_used = 0;
    evmc_status_code status = EVMC_SUCCESS;
};

/// Result of processing an entire block.
struct BlockResult
{
    std::vector<TxResult> tx_results;
    uint64_t total_gas = 0;
    evmc::bytes32 state_root{};
    double execution_time_ms = 0.0;
};

/// Batch-recover transaction senders using GPU ecrecover.
///
/// For every transaction with has_signature=true, submits (r, s, v, msg_hash)
/// to lux_gpu_ecrecover_batch() and populates the sender cache.
/// Falls back gracefully if GPU is unavailable: returns an empty cache,
/// and process_block uses the pre-set tx.sender field.
///
/// @param txs  Transactions with raw signature fields.
/// @return     Map from tx index to recovered sender address.
SenderCache batch_recover_senders(const std::vector<Transaction>& txs);

/// Process a block of transactions through evmone with full state.
///
/// For each transaction:
/// 1. Validate nonce and balance.
/// 2. Deduct gas_limit * gas_price from sender.
/// 3. Execute via evmone (or direct transfer for no-code recipients).
/// 4. Apply state changes (or revert on failure).
/// 5. Refund unused gas to sender.
/// 6. Add gas_used * gas_price to coinbase.
///
/// @param db       The state database (modified in place).
/// @param txs      Transactions to process.
/// @param vm       The evmone VM instance.
/// @param tx_ctx   Transaction/block context.
/// @param rev      EVM revision.
/// @param senders  Optional pre-recovered sender cache (from batch_recover_senders).
/// @return         Block result with per-tx gas usage and state root.
BlockResult process_block(
    StateDB& db,
    const std::vector<Transaction>& txs,
    evmc_vm* vm,
    const TxContext& tx_ctx,
    evmc_revision rev,
    const SenderCache& senders = {});

}  // namespace evm::state
