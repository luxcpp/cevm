// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "processor.hpp"

#include <chrono>

// Forward-declare evmone factory.
extern "C" struct evmc_vm* evmc_create_evmone(void) noexcept;

namespace evm::state
{

/// Intrinsic gas cost for a transaction (EIP-2028).
static uint64_t intrinsic_gas(const Transaction& tx) noexcept
{
    uint64_t gas = 21000;  // Base cost.

    // Calldata cost.
    for (const auto byte : tx.data)
        gas += (byte == 0) ? 4 : 16;

    // Contract creation adds 32000.
    if (tx.is_create)
        gas += 32000;

    return gas;
}

BlockResult process_block(
    StateDB& db,
    const std::vector<Transaction>& txs,
    evmc_vm* vm,
    const TxContext& tx_ctx,
    evmc_revision rev)
{
    BlockResult result;
    result.tx_results.resize(txs.size());

    const auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < txs.size(); ++i)
    {
        const auto& tx = txs[i];
        auto& tx_result = result.tx_results[i];

        // --- Validation ---

        // Nonce check.
        const auto db_nonce = db.get_nonce(tx.sender);
        if (tx.nonce != db_nonce)
        {
            tx_result.status = EVMC_REJECTED;
            tx_result.gas_used = 0;
            continue;
        }

        // Intrinsic gas.
        const auto base_gas = intrinsic_gas(tx);
        if (tx.gas_limit < base_gas)
        {
            tx_result.status = EVMC_REJECTED;
            tx_result.gas_used = 0;
            continue;
        }

        // Balance check: sender must cover gas_limit * gas_price + value.
        const auto gas_cost = tx.gas_price * intx::uint256{tx.gas_limit};
        const auto total_cost = gas_cost + tx.value;
        if (db.get_balance(tx.sender) < total_cost)
        {
            tx_result.status = EVMC_REJECTED;
            tx_result.gas_used = 0;
            continue;
        }

        // --- Pre-execution: deduct gas cost, increment nonce ---

        const auto snap = db.snapshot();
        db.sub_balance(tx.sender, gas_cost);
        db.increment_nonce(tx.sender);

        // Clear per-tx transient storage and access lists.
        db.clear_transient_storage();
        db.clear_access_lists();

        // Pre-warm sender and recipient (EIP-2929).
        db.warm_account(tx.sender);
        db.warm_account(tx_ctx.coinbase);
        if (!tx.is_create)
            db.warm_account(tx.recipient);

        // --- Execute ---

        uint64_t gas_used;

        if (!tx.is_create && tx.data.empty() && db.get_code_size(tx.recipient) == 0)
        {
            // Simple value transfer: no EVM execution needed.
            if (tx.value > 0)
            {
                db.add_balance(tx.recipient, tx.value);
                db.sub_balance(tx.sender, tx.value);
            }
            gas_used = base_gas;
            tx_result.status = EVMC_SUCCESS;
        }
        else
        {
            // Full EVM execution.
            TxContext per_tx_ctx = tx_ctx;
            per_tx_ctx.origin = tx.sender;
            per_tx_ctx.gas_price = intx::be::store<evmc::uint256be>(tx.gas_price);

            EvmcStateHost host(db, per_tx_ctx, rev);
            host.set_vm(vm);

            evmc_message msg{};
            msg.kind = tx.is_create ? EVMC_CREATE : EVMC_CALL;
            msg.gas = static_cast<int64_t>(tx.gas_limit - base_gas);
            msg.sender = tx.sender;
            msg.recipient = tx.recipient;
            msg.value = intx::be::store<evmc::uint256be>(tx.value);
            msg.input_data = tx.data.data();
            msg.input_size = tx.data.size();
            msg.depth = 0;

            // Transfer value for CALL.
            if (!tx.is_create && tx.value > 0)
            {
                db.sub_balance(tx.sender, tx.value);
                db.add_balance(tx.recipient, tx.value);
            }

            const uint8_t* code_ptr = nullptr;
            size_t code_size = 0;

            if (tx.is_create)
            {
                // Init code is in tx.data.
                code_ptr = tx.data.data();
                code_size = tx.data.size();
            }
            else
            {
                const auto& code = db.get_code(tx.recipient);
                code_ptr = code.data();
                code_size = code.size();
            }

            if (code_size > 0)
            {
                const auto& iface = evmc::Host::get_interface();
                auto* ctx = host.to_context();
                auto r = vm->execute(vm, &iface, ctx, rev, &msg, code_ptr, code_size);

                if (r.status_code != EVMC_SUCCESS)
                {
                    // Revert all state changes from this tx except gas deduction and nonce.
                    db.revert(snap);
                    // Re-apply gas deduction and nonce increment.
                    db.sub_balance(tx.sender, gas_cost);
                    db.increment_nonce(tx.sender);

                    // On failure, all gas is consumed.
                    gas_used = tx.gas_limit;
                    tx_result.status = r.status_code;
                }
                else
                {
                    gas_used = base_gas + static_cast<uint64_t>(msg.gas - r.gas_left);
                    tx_result.status = EVMC_SUCCESS;
                }

                if (r.release != nullptr)
                    r.release(&r);
            }
            else
            {
                // No code to execute (e.g., transfer to EOA with calldata).
                gas_used = base_gas;
                tx_result.status = EVMC_SUCCESS;
            }
        }

        // --- Post-execution: refund unused gas, pay coinbase ---

        const auto gas_refund = tx.gas_limit - gas_used;
        if (gas_refund > 0)
        {
            const auto refund_amount = tx.gas_price * intx::uint256{gas_refund};
            db.add_balance(tx.sender, refund_amount);
        }

        // Pay coinbase.
        const auto coinbase_reward = tx.gas_price * intx::uint256{gas_used};
        db.add_balance(tx_ctx.coinbase, coinbase_reward);

        tx_result.gas_used = gas_used;
        result.total_gas += gas_used;
    }

    // Compute state root.
    result.state_root = db.commit();

    const auto end = std::chrono::high_resolution_clock::now();
    result.execution_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    return result;
}

}  // namespace evm::state
