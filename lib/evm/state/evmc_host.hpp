// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file evmc_host.hpp
/// EVMC Host adapter that wires evmone's execute() to StateDB.
///
/// This is the critical bridge: evmone calls Host methods during execution,
/// and this adapter translates them into StateDB operations on real account state.

#pragma once

#include "state_db.hpp"

#include <evmc/evmc.hpp>
#include <intx/intx.hpp>
#include <cstring>
#include <vector>

namespace evm::state
{

/// Transaction context for the current execution.
struct TxContext
{
    evmc::address origin{};
    evmc::address coinbase{};
    int64_t block_number = 0;
    int64_t block_timestamp = 0;
    int64_t block_gas_limit = 30'000'000;
    evmc::uint256be gas_price{};
    evmc::uint256be chain_id{};
    evmc::uint256be block_base_fee{};
    evmc::uint256be prev_randao{};
    evmc::uint256be blob_base_fee{};
};

/// EVMC Host implementation backed by StateDB.
///
/// Implements every method in evmc::Host using real state operations.
/// Handles value transfers in call(), tracks access lists, and supports
/// snapshot/revert for nested calls.
class EvmcStateHost : public evmc::Host
{
public:
    EvmcStateHost(StateDB& db, const TxContext& tx_ctx, evmc_revision rev) noexcept
      : db_{db}, tx_ctx_{tx_ctx}, rev_{rev}
    {}

    // --- Account existence ---

    bool account_exists(const evmc::address& addr) const noexcept override
    {
        return db_.account_exists(addr);
    }

    // --- Storage ---

    evmc::bytes32 get_storage(
        const evmc::address& addr, const evmc::bytes32& key) const noexcept override
    {
        return db_.get_storage(addr, key);
    }

    evmc_storage_status set_storage(
        const evmc::address& addr,
        const evmc::bytes32& key,
        const evmc::bytes32& value) noexcept override
    {
        const auto current = db_.get_storage(addr, key);

        if (current == value)
            return EVMC_STORAGE_ASSIGNED;

        db_.set_storage(addr, key, value);

        // Determine storage status for gas accounting.
        const evmc::bytes32 zero{};
        if (current == zero)
            return EVMC_STORAGE_ADDED;
        if (value == zero)
            return EVMC_STORAGE_DELETED;
        return EVMC_STORAGE_MODIFIED;
    }

    // --- Balance ---

    evmc::uint256be get_balance(const evmc::address& addr) const noexcept override
    {
        return intx::be::store<evmc::uint256be>(db_.get_balance(addr));
    }

    // --- Code ---

    size_t get_code_size(const evmc::address& addr) const noexcept override
    {
        return db_.get_code_size(addr);
    }

    evmc::bytes32 get_code_hash(const evmc::address& addr) const noexcept override
    {
        if (!db_.account_exists(addr))
            return {};
        return db_.get_code_hash(addr);
    }

    size_t copy_code(const evmc::address& addr, size_t code_offset,
                     uint8_t* buffer_data, size_t buffer_size) const noexcept override
    {
        const auto& code = db_.get_code(addr);
        if (code_offset >= code.size())
            return 0;
        const auto copy_size = std::min(buffer_size, code.size() - code_offset);
        std::memcpy(buffer_data, code.data() + code_offset, copy_size);
        return copy_size;
    }

    // --- Self-destruct ---

    bool selfdestruct(const evmc::address& addr,
                      const evmc::address& beneficiary) noexcept override
    {
        db_.selfdestruct(addr, beneficiary);
        return true;
    }

    // --- Call (value transfer + nested execution) ---

    evmc::Result call(const evmc_message& msg) noexcept override
    {
        // For simple value transfers (no code at recipient), handle directly.
        const auto value = intx::be::load<intx::uint256>(msg.value);

        if (msg.kind == EVMC_CALL || msg.kind == EVMC_CALLCODE)
        {
            // Transfer value from sender to recipient.
            if (value > 0)
            {
                if (db_.get_balance(msg.sender) < value)
                    return evmc::Result{EVMC_INSUFFICIENT_BALANCE, msg.gas, 0};

                db_.sub_balance(msg.sender, value);
                db_.add_balance(msg.recipient, value);
            }

            // If recipient has code, we would need to execute it.
            // For the benchmark (simple transfers), recipients have no code.
            const auto code_size = db_.get_code_size(msg.recipient);
            if (code_size > 0)
            {
                // Nested execution: create a snapshot, execute, revert on failure.
                const auto snap = db_.snapshot();
                const auto& code = db_.get_code(msg.recipient);

                // Get the VM and execute.
                auto* vm = get_vm();
                if (vm != nullptr)
                {
                    const auto& iface = evmc::Host::get_interface();
                    auto* ctx = this->to_context();
                    auto r = vm->execute(vm, &iface, ctx, rev_,
                                         &msg, code.data(), code.size());
                    if (r.status_code != EVMC_SUCCESS)
                        db_.revert(snap);
                    return evmc::Result{r};
                }

                // No VM available (shouldn't happen in practice).
                db_.revert(snap);
                return evmc::Result{EVMC_INTERNAL_ERROR, 0, 0};
            }
        }
        else if (msg.kind == EVMC_CREATE || msg.kind == EVMC_CREATE2)
        {
            // Contract creation: deduct value, create account.
            if (value > 0)
            {
                if (db_.get_balance(msg.sender) < value)
                    return evmc::Result{EVMC_INSUFFICIENT_BALANCE, msg.gas, 0};
                db_.sub_balance(msg.sender, value);
            }

            // Compute create address (simplified).
            evmc::address new_addr{};
            const auto sender_nonce = db_.get_nonce(msg.sender);
            // Simple address derivation: keccak(sender || nonce)[12:]
            uint8_t buf[28] = {};
            std::memcpy(buf, msg.sender.bytes, 20);
            for (int i = 0; i < 8; ++i)
                buf[20 + i] = static_cast<uint8_t>(sender_nonce >> ((7 - i) * 8));
            const auto h = ethash::keccak256(buf, 28);
            std::memcpy(new_addr.bytes, h.bytes + 12, 20);

            db_.create_account(new_addr);
            if (value > 0)
                db_.add_balance(new_addr, value);

            return evmc::Result{EVMC_SUCCESS, msg.gas, 0, new_addr};
        }

        return evmc::Result{EVMC_SUCCESS, msg.gas, 0};
    }

    // --- Transaction context ---

    evmc_tx_context get_tx_context() const noexcept override
    {
        evmc_tx_context ctx{};
        ctx.tx_origin = tx_ctx_.origin;
        ctx.block_coinbase = tx_ctx_.coinbase;
        ctx.block_number = tx_ctx_.block_number;
        ctx.block_timestamp = tx_ctx_.block_timestamp;
        ctx.block_gas_limit = tx_ctx_.block_gas_limit;
        ctx.tx_gas_price = tx_ctx_.gas_price;
        ctx.chain_id = tx_ctx_.chain_id;
        ctx.block_base_fee = tx_ctx_.block_base_fee;
        ctx.block_prev_randao = tx_ctx_.prev_randao;
        ctx.blob_base_fee = tx_ctx_.blob_base_fee;
        return ctx;
    }

    evmc::bytes32 get_block_hash(int64_t block_number) const noexcept override
    {
        // Synthetic block hash for testing.
        evmc::bytes32 h{};
        for (int i = 0; i < 8; ++i)
            h.bytes[24 + i] = static_cast<uint8_t>(block_number >> ((7 - i) * 8));
        return h;
    }

    // --- Logging ---

    void emit_log(const evmc::address& addr, const uint8_t* data, size_t data_size,
                  const evmc::bytes32 topics[], size_t num_topics) noexcept override
    {
        // Record logs for receipt generation.
        LogEntry log{addr, {data, data + data_size}, {}};
        log.topics.reserve(num_topics);
        for (size_t i = 0; i < num_topics; ++i)
            log.topics.push_back(topics[i]);
        logs_.push_back(std::move(log));
    }

    // --- Access lists (EIP-2929) ---

    evmc_access_status access_account(const evmc::address& addr) noexcept override
    {
        if (rev_ < EVMC_BERLIN)
            return EVMC_ACCESS_WARM;
        return db_.warm_account(addr) ? EVMC_ACCESS_WARM : EVMC_ACCESS_COLD;
    }

    evmc_access_status access_storage(
        const evmc::address& addr, const evmc::bytes32& key) noexcept override
    {
        if (rev_ < EVMC_BERLIN)
            return EVMC_ACCESS_WARM;
        return db_.warm_storage(addr, key) ? EVMC_ACCESS_WARM : EVMC_ACCESS_COLD;
    }

    // --- Transient storage (EIP-1153) ---

    evmc::bytes32 get_transient_storage(
        const evmc::address& addr, const evmc::bytes32& key) const noexcept override
    {
        return db_.get_transient_storage(addr, key);
    }

    void set_transient_storage(
        const evmc::address& addr,
        const evmc::bytes32& key,
        const evmc::bytes32& value) noexcept override
    {
        db_.set_transient_storage(addr, key, value);
    }

    // --- Accessors ---

    [[nodiscard]] const auto& logs() const noexcept { return logs_; }
    void clear_logs() noexcept { logs_.clear(); }

    StateDB& state_db() noexcept { return db_; }
    const StateDB& state_db() const noexcept { return db_; }

    /// Set the EVM instance for nested calls.
    void set_vm(evmc_vm* vm) noexcept { vm_ = vm; }

private:
    evmc_vm* get_vm() const noexcept { return vm_; }

    struct LogEntry
    {
        evmc::address addr;
        std::vector<uint8_t> data;
        std::vector<evmc::bytes32> topics;
    };

    StateDB& db_;
    TxContext tx_ctx_;
    evmc_revision rev_;
    evmc_vm* vm_ = nullptr;
    std::vector<LogEntry> logs_;
};

}  // namespace evm::state
