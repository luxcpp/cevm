// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file state_db.hpp
/// In-memory state database with journal-based snapshot/revert.
///
/// This is the core state layer that sits between the EVMC Host and a future
/// Merkle Patricia Trie. All account data lives in flat maps; the "state root"
/// is computed as a Keccak-256 of the RLP-encoded account data.

#pragma once

#include "account.hpp"
#include "storage.hpp"

#include <evmc/evmc.hpp>
#include <intx/intx.hpp>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace evm::state
{

using intx::uint256;

/// In-memory state database with full Ethereum account model.
class StateDB
{
public:
    StateDB() = default;

    // --- Account existence ---

    [[nodiscard]] bool account_exists(const evmc::address& addr) const noexcept;

    /// Create an account. No-op if it already exists.
    void create_account(const evmc::address& addr);

    // --- Balance ---

    [[nodiscard]] uint256 get_balance(const evmc::address& addr) const noexcept;
    void set_balance(const evmc::address& addr, const uint256& amount);
    void add_balance(const evmc::address& addr, const uint256& amount);
    void sub_balance(const evmc::address& addr, const uint256& amount);

    // --- Nonce ---

    [[nodiscard]] uint64_t get_nonce(const evmc::address& addr) const noexcept;
    void set_nonce(const evmc::address& addr, uint64_t nonce);
    void increment_nonce(const evmc::address& addr);

    // --- Storage ---

    [[nodiscard]] evmc::bytes32 get_storage(
        const evmc::address& addr, const evmc::bytes32& key) const noexcept;
    void set_storage(
        const evmc::address& addr, const evmc::bytes32& key, const evmc::bytes32& value);

    // --- Code ---

    [[nodiscard]] const std::vector<uint8_t>& get_code(const evmc::address& addr) const noexcept;
    [[nodiscard]] size_t get_code_size(const evmc::address& addr) const noexcept;
    [[nodiscard]] evmc::bytes32 get_code_hash(const evmc::address& addr) const noexcept;
    void set_code(const evmc::address& addr, std::vector<uint8_t> code);

    // --- Self-destruct ---

    void selfdestruct(const evmc::address& addr, const evmc::address& beneficiary);

    // --- Access lists (EIP-2929) ---

    [[nodiscard]] bool is_account_warm(const evmc::address& addr) const noexcept;
    bool warm_account(const evmc::address& addr);

    [[nodiscard]] bool is_storage_warm(
        const evmc::address& addr, const evmc::bytes32& key) const noexcept;
    bool warm_storage(const evmc::address& addr, const evmc::bytes32& key);

    void clear_access_lists() noexcept;

    // --- Transient storage (EIP-1153) ---

    [[nodiscard]] evmc::bytes32 get_transient_storage(
        const evmc::address& addr, const evmc::bytes32& key) const noexcept;
    void set_transient_storage(
        const evmc::address& addr, const evmc::bytes32& key, const evmc::bytes32& value);

    void clear_transient_storage() noexcept;

    // --- Snapshot / revert ---

    /// Take a snapshot of current state. Returns a snapshot ID.
    [[nodiscard]] int snapshot();

    /// Revert to a previously taken snapshot. All changes after the snapshot are undone.
    void revert(int snapshot_id);

    // --- Commit ---

    /// Compute state root hash. Currently a Keccak-256 over all RLP-encoded accounts.
    /// This is a placeholder until Merkle Patricia Trie is implemented.
    [[nodiscard]] evmc::bytes32 commit();

    // --- Accessors ---

    [[nodiscard]] const Account* get_account(const evmc::address& addr) const noexcept;

private:
    Account& get_or_create(const evmc::address& addr);

    std::unordered_map<evmc::address, Account> accounts_;
    StateStorage storage_;
    Journal journal_;

    /// EIP-2929 warm accounts and storage slots.
    std::unordered_set<evmc::address> warm_accounts_;
    std::unordered_map<evmc::address, std::unordered_set<evmc::bytes32>> warm_storage_;

    /// EIP-1153 transient storage (cleared per transaction).
    std::unordered_map<evmc::address, std::unordered_map<evmc::bytes32, evmc::bytes32>>
        transient_storage_;

    /// Sentinel for empty code returns.
    static const std::vector<uint8_t> empty_code_;
};

}  // namespace evm::state
