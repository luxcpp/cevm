// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file storage.hpp
/// In-memory account storage with snapshot/revert journaling.
///
/// Pattern matches go-ethereum's journal.go: every state mutation is recorded
/// in a journal. Snapshots record journal positions. Revert replays the journal
/// backwards to undo changes.

#pragma once

#include <evmc/evmc.hpp>
#include <cstdint>
#include <unordered_map>
#include <variant>
#include <vector>

namespace evm::state
{

/// Per-account storage map: slot -> value.
using StorageMap = std::unordered_map<evmc::bytes32, evmc::bytes32>;

/// Full state storage: address -> (slot -> value).
using StateStorage = std::unordered_map<evmc::address, StorageMap>;

/// Journal entry types for undo operations.

struct JournalBalanceChange
{
    evmc::address addr;
    intx::uint256 prev_balance;
};

struct JournalNonceChange
{
    evmc::address addr;
    uint64_t prev_nonce;
};

struct JournalStorageChange
{
    evmc::address addr;
    evmc::bytes32 key;
    evmc::bytes32 prev_value;
    bool had_value;  ///< False if key didn't exist before.
};

struct JournalCodeChange
{
    evmc::address addr;
    std::vector<uint8_t> prev_code;
    evmc::bytes32 prev_code_hash;
};

struct JournalAccountCreate
{
    evmc::address addr;
};

struct JournalAccountDestroy
{
    evmc::address addr;
    uint64_t prev_nonce;
    intx::uint256 prev_balance;
    evmc::bytes32 prev_code_hash;
    std::vector<uint8_t> prev_code;
    StorageMap prev_storage;
};

struct JournalAccessListAccount
{
    evmc::address addr;
};

struct JournalAccessListStorage
{
    evmc::address addr;
    evmc::bytes32 key;
};

struct JournalTransientStorage
{
    evmc::address addr;
    evmc::bytes32 key;
    evmc::bytes32 prev_value;
};

/// Union of all journal entry types.
using JournalEntry = std::variant<
    JournalBalanceChange,
    JournalNonceChange,
    JournalStorageChange,
    JournalCodeChange,
    JournalAccountCreate,
    JournalAccountDestroy,
    JournalAccessListAccount,
    JournalAccessListStorage,
    JournalTransientStorage>;

/// Journal for tracking state mutations with snapshot/revert.
class Journal
{
public:
    /// Record a journal entry.
    void append(JournalEntry entry) { entries_.push_back(std::move(entry)); }

    /// Take a snapshot. Returns an ID that can be passed to revert().
    [[nodiscard]] int snapshot() noexcept
    {
        return static_cast<int>(entries_.size());
    }

    /// Get all entries after a snapshot point (in reverse order for undo).
    /// Caller is responsible for applying the undo logic.
    [[nodiscard]] auto entries_since(int snapshot_id) const
    {
        const auto start = static_cast<size_t>(snapshot_id);
        return std::make_pair(entries_.rbegin(),
                              entries_.rbegin() + static_cast<ptrdiff_t>(entries_.size() - start));
    }

    /// Truncate journal back to a snapshot point.
    void truncate(int snapshot_id)
    {
        entries_.resize(static_cast<size_t>(snapshot_id));
    }

    /// Clear the entire journal (after commit).
    void clear() noexcept { entries_.clear(); }

    [[nodiscard]] size_t size() const noexcept { return entries_.size(); }

private:
    std::vector<JournalEntry> entries_;
};

}  // namespace evm::state
