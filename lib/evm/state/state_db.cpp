// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "state_db.hpp"

#include <evmone_precompiles/keccak.hpp>
#include <algorithm>
#include <cassert>

namespace evm::state
{

const std::vector<uint8_t> StateDB::empty_code_{};

// --- Account existence ---

bool StateDB::account_exists(const evmc::address& addr) const noexcept
{
    return accounts_.contains(addr);
}

void StateDB::create_account(const evmc::address& addr)
{
    if (accounts_.contains(addr))
        return;
    accounts_.emplace(addr, Account{});
    journal_.append(JournalAccountCreate{addr});
}

// --- Balance ---

uint256 StateDB::get_balance(const evmc::address& addr) const noexcept
{
    if (const auto it = accounts_.find(addr); it != accounts_.end())
        return it->second.balance;
    return 0;
}

void StateDB::set_balance(const evmc::address& addr, const uint256& amount)
{
    auto& acct = get_or_create(addr);
    journal_.append(JournalBalanceChange{addr, acct.balance});
    acct.balance = amount;
}

void StateDB::add_balance(const evmc::address& addr, const uint256& amount)
{
    if (amount == 0)
        return;
    auto& acct = get_or_create(addr);
    journal_.append(JournalBalanceChange{addr, acct.balance});
    acct.balance += amount;
}

void StateDB::sub_balance(const evmc::address& addr, const uint256& amount)
{
    if (amount == 0)
        return;
    auto& acct = get_or_create(addr);
    assert(acct.balance >= amount);
    journal_.append(JournalBalanceChange{addr, acct.balance});
    acct.balance -= amount;
}

// --- Nonce ---

uint64_t StateDB::get_nonce(const evmc::address& addr) const noexcept
{
    if (const auto it = accounts_.find(addr); it != accounts_.end())
        return it->second.nonce;
    return 0;
}

void StateDB::set_nonce(const evmc::address& addr, uint64_t nonce)
{
    auto& acct = get_or_create(addr);
    journal_.append(JournalNonceChange{addr, acct.nonce});
    acct.nonce = nonce;
}

void StateDB::increment_nonce(const evmc::address& addr)
{
    auto& acct = get_or_create(addr);
    journal_.append(JournalNonceChange{addr, acct.nonce});
    ++acct.nonce;
}

// --- Storage ---

evmc::bytes32 StateDB::get_storage(
    const evmc::address& addr, const evmc::bytes32& key) const noexcept
{
    if (const auto it = storage_.find(addr); it != storage_.end())
    {
        if (const auto sit = it->second.find(key); sit != it->second.end())
            return sit->second;
    }
    return {};
}

void StateDB::set_storage(
    const evmc::address& addr, const evmc::bytes32& key, const evmc::bytes32& value)
{
    auto& slot_map = storage_[addr];
    const auto it = slot_map.find(key);
    const bool had = (it != slot_map.end());
    const auto prev = had ? it->second : evmc::bytes32{};

    journal_.append(JournalStorageChange{addr, key, prev, had});
    slot_map[key] = value;
}

// --- Code ---

const std::vector<uint8_t>& StateDB::get_code(const evmc::address& addr) const noexcept
{
    if (const auto it = accounts_.find(addr); it != accounts_.end())
        return it->second.code;
    return empty_code_;
}

size_t StateDB::get_code_size(const evmc::address& addr) const noexcept
{
    if (const auto it = accounts_.find(addr); it != accounts_.end())
        return it->second.code.size();
    return 0;
}

evmc::bytes32 StateDB::get_code_hash(const evmc::address& addr) const noexcept
{
    if (const auto it = accounts_.find(addr); it != accounts_.end())
        return it->second.code_hash;
    return {};
}

void StateDB::set_code(const evmc::address& addr, std::vector<uint8_t> code)
{
    auto& acct = get_or_create(addr);
    journal_.append(JournalCodeChange{addr, acct.code, acct.code_hash});

    if (code.empty())
    {
        acct.code_hash = empty_code_hash();
    }
    else
    {
        const auto h = ethash::keccak256(code.data(), code.size());
        __builtin_memcpy(acct.code_hash.bytes, h.bytes, 32);
    }
    acct.code = std::move(code);
}

// --- Self-destruct ---

void StateDB::selfdestruct(const evmc::address& addr, const evmc::address& beneficiary)
{
    const auto it = accounts_.find(addr);
    if (it == accounts_.end())
        return;

    // Transfer remaining balance to beneficiary.
    const auto& acct = it->second;
    if (acct.balance > 0)
        add_balance(beneficiary, acct.balance);

    // Record for journal revert.
    StorageMap prev_storage;
    if (auto sit = storage_.find(addr); sit != storage_.end())
        prev_storage = sit->second;

    journal_.append(JournalAccountDestroy{
        addr, acct.nonce, acct.balance, acct.code_hash, acct.code, std::move(prev_storage)});

    // Wipe account.
    accounts_.erase(addr);
    storage_.erase(addr);
}

// --- Access lists (EIP-2929) ---

bool StateDB::is_account_warm(const evmc::address& addr) const noexcept
{
    return warm_accounts_.contains(addr);
}

bool StateDB::warm_account(const evmc::address& addr)
{
    const auto [_, inserted] = warm_accounts_.insert(addr);
    if (inserted)
        journal_.append(JournalAccessListAccount{addr});
    return !inserted;  // Return true if was already warm.
}

bool StateDB::is_storage_warm(
    const evmc::address& addr, const evmc::bytes32& key) const noexcept
{
    if (const auto it = warm_storage_.find(addr); it != warm_storage_.end())
        return it->second.contains(key);
    return false;
}

bool StateDB::warm_storage(const evmc::address& addr, const evmc::bytes32& key)
{
    const auto [_, inserted] = warm_storage_[addr].insert(key);
    if (inserted)
        journal_.append(JournalAccessListStorage{addr, key});
    return !inserted;  // Return true if was already warm.
}

void StateDB::clear_access_lists() noexcept
{
    warm_accounts_.clear();
    warm_storage_.clear();
}

// --- Transient storage (EIP-1153) ---

evmc::bytes32 StateDB::get_transient_storage(
    const evmc::address& addr, const evmc::bytes32& key) const noexcept
{
    if (const auto it = transient_storage_.find(addr); it != transient_storage_.end())
    {
        if (const auto sit = it->second.find(key); sit != it->second.end())
            return sit->second;
    }
    return {};
}

void StateDB::set_transient_storage(
    const evmc::address& addr, const evmc::bytes32& key, const evmc::bytes32& value)
{
    auto& slot_map = transient_storage_[addr];
    const auto prev = get_transient_storage(addr, key);
    journal_.append(JournalTransientStorage{addr, key, prev});
    slot_map[key] = value;
}

void StateDB::clear_transient_storage() noexcept
{
    transient_storage_.clear();
}

// --- Snapshot / revert ---

int StateDB::snapshot()
{
    return journal_.snapshot();
}

void StateDB::revert(int snapshot_id)
{
    auto [begin, end] = journal_.entries_since(snapshot_id);
    for (auto it = begin; it != end; ++it)
    {
        std::visit([this](auto&& entry) {
            using T = std::decay_t<decltype(entry)>;

            if constexpr (std::is_same_v<T, JournalBalanceChange>)
            {
                if (auto ait = accounts_.find(entry.addr); ait != accounts_.end())
                    ait->second.balance = entry.prev_balance;
            }
            else if constexpr (std::is_same_v<T, JournalNonceChange>)
            {
                if (auto ait = accounts_.find(entry.addr); ait != accounts_.end())
                    ait->second.nonce = entry.prev_nonce;
            }
            else if constexpr (std::is_same_v<T, JournalStorageChange>)
            {
                if (entry.had_value)
                    storage_[entry.addr][entry.key] = entry.prev_value;
                else
                    storage_[entry.addr].erase(entry.key);
            }
            else if constexpr (std::is_same_v<T, JournalCodeChange>)
            {
                if (auto ait = accounts_.find(entry.addr); ait != accounts_.end())
                {
                    ait->second.code = entry.prev_code;
                    ait->second.code_hash = entry.prev_code_hash;
                }
            }
            else if constexpr (std::is_same_v<T, JournalAccountCreate>)
            {
                accounts_.erase(entry.addr);
            }
            else if constexpr (std::is_same_v<T, JournalAccountDestroy>)
            {
                auto& acct = accounts_[entry.addr];
                acct.nonce = entry.prev_nonce;
                acct.balance = entry.prev_balance;
                acct.code_hash = entry.prev_code_hash;
                acct.code = entry.prev_code;
                storage_[entry.addr] = entry.prev_storage;
            }
            else if constexpr (std::is_same_v<T, JournalAccessListAccount>)
            {
                warm_accounts_.erase(entry.addr);
            }
            else if constexpr (std::is_same_v<T, JournalAccessListStorage>)
            {
                if (auto ait = warm_storage_.find(entry.addr); ait != warm_storage_.end())
                    ait->second.erase(entry.key);
            }
            else if constexpr (std::is_same_v<T, JournalTransientStorage>)
            {
                transient_storage_[entry.addr][entry.key] = entry.prev_value;
            }
        }, *it);
    }
    journal_.truncate(snapshot_id);
}

// --- Commit ---

evmc::bytes32 StateDB::commit()
{
    // Compute a deterministic state root by hashing all accounts in sorted order.
    // This is NOT a Merkle Patricia Trie. It is a flat hash sufficient for correctness
    // testing and benchmarking. The MPT can be layered in later.

    // Collect and sort addresses for deterministic ordering.
    std::vector<evmc::address> addrs;
    addrs.reserve(accounts_.size());
    for (const auto& [addr, _] : accounts_)
        addrs.push_back(addr);
    std::sort(addrs.begin(), addrs.end());

    // Build a blob: for each account, RLP-encode it and append.
    std::vector<uint8_t> blob;
    blob.reserve(addrs.size() * 140);

    for (const auto& addr : addrs)
    {
        // Include address in the blob.
        blob.insert(blob.end(), addr.bytes, addr.bytes + 20);
        const auto encoded = rlp_encode(accounts_.at(addr));
        blob.insert(blob.end(), encoded.begin(), encoded.end());
    }

    // Hash the blob.
    evmc::bytes32 root{};
    if (!blob.empty())
    {
        const auto h = ethash::keccak256(blob.data(), blob.size());
        __builtin_memcpy(root.bytes, h.bytes, 32);
    }

    // Clear journal after commit.
    journal_.clear();
    return root;
}

// --- Internal ---

const Account* StateDB::get_account(const evmc::address& addr) const noexcept
{
    if (const auto it = accounts_.find(addr); it != accounts_.end())
        return &it->second;
    return nullptr;
}

Account& StateDB::get_or_create(const evmc::address& addr)
{
    auto it = accounts_.find(addr);
    if (it == accounts_.end())
    {
        it = accounts_.emplace(addr, Account{}).first;
        journal_.append(JournalAccountCreate{addr});
    }
    return it->second;
}

}  // namespace evm::state
