// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file gpu_state_db.hpp
/// GPU-native state database -- ALL state lives in GPU memory persistently.
///
/// On Apple Silicon (M1/M2/M3/M4), MTLResourceStorageModeShared means the
/// GPU hash table IS the state database. CPU reads it directly when needed
/// (e.g., for RPC queries). No CPU-side unordered_map for accounts or storage.
///
/// Architecture:
///   - Accounts:  GpuHashTable (persistent Metal buffer, open-addressing)
///   - Storage:   GpuHashTable (persistent Metal buffer, open-addressing)
///   - Code:      CPU-side map (bytecode is variable-length, poor fit for GPU HT)
///   - Journal:   CPU-side (only used for snapshot/revert within a block)
///   - Access lists / transient storage: CPU-side (per-transaction, cleared often)
///
/// The EVMC Host methods dispatch single-element GPU kernel calls for
/// individual reads/writes. For batch operations (block execution), the
/// parallel engine should use the batch lookup/insert methods directly.

#pragma once

#include "gpu_hashtable.hpp"
#include "storage.hpp"

#include <evmc/evmc.hpp>
#include <intx/intx.hpp>
#include <cstdint>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace evm::state
{

using intx::uint256;

/// GPU-native state database.
///
/// Accounts and storage live entirely in GPU memory (Metal buffers).
/// Code, journal, access lists, and transient storage remain on CPU
/// because they are either variable-length or per-transaction ephemeral.
class GpuNativeStateDB
{
public:
    /// Default capacity: 2^20 = ~1M account slots, ~1M storage slots.
    /// At 75% load factor, supports ~786K accounts and ~786K storage slots.
    static constexpr uint32_t DEFAULT_ACCOUNT_CAPACITY = 1u << 20;
    static constexpr uint32_t DEFAULT_STORAGE_CAPACITY = 1u << 22;  // 4M storage slots

    GpuNativeStateDB()
    {
        accounts_ = GpuHashTable::create(DEFAULT_ACCOUNT_CAPACITY);
        storage_ = GpuHashTable::create(DEFAULT_STORAGE_CAPACITY);
    }

    explicit GpuNativeStateDB(uint32_t account_capacity, uint32_t storage_capacity)
    {
        accounts_ = GpuHashTable::create(account_capacity);
        storage_ = GpuHashTable::create(storage_capacity);
    }

    /// Check if GPU tables are available.
    [[nodiscard]] bool gpu_available() const noexcept
    {
        return accounts_ != nullptr && storage_ != nullptr;
    }

    // --- Account existence ---

    [[nodiscard]] bool account_exists(const evmc::address& addr) const noexcept
    {
        GpuAccountData data{};
        uint32_t found = 0;
        const_cast<GpuHashTable*>(accounts_.get())->lookup_accounts(&addr, 1, &data, &found);
        return found != 0;
    }

    void create_account(const evmc::address& addr)
    {
        journal_.append(JournalAccountCreate{addr});

        GpuAccountData data{};
        data.nonce = 0;
        std::memset(data.balance, 0, 32);
        // code_hash = keccak256("") = empty_code_hash
        auto ech = empty_code_hash();
        std::memcpy(data.code_hash, ech.bytes, 32);
        std::memset(data.storage_root, 0, 32);
        accounts_->insert_accounts(&addr, &data, 1);
    }

    // --- Balance ---

    [[nodiscard]] uint256 get_balance(const evmc::address& addr) const noexcept
    {
        GpuAccountData data{};
        uint32_t found = 0;
        const_cast<GpuHashTable*>(accounts_.get())->lookup_accounts(&addr, 1, &data, &found);
        if (!found) return 0;

        uint256 result;
        std::memcpy(&result, data.balance, 32);
        return result;
    }

    void set_balance(const evmc::address& addr, const uint256& amount)
    {
        GpuAccountData data{};
        uint32_t found = 0;
        accounts_->lookup_accounts(&addr, 1, &data, &found);
        if (!found)
        {
            create_account(addr);
            accounts_->lookup_accounts(&addr, 1, &data, &found);
        }

        journal_.append(JournalBalanceChange{addr, get_balance(addr)});
        std::memcpy(data.balance, &amount, 32);
        accounts_->insert_accounts(&addr, &data, 1);
    }

    void add_balance(const evmc::address& addr, const uint256& amount)
    {
        set_balance(addr, get_balance(addr) + amount);
    }

    void sub_balance(const evmc::address& addr, const uint256& amount)
    {
        set_balance(addr, get_balance(addr) - amount);
    }

    // --- Nonce ---

    [[nodiscard]] uint64_t get_nonce(const evmc::address& addr) const noexcept
    {
        GpuAccountData data{};
        uint32_t found = 0;
        const_cast<GpuHashTable*>(accounts_.get())->lookup_accounts(&addr, 1, &data, &found);
        return found ? data.nonce : 0;
    }

    void set_nonce(const evmc::address& addr, uint64_t nonce)
    {
        GpuAccountData data{};
        uint32_t found = 0;
        accounts_->lookup_accounts(&addr, 1, &data, &found);
        if (!found)
        {
            create_account(addr);
            accounts_->lookup_accounts(&addr, 1, &data, &found);
        }

        journal_.append(JournalNonceChange{addr, data.nonce});
        data.nonce = nonce;
        accounts_->insert_accounts(&addr, &data, 1);
    }

    void increment_nonce(const evmc::address& addr)
    {
        set_nonce(addr, get_nonce(addr) + 1);
    }

    // --- Storage (GPU-native) ---

    [[nodiscard]] evmc::bytes32 get_storage(
        const evmc::address& addr, const evmc::bytes32& key) const noexcept
    {
        GpuStorageKey sk{};
        std::memcpy(sk.addr, addr.bytes, 20);
        std::memcpy(sk.slot, key.bytes, 32);

        evmc::bytes32 value{};
        uint32_t found = 0;
        const_cast<GpuHashTable*>(storage_.get())->lookup_storage(&sk, 1, &value, &found);
        return value;  // Returns zero if not found (buffer is zero-initialized).
    }

    void set_storage(
        const evmc::address& addr, const evmc::bytes32& key, const evmc::bytes32& value)
    {
        evmc::bytes32 prev = get_storage(addr, key);
        journal_.append(JournalStorageChange{addr, key, prev, prev != evmc::bytes32{}});

        GpuStorageKey sk{};
        std::memcpy(sk.addr, addr.bytes, 20);
        std::memcpy(sk.slot, key.bytes, 32);
        storage_->insert_storage(&sk, &value, 1);
    }

    // --- Code (CPU-side: variable-length bytecode) ---

    [[nodiscard]] const std::vector<uint8_t>& get_code(const evmc::address& addr) const noexcept
    {
        auto it = code_.find(addr);
        if (it != code_.end()) return it->second;
        return empty_code_;
    }

    [[nodiscard]] size_t get_code_size(const evmc::address& addr) const noexcept
    {
        return get_code(addr).size();
    }

    [[nodiscard]] evmc::bytes32 get_code_hash(const evmc::address& addr) const noexcept
    {
        GpuAccountData data{};
        uint32_t found = 0;
        const_cast<GpuHashTable*>(accounts_.get())->lookup_accounts(&addr, 1, &data, &found);
        if (!found) return {};
        evmc::bytes32 h{};
        std::memcpy(h.bytes, data.code_hash, 32);
        return h;
    }

    void set_code(const evmc::address& addr, std::vector<uint8_t> code)
    {
        // Update code_hash in the GPU account entry.
        GpuAccountData data{};
        uint32_t found = 0;
        accounts_->lookup_accounts(&addr, 1, &data, &found);
        if (!found)
        {
            create_account(addr);
            accounts_->lookup_accounts(&addr, 1, &data, &found);
        }

        evmc::bytes32 prev_hash{};
        std::memcpy(prev_hash.bytes, data.code_hash, 32);
        auto prev_code = std::move(code_[addr]);
        journal_.append(JournalCodeChange{addr, std::move(prev_code), prev_hash});

        if (code.empty())
        {
            auto ech = empty_code_hash();
            std::memcpy(data.code_hash, ech.bytes, 32);
        }
        else
        {
            const auto k = ethash::keccak256(code.data(), code.size());
            std::memcpy(data.code_hash, k.bytes, 32);
        }
        accounts_->insert_accounts(&addr, &data, 1);
        code_[addr] = std::move(code);
    }

    // --- Self-destruct ---

    void selfdestruct(const evmc::address& addr, const evmc::address& beneficiary)
    {
        auto bal = get_balance(addr);
        if (bal > 0)
        {
            add_balance(beneficiary, bal);
            set_balance(addr, 0);
        }
    }

    // --- Access lists (EIP-2929) ---

    [[nodiscard]] bool is_account_warm(const evmc::address& addr) const noexcept
    {
        return warm_accounts_.count(addr) > 0;
    }

    bool warm_account(const evmc::address& addr)
    {
        auto [_, inserted] = warm_accounts_.insert(addr);
        if (inserted)
            journal_.append(JournalAccessListAccount{addr});
        return !inserted;  // true if already warm
    }

    [[nodiscard]] bool is_storage_warm(
        const evmc::address& addr, const evmc::bytes32& key) const noexcept
    {
        auto it = warm_storage_.find(addr);
        return it != warm_storage_.end() && it->second.count(key) > 0;
    }

    bool warm_storage(const evmc::address& addr, const evmc::bytes32& key)
    {
        auto [_, inserted] = warm_storage_[addr].insert(key);
        if (inserted)
            journal_.append(JournalAccessListStorage{addr, key});
        return !inserted;
    }

    void clear_access_lists() noexcept
    {
        warm_accounts_.clear();
        warm_storage_.clear();
    }

    // --- Transient storage (EIP-1153) ---

    [[nodiscard]] evmc::bytes32 get_transient_storage(
        const evmc::address& addr, const evmc::bytes32& key) const noexcept
    {
        auto it = transient_storage_.find(addr);
        if (it == transient_storage_.end()) return {};
        auto jt = it->second.find(key);
        return jt != it->second.end() ? jt->second : evmc::bytes32{};
    }

    void set_transient_storage(
        const evmc::address& addr, const evmc::bytes32& key, const evmc::bytes32& value)
    {
        auto prev = get_transient_storage(addr, key);
        journal_.append(JournalTransientStorage{addr, key, prev});
        transient_storage_[addr][key] = value;
    }

    void clear_transient_storage() noexcept
    {
        transient_storage_.clear();
    }

    // --- Snapshot / revert ---

    [[nodiscard]] int snapshot()
    {
        return journal_.snapshot();
    }

    void revert(int snapshot_id)
    {
        auto [begin, end] = journal_.entries_since(snapshot_id);
        for (auto it = begin; it != end; ++it)
        {
            std::visit([this](const auto& entry) { undo(entry); }, *it);
        }
        journal_.truncate(snapshot_id);
    }

    // --- Commit (GPU state root) ---

    [[nodiscard]] evmc::bytes32 commit()
    {
        journal_.clear();
        if (accounts_)
            return accounts_->compute_state_root();
        return {};
    }

    // --- Direct GPU table access ---

    [[nodiscard]] GpuHashTable* account_table() noexcept { return accounts_.get(); }
    [[nodiscard]] GpuHashTable* storage_table() noexcept { return storage_.get(); }

    [[nodiscard]] const GpuHashTable* account_table() const noexcept { return accounts_.get(); }
    [[nodiscard]] const GpuHashTable* storage_table() const noexcept { return storage_.get(); }

private:
    std::unique_ptr<GpuHashTable> accounts_;
    std::unique_ptr<GpuHashTable> storage_;

    // Code stays on CPU (variable-length).
    std::unordered_map<evmc::address, std::vector<uint8_t>> code_;

    // Journal for snapshot/revert.
    Journal journal_;

    // EIP-2929 warm sets (CPU, per-transaction).
    std::unordered_set<evmc::address> warm_accounts_;
    std::unordered_map<evmc::address, std::unordered_set<evmc::bytes32>> warm_storage_;

    // EIP-1153 transient storage (CPU, per-transaction).
    std::unordered_map<evmc::address, std::unordered_map<evmc::bytes32, evmc::bytes32>>
        transient_storage_;

    static inline const std::vector<uint8_t> empty_code_{};

    // -- Journal undo methods ---

    void undo(const JournalBalanceChange& e)
    {
        GpuAccountData data{};
        uint32_t found = 0;
        accounts_->lookup_accounts(&e.addr, 1, &data, &found);
        if (found)
        {
            std::memcpy(data.balance, &e.prev_balance, 32);
            accounts_->insert_accounts(&e.addr, &data, 1);
        }
    }

    void undo(const JournalNonceChange& e)
    {
        GpuAccountData data{};
        uint32_t found = 0;
        accounts_->lookup_accounts(&e.addr, 1, &data, &found);
        if (found)
        {
            data.nonce = e.prev_nonce;
            accounts_->insert_accounts(&e.addr, &data, 1);
        }
    }

    void undo(const JournalStorageChange& e)
    {
        GpuStorageKey sk{};
        std::memcpy(sk.addr, e.addr.bytes, 20);
        std::memcpy(sk.slot, e.key.bytes, 32);
        storage_->insert_storage(&sk, &e.prev_value, 1);
    }

    void undo(const JournalCodeChange& e)
    {
        GpuAccountData data{};
        uint32_t found = 0;
        accounts_->lookup_accounts(&e.addr, 1, &data, &found);
        if (found)
        {
            std::memcpy(data.code_hash, e.prev_code_hash.bytes, 32);
            accounts_->insert_accounts(&e.addr, &data, 1);
        }
        if (e.prev_code.empty())
            code_.erase(e.addr);
        else
            code_[e.addr] = e.prev_code;
    }

    void undo(const JournalAccountCreate& e)
    {
        // Cannot truly delete from open-addressing table without tombstones.
        // Set the account to empty state (nonce=0, balance=0, empty code hash).
        GpuAccountData data{};
        auto ech = empty_code_hash();
        std::memcpy(data.code_hash, ech.bytes, 32);
        accounts_->insert_accounts(&e.addr, &data, 1);
    }

    void undo(const JournalAccountDestroy& e)
    {
        GpuAccountData data{};
        data.nonce = e.prev_nonce;
        std::memcpy(data.balance, &e.prev_balance, 32);
        std::memcpy(data.code_hash, e.prev_code_hash.bytes, 32);
        accounts_->insert_accounts(&e.addr, &data, 1);
        code_[e.addr] = e.prev_code;
        // Restore storage entries.
        for (const auto& [key, value] : e.prev_storage)
        {
            GpuStorageKey sk{};
            std::memcpy(sk.addr, e.addr.bytes, 20);
            std::memcpy(sk.slot, key.bytes, 32);
            storage_->insert_storage(&sk, &value, 1);
        }
    }

    void undo(const JournalAccessListAccount& e)
    {
        warm_accounts_.erase(e.addr);
    }

    void undo(const JournalAccessListStorage& e)
    {
        auto it = warm_storage_.find(e.addr);
        if (it != warm_storage_.end())
            it->second.erase(e.key);
    }

    void undo(const JournalTransientStorage& e)
    {
        transient_storage_[e.addr][e.key] = e.prev_value;
    }
};

}  // namespace evm::state
