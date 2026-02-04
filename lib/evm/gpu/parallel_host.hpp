// Copyright (C) 2026, The evmone Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file parallel_host.hpp
/// EVMC Host adapter for Block-STM parallel execution.
///
/// Wraps a base Host and intercepts storage reads/writes,
/// routing them through MvMemory. Records read-set and write-set
/// for conflict detection and validation.

#pragma once

#include "mv_memory.hpp"
#include <evmc/evmc.hpp>
#include <cstring>
#include <vector>

namespace evm::gpu
{

/// A recorded storage read: location + the version that was read.
struct ReadEntry
{
    MemoryLocation location;
    Version version;  ///< Version seen at read time. tx_index=UINT32_MAX means "from base state".
};

/// A recorded storage write: location + value written.
struct WriteEntry
{
    MemoryLocation location;
    evmc::bytes32 value;
};

/// Per-transaction read/write sets accumulated during execution.
struct ReadWriteSet
{
    std::vector<ReadEntry> reads;
    std::vector<WriteEntry> writes;

    void clear()
    {
        reads.clear();
        writes.clear();
    }
};

/// EVMC Host that intercepts storage access for Block-STM.
///
/// - get_storage: reads from MvMemory (latest version < tx_index), falls back to base host.
/// - set_storage: writes to MvMemory, records in write-set.
/// - All other calls delegate to base host unchanged.
///
/// This is the critical integration point between Block-STM and evmone.
class ParallelHost : public evmc::Host
{
public:
    /// @param base_host    The underlying state database host.
    /// @param mv_memory    Shared multi-version memory for this block.
    /// @param tx_index     Index of the transaction this host serves.
    /// @param incarnation  Current incarnation of this transaction.
    ParallelHost(evmc::Host& base_host, MvMemory& mv_memory,
                 uint32_t tx_index, uint32_t incarnation) noexcept
      : base_host_{base_host},
        mv_memory_{mv_memory},
        tx_index_{tx_index},
        incarnation_{incarnation}
    {}

    /// Reset for a new incarnation (re-execution after conflict).
    void reset(uint32_t incarnation) noexcept
    {
        incarnation_ = incarnation;
        rw_set_.clear();
    }

    /// Get the accumulated read/write set.
    const ReadWriteSet& rw_set() const noexcept { return rw_set_; }

    /// Flush all recorded writes to MvMemory.
    /// Called after successful execution, before validation.
    void flush_writes() noexcept
    {
        for (const auto& w : rw_set_.writes)
        {
            mv_memory_.write(w.location, tx_index_, incarnation_,
                             w.value.bytes, sizeof(w.value.bytes));
        }
    }

    /// Validate all recorded reads against current MvMemory state.
    /// Returns true if all reads are still valid (no conflicts).
    bool validate_reads() const noexcept
    {
        for (const auto& r : rw_set_.reads)
        {
            if (!mv_memory_.validate_read(r.location, tx_index_, r.version))
                return false;
        }
        return true;
    }

    // --- EVMC Host interface ---

    bool account_exists(const evmc::address& addr) const noexcept override
    {
        return base_host_.account_exists(addr);
    }

    evmc::bytes32 get_storage(const evmc::address& addr,
                              const evmc::bytes32& key) const noexcept override
    {
        MemoryLocation loc{};
        std::memcpy(loc.address, addr.bytes, 20);
        std::memcpy(loc.slot, key.bytes, 32);

        // Try MvMemory first (reads from transactions with index < tx_index)
        auto result = mv_memory_.read(loc, tx_index_);
        if (result)
        {
            // Record the read with the version we saw
            rw_set_.reads.push_back({loc, result->first});

            evmc::bytes32 val{};
            const auto& data = result->second;
            if (data.size() >= 32)
                std::memcpy(val.bytes, data.data(), 32);
            else if (!data.empty())
                std::memcpy(val.bytes + (32 - data.size()), data.data(), data.size());
            return val;
        }

        // Fall back to base state
        auto val = base_host_.get_storage(addr, key);

        // Record read from base state (version sentinel = UINT32_MAX)
        rw_set_.reads.push_back({loc, {UINT32_MAX, 0}});

        return val;
    }

    evmc_storage_status set_storage(const evmc::address& addr,
                                    const evmc::bytes32& key,
                                    const evmc::bytes32& value) noexcept override
    {
        MemoryLocation loc{};
        std::memcpy(loc.address, addr.bytes, 20);
        std::memcpy(loc.slot, key.bytes, 32);

        // Record the write
        rw_set_.writes.push_back({loc, value});

        // Write immediately to MvMemory so later transactions in this block can see it
        mv_memory_.write(loc, tx_index_, incarnation_, value.bytes, 32);

        // Return a generic status. Accurate status requires knowing the previous value,
        // which matters for gas refunds. For Block-STM we use EVMC_STORAGE_MODIFIED
        // as a safe default; the final sequential commit pass computes exact statuses.
        return EVMC_STORAGE_MODIFIED;
    }

    evmc::uint256be get_balance(const evmc::address& addr) const noexcept override
    {
        // Balance tracked through storage-like MvMemory entries for full accuracy
        // would require balance-specific location keys. For now, delegate to base host.
        // This is correct for simple transfers where balance is not read by EVM code.
        return base_host_.get_balance(addr);
    }

    size_t get_code_size(const evmc::address& addr) const noexcept override
    {
        return base_host_.get_code_size(addr);
    }

    evmc::bytes32 get_code_hash(const evmc::address& addr) const noexcept override
    {
        return base_host_.get_code_hash(addr);
    }

    size_t copy_code(const evmc::address& addr, size_t code_offset,
                     uint8_t* buffer_data, size_t buffer_size) const noexcept override
    {
        return base_host_.copy_code(addr, code_offset, buffer_data, buffer_size);
    }

    bool selfdestruct(const evmc::address& addr,
                      const evmc::address& beneficiary) noexcept override
    {
        return base_host_.selfdestruct(addr, beneficiary);
    }

    evmc::Result call(const evmc_message& msg) noexcept override
    {
        // Internal calls: delegate to base host.
        // A full implementation would create a nested ParallelHost for re-entrant calls,
        // but for simple transfers this path is not taken.
        return base_host_.call(msg);
    }

    evmc_tx_context get_tx_context() const noexcept override
    {
        return base_host_.get_tx_context();
    }

    evmc::bytes32 get_block_hash(int64_t block_number) const noexcept override
    {
        return base_host_.get_block_hash(block_number);
    }

    void emit_log(const evmc::address& addr, const uint8_t* data, size_t data_size,
                  const evmc::bytes32 topics[], size_t num_topics) noexcept override
    {
        base_host_.emit_log(addr, data, data_size, topics, num_topics);
    }

    evmc_access_status access_account(const evmc::address& addr) noexcept override
    {
        return base_host_.access_account(addr);
    }

    evmc_access_status access_storage(const evmc::address& addr,
                                      const evmc::bytes32& key) noexcept override
    {
        return base_host_.access_storage(addr, key);
    }

    evmc::bytes32 get_transient_storage(const evmc::address& addr,
                                        const evmc::bytes32& key) const noexcept override
    {
        return base_host_.get_transient_storage(addr, key);
    }

    void set_transient_storage(const evmc::address& addr,
                               const evmc::bytes32& key,
                               const evmc::bytes32& value) noexcept override
    {
        base_host_.set_transient_storage(addr, key, value);
    }

private:
    evmc::Host& base_host_;
    MvMemory& mv_memory_;
    uint32_t tx_index_;
    uint32_t incarnation_;
    mutable ReadWriteSet rw_set_;
};

}  // namespace evm::gpu
