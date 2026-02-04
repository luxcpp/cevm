// Copyright (C) 2026, The evmone Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file mv_memory.hpp
/// Multi-Version Memory for Block-STM parallel execution.
///
/// Port of ~/work/lux/evmgpu/core/parallel/mv_memory.go to C++.
/// Each storage location (address + slot) maintains a version chain
/// indexed by (tx_index, incarnation). Reads check for conflicts
/// by comparing the version they read against the latest write.

#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

namespace evm::gpu
{

/// A memory location is (address, storage_slot).
struct MemoryLocation
{
    uint8_t address[20];
    uint8_t slot[32];

    bool operator==(const MemoryLocation& other) const noexcept;
};

/// Hash function for MemoryLocation.
struct MemoryLocationHash
{
    size_t operator()(const MemoryLocation& loc) const noexcept;
};

/// Version identifier: which transaction incarnation wrote this value.
struct Version
{
    uint32_t tx_index;
    uint32_t incarnation;
};

/// A single entry in the version chain.
struct VersionEntry
{
    Version version;
    std::vector<uint8_t> value;  ///< 32-byte storage value
    bool is_estimate = false;    ///< True if this is a speculative write (may be invalid)
};

/// Multi-version data structure for Block-STM.
///
/// Thread-safe. Multiple workers can read/write concurrently.
/// Each storage location has an ordered chain of versions.
class MvMemory
{
public:
    explicit MvMemory(uint32_t num_txs);

    /// Record a write from transaction tx_index at incarnation.
    void write(const MemoryLocation& loc, uint32_t tx_index, uint32_t incarnation,
               const uint8_t* value, size_t value_len);

    /// Read the latest value written by a transaction with index < tx_index.
    /// Returns nullopt if no prior write exists (read from base state).
    std::optional<std::pair<Version, std::vector<uint8_t>>>
    read(const MemoryLocation& loc, uint32_t tx_index) const;

    /// Mark all writes from (tx_index, incarnation) as estimates.
    /// Called when a transaction is scheduled for re-execution.
    void mark_estimate(uint32_t tx_index, uint32_t incarnation);

    /// Check if a read at (loc, tx_index) is still valid.
    /// Returns false if the value has been overwritten since the read.
    bool validate_read(const MemoryLocation& loc, uint32_t tx_index,
                       const Version& read_version) const;

    /// Get conflict statistics.
    uint32_t num_conflicts() const { return conflicts_.load(std::memory_order_relaxed); }

private:
    using VersionChain = std::vector<VersionEntry>;
    mutable std::shared_mutex mu_;
    std::unordered_map<MemoryLocation, VersionChain, MemoryLocationHash> data_;
    mutable std::atomic<uint32_t> conflicts_{0};
    uint32_t num_txs_;
};

}  // namespace evm::gpu
