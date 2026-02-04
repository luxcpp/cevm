// Copyright (C) 2026, The evmone Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "mv_memory.hpp"
#include <algorithm>
#include <cstring>
#include <functional>

namespace evm::gpu
{

bool MemoryLocation::operator==(const MemoryLocation& other) const noexcept
{
    return std::memcmp(address, other.address, 20) == 0 &&
           std::memcmp(slot, other.slot, 32) == 0;
}

size_t MemoryLocationHash::operator()(const MemoryLocation& loc) const noexcept
{
    // FNV-1a over address + slot (52 bytes total)
    size_t hash = 14695981039346656037ULL;
    const auto* data = reinterpret_cast<const uint8_t*>(&loc);
    for (size_t i = 0; i < sizeof(MemoryLocation); ++i)
    {
        hash ^= data[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

MvMemory::MvMemory(uint32_t num_txs) : num_txs_{num_txs} {}

void MvMemory::write(const MemoryLocation& loc, uint32_t tx_index, uint32_t incarnation,
                     const uint8_t* value, size_t value_len)
{
    std::unique_lock lock(mu_);
    auto& chain = data_[loc];

    // Find or insert entry for this (tx_index, incarnation)
    auto it = std::find_if(chain.begin(), chain.end(),
        [tx_index](const VersionEntry& e) { return e.version.tx_index == tx_index; });

    if (it != chain.end())
    {
        it->version.incarnation = incarnation;
        it->value.assign(value, value + value_len);
        it->is_estimate = false;
    }
    else
    {
        chain.push_back({{tx_index, incarnation}, {value, value + value_len}, false});
        // Keep sorted by tx_index for binary search in read()
        std::sort(chain.begin(), chain.end(),
            [](const VersionEntry& a, const VersionEntry& b) {
                return a.version.tx_index < b.version.tx_index;
            });
    }
}

std::optional<std::pair<Version, std::vector<uint8_t>>>
MvMemory::read(const MemoryLocation& loc, uint32_t tx_index) const
{
    std::shared_lock lock(mu_);
    auto it = data_.find(loc);
    if (it == data_.end())
        return std::nullopt;

    const auto& chain = it->second;

    // Find the latest write by a transaction with index < tx_index
    const VersionEntry* best = nullptr;
    for (const auto& entry : chain)
    {
        if (entry.version.tx_index < tx_index && !entry.is_estimate)
        {
            if (!best || entry.version.tx_index > best->version.tx_index)
                best = &entry;
        }
    }

    if (!best)
        return std::nullopt;

    return std::make_pair(best->version, best->value);
}

void MvMemory::mark_estimate(uint32_t tx_index, uint32_t /*incarnation*/)
{
    std::unique_lock lock(mu_);
    for (auto& [loc, chain] : data_)
    {
        for (auto& entry : chain)
        {
            if (entry.version.tx_index == tx_index)
                entry.is_estimate = true;
        }
    }
}

bool MvMemory::validate_read(const MemoryLocation& loc, uint32_t tx_index,
                             const Version& read_version) const
{
    auto result = read(loc, tx_index);
    if (!result)
    {
        // No write exists now — valid only if we also read nothing
        return read_version.tx_index == UINT32_MAX;
    }

    bool valid = result->first.tx_index == read_version.tx_index &&
                 result->first.incarnation == read_version.incarnation;
    if (!valid)
        conflicts_.fetch_add(1, std::memory_order_relaxed);

    return valid;
}

}  // namespace evm::gpu
