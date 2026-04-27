// Copyright (C) 2026, The cevm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "conflict_spec.hpp"

#include <algorithm>
#include <cstring>
#include <list>
#include <unordered_map>

namespace evm::stm
{

namespace
{
    /// Empty-spec helper — used when a source declines to compose.
    ConflictSpec make_declared_empty(uint32_t tx_id)
    {
        ConflictSpec s{};
        s.tx_id = tx_id;
        const auto e = ConflictArena::empty_lanes();
        s.read_lane_offset    = e.first;
        s.read_lane_count     = e.second;
        s.write_lane_offset   = e.first;
        s.write_lane_count    = e.second;
        s.reducer_lane_offset = e.first;
        s.reducer_lane_count  = e.second;
        s.confidence = 0;
        s.source = static_cast<uint8_t>(ConflictSource::Declared);
        return s;
    }

    /// Read big-endian uint16 from raw bytes.
    inline uint32_t be_u32(const uint8_t* p) noexcept
    {
        return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16)
             | (uint32_t(p[2]) << 8)  |  uint32_t(p[3]);
    }

    /// Mainnet precompile range: 0x01..0x11 inclusive (Cancun + Prague).
    /// Address = 19 zero bytes + low byte in [1..0x11].
    bool is_precompile_address(const uint8_t addr[20]) noexcept
    {
        for (int i = 0; i < 19; ++i)
            if (addr[i] != 0) return false;
        return addr[19] >= 0x01 && addr[19] <= 0x11;
    }

    /// Compute slot for ERC-20 balance: keccak256(abi.encode(holder, slot=0)).
    /// We APPROXIMATE this without keccak256 here — the historical/abi
    /// sources only need a *deterministic* slot fingerprint, not the actual
    /// chain slot, because Prism placement keys on the lane fingerprint
    /// alone. Validation against MvMemory continues to use the real slot
    /// computed by the EVM. So: encode (holder, mapping_id) → 32-byte
    /// deterministic identifier. Collisions only hurt placement, not
    /// correctness — and across distinct holders the encoding is unique.
    void encode_mapping_slot(uint8_t out[32], const uint8_t holder[20], uint8_t mapping_id) noexcept
    {
        std::memset(out, 0, 32);
        out[0] = mapping_id;
        std::memcpy(out + 12, holder, 20);
    }
}  // namespace

// ===========================================================================
// Source 1 — EIP-2930 access list parser
// ===========================================================================
ConflictSpec compose_from_access_list(
    uint32_t tx_id,
    const uint8_t* warm_addresses, size_t warm_addresses_len,
    const uint8_t* warm_storage_keys, size_t warm_storage_keys_len,
    ConflictArena& arena)
{
    auto spec = make_declared_empty(tx_id);

    // Each address is 20 bytes, each storage key is (20 addr + 32 slot) = 52 bytes.
    if ((warm_addresses_len % 20) != 0) return spec;
    if ((warm_storage_keys_len % 52) != 0) return spec;
    if (warm_addresses_len == 0 && warm_storage_keys_len == 0) return spec;

    const auto n_addr = warm_addresses_len / 20;
    const auto n_slot = warm_storage_keys_len / 52;
    const auto total = n_addr + n_slot;

    std::vector<ConflictLane> tmp;
    tmp.reserve(total);

    // Address-only entries: lane with slot=zero.
    for (size_t i = 0; i < n_addr; ++i)
    {
        ConflictLane lane{};
        std::memcpy(lane.address, warm_addresses + i * 20, 20);
        // slot already zero from {}-init
        tmp.push_back(lane);
    }

    // (addr, slot) entries.
    for (size_t i = 0; i < n_slot; ++i)
    {
        ConflictLane lane{};
        std::memcpy(lane.address, warm_storage_keys + i * 52, 20);
        std::memcpy(lane.slot, warm_storage_keys + i * 52 + 20, 32);
        tmp.push_back(lane);
    }

    auto [off, cnt] = arena.push_lanes(tmp.data(), static_cast<uint16_t>(tmp.size()));
    spec.read_lane_offset = off;
    spec.read_lane_count = cnt;
    spec.confidence = 200;
    spec.source = static_cast<uint8_t>(ConflictSource::AccessList);
    return spec;
}

// ===========================================================================
// Source 2 — ABI selector table
// ===========================================================================
namespace
{
    enum class AbiKind : uint8_t
    {
        Unknown,
        Erc20Transfer,
        Erc20TransferFrom,
        Erc20Approve,
        UniV2Swap,
        UniV3ExactInputSingle,
        Erc721SafeTransferFrom,
    };

    AbiKind classify_selector(uint32_t selector) noexcept
    {
        switch (selector)
        {
            case 0xa9059cbbu: return AbiKind::Erc20Transfer;
            case 0x23b872ddu: return AbiKind::Erc20TransferFrom;   // also ERC-721; resolved by code class — caller hint
            case 0x095ea7b3u: return AbiKind::Erc20Approve;
            case 0x022c0d9fu: return AbiKind::UniV2Swap;
            case 0x414bf389u: return AbiKind::UniV3ExactInputSingle;
            case 0x42842e6eu: return AbiKind::Erc721SafeTransferFrom;
            default: return AbiKind::Unknown;
        }
    }

    /// Extract a 20-byte address from a 32-byte ABI-encoded word at offset.
    /// Returns false if calldata too short.
    bool read_addr_word(const uint8_t* calldata, size_t len, size_t word_off, uint8_t out[20])
    {
        const auto need = word_off + 32;
        if (len < need) return false;
        // ABI addresses are right-aligned in a 32-byte word.
        std::memcpy(out, calldata + word_off + 12, 20);
        return true;
    }
}  // namespace

ConflictSpec compose_from_abi(
    uint32_t tx_id,
    const uint8_t recipient[20],
    const uint8_t* calldata, size_t calldata_len,
    const uint8_t sender[20],
    ConflictArena& arena)
{
    auto spec = make_declared_empty(tx_id);
    if (recipient == nullptr || calldata_len < 4) return spec;

    const auto sel = be_u32(calldata);
    const auto kind = classify_selector(sel);
    if (kind == AbiKind::Unknown) return spec;

    std::vector<ConflictLane> reads;
    std::vector<ConflictLane> writes;
    reads.reserve(4);
    writes.reserve(4);

    // ERC-20 / ERC-721 standard storage layout assumption:
    //   slot 0: balances mapping (mapping_id = 0)
    //   slot 1: allowances mapping (mapping_id = 1)
    //   slot 2: owners mapping (ERC-721) / totalSupply (mapping_id = 2)
    constexpr uint8_t SLOT_BALANCES   = 0;
    constexpr uint8_t SLOT_ALLOWANCES = 1;
    constexpr uint8_t SLOT_OWNERS     = 2;

    auto make_lane = [&](const uint8_t addr[20], uint8_t mapping_id, const uint8_t holder[20]) -> ConflictLane
    {
        ConflictLane l{};
        std::memcpy(l.address, addr, 20);
        encode_mapping_slot(l.slot, holder, mapping_id);
        return l;
    };

    switch (kind)
    {
        case AbiKind::Erc20Transfer:
        {
            // transfer(address to, uint256)
            uint8_t to[20];
            if (!read_addr_word(calldata + 4, calldata_len - 4, 0, to)) return spec;
            // R/W: balances[sender], balances[to]
            writes.push_back(make_lane(recipient, SLOT_BALANCES, sender));
            writes.push_back(make_lane(recipient, SLOT_BALANCES, to));
            reads.push_back(make_lane(recipient, SLOT_BALANCES, sender));
            reads.push_back(make_lane(recipient, SLOT_BALANCES, to));
            break;
        }
        case AbiKind::Erc20TransferFrom:
        {
            // transferFrom(address from, address to, uint256)
            uint8_t from[20], to[20];
            if (!read_addr_word(calldata + 4, calldata_len - 4, 0, from)) return spec;
            if (!read_addr_word(calldata + 4, calldata_len - 4, 32, to)) return spec;
            // R/W: balances[from], balances[to], allowances[from][sender]
            writes.push_back(make_lane(recipient, SLOT_BALANCES, from));
            writes.push_back(make_lane(recipient, SLOT_BALANCES, to));
            writes.push_back(make_lane(recipient, SLOT_ALLOWANCES, from));
            reads = writes;  // every write is also a read
            break;
        }
        case AbiKind::Erc20Approve:
        {
            // approve(address spender, uint256)
            uint8_t spender[20];
            if (!read_addr_word(calldata + 4, calldata_len - 4, 0, spender)) return spec;
            // W: allowances[sender][spender]
            writes.push_back(make_lane(recipient, SLOT_ALLOWANCES, sender));
            // R: same slot (overwrite semantics)
            reads.push_back(make_lane(recipient, SLOT_ALLOWANCES, sender));
            break;
        }
        case AbiKind::UniV2Swap:
        {
            // Pair contract: reads/writes reserves slot (slot 8 in canonical V2)
            ConflictLane reserves{};
            std::memcpy(reserves.address, recipient, 20);
            reserves.slot[31] = 8;  // packed (reserve0, reserve1, blockTimestampLast)
            writes.push_back(reserves);
            reads.push_back(reserves);
            break;
        }
        case AbiKind::UniV3ExactInputSingle:
        {
            // Router calls into pool — we can only declare router-level reads;
            // pool-level lanes need code-hash dispatch (Historical takes over).
            ConflictLane router_state{};
            std::memcpy(router_state.address, recipient, 20);
            reads.push_back(router_state);
            break;
        }
        case AbiKind::Erc721SafeTransferFrom:
        {
            // safeTransferFrom(address from, address to, uint256 tokenId)
            uint8_t from[20], to[20];
            if (!read_addr_word(calldata + 4, calldata_len - 4, 0, from)) return spec;
            if (!read_addr_word(calldata + 4, calldata_len - 4, 32, to)) return spec;
            writes.push_back(make_lane(recipient, SLOT_OWNERS, from));
            writes.push_back(make_lane(recipient, SLOT_OWNERS, to));
            writes.push_back(make_lane(recipient, SLOT_BALANCES, from));
            writes.push_back(make_lane(recipient, SLOT_BALANCES, to));
            reads = writes;
            break;
        }
        case AbiKind::Unknown: return spec;
    }

    auto [r_off, r_cnt] = arena.push_lanes(reads.data(), static_cast<uint16_t>(reads.size()));
    auto [w_off, w_cnt] = arena.push_lanes(writes.data(), static_cast<uint16_t>(writes.size()));
    spec.read_lane_offset = r_off;
    spec.read_lane_count = r_cnt;
    spec.write_lane_offset = w_off;
    spec.write_lane_count = w_cnt;
    spec.confidence = 180;
    spec.source = static_cast<uint8_t>(ConflictSource::ABI);
    return spec;
}

// ===========================================================================
// Source 3 — Historical profile (LRU)
// ===========================================================================
namespace
{
    struct HistKey
    {
        uint8_t code_hash[32];
        uint32_t selector;

        bool operator==(const HistKey& o) const noexcept
        {
            return selector == o.selector
                && std::memcmp(code_hash, o.code_hash, 32) == 0;
        }
    };
    struct HistKeyHash
    {
        size_t operator()(const HistKey& k) const noexcept
        {
            // FNV-1a over (selector || code_hash). Cheap, deterministic, no deps.
            uint64_t h = 0xcbf29ce484222325ull;
            const auto mix = [&](uint8_t b)
            {
                h ^= b;
                h *= 0x100000001b3ull;
            };
            mix(uint8_t(k.selector >> 24));
            mix(uint8_t(k.selector >> 16));
            mix(uint8_t(k.selector >> 8));
            mix(uint8_t(k.selector));
            for (int i = 0; i < 32; ++i) mix(k.code_hash[i]);
            return static_cast<size_t>(h);
        }
    };

    struct HistEntry
    {
        std::vector<ConflictLane> reads;
        std::vector<ConflictLane> writes;
    };
}  // namespace

struct HistoricalProfile::Impl
{
    size_t capacity;
    std::list<std::pair<HistKey, HistEntry>> lru;
    std::unordered_map<HistKey, decltype(lru)::iterator, HistKeyHash> map;

    explicit Impl(size_t cap) : capacity{cap} {}
};

HistoricalProfile::HistoricalProfile(size_t capacity)
    : impl_{new Impl(capacity)}
{}

HistoricalProfile::~HistoricalProfile() { delete impl_; }

ConflictSpec HistoricalProfile::compose(
    uint32_t tx_id,
    const uint8_t code_hash[32],
    const uint8_t* calldata, size_t calldata_len,
    const uint8_t /*recipient*/[20],
    ConflictArena& arena)
{
    auto spec = make_declared_empty(tx_id);
    if (calldata_len < 4) { ++misses_; return spec; }

    HistKey key{};
    std::memcpy(key.code_hash, code_hash, 32);
    key.selector = be_u32(calldata);

    auto it = impl_->map.find(key);
    if (it == impl_->map.end()) { ++misses_; return spec; }

    // Touch LRU
    impl_->lru.splice(impl_->lru.begin(), impl_->lru, it->second);
    const auto& entry = it->second->second;

    auto [r_off, r_cnt] = arena.push_lanes(entry.reads.data(),
                                           static_cast<uint16_t>(entry.reads.size()));
    auto [w_off, w_cnt] = arena.push_lanes(entry.writes.data(),
                                           static_cast<uint16_t>(entry.writes.size()));
    spec.read_lane_offset = r_off;
    spec.read_lane_count = r_cnt;
    spec.write_lane_offset = w_off;
    spec.write_lane_count = w_cnt;
    spec.confidence = 150;
    spec.source = static_cast<uint8_t>(ConflictSource::Historical);
    ++hits_;
    return spec;
}

void HistoricalProfile::record(
    const uint8_t code_hash[32],
    const uint8_t* calldata, size_t calldata_len,
    const ConflictLane* read_lanes, uint16_t n_read,
    const ConflictLane* write_lanes, uint16_t n_write)
{
    if (calldata_len < 4) return;

    HistKey key{};
    std::memcpy(key.code_hash, code_hash, 32);
    key.selector = be_u32(calldata);

    auto it = impl_->map.find(key);
    if (it != impl_->map.end())
    {
        // Update entry & touch
        auto& entry = it->second->second;
        entry.reads.assign(read_lanes, read_lanes + n_read);
        entry.writes.assign(write_lanes, write_lanes + n_write);
        impl_->lru.splice(impl_->lru.begin(), impl_->lru, it->second);
        return;
    }

    // Insert new
    HistEntry entry;
    entry.reads.assign(read_lanes, read_lanes + n_read);
    entry.writes.assign(write_lanes, write_lanes + n_write);
    impl_->lru.emplace_front(key, std::move(entry));
    impl_->map[key] = impl_->lru.begin();

    if (impl_->lru.size() > impl_->capacity)
    {
        auto& last = impl_->lru.back();
        impl_->map.erase(last.first);
        impl_->lru.pop_back();
    }
}

// ===========================================================================
// Source 4 — Precompile id table
// ===========================================================================
ConflictSpec compose_from_precompile(
    uint32_t tx_id,
    const uint8_t recipient[20],
    ConflictArena& /*arena*/)
{
    auto spec = make_declared_empty(tx_id);
    if (recipient == nullptr) return spec;
    if (!is_precompile_address(recipient)) return spec;

    // Precompiles touch no storage. read_lane_count = write_lane_count = 0.
    spec.confidence = 220;
    spec.source = static_cast<uint8_t>(ConflictSource::Precompile);
    return spec;
}

// ===========================================================================
// Source 5 — Learned predictor (NOTIMPL stub)
// ===========================================================================
ConflictSpec LearnedPredictor::predict(
    uint32_t tx_id,
    const uint8_t /*code_hash*/[32],
    const uint8_t* /*calldata*/, size_t /*calldata_len*/,
    const uint8_t /*recipient*/[20],
    ConflictArena& /*arena*/)
{
    // Default impl: no prediction. Returns Declared/0-confidence so the
    // composer falls through to the next source (which is itself this
    // fallback, so the tx ends up using dynamic discovery — current
    // Block-STM behaviour preserved byte-equal).
    return make_declared_empty(tx_id);
}

// ===========================================================================
// Composer — fixed priority merge
// ===========================================================================
ConflictSpec compose(
    const ComposerInputs& in,
    HistoricalProfile* historical,
    LearnedPredictor* learned,
    ConflictArena& arena)
{
    // Try AccessList first.
    if (in.warm_addresses_len > 0 || in.warm_storage_keys_len > 0)
    {
        auto s = compose_from_access_list(
            in.tx_id,
            in.warm_addresses, in.warm_addresses_len,
            in.warm_storage_keys, in.warm_storage_keys_len,
            arena);
        if (s.confidence >= kSkipDiscoveryThreshold) return s;
    }

    // Try ABI selector.
    if (in.recipient != nullptr && in.sender != nullptr && in.calldata_len >= 4)
    {
        auto s = compose_from_abi(
            in.tx_id, in.recipient, in.calldata, in.calldata_len, in.sender, arena);
        if (s.confidence >= kSkipDiscoveryThreshold) return s;
    }

    // Try Historical.
    if (historical != nullptr && in.has_code_hash && in.calldata_len >= 4)
    {
        auto s = historical->compose(
            in.tx_id, in.code_hash, in.calldata, in.calldata_len,
            in.recipient ? in.recipient : (const uint8_t*)"\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
            arena);
        if (s.confidence >= kSkipDiscoveryThreshold) return s;
    }

    // Try Precompile.
    if (in.recipient != nullptr)
    {
        auto s = compose_from_precompile(in.tx_id, in.recipient, arena);
        if (s.confidence >= kSkipDiscoveryThreshold) return s;
    }

    // Try Learned (NOTIMPL stub returns 0-confidence Declared today).
    if (learned != nullptr)
    {
        auto s = learned->predict(
            in.tx_id, in.code_hash, in.calldata, in.calldata_len,
            in.recipient ? in.recipient : (const uint8_t*)"\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
            arena);
        if (s.confidence >= kSkipDiscoveryThreshold) return s;
    }

    // No source met threshold. Return Declared/0 — the validator will fall
    // back to the existing dynamic-discovery Block-STM path for this tx.
    return make_declared_empty(in.tx_id);
}

}  // namespace evm::stm
