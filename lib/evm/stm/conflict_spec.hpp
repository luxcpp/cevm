// Copyright (C) 2026, The cevm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file conflict_spec.hpp
/// LP-090 v0.50: ConflictSpec ABI — STM conflict declaration scaffold.
///
/// Block-STM today discovers all read/write sets dynamically by re-executing
/// aborted txs. Per the v0.50 executive review ("do not make STM discover
/// all conflicts dynamically"), we instead *declare* conflicts up front from
/// five sources, in fixed priority order:
///
///   AccessList (EIP-2930) > ABI selector > Historical profile >
///       Precompile id > Learned predictor > Declared (fallback)
///
/// The composed ConflictSpec is consumed by:
///   1. Prism refraction (pre-placement): fan tx into shards keyed on
///      write-lane offsets so contended writes land on the same shard,
///      avoiding hot-key cross-shard repairs.
///   2. QuasarSTM validation: when source confidence ≥ threshold, validation
///      may *skip* dynamic re-execution and trust the declared write set.
///      Below threshold, it falls back to the existing optimistic re-exec
///      path (today's Block-STM behaviour, byte-equal preserved).
///
/// Lanes are referenced by (offset, count) into a *block-scoped* lane arena
/// owned by the scheduler; ConflictSpec is therefore a 32-byte POD that can
/// be sorted, hashed, and copied freely without owning memory. This keeps
/// the ABI byte-stable across CPU and GPU backends.
///
/// AFT 2025 reference: declared conflicts deliver 1.75x over sequential and
/// 1.33x over PEVM on the standard contended block mix.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace evm::stm
{

/// Source of a conflict declaration. Lower numeric value = higher priority
/// when multiple sources fire on the same tx; ties broken by larger lane
/// count (broader covers narrower).
enum class ConflictSource : uint8_t
{
    AccessList = 0,  ///< EIP-2930 access_list parsed from tx envelope
    ABI        = 1,  ///< Known function selector → static read/write set
    Historical = 2,  ///< (code_hash, selector) seen in past blocks
    Precompile = 3,  ///< Known precompile address → static read/write set
    Learned    = 4,  ///< Online predictor (NOTIMPL stub today)
    Declared   = 5,  ///< Fallback / explicit caller declaration
};

/// A single tx conflict declaration. POD, copyable, hashable.
///
/// Layout matches the v0.50 executive ABI spec exactly: 28 bytes natural,
/// with the compiler inserting 2-byte alignment padding after each
/// (u32 + u16) lane triple so the next u32 starts 4-aligned. The static
/// asserts below pin both the total size and every field offset so the
/// wire encoding cannot drift across compilers or language bindings.
///
/// Lane fields reference offsets into the per-block ConflictArena.lanes
/// vector. (offset=UINT32_MAX, count=0) means "no lanes of this kind".
/// Reducer lanes are slots that accept commutative aggregation across
/// multiple writes (BalanceDelta, FeeAccumulate); they do NOT trigger
/// repairs even when multiple txs target the same slot.
struct ConflictSpec
{
    uint32_t tx_id;               ///< Index of the tx this spec describes
    uint32_t read_lane_offset;    ///< Index into ConflictArena.lanes
    uint16_t read_lane_count;     ///< Number of read lanes
    // 2 bytes natural alignment padding here
    uint32_t write_lane_offset;
    uint16_t write_lane_count;
    // 2 bytes natural alignment padding here
    uint32_t reducer_lane_offset; ///< Commutative-write lanes (no repair on conflict)
    uint16_t reducer_lane_count;
    uint8_t  confidence;          ///< 0..255; >=threshold => skip dynamic discovery
    uint8_t  source;              ///< ConflictSource cast to uint8_t

    /// Strongly-typed accessor. The wire field stays a uint8_t so the
    /// struct layout is ABI-fixed across compilers / language bindings.
    ConflictSource source_kind() const noexcept
    {
        return static_cast<ConflictSource>(source);
    }
};
static_assert(sizeof(ConflictSpec) == 28, "ConflictSpec must be 28 bytes (v0.50 ABI)");
static_assert(offsetof(ConflictSpec, tx_id) == 0,                "tx_id @ 0");
static_assert(offsetof(ConflictSpec, read_lane_offset) == 4,     "read_lane_offset @ 4");
static_assert(offsetof(ConflictSpec, read_lane_count) == 8,      "read_lane_count @ 8");
static_assert(offsetof(ConflictSpec, write_lane_offset) == 12,   "write_lane_offset @ 12");
static_assert(offsetof(ConflictSpec, write_lane_count) == 16,    "write_lane_count @ 16");
static_assert(offsetof(ConflictSpec, reducer_lane_offset) == 20, "reducer_lane_offset @ 20");
static_assert(offsetof(ConflictSpec, reducer_lane_count) == 24,  "reducer_lane_count @ 24");
static_assert(offsetof(ConflictSpec, confidence) == 26,          "confidence @ 26");
static_assert(offsetof(ConflictSpec, source) == 27,              "source @ 27");

/// A lane is a (20-byte address, 32-byte slot) tuple. Encoded as 52 bytes;
/// the arena packs them into a contiguous vector indexed by ConflictSpec
/// offsets. Address-only lanes (BALANCE, EXTCODESIZE) use slot=zero.
struct ConflictLane
{
    uint8_t address[20];
    uint8_t slot[32];

    bool operator==(const ConflictLane& o) const noexcept
    {
        return std::memcmp(address, o.address, 20) == 0
            && std::memcmp(slot, o.slot, 32) == 0;
    }
};
static_assert(sizeof(ConflictLane) == 52, "ConflictLane must be 52 bytes");

/// Per-block storage for ConflictSpec lane data. Specs are referenced by
/// index, so the arena owns the lane vector and survives spec sorting.
struct ConflictArena
{
    std::vector<ConflictLane> lanes;
    std::vector<ConflictSpec> specs;

    void clear() noexcept
    {
        lanes.clear();
        specs.clear();
    }

    /// Append a slice of lanes; returns (offset, count) for a ConflictSpec.
    std::pair<uint32_t, uint16_t> push_lanes(const ConflictLane* p, uint16_t n)
    {
        const auto off = static_cast<uint32_t>(lanes.size());
        lanes.insert(lanes.end(), p, p + n);
        return {off, n};
    }

    /// Empty lane sentinel: (UINT32_MAX, 0).
    static std::pair<uint32_t, uint16_t> empty_lanes() noexcept
    {
        return {UINT32_MAX, 0};
    }
};

// ---------------------------------------------------------------------------
// Source 1 — EIP-2930 access list parser.
//
// Input: flat-encoded access list per `Config::warm_addresses` /
// `Config::warm_storage_keys` (see lib/evm/gpu/gpu_dispatch.hpp). Output: a
// ConflictSpec with read_lanes covering every (addr, slot) entry plus
// addr-only entries with slot=zero. Confidence = 200 (very high — caller
// declared via tx envelope). Writes are NOT inferred from the access list:
// EIP-2930 declares accessed slots, not written slots. write_lane_count = 0
// from this source; ABI/Historical sources may add writes.
// ---------------------------------------------------------------------------
ConflictSpec compose_from_access_list(
    uint32_t tx_id,
    const uint8_t* warm_addresses,
    size_t warm_addresses_len,
    const uint8_t* warm_storage_keys,
    size_t warm_storage_keys_len,
    ConflictArena& arena);

// ---------------------------------------------------------------------------
// Source 2 — ABI selector table.
//
// Maps known 4-byte selectors to a static read/write set keyed off the
// recipient address. Covers the high-volume selectors that dominate
// real-world block conflict density:
//
//   ERC-20  transfer(address,uint256)            0xa9059cbb
//   ERC-20  transferFrom(address,address,uint256) 0x23b872dd
//   ERC-20  approve(address,uint256)             0x095ea7b3
//   Uniswap V2 swap(uint256,uint256,address,bytes) 0x022c0d9f
//   Uniswap V3 exactInputSingle(...)             0x414bf389
//   ERC-721 transferFrom(address,address,uint256) 0x23b872dd (collides; resolved by code_hash class)
//   ERC-721 safeTransferFrom(...)                0x42842e6e
//
// Confidence = 180 (high — selectors are stable, but storage layout
// assumed standard ERC-20/ERC-721 mapping slots).
// ---------------------------------------------------------------------------
ConflictSpec compose_from_abi(
    uint32_t tx_id,
    const uint8_t recipient[20],
    const uint8_t* calldata,
    size_t calldata_len,
    const uint8_t sender[20],
    ConflictArena& arena);

// ---------------------------------------------------------------------------
// Source 3 — Historical profile (LRU).
//
// Cache keyed on (code_hash, selector) → past observed (read_lanes,
// write_lanes). Hit rate is the dominant lever for the 1.33x AFT speedup;
// production target ≥0.85 hit rate on the EIP-1559 mainnet block mix.
// Confidence = 150 (medium — based on past observation, may not match
// today's calldata exactly).
// ---------------------------------------------------------------------------
class HistoricalProfile
{
public:
    explicit HistoricalProfile(size_t capacity = 4096);
    ~HistoricalProfile();

    /// Look up and compose. Returns a ConflictSpec with source=Historical
    /// on hit; on miss returns spec with source=Declared and confidence=0.
    ConflictSpec compose(
        uint32_t tx_id,
        const uint8_t code_hash[32],
        const uint8_t* calldata,
        size_t calldata_len,
        const uint8_t recipient[20],
        ConflictArena& arena);

    /// Update the profile after a successful execution.
    void record(
        const uint8_t code_hash[32],
        const uint8_t* calldata,
        size_t calldata_len,
        const ConflictLane* read_lanes, uint16_t n_read,
        const ConflictLane* write_lanes, uint16_t n_write);

    /// Stats.
    size_t hits() const noexcept { return hits_; }
    size_t misses() const noexcept { return misses_; }

private:
    struct Impl;
    Impl* impl_;
    size_t hits_ = 0;
    size_t misses_ = 0;
};

// ---------------------------------------------------------------------------
// Source 4 — Precompile id table.
//
// Mainnet precompile addresses 0x01..0x11 read no storage and write no
// storage (ecrecover, sha256, ripemd160, identity, modexp, bn254 add/mul/
// pairing, blake2f, point evaluation, secp256r1, ...). Confidence = 220
// (highest — the precompile set is consensus-fixed). Result spec has
// read_lane_count = 0, write_lane_count = 0.
// ---------------------------------------------------------------------------
ConflictSpec compose_from_precompile(
    uint32_t tx_id,
    const uint8_t recipient[20],
    ConflictArena& arena);

// ---------------------------------------------------------------------------
// Source 5 — Learned predictor (NOTIMPL stub).
//
// Online conflict predictor. Today: returns a Declared/0-confidence spec
// (i.e. fall-through to dynamic discovery). Retire plan: the
// LearnedPredictor interface stays stable; the impl swaps to a small MLP
// when offline training infrastructure (lux/zen-train) ships and produces
// a deterministic int8-quantised network we can run in the host without
// touching the GPU pipeline.
//
// NOT shipped as code today — only the interface and a NOTIMPL impl that
// composes a zero-confidence Declared spec. Doing the predictor in this
// scaffold would violate "no fake, no TODO" — the impl must wait until we
// have offline training data + a quantised network to inline.
// ---------------------------------------------------------------------------
class LearnedPredictor
{
public:
    virtual ~LearnedPredictor() = default;

    /// Default impl returns a Declared/0-confidence spec (no-op).
    virtual ConflictSpec predict(
        uint32_t tx_id,
        const uint8_t code_hash[32],
        const uint8_t* calldata,
        size_t calldata_len,
        const uint8_t recipient[20],
        ConflictArena& arena);
};

// ---------------------------------------------------------------------------
// Composer — fixed priority merge.
//
// Try each source in priority order. The FIRST source whose confidence
// meets the per-source minimum wins; lanes from lower-priority sources
// are discarded (they would only widen, never narrow, the spec, and
// widening costs throughput). The returned spec's source field reflects
// which source won.
//
// If no source meets its minimum, returns a Declared spec with
// confidence=0, write_lane_count=0, read_lane_count=0 — the QuasarSTM
// validator MUST then fall back to dynamic discovery for that tx.
// ---------------------------------------------------------------------------
struct ComposerInputs
{
    uint32_t tx_id;

    // EIP-2930 access list (already flat-encoded, see Config above)
    const uint8_t* warm_addresses = nullptr;
    size_t warm_addresses_len = 0;
    const uint8_t* warm_storage_keys = nullptr;
    size_t warm_storage_keys_len = 0;

    // Tx body
    const uint8_t* recipient = nullptr;     ///< 20 bytes (nullptr ⇒ contract create)
    const uint8_t* sender = nullptr;        ///< 20 bytes
    const uint8_t* calldata = nullptr;
    size_t calldata_len = 0;

    // Optional code hash (for Historical lookup); zeroed ⇒ skip Historical
    uint8_t code_hash[32] = {};
    bool has_code_hash = false;
};

/// Confidence threshold above which validation may skip dynamic discovery.
/// Picked so AccessList (200), ABI (180), Historical (150), Precompile (220)
/// all clear it; Declared (0) and zero-conf Learned never do.
inline constexpr uint8_t kSkipDiscoveryThreshold = 100;

/// Compose a ConflictSpec by trying each source in priority order.
ConflictSpec compose(
    const ComposerInputs& in,
    HistoricalProfile* historical,         // optional — nullptr skips this source
    LearnedPredictor* learned,             // optional — nullptr skips this source
    ConflictArena& arena);

}  // namespace evm::stm
