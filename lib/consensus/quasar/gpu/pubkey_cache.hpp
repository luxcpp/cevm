// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file pubkey_cache.hpp
/// Decompressed BLS12-381 G1 public-key cache for the Quasar same-message
/// hot path.
///
/// Motivation
/// ----------
/// The same-message batched verifier (`verify_bls_same_message_batch`) at
/// n=1024 spends ~30 ms in `blst_p1_uncompress` (~30 us each * 1024) on
/// the 130 ms hot path. Validators sign the same `certificate_subject`
/// every round AND the validator set is mostly stable across rounds, so
/// caching the decompressed `blst_p1_affine` keyed by the 48-byte
/// compressed pubkey eliminates the bulk of that work on warm runs.
///
/// Design
/// ------
/// * Fixed-capacity 2-way set-associative table. Capacity defaults to
///   4096 entries (4x a 1024-validator set). Eviction is per-bucket
///   round-robin (one bit of clock) — simple, branch-free, no global LRU
///   list to lock.
/// * Bucket index is FNV1a-64 of the 48-byte compressed pubkey, modulo
///   `kBuckets`. Collision check uses the full 48 bytes.
/// * Cached affine has already passed `blst_p1_affine_in_g1`. We do NOT
///   re-validate on hit. Validation correctness is preserved by the
///   "lookup OR uncompress+validate+insert" hot-path contract.
/// * Single `std::mutex` around the table. Critical sections are short
///   (memcmp + memcpy of ~144 bytes). On the same-message hot path the
///   contention is negligible because batches run on a single thread per
///   verify call.
///
/// Byte-equality contract
/// ----------------------
/// On every hit the cached `blst_p1_affine` is, by construction, the
/// exact output of `blst_p1_uncompress` on the same compressed bytes. blst
/// is deterministic. A debug build assertion (#ifndef NDEBUG) re-runs
/// `blst_p1_uncompress` on hit and `memcmp`s the result; this guards
/// against any future cache corruption introducing a divergence.
///
/// Out of scope
/// ------------
/// * G2 signature cache: the signatures change every round (subject is
///   different) so caching them yields nothing. Only the pubkey is
///   stable across rounds.
/// * Cross-process / cross-shard sharing. The cache is per-process.

#pragma once

#include <blst.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <mutex>

namespace quasar::gpu {

/// Capacity (number of cache slots, including both ways). 4096 = ~4x a
/// typical 1024-validator set; enough for two consecutive validator sets
/// to coexist without thrashing during a rotation.
inline constexpr std::size_t kPubkeyCacheBuckets = 2048;
inline constexpr std::size_t kPubkeyCacheWays    = 2;

class PubkeyAffineCache {
public:
    /// Try to fetch the decompressed affine for `pk_compressed` (48 bytes).
    /// Returns true on hit; `out` is filled with the cached affine. Hit
    /// implies the cached value previously passed `blst_p1_affine_in_g1`.
    bool lookup(const uint8_t pk_compressed[48], blst_p1_affine& out) noexcept
    {
        const uint64_t h = fnv1a64(pk_compressed, 48);
        const std::size_t b = static_cast<std::size_t>(h) % kPubkeyCacheBuckets;
        std::lock_guard<std::mutex> g(mu_);
        for (std::size_t w = 0; w < kPubkeyCacheWays; ++w) {
            Slot& s = table_[b][w];
            if (s.occupied && std::memcmp(s.key.data(), pk_compressed, 48) == 0) {
                out = s.value;
                return true;
            }
        }
        return false;
    }

    /// Store an already-decompressed-and-subgroup-checked affine for
    /// `pk_compressed`. Caller MUST have already called
    /// `blst_p1_affine_in_g1` on `value` and confirmed it returned true.
    void insert(const uint8_t pk_compressed[48], const blst_p1_affine& value) noexcept
    {
        const uint64_t h = fnv1a64(pk_compressed, 48);
        const std::size_t b = static_cast<std::size_t>(h) % kPubkeyCacheBuckets;
        std::lock_guard<std::mutex> g(mu_);
        // First pass: refresh-in-place if the same key is already there
        // (race: two threads decompress and both insert).
        for (std::size_t w = 0; w < kPubkeyCacheWays; ++w) {
            Slot& s = table_[b][w];
            if (s.occupied && std::memcmp(s.key.data(), pk_compressed, 48) == 0)
                return;  // already cached; no-op.
        }
        // Second pass: prefer an empty way; else evict the way pointed at
        // by the per-bucket clock bit and advance it.
        std::size_t target = clock_[b];
        for (std::size_t w = 0; w < kPubkeyCacheWays; ++w) {
            if (!table_[b][w].occupied) { target = w; break; }
        }
        Slot& s = table_[b][target];
        std::memcpy(s.key.data(), pk_compressed, 48);
        s.value    = value;
        s.occupied = true;
        clock_[b] = static_cast<uint8_t>((target + 1) % kPubkeyCacheWays);
    }

    /// Drop every entry. Used by tests to force cold runs.
    void clear() noexcept
    {
        std::lock_guard<std::mutex> g(mu_);
        for (auto& bucket : table_)
            for (auto& s : bucket) s.occupied = false;
        for (auto& c : clock_) c = 0;
    }

    /// Process-wide singleton. Lives for the program's lifetime.
    static PubkeyAffineCache& instance() noexcept
    {
        static PubkeyAffineCache c;
        return c;
    }

private:
    struct Slot {
        std::array<uint8_t, 48> key{};
        blst_p1_affine          value{};
        bool                    occupied = false;
    };

    static uint64_t fnv1a64(const uint8_t* p, std::size_t n) noexcept
    {
        // FNV1a 64-bit. Adequate for cache bucket selection — collisions
        // are resolved by the full 48-byte memcmp on lookup.
        uint64_t h = 0xcbf29ce484222325ULL;
        for (std::size_t i = 0; i < n; ++i) {
            h ^= p[i];
            h *= 0x100000001b3ULL;
        }
        return h;
    }

    std::mutex                                            mu_;
    std::array<std::array<Slot, kPubkeyCacheWays>,
               kPubkeyCacheBuckets>                       table_{};
    std::array<uint8_t, kPubkeyCacheBuckets>              clock_{};
};

}  // namespace quasar::gpu
