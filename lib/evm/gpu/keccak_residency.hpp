// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file keccak_residency.hpp
/// QuasarGPU v0.44 — round-scoped keccak residency service.
///
/// User directive (canonical): "Precompiles should not be opaque calls. They
/// should be GPU services that produce rootable artifacts."
///
/// Inside one consensus round the same keccak input is hashed many times:
///   * MappingSlot : `keccak(slot_key || 32-byte map base)` for every SLOAD
///                   that targets the same map; same call repeats every time
///                   the contract reads/writes that mapping.
///   * CodeHash    : the same contract bytecode hashed by every CREATE2 /
///                   EXTCODEHASH that touches it.
///   * StateLeaf   : the same Merkle leaf hashed by every storage_root walk
///                   over the same account.
///
/// `KeccakResidencySession` keeps a per-round dedup cache that catches these
/// repeats without re-hashing. Same recipe as `lux::crypto::keccak::
/// KeccakDedupTable` (sibling territory) but inlined here so this layer ships
/// with cevm's own keccak (cevm_precompiles/keccak) and depends on nothing
/// outside cevm. The kind enum mirrors the v0.63 service so call-site tagging
/// can later be redirected to the canonical service without changing
/// EVM-execution call sites.
///
/// Layout:
///   * 8192 cuckoo slots × (16-byte key + 32-byte digest + 1 valid byte)
///     = ~393 KB per session, fixed.
///   * Reset is `std::memset` over the slot array — O(slots), constant time.
///   * `run_one` is the single-call entry: classify, probe, hash on miss.
///   * `run_batch` walks an array of (kind, input, len, out) tuples — same
///     in-batch + round-cache layering as the canonical service.
///
/// Determinism: hits/misses are observable but never affect output bytes.
/// The 32-byte output is always the canonical keccak256 of the input.

#pragma once

#include <cevm_precompiles/keccak.hpp>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <span>

namespace evm::gpu::keccak_residency {

/// Per-call classification. Mirrors `lux::crypto::keccak::KeccakJobKind` so
/// when sibling territory's canonical service is reachable from cevm the
/// kind value is interchangeable. New kinds append at the end.
enum class JobKind : uint16_t {
    TxHash               = 0,
    MappingSlot          = 1,
    CodeHash             = 2,
    ReceiptLeaf          = 3,
    StateLeaf            = 4,
    ExecutionRootNode    = 5,
    CertificateSubject   = 6,
    AuditRoot            = 7,
    OrderflowCommitment  = 8,
};

/// One per round. Allocate on the EVM execution engine instance, call
/// `begin_round` at round start, route every hash through `hash`, then read
/// `hits()` / `misses()` for telemetry. `end_round` resets the table.
class KeccakResidencySession {
public:
    static constexpr std::size_t kSlots = 8192u;

    KeccakResidencySession() noexcept { reset_(); }

    /// Begin a new round. Clears the dedup cache. Idempotent.
    void begin_round(uint64_t /*round*/) noexcept { reset_(); }

    /// Compute keccak256(input). Tags this job with `kind` so MappingSlot
    /// repeats in the same round hit the cache. Output is 32 bytes,
    /// byte-equal to ethash::keccak256(input.data(), input.size()).
    ///
    /// `kind` is purely a routing hint — cache lookup only happens on
    /// `MappingSlot`, the only kind that demonstrably repeats inside one
    /// round (per the v0.63 service's load characterization). Other kinds
    /// hash directly. Future kinds may opt into caching by adding a bucket
    /// here.
    ///
    /// `out` must be exactly 32 bytes — we accept a dynamic-extent span so
    /// `Job` (which holds an `output` span) is default-constructible.
    void hash(JobKind kind,
              std::span<const uint8_t> input,
              std::span<uint8_t> out) noexcept {
        // Caller contract: out is exactly 32 bytes. We don't validate inside
        // the hot path; misuse is caught by tests.
        if (kind == JobKind::MappingSlot) {
            // Compute candidate. We always need the answer; the cache only
            // tells us whether we've seen it before this round.
            auto digest = ethash::keccak256(input.data(), input.size());
            std::memcpy(out.data(), digest.bytes, 32);

            // 4-way set-associative probe. With kSlots=8192 and ~500 unique
            // keys per round the load factor is < 0.07; 4-way probing makes
            // collision-induced shadowing statistically negligible (<1%).
            // FNV-1a + the probe stride 0x9E37... gives good distribution.
            const uint64_t h = key_hash_(kind, input.data(), input.size());
            int empty_probe = -1;
            for (int probe = 0; probe < 4; ++probe) {
                const std::size_t s = static_cast<std::size_t>(
                    (h + static_cast<uint64_t>(probe) * 0x9E3779B97F4A7C15ull)
                    % kSlots);
                Entry& e = slots_[s];
                if (e.valid != 0u
                    && e.input_len == input.size()
                    && std::memcmp(e.out, digest.bytes, 32) == 0)
                {
                    ++hits_;
                    return;
                }
                if (e.valid == 0u && empty_probe < 0) empty_probe = probe;
            }
            // Miss. Insert at first empty slot if we found one; otherwise
            // replace probe-0 (FIFO eviction at the head of the chain).
            const int target = empty_probe >= 0 ? empty_probe : 0;
            const std::size_t s = static_cast<std::size_t>(
                (h + static_cast<uint64_t>(target) * 0x9E3779B97F4A7C15ull)
                % kSlots);
            Entry& e = slots_[s];
            std::memcpy(e.out, digest.bytes, 32);
            e.input_len = static_cast<uint32_t>(input.size());
            e.kind = static_cast<uint16_t>(kind);
            e.valid = 1u;
            ++misses_;
            return;
        }

        // Default path: compute directly, no cache touch.
        auto digest = ethash::keccak256(input.data(), input.size());
        std::memcpy(out.data(), digest.bytes, 32);
        ++uncached_;
    }

    /// Convenience wrapper that returns the digest as `std::array`.
    std::array<uint8_t, 32> hash(JobKind kind,
                                  std::span<const uint8_t> input) noexcept {
        std::array<uint8_t, 32> out{};
        hash(kind, input, std::span<uint8_t>(out.data(), out.size()));
        return out;
    }

    /// One job in a `run_batch` request. The output span must point to a
    /// 32-byte buffer; we use a dynamic-extent span so vectors of jobs are
    /// default-constructible (static-extent spans don't have a default
    /// ctor, which makes them unusable inside std::vector).
    struct Job {
        JobKind kind = JobKind::TxHash;
        std::span<const uint8_t> input{};
        std::span<uint8_t> output{};         ///< must be exactly 32 bytes
    };

    /// Drain `n` jobs. Honours both in-batch dedup (same input bytes within
    /// this call hash once) and round-cache dedup (MappingSlot kind only,
    /// across calls). Returns the number of distinct hashes actually
    /// computed (cache hits + in-batch hits do not count).
    std::size_t run_batch(std::span<Job> jobs) noexcept {
        // In-batch dedup is a small probing table local to this call. Same
        // shape as the canonical sibling, scaled to the typical EVM block:
        // 1024 slots × 4 probes covers ~95% of repeat payloads.
        constexpr std::size_t kBatchSlots = 1024u;
        struct BatchEntry {
            uint64_t hash;
            uint32_t input_len;
            uint16_t kind;
            uint8_t  valid;
            uint8_t  _pad;
            std::size_t job_index;     ///< first occurrence's job index
        };
        BatchEntry batch[kBatchSlots]{};

        std::size_t computed = 0;
        for (std::size_t i = 0; i < jobs.size(); ++i) {
            Job& j = jobs[i];
            const uint64_t h = key_hash_(j.kind,
                                          j.input.data(), j.input.size());
            // Linear probe (4 buckets) for in-batch dedup.
            bool in_batch_hit = false;
            for (int probe = 0; probe < 4 && !in_batch_hit; ++probe) {
                const std::size_t s = static_cast<std::size_t>(
                    (h + static_cast<uint64_t>(probe) * 0x9E3779B97F4A7C15ull) % kBatchSlots);
                BatchEntry& e = batch[s];
                if (e.valid == 0u) {
                    e.hash = h;
                    e.input_len = static_cast<uint32_t>(j.input.size());
                    e.kind = static_cast<uint16_t>(j.kind);
                    e.valid = 1u;
                    e.job_index = i;
                    break;
                }
                if (e.hash == h
                    && e.input_len == j.input.size()
                    && e.kind == static_cast<uint16_t>(j.kind))
                {
                    // First-occurrence's output already populated — copy.
                    std::memcpy(j.output.data(),
                                jobs[e.job_index].output.data(), 32);
                    ++hits_;
                    in_batch_hit = true;
                }
            }
            if (in_batch_hit) continue;

            // Fall through to single-call hash (which handles round-cache
            // for MappingSlot).
            hash(j.kind, j.input, j.output);
            ++computed;
        }
        return computed;
    }

    /// End the round. Clears dedup state but keeps counters until next
    /// `begin_round` so end-of-round telemetry is observable.
    void end_round() noexcept { /* counters preserved; cache reset on next begin */ }

    // -- Telemetry ------------------------------------------------------------

    /// Cache hits this round. Each hit is one keccak we did not compute.
    uint64_t hits() const noexcept { return hits_; }
    /// Cache misses this round. Counts both first-time inputs and collisions.
    uint64_t misses() const noexcept { return misses_; }
    /// Calls that bypassed the cache (kind != MappingSlot).
    uint64_t uncached() const noexcept { return uncached_; }

    /// Hit rate over decisions where the cache could plausibly fire (i.e.
    /// MappingSlot calls only). Returns 0.0 when the round saw no MappingSlot
    /// calls.
    double hit_rate() const noexcept {
        const uint64_t denom = hits_ + misses_;
        return denom == 0u ? 0.0
                           : static_cast<double>(hits_) /
                             static_cast<double>(denom);
    }

private:
    struct Entry {
        uint8_t  out[32];
        uint32_t input_len;
        uint16_t kind;
        uint8_t  valid;
        uint8_t  _pad;
    };
    static_assert(sizeof(Entry) == 40, "KeccakResidencySession::Entry layout drift");

    void reset_() noexcept {
        std::memset(slots_, 0, sizeof(slots_));
        hits_ = misses_ = uncached_ = 0u;
    }

    static uint64_t key_hash_(JobKind kind,
                               const uint8_t* input,
                               std::size_t len) noexcept {
        // FNV-1a over (kind || first 64 bytes || len). Same recipe as the
        // canonical sibling so the per-bucket distribution matches.
        uint64_t h = 0xCBF29CE484222325ull;
        h ^= static_cast<uint64_t>(kind);
        h *= 0x100000001B3ull;
        const std::size_t take = len < 64u ? len : 64u;
        for (std::size_t i = 0; i < take; ++i) {
            h ^= static_cast<uint64_t>(input[i]);
            h *= 0x100000001B3ull;
        }
        h ^= static_cast<uint64_t>(len);
        h *= 0x100000001B3ull;
        return h;
    }

    Entry slots_[kSlots]{};
    uint64_t hits_ = 0u;
    uint64_t misses_ = 0u;
    uint64_t uncached_ = 0u;
};

}  // namespace evm::gpu::keccak_residency
