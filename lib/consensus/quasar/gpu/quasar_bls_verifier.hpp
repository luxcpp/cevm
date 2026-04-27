// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file quasar_bls_verifier.hpp
/// Host BLS12-381 aggregate verifier — Quasar 4.0 activation 2026-02-14.
///
/// v0.41 wires real BLS into the GPU substrate via a host shim:
///
///   GPU vote ring (host pushes)
///         │
///         ▼  run_epoch:
///   ┌───────────────────────────────────────────┐
///   │ host pump: for each unverified vote slot, │
///   │   call verify_bls_aggregate(pk, sig, msg) │
///   │   write vote_verified[slot] = 0/1         │
///   └───────────────────────────────────────────┘
///         │
///         ▼
///   wave-tick kernel reads vote_verified[] in drain_vote
///   and aggregates stake only for slots marked 1.
///
/// One round-trip per epoch on the host (~1-2 ms for 100 votes; blst
/// pairing dominates). v0.45 will move pairing to GPU for ≥10x speedup.
///
/// Wire format (matches the Ethereum / Eth2 "min-pubkey" scheme):
///   pk_aggregate  : 48 bytes — compressed BLS12-381 G1 (the public key)
///   signature     : 96 bytes — compressed BLS12-381 G2 (the signature)
///   subject       : 32 bytes — message digest (the consensus subject)
///
/// Domain separation tag: kQuasarBLSDST below. Sign side must hash-to-G2
/// with the SAME tag.
///
/// ML-DSA + Ringtail lanes: not yet wired; verify_lane returns false for
/// those sig_kinds. v0.42 wires ML-DSA, v0.43 wires Ringtail.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace quasar::gpu {

/// BLS12-381 domain separation tag (DST) for Quasar consensus subjects.
/// Must stay byte-identical between sign_subject() and verify_bls_aggregate().
inline constexpr std::array<uint8_t, 43> kQuasarBLSDST = {
    'B','L','S','_','S','I','G','_','B','L','S','1','2','3','8','1',
    'G','2','_','X','M','D',':','S','H','A','-','2','5','6','_','S',
    'S','W','U','_','R','O','_','Q','U','S','R'
};

/// Aggregate-verify a BLS signature against a consensus subject.
///
/// @param subject       32-byte consensus subject (the message hashed to G2).
/// @param signature     96-byte compressed G2 signature (or aggregate sig).
/// @param pk_aggregate  48-byte compressed G1 public key (or aggregate pk).
/// @return              true iff the pairing check holds:
///                      e(pk, H(subject)) == e(G1, sig)  (in min-pk form)
///
/// Returns false on any decoding / curve / subgroup failure — never throws.
[[nodiscard]] bool verify_bls_aggregate(
    const uint8_t subject[32],
    const uint8_t signature[96],
    const uint8_t pk_aggregate[48]) noexcept;

/// Per-lane verifier dispatcher. sig_kind is one of QuasarSigKind:
///   0 = BLS12-381  → real verifier
///   1 = Ringtail   → returns false (v0.43)
///   2 = ML-DSA-65  → returns false (v0.42)
///
/// Mainnet-safe: unknown / unwired lanes never accept. Quorum stake never
/// aggregates from a fake signature.
[[nodiscard]] bool verify_lane(
    uint32_t sig_kind,
    const uint8_t subject[32],
    const uint8_t signature[96],
    const uint8_t pk_aggregate[48]) noexcept;

// =============================================================================
// Test-only sign helper (also used by validator clients in dev). Wraps the
// blst min-pk sign-in-G2 path with the same DST as verify_bls_aggregate.
// =============================================================================

/// 32-byte BLS scalar secret key (big-endian).
struct BLSSecretKey { std::array<uint8_t, 32> bytes{}; };

/// 48-byte compressed G1 public key.
struct BLSPublicKey { std::array<uint8_t, 48> bytes{}; };

/// Generate a BLS keypair from 32 bytes of input keying material (IKM).
/// IKM_len must be at least 32. For tests, pass any deterministic seed.
[[nodiscard]] bool keygen_bls(
    const uint8_t* ikm, std::size_t ikm_len,
    BLSSecretKey& sk_out,
    BLSPublicKey& pk_out) noexcept;

/// Sign a 32-byte subject with the given secret key. Returns 96-byte
/// compressed G2 signature in `sig_out`. Same DST as verify_bls_aggregate.
[[nodiscard]] bool sign_subject(
    const BLSSecretKey& sk,
    const uint8_t subject[32],
    uint8_t sig_out[96]) noexcept;

// =============================================================================
// v0.45 — batched aggregate verify.
//
// `verify_bls_aggregate_batch` accumulates N (pk, sig, subject) tuples into a
// single blst pairing context and runs ONE Miller-loop accumulation followed
// by ONE final exponentiation. For N=1024 this is ~10-50× faster than calling
// verify_bls_aggregate N times because the dominating final_exp (~5 ms on
// host) runs once instead of N times.
//
// Behaviour: returns true iff EVERY input pair verifies. On any decode /
// subgroup / pairing failure the function returns false. Mainnet-safe: a
// single bad sig denies the whole batch (no per-element verdict leak).
//
// All 32-byte subjects are independent — i.e. signers may have signed
// different consensus subjects, the function batches across them.
// =============================================================================
[[nodiscard]] bool verify_bls_aggregate_batch(
    const uint8_t* const subjects[],   ///< n × pointer to 32-byte subject
    const uint8_t* const signatures[], ///< n × pointer to 96-byte compressed G2 sig
    const uint8_t* const pks[],        ///< n × pointer to 48-byte compressed G1 pk
    std::size_t n) noexcept;

/// Same-message BLS aggregate verify. All N signers signed the SAME subject.
/// Aggregates pks on G1 and sigs on G2 then runs ONE pairing equation.
///
/// Cost: O(N) decompress + O(N) point add + 2 Miller loops + 1 final_exp.
/// For N=1024 vs N×verify_bls_aggregate this is ≥50× faster on host blst.
///
/// This is the BLS lane's hot path in Quasar consensus: every validator
/// signs the same `subject` (the canonical certificate subject in
/// `compute_certificate_subject`). The general-message batch above
/// remains the fallback for cross-subject batches (e.g. cross-round
/// aggregation).
[[nodiscard]] bool verify_bls_same_message_batch(
    const uint8_t subject[32],
    const uint8_t* const signatures[], ///< n × 96-byte compressed G2
    const uint8_t* const pks[],        ///< n × 48-byte compressed G1
    std::size_t n) noexcept;

}  // namespace quasar::gpu
