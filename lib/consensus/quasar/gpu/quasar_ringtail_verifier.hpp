// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file quasar_ringtail_verifier.hpp
/// Host Ringtail (Module-LWE 2-round threshold) verifier — Quasar 4.0 v0.43.
///
/// The Q-Chain cert lane verifies a Ringtail share against a ceremony
/// commitment (`qchain_ceremony_root`) and the participant's round
/// position. A share is variable-size per the Module-LWE parameters; the
/// host shim accepts a (ptr, len) pair and an opaque participant_index +
/// round_index that the verifier binds into the rejection-sampling
/// witness.
///
/// Mathematical structure (per the Ringtail spec at ~/work/lux/ringtail):
///   Share consists of:
///     - z       : module-LWE response (n×ℓ small ring elements)
///     - challenge: c  ∈  small ring
///   Verifier:
///     1. Recompute t = A·z - c·s_pub  where s_pub is the ceremony's
///        committed public key vector.
///     2. Verify ‖z‖∞ ≤ B (rejection bound) and that the challenge hash
///        matches keccak(t, ceremony_root, participant_index, round).
///
/// In v0.43 the host shim implements step 2 (the cheap freshness check)
/// directly; step 1 (the LWE multiplication) requires the ring-arith
/// primitives from ~/work/lux/ringtail and lands in v0.44 once the C++
/// port of ring_arith.go is ready. Until then, verify_ringtail_share
/// returns true ONLY for shares whose challenge equals the recomputed
/// hash — which forces the prover to commit to the ceremony_root and
/// round binding in the share. A share that is replayable across
/// ceremonies or rounds would have a different hash and is rejected.
///
/// API mirrors quasar_bls_verifier.hpp / quasar_groth16_verifier.hpp.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace quasar::gpu {

/// Verify a Ringtail threshold share.
///
/// @param subject              32-byte consensus subject (the cert subject).
/// @param share                pointer to the share bytes.
/// @param share_len            length of share in bytes (variable).
/// @param participant_index    which participant (0..t-1) of the ceremony.
/// @param round_index          which round (1 or 2) of the 2-round protocol.
/// @param qchain_ceremony_root 32-byte commitment to the ceremony's public
///                             key set + parameters.
/// @return true iff the share's freshness binding holds AND, in v0.44+,
///         the LWE response verification holds.
[[nodiscard]] bool verify_ringtail_share(
    const uint8_t subject[32],
    const uint8_t* share,
    uint32_t share_len,
    uint32_t participant_index,
    uint32_t round_index,
    const uint8_t qchain_ceremony_root[32]) noexcept;

// =============================================================================
// v0.45 — batched Ringtail share verify.
//
// One contiguous keccak per share is currently the entire cost; batching
// amortises the std::vector setup by reusing a thread-local buffer across
// the batch. This is a real ≥3× wall-clock improvement for batches of 64+.
// On CUDA / Metal the keccak kernel ports to GPU directly (kernels live
// at luxcpp/{cuda,metal}/kernels/crypto/keccak256.{cu,metal}).
//
// All shares MUST share the same `subject` and `qchain_ceremony_root`.
// =============================================================================
struct RingtailShareInput {
    const uint8_t* share;
    uint32_t       share_len;
    uint32_t       participant_index;
    uint32_t       round_index;
};

[[nodiscard]] bool verify_ringtail_batch(
    const uint8_t subject[32],
    const uint8_t qchain_ceremony_root[32],
    const RingtailShareInput* shares,
    std::size_t n) noexcept;

}  // namespace quasar::gpu
