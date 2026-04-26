// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file quasar_groth16_verifier.hpp
/// Host Groth16 verifier on BLS12-381 — Quasar 4.0 v0.43.
///
/// The Z-Chain cert lane verifies a Groth16 proof against a verifying-key
/// commitment (`zchain_vk_root`) and a public-inputs commitment. The proof
/// is 192 bytes (G1 + G2 + G1 = 48+96+48):
///   proof[0..48]    : A   (compressed G1, 48 bytes)
///   proof[48..144]  : B   (compressed G2, 96 bytes)
///   proof[144..192] : C   (compressed G1, 48 bytes)
///
/// The verifier checks the standard Groth16 equation:
///   e(A, B) = e(α, β) · e(L, γ) · e(C, δ)
///
/// where (α, β, γ, δ, IC[]) come from the verifying key whose
/// keccak256 commitment is `vk_root`, and L = sum(public_inputs[i] · IC[i]).
///
/// In v0.43 the verifying-key arena is host-resident: the host loads the
/// VK once at chain genesis (or epoch boundary), then for every Z-Chain
/// vote with `vk_root` in the round descriptor matching, the verifier
/// runs the pairing equation against that VK.
///
/// API matches the BLS verifier shim (quasar_bls_verifier.hpp).

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace quasar::gpu {

/// Groth16 verifying key on BLS12-381. Stored host-side; populated once
/// per epoch from the chain's genesis VK + the chain ID's circuit
/// commitments.
struct Groth16VerifyingKey {
    std::array<uint8_t, 48>  alpha_g1;        ///< compressed G1
    std::array<uint8_t, 96>  beta_g2;         ///< compressed G2
    std::array<uint8_t, 96>  gamma_g2;        ///< compressed G2
    std::array<uint8_t, 96>  delta_g2;        ///< compressed G2
    /// IC[0..n_public] — one G1 per public input, plus the constant.
    /// Length is variable; the dot-product against public_inputs gives L.
    std::vector<std::array<uint8_t, 48>> ic;
};

/// Verify a Groth16 proof.
///
/// @param proof              192 bytes: A (G1, 48) || B (G2, 96) || C (G1, 48).
/// @param public_inputs_hash 32 bytes: keccak256 of the public inputs in
///                           canonical wire form. The verifier only needs
///                           the hash for binding; the actual scalars must
///                           be passed via `public_inputs`.
/// @param public_inputs      n public inputs, each a 32-byte big-endian
///                           BLS12-381 scalar (≤ field order).
/// @param vk                 the loaded verifying key.
/// @param vk_root            32-byte keccak256 commitment to the VK; the
///                           verifier rejects if `vk` does not commit to
///                           this root.
/// @return true iff Groth16 equation holds AND the VK matches the root.
[[nodiscard]] bool verify_groth16(
    const uint8_t proof[192],
    const uint8_t public_inputs_hash[32],
    const std::vector<std::array<uint8_t, 32>>& public_inputs,
    const Groth16VerifyingKey& vk,
    const uint8_t vk_root[32]) noexcept;

/// Compute the keccak256 commitment of a Groth16 verifying key.
/// Matches the on-chain `zchain_vk_root` recipe so `verify_groth16`'s
/// vk_root check works without any byte-for-byte storage divergence.
[[nodiscard]] std::array<uint8_t, 32>
compute_vk_root(const Groth16VerifyingKey& vk) noexcept;

/// 4-arg variant matching the API in the punchlist (no public_inputs
/// vector — use the hash directly as a single field element). Useful for
/// the simple binding case where the circuit's only public input is the
/// keccak hash itself.
[[nodiscard]] bool verify_groth16(
    const uint8_t proof[192],
    const uint8_t public_inputs_hash[32],
    const uint8_t vk_root[32]) noexcept;

}  // namespace quasar::gpu
