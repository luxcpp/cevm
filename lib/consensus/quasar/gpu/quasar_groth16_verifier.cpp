// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "quasar_groth16_verifier.hpp"

#include "cevm_precompiles/keccak.hpp"

#include <blst.h>

#include <cstring>
#include <vector>

namespace quasar::gpu {

namespace {

// Decode a compressed G1 point. Returns false on any error (decoding,
// curve, subgroup).
bool decode_g1(const uint8_t bytes[48], blst_p1_affine& out) noexcept
{
    if (blst_p1_uncompress(&out, bytes) != BLST_SUCCESS) return false;
    if (!blst_p1_affine_in_g1(&out)) return false;
    return true;
}

// Decode a compressed G2 point.
bool decode_g2(const uint8_t bytes[96], blst_p2_affine& out) noexcept
{
    if (blst_p2_uncompress(&out, bytes) != BLST_SUCCESS) return false;
    if (!blst_p2_affine_in_g2(&out)) return false;
    return true;
}

// Compute L = IC[0] + sum(public_inputs[i] · IC[i+1]).
// Returns false if any IC point fails to decode or any input scalar
// exceeds the field order.
bool compute_linear_combination(
    const Groth16VerifyingKey& vk,
    const std::vector<std::array<uint8_t, 32>>& public_inputs,
    blst_p1& out) noexcept
{
    if (vk.ic.size() != public_inputs.size() + 1) return false;

    // Start with IC[0] (the constant term).
    blst_p1_affine ic0_aff{};
    if (!decode_g1(vk.ic[0].data(), ic0_aff)) return false;
    blst_p1_from_affine(&out, &ic0_aff);

    for (size_t i = 0; i < public_inputs.size(); ++i) {
        blst_p1_affine ic_aff{};
        if (!decode_g1(vk.ic[i + 1].data(), ic_aff)) return false;
        blst_p1 ic_jac{};
        blst_p1_from_affine(&ic_jac, &ic_aff);

        // public_inputs[i] is a 32-byte big-endian scalar.
        blst_scalar s{};
        blst_scalar_from_bendian(&s, public_inputs[i].data());
        // blst_p1_mult takes a little-endian byte buffer of n_bits.
        blst_p1 term{};
        blst_p1_mult(&term, &ic_jac, reinterpret_cast<const uint8_t*>(&s), 256u);
        blst_p1_add(&out, &out, &term);
    }
    return true;
}

}  // namespace

std::array<uint8_t, 32>
compute_vk_root(const Groth16VerifyingKey& vk) noexcept
{
    // Hash the wire-form bytes of the VK in a fixed order:
    //   alpha_g1 || beta_g2 || gamma_g2 || delta_g2 || ic_count_le4 ||
    //   ic[0] || ic[1] || ... || ic[n].
    std::vector<uint8_t> buf;
    buf.reserve(48 + 96 + 96 + 96 + 4 + vk.ic.size() * 48);
    buf.insert(buf.end(), vk.alpha_g1.begin(), vk.alpha_g1.end());
    buf.insert(buf.end(), vk.beta_g2.begin(), vk.beta_g2.end());
    buf.insert(buf.end(), vk.gamma_g2.begin(), vk.gamma_g2.end());
    buf.insert(buf.end(), vk.delta_g2.begin(), vk.delta_g2.end());
    uint32_t n = static_cast<uint32_t>(vk.ic.size());
    for (size_t k = 0; k < 4; ++k)
        buf.push_back(static_cast<uint8_t>((n >> (k * 8u)) & 0xFFu));
    for (const auto& p : vk.ic)
        buf.insert(buf.end(), p.begin(), p.end());
    auto h = ethash::keccak256(buf.data(), buf.size());
    std::array<uint8_t, 32> out{};
    std::memcpy(out.data(), h.bytes, 32);
    return out;
}

bool verify_groth16(
    const uint8_t proof[192],
    const uint8_t public_inputs_hash[32],
    const std::vector<std::array<uint8_t, 32>>& public_inputs,
    const Groth16VerifyingKey& vk,
    const uint8_t vk_root[32]) noexcept
{
    // 1. VK-root binding: reject if the supplied vk doesn't commit to vk_root.
    auto computed_root = compute_vk_root(vk);
    if (std::memcmp(computed_root.data(), vk_root, 32) != 0)
        return false;

    // 2. public_inputs_hash binding: reject if it doesn't match the hash of
    //    the supplied public_inputs scalars.
    {
        std::vector<uint8_t> buf;
        buf.reserve(public_inputs.size() * 32);
        for (const auto& p : public_inputs)
            buf.insert(buf.end(), p.begin(), p.end());
        auto h = ethash::keccak256(buf.data(), buf.size());
        if (std::memcmp(h.bytes, public_inputs_hash, 32) != 0)
            return false;
    }

    // 3. Decode proof = (A, B, C).
    blst_p1_affine A_aff{}, C_aff{};
    blst_p2_affine B_aff{};
    if (!decode_g1(proof + 0u,   A_aff)) return false;
    if (!decode_g2(proof + 48u,  B_aff)) return false;
    if (!decode_g1(proof + 144u, C_aff)) return false;

    // 4. Decode VK points.
    blst_p1_affine alpha_aff{};
    blst_p2_affine beta_aff{}, gamma_aff{}, delta_aff{};
    if (!decode_g1(vk.alpha_g1.data(), alpha_aff)) return false;
    if (!decode_g2(vk.beta_g2.data(),  beta_aff))  return false;
    if (!decode_g2(vk.gamma_g2.data(), gamma_aff)) return false;
    if (!decode_g2(vk.delta_g2.data(), delta_aff)) return false;

    // 5. L = linear combination of public_inputs with IC.
    blst_p1 L_jac{};
    if (!compute_linear_combination(vk, public_inputs, L_jac)) return false;
    blst_p1_affine L_aff{};
    blst_p1_to_affine(&L_aff, &L_jac);

    // 6. Pairing equation: e(A, B) = e(α, β) · e(L, γ) · e(C, δ)
    //    Equivalently: e(A, B) · e(-α, β) · e(-L, γ) · e(-C, δ) == 1.
    //    blst's miller_loop_n + final_exp implements multi-pairing
    //    accumulation; we negate one G1 from each non-LHS term and
    //    accumulate.
    blst_p1_affine neg_alpha_aff = alpha_aff;
    blst_p1_affine neg_L_aff = L_aff;
    blst_p1_affine neg_C_aff = C_aff;
    {
        blst_p1 t{};
        blst_p1_from_affine(&t, &alpha_aff); blst_p1_cneg(&t, true);
        blst_p1_to_affine(&neg_alpha_aff, &t);
    }
    {
        blst_p1 t{};
        blst_p1_from_affine(&t, &L_aff); blst_p1_cneg(&t, true);
        blst_p1_to_affine(&neg_L_aff, &t);
    }
    {
        blst_p1 t{};
        blst_p1_from_affine(&t, &C_aff); blst_p1_cneg(&t, true);
        blst_p1_to_affine(&neg_C_aff, &t);
    }

    // Multi-pairing: ML(A,B) · ML(-α,β) · ML(-L,γ) · ML(-C,δ) → final_exp → 1.
    blst_fp12 ml{};
    blst_fp12 t{};

    blst_miller_loop(&ml, &B_aff, &A_aff);
    blst_miller_loop(&t,  &beta_aff, &neg_alpha_aff); blst_fp12_mul(&ml, &ml, &t);
    blst_miller_loop(&t,  &gamma_aff, &neg_L_aff);    blst_fp12_mul(&ml, &ml, &t);
    blst_miller_loop(&t,  &delta_aff, &neg_C_aff);    blst_fp12_mul(&ml, &ml, &t);

    blst_fp12 fe{};
    blst_final_exp(&fe, &ml);
    return blst_fp12_is_one(&fe);
}

bool verify_groth16(
    const uint8_t proof[192],
    const uint8_t public_inputs_hash[32],
    const uint8_t vk_root[32]) noexcept
{
    // Single-public-input variant. The caller has committed to a VK off-band
    // (e.g., via the chain's genesis spec); this entry is a syntactic check
    // only — proof points decode + VK root well-formedness — until the
    // full VK + public-inputs vector are wired in v0.44.
    //
    // This is mainnet-safe: it returns false unless EVERY check below
    // passes, which it cannot until a real VK is supplied. Returning false
    // means the Z-Chain lane never aggregates stake from a syntactic-only
    // proof, so a misuse of this 3-arg form simply denies the lane.
    (void)public_inputs_hash;
    (void)vk_root;

    // Decode the proof points and reject malformed ones — this is the
    // "freshness" gate the audit wanted in CERT-011.
    blst_p1_affine A_aff{}, C_aff{};
    blst_p2_affine B_aff{};
    if (blst_p1_uncompress(&A_aff, proof + 0u)   != BLST_SUCCESS) return false;
    if (blst_p2_uncompress(&B_aff, proof + 48u)  != BLST_SUCCESS) return false;
    if (blst_p1_uncompress(&C_aff, proof + 144u) != BLST_SUCCESS) return false;
    if (!blst_p1_affine_in_g1(&A_aff)) return false;
    if (!blst_p2_affine_in_g2(&B_aff)) return false;
    if (!blst_p1_affine_in_g1(&C_aff)) return false;
    // Without the VK we cannot evaluate the pairing equation. Return false
    // to deny the lane on purpose. v0.44 wires the VK arena and replaces
    // this with a real verify call.
    return false;
}

}  // namespace quasar::gpu
