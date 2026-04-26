// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "quasar_bls_verifier.hpp"
#include "quasar_groth16_verifier.hpp"
#include "quasar_ringtail_verifier.hpp"

#include <blst.h>

#include <cstring>

namespace quasar::gpu {

bool verify_bls_aggregate(
    const uint8_t subject[32],
    const uint8_t signature[96],
    const uint8_t pk_aggregate[48]) noexcept
{
    // Decode public key (48-byte compressed G1).
    blst_p1_affine pk{};
    if (blst_p1_uncompress(&pk, pk_aggregate) != BLST_SUCCESS)
        return false;
    if (!blst_p1_affine_in_g1(&pk))
        return false;

    // Decode signature (96-byte compressed G2).
    blst_p2_affine sig{};
    if (blst_p2_uncompress(&sig, signature) != BLST_SUCCESS)
        return false;
    if (!blst_p2_affine_in_g2(&sig))
        return false;

    // One-shot core-verify in min-pk form. blst hashes the subject to G2
    // with the supplied DST then performs:
    //     e(pk, H(subject)) == e(G1, sig)
    // BLST_SUCCESS iff the equality holds.
    const BLST_ERROR rc = blst_core_verify_pk_in_g1(
        &pk,
        &sig,
        /*hash_or_encode=*/true,
        subject, 32,
        kQuasarBLSDST.data(), kQuasarBLSDST.size(),
        /*aug=*/nullptr, /*aug_len=*/0);
    return rc == BLST_SUCCESS;
}

bool verify_lane(
    uint32_t sig_kind,
    const uint8_t subject[32],
    const uint8_t signature[96],
    const uint8_t pk_aggregate[48]) noexcept
{
    switch (sig_kind) {
        case 0:  // QuasarSigKind::BLS12_381
            return verify_bls_aggregate(subject, signature, pk_aggregate);
        case 1:  // QuasarSigKind::Ringtail — v0.43
            // Ringtail shares are variable-size and need a (ptr, len)
            // envelope; the 96-byte `signature` slot here can't carry
            // them. Routes through verify_ringtail_share at the engine
            // level where the host can pass the artifact arena.
            // Mainnet-safe: reject on this fixed-shape path.
            (void)pk_aggregate;
            return false;
        case 2:  // QuasarSigKind::MLDSA / Z-Chain Groth16 — v0.43
            // Same shape constraint: a Groth16 proof is 192 bytes (vs
            // 96-byte signature slot). The 3-arg verify_groth16 returns
            // false until v0.44 wires the VK arena, so accept-only
            // syntactic-decode never alone reaches mainnet quorum.
            (void)pk_aggregate;
            return false;
        default:
            return false;
    }
}

bool keygen_bls(
    const uint8_t* ikm, std::size_t ikm_len,
    BLSSecretKey& sk_out,
    BLSPublicKey& pk_out) noexcept
{
    if (ikm == nullptr || ikm_len < 32)
        return false;

    blst_scalar sk{};
    blst_keygen(&sk, ikm, ikm_len, /*info=*/nullptr, /*info_len=*/0);
    blst_bendian_from_scalar(sk_out.bytes.data(), &sk);

    blst_p1 pk_jac{};
    blst_sk_to_pk_in_g1(&pk_jac, &sk);
    blst_p1_compress(pk_out.bytes.data(), &pk_jac);
    return true;
}

bool sign_subject(
    const BLSSecretKey& sk,
    const uint8_t subject[32],
    uint8_t sig_out[96]) noexcept
{
    blst_scalar sk_scalar{};
    blst_scalar_from_bendian(&sk_scalar, sk.bytes.data());

    blst_p2 hash_jac{};
    blst_hash_to_g2(
        &hash_jac,
        subject, 32,
        kQuasarBLSDST.data(), kQuasarBLSDST.size(),
        /*aug=*/nullptr, /*aug_len=*/0);

    blst_p2 sig_jac{};
    blst_sign_pk_in_g1(&sig_jac, &hash_jac, &sk_scalar);
    blst_p2_compress(sig_out, &sig_jac);
    return true;
}

}  // namespace quasar::gpu
