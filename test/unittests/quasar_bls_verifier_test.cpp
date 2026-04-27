// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Real BLS12-381 aggregate verifier tests for Quasar 4.0 (v0.41).
// Activation 2026-02-14. Replaces the v0.38 keccak-stub with blst.
//
// Coverage:
//   1. test_bls_real_aggregate_verify_3_signers — round trip across 3 keypairs
//   2. test_bls_tampered_signature_rejected     — flip 1 bit in 96-byte sig
//   3. test_bls_wrong_subject_rejected          — verify against different subject
//   4. test_bls_wrong_pk_rejected               — verify with unrelated pk
//   5. test_bls_unwired_lanes_reject            — Ringtail/ML-DSA fail closed
//
// Self-contained — no GoogleTest dep, matches the style of
// quasar_gpu_engine_test.mm (EXPECT / PASS macros).

#include "consensus/quasar/gpu/quasar_bls_verifier.hpp"
#include "consensus/quasar/gpu/quasar_groth16_verifier.hpp"
#include "consensus/quasar/gpu/quasar_ringtail_verifier.hpp"
#include "cevm_precompiles/keccak.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using quasar::gpu::BLSPublicKey;
using quasar::gpu::BLSSecretKey;
using quasar::gpu::keygen_bls;
using quasar::gpu::sign_subject;
using quasar::gpu::verify_bls_aggregate;
using quasar::gpu::verify_lane;
using quasar::gpu::Groth16VerifyingKey;
using quasar::gpu::compute_vk_root;
using quasar::gpu::verify_ringtail_share;

namespace {

int g_passed = 0;
int g_failed = 0;

#define EXPECT(name, cond)                                                  \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::printf("  FAIL[%s]: %s\n", (name), #cond);                 \
            std::fflush(stdout);                                            \
            ++g_failed;                                                     \
            return;                                                         \
        }                                                                   \
    } while (0)

#define PASS(name)                                                          \
    do {                                                                    \
        std::printf("  ok  : %s\n", (name));                                \
        std::fflush(stdout);                                                \
        ++g_passed;                                                         \
    } while (0)

// Deterministic 32-byte IKM seeded by index — every keypair distinct.
std::array<uint8_t, 32> ikm_for(uint8_t idx)
{
    std::array<uint8_t, 32> out{};
    for (uint8_t k = 0; k < 32; ++k)
        out[k] = static_cast<uint8_t>(0xA0u ^ idx ^ k);
    return out;
}

// 1. Real aggregate verify across 3 signers.
//
// Each signer signs the SAME subject with their own SK. We verify each
// signature independently using its corresponding PK. (Per-validator
// votes are aggregated into a QC by the host downstream; full
// aggregate-of-signatures verification lives in the QC composer in v0.42+.)
void test_bls_real_aggregate_verify_3_signers()
{
    constexpr int N = 3;
    uint8_t subject[32]{};
    for (uint8_t k = 0; k < 32; ++k) subject[k] = static_cast<uint8_t>(0x42u ^ k);

    BLSSecretKey sks[N];
    BLSPublicKey pks[N];
    for (int i = 0; i < N; ++i) {
        auto ikm = ikm_for(static_cast<uint8_t>(i + 1));
        EXPECT("agg.keygen", keygen_bls(ikm.data(), ikm.size(), sks[i], pks[i]));
    }

    uint8_t sigs[N][96]{};
    for (int i = 0; i < N; ++i) {
        EXPECT("agg.sign", sign_subject(sks[i], subject, sigs[i]));
    }

    for (int i = 0; i < N; ++i) {
        EXPECT("agg.verify", verify_bls_aggregate(subject, sigs[i], pks[i].bytes.data()));
    }
    // Lane dispatcher: sig_kind = 0 (BLS) returns the verifier verdict.
    for (int i = 0; i < N; ++i) {
        EXPECT("agg.lane", verify_lane(0u, subject, sigs[i], pks[i].bytes.data()));
    }
    PASS("bls_real_aggregate_verify_3_signers");
}

// 2. Tampered signature rejected. Flip one bit in the 96-byte sig.
void test_bls_tampered_signature_rejected()
{
    BLSSecretKey sk{};
    BLSPublicKey pk{};
    auto ikm = ikm_for(0x10u);
    EXPECT("tamper.keygen", keygen_bls(ikm.data(), ikm.size(), sk, pk));

    uint8_t subject[32]{};
    for (uint8_t k = 0; k < 32; ++k) subject[k] = static_cast<uint8_t>(0xC0u ^ k);

    uint8_t sig[96]{};
    EXPECT("tamper.sign", sign_subject(sk, subject, sig));
    EXPECT("tamper.baseline", verify_bls_aggregate(subject, sig, pk.bytes.data()));

    // Byte 48 sits inside the second Fp coord of the compressed G2 point —
    // safely past the 3 high-bit encoding flags in byte 0.
    sig[48] ^= 0x01u;
    EXPECT("tamper.rejected", !verify_bls_aggregate(subject, sig, pk.bytes.data()));
    EXPECT("tamper.lane_rejected", !verify_lane(0u, subject, sig, pk.bytes.data()));
    PASS("bls_tampered_signature_rejected");
}

// 3. Wrong subject rejected.
void test_bls_wrong_subject_rejected()
{
    BLSSecretKey sk{};
    BLSPublicKey pk{};
    auto ikm = ikm_for(0x20u);
    EXPECT("wsubj.keygen", keygen_bls(ikm.data(), ikm.size(), sk, pk));

    uint8_t subject_a[32]{};
    uint8_t subject_b[32]{};
    for (uint8_t k = 0; k < 32; ++k) {
        subject_a[k] = static_cast<uint8_t>(0x11u ^ k);
        subject_b[k] = static_cast<uint8_t>(0x22u ^ k);
    }

    uint8_t sig_a[96]{};
    EXPECT("wsubj.sign",  sign_subject(sk, subject_a, sig_a));
    EXPECT("wsubj.right", verify_bls_aggregate(subject_a, sig_a, pk.bytes.data()));
    EXPECT("wsubj.wrong", !verify_bls_aggregate(subject_b, sig_a, pk.bytes.data()));
    EXPECT("wsubj.lane",  !verify_lane(0u, subject_b, sig_a, pk.bytes.data()));
    PASS("bls_wrong_subject_rejected");
}

// 4. Wrong pk rejected.
void test_bls_wrong_pk_rejected()
{
    BLSSecretKey sk_real{}, sk_other{};
    BLSPublicKey pk_real{}, pk_other{};
    auto ikm_real  = ikm_for(0x30u);
    auto ikm_other = ikm_for(0x31u);
    EXPECT("wpk.keygen.real",  keygen_bls(ikm_real.data(),  ikm_real.size(),  sk_real,  pk_real));
    EXPECT("wpk.keygen.other", keygen_bls(ikm_other.data(), ikm_other.size(), sk_other, pk_other));
    EXPECT("wpk.distinct",
           std::memcmp(pk_real.bytes.data(), pk_other.bytes.data(), 48) != 0);

    uint8_t subject[32]{};
    for (uint8_t k = 0; k < 32; ++k) subject[k] = static_cast<uint8_t>(0x77u ^ k);

    uint8_t sig[96]{};
    EXPECT("wpk.sign", sign_subject(sk_real, subject, sig));
    EXPECT("wpk.right", verify_bls_aggregate(subject, sig, pk_real.bytes.data()));
    EXPECT("wpk.wrong", !verify_bls_aggregate(subject, sig, pk_other.bytes.data()));
    EXPECT("wpk.lane",  !verify_lane(0u, subject, sig, pk_other.bytes.data()));
    PASS("bls_wrong_pk_rejected");
}

// 5. Unwired lanes (Ringtail / ML-DSA / unknown) always reject.
// Mainnet-safe: never aggregate stake from a fake signature.
void test_bls_unwired_lanes_reject()
{
    uint8_t subject[32]{};
    uint8_t sig[96]{};
    uint8_t pk[48]{};
    EXPECT("unwired.rt",     !verify_lane(1u, subject, sig, pk));
    EXPECT("unwired.mldsa_groth16",  !verify_lane(2u, subject, sig, pk));
    EXPECT("unwired.unknown",!verify_lane(99u, subject, sig, pk));
    PASS("bls_unwired_lanes_reject");
}

// =============================================================================
// v0.43 — Groth16 verifier tests
// =============================================================================
//
// Building a real Groth16 fixture requires (a) a verifying key generated by a
// trusted setup ceremony and (b) a proof generated by a circuit. Neither
// is in-tree. The tests below verify the *plumbing*: the verifier rejects
// well-formed but unbacked proofs, decodes proof points correctly, and
// produces deterministic vk_root commitments.

// Build a synthetic VK for plumbing tests. The points are NOT a valid VK
// (they reuse the BLS generator points); pairing equation will fail, but
// the keccak commitment + decode path are exercised end-to-end.
Groth16VerifyingKey make_synthetic_vk(uint8_t seed)
{
    Groth16VerifyingKey vk;
    // Encode the BLS12-381 G1 generator (compressed, with seed perturbation
    // applied via the IKM; we keygen and reuse).
    uint8_t ikm[32];
    for (int i = 0; i < 32; ++i) ikm[i] = static_cast<uint8_t>(seed + i);
    quasar::gpu::BLSSecretKey sk{};
    quasar::gpu::BLSPublicKey pk{};
    quasar::gpu::keygen_bls(ikm, 32, sk, pk);
    // alpha_g1 = pk (compressed G1, 48 bytes).
    std::memcpy(vk.alpha_g1.data(), pk.bytes.data(), 48);
    // For G2 fields we re-use the BLS signature output (96-byte G2) but
    // it's not a valid Groth16 element either — fine for plumbing tests.
    uint8_t subject[32]{};
    for (int i = 0; i < 32; ++i) subject[i] = static_cast<uint8_t>(0xAA ^ i);
    uint8_t sig96[96]{};
    quasar::gpu::sign_subject(sk, subject, sig96);
    std::memcpy(vk.beta_g2.data(),  sig96, 96);
    std::memcpy(vk.gamma_g2.data(), sig96, 96);
    std::memcpy(vk.delta_g2.data(), sig96, 96);
    // ic[0] only — supports 0 public inputs in this synthetic case.
    std::array<uint8_t, 48> ic0{};
    std::memcpy(ic0.data(), pk.bytes.data(), 48);
    vk.ic.push_back(ic0);
    return vk;
}

// 6. compute_vk_root determinism — same VK twice → same root.
void test_groth16_vk_root_deterministic()
{
    auto vk = make_synthetic_vk(0x10);
    auto r1 = compute_vk_root(vk);
    auto r2 = compute_vk_root(vk);
    EXPECT("g16_root.deterministic", std::memcmp(r1.data(), r2.data(), 32) == 0);
    PASS("groth16_vk_root_deterministic");
}

// 7. compute_vk_root binds every field — perturb each, get a different root.
void test_groth16_vk_root_binds_fields()
{
    auto base = make_synthetic_vk(0x20);
    auto h0   = compute_vk_root(base);

    auto vk_alpha = base; vk_alpha.alpha_g1[0] ^= 0xFF;
    auto vk_beta  = base; vk_beta.beta_g2[0]   ^= 0xFF;
    auto vk_gamma = base; vk_gamma.gamma_g2[0] ^= 0xFF;
    auto vk_delta = base; vk_delta.delta_g2[0] ^= 0xFF;

    EXPECT("g16_root.alpha", std::memcmp(h0.data(), compute_vk_root(vk_alpha).data(), 32) != 0);
    EXPECT("g16_root.beta",  std::memcmp(h0.data(), compute_vk_root(vk_beta).data(),  32) != 0);
    EXPECT("g16_root.gamma", std::memcmp(h0.data(), compute_vk_root(vk_gamma).data(), 32) != 0);
    EXPECT("g16_root.delta", std::memcmp(h0.data(), compute_vk_root(vk_delta).data(), 32) != 0);
    PASS("groth16_vk_root_binds_fields");
}

// 8. Malformed proof (bad G1/G2 point bytes) is rejected by the syntactic
// 3-arg verify_groth16 path.
void test_groth16_rejects_malformed_proof()
{
    uint8_t proof[192]{};
    // Fill with 0xFF — guaranteed not a valid compressed BLS12-381 point.
    std::memset(proof, 0xFF, 192);
    uint8_t pi_hash[32]{};
    uint8_t vk_root[32]{};
    EXPECT("g16_malformed.rejects",
           !quasar::gpu::verify_groth16(proof, pi_hash, vk_root));
    PASS("groth16_rejects_malformed_proof");
}

// 9. Wrong vk_root — a well-formed VK that does not match the supplied root
// causes verify_groth16 to return false (no public-input vector path).
void test_groth16_wrong_vk_root_rejected()
{
    auto vk = make_synthetic_vk(0x30);
    auto wrong_root = compute_vk_root(make_synthetic_vk(0x31));
    uint8_t proof[192]{};
    std::memcpy(proof + 0u,   vk.alpha_g1.data(), 48);
    std::memcpy(proof + 48u,  vk.beta_g2.data(),  96);
    std::memcpy(proof + 144u, vk.alpha_g1.data(), 48);
    uint8_t pi_hash[32]{};
    auto h = ethash::keccak256(nullptr, 0);
    std::memcpy(pi_hash, h.bytes, 32);
    std::vector<std::array<uint8_t, 32>> empty_inputs;
    EXPECT("g16_wrong_root.rejects",
           !quasar::gpu::verify_groth16(proof, pi_hash, empty_inputs, vk,
                                        wrong_root.data()));
    PASS("groth16_wrong_vk_root_rejected");
}

// =============================================================================
// v0.43 — Ringtail verifier tests
// =============================================================================
//
// Ringtail freshness binding: the share's challenge MUST equal
// keccak(subject || ceremony_root || participant || round || z_len || z ||
//        witness_hash). Replay across ceremonies / rounds is impossible.
// Until v0.44 lands the full Module-LWE response check, that's the
// strongest property we can test in unit form.

std::vector<uint8_t> make_ringtail_share(
    const uint8_t subject[32],
    uint32_t participant_index, uint32_t round_index,
    const uint8_t ceremony_root[32],
    const std::vector<uint8_t>& z,
    const uint8_t witness_hash[32])
{
    // Build the bytes-to-hash for the challenge.
    std::vector<uint8_t> tohash;
    tohash.insert(tohash.end(), subject, subject + 32);
    tohash.insert(tohash.end(), ceremony_root, ceremony_root + 32);
    for (size_t k = 0; k < 4; ++k)
        tohash.push_back(static_cast<uint8_t>((participant_index >> (k * 8u)) & 0xFFu));
    for (size_t k = 0; k < 4; ++k)
        tohash.push_back(static_cast<uint8_t>((round_index >> (k * 8u)) & 0xFFu));
    uint32_t z_len = static_cast<uint32_t>(z.size());
    for (size_t k = 0; k < 4; ++k)
        tohash.push_back(static_cast<uint8_t>((z_len >> (k * 8u)) & 0xFFu));
    tohash.insert(tohash.end(), z.begin(), z.end());
    tohash.insert(tohash.end(), witness_hash, witness_hash + 32);
    auto h = ethash::keccak256(tohash.data(), tohash.size());

    // Pack the share envelope.
    std::vector<uint8_t> share;
    share.insert(share.end(), h.bytes, h.bytes + 32);          // challenge
    share.insert(share.end(), witness_hash, witness_hash + 32);// witness_hash
    for (size_t k = 0; k < 4; ++k)
        share.push_back(static_cast<uint8_t>((z_len >> (k * 8u)) & 0xFFu));
    share.insert(share.end(), z.begin(), z.end());
    return share;
}

// 10. Round-trip — a well-formed share verifies against its own ceremony.
void test_ringtail_share_roundtrip()
{
    uint8_t subject[32]; for (int i=0;i<32;++i) subject[i] = static_cast<uint8_t>(0x42 ^ i);
    uint8_t ceremony[32]; for (int i=0;i<32;++i) ceremony[i] = static_cast<uint8_t>(0x55 ^ i);
    uint8_t witness[32]; for (int i=0;i<32;++i) witness[i] = static_cast<uint8_t>(0x99 ^ i);
    std::vector<uint8_t> z(64);
    for (size_t i = 0; i < z.size(); ++i) z[i] = static_cast<uint8_t>(i);

    auto share = make_ringtail_share(subject, 5u, 1u, ceremony, z, witness);
    EXPECT("rt_rt.verify",
           verify_ringtail_share(subject, share.data(),
                                 static_cast<uint32_t>(share.size()),
                                 5u, 1u, ceremony));
    PASS("ringtail_share_roundtrip");
}

// 11. Cross-ceremony replay rejected — share built for ceremony A fails on B.
void test_ringtail_cross_ceremony_rejected()
{
    uint8_t subject[32]; for (int i=0;i<32;++i) subject[i] = static_cast<uint8_t>(0x42 ^ i);
    uint8_t ceremony_a[32]; for (int i=0;i<32;++i) ceremony_a[i] = static_cast<uint8_t>(0x55 ^ i);
    uint8_t ceremony_b[32]; for (int i=0;i<32;++i) ceremony_b[i] = static_cast<uint8_t>(0x66 ^ i);
    uint8_t witness[32]{};
    std::vector<uint8_t> z(48, 0xCDu);

    auto share = make_ringtail_share(subject, 3u, 1u, ceremony_a, z, witness);
    EXPECT("rt_xcer.replay_rejected",
           !verify_ringtail_share(subject, share.data(),
                                  static_cast<uint32_t>(share.size()),
                                  3u, 1u, ceremony_b));
    PASS("ringtail_cross_ceremony_rejected");
}

// 12. Cross-round replay rejected — share built for round 1 fails on round 2.
void test_ringtail_cross_round_rejected()
{
    uint8_t subject[32]; for (int i=0;i<32;++i) subject[i] = static_cast<uint8_t>(0x12 ^ i);
    uint8_t ceremony[32]; for (int i=0;i<32;++i) ceremony[i] = static_cast<uint8_t>(0x34 ^ i);
    uint8_t witness[32]{};
    std::vector<uint8_t> z(72, 0x77u);

    auto share = make_ringtail_share(subject, 7u, 1u, ceremony, z, witness);
    EXPECT("rt_xround.r2_rejected",
           !verify_ringtail_share(subject, share.data(),
                                  static_cast<uint32_t>(share.size()),
                                  7u, 2u, ceremony));
    PASS("ringtail_cross_round_rejected");
}

// 13. Malformed share (too short, too long, null) rejected.
void test_ringtail_malformed_rejected()
{
    uint8_t subject[32]{};
    uint8_t ceremony[32]{};
    uint8_t tiny[16]{};
    EXPECT("rt_mal.null",  !verify_ringtail_share(subject, nullptr, 0u, 0, 0, ceremony));
    EXPECT("rt_mal.short", !verify_ringtail_share(subject, tiny, 16u, 0, 0, ceremony));
    std::vector<uint8_t> huge(70u * 1024u);  // > kMaxShareLen
    EXPECT("rt_mal.huge",
           !verify_ringtail_share(subject, huge.data(),
                                  static_cast<uint32_t>(huge.size()),
                                  0, 0, ceremony));
    PASS("ringtail_malformed_rejected");
}

}  // namespace

int main(int /*argc*/, char** /*argv*/)
{
    setvbuf(stdout, nullptr, _IOLBF, 0);
    std::printf("[quasar_bls_verifier_test] starting\n");
    std::fflush(stdout);
    test_bls_real_aggregate_verify_3_signers();
    test_bls_tampered_signature_rejected();
    test_bls_wrong_subject_rejected();
    test_bls_wrong_pk_rejected();
    test_bls_unwired_lanes_reject();
    // v0.43 — Groth16 verifier (Z-Chain).
    test_groth16_vk_root_deterministic();
    test_groth16_vk_root_binds_fields();
    test_groth16_rejects_malformed_proof();
    test_groth16_wrong_vk_root_rejected();
    // v0.43 — Ringtail verifier (Q-Chain).
    test_ringtail_share_roundtrip();
    test_ringtail_cross_ceremony_rejected();
    test_ringtail_cross_round_rejected();
    test_ringtail_malformed_rejected();
    std::printf("[quasar_bls_verifier_test] passed=%d failed=%d\n",
                g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
