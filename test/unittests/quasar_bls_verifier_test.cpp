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

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using quasar::gpu::BLSPublicKey;
using quasar::gpu::BLSSecretKey;
using quasar::gpu::keygen_bls;
using quasar::gpu::sign_subject;
using quasar::gpu::verify_bls_aggregate;
using quasar::gpu::verify_lane;

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
    EXPECT("unwired.mldsa",  !verify_lane(2u, subject, sig, pk));
    EXPECT("unwired.unknown",!verify_lane(99u, subject, sig, pk));
    PASS("bls_unwired_lanes_reject");
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
    std::printf("[quasar_bls_verifier_test] passed=%d failed=%d\n",
                g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
