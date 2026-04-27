// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file quasar_sig.hpp
/// Host-side mirror of the Quasar v0.42 GPU vote signature scheme.
///
/// The exact same recipe lives inline in quasar_wave.metal — see
/// `quasar_derive_secret` and `quasar_expected_sig` there. Both halves
/// MUST stay byte-for-byte identical or the GPU verifier will reject
/// host-signed votes.
///
/// v0.42 layout (CERT-001 / CERT-006 / CERT-021):
///   secret_i          = keccak256(domain_tag[16] || chain_id_le8
///                                  || validator_le4 || master_secret[32])
///   subject_with_vote = keccak256(subject[32] || stake_weight_le8
///                                  || validator_le4 || round_le8)
///   sig[0..32]        = keccak256(secret_i || subject_with_vote)
///   sig[32..96]       = zero (reserved for BLS aggregate / Ringtail share)
///
/// CERT-001: master_secret is read from the QUASAR_MASTER_SECRET env var
/// (placeholder until KMS lands). NEVER hard-coded in source.
///
/// Domain tags are lane-specific so a BLS-signed vote cannot satisfy
/// the ML-DSA verifier and vice versa.

#pragma once

#include "cevm_precompiles/keccak.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace quasar::gpu::sig {

// Lane domain tags — must match the constants in quasar_wave.metal.
inline constexpr std::array<uint8_t, 16> kBLSDomain = {
    'Q','U','A','S','A','R','-','B','L','S','-','v','0','3','8',0
};
inline constexpr std::array<uint8_t, 16> kRingtailDomain = {
    'Q','U','A','S','A','R','-','R','T','-','v','0','3','8',0,0
};
inline constexpr std::array<uint8_t, 16> kMLDSADomain = {
    'Q','U','A','S','A','R','-','M','D','S','-','v','0','3','8',0
};

inline const uint8_t* pick_domain(uint32_t sig_kind)
{
    switch (sig_kind) {
        case 0:  return kBLSDomain.data();
        case 1:  return kRingtailDomain.data();
        default: return kMLDSADomain.data();
    }
}

// CERT-001 — master secret loaded from env QUASAR_MASTER_SECRET. Returns
// keccak256(env-string-bytes || "QUASAR-MS-v042-salt") so even a short or
// empty env value produces a uniform 32-byte secret. Caller MUST cache.
inline std::array<uint8_t, 32> load_master_secret()
{
    static constexpr char kSalt[] = "QUASAR-MS-v042-salt";
    const char* env = std::getenv("QUASAR_MASTER_SECRET");
    std::string s = env ? env : "";
    std::vector<uint8_t> buf;
    buf.reserve(s.size() + sizeof(kSalt));
    for (char c : s) buf.push_back(static_cast<uint8_t>(c));
    for (size_t k = 0; k + 1 < sizeof(kSalt); ++k) buf.push_back(static_cast<uint8_t>(kSalt[k]));
    auto h = ethash::keccak256(buf.data(), buf.size());
    std::array<uint8_t, 32> out{};
    std::memcpy(out.data(), h.bytes, 32);
    return out;
}

// Derive secret_i = keccak256(domain || chain_id || validator || master).
inline std::array<uint8_t, 32> derive_secret(uint32_t sig_kind,
                                             uint64_t chain_id,
                                             uint32_t validator_index)
{
    auto master = load_master_secret();
    uint8_t buf[16 + 8 + 4 + 32];
    std::memcpy(buf, pick_domain(sig_kind), 16);
    for (size_t k = 0; k < 8; ++k)
        buf[16 + k] = uint8_t((chain_id >> (k * 8u)) & 0xFFu);
    for (size_t k = 0; k < 4; ++k)
        buf[24 + k] = uint8_t((validator_index >> (k * 8u)) & 0xFFu);
    std::memcpy(buf + 28, master.data(), 32);
    auto h = ethash::keccak256(buf, sizeof(buf));
    std::array<uint8_t, 32> out{};
    std::memcpy(out.data(), h.bytes, 32);
    return out;
}

// CERT-006/021 — Sign with full uint64 round + uint64 stake_weight bound.
// Returns a 96-byte signature (last 64 bytes are zero — reserved for the
// real BLS / Ringtail / Groth16 verifiers landing in v0.43..v0.45).
inline std::vector<uint8_t> sign(uint32_t sig_kind,
                                 uint64_t chain_id,
                                 uint32_t validator_index,
                                 uint64_t round,
                                 uint64_t stake_weight,
                                 const uint8_t subject[32])
{
    std::array<uint8_t, 32> secret = derive_secret(sig_kind, chain_id, validator_index);

    // Step 1: subject_with_vote = keccak(subject || stake_le8 || vi_le4 || round_le8).
    uint8_t buf1[32 + 8 + 4 + 8];
    std::memcpy(buf1, subject, 32);
    for (size_t k = 0; k < 8; ++k)
        buf1[32 + k] = uint8_t((stake_weight >> (k * 8u)) & 0xFFu);
    for (size_t k = 0; k < 4; ++k)
        buf1[40 + k] = uint8_t((validator_index >> (k * 8u)) & 0xFFu);
    for (size_t k = 0; k < 8; ++k)
        buf1[44 + k] = uint8_t((round >> (k * 8u)) & 0xFFu);
    auto h1 = ethash::keccak256(buf1, sizeof(buf1));

    // Step 2: keccak(secret || subject_with_vote).
    uint8_t buf2[32 + 32];
    std::memcpy(buf2, secret.data(), 32);
    std::memcpy(buf2 + 32, h1.bytes, 32);
    auto h2 = ethash::keccak256(buf2, sizeof(buf2));

    std::vector<uint8_t> sig(96, 0);
    std::memcpy(sig.data(), h2.bytes, 32);
    return sig;
}

// =============================================================================
// v0.42 cert ABI — three cert lanes + three cert modes
// =============================================================================
//
// `QuasarCertLane` is the cert ABI taxonomy: BLS aggregate, Ringtail threshold,
// and Groth16-over-ML-DSA. The third lane is `MLDSAGroth16`, NEVER raw `MLDSA`
// — the lane verifies a Groth16 SNARK that proves verification of N ML-DSA-65
// votes, then outputs a 192-byte proof. Raw ML-DSA-65 signing still happens on
// each vote (see `QuasarSigKind::MLDSA`); the Groth16 wrapper is what the cert
// lane ABI exposes.
//
// `QuasarCertMode` selects the round's lane policy:
//   * FastClassical   — only BLS aggregate runs (sub-second finality).
//   * HybridPQAsync   — BLS gates final cert; PQ lanes follow asynchronously
//                       and feed the next round's input.
//   * FullPQBlocking  — final cert blocks on all three lanes.
enum class QuasarCertLane : uint8_t {
    BLSAggregate      = 0,   ///< 48-byte aggregate sig + N-validator bitmap
    RingtailThreshold = 1,   ///< Ring-LWE threshold proof + ceremony root
    MLDSAGroth16      = 2,   ///< 192-byte Groth16 proof + N-validator bitmap
                              ///< (proves verification of N ML-DSA-65 sigs).
                              ///< Never refer to this lane as raw `MLDSA` —
                              ///< raw ML-DSA-65 signing lives on per-vote
                              ///< `QuasarSigKind::MLDSA` and feeds into
                              ///< this Groth16-wrapped cert lane.
};

enum class QuasarCertMode : uint8_t {
    FastClassical   = 0,   ///< BLS only — fast finality path
    HybridPQAsync   = 1,   ///< BLS now, PQ lanes follow async (non-blocking)
    FullPQBlocking  = 2,   ///< All required lanes pass before cert emits
};

// CertArtifact — per-lane proof envelope. The bytes live in a host-managed
// arena (artifact_offset + artifact_len); the verifier resolves them by
// offset. public_inputs_hash is the keccak digest of the lane's public
// inputs (chain roots, attestation_root, …) — the cert binds it so a
// downstream re-verification cannot substitute a different input set.
struct CertArtifact {
    QuasarCertLane lane;
    uint8_t  subject[32];           ///< == desc->certificate_subject
    uint64_t artifact_offset;       ///< arena offset where the proof lives
    uint32_t artifact_len;          ///< proof bytes (48|192|variable)
    uint8_t  public_inputs_hash[32];///< keccak of the public inputs
};

// Compute desc->certificate_subject per CERT-003 / v0.42 spec. Both kernel
// and host MUST agree byte-for-byte. The 9-chain canonical order is fixed:
// P, C, X, Q, Z, A, B, M, F. C reuses parent_block_hash because the cevm
// round IS the C-chain advance.
//
// Hash input (LE everywhere):
//   chain_id(8) || epoch(8) || round(8) || mode(4) || cert_mode(1)
//     || P(32) || C(32) || X(32) || Q(32) || Z(32)
//     || A(32) || B(32) || M(32) || F(32)
//     || attestation_root(32)
//     || parent_state_root(32) || parent_execution_root(32)
//     || gas_limit(8) || base_fee(8)
//
// Total:  8+8+8+4+1 + 32*9 + 32 + 32+32 + 8+8 = 429 bytes.
//
// cert_mode + attestation_root are bound into the subject so a tampered
// descriptor (different mode or different CPU/GPU TEE measurement root)
// produces a different cert subject — even when every chain root matches.
inline std::array<uint8_t, 32> compute_certificate_subject(
    uint64_t chain_id, uint64_t epoch, uint64_t round, uint32_t mode,
    QuasarCertMode cert_mode,
    const uint8_t pchain_validator_root[32],   ///< P
    const uint8_t cchain_block_root[32],       ///< C — parent_block_hash
    const uint8_t xchain_execution_root[32],   ///< X
    const uint8_t qchain_ceremony_root[32],    ///< Q
    const uint8_t zchain_vk_root[32],          ///< Z
    const uint8_t achain_state_root[32],       ///< A
    const uint8_t bchain_state_root[32],       ///< B
    const uint8_t mchain_state_root[32],       ///< M
    const uint8_t fchain_state_root[32],       ///< F
    const uint8_t attestation_root[32],        ///< CPU+GPU TEE measurement
    const uint8_t parent_state_root[32],
    const uint8_t parent_execution_root[32],
    uint64_t gas_limit, uint64_t base_fee)
{
    uint8_t buf[8 + 8 + 8 + 4 + 1 + 32 * 9 + 32 + 32 + 32 + 8 + 8];
    size_t off = 0;
    auto put_le64 = [&](uint64_t v) {
        for (size_t k = 0; k < 8; ++k) buf[off + k] = uint8_t((v >> (k * 8u)) & 0xFFu);
        off += 8;
    };
    auto put_le32 = [&](uint32_t v) {
        for (size_t k = 0; k < 4; ++k) buf[off + k] = uint8_t((v >> (k * 8u)) & 0xFFu);
        off += 4;
    };
    auto put_u8 = [&](uint8_t v) {
        buf[off] = v;
        off += 1;
    };
    auto put_32 = [&](const uint8_t* p) {
        std::memcpy(buf + off, p, 32);
        off += 32;
    };
    put_le64(chain_id);
    put_le64(epoch);
    put_le64(round);
    put_le32(mode);
    put_u8(static_cast<uint8_t>(cert_mode));
    // Canonical 9-chain order: P, C, X, Q, Z, A, B, M, F.
    put_32(pchain_validator_root);
    put_32(cchain_block_root);
    put_32(xchain_execution_root);
    put_32(qchain_ceremony_root);
    put_32(zchain_vk_root);
    put_32(achain_state_root);
    put_32(bchain_state_root);
    put_32(mchain_state_root);
    put_32(fchain_state_root);
    put_32(attestation_root);
    put_32(parent_state_root);
    put_32(parent_execution_root);
    put_le64(gas_limit);
    put_le64(base_fee);
    auto h = ethash::keccak256(buf, off);
    std::array<uint8_t, 32> out{};
    std::memcpy(out.data(), h.bytes, 32);
    return out;
}

}  // namespace quasar::gpu::sig
