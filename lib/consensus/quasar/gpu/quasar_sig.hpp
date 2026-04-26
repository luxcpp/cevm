// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file quasar_sig.hpp
/// Host-side mirror of the Quasar v0.38 GPU vote signature scheme.
///
/// The exact same recipe lives inline in quasar_wave.metal — see
/// `quasar_derive_secret` and `quasar_expected_sig` there. Both halves
/// MUST stay byte-for-byte identical or the GPU verifier will reject
/// host-signed votes.
///
/// Layout:
///   secret_i    = keccak256(domain_tag[16] || chain_id_le8
///                           || validator_le4 || master_secret_le32)
///   sig[0..32]  = keccak256(secret_i || subject[32] || round_le4)
///   sig[32..96] = zero (reserved for future BLS aggregate / Ringtail
///                       share material)
///
/// Domain tags are lane-specific so a BLS-signed vote cannot satisfy
/// the ML-DSA verifier and vice versa.

#pragma once

#include "cevm_precompiles/keccak.hpp"

#include <array>
#include <cstdint>
#include <cstring>
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

// Master secret — must match kQuasarMasterSecret in quasar_wave.metal.
inline constexpr std::array<uint8_t, 32> kMasterSecret = {
    0x51,0x55,0x41,0x53,0x41,0x52,0x2D,0x76,0x30,0x33,0x38,0x2D,0x6D,0x61,0x73,0x74,
    0x65,0x72,0x2D,0x73,0x65,0x63,0x72,0x65,0x74,0x2D,0x73,0x68,0x61,0x72,0x65,0x64,
};

inline const uint8_t* pick_domain(uint32_t sig_kind)
{
    switch (sig_kind) {
        case 0:  return kBLSDomain.data();
        case 1:  return kRingtailDomain.data();
        default: return kMLDSADomain.data();
    }
}

// Derive secret_i = keccak256(domain || chain_id || validator || master).
inline std::array<uint8_t, 32> derive_secret(uint32_t sig_kind,
                                             uint64_t chain_id,
                                             uint32_t validator_index)
{
    uint8_t buf[16 + 8 + 4 + 32];
    std::memcpy(buf, pick_domain(sig_kind), 16);
    for (size_t k = 0; k < 8; ++k)
        buf[16 + k] = uint8_t((chain_id >> (k * 8u)) & 0xFFu);
    for (size_t k = 0; k < 4; ++k)
        buf[24 + k] = uint8_t((validator_index >> (k * 8u)) & 0xFFu);
    std::memcpy(buf + 28, kMasterSecret.data(), 32);
    auto h = ethash::keccak256(buf, sizeof(buf));
    std::array<uint8_t, 32> out{};
    std::memcpy(out.data(), h.bytes, 32);
    return out;
}

// Sign: sig[0..32] = keccak256(secret || subject[32] || round_le4).
// Returns a 96-byte signature (last 64 bytes are zero — reserved).
inline std::vector<uint8_t> sign(uint32_t sig_kind,
                                 uint64_t chain_id,
                                 uint32_t validator_index,
                                 uint32_t round,
                                 const uint8_t subject[32])
{
    std::array<uint8_t, 32> secret = derive_secret(sig_kind, chain_id, validator_index);
    uint8_t buf[32 + 32 + 4];
    std::memcpy(buf, secret.data(), 32);
    std::memcpy(buf + 32, subject, 32);
    for (size_t k = 0; k < 4; ++k)
        buf[64 + k] = uint8_t((round >> (k * 8u)) & 0xFFu);
    auto h = ethash::keccak256(buf, sizeof(buf));
    std::vector<uint8_t> sig(96, 0);
    std::memcpy(sig.data(), h.bytes, 32);
    return sig;
}

}  // namespace quasar::gpu::sig
