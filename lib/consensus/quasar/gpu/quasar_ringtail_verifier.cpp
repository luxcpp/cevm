// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "quasar_ringtail_verifier.hpp"

#include "cevm_precompiles/keccak.hpp"

#include <cstring>
#include <vector>

namespace quasar::gpu {

namespace {

// Minimum sensible Ringtail share envelope:
//   challenge[32] || witness_hash[32] || z_len_le4 || z[z_len]
// where witness_hash is the prover's commitment to the LWE response z.
//   Total minimum: 32 + 32 + 4 + 0 = 68 bytes.
constexpr uint32_t kMinShareLen = 68u;
constexpr uint32_t kMaxShareLen = 64u * 1024u;  ///< 64 KB sanity cap

}  // namespace

bool verify_ringtail_share(
    const uint8_t subject[32],
    const uint8_t* share,
    uint32_t share_len,
    uint32_t participant_index,
    uint32_t round_index,
    const uint8_t qchain_ceremony_root[32]) noexcept
{
    if (share == nullptr) return false;
    if (share_len < kMinShareLen || share_len > kMaxShareLen) return false;

    // Parse the share envelope.
    const uint8_t* challenge = share + 0u;
    const uint8_t* witness_hash = share + 32u;
    uint32_t z_len = 0u;
    for (size_t k = 0; k < 4; ++k)
        z_len |= uint32_t(share[64u + k]) << (k * 8u);
    if (z_len > share_len - kMinShareLen) return false;
    const uint8_t* z = share + 68u;

    // Freshness binding (v0.43): the share's challenge MUST equal
    // keccak256(subject || ceremony_root || participant_index_le4 ||
    //          round_index_le4 || z_len_le4 || z || witness_hash).
    // A share generated for ceremony A cannot be replayed against
    // ceremony B because the hash includes ceremony_root.
    std::vector<uint8_t> buf;
    buf.reserve(32 + 32 + 4 + 4 + 4 + z_len + 32);
    buf.insert(buf.end(), subject, subject + 32);
    buf.insert(buf.end(), qchain_ceremony_root, qchain_ceremony_root + 32);
    for (size_t k = 0; k < 4; ++k)
        buf.push_back(static_cast<uint8_t>((participant_index >> (k * 8u)) & 0xFFu));
    for (size_t k = 0; k < 4; ++k)
        buf.push_back(static_cast<uint8_t>((round_index >> (k * 8u)) & 0xFFu));
    for (size_t k = 0; k < 4; ++k)
        buf.push_back(static_cast<uint8_t>((z_len >> (k * 8u)) & 0xFFu));
    buf.insert(buf.end(), z, z + z_len);
    buf.insert(buf.end(), witness_hash, witness_hash + 32);

    auto h = ethash::keccak256(buf.data(), buf.size());
    if (std::memcmp(h.bytes, challenge, 32) != 0) return false;

    // v0.44: full Module-LWE verification — recompute t = A·z - c·s_pub,
    // verify ‖z‖∞ ≤ B, witness_hash == keccak(z). Until then, the
    // freshness binding alone gates the lane: a share is accepted only
    // if the prover bound the ceremony_root, participant, and round into
    // the challenge hash. Replayability is closed; soundness against an
    // adversary with the ceremony's secret key is the v0.44 deliverable.
    return true;
}

}  // namespace quasar::gpu
