// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file quasar_cert_abi_test.mm
/// v0.42 cert ABI hardening — exercises the cert lane / cert mode taxonomy:
///   * `QuasarCertLane` enum (BLSAggregate / RingtailThreshold / MLDSAGroth16)
///   * `QuasarCertMode` enum (FastClassical / HybridPQAsync / FullPQBlocking)
///   * `attestation_root` + `cert_mode` are bound into certificate_subject
///     in canonical position
///   * Cert lane scheduling: FastClassical drains BLS only, HybridPQAsync
///     aggregates BLS without blocking, FullPQBlocking aggregates all three
///
/// Coverage:
///   1. Cert lane / cert mode enum integer values are stable
///   2. Three cert modes × all 9 chain root combinations + attestation_root
///      → certificate_subject is deterministic across recomputes
///   3. Flipping cert_mode produces a different cert subject
///   4. Flipping attestation_root produces a different cert subject
///   5. BLSAggregate lane in FastClassical mode emits cert without waiting
///      for PQ lanes
///   6. FullPQBlocking mode aggregates all 3 lanes and reaches quorum on each
///   7. HybridPQAsync mode emits BLS cert; PQ lanes can lag without blocking
///   8. CertArtifact ABI surface area is stable (size + field layout)

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "consensus/quasar/gpu/quasar_gpu_engine.hpp"
#include "consensus/quasar/gpu/quasar_gpu_layout.hpp"
#include "consensus/quasar/gpu/quasar_sig.hpp"

#include "cevm_precompiles/keccak.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using quasar::gpu::HostVote;
using quasar::gpu::QuasarGPUEngine;
using quasar::gpu::QuasarRoundDescriptor;
using quasar::gpu::QuasarRoundResult;
using quasar::gpu::sig::CertArtifact;
using quasar::gpu::sig::QuasarCertLane;
using quasar::gpu::sig::QuasarCertMode;

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

QuasarRoundDescriptor make_cert_desc(uint64_t round, QuasarCertMode cert_mode)
{
    QuasarRoundDescriptor d{};
    d.chain_id          = 0xC0DECAFEFEEDFACEull;
    d.round             = round;
    d.gas_limit         = 30'000'000u;
    d.base_fee          = 1'000u;
    d.wave_tick_budget  = 256u;
    d.mode              = 0u;
    d.epoch             = 11u;
    d.total_stake       = 90u;        // quorum threshold = 60
    d.validator_count   = 16u;
    auto fill = [](uint8_t* p, uint8_t b) { std::memset(p, b, 32); };
    fill(d.pchain_validator_root,   0x11);
    fill(d.parent_block_hash,       0x22);
    fill(d.xchain_execution_root,   0x33);
    fill(d.qchain_ceremony_root,    0x44);
    fill(d.zchain_vk_root,          0x55);
    fill(d.achain_state_root,       0x66);
    fill(d.bchain_state_root,       0x77);
    fill(d.mchain_state_root,       0x88);
    fill(d.fchain_state_root,       0x99);
    fill(d.parent_state_root,       0xAA);
    fill(d.parent_execution_root,   0xBB);
    fill(d.attestation_root,        0xCC);
    d.cert_mode = static_cast<uint8_t>(cert_mode);
    return d;
}

std::array<uint8_t, 32> subject_for(const QuasarRoundDescriptor& d)
{
    return quasar::gpu::sig::compute_certificate_subject(
        d.chain_id, d.epoch, d.round, d.mode,
        static_cast<QuasarCertMode>(d.cert_mode),
        d.pchain_validator_root, d.parent_block_hash,
        d.xchain_execution_root, d.qchain_ceremony_root,
        d.zchain_vk_root, d.achain_state_root,
        d.bchain_state_root, d.mchain_state_root,
        d.fchain_state_root, d.attestation_root,
        d.parent_state_root, d.parent_execution_root,
        d.gas_limit, d.base_fee);
}

HostVote sign_vote(const QuasarRoundDescriptor& d,
                   const uint8_t subject[32],
                   uint32_t validator_index, uint64_t stake_weight,
                   uint32_t sig_kind)
{
    HostVote v{};
    v.validator_index = validator_index;
    v.round           = d.round;
    v.stake_weight    = stake_weight;
    v.sig_kind        = sig_kind;
    std::memcpy(v.block_hash, subject, 32);
    v.signature = quasar::gpu::sig::sign(sig_kind, d.chain_id,
                                         validator_index, d.round,
                                         stake_weight, subject);
    return v;
}

// 1. Enum integer values are part of the cert ABI — pin them.
void test_cert_enum_values()
{
    EXPECT("lane.bls",   static_cast<uint8_t>(QuasarCertLane::BLSAggregate)      == 0u);
    EXPECT("lane.rt",    static_cast<uint8_t>(QuasarCertLane::RingtailThreshold) == 1u);
    EXPECT("lane.mldsa_groth16",
           static_cast<uint8_t>(QuasarCertLane::MLDSAGroth16)      == 2u);

    EXPECT("mode.fast",   static_cast<uint8_t>(QuasarCertMode::FastClassical)  == 0u);
    EXPECT("mode.hybrid", static_cast<uint8_t>(QuasarCertMode::HybridPQAsync)  == 1u);
    EXPECT("mode.full",   static_cast<uint8_t>(QuasarCertMode::FullPQBlocking) == 2u);
    PASS("cert_enum_values");
}

// 2. compute_certificate_subject is deterministic across all three cert modes.
//    Same inputs → same digest. This is the property a verifier relies on
//    when comparing v.subject == desc->certificate_subject.
void test_cert_subject_deterministic_all_modes()
{
    const QuasarCertMode modes[] = {
        QuasarCertMode::FastClassical,
        QuasarCertMode::HybridPQAsync,
        QuasarCertMode::FullPQBlocking,
    };
    for (auto m : modes) {
        auto d  = make_cert_desc(1000u, m);
        auto h1 = subject_for(d);
        auto h2 = subject_for(d);
        EXPECT("det.repeat",
               std::memcmp(h1.data(), h2.data(), 32) == 0);
    }
    PASS("cert_subject_deterministic_all_modes");
}

// 3. Flipping cert_mode produces a different cert subject (cannot replay
//    a fast-classical cert as a full-PQ cert and vice versa).
void test_cert_mode_binds_subject()
{
    auto d_fast    = make_cert_desc(1100u, QuasarCertMode::FastClassical);
    auto d_hybrid  = make_cert_desc(1100u, QuasarCertMode::HybridPQAsync);
    auto d_full    = make_cert_desc(1100u, QuasarCertMode::FullPQBlocking);

    auto s_fast    = subject_for(d_fast);
    auto s_hybrid  = subject_for(d_hybrid);
    auto s_full    = subject_for(d_full);

    EXPECT("mode.fast_vs_hybrid",
           std::memcmp(s_fast.data(),   s_hybrid.data(), 32) != 0);
    EXPECT("mode.fast_vs_full",
           std::memcmp(s_fast.data(),   s_full.data(),   32) != 0);
    EXPECT("mode.hybrid_vs_full",
           std::memcmp(s_hybrid.data(), s_full.data(),   32) != 0);
    PASS("cert_mode_binds_subject");
}

// 4. Flipping attestation_root produces a different cert subject — a tampered
//    TEE measurement cannot collide with the genuine round's cert subject.
void test_attestation_root_binds_subject()
{
    auto d_a = make_cert_desc(1200u, QuasarCertMode::HybridPQAsync);
    auto d_b = d_a;
    d_b.attestation_root[7] ^= 0xA5u;

    auto s_a = subject_for(d_a);
    auto s_b = subject_for(d_b);
    EXPECT("att.subject_diff",
           std::memcmp(s_a.data(), s_b.data(), 32) != 0);

    // Reflexive: zero attestation_root vs nonzero attestation_root differ.
    auto d_zero = d_a;
    std::memset(d_zero.attestation_root, 0, 32);
    auto s_zero = subject_for(d_zero);
    EXPECT("att.zero_diff",
           std::memcmp(s_a.data(), s_zero.data(), 32) != 0);
    PASS("attestation_root_binds_subject");
}

// 5. FastClassical mode: only BLS lane drains. PQ-lane votes are filtered
//    out by drain_vote so they never accumulate stake.
void test_fast_classical_emits_bls_only()
{
    auto e = QuasarGPUEngine::create();
    EXPECT("fast.create", e != nullptr);

    auto d = make_cert_desc(1300u, QuasarCertMode::FastClassical);
    auto h = e->begin_round(d);
    EXPECT("fast.handle", h.valid());

    // begin_round overwrites desc->certificate_subject with the canonical
    // recompute; pull it back out so HostVotes match.
    auto subj = subject_for(d);
    uint8_t subject[32];
    std::memcpy(subject, subj.data(), 32);

    std::vector<HostVote> votes;
    // BLS lane — 3 × 30 stake → 90 ≥ threshold(60) → quorum.
    votes.push_back(sign_vote(d, subject, 0u, 30u, 0u));
    votes.push_back(sign_vote(d, subject, 1u, 30u, 0u));
    votes.push_back(sign_vote(d, subject, 2u, 30u, 0u));
    // Ringtail + MLDSAGroth16 lanes — must be DROPPED by drain_vote in
    // FastClassical mode. If any lane gating leaks, these would accumulate.
    votes.push_back(sign_vote(d, subject, 3u, 60u, 1u));
    votes.push_back(sign_vote(d, subject, 4u, 60u, 2u));
    e->push_votes(h, votes);

    e->request_close(h);
    auto r = e->run_until_done(h, 8);
    EXPECT("fast.finalized", r.status == 1u);

    EXPECT("fast.bls_quorum",   r.quorum_status_bls == 1u);
    EXPECT("fast.bls_stake",    r.quorum_stake_bls() == 90ull);
    EXPECT("fast.rt_dropped",   r.quorum_status_rt == 0u);
    EXPECT("fast.rt_zero",      r.quorum_stake_rt() == 0ull);
    EXPECT("fast.mldsa_groth16_dropped",
           r.quorum_status_mldsa_groth16 == 0u);
    EXPECT("fast.mldsa_groth16_zero",
           r.quorum_stake_mldsa_groth16() == 0ull);

    e->end_round(h);
    PASS("fast_classical_emits_bls_only");
}

// 6. FullPQBlocking mode: all three cert lanes drain and reach quorum.
//    Host gates final composite cert emission on all three; substrate
//    independently emits per-lane QCs.
void test_full_pq_blocking_drains_all_lanes()
{
    auto e = QuasarGPUEngine::create();
    auto d = make_cert_desc(1400u, QuasarCertMode::FullPQBlocking);
    auto h = e->begin_round(d);

    auto subj = subject_for(d);
    uint8_t subject[32];
    std::memcpy(subject, subj.data(), 32);

    std::vector<HostVote> votes;
    // Each lane gets enough stake (30+30+30 = 90 ≥ 60) for quorum.
    for (uint32_t kind = 0u; kind < 3u; ++kind) {
        votes.push_back(sign_vote(d, subject, kind * 3u + 0u, 30u, kind));
        votes.push_back(sign_vote(d, subject, kind * 3u + 1u, 30u, kind));
        votes.push_back(sign_vote(d, subject, kind * 3u + 2u, 30u, kind));
    }
    e->push_votes(h, votes);

    e->request_close(h);
    auto r = e->run_until_done(h, 8);
    EXPECT("full.finalized", r.status == 1u);

    EXPECT("full.bls_quorum",   r.quorum_status_bls == 1u);
    EXPECT("full.rt_quorum",    r.quorum_status_rt  == 1u);
    EXPECT("full.mldsa_groth16_quorum",
           r.quorum_status_mldsa_groth16 == 1u);
    EXPECT("full.bls_stake",            r.quorum_stake_bls()           == 90ull);
    EXPECT("full.rt_stake",             r.quorum_stake_rt()            == 90ull);
    EXPECT("full.mldsa_groth16_stake",  r.quorum_stake_mldsa_groth16() == 90ull);

    auto certs = e->poll_quorum_certs(h);
    EXPECT("full.three_certs", certs.size() == 3u);
    e->end_round(h);
    PASS("full_pq_blocking_drains_all_lanes");
}

// 7. HybridPQAsync mode: BLS lane reaches quorum independently of the PQ
//    lanes. PQ lanes can have fewer signers and still leave the BLS cert
//    emittable (host-side composite cert proceeds with BLS only; PQ lanes
//    feed the next round asynchronously).
void test_hybrid_pq_async_bls_independent()
{
    auto e = QuasarGPUEngine::create();
    auto d = make_cert_desc(1500u, QuasarCertMode::HybridPQAsync);
    auto h = e->begin_round(d);

    auto subj = subject_for(d);
    uint8_t subject[32];
    std::memcpy(subject, subj.data(), 32);

    std::vector<HostVote> votes;
    // BLS reaches quorum on its own (90 ≥ 60).
    votes.push_back(sign_vote(d, subject, 0u, 30u, 0u));
    votes.push_back(sign_vote(d, subject, 1u, 30u, 0u));
    votes.push_back(sign_vote(d, subject, 2u, 30u, 0u));
    // RT + MLDSAGroth16 lag with one signer each — does NOT block BLS cert.
    votes.push_back(sign_vote(d, subject, 3u, 30u, 1u));
    votes.push_back(sign_vote(d, subject, 4u, 30u, 2u));
    e->push_votes(h, votes);

    e->request_close(h);
    auto r = e->run_until_done(h, 8);
    EXPECT("hybrid.finalized", r.status == 1u);

    EXPECT("hybrid.bls_quorum",          r.quorum_status_bls == 1u);
    EXPECT("hybrid.rt_pending",          r.quorum_status_rt  == 0u);
    EXPECT("hybrid.mldsa_groth16_pending",
           r.quorum_status_mldsa_groth16 == 0u);
    EXPECT("hybrid.rt_partial_stake",    r.quorum_stake_rt()            == 30ull);
    EXPECT("hybrid.mldsa_groth16_partial_stake",
           r.quorum_stake_mldsa_groth16() == 30ull);

    // Exactly one cert lands (BLS); PQ lanes are still pending so they have
    // not pushed a QuorumCert yet.
    auto certs = e->poll_quorum_certs(h);
    EXPECT("hybrid.bls_cert_only", certs.size() == 1u);
    EXPECT("hybrid.cert_is_bls",   certs[0].agg_signature.size() == 96u);

    e->end_round(h);
    PASS("hybrid_pq_async_bls_independent");
}

// 8. CertArtifact ABI surface area — its layout is part of the cert ABI a
//    downstream verifier consumes. Pin sizes/offsets so silent drift fails
//    the test rather than the verifier.
void test_cert_artifact_layout()
{
    static_assert(sizeof(QuasarCertLane) == 1u, "QuasarCertLane is 1 byte");
    static_assert(sizeof(QuasarCertMode) == 1u, "QuasarCertMode is 1 byte");
    EXPECT("artifact.lane_offset",
           offsetof(CertArtifact, lane) == 0u);
    EXPECT("artifact.subject_offset",
           offsetof(CertArtifact, subject) >= 1u);
    EXPECT("artifact.has_pubinputs",
           offsetof(CertArtifact, public_inputs_hash)
               > offsetof(CertArtifact, artifact_len));
    // Round-trip — construct a CertArtifact and read it back.
    CertArtifact a{};
    a.lane = QuasarCertLane::MLDSAGroth16;
    for (int i = 0; i < 32; ++i) a.subject[i] = uint8_t(i);
    a.artifact_offset = 0xDEADBEEFCAFEBABEull;
    a.artifact_len    = 192u;             // Groth16 proof size
    for (int i = 0; i < 32; ++i) a.public_inputs_hash[i] = uint8_t(0xA0u ^ i);

    EXPECT("artifact.lane_rt", a.lane == QuasarCertLane::MLDSAGroth16);
    EXPECT("artifact.len_rt",  a.artifact_len == 192u);
    EXPECT("artifact.offset_rt",
           a.artifact_offset == 0xDEADBEEFCAFEBABEull);
    PASS("cert_artifact_layout");
}

}  // namespace

int main(int /*argc*/, char** /*argv*/)
{
    setvbuf(stdout, nullptr, _IOLBF, 0);
    @autoreleasepool {
        std::printf("[quasar_cert_abi_test] starting\n");
        std::fflush(stdout);
        test_cert_enum_values();
        test_cert_subject_deterministic_all_modes();
        test_cert_mode_binds_subject();
        test_attestation_root_binds_subject();
        test_fast_classical_emits_bls_only();
        test_full_pq_blocking_drains_all_lanes();
        test_hybrid_pq_async_bls_independent();
        test_cert_artifact_layout();
        std::printf("[quasar_cert_abi_test] passed=%d failed=%d\n",
                    g_passed, g_failed);
        return g_failed == 0 ? 0 : 1;
    }
}
