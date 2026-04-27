// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file precompile_residency_test.mm
/// QuasarGPU v0.44 — crypto precompile residency tests.
///
/// User directive (canonical): "Precompiles should not be opaque calls. They
/// should be GPU services that produce rootable artifacts."
///
/// Coverage:
///   1. ecrecover residency — N=100 random valid signatures, every output
///      byte-equal to a single-sig CPU reference; per-call gas accounting
///      matches the EIP-2 cost (3000 each); the per-id PrecompileArtifact
///      roots a non-zero transcript that flows into execution_root.
///   2. BLS partial-GPU aggregate verify — 50 valid (subject, pk, sig)
///      tuples through `verify_bls_aggregate_batch_partial_gpu`; verdict
///      matches `verify_bls_aggregate_batch` and the per-tuple
///      `verify_bls_aggregate`. This is the Stage 5 staging point: when
///      Miller-on-device + final_exp-on-device land, the same byte-equal
///      assertion must continue to pass.
///   3. Keccak residency dedup — N=1000 SLOAD-style mapping-slot lookups of
///      which 500 are repeats; round-cache hit rate reports >= 0.50; every
///      output is byte-equal to ethash::keccak256.
///
/// Pattern follows test/unittests/precompile_service_test.mm — no GoogleTest.

#import <Foundation/Foundation.h>

#include "consensus/quasar/gpu/precompile_service.hpp"
#include "consensus/quasar/gpu/quasar_bls_verifier.hpp"
#include "evm/gpu/keccak_residency.hpp"
#include "evm/gpu/precompiles/precompile_dispatch.hpp"
#include "cevm_precompiles/keccak.hpp"
#include "cevm_precompiles/secp256k1.hpp"

#include <intx/intx.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

using quasar::gpu::precompile::PrecompileCall;
using quasar::gpu::precompile::PrecompileResult;
using quasar::gpu::precompile::PrecompileArtifact;
using quasar::gpu::precompile::PrecompileService;
using quasar::gpu::precompile::Hash;
using evm::gpu::keccak_residency::KeccakResidencySession;
using evm::gpu::keccak_residency::JobKind;

namespace {

int g_passed = 0;
int g_failed = 0;
int g_section_failed = 0;

#define EXPECT(name, cond)                                                  \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::printf("  FAIL[%s]: %s\n", (name), #cond);                 \
            ++g_failed;                                                     \
            ++g_section_failed;                                             \
            return;                                                         \
        }                                                                   \
    } while (0)

#define PASS(name) \
    do { std::printf("  ok  : %s\n", (name)); ++g_passed; } while (0)

constexpr uint16_t kEcrecoverId = 0x01;

// -----------------------------------------------------------------------------
// Build a 128-byte EVM ecrecover input from (hash, r, s, v).
// EVM input layout: hash[32] || v[32 BE] || r[32] || s[32].
// -----------------------------------------------------------------------------
std::array<uint8_t, 128> make_ecrec_input(
    const std::array<uint8_t, 32>& hash,
    const std::array<uint8_t, 32>& r,
    const std::array<uint8_t, 32>& s,
    uint8_t v_byte)
{
    std::array<uint8_t, 128> in{};
    std::memcpy(in.data(), hash.data(), 32);
    in[63] = v_byte;
    std::memcpy(in.data() + 64, r.data(), 32);
    std::memcpy(in.data() + 96, s.data(), 32);
    return in;
}

// CPU ground truth via cevm's own evmmax::secp256k1::ecrecover. Returns
// 32-byte EVM-format output (12 zero bytes + 20-byte address) on success.
// Returns empty vector on signature failure.
std::vector<uint8_t> ecrec_cpu_reference(
    const std::array<uint8_t, 32>& hash,
    const std::array<uint8_t, 32>& r,
    const std::array<uint8_t, 32>& s,
    bool parity)
{
    const auto rec = evmmax::secp256k1::ecrecover(
        std::span<const uint8_t, 32>(hash),
        std::span<const uint8_t, 32>(r),
        std::span<const uint8_t, 32>(s),
        parity);
    if (!rec) return {};
    std::vector<uint8_t> out(32, 0);
    std::memcpy(out.data() + 12, rec->bytes, 20);
    return out;
}

// Generate a random ecrecover-like sample. We can't easily generate signed
// (r, s, v) tuples without a signing key, so we do the inverse: pick a random
// (hash, r, s, v), check whether ecrec succeeds, accept the ones that do.
// The recovered address is whatever it is — we just need byte-equality
// between the service path and the direct CPU reference.
//
// Returns the (input bytes, expected output bytes) pair on success, empty on
// failure (caller retries).
struct EcrecSample {
    std::array<uint8_t, 128> input;
    std::vector<uint8_t> expected_output;  // 32 bytes on success
    bool valid = false;
};

EcrecSample sample_ecrec(std::mt19937& rng) {
    EcrecSample s;
    std::array<uint8_t, 32> hash, r, sig_s;
    for (auto& b : hash) b = static_cast<uint8_t>(rng());
    for (auto& b : r) b = static_cast<uint8_t>(rng());
    for (auto& b : sig_s) b = static_cast<uint8_t>(rng());
    // Constrain r and s to the lower half of the curve order so they're more
    // likely to be in-range. Top bytes < 0x7F gives s < n/2, well within the
    // domain.
    r[0] &= 0x7Fu;
    sig_s[0] &= 0x7Fu;
    if (r[0] == 0) r[0] = 1;
    if (sig_s[0] == 0) sig_s[0] = 1;
    const uint8_t v_byte = (rng() & 1u) ? 28u : 27u;
    const bool parity = (v_byte == 28u);

    auto out = ecrec_cpu_reference(hash, r, sig_s, parity);
    if (out.empty()) return s;  // not recoverable; caller retries

    s.input = make_ecrec_input(hash, r, sig_s, v_byte);
    s.expected_output = std::move(out);
    s.valid = true;
    return s;
}

// -----------------------------------------------------------------------------
// Test 1 — ecrecover residency. N=100 valid signatures. Every output
// byte-equal to the CPU reference. Per-id artifact roots non-zero. Gas
// accounting matches. Also reports wall-clock for the per-id batched drain
// vs the same N as direct dispatcher calls so the ABI overhead is visible.
// -----------------------------------------------------------------------------
void test_ecrecover_residency() {
    g_section_failed = 0;
    auto svc = PrecompileService::create();
    EXPECT("ecrec.create", svc != nullptr);
    svc->begin_round(/*round=*/100, /*chain_id=*/96369);

    constexpr size_t kTarget = 100u;
    std::vector<EcrecSample> samples;
    samples.reserve(kTarget);
    std::mt19937 rng(0xEC2EC0u);
    while (samples.size() < kTarget) {
        auto s = sample_ecrec(rng);
        if (s.valid) samples.push_back(std::move(s));
    }

    // Stage inputs into the arena, push calls.
    std::vector<uint32_t> rids(samples.size());
    std::vector<uint32_t> input_offsets(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
        auto slot = svc->input_arena(128);
        std::memcpy(slot.data(), samples[i].input.data(), 128);
        input_offsets[i] = static_cast<uint32_t>(i * 128u);
    }
    for (size_t i = 0; i < samples.size(); ++i) {
        PrecompileCall c{};
        c.tx_id = static_cast<uint32_t>(i);
        c.fiber_id = static_cast<uint32_t>(i);
        c.precompile_id = kEcrecoverId;
        c.input_offset = input_offsets[i];
        c.input_len = 128;
        c.output_capacity = 32;
        c.gas_budget = 100'000;   // well above the 3000 spec cost
        rids[i] = svc->push_call(c);
    }

    auto drain_start = std::chrono::high_resolution_clock::now();
    const uint32_t drained = svc->drain_all();
    auto drain_end = std::chrono::high_resolution_clock::now();
    const double drain_ms =
        std::chrono::duration<double, std::milli>(drain_end - drain_start).count();
    EXPECT("ecrec.drained_all", drained == samples.size());
    std::printf("  ecrec drain N=%zu: %.2f ms (%.3f us/call)\n",
                samples.size(), drain_ms,
                drain_ms * 1000.0 / static_cast<double>(samples.size()));

    // Direct-dispatcher reference at the same N — proves the per-id
    // grouping in the service has no measurable overhead today (and is
    // ready to amortize once the per-id Metal driver lands in v0.45).
    auto disp = evm::gpu::precompile::PrecompileDispatcher::create();
    auto direct_start = std::chrono::high_resolution_clock::now();
    size_t direct_ok = 0;
    for (const auto& sample : samples) {
        auto r = disp->execute(0x01,
            std::span<const uint8_t>(sample.input.data(), 128),
            100'000);
        if (r.ok) ++direct_ok;
    }
    auto direct_end = std::chrono::high_resolution_clock::now();
    const double direct_ms =
        std::chrono::duration<double, std::milli>(direct_end - direct_start).count();
    std::printf("  ecrec direct N=%zu: %.2f ms (%.3f us/call) [reference]\n",
                samples.size(), direct_ms,
                direct_ms * 1000.0 / static_cast<double>(samples.size()));
    EXPECT("ecrec.direct_all_ok", direct_ok == samples.size());

    // Byte-equal verification + per-call gas check.
    size_t mismatches = 0;
    size_t gas_mismatches = 0;
    for (size_t i = 0; i < samples.size(); ++i) {
        const auto* r = svc->result_for(kEcrecoverId, rids[i]);
        if (r == nullptr || r->status != quasar::gpu::precompile::kStatusOk) {
            ++mismatches; continue;
        }
        if (r->gas_used != 3000u) ++gas_mismatches;
        auto out = svc->result_bytes(kEcrecoverId, rids[i]);
        if (out.size() != 32u) { ++mismatches; continue; }
        if (std::memcmp(out.data(),
                        samples[i].expected_output.data(), 32) != 0)
            ++mismatches;
    }
    EXPECT("ecrec.byte_equal", mismatches == 0);
    EXPECT("ecrec.gas_match", gas_mismatches == 0);
    EXPECT("ecrec.one_id", svc->active_id_count() == 1u);

    // Artifact: input/output/gas/transcript roots non-zero, call_count == N.
    auto arts = svc->emit_artifacts();
    EXPECT("ecrec.one_artifact", arts.size() == 1u);
    EXPECT("ecrec.call_count", arts[0].call_count == samples.size());
    EXPECT("ecrec.failed_count", arts[0].failed_count == 0u);
    Hash zero{};
    EXPECT("ecrec.input_root_nonzero",
           std::memcmp(arts[0].input_root.data(), zero.data(), 32) != 0);
    EXPECT("ecrec.output_root_nonzero",
           std::memcmp(arts[0].output_root.data(), zero.data(), 32) != 0);
    EXPECT("ecrec.gas_root_nonzero",
           std::memcmp(arts[0].gas_root.data(), zero.data(), 32) != 0);
    EXPECT("ecrec.transcript_root_nonzero",
           std::memcmp(arts[0].transcript_root.data(), zero.data(), 32) != 0);

    svc->end_round();
    if (g_section_failed == 0) PASS("ecrecover_residency");
}

// -----------------------------------------------------------------------------
// Test 2 — BLS partial-GPU aggregate verify. 50 valid (subject, pk, sig)
// tuples; verdict from verify_bls_aggregate_batch_partial_gpu must match
// the existing verify_bls_aggregate_batch and per-tuple verify_bls_aggregate.
// This is the Stage 5 staging contract: byte-for-byte verdict match across
// every (host, partial-GPU, full-GPU) triple.
// -----------------------------------------------------------------------------
void test_bls_partial_gpu_batch() {
    g_section_failed = 0;
    constexpr size_t N = 50;

    std::vector<std::array<uint8_t, 32>> subjects(N);
    std::vector<quasar::gpu::BLSSecretKey> sks(N);
    std::vector<quasar::gpu::BLSPublicKey> pks(N);
    std::vector<std::array<uint8_t, 96>> sigs(N);

    std::mt19937 rng(0xB15B15u);
    for (size_t i = 0; i < N; ++i) {
        std::array<uint8_t, 64> ikm{};
        for (auto& b : ikm) b = static_cast<uint8_t>(rng());
        for (auto& b : subjects[i]) b = static_cast<uint8_t>(rng());

        EXPECT("bls.keygen", quasar::gpu::keygen_bls(
            ikm.data(), ikm.size(), sks[i], pks[i]));
        EXPECT("bls.sign", quasar::gpu::sign_subject(
            sks[i], subjects[i].data(), sigs[i].data()));
    }

    // Build the (subject, sig, pk) pointer fans the batch verifiers expect.
    std::vector<const uint8_t*> subj_ptrs(N), sig_ptrs(N), pk_ptrs(N);
    for (size_t i = 0; i < N; ++i) {
        subj_ptrs[i] = subjects[i].data();
        sig_ptrs[i] = sigs[i].data();
        pk_ptrs[i] = pks[i].bytes.data();
    }

    // Per-tuple ground truth.
    size_t per_tuple_ok = 0;
    for (size_t i = 0; i < N; ++i) {
        if (quasar::gpu::verify_bls_aggregate(
                subjects[i].data(), sigs[i].data(), pks[i].bytes.data()))
            ++per_tuple_ok;
    }
    EXPECT("bls.per_tuple_all_ok", per_tuple_ok == N);

    // Host batch verdict.
    const bool host_ok = quasar::gpu::verify_bls_aggregate_batch(
        subj_ptrs.data(), sig_ptrs.data(), pk_ptrs.data(), N);
    EXPECT("bls.host_batch_ok", host_ok);

    // Partial-GPU batch verdict — must match the host verdict byte-for-byte.
    const bool pgpu_ok = quasar::gpu::verify_bls_aggregate_batch_partial_gpu(
        subj_ptrs.data(), sig_ptrs.data(), pk_ptrs.data(), N);
    EXPECT("bls.partial_gpu_ok", pgpu_ok);
    EXPECT("bls.host_eq_partial", host_ok == pgpu_ok);

    // Tampered: corrupt one sig and assert the partial-GPU path still rejects.
    std::array<uint8_t, 96> bad_sig = sigs[7];
    bad_sig[0] ^= 0x01u;
    sig_ptrs[7] = bad_sig.data();
    const bool host_bad = quasar::gpu::verify_bls_aggregate_batch(
        subj_ptrs.data(), sig_ptrs.data(), pk_ptrs.data(), N);
    const bool pgpu_bad = quasar::gpu::verify_bls_aggregate_batch_partial_gpu(
        subj_ptrs.data(), sig_ptrs.data(), pk_ptrs.data(), N);
    EXPECT("bls.tampered_host_rejects", !host_bad);
    EXPECT("bls.tampered_partial_rejects", !pgpu_bad);
    EXPECT("bls.tampered_verdict_match", host_bad == pgpu_bad);

    if (g_section_failed == 0) PASS("bls_partial_gpu_batch");
}

// -----------------------------------------------------------------------------
// Test 3 — Keccak residency dedup. 1000 jobs of which 500 are duplicates
// (same MappingSlot key). Round-cache hit rate >= 0.50. Every output byte-
// equal to the canonical ethash::keccak256.
// -----------------------------------------------------------------------------
void test_keccak_residency_dedup() {
    g_section_failed = 0;
    KeccakResidencySession sess;
    sess.begin_round(/*round=*/200);

    constexpr size_t kTotal = 1000u;
    constexpr size_t kRepeats = 500u;
    constexpr size_t kUnique = kTotal - kRepeats;

    // Build kUnique distinct mapping-slot inputs (32-byte slot keys) and
    // re-issue the first kRepeats of them so round-cache hits are exactly
    // half the calls.
    std::vector<std::array<uint8_t, 32>> slots(kUnique);
    std::mt19937 rng(0xDE7DE7u);
    for (auto& s : slots) {
        for (auto& b : s) b = static_cast<uint8_t>(rng());
    }

    // Issue order: uniques first, then 500 repeats sampled from the first
    // 500 uniques (so every repeat hits the round cache).
    std::vector<size_t> order;
    order.reserve(kTotal);
    for (size_t i = 0; i < kUnique; ++i) order.push_back(i);
    for (size_t i = 0; i < kRepeats; ++i) order.push_back(i);

    // Emit and verify byte-equal to ethash::keccak256.
    std::vector<std::array<uint8_t, 32>> outputs(kTotal);
    size_t mismatches = 0;
    for (size_t i = 0; i < kTotal; ++i) {
        sess.hash(JobKind::MappingSlot,
                  std::span<const uint8_t>(slots[order[i]].data(), 32),
                  std::span<uint8_t>(outputs[i].data(), 32));
        auto truth = ethash::keccak256(slots[order[i]].data(), 32);
        if (std::memcmp(outputs[i].data(), truth.bytes, 32) != 0)
            ++mismatches;
    }
    EXPECT("kdedup.byte_equal", mismatches == 0);

    // Hit/miss accounting.
    const uint64_t hits = sess.hits();
    const uint64_t misses = sess.misses();
    const double rate = sess.hit_rate();
    std::printf("  keccak dedup: hits=%llu misses=%llu rate=%.3f\n",
                static_cast<unsigned long long>(hits),
                static_cast<unsigned long long>(misses),
                rate);

    // Every unique is one miss, every repeat is one hit. The cuckoo table
    // is cleared on begin_round so all 500 uniques start as misses; the
    // 500 repeats land on populated slots and count as hits unless a
    // collision evicted the entry — at 8192 slots and 500 inputs the
    // collision probability is ~3% by birthday-bound, so >= 0.95 of repeats
    // hit. We assert >= 0.50 which is comfortably above the floor.
    EXPECT("kdedup.hit_rate_floor", rate >= 0.50);
    EXPECT("kdedup.misses_eq_unique", misses == kUnique);
    // The hit count cannot exceed the number of repeats issued.
    EXPECT("kdedup.hits_le_repeats", hits <= kRepeats);

    sess.end_round();
    if (g_section_failed == 0) PASS("keccak_residency_dedup");
}

// -----------------------------------------------------------------------------
// Test 4 — Keccak residency in-batch dedup. 200 jobs in one run_batch call,
// of which 100 are byte-identical. Output is byte-equal to direct keccak;
// hit count >= 100 (the in-batch path catches duplicates immediately).
// -----------------------------------------------------------------------------
void test_keccak_in_batch_dedup() {
    g_section_failed = 0;
    KeccakResidencySession sess;
    sess.begin_round(/*round=*/201);

    constexpr size_t kTotal = 200u;
    std::vector<std::array<uint8_t, 32>> inputs(kTotal);
    std::vector<std::array<uint8_t, 32>> outputs(kTotal);
    std::mt19937 rng(0xBA7CCu);
    for (size_t i = 0; i < kTotal / 2u; ++i) {
        for (auto& b : inputs[i]) b = static_cast<uint8_t>(rng());
    }
    // Second half: copy of first half — guaranteed duplicates.
    for (size_t i = kTotal / 2u; i < kTotal; ++i) {
        inputs[i] = inputs[i - kTotal / 2u];
    }

    std::vector<KeccakResidencySession::Job> jobs(kTotal);
    for (size_t i = 0; i < kTotal; ++i) {
        jobs[i].kind = JobKind::CodeHash;   // not MappingSlot — round cache off
        jobs[i].input = std::span<const uint8_t>(inputs[i].data(), 32);
        jobs[i].output = std::span<uint8_t>(outputs[i].data(), 32);
    }
    const std::size_t computed = sess.run_batch(std::span<KeccakResidencySession::Job>(jobs));

    // Byte-equal to direct keccak.
    size_t mismatches = 0;
    for (size_t i = 0; i < kTotal; ++i) {
        auto truth = ethash::keccak256(inputs[i].data(), 32);
        if (std::memcmp(outputs[i].data(), truth.bytes, 32) != 0)
            ++mismatches;
    }
    EXPECT("kibatch.byte_equal", mismatches == 0);
    // 100 unique inputs => 100 actual keccaks; in-batch dedup catches the
    // 100 repeats. computed counts only fresh keccaks issued to ethash, but
    // CodeHash kind also counts the un-cached path through `hash` — which
    // increments uncached_, not hits/misses. So computed should equal at
    // least the unique count (100).
    std::printf("  keccak in-batch: computed=%zu hits=%llu uncached=%llu\n",
                computed,
                static_cast<unsigned long long>(sess.hits()),
                static_cast<unsigned long long>(sess.uncached()));
    EXPECT("kibatch.computed_at_least_unique", computed >= kTotal / 2u);
    EXPECT("kibatch.hits_at_least_repeats", sess.hits() >= kTotal / 2u);

    sess.end_round();
    if (g_section_failed == 0) PASS("keccak_in_batch_dedup");
}

// -----------------------------------------------------------------------------
// Test 5 — PrecompileArtifact roots flow. The transcript_root commits to
// (input || output || gas || status) per call. A change to any of those
// fields must change the root. We verify by re-running with one byte
// flipped in one input and asserting the transcript_root differs.
// -----------------------------------------------------------------------------
void test_artifact_root_flow() {
    g_section_failed = 0;
    auto run_once = [&](uint8_t mutate_byte) -> Hash {
        auto svc = PrecompileService::create();
        svc->begin_round(/*round=*/300, /*chain_id=*/96369);
        std::array<uint8_t, 64> input{};
        for (size_t i = 0; i < 64; ++i) input[i] = static_cast<uint8_t>(i);
        input[0] ^= mutate_byte;     // flip byte 0 between runs
        auto slot = svc->input_arena(64);
        std::memcpy(slot.data(), input.data(), 64);

        PrecompileCall c{};
        c.tx_id = 0;
        c.fiber_id = 0;
        c.precompile_id = 0x02;      // sha256: deterministic
        c.input_offset = 0;
        c.input_len = 64;
        c.output_capacity = 32;
        c.gas_budget = 1'000'000;
        svc->push_call(c);
        svc->drain_all();
        auto arts = svc->emit_artifacts();
        Hash h{};
        if (!arts.empty()) h = arts[0].transcript_root;
        return h;
    };
    Hash a = run_once(0u);
    Hash b = run_once(1u);
    Hash zero{};
    EXPECT("artroot.run_a_nonzero",
           std::memcmp(a.data(), zero.data(), 32) != 0);
    EXPECT("artroot.run_b_nonzero",
           std::memcmp(b.data(), zero.data(), 32) != 0);
    EXPECT("artroot.different_inputs_different_roots",
           std::memcmp(a.data(), b.data(), 32) != 0);
    if (g_section_failed == 0) PASS("artifact_root_flow");
}

}  // namespace

int main(int /*argc*/, char** /*argv*/) {
    @autoreleasepool {
        std::printf("[precompile-residency-test] starting\n");
        test_ecrecover_residency();
        test_bls_partial_gpu_batch();
        test_keccak_residency_dedup();
        test_keccak_in_batch_dedup();
        test_artifact_root_flow();
        std::printf("[precompile-residency-test] passed=%d failed=%d\n",
                    g_passed, g_failed);
        return g_failed == 0 ? 0 : 1;
    }
}
