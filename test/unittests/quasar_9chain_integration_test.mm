// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file quasar_9chain_integration_test.mm
/// v0.44 — Quasar substrate wires every chain's transition root.
///
/// Coverage:
///   1. Descriptor binds 9 distinct chain roots (P, C, X, Q, Z, A, B, M, F)
///      and produces a non-zero certificate_subject.
///   2. Subject keccak input is exactly H(... || P || C || X || Q || Z ||
///      A || B || M || F || ...) in canonical order — verified byte-for-byte
///      against an inline reference implementation.
///   3. Each of the 9 roots is "load-bearing": flipping a single bit in any
///      root produces a different subject (i.e. cert verification of a
///      tampered descriptor would fail).
///   4. The QuasarGPUEngine round echoes all 9 roots + the cert subject
///      back into QuasarRoundResult so downstream consumers can reconstruct
///      the cert subject from the result alone.
///   5. ServiceId enum exposes the 5 new transition services
///      (PlatformVMTransition .. MPCVMTransition) and ServiceId::Count == 17.

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

using quasar::gpu::QuasarGPUEngine;
using quasar::gpu::QuasarRoundDescriptor;
using quasar::gpu::QuasarRoundResult;
using quasar::gpu::ServiceId;

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

// Build a descriptor where every load-bearing field is distinct and non-zero
// so the subject keccak input is unambiguous. The 9 chain roots are filled
// with distinct repeating-byte patterns (0x10..0x90 stepping by 0x10) so a
// single-bit tamper test is unambiguous.
QuasarRoundDescriptor make_9chain_desc(uint64_t round)
{
    QuasarRoundDescriptor d{};
    d.chain_id          = 0xC0DECAFEFEEDFACEull;
    d.round             = round;
    d.timestamp_ns      = 0u;
    d.deadline_ns       = 0u;
    d.gas_limit         = 30'000'000u;
    d.base_fee          = 1'000u;
    d.wave_tick_budget  = 256u;
    d.mode              = 0u;
    d.epoch             = 7u;
    d.total_stake       = 100u;
    d.validator_count   = 16u;

    // Fill all 9 chain roots + parent state/execution with distinct patterns.
    // P/C/X/Q/Z/A/B/M/F = 0x11/0x22/0x33/0x44/0x55/0x66/0x77/0x88/0x99 fills.
    auto fill = [](uint8_t* p, uint8_t b) { std::memset(p, b, 32); };
    fill(d.pchain_validator_root,   0x11);   // P
    fill(d.parent_block_hash,       0x22);   // C
    fill(d.xchain_execution_root,   0x33);   // X
    fill(d.qchain_ceremony_root,    0x44);   // Q
    fill(d.zchain_vk_root,          0x55);   // Z
    fill(d.achain_state_root,       0x66);   // A
    fill(d.bchain_state_root,       0x77);   // B
    fill(d.mchain_state_root,       0x88);   // M
    fill(d.fchain_state_root,       0x99);   // F
    fill(d.parent_state_root,       0xAA);
    fill(d.parent_execution_root,   0xBB);
    return d;
}

// Inline reference: assemble the exact subject keccak input the goal spec
// requires (canonical order P, C, X, Q, Z, A, B, M, F) and hash it. Compare
// byte-for-byte against compute_certificate_subject. If this matches, the
// 9-chain canonical order is the one the substrate actually computes.
std::array<uint8_t, 32> reference_9chain_subject(const QuasarRoundDescriptor& d)
{
    uint8_t buf[8 + 8 + 8 + 4 + 32 * 11 + 8 + 8];
    size_t off = 0;
    auto put_le64 = [&](uint64_t v) {
        for (size_t k = 0; k < 8; ++k) buf[off + k] = uint8_t((v >> (k * 8u)) & 0xFFu);
        off += 8;
    };
    auto put_le32 = [&](uint32_t v) {
        for (size_t k = 0; k < 4; ++k) buf[off + k] = uint8_t((v >> (k * 8u)) & 0xFFu);
        off += 4;
    };
    auto put_32 = [&](const uint8_t* p) {
        std::memcpy(buf + off, p, 32);
        off += 32;
    };
    put_le64(d.chain_id);
    put_le64(d.epoch);
    put_le64(d.round);
    put_le32(d.mode);
    // Canonical 9-chain order: P, C, X, Q, Z, A, B, M, F.
    put_32(d.pchain_validator_root);     // P
    put_32(d.parent_block_hash);         // C
    put_32(d.xchain_execution_root);     // X
    put_32(d.qchain_ceremony_root);      // Q
    put_32(d.zchain_vk_root);            // Z
    put_32(d.achain_state_root);         // A
    put_32(d.bchain_state_root);         // B
    put_32(d.mchain_state_root);         // M
    put_32(d.fchain_state_root);         // F
    put_32(d.parent_state_root);
    put_32(d.parent_execution_root);
    put_le64(d.gas_limit);
    put_le64(d.base_fee);
    auto h = ethash::keccak256(buf, off);
    std::array<uint8_t, 32> out{};
    std::memcpy(out.data(), h.bytes, 32);
    return out;
}

bool is_zero32(const uint8_t* h)
{
    for (int i = 0; i < 32; ++i) if (h[i] != 0) return false;
    return true;
}

// 1. ServiceId enum exposes the 5 new transition services in the right order
//    after QuorumOut, with Count == 17.
void test_service_id_enum_extended()
{
    EXPECT("svc.platformvm",
           static_cast<uint32_t>(ServiceId::PlatformVMTransition) == 12u);
    EXPECT("svc.xvm",
           static_cast<uint32_t>(ServiceId::XVMTransition) == 13u);
    EXPECT("svc.aivm",
           static_cast<uint32_t>(ServiceId::AIVMTransition) == 14u);
    EXPECT("svc.bridgevm",
           static_cast<uint32_t>(ServiceId::BridgeVMTransition) == 15u);
    EXPECT("svc.mpcvm",
           static_cast<uint32_t>(ServiceId::MPCVMTransition) == 16u);
    EXPECT("svc.count_17",
           static_cast<uint32_t>(ServiceId::Count) == 17u);
    EXPECT("svc.knumservices_matches",
           quasar::gpu::kNumServices == 17u);
    PASS("service_id_enum_extended");
}

// 2. Descriptor + result sizes are stable. Catches any silent layout drift
//    between agent edits.
void test_descriptor_result_sizes()
{
    EXPECT("size.desc_480",  sizeof(QuasarRoundDescriptor) == 480u);
    // 64 atomics + 32 misc + 32*5 existing roots + 32*10 echoes
    // + 3 lanes * 8 words * 4 bytes = 64+32+160+320+96 = 672 bytes.
    EXPECT("size.result_672", sizeof(QuasarRoundResult) == 672u);
    PASS("descriptor_result_sizes");
}

// 3. compute_certificate_subject produces exactly the canonical 9-chain
//    keccak digest. The reference here is the inline reproduction of the
//    spec's hash input.
void test_subject_canonical_order()
{
    auto d = make_9chain_desc(100u);
    auto reference = reference_9chain_subject(d);
    auto actual = quasar::gpu::sig::compute_certificate_subject(
        d.chain_id, d.epoch, d.round, d.mode,
        d.pchain_validator_root,    // P
        d.parent_block_hash,        // C
        d.xchain_execution_root,    // X
        d.qchain_ceremony_root,     // Q
        d.zchain_vk_root,           // Z
        d.achain_state_root,        // A
        d.bchain_state_root,        // B
        d.mchain_state_root,        // M
        d.fchain_state_root,        // F
        d.parent_state_root, d.parent_execution_root,
        d.gas_limit, d.base_fee);

    EXPECT("canon.nonzero", !is_zero32(actual.data()));
    EXPECT("canon.matches_reference",
           std::memcmp(reference.data(), actual.data(), 32) == 0);
    PASS("subject_canonical_order");
}

// 4. Each of the 9 chain roots is load-bearing: flipping any single bit
//    produces a different subject — exactly what cert subject-binding needs
//    to prevent a tampered descriptor from passing verification.
void test_subject_binds_every_chain_root()
{
    auto base = make_9chain_desc(101u);
    auto h0 = quasar::gpu::sig::compute_certificate_subject(
        base.chain_id, base.epoch, base.round, base.mode,
        base.pchain_validator_root, base.parent_block_hash,
        base.xchain_execution_root, base.qchain_ceremony_root,
        base.zchain_vk_root, base.achain_state_root,
        base.bchain_state_root, base.mchain_state_root,
        base.fchain_state_root, base.parent_state_root,
        base.parent_execution_root, base.gas_limit, base.base_fee);

    struct Mutator { const char* name; uint8_t* (*get)(QuasarRoundDescriptor&); };
    const Mutator muts[] = {
        {"P", [](QuasarRoundDescriptor& d) -> uint8_t* { return d.pchain_validator_root; }},
        {"C", [](QuasarRoundDescriptor& d) -> uint8_t* { return d.parent_block_hash; }},
        {"X", [](QuasarRoundDescriptor& d) -> uint8_t* { return d.xchain_execution_root; }},
        {"Q", [](QuasarRoundDescriptor& d) -> uint8_t* { return d.qchain_ceremony_root; }},
        {"Z", [](QuasarRoundDescriptor& d) -> uint8_t* { return d.zchain_vk_root; }},
        {"A", [](QuasarRoundDescriptor& d) -> uint8_t* { return d.achain_state_root; }},
        {"B", [](QuasarRoundDescriptor& d) -> uint8_t* { return d.bchain_state_root; }},
        {"M", [](QuasarRoundDescriptor& d) -> uint8_t* { return d.mchain_state_root; }},
        {"F", [](QuasarRoundDescriptor& d) -> uint8_t* { return d.fchain_state_root; }},
    };

    for (const auto& mut : muts) {
        QuasarRoundDescriptor d = base;
        mut.get(d)[0] ^= 0x01;   // flip lowest bit of byte 0
        auto h = quasar::gpu::sig::compute_certificate_subject(
            d.chain_id, d.epoch, d.round, d.mode,
            d.pchain_validator_root, d.parent_block_hash,
            d.xchain_execution_root, d.qchain_ceremony_root,
            d.zchain_vk_root, d.achain_state_root,
            d.bchain_state_root, d.mchain_state_root,
            d.fchain_state_root, d.parent_state_root,
            d.parent_execution_root, d.gas_limit, d.base_fee);
        if (std::memcmp(h0.data(), h.data(), 32) == 0) {
            std::printf("  FAIL[bind.%s]: subject unchanged after bit flip\n", mut.name);
            std::fflush(stdout);
            ++g_failed;
            return;
        }
    }
    PASS("subject_binds_every_chain_root");
}

// 5. Roots in canonical order matter — swapping two roots produces a
//    different subject. Specifically swap A ↔ B; if the substrate hashed
//    them in some other order the digest could collide.
void test_subject_order_matters()
{
    auto d = make_9chain_desc(102u);
    auto h_orig = quasar::gpu::sig::compute_certificate_subject(
        d.chain_id, d.epoch, d.round, d.mode,
        d.pchain_validator_root, d.parent_block_hash,
        d.xchain_execution_root, d.qchain_ceremony_root,
        d.zchain_vk_root, d.achain_state_root,
        d.bchain_state_root, d.mchain_state_root,
        d.fchain_state_root, d.parent_state_root,
        d.parent_execution_root, d.gas_limit, d.base_fee);

    // Swap A and B contents (not the slots; the function arguments).
    auto h_swapped = quasar::gpu::sig::compute_certificate_subject(
        d.chain_id, d.epoch, d.round, d.mode,
        d.pchain_validator_root, d.parent_block_hash,
        d.xchain_execution_root, d.qchain_ceremony_root,
        d.zchain_vk_root,
        d.bchain_state_root,                // A slot fed B's bytes
        d.achain_state_root,                // B slot fed A's bytes
        d.mchain_state_root, d.fchain_state_root,
        d.parent_state_root, d.parent_execution_root,
        d.gas_limit, d.base_fee);

    EXPECT("order.differs",
           std::memcmp(h_orig.data(), h_swapped.data(), 32) != 0);
    PASS("subject_order_matters");
}

// 6. End-to-end through the engine: begin a 9-chain round, run it to
//    completion (no txs), and verify the engine echoes all 9 roots + the
//    cert subject into the result. This proves the substrate wires the
//    roots through to consumers.
void test_engine_round_echoes_9_roots()
{
    auto e = QuasarGPUEngine::create();
    EXPECT("e2e.create", e != nullptr);

    auto d = make_9chain_desc(200u);
    auto expected_subject = reference_9chain_subject(d);

    auto h = e->begin_round(d);
    EXPECT("e2e.handle", h.valid());
    e->request_close(h);
    auto r = e->run_until_done(h, 8);
    EXPECT("e2e.finalized", r.status == 1u);

    // Subject echo matches the canonical reference computation.
    EXPECT("e2e.subject_echo",
           std::memcmp(r.certificate_subject_echo,
                       expected_subject.data(), 32) == 0);

    // All 9 chain roots are echoed in canonical order.
    EXPECT("e2e.echo_P",
           std::memcmp(r.pchain_root_echo, d.pchain_validator_root, 32) == 0);
    EXPECT("e2e.echo_C",
           std::memcmp(r.cchain_root_echo, d.parent_block_hash, 32) == 0);
    EXPECT("e2e.echo_X",
           std::memcmp(r.xchain_root_echo, d.xchain_execution_root, 32) == 0);
    EXPECT("e2e.echo_Q",
           std::memcmp(r.qchain_root_echo, d.qchain_ceremony_root, 32) == 0);
    EXPECT("e2e.echo_Z",
           std::memcmp(r.zchain_root_echo, d.zchain_vk_root, 32) == 0);
    EXPECT("e2e.echo_A",
           std::memcmp(r.achain_root_echo, d.achain_state_root, 32) == 0);
    EXPECT("e2e.echo_B",
           std::memcmp(r.bchain_root_echo, d.bchain_state_root, 32) == 0);
    EXPECT("e2e.echo_M",
           std::memcmp(r.mchain_root_echo, d.mchain_state_root, 32) == 0);
    EXPECT("e2e.echo_F",
           std::memcmp(r.fchain_root_echo, d.fchain_state_root, 32) == 0);

    e->end_round(h);
    PASS("engine_round_echoes_9_roots");
}

// 7. Tampering the descriptor's certificate_subject would not match the
//    canonical recompute — i.e. a downstream verifier comparing
//    expected_subject vs result.certificate_subject_echo can detect the
//    forgery. We simulate this by computing the canonical subject for a
//    descriptor where one chain root has been flipped, and verifying that
//    the engine's subject_echo (computed from the unflipped descriptor)
//    differs from that tampered re-computation.
void test_tampered_descriptor_detected()
{
    auto e = QuasarGPUEngine::create();
    auto d = make_9chain_desc(201u);

    auto h = e->begin_round(d);
    e->request_close(h);
    auto r = e->run_until_done(h, 8);
    EXPECT("tamper.finalized", r.status == 1u);

    // What an attacker would compute if they flipped B-Chain root.
    QuasarRoundDescriptor d_tampered = d;
    d_tampered.bchain_state_root[5] ^= 0xAAu;
    auto tampered_subject = reference_9chain_subject(d_tampered);

    // Engine-echoed subject (from real, untampered descriptor) must NOT
    // match the tampered re-computation. This is the property a
    // QuasarCert verifier relies on.
    EXPECT("tamper.subject_diverges",
           std::memcmp(r.certificate_subject_echo,
                       tampered_subject.data(), 32) != 0);

    e->end_round(h);
    PASS("tampered_descriptor_detected");
}

}  // namespace

int main(int /*argc*/, char** /*argv*/)
{
    setvbuf(stdout, nullptr, _IOLBF, 0);
    @autoreleasepool {
        std::printf("[quasar_9chain_integration_test] starting\n");
        std::fflush(stdout);
        test_service_id_enum_extended();
        test_descriptor_result_sizes();
        test_subject_canonical_order();
        test_subject_binds_every_chain_root();
        test_subject_order_matters();
        test_engine_round_echoes_9_roots();
        test_tampered_descriptor_detected();
        std::printf("[quasar_9chain_integration_test] passed=%d failed=%d\n",
                    g_passed, g_failed);
        return g_failed == 0 ? 0 : 1;
    }
}
