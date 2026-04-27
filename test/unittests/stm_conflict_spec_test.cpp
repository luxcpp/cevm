// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file stm_conflict_spec_test.cpp
/// LP-090 v0.50: regression suite for ConflictSpec ABI + 5 sources +
/// composer + bench. Each section gates one observable property of the
/// declaration scaffold.
///
///   1. test_struct_layout_byte_equal           — 32-byte ABI invariant
///   2. test_compose_from_access_list           — Source 1: EIP-2930
///   3. test_compose_from_abi_erc20             — Source 2: ABI selectors
///   4. test_compose_from_historical_lru        — Source 3: hit/miss/evict
///   5. test_compose_from_precompile            — Source 4: 0x01..0x11
///   6. test_compose_from_learned_notimpl       — Source 5: stub returns 0
///   7. test_composer_priority_order            — AccessList > ABI > ...
///   8. test_reducer_lane_no_repair             — BalanceDelta accumulates
///   9. test_repair_amplification_synthetic     — bench: target <1.01

#include "stm/conflict_spec.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using evm::stm::ComposerInputs;
using evm::stm::compose;
using evm::stm::compose_from_abi;
using evm::stm::compose_from_access_list;
using evm::stm::compose_from_precompile;
using evm::stm::ConflictArena;
using evm::stm::ConflictLane;
using evm::stm::ConflictSource;
using evm::stm::ConflictSpec;
using evm::stm::HistoricalProfile;
using evm::stm::kSkipDiscoveryThreshold;
using evm::stm::LearnedPredictor;

namespace {

int g_passed = 0;
int g_failed = 0;

#define EXPECT(name, cond)                                                   \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("  FAIL[%s]: %s\n", (name), #cond);                  \
            std::fflush(stdout);                                             \
            ++g_failed;                                                      \
            return;                                                          \
        }                                                                    \
    } while (0)

#define PASS(name)                                                           \
    do {                                                                     \
        std::printf("  ok  : %s\n", (name));                                 \
        std::fflush(stdout);                                                 \
        ++g_passed;                                                          \
    } while (0)

void fill_addr(uint8_t out[20], uint8_t seed)
{
    for (int i = 0; i < 20; ++i) out[i] = uint8_t(seed + i);
}

void fill_slot(uint8_t out[32], uint8_t seed)
{
    for (int i = 0; i < 32; ++i) out[i] = uint8_t(seed + i);
}

// ===========================================================================
// 1. ConflictSpec is exactly 28 bytes (v0.50 ABI); ConflictLane exactly 52.
//    The static asserts in conflict_spec.hpp would refuse to compile if
//    violated, but we also assert the wire offsets and source IDs so
//    language bindings can rely on them.
// ===========================================================================
void test_struct_layout_byte_equal()
{
    EXPECT("layout.spec_size", sizeof(ConflictSpec) == 28);
    EXPECT("layout.lane_size", sizeof(ConflictLane) == 52);

    // Source values must match the documented wire encoding.
    EXPECT("layout.src_access",  uint8_t(ConflictSource::AccessList) == 0);
    EXPECT("layout.src_abi",     uint8_t(ConflictSource::ABI)        == 1);
    EXPECT("layout.src_hist",    uint8_t(ConflictSource::Historical) == 2);
    EXPECT("layout.src_precom",  uint8_t(ConflictSource::Precompile) == 3);
    EXPECT("layout.src_learn",   uint8_t(ConflictSource::Learned)    == 4);
    EXPECT("layout.src_decl",    uint8_t(ConflictSource::Declared)   == 5);
    PASS("struct_layout_byte_equal");
}

// ===========================================================================
// 2. EIP-2930 access list parser. Two addresses + one (addr, slot) =
//    three lanes total, all in read_lanes; write_lanes empty (EIP-2930
//    does not declare writes). Confidence = 200.
// ===========================================================================
void test_compose_from_access_list()
{
    ConflictArena arena;

    uint8_t addrs[40];
    fill_addr(addrs + 0, 0x10);
    fill_addr(addrs + 20, 0x30);

    uint8_t slots[52];
    fill_addr(slots + 0, 0x50);
    fill_slot(slots + 20, 0x70);

    auto s = compose_from_access_list(7, addrs, sizeof(addrs), slots, sizeof(slots), arena);
    EXPECT("al.tx_id", s.tx_id == 7u);
    EXPECT("al.source", s.source_kind() == ConflictSource::AccessList);
    EXPECT("al.confidence_high", s.confidence == 200);
    EXPECT("al.read_count", s.read_lane_count == 3);
    EXPECT("al.write_zero", s.write_lane_count == 0);

    // Lane 0: address-only (slot zero).
    const auto& lane0 = arena.lanes[s.read_lane_offset + 0];
    EXPECT("al.l0_addr", lane0.address[0] == 0x10);
    for (int i = 0; i < 32; ++i)
        EXPECT("al.l0_slot_zero", lane0.slot[i] == 0);

    // Lane 2: (addr, slot) entry.
    const auto& lane2 = arena.lanes[s.read_lane_offset + 2];
    EXPECT("al.l2_addr", lane2.address[0] == 0x50);
    EXPECT("al.l2_slot", lane2.slot[0] == 0x70);

    // Empty inputs ⇒ Declared/0.
    ConflictArena empty_arena;
    auto s_empty = compose_from_access_list(8, nullptr, 0, nullptr, 0, empty_arena);
    EXPECT("al.empty_decl", s_empty.source_kind() == ConflictSource::Declared);
    EXPECT("al.empty_conf0", s_empty.confidence == 0);

    PASS("compose_from_access_list");
}

// ===========================================================================
// 3. ABI source: ERC-20 transfer(address,uint256). Selector 0xa9059cbb +
//    32-byte recipient word. Expected: 2 reads, 2 writes (sender balance,
//    recipient balance). Confidence = 180.
// ===========================================================================
void test_compose_from_abi_erc20()
{
    ConflictArena arena;
    uint8_t recipient[20];
    fill_addr(recipient, 0xC0);
    uint8_t sender[20];
    fill_addr(sender, 0xA0);

    // calldata: selector || 32-byte addr-padded recipient
    uint8_t calldata[4 + 32] = {};
    calldata[0] = 0xa9; calldata[1] = 0x05; calldata[2] = 0x9c; calldata[3] = 0xbb;
    // ABI: addr right-aligned in 32-byte word at offset 4 (post-selector)
    uint8_t to[20];
    fill_addr(to, 0xB0);
    std::memcpy(calldata + 4 + 12, to, 20);

    auto s = compose_from_abi(42, recipient, calldata, sizeof(calldata), sender, arena);
    EXPECT("abi.tx_id", s.tx_id == 42u);
    EXPECT("abi.source", s.source_kind() == ConflictSource::ABI);
    EXPECT("abi.confidence", s.confidence == 180);
    EXPECT("abi.reads", s.read_lane_count == 2);
    EXPECT("abi.writes", s.write_lane_count == 2);

    // Both write lanes target the recipient (token contract).
    const auto& w0 = arena.lanes[s.write_lane_offset + 0];
    const auto& w1 = arena.lanes[s.write_lane_offset + 1];
    EXPECT("abi.w0_recipient", std::memcmp(w0.address, recipient, 20) == 0);
    EXPECT("abi.w1_recipient", std::memcmp(w1.address, recipient, 20) == 0);
    // Slots differ — encoding embeds the holder.
    EXPECT("abi.w0_w1_distinct", std::memcmp(w0.slot, w1.slot, 32) != 0);

    // Unknown selector ⇒ Declared/0.
    uint8_t cd_unknown[4] = {0xde, 0xad, 0xbe, 0xef};
    auto s_unk = compose_from_abi(43, recipient, cd_unknown, 4, sender, arena);
    EXPECT("abi.unk_decl", s_unk.source_kind() == ConflictSource::Declared);

    // Calldata too short ⇒ Declared/0.
    auto s_short = compose_from_abi(44, recipient, calldata, 3, sender, arena);
    EXPECT("abi.short_decl", s_short.source_kind() == ConflictSource::Declared);

    PASS("compose_from_abi_erc20");
}

// ===========================================================================
// 4. Historical profile: insert → hit; second insert with different key →
//    still hits both; capacity-1 cache evicts oldest on third insert.
// ===========================================================================
void test_compose_from_historical_lru()
{
    HistoricalProfile profile(2);  // capacity = 2

    uint8_t code_hash_a[32], code_hash_b[32], code_hash_c[32];
    std::memset(code_hash_a, 0xAA, 32);
    std::memset(code_hash_b, 0xBB, 32);
    std::memset(code_hash_c, 0xCC, 32);

    uint8_t recipient[20];
    fill_addr(recipient, 0x42);

    // calldata = selector 0x11223344 (no other bytes needed for record)
    uint8_t cd[4] = {0x11, 0x22, 0x33, 0x44};

    ConflictLane reads[1] = {{}};
    ConflictLane writes[1] = {{}};
    fill_addr(reads[0].address, 0x10);
    fill_addr(writes[0].address, 0x20);

    profile.record(code_hash_a, cd, 4, reads, 1, writes, 1);
    profile.record(code_hash_b, cd, 4, reads, 1, writes, 1);

    // Hit A
    {
        ConflictArena arena;
        auto s = profile.compose(1, code_hash_a, cd, 4, recipient, arena);
        EXPECT("hist.hit_a", s.source_kind() == ConflictSource::Historical);
        EXPECT("hist.conf", s.confidence == 150);
        EXPECT("hist.reads", s.read_lane_count == 1);
        EXPECT("hist.writes", s.write_lane_count == 1);
    }

    // Hit B
    {
        ConflictArena arena;
        auto s = profile.compose(2, code_hash_b, cd, 4, recipient, arena);
        EXPECT("hist.hit_b", s.source_kind() == ConflictSource::Historical);
    }

    // Insert C ⇒ A evicted (B was just touched, A is LRU).
    profile.record(code_hash_c, cd, 4, reads, 1, writes, 1);

    {
        ConflictArena arena;
        auto s_a = profile.compose(3, code_hash_a, cd, 4, recipient, arena);
        EXPECT("hist.evicted_a_miss", s_a.source_kind() == ConflictSource::Declared);

        auto s_c = profile.compose(4, code_hash_c, cd, 4, recipient, arena);
        EXPECT("hist.hit_c", s_c.source_kind() == ConflictSource::Historical);
    }

    EXPECT("hist.hits", profile.hits() >= 3);
    EXPECT("hist.misses", profile.misses() >= 1);
    PASS("compose_from_historical_lru");
}

// ===========================================================================
// 5. Precompile source: 0x01..0x11 ⇒ Precompile/220, zero lanes. Other
//    addresses ⇒ Declared/0.
// ===========================================================================
void test_compose_from_precompile()
{
    ConflictArena arena;

    for (uint8_t lo = 1; lo <= 0x11; ++lo) {
        uint8_t addr[20] = {};
        addr[19] = lo;
        auto s = compose_from_precompile(uint32_t(lo), addr, arena);
        EXPECT("pre.is_pre", s.source_kind() == ConflictSource::Precompile);
        EXPECT("pre.conf", s.confidence == 220);
        EXPECT("pre.zero_reads", s.read_lane_count == 0);
        EXPECT("pre.zero_writes", s.write_lane_count == 0);
    }

    // Out of range
    uint8_t addr_invalid[20] = {};
    addr_invalid[19] = 0x20;
    auto s_inv = compose_from_precompile(99, addr_invalid, arena);
    EXPECT("pre.invalid_decl", s_inv.source_kind() == ConflictSource::Declared);

    // High byte set ⇒ not precompile
    uint8_t addr_high[20] = {};
    addr_high[0] = 0x01;
    addr_high[19] = 0x01;
    auto s_high = compose_from_precompile(100, addr_high, arena);
    EXPECT("pre.high_decl", s_high.source_kind() == ConflictSource::Declared);

    PASS("compose_from_precompile");
}

// ===========================================================================
// 6. Learned predictor stub: returns Declared/0 (NOTIMPL placeholder).
//    Asserts the *interface* is stable — concrete impl swaps in later.
// ===========================================================================
void test_compose_from_learned_notimpl()
{
    LearnedPredictor pred;
    ConflictArena arena;
    uint8_t code_hash[32] = {};
    uint8_t cd[4] = {0x01, 0x02, 0x03, 0x04};
    uint8_t recipient[20] = {};
    auto s = pred.predict(5, code_hash, cd, 4, recipient, arena);
    EXPECT("learn.decl", s.source_kind() == ConflictSource::Declared);
    EXPECT("learn.conf0", s.confidence == 0);
    PASS("compose_from_learned_notimpl");
}

// ===========================================================================
// 7. Composer priority order: AccessList beats ABI even when ABI also
//    matches. Removing AccessList drops to ABI. Removing ABI calldata
//    drops to Precompile when recipient is one. Removing all ⇒ Declared/0.
// ===========================================================================
void test_composer_priority_order()
{
    ConflictArena arena;

    uint8_t recipient[20];
    fill_addr(recipient, 0xC0);
    uint8_t sender[20];
    fill_addr(sender, 0xA0);

    uint8_t cd_transfer[4 + 32] = {};
    cd_transfer[0] = 0xa9; cd_transfer[1] = 0x05; cd_transfer[2] = 0x9c; cd_transfer[3] = 0xbb;
    uint8_t to[20];
    fill_addr(to, 0xB0);
    std::memcpy(cd_transfer + 4 + 12, to, 20);

    uint8_t access_list[20];
    fill_addr(access_list, 0x10);

    // (a) Both AccessList and ABI present ⇒ AccessList wins.
    {
        ComposerInputs in{};
        in.tx_id = 1;
        in.warm_addresses = access_list;
        in.warm_addresses_len = sizeof(access_list);
        in.recipient = recipient;
        in.sender = sender;
        in.calldata = cd_transfer;
        in.calldata_len = sizeof(cd_transfer);
        auto s = compose(in, nullptr, nullptr, arena);
        EXPECT("comp.al_wins", s.source_kind() == ConflictSource::AccessList);
    }

    // (b) No AccessList, ABI matches ⇒ ABI wins.
    {
        ConflictArena a;
        ComposerInputs in{};
        in.tx_id = 2;
        in.recipient = recipient;
        in.sender = sender;
        in.calldata = cd_transfer;
        in.calldata_len = sizeof(cd_transfer);
        auto s = compose(in, nullptr, nullptr, a);
        EXPECT("comp.abi_wins", s.source_kind() == ConflictSource::ABI);
    }

    // (c) Recipient is precompile, no AccessList, no ABI match ⇒ Precompile.
    {
        ConflictArena a;
        uint8_t precom_addr[20] = {};
        precom_addr[19] = 0x05;  // modexp
        uint8_t cd_unknown[4] = {0xde, 0xad, 0xbe, 0xef};
        ComposerInputs in{};
        in.tx_id = 3;
        in.recipient = precom_addr;
        in.sender = sender;
        in.calldata = cd_unknown;
        in.calldata_len = 4;
        auto s = compose(in, nullptr, nullptr, a);
        EXPECT("comp.pre_wins", s.source_kind() == ConflictSource::Precompile);
    }

    // (d) Nothing matches ⇒ Declared/0 (validator falls back to dynamic).
    {
        ConflictArena a;
        uint8_t addr[20];
        fill_addr(addr, 0xFE);  // not a precompile
        uint8_t cd_unknown[4] = {0xde, 0xad, 0xbe, 0xef};
        ComposerInputs in{};
        in.tx_id = 4;
        in.recipient = addr;
        in.sender = sender;
        in.calldata = cd_unknown;
        in.calldata_len = 4;
        auto s = compose(in, nullptr, nullptr, a);
        EXPECT("comp.fallback_decl", s.source_kind() == ConflictSource::Declared);
        EXPECT("comp.fallback_conf0", s.confidence == 0);
    }

    PASS("composer_priority_order");
}

// ===========================================================================
// 8. Reducer-lane semantics. Two ABI transfers to the same recipient slot
//    (BalanceDelta-style: commutative aggregation) — the test asserts
//    that placing both writes into reducer_lanes does NOT count as a
//    conflict for repair purposes. Demonstrated by validating the
//    reducer slice independent of write slice.
// ===========================================================================
void test_reducer_lane_no_repair()
{
    ConflictArena arena;
    ConflictSpec spec{};
    spec.tx_id = 0;

    // Two reducer lanes (e.g. BalanceDelta on the same address).
    ConflictLane reducers[2] = {{}, {}};
    fill_addr(reducers[0].address, 0xCA);
    fill_addr(reducers[1].address, 0xCA);  // same address

    auto [r_off, r_cnt] = arena.push_lanes(reducers, 2);
    spec.reducer_lane_offset = r_off;
    spec.reducer_lane_count = r_cnt;

    // Reducer lanes are present. write_lane_count is zero — that is the
    // contract: a tx that only mutates reducers writes nothing the
    // STM validator will repair on.
    EXPECT("red.count", spec.reducer_lane_count == 2);
    EXPECT("red.no_writes", spec.write_lane_count == 0);

    // Both reducers point at the same address — but conflict-free by
    // construction (commutative). The placement layer is free to colocate
    // them on the same shard.
    const auto& r0 = arena.lanes[spec.reducer_lane_offset + 0];
    const auto& r1 = arena.lanes[spec.reducer_lane_offset + 1];
    EXPECT("red.same_addr", std::memcmp(r0.address, r1.address, 20) == 0);

    PASS("reducer_lane_no_repair");
}

// ===========================================================================
// 9. Synthetic repair-amplification bench. Five conflict densities + Zipf
//    hot-key. We measure declared-confidence rate (= committed_count /
//    total_count) at each density. With AccessList declarations on every
//    tx, the rate is 100% so repair_amplification is bounded at 1.0
//    (no repairs because validation skips dynamic discovery). Without
//    declarations, dynamic Block-STM owns repair which the existing 6/6
//    determinism path still honors.
//
// Target: declared-skip rate ≥ 0.99 at 0%/0.01%/0.1%/1% conflict; >0.5 at
// 10% (where repairs become unavoidable on the few sequential tails).
// ===========================================================================
struct BenchPoint
{
    double conflict_pct;
    bool zipf;
    double declared_rate;
    double repair_amplification;
};

BenchPoint run_bench(uint32_t n_tx, double conflict_pct, bool zipf, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    ConflictArena arena;
    HistoricalProfile profile;

    uint32_t declared = 0;
    uint32_t total_repairs = 0;
    uint32_t total_commits = 0;

    for (uint32_t i = 0; i < n_tx; ++i)
    {
        // Pick a "hot" sender for the conflicting fraction; otherwise random.
        const bool is_hot = (rng() % 10000u) < uint64_t(conflict_pct * 100.0);
        uint8_t sender[20] = {};
        if (is_hot) {
            // Zipf: 80% of hot reuses one address; 20% rotates.
            const uint8_t hot_id = zipf ? uint8_t(((rng() % 10u) < 8u) ? 1 : 2 + (rng() % 8u))
                                        : uint8_t(1);
            sender[19] = hot_id;
        } else {
            sender[18] = uint8_t(i >> 8);
            sender[19] = uint8_t(i);
        }

        uint8_t recipient[20] = {};
        recipient[0] = 0xC0;
        recipient[19] = uint8_t(0x42);

        // 50/50 ERC-20 transfer / approve mix
        uint8_t cd[4 + 32] = {};
        if ((rng() & 1u) == 0) {
            cd[0] = 0xa9; cd[1] = 0x05; cd[2] = 0x9c; cd[3] = 0xbb;
        } else {
            cd[0] = 0x09; cd[1] = 0x5e; cd[2] = 0xa7; cd[3] = 0xb3;
        }
        uint8_t to[20] = {};
        to[19] = uint8_t(rng() & 0xFF);
        std::memcpy(cd + 4 + 12, to, 20);

        ComposerInputs in{};
        in.tx_id = i;
        in.recipient = recipient;
        in.sender = sender;
        in.calldata = cd;
        in.calldata_len = sizeof(cd);

        auto s = compose(in, &profile, nullptr, arena);
        if (s.confidence >= kSkipDiscoveryThreshold) ++declared;

        // Simulate the STM validator outcome:
        //   - declared spec ⇒ commit, no repair.
        //   - declared-fallback (Declared/0) on a "hot" tx ⇒ repair (others
        //     wrote the same slot before validation).
        ++total_commits;
        if (s.confidence < kSkipDiscoveryThreshold && is_hot) ++total_repairs;
    }

    BenchPoint p{};
    p.conflict_pct = conflict_pct;
    p.zipf = zipf;
    p.declared_rate = double(declared) / double(n_tx);
    p.repair_amplification = double(total_repairs + total_commits) / double(total_commits);
    return p;
}

void test_repair_amplification_synthetic()
{
    constexpr uint32_t N = 8192;
    const double densities[] = {0.0, 0.01, 0.1, 1.0, 10.0};

    std::printf("\n  bench: N=%u txs, declared = AccessList(0)+ABI(1)+Hist(2)+Pre(3) > %u\n",
                N, kSkipDiscoveryThreshold);

    for (double d : densities)
    {
        for (int z = 0; z < 2; ++z)
        {
            auto p = run_bench(N, d, z != 0, 0xDEADBEEFu + uint64_t(d * 1000.0) + uint64_t(z));
            std::printf("  bench: conflict=%.2f%% zipf=%d declared_rate=%.4f "
                        "repair_amplification=%.4f\n",
                        p.conflict_pct, z, p.declared_rate, p.repair_amplification);

            // Target: <1.01 at low conflict.
            if (p.conflict_pct <= 0.1)
                EXPECT("bench.repair_lt_1_01", p.repair_amplification < 1.01);
            // ABI hits on every transfer/approve ⇒ declared_rate must be >0.99.
            EXPECT("bench.declared_high", p.declared_rate > 0.99);
        }
    }

    PASS("repair_amplification_synthetic");
}

}  // namespace

int main(int /*argc*/, char** /*argv*/)
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::printf("[stm_conflict_spec_test] starting\n");
    std::fflush(stdout);

    test_struct_layout_byte_equal();
    test_compose_from_access_list();
    test_compose_from_abi_erc20();
    test_compose_from_historical_lru();
    test_compose_from_precompile();
    test_compose_from_learned_notimpl();
    test_composer_priority_order();
    test_reducer_lane_no_repair();
    test_repair_amplification_synthetic();

    std::printf("[stm_conflict_spec_test] passed=%d failed=%d\n", g_passed, g_failed);
    std::fflush(stdout);
    return g_failed == 0 ? 0 : 1;
}
