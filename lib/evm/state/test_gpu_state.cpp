// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file test_gpu_state.cpp
/// Tests for GPU-native state database (GpuHashTable + GpuNativeStateDB).

#include "gpu_hashtable.hpp"
#include "gpu_state_db.hpp"
#include "account.hpp"

#include <evmc/evmc.hpp>
#include <intx/intx.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Test assertion that works regardless of NDEBUG.
#define CHECK(expr)                                                   \
    do {                                                              \
        if (!(expr)) {                                                \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n",                \
                         __FILE__, __LINE__, #expr);                  \
            std::abort();                                             \
        }                                                             \
    } while (0)

using namespace evm::state;
using intx::uint256;

static evmc::address make_addr(uint8_t byte)
{
    evmc::address addr{};
    addr.bytes[19] = byte;
    return addr;
}

static evmc::bytes32 make_key(uint8_t byte)
{
    evmc::bytes32 key{};
    key.bytes[31] = byte;
    return key;
}

static bool is_nonzero(const evmc::bytes32& h)
{
    for (int i = 0; i < 32; ++i)
        if (h.bytes[i] != 0) return true;
    return false;
}

static evmc::bytes32 make_val(uint8_t byte)
{
    evmc::bytes32 val{};
    val.bytes[31] = byte;
    return val;
}

// -- Test 1: GpuHashTable account insert + lookup ---

static void test_account_insert_lookup()
{
    auto ht = GpuHashTable::create(1024);
    if (!ht)
    {
        std::printf("  SKIP: Metal not available\n");
        return;
    }

    evmc::address addr = make_addr(0x42);
    GpuAccountData data{};
    data.nonce = 7;
    data.balance[0] = 1000;
    auto ech = empty_code_hash();
    std::memcpy(data.code_hash, ech.bytes, 32);

    // Insert.
    ht->insert_accounts(&addr, &data, 1);

    // Lookup.
    GpuAccountData result{};
    uint32_t found = 0;
    ht->lookup_accounts(&addr, 1, &result, &found);

    CHECK(found == 1);
    CHECK(result.nonce == 7);
    CHECK(result.balance[0] == 1000);
    std::printf("  PASS: account insert + lookup\n");
}

// -- Test 2: Account not found ---

static void test_account_not_found()
{
    auto ht = GpuHashTable::create(1024);
    if (!ht) { std::printf("  SKIP\n"); return; }

    evmc::address addr = make_addr(0x99);
    GpuAccountData result{};
    uint32_t found = 0;
    ht->lookup_accounts(&addr, 1, &result, &found);

    CHECK(found == 0);
    std::printf("  PASS: account not found\n");
}

// -- Test 3: Storage insert + lookup ---

static void test_storage_insert_lookup()
{
    auto ht = GpuHashTable::create(1024);
    if (!ht) { std::printf("  SKIP\n"); return; }

    GpuStorageKey sk{};
    evmc::address addr = make_addr(0x01);
    evmc::bytes32 slot = make_key(0x05);
    evmc::bytes32 val = make_val(0xAB);
    std::memcpy(sk.addr, addr.bytes, 20);
    std::memcpy(sk.slot, slot.bytes, 32);

    ht->insert_storage(&sk, &val, 1);

    evmc::bytes32 result{};
    uint32_t found = 0;
    ht->lookup_storage(&sk, 1, &result, &found);

    CHECK(found == 1);
    CHECK(result.bytes[31] == 0xAB);
    std::printf("  PASS: storage insert + lookup\n");
}

// -- Test 4: Batch operations ---

static void test_batch_operations()
{
    auto ht = GpuHashTable::create(4096);
    if (!ht) { std::printf("  SKIP\n"); return; }

    constexpr uint32_t N = 100;
    evmc::address addrs[N];
    GpuAccountData datas[N];

    for (uint32_t i = 0; i < N; ++i)
    {
        addrs[i] = make_addr(static_cast<uint8_t>(i));
        std::memset(&datas[i], 0, sizeof(GpuAccountData));
        datas[i].nonce = i * 10;
        datas[i].balance[0] = i * 100;
        auto ech = empty_code_hash();
        std::memcpy(datas[i].code_hash, ech.bytes, 32);
    }

    ht->insert_accounts(addrs, datas, N);

    GpuAccountData results[N];
    uint32_t founds[N];
    ht->lookup_accounts(addrs, N, results, founds);

    for (uint32_t i = 0; i < N; ++i)
    {
        CHECK(founds[i] == 1);
        CHECK(results[i].nonce == i * 10);
        CHECK(results[i].balance[0] == i * 100);
    }
    std::printf("  PASS: batch insert + lookup (%u accounts)\n", N);
}

// -- Test 5: Account update (overwrite) ---

static void test_account_update()
{
    auto ht = GpuHashTable::create(1024);
    if (!ht) { std::printf("  SKIP\n"); return; }

    evmc::address addr = make_addr(0x10);
    GpuAccountData data{};
    data.nonce = 1;
    data.balance[0] = 500;
    auto ech = empty_code_hash();
    std::memcpy(data.code_hash, ech.bytes, 32);

    ht->insert_accounts(&addr, &data, 1);

    // Update balance.
    data.nonce = 2;
    data.balance[0] = 999;
    ht->insert_accounts(&addr, &data, 1);

    GpuAccountData result{};
    uint32_t found = 0;
    ht->lookup_accounts(&addr, 1, &result, &found);

    CHECK(found == 1);
    CHECK(result.nonce == 2);
    CHECK(result.balance[0] == 999);
    std::printf("  PASS: account update\n");
}

// -- Test 6: State root computation ---

static void test_state_root()
{
    auto ht = GpuHashTable::create(1024);
    if (!ht) { std::printf("  SKIP\n"); return; }

    evmc::address addr = make_addr(0x01);
    GpuAccountData data{};
    data.nonce = 1;
    data.balance[0] = 1000;
    auto ech = empty_code_hash();
    std::memcpy(data.code_hash, ech.bytes, 32);

    ht->insert_accounts(&addr, &data, 1);

    auto root = ht->compute_state_root();

    CHECK(is_nonzero(root));

    // Same state should produce same root.
    CHECK(std::memcmp(root.bytes, ht->compute_state_root().bytes, 32) == 0);

    std::printf("  PASS: state root is non-zero and deterministic\n");
}

// -- Test 7: GpuNativeStateDB end-to-end ---

static void test_native_state_db()
{
    GpuNativeStateDB db;
    if (!db.gpu_available())
    {
        std::printf("  SKIP: Metal not available\n");
        return;
    }

    evmc::address alice = make_addr(0xA1);
    evmc::address bob = make_addr(0xB0);

    // Create accounts.
    db.create_account(alice);
    db.create_account(bob);

    CHECK(db.account_exists(alice));
    CHECK(db.account_exists(bob));

    // Set balances.
    db.set_balance(alice, 1000);
    db.set_balance(bob, 500);

    CHECK(db.get_balance(alice) == 1000);
    CHECK(db.get_balance(bob) == 500);

    // Transfer.
    db.sub_balance(alice, 200);
    db.add_balance(bob, 200);

    CHECK(db.get_balance(alice) == 800);
    CHECK(db.get_balance(bob) == 700);

    // Nonce.
    db.set_nonce(alice, 5);
    CHECK(db.get_nonce(alice) == 5);
    db.increment_nonce(alice);
    CHECK(db.get_nonce(alice) == 6);

    // Storage.
    evmc::bytes32 key = make_key(0x01);
    evmc::bytes32 val = make_val(0xFF);
    db.set_storage(alice, key, val);
    CHECK(db.get_storage(alice, key) == val);

    // Empty storage returns zero.
    CHECK(db.get_storage(alice, make_key(0x02)) == evmc::bytes32{});

    // Code.
    std::vector<uint8_t> code = {0x60, 0x00, 0x60, 0x00, 0xF3}; // PUSH0 PUSH0 RETURN
    db.set_code(alice, code);
    CHECK(db.get_code_size(alice) == 5);
    CHECK(db.get_code(alice) == code);

    // Commit produces non-zero state root.
    CHECK(is_nonzero(db.commit()));

    std::printf("  PASS: GpuNativeStateDB end-to-end\n");
}

// -- Test 8: Snapshot and revert ---

static void test_snapshot_revert()
{
    GpuNativeStateDB db;
    if (!db.gpu_available()) { std::printf("  SKIP\n"); return; }

    evmc::address addr = make_addr(0xAA);
    db.create_account(addr);
    db.set_balance(addr, 1000);

    int snap = db.snapshot();

    db.set_balance(addr, 2000);
    CHECK(db.get_balance(addr) == 2000);

    db.revert(snap);
    CHECK(db.get_balance(addr) == 1000);

    std::printf("  PASS: snapshot + revert\n");
}

// -- Main ---

int main()
{
    std::printf("=== GPU State Database Tests ===\n\n");

    std::printf("GpuHashTable:\n");
    test_account_insert_lookup();
    test_account_not_found();
    test_storage_insert_lookup();
    test_batch_operations();
    test_account_update();
    test_state_root();

    std::printf("\nGpuNativeStateDB:\n");
    test_native_state_db();
    test_snapshot_revert();

    std::printf("\nAll tests passed.\n");
    return 0;
}
