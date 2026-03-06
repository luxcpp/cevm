// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file bench_state.cpp
/// Benchmark: 10k ETH transfers through the full state layer.
///
/// Measures the complete transaction processing pipeline:
/// nonce validation, balance checks, gas accounting, state mutation,
/// journal snapshots, and state root computation.
///
/// Build: cmake --build build --target evm-bench-state
/// Usage: evm-bench-state [num_txs] [num_runs]

#include "processor.hpp"
#include "state_db.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

// Forward-declare cevm factory.
extern "C" struct evmc_vm* evmc_create_cevm(void) noexcept;

namespace
{

/// Generate N ETH transfer transactions.
/// Sender(i) -> Recipient(i + num_txs), 1 wei each.
std::vector<evm::state::Transaction> generate_transfers(uint32_t num_txs)
{
    std::vector<evm::state::Transaction> txs;
    txs.reserve(num_txs);

    for (uint32_t i = 0; i < num_txs; ++i)
    {
        evm::state::Transaction tx{};

        // Sender address derived from index.
        tx.sender = {};
        tx.sender.bytes[19] = static_cast<uint8_t>(i & 0xFF);
        tx.sender.bytes[18] = static_cast<uint8_t>((i >> 8) & 0xFF);
        tx.sender.bytes[17] = static_cast<uint8_t>((i >> 16) & 0xFF);

        // Recipient address: distinct from sender.
        tx.recipient = {};
        tx.recipient.bytes[19] = static_cast<uint8_t>((i + num_txs) & 0xFF);
        tx.recipient.bytes[18] = static_cast<uint8_t>(((i + num_txs) >> 8) & 0xFF);
        tx.recipient.bytes[17] = static_cast<uint8_t>(((i + num_txs) >> 16) & 0xFF);

        tx.value = 1;  // 1 wei
        tx.gas_limit = 21000;
        tx.gas_price = 1;
        tx.nonce = 0;  // Each sender sends exactly one tx.
        tx.is_create = false;

        txs.push_back(std::move(tx));
    }

    return txs;
}

/// Pre-fund all sender accounts with enough balance for gas + value.
void setup_state(evm::state::StateDB& db, const std::vector<evm::state::Transaction>& txs)
{
    for (const auto& tx : txs)
    {
        db.create_account(tx.sender);
        // Fund: gas_limit * gas_price + value + generous margin.
        const auto funding = intx::uint256{tx.gas_limit} * tx.gas_price + tx.value;
        db.set_balance(tx.sender, funding);
    }
    // Commit initial state (clears journal).
    (void)db.commit();
}

struct RunStats
{
    double mean_ms;
    double stddev_ms;
    double mean_mgas;
    double min_ms;
};

RunStats compute_stats(const std::vector<double>& times_ms, uint64_t total_gas)
{
    RunStats s{};
    const auto n = static_cast<double>(times_ms.size());

    s.mean_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / n;
    s.min_ms = *std::min_element(times_ms.begin(), times_ms.end());

    double var = 0;
    for (auto t : times_ms)
        var += (t - s.mean_ms) * (t - s.mean_ms);
    s.stddev_ms = std::sqrt(var / n);

    if (s.mean_ms > 0)
        s.mean_mgas = static_cast<double>(total_gas) / s.mean_ms / 1000.0;
    else
        s.mean_mgas = 0;

    return s;
}

}  // namespace

int main(int argc, char* argv[])
{
    uint32_t num_txs = 10000;
    uint32_t num_runs = 5;
    if (argc > 1)
        num_txs = static_cast<uint32_t>(std::atoi(argv[1]));
    if (argc > 2)
        num_runs = static_cast<uint32_t>(std::atoi(argv[2]));

    const uint64_t gas_per_tx = 21000;
    const uint64_t total_gas = static_cast<uint64_t>(num_txs) * gas_per_tx;

    std::printf("C++ EVM State Layer Benchmark\n");
    std::printf("==============================\n");
    std::printf("Transactions:  %u\n", num_txs);
    std::printf("Gas per tx:    %llu\n", static_cast<unsigned long long>(gas_per_tx));
    std::printf("Total gas:     %llu\n", static_cast<unsigned long long>(total_gas));
    std::printf("Runs:          %u\n", num_runs);
    std::printf("\n");

    const auto txs = generate_transfers(num_txs);

    // Block context.
    evm::state::TxContext tx_ctx{};
    tx_ctx.coinbase = {};
    tx_ctx.coinbase.bytes[19] = 0xFF;  // coinbase address
    tx_ctx.block_number = 1;
    tx_ctx.block_timestamp = 1700000000;
    tx_ctx.block_gas_limit = 30'000'000;
    tx_ctx.chain_id = {};
    tx_ctx.chain_id.bytes[31] = 1;

    auto* vm = evmc_create_cevm();

    // --- Full state benchmark ---
    std::vector<double> state_times;
    state_times.reserve(num_runs);
    evmc::bytes32 last_root{};

    for (uint32_t r = 0; r < num_runs; ++r)
    {
        evm::state::StateDB db;
        setup_state(db, txs);

        auto result = evm::state::process_block(db, txs, vm, tx_ctx, EVMC_SHANGHAI);
        state_times.push_back(result.execution_time_ms);
        last_root = result.state_root;

        // Verify all transactions succeeded.
        uint32_t failures = 0;
        for (const auto& tr : result.tx_results)
        {
            if (tr.status != EVMC_SUCCESS)
                ++failures;
        }
        if (failures > 0)
            std::printf("WARNING: %u transactions failed in run %u\n", failures, r);
    }

    auto state_stats = compute_stats(state_times, total_gas);

    // --- Print results ---
    std::printf("%-22s| %-16s| %-12s| %s\n",
        "Mode", "Time (ms)", "Mgas/s", "State Root (first 8 bytes)");
    std::printf("%-22s| %-16s| %-12s| %s\n",
        "----------------------", "----------------", "------------",
        "--------------------------");

    char time_buf[32];
    char mgas_buf[16];
    char root_buf[20];
    std::snprintf(time_buf, sizeof(time_buf), "%.2f +/- %.2f",
                  state_stats.mean_ms, state_stats.stddev_ms);
    std::snprintf(mgas_buf, sizeof(mgas_buf), "%.0f", state_stats.mean_mgas);
    std::snprintf(root_buf, sizeof(root_buf), "%02x%02x%02x%02x%02x%02x%02x%02x",
                  last_root.bytes[0], last_root.bytes[1], last_root.bytes[2], last_root.bytes[3],
                  last_root.bytes[4], last_root.bytes[5], last_root.bytes[6], last_root.bytes[7]);

    std::printf("%-22s| %-16s| %-12s| %s\n",
        "Full State Layer", time_buf, mgas_buf, root_buf);

    std::printf("\nBest run:  %.2f ms (%.0f Mgas/s)\n",
                state_stats.min_ms,
                state_stats.min_ms > 0
                    ? static_cast<double>(total_gas) / state_stats.min_ms / 1000.0
                    : 0.0);

    // Target comparison.
    std::printf("\nGo EVM target: 51ms for 10k transfers\n");
    if (num_txs == 10000)
    {
        const auto speedup = 51.0 / state_stats.mean_ms;
        std::printf("C++ vs Go:     %.1fx %s\n",
                    speedup, speedup > 1.0 ? "faster" : "slower");
    }

    std::printf("\n");

    vm->destroy(vm);
    return 0;
}
