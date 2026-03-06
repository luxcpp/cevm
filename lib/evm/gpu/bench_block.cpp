// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file bench_block.cpp
/// Benchmark comparing CPU_Sequential vs CPU_Parallel (Block-STM) execution
/// of ETH transfer blocks through cevm.
///
/// Build: cmake --build build --target evm-bench-block
/// Usage: evm-bench-block [num_txs] [num_runs]

#include "gpu_dispatch.hpp"
#include "parallel_engine.hpp"
#include <evmc/evmc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <thread>
#include <unordered_map>
#include <vector>

namespace
{

/// In-memory EVMC Host that tracks balances per address.
/// Each ETH transfer: deduct from sender, add to receiver.
class BenchHost : public evmc::Host
{
public:
    explicit BenchHost(uint64_t initial_balance) : initial_balance_{initial_balance} {}

    /// Reset all balances for a fresh run.
    void reset() { balances_.clear(); }

    bool account_exists(const evmc::address& addr) const noexcept override
    {
        return balances_.count(addr) > 0 || true;  // all accounts "exist"
    }

    evmc::bytes32 get_storage(const evmc::address&,
                              const evmc::bytes32&) const noexcept override
    {
        return {};
    }

    evmc_storage_status set_storage(const evmc::address&, const evmc::bytes32&,
                                    const evmc::bytes32&) noexcept override
    {
        return EVMC_STORAGE_MODIFIED;
    }

    evmc::uint256be get_balance(const evmc::address& addr) const noexcept override
    {
        uint64_t bal = initial_balance_;
        if (auto it = balances_.find(addr); it != balances_.end())
            bal = it->second;
        evmc::uint256be result{};
        for (int i = 0; i < 8; ++i)
            result.bytes[31 - i] = static_cast<uint8_t>(bal >> (i * 8));
        return result;
    }

    size_t get_code_size(const evmc::address&) const noexcept override { return 0; }

    evmc::bytes32 get_code_hash(const evmc::address&) const noexcept override { return {}; }

    size_t copy_code(const evmc::address&, size_t, uint8_t*,
                     size_t) const noexcept override
    {
        return 0;
    }

    bool selfdestruct(const evmc::address&, const evmc::address&) noexcept override
    {
        return false;
    }

    evmc::Result call(const evmc_message& msg) noexcept override
    {
        // Simple transfer: deduct value from sender, add to receiver
        uint64_t value = 0;
        for (int i = 0; i < 8; ++i)
            value |= static_cast<uint64_t>(msg.value.bytes[31 - i]) << (i * 8);

        if (value > 0)
        {
            auto& sender_bal = balances_[msg.sender];
            if (sender_bal == 0)
                sender_bal = initial_balance_;
            sender_bal -= std::min(sender_bal, value);

            auto& recip_bal = balances_[msg.recipient];
            if (recip_bal == 0)
                recip_bal = initial_balance_;
            recip_bal += value;
        }

        return evmc::Result{EVMC_SUCCESS, msg.gas, 0};
    }

    evmc_tx_context get_tx_context() const noexcept override
    {
        evmc_tx_context ctx{};
        ctx.block_gas_limit = 30'000'000;
        ctx.block_number = 1;
        ctx.block_timestamp = 1700000000;
        ctx.chain_id.bytes[31] = 1;
        return ctx;
    }

    evmc::bytes32 get_block_hash(int64_t) const noexcept override { return {}; }

    void emit_log(const evmc::address&, const uint8_t*, size_t,
                  const evmc::bytes32[], size_t) noexcept override {}

    evmc_access_status access_account(const evmc::address&) noexcept override
    {
        return EVMC_ACCESS_WARM;
    }

    evmc_access_status access_storage(const evmc::address&,
                                      const evmc::bytes32&) noexcept override
    {
        return EVMC_ACCESS_WARM;
    }

    evmc::bytes32 get_transient_storage(const evmc::address&,
                                        const evmc::bytes32&) const noexcept override
    {
        return {};
    }

    void set_transient_storage(const evmc::address&, const evmc::bytes32&,
                               const evmc::bytes32&) noexcept override {}

private:
    uint64_t initial_balance_;
    mutable std::unordered_map<evmc::address, uint64_t> balances_;
};

/// Generate N simple ETH transfer transactions as EVMC messages.
/// Each transfer sends 1 wei from address(i) to address(i + num_txs).
std::vector<evm::gpu::EvmTransaction> generate_transfers(uint32_t num_txs)
{
    std::vector<evm::gpu::EvmTransaction> txs;
    txs.reserve(num_txs);

    for (uint32_t i = 0; i < num_txs; ++i)
    {
        evm::gpu::EvmTransaction tx{};
        tx.msg.kind = EVMC_CALL;
        tx.msg.gas = 21000;
        tx.msg.depth = 0;

        // Sender: derived from index
        tx.msg.sender = {};
        tx.msg.sender.bytes[19] = static_cast<uint8_t>(i & 0xFF);
        tx.msg.sender.bytes[18] = static_cast<uint8_t>((i >> 8) & 0xFF);
        tx.msg.sender.bytes[17] = static_cast<uint8_t>((i >> 16) & 0xFF);

        // Recipient: distinct from sender
        tx.msg.recipient = {};
        tx.msg.recipient.bytes[19] = static_cast<uint8_t>((i + num_txs) & 0xFF);
        tx.msg.recipient.bytes[18] = static_cast<uint8_t>(((i + num_txs) >> 8) & 0xFF);
        tx.msg.recipient.bytes[17] = static_cast<uint8_t>(((i + num_txs) >> 16) & 0xFF);

        // Value: 1 wei
        tx.msg.value = {};
        tx.msg.value.bytes[31] = 1;

        // No bytecode for simple transfers
        tx.code.clear();

        txs.push_back(std::move(tx));
    }

    return txs;
}

struct RunStats
{
    double mean_ms;
    double stddev_ms;
    double mean_mgas;
    double stddev_mgas;
};

RunStats compute_stats(const std::vector<double>& times_ms, uint64_t total_gas)
{
    RunStats s{};
    const auto n = static_cast<double>(times_ms.size());

    s.mean_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / n;

    double var = 0;
    for (auto t : times_ms)
        var += (t - s.mean_ms) * (t - s.mean_ms);
    s.stddev_ms = std::sqrt(var / n);

    // Mgas/s for each run, then average
    std::vector<double> mgas_vals;
    mgas_vals.reserve(times_ms.size());
    for (auto t : times_ms)
    {
        if (t > 0)
            mgas_vals.push_back(static_cast<double>(total_gas) / t / 1000.0);
        else
            mgas_vals.push_back(0);
    }

    s.mean_mgas = std::accumulate(mgas_vals.begin(), mgas_vals.end(), 0.0) / n;

    double var_mgas = 0;
    for (auto m : mgas_vals)
        var_mgas += (m - s.mean_mgas) * (m - s.mean_mgas);
    s.stddev_mgas = std::sqrt(var_mgas / n);

    return s;
}

void print_row(const char* label, const RunStats& s, double baseline_ms)
{
    double speedup = (baseline_ms > 0) ? baseline_ms / s.mean_ms : 1.0;
    char time_buf[32];
    char mgas_buf[32];
    std::snprintf(time_buf, sizeof(time_buf), "%.1f +/- %.1f", s.mean_ms, s.stddev_ms);
    std::snprintf(mgas_buf, sizeof(mgas_buf), "%.0f +/- %.0f", s.mean_mgas, s.stddev_mgas);
    std::printf("%-22s| %-14s| %-14s| %.1fx\n", label, time_buf, mgas_buf, speedup);
}

}  // namespace

int main(int argc, char* argv[])
{
    uint32_t num_txs = 10000;
    uint32_t num_runs = 3;
    if (argc > 1)
        num_txs = static_cast<uint32_t>(std::atoi(argv[1]));
    if (argc > 2)
        num_runs = static_cast<uint32_t>(std::atoi(argv[2]));

    const uint64_t gas_per_tx = 21000;
    const uint64_t total_gas = static_cast<uint64_t>(num_txs) * gas_per_tx;
    const unsigned hw_threads = std::thread::hardware_concurrency();

    std::printf("C++ EVM Block Execution Benchmark\n");
    std::printf("==================================\n");
    std::printf("Transactions: %u\n", num_txs);
    std::printf("Gas per tx:   %llu\n", static_cast<unsigned long long>(gas_per_tx));
    std::printf("Hardware:     %u threads\n", hw_threads);
    std::printf("Runs:         %u\n", num_runs);
    std::printf("\n");

    auto txs = generate_transfers(num_txs);

    // --- CPU Sequential ---
    std::vector<double> seq_times;
    seq_times.reserve(num_runs);
    for (uint32_t r = 0; r < num_runs; ++r)
    {
        BenchHost host(1'000'000'000);
        auto result = evm::gpu::execute_sequential_cevm(txs, host, EVMC_SHANGHAI);
        seq_times.push_back(result.execution_time_ms);
    }
    auto seq_stats = compute_stats(seq_times, total_gas);

    // --- CPU Parallel at 4 threads ---
    std::vector<double> par4_times;
    par4_times.reserve(num_runs);
    for (uint32_t r = 0; r < num_runs; ++r)
    {
        BenchHost host(1'000'000'000);
        auto result = evm::gpu::execute_parallel_cevm(txs, host, EVMC_SHANGHAI, 4);
        par4_times.push_back(result.execution_time_ms);
    }
    auto par4_stats = compute_stats(par4_times, total_gas);

    // --- CPU Parallel at hardware thread count ---
    std::vector<double> parN_times;
    parN_times.reserve(num_runs);
    for (uint32_t r = 0; r < num_runs; ++r)
    {
        BenchHost host(1'000'000'000);
        auto result = evm::gpu::execute_parallel_cevm(txs, host, EVMC_SHANGHAI, 0);
        parN_times.push_back(result.execution_time_ms);
    }
    auto parN_stats = compute_stats(parN_times, total_gas);

    // --- Print results ---
    std::printf("%-22s| %-14s| %-14s| %s\n",
        "Mode", "Time (ms)", "Mgas/s", "Speedup");
    std::printf("%-22s| %-14s| %-14s| %s\n",
        "----------------------", "--------------", "--------------", "-------");

    print_row("CPU Sequential", seq_stats, seq_stats.mean_ms);

    char par4_label[64];
    std::snprintf(par4_label, sizeof(par4_label), "CPU Parallel (4T)");
    print_row(par4_label, par4_stats, seq_stats.mean_ms);

    char parN_label[64];
    std::snprintf(parN_label, sizeof(parN_label), "CPU Parallel (%uT)", hw_threads);
    print_row(parN_label, parN_stats, seq_stats.mean_ms);

    // GPU placeholder
    std::printf("%-22s| %-14s| %-14s| %s\n",
        "GPU (pending)", "--", "--", "--");

    std::printf("\n");
    return 0;
}
