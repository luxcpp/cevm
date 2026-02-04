// Copyright (C) 2026, The evmone Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file bench_parallel.cpp
/// Benchmark comparing CPU_Sequential vs CPU_Parallel (Block-STM) execution.
///
/// Generates N simple ETH transfer transactions and runs them through both
/// backends, printing timing comparison.
///
/// Build:
///   g++ -std=c++20 -O2 -pthread bench_parallel.cpp parallel_engine.cpp \
///       mv_memory.cpp scheduler.cpp gpu_dispatch.cpp -I../../.. -o bench_parallel

#include "gpu_dispatch.hpp"
#include "parallel_engine.hpp"
#include <evmc/evmc.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace
{

/// Minimal EVMC Host for benchmarking.
/// Stores nothing — returns zeros for all state queries.
/// Sufficient for measuring execution overhead of simple transfers.
class BenchHost : public evmc::Host
{
public:
    bool account_exists(const evmc::address&) const noexcept override { return true; }

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

    evmc::uint256be get_balance(const evmc::address&) const noexcept override
    {
        // Return a large balance so transfers succeed
        evmc::uint256be bal{};
        bal.bytes[0] = 0xFF;  // ~2^248
        return bal;
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
        return evmc::Result{EVMC_SUCCESS, msg.gas, 0};
    }

    evmc_tx_context get_tx_context() const noexcept override
    {
        evmc_tx_context ctx{};
        ctx.block_gas_limit = 30'000'000;
        ctx.block_number = 1;
        ctx.block_timestamp = 1700000000;
        // Set chain_id
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
};

/// Generate N simple ETH transfer transactions.
/// Each transfer sends 1 wei from address(i) to address(i + num_txs).
std::vector<evm::gpu::EvmTransaction> generate_transfers(uint32_t num_txs)
{
    std::vector<evm::gpu::EvmTransaction> txs;
    txs.reserve(num_txs);

    for (uint32_t i = 0; i < num_txs; ++i)
    {
        evm::gpu::EvmTransaction tx{};
        tx.msg.kind = EVMC_CALL;
        tx.msg.gas = 21000;  // Standard ETH transfer gas
        tx.msg.depth = 0;

        // Sender: address derived from index
        tx.msg.sender = {};
        tx.msg.sender.bytes[19] = static_cast<uint8_t>(i & 0xFF);
        tx.msg.sender.bytes[18] = static_cast<uint8_t>((i >> 8) & 0xFF);

        // Recipient: distinct from sender
        tx.msg.recipient = {};
        tx.msg.recipient.bytes[19] = static_cast<uint8_t>((i + num_txs) & 0xFF);
        tx.msg.recipient.bytes[18] = static_cast<uint8_t>(((i + num_txs) >> 8) & 0xFF);

        // Value: 1 wei
        tx.msg.value = {};
        tx.msg.value.bytes[31] = 1;

        // No code for simple transfers
        tx.code.clear();

        txs.push_back(std::move(tx));
    }

    return txs;
}

/// Generate N simple transfer transactions in the dispatch-layer format.
std::vector<evm::gpu::Transaction> generate_dispatch_transfers(uint32_t num_txs)
{
    std::vector<evm::gpu::Transaction> txs;
    txs.reserve(num_txs);

    for (uint32_t i = 0; i < num_txs; ++i)
    {
        evm::gpu::Transaction tx{};
        tx.from.resize(20, 0);
        tx.from[19] = static_cast<uint8_t>(i & 0xFF);
        tx.from[18] = static_cast<uint8_t>((i >> 8) & 0xFF);

        tx.to.resize(20, 0);
        tx.to[19] = static_cast<uint8_t>((i + num_txs) & 0xFF);
        tx.to[18] = static_cast<uint8_t>(((i + num_txs) >> 8) & 0xFF);

        tx.gas_limit = 21000;
        tx.value = 1;

        txs.push_back(std::move(tx));
    }

    return txs;
}

void print_result(const char* label, const evm::gpu::BlockResult& r, uint32_t num_txs)
{
    std::printf("  %-30s %8.3f ms  |  %u txs  |  total gas: %llu  |  "
                "conflicts: %u  re-exec: %u\n",
        label, r.execution_time_ms, num_txs,
        static_cast<unsigned long long>(r.total_gas),
        r.conflicts, r.re_executions);
}

}  // namespace

int main(int argc, char* argv[])
{
    uint32_t num_txs = 1000;
    if (argc > 1)
        num_txs = static_cast<uint32_t>(std::atoi(argv[1]));

    std::printf("Block-STM Parallel Execution Benchmark\n");
    std::printf("=======================================\n");
    std::printf("Transactions: %u\n", num_txs);
    std::printf("Hardware threads: %u\n", std::thread::hardware_concurrency());
    std::printf("\n");

    // --- Benchmark via dispatch layer (execute_block) ---
    std::printf("Dispatch layer (gpu_dispatch.hpp):\n");
    {
        auto txs = generate_dispatch_transfers(num_txs);
        BenchHost host;

        evm::gpu::Config seq_cfg{};
        seq_cfg.backend = evm::gpu::Backend::CPU_Sequential;
        auto r_seq = evm::gpu::execute_block(seq_cfg, txs, &host);
        print_result("CPU_Sequential", r_seq, num_txs);

        evm::gpu::Config par_cfg{};
        par_cfg.backend = evm::gpu::Backend::CPU_Parallel;
        auto r_par = evm::gpu::execute_block(par_cfg, txs, &host);
        print_result("CPU_Parallel (Block-STM)", r_par, num_txs);

        if (r_seq.execution_time_ms > 0)
        {
            std::printf("  Speedup: %.2fx\n",
                r_seq.execution_time_ms / r_par.execution_time_ms);
        }
    }

    std::printf("\n");

    // --- Benchmark via engine directly ---
    std::printf("Engine layer (parallel_engine.hpp):\n");
    {
        auto txs = generate_transfers(num_txs);
        BenchHost host;

        auto r_seq = evm::gpu::execute_sequential_evmone(txs, host, EVMC_SHANGHAI);
        print_result("Sequential evmone", r_seq, num_txs);

        auto r_par = evm::gpu::execute_parallel_evmone(txs, host, EVMC_SHANGHAI, 0);
        print_result("Parallel Block-STM + evmone", r_par, num_txs);

        if (r_seq.execution_time_ms > 0)
        {
            std::printf("  Speedup: %.2fx\n",
                r_seq.execution_time_ms / r_par.execution_time_ms);
        }
    }

    std::printf("\nDone.\n");
    return 0;
}
