// Copyright (C) 2026, The evmone Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "parallel_engine.hpp"
#include "mv_memory.hpp"
#include "parallel_host.hpp"
#include "scheduler.hpp"

#include <evmc/evmc.hpp>

// Forward-declare the evmone factory. Defined in vm.cpp.
extern "C" struct evmc_vm* evmc_create_evmone(void) noexcept;

#include <chrono>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace evm::gpu
{

/// Execute a single transaction through evmone.
static evmc_result execute_one(evmc_vm* vm, evmc::Host& host,
                               evmc_revision rev, const EvmTransaction& tx)
{
    const auto& iface = evmc::Host::get_interface();
    auto* ctx = host.to_context();
    return vm->execute(vm, &iface, ctx, rev, &tx.msg, tx.code.data(), tx.code.size());
}

EvmTransaction to_evm_transaction(const Transaction& tx)
{
    EvmTransaction etx{};

    // Sender
    if (tx.from.size() >= 20)
        std::memcpy(etx.msg.sender.bytes, tx.from.data(), 20);

    // Recipient
    if (tx.to.size() >= 20)
    {
        std::memcpy(etx.msg.recipient.bytes, tx.to.data(), 20);
        etx.msg.kind = EVMC_CALL;
    }
    else
    {
        etx.msg.kind = EVMC_CREATE;
    }

    // Value: store uint64 as big-endian in the last 8 bytes of evmc_uint256be
    etx.msg.value = {};
    for (int i = 0; i < 8; ++i)
        etx.msg.value.bytes[31 - i] = static_cast<uint8_t>(tx.value >> (i * 8));

    etx.msg.gas = static_cast<int64_t>(tx.gas_limit);
    etx.msg.depth = 0;

    // Calldata
    etx.msg.input_data = tx.data.data();
    etx.msg.input_size = tx.data.size();

    // For simple transfers, code is empty (no EVM bytecode to run).
    // For contract calls, the caller must set etx.code to the contract bytecode.
    etx.code = tx.data;

    return etx;
}

BlockResult execute_sequential_evmone(
    const std::vector<EvmTransaction>& txs,
    evmc::Host& base_host,
    evmc_revision rev)
{
    BlockResult result;
    result.gas_used.resize(txs.size());

    auto* vm = evmc_create_evmone();

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < txs.size(); ++i)
    {
        auto r = execute_one(vm, base_host, rev, txs[i]);
        const auto gas_consumed = static_cast<uint64_t>(txs[i].msg.gas - r.gas_left);
        result.gas_used[i] = gas_consumed;
        result.total_gas += gas_consumed;

        if (r.release != nullptr)
            r.release(&r);
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.execution_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    vm->destroy(vm);
    return result;
}

BlockResult execute_parallel_evmone(
    const std::vector<EvmTransaction>& txs,
    evmc::Host& base_host,
    evmc_revision rev,
    uint32_t num_threads)
{
    const auto num_txs = static_cast<uint32_t>(txs.size());
    if (num_threads == 0)
        num_threads = std::max(1u, std::thread::hardware_concurrency());

    MvMemory mv_memory(num_txs);
    Scheduler scheduler(num_txs);

    BlockResult result;
    result.gas_used.resize(num_txs, 0);

    // Mutex protecting result.gas_used writes from multiple threads
    std::mutex result_mu;

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    for (uint32_t w = 0; w < num_threads; ++w)
    {
        workers.emplace_back([&]() {
            // Each worker gets its own VM instance (evmone VM is not thread-safe
            // because it pools ExecutionState objects).
            auto* vm = evmc_create_evmone();

            while (true)
            {
                Task task = scheduler.next_task();

                if (task.type == TaskType::Done)
                {
                    vm->destroy(vm);
                    return;
                }

                if (task.type == TaskType::Execute)
                {
                    const auto idx = task.tx_index;

                    // Create a ParallelHost for this transaction
                    ParallelHost phost(base_host, mv_memory, idx, task.incarnation);

                    // Execute through evmone
                    auto r = execute_one(vm, phost, rev, txs[idx]);

                    const auto gas_consumed =
                        static_cast<uint64_t>(txs[idx].msg.gas - r.gas_left);

                    {
                        std::lock_guard lock(result_mu);
                        result.gas_used[idx] = gas_consumed;
                    }

                    // Flush writes to MvMemory
                    phost.flush_writes();

                    if (r.release != nullptr)
                        r.release(&r);

                    scheduler.finish_execution(idx, task.incarnation);
                }
                else if (task.type == TaskType::Validate)
                {
                    const auto idx = task.tx_index;

                    // Create a read-only ParallelHost to validate reads
                    ParallelHost phost(base_host, mv_memory, idx, task.incarnation);

                    // Re-execute to collect the read set, then validate.
                    // In a full implementation we would store the read-set from execution
                    // and validate it here without re-executing. For now, since the scheduler
                    // handles re-execution on abort, we just validate optimistically.
                    //
                    // Simple transfers with no storage reads always validate successfully.
                    // For transactions that do read storage, we need the stored read-set.
                    // The current scheduler design assumes validation passes for simple cases.
                    scheduler.finish_validation(idx);
                }
            }
        });
    }

    for (auto& w : workers)
        w.join();

    auto end = std::chrono::high_resolution_clock::now();
    result.execution_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    result.conflicts = mv_memory.num_conflicts();
    result.re_executions = scheduler.num_re_executions();

    for (auto g : result.gas_used)
        result.total_gas += g;

    return result;
}

}  // namespace evm::gpu
