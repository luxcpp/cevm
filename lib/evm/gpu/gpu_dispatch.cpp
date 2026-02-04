// Copyright (C) 2026, The evmone Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "gpu_dispatch.hpp"
#include "mv_memory.hpp"
#include "scheduler.hpp"

#include <chrono>
#include <thread>

namespace evm::gpu
{

const char* backend_name(Backend b)
{
    switch (b)
    {
    case Backend::CPU_Sequential: return "cpu-sequential";
    case Backend::CPU_Parallel:   return "cpu-parallel (Block-STM)";
    case Backend::GPU_Metal:      return "gpu-metal";
    case Backend::GPU_CUDA:       return "gpu-cuda";
    }
    return "unknown";
}

std::vector<Backend> available_backends()
{
    std::vector<Backend> backends;
    backends.push_back(Backend::CPU_Sequential);
    backends.push_back(Backend::CPU_Parallel);

#ifdef __APPLE__
    // Metal is available on macOS/iOS
    backends.push_back(Backend::GPU_Metal);
#endif

#ifdef EVM_CUDA
    backends.push_back(Backend::GPU_CUDA);
#endif

    return backends;
}

/// Execute a block sequentially (baseline).
static BlockResult execute_sequential(const std::vector<Transaction>& txs, void* /*state*/)
{
    BlockResult result;
    result.gas_used.resize(txs.size());

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < txs.size(); ++i)
    {
        // TODO: integrate with evmone's execute() function
        // For now, simulate execution with gas consumption
        result.gas_used[i] = txs[i].gas_limit;
        result.total_gas += txs[i].gas_limit;
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.execution_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    return result;
}

/// Execute a block in parallel using Block-STM.
static BlockResult execute_parallel(const Config& config,
                                    const std::vector<Transaction>& txs,
                                    void* /*state*/)
{
    const uint32_t num_txs = static_cast<uint32_t>(txs.size());
    const uint32_t num_threads = config.num_threads > 0
        ? config.num_threads
        : std::thread::hardware_concurrency();

    MvMemory mv_memory(num_txs);
    Scheduler scheduler(num_txs);

    BlockResult result;
    result.gas_used.resize(num_txs, 0);

    auto start = std::chrono::high_resolution_clock::now();

    // Spawn worker threads
    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    for (uint32_t w = 0; w < num_threads; ++w)
    {
        workers.emplace_back([&]() {
            while (true)
            {
                Task task = scheduler.next_task();
                if (task.type == TaskType::Done)
                    break;

                if (task.type == TaskType::Execute)
                {
                    // Execute transaction speculatively
                    // TODO: integrate with evmone's execute() using MvMemory for state access
                    result.gas_used[task.tx_index] = txs[task.tx_index].gas_limit;

                    scheduler.finish_execution(task.tx_index, task.incarnation);
                }
                else if (task.type == TaskType::Validate)
                {
                    // Validate: check if reads are still valid
                    // TODO: check MvMemory read set against current versions
                    // For now, assume all validations pass (no conflicts in simple transfers)
                    scheduler.finish_validation(task.tx_index);
                }
            }
        });
    }

    // Wait for all workers to finish
    for (auto& w : workers)
        w.join();

    auto end = std::chrono::high_resolution_clock::now();
    result.execution_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    result.conflicts = mv_memory.num_conflicts();
    result.re_executions = scheduler.num_re_executions();

    // Sum total gas
    for (auto g : result.gas_used)
        result.total_gas += g;

    return result;
}

BlockResult execute_block(const Config& config,
                          const std::vector<Transaction>& txs,
                          void* state)
{
    switch (config.backend)
    {
    case Backend::CPU_Sequential:
        return execute_sequential(txs, state);

    case Backend::CPU_Parallel:
        return execute_parallel(config, txs, state);

    case Backend::GPU_Metal:
        // TODO: Metal compute shader dispatch
        // Fall back to CPU parallel for now
        return execute_parallel(config, txs, state);

    case Backend::GPU_CUDA:
        // TODO: CUDA kernel dispatch
        // Fall back to CPU parallel for now
        return execute_parallel(config, txs, state);
    }

    return execute_sequential(txs, state);
}

}  // namespace evm::gpu
