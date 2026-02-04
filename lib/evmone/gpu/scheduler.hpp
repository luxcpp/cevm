// Copyright (C) 2026, The evmone Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file scheduler.hpp
/// Block-STM collaborative scheduler.
///
/// Assigns execution and validation tasks to worker threads.
/// Port of ~/work/lux/evmgpu/core/parallel/scheduler.go to C++.

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>

namespace evm::gpu
{

/// Task types for workers.
enum class TaskType : uint8_t
{
    Execute,    ///< Execute transaction at given index
    Validate,   ///< Validate transaction at given index
    Done,       ///< No more work
};

/// A task assigned to a worker.
struct Task
{
    TaskType type;
    uint32_t tx_index;
    uint32_t incarnation;
};

/// Collaborative scheduler for Block-STM.
///
/// Workers call next_task() to get work. When a validation fails,
/// the scheduler re-queues the transaction for re-execution and
/// cascades invalidation to dependent transactions.
class Scheduler
{
public:
    explicit Scheduler(uint32_t num_txs);

    /// Get the next task for a worker. Blocks if no work available.
    /// Returns Task with type=Done when all transactions are finalized.
    Task next_task();

    /// Report that execution of (tx_index, incarnation) completed.
    void finish_execution(uint32_t tx_index, uint32_t incarnation);

    /// Report that validation of tx_index failed (conflict detected).
    /// This re-queues tx_index and all transactions that read from it.
    void abort_validation(uint32_t tx_index);

    /// Report that validation of tx_index succeeded.
    void finish_validation(uint32_t tx_index);

    /// Check if all transactions are finalized.
    bool is_done() const;

    /// Get statistics.
    uint32_t num_re_executions() const { return re_executions_.load(std::memory_order_relaxed); }

private:
    uint32_t num_txs_;
    std::vector<std::atomic<uint32_t>> incarnations_;
    std::vector<std::atomic<bool>> validated_;
    std::atomic<uint32_t> execution_idx_{0};
    std::atomic<uint32_t> validation_idx_{0};
    std::atomic<uint32_t> done_count_{0};
    std::atomic<uint32_t> re_executions_{0};
    mutable std::mutex mu_;
    std::condition_variable cv_;
};

}  // namespace evm::gpu
