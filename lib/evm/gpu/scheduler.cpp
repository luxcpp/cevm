// Copyright (C) 2026, The cevm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "scheduler.hpp"
#include <algorithm>

namespace evm::gpu
{

Scheduler::Scheduler(uint32_t num_txs)
    : num_txs_{num_txs},
      incarnations_(num_txs),
      validated_(num_txs)
{
    for (auto& inc : incarnations_)
        inc.store(0, std::memory_order_relaxed);
    for (auto& v : validated_)
        v.store(false, std::memory_order_relaxed);
}

Task Scheduler::next_task()
{
    // Try to get an execution task first
    uint32_t idx = execution_idx_.load(std::memory_order_acquire);
    while (idx < num_txs_)
    {
        if (execution_idx_.compare_exchange_weak(idx, idx + 1, std::memory_order_acq_rel))
        {
            return {TaskType::Execute, idx, incarnations_[idx].load(std::memory_order_acquire)};
        }
        idx = execution_idx_.load(std::memory_order_acquire);
    }

    // Try to get a validation task
    uint32_t vidx = validation_idx_.load(std::memory_order_acquire);
    while (vidx < num_txs_)
    {
        if (validation_idx_.compare_exchange_weak(vidx, vidx + 1, std::memory_order_acq_rel))
        {
            return {TaskType::Validate, vidx, incarnations_[vidx].load(std::memory_order_acquire)};
        }
        vidx = validation_idx_.load(std::memory_order_acquire);
    }

    // Check if done
    if (is_done())
        return {TaskType::Done, 0, 0};

    // Wait for more work
    {
        std::unique_lock lock(mu_);
        cv_.wait_for(lock, std::chrono::microseconds(100));
    }

    // Retry
    return next_task();
}

void Scheduler::finish_execution(uint32_t tx_index, uint32_t /*incarnation*/)
{
    // After execution, this tx needs validation
    // Validation will happen when validation_idx_ reaches this tx
    cv_.notify_all();
}

void Scheduler::abort_validation(uint32_t tx_index)
{
    // Increment incarnation and re-queue for execution
    incarnations_[tx_index].fetch_add(1, std::memory_order_release);
    validated_[tx_index].store(false, std::memory_order_release);
    re_executions_.fetch_add(1, std::memory_order_relaxed);

    // Reset execution index to re-execute this tx
    uint32_t expected = execution_idx_.load(std::memory_order_acquire);
    while (expected > tx_index)
    {
        if (execution_idx_.compare_exchange_weak(expected, tx_index, std::memory_order_acq_rel))
            break;
    }

    // Also invalidate all later transactions that may have read from this one
    for (uint32_t i = tx_index + 1; i < num_txs_; ++i)
    {
        validated_[i].store(false, std::memory_order_release);
    }

    // Reset validation index
    uint32_t vexpected = validation_idx_.load(std::memory_order_acquire);
    while (vexpected > tx_index)
    {
        if (validation_idx_.compare_exchange_weak(vexpected, tx_index, std::memory_order_acq_rel))
            break;
    }

    cv_.notify_all();
}

void Scheduler::finish_validation(uint32_t tx_index)
{
    validated_[tx_index].store(true, std::memory_order_release);
    done_count_.fetch_add(1, std::memory_order_release);
    cv_.notify_all();
}

bool Scheduler::is_done() const
{
    for (uint32_t i = 0; i < num_txs_; ++i)
    {
        if (!validated_[i].load(std::memory_order_acquire))
            return false;
    }
    return true;
}

}  // namespace evm::gpu
