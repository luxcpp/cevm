// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file evm_stack.hpp
/// EVM stack for GPU kernel execution.
///
/// 1024-depth stack of uint256 values. On GPU, the active portion lives
/// in registers/local memory (thread-private). The EVM spec allows at most
/// 1024 items, but typical contract execution uses far fewer (<100).
///
/// To keep register pressure manageable on GPU, we store the stack in
/// thread-local memory (spills to device local memory automatically).

#pragma once

#include "uint256_gpu.hpp"

namespace evm::gpu::kernel {

/// Maximum EVM stack depth per the Yellow Paper.
static constexpr gpu_u32 STACK_LIMIT = 1024;

/// EVM execution status codes.
enum class ExecStatus : gpu_u32
{
    Ok          = 0,
    Stop        = 1,
    Return      = 2,
    Revert      = 3,
    OutOfGas    = 4,
    StackOverflow  = 5,
    StackUnderflow = 6,
    InvalidJump    = 7,
    InvalidOpcode  = 8,
    WriteProtection = 9,
    InvalidMemAccess = 10,
    CallNotSupported = 11,
};

/// GPU EVM stack.
///
/// Items are stored bottom-up: data[0] is the first pushed item.
/// `top` points to the next free slot (top == 0 means empty stack).
struct EvmStack
{
    uint256 data[STACK_LIMIT];
    gpu_u32 top;  // number of items on the stack

    GPU_INLINE EvmStack() : top(0) {}

    GPU_INLINE gpu_u32 size() const { return top; }

    GPU_INLINE ExecStatus push(const uint256& val)
    {
        if (top >= STACK_LIMIT)
            return ExecStatus::StackOverflow;
        data[top++] = val;
        return ExecStatus::Ok;
    }

    GPU_INLINE ExecStatus pop(uint256& out)
    {
        if (top == 0)
            return ExecStatus::StackUnderflow;
        out = data[--top];
        return ExecStatus::Ok;
    }

    /// Peek at the top item without removing it.
    GPU_INLINE ExecStatus peek(uint256& out) const
    {
        if (top == 0)
            return ExecStatus::StackUnderflow;
        out = data[top - 1];
        return ExecStatus::Ok;
    }

    /// Peek at item at depth n from top (0 = top, 1 = second from top, etc.).
    GPU_INLINE ExecStatus peek_at(gpu_u32 depth, uint256& out) const
    {
        if (depth >= top)
            return ExecStatus::StackUnderflow;
        out = data[top - 1 - depth];
        return ExecStatus::Ok;
    }

    /// SWAP: swap top with item at depth n (1-indexed: SWAP1 swaps top with second).
    GPU_INLINE ExecStatus swap(gpu_u32 n)
    {
        if (n >= top)
            return ExecStatus::StackUnderflow;
        gpu_u32 idx = top - 1 - n;
        uint256 tmp = data[top - 1];
        data[top - 1] = data[idx];
        data[idx] = tmp;
        return ExecStatus::Ok;
    }

    /// DUP: duplicate the item at depth n (1-indexed: DUP1 duplicates top).
    GPU_INLINE ExecStatus dup(gpu_u32 n)
    {
        if (n > top || n == 0)
            return ExecStatus::StackUnderflow;
        if (top >= STACK_LIMIT)
            return ExecStatus::StackOverflow;
        data[top] = data[top - n];
        ++top;
        return ExecStatus::Ok;
    }

    /// Pop and discard the top item.
    GPU_INLINE ExecStatus drop()
    {
        if (top == 0)
            return ExecStatus::StackUnderflow;
        --top;
        return ExecStatus::Ok;
    }

    /// Direct reference to top-of-stack (no bounds check).
    GPU_INLINE uint256& top_ref() { return data[top - 1]; }
    GPU_INLINE const uint256& top_ref() const { return data[top - 1]; }
};

}  // namespace evm::gpu::kernel
