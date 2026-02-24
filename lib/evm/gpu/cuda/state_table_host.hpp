// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file state_table_host.hpp
/// CUDA host interface for GPU-resident state hash table — STUB.
///
/// Mirrors the API surface that a future port of metal/state_table.metal
/// would expose. Today, StateTable::create() always returns nullptr;
/// callers must fall back to the CPU state path.
///
/// TODO: port the underlying kernels (state_table.cu) and finish this
/// host-side wrapper. Until then, treating cuda_available() as false
/// keeps the dispatcher correctness-equivalent to the existing Metal
/// behaviour on systems without Metal.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace evm::gpu::cuda
{

/// Returns false until state_table.cu is implemented.
inline bool state_table_cuda_available() { return false; }

class StateTable
{
public:
    virtual ~StateTable() = default;

    /// Always returns nullptr in this stub.
    static std::unique_ptr<StateTable> create() { return nullptr; }

    virtual const char* device_name() const = 0;

protected:
    StateTable() = default;
    StateTable(const StateTable&) = delete;
    StateTable& operator=(const StateTable&) = delete;
};

}  // namespace evm::gpu::cuda
