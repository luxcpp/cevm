// Copyright (C) 2026, The evmone Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file evm_kernel_host.hpp
/// CUDA host interface for the GPU EVM interpreter — STUB.
///
/// Mirrors the API of kernel/evm_kernel_host.hpp (Metal). Today,
/// EvmKernel::create() returns nullptr; the dispatcher falls back to
/// the CPU evmone interpreter.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace evm::gpu::cuda
{

/// Returns false until evm_kernel.cu is fully ported.
inline bool evm_kernel_cuda_available() { return false; }

/// CUDA-accelerated EVM interpreter — stub.
class EvmKernel
{
public:
    virtual ~EvmKernel() = default;

    /// Always returns nullptr until the kernel is ported.
    static std::unique_ptr<EvmKernel> create() { return nullptr; }

    virtual const char* device_name() const = 0;

protected:
    EvmKernel() = default;
    EvmKernel(const EvmKernel&) = delete;
    EvmKernel& operator=(const EvmKernel&) = delete;
};

}  // namespace evm::gpu::cuda
