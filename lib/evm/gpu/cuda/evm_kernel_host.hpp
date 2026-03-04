// Copyright (C) 2026, The evmone Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file evm_kernel_host.hpp
/// CUDA host interface for the GPU EVM interpreter.
///
/// Mirrors the API of kernel/evm_kernel_host.hpp (Metal). The CUDA backend
/// dispatches one thread per transaction. Transactions that hit a
/// CALL/CREATE-family opcode complete with status == CallNotSupported so
/// the caller can re-execute them on the CPU evmone interpreter.
///
/// Wire layout note: TxInput, TxOutput, and StorageEntry below are kept
/// byte-for-byte identical to evm_kernel.cu's device structs. The host
/// just memcpys the buffer to the device. static_assert at the bottom of
/// evm_kernel_host.cpp validates the sizes against the device side.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace evm::gpu::cuda
{

// -- Constants matching evm_kernel.cu ----------------------------------------

static constexpr uint32_t HOST_MAX_MEMORY_PER_TX  = 65536;
static constexpr uint32_t HOST_MAX_OUTPUT_PER_TX  = 1024;
static constexpr uint32_t HOST_MAX_STORAGE_PER_TX = 64;

// -- uint256 host mirror (matches device-side evm_kernel.cu uint256) ----------

struct alignas(8) uint256_host
{
    uint64_t w[4];  // little-endian: w[0] = low 64 bits

    constexpr uint256_host() : w{0, 0, 0, 0} {}
    constexpr explicit uint256_host(uint64_t lo) : w{lo, 0, 0, 0} {}
    constexpr uint256_host(uint64_t w0, uint64_t w1, uint64_t w2, uint64_t w3)
        : w{w0, w1, w2, w3} {}
};
static_assert(sizeof(uint256_host) == 32, "uint256_host size mismatch");

// -- GPU buffer descriptors (must match device structs in evm_kernel.cu) ------

struct TxInput
{
    uint32_t      code_offset;
    uint32_t      code_size;
    uint32_t      calldata_offset;
    uint32_t      calldata_size;
    uint64_t      gas_limit;
    uint256_host  caller;
    uint256_host  address;
    uint256_host  value;
};

struct TxOutput
{
    uint32_t  status;       // 0=stop, 1=return, 2=revert, 3=oog, 4=error, 5=call_not_supported
    uint64_t  gas_used;
    int64_t   gas_refund;   // EIP-2200 raw refund (signed). EIP-3529 cap applied at dispatcher.
    uint32_t  output_size;
};

struct StorageEntry
{
    uint256_host key;
    uint256_host value;
};

// -- Status enum --------------------------------------------------------------

enum class TxStatus : uint32_t
{
    Stop             = 0,
    Return           = 1,
    Revert           = 2,
    OutOfGas         = 3,
    Error            = 4,
    CallNotSupported = 5,  // CALL/CREATE/etc — needs CPU fallback
};

// -- Host-side input + result -------------------------------------------------

struct HostTransaction
{
    std::vector<uint8_t> code;
    std::vector<uint8_t> calldata;
    uint64_t     gas_limit = 0;
    uint256_host caller;
    uint256_host address;
    uint256_host value;
};

struct TxResult
{
    TxStatus              status = TxStatus::Error;
    uint64_t              gas_used = 0;
    int64_t               gas_refund = 0;  // EIP-2200 raw; cap applied at dispatcher
    std::vector<uint8_t>  output;
};

// -- Availability check -------------------------------------------------------

/// Returns true once nvcc has compiled the .cu kernel and CUDA reports a
/// usable device. The host create() will return nullptr if this is false.
bool evm_kernel_cuda_available();

// -- Public interface ---------------------------------------------------------

/// CUDA-accelerated EVM interpreter (V1: 1 thread per tx).
class EvmKernel
{
public:
    virtual ~EvmKernel() = default;

    /// Returns a CUDA-backed kernel host. Returns nullptr if no CUDA device
    /// is present or the build did not include CUDA support.
    static std::unique_ptr<EvmKernel> create();

    /// Execute a batch of transactions on the GPU.
    ///
    /// CALL/CREATE-family opcodes set status == CallNotSupported; the caller
    /// is responsible for re-running those transactions on the CPU.
    virtual std::vector<TxResult> execute(std::span<const HostTransaction> txs) = 0;

    /// Get the CUDA device name (e.g. "NVIDIA H100 PCIe").
    virtual const char* device_name() const = 0;

protected:
    EvmKernel() = default;
    EvmKernel(const EvmKernel&) = delete;
    EvmKernel& operator=(const EvmKernel&) = delete;
};

}  // namespace evm::gpu::cuda
