// Copyright (C) 2026, The evmone Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file evm_kernel.cu
/// CUDA port of kernel/evm_kernel.metal — STUB.
///
/// The Metal version is a 772-line single-thread-per-tx EVM opcode
/// interpreter. Each thread:
///   - reads its TxInput (from/to/calldata/etc.)
///   - maintains a private 1024-entry stack of uint256 (4 x uint64 limbs)
///   - dispatches opcodes via switch() jump table
///   - writes results to TxOutput
///   - issues storage reads/writes against a shared GPU storage pool
///
/// Major operations the kernel implements (see evm_kernel.metal):
///   ADD/SUB/MUL/DIV/MOD/EXP, SHL/SHR/SAR, AND/OR/XOR/NOT,
///   LT/GT/EQ/SLT/SGT/ISZERO,
///   PUSH1..PUSH32, DUP1..DUP16, SWAP1..SWAP16,
///   MSTORE/MLOAD/MSTORE8, SLOAD/SSTORE,
///   CALLDATALOAD/COPY, RETURN/REVERT,
///   KECCAK256 (inline keccak_f from keccak256.metal),
///   ADDRESS/BALANCE/CALLER/CALLVALUE/GASPRICE/CHAINID/SELFBALANCE.
///
/// TODO: port from kernel/evm_kernel.metal (772 lines).
/// CUDA mapping notes:
///   - Use uint256 from luxcpp/gpu's evm256.cu via __device__ inline
///     functions; the file already exports add/sub/mul/mod/cmp.
///   - The 1024-deep stack fits in 32 KB per thread which is too large
///     for registers; use a shared-memory pool sized by threads/block,
///     OR keep the V2 evm_stack.hpp pattern (compact 256-byte stack
///     with overflow to global memory).
///   - SLOAD/SSTORE need atomic access against the shared storage_pool
///     which on Metal uses atomic_compare_exchange on key_valid; CUDA
///     equivalent is atomicCAS on a uint32 sentinel.
///   - SSTORE refund counter (EIP-2200/3529) is per-tx, store in
///     TxOutput; matching commit a7fe004 / 8a4c1c3.
///
/// Until ported, the host wrapper (evm_kernel_host.hpp) returns nullptr
/// from EvmKernel::create() and the dispatcher falls back to the CPU
/// evmone interpreter.

#include <cstdint>
#include <cuda_runtime.h>

namespace evm::gpu::cuda
{

extern "C" cudaError_t evm_cuda_evm_execute_stub_launch(cudaStream_t /*stream*/)
{
    return cudaSuccess;
}

}  // namespace evm::gpu::cuda
