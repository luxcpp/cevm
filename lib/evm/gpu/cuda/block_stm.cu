// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file block_stm.cu
/// CUDA port of metal/block_stm.metal — STUB.
///
/// The Metal version dispatches a single block_stm_execute kernel with
/// per-thread workers competing for execution / validation slots via
/// atomic counters in shared sched_state. Each worker:
///   1. atomicAdd(execution_idx) — claim a tx to (re-)execute
///   2. atomicAdd(validation_idx) — claim a tx to validate
///   3. records reads/writes into MvMemory open-addressed by addr+slot
///   4. on validation conflict: bumps incarnation, requeues for execute
///
/// TODO: port from metal/block_stm.metal (600 lines).
/// CUDA mapping:
///   - Metal's atomic_uint maps to CUDA atomicAdd / atomicCAS
///   - Metal's memory_order_relaxed → CUDA's default __threadfence not
///     needed since most ops are independent counters; tighter ordering
///     for MV table writes via __threadfence_block / __threadfence
///   - Use one block per worker; thread count should be tuned for SM
///     occupancy (warp = 32 threads; aim for 128-256 threads/block)
///   - The MV table uses linear probing on a power-of-2 array; CUDA
///     can keep the same hash function (FNV-1a on addr+slot) and
///     implement CAS-based slot claim with atomicCAS on key_valid.
///
/// Until ported, the host wrapper (block_stm_host.hpp) returns nullptr
/// from BlockStmGpu::create() so the dispatcher falls back to the CPU
/// Block-STM scheduler in parallel_engine.cpp.

#include <cstdint>
#include <cuda_runtime.h>

namespace evm::gpu::cuda
{

extern "C" cudaError_t evm_cuda_block_stm_stub_launch(cudaStream_t /*stream*/)
{
    return cudaSuccess;
}

}  // namespace evm::gpu::cuda
