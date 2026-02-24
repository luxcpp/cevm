// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file state_table.cu
/// CUDA port of metal/state_table.metal — STUB.
///
/// The Metal version provides four batch hash-table kernels for the
/// GPU-resident Ethereum state:
///   - account_lookup_batch
///   - account_insert_batch
///   - storage_lookup_batch
///   - storage_insert_batch
/// Plus a state-root reduction pipeline:
///   - state_root_compact / state_root_sort /
///     state_root_hash_entries / state_root_reduce
///
/// TODO: port from metal/state_table.metal (580 lines).
/// Approach:
///   1. Port the open-addressing key/value structs (AccountEntry,
///      StorageEntry) into __device__ structs with the same byte layout
///      as the Metal versions so host buffers transfer 1:1.
///   2. Replace Metal's atomic_compare_exchange_weak_explicit with
///      CUDA's atomicCAS on the key_valid uint32.
///   3. The state-root reduction tree uses Metal's sort/reduce; CUDA
///      can use cub::DeviceRadixSort + cub::DeviceReduce, OR a
///      hand-rolled grid-stride reduction. Avoid Thrust to keep the
///      dep tree small (the project does not currently use Thrust).
///
/// Until ported, the host wrapper (state_table_host.hpp) returns nullptr
/// from StateTable::create() so callers fall through to the CPU path.

#include <cstdint>
#include <cuda_runtime.h>

namespace evm::gpu::cuda
{

extern "C" cudaError_t evm_cuda_state_table_stub_launch(cudaStream_t /*stream*/)
{
    // No-op stub. Returns success so the caller's launch path doesn't fail,
    // but the host wrapper short-circuits before reaching this anyway.
    return cudaSuccess;
}

}  // namespace evm::gpu::cuda
