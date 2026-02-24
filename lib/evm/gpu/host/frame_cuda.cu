// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file frame_cuda.cu
/// CUDA-side frame execution adapter for the host bridge.
///
/// Mirror of frame_metal.mm for the CUDA backend. When the CUDA kernel grows
/// a CALL handler with a code dictionary, this file owns the dispatch.
/// Today it forwards a single frame to `cuda::EvmKernel`.
///
/// The .cu suffix is purely so this file lives next to the other CUDA
/// sources; it contains no device code yet.

#include "../cuda/evm_kernel_host.hpp"

namespace evm::gpu::host {

[[nodiscard]] cuda::TxResult run_frame_cuda(
    cuda::EvmKernel& kernel,
    const cuda::HostTransaction& frame)
{
    cuda::HostTransaction txs[1] = {frame};
    auto results = kernel.execute({txs, 1});
    if (results.empty())
        return cuda::TxResult{cuda::TxStatus::Error, 0, 0, {}};
    return results.front();
}

}  // namespace evm::gpu::host
