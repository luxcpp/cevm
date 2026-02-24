// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file frame_metal.mm
/// Metal-side frame execution adapter for the host bridge.
///
/// The bridge's job is to pre-resolve all call targets and stage frames.
/// This file is the seam where, when the Metal kernel grows a CALL handler
/// that takes a code dictionary, the bridge will hand off the staged frame
/// stack to it. Today the kernel only runs leaf-pure frames, so the Metal
/// side is just a wrapper around `EvmKernelHost` that runs one frame.
///
/// Keeping this file separate from `frame_cuda.cu` lets the bridge link
/// either backend without pulling in the other.

#import <Foundation/Foundation.h>

#include "../kernel/evm_kernel_host.hpp"

namespace evm::gpu::host {

// -- Adapter that runs a single leaf frame on Metal --------------------------
//
// Built on top of the existing EvmKernelHost (Metal V1 kernel). Returns a
// kernel::TxResult — the bridge's frame manager translates that into an
// evmc::Result.
//
// The function is a thin wrapper today; it exists so the bridge can call a
// stable C++ entry point without including Foundation/Metal headers, and so
// the file owns the Metal-specific symbols if/when they grow.

[[nodiscard]] kernel::TxResult run_frame_metal(
    kernel::EvmKernelHost& host_obj,
    const kernel::HostTransaction& frame)
{
    kernel::HostTransaction txs[1] = {frame};
    auto results = host_obj.execute({txs, 1});
    if (results.empty())
        return kernel::TxResult{kernel::TxStatus::Error, 0, {}};
    return results.front();
}

}  // namespace evm::gpu::host
