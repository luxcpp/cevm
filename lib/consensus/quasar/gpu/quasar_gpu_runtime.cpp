// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file quasar_gpu_runtime.cpp
/// QuasarGPUEngine factory dispatcher. Picks the Metal backend on Apple,
/// the CUDA backend when EVM_CUDA is defined, and returns nullptr
/// otherwise. Both backends implement the same QuasarGPUEngine interface
/// (quasar_gpu_engine.hpp); this file is the single public entry point.

#include "quasar_gpu_engine.hpp"

namespace quasar::gpu {

#if defined(__APPLE__)
std::unique_ptr<QuasarGPUEngine> create_quasar_gpu_engine_metal();
#endif

#if defined(EVM_CUDA)
std::unique_ptr<QuasarGPUEngine> create_quasar_gpu_engine_cuda();
#endif

std::unique_ptr<QuasarGPUEngine> QuasarGPUEngine::create()
{
#if defined(__APPLE__)
    return create_quasar_gpu_engine_metal();
#elif defined(EVM_CUDA)
    return create_quasar_gpu_engine_cuda();
#else
    return nullptr;
#endif
}

}  // namespace quasar::gpu
