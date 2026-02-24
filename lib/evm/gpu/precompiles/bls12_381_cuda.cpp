// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// CUDA-backed BLS12-381 precompiles (0x0b..0x11) — installer stub.
//
// See bls12_381_metal.mm for the rationale: EIP-2537 requires consensus-exact
// input parsing and R-subgroup checks that the underlying GPU kernels in
// luxcpp/gpu/kernels/bls12_381.cu do not yet implement. The CPU path (blst)
// is correct and fast.

#include "precompile_dispatch.hpp"
#include <lux/gpu.h>

extern "C" void evm_precompile_install_bls12_381_cuda(void* dispatcher)
{
    (void)dispatcher;
    LuxGPU* g = lux_gpu_create_with_backend(LUX_BACKEND_CUDA);
    if (g) lux_gpu_destroy(g);
}
