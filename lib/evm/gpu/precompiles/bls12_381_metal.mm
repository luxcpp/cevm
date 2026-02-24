// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Metal-backed BLS12-381 precompiles (0x0b..0x11) — installer stub.
//
// The EIP-2537 BLS12-381 precompiles require strict EVM-format input parsing
// (64-byte field elements, leading 16-byte zero padding, R-subgroup checks),
// which blst handles natively in lib/evmone_precompiles/bls.cpp. The GPU
// kernels in luxcpp/gpu/kernels/bls12_381.metal currently implement primitive
// curve operations (add, mul, pairing) but do not implement the full
// EIP-2537 input validation and subgroup checks required for EVM consensus
// equivalence.
//
// Until the GPU kernels expose an EIP-2537-compatible API, the CPU path
// (blst, via evmone_precompiles) is the only consensus-safe choice and is
// also the fastest available option for BLS12-381 in this build.
//
// This file exists so the build wires correctly and the dispatcher can be
// upgraded to GPU later without changing the public API. Pairing-heavy
// workloads are the natural first target for GPU offload.

#import <Foundation/Foundation.h>

#include "precompile_dispatch.hpp"
#include <lux/gpu.h>

extern "C" void evm_precompile_install_bls12_381_metal(void* dispatcher)
{
    (void)dispatcher;
    // Probe Metal availability and the bls12_381 primitive. If/when the
    // EIP-2537 wrappers are added, register them here via
    // evm_precompile_set_impl(dispatcher, 0x0b..0x11, fn, /*Metal=*/2).
    LuxGPU* g = lux_gpu_create_with_backend(LUX_BACKEND_METAL);
    if (g) lux_gpu_destroy(g);
}
