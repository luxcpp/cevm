// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Metal-backed POINT_EVALUATION precompile (0x0a) — installer stub.
//
// EIP-4844 KZG point-evaluation requires:
//   1. SHA-256(commitment) == versioned_hash[1..]
//   2. KZG verify(commitment, z, y, proof)
//
// (2) is a single BLS12-381 pairing equality check. The pairing primitive
// exists in luxcpp/gpu/kernels/bls12_381.metal, but plumbing the
// trusted-setup G2 element and the EIP-4844-specific layout into a
// GPU-side verifier requires more than a one-shot wrapper. The CPU path
// uses blst's c-kzg implementation (verified via `crypto::kzg_verify_proof`)
// and is consensus-correct.
//
// This file exists so the install hook is wired. When a GPU KZG verifier
// is added it will register here via evm_precompile_set_impl(0x0a, ...).

#import <Foundation/Foundation.h>

#include "precompile_dispatch.hpp"
#include <lux/gpu.h>

extern "C" void evm_precompile_install_point_eval_metal(void* dispatcher)
{
    (void)dispatcher;
    LuxGPU* g = lux_gpu_create_with_backend(LUX_BACKEND_METAL);
    if (g) lux_gpu_destroy(g);
}
