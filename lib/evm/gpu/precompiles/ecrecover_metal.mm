// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Metal-backed ECRECOVER precompile (0x01).
//
// Wraps lux_gpu_ecrecover_batch (secp256k1_recover.metal). For compatibility
// with the EVM precompile format we translate the calldata layout into the
// LuxEcrecoverInput struct expected by the kernel:
//
//   EVM input (128 bytes): hash[32] || v[32 BE] || r[32] || s[32]
//   LuxEcrecoverInput     : r[32] || s[32] || v[1] || pad[3] || hash[32] || pad[28]
//
// The recovered Ethereum address is left-padded into a 32-byte output buffer.
//
// If the Metal kernel reports `valid=0` we fall back to the CPU implementation
// — there are corner cases (small-r signatures, certain k values) that the
// reference Metal shader does not yet handle and consensus equivalence is
// non-negotiable.

#import <Foundation/Foundation.h>

#include "precompile_dispatch.hpp"
#include <lux/gpu.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

namespace evm::gpu::precompile
{
// Defined in precompile_dispatch.cpp.
extern "C" void evm_precompile_set_impl(
    void* dispatcher, uint8_t address, Result (*fn)(std::span<const uint8_t>, uint64_t),
    int backend_id) noexcept;

// CPU fallback (defined in ecrecover_cpu.cpp).
Result ecrecover_cpu(std::span<const uint8_t>, uint64_t);

namespace
{
// Singleton: one LuxGPU per process. We don't own the dispatcher's lifetime.
LuxGPU* gpu_metal()
{
    static LuxGPU* g = []() -> LuxGPU* {
        return lux_gpu_create_with_backend(LUX_BACKEND_METAL);
    }();
    return g;
}

Result ecrecover_metal(std::span<const uint8_t> input, uint64_t gas_limit)
{
    constexpr uint64_t kGas = 3000;
    if (gas_limit < kGas)
    {
        Result r;
        r.out_of_gas = true;
        return r;
    }

    // Pad input to 128 bytes.
    uint8_t buf[128]{};
    std::memcpy(buf, input.data(), std::min<size_t>(input.size(), 128));

    // Validate v (must be 27 or 28 in big-endian uint256).
    // Bytes 32..63 hold v; only the last byte may be non-zero, and the value
    // must equal 27 or 28.
    for (size_t i = 32; i < 63; ++i)
    {
        if (buf[i] != 0)
        {
            Result r; r.gas_used = kGas; r.ok = false;
            return r;
        }
    }
    const uint8_t v = buf[63];
    if (v != 27 && v != 28)
    {
        Result r; r.gas_used = kGas; r.ok = false;
        return r;
    }

    LuxGPU* gpu = gpu_metal();
    if (!gpu)
        return ecrecover_cpu(input, gas_limit);

    LuxEcrecoverInput sig;
    std::memset(&sig, 0, sizeof(sig));
    std::memcpy(sig.r, buf + 64, 32);
    std::memcpy(sig.s, buf + 96, 32);
    sig.v = static_cast<uint8_t>(v - 27);  // kernel expects 0/1
    std::memcpy(sig.msg_hash, buf, 32);

    LuxEcrecoverOutput out;
    std::memset(&out, 0, sizeof(out));

    const LuxError err = lux_gpu_ecrecover_batch(gpu, &sig, &out, 1);
    if (err != LUX_OK || !out.valid)
    {
        // Metal kernel does not recover every valid signature.
        // Fall back to the CPU reference for consensus exactness.
        return ecrecover_cpu(input, gas_limit);
    }

    std::vector<uint8_t> output(32, 0);
    std::memcpy(output.data() + 12, out.address, 20);

    Result r;
    r.ok = true;
    r.gas_used = kGas;
    r.output = std::move(output);
    return r;
}

}  // namespace

}  // namespace evm::gpu::precompile

// Installer entry called from PrecompileDispatcher::create().
extern "C" void evm_precompile_install_metal(void* dispatcher)
{
    using namespace evm::gpu::precompile;

    // Probe Metal at install-time. If unavailable, leave CPU defaults intact.
    LuxGPU* g = lux_gpu_create_with_backend(LUX_BACKEND_METAL);
    if (!g) return;
    lux_gpu_destroy(g);  // gpu_metal() will create its own singleton

    // Backend::Metal == 2.
    evm_precompile_set_impl(dispatcher, 0x01, &ecrecover_metal, 2);
}
