// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// CUDA-backed ECRECOVER precompile (0x01).
//
// Same input/output translation as ecrecover_metal.mm but routes the
// LuxEcrecoverInput batch through LUX_BACKEND_CUDA.

#include "precompile_dispatch.hpp"
#include <lux/gpu.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

namespace evm::gpu::precompile
{
extern "C" void evm_precompile_set_impl(
    void* dispatcher, uint8_t address, Result (*fn)(std::span<const uint8_t>, uint64_t),
    int backend_id) noexcept;

// CPU fallback (defined in ecrecover_cpu.cpp).
Result ecrecover_cpu(std::span<const uint8_t>, uint64_t);

namespace
{
LuxGPU* gpu_cuda()
{
    static LuxGPU* g = []() -> LuxGPU* {
        return lux_gpu_create_with_backend(LUX_BACKEND_CUDA);
    }();
    return g;
}

Result ecrecover_cuda(std::span<const uint8_t> input, uint64_t gas_limit)
{
    constexpr uint64_t kGas = 3000;
    if (gas_limit < kGas)
    {
        Result r; r.out_of_gas = true; return r;
    }

    uint8_t buf[128]{};
    std::memcpy(buf, input.data(), std::min<size_t>(input.size(), 128));

    for (size_t i = 32; i < 63; ++i)
    {
        if (buf[i] != 0) { Result r; r.gas_used = kGas; return r; }
    }
    const uint8_t v = buf[63];
    if (v != 27 && v != 28) { Result r; r.gas_used = kGas; return r; }

    LuxGPU* gpu = gpu_cuda();
    if (!gpu)
        return ecrecover_cpu(input, gas_limit);

    LuxEcrecoverInput sig;
    std::memset(&sig, 0, sizeof(sig));
    std::memcpy(sig.r, buf + 64, 32);
    std::memcpy(sig.s, buf + 96, 32);
    sig.v = static_cast<uint8_t>(v - 27);
    std::memcpy(sig.msg_hash, buf, 32);

    LuxEcrecoverOutput out;
    std::memset(&out, 0, sizeof(out));

    const LuxError err = lux_gpu_ecrecover_batch(gpu, &sig, &out, 1);
    if (err != LUX_OK || !out.valid)
        return ecrecover_cpu(input, gas_limit);

    std::vector<uint8_t> output(32, 0);
    std::memcpy(output.data() + 12, out.address, 20);
    Result r;
    r.ok = true; r.gas_used = kGas; r.output = std::move(output);
    return r;
}

}  // namespace

}  // namespace evm::gpu::precompile

extern "C" void evm_precompile_install_cuda(void* dispatcher)
{
    using namespace evm::gpu::precompile;

    LuxGPU* g = lux_gpu_create_with_backend(LUX_BACKEND_CUDA);
    if (!g) return;
    lux_gpu_destroy(g);

    // Backend::Cuda == 3.
    evm_precompile_set_impl(dispatcher, 0x01, &ecrecover_cuda, 3);
}
