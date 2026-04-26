// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "internal.hpp"
#include "precompile_dispatch.hpp"

#include <array>
#include <cstdint>

namespace evm::gpu::precompile
{

const char* backend_name(Backend b) noexcept
{
    switch (b)
    {
    case Backend::None:  return "none";
    case Backend::Cpu:   return "cpu";
    case Backend::Metal: return "metal";
    case Backend::Cuda:  return "cuda";
    }
    return "unknown";
}

namespace
{
using Impl = Result (*)(std::span<const uint8_t>, uint64_t);

class DispatcherImpl final : public PrecompileDispatcher
{
public:
    DispatcherImpl()
    {
        impls_[0x01] = &ecrecover_cpu;
        impls_[0x02] = &sha256_cpu;
        impls_[0x03] = &ripemd160_cpu;
        impls_[0x04] = &identity_cpu;
        impls_[0x05] = &modexp_cpu;
        impls_[0x06] = &bn256_add_cpu;
        impls_[0x07] = &bn256_mul_cpu;
        impls_[0x08] = &bn256_pairing_cpu;
        impls_[0x09] = &blake2f_cpu;
        impls_[0x0a] = &point_eval_cpu;
        impls_[0x0b] = &bls12_g1add_cpu;
        impls_[0x0c] = &bls12_g1msm_cpu;
        impls_[0x0d] = &bls12_g2add_cpu;
        impls_[0x0e] = &bls12_g2msm_cpu;
        impls_[0x0f] = &bls12_pairing_cpu;
        impls_[0x10] = &bls12_map_fp_to_g1_cpu;
        impls_[0x11] = &bls12_map_fp2_to_g2_cpu;

        // Lux custom range (0x100..0x1ff). DEX_MATCH at 0x100.
        impls_[0x100] = &dex_match_cpu;

        for (size_t i = kFirstPrecompile; i <= kLastStandardPrecompile; ++i)
            backends_[i] = Backend::Cpu;
        backends_[0x100] = Backend::Cpu;
    }

    void set(uint16_t addr, Impl fn, Backend b) noexcept
    {
        if (addr <= kLastPrecompile && fn)
        {
            impls_[addr] = fn;
            backends_[addr] = b;
        }
    }

    Result execute(uint16_t address,
                   std::span<const uint8_t> input,
                   uint64_t gas_limit) const override
    {
        if (address > kLastPrecompile)
            return Result{};
        const auto fn = impls_[address];
        if (!fn) return Result{};
        return fn(input, gas_limit);
    }

    bool available(uint16_t address) const override
    {
        if (address > kLastPrecompile) return false;
        return impls_[address] != nullptr;
    }

    Backend backend(uint16_t address) const override
    {
        if (address > kLastPrecompile) return Backend::None;
        return backends_[address];
    }

private:
    // 512 entries: 0x000..0x1ff covers the standard range plus the Lux
    // custom range. Indexed directly by address; sparse but keeps lookup
    // branch-free.
    std::array<Impl, 512>    impls_{};
    std::array<Backend, 512> backends_{};
};

}  // namespace

namespace
{
// Weak references — resolved if the corresponding GPU object is linked in.
extern "C" __attribute__((weak)) void evm_precompile_install_metal(void* dispatcher);
extern "C" __attribute__((weak)) void evm_precompile_install_cuda(void* dispatcher);
extern "C" __attribute__((weak)) void evm_precompile_install_bls12_381_metal(void* dispatcher);
extern "C" __attribute__((weak)) void evm_precompile_install_bls12_381_cuda(void* dispatcher);
extern "C" __attribute__((weak)) void evm_precompile_install_point_eval_metal(void* dispatcher);
extern "C" __attribute__((weak)) void evm_precompile_install_dex_match_metal(void* dispatcher);
}  // namespace

std::unique_ptr<PrecompileDispatcher> PrecompileDispatcher::create()
{
    auto d = std::make_unique<DispatcherImpl>();

#if defined(__APPLE__)
    if (evm_precompile_install_metal)
        evm_precompile_install_metal(d.get());
    if (evm_precompile_install_bls12_381_metal)
        evm_precompile_install_bls12_381_metal(d.get());
    if (evm_precompile_install_point_eval_metal)
        evm_precompile_install_point_eval_metal(d.get());
    if (evm_precompile_install_dex_match_metal)
        evm_precompile_install_dex_match_metal(d.get());
#endif
#if defined(EVM_CUDA)
    if (evm_precompile_install_cuda)
        evm_precompile_install_cuda(d.get());
    if (evm_precompile_install_bls12_381_cuda)
        evm_precompile_install_bls12_381_cuda(d.get());
#endif

    return d;
}

// Public installer hook used by GPU wrappers (ecrecover_metal.mm,
// ecrecover_cuda.cpp, dex_match_metal.mm, ...). Lets a wrapper override a
// single precompile entry.
//
// backend_id matches the Backend enum: 1=Cpu, 2=Metal, 3=Cuda.
extern "C" void evm_precompile_set_impl(
    void* dispatcher, uint16_t address, Impl fn, int backend_id) noexcept
{
    if (!dispatcher) return;
    auto* d = static_cast<DispatcherImpl*>(dispatcher);
    d->set(address, fn, static_cast<Backend>(backend_id));
}

}  // namespace evm::gpu::precompile
