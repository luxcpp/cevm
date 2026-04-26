// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file precompile_dispatch.hpp
/// GPU precompile dispatcher.
///
/// Routes EVM precompile calls (addresses 0x01..0x11) to GPU primitives
/// when an accelerated implementation exists, falling back to the CPU path
/// (cevm_precompiles) otherwise. Output bytes and gas accounting match
/// cevm exactly so this can substitute for the CPU dispatcher with no
/// observable consensus difference.
///
/// Backends per precompile:
///   0x01 ECRECOVER          metal/cuda (via secp256k1_recover.{cu,metal})
///   0x02 SHA256             cpu        (cevm)
///   0x03 RIPEMD160          cpu        (cevm)
///   0x04 IDENTITY           cpu        (memcpy)
///   0x05 MODEXP             cpu        (cevm)
///   0x06 BN256_ADD          cpu        (cevm)
///   0x07 BN256_MUL          cpu        (cevm)
///   0x08 BN256_PAIRING      cpu        (cevm)
///   0x09 BLAKE2F            cpu        (cevm)
///   0x0a POINT_EVALUATION   metal/cuda (via bls12_381 + cpu kzg verify)
///   0x0b BLS12_G1ADD        metal/cuda (via bls12_381.{cu,metal})
///   0x0c BLS12_G1MSM        metal/cuda
///   0x0d BLS12_G2ADD        metal/cuda
///   0x0e BLS12_G2MSM        metal/cuda
///   0x0f BLS12_PAIRING      metal/cuda
///   0x10 BLS12_MAP_FP_G1    cpu        (cevm, no GPU primitive)
///   0x11 BLS12_MAP_FP2_G2   cpu        (cevm)
///
/// Usage:
///   auto disp = PrecompileDispatcher::create();
///   auto r = disp->execute(0x01, input_span, gas_limit);
///   if (r.ok) { use(r.output); deduct(r.gas_used); }
///

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace evm::gpu::precompile
{

/// Lowest precompile address (ECRECOVER).
inline constexpr uint16_t kFirstPrecompile = 0x01;

/// Highest standard EVM precompile address (BLS12_MAP_FP2_G2).
inline constexpr uint16_t kLastStandardPrecompile = 0x11;

/// Highest precompile address handled by this dispatcher. The range
/// 0x100..0x1ff is reserved for Lux custom precompiles (DEX_MATCH at 0x100,
/// future settlement / liquidity / TWAP entries).
inline constexpr uint16_t kLastPrecompile = 0x1ff;

/// Lowest Lux custom precompile address (DEX_MATCH).
inline constexpr uint16_t kFirstLuxPrecompile = 0x100;

/// Result of a precompile invocation.
///
/// `ok == false` means either out-of-gas or invalid input — the EVM caller
/// should propagate either OUT_OF_GAS (when gas_used == 0 and the dispatcher
/// reported the expected cost as exceeding gas_limit) or PRECOMPILE_FAILURE
/// otherwise. To keep the consumer side simple this struct also carries
/// `out_of_gas` to disambiguate.
struct Result
{
    bool ok = false;            ///< true if execution succeeded.
    bool out_of_gas = false;    ///< true if gas_limit < required gas.
    uint64_t gas_used = 0;      ///< gas charged (0 on out-of-gas).
    std::vector<uint8_t> output;///< output bytes (sized exactly).
};

/// Backend identifier for diagnostics.
enum class Backend : uint8_t
{
    None = 0,  ///< no implementation registered (precompile not handled).
    Cpu  = 1,  ///< CPU implementation (cevm).
    Metal= 2,  ///< Apple Metal GPU.
    Cuda = 3,  ///< NVIDIA CUDA GPU.
};

const char* backend_name(Backend b) noexcept;

/// GPU precompile dispatcher.
class PrecompileDispatcher
{
public:
    /// Construct a dispatcher and bind the best available backend per
    /// precompile. Always succeeds: if no GPU is available everything
    /// runs on CPU.
    static std::unique_ptr<PrecompileDispatcher> create();

    virtual ~PrecompileDispatcher() = default;

    /// Execute a precompile.
    ///
    /// @param address    Precompile address (1..0x11 standard, 0x100..0x1ff Lux).
    /// @param input      Calldata.
    /// @param gas_limit  Gas available for the call.
    /// @return           Execution result.
    virtual Result execute(uint16_t address,
                           std::span<const uint8_t> input,
                           uint64_t gas_limit) const = 0;

    /// True if a precompile is registered at this address.
    virtual bool available(uint16_t address) const = 0;

    /// Backend bound to a given precompile.
    virtual Backend backend(uint16_t address) const = 0;

    /// Human-readable backend name for diagnostics ("metal", "cuda", "cpu").
    const char* backend_name(uint16_t address) const noexcept
    {
        return precompile::backend_name(backend(address));
    }
};

}  // namespace evm::gpu::precompile
