// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Internal helpers shared across precompile implementations.
// Not a public header — included only by precompiles/*.cpp / *.mm in this
// library.

#pragma once

#include "precompile_dispatch.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

namespace evm::gpu::precompile::detail
{

inline constexpr uint64_t num_words(size_t bytes) noexcept
{
    return (static_cast<uint64_t>(bytes) + 31u) / 32u;
}

inline constexpr uint64_t cost_per_word(uint64_t base, uint64_t per_word, size_t n) noexcept
{
    return base + per_word * num_words(n);
}

inline Result make_failure(uint64_t gas)
{
    Result r;
    r.ok = false;
    r.out_of_gas = false;
    r.gas_used = gas;
    return r;
}

inline Result make_oog()
{
    Result r;
    r.out_of_gas = true;
    r.gas_used = 0;
    return r;
}

inline Result make_ok(uint64_t gas, std::vector<uint8_t> output)
{
    Result r;
    r.ok = true;
    r.gas_used = gas;
    r.output = std::move(output);
    return r;
}

// EIP-2537 layout constants.
inline constexpr size_t kBlsScalarSize     = 32;
inline constexpr size_t kBlsFieldSize      = 64;
inline constexpr size_t kBlsG1Size         = 2 * kBlsFieldSize;     // 128
inline constexpr size_t kBlsG2Size         = 4 * kBlsFieldSize;     // 256
inline constexpr size_t kBlsG1MulInputSize = kBlsG1Size + kBlsScalarSize; // 160
inline constexpr size_t kBlsG2MulInputSize = kBlsG2Size + kBlsScalarSize; // 288

}  // namespace evm::gpu::precompile::detail

namespace evm::gpu::precompile
{
// CPU implementations — defined in *_cpu.cpp.
Result ecrecover_cpu(std::span<const uint8_t>, uint64_t);
Result sha256_cpu(std::span<const uint8_t>, uint64_t);
Result ripemd160_cpu(std::span<const uint8_t>, uint64_t);
Result identity_cpu(std::span<const uint8_t>, uint64_t);
Result modexp_cpu(std::span<const uint8_t>, uint64_t);
Result bn256_add_cpu(std::span<const uint8_t>, uint64_t);
Result bn256_mul_cpu(std::span<const uint8_t>, uint64_t);
Result bn256_pairing_cpu(std::span<const uint8_t>, uint64_t);
Result blake2f_cpu(std::span<const uint8_t>, uint64_t);
Result point_eval_cpu(std::span<const uint8_t>, uint64_t);
Result bls12_g1add_cpu(std::span<const uint8_t>, uint64_t);
Result bls12_g1msm_cpu(std::span<const uint8_t>, uint64_t);
Result bls12_g2add_cpu(std::span<const uint8_t>, uint64_t);
Result bls12_g2msm_cpu(std::span<const uint8_t>, uint64_t);
Result bls12_pairing_cpu(std::span<const uint8_t>, uint64_t);
Result bls12_map_fp_to_g1_cpu(std::span<const uint8_t>, uint64_t);
Result bls12_map_fp2_to_g2_cpu(std::span<const uint8_t>, uint64_t);

// Lux custom precompile (0x100). Defined in dex_match_cpu.cpp.
Result dex_match_cpu(std::span<const uint8_t>, uint64_t);

}  // namespace evm::gpu::precompile
