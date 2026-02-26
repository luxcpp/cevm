// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// 0x0a POINT_EVALUATION precompile (EIP-4844). Gas = 50000 (fixed).
// Input = versioned_hash[32] || z[32] || y[32] || commitment[48] || proof[48]
//       = 192 bytes.
// Output on success = FIELD_ELEMENTS_PER_BLOB[32 BE] || BLS_MODULUS[32 BE].

#include "internal.hpp"

#include <evmone_precompiles/kzg.hpp>

#include <intx/intx.hpp>

namespace evm::gpu::precompile
{
Result point_eval_cpu(std::span<const uint8_t> input, uint64_t gas_limit)
{
    using namespace detail;
    constexpr uint64_t kGas = 50000;
    if (gas_limit < kGas)
        return make_oog();
    if (input.size() != 192)
        return make_failure(kGas);

    const auto p = reinterpret_cast<const std::byte*>(input.data());
    const bool ok = evmone::crypto::kzg_verify_proof(p, p + 32, p + 64, p + 96, p + 144);
    if (!ok)
        return make_failure(kGas);

    std::vector<uint8_t> out(64);
    intx::be::unsafe::store(out.data(), evmone::crypto::FIELD_ELEMENTS_PER_BLOB);
    intx::be::unsafe::store(out.data() + 32, evmone::crypto::BLS_MODULUS);
    return make_ok(kGas, std::move(out));
}
}  // namespace evm::gpu::precompile
