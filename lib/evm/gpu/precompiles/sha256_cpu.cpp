// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// 0x02 SHA-256 precompile (CPU). Gas = 60 + 12 * ceil(N/32). Output = 32 bytes.

#include "internal.hpp"

#include <evmone_precompiles/sha256.hpp>

namespace evm::gpu::precompile
{
Result sha256_cpu(std::span<const uint8_t> input, uint64_t gas_limit)
{
    using namespace detail;
    const uint64_t gas = cost_per_word(60, 12, input.size());
    if (gas_limit < gas)
        return make_oog();

    std::vector<uint8_t> out(32);
    evmone::crypto::sha256(reinterpret_cast<std::byte*>(out.data()),
        reinterpret_cast<const std::byte*>(input.data()), input.size());
    return make_ok(gas, std::move(out));
}
}  // namespace evm::gpu::precompile
