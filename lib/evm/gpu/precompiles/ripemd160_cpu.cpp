// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// 0x03 RIPEMD-160 precompile (CPU). Gas = 600 + 120 * ceil(N/32).
// Output = 32 bytes (12 zero bytes + 20-byte digest).

#include "internal.hpp"

#include <evmone_precompiles/ripemd160.hpp>

namespace evm::gpu::precompile
{
Result ripemd160_cpu(std::span<const uint8_t> input, uint64_t gas_limit)
{
    using namespace detail;
    const uint64_t gas = cost_per_word(600, 120, input.size());
    if (gas_limit < gas)
        return make_oog();

    std::vector<uint8_t> out(32, 0);
    evmone::crypto::ripemd160(reinterpret_cast<std::byte*>(out.data() + 12),
        reinterpret_cast<const std::byte*>(input.data()), input.size());
    return make_ok(gas, std::move(out));
}
}  // namespace evm::gpu::precompile
