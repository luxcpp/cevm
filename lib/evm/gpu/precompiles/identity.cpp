// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// 0x04 IDENTITY precompile. Gas = 15 + 3 * ceil(N/32). Output = input.

#include "internal.hpp"

namespace evm::gpu::precompile
{
Result identity_cpu(std::span<const uint8_t> input, uint64_t gas_limit)
{
    using namespace detail;
    const uint64_t gas = cost_per_word(15, 3, input.size());
    if (gas_limit < gas)
        return make_oog();

    return make_ok(gas, std::vector<uint8_t>(input.begin(), input.end()));
}
}  // namespace evm::gpu::precompile
