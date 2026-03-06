// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// 0x01 ECRECOVER precompile (CPU). Gas = 3000.
//
// Input  (128 bytes, zero-padded): hash[32] || v[32 BE] || r[32] || s[32]
// Output (32 bytes): 12 zero bytes + 20-byte recovered Ethereum address,
// or empty bytes on invalid signature.

#include "internal.hpp"

#include <cevm_precompiles/secp256k1.hpp>

#include <intx/intx.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace evm::gpu::precompile
{
Result ecrecover_cpu(std::span<const uint8_t> input, uint64_t gas_limit)
{
    using namespace detail;
    constexpr uint64_t kGas = 3000;
    if (gas_limit < kGas)
        return make_oog();

    uint8_t buf[128]{};
    std::memcpy(buf, input.data(), std::min<size_t>(input.size(), sizeof(buf)));

    const auto v_u256 = intx::be::unsafe::load<intx::uint256>(buf + 32);
    if (v_u256 != 27 && v_u256 != 28)
        return make_failure(kGas);

    const std::span<const uint8_t, 32> hash{buf, 32};
    const std::span<const uint8_t, 32> r{buf + 64, 32};
    const std::span<const uint8_t, 32> s{buf + 96, 32};
    const bool parity = (v_u256 == 28);

    const auto recovered = evmmax::secp256k1::ecrecover(hash, r, s, parity);
    if (!recovered)
        return make_failure(kGas);

    std::vector<uint8_t> out(32, 0);
    std::memcpy(out.data() + 12, recovered->bytes, 20);
    return make_ok(kGas, std::move(out));
}
}  // namespace evm::gpu::precompile
