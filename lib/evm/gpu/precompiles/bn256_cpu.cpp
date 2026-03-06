// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// 0x06 BN256_ADD     — gas 150
// 0x07 BN256_MUL     — gas 6000
// 0x08 BN256_PAIRING — gas 45000 + 34000*k
//
// (Istanbul+ pricing.) All implemented via the evmmax::bn254 primitives in
// lib/cevm_precompiles. CPU is the only consensus-safe path here: the
// pairing kernel in luxcpp/gpu does not currently implement BN254 pairings
// (only BLS12-381).

#include "internal.hpp"

#include <cevm_precompiles/bn254.hpp>
#include <cevm_precompiles/ecc.hpp>

#include <intx/intx.hpp>

#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace evm::gpu::precompile
{
namespace
{
using evmmax::bn254::AffinePoint;
using evmmax::bn254::Point;
using evmmax::bn254::ExtPoint;
using evmmax::bn254::validate;
using evmmax::bn254::mul;
using evmmax::bn254::pairing_check;
}  // namespace

Result bn256_add_cpu(std::span<const uint8_t> input, uint64_t gas_limit)
{
    using namespace detail;
    constexpr uint64_t kGas = 150;
    if (gas_limit < kGas)
        return make_oog();

    uint8_t buf[128]{};
    std::memcpy(buf, input.data(), std::min<size_t>(input.size(), sizeof(buf)));

    const std::span<const uint8_t, 64> a_bytes{buf, 64};
    const std::span<const uint8_t, 64> b_bytes{buf + 64, 64};
    const auto p = AffinePoint::from_bytes(a_bytes);
    const auto q = AffinePoint::from_bytes(b_bytes);
    if (!p || !q || !validate(*p) || !validate(*q))
        return make_failure(kGas);

    const auto r = evmmax::ecc::add_affine(*p, *q);
    std::vector<uint8_t> out(64);
    std::span<uint8_t, 64> out_span{out.data(), 64};
    r.to_bytes(out_span);
    return make_ok(kGas, std::move(out));
}

Result bn256_mul_cpu(std::span<const uint8_t> input, uint64_t gas_limit)
{
    using namespace detail;
    constexpr uint64_t kGas = 6000;
    if (gas_limit < kGas)
        return make_oog();

    uint8_t buf[96]{};
    std::memcpy(buf, input.data(), std::min<size_t>(input.size(), sizeof(buf)));

    const std::span<const uint8_t, 64> a_bytes{buf, 64};
    const auto p = AffinePoint::from_bytes(a_bytes);
    if (!p || !validate(*p))
        return make_failure(kGas);

    const auto c = intx::be::unsafe::load<intx::uint256>(buf + 64);
    const auto r = mul(*p, c);
    std::vector<uint8_t> out(64);
    std::span<uint8_t, 64> out_span{out.data(), 64};
    r.to_bytes(out_span);
    return make_ok(kGas, std::move(out));
}

Result bn256_pairing_cpu(std::span<const uint8_t> input, uint64_t gas_limit)
{
    using namespace detail;
    constexpr size_t kPairSize = 192;
    if (input.size() % kPairSize != 0)
        return make_oog();

    const uint64_t k = input.size() / kPairSize;
    const uint64_t gas = 45000 + 34000 * k;
    if (gas_limit < gas)
        return make_oog();

    std::vector<std::pair<Point, ExtPoint>> pairs;
    pairs.reserve(k);
    for (size_t i = 0; i < k; ++i)
    {
        const uint8_t* p = input.data() + i * kPairSize;
        const Point pt{
            intx::be::unsafe::load<intx::uint256>(p),
            intx::be::unsafe::load<intx::uint256>(p + 32),
        };
        const ExtPoint q{
            {intx::be::unsafe::load<intx::uint256>(p + 96),
                intx::be::unsafe::load<intx::uint256>(p + 64)},
            {intx::be::unsafe::load<intx::uint256>(p + 160),
                intx::be::unsafe::load<intx::uint256>(p + 128)},
        };
        pairs.emplace_back(pt, q);
    }

    const auto res = pairing_check(pairs);
    if (!res)
        return make_failure(gas);

    std::vector<uint8_t> out(32, 0);
    out[31] = *res ? 1 : 0;
    return make_ok(gas, std::move(out));
}
}  // namespace evm::gpu::precompile
