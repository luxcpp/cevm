// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// 0x09 BLAKE2F precompile (CPU). Gas = rounds (1 per round).
// Input = 213 bytes: rounds[4 BE] || h[64] || m[128] || t[16] || f[1].

#include "internal.hpp"

#include <evmone_precompiles/blake2b.hpp>

#include <intx/intx.hpp>

#include <cstdint>
#include <cstring>

namespace evm::gpu::precompile
{
Result blake2f_cpu(std::span<const uint8_t> input, uint64_t gas_limit)
{
    using namespace detail;
    if (input.size() != 213)
        return make_oog();

    const uint32_t rounds = intx::be::unsafe::load<uint32_t>(input.data());
    const uint64_t gas = rounds;
    if (gas_limit < gas)
        return make_oog();

    const uint8_t* p = input.data() + 4;
    uint64_t h[8];
    std::memcpy(h, p, sizeof(h));
    p += sizeof(h);
    uint64_t m[16];
    std::memcpy(m, p, sizeof(m));
    p += sizeof(m);
    uint64_t t[2];
    std::memcpy(t, p, sizeof(t));
    p += sizeof(t);
    const uint8_t f = *p;
    if (f != 0 && f != 1)
        return make_failure(gas);

    evmone::crypto::blake2b_compress(rounds, h, m, t, f != 0);

    std::vector<uint8_t> out(64);
    std::memcpy(out.data(), h, sizeof(h));
    return make_ok(gas, std::move(out));
}
}  // namespace evm::gpu::precompile
