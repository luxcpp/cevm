// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// 0x05 MODEXP precompile (CPU). EIP-2565 (Berlin) gas formula.
//
//   gas = max(200, complexity * adjusted_exp_len / 3)
//   complexity = ceil(max(base_len, mod_len) / 8)^2
//   adjusted_exp_len = max(8 * tail_len + bit_length(top32(exp)) - 1, 1)

#include "internal.hpp"

#include <cevm_precompiles/modexp.hpp>

#include <intx/intx.hpp>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>

namespace evm::gpu::precompile
{
namespace
{
struct Header { uint64_t base_len; uint64_t exp_len; uint64_t mod_len; };

std::optional<Header> parse_header(std::span<const uint8_t> input)
{
    uint8_t buf[96]{};
    std::memcpy(buf, input.data(), std::min<size_t>(input.size(), sizeof(buf)));

    const auto b = intx::be::unsafe::load<intx::uint256>(buf);
    const auto e = intx::be::unsafe::load<intx::uint256>(buf + 32);
    const auto m = intx::be::unsafe::load<intx::uint256>(buf + 64);

    constexpr auto kMax = std::numeric_limits<uint32_t>::max();
    if (b > kMax || e > kMax || m > kMax)
        return std::nullopt;

    return Header{static_cast<uint64_t>(b), static_cast<uint64_t>(e),
        static_cast<uint64_t>(m)};
}

uint64_t adjusted_exp_len(
    std::span<const uint8_t> input, uint64_t base_len, uint64_t exp_len) noexcept
{
    const uint64_t head_len = std::min<uint64_t>(exp_len, 32);
    const size_t head_off = 96 + base_len;
    uint8_t head[32]{};
    if (head_off < input.size())
    {
        const size_t avail = input.size() - head_off;
        std::memcpy(head, input.data() + head_off, std::min<size_t>(avail, head_len));
    }

    size_t top = head_len;
    for (size_t i = 0; i < head_len; ++i)
        if (head[i] != 0) { top = i; break; }

    uint64_t bit_width = 0;
    if (top < head_len)
        bit_width = 8 * (head_len - top - 1) +
                    static_cast<uint64_t>(std::bit_width(head[top]));

    const uint64_t tail_len = exp_len - head_len;
    const uint64_t head_bits = (bit_width > 0) ? bit_width - 1 : 0;
    return std::max<uint64_t>(8 * tail_len + head_bits, 1);
}
}  // namespace

Result modexp_cpu(std::span<const uint8_t> input, uint64_t gas_limit)
{
    using namespace detail;
    const auto h = parse_header(input);
    if (!h)
        return make_oog();

    const uint64_t max_len = std::max(h->base_len, h->mod_len);
    const uint64_t words = (max_len + 7) / 8;
    const uint64_t complexity = words * words;
    const uint64_t adj = adjusted_exp_len(input, h->base_len, h->exp_len);
    uint64_t gas = (complexity * adj) / 3;
    if (gas < 200) gas = 200;

    if (gas_limit < gas)
        return make_oog();

    std::vector<uint8_t> out(static_cast<size_t>(h->mod_len), 0);

    if (h->mod_len == 0)
        return make_ok(gas, std::move(out));

    const size_t header = 96;
    std::vector<uint8_t> base(static_cast<size_t>(h->base_len), 0);
    std::vector<uint8_t> exp(static_cast<size_t>(h->exp_len), 0);
    std::vector<uint8_t> mod(static_cast<size_t>(h->mod_len), 0);
    auto copy_seg = [&](size_t off, std::vector<uint8_t>& dst) {
        if (off >= input.size() || dst.empty()) return;
        const size_t avail = input.size() - off;
        std::memcpy(dst.data(), input.data() + off, std::min(avail, dst.size()));
    };
    copy_seg(header, base);
    copy_seg(header + h->base_len, exp);
    copy_seg(header + h->base_len + h->exp_len, mod);

    if (std::all_of(mod.begin(), mod.end(), [](uint8_t b) { return b == 0; }))
        return make_ok(gas, std::move(out));

    cevm::crypto::modexp(base, exp, mod, out.data());
    return make_ok(gas, std::move(out));
}
}  // namespace evm::gpu::precompile
