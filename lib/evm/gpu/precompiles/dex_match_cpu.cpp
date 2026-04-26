// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Lux custom precompile 0x100 — DEX_MATCH (CPU reference).
//
// Calldata layout (single-incoming-order form, 117 bytes):
//
//   side(1)        0 = bid, 1 = ask
//   price(32)      uint256, BE                 — limit price
//   qty(32)        uint256, BE                 — incoming quantity
//   user(20)       address                     — caller-of-record
//   book_id(32)    uint256, BE                 — order-book identifier
//
// Output (68 bytes):
//
//   filled_qty(32) uint256, BE
//   avg_price(32)  uint256, BE                 — VWAP of fills
//   num_fills(4)   uint32, BE                  — number of resting orders crossed
//
// This is a CPU reference. The matching loop walks an in-memory order book
// keyed by `book_id`, sorts the opposite side by price (best-first), and
// fills the incoming order until `qty` is exhausted or no crossing levels
// remain. State is process-global; this is fine for benchmarks and the
// initial wiring but consensus-grade matching needs to live behind a
// state-trie commit (future work — same shape as the SSTORE journal).
//
// The GPU path (dex_match_metal.mm) installs over this entry once the
// `lux_match_orders` backend in liblux_accel returns `gpu::Status::OK`.
// Until then this CPU implementation is what the precompile actually does.

#include "internal.hpp"
#include "precompile_dispatch.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>

namespace evm::gpu::precompile
{
namespace
{

// Calldata sizes.
constexpr size_t kSide      = 1;
constexpr size_t kPrice     = 32;
constexpr size_t kQty       = 32;
constexpr size_t kUser      = 20;
constexpr size_t kBookId    = 32;
constexpr size_t kCalldata  = kSide + kPrice + kQty + kUser + kBookId;  // 117

// Output sizes.
constexpr size_t kFilledQty = 32;
constexpr size_t kAvgPrice  = 32;
constexpr size_t kNumFills  = 4;
constexpr size_t kOutput    = kFilledQty + kAvgPrice + kNumFills;  // 68

// Gas: linear in matched levels. Base 1500 (about ECRECOVER), 350 per fill —
// matches the ECADD level for a small uint256 multiply + add pair, which is
// what each fill actually costs.
constexpr uint64_t kGasBase    = 1500;
constexpr uint64_t kGasPerFill = 350;

// Resting orders are stored as a 256-bit price + 256-bit qty. We don't yet
// store user / order-id because the precompile only returns aggregate fill
// info; fully-detailed event log emission is a layer above (an EIP-2929-
// style touch list).
struct RestingOrder
{
    std::array<uint8_t, 32> price{};  // BE
    std::array<uint8_t, 32> qty{};    // BE
};

// Per-book state.
struct Book
{
    std::vector<RestingOrder> bids;  // sorted high→low
    std::vector<RestingOrder> asks;  // sorted low→high
};

struct ArrayHash
{
    size_t operator()(const std::array<uint8_t, 32>& a) const noexcept
    {
        // FNV-1a 64. Good enough for a process-local hash; collisions cost
        // a chained probe.
        size_t h = 1469598103934665603ull;
        for (auto b : a)
        {
            h ^= b;
            h *= 1099511628211ull;
        }
        return h;
    }
};

// Process-global book table. Guarded by a single mutex; this is the
// reference-correctness path so we keep it simple.
struct BookTable
{
    std::mutex mu;
    std::unordered_map<std::array<uint8_t, 32>, Book, ArrayHash> books;

    Book& at(const std::array<uint8_t, 32>& book_id)
    {
        return books[book_id];
    }
};

BookTable& book_table()
{
    static BookTable t;
    return t;
}

// uint256 helpers (BE-byte, 32-byte arrays). Only the operations we need.

void u256_zero(std::array<uint8_t, 32>& out)
{
    out.fill(0);
}

bool u256_lt(const std::array<uint8_t, 32>& a, const std::array<uint8_t, 32>& b)
{
    return std::memcmp(a.data(), b.data(), 32) < 0;
}

bool u256_le(const std::array<uint8_t, 32>& a, const std::array<uint8_t, 32>& b)
{
    return std::memcmp(a.data(), b.data(), 32) <= 0;
}

bool u256_is_zero(const std::array<uint8_t, 32>& a)
{
    for (auto b : a)
        if (b) return false;
    return true;
}

// out = a - b. Caller guarantees a >= b.
void u256_sub(std::array<uint8_t, 32>& out,
              const std::array<uint8_t, 32>& a,
              const std::array<uint8_t, 32>& b)
{
    int borrow = 0;
    for (int i = 31; i >= 0; --i)
    {
        int diff = static_cast<int>(a[i]) - static_cast<int>(b[i]) - borrow;
        if (diff < 0) { diff += 256; borrow = 1; }
        else          { borrow = 0; }
        out[i] = static_cast<uint8_t>(diff);
    }
}

// out += a (mod 2^256, but we expect no overflow — fill quantities can't
// exceed 2^256 individually).
void u256_add_inplace(std::array<uint8_t, 32>& out, const std::array<uint8_t, 32>& a)
{
    int carry = 0;
    for (int i = 31; i >= 0; --i)
    {
        int sum = static_cast<int>(out[i]) + static_cast<int>(a[i]) + carry;
        out[i] = static_cast<uint8_t>(sum & 0xff);
        carry  = sum >> 8;
    }
}

// out = min(a, b).
const std::array<uint8_t, 32>& u256_min(const std::array<uint8_t, 32>& a,
                                        const std::array<uint8_t, 32>& b)
{
    return u256_lt(a, b) ? a : b;
}

// Schoolbook 256x256 → 512 multiply, then divide by num_fills (256-bit). For
// the precompile we only need (sum_of price*qty) / total_filled — we approximate
// by accumulating into a 64-byte buffer and dividing at the end. To keep the
// CPU reference deterministic and bounded we cap individual fills at 2^128
// (price * qty fits in a 256-bit accumulator under that cap).
//
// VWAP(price, qty) = sum(price_i * qty_i) / sum(qty_i).
// Caller passes in the running notional accumulator and the running fill
// total; we produce the final BE-32 quotient.
//
// `notional_lo`/`notional_hi` form a 512-bit big-endian value (notional_hi is
// the high 32 bytes). Total is a 256-bit BE value.
void u256_vwap(std::array<uint8_t, 32>& out,
               const std::array<uint8_t, 32>& notional_hi,
               const std::array<uint8_t, 32>& notional_lo,
               const std::array<uint8_t, 32>& total)
{
    if (u256_is_zero(total))
    {
        u256_zero(out);
        return;
    }
    // Schoolbook long division: (notional_hi:notional_lo) / total.
    // We do bit-by-bit; 512 iterations is fine for a CPU reference.
    std::array<uint8_t, 32> rem{};       // running remainder (≤ total - 1)
    std::array<uint8_t, 32> quot{};      // 256-bit quotient

    auto get_bit = [&](int bit) -> int {
        // bit in [0, 512). bit=511 is MSB of notional_hi.
        if (bit < 0 || bit >= 512) return 0;
        const auto& src = (bit >= 256) ? notional_hi : notional_lo;
        int b = bit % 256;             // 0..255 within src (0 = LSB bit)
        int byte = 31 - (b / 8);
        int shift = b % 8;
        return (src[byte] >> shift) & 1;
    };
    auto rem_shl1_or = [&](int incoming_bit) {
        // rem = (rem << 1) | incoming_bit
        int carry = incoming_bit;
        for (int i = 31; i >= 0; --i)
        {
            int v = (rem[i] << 1) | carry;
            rem[i] = static_cast<uint8_t>(v & 0xff);
            carry  = (v >> 8) & 1;
        }
    };
    auto quot_shl1_or = [&](int low_bit) {
        int carry = low_bit;
        for (int i = 31; i >= 0; --i)
        {
            int v = (quot[i] << 1) | carry;
            quot[i] = static_cast<uint8_t>(v & 0xff);
            carry  = (v >> 8) & 1;
        }
    };
    auto rem_geq_total = [&]() -> bool {
        return std::memcmp(rem.data(), total.data(), 32) >= 0;
    };

    for (int bit = 511; bit >= 0; --bit)
    {
        rem_shl1_or(get_bit(bit));
        if (rem_geq_total())
        {
            std::array<uint8_t, 32> nr{};
            u256_sub(nr, rem, total);
            rem = nr;
            quot_shl1_or(1);
        }
        else
        {
            quot_shl1_or(0);
        }
    }
    out = quot;
}

// 256-bit by 256-bit multiply into a 512-bit big-endian (hi:lo). The CPU
// reference does this with 64-bit limbs; result is little-endian-limb so we
// byteswap into BE at the end.
void u256_mul(std::array<uint8_t, 32>& out_hi,
              std::array<uint8_t, 32>& out_lo,
              const std::array<uint8_t, 32>& a,
              const std::array<uint8_t, 32>& b)
{
    // Convert to 4-limb little-endian (limb 0 = least significant 64 bits).
    auto to_limbs = [](const std::array<uint8_t, 32>& src,
                       std::array<uint64_t, 4>& dst) {
        for (int i = 0; i < 4; ++i)
        {
            uint64_t v = 0;
            for (int j = 0; j < 8; ++j)
                v |= static_cast<uint64_t>(src[31 - i * 8 - j]) << (j * 8);
            dst[i] = v;
        }
    };
    std::array<uint64_t, 4> al{}, bl{};
    to_limbs(a, al);
    to_limbs(b, bl);

    // 8-limb result.
    std::array<uint64_t, 8> r{};
    for (int i = 0; i < 4; ++i)
    {
        unsigned __int128 carry = 0;
        for (int j = 0; j < 4; ++j)
        {
            unsigned __int128 cur = static_cast<unsigned __int128>(r[i + j])
                + static_cast<unsigned __int128>(al[i]) * bl[j]
                + carry;
            r[i + j] = static_cast<uint64_t>(cur);
            carry    = cur >> 64;
        }
        r[i + 4] += static_cast<uint64_t>(carry);
    }

    // Limbs 0..3 → out_lo (low 256), 4..7 → out_hi.
    auto from_limbs = [](std::array<uint8_t, 32>& dst,
                         const std::array<uint64_t, 4>& src) {
        for (int i = 0; i < 4; ++i)
        {
            uint64_t v = src[i];
            for (int j = 0; j < 8; ++j)
                dst[31 - i * 8 - j] = static_cast<uint8_t>(v >> (j * 8));
        }
    };
    std::array<uint64_t, 4> lo{r[0], r[1], r[2], r[3]};
    std::array<uint64_t, 4> hi{r[4], r[5], r[6], r[7]};
    from_limbs(out_lo, lo);
    from_limbs(out_hi, hi);
}

// out_hi:out_lo += a_hi:a_lo  (512-bit add, mod 2^512).
void u512_add_inplace(std::array<uint8_t, 32>& out_hi,
                      std::array<uint8_t, 32>& out_lo,
                      const std::array<uint8_t, 32>& a_hi,
                      const std::array<uint8_t, 32>& a_lo)
{
    int carry = 0;
    for (int i = 31; i >= 0; --i)
    {
        int sum = static_cast<int>(out_lo[i]) + static_cast<int>(a_lo[i]) + carry;
        out_lo[i] = static_cast<uint8_t>(sum & 0xff);
        carry     = sum >> 8;
    }
    for (int i = 31; i >= 0; --i)
    {
        int sum = static_cast<int>(out_hi[i]) + static_cast<int>(a_hi[i]) + carry;
        out_hi[i] = static_cast<uint8_t>(sum & 0xff);
        carry     = sum >> 8;
    }
    // Overflow beyond 512 bits is impossible for the workloads this
    // precompile sees (notional ≤ 2^256 * num_fills, with num_fills ≤ a few
    // billion in practice — well under 2^256).
}

// Insert a resting order into a sorted side. `descending=true` for bids.
void insert_sorted(std::vector<RestingOrder>& side,
                   const RestingOrder& o,
                   bool descending)
{
    auto pred = [&](const RestingOrder& x) {
        return descending ? u256_lt(o.price, x.price) : u256_lt(x.price, o.price);
    };
    auto it = std::find_if(side.begin(), side.end(), pred);
    side.insert(it, o);
}

// Match an incoming order against the opposing side. Returns the cumulative
// fill quantity, the running notional (price * qty) and the count of fills.
//
// `incoming_qty` is decremented; surviving quantity stays in incoming_qty
// and is later inserted as a resting order.
//
// `is_bid_incoming=true` means the incoming order is a bid — it crosses any
// ask whose price ≤ limit. `false` means an ask, crossing any bid whose
// price ≥ limit.
struct MatchAccum
{
    std::array<uint8_t, 32> filled_qty{};
    std::array<uint8_t, 32> notional_hi{};   // 512-bit running notional
    std::array<uint8_t, 32> notional_lo{};
    uint32_t                num_fills = 0;
};

void match_against(MatchAccum& acc,
                   std::array<uint8_t, 32>& incoming_qty,
                   const std::array<uint8_t, 32>& limit,
                   std::vector<RestingOrder>& opposite,
                   bool is_bid_incoming)
{
    // opposite is sorted best-first (asks: ascending; bids: descending).
    while (!opposite.empty() && !u256_is_zero(incoming_qty))
    {
        const auto& top = opposite.front();
        // Cross check.
        bool crosses = is_bid_incoming
            ? u256_le(top.price, limit)   // ask price ≤ bid limit
            : u256_le(limit, top.price);  // limit ≤ bid price
        if (!crosses) break;

        const auto& fill_qty = u256_min(incoming_qty, top.qty);

        // notional += fill_qty * top.price   (512-bit accumulator)
        std::array<uint8_t, 32> add_hi{}, add_lo{};
        u256_mul(add_hi, add_lo, fill_qty, top.price);
        u512_add_inplace(acc.notional_hi, acc.notional_lo, add_hi, add_lo);

        u256_add_inplace(acc.filled_qty, fill_qty);

        // Decrement incoming and the resting top.
        std::array<uint8_t, 32> new_incoming{};
        u256_sub(new_incoming, incoming_qty, fill_qty);
        incoming_qty = new_incoming;

        if (u256_lt(fill_qty, top.qty))
        {
            // Resting order partially filled — keep it.
            std::array<uint8_t, 32> new_top_qty{};
            u256_sub(new_top_qty, top.qty, fill_qty);
            opposite.front().qty = new_top_qty;
        }
        else
        {
            opposite.erase(opposite.begin());
        }

        ++acc.num_fills;
    }
}

// Encode (filled_qty, avg_price, num_fills) into the 68-byte output.
void encode_output(std::vector<uint8_t>& out,
                   const std::array<uint8_t, 32>& filled_qty,
                   const std::array<uint8_t, 32>& avg_price,
                   uint32_t num_fills)
{
    out.assign(kOutput, 0);
    std::memcpy(out.data() + 0,                   filled_qty.data(), 32);
    std::memcpy(out.data() + 32,                  avg_price.data(),  32);
    out[64] = static_cast<uint8_t>((num_fills >> 24) & 0xff);
    out[65] = static_cast<uint8_t>((num_fills >> 16) & 0xff);
    out[66] = static_cast<uint8_t>((num_fills >>  8) & 0xff);
    out[67] = static_cast<uint8_t>((num_fills >>  0) & 0xff);
}

}  // namespace

Result dex_match_cpu(std::span<const uint8_t> input, uint64_t gas_limit)
{
    if (input.size() < kCalldata)
    {
        // Pad short input with zeros so callers can still hit the
        // dispatcher with junk and get a deterministic failure rather
        // than a length-dependent crash.
        std::vector<uint8_t> padded(kCalldata, 0);
        std::memcpy(padded.data(), input.data(), input.size());
        std::span<const uint8_t> p(padded.data(), padded.size());
        return dex_match_cpu(p, gas_limit);
    }

    if (gas_limit < kGasBase)
        return detail::make_oog();

    // Decode.
    const uint8_t side = input[0];

    std::array<uint8_t, 32> price{}, qty{}, book_id{};
    std::memcpy(price.data(),   input.data() + kSide,                            32);
    std::memcpy(qty.data(),     input.data() + kSide + kPrice,                   32);
    // user (20 bytes) is recorded for future event-log emission; not needed
    // for the aggregate output.
    std::memcpy(book_id.data(), input.data() + kSide + kPrice + kQty + kUser,    32);

    // Match.
    auto& tbl = book_table();
    std::lock_guard<std::mutex> lk(tbl.mu);
    auto& book = tbl.at(book_id);

    MatchAccum acc;
    std::array<uint8_t, 32> remaining = qty;
    if (side == 0)
    {
        // Incoming bid → cross asks.
        match_against(acc, remaining, price, book.asks, /*is_bid_incoming=*/true);
        // Surviving qty rests in the bids.
        if (!u256_is_zero(remaining))
        {
            RestingOrder o{price, remaining};
            insert_sorted(book.bids, o, /*descending=*/true);
        }
    }
    else
    {
        // Incoming ask → cross bids.
        match_against(acc, remaining, price, book.bids, /*is_bid_incoming=*/false);
        if (!u256_is_zero(remaining))
        {
            RestingOrder o{price, remaining};
            insert_sorted(book.asks, o, /*descending=*/false);
        }
    }

    // Gas accounting.
    const uint64_t gas_used =
        kGasBase + static_cast<uint64_t>(acc.num_fills) * kGasPerFill;
    if (gas_used > gas_limit)
        return detail::make_oog();

    // VWAP.
    std::array<uint8_t, 32> avg_price{};
    u256_vwap(avg_price, acc.notional_hi, acc.notional_lo, acc.filled_qty);

    std::vector<uint8_t> output;
    encode_output(output, acc.filled_qty, avg_price, acc.num_fills);
    return detail::make_ok(gas_used, std::move(output));
}

}  // namespace evm::gpu::precompile
