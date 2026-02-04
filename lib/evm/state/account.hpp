// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file account.hpp
/// Ethereum account representation with RLP serialization.

#pragma once

#include <evmc/evmc.hpp>
#include <evmone_precompiles/keccak.hpp>
#include <intx/intx.hpp>
#include <cstdint>
#include <vector>

namespace evm::state
{

using intx::uint256;

/// Keccak-256 of empty byte sequence. Used as code_hash for accounts without code.
inline evmc::bytes32 empty_code_hash() noexcept
{
    static const auto h = [] {
        const auto k = ethash::keccak256(nullptr, 0);
        evmc::bytes32 r{};
        __builtin_memcpy(r.bytes, k.bytes, 32);
        return r;
    }();
    return h;
}

/// Ethereum account state.
struct Account
{
    uint64_t nonce = 0;
    uint256 balance = 0;
    evmc::bytes32 code_hash = empty_code_hash();
    evmc::bytes32 storage_root{};  ///< Placeholder until MPT.
    std::vector<uint8_t> code;     ///< Deployed bytecode.

    /// True if this account has never been touched (default state).
    [[nodiscard]] bool is_empty() const noexcept
    {
        return nonce == 0 && balance == 0 && code.empty();
    }
};

/// Minimal RLP encoding utilities.
/// Only what is needed for account serialization and state root hashing.
namespace rlp
{

/// Encode a single byte string (raw bytes) as RLP.
inline void encode_bytes(std::vector<uint8_t>& out, const uint8_t* data, size_t len)
{
    if (len == 1 && data[0] < 0x80)
    {
        out.push_back(data[0]);
    }
    else if (len < 56)
    {
        out.push_back(static_cast<uint8_t>(0x80 + len));
        out.insert(out.end(), data, data + len);
    }
    else
    {
        // Encode length of length
        uint8_t len_bytes[8];
        int len_len = 0;
        auto tmp = len;
        while (tmp > 0)
        {
            len_bytes[7 - len_len] = static_cast<uint8_t>(tmp & 0xFF);
            tmp >>= 8;
            ++len_len;
        }
        out.push_back(static_cast<uint8_t>(0xB7 + len_len));
        out.insert(out.end(), len_bytes + (8 - len_len), len_bytes + 8);
        out.insert(out.end(), data, data + len);
    }
}

/// Encode a uint64 as RLP.
inline void encode_uint64(std::vector<uint8_t>& out, uint64_t v)
{
    if (v == 0)
    {
        out.push_back(0x80);  // Empty byte string = integer 0
        return;
    }
    uint8_t buf[8];
    int len = 0;
    auto tmp = v;
    while (tmp > 0)
    {
        buf[7 - len] = static_cast<uint8_t>(tmp & 0xFF);
        tmp >>= 8;
        ++len;
    }
    encode_bytes(out, buf + (8 - len), static_cast<size_t>(len));
}

/// Encode a uint256 as RLP (big-endian, no leading zeros).
inline void encode_uint256(std::vector<uint8_t>& out, const uint256& v)
{
    if (v == 0)
    {
        out.push_back(0x80);
        return;
    }
    const auto be = intx::be::store<evmc::bytes32>(v);
    // Skip leading zeros.
    size_t start = 0;
    while (start < 32 && be.bytes[start] == 0)
        ++start;
    encode_bytes(out, be.bytes + start, 32 - start);
}

/// Encode a bytes32 as RLP (always 32 bytes, no stripping).
inline void encode_bytes32(std::vector<uint8_t>& out, const evmc::bytes32& v)
{
    encode_bytes(out, v.bytes, 32);
}

/// Wrap already-encoded items into an RLP list.
inline void encode_list(std::vector<uint8_t>& out, const std::vector<uint8_t>& payload)
{
    const auto len = payload.size();
    if (len < 56)
    {
        out.push_back(static_cast<uint8_t>(0xC0 + len));
    }
    else
    {
        uint8_t len_bytes[8];
        int len_len = 0;
        auto tmp = len;
        while (tmp > 0)
        {
            len_bytes[7 - len_len] = static_cast<uint8_t>(tmp & 0xFF);
            tmp >>= 8;
            ++len_len;
        }
        out.push_back(static_cast<uint8_t>(0xF7 + len_len));
        out.insert(out.end(), len_bytes + (8 - len_len), len_bytes + 8);
    }
    out.insert(out.end(), payload.begin(), payload.end());
}

}  // namespace rlp

/// RLP-encode an account (nonce, balance, storage_root, code_hash) per Yellow Paper.
inline std::vector<uint8_t> rlp_encode(const Account& acct)
{
    std::vector<uint8_t> payload;
    payload.reserve(128);
    rlp::encode_uint64(payload, acct.nonce);
    rlp::encode_uint256(payload, acct.balance);
    rlp::encode_bytes32(payload, acct.storage_root);
    rlp::encode_bytes32(payload, acct.code_hash);

    std::vector<uint8_t> out;
    out.reserve(payload.size() + 4);
    rlp::encode_list(out, payload);
    return out;
}

}  // namespace evm::state
