// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file evm_backend.hpp
/// C++ EVM backend for the ZAP VM plugin.
///
/// Wraps StateDB + processor + in-memory block storage.
/// Implements the VmBackend interface from zap_server.hpp.

#pragma once

#include "zap_server.hpp"

#include <state/state_db.hpp>
#include <state/processor.hpp>

#include <evmc/evmc.hpp>
#include <cstring>
#include <ctime>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

// Forward-declare evmone factory.
extern "C" struct evmc_vm* evmc_create_evmone(void) noexcept;

namespace cevm
{

// ---------------------------------------------------------------------------
// SHA-256 (minimal, for block IDs — no external dep)
// ---------------------------------------------------------------------------

namespace sha256_impl
{
// Minimal SHA-256 implementation for block ID computation.
// Uses the same algorithm as the Rust reference (sha2 crate).

static constexpr uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

inline void hash(const uint8_t* data, size_t len, uint8_t out[32])
{
    uint32_t h0 = 0x6a09e667, h1 = 0xbb67ae85, h2 = 0x3c6ef372, h3 = 0xa54ff53a;
    uint32_t h4 = 0x510e527f, h5 = 0x9b05688c, h6 = 0x1f83d9ab, h7 = 0x5be0cd19;

    // Pad message
    size_t bit_len = len * 8;
    size_t padded_len = ((len + 9 + 63) / 64) * 64;
    std::vector<uint8_t> padded(padded_len, 0);
    std::memcpy(padded.data(), data, len);
    padded[len] = 0x80;
    for (size_t i = 0; i < 8; ++i)
        padded[padded_len - 1 - i] = static_cast<uint8_t>(bit_len >> (i * 8));

    // Process blocks
    for (size_t offset = 0; offset < padded_len; offset += 64)
    {
        uint32_t w[64];
        for (size_t i = 0; i < 16; ++i)
        {
            w[i] = static_cast<uint32_t>(padded[offset + i * 4]) << 24 |
                   static_cast<uint32_t>(padded[offset + i * 4 + 1]) << 16 |
                   static_cast<uint32_t>(padded[offset + i * 4 + 2]) << 8 |
                   static_cast<uint32_t>(padded[offset + i * 4 + 3]);
        }
        for (int i = 16; i < 64; ++i)
        {
            uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        uint32_t a = h0, b = h1, c = h2, d = h3;
        uint32_t e = h4, f = h5, g = h6, h = h7;

        for (int i = 0; i < 64; ++i)
        {
            uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t temp1 = h + S1 + ch + K[i] + w[i];
            uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t temp2 = S0 + maj;

            h = g; g = f; f = e; e = d + temp1;
            d = c; c = b; b = a; a = temp1 + temp2;
        }

        h0 += a; h1 += b; h2 += c; h3 += d;
        h4 += e; h5 += f; h6 += g; h7 += h;
    }

    uint32_t vals[8] = {h0, h1, h2, h3, h4, h5, h6, h7};
    for (int i = 0; i < 8; ++i)
    {
        out[i * 4]     = static_cast<uint8_t>(vals[i] >> 24);
        out[i * 4 + 1] = static_cast<uint8_t>(vals[i] >> 16);
        out[i * 4 + 2] = static_cast<uint8_t>(vals[i] >> 8);
        out[i * 4 + 3] = static_cast<uint8_t>(vals[i]);
    }
}

}  // namespace sha256_impl

// ---------------------------------------------------------------------------
// Block encoding/decoding — matches Rust SimpleBlock exactly
// ---------------------------------------------------------------------------

struct Block
{
    uint8_t parent_id[32] = {};
    uint64_t height = 0;
    int64_t timestamp = 0;
    std::vector<uint8_t> data;

    /// Encode: parent_id(32) || height(8 BE) || timestamp(8 BE) || data
    std::vector<uint8_t> encode() const
    {
        std::vector<uint8_t> out;
        out.reserve(48 + data.size());
        out.insert(out.end(), parent_id, parent_id + 32);
        for (int i = 7; i >= 0; --i)
            out.push_back(static_cast<uint8_t>(height >> (i * 8)));
        auto ts = static_cast<uint64_t>(timestamp);
        for (int i = 7; i >= 0; --i)
            out.push_back(static_cast<uint8_t>(ts >> (i * 8)));
        out.insert(out.end(), data.begin(), data.end());
        return out;
    }

    static bool decode(const std::vector<uint8_t>& bytes, Block& blk)
    {
        if (bytes.size() < 48) return false;
        std::memcpy(blk.parent_id, bytes.data(), 32);
        blk.height = 0;
        for (size_t i = 0; i < 8; ++i)
            blk.height = (blk.height << 8) | bytes[32 + i];
        uint64_t ts = 0;
        for (size_t i = 0; i < 8; ++i)
            ts = (ts << 8) | bytes[40 + i];
        blk.timestamp = static_cast<int64_t>(ts);
        blk.data.assign(bytes.begin() + 48, bytes.end());
        return true;
    }

    /// Compute block ID = SHA-256(encoded).
    static void compute_id(const std::vector<uint8_t>& encoded, uint8_t out[32])
    {
        sha256_impl::hash(encoded.data(), encoded.size(), out);
    }
};

// ---------------------------------------------------------------------------
// Hash key for uint8_t[32] arrays in unordered_map
// ---------------------------------------------------------------------------

struct Hash32
{
    size_t operator()(const std::array<uint8_t, 32>& k) const noexcept
    {
        size_t h = 0;
        for (size_t i = 0; i < 32; i += 8)
        {
            uint64_t v;
            std::memcpy(&v, k.data() + i, 8);
            h ^= v + 0x9e3779b97f4a7c15ULL + (h << 12) + (h >> 4);
        }
        return h;
    }
};

using BlockId = std::array<uint8_t, 32>;

inline BlockId to_block_id(const uint8_t* p)
{
    BlockId id;
    std::memcpy(id.data(), p, 32);
    return id;
}

inline BlockId to_block_id(const std::vector<uint8_t>& v)
{
    BlockId id{};
    if (v.size() >= 32)
        std::memcpy(id.data(), v.data(), 32);
    return id;
}

// ---------------------------------------------------------------------------
// Hex encoding for logging
// ---------------------------------------------------------------------------

inline std::string hex_encode(const uint8_t* data, size_t len)
{
    static const char hex[] = "0123456789abcdef";
    std::string out;
    out.reserve(len * 2);
    for (size_t i = 0; i < len; ++i)
    {
        out.push_back(hex[data[i] >> 4]);
        out.push_back(hex[data[i] & 0xF]);
    }
    return out;
}

// ---------------------------------------------------------------------------
// EvmBackend — VmBackend implementation wrapping the C++ EVM
// ---------------------------------------------------------------------------

class EvmBackend final : public VmBackend
{
public:
    EvmBackend()
    {
        vm_ = evmc_create_evmone();
    }

    ~EvmBackend() override
    {
        if (vm_)
            vm_->destroy(vm_);
    }

    bool initialize(Reader& r, Writer& w) override
    {
        std::lock_guard lock(mu_);

        // Decode InitializeRequest fields.
        uint32_t network_id;
        std::vector<uint8_t> chain_id, node_id, public_key;
        std::vector<uint8_t> x_chain_id, c_chain_id, lux_asset_id;
        std::string chain_data_dir;
        std::vector<uint8_t> genesis_bytes, upgrade_bytes, config_bytes;
        std::string db_server_addr, server_addr;

        if (!r.read_u32(network_id)) return false;
        if (!r.read_bytes(chain_id)) return false;
        if (!r.read_bytes(node_id)) return false;
        if (!r.read_bytes(public_key)) return false;
        if (!r.read_bytes(x_chain_id)) return false;
        if (!r.read_bytes(c_chain_id)) return false;
        if (!r.read_bytes(lux_asset_id)) return false;
        if (!r.read_string(chain_data_dir)) return false;
        if (!r.read_bytes(genesis_bytes)) return false;
        if (!r.read_bytes(upgrade_bytes)) return false;
        if (!r.read_bytes(config_bytes)) return false;
        if (!r.read_string(db_server_addr)) return false;
        if (!r.read_string(server_addr)) return false;

        // Build genesis block.
        Block genesis;
        std::memset(genesis.parent_id, 0, 32);
        genesis.height = 0;
        genesis.timestamp = 0;
        genesis.data = genesis_bytes;

        auto encoded = genesis.encode();
        uint8_t genesis_id_raw[32];
        Block::compute_id(encoded, genesis_id_raw);
        auto genesis_id = to_block_id(genesis_id_raw);

        blocks_[genesis_id] = encoded;
        height_index_[0] = genesis_id;
        last_accepted_ = genesis_id;
        preferred_ = genesis_id;

        // Create fresh state DB.
        state_db_ = evm::state::StateDB{};

        fprintf(stderr, "[cevm] initialized: genesis_id=%s genesis_bytes_len=%zu\n",
                hex_encode(genesis_id.data(), 32).c_str(), genesis_bytes.size());

        // Encode InitializeResponse.
        BlockId zero_id{};
        w.write_bytes(genesis_id.data(), 32);        // last_accepted_id
        w.write_bytes(zero_id.data(), 32);            // last_accepted_parent_id
        w.write_u64(0);                               // height
        w.write_bytes(encoded);                       // bytes
        w.write_i64(0);                               // timestamp

        return true;
    }

    bool set_state(Reader& r) override
    {
        std::lock_guard lock(mu_);
        uint8_t state;
        if (!r.read_u8(state)) return false;
        fprintf(stderr, "[cevm] set_state: %u\n", state);
        return true;
    }

    bool shutdown() override
    {
        fprintf(stderr, "[cevm] shutdown requested\n");
        return true;
    }

    bool build_block(Writer& w) override
    {
        std::lock_guard lock(mu_);

        auto it = blocks_.find(preferred_);
        if (it == blocks_.end()) return false;

        Block parent;
        if (!Block::decode(it->second, parent)) return false;

        auto now = static_cast<int64_t>(std::time(nullptr));

        Block blk;
        std::memcpy(blk.parent_id, preferred_.data(), 32);
        blk.height = parent.height + 1;
        blk.timestamp = now;

        auto encoded = blk.encode();
        uint8_t id_raw[32];
        Block::compute_id(encoded, id_raw);
        auto id = to_block_id(id_raw);

        blocks_[id] = encoded;
        height_index_[blk.height] = id;

        fprintf(stderr, "[cevm] build_block: height=%llu id=%s\n",
                static_cast<unsigned long long>(blk.height),
                hex_encode(id.data(), 32).c_str());

        // Encode BlockResponse.
        encode_block_response(w, id, preferred_, encoded, blk.height, blk.timestamp);
        return true;
    }

    bool parse_block(Reader& r, Writer& w) override
    {
        std::lock_guard lock(mu_);

        std::vector<uint8_t> bytes;
        if (!r.read_bytes(bytes)) return false;

        Block blk;
        if (!Block::decode(bytes, blk)) return false;

        uint8_t id_raw[32];
        Block::compute_id(bytes, id_raw);
        auto id = to_block_id(id_raw);
        auto parent_id = to_block_id(blk.parent_id);

        blocks_[id] = bytes;
        height_index_[blk.height] = id;

        encode_block_response(w, id, parent_id, bytes, blk.height, blk.timestamp);
        return true;
    }

    bool get_block(Reader& r, Writer& w) override
    {
        std::lock_guard lock(mu_);

        std::vector<uint8_t> id_bytes;
        if (!r.read_bytes(id_bytes)) return false;
        if (id_bytes.size() != 32) return false;

        auto id = to_block_id(id_bytes);
        auto it = blocks_.find(id);
        if (it == blocks_.end()) return false;

        Block blk;
        if (!Block::decode(it->second, blk)) return false;
        auto parent_id = to_block_id(blk.parent_id);

        encode_block_response(w, id, parent_id, it->second, blk.height, blk.timestamp);
        return true;
    }

    bool block_verify(Reader& r) override
    {
        std::lock_guard lock(mu_);

        std::vector<uint8_t> bytes;
        if (!r.read_bytes(bytes)) return false;

        // Consume optional p_chain_height.
        bool has_pch;
        if (r.read_bool(has_pch) && has_pch)
        {
            uint64_t dummy;
            r.read_u64(dummy);
        }

        Block blk;
        if (!Block::decode(bytes, blk)) return false;

        // Check parent exists.
        auto parent_id = to_block_id(blk.parent_id);
        auto it = blocks_.find(parent_id);
        if (it == blocks_.end()) return false;

        // Check height.
        Block parent;
        if (!Block::decode(it->second, parent)) return false;
        if (blk.height != parent.height + 1) return false;

        return true;
    }

    bool block_accept(Reader& r) override
    {
        std::lock_guard lock(mu_);

        std::vector<uint8_t> id_bytes;
        if (!r.read_bytes(id_bytes)) return false;
        if (id_bytes.size() != 32) return false;

        auto id = to_block_id(id_bytes);
        if (blocks_.find(id) == blocks_.end()) return false;

        last_accepted_ = id;
        fprintf(stderr, "[cevm] block_accept: id=%s\n",
                hex_encode(id.data(), 32).c_str());
        return true;
    }

    bool block_reject(Reader& r) override
    {
        std::lock_guard lock(mu_);

        std::vector<uint8_t> id_bytes;
        if (!r.read_bytes(id_bytes)) return false;
        if (id_bytes.size() != 32) return false;

        auto id = to_block_id(id_bytes);
        blocks_.erase(id);
        fprintf(stderr, "[cevm] block_reject: id=%s\n",
                hex_encode(id.data(), 32).c_str());
        return true;
    }

    bool set_preference(Reader& r) override
    {
        std::lock_guard lock(mu_);

        std::vector<uint8_t> id_bytes;
        if (!r.read_bytes(id_bytes)) return false;
        if (id_bytes.size() != 32) return false;

        preferred_ = to_block_id(id_bytes);
        return true;
    }

    bool health(Writer& w) override
    {
        // Empty health details = healthy.
        w.write_bytes(nullptr, 0);
        return true;
    }

    bool version(Writer& w) override
    {
        w.write_string("cevm/0.1.0");
        return true;
    }

    bool create_handlers(Writer& w) override
    {
        // Zero handlers.
        w.write_u32(0);
        return true;
    }

    bool wait_for_event(Writer& w) override
    {
        // Sleep briefly, then signal PendingTxs.
        std::this_thread::sleep_for(std::chrono::seconds(1));
        w.write_u8(1);  // PendingTxs
        return true;
    }

private:
    void encode_block_response(Writer& w, const BlockId& id, const BlockId& parent_id,
                               const std::vector<uint8_t>& bytes, uint64_t height,
                               int64_t timestamp)
    {
        w.write_bytes(id.data(), 32);         // id
        w.write_bytes(parent_id.data(), 32);  // parent_id
        w.write_bytes(bytes);                 // bytes
        w.write_u64(height);                  // height
        w.write_i64(timestamp);               // timestamp
        w.write_bool(false);                  // verify_with_context
        w.write_u8(0);                        // err = Unspecified (success)
    }

    std::mutex mu_;

    // Block storage.
    std::unordered_map<BlockId, std::vector<uint8_t>, Hash32> blocks_;
    std::unordered_map<uint64_t, BlockId> height_index_;
    BlockId last_accepted_{};
    BlockId preferred_{};

    // EVM state.
    evm::state::StateDB state_db_;
    evmc_vm* vm_ = nullptr;
};

}  // namespace cevm
