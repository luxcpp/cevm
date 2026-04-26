// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file quasar_cpu_reference.cpp
/// CPU reference for cross-backend determinism (STM-004).
///
/// Reproduces the v0.36 synthetic substrate's commit chain byte-for-byte:
/// identical receipt_hash recipe, identical receipts_root/execution_root
/// accumulator, identical block_hash header format. Block-STM scheduling
/// nondeterminism is collapsed by committing in tx_index order — which
/// is exactly the order drain_validate pops CommitItems from validate_hdr
/// (single-writer, FIFO). The determinism contract is on roots, not on
/// conflict_count or repair_count.

#include "quasar_cpu_reference.hpp"

#include <array>
#include <cstring>
#include <vector>

namespace quasar::gpu::ref {

namespace {

constexpr std::array<uint64_t, 24> kKeccakRC = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL,
};

constexpr std::array<uint32_t, 25> kKeccakRot = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
};

inline uint64_t rotl64(uint64_t x, uint32_t n) {
    return (x << n) | (x >> (64u - n));
}

void keccak_f1600(uint64_t* s) {
    for (uint32_t round = 0; round < 24u; ++round) {
        uint64_t c[5];
        for (uint32_t x = 0; x < 5u; ++x)
            c[x] = s[x] ^ s[x+5] ^ s[x+10] ^ s[x+15] ^ s[x+20];
        uint64_t d[5];
        for (uint32_t x = 0; x < 5u; ++x)
            d[x] = c[(x + 4u) % 5u] ^ rotl64(c[(x + 1u) % 5u], 1);
        for (uint32_t y = 0; y < 25u; y += 5u)
            for (uint32_t x = 0; x < 5u; ++x)
                s[y + x] ^= d[x];
        uint64_t b[25];
        for (uint32_t y = 0; y < 5u; ++y)
            for (uint32_t x = 0; x < 5u; ++x) {
                uint32_t i = x + 5u * y;
                uint32_t j = y + 5u * ((2u * x + 3u * y) % 5u);
                b[j] = rotl64(s[i], kKeccakRot[i]);
            }
        for (uint32_t y = 0; y < 25u; y += 5u) {
            uint64_t t0 = b[y+0], t1 = b[y+1], t2 = b[y+2], t3 = b[y+3], t4 = b[y+4];
            s[y+0] = t0 ^ ((~t1) & t2);
            s[y+1] = t1 ^ ((~t2) & t3);
            s[y+2] = t2 ^ ((~t3) & t4);
            s[y+3] = t3 ^ ((~t4) & t0);
            s[y+4] = t4 ^ ((~t0) & t1);
        }
        s[0] ^= kKeccakRC[round];
    }
}

void keccak256(const uint8_t* data, uint64_t len, uint8_t* out) {
    uint64_t s[25] = {};
    constexpr uint32_t rate = 136;
    uint64_t off = 0;
    while (len - off >= rate) {
        for (uint32_t i = 0; i < rate; ++i) {
            uint32_t lane = i / 8u, sh = (i % 8u) * 8u;
            s[lane] ^= uint64_t(data[off + i]) << sh;
        }
        keccak_f1600(s);
        off += rate;
    }
    uint8_t block[rate] = {};
    uint64_t rem = len - off;
    for (uint64_t i = 0; i < rem; ++i) block[i] = data[off + i];
    block[rem]      ^= 0x01;
    block[rate - 1] ^= 0x80;
    for (uint32_t i = 0; i < rate; ++i) {
        uint32_t lane = i / 8u, sh = (i % 8u) * 8u;
        s[lane] ^= uint64_t(block[i]) << sh;
    }
    keccak_f1600(s);
    for (uint32_t i = 0; i < 32u; ++i) {
        uint32_t lane = i / 8u, sh = (i % 8u) * 8u;
        out[i] = uint8_t((s[lane] >> sh) & 0xFFu);
    }
}

void receipt_hash(uint32_t tx_index, uint32_t origin_lo, uint32_t origin_hi,
                  uint64_t gas_limit, uint64_t gas_used,
                  uint64_t round, uint64_t chain_id, uint8_t* out)
{
    uint8_t leaf[40] = {};
    for (uint32_t k = 0; k < 4u; ++k) leaf[0  + k] = uint8_t((tx_index  >> (k*8)) & 0xFFu);
    for (uint32_t k = 0; k < 4u; ++k) leaf[4  + k] = uint8_t((origin_lo >> (k*8)) & 0xFFu);
    for (uint32_t k = 0; k < 4u; ++k) leaf[8  + k] = uint8_t((origin_hi >> (k*8)) & 0xFFu);
    for (uint32_t k = 0; k < 8u; ++k) leaf[12 + k] = uint8_t((gas_limit >> (k*8)) & 0xFFu);
    for (uint32_t k = 0; k < 4u; ++k) leaf[20 + k] = uint8_t((gas_used  >> (k*8)) & 0xFFu);
    for (uint32_t k = 0; k < 4u; ++k) leaf[24 + k] = uint8_t((round     >> (k*8)) & 0xFFu);
    for (uint32_t k = 0; k < 8u; ++k) leaf[28 + k] = uint8_t((chain_id  >> (k*8)) & 0xFFu);
    leaf[36] = leaf[37] = leaf[38] = leaf[39] = 0;
    keccak256(leaf, 40, out);
}

struct MvccSlotCpu {
    uint64_t key_lo = 0;
    uint64_t key_hi = 0;
    uint32_t version = 0;
    bool     occupied = false;
};

uint32_t mvcc_index_cpu(uint64_t key_lo, uint64_t key_hi, uint32_t mask) {
    uint64_t h = 0xcbf29ce484222325ULL;
    h = (h ^ key_lo) * 0x100000001b3ULL;
    h = (h ^ key_hi) * 0x100000001b3ULL;
    return uint32_t(h) & mask;
}

uint32_t mvcc_locate_cpu(std::vector<MvccSlotCpu>& tab,
                         uint64_t key_lo, uint64_t key_hi)
{
    uint32_t mask = uint32_t(tab.size()) - 1u;
    uint32_t idx  = mvcc_index_cpu(key_lo, key_hi, mask);
    for (uint32_t probe = 0; probe < tab.size(); ++probe) {
        auto& s = tab[idx];
        if (!s.occupied) {
            s.key_lo = key_lo;
            s.key_hi = key_hi;
            s.occupied = true;
            return idx;
        }
        if (s.key_lo == key_lo && s.key_hi == key_hi) return idx;
        idx = (idx + 1u) & mask;
    }
    return 0xFFFFFFFFu;
}

constexpr uint32_t kNeedsExec   = 0x40000000u;
constexpr uint32_t kFlagMask    = 0xC0000000u;
constexpr uint32_t kStatusReturn = 1u;

}  // anonymous namespace

CpuReferenceResult run_reference(const QuasarRoundDescriptor& desc,
                                 std::span<const HostInputTx> txs)
{
    CpuReferenceResult r{};
    r.mode = desc.mode;

    constexpr uint32_t kSlots = 8192u;
    std::vector<MvccSlotCpu> mvcc(kSlots);

    uint8_t receipts_root[32] = {};
    uint8_t execution_root[32] = {};

    for (uint32_t tx_index = 0; tx_index < uint32_t(txs.size()); ++tx_index) {
        const auto& t = txs[tx_index];

        uint32_t origin_lo = uint32_t(t.origin & 0xFFFFFFFFu);
        uint32_t origin_hi = uint32_t(t.origin >> 32);
        origin_hi &= 0x3FFFFFFFu;
        if (t.needs_state) origin_hi |= 0x80000000u;
        if (t.needs_exec)  origin_hi |= 0x40000000u;

        uint64_t gas_used = 21000;
        uint32_t status = kStatusReturn;

        if ((origin_hi & kNeedsExec) != 0u) {
            uint64_t key_lo = uint64_t(origin_lo);
            uint64_t key_hi = uint64_t(origin_hi & ~kFlagMask);
            if (key_lo == 0 && key_hi == 0) key_lo = 1;
            uint32_t idx = mvcc_locate_cpu(mvcc, key_lo, key_hi);
            if (idx != 0xFFFFFFFFu) {
                ++mvcc[idx].version;
            }
        }

        uint8_t digest[32];
        receipt_hash(tx_index, origin_lo, origin_hi,
                     t.gas_limit, gas_used,
                     desc.round, desc.chain_id, digest);

        uint8_t buf[64];
        std::memcpy(buf, receipts_root, 32);
        std::memcpy(buf + 32, digest, 32);
        keccak256(buf, 64, receipts_root);

        uint8_t erbuf[64];
        std::memcpy(erbuf, execution_root, 32);
        for (uint32_t k = 0; k < 4u; ++k) erbuf[32 + k] = uint8_t((tx_index >> (k*8)) & 0xFFu);
        for (uint32_t k = 0; k < 4u; ++k) erbuf[36 + k] = uint8_t((status   >> (k*8)) & 0xFFu);
        for (uint32_t k = 0; k < 8u; ++k) erbuf[40 + k] = uint8_t((gas_used >> (k*8)) & 0xFFu);
        for (uint32_t k = 0; k < 20u; ++k) erbuf[48 + k] = digest[k];
        keccak256(erbuf, 64, execution_root);

        ++r.tx_count;
        r.gas_used += gas_used;
    }

    std::memcpy(r.receipts_root, receipts_root, 32);
    std::memcpy(r.execution_root, execution_root, 32);

    uint8_t header[8 + 4 + 32 * 4] = {};
    uint32_t o = 0;
    for (uint32_t k = 0; k < 8u; ++k) header[o++] = uint8_t((desc.round >> (k*8)) & 0xFFu);
    for (uint32_t k = 0; k < 4u; ++k) header[o++] = uint8_t((desc.mode  >> (k*8)) & 0xFFu);
    for (uint32_t k = 0; k < 32u; ++k) header[o++] = receipts_root[k];
    for (uint32_t k = 0; k < 32u; ++k) header[o++] = execution_root[k];
    for (uint32_t k = 0; k < 32u; ++k) header[o++] = 0;
    for (uint32_t k = 0; k < 32u; ++k) header[o++] = 0;
    keccak256(header, o, r.block_hash);
    std::memcpy(r.mode_root, r.block_hash, 32);

    r.status = 1;
    return r;
}

}  // namespace quasar::gpu::ref
