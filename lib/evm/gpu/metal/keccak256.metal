// Copyright (C) 2026, The evmone Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file keccak256.metal
/// Metal compute shader for parallel Keccak-256 hashing.
///
/// Each thread group processes one hash. Input is a buffer of (offset, length)
/// pairs pointing into a contiguous data buffer. Output is a buffer of 32-byte
/// digests.
///
/// Algorithm: Keccak-256 (Ethereum variant, NOT NIST SHA-3)
///   - State: 5x5 x 64-bit = 1600-bit sponge
///   - Rate:  1088 bits (136 bytes)
///   - Capacity: 512 bits
///   - Rounds: 24
///   - Padding: 0x01 || 0x00...0x00 || 0x80 (Keccak, not SHA-3's 0x06)

#include <metal_stdlib>
using namespace metal;

// -- Round constants ----------------------------------------------------------

constant ulong RC[24] = {
    0x0000000000000001UL, 0x0000000000008082UL,
    0x800000000000808AUL, 0x8000000080008000UL,
    0x000000000000808BUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL,
    0x000000000000008AUL, 0x0000000000000088UL,
    0x0000000080008009UL, 0x000000008000000AUL,
    0x000000008000808BUL, 0x800000000000008BUL,
    0x8000000000008089UL, 0x8000000000008003UL,
    0x8000000000008002UL, 0x8000000000000080UL,
    0x000000000000800AUL, 0x800000008000000AUL,
    0x8000000080008081UL, 0x8000000000008080UL,
    0x0000000080000001UL, 0x8000000080008008UL,
};

// -- Pi-lane destination indices for the rho+pi "moving lane" sequence --------
// Starting from lane 1, each step moves to PI_LANE[i] with rotation RHO[i].

constant int PI_LANE[24] = {
    10,  7, 11, 17, 18,  3,  5, 16,  8, 21, 24,  4,
    15, 23, 19, 13, 12,  2, 20, 14, 22,  9,  6,  1
};

constant int RHO[24] = {
     1,  3,  6, 10, 15, 21, 28, 36, 45, 55,  2, 14,
    27, 41, 56,  8, 25, 43, 62, 18, 39, 61, 20, 44
};

// -- Helpers ------------------------------------------------------------------

inline ulong rotl64(ulong x, int n) {
    return (x << n) | (x >> (64 - n));
}

// -- Keccak-f[1600] permutation -----------------------------------------------

void keccak_f(thread ulong st[25]) {
    for (int round = 0; round < 24; ++round) {

        // Theta
        ulong C[5];
        for (int x = 0; x < 5; ++x)
            C[x] = st[x] ^ st[x + 5] ^ st[x + 10] ^ st[x + 15] ^ st[x + 20];

        for (int x = 0; x < 5; ++x) {
            ulong d = C[(x + 4) % 5] ^ rotl64(C[(x + 1) % 5], 1);
            for (int y = 0; y < 5; ++y)
                st[x + 5 * y] ^= d;
        }

        // Rho + Pi (unrolled moving-lane sequence)
        ulong t = st[1];
        for (int i = 0; i < 24; ++i) {
            ulong tmp = st[PI_LANE[i]];
            st[PI_LANE[i]] = rotl64(t, RHO[i]);
            t = tmp;
        }

        // Chi
        for (int y = 0; y < 5; ++y) {
            ulong row[5];
            for (int x = 0; x < 5; ++x)
                row[x] = st[x + 5 * y];
            for (int x = 0; x < 5; ++x)
                st[x + 5 * y] = row[x] ^ ((~row[(x + 1) % 5]) & row[(x + 2) % 5]);
        }

        // Iota
        st[0] ^= RC[round];
    }
}

// -- Input descriptor ---------------------------------------------------------
// Each hash input is described by an offset into the data buffer and a length.

struct HashInput {
    uint offset;   // byte offset into the data buffer
    uint length;   // number of bytes to hash
};

// -- Kernel -------------------------------------------------------------------
// One thread per hash. No thread-group cooperation needed since each hash is
// independent.

kernel void keccak256_batch(
    device const HashInput* inputs  [[buffer(0)]],
    device const uchar*     data    [[buffer(1)]],
    device uchar*           outputs [[buffer(2)]],
    uint tid                        [[thread_position_in_grid]])
{
    const uint offset = inputs[tid].offset;
    const uint len    = inputs[tid].length;
    const uint rate   = 136;  // 1088 bits / 8

    // Initialize state to zero.
    ulong state[25] = {};

    // Absorb phase: process full rate-sized blocks.
    uint absorbed = 0;
    while (absorbed + rate <= len) {
        // XOR rate bytes into state (as little-endian 64-bit words).
        for (uint w = 0; w < rate / 8; ++w) {
            ulong lane = 0;
            for (uint b = 0; b < 8; ++b)
                lane |= ulong(data[offset + absorbed + w * 8 + b]) << (b * 8);
            state[w] ^= lane;
        }
        keccak_f(state);
        absorbed += rate;
    }

    // Absorb remaining bytes with padding.
    // We build the final padded block in a local buffer.
    uchar padded[136] = {};
    uint remaining = len - absorbed;
    for (uint i = 0; i < remaining; ++i)
        padded[i] = data[offset + absorbed + i];

    // Keccak padding (NOT SHA-3): first pad byte = 0x01, last = 0x80.
    // If remaining == rate-1, both bits land on the same byte (0x81).
    padded[remaining] = 0x01;
    padded[rate - 1] |= 0x80;

    // XOR padded block into state.
    for (uint w = 0; w < rate / 8; ++w) {
        ulong lane = 0;
        for (uint b = 0; b < 8; ++b)
            lane |= ulong(padded[w * 8 + b]) << (b * 8);
        state[w] ^= lane;
    }
    keccak_f(state);

    // Squeeze: extract first 256 bits (32 bytes) from state.
    device uchar* out = outputs + tid * 32;
    for (uint w = 0; w < 4; ++w) {
        ulong lane = state[w];
        for (uint b = 0; b < 8; ++b)
            out[w * 8 + b] = uchar(lane >> (b * 8));
    }
}
