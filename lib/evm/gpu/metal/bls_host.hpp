// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file bls_host.hpp
/// C++ host-side interface for GPU-accelerated BLS12-381 batch verification.
///
/// Dispatches bls12_381.metal on Apple Metal for consensus vote verification.
/// Each signature is a compressed G1 point (48 bytes), each public key is a
/// compressed G2 point (96 bytes), each message is a 32-byte pre-hash.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace evm::gpu::metal
{

/// GPU-accelerated BLS12-381 batch verifier.
///
/// Dispatches bls_verify_batch from bls12_381.metal.
/// The Metal kernel handles G1 deserialization and curve checks.
/// Pairing verification is deferred to the host for correctness.
class BlsVerifier
{
public:
    virtual ~BlsVerifier() = default;

    /// Create a BlsVerifier. Returns nullptr if Metal is unavailable.
    static std::unique_ptr<BlsVerifier> create();

    /// Batch-verify BLS signatures on GPU.
    ///
    /// @param sigs     Compressed G1 signatures (48 bytes each).
    /// @param pubkeys  Compressed G2 public keys (96 bytes each).
    /// @param messages Pre-hashed messages (32 bytes each).
    /// @param count    Number of signatures to verify.
    /// @return         Per-signature results: true if G1 deser + curve check passed.
    virtual std::vector<bool> verify_batch(
        const uint8_t* sigs,
        const uint8_t* pubkeys,
        const uint8_t* messages,
        size_t count) = 0;

    /// Get the Metal device name.
    virtual const char* device_name() const = 0;

protected:
    BlsVerifier() = default;
    BlsVerifier(const BlsVerifier&) = delete;
    BlsVerifier& operator=(const BlsVerifier&) = delete;
};

}  // namespace evm::gpu::metal
