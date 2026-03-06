// Copyright (C) 2026, The cevm Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file keccak_host.hpp
/// C++ host-side interface for Metal-accelerated Keccak-256 hashing.
///
/// Usage:
///   auto hasher = evm::gpu::metal::KeccakHasher::create();
///   auto digests = hasher->batch_hash(inputs, num_inputs);
///
/// Each input is a (pointer, length) pair. The output is a flat vector of
/// 32-byte digests, one per input.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace evm::gpu::metal
{

/// A single hash input: pointer to data and its length in bytes.
struct HashInput
{
    const uint8_t* data;
    uint32_t length;
};

/// GPU-accelerated Keccak-256 batch hasher using Apple Metal.
///
/// Compiles the Metal shader on first use and caches the pipeline state.
/// Thread-safe for batch_hash calls from a single thread. Create one instance
/// per thread if concurrent hashing is needed.
class KeccakHasher
{
public:
    virtual ~KeccakHasher() = default;

    /// Create a KeccakHasher. Returns nullptr if Metal is unavailable.
    static std::unique_ptr<KeccakHasher> create();

    /// Hash a batch of inputs in parallel on the GPU.
    ///
    /// @param inputs     Array of (data, length) pairs.
    /// @param num_inputs Number of inputs to hash.
    /// @return           Flat vector of 32 * num_inputs bytes. Each consecutive
    ///                   32-byte slice is the Keccak-256 digest of the
    ///                   corresponding input.
    virtual std::vector<uint8_t> batch_hash(const HashInput* inputs, size_t num_inputs) = 0;

    /// Convenience overload for span.
    std::vector<uint8_t> batch_hash(std::span<const HashInput> inputs)
    {
        return batch_hash(inputs.data(), inputs.size());
    }

    /// Get the Metal device name (for diagnostics).
    virtual const char* device_name() const = 0;

protected:
    KeccakHasher() = default;
    KeccakHasher(const KeccakHasher&) = delete;
    KeccakHasher& operator=(const KeccakHasher&) = delete;
};

/// CPU reference implementation of Keccak-256 (single input).
/// Used for correctness verification and benchmark comparison.
void keccak256_cpu(const uint8_t* data, size_t length, uint8_t out[32]);

}  // namespace evm::gpu::metal
