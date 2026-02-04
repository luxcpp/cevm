// Copyright (C) 2026, The evmone Authors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file gpu_state_hasher.hpp
/// State trie Keccak-256 batch hasher using luxcpp/gpu.
///
/// Wraps the lux_gpu_keccak256_batch() C API for use from the EVM parallel
/// engine. Supports CPU and GPU backends transparently via luxcpp/gpu's
/// backend dispatch.

#pragma once

#include <lux/gpu.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace evm::gpu
{

/// Batch Keccak-256 hasher backed by luxcpp/gpu.
/// One instance per execution context. Not thread-safe.
class GpuStateHasher
{
public:
    /// Create a hasher with the specified backend.
    /// Falls back to CPU if the requested backend is unavailable.
    explicit GpuStateHasher(LuxBackend backend = LUX_BACKEND_AUTO)
    {
        gpu_ = lux_gpu_create_with_backend(backend);
    }

    ~GpuStateHasher()
    {
        if (gpu_)
            lux_gpu_destroy(gpu_);
    }

    GpuStateHasher(const GpuStateHasher&) = delete;
    GpuStateHasher& operator=(const GpuStateHasher&) = delete;

    GpuStateHasher(GpuStateHasher&& o) noexcept : gpu_(o.gpu_) { o.gpu_ = nullptr; }
    GpuStateHasher& operator=(GpuStateHasher&& o) noexcept
    {
        if (this != &o)
        {
            if (gpu_)
                lux_gpu_destroy(gpu_);
            gpu_ = o.gpu_;
            o.gpu_ = nullptr;
        }
        return *this;
    }

    /// Returns true if the underlying GPU context was created successfully.
    bool available() const { return gpu_ != nullptr; }

    /// Get the backend name string.
    const char* backend_name() const
    {
        return gpu_ ? lux_gpu_backend_name(gpu_) : "none";
    }

    /// Hash a single input (convenience wrapper).
    /// Returns false on error.
    bool hash(const uint8_t* data, size_t length, uint8_t out[32])
    {
        if (!gpu_)
            return false;
        LuxError err = lux_gpu_keccak256_batch(gpu_, data, out, &length, 1);
        return err == LUX_OK;
    }

    /// Hash a batch of inputs.
    ///
    /// @param data        Concatenated input data.
    /// @param input_lens  Length of each input.
    /// @param num_inputs  Number of inputs.
    /// @param outputs     Output buffer: num_inputs * 32 bytes.
    /// @return            true on success.
    bool batch_hash(const uint8_t* data,
                    const size_t* input_lens,
                    size_t num_inputs,
                    uint8_t* outputs)
    {
        if (!gpu_)
            return false;
        LuxError err = lux_gpu_keccak256_batch(gpu_, data, outputs, input_lens, num_inputs);
        return err == LUX_OK;
    }

    /// Hash a batch of non-contiguous inputs.
    ///
    /// @param inputs      Array of pointers to input data.
    /// @param input_lens  Length of each input.
    /// @param num_inputs  Number of inputs.
    /// @param outputs     Output buffer: num_inputs * 32 bytes.
    /// @return            true on success.
    bool batch_hash_scattered(const uint8_t* const* inputs,
                              const size_t* input_lens,
                              size_t num_inputs,
                              uint8_t* outputs)
    {
        if (!gpu_ || num_inputs == 0)
            return num_inputs == 0;

        // Flatten scattered inputs into a contiguous buffer.
        size_t total = 0;
        for (size_t i = 0; i < num_inputs; ++i)
            total += input_lens[i];

        flat_buf_.resize(total);
        size_t offset = 0;
        for (size_t i = 0; i < num_inputs; ++i)
        {
            if (input_lens[i] > 0)
                std::memcpy(flat_buf_.data() + offset, inputs[i], input_lens[i]);
            offset += input_lens[i];
        }

        return batch_hash(flat_buf_.data(), input_lens, num_inputs, outputs);
    }

private:
    LuxGPU* gpu_ = nullptr;
    std::vector<uint8_t> flat_buf_;  // Reusable buffer for scatter->contiguous
};

}  // namespace evm::gpu
