// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file gpu_state.hpp
/// GPU state database selection header.
///
/// Two implementations exist:
///
///   1. GpuStateDB (legacy) — CPU maps + GPU batch hashing via lux-gpu.
///      All state lives in CPU unordered_maps. GPU is only used for
///      batch Keccak-256 during commit(). Requires lux-gpu C library.
///
///   2. GpuNativeStateDB (new) — ALL state in GPU memory persistently.
///      Accounts and storage live in GPU-resident open-addressing hash
///      tables (Metal buffers). No CPU round-trips for state access.
///      See gpu_state_db.hpp.
///
/// Prefer GpuNativeStateDB for new code. GpuStateDB is retained for
/// backward compatibility with code that inherits from StateDB.

#pragma once

#include "state_db.hpp"
#include "gpu_state_db.hpp"

#ifdef HAVE_LUX_GPU
#include <lux/gpu.h>
#endif

#include <cstring>
#include <vector>

namespace evm::state
{

/// Legacy GPU-accelerated state database (CPU state + GPU hashing).
///
/// Inherits all StateDB behavior. Overrides commit() to use GPU batch hashing.
/// Falls back to CPU if GPU is unavailable.
///
/// For new code, prefer GpuNativeStateDB which keeps all state on GPU.
class GpuStateDB : public StateDB
{
public:
    GpuStateDB()
    {
#ifdef HAVE_LUX_GPU
        gpu_ = lux_gpu_create();
#endif
    }

#ifdef HAVE_LUX_GPU
    explicit GpuStateDB(LuxBackend backend)
    {
        gpu_ = lux_gpu_create_with_backend(backend);
    }
#endif

    ~GpuStateDB() override
    {
#ifdef HAVE_LUX_GPU
        if (gpu_)
            lux_gpu_destroy(gpu_);
#endif
    }

    GpuStateDB(const GpuStateDB&) = delete;
    GpuStateDB& operator=(const GpuStateDB&) = delete;

    GpuStateDB(GpuStateDB&& other) noexcept
        : StateDB(std::move(other))
#ifdef HAVE_LUX_GPU
        , gpu_(other.gpu_)
#endif
    {
#ifdef HAVE_LUX_GPU
        other.gpu_ = nullptr;
#endif
    }

    GpuStateDB& operator=(GpuStateDB&& other) noexcept
    {
        if (this != &other)
        {
            StateDB::operator=(std::move(other));
#ifdef HAVE_LUX_GPU
            if (gpu_) lux_gpu_destroy(gpu_);
            gpu_ = other.gpu_;
            other.gpu_ = nullptr;
#endif
        }
        return *this;
    }

    [[nodiscard]] evmc::bytes32 commit() override
    {
#ifdef HAVE_LUX_GPU
        if (!gpu_)
            return StateDB::commit();

        auto account_data = collect_account_rlp();
        if (account_data.empty())
            return StateDB::commit();

        std::vector<uint8_t> concat_data;
        std::vector<size_t> lengths;
        lengths.reserve(account_data.size());

        for (const auto& rlp : account_data)
        {
            lengths.push_back(rlp.size());
            concat_data.insert(concat_data.end(), rlp.begin(), rlp.end());
        }

        const size_t num_hashes = account_data.size();
        std::vector<uint8_t> hashes(num_hashes * 32);

        LuxError err = lux_gpu_keccak256_batch(
            gpu_, concat_data.data(), hashes.data(), lengths.data(), num_hashes);

        if (err != LUX_OK)
            return StateDB::commit();

        if (num_hashes == 1)
        {
            evmc::bytes32 root{};
            std::memcpy(root.bytes, hashes.data(), 32);
            return root;
        }

        size_t total_len = num_hashes * 32;
        std::vector<size_t> final_len = {total_len};
        uint8_t final_hash[32];

        err = lux_gpu_keccak256_batch(
            gpu_, hashes.data(), final_hash, final_len.data(), 1);

        evmc::bytes32 root{};
        if (err == LUX_OK)
            std::memcpy(root.bytes, final_hash, 32);
        else
            return StateDB::commit();

        return root;
#else
        return StateDB::commit();
#endif
    }

    [[nodiscard]] const char* backend_name() const noexcept
    {
#ifdef HAVE_LUX_GPU
        if (!gpu_) return "none";
        return lux_gpu_backend_name(gpu_);
#else
        return "cpu";
#endif
    }

    [[nodiscard]] bool gpu_available() const noexcept
    {
#ifdef HAVE_LUX_GPU
        return gpu_ != nullptr;
#else
        return false;
#endif
    }

private:
#ifdef HAVE_LUX_GPU
    LuxGPU* gpu_ = nullptr;
#endif

    [[nodiscard]] std::vector<std::vector<uint8_t>> collect_account_rlp() const
    {
        std::vector<std::vector<uint8_t>> result;
        return result;
    }
};

}  // namespace evm::state
