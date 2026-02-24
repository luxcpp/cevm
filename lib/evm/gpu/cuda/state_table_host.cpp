// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file state_table_host.cpp
/// CUDA Runtime API host wrapper for the GPU-resident state hash tables.
///
/// Holds two persistent device-memory tables (account + storage) and exposes
/// batch insert/lookup plus the state-root reduction pipeline. All kernels
/// live in state_table.cu; this file only manages buffers and dispatches.

#include "state_table_host.hpp"

#include <cuda_runtime.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace evm::gpu::cuda
{

// Defined in state_table.cu.
extern "C" cudaError_t evm_cuda_account_lookup_launch(
    const void*, const void*, void*, void*,
    uint32_t, uint32_t, cudaStream_t);

extern "C" cudaError_t evm_cuda_account_insert_launch(
    const void*, const void*, void*,
    uint32_t, uint32_t, cudaStream_t);

extern "C" cudaError_t evm_cuda_storage_lookup_launch(
    const void*, const void*, void*, void*,
    uint32_t, uint32_t, cudaStream_t);

extern "C" cudaError_t evm_cuda_storage_insert_launch(
    const void*, const void*, void*,
    uint32_t, uint32_t, cudaStream_t);

extern "C" cudaError_t evm_cuda_state_root_compact_launch(
    const void*, void*, void*,
    uint32_t, cudaStream_t);

extern "C" cudaError_t evm_cuda_state_root_sort_launch(
    void*, uint32_t, uint32_t, cudaStream_t);

extern "C" cudaError_t evm_cuda_state_root_hash_launch(
    const void*, void*,
    uint32_t, cudaStream_t);

extern "C" cudaError_t evm_cuda_state_root_reduce_launch(
    void*, uint32_t, cudaStream_t);

namespace
{

inline void cuda_check(cudaError_t err, const char* what)
{
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("CUDA error in ") + what +
                                 ": " + cudaGetErrorString(err));
    }
}

inline bool is_pow2(uint32_t v) { return v != 0 && (v & (v - 1)) == 0; }

}  // namespace

bool state_table_cuda_available()
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return false;
    return count > 0;
}

class StateTableCuda final : public StateTable
{
public:
    StateTableCuda(int device, cudaStream_t stream, std::string name,
                   uint32_t account_capacity, uint32_t storage_capacity)
        : device_(device), stream_(stream), name_(std::move(name)),
          account_capacity_(account_capacity),
          storage_capacity_(storage_capacity)
    {
        cuda_check(cudaSetDevice(device_), "cudaSetDevice");
        cuda_check(cudaMalloc(&d_account_table_,
                              account_capacity_ * sizeof(GpuAccountEntry)),
                   "cudaMalloc account_table");
        cuda_check(cudaMalloc(&d_storage_table_,
                              storage_capacity_ * sizeof(GpuStorageEntry)),
                   "cudaMalloc storage_table");
        clear();
    }

    ~StateTableCuda() override
    {
        if (d_account_table_)   cudaFree(d_account_table_);
        if (d_storage_table_)   cudaFree(d_storage_table_);
        if (d_keys_scratch_)    cudaFree(d_keys_scratch_);
        if (d_data_scratch_)    cudaFree(d_data_scratch_);
        if (d_storage_keys_)    cudaFree(d_storage_keys_);
        if (d_values_scratch_)  cudaFree(d_values_scratch_);
        if (d_results_scratch_) cudaFree(d_results_scratch_);
        if (d_found_scratch_)   cudaFree(d_found_scratch_);
        if (d_compact_buf_)     cudaFree(d_compact_buf_);
        if (d_counter_)         cudaFree(d_counter_);
        if (d_hash_buf_)        cudaFree(d_hash_buf_);
        if (stream_)            cudaStreamDestroy(stream_);
    }

    const char* device_name() const override { return name_.c_str(); }

    void clear() override
    {
        cuda_check(cudaSetDevice(device_), "cudaSetDevice");
        cuda_check(cudaMemsetAsync(d_account_table_, 0,
                                   account_capacity_ * sizeof(GpuAccountEntry),
                                   stream_),
                   "memset account_table");
        cuda_check(cudaMemsetAsync(d_storage_table_, 0,
                                   storage_capacity_ * sizeof(GpuStorageEntry),
                                   stream_),
                   "memset storage_table");
        cuda_check(cudaStreamSynchronize(stream_), "clear sync");
    }

    // -- Account ops ---------------------------------------------------------

    void account_insert(
        const uint8_t*        keys_20bytes,
        const GpuAccountData* values,
        uint32_t              count) override
    {
        if (count == 0) return;
        cuda_check(cudaSetDevice(device_), "cudaSetDevice");

        const size_t keys_bytes = count * 20;
        const size_t data_bytes = count * sizeof(GpuAccountData);

        ensure_capacity(d_keys_scratch_, cap_keys_, keys_bytes);
        ensure_capacity(d_data_scratch_, cap_data_, data_bytes);

        cuda_check(cudaMemcpyAsync(d_keys_scratch_, keys_20bytes, keys_bytes,
                                   cudaMemcpyHostToDevice, stream_),
                   "H2D account keys");
        cuda_check(cudaMemcpyAsync(d_data_scratch_, values, data_bytes,
                                   cudaMemcpyHostToDevice, stream_),
                   "H2D account data");

        cuda_check(evm_cuda_account_insert_launch(
                       d_keys_scratch_, d_data_scratch_, d_account_table_,
                       count, account_capacity_, stream_),
                   "account_insert launch");
        cuda_check(cudaStreamSynchronize(stream_), "account_insert sync");
    }

    void account_lookup(
        const uint8_t*  keys_20bytes,
        GpuAccountData* out_values,
        uint32_t*       out_found_flags,
        uint32_t        count) override
    {
        if (count == 0) return;
        cuda_check(cudaSetDevice(device_), "cudaSetDevice");

        const size_t keys_bytes  = count * 20;
        const size_t data_bytes  = count * sizeof(GpuAccountData);
        const size_t found_bytes = count * sizeof(uint32_t);

        ensure_capacity(d_keys_scratch_,    cap_keys_,    keys_bytes);
        ensure_capacity(d_results_scratch_, cap_results_, data_bytes);
        ensure_capacity(d_found_scratch_,   cap_found_,   found_bytes);

        cuda_check(cudaMemcpyAsync(d_keys_scratch_, keys_20bytes, keys_bytes,
                                   cudaMemcpyHostToDevice, stream_),
                   "H2D account keys");

        cuda_check(evm_cuda_account_lookup_launch(
                       d_keys_scratch_, d_account_table_,
                       d_results_scratch_, d_found_scratch_,
                       count, account_capacity_, stream_),
                   "account_lookup launch");

        cuda_check(cudaMemcpyAsync(out_values, d_results_scratch_, data_bytes,
                                   cudaMemcpyDeviceToHost, stream_),
                   "D2H account values");
        cuda_check(cudaMemcpyAsync(out_found_flags, d_found_scratch_, found_bytes,
                                   cudaMemcpyDeviceToHost, stream_),
                   "D2H account found");
        cuda_check(cudaStreamSynchronize(stream_), "account_lookup sync");
    }

    // -- Storage ops ---------------------------------------------------------

    void storage_insert(
        const GpuStorageKey* keys,
        const uint8_t*       values_32bytes,
        uint32_t             count) override
    {
        if (count == 0) return;
        cuda_check(cudaSetDevice(device_), "cudaSetDevice");

        const size_t keys_bytes   = count * sizeof(GpuStorageKey);
        const size_t values_bytes = count * 32;

        ensure_capacity(d_storage_keys_,   cap_storage_keys_,   keys_bytes);
        ensure_capacity(d_values_scratch_, cap_values_,         values_bytes);

        cuda_check(cudaMemcpyAsync(d_storage_keys_, keys, keys_bytes,
                                   cudaMemcpyHostToDevice, stream_),
                   "H2D storage keys");
        cuda_check(cudaMemcpyAsync(d_values_scratch_, values_32bytes, values_bytes,
                                   cudaMemcpyHostToDevice, stream_),
                   "H2D storage values");

        cuda_check(evm_cuda_storage_insert_launch(
                       d_storage_keys_, d_values_scratch_, d_storage_table_,
                       count, storage_capacity_, stream_),
                   "storage_insert launch");
        cuda_check(cudaStreamSynchronize(stream_), "storage_insert sync");
    }

    void storage_lookup(
        const GpuStorageKey* keys,
        uint8_t*             out_values_32bytes,
        uint32_t*            out_found_flags,
        uint32_t             count) override
    {
        if (count == 0) return;
        cuda_check(cudaSetDevice(device_), "cudaSetDevice");

        const size_t keys_bytes   = count * sizeof(GpuStorageKey);
        const size_t values_bytes = count * 32;
        const size_t found_bytes  = count * sizeof(uint32_t);

        ensure_capacity(d_storage_keys_,    cap_storage_keys_,  keys_bytes);
        ensure_capacity(d_results_scratch_, cap_results_,       values_bytes);
        ensure_capacity(d_found_scratch_,   cap_found_,         found_bytes);

        cuda_check(cudaMemcpyAsync(d_storage_keys_, keys, keys_bytes,
                                   cudaMemcpyHostToDevice, stream_),
                   "H2D storage keys");

        cuda_check(evm_cuda_storage_lookup_launch(
                       d_storage_keys_, d_storage_table_,
                       d_results_scratch_, d_found_scratch_,
                       count, storage_capacity_, stream_),
                   "storage_lookup launch");

        cuda_check(cudaMemcpyAsync(out_values_32bytes, d_results_scratch_, values_bytes,
                                   cudaMemcpyDeviceToHost, stream_),
                   "D2H storage values");
        cuda_check(cudaMemcpyAsync(out_found_flags, d_found_scratch_, found_bytes,
                                   cudaMemcpyDeviceToHost, stream_),
                   "D2H storage found");
        cuda_check(cudaStreamSynchronize(stream_), "storage_lookup sync");
    }

    // -- State root pipeline -------------------------------------------------

    std::vector<uint8_t> compute_state_root() override
    {
        cuda_check(cudaSetDevice(device_), "cudaSetDevice");

        // 1. Compact occupied entries to the front of d_compact_buf_.
        ensure_capacity(d_compact_buf_, cap_compact_,
                        account_capacity_ * sizeof(GpuAccountEntry));
        ensure_capacity(d_counter_, cap_counter_, sizeof(uint32_t));
        cuda_check(cudaMemsetAsync(d_counter_, 0, sizeof(uint32_t), stream_),
                   "memset counter");

        cuda_check(evm_cuda_state_root_compact_launch(
                       d_account_table_, d_compact_buf_, d_counter_,
                       account_capacity_, stream_),
                   "state_root_compact launch");

        uint32_t occupied = 0;
        cuda_check(cudaMemcpyAsync(&occupied, d_counter_, sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost, stream_),
                   "D2H counter");
        cuda_check(cudaStreamSynchronize(stream_), "compact sync");

        if (occupied == 0)
            return std::vector<uint8_t>(32, 0);

        // 2. Bitonic sort the compacted entries by address.
        // Round count up to next power of two for the bitonic network; entries
        // beyond `occupied` will be ignored by the sort kernel itself.
        uint32_t pow2 = 1;
        while (pow2 < occupied) pow2 <<= 1;

        for (uint32_t step = 1; step < pow2; step <<= 1)
        {
            uint32_t dir_mask = step << 1;
            for (uint32_t substep = step; substep > 0; substep >>= 1)
            {
                uint32_t packed = (dir_mask << 16) | (substep & 0xFFFFu);
                cuda_check(evm_cuda_state_root_sort_launch(
                               d_compact_buf_, occupied, packed, stream_),
                           "state_root_sort launch");
            }
        }

        // 3. Pad the compacted buffer to a power of two for clean reduction.
        // We hash entries at indices [occupied, pow2) -> zero (already done by
        // the hash kernel for empty slots, but the compacted buffer slots in
        // that range may hold stale data from the table copy. Zero them.)
        if (pow2 > occupied)
        {
            // Zero out the trailing entries' key_valid flag so the hash kernel
            // sees them as empty and emits zero hashes.
            std::vector<GpuAccountEntry> zeros(pow2 - occupied);
            std::memset(zeros.data(), 0, zeros.size() * sizeof(GpuAccountEntry));
            cuda_check(cudaMemcpyAsync(
                static_cast<uint8_t*>(d_compact_buf_) + occupied * sizeof(GpuAccountEntry),
                zeros.data(),
                zeros.size() * sizeof(GpuAccountEntry),
                cudaMemcpyHostToDevice, stream_),
                "H2D zero pad");
        }

        // 4. Hash each account entry: keccak256(RLP(account)).
        ensure_capacity(d_hash_buf_, cap_hash_, pow2 * 32);
        cuda_check(evm_cuda_state_root_hash_launch(
                       d_compact_buf_, d_hash_buf_, pow2, stream_),
                   "state_root_hash launch");

        // 5. Pairwise reduce: keccak(a || b) -> single 32-byte digest.
        for (uint32_t active = pow2; active > 1; active >>= 1)
        {
            cuda_check(evm_cuda_state_root_reduce_launch(
                           d_hash_buf_, active, stream_),
                       "state_root_reduce launch");
        }

        std::vector<uint8_t> root(32, 0);
        cuda_check(cudaMemcpyAsync(root.data(), d_hash_buf_, 32,
                                   cudaMemcpyDeviceToHost, stream_),
                   "D2H state_root");
        cuda_check(cudaStreamSynchronize(stream_), "compute_state_root sync");
        return root;
    }

private:
    void ensure_capacity(void*& ptr, size_t& cap, size_t needed)
    {
        if (cap >= needed) return;
        if (ptr) cudaFree(ptr);
        cuda_check(cudaMalloc(&ptr, needed), "cudaMalloc");
        cap = needed;
    }

    int          device_ = 0;
    cudaStream_t stream_ = nullptr;
    std::string  name_;

    uint32_t account_capacity_ = DEFAULT_ACCOUNT_CAPACITY;
    uint32_t storage_capacity_ = DEFAULT_STORAGE_CAPACITY;

    void* d_account_table_   = nullptr;
    void* d_storage_table_   = nullptr;

    void*  d_keys_scratch_    = nullptr; size_t cap_keys_         = 0;
    void*  d_data_scratch_    = nullptr; size_t cap_data_         = 0;
    void*  d_storage_keys_    = nullptr; size_t cap_storage_keys_ = 0;
    void*  d_values_scratch_  = nullptr; size_t cap_values_       = 0;
    void*  d_results_scratch_ = nullptr; size_t cap_results_      = 0;
    void*  d_found_scratch_   = nullptr; size_t cap_found_        = 0;
    void*  d_compact_buf_     = nullptr; size_t cap_compact_      = 0;
    void*  d_counter_         = nullptr; size_t cap_counter_      = 0;
    void*  d_hash_buf_        = nullptr; size_t cap_hash_         = 0;
};

std::unique_ptr<StateTable> StateTable::create(
    uint32_t account_capacity, uint32_t storage_capacity)
{
    if (!is_pow2(account_capacity) || !is_pow2(storage_capacity))
        throw std::invalid_argument("StateTable capacities must be powers of two");

    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0)
        return nullptr;

    int device = 0;
    if (cudaSetDevice(device) != cudaSuccess)
        return nullptr;

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess)
        return nullptr;

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess)
        return nullptr;

    return std::make_unique<StateTableCuda>(
        device, stream, std::string(prop.name),
        account_capacity, storage_capacity);
}

}  // namespace evm::gpu::cuda
