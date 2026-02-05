// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file gpu_hashtable.mm
/// Objective-C++ implementation of the GPU-resident hash table.
///
/// Compiles and dispatches Metal compute kernels from state_table.metal.
/// All state data lives in persistent Metal buffers across blocks.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "gpu_hashtable.hpp"

#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace evm::state
{

// -- Metal shader struct mirrors (must match state_table.metal) ----------------

struct alignas(4) MetalBatchParams
{
    uint32_t count;
    uint32_t capacity;
};
static_assert(sizeof(MetalBatchParams) == 8);

// AccountEntry: key[20] + valid(4) + _pad(4) + data(104) = 132.
// GpuAccountData is alignas(8), so offset of data within this struct
// must be 8-aligned. 20+4+4 = 28, needs 4 bytes padding -> 32.
// Total = 32 + 104 = 136.
struct MetalAccountEntry
{
    uint8_t        key[20];
    uint32_t       key_valid;
    uint32_t       _pad;
    GpuAccountData data;
};
static_assert(sizeof(MetalAccountEntry) == 136);

// StorageEntry: addr[20] + slot[32] + valid(4) + _pad(4) + value[32] = 92.
struct MetalStorageEntry
{
    uint8_t  key_addr[20];
    uint8_t  key_slot[32];
    uint32_t key_valid;
    uint32_t _pad;
    uint8_t  value[32];
};
static_assert(sizeof(MetalStorageEntry) == 92);

// -- Metal library loader -----------------------------------------------------

static id<MTLLibrary> load_state_table_library(id<MTLDevice> device)
{
    NSError* error = nil;

    // Try pre-compiled metallib.
    NSBundle* bundle = [NSBundle mainBundle];
    NSString* libPath = [bundle pathForResource:@"state_table" ofType:@"metallib"];
    if (libPath)
    {
        NSURL* url = [NSURL fileURLWithPath:libPath];
        id<MTLLibrary> lib = [device newLibraryWithURL:url error:&error];
        if (lib) return lib;
    }

    // Runtime compilation from source.
    std::filesystem::path src(__FILE__);
    std::filesystem::path metal_path = src.parent_path().parent_path() / "gpu" / "metal" / "state_table.metal";

    if (!std::filesystem::exists(metal_path))
    {
        // Try relative to CWD.
        metal_path = std::filesystem::current_path() / "lib" / "evm" / "gpu" / "metal" / "state_table.metal";
    }

    NSString* path = [NSString stringWithUTF8String:metal_path.c_str()];
    NSString* source = [NSString stringWithContentsOfFile:path
                                 encoding:NSUTF8StringEncoding
                                 error:&error];
    if (!source)
        return nil;

    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    opts.mathMode = MTLMathModeFast;
    opts.languageVersion = MTLLanguageVersion3_0;

    id<MTLLibrary> lib = [device newLibraryWithSource:source options:opts error:&error];
    return lib;
}

// -- Metal implementation -----------------------------------------------------

class GpuHashTableMetal final : public GpuHashTable
{
public:
    GpuHashTableMetal(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLBuffer> account_table,
        id<MTLBuffer> storage_table,
        uint32_t account_capacity,
        uint32_t storage_capacity,
        bool owns_buffers,
        id<MTLComputePipelineState> account_lookup_pipeline,
        id<MTLComputePipelineState> account_insert_pipeline,
        id<MTLComputePipelineState> storage_lookup_pipeline,
        id<MTLComputePipelineState> storage_insert_pipeline,
        id<MTLComputePipelineState> state_root_hash_pipeline,
        id<MTLComputePipelineState> state_root_reduce_pipeline,
        id<MTLComputePipelineState> state_root_compact_pipeline,
        id<MTLComputePipelineState> state_root_sort_pipeline)
        : device_(device)
        , queue_(queue)
        , account_table_(account_table)
        , storage_table_(storage_table)
        , account_capacity_(account_capacity)
        , storage_capacity_(storage_capacity)
        , owns_buffers_(owns_buffers)
        , account_lookup_pso_(account_lookup_pipeline)
        , account_insert_pso_(account_insert_pipeline)
        , storage_lookup_pso_(storage_lookup_pipeline)
        , storage_insert_pso_(storage_insert_pipeline)
        , state_root_hash_pso_(state_root_hash_pipeline)
        , state_root_reduce_pso_(state_root_reduce_pipeline)
        , state_root_compact_pso_(state_root_compact_pipeline)
        , state_root_sort_pso_(state_root_sort_pipeline)
    {}

    ~GpuHashTableMetal() override = default;

    // -- Account operations ---

    void lookup_accounts(
        const evmc::address* addrs, uint32_t n,
        GpuAccountData* results, uint32_t* found) override
    {
        if (n == 0) return;

        @autoreleasepool {
            // Keys buffer: N * 20 bytes
            id<MTLBuffer> keys_buf = [device_ newBufferWithBytes:addrs
                                              length:n * 20
                                              options:MTLResourceStorageModeShared];
            id<MTLBuffer> results_buf = [device_ newBufferWithLength:n * sizeof(GpuAccountData)
                                                  options:MTLResourceStorageModeShared];
            id<MTLBuffer> found_buf = [device_ newBufferWithLength:n * sizeof(uint32_t)
                                               options:MTLResourceStorageModeShared];

            MetalBatchParams p{n, account_capacity_};
            id<MTLBuffer> params_buf = [device_ newBufferWithBytes:&p
                                                length:sizeof(p)
                                                options:MTLResourceStorageModeShared];

            id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:account_lookup_pso_];
            [enc setBuffer:keys_buf     offset:0 atIndex:0];
            [enc setBuffer:account_table_ offset:0 atIndex:1];
            [enc setBuffer:results_buf  offset:0 atIndex:2];
            [enc setBuffer:found_buf    offset:0 atIndex:3];
            [enc setBuffer:params_buf   offset:0 atIndex:4];

            NSUInteger tpg = account_lookup_pso_.maxTotalThreadsPerThreadgroup;
            if (tpg > n) tpg = n;
            [enc dispatchThreads:MTLSizeMake(n, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];

            std::memcpy(results, [results_buf contents], n * sizeof(GpuAccountData));
            std::memcpy(found, [found_buf contents], n * sizeof(uint32_t));
        }
    }

    void insert_accounts(
        const evmc::address* addrs, const GpuAccountData* data, uint32_t n) override
    {
        if (n == 0) return;

        @autoreleasepool {
            id<MTLBuffer> keys_buf = [device_ newBufferWithBytes:addrs
                                              length:n * 20
                                              options:MTLResourceStorageModeShared];
            id<MTLBuffer> data_buf = [device_ newBufferWithBytes:data
                                              length:n * sizeof(GpuAccountData)
                                              options:MTLResourceStorageModeShared];

            MetalBatchParams p{n, account_capacity_};
            id<MTLBuffer> params_buf = [device_ newBufferWithBytes:&p
                                                length:sizeof(p)
                                                options:MTLResourceStorageModeShared];

            id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:account_insert_pso_];
            [enc setBuffer:keys_buf       offset:0 atIndex:0];
            [enc setBuffer:data_buf       offset:0 atIndex:1];
            [enc setBuffer:account_table_ offset:0 atIndex:2];
            [enc setBuffer:params_buf     offset:0 atIndex:3];

            NSUInteger tpg = account_insert_pso_.maxTotalThreadsPerThreadgroup;
            if (tpg > n) tpg = n;
            [enc dispatchThreads:MTLSizeMake(n, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];

            account_count_ += n;  // Approximate; exact count tracked by occupancy.
        }
    }

    // -- Storage operations ---

    void lookup_storage(
        const GpuStorageKey* keys, uint32_t n,
        evmc::bytes32* values, uint32_t* found) override
    {
        if (n == 0) return;

        @autoreleasepool {
            id<MTLBuffer> keys_buf = [device_ newBufferWithBytes:keys
                                              length:n * sizeof(GpuStorageKey)
                                              options:MTLResourceStorageModeShared];
            id<MTLBuffer> results_buf = [device_ newBufferWithLength:n * 32
                                                  options:MTLResourceStorageModeShared];
            id<MTLBuffer> found_buf = [device_ newBufferWithLength:n * sizeof(uint32_t)
                                               options:MTLResourceStorageModeShared];

            MetalBatchParams p{n, storage_capacity_};
            id<MTLBuffer> params_buf = [device_ newBufferWithBytes:&p
                                                length:sizeof(p)
                                                options:MTLResourceStorageModeShared];

            id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:storage_lookup_pso_];
            [enc setBuffer:keys_buf       offset:0 atIndex:0];
            [enc setBuffer:storage_table_ offset:0 atIndex:1];
            [enc setBuffer:results_buf    offset:0 atIndex:2];
            [enc setBuffer:found_buf      offset:0 atIndex:3];
            [enc setBuffer:params_buf     offset:0 atIndex:4];

            NSUInteger tpg = storage_lookup_pso_.maxTotalThreadsPerThreadgroup;
            if (tpg > n) tpg = n;
            [enc dispatchThreads:MTLSizeMake(n, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];

            std::memcpy(values, [results_buf contents], n * 32);
            std::memcpy(found, [found_buf contents], n * sizeof(uint32_t));
        }
    }

    void insert_storage(
        const GpuStorageKey* keys, const evmc::bytes32* values, uint32_t n) override
    {
        if (n == 0) return;

        @autoreleasepool {
            id<MTLBuffer> keys_buf = [device_ newBufferWithBytes:keys
                                              length:n * sizeof(GpuStorageKey)
                                              options:MTLResourceStorageModeShared];
            id<MTLBuffer> vals_buf = [device_ newBufferWithBytes:values
                                              length:n * 32
                                              options:MTLResourceStorageModeShared];

            MetalBatchParams p{n, storage_capacity_};
            id<MTLBuffer> params_buf = [device_ newBufferWithBytes:&p
                                                length:sizeof(p)
                                                options:MTLResourceStorageModeShared];

            id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:storage_insert_pso_];
            [enc setBuffer:keys_buf       offset:0 atIndex:0];
            [enc setBuffer:vals_buf       offset:0 atIndex:1];
            [enc setBuffer:storage_table_ offset:0 atIndex:2];
            [enc setBuffer:params_buf     offset:0 atIndex:3];

            NSUInteger tpg = storage_insert_pso_.maxTotalThreadsPerThreadgroup;
            if (tpg > n) tpg = n;
            [enc dispatchThreads:MTLSizeMake(n, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];

            storage_count_ += n;
        }
    }

    // -- State root ---

    evmc::bytes32 compute_state_root() override
    {
        @autoreleasepool {
            // Phase 0: Compact occupied entries to a contiguous buffer
            // so we can sort them by address for deterministic ordering.
            size_t compact_buf_size = account_capacity_ * sizeof(MetalAccountEntry);
            id<MTLBuffer> compact_buf = [device_ newBufferWithLength:compact_buf_size
                                                   options:MTLResourceStorageModeShared];
            id<MTLBuffer> counter_buf = [device_ newBufferWithLength:sizeof(uint32_t)
                                                   options:MTLResourceStorageModeShared];
            std::memset([counter_buf contents], 0, sizeof(uint32_t));

            MetalBatchParams pc{0, account_capacity_};
            id<MTLBuffer> compact_params = [device_ newBufferWithBytes:&pc
                                                     length:sizeof(pc)
                                                     options:MTLResourceStorageModeShared];
            {
                id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:state_root_compact_pso_];
                [enc setBuffer:account_table_ offset:0 atIndex:0];
                [enc setBuffer:compact_buf    offset:0 atIndex:1];
                [enc setBuffer:counter_buf    offset:0 atIndex:2];
                [enc setBuffer:compact_params offset:0 atIndex:3];

                NSUInteger tpg = state_root_compact_pso_.maxTotalThreadsPerThreadgroup;
                NSUInteger n = account_capacity_;
                if (tpg > n) tpg = n;
                [enc dispatchThreads:MTLSizeMake(n, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
                [enc endEncoding];
                [cmd commit];
                [cmd waitUntilCompleted];
            }

            uint32_t num_occupied = 0;
            std::memcpy(&num_occupied, [counter_buf contents], sizeof(uint32_t));

            // Phase 0b: Bitonic sort the compacted entries by address.
            // This ensures deterministic hash ordering across all nodes.
            if (num_occupied > 1) {
                for (uint32_t step = 1; step < num_occupied; step <<= 1) {
                    for (uint32_t substep = step; substep > 0; substep >>= 1) {
                        uint32_t dir_mask = step << 1;
                        uint32_t encoded = (dir_mask << 16) | (substep & 0xFFFF);
                        MetalBatchParams ps{num_occupied, encoded};
                        id<MTLBuffer> sort_params = [device_ newBufferWithBytes:&ps
                                                               length:sizeof(ps)
                                                               options:MTLResourceStorageModeShared];

                        id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
                        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                        [enc setComputePipelineState:state_root_sort_pso_];
                        [enc setBuffer:compact_buf offset:0 atIndex:0];
                        [enc setBuffer:sort_params offset:0 atIndex:1];

                        NSUInteger tpg = state_root_sort_pso_.maxTotalThreadsPerThreadgroup;
                        if (tpg > num_occupied) tpg = num_occupied;
                        [enc dispatchThreads:MTLSizeMake(num_occupied, 1, 1)
                             threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
                        [enc endEncoding];
                        [cmd commit];
                        [cmd waitUntilCompleted];
                    }
                }
            }

            // Phase 1: Hash each sorted entry (use compact_buf, not original table).
            size_t hash_buf_size = (num_occupied > 0 ? num_occupied : 1) * 32;
            id<MTLBuffer> hash_buf = [device_ newBufferWithLength:hash_buf_size
                                               options:MTLResourceStorageModeShared];

            if (num_occupied == 0) {
                // No accounts: return zero hash.
                evmc::bytes32 root{};
                return root;
            }

            MetalBatchParams p1{0, num_occupied};
            id<MTLBuffer> params1 = [device_ newBufferWithBytes:&p1
                                              length:sizeof(p1)
                                              options:MTLResourceStorageModeShared];

            {
                id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:state_root_hash_pso_];
                [enc setBuffer:compact_buf offset:0 atIndex:0];
                [enc setBuffer:hash_buf    offset:0 atIndex:1];
                [enc setBuffer:params1     offset:0 atIndex:2];

                NSUInteger tpg = state_root_hash_pso_.maxTotalThreadsPerThreadgroup;
                NSUInteger n = num_occupied;
                if (tpg > n) tpg = n;
                [enc dispatchThreads:MTLSizeMake(n, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
                [enc endEncoding];
                [cmd commit];
                [cmd waitUntilCompleted];
            }

            // Phase 2: Parallel reduce via pairwise keccak256.
            uint32_t active = num_occupied;
            while (active > 1) {
                MetalBatchParams p2{active, 0};
                id<MTLBuffer> params2 = [device_ newBufferWithBytes:&p2
                                                  length:sizeof(p2)
                                                  options:MTLResourceStorageModeShared];

                id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:state_root_reduce_pso_];
                [enc setBuffer:hash_buf offset:0 atIndex:0];
                [enc setBuffer:params2  offset:0 atIndex:1];

                uint32_t pairs = active / 2;
                NSUInteger tpg = state_root_reduce_pso_.maxTotalThreadsPerThreadgroup;
                if (tpg > pairs) tpg = pairs;
                if (tpg == 0) tpg = 1;
                [enc dispatchThreads:MTLSizeMake(pairs, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
                [enc endEncoding];
                [cmd commit];
                [cmd waitUntilCompleted];

                active = (active + 1) / 2;
            }

            // Read final hash from hash_buf[0].
            evmc::bytes32 root{};
            std::memcpy(root.bytes, [hash_buf contents], 32);
            return root;
        }
    }

    // -- Accessors ---

    void* table_contents() const noexcept override
    {
        return [account_table_ contents];
    }

    void* metal_buffer() const noexcept override
    {
        return (__bridge void*)account_table_;
    }

    uint32_t capacity() const noexcept override { return account_capacity_; }
    uint32_t count() const noexcept override { return account_count_; }

private:
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLBuffer> account_table_;
    id<MTLBuffer> storage_table_;
    uint32_t account_capacity_;
    uint32_t storage_capacity_;
    uint32_t account_count_ = 0;
    uint32_t storage_count_ = 0;
    [[maybe_unused]] bool owns_buffers_;

    id<MTLComputePipelineState> account_lookup_pso_;
    id<MTLComputePipelineState> account_insert_pso_;
    id<MTLComputePipelineState> storage_lookup_pso_;
    id<MTLComputePipelineState> storage_insert_pso_;
    id<MTLComputePipelineState> state_root_hash_pso_;
    id<MTLComputePipelineState> state_root_reduce_pso_;
    id<MTLComputePipelineState> state_root_compact_pso_;
    id<MTLComputePipelineState> state_root_sort_pso_;
};

// -- Factory ------------------------------------------------------------------

static id<MTLComputePipelineState> make_pipeline(
    id<MTLDevice> device, id<MTLLibrary> lib, NSString* name)
{
    id<MTLFunction> func = [lib newFunctionWithName:name];
    if (!func) return nil;

    NSError* error = nil;
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:func error:&error];
    return pso;
}

std::unique_ptr<GpuHashTable> GpuHashTable::create(uint32_t capacity, void* device_ptr)
{
    @autoreleasepool {
        id<MTLDevice> device;
        if (device_ptr)
            device = (__bridge id<MTLDevice>)device_ptr;
        else
            device = MTLCreateSystemDefaultDevice();
        if (!device) return nullptr;

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) return nullptr;

        id<MTLLibrary> lib = load_state_table_library(device);
        if (!lib) return nullptr;

        // Create pipelines for all 8 kernels.
        auto acct_lookup   = make_pipeline(device, lib, @"account_lookup_batch");
        auto acct_insert   = make_pipeline(device, lib, @"account_insert_batch");
        auto stor_lookup   = make_pipeline(device, lib, @"storage_lookup_batch");
        auto stor_insert   = make_pipeline(device, lib, @"storage_insert_batch");
        auto root_hash     = make_pipeline(device, lib, @"state_root_hash_entries");
        auto root_reduce   = make_pipeline(device, lib, @"state_root_reduce");
        auto root_compact  = make_pipeline(device, lib, @"state_root_compact");
        auto root_sort     = make_pipeline(device, lib, @"state_root_sort");

        if (!acct_lookup || !acct_insert || !stor_lookup || !stor_insert ||
            !root_hash || !root_reduce || !root_compact || !root_sort)
            return nullptr;

        // Allocate persistent table buffers.
        size_t acct_buf_size = capacity * sizeof(MetalAccountEntry);
        size_t stor_buf_size = capacity * sizeof(MetalStorageEntry);

        id<MTLBuffer> acct_buf = [device newBufferWithLength:acct_buf_size
                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> stor_buf = [device newBufferWithLength:stor_buf_size
                                         options:MTLResourceStorageModeShared];
        if (!acct_buf || !stor_buf) return nullptr;

        // Zero-initialize (all key_valid = 0 means empty).
        std::memset([acct_buf contents], 0, acct_buf_size);
        std::memset([stor_buf contents], 0, stor_buf_size);

        return std::make_unique<GpuHashTableMetal>(
            device, queue, acct_buf, stor_buf,
            capacity, capacity, true,
            acct_lookup, acct_insert,
            stor_lookup, stor_insert,
            root_hash, root_reduce,
            root_compact, root_sort);
    }
}

std::unique_ptr<GpuHashTable> GpuHashTable::create_with_buffer(
    uint32_t capacity, void* buffer, void* device_ptr, void* queue_ptr)
{
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)device_ptr;
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queue_ptr;
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffer;
        if (!device || !queue || !buf) return nullptr;

        id<MTLLibrary> lib = load_state_table_library(device);
        if (!lib) return nullptr;

        auto acct_lookup   = make_pipeline(device, lib, @"account_lookup_batch");
        auto acct_insert   = make_pipeline(device, lib, @"account_insert_batch");
        auto stor_lookup   = make_pipeline(device, lib, @"storage_lookup_batch");
        auto stor_insert   = make_pipeline(device, lib, @"storage_insert_batch");
        auto root_hash     = make_pipeline(device, lib, @"state_root_hash_entries");
        auto root_reduce   = make_pipeline(device, lib, @"state_root_reduce");
        auto root_compact  = make_pipeline(device, lib, @"state_root_compact");
        auto root_sort     = make_pipeline(device, lib, @"state_root_sort");

        if (!acct_lookup || !acct_insert || !stor_lookup || !stor_insert ||
            !root_hash || !root_reduce || !root_compact || !root_sort)
            return nullptr;

        // Use the same buffer for both account and storage (caller manages layout).
        // For the pre-allocated pool case, the caller passes separate buffers.
        // Here we just use the one buffer as the account table.
        size_t stor_buf_size = capacity * sizeof(MetalStorageEntry);
        id<MTLBuffer> stor_buf = [device newBufferWithLength:stor_buf_size
                                         options:MTLResourceStorageModeShared];
        if (!stor_buf) return nullptr;
        std::memset([stor_buf contents], 0, stor_buf_size);

        return std::make_unique<GpuHashTableMetal>(
            device, queue, buf, stor_buf,
            capacity, capacity, false,
            acct_lookup, acct_insert,
            stor_lookup, stor_insert,
            root_hash, root_reduce,
            root_compact, root_sort);
    }
}

}  // namespace evm::state
