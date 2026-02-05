// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file persistent_pool.mm
/// Objective-C++ implementation of PersistentBufferPool.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "persistent_pool.hpp"

#include <cstring>

namespace evm::gpu
{

static PersistentBuffer allocate_buffer(id<MTLDevice> device, size_t size)
{
    PersistentBuffer buf;
    buf.size = size;

    // MTLResourceStorageModeShared: zero-copy on Apple Silicon unified memory.
    // CPU and GPU share the same physical pages.
    id<MTLBuffer> mtl_buf = [device newBufferWithLength:size
                                    options:MTLResourceStorageModeShared];
    if (!mtl_buf)
        return {};

    // Zero-initialize.
    std::memset([mtl_buf contents], 0, size);

    buf.metal_buffer = (void*)CFBridgingRetain(mtl_buf);
    buf.contents = [mtl_buf contents];
    return buf;
}

static void release_buffer(PersistentBuffer& buf)
{
    if (buf.metal_buffer)
    {
        // Release the retained MTLBuffer.
        CFRelease(buf.metal_buffer);
        buf.metal_buffer = nullptr;
        buf.contents = nullptr;
        buf.size = 0;
    }
}

bool PersistentBufferPool::init(void* device, const PoolSizes& sizes)
{
    if (!device)
        return false;

    device_ = device;
    id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device;

    account_table_ = allocate_buffer(mtl_device, sizes.account_table);
    if (!account_table_.metal_buffer)
        return false;

    storage_table_ = allocate_buffer(mtl_device, sizes.storage_table);
    if (!storage_table_.metal_buffer)
    {
        release_buffer(account_table_);
        return false;
    }

    tx_input_ = allocate_buffer(mtl_device, sizes.tx_input);
    tx_output_ = allocate_buffer(mtl_device, sizes.tx_output);
    evm_memory_ = allocate_buffer(mtl_device, sizes.evm_memory);
    mv_memory_ = allocate_buffer(mtl_device, sizes.mv_memory);

    return true;
}

void PersistentBufferPool::destroy() noexcept
{
    release_buffer(account_table_);
    release_buffer(storage_table_);
    release_buffer(tx_input_);
    release_buffer(tx_output_);
    release_buffer(evm_memory_);
    release_buffer(mv_memory_);
    device_ = nullptr;
}

PersistentBufferPool::~PersistentBufferPool()
{
    destroy();
}

PersistentBufferPool::PersistentBufferPool(PersistentBufferPool&& other) noexcept
    : device_(other.device_)
    , account_table_(other.account_table_)
    , storage_table_(other.storage_table_)
    , tx_input_(other.tx_input_)
    , tx_output_(other.tx_output_)
    , evm_memory_(other.evm_memory_)
    , mv_memory_(other.mv_memory_)
{
    other.device_ = nullptr;
    other.account_table_ = {};
    other.storage_table_ = {};
    other.tx_input_ = {};
    other.tx_output_ = {};
    other.evm_memory_ = {};
    other.mv_memory_ = {};
}

PersistentBufferPool& PersistentBufferPool::operator=(PersistentBufferPool&& other) noexcept
{
    if (this != &other)
    {
        destroy();
        device_ = other.device_;
        account_table_ = other.account_table_;
        storage_table_ = other.storage_table_;
        tx_input_ = other.tx_input_;
        tx_output_ = other.tx_output_;
        evm_memory_ = other.evm_memory_;
        mv_memory_ = other.mv_memory_;
        other.device_ = nullptr;
        other.account_table_ = {};
        other.storage_table_ = {};
        other.tx_input_ = {};
        other.tx_output_ = {};
        other.evm_memory_ = {};
        other.mv_memory_ = {};
    }
    return *this;
}

}  // namespace evm::gpu
