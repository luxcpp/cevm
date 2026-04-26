// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Lux DEX_MATCH precompile (0x100) — Metal/lux-accel wrapper.
//
// Routes the EVM calldata layout into the lux-accel `lux_match_orders`
// entry point (luxcpp/lux-accel/include/lux/accel/c_api.h), which dispatches
// to the GPU shaders in luxcpp/{cuda,metal}/kernels/gpu/dex_swap.* once the
// lux-accel C++ implementation in lux-accel/src/dex.cpp returns
// gpu::Status::OK. Until then lux_match_orders returns LUX_NOT_SUPPORTED
// and we fall back to the CPU reference (dex_match_cpu.cpp) — same pattern
// ecrecover_metal.mm uses when the kernel reports `valid=0`.
//
// Tensor layout for the lux-accel API:
//
//   bids:    shape [N_bids, 3] u64  (price, qty, order_id)
//   asks:    shape [N_asks, 3] u64  (price, qty, order_id)
//   matches: shape [k, 4]      u64  (bid_id, ask_id, qty, price)
//   prices:  shape [k]         u64
//   amounts: shape [k]         u64
//
// The EVM precompile feeds in a single incoming order; the wrapper builds a
// 1-row "incoming" tensor on the appropriate side and a tensor view of the
// resting opposite side. For the first wiring we shape the call so the GPU
// kernel receives both sides as full snapshots; this trades a copy for a
// stable invariant — the kernel never mutates the in-process book table,
// the wrapper does.
//
// This wrapper is consensus-correct only when the GPU produces the same
// VWAP-and-fills tuple as the CPU reference. Because lux_match_orders is
// currently a stub, this path always falls back to CPU — the wiring is
// proven by the precompile dispatcher reporting backend == metal at
// install time, but the bytes returned to the EVM are produced by the
// CPU reference until lux-accel ships the real kernel.

#import <Foundation/Foundation.h>

#include "internal.hpp"
#include "precompile_dispatch.hpp"
#include <lux/accel/c_api.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

namespace evm::gpu::precompile
{
extern "C" void evm_precompile_set_impl(
    void* dispatcher, uint16_t address, Result (*fn)(std::span<const uint8_t>, uint64_t),
    int backend_id) noexcept;

namespace
{

// Process-singleton lux-accel session bound to Metal. Created lazily on first
// call; freed at process exit. lux-accel's session_destroy is reentrant so
// concurrent calls into the precompile are safe — the wrapper above us
// (cevm dispatcher) does its own batching.
lux_session accel_session()
{
    static lux_session s = []() -> lux_session {
        lux_session out = nullptr;
        if (lux_init() != LUX_OK)
            return nullptr;
        // Auto picks the best backend on the host. On macOS that's Metal.
        if (lux_session_create_with_backend(LUX_BACKEND_METAL, &out) != LUX_OK)
            return nullptr;
        return out;
    }();
    return s;
}

Result dex_match_accel(std::span<const uint8_t> input, uint64_t gas_limit)
{
    // Shortcut: any decode error or out-of-gas is identical between CPU and
    // GPU paths, so let the CPU reference handle them. This also keeps the
    // calldata layout owned in exactly one place (dex_match_cpu.cpp).
    lux_session sess = accel_session();
    if (!sess)
        return dex_match_cpu(input, gas_limit);

    // Probe the lux-accel match_orders entry point with a minimal 1x1 call.
    // If it returns LUX_NOT_SUPPORTED (the current state — stub
    // implementation in lux-accel/src/dex.cpp) we fall through to CPU. We
    // do this on every call because the binding is dlopen-style; a future
    // build of liblux_accel that ships the real kernel takes over without
    // any cevm-side change.
    //
    // Once lux_match_orders returns LUX_OK with consensus-grade output we
    // remove this probe and translate the EVM calldata into tensors here.
    lux_tensor bids_t = nullptr;
    const size_t one[2] = {1u, 3u};
    if (lux_tensor_create(sess, LUX_DTYPE_U64, one, 2, &bids_t) != LUX_OK)
        return dex_match_cpu(input, gas_limit);

    lux_tensor asks_t = nullptr;
    if (lux_tensor_create(sess, LUX_DTYPE_U64, one, 2, &asks_t) != LUX_OK)
    {
        lux_tensor_destroy(bids_t);
        return dex_match_cpu(input, gas_limit);
    }

    const size_t mshape[2] = {0u, 4u};
    lux_tensor matches_t = nullptr;
    lux_tensor prices_t = nullptr;
    lux_tensor amounts_t = nullptr;
    const size_t kone[1] = {0u};
    if (lux_tensor_create(sess, LUX_DTYPE_U64, mshape, 2, &matches_t) != LUX_OK ||
        lux_tensor_create(sess, LUX_DTYPE_U64, kone, 1, &prices_t) != LUX_OK ||
        lux_tensor_create(sess, LUX_DTYPE_U64, kone, 1, &amounts_t) != LUX_OK)
    {
        if (matches_t) lux_tensor_destroy(matches_t);
        if (prices_t)  lux_tensor_destroy(prices_t);
        if (amounts_t) lux_tensor_destroy(amounts_t);
        lux_tensor_destroy(asks_t);
        lux_tensor_destroy(bids_t);
        return dex_match_cpu(input, gas_limit);
    }

    const lux_status st = lux_match_orders(sess, bids_t, asks_t, matches_t,
                                           prices_t, amounts_t);

    lux_tensor_destroy(amounts_t);
    lux_tensor_destroy(prices_t);
    lux_tensor_destroy(matches_t);
    lux_tensor_destroy(asks_t);
    lux_tensor_destroy(bids_t);

    if (st != LUX_OK)
    {
        // Stub returns LUX_NOT_SUPPORTED. This is the expected state today.
        return dex_match_cpu(input, gas_limit);
    }

    // Real kernel path: translate the EVM calldata into tensors, run, copy
    // back. Out of scope until lux-accel actually returns LUX_OK; for now
    // even a "successful" probe falls back to the CPU reference because we
    // haven't filled in the GPU output → uint256 BE encoding.
    return dex_match_cpu(input, gas_limit);
}

}  // namespace

}  // namespace evm::gpu::precompile

extern "C" void evm_precompile_install_dex_match_metal(void* dispatcher)
{
    using namespace evm::gpu::precompile;

    // Probe lux-accel availability. If the dylib didn't load or no Metal
    // device is present, leave the CPU default in place.
    if (!accel_session())
        return;

    // Backend::Metal == 2.
    evm_precompile_set_impl(dispatcher, 0x100, &dex_match_accel, 2);
}
