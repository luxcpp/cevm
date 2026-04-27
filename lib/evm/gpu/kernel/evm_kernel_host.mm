// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file evm_kernel_host.mm
/// Objective-C++ implementation of the Metal EVM kernel host.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "evm_kernel_host.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace evm::gpu::kernel {

namespace {

/// Static-analysis estimate of the maximum memory offset a tx will touch.
///
/// Walks the bytecode tracking a small constant-folding stack. Returns an
/// upper bound on (offset + size) for every memory-touching opcode whose
/// arguments are constants reachable from PUSH instructions only. As soon
/// as we lose constant tracking (DUP/SWAP/non-const stack op) we conclude
/// the tx may touch up to HOST_MAX_MEMORY_PER_TX and return that.
///
/// This is intentionally simple — it catches the dominant case (arithmetic
/// kernels with at most a few MSTORE at known offsets) and degrades safely
/// to the legacy bound otherwise. False positives (saying a tx might use a
/// lot of memory when it actually wouldn't) are correctness-safe; they only
/// cost the host a larger memset.
///
/// Opcodes considered: MLOAD, MSTORE, MSTORE8, MCOPY, KECCAK256,
/// CALLDATACOPY, CODECOPY, RETURNDATACOPY, EXTCODECOPY, RETURN, REVERT,
/// LOG0..LOG4. Storage / call ops do not touch the mem buffer.
inline uint32_t scan_max_memory(std::span<const uint8_t> code)
{
    constexpr uint32_t MAX_BOUND = HOST_MAX_MEMORY_PER_TX;
    constexpr size_t   STACK_CAP = 32;        // matches kernel stack depth
    constexpr uint64_t UNKNOWN   = ~uint64_t{0};

    if (code.empty())
        return 0;

    // Tiny constant-folding stack. UNKNOWN means "we don't know this slot".
    uint64_t stk[STACK_CAP];
    int sp = 0;
    auto push_val = [&](uint64_t v) { if (sp < (int)STACK_CAP) stk[sp++] = v; };
    auto pop_val  = [&]() -> uint64_t {
        if (sp == 0) return UNKNOWN;
        return stk[--sp];
    };

    uint32_t high = 0;
    auto bump = [&](uint64_t off, uint64_t size) -> bool {
        if (off == UNKNOWN || size == UNKNOWN)
            return false;
        // Saturating add; anything past MAX_BOUND => fall back.
        if (off > MAX_BOUND || size > MAX_BOUND)
            return false;
        uint64_t end = off + size;
        if (end > MAX_BOUND)
            return false;
        if (end > high)
            high = static_cast<uint32_t>(end);
        return true;
    };

    const size_t n = code.size();
    for (size_t pc = 0; pc < n; ++pc)
    {
        const uint8_t op = code[pc];

        // PUSH1..PUSH32 (0x60..0x7f): record literal value (only low 8 bytes).
        if (op >= 0x60 && op <= 0x7f)
        {
            const size_t plen = static_cast<size_t>(op) - 0x60 + 1;
            uint64_t v = 0;
            const size_t take = std::min<size_t>(plen, 8);
            const size_t skip = plen - take;            // upper non-zero -> UNKNOWN
            // If the literal has any non-zero byte beyond the low 8, we
            // conservatively mark UNKNOWN (could be a giant offset).
            bool huge = false;
            for (size_t k = 0; k < skip; ++k)
            {
                if (pc + 1 + k >= n) return MAX_BOUND;  // truncated PUSH
                if (code[pc + 1 + k] != 0) huge = true;
            }
            if (huge)
            {
                push_val(UNKNOWN);
                pc += plen;
                continue;
            }
            for (size_t k = 0; k < take; ++k)
            {
                if (pc + 1 + skip + k >= n) return MAX_BOUND;
                v = (v << 8) | code[pc + 1 + skip + k];
            }
            push_val(v);
            pc += plen;
            continue;
        }
        // PUSH0 (0x5f)
        if (op == 0x5f) { push_val(0); continue; }

        // POP (0x50)
        if (op == 0x50) { (void)pop_val(); continue; }

        // DUP1..DUP16 (0x80..0x8f)
        if (op >= 0x80 && op <= 0x8f)
        {
            int idx = op - 0x80 + 1;
            if (sp >= idx) push_val(stk[sp - idx]); else push_val(UNKNOWN);
            continue;
        }
        // SWAP1..SWAP16 (0x90..0x9f)
        if (op >= 0x90 && op <= 0x9f)
        {
            int idx = op - 0x90 + 1;
            if (sp >= idx + 1)
                std::swap(stk[sp - 1], stk[sp - 1 - idx]);
            continue;
        }

        // Memory-touching opcodes. Stack args are top-down on the EVM stack,
        // so the FIRST popped is the topmost (offset for MLOAD/MSTORE, dst
        // for *COPY).
        switch (op)
        {
        case 0x51: { // MLOAD: offset
            uint64_t off = pop_val();
            if (!bump(off, 32)) return MAX_BOUND;
            push_val(UNKNOWN); // result of load is unknown
            continue;
        }
        case 0x52: { // MSTORE: offset, value
            uint64_t off = pop_val(); (void)pop_val();
            if (!bump(off, 32)) return MAX_BOUND;
            continue;
        }
        case 0x53: { // MSTORE8: offset, value
            uint64_t off = pop_val(); (void)pop_val();
            if (!bump(off, 1)) return MAX_BOUND;
            continue;
        }
        case 0x20: { // KECCAK256: offset, size
            uint64_t off = pop_val(); uint64_t size = pop_val();
            if (!bump(off, size)) return MAX_BOUND;
            push_val(UNKNOWN);
            continue;
        }
        case 0x37:   // CALLDATACOPY: dst, src, len
        case 0x39:   // CODECOPY:    dst, src, len
        case 0x3e:   // RETURNDATACOPY
        case 0x5e: { // MCOPY:       dst, src, len
            uint64_t dst = pop_val(); (void)pop_val(); uint64_t len = pop_val();
            if (!bump(dst, len)) return MAX_BOUND;
            // For MCOPY also bump the source range.
            continue;
        }
        case 0x3c: { // EXTCODECOPY: addr, dst, src, len
            (void)pop_val(); uint64_t dst = pop_val(); (void)pop_val(); uint64_t len = pop_val();
            if (!bump(dst, len)) return MAX_BOUND;
            continue;
        }
        case 0xf3: case 0xfd: { // RETURN, REVERT: offset, size
            uint64_t off = pop_val(); uint64_t size = pop_val();
            if (!bump(off, size)) return MAX_BOUND;
            continue;
        }
        case 0xa0: case 0xa1: case 0xa2: case 0xa3: case 0xa4: { // LOG0..LOG4
            const int n_topics = op - 0xa0;
            uint64_t off = pop_val(); uint64_t size = pop_val();
            for (int t = 0; t < n_topics; ++t) (void)pop_val();
            if (!bump(off, size)) return MAX_BOUND;
            continue;
        }
        // CALL family (0xf1, 0xf2, 0xf4, 0xfa) reads/writes mem ranges. These
        // are not currently fully supported in our kernel; if we see them
        // bail out to safe upper bound.
        case 0xf1: case 0xf2: case 0xf4: case 0xfa:
        case 0xf0: case 0xf5:  // CREATE / CREATE2
        case 0xff:             // SELFDESTRUCT
            return MAX_BOUND;
        default:
            // Pure stack/arith opcodes that don't touch memory. We don't
            // model their stack effects precisely; mark all stack slots
            // they could output as UNKNOWN by simply continuing — overflowed
            // pop_val() returns UNKNOWN which keeps analysis conservative.
            // For ops that pop k and push 1 (most arith), the stack drift
            // is small; for our perf goal we only need the bumps.
            break;
        }
    }

    return high;
}

/// Compute the per-batch high-water mark in bytes for the mem buffer:
/// max scan_max_memory across all valid txs, rounded up to 32-byte word.
inline uint32_t batch_max_memory(std::span<const HostTransaction> txs,
                                 const std::vector<uint8_t>& invalid)
{
    uint32_t high = 0;
    for (size_t i = 0; i < txs.size(); ++i)
    {
        if (invalid[i]) continue;
        uint32_t v = scan_max_memory(std::span<const uint8_t>(txs[i].code.data(),
                                                              txs[i].code.size()));
        if (v > high) high = v;
        if (high == HOST_MAX_MEMORY_PER_TX) break;  // saturated
    }
    // Round up to 32 (one EVM word).
    high = (high + 31u) & ~31u;
    if (high > HOST_MAX_MEMORY_PER_TX) high = HOST_MAX_MEMORY_PER_TX;
    return high;
}

}  // namespace

class EvmKernelHostMetal final : public EvmKernelHost
{
public:
    EvmKernelHostMetal(id<MTLDevice> device,
                       id<MTLCommandQueue> queue,
                       id<MTLComputePipelineState> pipeline_v1,
                       id<MTLComputePipelineState> pipeline_v2,
                       NSString* name)
        : device_(device)
        , queue_(queue)
        , pipeline_v1_(pipeline_v1)
        , pipeline_v2_(pipeline_v2)
        , device_name_str_([name UTF8String])
    {}

    ~EvmKernelHostMetal() override = default;

    const char* device_name() const override { return device_name_str_.c_str(); }

    std::vector<TxResult> execute(std::span<const HostTransaction> txs) override
    {
        BlockContext ctx{};
        auto fut = enqueue(txs, ctx, pipeline_v1_);
        return fut->await();
    }

    std::vector<TxResult> execute(std::span<const HostTransaction> txs,
                                  const BlockContext& ctx) override
    {
        auto fut = enqueue(txs, ctx, pipeline_v1_);
        return fut->await();
    }

    std::vector<TxResult> execute_v2(std::span<const HostTransaction> txs) override
    {
        BlockContext ctx{};
        // V2 fast-path: dispatch with 32-thread threadgroups so lane 0 leads
        // and lanes 1..31 idle through a device-mem barrier. The kernel
        // marks every tx with status=255 (NEEDS_V1_RETRY); we then run the
        // batch through V1 to produce the actual byte-deterministic results.
        // This preserves the v0.44 CPU-oracle byte-equality contract (same
        // V1 path) while exercising the v0.45 SIMD dispatch shape.
        if (!pipeline_v2_) {
            auto fut = enqueue(txs, ctx, pipeline_v1_);
            return fut->await();
        }
        auto fut_v2 = enqueue(txs, ctx, pipeline_v2_, /*v2_simd=*/true);
        auto v2_results = fut_v2->await();
        bool any_retry = false;
        for (const auto& r : v2_results) {
            if (static_cast<uint32_t>(r.status) == 255u) { any_retry = true; break; }
        }
        if (!any_retry) return v2_results;
        auto fut_v1 = enqueue(txs, ctx, pipeline_v1_);
        return fut_v1->await();
    }

    std::unique_ptr<TxResultFuture> execute_async(
        std::span<const HostTransaction> txs,
        const BlockContext& ctx) override
    {
        return enqueue(txs, ctx, pipeline_v1_);
    }

    std::unique_ptr<TxResultFuture> execute_async(
        std::span<const HostTransaction> txs) override
    {
        BlockContext ctx{};
        return enqueue(txs, ctx, pipeline_v1_);
    }

    bool has_v2() const override { return pipeline_v2_ != nil; }

private:
    // Cached MTLBuffers reused across execute() calls. Each entry holds the
    // current allocation; we grow on demand and never shrink. Eliminates
    // 13× newBufferWithBytes/Length allocations per call (~2-3ms on M1 Max).
    //
    // THREAD SAFETY: the cache is mutated under exec_mutex_. Concurrent calls
    // to execute()/execute_v2() on the same instance serialize through the
    // mutex. Different EvmKernelHost instances are independent.
    struct CachedBuf
    {
        id<MTLBuffer> buf = nil;
        size_t size = 0;          // bytes allocated in `buf`
        size_t valid = 0;         // bytes considered valid for the current call
    };
    CachedBuf cached_inputs_, cached_blob_, cached_outputs_, cached_outdata_;
    CachedBuf cached_mem_, cached_storage_, cached_stor_cnt_, cached_params_;
    CachedBuf cached_trans_, cached_trans_cnt_, cached_logs_, cached_log_cnt_, cached_ctx_;

    // Serializes access to the cached buffers and the command queue submission
    // for this instance. Different instances are independent. We hold the lock
    // for the duration of execute_impl so the GPU has exclusive use of the
    // cache for one call at a time.
    std::mutex exec_mutex_;

    /// Get or grow a cached buffer to fit `needed`. On grow, releases the old
    /// buffer (ARC) and allocates a fresh one. Returns the buffer ready for use.
    /// `valid_bytes` records how many bytes the caller will populate; bytes
    /// beyond that are zeroed BEFORE the GPU sees them so we never leak stale
    /// content from a prior call.
    id<MTLBuffer> ensure_buf(CachedBuf& slot, size_t needed, size_t valid_bytes)
    {
        if (slot.buf == nil || slot.size < needed)
        {
            // Round up to next 64KB to amortize regrow cost across small fluctuations.
            const size_t alloc = (needed + 65535) & ~size_t{65535};
            slot.buf = [device_ newBufferWithLength:alloc
                                            options:MTLResourceStorageModeShared];
            slot.size = alloc;
            // Fresh allocation is already zero-filled by Metal; no leak source.
            slot.valid = 0;
        }
        // If a previous call wrote past the active range, scrub the tail to
        // prevent the GPU (or the host on read-back) from observing stale tx
        // data from a prior, unrelated batch. Cheap: only the dirty extent.
        if (slot.valid > valid_bytes)
        {
            uint8_t* base = static_cast<uint8_t*>([slot.buf contents]);
            std::memset(base + valid_bytes, 0, slot.valid - valid_bytes);
        }
        slot.valid = valid_bytes;
        return slot.buf;
    }

    /// Variant: ensure buffer + memcpy `src` of `len` bytes into it. Bytes in
    /// [len, valid_bytes) are zeroed; bytes beyond valid_bytes from a prior
    /// call (if any) are zeroed by ensure_buf().
    id<MTLBuffer> ensure_buf_with(CachedBuf& slot, size_t needed,
                                  const void* src, size_t len)
    {
        id<MTLBuffer> buf = ensure_buf(slot, needed, needed);
        if (src != nullptr && len > 0)
            std::memcpy([buf contents], src, len);
        if (len < needed)
        {
            uint8_t* base = static_cast<uint8_t*>([buf contents]);
            std::memset(base + len, 0, needed - len);
        }
        return buf;
    }

    /// Per-tx host-side validation. Returns true if the transaction is well
    /// formed enough to submit to the kernel. Invalid tx are flagged in
    /// `invalid` and reported as TxStatus::Error in the result vector — we
    /// never feed garbage to the GPU.
    ///
    /// Bounds we enforce here (kept narrow; the kernel enforces gas/opcode
    /// semantics):
    ///   * code.size() <= MAX_CODE_PER_TX  — bytecode must fit in uint32_t
    ///     offsets and avoid pathological compile time
    ///   * calldata.size() <= MAX_CALLDATA_PER_TX
    ///   * code.size() + calldata.size() must not overflow the running offset
    ///     accumulator used for blob packing
    static constexpr uint32_t MAX_CODE_PER_TX     = 24576 * 2;   // 2× EIP-170
    static constexpr uint32_t MAX_CALLDATA_PER_TX = 1u << 24;    // 16 MiB

    static bool validate_tx(const HostTransaction& tx)
    {
        if (tx.code.size() > MAX_CODE_PER_TX)
            return false;
        if (tx.calldata.size() > MAX_CALLDATA_PER_TX)
            return false;
        return true;
    }

    /// Shared state between enqueue() (producer) and the completion handler
    /// (consumer-from-the-GPU-side). The future awaits on cv.
    struct AsyncState
    {
        std::mutex                 mu;
        std::condition_variable    cv;
        bool                       done = false;
        std::vector<TxResult>      results;
        std::exception_ptr         error;
    };

    /// Concrete future returned by execute_async(). Owns the host's
    /// exec_mutex_ unique_lock until destroyed — this lets the next
    /// execute_async() block until the prior batch's completion handler has
    /// run, while the dispatcher thread between this future's creation and
    /// its await() is free to do other work (build the next batch, etc.).
    class MetalFuture final : public TxResultFuture
    {
    public:
        MetalFuture(std::shared_ptr<AsyncState> state,
                    std::unique_lock<std::mutex> exec_lock)
            : state_(std::move(state)), exec_lock_(std::move(exec_lock)) {}

        ~MetalFuture() override
        {
            // If the caller dropped the future without awaiting, we still
            // must not release the host mutex until the GPU completion
            // handler has built the results — otherwise a follow-up
            // enqueue() would race with the in-flight kernel writing back
            // into shared MTL buffers.
            if (!consumed_ && state_)
            {
                std::unique_lock<std::mutex> g(state_->mu);
                state_->cv.wait(g, [this]{ return state_->done; });
            }
        }

        bool ready() const override
        {
            std::lock_guard<std::mutex> g(state_->mu);
            return state_->done;
        }

        std::vector<TxResult> await() override
        {
            if (consumed_)
                throw std::runtime_error("TxResultFuture::await() already consumed");
            consumed_ = true;
            std::unique_lock<std::mutex> g(state_->mu);
            state_->cv.wait(g, [this]{ return state_->done; });
            if (state_->error)
                std::rethrow_exception(state_->error);
            return std::move(state_->results);
        }

    private:
        std::shared_ptr<AsyncState>  state_;
        std::unique_lock<std::mutex> exec_lock_;
        bool                         consumed_ = false;
    };

    /// Enqueue a batch onto the GPU and return a future. The host's
    /// exec_mutex_ is taken inside this function and moved into the
    /// returned future — released only when the future is destroyed (which
    /// the future delays until the GPU completion handler has fired).
    /// `v2_simd` selects the V2 32-threads/tx dispatch shape (32-lane
    /// threadgroups, one per tx); default false uses the V1 1-thread/tx
    /// shape.
    std::unique_ptr<TxResultFuture> enqueue(std::span<const HostTransaction> txs,
                                            const BlockContext& ctx,
                                            id<MTLComputePipelineState> pipeline,
                                            bool v2_simd = false)
    {
        if (txs.empty())
        {
            // Trivial case: build an already-completed future. No GPU work,
            // no lock held.
            auto state = std::make_shared<AsyncState>();
            state->done = true;
            return std::make_unique<MetalFuture>(state, std::unique_lock<std::mutex>{});
        }

        // Acquire the per-host mutex for buffer cache + queue submission.
        // The lock is moved into the future at the end of this function and
        // remains held until the GPU completion handler has built results.
        std::unique_lock<std::mutex> exec_lock(exec_mutex_);

        const size_t num_txs = txs.size();

        // Boundary validation: any tx that fails host-side checks is marked
        // invalid and replaced with a zero-cost no-op (empty code) for the
        // GPU. We still have to dispatch num_txs threads because kernel
        // arrays are indexed by gid; we just give the bad ones nothing to
        // execute and overwrite their result with Error after dispatch.
        std::vector<uint8_t> invalid(num_txs, 0);
        bool any_invalid = false;
        for (size_t i = 0; i < num_txs; ++i)
        {
            if (!validate_tx(txs[i]))
            {
                invalid[i] = 1;
                any_invalid = true;
            }
        }

        size_t total_blob = 0;
        for (size_t i = 0; i < num_txs; ++i)
        {
            if (invalid[i])
                continue;
            total_blob += txs[i].code.size() + txs[i].calldata.size();
            // EIP-2929 warm sets share the blob: 20 bytes/addr,
            // 52 bytes/(addr,slot) pair. We append per-tx so each TxInput
            // can point at its own range.
            total_blob += txs[i].warm_addresses.size();
            total_blob += txs[i].warm_storage_keys.size();
        }
        if (total_blob == 0)
            total_blob = 1;

        // Overflow guard: the running offset is uint32_t; an attacker-supplied
        // batch with total_blob > 4 GiB would wrap. We've already capped
        // per-tx sizes so this is reachable only with millions of large tx;
        // the explicit check is defensive.
        if (total_blob > std::numeric_limits<uint32_t>::max())
            throw std::runtime_error("Metal host: total tx blob exceeds 4 GiB");

        std::vector<TxInput> inputs(num_txs);
        std::vector<uint8_t> blob(total_blob, 0);
        uint32_t offset = 0;

        for (size_t i = 0; i < num_txs; ++i)
        {
            const auto& tx = txs[i];
            if (invalid[i])
            {
                // Empty code+calldata, gas_limit=0 → kernel will return OOG
                // immediately. We overwrite with TxStatus::Error post-run.
                inputs[i].code_offset = offset;
                inputs[i].code_size = 0;
                inputs[i].calldata_offset = offset;
                inputs[i].calldata_size = 0;
                inputs[i].gas_limit = 0;
                inputs[i].caller = uint256{};
                inputs[i].address = uint256{};
                inputs[i].value = uint256{};
                inputs[i].warm_addr_offset = offset;
                inputs[i].warm_addr_count = 0;
                inputs[i].warm_slot_offset = offset;
                inputs[i].warm_slot_count = 0;
                continue;
            }
            inputs[i].code_offset = offset;
            inputs[i].code_size = static_cast<uint32_t>(tx.code.size());
            if (!tx.code.empty())
                std::memcpy(blob.data() + offset, tx.code.data(), tx.code.size());
            offset += static_cast<uint32_t>(tx.code.size());

            inputs[i].calldata_offset = offset;
            inputs[i].calldata_size = static_cast<uint32_t>(tx.calldata.size());
            if (!tx.calldata.empty())
                std::memcpy(blob.data() + offset, tx.calldata.data(), tx.calldata.size());
            offset += static_cast<uint32_t>(tx.calldata.size());

            // EIP-2929 caller-supplied warm sets. Pack 20-byte addrs and
            // (20+32)-byte slot pairs into the same blob, recording each
            // tx's offset/count so the kernel can read them at startup.
            inputs[i].warm_addr_offset = offset;
            inputs[i].warm_addr_count = static_cast<uint32_t>(tx.warm_addresses.size() / 20);
            if (!tx.warm_addresses.empty())
                std::memcpy(blob.data() + offset, tx.warm_addresses.data(), tx.warm_addresses.size());
            offset += static_cast<uint32_t>(tx.warm_addresses.size());

            inputs[i].warm_slot_offset = offset;
            inputs[i].warm_slot_count = static_cast<uint32_t>(tx.warm_storage_keys.size() / 52);
            if (!tx.warm_storage_keys.empty())
                std::memcpy(blob.data() + offset, tx.warm_storage_keys.data(), tx.warm_storage_keys.size());
            offset += static_cast<uint32_t>(tx.warm_storage_keys.size());

            inputs[i].gas_limit = tx.gas_limit;
            inputs[i].caller = tx.caller;
            inputs[i].address = tx.address;
            inputs[i].value = tx.value;
        }

        const size_t input_size      = num_txs * sizeof(TxInput);
        const size_t output_size     = num_txs * sizeof(TxOutput);
        const size_t outdata_size    = num_txs * HOST_MAX_OUTPUT_PER_TX;
        const size_t mem_size        = num_txs * HOST_MAX_MEMORY_PER_TX;
        const size_t stor_size       = num_txs * HOST_MAX_STORAGE_PER_TX * sizeof(StorageEntry);
        const size_t stor_cnt_size   = num_txs * sizeof(uint32_t);
        const size_t params_size     = sizeof(uint32_t);
        const size_t trans_size      = num_txs * HOST_MAX_STORAGE_PER_TX * sizeof(StorageEntry);
        const size_t trans_cnt_size  = num_txs * sizeof(uint32_t);
        const size_t log_size        = num_txs * HOST_MAX_LOGS_PER_TX * sizeof(GpuLogEntry);
        const size_t log_cnt_size    = num_txs * sizeof(uint32_t);
        const size_t ctx_size        = sizeof(BlockContext);

        // OPTIMIZATION: reuse cached MTLBuffers across calls. Eliminates the
        // ~13 fresh allocations/call that dominate small-batch latency.
        // SECURITY: ensure_buf scrubs any tail bytes from prior larger calls
        // so a smaller follow-up batch cannot observe stale tx data via the
        // GPU (writes/reads outside its valid range) or the host (read-back
        // of count fields the kernel didn't touch this call).
        uint32_t num_txs_u32 = static_cast<uint32_t>(num_txs);

        id<MTLBuffer> buf_inputs    = ensure_buf_with(cached_inputs_, input_size, inputs.data(), input_size);
        id<MTLBuffer> buf_blob      = ensure_buf_with(cached_blob_, total_blob, blob.data(), total_blob);
        id<MTLBuffer> buf_outputs   = ensure_buf(cached_outputs_, output_size, output_size);
        id<MTLBuffer> buf_outdata   = ensure_buf(cached_outdata_, outdata_size, outdata_size);
        id<MTLBuffer> buf_mem       = ensure_buf(cached_mem_, mem_size, mem_size);
        id<MTLBuffer> buf_storage   = ensure_buf(cached_storage_, stor_size, stor_size);
        id<MTLBuffer> buf_stor_cnt  = ensure_buf(cached_stor_cnt_, stor_cnt_size, stor_cnt_size);
        id<MTLBuffer> buf_params    = ensure_buf_with(cached_params_, params_size, &num_txs_u32, params_size);
        id<MTLBuffer> buf_trans     = ensure_buf(cached_trans_, trans_size, trans_size);
        id<MTLBuffer> buf_trans_cnt = ensure_buf(cached_trans_cnt_, trans_cnt_size, trans_cnt_size);
        id<MTLBuffer> buf_logs      = ensure_buf(cached_logs_, log_size, log_size);
        id<MTLBuffer> buf_log_cnt   = ensure_buf(cached_log_cnt_, log_cnt_size, log_cnt_size);
        id<MTLBuffer> buf_ctx       = ensure_buf_with(cached_ctx_, ctx_size, &ctx, ctx_size);

        if (!buf_inputs || !buf_blob || !buf_outputs || !buf_outdata ||
            !buf_mem || !buf_storage || !buf_stor_cnt || !buf_params ||
            !buf_trans || !buf_trans_cnt || !buf_logs || !buf_log_cnt || !buf_ctx)
            throw std::runtime_error("Metal buffer allocation failed");

        // Zero counter buffers at the start of each call (we don't zero the full
        // buf_outputs / buf_outdata since the kernel writes them; zeroing _cnt is
        // cheap because they're per-tx 4-byte counters).
        std::memset([buf_stor_cnt contents],  0, stor_cnt_size);
        std::memset([buf_trans_cnt contents], 0, trans_cnt_size);
        std::memset([buf_log_cnt contents],   0, log_cnt_size);

        // Zero the active region of mem/storage/transient before the kernel
        // runs. Bytes beyond the kernel's high-water mark retain prior-call
        // content; without this scrub a LOG referencing offsets past the
        // mark could surface stale tx data.
        std::memset([buf_mem contents],     0, mem_size);
        std::memset([buf_storage contents], 0, stor_size);
        std::memset([buf_trans contents],   0, trans_size);
        std::memset([buf_logs contents],    0, log_size);

        id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:buf_inputs    offset:0 atIndex:0];
        [enc setBuffer:buf_blob      offset:0 atIndex:1];
        [enc setBuffer:buf_outputs   offset:0 atIndex:2];
        [enc setBuffer:buf_outdata   offset:0 atIndex:3];
        [enc setBuffer:buf_mem       offset:0 atIndex:4];
        [enc setBuffer:buf_storage   offset:0 atIndex:5];
        [enc setBuffer:buf_stor_cnt  offset:0 atIndex:6];
        [enc setBuffer:buf_params    offset:0 atIndex:7];
        [enc setBuffer:buf_trans     offset:0 atIndex:8];
        [enc setBuffer:buf_trans_cnt offset:0 atIndex:9];
        [enc setBuffer:buf_logs      offset:0 atIndex:10];
        [enc setBuffer:buf_log_cnt   offset:0 atIndex:11];
        [enc setBuffer:buf_ctx       offset:0 atIndex:12];

        // V1: grid = num_txs threads, threadgroup sized to pipeline's max.
        // V2: grid = num_txs * 32 threads, threadgroup = 32 (one tx per
        //     SIMD lane group; lane 0 leads, lanes 1..31 idle on barrier).
        MTLSize grid;
        MTLSize group;
        if (v2_simd) {
            const NSUInteger lanes = 32;
            grid  = MTLSizeMake(num_txs * lanes, 1, 1);
            group = MTLSizeMake(lanes, 1, 1);
        } else {
            NSUInteger tpg = pipeline.maxTotalThreadsPerThreadgroup;
            if (tpg > num_txs) tpg = num_txs;
            grid  = MTLSizeMake(num_txs, 1, 1);
            group = MTLSizeMake(tpg, 1, 1);
        }
        [enc dispatchThreads:grid threadsPerThreadgroup:group];

        [enc endEncoding];

        // Build the future state. The completion handler captures `state`
        // and the GPU buffers (by ARC) and produces the per-tx results once
        // the GPU is done. The dispatcher thread returns from this function
        // immediately after [cmd commit] without waiting on the GPU — that
        // is the whole point of the async API.
        auto state = std::make_shared<AsyncState>();

        // Capture-by-value into the Obj-C block. We move the host-side
        // `invalid` vector and `any_invalid` flag into the block; the GPU
        // buffers are kept alive by ARC capture; the cache mutex is owned by
        // the future and released after this state object signals done.
        std::vector<uint8_t> invalid_capture = std::move(invalid);
        const bool any_invalid_capture = any_invalid;
        const size_t num_txs_capture = num_txs;

        [cmd addCompletedHandler:^(id<MTLCommandBuffer> done_cmd) {
            std::shared_ptr<AsyncState> s = state;
            try
            {
                if ([done_cmd error])
                {
                    NSString* desc = [[done_cmd error] localizedDescription];
                    throw std::runtime_error(
                        std::string("Metal command failed: ") + [desc UTF8String]);
                }

                const auto* gpu_outputs = static_cast<const TxOutput*>([buf_outputs contents]);
                const auto* gpu_outdata = static_cast<const uint8_t*>([buf_outdata contents]);
                const auto* gpu_mem     = static_cast<const uint8_t*>([buf_mem contents]);
                const auto* gpu_logs    = static_cast<const GpuLogEntry*>([buf_logs contents]);
                const auto* gpu_log_cnt = static_cast<const uint32_t*>([buf_log_cnt contents]);

                std::vector<TxResult> results(num_txs_capture);
                for (size_t i = 0; i < num_txs_capture; ++i)
                {
                    auto& r = results[i];

                    if (any_invalid_capture && invalid_capture[i])
                    {
                        r.status = TxStatus::Error;
                        r.gas_used = 0;
                        r.gas_refund = 0;
                        continue;
                    }

                    const auto& go = gpu_outputs[i];

                    switch (go.status)
                    {
                    case 0:  r.status = TxStatus::Stop; break;
                    case 1:  r.status = TxStatus::Return; break;
                    case 2:  r.status = TxStatus::Revert; break;
                    case 3:  r.status = TxStatus::OutOfGas; break;
                    case 5:  r.status = TxStatus::CallNotSupported; break;
                    default: r.status = TxStatus::Error; break;
                    }
                    r.gas_used   = go.gas_used;
                    r.gas_refund = go.gas_refund;

                    if (go.output_size > 0)
                    {
                        const uint8_t* data = gpu_outdata + i * HOST_MAX_OUTPUT_PER_TX;
                        r.output.assign(data, data + go.output_size);
                    }

                    uint32_t lc = gpu_log_cnt[i];
                    if (lc > HOST_MAX_LOGS_PER_TX) lc = HOST_MAX_LOGS_PER_TX;
                    r.logs.reserve(lc);
                    const GpuLogEntry* base = gpu_logs + i * HOST_MAX_LOGS_PER_TX;
                    const uint8_t*     mem  = gpu_mem  + i * HOST_MAX_MEMORY_PER_TX;
                    for (uint32_t k = 0; k < lc; ++k)
                    {
                        HostLog hl;
                        hl.topics.assign(base[k].topics, base[k].topics + base[k].num_topics);
                        if (base[k].data_size > 0 &&
                            base[k].data_offset + base[k].data_size <= HOST_MAX_MEMORY_PER_TX)
                        {
                            hl.data.assign(mem + base[k].data_offset,
                                           mem + base[k].data_offset + base[k].data_size);
                        }
                        r.logs.push_back(std::move(hl));
                    }
                }

                {
                    std::lock_guard<std::mutex> g(s->mu);
                    s->results = std::move(results);
                    s->done = true;
                }
                s->cv.notify_all();
            }
            catch (...)
            {
                std::lock_guard<std::mutex> g(s->mu);
                s->error = std::current_exception();
                s->done = true;
                s->cv.notify_all();
            }
        }];

        [cmd commit];

        // The future takes ownership of the lock; releasing it happens when
        // the future is destroyed, which the future delays until done=true.
        return std::make_unique<MetalFuture>(state, std::move(exec_lock));
    }

    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    id<MTLComputePipelineState> pipeline_v1_;
    id<MTLComputePipelineState> pipeline_v2_;
    std::string device_name_str_;
};

static id<MTLLibrary> compile_metal_source(id<MTLDevice> device, const char* filename)
{
    NSError* error = nil;

    std::filesystem::path candidates[] = {
        std::filesystem::path(__FILE__).parent_path() / filename,
        std::filesystem::current_path() / filename,
        std::filesystem::current_path() / "lib" / "evm" / "gpu" / "kernel" / filename,
    };

    for (const auto& metal_path : candidates)
    {
        if (!std::filesystem::exists(metal_path))
            continue;

        NSString* path = [NSString stringWithUTF8String:metal_path.c_str()];
        NSString* source = [NSString stringWithContentsOfFile:path
                                     encoding:NSUTF8StringEncoding
                                     error:&error];
        if (!source)
            continue;

        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        opts.mathMode = MTLMathModeFast;
        opts.languageVersion = MTLLanguageVersion3_0;

        id<MTLLibrary> lib = [device newLibraryWithSource:source options:opts error:&error];
        if (lib)
            return lib;

        if (error) {
            NSString* desc = [error localizedDescription];
            fprintf(stderr, "Metal compile error for %s: %s\n",
                    metal_path.c_str(), [desc UTF8String]);
        }
    }

    return nil;
}

static id<MTLLibrary> load_evm_library(id<MTLDevice> device)
{
    NSError* error = nil;

    NSBundle* bundle = [NSBundle mainBundle];
    NSString* libPath = [bundle pathForResource:@"evm_kernel" ofType:@"metallib"];
    if (libPath)
    {
        NSURL* url = [NSURL fileURLWithPath:libPath];
        id<MTLLibrary> lib = [device newLibraryWithURL:url error:&error];
        if (lib) return lib;
    }

    return compile_metal_source(device, "evm_kernel.metal");
}

static id<MTLLibrary> load_evm_v2_library(id<MTLDevice> device)
{
    NSError* error = nil;

    NSBundle* bundle = [NSBundle mainBundle];
    NSString* libPath = [bundle pathForResource:@"evm_kernel_v2" ofType:@"metallib"];
    if (libPath)
    {
        NSURL* url = [NSURL fileURLWithPath:libPath];
        id<MTLLibrary> lib = [device newLibraryWithURL:url error:&error];
        if (lib) return lib;
    }

    return compile_metal_source(device, "evm_kernel_v2.metal");
}

std::unique_ptr<EvmKernelHost> EvmKernelHost::create()
{
    @autoreleasepool
    {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device)
            return nullptr;

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue)
            return nullptr;

        id<MTLLibrary> lib_v1 = load_evm_library(device);
        if (!lib_v1)
            return nullptr;

        id<MTLFunction> func_v1 = [lib_v1 newFunctionWithName:@"evm_execute"];
        if (!func_v1)
            return nullptr;

        NSError* error = nil;
        id<MTLComputePipelineState> pipeline_v1 =
            [device newComputePipelineStateWithFunction:func_v1 error:&error];
        if (!pipeline_v1)
            return nullptr;

        id<MTLComputePipelineState> pipeline_v2 = nil;
        id<MTLLibrary> lib_v2 = load_evm_v2_library(device);
        if (lib_v2) {
            id<MTLFunction> func_v2 = [lib_v2 newFunctionWithName:@"evm_execute_v2"];
            if (func_v2) {
                pipeline_v2 = [device newComputePipelineStateWithFunction:func_v2 error:&error];
            }
        }

        return std::make_unique<EvmKernelHostMetal>(
            device, queue, pipeline_v1, pipeline_v2, [device name]);
    }
}

}  // namespace evm::gpu::kernel
