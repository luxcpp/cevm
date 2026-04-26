// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file slot_kernel.metal
/// GPU Slot Engine — bounded-epoch scheduler kernel.
///
/// One invocation processes ONE epoch. The host re-launches the same kernel
/// until SlotResult::status != 0. Inside one epoch:
///
///   * Each workgroup picks a service (Ingress/Decode/Crypto/Commit/...)
///     by gid % kNumServices, then drains up to epoch_budget_items work
///     items from that service's ring before exiting. Services with no
///     work return immediately so other workgroups don't block.
///
///   * Cross-service ring writes use the v0.30 release pattern:
///     write items[tail], threadgroup_barrier(mem_flags::mem_device),
///     atomic_store(tail). Cross-service ring reads pair the relaxed
///     atomic_load(tail) with another mem_device barrier before the
///     items[head] load, so producer writes are visible.
///
///   * No hot-spinning. When a service is empty the workgroup exits.
///     This is the v0.29 starvation fix: workgroups never hold the
///     compute units waiting for new work — the host pumps the kernel
///     once per epoch and the GPU scheduler reschedules cleanly each
///     time.

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Layout — must match lib/evm/gpu/slot/slot_layout.hpp byte-for-byte
// =============================================================================

constant uint kNumServices = 8u;

struct alignas(16) RingHeader {
    atomic_uint head;
    atomic_uint tail;
    uint        capacity;
    uint        mask;
    ulong       items_ofs;
    uint        item_size;
    uint        _pad0;
    atomic_uint pushed;
    atomic_uint consumed;
    uint        _pad1;
    uint        _pad2;
};

struct alignas(16) SlotDescriptor {
    ulong chain_id;
    ulong slot;
    ulong timestamp_ns;
    ulong deadline_ns;
    ulong gas_limit;
    ulong base_fee;
    uint  epoch_budget_items;
    uint  epoch_index;
    uint  closing_flag;
    uint  _pad0;
};

struct alignas(16) SlotResult {
    atomic_uint status;
    atomic_uint tx_count;
    atomic_uint gas_used_lo;
    atomic_uint gas_used_hi;
    atomic_uint epoch_count;
    uint        _pad0;
    ulong       _pad1;
    uchar block_hash[32];
    uchar state_root[32];
    uchar receipts_root[32];
};

struct alignas(16) IngressTx {
    uint  blob_offset;
    uint  blob_size;
    ulong gas_limit;
    uint  nonce;
    uint  _pad0;
    uint  origin_lo;
    uint  origin_hi;
};

struct alignas(16) DecodedTx {
    uint  tx_index;
    uint  blob_offset;
    uint  blob_size;
    ulong gas_limit;
    uint  nonce;
    uint  origin_lo;
    uint  origin_hi;
    uint  status;
};

struct alignas(16) VerifiedTx {
    uint  tx_index;
    uint  admission;
    ulong gas_limit;
    uint  origin_lo;
    uint  origin_hi;
    ulong _pad0;
};

struct alignas(16) CommitItem {
    uint  tx_index;
    uint  status;
    ulong gas_used;
    ulong cumulative_gas;
    ulong _pad0;
};

// =============================================================================
// Ring helpers — relaxed atomics + threadgroup_barrier(mem_device).
// Same pattern proven in v3_persistent.metal (v0.30 wave-dispatch).
// =============================================================================

template<typename T>
static inline bool ring_try_push(device RingHeader* h,
                                 device T* items,
                                 thread const T& v)
{
    uint head = atomic_load_explicit(&h->head, memory_order_relaxed);
    uint tail = atomic_load_explicit(&h->tail, memory_order_relaxed);
    if (tail - head >= h->capacity) {
        return false;  // ring full — caller retries next epoch
    }
    items[tail & h->mask] = v;
    threadgroup_barrier(mem_flags::mem_device);
    atomic_store_explicit(&h->tail, tail + 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(&h->pushed, 1u, memory_order_relaxed);
    return true;
}

template<typename T>
static inline bool ring_try_pop(device RingHeader* h,
                                device T* items,
                                thread T& out)
{
    while (true) {
        uint head = atomic_load_explicit(&h->head, memory_order_relaxed);
        uint tail = atomic_load_explicit(&h->tail, memory_order_relaxed);
        if (head >= tail) {
            return false;
        }
        uint expected = head;
        if (atomic_compare_exchange_weak_explicit(
                &h->head, &expected, head + 1u,
                memory_order_relaxed, memory_order_relaxed))
        {
            threadgroup_barrier(mem_flags::mem_device);
            out = items[head & h->mask];
            atomic_fetch_add_explicit(&h->consumed, 1u, memory_order_relaxed);
            return true;
        }
        // CAS lost — retry with fresh head value.
    }
}

// =============================================================================
// Per-service drain functions. Each returns the number of items processed
// in this epoch. The scheduler stops the workgroup once the budget is hit.
// =============================================================================

static inline uint drain_ingress(
    device RingHeader* ingress_hdr, device IngressTx* ingress_items,
    device RingHeader* decode_hdr,  device DecodedTx*  decode_items,
    device atomic_uint* tx_index_seq,
    uint budget)
{
    uint processed = 0u;
    for (uint i = 0u; i < budget; ++i) {
        IngressTx in;
        if (!ring_try_pop(ingress_hdr, ingress_items, in))
            break;
        DecodedTx out;
        out.tx_index    = atomic_fetch_add_explicit(tx_index_seq, 1u,
                                                     memory_order_relaxed);
        out.blob_offset = in.blob_offset;
        out.blob_size   = in.blob_size;
        out.gas_limit   = in.gas_limit;
        out.nonce       = in.nonce;
        out.origin_lo   = in.origin_lo;
        out.origin_hi   = in.origin_hi;
        out.status      = 0u;
        if (!ring_try_push(decode_hdr, decode_items, out)) {
            // Decode ring full — spill back: re-push to ingress so we
            // don't lose the tx. Backpressure flows naturally.
            (void)ring_try_push(ingress_hdr, ingress_items, in);
            break;
        }
        ++processed;
    }
    return processed;
}

static inline uint drain_decode(
    device RingHeader* decode_hdr, device DecodedTx*  decode_items,
    device RingHeader* crypto_hdr, device VerifiedTx* crypto_items,
    uint budget)
{
    // v0.31 stub: no real sender recovery yet. We pass origin straight
    // through; the secp256k1 step lands here in v0.32 (the existing
    // metal_op_ecrecover_batch can be invoked over the slot's input arena).
    uint processed = 0u;
    for (uint i = 0u; i < budget; ++i) {
        DecodedTx d;
        if (!ring_try_pop(decode_hdr, decode_items, d))
            break;
        VerifiedTx v;
        v.tx_index  = d.tx_index;
        v.admission = (d.status == 0u) ? 0u : 1u;  // 0=admit
        v.gas_limit = d.gas_limit;
        v.origin_lo = d.origin_lo;
        v.origin_hi = d.origin_hi;
        if (!ring_try_push(crypto_hdr, crypto_items, v)) {
            (void)ring_try_push(decode_hdr, decode_items, d);
            break;
        }
        ++processed;
    }
    return processed;
}

static inline uint drain_crypto(
    device RingHeader* crypto_hdr, device VerifiedTx* crypto_items,
    device RingHeader* commit_hdr, device CommitItem* commit_items,
    uint budget)
{
    // v0.31 stub: admission accepts everything; a tx that passed
    // decode + recovery becomes committable. The Block-STM scheduler +
    // EVM fiber VM (v0.32+) will replace this passthrough with real
    // execution. The contract this stub asserts is that the slot
    // pipeline preserves count and ordering.
    uint processed = 0u;
    for (uint i = 0u; i < budget; ++i) {
        VerifiedTx v;
        if (!ring_try_pop(crypto_hdr, crypto_items, v))
            break;
        if (v.admission != 0u) {
            ++processed;
            continue;  // dropped by admission
        }
        CommitItem c;
        c.tx_index       = v.tx_index;
        c.status         = 1u;       // TxStatus::Return
        c.gas_used       = 21000u;   // synthetic until EVM lands
        c.cumulative_gas = 0u;       // commit stage fills this in
        if (!ring_try_push(commit_hdr, commit_items, c)) {
            (void)ring_try_push(crypto_hdr, crypto_items, v);
            break;
        }
        ++processed;
    }
    return processed;
}

static inline uint drain_commit(
    device RingHeader* commit_hdr, device CommitItem* commit_items,
    device SlotResult* result,
    uint budget)
{
    uint processed = 0u;
    for (uint i = 0u; i < budget; ++i) {
        CommitItem c;
        if (!ring_try_pop(commit_hdr, commit_items, c))
            break;
        // Finalize: bump tx_count + gas_used. The block_hash / state_root
        // / receipts_root remain placeholder until v0.34 wires keccak on
        // the slot arena. Counters, however, are correct end-to-end.
        atomic_fetch_add_explicit(&result->tx_count, 1u, memory_order_relaxed);
        // Split 64-bit gas accumulation: bump lo, propagate carry to hi
        // when it rolls over. MSL has no 64-bit atomic_fetch_add.
        const uint gas_lo = uint(c.gas_used & 0xFFFFFFFFu);
        const uint gas_hi = uint((c.gas_used >> 32) & 0xFFFFFFFFu);
        const uint prev = atomic_fetch_add_explicit(&result->gas_used_lo,
                                                    gas_lo,
                                                    memory_order_relaxed);
        if (gas_hi != 0u) {
            atomic_fetch_add_explicit(&result->gas_used_hi, gas_hi,
                                      memory_order_relaxed);
        }
        if (prev + gas_lo < prev) {  // unsigned wrap = carry into hi
            atomic_fetch_add_explicit(&result->gas_used_hi, 1u,
                                      memory_order_relaxed);
        }
        ++processed;
    }
    return processed;
}

// =============================================================================
// slot_kernel — one kernel grid per epoch.
//
// Layout:
//   buffer 0   slot descriptor (1 element, read-only this epoch)
//   buffer 1   slot result     (1 element, atomically updated)
//   buffer 2   ring headers    (kNumServices RingHeader records)
//   buffer 3   ring items arena (typed views chosen via per-service offset)
//   buffer 4   tx_index sequence atomic counter
//
// Threadgroup grid:
//   N = kNumServices (one workgroup per service)
//   T = 32 lanes per workgroup, only tid==0 does work — leaves lanes for
//   the future SIMT EVM interpreter to fan out per-tx instruction streams.
// =============================================================================

kernel void slot_scheduler_kernel(
    device const SlotDescriptor* desc          [[buffer(0)]],
    device SlotResult*           result        [[buffer(1)]],
    device RingHeader*           hdrs          [[buffer(2)]],
    device uchar*                items_arena   [[buffer(3)]],
    device atomic_uint*          tx_index_seq  [[buffer(4)]],
    uint   tid                                  [[thread_index_in_threadgroup]],
    uint   gid                                  [[threadgroup_position_in_grid]])
{
    if (tid != 0u) return;
    if (gid >= kNumServices) return;

    // The kernel is launched fresh each epoch; bump the counter for the
    // host-side stats. Only one workgroup needs to do this.
    if (gid == 0u) {
        atomic_fetch_add_explicit(&result->epoch_count, 1u,
                                  memory_order_relaxed);
    }

    // Each workgroup processes its assigned service for this epoch.
    const uint budget = max(uint(64), desc->epoch_budget_items);

    // Derive per-service ring views from the headers' items_ofs fields.
    // Items are interleaved in one MTLBuffer; the host computed offsets
    // when allocating the arena.
    device RingHeader* ingress_hdr = hdrs + 0;
    device RingHeader* decode_hdr  = hdrs + 1;
    device RingHeader* crypto_hdr  = hdrs + 2;
    device RingHeader* commit_hdr  = hdrs + 3;

    device IngressTx*  ingress_items = (device IngressTx*) (items_arena + ingress_hdr->items_ofs);
    device DecodedTx*  decode_items  = (device DecodedTx*) (items_arena + decode_hdr->items_ofs);
    device VerifiedTx* crypto_items  = (device VerifiedTx*)(items_arena + crypto_hdr->items_ofs);
    device CommitItem* commit_items  = (device CommitItem*)(items_arena + commit_hdr->items_ofs);

    if (gid == uint(0)) {
        (void)drain_ingress(ingress_hdr, ingress_items,
                            decode_hdr, decode_items,
                            tx_index_seq, budget);
    } else if (gid == uint(1)) {
        (void)drain_decode(decode_hdr, decode_items,
                           crypto_hdr, crypto_items,
                           budget);
    } else if (gid == uint(2)) {
        (void)drain_crypto(crypto_hdr, crypto_items,
                           commit_hdr, commit_items,
                           budget);
    } else if (gid == uint(3)) {
        (void)drain_commit(commit_hdr, commit_items, result, budget);
    } else {
        // ServiceId::StateRequest, StateResp, Vote, QuorumOut land in
        // v0.32+ — for v0.31 these workgroups exit immediately.
    }

    // Finalization. The naive "all rings empty" check fires too early —
    // workgroups run concurrently and a tx can be in transit between
    // stages when WG0 samples. The robust invariant is end-to-end count
    // equality: every tx that entered ingress reached commit. Only one
    // workgroup writes the status to avoid interleaving.
    if (gid == 0u && desc->closing_flag != 0u) {
        const uint ingress_pushed = atomic_load_explicit(
            &hdrs[0].pushed, memory_order_relaxed);
        const uint commit_consumed = atomic_load_explicit(
            &hdrs[3].consumed, memory_order_relaxed);
        if (ingress_pushed == commit_consumed) {
            atomic_store_explicit(&result->status, 1u, memory_order_relaxed);
        }
    }
}
