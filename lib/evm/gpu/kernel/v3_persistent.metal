// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file v3_persistent.metal
/// V3 persistent kernels — the queue-driven pipeline.
///
/// Three kernels, all persistent (single launch covers the lifetime of a
/// V3PersistentRunner):
///   * exec_worker     — pulls (tx_idx, incarnation) off exec_q,
///                       runs the interpreter, pushes onto validate_q
///   * validate_worker — pulls off validate_q, confirms read-set, pushes
///                       onto commit_q
///   * commit_worker   — pulls off commit_q, marks tx committed
///
/// All three spin on `V3Control::shutdown_flag`; once flipped they drain
/// the queue they're servicing and return.
///
/// Metal MSL note: only memory_order_relaxed is exposed on device atomics.
/// We rely on:
///   * atomic_compare_exchange_weak_explicit for queue-pop CAS — failures
///     just retry, so the visibility of the loser's read doesn't matter
///   * Apple Silicon's unified memory + the fact that the host's
///     [cmd commit] establishes a release fence visible to the kernel,
///     and the kernel's writes become visible to the host on
///     [cmd waitUntilCompleted] — so cross-kernel ordering is structural
///   * For inter-kernel ordering (exec → validate → commit) the CAS
///     ensures a workitem isn't double-consumed; the data dependency
///     (validate reads results[w.tx_index] written by exec) is honored
///     via the queue itself: a slot is only enqueued AFTER its result is
///     written
///
/// v0.29 scope: end-to-end queue mechanics with a simplified executor that
/// proves persistence + pipelining + backpressure + shutdown. The full
/// EVM interpreter (matching evm_kernel.metal) is wired in v0.30 alongside
/// MVCC. The simplified executor returns deterministic synthetic results
/// keyed off code_size, so V3 tests can assert the exact value seen at
/// commit time without depending on the legacy evm_execute opcode set.

#include <metal_stdlib>
using namespace metal;

// -----------------------------------------------------------------------------
// Layout — matches lib/evm/gpu/kernel/v3_queue.hpp
// -----------------------------------------------------------------------------

constant uint Q_LOG2_CAPACITY = 14u;
constant uint Q_CAPACITY      = 1u << Q_LOG2_CAPACITY;
constant uint Q_MASK          = Q_CAPACITY - 1u;

struct WorkItem {
    uint tx_index;
    uint incarnation;
    uint wave_id;
    uint flags;
};

struct alignas(16) QueueHeader {
    atomic_uint head;
    atomic_uint tail;
    uint        mask;
    uint        _pad0;
};

struct alignas(16) V3Control {
    atomic_uint shutdown_flag;
    atomic_uint exec_alive;
    atomic_uint validate_alive;
    atomic_uint commit_alive;

    atomic_uint exec_done;
    atomic_uint validate_done;
    atomic_uint commit_done;
    uint        _pad0;
};

// -----------------------------------------------------------------------------
// Per-tx slots in unified memory.
// -----------------------------------------------------------------------------

struct V3TxInput {
    uint  code_size;
    uint  code_offset;
    uint  calldata_size;
    uint  calldata_offset;
    ulong gas_limit;
    uint  read_set_size;
    uint  read_set_offset;
    uint  _pad0;
    uint  _pad1;
};

struct V3TxResult {
    uint  status;
    ulong gas_used;
    uint  output_size;
    uint  _pad0;
};

// -----------------------------------------------------------------------------
// Queue helpers — Metal MSL only supports memory_order_relaxed on device
// atomics. The CAS pop relies on the loser-retries pattern; visibility
// across kernels is established by Apple Silicon's unified-memory model
// and the host's command-buffer dependency between [cmd commit] /
// [cmd waitUntilCompleted].
// -----------------------------------------------------------------------------

static inline bool q_push(device QueueHeader* q,
                          device WorkItem*    items,
                          WorkItem            w)
{
    uint head = atomic_load_explicit(&q->head, memory_order_relaxed);
    uint tail = atomic_load_explicit(&q->tail, memory_order_relaxed);
    if (tail - head >= Q_CAPACITY) {
        return false;  // full
    }
    items[tail & Q_MASK] = w;
    atomic_store_explicit(&q->tail, tail + 1u, memory_order_relaxed);
    return true;
}

static inline bool q_pop(device QueueHeader* q,
                         device WorkItem*    items,
                         thread WorkItem&    out)
{
    while (true) {
        uint head = atomic_load_explicit(&q->head, memory_order_relaxed);
        uint tail = atomic_load_explicit(&q->tail, memory_order_relaxed);
        if (head >= tail) {
            return false;
        }
        uint expected = head;
        if (atomic_compare_exchange_weak_explicit(
                &q->head, &expected, head + 1u,
                memory_order_relaxed, memory_order_relaxed))
        {
            out = items[head & Q_MASK];
            return true;
        }
        // CAS lost — retry.
    }
}

static inline bool shutdown_set(device V3Control* ctl)
{
    return atomic_load_explicit(&ctl->shutdown_flag,
                                memory_order_relaxed) != 0u;
}

static inline void exec_one_tx(device const V3TxInput& in,
                               device V3TxResult&      r)
{
    r.status = 1u;  // Return
    r.gas_used = ulong(in.code_size) * 21u + 21000u;
    r.output_size = (in.code_size < 32u) ? in.code_size : 32u;
}

// -----------------------------------------------------------------------------
// exec_worker — drains exec_q, runs exec_one_tx, pushes onto validate_q.
// -----------------------------------------------------------------------------

kernel void v3_exec_worker(
    device QueueHeader*       exec_q       [[buffer(0)]],
    device WorkItem*          exec_items   [[buffer(1)]],
    device QueueHeader*       validate_q   [[buffer(2)]],
    device WorkItem*          validate_items[[buffer(3)]],
    device V3Control*         ctl          [[buffer(4)]],
    device const V3TxInput*   inputs       [[buffer(5)]],
    device V3TxResult*        results      [[buffer(6)]],
    uint   tid                              [[thread_index_in_threadgroup]],
    uint   gid                              [[threadgroup_position_in_grid]])
{
    if (tid != 0) return;

    atomic_fetch_add_explicit(&ctl->exec_alive, 1u, memory_order_relaxed);

    bool keep_going = true;
    uint backoff = 0u;
    while (keep_going) {
        WorkItem w;
        if (q_pop(exec_q, exec_items, w)) {
            exec_one_tx(inputs[w.tx_index], results[w.tx_index]);
            atomic_fetch_add_explicit(&ctl->exec_done, 1u, memory_order_relaxed);

            // Push downstream — spin if validate_q is full unless we're shutting down.
            bool pushed = false;
            while (!pushed) {
                pushed = q_push(validate_q, validate_items, w);
                if (!pushed && shutdown_set(ctl)) {
                    keep_going = false;
                    break;
                }
            }
            backoff = 0u;
            continue;
        }
        // Empty queue — exit if shutdown signalled.
        if (shutdown_set(ctl)) {
            keep_going = false;
            break;
        }
        backoff = (backoff < 64u) ? (backoff + 1u) : 64u;
        for (uint i = 0; i < backoff; ++i) {
            (void)atomic_load_explicit(&exec_q->head, memory_order_relaxed);
        }
    }

    atomic_fetch_sub_explicit(&ctl->exec_alive, 1u, memory_order_relaxed);
}

// -----------------------------------------------------------------------------
// validate_worker — empty read-set ⇒ pass to commit_q. v0.30 wires MVCC.
// -----------------------------------------------------------------------------

kernel void v3_validate_worker(
    device QueueHeader*       validate_q     [[buffer(0)]],
    device WorkItem*          validate_items [[buffer(1)]],
    device QueueHeader*       commit_q       [[buffer(2)]],
    device WorkItem*          commit_items   [[buffer(3)]],
    device V3Control*         ctl            [[buffer(4)]],
    device const V3TxInput*   inputs         [[buffer(5)]],
    device V3TxResult*        results        [[buffer(6)]],
    uint   tid                                [[thread_index_in_threadgroup]],
    uint   gid                                [[threadgroup_position_in_grid]])
{
    if (tid != 0) return;

    atomic_fetch_add_explicit(&ctl->validate_alive, 1u, memory_order_relaxed);

    bool keep_going = true;
    uint backoff = 0u;
    while (keep_going) {
        WorkItem w;
        if (q_pop(validate_q, validate_items, w)) {
            const device V3TxInput& in = inputs[w.tx_index];
            if (in.read_set_size != 0u) {
                // Non-empty read set — flag as Error. v0.30 will re-enqueue.
                results[w.tx_index].status = 4u;
            }
            atomic_fetch_add_explicit(&ctl->validate_done, 1u, memory_order_relaxed);

            bool pushed = false;
            while (!pushed) {
                pushed = q_push(commit_q, commit_items, w);
                if (!pushed && shutdown_set(ctl)) {
                    keep_going = false;
                    break;
                }
            }
            backoff = 0u;
            continue;
        }
        if (shutdown_set(ctl)) {
            keep_going = false;
            break;
        }
        backoff = (backoff < 64u) ? (backoff + 1u) : 64u;
        for (uint i = 0; i < backoff; ++i) {
            (void)atomic_load_explicit(&validate_q->head, memory_order_relaxed);
        }
    }

    atomic_fetch_sub_explicit(&ctl->validate_alive, 1u, memory_order_relaxed);
}

// -----------------------------------------------------------------------------
// commit_worker — drains commit_q, flips per-tx committed[] flag.
// -----------------------------------------------------------------------------

kernel void v3_commit_worker(
    device QueueHeader*       commit_q       [[buffer(0)]],
    device WorkItem*          commit_items   [[buffer(1)]],
    device V3Control*         ctl            [[buffer(2)]],
    device atomic_uint*       committed      [[buffer(3)]],
    uint   tid                                [[thread_index_in_threadgroup]],
    uint   gid                                [[threadgroup_position_in_grid]])
{
    if (tid != 0) return;

    atomic_fetch_add_explicit(&ctl->commit_alive, 1u, memory_order_relaxed);

    bool keep_going = true;
    uint backoff = 0u;
    while (keep_going) {
        WorkItem w;
        if (q_pop(commit_q, commit_items, w)) {
            atomic_store_explicit(&committed[w.tx_index], 1u,
                                  memory_order_relaxed);
            atomic_fetch_add_explicit(&ctl->commit_done, 1u,
                                      memory_order_relaxed);
            backoff = 0u;
            continue;
        }
        if (shutdown_set(ctl)) {
            keep_going = false;
            break;
        }
        backoff = (backoff < 64u) ? (backoff + 1u) : 64u;
        for (uint i = 0; i < backoff; ++i) {
            (void)atomic_load_explicit(&commit_q->head, memory_order_relaxed);
        }
    }

    atomic_fetch_sub_explicit(&ctl->commit_alive, 1u, memory_order_relaxed);
}
