// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file slot_layout.hpp
/// Shared host/GPU memory layouts for the GPU Slot Engine (v0.31+).
///
/// Architecture in one sentence: the GPU owns the slot-local control plane
/// (admission, decode, recovery, DAG, Block-STM scheduling, EVM execution,
/// validate, repair, commit, root, vote, quorum). The CPU only provides
/// ingress/egress (network packets, cold-state pages) and re-launches the
/// scheduler kernel once per epoch until slot_result is final.
///
/// This header defines the byte-for-byte layout shared between the host
/// driver (slot_engine.mm) and the device kernel (slot_kernel.metal).
/// Every offset and field count below MUST match the corresponding MSL
/// declaration. static_assert covers what cmake can't enforce.

#pragma once

#include <cstdint>

namespace evm::gpu::slot {

// =============================================================================
// Service identifiers
// =============================================================================

/// Service IDs route work items through the slot scheduler. Each ID owns one
/// device ring; workgroups pick a service to advance via the priority
/// scheduler. Add new services at the end — never reorder or recycle IDs.
enum class ServiceId : uint32_t {
    Ingress      = 0,  ///< raw tx blobs from the host
    Decode       = 1,  ///< decoded txs awaiting sender recovery
    Crypto       = 2,  ///< sig-verified txs awaiting admission
    Commit       = 3,  ///< committable txs awaiting receipt + finalize
    StateRequest = 4,  ///< GPU-emitted cold-state requests (out)
    StateResp    = 5,  ///< host-emitted cold-state responses (in)
    Vote         = 6,  ///< raw consensus votes from the host
    QuorumOut    = 7,  ///< GPU-emitted quorum certs (out)
    Count        = 8
};

inline constexpr uint32_t kNumServices = static_cast<uint32_t>(ServiceId::Count);

// =============================================================================
// Device-resident ring buffer
// =============================================================================
//
// Single-producer-single-consumer at a service granularity (one ring per
// service). Internally MPMC across workgroups via relaxed atomics on
// head/tail; the threadgroup_barrier(mem_flags::mem_device) trick from V3
// makes non-atomic items[] writes visible across workgroups.
//
// Layout: [ RingHeader | items[capacity] ] in one MTLBuffer.
// The driver computes per-service offsets and binds each ring as an offset
// view into a single arena buffer.

struct alignas(16) RingHeader {
    uint32_t head;       ///< next index to consume
    uint32_t tail;       ///< next index to produce
    uint32_t capacity;   ///< power of two
    uint32_t mask;       ///< capacity - 1
    uint64_t items_ofs;  ///< byte offset to items[] from this header
    uint32_t item_size;  ///< bytes per item
    uint32_t _pad0;
    uint32_t pushed;     ///< monotonic producer counter (debug + tests)
    uint32_t consumed;   ///< monotonic consumer counter (debug + tests)
    uint32_t _pad1;
    uint32_t _pad2;
};

static_assert(sizeof(RingHeader) == 48, "RingHeader layout drift");

// =============================================================================
// Per-service ring records
// =============================================================================
//
// Item shapes for v0.31. These are intentionally small — the EVM runtime
// fibers are not yet on the device, so the items model the tx envelope and
// the receipt envelope. v0.32+ adds DagNode, TxFiber, CommitItem with the
// MVCC arena.

struct alignas(16) IngressTx {
    uint32_t blob_offset;   ///< into the per-slot tx_blob_arena
    uint32_t blob_size;
    uint64_t gas_limit;
    uint32_t nonce;
    uint32_t _pad0;
    uint32_t origin_lo;     ///< first 8 bytes of origin address (host-side ID)
    uint32_t origin_hi;
};

static_assert(sizeof(IngressTx) == 32, "IngressTx layout drift");

struct alignas(16) DecodedTx {
    uint32_t tx_index;      ///< stable index into the slot's input arena
    uint32_t blob_offset;
    uint32_t blob_size;
    uint64_t gas_limit;
    uint32_t nonce;
    uint32_t origin_lo;
    uint32_t origin_hi;
    uint32_t status;        ///< 0=ok, !=0=decode-failed enum
};

static_assert(sizeof(DecodedTx) == 48, "DecodedTx layout drift");

struct alignas(16) VerifiedTx {
    uint32_t tx_index;
    uint32_t admission;     ///< 0=admit, !=0=reject reason
    uint64_t gas_limit;
    uint32_t origin_lo;     ///< sender (here: pass-through of origin until
    uint32_t origin_hi;     ///< secp256k1 lands inside slot kernel)
    uint64_t _pad0;
};

static_assert(sizeof(VerifiedTx) == 32, "VerifiedTx layout drift");

struct alignas(16) CommitItem {
    uint32_t tx_index;
    uint32_t status;        ///< 1=Return, 2=Revert, 3=OutOfGas, 4=Error
    uint64_t gas_used;
    uint64_t cumulative_gas;
    uint64_t _pad0;
};

static_assert(sizeof(CommitItem) == 32, "CommitItem layout drift");

// =============================================================================
// Cold-state page-fault rings (v0.32)
// =============================================================================
//
// A tx that misses GPU-resident state emits a StateRequest and suspends.
// The host drains the request ring, services it via mmap/LSM/cache, and
// pushes a StatePage onto the response ring. The slot kernel then wakes
// the suspended tx fiber and re-runs it from the suspend point.
//
// v0.32 ships the ring infrastructure + host APIs end-to-end so we can
// validate the page-fault round-trip. Full fiber suspension/resume lives
// in v0.36 alongside the on-device EVM interpreter; until then a tx is
// classified as "needs_state" by a host-supplied flag (HostTxBlob::
// needs_state) and the kernel emits a request on its behalf at the
// decode stage.

enum class StateKeyType : uint32_t {
    Account = 0,
    Storage = 1,
    Code    = 2,
};

struct alignas(16) StateRequest {
    uint32_t tx_index;
    uint32_t key_type;       ///< StateKeyType
    uint32_t priority;       ///< 0=normal, 1=deadline-critical
    uint32_t _pad0;
    uint64_t key_lo;         ///< first half of the 16-byte key digest
    uint64_t key_hi;         ///< second half
};

static_assert(sizeof(StateRequest) == 32, "StateRequest layout drift");

struct alignas(16) StatePage {
    uint32_t tx_index;       ///< matches the request
    uint32_t key_type;
    uint32_t status;         ///< 0=ok, 1=missing, 2=fault
    uint32_t data_size;      ///< up to 4 KiB inline; larger pages out-of-band
    uint64_t key_lo;
    uint64_t key_hi;
    uint8_t  data[64];       ///< inline payload — accounts/storage slots fit
};

static_assert(sizeof(StatePage) == 96, "StatePage layout drift");

// =============================================================================
// Slot descriptor (host -> GPU, written once per slot)
// =============================================================================

struct alignas(16) SlotDescriptor {
    uint64_t chain_id;
    uint64_t slot;
    uint64_t timestamp_ns;
    uint64_t deadline_ns;       ///< wall time at which the slot must close
    uint64_t gas_limit;
    uint64_t base_fee;
    uint32_t epoch_budget_items;///< hint: max items each service drains per
                                ///< epoch. The kernel uses this to bound
                                ///< per-workgroup work and yield politely
                                ///< back to the GPU scheduler — directly
                                ///< addressing the v0.29 starvation issue.
    uint32_t epoch_index;       ///< monotonic across re-launches in this slot
    uint32_t closing_flag;      ///< host sets to 1 to request a clean exit
                                ///< after the next epoch
    uint32_t _pad0;
};

static_assert(sizeof(SlotDescriptor) == 64, "SlotDescriptor layout drift");

// =============================================================================
// Slot result (GPU -> host)
// =============================================================================

struct alignas(16) SlotResult {
    uint32_t status;            ///< 0=in-progress, 1=finalized, 2=needs_state, 3=failed
    uint32_t tx_count;          ///< number of committed txs
    uint32_t gas_used_lo;       ///< low 32 bits of total gas (MSL has no 64-bit atomics)
    uint32_t gas_used_hi;       ///< high 32 bits — bumped via CAS rollover
    uint32_t epoch_count;       ///< how many epochs the kernel ran
    uint32_t _pad0;
    uint64_t _pad1;
    uint8_t  block_hash[32];    ///< placeholder until v0.34 wires keccak
    uint8_t  state_root[32];    ///< placeholder until v0.34
    uint8_t  receipts_root[32]; ///< placeholder until v0.34

    /// Helper for callers — recombine the split gas counter.
    uint64_t gas_used() const {
        return (uint64_t(gas_used_hi) << 32) | uint64_t(gas_used_lo);
    }
};

static_assert(sizeof(SlotResult) == 128, "SlotResult layout drift");

}  // namespace evm::gpu::slot
