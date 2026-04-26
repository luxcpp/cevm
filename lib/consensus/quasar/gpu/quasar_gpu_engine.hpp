// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file quasar_gpu_engine.hpp
/// QuasarGPU — shared GPU substrate for the Quasar consensus family.
///
/// Underneath:
///   * Nova   (linear) — protocol/nova in luxfi/consensus
///   * Nebula (DAG)    — protocol/nebula in luxfi/consensus
///
/// Inside one round:
///   * Block-STM execution + validate + repair (v0.33)
///   * GPU keccak for block_hash, state_root, receipts_root (v0.34)
///   * Predicted-access-set DAG ready set (v0.35; Nebula mode)
///   * EVM fiber VM with cold-state suspend/resume (v0.36)
///   * BLS / ML-DSA / Ringtail vote ingestion + QC emission (v0.37)
///
/// The host's job is reduced to:
///   begin_round(QuasarRoundDescriptor)        # once per consensus round
///   push_txs / push_votes / push_state_pages  # ingress doorbells
///   poll_state_requests                       # service GPU-emitted faults
///   run_epoch / run_until_done                # bounded scheduler ticks
///   poll_round_result                         # finalized hash + QC

#pragma once

#include "quasar_gpu_layout.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace quasar::gpu {

struct QuasarRoundHandle {
    uint64_t opaque = 0;
    bool valid() const { return opaque != 0; }
};

struct HostTxBlob {
    std::vector<uint8_t> bytes;
    uint64_t gas_limit = 0;
    uint32_t nonce = 0;
    uint64_t origin = 0;
    bool needs_state = false;
    bool needs_exec  = false;       ///< route through Exec/Validate/Commit
                                    ///< full Block-STM pipeline rather
                                    ///< than the fast Crypto→Commit lane
    uint64_t exec_key_lo = 0;       ///< MVCC key the tx reads+writes
    uint64_t exec_key_hi = 0;       ///< (v0.36 substrate; full EVM fiber
                                    ///< replaces this with bytecode-driven
                                    ///< RW set generation in v0.39)

    /// Optional: predicted read/write keys for DAG construction (Nebula).
    /// Empty vector → tx is conservatively scheduled as touching all
    /// hot keys. Populating these lets the DAG service form parallel
    /// antichains.
    struct PredictedAccess {
        uint64_t key_lo;
        uint64_t key_hi;
        bool is_write;
    };
    std::vector<PredictedAccess> predicted_access;
};

struct HostStateRequest {
    uint32_t tx_index = 0;
    uint32_t key_type = 0;
    uint32_t priority = 0;
    uint64_t key_lo = 0;
    uint64_t key_hi = 0;
};

struct HostStatePage {
    uint32_t tx_index = 0;
    uint32_t key_type = 0;
    uint32_t status = 0;
    uint64_t key_lo = 0;
    uint64_t key_hi = 0;
    std::vector<uint8_t> data;
};

struct HostVote {
    uint32_t validator_index = 0;
    uint32_t round = 0;
    uint32_t stake_weight = 0;
    uint32_t sig_kind = 0;        ///< 0=BLS, 1=ML-DSA, 2=Ringtail
    uint8_t  block_hash[32]{};
    std::vector<uint8_t> signature;
};

struct HostQuorumCert {
    uint32_t round = 0;
    uint32_t status = 0;
    uint32_t signers_count = 0;
    uint32_t total_stake = 0;
    uint8_t  block_hash[32]{};
    std::vector<uint8_t> agg_signature;
};

class QuasarGPUEngine {
public:
    virtual ~QuasarGPUEngine() = default;

    static std::unique_ptr<QuasarGPUEngine> create();

    virtual QuasarRoundHandle begin_round(const QuasarRoundDescriptor& desc) = 0;

    virtual void push_txs(QuasarRoundHandle h,
                          std::span<const HostTxBlob> txs) = 0;

    virtual void push_votes(QuasarRoundHandle h,
                            std::span<const HostVote> votes) = 0;

    virtual std::vector<HostStateRequest>
        poll_state_requests(QuasarRoundHandle h) = 0;

    virtual void push_state_pages(QuasarRoundHandle h,
                                  std::span<const HostStatePage> pages) = 0;

    virtual std::vector<HostQuorumCert>
        poll_quorum_certs(QuasarRoundHandle h) = 0;

    virtual QuasarRoundResult run_epoch(QuasarRoundHandle h) = 0;

    virtual QuasarRoundResult run_until_done(QuasarRoundHandle h,
                                             std::size_t max_epochs = 1024) = 0;

    virtual QuasarRoundResult poll_round_result(QuasarRoundHandle h) const = 0;

    virtual void request_close(QuasarRoundHandle h) = 0;

    virtual void end_round(QuasarRoundHandle h) = 0;

    virtual bool round_active() const = 0;

    virtual const char* device_name() const = 0;

    struct RingStats {
        uint32_t pushed = 0;
        uint32_t consumed = 0;
        uint32_t head = 0;
        uint32_t tail = 0;
        uint32_t capacity = 0;
    };
    virtual RingStats ring_stats(QuasarRoundHandle h, ServiceId s) const = 0;

protected:
    QuasarGPUEngine() = default;
    QuasarGPUEngine(const QuasarGPUEngine&) = delete;
    QuasarGPUEngine& operator=(const QuasarGPUEngine&) = delete;
};

}  // namespace quasar::gpu
