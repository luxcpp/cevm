// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file unified_pipeline.hpp
/// Unified GPU pipeline: EVM + consensus state hash + post-quantum crypto on
/// one device, sharing a single LuxGPU handle (and on Apple, a single MTLDevice).
///
/// Composes the existing primitives:
///   - evm::gpu::metal::BlockStmGpu       — Block-STM EVM execution
///   - evm::gpu::GpuStateHasher            — Keccak-256 batch (state root)
///   - evm::gpu::metal::BlsVerifier        — BLS12-381 batch verify
///   - lux_gpu_keccak256_batch             — Quasar consensus block hash
///   - lux_gpu_mldsa_verify_batch          — ML-DSA-65 lattice signature
///                                            verify (Quasar finality round 2)
///
/// Pipelining: when invoked with N blocks, the host overlaps stages so that
/// block N+1's signature verification runs concurrently with block N's EVM
/// execution. All GPU work targets the same Metal device, so the MTL command
/// queues serialize naturally without inter-queue fences. CPU-side host work
/// (buffer prep, result copy) overlaps with GPU work via std::async.
///
/// Usage:
///   auto pipe = evm::gpu::unified::UnifiedPipeline::create();
///   auto result = pipe->process_block(txs, accounts, ctx);
///
/// This is built NEXT TO the legacy GpuPipeline (pipeline.hpp). The legacy
/// type is unchanged for backward compatibility.

#pragma once

#include "gpu_dispatch.hpp"  // Transaction, BlockResult
#include "pipeline.hpp"      // AccountInfo

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace evm::gpu::unified {

// -- Public configuration -----------------------------------------------------

/// Configuration knobs for the unified pipeline.
struct UnifiedConfig {
    /// Maximum number of blocks in flight. Pipeline depth.
    /// The Metal command queue serializes GPU work; this controls how much
    /// CPU host work can overlap with GPU execution.
    uint32_t max_concurrent_blocks = 4;

    /// If true, compute the post-block state root (Keccak-256 over modified
    /// accounts).
    bool enable_state_root = true;

    /// If true, compute the consensus block hash (Quasar's canonical block
    /// hash = Keccak-256 of the RLP-encoded header).
    bool enable_consensus_hash = true;

    /// If true, batch-verify any signatures attached to the block context
    /// (BLS12-381 by default; ML-DSA if pq_signatures is set).
    bool enable_signature_verify = true;
};

/// Per-block input metadata that is not part of the EVM transaction list.
/// Roughly mirrors what Lux Quasar consensus needs to bind a block to its
/// finality proof.
struct BlockContext {
    /// Block height. Used as part of the consensus hash preimage.
    uint64_t height = 0;

    /// Parent block hash (32 bytes).
    std::vector<uint8_t> parent_hash;

    /// Pre-encoded canonical header bytes (RLP). Hashed for the consensus
    /// block hash.
    std::vector<uint8_t> header_bytes;

    /// BLS12-381 validator signatures over (height, parent_hash, ...).
    /// Each sig is 48 bytes (compressed G1).
    std::vector<uint8_t> bls_signatures;
    /// Matching compressed G2 public keys (96 bytes each).
    std::vector<uint8_t> bls_pubkeys;
    /// Matching pre-hashed messages (32 bytes each).
    std::vector<uint8_t> bls_messages;

    /// ML-DSA-65 (CRYSTALS-Dilithium) post-quantum signatures from the
    /// lattice round of Quasar finality. Optional; absent on most blocks.
    std::vector<const uint8_t*> pq_signatures;   // 3360 bytes each
    std::vector<const uint8_t*> pq_pubkeys;      // 1952 bytes each
    std::vector<const uint8_t*> pq_messages;     // 64 bytes each
};

/// Aggregate result of processing a single block.
struct PipelineResult {
    /// EVM execution result (gas used, state root from Block-STM, etc.).
    BlockResult evm;

    /// Post-block state root (32 bytes). May be the same value as evm.state_root
    /// when state-root computation is delegated to Block-STM, or freshly hashed
    /// here when enable_state_root is true.
    std::vector<uint8_t> state_root;

    /// Quasar consensus block hash (32 bytes, Keccak-256 of header_bytes).
    std::vector<uint8_t> consensus_hash;

    /// Number of BLS signatures successfully verified.
    uint64_t bls_verifies = 0;
    /// Number of ML-DSA signatures successfully verified.
    uint64_t pq_verifies = 0;

    // Stage timings (wall-clock, ms).
    double total_ms = 0.0;
    double evm_ms = 0.0;
    double state_root_ms = 0.0;
    double consensus_ms = 0.0;
    double sig_verify_ms = 0.0;
};

// -- Unified pipeline ---------------------------------------------------------

class UnifiedPipeline {
public:
    virtual ~UnifiedPipeline() = default;

    /// Create the pipeline. Returns nullptr if the GPU is unavailable.
    /// Falls back to CPU for any subsystem whose GPU kernel cannot be created.
    static std::unique_ptr<UnifiedPipeline> create(const UnifiedConfig& cfg = {});

    /// Process a single block end-to-end on the GPU.
    virtual PipelineResult process_block(
        std::span<const Transaction> txs,
        std::span<const AccountInfo> accounts,
        const BlockContext& ctx) = 0;

    /// Process a backlog of blocks with stage-overlap pipelining.
    /// Block N+1's signature verification runs concurrently with block N's
    /// EVM execution. All blocks share the same base account snapshot.
    virtual std::vector<PipelineResult> process_blocks(
        std::span<const std::span<const Transaction>> blocks,
        std::span<const AccountInfo> base_accounts,
        std::span<const BlockContext> ctxs) = 0;

    /// Diagnostic: device name (e.g. "Apple M1 Max", "NVIDIA H100", "CPU").
    virtual const char* device_name() const = 0;

    /// Diagnostic: backend name (Metal/CUDA/CPU).
    virtual const char* backend_name() const = 0;

protected:
    UnifiedPipeline() = default;
    UnifiedPipeline(const UnifiedPipeline&) = delete;
    UnifiedPipeline& operator=(const UnifiedPipeline&) = delete;
};

}  // namespace evm::gpu::unified
