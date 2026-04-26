// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file quasar_cpu_reference.hpp
/// CPU reference implementation of the Quasar wave-tick scheduler — the
/// differential-fuzz oracle for cross-backend determinism testing
/// (STM-004 in the Q3.0 STM Red review). Metal vs CPU vs CUDA must
/// produce byte-identical roots on the same input.
///
/// Mirrors the v0.36 synthetic substrate: each tx performs one
/// (read, write) pair on a single MVCC slot derived from origin_lo/hi.
/// Block-STM repair semantics + LP-010 anti-livelock cap match the
/// drain_validate path exactly. EVM bytecode interpretation is not in
/// scope (lands in v0.41 separately) — this reference proves the
/// substrate's determinism story before interpreter complexity.

#pragma once

#include "quasar_gpu_layout.hpp"

#include <cstdint>
#include <span>

namespace quasar::gpu::ref {

struct HostInputTx {
    uint64_t gas_limit = 0;
    uint64_t origin = 0;
    bool needs_state = false;
    bool needs_exec  = false;
};

struct CpuReferenceResult {
    uint8_t  block_hash[32]{};
    uint8_t  state_root[32]{};
    uint8_t  receipts_root[32]{};
    uint8_t  execution_root[32]{};
    uint8_t  mode_root[32]{};
    uint32_t tx_count       = 0;
    uint64_t gas_used       = 0;
    uint32_t conflict_count = 0;
    uint32_t repair_count   = 0;
    uint32_t status         = 0;
    uint32_t mode           = 0;
};

CpuReferenceResult run_reference(const QuasarRoundDescriptor& desc,
                                 std::span<const HostInputTx> txs);

}  // namespace quasar::gpu::ref
