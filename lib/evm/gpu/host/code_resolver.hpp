// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file code_resolver.hpp
/// Static analysis of EVM bytecode that classifies CALL/CREATE targets.
///
/// The bridge needs to know up-front:
///   * Does this bytecode invoke any CALL/CREATE family opcode at all?
///   * If so, can every target address be determined statically?
///   * Does it have SELFDESTRUCT? (treated as dynamic — needs host).
///
/// A target is "static" when the address pushed onto the stack right before
/// the CALL opcode came from a single PUSH20 instruction (or shorter PUSH that
/// produces a 20-byte address). This is enough to handle the overwhelming
/// majority of compiler output: Solidity emits calls as
///
///     PUSH20 <addr> ; gas ; mstore stuff ; CALL
///
/// Anything else (CALLDATALOAD, SLOAD, arithmetic, MLOAD …) is dynamic and
/// makes the bridge bail out so cevm can take over.
///
/// We're not building a full data-flow analyzer here. We only walk forwards
/// from each CALL/CREATE site and look at the stack picture immediately
/// before it, modelling only the easy cases (PUSH/DUP/SWAP/JUMPDEST).
/// On any unrecognised producer we say "dynamic". Conservative is correct.

#pragma once

#include <evmc/evmc.hpp>

#include <cstdint>
#include <span>
#include <variant>
#include <vector>

namespace evm::gpu::host {

/// One CALL/CREATE site that the analyzer found.
struct CallSite
{
    /// PC of the CALL/CREATE opcode itself.
    uint32_t pc = 0;
    /// The opcode (0xf0 = CREATE, 0xf1 = CALL, 0xf2 = CALLCODE,
    /// 0xf4 = DELEGATECALL, 0xf5 = CREATE2, 0xfa = STATICCALL,
    /// 0xff = SELFDESTRUCT).
    uint8_t op = 0;
    /// True if every input operand needed by the bridge is constant.
    bool fully_static = false;
    /// Statically resolved target address (only valid when `fully_static`).
    evmc::address target{};
};

/// Output of `analyze`.
struct CodeAnalysis
{
    /// Every CALL/CREATE/SELFDESTRUCT site found.
    std::vector<CallSite> sites;
    /// True iff the bytecode has zero CALL/CREATE/SELFDESTRUCT opcodes.
    bool pure = true;
    /// True iff every site in `sites` has `fully_static`.
    bool all_static = true;
};

/// Analyze `code`. Returns conservative classification — never produces a
/// false "static" claim.
CodeAnalysis analyze(std::span<const uint8_t> code);

/// GPU kernel resource caps. Hitting any of these means the kernel either
/// silently truncates (storage, output), silently OOGs (memory), or hits an
/// ERR (logs). All four are consensus divergences vs cevm, so the
/// dispatcher must spot them up front and fall back to CPU.
///
/// The numbers come from `lib/evm/gpu/kernel/evm_kernel_host.hpp` and the
/// matching constants inside the Metal/CUDA kernels — see
/// `HOST_MAX_MEMORY_PER_TX` etc. Keep them in sync.
constexpr uint32_t KERNEL_MAX_MEMORY_PER_TX  = 65'536;
constexpr uint32_t KERNEL_MAX_STORAGE_PER_TX = 64;
constexpr uint32_t KERNEL_MAX_LOGS_PER_TX    = 16;
constexpr uint32_t KERNEL_MAX_OUTPUT_PER_TX  = 1024;

/// Static-analysis upper bounds on what one execution of `code` could
/// demand from the GPU kernel. The kernel cannot service:
///   * BALANCE / EXTCODE* / SELFBALANCE / BLOCKHASH (returns hardcoded 0)
///   * memory > KERNEL_MAX_MEMORY_PER_TX (silent OOG)
///   * storage writes > KERNEL_MAX_STORAGE_PER_TX (silent drop)
///   * logs > KERNEL_MAX_LOGS_PER_TX (ERR)
///   * RETURN/REVERT size > KERNEL_MAX_OUTPUT_PER_TX (silent truncate)
///
/// `analyze_requirements` walks the bytecode once and reports an upper
/// bound for each. The values are intentionally conservative — if the
/// analyzer cannot prove a tight bound, it returns one that triggers
/// fallback. There is no false-positive cost (we run on CPU; correct but
/// slower) and no false-negative path (we never claim a tx is GPU-safe
/// when it isn't).
struct TxRequirements
{
    /// True if `code` reaches an opcode whose result depends on the
    /// account-state oracle: BALANCE (0x31), EXTCODESIZE (0x3b),
    /// EXTCODECOPY (0x3c), EXTCODEHASH (0x3f), BLOCKHASH (0x40),
    /// SELFBALANCE (0x47). These return zero on the GPU and can never be
    /// served correctly without a real host.
    bool reads_account_state = false;
    /// Highest static constant memory offset+size we observed in any
    /// MLOAD/MSTORE/MSTORE8/MCOPY/RETURN/REVERT/CALLDATACOPY/CODECOPY/
    /// EXTCODECOPY/RETURNDATACOPY/LOG*. If any operand was non-constant
    /// we set this to UINT32_MAX so the dispatcher falls back.
    uint32_t max_memory_used = 0;
    /// Number of distinct constant storage slots we saw written through
    /// SSTORE. Slots loaded via SLOAD are counted too because the kernel
    /// promotes loaded slots to its small storage table. Non-constant
    /// keys → UINT32_MAX.
    uint32_t max_storage_keys = 0;
    /// Count of reachable LOG0..LOG4 opcodes. We do not deduplicate by
    /// path — every LOG opcode in the bytecode counts once. Loops can
    /// emit more logs at runtime than this bound, so any LOG following a
    /// JUMP/JUMPI raises the count to UINT32_MAX.
    uint32_t max_log_count = 0;
    /// Largest static constant size operand we saw on a RETURN/REVERT.
    /// Non-constant size → UINT32_MAX.
    uint32_t max_output_size = 0;
};

/// Walk `code` once and compute conservative upper bounds.
TxRequirements analyze_requirements(std::span<const uint8_t> code);

/// Bitmask describing why a tx cannot run safely on the GPU kernel. Used
/// by the dispatcher to populate `BlockResult::warnings[i]` so callers
/// can tell whether their per-tx result is consensus-equivalent to cevm.
enum TxWarning : uint32_t
{
    /// Tx reads account state (BALANCE / EXTCODE* / SELFBALANCE / BLOCKHASH)
    /// and was executed on the GPU without a real host — result diverges
    /// from mainnet.
    TX_WARN_ACCOUNT_STATE_ON_GPU = 1u << 0,
    /// Tx exceeds KERNEL_MAX_MEMORY_PER_TX and was executed on the GPU.
    TX_WARN_MEMORY_OVERFLOW      = 1u << 1,
    /// Tx exceeds KERNEL_MAX_STORAGE_PER_TX and was executed on the GPU.
    TX_WARN_STORAGE_OVERFLOW     = 1u << 2,
    /// Tx exceeds KERNEL_MAX_LOGS_PER_TX and was executed on the GPU.
    TX_WARN_LOG_OVERFLOW         = 1u << 3,
    /// Tx exceeds KERNEL_MAX_OUTPUT_PER_TX and was executed on the GPU.
    TX_WARN_OUTPUT_OVERFLOW      = 1u << 4,
};

/// Returns the bitmask of warnings this `req` would produce when executed
/// on a GPU kernel without a host fallback. Empty (0) means GPU-safe.
uint32_t classify_warnings(const TxRequirements& req) noexcept;

}  // namespace evm::gpu::host
