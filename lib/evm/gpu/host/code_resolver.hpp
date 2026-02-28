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
/// makes the bridge bail out so evmone can take over.
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

}  // namespace evm::gpu::host
