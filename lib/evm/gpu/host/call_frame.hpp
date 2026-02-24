// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file call_frame.hpp
/// Call frame layout used by the bridge.
///
/// One frame represents a single EVM execution context: a piece of bytecode
/// to run, the calldata it sees, the address it's executing as, the value it
/// was called with, and the gas it has. Frames are stacked: when the inner
/// frame returns, its output becomes the next CALL/CREATE's return data in
/// the outer frame.
///
/// The bridge maintains its own stack of frames so that nested calls don't
/// recurse into the interpreter — the interpreter only ever runs leaf frames.

#pragma once

#include <evmc/evmc.hpp>

#include <cstdint>
#include <vector>

namespace evm::gpu::host {

/// What kind of call produced this frame.
enum class FrameKind : uint8_t
{
    Top,          ///< The outer transaction frame.
    Call,         ///< OP_CALL.
    CallCode,     ///< OP_CALLCODE.
    DelegateCall, ///< OP_DELEGATECALL.
    StaticCall,   ///< OP_STATICCALL.
    Create,       ///< OP_CREATE.
    Create2,      ///< OP_CREATE2.
};

/// A single call frame.
struct CallFrame
{
    FrameKind kind = FrameKind::Top;

    /// Address of the executing code (i.e. ADDRESS opcode result).
    evmc::address recipient{};
    /// Address that was the *target* of the call (may differ from recipient
    /// for DELEGATECALL/CALLCODE: code runs as recipient but is from code_address).
    evmc::address code_address{};
    /// Address that initiated the call (i.e. CALLER opcode result).
    evmc::address caller{};

    /// Value passed with the call (zero for static, delegate, create2-without-value).
    evmc::uint256be value{};
    /// For DELEGATECALL we surface the *outer* msg.value as CALLVALUE.
    /// `apparent_value` lets the frame report a different value than what
    /// was actually transferred.
    evmc::uint256be apparent_value{};

    /// The bytecode being executed.
    std::vector<uint8_t> code;
    /// The input the EVM sees as calldata.
    std::vector<uint8_t> input;

    /// Gas budget for this frame (post-63/64 stipend rule).
    int64_t gas = 0;
    /// Whether STATICCALL forbids state modifications below.
    bool is_static = false;
    /// Depth in the call stack (0 = top, capped by MAX_FRAME_DEPTH).
    unsigned depth = 0;
};

/// Output of running a single frame.
struct FrameResult
{
    evmc_status_code status = EVMC_FAILURE;
    int64_t gas_left = 0;
    int64_t gas_refund = 0;
    std::vector<uint8_t> output;
    /// For CREATE/CREATE2: the address of the new contract on success.
    evmc::address create_address{};
};

}  // namespace evm::gpu::host
