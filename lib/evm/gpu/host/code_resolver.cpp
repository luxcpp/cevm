// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "code_resolver.hpp"

#include <cstring>

namespace evm::gpu::host {

namespace {

// Opcodes we care about classifying as CALL-family. Anything outside this set
// either modifies the stack in a way we can model, or is irrelevant.
constexpr bool is_call_op(uint8_t op) noexcept
{
    return op == 0xf0 || op == 0xf1 || op == 0xf2 || op == 0xf4 ||
           op == 0xf5 || op == 0xfa || op == 0xff;
}

constexpr bool is_push(uint8_t op) noexcept { return op >= 0x60 && op <= 0x7f; }
constexpr bool is_dup(uint8_t op)  noexcept { return op >= 0x80 && op <= 0x8f; }
constexpr bool is_swap(uint8_t op) noexcept { return op >= 0x90 && op <= 0x9f; }

// Stack item produced by simple constant folding. Either a known 32-byte
// constant (for which we may fail to interpret it as an address but the value
// is fixed) or "unknown" — anything we can't model.
struct StackItem
{
    bool known = false;
    uint8_t bytes[32] = {};  // big-endian
};

// Linear scan over `code`, tracking a small symbolic stack. Whenever we see
// a CALL/CREATE/SELFDESTRUCT, we look at the relevant stack slot and decide
// whether the target address is statically resolvable.
//
// Stack layout for the EVM CALL family (top-down):
//   CALL/CALLCODE:        gas, addr, value, in_off, in_size, out_off, out_size
//   DELEGATECALL/STATIC:  gas, addr, in_off, in_size, out_off, out_size
//   CREATE:               value, in_off, in_size
//   CREATE2:              value, in_off, in_size, salt
//   SELFDESTRUCT:         beneficiary
//
// We only need the address position to make the static/dynamic call.
//
// JUMPs make precise tracking impossible. As soon as we encounter a JUMP/JUMPI
// we mark the whole frame "uncertain" by clearing the stack. JUMPDEST resets
// us to an empty stack (worst case).
//
// This is conservative — a contract with a JUMP somewhere can still be
// classified static if every site we encounter has its operand pushed in a
// straight line right before the call. We just can't carry the model across
// jumps.
class Analyzer
{
public:
    explicit Analyzer(std::span<const uint8_t> code) noexcept : code_{code} {}

    CodeAnalysis run()
    {
        CodeAnalysis out;
        std::vector<StackItem> stack;
        stack.reserve(64);

        for (uint32_t pc = 0; pc < code_.size();)
        {
            const uint8_t op = code_[pc];

            if (is_call_op(op))
            {
                CallSite site;
                site.pc = pc;
                site.op = op;
                classify(op, stack, site);
                out.sites.push_back(site);
                if (!site.fully_static)
                    out.all_static = false;
                out.pure = false;

                // Conservatively clear the stack — modelling the call's effect
                // on the stack precisely would require executing the call.
                stack.clear();
                ++pc;
                continue;
            }

            // -- Producers of known stack items ----------------------------
            if (op == 0x5f)  // PUSH0
            {
                push_zero(stack);
                ++pc;
                continue;
            }
            if (is_push(op))
            {
                const uint32_t n = static_cast<uint32_t>(op - 0x60 + 1);
                StackItem it;
                it.known = true;
                if (pc + 1 + n <= code_.size())
                {
                    // Right-align bytes, big-endian.
                    std::memcpy(&it.bytes[32 - n], &code_[pc + 1], n);
                }
                stack.push_back(it);
                pc += 1 + n;
                continue;
            }

            // -- Stack manipulators we can model ---------------------------
            if (is_dup(op))
            {
                const uint32_t k = static_cast<uint32_t>(op - 0x80 + 1);
                if (k <= stack.size())
                    stack.push_back(stack[stack.size() - k]);
                else
                    stack.push_back({});  // unknown
                ++pc;
                continue;
            }
            if (is_swap(op))
            {
                const uint32_t k = static_cast<uint32_t>(op - 0x90 + 1);
                if (k < stack.size())
                {
                    const auto top = stack.back();
                    stack.back() = stack[stack.size() - 1 - k];
                    stack[stack.size() - 1 - k] = top;
                }
                ++pc;
                continue;
            }

            if (op == 0x50)  // POP
            {
                if (!stack.empty())
                    stack.pop_back();
                ++pc;
                continue;
            }
            if (op == 0x5b)  // JUMPDEST
            {
                // Treat join points as full unknowns.
                stack.clear();
                ++pc;
                continue;
            }
            if (op == 0x56 || op == 0x57)  // JUMP / JUMPI
            {
                stack.clear();
                ++pc;
                continue;
            }
            if (op == 0x00 || op == 0xf3 || op == 0xfd || op == 0xfe)
            {
                // STOP / RETURN / REVERT / INVALID — execution path ends.
                stack.clear();
                ++pc;
                continue;
            }

            // -- Anything else makes its outputs unknown -------------------
            const auto [pops, pushes] = stack_effect(op);
            for (uint32_t i = 0; i < pops && !stack.empty(); ++i)
                stack.pop_back();
            for (uint32_t i = 0; i < pushes; ++i)
                stack.push_back({});  // unknown

            ++pc;
        }

        return out;
    }

private:
    // Push a known-zero item.
    static void push_zero(std::vector<StackItem>& stack)
    {
        StackItem it;
        it.known = true;
        // bytes already zero
        stack.push_back(it);
    }

    // Look at the stack picture before a call op and decide if the target
    // address is statically known.
    void classify(uint8_t op, const std::vector<StackItem>& stack, CallSite& site)
    {
        // Position of the address operand from the top of stack:
        //   CALL/CALLCODE/DELEGATECALL/STATICCALL: index 1 (gas, addr, ...)
        //   CREATE/CREATE2/SELFDESTRUCT: no addr, but result is a *new*
        //                                   account — for CREATE-family we
        //                                   compute the address ourselves
        //                                   below (deterministic). For
        //                                   SELFDESTRUCT the operand is the
        //                                   beneficiary at index 0.
        if (op == 0xf1 || op == 0xf2 || op == 0xf4 || op == 0xfa)
        {
            if (stack.size() < 2)
                return;
            const auto& it = stack[stack.size() - 2];
            if (!it.known)
                return;
            site.fully_static = true;
            // Bytes 12..31 are the address (right-aligned).
            std::memcpy(site.target.bytes, &it.bytes[12], 20);
            return;
        }

        if (op == 0xff)
        {
            if (stack.empty())
                return;
            const auto& it = stack.back();
            if (!it.known)
                return;
            site.fully_static = true;
            std::memcpy(site.target.bytes, &it.bytes[12], 20);
            return;
        }

        // CREATE / CREATE2 — we don't need the target address for the
        // analysis, the host computes it. Mark static iff the value is
        // known (we don't even need that — the bridge will compute the
        // result either way).
        site.fully_static = true;
    }

    // Stack effect for opcodes we don't model precisely.
    // Returns {pops, pushes}.
    static std::pair<uint32_t, uint32_t> stack_effect(uint8_t op)
    {
        // Most arithmetic / comparison / bitwise ops: 2 pops, 1 push.
        // We don't need the values for static classification.
        if (op >= 0x01 && op <= 0x07) return {2, 1};   // ADD..SMOD
        if (op == 0x08 || op == 0x09) return {3, 1};   // ADDMOD/MULMOD
        if (op == 0x0a || op == 0x0b) return {2, 1};   // EXP/SIGNEXTEND
        if (op >= 0x10 && op <= 0x14) return {2, 1};   // LT..EQ
        if (op == 0x15)               return {1, 1};   // ISZERO
        if (op >= 0x16 && op <= 0x18) return {2, 1};   // AND/OR/XOR
        if (op == 0x19)               return {1, 1};   // NOT
        if (op == 0x1a)               return {2, 1};   // BYTE
        if (op >= 0x1b && op <= 0x1d) return {2, 1};   // SHL/SHR/SAR
        if (op == 0x20)               return {2, 1};   // KECCAK256

        if (op == 0x30)               return {0, 1};   // ADDRESS
        if (op == 0x31)               return {1, 1};   // BALANCE
        if (op == 0x32)               return {0, 1};   // ORIGIN
        if (op == 0x33)               return {0, 1};   // CALLER
        if (op == 0x34)               return {0, 1};   // CALLVALUE
        if (op == 0x35)               return {1, 1};   // CALLDATALOAD
        if (op == 0x36)               return {0, 1};   // CALLDATASIZE
        if (op == 0x37)               return {3, 0};   // CALLDATACOPY
        if (op == 0x38)               return {0, 1};   // CODESIZE
        if (op == 0x39)               return {3, 0};   // CODECOPY
        if (op == 0x3a)               return {0, 1};   // GASPRICE
        if (op == 0x3b)               return {1, 1};   // EXTCODESIZE
        if (op == 0x3c)               return {4, 0};   // EXTCODECOPY
        if (op == 0x3d)               return {0, 1};   // RETURNDATASIZE
        if (op == 0x3e)               return {3, 0};   // RETURNDATACOPY
        if (op == 0x3f)               return {1, 1};   // EXTCODEHASH
        if (op == 0x40)               return {1, 1};   // BLOCKHASH
        if (op >= 0x41 && op <= 0x4a) return {0, 1};   // COINBASE..BLOBBASEFEE

        if (op == 0x51)               return {1, 1};   // MLOAD
        if (op == 0x52 || op == 0x53) return {2, 0};   // MSTORE/MSTORE8
        if (op == 0x54)               return {1, 1};   // SLOAD
        if (op == 0x55)               return {2, 0};   // SSTORE
        if (op == 0x58)               return {0, 1};   // PC
        if (op == 0x59)               return {0, 1};   // MSIZE
        if (op == 0x5a)               return {0, 1};   // GAS

        if (op == 0x5c)               return {1, 1};   // TLOAD
        if (op == 0x5d)               return {2, 0};   // TSTORE
        if (op == 0x5e)               return {3, 0};   // MCOPY

        if (op >= 0xa0 && op <= 0xa4)
            return {2u + static_cast<uint32_t>(op - 0xa0), 0};  // LOG0..LOG4

        // Unknown — be safe: pop one, push none. Worst case stack is too
        // shallow and we'll bail out anyway.
        return {1, 0};
    }

    std::span<const uint8_t> code_;
};

}  // namespace

CodeAnalysis analyze(std::span<const uint8_t> code)
{
    Analyzer a{code};
    return a.run();
}

}  // namespace evm::gpu::host
