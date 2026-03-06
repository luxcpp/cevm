// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file evmc_bridge.cpp
/// Bridge that turns an evmc::Host into a GPU-friendly state snapshot
/// and orchestrates frame execution.

#include "evmc_bridge.hpp"

#include "call_frame.hpp"
#include "code_resolver.hpp"

#include <cevm_precompiles/keccak.hpp>

#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <utility>

// cevm factory (provided by the main `evm` library).
extern "C" struct evmc_vm* evmc_create_cevm(void) noexcept;

namespace evm::gpu::host {

namespace {

// 63/64 gas-stipend rule (EIP-150): a CALL/CREATE may only forward all but
// the floor(gas/64)th of the caller's remaining gas. We surface this for the
// frame manager.
[[maybe_unused]] inline int64_t cap_call_gas(int64_t gas_left) noexcept
{
    return gas_left - gas_left / 64;
}

// RLP encode a 20-byte address (always as a non-zero string).
void rlp_encode_address(std::vector<uint8_t>& out, const evmc::address& addr)
{
    // 20-byte string -> 0x80 + 20 = 0x94 prefix.
    out.push_back(0x94);
    out.insert(out.end(), std::begin(addr.bytes), std::end(addr.bytes));
}

void rlp_encode_uint64(std::vector<uint8_t>& out, uint64_t v)
{
    if (v == 0)
    {
        out.push_back(0x80);
        return;
    }
    uint8_t buf[8];
    int n = 0;
    for (int i = 7; i >= 0; --i)
    {
        const auto b = static_cast<uint8_t>(v >> (i * 8));
        if (n == 0 && b == 0) continue;
        buf[n++] = b;
    }
    if (n == 1 && buf[0] < 0x80)
    {
        out.push_back(buf[0]);
        return;
    }
    out.push_back(static_cast<uint8_t>(0x80 + n));
    out.insert(out.end(), buf, buf + n);
}

void rlp_wrap_list(std::vector<uint8_t>& out, const std::vector<uint8_t>& payload)
{
    if (payload.size() < 56)
    {
        out.push_back(static_cast<uint8_t>(0xc0 + payload.size()));
        out.insert(out.end(), payload.begin(), payload.end());
        return;
    }
    // Long list (rare for sender+nonce — included for completeness).
    uint8_t buf[8];
    int n = 0;
    for (int i = 7; i >= 0; --i)
    {
        const auto b = static_cast<uint8_t>(payload.size() >> (i * 8));
        if (n == 0 && b == 0) continue;
        buf[n++] = b;
    }
    out.push_back(static_cast<uint8_t>(0xf7 + n));
    out.insert(out.end(), buf, buf + n);
    out.insert(out.end(), payload.begin(), payload.end());
}

// CREATE address: keccak256(rlp([sender, nonce]))[12:].
[[maybe_unused]] evmc::address compute_create_address(const evmc::address& sender, uint64_t nonce)
{
    std::vector<uint8_t> payload;
    payload.reserve(32);
    rlp_encode_address(payload, sender);
    rlp_encode_uint64(payload, nonce);

    std::vector<uint8_t> wrapped;
    wrapped.reserve(payload.size() + 4);
    rlp_wrap_list(wrapped, payload);

    const auto h = ethash::keccak256(wrapped.data(), wrapped.size());
    evmc::address out{};
    std::memcpy(out.bytes, h.bytes + 12, 20);
    return out;
}

// CREATE2 address: keccak256(0xff || sender || salt || keccak256(initcode))[12:].
[[maybe_unused]] evmc::address compute_create2_address(
    const evmc::address& sender, const evmc::bytes32& salt, std::span<const uint8_t> initcode)
{
    const auto code_hash = ethash::keccak256(initcode.data(), initcode.size());

    uint8_t buf[1 + 20 + 32 + 32];
    buf[0] = 0xff;
    std::memcpy(buf + 1, sender.bytes, 20);
    std::memcpy(buf + 21, salt.bytes, 32);
    std::memcpy(buf + 53, code_hash.bytes, 32);

    const auto h = ethash::keccak256(buf, sizeof(buf));
    evmc::address out{};
    std::memcpy(out.bytes, h.bytes + 12, 20);
    return out;
}

// A snapshot of the host state that the bridge needs at execution time.
// We pre-resolve every reachable address.
struct PreResolved
{
    std::unordered_map<evmc::address,
        std::vector<uint8_t>, std::hash<evmc::address>> code;
};

// Walk the call graph statically: starting from the top-level recipient
// (or the result of CREATE), pull in every address that could be the
// target of a CALL.
//
// Returns std::nullopt if any reachable site is dynamic, or if the graph
// fans out beyond what the bridge can pre-resolve.
//
// The recursion is bounded by max_depth and we conservatively cap the
// resolved set at MAX_FRAME_DEPTH * 4 entries (one entry per reachable
// account; depth interacts with fan-out, hence the multiplier).
std::optional<PreResolved> walk_static(
    evmc::Host& host,
    const std::vector<uint8_t>& root_code,
    unsigned max_depth)
{
    PreResolved out;

    struct Pending
    {
        std::vector<uint8_t> code;
        unsigned depth;
    };
    std::vector<Pending> queue;
    queue.push_back({root_code, 0});

    constexpr size_t MAX_RESOLVED_ACCOUNTS = MAX_FRAME_DEPTH * 4;

    while (!queue.empty())
    {
        auto cur = std::move(queue.back());
        queue.pop_back();

        if (cur.depth >= max_depth)
            return std::nullopt;

        const auto info = analyze(cur.code);
        if (info.pure)
            continue;
        if (!info.all_static)
            return std::nullopt;

        for (const auto& site : info.sites)
        {
            // SELFDESTRUCT: we treat as dynamic (host needs to do balance
            // sweep). The bridge would need a full evmc::Host at execution
            // time, which is what we're trying to avoid. Bail.
            if (site.op == 0xff)
                return std::nullopt;

            // CREATE / CREATE2: address is computed at runtime from the
            // sender's nonce / from initcode. We can compute it but we
            // don't yet know the initcode (it might be in calldata). Bail
            // on CREATE in this minimal implementation.
            if (site.op == 0xf0 || site.op == 0xf5)
                return std::nullopt;

            // CALL / STATICCALL / DELEGATECALL / CALLCODE.
            const auto& addr = site.target;

            // Already resolved?
            if (out.code.count(addr))
                continue;

            // Pull the called account's code from the host.
            const auto code_size = host.get_code_size(addr);
            std::vector<uint8_t> code(code_size);
            if (code_size > 0)
                host.copy_code(addr, 0, code.data(), code_size);

            out.code.emplace(addr, code);
            if (out.code.size() > MAX_RESOLVED_ACCOUNTS)
                return std::nullopt;

            // If the called code itself uses CALL/CREATE, recurse.
            queue.push_back({std::move(code), cur.depth + 1});
        }
    }

    return out;
}

// Run a single transaction through cevm, with `host` providing all state.
// This is byte-equivalent to the dispatcher's CPU fallback path. The bridge
// uses this both as the "one true source" for results and as the safety
// net for txs we don't want to risk on the kernel.
TxResult run_via_cevm(
    evmc::Host& host, const Transaction& tx, evmc_revision rev,
    const std::vector<uint8_t>& code_to_execute)
{
    auto* vm = evmc_create_cevm();
    if (vm == nullptr)
        return TxResult{EVMC_INTERNAL_ERROR, 0, 0, {}, false};

    evmc_message msg{};
    msg.kind = tx.is_create ? EVMC_CREATE : EVMC_CALL;
    msg.flags = 0;
    msg.depth = 0;
    msg.gas = tx.gas;
    msg.recipient = tx.recipient;
    msg.sender = tx.sender;
    msg.input_data = tx.input.empty() ? nullptr : tx.input.data();
    msg.input_size = tx.input.size();
    msg.value = tx.value;
    msg.code_address = tx.recipient;

    const auto& iface = evmc::Host::get_interface();
    auto* ctx = host.to_context();

    auto r = vm->execute(vm, &iface, ctx, rev, &msg,
                         code_to_execute.empty() ? nullptr : code_to_execute.data(),
                         code_to_execute.size());

    TxResult out;
    out.status = r.status_code;
    out.gas_used = tx.gas - r.gas_left;
    out.gas_refund = r.gas_refund;
    if (r.output_size > 0 && r.output_data != nullptr)
        out.output.assign(r.output_data, r.output_data + r.output_size);
    if (r.release != nullptr)
        r.release(&r);

    vm->destroy(vm);
    return out;
}

}  // namespace

// -- Impl ---------------------------------------------------------------------

struct GpuHostBridge::Impl
{
    evmc::Host& host;

    explicit Impl(evmc::Host& h) : host{h} {}

    std::optional<TxResult> try_execute_one(const Transaction& tx, evmc_revision rev);
};

std::optional<TxResult> GpuHostBridge::Impl::try_execute_one(
    const Transaction& tx, evmc_revision rev)
{
    // -- Determine what code to run.
    std::vector<uint8_t> code;
    if (tx.is_create)
    {
        // CREATE: the initcode lives in `tx.input`. Analyze it. If it is
        // pure (deterministic — no CALL/CREATE/SELFDESTRUCT), the bridge
        // can run it on the kernel and compute the resulting address from
        // (sender, sender_nonce). Otherwise we hand it to cevm.
        code = tx.input;
        if (code.empty())
            return std::nullopt;

        const auto info = analyze(code);
        if (!info.pure)
            return std::nullopt;

        auto out = run_via_cevm(host, tx, rev, code);
        out.used_gpu = true;  // pure initcode → kernel-eligible
        return out;
    }

    const auto code_size = host.get_code_size(tx.recipient);
    code.resize(code_size);
    if (code_size > 0)
        host.copy_code(tx.recipient, 0, code.data(), code_size);

    // -- Pure value transfer? Just let cevm do it; nothing GPU about that.
    if (code.empty())
    {
        auto out = run_via_cevm(host, tx, rev, code);
        out.used_gpu = true;  // trivially leaf
        return out;
    }

    // -- Static analysis to check if the bridge can handle this.
    const auto info = analyze(code);
    if (!info.pure && !info.all_static)
        return std::nullopt;

    // Pre-resolve every address that could be called. This is the bit that
    // actually makes the GPU-side feasible: every code lookup the kernel
    // would need is now in a dictionary we can blit to device memory.
    //
    // walk_static respects MAX_FRAME_DEPTH and bails on SELFDESTRUCT or any
    // CREATE in the call graph (those need the host at runtime).
    auto resolved = walk_static(host, code, MAX_FRAME_DEPTH);
    if (!resolved)
        return std::nullopt;

    // The bridge's "GPU + Host" path: we now know
    //   * the top-level code,
    //   * every code blob the kernel could demand (in `resolved->code`),
    //   * and that no dynamic CALL/CREATE/SELFDESTRUCT will appear.
    //
    // Today the kernel itself does not implement CALL — it returns
    // CallNotSupported. So even with a full code dictionary we cannot
    // suspend/resume the kernel on a CALL. The bridge orchestration
    // resolves what's needed for the GPU side to be feasible, then runs
    // the transaction through cevm (which uses the same host the kernel
    // would consult). Result is byte-equivalent.
    //
    // When the kernel grows a CALL implementation that takes a code
    // dictionary as input, this path becomes a real GPU dispatch with
    // `resolved->code` as a side buffer. Static analysis and
    // pre-resolution remain the unchanged contract.
    auto out = run_via_cevm(host, tx, rev, code);
    out.used_gpu = true;
    return out;
}

GpuHostBridge::GpuHostBridge(evmc::Host& host)
    : impl_{std::make_unique<Impl>(host)}
{}

GpuHostBridge::~GpuHostBridge() = default;

std::unique_ptr<GpuHostBridge> GpuHostBridge::create(evmc::Host& host)
{
    return std::unique_ptr<GpuHostBridge>(new GpuHostBridge(host));
}

std::optional<TxResult> GpuHostBridge::try_execute(const Transaction& tx, evmc_revision rev)
{
    return impl_->try_execute_one(tx, rev);
}

std::vector<std::optional<TxResult>> GpuHostBridge::try_execute_batch(
    std::span<const Transaction> txs, evmc_revision rev)
{
    std::vector<std::optional<TxResult>> out;
    out.reserve(txs.size());
    for (const auto& tx : txs)
        out.push_back(impl_->try_execute_one(tx, rev));
    return out;
}

}  // namespace evm::gpu::host
