// Copyright (C) 2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.
//
// C-linkage bridge between Go CGo and evm::gpu C++ API.
// Compiled as C++ but exports C symbols for CGo consumption.

#include "go_bridge.h"
#include "gpu_dispatch.hpp"
#include <cstring>
#include <cstdlib>

extern "C" {

CGpuBlockResult gpu_execute_block(
    const CGpuTx* txs,
    uint32_t      num_txs,
    uint8_t       backend
) {
    CGpuBlockResult cresult;
    std::memset(&cresult, 0, sizeof(cresult));

    std::vector<evm::gpu::Transaction> evm_txs;
    evm_txs.reserve(num_txs);
    for (uint32_t i = 0; i < num_txs; ++i) {
        evm::gpu::Transaction etx;
        etx.from.assign(txs[i].from, txs[i].from + 20);
        if (txs[i].has_to) {
            etx.to.assign(txs[i].to, txs[i].to + 20);
        }
        if (txs[i].data != nullptr && txs[i].data_len > 0) {
            etx.data.assign(txs[i].data, txs[i].data + txs[i].data_len);
        }
        if (txs[i].code != nullptr && txs[i].code_len > 0) {
            etx.code.assign(txs[i].code, txs[i].code + txs[i].code_len);
        }
        etx.gas_limit = txs[i].gas_limit;
        etx.value     = txs[i].value;
        etx.nonce     = txs[i].nonce;
        etx.gas_price = txs[i].gas_price;
        evm_txs.push_back(std::move(etx));
    }

    evm::gpu::Config config;
    config.backend = static_cast<evm::gpu::Backend>(backend);
    config.enable_state_trie_gpu = true;

    evm::gpu::BlockResult result = evm::gpu::execute_block(config, evm_txs, nullptr);

    cresult.num_txs       = num_txs;
    cresult.total_gas     = result.total_gas;
    cresult.exec_time_ms  = result.execution_time_ms;
    cresult.conflicts     = result.conflicts;
    cresult.re_executions = result.re_executions;
    cresult.ok            = 1;

    cresult.gas_used = static_cast<uint64_t*>(std::malloc(num_txs * sizeof(uint64_t)));
    if (cresult.gas_used == nullptr) {
        cresult.ok = 0;
        return cresult;
    }
    for (uint32_t i = 0; i < num_txs && i < result.gas_used.size(); ++i) {
        cresult.gas_used[i] = result.gas_used[i];
    }

    return cresult;
}

// Convert a flat C tx batch into the C++ Transaction vector. Shared by V1, V2,
// and V3 entry points so input handling stays identical across ABI versions.
static std::vector<evm::gpu::Transaction>
collect_txs(const CGpuTx* txs, uint32_t num_txs)
{
    std::vector<evm::gpu::Transaction> evm_txs;
    evm_txs.reserve(num_txs);
    for (uint32_t i = 0; i < num_txs; ++i) {
        evm::gpu::Transaction etx;
        etx.from.assign(txs[i].from, txs[i].from + 20);
        if (txs[i].has_to) {
            etx.to.assign(txs[i].to, txs[i].to + 20);
        }
        if (txs[i].data != nullptr && txs[i].data_len > 0) {
            etx.data.assign(txs[i].data, txs[i].data + txs[i].data_len);
        }
        if (txs[i].code != nullptr && txs[i].code_len > 0) {
            etx.code.assign(txs[i].code, txs[i].code + txs[i].code_len);
        }
        etx.gas_limit = txs[i].gas_limit;
        etx.value     = txs[i].value;
        etx.nonce     = txs[i].nonce;
        etx.gas_price = txs[i].gas_price;
        evm_txs.push_back(std::move(etx));
    }
    return evm_txs;
}

// Pack the C++ BlockResult into the V2 wire shape. Allocates gas_used/status
// arrays via std::malloc; caller frees with gpu_free_result_v2. On allocation
// failure the result is marked ok=0 and any partial allocation is freed.
//
// `surface_kernel_status` controls how per-tx status is reported:
//   - false (V2 wire shape): every tx reports EVM_GPU_TX_OK as long as the
//     dispatcher returned without error. This is the legacy ABI v4 contract
//     callers depend on for the dispatcher's cevm path which doesn't yet
//     populate result.status[].
//   - true (V3+ wire shape): tx[i].status comes straight from
//     result.status[i] when populated, falling back to EVM_GPU_TX_OK only
//     for slots the dispatcher left unfilled (no-bytecode value-transfer
//     batches). This is the ABI v5 contract — callers that opt in via V3
//     get the kernel-accurate per-tx outcome.
static void pack_v2_result(const evm::gpu::BlockResult& result,
                           uint32_t num_txs,
                           bool surface_kernel_status,
                           CGpuBlockResultV2& cresult)
{
    cresult.num_txs       = num_txs;
    cresult.total_gas     = result.total_gas;
    cresult.exec_time_ms  = result.execution_time_ms;
    cresult.conflicts     = result.conflicts;
    cresult.re_executions = result.re_executions;
    cresult.abi_version   = EVM_GPU_ABI_VERSION;
    cresult.ok            = 1;

    if (!result.state_root.empty() && result.state_root.size() >= 32) {
        std::memcpy(cresult.state_root, result.state_root.data(), 32);
    }

    if (num_txs == 0) {
        return;
    }

    cresult.gas_used = static_cast<uint64_t*>(std::malloc(num_txs * sizeof(uint64_t)));
    cresult.status   = static_cast<uint8_t*>(std::malloc(num_txs * sizeof(uint8_t)));
    if (cresult.gas_used == nullptr || cresult.status == nullptr) {
        std::free(cresult.gas_used);
        std::free(cresult.status);
        cresult.gas_used = nullptr;
        cresult.status   = nullptr;
        cresult.ok = 0;
        return;
    }
    for (uint32_t i = 0; i < num_txs; ++i) {
        cresult.gas_used[i] = (i < result.gas_used.size()) ? result.gas_used[i] : 0;
        if (surface_kernel_status && i < result.status.size()) {
            cresult.status[i] = static_cast<uint8_t>(result.status[i]);
        } else {
            // V2 wire shape: status is implied OK when the dispatcher
            // returned without error. Keeps V4 callers behaviour
            // identical across an ABI v4→v5 library upgrade.
            cresult.status[i] = EVM_GPU_TX_OK;
        }
    }
}

CGpuBlockResultV2 gpu_execute_block_v2(
    const CGpuTx* txs,
    uint32_t      num_txs,
    uint8_t       backend,
    uint32_t      num_threads,
    uint8_t       revision
) {
    CGpuBlockResultV2 cresult;
    std::memset(&cresult, 0, sizeof(cresult));
    cresult.abi_version = EVM_GPU_ABI_VERSION;

    auto evm_txs = collect_txs(txs, num_txs);

    evm::gpu::Config config;
    config.backend = static_cast<evm::gpu::Backend>(backend);
    config.num_threads = num_threads;
    config.enable_state_trie_gpu = true;
    config.revision = static_cast<evmc_revision>(revision);
    // V2 callers don't supply BlockContext — leave it zero-initialised.

    evm::gpu::BlockResult result = evm::gpu::execute_block(config, evm_txs, nullptr);
    pack_v2_result(result, num_txs, /*surface_kernel_status=*/false, cresult);
    return cresult;
}

CGpuBlockResultV2 gpu_execute_block_v3(
    const CGpuTx*         txs,
    uint32_t              num_txs,
    uint8_t               backend,
    uint32_t              num_threads,
    uint8_t               revision,
    const CBlockContext*  block_ctx
) {
    CGpuBlockResultV2 cresult;
    std::memset(&cresult, 0, sizeof(cresult));
    cresult.abi_version = EVM_GPU_ABI_VERSION;

    auto evm_txs = collect_txs(txs, num_txs);

    evm::gpu::Config config;
    config.backend = static_cast<evm::gpu::Backend>(backend);
    config.num_threads = num_threads;
    config.enable_state_trie_gpu = true;
    config.revision = static_cast<evmc_revision>(revision);

    if (block_ctx != nullptr) {
        // Wire format is layout-compatible: dispatcher BlockContext is the
        // exact same byte sequence as CBlockContext. Copy through memcpy
        // rather than field-by-field so the layout stays in lockstep with
        // any future addition that lands on both sides at once.
        static_assert(sizeof(evm::gpu::BlockContext) == sizeof(CBlockContext),
            "evm::gpu::BlockContext and CBlockContext must stay layout-compatible");
        std::memcpy(&config.block_context, block_ctx, sizeof(CBlockContext));
    }

    evm::gpu::BlockResult result = evm::gpu::execute_block(config, evm_txs, nullptr);
    pack_v2_result(result, num_txs, /*surface_kernel_status=*/true, cresult);
    return cresult;
}

void gpu_free_result(CGpuBlockResult* result) {
    if (result == nullptr) return;
    if (result->gas_used != nullptr) {
        std::free(result->gas_used);
        result->gas_used = nullptr;
    }
}

void gpu_free_result_v2(CGpuBlockResultV2* result) {
    if (result == nullptr) return;
    if (result->gas_used != nullptr) {
        std::free(result->gas_used);
        result->gas_used = nullptr;
    }
    if (result->status != nullptr) {
        std::free(result->status);
        result->status = nullptr;
    }
}

uint8_t gpu_auto_detect_backend(void) {
    return static_cast<uint8_t>(evm::gpu::auto_detect());
}

const char* gpu_backend_name(uint8_t backend) {
    return evm::gpu::backend_name(static_cast<evm::gpu::Backend>(backend));
}

uint32_t gpu_available_backends(uint8_t* out, uint32_t max) {
    auto backends = evm::gpu::available_backends();
    uint32_t n = static_cast<uint32_t>(backends.size());
    if (out == nullptr || max == 0) return n;
    uint32_t to_write = (n < max) ? n : max;
    for (uint32_t i = 0; i < to_write; ++i) {
        out[i] = static_cast<uint8_t>(backends[i]);
    }
    return to_write;
}

uint32_t gpu_abi_version(void) {
    return EVM_GPU_ABI_VERSION;
}

}  // extern "C"
