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

CGpuBlockResultV2 gpu_execute_block_v2(
    const CGpuTx* txs,
    uint32_t      num_txs,
    uint8_t       backend,
    uint32_t      num_threads
) {
    CGpuBlockResultV2 cresult;
    std::memset(&cresult, 0, sizeof(cresult));
    cresult.abi_version = EVM_GPU_ABI_VERSION;

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
    config.num_threads = num_threads;
    config.enable_state_trie_gpu = true;

    evm::gpu::BlockResult result = evm::gpu::execute_block(config, evm_txs, nullptr);

    cresult.num_txs       = num_txs;
    cresult.total_gas     = result.total_gas;
    cresult.exec_time_ms  = result.execution_time_ms;
    cresult.conflicts     = result.conflicts;
    cresult.re_executions = result.re_executions;
    cresult.ok            = 1;

    if (!result.state_root.empty() && result.state_root.size() >= 32) {
        std::memcpy(cresult.state_root, result.state_root.data(), 32);
    }

    if (num_txs > 0) {
        cresult.gas_used = static_cast<uint64_t*>(std::malloc(num_txs * sizeof(uint64_t)));
        cresult.status   = static_cast<uint8_t*>(std::malloc(num_txs * sizeof(uint8_t)));
        if (cresult.gas_used == nullptr || cresult.status == nullptr) {
            std::free(cresult.gas_used);
            std::free(cresult.status);
            cresult.gas_used = nullptr;
            cresult.status   = nullptr;
            cresult.ok = 0;
            return cresult;
        }
        for (uint32_t i = 0; i < num_txs; ++i) {
            cresult.gas_used[i] = (i < result.gas_used.size()) ? result.gas_used[i] : 0;
            // The dispatcher does not currently propagate per-tx status.
            // Until that is wired through evm::gpu::BlockResult, success
            // is implied by the engine completing without throwing.
            cresult.status[i] = EVM_GPU_TX_OK;
        }
    }

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
