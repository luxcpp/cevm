// Copyright (C) 2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.
//
// C-linkage bridge between Go CGo and evm::gpu C++ API.
// Compiled as C++ but exports C symbols for CGo consumption.

#include "gpu_dispatch.hpp"
#include <cstring>
#include <cstdlib>

extern "C" {

typedef struct {
    uint8_t  from[20];
    uint8_t  to[20];
    uint8_t* data;
    uint32_t data_len;
    uint64_t gas_limit;
    uint64_t value;
    uint64_t nonce;
    uint64_t gas_price;
    uint8_t  has_to;
} CGpuTx;

typedef struct {
    uint64_t* gas_used;
    uint32_t  num_txs;
    uint64_t  total_gas;
    double    exec_time_ms;
    uint32_t  conflicts;
    uint32_t  re_executions;
    int       ok;
} CGpuBlockResult;

CGpuBlockResult gpu_execute_block(
    const CGpuTx* txs,
    uint32_t      num_txs,
    uint8_t       backend
) {
    CGpuBlockResult cresult;
    std::memset(&cresult, 0, sizeof(cresult));

    // Convert C structs to evm::gpu::Transaction.
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
        etx.gas_limit = txs[i].gas_limit;
        etx.value     = txs[i].value;
        etx.nonce     = txs[i].nonce;
        etx.gas_price = txs[i].gas_price;
        evm_txs.push_back(std::move(etx));
    }

    // Configure backend.
    evm::gpu::Config config;
    config.backend = static_cast<evm::gpu::Backend>(backend);
    config.enable_state_trie_gpu = true;

    // Execute.
    evm::gpu::BlockResult result = evm::gpu::execute_block(config, evm_txs, nullptr);

    // Pack into C result.
    cresult.num_txs       = num_txs;
    cresult.total_gas     = result.total_gas;
    cresult.exec_time_ms  = result.execution_time_ms;
    cresult.conflicts     = result.conflicts;
    cresult.re_executions = result.re_executions;
    cresult.ok            = 1;

    // Allocate gas_used array for Go to read.
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

void gpu_free_result(CGpuBlockResult* result) {
    if (result != nullptr && result->gas_used != nullptr) {
        std::free(result->gas_used);
        result->gas_used = nullptr;
    }
}

uint8_t gpu_auto_detect_backend(void) {
    return static_cast<uint8_t>(evm::gpu::auto_detect());
}

}  // extern "C"
