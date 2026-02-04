// Copyright (C) 2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.
//
// C-linkage header for Go CGo bridge to evm::gpu.

#ifndef EVM_GPU_GO_BRIDGE_H
#define EVM_GPU_GO_BRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

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
);

void gpu_free_result(CGpuBlockResult* result);
uint8_t gpu_auto_detect_backend(void);

#ifdef __cplusplus
}
#endif

#endif  // EVM_GPU_GO_BRIDGE_H
