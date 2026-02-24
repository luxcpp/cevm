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

#define EVM_GPU_ABI_VERSION 2

// Backend constants — must match evm::gpu::Backend.
#define EVM_GPU_BACKEND_CPU_SEQUENTIAL 0
#define EVM_GPU_BACKEND_CPU_PARALLEL   1
#define EVM_GPU_BACKEND_METAL          2
#define EVM_GPU_BACKEND_CUDA           3

// Per-tx status codes — must match the kernel's TxOutput.status.
#define EVM_GPU_TX_OK              0
#define EVM_GPU_TX_RETURN          1
#define EVM_GPU_TX_REVERT          2
#define EVM_GPU_TX_OOG             3
#define EVM_GPU_TX_ERROR           4
#define EVM_GPU_TX_CALL_NOT_SUPP   5

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

// V2 result — exposes per-tx status and post-execution state root.
typedef struct {
    uint8_t   state_root[32];   // post-execution state root (Keccak-256)
    uint64_t* gas_used;          // per-tx, length=num_txs
    uint8_t*  status;            // per-tx, EVM_GPU_TX_*
    uint32_t  num_txs;
    uint64_t  total_gas;
    double    exec_time_ms;
    uint32_t  conflicts;
    uint32_t  re_executions;
    uint32_t  abi_version;
    int       ok;
} CGpuBlockResultV2;

// Original block execution (kept for ABI stability).
CGpuBlockResult gpu_execute_block(
    const CGpuTx* txs,
    uint32_t      num_txs,
    uint8_t       backend
);

// Extended block execution: returns state root + per-tx status.
// num_threads=0 selects hardware concurrency for CPU_PARALLEL.
CGpuBlockResultV2 gpu_execute_block_v2(
    const CGpuTx* txs,
    uint32_t      num_txs,
    uint8_t       backend,
    uint32_t      num_threads
);

void gpu_free_result(CGpuBlockResult* result);
void gpu_free_result_v2(CGpuBlockResultV2* result);

uint8_t gpu_auto_detect_backend(void);

// Returns a NUL-terminated static string for the given backend ID.
// Static lifetime — caller must NOT free.
const char* gpu_backend_name(uint8_t backend);

// Fills `out` with up to `max` available backend IDs and returns the
// number written. If out==NULL or max==0, returns the available count
// without writing.
uint32_t gpu_available_backends(uint8_t* out, uint32_t max);

// ABI version of this build. Go side compares against a known constant
// to detect mismatched libraries.
uint32_t gpu_abi_version(void);

#ifdef __cplusplus
}
#endif

#endif  // EVM_GPU_GO_BRIDGE_H
