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

#define EVM_GPU_ABI_VERSION 5

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

// EVM revision constants — numeric values match enum evmc_revision in
// evmc/evmc.h. Listed for completeness; Go callers should pass the raw
// uint8 from the evmc binding's Revision constants.
#define EVM_GPU_REV_FRONTIER         0
#define EVM_GPU_REV_HOMESTEAD        1
#define EVM_GPU_REV_TANGERINE        2
#define EVM_GPU_REV_SPURIOUS         3
#define EVM_GPU_REV_BYZANTIUM        4
#define EVM_GPU_REV_CONSTANTINOPLE   5
#define EVM_GPU_REV_PETERSBURG       6
#define EVM_GPU_REV_ISTANBUL         7
#define EVM_GPU_REV_BERLIN           8
#define EVM_GPU_REV_LONDON           9
#define EVM_GPU_REV_PARIS           10
#define EVM_GPU_REV_SHANGHAI        11
#define EVM_GPU_REV_CANCUN          12
#define EVM_GPU_REV_DEFAULT         EVM_GPU_REV_CANCUN

typedef struct {
    uint8_t  from[20];
    uint8_t  to[20];
    uint8_t* data;       // calldata
    uint32_t data_len;
    uint8_t* code;       // EVM bytecode (NULL/0 ⇒ scheduler-only path, no opcode execution)
    uint32_t code_len;
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

// Block-level context shared by every tx in a block. Field layout matches
// evm::gpu::BlockContext (gpu_dispatch.hpp) byte-for-byte. All-zero ⇒
// "no context provided", matching the dispatcher's default Config behaviour.
typedef struct {
    uint8_t  origin[20];
    uint64_t gas_price;
    uint64_t timestamp;
    uint64_t number;
    uint8_t  prevrandao[32];
    uint64_t gas_limit;
    uint64_t chain_id;
    uint64_t base_fee;
    uint64_t blob_base_fee;
    uint8_t  coinbase[20];
    uint8_t  blob_hashes[8][32];
    uint32_t num_blob_hashes;
} CBlockContext;

// Original block execution (kept for ABI stability).
CGpuBlockResult gpu_execute_block(
    const CGpuTx* txs,
    uint32_t      num_txs,
    uint8_t       backend
);

// Extended block execution: returns state root + per-tx status.
// num_threads=0 selects hardware concurrency for CPU_PARALLEL.
// revision is an evmc_revision value (e.g. EVM_GPU_REV_CANCUN). It governs
// the cevm fallback path; kernel CPU/GPU paths implement Cancun
// unconditionally and ignore this field. Pass EVM_GPU_REV_DEFAULT for the
// production default (Cancun).
CGpuBlockResultV2 gpu_execute_block_v2(
    const CGpuTx* txs,
    uint32_t      num_txs,
    uint8_t       backend,
    uint32_t      num_threads,
    uint8_t       revision
);

// V3 block execution: V2 + block context (CHAINID, TIMESTAMP, NUMBER,
// BASEFEE, COINBASE, etc.). Pass NULL for `block_ctx` to get V2 semantics
// (zero-initialised context). The result struct is the same V2 shape; the
// abi_version field reports the loaded library's EVM_GPU_ABI_VERSION (5+).
CGpuBlockResultV2 gpu_execute_block_v3(
    const CGpuTx*         txs,
    uint32_t              num_txs,
    uint8_t               backend,
    uint32_t              num_threads,
    uint8_t               revision,
    const CBlockContext*  block_ctx
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
