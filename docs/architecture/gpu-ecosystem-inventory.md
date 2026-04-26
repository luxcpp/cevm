# Lux GPU Ecosystem Inventory

Single canonical map of every GPU code path that the cevm dispatcher can
reach, what state it is in, and how cevm → lux-accel → luxcpp/{cuda,metal}
fit together. Updated alongside the code; if a row in this doc disagrees
with the code, the code is right and this doc is stale.

## 1. Layered architecture

```
                       ┌────────────────────────────────────┐
        EVM bytecode → │ cevm/lib/evm/gpu/gpu_dispatch.cpp │  CPU_Sequential, CPU_Parallel,
                       │                                    │  GPU_Metal, GPU_CUDA
                       └─────────────┬──────────────────────┘
                                     │  STATICCALL
                                     ▼
                     ┌─────────────────────────────────────┐
                     │ cevm/lib/evm/gpu/precompiles/       │
                     │   precompile_dispatch.{cpp,hpp}     │   address-keyed
                     │   ecrecover_metal.mm   (0x01)       │   table.
                     │   bn254_metal.mm       (0x06,0x07)  │
                     │   bls12_381_metal.mm   (0x0b..0x0f) │
                     │   point_eval_metal.mm  (0x0a)       │
                     │   dex_match_*.{cpp,mm}  (0x100)     │
                     └─────────────┬──────────────────────┘
                                   │
              ┌────────────────────┼─────────────────────┐
              ▼                    ▼                     ▼
   ┌────────────────────┐  ┌──────────────────┐  ┌──────────────────┐
   │ luxcpp/gpu (C ABI) │  │ luxcpp/lux-accel │  │ cevm/lib/cevm_   │
   │  lux/gpu.h         │  │ liblux_accel.so  │  │ precompiles/     │
   │  Metal+CUDA+WebGPU │  │ Tensor-shaped C  │  │ blst, intx,      │
   │  primitives        │  │ API; consumed by │  │ pure-C++ refs    │
   │                    │  │ Go via luxfi/    │  │ (parity refs)    │
   │                    │  │ accel            │  │                  │
   └─────────┬──────────┘  └────────┬─────────┘  └──────────────────┘
             │                      │
             ▼                      ▼
   ┌────────────────────┐  ┌──────────────────┐
   │ luxcpp/{cuda,metal}│  │ luxcpp/{cuda,    │
   │ /kernels/{crypto,  │  │ metal}/kernels/  │
   │ gpu}/*.{cu,metal}  │  │ gpu/dex_swap.*   │
   └────────────────────┘  └──────────────────┘
```

The four arrows out of the precompile dispatcher are the only way EVM
bytecode reaches GPU work. Anything else is either CPU or a primitive that
is not yet wired.

## 2. Standard EVM precompiles (0x01..0x11)

Source of truth: `cevm/lib/evm/gpu/precompiles/precompile_dispatch.cpp`,
`cevm/lib/cevm_precompiles/`, `luxcpp/gpu/include/lux/gpu.h`.

| Addr | Name              | CPU impl (cevm)          | GPU primitive (luxcpp)      | Wired in cevm dispatcher                |
|-----:|-------------------|--------------------------|-----------------------------|-----------------------------------------|
| 0x01 | ECRECOVER         | `secp256k1.cpp`          | `lux_gpu_ecrecover_batch`   | yes — `ecrecover_metal.mm`, falls back to CPU on `valid==0` |
| 0x02 | SHA256            | `sha256.cpp`             | `lux_gpu_keccak256_batch` (Keccak only; SHA256 is a separate kernel in `metal/kernels/crypto/`, no public API) | no — CPU only |
| 0x03 | RIPEMD160         | `ripemd160.cpp`          | none                        | no — CPU only                           |
| 0x04 | IDENTITY          | `identity.cpp`           | n/a (memcpy)                | no — CPU only                           |
| 0x05 | MODEXP            | `modexp.cpp` (intx)      | none in public lux/gpu API; `modular.{cu,metal}` exists in luxcpp/cuda but is unrelated NTT/lattice work | no — CPU only |
| 0x06 | BN256_ADD         | `bn254.cpp` → `pairing/bn254/` | `lux_bn254_add` (G1 + G2 batch add) | YES (this commit) — `bn254_metal.mm` |
| 0x07 | BN256_MUL         | `bn254.cpp`              | `lux_bn254_mul` (G1 + G2 batch scalar mul) | YES (this commit) — `bn254_metal.mm` |
| 0x08 | BN256_PAIRING     | `pairing/bn254/pairing.cpp` (113K gas — the big one) | **none** — `lux/gpu.h` exposes BLS12-381 pairing but no BN254 pairing | NO — biggest gas win is blocked on a missing kernel |
| 0x09 | BLAKE2F           | `blake2b.cpp`            | none in public API; no `blake2f` in luxcpp/{cuda,metal} | no — CPU only |
| 0x0a | KZG_POINT_EVAL    | `kzg.cpp` (c-kzg/blst)   | `lux_kzg_verify` exists; needs versioned-hash + SHA-256 + EIP-4844 layout glue | no — installer stub only; CPU ref is consensus-correct |
| 0x0b | BLS12_G1ADD       | `bls.cpp` (blst)         | `lux_bls12_381_add(is_g2=false)` | no — installer stub only |
| 0x0c | BLS12_G1MSM       | `bls.cpp`                | `lux_bls12_381_mul` + `lux_msm` | no — installer stub only |
| 0x0d | BLS12_G2ADD       | `bls.cpp`                | `lux_bls12_381_add(is_g2=true)` | no — installer stub only |
| 0x0e | BLS12_G2MSM       | `bls.cpp`                | `lux_bls12_381_mul(is_g2=true)` | no — installer stub only |
| 0x0f | BLS12_PAIRING     | `bls.cpp`                | `lux_bls12_381_pairing`     | no — installer stub only; would be a large gas win |
| 0x10 | BLS12_MAP_FP_G1   | `bls.cpp`                | none                        | no — CPU only                           |
| 0x11 | BLS12_MAP_FP2_G2  | `bls.cpp`                | none                        | no — CPU only                           |

**Why several rows say "stub" on the GPU side**: the EIP-2537 BLS12-381
precompiles require strict input parsing (64-byte field elements, leading
16-byte zero padding, R-subgroup checks) which `blst` handles natively in
`bls.cpp`. The Metal primitives in `luxcpp/gpu/kernels/bls12_381.metal`
implement curve operations but do not implement EIP-2537 input validation.
Wiring them today would be wrong (consensus divergence) — the dispatcher's
CPU default is the consensus-safe choice. Same shape applies to KZG: the
pairing primitive exists; the EIP-4844 SHA-256 + versioned-hash + trusted
setup glue does not.

## 3. Lux custom precompiles (0x100..0x1ff)

| Addr  | Name        | Calldata layout                                          | CPU ref                        | GPU primitive       | Wired |
|------:|-------------|----------------------------------------------------------|--------------------------------|---------------------|------|
| 0x100 | DEX_MATCH   | `side(1) ‖ price(32) ‖ qty(32) ‖ user(20) ‖ book_id(32)` | new — `dex_match_cpu.cpp`      | `lux_match_orders`  | YES (this commit) |

Range `0x101..0x1ff` is reserved for additional Lux precompiles (settlement,
liquidity, on-chain TWAP). New entries get registered the same way 0x100 is
in `precompile_dispatch.cpp`.

The dispatcher's address parameter was widened from `uint8_t` to `uint16_t`
so this range fits without changing how 0x01..0x11 are addressed.

## 4. lux-accel C ABI surface

Source of truth: `luxcpp/lux-accel/include/lux/accel/c_api.h`, `nm
liblux_accel.0.1.0.dylib`.

### 4.1 What the symbol table actually exports

```
$ nm liblux_accel.0.1.0.dylib | grep " T _lux_" | sort
_lux_attention             _lux_kyber_decaps           _lux_session_create
_lux_bfv_add               _lux_kyber_encaps           _lux_session_create_with_backend
_lux_bfv_decrypt           _lux_kyber_keygen           _lux_session_create_with_device
_lux_bfv_encrypt           _lux_layer_norm             _lux_session_destroy
_lux_bfv_multiply          _lux_load_backend           _lux_session_get_device_info
_lux_bls_verify_batch      _lux_match_orders           _lux_session_sync
_lux_compute_twap          _lux_matmul                 _lux_sha256
_lux_constant_product_swap _lux_merkle_root            _lux_shutdown
_lux_dilithium_sign        _lux_msm                    _lux_softmax
_lux_dilithium_verify      _lux_ntt                    _lux_tensor_*
_lux_ecdsa_verify_batch    _lux_poly_mul               _lux_version
_lux_ed25519_verify_batch  _lux_poseidon
_lux_gelu                  _lux_relu
_lux_init                  _lux_keccak256
_lux_intt
```

### 4.2 DEX subset, current state

| Symbol                       | C++ impl in `lux-accel/src/dex.cpp` | Notes                                      |
|------------------------------|-------------------------------------|--------------------------------------------|
| `lux_match_orders`           | stub (`NotSupported`)               | wired through Go `accel.DEXOps.MatchOrders`; precompile 0x100 calls this |
| `lux_constant_product_swap`  | stub (`NotSupported`)               | Go `accel.DEXOps.ConstantProductSwap`       |
| `lux_compute_twap`           | stub (`NotSupported`)               | Go `accel.DEXOps.ComputeTWAP`               |

The C symbols exist (so the precompile dispatcher and Go binding compile
and run), but the implementations in `lux-accel/src/dex.cpp` return
`gpu::Status::Error(NotSupported, ...)`. The Metal/CUDA shaders below
exist in `luxcpp/{cuda,metal}/kernels/gpu/dex_swap.*` but are **not** linked
into `liblux_accel`. Wiring them is the next step; the precompile
dispatcher today falls back to the cevm CPU implementation when it gets
`NotSupported`, which keeps consensus correct while we land the GPU path.

### 4.3 What the Go `accel.DEXOps` interface advertises but C ABI does not expose

| Go method                    | C ABI symbol                            | Status              |
|------------------------------|-----------------------------------------|---------------------|
| `MatchOrdersWithPriority`    | (not in `c_api.h`)                      | Go returns `ErrNotSupported` |
| `ComputeLiquidity`           | (not in `c_api.h`)                      | same                |
| `ComputePositionValue`       | (not in `c_api.h`)                      | same                |
| `CalculateFees`              | (not in `c_api.h`)                      | same                |
| `BatchSettlement`            | (not in `c_api.h`)                      | same                |
| `ConstantProductSwapBatch`   | (not in `c_api.h`)                      | same                |

These appear in `lux/accel/ops_dex.go` as interface methods but have no
backing C symbols. They are not callable today.

## 5. Raw GPU kernels in luxcpp

### 5.1 Crypto (`luxcpp/{cuda,metal}/kernels/crypto/`)

Every file is mirrored across CUDA and Metal unless noted.

| File                       | Lines (Metal) | Public API in `lux/gpu.h`?         |
|----------------------------|---------------|------------------------------------|
| `secp256k1.metal`          | recover, multi-sign | `lux_gpu_ecrecover_batch`     |
| `bn254.metal`              | add (G1+G2), scalar mul (G1+G2), pedersen_commit | `lux_bn254_add`, `lux_bn254_mul` |
| `bls12_381.metal`          | add, mul, pairing  | `lux_bls12_381_add`, `lux_bls12_381_mul`, `lux_bls12_381_pairing` |
| `kzg.metal`                | commit, verify | `lux_kzg_commit`, `lux_kzg_verify` |
| `poseidon.metal`           |                | `lux_poseidon2_hash` (BN254 variant) |
| `poseidon2_bn254.metal`    |                | (used internally by lux_poseidon2_hash) |
| `blake3.metal`             |                | `lux_blake3_hash`                  |
| `frost_*.metal`            |                | (FROST threshold sigs, no public API yet) |
| `ringtail_*.metal`         |                | (Ringtail PQ, no public API yet)   |
| `mldsa_verify.metal`       |                | (ML-DSA, no public API yet)        |
| `goldilocks.metal`         |                | (Plonky2 field, no public API yet) |
| `msm.metal`                |                | `lux_msm`                          |
| `modular.metal`            |                | (NTT/lattice helper)               |
| `attestation.metal`        |                | (private)                          |
| `shamir_interpolate.metal` |                | (private)                          |

### 5.2 GPU primitives (`luxcpp/{cuda,metal}/kernels/gpu/`)

| File                                | Kernels                                                                      |
|-------------------------------------|------------------------------------------------------------------------------|
| `cuda/kernels/gpu/dex_swap.cu`      | `match_order_kernel`, `batch_match_kernel`, `aggregate_levels_kernel`, `find_best_prices_kernel`, `bitonic_sort_step_kernel`, `execute_swaps_kernel`, `aggregate_balance_updates_kernel` |
| `metal/kernels/gpu/dex_swap.metal`  | `batch_swap`, `batch_liquidity`, `batch_route`, `batch_tick_to_sqrt_price`, `batch_next_initialized_tick`, `batch_price_impact` |

CUDA covers order-book matching; Metal covers AMM swap & liquidity.
**They are not symmetric**. A future cross-platform DEX kernel set must
either standardise on one of these surfaces or maintain two backends in
lux-accel — currently neither shader is compiled into `liblux_accel`.

### 5.3 EVM-related GPU primitives

`luxcpp/gpu/kernels/evm256.{cu,metal}` — uint256 arithmetic for the GPU EVM
interpreter. Used by cevm's own kernel
(`cevm/lib/evm/gpu/kernel/evm_kernel.metal`), not by lux-accel.

`luxcpp/gpu/kernels/keccak256.{cu,metal}` and
`cevm/lib/evm/gpu/cuda/keccak256.cu` — both exist; cevm uses the latter
directly when `EVM_CUDA` is set, otherwise it routes through luxcpp/gpu's
`lux_gpu_keccak256_batch`.

`luxcpp/gpu/kernels/secp256k1_recover.{cu,metal}` — used directly by
`ecrecover_metal.mm` / `ecrecover_cuda.cpp`.

## 6. cevm's own GPU EVM kernel

Not part of the precompile path — this is the opcode-level interpreter
that runs when `Config::backend == GPU_Metal` or `GPU_CUDA`.

| File                                              | Lines | Role                                              |
|---------------------------------------------------|------:|---------------------------------------------------|
| `cevm/lib/evm/gpu/kernel/evm_kernel.metal`        | ~1300 | Metal compute shader: full Cancun-revision interpreter |
| `cevm/lib/evm/gpu/kernel/evm_kernel_host.{hpp,mm}`| ~700  | Metal host: device, queue, pipeline cache, dispatch |
| `cevm/lib/evm/gpu/cuda/evm_kernel.cu`             | ~740  | CUDA equivalent (gated on `CEVM_CUDA`)            |
| `cevm/lib/evm/gpu/cuda/evm_kernel_host.{cpp,hpp}` |       | CUDA host                                         |
| `cevm/lib/evm/gpu/gpu_dispatch.{cpp,hpp}`         |       | The Backend selector + `execute_block` entry      |
| `cevm/lib/evm/gpu/parallel_engine.{cpp,hpp}`      |       | CPU Block-STM scheduler used by `CPU_Parallel`    |

The 4-way parity test (`evm-parity-test`) drives every vector through
`CPU_Sequential`, `CPU_Parallel`, `GPU_Metal`, and (when CUDA is built)
`GPU_CUDA`. Current count: **133 / 133** all backends agree.

## 7. Status summary (one-liner)

- 1 EVM precompile (0x01) — wired to GPU, falls back to CPU on edge cases.
- 2 EVM precompiles (0x06, 0x07) — wired to GPU in this commit.
- 1 Lux precompile (0x100) — added in this commit; CPU first, GPU path
  installed but currently lands on CPU because `lux_match_orders` is a
  stub. Activates the moment `lux-accel/src/dex.cpp::match_orders` returns
  `gpu::Status::OK`.
- 14 EVM precompiles + 5 BLS12-381 — CPU only. Most have a kernel-level
  primitive in luxcpp/{cuda,metal} but are blocked on either (a) EIP layout
  glue (BLS, KZG) or (b) a missing kernel (BN256_PAIRING, BLAKE2F,
  RIPEMD160, MODEXP, BLS12_MAP_*).

When a row in §2 or §3 changes status, update it here in the same commit.
