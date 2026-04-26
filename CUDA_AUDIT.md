# Quasar CUDA backend — syntactic audit (2026-04-26)

Audit of `lib/consensus/quasar/gpu/quasar_wave.cu` (1034 lines) and
`lib/consensus/quasar/gpu/quasar_gpu_engine_cuda.cpp` (653 lines).
This is a pure walkthrough — no NVCC available on the Apple host. All
issues called out here are syntactic / API-level, not behavioural; the
behavioural test will be in `.github/workflows/quasar-cuda-build.yml`
on a self-hosted H100 runner.

## Audit checklist

### 1. `__global__` vs `__device__` qualifiers

* `quasar_wave_kernel` — **`__global__`** at line 893. Correct: this is
  the kernel entry launched by `quasar_wave_launch` via `<<<grid,
  block, 0, stream>>>`. Returns `void`, takes only device pointers.
* All drain helpers (`drain_ingress`, `drain_decode`, `drain_crypto`,
  `drain_dagready`, `drain_exec`, `drain_validate`, `drain_repair`,
  `drain_commit`, `drain_state_resp`, `drain_vote`) — **`__device__
  static`**. Correct: called only from the kernel, no host visibility
  needed.
* `keccak_f1600`, `keccak256_local`, `mvcc_locate`, `mvcc_index`,
  `mvcc_check_consistent`, `mvcc_apply_writes`, `receipt_hash`,
  `verify_signature_stub` — **`__device__ static`** (or
  `__device__ __forceinline__` for hot helpers). Correct.
* `ring_try_push<T>` / `ring_try_pop<T>` —
  **`__device__ __forceinline__`** template. Correct; templates are
  resolved at compile time and the `__device__` annotation propagates
  to every instantiation.
* `quasar_wave_launch` — **`extern "C"` host launcher** in `.cu` file,
  has no `__global__`/`__device__` qualifier. Correct: this is host
  code that emits the `<<<>>>` launch (only NVCC parses it), and the
  `.cpp` driver picks it up via the matching `extern "C"` declaration
  at `quasar_gpu_engine_cuda.cpp:43`.

**Verdict: PASS.** Qualifier hygiene matches the CUDA programming
guide. No functions called from host without `__global__` or
`extern "C"` linkage.

### 2. `atomicCAS` / `atomicAdd` signatures

CUDA SDK signatures (CUDA 12.x):

```c
unsigned int atomicAdd (unsigned int* address, unsigned int val);
unsigned int atomicCAS (unsigned int* address, unsigned int compare, unsigned int val);
unsigned int atomicExch(unsigned int* address, unsigned int val);
```

Usages in `quasar_wave.cu`:

| Site | Call | Args | Notes |
| --- | --- | --- | --- |
| L278 | `atomicAdd(&h->head, 0u)` | `uint32_t*, uint32_t` | Idiomatic atomic load. Returns the *current* value because val=0. PASS. |
| L279 | `atomicAdd(&h->tail, 0u)` | `uint32_t*, uint32_t` | Same pattern. PASS. |
| L283 | `atomicExch(&h->tail, tail + 1u)` | `uint32_t*, uint32_t` | Publish a new tail value. PASS. |
| L284 | `atomicAdd(&h->pushed, 1u)` | `uint32_t*, uint32_t` | Counter increment. PASS. |
| L295 | `atomicCAS(&h->head, head, head + 1u)` | `uint32_t*, uint32_t, uint32_t` | Returns observed; success if `observed == head`. PASS. |
| L489 | `atomicAdd(tx_index_seq, 1u)` | `uint32_t*, uint32_t` | Monotonic tx_index. PASS. |
| L529 | `atomicAdd(&result->fibers_suspended, 1u)` | `uint32_t*, uint32_t` | Counter. PASS. |
| L619 | `atomicAdd(&table[idx].version, 0u)` | `uint32_t*, uint32_t` | Atomic load of version. PASS. |
| L633-635 | `atomicExch`/`atomicAdd` on MvccSlot fields | `uint32_t*, uint32_t` | Writer publishes (last_writer_tx, last_writer_inc) + bumps version. PASS. |
| L704 | `atomicAdd(&result->conflict_count, 1u)` | `uint32_t*, uint32_t` | Counter. PASS. |
| L759, L787, L802 | counter increments on QuasarRoundResult | all `uint32_t*, uint32_t` | PASS. |
| L805-810 | `atomicAdd(&result->gas_used_lo, gas_lo)` + carry | `uint32_t*, uint32_t` | Lo + carry-into-hi pattern. PASS — the host reassembles via `(hi << 32) | lo`. |
| L863 | `atomicAdd(stake_acc, v.stake_weight)` | `uint32_t*, uint32_t` | Quorum stake accumulator. PASS. |
| L878, L911, L990 | `atomicExch` on status fields | `uint32_t*, uint32_t` | One-shot transitions. PASS. |

**Verdict: PASS.** All atomic calls are well-typed `(uint32_t*, ...)`
forms that match CUDA 12.x signatures. No 64-bit atomic uses (which
would require sm_60+ specifically — H100 has them, but the kernel
sticks to 32-bit ops everywhere, which is portable down to sm_50).

### 3. `__threadfence()` placement vs MSL `threadgroup_barrier(mem_flags::mem_device)`

The Metal kernel uses `threadgroup_barrier(mem_flags::mem_device)`
between the `items[tail]` write and the `atomic_store` of `tail` on
the producer side, and between the `atomic_load` of `tail` and the
`items[head]` read on the consumer side.

CUDA equivalent is `__threadfence()` (device-wide visibility, weaker
than `__threadfence_system` which crosses the host boundary).
`__threadfence()` is the right primitive because the producer and
consumer CTAs are both on the device and we don't need host ordering
(host sees state via `cudaMemcpy` after `cudaStreamSynchronize`).

| Producer side (`ring_try_push`, line 276-286): | Status |
| --- | --- |
| L281: `items[tail & h->mask] = v;` | Write payload first |
| L282: `__threadfence();` | Publish before tail update |
| L283: `atomicExch(&h->tail, tail + 1u);` | Atomically advance tail |
| L284: `atomicAdd(&h->pushed, 1u);` | Counter post-publish |
| **PASS** — same write-fence-publish recipe as MSL. | |

| Consumer side (`ring_try_pop`, line 289-304): | Status |
| --- | --- |
| L295: `atomicCAS(&h->head, head, head + 1u)` | Reserve a slot |
| L297: `__threadfence();` | Acquire-fence before payload read |
| L298: `out = items[head & h->mask];` | Read payload after fence |
| L299: `atomicAdd(&h->consumed, 1u);` | Counter |
| **PASS** — fence-on-acquire matches MSL release-acquire ring. | |

The kernel itself does not need a between-service barrier because each
service is its own CTA (`gridDim.x = kNumServices = 12`) and the
between-service hand-off is via the rings, which already have the
fence around them. **Verdict: PASS.**

### 4. Shared `__constant__` memory for keccak constants

Current state: `kKeccakRC[24]` and `kKeccakRot[25]` are declared
**`__device__ static const`** at lines 311 / 326. This places them in
`.rodata` (constant memory in CUDA terms — but specifically the
*generic* constant memory, not the explicit `__constant__` pool).

For tables this small (24 × 8 = 192 bytes; 25 × 4 = 100 bytes) the
compiler will normally promote them into the constant cache anyway,
but there is a small win to be had by using `__constant__ static const`
explicitly: it guarantees they're routed through the constant cache
even if the compiler chooses not to promote, and on H100 the constant
cache has a dedicated 64 KiB pool that doesn't compete with the L1.

**Recommendation (low-priority, perf only — not a correctness issue):**

```cuda
__constant__ static const uint64_t kKeccakRC[24]  = { /* … */ };
__constant__ static const uint32_t kKeccakRot[25] = { /* … */ };
```

Caveat: `__constant__` arrays must be initialised at file scope and
cannot be modified at runtime; the current values satisfy that.

**Verdict: ACCEPTABLE AS-IS.** Both placements give constant-cache
routing on modern arches; switching to `__constant__` is a clean-up
nit, not a fix. Mirrors the MSL `constant uchar kKeccakRC[…]` exactly.

### 5. Buffer indexing parity vs Metal kernel

The decisive structural points to verify:

* **Service count / CTA grid.** MSL: `gid % kNumServices` over a
  flat workgroup count of 12. CUDA: `gridDim.x = kNumServices = 12`,
  `gid = blockIdx.x`, `if (gid >= kNumServices) return`. Match. PASS.
* **Per-service drain dispatch.** MSL: an `if (gid == 0)` ladder.
  CUDA: identical `if (gid == 0u) ...; else if (gid == 1u) ...`
  ladder at line 944-970, gid 0..7 + 9 + 10. **PASS.**
* **gid 8 (StateRequest) and gid 11 (QuorumOut)** are intentionally
  empty in both backends — the host polls these rings via
  `poll_state_requests` / `poll_quorum_certs`, no on-device drain
  needed. Match. PASS.
* **`thread 0 only` filter.** Both backends: `if (tid != 0u) return;`
  at the top of the kernel body. CUDA line 907; MSL has the same
  `tid != 0u` guard (each block has 32 threads but only thread 0 does
  the drain). PASS.
* **Ring header / items split.** MSL: `device RingHeader* hdr` +
  `items_ofs` byte offset into a flat `device uchar* items_arena`.
  CUDA mirrors exactly: line 916-928 reconstructs each `RingHeader*`
  by indexing `hdrs + N`, and each items pointer via
  `(T*)(items_arena + hdr->items_ofs)`. PASS — matches the Metal host
  arena layout.
* **`closing_flag` finalization.** Both backends gate the
  block_hash / mode_root composition on `gid == 0 && closing_flag != 0`
  AND `ingress.pushed == commit.consumed`. CUDA line 974-992 mirrors
  MSL line 1240-1280. PASS.

**Verdict: PASS.** Buffer indexing is byte-identical with the MSL
kernel modulo language syntax. Same arena geometry, same per-service
CTA placement, same gid → drain mapping.

## Other findings

### Minor (cosmetic)

* `quasar_wave.cu` line 489: `out.tx_index = atomicAdd(tx_index_seq, 1u);`
  The MSL kernel has a v0.42 in-flight tweak that **peeks** the
  counter before the decode-ring push and only increments after a
  successful push, to avoid orphaned tx_index slots when the decode
  ring is full. The CUDA kernel still uses the v0.40 unconditional
  increment. This is a behavioural difference, not a syntax bug —
  both work, but the v0.42 MSL pattern is strictly better when the
  decode ring is at capacity. **Recommendation:** port the v0.42 MSL
  pattern to CUDA when v0.42 lands on main.
* `quasar_wave.cu` line 233: `quorum_stake_*` are flat `uint32_t`
  fields, matching the v0.40 layout. The in-flight v0.42 MSL kernel
  splits these into `_lo` / `_hi` for 64-bit stake. Not a CUDA
  problem — the CUDA backend simply needs to track whichever shape
  ends up landing in `quasar_gpu_layout.hpp`.

### Not observed (i.e. things we checked for and didn't find)

* No use of `atomicAdd_block` / `atomicCAS_block` — kernel uses the
  full-device variants throughout, which is correct because the
  rings are written/read across CTAs.
* No `__syncthreads()` — would only be needed for in-CTA sharing,
  but the kernel is single-thread-per-CTA. Correct to omit.
* No `cooperative_groups::*` headers — not needed at this layer.
* No `__shfl_*` / `__ballot_*` warp primitives — correct;
  lanes 1..31 are dormant in v0.40 and only get woken in v0.42+
  for the SIMT EVM ops.
* No host-side memory passed without `cudaMemcpy*` first — the
  `quasar_gpu_engine_cuda.cpp` driver allocates everything via
  `cudaMalloc` and only memcpys the diff regions per push.

## How to verify on real hardware

The Apple host has no NVCC. To finish the verification:

```bash
# On a machine with CUDA Toolkit ≥ 12.0 + sm_80 or sm_90 GPU:
cmake -S . -B build-cuda \
      -DCEVM_CUDA=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="80;90"
cmake --build build-cuda --target evm-cuda -j
cmake --build build-cuda --target quasar-determinism-test -j
./build-cuda/lib/evm/quasar-determinism-test
```

Expected: 6/6 cases pass with byte-identical `QuasarRoundResult` snapshots
(same-engine repeats and two-engine cross-instance), matching the Metal
output of `./build-cov/lib/evm/quasar-determinism-test`.

Cross-backend Metal-vs-CUDA byte parity is a separate test
(`quasar_xbackend_determinism_test`, deferred until both backends are
available on the same runner — likely v0.47).

## Summary

| Item | Result |
| --- | --- |
| `__global__` / `__device__` hygiene | PASS |
| `atomicCAS` / `atomicAdd` / `atomicExch` signatures | PASS |
| `__threadfence()` placement vs MSL `mem_device` barriers | PASS |
| `__constant__` memory for keccak tables | ACCEPTABLE (`__device__ static const` is equivalent on modern arches; a `__constant__` upgrade is a perf nit) |
| Buffer indexing / CTA grid parity | PASS |

No CUDA backend syntax issues found that would prevent NVCC compilation
or cause divergence from the Metal kernel. The CUDA kernel is ready for
its first run on a real H100 via the workflow at
`.github/workflows/quasar-cuda-build.yml`.
