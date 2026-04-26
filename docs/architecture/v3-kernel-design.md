# cevm V3 GPU kernel — saturating the device with Block-STM + Wave/FPC

This is the v0.28+ architecture target. v0.28 lands the foundation
(per-tx scratch sizing, async dispatch, SIMD-cooperative interpreter).
v0.29 builds the persistent + pipelined V3 kernel on top.

The goal is not "GPU faster than CPU at 16K txs". The goal is **the
GPU never goes idle while there is consensus work to do** — meaning:

  * Block-STM's optimistic execution + validation runs continuously
  * Lux Quasar Waves overlap on the GPU (Wave[t+1] starts before
    Wave[t] finalises)
  * FPC's early-commit signal short-circuits validation for txs that
    are already certified

Below is the layered design, top-down: hardware mapping → parallelism
axes → MVCC layout → Wave/FPC integration → persistent-kernel skeleton
→ honest limits.

---

## 1. Hardware mapping (Apple M1 Max numbers; CUDA analogues parenthesised)

| Tier               | M1 Max                              | What lives there                         |
|--------------------|-------------------------------------|------------------------------------------|
| Registers          | per-thread, ~256 × 32-bit           | tx PC, gas, top-of-stack pointer         |
| Threadgroup memory | 32 KiB per workgroup (≅ smem)       | warp-shared hot state, opcode dispatch LUT, recent stack window |
| L1 / L2            | implicit                            | bytecode, calldata, MVCC version index   |
| Device memory      | unified 64 GiB                      | per-tx scratch (sized!), MVCC stores, queues |
| Shared with CPU    | unified (M1) / pinned (CUDA)        | block headers, FPC certificates, mempool |

M1 Max steady-state: 32 cores × 32 SIMD lanes = **1024 hw threads**.
That's the saturation target. Anything that leaves the device idle for
>5 µs at this scale is a bug.

---

## 2. Parallelism axes (orthogonal — exploit all four)

```
  axis A: inter-tx Block-STM      ──> N txs in a wave run speculatively
  axis B: intra-tx SIMD           ──> 32 threads cooperate on one tx's
                                      hot opcode (keccak, ecrecover,
                                      modexp, bn254/bls12 pairings)
  axis C: inter-wave pipeline     ──> Wave[t+1] starts while Wave[t]
                                      validates / commits
  axis D: inter-block speculation ──> if FPC certified blocks A and B,
                                      pre-execute their successors
                                      against a speculative MVCC frontier
```

Most current EVM-on-GPU papers exploit axis A only. Lux Quasar gives us
axes C and D as a free dividend on top of Block-STM, *if* the kernel is
queue-driven instead of dispatch-driven.

---

## 3. MVCC layout on device

Block-STM needs multi-version reads. Naïve "copy state per tx" doesn't
fit on GPU. Use a Structure-of-Arrays layout indexed by (slot_hash,
incarnation):

```c++
// device-side, in evm_kernel_v3.metal:
struct SlotEntry {
    uint64_t key_hi;      // first 8 bytes of slot hash
    uint64_t key_lo;      // last 8 bytes of slot hash
    uint32_t head_idx;    // index into Versions[] of latest committed
    uint32_t pending_idx; // index of latest speculative (if any)
};

struct Version {
    uint8_t  value[32];   // 256-bit slot value
    uint32_t writer_tx;   // tx index that wrote this version
    uint32_t incarnation; // Block-STM incarnation number
    uint32_t prev_idx;    // chain-back to earlier version
    uint32_t flags;       // {COMMITTED, FPC_CERTIFIED, SPECULATIVE}
};

device SlotEntry  slots[NUM_SLOTS];   // open-addressed hash table
device Version    versions[NUM_VERS]; // arena
device atomic_uint version_alloc;     // bump pointer
```

Reads:
  1. Hash the slot key, probe `slots[]`.
  2. Walk `versions[]` chain back from `pending_idx` until we find a
     version with `incarnation ≤ this_tx_incarnation`.
  3. Append `(slot_id, observed_version_idx)` to this tx's read-set.

Writes:
  1. Allocate a new `Version` via `atomic_fetch_add(version_alloc, 1)`.
  2. CAS `slots[slot_id].pending_idx` to the new index (replay tx if CAS
     fails — another tx beat us; that's a Block-STM conflict).
  3. Append `(slot_id, new_version_idx)` to this tx's write-set.

Validation (run after the wave finishes execution):
  * Every tx checks its read-set: for each (slot_id, observed_idx),
    confirm no committed write between `observed_idx` and `slots[slot_id]
    .head_idx` came from a tx with smaller index.
  * If validation fails, mark tx for re-execution at incarnation+1.
  * Validation is itself a parallel kernel — every read-set entry is
    one independent thread.

---

## 4. Wave / FPC integration

Quasar emits two consensus signals the GPU can use:

  * **Wave**: a topological frontier of txs that can be executed
    concurrently — these are Block-STM's `N` per launch.
  * **FPC certificate**: probabilistic early-commit for a tx, meaning
    enough validators have seen it that final ordering is determined.
    Fires *before* finality — typically 100–300 ms early.

V3 reacts to these signals *on the GPU* without round-tripping to the
CPU scheduler:

```
host queues:
  +-----------------+  +-----------------+  +-----------------+
  | wave_in[]       |  | fpc_certs[]     |  | finality_in[]   |
  +-----------------+  +-----------------+  +-----------------+
        |                     |                     |
        v                     v                     v
   exec_workers          fpc_promote          finalise_commit
  (persistent kernel)   (persistent kernel)  (persistent kernel)
        |                     |                     |
        v                     v                     v
  +----------------------------------------+
  | shared MVCC store (versions[], slots[])|
  +----------------------------------------+
        |
        v
  +----------------------------------------+
  | committed state delta (out to CPU)     |
  +----------------------------------------+
```

`exec_workers`: takes (wave_id, tx_index, incarnation), executes,
writes pending versions.

`fpc_promote`: takes a tx_id, walks its write-set, flips every pending
version's flag from `SPECULATIVE` to `FPC_CERTIFIED`. Subsequent reads
from later txs treat FPC_CERTIFIED as "do not re-validate against
this" — saves a validation roundtrip per dependent tx.

`finalise_commit`: takes a wave_id, walks the wave's write-set,
flips `FPC_CERTIFIED → COMMITTED`, advances `slots[].head_idx`, garbage-
collects dropped speculative versions.

All three are **persistent kernels**: they spin on their input queue and
self-dispatch. Host emits work via lock-free SPSC queues in unified
memory. No `dispatch_thread()` per wave; one launch covers the entire
block-production lifetime.

---

## 5. Persistent kernel skeleton (Metal-flavoured pseudocode)

```metal
kernel void exec_worker(
    device WorkItem*       work_q   [[ buffer(0) ]],
    device atomic_uint*    work_head[[ buffer(1) ]],
    device atomic_uint*    work_tail[[ buffer(2) ]],
    device SlotEntry*      slots    [[ buffer(3) ]],
    device Version*        versions [[ buffer(4) ]],
    device atomic_uint*    version_alloc[[ buffer(5) ]],
    device const uint8_t*  bytecode_arena [[ buffer(6) ]],
    device const uint32_t* bytecode_offsets [[ buffer(7) ]],
    device       uint8_t*  scratch_arena   [[ buffer(8) ]],
    device const uint32_t* scratch_offsets [[ buffer(9) ]],
    device TxResult*       results   [[ buffer(10) ]],
    threadgroup uint8_t*   tg_hot   [[ threadgroup(0) ]], // 16 KiB
    uint   tid              [[ thread_index_in_threadgroup ]],
    uint   gid              [[ threadgroup_position_in_grid ]],
    uint   gtid             [[ thread_position_in_grid ]])
{
    while (true) {
        // One workgroup pulls one work item; 32 threads cooperate on it.
        threadgroup uint local_item;
        if (tid == 0) {
            uint h = atomic_fetch_add_explicit(work_head, 1, memory_order_relaxed);
            uint t = atomic_load_explicit(work_tail, memory_order_acquire);
            local_item = (h < t) ? h : UINT32_MAX;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (local_item == UINT32_MAX) {
            // Queue empty → backoff (yield via dummy threadgroup_barrier),
            // then re-check. Or break if shutdown flag set.
            if (atomic_load_explicit(shutdown_flag, memory_order_acquire))
                return;
            continue;
        }

        WorkItem w = work_q[local_item];

        uint scratch_off = scratch_offsets[w.tx_index];
        uint scratch_len = scratch_offsets[w.tx_index + 1] - scratch_off;
        device uint8_t* my_scratch = scratch_arena + scratch_off;

        uint code_off = bytecode_offsets[w.tx_index];
        uint code_len = bytecode_offsets[w.tx_index + 1] - code_off;

        // Co-operative interpreter: thread 0 owns PC+stack. Other threads
        // are recruited on intrinsically parallel ops (keccak, ecrecover,
        // big-int mul, MCOPY). Idle threads spin on a shared opcode token.
        execute_tx_simd_coop(
            bytecode_arena + code_off, code_len,
            my_scratch, scratch_len,
            tg_hot,
            slots, versions, version_alloc,
            w.tx_index, w.incarnation,
            &results[w.tx_index],
            tid);

        // Result writeback by thread 0 only.
        if (tid == 0) {
            // Append to validation queue if execution succeeded.
            uint vt = atomic_fetch_add_explicit(validation_tail, 1,
                                                memory_order_release);
            validation_q[vt] = w.tx_index;
        }
    }
}
```

Key details:

  * **One workgroup per tx, 32 threads cooperating** (axis B).
  * Workgroup pulls from a SPMC queue (single producer = host scheduler,
    multi consumer = GPU workgroups). M1 supports `atomic_uint` with
    relaxed memory order for the head; the producer uses release on tail.
  * `tg_hot` is threadgroup memory: 16 KiB scratch for the warp's
    interpreter state — top-of-stack window, recent memory page, opcode
    fetch buffer. Stays resident for the whole tx; only spills to
    `my_scratch` (device memory) on overflow.
  * Persistent: workers live for the whole block-production session.
    Host queues new waves; workers drain.

---

## 6. Intra-tx SIMD-cooperative opcodes (axis B detail)

Most opcodes (PUSH, ADD, JUMP, MSTORE) are intrinsically serial within
one tx — thread 0 executes, threads 1–31 idle on a barrier. That's
fine; the inter-tx parallelism via different workgroups soaks up the
device.

But four opcode families dominate gas:

| Opcode               | Gas cost | Intra-tx parallel work | Speedup target |
|----------------------|----------|------------------------|----------------|
| KECCAK256            | 30+6/w   | 1600-bit state, 24 rounds, theta/rho/pi/chi parallel by 5×5 | 8–16× |
| ECRECOVER (precomp.) | 3000     | secp256k1 scalar mul, ~256 doublings + adds | 16× |
| BN254/BLS pairings   | 113K+    | Miller loop, final exponentiation | 32× |
| MODEXP               | dyn.     | multi-limb mul/redc, ~32 limbs cooperatively | 4–32× |

For each, V3 has a 32-thread SIMD-cooperative variant that thread 0
calls into, keeping the rest of the warp busy on the same tx instead
of spinning.

---

## 7. Wave-pipelining (axis C)

Without pipelining, GPU is idle during validation:

```
[exec wave 0][validate 0][exec wave 1][validate 1]...
              ^idle GPU       ^idle GPU
```

With pipelining:

```
[exec wave 0]                     [exec wave 2]
              [validate 0]
                            [exec wave 1]
                                          [validate 1]
                                                       ...
GPU utilisation: ~100% as long as queue depth ≥ 2 waves.
```

Implementation: separate `validate_worker` persistent kernel, separate
queue. Once `exec_worker` finishes a tx, it pushes to the validate
queue. Once `validate_worker` confirms a tx's read-set, it pushes to
`finalise_commit`. Three pipelines, three queues, all draining
concurrently.

---

## 8. Inter-block speculation via FPC (axis D)

Lux's FPC certifies a tx as "will be in the canonical history" before
final ordering. Concretely: enough validators have voted such that
even adversarial reordering cannot evict the tx.

V3 use:
  * On FPC arrival for tx T, mark T's writes `FPC_CERTIFIED` (atomic
    flag flip on each version).
  * Schedule wave[t+1] *immediately* against the FPC-certified
    frontier without waiting for finality.
  * On the rare FPC reversal (~10⁻⁶ per tx in healthy networks),
    re-execute wave[t+1] from a snapshot. Cost: negligible at expected
    failure rate.

This is the lever that lets us "stack" three or four waves in flight
and saturate the GPU even when one wave's bytecode is light.

---

## 9. Static bytecode analysis (host-side, runs once per tx admission)

The host scheduler runs a single-pass analysis on each tx's bytecode
before queuing:

```c++
struct TxFootprint {
    uint16_t max_stack;      // 0..1024
    uint32_t max_mem_bytes;  // round to 32 (one EVM word)
    uint16_t max_storage_n;  // distinct slots SSTORE/SLOAD touched
    uint16_t max_output;     // RETURN/REVERT max size
    uint8_t  has_call;       // forces CPU-evmone fallback if true (no host on GPU)
    uint8_t  has_keccak;     // hint for kernel variant selection
    uint8_t  has_ecrecover;  // ditto
};
```

Walk: for each opcode, simulate stack delta, peek operand for static
PUSH/MSTORE/SSTORE targets, max-reduce per field. Unresolvable cases
fall back to per-opcode worst-case (e.g. dynamic SSTORE → assume each
op touches a new slot; bound at 64).

Output: the per-tx scratch size in bytes:
```
scratch[i] = max_stack[i] * 32       // EVM stack
           + max_mem_bytes[i]        // memory
           + max_storage_n[i] * 64   // (key, value) pairs
           + max_output[i]           // return data
           + 256                     // bookkeeping (pc, gas, flags, read-set head)
```

For the typical AMM swap workload above, this drops from 64 KiB to
~256 B — **256× less device memory per tx**.

---

## 10. Honest limits

* **Branch divergence**: EVM is dispatch-by-opcode. Threads in the same
  workgroup may diverge on JUMPI. Cooperative interpreter mitigates by
  having thread 0 do the dispatch and broadcast the PC, but cross-
  workgroup divergence (different txs at different opcodes) is
  unavoidable. Mitigation: workgroup-level scheduling so workgroups
  process one tx at a time, never split.

* **MVCC contention**: Hot slots (e.g. AMM reserve0) become atomic
  hotspots on `slots[].pending_idx`. CAS retry loops bound the
  damage. Counter-measure: bucket hot slots into local-shadow caches
  per workgroup, with a periodic flush. Pays off only when contention
  >>50%.

* **Small N regime**: Persistent-kernel overhead (queue probing,
  threadgroup barriers) dominates when wave size <256. Solution: keep
  CPU-Sequential as the small-N baseline; auto-route through the
  dispatcher.

* **Bytecode arena bandwidth**: 1M txs × 4 KiB bytecode worst case =
  4 GiB; M1 Max can stream that in ~10 ms but sustained reads cost
  cache pressure. If the same bytecode repeats (token contracts), use
  a content-addressed dedup arena.

* **Validation parallelism limit**: read-set sizes can vary 100×.
  Use prefix-scan partitioning to keep all 1024 hw threads busy
  during validation (a power-of-two tx index → power-of-two read
  segment).

---

## 11. Performance targets

Concrete numbers for the v0.29 V3 kernel on M1 Max, all relative to
v0.27 CPU-Par on the same hardware:

| Workload         | N        | v0.27 CPU-Par (ms) | V3 target (ms) | Target speedup |
|------------------|----------|-------------------:|---------------:|---------------:|
| amm_swap         | 16,384   | 29.2               | ≤ 2.9          | 10×            |
| amm_swap         | 1,000,000| ~1800 (extrap.)    | ≤ 50           | 36×            |
| compute_500      | 16,384   | 48.4               | ≤ 4.8          | 10×            |
| keccak_10        | 16,384   | 35.7               | ≤ 1.5          | 24× (intra-tx SIMD) |
| erc20_transfer   | 1,000,000| ~1800 (extrap.)    | ≤ 50           | 36×            |
| Wave-pipelined block stream (3 waves in flight) | continuous | n/a | n/a | ≥ 90% device util |

V3 is overdetermined: any one of {axis A, B, C, D} bought alone gets
to 4–6×. All four together is where 30–40× lives.

---

## 12. Build out order

* **v0.28** — landed by GPU specialist (in flight): per-tx scratch
  sizing, async dispatch, SIMD-cooperative interpreter for hot opcodes
  (keccak, ecrecover). Single-wave throughput 10× target.

* **v0.29** — V3 persistent kernel + MVCC on device + validation
  pipeline. Wave-pipelined throughput.

* **v0.30** — FPC integration: subscribe to certificates from luxd's
  consensus path, propagate to GPU's FPC promotion kernel. Inter-block
  speculation enabled.

* **v0.31** — opcode-level SIMD: 32-thread keccak, ecrecover, modexp.
  Exotic: 32-thread BN254/BLS pairings (the precompile-heavy workloads
  this finally lets the GPU shine on).

Each tag: parity 133/133, no fakes, real bench numbers committed under
`docs/benchmarks/`.
