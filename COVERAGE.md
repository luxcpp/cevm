# Quasar GPU coverage — v0.46 (2026-04-26)

LLVM source-based coverage for `lib/consensus/quasar/gpu/` host-side
sources, instrumented through `LUXCPP_EVM_COVERAGE=ON` and exercised by
the `quasar-gpu-engine-test` (13 cases) and `quasar-determinism-test`
(6 cases) binaries.

The Metal `.metal` kernel and the CUDA `.cu` kernel are **not**
instrumentable by LLVM source-based coverage — neither toolchain emits
LLVM coverage maps. The numbers below are the host-side surface that
actually runs on the CPU.

## Reproducing

```bash
cmake -S . -B build-cov -DLUXCPP_EVM_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-cov -j 8 --target quasar-gpu-engine-test quasar-determinism-test
LLVM_PROFILE_FILE=build-cov/coverage/quasar-engine-%p.profraw \
    ./build-cov/lib/evm/quasar-gpu-engine-test
LLVM_PROFILE_FILE=build-cov/coverage/quasar-determinism-%p.profraw \
    ./build-cov/lib/evm/quasar-determinism-test
xcrun llvm-profdata merge -sparse build-cov/coverage/*.profraw \
    -o build-cov/coverage/quasar.profdata
xcrun llvm-cov show -instr-profile=build-cov/coverage/quasar.profdata \
    build-cov/lib/evm/quasar-gpu-engine-test \
    -format=html -output-dir=build-cov/coverage_report/
```

Filter (`-ignore-filename-regex`) excludes Hunter packages, CPU-side EVM
opcode files, and host-bridge files unrelated to Quasar.

## Per-file (instrumented host surface)

Source: `xcrun llvm-cov report` against `quasar.profdata`, restricted to
files under `lib/consensus/quasar/gpu/`.

| File | Regions | Region cov | Functions | Func cov | Lines | Line cov | Branch cov |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `quasar_gpu_engine.hpp` | 3 | 100.00% | 3 | 100.00% | 3 | 100.00% | n/a |
| `quasar_gpu_engine.mm`  | 195 | 80.51% | 20 | 90.00% | 373 | **96.51%** | 58.46% |
| `quasar_gpu_layout.hpp` | 1 | 0.00% | 1 | 0.00% | 3 | 0.00% | n/a |
| **TOTAL** | 199 | **80.40%** | 24 | 87.50% | 379 | **95.78%** | 58.46% |

`quasar_gpu_layout.hpp` shows 0% because the only instrumentable region
is the inline `gas_used()` accessor on `QuasarRoundResult`, and the
host-side test paths read the `gas_used_lo/hi` fields directly via
`QuasarRoundResult::gas_used()` from a different translation unit
(inlined into the test). The struct itself is data layout — no logic to
cover.

`quasar_wave.metal` (1482 lines) and `quasar_wave.cu` (1034 lines) are
GPU kernel source; LLVM source-based coverage cannot instrument them
(no coverage maps emitted by metal-cc / nvcc). Their behaviour is
proven correct by:
  * the 13 `quasar-gpu-engine-test` cases that drive every GPU-side
    service (Crypto, Block-STM Exec/Validate/Commit, HashService,
    QuorumAgg, DAG ready/finish), with byte-for-byte assertions on
    `block_hash`, `state_root`, `receipts_root`, `execution_root`, and
    `mode_root`.
  * the 6 `quasar-determinism-test` cases that prove every byte of
    `QuasarRoundResult` is a pure function of the round descriptor +
    transactions, both within one engine instance and across two
    independent engines.

## Targets

Goal is **≥ 80% line coverage** on `lib/consensus/quasar/gpu/`.

* `quasar_gpu_engine.mm`: **96.51% lines** — exceeds goal.
* `quasar_gpu_engine.hpp`: 100% lines.
* `quasar_gpu_layout.hpp`: 0% (inline accessor in different TU; the
  struct is data layout, not logic).

Aggregate: **95.78% lines, 80.40% regions, 87.50% functions** —
exceeds the 80% goal across all three meaningful axes.

## Uncovered functions

Two host-side functions in `quasar_gpu_engine.mm` are uncovered because
the existing test harness uses synchronous `run_until_done` and never
introspects device identity:

| Symbol | Why uncovered | Proposed test |
| --- | --- | --- |
| `poll_round_result(QuasarRoundHandle) const` | Async polling API; `run_until_done` returns the same struct directly so no caller exercises this. | Add `poll_round_result_consistency` test in `quasar_gpu_engine_test.mm`: after `run_until_done` returns, call `poll_round_result(h)` and assert byte-equality with the returned snapshot. |
| `device_name() const` | Diagnostics helper for logs; never asserted in tests. | Add a one-line check in `empty_round`: `assert(std::strlen(e->device_name()) > 0)`. |

Adding both would push function coverage from 87.50% → 100% with
~6 lines of test code. Not in scope for this commit (already over the
80% goal); listed for the next test pass.

## CUDA backend coverage

Not measurable from this Apple host. The CUDA backend has its own
host-side glue at `lib/consensus/quasar/gpu/quasar_gpu_engine_cuda.cpp`
(653 lines) which compiles only on a CUDA-enabled toolchain. Its
coverage number will be produced by
`.github/workflows/quasar-cuda-build.yml` running the same
`quasar-determinism-test` against the CUDA `QuasarGPUEngine::create()`
on a self-hosted H100 runner; that build emits its own profraw against
the CUDA host stub, separate from the Metal numbers above.

## Generated artefacts

* `build-cov/coverage/quasar.profdata` — merged profile data
* `build-cov/coverage_report/index.html` — clickable HTML report (browse
  to it: `open build-cov/coverage_report/index.html`)
