# Coverage for the four-backend EVM

Source-based code coverage for the targets that make up the parity surface:

| Target              | Role                                                       |
| ------------------- | ---------------------------------------------------------- |
| `evm`               | cevm interpreter (CPU baseline + Block-STM worker)       |
| `evm-gpu`           | dispatcher + Block-STM scheduler + state hasher            |
| `evm-kernel-metal`  | Apple Metal kernel host (compiles `evm_kernel.metal`)      |
| `evm-cuda`          | NVIDIA CUDA kernels + host launchers (when `CEVM_CUDA`)  |

Only these four targets carry coverage instrumentation. Hunter-installed
dependencies and CMake helpers are excluded so the headline numbers reflect
the real opcode-dispatch surface.

## Running

```sh
cd build
cmake .. -DLUXCPP_EVM_COVERAGE=ON
cmake --build . -j
make coverage
```

The `coverage` target:

1. Builds each test driver if the source has changed (`evm-parity-test`,
   `evm-test-opcodes`, `evm-precompile-test`, `evm-test-host-bridge`,
   `evm-test-modes`).
2. Runs each one with a per-driver `LLVM_PROFILE_FILE` under
   `build/coverage_report/profraw/`.
3. Merges the resulting `.profraw` files via `llvm-profdata merge -sparse`.
4. Emits an HTML report at `build/coverage_report/html/index.html` and
   prints a summary table to stdout.

## Reading the report

`llvm-cov report` prints three columns per target:

* **Lines** — fraction of source lines executed at least once.
* **Regions** (branches) — fraction of branch-taken / branch-not-taken pairs.
* **Functions** — fraction of functions entered at least once.

The HTML view shows the same data per file, with a heatmap of unexecuted
lines. Files filtered out via `-ignore-filename-regex=.*\\.hunter.*` etc.
do not appear, keeping the report focused on `lib/evm/`.

## What the targets cover

* `evm` — cevm's `baseline_execution.cpp`, `vm.cpp`, opcode
  implementations in `instructions_*.cpp`. The parity test drives the CPU
  bytecode path through `kernel::execute_cpu` (in `evm-gpu`), so `evm`'s
  coverage here comes from `precompile_test` and `host_bridge_test` (which
  invoke cevm via `evmc_create_cevm`).
* `evm-gpu` — `gpu_dispatch.cpp` (the public `execute_block` entry point
  and Backend routing), `parallel_engine.cpp` (Block-STM glue),
  `mv_memory.cpp` and `scheduler.cpp` (Block-STM internals), and the kernel
  CPU interpreter `evm_interpreter.hpp` (header-only; coverage shows up via
  every TU that includes it).
* `evm-kernel-metal` — `evm_kernel_host.mm` (Objective-C++ glue from C++
  `HostTransaction` to Metal's `MTLBuffer`s and dispatch). The .metal
  shader code itself is not LLVM-instrumented (Metal compiles to AIR, not
  IR with PGO support); coverage here measures only the host launcher.
* `evm-cuda` — host-side `*_host.cpp` files. The `.cu` device code is
  similarly outside LLVM's coverage scope; nvcc emits PTX, not coverage-
  mapped IR.

## What "good coverage" means here

The test corpus is intentionally per-opcode plus a few control-flow shapes.
For the parity targets the goal is:

* `evm-gpu` — **>90% line, >85% region** on `gpu_dispatch.cpp` and
  `parallel_engine.cpp`. The remaining slice is fallback paths that only
  fire when a backend factory returns nullptr.
* `evm-kernel-metal` — **>80% line** on `evm_kernel_host.mm`. The handful
  of unexecuted lines are MPS device-loss handlers and `execute_v2`'s
  warp-coalesced fast path, which the parity corpus only triggers on
  modern Apple Silicon.
* `evm-cuda` — **N/A on macOS** (excluded). On Linux/CUDA hosts the same
  >80% target applies to the host launchers.
* `evm` — secondary; the cevm unit suite provides its own >95% baseline
  (see `circle.yml`'s `cevm-coverage` job). The parity-driven number on
  this target is informational.

## When coverage drops

A regression here usually means one of:

1. A new opcode handler in `evm_interpreter.hpp` that the parity corpus
   doesn't cover yet — add a vector to `parity_test.cpp` for it.
2. A new dispatcher branch in `gpu_dispatch.cpp` (e.g. a GPU device-id
   filter) that no test triggers — add a config to the corpus.
3. A backend factory that started returning `nullptr` because of a runtime
   probe (no Metal device, no CUDA driver) — coverage drops because the
   GPU branches stop firing. Verify on an instrumented build that
   `available_backends()` reports the expected set.

For (1) and (2) the fix is to extend the corpus. For (3) it's an
environment problem, not a coverage problem.
