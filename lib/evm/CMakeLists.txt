# evm: Fast Ethereum Virtual Machine implementation
# Copyright 2019 The cevm Authors.
# SPDX-License-Identifier: Apache-2.0

include(LibraryTools)

# -----------------------------------------------------------------------------
# Targeted coverage instrumentation.
#
# When LUXCPP_EVM_COVERAGE=ON, a curated set of targets (evm, evm-gpu,
# evm-kernel-metal, evm-cuda) is built with LLVM source-based coverage
# (-fprofile-instr-generate -fcoverage-mapping). The custom `coverage`
# target below builds these, runs the tests, merges the .profraw files, and
# emits per-target HTML reports under ${CMAKE_BINARY_DIR}/coverage_report/.
#
# Why a separate option from -DCOVERAGE? -DCOVERAGE instruments *everything*
# the project compiles, including hunter-installed dependencies and CMake
# helpers — that produces noisy reports. LUXCPP_EVM_COVERAGE focuses on the
# dispatch + kernel surface that the parity test exercises, so the % numbers
# represent how much of the four-backend code actually runs.
function(_luxcpp_evm_apply_coverage tgt)
    if(NOT LUXCPP_EVM_COVERAGE)
        return()
    endif()
    if(NOT (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
            CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang"))
        message(FATAL_ERROR "LUXCPP_EVM_COVERAGE requires Clang/AppleClang "
                            "(found ${CMAKE_CXX_COMPILER_ID})")
    endif()
    if(NOT TARGET ${tgt})
        return()
    endif()
    target_compile_options(${tgt} PRIVATE -fprofile-instr-generate -fcoverage-mapping)
    # Propagate the linker flag publicly so executables that link an
    # instrumented static library pick up the LLVM profile runtime.
    target_link_options(${tgt} PUBLIC -fprofile-instr-generate -fcoverage-mapping)
endfunction()

add_library(evm
    ${include_dir}/cevm/cevm.h
    advanced_analysis.cpp
    advanced_analysis.hpp
    advanced_execution.cpp
    advanced_execution.hpp
    advanced_instructions.cpp
    baseline.hpp
    baseline_analysis.cpp
    baseline_execution.cpp
    baseline_instruction_table.cpp
    baseline_instruction_table.hpp
    constants.hpp
    delegation.cpp
    delegation.hpp
    instructions.hpp
    instructions_calls.cpp
    instructions_opcodes.hpp
    instructions_storage.cpp
    instructions_traits.hpp
    instructions_xmacro.hpp
    lru_cache.hpp
    tracing.cpp
    tracing.hpp
    vm.cpp
    vm.hpp
)
target_compile_features(evm PUBLIC cxx_std_20)
target_link_libraries(evm PUBLIC evmc::evmc intx::intx PRIVATE cevm::precompiles)
target_include_directories(evm PUBLIC
    $<BUILD_INTERFACE:${include_dir}>$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(CEVM_X86_64_ARCH_LEVEL GREATER_EQUAL 2)
    # Add CPU architecture runtime check. The CEVM_X86_64_ARCH_LEVEL has a valid value.
    target_sources(evm PRIVATE cpu_check.cpp)
    set_source_files_properties(cpu_check.cpp PROPERTIES COMPILE_DEFINITIONS CEVM_X86_64_ARCH_LEVEL=${CEVM_X86_64_ARCH_LEVEL})
endif()

if(CABLE_COMPILER_GNULIKE)
    target_compile_options(
        evm PRIVATE
        -fno-exceptions
        $<$<CXX_COMPILER_ID:GNU>:-Wstack-usage=2900>
    )
    if(NOT SANITIZE MATCHES undefined)
        # RTTI can be disabled except for UBSan which checks vptr integrity.
        target_compile_options(evm PRIVATE -fno-rtti)
    endif()
endif()
set_target_properties(
    evm
    PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_SOVERSION}
)

if(NOT SANITIZE)
    # On Linux, check if all symbols in evm are resolved during linking.
    target_link_options(evm PRIVATE $<$<PLATFORM_ID:Linux>:LINKER:--no-undefined>)
endif()

set_source_files_properties(vm.cpp PROPERTIES COMPILE_DEFINITIONS PROJECT_VERSION="${PROJECT_VERSION}")

add_standalone_library(evm)
_luxcpp_evm_apply_coverage(evm)

# LP-090 v0.50: ConflictSpec ABI — STM conflict declaration scaffold.
# Five sources (AccessList > ABI > Historical > Precompile > Learned > Declared)
# composed in fixed priority order. Consumed by Prism placement and
# QuasarSTM validation; below threshold, validation falls back to today's
# dynamic-discovery Block-STM (byte-equal to v0.47.x).
add_library(evm-stm STATIC
    stm/conflict_spec.cpp
    stm/conflict_spec.hpp
)
target_compile_features(evm-stm PUBLIC cxx_std_20)
target_include_directories(evm-stm PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${include_dir}>
)
_luxcpp_evm_apply_coverage(evm-stm)

# luxcpp/gpu: unified GPU acceleration library
set(LUXCPP_GPU_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../gpu" CACHE PATH "Path to luxcpp/gpu")
if(EXISTS "${LUXCPP_GPU_DIR}/CMakeLists.txt")
    set(LUX_GPU_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LUX_GPU_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    add_subdirectory("${LUXCPP_GPU_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/luxcpp-gpu" EXCLUDE_FROM_ALL)
    set(LUXCPP_GPU_FOUND TRUE)
else()
    message(STATUS "luxcpp/gpu not found at ${LUXCPP_GPU_DIR}, GPU state hashing disabled")
    set(LUXCPP_GPU_FOUND FALSE)
endif()

# GPU / parallel execution sources (compiled into a static lib for the benchmark)
add_library(evm-gpu STATIC
    gpu/gpu_dispatch.cpp
    gpu/gpu_dispatch.hpp
    gpu/gpu_state_hasher.hpp
    gpu/go_bridge.cpp
    gpu/go_bridge.h
    gpu/mv_memory.cpp
    gpu/mv_memory.hpp
    gpu/parallel_engine.cpp
    gpu/parallel_engine.hpp
    gpu/parallel_host.hpp
    gpu/scheduler.cpp
    gpu/scheduler.hpp
)
target_compile_features(evm-gpu PUBLIC cxx_std_20)
target_link_libraries(evm-gpu PUBLIC evm evmc::evmc intx::intx)
if(LUXCPP_GPU_FOUND)
    target_link_libraries(evm-gpu PUBLIC luxgpu_core_static)
endif()
target_include_directories(evm-gpu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${include_dir}>
)
_luxcpp_evm_apply_coverage(evm-gpu)

# CUDA backend (optional). Adds the .cu kernels and host launchers, and
# defines EVM_CUDA on evm-gpu so gpu_dispatch.cpp routes through the
# direct CUDA path.
if(CEVM_CUDA)
    add_library(evm-cuda STATIC
        gpu/cuda/keccak256.cu
        gpu/cuda/keccak_host.cpp
        gpu/cuda/keccak_host.hpp
        gpu/cuda/tx_validate.cu
        gpu/cuda/tx_validate_host.cpp
        gpu/cuda/tx_validate_host.hpp
        gpu/cuda/state_table.cu
        gpu/cuda/state_table_host.cpp
        gpu/cuda/state_table_host.hpp
        gpu/cuda/block_stm.cu
        gpu/cuda/block_stm_host.cpp
        gpu/cuda/block_stm_host.hpp
        gpu/cuda/evm_kernel.cu
        gpu/cuda/evm_kernel_host.cpp
        gpu/cuda/evm_kernel_host.hpp
        # QuasarGPU CUDA backend (mirror of lib/consensus/quasar/gpu/quasar_wave.metal +
        # quasar_gpu_engine.mm). Same QuasarGPUEngine API, persistent CTAs.
        ${CMAKE_CURRENT_SOURCE_DIR}/../consensus/quasar/gpu/quasar_wave.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/../consensus/quasar/gpu/quasar_gpu_engine_cuda.cpp
    )
    # On non-Apple CUDA-only builds, evm-kernel-metal is not built, so the
    # runtime dispatcher needs to live in evm-cuda. On Apple+CUDA dual
    # builds the dispatcher already lives in evm-kernel-metal — don't
    # double-include.
    if(NOT APPLE)
        target_sources(evm-cuda PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../consensus/quasar/gpu/quasar_gpu_runtime.cpp
        )
    endif()
    target_compile_features(evm-cuda PUBLIC cxx_std_20)
    set_target_properties(evm-cuda PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
    target_include_directories(evm-cuda PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>   # lib/ for consensus/quasar/gpu/...
        $<BUILD_INTERFACE:${include_dir}>
    )
    target_link_libraries(evm-cuda PUBLIC
        evm evmc::evmc intx::intx
        CUDA::cudart
    )

    # Wire EVM_CUDA into evm-gpu and the QuasarGPU runtime dispatcher.
    target_compile_definitions(evm-gpu PUBLIC EVM_CUDA)
    target_compile_definitions(evm-cuda PUBLIC EVM_CUDA)
    target_link_libraries(evm-gpu PUBLIC evm-cuda)
    _luxcpp_evm_apply_coverage(evm-cuda)
endif()

# Block execution benchmark
add_executable(evm-bench-block gpu/bench_block.cpp)
target_compile_features(evm-bench-block PRIVATE cxx_std_20)
target_link_libraries(evm-bench-block PRIVATE evm-gpu evm evmc::evmc)

# GPU mode consistency test
if(LUXCPP_GPU_FOUND)
    add_executable(evm-test-modes gpu/test_modes.cpp)
    target_compile_features(evm-test-modes PRIVATE cxx_std_20)
    target_link_libraries(evm-test-modes PRIVATE evm-gpu evm evmc::evmc)
    if(APPLE)
        target_link_libraries(evm-test-modes PRIVATE evm-metal-hosts evm-kernel-metal)
    endif()
endif()

# evm-bench-block also needs the Metal hosts to satisfy evm-gpu's references.
if(APPLE)
    target_link_libraries(evm-bench-block PRIVATE evm-metal-hosts evm-kernel-metal)
endif()

# GPU precompile dispatcher (lib/evm/gpu/precompiles/).
add_library(evm-precompiles STATIC
    gpu/precompiles/precompile_dispatch.cpp
    gpu/precompiles/precompile_dispatch.hpp
    gpu/precompiles/internal.hpp
    gpu/precompiles/ecrecover_cpu.cpp
    gpu/precompiles/sha256_cpu.cpp
    gpu/precompiles/ripemd160_cpu.cpp
    gpu/precompiles/identity.cpp
    gpu/precompiles/modexp_cpu.cpp
    gpu/precompiles/blake2f_cpu.cpp
    gpu/precompiles/bn256_cpu.cpp
    gpu/precompiles/point_eval_cpu.cpp
    gpu/precompiles/bls12_381_cpu.cpp
    gpu/precompiles/dex_match_cpu.cpp
)
target_compile_features(evm-precompiles PUBLIC cxx_std_20)
target_link_libraries(evm-precompiles PUBLIC cevm::precompiles intx::intx)
target_include_directories(evm-precompiles PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${include_dir}>
)
if(LUXCPP_GPU_FOUND)
    target_link_libraries(evm-precompiles PUBLIC luxgpu_core_static)
endif()

# lux-accel: optional GPU DEX/crypto kernels via the C ABI in
# lux/accel/c_api.h. When present, the dex_match_metal wrapper installs a
# Metal-backed override for precompile 0x100. When absent, the precompile
# stays on its CPU reference.
find_package(lux-accel CONFIG QUIET
    PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../../../install/lib/cmake/lux-accel"
    NO_DEFAULT_PATH)
if(lux-accel_FOUND)
    set(LUXCPP_ACCEL_FOUND TRUE)
    message(STATUS "lux-accel: found via cmake config")
else()
    # Manual fallback: probe the install tree directly.
    set(_lux_accel_install "${CMAKE_CURRENT_SOURCE_DIR}/../../../install")
    if(EXISTS "${_lux_accel_install}/include/lux/accel/c_api.h")
        add_library(lux::luxaccel SHARED IMPORTED)
        if(APPLE)
            set_target_properties(lux::luxaccel PROPERTIES
                IMPORTED_LOCATION "${_lux_accel_install}/lib/libluxaccel.dylib"
                INTERFACE_INCLUDE_DIRECTORIES "${_lux_accel_install}/include")
        else()
            set_target_properties(lux::luxaccel PROPERTIES
                IMPORTED_LOCATION "${_lux_accel_install}/lib/libluxaccel.so"
                INTERFACE_INCLUDE_DIRECTORIES "${_lux_accel_install}/include")
        endif()
        set(LUXCPP_ACCEL_FOUND TRUE)
        message(STATUS "lux-accel: found via manual probe at ${_lux_accel_install}")
    else()
        set(LUXCPP_ACCEL_FOUND FALSE)
        message(STATUS "lux-accel: NOT found, DEX_MATCH (0x100) stays on CPU")
    endif()
endif()

if(APPLE AND LUXCPP_GPU_FOUND)
    target_sources(evm-precompiles PRIVATE
        gpu/precompiles/ecrecover_metal.mm
        gpu/precompiles/bls12_381_metal.mm
        gpu/precompiles/point_eval_metal.mm
    )
    if(LUXCPP_ACCEL_FOUND)
        target_sources(evm-precompiles PRIVATE
            gpu/precompiles/dex_match_metal.mm
        )
        target_link_libraries(evm-precompiles PUBLIC lux::luxaccel)
    endif()
    target_link_libraries(evm-precompiles PUBLIC "-framework Metal" "-framework Foundation")
endif()
if(CEVM_CUDA AND LUXCPP_GPU_FOUND)
    target_sources(evm-precompiles PRIVATE
        gpu/precompiles/ecrecover_cuda.cpp
        gpu/precompiles/bls12_381_cuda.cpp
    )
    target_compile_definitions(evm-precompiles PUBLIC EVM_CUDA)
endif()

find_package(GTest CONFIG QUIET)
if(GTest_FOUND)
    add_executable(evm-precompile-test
        ${CMAKE_CURRENT_SOURCE_DIR}/../../test/unittests/precompile_test.cpp
    )
    target_compile_features(evm-precompile-test PRIVATE cxx_std_20)
    target_include_directories(evm-precompile-test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${include_dir}
    )
    target_link_libraries(evm-precompile-test PRIVATE
        evm-precompiles GTest::gtest GTest::gtest_main evmc::evmc
    )
endif()

# 4-way parity test: drives every parity vector through CPU_Sequential,
# CPU_Parallel, GPU_Metal, and (when EVM_CUDA is set) GPU_CUDA. Built only
# when GoogleTest is available (every other unit test in this repo gates
# on the same).
if(GTest_FOUND)
    add_executable(evm-parity-test
        ${CMAKE_CURRENT_SOURCE_DIR}/../../test/unittests/parity_test.cpp
    )
    target_compile_features(evm-parity-test PRIVATE cxx_std_20)
    target_include_directories(evm-parity-test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${include_dir}
    )
    target_link_libraries(evm-parity-test PRIVATE
        evm-gpu evm evmc::evmc GTest::gtest GTest::gtest_main
    )
    # The Metal kernel host is the actual implementation of
    # kernel::EvmKernelHost::create() on Apple. Without it the link
    # resolves to nullptr at runtime and the GPU_Metal backend falls
    # back to the CPU dispatcher (which would silently mask divergence).
    if(APPLE)
        target_link_libraries(evm-parity-test PRIVATE
            evm-metal-hosts evm-kernel-metal
        )
    endif()

    # Dispatcher unit tests
    add_executable(dispatch-test
        ${CMAKE_CURRENT_SOURCE_DIR}/../../test/unittests/dispatch_test.cpp
    )
    target_compile_features(dispatch-test PRIVATE cxx_std_20)
    target_include_directories(dispatch-test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${include_dir}
    )
    target_link_libraries(dispatch-test PRIVATE
        evm-gpu evm evmc::evmc GTest::gtest GTest::gtest_main
    )
    if(APPLE)
        target_link_libraries(dispatch-test PRIVATE evm-metal-hosts evm-kernel-metal)
    endif()

    # EIP-2200 signed refund + EIP-3529 cap test: covers the kernel CPU
    # interpreter's refund tracking and the dispatcher's cap application.
    # The kernel CPU path is the parity reference for Metal/CUDA, so this
    # test indirectly exercises the GPU kernels as well.
    add_executable(refund-test
        ${CMAKE_CURRENT_SOURCE_DIR}/../../test/unittests/refund_test.cpp
    )
    target_compile_features(refund-test PRIVATE cxx_std_20)
    target_include_directories(refund-test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${include_dir}
    )
    target_link_libraries(refund-test PRIVATE
        evm-gpu evm evmc::evmc GTest::gtest GTest::gtest_main
    )
    if(APPLE)
        target_link_libraries(refund-test PRIVATE evm-metal-hosts evm-kernel-metal)
    endif()

    # Stack-depth parity test: bytecode with stack depth >32, proves
    # Metal/CUDA match CPU cevm now that the spec's 1024-deep stack is wired.
    add_executable(stack-depth-test
        ${CMAKE_CURRENT_SOURCE_DIR}/../../test/unittests/stack_depth_test.cpp
    )
    target_compile_features(stack-depth-test PRIVATE cxx_std_20)
    target_include_directories(stack-depth-test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${include_dir}
    )
    target_link_libraries(stack-depth-test PRIVATE
        evm-gpu evm evmc::evmc GTest::gtest GTest::gtest_main
    )
    if(APPLE)
        target_link_libraries(stack-depth-test PRIVATE evm-metal-hosts evm-kernel-metal)
    endif()

    # Storage-overflow detection: 65th SSTORE/TSTORE signals exceptional halt
    # instead of silently dropping the write.
    add_executable(storage-overflow-test
        ${CMAKE_CURRENT_SOURCE_DIR}/../../test/unittests/storage_overflow_test.cpp
    )
    target_compile_features(storage-overflow-test PRIVATE cxx_std_20)
    target_include_directories(storage-overflow-test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${include_dir}
    )
    target_link_libraries(storage-overflow-test PRIVATE
        evm-gpu evm evmc::evmc GTest::gtest GTest::gtest_main
    )
    if(APPLE)
        target_link_libraries(storage-overflow-test PRIVATE evm-metal-hosts evm-kernel-metal)
    endif()

    # Kernel edge-cases: uint32 wrap, 2048-byte RETURN, BLOBHASH OOB, etc.
    add_executable(kernel-edge-cases-test
        ${CMAKE_CURRENT_SOURCE_DIR}/../../test/unittests/kernel_edge_cases_test.cpp
    )
    target_compile_features(kernel-edge-cases-test PRIVATE cxx_std_20)
    target_include_directories(kernel-edge-cases-test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${include_dir}
    )
    target_link_libraries(kernel-edge-cases-test PRIVATE
        evm-gpu evm evmc::evmc GTest::gtest GTest::gtest_main
    )
    if(APPLE)
        target_link_libraries(kernel-edge-cases-test PRIVATE evm-metal-hosts evm-kernel-metal)
    endif()

    # EIP-2929 access-list test: cold/warm gas accounting for SLOAD,
    # SSTORE, BALANCE, EXTCODE family on the kernel CPU + GPU paths.
    # Verifies that caller-supplied `Config::warm_addresses` and
    # `Config::warm_storage_keys` reach the kernel and pre-warm the
    # right keys, and that CPU + Metal agree on gas.
    add_executable(access-list-test
        ${CMAKE_CURRENT_SOURCE_DIR}/../../test/unittests/access_list_test.cpp
    )
    target_compile_features(access-list-test PRIVATE cxx_std_20)
    target_include_directories(access-list-test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${include_dir}
    )
    target_link_libraries(access-list-test PRIVATE
        evm-gpu evm evmc::evmc GTest::gtest GTest::gtest_main
    )
    if(APPLE)
        target_link_libraries(access-list-test PRIVATE evm-metal-hosts evm-kernel-metal)
    endif()

    # Per-opcode spec test for the kernel CPU interpreter. Each new opcode in
    # evm_interpreter.hpp gets a vector here; passing means status, gas, and
    # output match the EIP-cited cost (cross-checked against cevm CANCUN).
    add_executable(cpu-interpreter-spec-test
        ${CMAKE_CURRENT_SOURCE_DIR}/../../test/unittests/cpu_interpreter_spec_test.cpp
    )
    target_compile_features(cpu-interpreter-spec-test PRIVATE cxx_std_20)
    target_include_directories(cpu-interpreter-spec-test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${include_dir}
    )
    target_link_libraries(cpu-interpreter-spec-test PRIVATE
        evm-kernel evm evmc::evmc GTest::gtest GTest::gtest_main
    )
endif()

# LP-090 v0.50: STM ConflictSpec ABI regression suite. No GTest dependency
# (plain main + EXPECT macros), so always builds. Exercises the 5 sources +
# composer priority + reducer-lane semantics + repair_amplification bench.
add_executable(stm-conflict-spec-test
    ${CMAKE_CURRENT_SOURCE_DIR}/../../test/unittests/stm_conflict_spec_test.cpp
)
target_compile_features(stm-conflict-spec-test PRIVATE cxx_std_20)
target_include_directories(stm-conflict-spec-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR} ${include_dir}
)
target_link_libraries(stm-conflict-spec-test PRIVATE evm-stm)

# GPU EVM kernel interpreter (header-only library for CPU path)
add_library(evm-kernel INTERFACE)
target_include_directories(evm-kernel INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gpu/kernel>
)

# GPU EVM kernel benchmark (CPU path — always builds)
add_executable(evm-bench-kernel gpu/kernel/bench_kernel.cpp)
target_compile_features(evm-bench-kernel PRIVATE cxx_std_20)
target_link_libraries(evm-bench-kernel PRIVATE evm-kernel)

# Metal kernel host (macOS only)
if(APPLE)
    add_library(evm-kernel-metal STATIC
        gpu/kernel/evm_kernel_host.mm
        gpu/kernel/v3_persistent_host.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/../consensus/quasar/gpu/quasar_gpu_engine.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/../consensus/quasar/gpu/quasar_gpu_runtime.cpp
        # v0.46.1 — substrate-threshold gate dispatches small N to the CPU
        # reference (byte-equal to Metal). Linked into the engine library so
        # every consumer of QuasarGPUEngine gets the gate transparently.
        ${CMAKE_CURRENT_SOURCE_DIR}/../consensus/quasar/gpu/quasar_cpu_reference.cpp
    )
    target_include_directories(evm-kernel-metal PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>
    )
    target_compile_features(evm-kernel-metal PUBLIC cxx_std_20)
    target_link_libraries(evm-kernel-metal PUBLIC evm-kernel)
    target_link_libraries(evm-kernel-metal PUBLIC "-framework Metal" "-framework Foundation")
    target_link_libraries(evm-bench-kernel PRIVATE evm-kernel-metal)
    _luxcpp_evm_apply_coverage(evm-kernel-metal)

    # GPU-native state: persistent buffer pool + hash table
    add_library(evm-gpu-state STATIC
        gpu/persistent_pool.mm
        state/gpu_hashtable.mm
    )
    target_compile_features(evm-gpu-state PUBLIC cxx_std_20)
    target_link_libraries(evm-gpu-state PUBLIC evm evmc::evmc intx::intx cevm::precompiles)
    target_link_libraries(evm-gpu-state PUBLIC "-framework Metal" "-framework Foundation")
    target_include_directories(evm-gpu-state PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${include_dir}>
    )

    # GPU state tests
    add_executable(evm-test-gpu-state state/test_gpu_state.cpp)
    target_compile_features(evm-test-gpu-state PRIVATE cxx_std_20)
    target_link_libraries(evm-test-gpu-state PRIVATE evm-gpu-state evm evmc::evmc intx::intx cevm::precompiles)

    # Metal host dispatchers: tx_validate, bls, block_stm, keccak
    add_library(evm-metal-hosts STATIC
        gpu/metal/tx_validate_host.mm
        gpu/metal/bls_host.mm
        gpu/metal/block_stm_host.mm
        gpu/metal/keccak_host.mm
    )
    target_compile_features(evm-metal-hosts PUBLIC cxx_std_20)
    target_link_libraries(evm-metal-hosts PUBLIC evm-gpu "-framework Metal" "-framework Foundation")
    target_include_directories(evm-metal-hosts PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gpu>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${include_dir}>
    )

    # End-to-end GPU pipeline test
    add_executable(evm-test-pipeline gpu/test_pipeline.cpp)
    target_compile_features(evm-test-pipeline PRIVATE cxx_std_20)
    target_link_libraries(evm-test-pipeline PRIVATE
        evm-metal-hosts evm-kernel-metal evm-gpu-state evm-gpu evm-state
        evm evmc::evmc intx::intx cevm::precompiles
        "-framework Metal" "-framework Foundation"
    )
    target_include_directories(evm-test-pipeline PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gpu>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${include_dir}>
    )
    if(LUXCPP_GPU_FOUND)
        target_link_libraries(evm-test-pipeline PRIVATE luxgpu_core_static)
    endif()

    # Per-opcode coverage test for the GPU EVM kernel (Metal).
    add_executable(evm-test-opcodes gpu/test_opcodes.cpp)
    target_compile_features(evm-test-opcodes PRIVATE cxx_std_20)
    target_link_libraries(evm-test-opcodes PRIVATE
        evm-kernel-metal evm-kernel evm evmc::evmc intx::intx
        "-framework Metal" "-framework Foundation"
    )
    target_include_directories(evm-test-opcodes PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gpu>
        $<BUILD_INTERFACE:${include_dir}>
    )

    # Targeted host-layer tests for evm_kernel_host.mm (cache, concurrency,
    # boundary validation, async API). Mirrors the standalone test_opcodes
    # pattern — no GoogleTest dep so it builds on bare Apple toolchains.
    add_executable(metal-host-test
        ${CMAKE_CURRENT_SOURCE_DIR}/../../test/unittests/metal_host_test.mm)
    target_compile_features(metal-host-test PRIVATE cxx_std_20)
    target_link_libraries(metal-host-test PRIVATE
        evm-kernel-metal evm-kernel evm evmc::evmc intx::intx
        "-framework Metal" "-framework Foundation"
    )
    target_include_directories(metal-host-test PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gpu>
        $<BUILD_INTERFACE:${include_dir}>
    )

    # V3 wave-dispatch runner tests (drain, backpressure, shutdown,
    # counter monotonicity, empty-wave). Same bare-Apple pattern as
    # metal-host-test — no GoogleTest dep.
    add_executable(v3-persistent-test
        ${CMAKE_CURRENT_SOURCE_DIR}/../../test/unittests/v3_persistent_test.mm)
    target_compile_features(v3-persistent-test PRIVATE cxx_std_20)
    target_link_libraries(v3-persistent-test PRIVATE
        evm-kernel-metal evm-kernel evm evmc::evmc intx::intx
        "-framework Metal" "-framework Foundation"
    )
    target_include_directories(v3-persistent-test PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gpu>
        $<BUILD_INTERFACE:${include_dir}>
    )

    # QuasarGPU Engine tests (v0.31..v0.37): wave-tick scheduler,
    # device-resident rings, real keccak roots, async cold-state, Quasar
    # quorum aggregation. CEVM execution adapter for Quasar consensus
    # rounds (luxfi/consensus protocol/quasar).
    add_executable(quasar-gpu-engine-test
        ${CMAKE_CURRENT_SOURCE_DIR}/../../test/unittests/quasar_gpu_engine_test.mm)
    target_compile_features(quasar-gpu-engine-test PRIVATE cxx_std_20)
    target_link_libraries(quasar-gpu-engine-test PRIVATE
        evm-kernel-metal evm-kernel evm evmc::evmc intx::intx cevm::precompiles
        "-framework Metal" "-framework Foundation"
    )
    target_include_directories(quasar-gpu-engine-test PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gpu>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${include_dir}>
    )

    # Cross-backend determinism harness (Quasar 4.0 activation 2026-02-14).
    # Runs three deterministic workloads (1024-tx fast lane, 16-same-key
    # Block-STM, 100-tx Nebula DAG) twice on the same engine and once on
    # two separate engines; asserts every byte of QuasarRoundResult is
    # identical across runs. Same source file is reused by the CUDA
    # workflow in .github/workflows/quasar-cuda-build.yml — the only
    # difference at link time is whether QuasarGPUEngine::create() returns
    # the Metal or CUDA backend.
    add_executable(quasar-determinism-test
        ${CMAKE_CURRENT_SOURCE_DIR}/../../test/unittests/quasar_determinism_test.mm)
    target_compile_features(quasar-determinism-test PRIVATE cxx_std_20)
    target_link_libraries(quasar-determinism-test PRIVATE
        evm-kernel-metal evm-kernel evm evmc::evmc intx::intx cevm::precompiles
        "-framework Metal" "-framework Foundation"
    )
    target_include_directories(quasar-determinism-test PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gpu>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${include_dir}>
    )

    # STM-004 cross-backend determinism + v0.46.1 substrate gate regression.
    # Asserts Metal output is byte-equal to the CPU reference at small N
    # (gated path) and at force-Metal path. See quasar_stm_red_review_test.mm
    # — sections "stm-004", "v0.46.1 gate".
    add_executable(quasar-stm-red-review-test
        ${CMAKE_CURRENT_SOURCE_DIR}/../../test/unittests/quasar_stm_red_review_test.mm)
    target_compile_features(quasar-stm-red-review-test PRIVATE cxx_std_20)
    target_link_libraries(quasar-stm-red-review-test PRIVATE
        evm-kernel-metal evm-kernel evm evmc::evmc intx::intx cevm::precompiles
        "-framework Metal" "-framework Foundation"
    )
    target_include_directories(quasar-stm-red-review-test PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gpu>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${include_dir}>
    )

    # v0.44 — Quasar 9-chain integration. Validates the substrate wires
    # every LP-134 chain's transition root into QuasarRoundDescriptor and
    # the certificate_subject keccak in canonical P, C, X, Q, Z, A, B, M, F
    # order. Five new ServiceId entries (PlatformVMTransition ..
    # MPCVMTransition) reserve work-queue addresses for v0.45+ per-VM
    # ingest. Same Metal-only pattern as quasar-gpu-engine-test.
    add_executable(quasar-9chain-integration-test
        ${CMAKE_CURRENT_SOURCE_DIR}/../../test/unittests/quasar_9chain_integration_test.mm)
    target_compile_features(quasar-9chain-integration-test PRIVATE cxx_std_20)
    target_link_libraries(quasar-9chain-integration-test PRIVATE
        evm-kernel-metal evm-kernel evm evmc::evmc intx::intx cevm::precompiles
        "-framework Metal" "-framework Foundation"
    )
    target_include_directories(quasar-9chain-integration-test PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gpu>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${include_dir}>
    )

    # v0.42 cert ABI hardening — three cert lanes (BLSAggregate,
    # RingtailThreshold, MLDSAGroth16) × three cert modes (FastClassical,
    # HybridPQAsync, FullPQBlocking). Validates attestation_root + cert_mode
    # are bound into certificate_subject and cert lane scheduling honours
    # cert_mode. Same Metal-only pattern as quasar-9chain-integration-test.
    add_executable(quasar-cert-abi-test
        ${CMAKE_CURRENT_SOURCE_DIR}/../../test/unittests/quasar_cert_abi_test.mm)
    target_compile_features(quasar-cert-abi-test PRIVATE cxx_std_20)
    target_link_libraries(quasar-cert-abi-test PRIVATE
        evm-kernel-metal evm-kernel evm evmc::evmc intx::intx cevm::precompiles
        "-framework Metal" "-framework Foundation"
    )
    target_include_directories(quasar-cert-abi-test PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gpu>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${include_dir}>
    )

    # v0.44.1 — Quasar round wall-clock bench: Metal substrate vs CPU
    # reference on the three deterministic workloads, plus per-call
    # timings for the BLS / Groth16 / Ringtail host verifiers. Output is
    # plain stdout consumed by BENCHMARKS.md.
    add_executable(quasar-round-bench
        ${CMAKE_CURRENT_SOURCE_DIR}/../../test/unittests/quasar_round_bench.mm
        # quasar_cpu_reference.cpp comes via evm-kernel-metal (v0.46.1 — substrate gate).
        ${CMAKE_CURRENT_SOURCE_DIR}/../consensus/quasar/gpu/quasar_bls_verifier.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../consensus/quasar/gpu/quasar_groth16_verifier.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../consensus/quasar/gpu/quasar_ringtail_verifier.cpp
        # Stage 5 — on-device pairing C++ surface (host blst body today,
        # Stage 5b swaps to Metal pipeline; verdicts byte-equal by Stage 3
        # contract, 2746 vectors validated).
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../crypto/bls/cpp/bls_pairing.cpp)
    target_compile_features(quasar-round-bench PRIVATE cxx_std_20)
    target_link_libraries(quasar-round-bench PRIVATE
        evm-kernel-metal evm-kernel evm evmc::evmc intx::intx cevm::precompiles
        blst::oracle
        "-framework Metal" "-framework Foundation"
    )
    target_include_directories(quasar-round-bench PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gpu>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${include_dir}>
    )

    # v0.46 — Metal-vs-CPU substrate cutover sweep. Measures wall-clock at
    # each N and reports the smallest N where Metal beats CPU. Drives the
    # kQuasarSubstrateMetalThreshold constant in quasar_gpu_engine.hpp.
    add_executable(quasar-threshold-sweep
        ${CMAKE_CURRENT_SOURCE_DIR}/../../test/unittests/quasar_threshold_sweep.mm)
    # quasar_cpu_reference.cpp comes via evm-kernel-metal (v0.46.1 — substrate gate).
    target_compile_features(quasar-threshold-sweep PRIVATE cxx_std_20)
    target_link_libraries(quasar-threshold-sweep PRIVATE
        evm-kernel-metal evm-kernel evm evmc::evmc intx::intx cevm::precompiles
        "-framework Metal" "-framework Foundation"
    )
    target_include_directories(quasar-threshold-sweep PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gpu>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${include_dir}>
    )

    # v0.41 — real BLS12-381 aggregate verifier tests (Quasar 4.0
    # activation 2026-02-14). Replaces the v0.38 keccak-stub on the BLS
    # lane. Verifier is platform-independent (host blst), so the test is
    # a plain C++ executable — no Metal / CUDA dependency. Lives next to
    # the GPU substrate (lib/consensus/quasar/gpu/) so it ships on the
    # same release cadence.
    # v0.43 — add Groth16 + Ringtail verifiers in the same lane.
    add_executable(quasar-bls-verifier-test
        ${CMAKE_CURRENT_SOURCE_DIR}/../../test/unittests/quasar_bls_verifier_test.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../consensus/quasar/gpu/quasar_bls_verifier.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../consensus/quasar/gpu/quasar_groth16_verifier.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../consensus/quasar/gpu/quasar_ringtail_verifier.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../crypto/bls/cpp/bls_pairing.cpp)
    target_compile_features(quasar-bls-verifier-test PRIVATE cxx_std_20)
    target_link_libraries(quasar-bls-verifier-test PRIVATE blst::oracle cevm::precompiles)
    target_include_directories(quasar-bls-verifier-test PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${include_dir}>
    )

    # v0.43 — Precompile service ABI: PrecompileCall/PrecompileResult/
    # PrecompileArtifact + per-precompile-id batched queues + fiber
    # suspend/resume kernels. Per-precompile batched implementations land
    # in v0.44; this is the dispatcher + queues + state machine.
    add_library(precompile-service STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../consensus/quasar/gpu/precompile_service.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../consensus/quasar/gpu/precompile_service.mm)
    target_compile_features(precompile-service PUBLIC cxx_std_20)
    target_link_libraries(precompile-service
        PUBLIC evm-precompiles cevm::precompiles
        PUBLIC "-framework Metal" "-framework Foundation")
    target_include_directories(precompile-service PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gpu>
        $<BUILD_INTERFACE:${include_dir}>
    )

    add_executable(precompile-service-test
        ${CMAKE_CURRENT_SOURCE_DIR}/../../test/unittests/precompile_service_test.mm)
    target_compile_features(precompile-service-test PRIVATE cxx_std_20)
    target_link_libraries(precompile-service-test PRIVATE
        precompile-service evm-precompiles evm-metal-hosts
        evm-kernel-metal evm-kernel evm evmc::evmc intx::intx
        cevm::precompiles
        "-framework Metal" "-framework Foundation"
    )
    target_include_directories(precompile-service-test PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gpu>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${include_dir}>
    )

    # v0.44 — crypto precompile residency: ecrecover GPU-only batched
    # (Stage A inv via the v0.63 pipeline), BLS partial-GPU aggregate verify
    # (Miller-on-device + final_exp-host; Stage 5 wire-back lands when the
    # bls_miller driver is exposed publicly), Keccak residency dedup
    # (KeccakJobKind round-cache + in-batch table). PrecompileArtifact
    # roots flow into execution_root unchanged.
    add_executable(precompile-residency-test
        ${CMAKE_CURRENT_SOURCE_DIR}/../../test/unittests/precompile_residency_test.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/../consensus/quasar/gpu/quasar_bls_verifier.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../consensus/quasar/gpu/quasar_groth16_verifier.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../consensus/quasar/gpu/quasar_ringtail_verifier.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../crypto/bls/cpp/bls_pairing.cpp)
    target_compile_features(precompile-residency-test PRIVATE cxx_std_20)
    target_link_libraries(precompile-residency-test PRIVATE
        precompile-service evm-precompiles evm-metal-hosts
        evm-kernel-metal evm-kernel evm evmc::evmc intx::intx
        cevm::precompiles blst::oracle
        "-framework Metal" "-framework Foundation"
    )
    target_include_directories(precompile-residency-test PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gpu>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${include_dir}>
    )

    # v0.45 — precompiles bench: BLS / Groth16 / Ringtail batched-vs-unbatched.
    # Lives at the build root as `precompiles-bench` (matching the
    # acceptance script `./build-bench/precompiles-bench`). Self-contained,
    # no GoogleTest, no google-benchmark — same harness pattern as
    # quasar-round-bench.
    add_executable(precompiles-bench
        ${CMAKE_CURRENT_SOURCE_DIR}/../../test/unittests/precompiles_bench_v045.mm
        ${CMAKE_CURRENT_SOURCE_DIR}/../consensus/quasar/gpu/quasar_bls_verifier.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../consensus/quasar/gpu/quasar_groth16_verifier.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../consensus/quasar/gpu/quasar_ringtail_verifier.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../crypto/bls/cpp/bls_pairing.cpp)
    target_compile_features(precompiles-bench PRIVATE cxx_std_20)
    target_link_libraries(precompiles-bench PRIVATE
        blst::oracle cevm::precompiles
        "-framework Foundation")
    target_include_directories(precompiles-bench PRIVATE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
        $<BUILD_INTERFACE:${include_dir}>
    )
    set_target_properties(precompiles-bench PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

    # Mirror quasar-round-bench to the build root so the acceptance
    # script's `./build-bench/quasar-round-bench` resolves directly.
    if(TARGET quasar-round-bench)
        set_target_properties(quasar-round-bench PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    endif()

    # Stage 5 closure-proof: assert production cevm binaries do NOT link
    # blst.  Today this fails (cevm/cmake/blst.cmake is wired into the
    # production EIP-2537 G1/G2 precompile path).  Phase 5b turns it green
    # by routing those precompiles through the Stage 3 Metal pipeline.
    # Shipping the assertion now encodes the invariant.
    add_executable(no-blst-in-production-check
        ${CMAKE_CURRENT_SOURCE_DIR}/../../test/unittests/no_blst_in_production_test.cpp)
    target_compile_features(no-blst-in-production-check PRIVATE cxx_std_20)
    set_target_properties(no-blst-in-production-check PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    add_test(NAME no-blst-in-production-check
        COMMAND no-blst-in-production-check
                "${CMAKE_BINARY_DIR}"
                "${CMAKE_CURRENT_SOURCE_DIR}/../../test/unittests/no_blst_in_production_test.sh")
    set_tests_properties(no-blst-in-production-check PROPERTIES
        WILL_FAIL FALSE)
    # Phase 5b complete: WILL_FAIL flipped to FALSE.  cevm_precompiles is
    # blst-free in production -- bls.cpp/kzg.cpp call extern "C" symbols
    # that resolve to the canonical luxcpp/crypto adapter
    # (cevm_bls_kzg_canonical_cpu in lib/cevm_precompiles/CMakeLists.txt).
    # blst lives only in test-time oracle binaries via blst::oracle.

    # Register every Quasar / EVM kernel / parity test target with ctest so
    # `ctest --test-dir build-bench --output-on-failure` runs them. Each
    # test's command is the binary itself; they exit non-zero on failure
    # via the per-suite EXPECT macros.
    foreach(t IN ITEMS
        quasar-bls-verifier-test
        quasar-9chain-integration-test
        quasar-gpu-engine-test
        quasar-determinism-test
        quasar-stm-red-review-test
        quasar-cert-abi-test
        evm-test-opcodes
        evm-test-pipeline
        evm-test-modes
        evm-test-host-bridge
        evm-test-gpu-state
        evm-parity-test
        dispatch-test
        refund-test
        stack-depth-test
        storage-overflow-test
        kernel-edge-cases-test
        access-list-test
        cpu-interpreter-spec-test
        evm-precompile-test
        metal-host-test
        v3-persistent-test
        evm-unified-pipeline-test
        precompile-service-test
        precompile-residency-test
        stm-conflict-spec-test
    )
        if(TARGET ${t})
            add_test(NAME ${t} COMMAND ${t})
        endif()
    endforeach()
endif()

# State layer: full Ethereum state (accounts, storage, journal) for cevm
add_library(evm-state STATIC
    state/state_db.cpp
    state/processor.cpp
)
target_compile_features(evm-state PUBLIC cxx_std_20)
target_link_libraries(evm-state PUBLIC evm evmc::evmc intx::intx cevm::precompiles)
target_include_directories(evm-state PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${include_dir}>
)

# State layer: link luxgpu for GPU batch ecrecover when available
if(LUXCPP_GPU_FOUND)
    target_link_libraries(evm-state PUBLIC luxgpu_core_static)
    target_compile_definitions(evm-state PRIVATE HAVE_LUX_GPU=1)
endif()

# State layer benchmark
add_executable(evm-bench-state state/bench_state.cpp)
target_compile_features(evm-bench-state PRIVATE cxx_std_20)
target_link_libraries(evm-bench-state PRIVATE evm-state evm evmc::evmc)

# cevm: C++ EVM Lux VM plugin binary (ZAP protocol)
add_executable(cevm
    ${CMAKE_CURRENT_SOURCE_DIR}/../../bin/cevm/main.cpp
)
target_compile_features(cevm PRIVATE cxx_std_20)
target_link_libraries(cevm PRIVATE evm-state evm-gpu evm evmc::evmc intx::intx)
if(LUXCPP_GPU_FOUND)
    target_link_libraries(cevm PRIVATE luxgpu_core_static)
endif()
target_include_directories(cevm PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${include_dir}>
)

# Unified GPU pipeline test (consensus + EVM + FHE on same device)
if(APPLE AND LUXCPP_GPU_FOUND)
    add_executable(evm-test-unified gpu/test_unified.cpp)
    target_compile_features(evm-test-unified PRIVATE cxx_std_20)
    target_include_directories(evm-test-unified PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu
    )
    target_link_libraries(evm-test-unified PRIVATE
        evm evm-gpu luxgpu_core_static
        evm-kernel-metal
        "-framework Metal" "-framework Foundation"
    )
endif()

# Unified pipeline: EVM + consensus state hash + post-quantum sigs on one
# device, sharing a single LuxGPU handle. Built next to the legacy
# GpuPipeline (pipeline.hpp) for backwards compatibility.
if(APPLE AND LUXCPP_GPU_FOUND)
    add_library(evm-unified-pipeline STATIC
        gpu/unified_pipeline.cpp
        gpu/unified_pipeline.hpp
    )
    target_compile_features(evm-unified-pipeline PUBLIC cxx_std_20)
    target_include_directories(evm-unified-pipeline PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gpu>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${include_dir}>
    )
    target_link_libraries(evm-unified-pipeline PUBLIC
        evm-metal-hosts evm-gpu luxgpu_core_static
        evm evmc::evmc intx::intx
        "-framework Metal" "-framework Foundation"
    )

    add_executable(evm-unified-pipeline-test gpu/test_unified_pipeline.cpp)
    target_compile_features(evm-unified-pipeline-test PRIVATE cxx_std_20)
    target_compile_definitions(evm-unified-pipeline-test PRIVATE
        LUX_GPU_BACKEND_PATH_DEFAULT="${CMAKE_CURRENT_BINARY_DIR}/luxcpp-gpu"
    )
    target_link_libraries(evm-unified-pipeline-test PRIVATE
        evm-unified-pipeline
    )
elseif(LUXCPP_GPU_FOUND)
    # Non-Apple: header still usable, .cpp builds with CPU fallback paths only.
    add_library(evm-unified-pipeline STATIC
        gpu/unified_pipeline.cpp
        gpu/unified_pipeline.hpp
    )
    target_compile_features(evm-unified-pipeline PUBLIC cxx_std_20)
    target_include_directories(evm-unified-pipeline PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gpu>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${include_dir}>
    )
    target_link_libraries(evm-unified-pipeline PUBLIC
        evm-gpu luxgpu_core_static
        evm evmc::evmc intx::intx
    )
endif()

# -----------------------------------------------------------------------------
# EVMC Host bridge (feat/gpu-host-bridge).
#
# Turns evmc::Host* into a code dictionary that the GPU EVM kernel can run
# against. Pure leaf transactions and statically call-resolvable transactions
# go through the bridge; everything else falls back to cevm via the
# dispatcher.
# -----------------------------------------------------------------------------
add_library(evm-host-bridge STATIC
    gpu/host/evmc_bridge.cpp
    gpu/host/code_resolver.cpp
)
target_compile_features(evm-host-bridge PUBLIC cxx_std_20)
target_link_libraries(evm-host-bridge
    PUBLIC evmc::evmc evm-gpu evm
    PRIVATE cevm::precompiles)
target_include_directories(evm-host-bridge PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${include_dir}>
)
if(APPLE)
    target_sources(evm-host-bridge PRIVATE gpu/host/frame_metal.mm)
    target_link_libraries(evm-host-bridge PUBLIC evm-kernel-metal)
endif()
if(CEVM_CUDA)
    target_sources(evm-cuda PRIVATE gpu/host/frame_cuda.cu)
endif()

# Self-contained host bridge unit-test executable. Uses the GoogleTest already
# pulled in by Hunter so we don't need the full cevm test framework.
hunter_add_package(GTest)
find_package(GTest CONFIG QUIET)
if(GTest_FOUND)
    add_executable(evm-test-host-bridge
        ${PROJECT_SOURCE_DIR}/test/unittests/host_bridge_test.cpp
    )
    target_compile_features(evm-test-host-bridge PRIVATE cxx_std_20)
    target_link_libraries(evm-test-host-bridge PRIVATE
        evm-host-bridge evm-gpu evm evmc::evmc
        GTest::gtest GTest::gtest_main
    )
endif()

# -----------------------------------------------------------------------------
# `coverage` target.
#
# Drives the four-backend test executables, merges the resulting .profraw
# files via llvm-profdata, and emits per-target HTML reports under
# ${CMAKE_BINARY_DIR}/coverage_report/.
#
#   $ cmake .. -DLUXCPP_EVM_COVERAGE=ON
#   $ cmake --build . -j
#   $ make coverage
#
# Reports cover line, branch (region), and function metrics for the four
# instrumented targets: evm, evm-gpu, evm-kernel-metal, evm-cuda. The
# percentages reflect what fraction of each target's opcode dispatch / glue
# code the parity corpus + dedicated tests exercise.
# -----------------------------------------------------------------------------
if(LUXCPP_EVM_COVERAGE)
    # Apple ships the LLVM tools inside the Xcode toolchain, not on PATH.
    # Probe `xcrun -f` first, then fall back to PATH so Linux builds work.
    if(APPLE)
        execute_process(
            COMMAND xcrun -f llvm-profdata
            OUTPUT_VARIABLE LLVM_PROFDATA
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        execute_process(
            COMMAND xcrun -f llvm-cov
            OUTPUT_VARIABLE LLVM_COV
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
    endif()
    if(NOT LLVM_PROFDATA)
        find_program(LLVM_PROFDATA llvm-profdata)
    endif()
    if(NOT LLVM_COV)
        find_program(LLVM_COV llvm-cov)
    endif()
    if(NOT LLVM_PROFDATA OR NOT LLVM_COV)
        message(WARNING "LUXCPP_EVM_COVERAGE: llvm-profdata or llvm-cov not "
                        "found in PATH; skipping `coverage` target.")
    else()
        message(STATUS "LUXCPP_EVM_COVERAGE: profdata=${LLVM_PROFDATA} cov=${LLVM_COV}")
        # Test executables that drive the instrumented targets.
        set(_cov_drivers)
        foreach(t IN ITEMS
                evm-parity-test
                evm-test-opcodes
                evm-precompile-test
                evm-test-host-bridge
                evm-test-modes
                evm-test-pipeline
                evm-unified-pipeline-test
                evm-test-gpu-state)
            if(TARGET ${t})
                list(APPEND _cov_drivers ${t})
            endif()
        endforeach()

        set(_cov_dir "${CMAKE_BINARY_DIR}/coverage_report")
        set(_cov_raw "${_cov_dir}/profraw")
        set(_cov_data "${_cov_dir}/merged.profdata")

        # Per-target arguments for llvm-cov: each binary or .a is one
        # `-object` flag. We report on the .a/.dylib files because the test
        # executables themselves carry the instrumentation symbols of every
        # library they link.
        set(_cov_objects)
        foreach(t IN ITEMS evm evm-gpu evm-kernel-metal evm-cuda)
            if(TARGET ${t})
                list(APPEND _cov_objects "-object" "$<TARGET_FILE:${t}>")
            endif()
        endforeach()

        add_custom_target(coverage
            COMMENT "Run instrumented tests, merge profraw, emit HTML report"
            COMMAND ${CMAKE_COMMAND} -E rm -rf "${_cov_dir}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${_cov_raw}"
            # Run each driver with a unique LLVM_PROFILE_FILE so multiple
            # runs don't clobber each other.
            COMMAND ${CMAKE_COMMAND} -E env
                "LLVM_PROFILE_FILE=${_cov_raw}/parity-%p.profraw"
                $<TARGET_FILE:evm-parity-test>
            COMMAND ${CMAKE_COMMAND} -E env
                "LLVM_PROFILE_FILE=${_cov_raw}/opcodes-%p.profraw"
                $<TARGET_FILE:evm-test-opcodes>
            COMMAND ${CMAKE_COMMAND} -E env
                "LLVM_PROFILE_FILE=${_cov_raw}/precompile-%p.profraw"
                $<TARGET_FILE:evm-precompile-test>
            COMMAND ${CMAKE_COMMAND} -E env
                "LLVM_PROFILE_FILE=${_cov_raw}/host-bridge-%p.profraw"
                $<TARGET_FILE:evm-test-host-bridge>
            COMMAND ${CMAKE_COMMAND} -E env
                "LLVM_PROFILE_FILE=${_cov_raw}/modes-%p.profraw"
                $<TARGET_FILE:evm-test-modes>
            # Merge.
            COMMAND ${CMAKE_COMMAND} -E echo "[coverage] merging profraw..."
            COMMAND sh -c "${LLVM_PROFDATA} merge -sparse ${_cov_raw}/*.profraw -o ${_cov_data}"
            # Per-target HTML reports.
            COMMAND ${CMAKE_COMMAND} -E echo "[coverage] generating HTML reports..."
            COMMAND ${LLVM_COV} show
                ${_cov_objects}
                -instr-profile=${_cov_data}
                -format=html
                -output-dir=${_cov_dir}/html
                -ignore-filename-regex=.*\\.hunter.*
                -ignore-filename-regex=.*/_3rdParty/.*
                -ignore-filename-regex=.*/build.*
            COMMAND ${LLVM_COV} report
                ${_cov_objects}
                -instr-profile=${_cov_data}
                -ignore-filename-regex=.*\\.hunter.*
                -ignore-filename-regex=.*/_3rdParty/.*
                -ignore-filename-regex=.*/build.*
            COMMAND ${CMAKE_COMMAND} -E echo
                "[coverage] HTML at ${_cov_dir}/html/index.html"
            DEPENDS ${_cov_drivers}
            VERBATIM
        )
    endif()
endif()
