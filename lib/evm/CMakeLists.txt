# evm: Fast Ethereum Virtual Machine implementation
# Copyright 2019 The evmone Authors.
# SPDX-License-Identifier: Apache-2.0

include(LibraryTools)

add_library(evm
    ${include_dir}/evmone/evmone.h
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
target_link_libraries(evm PUBLIC evmc::evmc intx::intx PRIVATE evmone::precompiles)
target_include_directories(evm PUBLIC
    $<BUILD_INTERFACE:${include_dir}>$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(EVMONE_X86_64_ARCH_LEVEL GREATER_EQUAL 2)
    # Add CPU architecture runtime check. The EVMONE_X86_64_ARCH_LEVEL has a valid value.
    target_sources(evm PRIVATE cpu_check.cpp)
    set_source_files_properties(cpu_check.cpp PROPERTIES COMPILE_DEFINITIONS EVMONE_X86_64_ARCH_LEVEL=${EVMONE_X86_64_ARCH_LEVEL})
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

# CUDA backend (optional). Adds the .cu kernels and host launchers, and
# defines EVM_CUDA on evm-gpu so gpu_dispatch.cpp routes through the
# direct CUDA path.
if(EVMONE_CUDA)
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
    )
    target_compile_features(evm-cuda PUBLIC cxx_std_20)
    set_target_properties(evm-cuda PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
    target_include_directories(evm-cuda PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${include_dir}>
    )
    target_link_libraries(evm-cuda PUBLIC
        evm evmc::evmc intx::intx
        CUDA::cudart
    )

    # Wire EVM_CUDA into evm-gpu so the dispatcher routes through CUDA.
    target_compile_definitions(evm-gpu PUBLIC EVM_CUDA)
    target_link_libraries(evm-gpu PUBLIC evm-cuda)
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
)
target_compile_features(evm-precompiles PUBLIC cxx_std_20)
target_link_libraries(evm-precompiles PUBLIC evmone::precompiles intx::intx)
target_include_directories(evm-precompiles PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${include_dir}>
)
if(LUXCPP_GPU_FOUND)
    target_link_libraries(evm-precompiles PUBLIC luxgpu_core_static)
endif()
if(APPLE AND LUXCPP_GPU_FOUND)
    target_sources(evm-precompiles PRIVATE
        gpu/precompiles/ecrecover_metal.mm
        gpu/precompiles/bls12_381_metal.mm
        gpu/precompiles/point_eval_metal.mm
    )
    target_link_libraries(evm-precompiles PUBLIC "-framework Metal" "-framework Foundation")
endif()
if(EVMONE_CUDA AND LUXCPP_GPU_FOUND)
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
    add_library(evm-kernel-metal STATIC gpu/kernel/evm_kernel_host.mm)
    target_compile_features(evm-kernel-metal PUBLIC cxx_std_20)
    target_link_libraries(evm-kernel-metal PUBLIC evm-kernel)
    target_link_libraries(evm-kernel-metal PUBLIC "-framework Metal" "-framework Foundation")
    target_link_libraries(evm-bench-kernel PRIVATE evm-kernel-metal)

    # GPU-native state: persistent buffer pool + hash table
    add_library(evm-gpu-state STATIC
        gpu/persistent_pool.mm
        state/gpu_hashtable.mm
    )
    target_compile_features(evm-gpu-state PUBLIC cxx_std_20)
    target_link_libraries(evm-gpu-state PUBLIC evm evmc::evmc intx::intx evmone::precompiles)
    target_link_libraries(evm-gpu-state PUBLIC "-framework Metal" "-framework Foundation")
    target_include_directories(evm-gpu-state PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${include_dir}>
    )

    # GPU state tests
    add_executable(evm-test-gpu-state state/test_gpu_state.cpp)
    target_compile_features(evm-test-gpu-state PRIVATE cxx_std_20)
    target_link_libraries(evm-test-gpu-state PRIVATE evm-gpu-state evm evmc::evmc intx::intx evmone::precompiles)

    # Metal host dispatchers: tx_validate, bls, block_stm
    add_library(evm-metal-hosts STATIC
        gpu/metal/tx_validate_host.mm
        gpu/metal/bls_host.mm
        gpu/metal/block_stm_host.mm
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
        evm evmc::evmc intx::intx evmone::precompiles
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
endif()

# State layer: full Ethereum state (accounts, storage, journal) for evmone
add_library(evm-state STATIC
    state/state_db.cpp
    state/processor.cpp
)
target_compile_features(evm-state PUBLIC cxx_std_20)
target_link_libraries(evm-state PUBLIC evm evmc::evmc intx::intx evmone::precompiles)
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
