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

# GPU / parallel execution sources (compiled into a static lib for the benchmark)
add_library(evm-gpu STATIC
    gpu/gpu_dispatch.cpp
    gpu/gpu_dispatch.hpp
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
target_include_directories(evm-gpu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${include_dir}>
)

# Block execution benchmark
add_executable(evm-bench-block gpu/bench_block.cpp)
target_compile_features(evm-bench-block PRIVATE cxx_std_20)
target_link_libraries(evm-bench-block PRIVATE evm-gpu evm evmc::evmc)
