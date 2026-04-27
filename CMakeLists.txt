# evm: Ethereum Virtual Machine
# Copyright 2019 The cevm Authors.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.19...4.2)

if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/evmc/.git)
    message(FATAL_ERROR "Git submodules not initialized, execute:\n  git submodule update --init")
endif()

option(BUILD_SHARED_LIBS "Build evm as a shared library" ON)
option(COVERAGE "Build with coverage instrumentation" OFF)
option(ASSERTIONS "Force enable the assertions in the code ignoring the CMAKE_BUILD_TYPE default" OFF)
option(CEVM_TESTING "Build tests and test tools" OFF)
option(CEVM_FUZZING "Instrument libraries and build fuzzing tools" OFF)
option(CEVM_CUDA "Enable CUDA backend (NVIDIA GPU acceleration)" OFF)

# v0.45 build flags. Metal is auto-enabled on Apple; this option is accepted
# (and asserted) so callers can pin it on/off without surprise. The two
# kernel/pairing flags switch on the V2 EVM kernel and the GPU-batched
# pairing path respectively. Off by default so v0.44 builds unchanged.
option(LUX_CEVM_ENABLE_METAL    "Enable Apple Metal backend (auto on APPLE)" ON)
option(LUX_EVM_KERNEL_V2        "Build the V2 32-threads/tx EVM kernel"       OFF)
option(LUX_QUASAR_GPU_PAIRING   "Enable batched BLS/Groth16/Ringtail pairing" OFF)
if(LUX_EVM_KERNEL_V2)
    add_compile_definitions(LUX_EVM_KERNEL_V2=1)
endif()
if(LUX_QUASAR_GPU_PAIRING)
    add_compile_definitions(LUX_QUASAR_GPU_PAIRING=1)
endif()
# luxcpp/evm targeted coverage. Distinct from -DCOVERAGE which instruments
# every TU in the project; LUXCPP_EVM_COVERAGE only instruments the targets
# that constitute the four-backend parity surface (evm, evm-gpu,
# evm-kernel-metal, evm-cuda) so the report is focused on the opcode
# dispatch paths rather than CMake helpers / Hunter packages.
option(LUXCPP_EVM_COVERAGE "Instrument the four-backend EVM targets with LLVM source-based coverage" OFF)

include(cmake/cable/bootstrap.cmake)
include(CableBuildType)
include(CableCompilerSettings)
include(CablePackage)
include(CableToolchains)
include(CMakePackageConfigHelpers)

cable_configure_toolchain(DEFAULT cxx17-pic)
cable_set_build_type(DEFAULT Release CONFIGURATION_TYPES Release Debug)

include(Hunter/init)

project(evm LANGUAGES CXX C OBJCXX)
set(PROJECT_VERSION 0.19.0)

# CUDA backend (optional). Requires CUDA Toolkit 11.8+.
# Targets compute capability 7.0+ (Turing, Ampere, Hopper, Ada).
if(CEVM_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit 11.8 REQUIRED)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # 70 = Volta/Turing, 80 = Ampere, 89 = Ada, 90 = Hopper.
        set(CMAKE_CUDA_ARCHITECTURES 70 80 89 90)
    endif()
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    message(STATUS "CUDA backend: ENABLED (CUDA ${CUDAToolkit_VERSION}, archs: ${CMAKE_CUDA_ARCHITECTURES})")
else()
    message(STATUS "CUDA backend: DISABLED (-DCEVM_CUDA=ON to enable)")
endif()

string(REGEX MATCH "([0-9]+)\\.([0-9]+)" _ ${PROJECT_VERSION})
set(PROJECT_VERSION_MAJOR ${CMAKE_MATCH_1})
set(PROJECT_VERSION_MINOR ${CMAKE_MATCH_2})

set(PROJECT_SOVERSION ${PROJECT_VERSION_MAJOR})
if(PROJECT_VERSION_MAJOR EQUAL 0)
    set(PROJECT_SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR})
endif()

option(EVMC_TOOLS "Build EVMC test tools" ${CEVM_TESTING})
option(EVMC_INSTALL "Install EVMC" OFF)

if(COMMAND block) # TODO(cmake-3.25)
    block()
        set(CMAKE_C_CLANG_TIDY "")
        set(CMAKE_CXX_CLANG_TIDY "")
        add_subdirectory(evmc)
    endblock()
else()
    add_subdirectory(evmc)
endif()

cable_configure_compiler(NO_STACK_PROTECTION)
if(CABLE_COMPILER_GNULIKE)
    add_compile_options(
        -Wmissing-declarations
        $<$<COMPILE_LANGUAGE:CXX>:-Wextra-semi>
        $<$<COMPILE_LANGUAGE:CXX>:-Wno-missing-field-initializers>

        $<$<CXX_COMPILER_ID:GNU>:-Wno-attributes>
        $<$<CXX_COMPILER_ID:GNU>:-Wduplicated-cond>
        $<$<CXX_COMPILER_ID:GNU>:-Wlogical-op>

        $<$<CXX_COMPILER_ID:Clang,AppleClang>:-Wno-unknown-attributes>
        $<$<CXX_COMPILER_ID:Clang,AppleClang>:-Wduplicate-enum>
        $<$<CXX_COMPILER_ID:Clang,AppleClang>:-Wnewline-eof>
        $<$<CXX_COMPILER_ID:Clang,AppleClang>:-Wunreachable-code-aggressive>
    )
elseif(MSVC)
    add_compile_options(/wd4324)  # Disabled warning about alignment caused by alignas.
    add_compile_options(/wd5030)  # Allow using unknown attributes.
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Coverage")
    message(FATAL_ERROR "Use -DCOVERAGE=1")
endif()

if(COVERAGE)
    add_compile_options(
        $<$<CXX_COMPILER_ID:GNU>:--coverage>

        # Enable Clang's Source Based Coverage.
        # https://clang.llvm.org/docs/SourceBasedCodeCoverage.html
        $<$<CXX_COMPILER_ID:Clang,AppleClang>:-fprofile-instr-generate>
        $<$<CXX_COMPILER_ID:Clang,AppleClang>:-fcoverage-mapping>
        $<$<CXX_COMPILER_ID:Clang,AppleClang>:-fcoverage-mcdc>
    )
    add_link_options(
        $<$<CXX_COMPILER_ID:GNU>:--coverage>
        $<$<CXX_COMPILER_ID:Clang,AppleClang>:-fprofile-instr-generate>
    )
endif()

if(ASSERTIONS)
    if(CMAKE_CONFIGURATION_TYPES)
        message(FATAL_ERROR "ASSERTIONS option is not supported with multi-configuration generators")
    endif()
    # Remove -DNDEBUG from default build type flags.
    string(TOUPPER "${CMAKE_BUILD_TYPE}" BUILD_TYPE_UPPER)
    get_property(ENABLED_LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)
    foreach(LANG IN LISTS ENABLED_LANGUAGES)
        string(REGEX REPLACE " *-DNDEBUG" "" CMAKE_${LANG}_FLAGS_${BUILD_TYPE_UPPER} "${CMAKE_${LANG}_FLAGS_${BUILD_TYPE_UPPER}}")
    endforeach()
endif()

set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL x86_64)
    # Setup options for x86_64 micro-architecture levels.
    # https://clang.llvm.org/docs/UsersManual.html#x86

    set(CEVM_X86_64_ARCH_LEVEL_INIT 2)
    if(APPLE)
        # On macos with Apple Silicon CPU (arm64) the x86 is emulated and SSE4.2 is not available.
        set(CEVM_X86_64_ARCH_LEVEL_INIT 1)
    endif()

    set(CEVM_X86_64_ARCH_LEVEL ${CEVM_X86_64_ARCH_LEVEL_INIT} CACHE STRING "The x86_64 micro-architecture level")
    if(CEVM_X86_64_ARCH_LEVEL GREATER_EQUAL 1 AND CEVM_X86_64_ARCH_LEVEL LESS_EQUAL 4)
        message(STATUS "x86_64 micro-architecture level: ${CEVM_X86_64_ARCH_LEVEL}")
        if(CEVM_X86_64_ARCH_LEVEL GREATER_EQUAL 2)
            add_compile_options(-march=x86-64-v${CEVM_X86_64_ARCH_LEVEL})
        endif()
    else()
        message(FATAL_ERROR "Invalid CEVM_X86_64_ARCH_LEVEL: ${CEVM_X86_64_ARCH_LEVEL}")
    endif()
endif()

include(GNUInstallDirs)

if(CEVM_FUZZING)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build evm as a shared library" FORCE)
    set(CEVM_TESTING ON CACHE BOOL "Build tests and test tools" FORCE)

    if(NOT COVERAGE)
        # Add fuzzing instrumentation only for non-coverage builds.
        # The coverage builds should be without fuzzing instrumentation to allow
        # running fuzzing corpus once and getting code coverage.
        set(fuzzing_flags -fsanitize=fuzzer-no-link,address,undefined,shift-exponent,implicit-conversion,nullability)
        add_compile_options(${fuzzing_flags})
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${fuzzing_flags}")
    endif()
endif()

set(CEVM_INTX_DIR "" CACHE PATH "Path to local intx source directory")
if(CEVM_INTX_DIR)
    add_subdirectory(${CEVM_INTX_DIR} intx)
else()
    hunter_add_package(intx)
    find_package(intx CONFIG REQUIRED)
endif()

set(include_dir ${CMAKE_CURRENT_SOURCE_DIR}/include)

enable_testing()
add_subdirectory(lib)

if(CEVM_TESTING)
    add_subdirectory(test)
endif()


# INSTALL

set(install_targets evm)
if(TARGET evm-standalone)
    list(APPEND install_targets evm-standalone)
endif()
if(TARGET evm-test)
    list(APPEND install_targets evm-test)
endif()
if(TARGET evm-bench-block)
    list(APPEND install_targets evm-bench-block)
endif()
if(TARGET evm-bench-state)
    list(APPEND install_targets evm-bench-state)
endif()
if(TARGET cevm)
    list(APPEND install_targets cevm)
endif()

set_target_properties(
    ${install_targets} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}
)

install(TARGETS ${install_targets} EXPORT evmTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY ${include_dir}/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

cable_add_archive_package()
