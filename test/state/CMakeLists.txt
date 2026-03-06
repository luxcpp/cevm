# cevm: Fast Ethereum Virtual Machine implementation
# Copyright 2022 The cevm Authors.
# SPDX-License-Identifier: Apache-2.0

include(FetchContent)

add_library(cevm-state STATIC)
add_library(cevm::state ALIAS cevm-state)
target_compile_features(cevm-state PUBLIC cxx_std_20)
target_link_libraries(cevm-state PUBLIC cevm::precompiles evmc::evmc_cpp PRIVATE cevm)
target_include_directories(cevm-state PUBLIC ${PROJECT_SOURCE_DIR})
target_sources(
    cevm-state PRIVATE
    account.hpp
    blob_params.hpp
    block.hpp
    block.cpp
    bloom_filter.hpp
    bloom_filter.cpp
    errors.hpp
    ethash_difficulty.hpp
    ethash_difficulty.cpp
    hash_utils.hpp
    host.hpp
    host.cpp
    precompiles.hpp
    precompiles.cpp
    precompiles_internal.hpp
    requests.hpp
    requests.cpp
    state.hpp
    state.cpp
    state_diff.hpp
    state_view.hpp
    system_contracts.hpp
    system_contracts.cpp
    transaction.hpp
)

option(CEVM_PRECOMPILES_LIBSECP256K1 "Enable precompiles implementations using libsecp256k1 library" OFF)
if(CEVM_PRECOMPILES_LIBSECP256K1)
    FetchContent_Declare(
        libsecp256k1
        URL https://github.com/bitcoin-core/secp256k1/archive/refs/tags/v0.7.1.tar.gz
        URL_HASH SHA256=958f204dbafc117e73a2604285dc2eb2a5128344d3499c114dcba5de54cb7a9e
    )
    # TODO(cmake-3.25): Use block().
    set(BUILD_SHARED_LIBS_ORIG ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)
    set(SECP256K1_ENABLE_MODULE_ECDH OFF)
    set(SECP256K1_ENABLE_MODULE_RECOVERY ON)
    set(SECP256K1_ENABLE_MODULE_EXTRAKEYS OFF)
    set(SECP256K1_ENABLE_MODULE_SCHNORRSIG OFF)
    set(SECP256K1_ENABLE_MODULE_MUSIG OFF)
    set(SECP256K1_ENABLE_MODULE_ELLSWIFT OFF)
    # On x86_64 using asm is ~1% worse. Untested on ARM.
    set(SECP256K1_ASM OFF)
    set(SECP256K1_BUILD_BENCHMARK OFF)
    set(SECP256K1_BUILD_TESTS OFF)
    set(SECP256K1_BUILD_EXHAUSTIVE_TESTS OFF)
    set(SECP256K1_BUILD_CTIME_TESTS OFF)
    set(SECP256K1_BUILD_EXAMPLES OFF)
    # Not needed for static library.
    set(SECP256K1_ENABLE_API_VISIBILITY_ATTRIBUTES OFF)
    set(SECP256K1_VALGRIND OFF)
    FetchContent_MakeAvailable(libsecp256k1)
    foreach(TARGET secp256k1 secp256k1_precomputed)
        set_target_properties(
            ${TARGET}
            PROPERTIES
            COMPILE_OPTIONS $<$<C_COMPILER_ID:GNU,Clang,AppleClang>:-w>  # Disable warnings.
            CXX_CLANG_TIDY ""
        )
    endforeach()
    add_library(libsecp256k1::secp256k1 ALIAS secp256k1)

    target_link_libraries(cevm-state PRIVATE libsecp256k1::secp256k1)
    target_compile_definitions(cevm-state PUBLIC CEVM_PRECOMPILES_LIBSECP256K1=1)
    target_sources(
        cevm-state PRIVATE
        precompiles_libsecp256k1.hpp
        precompiles_libsecp256k1.cpp
    )
    set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS_ORIG})
endif()

option(CEVM_PRECOMPILES_GMP "Enable precompiles implementations via the GMP/MPIR library" OFF)
if(CEVM_PRECOMPILES_GMP)
    find_package(GMP REQUIRED)
    target_link_libraries(cevm-state PRIVATE GMP::gmp)
    target_compile_definitions(cevm-state PUBLIC CEVM_PRECOMPILES_GMP=1)
    target_sources(
        cevm-state PRIVATE
        precompiles_gmp.hpp
        precompiles_gmp.cpp
    )
endif()
