# cevm: Fast Ethereum Virtual Machine implementation
# Copyright 2023 The cevm Authors.
# SPDX-License-Identifier: Apache-2.0

# Phase 5b — production cevm_precompiles is blst-free.  bls.cpp/kzg.cpp
# were rewritten to call the extern "C" symbols `bls12_381_*` and
# `bls12_381_kzg_verify_proof`.  Those symbols come from the canonical
# luxcpp/crypto adapter library `cevm_bls_kzg_canonical_cpu` defined
# below, which links blst privately.  The static archive
# `libcevm_precompiles.a` no longer carries blst symbol references; the
# CI assertion (no_blst_in_production_test) passes.

# ----------------------------------------------------------------------------
# Canonical CPU adapter — wraps luxcpp/crypto/bls + luxcpp/crypto/kzg with
# extern "C" entry points.  Links blst privately (test-time oracle).
# ----------------------------------------------------------------------------

set(_LUX_CRYPTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../crypto)

# Pull in the test-time blst oracle that already lives at
# luxcpp/crypto/bls/test/cmake/blst.cmake.  cevm/cmake/blst.cmake was
# deleted in Phase 5b; this is the only blst dependency the cevm tree
# carries from now on, and it is consumed by the canonical adapter
# library only -- not by libcevm_precompiles.a.
include(${_LUX_CRYPTO_DIR}/bls/test/cmake/blst.cmake)

# Note: sha256.{hpp,cpp} are NOT compiled here; cevm_precompiles already
# carries the same first-party sha256 (cevm::crypto::sha256), and the
# kzg.cpp body's `sha256` reference resolves to that single definition
# at the final dylib/executable link.
add_library(cevm_bls_kzg_canonical_cpu STATIC
    ${_LUX_CRYPTO_DIR}/bls/c-abi/c_bls_pairing.cpp
    ${_LUX_CRYPTO_DIR}/bls/cpp/bls.cpp
    ${_LUX_CRYPTO_DIR}/bls/cpp/bls_pairing.cpp
    ${_LUX_CRYPTO_DIR}/kzg/c-abi/c_kzg.cpp
    ${_LUX_CRYPTO_DIR}/kzg/cpp/kzg.cpp
)
target_compile_features(cevm_bls_kzg_canonical_cpu PUBLIC cxx_std_20)
target_include_directories(cevm_bls_kzg_canonical_cpu
    PUBLIC
        $<BUILD_INTERFACE:${_LUX_CRYPTO_DIR}/bls/c-abi>
        $<BUILD_INTERFACE:${_LUX_CRYPTO_DIR}/bls/cpp>
        $<BUILD_INTERFACE:${_LUX_CRYPTO_DIR}/kzg/cpp>
        $<BUILD_INTERFACE:${_LUX_CRYPTO_DIR}/sha256/cpp>
        $<BUILD_INTERFACE:${_LUX_CRYPTO_DIR}/c-abi>
        $<BUILD_INTERFACE:${_LUX_CRYPTO_DIR}/include>
)
target_link_libraries(cevm_bls_kzg_canonical_cpu PRIVATE intx::intx blst::oracle)
add_library(cevm::bls_kzg_canonical_cpu ALIAS cevm_bls_kzg_canonical_cpu)

# ----------------------------------------------------------------------------
# Production cevm_precompiles -- first-party sources, NO blst.
# ----------------------------------------------------------------------------

add_library(cevm_precompiles STATIC)
add_library(cevm::precompiles ALIAS cevm_precompiles)
target_link_libraries(cevm_precompiles
    PUBLIC evmc::evmc_cpp cevm_bls_kzg_canonical_cpu
    PRIVATE cevm::evmmax)
target_include_directories(cevm_precompiles INTERFACE ..)
target_sources(
    cevm_precompiles PRIVATE
    blake2b.hpp
    blake2b.cpp
    # Phase 5b: bls.{hpp,cpp} declarations remain (callers include the
    # header), but the implementation is supplied by
    # cevm_bls_kzg_canonical_cpu (linked PUBLIC above).  No bls.cpp here.
    bls.hpp
    bn254.hpp
    bn254.cpp
    ecc.hpp
    hash_types.h
    keccak.h
    keccak.hpp
    keccak.c
    pairing/bn254/fields.hpp
    pairing/bn254/pairing.cpp
    pairing/bn254/utils.hpp
    pairing/field_template.hpp
    modexp.hpp
    modexp.cpp
    mulmod.hpp
    mulmod.cpp
    ripemd160.hpp
    ripemd160.cpp
    secp256k1.hpp
    secp256k1.cpp
    secp256r1.hpp
    secp256r1.cpp
    sha256.hpp
    sha256.cpp
    # Phase 5b: kzg.{hpp,cpp} same pattern -- header here, body in
    # cevm_bls_kzg_canonical_cpu.
    kzg.hpp
)
