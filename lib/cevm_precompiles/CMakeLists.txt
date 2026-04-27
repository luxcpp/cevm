# cevm: Fast Ethereum Virtual Machine implementation
# Copyright 2023 The cevm Authors.
# SPDX-License-Identifier: Apache-2.0

# Phase 5c (cevm v0.47.0) — first-party algorithm bodies relocated to the
# canonical luxcpp/crypto/<alg>/cpp/ tree. cevm_precompiles/ now contains
# only:
#   * shim headers that forward to <_LUX_CRYPTO_DIR>/<alg>/cpp/<alg>.hpp
#   * keccak.{c,h,hpp} + secp256k1.{cpp,hpp} (legacy ethash::/evmmax::
#     consumer API; superseded by canonical lux/crypto/keccak.h and
#     lux/crypto/secp256k1.h but consumer migration is out of scope here)
#   * shared infra (ecc.hpp, hash_types.h, expmod.tmpl)
#   * bls.hpp + kzg.hpp declarations (bodies in cevm_bls_kzg_canonical_cpu)
# Net effect: zero duplicate first-party implementation source files in the
# cevm tree. Canonical sources are compiled directly from luxcpp/crypto.

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

# Note: sha256 body is supplied by the canonical luxcpp/crypto/sha256/cpp/sha256.cpp
# linked into cevm_precompiles below; the kzg.cpp body's `sha256` reference
# resolves to that single definition at the final dylib/executable link.
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
# Production cevm_precompiles -- canonical luxcpp/crypto sources, NO blst.
# ----------------------------------------------------------------------------

add_library(cevm_precompiles STATIC)
add_library(cevm::precompiles ALIAS cevm_precompiles)
target_link_libraries(cevm_precompiles
    PUBLIC evmc::evmc_cpp cevm_bls_kzg_canonical_cpu
    PRIVATE cevm::evmmax)
# INTERFACE includes:
#   * `..` keeps `<cevm_precompiles/<alg>.hpp>` reachable for legacy consumers.
#   * `_LUX_CRYPTO_DIR` lets the shim headers resolve `"<alg>/cpp/<alg>.hpp"`.
target_include_directories(cevm_precompiles
    INTERFACE
        ..
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${_LUX_CRYPTO_DIR})
# Canonical sources need the same include roots when compiled as part of
# cevm_precompiles (they reach into shared infra: ecc.hpp, hash_types.h,
# intx, evmc).
target_include_directories(cevm_precompiles
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${_LUX_CRYPTO_DIR}
        ${_LUX_CRYPTO_DIR}/include
        ${_LUX_CRYPTO_DIR}/c-abi)
target_sources(
    cevm_precompiles PRIVATE
    # ---- shim headers (forward to canonical crypto/<alg>/cpp/<alg>.hpp) ----
    blake2b.hpp
    bn254.hpp
    modexp.hpp
    mulmod.hpp
    ripemd160.hpp
    secp256r1.hpp
    sha256.hpp
    # ---- canonical first-party bodies ----
    ${_LUX_CRYPTO_DIR}/blake2b/cpp/blake2b.cpp
    ${_LUX_CRYPTO_DIR}/blake2b/cpp/blake2b.hpp
    ${_LUX_CRYPTO_DIR}/bn254/cpp/bn254.cpp
    ${_LUX_CRYPTO_DIR}/bn254/cpp/bn254.hpp
    # Note: bn254 pairing files have relative includes ("../../bn254.hpp",
    # "../field_template.hpp") that only resolve when the files live in a
    # directory layout matching cevm/lib/cevm_precompiles/pairing/bn254/.
    # The canonical luxcpp/crypto/bn254/cpp/pairing/ copy uses identical
    # source text but its layout breaks those relative includes. Until the
    # canonical headers are flattened (out of scope for Phase 5c), the
    # pairing impl stays compiled from this location.
    pairing/bn254/fields.hpp
    pairing/bn254/pairing.cpp
    pairing/bn254/utils.hpp
    pairing/field_template.hpp
    ${_LUX_CRYPTO_DIR}/modexp/cpp/modexp.cpp
    ${_LUX_CRYPTO_DIR}/modexp/cpp/modexp.hpp
    ${_LUX_CRYPTO_DIR}/modexp/cpp/mulmod.cpp
    ${_LUX_CRYPTO_DIR}/modexp/cpp/mulmod.hpp
    ${_LUX_CRYPTO_DIR}/ripemd160/cpp/ripemd160.cpp
    ${_LUX_CRYPTO_DIR}/ripemd160/cpp/ripemd160.hpp
    ${_LUX_CRYPTO_DIR}/secp256r1/cpp/secp256r1.cpp
    ${_LUX_CRYPTO_DIR}/secp256r1/cpp/secp256r1.hpp
    ${_LUX_CRYPTO_DIR}/sha256/cpp/sha256.cpp
    ${_LUX_CRYPTO_DIR}/sha256/cpp/sha256.hpp
    # ---- legacy ethash:: / evmmax::secp256k1 (consumer API not yet migrated) ----
    bls.hpp
    ecc.hpp
    hash_types.h
    keccak.h
    keccak.hpp
    keccak.c
    secp256k1.hpp
    secp256k1.cpp
    # Phase 5b: kzg.hpp declarations -- body in cevm_bls_kzg_canonical_cpu.
    kzg.hpp
)
