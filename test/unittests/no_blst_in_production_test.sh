#!/bin/bash
# CI assertion: production cevm binaries must not link blst.
#
# Stage 5 closure proof.  Today the cevm production library still links
# blst (cevm/cmake/blst.cmake is wired into cevm-precompiles for the
# EIP-2537 G1/G2 add/mul precompiles in cevm/lib/cevm_precompiles/bls.cpp).
# Phase 5b drops that dependency by routing those precompiles through the
# Stage 3 Metal pipeline.  This script encodes the invariant: when 5b
# lands, this assertion passes.  Until then, it documents the gap.
#
# Usage: no_blst_in_production_test.sh [BUILD_DIR]
#   BUILD_DIR defaults to "build".

set -euo pipefail

BUILD_DIR="${1:-build}"

# Production binaries that MUST be free of blst symbols.  Test/bench
# binaries (quasar-bls-verifier-test, quasar-round-bench, etc) are
# allowed to link blst as the test-time oracle; they are not on this
# list.
PRODUCTION_BINARIES=(
    "${BUILD_DIR}/lib/libevm.dylib"
    "${BUILD_DIR}/lib/libevm.so"
    "${BUILD_DIR}/lib/cevm_precompiles/libcevm_precompiles.a"
    "${BUILD_DIR}/lib/evm/libevm-precompiles.a"
    "${BUILD_DIR}/lib/evm/libevm-kernel-metal.a"
    "${BUILD_DIR}/lib/evm/libevm-gpu.a"
    "${BUILD_DIR}/lib/evm/libevm-metal-hosts.a"
    "${BUILD_DIR}/lib/evm/libprecompile-service.a"
)

EXIT_CODE=0
CHECKED=0
for bin in "${PRODUCTION_BINARIES[@]}"; do
    if [[ ! -f "$bin" ]]; then
        continue
    fi
    CHECKED=$((CHECKED + 1))
    BLST_SYMS=$(nm "$bin" 2>/dev/null | grep -cE "_blst_[a-z_]+" || true)
    if [[ "$BLST_SYMS" -gt 0 ]]; then
        echo "FAIL: $bin contains $BLST_SYMS blst symbols (production must not link blst)"
        nm "$bin" 2>/dev/null | grep -E "_blst_[a-z_]+" | head -10 || true
        EXIT_CODE=1
    else
        echo "PASS: $bin (no blst symbols)"
    fi
done

if [[ "$CHECKED" -eq 0 ]]; then
    echo "no-blst-in-production-check: no production binaries found at expected paths under ${BUILD_DIR}"
    echo "  This means the build hasn't produced them yet.  Re-run after a full build."
    exit 0
fi

if [[ "$EXIT_CODE" -ne 0 ]]; then
    echo ""
    echo "Phase 5b cleanup pending: route cevm_precompiles/bls.cpp (EIP-2537 G1/G2"
    echo "add/mul/msm) through the Stage 3 Metal pipeline + drop blst.cmake from"
    echo "the production cevm library.  This assertion is the closure proof —"
    echo "it ships today and passes once 5b lands.  Test binaries (quasar-bls-"
    echo "verifier-test etc) are allowed to link blst as the test-time oracle."
    exit 1
fi

echo "All production binaries clear of blst."
