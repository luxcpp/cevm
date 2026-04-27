#!/bin/bash
# CI assertion: production cevm binaries must not link blst.
#
# Phase 5b closure proof.  cevm_precompiles' bls.cpp + kzg.cpp now call
# the `bls12_381_*` and `bls12_381_kzg_verify_proof` extern "C" symbols
# resolved by the canonical luxcpp/crypto adapter (cevm_bls_kzg_canonical_cpu).
# That adapter links blst privately as a test-time oracle.  cevm's own
# production static archives carry no blst symbol references.
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
    echo "Production library carries blst symbol references.  Phase 5b's"
    echo "invariant is: cevm_precompiles, evm-precompiles, evm-kernel-metal,"
    echo "evm-gpu, evm-metal-hosts, precompile-service must all be blst-free."
    echo "Test binaries (quasar-bls-verifier-test etc) link blst::oracle as the"
    echo "test-time reference; that is allowed and not checked here."
    exit 1
fi

echo "All production binaries clear of blst."
