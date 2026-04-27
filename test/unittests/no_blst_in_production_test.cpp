// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Phase 5b closure-proof test runner.
//
// Spawns the no_blst_in_production_test.sh assertion against the build
// directory the test was launched from.  Exits with the script's exit
// code so ctest reports it as PASS/FAIL.
//
// Phase 5b: cevm_precompiles' bls.cpp + kzg.cpp call extern "C" symbols
// (`bls12_381_*`, `bls12_381_kzg_verify_proof`) defined by the canonical
// luxcpp/crypto adapter (cevm_bls_kzg_canonical_cpu).  No production
// static archive carries blst symbols; this test asserts that invariant.

#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv)
{
    const char* build_dir = (argc >= 2) ? argv[1] : "build";
    const char* script_path = (argc >= 3)
        ? argv[2]
        : "test/unittests/no_blst_in_production_test.sh";

    std::string cmd = "/bin/bash ";
    cmd += script_path;
    cmd += " ";
    cmd += build_dir;

    std::printf("[no_blst_in_production_test] running: %s\n", cmd.c_str());
    std::fflush(stdout);

    int rc = std::system(cmd.c_str());
    if (rc == -1) {
        std::fprintf(stderr,
            "no_blst_in_production_test: failed to spawn %s\n",
            script_path);
        return 2;
    }
    // POSIX: use WEXITSTATUS, but for portability we just compare to 0.
    return rc == 0 ? 0 : 1;
}
