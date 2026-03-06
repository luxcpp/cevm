# cevm: Fast Ethereum Virtual Machine implementation
# Copyright 2024 The cevm Authors.
# SPDX-License-Identifier: Apache-2.0

add_executable(cevm-precompiles-bench)
target_compile_features(cevm-precompiles-bench PRIVATE cxx_std_20)
target_include_directories(cevm-precompiles-bench PRIVATE ..)
target_link_libraries(cevm-precompiles-bench PRIVATE cevm::state benchmark::benchmark)
target_sources(
    cevm-precompiles-bench PRIVATE
    precompiles_bench.cpp
)

add_test(NAME cevm/cevm-precompiles-bench COMMAND cevm-precompiles-bench --benchmark_min_time=1x)
set_tests_properties(cevm/cevm-precompiles-bench PROPERTIES FAIL_REGULAR_EXPRESSION "ERROR")
