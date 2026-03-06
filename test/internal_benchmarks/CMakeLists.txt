# cevm: Fast Ethereum Virtual Machine implementation
# Copyright 2019-2020 The cevm Authors.
# SPDX-License-Identifier: Apache-2.0

add_executable(
    cevm-bench-internal
    evmmax_bench.cpp
    find_jumpdest_bench.cpp
    memory_allocation.cpp
    lru_cache_bench.cpp
)

# TODO: Not sure why cevm::precompiles must be here, but otherwise this don't link.
target_link_libraries(cevm-bench-internal PRIVATE cevm cevm::precompiles cevm::evmmax benchmark::benchmark)
target_include_directories(cevm-bench-internal PRIVATE ${cevm_private_include_dir})
