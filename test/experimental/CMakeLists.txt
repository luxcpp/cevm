# cevm: Fast Ethereum Virtual Machine implementation
# Copyright 2020 The cevm Authors.
# SPDX-License-Identifier: Apache-2.0

add_library(cevm-experimental STATIC)
add_library(cevm::experimental ALIAS cevm-experimental)
target_link_libraries(cevm-experimental PRIVATE cevm)
target_include_directories(cevm-experimental PRIVATE ${cevm_private_include_dir})
target_sources(
    cevm-experimental PRIVATE
    jumpdest_analysis.hpp
    jumpdest_analysis.cpp
)
