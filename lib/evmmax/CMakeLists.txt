# cevm: Fast Ethereum Virtual Machine implementation
# Copyright 2023 The cevm Authors.
# SPDX-License-Identifier: Apache-2.0

add_library(evmmax INTERFACE)
add_library(cevm::evmmax ALIAS evmmax)
target_compile_features(evmmax INTERFACE cxx_std_20)
target_include_directories(evmmax INTERFACE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(evmmax INTERFACE intx::intx)
target_sources(
    evmmax
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include/evmmax/evmmax.hpp
)
