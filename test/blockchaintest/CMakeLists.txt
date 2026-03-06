# cevm: Fast Ethereum Virtual Machine implementation
# Copyright 2023 The cevm Authors.
# SPDX-License-Identifier: Apache-2.0

add_executable(cevm-blockchaintest)
target_link_libraries(cevm-blockchaintest PRIVATE cevm::testutils cevm cevm-buildinfo GTest::gtest)
target_sources(
    cevm-blockchaintest PRIVATE
    blockchaintest.cpp
    blockchaintest_runner.hpp
    blockchaintest_runner.cpp
)
