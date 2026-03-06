# cevm: Fast Ethereum Virtual Machine implementation
# Copyright 2022 The cevm Authors.
# SPDX-License-Identifier: Apache-2.0

add_executable(cevm-statetest)
target_link_libraries(cevm-statetest PRIVATE cevm::testutils cevm cevm-buildinfo GTest::gtest)
target_sources(
    cevm-statetest PRIVATE
    statetest.cpp
    statetest_runner.cpp
)
