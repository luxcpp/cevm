# cevm-fuzzer: LibFuzzer based testing tool for EVMC-compatible EVM implementations.
# Copyright 2019 The cevm Authors.
# SPDX-License-Identifier: Apache-2.0
include(ExternalProject)

get_target_property(type cevm TYPE)
if(NOT type STREQUAL STATIC_LIBRARY)
    message(FATAL_ERROR "The cevm must be built as static library")
endif()

if(COVERAGE)
    set(CMAKE_EXE_LINKER_FLAGS "-fsanitize=fuzzer")
else()
    string(REPLACE fuzzer-no-link fuzzer CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS})
endif()

add_executable(cevm-fuzzer fuzzer.cpp)
target_link_libraries(cevm-fuzzer PRIVATE cevm cevm::testutils evmc::mocked_host)
