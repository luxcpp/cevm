# cevm: Fast Ethereum Virtual Machine implementation
# Copyright 2019 The cevm Authors.
# SPDX-License-Identifier: Apache-2.0

add_library(cevm.testutils STATIC)
add_library(cevm::testutils ALIAS cevm.testutils)
target_link_libraries(cevm.testutils PUBLIC cevm::state evmc::evmc_cpp nlohmann_json::nlohmann_json)

target_sources(
    cevm.testutils
    PRIVATE
    stdx/utility.hpp
    blob_schedule.hpp
    blob_schedule.cpp
    blockchaintest.hpp
    blockchaintest_loader.cpp
    bytecode.hpp
    mpt.hpp
    mpt.cpp
    mpt_hash.hpp
    mpt_hash.cpp
    rlp.hpp
    rlp_encode.hpp
    rlp_encode.cpp
    statetest.hpp
    statetest_export.cpp
    statetest_loader.cpp
    statetest_logs_hash.cpp
    test_state.hpp
    test_state.cpp
    utils.hpp
    utils.cpp
)
