# cevm: Fast Ethereum Virtual Machine implementation
# Copyright 2022 The cevm Authors.
# SPDX-License-Identifier: Apache-2.0

add_executable(cevm-t8n)
target_link_libraries(cevm-t8n PRIVATE cevm::state cevm::testutils evmc::evmc cevm cevm-buildinfo)
target_sources(cevm-t8n PRIVATE t8n.cpp)
