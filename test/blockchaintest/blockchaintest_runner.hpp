// cevm: Fast Ethereum Virtual Machine implementation
// Copyright 2023 The cevm Authors.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <test/utils/blockchaintest.hpp>

namespace cevm::test
{
void run_blockchain_tests(std::span<const BlockchainTest> tests, evmc::VM& vm);
}  // namespace cevm::test
