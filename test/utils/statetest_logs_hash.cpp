// cevm: Fast Ethereum Virtual Machine implementation
// Copyright 2022 The cevm Authors.
// SPDX-License-Identifier: Apache-2.0

#include "rlp.hpp"
#include "rlp_encode.hpp"
#include "statetest.hpp"

namespace cevm::test
{
hash256 logs_hash(const std::vector<state::Log>& logs)
{
    return keccak256(rlp::encode(logs));
}
}  // namespace cevm::test
