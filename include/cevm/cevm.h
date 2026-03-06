// cevm: Fast Ethereum Virtual Machine implementation
// Copyright 2018-2019 The cevm Authors.
// SPDX-License-Identifier: Apache-2.0

#ifndef CEVM_H
#define CEVM_H

#include <evmc/evmc.h>
#include <evmc/utils.h>

#if __cplusplus
extern "C" {
#endif

EVMC_EXPORT struct evmc_vm* evmc_create_cevm(void) EVMC_NOEXCEPT;

#if __cplusplus
}
#endif

#endif  // CEVM_H
