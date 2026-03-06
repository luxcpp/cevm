// cevm: Fast Ethereum Virtual Machine implementation
// Copyright 2024 The cevm Authors.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <gtest/gtest.h>

namespace cevm::test
{
class ExportableFixture : public testing::Test
{
protected:
    std::string_view export_test_name;
    std::string export_file_path;

    ExportableFixture();
};
}  // namespace cevm::test
