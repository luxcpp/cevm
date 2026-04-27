// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file stm_conflict_spec_test.mm
/// Apple shim for stm_conflict_spec_test.cpp. The test is pure portable
/// C++ — there is no Metal or Foundation dependency. The .mm extension
/// exists so the cevm Apple build pattern (uses .mm for ObjC++ tests)
/// can include this target unconditionally; the .cpp body is reused
/// verbatim across Linux + macOS.

#include "stm_conflict_spec_test.cpp"
