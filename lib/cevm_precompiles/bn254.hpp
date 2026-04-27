// Phase 5c: header forward to canonical luxcpp/crypto/bn254.
// The single source of truth lives at luxcpp/crypto/bn254/cpp/bn254.hpp.
// Note: that header includes "ecc.hpp" which still lives in this directory
// (ecc.hpp is shared infrastructure, not a per-algo file).
#pragma once
#include "bn254/cpp/bn254.hpp"
