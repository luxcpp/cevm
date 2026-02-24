// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// EIP-2537 BLS12-381 precompiles (CPU, via blst):
//   0x0b G1ADD            — gas 375
//   0x0c G1MSM            — gas 12000 * k * discount(k) / 1000
//   0x0d G2ADD            — gas 600
//   0x0e G2MSM            — gas 22500 * k * discount(k) / 1000
//   0x0f PAIRING_CHECK    — gas 37700 + 32600 * k
//   0x10 MAP_FP_TO_G1     — gas 5500
//   0x11 MAP_FP2_TO_G2    — gas 23800

#include "internal.hpp"

#include <evmone_precompiles/bls.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <vector>

namespace evm::gpu::precompile
{
namespace
{
// Source: EIP-2537 § Discount table.
constexpr uint16_t kG1Discounts[] = {
    1000, 949, 848, 797, 764, 750, 738, 728, 719, 712, 705, 698, 692, 687, 682,
    677, 673, 669, 665, 661, 658, 654, 651, 648, 645, 642, 640, 637, 635, 632,
    630, 627, 625, 623, 621, 619, 617, 615, 613, 611, 609, 608, 606, 604, 603,
    601, 599, 598, 596, 595, 593, 592, 591, 589, 588, 586, 585, 584, 582, 581,
    580, 579, 577, 576, 575, 574, 573, 572, 570, 569, 568, 567, 566, 565, 564,
    563, 562, 561, 560, 559, 558, 557, 556, 555, 554, 553, 552, 551, 550, 549,
    548, 547, 547, 546, 545, 544, 543, 542, 541, 540, 540, 539, 538, 537, 536,
    536, 535, 534, 533, 532, 532, 531, 530, 529, 528, 528, 527, 526, 525, 525,
    524, 523, 522, 522, 521, 520, 520, 519};
constexpr uint16_t kG2Discounts[] = {
    1000, 1000, 923, 884, 855, 832, 812, 796, 782, 770, 759, 749, 740, 732, 724,
    717, 711, 704, 699, 693, 688, 683, 679, 674, 670, 666, 663, 659, 655, 652,
    649, 646, 643, 640, 637, 634, 632, 629, 627, 624, 622, 620, 618, 615, 613,
    611, 609, 607, 606, 604, 602, 600, 598, 597, 595, 593, 592, 590, 589, 587,
    586, 584, 583, 582, 580, 579, 578, 576, 575, 574, 573, 571, 570, 569, 568,
    567, 566, 565, 563, 562, 561, 560, 559, 558, 557, 556, 555, 554, 553, 552,
    552, 551, 550, 549, 548, 547, 546, 545, 545, 544, 543, 542, 541, 541, 540,
    539, 538, 537, 537, 536, 535, 535, 534, 533, 532, 532, 531, 530, 530, 529,
    528, 528, 527, 526, 526, 525, 524, 524};
}  // namespace

Result bls12_g1add_cpu(std::span<const uint8_t> input, uint64_t gas_limit)
{
    using namespace detail;
    constexpr uint64_t kGas = 375;
    if (gas_limit < kGas)
        return make_oog();
    if (input.size() != 2 * kBlsG1Size)
        return make_failure(kGas);

    std::vector<uint8_t> out(kBlsG1Size);
    if (!evmone::crypto::bls::g1_add(out.data(), out.data() + 64,
            input.data(), input.data() + 64, input.data() + 128, input.data() + 192))
        return make_failure(kGas);
    return make_ok(kGas, std::move(out));
}

Result bls12_g1msm_cpu(std::span<const uint8_t> input, uint64_t gas_limit)
{
    using namespace detail;
    if (input.empty() || input.size() % kBlsG1MulInputSize != 0)
        return make_oog();
    const uint64_t k = input.size() / kBlsG1MulInputSize;
    const uint64_t discount =
        kG1Discounts[std::min<size_t>(k, std::size(kG1Discounts)) - 1];
    const uint64_t gas = (12000ULL * discount * k) / 1000;
    if (gas_limit < gas)
        return make_oog();

    std::vector<uint8_t> out(kBlsG1Size);
    const bool ok = (k == 1)
        ? evmone::crypto::bls::g1_mul(out.data(), out.data() + 64,
              input.data(), input.data() + 64, input.data() + 128)
        : evmone::crypto::bls::g1_msm(out.data(), out.data() + 64,
              input.data(), input.size());
    if (!ok) return make_failure(gas);
    return make_ok(gas, std::move(out));
}

Result bls12_g2add_cpu(std::span<const uint8_t> input, uint64_t gas_limit)
{
    using namespace detail;
    constexpr uint64_t kGas = 600;
    if (gas_limit < kGas)
        return make_oog();
    if (input.size() != 2 * kBlsG2Size)
        return make_failure(kGas);

    std::vector<uint8_t> out(kBlsG2Size);
    if (!evmone::crypto::bls::g2_add(out.data(), out.data() + 128,
            input.data(), input.data() + 128, input.data() + 256, input.data() + 384))
        return make_failure(kGas);
    return make_ok(kGas, std::move(out));
}

Result bls12_g2msm_cpu(std::span<const uint8_t> input, uint64_t gas_limit)
{
    using namespace detail;
    if (input.empty() || input.size() % kBlsG2MulInputSize != 0)
        return make_oog();
    const uint64_t k = input.size() / kBlsG2MulInputSize;
    const uint64_t discount =
        kG2Discounts[std::min<size_t>(k, std::size(kG2Discounts)) - 1];
    const uint64_t gas = (22500ULL * discount * k) / 1000;
    if (gas_limit < gas)
        return make_oog();

    std::vector<uint8_t> out(kBlsG2Size);
    const bool ok = (k == 1)
        ? evmone::crypto::bls::g2_mul(out.data(), out.data() + 128,
              input.data(), input.data() + 128, input.data() + 256)
        : evmone::crypto::bls::g2_msm(out.data(), out.data() + 128,
              input.data(), input.size());
    if (!ok) return make_failure(gas);
    return make_ok(gas, std::move(out));
}

Result bls12_pairing_cpu(std::span<const uint8_t> input, uint64_t gas_limit)
{
    using namespace detail;
    constexpr size_t kPairSize = kBlsG1Size + kBlsG2Size;  // 384
    if (input.empty() || input.size() % kPairSize != 0)
        return make_oog();

    const uint64_t k = input.size() / kPairSize;
    const uint64_t gas = 37700 + 32600 * k;
    if (gas_limit < gas)
        return make_oog();

    std::vector<uint8_t> out(32, 0);
    if (!evmone::crypto::bls::pairing_check(out.data(), input.data(), input.size()))
        return make_failure(gas);
    return make_ok(gas, std::move(out));
}

Result bls12_map_fp_to_g1_cpu(std::span<const uint8_t> input, uint64_t gas_limit)
{
    using namespace detail;
    constexpr uint64_t kGas = 5500;
    if (gas_limit < kGas)
        return make_oog();
    if (input.size() != kBlsFieldSize)
        return make_failure(kGas);

    std::vector<uint8_t> out(kBlsG1Size);
    if (!evmone::crypto::bls::map_fp_to_g1(out.data(), out.data() + 64, input.data()))
        return make_failure(kGas);
    return make_ok(kGas, std::move(out));
}

Result bls12_map_fp2_to_g2_cpu(std::span<const uint8_t> input, uint64_t gas_limit)
{
    using namespace detail;
    constexpr uint64_t kGas = 23800;
    if (gas_limit < kGas)
        return make_oog();
    if (input.size() != 2 * kBlsFieldSize)
        return make_failure(kGas);

    std::vector<uint8_t> out(kBlsG2Size);
    if (!evmone::crypto::bls::map_fp2_to_g2(out.data(), out.data() + 128, input.data()))
        return make_failure(kGas);
    return make_ok(kGas, std::move(out));
}
}  // namespace evm::gpu::precompile
