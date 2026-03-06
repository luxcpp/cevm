// Copyright (C) 2026, Lux Partners Limited. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for the GPU precompile dispatcher.
//
// Each precompile is exercised with at least one known-good vector pulled
// from the EIPs, the official Ethereum tests (ethereum/tests), or the
// go-ethereum precompile test corpus, and compared against the expected
// output and gas cost. The CPU and GPU paths are required to produce
// byte-identical results since they share the dispatcher's gas formulas
// and only differ in how the underlying primitive is computed.

#include <evm/gpu/precompiles/precompile_dispatch.hpp>
#include <evmc/hex.hpp>
#include <gtest/gtest.h>

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace
{
using evm::gpu::precompile::PrecompileDispatcher;
using evm::gpu::precompile::Result;

std::vector<uint8_t> hex_to_bytes(std::string_view hex)
{
    auto v = evmc::from_hex(hex);
    if (!v) return {};
    return std::vector<uint8_t>{v->begin(), v->end()};
}

std::string bytes_to_hex(const std::vector<uint8_t>& bytes)
{
    return evmc::hex({bytes.data(), bytes.size()});
}
}  // namespace

class PrecompileTest : public ::testing::Test
{
protected:
    std::unique_ptr<PrecompileDispatcher> disp_ = PrecompileDispatcher::create();
};

// =============================================================================
// 0x01 ECRECOVER
// =============================================================================
//
// Vector from go-ethereum core/vm/testdata/precompiles/ecRecover.json:
//   hash = "18c547e4f7b0f325ad1e56f57e26c745b09a3e503d86e00e5255ff7f715d3d1c"
//   v    = 28
//   r    = "73b1693892219d736caba55bdb67216e485557ea6b6af75f37096c9aa6a5a75f"
//   s    = "eeb940b1d03b21e36b0e47e79769f095fe2ab855bd91e3a38756b7d75a9c4549"
//   addr = "a94f5374fce5edbc8e2a8697c15331677e6ebf0b"
TEST_F(PrecompileTest, Ecrecover)
{
    const auto input = hex_to_bytes(
        "18c547e4f7b0f325ad1e56f57e26c745b09a3e503d86e00e5255ff7f715d3d1c"
        "000000000000000000000000000000000000000000000000000000000000001c"
        "73b1693892219d736caba55bdb67216e485557ea6b6af75f37096c9aa6a5a75f"
        "eeb940b1d03b21e36b0e47e79769f095fe2ab855bd91e3a38756b7d75a9c4549");
    ASSERT_EQ(input.size(), 128u);

    const auto r = disp_->execute(0x01, input, 100000);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.gas_used, 3000u);
    EXPECT_EQ(bytes_to_hex(r.output),
        "000000000000000000000000a94f5374fce5edbc8e2a8697c15331677e6ebf0b");
}

TEST_F(PrecompileTest, EcrecoverInvalidV)
{
    // v = 26 (must be 27 or 28) -> empty output, gas still charged.
    auto input = hex_to_bytes(
        "18c547e4f7b0f325ad1e56f57e26c745b09a3e503d86e00e5255ff7f715d3d1c"
        "000000000000000000000000000000000000000000000000000000000000001a"
        "73b1693892219d736caba55bdb67216e485557ea6b6af75f37096c9aa6a5a75f"
        "eeb940b1d03b21e36b0e47e79769f095fe2ab855bd91e3a38756b7d75a9c4549");
    const auto r = disp_->execute(0x01, input, 100000);
    EXPECT_FALSE(r.ok);
    EXPECT_EQ(r.gas_used, 3000u);
}

TEST_F(PrecompileTest, EcrecoverOutOfGas)
{
    const auto input = hex_to_bytes(
        "18c547e4f7b0f325ad1e56f57e26c745b09a3e503d86e00e5255ff7f715d3d1c"
        "000000000000000000000000000000000000000000000000000000000000001c"
        "73b1693892219d736caba55bdb67216e485557ea6b6af75f37096c9aa6a5a75f"
        "eeb940b1d03b21e36b0e47e79769f095fe2ab855bd91e3a38756b7d75a9c4549");
    const auto r = disp_->execute(0x01, input, 100);  // < 3000
    EXPECT_TRUE(r.out_of_gas);
    EXPECT_FALSE(r.ok);
}

// =============================================================================
// 0x02 SHA-256
// =============================================================================
//
// Vector from FIPS 180-4 / NIST CAVS:
//   sha256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
TEST_F(PrecompileTest, Sha256)
{
    const std::vector<uint8_t> input{'a', 'b', 'c'};
    const auto r = disp_->execute(0x02, input, 100000);
    ASSERT_TRUE(r.ok);
    // Gas = 60 + 12 * ceil(3/32) = 72.
    EXPECT_EQ(r.gas_used, 72u);
    EXPECT_EQ(bytes_to_hex(r.output),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

TEST_F(PrecompileTest, Sha256Empty)
{
    const auto r = disp_->execute(0x02, {}, 100000);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.gas_used, 60u);
    EXPECT_EQ(bytes_to_hex(r.output),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

// =============================================================================
// 0x03 RIPEMD-160
// =============================================================================
//
// Vector from RIPEMD-160 reference paper:
//   ripemd160("abc") = 8eb208f7e05d987a9b044a8e98c6b087f15a0bfc
TEST_F(PrecompileTest, Ripemd160)
{
    const std::vector<uint8_t> input{'a', 'b', 'c'};
    const auto r = disp_->execute(0x03, input, 100000);
    ASSERT_TRUE(r.ok);
    // Gas = 600 + 120 * 1 = 720.
    EXPECT_EQ(r.gas_used, 720u);
    // Output is 12 zero bytes + 20-byte digest, total 32 bytes.
    EXPECT_EQ(bytes_to_hex(r.output),
        "0000000000000000000000008eb208f7e05d987a9b044a8e98c6b087f15a0bfc");
}

// =============================================================================
// 0x04 IDENTITY
// =============================================================================
TEST_F(PrecompileTest, Identity)
{
    const auto input = hex_to_bytes("deadbeefcafebabe");
    const auto r = disp_->execute(0x04, input, 100000);
    ASSERT_TRUE(r.ok);
    // 8 bytes => 1 word => gas = 15 + 3 = 18.
    EXPECT_EQ(r.gas_used, 18u);
    EXPECT_EQ(bytes_to_hex(r.output), "deadbeefcafebabe");
}

TEST_F(PrecompileTest, IdentityEmpty)
{
    const auto r = disp_->execute(0x04, {}, 100000);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.gas_used, 15u);
    EXPECT_TRUE(r.output.empty());
}

// =============================================================================
// 0x05 MODEXP
// =============================================================================
//
// Vector from EIP-2565: 3^(2^256-1) mod (2^256 - 2^32 - 977).
// Using a smaller verifiable case: 3^5 mod 100 = 243 mod 100 = 43.
TEST_F(PrecompileTest, ModexpSmall)
{
    // base_len=1, exp_len=1, mod_len=1, base=3, exp=5, mod=100.
    auto input = hex_to_bytes(
        "0000000000000000000000000000000000000000000000000000000000000001"  // base_len
        "0000000000000000000000000000000000000000000000000000000000000001"  // exp_len
        "0000000000000000000000000000000000000000000000000000000000000001"  // mod_len
        "03"   // base = 3
        "05"   // exp = 5
        "64"); // mod = 100

    const auto r = disp_->execute(0x05, input, 100000);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.gas_used, 200u);  // EIP-2565 minimum
    ASSERT_EQ(r.output.size(), 1u);
    EXPECT_EQ(r.output[0], 43);
}

// =============================================================================
// 0x06 BN256_ADD
// =============================================================================
//
// Vector from go-ethereum precompiles_test.go (CALLBN256ADD):
//   P + 0 = P, where P = (1, 2)
TEST_F(PrecompileTest, Bn256AddIdentity)
{
    auto input = hex_to_bytes(
        "0000000000000000000000000000000000000000000000000000000000000001"
        "0000000000000000000000000000000000000000000000000000000000000002"
        "0000000000000000000000000000000000000000000000000000000000000000"
        "0000000000000000000000000000000000000000000000000000000000000000");

    const auto r = disp_->execute(0x06, input, 100000);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.gas_used, 150u);
    EXPECT_EQ(bytes_to_hex(r.output),
        "00000000000000000000000000000000000000000000000000000000000000010"
        "000000000000000000000000000000000000000000000000000000000000002");
}

// =============================================================================
// 0x07 BN256_MUL
// =============================================================================
//
// Vector: 2 * (1, 2) = (1368015179489954701390400359078579693043519447331113978918064868415326638035,
//                      9918110051302171585080402603319702774565515993150576347155970296011453393214)
TEST_F(PrecompileTest, Bn256MulByTwo)
{
    auto input = hex_to_bytes(
        "0000000000000000000000000000000000000000000000000000000000000001"   // x
        "0000000000000000000000000000000000000000000000000000000000000002"   // y
        "0000000000000000000000000000000000000000000000000000000000000002"); // scalar

    const auto r = disp_->execute(0x07, input, 100000);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.gas_used, 6000u);
    // Reference output computed by cevm bn254 (matches geth bn256
    // ScalarMult of (1, 2) by 2 over the curve in EIP-196 layout).
    EXPECT_EQ(bytes_to_hex(r.output),
        "030644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd3"
        "15ed738c0e0a7c92e7845f96b2ae9c0a68a6a449e3538fc7ff3ebf7a5a18a2c4");
}

// =============================================================================
// 0x08 BN256_PAIRING (EIP-197)
// =============================================================================
//
// Empty input vector: e() = 1 (vacuous truth — pairing of zero pairs is 1).
// Source: go-ethereum core/vm/contracts_test.go (precompile bn256Pairing) and
// the canonical spec test (https://eips.ethereum.org/EIPS/eip-197).
// Output is 32-byte big-endian 1; gas = 45000 + 34000 * 0 = 45000.
TEST_F(PrecompileTest, Bn256PairingEmpty)
{
    const std::vector<uint8_t> input;  // 0 pairs
    const auto r = disp_->execute(0x08, input, 100000);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.gas_used, 45000u);
    EXPECT_EQ(bytes_to_hex(r.output),
        "0000000000000000000000000000000000000000000000000000000000000001");
}

// EIP-197 §5 test: e(P, Q) * e(-P, Q) = 1.
// Vector from go-ethereum core/vm/testdata/precompiles/bn256Pairing.json
// (case "jeff1") — two pairs that cancel, expected output = 1.
TEST_F(PrecompileTest, Bn256PairingJeff1)
{
    const auto input = hex_to_bytes(
        "1c76476f4def4bb94541d57ebba1193381ffa7aa76ada664dd31c16024c43f59"
        "3034dd2920f673e204fee2811c678745fc819b55d3e9d294e45c9b03a76aef41"
        "209dd15ebff5d46c4bd888e51a93cf99a7329636c63514396b4a452003a35bf7"
        "04bf11ca01483bfa8b34b43561848d28905960114c8ac04049af4b6315a41678"
        "2bb8324af6cfc93537a2ad1a445cfd0ca2a71acd7ac41fadbf933c2a51be344d"
        "120a2a4cf30c1bf9845f20c6fe39e07ea2cce61f0c9bb048165fe5e4de877550"
        "111e129f1cf1097710d41c4ac70fcdfa5ba2023c6ff1cbeac322de49d1b6df7c"
        "2032c61a830e3c17286de9462bf242fca2883585b93870a73853face6a6bf411"
        "198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c2"
        "1800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed"
        "090689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b"
        "12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa");
    ASSERT_EQ(input.size(), 384u);
    const auto r = disp_->execute(0x08, input, 200000);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.gas_used, 45000u + 2u * 34000u);
    EXPECT_EQ(bytes_to_hex(r.output),
        "0000000000000000000000000000000000000000000000000000000000000001");
}

// =============================================================================
// 0x09 BLAKE2F (EIP-152)
// =============================================================================
//
// Vector from EIP-152 reference: rounds=12 with empty state.
TEST_F(PrecompileTest, Blake2fInvalidSize)
{
    // 200-byte input: invalid (must be exactly 213).
    std::vector<uint8_t> input(200, 0);
    const auto r = disp_->execute(0x09, input, 100000);
    EXPECT_TRUE(r.out_of_gas);
}

TEST_F(PrecompileTest, Blake2fInvalidF)
{
    std::vector<uint8_t> input(213, 0);
    input[0] = 0; input[1] = 0; input[2] = 0; input[3] = 12;  // rounds = 12
    input[212] = 2;  // f must be 0 or 1
    const auto r = disp_->execute(0x09, input, 100000);
    EXPECT_FALSE(r.ok);
    EXPECT_EQ(r.gas_used, 12u);
}

// EIP-152 test vector 4 (rounds=12, BLAKE2b("abc"), f=1).
// Source: https://eips.ethereum.org/EIPS/eip-152#test-cases
// rounds=12, h=IV with h[0] ^= 0x01010040, m="abc"+pad, t=(3,0), f=1.
// h[0] = 0x6a09e667f3bcc908 ^ 0x01010040 = 0x6a09e667f2bdc948,
// stored little-endian as bytes "48c9bdf267e6096a..." (8 bytes per word).
// Expected output is the standard BLAKE2b("abc") digest.
TEST_F(PrecompileTest, Blake2fEip152Vector4)
{
    const auto input = hex_to_bytes(
        "0000000c"                                                          // rounds = 12
        "48c9bdf267e6096a3ba7ca8485ae67bb2bf894fe72f36e3cf1361d5f3af54fa5"  // h[0..3]
        "d182e6ad7f520e511f6c3e2b8c68059b6bbd41fbabd9831f79217e1319cde05b"  // h[4..7]
        "6162630000000000000000000000000000000000000000000000000000000000"  // m "abc"
        "0000000000000000000000000000000000000000000000000000000000000000"
        "0000000000000000000000000000000000000000000000000000000000000000"
        "0000000000000000000000000000000000000000000000000000000000000000"
        "0300000000000000"                                                  // t[0] = 3
        "0000000000000000"                                                  // t[1] = 0
        "01");                                                              // f = 1
    ASSERT_EQ(input.size(), 213u);

    const auto r = disp_->execute(0x09, input, 100000);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.gas_used, 12u);  // gas = rounds = 12
    EXPECT_EQ(bytes_to_hex(r.output),
        "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d1"
        "7d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923");
}

// =============================================================================
// 0x0a POINT_EVALUATION (EIP-4844)
// =============================================================================
//
// Vector: invalid input size => failure.
TEST_F(PrecompileTest, PointEvaluationInvalidSize)
{
    const std::vector<uint8_t> input(100, 0);
    const auto r = disp_->execute(0x0a, input, 100000);
    EXPECT_FALSE(r.ok);
    EXPECT_EQ(r.gas_used, 50000u);
}

// EIP-4844 KZG verify proof: f(x) = 0 polynomial.
// Source: precompiles_kzg_test.cpp TEST(kzg, verify_proof_zero) — same
// vector, exercised here through the precompile dispatcher boundary.
// Input layout: versioned_hash[32] || z[32] || y[32] || commitment[48] || proof[48].
// commitment & proof = G1 point at infinity (compressed: 0xC0 + 47 zero bytes).
// versioned_hash = sha256(commitment) with first byte replaced by 0x01:
//   sha256(c0 || 00*47) = "0657f37554c781402a22917dee2f75def7ab966d7b770905398eba3c44440140"
//   versioned hash = "01" || tail_31 -> "010657f37554c781402a22917dee2f75def7ab966d7b770905398eba3c444014"
// y = 0 (f(z) = 0 for any z), z = arbitrary nonzero.
// Expected output: FIELD_ELEMENTS_PER_BLOB[32 BE] || BLS_MODULUS[32 BE]
//   = 4096 || 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001.
TEST_F(PrecompileTest, PointEvaluationZeroPoly)
{
    const auto input = hex_to_bytes(
        // versioned_hash = 0x01 || tail-31-of-sha256(point_at_infinity)
        "010657f37554c781402a22917dee2f75def7ab966d7b770905398eba3c444014"
        // z (arbitrary, byte 13 = 17 — same as kzg unit test)
        "0000000000000000000000000011000000000000000000000000000000000000"
        // y = 0
        "0000000000000000000000000000000000000000000000000000000000000000"
        // commitment = G1 point at infinity (compressed: 0xc0 + 47 zeros)
        "c000000000000000000000000000000000000000000000000000000000000000"
        "00000000000000000000000000000000"
        // proof = G1 point at infinity
        "c000000000000000000000000000000000000000000000000000000000000000"
        "00000000000000000000000000000000");
    ASSERT_EQ(input.size(), 192u);

    const auto r = disp_->execute(0x0a, input, 100000);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.gas_used, 50000u);
    // 4096 = 0x1000 (FIELD_ELEMENTS_PER_BLOB) || BLS_MODULUS.
    EXPECT_EQ(bytes_to_hex(r.output),
        "0000000000000000000000000000000000000000000000000000000000001000"
        "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001");
}

// =============================================================================
// 0x0b BLS12_G1ADD (EIP-2537)
// =============================================================================
//
// Vector from EIP-2537 spec test corpus: G1 + (-G1) = 0.
TEST_F(PrecompileTest, Bls12G1AddInvalidSize)
{
    const std::vector<uint8_t> input(100, 0);  // wrong size
    const auto r = disp_->execute(0x0b, input, 100000);
    EXPECT_FALSE(r.ok);
    EXPECT_EQ(r.gas_used, 375u);
}

// EIP-2537 G1ADD: G1_1 + G1_2 == G1_3.
// Source: precompiles_bls_test.cpp TEST(bls, g1_add) — vectors mirror
// https://github.com/ethereum/EIPs/blob/master/assets/eip-2537/add_G1_bls.json
// G1_1 is the BLS12-381 G1 generator; the addition checks the underlying
// blst affine add. Each coordinate is 64 bytes (16-byte zero pad + 48-byte fp).
TEST_F(PrecompileTest, Bls12G1AddGenerators)
{
    const auto input = hex_to_bytes(
        // G1_1 (generator)
        "0000000000000000000000000000000017f1d3a73197d7942695638c4fa9ac0f"
        "c3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb"
        "0000000000000000000000000000000008b3f481e3aaa0f1a09e30ed741d8ae4"
        "fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1"
        // G1_2
        "00000000000000000000000000000000112b98340eee2777cc3c14163dea3ec9"
        "7977ac3dc5c70da32e6e87578f44912e902ccef9efe28d4a78b8999dfbca9426"
        "00000000000000000000000000000000186b28d92356c4dfec4b5201ad099dbd"
        "ede3781f8998ddf929b4cd7756192185ca7b8f4ef7088f813270ac3d48868a21");
    ASSERT_EQ(input.size(), 256u);

    const auto r = disp_->execute(0x0b, input, 10000);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.gas_used, 375u);
    EXPECT_EQ(bytes_to_hex(r.output),
        "000000000000000000000000000000000a40300ce2dec9888b60690e9a41d300"
        "4fda4886854573974fab73b046d3147ba5b7a5bde85279ffede1b45b3918d82d"
        "0000000000000000000000000000000006d3d887e9f53b9ec4eb6cedf5607226"
        "754b07c01ace7834f57f3e7315faefb739e59018e22c492006190fba4a870025");
}

// =============================================================================
// 0x0c BLS12_G1MSM (EIP-2537)
// =============================================================================
//
// Single-pair MSM: 2 * G1 generator. k=1, gas = 12000 * 1000/1000 = 12000.
// Source: precompiles_bls_test.cpp TEST(bls, g1_mul) and (bls, g1_msm) —
// matches EIP-2537 spec corpus mul_G1_bls.json case "bls_g1mul_(g1+g1=2*g1)".
TEST_F(PrecompileTest, Bls12G1MsmGeneratorTimes2)
{
    const auto input = hex_to_bytes(
        // G1 generator
        "0000000000000000000000000000000017f1d3a73197d7942695638c4fa9ac0f"
        "c3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb"
        "0000000000000000000000000000000008b3f481e3aaa0f1a09e30ed741d8ae4"
        "fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1"
        // scalar = 2
        "0000000000000000000000000000000000000000000000000000000000000002");
    ASSERT_EQ(input.size(), 160u);

    const auto r = disp_->execute(0x0c, input, 100000);
    ASSERT_TRUE(r.ok);
    // EIP-2537: k=1 => discount=1000, gas = 12000 * 1000 * 1 / 1000 = 12000.
    EXPECT_EQ(r.gas_used, 12000u);
    EXPECT_EQ(bytes_to_hex(r.output),
        "000000000000000000000000000000000572cbea904d67468808c8eb50a9450c"
        "9721db309128012543902d0ac358a62ae28f75bb8f1c7c42c39a8c5529bf0f4e"
        "00000000000000000000000000000000166a9d8cabc673a322fda673779d8e38"
        "22ba3ecb8670e461f73bb9021d5fd76a4c56d9d4cd16bd1bba86881979749d28");
}

// =============================================================================
// 0x0d BLS12_G2ADD (EIP-2537)
// =============================================================================
//
// G2 generator + a known G2 point.
// Source: precompiles_bls_test.cpp TEST(bls, g2_add) — matches EIP-2537 spec
// add_G2_bls.json. Each G2 point is 256 bytes (4 × 64-byte fp).
TEST_F(PrecompileTest, Bls12G2AddGenerators)
{
    const auto input = hex_to_bytes(
        // G2_1 x: c0 || c1
        "00000000000000000000000000000000024aa2b2f08f0a91260805272dc51051"
        "c6e47ad4fa403b02b4510b647ae3d1770bac0326a805bbefd48056c8c121bdb8"
        "0000000000000000000000000000000013e02b6052719f607dacd3a088274f65"
        "596bd0d09920b61ab5da61bbdc7f5049334cf11213945d57e5ac7d055d042b7e"
        // G2_1 y: c0 || c1
        "000000000000000000000000000000000ce5d527727d6e118cc9cdc6da2e351a"
        "adfd9baa8cbdd3a76d429a695160d12c923ac9cc3baca289e193548608b82801"
        "000000000000000000000000000000000606c4a02ea734cc32acd2b02bc28b99"
        "cb3e287e85a763af267492ab572e99ab3f370d275cec1da1aaa9075ff05f79be"
        // Second point x
        "00000000000000000000000000000000103121a2ceaae586d240843a39896732"
        "5f8eb5a93e8fea99b62b9f88d8556c80dd726a4b30e84a36eeabaf3592937f27"
        "00000000000000000000000000000000086b990f3da2aeac0a36143b7d7c8244"
        "28215140db1bb859338764cb58458f081d92664f9053b50b3fbd2e4723121b68"
        // Second point y
        "000000000000000000000000000000000f9e7ba9a86a8f7624aa2b42dcc8772e"
        "1af4ae115685e60abc2c9b90242167acef3d0be4050bf935eed7c3b6fc7ba77e"
        "000000000000000000000000000000000d22c3652d0dc6f0fc9316e14268477c"
        "2049ef772e852108d269d9c38dba1d4802e8dae479818184c08f9a569d878451");
    ASSERT_EQ(input.size(), 512u);

    const auto r = disp_->execute(0x0d, input, 10000);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.gas_used, 600u);
    EXPECT_EQ(bytes_to_hex(r.output),
        "000000000000000000000000000000000b54a8a7b08bd6827ed9a797de216b8c"
        "9057b3a9ca93e2f88e7f04f19accc42da90d883632b9ca4dc38d013f71ede4db"
        "00000000000000000000000000000000077eba4eecf0bd764dce8ed5f45040dd"
        "8f3b3427cb35230509482c14651713282946306247866dfe39a8e33016fcbe52"
        "0000000000000000000000000000000014e60a76a29ef85cbd69f251b9f29147"
        "b67cfe3ed2823d3f9776b3a0efd2731941d47436dc6d2b58d9e65f8438bad073"
        "000000000000000000000000000000001586c3c910d95754fef7a732df78e279"
        "c3d37431c6a2b77e67a00c7c130a8fcd4d19f159cbeb997a178108fffffcbd20");
}

// =============================================================================
// 0x0e BLS12_G2MSM (EIP-2537)
// =============================================================================
//
// Single-pair MSM: 2 * G2 generator. k=1, gas = 22500 * 1000/1000 = 22500.
// Source: precompiles_bls_test.cpp TEST(bls, g2_mul) — EIP-2537 spec corpus
// mul_G2_bls.json case "bls_g2mul_(g2+g2=2*g2)".
TEST_F(PrecompileTest, Bls12G2MsmGeneratorTimes2)
{
    const auto input = hex_to_bytes(
        // G2 generator
        "00000000000000000000000000000000024aa2b2f08f0a91260805272dc51051"
        "c6e47ad4fa403b02b4510b647ae3d1770bac0326a805bbefd48056c8c121bdb8"
        "0000000000000000000000000000000013e02b6052719f607dacd3a088274f65"
        "596bd0d09920b61ab5da61bbdc7f5049334cf11213945d57e5ac7d055d042b7e"
        "000000000000000000000000000000000ce5d527727d6e118cc9cdc6da2e351a"
        "adfd9baa8cbdd3a76d429a695160d12c923ac9cc3baca289e193548608b82801"
        "000000000000000000000000000000000606c4a02ea734cc32acd2b02bc28b99"
        "cb3e287e85a763af267492ab572e99ab3f370d275cec1da1aaa9075ff05f79be"
        // scalar = 2
        "0000000000000000000000000000000000000000000000000000000000000002");
    ASSERT_EQ(input.size(), 288u);

    const auto r = disp_->execute(0x0e, input, 100000);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.gas_used, 22500u);
    EXPECT_EQ(bytes_to_hex(r.output),
        "000000000000000000000000000000001638533957d540a9d2370f17cc7ed586"
        "3bc0b995b8825e0ee1ea1e1e4d00dbae81f14b0bf3611b78c952aacab827a053"
        "000000000000000000000000000000000a4edef9c1ed7f729f520e47730a124f"
        "d70662a904ba1074728114d1031e1572c6c886f6b57ec72a6178288c47c33577"
        "000000000000000000000000000000000468fb440d82b0630aeb8dca2b525678"
        "9a66da69bf91009cbfe6bd221e47aa8ae88dece9764bf3bd999d95d71e4c9899"
        "000000000000000000000000000000000f6d4552fa65dd2638b361543f887136"
        "a43253d9c66c411697003f7a13c308f5422e1aa0a59c8967acdefd8b6e36ccf3");
}

// =============================================================================
// 0x0f BLS12_PAIRING_CHECK (EIP-2537)
// =============================================================================
//
// One pair: e(G1_inf, G2_1) = 1 (any pairing with infinity is the identity).
// Source: precompiles_bls_test.cpp TEST(bls, paring_check_one_pair_g1_inf) —
// matches the EIP-2537 spec pairing_check.json case for inf-G1.
// Gas = 37700 + 32600 * 1 = 70300.
TEST_F(PrecompileTest, Bls12PairingCheckG1InfG2Gen)
{
    const auto input = hex_to_bytes(
        // G1 point at infinity (128 zero bytes = 256 hex chars)
        "0000000000000000000000000000000000000000000000000000000000000000"
        "0000000000000000000000000000000000000000000000000000000000000000"
        "0000000000000000000000000000000000000000000000000000000000000000"
        "0000000000000000000000000000000000000000000000000000000000000000"
        // G2 generator (256 bytes)
        "00000000000000000000000000000000024aa2b2f08f0a91260805272dc51051"
        "c6e47ad4fa403b02b4510b647ae3d1770bac0326a805bbefd48056c8c121bdb8"
        "0000000000000000000000000000000013e02b6052719f607dacd3a088274f65"
        "596bd0d09920b61ab5da61bbdc7f5049334cf11213945d57e5ac7d055d042b7e"
        "000000000000000000000000000000000ce5d527727d6e118cc9cdc6da2e351a"
        "adfd9baa8cbdd3a76d429a695160d12c923ac9cc3baca289e193548608b82801"
        "000000000000000000000000000000000606c4a02ea734cc32acd2b02bc28b99"
        "cb3e287e85a763af267492ab572e99ab3f370d275cec1da1aaa9075ff05f79be");
    ASSERT_EQ(input.size(), 384u);

    const auto r = disp_->execute(0x0f, input, 100000);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.gas_used, 70300u);  // 37700 + 32600 * 1
    EXPECT_EQ(bytes_to_hex(r.output),
        "0000000000000000000000000000000000000000000000000000000000000001");
}

// =============================================================================
// 0x10 BLS12_MAP_FP_TO_G1 (EIP-2537)
// =============================================================================
//
// Map a 64-byte fp element to a G1 point.
// Source: precompiles_bls_test.cpp TEST(bls, map_fp_to_g1) — corresponds to
// EIP-2537 spec corpus fp_to_G1_bls.json.
TEST_F(PrecompileTest, Bls12MapFpToG1)
{
    const auto input = hex_to_bytes(
        "00000000000000000000000000000000156c8a6a2c184569d69a76be144b5cdc"
        "5141d2d2ca4fe341f011e25e3969c55ad9e9b9ce2eb833c81a908e5fa4ac5f03");
    ASSERT_EQ(input.size(), 64u);

    const auto r = disp_->execute(0x10, input, 100000);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.gas_used, 5500u);
    EXPECT_EQ(bytes_to_hex(r.output),
        "00000000000000000000000000000000184bb665c37ff561a89ec2122dd343f2"
        "0e0f4cbcaec84e3c3052ea81d1834e192c426074b02ed3dca4e7676ce4ce48ba"
        "0000000000000000000000000000000004407b8d35af4dacc809927071fc0405"
        "218f1401a6d15af775810e4e460064bcc9468beeba82fdc751be70476c888bf3");
}

// =============================================================================
// 0x11 BLS12_MAP_FP2_TO_G2 (EIP-2537)
// =============================================================================
//
// Map a 128-byte fp2 element to a G2 point.
// Source: precompiles_bls_test.cpp TEST(bls, map_fp2_to_g2) — matches EIP-2537
// fp2_to_G2_bls.json.
TEST_F(PrecompileTest, Bls12MapFp2ToG2)
{
    const auto input = hex_to_bytes(
        "0000000000000000000000000000000007355d25caf6e7f2f0cb2812ca0e513b"
        "d026ed09dda65b177500fa31714e09ea0ded3a078b526bed3307f804d4b93b04"
        "0000000000000000000000000000000002829ce3c021339ccb5caf3e187f6370"
        "e1e2a311dec9b75363117063ab2015603ff52c3d3b98f19c2f65575e99e8b78c");
    ASSERT_EQ(input.size(), 128u);

    const auto r = disp_->execute(0x11, input, 100000);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.gas_used, 23800u);
    EXPECT_EQ(bytes_to_hex(r.output),
        "0000000000000000000000000000000000e7f4568a82b4b7dc1f14c6aaa055ed"
        "f51502319c723c4dc2688c7fe5944c213f510328082396515734b6612c4e7bb7"
        "00000000000000000000000000000000126b855e9e69b1f691f816e48ac69776"
        "64d24d99f8724868a184186469ddfd4617367e94527d4b74fc86413483afb35b"
        "000000000000000000000000000000000caead0fd7b6176c01436833c79d305c"
        "78be307da5f6af6c133c47311def6ff1e0babf57a0fb5539fce7ee12407b0a42"
        "000000000000000000000000000000001498aadcf7ae2b345243e281ae076df6"
        "de84455d766ab6fcdaad71fab60abb2e8b980a440043cd305db09d283c895e3d");
}

// =============================================================================
// Out-of-range and unhandled
// =============================================================================
TEST_F(PrecompileTest, Address0x00Unhandled)
{
    const auto r = disp_->execute(0x00, {}, 100000);
    EXPECT_FALSE(r.ok);
    EXPECT_EQ(r.gas_used, 0u);
}

TEST_F(PrecompileTest, Address0x12Unhandled)
{
    const auto r = disp_->execute(0x12, {}, 100000);
    EXPECT_FALSE(r.ok);
    EXPECT_EQ(r.gas_used, 0u);
}

TEST_F(PrecompileTest, AvailabilityRange)
{
    for (uint8_t a = 0x01; a <= 0x11; ++a)
        EXPECT_TRUE(disp_->available(a)) << "addr=0x" << std::hex << int(a);
    EXPECT_FALSE(disp_->available(0x00));
    EXPECT_FALSE(disp_->available(0x12));
}

TEST_F(PrecompileTest, BackendNamesStrings)
{
    // Backend names shouldn't be null and should be a recognised tag.
    for (uint8_t a = 0x01; a <= 0x11; ++a)
    {
        const char* n = disp_->backend_name(a);
        ASSERT_NE(n, nullptr);
        const std::string s = n;
        EXPECT_TRUE(s == "cpu" || s == "metal" || s == "cuda")
            << "addr=0x" << std::hex << int(a) << " backend=" << s;
    }
}
