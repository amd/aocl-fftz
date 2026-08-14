// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fused_twiddle_transpose_kernel_gtest.cpp
 *
 *  @brief GTest unit tests for the fused four-step twiddle + transpose kernels.
 *
 *  Exercises every registered kernel variant (scalar always; AVX128/256/512
 *  when built) across both precisions, both directions, square/non-square
 *  geometries and padded row strides, comparing each against an independent
 *  reference.
 *
 *  @author Ashwin K. Godbole
 */

#include <string>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>

#include "test/gtest/fused_twiddle_transpose/fused_twiddle_transpose_kernel_gtest_base.h"

// Builds a human-readable test name: <n1>x<n2>_is<pad>_os<pad>_<dir>_<isa>
static std::string tt_name(const FusedTwiddleTransposeParam &param)
{
    const FFTZ_INTP n1 = std::get<0>(std::get<0>(param));
    const FFTZ_INTP n2 = std::get<1>(std::get<0>(param));
    const FFTZ_INTP in_pad = std::get<1>(param);
    const FFTZ_INTP out_pad = std::get<2>(param);
    const FFTZ_INT32 dir = std::get<3>(param);
    const FFTZ_INT32 isa = std::get<4>(param);

    std::string isa_name = "isa" + std::to_string(isa);
    for (const auto &v : tt_isa_variants())
    {
        if (v.first == isa)
        {
            isa_name = v.second;
            break;
        }
    }

    std::string name = std::to_string(n1) + "x" + std::to_string(n2);
    name += "_is" + std::to_string(in_pad);
    name += "_os" + std::to_string(out_pad);
    name += (dir == FORWARD_FFT_DIR) ? "_fwd" : "_bwd";
    name += "_" + isa_name;
    return name;
}

static auto name_generator =
    [](const ::testing::TestParamInfo<FusedTwiddleTransposeParam> &info)
{ return tt_name(info.param); };

TEST_P(AoclfftzFusedTwiddleTransposeKernelTestF32, TEST_FLOAT_COMPLEX_KERNEL)
{
    test_kernel();
}

TEST_P(AoclfftzFusedTwiddleTransposeKernelTestF64, TEST_DOUBLE_COMPLEX_KERNEL)
{
    test_kernel();
}

// Geometries: square and non-square, edges multiples of 8 (both micro-tiles)
// and large enough that the blocked walk crosses cache-block boundaries.
static std::vector<std::tuple<FFTZ_INTP, FFTZ_INTP>> tt_dims = {
    {16, 16}, /* square */
    {16, 32}, /* non-square */
    {32, 16}, /* non-square (transposed aspect) */
    {24, 40}, /* non-square, not a multiple of the cache block */
};

// Row-stride padding: 0 (tight) and a positive pad so in_row_stride > n2 and
// out_row_stride > n1, exercising the padded strides the solver uses.
static std::vector<FFTZ_INTP> tt_pads = {0, 5};

// Directions: forward (multiply by twiddle) and backward (multiply by conj).
static std::vector<FFTZ_INT32> tt_dirs = {FORWARD_FFT_DIR, BACKWARD_FFT_DIR};

static std::vector<FFTZ_INT32> tt_isas()
{
    std::vector<FFTZ_INT32> isas;
    for (const auto &v : tt_isa_variants())
    {
        isas.push_back(v.first);
    }
    return isas;
}

INSTANTIATE_TEST_SUITE_P(
    FusedTwiddleTransposeKernelTest, AoclfftzFusedTwiddleTransposeKernelTestF32,
    ::testing::Combine(::testing::ValuesIn(tt_dims),
                       ::testing::ValuesIn(tt_pads), // in-row pad
                       ::testing::ValuesIn(tt_pads), // out-row pad
                       ::testing::ValuesIn(tt_dirs),
                       ::testing::ValuesIn(tt_isas())),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    FusedTwiddleTransposeKernelTest, AoclfftzFusedTwiddleTransposeKernelTestF64,
    ::testing::Combine(::testing::ValuesIn(tt_dims),
                       ::testing::ValuesIn(tt_pads), // in-row pad
                       ::testing::ValuesIn(tt_pads), // out-row pad
                       ::testing::ValuesIn(tt_dirs),
                       ::testing::ValuesIn(tt_isas())),
    name_generator);

