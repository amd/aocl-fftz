// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file transpose_outofplace_kernel_gtest.cpp
 *
 *  @brief File that contains the GTest based transpose kernel unit tests.
 *
 *  This file contains the entry point of the GTest. It contains tests for all
 *  the out of place transpose kernels
 *
 * @author Ashwin K. Godbole
 */

#include <string>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>
#include "test/gtest/transpose/transpose_outofplace_kernel_gtest_base.h"
#include "test/gtest/transpose/transpose_kernel_gtest_utils.h"

/**
 * @brief An utility function to return the test name based on the test_type
 *
 */
auto name_generator =
    [](const ::testing::TestParamInfo< std::tuple<std::tuple<FFTZ_INTP,
       FFTZ_INTP>, FFTZ_INTP, FFTZ_INTP, FFTZ_INT32>> &info)
{
    FFTZ_INTP rows = std::get<0>(std::get<0>(info.param));
    FFTZ_INTP cols = std::get<1>(std::get<0>(info.param));
    FFTZ_INTP in_stride = std::get<1>(info.param);
    FFTZ_INTP out_stride = std::get<2>(info.param);
    FFTZ_INT32 kernel_idx = std::get<3>(info.param);
    std::string kernel_name = transpose_kernel_names_table[kernel_idx];

    std::string test_name = std::to_string(rows) + "x" + std::to_string(cols);
    test_name += "_is" + std::to_string(in_stride);
    test_name += "_os" + std::to_string(out_stride);
    test_name += "_" + kernel_name;
    return test_name;
};

TEST_P(AoclfftzOutOfPlaceTransposeKernelTestF32, TEST_FLOAT_KERNEL)
{
    test_kernel();
}

TEST_P(AoclfftzOutOfPlaceTransposeKernelTestF64, TEST_DOUBLE_KERNEL)
{
    test_kernel();
}

TEST_P(AoclfftzOutOfPlaceTransposeKernelTestF32C, TEST_FLOAT_COMPLEX_KERNEL)
{
    test_kernel();
}

TEST_P(AoclfftzOutOfPlaceTransposeKernelTestF64C, TEST_DOUBLE_COMPLEX_KERNEL)
{
    test_kernel();
}

std::vector<std::tuple<FFTZ_INTP, FFTZ_INTP>> dims = {
    {10, 10}, {69, 69}, {360, 360}, {512, 512},
    {10, 40}, {82, 69}, {341, 3},   {16, 1024}};

std::vector<FFTZ_INTP> strides = {1, 35, 102};

// Arbitrarily (General) strided, out-of-place matrix transpose
// -----------------------------------------------------------------------------
INSTANTIATE_TEST_SUITE_P(
    OOPGeneralStrideTransposeKernelTest,
    AoclfftzOutOfPlaceTransposeKernelTestF32,
    ::testing::Combine(::testing::ValuesIn(dims),
                       ::testing::ValuesIn(strides), // in-stride
                       ::testing::ValuesIn(strides), // out-stride
                       ::testing::Values(5,          // tos_iterative
                                         6)          // tos_blocked
                       ),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    OOPGeneralStrideTransposeKernelTest,
    AoclfftzOutOfPlaceTransposeKernelTestF64,
    ::testing::Combine(::testing::ValuesIn(dims),
                       ::testing::ValuesIn(strides), // in-stride
                       ::testing::ValuesIn(strides), // out-stride
                       ::testing::Values(5,          // tos_iterative
                                         6)          // tos_blocked
                       ),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    OOPGeneralStrideTransposeKernelTest,
    AoclfftzOutOfPlaceTransposeKernelTestF32C,
    ::testing::Combine(::testing::ValuesIn(dims),
                       ::testing::ValuesIn(strides), // in-stride
                       ::testing::ValuesIn(strides), // out-stride
                       ::testing::Values(5,          // tos_iterative
                                         6)          // tos_blocked
                       ),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    OOPGeneralStrideTransposeKernelTest,
    AoclfftzOutOfPlaceTransposeKernelTestF64C,
    ::testing::Combine(::testing::ValuesIn(dims),
                       ::testing::ValuesIn(strides), // in-stride
                       ::testing::ValuesIn(strides), // out-stride
                       ::testing::Values(5,          // tos_iterative
                                         6)          // tos_blocked
                       ),
    name_generator);
