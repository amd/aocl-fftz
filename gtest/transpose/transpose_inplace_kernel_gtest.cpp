/**
 * Copyright (C) 2024, Advanced Micro Devices. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/** @file transpose_inplace_kernel_gtest.cpp
 *
 *  @brief File that contains the GTest based transpose kernel unit tests.
 *
 *  This file contains the entry point of the GTest. It contains tests for all
 *  the inplace transpose kernels
 *
 * @author Ashwin K. Godbole
 */

#include <string>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>
#include "gtest/transpose/transpose_inplace_kernel_gtest_base.h"
#include "gtest/transpose/transpose_kernel_gtest_utils.h"

/**
 * @brief An utility function to return the test name based on the test_type
 *
 */
auto name_generator =
    [](const ::testing::TestParamInfo<
        std::tuple<std::tuple<INTP, INTP>, INTP, INT32>> &info)
{
    INTP rows = std::get<0>(std::get<0>(info.param));
    INTP cols = std::get<1>(std::get<0>(info.param));
    INTP stride = std::get<1>(info.param);
    INT32 kernel_idx = std::get<2>(info.param);
    std::string kernel_name = transpose_kernel_names_table[kernel_idx];

    std::string test_name = std::to_string(rows) + "x" + std::to_string(cols);
    test_name += "_s" + std::to_string(stride);
    test_name += "_" + kernel_name;
    return test_name;
};

TEST_P(AoclfftzInplaceTransposeKernelTestFloat, TESTFloat_KERNEL)
{
    test_kernel();
}

TEST_P(AoclfftzInplaceTransposeKernelTestDouble, TESTDouble_KERNEL)
{
    test_kernel();
}

TEST_P(AoclfftzInplaceTransposeKernelTestFloatComplex,
       TESTFloat_COMPLEX_KERNEL)
{
    test_kernel();
}

TEST_P(AoclfftzInplaceTransposeKernelTestDoubleComplex,
       TESTDouble_COMPLEX_KERNEL)
{
    test_kernel();
}

std::vector<std::tuple<INTP, INTP>> dims = {
    {1, 1}, {2, 2}, {13, 13}, {56, 56}, {64, 64}};

// Unit strided, square, inplace matrix transpose
// -----------------------------------------------------------------------------
INSTANTIATE_TEST_SUITE_P(
    IpUnitStrideTransposeKernelTest, AoclfftzInplaceTransposeKernelTestFloat,
    ::testing::Combine(::testing::ValuesIn(dims),
                       ::testing::Values(1), // stride
                       ::testing::Values(0)  // kernel :: tiq_iterative
                       ),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    IpUnitStrideTransposeKernelTest, AoclfftzInplaceTransposeKernelTestDouble,
    ::testing::Combine(::testing::ValuesIn(dims),
                       ::testing::Values(1), // stride
                       ::testing::Values(0)  // kernel :: tiq_iterative
                       ),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    IpUnitStrideTransposeKernelTest,
    AoclfftzInplaceTransposeKernelTestFloatComplex,
    ::testing::Combine(::testing::ValuesIn(dims),
                       ::testing::Values(1), // stride
                       ::testing::Values(0)  // kernel :: tiq_iterative
                       ),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    IpUnitStrideTransposeKernelTest,
    AoclfftzInplaceTransposeKernelTestDoubleComplex,
    ::testing::Combine(::testing::ValuesIn(dims),
                       ::testing::Values(1), // stride
                       ::testing::Values(0)  // kernel :: tiq_iterative
                       ),
    name_generator);

std::vector<INTP> strides = {1, 3, 8, 35};

// Arbitrarily (General) strided, square, inplace matrix transpose
// -----------------------------------------------------------------------------
INSTANTIATE_TEST_SUITE_P(
    IpGeneralStrideTransposeKernelTest,
    AoclfftzInplaceTransposeKernelTestFloat,
    ::testing::Combine(::testing::ValuesIn(dims),
                       ::testing::ValuesIn(strides), // stride
                       ::testing::Values(1)          // kernel :: tisq_iterative
                       ),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    IpGeneralStrideTransposeKernelTest,
    AoclfftzInplaceTransposeKernelTestDouble,
    ::testing::Combine(::testing::ValuesIn(dims),
                       ::testing::ValuesIn(strides), // stride
                       ::testing::Values(1)          // kernel :: tisq_iterative
                       ),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    IpGeneralStrideTransposeKernelTest,
    AoclfftzInplaceTransposeKernelTestFloatComplex,
    ::testing::Combine(::testing::ValuesIn(dims),
                       ::testing::ValuesIn(strides), // stride
                       ::testing::Values(1)          // kernel :: tisq_iterative
                       ),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    IpGeneralStrideTransposeKernelTest,
    AoclfftzInplaceTransposeKernelTestDoubleComplex,
    ::testing::Combine(::testing::ValuesIn(dims),
                       ::testing::ValuesIn(strides), // stride
                       ::testing::Values(1)          // kernel :: tisq_iterative
                       ),
    name_generator);
