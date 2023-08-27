/**
 * Copyright (C) 2023, Advanced Micro Devices. All rights reserved.
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

/** @file kernel_gtest.cpp
 *
 *  @brief File that contains the GTest based kernel unit tests.
 *
 *  This file contains the entry point of the GTest. It contains the kernel
 *  table initializations for kernel unit tests.
 *
 *  @author Srirammaswamy Srinivasan
 *  @author Prasandh Sankarankutty
 */

#include <gtest/gtest.h>
#include "kernel_gtest_base.h"

// FLOAT test params consisting of radix, kernel-type and test-type
aoclfftz_kernel_test_params_t param_float_kernels[] = {
    {2, aocl_fftz_kernel_type::STANDARD_C, aoclfftz_kernel_test_type::ALL},
    {3, aocl_fftz_kernel_type::STANDARD_C, aoclfftz_kernel_test_type::ALL},
    {4, aocl_fftz_kernel_type::STANDARD_C, aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::STANDARD_C, aoclfftz_kernel_test_type::ALL},
    {6, aocl_fftz_kernel_type::STANDARD_C, aoclfftz_kernel_test_type::ALL},
    {7, aocl_fftz_kernel_type::STANDARD_C, aoclfftz_kernel_test_type::ALL},
    {8, aocl_fftz_kernel_type::STANDARD_C, aoclfftz_kernel_test_type::ALL},
    {9, aocl_fftz_kernel_type::STANDARD_C, aoclfftz_kernel_test_type::ALL},
    {10, aocl_fftz_kernel_type::STANDARD_C, aoclfftz_kernel_test_type::ALL},
    {11, aocl_fftz_kernel_type::STANDARD_C, aoclfftz_kernel_test_type::ALL},
    {12, aocl_fftz_kernel_type::STANDARD_C, aoclfftz_kernel_test_type::ALL},
    {13, aocl_fftz_kernel_type::STANDARD_C, aoclfftz_kernel_test_type::ALL},
    {14, aocl_fftz_kernel_type::STANDARD_C, aoclfftz_kernel_test_type::ALL},
    {15, aocl_fftz_kernel_type::STANDARD_C, aoclfftz_kernel_test_type::ALL},
    {16, aocl_fftz_kernel_type::STANDARD_C, aoclfftz_kernel_test_type::ALL},
    {2, aocl_fftz_kernel_type::PERMUTED_C, aoclfftz_kernel_test_type::ALL},
    {3, aocl_fftz_kernel_type::PERMUTED_C, aoclfftz_kernel_test_type::ALL},
    {4, aocl_fftz_kernel_type::PERMUTED_C, aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::PERMUTED_C, aoclfftz_kernel_test_type::ALL},
    {6, aocl_fftz_kernel_type::PERMUTED_C, aoclfftz_kernel_test_type::ALL},
    {7, aocl_fftz_kernel_type::PERMUTED_C, aoclfftz_kernel_test_type::ALL},
    {8, aocl_fftz_kernel_type::PERMUTED_C, aoclfftz_kernel_test_type::ALL},
    {9, aocl_fftz_kernel_type::PERMUTED_C, aoclfftz_kernel_test_type::ALL},
    {10, aocl_fftz_kernel_type::PERMUTED_C, aoclfftz_kernel_test_type::ALL},
    {11, aocl_fftz_kernel_type::PERMUTED_C, aoclfftz_kernel_test_type::ALL},
    {12, aocl_fftz_kernel_type::PERMUTED_C, aoclfftz_kernel_test_type::ALL},
    {13, aocl_fftz_kernel_type::PERMUTED_C, aoclfftz_kernel_test_type::ALL},
    {14, aocl_fftz_kernel_type::PERMUTED_C, aoclfftz_kernel_test_type::ALL},
    {15, aocl_fftz_kernel_type::PERMUTED_C, aoclfftz_kernel_test_type::ALL},
    {16, aocl_fftz_kernel_type::PERMUTED_C, aoclfftz_kernel_test_type::ALL}};

// DOUBLE test params consisting of radix, kernel-type and test-type
aoclfftz_kernel_test_params_t param_double_kernels[] = {
    {2, aocl_fftz_kernel_type::STANDARD_C, aoclfftz_kernel_test_type::ALL},
    {3, aocl_fftz_kernel_type::STANDARD_C, aoclfftz_kernel_test_type::ALL},
    {4, aocl_fftz_kernel_type::STANDARD_C, aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::STANDARD_C, aoclfftz_kernel_test_type::ALL},
    {6, aocl_fftz_kernel_type::STANDARD_C, aoclfftz_kernel_test_type::ALL},
    {7, aocl_fftz_kernel_type::STANDARD_C, aoclfftz_kernel_test_type::ALL},
    {8, aocl_fftz_kernel_type::STANDARD_C, aoclfftz_kernel_test_type::ALL},
    {9, aocl_fftz_kernel_type::STANDARD_C, aoclfftz_kernel_test_type::ALL},
    {10, aocl_fftz_kernel_type::STANDARD_C, aoclfftz_kernel_test_type::ALL},
    {11, aocl_fftz_kernel_type::STANDARD_C, aoclfftz_kernel_test_type::ALL},
    {12, aocl_fftz_kernel_type::STANDARD_C, aoclfftz_kernel_test_type::ALL},
    {13, aocl_fftz_kernel_type::STANDARD_C, aoclfftz_kernel_test_type::ALL},
    {14, aocl_fftz_kernel_type::STANDARD_C, aoclfftz_kernel_test_type::ALL},
    {15, aocl_fftz_kernel_type::STANDARD_C, aoclfftz_kernel_test_type::ALL},
    {16, aocl_fftz_kernel_type::STANDARD_C, aoclfftz_kernel_test_type::ALL},
    {2, aocl_fftz_kernel_type::PERMUTED_C, aoclfftz_kernel_test_type::ALL},
    {3, aocl_fftz_kernel_type::PERMUTED_C, aoclfftz_kernel_test_type::ALL},
    {4, aocl_fftz_kernel_type::PERMUTED_C, aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::PERMUTED_C, aoclfftz_kernel_test_type::ALL},
    {6, aocl_fftz_kernel_type::PERMUTED_C, aoclfftz_kernel_test_type::ALL},
    {7, aocl_fftz_kernel_type::PERMUTED_C, aoclfftz_kernel_test_type::ALL},
    {8, aocl_fftz_kernel_type::PERMUTED_C, aoclfftz_kernel_test_type::ALL},
    {9, aocl_fftz_kernel_type::PERMUTED_C, aoclfftz_kernel_test_type::ALL},
    {10, aocl_fftz_kernel_type::PERMUTED_C, aoclfftz_kernel_test_type::ALL},
    {11, aocl_fftz_kernel_type::PERMUTED_C, aoclfftz_kernel_test_type::ALL},
    {12, aocl_fftz_kernel_type::PERMUTED_C, aoclfftz_kernel_test_type::ALL},
    {13, aocl_fftz_kernel_type::PERMUTED_C, aoclfftz_kernel_test_type::ALL},
    {14, aocl_fftz_kernel_type::PERMUTED_C, aoclfftz_kernel_test_type::ALL},
    {15, aocl_fftz_kernel_type::PERMUTED_C, aoclfftz_kernel_test_type::ALL},
    {16, aocl_fftz_kernel_type::PERMUTED_C, aoclfftz_kernel_test_type::ALL}};

// stride values as in-stride, out-stride pairs
std::vector<std::pair<INTP, INTP>> strides = {{1, 1}, {2, 9},  {7, 3},
                                              {4, 4}, {11, 1}, {1, 6}};

TEST_P(AoclfftzKernelTestFloat, TEST_FLOAT_KERNEL)
{
    run_kernel_test(1); // run test with normal values and with 1 set
}

TEST_P(AoclfftzKernelTestDouble, TEST_DOUBLE_KERNEL)
{
    run_kernel_test(1); // run test with normal values and with 1 set
}

TEST_P(AoclfftzKernelTestFloat, TEST_FLOAT_KERNEL_BATCHED)
{
    run_kernel_test(0); // run test with normal values and with multiple sets
}

TEST_P(AoclfftzKernelTestDouble, TEST_DOUBLE_KERNEL_BATCHED)
{
    run_kernel_test(0); // run test with normal values and with multiple sets
}

TEST_P(AoclfftzKernelTestFloat, TEST_FLOAT_KERNEL_SPECIAL)
{
    run_kernel_test(1, true); // run test with normal and special values
}

TEST_P(AoclfftzKernelTestDouble, TEST_DOUBLE_KERNEL_SPECIAL)
{
    run_kernel_test(1, true); // run test with normal and special values
}

/**
 * @brief An utility function to return the test name based on the test_type
 *
 */
auto name_generator =
    [](const ::testing::TestParamInfo<
        std::tuple<aoclfftz_kernel_test_params_t, std::pair<INTP, INTP>, UINT8>>
           &info) {
        auto param = std::get<0>(info.param);
        INTP istride = std::get<1>(info.param).first;
        INTP ostride = std::get<1>(info.param).second;
        UINT8 is_bwd = std::get<2>(info.param);
        UINT32 radix = std::get<0>(param);
        UINT8 kernel_type = std::get<1>(param);
        UINT8 test_type = std::get<2>(param);

        std::string test_name = std::to_string(radix);
        if (is_bwd)
        {
            test_name += "_BWD";
        }
        else
        {
            test_name += "_FWD";
        }
        test_name += "_IS" + std::to_string(istride);
        test_name += "_OS" + std::to_string(ostride);
        test_name += get_kernel_type_as_string(kernel_type);
        if (test_type == aoclfftz_kernel_test_type::ALL)
        {
            test_name += "_ALL";
        }
        else
        {
            if (test_type & aoclfftz_kernel_test_type::LINEARITY)
                test_name += "_LINEARITY";
            if (test_type & aoclfftz_kernel_test_type::TRANSFORMATION)
                test_name += "_TRANSFORMATION";
            if (test_type & aoclfftz_kernel_test_type::TIMESHIFT)
                test_name += "_TIMESHIFT";
        }
        return test_name;
    };

INSTANTIATE_TEST_SUITE_P(
    KernelTest, AoclfftzKernelTestFloat,
    ::testing::Combine(::testing::ValuesIn(param_float_kernels),
                       ::testing::ValuesIn(strides),
                       ::testing::Values(0, 1)), // 0 -> FWD, 1 -> BWD
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    KernelTest, AoclfftzKernelTestDouble,
    ::testing::Combine(::testing::ValuesIn(param_double_kernels),
                       ::testing::ValuesIn(strides),
                       ::testing::Values(0, 1)), // 0 -> FWD, 1 -> BWD
    name_generator);