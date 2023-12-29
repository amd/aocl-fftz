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
aoclfftz_kernel_test_params_t param_float_kernels[] =
{
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
    {16, aocl_fftz_kernel_type::PERMUTED_C, aoclfftz_kernel_test_type::ALL}
};

// DOUBLE test params consisting of radix, kernel-type and test-type
aoclfftz_kernel_test_params_t param_double_kernels[] =
{
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
    {16, aocl_fftz_kernel_type::PERMUTED_C, aoclfftz_kernel_test_type::ALL}
};

aoclfftz_kernel_test_params_t param_float_avx128_kernels[] =
{
    {2, aocl_fftz_kernel_type::STANDARD_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {3, aocl_fftz_kernel_type::STANDARD_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {4, aocl_fftz_kernel_type::STANDARD_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::STANDARD_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {6, aocl_fftz_kernel_type::STANDARD_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {7, aocl_fftz_kernel_type::STANDARD_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {8, aocl_fftz_kernel_type::STANDARD_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {9, aocl_fftz_kernel_type::STANDARD_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {10, aocl_fftz_kernel_type::STANDARD_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {11, aocl_fftz_kernel_type::STANDARD_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {12, aocl_fftz_kernel_type::STANDARD_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {14, aocl_fftz_kernel_type::STANDARD_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {15, aocl_fftz_kernel_type::STANDARD_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {16, aocl_fftz_kernel_type::STANDARD_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {2, aocl_fftz_kernel_type::PERMUTED_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {3, aocl_fftz_kernel_type::PERMUTED_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {4, aocl_fftz_kernel_type::PERMUTED_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::PERMUTED_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {6, aocl_fftz_kernel_type::PERMUTED_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {7, aocl_fftz_kernel_type::PERMUTED_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {8, aocl_fftz_kernel_type::PERMUTED_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {9, aocl_fftz_kernel_type::PERMUTED_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {10, aocl_fftz_kernel_type::PERMUTED_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {11, aocl_fftz_kernel_type::PERMUTED_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {12, aocl_fftz_kernel_type::PERMUTED_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {14, aocl_fftz_kernel_type::PERMUTED_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {15, aocl_fftz_kernel_type::PERMUTED_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {16, aocl_fftz_kernel_type::PERMUTED_AVX128,
    aoclfftz_kernel_test_type::ALL}
};

aoclfftz_kernel_test_params_t param_double_avx128_kernels[] =
{
    {2, aocl_fftz_kernel_type::STANDARD_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {3, aocl_fftz_kernel_type::STANDARD_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {4, aocl_fftz_kernel_type::STANDARD_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::STANDARD_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {6, aocl_fftz_kernel_type::STANDARD_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {7, aocl_fftz_kernel_type::STANDARD_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {8, aocl_fftz_kernel_type::STANDARD_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {9, aocl_fftz_kernel_type::STANDARD_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {10, aocl_fftz_kernel_type::STANDARD_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {11, aocl_fftz_kernel_type::STANDARD_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {12, aocl_fftz_kernel_type::STANDARD_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {14, aocl_fftz_kernel_type::STANDARD_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {15, aocl_fftz_kernel_type::STANDARD_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {16, aocl_fftz_kernel_type::STANDARD_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {2, aocl_fftz_kernel_type::PERMUTED_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {3, aocl_fftz_kernel_type::PERMUTED_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {4, aocl_fftz_kernel_type::PERMUTED_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::PERMUTED_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {6, aocl_fftz_kernel_type::PERMUTED_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {7, aocl_fftz_kernel_type::PERMUTED_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {8, aocl_fftz_kernel_type::PERMUTED_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {9, aocl_fftz_kernel_type::PERMUTED_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {10, aocl_fftz_kernel_type::PERMUTED_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {11, aocl_fftz_kernel_type::PERMUTED_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {12, aocl_fftz_kernel_type::PERMUTED_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {14, aocl_fftz_kernel_type::PERMUTED_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {15, aocl_fftz_kernel_type::PERMUTED_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {16, aocl_fftz_kernel_type::PERMUTED_AVX128,
    aoclfftz_kernel_test_type::ALL}
};

//AVX256 kernels
aoclfftz_kernel_test_params_t param_float_avx256_kernels[] =
{
    {2, aocl_fftz_kernel_type::STANDARD_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {3, aocl_fftz_kernel_type::STANDARD_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {4, aocl_fftz_kernel_type::STANDARD_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::STANDARD_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {6, aocl_fftz_kernel_type::STANDARD_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {7, aocl_fftz_kernel_type::STANDARD_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {8, aocl_fftz_kernel_type::STANDARD_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {9, aocl_fftz_kernel_type::STANDARD_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {10, aocl_fftz_kernel_type::STANDARD_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {11, aocl_fftz_kernel_type::STANDARD_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {12, aocl_fftz_kernel_type::STANDARD_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {14, aocl_fftz_kernel_type::STANDARD_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {15, aocl_fftz_kernel_type::STANDARD_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {2, aocl_fftz_kernel_type::PERMUTED_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {3, aocl_fftz_kernel_type::PERMUTED_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {4, aocl_fftz_kernel_type::PERMUTED_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::PERMUTED_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {6, aocl_fftz_kernel_type::PERMUTED_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {7, aocl_fftz_kernel_type::PERMUTED_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {8, aocl_fftz_kernel_type::PERMUTED_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {9, aocl_fftz_kernel_type::PERMUTED_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {10, aocl_fftz_kernel_type::PERMUTED_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {11, aocl_fftz_kernel_type::PERMUTED_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {12, aocl_fftz_kernel_type::PERMUTED_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {14, aocl_fftz_kernel_type::PERMUTED_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {15, aocl_fftz_kernel_type::PERMUTED_AVX256,
     aoclfftz_kernel_test_type::ALL}
};

aoclfftz_kernel_test_params_t param_double_avx256_kernels[] =
{
    {2, aocl_fftz_kernel_type::STANDARD_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {3, aocl_fftz_kernel_type::STANDARD_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {4, aocl_fftz_kernel_type::STANDARD_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::STANDARD_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {6, aocl_fftz_kernel_type::STANDARD_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {7, aocl_fftz_kernel_type::STANDARD_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {8, aocl_fftz_kernel_type::STANDARD_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {9, aocl_fftz_kernel_type::STANDARD_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {10, aocl_fftz_kernel_type::STANDARD_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {11, aocl_fftz_kernel_type::STANDARD_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {12, aocl_fftz_kernel_type::STANDARD_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {14, aocl_fftz_kernel_type::STANDARD_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {15, aocl_fftz_kernel_type::STANDARD_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {2, aocl_fftz_kernel_type::PERMUTED_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {3, aocl_fftz_kernel_type::PERMUTED_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {4, aocl_fftz_kernel_type::PERMUTED_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::PERMUTED_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {6, aocl_fftz_kernel_type::PERMUTED_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {7, aocl_fftz_kernel_type::PERMUTED_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {8, aocl_fftz_kernel_type::PERMUTED_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {9, aocl_fftz_kernel_type::PERMUTED_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {10, aocl_fftz_kernel_type::PERMUTED_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {11, aocl_fftz_kernel_type::PERMUTED_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {12, aocl_fftz_kernel_type::PERMUTED_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {14, aocl_fftz_kernel_type::PERMUTED_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {15, aocl_fftz_kernel_type::PERMUTED_AVX256,
     aoclfftz_kernel_test_type::ALL}
};

// IO params as {in-stride, out-stride , batch size, dir of FFT(0->FWD/1-> BWD)}
// Batch size fixed as 1-7 to cover all the tail case in AVX128 & AVX256 kernels
std::vector<std::tuple<INTP, INTP, INTP, UINT8>> io_params = {{1,  1, 1, 0},
                                                              {2,  9, 2, 1},
                                                              {7,  3, 3, 0},
                                                              {4,  4, 4, 1},
                                                              {11, 1, 5, 0},
                                                              {1,  6, 6, 1},
                                                              {10, 5, 7, 0}};

TEST_P(AoclfftzKernelTestFloat, TEST_FLOAT_KERNEL)
{
    run_kernel_test(); // run test with normal values
}

TEST_P(AoclfftzKernelTestDouble, TEST_DOUBLE_KERNEL)
{
    run_kernel_test(); // run test with normal values
}

TEST_P(AoclfftzKernelTestFloat, TEST_FLOAT_KERNEL_SPECIAL)
{
    run_kernel_test(true); // run test with normal and special values
}

TEST_P(AoclfftzKernelTestDouble, TEST_DOUBLE_KERNEL_SPECIAL)
{
    run_kernel_test(true); // run test with normal and special values
}

/**
 * @brief An utility function to return the test name based on the test_type
 *
 */
auto name_generator =
    [](const ::testing::TestParamInfo<
        std::tuple<aoclfftz_kernel_test_params_t,
        std::tuple<INTP, INTP, INTP, UINT8>>> &info)
    {
        auto param = std::get<0>(info.param);
        auto io_param = std::get<1>(info.param);
        INTP istride  = std::get<0>(io_param);
        INTP ostride  = std::get<1>(io_param);
        INTP batch_sz = std::get<2>(io_param);
        UINT8 is_bwd  = std::get<3>(io_param);
        UINT32 radix  = std::get<0>(param);
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
        test_name += "_IS_" + std::to_string(istride);
        test_name += "_OS_" + std::to_string(ostride);
        test_name += "_BATCH_" + std::to_string(batch_sz);
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
                       ::testing::ValuesIn(io_params)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    KernelTest, AoclfftzKernelTestDouble,
    ::testing::Combine(::testing::ValuesIn(param_double_kernels),
                       ::testing::ValuesIn(io_params)),
    name_generator);

#ifdef ENABLE_AVX128
//AVX128 TEST SUITE
INSTANTIATE_TEST_SUITE_P(
    AVXKernelTest, AoclfftzKernelTestFloat,
    ::testing::Combine(::testing::ValuesIn(param_float_avx128_kernels),
                       ::testing::ValuesIn(io_params)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    AVXKernelTest, AoclfftzKernelTestDouble,
    ::testing::Combine(::testing::ValuesIn(param_double_avx128_kernels),
                       ::testing::ValuesIn(io_params)),
    name_generator);
#endif
#ifdef ENABLE_AVX256
//AVX256 TEST SUITE
INSTANTIATE_TEST_SUITE_P(
    AVX256KernelTest, AoclfftzKernelTestFloat,
    ::testing::Combine(::testing::ValuesIn(param_float_avx256_kernels),
                       ::testing::ValuesIn(io_params)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    AVX256KernelTest, AoclfftzKernelTestDouble,
    ::testing::Combine(::testing::ValuesIn(param_double_avx256_kernels),
                       ::testing::ValuesIn(io_params)),
    name_generator);
#endif
