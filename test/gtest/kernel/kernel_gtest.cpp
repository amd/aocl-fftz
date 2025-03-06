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
#include "complex_kernel_gtest.h"
#include "real_kernel_gtest.h"

// C2C - C Kernels - Float
aoclfftz_kernel_test_params_t param_float_c2c_c_kernels[] =
{
    {2, aocl_fftz_kernel_type::STANDARD_C2C_C, aoclfftz_kernel_test_type::ALL},
    {3, aocl_fftz_kernel_type::STANDARD_C2C_C, aoclfftz_kernel_test_type::ALL},
    {4, aocl_fftz_kernel_type::STANDARD_C2C_C, aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::STANDARD_C2C_C, aoclfftz_kernel_test_type::ALL},
    {6, aocl_fftz_kernel_type::STANDARD_C2C_C, aoclfftz_kernel_test_type::ALL},
    {7, aocl_fftz_kernel_type::STANDARD_C2C_C, aoclfftz_kernel_test_type::ALL},
    {8, aocl_fftz_kernel_type::STANDARD_C2C_C, aoclfftz_kernel_test_type::ALL},
    {9, aocl_fftz_kernel_type::STANDARD_C2C_C, aoclfftz_kernel_test_type::ALL},
    {10, aocl_fftz_kernel_type::STANDARD_C2C_C, aoclfftz_kernel_test_type::ALL},
    {11, aocl_fftz_kernel_type::STANDARD_C2C_C, aoclfftz_kernel_test_type::ALL},
    {12, aocl_fftz_kernel_type::STANDARD_C2C_C, aoclfftz_kernel_test_type::ALL},
    {13, aocl_fftz_kernel_type::STANDARD_C2C_C, aoclfftz_kernel_test_type::ALL},
    {14, aocl_fftz_kernel_type::STANDARD_C2C_C, aoclfftz_kernel_test_type::ALL},
    {15, aocl_fftz_kernel_type::STANDARD_C2C_C, aoclfftz_kernel_test_type::ALL},
    {16, aocl_fftz_kernel_type::STANDARD_C2C_C, aoclfftz_kernel_test_type::ALL},
    {2, aocl_fftz_kernel_type::PERMUTED_C2C_C, aoclfftz_kernel_test_type::ALL},
    {3, aocl_fftz_kernel_type::PERMUTED_C2C_C, aoclfftz_kernel_test_type::ALL},
    {4, aocl_fftz_kernel_type::PERMUTED_C2C_C, aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::PERMUTED_C2C_C, aoclfftz_kernel_test_type::ALL},
    {6, aocl_fftz_kernel_type::PERMUTED_C2C_C, aoclfftz_kernel_test_type::ALL},
    {7, aocl_fftz_kernel_type::PERMUTED_C2C_C, aoclfftz_kernel_test_type::ALL},
    {8, aocl_fftz_kernel_type::PERMUTED_C2C_C, aoclfftz_kernel_test_type::ALL},
    {9, aocl_fftz_kernel_type::PERMUTED_C2C_C, aoclfftz_kernel_test_type::ALL},
    {10, aocl_fftz_kernel_type::PERMUTED_C2C_C, aoclfftz_kernel_test_type::ALL},
    {11, aocl_fftz_kernel_type::PERMUTED_C2C_C, aoclfftz_kernel_test_type::ALL},
    {12, aocl_fftz_kernel_type::PERMUTED_C2C_C, aoclfftz_kernel_test_type::ALL},
    {13, aocl_fftz_kernel_type::PERMUTED_C2C_C, aoclfftz_kernel_test_type::ALL},
    {14, aocl_fftz_kernel_type::PERMUTED_C2C_C, aoclfftz_kernel_test_type::ALL},
    {15, aocl_fftz_kernel_type::PERMUTED_C2C_C, aoclfftz_kernel_test_type::ALL},
    {16, aocl_fftz_kernel_type::PERMUTED_C2C_C, aoclfftz_kernel_test_type::ALL}
};

// C2C - C Kernels - Double
aoclfftz_kernel_test_params_t param_double_c2c_c_kernels[] =
{
    {2, aocl_fftz_kernel_type::STANDARD_C2C_C, aoclfftz_kernel_test_type::ALL},
    {3, aocl_fftz_kernel_type::STANDARD_C2C_C, aoclfftz_kernel_test_type::ALL},
    {4, aocl_fftz_kernel_type::STANDARD_C2C_C, aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::STANDARD_C2C_C, aoclfftz_kernel_test_type::ALL},
    {6, aocl_fftz_kernel_type::STANDARD_C2C_C, aoclfftz_kernel_test_type::ALL},
    {7, aocl_fftz_kernel_type::STANDARD_C2C_C, aoclfftz_kernel_test_type::ALL},
    {8, aocl_fftz_kernel_type::STANDARD_C2C_C, aoclfftz_kernel_test_type::ALL},
    {9, aocl_fftz_kernel_type::STANDARD_C2C_C, aoclfftz_kernel_test_type::ALL},
    {10, aocl_fftz_kernel_type::STANDARD_C2C_C, aoclfftz_kernel_test_type::ALL},
    {11, aocl_fftz_kernel_type::STANDARD_C2C_C, aoclfftz_kernel_test_type::ALL},
    {12, aocl_fftz_kernel_type::STANDARD_C2C_C, aoclfftz_kernel_test_type::ALL},
    {13, aocl_fftz_kernel_type::STANDARD_C2C_C, aoclfftz_kernel_test_type::ALL},
    {14, aocl_fftz_kernel_type::STANDARD_C2C_C, aoclfftz_kernel_test_type::ALL},
    {15, aocl_fftz_kernel_type::STANDARD_C2C_C, aoclfftz_kernel_test_type::ALL},
    {16, aocl_fftz_kernel_type::STANDARD_C2C_C, aoclfftz_kernel_test_type::ALL},
    {2, aocl_fftz_kernel_type::PERMUTED_C2C_C, aoclfftz_kernel_test_type::ALL},
    {3, aocl_fftz_kernel_type::PERMUTED_C2C_C, aoclfftz_kernel_test_type::ALL},
    {4, aocl_fftz_kernel_type::PERMUTED_C2C_C, aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::PERMUTED_C2C_C, aoclfftz_kernel_test_type::ALL},
    {6, aocl_fftz_kernel_type::PERMUTED_C2C_C, aoclfftz_kernel_test_type::ALL},
    {7, aocl_fftz_kernel_type::PERMUTED_C2C_C, aoclfftz_kernel_test_type::ALL},
    {8, aocl_fftz_kernel_type::PERMUTED_C2C_C, aoclfftz_kernel_test_type::ALL},
    {9, aocl_fftz_kernel_type::PERMUTED_C2C_C, aoclfftz_kernel_test_type::ALL},
    {10, aocl_fftz_kernel_type::PERMUTED_C2C_C, aoclfftz_kernel_test_type::ALL},
    {11, aocl_fftz_kernel_type::PERMUTED_C2C_C, aoclfftz_kernel_test_type::ALL},
    {12, aocl_fftz_kernel_type::PERMUTED_C2C_C, aoclfftz_kernel_test_type::ALL},
    {13, aocl_fftz_kernel_type::PERMUTED_C2C_C, aoclfftz_kernel_test_type::ALL},
    {14, aocl_fftz_kernel_type::PERMUTED_C2C_C, aoclfftz_kernel_test_type::ALL},
    {15, aocl_fftz_kernel_type::PERMUTED_C2C_C, aoclfftz_kernel_test_type::ALL},
    {16, aocl_fftz_kernel_type::PERMUTED_C2C_C, aoclfftz_kernel_test_type::ALL}
};

// C2C - AVX128 Kernels - Float
aoclfftz_kernel_test_params_t param_float_c2c_avx128_kernels[] =
{
    {2, aocl_fftz_kernel_type::STANDARD_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {3, aocl_fftz_kernel_type::STANDARD_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {4, aocl_fftz_kernel_type::STANDARD_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::STANDARD_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {6, aocl_fftz_kernel_type::STANDARD_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {7, aocl_fftz_kernel_type::STANDARD_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {8, aocl_fftz_kernel_type::STANDARD_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {9, aocl_fftz_kernel_type::STANDARD_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {10, aocl_fftz_kernel_type::STANDARD_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {11, aocl_fftz_kernel_type::STANDARD_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {12, aocl_fftz_kernel_type::STANDARD_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {13, aocl_fftz_kernel_type::STANDARD_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {14, aocl_fftz_kernel_type::STANDARD_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {15, aocl_fftz_kernel_type::STANDARD_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {16, aocl_fftz_kernel_type::STANDARD_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {2, aocl_fftz_kernel_type::PERMUTED_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {3, aocl_fftz_kernel_type::PERMUTED_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {4, aocl_fftz_kernel_type::PERMUTED_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::PERMUTED_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {6, aocl_fftz_kernel_type::PERMUTED_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {7, aocl_fftz_kernel_type::PERMUTED_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {8, aocl_fftz_kernel_type::PERMUTED_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {9, aocl_fftz_kernel_type::PERMUTED_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {10, aocl_fftz_kernel_type::PERMUTED_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {11, aocl_fftz_kernel_type::PERMUTED_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {12, aocl_fftz_kernel_type::PERMUTED_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {13, aocl_fftz_kernel_type::PERMUTED_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {14, aocl_fftz_kernel_type::PERMUTED_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {15, aocl_fftz_kernel_type::PERMUTED_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {16, aocl_fftz_kernel_type::PERMUTED_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL}
};

// C2C - AVX128 Kernels - Double
aoclfftz_kernel_test_params_t param_double_c2c_avx128_kernels[] =
{
    {2, aocl_fftz_kernel_type::STANDARD_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {3, aocl_fftz_kernel_type::STANDARD_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {4, aocl_fftz_kernel_type::STANDARD_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::STANDARD_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {6, aocl_fftz_kernel_type::STANDARD_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {7, aocl_fftz_kernel_type::STANDARD_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {8, aocl_fftz_kernel_type::STANDARD_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {9, aocl_fftz_kernel_type::STANDARD_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {10, aocl_fftz_kernel_type::STANDARD_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {11, aocl_fftz_kernel_type::STANDARD_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {12, aocl_fftz_kernel_type::STANDARD_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {13, aocl_fftz_kernel_type::STANDARD_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {14, aocl_fftz_kernel_type::STANDARD_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {15, aocl_fftz_kernel_type::STANDARD_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {16, aocl_fftz_kernel_type::STANDARD_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {2, aocl_fftz_kernel_type::PERMUTED_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {3, aocl_fftz_kernel_type::PERMUTED_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {4, aocl_fftz_kernel_type::PERMUTED_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::PERMUTED_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {6, aocl_fftz_kernel_type::PERMUTED_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {7, aocl_fftz_kernel_type::PERMUTED_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {8, aocl_fftz_kernel_type::PERMUTED_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {9, aocl_fftz_kernel_type::PERMUTED_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {10, aocl_fftz_kernel_type::PERMUTED_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {11, aocl_fftz_kernel_type::PERMUTED_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {12, aocl_fftz_kernel_type::PERMUTED_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {13, aocl_fftz_kernel_type::PERMUTED_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {14, aocl_fftz_kernel_type::PERMUTED_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {15, aocl_fftz_kernel_type::PERMUTED_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL},
    {16, aocl_fftz_kernel_type::PERMUTED_C2C_AVX128,
    aoclfftz_kernel_test_type::ALL}
};

// C2C - AVX256 Kernels - Float
aoclfftz_kernel_test_params_t param_float_c2c_avx256_kernels[] =
{
    {2, aocl_fftz_kernel_type::STANDARD_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {3, aocl_fftz_kernel_type::STANDARD_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {4, aocl_fftz_kernel_type::STANDARD_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::STANDARD_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {6, aocl_fftz_kernel_type::STANDARD_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {7, aocl_fftz_kernel_type::STANDARD_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {8, aocl_fftz_kernel_type::STANDARD_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {9, aocl_fftz_kernel_type::STANDARD_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {10, aocl_fftz_kernel_type::STANDARD_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {11, aocl_fftz_kernel_type::STANDARD_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {12, aocl_fftz_kernel_type::STANDARD_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {13, aocl_fftz_kernel_type::STANDARD_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {14, aocl_fftz_kernel_type::STANDARD_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {15, aocl_fftz_kernel_type::STANDARD_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {16, aocl_fftz_kernel_type::STANDARD_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {2, aocl_fftz_kernel_type::PERMUTED_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {3, aocl_fftz_kernel_type::PERMUTED_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {4, aocl_fftz_kernel_type::PERMUTED_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::PERMUTED_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {6, aocl_fftz_kernel_type::PERMUTED_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {7, aocl_fftz_kernel_type::PERMUTED_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {8, aocl_fftz_kernel_type::PERMUTED_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {9, aocl_fftz_kernel_type::PERMUTED_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {10, aocl_fftz_kernel_type::PERMUTED_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {11, aocl_fftz_kernel_type::PERMUTED_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {12, aocl_fftz_kernel_type::PERMUTED_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {13, aocl_fftz_kernel_type::PERMUTED_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {14, aocl_fftz_kernel_type::PERMUTED_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {15, aocl_fftz_kernel_type::PERMUTED_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {16, aocl_fftz_kernel_type::PERMUTED_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL}
};

// C2C - AVX256 Kernels - Double
aoclfftz_kernel_test_params_t param_double_c2c_avx256_kernels[] =
{
    {2, aocl_fftz_kernel_type::STANDARD_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {3, aocl_fftz_kernel_type::STANDARD_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {4, aocl_fftz_kernel_type::STANDARD_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::STANDARD_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {6, aocl_fftz_kernel_type::STANDARD_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {7, aocl_fftz_kernel_type::STANDARD_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {8, aocl_fftz_kernel_type::STANDARD_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {9, aocl_fftz_kernel_type::STANDARD_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {10, aocl_fftz_kernel_type::STANDARD_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {11, aocl_fftz_kernel_type::STANDARD_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {12, aocl_fftz_kernel_type::STANDARD_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {13, aocl_fftz_kernel_type::STANDARD_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {14, aocl_fftz_kernel_type::STANDARD_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {15, aocl_fftz_kernel_type::STANDARD_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {16, aocl_fftz_kernel_type::STANDARD_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {2, aocl_fftz_kernel_type::PERMUTED_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {3, aocl_fftz_kernel_type::PERMUTED_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {4, aocl_fftz_kernel_type::PERMUTED_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::PERMUTED_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {6, aocl_fftz_kernel_type::PERMUTED_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {7, aocl_fftz_kernel_type::PERMUTED_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {8, aocl_fftz_kernel_type::PERMUTED_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {9, aocl_fftz_kernel_type::PERMUTED_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {10, aocl_fftz_kernel_type::PERMUTED_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {11, aocl_fftz_kernel_type::PERMUTED_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {12, aocl_fftz_kernel_type::PERMUTED_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {13, aocl_fftz_kernel_type::PERMUTED_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {14, aocl_fftz_kernel_type::PERMUTED_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {15, aocl_fftz_kernel_type::PERMUTED_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL},
    {16, aocl_fftz_kernel_type::PERMUTED_C2C_AVX256,
     aoclfftz_kernel_test_type::ALL}
};

// C2C - AVX512 Kernels - Float
aoclfftz_kernel_test_params_t param_float_c2c_avx512_kernels[] =
{
    {2, aocl_fftz_kernel_type::STANDARD_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {3, aocl_fftz_kernel_type::STANDARD_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {4, aocl_fftz_kernel_type::STANDARD_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::STANDARD_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {6, aocl_fftz_kernel_type::STANDARD_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {7, aocl_fftz_kernel_type::STANDARD_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {8, aocl_fftz_kernel_type::STANDARD_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {9, aocl_fftz_kernel_type::STANDARD_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {10, aocl_fftz_kernel_type::STANDARD_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {11, aocl_fftz_kernel_type::STANDARD_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {12, aocl_fftz_kernel_type::STANDARD_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {13, aocl_fftz_kernel_type::STANDARD_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {14, aocl_fftz_kernel_type::STANDARD_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {15, aocl_fftz_kernel_type::STANDARD_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {16, aocl_fftz_kernel_type::STANDARD_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {2, aocl_fftz_kernel_type::PERMUTED_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {3, aocl_fftz_kernel_type::PERMUTED_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {4, aocl_fftz_kernel_type::PERMUTED_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::PERMUTED_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {6, aocl_fftz_kernel_type::PERMUTED_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {7, aocl_fftz_kernel_type::PERMUTED_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {8, aocl_fftz_kernel_type::PERMUTED_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {9, aocl_fftz_kernel_type::PERMUTED_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {10, aocl_fftz_kernel_type::PERMUTED_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {11, aocl_fftz_kernel_type::PERMUTED_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {12, aocl_fftz_kernel_type::PERMUTED_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {13, aocl_fftz_kernel_type::PERMUTED_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {14, aocl_fftz_kernel_type::PERMUTED_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {15, aocl_fftz_kernel_type::PERMUTED_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {16, aocl_fftz_kernel_type::PERMUTED_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL}
};

// C2C - AVX512 Kernels - Double
aoclfftz_kernel_test_params_t param_double_c2c_avx512_kernels[] =
{
    {2, aocl_fftz_kernel_type::STANDARD_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {3, aocl_fftz_kernel_type::STANDARD_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {4, aocl_fftz_kernel_type::STANDARD_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::STANDARD_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {6, aocl_fftz_kernel_type::STANDARD_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {7, aocl_fftz_kernel_type::STANDARD_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {8, aocl_fftz_kernel_type::STANDARD_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {9, aocl_fftz_kernel_type::STANDARD_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {10, aocl_fftz_kernel_type::STANDARD_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {11, aocl_fftz_kernel_type::STANDARD_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {12, aocl_fftz_kernel_type::STANDARD_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {13, aocl_fftz_kernel_type::STANDARD_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {14, aocl_fftz_kernel_type::STANDARD_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {15, aocl_fftz_kernel_type::STANDARD_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {16, aocl_fftz_kernel_type::STANDARD_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {2, aocl_fftz_kernel_type::PERMUTED_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {3, aocl_fftz_kernel_type::PERMUTED_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {4, aocl_fftz_kernel_type::PERMUTED_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::PERMUTED_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {6, aocl_fftz_kernel_type::PERMUTED_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {7, aocl_fftz_kernel_type::PERMUTED_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {8, aocl_fftz_kernel_type::PERMUTED_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {9, aocl_fftz_kernel_type::PERMUTED_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {10, aocl_fftz_kernel_type::PERMUTED_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {11, aocl_fftz_kernel_type::PERMUTED_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {12, aocl_fftz_kernel_type::PERMUTED_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {13, aocl_fftz_kernel_type::PERMUTED_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {14, aocl_fftz_kernel_type::PERMUTED_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {15, aocl_fftz_kernel_type::PERMUTED_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL},
    {16, aocl_fftz_kernel_type::PERMUTED_C2C_AVX512,
     aoclfftz_kernel_test_type::ALL}
};

// R2HC - C Kernels - Float
aoclfftz_kernel_test_params_t param_float_r2hc_c_kernels[] =
{
    {2, aocl_fftz_kernel_type::STANDARD_R2HC_C,
        aoclfftz_kernel_test_type::ALL},
    {3, aocl_fftz_kernel_type::STANDARD_R2HC_C,
        aoclfftz_kernel_test_type::ALL},
    {4, aocl_fftz_kernel_type::STANDARD_R2HC_C,
        aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::STANDARD_R2HC_C,
        aoclfftz_kernel_test_type::ALL},
    {6, aocl_fftz_kernel_type::STANDARD_R2HC_C,
        aoclfftz_kernel_test_type::ALL},
    {7, aocl_fftz_kernel_type::STANDARD_R2HC_C,
        aoclfftz_kernel_test_type::ALL},
    {8, aocl_fftz_kernel_type::STANDARD_R2HC_C,
        aoclfftz_kernel_test_type::ALL},
    {10, aocl_fftz_kernel_type::STANDARD_R2HC_C,
         aoclfftz_kernel_test_type::ALL}
};

// R2HC - C Kernels - Double
aoclfftz_kernel_test_params_t param_double_r2hc_c_kernels[] =
{
    {2, aocl_fftz_kernel_type::STANDARD_R2HC_C,
        aoclfftz_kernel_test_type::ALL},
    {3, aocl_fftz_kernel_type::STANDARD_R2HC_C,
        aoclfftz_kernel_test_type::ALL},
    {4, aocl_fftz_kernel_type::STANDARD_R2HC_C,
        aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::STANDARD_R2HC_C,
        aoclfftz_kernel_test_type::ALL},
    {6, aocl_fftz_kernel_type::STANDARD_R2HC_C,
        aoclfftz_kernel_test_type::ALL},
    {7, aocl_fftz_kernel_type::STANDARD_R2HC_C,
        aoclfftz_kernel_test_type::ALL},
    {8, aocl_fftz_kernel_type::STANDARD_R2HC_C,
        aoclfftz_kernel_test_type::ALL},
    {10, aocl_fftz_kernel_type::STANDARD_R2HC_C,
         aoclfftz_kernel_test_type::ALL}
};

// R2HC-Fused - C Kernels - Float
aoclfftz_kernel_test_params_t param_float_r2hcf_c_kernels[] =
{
    {2, aocl_fftz_kernel_type::STANDARD_R2HCF_C,
        aoclfftz_kernel_test_type::ALL},
    {3, aocl_fftz_kernel_type::STANDARD_R2HCF_C,
        aoclfftz_kernel_test_type::ALL},
    {4, aocl_fftz_kernel_type::STANDARD_R2HCF_C,
        aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::STANDARD_R2HCF_C,
        aoclfftz_kernel_test_type::ALL},
    {6, aocl_fftz_kernel_type::STANDARD_R2HCF_C,
        aoclfftz_kernel_test_type::ALL},
    {7, aocl_fftz_kernel_type::STANDARD_R2HCF_C,
        aoclfftz_kernel_test_type::ALL},
    {8, aocl_fftz_kernel_type::STANDARD_R2HCF_C,
        aoclfftz_kernel_test_type::ALL},
    {10, aocl_fftz_kernel_type::STANDARD_R2HCF_C,
         aoclfftz_kernel_test_type::ALL}
};

// R2HC-Fused - C Kernels - Double
aoclfftz_kernel_test_params_t param_double_r2hcf_c_kernels[] =
{
    {2, aocl_fftz_kernel_type::STANDARD_R2HCF_C,
        aoclfftz_kernel_test_type::ALL},
    {3, aocl_fftz_kernel_type::STANDARD_R2HCF_C,
        aoclfftz_kernel_test_type::ALL},
    {4, aocl_fftz_kernel_type::STANDARD_R2HCF_C,
        aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::STANDARD_R2HCF_C,
        aoclfftz_kernel_test_type::ALL},
    {6, aocl_fftz_kernel_type::STANDARD_R2HCF_C,
        aoclfftz_kernel_test_type::ALL},
    {7, aocl_fftz_kernel_type::STANDARD_R2HCF_C,
        aoclfftz_kernel_test_type::ALL},
    {8, aocl_fftz_kernel_type::STANDARD_R2HCF_C,
        aoclfftz_kernel_test_type::ALL},
    {10, aocl_fftz_kernel_type::STANDARD_R2HCF_C,
         aoclfftz_kernel_test_type::ALL}
};

#ifdef ENABLE_AVX128
// R2HC - AVX128 Kernels - Double
aoclfftz_kernel_test_params_t param_double_r2hc_avx128_kernels[] =
{
    {2, aocl_fftz_kernel_type::STANDARD_R2HC_AVX128,
        aoclfftz_kernel_test_type::ALL},
    {2, aocl_fftz_kernel_type::PERMUTED_R2HC_AVX128,
        aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::STANDARD_R2HC_AVX128,
        aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::PERMUTED_R2HC_AVX128,
        aoclfftz_kernel_test_type::ALL},
};

// R2HC - AVX128 Kernels - Float
aoclfftz_kernel_test_params_t param_float_r2hc_avx128_kernels[] =
{
    {2, aocl_fftz_kernel_type::STANDARD_R2HC_AVX128,
        aoclfftz_kernel_test_type::ALL},
    {2, aocl_fftz_kernel_type::PERMUTED_R2HC_AVX128,
        aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::STANDARD_R2HC_AVX128,
        aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::PERMUTED_R2HC_AVX128,
        aoclfftz_kernel_test_type::ALL},
};

// R2HC-Fused - AVX128 Kernels - Double
aoclfftz_kernel_test_params_t param_double_r2hcf_avx128_kernels[] =
{
    {7, aocl_fftz_kernel_type::STANDARD_R2HCF_AVX128,
        aoclfftz_kernel_test_type::ALL},
    {7, aocl_fftz_kernel_type::PERMUTED_R2HCF_AVX128,
        aoclfftz_kernel_test_type::ALL},
};

// R2HC-Fused - AVX128 Kernels - Float
aoclfftz_kernel_test_params_t param_float_r2hcf_avx128_kernels[] =
{
    {7, aocl_fftz_kernel_type::STANDARD_R2HCF_AVX128,
        aoclfftz_kernel_test_type::ALL},
    {7, aocl_fftz_kernel_type::PERMUTED_R2HCF_AVX128,
        aoclfftz_kernel_test_type::ALL},
};
#endif

#ifdef ENABLE_AVX256
// R2HC - AVX256 Kernels - Double
aoclfftz_kernel_test_params_t param_double_r2hc_avx256_kernels[] =
{
    {2, aocl_fftz_kernel_type::STANDARD_R2HC_AVX256,
        aoclfftz_kernel_test_type::ALL},
    {2, aocl_fftz_kernel_type::PERMUTED_R2HC_AVX256,
        aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::STANDARD_R2HC_AVX256,
        aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::PERMUTED_R2HC_AVX256,
        aoclfftz_kernel_test_type::ALL},
};

// R2HC - AVX256 Kernels - Float
aoclfftz_kernel_test_params_t param_float_r2hc_avx256_kernels[] =
{
    {2, aocl_fftz_kernel_type::STANDARD_R2HC_AVX256,
        aoclfftz_kernel_test_type::ALL},
    {2, aocl_fftz_kernel_type::PERMUTED_R2HC_AVX256,
        aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::STANDARD_R2HC_AVX256,
        aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::PERMUTED_R2HC_AVX256,
        aoclfftz_kernel_test_type::ALL},
};

// R2HC-Fused - AVX256 Kernels - Double
aoclfftz_kernel_test_params_t param_double_r2hcf_avx256_kernels[] =
{
    {2, aocl_fftz_kernel_type::STANDARD_R2HCF_AVX256,
        aoclfftz_kernel_test_type::ALL},
    {2, aocl_fftz_kernel_type::PERMUTED_R2HCF_AVX256,
        aoclfftz_kernel_test_type::ALL},
};

// R2HC-Fused - AVX256 Kernels - Float
aoclfftz_kernel_test_params_t param_float_r2hcf_avx256_kernels[] =
{
    {2, aocl_fftz_kernel_type::STANDARD_R2HCF_AVX256,
        aoclfftz_kernel_test_type::ALL},
    {2, aocl_fftz_kernel_type::PERMUTED_R2HCF_AVX256,
        aoclfftz_kernel_test_type::ALL},
};
#endif

#ifdef ENABLE_AVX512
// R2HC - AVX512 Kernels - Double
aoclfftz_kernel_test_params_t param_double_r2hc_avx512_kernels[] =
{
    {2, aocl_fftz_kernel_type::STANDARD_R2HC_AVX512,
        aoclfftz_kernel_test_type::ALL},
    {2, aocl_fftz_kernel_type::PERMUTED_R2HC_AVX512,
        aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::STANDARD_R2HC_AVX512,
        aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::PERMUTED_R2HC_AVX512,
        aoclfftz_kernel_test_type::ALL},
};

// R2HC - AVX512 Kernels - Float
aoclfftz_kernel_test_params_t param_float_r2hc_avx512_kernels[] =
{
    {2, aocl_fftz_kernel_type::STANDARD_R2HC_AVX512,
        aoclfftz_kernel_test_type::ALL},
    {2, aocl_fftz_kernel_type::PERMUTED_R2HC_AVX512,
        aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::STANDARD_R2HC_AVX512,
        aoclfftz_kernel_test_type::ALL},
    {5, aocl_fftz_kernel_type::PERMUTED_R2HC_AVX512,
        aoclfftz_kernel_test_type::ALL},
};

// R2HC-Fused - AVX512 Kernels - Double
aoclfftz_kernel_test_params_t param_double_r2hcf_avx512_kernels[] =
{
    {2, aocl_fftz_kernel_type::STANDARD_R2HCF_AVX512,
        aoclfftz_kernel_test_type::ALL},
    {2, aocl_fftz_kernel_type::PERMUTED_R2HCF_AVX512,
        aoclfftz_kernel_test_type::ALL},
};

// R2HC-Fused - AVX512 Kernels - Float
aoclfftz_kernel_test_params_t param_float_r2hcf_avx512_kernels[] =
{
    {2, aocl_fftz_kernel_type::STANDARD_R2HCF_AVX512,
        aoclfftz_kernel_test_type::ALL},
    {2, aocl_fftz_kernel_type::PERMUTED_R2HCF_AVX512,
        aoclfftz_kernel_test_type::ALL},
};
#endif
// IO params as {in-stride, out-stride , batch size, dir of FFT(0->FWD/1-> BWD),
//               result placement(0 -> inplace, 1 -> out-of-place)}
// Batch size set to cover all the tail case in AVX128 & AVX256 kernels
std::vector<std::tuple<INTP, INTP, INTP, UINT8, UINT8>> io_params =
                                                            {{1,  1, 1, 0, 0},
                                                             {2,  9, 3, 1, 1},
                                                             {7,  3, 4, 0, 1},
                                                             {4,  4, 7, 1, 0},
                                                             {11, 1, 8, 0, 1},
                                                             {1,  6, 15, 1, 1},
                                                             {10, 5, 16, 0, 1}};

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

TEST_P(AoclfftzKernelTestFloatReal, TEST_FLOAT_REAL_KERNEL)
{
    run_kernel_test(); // run test with normal values
}

TEST_P(AoclfftzKernelTestDoubleReal, TEST_DOUBLE_REAL_KERNEL)
{
    run_kernel_test(); // run test with normal values
}

TEST_P(AoclfftzKernelTestFloatFused, TEST_FLOAT_REAL_KERNEL)
{
    run_kernel_test(); // run test with normal values
}

TEST_P(AoclfftzKernelTestDoubleFused, TEST_DOUBLE_REAL_KERNEL)
{
    run_kernel_test(); // run test with normal values
}

/**
 * @brief An utility function to return the test name based on the test_type
 *
 */
auto name_generator =
    [](const ::testing::TestParamInfo<std::tuple<aoclfftz_kernel_test_params_t,
                                      std::tuple<INTP, INTP, INTP, UINT8,
                                                 UINT8>>> &info)
    {
        auto param = std::get<0>(info.param);
        auto io_param = std::get<1>(info.param);
        INTP istride = std::get<0>(io_param);
        INTP ostride = std::get<1>(io_param);
        INTP batch_sz = std::get<2>(io_param);
        UINT8 is_bwd  = std::get<3>(io_param);
        UINT8 is_out_of_place  = std::get<4>(io_param);
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
        if (is_out_of_place)
        {
            test_name += "_OUTOFPLACE";
        }
        else
        {
            test_name += "_INPLACE";
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
            {
                test_name += "_LINEARITY";
            }
            if (test_type & aoclfftz_kernel_test_type::TRANSFORMATION)
            {
                test_name += "_TRANSFORMATION";
            }
            if (test_type & aoclfftz_kernel_test_type::TIMESHIFT)
            {
                test_name += "_TIMESHIFT";
            }
            if (test_type & aoclfftz_kernel_test_type::DFT_REFERENCE)
            {
                test_name += "_DFTREF";
            }
        }
        return test_name;
    };

// C2C Kernels
INSTANTIATE_TEST_SUITE_P(
    C2C_C_KernelTest, AoclfftzKernelTestFloat,
    ::testing::Combine(::testing::ValuesIn(param_float_c2c_c_kernels),
                       ::testing::ValuesIn(io_params)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    C2C_C_KernelTest, AoclfftzKernelTestDouble,
    ::testing::Combine(::testing::ValuesIn(param_double_c2c_c_kernels),
                       ::testing::ValuesIn(io_params)),
    name_generator);

#ifdef ENABLE_AVX128
INSTANTIATE_TEST_SUITE_P(
    C2C_AVX128_KernelTest, AoclfftzKernelTestFloat,
    ::testing::Combine(::testing::ValuesIn(param_float_c2c_avx128_kernels),
                       ::testing::ValuesIn(io_params)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    C2C_AVX128_KernelTest, AoclfftzKernelTestDouble,
    ::testing::Combine(::testing::ValuesIn(param_double_c2c_avx128_kernels),
                       ::testing::ValuesIn(io_params)),
    name_generator);
#endif

#ifdef ENABLE_AVX256
INSTANTIATE_TEST_SUITE_P(
    C2C_AVX256_KernelTest, AoclfftzKernelTestFloat,
    ::testing::Combine(::testing::ValuesIn(param_float_c2c_avx256_kernels),
                       ::testing::ValuesIn(io_params)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    C2C_AVX256_KernelTest, AoclfftzKernelTestDouble,
    ::testing::Combine(::testing::ValuesIn(param_double_c2c_avx256_kernels),
                       ::testing::ValuesIn(io_params)),
    name_generator);
#endif

// Separate io params for c2c AVX512 Kernels and r2hc AVX256 kernel onwards to
// avoid batch-sizes 8-15 repetition in c2c AVX128 & AVX256 kernels and r2hc
// AVX128 kernels
// IO params as {in-stride, out-stride , batch size, dir of FFT(0->FWD/1-> BWD),
//               result placement(0 -> inplace, 1 -> out-of-place)}
std::vector<std::tuple<INTP, INTP, INTP, UINT8, UINT8>> io_params_batch16 = {
                                                            {1,   1,  1, 0, 0},
                                                            {1,   1,  2, 1, 0},
                                                            {5,   3,  3, 0, 1},
                                                            {10, 15,  4, 1, 1},
                                                            {11,  1,  5, 0, 1},
                                                            {8,   1,  6, 1, 1},
                                                            {1,   7,  7, 0, 1},
                                                            {1,  13,  8, 1, 1},
                                                            {7,   3,  9, 0, 1},
                                                            {12,  4, 10, 1, 1},
                                                            {11, 21, 11, 0, 1},
                                                            {8,  15, 12, 1, 1},
                                                            {10,  5, 13, 0, 1},
                                                            {5,   5, 14, 1, 0},
                                                            {15, 15, 15, 0, 0}};


// Separate io params for r2hc AVX512 Kernels to avoid batch-sizes 30-32
// repetition in AVX128/AVX256 kernels
// IO params as {in-stride, out-stride , batch size, dir of FFT(0->FWD/1-> BWD),
//               result placement(0 -> inplace, 1 -> out-of-place)}
// The inbetween batch size [17-30] are skipped because the flow is
// tested by other batch size
std::vector<std::tuple<INTP, INTP, INTP, UINT8, UINT8>> io_params_batch32 = {
                                                            {1,   1,  1, 0, 0},
                                                            {1,   1,  2, 1, 0},
                                                            {5,   3,  3, 0, 1},
                                                            {10, 15,  4, 1, 1},
                                                            {11,  1,  5, 0, 1},
                                                            {8,   1,  6, 1, 1},
                                                            {1,   7,  7, 0, 1},
                                                            {1,  13,  8, 1, 1},
                                                            {7,   3,  9, 0, 1},
                                                            {12,  4, 10, 1, 1},
                                                            {11, 21, 11, 0, 1},
                                                            {8,  15, 12, 1, 1},
                                                            {10,  5, 13, 0, 1},
                                                            {5,   5, 14, 1, 0},
                                                            {15, 15, 15, 0, 0},
                                                            {11,  5, 16, 1, 1},
                                                            {3,  10, 31, 0, 1},
                                                            {17, 2, 32, 1, 1}};
#ifdef ENABLE_AVX512
INSTANTIATE_TEST_SUITE_P(
    C2C_AVX512_KernelTest, AoclfftzKernelTestFloat,
    ::testing::Combine(::testing::ValuesIn(param_float_c2c_avx512_kernels),
                       ::testing::ValuesIn(io_params_batch16)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    C2C_AVX512_KernelTest, AoclfftzKernelTestDouble,
    ::testing::Combine(::testing::ValuesIn(param_double_c2c_avx512_kernels),
                       ::testing::ValuesIn(io_params_batch16)),
    name_generator);
#endif

// R2HC Kernels
INSTANTIATE_TEST_SUITE_P(
    R2HC_C_KernelTest, AoclfftzKernelTestFloatReal,
    ::testing::Combine(::testing::ValuesIn(param_float_r2hc_c_kernels),
                       ::testing::ValuesIn(io_params)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    R2HC_C_KernelTest, AoclfftzKernelTestDoubleReal,
    ::testing::Combine(::testing::ValuesIn(param_double_r2hc_c_kernels),
                       ::testing::ValuesIn(io_params)),
    name_generator);

// R2HC-Fused Kernels
INSTANTIATE_TEST_SUITE_P(
    R2HCF_C_KernelTest, AoclfftzKernelTestFloatFused,
    ::testing::Combine(::testing::ValuesIn(param_float_r2hcf_c_kernels),
                       ::testing::ValuesIn(io_params)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    R2HCF_C_KernelTest, AoclfftzKernelTestDoubleFused,
    ::testing::Combine(::testing::ValuesIn(param_double_r2hcf_c_kernels),
                       ::testing::ValuesIn(io_params)),
    name_generator);

#ifdef ENABLE_AVX128
// R2HC AVX128 Kernels
INSTANTIATE_TEST_SUITE_P(
    R2HC_AVX128_KernelTest, AoclfftzKernelTestFloatReal,
    ::testing::Combine(::testing::ValuesIn(param_float_r2hc_avx128_kernels),
                       ::testing::ValuesIn(io_params)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    R2HC_AVX128_KernelTest, AoclfftzKernelTestDoubleReal,
    ::testing::Combine(::testing::ValuesIn(param_double_r2hc_avx128_kernels),
                       ::testing::ValuesIn(io_params)),
    name_generator);

// R2HC-Fused AVX128 Kernels
INSTANTIATE_TEST_SUITE_P(
    R2HCF_AVX128_KernelTest, AoclfftzKernelTestFloatFused,
    ::testing::Combine(::testing::ValuesIn(param_float_r2hcf_avx128_kernels),
                       ::testing::ValuesIn(io_params)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    R2HCF_AVX128_KernelTest, AoclfftzKernelTestDoubleFused,
    ::testing::Combine(::testing::ValuesIn(param_double_r2hcf_avx128_kernels),
                       ::testing::ValuesIn(io_params)),
    name_generator);
#endif

#ifdef ENABLE_AVX256
// R2HC AVX256 Kernels
INSTANTIATE_TEST_SUITE_P(
    R2HC_AVX256_KernelTest, AoclfftzKernelTestFloatReal,
    ::testing::Combine(::testing::ValuesIn(param_float_r2hc_avx256_kernels),
                       ::testing::ValuesIn(io_params_batch16)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    R2HC_AVX256_KernelTest, AoclfftzKernelTestDoubleReal,
    ::testing::Combine(::testing::ValuesIn(param_double_r2hc_avx256_kernels),
                       ::testing::ValuesIn(io_params_batch16)),
    name_generator);

// R2HC-Fused AVX256 Kernels
INSTANTIATE_TEST_SUITE_P(
    R2HCF_AVX256_KernelTest, AoclfftzKernelTestFloatFused,
    ::testing::Combine(::testing::ValuesIn(param_float_r2hcf_avx256_kernels),
                       ::testing::ValuesIn(io_params_batch16)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    R2HCF_AVX256_KernelTest, AoclfftzKernelTestDoubleFused,
    ::testing::Combine(::testing::ValuesIn(param_double_r2hcf_avx256_kernels),
                       ::testing::ValuesIn(io_params_batch16)),
    name_generator);
#endif

#ifdef ENABLE_AVX512
// R2HC AVX512 Kernels
INSTANTIATE_TEST_SUITE_P(
    R2HC_AVX512_KernelTest, AoclfftzKernelTestFloatReal,
    ::testing::Combine(::testing::ValuesIn(param_float_r2hc_avx512_kernels),
                       ::testing::ValuesIn(io_params_batch32)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    R2HC_AVX512_KernelTest, AoclfftzKernelTestDoubleReal,
    ::testing::Combine(::testing::ValuesIn(param_double_r2hc_avx512_kernels),
                       ::testing::ValuesIn(io_params_batch32)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    R2HCF_AVX512_KernelTest, AoclfftzKernelTestFloatFused,
    ::testing::Combine(::testing::ValuesIn(param_float_r2hcf_avx512_kernels),
                       ::testing::ValuesIn(io_params_batch32)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    R2HCF_AVX512_KernelTest, AoclfftzKernelTestDoubleFused,
    ::testing::Combine(::testing::ValuesIn(param_double_r2hcf_avx512_kernels),
                       ::testing::ValuesIn(io_params_batch32)),
    name_generator);
#endif
