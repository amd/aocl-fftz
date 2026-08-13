// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file kernel_gtest.cpp
 *
 *  @brief File that contains the GTest based kernel unit tests.
 *
 *  This file contains the entry point of the GTest. It contains the kernel
 *  table initializations for kernel unit tests.
 *
 *  @author Srirammaswamy Srinivasan
 *  @author Prasandh Sankarankutty
 *  @author Jeevanantham N
 */

#include <gtest/gtest.h>
#include "kernel_gtest_base.h"
#include "dft_reference.h"

// C2C - C Kernels
std::vector<aoclfftz_kernel_test_params_t> param_c2c_c_kernels =
{
    {2, aocl_fftz_kernel_type::C2C_C},
    {3, aocl_fftz_kernel_type::C2C_C},
    {4, aocl_fftz_kernel_type::C2C_C},
    {5, aocl_fftz_kernel_type::C2C_C},
    {6, aocl_fftz_kernel_type::C2C_C},
    {7, aocl_fftz_kernel_type::C2C_C},
    {8, aocl_fftz_kernel_type::C2C_C},
    {9, aocl_fftz_kernel_type::C2C_C},
    {10, aocl_fftz_kernel_type::C2C_C},
    {11, aocl_fftz_kernel_type::C2C_C},
    {12, aocl_fftz_kernel_type::C2C_C},
    {13, aocl_fftz_kernel_type::C2C_C},
    {14, aocl_fftz_kernel_type::C2C_C},
    {15, aocl_fftz_kernel_type::C2C_C},
    {16, aocl_fftz_kernel_type::C2C_C},
    {20, aocl_fftz_kernel_type::C2C_C},
    {48, aocl_fftz_kernel_type::C2C_C},
};

// C2C - AVX128 Kernels
std::vector<aoclfftz_kernel_test_params_t> param_c2c_avx128_kernels =
{
    {2, aocl_fftz_kernel_type::C2C_AVX128},
    {3, aocl_fftz_kernel_type::C2C_AVX128},
    {4, aocl_fftz_kernel_type::C2C_AVX128},
    {5, aocl_fftz_kernel_type::C2C_AVX128},
    {6, aocl_fftz_kernel_type::C2C_AVX128},
    {7, aocl_fftz_kernel_type::C2C_AVX128},
    {8, aocl_fftz_kernel_type::C2C_AVX128},
    {9, aocl_fftz_kernel_type::C2C_AVX128},
    {10, aocl_fftz_kernel_type::C2C_AVX128},
    {11, aocl_fftz_kernel_type::C2C_AVX128},
    {12, aocl_fftz_kernel_type::C2C_AVX128},
    {13, aocl_fftz_kernel_type::C2C_AVX128},
    {14, aocl_fftz_kernel_type::C2C_AVX128},
    {15, aocl_fftz_kernel_type::C2C_AVX128},
    {16, aocl_fftz_kernel_type::C2C_AVX128},
    {20, aocl_fftz_kernel_type::C2C_AVX128},
    {48, aocl_fftz_kernel_type::C2C_AVX128},
};

// C2C - AVX256 Kernels
std::vector<aoclfftz_kernel_test_params_t> param_c2c_avx256_kernels =
{
    {2, aocl_fftz_kernel_type::C2C_AVX256},
    {3, aocl_fftz_kernel_type::C2C_AVX256},
    {4, aocl_fftz_kernel_type::C2C_AVX256},
    {5, aocl_fftz_kernel_type::C2C_AVX256},
    {6, aocl_fftz_kernel_type::C2C_AVX256},
    {7, aocl_fftz_kernel_type::C2C_AVX256},
    {8, aocl_fftz_kernel_type::C2C_AVX256},
    {9, aocl_fftz_kernel_type::C2C_AVX256},
    {10, aocl_fftz_kernel_type::C2C_AVX256},
    {11, aocl_fftz_kernel_type::C2C_AVX256},
    {12, aocl_fftz_kernel_type::C2C_AVX256},
    {13, aocl_fftz_kernel_type::C2C_AVX256},
    {14, aocl_fftz_kernel_type::C2C_AVX256},
    {15, aocl_fftz_kernel_type::C2C_AVX256},
    {16, aocl_fftz_kernel_type::C2C_AVX256},
    {20, aocl_fftz_kernel_type::C2C_AVX256},
    {48, aocl_fftz_kernel_type::C2C_AVX256},
};

// C2C - AVX512 Kernels
std::vector<aoclfftz_kernel_test_params_t> param_c2c_avx512_kernels =
{
    {2, aocl_fftz_kernel_type::C2C_AVX512},
    {3, aocl_fftz_kernel_type::C2C_AVX512},
    {4, aocl_fftz_kernel_type::C2C_AVX512},
    {5, aocl_fftz_kernel_type::C2C_AVX512},
    {6, aocl_fftz_kernel_type::C2C_AVX512},
    {7, aocl_fftz_kernel_type::C2C_AVX512},
    {8, aocl_fftz_kernel_type::C2C_AVX512},
    {9, aocl_fftz_kernel_type::C2C_AVX512},
    {10, aocl_fftz_kernel_type::C2C_AVX512},
    {11, aocl_fftz_kernel_type::C2C_AVX512},
    {12, aocl_fftz_kernel_type::C2C_AVX512},
    {13, aocl_fftz_kernel_type::C2C_AVX512},
    {14, aocl_fftz_kernel_type::C2C_AVX512},
    {15, aocl_fftz_kernel_type::C2C_AVX512},
    {16, aocl_fftz_kernel_type::C2C_AVX512},
    {20, aocl_fftz_kernel_type::C2C_AVX512},
    {48, aocl_fftz_kernel_type::C2C_AVX512},
};

// C2C Twiddle Forward - C Kernels
std::vector<aoclfftz_kernel_test_params_t> param_c2c_twid_fwd_c_kernels =
{
    {2, aocl_fftz_kernel_type::C2C_TWID_FWD_C},
    {3, aocl_fftz_kernel_type::C2C_TWID_FWD_C},
    {4, aocl_fftz_kernel_type::C2C_TWID_FWD_C},
    {5, aocl_fftz_kernel_type::C2C_TWID_FWD_C},
    {6, aocl_fftz_kernel_type::C2C_TWID_FWD_C},
    {7, aocl_fftz_kernel_type::C2C_TWID_FWD_C},
    {8, aocl_fftz_kernel_type::C2C_TWID_FWD_C},
    {9, aocl_fftz_kernel_type::C2C_TWID_FWD_C},
    {10, aocl_fftz_kernel_type::C2C_TWID_FWD_C},
    {11, aocl_fftz_kernel_type::C2C_TWID_FWD_C},
    {12, aocl_fftz_kernel_type::C2C_TWID_FWD_C},
    {13, aocl_fftz_kernel_type::C2C_TWID_FWD_C},
    {14, aocl_fftz_kernel_type::C2C_TWID_FWD_C},
    {15, aocl_fftz_kernel_type::C2C_TWID_FWD_C},
    {16, aocl_fftz_kernel_type::C2C_TWID_FWD_C},
};

// C2C Twiddle Backward - C Kernels
std::vector<aoclfftz_kernel_test_params_t> param_c2c_twid_bwd_c_kernels =
{
    {2, aocl_fftz_kernel_type::C2C_TWID_BWD_C},
    {3, aocl_fftz_kernel_type::C2C_TWID_BWD_C},
    {4, aocl_fftz_kernel_type::C2C_TWID_BWD_C},
    {5, aocl_fftz_kernel_type::C2C_TWID_BWD_C},
    {6, aocl_fftz_kernel_type::C2C_TWID_BWD_C},
    {7, aocl_fftz_kernel_type::C2C_TWID_BWD_C},
    {8, aocl_fftz_kernel_type::C2C_TWID_BWD_C},
    {9, aocl_fftz_kernel_type::C2C_TWID_BWD_C},
    {10, aocl_fftz_kernel_type::C2C_TWID_BWD_C},
    {11, aocl_fftz_kernel_type::C2C_TWID_BWD_C},
    {12, aocl_fftz_kernel_type::C2C_TWID_BWD_C},
    {13, aocl_fftz_kernel_type::C2C_TWID_BWD_C},
    {14, aocl_fftz_kernel_type::C2C_TWID_BWD_C},
    {15, aocl_fftz_kernel_type::C2C_TWID_BWD_C},
    {16, aocl_fftz_kernel_type::C2C_TWID_BWD_C},
};

// C2C Twiddle Forward - AVX128 Kernels
std::vector<aoclfftz_kernel_test_params_t> param_c2c_twid_fwd_avx128_kernels =
{
    {2, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX128},
    {3, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX128},
    {4, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX128},
    {5, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX128},
    {6, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX128},
    {7, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX128},
    {8, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX128},
    {9, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX128},
    {10, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX128},
    {11, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX128},
    {12, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX128},
    {13, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX128},
    {14, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX128},
    {15, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX128},
    {16, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX128},
};

// C2C Twiddle Backward - AVX128 Kernels
std::vector<aoclfftz_kernel_test_params_t> param_c2c_twid_bwd_avx128_kernels =
{
    {2, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX128},
    {3, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX128},
    {4, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX128},
    {5, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX128},
    {6, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX128},
    {7, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX128},
    {8, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX128},
    {9, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX128},
    {10, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX128},
    {11, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX128},
    {12, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX128},
    {13, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX128},
    {14, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX128},
    {15, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX128},
    {16, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX128},
};

// C2C Twiddle Forward - AVX256 Kernels
std::vector<aoclfftz_kernel_test_params_t> param_c2c_twid_fwd_avx256_kernels =
{
    {2, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX256},
    {3, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX256},
    {4, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX256},
    {5, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX256},
    {6, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX256},
    {7, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX256},
    {8, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX256},
    {9, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX256},
    {10, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX256},
    {11, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX256},
    {12, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX256},
    {13, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX256},
    {14, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX256},
    {15, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX256},
    {16, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX256},
};

// C2C Twiddle Backward - AVX256 Kernels
std::vector<aoclfftz_kernel_test_params_t> param_c2c_twid_bwd_avx256_kernels =
{
    {2, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX256},
    {3, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX256},
    {4, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX256},
    {5, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX256},
    {6, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX256},
    {7, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX256},
    {8, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX256},
    {9, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX256},
    {10, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX256},
    {11, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX256},
    {12, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX256},
    {13, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX256},
    {14, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX256},
    {15, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX256},
    {16, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX256},
};

// C2C Twiddle Forward - AVX512 Kernels
std::vector<aoclfftz_kernel_test_params_t> param_c2c_twid_fwd_avx512_kernels =
{
    {2, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX512},
    {3, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX512},
    {4, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX512},
    {5, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX512},
    {6, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX512},
    {7, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX512},
    {8, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX512},
    {9, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX512},
    {10, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX512},
    {11, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX512},
    {12, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX512},
    {13, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX512},
    {14, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX512},
    {15, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX512},
    {16, aocl_fftz_kernel_type::C2C_TWID_FWD_AVX512},
};

// C2C Twiddle Backward - AVX512 Kernels
std::vector<aoclfftz_kernel_test_params_t> param_c2c_twid_bwd_avx512_kernels =
{
    {2, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX512},
    {3, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX512},
    {4, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX512},
    {5, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX512},
    {6, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX512},
    {7, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX512},
    {8, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX512},
    {9, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX512},
    {10, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX512},
    {11, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX512},
    {12, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX512},
    {13, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX512},
    {14, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX512},
    {15, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX512},
    {16, aocl_fftz_kernel_type::C2C_TWID_BWD_AVX512},
};

// R2HC - C Kernels
std::vector<aoclfftz_kernel_test_params_t> param_r2hc_c_kernels =
{
    {2, aocl_fftz_kernel_type::R2HC_C},
    {3, aocl_fftz_kernel_type::R2HC_C},
    {4, aocl_fftz_kernel_type::R2HC_C},
    {5, aocl_fftz_kernel_type::R2HC_C},
    {6, aocl_fftz_kernel_type::R2HC_C},
    {7, aocl_fftz_kernel_type::R2HC_C},
    {8, aocl_fftz_kernel_type::R2HC_C},
    {9, aocl_fftz_kernel_type::R2HC_C},
    {10, aocl_fftz_kernel_type::R2HC_C},
    {11, aocl_fftz_kernel_type::R2HC_C},
    {12, aocl_fftz_kernel_type::R2HC_C},
    {13, aocl_fftz_kernel_type::R2HC_C},
    {14, aocl_fftz_kernel_type::R2HC_C},
    {15, aocl_fftz_kernel_type::R2HC_C},
    {16, aocl_fftz_kernel_type::R2HC_C},
};

// R2HC-Fused - C Kernels
std::vector<aoclfftz_kernel_test_params_t> param_r2hcf_c_kernels =
{
    {2, aocl_fftz_kernel_type::R2HCF_C},
    {3, aocl_fftz_kernel_type::R2HCF_C},
    {4, aocl_fftz_kernel_type::R2HCF_C},
    {5, aocl_fftz_kernel_type::R2HCF_C},
    {6, aocl_fftz_kernel_type::R2HCF_C},
    {7, aocl_fftz_kernel_type::R2HCF_C},
    {8, aocl_fftz_kernel_type::R2HCF_C},
    {9, aocl_fftz_kernel_type::R2HCF_C},
    {10, aocl_fftz_kernel_type::R2HCF_C},
    {11, aocl_fftz_kernel_type::R2HCF_C},
    {12, aocl_fftz_kernel_type::R2HCF_C},
    {13, aocl_fftz_kernel_type::R2HCF_C},
    {14, aocl_fftz_kernel_type::R2HCF_C},
    {15, aocl_fftz_kernel_type::R2HCF_C},
    {16, aocl_fftz_kernel_type::R2HCF_C},
};

#ifdef ENABLE_AVX128
// R2HC - AVX128 Kernels
std::vector<aoclfftz_kernel_test_params_t> param_r2hc_avx128_kernels =
{
    {2, aocl_fftz_kernel_type::R2HC_AVX128},
    {3, aocl_fftz_kernel_type::R2HC_AVX128},
    {4, aocl_fftz_kernel_type::R2HC_AVX128},
    {5, aocl_fftz_kernel_type::R2HC_AVX128},
    {6, aocl_fftz_kernel_type::R2HC_AVX128},
    {7, aocl_fftz_kernel_type::R2HC_AVX128},
    {8, aocl_fftz_kernel_type::R2HC_AVX128},
    {9, aocl_fftz_kernel_type::R2HC_AVX128},
    {10, aocl_fftz_kernel_type::R2HC_AVX128},
    {11, aocl_fftz_kernel_type::R2HC_AVX128},
    {12, aocl_fftz_kernel_type::R2HC_AVX128},
    {13, aocl_fftz_kernel_type::R2HC_AVX128},
    {14, aocl_fftz_kernel_type::R2HC_AVX128},
    {15, aocl_fftz_kernel_type::R2HC_AVX128},
    {16, aocl_fftz_kernel_type::R2HC_AVX128},
};

// R2HC-Fused - AVX128 Kernels
std::vector<aoclfftz_kernel_test_params_t> param_r2hcf_avx128_kernels =
{
    {2, aocl_fftz_kernel_type::R2HCF_AVX128},
    {3, aocl_fftz_kernel_type::R2HCF_AVX128},
    {4, aocl_fftz_kernel_type::R2HCF_AVX128},
    {5, aocl_fftz_kernel_type::R2HCF_AVX128},
    {6, aocl_fftz_kernel_type::R2HCF_AVX128},
    {7, aocl_fftz_kernel_type::R2HCF_AVX128},
    {8, aocl_fftz_kernel_type::R2HCF_AVX128},
    {9, aocl_fftz_kernel_type::R2HCF_AVX128},
    {10, aocl_fftz_kernel_type::R2HCF_AVX128},
    {11, aocl_fftz_kernel_type::R2HCF_AVX128},
    {12, aocl_fftz_kernel_type::R2HCF_AVX128},
    {13, aocl_fftz_kernel_type::R2HCF_AVX128},
    {14, aocl_fftz_kernel_type::R2HCF_AVX128},
    {15, aocl_fftz_kernel_type::R2HCF_AVX128},
    {16, aocl_fftz_kernel_type::R2HCF_AVX128},
};
#endif

#ifdef ENABLE_AVX256
// R2HC - AVX256 Kernels
std::vector<aoclfftz_kernel_test_params_t> param_r2hc_avx256_kernels =
{
    {2, aocl_fftz_kernel_type::R2HC_AVX256},
    {3, aocl_fftz_kernel_type::R2HC_AVX256},
    {4, aocl_fftz_kernel_type::R2HC_AVX256},
    {5, aocl_fftz_kernel_type::R2HC_AVX256},
    {6, aocl_fftz_kernel_type::R2HC_AVX256},
    {7, aocl_fftz_kernel_type::R2HC_AVX256},
    {8, aocl_fftz_kernel_type::R2HC_AVX256},
    {9, aocl_fftz_kernel_type::R2HC_AVX256},
    {10, aocl_fftz_kernel_type::R2HC_AVX256},
    {11, aocl_fftz_kernel_type::R2HC_AVX256},
    {12, aocl_fftz_kernel_type::R2HC_AVX256},
    {13, aocl_fftz_kernel_type::R2HC_AVX256},
    {14, aocl_fftz_kernel_type::R2HC_AVX256},
    {15, aocl_fftz_kernel_type::R2HC_AVX256},
    {16, aocl_fftz_kernel_type::R2HC_AVX256},
};

// R2HC-Fused - AVX256 Kernels
std::vector<aoclfftz_kernel_test_params_t> param_r2hcf_avx256_kernels =
{
    {2, aocl_fftz_kernel_type::R2HCF_AVX256},
    {3, aocl_fftz_kernel_type::R2HCF_AVX256},
    {4, aocl_fftz_kernel_type::R2HCF_AVX256},
    {5, aocl_fftz_kernel_type::R2HCF_AVX256},
    {6, aocl_fftz_kernel_type::R2HCF_AVX256},
    {7, aocl_fftz_kernel_type::R2HCF_AVX256},
    {8, aocl_fftz_kernel_type::R2HCF_AVX256},
    {9, aocl_fftz_kernel_type::R2HCF_AVX256},
    {10, aocl_fftz_kernel_type::R2HCF_AVX256},
    {11, aocl_fftz_kernel_type::R2HCF_AVX256},
    {12, aocl_fftz_kernel_type::R2HCF_AVX256},
    {13, aocl_fftz_kernel_type::R2HCF_AVX256},
    {14, aocl_fftz_kernel_type::R2HCF_AVX256},
    {15, aocl_fftz_kernel_type::R2HCF_AVX256},
    {16, aocl_fftz_kernel_type::R2HCF_AVX256},
};
#endif

#ifdef ENABLE_AVX512
// R2HC - AVX512 Kernels
std::vector<aoclfftz_kernel_test_params_t> param_r2hc_avx512_kernels =
{
    {2, aocl_fftz_kernel_type::R2HC_AVX512},
    {3, aocl_fftz_kernel_type::R2HC_AVX512},
    {4, aocl_fftz_kernel_type::R2HC_AVX512},
    {5, aocl_fftz_kernel_type::R2HC_AVX512},
    {6, aocl_fftz_kernel_type::R2HC_AVX512},
    {7, aocl_fftz_kernel_type::R2HC_AVX512},
    {8, aocl_fftz_kernel_type::R2HC_AVX512},
    {9, aocl_fftz_kernel_type::R2HC_AVX512},
    {10, aocl_fftz_kernel_type::R2HC_AVX512},
    {11, aocl_fftz_kernel_type::R2HC_AVX512},
    {12, aocl_fftz_kernel_type::R2HC_AVX512},
    {13, aocl_fftz_kernel_type::R2HC_AVX512},
    {14, aocl_fftz_kernel_type::R2HC_AVX512},
    {15, aocl_fftz_kernel_type::R2HC_AVX512},
    {16, aocl_fftz_kernel_type::R2HC_AVX512},
};

// R2HC-Fused - AVX512 Kernels
std::vector<aoclfftz_kernel_test_params_t> param_r2hcf_avx512_kernels =
{
    {2, aocl_fftz_kernel_type::R2HCF_AVX512},
    {3, aocl_fftz_kernel_type::R2HCF_AVX512},
    {4, aocl_fftz_kernel_type::R2HCF_AVX512},
    {5, aocl_fftz_kernel_type::R2HCF_AVX512},
    {6, aocl_fftz_kernel_type::R2HCF_AVX512},
    {7, aocl_fftz_kernel_type::R2HCF_AVX512},
    {8, aocl_fftz_kernel_type::R2HCF_AVX512},
    {9, aocl_fftz_kernel_type::R2HCF_AVX512},
    {10, aocl_fftz_kernel_type::R2HCF_AVX512},
    {11, aocl_fftz_kernel_type::R2HCF_AVX512},
    {12, aocl_fftz_kernel_type::R2HCF_AVX512},
    {13, aocl_fftz_kernel_type::R2HCF_AVX512},
    {14, aocl_fftz_kernel_type::R2HCF_AVX512},
    {15, aocl_fftz_kernel_type::R2HCF_AVX512},
    {16, aocl_fftz_kernel_type::R2HCF_AVX512},
};
#endif
// IO params as {in-stride, out-stride , batch size, dir of FFT(0->FWD/1-> BWD),
//               result placement(0 -> inplace, 1 -> out-of-place),
//               load_multi_cols (0 -> broadcast same twiddle, 1 -> load
//               different twiddles, used only for twiddle kernels)}
// Batch size set to cover all the tail case in AVX128 & AVX256 kernels.
std::vector<std::tuple<FFTZ_INTP, FFTZ_INTP, FFTZ_INTP, FFTZ_UINT8, FFTZ_UINT8,
                       FFTZ_UINT8>>
    io_params = {{1, 1, 1, 0, 0, 1},  {2, 2, 2, 0, 1, 1},  {7, 3, 4, 0, 1, 0},
                 {4, 4, 7, 1, 0, 1},  {11, 1, 8, 0, 1, 0}, {1, 6, 15, 1, 1, 1},
                 {10, 5, 31, 0, 1, 0}};

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

TEST_P(AoclfftzTwiddleKernelTestFloat, TEST_FLOAT_C2C_TWID_KERNEL)
{
    run_twiddle_kernel_test(); // run test with normal values
}

TEST_P(AoclfftzTwiddleKernelTestDouble, TEST_DOUBLE_C2C_TWID_KERNEL)
{
    run_twiddle_kernel_test(); // run test with normal values
}

TEST_P(AoclfftzR2CTwiddleKernelTestFloat, TEST_FLOAT_R2C_TWIDDLE_KERNEL)
{
    run_twiddle_kernel_test();
}

TEST_P(AoclfftzR2CTwiddleKernelTestDouble, TEST_DOUBLE_R2C_TWIDDLE_KERNEL)
{
    run_twiddle_kernel_test();
}

TEST_P(AoclfftzC2RTwiddleKernelTestFloat, TEST_FLOAT_C2R_TWIDDLE_KERNEL)
{
    run_twiddle_kernel_test();
}

TEST_P(AoclfftzC2RTwiddleKernelTestDouble, TEST_DOUBLE_C2R_TWIDDLE_KERNEL)
{
    run_twiddle_kernel_test();
}

/**
 * @brief An utility function to return the test name based on the test_type
 *
 */
auto name_generator =
    [](const ::testing::TestParamInfo<std::tuple<aoclfftz_kernel_test_params_t,
       std::tuple<FFTZ_INTP, FFTZ_INTP, FFTZ_INTP, FFTZ_UINT8, FFTZ_UINT8,
       FFTZ_UINT8>>> &info)
    {
        auto param = std::get<0>(info.param);
        auto io_param = std::get<1>(info.param);
        FFTZ_INTP istride = std::get<0>(io_param);
        FFTZ_INTP ostride = std::get<1>(io_param);
        FFTZ_INTP batch_sz = std::get<2>(io_param);
        FFTZ_UINT8 is_bwd  = std::get<3>(io_param);
        FFTZ_UINT8 is_out_of_place  = std::get<4>(io_param);
        FFTZ_UINT8 load_multi_cols  = std::get<5>(io_param);
        FFTZ_UINT32 radix  = std::get<0>(param);
        FFTZ_UINT8 kernel_type = std::get<1>(param);

        // Twiddle kernels' direction is fixed by kernel_type itself (see
        // twiddle_kind_from_kernel_type()); run_twiddle_kernel_test()
        // ignores the IO tuple's `is_bwd` field for these, so name them by
        // the kernel-derived direction instead, which is what actually
        // ran. Standard/R2HC/R2HCF kernels keep using the tuple's value.
        twiddle_kind kind;
        bool is_twiddle_kernel =
            twiddle_kind_from_kernel_type(kernel_type, kind);
        if (is_twiddle_kernel)
        {
            is_bwd = twiddle_kind_is_bwd(kind);
        }

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
        // Only add BROADCAST suffix for twiddle kernels when load_multi_cols =
        // 0. Non-twiddle kernels (C2C, R2C, etc.) don't use this parameter.
        if (is_twiddle_kernel && load_multi_cols == 0)
        {
            test_name += "_BROADCAST";
        }
        test_name += get_kernel_type_as_string(kernel_type);
        return test_name;
    };

// C2C Kernels
INSTANTIATE_TEST_SUITE_P(
    C2C_C_KernelTest, AoclfftzKernelTestFloat,
    ::testing::Combine(
        ::testing::ValuesIn(filter_kernels_for_type<FFTZ_FLOAT>(param_c2c_c_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    C2C_C_KernelTest, AoclfftzKernelTestDouble,
    ::testing::Combine(
        ::testing::ValuesIn(filter_kernels_for_type<FFTZ_DOUBLE>(param_c2c_c_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);

#if defined(AVX128_SUPPORTED) && defined(ENABLE_AVX128)
INSTANTIATE_TEST_SUITE_P(
    C2C_AVX128_KernelTest, AoclfftzKernelTestFloat,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_FLOAT>(param_c2c_avx128_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    C2C_AVX128_KernelTest, AoclfftzKernelTestDouble,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_DOUBLE>(param_c2c_avx128_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);
#endif

#if defined(AVX256_SUPPORTED) && defined(ENABLE_AVX256)
INSTANTIATE_TEST_SUITE_P(
    C2C_AVX256_KernelTest, AoclfftzKernelTestFloat,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_FLOAT>(param_c2c_avx256_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    C2C_AVX256_KernelTest, AoclfftzKernelTestDouble,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_DOUBLE>(param_c2c_avx256_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);
#endif

// C2C Twiddle Kernels (Forward and Backward)
INSTANTIATE_TEST_SUITE_P(
    C2C_TWID_FWD_C_KernelTest, AoclfftzTwiddleKernelTestFloat,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_FLOAT>(param_c2c_twid_fwd_c_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    C2C_TWID_FWD_C_KernelTest, AoclfftzTwiddleKernelTestDouble,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_DOUBLE>(param_c2c_twid_fwd_c_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    C2C_TWID_BWD_C_KernelTest, AoclfftzTwiddleKernelTestFloat,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_FLOAT>(param_c2c_twid_bwd_c_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    C2C_TWID_BWD_C_KernelTest, AoclfftzTwiddleKernelTestDouble,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_DOUBLE>(param_c2c_twid_bwd_c_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);

#if defined(AVX128_SUPPORTED) && defined(ENABLE_AVX128)
INSTANTIATE_TEST_SUITE_P(
    C2C_TWID_FWD_AVX128_KernelTest, AoclfftzTwiddleKernelTestFloat,
    ::testing::Combine(
        ::testing::ValuesIn(filter_kernels_for_type<FFTZ_FLOAT>(
            param_c2c_twid_fwd_avx128_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    C2C_TWID_FWD_AVX128_KernelTest, AoclfftzTwiddleKernelTestDouble,
    ::testing::Combine(
        ::testing::ValuesIn(filter_kernels_for_type<FFTZ_DOUBLE>(
            param_c2c_twid_fwd_avx128_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    C2C_TWID_BWD_AVX128_KernelTest, AoclfftzTwiddleKernelTestFloat,
    ::testing::Combine(
        ::testing::ValuesIn(filter_kernels_for_type<FFTZ_FLOAT>(
            param_c2c_twid_bwd_avx128_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    C2C_TWID_BWD_AVX128_KernelTest, AoclfftzTwiddleKernelTestDouble,
    ::testing::Combine(
        ::testing::ValuesIn(filter_kernels_for_type<FFTZ_DOUBLE>(
            param_c2c_twid_bwd_avx128_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);

#endif

#if defined(AVX256_SUPPORTED) && defined(ENABLE_AVX256)
INSTANTIATE_TEST_SUITE_P(
    C2C_TWID_FWD_AVX256_KernelTest, AoclfftzTwiddleKernelTestFloat,
    ::testing::Combine(
        ::testing::ValuesIn(filter_kernels_for_type<FFTZ_FLOAT>(
            param_c2c_twid_fwd_avx256_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    C2C_TWID_FWD_AVX256_KernelTest, AoclfftzTwiddleKernelTestDouble,
    ::testing::Combine(
        ::testing::ValuesIn(filter_kernels_for_type<FFTZ_DOUBLE>(
            param_c2c_twid_fwd_avx256_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    C2C_TWID_BWD_AVX256_KernelTest, AoclfftzTwiddleKernelTestFloat,
    ::testing::Combine(
        ::testing::ValuesIn(filter_kernels_for_type<FFTZ_FLOAT>(
            param_c2c_twid_bwd_avx256_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    C2C_TWID_BWD_AVX256_KernelTest, AoclfftzTwiddleKernelTestDouble,
    ::testing::Combine(
        ::testing::ValuesIn(filter_kernels_for_type<FFTZ_DOUBLE>(
            param_c2c_twid_bwd_avx256_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);

#endif

// Separate io params for c2c AVX512 Kernels and r2hc AVX256 kernel onwards to
// avoid batch-sizes 8-15 repetition in c2c AVX128 & AVX256 kernels and r2hc
// AVX128 kernels
// IO params as {in-stride, out-stride , batch size, dir of FFT(0->FWD/1-> BWD),
//               result placement(0 -> inplace, 1 -> out-of-place),
//               load_multi_cols (0 -> broadcast same twiddle, 1 -> load
//               different twiddles)}
std::vector<std::tuple<FFTZ_INTP, FFTZ_INTP, FFTZ_INTP, FFTZ_UINT8, FFTZ_UINT8,
                       FFTZ_UINT8>>
    io_params_batch16 = {
        {1, 1, 1, 0, 0, 0},   {1, 1, 1, 1, 0, 1},   {1, 1, 2, 1, 0, 0},
        {5, 3, 3, 0, 1, 1},   {10, 15, 4, 1, 1, 0}, {11, 1, 5, 0, 1, 1},
        {8, 1, 6, 1, 1, 0},   {1, 7, 7, 0, 1, 1},   {1, 13, 8, 1, 1, 0},
        {7, 3, 9, 0, 1, 1},   {12, 4, 10, 1, 1, 0}, {11, 21, 11, 0, 1, 1},
        {8, 15, 12, 1, 1, 0}, {10, 5, 13, 0, 1, 1}, {5, 5, 14, 1, 0, 0},
        {15, 15, 15, 0, 0, 1}, {6, 7, 31, 1, 1, 0}};

// Separate io params for r2hc AVX512 Kernels to avoid batch-sizes 30-32
// repetition in AVX128/AVX256 kernels
// IO params as {in-stride, out-stride , batch size, dir of FFT(0->FWD/1-> BWD),
//               result placement(0 -> inplace, 1 -> out-of-place),
//               load_multi_cols (0 -> broadcast same twiddle, 1 -> load
//               different twiddles)}
// The inbetween batch size [17-30] are skipped because the flow is
// tested by other batch size
std::vector<std::tuple<FFTZ_INTP, FFTZ_INTP, FFTZ_INTP, FFTZ_UINT8, FFTZ_UINT8,
                       FFTZ_UINT8>>
    io_params_batch32 = {
        {1, 1, 1, 0, 0, 1},    {1, 1, 1, 1, 0, 1},   {1, 1, 2, 1, 0, 1},
        {5, 3, 3, 0, 1, 1},    {10, 15, 4, 1, 1, 1}, {11, 1, 5, 0, 1, 1},
        {8, 1, 6, 1, 1, 1},    {1, 7, 7, 0, 1, 1},   {1, 13, 8, 1, 1, 1},
        {7, 3, 9, 0, 1, 1},    {12, 4, 10, 1, 1, 1}, {11, 21, 11, 0, 1, 1},
        {8, 15, 12, 1, 1, 1},  {10, 5, 13, 0, 1, 1}, {5, 5, 14, 1, 0, 1},
        {15, 15, 15, 0, 0, 1}, {11, 5, 16, 1, 1, 1}, {3, 10, 31, 0, 1, 1},
        {17, 2, 32, 1, 1, 1}};

#if defined(AVX512_SUPPORTED) && defined(ENABLE_AVX512)
INSTANTIATE_TEST_SUITE_P(
    C2C_AVX512_KernelTest, AoclfftzKernelTestFloat,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_FLOAT>(param_c2c_avx512_kernels)),
        ::testing::ValuesIn(io_params_batch16)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    C2C_AVX512_KernelTest, AoclfftzKernelTestDouble,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_DOUBLE>(param_c2c_avx512_kernels)),
        ::testing::ValuesIn(io_params_batch16)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    C2C_TWID_FWD_AVX512_KernelTest, AoclfftzTwiddleKernelTestFloat,
    ::testing::Combine(
        ::testing::ValuesIn(filter_kernels_for_type<FFTZ_FLOAT>(
            param_c2c_twid_fwd_avx512_kernels)),
        ::testing::ValuesIn(io_params_batch16)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    C2C_TWID_FWD_AVX512_KernelTest, AoclfftzTwiddleKernelTestDouble,
    ::testing::Combine(
        ::testing::ValuesIn(filter_kernels_for_type<FFTZ_DOUBLE>(
            param_c2c_twid_fwd_avx512_kernels)),
        ::testing::ValuesIn(io_params_batch16)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    C2C_TWID_BWD_AVX512_KernelTest, AoclfftzTwiddleKernelTestFloat,
    ::testing::Combine(
        ::testing::ValuesIn(filter_kernels_for_type<FFTZ_FLOAT>(
            param_c2c_twid_bwd_avx512_kernels)),
        ::testing::ValuesIn(io_params_batch16)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    C2C_TWID_BWD_AVX512_KernelTest, AoclfftzTwiddleKernelTestDouble,
    ::testing::Combine(
        ::testing::ValuesIn(filter_kernels_for_type<FFTZ_DOUBLE>(
            param_c2c_twid_bwd_avx512_kernels)),
        ::testing::ValuesIn(io_params_batch16)),
    name_generator);

#endif

// C2R Twiddle Kernels
// C2R twiddle kernels implement the backward (inverse) direction only;
// run_twiddle_kernel_test() always exercises the backward reference path
// for these kernels regardless of the IO tuple's `is_bwd` field (kernel
// direction is derived from kernel_type -- see twiddle_kind_from_kernel_type()
// -- and the gtest name is generated the same way, so `is_bwd` in `io_params`
// below is a documented no-op for this suite). C2R/R2C/C2C non-AVX512
// twiddle suites all reuse the same shared `io_params` matrix since none of
// them are sensitive to its `is_bwd` field.

// C2R Twiddle - C Kernels
std::vector<aoclfftz_kernel_test_params_t> param_c2r_twid_c_kernels =
{
    {2, aocl_fftz_kernel_type::C2C_TWID_C2R_C},
    {3, aocl_fftz_kernel_type::C2C_TWID_C2R_C},
    {4, aocl_fftz_kernel_type::C2C_TWID_C2R_C},
    {5, aocl_fftz_kernel_type::C2C_TWID_C2R_C},
    {6, aocl_fftz_kernel_type::C2C_TWID_C2R_C},
    {7, aocl_fftz_kernel_type::C2C_TWID_C2R_C},
    {8, aocl_fftz_kernel_type::C2C_TWID_C2R_C},
    {9, aocl_fftz_kernel_type::C2C_TWID_C2R_C},
    {10, aocl_fftz_kernel_type::C2C_TWID_C2R_C},
    {11, aocl_fftz_kernel_type::C2C_TWID_C2R_C},
    {12, aocl_fftz_kernel_type::C2C_TWID_C2R_C},
    {13, aocl_fftz_kernel_type::C2C_TWID_C2R_C},
    {14, aocl_fftz_kernel_type::C2C_TWID_C2R_C},
    {15, aocl_fftz_kernel_type::C2C_TWID_C2R_C},
    {16, aocl_fftz_kernel_type::C2C_TWID_C2R_C},
};

INSTANTIATE_TEST_SUITE_P(
    C2R_TWID_C_KernelTest, AoclfftzC2RTwiddleKernelTestFloat,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_FLOAT>(param_c2r_twid_c_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    C2R_TWID_C_KernelTest, AoclfftzC2RTwiddleKernelTestDouble,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_DOUBLE>(param_c2r_twid_c_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);

#if defined(AVX128_SUPPORTED) && defined(ENABLE_AVX128)
// C2R Twiddle - AVX128 Kernels
std::vector<aoclfftz_kernel_test_params_t> param_c2r_twid_avx128_kernels =
{
    {2, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX128},
    {3, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX128},
    {4, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX128},
    {5, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX128},
    {6, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX128},
    {7, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX128},
    {8, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX128},
    {9, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX128},
    {10, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX128},
    {11, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX128},
    {12, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX128},
    {13, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX128},
    {14, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX128},
    {15, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX128},
    {16, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX128},
};

INSTANTIATE_TEST_SUITE_P(
    C2R_TWID_AVX128_KernelTest, AoclfftzC2RTwiddleKernelTestFloat,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_FLOAT>(param_c2r_twid_avx128_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    C2R_TWID_AVX128_KernelTest, AoclfftzC2RTwiddleKernelTestDouble,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_DOUBLE>(param_c2r_twid_avx128_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);

#endif

#if defined(AVX256_SUPPORTED) && defined(ENABLE_AVX256)
// C2R Twiddle - AVX256 Kernels
std::vector<aoclfftz_kernel_test_params_t> param_c2r_twid_avx256_kernels =
{
    {2, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX256},
    {3, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX256},
    {4, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX256},
    {5, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX256},
    {6, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX256},
    {7, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX256},
    {8, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX256},
    {9, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX256},
    {10, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX256},
    {11, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX256},
    {12, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX256},
    {13, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX256},
    {14, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX256},
    {15, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX256},
    {16, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX256},
};

INSTANTIATE_TEST_SUITE_P(
    C2R_TWID_AVX256_KernelTest, AoclfftzC2RTwiddleKernelTestFloat,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_FLOAT>(param_c2r_twid_avx256_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    C2R_TWID_AVX256_KernelTest, AoclfftzC2RTwiddleKernelTestDouble,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_DOUBLE>(param_c2r_twid_avx256_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);

#endif

#if defined(AVX512_SUPPORTED) && defined(ENABLE_AVX512)
// C2R Twiddle - AVX512 Kernels
// Reuses io_params_batch16 (same table as C2C_TWID_FWD/BWD_AVX512): its
// `is_bwd` (column 4) is a documented no-op for C2R -- see the non-AVX512
// comment above.
std::vector<aoclfftz_kernel_test_params_t> param_c2r_twid_avx512_kernels =
{
    {2, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX512},
    {3, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX512},
    {4, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX512},
    {5, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX512},
    {6, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX512},
    {7, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX512},
    {8, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX512},
    {9, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX512},
    {10, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX512},
    {11, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX512},
    {12, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX512},
    {13, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX512},
    {14, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX512},
    {15, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX512},
    {16, aocl_fftz_kernel_type::C2C_TWID_C2R_AVX512},
};

INSTANTIATE_TEST_SUITE_P(
    C2R_TWID_AVX512_KernelTest, AoclfftzC2RTwiddleKernelTestFloat,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_FLOAT>(param_c2r_twid_avx512_kernels)),
        ::testing::ValuesIn(io_params_batch16)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    C2R_TWID_AVX512_KernelTest, AoclfftzC2RTwiddleKernelTestDouble,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_DOUBLE>(param_c2r_twid_avx512_kernels)),
        ::testing::ValuesIn(io_params_batch16)),
    name_generator);

#endif

// R2C Twiddle Kernels
// R2C twiddle kernels implement the forward direction only;
// run_twiddle_kernel_test() always exercises the forward reference path for
// these kernels regardless of the IO tuple's `is_bwd` field (see the C2R
// comment above -- the same reasoning applies here). Reuses the shared
// `io_params` matrix.

// R2C Twiddle - C Kernels
std::vector<aoclfftz_kernel_test_params_t> param_r2c_twid_c_kernels =
{
    {2, aocl_fftz_kernel_type::C2C_TWID_R2C_C},
    {3, aocl_fftz_kernel_type::C2C_TWID_R2C_C},
    {4, aocl_fftz_kernel_type::C2C_TWID_R2C_C},
    {5, aocl_fftz_kernel_type::C2C_TWID_R2C_C},
    {6, aocl_fftz_kernel_type::C2C_TWID_R2C_C},
    {7, aocl_fftz_kernel_type::C2C_TWID_R2C_C},
    {8, aocl_fftz_kernel_type::C2C_TWID_R2C_C},
    {9, aocl_fftz_kernel_type::C2C_TWID_R2C_C},
    {10, aocl_fftz_kernel_type::C2C_TWID_R2C_C},
    {11, aocl_fftz_kernel_type::C2C_TWID_R2C_C},
    {12, aocl_fftz_kernel_type::C2C_TWID_R2C_C},
    {13, aocl_fftz_kernel_type::C2C_TWID_R2C_C},
    {14, aocl_fftz_kernel_type::C2C_TWID_R2C_C},
    {15, aocl_fftz_kernel_type::C2C_TWID_R2C_C},
    {16, aocl_fftz_kernel_type::C2C_TWID_R2C_C},
};

INSTANTIATE_TEST_SUITE_P(
    R2C_TWID_C_KernelTest, AoclfftzR2CTwiddleKernelTestFloat,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_FLOAT>(param_r2c_twid_c_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    R2C_TWID_C_KernelTest, AoclfftzR2CTwiddleKernelTestDouble,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_DOUBLE>(param_r2c_twid_c_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);

#if defined(AVX128_SUPPORTED) && defined(ENABLE_AVX128)
// R2C Twiddle - AVX128 Kernels
std::vector<aoclfftz_kernel_test_params_t> param_r2c_twid_avx128_kernels =
{
    {2, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX128},
    {3, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX128},
    {4, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX128},
    {5, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX128},
    {6, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX128},
    {7, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX128},
    {8, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX128},
    {9, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX128},
    {10, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX128},
    {11, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX128},
    {12, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX128},
    {13, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX128},
    {14, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX128},
    {15, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX128},
    {16, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX128},
};

INSTANTIATE_TEST_SUITE_P(
    R2C_TWID_AVX128_KernelTest, AoclfftzR2CTwiddleKernelTestFloat,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_FLOAT>(param_r2c_twid_avx128_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    R2C_TWID_AVX128_KernelTest, AoclfftzR2CTwiddleKernelTestDouble,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_DOUBLE>(param_r2c_twid_avx128_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);

#endif

#if defined(AVX256_SUPPORTED) && defined(ENABLE_AVX256)
// R2C Twiddle - AVX256 Kernels
std::vector<aoclfftz_kernel_test_params_t> param_r2c_twid_avx256_kernels =
{
    {2, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX256},
    {3, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX256},
    {4, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX256},
    {5, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX256},
    {6, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX256},
    {7, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX256},
    {8, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX256},
    {9, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX256},
    {10, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX256},
    {11, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX256},
    {12, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX256},
    {13, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX256},
    {14, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX256},
    {15, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX256},
    {16, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX256},
};

INSTANTIATE_TEST_SUITE_P(
    R2C_TWID_AVX256_KernelTest, AoclfftzR2CTwiddleKernelTestFloat,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_FLOAT>(param_r2c_twid_avx256_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    R2C_TWID_AVX256_KernelTest, AoclfftzR2CTwiddleKernelTestDouble,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_DOUBLE>(param_r2c_twid_avx256_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);

#endif

#if defined(AVX512_SUPPORTED) && defined(ENABLE_AVX512)
// R2C Twiddle - AVX512 Kernels
// Reuses io_params_batch16 (same table as C2C_TWID_FWD/BWD_AVX512): its
// `is_bwd` (column 4) is a documented no-op for R2C -- see the non-AVX512
// comment above.
std::vector<aoclfftz_kernel_test_params_t> param_r2c_twid_avx512_kernels =
{
    {2, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX512},
    {3, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX512},
    {4, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX512},
    {5, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX512},
    {6, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX512},
    {7, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX512},
    {8, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX512},
    {9, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX512},
    {10, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX512},
    {11, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX512},
    {12, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX512},
    {13, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX512},
    {14, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX512},
    {15, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX512},
    {16, aocl_fftz_kernel_type::C2C_TWID_R2C_AVX512},
};

INSTANTIATE_TEST_SUITE_P(
    R2C_TWID_AVX512_KernelTest, AoclfftzR2CTwiddleKernelTestFloat,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_FLOAT>(param_r2c_twid_avx512_kernels)),
        ::testing::ValuesIn(io_params_batch16)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    R2C_TWID_AVX512_KernelTest, AoclfftzR2CTwiddleKernelTestDouble,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_DOUBLE>(param_r2c_twid_avx512_kernels)),
        ::testing::ValuesIn(io_params_batch16)),
    name_generator);

#endif

// R2HC Kernels
INSTANTIATE_TEST_SUITE_P(
    R2HC_C_KernelTest, AoclfftzKernelTestFloatReal,
    ::testing::Combine(
        ::testing::ValuesIn(filter_kernels_for_type<FFTZ_FLOAT>(param_r2hc_c_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    R2HC_C_KernelTest, AoclfftzKernelTestDoubleReal,
    ::testing::Combine(
        ::testing::ValuesIn(filter_kernels_for_type<FFTZ_DOUBLE>(param_r2hc_c_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);

// R2HC-Fused Kernels
INSTANTIATE_TEST_SUITE_P(
    R2HCF_C_KernelTest, AoclfftzKernelTestFloatFused,
    ::testing::Combine(
        ::testing::ValuesIn(filter_kernels_for_type<FFTZ_FLOAT>(param_r2hcf_c_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    R2HCF_C_KernelTest, AoclfftzKernelTestDoubleFused,
    ::testing::Combine(
        ::testing::ValuesIn(filter_kernels_for_type<FFTZ_DOUBLE>(param_r2hcf_c_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);

#if defined(AVX128_SUPPORTED) && defined(ENABLE_AVX128)
// R2HC AVX128 Kernels
INSTANTIATE_TEST_SUITE_P(
    R2HC_AVX128_KernelTest, AoclfftzKernelTestFloatReal,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_FLOAT>(param_r2hc_avx128_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    R2HC_AVX128_KernelTest, AoclfftzKernelTestDoubleReal,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_DOUBLE>(param_r2hc_avx128_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);

// R2HC-Fused AVX128 Kernels
INSTANTIATE_TEST_SUITE_P(
    R2HCF_AVX128_KernelTest, AoclfftzKernelTestFloatFused,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_FLOAT>(param_r2hcf_avx128_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    R2HCF_AVX128_KernelTest, AoclfftzKernelTestDoubleFused,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_DOUBLE>(param_r2hcf_avx128_kernels)),
        ::testing::ValuesIn(io_params)),
    name_generator);
#endif

#if defined(AVX256_SUPPORTED) && defined(ENABLE_AVX256)
// R2HC AVX256 Kernels
INSTANTIATE_TEST_SUITE_P(
    R2HC_AVX256_KernelTest, AoclfftzKernelTestFloatReal,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_FLOAT>(param_r2hc_avx256_kernels)),
        ::testing::ValuesIn(io_params_batch16)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    R2HC_AVX256_KernelTest, AoclfftzKernelTestDoubleReal,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_DOUBLE>(param_r2hc_avx256_kernels)),
        ::testing::ValuesIn(io_params_batch16)),
    name_generator);

// R2HC-Fused AVX256 Kernels
INSTANTIATE_TEST_SUITE_P(
    R2HCF_AVX256_KernelTest, AoclfftzKernelTestFloatFused,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_FLOAT>(param_r2hcf_avx256_kernels)),
        ::testing::ValuesIn(io_params_batch16)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    R2HCF_AVX256_KernelTest, AoclfftzKernelTestDoubleFused,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_DOUBLE>(param_r2hcf_avx256_kernels)),
        ::testing::ValuesIn(io_params_batch16)),
    name_generator);
#endif

#if defined(AVX512_SUPPORTED) && defined(ENABLE_AVX512)
// R2HC AVX512 Kernels
INSTANTIATE_TEST_SUITE_P(
    R2HC_AVX512_KernelTest, AoclfftzKernelTestFloatReal,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_FLOAT>(param_r2hc_avx512_kernels)),
        ::testing::ValuesIn(io_params_batch32)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    R2HC_AVX512_KernelTest, AoclfftzKernelTestDoubleReal,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_DOUBLE>(param_r2hc_avx512_kernels)),
        ::testing::ValuesIn(io_params_batch32)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    R2HCF_AVX512_KernelTest, AoclfftzKernelTestFloatFused,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_FLOAT>(param_r2hcf_avx512_kernels)),
        ::testing::ValuesIn(io_params_batch32)),
    name_generator);

INSTANTIATE_TEST_SUITE_P(
    R2HCF_AVX512_KernelTest, AoclfftzKernelTestDoubleFused,
    ::testing::Combine(
        ::testing::ValuesIn(
            filter_kernels_for_type<FFTZ_DOUBLE>(param_r2hcf_avx512_kernels)),
        ::testing::ValuesIn(io_params_batch32)),
    name_generator);
#endif
