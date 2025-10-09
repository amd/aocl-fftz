/**
 * Copyright (C) 2023-2025, Advanced Micro Devices. All rights reserved.
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

/** @file gtest_types.h
 *
 *  @brief Enums and typedef declarations used for GTest.
 *
 *  This file contains the enums for test input, test type and kernel type; and
 * typedef declaration for test param.
 *
 *  @author Srirammaswamy Srinivasan
 */

#ifndef AOCLFFTZ_GTEST_TYPES_H
#define AOCLFFTZ_GTEST_TYPES_H

#include <tuple>
extern "C"
{
#include "core/solvers/solver.h"
}

/**
 * @brief An enum for the supported input types for tests
 * RANDOM: generates random input for all points
 * RANDOM_SPECIAL: generates random input with special values like NaN,
 * infinity, sub-normals and with floating-point min/max values
 *
 */
enum aocl_fftz_test_input
{
    RANDOM,
    RANDOM_SPECIAL
};

/**
 * @brief An enum for the supported kernel types
 *
 */
enum aocl_fftz_kernel_type
{
    C2C_C = 0,
    C2C_AVX128,
    C2C_AVX256,
    C2C_AVX512,
    C2C_TWID_C,
    C2C_TWID_AVX128,
    C2C_TWID_AVX256,
    C2C_TWID_AVX512,
    R2HC_C,
    R2HCF_C,
    R2HC_AVX128,
    R2HCF_AVX128,
    R2HC_AVX256,
    R2HCF_AVX256,
    R2HC_AVX512,
    R2HCF_AVX512,
};

/**
 * @brief A type used as the parameter for the parameterized tests (TEST_P)
 *
 * std::tuple<radix, aocl_fftz_kernel_type>
 * used UINT8 instead of aocl_fftz_kernel_type to perform bitwise operations
 * to make decisions.
 */
typedef std::tuple<UINT32, UINT8> aoclfftz_kernel_test_params_t;

/**
 * @brief A type used as the parameter for the paraterized selector tests
 * (TEST_P)
 *
 */
typedef std::tuple<std::string, INT32, INT32, std::vector<aoclfftz_solver_type>>
    aoclfftz_selector_test_params_t;

// function pointer to dft_solver
typedef INT32 (*dft_solver_)(aoclfftz_solution_t *sol);

#endif // AOCLFFTZ_GTEST_TYPES_H
