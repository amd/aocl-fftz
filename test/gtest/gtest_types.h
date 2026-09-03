// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

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

#ifdef MULTI_THREADING
#include <omp.h>
#endif

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
    C2C_TWID_FWD_C,
    C2C_TWID_FWD_AVX128,
    C2C_TWID_FWD_AVX256,
    C2C_TWID_FWD_AVX512,
    C2C_TWID_BWD_C,
    C2C_TWID_BWD_AVX128,
    C2C_TWID_BWD_AVX256,
    C2C_TWID_BWD_AVX512,
    R2HC_C,
    R2HCF_C,
    R2HC_AVX128,
    R2HCF_AVX128,
    R2HC_AVX256,
    R2HCF_AVX256,
    R2HC_AVX512,
    R2HCF_AVX512,
    C2C_TWID_R2C_C,
    C2C_TWID_R2C_AVX128,
    C2C_TWID_R2C_AVX256,
    C2C_TWID_R2C_AVX512,
    C2C_TWID_C2R_C,
    C2C_TWID_C2R_AVX128,
    C2C_TWID_C2R_AVX256,
    C2C_TWID_C2R_AVX512,
};

/**
 * @brief A type used as the parameter for the parameterized tests (TEST_P)
 *
 * std::tuple<radix, aocl_fftz_kernel_type>
 * used FFTZ_UINT8 instead of aocl_fftz_kernel_type to perform bitwise
 * operations to make decisions.
 */
typedef std::tuple<FFTZ_UINT32, FFTZ_UINT8> aoclfftz_kernel_test_params_t;

/**
 * @brief A type used as the parameter for the paraterized selector tests
 * (TEST_P)
 *
 */
typedef std::tuple<std::string, FFTZ_INT32, FFTZ_INT32,
                   std::vector<aoclfftz_solver_type>>
    aoclfftz_selector_test_params_t;

// function pointer to dft_solver
typedef FFTZ_INT32 (*dft_solver_)(aoclfftz_solution_t *sol,
                                  aoclfftz_mutable_ctx_t *ctx);

#endif // AOCLFFTZ_GTEST_TYPES_H
