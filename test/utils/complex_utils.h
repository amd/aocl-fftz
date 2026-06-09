// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file complex_utils.h
 *
 *  @brief Utility functions for complex operations.
 *
 *  This file contains the utility functions (function-like macros) used to do
 *  complex arithmetic and to print complex numbers.
 *
 *  @author Srirammaswamy Srinivasan
 *
 */

#ifndef AOCLFFTZ_COMPLEX_UTILS_H
#define AOCLFFTZ_COMPLEX_UTILS_H

#include <math.h>
#include <stdio.h>

/*************** MACRO FUNCTIONS ***************/
/**
 * @brief complex addition: res = c1 + c2
 *
 */
#define CADD(c1, c2, res)                                                      \
{                                                                              \
    (res)[0] = (c1)[0] + (c2)[0];                                              \
    (res)[1] = (c1)[1] + (c2)[1];                                              \
}

/**
 * @brief complex multiplication: res = c1 * c2 (NOTE: mtemp is used to store
 * the temporary result)
 *
 */
#define CMUL(c1, c2, res, mtemp)                                               \
{                                                                              \
    (mtemp)[0] = (c1)[0] * (c2)[0] - (c1)[1] * (c2)[1];                        \
    (mtemp)[1] = (c1)[0] * (c2)[1] + (c1)[1] * (c2)[0];                        \
    (res)[0] = (mtemp)[0];                                                     \
    (res)[1] = (mtemp)[1];                                                     \
}

/**
 * @brief euler's formula: res = exp(ix), i.e. res = cos(x) + i sin(x)
 *
 */
#define EULER(x, res)                                                          \
{                                                                              \
    (res)[0] = cos(x);                                                         \
    (res)[1] = sin(x);                                                         \
}

/**
 * @brief prints the float complex array `arr` of length `length`
 *
 */
#define PRINT_CARRAY_FP32(arr, size)                                           \
{                                                                              \
    FLOAT *arr_f = (FLOAT *)arr;                                               \
    for (INTP i = 0; i < size; ++i)                                            \
    {                                                                          \
        printf("%td: %12.6f + %12.6fj\n", i, arr_f[i * 2],                     \
               arr_f[i * 2 + 1]);                                              \
    }                                                                          \
    printf("\n");                                                              \
}

/**
 * @brief prints the double complex array `arr` of size `length`
 *
 */
#define PRINT_CARRAY_FP64(arr, size)                                           \
{                                                                              \
    DOUBLE *arr_d = (DOUBLE *)arr;                                             \
    for (INTP i = 0; i < size; ++i)                                            \
    {                                                                          \
        printf("%td: %20.14lf + %20.14lfj\n", i, arr_d[i * 2],                 \
               arr_d[i * 2 + 1]);                                              \
    }                                                                          \
    printf("\n");                                                              \
}

/**
 * @brief prints a single float complex value `val`
 *
 */
#define PRINT_COMPLEX_FP32(val)                                                \
{                                                                              \
    printf("%12.6f + %12.6fj", (val)[0], (val)[1]);                            \
}

/**
 * @brief prints a single float complex value `val`
 *
 */
#define PRINT_COMPLEX_FP64(val)                                                \
{                                                                              \
    printf("%20.14lf + %20.14lfj", (val)[0], (val)[1]);                        \
}

#endif // AOCLFFTZ_COMPLEX_UTILS_H
