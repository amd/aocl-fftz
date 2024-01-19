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
 * @brief load a single complex number from `res` to `input`
 */
#define LOAD_INPUT(in_r, in_i, res)                                            \
    {                                                                          \
        (res)[0] = (in_r)[0];                                                  \
        (res)[1] = (in_i)[0];                                                  \
    }

/**
 * @brief load a single complex number from `res` to `output`
 */
#define STORE_OUTPUT(res, out_r, out_i)                                        \
    {                                                                          \
        (out_r)[0] = (res)[0];                                                 \
        (out_i)[0] = (res)[1];                                                 \
    }

/**
 * @brief complex addition: res = c1 + c2
 *
 */
#define CADD(c1, c2, res)                                                      \
    {                                                                          \
        (res)[0] = (c1)[0] + (c2)[0];                                          \
        (res)[1] = (c1)[1] + (c2)[1];                                          \
    }

/**
 * @brief complex subtraction: res = c1 - c2
 *
 */
#define CSUB(c1, c2, res)                                                      \
    {                                                                          \
        (res)[0] = (c1)[0] - (c2)[0];                                          \
        (res)[1] = (c1)[1] - (c2)[1];                                          \
    }

/**
 * @brief complex multiplication: res = c1 * c2 (NOTE: mtemp is used to store
 * the temporary result)
 *
 */
#define CMUL(c1, c2, res, mtemp)                                               \
    {                                                                          \
        (mtemp)[0] = (c1)[0] * (c2)[0] - (c1)[1] * (c2)[1];                    \
        (mtemp)[1] = (c1)[0] * (c2)[1] + (c1)[1] * (c2)[0];                    \
        (res)[0] = (mtemp)[0];                                                 \
        (res)[1] = (mtemp)[1];                                                 \
    }

/**
 * @brief power of a complex number: res = c^p (NOTE: mtemp and ptemp are used
 * to store the temporary results of multiply and power operations. Here, power
 * is achieved using repeated multiplication)
 *
 */
#define CPOW(c, p, res, mtemp, ptemp)                                          \
    {                                                                          \
        int _p = p;                                                            \
        (ptemp)[0] = 1.0;                                                      \
        (ptemp)[1] = 0.0;                                                      \
        while (_p--)                                                           \
        {                                                                      \
            CMUL(ptemp, c, ptemp, mtemp);                                      \
        }                                                                      \
        (res)[0] = (ptemp)[0];                                                 \
        (res)[1] = (ptemp)[1];                                                 \
    }

/**
 * @brief A wrapper of CMUL, CADD and CPOW functions
 *
 */
#define CMUL_CADD_CPOW(in, tf, pow_idx, temp1, temp2, temp_mul, temp_CPOW)     \
    {                                                                          \
        CMUL(in, temp1, temp1, temp_mul);                                      \
        CADD(temp2, temp1, temp2);                                             \
        CPOW(tf, pow_idx, temp1, temp_mul, temp_CPOW);                         \
    }

/**
 * @brief A wrapper of CMUL, CADD functions
 *
 */
#define CMUL_CADD(in, pow_tf, temp_CPOW, res, temp_mul)                        \
    {                                                                          \
        CMUL(in, pow_tf, temp_CPOW, temp_mul);                                 \
        CADD(res, temp_CPOW, res);                                             \
    }

/**
 * @brief euler's formula: res = exp(ix), i.e. res = cos(x) + i sin(x)
 *
 */
#define EULER(x, res)                                                          \
    {                                                                          \
        (res)[0] = cos(x);                                                     \
        (res)[1] = sin(x);                                                     \
    }

/**
 * @brief prints the float complex array `arr` of length `length`
 *
 */
#define PRINT_CARRAY_FP32(arr, size)                                           \
    {                                                                          \
        FLOAT *arr_f = (FLOAT *)arr;                                           \
        for (int i = 0; i < size; ++i)                                         \
        {                                                                      \
            printf("%td: %12.6f + %12.6fj\n", (INTP)i, arr_f[i * DATA_STRIDE], \
                   arr_f[i * DATA_STRIDE + 1]);                                \
        }                                                                      \
        printf("\n");                                                          \
    }

/**
 * @brief prints the double complex array `arr` of size `length`
 *
 */
#define PRINT_CARRAY_FP64(arr, size)                                           \
    {                                                                          \
        DOUBLE *arr_d = (DOUBLE *)arr;                                         \
        for (int i = 0; i < size; ++i)                                         \
        {                                                                      \
            printf("%td: %20.14lf + %20.14lfj\n", (INTP)i,                     \
                   arr_d[i * DATA_STRIDE], arr_d[i * DATA_STRIDE + 1]);        \
        }                                                                      \
        printf("\n");                                                          \
    }

/**
 * @brief prints a single float complex value `val`
 *
 */
#define PRINT_COMPLEX_FP32(val)                                                \
    {                                                                          \
        printf("%12.6f + %12.6fj", (val)[0], (val)[1]);                        \
    }

/**
 * @brief prints a single float complex value `val`
 *
 */
#define PRINT_COMPLEX_FP64(val)                                                \
    {                                                                          \
        printf("%20.14lf + %20.14lfj", (val)[0], (val)[1]);                    \
    }

#endif // AOCLFFTZ_COMPLEX_UTILS_H
