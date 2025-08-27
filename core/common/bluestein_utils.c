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

/** @file bluestein_utils.c
 *
 *  @brief Contains utility functions related to Bluestein solver.
 *
 *  This file contains the functions for computing extended length, computing
 *  bluestein sequence, elementwise multiplication and normalizing the data.
 *
 *  @author Srirammaswamy Srinivasan
 */

#include <math.h>
#include <string.h> /* for memset */
#include "core/common/bluestein_utils.h"

INTP get_extended_length(INTP n)
{
    INTP m = 2 * n - 1;
    // check if all the factors of m are within the supported kernels' range
    // i.e. prime numbers in range 2 to 16
    // if not, adjust the m to a nearest larger number
    // which satisfies the above condition

    // TODO: Move this list to a common file to be used across
    // library and selector gtest
    INTP supported_primes[] = {2, 3, 5, 7, 11, 13};
    UINT32 prime_count = sizeof(supported_primes) / sizeof(supported_primes[0]);
    for (INTP next_m = m, quo = 0; quo != 1; next_m++)
    {
        quo = m = next_m;
        for (UINT32 i = 0; i < prime_count; i++)
        {
            while (quo % supported_primes[i] == 0)
            {
                quo /= supported_primes[i];
            }
            if (quo == 1)
            {
                break; // solvable m value
            }
        }
    }
    return m;
}

INT32 prepare_bluestein_sequence(aoclfftz_solution_t *sol, INTP m)
{
    /*            Bluestein sequence B of length m
        <------ (n) -----><-- (m-2n-1) --><----- (n-1) ----->
        |     values      |     zeros     | reversed values |
                                                                */
    UINT32 precision = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    INTP n = sol->decomp_scheme->dims[0].n;
    INTP n2 = n << 1;

    if (precision == DT_FLOAT)
    {
        FLOAT *B = (FLOAT *)sol->dft_bufs->bluestein->B;
        for (INTP i = 0; i < n; i++)
        {
            // Handle overflow to improve accuracy for larger values
            INTP m = (i * i) % n2;
            FLOAT angle = (AOCLFFTZ_2_PIf * m) / n2;
            B[i * DATA_STRIDE] = cosf(angle);
            B[i * DATA_STRIDE + 1] = sinf(angle);
        }
        memset(B + n * DATA_STRIDE, 0,
               (m - n - 1) * DATA_STRIDE * sizeof(FLOAT));
        for (INTP i = 1; i < n; i++)
        {
            B[(m - i) * DATA_STRIDE] = B[i * DATA_STRIDE];
            B[(m - i) * DATA_STRIDE + 1] = B[i * DATA_STRIDE + 1];
        }
    }
    else
    {
        DOUBLE *B = (DOUBLE *)sol->dft_bufs->bluestein->B;
        for (INTP i = 0; i < n; i++)
        {
            // Handle overflow to improve accuracy for larger values
            INTP m = (i * i) % n2;
            DOUBLE angle = (AOCLFFTZ_2_PI * m) / n2;
            B[i * DATA_STRIDE] = cos(angle);
            B[i * DATA_STRIDE + 1] = sin(angle);
        }
        memset(B + n * DATA_STRIDE, 0,
               (m - n - 1) * DATA_STRIDE * sizeof(DOUBLE));
        for (INTP i = 1; i < n; i++)
        {
            B[(m - i) * DATA_STRIDE] = B[i * DATA_STRIDE];
            B[(m - i) * DATA_STRIDE + 1] = B[i * DATA_STRIDE + 1];
        }
    }

    return BLUESTEIN_SUCCESS;
}

INT32 elementwise_multiplication(VOID *out, VOID *a, VOID *b, INTP n,
                                 UINT8 sign, UINT8 precision)
{
    if (precision == DT_FLOAT)
    {
        FLOAT *a_f = (FLOAT *)a;
        FLOAT *b_f = (FLOAT *)b;
        FLOAT *out_f = (FLOAT *)out;
        FLOAT temp[DATA_STRIDE];
        // Complex multiplication of a and b
        // out.re = (a.re * b.re) - (a.im * b.im)
        // out.im = (a.re * b.im) + (a.im * b.re)
        if (!sign)
        {
            for (INTP i = 0; i < n * DATA_STRIDE; i += DATA_STRIDE)
            {
                temp[0] = (a_f[i] * b_f[i]) - (a_f[i + 1] * b_f[i + 1]);
                temp[1] = (a_f[i] * b_f[i + 1]) + (a_f[i + 1] * b_f[i]);
                out_f[i]     = temp[0];
                out_f[i + 1] = temp[1];
            }
        }
        // Complex multiplication with b.im sign reversed
        // out.re =  (a.re * b.re) + (a.im * b.im)
        // out.im = -(a.re * b.im) + (a.im * b.re)
        else
        {
            for (INTP i = 0; i < n * DATA_STRIDE; i += DATA_STRIDE)
            {
                temp[0] =  (a_f[i] * b_f[i]) + (a_f[i + 1] * b_f[i + 1]);
                temp[1] = -(a_f[i] * b_f[i + 1]) + (a_f[i + 1] * b_f[i]);
                out_f[i]     = temp[0];
                out_f[i + 1] = temp[1];
            }
        }
    }
    else
    {
        DOUBLE *a_d = (DOUBLE *)a;
        DOUBLE *b_d = (DOUBLE *)b;
        DOUBLE *out_d = (DOUBLE *)out;
        DOUBLE temp[DATA_STRIDE];
        // Complex multiplication of a and b
        // out.re = (a.re * b.re) - (a.im * b.im)
        // out.im = (a.re * b.im) + (a.im * b.re)
        if (!sign)
        {
            for (INTP i = 0; i < n * DATA_STRIDE; i += DATA_STRIDE)
            {
                temp[0] = (a_d[i] * b_d[i]) - (a_d[i + 1] * b_d[i + 1]);
                temp[1] = (a_d[i] * b_d[i + 1]) + (a_d[i + 1] * b_d[i]);
                out_d[i]     = temp[0];
                out_d[i + 1] = temp[1];
            }
        }
        // Complex multiplication with b.im sign reversed
        // out.re =  (a.re * b.re) + (a.im * b.im)
        // out.im = -(a.re * b.im) + (a.im * b.re)
        else
        {
            for (INTP i = 0; i < n * DATA_STRIDE; i += DATA_STRIDE)
            {
                temp[0] =  (a_d[i] * b_d[i]) + (a_d[i + 1] * b_d[i + 1]);
                temp[1] = -(a_d[i] * b_d[i + 1]) + (a_d[i + 1] * b_d[i]);
                out_d[i]     = temp[0];
                out_d[i + 1] = temp[1];
            }
        }
    }

    return BLUESTEIN_SUCCESS;
}

INT32 normalize_data(VOID *data, INTP n, DOUBLE normalize_factor,
                     UINT8 precision)
{
    if (precision == DT_FLOAT)
    {
        FLOAT *data_f = (FLOAT *)data;
        for (INTP i = 0; i < n * DATA_STRIDE; i++)
        {
            data_f[i] *= (FLOAT)normalize_factor;
        }
    }
    else
    {
        DOUBLE *data_d = (DOUBLE *)data;
        for (INTP i = 0; i < n * DATA_STRIDE; i++)
        {
            data_d[i] *= normalize_factor;
        }
    }

    return BLUESTEIN_SUCCESS;
}
