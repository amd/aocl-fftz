
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
/** @file data_generation.h
 *
 *  @brief Input data generation utility functions.
 *
 *  This file contains the utility functions and macros for input data
 *  generation for test bench.
 *
 *  @author V. Murugan
 *  @author Srirammaswamy Srinivasan
 */

#ifndef DATA_GENERATION_H
#define DATA_GENERATION_H

#include "api/types.h"
#include "utils/complex_utils.h"

#define PREPARE_LINEAR_TEST_INPUTS(in1, in2, in_combined, size, factors,       \
                                   precision)                                  \
    {                                                                          \
        if (precision == FLOAT_P)                                              \
        {                                                                      \
            PREPARE_LINEAR_TEST_INPUTS_IMPL(in1, in2, in_combined, size,       \
                                            factors, FLOAT);                   \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            PREPARE_LINEAR_TEST_INPUTS_IMPL(in1, in2, in_combined, size,       \
                                            factors, DOUBLE);                  \
        }                                                                      \
    }

#define PREPARE_LINEAR_TEST_INPUTS_IMPL(in1, in2, in_combined, size, factors,  \
                                        dt_t)                                  \
    dt_t *in1_t = (dt_t *)in1;                                                 \
    dt_t *in2_t = (dt_t *)in2;                                                 \
    dt_t *in_combined_t = (dt_t *)in_combined;                                 \
    dt_t *factors_t = (dt_t *)factors;                                         \
    factors_t[0] = (dt_t)((rand() % 200) / 20.0 - 10.0);                       \
    factors_t[1] = 0.0;                                                        \
    factors_t[2] = (dt_t)((rand() % 200) / 20.0 - 10.0);                       \
    factors_t[3] = 0.0;                                                        \
    dt_t temp1[T_DATA_STRIDE] = {0.0, 0.0};                                    \
    dt_t temp2[T_DATA_STRIDE] = {0.0, 0.0};                                    \
    dt_t cmul_temp[T_DATA_STRIDE] = {0.0, 0.0};                                \
    for (INTP idx = 0; idx < size; ++idx)                                      \
    {                                                                          \
        CMUL(factors_t, in1_t + idx * T_DATA_STRIDE, temp1, cmul_temp);        \
        CMUL(factors_t + 2, in2_t + idx * T_DATA_STRIDE, temp2, cmul_temp);    \
        CADD(temp1, temp2, in_combined_t + idx * T_DATA_STRIDE);               \
    }

#define PREPARE_LINEAR_TEST_OUTPUTS(out1, out2, out_added, size, factors,      \
                                    precision)                                 \
    {                                                                          \
        if (precision == FLOAT_P)                                              \
        {                                                                      \
            PREPARE_LINEAR_TEST_OUTPUTS_IMPL(out1, out2, out_added, size,      \
                                             factors, FLOAT);                  \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            PREPARE_LINEAR_TEST_OUTPUTS_IMPL(out1, out2, out_added, size,      \
                                             factors, DOUBLE);                 \
        }                                                                      \
    }

#define PREPARE_LINEAR_TEST_OUTPUTS_IMPL(out1, out2, out_added, size, factors, \
                                         dt_t)                                 \
    {                                                                          \
        dt_t *out1_t = (dt_t *)out1;                                           \
        dt_t *out2_t = (dt_t *)out2;                                           \
        dt_t *out_added_t = (dt_t *)out_added;                                 \
        dt_t *factors_t = (dt_t *)factors;                                     \
        dt_t temp1[T_DATA_STRIDE] = {0.0, 0.0};                                \
        dt_t temp2[T_DATA_STRIDE] = {0.0, 0.0};                                \
        dt_t cmul_temp[T_DATA_STRIDE] = {0.0, 0.0};                            \
        for (INTP idx = 0; idx < size; ++idx)                                  \
        {                                                                      \
            CMUL(factors_t, out1_t + idx * T_DATA_STRIDE, temp1, cmul_temp);   \
            CMUL(factors_t + 2, out2_t + idx * T_DATA_STRIDE, temp2,           \
                 cmul_temp);                                                   \
            CADD(temp1, temp2, out_added_t + idx * T_DATA_STRIDE);             \
        }                                                                      \
    }

#define PREPARE_TIMESHIFT_TEST_INPUTS(in1, in2, n, m, imap, precision)         \
    {                                                                          \
        if (precision == FLOAT_P)                                              \
        {                                                                      \
            PREPARE_TIMESHIFT_TEST_INPUTS_IMPL(in1, in2, n, m, imap, FLOAT);   \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            PREPARE_TIMESHIFT_TEST_INPUTS_IMPL(in1, in2, n, m, imap, DOUBLE);  \
        }                                                                      \
    }

#define PREPARE_TIMESHIFT_TEST_INPUTS_IMPL(in1, in2, n, m, imap, dt_t)         \
    {                                                                          \
        dt_t *in1_t = (dt_t *)in1;                                             \
        dt_t *in2_t = (dt_t *)in2;                                             \
        /* Handle overflow to avoid negative indexing */                       \
        for (INTP idx = 0; idx < n; idx++)                                     \
        {                                                                      \
            INTP src = imap[(idx + (n - m)) % n] * T_DATA_STRIDE;              \
            INTP dst = imap[idx] * T_DATA_STRIDE;                              \
            in2_t[dst] = in1_t[src];                                           \
            in2_t[dst + 1] = in1_t[src + 1];                                   \
        }                                                                      \
    }

#define PREPARE_TIMESHIFT_TEST_OUTPUTS(out1, out_combined, n, m, unit_m, omap, \
                                        dir, precision)                        \
    {                                                                          \
        if (precision == FLOAT_P)                                              \
        {                                                                      \
            PREPARE_TIMESHIFT_TEST_OUTPUTS_IMPL(out1, out_combined, n, m,      \
                                                unit_m, omap, dir, FLOAT);     \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            PREPARE_TIMESHIFT_TEST_OUTPUTS_IMPL(out1, out_combined, n, m,      \
                                                unit_m, omap, dir, DOUBLE);    \
        }                                                                      \
    }

#define PREPARE_TIMESHIFT_TEST_OUTPUTS_IMPL(out1, out_combined, n, m, unit_m,  \
                                            omap, dir, dt_t)                   \
    {                                                                          \
        dt_t *out1_t = (dt_t *)out1;                                           \
        dt_t *out_combined_t = (dt_t *)out_combined;                           \
        dt_t cmul_temp[T_DATA_STRIDE] = {0.0, 0.0};                            \
        dt_t e_k[T_DATA_STRIDE] = {1.0, 0.0};                                  \
        dt_t sign = (dir == FORWARD) ? -1.0 : 1.0;                             \
        for (INTP k = 0; k < n; k++)                                           \
        {                                                                      \
            /* Handle overflow to improve accuracy for larger values */        \
            INTP mk = (m * k) % n;                                             \
            dt_t angle = (sign * BENCH_2_PI * mk / n);                         \
            EULER(angle, e_k);                                                 \
            for (INTP i = 0; i < unit_m; i++)                                  \
            {                                                                  \
                CMUL(out1_t + omap[(k*unit_m+i)] * T_DATA_STRIDE, e_k,         \
                     out_combined_t + omap[(k*unit_m+i)] * T_DATA_STRIDE,      \
                     cmul_temp);                                               \
            }                                                                  \
        }                                                                      \
    }

#define NORMALIZE_IFFT_DATA(arr, length, n, precision)                         \
    {                                                                          \
        if (precision == FLOAT_P)                                              \
        {                                                                      \
            NORMALIZE_IFFT_DATA_IMPL(arr, length, n, FLOAT);                   \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            NORMALIZE_IFFT_DATA_IMPL(arr, length, n, DOUBLE);                  \
        }                                                                      \
    }

#define NORMALIZE_IFFT_DATA_IMPL(arr, length, n, dt_t)                         \
    {                                                                          \
        dt_t *arr_t = (dt_t *)arr;                                             \
        for (INTP idx = 0; idx < length; ++idx)                                \
        {                                                                      \
            arr_t[idx * T_DATA_STRIDE] /= n;                                   \
            arr_t[idx * T_DATA_STRIDE + 1] /= n;                               \
        }                                                                      \
    }

VOID prepare_input_data_f(VOID *input, INTP n, INTP *idx_map, INT32 input_type);
VOID prepare_input_data_d(VOID *input, INTP n, INTP *idx_map, INT32 input_type);

#endif // DATA_GENERATION_H
