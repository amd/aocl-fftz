
// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause
/** @file data_generation.h
 *
 *  @brief Input data generation utility functions.
 *
 *  This file contains the utility functions and macros for input data
 *  generation for test bench.
 *
 *  @author V. Murugan
 *  @author Srirammaswamy Srinivasan
 *  @author Jeya R
 */

#ifndef DATA_GENERATION_H
#define DATA_GENERATION_H

#include "api/types.h"
#include "test/utils/complex_utils.h"

#define PREPARE_LINEAR_TEST_INPUTS(in1, in2, in_combined, size, factors,       \
                                   precision, data_stride)                     \
    do                                                                         \
    {                                                                          \
        if (precision == FLOAT_P)                                              \
        {                                                                      \
            PREPARE_LINEAR_TEST_INPUTS_IMPL(in1, in2, in_combined, size,       \
                                            factors, FLOAT, data_stride);      \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            PREPARE_LINEAR_TEST_INPUTS_IMPL(in1, in2, in_combined, size,       \
                                            factors, DOUBLE, data_stride);     \
        }                                                                      \
    } while (0)

#define PREPARE_LINEAR_TEST_INPUTS_IMPL(in1, in2, in_combined, size, factors,  \
                                        dt_t, data_stride)                     \
    do                                                                         \
    {                                                                          \
        dt_t *in1_t = (dt_t *)in1;                                             \
        dt_t *in2_t = (dt_t *)in2;                                             \
        dt_t *in_combined_t = (dt_t *)in_combined;                             \
        dt_t *factors_t = (dt_t *)factors;                                     \
        factors_t[0] = (dt_t)((rand() % 200) / 20.0 - 10.0);                   \
        factors_t[1] = 0.0;                                                    \
        factors_t[2] = (dt_t)((rand() % 200) / 20.0 - 10.0);                   \
        factors_t[3] = 0.0;                                                    \
        if (data_stride == 1)                                                  \
        {                                                                      \
            for (INTP idx = 0; idx < size; ++idx)                              \
            {                                                                  \
                in_combined_t[idx] =                                           \
                    factors_t[0] * in1_t[idx] + factors_t[2] * in2_t[idx];     \
            }                                                                  \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            dt_t temp1[2] = {0.0, 0.0};                                        \
            dt_t temp2[2] = {0.0, 0.0};                                        \
            dt_t cmul_temp[2] = {0.0, 0.0};                                    \
            for (INTP idx = 0; idx < size; ++idx)                              \
            {                                                                  \
                CMUL(factors_t, in1_t + idx * 2, temp1, cmul_temp);            \
                CMUL(factors_t + 2, in2_t + idx * 2, temp2, cmul_temp);        \
                CADD(temp1, temp2, in_combined_t + idx * 2);                   \
            }                                                                  \
        }                                                                      \
    } while (0)

#define PREPARE_LINEAR_TEST_OUTPUTS(out1, out2, out_added, size, factors,      \
                                    precision, data_stride)                    \
    do                                                                         \
    {                                                                          \
        if (precision == FLOAT_P)                                              \
        {                                                                      \
            PREPARE_LINEAR_TEST_OUTPUTS_IMPL(out1, out2, out_added, size,      \
                                             factors, FLOAT, data_stride);     \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            PREPARE_LINEAR_TEST_OUTPUTS_IMPL(out1, out2, out_added, size,      \
                                             factors, DOUBLE, data_stride);    \
        }                                                                      \
    } while (0)

#define PREPARE_LINEAR_TEST_OUTPUTS_IMPL(out1, out2, out_added, size, factors, \
                                         dt_t, data_stride)                    \
    do                                                                         \
    {                                                                          \
        dt_t *out1_t = (dt_t *)out1;                                           \
        dt_t *out2_t = (dt_t *)out2;                                           \
        dt_t *out_added_t = (dt_t *)out_added;                                 \
        dt_t *factors_t = (dt_t *)factors;                                     \
        if (data_stride == 1)                                                  \
        {                                                                      \
            for (INTP idx = 0; idx < size; ++idx)                              \
            {                                                                  \
                out_added_t[idx] =                                             \
                    factors_t[0] * out1_t[idx] + factors_t[2] * out2_t[idx];   \
            }                                                                  \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            dt_t temp1[2] = {0.0, 0.0};                                        \
            dt_t temp2[2] = {0.0, 0.0};                                        \
            dt_t cmul_temp[2] = {0.0, 0.0};                                    \
            for (INTP idx = 0; idx < size; ++idx)                              \
            {                                                                  \
                CMUL(factors_t, out1_t + idx * data_stride, temp1, cmul_temp); \
                CMUL(factors_t + 2, out2_t + idx * data_stride, temp2,         \
                     cmul_temp);                                               \
                CADD(temp1, temp2, out_added_t + idx * 2);                     \
            }                                                                  \
        }                                                                      \
    } while (0)

#define PREPARE_TIMESHIFT_TEST_INPUTS(in1, in2, n, m, imap, precision,         \
                                      data_stride)                             \
    do                                                                         \
    {                                                                          \
        if (precision == FLOAT_P)                                              \
        {                                                                      \
            PREPARE_TIMESHIFT_TEST_INPUTS_IMPL(in1, in2, n, m, imap, FLOAT,    \
                                               data_stride);                   \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            PREPARE_TIMESHIFT_TEST_INPUTS_IMPL(in1, in2, n, m, imap, DOUBLE,   \
                                               data_stride);                   \
        }                                                                      \
    } while (0)

#define PREPARE_TIMESHIFT_TEST_INPUTS_IMPL(in1, in2, n, m, imap, dt_t,         \
                                           data_stride)                        \
    do                                                                         \
    {                                                                          \
        dt_t *in1_t = (dt_t *)in1;                                             \
        dt_t *in2_t = (dt_t *)in2;                                             \
        INTP *imap_t = (INTP *)imap;                                           \
        /* Handle overflow to avoid negative indexing */                       \
        if (imap_t == NULL)                                                    \
        {                                                                      \
            for (INTP idx = 0; idx < n * data_stride; idx++)                   \
            {                                                                  \
                INTP src = ((idx / data_stride + (n - m)) % n) * data_stride + \
                           idx % data_stride;                                  \
                INTP dst =                                                     \
                    (idx / data_stride) * data_stride + idx % data_stride;     \
                in2_t[dst] = in1_t[src];                                       \
            }                                                                  \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            for (INTP idx = 0; idx < n * data_stride; idx++)                   \
            {                                                                  \
                INTP src =                                                     \
                    imap_t[(idx / data_stride + (n - m)) % n] * data_stride +  \
                    idx % data_stride;                                         \
                INTP dst = imap_t[idx / data_stride] * data_stride +           \
                           idx % data_stride;                                  \
                in2_t[dst] = in1_t[src];                                       \
            }                                                                  \
        }                                                                      \
    } while (0)

#define PREPARE_TIMESHIFT_TEST_OUTPUTS(out1, out_combined, n, m, unit_m, omap, \
                                       dir, precision, data_stride)            \
    do                                                                         \
    {                                                                          \
        if (precision == FLOAT_P)                                              \
        {                                                                      \
            PREPARE_TIMESHIFT_TEST_OUTPUTS_IMPL(out1, out_combined, n, m,      \
                                                unit_m, omap, dir, FLOAT,      \
                                                data_stride);                  \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            PREPARE_TIMESHIFT_TEST_OUTPUTS_IMPL(out1, out_combined, n, m,      \
                                                unit_m, omap, dir, DOUBLE,     \
                                                data_stride);                  \
        }                                                                      \
    } while (0)

#define PREPARE_TIMESHIFT_TEST_OUTPUTS_IMPL(out1, out_combined, n, m, unit_m,  \
                                            omap, dir, dt_t, data_stride)      \
    do                                                                         \
    {                                                                          \
        dt_t *out1_t = (dt_t *)out1;                                           \
        dt_t *out_combined_t = (dt_t *)out_combined;                           \
        dt_t cmul_temp[2] = {0.0, 0.0};                                        \
        dt_t e_k[2] = {1.0, 0.0};                                              \
        for (INTP k = 0; k < n; k++)                                           \
        {                                                                      \
            /* Handle overflow to improve accuracy for larger values */        \
            INTP mk = (m * k) % n;                                             \
            dt_t angle = (-1 * BENCH_2_PI * mk / n);                           \
            EULER(angle, e_k);                                                 \
            if (omap == NULL)                                                  \
            {                                                                  \
                for (INTP i = 0; i < unit_m; i++)                              \
                {                                                              \
                    CMUL(out1_t + (k * unit_m + i) * data_stride, e_k,         \
                         out_combined_t + (k * unit_m + i) * data_stride,      \
                         cmul_temp);                                           \
                }                                                              \
            }                                                                  \
            else                                                               \
            {                                                                  \
                INTP *omap_t = (INTP *)omap;                                   \
                for (INTP i = 0; i < unit_m; i++)                              \
                {                                                              \
                    CMUL(out1_t + omap_t[(k * unit_m + i)] * data_stride, e_k, \
                         out_combined_t +                                      \
                             omap_t[(k * unit_m + i)] * data_stride,           \
                         cmul_temp);                                           \
                }                                                              \
            }                                                                  \
        }                                                                      \
    } while (0)

#define NORMALIZE_IFFT_DATA(arr, length, n, precision, data_stride)            \
    do                                                                         \
    {                                                                          \
        if (precision == FLOAT_P)                                              \
        {                                                                      \
            NORMALIZE_IFFT_DATA_IMPL(arr, length, n, FLOAT, data_stride);      \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            NORMALIZE_IFFT_DATA_IMPL(arr, length, n, DOUBLE, data_stride);     \
        }                                                                      \
    } while (0)

#define NORMALIZE_IFFT_DATA_IMPL(arr, length, n, dt_t, data_stride)            \
    do                                                                         \
    {                                                                          \
        dt_t *arr_t = (dt_t *)arr;                                             \
        for (INTP idx = 0; idx < length * data_stride; ++idx)                  \
        {                                                                      \
            arr_t[idx] /= n;                                                   \
        }                                                                      \
    } while (0)

VOID prepare_input_data_f(VOID *input, INTP n, INTP *idx_map, INT32 input_type,
                          INT32 data_stride);
VOID prepare_input_data_d(VOID *input, INTP n, INTP *idx_map, INT32 input_type,
                          INT32 data_stride);

#endif // DATA_GENERATION_H
