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

/** @file aoclfftz_corebench_utils.h
 *
 *  @brief Core test bench utility functions.
 *
 *  This file contains the utility and helper functions for core test bench.
 *
 *  @author S. Biplab Raut
 *  @author V. Murugan
 *  @author Srirammaswamy Srinivasan
 */

#ifndef AOCLFFTZ_COREBENCH_UTILS_H
#define AOCLFFTZ_COREBENCH_UTILS_H

#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "test/aoclfftz_corebench.h"
#include "utils/allocator.h"
#include "utils/utils.h"

// Defining DATA_STRIDE in corebench internally to avoid using internal headers
#define T_DATA_STRIDE 2

#define MAX(a, b) ((a > b) ? a : b)

#ifdef WIN32
#define SSCANF sscanf_s
#define STRCPY(dst, size, src) strcpy_s(dst, size, src)
#else
#define SSCANF sscanf
#define STRCPY(dst, size, src) strcpy(dst, src)
#endif

#define PRINT_SUCCESS(str) printf("\033[1;32m" str "\033[1;0m");

#define PRINT_FAILURE(str) printf("\033[1;31m" str "\033[1;0m");

#define VALIDATE_AND_GET_INT(str, buff, result, ret, min_val)                  \
    {                                                                          \
        result = atoi(str);                                                    \
        sprintf(buff, "%d", result);                                           \
        ret = strcmp(str, buff);                                               \
        ret |= (result < min_val);                                             \
    }

#define VALIDATE_AND_GET_DOUBLE(str, buff, result, ret, min_val, max_val)      \
    {                                                                          \
        INT32 length;                                                          \
        DOUBLE temp;                                                           \
        result = atof(str);                                                    \
        ret = SSCANF(str, "%lf %n", &temp, &length) != 1;                      \
        ret |= strlen(str) != length;                                          \
        ret |= (result < min_val);                                             \
        ret |= (result > max_val);                                             \
    }

/**
 * @brief check whether the dims are currently supported or not
 *
 */
#define CHECK_SUPPORTED_DIMS(dims, vecs, dim_rank, vec_rank, status)           \
    {                                                                          \
        if (dim_rank != 1 || vec_rank != 1)                                    \
        {                                                                      \
            status = UNSUPPORTED_SIZE_ERROR;                                   \
        }                                                                      \
    }

/**
 * @brief handle the errors codes returned from parsing function
 *
 */
#define HANDLE_PARSER_ERROR_MESSAGE(status)                                    \
    {                                                                          \
        if (status != PARSER_SUCCESS)                                          \
        {                                                                      \
            switch (status)                                                    \
            {                                                                  \
            case HELP_MENU:                                                    \
                show_help_menu();                                              \
                break;                                                         \
            case SIZE_REQUIRED_ERROR:                                          \
                printf("\nProblem size must be provided.\n");                  \
                break;                                                         \
            case SIZE_PARSING_ERROR:                                           \
                printf("\nInvalid problem size provided.\n");                  \
                break;                                                         \
            case UNSUPPORTED_SIZE_ERROR:                                       \
                printf("\nUnsupported problem size provided.\nOnly 1D is "     \
                       "supported with dim values 2 to 16 and vec value "      \
                       "1.\n");                                                \
                break;                                                         \
            case UNSUPPORTED_OPTION_ERROR:                                     \
                printf("\nUnsupported option provided.\n");                    \
                break;                                                         \
            case NON_OPTION_ARGUMENTS_ERROR:                                   \
                printf("\nMore than one non-option arguments provided.\nOnly " \
                       "one non-option argument must be provided which is "    \
                       "problem size.\n");                                     \
                break;                                                         \
            default:                                                           \
                printf("\nInvalid arguments provided.\n");                     \
            }                                                                  \
            if (status != HELP_MENU)                                           \
            {                                                                  \
                printf("Use -h / --help for more information.\n");             \
                PRINT_FAILURE("\nTest bench failed [REASON: Argument parsing " \
                              "failed]\n\n");                                  \
            }                                                                  \
        }                                                                      \
    }

/**
 * @brief handle the bench status error codes
 *
 */
#define HANDLE_BENCH_STATUS(status)                                            \
    {                                                                          \
        switch (status)                                                        \
        {                                                                      \
        case BENCH_SUCCESS:                                                    \
            break;                                                             \
        case SETUP_FAILURE:                                                    \
            PRINT_FAILURE(                                                     \
                "\nTest bench failed [REASON: Setup problem failed]\n\n");     \
            break;                                                             \
        case EXECUTION_FAILURE:                                                \
            PRINT_FAILURE(                                                     \
                "\nTest bench failed [REASON: Execute problem failed]\n\n");   \
            break;                                                             \
        case VERIFICATION_FAILURE:                                             \
            PRINT_FAILURE(                                                     \
                "\nTest bench failed [REASON: Result mismatch]\n\n");          \
            break;                                                             \
        default:                                                               \
            PRINT_FAILURE("\nTest bench failed [REASON: Unknown]\n\n");        \
        }                                                                      \
    }

/**
 * @brief Initialize problem descriptor with the bench params
 *
 */
#define INIT_PD(params, p_desc, dt_t, dim_t)                                   \
    {                                                                          \
        p_desc->dim_rank = params->dim_rank;                                   \
        p_desc->vec_rank = params->vec_rank;                                   \
        p_desc->dims = (dim_t *)malloc(sizeof(dim_t) * p_desc->dim_rank);      \
        for (INT32 i = 0; i < p_desc->dim_rank; i++)                           \
        {                                                                      \
            p_desc->dims[i].n = (dt_t)params->dims[i].n;                       \
            p_desc->dims[i].in_stride = (dt_t)params->dims[i].in_stride;       \
            p_desc->dims[i].out_stride = (dt_t)params->dims[i].out_stride;     \
        }                                                                      \
        p_desc->vecs = (dim_t *)malloc(sizeof(dim_t) * p_desc->vec_rank);      \
        for (INT32 i = 0; i < p_desc->vec_rank; i++)                           \
        {                                                                      \
            p_desc->vecs[i].n = (dt_t)params->vecs[i].n;                       \
            p_desc->vecs[i].in_stride = (dt_t)params->vecs[i].in_stride;       \
            p_desc->vecs[i].out_stride = (dt_t)params->vecs[i].out_stride;     \
        }                                                                      \
        if (params->num_threads > 0)                                           \
        {                                                                      \
            p_desc->pthr_fft.num_threads = params->num_threads;                \
            p_desc->pthr_fft.dynamic_load_model = 0;                           \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            p_desc->pthr_fft.num_threads = 0;                                  \
            p_desc->pthr_fft.dynamic_load_model = 1;                           \
        }                                                                      \
        p_desc->cntrl_params.opt_level = params->opt_level;                    \
        p_desc->flags = set_flag(params);                                      \
        if (params->opt_level == -1)                                           \
        {                                                                      \
            p_desc->cntrl_params.opt_off = 1;                                  \
        }                                                                      \
        p_desc->cntrl_params.logger_mode = params->logger_mode;                \
        p_desc->cntrl_params.measure_stats = 0;                                \
    }

/**
 * @brief Destroy the problem descriptor
 *
 */
#define DESTROY_PD(p_desc)                                                     \
    {                                                                          \
        if (p_desc != NULL)                                                    \
        {                                                                      \
            FREE_ALLOCATED_MEM(p_desc->dims);                                  \
            FREE_ALLOCATED_MEM(p_desc->vecs);                                  \
            FREE_ALLOCATED_MEM(p_desc);                                        \
        }                                                                      \
    }

/**
 * @brief Initialize dst params object and copy the values from src to dst
 *        NOTE: This function will not create in and out buffers, it needs to be
 *              created manually
 *
 */
#define ALLOC_AND_COPY_PARAMS(dst, src)                                        \
    {                                                                          \
        dst = (aoclfftz_bench_params_t *)ALLOC_UNALIGN_UNINIT(                 \
            sizeof(aoclfftz_bench_params_t));                                  \
        memcpy(dst, src, sizeof(aoclfftz_bench_params_t));                     \
        dst->dims = (aoclfftz_dim_t_64_ *)ALLOC_UNALIGN_UNINIT(                \
            sizeof(aoclfftz_bench_params_t));                                  \
        dst->vecs = (aoclfftz_dim_t_64_ *)ALLOC_UNALIGN_UNINIT(                \
            sizeof(aoclfftz_bench_params_t));                                  \
        memcpy(dst->dims, src->dims,                                           \
               sizeof(aoclfftz_dim_t_64_) * src->dim_rank);                    \
        memcpy(dst->vecs, src->vecs,                                           \
               sizeof(aoclfftz_dim_t_64_) * src->vec_rank);                    \
    }

/**
 * @brief Log the bench param values in INFO mode
 *
 */
#define LOG_BENCH_PARAMS(params)                                               \
    {                                                                          \
        AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode,                      \
                               "dim_rank      : %d", params->dim_rank);        \
        for (INT32 i = 0; i < params->dim_rank; i++)                           \
        {                                                                      \
            AOCLFFTZ_LOG_FORMATTED(                                            \
                INFO, params->logger_mode, "    dims[%d]   : %td:%td:%td", i,  \
                params->dims[i].n, params->dims[i].in_stride,                  \
                params->dims[i].out_stride);                                   \
        }                                                                      \
        AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode,                      \
                               "vec_rank      : %d", params->vec_rank);        \
        for (INT32 i = 0; i < params->dim_rank; i++)                           \
        {                                                                      \
            AOCLFFTZ_LOG_FORMATTED(                                            \
                INFO, params->logger_mode, "    vecs[%d]   : %td:%td:%td", i,  \
                params->vecs[i].n, params->vecs[i].in_stride,                  \
                params->vecs[i].out_stride);                                   \
        }                                                                      \
        AOCLFFTZ_LOG_FORMATTED(                                                \
            INFO, params->logger_mode, "precision     : %s",                   \
            params->precision == FLOAT_P ? "FLOAT" : "DOUBLE");                \
        AOCLFFTZ_LOG_FORMATTED(                                                \
            INFO, params->logger_mode, "data_model    : %s",                   \
            params->data_model == ILP64 ? "ILP64" : "LP64");                   \
        AOCLFFTZ_LOG_FORMATTED(                                                \
            INFO, params->logger_mode, "bench_type    : %s",                   \
            params->bench_type == ACCURACY ? "ACCURACY" : "PERFORMANCE");      \
        AOCLFFTZ_LOG_FORMATTED(                                                \
            INFO, params->logger_mode, "res_placement : %s",                   \
            params->res_placement == IN_PLACE ? "IN_PLACE" : "OUT_OF_PLACE");  \
        AOCLFFTZ_LOG_FORMATTED(                                                \
            INFO, params->logger_mode, "order         : %s",                   \
            params->order == OUT_OF_ORDER ? "OUT_OF_ORDER" : "IN_ORDER");      \
        AOCLFFTZ_LOG_FORMATTED(                                                \
            INFO, params->logger_mode, "direction     : %s",                   \
            params->dir == BACKWARD ? "BACKWARD" : "FORWARD");                 \
        AOCLFFTZ_LOG_FORMATTED(                                                \
            INFO, params->logger_mode, "fft_type      : %s",                   \
            params->fft_type == COMPLEX_TO_COMPLEX                             \
                ? "COMPLEX_TO_COMPLEX"                                         \
                : (params->fft_type == REAL_TO_COMPLEX ? "REAL_TO_COMPLEX"     \
                                                       : "COMPLEX_TO_REAL"));  \
        AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode,                      \
                               "iterations    : %d", params->num_iterations);  \
        if (params->bench_type == PERFORMANCE)                                 \
        {                                                                      \
            AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode,                  \
                                   "warmup-iters  : %d",                       \
                                   params->warmup_iterations);                 \
        }                                                                      \
        AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode,                      \
                               "random_seed   : %s",                           \
                               params->use_random_seed ? "TRUE" : "FALSE");    \
        AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode,                      \
                               "opt_level     : %d", params->opt_level);       \
        AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode,                      \
                               "logger_mode   : %d", params->logger_mode);     \
        AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode,                      \
                               "tolerance     : %.20lf (%.5lE)",               \
                               params->tolerance, params->tolerance);          \
        AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode,                      \
                               "num_threads   : %d", params->num_threads);     \
        AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode,                      \
                               "selector_time : %s",                           \
                               params->selector_time ? "TRUE" : "FALSE");      \
        AOCLFFTZ_LOG_FORMATTED(                                                \
            INFO, params->logger_mode, "bit_reproducibility : %s",             \
            params->bit_reproducibility ? "TRUE" : "FALSE");                   \
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
 * @brief complex addition: res = c1 + c2
 *
 */
#define CADD(c1, c2, res)                                                      \
    {                                                                          \
        (res)[0] = (c1)[0] + (c2)[0];                                          \
        (res)[1] = (c1)[1] + (c2)[1];                                          \
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
 * @brief prints the float complex array `arr` of length `length`
 *
 */
#define PRINT_CARRAY_FP32(arr, size)                                           \
    {                                                                          \
        FLOAT *arr_f = (FLOAT *)arr;                                           \
        for (INTP i = 0; i < size; ++i)                                        \
        {                                                                      \
            printf("%td: %12.6f + %12.6fj\n", (INTP)i,                         \
                   (arr_f)[i * T_DATA_STRIDE],                                 \
                   (arr_f)[i * T_DATA_STRIDE + 1]);                            \
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
        for (INTP i = 0; i < size; ++i)                                        \
        {                                                                      \
            printf("%td: %20.14lf + %20.14lfj\n", (INTP)i,                     \
                   (arr_d)[i * T_DATA_STRIDE],                                 \
                   (arr_d)[i * T_DATA_STRIDE + 1]);                            \
        }                                                                      \
        printf("\n");                                                          \
    }

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

#define PREPARE_TIMESHIFT_TEST_INPUTS(in1, in2, n, m, stride, precision)       \
    {                                                                          \
        if (precision == FLOAT_P)                                              \
        {                                                                      \
            PREPARE_TIMESHIFT_TEST_INPUTS_IMPL(in1, in2, n, m, stride, FLOAT); \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            PREPARE_TIMESHIFT_TEST_INPUTS_IMPL(in1, in2, n, m, stride,         \
                                               DOUBLE);                        \
        }                                                                      \
    }

#define PREPARE_TIMESHIFT_TEST_INPUTS_IMPL(in1, in2, n, m, stride, dt_t)       \
    {                                                                          \
        dt_t *in1_t = (dt_t *)in1;                                             \
        dt_t *in2_t = (dt_t *)in2;                                             \
        for (INTP idx = 0; idx < n; idx++)                                     \
        {                                                                      \
            for (INTP is = 0; is < stride; is++)                               \
            {                                                                  \
                INTP src = (((idx + (n - m)) * stride) % (n * stride) + is) *  \
                           T_DATA_STRIDE;                                      \
                INTP dst = ((idx)*stride + is) * T_DATA_STRIDE;                \
                in2_t[dst] = in1_t[src];                                       \
                in2_t[dst + 1] = in1_t[src + 1];                               \
            }                                                                  \
        }                                                                      \
    }

#define PREPARE_TIMESHIFT_TEST_OUTPUTS(out1, out_combined, n, m, stride, dir,  \
                                       precision)                              \
    {                                                                          \
        if (precision == FLOAT_P)                                              \
        {                                                                      \
            PREPARE_TIMESHIFT_TEST_OUTPUTS_IMPL(out1, out_combined, n, m,      \
                                                stride, dir, FLOAT);           \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            PREPARE_TIMESHIFT_TEST_OUTPUTS_IMPL(out1, out_combined, n, m,      \
                                                stride, dir, DOUBLE);          \
        }                                                                      \
    }

#define PREPARE_TIMESHIFT_TEST_OUTPUTS_IMPL(out1, out_combined, n, m, stride,  \
                                            dir, dt_t)                         \
    {                                                                          \
        dt_t *out1_t = (dt_t *)out1;                                           \
        dt_t *out_combined_t = (dt_t *)out_combined;                           \
        dt_t cmul_temp[T_DATA_STRIDE] = {0.0, 0.0};                            \
        dt_t e_k[T_DATA_STRIDE] = {1.0, 0.0};                                  \
        dt_t two = (dir == FORWARD) ? -2.0 : 2.0;                              \
        for (INTP k = 0; k < n; k++)                                           \
        {                                                                      \
            EULER((two * M_PI * m * k / n), e_k);                              \
            CMUL(out1_t + k * stride * T_DATA_STRIDE, e_k,                     \
                 out_combined_t + k * stride * T_DATA_STRIDE, cmul_temp);      \
        }                                                                      \
    }

#define NORMALIZE_DATA(arr, length, n, precision)                              \
    {                                                                          \
        if (precision == FLOAT_P)                                              \
        {                                                                      \
            NORMALIZE_DATA_IMPL(arr, length, n, FLOAT);                        \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            NORMALIZE_DATA_IMPL(arr, length, n, DOUBLE);                       \
        }                                                                      \
    }

#define NORMALIZE_DATA_IMPL(arr, length, n, dt_t)                              \
    {                                                                          \
        dt_t *arr_t = (dt_t *)arr;                                             \
        for (INTP idx = 0; idx < length; ++idx)                                \
        {                                                                      \
            arr_t[idx * T_DATA_STRIDE] /= n;                                   \
            arr_t[idx * T_DATA_STRIDE + 1] /= n;                               \
        }                                                                      \
    }

// Function pointers
VOID (*prepare_input_data)(VOID *, INTP, INTP, INT32);
VOID (*dft_ref)(VOID *, VOID *, INTP, INTP, INTP, INTP, INTP, INTP, INT32);
INT32 (*compare)(VOID *, VOID *, INTP, INTP, DOUBLE, INT32);

// Function declarations
INT32 set_flag(aoclfftz_bench_params_t *params);
INT32 find_dim_rank(CHAR *arg);
INT32 allocate_and_fill_dims_vecs(CHAR *arg, INT32 dim_rank, INT32 vec_rank,
                                  aoclfftz_dim_t_64_ **dims,
                                  aoclfftz_dim_t_64_ **vecs,
                                  INTP default_stride);
VOID prepare_input_data_f(VOID *input, INTP n, INTP stride, INT32 input_type);
VOID prepare_input_data_d(VOID *input, INTP n, INTP stride, INT32 input_type);
VOID dft_ref_f(VOID *in, VOID *out, INTP n, INTP in_stride, INTP out_stride,
               INTP batch, INTP v_in_stride, INTP v_out_stride, INT32 is_bwd);
VOID dft_ref_d(VOID *in, VOID *out, INTP n, INTP in_stride, INTP out_stride,
               INTP batch, INTP v_in_stride, INTP v_out_stride, INT32 is_bwd);
INT32 compare_f(VOID *a, VOID *b, INTP n, INTP stride, DOUBLE tol,
                INT32 logger_mode);
INT32 compare_d(VOID *a, VOID *b, INTP n, INTP stride, DOUBLE tol,
                INT32 logger_mode);

// Function definitions

/**
 * @brief set the flag value based on bench params
 *
 * @param params bench params
 * @return INT32 encoded flag value
 */
INT32 set_flag(aoclfftz_bench_params_t *params)
{
    UINT32 flag = 0;
    if (params->res_placement == IN_PLACE)
    {
        flag = flag | (0 << 0);
    }
    else
    {
        flag = flag | (1 << 0);
    }
    if (params->order == IN_ORDER)
    {
        flag = flag | (0 << 1);
    }
    else
    {
        flag = flag | (1 << 1);
    }
    if (params->dir == FORWARD)
    {
        flag = flag | (0 << 2);
    }
    else
    {
        flag = flag | (1 << 2);
    }
    if (params->fft_type == COMPLEX_TO_COMPLEX)
    {
        flag = flag | (0 << 3);
    }
    else
    {
        flag = flag | (1 << 3);
    }
    return flag;
}

/**
 * @brief find the rank of the dimensions from the given dimension argument
 *
 * @param arg dimension string
 * @param dim_rank reference variable to store the dim-rank
 * @param vec_rank reference variable to store the vec-rank
 * @return INT32 status code
 */
INT32 find_dim_vec_ranks(CHAR *arg, INT32 *dim_rank, INT32 *vec_rank)
{
    INT32 dims = 1;
    INT32 vecs = 0;
    for (INT32 i = 0; i < strlen(arg); i++)
    {
        if (arg[i] == 'x')
        {
            dims++;
        }
    }
    for (INT32 i = 0; i < strlen(arg); i++)
    {
        if (arg[i] == 'v')
        {
            vecs++;
        }
    }
    if (vecs > dims)
    {
        return SIZE_PARSING_ERROR;
    }
    if (vecs == 0)
    {
        vecs = 1;
    }
    (*dim_rank) = dims;
    (*vec_rank) = vecs;
    return PARSER_SUCCESS;
}

/**
 * @brief allocate and fill the values of dims and vecs parsed from the
 * string argument.
 *
 * @param arg dimension string
 * @param dim_rank rank of the dimensions
 * @param dim_rank rank of the vectors
 * @param dims pointer to store the dims structure
 * @param vecs pointer to stroe the vecs structure
 * @param default_stride default stride value to be used for dims and vecs
 * @return INT32
 */
// FIXME : multi-dimension support
INT32 allocate_and_fill_dims_vecs(CHAR *arg, INT32 dim_rank, INT32 vec_rank,
                                  aoclfftz_dim_t_64_ **dims,
                                  aoclfftz_dim_t_64_ **vecs,
                                  INTP default_stride)
{
    (*dims) = (aoclfftz_dim_t_64_ *)ALLOC_UNALIGN_UNINIT(
        dim_rank * sizeof(aoclfftz_dim_t_64_));
    (*vecs) = (aoclfftz_dim_t_64_ *)ALLOC_UNALIGN_UNINIT(
        vec_rank * sizeof(aoclfftz_dim_t_64_));
    INT32 is_stride = 0;
    INT32 is_vec_stride = 0;
    INT32 rank_count = 0;
    INT32 vec_count = 0;
    INT32 def_stride_rank = -1;
    INT32 def_vec_stride_rank = -1;
    CHAR last_char = '\0';
    INT32 start = 0;
    CHAR val_str[strlen(arg)];
    INT32 status = PARSER_SUCCESS;
    for (INT32 i = 0; i < strlen(arg); ++i)
    {
        if (arg[i] == 'x')
        {
            // 1D is currently supported
            return UNSUPPORTED_SIZE_ERROR;
            // TODO: Fix this code to support more dimensions
            if ((is_stride != 0 && is_stride != 2) ||
                (is_vec_stride != 0 && is_vec_stride != 2))
            {
                return SIZE_PARSING_ERROR;
            }
            if (is_stride != 2)
            {
                def_stride_rank = rank_count;
            }
            is_stride = 0;
            rank_count++;
            last_char = 'x';
        }
        else if (arg[i] == 'v')
        {
            // by default the data is stored in dims always
            // once "v" is encountered, its moved to vecs and then dims is reset
            // FIXME : this needs to be fixed properly
            vecs[rank_count]->n = dims[rank_count]->n;
            vecs[rank_count]->in_stride =
                (is_stride >= 1) ? dims[rank_count]->in_stride : 0;
            vecs[rank_count]->out_stride =
                (is_stride == 2) ? dims[rank_count]->out_stride : 0;
            dims[rank_count]->in_stride = default_stride;
            dims[rank_count]->out_stride = default_stride;
            def_vec_stride_rank = def_stride_rank;
            def_stride_rank = 0;
            is_vec_stride = 0;
            vec_count++;
            last_char = 'v';
        }
        else if (arg[i] == ':')
        {
            if (last_char == 'v')
            {
                is_vec_stride++;
            }
            else
            {
                is_stride++;
            }
        }
        else if (isdigit(arg[i]))
        {
            start = i;
            val_str[0] = arg[i];
            while (isdigit(arg[++i]))
            {
                val_str[i - start] = arg[i];
            }
            val_str[i - start] = '\0';
            INTP val = atol(val_str);
            if (val < 0)
            {
                return SIZE_PARSING_ERROR;
            }
            if (last_char == 'v')
            {
                if (is_vec_stride == 0) // no strides encountered till now
                {
                    dims[rank_count]->n = val;
                }
                else if (is_vec_stride == 1)
                {
                    dims[rank_count]->in_stride = val;
                    def_stride_rank++;
                }
                else if (is_vec_stride == 2)
                {
                    dims[rank_count]->out_stride = val;
                    def_stride_rank++;
                }
            }
            else
            {
                if (is_stride == 0)
                {
                    dims[rank_count]->n = val;
                }
                else if (is_stride == 1)
                {
                    dims[rank_count]->in_stride = val;
                    def_stride_rank++;
                }
                else if (is_stride == 2)
                {
                    dims[rank_count]->out_stride = val;
                    def_stride_rank++;
                }
            }
            i--;
        }
        else
        {
            return SIZE_PARSING_ERROR;
        }
    }

    // TODO: align with FFTW's bench behaviour
    // for now, either no stride or both in/out strides should be provided
    // only in_stride will not be parsed
    if ((is_stride != 0 && is_stride != 2) ||
        (is_vec_stride != 0 && is_vec_stride != 2))
    {
        return SIZE_PARSING_ERROR;
    }

    // if both strides are provided, check if vec strides are as expected
    if(is_stride == 2 && is_vec_stride == 2)
    {
        for(INT32 i = 0; i < dim_rank; i++)
        {
            if( (vecs[i]->in_stride != (dims[i]->in_stride * dims[i]->n)) ||
                (vecs[i]->out_stride != (dims[i]->out_stride * dims[i]->n)) )
            {
                return SIZE_PARSING_ERROR;
            }
        }
    }

    // Fill the default stride values for the last dimension if missed
    if (def_stride_rank < dim_rank * 2 - 1)
    {
        for (INT32 i = def_stride_rank + 1; i < dim_rank; ++i)
        {
            dims[i]->in_stride = default_stride;
            dims[i]->out_stride = default_stride;
        }
    }

    if (def_vec_stride_rank < vec_rank * 2 - 1)
    {
        for (INT32 i = def_vec_stride_rank + 1; i < vec_rank; ++i)
        {
            // default value of vec stride is atleast the size of its corresponding dim
            vecs[i]->in_stride = dims[i]->n * dims[i]->in_stride;
            vecs[i]->out_stride = dims[i]->n * dims[i]->out_stride;
        }
    }

    // Move the vecs to dims and fill the default values for vecs if they are
    // not provided
    if (vec_count == 0 && vec_rank > 0)
    {
        for (INT32 i = 0; i < vec_rank; ++i)
        {
            vecs[i]->n = 1;
            vecs[i]->in_stride = default_stride;
            vecs[i]->out_stride = default_stride;
        }
    }

    CHECK_SUPPORTED_DIMS(dims, vecs, dim_rank, vec_rank, status);
    return status;
}

/**
 * @brief Prepare FLOAT input data of size `n * stride`.
 *
 * @param input array to store input data
 * @param n input size
 * @param stride strides
 * @param input_type type of input data : RANDOM, IMPULSE or SIGNAL
 * @return VOID
 */
VOID prepare_input_data_f(VOID *input, INTP n, INTP stride, INT32 input_type)
{
    FLOAT *input_f = (FLOAT *)input;
    // random input
    if (input_type == RANDOM_INPUT)
    {
        for (INTP idx = 0; idx < n * stride * T_DATA_STRIDE; ++idx)
        {
            // range: [-10.0, 10.0) with 3 decimal precision
            input_f[idx] = (FLOAT)((rand() % 2000) / 200.0) - 10.0;
        }
    }
    // impulse input
    else if (input_type == IMPULSE_INPUT)
    {
        INTP idx = (INTP)(rand() % n) * stride;
        memset(input_f, 0, n * stride * T_DATA_STRIDE);
        // range: [-10.0, 10.0) with 3 decimal precision
        input_f[idx * T_DATA_STRIDE] = (FLOAT)((rand() % 2000) / 200.0 - 10.0);
        input_f[idx * T_DATA_STRIDE + 1] =
            (FLOAT)((rand() % 2000) / 200.0 - 10.0);
    }
    // sinusoidal signal input
    else if (input_type == SINUSOIDAL_SIGNAL_INPUT)
    {
        INTP length = n * stride;
        // Sine wave cycles
        INTP cycles = (rand() % (length / 2)) + 2;
        FLOAT size = 2.0 * M_PI * cycles;
        // Shift the origin of the wave from 0 to a positive integer `shift`,
        // shift range: [0, length)
        INTP shift = rand() % length;
        // scale the amplitude of the wave by `scale` times, scale range:
        // [0.0 5.0)
        FLOAT scale = ((FLOAT)rand() / RAND_MAX) * 5.0;
        for (INTP i = 0; i < length; i++)
        {
            input_f[((i + shift) % length) * T_DATA_STRIDE] =
                sin((FLOAT)(i * size) / length) * scale;
            input_f[((i + shift) % length) * T_DATA_STRIDE + 1] = 0.0;
        }
    }
}

/**
 * @brief Prepare DOUBLE input data of size `n * stride`.
 *
 * @param input array to store input data
 * @param n input size
 * @param stride strides
 * @param input_type type of input data : RANDOM, IMPULSE or SIGNAL
 * @return VOID
 */
VOID prepare_input_data_d(VOID *input, INTP n, INTP stride, INT32 input_type)
{
    DOUBLE *input_d = (DOUBLE *)input;
    // random input
    if (input_type == RANDOM_INPUT)
    {
        for (INTP idx = 0; idx < n * stride * T_DATA_STRIDE; ++idx)
        {
            // range: [-10.0, 10.0) with 3 decimal precision
            input_d[idx] = (DOUBLE)((rand() % 2000) / 200.0) - 10.0;
        }
    }
    // impulse input
    else if (input_type == IMPULSE_INPUT)
    {
        INTP idx = (INTP)(rand() % n) * stride;
        memset(input_d, 0, n * stride * T_DATA_STRIDE);
        // range: [-10.0, 10.0) with 3 decimal precision */
        input_d[idx * T_DATA_STRIDE] = (DOUBLE)((rand() % 2000) / 200.0 - 10.0);
        input_d[idx * T_DATA_STRIDE + 1] =
            (DOUBLE)((rand() % 2000) / 200.0 - 10.0);
    }
    // sinusoidal signal input
    else if (input_type == SINUSOIDAL_SIGNAL_INPUT)
    {
        INTP length = n * stride;
        // Sine wave cycles
        INTP cycles = (rand() % (length / 2)) + 2;
        DOUBLE size = 2.0 * M_PI * cycles;
        // Shift the origin of the wave from 0 to a positive integer `shift`,
        // shift range: [0, length)
        INTP shift = rand() % length;
        // scale the amplitude of the wave by `scale` times, scale range:
        // [0.0 5.0)
        DOUBLE scale = ((DOUBLE)rand() / RAND_MAX) * 5.0;
        for (INTP i = 0; i < length; i++)
        {
            input_d[((i + shift) % length) * T_DATA_STRIDE] =
                sin((DOUBLE)(i * size) / length) * scale;
            input_d[((i + shift) % length) * T_DATA_STRIDE + 1] = 0.0;
        }
    }
}

/**
 * @brief DFT reference implementation for FLOAT type
 *
 * @param in input data
 * @param out output data
 * @param n input length
 * @param in_stride input stride value
 * @param out_stride output stride value
 * @param is_bwd 1 for backward dir, 0 for forward dir
 * @return VOID
 */
VOID dft_ref_f(VOID *in, VOID *out, INTP n, INTP in_stride, INTP out_stride,
               INTP batch, INTP v_in_stride, INTP v_out_stride, INT32 is_bwd)
{
    FLOAT e[T_DATA_STRIDE], mul_buf[T_DATA_STRIDE];
    FLOAT two = is_bwd ? 2.0 : -2.0;
    FLOAT *in_f = (FLOAT *)in;
    FLOAT *out_f = (FLOAT *)out;
    in_stride = in_stride * T_DATA_STRIDE;
    out_stride = out_stride * T_DATA_STRIDE;
    v_in_stride = v_in_stride * T_DATA_STRIDE;
    v_out_stride = v_out_stride * T_DATA_STRIDE;

    for (INTP b = 0; b < batch; b++)
    {
        for (INTP k = 0; k < n; k++)
        {
            INTP out_idx = k * out_stride;
            for (INTP i = 0; i < n; i++)
            {
                INTP in_idx = i * in_stride;

                FLOAT x = (two * M_PI * i * k / n);
                e[0] = cos(x);
                e[1] = sin(x);

                mul_buf[0] = (in_f[in_idx] * e[0]) - (in_f[in_idx + 1] * e[1]);
                mul_buf[1] = (in_f[in_idx] * e[1]) + (in_f[in_idx + 1] * e[0]);

                out_f[out_idx]     = out_f[out_idx] + mul_buf[0];
                out_f[out_idx + 1] = out_f[out_idx + 1] + mul_buf[1];
            }
        }
        in_f  += v_in_stride;
        out_f += v_out_stride;
    }

}

/**
 * @brief DFT reference implementation for DOUBLE type
 *
 * @param in input data
 * @param out output data
 * @param n input length
 * @param in_stride input stride value
 * @param out_stride output stride value
 * @param is_bwd 1 for backward dir, 0 for forward dir
 * @return VOID
 */
VOID dft_ref_d(VOID *in, VOID *out, INTP n, INTP in_stride, INTP out_stride,
               INTP batch, INTP v_in_stride, INTP v_out_stride, INT32 is_bwd)
{
    DOUBLE e[T_DATA_STRIDE], mul_buf[T_DATA_STRIDE];
    DOUBLE two = is_bwd ? 2.0 : -2.0;
    DOUBLE *in_d = (DOUBLE *)in;
    DOUBLE *out_d = (DOUBLE *)out;
    in_stride = in_stride * T_DATA_STRIDE;
    out_stride = out_stride * T_DATA_STRIDE;
    v_in_stride = v_in_stride * T_DATA_STRIDE;
    v_out_stride = v_out_stride * T_DATA_STRIDE;

    for (INTP b = 0; b < batch; b++)
    {
        for (INTP k = 0; k < n; k++)
        {
            INTP out_idx = k * out_stride;
            for (INTP i = 0; i < n; i++)
            {
                INTP in_idx = i * in_stride;
                DOUBLE x = (two * M_PI * i * k / n);
                e[0] = cos(x);
                e[1] = sin(x);

                mul_buf[0] = (in_d[in_idx] * e[0]) - (in_d[in_idx + 1] * e[1]);
                mul_buf[1] = (in_d[in_idx] * e[1]) + (in_d[in_idx + 1] * e[0]);

                out_d[out_idx]     = out_d[out_idx] + mul_buf[0];
                out_d[out_idx + 1] = out_d[out_idx + 1] + mul_buf[1];
            }
        }
        in_d  += v_in_stride;
        out_d += v_out_stride;
    }
}

/**
 * @brief Compare the two data of length n (FLOAT type)
 *        Used to compare output with reference output.
 *
 * @param a first data
 * @param b second data
 * @param n data length
 * @param stride stride value
 * @param tol error tolerance value
 * @param logger_mode logger mode from bench_params
 * @return INT32 if the data points are same, return 1 else 0
 */
INT32 compare_f(VOID *a, VOID *b, INTP n, INTP stride, DOUBLE tol,
                INT32 logger_mode)
{
    FLOAT *a_f = (FLOAT *)a;
    FLOAT *b_f = (FLOAT *)b;
    FLOAT tol_f = (FLOAT)tol;
    FLOAT max_abs_err = 0.0;
    FLOAT max_mag = 0.0;
    INTP max_err_idx = -1;
    INTP first_err_idx = INT64_MAX;
    for (INTP idx = 0; idx < n; idx += stride)
    {
        FLOAT abs_err = fmaxf(
            fabsf(a_f[idx * T_DATA_STRIDE] - b_f[idx * T_DATA_STRIDE]),
            fabsf(a_f[idx * T_DATA_STRIDE + 1] - b_f[idx * T_DATA_STRIDE + 1]));
        FLOAT mag = fminf(fmaxf(fabsf(a_f[idx * T_DATA_STRIDE]),
                                fabsf(a_f[idx * T_DATA_STRIDE + 1])),
                          fmaxf(fabsf(b_f[idx * T_DATA_STRIDE]),
                                fabsf(b_f[idx * T_DATA_STRIDE + 1])));
        if (abs_err > max_abs_err)
        {
            max_abs_err = abs_err;
            max_err_idx = idx;
            if (idx < first_err_idx)
            {
                first_err_idx = idx;
            }
        }
        if (mag > max_mag)
        {
            max_mag = mag;
        }
    }
    FLOAT rel_err;
    if (max_abs_err == 0.0 && max_mag == 0.0)
    {
        rel_err = 0.0;
    }
    else if (max_mag == 0.0)
    {
        rel_err = max_abs_err;
    }
    else
    {
        rel_err = max_abs_err / max_mag;
    }
    AOCLFFTZ_LOG_FORMATTED(INFO, logger_mode, "Error = %.20f (%.5e)", rel_err,
                           rel_err);
    if (rel_err > tol_f)
    {
        AOCLFFTZ_LOG_FORMATTED(DEBUG, logger_mode,
                               "Relative error  = %.10f (%8.5e)", rel_err,
                               rel_err);
        AOCLFFTZ_LOG_FORMATTED(DEBUG, logger_mode,
                               "Tolerance       = %.10f (%8.5e)", tol, tol);
        AOCLFFTZ_LOG_FORMATTED(DEBUG, logger_mode,
                               "First absolute error at index %td",
                               first_err_idx);
        AOCLFFTZ_LOG_FORMATTED(DEBUG, logger_mode,
                               "Max absolute error at index %td", max_err_idx);
        AOCLFFTZ_LOG_FORMATTED(DEBUG, logger_mode, "  expected = %.10f + %.10f",
                               b_f[max_err_idx * T_DATA_STRIDE],
                               b_f[max_err_idx * T_DATA_STRIDE + 1]);
        AOCLFFTZ_LOG_FORMATTED(DEBUG, logger_mode, "  got      = %.10f + %.10f",
                               a_f[max_err_idx * T_DATA_STRIDE],
                               a_f[max_err_idx * T_DATA_STRIDE + 1]);
        AOCLFFTZ_LOG_FORMATTED(DEBUG, logger_mode,
                               "  max abs error = %.10f (%8.5e)", max_abs_err,
                               max_abs_err);
        return VERIFICATION_FAILURE;
    }
    return BENCH_SUCCESS;
}

/**
 * @brief Compare the two data of length n (DOUBLE type)
 *        Used to compare output with reference output.
 *
 * @param a first data
 * @param b second data
 * @param n data length
 * @param stride stride value
 * @param tol error tolerance value
 * @param logger_mode logger mode from bench_params
 * @return INT32 if the data points are same, return 1 else 0
 */
INT32 compare_d(VOID *a, VOID *b, INTP n, INTP stride, DOUBLE tol,
                INT32 logger_mode)
{
    DOUBLE *a_d = (DOUBLE *)a;
    DOUBLE *b_d = (DOUBLE *)b;
    DOUBLE max_abs_err = 0.0;
    DOUBLE max_mag = 0.0;
    INTP max_err_idx = -1;
    INTP first_err_idx = INT64_MAX;
    DOUBLE first_abs_err = 0.0;
    for (INTP idx = 0; idx < n; idx += stride)
    {
        DOUBLE abs_err = fmax(
            fabs(a_d[idx * T_DATA_STRIDE] - b_d[idx * T_DATA_STRIDE]),
            fabs(a_d[idx * T_DATA_STRIDE + 1] - b_d[idx * T_DATA_STRIDE + 1]));
        DOUBLE mag = fmin(fmax(fabs(a_d[idx * T_DATA_STRIDE]),
                               fabs(a_d[idx * T_DATA_STRIDE + 1])),
                          fmax(fabs(b_d[idx * T_DATA_STRIDE]),
                               fabs(b_d[idx * T_DATA_STRIDE + 1])));
        if (abs_err > max_abs_err)
        {
            max_abs_err = abs_err;
            max_err_idx = idx;
            if (idx < first_err_idx && abs_err > tol)
            {
                first_err_idx = idx;
                first_abs_err = abs_err;
            }
        }
        if (mag > max_mag)
        {
            max_mag = mag;
        }
    }
    DOUBLE rel_err;
    if (max_abs_err == 0.0 && max_mag == 0.0)
    {
        rel_err = 0.0;
    }
    else if (max_mag == 0.0)
    {
        rel_err = max_abs_err;
    }
    else
    {
        rel_err = max_abs_err / max_mag;
    }
    AOCLFFTZ_LOG_FORMATTED(INFO, logger_mode, "Error = %.20lf (%.5le)", rel_err,
                           rel_err);
    if (rel_err > tol)
    {
        AOCLFFTZ_LOG_FORMATTED(DEBUG, logger_mode,
                               "Relative error  = %.20lf (%8.5le)", rel_err,
                               rel_err);
        AOCLFFTZ_LOG_FORMATTED(DEBUG, logger_mode,
                               "Tolerance       = %.20lf (%8.5le)", tol, tol);
        if (first_err_idx < INT64_MAX)
        {
            AOCLFFTZ_LOG_FORMATTED(DEBUG, logger_mode,
                                   "First absolute error at index %td",
                                   first_err_idx);
            AOCLFFTZ_LOG_FORMATTED(DEBUG, logger_mode,
                                   "  expected = %.20lf + %.20lf",
                                   b_d[first_err_idx * T_DATA_STRIDE],
                                   b_d[first_err_idx * T_DATA_STRIDE + 1]);
            AOCLFFTZ_LOG_FORMATTED(DEBUG, logger_mode,
                                   "  got      = %.20lf + %.20lf",
                                   a_d[first_err_idx * T_DATA_STRIDE],
                                   a_d[first_err_idx * T_DATA_STRIDE + 1]);
        }
        AOCLFFTZ_LOG_FORMATTED(DEBUG, logger_mode,
                               "  max abs error = %.20lf (%8.5le)",
                               first_abs_err, first_abs_err);
        AOCLFFTZ_LOG_FORMATTED(DEBUG, logger_mode,
                               "Max absolute error at index %td", max_err_idx);
        AOCLFFTZ_LOG_FORMATTED(DEBUG, logger_mode,
                               "  expected = %.20lf + %.20lf",
                               b_d[max_err_idx * T_DATA_STRIDE],
                               b_d[max_err_idx * T_DATA_STRIDE + 1]);
        AOCLFFTZ_LOG_FORMATTED(DEBUG, logger_mode,
                               "  got      = %.20lf + %.20lf",
                               a_d[max_err_idx * T_DATA_STRIDE],
                               a_d[max_err_idx * T_DATA_STRIDE + 1]);
        AOCLFFTZ_LOG_FORMATTED(DEBUG, logger_mode,
                               "  max abs error = %.20lf (%8.5le)", max_abs_err,
                               max_abs_err);
        return VERIFICATION_FAILURE;
    }
    return BENCH_SUCCESS;
}

#endif // AOCLFFTZ_COREBENCH_UTILS_H