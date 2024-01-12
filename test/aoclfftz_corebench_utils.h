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
#include "utils/complex_utils.h"
#include "utils/utils.h"

// Defining DATA_STRIDE in corebench internally to avoid using internal headers
#define T_DATA_STRIDE 2
#define DATA_STRIDE 2

// TODO : something else maybe ?
#define IN_STRIDE 0
#define OUT_STRIDE 1

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
        if (vec_rank > 3)                                                      \
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
                printf("\nUnsupported problem size provided.\nOnly 1D vec is " \
                       "supported.\n");                                        \
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
        UINT32 is_align = params->aligned_alloc;                                \
        ALLOC_UNINIT(p_desc->dims, dim_t, sizeof(dim_t) * p_desc->dim_rank,    \
                        is_align);                                             \
        for (INT32 i = 0; i < p_desc->dim_rank; i++)                           \
        {                                                                      \
            p_desc->dims[i].n = (dt_t)params->dims[i].n;                       \
            p_desc->dims[i].in_stride = (dt_t)params->dims[i].in_stride;       \
            p_desc->dims[i].out_stride = (dt_t)params->dims[i].out_stride;     \
        }                                                                      \
        ALLOC_UNINIT(p_desc->vecs, dim_t, sizeof(dim_t) * p_desc->vec_rank,    \
                        is_align);                                             \
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
#define DESTROY_PD(p_desc, is_align)                                                     \
    {                                                                          \
        if (p_desc != NULL)                                                    \
        {                                                                      \
            FREE_ALLOCATED_MEM(p_desc->dims, is_align);                        \
            FREE_ALLOCATED_MEM(p_desc->vecs, is_align);                        \
            FREE_ALLOCATED_MEM(p_desc, is_align);                              \
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
        UINT32 is_align = src->aligned_alloc;                                  \
        ALLOC_UNINIT(dst, aoclfftz_bench_params_t,                             \
                        sizeof(aoclfftz_bench_params_t), is_align);            \
        memcpy(dst, src, sizeof(aoclfftz_bench_params_t));                     \
        ALLOC_UNINIT(dst->dims, aoclfftz_dim_t_64_,                            \
                        sizeof(aoclfftz_dim_t_64_) * src->dim_rank, is_align); \
        ALLOC_UNINIT(dst->vecs, aoclfftz_dim_t_64_,                            \
                        sizeof(aoclfftz_dim_t_64_) * src->vec_rank, is_align); \
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
        for (INT32 i = 0; i < params->vec_rank; i++)                           \
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
            PREPARE_TIMESHIFT_TEST_INPUTS_IMPL(in1, in2, n, m, imap,           \
                                               DOUBLE);                        \
        }                                                                      \
    }

#define PREPARE_TIMESHIFT_TEST_INPUTS_IMPL(in1, in2, n, m, imap, dt_t)         \
    {                                                                          \
        dt_t *in1_t = (dt_t *)in1;                                             \
        dt_t *in2_t = (dt_t *)in2;                                             \
        /* Handle overflow to avoid negative indexing */                       \
        for (INTP idx = 0; idx < n; idx++)                                     \
        {                                                                      \
            for (INTP is = 0; is < 1; is++)                                    \
            {                                                                  \
                INTP src = imap[(idx + (n - m)) % n + is] *                    \
                           T_DATA_STRIDE;                                      \
                INTP dst = imap[idx + is] * T_DATA_STRIDE;                     \
                in2_t[dst] = in1_t[src];                                       \
                in2_t[dst + 1] = in1_t[src + 1];                               \
            }                                                                  \
        }                                                                      \
    }

#define PREPARE_TIMESHIFT_TEST_OUTPUTS(out1, out_combined, n, m, omap, dir,    \
                                       precision)                              \
    {                                                                          \
        if (precision == FLOAT_P)                                              \
        {                                                                      \
            PREPARE_TIMESHIFT_TEST_OUTPUTS_IMPL(out1, out_combined, n, m,      \
                                                omap, dir, FLOAT);             \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            PREPARE_TIMESHIFT_TEST_OUTPUTS_IMPL(out1, out_combined, n, m,      \
                                                omap, dir, DOUBLE);            \
        }                                                                      \
    }

#define PREPARE_TIMESHIFT_TEST_OUTPUTS_IMPL(out1, out_combined, n, m, omap,    \
                                            dir, dt_t)                         \
    {                                                                          \
        dt_t *out1_t = (dt_t *)out1;                                           \
        dt_t *out_combined_t = (dt_t *)out_combined;                           \
        dt_t cmul_temp[T_DATA_STRIDE] = {0.0, 0.0};                            \
        dt_t e_k[T_DATA_STRIDE] = {1.0, 0.0};                                  \
        dt_t sign = (dir == FORWARD) ? -1.0 : 1.0;                             \
        for (INTP k = 0; k < n; k++)                                           \
        {                                                                      \
            dt_t angle = (sign * BENCH_2_PI * k / n);                          \
            EULER(angle, e_k);                                                 \
            for (INTP i = 0; i < m; i++)                                       \
            {                                                                  \
                CMUL(out1_t + omap[k * m + i] * T_DATA_STRIDE, e_k,            \
                     out_combined_t + omap[k * m + i] * T_DATA_STRIDE,         \
                     cmul_temp);                                               \
            }                                                                  \
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

#define INIT_ERR_COORDS(arr, length, val)                                      \
    {                                                                          \
        for (INTP idx = 0; idx < length; ++idx)                                \
        {                                                                      \
            arr[idx] = val;                                                    \
        }                                                                      \
    }

#define PRINT_ERR_COORDS(enablelog, arr, dim_length, vec_length)               \
    {                                                                          \
        if(DEBUG <= enablelog)                                                 \
        {                                                                      \
            INTP idx;                                                          \
            for (idx = vec_length-1; idx > 0; idx--)                           \
            {                                                                  \
                printf("%ldx",arr[dim_length + idx]);                          \
            }                                                                  \
            printf("%ldv",arr[dim_length + idx]);                              \
            for (INTP idx = dim_length-1; idx >= 0; idx--)                     \
            {                                                                  \
                printf("[%ld]",arr[idx]);                                      \
            }                                                                  \
            printf("\n");                                                      \
        }                                                                      \
    }

/**
 * @brief angle = angle * [(i0+k0)/n0 + (i0+k0)/n0 + ... + (iR+kR)/nR]
 * where R = dim rank
 *
 */
#define UPDATE_ANGLE(angle, in, out, dims, rank, dt_t)                         \
    {                                                                          \
        dt_t x = 0.0;                                                          \
        for (INTP i = 0; i < rank; i++)                                        \
        {                                                                      \
            x += (((dt_t)in[i] * out[i]) / dims[i].n);                         \
        }                                                                      \
        angle = angle * x;                                                     \
    }

/**
 * @brief incrmement the nD counter by 1 value, used to map 1D index to nD
 *
 */
#define INCREMENT_ND_COUNTER(cur_dims, max_dims, rank)                         \
    {                                                                          \
        UINT8 incremented = 0;                                                 \
        INTP cur_rank = 0;                                                     \
        do                                                                     \
        {                                                                      \
            if (cur_dims[cur_rank] < max_dims[cur_rank].n - 1)                 \
            {                                                                  \
                cur_dims[cur_rank]++;                                          \
                incremented = 1;                                               \
            }                                                                  \
            else                                                               \
            {                                                                  \
                for (INTP i = 0; i <= cur_rank; i++)                           \
                {                                                              \
                    cur_dims[i] = 0;                                           \
                }                                                              \
                cur_rank++;                                                    \
            }                                                                  \
        } while (!incremented && cur_rank < rank);                             \
    }

/**
 * @brief reset the nD counter values to 0
 *
 */
#define RESET_ND_COUNTER(cur_dims, rank)                                       \
    {                                                                          \
        for (INT32 i = 0; i < rank; i++)                                       \
        {                                                                      \
            cur_dims[i] = 0;                                                   \
        }                                                                      \
    }

/**
 * @brief prepare suitable selector time unit
 *
 */
#define ADJUST_SELECTOR_TIME_UNIT(time_taken, time_unit)                       \
    {                                                                          \
        /* print time in seconds */                                            \
        if (time_taken > 1E9)                                                  \
        {                                                                      \
            time_taken *= 1E-9;                                                \
            STRCPY(time_unit, 3, "s");                                         \
        }                                                                      \
        /* print time in milli-seconds */                                      \
        else if (time_taken > 1E6)                                             \
        {                                                                      \
            time_taken *= 1E-6;                                                \
            STRCPY(time_unit, 3, "ms");                                        \
        }                                                                      \
        /* print time in micro-seconds */                                      \
        else if (time_taken > 1E3)                                             \
        {                                                                      \
            time_taken *= 1E-3;                                                \
            STRCPY(time_unit, 3, "us");                                        \
        }                                                                      \
        /* print time in nano-seconds */                                       \
        else                                                                   \
        {                                                                      \
            STRCPY(time_unit, 3, "ns");                                        \
        }                                                                      \
    }

// Function pointers
VOID (*prepare_input_data)(VOID *, INTP, INTP *, INT32);
VOID (*dft_ref)(aoclfftz_bench_params_t *, VOID *, INTP *, INTP *);
INT32 (*compare)(VOID *, VOID *, INTP, INTP *, DOUBLE tol, INT32);

// Function declarations
INT32 set_flag(aoclfftz_bench_params_t *params);
INT32 find_dim_rank(CHAR *arg);
INT32 allocate_and_fill_dims_vecs(CHAR *arg, INT32 dim_rank, INT32 vec_rank,
                                  aoclfftz_dim_t_64_ **dims,
                                  aoclfftz_dim_t_64_ **vecs,
                                  INTP default_stride);
VOID prepare_input_data_f(VOID *input, INTP n, INTP *idx_map, INT32 input_type);
VOID prepare_input_data_d(VOID *input, INTP n, INTP *idx_map, INT32 input_type);
VOID dft_ref_f(aoclfftz_bench_params_t *params, VOID *out_buf, INTP *in_idx_map,
               INTP *out_idx_map);
VOID dft_ref_d(aoclfftz_bench_params_t *params, VOID *out_buf, INTP *in_idx_map,
               INTP *out_idx_map);
INT32 compare_f(VOID *a, VOID *b, INTP n, INTP *idx_map, DOUBLE tol,
                INT32 logger_mode);
INT32 compare_d(VOID *a, VOID *b, INTP n, INTP *idx_map, DOUBLE tol,
                INT32 logger_mode);
INT32 get_fusable_dims_ref(aoclfftz_bench_params_t *params, INT32 dim_rank);
INTP calculate_size(aoclfftz_dim_t_64_ *dims, INT32 rank);
VOID calculate_buffer_sizes(aoclfftz_bench_params_t *params,
                            INTP *in_buffer_size, INTP *out_buffer_size);
INT32 check_inplace_strides(aoclfftz_dim_t_64_ *dims, aoclfftz_dim_t_64_ *vecs,
                            INTP dim_rank, INTP vec_rank);
VOID prepare_index_map(aoclfftz_bench_params_t *params, INTP *in_idx_map,
                       INTP *out_idx_map);
VOID compute_index_map(INTP *in_idx_map, INTP *out_idx_map, INTP *src_idx,
                       INTP dst_in_idx, INTP dst_out_idx,
                       aoclfftz_dim_t_64_ *dims, INT32 rank);
INT32 caliberate_iterations(VOID *handle, DOUBLE min_bench_time);

// Function definitions

/**
 * @brief Free the structures used for FFT problem
 *
 * @param params aoclfftz_bench_params_t type contains parsed arguments
 * @return VOID
 */
VOID destroy_bench_param(aoclfftz_bench_params_t *params)
{
    if (params != NULL)
    {
        UINT32 is_align = params->aligned_alloc;
        FREE_ALLOCATED_MEM(params->in, is_align);
        if(params->res_placement == OUT_OF_PLACE)
        {
            FREE_ALLOCATED_MEM(params->out, is_align);
        }
        FREE_ALIGN_ALLOCATED_MEM(params->dims);
        FREE_ALIGN_ALLOCATED_MEM(params->vecs);
        FREE_ALIGN_ALLOCATED_MEM(params);
    }
}

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

// FIXME : this needs to be modified, club with allocate or exploit this result
INT32 find_dim_vec_ranks(CHAR *arg, INT32 *dim_rank, INT32 *vec_rank)
{
    INT32 dr = 1;
    INT32 vr = 1; // can we retain this as O when no vec ?
    for (INT32 i = 0; i < strlen(arg); i++)
    {
        if (arg[i] == 'x')
        {
            dr++;
        }
        else if (arg[i] == 'v')
        {
            vr = dr;
            dr = 1;
        }
    }
    (*dim_rank) = dr;
    (*vec_rank) = vr;
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
    ALLOC_ALIGN_INIT((*dims),
        aoclfftz_dim_t_64_, dim_rank * sizeof(aoclfftz_dim_t_64_));
    ALLOC_ALIGN_INIT((*vecs),
        aoclfftz_dim_t_64_, vec_rank * sizeof(aoclfftz_dim_t_64_));
    INT32 max_rank = dim_rank > vec_rank ? dim_rank : vec_rank;
    aoclfftz_dim_t_64_ *desc = NULL;
    ALLOC_ALIGN_INIT(desc, aoclfftz_dim_t_64_,
                     max_rank * sizeof(aoclfftz_dim_t_64_));

    INT32 is_stride = 0;
    INT32 is_vec_stride = 0;
    INT32 rank_count = 0;
    INT32 vec_count = 0;
    CHAR last_char = '\0';
    INT32 start = 0;
    CHAR val_str[strlen(arg) + 1];
    INT32 status = PARSER_SUCCESS;
    for (INT32 i = 0; i < strlen(arg); ++i)
    {
        if (arg[i] == 'x' || arg[i] == 'X')
        {
            // TODO: Fix this code to support more dimensions
            if ((is_stride != 0 && is_stride != 2) ||
                (is_vec_stride != 0 && is_vec_stride != 2))
            {
                status = SIZE_PARSING_ERROR;
                goto exit_func;
            }
            is_stride = 0;
            rank_count++;
            last_char = 'x';
        }
        else if (arg[i] == 'v' || arg[i] == 'V')
        {
            // by default the data is stored in dims always
            // once "v" is encountered, its moved to vecs and then dims is reset
            // FIXME : this needs to be fixed properly
            for(INT32 i = vec_rank - 1, j = 0; i >= 0; i--, j++)
            {
                (*vecs)[i].n = desc[j].n;
                (*vecs)[i].in_stride =
                    (is_stride >= 1) ? desc[j].in_stride : 0;
                (*vecs)[i].out_stride =
                    (is_stride == 2) ? desc[j].out_stride : 0;
            }
            // reset buffer to store dims config
            memset(desc, 0, max_rank * sizeof(aoclfftz_dim_t_64_));
            rank_count = 0;
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
                status = SIZE_PARSING_ERROR;
                goto exit_func;
            }
            if (last_char == 'v')
            {
                if (is_vec_stride == 0) // no strides encountered till now
                {
                    desc[rank_count].n = val;
                }
                else if (is_vec_stride == 1)
                {
                    desc[rank_count].in_stride = val;
                }
                else if (is_vec_stride == 2)
                {
                    desc[rank_count].out_stride = val;
                }
            }
            else
            {
                if (is_stride == 0)
                {
                    desc[rank_count].n = val;
                }
                else if (is_stride == 1)
                {
                    desc[rank_count].in_stride = val;
                }
                else if (is_stride == 2)
                {
                    desc[rank_count].out_stride = val;
                }
            }
            i--;
        }
        else
        {
            status =  SIZE_PARSING_ERROR;
            goto exit_func;
        }
    }

    // copy desc to dims in reverse
    for(INT32 i = dim_rank - 1, j = 0; i >= 0; i--, j++)
    {
        (*dims)[i].n = desc[j].n;
        (*dims)[i].in_stride = desc[j].in_stride;
        (*dims)[i].out_stride = desc[j].out_stride;
    }

    // TODO: if only one stride value is provided, then use it for in-stride and out-stride
    // for now, either no stride or both in/out strides should be provided
    // only in_stride will not be parsed
    if ((is_stride != 0 && is_stride != 2) ||
        (is_vec_stride != 0 && is_vec_stride != 2))
    {
        status = SIZE_PARSING_ERROR;
        goto exit_func;
    }

    // TODO : this needs to be moved at the last and done for ND
    // if both strides are provided, check if vec strides are as expected
    // if(is_stride == 2 && is_vec_stride == 2)
    // {
    //     for(INT32 i = 0; i < dim_rank; i++)
    //     {
    //         if( ((*vecs)[i].in_stride < ((*dims)[i].in_stride *
    //         (*dims)[i].n)) ||
    //             ((*vecs)[i].out_stride < ((*dims)[i].out_stride *
    //             (*dims)[i].n)))
    //         {
    //             status = SIZE_PARSING_ERROR;
    //             goto exit_func;
    //         }
    //     }
    // }

    // set strides for dims if not provided
    for(INT32 i = 0; i < dim_rank; i++)
    {
        if((*dims)[i].in_stride == 0)
        {
            (*dims)[i].in_stride = (i == 0) ?
                1 : ((*dims)[i - 1]. n * (*dims)[i - 1].in_stride);
        }
        if((*dims)[i].out_stride == 0)
        {
            (*dims)[i].out_stride = (i == 0) ?
                1 : ((*dims)[i - 1]. n * (*dims)[i - 1].out_stride);
        }
    }

    // set strides for vecs if not provided
    for(INT32 i = 0; i < vec_rank; i++)
    {
        if((*vecs)[i].in_stride == 0)
        {
            // stride of fcd should atleast be the length of dims
            (*vecs)[i].in_stride = (i == 0) ?
                (*dims)[dim_rank -1].n * (*dims)[dim_rank - 1].in_stride :
                    ((*vecs) [i - 1]. n * (*vecs) [i - 1].in_stride);
        }
        if((*vecs)[i].out_stride == 0)
        {
            // stride of fcd should atleast be the length of dims
            (*vecs)[i].out_stride = (i == 0) ?
                (*dims)[dim_rank -1].n * (*dims)[dim_rank - 1].out_stride :
                    ((*vecs) [i - 1]. n * (*vecs) [i - 1].out_stride);
        }
    }

    // FIXME : this may not be necessary ?
    // Move the vecs to dims and fill the default values for vecs if they are
    // not provided
    if (vec_count == 0 && vec_rank > 0)
    {
        for (INT32 i = 0; i < vec_rank; ++i)
        {
            (*vecs)[i].n = 1;
            (*vecs)[i].in_stride = default_stride;
            (*vecs)[i].out_stride = default_stride;
        }
    }

    CHECK_SUPPORTED_DIMS(dims, vecs, dim_rank, vec_rank, status);

exit_func :
    FREE_ALIGN_ALLOCATED_MEM(desc);
    return status;
}

/**
 * @brief Prepare FLOAT input data of size `n * stride`.
 *
 * @param input array to store input data
 * @param n input size
 * @param idx_map index map of size n, specify NULL to disable mapping
 * @param input_type type of input data : RANDOM, IMPULSE or SIGNAL
 * @return VOID
 */
VOID prepare_input_data_f(VOID *input, INTP n, INTP *idx_map, INT32 input_type)
{
    FLOAT *input_f = (FLOAT *)input;
    INTP idx = 0;
    // random input
    // range: [-10.0, 10.0)
    if (input_type == RANDOM_INPUT)
    {
        if (idx_map == NULL)
        {
            for (idx = 0; idx < n; ++idx)
            {
                input_f[idx + T_DATA_STRIDE] =
                    (20.0f / (FLOAT)RAND_MAX) * rand() - 10.0f;
                input_f[idx + T_DATA_STRIDE + 1] =
                    (20.0f / (FLOAT)RAND_MAX) * rand() - 10.0f;
            }
        }
        else
        {
            for (idx = 0; idx < n; ++idx)
            {
                input_f[idx_map[idx] + T_DATA_STRIDE] =
                    (20.0f / (FLOAT)RAND_MAX) * rand() - 10.0f;
                input_f[idx_map[idx] + T_DATA_STRIDE + 1] =
                    (20.0f / (FLOAT)RAND_MAX) * rand() - 10.0f;
            }
        }
    }
    // impulse input
    // range: [-10.0, 10.0)
    else if (input_type == IMPULSE_INPUT)
    {
        if (idx_map == NULL)
        {
            idx = rand() % n;
        }
        else
        {
            idx = idx_map[rand() % n];
        }
        memset(input_f, 0, n * T_DATA_STRIDE);
        input_f[idx * T_DATA_STRIDE] =
            (20.0f / (FLOAT)RAND_MAX) * rand() - 10.0f;
        input_f[idx * T_DATA_STRIDE + 1] =
            (20.0f / (FLOAT)RAND_MAX) * rand() - 10.0f;
    }
    // sinusoidal signal input
    else if (input_type == SINUSOIDAL_SIGNAL_INPUT)
    {
        // Sine wave cycles
        INTP cycles = (rand() % (n / 2)) + 2;
        FLOAT size = BENCH_2_PI * cycles;
        // Shift the origin of the wave from 0 to a positive integer `shift`,
        // shift range: [0, n)
        INTP shift = rand() % n;
        // scale the amplitude of the wave by `scale` times,
        // scale range: [0.0, 5.0)
        FLOAT scale = ((FLOAT)rand() / (FLOAT)RAND_MAX) * 5.0f;
        if (idx_map == NULL)
        {
            for (INTP i = 0; i < n; i++)
            {
                input_f[((i + shift) % n) * T_DATA_STRIDE] =
                    sinf((FLOAT)(i * size) / n) * scale;
                input_f[((i + shift) % n) * T_DATA_STRIDE + 1] = 0.0f;
            }
        }
        else
        {
            for (INTP i = 0; i < n; i++)
            {
                input_f[idx_map[(i + shift) % n] * T_DATA_STRIDE] =
                    sinf((FLOAT)(i * size) / n) * scale;
                input_f[idx_map[(i + shift) % n] * T_DATA_STRIDE + 1] = 0.0f;
            }
        }
    }
}

/**
 * @brief Prepare DOUBLE input data of size `n * stride`.
 *
 * @param input array to store input data
 * @param n input size
 * @param idx_map index map of size n, specify NULL to disable mapping
 * @param input_type type of input data : RANDOM, IMPULSE or SIGNAL
 * @return VOID
 */
VOID prepare_input_data_d(VOID *input, INTP n, INTP *idx_map, INT32 input_type)
{
    DOUBLE *input_d = (DOUBLE *)input;
    INTP idx = 0;
    // random input
    // range: [-10.0, 10.0)
    if (input_type == RANDOM_INPUT)
    {
        if (idx_map == NULL)
        {
            for (idx = 0; idx < n; ++idx)
            {
                input_d[idx + T_DATA_STRIDE] =
                    (20.0 / RAND_MAX) * rand() - 10.0;
                input_d[idx + T_DATA_STRIDE + 1] =
                    (20.0 / RAND_MAX) * rand() - 10.0;
            }
        }
        else
        {
            for (idx = 0; idx < n; ++idx)
            {
                input_d[idx_map[idx] + T_DATA_STRIDE] =
                    (20.0 / RAND_MAX) * rand() - 10.0;
                input_d[idx_map[idx] + T_DATA_STRIDE + 1] =
                    (20.0 / RAND_MAX) * rand() - 10.0;
            }
        }
    }
    // impulse input
    // range: [-10.0, 10.0)
    else if (input_type == IMPULSE_INPUT)
    {
        if (idx_map == NULL)
        {
            idx = rand() % n;
        }
        else
        {
            idx = idx_map[rand() % n];
        }
        memset(input_d, 0, n * T_DATA_STRIDE);
        input_d[idx * T_DATA_STRIDE] = (20.0 / RAND_MAX) * rand() - 10.0;
        input_d[idx * T_DATA_STRIDE + 1] = (20.0 / RAND_MAX) * rand() - 10.0;
    }
    // sinusoidal signal input
    else if (input_type == SINUSOIDAL_SIGNAL_INPUT)
    {
        // Sine wave cycles
        INTP cycles = (rand() % (n / 2)) + 2;
        DOUBLE size = BENCH_2_PI * cycles;
        // Shift the origin of the wave from 0 to a positive integer `shift`,
        // shift range: [0, n)
        INTP shift = rand() % n;
        // scale the amplitude of the wave by `scale` times,
        // scale range: [0.0, 5.0)
        DOUBLE scale = ((DOUBLE)rand() / RAND_MAX) * 5.0;
        if (idx_map == NULL)
        {
            for (INTP i = 0; i < n; i++)
            {
                input_d[((i + shift) % n) * T_DATA_STRIDE] =
                    sin((DOUBLE)(i * size) / n) * scale;
                input_d[((i + shift) % n) * T_DATA_STRIDE + 1] = 0.0;
            }
        }
        else
        {
            for (INTP i = 0; i < n; i++)
            {
                input_d[idx_map[(i + shift) % n] * T_DATA_STRIDE] =
                    sin((DOUBLE)(i * size) / n) * scale;
                input_d[idx_map[(i + shift) % n] * T_DATA_STRIDE + 1] = 0.0;
            }
        }
    }
}

/**
 * @brief DFT reference implementation for FLOAT type
 *
 * @param params aoclfftz_bench_params_t struct containing the req info
 * @param out_buf buffer to store the DFT output
 * @param in_idx_map index map input indices
 * @param out_idx_map index map for output indices
 * @return VOID
 */
VOID dft_ref_f(aoclfftz_bench_params_t *params, VOID *out_buf, INTP *in_idx_map,
               INTP *out_idx_map)
{
    FLOAT e[T_DATA_STRIDE], mul_buf[T_DATA_STRIDE];
    FLOAT sign = params->dir == BACKWARD ? 1.0 : -1.0;
    FLOAT *in_d = (FLOAT *)params->in;
    FLOAT *out_d = (FLOAT *)out_buf;
    INT32 rank = params->dim_rank;
    aoclfftz_dim_t_64_ *dims = params->dims;
    INTP n = calculate_size(params->dims, params->dim_rank);
    INTP batches = calculate_size(params->vecs, params->vec_rank);
    UINT32 is_align = params->aligned_alloc;

    INTP *in_counter = NULL;
    ALLOC_INIT(in_counter, INTP, rank * sizeof(INTP), is_align);
    INTP *out_counter = NULL;
    ALLOC_INIT(out_counter, INTP, rank * sizeof(INTP), is_align);

    // iterate over the total batches
    for (INTP b = 0; b < batches; b++)
    {
        RESET_ND_COUNTER(out_counter, rank);
        // iterate over output points
        for (INTP k = 0; k < n; k++)
        {
            INTP out_idx = out_idx_map[b * n + k] * T_DATA_STRIDE;
            RESET_ND_COUNTER(in_counter, rank);
            // iterate over input points
            for (INTP i = 0; i < n; i++)
            {
                INTP in_idx = in_idx_map[b * n + i] * T_DATA_STRIDE;
                FLOAT angle = sign * BENCH_2_PI;
                // angle = angle * [(i0+k0)/n0 + (i0+k0)/n0 + ... + (iR+kR)/nR]
                UPDATE_ANGLE(angle, in_counter, out_counter, dims, rank, FLOAT);
                e[0] = cosf(angle);
                e[1] = sinf(angle);

                mul_buf[0] = (in_d[in_idx] * e[0]) - (in_d[in_idx + 1] * e[1]);
                mul_buf[1] = (in_d[in_idx] * e[1]) + (in_d[in_idx + 1] * e[0]);

                out_d[out_idx] = out_d[out_idx] + mul_buf[0];
                out_d[out_idx + 1] = out_d[out_idx + 1] + mul_buf[1];

                INCREMENT_ND_COUNTER(in_counter, dims, rank);
            }
            INCREMENT_ND_COUNTER(out_counter, dims, rank);
        }
    }
    FREE_ALLOCATED_MEM(in_counter, is_align);
    FREE_ALLOCATED_MEM(out_counter, is_align);
}

/**
 * @brief DFT reference implementation for DOUBLE type
 *
 * @param params aoclfftz_bench_params_t struct containing the req info
 * @param out_buf buffer to store the DFT output
 * @param in_idx_map index map input indices
 * @param out_idx_map index map for output indices
 * @return VOID
 */
VOID dft_ref_d(aoclfftz_bench_params_t *params, VOID *out_buf, INTP *in_idx_map,
               INTP *out_idx_map)
{
    DOUBLE e[T_DATA_STRIDE], mul_buf[T_DATA_STRIDE];
    DOUBLE sign = params->dir == BACKWARD ? 1.0 : -1.0;
    DOUBLE *in_d = (DOUBLE *)params->in;
    DOUBLE *out_d = (DOUBLE *)out_buf;
    INT32 rank = params->dim_rank;
    aoclfftz_dim_t_64_ *dims = params->dims;
    INTP n = calculate_size(params->dims, params->dim_rank);
    INTP batches = calculate_size(params->vecs, params->vec_rank);
    UINT32 is_align = params->aligned_alloc;

    INTP *in_counter = NULL;
    ALLOC_INIT(in_counter, INTP, rank * sizeof(INTP), is_align);
    INTP *out_counter = NULL;
    ALLOC_INIT(out_counter, INTP, rank * sizeof(INTP), is_align);

    // iterate over the total batches
    for (INTP b = 0; b < batches; b++)
    {
        RESET_ND_COUNTER(out_counter, rank);
        // iterate over output points
        for (INTP k = 0; k < n; k++)
        {
            INTP out_idx = out_idx_map[b * n + k] * T_DATA_STRIDE;
            RESET_ND_COUNTER(in_counter, rank);
            // iterate over input points
            for (INTP i = 0; i < n; i++)
            {
                INTP in_idx = in_idx_map[b * n + i] * T_DATA_STRIDE;
                DOUBLE angle = sign * BENCH_2_PI;
                // angle = angle * [(i0+k0)/n0 + (i0+k0)/n0 + ... + (iR+kR)/nR]
                UPDATE_ANGLE(angle, in_counter, out_counter, dims, rank,
                             DOUBLE);
                e[0] = cos(angle);
                e[1] = sin(angle);

                mul_buf[0] = (in_d[in_idx] * e[0]) - (in_d[in_idx + 1] * e[1]);
                mul_buf[1] = (in_d[in_idx] * e[1]) + (in_d[in_idx + 1] * e[0]);

                out_d[out_idx] = out_d[out_idx] + mul_buf[0];
                out_d[out_idx + 1] = out_d[out_idx + 1] + mul_buf[1];

                INCREMENT_ND_COUNTER(in_counter, dims, rank);
            }
            INCREMENT_ND_COUNTER(out_counter, dims, rank);
        }
    }
    FREE_ALLOCATED_MEM(in_counter, is_align);
    FREE_ALLOCATED_MEM(out_counter, is_align);
}

/**
 * @brief Compare the two data of length n (DOUBLE type)
 *        Used to compare output with reference output.
 *
 * @param a first data
 * @param b second data
 * @param n data length
 * @param idx_map index map
 * @param tol error tolerance value
 * @param logger_mode logger mode from bench_params
 * @return INT32 if the data points are same, return 1 else 0
 */
INT32 compare_f(VOID *a, VOID *b, INTP n, INTP *idx_map, DOUBLE tol,
                INT32 logger_mode)
{
    FLOAT *a_f = (FLOAT *)a;
    FLOAT *b_f = (FLOAT *)b;
    FLOAT max_abs_err = 0.0;
    FLOAT max_mag = 0.0;
    INTP max_err_idx = -1;
    INTP first_err_idx = INT64_MAX;
    FLOAT first_abs_err = 0.0;
    for (INTP i = 0; i < n; i++)
    {
        INTP idx = idx_map[i];
        FLOAT abs_err = fmaxf(
            fabsf(a_f[idx * T_DATA_STRIDE] - b_f[idx * T_DATA_STRIDE]),
            fabsf(a_f[idx * T_DATA_STRIDE + 1] - b_f[idx * T_DATA_STRIDE + 1]));
        FLOAT mag = fmin(fmaxf(fabsf(a_f[idx * T_DATA_STRIDE]),
                               fabsf(a_f[idx * T_DATA_STRIDE + 1])),
                          fmaxf(fabsf(b_f[idx * T_DATA_STRIDE]),
                               fabsf(b_f[idx * T_DATA_STRIDE + 1])));
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
    AOCLFFTZ_LOG_FORMATTED(INFO, logger_mode, "Error = %.10f (%.5e)", rel_err,
                           rel_err);
    if (rel_err > tol)
    {
        AOCLFFTZ_LOG_FORMATTED(DEBUG, logger_mode,
                               "Relative error  = %.10f (%8.5e)", rel_err,
                               rel_err);
        AOCLFFTZ_LOG_FORMATTED(DEBUG, logger_mode,
                               "Tolerance       = %.10f (%8.5e)", tol, tol);
        if (first_err_idx < INT64_MAX)
        {
            AOCLFFTZ_LOG_FORMATTED(DEBUG, logger_mode,
                                   "First absolute error at index %td",
                                   first_err_idx);
            AOCLFFTZ_LOG_FORMATTED(DEBUG, logger_mode,
                                   "  expected = %.10f + %.10f",
                                   b_f[first_err_idx * T_DATA_STRIDE],
                                   b_f[first_err_idx * T_DATA_STRIDE + 1]);
            AOCLFFTZ_LOG_FORMATTED(DEBUG, logger_mode,
                                   "  got      = %.10f + %.10f",
                                   a_f[first_err_idx * T_DATA_STRIDE],
                                   a_f[first_err_idx * T_DATA_STRIDE + 1]);
        }
        AOCLFFTZ_LOG_FORMATTED(DEBUG, logger_mode,
                               "  max abs error = %.10f (%8.5le)",
                               first_abs_err, first_abs_err);
        AOCLFFTZ_LOG_FORMATTED(DEBUG, logger_mode,
                               "Max absolute error at index %td", max_err_idx);
        AOCLFFTZ_LOG_FORMATTED(DEBUG, logger_mode,
                               "  expected = %.10f + %.10f",
                               b_f[max_err_idx * T_DATA_STRIDE],
                               b_f[max_err_idx * T_DATA_STRIDE + 1]);
        AOCLFFTZ_LOG_FORMATTED(DEBUG, logger_mode,
                               "  got      = %.10f + %.10f",
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
 * @param idx_map index map
 * @param tol error tolerance value
 * @param logger_mode logger mode from bench_params
 * @return INT32 if the data points are same, return 1 else 0
 */
INT32 compare_d(VOID *a, VOID *b, INTP n, INTP *idx_map, DOUBLE tol,
                INT32 logger_mode)
{
    DOUBLE *a_d = (DOUBLE *)a;
    DOUBLE *b_d = (DOUBLE *)b;
    DOUBLE max_abs_err = 0.0;
    DOUBLE max_mag = 0.0;
    INTP max_err_idx = -1;
    INTP first_err_idx = INT64_MAX;
    DOUBLE first_abs_err = 0.0;
    for (INTP i = 0; i < n; i++)
    {
        INTP idx = idx_map[i];
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

/**
 * @brief calculates the total size without strides
 *
 * @param dims holds dims related info
 * @param rank rank of the provided dims
 * @return INTP length of the input
 */
INTP calculate_size(aoclfftz_dim_t_64_ *dims, INT32 rank)
{
    INTP len = 1;

    for (INTP i = 0; i < rank; i++)
    {
        len = len * dims[i].n;
    }

    return len;
}

/**
 * @brief Function to find the required buffer size for memory allocation.
 * calculates the total length of input & output buffers with strides included
 *
 * @param params aoclfftz_bench_params_t struct containing the req info
 * @param in_buffer_size register to store input buffer size
 * @param out_buffer_size register to store output buffer size
 * @return VOID
 */
VOID calculate_buffer_sizes(aoclfftz_bench_params_t *params,
                            INTP *in_buffer_size, INTP *out_buffer_size)
{
    // Data arrangement considered :
    // [1, 2, 3, 4]<0, 0>[5, 6, 7, 8]<0, 0>[9, 10, 11, 12]
    // <---vec stride--->
    // <-------------(Batches -1)---------><--- Problem size * dim stride --->
    // ((Batches -1) * (vec_stride)) + (Problem size * dim stride)

    INT32 dim_rank = params->dim_rank;
    INT32 vec_rank = params->vec_rank;
    in_buffer_size[0] = 0;
    out_buffer_size[0] = 0;
    INT32 in_size = 1; // rank-0 problem where its a constant ?
    INT32 out_size = 1;

    for (INT32 i = 0; i < dim_rank; i++)
    {
        in_size += ((params->dims[i].n - 1) * (params->dims[i].in_stride));
        out_size += ((params->dims[i].n - 1) * (params->dims[i].out_stride));
    }

    for (INT32 i = 0; i < vec_rank; i++)
    {
        in_size += ((params->vecs[i].n - 1) * (params->vecs[i].in_stride));
        out_size += ((params->vecs[i].n - 1) * (params->vecs[i].out_stride));
    }

    in_buffer_size[0] = in_size;
    out_buffer_size[0] = out_size;
}

INT32 get_fusable_dims_ref(aoclfftz_bench_params_t *params, INT32 dim_rank)
{
    // FIXME : considered only cases where instride == outstride

    INT32 fusable_dims = 1; // default append batch to 1D

    if (params->dims[0].in_stride != params->dims[0].out_stride)
    {
        return fusable_dims;
    }
    INTP expected_stride = params->dims[0].n * params->dims[0].in_stride;
    for (INT32 i = 1; i < dim_rank; i++)
    {
        if (params->dims[i].in_stride != params->dims[i].out_stride)
        {
            break;
        }

        INTP actual_stride = params->dims[i].in_stride;
        if (expected_stride != actual_stride)
        {
            break;
        }
        fusable_dims += 1;
        expected_stride = expected_stride * params->dims[i].n;
    }

    return fusable_dims;
}

/**
 * @brief checks if input & output strides are the same for of an inplace
 * problem
 *
 * @param dims holds dims related stride info
 * @param vecs holds vecs related stride info
 * @param dim_rank rank of the dimension
 * @param vec_rank rank of the vector
 * @return INT32
 */
INT32 check_inplace_strides(aoclfftz_dim_t_64_ *dims, aoclfftz_dim_t_64_ *vecs,
                            INTP dim_rank, INTP vec_rank)
{
    for (INTP i = 0; i < dim_rank; i++)
    {
        if (dims[i].in_stride != dims[i].out_stride)
        {
            return SIZE_PARSING_ERROR;
        }
    }
    for (INTP i = 0; i < vec_rank; i++)
    {
        if (vecs[i].in_stride != vecs[i].out_stride)
        {
            return SIZE_PARSING_ERROR;
        }
    }

    return PARSER_SUCCESS;
}

/**
 * @brief prepare the index map to map non strided indices to strides ones
 *
 * index map is used to simplify the property tests for strided problems
 *
 * Example: for an 1D problem with 1D batch
 * Problem size : 3:6:6v4:1:1
 * Data arrangement considered :
 * [1, 2, 3, 4]<0, 0>[5, 6, 7, 8]<0, 0>[9, 10, 11, 12]
 * <---vec stride--->
 * <------------(Batches - 1)--------->
 *
 * data buffer   => [1, 2, 3, 4, 0, 0, 5, 6, 7, 8,  0,  0,  9, 10, 11, 12]
 * (here data in the strides are considered as 0)
 * indices       => [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
 * valid indices => [0, 1, 2, 3,       6, 7, 8, 9,         12, 13, 14, 15]
 * (ignoring the indices of strides)
 *
 * index map for this configuration in key, value representation :
 * key : valid data index for the given problem without strides
 * value : valid data index for the given problem with strides
 * [(0,0), (1,1), (2,2), (3,3), (4,6), (5,7), (6,8), (7,9), (8,12), (9,13), (10,14), (11,15)]
 * simplified index map (in array representation, where array index is the key) :
 * [0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15]
 *
 * @param params aoclfftz_bench_params_t struct containing the req info
 * @param in_idx_map buffer to store input index map
 * @param out_idx_map buffer to store output index map
 * @return VOID
 */
VOID prepare_index_map(aoclfftz_bench_params_t *params, INTP *in_idx_map,
                       INTP *out_idx_map)
{
    // combine dims and vecs
    aoclfftz_dim_t_64_ *combined_dims = NULL;
    ALLOC_UNINIT(combined_dims, aoclfftz_dim_t_64_,
            sizeof(aoclfftz_dim_t_64_) * (params->dim_rank + params->vec_rank),
            params->aligned_alloc);
    memcpy(combined_dims, params->dims,
           (sizeof(aoclfftz_dim_t_64_) * params->dim_rank));
    memcpy((combined_dims + params->dim_rank), params->vecs,
           (sizeof(aoclfftz_dim_t_64_) * params->vec_rank));

    INT32 combined_rank = params->dim_rank + params->vec_rank;
    INTP src_idx = 0;
    INTP dst_in_idx = 0;
    INTP dst_out_idx = 0;
    compute_index_map(in_idx_map, out_idx_map, &src_idx, dst_in_idx,
                      dst_out_idx, combined_dims, combined_rank);

    FREE_ALLOCATED_MEM(combined_dims, params->aligned_alloc);
}

/**
 * @brief recursive algorithm to compute index map
 *
 * @param in_idx_map buffer to store input index map
 * @param out_idx_map buffer to store output index map
 * @param src_idx source index for input and output buffers
 * @param dst_in_idx destination index for input buffer
 * @param dst_out_idx destination index for output buffer
 * @param dims aoclfftz_dim_t_64_ struct which holds problem size and strides
 * @param rank current rank of the nD problem
 * @return VOID
 */
VOID compute_index_map(INTP *in_idx_map, INTP *out_idx_map, INTP *src_idx,
                       INTP dst_in_idx, INTP dst_out_idx,
                       aoclfftz_dim_t_64_ *dims, INT32 rank)
{
    if (rank == 0)
    {
        in_idx_map[*src_idx] = dst_in_idx;
        out_idx_map[*src_idx] = dst_out_idx;
        (*src_idx)++;
    }
    else
    {
        INTP n = dims[rank - 1].n;
        INTP in_stride = dims[rank - 1].in_stride;
        INTP out_stride = dims[rank - 1].out_stride;
        for (INTP i = 0; i < n; i++)
        {
            compute_index_map(in_idx_map, out_idx_map, src_idx, dst_in_idx,
                              dst_out_idx, dims, rank - 1);
            dst_in_idx += in_stride;
            dst_out_idx += out_stride;
        }
    }
}

/**
 * @brief Computes the number of iterations for benchmarking
 *
 * @return INT32 iterations
 */
INT32 caliberate_iterations(VOID *handle, DOUBLE min_bench_time)
{
    DOUBLE minq_time = 1000; // minimum quantifiable time 1 us
    DOUBLE min_acceptable_time = min_bench_time; //min_acceptable_time
    INT32 increase_iterations = 1;
    DOUBLE cur_time = 0;
#ifdef WIN32
    timer clk_tick;
#endif
    timeVal start_time, end_time;
    initTimer(clk_tick);
    INT32 iters = 1;
    for(iters = 1; increase_iterations && iters < INT32_MAX; iters *= 5)
    {
        getTime(start_time);
        for (INT32 j = 0; j < iters; j++)
        {
            aoclfftz_execute(handle);
        }
        getTime(end_time);
        cur_time = (DOUBLE)diffTime(clk_tick, start_time, end_time);
        // if execution time is above the minimum quantifiable limit,
        // then stop the iterations
        if (cur_time >= minq_time)
        {
            increase_iterations = 0;
            cur_time = cur_time / 1000;   //change time unit from ns to us
            if(cur_time > min_acceptable_time)
            {
                return iters;
            }
            //Scaling the iteration for min_acceptable_time
            return (iters * min_acceptable_time / cur_time);
        }
    }
    return iters;
}

#endif // AOCLFFTZ_COREBENCH_UTILS_H
