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
#include "api/aoclfftz_internal.h"
#include "test/aoclfftz_corebench.h"
#include "test/utils/size_and_index_mapper.h"
#include "utils/allocator.h"
#include "utils/complex_utils.h"
#include "utils/utils.h"

#define MAX(a, b) ((a > b) ? a : b)

#define PRINT_SUCCESS(str) printf("\033[1;32m" str "\033[1;0m");

#define PRINT_FAILURE(str) printf("\033[1;31m" str "\033[1;0m");

#define VALIDATE_AND_GET_INT(str, buff, result, ret, min_val)                  \
{                                                                              \
    result = atoi(str);                                                        \
    sprintf(buff, "%d", result);                                               \
    ret = strcmp(str, buff);                                                   \
    ret |= (result < min_val);                                                 \
}

#define VALIDATE_AND_GET_DOUBLE(str, buff, result, ret, min_val, max_val)      \
{                                                                              \
    INT32 length;                                                              \
    DOUBLE temp;                                                               \
    result = atof(str);                                                        \
    ret = SSCANF(str, "%lf %n", &temp, &length) != 1;                          \
    ret |= strlen(str) != length;                                              \
    ret |= (result < min_val);                                                 \
    ret |= (result > max_val);                                                 \
}

/**
 * @brief handle the errors codes returned from parsing function
 *
 */
#define HANDLE_PARSER_ERROR_MESSAGE(status)                                    \
{                                                                              \
    if (status != PARSER_SUCCESS)                                              \
    {                                                                          \
        switch (status)                                                        \
        {                                                                      \
        case HELP_MENU:                                                        \
            show_help_menu();                                                  \
            break;                                                             \
        case SIZE_REQUIRED_ERROR:                                              \
            printf("\nProblem size must be provided.\n");                      \
            break;                                                             \
        case SIZE_PARSING_ERROR:                                               \
            printf("\nInvalid problem size provided.\n");                      \
            break;                                                             \
        case UNSUPPORTED_SIZE_ERROR:                                           \
            printf("\nUnsupported problem size provided.\nOnly 1D vec is "     \
                   "supported.\n");                                            \
            break;                                                             \
        case UNSUPPORTED_OPTION_ERROR:                                         \
            printf("\nUnsupported option provided.\n");                        \
            break;                                                             \
        case NON_OPTION_ARGUMENTS_ERROR:                                       \
            printf("\nMore than one non-option arguments provided.\nOnly "     \
                   "one non-option argument must be provided which is "        \
                   "problem size.\n");                                         \
            break;                                                             \
        default:                                                               \
            printf("\nInvalid arguments provided.\n");                         \
        }                                                                      \
        if (status == HELP_MENU)                                               \
        {                                                                      \
            status = PARSER_SUCCESS;                                           \
            goto exit_main;                                                    \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            printf("Use -h / --help for more information.\n");                 \
            PRINT_FAILURE("\nTest bench failed [REASON: Argument parsing "     \
                          "failed]\n\n");                                      \
        }                                                                      \
    }                                                                          \
}

/**
 * @brief handle the bench status error codes
 *
 */
#define HANDLE_BENCH_STATUS(status)                                            \
{                                                                              \
    switch (status)                                                            \
    {                                                                          \
    case BENCH_SUCCESS:                                                        \
        break;                                                                 \
    case SETUP_FAILURE:                                                        \
        PRINT_FAILURE(                                                         \
            "\nTest bench failed [REASON: Setup problem failed]\n\n");         \
        break;                                                                 \
    case EXECUTION_FAILURE:                                                    \
        PRINT_FAILURE(                                                         \
            "\nTest bench failed [REASON: Execute problem failed]\n\n");       \
        break;                                                                 \
    case VERIFICATION_FAILURE:                                                 \
        PRINT_FAILURE(                                                         \
            "\nTest bench failed [REASON: Result mismatch]\n\n");              \
        break;                                                                 \
    default:                                                                   \
        PRINT_FAILURE("\nTest bench failed [REASON: Unknown]\n\n");            \
    }                                                                          \
}

/**
 * @brief Initialize problem descriptor with the bench params
 *
 */
#define INIT_PD(params, p_desc, dt_t, dim_t)                                   \
{                                                                              \
    p_desc->dim_rank = params->dim_rank;                                       \
    p_desc->vec_rank = params->vec_rank;                                       \
    UINT32 is_align = params->aligned_alloc;                                   \
    ALLOC_UNINIT(p_desc->dims, dim_t, sizeof(dim_t) * p_desc->dim_rank,        \
                 is_align);                                                    \
    for (INT32 i = 0; i < p_desc->dim_rank; i++)                               \
    {                                                                          \
        p_desc->dims[i].n = (dt_t)params->dims[i].n;                           \
        p_desc->dims[i].in_stride = (dt_t)params->dims[i].in_stride;           \
        p_desc->dims[i].out_stride = (dt_t)params->dims[i].out_stride;         \
    }                                                                          \
    ALLOC_UNINIT(p_desc->vecs, dim_t, sizeof(dim_t) * p_desc->vec_rank,        \
                 is_align);                                                    \
    for (INT32 i = 0; i < p_desc->vec_rank; i++)                               \
    {                                                                          \
        p_desc->vecs[i].n = (dt_t)params->vecs[i].n;                           \
        p_desc->vecs[i].in_stride = (dt_t)params->vecs[i].in_stride;           \
        p_desc->vecs[i].out_stride = (dt_t)params->vecs[i].out_stride;         \
    }                                                                          \
    if (params->num_threads > 0)                                               \
    {                                                                          \
        p_desc->pthr_fft.num_threads = params->num_threads;                    \
        p_desc->pthr_fft.dynamic_load_model = 0;                               \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        p_desc->pthr_fft.num_threads = 0;                                      \
        p_desc->pthr_fft.dynamic_load_model = 1;                               \
    }                                                                          \
    p_desc->cntrl_params.opt_level = params->opt_level;                        \
    p_desc->flags = set_flag(params);                                          \
    p_desc->cntrl_params.opt_off = 0;                                          \
    if (params->opt_level == -1)                                               \
    {                                                                          \
        p_desc->cntrl_params.opt_off = 1;                                      \
    }                                                                          \
    p_desc->cntrl_params.logger_mode = params->logger_mode;                    \
    p_desc->cntrl_params.measure_stats = 0;                                    \
}

/**
 * @brief Destroy the problem descriptor
 *
 */
#define DESTROY_PD(p_desc, is_align)                                           \
{                                                                              \
    if (p_desc != NULL)                                                        \
    {                                                                          \
        FREE_ALLOCATED_MEM(p_desc->dims, is_align);                            \
        FREE_ALLOCATED_MEM(p_desc->vecs, is_align);                            \
        FREE_ALLOCATED_MEM(p_desc, is_align);                                  \
    }                                                                          \
}

/**
 * @brief Initialize dst params object and copy the values from src to dst
 *        NOTE: This function will not create in and out buffers, it needs to be
 *              created manually
 *
 */
#define ALLOC_AND_COPY_PARAMS(dst, src)                                        \
{                                                                              \
    UINT32 is_align = src->aligned_alloc;                                      \
    ALLOC_UNINIT(dst, aoclfftz_bench_params_t,                                 \
                 sizeof(aoclfftz_bench_params_t), is_align);                   \
    memcpy(dst, src, sizeof(aoclfftz_bench_params_t));                         \
    ALLOC_UNINIT(dst->dims, aoclfftz_dim_t_64_,                                \
                 sizeof(aoclfftz_dim_t_64_) * src->dim_rank, is_align);        \
    ALLOC_UNINIT(dst->vecs, aoclfftz_dim_t_64_,                                \
                 sizeof(aoclfftz_dim_t_64_) * src->vec_rank, is_align);        \
    memcpy(dst->dims, src->dims,                                               \
           sizeof(aoclfftz_dim_t_64_) * src->dim_rank);                        \
    memcpy(dst->vecs, src->vecs,                                               \
           sizeof(aoclfftz_dim_t_64_) * src->vec_rank);                        \
}

/**
 * @brief Log the bench param values in INFO mode
 *
 */
#define LOG_BENCH_PARAMS(params)                                               \
{                                                                              \
    AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode,                          \
                           "dim_rank      : %d", params->dim_rank);            \
    for (INT32 i = 0; i < params->dim_rank; i++)                               \
    {                                                                          \
        AOCLFFTZ_LOG_FORMATTED(                                                \
            INFO, params->logger_mode, "    dims[%d]   : %td:%td:%td", i,      \
            params->dims[i].n, params->dims[i].in_stride,                      \
            params->dims[i].out_stride);                                       \
    }                                                                          \
    AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode,                          \
                           "vec_rank      : %d", params->vec_rank);            \
    for (INT32 i = 0; i < params->vec_rank; i++)                               \
    {                                                                          \
        AOCLFFTZ_LOG_FORMATTED(                                                \
            INFO, params->logger_mode, "    vecs[%d]   : %td:%td:%td", i,      \
            params->vecs[i].n, params->vecs[i].in_stride,                      \
            params->vecs[i].out_stride);                                       \
    }                                                                          \
    AOCLFFTZ_LOG_FORMATTED(                                                    \
        INFO, params->logger_mode, "precision     : %s",                       \
        params->precision == FLOAT_P ? "FLOAT" : "DOUBLE");                    \
    AOCLFFTZ_LOG_FORMATTED(                                                    \
        INFO, params->logger_mode, "data_model    : %s",                       \
        params->data_model == ILP64 ? "ILP64" : "LP64");                       \
    AOCLFFTZ_LOG_FORMATTED(                                                    \
        INFO, params->logger_mode, "bench_type    : %s",                       \
        params->bench_type == ACCURACY ? "ACCURACY" : "PERFORMANCE");          \
    AOCLFFTZ_LOG_FORMATTED(                                                    \
        INFO, params->logger_mode, "res_placement : %s",                       \
        params->res_placement == IN_PLACE ? "IN_PLACE" : "OUT_OF_PLACE");      \
    AOCLFFTZ_LOG_FORMATTED(                                                    \
        INFO, params->logger_mode, "order         : %s",                       \
        params->order == OUT_OF_ORDER ? "OUT_OF_ORDER" : "IN_ORDER");          \
    AOCLFFTZ_LOG_FORMATTED(                                                    \
        INFO, params->logger_mode, "direction     : %s",                       \
        params->dir == BACKWARD ? "BACKWARD" : "FORWARD");                     \
    AOCLFFTZ_LOG_FORMATTED(                                                    \
        INFO, params->logger_mode, "fft_type      : %s",                       \
        params->fft_type == COMPLEX_TO_COMPLEX                                 \
            ? "COMPLEX_TO_COMPLEX"                                             \
            : (params->fft_type == REAL_TO_COMPLEX ? "REAL_TO_COMPLEX"         \
                                                   : "COMPLEX_TO_REAL"));      \
    AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode,                          \
                           "iterations    : %d", params->num_iterations);      \
    AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode,                          \
                           "random_seed   : %s",                               \
                           params->use_random_seed ? "TRUE" : "FALSE");        \
    AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode,                          \
                           "opt_level     : %d", params->opt_level);           \
    AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode,                          \
                           "logger_mode   : %d", params->logger_mode);         \
    AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode,                          \
                           "tolerance     : %.6g", params->tolerance);         \
    AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode,                          \
                           "num_threads   : %d", params->num_threads);         \
    AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode,                          \
                           "selector_time : %s",                               \
                           params->selector_time ? "TRUE" : "FALSE");          \
    AOCLFFTZ_LOG_FORMATTED(                                                    \
        INFO, params->logger_mode, "bit_reproducibility : %s",                 \
        params->bit_reproducibility ? "TRUE" : "FALSE");                       \
}

/**
 * @brief prepare suitable selector time unit
 *
 */
#define ADJUST_SELECTOR_TIME_UNIT(time_taken, time_unit)                       \
{                                                                              \
    /* print time in seconds */                                                \
    if (time_taken > 1E9)                                                      \
    {                                                                          \
        time_taken *= 1E-9;                                                    \
        STRCPY(time_unit, 3, "s");                                             \
    }                                                                          \
    /* print time in milli-seconds */                                          \
    else if (time_taken > 1E6)                                                 \
    {                                                                          \
        time_taken *= 1E-6;                                                    \
        STRCPY(time_unit, 3, "ms");                                            \
    }                                                                          \
    /* print time in micro-seconds */                                          \
    else if (time_taken > 1E3)                                                 \
    {                                                                          \
        time_taken *= 1E-3;                                                    \
        STRCPY(time_unit, 3, "us");                                            \
    }                                                                          \
    /* print time in nano-seconds */                                           \
    else                                                                       \
    {                                                                          \
        STRCPY(time_unit, 3, "ns");                                            \
    }                                                                          \
}

#define MINQ_TIME 1e5
#define SLEEP_TIME 1e8

// Function declarations
INT32 set_flag(aoclfftz_bench_params_t *params);
INT32 find_dim_rank(CHAR *arg);
INT32 allocate_and_fill_dims_vecs(CHAR *arg, INT32 dim_rank, INT32 vec_rank,
                                  aoclfftz_dim_t_64_ **dims,
                                  aoclfftz_dim_t_64_ **vecs,
                                  INTP default_stride);
INT32 calibrate_iterations(VOID *handle, aoclfftz_bench_params_t *params);
VOID bench_sleep(INT64 nano_seconds);

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

INT32 allocate_and_fill_dims_vecs(CHAR *arg, INT32 dim_rank, INT32 vec_rank,
                                  aoclfftz_dim_t_64_ **dims,
                                  aoclfftz_dim_t_64_ **vecs,
                                  INTP default_stride)
{
    ALLOC_ALIGN_INIT((*dims), aoclfftz_dim_t_64_,
                     dim_rank * sizeof(aoclfftz_dim_t_64_));
    ALLOC_ALIGN_INIT((*vecs), aoclfftz_dim_t_64_,
                     vec_rank * sizeof(aoclfftz_dim_t_64_));
    INT32 max_rank = dim_rank > vec_rank ? dim_rank : vec_rank;
    aoclfftz_dim_t_64_ *desc = NULL;
    ALLOC_ALIGN_INIT(desc, aoclfftz_dim_t_64_,
                     max_rank * sizeof(aoclfftz_dim_t_64_));

    INT32 is_stride = 0;
    INT32 rank_count = 0;
    INT32 vec_count = 0;
    INT32 start = 0;
    CHAR val_str[strlen(arg) + 1];
    INT32 status = PARSER_SUCCESS;
    for (INT32 i = 0; i < strlen(arg); ++i)
    {
        if (arg[i] == 'x' || arg[i] == 'X')
        {
            if (i + 1 >= strlen(arg) || !isdigit(arg[i + 1]))
            {
                printf("Integer value expected after 'x' character.\n");
                status = SIZE_PARSING_ERROR;
                goto exit_func;
            }
            if ((is_stride != 0 && is_stride != 2))
            {
                printf("Only in_stride is not accepted. "
                       "Please pass both in & out strides (or) no strides.\n");
                status = SIZE_PARSING_ERROR;
                goto exit_func;
            }
            is_stride = 0;
            rank_count++;
        }
        else if (arg[i] == 'v' || arg[i] == 'V')
        {
            if (i + 1 >= strlen(arg) || !isdigit(arg[i + 1]))
            {
                printf("Integer value expected after 'v' character.\n");
                status = SIZE_PARSING_ERROR;
                goto exit_func;
            }
            // by default the data is stored in desc always
            // once "v" is encountered, its moved to vecs and then desc is reset
            // FIXME : this needs to be fixed properly
            for (INT32 i = vec_rank - 1, j = 0; i >= 0; i--, j++)
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
            is_stride = 0;
            vec_count++;
        }
        else if (arg[i] == ':')
        {
            if (i + 1 >= strlen(arg) || !isdigit(arg[i + 1]))
            {
                printf("Integer value expected after ':' character.\n");
                status = SIZE_PARSING_ERROR;
                goto exit_func;
            }
            is_stride++;
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
            if (is_stride == 0)
            {
                if (val == 0)
                {
                    printf("Invalid dim/vec size (zero) at rank %d",
                           rank_count);
                    status = SIZE_PARSING_ERROR;
                    goto exit_func;
                }
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
            i--;
        }
        else
        {
            status = SIZE_PARSING_ERROR;
            goto exit_func;
        }
    }

    // copy desc to dims in reverse
    for (INT32 i = dim_rank - 1, j = 0; i >= 0; i--, j++)
    {
        (*dims)[i].n = desc[j].n;
        (*dims)[i].in_stride = desc[j].in_stride;
        (*dims)[i].out_stride = desc[j].out_stride;
    }

    // validate & set strides for dims if not provided
    for (INT32 i = 0; i < dim_rank; i++)
    {
        INTP min_stride =
            (i == 0) ? 1 : ((*dims)[i - 1].n * (*dims)[i - 1].in_stride);
        if ((*dims)[i].in_stride == 0)
        {
            (*dims)[i].in_stride = min_stride;
        }
        else if ((*dims)[i].in_stride < min_stride)
        {
            printf("Invalid in stride value : %td provided for (%d) dim."
                   "minimum value expected : %td\n",
                   (*dims)[i].in_stride, i + 1, min_stride);
            status = SIZE_PARSING_ERROR;
            goto exit_func;
        }
        min_stride =
            (i == 0) ? 1 : ((*dims)[i - 1].n * (*dims)[i - 1].out_stride);
        if ((*dims)[i].out_stride == 0)
        {
            (*dims)[i].out_stride = min_stride;
        }
        else if ((*dims)[i].out_stride < min_stride)
        {
            printf("Invalid out stride value : %td provided for (%d) dim."
                   "minimum value expected : %td\n",
                   (*dims)[i].out_stride, i + 1, min_stride);
            status = SIZE_PARSING_ERROR;
            goto exit_func;
        }
    }

    // validate & set strides for vecs if not provided
    for (INT32 i = 0; i < vec_rank; i++)
    {
        INTP min_stride = (i == 0) ?
                (*dims)[dim_rank -1].n * (*dims)[dim_rank - 1].in_stride :
                    ((*vecs) [i - 1]. n * (*vecs) [i - 1].in_stride);
        if ((*vecs)[i].in_stride == 0)
        {
            // stride of fcd should atleast be the length of dims
            (*vecs)[i].in_stride = min_stride;
        }
        else if ((*vecs)[i].in_stride < min_stride)
        {
            printf("Invalid in stride value : %td provided for (%d) vec."
                   "minimum value expected : %td\n",
                   (*vecs)[i].in_stride, i + 1, min_stride);
            status = SIZE_PARSING_ERROR;
            goto exit_func;
        }
        min_stride = (i == 0) ? (*dims)[dim_rank - 1].n *
                                    (*dims)[dim_rank - 1].out_stride
                              : ((*vecs)[i - 1].n * (*vecs)[i - 1].out_stride);
        if ((*vecs)[i].out_stride == 0)
        {
            // stride of fcd should atleast be the length of dims
            (*vecs)[i].out_stride = min_stride;
        }
        else if ((*vecs)[i].out_stride < min_stride)
        {
            printf("Invalid out stride value : %td provided for (%d) vec."
                   "minimum value expected : %td\n",
                   (*vecs)[i].out_stride, i + 1, min_stride);
            status = SIZE_PARSING_ERROR;
            goto exit_func;
        }
    }

    // Initialize vector size to default value when no vector is encountered.
    if (vec_count == 0)
    {
        (*vecs)[0].n = 1;
        (*vecs)[0].in_stride = default_stride;
        (*vecs)[0].out_stride = default_stride;
    }

    CHECK_SUPPORTED_DIMS(dims, vecs, dim_rank, vec_rank, status);

exit_func:
    FREE_ALIGN_ALLOCATED_MEM(desc);
    return status;
}

/**
 * @brief Computes the number of iterations for benchmarking
 *
 * @param handle handle object of VOID* type
 * @param params aoclfftz_bench_params_t type contains parsed arguments
 * @return INT32 iterations
 */
INT32 calibrate_iterations(VOID *handle, aoclfftz_bench_params_t *params)
{
    DOUBLE minq_time = MINQ_TIME; // minimum quantifiable time 100 us
    INT32 increase_iterations = 1;
    DOUBLE cur_time = 0;
#ifdef WIN32
    timer clk_tick;
#endif
    timeVal start_time, end_time;
    initTimer(clk_tick);
    INT32 iters = 1;

    for (iters = 1; increase_iterations && iters < INT32_MAX; iters *= 5)
    {
        INT32 j = iters + 1;
        getTime(start_time);
        while (--j)
        {
            params->aoclfftz_execute(handle);
        }
        getTime(end_time);
        cur_time = (DOUBLE)diffTime(clk_tick, start_time, end_time);
        // if execution time is above the minimum quantifiable limit,
        // then stop the iterations
        if (cur_time >= minq_time)
        {
            increase_iterations = 0;
            if (cur_time > params->min_bench_time)
            {
                return iters;
            }
            // Scaling the iteration for min_acceptable_time
            return (iters * params->min_bench_time / cur_time);
        }
        bench_sleep(SLEEP_TIME); // 0.1 seconds
    }
    return iters;
}

/**
 * @brief Wrapper function for nanosleep
 *
 * @param nano_seconds sleep time in nano seconds
 * @return VOID
 */
VOID bench_sleep(INT64 nano_seconds)
{
#ifdef WIN32
    DWORD milli_seconds = (nano_seconds / 1e6);
    Sleep(milli_seconds);
#else
    timeVal t;
    t.tv_sec = nano_seconds / (INT64)1e9;  // 1 second
    t.tv_nsec = nano_seconds % (INT64)1e9; // 1 second
    nanosleep(&t, &t);
#endif
}

#endif // AOCLFFTZ_COREBENCH_UTILS_H
