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

/** @file bench_utils.h
 *
 *  @brief Test bench utility functions.
 *
 *  This file contains utility functions and macros related to test bench
 *  params and common utilities.
 *
 *  @author V. Murugan
 *  @author Srirammaswamy Srinivasan
 *  @author Jeya R
 */

#ifndef BENCH_UTILS_H
#define BENCH_UTILS_H

#include <stdio.h>
#include "api/types.h"

// Defining DATA_STRIDE in corebench internally to avoid using internal headers
#define T_DATA_STRIDE 2
#define PATH_SIZE_MAX 200

#define MAX(a, b) ((a > b) ? a : b)

#define PRINT_SUCCESS(str) printf("\033[1;32m" str "\033[1;0m");

#define PRINT_FAILURE(str) printf("\033[1;31m" str "\033[1;0m");

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
            case UNSUPPORTED_OPTION_ERROR:                                     \
                printf("\nUnsupported option provided.\n");                    \
                break;                                                         \
            case NON_OPTION_ARGUMENTS_ERROR:                                   \
                printf("\nMore than one non-option arguments provided.\nOnly " \
                       "one non-option argument must be provided which is "    \
                       "problem size.\n");                                     \
                break;                                                         \
            case MEMORY_FAILURE:                                               \
                printf("\nCould not allocate memory for data buffers.\n");     \
                break;                                                         \
            default:                                                           \
                printf("\nInvalid arguments provided.\n");                     \
            }                                                                  \
            if (status == HELP_MENU)                                           \
            {                                                                  \
                status = PARSER_SUCCESS;                                       \
                goto exit_main;                                                \
            }                                                                  \
            else                                                               \
            {                                                                  \
                if (status > 0)                                                \
                {                                                              \
                    printf("Use -h / --help for more information.\n");         \
                    PRINT_FAILURE("\nTest bench failed [REASON: Argument       \
                                   parsing failed]\n\n");                      \
                }                                                              \
                else                                                           \
                {                                                              \
                    PRINT_FAILURE("\nTest bench failed\n\n");                  \
                }                                                              \
            }                                                                  \
        }                                                                      \
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
        AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode,                      \
                               "random_seed   : %s",                           \
                               params->use_random_seed ? "TRUE" : "FALSE");    \
        AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode,                      \
                               "opt_level     : %d", params->opt_level);       \
        AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode,                      \
                               "logger_mode   : %d", params->logger_mode);     \
        AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode,                      \
                               "tolerance     : %.6g", params->tolerance);     \
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

#endif // BENCH_UTILS_H
