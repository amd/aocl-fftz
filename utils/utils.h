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

/** @file utils.h
 *
 *  @brief Utility functions related to logger, timer, cpuid and others.
 *
 *  This file contains the utility functions related to logging mechanism,
 *  timer and others along with the associated data structures and macros.
 *
 *  @author S. Biplab Raut
 */

#ifndef AOCLFFTZ_UTILS_H
#define AOCLFFTZ_UTILS_H

#include <stdio.h>
#include "api/types.h"
#include "api/aoclfftz.h"

extern INT32 global_logger_mode;

#define AOCLFFTZ_STATS

#define AOCLFFTZ_CPUID_SIMD_DETECTION

#if defined(__GNUC__) && __GNUC__ >= 3
#define FUNC_NAME __func__
#else
#define FUNC_NAME __FUNCTION__
#endif

#define INFO     1
#define TRACE    2
#define DEBUG    3

#define SET_PROBLEM_LOGGER_MODE(problem)                                       \
{                                                                              \
    if (problem != NULL)                                                       \
    {                                                                          \
        global_logger_mode = problem->cntrl_params.logger_mode;                \
    }                                                                          \
}

#ifdef AOCLFFTZ_STATS
#ifdef _WINDOWS
#include <windows.h>
typedef LARGE_INTEGER timer;
typedef LARGE_INTEGER timeVal;
#else
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
typedef struct timespec timer;
typedef struct timespec timeVal;
#endif
#endif

#ifdef AOCL_ENABLE_LOG

#define AOCLFFTZ_LOG__INTERNAL(str, ...)                                       \
    printf("[%s] : %s : %s : %d : " str "%s", type, __FILE__, FUNC_NAME,       \
           __LINE__, __VA_ARGS__)

#define AOCLFFTZ_LOG(logType, enableLog, ...)                                  \
    do                                                                         \
    {                                                                          \
        if (enableLog)                                                         \
        {                                                                      \
            const CHAR *type = NULL;                                           \
            if (logType == INFO)                                               \
            {                                                                  \
                type = "INFO";                                                 \
            }                                                                  \
            else if (logType == TRACE)                                         \
            {                                                                  \
                type = "TRACE";                                                \
            }                                                                  \
            else if (logType == DEBUG)                                         \
            {                                                                  \
                type = "DEBUG";                                                \
            }                                                                  \
            else                                                               \
            {                                                                  \
                type = "UNKNOWN";                                              \
            }                                                                  \
            if (logType <= enableLog)                                          \
            {                                                                  \
                AOCLFFTZ_LOG__INTERNAL(__VA_ARGS__, "\n");                     \
            }                                                                  \
        }                                                                      \
    } while (0)
#else
#define AOCLFFTZ_LOG(logType, enableLog, ...) ((VOID)enableLog);
#endif // AOCL_ENABLE_LOG

#define AOCLFFTZ_ERROR__INTERNAL(str, ...)                                     \
    fprintf(stderr, "[ERROR] : %s : %s : %d : " str "%s", __FILE__,            \
            FUNC_NAME, __LINE__, __VA_ARGS__)

#define AOCLFFTZ_ERROR(...) AOCLFFTZ_ERROR__INTERNAL(__VA_ARGS__, "\n");

// Timer and stats keeping
#ifdef AOCLFFTZ_STATS
#ifdef _WINDOWS
#define initTimer(timerClk)                                                    \
{                                                                              \
    if (!QueryPerformanceFrequency(&timerClk))                                 \
    {                                                                          \
        AOCLFFTZ_ERROR("QueryPerformanceFrequency based Timer failed.");       \
    }                                                                          \
}
#define getTime(timeVal) QueryPerformanceCounter(&timeVal)
#define diffTime(timerClk, startTime, endTime)                                 \
    ((1000000000ULL * (endTime.QuadPart - startTime.QuadPart)) /               \
     timerClk.QuadPart)
#else
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#define initTimer(timer)
#define getTime(timeVal) clock_gettime(CLOCK_REALTIME, &timeVal)
#define diffTime(timer, startTime, endTime)                                    \
    (1000000000ULL * (endTime.tv_sec - startTime.tv_sec) +                     \
     endTime.tv_nsec - startTime.tv_nsec)
#endif
#endif

const CHAR* get_status_string(aoclfftz_error_type status);

INT32 setup_dynamic_dispatcher(INT32 opt_off, INT32 opt_level);

#ifdef ENABLE_APP_INFO_LOGS
#define PRINT_PROBLEM_DESCRIPTOR(problem, dt_type, f_specifier, data_model)    \
{                                                                              \
    printf("\n[AOCL-FFTZ] Problem : ");                                        \
    const CHAR *data_type = (dt_type == DT_FLOAT) ? "Float" :                  \
                            (dt_type == DT_DOUBLE) ? "Double" : "Unknown";     \
    const CHAR *fft_type = (problem->flags.fft_type == 0) ? "Complex" : "Real";\
    const CHAR *direction = (problem->flags.fft_direction == FORWARD_FFT_DIR) ?\
                            "Forward" : "Backward";                            \
    const CHAR *placement = (problem->flags.fft_placement == 0) ?              \
                            "InPlace" : "OutOfPlace";                          \
    const CHAR *storage_order = (problem->flags.storage_order == 0) ?          \
                                "InOrder" : "OutOfOrder";                      \
    INT32 threads = problem->pthr_fft.num_threads;                             \
    for (INT32 i = problem->vec_rank - 1; i >= 0; i--)                         \
    {                                                                          \
        printf(f_specifier ":" f_specifier ":" f_specifier, problem->vecs[i].n,\
               problem->vecs[i].in_stride, problem->vecs[i].out_stride);       \
        if (i > 0)                                                             \
        {                                                                      \
            printf("x"); /* dims delimiter*/                                   \
        }                                                                      \
    }                                                                          \
    printf("v"); /* vecs delimiter*/                                           \
    for (INT32 i = problem->dim_rank - 1; i >= 0; i--)                         \
    {                                                                          \
        printf(f_specifier ":" f_specifier ":" f_specifier, problem->dims[i].n,\
               problem->dims[i].in_stride, problem->dims[i].out_stride);       \
        if (i > 0)                                                             \
        {                                                                      \
            printf("x"); /* dims delimiter*/                                   \
        }                                                                      \
    }                                                                          \
    printf(", "); /* delimiter to adapt csv format */                          \
    printf("%s, %s, %s, %s, %s, %s, Threads: %d\n", data_type, fft_type,       \
           direction, placement, storage_order, data_model, threads);          \
}

#define PRINT_LP64_PROBLEM_DESCRIPTOR(problem, data_type)\
    PRINT_PROBLEM_DESCRIPTOR(problem, data_type, "%d", "LP64")
#define PRINT_ILP64_PROBLEM_DESCRIPTOR(problem, data_type)\
    PRINT_PROBLEM_DESCRIPTOR(problem, data_type, "%ld", "ILP64")

#else

#define PRINT_LP64_PROBLEM_DESCRIPTOR(problem, data_type) ((VOID)0);
#define PRINT_ILP64_PROBLEM_DESCRIPTOR(problem, data_type) ((VOID)0);

#endif

#endif // AOCLFFTZ_UTILS_H
