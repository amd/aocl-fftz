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

#include "api/types.h"

#define AOCLFFTZ_DTL

#define AOCLFFTZ_STATS

#define AOCLFFTZ_CPUID_SIMD_DETECTION

#if defined(__GNUC__) && __GNUC__ >= 3
#define FUNC_NAME __func__
#else
#define FUNC_NAME __FUNCTION__
#endif

#ifdef AOCLFFTZ_DTL
#define ERR      1
#define INFO     2
#define DEBUG    3
#define TRACE    4
#else
#define ERR      0
#define INFO     0
#define DEBUG    0
#define TRACE    0

#endif

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

//Logger - DTL
#ifdef AOCLFFTZ_DTL
#include <stdio.h>
#define AOCLFFTZ_LOG_UNFORMATTED(logType, enableLog, str)     do {\
                            if (enableLog)\
                            {\
                                const char *type=NULL;\
                                if (logType == ERR)\
                                    type = "ERR";\
                                else if (logType == INFO)\
                                    type = "INFO";\
                                else if (logType == DEBUG)\
                                    type = "DEBUG";\
                                else if (logType == TRACE)\
                                    type = "TRACE";\
                                if (logType <= enableLog)\
                                    printf ("[%s] : %s : %s : %d : "str"\n",\
                                    type, __FILE__, FUNC_NAME, __LINE__);\
                            }\
                        } while (0)
#define AOCLFFTZ_LOG_FORMATTED(logType, enableLog, str, ...)     do {\
                            if (enableLog)\
                            {\
                                const char *type=NULL;\
                                if (logType == ERR)\
                                    type = "ERR";\
                                else if (logType == INFO)\
                                    type = "INFO";\
                                else if (logType == DEBUG)\
                                    type = "DEBUG";\
                                else if (logType == TRACE)\
                                    type = "TRACE";\
                                if (logType <= enableLog)\
                                    printf ("[%s] : %s : %s : %d : "str"\n",\
                                    type, __FILE__, FUNC_NAME, __LINE__, __VA_ARGS__);\
                            }\
                        } while (0)
#else
#define AOCLFFTZ_LOG_UNFORMATTED(logType, enableLog, str)
#define AOCLFFTZ_LOG_FORMATTED(logType, enableLog, str, ...)
#endif

//Timer and stats keeping
#ifdef AOCLFFTZ_STATS
#ifdef _WINDOWS
#define initTimer(timerClk) if(!QueryPerformanceFrequency(&timerClk))\
                         {\
                             AOCLFFTZ_LOG_UNFORMATTED(ERR, 1, \
                             "QueryPerformanceFrequency based Timer failed.");\
                         }
#define getTime(timeVal) QueryPerformanceCounter(&timeVal)
#define diffTime(timerClk, startTime, endTime) ((1000000000ULL * \
                (endTime.QuadPart - startTime.QuadPart))/timerClk.QuadPart)
#else
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#define initTimer(timer)
#define getTime(timeVal) clock_gettime(CLOCK_REALTIME, &timeVal)
#define diffTime(timer, startTime, endTime) (1000000000ULL * \
                (endTime.tv_sec - startTime.tv_sec) + \
                endTime.tv_nsec - startTime.tv_nsec)
#endif
#endif

VOID cpu_features_detection(INTP fn, INTP optVal,
                            INTP *eax, INTP *ebx,
                            INTP *ecx, INTP *edx);

INT32 setup_dynamic_dispatcher(INT32 opt_off, INT32 opt_level, INT32 logger_mode);

#endif //AOCLFFTZ_UTILS_H