// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file performance.h
 *
 *  @brief Performance test utility functions.
 *
 *  This file contains function declarations of performance test related
 *  utility functions for test bench.
 *
 *  @author V. Murugan
 *  @author Srirammaswamy Srinivasan
 *  @author Jeya R
 */

#ifndef PERFORMANCE_H
#define PERFORMANCE_H

#include <float.h>
#include "test/bench/aoclfftz_bench.h"

// Structures for test bench
typedef struct perf_stats
{
    FFTZ_DOUBLE min_time;
    FFTZ_DOUBLE avg_time;
    FFTZ_DOUBLE avg_mflops;
    FFTZ_DOUBLE max_mflops;
} perf_stats_t;


FFTZ_INT32 run_bench_on_performance_mode(aoclfftz_bench_params_t *params);
FFTZ_INT32 run_problem_on_performance_mode(aoclfftz_bench_params_t *params,
                                      FFTZ_VOID *handle, perf_stats_t *stats);
FFTZ_INT32 calibrate_iterations(FFTZ_VOID *handle, FFTZ_DOUBLE min_bench_time,
                            aoclfftz_bench_params_t *params);
FFTZ_VOID bench_sleep(FFTZ_INT64 nano_seconds);
FFTZ_VOID print_perf_stats(perf_stats_t *stats);
FFTZ_VOID calculate_and_print_scaling(perf_stats_t single, perf_stats_t multi);

#endif // PERFORMANCE_H
