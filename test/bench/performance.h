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
    DOUBLE min_time;
    DOUBLE avg_time;
    DOUBLE avg_mflops;
    DOUBLE max_mflops;
} perf_stats_t;


INT32 run_bench_on_performance_mode(aoclfftz_bench_params_t *params);
INT32 run_problem_on_performance_mode(aoclfftz_bench_params_t *params,
                                      VOID *handle, perf_stats_t *stats);
INT32 calibrate_iterations(VOID *handle, DOUBLE min_bench_time,
                            aoclfftz_bench_params_t *params);
VOID bench_sleep(INT64 nano_seconds);
VOID print_perf_stats(perf_stats_t *stats);
VOID calculate_and_print_scaling(perf_stats_t single, perf_stats_t multi);

#endif // PERFORMANCE_H
