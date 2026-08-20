// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file performance.c
 *
 *  @brief Performance test utility functions.
 *
 *  This file contains performance test related utility functions
 *  for test bench.
 *
 *  @author V. Murugan
 *  @author Srirammaswamy Srinivasan
 *  @author Jeya R
 */

#include <limits.h>
#include <time.h>
#include "test/bench/performance.h"
#include "test/bench/utils/register_functions.h"
#include "test/bench/utils/bench_utils.h"
#include "test/bench/utils/size_and_index_mapper.h"

// warmup iterations are set to 2 since each iteration will internally
// run calibrated no. of iterations which will be sufficient for different
// sized problems.
#define WARMUP_ITERATIONS 2

/**
 * @brief Run and benchmark the FFT problem
 *
 * @param params aoclfftz_bench_params_t type contains parsed arguments
 * @param handle handle object of FFTZ_VOID* type
 * @param stats performance statistics object
 * @return FFTZ_INT32 status code
 */
FFTZ_INT32 run_problem_on_performance_mode(aoclfftz_bench_params_t *params,
                                      FFTZ_VOID *handle, perf_stats_t *stats)
{
    AOCLFFTZ_LOG(TRACE, params->logger_mode, "ENTER");
    FFTZ_INT32 status;
#ifdef WIN32
    timer clk_tick;
#endif
    timeVal start_time, end_time;
    FFTZ_DOUBLE tot_time = 0.0, cur_time = 0.0;
    FFTZ_INTP n = calculate_size(params->dims, params->dim_rank);
    FFTZ_INTP batches = calculate_size(params->vecs, params->vec_rank);
    FFTZ_UINTP input_size = 0;
    FFTZ_UINTP output_size = 0;
    calculate_buffer_sizes(params->dim_rank, params->vec_rank, params->dims,
                           params->vecs, &input_size, &output_size, 
                           params->fft_type);

    // prepare random seed value
    if (params->use_random_seed)
    {
        params->seed = (FFTZ_INT32)((FFTZ_INT64)time(0) % (INT_MAX));
    }

    AOCLFFTZ_LOG(INFO, params->logger_mode, "seed   : %d",
                          params->seed);


    // prepare random input data
    params->prepare_input_data(params->in, input_size, NULL, RANDOM_INPUT,
                               params->sz_info.in_data_stride);

    status = aoclfftz_execute(handle);
    if (status != AOCLFFTZ_SUCCESS)
    {
        PRINT_FAILURE(
            "\nTest bench failed [REASON: aoclfftz_execute failed]\n\n");
        return EXECUTION_FAILURE;
    }

    FFTZ_INT32 iter = calibrate_iterations(handle, params->min_bench_time,
                                           params);

    // warmup iterations (skipped from profiling)
    // TODO: improvise this logic
    AOCLFFTZ_LOG(TRACE, params->logger_mode, "WARM-UP START");
    for (FFTZ_INT32 i = 0; i < WARMUP_ITERATIONS; ++i)
    {
        FFTZ_INT32 j = iter + 1;
        while (--j)
        {
            aoclfftz_execute(handle);
        }
    }
    AOCLFFTZ_LOG(TRACE, params->logger_mode, "WARM-UP END");

    initTimer(clk_tick);
    for (FFTZ_INT32 i = 0; i < params->num_iterations; i++)
    {
        AOCLFFTZ_LOG(INFO, params->logger_mode, "Iteration: %d",
                               i + 1);

        FFTZ_INT32 j = iter + 1;
        getTime(start_time);
        while (--j)
        {
                aoclfftz_execute(handle);
        }
        getTime(end_time);
        cur_time = diffTime(clk_tick, start_time, end_time);
        cur_time = cur_time / iter;
        tot_time = tot_time + cur_time;
        if (cur_time < stats->min_time)
        {
            stats->min_time = cur_time;
        }
        stats->avg_time = (FFTZ_DOUBLE)tot_time / params->num_iterations;
        bench_sleep(1e8); // 0.1 seconds
    }

    // compute MFLOPS from execution time
    if (params->fft_type == C2C)
    {
        stats->max_mflops = (5.0 * n * batches * log2(n)) /
                             (stats->min_time * 1E-3);
        stats->avg_mflops = (5.0 * n * batches * log2(n)) /
                             (stats->avg_time * 1E-3);
    }
    else
    {
        stats->max_mflops = (2.5 * n * batches * log2(n)) /
                             (stats->min_time * 1E-3);
        stats->avg_mflops = (2.5 * n * batches * log2(n)) /
                             (stats->avg_time * 1E-3);
    }

    AOCLFFTZ_LOG(TRACE, params->logger_mode, "EXIT");
    return BENCH_SUCCESS;
}

/**
 * @brief prints performance stats
 */
FFTZ_VOID print_perf_stats(perf_stats_t *stats)
{
    // prepare suitable execution time unit
    FFTZ_DOUBLE time_multiplier = 1.0;
    FFTZ_CHAR time_unit[3];
    // units will be decided based on minimum of min_time and avg_time
    // which is min_time
    // print time in seconds
    if (stats->min_time > 1E9)
    {
        time_multiplier = 1E-9;
        STRCPY(time_unit, 3, "s");
    }
    // print time in milli-seconds
    else if (stats->min_time > 1E6)
    {
        time_multiplier = 1E-6;
        STRCPY(time_unit, 3, "ms");
    }
    // print time in micro-seconds
    else if (stats->min_time > 1E3)
    {
        time_multiplier = 1E-3;
        STRCPY(time_unit, 3, "us");
    }
    // print time in nano-seconds
    else
    {
        time_multiplier = 1.0;
        STRCPY(time_unit, 3, "ns");
    }

    printf("\n=====================================\n");
    printf("  Min Execution time : %6.3lf %s\n",
           stats->min_time * time_multiplier, time_unit);
    printf("  Avg Execution time : %6.3lf %s\n",
           stats->avg_time * time_multiplier, time_unit);
    printf("=====================================\n");
    printf("      Max MFLOPS : %9.6lf\n", stats->max_mflops);
    printf("      Avg MFLOPS : %9.6lf\n", stats->avg_mflops);
    printf("=====================================\n");
}

FFTZ_VOID calculate_and_print_scaling(perf_stats_t st, perf_stats_t mt)
{
    printf("\nPerformance numbers in Multi threaded mode\n");
    print_perf_stats(&mt);
    printf("\nPerformance numbers in Single threaded mode\n");
    print_perf_stats(&st);
    FFTZ_DOUBLE scaling_factor = st.avg_time / mt.avg_time;
    if (scaling_factor < 1)
    {
        PRINT_FAILURE_FORMATTED("Scaling Factor (Single->Multi): %.2fx\n",
                                 scaling_factor);
    }
    else
    {
        PRINT_SUCCESS_FORMATTED("Scaling Factor (Single->Multi): %.2fx\n",
                                 scaling_factor);
    }
}

/**
 * @brief run the test bench on performance mode and calculate MFLOPS.
 *
 * @param params bench params object
 * @param stats performance statistic object
 * @return FFTZ_INT32 bench status code
 */
FFTZ_INT32 run_bench_on_perf_mode_and_get_stats(aoclfftz_bench_params_t *params,
                                           perf_stats_t *stats)
{
    FFTZ_INT32 status = BENCH_SUCCESS;

    FFTZ_VOID *handle = params->setup_problem(params);
    if (handle == NULL)
    {
        PRINT_FAILURE("\nTest bench failed [REASON: Setup problem failed]\n\n");
        status = SETUP_FAILURE;
        goto exit;
    }

    status = run_problem_on_performance_mode(params, handle, stats);
    if (status != BENCH_SUCCESS)
    {
        PRINT_FAILURE(
            "\nTest bench failed [REASON: Execute problem failed]\n\n");
        status = EXECUTION_FAILURE;
        goto exit;
    }

exit:
    aoclfftz_destroy(handle);
    return status;
}

/**
 * @brief run the test bench on performance mode and calculate MFLOPS.
 *
 * @param params bench params object
 * @return FFTZ_INT32 bench status code
 */
FFTZ_INT32 run_bench_on_performance_mode(aoclfftz_bench_params_t *params)
{
    FFTZ_INT32 status = BENCH_SUCCESS;
    perf_stats_t stats =
    {
        .min_time = DBL_MAX,
        .avg_time = 0.0,
        .avg_mflops = 0.0,
        .max_mflops = 0.0
    };

    status = run_bench_on_perf_mode_and_get_stats(params, &stats);
    if (status != BENCH_SUCCESS)
    {
        goto exit;
    }

    // Multi-threaded scaling comparisons against single-threaded mode are
    // performed only when logging is set to INFO level or higher.
    if (params->logger_mode >= AOCLFFTZ_LOG_INFO && params->num_threads > 1)
    {
        // Running in single threaded mode
        params->num_threads = 1;
        perf_stats_t st_stats =
        {
            .min_time = DBL_MAX,
            .avg_time = 0.0,
            .avg_mflops = 0.0,
            .max_mflops = 0.0
        };

        status = run_bench_on_perf_mode_and_get_stats(params, &st_stats);
        if (status != BENCH_SUCCESS)
        {
            goto exit;
        }

        calculate_and_print_scaling(st_stats, stats);
    }
    else
    {
        // print stat and exit
        print_perf_stats(&stats);
    }

    PRINT_SUCCESS("\nTest bench completed on performance mode\n\n");
exit:
    return status;
}

/**
 * @brief Computes the number of iterations for benchmarking
 *
 * @param handle handle object of FFTZ_VOID* type
 * @param min_bench_time minimum time to run benchmark
 * @param params aoclfftz_bench_params_t type contains parsed arguments
 * @return FFTZ_INT32 iterations
 */
FFTZ_INT32 calibrate_iterations(FFTZ_VOID *handle, FFTZ_DOUBLE min_bench_time,
                            aoclfftz_bench_params_t *params)
{
    FFTZ_DOUBLE minq_time = 1e5; // minimum quantifiable time 100 us
    FFTZ_INT32 increase_iterations = 1;
    FFTZ_DOUBLE cur_time = 0;
#ifdef WIN32
    timer clk_tick;
#endif
    timeVal start_time, end_time;
    initTimer(clk_tick);
    FFTZ_INT32 iters = 1;

    for (iters = 1; increase_iterations && iters < INT32_MAX; iters *= 5)
    {
        FFTZ_INT32 j = iters + 1;
        getTime(start_time);
        while (--j)
        {
            aoclfftz_execute(handle);
        }
        getTime(end_time);
        cur_time = (FFTZ_DOUBLE)diffTime(clk_tick, start_time, end_time);
        // if execution time is above the minimum quantifiable limit,
        // then stop the iterations
        if (cur_time >= minq_time)
        {
            increase_iterations = 0;
            if (cur_time > min_bench_time)
            {
                return iters;
            }
            //Scaling the iteration for min_acceptable_time
            return (iters * min_bench_time / cur_time);
        }
        bench_sleep(1e8); // 0.1 seconds
    }
    return iters;
}

/**
 * @brief Wrapper function for nanosleep
 *
 * @param nano_seconds sleep time in nano seconds
 * @return FFTZ_VOID
 */
FFTZ_VOID bench_sleep(FFTZ_INT64 nano_seconds)
{
#ifdef WIN32
    DWORD milli_seconds = (nano_seconds / 1e6);
    Sleep(milli_seconds);
#else
    timeVal t;
    t.tv_sec = nano_seconds / (FFTZ_INT64)1e9; // 1 second
    t.tv_nsec = nano_seconds % (FFTZ_INT64)1e9; // 1 second
    nanosleep(&t, &t);
#endif
}
