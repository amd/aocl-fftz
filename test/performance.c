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

#include <time.h>
#include "test/performance.h"
#include "test/utils/register_functions.h"
#include "test/utils/bench_utils.h"
#include "test/utils/size_and_index_mapper.h"

// warmup iterations are set to 2 since each iteration will internally
// run calibrated no. of iterations which will be sufficient for different
// sized problems.
#define WARMUP_ITERATIONS 2

/**
 * @brief Run and benchmark the FFT problem
 *
 * @param params aoclfftz_bench_params_t type contains parsed arguments
 * @param handle handle object of VOID* type
 * @return INT32 status code
 */
INT32 run_problem_on_performance_mode(aoclfftz_bench_params_t *params,
                                      VOID *handle)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "ENTER");
#endif
    INT32 status;
#ifdef WIN32
    timer clk_tick;
#endif
    timeVal start_time, end_time;
    DOUBLE min_time = DBL_MAX, avg_time = 0.0, tot_time = 0.0, cur_time = 0.0;
    DOUBLE avg_mflops = 0.0, max_mflops = 0.0;
    INTP n = calculate_size(params->dims, params->dim_rank);
    INTP batches = calculate_size(params->vecs, params->vec_rank);
    INTP input_size = 0;
    INTP output_size = 0;
    calculate_buffer_sizes(params, &input_size, &output_size);

    // prepare random seed value
    if (params->use_random_seed)
    {
        params->seed = time(0);
    }

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode, "seed   : %d",
                          params->seed);
#endif

    // prepare random input data
    params->prepare_input_data(params->in, input_size, NULL, RANDOM_INPUT);

    status = params->aoclfftz_execute(handle);
    if (status != AOCLFFTZ_SUCCESS)
    {
        PRINT_FAILURE(
            "\nTest bench failed [REASON: aoclfftz_execute  failed]\n\n");
        return EXECUTION_FAILURE;
    }

    INT32 iter = calibrate_iterations(handle, params->min_bench_time, params);

    // warmup iterations (skipped from profiling)
    // TODO: improvise this logic
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "WARM-UP START");
#endif
    for (INT32 i = 0; i < WARMUP_ITERATIONS; ++i)
    {
        INT32 j = iter + 1;
        while (--j)
        {
            params->aoclfftz_execute(handle);
        }
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "WARM-UP END");
#endif

    initTimer(clk_tick);
    for (INT32 i = 0; i < params->num_iterations; i++)
    {
#ifdef AOCL_ENABLE_LOG
        AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode, "Iteration: %d",
                               i + 1);
#endif
        INT32 j = iter + 1;
        getTime(start_time);
        while (--j)
        {
            params->aoclfftz_execute(handle);
        }
        getTime(end_time);
        cur_time = diffTime(clk_tick, start_time, end_time);
        cur_time = cur_time / iter;
        tot_time = tot_time + cur_time;
        if (cur_time < min_time)
        {
            min_time = cur_time;
        }
        avg_time = (DOUBLE)tot_time / params->num_iterations;
        bench_sleep(1e8); // 0.1 seconds
    }

    // compute MFLOPS from execution time
    max_mflops = (5.0 * n * batches * log2(n)) / (min_time * 1E-3);
    avg_mflops = (5.0 * n * batches * log2(n)) / (avg_time * 1E-3);
    print_perf_stats(min_time, avg_time, avg_mflops, max_mflops);

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "EXIT");
#endif
    return BENCH_SUCCESS;
}

/**
 * @brief prints performance stats
 */
VOID print_perf_stats(DOUBLE min_time, DOUBLE avg_time, DOUBLE avg_mflops,
                      DOUBLE max_mflops)
{
    // prepare suitable execution time unit
    DOUBLE time_multiplier = 1.0;
    CHAR time_unit[3];
    // units will be decided based on minimum of min_time and avg_time
    // which is min_time
    // print time in seconds
    if (min_time > 1E9)
    {
        time_multiplier = 1E-9;
        STRCPY(time_unit, 3, "s");
    }
    // print time in milli-seconds
    else if (min_time > 1E6)
    {
        time_multiplier = 1E-6;
        STRCPY(time_unit, 3, "ms");
    }
    // print time in micro-seconds
    else if (min_time > 1E3)
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
    printf("  Min Execution time : %6.3lf %s\n", min_time * time_multiplier,
           time_unit);
    printf("  Avg Execution time : %6.3lf %s\n", avg_time * time_multiplier,
           time_unit);
    printf("=====================================\n");
    printf("      Max MFLOPS : %9.6lf\n", max_mflops);
    printf("      Avg MFLOPS : %9.6lf\n", avg_mflops);
    printf("=====================================\n");
}

/**
 * @brief run the test bench on performance mode and calculate MFLOPS.
 *
 * @param params bench params object
 * @return INT32 bench status code
 */
INT32 run_bench_on_performance_mode(aoclfftz_bench_params_t *params)
{
    INT32 status = BENCH_SUCCESS;

    // setup the FFT problem
    VOID *handle = params->setup_problem(params);
    if (handle == NULL)
    {
        PRINT_FAILURE("\nTest bench failed [REASON: Setup problem failed]\n\n");
        status = SETUP_FAILURE;
        goto exit_performance_mode;
    }

    // run the FFT problem
    status = run_problem_on_performance_mode(params, handle);
    if (status != BENCH_SUCCESS)
    {
        PRINT_FAILURE(
            "\nTest bench failed [REASON: Execute problem failed]\n\n");
        status = EXECUTION_FAILURE;
        goto exit_performance_mode;
    }

    PRINT_SUCCESS("\nTest bench completed on performance mode\n\n");

exit_performance_mode:
    // destroy the handle object
    params->aoclfftz_destroy(handle);
    return BENCH_SUCCESS;
}

/**
 * @brief Computes the number of iterations for benchmarking
 *
 * @param handle handle object of VOID* type
 * @param min_bench_time minimum time to run benchmark
 * @param params aoclfftz_bench_params_t type contains parsed arguments
 * @return INT32 iterations
 */
INT32 calibrate_iterations(VOID *handle, DOUBLE min_bench_time,
                            aoclfftz_bench_params_t *params)
{
    DOUBLE minq_time = 1e5; // minimum quantifiable time 100 us
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
 * @return VOID
 */
VOID bench_sleep(INT64 nano_seconds)
{
#ifdef WIN32
    DWORD milli_seconds = (nano_seconds / 1e6);
    Sleep(milli_seconds);
#else
    timeVal t;
    t.tv_sec = nano_seconds / (INT64)1e9; // 1 second
    t.tv_nsec = nano_seconds % (INT64)1e9; // 1 second
    nanosleep(&t, &t);
#endif
}
