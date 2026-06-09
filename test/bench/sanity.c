// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file sanity.c
 *
 *  @brief functions for sanity mode of test bench.
 *
 *  This file contains function implementation of sanity mode related
 *  functions for test bench.
 *
 *  @author Avinash Thakur
 */

#include "sanity.h"
#include "test/bench/utils/bench_utils.h"
#include "utils/utils.h"


INT32 run_bench_on_sanity_mode(aoclfftz_bench_params_t *params)
{
    AOCLFFTZ_LOG(TRACE, params->logger_mode, "ENTER");
    aoclfftz_bench_status_t bench_status = BENCH_SUCCESS;
    // Initialize FFT plan for testing
    VOID *handle = params->setup_problem(params);
    if (handle == NULL)
    {
        bench_status = SETUP_FAILURE;
        HANDLE_BENCH_STATUS(bench_status);
        goto exit_sanity_mode;
    }
    for (INT32 i = 0; i < params->num_iterations; i++)
    {
        aoclfftz_error_type execute_status = aoclfftz_execute(handle);
        if (AOCLFFTZ_SUCCESS != execute_status)
        {
            bench_status = EXECUTION_FAILURE;
            HANDLE_BENCH_STATUS(bench_status);
            goto exit_sanity_mode;
        }
    }

exit_sanity_mode:
    // Cleanup resources
    aoclfftz_destroy(handle);
    return bench_status;
}

