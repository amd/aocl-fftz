// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file aoclfftz_bench.c
 *
 *  @brief Tests the single-threaded core fft library for functional and
 *  performance tests.
 *
 *  This file contains the functions to setup and execute single-threaded core
 *  library for testing the library APIs.
 *
 *  @author S. Biplab Raut
 *  @author V. Murugan
 *  @author Srirammaswamy Srinivasan
 *  @author Jeya R
 */

#include "aoclfftz_bench.h"
#include "test/bench/accuracy.h"
#include "test/bench/performance.h"
#include "test/bench/sanity.h"

/**
 * @brief Entry function to test bench
 *
 * @param argc command-line argument count
 * @param argv command-line argument values as char array
 * @return FFTZ_INT32 status code: 0 indicates success
 *                 negative value indicates bench error code
 *                 positive value indicates specific parser error code
 */
FFTZ_INT32 main(FFTZ_INT32 argc, FFTZ_CHAR **argv)
{
    printf("\nAOCL-FFTZ version: %s\n\n", aoclfftz_version());

    FFTZ_INT32 status = BENCH_SUCCESS;

    // prepare bench params from user inputs
    aoclfftz_bench_params_t *params = NULL;
    ALLOC_ALIGN_UNINIT(params, aoclfftz_bench_params_t,
                         sizeof(aoclfftz_bench_params_t));
    if (params == NULL)
    {
        status = MEMORY_FAILURE;
        goto exit_main;
    }

    status = prepare_bench_params(argc, argv, params);
    HANDLE_PARSER_ERROR_MESSAGE(status);
    if (status != PARSER_SUCCESS)
    {
        goto exit_main;
    }

    // log the user params in INFO mode
    LOG_BENCH_PARAMS(params);

    if (params->bench_type == PERFORMANCE)
    {
        printf("\nRunning bench on performance mode\n");
        status = run_bench_on_performance_mode(params);
    }
    else if (params->bench_type == SANITY)
    {
        printf("\nRunning bench on sanity mode\n");
        status = run_bench_on_sanity_mode(params);
    }
    else // params->bench_type == ACCURACY
    {
        printf("\nRunning bench on accuracy mode\n");
        status = run_bench_on_accuracy_mode(params);
    }

exit_main:
    destroy_bench_param(params);
    return status;
}
