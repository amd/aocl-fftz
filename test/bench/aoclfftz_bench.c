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

/**
 * @brief Entry function to test bench
 *
 * @param argc command-line argument count
 * @param argv command-line argument values as char array
 * @return INT32 status code: 0 indicates success
 *                 negative value indicates bench error code
 *                 positive value indicates specific parser error code
 */
INT32 main(INT32 argc, CHAR **argv)
{
    printf("\nAOCL-FFTZ version: %s\n\n", aoclfftz_version());

    INT32 status = BENCH_SUCCESS;

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
    else // params->bench_type == ACCURACY
    {
        printf("\nRunning bench on accuracy mode\n");
        status = run_bench_on_accuracy_mode(params);
    }

exit_main:
    destroy_bench_param(params);
    return status;
}
