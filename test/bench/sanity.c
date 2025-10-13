/**
 * Copyright (C) 2025, Advanced Micro Devices. All rights reserved.
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

