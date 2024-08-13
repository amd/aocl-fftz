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

/** @file accuracy.c
 *
 *  @brief Accuracy test utility functions.
 *
 *  This file contains accuracy test related utility functions for test bench.
 *
 *  @author S. Biplab Raut
 *  @author V. Murugan
 *  @author Srirammaswamy Srinivasan
 */

#include <time.h>
#include "test/accuracy.h"
#include "test/utils/bench_utils.h"
#include "test/utils/size_and_index_mapper.h"

/**
 * @brief run the FFT execute api verify the linearity property
 *
 * @param params bench params object
 * @return INT32 bench status code
 */
INT32 run_linearity_test(aoclfftz_bench_params_t *params, INTP *in_idx_map,
                         INTP *out_idx_map)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "ENTER");
#endif
    INT32 status = BENCH_SUCCESS;
    INT32 ret = AOCLFFTZ_SUCCESS;
    INT32 dt_bytes = (params->precision == FLOAT_P) ?
                     sizeof(FLOAT) : sizeof(DOUBLE);
    INTP input_size = 0;
    INTP output_size = 0;

    INTP n = calculate_size(params->dims, params->dim_rank);
    INTP batches = calculate_size(params->vecs, params->vec_rank);
    UINT32 is_align = params->aligned_alloc;

    calculate_buffer_sizes(params, &input_size, &output_size);

    VOID *factors, *in1, *in2, *out1, *out2, *out_combined, *handle;
    factors = in1 = in2 = out1 = out2 = out_combined = handle = NULL;

    // create buffer to store 2 complex constant values
    ALLOC_INIT(factors, VOID, 4 * dt_bytes, is_align);
    // create locals buffer to store inputs and outputs
    ALLOC_UNINIT(in1, VOID, dt_bytes * input_size * T_DATA_STRIDE, is_align);
    ALLOC_UNINIT(in2, VOID, dt_bytes * input_size * T_DATA_STRIDE, is_align);
    if (factors == NULL || in1 == NULL || in2 == NULL)
    {
        printf("run_linearity_test : input buffer creation failed\n");
        status = MEMORY_FAILURE;
        goto exit_linearity_test;
    }
    ALLOC_INIT(out1, VOID, output_size * T_DATA_STRIDE * dt_bytes, is_align);
    ALLOC_INIT(out2, VOID, output_size * T_DATA_STRIDE * dt_bytes, is_align);
    ALLOC_INIT(out_combined, VOID,
                     output_size * T_DATA_STRIDE * dt_bytes, is_align);
    if (out1 == NULL || out2 == NULL || out_combined == NULL)
    {
        printf("run_linearity_test : output buffer creation failed\n");
        status = SETUP_FAILURE;
        goto exit_linearity_test;
    }

    // setup FFT problem
    handle = params->setup_problem(params);
    if (handle == NULL)
    {
        status = SETUP_FAILURE;
        goto exit_linearity_test;
    }

    // initialize the random seed value based on current time
    if (params->use_random_seed)
    {
        srand(time(0));
    }

    for (INT32 i = 0; i < params->num_iterations; i++)
    {
        // set the random seed value for each iteration
        if (params->use_random_seed)
        {
            params->seed = rand();
        }
        srand(params->seed);
#ifdef AOCL_ENABLE_LOG
        AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode,
                               "Iteration: %d, Seed: %d", i, params->seed);
#endif

        // prepare random input data
        // use in_stride as 1 to fill random data in all points
        params->prepare_input_data(in1, input_size, NULL, RANDOM_INPUT);
        params->prepare_input_data(in2, input_size, NULL, RANDOM_INPUT);

        // perform FFT for first input
        memcpy(params->in, in1, dt_bytes * input_size * T_DATA_STRIDE);
        ret = aoclfftz_execute(handle);
        if (ret != AOCLFFTZ_SUCCESS)
        {
            status = EXECUTION_FAILURE;
            goto exit_linearity_test;
        }
        memcpy(out1, params->out, dt_bytes * output_size * T_DATA_STRIDE);

        // perform FFT for second input
        memcpy(params->in, in2, dt_bytes * input_size * T_DATA_STRIDE);
        ret = aoclfftz_execute(handle);
        if (ret != AOCLFFTZ_SUCCESS)
        {
            status = EXECUTION_FAILURE;
            goto exit_linearity_test;
        }
        memcpy(out2, params->out, dt_bytes * output_size * T_DATA_STRIDE);

        // combine in1 and in2 and store the result in in1
        PREPARE_LINEAR_TEST_INPUTS(in1, in2, in1, input_size, factors,
                                   params->precision);

        // perform FFT for combined input
        memcpy(params->in, in1, dt_bytes * input_size * T_DATA_STRIDE);
        ret = aoclfftz_execute(handle);
        if (ret != AOCLFFTZ_SUCCESS)
        {
            status = EXECUTION_FAILURE;
            goto exit_linearity_test;
        }
        memcpy(out_combined, params->out,
               dt_bytes * output_size * T_DATA_STRIDE);

        // combine out1 and out2 and store the result in out1
        PREPARE_LINEAR_TEST_OUTPUTS(out1, out2, out1, output_size, factors,
                                    params->precision);

        // compare the outputs
        status = params->compare(params, out1, out_combined, batches, n,
                              out_idx_map);
        if (status != BENCH_SUCCESS)
        {
            printf("\nResults mismatch on accuracy mode => property: "
                   "linearity, iteration: %d/%d, seed: %d\n",
                   i, params->num_iterations, params->seed);
            status = VERIFICATION_FAILURE;
            goto exit_linearity_test;
        }
    }

exit_linearity_test:
    // destroy internal buffers
    FREE_ALLOCATED_MEM(in1, is_align);
    FREE_ALLOCATED_MEM(in2, is_align);
    FREE_ALLOCATED_MEM(out1, is_align);
    FREE_ALLOCATED_MEM(out2, is_align);
    FREE_ALLOCATED_MEM(out_combined, is_align);
    FREE_ALLOCATED_MEM(factors, is_align);
    // destroy handle
    aoclfftz_destroy(handle);
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "EXIT");
#endif
    return status;
}

/**
 * @brief run the FFT execute api verify the transformation property
 *
 * @param params bench params object
 * @return INT32 bench status code
 */
INT32 run_impulse_transform_test(aoclfftz_bench_params_t *params,
                                      INTP *in_idx_map, INTP *out_idx_map)
{
    INT32 status = BENCH_SUCCESS;
    INT32 ret = AOCLFFTZ_SUCCESS;
    INTP n = calculate_size(params->dims, params->dim_rank);
    INTP batches = calculate_size(params->vecs, params->vec_rank);
    INT32 dt_bytes = (params->precision == FLOAT_P) ?
                     sizeof(FLOAT) : sizeof(DOUBLE);
    INTP input_size = 0;
    INTP output_size = 0;
    UINT32 is_align = params->aligned_alloc;

    calculate_buffer_sizes(params, &input_size, &output_size);
    // create a new bench params for reverse FFT direction
    aoclfftz_bench_params_t *params_reverse = NULL;
    ALLOC_AND_COPY_PARAMS(params_reverse, params);
    if (params_reverse == NULL)
    {
        printf("run_impulse_transform_test : creating new bench params "
               "failed\n");
        return SETUP_FAILURE;
    }

    VOID *in, *handle, *handle_reverse;
    handle = handle_reverse = in = NULL;

    // create buffer to store input
    ALLOC_INIT(in, VOID, input_size * T_DATA_STRIDE * dt_bytes, is_align);
    if (in == NULL)
    {
        printf(
            "run_impulse_transform_test : input buffer creation failed\n");
        return SETUP_FAILURE;
    }

    // reverse the FFT direction and swap input output strides
    if (params->dir == FORWARD)
    {
        params_reverse->dir = BACKWARD;
    }
    else
    {
        params_reverse->dir = FORWARD;
    }
    // create in and out buffers for params_reverse object
    ALLOC_INIT(params_reverse->in, VOID,
                         output_size * T_DATA_STRIDE * dt_bytes, is_align);
    if (params_reverse->in == NULL)
    {
        status = MEMORY_FAILURE;
        goto exit_impulse_transform_test;
    }

    if (params->res_placement == IN_PLACE)
    {
        params_reverse->out = params_reverse->in;
    }
    else
    {
        ALLOC_INIT(params_reverse->out, VOID,
                         input_size * T_DATA_STRIDE * dt_bytes, is_align);
        if (params_reverse->out == NULL)
        {
            status = MEMORY_FAILURE;
            goto exit_impulse_transform_test;
        }
        for (INT32 i = 0; i < params->dim_rank; i++)
        {
            params_reverse->dims[i].in_stride = params->dims[i].out_stride;
            params_reverse->dims[i].out_stride = params->dims[i].in_stride;
        }

        for (INT32 i = 0; i < params->vec_rank; i++)
        {
            params_reverse->vecs[i].in_stride = params->vecs[i].out_stride;
            params_reverse->vecs[i].out_stride = params->vecs[i].in_stride;
        }
    }

    // setup FFT problem
    handle = params->setup_problem(params);
    if (handle == NULL)
    {
        status = SETUP_FAILURE;
        goto exit_impulse_transform_test;
    }

    // setup reverse FFT problem
    handle_reverse = params->setup_problem(params_reverse);
    if (handle_reverse == NULL)
    {
        status = SETUP_FAILURE;
        goto exit_impulse_transform_test;
    }

    // initialize the random seed value based on current time
    if (params->use_random_seed)
    {
        srand(time(0));
    }

    for (INT32 i = 0; i < params->num_iterations; i++)
    {
        // set the random seed value for each iteration
        if (params->use_random_seed)
        {
            params->seed = rand();
        }
        srand(params->seed);
#ifdef AOCL_ENABLE_LOG
        AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode,
                               "Iteration: %d, Seed: %d", i, params->seed);
#endif

        params->prepare_input_data(in, n * batches, in_idx_map, IMPULSE_INPUT);

        // perform FFT
        memcpy(params->in, in, dt_bytes * input_size * T_DATA_STRIDE);
        ret = aoclfftz_execute(handle);
        if (ret != AOCLFFTZ_SUCCESS)
        {
            status = EXECUTION_FAILURE;
            goto exit_impulse_transform_test;
        }

        // perform reversed FFT
        memcpy(params_reverse->in, params->out,
               dt_bytes * output_size * T_DATA_STRIDE);
        ret = aoclfftz_execute(handle_reverse);
        if (ret != AOCLFFTZ_SUCCESS)
        {
            status = EXECUTION_FAILURE;
            goto exit_impulse_transform_test;
        }
        NORMALIZE_IFFT_DATA(params_reverse->out, input_size, n,
                            params->precision);

        // compare reversed output with the input
        status = params->compare(params, in, params_reverse->out,
                              batches, n, in_idx_map);
        if (status != BENCH_SUCCESS)
        {
            printf("\nResults mismatch on accuracy mode => property: "
                   "transformation, iteration: %d/%d, seed: %d\n",
                   i, params->num_iterations, params->seed);
            status = VERIFICATION_FAILURE;
            goto exit_impulse_transform_test;
        }
    }

exit_impulse_transform_test:
    // destroy local buffer
    FREE_ALLOCATED_MEM(in, is_align);
    // destroy handles
    aoclfftz_destroy(handle);
    aoclfftz_destroy(handle_reverse);
    // destroy locally created bench param
    destroy_bench_param(params_reverse);
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "EXIT");
#endif
    return status;
}

/**
 * @brief run the FFT execute api verify the timeshift property
 *
 * @param params bench params object
 * @return INT32 bench status code
 */
INT32 run_timeshift_test(aoclfftz_bench_params_t *params, INTP *in_idx_map,
                         INTP *out_idx_map)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "ENTER");
#endif
    INT32 status = BENCH_SUCCESS;
    INT32 ret = AOCLFFTZ_SUCCESS;
    INTP n = calculate_size(params->dims, params->dim_rank);
    INTP batches = calculate_size(params->vecs, params->vec_rank);
    INT32 dt_bytes = (params->precision == FLOAT_P) ?
                     sizeof(FLOAT) : sizeof(DOUBLE);
    INTP input_size = 0;
    INTP output_size = 0;
    UINT32 is_align = params->aligned_alloc;

    calculate_buffer_sizes(params, &input_size, &output_size);

    VOID *in1, *in2, *out1, *out2, *handle;
    in1 = in2 = out1 = out2 = handle = NULL;
    // create buffers for inputs and outputs
    ALLOC_UNINIT(in1, VOID, dt_bytes * input_size * T_DATA_STRIDE, is_align);
    ALLOC_UNINIT(in2, VOID, dt_bytes * input_size * T_DATA_STRIDE, is_align);
    if (in1 == NULL || in2 == NULL)
    {
        printf("run_timeshift_test : input buffer creation failed\n");
        status = SETUP_FAILURE;
        goto exit_timeshift_test;
    }
    ALLOC_INIT(out1, VOID, output_size * T_DATA_STRIDE * dt_bytes, is_align);
    ALLOC_INIT(out2, VOID, output_size * T_DATA_STRIDE * dt_bytes, is_align);
    if (out1 == NULL || out2 == NULL)
    {
        printf("run_timeshift_test : output buffer creation failed\n");
        status = SETUP_FAILURE;
        goto exit_timeshift_test;
    }

    // setup FFT problem
    handle = params->setup_problem(params);
    if (handle == NULL)
    {
        status = SETUP_FAILURE;
        goto exit_timeshift_test;
    }

    // initialize the random seed value based on current time
    if (params->use_random_seed)
    {
        srand(time(0));
    }

    for (INT32 i = 0; i < params->num_iterations; i++)
    {
        // set the random seed value for each iteration
        if (params->use_random_seed)
        {
            params->seed = rand();
        }
        srand(params->seed);
#ifdef AOCL_ENABLE_LOG
        AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode,
                               "Iteration: %d, Seed: %d", i, params->seed);
#endif

        params->prepare_input_data(in1, n * batches, in_idx_map, RANDOM_INPUT);

        INTP cur_n;
        INTP outer_n = 1;
        INTP inner_n = n;
        INTP unit_m = n;

        for (INTP d = params->dim_rank - 1; d >= 0; d--)
        {
            cur_n = params->dims[d].n;
            unit_m /= cur_n;
            // random no. of shifts in the range of 0 to current dimension size
            INTP m = rand() % cur_n;
            // shifting units wrt linear representation
            // for example: A ND problem of 3x4x5 when dim=2 is processed
            // if m = 2, then the actual unit to be shifted for every element
            // would be 2*20
            INTP shifts = m * unit_m;

            // time shift for each batch
            for (INTP b = 0; b < batches ; b++)
            {
                // time shift the input for each outer dimension
                // for example: A ND problem of 3x4x5 when dim=1 is processed
                // it will shift the input m times in every 4x5 matrix for
                // 3(outer_n) times
                for (INTP o = 0; o < outer_n; o++)
                {
                    PREPARE_TIMESHIFT_TEST_INPUTS(
                        in1 + in_idx_map[b * n + o * inner_n] * T_DATA_STRIDE,
                        in2 + in_idx_map[b * n + o * inner_n] * T_DATA_STRIDE,
                        inner_n, shifts, in_idx_map, params->precision);
                }
            }

            // perform FFT for input
            memcpy(params->in, in1, dt_bytes * input_size * T_DATA_STRIDE);
            ret = aoclfftz_execute(handle);
            if (ret != AOCLFFTZ_SUCCESS)
            {
                status = EXECUTION_FAILURE;
                goto exit_timeshift_test;
            }
            memcpy(out1, params->out, dt_bytes * output_size * T_DATA_STRIDE);

            // perform FFT for shifted input
            memcpy(params->in, in2, dt_bytes * input_size * T_DATA_STRIDE);
            status |= aoclfftz_execute(handle);
            memcpy(out2, params->out, dt_bytes * output_size * T_DATA_STRIDE);

            // perform phase shift on FFT(input)
            for (INTP b = 0; b < batches; b++)
            {
                for (INTP o = 0; o < outer_n; o++)
                {
                    PREPARE_TIMESHIFT_TEST_OUTPUTS(
                        out1 + out_idx_map[b * n + o * inner_n] * T_DATA_STRIDE,
                        out1 + out_idx_map[b * n + o * inner_n] * T_DATA_STRIDE,
                        cur_n, m, unit_m, out_idx_map, params->dir,
                        params->precision);
                }
            }

            inner_n /= cur_n;
            outer_n *= cur_n;

            // compare the outputs
            status = params->compare(params, out1, out2,
                                     batches, n, out_idx_map);
            if (status != BENCH_SUCCESS)
            {
                printf("\nResults mismatch on accuracy mode => property: "
                    "timeshift (dim = %td), iteration: %d/%d, seed: %d\n",
                    d, i, params->num_iterations, params->seed);
                status = VERIFICATION_FAILURE;
                goto exit_timeshift_test;
            }
        }
    }

exit_timeshift_test:
    // destroy internal buffers
    FREE_ALLOCATED_MEM(in1, is_align);
    FREE_ALLOCATED_MEM(in2, is_align);
    FREE_ALLOCATED_MEM(out1, is_align);
    FREE_ALLOCATED_MEM(out2, is_align);
    // destroy handle
    aoclfftz_destroy(handle);
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "EXIT");
#endif
    return status;
}

/**
 * @brief run the test bench on accuracy mode by verifying the following DFT
 * properties.
 *          1. Linearity
 *          2. Impulse Transformation
 *          3. Timeshift
 *
 * @param params bench params object
 * @return INT32 bench status code
 */
INT32 run_bench_on_accuracy_mode(aoclfftz_bench_params_t *params)
{
    INT32 status = BENCH_SUCCESS;

    INTP n = calculate_size(params->dims, params->dim_rank);
    INTP batches = calculate_size(params->vecs, params->vec_rank);
    UINT32 is_align = params->aligned_alloc;
    INTP *in_idx_map = NULL;
    ALLOC_UNINIT(in_idx_map, INTP, n * batches * sizeof(INTP), is_align);
    INTP *out_idx_map = NULL;
    ALLOC_UNINIT(out_idx_map, INTP, n * batches * sizeof(INTP), is_align);
    if (in_idx_map == NULL || out_idx_map == NULL)
    {
        printf("run_bench_on_accuracy_mode : idx_map buffer creation failed\n");
        status = MEMORY_FAILURE;
        goto exit_accuracy_mode;
    }
    // prepare index map which maps the strided indices with non-strided ones
    prepare_index_map(params, in_idx_map, out_idx_map);

#ifdef ENABLE_DFT_REFERENCE
    status = run_dft_reference_test(params, in_idx_map, out_idx_map);
    HANDLE_BENCH_STATUS(status);
    if (status != BENCH_SUCCESS)
    {
        goto exit_accuracy_mode;
    }
#endif

    // run property tests
    // 1. linearity property
    status = run_linearity_test(params, in_idx_map, out_idx_map);
    HANDLE_BENCH_STATUS(status);
    if (status != BENCH_SUCCESS)
    {
        goto exit_accuracy_mode;
    }

    // 2. transformation test
    status = run_impulse_transform_test(params, in_idx_map, out_idx_map);
    HANDLE_BENCH_STATUS(status);
    if (status != BENCH_SUCCESS)
    {
        goto exit_accuracy_mode;
    }

    // 3. timeshift test
    status = run_timeshift_test(params, in_idx_map, out_idx_map);
    HANDLE_BENCH_STATUS(status);
    if (status != BENCH_SUCCESS)
    {
        goto exit_accuracy_mode;
    }

    PRINT_SUCCESS("\nTest bench completed on accuracy mode\n\n");

exit_accuracy_mode:
    FREE_ALLOCATED_MEM(in_idx_map, is_align);
    FREE_ALLOCATED_MEM(out_idx_map, is_align);
    return status;
}
