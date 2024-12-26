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
#include "test/bench/accuracy.h"
#include "test/bench/utils/bench_utils.h"
#include "test/bench/utils/data_conversion.h"
#include "test/bench/utils/size_and_index_mapper.h"

/**
 * @brief run the FFT execute api verify the linearity property
 *
 * @param params bench params object
 * @return INT32 bench status code
 */
INT32 run_linearity_test(aoclfftz_bench_params_t *params, INTP *in_idx_map,
                         INTP *out_idx_map, VOID *handle, VOID *input_buffer)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "ENTER");
#endif
    INT32 status = BENCH_SUCCESS;
    INT32 ret = AOCLFFTZ_SUCCESS;
    INT32 dt_bytes = (params->precision == FLOAT_P) ?
                     sizeof(FLOAT) : sizeof(DOUBLE);
    UINT32 is_align = params->aligned_alloc;
    INTP input_bytes =
        dt_bytes * params->sz_info.input_size * params->in_data_stride;
    INTP output_bytes =
        dt_bytes * params->sz_info.output_size * params->out_data_stride;

    // In an R2C problem, input will have N real points and output will have N
    // complex points and vise-versa for C2R.
    // For in-place R2C/C2R problems, since the same buffer will be used for
    // input and output, it should be large enough to hold the N complex points.
    if (params->fft_type != C2C && params->res_placement == IN_PLACE)
    {
        input_bytes = MAX(input_bytes, output_bytes);
        output_bytes = input_bytes;
    }

    VOID *factors, *in1, *in2, *out1, *out2, *out_combined;
    factors = in1 = in2 = out1 = out2 = out_combined = NULL;

    // create buffer to store 2 complex constant values
    ALLOC_INIT(factors, VOID, 4 * dt_bytes, is_align);
    // create locals buffer to store inputs and outputs
    ALLOC_INIT(in1, VOID, input_bytes, is_align);
    ALLOC_INIT(in2, VOID, input_bytes, is_align);
    if (factors == NULL || in1 == NULL || in2 == NULL)
    {
        AOCLFFTZ_LOG_UNFORMATTED(ERR, ERR, "input buffer creation failed");
        status = MEMORY_FAILURE;
        goto exit_linearity_test;
    }
    ALLOC_INIT(out1, VOID, output_bytes, is_align);
    ALLOC_INIT(out2, VOID, output_bytes, is_align);
    ALLOC_INIT(out_combined, VOID, output_bytes, is_align);
    if (out1 == NULL || out2 == NULL || out_combined == NULL)
    {
        AOCLFFTZ_LOG_UNFORMATTED(ERR, ERR, "output buffer creation failed");
        status = MEMORY_FAILURE;
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
        params->prepare_input_data(in2, params->sz_info.input_size, NULL,
                                   RANDOM_INPUT, params->in_data_stride);

        if (input_buffer == NULL)
        {
            params->prepare_input_data(in1, params->sz_info.input_size, NULL,
                                       RANDOM_INPUT, params->in_data_stride);
        }
        else
        {
            memcpy(in1, input_buffer, input_bytes);
        }
        // perform FFT for first input
        memcpy(params->in, in1, input_bytes);
        ret = aoclfftz_execute(handle);
        if (ret != AOCLFFTZ_SUCCESS)
        {
            status = EXECUTION_FAILURE;
            goto exit_linearity_test;
        }
        memcpy(out1, params->out, output_bytes);

        // perform FFT for second input
        memcpy(params->in, in2, input_bytes);
        ret = aoclfftz_execute(handle);
        if (ret != AOCLFFTZ_SUCCESS)
        {
            status = EXECUTION_FAILURE;
            goto exit_linearity_test;
        }
        memcpy(out2, params->out, output_bytes);

        // combine in1 and in2 and store the result in in1
        PREPARE_LINEAR_TEST_INPUTS(in1, in2, in1, params->sz_info.input_size,
                                   factors, params->precision,
                                   params->in_data_stride);

        // perform FFT for combined input
        memcpy(params->in, in1, input_bytes);
        ret = aoclfftz_execute(handle);
        if (ret != AOCLFFTZ_SUCCESS)
        {
            status = EXECUTION_FAILURE;
            goto exit_linearity_test;
        }
        memcpy(out_combined, params->out, output_bytes);

        // combine out1 and out2 and store the result in out1
        PREPARE_LINEAR_TEST_OUTPUTS(out1, out2, out1,
                                    params->sz_info.output_size, factors,
                                    params->precision, params->out_data_stride);
        // compare the outputs
        status = params->compare(params, out1, out_combined,
                                 params->sz_info.batches, params->sz_info.n,
                                 out_idx_map, params->out_data_stride);

        if (status != BENCH_SUCCESS)
        {
            printf("\nResults mismatch on accuracy mode => property: "
                   "linearity, iteration: %d/%d, seed: %d\n",
                   i, params->num_iterations, params->seed);
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
                                 INTP *in_idx_map, INTP *out_idx_map,
                                 VOID *handle, VOID *input_buffer)
{
    INT32 status = BENCH_SUCCESS;
    INT32 ret = AOCLFFTZ_SUCCESS;
    INT32 dt_bytes = (params->precision == FLOAT_P) ?
                     sizeof(FLOAT) : sizeof(DOUBLE);
    UINT32 is_align = params->aligned_alloc;
    INTP input_bytes =
        dt_bytes * params->sz_info.input_size * params->in_data_stride;
    INTP output_bytes =
        dt_bytes * params->sz_info.output_size * params->out_data_stride;

    // In an R2C problem, input will have N real points and output will have N
    // complex points and vise-versa for C2R.
    // For in-place R2C/C2R problems, since the same buffer will be used for
    // input and output, it should be large enough to hold the N complex points.
    if (params->fft_type != C2C && params->res_placement == IN_PLACE)
    {
        input_bytes = MAX(input_bytes, output_bytes);
        output_bytes = input_bytes;
    }

    // create a new bench params for reverse FFT direction
    aoclfftz_bench_params_t *params_reverse = NULL;
    VOID *in, *handle_reverse;
    handle_reverse = in = NULL;

    ALLOC_AND_COPY_PARAMS(params_reverse, params);
    if (params_reverse == NULL)
    {
        AOCLFFTZ_LOG_UNFORMATTED(ERR, ERR, "params_reverse creation failed");
        status = MEMORY_FAILURE;
        goto exit_impulse_transform_test;
    }

    // create buffer to store input
    ALLOC_INIT(in, VOID, input_bytes, is_align);
    if (in == NULL)
    {
        AOCLFFTZ_LOG_UNFORMATTED(ERR, ERR, "input buffer creation failed");
        status = MEMORY_FAILURE;
        goto exit_impulse_transform_test;
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
    ALLOC_INIT(params_reverse->in, VOID, output_bytes, is_align);
    if (params_reverse->in == NULL)
    {
        AOCLFFTZ_LOG_UNFORMATTED(ERR, ERR, "input buffer creation failed");
        status = MEMORY_FAILURE;
        goto exit_impulse_transform_test;
    }

    if (params->res_placement == IN_PLACE)
    {
        params_reverse->out = params_reverse->in;
    }
    else
    {
        ALLOC_INIT(params_reverse->out, VOID, input_bytes, is_align);
        if (params_reverse->out == NULL)
        {
            AOCLFFTZ_LOG_UNFORMATTED(ERR, ERR, "output buffer creation failed");
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
        if (input_buffer == NULL)
        {
            params->prepare_input_data(in, params->sz_info.n *
                                       params->sz_info.batches, in_idx_map,
                                       IMPULSE_INPUT, params->in_data_stride);
        }
        else
        {
            memcpy(in, input_buffer, input_bytes);
        }
        if (params->fft_type == R2C && params->dir == BACKWARD)
        {
            convert_complex_to_half_complex(in, params->sz_info.n,
                                            params->sz_info.batches,
                                            in_idx_map, params->precision);
        }

        // perform FFT
        memcpy(params->in, in, input_bytes);
        ret = aoclfftz_execute(handle);
        if (ret != AOCLFFTZ_SUCCESS)
        {
            status = EXECUTION_FAILURE;
            goto exit_impulse_transform_test;
        }

        // perform reversed FFT
        if (params->fft_type == R2C && params->dir == FORWARD)
        {
            convert_half_complex_to_complex(params_reverse->in, params->out,
                                            params->sz_info.n,
                                            params->sz_info.batches,
                                            out_idx_map,
                                            params->precision);
        }
        else
        {
            memcpy(params_reverse->in, params->out, output_bytes);
        }
        ret = aoclfftz_execute(handle_reverse);
        if (ret != AOCLFFTZ_SUCCESS)
        {
            status = EXECUTION_FAILURE;
            goto exit_impulse_transform_test;
        }

        if (params->fft_type == R2C && params->dir == BACKWARD)
        {
            convert_half_complex_to_complex(params_reverse->out,
                                            params_reverse->out,
                                            params->sz_info.n,
                                            params->sz_info.batches,
                                            in_idx_map,
                                            params->precision);
        }
        NORMALIZE_IFFT_DATA(params_reverse->out, params->sz_info.input_size,
                            params->sz_info.n, params->precision,
                            params->in_data_stride);

        // compare reversed output with the input
        status = params->compare(params, in, params_reverse->out,
                                 params->sz_info.batches, params->sz_info.n,
                                 in_idx_map, params->in_data_stride);
        if (status != BENCH_SUCCESS)
        {
            printf("\nResults mismatch on accuracy mode => property: "
                   "transformation, iteration: %d/%d, seed: %d\n",
                   i, params->num_iterations, params->seed);
            goto exit_impulse_transform_test;
        }
    }

exit_impulse_transform_test:
    // destroy local buffer
    FREE_ALLOCATED_MEM(in, is_align);
    aoclfftz_destroy(handle_reverse);
    // destroy locally created bench param
    destroy_bench_param(params_reverse);
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "EXIT");
#endif
    return status;
}

#define TIME_SHIFT(in, out, idx_map, data_stride)                              \
{                                                                              \
    for (INTP b = 0; b < batches; b++)                                         \
    {                                                                          \
        /* time shift the input for each outer dimension                       \
           for example: An ND problem of 3x4x5 when dim=2 (i.e. outer dim,     \
           where n=3) is processed it will shift the input m times in          \
           every 4x5 matrix for 3 (outer_n) times */                           \
        for (INTP o = 0; o < outer_n; o++)                                     \
        {                                                                      \
            PREPARE_TIMESHIFT_TEST_INPUTS(                                     \
                in + idx_map[b * n + o * inner_n] * data_stride,               \
                out + idx_map[b * n + o * inner_n] * data_stride,              \
                inner_n, shifts, idx_map, params->precision,                   \
                data_stride);                                                  \
        }                                                                      \
    }                                                                          \
}

#define PHASE_SHIFT(in, out, idx_map, data_stride)                             \
{                                                                              \
    for (INTP b = 0; b < batches; b++)                                         \
    {                                                                          \
        for (INTP o = 0; o < outer_n; o++)                                     \
        {                                                                      \
            PREPARE_TIMESHIFT_TEST_OUTPUTS(                                    \
                in + idx_map[b * n + o * inner_n] * data_stride,               \
                out + idx_map[b * n + o * inner_n] * data_stride,              \
                cur_n, m, unit_m, idx_map, params->dir,                        \
                params->precision, 2);                                         \
        }                                                                      \
    }                                                                          \
}

/**
 * @brief run the FFT execute api verify the timeshift property
 *
 * @param params bench params object
 * @return INT32 bench status code
 */
INT32 run_timeshift_test(aoclfftz_bench_params_t *params, INTP *in_idx_map,
                         INTP *out_idx_map, VOID * handle, VOID *input_buffer)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "ENTER");
#endif
    INT32 status = BENCH_SUCCESS;
    INT32 ret = AOCLFFTZ_SUCCESS;
    INTP n = params->sz_info.n;
    INTP batches = params->sz_info.batches;
    INT32 dt_bytes = (params->precision == FLOAT_P) ?
                     sizeof(FLOAT) : sizeof(DOUBLE);
    UINT32 is_align = params->aligned_alloc;

    VOID *in1, *in2, *out1, *out2, *out_temp;
    in1 = in2 = out1 = out2 = out_temp = NULL;
    INTP input_bytes =
        dt_bytes * params->sz_info.input_size * params->in_data_stride;
    INTP output_bytes =
        dt_bytes * params->sz_info.output_size * params->out_data_stride;
    INTP complex_output_bytes =
        dt_bytes * params->sz_info.output_size * 2;

    // In an R2C problem, input will have N real points and output will have N
    // complex points and vise-versa for C2R.
    // For in-place R2C/C2R problems, since the same buffer will be used for
    // input and output, it should be large enough to hold the N complex points.
    if (params->fft_type != C2C && params->res_placement == IN_PLACE)
    {
        input_bytes = MAX(input_bytes, output_bytes);
        output_bytes = input_bytes;
    }

    // create buffers for inputs and outputs
    ALLOC_UNINIT(in1, VOID, input_bytes, is_align);
    ALLOC_UNINIT(in2, VOID, input_bytes, is_align);
    if (in1 == NULL || in2 == NULL)
    {
        AOCLFFTZ_LOG_UNFORMATTED(ERR, ERR, "input buffer creation failed");
        status = MEMORY_FAILURE;
        goto exit_timeshift_test;
    }
    ALLOC_INIT(out1, VOID, output_bytes, is_align);
    ALLOC_INIT(out2, VOID, complex_output_bytes, is_align);
    if (out1 == NULL || out2 == NULL)
    {
        AOCLFFTZ_LOG_UNFORMATTED(ERR, ERR, "output buffer creation failed");
        status = MEMORY_FAILURE;
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

        if (input_buffer == NULL)
        {
            params->prepare_input_data(in1, n * batches, in_idx_map,
                                       RANDOM_INPUT, params->in_data_stride);
        }
        else
        {
            memcpy(in1, input_buffer, input_bytes);
        }
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

            if (params->dir == FORWARD)
            {
                // shifting units wrt linear representation
                // for example: An ND problem of 3x4x5 when dim=2 is processed
                // if m = 2, then the actual unit to be shifted for every
                // element would be 2*20
                INTP shifts = m * unit_m;
                TIME_SHIFT(in1, in2, in_idx_map, params->in_data_stride)
            }
            else
            {
                PHASE_SHIFT(in1, in2, in_idx_map, params->in_data_stride)
            }

            // perform FFT for input
            memcpy(params->in, in1, input_bytes);
            ret = aoclfftz_execute(handle);
            if (ret != AOCLFFTZ_SUCCESS)
            {
                status = EXECUTION_FAILURE;
                goto exit_timeshift_test;
            }
            memcpy(out1, params->out, output_bytes);

            // perform FFT for shifted input
            memcpy(params->in, in2, input_bytes);
            status |= aoclfftz_execute(handle);
            memcpy(out2, params->out, output_bytes);

            if (params->fft_type == R2C && params->dir == FORWARD)
            {
                convert_half_complex_to_complex(out1, out1, n, batches,
                                                out_idx_map, params->precision);
                convert_half_complex_to_complex(out2, out2, n, batches,
                                                out_idx_map, params->precision);
            }
            if (params->dir == FORWARD)
            {
                PHASE_SHIFT(out1, out1, out_idx_map, params->out_data_stride)
            }
            else
            {
                ALLOC_INIT(out_temp, VOID, output_bytes, is_align);
                if (out_temp == NULL)
                {
                    AOCLFFTZ_LOG_UNFORMATTED(ERR, ERR,
                                             "output buffer creation failed");
                    status = MEMORY_FAILURE;
                    goto exit_timeshift_test;
                }
                INTP shifts = m * unit_m;
                TIME_SHIFT(out1, out_temp, out_idx_map,
                           params->out_data_stride);
                FREE_ALLOCATED_MEM(out1, is_align);
                out1 = out_temp;
            }

            inner_n /= cur_n;
            outer_n *= cur_n;

            // compare the outputs
            status = params->compare(params, out1, out2, batches, n,
                                     out_idx_map, params->out_data_stride);
            if (status != BENCH_SUCCESS)
            {
                printf("\nResults mismatch on accuracy mode => property: "
                    "timeshift (dim = %td), iteration: %d/%d, seed: %d\n",
                    d, i, params->num_iterations, params->seed);
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
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "ENTER");
#endif
    INT32 status = BENCH_SUCCESS;
    VOID *handle = NULL;

    UINT32 is_align = params->aligned_alloc;
    params->sz_info.n = calculate_size(params->dims, params->dim_rank);
    params->sz_info.batches = calculate_size(params->vecs, params->vec_rank);
    INTP *in_idx_map = NULL;
    ALLOC_UNINIT(in_idx_map, INTP, params->sz_info.n * params->sz_info.batches
                 * sizeof(INTP), is_align);
    INTP *out_idx_map = NULL;
    ALLOC_UNINIT(out_idx_map, INTP, params->sz_info.n *
                 params->sz_info.batches * sizeof(INTP), is_align);
    if (in_idx_map == NULL || out_idx_map == NULL)
    {
        AOCLFFTZ_LOG_UNFORMATTED(ERR, ERR, "idx_map creation failed");
        status = MEMORY_FAILURE;
        goto exit_accuracy_mode;
    }

    // prepare index map which maps the strided indices with non-strided ones
    prepare_index_map(params->dim_rank, params->vec_rank, params->dims,
                    params->vecs, in_idx_map, out_idx_map, is_align);
    calculate_buffer_sizes(params->dim_rank, params->vec_rank, params->dims,
                             params->vecs, &(params->sz_info.input_size),
                             &(params->sz_info.output_size));

    // setup FFT problem
    handle = params->setup_problem(params);
    if (handle == NULL)
    {
        status = SETUP_FAILURE;
        HANDLE_BENCH_STATUS(status);
        goto exit_accuracy_mode;
    }

#ifdef ENABLE_DFT_REFERENCE
    status = run_dft_reference_test(params, in_idx_map, out_idx_map, handle,
                                    NULL);
    HANDLE_BENCH_STATUS(status);
    if (status != BENCH_SUCCESS)
    {
        goto exit_accuracy_mode;
    }
#endif

    // run property tests
    // 1. linearity property
    status = run_linearity_test(params, in_idx_map, out_idx_map, handle, NULL);
    HANDLE_BENCH_STATUS(status);
    if (status != BENCH_SUCCESS)
    {
        goto exit_accuracy_mode;
    }

    // 2. timeshift test
    status = run_timeshift_test(params, in_idx_map, out_idx_map, handle, NULL);
    HANDLE_BENCH_STATUS(status);
    if (status != BENCH_SUCCESS)
    {
        goto exit_accuracy_mode;
    }

    // 3. transformation test
    status = run_impulse_transform_test(params, in_idx_map, out_idx_map, handle,
                                        NULL);
    HANDLE_BENCH_STATUS(status);
    if (status != BENCH_SUCCESS)
    {
        goto exit_accuracy_mode;
    }

    PRINT_SUCCESS("\nTest bench completed on accuracy mode\n\n");

exit_accuracy_mode:

    FREE_ALLOCATED_MEM(in_idx_map, is_align);
    FREE_ALLOCATED_MEM(out_idx_map, is_align);
    aoclfftz_destroy(handle);
    return status;
}
