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

/** @file dft_reference.c
 *
 *  @brief DFT reference implementation.
 *
 *  This file contains the implementation of DFT reference and its helper
 *  functions.
 *
 *  @author V. Murugan
 *  @author Srirammaswamy Srinivasan
 *  @author Jeya R
 */

#ifdef _WINDOWS
#include <time.h>
#endif
#include "test/dft_reference.h"
#include "test/utils/compare.h"
#include "test/utils/bench_utils.h"
#include "test/utils/size_and_index_mapper.h"

#ifdef ENABLE_DFT_REFERENCE

/**
 * @brief run the FFT execute api and compare the output with DFT reference
 * output
 *
 * @param params bench params object
 * @return INT32 bench status code
 */
INT32 run_dft_reference_test(aoclfftz_bench_params_t *params, INTP *in_idx_map,
                             INTP *out_idx_map)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "ENTER");
#endif
    INT32 status = BENCH_SUCCESS;
    INT32 compare_status = AOCLFFTZ_SUCCESS;
    INT32 dt_bytes = (params->precision == FLOAT_P) ?
                     sizeof(FLOAT) : sizeof(DOUBLE);
    INTP input_size = 0;
    INTP output_size = 0;
    calculate_buffer_sizes(params, &input_size, &output_size);

    INTP n = calculate_size(params->dims, params->dim_rank);
    INTP batches = calculate_size(params->vecs, params->vec_rank);
    UINT32 is_align = params->aligned_alloc;

    // setup FFT problem
    VOID *handle = params->setup_problem(params);
    if (handle == NULL)
    {
        return SETUP_FAILURE;
    }

    // create local buffer to store DFT reference output
    VOID *out_ref;
    ALLOC_INIT(out_ref, VOID, output_size * T_DATA_STRIDE * dt_bytes, is_align);

    // initialize the random seed value based on current time
    if (params->use_random_seed)
    {
        srand(time(0));
    }

    for (INT32 i = 0; i < params->num_iterations && status == 0; i++)
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
        params->prepare_input_data(params->in, input_size, NULL, RANDOM_INPUT);

        // get the DFT reference output
        params->dft_ref(params, out_ref, in_idx_map, out_idx_map);
        status |= aoclfftz_execute(handle);

        if (status != BENCH_SUCCESS)
        {
            // destroy reference output buffer
            FREE_ALLOCATED_MEM(out_ref, is_align);
            // destroy handle
            aoclfftz_destroy(handle);
            return EXECUTION_FAILURE;
        }

        // compare the FFT output with DFT reference output
        compare_status =
            params->compare(params, out_ref, params->out, batches, n, out_idx_map);
        if (compare_status != AOCLFFTZ_SUCCESS)
        {
            printf("\nResults mismatch on accuracy mode => DFT reference, "
                   "iteration: %d/%d, seed: %d\n",
                   i, params->num_iterations, params->seed);
            // destroy reference output buffer
            FREE_ALLOCATED_MEM(out_ref, is_align);
            // destroy handle
            aoclfftz_destroy(handle);
            return VERIFICATION_FAILURE;
        }
    }
    // destroy reference output buffer
    FREE_ALLOCATED_MEM(out_ref, is_align);
    // destroy handle
    aoclfftz_destroy(handle);
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "EXIT");
#endif
    return BENCH_SUCCESS;
}

/**
 * @brief DFT reference implementation for FLOAT type
 * Computes multi dimensional DFT using the below formula
 *  X[k] = ∑x[n]⋅e ^-j2πk.n/N
 *  where n = (n1, n2,...,nd)
 *  k = (k1, k2,...,kd)
 *  N = (N1, N2,...,Nd)
 *  k.n/N is the dot product of Summation(ki*ni/Ni)on i=0 to d
 * @param params aoclfftz_bench_params_t struct containing the req info
 * @param out_buf buffer to store the DFT output
 * @param in_idx_map index map input indices
 * @param out_idx_map index map for output indices
 * @return VOID
 */
VOID dft_ref_f(aoclfftz_bench_params_t *params, VOID *out_buf, INTP *in_idx_map,
               INTP *out_idx_map)
{
    // intermediate variables are in DOUBLE  data type to improve accuracy
    DOUBLE e[T_DATA_STRIDE], mul_buf[T_DATA_STRIDE];
    FLOAT sign = params->dir == BACKWARD ? 1.0 : -1.0;
    FLOAT *in_f = (FLOAT *)params->in;
    FLOAT *out_f = (FLOAT *)out_buf;
    INT32 rank = params->dim_rank;
    aoclfftz_dim_t_64_ *dims = params->dims;
    INTP n = calculate_size(params->dims, params->dim_rank);
    INTP batches = calculate_size(params->vecs, params->vec_rank);
    UINT32 is_align = params->aligned_alloc;

    INTP *in_counter = NULL;
    ALLOC_INIT(in_counter, INTP, rank * sizeof(INTP), is_align);
    INTP *out_counter = NULL;
    ALLOC_INIT(out_counter, INTP, rank * sizeof(INTP), is_align);

    // iterate over the total batches
    for (INTP b = 0; b < batches; b++)
    {
        RESET_ND_COUNTER(out_counter, rank);
        // iterate over output points
        for (INTP k = 0; k < n; k++)
        {
            INTP out_idx = out_idx_map[b * n + k] * T_DATA_STRIDE;
            RESET_ND_COUNTER(in_counter, rank);
            // iterate over input points
            for (INTP i = 0; i < n; i++)
            {
                INTP in_idx = in_idx_map[b * n + i] * T_DATA_STRIDE;
                DOUBLE angle = sign * BENCH_2_PI;
                // angle = angle * [(i0+k0)/n0 + (i0+k0)/n0 + ... + (iR+kR)/nR]
                UPDATE_ANGLE(angle, in_counter, out_counter, dims, rank);
                e[0] = cos(angle);
                e[1] = sin(angle);

                mul_buf[0] = (in_f[in_idx] * e[0]) - (in_f[in_idx + 1] * e[1]);
                mul_buf[1] = (in_f[in_idx] * e[1]) + (in_f[in_idx + 1] * e[0]);

                out_f[out_idx] = out_f[out_idx] + mul_buf[0];
                out_f[out_idx + 1] = out_f[out_idx + 1] + mul_buf[1];

                INCREMENT_ND_COUNTER(in_counter, dims, rank);
            }
            INCREMENT_ND_COUNTER(out_counter, dims, rank);
        }
    }
    FREE_ALLOCATED_MEM(in_counter, is_align);
    FREE_ALLOCATED_MEM(out_counter, is_align);
}

/**
 * @brief DFT reference implementation for DOUBLE type
 * Computes multi dimensional DFT using the below formula
 *  X[k] = ∑x[n]⋅e ^-j2πk.n/N
 *  where n = (n1, n2,...,nd)
 *  k = (k1, k2,...,kd)
 *  N = (N1, N2,...,Nd)
 *  k.n/N is the dot product of Summation(ki*ni/Ni)on i=0 to d
 * @param params aoclfftz_bench_params_t struct containing the req info
 * @param out_buf buffer to store the DFT output
 * @param in_idx_map index map input indices
 * @param out_idx_map index map for output indices
 * @return VOID
 */
VOID dft_ref_d(aoclfftz_bench_params_t *params, VOID *out_buf, INTP *in_idx_map,
               INTP *out_idx_map)
{
    DOUBLE e[T_DATA_STRIDE], mul_buf[T_DATA_STRIDE];
    DOUBLE sign = params->dir == BACKWARD ? 1.0 : -1.0;
    DOUBLE *in_d = (DOUBLE *)params->in;
    DOUBLE *out_d = (DOUBLE *)out_buf;
    INT32 rank = params->dim_rank;
    aoclfftz_dim_t_64_ *dims = params->dims;
    INTP n = calculate_size(params->dims, params->dim_rank);
    INTP batches = calculate_size(params->vecs, params->vec_rank);
    UINT32 is_align = params->aligned_alloc;

    INTP *in_counter = NULL;
    ALLOC_INIT(in_counter, INTP, rank * sizeof(INTP), is_align);
    INTP *out_counter = NULL;
    ALLOC_INIT(out_counter, INTP, rank * sizeof(INTP), is_align);

    // iterate over the total batches
    for (INTP b = 0; b < batches; b++)
    {
        RESET_ND_COUNTER(out_counter, rank);
        // iterate over output points
        for (INTP k = 0; k < n; k++)
        {
            INTP out_idx = out_idx_map[b * n + k] * T_DATA_STRIDE;
            RESET_ND_COUNTER(in_counter, rank);
            // iterate over input points
            for (INTP i = 0; i < n; i++)
            {
                INTP in_idx = in_idx_map[b * n + i] * T_DATA_STRIDE;
                DOUBLE angle = sign * BENCH_2_PI;
                // angle = angle * [(i0+k0)/n0 + (i0+k0)/n0 + ... + (iR+kR)/nR]
                UPDATE_ANGLE(angle, in_counter, out_counter, dims, rank);
                e[0] = cos(angle);
                e[1] = sin(angle);

                mul_buf[0] = (in_d[in_idx] * e[0]) - (in_d[in_idx + 1] * e[1]);
                mul_buf[1] = (in_d[in_idx] * e[1]) + (in_d[in_idx + 1] * e[0]);

                out_d[out_idx] = out_d[out_idx] + mul_buf[0];
                out_d[out_idx + 1] = out_d[out_idx + 1] + mul_buf[1];

                INCREMENT_ND_COUNTER(in_counter, dims, rank);
            }
            INCREMENT_ND_COUNTER(out_counter, dims, rank);
        }
    }
    FREE_ALLOCATED_MEM(in_counter, is_align);
    FREE_ALLOCATED_MEM(out_counter, is_align);
}
#endif
