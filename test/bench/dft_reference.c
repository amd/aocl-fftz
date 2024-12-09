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
#include "test/bench/dft_reference.h"
#include "test/bench/utils/compare.h"
#include "test/bench/utils/data_conversion.h"
#include "test/bench/utils/size_and_index_mapper.h"
#include "test/bench/utils/bench_utils.h"
#include "test/bench/utils/size_and_index_mapper.h"

#ifdef ENABLE_DFT_REFERENCE

#include "utils/complex_utils.h"

/**
 * @brief run the FFT execute api and compare the output with DFT reference
 * output
 *
 * DFT reference execution flow for different FFT types:
 *
 * 1. C2C:
 *
 *    library:
 *      [in-place]     params->out = params->in (pointer reference)
 *      [out-of-place] params->out = FFT(params->in)
 *    dft-ref:
 *      in_ref  = params->in (copy)
 *      out_ref = DFT_REF(in_ref)
 *    compare:
 *      compare(params->out, out_ref)
 *
 * 2. R2C a.k.a. R2C FORWARD:
 *
 *    library:
 *      [in-place]     params->out = params->in (pointer reference)
 *      [out-of-place] params->out = copy(params->in) (mem-copy)
 *                     params->out = FFT(params->in)
 *    dft-ref:
 *      complex_in  = convert_real_to_complex(params->in)
 *      in_ref      = complex_in (pointer reference)
 *      out_ref     = DFT_REF(in_ref)
 *    compare:
 *      complex_out = convert_half_complex_to_complex(params->out)
 *      compare(complex_out, out_ref)
 *
 * 3. C2R a.k.a. R2C BACKWARD:
 *
 *    library:
 *      [in-place]     params->out = params->in (pointer reference)
 *      [out-of-place] params->out = copy(params->in) (mem-copy)
 *                     params->in  = convert_complex_to_half_complex(params->in)
 *                     params->out = FFT(params->in)
 *    dft-ref:
 *        complex_in = copy(params->in)
 *        in_ref     = complex_in (pointer reference)
 *        out_ref    = DFT_REF(in_ref)
 *    compare:
 *        complex_out = convert_real_to_complex(params->out)
 *        compare(complex_out, out_ref)
 *
 * @param params bench params object
 * @param in_idx_map index map for input
 * @param out_idx_map index map for output
 * @param handle problem handle
 * @param input_buffer input data buffer
 * @return INT32
 */
INT32 run_dft_reference_test(aoclfftz_bench_params_t *params, INTP *in_idx_map,
                             INTP *out_idx_map, VOID *handle,
                             VOID *input_buffer)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "ENTER");
#endif
    INT32 status = BENCH_SUCCESS;
    INT32 ret = AOCLFFTZ_SUCCESS;
    INT32 dt_bytes = (params->precision == FLOAT_P) ?
                     sizeof(FLOAT) : sizeof(DOUBLE);
    UINT32 is_align = params->aligned_alloc;
    INTP n = params->sz_info.n;
    INTP batches = params->sz_info.batches;
    // DFT reference mostly deals with complex buffers even for real problems
    // so using the data stride value as 2 always
    INTP complex_input_bytes = params->sz_info.input_size * 2 * dt_bytes;
    INTP complex_output_bytes = params->sz_info.output_size * 2 * dt_bytes;

    // initialize the random seed value based on current time
    if (params->use_random_seed)
    {
        srand(time(0));
    }

    // local buffers
    VOID *complex_in = NULL;
    VOID *complex_out = NULL;
    VOID *in_ref = NULL;
    VOID *out_ref = NULL;

    if (params->fft_type == R2C)
    {
        ALLOC_INIT(complex_in, VOID, complex_input_bytes, is_align);
        if (complex_in == NULL)
        {
            AOCLFFTZ_LOG_UNFORMATTED(ERR, ERR, "input buffer creation failed");
            status = MEMORY_FAILURE;
            goto exit_dft_reference_test;
        }

        ALLOC_INIT(complex_out, VOID, complex_output_bytes, is_align);
        if (complex_out == NULL)
        {
            AOCLFFTZ_LOG_UNFORMATTED(ERR, ERR, "output buffer creation failed");
            status = MEMORY_FAILURE;
            goto exit_dft_reference_test;
        }
    }

    if (params->fft_type == R2C)
    {
        in_ref = complex_in;
    }
    else if (params->res_placement == IN_PLACE)
    {
        ALLOC_INIT(in_ref, VOID, complex_input_bytes, is_align);
        if (in_ref == NULL)
        {
            AOCLFFTZ_LOG_UNFORMATTED(ERR, ERR, "input buffer creation failed");
            status = MEMORY_FAILURE;
            goto exit_dft_reference_test;
        }
    }
    else
    {
        in_ref = params->in;
    }

    ALLOC_INIT(out_ref, VOID, complex_output_bytes, is_align);
    if (out_ref == NULL)
    {
        AOCLFFTZ_LOG_UNFORMATTED(ERR, ERR, "output buffer creation failed");
        status = MEMORY_FAILURE;
        goto exit_dft_reference_test;
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
        params->prepare_input_data(params->in, n * batches, in_idx_map,
                                   RANDOM_INPUT, params->in_data_stride);

        if (params->fft_type == R2C && params->dir == FORWARD)
        {
            // Make the DFT-reference input from real to complex
            memset(complex_in, 0, complex_input_bytes);
            convert_real_to_complex(complex_in, params->in, n, batches,
                                    in_idx_map, params->precision);

            // Library execution
            ret = aoclfftz_execute(handle);
            if (ret != AOCLFFTZ_SUCCESS)
            {
                status = EXECUTION_FAILURE;
                goto exit_dft_reference_test;
            }

            // Make the library output from half-complex to complex
            memset(complex_out, 0, complex_output_bytes);
            convert_half_complex_to_complex(complex_out, params->out, n,
                                            batches, out_idx_map,
                                            params->precision);
        }
        else if (params->fft_type == R2C && params->dir == BACKWARD)
        {
            // Make the library input from complex to half-complex
            // TODO: do it in out-of-place if needed
            convert_complex_to_half_complex(params->in, n, batches,
                                            in_idx_map, params->precision);
            memcpy(complex_in, params->in, complex_input_bytes);

            // Library execution
            ret = aoclfftz_execute(handle);
            if (ret != AOCLFFTZ_SUCCESS)
            {
                status = EXECUTION_FAILURE;
                goto exit_dft_reference_test;
            }

            // Make the library output from real to complex
            convert_real_to_complex(complex_out, params->out, n, batches,
                                    out_idx_map, params->precision);
        }
        else // params->fft_type == C2C
        {
            if (params->res_placement == IN_PLACE)
            {
                memcpy(in_ref, params->in, complex_input_bytes);
            }
            // Library execution
            ret = aoclfftz_execute(handle);
            if (ret != AOCLFFTZ_SUCCESS)
            {
                status = EXECUTION_FAILURE;
                goto exit_dft_reference_test;
            }

            complex_out = params->out;
        }

        // get the DFT reference output
        params->dft_ref(params, in_ref, out_ref, in_idx_map, out_idx_map);

        // compare the FFT output with DFT reference output
        status = params->compare(params, out_ref, complex_out, batches,
                                         n, out_idx_map, 2);
        if (status != AOCLFFTZ_SUCCESS)
        {
            printf("\nResults mismatch on accuracy mode => DFT reference, "
                   "iteration: %d/%d, seed: %d\n",
                   i, params->num_iterations, params->seed);
            goto exit_dft_reference_test;
        }
    }

exit_dft_reference_test:
    // destroy temporary buffers
    FREE_ALLOCATED_MEM(out_ref, is_align);
    if (params->fft_type == C2C &&
        params->res_placement == IN_PLACE)
    {
        FREE_ALLOCATED_MEM(in_ref, is_align);
    }
    if (params->fft_type == R2C)
    {
        FREE_ALLOCATED_MEM(complex_in, is_align);
        FREE_ALLOCATED_MEM(complex_out, is_align);
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "EXIT");
#endif
    return status;
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
 * @param in input data buffer
 * @param out buffer to store the DFT output
 * @param in_idx_map index map input indices
 * @param out_idx_map index map for output indices
 * @return VOID
 */
VOID dft_ref_f(aoclfftz_bench_params_t *params, VOID *in, VOID *out,
               INTP *in_idx_map, INTP *out_idx_map)
{
    INT32 data_stride = 2;
    // intermediate variables are in DOUBLE  data type to improve accuracy
    DOUBLE e[data_stride], mul_buf[data_stride];
    FLOAT sign = params->dir == BACKWARD ? 1.0 : -1.0;
    FLOAT *in_f = (FLOAT *)in;
    FLOAT *out_f = (FLOAT *)out;
    INT32 rank = params->dim_rank;
    aoclfftz_dim_t_64_ *dims = params->dims;
    INTP n = params->sz_info.n;
    INTP batches = params->sz_info.batches;
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
            INTP out_idx = out_idx_map[b * n + k] * data_stride;
            RESET_ND_COUNTER(in_counter, rank);
            // iterate over input points
            for (INTP i = 0; i < n; i++)
            {
                INTP in_idx = in_idx_map[b * n + i] * data_stride;
                DOUBLE angle = sign * BENCH_2_PI;
                // angle = angle * [(i0+k0)/n0 + (i0+k0)/n0 + ... + (iR+kR)/nR]
                UPDATE_ANGLE(angle, in_counter, out_counter, dims, rank);
                e[0] = cos(angle);
                e[1] = sin(angle);

                if (data_stride == 1)
                {
                    mul_buf[0] = (in_f[in_idx] * e[0]);
                    mul_buf[1] = (in_f[in_idx] * e[1]);
                }
                else
                {
                    mul_buf[0] =
                        (in_f[in_idx] * e[0]) - (in_f[in_idx + 1] * e[1]);
                    mul_buf[1] =
                        (in_f[in_idx] * e[1]) + (in_f[in_idx + 1] * e[0]);
                }

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
 * @param in input data buffer
 * @param out buffer to store the DFT output
 * @param in_idx_map index map input indices
 * @param out_idx_map index map for output indices
 * @return VOID
 */
VOID dft_ref_d(aoclfftz_bench_params_t *params, VOID *in, VOID *out,
               INTP *in_idx_map, INTP *out_idx_map)
{
    INT32 data_stride = 2;
    DOUBLE e[data_stride], mul_buf[data_stride];
    DOUBLE sign = params->dir == BACKWARD ? 1.0 : -1.0;
    DOUBLE *in_d = (DOUBLE *)in;
    DOUBLE *out_d = (DOUBLE *)out;
    INT32 rank = params->dim_rank;
    aoclfftz_dim_t_64_ *dims = params->dims;
    INTP n = params->sz_info.n;
    INTP batches = params->sz_info.batches;
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
            INTP out_idx = out_idx_map[b * n + k] * 2;
            RESET_ND_COUNTER(in_counter, rank);
            // iterate over input points
            for (INTP i = 0; i < n; i++)
            {
                INTP in_idx = in_idx_map[b * n + i] * data_stride;
                DOUBLE angle = sign * BENCH_2_PI;
                // angle = angle * [(i0+k0)/n0 + (i0+k0)/n0 + ... + (iR+kR)/nR]
                UPDATE_ANGLE(angle, in_counter, out_counter, dims, rank);
                e[0] = cos(angle);
                e[1] = sin(angle);

                if (data_stride == 1)
                {
                    mul_buf[0] = (in_d[in_idx] * e[0]);
                    mul_buf[1] = (in_d[in_idx] * e[1]);
                }
                else
                {
                    mul_buf[0] =
                        (in_d[in_idx] * e[0]) - (in_d[in_idx + 1] * e[1]);
                    mul_buf[1] =
                        (in_d[in_idx] * e[1]) + (in_d[in_idx + 1] * e[0]);
                }

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
