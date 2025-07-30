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
 *  The accuracy tests verify fundamental DFT properties:
 *  1. Linearity: DFT(a*x + b*Y) = a*DFT(x) + b*DFT(y)
 *  2. Impulse Response: DFT(δ(x)) = 1 for all x, IDFT(DFT(x)) = x
 *  3. Timeshift: DFT(x(n-m)) = DFT(x(n)) * e^(-j2πkm/N)
 *
 *  @author S. Biplab Raut
 *  @author V. Murugan
 *  @author Srirammaswamy Srinivasan
 */

#include <time.h>
#include "test/bench/accuracy.h"
#include "api/aoclfftz_internal.h"
#include "test/bench/aoclfftz_bench.h"
#include "test/bench/utils/bench_utils.h"
#include "test/bench/utils/data_conversion.h"
#include "test/bench/utils/size_and_index_mapper.h"

/**
 * @brief Verifies DFT linearity property: DFT(a*x + b*y) = a*DFT(x) + b*DFT(y)
 *
 * Tests the fundamental linearity property of the Discrete Fourier Transform.
 * For two input signals x[n] and y[n] and complex constants a, b:
 * DFT{a*x[n] + b*y[n]} should equal a*DFT{x[n]} + b*DFT{y[n]}
 *
 * @param params bench params object containing test configuration
 * @param in_idx_map input index mapping for strided access
 * @param out_idx_map output index mapping for strided access
 * @param handle FFT plan handle
 * @param input_buffer optional pre-allocated input buffer
 * @return INT32 BENCH_SUCCESS if linearity property holds, error code otherwise
 */
INT32 run_linearity_test(aoclfftz_bench_params_t *params, INTP *in_idx_map,
                         INTP *out_idx_map, VOID *handle, VOID *input_buffer)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "ENTER");
#endif
    INT32 status = BENCH_SUCCESS;
    INT32 ret = AOCLFFTZ_SUCCESS;
    UINT32 is_align = params->aligned_alloc;
    INTP input_bytes = params->sz_info.input_bytes;
    INTP output_bytes = params->sz_info.output_bytes;

    // Buffer size adjustment for R2C/C2R in-place transforms
    // In-place requires max(input_size, output_size) buffer allocation
    // TODO: Check and remove this MAX logic if possible
    if (params->fft_type != C2C && params->res_placement == IN_PLACE)
    {
        input_bytes = MAX(input_bytes, output_bytes);
        output_bytes = input_bytes;
    }

    VOID *constants, *in1, *in2, *out1, *out2, *out_combined;
    constants = in1 = in2 = out1 = out2 = out_combined = NULL;

    // Allocate buffers for linearity test:
    // constants: stores complex constants a, b
    // in1, in2: input signals x[n], y[n]
    // out1, out2: DFT(x), DFT(y)
    // out_combined: DFT(a*x + b*y)
    ALLOC_INIT(constants, VOID, 4 * params->sz_info.dt_bytes, is_align);
    ALLOC_INIT(in1, VOID, input_bytes, is_align);
    ALLOC_INIT(in2, VOID, input_bytes, is_align);
    if (constants == NULL || in1 == NULL || in2 == NULL)
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

    if (params->use_random_seed)
    {
        srand(time(0));
    }

    // Run linearity test over multiple iterations with different random data
    for (INT32 i = 0; i < params->num_iterations; i++)
    {
        if (params->use_random_seed)
        {
            params->seed = rand();
        }
        srand(params->seed);
#ifdef AOCL_ENABLE_LOG
        AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode,
                               "Iteration: %d, Seed: %d", i, params->seed);
#endif

        // Generate random input signals x[n] and y[n]
        params->prepare_input_data(
            in2, params->sz_info.n_in * params->sz_info.batches, NULL,
            RANDOM_INPUT, params->sz_info.in_data_stride);

        if (input_buffer == NULL)
        {
            params->prepare_input_data(
                in1, params->sz_info.n_in * params->sz_info.batches, NULL,
                RANDOM_INPUT, params->sz_info.in_data_stride);
        }
        else
        {
            memcpy(in1, input_buffer, input_bytes);
        }

        // Compute DFT(x[n]) → out1
        memcpy(params->in, in1, input_bytes);
        ret = aoclfftz_execute(handle);
        if (ret != AOCLFFTZ_SUCCESS)
        {
            status = EXECUTION_FAILURE;
            goto exit_linearity_test;
        }
        memcpy(out1, params->out, output_bytes);

        // Compute DFT(y[n]) → out2
        memcpy(params->in, in2, input_bytes);
        ret = aoclfftz_execute(handle);
        if (ret != AOCLFFTZ_SUCCESS)
        {
            status = EXECUTION_FAILURE;
            goto exit_linearity_test;
        }
        memcpy(out2, params->out, output_bytes);

        // Create linear combination: in1 = a*x[n] + b*y[n] using random
        // constants
        PREPARE_LINEAR_TEST_INPUTS(in1, in2, in1, params->sz_info.input_size,
                                   constants, params->precision,
                                   params->sz_info.in_data_stride);

        // Compute DFT(a*x[n] * b*y[n]) → out_combined
        memcpy(params->in, in1, input_bytes);
        ret = aoclfftz_execute(handle);
        if (ret != AOCLFFTZ_SUCCESS)
        {
            status = EXECUTION_FAILURE;
            goto exit_linearity_test;
        }
        memcpy(out_combined, params->out, output_bytes);

        // Create expected result: out1 = a*DFT(x) + b*DFT(y)
        PREPARE_LINEAR_TEST_OUTPUTS(
            out1, out2, out1, params->sz_info.output_size, constants,
            params->precision, params->sz_info.out_data_stride);

        // Verify linearity: DFT(a*x + b*y) ?= a*DFT(x) + b*DFT(y)
        status =
            params->compare(params, out1, out_combined, params->sz_info.batches,
                            params->sz_info.n_out, out_idx_map, out_idx_map,
                            params->sz_info.out_data_stride);

        if (status != BENCH_SUCCESS)
        {
            printf("\nResults mismatch on accuracy mode => property: "
                   "linearity, iteration: %d/%d, seed: %d\n",
                   i, params->num_iterations, params->seed);
            goto exit_linearity_test;
        }
    }

exit_linearity_test:
    // Cleanup allocated memory
    FREE_ALLOCATED_MEM(in1, is_align);
    FREE_ALLOCATED_MEM(in2, is_align);
    FREE_ALLOCATED_MEM(out1, is_align);
    FREE_ALLOCATED_MEM(out2, is_align);
    FREE_ALLOCATED_MEM(out_combined, is_align);
    FREE_ALLOCATED_MEM(constants, is_align);
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "EXIT");
#endif
    return status;
}

/**
 * @brief Verifies DFT impulse response and reversibility: IDFT(DFT(x)) = x
 *
 * Tests the fundamental transformation property by verifying that:
 * 1. Impulse input is applied at random position
 * 2. Forward transform followed by inverse transform recovers original signal
 * 3. Proper scaling factor (1/N) is applied for IDFT normalization
 *
 * @param params bench params object containing test configuration
 * @param in_idx_map input index mapping for strided access
 * @param out_idx_map output index mapping for strided access
 * @param handle FFT plan handle
 * @param input_buffer optional pre-allocated input buffer
 * @return INT32 BENCH_SUCCESS if transformation property holds, error code otherwise
 */
INT32 run_impulse_transform_test(aoclfftz_bench_params_t *params,
                                 INTP *in_idx_map, INTP *out_idx_map,
                                 VOID *handle, VOID *input_buffer)
{
    INT32 status = BENCH_SUCCESS;
    INT32 ret = AOCLFFTZ_SUCCESS;
    UINT32 is_align = params->aligned_alloc;
    INTP input_bytes = params->sz_info.input_bytes;
    INTP output_bytes = params->sz_info.output_bytes;

    // Buffer size adjustment for R2C/C2R in-place operations
    // TODO: Check and remove this MAX logic if possible
    if (params->fft_type != C2C && params->res_placement == IN_PLACE)
    {
        input_bytes = MAX(input_bytes, output_bytes);
        output_bytes = input_bytes;
    }

    // Create reverse transform parameters for IDFT operation
    aoclfftz_bench_params_t *params_reverse = NULL;
    VOID *in, *handle_reverse;
    handle_reverse = in = NULL;

    ALLOC_AND_COPY_PARAMS(params_reverse, params);
    if (params_reverse == NULL)
    {
        AOCLFFTZ_LOG_UNFORMATTED(ERR, params->logger_mode,
                                 "params_reverse creation failed");
        status = MEMORY_FAILURE;
        goto exit_impulse_transform_test;
    }

    // Allocate input buffer for impulse signal generation
    ALLOC_INIT(in, VOID, input_bytes, is_align);
    if (in == NULL)
    {
        AOCLFFTZ_LOG_UNFORMATTED(ERR, params->logger_mode,
                                 "input buffer creation failed");
        status = MEMORY_FAILURE;
        goto exit_impulse_transform_test;
    }

    // Configure reverse transform based on original transform type:
    // R2C ↔ C2R: swap transform types
    // C2C: swap FORWARD ↔ BACKWARD directions
    if (params->fft_type == R2C)
    {
        params_reverse->fft_type = C2R;
        params_reverse->dir = params->dir;
    }
    else if (params->fft_type == C2R)
    {
        params_reverse->fft_type = R2C;
        params_reverse->dir = params->dir;
    }
    else if (params->fft_type == C2C && params->dir == FORWARD)
    {
        params_reverse->dir = BACKWARD;
    }
    else // C2C BACKWARD
    {
        params_reverse->dir = FORWARD;
    }

    // Allocate buffers for reverse transform
    // (output of forward becomes input of reverse)
    ALLOC_INIT(params_reverse->in, VOID, output_bytes, is_align);
    if (params_reverse->in == NULL)
    {
        AOCLFFTZ_LOG_UNFORMATTED(ERR, params->logger_mode,
                                 "input buffer creation failed");
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
            AOCLFFTZ_LOG_UNFORMATTED(ERR, params->logger_mode,
                                     "output buffer creation failed");
            status = MEMORY_FAILURE;
            goto exit_impulse_transform_test;
        }
    }

    // Swap stride configuration for reverse transform to match data layout
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

    // Setup reverse FFT plan
    handle_reverse = params->setup_problem(params_reverse);
    if (handle_reverse == NULL)
    {
        status = SETUP_FAILURE;
        goto exit_impulse_transform_test;
    }

    if (params->use_random_seed)
    {
        srand(time(0));
    }

    // Test impulse response over multiple iterations
    for (INT32 i = 0; i < params->num_iterations; i++)
    {
        if (params->use_random_seed)
        {
            params->seed = rand();
        }
        srand(params->seed);
#ifdef AOCL_ENABLE_LOG
        AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode,
                               "Iteration: %d, Seed: %d", i, params->seed);
#endif

        // Generate impulse signal δ(n) at random position
        if (input_buffer == NULL)
        {
            params->prepare_input_data(
                in, params->sz_info.n_in * params->sz_info.batches, in_idx_map,
                IMPULSE_INPUT, params->sz_info.in_data_stride);
        }
        else
        {
            memcpy(in, input_buffer, input_bytes);
        }

        // For C2R transforms, ensure DC and Nyquist components are real-valued
        if (params->fft_type == C2R)
        {
            set_zero_for_dc_and_nyquist_nd(
                in, params->sz_info.n, params->dims[0].n,
                params->sz_info.batches, in_idx_map, params->precision);
        }

        // Compute forward transform: x[n] → X[k]
        memcpy(params->in, in, input_bytes);
        ret = aoclfftz_execute(handle);
        if (ret != AOCLFFTZ_SUCCESS)
        {
            status = EXECUTION_FAILURE;
            goto exit_impulse_transform_test;
        }

        // Compute reverse transform: X[k] → x_recovered[n]
        memcpy(params_reverse->in, params->out, output_bytes);
        ret = aoclfftz_execute(handle_reverse);
        if (ret != AOCLFFTZ_SUCCESS)
        {
            status = EXECUTION_FAILURE;
            goto exit_impulse_transform_test;
        }

        // Apply IDFT normalization factor (1/N) to recover original amplitude
        NORMALIZE_IFFT_DATA(params_reverse->out, params->sz_info.input_size,
                            params->sz_info.n, params->precision,
                            params->sz_info.in_data_stride);

        // Verify x_recovered[n] == x[n] within numerical precision
        status = params->compare(params, in, params_reverse->out,
                                 params->sz_info.batches, params->sz_info.n_in,
                                 in_idx_map, in_idx_map,
                                 params->sz_info.in_data_stride);
        if (status != BENCH_SUCCESS)
        {
            printf("\nResults mismatch on accuracy mode => property: "
                   "transformation, iteration: %d/%d, seed: %d\n",
                   i, params->num_iterations, params->seed);
            goto exit_impulse_transform_test;
        }
    }

exit_impulse_transform_test:
    // Cleanup resources
    FREE_ALLOCATED_MEM(in, is_align);
    aoclfftz_destroy(handle_reverse);
    destroy_bench_param(params_reverse);
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "EXIT");
#endif
    return status;
}

/**
 * @brief Macro for time-domain shifting: x(n-m) for input signals
 *
 * Applies circular time shift to input data for timeshift property verification.
 * For ND transforms, shifts are applied per dimension maintaining data structure.
 */
#define TIME_SHIFT(in, out, idx_map, data_stride)                              \
    do                                                                         \
    {                                                                          \
        for (INTP b = 0; b < batches; b++)                                     \
        {                                                                      \
            /* Apply time shift per outer dimension                       */   \
            /* Example: 3x4x5 ND problem, dim=2 processing:               */   \
            /* shifts input m times in each 4x5 matrix, repeated 3 times  */   \
            if (idx_map == NULL)                                               \
            {                                                                  \
                for (INTP o = 0; o < outer_n; o++)                             \
                {                                                              \
                    PREPARE_TIMESHIFT_TEST_INPUTS(                             \
                        in + (b * n + o * inner_n) * data_stride,              \
                        out + (b * n + o * inner_n) * data_stride, inner_n,    \
                        shifts, NULL, params->precision, data_stride);         \
                }                                                              \
            }                                                                  \
            else                                                               \
            {                                                                  \
                INTP *idx_map_t = (INTP *)idx_map;                             \
                for (INTP o = 0; o < outer_n; o++)                             \
                {                                                              \
                    PREPARE_TIMESHIFT_TEST_INPUTS(                             \
                        in + idx_map_t[b * n + o * inner_n] * data_stride,     \
                        out + idx_map_t[b * n + o * inner_n] * data_stride,    \
                        inner_n, shifts, idx_map_t, params->precision,         \
                        data_stride);                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    } while (0)

/**
 * @brief Macro for frequency-domain phase shift: X[k] * e^(-j2πkm/N)
 *
 * Applies phase shift in frequency domain equivalent to time domain shift.
 * Phase shift formula: φ = -2πkm/N where k=frequency bin, m=shift amount, N=size
 */
#define PHASE_SHIFT(in, out, idx_map, data_stride)                             \
    do                                                                         \
    {                                                                          \
        for (INTP b = 0; b < batches; b++)                                     \
        {                                                                      \
            if (idx_map == NULL)                                               \
            {                                                                  \
                for (INTP o = 0; o < outer_n; o++)                             \
                {                                                              \
                    PREPARE_TIMESHIFT_TEST_OUTPUTS(                            \
                        in + (b * n0_in + o * inner_n) * data_stride,          \
                        out + (b * n0_in + o * inner_n) * data_stride, cur_n,  \
                        m, unit_m, idx_map, params->dir, params->precision,    \
                        DATA_STRIDE);                                          \
                }                                                              \
            }                                                                  \
            else                                                               \
            {                                                                  \
                INTP *idx_map_t = (INTP *)idx_map;                             \
                for (INTP o = 0; o < outer_n; o++)                             \
                {                                                              \
                    PREPARE_TIMESHIFT_TEST_OUTPUTS(                            \
                        in + idx_map_t[b * n0_in + o * inner_n] * data_stride, \
                        out +                                                  \
                            idx_map_t[b * n0_in + o * inner_n] * data_stride,  \
                        cur_n, m, unit_m, idx_map, params->dir,                \
                        params->precision, DATA_STRIDE);                       \
                }                                                              \
            }                                                                  \
        }                                                                      \
    } while (0)

/**
 * @brief Verifies DFT timeshift property: DFT{x(n-m)} = DFT{x(n)} * e^(-j2πkm/N)
 *
 * Tests the timeshift theorem stating that a shift in time domain corresponds to
 * a linear phase shift in frequency domain. For each dimension d:
 * - Time shift: x(n-m) → circular shift by m samples
 * - Frequency shift: X[k] * exp(-j2πkm/N) → phase rotation
 * Both should produce identical results.
 *
 * @param params bench params object containing test configuration
 * @param in_idx_map input index mapping for strided access
 * @param out_idx_map output index mapping for strided access
 * @param handle FFT plan handle
 * @param input_buffer optional pre-allocated input buffer
 * @return INT32 BENCH_SUCCESS if timeshift property holds, error code otherwise
 */
INT32 run_timeshift_test(aoclfftz_bench_params_t *params, INTP *in_idx_map,
                         INTP *out_idx_map, VOID *handle, VOID *input_buffer)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "ENTER");
#endif
    INT32 status = BENCH_SUCCESS;
    INT32 ret = AOCLFFTZ_SUCCESS;
    INTP n = params->sz_info.n;
    INTP batches = params->sz_info.batches;
    UINT32 is_align = params->aligned_alloc;

    VOID *in1, *in2, *out1, *out2, *out1_fc, *out2_fc, *out_temp;
    in1 = in2 = out1 = out2 = out_temp = out1_fc = out2_fc = NULL;
    INTP input_bytes = params->sz_info.input_bytes;
    INTP output_bytes = params->sz_info.output_bytes;
    INTP full_complex_bytes =
        n * batches * DATA_STRIDE * params->sz_info.dt_bytes;

    // Buffer size adjustment for R2C/C2R in-place operations
    // TODO: Check and remove this MAX logic if possible
    if (params->fft_type != C2C && params->res_placement == IN_PLACE)
    {
        input_bytes = MAX(input_bytes, output_bytes);
        output_bytes = input_bytes;
    }

    // Allocate buffers for timeshift test:
    // in1, in2: original and time-shifted input signals
    // out1, out2: DFT outputs from both signals
    // out1_fc, out2_fc: full complex representations for comparison
    ALLOC_UNINIT(in1, VOID, input_bytes, is_align);
    ALLOC_UNINIT(in2, VOID, input_bytes, is_align);
    if (in1 == NULL || in2 == NULL)
    {
        AOCLFFTZ_LOG_UNFORMATTED(ERR, params->logger_mode,
                                 "input buffer creation failed");
        status = MEMORY_FAILURE;
        goto exit_timeshift_test;
    }
    ALLOC_INIT(out1, VOID, output_bytes, is_align);
    ALLOC_INIT(out2, VOID, output_bytes, is_align);
    ALLOC_INIT(out1_fc, VOID, full_complex_bytes, is_align);
    ALLOC_INIT(out2_fc, VOID, full_complex_bytes, is_align);
    if (out1 == NULL || out2 == NULL || out1_fc == NULL || out2_fc == NULL)
    {
        AOCLFFTZ_LOG_UNFORMATTED(ERR, params->logger_mode,
                                 "output buffer creation failed");
        status = MEMORY_FAILURE;
        goto exit_timeshift_test;
    }

    if (params->use_random_seed)
    {
        srand(time(0));
    }

    // Test timeshift property over multiple iterations
    for (INT32 i = 0; i < params->num_iterations; i++)
    {
        if (params->use_random_seed)
        {
            params->seed = rand();
        }
        srand(params->seed);
#ifdef AOCL_ENABLE_LOG
        AOCLFFTZ_LOG_FORMATTED(INFO, params->logger_mode,
                               "Iteration: %d, Seed: %d", i, params->seed);
#endif

        // Generate random input signal
        if (input_buffer == NULL)
        {
            params->prepare_input_data(in1, params->sz_info.n_in * batches,
                                       in_idx_map, RANDOM_INPUT,
                                       params->sz_info.in_data_stride);
        }
        else
        {
            memcpy(in1, input_buffer, input_bytes);
        }

        // Initialize dimension parameters for ND timeshift testing
        INTP cur_n;
        INTP outer_n = 1;
        INTP inner_n = n;
        INTP unit_m = n;
        INTP n0 = params->dims[0].n;
        INTP n0_in = params->fft_type == C2R ? n0 / 2 + 1 : n0;

        // Test timeshift property for each dimension (outermost to innermost)
        for (INTP d = params->dim_rank - 1; d >= 0; d--)
        {
            cur_n = params->dims[d].n;
            // Adjust dimension size for C2R half-complex format
            if (d == 0 && params->fft_type == C2R)
            {
                cur_n = cur_n / 2 + 1;
            }
            unit_m /= cur_n;

            // Generate random shift amount m ∈ [0, cur_n)
            INTP m = rand() % cur_n;

            // Apply timeshift based on transform direction:
            // Forward/R2C: shift in time domain, verify in frequency domain
            // Backward/C2R: shift in frequency domain, verify in time domain
            if (params->fft_type == R2C ||
                (params->fft_type == C2C && params->dir == FORWARD))
            {
                // Time domain shift: x(n-m) - circular shift by m samples
                // Linear shift amount for ND: m * unit_m
                // Example: 3x4x5 ND, dim=2, m=2 → shift = 2*20 = 40 elements
                INTP shifts = m * unit_m;
                TIME_SHIFT(in1, in2, in_idx_map,
                           params->sz_info.in_data_stride);
            }
            else
            {
                // Frequency domain phase shift: X[k] * e^(-j2πkm/N)
                PHASE_SHIFT(in1, in2, in_idx_map,
                            params->sz_info.in_data_stride);
            }

            // Compute DFT of original signal
            memcpy(params->in, in1, input_bytes);
            ret = aoclfftz_execute(handle);
            if (ret != AOCLFFTZ_SUCCESS)
            {
                status = EXECUTION_FAILURE;
                goto exit_timeshift_test;
            }
            memcpy(out1, params->out, output_bytes);

            // Compute DFT of shifted signal
            memcpy(params->in, in2, input_bytes);
            status |= aoclfftz_execute(handle);
            memcpy(out2, params->out, output_bytes);

            // Convert R2C half-complex output to full complex for comparison
            // if (params->fft_type == R2C && params->dir == FORWARD)
            if (params->fft_type == R2C)
            {
                convert_half_complex_to_complex(out1_fc, out1, n, batches,
                                                out_idx_map, params->precision);
                convert_half_complex_to_complex(out2_fc, out2, n, batches,
                                                out_idx_map, params->precision);
            }

            // Apply corresponding transformation to match shifted result
            if (params->fft_type == R2C || params->dir == FORWARD)
            {
                // Apply frequency domain phase shift to match time domain shift
                PHASE_SHIFT(out1_fc, out1_fc, NULL,
                            params->sz_info.out_data_stride);
            }
            else
            {
                // Apply time domain shift to match frequency domain phase shift
                ALLOC_INIT(out_temp, VOID, full_complex_bytes, is_align);
                if (out_temp == NULL)
                {
                    AOCLFFTZ_LOG_UNFORMATTED(ERR, ERR,
                                             "output buffer creation failed");
                    status = MEMORY_FAILURE;
                    goto exit_timeshift_test;
                }
                INTP shifts = m * unit_m;
                TIME_SHIFT(out1_fc, out_temp, NULL,
                           params->sz_info.out_data_stride);
                FREE_ALLOCATED_MEM(out1_fc, is_align);
                out1_fc = out_temp;
            }

            // Update dimension traversal parameters
            inner_n /= cur_n;
            outer_n *= cur_n;

            // Verify timeshift property: both paths should yield identical results
            status = params->compare(params, out1_fc, out2_fc, batches,
                                     params->sz_info.n_out, NULL, NULL,
                                     params->sz_info.out_data_stride);
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
    // Cleanup allocated buffers
    FREE_ALLOCATED_MEM(in1, is_align);
    FREE_ALLOCATED_MEM(in2, is_align);
    FREE_ALLOCATED_MEM(out1, is_align);
    FREE_ALLOCATED_MEM(out2, is_align);
    FREE_ALLOCATED_MEM(out1_fc, is_align);
    FREE_ALLOCATED_MEM(out2_fc, is_align);
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "EXIT");
#endif
    return status;
}

/**
 * @brief Main accuracy test coordinator - verifies fundamental DFT properties
 *
 * Orchestrates comprehensive accuracy testing by running three core DFT property tests:
 * 1. Linearity: DFT(a*x + b*y) = a*DFT(x) + b*DFT(y)
 * 2. Impulse Response: IDFT(DFT(x)) = x (reversibility)
 * 3. Timeshift: DFT{x(n-m)} = DFT{x(n)} * e^(-j2πkm/N)
 *
 * These properties are fundamental to any correct DFT implementation and provide
 * robust validation across all transform types (C2C, R2C, C2R) and dimensions.
 *
 * @param params bench params object containing complete test configuration
 * @return INT32 BENCH_SUCCESS if all properties verified, error code otherwise
 */
INT32 run_bench_on_accuracy_mode(aoclfftz_bench_params_t *params)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "ENTER");
#endif
    INT32 status = BENCH_SUCCESS;
    VOID *handle = NULL;

    UINT32 is_align = params->aligned_alloc;

    // Calculate transform dimensions and batch sizes
    params->sz_info.n = calculate_size(params->dims, params->dim_rank);
    params->sz_info.batches = calculate_size(params->vecs, params->vec_rank);

    // Initialize input/output sizes for different transform types
    params->sz_info.n_in = params->sz_info.n;
    params->sz_info.n_out = params->sz_info.n;
    INTP n0 = params->dims[0].n;
    INTP n0_hc = n0 / 2 + 1; // half-complex size for R2C/C2R

    // Adjust sizes for half-complex transforms:
    // R2C: N real → (N/2+1) complex (Hermitian symmetry reduces storage)
    // C2R: (N/2+1) complex → N real
    if (params->fft_type == R2C)
    {
        params->sz_info.n_out = (params->sz_info.n * n0_hc) / n0;
    }
    else if (params->fft_type == C2R)
    {
        params->sz_info.n_in = (params->sz_info.n * n0_hc) / n0;
    }

    // Allocate index mapping arrays for strided data access
    INTP *in_idx_map = NULL;
    ALLOC_UNINIT(in_idx_map, INTP,
                 params->sz_info.n * params->sz_info.batches * sizeof(INTP),
                 is_align);
    INTP *out_idx_map = NULL;
    ALLOC_UNINIT(out_idx_map, INTP,
                 params->sz_info.n * params->sz_info.batches * sizeof(INTP),
                 is_align);
    if (in_idx_map == NULL || out_idx_map == NULL)
    {
        AOCLFFTZ_LOG_UNFORMATTED(ERR, params->logger_mode,
                                 "idx_map creation failed");
        status = MEMORY_FAILURE;
        goto exit_accuracy_mode;
    }

    // Generate index mappings for strided access patterns
    // Strided access is required for problem->in and problem->out buffers:
    //
    // problem->in can be used in the following places which requires strided
    // access using in_idx_map:
    //   * preparing input data
    //   * comparing IFFT output with input
    //   * performing time-shift / phase-shift on input
    //
    // problem->out can be used in the following places which requires strided
    // access using out_idx_map:
    //   * comparing outputs for different properties
    //   * performing time-shift / phase-shift on output
    //
    // other internal buffers which are created for full-complex conversion,
    // DFT reference computation, etc. do not require strided access.
    // In those cases, NULL is passed as in/out index map
    //
    prepare_index_map(params->dim_rank, params->vec_rank, params->dims,
                      params->vecs, in_idx_map, out_idx_map, params->fft_type,
                      is_align);

    // Initialize FFT plan for testing
    handle = params->setup_problem(params);
    if (handle == NULL)
    {
        status = SETUP_FAILURE;
        HANDLE_BENCH_STATUS(status);
        goto exit_accuracy_mode;
    }

#ifdef ENABLE_DFT_REFERENCE
    // Optional: Compare against reference DFT implementation
    status =
        run_dft_reference_test(params, in_idx_map, out_idx_map, handle, NULL);
    HANDLE_BENCH_STATUS(status);
    if (status != BENCH_SUCCESS)
    {
        goto exit_accuracy_mode;
    }
#endif

    // Execute core DFT property verification tests

    // Test 1: Linearity Property - DFT(a*x + b*y) = a*DFT(x) + b*DFT(y)
    status = run_linearity_test(params, in_idx_map, out_idx_map, handle, NULL);
    HANDLE_BENCH_STATUS(status);
    if (status != BENCH_SUCCESS)
    {
        goto exit_accuracy_mode;
    }

    // Test 2: Transformation Property - IDFT(DFT(x)) = x
    status = run_impulse_transform_test(params, in_idx_map, out_idx_map, handle,
                                        NULL);
    HANDLE_BENCH_STATUS(status);
    if (status != BENCH_SUCCESS)
    {
        goto exit_accuracy_mode;
    }

    // Test 3: Timeshift Property - DFT{x(n-m)} = DFT{x(n)} * e^(-j2πkm/N)
    status = run_timeshift_test(params, in_idx_map, out_idx_map, handle, NULL);
    HANDLE_BENCH_STATUS(status);
    if (status != BENCH_SUCCESS)
    {
        goto exit_accuracy_mode;
    }

    PRINT_SUCCESS("\nTest bench completed on accuracy mode\n\n");

exit_accuracy_mode:
    // Cleanup resources
    FREE_ALLOCATED_MEM(in_idx_map, is_align);
    FREE_ALLOCATED_MEM(out_idx_map, is_align);
    aoclfftz_destroy(handle);
    return status;
}
