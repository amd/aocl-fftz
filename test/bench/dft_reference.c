/**
 * Copyright (C) 2024-2025, Advanced Micro Devices. All rights reserved.
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

#include "api/aoclfftz_internal.h"
#include "api/types.h"
#include "test/bench/aoclfftz_bench.h"
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
 * @brief Execute FFT library and compare output with DFT reference
 * implementation
 *
 * This function performs accuracy testing by comparing the FFT library output
 * with a mathematical DFT reference implementation. It handles different FFT
 * configurations and data formats through multiple iterations with random input
 * data.
 *
 * Execution Flow for Different FFT Types:
 *
 * 1. Complex-to-Complex (C2C):
 *    - Library: Executes FFT directly on complex input data
 *    - Reference: Copies input to reference buffer (if in-place), computes DFT
 *    - Comparison: Direct comparison between library and reference outputs
 *
 * 2. Real-to-Complex (R2C) a.k.a. R2C FORWARD:
 *    - Library: Converts real input to complex, executes FFT, produces
 * half-complex output
 *    - Reference: Converts real input to full complex, computes DFT
 *    - Comparison: Converts library's half-complex output to full complex, then
 * compares
 *
 * 3. Complex-to-Real (C2R) a.k.a. R2C BACKWARD:
 *    - Library: Converts complex input to half-complex format, executes FFT,
 * produces real output
 *    - Reference: Uses full complex input, computes DFT
 *    - Comparison: Converts library's real output to complex, then compares
 *
 * Buffer Management:
 * - Allocates temporary buffers for data format conversions (R2C/C2R)
 * - Handles in-place vs out-of-place configurations
 * - Manages complex data representation for reference computation
 *
 * Iteration Process:
 * - Runs multiple iterations with different random seeds
 * - Prepares random input data for each iteration
 * - Performs library execution and reference computation
 * - Compares results and reports accuracy status
 *
 * @param params Benchmark parameters containing FFT configuration, dimensions,
 *               precision, placement mode, and iteration settings
 * @param in_idx_map Index mapping for input data access patterns (can be NULL)
 * @param out_idx_map Index mapping for output data access patterns (can be
 * NULL)
 * @param handle FFT problem handle for library execution
 * @param input_buffer Input data buffer (currently unused parameter)
 * @return INT32 Status code: BENCH_SUCCESS on success, error codes on failure
 *         (MEMORY_FAILURE, EXECUTION_FAILURE, or comparison mismatch)
 */
INT32 run_dft_reference_test(aoclfftz_bench_params_t *params, INTP *in_idx_map,
                             INTP *out_idx_map, VOID *handle,
                             VOID *input_buffer)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "ENTER");
#endif

    // Return codes and configuration
    INT32 status = BENCH_SUCCESS;
    UINT32 is_align = params->aligned_alloc;
    INTP problem_bytes_without_strides =
        params->sz_info.n * params->sz_info.batches * DATA_STRIDE *
        params->sz_info.in_data_stride * params->sz_info.dt_bytes;

    if (params->use_random_seed)
    {
        srand(time(0));
    }

    // Buffer structure for organized memory management
    dft_ref_buffers_t buffers = {.complex_in = NULL,
                                 .complex_out = NULL,
                                 .in_ref = NULL,
                                 .out_ref = NULL};

    // Allocate buffers based on FFT type and placement
    status = allocate_dftref_buffers(params, &buffers, is_align);
    if (status != BENCH_SUCCESS)
    {
        goto cleanup;
    }

    // Main iteration loop
    for (INT32 i = 0; i < params->num_iterations && status == 0; i++)
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

        if (input_buffer == NULL)
        {
            // Prepare random input data for this iteration
            params->prepare_input_data(
                params->in, params->sz_info.n_in * params->sz_info.batches,
                in_idx_map, RANDOM_INPUT, params->sz_info.in_data_stride);
        }
        else
        {
            memcpy(params->in, input_buffer, params->sz_info.input_bytes);
        }

        // Execute FFT and prepare data for comparison (handles C2C, R2C & C2R)
        status = execute_fft_and_postprocess(params, &buffers, handle,
                                             in_idx_map, out_idx_map);
        if (status != BENCH_SUCCESS)
        {
            goto cleanup;
        }

        // Set up index maps for DFT reference computation and comparison
        //
        // Index maps control data access patterns for non-contiguous memory
        // layouts. Different FFT types require different indexing strategies:
        //
        // C2C transforms:
        //   - Use original input/output index maps to maintain stride patterns
        //   - DFT reference needs same input indexing as library
        //   - Comparison needs same output indexing as library
        //
        // R2C/C2R transforms:
        //   - Use NULL (contiguous indexing) for DFT reference and comparison
        //   - Data conversion (real/half-complex to complex) functions handle
        //   stride patterns internally
        //   - Converted buffers (complex_in/complex_out) are always contiguous

        // Input index map for DFT reference computation
        // C2C: preserve original input stride pattern, R2C/C2R: use contiguous
        INTP *dftref_in_map = (params->fft_type == C2C) ? in_idx_map : NULL;
        // Output index map for DFT reference computation
        // Always NULL - DFT reference output is computed in contiguous format
        INTP *dftref_out_map = NULL;
        // Input index map for comparison function
        // Always NULL - comparison input (reference) is in contiguous format
        INTP *cmp_in_map = NULL;
        // Output index map for comparison function
        // C2C: preserve original output stride pattern, R2C/C2R: use contiguous
        INTP *cmp_out_map = (params->fft_type == C2C) ? out_idx_map : NULL;

        // Clear reference output buffer before DFT computation
        memset(buffers.out_ref, 0, problem_bytes_without_strides);

        // Compute DFT reference using the provided function pointer
        params->dft_ref(params, buffers.in_ref, buffers.out_ref,
                        dftref_in_map, dftref_out_map);

        // Compare library output with DFT reference output
        status = params->compare(params, buffers.out_ref, buffers.complex_out,
                                 params->sz_info.batches, params->sz_info.n_out,
                                 cmp_in_map, cmp_out_map,
                                 params->sz_info.out_data_stride);
        if (status != BENCH_SUCCESS)
        {
            printf("\nResults mismatch on accuracy mode => DFT reference, "
                "iteration: %d/%d, seed: %d\n",
                i, params->num_iterations, params->seed);
            goto cleanup;
        }
    }

cleanup:
    // Free all allocated buffers before returning
    cleanup_buffers(params, &buffers, is_align);

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, params->logger_mode, "EXIT");
#endif
    return status;
}

/**
 * Buffer size requirements for different FFT types and placement modes
 *
 * The following table summarizes the buffer sizes (in bytes) required for each
 * FFT type/placement, with FFT types/placement as rows and buffer names as
 * columns:
 *
 * +-------------------+---------------------------+---------------------------+---------------------------+---------------------------+
 * | FFT Type          | complex_in                | complex_out               | in_ref                    | out_ref                   |
 * +-------------------+---------------------------+---------------------------+---------------------------+---------------------------+
 * | C2C in-place      | (nil)                     | (nil)                     | input size (with strides) | problem size (no strides) |
 * | C2C out-of-place  | (nil)                     | (nil)                     | params->in                | problem size (no strides) |
 * | R2C in-place      | problem size (no strides) | problem size (no strides) | complex_in                | problem size (no strides) |
 * | R2C out-of-place  | problem size (no strides) | problem size (no strides) | complex_in                | problem size (no strides) |
 * | C2R in-place      | problem size (no strides) | problem size (no strides) | complex_in                | problem size (no strides) |
 * | C2R out-of-place  | problem size (no strides) | problem size (no strides) | complex_in                | problem size (no strides) |
 * +-------------------+---------------------------+---------------------------+---------------------------+---------------------------+
 *
 * Here,
 *   problem size (no strides) = batches * transform size * complex data stride * dt_bytes
 *   input_size = (size of batches with vec_in_strides and transforms with in_strides) * data stride * dt_bytes
 */
INT32 allocate_dftref_buffers(aoclfftz_bench_params_t *params,
                              dft_ref_buffers_t *buffers, UINT32 is_align)
{
    INTP problem_bytes_without_strides =
        params->sz_info.n * params->sz_info.batches * DATA_STRIDE *
        params->sz_info.in_data_stride * params->sz_info.dt_bytes;

    // Always allocate reference output buffer for storing DFT results
    ALLOC_INIT(buffers->out_ref, VOID, problem_bytes_without_strides, is_align);
    if (buffers->out_ref == NULL)
    {
        AOCLFFTZ_LOG_UNFORMATTED(ERR, params->logger_mode,
                                 "Reference output buffer allocation failed");
        return MEMORY_FAILURE;
    }

    if (params->fft_type == C2C)
    {
        if (params->res_placement == IN_PLACE)
        {
            // For in-place C2C, need separate reference input buffer
            ALLOC_INIT(buffers->in_ref, VOID, params->sz_info.input_bytes,
                       is_align);
            if (buffers->in_ref == NULL)
            {
                AOCLFFTZ_LOG_UNFORMATTED(
                    ERR, params->logger_mode,
                    "Reference input buffer allocation failed");
                return MEMORY_FAILURE;
            }
        }
        else
        {
            // For out-of-place C2C, reference input points to original input
            buffers->in_ref = params->in;
        }
    }
    else // R2C & C2R transforms
    {
        // Allocate buffer for complex input (real->complex conversion)
        ALLOC_INIT(buffers->complex_in, VOID, problem_bytes_without_strides,
                   is_align);
        if (buffers->complex_in == NULL)
        {
            AOCLFFTZ_LOG_UNFORMATTED(ERR, params->logger_mode,
                                     "Complex input buffer allocation failed");
            return MEMORY_FAILURE;
        }

        // Allocate buffer for complex output (half-complex->complex conversion)
        ALLOC_INIT(buffers->complex_out, VOID, problem_bytes_without_strides,
                   is_align);
        if (buffers->complex_out == NULL)
        {
            AOCLFFTZ_LOG_UNFORMATTED(ERR, params->logger_mode,
                                     "Complex output buffer allocation failed");
            return MEMORY_FAILURE;
        }

        // For R2C & C2R, in_ref points to the complex input buffer
        buffers->in_ref = buffers->complex_in;
    }

    return BENCH_SUCCESS;
}

/**
 * @brief Execute FFT and prepare data based on FFT type and direction
 *
 * This function performs FFT computation with appropriate data preparation
 * based on the specified FFT type and direction.
 *
 * Data flow for different FFT types:
 *
 * C2C (Complex-to-Complex):
 *   FFTZ forward  : params->in => [FFT]  => params->out
 *   FFTZ backward : params->in => [IFFT] => params->out
 *   DFT-REF input : params->in => buffers->complex_in
 *
 * R2C (Real-to-Complex):
 *   FFTZ          : params->in => [FFT] => params->out => [HC2C conversion] => buffers->complex_out
 *   DFT-REF input : params->in => [R2C conversion] => buffers->complex_in
 *
 * C2R (Complex-to-Real):
 *   FFTZ          : params->in => [FFT] => params->out => [R2C conversion] => buffers->complex_out
 *   DFT-REF input : params->in => [HC2C conversion] => buffers->complex_in
 *
 * @param[in] params Pointer to FFT parameters structure containing input/output buffers
 * @param[in] buffers Pointer to intermediate buffer structure for data conversions
 * @param[in] fft_type Type of FFT operation (C2C, R2C, C2R, R2R)
 * @param[in] direction FFT direction (forward or inverse)
 *
 * @return Status code indicating success or failure of the operation
 */
INT32 execute_fft_and_postprocess(aoclfftz_bench_params_t *params,
                                  dft_ref_buffers_t *buffers, VOID *handle,
                                  INTP *in_idx_map, INTP *out_idx_map)
{
    INT32 ret;
    INTP problem_bytes_without_strides = params->sz_info.n *
                                         params->sz_info.batches * DATA_STRIDE *
                                         params->sz_info.dt_bytes;
    INTP complex_input_bytes = (params->fft_type == C2C)
                                   ? params->sz_info.input_bytes
                                   : problem_bytes_without_strides;
    INTP complex_output_bytes = problem_bytes_without_strides;

    if (params->fft_type == R2C)
    {
        // R2C Forward: Real to Complex
        // Zero out the complex input buffer before conversion
        memset(buffers->complex_in, 0, complex_input_bytes);

        // Convert real input to complex format for FFT
        convert_real_to_complex(buffers->complex_in, params->in,
                                params->sz_info.n, params->sz_info.batches,
                                in_idx_map, params->precision);

        // Execute library FFT (params->in -> params->out)
        ret = aoclfftz_execute(handle);
        if (ret != AOCLFFTZ_SUCCESS)
        {
            return EXECUTION_FAILURE;
        }

        // Convert library output from half-complex to full complex for
        // comparison
        memset(buffers->complex_out, 0, complex_output_bytes);
        convert_half_complex_to_complex(
            buffers->complex_out, params->out, params->sz_info.n,
            params->sz_info.batches, out_idx_map, params->precision);
    }
    else if (params->fft_type == C2R)
    {
        // Set DC and Nyquist components to zero for correct input
        set_zero_for_dc_and_nyquist_nd(
            params->in, params->sz_info.n, params->dims[0].n,
            params->sz_info.batches, in_idx_map, params->precision);

        // Convert half-complex input to full complex for FFT
        convert_half_complex_to_complex(
            buffers->complex_in, params->in, params->sz_info.n,
            params->sz_info.batches, in_idx_map, params->precision);

        // Execute library FFT (params->in -> params->out)
        ret = aoclfftz_execute(handle);
        if (ret != AOCLFFTZ_SUCCESS)
        {
            return EXECUTION_FAILURE;
        }

        // Convert library output from real to complex for comparison
        convert_real_to_complex(buffers->complex_out, params->out,
                                params->sz_info.n, params->sz_info.batches,
                                out_idx_map, params->precision);
    }
    else // C2C transforms
    {
        if (params->res_placement == IN_PLACE)
        {
            // Copy input to reference buffer before in-place operation
            memcpy(buffers->in_ref, params->in, complex_input_bytes);
        }

        // Execute library FFT (params->in -> params->out)
        ret = aoclfftz_execute(handle);
        if (ret != AOCLFFTZ_SUCCESS)
        {
            return EXECUTION_FAILURE;
        }

        // For C2C, library output is already in complex format
        buffers->complex_out = params->out;
    }

    return BENCH_SUCCESS;
}

/**
 * @brief Clean up allocated buffers
 */
VOID cleanup_buffers(aoclfftz_bench_params_t *params,
                     dft_ref_buffers_t *buffers, UINT32 is_align)
{
    // Always free reference output buffer
    FREE_ALLOCATED_MEM(buffers->out_ref, is_align);

    // Free buffers specific to R2C transforms
    if (params->fft_type == R2C || params->fft_type == C2R)
    {
        FREE_ALLOCATED_MEM(buffers->complex_in, is_align);
        FREE_ALLOCATED_MEM(buffers->complex_out, is_align);
    }
    // Free buffers specific to in-place C2C transforms
    else if (params->fft_type == C2C && params->res_placement == IN_PLACE)
    {
        FREE_ALLOCATED_MEM(buffers->in_ref, is_align);
    }
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
    // Use DOUBLE for intermediate variables to improve accuracy of accumulation
    DOUBLE e[DATA_STRIDE], mul_buf[DATA_STRIDE];
    FLOAT sign =
        (params->dir == BACKWARD || params->fft_type == C2R) ? 1.0f : -1.0f;
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

    DOUBLE *inv_dims = NULL;
    ALLOC_INIT(inv_dims, DOUBLE, rank * sizeof(DOUBLE), is_align);
    // Precompute angle inv_dims
    for (INTP i = 0; i < rank; i++)
    {
        inv_dims[i] = 1.0 / dims[i].n;
    }

    // Iterate over all batches
    for (INTP b = 0; b < batches; b++)
    {
        RESET_ND_COUNTER(out_counter, rank);
        // For each output point k (multi-dimensional index)
        for (INTP k = 0; k < n; k++)
        {
            // Compute output index, using out_idx_map if provided
            INTP out_idx = out_idx_map ? out_idx_map[b * n + k] * DATA_STRIDE
                                       : (b * n + k) * DATA_STRIDE;
            RESET_ND_COUNTER(in_counter, rank);
            // For each input point i (multi-dimensional index)
            for (INTP i = 0; i < n; i++)
            {
                // Compute input index, using in_idx_map if provided
                INTP in_idx = in_idx_map ? in_idx_map[b * n + i] * DATA_STRIDE
                                         : (b * n + i) * DATA_STRIDE;

                // Compute the DFT angle for this (k, i) pair:
                // angle = sign * 2π * dot(k, i) / N
                DOUBLE angle = sign * BENCH_2_PI;
                // The macro UPDATE_ANGLE updates 'angle' in-place based on the
                // current multi-dimensional indices and dimensions
                UPDATE_ANGLE(angle, in_counter, out_counter, inv_dims, rank);

                // Compute the complex exponential e^{-j*angle}
                e[0] = cos(angle); // real part
                e[1] = sin(angle); // imag part

                // Multiply input by the twiddle factor
                mul_buf[0] = (in_f[in_idx] * e[0]) - (in_f[in_idx + 1] * e[1]);
                mul_buf[1] = (in_f[in_idx] * e[1]) + (in_f[in_idx + 1] * e[0]);

                // Accumulate the result into the output buffer
                out_f[out_idx] = out_f[out_idx] + (FLOAT)mul_buf[0];
                out_f[out_idx + 1] = out_f[out_idx + 1] + (FLOAT)mul_buf[1];

                // Move to next input multi-dimensional index
                INCREMENT_ND_COUNTER(in_counter, dims, rank);
            }
            // Move to next output multi-dimensional index
            INCREMENT_ND_COUNTER(out_counter, dims, rank);
        }
    }
    // Free temporary counters
    FREE_ALLOCATED_MEM(in_counter, is_align);
    FREE_ALLOCATED_MEM(out_counter, is_align);
    FREE_ALLOCATED_MEM(inv_dims, is_align);
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
    DOUBLE e[DATA_STRIDE], mul_buf[DATA_STRIDE];
    DOUBLE sign =
        (params->dir == BACKWARD || params->fft_type == C2R) ? 1.0f : -1.0f;
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

    DOUBLE *inv_dims = NULL;
    ALLOC_INIT(inv_dims, DOUBLE, rank * sizeof(DOUBLE), is_align);
    // Precompute angle inv_dims
    for (INTP i = 0; i < rank; i++)
    {
        inv_dims[i] = 1.0 / dims[i].n;
    }

    // Iterate over all batches
    // TODO: add parallelization over batches to speed up, cache twiddles
    for (INTP b = 0; b < batches; b++)
    {
        RESET_ND_COUNTER(out_counter, rank);
        // For each output point k (multi-dimensional index)
        for (INTP k = 0; k < n; k++)
        {
            // Compute output index, using out_idx_map if provided
            INTP out_idx = out_idx_map ? out_idx_map[b * n + k] * DATA_STRIDE
                                       : (b * n + k) * DATA_STRIDE;
            RESET_ND_COUNTER(in_counter, rank);
            // For each input point i (multi-dimensional index)
            for (INTP i = 0; i < n; i++)
            {
                // Compute input index, using in_idx_map if provided
                INTP in_idx = in_idx_map ? in_idx_map[b * n + i] * DATA_STRIDE
                                         : (b * n + i) * DATA_STRIDE;

                // Compute the DFT angle for this (k, i) pair:
                // angle = sign * 2π * dot(k, i) / N
                DOUBLE angle = sign * BENCH_2_PI;
                // The macro UPDATE_ANGLE updates 'angle' in-place based on the
                // current multi-dimensional indices and dimensions
                UPDATE_ANGLE(angle, in_counter, out_counter, inv_dims, rank);

                // Compute the complex exponential e^{-j*angle}
                e[0] = cos(angle); // real part
                e[1] = sin(angle); // imag part

                // Multiply input by the twiddle factor
                mul_buf[0] = (in_d[in_idx] * e[0]) - (in_d[in_idx + 1] * e[1]);
                mul_buf[1] = (in_d[in_idx] * e[1]) + (in_d[in_idx + 1] * e[0]);

                // Accumulate the result into the output buffer
                out_d[out_idx] = out_d[out_idx] + mul_buf[0];
                out_d[out_idx + 1] = out_d[out_idx + 1] + mul_buf[1];

                // Move to next input multi-dimensional index
                INCREMENT_ND_COUNTER(in_counter, dims, rank);
            }
            // Move to next output multi-dimensional index
            INCREMENT_ND_COUNTER(out_counter, dims, rank);
        }
    }
    // Free temporary counters
    FREE_ALLOCATED_MEM(in_counter, is_align);
    FREE_ALLOCATED_MEM(out_counter, is_align);
    FREE_ALLOCATED_MEM(inv_dims, is_align);
}
#endif
