/**
 * Copyright (C) 2023, Advanced Micro Devices. All rights reserved.
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

/** @file twiddle.c
 *
 *  @brief Computes and applies Twiddle factor.
 *
 *  This file contains the functions related to computing and multiplying
 *  twiddle factors to the values as needed between FFT stages
 *
 *  @author S. Biplab Raut
 *  @author Prasandh Sankarankutty
 *  @author Ashwin K. Godbole
 */

#include <math.h>
#include "core/common/twiddle.h"
#include "core/kernels/kernel.h"
#include "utils/allocator.h"
#include "api/aoclfftz_internal.h"
#include "selector/selector.h"
#include "core/kernels/transpose/transpose_utils.h"

INT32 twiddle_multiplier_float(aoclfftz_solution_t *sol)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    FLOAT *in_real = (FLOAT *)sol->decomp_scheme->in_real;
    FLOAT *in_imag = (FLOAT *)sol->decomp_scheme->in_imag;
    FLOAT *out_real = (FLOAT *)sol->decomp_scheme->out_real;
    FLOAT *out_imag = (FLOAT *)sol->decomp_scheme->out_imag;

    INTP sets = sol->decomp_scheme->vecs[0].n;
    INTP radix = sol->decomp_scheme->dims[0].n;
    INTP N = sets * radix;

    // out-of-order -> out-of-order multiplication of twiddle
    // output_buffer = output_buffer * twiddle_val
    INTP in_stride = sol->decomp_scheme->dims[0].in_stride * DATA_STRIDE;
    INTP out_stride = sol->decomp_scheme->dims[0].out_stride * DATA_STRIDE;
    INTP v_in_stride = sol->decomp_scheme->vecs[0].in_stride * DATA_STRIDE;
    INTP v_out_stride =
        sol->decomp_scheme->vecs[0].out_stride * DATA_STRIDE;

    for (INTP s = 0; s < sets; s++)
    {
        INTP out_idx = 0;
        INTP in_idx = 0;
        for (INTP r = 0; r < radix; r++)
        {
            FLOAT x = (-AOCLFFTZ_2_PIf * r * s) / ((FLOAT)(N));

            FLOAT TW_real = cosf(x);
            FLOAT TW_imag = sinf(x);
            FLOAT real = out_real[out_idx];
            FLOAT imag = out_imag[out_idx];

            in_real[in_idx] = real * TW_real - imag * TW_imag;
            in_imag[in_idx] = real * TW_imag + imag * TW_real;

            in_idx += in_stride;
            out_idx += out_stride;
        }

        in_real += v_in_stride;
        in_imag += v_in_stride;
        out_real += v_out_stride;
        out_imag += v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
    return TW_SUCCESS;
}

INT32 twiddle_multiplier_double(aoclfftz_solution_t *sol)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    DOUBLE *in_real = (DOUBLE *)sol->decomp_scheme->in_real;
    DOUBLE *in_imag = (DOUBLE *)sol->decomp_scheme->in_imag;
    DOUBLE *out_real = (DOUBLE *)sol->decomp_scheme->out_real;
    DOUBLE *out_imag = (DOUBLE *)sol->decomp_scheme->out_imag;

    INTP sets = sol->decomp_scheme->vecs[0].n;
    INTP radix = sol->decomp_scheme->dims[0].n;
    INTP N = sets * radix;

    // out-of-order -> out-of-order multiplication of twiddle
    // output_buffer = output_buffer * twiddle_val
    INTP in_stride = sol->decomp_scheme->dims[0].in_stride * DATA_STRIDE;
    INTP out_stride = sol->decomp_scheme->dims[0].out_stride * DATA_STRIDE;
    INTP v_in_stride = sol->decomp_scheme->vecs[0].in_stride * DATA_STRIDE;
    INTP v_out_stride =
        sol->decomp_scheme->vecs[0].out_stride * DATA_STRIDE;

    for (INTP s = 0; s < sets; s++)
    {
        INTP out_idx = 0;
        INTP in_idx = 0;
        for (INTP r = 0; r < radix; r++)
        {
            DOUBLE x = (-AOCLFFTZ_2_PI * r * s) / ((DOUBLE)(N));

            DOUBLE TW_real = cos(x);
            DOUBLE TW_imag = sin(x);

            DOUBLE real = out_real[out_idx];
            DOUBLE imag = out_imag[out_idx];

            in_real[in_idx] = real * TW_real - imag * TW_imag;
            in_imag[in_idx] = real * TW_imag + imag * TW_real;

            in_idx += in_stride;
            out_idx += out_stride;
        }

        in_real += v_in_stride;
        in_imag += v_in_stride;
        out_real += v_out_stride;
        out_imag += v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
    return TW_SUCCESS;
}

INT32 twiddle_multiplier(aoclfftz_solution_t *sol)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    UINT32 precision = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    INT32 status = TW_FAILURE;

    if (precision == DT_FLOAT)
    {
        status = twiddle_multiplier_float(sol);
    }
    else
    {
        status = twiddle_multiplier_double(sol);
    }

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
    return status;
}

/* The reordering of radix-m outputs is done by following a cyclic permutation
 * pattern to convert the out-of-order output into in-order output.
 * It is similar to out-of-place radix-m transform operation that writes to
 * a separate output buffer from out-of-order output into in-order output.
 *
 * Algorithm :
 *
 * Step 0 : Start from an index i
 * Step 1 : Keep processing the indices in cyclic manner until
 *          the start index "i" is encountered again.
 *          Mark the visited indices with 1.
 * Step 2 : Traverse along the 'visited' buffer to find the next unprocessed idx
 * Step 3 : Repeat Step 0 - 2 until the entire buffer is processed.
 */
INT32 twiddle_multiplier_inplace_nonbuffered_float(aoclfftz_solution_t *sol)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    // both buffers point to same memory location in an inplace problem
    FLOAT *in_real = (FLOAT *)sol->decomp_scheme->out_real;
    FLOAT *in_imag = (FLOAT *)sol->decomp_scheme->out_imag;
    FLOAT *out_real = (FLOAT *)sol->decomp_scheme->in_real;
    FLOAT *out_imag = (FLOAT *)sol->decomp_scheme->in_imag;

    INTP sets = sol->decomp_scheme->vecs[0].n;
    INTP radix = sol->decomp_scheme->dims[0].n;
    INTP N = sets * radix;

    // buffer to mark processed index
    INT8 *visited = NULL;
    ALLOC_ALIGN_INIT(visited, INT8, N * sizeof(INT8));

    INTP in_stride =
        (sol->decomp_scheme->dims[0].in_stride / sets) * DATA_STRIDE;
    INTP out_stride =
        (sol->decomp_scheme->dims[0].out_stride / sets) * DATA_STRIDE;

    INTP count = 0;
    INTP start_idx = 0;
    while (count != N)
    {
        // find the unprocessed index
        for (INTP i = start_idx; i < N; i++)
        {
            if (visited[i] != 1)
            {
                start_idx = i;
                break;
            }
        }

        INTP src_idx = start_idx;
        INTP src = src_idx * in_stride;
        FLOAT src_real = in_real[src];
        FLOAT src_imag = in_imag[src];

        // process the current cycle until the start index is reached again
        do
        {
            // find the permuted index for the src index
            INTP s = src_idx / radix;
            INTP r = (src_idx % radix);
            INTP dst_idx = (s + (sets * r));
            INTP dst = dst_idx * out_stride;
            // store the dst's current data as it would get overwritten
            FLOAT next_real = in_real[dst];
            FLOAT next_imag = in_imag[dst];

            FLOAT x = (-AOCLFFTZ_2_PIf * r * s) / ((FLOAT)(N));
            FLOAT TW_real = cosf(x);
            FLOAT TW_imag = sinf(x);

            out_real[dst] = src_real * TW_real - src_imag * TW_imag;
            out_imag[dst] = src_real * TW_imag + src_imag * TW_real;

            visited[src_idx] = 1;
            count++;
            src_real = next_real;
            src_imag = next_imag;
            src_idx = dst_idx;
        } while (count != N && start_idx != src_idx);
    }
    FREE_ALIGN_ALLOCATED_MEM(visited);
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
    return TW_SUCCESS;
}

/* The reordering of radix-m outputs is done by following a cyclic permutation
 * pattern to convert the out-of-order output into in-order output.
 * It is similar to out-of-place radix-m transform operation that writes to
 * a separate output buffer from out-of-order output into in-order output.
 *
 * Algorithm :
 *
 * Step 0 : Start from an index i
 * Step 1 : Keep processing the indices in cyclic manner until
 *          the start index "i" is encountered again.
 *          Mark the visited indices with 1.
 * Step 2 : Traverse along the 'visited' buffer to find the next unprocessed idx
 * Step 3 : Repeat Step 0 - 2 until the entire buffer is processed.
 */
INT32 twiddle_multiplier_inplace_nonbuffered_double(aoclfftz_solution_t *sol)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    // both buffers point to same memory location in an inplace problem
    DOUBLE *in_real = (DOUBLE *)sol->decomp_scheme->out_real;
    DOUBLE *in_imag = (DOUBLE *)sol->decomp_scheme->out_imag;
    DOUBLE *out_real = (DOUBLE *)sol->decomp_scheme->in_real;
    DOUBLE *out_imag = (DOUBLE *)sol->decomp_scheme->in_imag;

    INTP sets = sol->decomp_scheme->vecs[0].n;
    INTP radix = sol->decomp_scheme->dims[0].n;
    INTP N = sets * radix;

    // buffer to mark processed index
    INT8 *visited = NULL;
    ALLOC_ALIGN_INIT(visited, INT8, N * sizeof(INT8));

    INTP in_stride =
        (sol->decomp_scheme->dims[0].in_stride / sets) * DATA_STRIDE;
    INTP out_stride =
        (sol->decomp_scheme->dims[0].out_stride / sets) * DATA_STRIDE;

    INTP count = 0;
    INTP start_idx = 0;
    while (count != N)
    {
        // find the unprocessed index
        for (INTP i = start_idx; i < N; i++)
        {
            if (visited[i] != 1)
            {
                start_idx = i;
                break;
            }
        }

        INTP src_idx = start_idx;
        INTP src = src_idx * in_stride;
        DOUBLE src_real = in_real[src];
        DOUBLE src_imag = in_imag[src];

        // process the current cycle until the start index is reached again
        do
        {
            // find the permuted index for the src index
            INTP s = src_idx / radix;
            INTP r = (src_idx % radix);
            INTP dst_idx = (s + (sets * r));
            INTP dst = dst_idx * out_stride;
            // store the dst's current data as it would get overwritten
            DOUBLE next_real = in_real[dst];
            DOUBLE next_imag = in_imag[dst];

            DOUBLE x = (-AOCLFFTZ_2_PI * r * s) / ((DOUBLE)(N));
            DOUBLE TW_real = cos(x);
            DOUBLE TW_imag = sin(x);

            out_real[dst] = src_real * TW_real - src_imag * TW_imag;
            out_imag[dst] = src_real * TW_imag + src_imag * TW_real;

            visited[src_idx] = 1;
            count++;
            src_real = next_real;
            src_imag = next_imag;
            src_idx = dst_idx;
        } while (count != N && start_idx != src_idx);
    }
    FREE_ALIGN_ALLOCATED_MEM(visited);
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
    return TW_SUCCESS;
}

static inline INT32
twiddle_multiplier_inplace_buffered_float(aoclfftz_solution_t *sol)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    INTP sets = sol->decomp_scheme->vecs[0].n;
    INTP radix = sol->decomp_scheme->dims[0].n;
    INTP N = sets * radix;

    INTP rows = sets;
    INTP cols = radix;
    INTP col_stride = sol->decomp_scheme->vecs[0].in_stride * DATA_STRIDE;
    INTP row_stride = col_stride * cols;

    INTP sc_rows = radix;
    INTP sc_cols = sets;
    INTP scratch_col_stride = DATA_STRIDE;
    INTP scratch_row_stride = sc_cols * DATA_STRIDE;

    FLOAT *in_real = (FLOAT *)sol->decomp_scheme->out_real;
    FLOAT *in_imag = (FLOAT *)sol->decomp_scheme->out_imag;
    aoclfftz_complex_f_t *in; // decided based on the direction of FFT

    FLOAT *scratch_real; // decided based on the direction of FFT
    FLOAT *scratch_imag; // decided based on the direction of FFT
    aoclfftz_complex_f_t *sc = (aoclfftz_complex_f_t *)sol->scratch_space;

    if (FFT_DIR(sol->decomp_scheme->flags) == FORWARD_FFT_DIR)
    {
        in = (aoclfftz_complex_f_t *)sol->decomp_scheme->out_real;
        scratch_real = (FLOAT *)sol->scratch_space;
        scratch_imag = scratch_real + 1;
    }
    else
    {
        in = (aoclfftz_complex_f_t *)sol->decomp_scheme->out_imag;
        scratch_imag = (FLOAT *)sol->scratch_space;
        scratch_real = scratch_imag + 1;
    }

    INTP sc_cs = 1;
    INTP sc_rs = rows;
    INTP in_cs = col_stride / DATA_STRIDE;
    INTP in_rs = row_stride / DATA_STRIDE;

    // transpose the input into the scratch space
    for (INTP i = 0; i < rows; i++)
    {
        for (INTP j = 0; j < cols; j++)
        {
            sc[LINEAR_IDX_2D(j, i, sc_cs, sc_rs)] =
                in[LINEAR_IDX_2D(i, j, in_cs, in_rs)];
        }
    }

    row_stride = sol->decomp_scheme->dims[0].in_stride * DATA_STRIDE;
    col_stride = sol->decomp_scheme->vecs[0].in_stride * DATA_STRIDE;

    for (INTP i = 0; i < sc_rows; i++)
    {
        for (INTP j = 0; j < sc_cols; j++)
        {
            FLOAT x = (-AOCLFFTZ_2_PIf * i * j) / ((FLOAT)(N));

            FLOAT TW_real = cosf(x);
            FLOAT TW_imag = sinf(x);

            FLOAT real = scratch_real[LINEAR_IDX_2D(i, j, scratch_col_stride,
                                                    scratch_row_stride)];
            FLOAT imag = scratch_imag[LINEAR_IDX_2D(i, j, scratch_col_stride,
                                                    scratch_row_stride)];

            in_real[LINEAR_IDX_2D(i, j, col_stride, row_stride)] =
                real * TW_real - imag * TW_imag;
            in_imag[LINEAR_IDX_2D(i, j, col_stride, row_stride)] =
                real * TW_imag + imag * TW_real;
        }
    }

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
    return TW_SUCCESS;
}

static inline INT32
twiddle_multiplier_inplace_buffered_double(aoclfftz_solution_t *sol)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    INTP sets = sol->decomp_scheme->vecs[0].n;
    INTP radix = sol->decomp_scheme->dims[0].n;
    INTP N = sets * radix;

    INTP rows = sets;
    INTP cols = radix;
    INTP col_stride = sol->decomp_scheme->vecs[0].in_stride * DATA_STRIDE;
    INTP row_stride = col_stride * cols;

    INTP sc_rows = radix;
    INTP sc_cols = sets;
    INTP scratch_col_stride = DATA_STRIDE;
    INTP scratch_row_stride = sc_cols * DATA_STRIDE;

    DOUBLE *in_real = (DOUBLE *)sol->decomp_scheme->out_real;
    DOUBLE *in_imag = (DOUBLE *)sol->decomp_scheme->out_imag;
    aoclfftz_complex_d_t *in; // decided based on the direction of FFT

    DOUBLE *scratch_real; // decided based on the direction of FFT
    DOUBLE *scratch_imag; // decided based on the direction of FFT
    aoclfftz_complex_d_t *sc = (aoclfftz_complex_d_t *)sol->scratch_space;

    if (FFT_DIR(sol->decomp_scheme->flags) == FORWARD_FFT_DIR)
    {
        in = (aoclfftz_complex_d_t *)sol->decomp_scheme->out_real;
        scratch_real = (DOUBLE *)sol->scratch_space;
        scratch_imag = scratch_real + 1;
    }
    else
    {
        in = (aoclfftz_complex_d_t *)sol->decomp_scheme->out_imag;
        scratch_imag = (DOUBLE *)sol->scratch_space;
        scratch_real = scratch_imag + 1;
    }

    INTP sc_cs = 1;
    INTP sc_rs = rows;
    INTP in_cs = col_stride / DATA_STRIDE;
    INTP in_rs = row_stride / DATA_STRIDE;

    for (INTP i = 0; i < rows; i++)
    {
        for (INTP j = 0; j < cols; j++)
        {
            sc[LINEAR_IDX_2D(j, i, sc_cs, sc_rs)] =
                in[LINEAR_IDX_2D(i, j, in_cs, in_rs)];
        }
    }

    row_stride = sol->decomp_scheme->dims[0].in_stride * DATA_STRIDE;
    col_stride = sol->decomp_scheme->vecs[0].in_stride * DATA_STRIDE;

    for (INTP i = 0; i < sc_rows; i++)
    {
        for (INTP j = 0; j < sc_cols; j++)
        {
            DOUBLE x = (-AOCLFFTZ_2_PI * i * j) / ((DOUBLE)(N));

            DOUBLE TW_real = cos(x);
            DOUBLE TW_imag = sin(x);

            DOUBLE real = scratch_real[LINEAR_IDX_2D(i, j, scratch_col_stride,
                                                     scratch_row_stride)];
            DOUBLE imag = scratch_imag[LINEAR_IDX_2D(i, j, scratch_col_stride,
                                                     scratch_row_stride)];

            in_real[LINEAR_IDX_2D(i, j, col_stride, row_stride)] =
                real * TW_real - imag * TW_imag;
            in_imag[LINEAR_IDX_2D(i, j, col_stride, row_stride)] =
                real * TW_imag + imag * TW_real;
        }
    }

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
    return TW_SUCCESS;
}

// An in-place twiddle multiplication function that uses the scratch space to
// store intermediate results
INT32 twiddle_multiplier_inplace(aoclfftz_solution_t *sol)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    UINT32 precision = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    INTP sets = sol->decomp_scheme->vecs[0].n;
    INTP radix = sol->decomp_scheme->dims[0].n;
    INT32 status = TW_FAILURE;

    if (sol->scratch_space == NULL)
    {
        if (precision == DT_FLOAT)
        {
            status = twiddle_multiplier_inplace_nonbuffered_float(sol);
        }
        else
        {
            status = twiddle_multiplier_inplace_nonbuffered_double(sol);
        }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
        return status;
    }

    INTP total_size = radix * sets * DATA_STRIDE;
    INTP scratch_available = scratch_space_capacity / (2 * (1 << precision));

    if (total_size > scratch_available)
    {
        if (precision == DT_FLOAT)
        {
            status = twiddle_multiplier_inplace_nonbuffered_float(sol);
        }
        else
        {
            status = twiddle_multiplier_inplace_nonbuffered_double(sol);
        }
    }
    else
    {
        if (precision == DT_FLOAT)
        {
            status = twiddle_multiplier_inplace_buffered_float(sol);
        }
        else
    {
            status = twiddle_multiplier_inplace_buffered_double(sol);
        }
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
    return status;
}

INT32 twiddle_multiplier_for_real(aoclfftz_solution_t *sol, INTP p)
{
    UINT32 precision = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    UINT32 is_backward = FFT_DIR(sol->decomp_scheme->flags);
    VOID *in = sol->decomp_scheme->in_real;
    VOID *out = sol->decomp_scheme->out_real;
    INTP *batches = sol->solver->batches;
    INTP radix = sol->decomp_scheme->dims[0].n;
    aoclfftz_strides_t *strides = sol->strides;
    // FIXIT: Fix for fully strided CT problems (i.e. strides in all CT stages)
    INTP base_stride = 1;

    INTP *stride_arr = NULL;
    INTP vec_stride = 1;
    if (is_backward)
    {
        stride_arr = strides->out_strides;
        vec_stride = strides->v_out_stride;
    }
    else
    {
        stride_arr = strides->in_strides;
        vec_stride = strides->v_in_stride;
    }

    INTP no_of_groups = batches[R2HCF_KERNEL] > 0 ?
                        batches[R2HCF_KERNEL] :
                        batches[R2HC_KERNEL];
    INTP group_size = batches[C2C_KERNEL] / no_of_groups;

    if (precision == DT_FLOAT)
    {
        FLOAT sign = is_backward ? 1.0 : -1.0;
        FLOAT *data_r = is_backward ? (FLOAT *)out : (FLOAT *)in;
        FLOAT *data_i = is_backward ? (FLOAT *)out + 1 : (FLOAT *)in + 1;

        // move the in_r to point first C2C point
        data_r += base_stride;
        data_i += base_stride;

        for (INTP i = 0; i < group_size; i++)
        {
            for (INTP j = 1; j < radix; j++)
            {
                FLOAT *cur_data_r = data_r;
                FLOAT *cur_data_i = data_i;

                FLOAT x = (sign * AOCLFFTZ_2_PI * (i + 1) * j) / p;
                FLOAT TW_real = cos(x);
                FLOAT TW_imag = sin(x);

                INTP stride = stride_arr[j];

                for (INTP k = 0; k < no_of_groups; k++)
                {
                    FLOAT a = cur_data_r[stride];
                    FLOAT b = cur_data_i[stride];
                    cur_data_r[stride] = a * TW_real - b * TW_imag;
                    cur_data_i[stride] = b * TW_real + a * TW_imag;
                    stride += vec_stride;
                }
            }
            data_r += base_stride * 2;
            data_i += base_stride * 2;
        }
    }
    else
    {
        DOUBLE sign = is_backward ? 1.0 : -1.0;
        DOUBLE *data_r = is_backward ? (DOUBLE *)out : (DOUBLE *)in;
        DOUBLE *data_i = is_backward ? (DOUBLE *)out + 1 : (DOUBLE *)in + 1;

        // move the in_r to point first C2C point
        data_r += base_stride;
        data_i += base_stride;

        for (INTP i = 0; i < group_size; i++)
        {
            for (INTP j = 1; j < radix; j++)
            {
                DOUBLE *cur_data_r = data_r;
                DOUBLE *cur_data_i = data_i;

                DOUBLE x = (sign * AOCLFFTZ_2_PI * (i + 1) * j) / p;
                DOUBLE TW_real = cos(x);
                DOUBLE TW_imag = sin(x);

                INTP stride = stride_arr[j];

                for (INTP k = 0; k < no_of_groups; k++)
                {
                    DOUBLE a = cur_data_r[stride];
                    DOUBLE b = cur_data_i[stride];
                    cur_data_r[stride] = a * TW_real - b * TW_imag;
                    cur_data_i[stride] = b * TW_real + a * TW_imag;
                    stride += vec_stride;
                }
            }
            data_r += base_stride * 2;
            data_i += base_stride * 2;
        }
    }

    return TW_SUCCESS;
}
