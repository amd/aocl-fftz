/**
 * Copyright (C) 2023-2025, Advanced Micro Devices. All rights reserved.
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

// The first row and first column are not processed because the twiddle
// factors for them are such that they have to just be copied to the
// same location as where they are read from.
//
// since sin(0) = 0 and cos(0) = 1, the first row and first column
// elements are processed as:
//
// in_real[idx] = real * TW_real - imag * TW_imag
//              = real * cos(0) - imag * sin(0)
//              = real * 1 - imag * 0
//              = real
//
// in_imag[idx] = real * TW_imag + imag * TW_real
//              = real * sin(0) + imag * cos(0)
//              = real * 0 + imag * 1
//              = imag
//
// where
//      real = out_real[idx] and imag = out_imag[idx]
// but
//      out_real = in_real and out_imag = in_imag
//
// so
//      real = in_real[idx]
//      imag = in_imag[idx]
//
// so
//      in_real[idx] = real = in_real[idx]
//      in_imag[idx] = imag = in_imag[idx]

static VOID twiddle_multiplier_float(aoclfftz_solution_t *sol)
{
    FLOAT *in_real, *in_imag;

    if (FFT_DIR(sol->decomp_scheme->flags) == FORWARD_FFT_DIR)
    {
        in_real  = (FLOAT *)sol->decomp_scheme->in_real;
        in_imag  = (FLOAT *)sol->decomp_scheme->in_imag;
    }
    else
    {
        in_real  = (FLOAT *)sol->decomp_scheme->in_imag;
        in_imag  = (FLOAT *)sol->decomp_scheme->in_real;
    }

    INTP sets = sol->decomp_scheme->vecs[0].n;
    INTP radix = sol->decomp_scheme->dims[0].n;
    INTP N = sets * radix;

    INTP in_stride = sol->decomp_scheme->dims[0].in_stride * DATA_STRIDE;
    INTP v_in_stride = sol->decomp_scheme->vecs[0].in_stride * DATA_STRIDE;

#if IN_MEMORY_TWIDDLE_FACTORS == 1
    FLOAT *twiddle_buffer_real = (FLOAT *)sol->twiddle->TW;

    if (sol->twiddle->TW)
    {
        FLOAT *twiddle_buffer_imag = twiddle_buffer_real + 1;
        for (INTP r = 1; r < radix; r++)
        {
            INTP in_index = r * in_stride + v_in_stride;
            INTP tw_in_index =  DATA_STRIDE * (r * sets + 1);

            for (INTP s = 1; s < sets; s++)
            {
                FLOAT TW_real = twiddle_buffer_real[tw_in_index];
                FLOAT TW_imag = twiddle_buffer_imag[tw_in_index];

                FLOAT real = in_real[in_index];
                FLOAT imag = in_imag[in_index];

                in_real[in_index] = real * TW_real - imag * TW_imag;
                in_imag[in_index] = real * TW_imag + imag * TW_real;

                in_index += v_in_stride;
                tw_in_index += DATA_STRIDE;
            }
        }
    }
    else
#endif
    {
        for (INTP r = 1; r < radix; r++)
        {
            INTP in_index = r * in_stride + v_in_stride;

            for (INTP s = 1; s < sets; s++)
            {
                FLOAT x = (-AOCLFFTZ_2_PIf * r * s) / ((FLOAT)(N));

                FLOAT TW_real = cosf(x);
                FLOAT TW_imag = sinf(x);
                FLOAT real = in_real[in_index];
                FLOAT imag = in_imag[in_index];

                in_real[in_index] = real * TW_real - imag * TW_imag;
                in_imag[in_index] = real * TW_imag + imag * TW_real;

                in_index += v_in_stride;
            }
        }
    }
}

// The first row and first column are not processed because the twiddle
// factors for them are such that they have to just be copied to the
// same location as where they are read from.
//
// since sin(0) = 0 and cos(0) = 1, the first row and first column
// elements are processed as:
//
// in_real[idx] = real * TW_real - imag * TW_imag
//              = real * cos(0) - imag * sin(0)
//              = real * 1 - imag * 0
//              = real
//
// in_imag[idx] = real * TW_imag + imag * TW_real
//              = real * sin(0) + imag * cos(0)
//              = real * 0 + imag * 1
//              = imag
//
// where
//      real = out_real[idx] and imag = out_imag[idx]
// but
//      out_real = in_real and out_imag = in_imag
//
// so
//      real = in_real[idx]
//      imag = in_imag[idx]
//
// so
//      in_real[idx] = real = in_real[idx]
//      in_imag[idx] = imag = in_imag[idx]

static VOID twiddle_multiplier_double(aoclfftz_solution_t *sol)
{
    DOUBLE *in_real, *in_imag;

    if (FFT_DIR(sol->decomp_scheme->flags) == FORWARD_FFT_DIR)
    {
        in_real  = (DOUBLE *)sol->decomp_scheme->in_real;
        in_imag  = (DOUBLE *)sol->decomp_scheme->in_imag;
    }
    else
    {
        in_real  = (DOUBLE *)sol->decomp_scheme->in_imag;
        in_imag  = (DOUBLE *)sol->decomp_scheme->in_real;
    }

    INTP sets = sol->decomp_scheme->vecs[0].n;
    INTP radix = sol->decomp_scheme->dims[0].n;
    INTP N = sets * radix;

    INTP in_stride = sol->decomp_scheme->dims[0].in_stride * DATA_STRIDE;
    INTP v_in_stride = sol->decomp_scheme->vecs[0].in_stride * DATA_STRIDE;

#if IN_MEMORY_TWIDDLE_FACTORS == 1
    DOUBLE *twiddle_buffer_real = (DOUBLE *)sol->twiddle->TW;

    if (sol->twiddle->TW)
    {
        DOUBLE *twiddle_buffer_imag = twiddle_buffer_real + 1;
        for (INTP r = 1; r < radix; r++)
        {
            INTP in_index = r * in_stride + v_in_stride;
            INTP tw_in_index =  DATA_STRIDE * (r * sets + 1);

            for (INTP s = 1; s < sets; s++)
            {
                DOUBLE TW_real = twiddle_buffer_real[tw_in_index];
                DOUBLE TW_imag = twiddle_buffer_imag[tw_in_index];

                DOUBLE real = in_real[in_index];
                DOUBLE imag = in_imag[in_index];

                in_real[in_index] = real * TW_real - imag * TW_imag;
                in_imag[in_index] = real * TW_imag + imag * TW_real;

                in_index += v_in_stride;
                tw_in_index += DATA_STRIDE;
            }
        }
    }
    else
#endif
    {
        for (INTP r = 1; r < radix; r++)
        {
            INTP in_index = r * in_stride + v_in_stride;

            for (INTP s = 1; s < sets; s++)
            {
                DOUBLE x = (-AOCLFFTZ_2_PI * r * s) / ((DOUBLE)(N));

                DOUBLE TW_real = cos(x);
                DOUBLE TW_imag = sin(x);
                DOUBLE real = in_real[in_index];
                DOUBLE imag = in_imag[in_index];

                in_real[in_index] = real * TW_real - imag * TW_imag;
                in_imag[in_index] = real * TW_imag + imag * TW_real;

                in_index += v_in_stride;
            }
        }
    }
}

INT32 twiddle_multiplier(aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");
    UINT32 precision = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    if (precision == DT_FLOAT)
    {
        twiddle_multiplier_float(sol);
    }
    else
    {
        twiddle_multiplier_double(sol);
    }
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
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
static INT32 twiddle_multiplier_inplace_nonbuffered_float(aoclfftz_solution_t *sol)
{
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

#if IN_MEMORY_TWIDDLE_FACTORS == 1
    FLOAT *twiddle_buffer_real = (FLOAT *)sol->twiddle->TW;
    FLOAT *twiddle_buffer_imag = twiddle_buffer_real + 1;
#endif

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
            INTP s = floor(src_idx / radix);
            INTP r = (src_idx % radix);
            INTP dst_idx = (s + (sets * r));
            INTP dst = dst_idx * out_stride;
            // store the dst's current data as it would get overwritten
            FLOAT next_real = in_real[dst];
            FLOAT next_imag = in_imag[dst];

            FLOAT TW_real, TW_imag;

            if (r == 0 || s == 0)
            {
                out_real[dst] = src_real;
                out_imag[dst] = src_imag;
            }
            else
            {
#if IN_MEMORY_TWIDDLE_FACTORS == 1
                if (twiddle_buffer_real)
                {
                    TW_real = twiddle_buffer_real[LINEAR_IDX_2D(r, s,
                            DATA_STRIDE, DATA_STRIDE * sets)];
                    TW_imag = twiddle_buffer_imag[LINEAR_IDX_2D(r, s,
                            DATA_STRIDE, DATA_STRIDE * sets)];
                }
                else
#endif
                {
                    FLOAT x = (-AOCLFFTZ_2_PIf * r * s) / ((FLOAT)(N));
                    TW_real = cosf(x);
                    TW_imag = sinf(x);
                }

                out_real[dst] = src_real * TW_real - src_imag * TW_imag;
                out_imag[dst] = src_real * TW_imag + src_imag * TW_real;
            }

            visited[src_idx] = 1;
            count++;
            src_real = next_real;
            src_imag = next_imag;
            src_idx = dst_idx;
        } while (count != N && start_idx != src_idx);
    }
    FREE_ALIGN_ALLOCATED_MEM(visited);
    return TW_SUCCESS;
}

static INT32 twiddle_multiplier_inplace_nonbuffered_double(
    aoclfftz_solution_t *sol)
{
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

#if IN_MEMORY_TWIDDLE_FACTORS == 1
    DOUBLE *twiddle_buffer_real = (DOUBLE *)sol->twiddle->TW;
    DOUBLE *twiddle_buffer_imag = twiddle_buffer_real + 1;
#endif

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
            INTP s = floor(src_idx / radix);
            INTP r = (src_idx % radix);
            INTP dst_idx = (s + (sets * r));
            INTP dst = dst_idx * out_stride;
            // store the dst's current data as it would get overwritten
            DOUBLE next_real = in_real[dst];
            DOUBLE next_imag = in_imag[dst];

            DOUBLE TW_real, TW_imag;

            if (r == 0 || s == 0)
            {
                out_real[dst] = src_real;
                out_imag[dst] = src_imag;
            }
            else
            {
#if IN_MEMORY_TWIDDLE_FACTORS == 1
                if (twiddle_buffer_real)
                {
                    TW_real = twiddle_buffer_real[LINEAR_IDX_2D(
                        r, s, DATA_STRIDE, DATA_STRIDE * sets)];
                    TW_imag = twiddle_buffer_imag[LINEAR_IDX_2D(
                        r, s, DATA_STRIDE, DATA_STRIDE * sets)];
                }
                else
#endif
                {
                    DOUBLE x = (-AOCLFFTZ_2_PI * r * s) / ((DOUBLE)(N));
                    TW_real = cos(x);
                    TW_imag = sin(x);
                }

                out_real[dst] = src_real * TW_real - src_imag * TW_imag;
                out_imag[dst] = src_real * TW_imag + src_imag * TW_real;
            }

            visited[src_idx] = 1;
            count++;
            src_real = next_real;
            src_imag = next_imag;
            src_idx = dst_idx;
        } while (count != N && start_idx != src_idx);
    }
    FREE_ALIGN_ALLOCATED_MEM(visited);
    return TW_SUCCESS;
}

static inline INT32 twiddle_multiplier_inplace_buffered_float(
    aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");
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

    // decided based on the direction of FFT
    FLOAT *in_real, *in_imag;
    FLOAT *scratch_real;
    FLOAT *scratch_imag;
    aoclfftz_complex_f_t *in;

    if (FFT_DIR(sol->decomp_scheme->flags) == FORWARD_FFT_DIR)
    {
        in_real = (FLOAT *)sol->decomp_scheme->out_real;
        in_imag = (FLOAT *)sol->decomp_scheme->out_imag;
        scratch_real = (FLOAT *)sol->dft_bufs->scratch_space;
        scratch_imag = scratch_real + 1;
    }
    else
    {
        in_real = (FLOAT *)sol->decomp_scheme->out_imag;
        in_imag = (FLOAT *)sol->decomp_scheme->out_real;
        scratch_imag = (FLOAT *)sol->dft_bufs->scratch_space;
        scratch_real = scratch_imag + 1;
    }

    in = (aoclfftz_complex_f_t *)sol->decomp_scheme->out_real;

    aoclfftz_complex_f_t *sc =
        (aoclfftz_complex_f_t *)sol->dft_bufs->scratch_space;

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

#if IN_MEMORY_TWIDDLE_FACTORS == 1
    FLOAT *twiddle_buffer_real = (FLOAT *)sol->twiddle->TW;
    FLOAT *twiddle_buffer_imag = twiddle_buffer_real + 1;
#endif

    // process the first column
    for (INTP i = 0; i < sc_rows; i++)
    {
        FLOAT real = scratch_real[LINEAR_IDX_2D(i, 0, scratch_col_stride,
                                                scratch_row_stride)];
        FLOAT imag = scratch_imag[LINEAR_IDX_2D(i, 0, scratch_col_stride,
                                                scratch_row_stride)];

        in_real[LINEAR_IDX_2D(i, 0, col_stride, row_stride)] = real;
        in_imag[LINEAR_IDX_2D(i, 0, col_stride, row_stride)] = imag;
    }

    // process the first row
    for (INTP j = 1; j < sc_cols; j++)
    {
        FLOAT real = scratch_real[LINEAR_IDX_2D(0, j, scratch_col_stride,
                                                scratch_row_stride)];
        FLOAT imag = scratch_imag[LINEAR_IDX_2D(0, j, scratch_col_stride,
                                                scratch_row_stride)];

        in_real[LINEAR_IDX_2D(0, j, col_stride, row_stride)] = real;
        in_imag[LINEAR_IDX_2D(0, j, col_stride, row_stride)] = imag;
    }

    for (INTP i = 1; i < sc_rows; i++)
    {
        for (INTP j = 1; j < sc_cols; j++)
        {
            FLOAT TW_real, TW_imag;

#if IN_MEMORY_TWIDDLE_FACTORS == 1
            if (twiddle_buffer_real)
            {
                TW_real = twiddle_buffer_real[LINEAR_IDX_2D(
                    i, j, DATA_STRIDE, DATA_STRIDE * sets)];
                TW_imag = twiddle_buffer_imag[LINEAR_IDX_2D(
                    i, j, DATA_STRIDE, DATA_STRIDE * sets)];
            }
            else
#endif
            {
                FLOAT x = (-AOCLFFTZ_2_PIf * i * j) / ((FLOAT)(N));
                TW_real = cosf(x);
                TW_imag = sinf(x);
            }

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

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return TW_SUCCESS;
}

static inline INT32 twiddle_multiplier_inplace_buffered_double(
    aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");
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

    // decided based on the direction of FFT
    DOUBLE *in_real, *in_imag;
    DOUBLE *scratch_real;
    DOUBLE *scratch_imag;
    aoclfftz_complex_d_t *in;

    if (FFT_DIR(sol->decomp_scheme->flags) == FORWARD_FFT_DIR)
    {
        in_real = (DOUBLE *)sol->decomp_scheme->out_real;
        in_imag = (DOUBLE *)sol->decomp_scheme->out_imag;
        scratch_real = (DOUBLE *)sol->dft_bufs->scratch_space;
        scratch_imag = scratch_real + 1;
    }
    else
    {
        in_real = (DOUBLE *)sol->decomp_scheme->out_imag;
        in_imag = (DOUBLE *)sol->decomp_scheme->out_real;
        scratch_imag = (DOUBLE *)sol->dft_bufs->scratch_space;
        scratch_real = scratch_imag + 1;
    }

    in = (aoclfftz_complex_d_t *)sol->decomp_scheme->out_real;

    aoclfftz_complex_d_t *sc =
        (aoclfftz_complex_d_t *)sol->dft_bufs->scratch_space;

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

#if IN_MEMORY_TWIDDLE_FACTORS == 1
    DOUBLE *twiddle_buffer_real = (DOUBLE *)sol->twiddle->TW;
    DOUBLE *twiddle_buffer_imag = twiddle_buffer_real + 1;
#endif

    for (INTP i = 0; i < sc_rows; i++)
    {
        DOUBLE real = scratch_real[LINEAR_IDX_2D(i, 0, scratch_col_stride,
                                                 scratch_row_stride)];
        DOUBLE imag = scratch_imag[LINEAR_IDX_2D(i, 0, scratch_col_stride,
                                                 scratch_row_stride)];

        in_real[LINEAR_IDX_2D(i, 0, col_stride, row_stride)] = real;
        in_imag[LINEAR_IDX_2D(i, 0, col_stride, row_stride)] = imag;
    }

    for (INTP j = 1; j < sc_cols; j++)
    {
        DOUBLE real = scratch_real[LINEAR_IDX_2D(0, j, scratch_col_stride,
                                                 scratch_row_stride)];
        DOUBLE imag = scratch_imag[LINEAR_IDX_2D(0, j, scratch_col_stride,
                                                 scratch_row_stride)];

        in_real[LINEAR_IDX_2D(0, j, col_stride, row_stride)] = real;
        in_imag[LINEAR_IDX_2D(0, j, col_stride, row_stride)] = imag;
    }

    for (INTP i = 1; i < sc_rows; i++)
    {
        for (INTP j = 1; j < sc_cols; j++)
        {
            DOUBLE TW_real, TW_imag;

#if IN_MEMORY_TWIDDLE_FACTORS == 1
            if (twiddle_buffer_real)
            {
                TW_real = twiddle_buffer_real[LINEAR_IDX_2D(
                    i, j, DATA_STRIDE, DATA_STRIDE * sets)];
                TW_imag = twiddle_buffer_imag[LINEAR_IDX_2D(
                    i, j, DATA_STRIDE, DATA_STRIDE * sets)];
            }
            else
#endif
            {
                DOUBLE x = (-AOCLFFTZ_2_PI * i * j) / ((DOUBLE)(N));
                TW_real = cos(x);
                TW_imag = sin(x);
            }

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

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return TW_SUCCESS;
}

// An in-place twiddle multiplication function that uses the scratch space to
// store intermediate results
INT32 twiddle_multiplier_inplace(aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");
    UINT32 precision = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    INTP sets = sol->decomp_scheme->vecs[0].n;
    INTP radix = sol->decomp_scheme->dims[0].n;
    INT32 status = TW_FAILURE;

    if (sol->dft_bufs->scratch_space == NULL)
    {
        if (precision == DT_FLOAT)
        {
            status = twiddle_multiplier_inplace_nonbuffered_float(sol);
        }
        else
        {
            status = twiddle_multiplier_inplace_nonbuffered_double(sol);
        }
        AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
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
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return status;
}

INT32 twiddle_multiplier_for_real_float(aoclfftz_solution_t *sol,
                                        INTP freq_factor)
{
    UINT32 is_backward = FFT_DIR(sol->decomp_scheme->flags);
    VOID *in = sol->decomp_scheme->in_real;
    VOID *out = sol->decomp_scheme->out_real;
    INTP radix = sol->decomp_scheme->dims[0].n;
    aoclfftz_strides_t *strides = sol->strides_grp->strides;
    // base_stride is 1 since all intermediate CT stages use unit strided
    // buffers
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

    INTP num_groups = NUM_RFFT_GROUPS(sol->solver);
    INTP num_c2c_per_group = sol->solver->kernel_c2c->count / num_groups;

    FLOAT sign = is_backward ? 1.0 : -1.0;
    FLOAT *data_r = is_backward ? (FLOAT *)out : (FLOAT *)in;
    FLOAT *data_i = is_backward ? (FLOAT *)out + 1 : (FLOAT *)in + 1;

    // move the in_r to point first C2C point
    data_r += base_stride;
    data_i += base_stride;

#if IN_MEMORY_TWIDDLE_FACTORS == 1
    if (sol->twiddle->TW)
    {
        FLOAT *twiddle_buffer_real = (FLOAT *)sol->twiddle->TW;
        FLOAT *twiddle_buffer_imag = twiddle_buffer_real + 1;
        for (INTP i = 1; i < radix; i++)
        {
            for (INTP j = 0; j < num_c2c_per_group; j++)
            {
                FLOAT *cur_data_r = data_r;
                FLOAT *cur_data_i = data_i;
                // every `num_groups` columns are duplicates
                // so skip those columns to avoid redundant loads
                INTP tw_in_index =
                    LINEAR_IDX_2D(i, j, DATA_STRIDE,
                                  num_c2c_per_group * DATA_STRIDE);
                FLOAT TW_real = twiddle_buffer_real[tw_in_index];
                FLOAT TW_imag = twiddle_buffer_imag[tw_in_index];

                for (INTP k = 0; k < num_groups; k++)
                {
                    INTP data_index = stride_arr[i] + (k * vec_stride) +
                                      (j * base_stride * 2);
                    FLOAT a = cur_data_r[data_index];
                    FLOAT b = cur_data_i[data_index];
                    cur_data_r[data_index] = a * TW_real - b * TW_imag;
                    cur_data_i[data_index] = b * TW_real + a * TW_imag;
                }
            }
        }
    }
    else
#endif
    {
        for (INTP i = 0; i < num_c2c_per_group; i++)
        {
            for (INTP j = 1; j < radix; j++)
            {
                FLOAT *cur_data_r = data_r;
                FLOAT *cur_data_i = data_i;

                FLOAT x = (sign * AOCLFFTZ_2_PI * (i + 1) * j) / freq_factor;
                FLOAT TW_real = cosf(x);
                FLOAT TW_imag = sinf(x);

                INTP stride = stride_arr[j];

                for (INTP k = 0; k < num_groups; k++)
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
    return TW_SUCCESS;
}

INT32 twiddle_multiplier_for_real_double(aoclfftz_solution_t *sol,
                                         INTP freq_factor)
{
    UINT32 is_backward = FFT_DIR(sol->decomp_scheme->flags);
    VOID *in = sol->decomp_scheme->in_real;
    VOID *out = sol->decomp_scheme->out_real;
    INTP radix = sol->decomp_scheme->dims[0].n;
    aoclfftz_strides_t *strides = sol->strides_grp->strides;
    // base_stride is 1 since all intermediate CT stages use unit strided
    // buffers
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

    INTP num_groups = NUM_RFFT_GROUPS(sol->solver);
    INTP num_c2c_per_group = sol->solver->kernel_c2c->count / num_groups;
    DOUBLE sign = is_backward ? 1.0 : -1.0;
    DOUBLE *data_r = is_backward ? (DOUBLE *)out : (DOUBLE *)in;
    DOUBLE *data_i = is_backward ? (DOUBLE *)out + 1 : (DOUBLE *)in + 1;

    // move the in_r to point first C2C point
    data_r += base_stride;
    data_i += base_stride;

#if IN_MEMORY_TWIDDLE_FACTORS == 1
    if (sol->twiddle->TW)
    {
        DOUBLE *twiddle_buffer_real = (DOUBLE *)sol->twiddle->TW;
        DOUBLE *twiddle_buffer_imag = twiddle_buffer_real + 1;
        for (INTP i = 1; i < radix; i++)
        {
            for (INTP j = 0; j < num_c2c_per_group; j++)
            {
                DOUBLE *cur_data_r = data_r;
                DOUBLE *cur_data_i = data_i;
                // every `num_groups` columns are duplicates
                // so skip those columns to avoid redundant loads
                INTP tw_in_index =
                    LINEAR_IDX_2D(i, j, DATA_STRIDE,
                                  num_c2c_per_group * DATA_STRIDE);
                DOUBLE TW_real = twiddle_buffer_real[tw_in_index];
                DOUBLE TW_imag = twiddle_buffer_imag[tw_in_index];

                for (INTP k = 0; k < num_groups; k++)
                {
                    INTP data_index = stride_arr[i] + (k * vec_stride) +
                                      (j * base_stride * 2);
                    DOUBLE a = cur_data_r[data_index];
                    DOUBLE b = cur_data_i[data_index];
                    cur_data_r[data_index] = a * TW_real - b * TW_imag;
                    cur_data_i[data_index] = b * TW_real + a * TW_imag;
                }
            }
        }
    }
    else
#endif
    {
        for (INTP i = 0; i < num_c2c_per_group; i++)
        {
            for (INTP j = 1; j < radix; j++)
            {
                DOUBLE *cur_data_r = data_r;
                DOUBLE *cur_data_i = data_i;

                DOUBLE x = (sign * AOCLFFTZ_2_PI * (i + 1) * j) / freq_factor;
                DOUBLE TW_real = cos(x);
                DOUBLE TW_imag = sin(x);
                INTP stride = stride_arr[j];

                for (INTP k = 0; k < num_groups; k++)
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

// TODO: Support twiddle kernels for C2R problems
INT32 twiddle_multiplier_for_real(aoclfftz_solution_t *sol, INTP freq_factor)
{
    UINT32 dt_prec = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    if (dt_prec == DT_FLOAT)
    {
        twiddle_multiplier_for_real_float(sol, freq_factor);
    }
    else
    {
        twiddle_multiplier_for_real_double(sol, freq_factor);
    }
    return TW_SUCCESS;
}

#ifdef MULTI_THREADING
INT32 twiddle_multiplier_mt_for_real_float(aoclfftz_solution_t *sol,
                                           INTP freq_factor,
                                           INT32 n_threads_c2c_outer,
                                           INT32 n_threads_c2c_inner)
{
    UINT32 is_backward = FFT_DIR(sol->decomp_scheme->flags);
    VOID *in = sol->decomp_scheme->in_real;
    VOID *out = sol->decomp_scheme->out_real;
    INTP radix = sol->decomp_scheme->dims[0].n;
    aoclfftz_strides_t *strides = sol->strides_grp->strides;
    // base_stride is 1 since all intermediate CT stages use unit strided
    // buffers
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

    INTP num_groups = NUM_RFFT_GROUPS(sol->solver);
    INTP num_c2c_per_group = sol->solver->kernel_c2c->count / num_groups;

    FLOAT sign = is_backward ? 1.0 : -1.0;
    FLOAT *data_r = is_backward ? (FLOAT *)out : (FLOAT *)in;
    FLOAT *data_i = is_backward ? (FLOAT *)out + 1 : (FLOAT *)in + 1;

    // move the in_r to point first C2C point
    data_r += base_stride;
    data_i += base_stride;

#if IN_MEMORY_TWIDDLE_FACTORS == 1
    if (sol->twiddle->TW)
    {
        FLOAT *twiddle_buffer_real = (FLOAT *)sol->twiddle->TW;
        FLOAT *twiddle_buffer_imag = twiddle_buffer_real + 1;
        #pragma omp parallel for num_threads(n_threads_c2c_outer)
        for (INTP i = 0; i < num_c2c_per_group; i++)
        {
            #pragma omp parallel for num_threads(n_threads_c2c_inner)
            for (INTP j = 0; j < num_groups; j++)
            {
                for (INTP k = 1; k < radix; k++)
                {
                    FLOAT *cur_data_r = data_r;
                    FLOAT *cur_data_i = data_i;
                    // every `num_groups` columns are duplicates
                    // so skip those columns to avoid redundant loads
                    INTP tw_in_index = LINEAR_IDX_2D(
                        k, i, DATA_STRIDE, num_c2c_per_group * DATA_STRIDE);
                    FLOAT TW_real = twiddle_buffer_real[tw_in_index];
                    FLOAT TW_imag = twiddle_buffer_imag[tw_in_index];

                    INTP data_index = stride_arr[k] + (j * vec_stride) +
                                      (i * base_stride * 2);
                    FLOAT a = cur_data_r[data_index];
                    FLOAT b = cur_data_i[data_index];
                    cur_data_r[data_index] = a * TW_real - b * TW_imag;
                    cur_data_i[data_index] = b * TW_real + a * TW_imag;
                }
            }
        }
    }
    else
#endif
    {
        // TODO: Parallelize these loops
        for (INTP i = 0; i < num_c2c_per_group; i++)
        {
            for (INTP j = 1; j < radix; j++)
            {
                FLOAT *cur_data_r = data_r;
                FLOAT *cur_data_i = data_i;

                FLOAT x = (sign * AOCLFFTZ_2_PI * (i + 1) * j) / freq_factor;
                FLOAT TW_real = cosf(x);
                FLOAT TW_imag = sinf(x);

                INTP stride = stride_arr[j];

                for (INTP k = 0; k < num_groups; k++)
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
    return TW_SUCCESS;
}

INT32 twiddle_multiplier_mt_for_real_double(aoclfftz_solution_t *sol,
                                            INTP freq_factor,
                                            INT32 n_threads_c2c_outer,
                                            INT32 n_threads_c2c_inner)
{
    UINT32 is_backward = FFT_DIR(sol->decomp_scheme->flags);
    VOID *in = sol->decomp_scheme->in_real;
    VOID *out = sol->decomp_scheme->out_real;
    INTP radix = sol->decomp_scheme->dims[0].n;
    aoclfftz_strides_t *strides = sol->strides_grp->strides;
    // base_stride is 1 since all intermediate CT stages use unit strided
    // buffers
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

    INTP num_groups = NUM_RFFT_GROUPS(sol->solver);
    INTP num_c2c_per_group = sol->solver->kernel_c2c->count / num_groups;
    DOUBLE sign = is_backward ? 1.0 : -1.0;
    DOUBLE *data_r = is_backward ? (DOUBLE *)out : (DOUBLE *)in;
    DOUBLE *data_i = is_backward ? (DOUBLE *)out + 1 : (DOUBLE *)in + 1;

    // move the in_r to point first C2C point
    data_r += base_stride;
    data_i += base_stride;

#if IN_MEMORY_TWIDDLE_FACTORS == 1
    if (sol->twiddle->TW)
    {
        DOUBLE *twiddle_buffer_real = (DOUBLE *)sol->twiddle->TW;
        DOUBLE *twiddle_buffer_imag = twiddle_buffer_real + 1;
        #pragma omp parallel for num_threads(n_threads_c2c_outer)
        for (INTP i = 0; i < num_c2c_per_group; i++)
        {
            #pragma omp parallel for num_threads(n_threads_c2c_inner)
            for (INTP j = 0; j < num_groups; j++)
            {
                for (INTP k = 1; k < radix; k++)
                {
                    DOUBLE *cur_data_r = data_r;
                    DOUBLE *cur_data_i = data_i;
                    // every `num_groups` columns are duplicates
                    // so skip those columns to avoid redundant loads
                    INTP tw_in_index = LINEAR_IDX_2D(
                        k, i, DATA_STRIDE, num_c2c_per_group * DATA_STRIDE);
                    DOUBLE TW_real = twiddle_buffer_real[tw_in_index];
                    DOUBLE TW_imag = twiddle_buffer_imag[tw_in_index];

                    INTP data_index = stride_arr[k] + (j * vec_stride) +
                                      (i * base_stride * 2);
                    DOUBLE a = cur_data_r[data_index];
                    DOUBLE b = cur_data_i[data_index];
                    cur_data_r[data_index] = a * TW_real - b * TW_imag;
                    cur_data_i[data_index] = b * TW_real + a * TW_imag;
                }
            }
        }
    }
    else
#endif
    {
        // TODO: Parallelize these loops
        for (INTP i = 0; i < num_c2c_per_group; i++)
        {
            for (INTP j = 1; j < radix; j++)
            {
                DOUBLE *cur_data_r = data_r;
                DOUBLE *cur_data_i = data_i;

                DOUBLE x = (sign * AOCLFFTZ_2_PI * (i + 1) * j) / freq_factor;
                DOUBLE TW_real = cos(x);
                DOUBLE TW_imag = sin(x);
                INTP stride = stride_arr[j];

                for (INTP k = 0; k < num_groups; k++)
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

INT32 twiddle_multiplier_mt_for_real(aoclfftz_solution_t *sol, INTP freq_factor,
                                     INT32 n_threads_c2c_outer,
                                     INT32 n_threads_c2c_inner)
{
    UINT32 dt_prec = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    if (dt_prec == DT_FLOAT)
    {
        twiddle_multiplier_mt_for_real_float(
            sol, freq_factor, n_threads_c2c_outer, n_threads_c2c_inner);
    }
    else
    {
        twiddle_multiplier_mt_for_real_double(
            sol, freq_factor, n_threads_c2c_outer, n_threads_c2c_inner);
    }
    return TW_SUCCESS;
}
#endif

#if IN_MEMORY_TWIDDLE_FACTORS == 1
// Setup a twiddle buffer of size `radix x n_tw_batches`
VOID compute_twiddle_buffer_float(VOID *twiddle_buffer, INTP radix,
                                  INTP n_tw_batches)
{
    FLOAT *twiddle_buffer_real = (FLOAT *)twiddle_buffer;
    FLOAT *twiddle_buffer_imag = twiddle_buffer_real + 1;

    FLOAT angle_base = -AOCLFFTZ_2_PIf / (FLOAT)(radix * n_tw_batches);
    INTP c_stride = 1 * DATA_STRIDE;
    INTP r_stride = n_tw_batches * DATA_STRIDE;

    for (INTP i = 0; i < radix; ++i)
    {
        twiddle_buffer_real[LINEAR_IDX_2D(i, 0, c_stride, r_stride)] = 1.0f;
        twiddle_buffer_imag[LINEAR_IDX_2D(i, 0, c_stride, r_stride)] = 0.0f;
    }
    for (INTP j = 1; j < n_tw_batches; ++j)
    {
        twiddle_buffer_real[LINEAR_IDX_2D(0, j, c_stride, r_stride)] = 1.0f;
        twiddle_buffer_imag[LINEAR_IDX_2D(0, j, c_stride, r_stride)] = 0.0f;
    }

    for (INTP i = 1; i < radix; ++i)
    {
        for (INTP j = 1; j < n_tw_batches; ++j)
        {
            FLOAT angle = angle_base * i * j;
            FLOAT sin_val = sinf(angle);
            FLOAT cos_val = cosf(angle);
            twiddle_buffer_real[LINEAR_IDX_2D(i, j, c_stride, r_stride)] =
                cos_val;
            twiddle_buffer_imag[LINEAR_IDX_2D(i, j, c_stride, r_stride)] =
                sin_val;
        }
    }
}

// Setup a twiddle buffer of size `radix x n_tw_batches`
VOID compute_twiddle_buffer_double(VOID *twiddle_buffer, INTP radix,
                                   INTP n_tw_batches)
{
    DOUBLE *twiddle_buffer_real = (DOUBLE *)twiddle_buffer;
    DOUBLE *twiddle_buffer_imag = twiddle_buffer_real + 1;

    DOUBLE angle_base = -AOCLFFTZ_2_PI / (DOUBLE)(radix * n_tw_batches);
    INTP c_stride = 1 * DATA_STRIDE;
    INTP r_stride = n_tw_batches * DATA_STRIDE;

    for (INTP i = 0; i < radix; ++i)
    {
        twiddle_buffer_real[LINEAR_IDX_2D(i, 0, c_stride, r_stride)] = 1.0;
        twiddle_buffer_imag[LINEAR_IDX_2D(i, 0, c_stride, r_stride)] = 0.0;
    }
    for (INTP j = 1; j < n_tw_batches; ++j)
    {
        twiddle_buffer_real[LINEAR_IDX_2D(0, j, c_stride, r_stride)] = 1.0;
        twiddle_buffer_imag[LINEAR_IDX_2D(0, j, c_stride, r_stride)] = 0.0;
    }

    for (INTP i = 1; i < radix; ++i)
    {
        for (INTP j = 1; j < n_tw_batches; ++j)
        {
            DOUBLE angle = angle_base * i * j;
            DOUBLE sin_val = sin(angle);
            DOUBLE cos_val = cos(angle);
            twiddle_buffer_real[LINEAR_IDX_2D(i, j, c_stride, r_stride)] =
                cos_val;
            twiddle_buffer_imag[LINEAR_IDX_2D(i, j, c_stride, r_stride)] =
                sin_val;
        }
    }
}

VOID compute_twiddle_buffer(VOID *twiddle_buffer, INTP radix, INTP n_tw_batches,
                            UINT32 dt_prec)
{
    if (dt_prec == DT_FLOAT)
    {
        compute_twiddle_buffer_float(twiddle_buffer, radix, n_tw_batches);
    }
    else
    {
        compute_twiddle_buffer_double(twiddle_buffer, radix, n_tw_batches);
    }
}
#endif

#if IN_MEMORY_TWIDDLE_FACTORS == 1
VOID compute_twiddle_buffer_real_float(VOID *twiddle_buffer, INTP radix,
                                       INTP num_c2c_per_group, INTP num_groups,
                                       INTP freq_factor, UINT8 dir)
{
    FLOAT *twiddle_buffer_real = (FLOAT *)twiddle_buffer;
    FLOAT *twiddle_buffer_imag = twiddle_buffer_real + 1;

    FLOAT sign = (dir == BACKWARD_FFT_DIR) ? 1.0 : -1.0;
    FLOAT angle_base = (sign * AOCLFFTZ_2_PI) / (FLOAT)freq_factor;
    INTP c_stride = DATA_STRIDE;
    INTP r_stride = num_c2c_per_group * DATA_STRIDE;

    // set the first column of the twiddle matrix to (1 + 0i)
    for (INTP j = 0; j < num_c2c_per_group; ++j)
    {
        INTP buffer_index =
            LINEAR_IDX_2D(0, j, c_stride, r_stride);
        twiddle_buffer_real[buffer_index] = 1.0;
        twiddle_buffer_imag[buffer_index] = 0.0;
    }

    for (INTP i = 1; i < radix; ++i)
    {
        for (INTP j = 0; j < num_c2c_per_group; ++j)
        {
           // twiddle matrix size `r x num_c2c_per_group`
            FLOAT angle = angle_base * i * (j + 1);
            FLOAT sin_val = sinf(angle);
            FLOAT cos_val = cosf(angle);
            INTP buffer_index =
                LINEAR_IDX_2D(i, j, c_stride, r_stride);
            twiddle_buffer_real[buffer_index] = cos_val;
            twiddle_buffer_imag[buffer_index] = sin_val;
        }
    }
}

VOID compute_twiddle_buffer_real_double(VOID *twiddle_buffer, INTP radix,
                                        INTP num_c2c_per_group, INTP num_groups,
                                        INTP freq_factor, UINT8 dir)
{
    DOUBLE *twiddle_buffer_real = (DOUBLE *)twiddle_buffer;
    DOUBLE *twiddle_buffer_imag = twiddle_buffer_real + 1;

    DOUBLE sign = (dir == BACKWARD_FFT_DIR) ? 1.0 : -1.0;
    DOUBLE angle_base = (sign * AOCLFFTZ_2_PI) / (DOUBLE)freq_factor;
    INTP c_stride = DATA_STRIDE;
    INTP r_stride = num_c2c_per_group * DATA_STRIDE;

    // set the first column of the twiddle matrix to (1 + 0i)
    for (INTP j = 0; j < num_c2c_per_group; ++j)
    {
        INTP buffer_index =
            LINEAR_IDX_2D(0, j, c_stride, r_stride);
        twiddle_buffer_real[buffer_index] = 1.0;
        twiddle_buffer_imag[buffer_index] = 0.0;
    }

    for (INTP i = 1; i < radix; ++i)
    {
        for (INTP j = 0; j < num_c2c_per_group; ++j)
        {
            DOUBLE angle = angle_base * i * (j + 1);
            DOUBLE sin_val = sin(angle);
            DOUBLE cos_val = cos(angle);
            // twiddle matrix size `r x num_c2c_per_group`
            INTP buffer_index =
                LINEAR_IDX_2D(i, j, c_stride, r_stride);
            twiddle_buffer_real[buffer_index] = cos_val;
            twiddle_buffer_imag[buffer_index] = sin_val;
        }
    }
}

VOID compute_twiddle_buffer_real(VOID *twiddle_buffer, INTP radix,
                                 INTP num_c2c_per_group, INTP num_groups,
                                 INTP freq_factor, UINT8 dir, UINT32 dt_prec)
{
    if (dt_prec == DT_FLOAT)
    {
        compute_twiddle_buffer_real_float(twiddle_buffer, radix,
                                          num_c2c_per_group, num_groups,
                                          freq_factor, dir);
    }
    else
    {
        compute_twiddle_buffer_real_double(twiddle_buffer, radix,
                                           num_c2c_per_group, num_groups,
                                           freq_factor, dir);
    }
}

static VOID compute_sr_twiddle_buffer_float(VOID *twiddle_buffer, INTP n)
{
    INTP n4 = n / 4;
    aoclfftz_complex_f_t *tw = (aoclfftz_complex_f_t *)twiddle_buffer;

    /* Compute twiddle factors for split-radix */
    /* For each k, we need W_n^k and W_n^(3k) */
    for (INTP k = 0; k < n4; k++)
    {
        FLOAT angle1 = -AOCLFFTZ_2_PIf * k / n;  /* W_n^k */
        FLOAT angle3 = -3.0f * AOCLFFTZ_2_PIf * k / n;  /* W_n^(3k) */

        /* Store W^k at even indices */
        tw[k * 2].real = cosf(angle1);
        tw[k * 2].imag = sinf(angle1);

        /* Store W^(3k) at odd indices */
        tw[k * 2 + 1].real = cosf(angle3);
        tw[k * 2 + 1].imag = sinf(angle3);
    }
}

static VOID compute_sr_twiddle_buffer_double(VOID *twiddle_buffer, INTP n)
{
    INTP n4 = n / 4;
    aoclfftz_complex_d_t *tw = (aoclfftz_complex_d_t *)twiddle_buffer;

    /* Compute twiddle factors for split-radix */
    /* For each k, we need W_n^k and W_n^(3k) */
    for (INTP k = 0; k < n4; k++)
    {
        DOUBLE angle1 = -AOCLFFTZ_2_PI * k / n;  /* W_n^k */
        DOUBLE angle3 = -3.0 * AOCLFFTZ_2_PI * k / n;  /* W_n^(3k) */

        /* Store W^k at even indices */
        tw[k * 2].real = cos(angle1);
        tw[k * 2].imag = sin(angle1);

        /* Store W^(3k) at odd indices */
        tw[k * 2 + 1].real = cos(angle3);
        tw[k * 2 + 1].imag = sin(angle3);
    }
}

VOID compute_sr_twiddle_buffer(VOID *twiddle_buffer, INTP n, UINT32 dt_prec)
{
    if (dt_prec == DT_FLOAT)
    {
        compute_sr_twiddle_buffer_float(twiddle_buffer, n);
    }
    else
    {
        compute_sr_twiddle_buffer_double(twiddle_buffer, n);
    }
}
#endif
