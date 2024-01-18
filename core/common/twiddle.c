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
 */

#include <math.h>
#include "core/common/twiddle.h"
#include "utils/allocator.h"

// TODO : Add support for In-Place problems
INT32 twiddle_multiplier(aoclfftz_solution_t* sol)
{
    UINT32 precision = DT_PRECISION_FLAG(sol->decomp_scheme->flags);

    if(precision == DT_FLOAT)
    {
        FLOAT *in_real  = (FLOAT *)sol->decomp_scheme->in_real;
        FLOAT *in_imag  = (FLOAT *)sol->decomp_scheme->in_imag;
        FLOAT *out_real = (FLOAT *)sol->decomp_scheme->out_real;
        FLOAT *out_imag = (FLOAT *)sol->decomp_scheme->out_imag;

        INTP sets = sol->decomp_scheme->vecs[0].n;
        INTP radix = sol->decomp_scheme->dims[0].n;
        INTP N = sets * radix; // actual problem length

        // out-of-order -> out-of-order multiplication of twiddle
        // output_buffer = output_buffer * twiddle_val
        INTP in_stride = sol->decomp_scheme->dims[0].in_stride * DATA_STRIDE;
        INTP out_stride = sol->decomp_scheme->dims[0].out_stride * DATA_STRIDE;
        INTP v_in_stride = sol->decomp_scheme->vecs[0].in_stride * DATA_STRIDE;
        INTP v_out_stride = sol->decomp_scheme->vecs[0].out_stride * DATA_STRIDE;

        for(INTP s = 0; s < sets; s++)
        {
            INTP out_idx = 0;
            INTP in_idx = 0;
            for(INTP r = 0; r < radix; r++)
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

            in_real  += v_in_stride;
            in_imag  += v_in_stride;
            out_real += v_out_stride;
            out_imag += v_out_stride;
        }
    }
    else
    {
        DOUBLE *in_real  = (DOUBLE *)sol->decomp_scheme->in_real;
        DOUBLE *in_imag  = (DOUBLE *)sol->decomp_scheme->in_imag;
        DOUBLE *out_real = (DOUBLE *)sol->decomp_scheme->out_real;
        DOUBLE *out_imag = (DOUBLE *)sol->decomp_scheme->out_imag;

        INTP sets = sol->decomp_scheme->vecs[0].n;
        INTP radix = sol->decomp_scheme->dims[0].n;
        INTP N = sets * radix; // actual problem length

        // out-of-order -> out-of-order multiplication of twiddle
        // output_buffer = output_buffer * twiddle_val
        INTP in_stride = sol->decomp_scheme->dims[0].in_stride * DATA_STRIDE;
        INTP out_stride = sol->decomp_scheme->dims[0].out_stride * DATA_STRIDE;
        INTP v_in_stride = sol->decomp_scheme->vecs[0].in_stride * DATA_STRIDE;
        INTP v_out_stride = sol->decomp_scheme->vecs[0].out_stride * DATA_STRIDE;

        for(INTP s = 0; s < sets; s++)
        {
            INTP out_idx = 0;
            INTP in_idx = 0;
            for(INTP r = 0; r < radix; r++)
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

            in_real  += v_in_stride;
            in_imag  += v_in_stride;
            out_real += v_out_stride;
            out_imag += v_out_stride;
        }
    }

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
INT32 twiddle_multiplier_inplace(aoclfftz_solution_t* sol)
{
    UINT32 precision = DT_PRECISION_FLAG(sol->decomp_scheme->flags);

    if(precision == DT_FLOAT)
    {
        // both buffers point to same memory location in an inplace problem
        FLOAT *in_real  = (FLOAT *)sol->decomp_scheme->out_real;
        FLOAT *in_imag  = (FLOAT *)sol->decomp_scheme->out_imag;
        FLOAT *out_real = (FLOAT *)sol->decomp_scheme->in_real;
        FLOAT *out_imag = (FLOAT *)sol->decomp_scheme->in_imag;

        INTP sets  = sol->decomp_scheme->vecs[0].n;
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
        while(count != N)
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

                FLOAT x = (-AOCLFFTZ_2_PIf * r * s) / ((FLOAT)(N));
                FLOAT TW_real = cosf(x);
                FLOAT TW_imag = sinf(x);

                out_real[dst] = src_real * TW_real - src_imag * TW_imag;
                out_imag[dst] = src_real * TW_imag + src_imag * TW_real;

                visited[src_idx] = 1; count++;
                src_real = next_real;
                src_imag = next_imag;
                src_idx = dst_idx;
            } while (count != N && start_idx != src_idx);
        }
    }
    else
    {
        // both buffers point to same memory location in an inplace problem
        DOUBLE *in_real  = (DOUBLE *)sol->decomp_scheme->out_real;
        DOUBLE *in_imag  = (DOUBLE *)sol->decomp_scheme->out_imag;
        DOUBLE *out_real = (DOUBLE *)sol->decomp_scheme->in_real;
        DOUBLE *out_imag = (DOUBLE *)sol->decomp_scheme->in_imag;

        INTP sets  = sol->decomp_scheme->vecs[0].n;
        INTP radix = sol->decomp_scheme->dims[0].n;
        INTP N = sets * radix;

        // buffer to mark processed index
        INT8 *visited = NULL;;
        ALLOC_ALIGN_INIT(visited, INT8, N * sizeof(INT8));

        INTP in_stride =
                (sol->decomp_scheme->dims[0].in_stride / sets) * DATA_STRIDE;
        INTP out_stride =
                (sol->decomp_scheme->dims[0].out_stride / sets) * DATA_STRIDE;

        INTP count = 0;
        INTP start_idx = 0;
        while(count != N)
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

                DOUBLE x = (-AOCLFFTZ_2_PI * r * s) / ((DOUBLE)(N));
                DOUBLE TW_real = cos(x);
                DOUBLE TW_imag = sin(x);

                out_real[dst] = src_real * TW_real - src_imag * TW_imag;
                out_imag[dst] = src_real * TW_imag + src_imag * TW_real;

                visited[src_idx] = 1; count++;
                src_real = next_real;
                src_imag = next_imag;
                src_idx = dst_idx;
            } while (count != N && start_idx != src_idx);
        }

    }

    return TW_SUCCESS;
}
