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
                FLOAT x = (-2 * M_PI * r * s) / ((FLOAT)(N));
                FLOAT TW_real = cos(x);
                FLOAT TW_imag = sin(x);
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
                DOUBLE x = (-2 * M_PI * r * s) / ((DOUBLE)(N));
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