/**
 * Copyright (C) 2025, Advanced Micro Devices. All rights reserved.
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

/** @file realfft_utils.c
 *
 *  @brief RealFFT utility functions.
 *
 *  This file contains the function definitions of utility functions related to
 *  RealFFT problems.
 *
 *  @author Srirammaswamy Srinivasan
 *  @author Avinash Thakur
 */

#include "core/common/realfft_utils.h"
#include "core/common/strides.h"
#include "core/kernels/kernel.h"

/**
 * @brief Compute complex conjugates for a set of selective points.
 *        Conjugate the second half of the complex numbers in the input buffer.
 *
 * Example:
 * Given an input for radix 4 (4 complex numbers in interleaved format)
 * input  -> (1,  2), (3,  4), (5,  6), (7,  8)
 * output -> (1,  2), (3,  4), (5, -6), (7, -8)
 *
 * @param data in/out data buffer
 * @param radix radix of the C2C kernel
 * @param n batch of the C2C kernels
 * @param strides strides array for the buffer
 * @param v_out_stride vector stride for the buffer
 * @param prec precision flag (DT_FLOAT or DT_DOUBLE)
 * @return VOID
 */
VOID compute_conjugates(VOID *data, INTP radix, INTP n, INTP *strides,
                        INTP v_out_stride, UINT32 prec)
{
    INTP points = (radix + 1) >> 1; // ceil div
    if (prec == DT_FLOAT)
    {
        FLOAT *data_i = (FLOAT *)data + 1;
        for (INTP i = 0; i < n; i++)
        {
            for (INTP j = points; j < radix; j++)
            {
                data_i[strides[j]] = -data_i[strides[j]];
            }
            data_i += v_out_stride;
        }
    }
    else
    {
        DOUBLE *data_i = (DOUBLE *)data + 1;
        for (INTP i = 0; i < n; i++)
        {
            for (INTP j = points; j < radix; j++)
            {
                data_i[strides[j]] = -data_i[strides[j]];
            }
            data_i += v_out_stride;
        }
    }
}

/**
 * @brief Sets imaginary parts of DC and Nyquist frequencies to zero for
 * batched R2C transforms.
 *
 * For R2C (real forward) problems, this function sets the imaginary part of
 * the first complex number (DC component) and the complex number
 * corresponding to the Nyquist frequency to zero. This is a property of the
 * discrete Fourier transform of a real-valued signal. This function handles
 * batched transforms.
 *
 * @param sol [in, out] The solution object containing problem details,
 *            including the output buffer to be modified.
 */
VOID set_zero_for_dc_and_nyquist_batched(aoclfftz_solution_t *sol)
{
    UINTP transform_len = sol->decomp_scheme->dims[0].n;
    UINTP num_batches = sol->decomp_scheme->vecs[0].n;
    UINTP out_stride = sol->decomp_scheme->dims[0].out_stride;
    UINTP v_out_stride = sol->decomp_scheme->vecs[0].out_stride * 2;
    UINTP nyquist_im_offset =
        transform_len % 2 == 0 ? transform_len * out_stride + 1 : 1;
    if (DT_PRECISION_FLAG(sol->decomp_scheme->flags) == DT_FLOAT)
    {
        FLOAT *out = (FLOAT *)sol->decomp_scheme->out_real;
        for (UINTP b = 0; b < num_batches; b++)
        {
            out[1] = 0.0f;
            out[nyquist_im_offset] = 0.0f;
            out += v_out_stride;
        }
    }
    else
    {
        DOUBLE *out = (DOUBLE *)sol->decomp_scheme->out_real;
        for (UINTP b = 0; b < num_batches; b++)
        {
            out[1] = 0.0;
            out[nyquist_im_offset] = 0.0;
            out += v_out_stride;
        }
    }
}

/**
 * @brief Sets imaginary parts of DC and Nyquist frequencies to zero for R2C
 * transforms.
 *
 * For R2C (real forward) problems, this function sets the imaginary part of
 * the first complex number (DC component) and the complex number
 * corresponding to the Nyquist frequency to zero. This is a property of the
 * discrete Fourier transform of a real-valued signal.
 *
 * @param sol [in, out] The solution object containing problem details,
 *            including the output buffer to be modified.
 */
VOID set_zero_for_dc_and_nyquist(aoclfftz_solution_t *sol)
{
    // For R2C (real forward) problems, set the imaginary part of first and
    // last points in half-complex buffer to 0.
    INTP transform_len =
        sol->decomp_scheme->dims[0].n * sol->decomp_scheme->vecs[0].n;
    INTP out_stride = sol->decomp_scheme->dims[0].out_stride;
    INTP nyquist_im_offset =
        transform_len % 2 == 0 ? transform_len * out_stride + 1 : 1;

    if (DT_PRECISION_FLAG(sol->decomp_scheme->flags) == DT_FLOAT)
    {
        FLOAT *out = (FLOAT *)sol->decomp_scheme->out_real;
        out[1] = 0.0f;
        out[nyquist_im_offset] = 0.0f;
    }
    else
    {
        DOUBLE *out = (DOUBLE *)sol->decomp_scheme->out_real;
        out[1] = 0.0;
        out[nyquist_im_offset] = 0.0;
    }
}
