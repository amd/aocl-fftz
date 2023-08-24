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

 /** @file fft2c.c
 *
 *  @brief Radix-2 FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-2 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author S. Biplab Raut
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = { { 0, 4, 12, 12, 0, 0 },
                                                      { 0, 4, 12, 12, 0, 0 } };
ops_cycles_t get_ops_cnt_fft3c(INT32 precision)
{
    return ops_cnt[precision - 1];
}

kfft_ register_kernel_fft3c(INT32 precision)
{
    if (precision == DT_FLOAT)
        return fft3c_fp32;
    else if (precision == DT_DOUBLE)
        return fft3c_fp64;
    else
        return NULL;
}

VOID fft3c_fp32(VOID *in_real, VOID *in_imag,
                VOID *out_real, VOID *out_imag,
                ptrdiff_t n,
                aoclfftz_strides_t *strides)
{
    const FLOAT CRTM_3_1 = +0.500000000000000000000000000000000000000000000;
    const FLOAT CRTM_3_2 = +0.866025403784438646763723170752936183471402627;

    FLOAT *in_r = (FLOAT*)in_real;
    FLOAT *in_i = (FLOAT*)in_imag;
    FLOAT *out_r = (FLOAT*)out_real;
    FLOAT* out_i = (FLOAT*)out_imag;
    INTP in_stride = (strides->in_stride << 1);
    INTP out_stride = (strides->out_stride << 1);
    INTP v_in_stride = (strides->v_in_stride << 1);
    INTP v_out_stride = (strides->v_out_stride << 1);
    INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        FLOAT v1r, v1i, v2r, v2i, v3r, v3i, tv1r, tv1i, tv2r, tv2i, tv3r, tv3i;
        //Input point 1: x(0)
        v1r = *in_r;
        v1i = *in_i;
        //Input point 2: x(1)
        v2r = in_r[in_stride];
        v2i = in_i[in_stride];
        //Input point 2: x(1)
        v3r = in_r[(in_stride << 1)];
        v3i = in_i[(in_stride << 1)];

        tv3r = v2r + v3r;
        tv3i = v2i - v3i;
        tv1r = CRTM_3_1 * (tv3r);
        tv1i = CRTM_3_2 * (tv3i);
        tv2r = CRTM_3_2 * (v3r - v2r);
        tv2i = CRTM_3_1 * (v2i + v3i);

        //Output point 1: X(0)
        *out_r = v1r + tv3r;
        *out_i = v1i + tv3i;
        //Output point 2: X(1)
        tv3r = v1r + tv1r;
        tv3i = tv2r - v1i;
        out_r[out_stride] = tv3r + tv1i;
        out_i[out_stride] = tv3i + tv2i;
        //Output point 3: X(2)
        out_r[(out_stride << 1)] = tv3r - tv1i;
        out_i[(out_stride << 1)] = tv3i - tv2i;

        in_r = in_r + v_in_stride;
        in_i = in_i + v_in_stride;
        out_r = out_r + v_out_stride;
        out_i = out_i + v_out_stride;
    }
}

VOID fft3c_fp64(VOID* in_real, VOID* in_imag,
                VOID* out_real, VOID* out_imag,
                ptrdiff_t n,
                aoclfftz_strides_t *strides)
{
    const DOUBLE CRTM_3_1 = +0.500000000000000000000000000000000000000000000;
    const DOUBLE CRTM_3_2 = +0.866025403784438646763723170752936183471402627;

    DOUBLE* in_r = (DOUBLE*)in_real;
    DOUBLE* in_i = (DOUBLE*)in_imag;
    DOUBLE* out_r = (DOUBLE*)out_real;
    DOUBLE* out_i = (DOUBLE*)out_imag;
    INTP in_stride = (strides->in_stride << 1);
    INTP out_stride = (strides->out_stride << 1);
    INTP v_in_stride = (strides->v_in_stride << 1);
    INTP v_out_stride = (strides->v_out_stride << 1);
    INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        DOUBLE v1r, v1i, v2r, v2i, v3r, v3i, tv1r, tv1i, tv2r, tv2i, tv3r, tv3i;
        //Input point 1: x(0)
        v1r = *in_r;
        v1i = *in_i;
        //Input point 2: x(1)
        v2r = in_r[in_stride];
        v2i = in_i[in_stride];
        //Input point 2: x(1)
        v3r = in_r[(in_stride << 1)];
        v3i = in_i[(in_stride << 1)];

        tv3r = v2r + v3r;
        tv3i = v2i - v3i;
        tv1r = CRTM_3_1 * (tv3r);
        tv1i = CRTM_3_2 * (tv3i);
        tv2r = CRTM_3_2 * (v3r - v2r);
        tv2i = CRTM_3_1 * (v2i + v3i);

        //Output point 1: X(0)
        *out_r = v1r + tv3r;
        *out_i = v1i + tv3i;
        //Output point 2: X(1)
        tv3r = v1r + tv1r;
        tv3i = tv2r - v1i;
        out_r[out_stride] = tv3r + tv1i;
        out_i[out_stride] = tv3i + tv2i;
        //Output point 3: X(2)
        out_r[(out_stride << 1)] = tv3r - tv1i;
        out_i[(out_stride << 1)] = tv3i - tv2i;

        in_r = in_r + v_in_stride;
        in_i = in_i + v_in_stride;
        out_r = out_r + v_out_stride;
        out_i = out_i + v_out_stride;
    }
}