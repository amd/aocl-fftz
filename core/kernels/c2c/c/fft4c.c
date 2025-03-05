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

/** @file fft4c.c
 *
 *  @brief Radix-4 FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-4 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author S. Biplab Raut
 *  @author Varun Sanjay
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 0, 16, 16, 0, 0},
                                                     {0, 0, 16, 16, 0, 0}};

ops_cycles_t get_ops_cnt_fft4c(UINT8 precision, UINT8 direction)
{
    if (precision == DT_FLOAT)
    {
        return ops_cnt[0];
    }
    else
    {
        return ops_cnt[1];
    }
}

static VOID fft4c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                       VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                       UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    DOUBLE *in_r = (DOUBLE *)in_real;
    DOUBLE *in_i = (DOUBLE *)in_imag;
    DOUBLE *out_r = (DOUBLE *)out_real;
    DOUBLE *out_i = (DOUBLE *)out_imag;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;
    INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        DOUBLE v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, tvri, tvir, tvii, tvrr,
               v13r, v24r, v13i, v24i;

        // Input point 1: x(0)
        v1r = *in_r;
        v1i = *in_i;

        // Input point 2: x(1)
        v2r = in_r[in_strides[1]];
        v2i = in_i[in_strides[1]];

        // Input point 3: x(2)
        v3r = in_r[in_strides[2]];
        v3i = in_i[in_strides[2]];

        // Input point 4: x(3)
        v4r = in_r[in_strides[3]];
        v4i = in_i[in_strides[3]];

        v13r = v1r + v3r;
        v24r = v2r + v4r;
        v13i = v1i + v3i;
        v24i = v2i + v4i;

        // Output point 1: X(0)
        *out_r = v13r + v24r;
        *out_i = v13i + v24i;

        // Output point 2: X(1)
        tvri = v4i - v2i;
        tvir = v4r - v2r;

        tvrr = v1r - v3r;
        tvii = v1i - v3i;

        out_r[out_strides[1]] = tvrr - tvri;
        out_i[out_strides[1]] = tvii + tvir;

        // Output point 4: X(3)
        out_r[out_strides[3]] = tvrr + tvri;
        out_i[out_strides[3]] = tvii - tvir;

        // Output point 3: X(2)
        tvrr = v13r - v24r;
        tvii = v13i - v24i;

        out_r[out_strides[2]] = tvrr;
        out_i[out_strides[2]] = tvii;

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID fft4c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                       VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                       UINT8 flag)
{
    FLOAT *in_r = (FLOAT *)in_real;
    FLOAT *in_i = (FLOAT *)in_imag;
    FLOAT *out_r = (FLOAT *)out_real;
    FLOAT *out_i = (FLOAT *)out_imag;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;
    INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        FLOAT v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, tvri, tvir, tvii, tvrr,
              v13r, v24r, v13i, v24i;

        // Input point 1: x(0)
        v1r = *in_r;
        v1i = *in_i;

        // Input point 2: x(1)
        v2r = in_r[in_strides[1]];
        v2i = in_i[in_strides[1]];

        // Input point 3: x(2)
        v3r = in_r[in_strides[2]];
        v3i = in_i[in_strides[2]];

        // Input point 4: x(3)
        v4r = in_r[in_strides[3]];
        v4i = in_i[in_strides[3]];

        v13r = v1r + v3r;
        v24r = v2r + v4r;
        v13i = v1i + v3i;
        v24i = v2i + v4i;

        // Output point 1: X(0)
        *out_r = v13r + v24r;
        *out_i = v13i + v24i;

        // Output point 2: X(1)
        tvri = v4i - v2i;
        tvir = v4r - v2r;

        tvrr = v1r - v3r;
        tvii = v1i - v3i;

        out_r[out_strides[1]] = tvrr - tvri;
        out_i[out_strides[1]] = tvii + tvir;

        // Output point 4: X(3)
        out_r[out_strides[3]] = tvrr + tvri;
        out_i[out_strides[3]] = tvii - tvir;

        // Output point 3: X(2)
        tvrr = v13r - v24r;
        tvii = v13i - v24i;

        out_r[out_strides[2]] = tvrr;
        out_i[out_strides[2]] = tvii;

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

kfft_ register_kernel_fft4c(UINT8 precision, UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return fft4c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft4c_fp64;
    }
    else
    {
        return NULL;
    }
}
