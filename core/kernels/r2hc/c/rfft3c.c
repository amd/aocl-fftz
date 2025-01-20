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

/** @file rfft3c.c
 *
 *  @brief Radix-3 r2hc Real-FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-3 real-to-halfcomplex implementations
 *  using scalar operations for single-precision and double-precision inputs.
 *
 *  @author Srirammaswamy Srinivasan
 */

#include "core/kernels/kernel.h"

// Forward and backward opscount are identical for float and double
static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 2, 4, 6, 0, 0},
                                                     {0, 2, 4, 6, 0, 0}};

ops_cycles_t get_ops_cnt_r2hc_rfft3c(UINT8 precision, UINT8 direction)
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

static VOID r2hc_rfft3c_fp32_fwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                 VOID *out_imag, INTP n,
                                 aoclfftz_strides_t *strides, UINT8 flag)
{
    const FLOAT CRTM_3_1 = 0.500000000000000000000000000000000000000000000f;
    const FLOAT CRTM_3_2 = 0.866025403784438646763723170752936183471402627f;

    FLOAT *in = (FLOAT *)in_real;
    FLOAT *out = (FLOAT *)out_real;
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
        FLOAT v0, v1, v2;
        FLOAT t0, t1, t2, t3;

        // Input point 1: x(0)
        v0 = *in;
        // Input point 2: x(1)
        v1 = in[in_strides[1]];
        // Input point 3: x(2)
        v2 = in[in_strides[2]];

        t0 = v2 + v1;
        t1 = v2 - v1;

        t2 = CRTM_3_1 * t0;
        t3 = CRTM_3_2 * t1;

        // Output point 1: X(0)
        *out = v0 + t0;

        // Output point 2: X(1)
        out[out_strides[1]] = v0 - t2;

        // Output point 3: X(2)
        out[out_strides[2]] = t3;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
}

static VOID r2hc_rfft3c_fp32_bwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                 VOID *out_imag, INTP n,
                                 aoclfftz_strides_t *strides, UINT8 flag)
{
    const FLOAT CRTM_3_1 = 1.732050807568877293527446341505872366942805254f;
    const FLOAT CRTM_3_2 = 2.0f;

    FLOAT *in = (FLOAT *)in_real;
    FLOAT *out = (FLOAT *)out_real;
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
        FLOAT v0, v1, v2;
        FLOAT t0, t1;

        // Input point 1: x(0)
        v0 = *in;
        // Input point 2: x(1)
        v1 = in[in_strides[1]];
        // Input point 3: x(2)
        v2 = in[in_strides[2]];

        t0 = v0 - v1;
        t1 = CRTM_3_1 * v2;

        // Output point 1: X(0)
        *out = v0 + CRTM_3_2 * v1;

        // Output point 2: X(1)
        out[out_strides[1]] = t0 - t1;

        // Output point 3: X(2)
        out[out_strides[2]] = t0 + t1;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
}

static VOID r2hc_rfft3c_fp64_fwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                 VOID *out_imag, INTP n,
                                 aoclfftz_strides_t *strides, UINT8 flag)
{
    const DOUBLE CRTM_3_1 = 0.500000000000000000000000000000000000000000000;
    const DOUBLE CRTM_3_2 = 0.866025403784438646763723170752936183471402627;

    DOUBLE *in = (DOUBLE *)in_real;
    DOUBLE *out = (DOUBLE *)out_real;
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
        DOUBLE v0, v1, v2;
        DOUBLE t0, t1, t2, t3;

        // Input point 1: x(0)
        v0 = *in;
        // Input point 2: x(1)
        v1 = in[in_strides[1]];
        // Input point 3: x(2)
        v2 = in[in_strides[2]];

        t0 = v2 + v1;
        t1 = v2 - v1;

        t2 = CRTM_3_1 * t0;
        t3 = CRTM_3_2 * t1;

        // Output point 1: X(0)
        *out = v0 + t0;

        // Output point 2: X(1)
        out[out_strides[1]] = v0 - t2;

        // Output point 3: X(2)
        out[out_strides[2]] = t3;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
}

static VOID r2hc_rfft3c_fp64_bwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                 VOID *out_imag, INTP n,
                                 aoclfftz_strides_t *strides, UINT8 flag)
{
    const DOUBLE CRTM_3_1 = 1.732050807568877293527446341505872366942805254;
    const DOUBLE CRTM_3_2 = 2.0;

    DOUBLE *in = (DOUBLE *)in_real;
    DOUBLE *out = (DOUBLE *)out_real;
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
        DOUBLE v0, v1, v2;
        DOUBLE t0, t1;

        // Input point 1: x(0)
        v0 = *in;
        // Input point 2: x(1)
        v1 = in[in_strides[1]];
        // Input point 3: x(2)
        v2 = in[in_strides[2]];

        t0 = v0 - v1;
        t1 = CRTM_3_1 * v2;

        // Output point 1: X(0)
        *out = v0 + CRTM_3_2 * v1;

        // Output point 2: X(1)
        out[out_strides[1]] = t0 - t1;

        // Output point 3: X(2)
        out[out_strides[2]] = t0 + t1;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
}

kfft_ register_kernel_r2hc_rfft3c(UINT8 precision, UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hc_rfft3c_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft3c_fp64_fwd;
        }
        else
        {
            return NULL;
        }
    }
    else
    {
        if (precision == DT_FLOAT)
        {
            return r2hc_rfft3c_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft3c_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
