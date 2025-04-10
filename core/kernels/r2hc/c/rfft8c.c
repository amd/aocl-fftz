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

/** @file rfft8c.c
 *
 *  @brief Radix-8 r2hc Real-FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-8 real-to-halfcomplex implementations
 *  using scalar operations for single-precision and double-precision inputs.
 *
 *  @author Dr. Pritam Giri
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                    {{{0, 2, 20, 16, 0, 0},
                                                      {0, 4, 22, 16, 0, 0}},
                                                     {{0, 2, 20, 16, 0, 0},
                                                      {0, 4, 22, 16, 0, 0}}};

ops_cycles_t get_ops_cnt_r2hc_rfft8c(UINT8 precision, UINT8 direction)
{
    if (precision == DT_FLOAT)
    {
        if (direction == FORWARD_FFT_DIR)
        {
            return ops_cnt[0][0];
        }
        else
        {
            return ops_cnt[0][1];
        }
    }
    else
    {
        if (direction == FORWARD_FFT_DIR)
        {
            return ops_cnt[1][0];
        }
        else
        {
            return ops_cnt[1][1];
        }
    }
}

static VOID r2hc_rfft8c_fp32_fwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                              VOID *out_imag, INTP n,
                              aoclfftz_strides_t *strides, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const FLOAT CRTM_8_1 = 0.7071067811865475244008443621048490392848359377f;

    FLOAT *in = (FLOAT *)in_real;
    FLOAT *out = (FLOAT *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = (strides->v_in_stride);
    INTP v_out_stride = (strides->v_out_stride);
    INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        FLOAT v0, v1, v2, v3, v4, v5, v6, v7;
        FLOAT t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13;

        // Input point 1: x(0)
        v0 = *in;
        // Input point 2: x(1)
        v1 = in[in_strides[1]];
        // Input point 3: x(2)
        v2 = in[in_strides[2]];
        // Input point 4: x(3)
        v3 = in[in_strides[3]];
        // Input point 5: x(4)
        v4 = in[in_strides[4]];
        // Input point 6: x(5)
        v5 = in[in_strides[5]];
        // Input point 7: x(6)
        v6 = in[in_strides[6]];
        // Input point 8: x(7)
        v7 = in[in_strides[7]];

        t0 = v7 + v5;
        t1 = v7 - v5;
        t2 = v6 + v2;
        t3 = v6 - v2;
        t4 = v0 + v4;
        t5 = v0 - v4;
        t6 = v3 + v1;
        t7 = v3 - v1;

        t8 = t7 - t1;
        t9 = t0 - t6;
        t10 = t0 + t6;
        t11 = t4 + t2;
        t12 = CRTM_8_1 * t8;
        t13 = CRTM_8_1 * t9;

        *out = t11 + t10;                   // Output pt 1: X(0)
        out[out_strides[1]] = t5 - t12;     // Output pt 2: X(1)
        out[out_strides[2]] = t3 + t13;     // Output pt 3: X(2)
        out[out_strides[3]] = t4 - t2;      // Output pt 4: X(3)
        out[out_strides[4]] = t7 + t1;      // Output pt 5: X(4)
        out[out_strides[5]] = t12 + t5;     // Output pt 6: X(5)
        out[out_strides[6]] = t13 - t3;     // Output pt 7: X(6)
        out[out_strides[7]] = t11 - t10;    // Output pt 8: X(7)

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID r2hc_rfft8c_fp32_bwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                              VOID *out_imag, INTP n,
                              aoclfftz_strides_t *strides, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const FLOAT CRTM_8_1 = 1.414213562373095048801688724209698078569671875f;
    const FLOAT CRTM_8_2 = 2.000000000000000000000000000000000000000000000f;

    FLOAT *in = (FLOAT *)in_real;
    FLOAT *out = (FLOAT *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = (strides->v_in_stride);
    INTP v_out_stride = (strides->v_out_stride);
    INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        FLOAT v0, v1, v2, v3, v4, v5, v6, v7;
        FLOAT t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13,
              t14, t15, t16, t17;

        // Input point 1: x(0)
        v0 = *in;
        // Input point 2: x(1)
        v1 = in[in_strides[1]];
        // Input point 3: x(2)
        v2 = in[in_strides[2]];
        // Input point 4: x(3)
        v3 = in[in_strides[3]];
        // Input point 5: x(4)
        v4 = in[in_strides[4]];
        // Input point 6: x(5)
        v5 = in[in_strides[5]];
        // Input point 7: x(6)
        v6 = in[in_strides[6]];
        // Input point 8: x(7)
        v7 = in[in_strides[7]];

        t0 = v7 + v0;
        t1 = v7 - v0;
        t2 = v6 + v2;
        t3 = v6 - v2;
        t4 = v5 + v1;
        t5 = v5 - v1;
        t6 = CRTM_8_2 * t3;
        t7 = CRTM_8_2 * t4;
        t8 = CRTM_8_2 * v3;
        t9 = CRTM_8_2 * v4;

        t10 = t5 + t2;
        t11 = t5 - t2;
        t12 = CRTM_8_1 * t10;
        t13 = CRTM_8_1 * t11;
        t14 = t0 + t8;
        t15 = t0 - t8;
        t16 = t9 + t1;
        t17 = t9 - t1;

        *out = t14 + t7;                   // Output pt 1: X(0)
        out[out_strides[1]] = -t12 - t16;  // Output pt 2: X(1)
        out[out_strides[2]] = t15 + t6;    // Output pt 3: X(2)
        out[out_strides[3]] = t17 + t13;   // Output pt 4: X(3)
        out[out_strides[4]] = t14 - t7;    // Output pt 5: X(4)
        out[out_strides[5]] = t12 - t16;   // Output pt 6: X(5)
        out[out_strides[6]] = t15 - t6;    // Output pt 7: X(6)
        out[out_strides[7]] = t17 - t13;   // Output pt 8: X(7)

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID r2hc_rfft8c_fp64_fwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                              VOID *out_imag, INTP n,
                              aoclfftz_strides_t *strides, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const DOUBLE CRTM_8_1 = 0.7071067811865475244008443621048490392848359377;

    DOUBLE *in = (DOUBLE *)in_real;
    DOUBLE *out = (DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = (strides->v_in_stride);
    INTP v_out_stride = (strides->v_out_stride);
    INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        DOUBLE v0, v1, v2, v3, v4, v5, v6, v7;
        DOUBLE t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13;

        // Input point 1: x(0)
        v0 = *in;
        // Input point 2: x(1)
        v1 = in[in_strides[1]];
        // Input point 3: x(2)
        v2 = in[in_strides[2]];
        // Input point 4: x(3)
        v3 = in[in_strides[3]];
        // Input point 5: x(4)
        v4 = in[in_strides[4]];
        // Input point 6: x(5)
        v5 = in[in_strides[5]];
        // Input point 7: x(6)
        v6 = in[in_strides[6]];
        // Input point 8: x(7)
        v7 = in[in_strides[7]];

        t0 = v7 + v5;
        t1 = v7 - v5;
        t2 = v6 + v2;
        t3 = v6 - v2;
        t4 = v0 + v4;
        t5 = v0 - v4;
        t6 = v3 + v1;
        t7 = v3 - v1;

        t8 = t7 - t1;
        t9 = t0 - t6;
        t10 = t0 + t6;
        t11 = t4 + t2;
        t12 = CRTM_8_1 * t8;
        t13 = CRTM_8_1 * t9;

        *out = t11 + t10;                   // Output pt 1: X(0)
        out[out_strides[1]] = t5 - t12;     // Output pt 2: X(1)
        out[out_strides[2]] = t3 + t13;     // Output pt 3: X(2)
        out[out_strides[3]] = t4 - t2;      // Output pt 4: X(3)
        out[out_strides[4]] = t7 + t1;      // Output pt 5: X(4)
        out[out_strides[5]] = t12 + t5;     // Output pt 6: X(5)
        out[out_strides[6]] = t13 - t3;     // Output pt 7: X(6)
        out[out_strides[7]] = t11 - t10;    // Output pt 8: X(7)

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID r2hc_rfft8c_fp64_bwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                              VOID *out_imag, INTP n,
                              aoclfftz_strides_t *strides, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const DOUBLE CRTM_8_1 = 1.414213562373095048801688724209698078569671875;
    const DOUBLE CRTM_8_2 = 2.000000000000000000000000000000000000000000000;

    DOUBLE *in = (DOUBLE *)in_real;
    DOUBLE *out = (DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = (strides->v_in_stride);
    INTP v_out_stride = (strides->v_out_stride);
    INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        DOUBLE v0, v1, v2, v3, v4, v5, v6, v7;
        DOUBLE t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13,
               t14, t15, t16, t17;

        // Input point 1: x(0)
        v0 = *in;
        // Input point 2: x(1)
        v1 = in[in_strides[1]];
        // Input point 3: x(2)
        v2 = in[in_strides[2]];
        // Input point 4: x(3)
        v3 = in[in_strides[3]];
        // Input point 5: x(4)
        v4 = in[in_strides[4]];
        // Input point 6: x(5)
        v5 = in[in_strides[5]];
        // Input point 7: x(6)
        v6 = in[in_strides[6]];
        // Input point 8: x(7)
        v7 = in[in_strides[7]];

        t0 = v7 + v0;
        t1 = v7 - v0;
        t2 = v6 + v2;
        t3 = v6 - v2;
        t4 = v5 + v1;
        t5 = v5 - v1;
        t6 = CRTM_8_2 * t3;
        t7 = CRTM_8_2 * t4;
        t8 = CRTM_8_2 * v3;
        t9 = CRTM_8_2 * v4;

        t10 = t5 + t2;
        t11 = t5 - t2;
        t12 = CRTM_8_1 * t10;
        t13 = CRTM_8_1 * t11;
        t14 = t0 + t8;
        t15 = t0 - t8;
        t16 = t9 + t1;
        t17 = t9 - t1;

        *out = t14 + t7;                   // Output pt 1: X(0)
        out[out_strides[1]] = -t12 - t16;  // Output pt 2: X(1)
        out[out_strides[2]] = t15 + t6;    // Output pt 3: X(2)
        out[out_strides[3]] = t17 + t13;   // Output pt 4: X(3)
        out[out_strides[4]] = t14 - t7;    // Output pt 5: X(4)
        out[out_strides[5]] = t12 - t16;   // Output pt 6: X(5)
        out[out_strides[6]] = t15 - t6;    // Output pt 7: X(6)
        out[out_strides[7]] = t17 - t13;   // Output pt 8: X(7)

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

kfft_ register_kernel_r2hc_rfft8c(UINT8 precision, UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hc_rfft8c_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft8c_fp64_fwd;
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
            return r2hc_rfft8c_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft8c_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
