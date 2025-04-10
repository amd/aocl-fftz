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

/** @file rfft7c.c
 *
 *  @brief Radix-7 r2hc Real-FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-7 real-to-halfcomplex implementations
 *  using scalar operations for single-precision and double-precision inputs.
 *
 *  @author Dr. Pritam Giri
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                    {{{0, 18, 24, 14, 0, 0},
                                                      {0, 19, 24, 14, 0, 0}},
                                                     {{0, 18, 24, 14, 0, 0},
                                                      {0, 19, 24, 14, 0, 0}}};

ops_cycles_t get_ops_cnt_r2hc_rfft7c(UINT8 precision, UINT8 direction)
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

static VOID r2hc_rfft7c_fp32_fwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                              VOID *out_imag, INTP n,
                              aoclfftz_strides_t *strides, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const FLOAT CRTM_7_1 = 0.900968867902419126236102319507445051165919162f;
    const FLOAT CRTM_7_2 = 0.433883739117558120475768332848358754609990728f;
    const FLOAT CRTM_7_3 = 0.623489801858733530525004884004239810632274731f;
    const FLOAT CRTM_7_4 = 0.781831482468029808708444526674057750232334519f;
    const FLOAT CRTM_7_5 = 0.222520933956314404288902564496794759466355569f;
    const FLOAT CRTM_7_6 = 0.974927912181823607018131682993931217232785801f;

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
        FLOAT v0, v1, v2, v3, v4, v5, v6;
        FLOAT t0, t1, t2, t3, t4, t5, t6, t7, t8, t9,
              t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
              t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
              t30, t31, t32, t33, t34;

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

        t0 = v6 + v1;
        t1 = v6 - v1;
        t2 = v5 + v2;
        t3 = v5 - v2;
        t4 = v4 + v3;
        t5 = v4 - v3;
        t6 = v0 + t0;
        t7 = t2 + t4;

        t8 = CRTM_7_1 * t4;
        t9 = CRTM_7_3 * t0;
        t10 = CRTM_7_5 * t2;
        t11 = CRTM_7_2 * t5;
        t12 = CRTM_7_4 * t1;
        t13 = v0 - t8;
        t14 = t9 - t10;

        t15 = CRTM_7_6 * t3;
        t16 = t11 + t12;
        t17 = CRTM_7_1 * t2;
        t18 = CRTM_7_3 * t4;
        t19 = CRTM_7_5 * t0;

        t20 = v0 - t17;
        t21 = t18 - t19;
        t22 = CRTM_7_2 * t3;
        t23 = CRTM_7_4 * t5;

        t24 = CRTM_7_6 * t1;
        t25 = CRTM_7_1 * t0;
        t26 = CRTM_7_3 * t2;
        t27 = CRTM_7_5 * t4;
        t28 = t22 + t23;

        t29 = v0 - t25;
        t30 = t26 - t27;
        t31 = CRTM_7_2 * t1;
        t32 = CRTM_7_4 * t3;
        t33 = CRTM_7_6 * t5;
        t34 = t31 - t32;

        *out = t6 + t7;                      // Output pt 1: X(0)
        out[out_strides[1]] = t13 + t14;     // Output pt 2: X(1)
        out[out_strides[2]] = t15 + t16;     // Output pt 3: X(2)
        out[out_strides[3]] = t20 + t21;     // Output pt 4: X(3)
        out[out_strides[4]] = t24 - t28;     // Output pt 5: X(4)
        out[out_strides[5]] = t29 + t30;     // Output pt 6: X(5)
        out[out_strides[6]] = t34 + t33;     // Output pt 7: X(6)

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID r2hc_rfft7c_fp32_bwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                              VOID *out_imag, INTP n,
                              aoclfftz_strides_t *strides, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const FLOAT CRTM_7_1 = 1.801937735804838252472204639014890102331838324f;
    const FLOAT CRTM_7_2 = 0.867767478235116240951536665696717509219981456f;
    const FLOAT CRTM_7_3 = 1.246979603717467061050009768008479621264549462f;
    const FLOAT CRTM_7_4 = 1.563662964936059617416889053348115500464669038f;
    const FLOAT CRTM_7_5 = 0.445041867912628808577805128993589518932711138f;
    const FLOAT CRTM_7_6 = 1.949855824363647214036263365987862434465571602f;
    const FLOAT CRTM_7_7 = 2.000000000000000000000000000000000000000000000f;

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
        FLOAT v0, v1, v2, v3, v4, v5, v6;
        FLOAT t0, t1, t2, t3, t4, t5, t6, t7, t8, t9,
              t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
              t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
              t30, t31, t32, t33, t34, t35;

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

        t0 = CRTM_7_1 * v5;
        t1 = CRTM_7_3 * v1;
        t2 = CRTM_7_5 * v3;
        t3 = v0 - t0;
        t4 = t1 - t2;

        t5 = CRTM_7_2 * v6;
        t6 = CRTM_7_4 * v2;
        t7 = CRTM_7_6 * v4;
        t8 = t5 + t6;
        t9 = t3 + t4;
        t10 = t8 + t7;

        t11 = CRTM_7_1 * v3;
        t12 = CRTM_7_3 * v5;
        t13 = CRTM_7_5 * v1;
        t14 = v0 - t11;
        t15 = t12 - t13;

        t16 = CRTM_7_2 * v4;
        t17 = CRTM_7_4 * v6;
        t18 = CRTM_7_6 * v2;
        t19 = t16 + t17;

        t20 = t14 + t15;
        t21 = t18 - t19;

        t22 = CRTM_7_1 * v1;
        t23 = CRTM_7_3 * v3;
        t24 = CRTM_7_5 * v5;
        t25 = v0 - t22;
        t26 = t23 - t24;

        t27 = CRTM_7_2 * v2;
        t28 = CRTM_7_4 * v4;
        t29 = CRTM_7_6 * v6;
        t30 = t27 - t28;

        t31 = v1 + v3;
        t32 = t25 + t26;
        t33 = t31 + v5;
        t34 = t30 + t29;
        t35 = CRTM_7_7 * t33;

        *out = v0 + t35;                    // Output pt 1: X(0)
        out[out_strides[1]] = t9 - t10;     // Output pt 2: X(1)
        out[out_strides[2]] = t20 - t21;    // Output pt 3: X(2)
        out[out_strides[3]] = t32 - t34;    // Output pt 4: X(3)
        out[out_strides[4]] = t32 + t34;    // Output pt 5: X(4)
        out[out_strides[5]] = t20 + t21;    // Output pt 6: X(5)
        out[out_strides[6]] = t9 + t10;     // Output pt 7: X(6)

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID r2hc_rfft7c_fp64_fwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                              VOID *out_imag, INTP n,
                              aoclfftz_strides_t *strides, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const DOUBLE CRTM_7_1 = 0.900968867902419126236102319507445051165919162;
    const DOUBLE CRTM_7_2 = 0.433883739117558120475768332848358754609990728;
    const DOUBLE CRTM_7_3 = 0.623489801858733530525004884004239810632274731;
    const DOUBLE CRTM_7_4 = 0.781831482468029808708444526674057750232334519;
    const DOUBLE CRTM_7_5 = 0.222520933956314404288902564496794759466355569;
    const DOUBLE CRTM_7_6 = 0.974927912181823607018131682993931217232785801;

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
        DOUBLE v0, v1, v2, v3, v4, v5, v6;
        DOUBLE t0, t1, t2, t3, t4, t5, t6, t7, t8, t9,
               t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
               t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
               t30, t31, t32, t33, t34;

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

        t0 = v6 + v1;
        t1 = v6 - v1;
        t2 = v5 + v2;
        t3 = v5 - v2;
        t4 = v4 + v3;
        t5 = v4 - v3;
        t6 = v0 + t0;
        t7 = t2 + t4;

        t8 = CRTM_7_1 * t4;
        t9 = CRTM_7_3 * t0;
        t10 = CRTM_7_5 * t2;
        t11 = CRTM_7_2 * t5;
        t12 = CRTM_7_4 * t1;
        t13 = v0 - t8;
        t14 = t9 - t10;

        t15 = CRTM_7_6 * t3;
        t16 = t11 + t12;
        t17 = CRTM_7_1 * t2;
        t18 = CRTM_7_3 * t4;
        t19 = CRTM_7_5 * t0;

        t20 = v0 - t17;
        t21 = t18 - t19;
        t22 = CRTM_7_2 * t3;
        t23 = CRTM_7_4 * t5;

        t24 = CRTM_7_6 * t1;
        t25 = CRTM_7_1 * t0;
        t26 = CRTM_7_3 * t2;
        t27 = CRTM_7_5 * t4;
        t28 = t22 + t23;

        t29 = v0 - t25;
        t30 = t26 - t27;
        t31 = CRTM_7_2 * t1;
        t32 = CRTM_7_4 * t3;
        t33 = CRTM_7_6 * t5;
        t34 = t31 - t32;

        *out = t6 + t7;                      // Output pt 1: X(0)
        out[out_strides[1]] = t13 + t14;     // Output pt 2: X(1)
        out[out_strides[2]] = t15 + t16;     // Output pt 3: X(2)
        out[out_strides[3]] = t20 + t21;     // Output pt 4: X(3)
        out[out_strides[4]] = t24 - t28;     // Output pt 5: X(4)
        out[out_strides[5]] = t29 + t30;     // Output pt 6: X(5)
        out[out_strides[6]] = t34 + t33;     // Output pt 7: X(6)

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID r2hc_rfft7c_fp64_bwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                              VOID *out_imag, INTP n,
                              aoclfftz_strides_t *strides, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const DOUBLE CRTM_7_1 = 1.801937735804838252472204639014890102331838324;
    const DOUBLE CRTM_7_2 = 0.867767478235116240951536665696717509219981456;
    const DOUBLE CRTM_7_3 = 1.246979603717467061050009768008479621264549462;
    const DOUBLE CRTM_7_4 = 1.563662964936059617416889053348115500464669038;
    const DOUBLE CRTM_7_5 = 0.445041867912628808577805128993589518932711138;
    const DOUBLE CRTM_7_6 = 1.949855824363647214036263365987862434465571602;
    const DOUBLE CRTM_7_7 = 2.000000000000000000000000000000000000000000000;

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
        DOUBLE v0, v1, v2, v3, v4, v5, v6;
        DOUBLE t0, t1, t2, t3, t4, t5, t6, t7, t8, t9,
               t10, t11, t12, t13, t14, t15, t16, t17, t18, t19,
               t20, t21, t22, t23, t24, t25, t26, t27, t28, t29,
               t30, t31, t32, t33, t34, t35;

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

        t0 = CRTM_7_1 * v5;
        t1 = CRTM_7_3 * v1;
        t2 = CRTM_7_5 * v3;
        t3 = v0 - t0;
        t4 = t1 - t2;

        t5 = CRTM_7_2 * v6;
        t6 = CRTM_7_4 * v2;
        t7 = CRTM_7_6 * v4;
        t8 = t5 + t6;
        t9 = t3 + t4;
        t10 = t8 + t7;

        t11 = CRTM_7_1 * v3;
        t12 = CRTM_7_3 * v5;
        t13 = CRTM_7_5 * v1;
        t14 = v0 - t11;
        t15 = t12 - t13;

        t16 = CRTM_7_2 * v4;
        t17 = CRTM_7_4 * v6;
        t18 = CRTM_7_6 * v2;
        t19 = t16 + t17;

        t20 = t14 + t15;
        t21 = t18 - t19;

        t22 = CRTM_7_1 * v1;
        t23 = CRTM_7_3 * v3;
        t24 = CRTM_7_5 * v5;
        t25 = v0 - t22;
        t26 = t23 - t24;

        t27 = CRTM_7_2 * v2;
        t28 = CRTM_7_4 * v4;
        t29 = CRTM_7_6 * v6;
        t30 = t27 - t28;

        t31 = v1 + v3;
        t32 = t25 + t26;
        t33 = t31 + v5;
        t34 = t30 + t29;
        t35 = CRTM_7_7 * t33;

        *out = v0 + t35;                    // Output pt 1: X(0)
        out[out_strides[1]] = t9 - t10;     // Output pt 2: X(1)
        out[out_strides[2]] = t20 - t21;    // Output pt 3: X(2)
        out[out_strides[3]] = t32 - t34;    // Output pt 4: X(3)
        out[out_strides[4]] = t32 + t34;    // Output pt 5: X(4)
        out[out_strides[5]] = t20 + t21;    // Output pt 6: X(5)
        out[out_strides[6]] = t9 + t10;     // Output pt 7: X(6)

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

kfft_ register_kernel_r2hc_rfft7c(UINT8 precision, UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hc_rfft7c_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft7c_fp64_fwd;
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
            return r2hc_rfft7c_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft7c_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
