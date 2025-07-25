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

/** @file rfft12c.c
 *
 *  @brief Radix-12 r2hc Real-FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-12 real-to-halfcomplex implementations
 *  using scalar operations for single-precision and double-precision inputs.
 *
 *  @author Partiksha
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                    {{{0, 8, 38, 48, 0, 0},
                                                      {0, 10, 38, 48, 0, 0}},
                                                     {{0, 8, 38, 48, 0, 0},
                                                      {0, 10, 38, 48, 0, 0}}};

ops_cycles_t get_ops_cnt_r2hc_rfft12c(UINT8 precision, UINT8 direction)
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

static VOID r2hc_rfft12c_fp32_fwd(VOID *in_real, VOID *in_imag,
                                  VOID *out_real, VOID *out_imag, INTP n,
                                  aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const FLOAT CRTM_12_1 = 0.866025403784438646763723170752936183471402627f;
    const FLOAT CRTM_12_2 = 0.500000000000000000000000000000000000000000000f;

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
        FLOAT v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11;
        FLOAT t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13,
              t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25,
              t26, t27, t28, t29, t30, t31, t32, t33;

        v0 = *in;                   // Input point 1: x(0)
        v1 = in[in_strides[1]];     // Input point 2: x(1)
        v2 = in[in_strides[2]];     // Input point 3: x(2)
        v3 = in[in_strides[3]];     // Input point 4: x(3)
        v4 = in[in_strides[4]];     // Input point 5: x(4)
        v5 = in[in_strides[5]];     // Input point 6: x(5)
        v6 = in[in_strides[6]];     // Input point 7: x(6)
        v7 = in[in_strides[7]];     // Input point 8: x(7)
        v8 = in[in_strides[8]];     // Input point 9: x(8)
        v9 = in[in_strides[9]];     // Input point 10: x(9)
        v10 = in[in_strides[10]];   // Input point 11: x(10)
        v11 = in[in_strides[11]];   // Input point 12: x(11)

        t0 = v11 + v1;
        t1 = v11 - v1;
        t2 = v5 + v7;
        t3 = v5 - v7;
        t4 = v0 + v6;
        t5 = v0 - v6;
        t6 = v10 + v2;
        t7 = v10 - v2;
        t8 = v4 + v8;
        t9 = v4 - v8;
        t10 = v9 + v3;
        t11 = v9 - v3;

        t12 = t0 + t2;
        t13 = t0 - t2;
        t14 = t6 + t8;
        t15 = t6 - t8;
        t16 = t1 + t3;
        t17 = t1 - t3;
        t18 = t4 + t10;
        t19 = t4 - t10;
        t20 = t12 + t14;
        t21 = t12 - t14;

        t22 = CRTM_12_2 * t15;
        t23 = CRTM_12_2 * t17;
        t24 = t5 + t22;
        t25 = t23 + t11;
        t26 = t7 + t9;
        t27 = t7 - t9;
        t28 = t16 + t26;
        t29 = t16 - t26;

        t30 = CRTM_12_1 * t13;
        t31 = CRTM_12_2 * t20;
        t32 = CRTM_12_2 * t21;
        t33 = CRTM_12_1 * t27;

        *out = t20 + t18;                           // Output pt 1: X(0)
        out[out_strides[1]]  = t24 + t30;           // Output pt 2: X(1)
        out[out_strides[2]]  = t25 + t33;           // Output pt 2: X(1)
        out[out_strides[3]]  = t32 + t19;           // Output pt 4: X(3)
        out[out_strides[4]]  = CRTM_12_1 * t28;     // Output pt 5: X(4)
        out[out_strides[5]]  = t5 - t15;            // Output pt 6: X(5)
        out[out_strides[6]]  = t17 - t11;           // Output pt 7: X(6)
        out[out_strides[7]]  = t18 - t31;           // Output pt 8: X(7)
        out[out_strides[8]]  = CRTM_12_1 * t29;     // Output pt 9: X(8)
        out[out_strides[9]]  = t24 - t30;           // Output pt 10: X(9)
        out[out_strides[10]] = t25 - t33;           // Output pt 11: X(10)
        out[out_strides[11]] = t19 - t21;           // Output pt 12: X(11)

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID r2hc_rfft12c_fp32_bwd(VOID *in_real, VOID *in_imag,
                                  VOID *out_real, VOID *out_imag, INTP n,
                                  aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const FLOAT CRTM_12_1 = 1.732050807568877293527446341505872366942805254f;
    const FLOAT CRTM_12_2 = 2.000000000000000000000000000000000000000000000f;

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
        FLOAT v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11;
        FLOAT t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13,
              t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25,
              t26, t27, t28, t29, t30, t31, t32, t33, t34, t35;

        v0 = *in;                   // Input point 1: X(0)
        v1 = in[in_strides[1]];     // Input point 2: X(1)
        v2 = in[in_strides[2]];     // Input point 3: X(2)
        v3 = in[in_strides[3]];     // Input point 4: X(3)
        v4 = in[in_strides[4]];     // Input point 5: X(4)
        v5 = in[in_strides[5]];     // Input point 6: X(5)
        v6 = in[in_strides[6]];     // Input point 7: X(6)
        v7 = in[in_strides[7]];     // Input point 8: X(7)
        v8 = in[in_strides[8]];     // Input point 9: X(8)
        v9 = in[in_strides[9]];     // Input point 10: X(9)
        v10 = in[in_strides[10]];   // Input point 11: X(10)
        v11 = in[in_strides[11]];   // Input point 12: X(11)

        t0 = v0 + v11;
        t1 = v0 - v11;
        t2 = v1 + v9;
        t3 = v1 - v9;
        t4 = v3 + v7;
        t5 = v3 - v7;
        t6 = v4 + v8;
        t7 = v4 - v8;
        t8 = v10 + v2;
        t9 = v10 - v2;

        t10 = CRTM_12_2 * v5;
        t11 = CRTM_12_2 * v6;

        t12 = t2 + t4;
        t13 = t0 + t10;
        t14 = t0 - t10;
        t15 = CRTM_12_2 * t12;

        t16 = t3 + t6;
        t17 = t3 - t6;
        t18 = CRTM_12_1 * t16;
        t19 = CRTM_12_1 * t17;

        t20 = t9 + t7;
        t21 = t9 - t7;
        t22 = CRTM_12_1 * t20;
        t23 = CRTM_12_1 * t21;

        t24 = t5 + t8;
        t25 = t5 - t8;
        t26 = t2 - t4;
        t27 = CRTM_12_2 * t24;
        t28 = CRTM_12_2 * t25;
        t29 = CRTM_12_2 * t26;

        t30 = t1 - t11;
        t31 = t1 + t11;
        t32 = t25 + t30;
        t33 = t31 + t24;
        t34 = t13 - t12;
        t35 = t14 + t26;

        *out = t13 + t15;                           // Output pt 1: x(0)
        out[out_strides[1]] = t19 + t32;            // Output pt 2: x(1)
        out[out_strides[2]] = t23 + t35;            // Output pt 3: x(2)
        out[out_strides[3]] = t31 - t27;            // Output pt 4: x(3)
        out[out_strides[4]] = t34 + t22;            // Output pt 5: x(4)
        out[out_strides[5]] = t32 - t19;            // Output pt 6: x(5)
        out[out_strides[6]] = t14 - t29;            // Output pt 7: x(6)
        out[out_strides[7]] = t33 - t18;            // Output pt 8: x(7)
        out[out_strides[8]] = t34 - t22;            // Output pt 9: x(8)
        out[out_strides[9]] = t30 - t28;            // Output pt 10: x(9)
        out[out_strides[10]] = t35 - t23;           // Output pt 11: x(10)
        out[out_strides[11]] = t33 + t18;           // Output pt 12: x(11)

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
 #ifdef AOCL_ENABLE_LOG
     AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
 #endif
 }

static VOID r2hc_rfft12c_fp64_fwd(VOID *in_real, VOID *in_imag,
                                  VOID *out_real, VOID *out_imag, INTP n,
                                  aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{

#ifdef AOCL_ENABLE_LOG
AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const DOUBLE CRTM_12_1 = 0.866025403784438646763723170752936183471402627;
    const DOUBLE CRTM_12_2 = 0.500000000000000000000000000000000000000000000;

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
        DOUBLE v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11;
        DOUBLE t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13,
              t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25,
              t26, t27, t28, t29, t30, t31, t32, t33;

        v0 = *in;                   // Input point 1: x(0)
        v1 = in[in_strides[1]];     // Input point 2: x(1)
        v2 = in[in_strides[2]];     // Input point 3: x(2)
        v3 = in[in_strides[3]];     // Input point 4: x(3)
        v4 = in[in_strides[4]];     // Input point 5: x(4)
        v5 = in[in_strides[5]];     // Input point 6: x(5)
        v6 = in[in_strides[6]];     // Input point 7: x(6)
        v7 = in[in_strides[7]];     // Input point 8: x(7)
        v8 = in[in_strides[8]];     // Input point 9: x(8)
        v9 = in[in_strides[9]];     // Input point 10: x(9)
        v10 = in[in_strides[10]];   // Input point 11: x(10)
        v11 = in[in_strides[11]];   // Input point 12: x(11)

        t0 = v11 + v1;
        t1 = v11 - v1;
        t2 = v5 + v7;
        t3 = v5 - v7;
        t4 = v0 + v6;
        t5 = v0 - v6;
        t6 = v10 + v2;
        t7 = v10 - v2;
        t8 = v4 + v8;
        t9 = v4 - v8;
        t10 = v9 + v3;
        t11 = v9 - v3;

        t12 = t0 + t2;
        t13 = t0 - t2;
        t14 = t6 + t8;
        t15 = t6 - t8;
        t16 = t1 + t3;
        t17 = t1 - t3;
        t18 = t4 + t10;
        t19 = t4 - t10;
        t20 = t12 + t14;
        t21 = t12 - t14;

        t22 = CRTM_12_2 * t15;
        t23 = CRTM_12_2 * t17;
        t24 = t5 + t22;
        t25 = t23 + t11;
        t26 = t7 + t9;
        t27 = t7 - t9;
        t28 = t16 + t26;
        t29 = t16 - t26;

        t30 = CRTM_12_1 * t13;
        t31 = CRTM_12_2 * t20;
        t32 = CRTM_12_2 * t21;
        t33 = CRTM_12_1 * t27;

        *out = t20 + t18;                           // Output pt 1: X(0)
        out[out_strides[1]]  = t24 + t30;           // Output pt 2: X(1)
        out[out_strides[2]]  = t25 + t33;           // Output pt 2: X(1)
        out[out_strides[3]]  = t32 + t19;           // Output pt 4: X(3)
        out[out_strides[4]]  = CRTM_12_1 * t28;     // Output pt 5: X(4)
        out[out_strides[5]]  = t5 - t15;            // Output pt 6: X(5)
        out[out_strides[6]]  = t17 - t11;           // Output pt 7: X(6)
        out[out_strides[7]]  = t18 - t31;           // Output pt 8: X(7)
        out[out_strides[8]]  = CRTM_12_1 * t29;     // Output pt 9: X(8)
        out[out_strides[9]]  = t24 - t30;           // Output pt 10: X(9)
        out[out_strides[10]] = t25 - t33;           // Output pt 11: X(10)
        out[out_strides[11]] = t19 - t21;           // Output pt 12: X(11)

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID r2hc_rfft12c_fp64_bwd(VOID *in_real, VOID *in_imag,
                                  VOID *out_real, VOID *out_imag, INTP n,
                                  aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const DOUBLE CRTM_12_1 = 1.732050807568877293527446341505872366942805254;
    const DOUBLE CRTM_12_2 = 2.000000000000000000000000000000000000000000000;

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
        DOUBLE v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11;
        DOUBLE t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13,
               t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25,
               t26, t27, t28, t29, t30, t31, t32, t33, t34, t35;

        v0 = *in;                   // Input point 1: X(0)
        v1 = in[in_strides[1]];     // Input point 2: X(1)
        v2 = in[in_strides[2]];     // Input point 3: X(2)
        v3 = in[in_strides[3]];     // Input point 4: X(3)
        v4 = in[in_strides[4]];     // Input point 5: X(4)
        v5 = in[in_strides[5]];     // Input point 6: X(5)
        v6 = in[in_strides[6]];     // Input point 7: X(6)
        v7 = in[in_strides[7]];     // Input point 8: X(7)
        v8 = in[in_strides[8]];     // Input point 9: X(8)
        v9 = in[in_strides[9]];     // Input point 10: X(9)
        v10 = in[in_strides[10]];   // Input point 11: X(10)
        v11 = in[in_strides[11]];   // Input point 12: X(11)

        t0 = v0 + v11;
        t1 = v0 - v11;
        t2 = v1 + v9;
        t3 = v1 - v9;
        t4 = v3 + v7;
        t5 = v3 - v7;
        t6 = v4 + v8;
        t7 = v4 - v8;
        t8 = v10 + v2;
        t9 = v10 - v2;

        t10 = CRTM_12_2 * v5;
        t11 = CRTM_12_2 * v6;

        t12 = t2 + t4;
        t13 = t0 + t10;
        t14 = t0 - t10;
        t15 = CRTM_12_2 * t12;

        t16 = t3 + t6;
        t17 = t3 - t6;
        t18 = CRTM_12_1 * t16;
        t19 = CRTM_12_1 * t17;

        t20 = t9 + t7;
        t21 = t9 - t7;
        t22 = CRTM_12_1 * t20;
        t23 = CRTM_12_1 * t21;

        t24 = t5 + t8;
        t25 = t5 - t8;
        t26 = t2 - t4;
        t27 = CRTM_12_2 * t24;
        t28 = CRTM_12_2 * t25;
        t29 = CRTM_12_2 * t26;

        t30 = t1 - t11;
        t31 = t1 + t11;
        t32 = t25 + t30;
        t33 = t31 + t24;
        t34 = t13 - t12;
        t35 = t14 + t26;

        *out = t13 + t15;                           // Output pt 1: x(0)
        out[out_strides[1]] = t19 + t32;            // Output pt 2: x(1)
        out[out_strides[2]] = t23 + t35;            // Output pt 3: x(2)
        out[out_strides[3]] = t31 - t27;            // Output pt 4: x(3)
        out[out_strides[4]] = t34 + t22;            // Output pt 5: x(4)
        out[out_strides[5]] = t32 - t19;            // Output pt 6: x(5)
        out[out_strides[6]] = t14 - t29;            // Output pt 7: x(6)
        out[out_strides[7]] = t33 - t18;            // Output pt 8: x(7)
        out[out_strides[8]] = t34 - t22;            // Output pt 9: x(8)
        out[out_strides[9]] = t30 - t28;            // Output pt 10: x(9)
        out[out_strides[10]] = t35 - t23;           // Output pt 11: x(10)
        out[out_strides[11]] = t33 + t18;           // Output pt 12: x(11)

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

kfft_ register_kernel_r2hc_rfft12c(UINT8 precision, UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hc_rfft12c_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft12c_fp64_fwd;
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
            return r2hc_rfft12c_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft12c_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
