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

/** @file rfft14c.c
 *
 *  @brief Radix-14 r2hc Real-FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-14 real-to-halfcomplex implementations
 *  using scalar operations for single-precision and double-precision inputs.
 *
 *  @author Jeya R
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                    {{{0, 36, 62, 28, 0, 0},
                                                      {0, 38, 62, 28, 0, 0}},
                                                     {{0, 36, 62, 28, 0, 0},
                                                      {0, 38, 62, 28, 0, 0}}};

ops_cycles_t get_ops_cnt_r2hc_rfft14c(UINT8 precision, UINT8 direction)
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

static VOID r2hc_rfft14c_fp32_fwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                  VOID *out_imag, INTP n,
                                  aoclfftz_strides_t *strides, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const FLOAT CRTM_14_1 = 0.900968867902419126236102319507445051165919162f;
    const FLOAT CRTM_14_2 = 0.433883739117558120475768332848358754609990728f;
    const FLOAT CRTM_14_3 = 0.623489801858733530525004884004239810632274731f;
    const FLOAT CRTM_14_4 = 0.781831482468029808708444526674057750232334519f;
    const FLOAT CRTM_14_5 = 0.222520933956314404288902564496794759466355569f;
    const FLOAT CRTM_14_6 = 0.974927912181823607018131682993931217232785801f;

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
        FLOAT in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11,
              in12, in13;
        FLOAT s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14,
              s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25;
        FLOAT t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14,
              t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27,
              t28, t29, t30, t31, t32, t33, t34, t35;

        // Input point 1: x(0)
        in0 = *in;
        // Input point 2: x(1)
        in1 = in[in_strides[1]];
        // Input point 3: x(2)
        in2 = in[in_strides[2]];
        // Input point 4: x(3)
        in3 = in[in_strides[3]];
        // Input point 5: x(4)
        in4 = in[in_strides[4]];
        // Input point 6: x(5)
        in5 = in[in_strides[5]];
        // Input point 7: x(6)
        in6 = in[in_strides[6]];
        // Input point 8: x(7)
        in7 = in[in_strides[7]];
        // Input point 9: x(8)
        in8 = in[in_strides[8]];
        // Input point 10: x(9)
        in9 = in[in_strides[9]];
        // Input point 11: x(10)
        in10 = in[in_strides[10]];
        // Input point 12: x(11)
        in11 = in[in_strides[11]];
        // Input point 13: x(12)
        in12 = in[in_strides[12]];
        // Input point 14: x(13)
        in13 = in[in_strides[13]];

        s0 = in0 - in7;
        s1 = in0 + in7;
        s2 = in13 - in1;
        s3 = in13 + in1;
        s4 = in12 - in2;
        s5 = in12 + in2;
        s6 = in11 - in3;
        s7 = in11 + in3;
        s8 = in10 - in4;
        s9 = in10 + in4;
        s10 = in9 - in5;
        s11 = in9 + in5;
        s12 = in8 - in6;
        s13 = in8 + in6;

        s14 = s3 + s13;
        s15 = s5 + s11;
        s16 = s7 + s9;

        s17 = s13 - s3;
        s18 = s5 - s11;
        s19 = s9 - s7;

        // Output point 1: X(0)
        *out = s1 + s14 + s15 + s16;
        // Output point 13: X(14)
        out[out_strides[13]] = s0 + s17 + s18 + s19;

        t0 = CRTM_14_1 * s17;
        t1 = CRTM_14_3 * s18;
        t2 = CRTM_14_5 * s19;

        // Output point 2: X(1)
        out[out_strides[1]] = s0 - t0 + t1 - t2;

        s20 = s2 + s12;
        s21 = s4 + s10;
        s22 = s6 + s8;

        t3 = CRTM_14_2 * s20;
        t4 = CRTM_14_4 * s21;
        t5 = CRTM_14_6 * s22;

        // Output point 3: X(2)
        out[out_strides[2]] = t3 + t4 + t5;

        t6 = CRTM_14_1 * s16;
        t7 = CRTM_14_3 * s14;
        t8 = CRTM_14_5 * s15;

        // Output point 4: X(3)
        out[out_strides[3]] = s1 - t6 + t7 - t8;

        s23 = s2 - s12;
        s24 = s4 - s10;
        s25 = s6 - s8;

        t9 = CRTM_14_2 * s25;
        t10 = CRTM_14_4 * s23;
        t11 = CRTM_14_6 * s24;

        // Output point 5: X(4)
        out[out_strides[4]] = t9 + t10 + t11;

        t12 = CRTM_14_1 * s18;
        t13 = CRTM_14_3 * s19;
        t14 = CRTM_14_5 * s17;
        // Output point 6: X(5)
        out[out_strides[5]] = s0 - t12 + t13 - t14;
        t15 = CRTM_14_2 * s21;
        t16 = CRTM_14_4 * s22;
        t17 = CRTM_14_6 * s20;
        // Output point 7: X(6)
        out[out_strides[6]] = t15 - t16 + t17;

        t18 = CRTM_14_1 * s15;
        t19 = CRTM_14_3 * s16;
        t20 = CRTM_14_5 * s14;
        // Output point 8: X(7)
        out[out_strides[7]] = s1 - t18 + t19 - t20;
        t21 = CRTM_14_2 * s24;
        t22 = CRTM_14_4 * s25;
        t23 = CRTM_14_6 * s23;
        // Output point 9: X(8)
        out[out_strides[8]] = t23 - t22 - t21;

        t24 = CRTM_14_1 * s19;
        t25 = CRTM_14_3 * s17;
        t26 = CRTM_14_5 * s18;
        // Output point 10: X(9)
        out[out_strides[9]] = s0 - t24 + t25 - t26;
        t27 = CRTM_14_2 * s22;
        t28 = CRTM_14_4 * s20;
        t29 = CRTM_14_6 * s21;
        // Output point 11: X(10)
        out[out_strides[10]] = t27 + t28 - t29;

        t30 = CRTM_14_1 * s14;
        t31 = CRTM_14_3 * s15;
        t32 = CRTM_14_5 * s16;
        // Output point 5: X(4)
        out[out_strides[11]] = s1 - t30 + t31 - t32;
        t33 = CRTM_14_2 * s23;
        t34 = CRTM_14_4 * s24;
        t35 = CRTM_14_6 * s25;
        // Output point 5: X(4)
        out[out_strides[12]] = t33 - t34 + t35;

        in = in + v_in_stride;
        out = out + v_out_stride;
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
    }
}

static VOID r2hc_rfft14c_fp32_bwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                  VOID *out_imag, INTP n,
                                  aoclfftz_strides_t *strides, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const FLOAT CRTM_14_1 = 0.867767478235116240951536665696717509219981456f;
    const FLOAT CRTM_14_2 = 1.801937735804838252472204639014890102331838324f;
    const FLOAT CRTM_14_3 = 1.563662964936059617416889053348115500464669038f;
    const FLOAT CRTM_14_4 = 1.246979603717467061050009768008479621264549462f;
    const FLOAT CRTM_14_5 = 1.949855824363647214036263365987862434465571602f;
    const FLOAT CRTM_14_6 = 0.445041867912628808577805128993589518932711138f;
    const FLOAT CRTM_14_7 = 2.000000000000000000000000000000000000000000000f;

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
        FLOAT in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11,
              in12, in13;
        FLOAT s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14,
              s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25;
        FLOAT t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14,
              t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27,
              t28, t29, t30, t31, t32, t33, t34, t35;
         

        // Input point 1: x(0)
        in0 = *in;
        // Input point 2: x(1)
        in1 = in[in_strides[1]];
        // Input point 3: x(2)
        in2 = in[in_strides[2]];
        // Input point 4: x(3)
        in3 = in[in_strides[3]];
        // Input point 5: x(4)
        in4 = in[in_strides[4]];
        // Input point 6: x(5)
        in5 = in[in_strides[5]];
        // Input point 7: x(6)
        in6 = in[in_strides[6]];
        // Input point 8: x(7)
        in7 = in[in_strides[7]];
        // Input point 9: x(8)
        in8 = in[in_strides[8]];
        // Input point 10: x(9)
        in9 = in[in_strides[9]];
        // Input point 11: x(10)
        in10 = in[in_strides[10]];
        // Input point 12: x(11)
        in11 = in[in_strides[11]];
        // Input point 13: x(12)
        in12 = in[in_strides[12]];
        // Input point 14: x(13)
        in13 = in[in_strides[13]];

        s0 = in0 - in13;
        s1 = in0 + in13;
        s2 = in11 - in1;
        s3 = in1 + in11;
        s4 = in2 - in12;
        s5 = in2 + in12;
        s6 = in3 - in9;
        s7 = in3 + in9;
        s8 = in4 - in10;
        s9 = in4 + in10;
        s10 = in7 - in5;
        s11 = in5 + in7;
        s12 = in6 - in8;
        s13 = in6 + in8;

        *out = CRTM_14_7 * (s11 + s3 + s7) + s1;
        out[out_strides[7]] = CRTM_14_7 * (s2 + s6 + s10) + s0;

        t0 = CRTM_14_1 * s5;
        t1 = CRTM_14_3 * s9;
        t2 = CRTM_14_5 * s13;
        t3 = CRTM_14_2 * s2;
        t4 = CRTM_14_4 * s6;
        t5 = CRTM_14_6 * s10;

        s14 = t4 - t5 - t3 + s0;
        s15 = t0 + t1 + t2;
        out[out_strides[1]] = s14 - s15;
        out[out_strides[13]] = s14 + s15;

        t6 = CRTM_14_1 * s12;
        t7 = CRTM_14_3 * s4;
        t8 = CRTM_14_5 * s8;

        t9 = CRTM_14_2 * s11;
        t10 = CRTM_14_4 * s3;
        t11 = CRTM_14_6 * s7;

        s16 = t10 - t9 - t11 + s1;
        s17 = t6 + t7 + t8;
        out[out_strides[2]] = s16 - s17;
        out[out_strides[12]] = s16 + s17;

        t12 = CRTM_14_1 * s9;
        t13 = CRTM_14_3 * s13;
        t14 = CRTM_14_5 * s5;
        t15 = CRTM_14_2 * s6;
        t16 = CRTM_14_4 * s10;
        t17 = CRTM_14_6 * s2;

        s18 = t16 - t15 - t17 + s0;
        s19 = t13 - t12 - t14;
        out[out_strides[3]] = s18 + s19;
        out[out_strides[11]] = s18 - s19;

        t18 = CRTM_14_1 * s8;
        t19 = CRTM_14_3 * s12;
        t20 = CRTM_14_5 * s4;
        t21 = CRTM_14_2 * s7;
        t22 = CRTM_14_4 * s11;
        t23 = CRTM_14_6 * -s3;

        s20 = t22 + t23 - t21 + s1;
        s21 = t18 + t19 - t20;
        out[out_strides[4]] = s20 + s21;
        out[out_strides[10]] = s20 - s21;

        t24 = CRTM_14_1 * s13;
        t25 = CRTM_14_3 * s5;
        t26 = CRTM_14_5 * s9;
        t27 = CRTM_14_2 * s10;
        t28 = CRTM_14_4 * s2;
        t29 = CRTM_14_6 * s6;

        s22 = t28 - t27 - t29 + s0;
        s23 = t26 - t24 - t25;
        out[out_strides[5]] = s22 + s23;
        out[out_strides[9]] = s22 - s23;

        t30 = CRTM_14_1 * s4;
        t31 = CRTM_14_3 * s8;
        t32 = CRTM_14_5 * s12;
        t33 = CRTM_14_2 * s3;
        t34 = CRTM_14_4 * s7;
        t35 = CRTM_14_6 * s11;

        s24 = t34 - t33 - t35 + s1;
        s25 = t31 - t30 - t32;
        out[out_strides[6]] = s24 + s25;
        out[out_strides[8]] = s24 - s25;

        in = in + v_in_stride;
        out = out + v_out_stride;
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
    }
}

static VOID r2hc_rfft14c_fp64_fwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                  VOID *out_imag, INTP n,
                                  aoclfftz_strides_t *strides, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const DOUBLE CRTM_14_1 = 0.900968867902419126236102319507445051165919162;
    const DOUBLE CRTM_14_2 = 0.433883739117558120475768332848358754609990728;
    const DOUBLE CRTM_14_3 = 0.623489801858733530525004884004239810632274731;
    const DOUBLE CRTM_14_4 = 0.781831482468029808708444526674057750232334519;
    const DOUBLE CRTM_14_5 = 0.222520933956314404288902564496794759466355569;
    const DOUBLE CRTM_14_6 = 0.974927912181823607018131682993931217232785801;

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
        DOUBLE in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11,
              in12, in13;
        DOUBLE s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14,
              s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25;
        DOUBLE t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14,
              t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27,
              t28, t29, t30, t31, t32, t33, t34, t35;

        // Input point 1: x(0)
        in0 = *in;
        // Input point 2: x(1)
        in1 = in[in_strides[1]];
        // Input point 3: x(2)
        in2 = in[in_strides[2]];
        // Input point 4: x(3)
        in3 = in[in_strides[3]];
        // Input point 5: x(4)
        in4 = in[in_strides[4]];
        // Input point 6: x(5)
        in5 = in[in_strides[5]];
        // Input point 7: x(6)
        in6 = in[in_strides[6]];
        // Input point 8: x(7)
        in7 = in[in_strides[7]];
        // Input point 9: x(8)
        in8 = in[in_strides[8]];
        // Input point 10: x(9)
        in9 = in[in_strides[9]];
        // Input point 11: x(10)
        in10 = in[in_strides[10]];
        // Input point 12: x(11)
        in11 = in[in_strides[11]];
        // Input point 13: x(12)
        in12 = in[in_strides[12]];
        // Input point 14: x(13)
        in13 = in[in_strides[13]];

        s0 = in0 - in7;
        s1 = in0 + in7;
        s2 = in13 - in1;
        s3 = in13 + in1;
        s4 = in12 - in2;
        s5 = in12 + in2;
        s6 = in11 - in3;
        s7 = in11 + in3;
        s8 = in10 - in4;
        s9 = in10 + in4;
        s10 = in9 - in5;
        s11 = in9 + in5;
        s12 = in8 - in6;
        s13 = in8 + in6;

        s14 = s3 + s13;
        s15 = s5 + s11;
        s16 = s7 + s9;

        s17 = s13 - s3;
        s18 = s5 - s11;
        s19 = s9 - s7;

        // Output point 1: X(0)
        *out = s1 + s14 + s15 + s16;
        // Output point 13: X(14)
        out[out_strides[13]] = s0 + s17 + s18 + s19;

        t0 = CRTM_14_1 * s17;
        t1 = CRTM_14_3 * s18;
        t2 = CRTM_14_5 * s19;

        // Output point 2: X(1)
        out[out_strides[1]] = s0 - t0 + t1 - t2;

        s20 = s2 + s12;
        s21 = s4 + s10;
        s22 = s6 + s8;

        t3 = CRTM_14_2 * s20;
        t4 = CRTM_14_4 * s21;
        t5 = CRTM_14_6 * s22;

        // Output point 3: X(2)
        out[out_strides[2]] = t3 + t4 + t5;

        t6 = CRTM_14_1 * s16;
        t7 = CRTM_14_3 * s14;
        t8 = CRTM_14_5 * s15;

        // Output point 4: X(3)
        out[out_strides[3]] = s1 - t6 + t7 - t8;

        s23 = s2 - s12;
        s24 = s4 - s10;
        s25 = s6 - s8;

        t9 = CRTM_14_2 * s25;
        t10 = CRTM_14_4 * s23;
        t11 = CRTM_14_6 * s24;

        // Output point 5: X(4)
        out[out_strides[4]] = t9 + t10 + t11;

        t12 = CRTM_14_1 * s18;
        t13 = CRTM_14_3 * s19;
        t14 = CRTM_14_5 * s17;
        // Output point 6: X(5)
        out[out_strides[5]] = s0 - t12 + t13 - t14;
        t15 = CRTM_14_2 * s21;
        t16 = CRTM_14_4 * s22;
        t17 = CRTM_14_6 * s20;
        // Output point 7: X(6)
        out[out_strides[6]] = t15 - t16 + t17;

        t18 = CRTM_14_1 * s15;
        t19 = CRTM_14_3 * s16;
        t20 = CRTM_14_5 * s14;
        // Output point 8: X(7)
        out[out_strides[7]] = s1 - t18 + t19 - t20;
        t21 = CRTM_14_2 * s24;
        t22 = CRTM_14_4 * s25;
        t23 = CRTM_14_6 * s23;
        // Output point 9: X(8)
        out[out_strides[8]] = t23 - t22 - t21;

        t24 = CRTM_14_1 * s19;
        t25 = CRTM_14_3 * s17;
        t26 = CRTM_14_5 * s18;
        // Output point 10: X(9)
        out[out_strides[9]] = s0 - t24 + t25 - t26;
        t27 = CRTM_14_2 * s22;
        t28 = CRTM_14_4 * s20;
        t29 = CRTM_14_6 * s21;
        // Output point 11: X(10)
        out[out_strides[10]] = t27 + t28 - t29;

        t30 = CRTM_14_1 * s14;
        t31 = CRTM_14_3 * s15;
        t32 = CRTM_14_5 * s16;
        // Output point 5: X(4)
        out[out_strides[11]] = s1 - t30 + t31 - t32;
        t33 = CRTM_14_2 * s23;
        t34 = CRTM_14_4 * s24;
        t35 = CRTM_14_6 * s25;
        // Output point 5: X(4)
        out[out_strides[12]] = t33 - t34 + t35;

        in = in + v_in_stride;
        out = out + v_out_stride;
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
    }
}

static VOID r2hc_rfft14c_fp64_bwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                  VOID *out_imag, INTP n,
                                  aoclfftz_strides_t *strides, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const DOUBLE CRTM_14_1 = 0.867767478235116240951536665696717509219981456;
    const DOUBLE CRTM_14_2 = 1.801937735804838252472204639014890102331838324;
    const DOUBLE CRTM_14_3 = 1.563662964936059617416889053348115500464669038;
    const DOUBLE CRTM_14_4 = 1.246979603717467061050009768008479621264549462;
    const DOUBLE CRTM_14_5 = 1.949855824363647214036263365987862434465571602;
    const DOUBLE CRTM_14_6 = 0.445041867912628808577805128993589518932711138;
    const DOUBLE CRTM_14_7 = 2.000000000000000000000000000000000000000000000;

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
        DOUBLE in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11,
               in12, in13;
        DOUBLE s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14,
               s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25;
        DOUBLE t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14,
               t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27,
               t28, t29, t30, t31, t32, t33, t34, t35;
         

        // Input point 1: x(0)
        in0 = *in;
        // Input point 2: x(1)
        in1 = in[in_strides[1]];
        // Input point 3: x(2)
        in2 = in[in_strides[2]];
        // Input point 4: x(3)
        in3 = in[in_strides[3]];
        // Input point 5: x(4)
        in4 = in[in_strides[4]];
        // Input point 6: x(5)
        in5 = in[in_strides[5]];
        // Input point 7: x(6)
        in6 = in[in_strides[6]];
        // Input point 8: x(7)
        in7 = in[in_strides[7]];
        // Input point 9: x(8)
        in8 = in[in_strides[8]];
        // Input point 10: x(9)
        in9 = in[in_strides[9]];
        // Input point 11: x(10)
        in10 = in[in_strides[10]];
        // Input point 12: x(11)
        in11 = in[in_strides[11]];
        // Input point 13: x(12)
        in12 = in[in_strides[12]];
        // Input point 14: x(13)
        in13 = in[in_strides[13]];

        s0 = in0 - in13;
        s1 = in0 + in13;
        s2 = in11 - in1;
        s3 = in1 + in11;
        s4 = in2 - in12;
        s5 = in2 + in12;
        s6 = in3 - in9;
        s7 = in3 + in9;
        s8 = in4 - in10;
        s9 = in4 + in10;
        s10 = in7 - in5;
        s11 = in5 + in7;
        s12 = in6 - in8;
        s13 = in6 + in8;

        *out = CRTM_14_7 * (s11 + s3 + s7) + s1;
        out[out_strides[7]] = CRTM_14_7 * (s2 + s6 + s10) + s0;

        t0 = CRTM_14_1 * s5;
        t1 = CRTM_14_3 * s9;
        t2 = CRTM_14_5 * s13;
        t3 = CRTM_14_2 * s2;
        t4 = CRTM_14_4 * s6;
        t5 = CRTM_14_6 * s10;

        s14 = t4 - t5 - t3 + s0;
        s15 = t0 + t1 + t2;
        out[out_strides[1]] = s14 - s15;
        out[out_strides[13]] = s14 + s15;

        t6 = CRTM_14_1 * s12;
        t7 = CRTM_14_3 * s4;
        t8 = CRTM_14_5 * s8;

        t9 = CRTM_14_2 * s11;
        t10 = CRTM_14_4 * s3;
        t11 = CRTM_14_6 * s7;

        s16 = t10 - t9 - t11 + s1;
        s17 = t6 + t7 + t8;
        out[out_strides[2]] = s16 - s17;
        out[out_strides[12]] = s16 + s17;

        t12 = CRTM_14_1 * s9;
        t13 = CRTM_14_3 * s13;
        t14 = CRTM_14_5 * s5;
        t15 = CRTM_14_2 * s6;
        t16 = CRTM_14_4 * s10;
        t17 = CRTM_14_6 * s2;

        s18 = t16 - t15 - t17 + s0;
        s19 = t13 - t12 - t14;
        out[out_strides[3]] = s18 + s19;
        out[out_strides[11]] = s18 - s19;

        t18 = CRTM_14_1 * s8;
        t19 = CRTM_14_3 * s12;
        t20 = CRTM_14_5 * s4;
        t21 = CRTM_14_2 * s7;
        t22 = CRTM_14_4 * s11;
        t23 = CRTM_14_6 * -s3;

        s20 = t22 + t23 - t21 + s1;
        s21 = t18 + t19 - t20;
        out[out_strides[4]] = s20 + s21;
        out[out_strides[10]] = s20 - s21;

        t24 = CRTM_14_1 * s13;
        t25 = CRTM_14_3 * s5;
        t26 = CRTM_14_5 * s9;
        t27 = CRTM_14_2 * s10;
        t28 = CRTM_14_4 * s2;
        t29 = CRTM_14_6 * s6;

        s22 = t28 - t27 - t29 + s0;
        s23 = t26 - t24 - t25;
        out[out_strides[5]] = s22 + s23;
        out[out_strides[9]] = s22 - s23;

        t30 = CRTM_14_1 * s4;
        t31 = CRTM_14_3 * s8;
        t32 = CRTM_14_5 * s12;
        t33 = CRTM_14_2 * s3;
        t34 = CRTM_14_4 * s7;
        t35 = CRTM_14_6 * s11;

        s24 = t34 - t33 - t35 + s1;
        s25 = t31 - t30 - t32;
        out[out_strides[6]] = s24 + s25;
        out[out_strides[8]] = s24 - s25;

        in = in + v_in_stride;
        out = out + v_out_stride;
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
    }
}

kfft_ register_kernel_r2hc_rfft14c(UINT8 precision, UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hc_rfft14c_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft14c_fp64_fwd;
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
            return r2hc_rfft14c_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft14c_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
