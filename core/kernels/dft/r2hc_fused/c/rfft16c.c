// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft16c.c
 *
 *  @brief Radix-16 r2hc_fused Real-FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-16 real-to-halfcomplex fused of two
 *  different implementations (Standard DFT and Shifted DFT that differs in DFT
 *  weight matrix) using scalar operations for single-precision
 *  and double-precision inputs.
 *
 *  @author Jeya R
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                    {{{0, 42, 124, 64, 0, 0},
                                                      {0, 50, 124, 64, 0, 0}},
                                                     {{0, 42, 124, 64, 0, 0},
                                                      {0, 50, 124, 64, 0, 0}}};

ops_cycles_t get_ops_cnt_r2hcf_rfft16c(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction)
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

static FFTZ_VOID r2hcf_rfft16c_fp32_fwd(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                        FFTZ_VOID *out_real,
                                        FFTZ_VOID *out_imag, FFTZ_INTP n,
                                        aoclfftz_strides_t *strides,
                                        FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_16_1 =
        0.923879532511286756128183189396788286822416626f;
    const FFTZ_FLOAT CRTM_16_2 =
        0.382683432365089771728459984030398866761344562f;
    const FFTZ_FLOAT CRTM_16_3 =
        0.707106781186547524400844362104849039284835938f;
    const FFTZ_FLOAT CRTM_16_4 =
        0.555570233019602224742830813948532874374937191f;
    const FFTZ_FLOAT CRTM_16_5 =
        0.831469612302545237078788377617905756738560812f;
    const FFTZ_FLOAT CRTM_16_6 =
        0.980785280403230449126182236134239036973933731f;
    const FFTZ_FLOAT CRTM_16_7 =
        0.195090322016128267848284868477022240927691618f;

    FFTZ_FLOAT *in = (FFTZ_FLOAT *)in_real;
    FFTZ_FLOAT *out = (FFTZ_FLOAT *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        FFTZ_FLOAT a_in0, a_in1, a_in2, a_in3, a_in4, a_in5, a_in6, a_in7,
            a_in8, a_in9, a_in10, a_in11, a_in12, a_in13, a_in14, a_in15;
        FFTZ_FLOAT a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_s9,
            a_s10, a_s11, a_s12, a_s13, a_s14, a_s15, a_s16, a_s17, a_s18,
            a_s19, a_s20, a_s21, a_s22, a_s23, a_s24, a_s25, a_s26, a_s27,
            a_s28, a_s29, a_s30, a_s31, a_s32;
        FFTZ_FLOAT a_t0, a_t1, a_t2, a_t3, a_t4, a_t5, a_t6, a_t7, a_t8, a_t9,
            a_t10, a_t11, a_t12, a_t13, a_t14, a_t15, a_t16, a_t17, a_t18,
            a_t19, a_t20;

        // Input point 1: x(0)
        a_in0 = *in;
        // Input point 3: x(2)
        a_in1 = in[in_strides[2]];
        // Input point 5: x(4)
        a_in2 = in[in_strides[4]];
        // Input point 7: x(6)
        a_in3 = in[in_strides[6]];
        // Input point 9: x(8)
        a_in4 = in[in_strides[8]];
        // Input point 11: x(10)
        a_in5 = in[in_strides[10]];
        // Input point 13: x(12)
        a_in6 = in[in_strides[12]];
        // Input point 15: x(14)
        a_in7 = in[in_strides[14]];
        // Input point 17: x(16)
        a_in8 = in[in_strides[16]];
        // Input point 19: x(18)
        a_in9 = in[in_strides[18]];
        // Input point 21: x(20)
        a_in10 = in[in_strides[20]];
        // Input point 23: x(22)
        a_in11 = in[in_strides[22]];
        // Input point 25: x(24)
        a_in12 = in[in_strides[24]];
        // Input point 27: x(26)
        a_in13 = in[in_strides[26]];
        // Input point 29: x(28)
        a_in14 = in[in_strides[28]];
        // Input point 31: x(30)
        a_in15 = in[in_strides[30]];

        a_s0 = a_in0 + a_in8;
        a_s1 = a_in0 - a_in8;
        a_s2 = a_in1 + a_in15;
        a_s3 = a_in1 - a_in15;
        a_s4 = a_in2 + a_in6;
        a_s5 = a_in2 - a_in6;
        a_s6 = a_in3 + a_in5;
        a_s7 = a_in3 - a_in5;
        a_s8 = a_in4 + a_in12;
        a_s9 = a_in4 - a_in12;
        a_s10 = a_in7 + a_in9;
        a_s11 = a_in7 - a_in9;
        a_s12 = a_in10 + a_in14;
        a_s13 = a_in10 - a_in14;
        a_s14 = a_in11 + a_in13;
        a_s15 = a_in11 - a_in13;

        a_s16 = a_s0 + a_s8;
        a_s17 = a_s0 - a_s8;
        a_s18 = a_s2 + a_s10;
        a_s19 = a_s2 - a_s10;
        a_s20 = a_s3 + a_s11;
        a_s21 = a_s3 - a_s11;
        a_s22 = a_s4 + a_s12;
        a_s23 = a_s4 - a_s12;
        // Output point 16: X(15)
        out[out_strides[15]] = a_s16 - a_s22;

        a_t12 = a_s5 + a_s13;
        a_t16 = a_s5 - a_s13;
        a_t14 = a_s6 + a_s14;
        a_s24 = a_s6 - a_s14;
        a_t13 = a_s7 + a_s15;
        a_s25 = a_s7 - a_s15;
        // Output point 17: X(16)
        out[out_strides[16]] = a_t13 - a_s21;

        a_t15 = a_s18 + a_t14;
        a_s26 = a_s18 - a_t14;
        a_s27 = a_s16 + a_s22;
        a_t17 = a_s21 + a_t13;
        // Output point 1: X(0)
        *out = a_t15 + a_s27;
        // Output point 32: X(31)
        out[out_strides[31]] = a_s27 - a_t15;

        a_t0 = CRTM_16_1 * a_s19;
        a_t1 = CRTM_16_1 * a_s20;
        a_t2 = CRTM_16_1 * a_s24;
        a_t3 = CRTM_16_1 * a_s25;
        a_t4 = CRTM_16_2 * a_s19;
        a_t5 = CRTM_16_2 * a_s20;
        a_t6 = CRTM_16_2 * a_s24;
        a_t7 = CRTM_16_2 * a_s25;
        a_t8 = CRTM_16_3 * a_s23;
        a_t9 = CRTM_16_3 * a_t16;
        a_t10 = CRTM_16_3 * a_s26;
        // Output point 8: X(7)
        out[out_strides[7]] = a_s17 + a_t10;
        // Output point 24: X(23)
        out[out_strides[23]] = a_s17 - a_t10;

        a_t11 = CRTM_16_3 * a_t17;
        // Output point 9: X(8)
        out[out_strides[8]] = -(a_t12 + a_t11);
        // Output point 25: X(24)
        out[out_strides[24]] = a_t12 - a_t11;

        a_s28 = a_t0 + a_t7;
        a_t18 = a_t9 + a_s1;
        // Output point 4: X(3)
        out[out_strides[3]] = a_s28 + a_t18;
        // Output point 28: X(27)
        out[out_strides[27]] = a_t18 - a_s28;

        a_t20 = a_t2 + a_t5;
        a_t19 = a_t8 + a_s9;
        // Output point 5: X(4)
        out[out_strides[4]] = -(a_t20 + a_t19);
        // Output point 29: X(28)
        out[out_strides[28]] = a_t19 - a_t20;

        a_s29 = a_t3 - a_t4;
        a_s30 = a_s1 - a_t9;
        // Output point 12: X(11)
        out[out_strides[11]] = a_s30 - a_s29;
        // Output point 20: X(19)
        out[out_strides[19]] = a_s29 + a_s30;

        a_s31 = a_t6 - a_t1;
        a_s32 = a_s9 - a_t8;
        // Output point 13: X(12)
        out[out_strides[12]] = a_s31 + a_s32;
        // Output point 21: X(20)
        out[out_strides[20]] = a_s31 - a_s32;

        // Shifted DFT
        FFTZ_FLOAT b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7,
            b_in8, b_in9, b_in10, b_in11, b_in12, b_in13, b_in14, b_in15;
        FFTZ_FLOAT b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9,
              b_s10, b_s11, b_s12, b_s13, b_s14, b_s15, b_s16, b_s17, b_s18,
              b_s19, b_s20, b_s21, b_s22, b_s23, b_s24, b_s25, b_s26, b_s27,
              b_s28, b_s29, b_s30, b_s31, b_s32, b_s33, b_s34;
        FFTZ_FLOAT b_t0, b_t1, b_t2, b_t3, b_t4, b_t5, b_t6, b_t7, b_t8, b_t9,
            b_t10, b_t11, b_t12, b_t13, b_t14, b_t15, b_t16, b_t17, b_t18,
            b_t19, b_t20;

        // Input point 2: x(1)
        b_in0 = in[in_strides[1]];
        // Input point 4: x(3)
        b_in1 = in[in_strides[3]];
        // Input point 6: x(5)
        b_in2 = in[in_strides[5]];
        // Input point 8: x(7)
        b_in3 = in[in_strides[7]];
        // Input point 10: x(9)
        b_in4 = in[in_strides[9]];
        // Input point 12: x(11)
        b_in5 = in[in_strides[11]];
        // Input point 14: x(13)
        b_in6 = in[in_strides[13]];
        // Input point 16: x(15)
        b_in7 = in[in_strides[15]];
        // Input point 18: x(17)
        b_in8 = in[in_strides[17]];
        // Input point 20: x(19)
        b_in9 = in[in_strides[19]];
        // Input point 22: x(21)
        b_in10 = in[in_strides[21]];
        // Input point 24: x(23)
        b_in11 = in[in_strides[23]];
        // Input point 26: x(25)
        b_in12 = in[in_strides[25]];
        // Input point 28: x(27)
        b_in13 = in[in_strides[27]];
        // Input point 30: x(29)
        b_in14 = in[in_strides[29]];
        // Input point 32: x(31)
        b_in15 = in[in_strides[31]];

        b_s0 = b_in3 - b_in11;
        b_s1 = b_in3 + b_in11;
        b_s2 = b_in4 - b_in12;
        b_s3 = b_in4 + b_in12;
        b_s4 = b_in5 - b_in13;
        b_s5 = b_in5 + b_in13;

        b_t0 = CRTM_16_3 * b_s2;
        b_s6 = b_t0 + b_in0;
        b_t1 = (CRTM_16_1 * b_in2) - (CRTM_16_2 * b_in10);
        b_t2 = (CRTM_16_2 * b_in6) - (CRTM_16_1 * b_in14);
        b_s7 = b_t1 + b_t2;
        b_s8 = b_s6 - b_s7;
        b_s9 = b_s6 + b_s7;

        b_t3 = CRTM_16_3 * b_s4;
        b_s10 = b_t3 + b_in1;
        b_t4 = CRTM_16_3 * b_s5;
        b_s11 = b_t4 + b_in9;

        b_t5 = (CRTM_16_6 * b_s10) - (CRTM_16_7 * b_s11);

        b_t6 = CRTM_16_3 * b_s1;
        b_s12 = b_t6 + b_in7;
        b_t7 = CRTM_16_3 * b_s0;
        b_s13 = b_t7 - b_in15;

        b_t8 = (CRTM_16_7 * b_s12) + (CRTM_16_6 * b_s13);

        b_s14 = b_t5 + b_t8;
        b_s15 = b_t8 - b_t5;

        // Output point 2: X(1)
        out[out_strides[1]] = b_s9 + b_s14;
        // Output point 30: X(29)
        out[out_strides[29]] = b_s9 - b_s14;

        b_t9 = CRTM_16_3 * b_s3;
        b_s16 = b_t9 + b_in8;

        b_t10 = (CRTM_16_2 * b_in2) + (CRTM_16_1 * b_in10);
        b_t11 = (CRTM_16_1 * b_in6) + (CRTM_16_2 * b_in14);
        b_s17 = b_t10 + b_t11;
        b_s18 = b_s16 - b_s17;
        b_s19 = b_s16 + b_s17;
        // Output point 15: X(14)
        out[out_strides[14]] = b_s18 + b_s15;
        // Output point 19: X(18)
        out[out_strides[18]] = b_s15 - b_s18;

        b_t12 = (CRTM_16_6 * b_s11) + (CRTM_16_7 * b_s10);
        b_t13 = (CRTM_16_7 * b_s13) - (CRTM_16_6 * b_s12);

        b_s20 = b_t13 + b_t12;
        b_s21 = b_t13 - b_t12;

        // Output point 3: X(2)
        out[out_strides[2]] = b_s21 - b_s19;
        // Output point 31: X(30)
        out[out_strides[30]] = b_s21 + b_s19;

        // Output point 18: X(17)
        out[out_strides[17]] = b_s8 - b_s20;
        // Output point 14: X(13)
        out[out_strides[13]] = b_s8 + b_s20;

        b_s22 = b_in0 - b_t0;

        b_t18 = b_t10 - b_t11;
        b_t19 = b_s22 + b_t18;
        b_s23 = b_s22 - b_t18;

        b_s24 = b_in9 - b_t4;
        b_t20 = b_in1 - b_t3;

        b_t14 = (CRTM_16_4 * b_s24) + (CRTM_16_5 * b_t20);

        b_s25 = b_in7 - b_t6;
        b_s26 = b_in15 + b_t7;

        b_t15 = (CRTM_16_4 * b_s25) + (CRTM_16_5 * b_s26);

        b_s27 = b_t14 - b_t15;
        b_s28 = b_t14 + b_t15;

        // Output point 6: X(5)
        out[out_strides[5]] = b_t19 + b_s27;
        // Output point 26: X(25)
        out[out_strides[25]] = b_t19 - b_s27;

        b_s29 = b_in8 - b_t9;
        b_s30 = b_t2 - b_t1;
        b_s31 = b_s30 - b_s29;
        b_s32 = b_s30 + b_s29;
        // Output point 11: X(10)
        out[out_strides[10]] = b_s31 - b_s28;
        // Output point 23: X(22)
        out[out_strides[22]] = -(b_s31 + b_s28);

        b_t16 = (CRTM_16_5 * b_s25) - (CRTM_16_4 * b_s26);
        b_t17 = (CRTM_16_5 * b_s24) - (CRTM_16_4 * b_t20);

        b_s33 = b_t16 - b_t17;
        b_s34 = b_t17 + b_t16;

        // Output point 22: X(21)
        out[out_strides[21]] = b_s23 - b_s33;
        // Output point 10: X(9)
        out[out_strides[9]] = b_s23 + b_s33;

        // Output point 27: X(26)
        out[out_strides[26]] = b_s34 - b_s32;
        // Output point 7: X(6)
        out[out_strides[6]] = b_s34 + b_s32;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft16c_fp32_bwd(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                        FFTZ_VOID *out_real,
                                        FFTZ_VOID *out_imag, FFTZ_INTP n,
                                        aoclfftz_strides_t *strides,
                                        FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_16_1 =
        1.847759065022573512256366378793576573644833252f;
    const FFTZ_FLOAT CRTM_16_2 =
        0.765366864730179543456919968060797733522689125f;
    const FFTZ_FLOAT CRTM_16_3 =
        1.414213562373095048801688724209698078569671875f;
    const FFTZ_FLOAT CRTM_16_4 =
        2.000000000000000000000000000000000000000000000f;
    const FFTZ_FLOAT CRTM_16_5 =
        0.707106781186547524400844362104849039284835938f;
    const FFTZ_FLOAT CRTM_16_6 =
        1.961570560806460898252364472268478073947867462f;
    const FFTZ_FLOAT CRTM_16_7 =
        0.390180644032256535696569736954044481855383236f;
    const FFTZ_FLOAT CRTM_16_8 =
        1.111140466039204449485661627897065748749874382f;
    const FFTZ_FLOAT CRTM_16_9 =
        1.662939224605090474157576755235811513477121624f;

    FFTZ_FLOAT *in = (FFTZ_FLOAT *)in_real;
    FFTZ_FLOAT *out = (FFTZ_FLOAT *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        FFTZ_FLOAT a_in0, a_in1, a_in2, a_in3, a_in4, a_in5, a_in6, a_in7,
            a_in8, a_in9, a_in10, a_in11, a_in12, a_in13, a_in14, a_in15;
        FFTZ_FLOAT a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_s9,
            a_s10, a_s11, a_s12, a_s13, a_s14, a_s15, a_s16, a_s17, a_s18,
            a_s19, a_s20, a_s21, a_s22, a_s23, a_s24, a_s25, a_s26, a_s27,
            a_s28, a_s29, a_s30, a_s31, a_s32, a_s33, a_s34, a_s35, a_s36,
            a_s37, a_s38;
        FFTZ_FLOAT a_t0, a_t1, a_t2, a_t3, a_t4, a_t5, a_t6, a_t7, a_t8, a_t9,
            a_t10, a_t11, a_t12, a_t13, a_t14, a_t15, a_t16, a_t17, a_t18,
            a_t19, a_t20;

        // Input point 1: x(0)
        a_in0 = *in;
        // Input point 4: x(3)
        a_in1 = in[in_strides[3]];
        // Input point 5: x(4)
        a_in2 = in[in_strides[4]];
        // Input point 8: x(7)
        a_in3 = in[in_strides[7]];
        // Input point 9: x(8)
        a_in4 = in[in_strides[8]];
        // Input point 12: x(11)
        a_in5 = in[in_strides[11]];
        // Input point 13: x(12)
        a_in6 = in[in_strides[12]];
        // Input point 16: x(15)
        a_in7 = in[in_strides[15]];
        // Input point 17: x(16)
        a_in8 = in[in_strides[16]];
        // Input point 20: x(19)
        a_in9 = in[in_strides[19]];
        // Input point 21: x(20)
        a_in10 = in[in_strides[20]];
        // Input point 24: x(23)
        a_in11 = in[in_strides[23]];
        // Input point 25: x(24)
        a_in12 = in[in_strides[24]];
        // Input point 28: x(27)
        a_in13 = in[in_strides[27]];
        // Input point 29: x(28)
        a_in14 = in[in_strides[28]];
        // Input point 32: x(31)
        a_in15 = in[in_strides[31]];

        a_s0 = a_in0 + a_in15;
        a_s1 = a_in0 - a_in15;
        a_s2 = a_in1 + a_in13;
        a_s3 = a_in1 - a_in13;
        a_s4 = a_in2 + a_in14;
        a_s5 = a_in2 - a_in14;
        a_s6 = a_in3 + a_in11;
        a_s7 = a_in3 - a_in11;
        a_s8 = a_in4 + a_in12;
        a_s9 = a_in4 - a_in12;
        a_s10 = a_in5 + a_in9;
        a_s11 = a_in5 - a_in9;
        a_s12 = a_in6 + a_in10;
        a_s13 = a_in6 - a_in10;

        a_s14 = a_s2 + a_s10;
        a_s15 = a_s2 - a_s10;
        a_s16 = a_s3 + a_s12;
        a_s17 = a_s3 - a_s12;
        a_s18 = a_s4 + a_s11;
        a_s19 = a_s4 - a_s11;
        a_s20 = a_s5 + a_s13;
        a_s21 = a_s5 - a_s13;
        a_s22 = a_s7 + a_s8;
        a_s23 = a_s7 - a_s8;

        a_t12 = CRTM_16_4 * a_in7;
        a_t13 = CRTM_16_4 * a_s14;
        a_t14 = CRTM_16_4 * a_s6;
        a_s24 = a_s0 + a_t12;
        a_s26 = a_t14 + a_s24;
        // Output point 1: X(0)
        *out = a_s26 + a_t13;
        // Output point 17: X(16)
        out[out_strides[16]] = a_s26 - a_t13;

        a_t15 = CRTM_16_4 * a_s21;
        a_s27 = a_s24 - a_t14;
        // Output point 9: X(8)
        out[out_strides[8]] = a_s27 - a_t15;
        // Output point 25: X(24)
        out[out_strides[24]] = a_s27 + a_t15;

        a_t16 = CRTM_16_4 * a_in8;
        a_t17 = CRTM_16_3 * a_s23;
        a_t0 = CRTM_16_1 * a_s17;
        a_t1 = CRTM_16_2 * a_s19;
        a_t3 = a_s1 - a_t16;
        a_s29 = a_t17 + a_t3;
        a_s30 = a_t0 - a_t1;
        // Output point 3: X(2)
        out[out_strides[2]] = a_s29 + a_s30;
        // Output point 19: X(18)
        out[out_strides[18]] = a_s29 - a_s30;

        a_s25 = a_s0 - a_t12;
        a_t2 = a_s1 + a_t16;
        a_t4 = CRTM_16_3 * a_s22;
        a_t5 = CRTM_16_1 * a_s18;
        a_t6 = CRTM_16_2 * a_s16;
        a_s31 = a_t2 - a_t4;
        a_s32 = a_t6 - a_t5;
        // Output point 7: X(6)
        out[out_strides[6]] = a_s31 + a_s32;
        // Output point 23: X(22)
        out[out_strides[22]] = a_s31 - a_s32;

        a_t9 = CRTM_16_1 * a_s16;
        a_t10 = CRTM_16_2 * a_s18;
        a_s35 = a_t2 + a_t4;
        a_s36 = a_t9 + a_t10;
        // Output point 15: X(14)
        out[out_strides[14]] = a_s35 - a_s36;
        // Output point 31: X(30)
        out[out_strides[30]] = a_s35 + a_s36;

        a_t7 = CRTM_16_1 * a_s19;
        a_t8 = CRTM_16_2 * a_s17;
        a_s33 = a_t3 - a_t17;
        a_s34 = a_t7 + a_t8;
        // Output point 11: X(10)
        out[out_strides[10]] = a_s33 - a_s34;
        // Output point 27: X(26)
        out[out_strides[26]] = a_s33 + a_s34;

        a_s28 = a_s15 - a_s20;
        a_t18 = CRTM_16_3 * a_s28;
        a_t19 = CRTM_16_4 * a_s9;
        a_s37 = a_s25 - a_t19;
        // Output point 5: X(4)
        out[out_strides[4]] = a_s37 + a_t18;
        // Output point 21: X(20)
        out[out_strides[20]] = a_s37 - a_t18;

        a_t11 = a_s15 + a_s20;
        a_t20 = CRTM_16_3 * a_t11;
        a_s38 = a_s25 + a_t19;
        // Output point 13: X(12)
        out[out_strides[12]] = a_s38 - a_t20;
        // Output point 29: X(28)
        out[out_strides[28]] = a_s38 + a_t20;

        // Shifted DFT
        FFTZ_FLOAT b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7,
            b_in8, b_in9, b_in10, b_in11, b_in12, b_in13, b_in14, b_in15;
        FFTZ_FLOAT b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9, b_s10,
              b_s11, b_s12, b_s13, b_s14, b_s15, b_s16, b_s17, b_s18, b_s19,
              b_s20, b_s21, b_s22, b_s23, b_s24, b_s25, b_s26, b_s27, b_s28,
              b_s29, b_s30, b_s31;
        FFTZ_FLOAT b_t0, b_t1, b_t2, b_t3, b_t4, b_t5, b_t6, b_t7, b_t8, b_t9,
            b_t10, b_t11, b_t12, b_t13, b_t14, b_t15, b_t16, b_t17, b_t18;

        //  Input point 2: x(1)
        b_in0 = in[in_strides[1]];
        // Input point 3: x(2)
        b_in1 = in[in_strides[2]];
        // Input point 6: x(5)
        b_in2 = in[in_strides[5]];
        // Input point 7: x(6)
        b_in3 = in[in_strides[6]];
        // Input point 10: x(9)
        b_in4 = in[in_strides[9]];
        // Input point 11: x(10)
        b_in5 = in[in_strides[10]];
        // Input point 14: x(13)
        b_in6 = in[in_strides[13]];
        // Input point 15: x(14)
        b_in7 = in[in_strides[14]];
        // Input point 18: x(17)
        b_in8 = in[in_strides[17]];
        // Input point 19: x(18)
        b_in9 = in[in_strides[18]];
        // Input point 22: x(21)
        b_in10 = in[in_strides[21]];
        // Input point 23: x(22)
        b_in11 = in[in_strides[22]];
        // Input point 26: x(25)
        b_in12 = in[in_strides[25]];
        // Input point 27: x(26)
        b_in13 = in[in_strides[26]];
        // Input point 30: x(29)
        b_in14 = in[in_strides[29]];
        // Input point 31: x(30)
        b_in15 = in[in_strides[30]];

        b_s1 = b_in14 + b_in0;
        b_s2 = b_in14 - b_in0;
        b_s3 = b_in15 + b_in1;
        b_s4 = b_in15 - b_in1;
        b_s5 = b_in12 + b_in2;
        b_t0 = b_in12 - b_in2;
        b_s6 = b_in13 + b_in3;
        b_t1 = b_in13 - b_in3;
        b_t2 = b_in10 + b_in4;
        b_s7 = b_in10 - b_in4;
        b_s8 = b_in11 + b_in5;
        b_s9 = b_in11 - b_in5;
        b_t3 = b_in8 + b_in6;
        b_s10 = b_in8 - b_in6;
        b_t4 = b_in9 + b_in7;
        b_s11 = b_in9 - b_in7;

        b_t5 = b_s1 - b_t3;
        b_t6 = b_s9 + b_t1;
        b_s12 = b_t5 + b_t6;
        b_t7 = b_t6 - b_t5;

        b_s13 = b_s4 + b_s11;
        b_t8 = b_s5 - b_t2;
        b_s14 = b_s13 + b_t8;
        b_s15 = b_s13 - b_t8;

        // Output point 6: X(5)
        out[out_strides[5]] = (CRTM_16_1 * b_s12) + (CRTM_16_2 * b_s14);
        // Output point 22: X(21)
        out[out_strides[21]] = (CRTM_16_1 * b_s14) - (CRTM_16_2 * b_s12);
        // Output point 30: X(29)
        out[out_strides[29]] = (CRTM_16_1 * b_t7) + (CRTM_16_2 * b_s15);
        // Output point 14: X(13)
        out[out_strides[13]] = (CRTM_16_1 * b_s15) - (CRTM_16_2 * b_t7);

        b_t9 = b_s1 + b_t3;
        b_s16 = b_s5 + b_t2;
        // Output point 2: X(1)
        out[out_strides[1]] = CRTM_16_4 * (b_t9 + b_s16);

        b_t10 = b_s4 - b_s11;
        b_t11 = b_s9 - b_t1;
        // Output point 18: X(17)
        out[out_strides[17]] = CRTM_16_4 * (b_t10 + b_t11);

        b_s17 = b_t9 - b_s16;
        b_s18 = b_t10 - b_t11;

        // Output point 10: X(9)
        out[out_strides[9]] = CRTM_16_3 * (b_s17 + b_s18);
        // Output point 26: X(25)
        out[out_strides[25]] = CRTM_16_3 * (b_s18 - b_s17);

        b_s19 = b_s2 + b_t4;
        b_t12 = -b_s3 - b_s10;

        b_t13 = b_s10 - b_s3;
        b_s20 = b_t4 - b_s2;

        b_s21 = b_s8 + b_t0;
        b_s22 = b_s7 + b_s6;

        b_t16 = CRTM_16_5 * (b_s21 + b_s22);
        b_t17 = CRTM_16_5 * (b_s22 - b_s21);

        b_s25 = b_t16 + b_s19;
        b_s26 = b_t12 + b_t17;
        b_t15 = b_s19 - b_t16;
        b_s27 = b_t12 - b_t17;

        // Output point 4: X(3)
        out[out_strides[3]] = (CRTM_16_7 * b_s26) - (CRTM_16_6 * b_s25);
        // Output point 12: X(11)
        out[out_strides[11]] = (CRTM_16_9 * b_s27) - (CRTM_16_8 * b_t15);
        // Output point 28: X(27)
        out[out_strides[27]] = (CRTM_16_9 * b_t15) + (CRTM_16_8 * b_s27);
        // Output point 20: X(19)
        out[out_strides[19]] = (CRTM_16_6 * b_s26) + (CRTM_16_7 * b_s25);

        b_s23 = b_t0 - b_s8;
        b_s24 = b_s7 - b_s6;

        b_t18 = CRTM_16_5 * (b_s23 - b_s24);
        b_t14 = CRTM_16_5 * (b_s23 + b_s24);

        b_s28 = b_t13 + b_t18;
        b_s29 = b_s20 + b_t14;
        b_s30 = b_t13 - b_t18;
        b_s31 = b_s20 - b_t14;

        // Output point 16: X(15)
        out[out_strides[15]] = (CRTM_16_6 * b_s28) + (CRTM_16_7 * b_s31);
        // Output point 32: X(31)
        out[out_strides[31]] = (CRTM_16_7 * b_s28) - (CRTM_16_6 * b_s31);
        // Output point 8: X(7)
        out[out_strides[7]] = (CRTM_16_9 * b_s29) + (CRTM_16_8 * b_s30);
        // Output point 24: X(23)
        out[out_strides[23]] = (CRTM_16_9 * b_s30) - (CRTM_16_8 * b_s29);

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft16c_fp64_fwd(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                        FFTZ_VOID *out_real,
                                        FFTZ_VOID *out_imag, FFTZ_INTP n,
                                        aoclfftz_strides_t *strides,
                                        FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_16_1 =
        0.923879532511286756128183189396788286822416626;
    const FFTZ_DOUBLE CRTM_16_2 =
        0.382683432365089771728459984030398866761344562;
    const FFTZ_DOUBLE CRTM_16_3 =
        0.707106781186547524400844362104849039284835938;
    const FFTZ_DOUBLE CRTM_16_4 =
        0.555570233019602224742830813948532874374937191;
    const FFTZ_DOUBLE CRTM_16_5 =
        0.831469612302545237078788377617905756738560812;
    const FFTZ_DOUBLE CRTM_16_6 =
        0.980785280403230449126182236134239036973933731;
    const FFTZ_DOUBLE CRTM_16_7 =
        0.195090322016128267848284868477022240927691618;

    FFTZ_DOUBLE *in = (FFTZ_DOUBLE *)in_real;
    FFTZ_DOUBLE *out = (FFTZ_DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        FFTZ_DOUBLE a_in0, a_in1, a_in2, a_in3, a_in4, a_in5, a_in6, a_in7,
            a_in8, a_in9, a_in10, a_in11, a_in12, a_in13, a_in14, a_in15;
        FFTZ_DOUBLE a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_s9,
               a_s10, a_s11, a_s12, a_s13, a_s14, a_s15, a_s16, a_s17, a_s18,
               a_s19, a_s20, a_s21, a_s22, a_s23, a_s24, a_s25, a_s26, a_s27,
               a_s28, a_s29, a_s30, a_s31, a_s32;
        FFTZ_DOUBLE a_t0, a_t1, a_t2, a_t3, a_t4, a_t5, a_t6, a_t7, a_t8, a_t9,
               a_t10, a_t11, a_t12, a_t13, a_t14, a_t15, a_t16, a_t17, a_t18,
               a_t19, a_t20;

        // Input point 1: x(0)
        a_in0 = *in;
        // Input point 3: x(2)
        a_in1 = in[in_strides[2]];
        // Input point 5: x(4)
        a_in2 = in[in_strides[4]];
        // Input point 7: x(6)
        a_in3 = in[in_strides[6]];
        // Input point 9: x(8)
        a_in4 = in[in_strides[8]];
        // Input point 11: x(10)
        a_in5 = in[in_strides[10]];
        // Input point 13: x(12)
        a_in6 = in[in_strides[12]];
        // Input point 15: x(14)
        a_in7 = in[in_strides[14]];
        // Input point 17: x(16)
        a_in8 = in[in_strides[16]];
        // Input point 19: x(18)
        a_in9 = in[in_strides[18]];
        // Input point 21: x(20)
        a_in10 = in[in_strides[20]];
        // Input point 23: x(22)
        a_in11 = in[in_strides[22]];
        // Input point 25: x(24)
        a_in12 = in[in_strides[24]];
        // Input point 27: x(26)
        a_in13 = in[in_strides[26]];
        // Input point 29: x(28)
        a_in14 = in[in_strides[28]];
        // Input point 31: x(30)
        a_in15 = in[in_strides[30]];

        a_s0 = a_in0 + a_in8;
        a_s1 = a_in0 - a_in8;
        a_s2 = a_in1 + a_in15;
        a_s3 = a_in1 - a_in15;
        a_s4 = a_in2 + a_in6;
        a_s5 = a_in2 - a_in6;
        a_s6 = a_in3 + a_in5;
        a_s7 = a_in3 - a_in5;
        a_s8 = a_in4 + a_in12;
        a_s9 = a_in4 - a_in12;
        a_s10 = a_in7 + a_in9;
        a_s11 = a_in7 - a_in9;
        a_s12 = a_in10 + a_in14;
        a_s13 = a_in10 - a_in14;
        a_s14 = a_in11 + a_in13;
        a_s15 = a_in11 - a_in13;

        a_s16 = a_s0 + a_s8;
        a_s17 = a_s0 - a_s8;
        a_s18 = a_s2 + a_s10;
        a_s19 = a_s2 - a_s10;
        a_s20 = a_s3 + a_s11;
        a_s21 = a_s3 - a_s11;
        a_s22 = a_s4 + a_s12;
        a_s23 = a_s4 - a_s12;
        // Output point 16: X(15)
        out[out_strides[15]] = a_s16 - a_s22;

        a_t12 = a_s5 + a_s13;
        a_t16 = a_s5 - a_s13;
        a_t14 = a_s6 + a_s14;
        a_s24 = a_s6 - a_s14;
        a_t13 = a_s7 + a_s15;
        a_s25 = a_s7 - a_s15;
        // Output point 17: X(16)
        out[out_strides[16]] = a_t13 - a_s21;

        a_t15 = a_s18 + a_t14;
        a_s26 = a_s18 - a_t14;
        a_s27 = a_s16 + a_s22;
        a_t17 = a_s21 + a_t13;
        // Output point 1: X(0)
        *out = a_t15 + a_s27;
        // Output point 32: X(31)
        out[out_strides[31]] = a_s27 - a_t15;

        a_t0 = CRTM_16_1 * a_s19;
        a_t1 = CRTM_16_1 * a_s20;
        a_t2 = CRTM_16_1 * a_s24;
        a_t3 = CRTM_16_1 * a_s25;
        a_t4 = CRTM_16_2 * a_s19;
        a_t5 = CRTM_16_2 * a_s20;
        a_t6 = CRTM_16_2 * a_s24;
        a_t7 = CRTM_16_2 * a_s25;
        a_t8 = CRTM_16_3 * a_s23;
        a_t9 = CRTM_16_3 * a_t16;
        a_t10 = CRTM_16_3 * a_s26;
        // Output point 8: X(7)
        out[out_strides[7]] = a_s17 + a_t10;
        // Output point 24: X(23)
        out[out_strides[23]] = a_s17 - a_t10;

        a_t11 = CRTM_16_3 * a_t17;
        // Output point 9: X(8)
        out[out_strides[8]] = -(a_t12 + a_t11);
        // Output point 25: X(24)
        out[out_strides[24]] = a_t12 - a_t11;

        a_s28 = a_t0 + a_t7;
        a_t18 = a_t9 + a_s1;
        // Output point 4: X(3)
        out[out_strides[3]] = a_s28 + a_t18;
        // Output point 28: X(27)
        out[out_strides[27]] = a_t18 - a_s28;

        a_t20 = a_t2 + a_t5;
        a_t19 = a_t8 + a_s9;
        // Output point 5: X(4)
        out[out_strides[4]] = -(a_t20 + a_t19);
        // Output point 29: X(28)
        out[out_strides[28]] = a_t19 - a_t20;

        a_s29 = a_t3 - a_t4;
        a_s30 = a_s1 - a_t9;
        // Output point 12: X(11)
        out[out_strides[11]] = a_s30 - a_s29;
        // Output point 20: X(19)
        out[out_strides[19]] = a_s29 + a_s30;

        a_s31 = a_t6 - a_t1;
        a_s32 = a_s9 - a_t8;
        // Output point 13: X(12)
        out[out_strides[12]] = a_s31 + a_s32;
        // Output point 21: X(20)
        out[out_strides[20]] = a_s31 - a_s32;

        // Shifted DFT
        FFTZ_DOUBLE b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7,
            b_in8, b_in9, b_in10, b_in11, b_in12, b_in13, b_in14, b_in15;
        FFTZ_DOUBLE b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9,
               b_s10, b_s11, b_s12, b_s13, b_s14, b_s15, b_s16, b_s17, b_s18,
               b_s19, b_s20, b_s21, b_s22, b_s23, b_s24, b_s25, b_s26, b_s27,
               b_s28, b_s29, b_s30, b_s31, b_s32, b_s33, b_s34;
        FFTZ_DOUBLE b_t0, b_t1, b_t2, b_t3, b_t4, b_t5, b_t6, b_t7, b_t8, b_t9,
               b_t10, b_t11, b_t12, b_t13, b_t14, b_t15, b_t16, b_t17, b_t18,
               b_t19, b_t20;

        // Input point 2: x(1)
        b_in0 = in[in_strides[1]];
        // Input point 4: x(3)
        b_in1 = in[in_strides[3]];
        // Input point 6: x(5)
        b_in2 = in[in_strides[5]];
        // Input point 8: x(7)
        b_in3 = in[in_strides[7]];
        // Input point 10: x(9)
        b_in4 = in[in_strides[9]];
        // Input point 12: x(11)
        b_in5 = in[in_strides[11]];
        // Input point 14: x(13)
        b_in6 = in[in_strides[13]];
        // Input point 16: x(15)
        b_in7 = in[in_strides[15]];
        // Input point 18: x(17)
        b_in8 = in[in_strides[17]];
        // Input point 20: x(19)
        b_in9 = in[in_strides[19]];
        // Input point 22: x(21)
        b_in10 = in[in_strides[21]];
        // Input point 24: x(23)
        b_in11 = in[in_strides[23]];
        // Input point 26: x(25)
        b_in12 = in[in_strides[25]];
        // Input point 28: x(27)
        b_in13 = in[in_strides[27]];
        // Input point 30: x(29)
        b_in14 = in[in_strides[29]];
        // Input point 32: x(31)
        b_in15 = in[in_strides[31]];

        b_s0 = b_in3 - b_in11;
        b_s1 = b_in3 + b_in11;
        b_s2 = b_in4 - b_in12;
        b_s3 = b_in4 + b_in12;
        b_s4 = b_in5 - b_in13;
        b_s5 = b_in5 + b_in13;

        b_t0 = CRTM_16_3 * b_s2;
        b_s6 = b_t0 + b_in0;
        b_t1 = (CRTM_16_1 * b_in2) - (CRTM_16_2 * b_in10);
        b_t2 = (CRTM_16_2 * b_in6) - (CRTM_16_1 * b_in14);
        b_s7 = b_t1 + b_t2;
        b_s8 = b_s6 - b_s7;
        b_s9 = b_s6 + b_s7;

        b_t3 = CRTM_16_3 * b_s4;
        b_s10 = b_t3 + b_in1;
        b_t4 = CRTM_16_3 * b_s5;
        b_s11 = b_t4 + b_in9;

        b_t5 = (CRTM_16_6 * b_s10) - (CRTM_16_7 * b_s11);

        b_t6 = CRTM_16_3 * b_s1;
        b_s12 = b_t6 + b_in7;
        b_t7 = CRTM_16_3 * b_s0;
        b_s13 = b_t7 - b_in15;

        b_t8 = (CRTM_16_7 * b_s12) + (CRTM_16_6 * b_s13);

        b_s14 = b_t5 + b_t8;
        b_s15 = b_t8 - b_t5;

        // Output point 2: X(1)
        out[out_strides[1]] = b_s9 + b_s14;
        // Output point 30: X(29)
        out[out_strides[29]] = b_s9 - b_s14;

        b_t9 = CRTM_16_3 * b_s3;
        b_s16 = b_t9 + b_in8;

        b_t10 = (CRTM_16_2 * b_in2) + (CRTM_16_1 * b_in10);
        b_t11 = (CRTM_16_1 * b_in6) + (CRTM_16_2 * b_in14);
        b_s17 = b_t10 + b_t11;
        b_s18 = b_s16 - b_s17;
        b_s19 = b_s16 + b_s17;
        // Output point 15: X(14)
        out[out_strides[14]] = b_s18 + b_s15;
        // Output point 19: X(18)
        out[out_strides[18]] = b_s15 - b_s18;

        b_t12 = (CRTM_16_6 * b_s11) + (CRTM_16_7 * b_s10);
        b_t13 = (CRTM_16_7 * b_s13) - (CRTM_16_6 * b_s12);

        b_s20 = b_t13 + b_t12;
        b_s21 = b_t13 - b_t12;

        // Output point 3: X(2)
        out[out_strides[2]] = b_s21 - b_s19;
        // Output point 31: X(30)
        out[out_strides[30]] = b_s21 + b_s19;

        // Output point 18: X(17)
        out[out_strides[17]] = b_s8 - b_s20;
        // Output point 14: X(13)
        out[out_strides[13]] = b_s8 + b_s20;

        b_s22 = b_in0 - b_t0;

        b_t18 = b_t10 - b_t11;
        b_t19 = b_s22 + b_t18;
        b_s23 = b_s22 - b_t18;

        b_s24 = b_in9 - b_t4;
        b_t20 = b_in1 - b_t3;

        b_t14 = (CRTM_16_4 * b_s24) + (CRTM_16_5 * b_t20);

        b_s25 = b_in7 - b_t6;
        b_s26 = b_in15 + b_t7;

        b_t15 = (CRTM_16_4 * b_s25) + (CRTM_16_5 * b_s26);

        b_s27 = b_t14 - b_t15;
        b_s28 = b_t14 + b_t15;

        // Output point 6: X(5)
        out[out_strides[5]] = b_t19 + b_s27;
        // Output point 26: X(25)
        out[out_strides[25]] = b_t19 - b_s27;

        b_s29 = b_in8 - b_t9;
        b_s30 = b_t2 - b_t1;
        b_s31 = b_s30 - b_s29;
        b_s32 = b_s30 + b_s29;
        // Output point 11: X(10)
        out[out_strides[10]] = b_s31 - b_s28;
        // Output point 23: X(22)
        out[out_strides[22]] = -(b_s31 + b_s28);

        b_t16 = (CRTM_16_5 * b_s25) - (CRTM_16_4 * b_s26);
        b_t17 = (CRTM_16_5 * b_s24) - (CRTM_16_4 * b_t20);

        b_s33 = b_t16 - b_t17;
        b_s34 = b_t17 + b_t16;

        // Output point 22: X(21)
        out[out_strides[21]] = b_s23 - b_s33;
        // Output point 10: X(9)
        out[out_strides[9]] = b_s23 + b_s33;

        // Output point 27: X(26)
        out[out_strides[26]] = b_s34 - b_s32;
        // Output point 7: X(6)
        out[out_strides[6]] = b_s34 + b_s32;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft16c_fp64_bwd(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                        FFTZ_VOID *out_real,
                                        FFTZ_VOID *out_imag, FFTZ_INTP n,
                                        aoclfftz_strides_t *strides,
                                        FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_16_1 =
        1.847759065022573512256366378793576573644833252;
    const FFTZ_DOUBLE CRTM_16_2 =
        0.765366864730179543456919968060797733522689125;
    const FFTZ_DOUBLE CRTM_16_3 =
        1.414213562373095048801688724209698078569671875;
    const FFTZ_DOUBLE CRTM_16_4 =
        2.000000000000000000000000000000000000000000000;
    const FFTZ_DOUBLE CRTM_16_5 =
        0.707106781186547524400844362104849039284835938;
    const FFTZ_DOUBLE CRTM_16_6 =
        1.961570560806460898252364472268478073947867462;
    const FFTZ_DOUBLE CRTM_16_7 =
        0.390180644032256535696569736954044481855383236;
    const FFTZ_DOUBLE CRTM_16_8 =
        1.111140466039204449485661627897065748749874382;
    const FFTZ_DOUBLE CRTM_16_9 =
        1.662939224605090474157576755235811513477121624;

    FFTZ_DOUBLE *in = (FFTZ_DOUBLE *)in_real;
    FFTZ_DOUBLE *out = (FFTZ_DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        FFTZ_DOUBLE a_in0, a_in1, a_in2, a_in3, a_in4, a_in5, a_in6, a_in7,
            a_in8, a_in9, a_in10, a_in11, a_in12, a_in13, a_in14, a_in15;
        FFTZ_DOUBLE a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_s9,
               a_s10, a_s11, a_s12, a_s13, a_s14, a_s15, a_s16, a_s17, a_s18,
               a_s19, a_s20, a_s21, a_s22, a_s23, a_s24, a_s25, a_s26, a_s27,
               a_s28, a_s29, a_s30, a_s31, a_s32, a_s33, a_s34, a_s35, a_s36,
               a_s37, a_s38;
        FFTZ_DOUBLE a_t0, a_t1, a_t2, a_t3, a_t4, a_t5, a_t6, a_t7, a_t8, a_t9,
               a_t10, a_t11, a_t12, a_t13, a_t14, a_t15, a_t16, a_t17, a_t18,
               a_t19, a_t20;

        // Input point 1: x(0)
        a_in0 = *in;
        // Input point 4: x(3)
        a_in1 = in[in_strides[3]];
        // Input point 5: x(4)
        a_in2 = in[in_strides[4]];
        // Input point 8: x(7)
        a_in3 = in[in_strides[7]];
        // Input point 9: x(8)
        a_in4 = in[in_strides[8]];
        // Input point 12: x(11)
        a_in5 = in[in_strides[11]];
        // Input point 13: x(12)
        a_in6 = in[in_strides[12]];
        // Input point 16: x(15)
        a_in7 = in[in_strides[15]];
        // Input point 17: x(16)
        a_in8 = in[in_strides[16]];
        // Input point 20: x(19)
        a_in9 = in[in_strides[19]];
        // Input point 21: x(20)
        a_in10 = in[in_strides[20]];
        // Input point 24: x(23)
        a_in11 = in[in_strides[23]];
        // Input point 25: x(24)
        a_in12 = in[in_strides[24]];
        // Input point 28: x(27)
        a_in13 = in[in_strides[27]];
        // Input point 29: x(28)
        a_in14 = in[in_strides[28]];
        // Input point 32: x(31)
        a_in15 = in[in_strides[31]];

        a_s0 = a_in0 + a_in15;
        a_s1 = a_in0 - a_in15;
        a_s2 = a_in1 + a_in13;
        a_s3 = a_in1 - a_in13;
        a_s4 = a_in2 + a_in14;
        a_s5 = a_in2 - a_in14;
        a_s6 = a_in3 + a_in11;
        a_s7 = a_in3 - a_in11;
        a_s8 = a_in4 + a_in12;
        a_s9 = a_in4 - a_in12;
        a_s10 = a_in5 + a_in9;
        a_s11 = a_in5 - a_in9;
        a_s12 = a_in6 + a_in10;
        a_s13 = a_in6 - a_in10;

        a_s14 = a_s2 + a_s10;
        a_s15 = a_s2 - a_s10;
        a_s16 = a_s3 + a_s12;
        a_s17 = a_s3 - a_s12;
        a_s18 = a_s4 + a_s11;
        a_s19 = a_s4 - a_s11;
        a_s20 = a_s5 + a_s13;
        a_s21 = a_s5 - a_s13;
        a_s22 = a_s7 + a_s8;
        a_s23 = a_s7 - a_s8;

        a_t12 = CRTM_16_4 * a_in7;
        a_t13 = CRTM_16_4 * a_s14;
        a_t14 = CRTM_16_4 * a_s6;
        a_s24 = a_s0 + a_t12;
        a_s26 = a_t14 + a_s24;
        // Output point 1: X(0)
        *out = a_s26 + a_t13;
        // Output point 17: X(16)
        out[out_strides[16]] = a_s26 - a_t13;

        a_t15 = CRTM_16_4 * a_s21;
        a_s27 = a_s24 - a_t14;
        // Output point 9: X(8)
        out[out_strides[8]] = a_s27 - a_t15;
        // Output point 25: X(24)
        out[out_strides[24]] = a_s27 + a_t15;

        a_t16 = CRTM_16_4 * a_in8;
        a_t17 = CRTM_16_3 * a_s23;
        a_t0 = CRTM_16_1 * a_s17;
        a_t1 = CRTM_16_2 * a_s19;
        a_t3 = a_s1 - a_t16;
        a_s29 = a_t17 + a_t3;
        a_s30 = a_t0 - a_t1;
        // Output point 3: X(2)
        out[out_strides[2]] = a_s29 + a_s30;
        // Output point 19: X(18)
        out[out_strides[18]] = a_s29 - a_s30;

        a_s25 = a_s0 - a_t12;
        a_t2 = a_s1 + a_t16;
        a_t4 = CRTM_16_3 * a_s22;
        a_t5 = CRTM_16_1 * a_s18;
        a_t6 = CRTM_16_2 * a_s16;
        a_s31 = a_t2 - a_t4;
        a_s32 = a_t6 - a_t5;
        // Output point 7: X(6)
        out[out_strides[6]] = a_s31 + a_s32;
        // Output point 23: X(22)
        out[out_strides[22]] = a_s31 - a_s32;

        a_t9 = CRTM_16_1 * a_s16;
        a_t10 = CRTM_16_2 * a_s18;
        a_s35 = a_t2 + a_t4;
        a_s36 = a_t9 + a_t10;
        // Output point 15: X(14)
        out[out_strides[14]] = a_s35 - a_s36;
        // Output point 31: X(30)
        out[out_strides[30]] = a_s35 + a_s36;

        a_t7 = CRTM_16_1 * a_s19;
        a_t8 = CRTM_16_2 * a_s17;
        a_s33 = a_t3 - a_t17;
        a_s34 = a_t7 + a_t8;
        // Output point 11: X(10)
        out[out_strides[10]] = a_s33 - a_s34;
        // Output point 27: X(26)
        out[out_strides[26]] = a_s33 + a_s34;

        a_s28 = a_s15 - a_s20;
        a_t18 = CRTM_16_3 * a_s28;
        a_t19 = CRTM_16_4 * a_s9;
        a_s37 = a_s25 - a_t19;
        // Output point 5: X(4)
        out[out_strides[4]] = a_s37 + a_t18;
        // Output point 21: X(20)
        out[out_strides[20]] = a_s37 - a_t18;

        a_t11 = a_s15 + a_s20;
        a_t20 = CRTM_16_3 * a_t11;
        a_s38 = a_s25 + a_t19;
        // Output point 13: X(12)
        out[out_strides[12]] = a_s38 - a_t20;
        // Output point 29: X(28)
        out[out_strides[28]] = a_s38 + a_t20;

        // Shifted DFT
        FFTZ_DOUBLE b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7,
            b_in8, b_in9, b_in10, b_in11, b_in12, b_in13, b_in14, b_in15;
        FFTZ_DOUBLE b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9, b_s10,
               b_s11, b_s12, b_s13, b_s14, b_s15, b_s16, b_s17, b_s18, b_s19,
               b_s20, b_s21, b_s22, b_s23, b_s24, b_s25, b_s26, b_s27, b_s28,
               b_s29, b_s30, b_s31;
        FFTZ_DOUBLE b_t0, b_t1, b_t2, b_t3, b_t4, b_t5, b_t6, b_t7, b_t8, b_t9,
               b_t10, b_t11, b_t12, b_t13, b_t14, b_t15, b_t16, b_t17, b_t18;

        //  Input point 2: x(1)
        b_in0 = in[in_strides[1]];
        // Input point 3: x(2)
        b_in1 = in[in_strides[2]];
        // Input point 6: x(5)
        b_in2 = in[in_strides[5]];
        // Input point 7: x(6)
        b_in3 = in[in_strides[6]];
        // Input point 10: x(9)
        b_in4 = in[in_strides[9]];
        // Input point 11: x(10)
        b_in5 = in[in_strides[10]];
        // Input point 14: x(13)
        b_in6 = in[in_strides[13]];
        // Input point 15: x(14)
        b_in7 = in[in_strides[14]];
        // Input point 18: x(17)
        b_in8 = in[in_strides[17]];
        // Input point 19: x(18)
        b_in9 = in[in_strides[18]];
        // Input point 22: x(21)
        b_in10 = in[in_strides[21]];
        // Input point 23: x(22)
        b_in11 = in[in_strides[22]];
        // Input point 26: x(25)
        b_in12 = in[in_strides[25]];
        // Input point 27: x(26)
        b_in13 = in[in_strides[26]];
        // Input point 30: x(29)
        b_in14 = in[in_strides[29]];
        // Input point 31: x(30)
        b_in15 = in[in_strides[30]];

        b_s1 = b_in14 + b_in0;
        b_s2 = b_in14 - b_in0;
        b_s3 = b_in15 + b_in1;
        b_s4 = b_in15 - b_in1;
        b_s5 = b_in12 + b_in2;
        b_t0 = b_in12 - b_in2;
        b_s6 = b_in13 + b_in3;
        b_t1 = b_in13 - b_in3;
        b_t2 = b_in10 + b_in4;
        b_s7 = b_in10 - b_in4;
        b_s8 = b_in11 + b_in5;
        b_s9 = b_in11 - b_in5;
        b_t3 = b_in8 + b_in6;
        b_s10 = b_in8 - b_in6;
        b_t4 = b_in9 + b_in7;
        b_s11 = b_in9 - b_in7;

        b_t5 = b_s1 - b_t3;
        b_t6 = b_s9 + b_t1;
        b_s12 = b_t5 + b_t6;
        b_t7 = b_t6 - b_t5;

        b_s13 = b_s4 + b_s11;
        b_t8 = b_s5 - b_t2;
        b_s14 = b_s13 + b_t8;
        b_s15 = b_s13 - b_t8;

        // Output point 6: X(5)
        out[out_strides[5]] = (CRTM_16_1 * b_s12) + (CRTM_16_2 * b_s14);
        // Output point 22: X(21)
        out[out_strides[21]] = (CRTM_16_1 * b_s14) - (CRTM_16_2 * b_s12);
        // Output point 30: X(29)
        out[out_strides[29]] = (CRTM_16_1 * b_t7) + (CRTM_16_2 * b_s15);
        // Output point 14: X(13)
        out[out_strides[13]] = (CRTM_16_1 * b_s15) - (CRTM_16_2 * b_t7);

        b_t9 = b_s1 + b_t3;
        b_s16 = b_s5 + b_t2;
        // Output point 2: X(1)
        out[out_strides[1]] = CRTM_16_4 * (b_t9 + b_s16);

        b_t10 = b_s4 - b_s11;
        b_t11 = b_s9 - b_t1;
        // Output point 18: X(17)
        out[out_strides[17]] = CRTM_16_4 * (b_t10 + b_t11);

        b_s17 = b_t9 - b_s16;
        b_s18 = b_t10 - b_t11;

        // Output point 10: X(9)
        out[out_strides[9]] = CRTM_16_3 * (b_s17 + b_s18);
        // Output point 26: X(25)
        out[out_strides[25]] = CRTM_16_3 * (b_s18 - b_s17);

        b_s19 = b_s2 + b_t4;
        b_t12 = -b_s3 - b_s10;

        b_t13 = b_s10 - b_s3;
        b_s20 = b_t4 - b_s2;

        b_s21 = b_s8 + b_t0;
        b_s22 = b_s7 + b_s6;

        b_t16 = CRTM_16_5 * (b_s21 + b_s22);
        b_t17 = CRTM_16_5 * (b_s22 - b_s21);

        b_s25 = b_t16 + b_s19;
        b_s26 = b_t12 + b_t17;
        b_t15 = b_s19 - b_t16;
        b_s27 = b_t12 - b_t17;

        // Output point 4: X(3)
        out[out_strides[3]] = (CRTM_16_7 * b_s26) - (CRTM_16_6 * b_s25);
        // Output point 12: X(11)
        out[out_strides[11]] = (CRTM_16_9 * b_s27) - (CRTM_16_8 * b_t15);
        // Output point 28: X(27)
        out[out_strides[27]] = (CRTM_16_9 * b_t15) + (CRTM_16_8 * b_s27);
        // Output point 20: X(19)
        out[out_strides[19]] = (CRTM_16_6 * b_s26) + (CRTM_16_7 * b_s25);

        b_s23 = b_t0 - b_s8;
        b_s24 = b_s7 - b_s6;

        b_t18 = CRTM_16_5 * (b_s23 - b_s24);
        b_t14 = CRTM_16_5 * (b_s23 + b_s24);

        b_s28 = b_t13 + b_t18;
        b_s29 = b_s20 + b_t14;
        b_s30 = b_t13 - b_t18;
        b_s31 = b_s20 - b_t14;

        // Output point 16: X(15)
        out[out_strides[15]] = (CRTM_16_6 * b_s28) + (CRTM_16_7 * b_s31);
        // Output point 32: X(31)
        out[out_strides[31]] = (CRTM_16_7 * b_s28) - (CRTM_16_6 * b_s31);
        // Output point 8: X(7)
        out[out_strides[7]] = (CRTM_16_9 * b_s29) + (CRTM_16_8 * b_s30);
        // Output point 24: X(23)
        out[out_strides[23]] = (CRTM_16_9 * b_s30) - (CRTM_16_8 * b_s29);

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hcf_rfft16c(FFTZ_UINT8 precision, FFTZ_UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hcf_rfft16c_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft16c_fp64_fwd;
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
            return r2hcf_rfft16c_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft16c_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
