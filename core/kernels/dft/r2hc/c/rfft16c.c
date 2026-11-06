// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft16c.c
 *
 *  @brief Radix-16 r2hc Real-FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-16 real-to-halfcomplex implementations
 *  using scalar operations for single-precision and double-precision inputs.
 *
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                    {{{0, 12, 58, 32, 0, 0},
                                                      {0, 18, 58, 32, 0, 0}},
                                                     {{0, 12, 58, 32, 0, 0},
                                                      {0, 18, 58, 32, 0, 0}}};

ops_cycles_t get_ops_cnt_r2hc_rfft16c(FFTZ_UINT8 precision,
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

static FFTZ_VOID r2hc_rfft16c_fp32_fwd(FFTZ_VOID *in_real,
                                       FFTZ_VOID *in_complex,
                                       FFTZ_VOID *out_real,
                                       FFTZ_VOID *out_complex, FFTZ_INTP n,
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

    FFTZ_FLOAT *in_r = (FFTZ_FLOAT *)in_real;
    FFTZ_FLOAT *out_r = (FFTZ_FLOAT *)out_real;
    FFTZ_FLOAT *out_cp = (FFTZ_FLOAT *)out_complex;

#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_INTP v_out_dc_nyq_stride = strides->v_out_sym_stride;
    FFTZ_INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        FFTZ_FLOAT in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11,
              in12, in13, in14, in15;
        FFTZ_FLOAT s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13,
            s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26,
            s27, s28, s29, s30, s31, s32, s33, s34, s35, s36, s37, s38, s39,
            s40, s41;
        FFTZ_FLOAT t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11;

        // Input point 1: x(0)
        in0 = *in_r;
        // Input point 2: x(1)
        in1 = in_r[in_strides[1]];
        // Input point 3: x(2)
        in2 = in_r[in_strides[2]];
        // Input point 4: x(3)
        in3 = in_r[in_strides[3]];
        // Input point 5: x(4)
        in4 = in_r[in_strides[4]];
        // Input point 6: x(5)
        in5 = in_r[in_strides[5]];
        // Input point 7: x(6)
        in6 = in_r[in_strides[6]];
        // Input point 8: x(7)
        in7 = in_r[in_strides[7]];
        // Input point 9: x(8)
        in8 = in_r[in_strides[8]];
        // Input point 10: x(9)
        in9 = in_r[in_strides[9]];
        // Input point 11: x(10)
        in10 = in_r[in_strides[10]];
        // Input point 12: x(11)
        in11 = in_r[in_strides[11]];
        // Input point 13: x(12)
        in12 = in_r[in_strides[12]];
        // Input point 14: x(13)
        in13 = in_r[in_strides[13]];
        // Input point 15: x(14)
        in14 = in_r[in_strides[14]];
        // Input point 16: x(15)
        in15 = in_r[in_strides[15]];

        s0 = in0 + in8;
        s1 = in0 - in8;
        s2 = in1 + in15;
        s3 = in1 - in15;
        s4 = in2 + in6;
        s5 = in2 - in6;
        s6 = in3 + in5;
        s7 = in3 - in5;
        s8 = in4 + in12;
        s9 = in4 - in12;
        s10 = in7 + in9;
        s11 = in7 - in9;
        s12 = in10 + in14;
        s13 = in10 - in14;
        s14 = in11 + in13;
        s15 = in11 - in13;

        s16 = s0 + s8;
        s17 = s0 - s8;
        s18 = s2 + s10;
        s19 = s2 - s10;
        s20 = s3 + s11;
        t0 = CRTM_16_1 * s19;
        t1 = CRTM_16_1 * s20;
        t2 = CRTM_16_2 * s19;
        t3 = CRTM_16_2 * s20;
        s21 = s3 - s11;
        s22 = s4 + s12;
        s23 = s4 - s12;
        // Output point 8: X(7)
        out_cp[out_strides[7]] = s16 - s22;

        t4 = CRTM_16_3 * s23;
        s24 = s5 + s13;
        s25 = s5 - s13;
        t5 = CRTM_16_3 * s25;
        s26 = s6 + s14;
        s27 = s6 - s14;
        t6 = CRTM_16_1 * s27;
        t7 = CRTM_16_2 * s27;
        s28 = s7 + s15;
        s29 = s7 - s15;
        t8 = CRTM_16_1 * s29;
        t9 = CRTM_16_2 * s29;
        // Output point 9: X(8)
        out_cp[out_strides[8]] = s28 - s21;

        s34 = t0 + t9;
        s35 = t5 + s1;
        // Output point 2: X(1)
        out_cp[out_strides[1]] = s34 + s35;
        // Output point 14: X(13)
        out_cp[out_strides[13]] = s35 - s34;

        s30 = s18 + s26;
        s31 = s18 - s26;
        t10 = CRTM_16_3 * s31;
        // Output point 4: X(3)
        out_cp[out_strides[3]] = s17 + t10;
        // Output point 12: X(11)
        out_cp[out_strides[11]] = s17 - t10;

        s32 = s16 + s22;
        s33 = s21 + s28;
        // Output point 1: X(0)
        *out_r = s30 + s32;
        // Output point 16: X(15)
        out_r[out_strides[15]] = s32 - s30;

        t11 = CRTM_16_3 * s33;
        // Output point 5: X(4)
        out_cp[out_strides[4]] = -(s24 + t11);
        // Output point 13: X(12)
        out_cp[out_strides[12]] = s24 - t11;

        s36 = t6 + t3;
        s37 = t4 + s9;
        // Output point 3: X(2)
        out_cp[out_strides[2]] = -(s36 + s37);
        // Output point 15: X(14)
        out_cp[out_strides[14]] = s37 - s36;

        s38 = t8 - t2;
        s39 = s1 - t5;
        // Output point 6: X(5)
        out_cp[out_strides[5]] = s39 - s38;
        // Output point 10: X(9)
        out_cp[out_strides[9]] = s38 + s39;

        s40 = t7 - t1;
        s41 = s9 - t4;
        // Output point 7: X(6)
        out_cp[out_strides[6]] = s40 + s41;
        // Output point 11: X(10)
        out_cp[out_strides[10]] = s40 - s41;

        in_r = in_r + v_in_stride;
        out_cp = out_cp + v_out_stride;
        out_r = out_r + v_out_dc_nyq_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft16c_fp32_bwd(FFTZ_VOID *in_real,
                                       FFTZ_VOID *in_complex,
                                       FFTZ_VOID *out_real,
                                       FFTZ_VOID *out_complex, FFTZ_INTP n,
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

    FFTZ_FLOAT *in_r = (FFTZ_FLOAT *)in_real;
    FFTZ_FLOAT *in_cp = (FFTZ_FLOAT *)in_complex;
    FFTZ_FLOAT *out_r = (FFTZ_FLOAT *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_in_dc_nyq_stride = strides->v_in_sym_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        FFTZ_FLOAT in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11,
              in12, in13, in14, in15;
        FFTZ_FLOAT s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13,
            s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26,
            s27, s28, s29, s30, s31, s32, s33, s34, s35, s36, s37, s38, s39,
            s40, s41;
        FFTZ_FLOAT t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13,
            t14, t15, t16, t17;

        // Input point 1: x(0)
        in0 = *in_r;
        // Input point 2: x(1)
        in1 = in_cp[in_strides[1]];
        // Input point 3: x(2)
        in2 = in_cp[in_strides[2]];
        // Input point 4: x(3)
        in3 = in_cp[in_strides[3]];
        // Input point 5: x(4)
        in4 = in_cp[in_strides[4]];
        // Input point 6: x(5)
        in5 = in_cp[in_strides[5]];
        // Input point 7: x(6)
        in6 = in_cp[in_strides[6]];
        // Input point 8: x(7)
        in7 = in_cp[in_strides[7]];
        // Input point 9: x(8)
        in8 = in_cp[in_strides[8]];
        // Input point 10: x(9)
        in9 = in_cp[in_strides[9]];
        // Input point 11: x(10)
        in10 = in_cp[in_strides[10]];
        // Input point 12: x(11)
        in11 = in_cp[in_strides[11]];
        // Input point 13: x(12)
        in12 = in_cp[in_strides[12]];
        // Input point 14: x(13)
        in13 = in_cp[in_strides[13]];
        // Input point 15: x(14)
        in14 = in_cp[in_strides[14]];
        // Input point 16: x(15)
        in15 = in_r[in_strides[15]];

        s0 = in0 + in15;
        s1 = in0 - in15;
        s2 = in1 + in13;
        s3 = in1 - in13;
        s4 = in2 + in14;
        s5 = in2 - in14;
        s6 = in3 + in11;
        s7 = in3 - in11;
        s8 = in4 + in12;
        s9 = in4 - in12;
        s10 = in5 + in9;
        s11 = in5 - in9;
        s12 = in6 + in10;
        s13 = in6 - in10;

        s14 = s2 + s10;
        s15 = s2 - s10;
        s16 = s3 + s12;
        s17 = s3 - s12;
        s18 = s4 + s11;
        s19 = s4 - s11;
        s20 = s5 + s13;
        s21 = s5 - s13;
        s22 = s7 + s8;
        s23 = s7 - s8;

        t0 = CRTM_16_4 * in7;
        t1 = CRTM_16_4 * s14;
        t2 = CRTM_16_4 * s6;
        s24 = s0 + t0;
        s26 = t2 + s24;
        // Output point 1: X(0)
        *out_r = s26 + t1;
        // Output point 9: X(8)
        out_r[out_strides[8]] = s26 - t1;

        t3 = CRTM_16_4 * s21;
        s27 = s24 - t2;
        // Output point 5: X(4)
        out_r[out_strides[4]] = s27 - t3;
        // Output point 3: X(2)
        out_r[out_strides[12]] = s27 + t3;

        t4 = CRTM_16_4 * in8;
        t5 = CRTM_16_3 * s23;
        t6 = CRTM_16_1 * s17;
        t7 = CRTM_16_2 * s19;
        s29 = s1 - t4;
        s32 = t5 + s29;
        s33 = t6 - t7;
        // Output point 2: X(1)
        out_r[out_strides[1]] = s32 + s33;
        // Output point 10: X(9)
        out_r[out_strides[9]] = s32 - s33;

        s25 = s0 - t0;
        s28 = s1 + t4;
        t8 = CRTM_16_3 * s22;
        t9 = CRTM_16_1 * s18;
        t10 = CRTM_16_2 * s16;
        s34 = s28 - t8;
        s35 = t10 - t9;
        // Output point 4: X(3)
        out_r[out_strides[3]] = s34 + s35;
        // Output point 12: X(11)
        out_r[out_strides[11]] = s34 - s35;

        t11 = CRTM_16_1 * s16;
        t12 = CRTM_16_2 * s18;
        s38 = s28 + t8;
        s39 = t11 + t12;
        // Output point 8: X(7)
        out_r[out_strides[7]] = s38 - s39;
        // Output point 16: X(15)
        out_r[out_strides[15]] = s38 + s39;

        t13 = CRTM_16_1 * s19;
        t14 = CRTM_16_2 * s17;
        s36 = s29 - t5;
        s37 = t13 + t14;
        // Output point 6: X(5)
        out_r[out_strides[5]] = s36 - s37;
        // Output point 14: X(13)
        out_r[out_strides[13]] = s36 + s37;

        s31 = s15 - s20;
        t15 = CRTM_16_3 * s31;
        t16 = CRTM_16_4 * s9;
        s40 = s25 - t16;
        // Output point 3: X(2)
        out_r[out_strides[2]] = s40 + t15;
        // Output point 11: X(10)
        out_r[out_strides[10]] = s40 - t15;

        s30 = s15 + s20;
        t17 = CRTM_16_3 * s30;
        s41 = s25 + t16;
        // Output point 7: X(6)
        out_r[out_strides[6]] = s41 - t17;
        // Output point 15: X(14)
        out_r[out_strides[14]] = s41 + t17;

        in_cp = in_cp + v_in_stride;
        in_r = in_r + v_in_dc_nyq_stride;
        out_r = out_r + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft16c_fp64_fwd(FFTZ_VOID *in_real,
                                       FFTZ_VOID *in_complex,
                                       FFTZ_VOID *out_real,
                                       FFTZ_VOID *out_complex, FFTZ_INTP n,
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

    FFTZ_DOUBLE *in_r = (FFTZ_DOUBLE *)in_real;
    FFTZ_DOUBLE *out_r = (FFTZ_DOUBLE *)out_real;
    FFTZ_DOUBLE *out_cp = (FFTZ_DOUBLE *)out_complex;

#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_INTP v_out_dc_nyq_stride = strides->v_out_sym_stride;
    FFTZ_INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        FFTZ_DOUBLE in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10,
            in11, in12, in13, in14, in15;
        FFTZ_DOUBLE s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13,
            s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26,
            s27, s28, s29, s30, s31, s32, s33, s34, s35, s36, s37, s38, s39,
            s40, s41;
        FFTZ_DOUBLE t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11;

        // Input point 1: x(0)
        in0 = *in_r;
        // Input point 2: x(1)
        in1 = in_r[in_strides[1]];
        // Input point 3: x(2)
        in2 = in_r[in_strides[2]];
        // Input point 4: x(3)
        in3 = in_r[in_strides[3]];
        // Input point 5: x(4)
        in4 = in_r[in_strides[4]];
        // Input point 6: x(5)
        in5 = in_r[in_strides[5]];
        // Input point 7: x(6)
        in6 = in_r[in_strides[6]];
        // Input point 8: x(7)
        in7 = in_r[in_strides[7]];
        // Input point 9: x(8)
        in8 = in_r[in_strides[8]];
        // Input point 10: x(9)
        in9 = in_r[in_strides[9]];
        // Input point 11: x(10)
        in10 = in_r[in_strides[10]];
        // Input point 12: x(11)
        in11 = in_r[in_strides[11]];
        // Input point 13: x(12)
        in12 = in_r[in_strides[12]];
        // Input point 14: x(13)
        in13 = in_r[in_strides[13]];
        // Input point 15: x(14)
        in14 = in_r[in_strides[14]];
        // Input point 16: x(15)
        in15 = in_r[in_strides[15]];

        s0 = in0 + in8;
        s1 = in0 - in8;
        s2 = in1 + in15;
        s3 = in1 - in15;
        s4 = in2 + in6;
        s5 = in2 - in6;
        s6 = in3 + in5;
        s7 = in3 - in5;
        s8 = in4 + in12;
        s9 = in4 - in12;
        s10 = in7 + in9;
        s11 = in7 - in9;
        s12 = in10 + in14;
        s13 = in10 - in14;
        s14 = in11 + in13;
        s15 = in11 - in13;

        s16 = s0 + s8;
        s17 = s0 - s8;
        s18 = s2 + s10;
        s19 = s2 - s10;
        s20 = s3 + s11;
        t0 = CRTM_16_1 * s19;
        t1 = CRTM_16_1 * s20;
        t2 = CRTM_16_2 * s19;
        t3 = CRTM_16_2 * s20;
        s21 = s3 - s11;
        s22 = s4 + s12;
        s23 = s4 - s12;
        // Output point 8: X(7)
        out_cp[out_strides[7]] = s16 - s22;

        t4 = CRTM_16_3 * s23;
        s24 = s5 + s13;
        s25 = s5 - s13;
        t5 = CRTM_16_3 * s25;
        s26 = s6 + s14;
        s27 = s6 - s14;
        t6 = CRTM_16_1 * s27;
        t7 = CRTM_16_2 * s27;
        s28 = s7 + s15;
        s29 = s7 - s15;
        t8 = CRTM_16_1 * s29;
        t9 = CRTM_16_2 * s29;
        // Output point 9: X(8)
        out_cp[out_strides[8]] = s28 - s21;

        s34 = t0 + t9;
        s35 = t5 + s1;
        // Output point 2: X(1)
        out_cp[out_strides[1]] = s34 + s35;
        // Output point 14: X(13)
        out_cp[out_strides[13]] = s35 - s34;

        s30 = s18 + s26;
        s31 = s18 - s26;
        t10 = CRTM_16_3 * s31;
        // Output point 4: X(3)
        out_cp[out_strides[3]] = s17 + t10;
        // Output point 12: X(11)
        out_cp[out_strides[11]] = s17 - t10;

        s32 = s16 + s22;
        s33 = s21 + s28;
        // Output point 1: X(0)
        *out_r = s30 + s32;
        // Output point 16: X(15)
        out_r[out_strides[15]] = s32 - s30;

        t11 = CRTM_16_3 * s33;
        // Output point 5: X(4)
        out_cp[out_strides[4]] = -(s24 + t11);
        // Output point 13: X(12)
        out_cp[out_strides[12]] = s24 - t11;

        s36 = t6 + t3;
        s37 = t4 + s9;
        // Output point 3: X(2)
        out_cp[out_strides[2]] = -(s36 + s37);
        // Output point 15: X(14)
        out_cp[out_strides[14]] = s37 - s36;

        s38 = t8 - t2;
        s39 = s1 - t5;
        // Output point 6: X(5)
        out_cp[out_strides[5]] = s39 - s38;
        // Output point 10: X(9)
        out_cp[out_strides[9]] = s38 + s39;

        s40 = t7 - t1;
        s41 = s9 - t4;
        // Output point 7: X(6)
        out_cp[out_strides[6]] = s40 + s41;
        // Output point 11: X(10)
        out_cp[out_strides[10]] = s40 - s41;

        in_r = in_r + v_in_stride;
        out_cp = out_cp + v_out_stride;
        out_r = out_r + v_out_dc_nyq_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft16c_fp64_bwd(FFTZ_VOID *in_real,
                                       FFTZ_VOID *in_complex,
                                       FFTZ_VOID *out_real,
                                       FFTZ_VOID *out_complex, FFTZ_INTP n,
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

    FFTZ_DOUBLE *in_r = (FFTZ_DOUBLE *)in_real;
    FFTZ_DOUBLE *in_cp = (FFTZ_DOUBLE *)in_complex;
    FFTZ_DOUBLE *out_r = (FFTZ_DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_in_dc_nyq_stride = strides->v_in_sym_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        FFTZ_DOUBLE in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10,
            in11, in12, in13, in14, in15;
        FFTZ_DOUBLE s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13,
            s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26,
            s27, s28, s29, s30, s31, s32, s33, s34, s35, s36, s37, s38, s39,
            s40, s41;
        FFTZ_DOUBLE t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13,
            t14, t15, t16, t17;

        // Input point 1: x(0)
        in0 = *in_r;
        // Input point 2: x(1)
        in1 = in_cp[in_strides[1]];
        // Input point 3: x(2)
        in2 = in_cp[in_strides[2]];
        // Input point 4: x(3)
        in3 = in_cp[in_strides[3]];
        // Input point 5: x(4)
        in4 = in_cp[in_strides[4]];
        // Input point 6: x(5)
        in5 = in_cp[in_strides[5]];
        // Input point 7: x(6)
        in6 = in_cp[in_strides[6]];
        // Input point 8: x(7)
        in7 = in_cp[in_strides[7]];
        // Input point 9: x(8)
        in8 = in_cp[in_strides[8]];
        // Input point 10: x(9)
        in9 = in_cp[in_strides[9]];
        // Input point 11: x(10)
        in10 = in_cp[in_strides[10]];
        // Input point 12: x(11)
        in11 = in_cp[in_strides[11]];
        // Input point 13: x(12)
        in12 = in_cp[in_strides[12]];
        // Input point 14: x(13)
        in13 = in_cp[in_strides[13]];
        // Input point 15: x(14)
        in14 = in_cp[in_strides[14]];
        // Input point 16: x(15)
        in15 = in_r[in_strides[15]];

        s0 = in0 + in15;
        s1 = in0 - in15;
        s2 = in1 + in13;
        s3 = in1 - in13;
        s4 = in2 + in14;
        s5 = in2 - in14;
        s6 = in3 + in11;
        s7 = in3 - in11;
        s8 = in4 + in12;
        s9 = in4 - in12;
        s10 = in5 + in9;
        s11 = in5 - in9;
        s12 = in6 + in10;
        s13 = in6 - in10;

        s14 = s2 + s10;
        s15 = s2 - s10;
        s16 = s3 + s12;
        s17 = s3 - s12;
        s18 = s4 + s11;
        s19 = s4 - s11;
        s20 = s5 + s13;
        s21 = s5 - s13;
        s22 = s7 + s8;
        s23 = s7 - s8;

        t0 = CRTM_16_4 * in7;
        t1 = CRTM_16_4 * s14;
        t2 = CRTM_16_4 * s6;
        s24 = s0 + t0;
        s26 = t2 + s24;
        // Output point 1: X(0)
        *out_r = s26 + t1;
        // Output point 9: X(8)
        out_r[out_strides[8]] = s26 - t1;

        t3 = CRTM_16_4 * s21;
        s27 = s24 - t2;
        // Output point 5: X(4)
        out_r[out_strides[4]] = s27 - t3;
        // Output point 3: X(2)
        out_r[out_strides[12]] = s27 + t3;

        t4 = CRTM_16_4 * in8;
        t5 = CRTM_16_3 * s23;
        t6 = CRTM_16_1 * s17;
        t7 = CRTM_16_2 * s19;
        s29 = s1 - t4;
        s32 = t5 + s29;
        s33 = t6 - t7;
        // Output point 2: X(1)
        out_r[out_strides[1]] = s32 + s33;
        // Output point 10: X(9)
        out_r[out_strides[9]] = s32 - s33;

        s25 = s0 - t0;
        s28 = s1 + t4;
        t8 = CRTM_16_3 * s22;
        t9 = CRTM_16_1 * s18;
        t10 = CRTM_16_2 * s16;
        s34 = s28 - t8;
        s35 = t10 - t9;
        // Output point 4: X(3)
        out_r[out_strides[3]] = s34 + s35;
        // Output point 12: X(11)
        out_r[out_strides[11]] = s34 - s35;

        t11 = CRTM_16_1 * s16;
        t12 = CRTM_16_2 * s18;
        s38 = s28 + t8;
        s39 = t11 + t12;
        // Output point 8: X(7)
        out_r[out_strides[7]] = s38 - s39;
        // Output point 16: X(15)
        out_r[out_strides[15]] = s38 + s39;

        t13 = CRTM_16_1 * s19;
        t14 = CRTM_16_2 * s17;
        s36 = s29 - t5;
        s37 = t13 + t14;
        // Output point 6: X(5)
        out_r[out_strides[5]] = s36 - s37;
        // Output point 14: X(13)
        out_r[out_strides[13]] = s36 + s37;

        s31 = s15 - s20;
        t15 = CRTM_16_3 * s31;
        t16 = CRTM_16_4 * s9;
        s40 = s25 - t16;
        // Output point 3: X(2)
        out_r[out_strides[2]] = s40 + t15;
        // Output point 11: X(10)
        out_r[out_strides[10]] = s40 - t15;

        s30 = s15 + s20;
        t17 = CRTM_16_3 * s30;
        s41 = s25 + t16;
        // Output point 7: X(6)
        out_r[out_strides[6]] = s41 - t17;
        // Output point 15: X(14)
        out_r[out_strides[14]] = s41 + t17;

        in_cp = in_cp + v_in_stride;
        in_r = in_r + v_in_dc_nyq_stride;
        out_r = out_r + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hc_rfft16c(FFTZ_UINT8 precision, FFTZ_UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hc_rfft16c_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft16c_fp64_fwd;
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
            return r2hc_rfft16c_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft16c_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
