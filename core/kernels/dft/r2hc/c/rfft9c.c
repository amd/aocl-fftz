// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft9c.c
 *
 *  @brief Radix-9 r2hc Real-FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-9 real-to-halfcomplex implementations
 *  using scalar operations for single-precision and double-precision inputs.
 *
 *  @author Amrin Fathima
 */
#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] = {
                                                    {{0, 22, 37, 18, 0, 0},
                                                     {0, 18, 32, 18, 0, 0}},
                                                    {{0, 22, 37, 18, 0, 0},
                                                     {0, 18, 32, 18, 0, 0}}};

ops_cycles_t get_ops_cnt_r2hc_rfft9c(UINT8 precision, UINT8 direction)
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

static VOID r2hc_rfft9c_fp32_fwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                 VOID *out_imag, INTP n,
                                 aoclfftz_strides_t *strides, VOID *twd,
                                 UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_9_0 = 0.766044443118978035202392650555416673935832457f;
    const FLOAT CRTM_9_1 = 0.642787609686539326322643409907263432907559884f;
    const FLOAT CRTM_9_2 = 0.173648177666930348851716626769314796000375677f;
    const FLOAT CRTM_9_3 = 0.984807753012208059366743024589523013670643252f;
    const FLOAT CRTM_9_4 = 0.500000000000000000000000000000000000000000000f;
    const FLOAT CRTM_9_5 = 0.866025403784438646763723170752936183471402627f;
    const FLOAT CRTM_9_6 = 0.939692620785908384054109277324975766871890789f;
    const FLOAT CRTM_9_7 = 0.342020143325668733044099614682259580763083320f;

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
        FLOAT in0, in1, in2, in3, in4, in5, in6, in7, in8;
        FLOAT s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14,
              s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27,
              s28;
        FLOAT m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14,
              m15, m16, m17, m18, m19, m20, m21;

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

        s0 = in1 + in8;
        s1 = in1 - in8;
        s2 = in2 + in7;
        s3 = in2 - in7;
        s4 = in3 + in6;
        s5 = in3 - in6;
        s6 = in4 + in5;
        s7 = in4 - in5;

        s8 = s0 + s6;
        s9 = s8 + s2;
        s10 = s1 - s3;
        s11 = s10 + s7;
        s12 = s9 + s4;
        s13 = s12 + in0;

        m0 = CRTM_9_4 * s4;
        m1 = CRTM_9_5 * s5;
        s14 = in0 - m0;
        // Output point 1: X(0)
        *out = s13;

        m2 = CRTM_9_0 * s0;
        m3 = CRTM_9_2 * s2;
        m4 = CRTM_9_6 * s6;
        s15 = s14 + m2;
        s16 = s15 + m3;
        // Output point 2: X(1)
        out[out_strides[1]] = s16 - m4;

        m5 = CRTM_9_1 * s1;
        m6 = CRTM_9_3 * s3;
        m7 = CRTM_9_7 * s7;
        s17 = m5 + m6;
        s18 = s17 + m1;
        s19 = -(s18 + m7);
        // Output point 3: X(2)
        out[out_strides[2]] = s19;

        m8 = CRTM_9_0 * s6;
        m9 = CRTM_9_2 * s0;
        m10 = CRTM_9_6 * s2;
        s20 = s14 + m8;
        s21 = s20 + m9;
        // Output point 4: X(3)
        out[out_strides[3]] = s21 - m10;

        m11 = CRTM_9_1 * s7;
        m12 = CRTM_9_3 * s1;
        m13 = CRTM_9_7 * s3;
        s22 = m11 - m12;
        s23 = s22 + m1;
        // Output point 5: X(4)
        out[out_strides[4]] = s23 - m13;

        m14 = CRTM_9_4 * s9;
        s24 = in0 + s4;
        // Output point 6: X(5)
        out[out_strides[5]] = s24 - m14;

        m15 = -CRTM_9_5 * s11;
        // Output point 7: X(6)
        out[out_strides[6]] = m15;

        m16 = CRTM_9_0 * s2;
        m17 = CRTM_9_2 * s6;
        m18 = CRTM_9_6 * s0;
        s25 = s14 + m16;
        s26 = s25 + m17;
        // Output point 8: X(7)
        out[out_strides[7]] = s26 - m18;

        m19 = CRTM_9_1 * s3;
        m20 = CRTM_9_3 * s7;
        m21 = CRTM_9_7 * s1;
        s27 = m19 + m20;
        s28 = s27 - m1;
        // Output point 9: X(8)
        out[out_strides[8]] = s28 - m21;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hc_rfft9c_fp32_bwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                 VOID *out_imag, INTP n,
                                 aoclfftz_strides_t *strides, VOID *twd,
                                 UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_9_0 = 0.766044443118978035202392650555416673935832457f;
    const FLOAT CRTM_9_1 = 0.642787609686539326322643409907263432907559884f;
    const FLOAT CRTM_9_2 = 0.173648177666930348851716626769314796000375677f;
    const FLOAT CRTM_9_3 = 0.984807753012208059366743024589523013670643252f;
    const FLOAT CRTM_9_4 = 0.500000000000000000000000000000000000000000000f;
    const FLOAT CRTM_9_5 = 0.866025403784438646763723170752936183471402627f;
    const FLOAT CRTM_9_6 = 2.000000000000000000000000000000000000000000000f;
    // Below CRTMs are the product or sum of the above CRTMs, Precomputed
    // to save multiplications and additions on the fly.
    // CRTM_9_7 = CRTM_9_6 * CRTM_9_5
    const FLOAT CRTM_9_7 = 1.732050807568877293527446341505872366942805254f;
    // CRTM_9_8 = CRTM_9_6 * CRTM_9_5 * CRTM_9_3
    const FLOAT CRTM_9_8 = 1.705737063904886419256501927880148143872040592f;
    // CRTM_9_9 = CRTM_9_7 * CRTM_9_2
    const FLOAT CRTM_9_9 = 0.300767466360870593278543795225003852144476516f;
    // CRTM_9_10 = CRTM_9_8 - CRTM_9_0 + CRTM_9_2
    const FLOAT CRTM_9_10 = 1.113340798452838732905825904094046265936583812f;
    // CRTM_9_11 = CRTM_9_3 + CRTM_9_6 * CRTM_9_2 * CRTM_9_3
    const FLOAT CRTM_9_11 = 1.326827896337876792410842639271782594433726619f;

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
        FLOAT in0, in1, in2, in3, in4, in5, in6, in7, in8;
        FLOAT s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14,
              s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27,
              s28, s29, s30, s31;
        FLOAT m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14,
              m15, m16, m17;

        // Input point 1: X(0)
        in0 = *in;
        // Input point 2: X(1)
        in1 = in[in_strides[1]];
        // Input point 3: X(2)
        in2 = in[in_strides[2]];
        // Input point 4: X(3)
        in3 = in[in_strides[3]];
        // Input point 5: X(4)
        in4 = in[in_strides[4]];
        // Input point 6: X(5)
        in5 = in[in_strides[5]];
        // Input point 7: X(6)
        in6 = in[in_strides[6]];
        // Input point 8: X(7)
        in7 = in[in_strides[7]];
        // Input point 9: X(8)
        in8 = in[in_strides[8]];

        m0 = CRTM_9_7 * in6;
        s0 = in0 - in5;
        m1 = CRTM_9_6 * in5;
        s1 = in0 + m1;
        s2 = s0 - m0;
        s3 = s0 + m0;
        s4 = in7 + in3;
        s5 = in7 - in3;
        m2 = CRTM_9_5 * s5;
        s6 = in8 + in4;
        m3 = CRTM_9_5 * s6;
        s7 = in4 - in8;
        s8 = in1 + s4;
        m4 = CRTM_9_4 * s7;
        s9 = in2 + m4;
        s10 = m2 + s9;
        s11 = s9 - m2;
        m5 = CRTM_9_4 * s4;
        s12 = in1 - m5;
        s13 = s12 - m3;
        s14 = s12 + m3;
        m6 = CRTM_9_6 * s8;
        s15 = s1 + m6;
        // Output point 1: x(0)
        *out = s15;

        s16 = s1 - s8;
        s17 = in2 - s7;
        m7 = CRTM_9_7 * s17;
        s18 = s16 - m7;
        s19 = s16 + m7;
        // Output point 4: x(3)
        out[out_strides[3]] = s18;
        // Output point 7: x(6)
        out[out_strides[6]] = s19;

        m8 = CRTM_9_0 * s13;
        m9 = CRTM_9_1 * s10;
        s20 = m8 - m9;
        m10 = CRTM_9_11 * s10;
        m11 = CRTM_9_10 * s13;
        s21 = m11 + m10;
        s22 = s2 - s20;
        m12 = CRTM_9_6 * s20;
        s23 = s2 + m12;
        // Output point 2: x(1)
        out[out_strides[1]] = s23;

        s24 = s22 + s21;
        s25 = s22 - s21;
        // Output point 8: x(7)
        out[out_strides[7]] = s24;
        // Output point 5: x(4)
        out[out_strides[4]] = s25;

        m13 = CRTM_9_8 * s14;
        m14 = CRTM_9_9 * s11;
        s26 = m13 + m14;
        m15 = CRTM_9_2 * s14;
        m16 = CRTM_9_3 * s11;
        s27 = m15 - m16;
        s28 = s3 - s27;
        m17 = CRTM_9_6 * s27;
        s29 = s3 + m17;
        // Output point 3: x(2)
        out[out_strides[2]] = s29;

        s30 = s28 + s26;
        s31 = s28 - s26;
        // Output point 9: x(8)
        out[out_strides[8]] = s30;
        // Output point 6: x(5)
        out[out_strides[5]] = s31;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hc_rfft9c_fp64_fwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                 VOID *out_imag, INTP n,
                                 aoclfftz_strides_t *strides, VOID *twd,
                                 UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_9_0 = 0.766044443118978035202392650555416673935832457;
    const DOUBLE CRTM_9_1 = 0.642787609686539326322643409907263432907559884;
    const DOUBLE CRTM_9_2 = 0.173648177666930348851716626769314796000375677;
    const DOUBLE CRTM_9_3 = 0.984807753012208059366743024589523013670643252;
    const DOUBLE CRTM_9_4 = 0.500000000000000000000000000000000000000000000;
    const DOUBLE CRTM_9_5 = 0.866025403784438646763723170752936183471402627;
    const DOUBLE CRTM_9_6 = 0.939692620785908384054109277324975766871890789;
    const DOUBLE CRTM_9_7 = 0.342020143325668733044099614682259580763083320;

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
        DOUBLE in0, in1, in2, in3, in4, in5, in6, in7, in8;
        DOUBLE s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14,
               s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27,
               s28;
        DOUBLE m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14,
               m15, m16, m17, m18, m19, m20, m21;

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

        s0 = in1 + in8;
        s1 = in1 - in8;
        s2 = in2 + in7;
        s3 = in2 - in7;
        s4 = in3 + in6;
        s5 = in3 - in6;
        s6 = in4 + in5;
        s7 = in4 - in5;

        s8 = s0 + s6;
        s9 = s8 + s2;
        s10 = s1 - s3;
        s11 = s10 + s7;
        s12 = s9 + s4;
        s13 = s12 + in0;

        m0 = CRTM_9_4 * s4;
        m1 = CRTM_9_5 * s5;
        s14 = in0 - m0;

        // Output point 1: X(0)
        *out = s13;

        m2 = CRTM_9_0 * s0;
        m3 = CRTM_9_2 * s2;
        m4 = CRTM_9_6 * s6;
        s15 = s14 + m2;
        s16 = s15 + m3;
        // Output point 2: X(1)
        out[out_strides[1]] = s16 - m4;

        m5 = CRTM_9_1 * s1;
        m6 = CRTM_9_3 * s3;
        m7 = CRTM_9_7 * s7;
        s17 = m5 + m6;
        s18 = s17 + m1;
        s19 = -(s18 + m7);
        // Output point 3: X(2)
        out[out_strides[2]] = s19;

        m8 = CRTM_9_0 * s6;
        m9 = CRTM_9_2 * s0;
        m10 = CRTM_9_6 * s2;
        s20 = s14 + m8;
        s21 = s20 + m9;
        // Output point 4: X(3)
        out[out_strides[3]] = s21 - m10;

        m11 = CRTM_9_1 * s7;
        m12 = CRTM_9_3 * s1;
        m13 = CRTM_9_7 * s3;
        s22 = m11 - m12;
        s23 = s22 + m1;
        // Output point 5: X(4)
        out[out_strides[4]] = s23 - m13;

        m14 = CRTM_9_4 * s9;
        s24 = in0 + s4;
        // Output point 6: X(5)
        out[out_strides[5]] = s24 - m14;

        m15 = -CRTM_9_5 * s11;
        // Output point 7: X(6)
        out[out_strides[6]] = m15;

        m16 = CRTM_9_0 * s2;
        m17 = CRTM_9_2 * s6;
        m18 = CRTM_9_6 * s0;
        s25 = s14 + m16;
        s26 = s25 + m17;
        // Output point 8: X(7)
        out[out_strides[7]] = s26 - m18;

        m19 = CRTM_9_1 * s3;
        m20 = CRTM_9_3 * s7;
        m21 = CRTM_9_7 * s1;
        s27 = m19 + m20;
        s28 = s27 - m1;
        // Output point 9: X(8)
        out[out_strides[8]] = s28 - m21;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hc_rfft9c_fp64_bwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                 VOID *out_imag, INTP n,
                                 aoclfftz_strides_t *strides, VOID *twd,
                                 UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_9_0 = 0.766044443118978035202392650555416673935832457;
    const DOUBLE CRTM_9_1 = 0.642787609686539326322643409907263432907559884;
    const DOUBLE CRTM_9_2 = 0.173648177666930348851716626769314796000375677;
    const DOUBLE CRTM_9_3 = 0.984807753012208059366743024589523013670643252;
    const DOUBLE CRTM_9_4 = 0.500000000000000000000000000000000000000000000;
    const DOUBLE CRTM_9_5 = 0.866025403784438646763723170752936183471402627;
    const DOUBLE CRTM_9_6 = 2.000000000000000000000000000000000000000000000;
    // Below CRTMs are the product or sum of the above CRTMs, Precomputed
    // to save multiplications and additions on the fly.
    // CRTM_9_7 = CRTM_9_6 * CRTM_9_5
    const DOUBLE CRTM_9_7 = 1.732050807568877293527446341505872366942805254;
    // CRTM_9_8 = CRTM_9_6 * CRTM_9_5 * CRTM_9_3
    const DOUBLE CRTM_9_8 = 1.705737063904886419256501927880148143872040592;
    // CRTM_9_9 = CRTM_9_7 * CRTM_9_2
    const DOUBLE CRTM_9_9 = 0.300767466360870593278543795225003852144476516;
    // CRTM_9_10 = CRTM_9_8 - CRTM_9_0 + CRTM_9_2
    const DOUBLE CRTM_9_10 = 1.113340798452838732905825904094046265936583812;
    // CRTM_9_11 = CRTM_9_3 + CRTM_9_6 * CRTM_9_2 * CRTM_9_3
    const DOUBLE CRTM_9_11 = 1.326827896337876792410842639271782594433726619;

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
        DOUBLE in0, in1, in2, in3, in4, in5, in6, in7, in8;
        DOUBLE s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14,
               s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27,
               s28, s29, s30, s31;
        DOUBLE m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14,
               m15, m16, m17;

        // Input point 1: X(0)
        in0 = *in;
        // Input point 2: X(1)
        in1 = in[in_strides[1]];
        // Input point 3: X(2)
        in2 = in[in_strides[2]];
        // Input point 4: X(3)
        in3 = in[in_strides[3]];
        // Input point 5: X(4)
        in4 = in[in_strides[4]];
        // Input point 6: X(5)
        in5 = in[in_strides[5]];
        // Input point 7: X(6)
        in6 = in[in_strides[6]];
        // Input point 8: X(7)
        in7 = in[in_strides[7]];
        // Input point 9: X(8)
        in8 = in[in_strides[8]];

        m0 = CRTM_9_7 * in6;
        s0 = in0 - in5;
        m1 = CRTM_9_6 * in5;
        s1 = in0 + m1;
        s2 = s0 - m0;
        s3 = s0 + m0;
        s4 = in7 + in3;
        s5 = in7 - in3;
        m2 = CRTM_9_5 * s5;
        s6 = in8 + in4;
        m3 = CRTM_9_5 * s6;
        s7 = in4 - in8;
        s8 = in1 + s4;
        m4 = CRTM_9_4 * s7;
        s9 = in2 + m4;
        s10 = m2 + s9;
        s11 = s9 - m2;
        m5 = CRTM_9_4 * s4;
        s12 = in1 - m5;
        s13 = s12 - m3;
        s14 = s12 + m3;
        m6 = CRTM_9_6 * s8;
        s15 = s1 + m6;
        // Output point 1: x(0)
        *out = s15;

        s16 = s1 - s8;
        s17 = in2 - s7;
        m7 = CRTM_9_7 * s17;
        s18 = s16 - m7;
        s19 = s16 + m7;
        // Output point 4: x(3)
        out[out_strides[3]] = s18;
        // Output point 7: x(6)
        out[out_strides[6]] = s19;

        m8 = CRTM_9_0 * s13;
        m9 = CRTM_9_1 * s10;
        s20 = m8 - m9;
        m10 = CRTM_9_11 * s10;
        m11 = CRTM_9_10 * s13;
        s21 = m11 + m10;
        s22 = s2 - s20;
        m12 = CRTM_9_6 * s20;
        s23 = s2 + m12;
        // Output point 2: x(1)
        out[out_strides[1]] = s23;

        s24 = s22 + s21;
        s25 = s22 - s21;
        // Output point 8: x(7)
        out[out_strides[7]] = s24;
        // Output point 5: x(4)
        out[out_strides[4]] = s25;

        m13 = CRTM_9_8 * s14;
        m14 = CRTM_9_9 * s11;
        s26 = m13 + m14;
        m15 = CRTM_9_2 * s14;
        m16 = CRTM_9_3 * s11;
        s27 = m15 - m16;
        s28 = s3 - s27;
        m17 = CRTM_9_6 * s27;
        s29 = s3 + m17;
        // Output point 3: x(2)
        out[out_strides[2]] = s29;

        s30 = s28 + s26;
        s31 = s28 - s26;
        // Output point 9: x(8)
        out[out_strides[8]] = s30;
        // Output point 6: x(5)
        out[out_strides[5]] = s31;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hc_rfft9c(UINT8 precision, UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hc_rfft9c_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft9c_fp64_fwd;
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
            return r2hc_rfft9c_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft9c_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
