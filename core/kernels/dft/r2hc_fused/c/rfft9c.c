// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft9c.c
 *
 *  @brief Radix-9 r2hc_fused Real-FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-9 real-to-halfcomplex fused of
 *  two different implementations (Standard DFT and Shifted DFT that
 *  differs in DFT weight matrix) using scalar operations for
 *  single-precision and double-precision inputs.
 *
 *  @author Amrin Fathima
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] = {
                                                    {{0, 44, 76, 32, 0, 0},
                                                     {0, 46, 75, 32, 0, 0}},
                                                    {{0, 44, 76, 32, 0, 0},
                                                     {0, 46, 75, 32, 0, 0}}};

ops_cycles_t get_ops_cnt_r2hcf_rfft9c(UINT8 precision, UINT8 direction)
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

static VOID r2hcf_rfft9c_fp32_fwd(VOID *in_real, VOID *in_imag, VOID *out_real,
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
    INTP v_in_stride = (strides->v_in_stride);
    INTP v_out_stride = (strides->v_out_stride);
    INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        /* Standard DFT */
        FLOAT a_in0, a_in1, a_in2, a_in3, a_in4, a_in5, a_in6, a_in7, a_in8;
        FLOAT a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_s9, a_s10,
              a_s11, a_s12, a_s13, a_s14, a_s15, a_s16, a_s17, a_s18, a_s19,
              a_s20, a_s21, a_s22, a_s23, a_s24, a_s25, a_s26, a_s27, a_s28;
        FLOAT a_m0, a_m1, a_m2, a_m3, a_m4, a_m5, a_m6, a_m7, a_m8, a_m9, a_m10,
              a_m11, a_m12, a_m13, a_m14, a_m15, a_m16, a_m17, a_m18, a_m19,
              a_m20, a_m21;

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

        a_s0 = a_in1 + a_in8;
        a_s1 = a_in1 - a_in8;
        a_s2 = a_in2 + a_in7;
        a_s3 = a_in2 - a_in7;
        a_s4 = a_in3 + a_in6;
        a_s5 = a_in3 - a_in6;
        a_s6 = a_in4 + a_in5;
        a_s7 = a_in4 - a_in5;

        a_s8 = a_s0 + a_s6;
        a_s9 = a_s8 + a_s2;
        a_s10 = a_s1 - a_s3;
        a_s11 = a_s10 + a_s7;
        a_s12 = a_s9 + a_s4;
        a_s13 = a_s12 + a_in0;

        a_m0 = CRTM_9_4 * a_s4;
        a_m1 = CRTM_9_5 * a_s5;
        a_s14 = a_in0 - a_m0;

        // Output point 1: X(0)
        *out = a_s13;

        a_m2 = CRTM_9_0 * a_s0;
        a_m3 = CRTM_9_2 * a_s2;
        a_m4 = CRTM_9_6 * a_s6;
        a_s15 = a_s14 + a_m2;
        a_s16 = a_s15 + a_m3;
        // Output point 4: X(3)
        out[out_strides[3]] = a_s16 - a_m4;

        a_m5 = CRTM_9_1 * a_s1;
        a_m6 = CRTM_9_3 * a_s3;
        a_m7 = CRTM_9_7 * a_s7;
        a_s17 = a_m5 + a_m6;
        a_s18 = a_s17 + a_m1;
        a_s19 = -(a_s18 + a_m7);
        // Output point 5: X(4)
        out[out_strides[4]] = a_s19;

        a_m8 = CRTM_9_0 * a_s6;
        a_m9 = CRTM_9_2 * a_s0;
        a_m10 = CRTM_9_6 * a_s2;
        a_s20 = a_s14 + a_m8;
        a_s21 = a_s20 + a_m9;
        // Output point 8: X(7)
        out[out_strides[7]] = a_s21 - a_m10;

        a_m11 = CRTM_9_1 * a_s7;
        a_m12 = CRTM_9_3 * a_s1;
        a_m13 = CRTM_9_7 * a_s3;
        a_s22 = a_m11 - a_m12;
        a_s23 = a_s22 + a_m1;
        // Output point 9: X(8)
        out[out_strides[8]] = a_s23 - a_m13;

        a_m14 = CRTM_9_4 * a_s9;
        a_s24 = a_in0 + a_s4;
        // Output point 12: X(11)
        out[out_strides[11]] = a_s24 - a_m14;

        a_m15 = -CRTM_9_5 * a_s11;
        // Output point 13: X(12)
        out[out_strides[12]] = a_m15;

        a_m16 = CRTM_9_0 * a_s2;
        a_m17 = CRTM_9_2 * a_s6;
        a_m18 = CRTM_9_6 * a_s0;
        a_s25 = a_s14 + a_m16;
        a_s26 = a_s25 + a_m17;
        // Output point 16: X(15)
        out[out_strides[15]] = a_s26 - a_m18;

        a_m19 = CRTM_9_1 * a_s3;
        a_m20 = CRTM_9_3 * a_s7;
        a_m21 = CRTM_9_7 * a_s1;
        a_s27 = a_m19 + a_m20;
        a_s28 = a_s27 - a_m1;
        // Output point 17: X(16)
        out[out_strides[16]] = a_s28 - a_m21;

        /* Shifted DFT */
        FLOAT b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7, b_in8;
        FLOAT b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9, b_s10,
              b_s11, b_s12, b_s13, b_s14, b_s15, b_s16, b_s17, b_s18, b_s19,
              b_s20, b_s21, b_s22, b_s23, b_s24, b_s25, b_s26, b_s27, b_s28,
              b_s29, b_s30, b_s31, b_s32, b_s33, b_s34, b_s35, b_s36, b_s37,
              b_s38, b_s39, b_s40;
        FLOAT b_m0, b_m1, b_m2, b_m3, b_m4, b_m5, b_m6, b_m7, b_m8, b_m9, b_m10,
              b_m11, b_m12, b_m13, b_m14, b_m15, b_m16, b_m17, b_m18, b_m19;

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

        b_s0 = b_in1 + b_in8;
        b_s1 = b_in1 - b_in8;
        b_s2 = b_in2 + b_in7;
        b_s3 = b_in2 - b_in7;
        b_s4 = b_in3 + b_in6;
        b_s5 = b_in3 - b_in6;
        b_s6 = b_in4 + b_in5;
        b_s7 = b_in4 - b_in5;

        b_m0 = CRTM_9_6 * b_s1;
        b_m1 = CRTM_9_6 * b_s3;
        b_m2 = CRTM_9_6 * b_s7;
        b_m3 = CRTM_9_0 * b_s3;
        b_m4 = CRTM_9_0 * b_s7;
        b_m5 = CRTM_9_0 * b_s1;
        b_m6 = CRTM_9_4 * b_s5;
        b_m7 = CRTM_9_2 * b_s7;
        b_m8 = CRTM_9_2 * b_s1;
        b_m9 = CRTM_9_2 * b_s3;

        b_m10 = CRTM_9_7 * b_s0;
        b_m11 = CRTM_9_7 * b_s2;
        b_m12 = CRTM_9_7 * b_s6;
        b_m13 = CRTM_9_1 * b_s2;
        b_m14 = CRTM_9_1 * b_s6;
        b_m15 = CRTM_9_1 * b_s0;
        b_m16 = CRTM_9_3 * b_s0;
        b_m17 = CRTM_9_3 * b_s2;
        b_m18 = CRTM_9_3 * b_s6;
        b_m19 = CRTM_9_5 * b_s4;

        b_s8 = b_m0 + b_m3;
        b_s9 = b_s8 + b_m6;
        b_s10 = b_s9 + b_m7;
        b_s11 = b_s10 + b_in0;
        b_s12 = b_m10 + b_m13;
        b_s13 = b_s12 + b_m19;
        b_s14 = -(b_s13 + b_m18);
        // Output point 2: X(1)
        out[out_strides[1]] = b_s11;
        // Output point 3: X(2)
        out[out_strides[2]] = b_s14;

        b_s15 = b_s1 - b_s3;
        b_s16 = b_s15 - b_s7;
        b_s17 = CRTM_9_4 * b_s16;
        b_s18 = b_s6 - b_s0;
        b_s19 = b_s18 - b_s2;
        b_s20 = CRTM_9_5 * b_s19;
        b_s21 = b_s17 + b_in0;
        b_s22 = b_s21 - b_s5;
        // Output point 6: X(5)
        out[out_strides[5]] = b_s22;
        // Output point 7: X(6)
        out[out_strides[6]] = b_s20;

        b_s23 = b_m4 - b_m1;
        b_s24 = b_s23 + b_m6;
        b_s25 = b_s24 - b_m8;
        b_s26 = b_s25 + b_in0;
        b_s27 = b_m11 - b_m14;
        b_s28 = b_s27 + b_m19;
        b_s29 = b_s28 - b_m16;
        // Output point 10: X(9)
        out[out_strides[9]] = b_s26;
        // Output point 11: X(10)
        out[out_strides[10]] = b_s29;

        b_s30 = b_m6 - b_m2;
        b_s31 = b_s30 - b_m5;
        b_s32 = b_s31 + b_m9;
        b_s33 = b_s32 + b_in0;
        b_s34 = b_m12 - b_m15;
        b_s35 = b_s34 - b_m19;
        b_s36 = b_s35 + b_m17;
        // Output point 14: X(13)
        out[out_strides[13]] = b_s33;
        // Output point 15: X(14)
        out[out_strides[14]] = b_s36;

        b_s37 = b_s3 - b_s5;
        b_s38 = b_s37 - b_s1;
        b_s39 = b_s38 + b_in0;
        b_s40 = b_s39 + b_s7;
        // Output point 18: X(17)
        out[out_strides[17]] = b_s40;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hcf_rfft9c_fp32_bwd(VOID *in_real, VOID *in_imag, VOID *out_real,
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
    // to save multiplications on the fly.
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
    INTP v_in_stride = (strides->v_in_stride);
    INTP v_out_stride = (strides->v_out_stride);
    INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        /* Standard DFT */
        FLOAT a_in0, a_in1, a_in2, a_in3, a_in4, a_in5, a_in6, a_in7, a_in8;
        FLOAT a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_s9, a_s10,
              a_s11, a_s12, a_s13, a_s14, a_s15, a_s16, a_s17, a_s18, a_s19,
              a_s20, a_s21, a_s22, a_s23, a_s24, a_s25, a_s26, a_s27, a_s28,
              a_s29, a_s30, a_s31;
        FLOAT a_m0, a_m1, a_m2, a_m3, a_m4, a_m5, a_m6, a_m7, a_m8, a_m9, a_m10,
              a_m11, a_m12, a_m13, a_m14, a_m15, a_m16, a_m17;

        // Input point 1: X(0)
        a_in0 = *in;
        // Input point 4: X(3)
        a_in1 = in[in_strides[3]];
        // Input point 5: X(4)
        a_in2 = in[in_strides[4]];
        // Input point 8: X(7)
        a_in3 = in[in_strides[7]];
        // Input point 9: X(8)
        a_in4 = in[in_strides[8]];
        // Input point 12: X(11)
        a_in5 = in[in_strides[11]];
        // Input point 13: X(12)
        a_in6 = in[in_strides[12]];
        // Input point 16: X(15)
        a_in7 = in[in_strides[15]];
        // Input point 17: X(16)
        a_in8 = in[in_strides[16]];

        a_m0 = CRTM_9_7 * a_in6;
        a_s0 = a_in0 - a_in5;
        a_m1 = CRTM_9_6 * a_in5;
        a_s1 = a_in0 + a_m1;
        a_s2 = a_s0 - a_m0;
        a_s3 = a_s0 + a_m0;
        a_s4 = a_in7 + a_in3;
        a_s5 = a_in7 - a_in3;
        a_m2 = CRTM_9_5 * a_s5;
        a_s6 = a_in8 + a_in4;
        a_m3 = CRTM_9_5 * a_s6;
        a_s7 = a_in4 - a_in8;
        a_s8 = a_in1 + a_s4;
        a_m4 = CRTM_9_4 * a_s7;
        a_s9 = a_in2 + a_m4;
        a_s10 = a_m2 + a_s9;
        a_s11 = a_s9 - a_m2;
        a_m5 = CRTM_9_4 * a_s4;
        a_s12 = a_in1 - a_m5;
        a_s13 = a_s12 - a_m3;
        a_s14 = a_s12 + a_m3;
        a_m6 = CRTM_9_6 * a_s8;
        a_s15 = a_s1 + a_m6;
        // Output point 1: x(0)
        *out = a_s15;

        a_s16 = a_s1 - a_s8;
        a_s17 = a_in2 - a_s7;
        a_m7 = CRTM_9_7 * a_s17;
        a_s18 = a_s16 - a_m7;
        a_s19 = a_s16 + a_m7;
        // Output point 7: x(6)
        out[out_strides[6]] = a_s18;
        // Output point 13: x(12)
        out[out_strides[12]] = a_s19;

        a_m8 = CRTM_9_0 * a_s13;
        a_m9 = CRTM_9_1 * a_s10;
        a_s20 = a_m8 - a_m9;
        a_m10 = CRTM_9_11 * a_s10;
        a_m11 = CRTM_9_10 * a_s13;
        a_s21 = a_m11 + a_m10;
        a_s22 = a_s2 - a_s20;
        a_m12 = CRTM_9_6 * a_s20;
        a_s23 = a_s2 + a_m12;
        // Output point 3: x(2)
        out[out_strides[2]] = a_s23;

        a_s24 = a_s22 + a_s21;
        a_s25 = a_s22 - a_s21;
        // Output point 15: x(14)
        out[out_strides[14]] = a_s24;
        // Output point 9: x(8)
        out[out_strides[8]] = a_s25;

        a_m13 = CRTM_9_8 * a_s14;
        a_m14 = CRTM_9_9 * a_s11;
        a_s26 = a_m13 + a_m14;
        a_m15 = CRTM_9_2 * a_s14;
        a_m16 = CRTM_9_3 * a_s11;
        a_s27 = a_m15 - a_m16;
        a_s28 = a_s3 - a_s27;
        a_m17 = CRTM_9_6 * a_s27;
        a_s29 = a_s3 + a_m17;
        // Output point 5: x(4)
        out[out_strides[4]] = a_s29;

        a_s30 = a_s28 + a_s26;
        a_s31 = a_s28 - a_s26;
        // Output point 17: x(16)
        out[out_strides[16]] = a_s30;
        // Output point 11: x(10)
        out[out_strides[10]] = a_s31;

        /* Shifted DFT */
        FLOAT b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7, b_in8;
        FLOAT b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9, b_s10,
              b_s11, b_s12, b_s13, b_s14, b_s15, b_s16, b_s17, b_s18, b_s19,
              b_s20, b_s21, b_s22, b_s23, b_s24, b_s25, b_s26, b_s27, b_s28,
              b_s29, b_s30, b_s31;
        FLOAT b_m0, b_m1, b_m2, b_m3, b_m4, b_m5, b_m6, b_m7, b_m8, b_m9, b_m10,
              b_m11, b_m12, b_m13, b_m14, b_m15, b_m16, b_m17;

        // Input point 2: X(1)
        b_in0 = in[in_strides[1]];
        // Input point 3: X(2)
        b_in1 = in[in_strides[2]];
        // Input point 6: X(5)
        b_in2 = in[in_strides[5]];
        // Input point 7: X(6)
        b_in3 = in[in_strides[6]];
        // Input point 10: X(9)
        b_in4 = in[in_strides[9]];
        // Input point 11: X(10)
        b_in5 = in[in_strides[10]];
        // Input point 14: X(13)
        b_in6 = in[in_strides[13]];
        // Input point 15: X(14)
        b_in7 = in[in_strides[14]];
        // Input point 18: X(17)
        b_in8 = in[in_strides[17]];

        b_m0 = CRTM_9_7 * b_in3;
        b_s0 = b_in2 - b_in8;
        b_m1 = CRTM_9_6 * b_in2;
        b_s1 = b_m1 + b_in8;
        b_s2 = b_s0 - b_m0;
        b_s3 = b_s0 + b_m0;

        b_s4 = b_in0 + b_in4;
        b_s5 = b_in4 - b_in0;
        b_m2 = CRTM_9_5 * b_s5;
        b_s6 = b_in5 - b_in1;
        b_s7 = b_in1 + b_in5;
        b_m3 = CRTM_9_5 * b_s7;
        b_s8 = b_in6 + b_s4;
        b_m4 = CRTM_9_4 * b_s6;
        b_s9 = b_m4 + b_in7;
        b_s10 = b_m2 - b_s9;
        b_s11 = b_m2 + b_s9;
        b_m5 = CRTM_9_4 * b_s4;
        b_s12 = b_m5 - b_in6;
        b_s13 = b_s12 + b_m3;
        b_s14 = b_s12 - b_m3;

        b_s15 = b_s8 - b_s1;
        b_s16 = b_s6 - b_in7;
        b_m6 = CRTM_9_7 * b_s16;
        b_m7 = CRTM_9_9 * b_s10;
        b_m8 = CRTM_9_8 * b_s13;
        b_s17 = b_m7 - b_m8;
        b_m9 = CRTM_9_2 * b_s13;
        b_m10 = CRTM_9_3 * b_s10;
        b_s18 = b_m9 + b_m10;
        b_s19 = b_s3 - b_s18;
        b_m11 = CRTM_9_10 * b_s14;
        b_m12 = CRTM_9_11 * b_s11;
        b_s20 = b_m11 + b_m12;
        b_m13 = CRTM_9_0 * b_s14;
        b_m14 = CRTM_9_1 * b_s11;
        b_s21 = b_m13 - b_m14;
        b_s22 = b_s21 - b_s2;

        b_m15 = CRTM_9_6 * b_s8;
        b_s23 = b_m15 + b_s1;
        // Output point 2: x(1)
        out[out_strides[1]] = b_s23;

        b_m16 = CRTM_9_6 * b_s21;
        b_s24 = b_m16 + b_s2;
        // Output point 4: x(3)
        out[out_strides[3]] = b_s24;

        b_m17 = CRTM_9_6 * b_s18;
        b_s25 = -(b_m17 + b_s3);
        // Output point 6: x(5)
        out[out_strides[5]] = b_s25;

        b_s26 = b_s15 + b_m6;
        // Output point 8: x(7)
        out[out_strides[7]] = b_s26;

        b_s27 = b_s22 + b_s20;
        // Output point 10: x(9)
        out[out_strides[9]] = b_s27;

        b_s28 = b_s19 + b_s17;
        // Output point 12: x(11)
        out[out_strides[11]] = b_s28;

        b_s29 = b_m6 - b_s15;
        // Output point 14: x(13)
        out[out_strides[13]] = b_s29;

        b_s30 = b_s20 - b_s22;
        // Output point 16: x(15)
        out[out_strides[15]] = b_s30;

        b_s31 = b_s17 - b_s19;
        // Output point 18: x(17)
        out[out_strides[17]] = b_s31;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hcf_rfft9c_fp64_fwd(VOID *in_real, VOID *in_imag, VOID *out_real,
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
    INTP v_in_stride = (strides->v_in_stride);
    INTP v_out_stride = (strides->v_out_stride);
    INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        /* Standard DFT */
        DOUBLE a_in0, a_in1, a_in2, a_in3, a_in4, a_in5, a_in6, a_in7, a_in8;
        DOUBLE a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_s9,
               a_s10, a_s11, a_s12, a_s13, a_s14, a_s15, a_s16, a_s17, a_s18,
               a_s19, a_s20, a_s21, a_s22, a_s23, a_s24, a_s25, a_s26, a_s27,
               a_s28;
        DOUBLE a_m0, a_m1, a_m2, a_m3, a_m4, a_m5, a_m6, a_m7, a_m8, a_m9,
               a_m10, a_m11, a_m12, a_m13, a_m14, a_m15, a_m16, a_m17, a_m18,
               a_m19, a_m20, a_m21;

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

        a_s0 = a_in1 + a_in8;
        a_s1 = a_in1 - a_in8;
        a_s2 = a_in2 + a_in7;
        a_s3 = a_in2 - a_in7;
        a_s4 = a_in3 + a_in6;
        a_s5 = a_in3 - a_in6;
        a_s6 = a_in4 + a_in5;
        a_s7 = a_in4 - a_in5;

        a_s8 = a_s0 + a_s6;
        a_s9 = a_s8 + a_s2;
        a_s10 = a_s1 - a_s3;
        a_s11 = a_s10 + a_s7;
        a_s12 = a_s9 + a_s4;
        a_s13 = a_s12 + a_in0;

        a_m0 = CRTM_9_4 * a_s4;
        a_m1 = CRTM_9_5 * a_s5;
        a_s14 = a_in0 - a_m0;

        // Output point 1: X(0)
        *out = a_s13;

        a_m2 = CRTM_9_0 * a_s0;
        a_m3 = CRTM_9_2 * a_s2;
        a_m4 = CRTM_9_6 * a_s6;
        a_s15 = a_s14 + a_m2;
        a_s16 = a_s15 + a_m3;
        // Output point 4: X(3)
        out[out_strides[3]] = a_s16 - a_m4;

        a_m5 = CRTM_9_1 * a_s1;
        a_m6 = CRTM_9_3 * a_s3;
        a_m7 = CRTM_9_7 * a_s7;
        a_s17 = a_m5 + a_m6;
        a_s18 = a_s17 + a_m1;
        a_s19 = -(a_s18 + a_m7);
        // Output point 5: X(4)
        out[out_strides[4]] = a_s19;

        a_m8 = CRTM_9_0 * a_s6;
        a_m9 = CRTM_9_2 * a_s0;
        a_m10 = CRTM_9_6 * a_s2;
        a_s20 = a_s14 + a_m8;
        a_s21 = a_s20 + a_m9;
        // Output point 8: X(7)
        out[out_strides[7]] = a_s21 - a_m10;

        a_m11 = CRTM_9_1 * a_s7;
        a_m12 = CRTM_9_3 * a_s1;
        a_m13 = CRTM_9_7 * a_s3;
        a_s22 = a_m11 - a_m12;
        a_s23 = a_s22 + a_m1;
        // Output point 9: X(8)
        out[out_strides[8]] = a_s23 - a_m13;

        a_m14 = CRTM_9_4 * a_s9;
        a_s24 = a_in0 + a_s4;
        // Output point 12: X(11)
        out[out_strides[11]] = a_s24 - a_m14;

        a_m15 = -CRTM_9_5 * a_s11;
        // Output point 13: X(12)
        out[out_strides[12]] = a_m15;

        a_m16 = CRTM_9_0 * a_s2;
        a_m17 = CRTM_9_2 * a_s6;
        a_m18 = CRTM_9_6 * a_s0;
        a_s25 = a_s14 + a_m16;
        a_s26 = a_s25 + a_m17;
        // Output point 16: X(15)
        out[out_strides[15]] = a_s26 - a_m18;

        a_m19 = CRTM_9_1 * a_s3;
        a_m20 = CRTM_9_3 * a_s7;
        a_m21 = CRTM_9_7 * a_s1;
        a_s27 = a_m19 + a_m20;
        a_s28 = a_s27 - a_m1;
        // Output point 17: X(16)
        out[out_strides[16]] = a_s28 - a_m21;

        /* Shifted DFT */
        DOUBLE b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7, b_in8;
        DOUBLE b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9,
               b_s10, b_s11, b_s12, b_s13, b_s14, b_s15, b_s16, b_s17, b_s18,
               b_s19, b_s20, b_s21, b_s22, b_s23, b_s24, b_s25, b_s26, b_s27,
               b_s28, b_s29, b_s30, b_s31, b_s32, b_s33, b_s34, b_s35, b_s36,
               b_s37, b_s38, b_s39, b_s40;
        DOUBLE b_m0, b_m1, b_m2, b_m3, b_m4, b_m5, b_m6, b_m7, b_m8, b_m9,
               b_m10, b_m11, b_m12, b_m13, b_m14, b_m15, b_m16, b_m17, b_m18,
               b_m19;

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

        b_s0 = b_in1 + b_in8;
        b_s1 = b_in1 - b_in8;
        b_s2 = b_in2 + b_in7;
        b_s3 = b_in2 - b_in7;
        b_s4 = b_in3 + b_in6;
        b_s5 = b_in3 - b_in6;
        b_s6 = b_in4 + b_in5;
        b_s7 = b_in4 - b_in5;

        b_m0 = CRTM_9_6 * b_s1;
        b_m1 = CRTM_9_6 * b_s3;
        b_m2 = CRTM_9_6 * b_s7;
        b_m3 = CRTM_9_0 * b_s3;
        b_m4 = CRTM_9_0 * b_s7;
        b_m5 = CRTM_9_0 * b_s1;
        b_m6 = CRTM_9_4 * b_s5;
        b_m7 = CRTM_9_2 * b_s7;
        b_m8 = CRTM_9_2 * b_s1;
        b_m9 = CRTM_9_2 * b_s3;

        b_m10 = CRTM_9_7 * b_s0;
        b_m11 = CRTM_9_7 * b_s2;
        b_m12 = CRTM_9_7 * b_s6;
        b_m13 = CRTM_9_1 * b_s2;
        b_m14 = CRTM_9_1 * b_s6;
        b_m15 = CRTM_9_1 * b_s0;
        b_m16 = CRTM_9_3 * b_s0;
        b_m17 = CRTM_9_3 * b_s2;
        b_m18 = CRTM_9_3 * b_s6;
        b_m19 = CRTM_9_5 * b_s4;

        b_s8 = b_m0 + b_m3;
        b_s9 = b_s8 + b_m6;
        b_s10 = b_s9 + b_m7;
        b_s11 = b_s10 + b_in0;
        b_s12 = b_m10 + b_m13;
        b_s13 = b_s12 + b_m19;
        b_s14 = -(b_s13 + b_m18);
        // Output point 2: X(1)
        out[out_strides[1]] = b_s11;
        // Output point 3: X(2)
        out[out_strides[2]] = b_s14;

        b_s15 = b_s1 - b_s3;
        b_s16 = b_s15 - b_s7;
        b_s17 = CRTM_9_4 * b_s16;
        b_s18 = b_s6 - b_s0;
        b_s19 = b_s18 - b_s2;
        b_s20 = CRTM_9_5 * b_s19;
        b_s21 = b_s17 + b_in0;
        b_s22 = b_s21 - b_s5;
        // Output point 6: X(5)
        out[out_strides[5]] = b_s22;
        // Output point 7: X(6)
        out[out_strides[6]] = b_s20;

        b_s23 = b_m4 - b_m1;
        b_s24 = b_s23 + b_m6;
        b_s25 = b_s24 - b_m8;
        b_s26 = b_s25 + b_in0;
        b_s27 = b_m11 - b_m14;
        b_s28 = b_s27 + b_m19;
        b_s29 = b_s28 - b_m16;
        // Output point 10: X(9)
        out[out_strides[9]] = b_s26;
        // Output point 11: X(10)
        out[out_strides[10]] = b_s29;

        b_s30 = b_m6 - b_m2;
        b_s31 = b_s30 - b_m5;
        b_s32 = b_s31 + b_m9;
        b_s33 = b_s32 + b_in0;
        b_s34 = b_m12 - b_m15;
        b_s35 = b_s34 - b_m19;
        b_s36 = b_s35 + b_m17;
        // Output point 14: X(13)
        out[out_strides[13]] = b_s33;
        // Output point 15: X(14)
        out[out_strides[14]] = b_s36;

        b_s37 = b_s3 - b_s5;
        b_s38 = b_s37 - b_s1;
        b_s39 = b_s38 + b_in0;
        b_s40 = b_s39 + b_s7;
        // Output point 18: X(17)
        out[out_strides[17]] = b_s40;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hcf_rfft9c_fp64_bwd(VOID *in_real, VOID *in_imag, VOID *out_real,
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
    // to save multiplications on the fly.
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
    INTP v_in_stride = (strides->v_in_stride);
    INTP v_out_stride = (strides->v_out_stride);
    INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        /* Standard DFT */
        DOUBLE a_in0, a_in1, a_in2, a_in3, a_in4, a_in5, a_in6, a_in7, a_in8;
        DOUBLE a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_s9,
               a_s10, a_s11, a_s12, a_s13, a_s14, a_s15, a_s16, a_s17, a_s18,
               a_s19, a_s20, a_s21, a_s22, a_s23, a_s24, a_s25, a_s26, a_s27,
               a_s28, a_s29, a_s30, a_s31;
        DOUBLE a_m0, a_m1, a_m2, a_m3, a_m4, a_m5, a_m6, a_m7, a_m8, a_m9,
               a_m10, a_m11, a_m12, a_m13, a_m14, a_m15, a_m16, a_m17;

        // Input point 1: X(0)
        a_in0 = *in;
        // Input point 4: X(3)
        a_in1 = in[in_strides[3]];
        // Input point 5: X(4)
        a_in2 = in[in_strides[4]];
        // Input point 8: X(7)
        a_in3 = in[in_strides[7]];
        // Input point 9: X(8)
        a_in4 = in[in_strides[8]];
        // Input point 12: X(11)
        a_in5 = in[in_strides[11]];
        // Input point 13: X(12)
        a_in6 = in[in_strides[12]];
        // Input point 16: X(15)
        a_in7 = in[in_strides[15]];
        // Input point 17: X(16)
        a_in8 = in[in_strides[16]];

        a_m0 = CRTM_9_7 * a_in6;
        a_s0 = a_in0 - a_in5;
        a_m1 = CRTM_9_6 * a_in5;
        a_s1 = a_in0 + a_m1;
        a_s2 = a_s0 - a_m0;
        a_s3 = a_s0 + a_m0;
        a_s4 = a_in7 + a_in3;
        a_s5 = a_in7 - a_in3;
        a_m2 = CRTM_9_5 * a_s5;
        a_s6 = a_in8 + a_in4;
        a_m3 = CRTM_9_5 * a_s6;
        a_s7 = a_in4 - a_in8;
        a_s8 = a_in1 + a_s4;
        a_m4 = CRTM_9_4 * a_s7;
        a_s9 = a_in2 + a_m4;
        a_s10 = a_m2 + a_s9;
        a_s11 = a_s9 - a_m2;
        a_m5 = CRTM_9_4 * a_s4;
        a_s12 = a_in1 - a_m5;
        a_s13 = a_s12 - a_m3;
        a_s14 = a_s12 + a_m3;
        a_m6 = CRTM_9_6 * a_s8;
        a_s15 = a_s1 + a_m6;
        // Output point 1: x(0)
        *out = a_s15;

        a_s16 = a_s1 - a_s8;
        a_s17 = a_in2 - a_s7;
        a_m7 = CRTM_9_7 * a_s17;
        a_s18 = a_s16 - a_m7;
        a_s19 = a_s16 + a_m7;
        // Output point 7: x(6)
        out[out_strides[6]] = a_s18;
        // Output point 13: x(12)
        out[out_strides[12]] = a_s19;

        a_m8 = CRTM_9_0 * a_s13;
        a_m9 = CRTM_9_1 * a_s10;
        a_s20 = a_m8 - a_m9;
        a_m10 = CRTM_9_11 * a_s10;
        a_m11 = CRTM_9_10 * a_s13;
        a_s21 = a_m11 + a_m10;
        a_s22 = a_s2 - a_s20;
        a_m12 = CRTM_9_6 * a_s20;
        a_s23 = a_s2 + a_m12;
        // Output point 3: x(2)
        out[out_strides[2]] = a_s23;

        a_s24 = a_s22 + a_s21;
        a_s25 = a_s22 - a_s21;
        // Output point 15: x(14)
        out[out_strides[14]] = a_s24;
        // Output point 9: x(8)
        out[out_strides[8]] = a_s25;

        a_m13 = CRTM_9_8 * a_s14;
        a_m14 = CRTM_9_9 * a_s11;
        a_s26 = a_m13 + a_m14;
        a_m15 = CRTM_9_2 * a_s14;
        a_m16 = CRTM_9_3 * a_s11;
        a_s27 = a_m15 - a_m16;
        a_s28 = a_s3 - a_s27;
        a_m17 = CRTM_9_6 * a_s27;
        a_s29 = a_s3 + a_m17;
        // Output point 5: x(4)
        out[out_strides[4]] = a_s29;

        a_s30 = a_s28 + a_s26;
        a_s31 = a_s28 - a_s26;
        // Output point 17: x(16)
        out[out_strides[16]] = a_s30;
        // Output point 11: x(10)
        out[out_strides[10]] = a_s31;

        /* Shifted DFT */
        DOUBLE b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7, b_in8;
        DOUBLE b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9,
               b_s10, b_s11, b_s12, b_s13, b_s14, b_s15, b_s16, b_s17, b_s18,
               b_s19, b_s20, b_s21, b_s22, b_s23, b_s24, b_s25, b_s26, b_s27,
               b_s28, b_s29, b_s30, b_s31;
        DOUBLE b_m0, b_m1, b_m2, b_m3, b_m4, b_m5, b_m6, b_m7, b_m8, b_m9,
               b_m10, b_m11, b_m12, b_m13, b_m14, b_m15, b_m16, b_m17;

        // Input point 2: X(1)
        b_in0 = in[in_strides[1]];
        // Input point 3: X(2)
        b_in1 = in[in_strides[2]];
        // Input point 6: X(5)
        b_in2 = in[in_strides[5]];
        // Input point 7: X(6)
        b_in3 = in[in_strides[6]];
        // Input point 10: X(9)
        b_in4 = in[in_strides[9]];
        // Input point 11: X(10)
        b_in5 = in[in_strides[10]];
        // Input point 14: X(13)
        b_in6 = in[in_strides[13]];
        // Input point 15: X(14)
        b_in7 = in[in_strides[14]];
        // Input point 18: X(17)
        b_in8 = in[in_strides[17]];

        b_m0 = CRTM_9_7 * b_in3;
        b_s0 = b_in2 - b_in8;
        b_m1 = CRTM_9_6 * b_in2;
        b_s1 = b_m1 + b_in8;
        b_s2 = b_s0 - b_m0;
        b_s3 = b_s0 + b_m0;

        b_s4 = b_in0 + b_in4;
        b_s5 = b_in4 - b_in0;
        b_m2 = CRTM_9_5 * b_s5;
        b_s6 = b_in5 - b_in1;
        b_s7 = b_in1 + b_in5;
        b_m3 = CRTM_9_5 * b_s7;
        b_s8 = b_in6 + b_s4;
        b_m4 = CRTM_9_4 * b_s6;
        b_s9 = b_m4 + b_in7;
        b_s10 = b_m2 - b_s9;
        b_s11 = b_m2 + b_s9;
        b_m5 = CRTM_9_4 * b_s4;
        b_s12 = b_m5 - b_in6;
        b_s13 = b_s12 + b_m3;
        b_s14 = b_s12 - b_m3;

        b_s15 = b_s8 - b_s1;
        b_s16 = b_s6 - b_in7;
        b_m6 = CRTM_9_7 * b_s16;
        b_m7 = CRTM_9_9 * b_s10;
        b_m8 = CRTM_9_8 * b_s13;
        b_s17 = b_m7 - b_m8;
        b_m9 = CRTM_9_2 * b_s13;
        b_m10 = CRTM_9_3 * b_s10;
        b_s18 = b_m9 + b_m10;
        b_s19 = b_s3 - b_s18;
        b_m11 = CRTM_9_10 * b_s14;
        b_m12 = CRTM_9_11 * b_s11;
        b_s20 = b_m11 + b_m12;
        b_m13 = CRTM_9_0 * b_s14;
        b_m14 = CRTM_9_1 * b_s11;
        b_s21 = b_m13 - b_m14;
        b_s22 = b_s21 - b_s2;

        b_m15 = CRTM_9_6 * b_s8;
        b_s23 = b_m15 + b_s1;
        // Output point 2: x(1)
        out[out_strides[1]] = b_s23;

        b_m16 = CRTM_9_6 * b_s21;
        b_s24 = b_m16 + b_s2;
        // Output point 4: x(3)
        out[out_strides[3]] = b_s24;

        b_m17 = CRTM_9_6 * b_s18;
        b_s25 = -(b_m17 + b_s3);
        // Output point 6: x(5)
        out[out_strides[5]] = b_s25;

        b_s26 = b_s15 + b_m6;
        // Output point 8: x(7)
        out[out_strides[7]] = b_s26;

        b_s27 = b_s22 + b_s20;
        // Output point 10: x(9)
        out[out_strides[9]] = b_s27;

        b_s28 = b_s19 + b_s17;
        // Output point 12: x(11)
        out[out_strides[11]] = b_s28;

        b_s29 = b_m6 - b_s15;
        // Output point 14: x(13)
        out[out_strides[13]] = b_s29;

        b_s30 = b_s20 - b_s22;
        // Output point 16: x(15)
        out[out_strides[15]] = b_s30;

        b_s31 = b_s17 - b_s19;
        // Output point 18: x(17)
        out[out_strides[17]] = b_s31;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hcf_rfft9c(UINT8 precision, UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hcf_rfft9c_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft9c_fp64_fwd;
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
            return r2hcf_rfft9c_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft9c_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
