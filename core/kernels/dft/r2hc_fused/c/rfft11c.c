// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft11c.c
 *
 *  @brief Radix-11 r2hc_fused Real-FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-11 real-to-halfcomplex fused of two
 *  different implementations (Standard DFT and Shifted DFT that differs in DFT
 *  weight matrix) using scalar operations for single-precision
 *  and double-precision inputs.
 *
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                   {{{0, 100, 124, 44, 0, 0},
                                                     {0, 102, 124, 44, 0, 0}},
                                                    {{0, 100, 124, 44, 0, 0},
                                                     {0, 102, 124, 44, 0, 0}}};

ops_cycles_t get_ops_cnt_r2hcf_rfft11c(FFTZ_UINT8 precision,
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

static FFTZ_VOID r2hcf_rfft11c_fp32_fwd(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                        FFTZ_VOID *out_real,
                                        FFTZ_VOID *out_imag, FFTZ_INTP n,
                                        aoclfftz_strides_t *strides,
                                        FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_11_1 =
        0.841253532831181168861811648919367717513292498f;
    const FFTZ_FLOAT CRTM_11_2 =
        0.540640817455597582107635954318691695431770608f;
    const FFTZ_FLOAT CRTM_11_3 =
        0.415415013001886425529274149229623203524004910f;
    const FFTZ_FLOAT CRTM_11_4 =
        0.909631995354518371411715383079028460060241051f;
    const FFTZ_FLOAT CRTM_11_5 =
        0.142314838273285140443792668616369668791051361f;
    const FFTZ_FLOAT CRTM_11_6 =
        0.989821441880932732376092037776718787376519372f;
    const FFTZ_FLOAT CRTM_11_7 =
        0.654860733945285064056925072466293553183791199f;
    const FFTZ_FLOAT CRTM_11_8 =
        0.755749574354258283774035843972344420179717445f;
    const FFTZ_FLOAT CRTM_11_9 =
        0.959492973614497389890368057066327699062454848f;
    const FFTZ_FLOAT CRTM_11_10 =
        0.281732556841429697711417915346616899035777899f;

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
        /* Standard DFT */
        FFTZ_FLOAT a_in0, a_in1, a_in2, a_in3, a_in4, a_in5, a_in6, a_in7,
            a_in8, a_in9, a_in10;
        FFTZ_FLOAT a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_s9,
              a_s10, a_s11, a_s12, a_s13, a_s14, a_s15, a_s16, a_s17, a_s18,
              a_s19, a_s20, a_s21, a_s22, a_s23, a_s24, a_s25, a_s26, a_s27,
              a_s28, a_s29, a_s30, a_s31, a_s32, a_s33, a_s34, a_s35, a_s36,
              a_s37, a_s38, a_s39, a_s40, a_s41, a_s42, a_s43, a_s44, a_s45,
              a_s46, a_s47, a_s48;
        FFTZ_FLOAT a_m0, a_m1, a_m2, a_m3, a_m4, a_m5, a_m6, a_m7, a_m8, a_m9,
              a_m10, a_m11, a_m12, a_m13, a_m14, a_m15, a_m16, a_m17, a_m18,
              a_m19, a_m20, a_m21, a_m22, a_m23, a_m24, a_m25, a_m26, a_m27,
              a_m28, a_m29, a_m30, a_m31, a_m32, a_m33, a_m34, a_m35, a_m36,
              a_m37, a_m38, a_m39, a_m40, a_m41, a_m42, a_m43, a_m44, a_m45,
              a_m46, a_m47, a_m48, a_m49;

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

        a_s0 = a_in1 + a_in10;
        a_s1 = a_in2 + a_in9;
        a_s2 = a_in3 + a_in8;
        a_s3 = a_in4 + a_in7;
        a_s4 = a_in5 + a_in6;
        a_s5 = a_in1 - a_in10;
        a_s6 = a_in2 - a_in9;
        a_s7 = a_in3 - a_in8;
        a_s8 = a_in4 - a_in7;
        a_s9 = a_in5 - a_in6;

        a_s10 = a_s0 + a_s1;
        a_s11 = a_s2 + a_s3;
        a_s12 = a_s4 + a_in0;
        a_s13 = a_s10 + a_s11;

        // Output point 1: X(0)
        *out = a_s12 + a_s13;

        a_m0 = CRTM_11_1 * a_s0;
        a_m1 = CRTM_11_3 * a_s1;
        a_m2 = CRTM_11_5 * a_s2;
        a_m3 = CRTM_11_7 * a_s3;
        a_m4 = CRTM_11_9 * a_s4;

        a_s14 = a_m0 + a_m1;
        a_s15 = a_m2 + a_m3;
        a_s16 = a_in0 - a_m4;
        a_s17 = a_s14 - a_s15;

        // Output point 4: X(3)
        out[out_strides[3]] = a_s16 + a_s17;

        a_m5 = CRTM_11_2 * a_s5;
        a_m6 = CRTM_11_4 * a_s6;
        a_m7 = CRTM_11_6 * a_s7;
        a_m8 = CRTM_11_8 * a_s8;
        a_m9 = CRTM_11_10 * a_s9;

        a_s18 = a_m5 + a_m6;
        a_s19 = a_m7 + a_m8;
        a_s20 = a_s19 + a_m9;

        // Output point 5: X(4)
        out[out_strides[4]] = -(a_s18 + a_s20);

        a_m10 = CRTM_11_1 * a_s4;
        a_m11 = CRTM_11_3 * a_s0;
        a_m12 = CRTM_11_5 * a_s3;
        a_m13 = CRTM_11_7 * a_s1;
        a_m14 = CRTM_11_9 * a_s2;

        a_s21 = a_m10 + a_m11;
        a_s22 = a_m12 + a_m13;
        a_s23 = a_in0 - a_m14;
        a_s24 = a_s21 - a_s22;

        // Output point 8: X(7)
        out[out_strides[7]] = a_s23 + a_s24;

        a_m15 = CRTM_11_2 * a_s9;
        a_m16 = CRTM_11_4 * a_s5;
        a_m17 = CRTM_11_6 * a_s8;
        a_m18 = CRTM_11_8 * a_s6;
        a_m19 = CRTM_11_10 * a_s7;

        a_s25 = a_m15 - a_m16;
        a_s26 = a_m17 - a_m18;
        a_s27 = a_s26 + a_m19;

        // Output point 9: X(8)
        out[out_strides[8]] = a_s25 + a_s27;

        a_m20 = CRTM_11_1 * a_s3;
        a_m21 = CRTM_11_3 * a_s2;
        a_m22 = CRTM_11_5 * a_s0;
        a_m23 = CRTM_11_7 * a_s4;
        a_m24 = CRTM_11_9 * a_s1;

        a_s28 = a_m20 + a_m21;
        a_s29 = a_m22 + a_m23;
        a_s30 = a_in0 - a_m24;
        a_s31 = a_s30 - a_s29;

        // Output point 12: X(11)
        out[out_strides[11]] = a_s28 + a_s31;

        a_m25 = CRTM_11_2 * a_s8;
        a_m26 = CRTM_11_4 * a_s7;
        a_m27 = CRTM_11_6 * a_s5;
        a_m28 = CRTM_11_8 * a_s9;
        a_m29 = CRTM_11_10 * a_s6;

        a_s32 = a_m26 - a_m25;
        a_s33 = a_m27 + a_m28;
        a_s34 = a_s32 - a_s33;

        // Output point 13: X(12)
        out[out_strides[12]] = a_s34 + a_m29;

        a_m30 = CRTM_11_1 * a_s2;
        a_m31 = CRTM_11_3 * a_s4;
        a_m32 = CRTM_11_5 * a_s1;
        a_m33 = CRTM_11_7 * a_s0;
        a_m34 = CRTM_11_9 * a_s3;

        a_s35 = a_m30 + a_m31;
        a_s36 = a_m32 + a_m33;
        a_s37 = a_in0 - a_m34;
        a_s38 = a_s35 - a_s36;

        // Output point 16: X(15)
        out[out_strides[15]] = a_s38 + a_s37;

        a_m35 = CRTM_11_2 * a_s7;
        a_m36 = CRTM_11_4 * a_s9;
        a_m37 = CRTM_11_6 * a_s6;
        a_m38 = CRTM_11_8 * a_s5;
        a_m39 = CRTM_11_10 * a_s8;

        a_s39 = a_m36 - a_m35;
        a_s40 = a_m37 - a_m38;
        a_s41 = a_s40 - a_m39;

        // Output point 17: X(16)
        out[out_strides[16]] = a_s39 + a_s41;

        a_m40 = CRTM_11_1 * a_s1;
        a_m41 = CRTM_11_3 * a_s3;
        a_m42 = CRTM_11_5 * a_s4;
        a_m43 = CRTM_11_7 * a_s2;
        a_m44 = CRTM_11_9 * a_s0;

        a_s42 = a_m40 + a_m41;
        a_s43 = a_m42 + a_m43;
        a_s44 = a_in0 - a_m44;
        a_s45 = a_s42 - a_s43;

        // Output point 20: X(19)
        out[out_strides[19]] = a_s44 + a_s45;

        a_m45 = CRTM_11_2 * a_s6;
        a_m46 = CRTM_11_4 * a_s8;
        a_m47 = CRTM_11_6 * a_s9;
        a_m48 = CRTM_11_8 * a_s7;
        a_m49 = CRTM_11_10 * a_s5;

        a_s46 = a_m45 + a_m46;
        a_s47 = a_m47 + a_m48;
        a_s48 = a_s46 - a_s47;

        // Output point 21: X(20)
        out[out_strides[20]] = a_s48 - a_m49;

        /* Shifted DFT */
        FFTZ_FLOAT b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7,
            b_in8, b_in9, b_in10;
        FFTZ_FLOAT b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9,
              b_s10, b_s11, b_s12, b_s13, b_s14, b_s15, b_s16, b_s17, b_s18,
              b_s19, b_s20, b_s21, b_s22, b_s23, b_s24, b_s25, b_s26, b_s27,
              b_s28, b_s29, b_s30, b_s31, b_s32, b_s33, b_s34, b_s35, b_s36,
              b_s37, b_s38, b_s39, b_s40, b_s41, b_s42, b_s43, b_s44, b_s45,
              b_s46, b_s47, b_s48, b_s49, b_s50, b_s51, b_s52, b_s53, b_s54;
        FFTZ_FLOAT b_m0, b_m1, b_m2, b_m3, b_m4, b_m5, b_m6, b_m7, b_m8, b_m9,
              b_m10, b_m11, b_m12, b_m13, b_m14, b_m15, b_m16, b_m17, b_m18,
              b_m19, b_m20, b_m21, b_m22, b_m23, b_m24, b_m25, b_m26, b_m27,
              b_m28, b_m29, b_m30, b_m31, b_m32, b_m33, b_m34, b_m35, b_m36,
              b_m37, b_m38, b_m39, b_m40, b_m41, b_m42, b_m43, b_m44, b_m45,
              b_m46, b_m47, b_m48, b_m49;

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

        b_s0 = b_in1 + b_in10;
        b_s1 = b_in2 + b_in9;
        b_s2 = b_in3 + b_in8;
        b_s3 = b_in4 + b_in7;
        b_s4 = b_in5 + b_in6;
        b_s5 = b_in1 - b_in10;
        b_s6 = b_in2 - b_in9;
        b_s7 = b_in3 - b_in8;
        b_s8 = b_in4 - b_in7;
        b_s9 = b_in5 - b_in6;

        b_m0 = CRTM_11_9 * b_s5;
        b_m1 = CRTM_11_1 * b_s6;
        b_s10 = b_m0 + b_m1;
        b_m2 = CRTM_11_7 * b_s7;
        b_s11 = b_s10 + b_m2;
        b_m3 = CRTM_11_3 * b_s8;
        b_s12 = b_s11 + b_m3;
        b_m4 = CRTM_11_5 * b_s9;
        b_s13 = b_s12 + b_m4;

        // Output point 2: X(1)
        out[out_strides[1]] = b_in0 + b_s13;

        b_m5 = CRTM_11_10 * b_s0;
        b_m6 = CRTM_11_2 * b_s1;
        b_s14 = b_m5 + b_m6;
        b_m7 = CRTM_11_8 * b_s2;
        b_s15 = b_s14 + b_m7;
        b_m8 = CRTM_11_4 * b_s3;
        b_s16 = b_s15 + b_m8;
        b_m9 = CRTM_11_6 * b_s4;
        b_s17 = b_s16 + b_m9;

        // Output point 3: X(2)
        out[out_strides[2]] = -b_s17;

        b_m10 = CRTM_11_7 * b_s5;
        b_m11 = CRTM_11_5 * b_s6;
        b_s18 = b_m10 - b_m11;
        b_m12 = CRTM_11_1 * b_s7;
        b_s19 = b_s18 - b_m12;
        b_m13 = CRTM_11_9 * b_s8;
        b_s20 = b_s19 - b_m13;
        b_m14 = CRTM_11_3 * b_s9;
        b_s21 = b_s20 - b_m14;

        // Output point 6: X(5)
        out[out_strides[5]] = b_in0 + b_s21;

        b_m15 = CRTM_11_8 * b_s0;
        b_m16 = CRTM_11_6 * b_s1;
        b_s22 = b_m15 + b_m16;
        b_m17 = CRTM_11_2 * b_s2;
        b_s23 = b_s22 + b_m17;
        b_m18 = CRTM_11_10 * b_s3;
        b_s24 = b_s23 - b_m18;
        b_m19 = CRTM_11_4 * b_s4;
        b_s25 = b_s24 - b_m19;

        // Output point 7: X(6)
        out[out_strides[6]] = -b_s25;

        b_m20 = CRTM_11_5 * b_s5;
        b_m21 = CRTM_11_9 * b_s6;
        b_s26 = b_m20 - b_m21;
        b_m22 = CRTM_11_3 * b_s7;
        b_s27 = b_s26 - b_m22;
        b_m23 = CRTM_11_1 * b_s8;
        b_s28 = b_s27 + b_m23;
        b_m24 = CRTM_11_7 * b_s9;
        b_s29 = b_s28 + b_m24;

        // Output point 10: X(9)
        out[out_strides[9]] = b_in0 + b_s29;

        b_m25 = CRTM_11_6 * b_s0;
        b_m26 = CRTM_11_10 * b_s1;
        b_s30 = b_m25 + b_m26;
        b_m27 = CRTM_11_4 * b_s2;
        b_s31 = b_s30 - b_m27;
        b_m28 = CRTM_11_2 * b_s3;
        b_s32 = b_s31 - b_m28;
        b_m29 = CRTM_11_8 * b_s4;
        b_s33 = b_s32 + b_m29;

        // Output point 11: X(10)
        out[out_strides[10]] = -b_s33;

        b_m30 = CRTM_11_3 * b_s5;
        b_m31 = CRTM_11_7 * b_s6;
        b_s34 = b_m30 + b_m31;
        b_m32 = CRTM_11_9 * b_s7;
        b_s35 = b_m32 - b_s34;
        b_m33 = CRTM_11_5 * b_s8;
        b_s36 = b_s35 - b_m33;
        b_m34 = CRTM_11_1 * b_s9;
        b_s37 = b_s36 - b_m34;

        // Output point 14: X(13)
        out[out_strides[13]] = b_in0 + b_s37;

        b_m35 = CRTM_11_4 * b_s0;
        b_m36 = CRTM_11_8 * b_s1;
        b_s38 = b_m35 - b_m36;
        b_m37 = CRTM_11_10 * b_s2;
        b_s39 = b_s38 - b_m37;
        b_m38 = CRTM_11_6 * b_s3;
        b_s40 = b_s39 + b_m38;
        b_m39 = CRTM_11_2 * b_s4;
        b_s41 = b_s40 - b_m39;

        // Output point 15: X(14)
        out[out_strides[14]] = -b_s41;

        b_m40 = CRTM_11_1 * b_s5;
        b_m41 = CRTM_11_3 * b_s6;
        b_s42 = b_m41 - b_m40;
        b_m42 = CRTM_11_5 * b_s7;
        b_s43 = b_s42 + b_m42;
        b_m43 = CRTM_11_7 * b_s8;
        b_s44 = b_s43 - b_m43;
        b_m44 = CRTM_11_9 * b_s9;
        b_s45 = b_s44 + b_m44;

        // Output point 18: X(17)
        out[out_strides[17]] = b_in0 + b_s45;

        b_m45 = CRTM_11_2 * b_s0;
        b_m46 = CRTM_11_4 * b_s1;
        b_s46 = b_m45 - b_m46;
        b_m47 = CRTM_11_6 * b_s2;
        b_s47 = b_s46 + b_m47;
        b_m48 = CRTM_11_8 * b_s3;
        b_s48 = b_s47 - b_m48;
        b_m49 = CRTM_11_10 * b_s4;
        b_s49 = b_s48 + b_m49;

        // Output point 19: X(18)
        out[out_strides[18]] = -b_s49;

        b_s50 = b_in0 - b_s5;
        b_s51 = b_s50 + b_s6;
        b_s52 = b_s51 - b_s7;
        b_s53 = b_s52 + b_s8;
        b_s54 = b_s53 - b_s9;

        // Output point 22: X(21)
        out[out_strides[21]] = b_s54;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft11c_fp32_bwd(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                        FFTZ_VOID *out_real,
                                        FFTZ_VOID *out_imag, FFTZ_INTP n,
                                        aoclfftz_strides_t *strides,
                                        FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");

    const FFTZ_FLOAT CRTM_11_1 =
        1.682507065662362337723623297838735435026584997f;
    const FFTZ_FLOAT CRTM_11_2 =
        1.081281634911195164215271908637383390863541216f;
    const FFTZ_FLOAT CRTM_11_3 =
        0.830830026003772851058548298459246407048009821f;
    const FFTZ_FLOAT CRTM_11_4 =
        1.819263990709036742823430766158056920120482102f;
    const FFTZ_FLOAT CRTM_11_5 =
        0.284629676546570280887585337232739337582102722f;
    const FFTZ_FLOAT CRTM_11_6 =
        1.979642883761865464752184075553437574753038744f;
    const FFTZ_FLOAT CRTM_11_7 =
        1.309721467890570128113850144932587106367582399f;
    const FFTZ_FLOAT CRTM_11_8 =
        1.511499148708516567548071687944688840359434890f;
    const FFTZ_FLOAT CRTM_11_9 =
        1.918985947228994779780736114132655398124909697f;
    const FFTZ_FLOAT CRTM_11_10 =
        0.563465113682859395422835830693233798071555798f;
    const FFTZ_FLOAT CRTM_11_11 =
        2.000000000000000000000000000000000000000000000f;

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
            a_in8, a_in9, a_in10;
        FFTZ_FLOAT a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_s9,
              a_s10, a_s11, a_s12, a_s13, a_s14, a_s15, a_s16, a_s17, a_s18,
              a_s19, a_s20, a_s21, a_s22, a_s23, a_s24, a_s25, a_s26, a_s27,
              a_s28, a_s29, a_s30, a_s31, a_s32, a_s33, a_s34, a_s35, a_s36,
              a_s37, a_s38, a_s39, a_s40, a_s41, a_s42, a_s43, a_s44, a_s45,
              a_s46, a_s47, a_s48;
        FFTZ_FLOAT a_m0, a_m1, a_m2, a_m3, a_m4, a_m5, a_m6, a_m7, a_m8, a_m9,
              a_m10, a_m11, a_m12, a_m13, a_m14, a_m15, a_m16, a_m17, a_m18,
              a_m19, a_m20, a_m21, a_m22, a_m23, a_m24, a_m25, a_m26, a_m27,
              a_m28, a_m29, a_m30, a_m31, a_m32, a_m33, a_m34, a_m35, a_m36,
              a_m37, a_m38, a_m39, a_m40, a_m41, a_m42, a_m43, a_m44, a_m45,
              a_m46, a_m47, a_m48, a_m49, a_m50;

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
        // Input point 20: X(19)
        a_in9 = in[in_strides[19]];
        // Input point 21: X(20)
        a_in10 = in[in_strides[20]];

        a_s0 = a_in1 + a_in3;
        a_s1 = a_in5 + a_in7;
        a_s2 = a_s0 + a_s1;
        a_s3 = a_s2 + a_in9;
        a_m0 = CRTM_11_11 * a_s3;

        // Output point 1: x(0)
        *out = a_in0 + a_m0;

        a_m1 = CRTM_11_1 * a_in1;
        a_m2 = CRTM_11_2 * a_in2;
        a_m3 = CRTM_11_3 * a_in3;
        a_m4 = CRTM_11_4 * a_in4;
        a_m5 = CRTM_11_5 * a_in5;
        a_m6 = CRTM_11_6 * a_in6;
        a_m7 = CRTM_11_7 * a_in7;
        a_m8 = CRTM_11_8 * a_in8;
        a_m9 = CRTM_11_9 * a_in9;
        a_m10 = CRTM_11_10 * a_in10;

        a_s4 = a_m1 + a_m3;
        a_s5 = a_s4 + a_in0;
        a_s6 = a_m5 + a_m7;
        a_s7 = a_s6 + a_m9;
        a_s8 = a_s5 - a_s7;

        a_s9 = a_m2 + a_m4;
        a_s10 = a_m6 + a_m8;
        a_s11 = a_s9 + a_m10;
        a_s12 = a_s10 + a_s11;

        // Output point 3: x(2)
        out[out_strides[2]] = a_s8 - a_s12;
        // Output point 21: x(20)
        out[out_strides[20]] = a_s8 + a_s12;

        a_m11 = CRTM_11_1 * a_in9;
        a_m12 = CRTM_11_2 * a_in10;
        a_m13 = CRTM_11_3 * a_in1;
        a_m14 = CRTM_11_4 * a_in2;
        a_m15 = CRTM_11_5 * a_in7;
        a_m16 = CRTM_11_6 * a_in8;
        a_m17 = CRTM_11_7 * a_in3;
        a_m18 = CRTM_11_8 * a_in4;
        a_m19 = CRTM_11_9 * a_in5;
        a_m20 = CRTM_11_10 * a_in6;

        a_s13 = a_m11 + a_m13;
        a_s14 = a_s13 + a_in0;
        a_s15 = a_m15 + a_m17;
        a_s16 = a_s15 + a_m19;
        a_s17 = a_s14 - a_s16;

        a_s18 = a_m12 - a_m14;
        a_s19 = a_m16 - a_m18;
        a_s20 = a_s19 + a_m20;
        a_s21 = a_s18 + a_s20;

        // Output point 5: x(4)
        out[out_strides[4]] = a_s17 + a_s21;
        // Output point 19: x(18)
        out[out_strides[18]] = a_s17 - a_s21;

        a_m21 = CRTM_11_1 * a_in7;
        a_m22 = CRTM_11_2 * a_in8;
        a_m23 = CRTM_11_3 * a_in5;
        a_m24 = CRTM_11_4 * a_in6;
        a_m25 = CRTM_11_5 * a_in1;
        a_m26 = CRTM_11_6 * a_in2;
        a_m27 = CRTM_11_7 * a_in9;
        a_m28 = CRTM_11_8 * a_in10;
        a_m29 = CRTM_11_9 * a_in3;
        a_m30 = CRTM_11_10 * a_in4;

        a_s22 = a_m21 + a_m23;
        a_s23 = a_s22 + a_in0;
        a_s24 = a_m25 + a_m27;
        a_s25 = a_s24 + a_m29;
        a_s26 = a_s23 - a_s25;

        a_s27 = a_m22 - a_m24;
        a_s28 = a_m26 + a_m28;
        a_s29 = a_s28 - a_m30;
        a_s30 = a_s27 + a_s29;

        // Output point 7: x(6)
        out[out_strides[6]] = a_s26 - a_s30;
        // Output point 17: x(16)
        out[out_strides[16]] = a_s26 + a_s30;

        a_m31 = CRTM_11_1 * a_in5;
        a_m32 = CRTM_11_2 * a_in6;
        a_m33 = CRTM_11_3 * a_in9;
        a_m34 = CRTM_11_4 * a_in10;
        a_m35 = CRTM_11_5 * a_in3;
        a_m36 = CRTM_11_6 * a_in4;
        a_m37 = CRTM_11_7 * a_in1;
        a_m38 = CRTM_11_8 * a_in2;
        a_m39 = CRTM_11_9 * a_in7;
        a_m40 = CRTM_11_10 * a_in8;

        a_s31 = a_m31 + a_m33;
        a_s32 = a_s31 + a_in0;
        a_s33 = a_m35 + a_m37;
        a_s34 = a_s33 + a_m39;
        a_s35 = a_s32 - a_s34;

        a_s36 = a_m32 - a_m34;
        a_s37 = a_m38 - a_m36;
        a_s38 = a_s37 + a_m40;
        a_s39 = a_s36 + a_s38;

        // Output point 9: x(8)
        out[out_strides[8]] = a_s35 - a_s39;
        // Output point 15: x(14)
        out[out_strides[14]] = a_s35 + a_s39;

        a_m41 = CRTM_11_1 * a_in3;
        a_m42 = CRTM_11_2 * a_in4;
        a_m43 = CRTM_11_3 * a_in7;
        a_m44 = CRTM_11_4 * a_in8;
        a_m45 = CRTM_11_5 * a_in9;
        a_m46 = CRTM_11_6 * a_in10;
        a_m47 = CRTM_11_7 * a_in5;
        a_m48 = CRTM_11_8 * a_in6;
        a_m49 = CRTM_11_9 * a_in1;
        a_m50 = CRTM_11_10 * a_in2;

        a_s40 = a_m41 + a_m43;
        a_s41 = a_s40 + a_in0;
        a_s42 = a_m45 + a_m47;
        a_s43 = a_s42 + a_m49;
        a_s44 = a_s41 - a_s43;

        a_s45 = a_m42 + a_m44;
        a_s46 = a_m46 + a_m48;
        a_s47 = a_s45 - a_s46;
        a_s48 = a_s47 - a_m50;

        // Output point 11: x(10)
        out[out_strides[10]] = a_s44 + a_s48;
        // Output point 13: x(12)
        out[out_strides[12]] = a_s44 - a_s48;

        /* Shifted DFT */
        FFTZ_FLOAT b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7,
            b_in8, b_in9, b_in10;
        FFTZ_FLOAT b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9,
              b_s10, b_s11, b_s12, b_s13, b_s14, b_s15, b_s16, b_s17, b_s18,
              b_s19, b_s20, b_s21, b_s22, b_s23, b_s24, b_s25, b_s26, b_s27,
              b_s28, b_s29, b_s30, b_s31, b_s32, b_s33, b_s34, b_s35, b_s36,
              b_s37, b_s38, b_s39, b_s40, b_s41, b_s42, b_s43, b_s44, b_s45,
              b_s46, b_s47, b_s48;
        FFTZ_FLOAT b_m0, b_m1, b_m2, b_m3, b_m4, b_m5, b_m6, b_m7, b_m8, b_m9,
              b_m10, b_m11, b_m12, b_m13, b_m14, b_m15, b_m16, b_m17, b_m18,
              b_m19, b_m20, b_m21, b_m22, b_m23, b_m24, b_m25, b_m26, b_m27,
              b_m28, b_m29, b_m30, b_m31, b_m32, b_m33, b_m34, b_m35, b_m36,
              b_m37, b_m38, b_m39, b_m40, b_m41, b_m42, b_m43, b_m44, b_m45,
              b_m46, b_m47, b_m48, b_m49, b_m50;

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
        // Input point 19: X(18)
        b_in9 = in[in_strides[18]];
        // Input point 22: X(21)
        b_in10 = in[in_strides[21]];

        b_s0 = b_in0 + b_in2;
        b_s1 = b_in4 + b_in6;
        b_s2 = b_s0 + b_s1;
        b_s3 = b_s2 + b_in8;
        b_m0 = CRTM_11_11 * b_s3;

        // Output point 2: x(1)
        out[out_strides[1]] = b_m0 + b_in10;

        b_m1 = CRTM_11_9 * b_in0;
        b_m2 = CRTM_11_10 * b_in1;
        b_m3 = CRTM_11_1 * b_in8;
        b_m4 = CRTM_11_2 * b_in9;
        b_m5 = CRTM_11_7 * b_in2;
        b_m6 = CRTM_11_8 * b_in3;
        b_m7 = CRTM_11_3 * b_in6;
        b_m8 = CRTM_11_4 * b_in7;
        b_m9 = CRTM_11_5 * b_in4;
        b_m10 = CRTM_11_6 * b_in5;

        b_s4 = b_m2 + b_m4;
        b_s5 = b_m6 + b_m8;
        b_s6 = b_s4 + b_s5;
        b_s7 = b_s6 + b_m10;
        b_s8 = b_m1 - b_m3;
        b_s9 = b_m5 - b_m7;
        b_s10 = b_s8 + b_s9;
        b_s11 = b_m9 - b_in10;
        b_s12 = b_s10 + b_s11;

        // Output point 4: x(3)
        out[out_strides[3]] = b_s12 - b_s7;
        // Output point 22: x(21)
        out[out_strides[21]] = -(b_s12 + b_s7);

        b_m11 = CRTM_11_1 * b_in0;
        b_m12 = CRTM_11_2 * b_in1;
        b_m13 = CRTM_11_5 * b_in2;
        b_m14 = CRTM_11_6 * b_in3;
        b_m15 = CRTM_11_9 * b_in4;
        b_m16 = CRTM_11_10 * b_in5;
        b_m17 = CRTM_11_7 * b_in6;
        b_m18 = CRTM_11_8 * b_in7;
        b_m19 = CRTM_11_3 * b_in8;
        b_m20 = CRTM_11_4 * b_in9;

        b_s13 = b_m12 + b_m16;
        b_s14 = b_m14 - b_m20;
        b_s15 = b_s13 + b_s14;
        b_s16 = b_s15 - b_m18;
        b_s17 = b_m11 + b_m19;
        b_s18 = b_m13 + b_m17;
        b_s19 = b_s17 - b_s18;
        b_s20 = b_in10 - b_m15;
        b_s21 = b_s19 + b_s20;

        // Output point 6: x(5)
        out[out_strides[5]] = b_s21 - b_s16;
        // Output point 20: x(19)
        out[out_strides[19]] = -(b_s21 + b_s16);

        b_m21 = CRTM_11_7 * b_in0;
        b_m22 = CRTM_11_8 * b_in1;
        b_m23 = CRTM_11_1 * b_in2;
        b_m24 = CRTM_11_2 * b_in3;
        b_m25 = CRTM_11_3 * b_in4;
        b_m26 = CRTM_11_4 * b_in5;
        b_m27 = CRTM_11_9 * b_in6;
        b_m28 = CRTM_11_10 * b_in7;
        b_m29 = CRTM_11_5 * b_in8;
        b_m30 = CRTM_11_6 * b_in9;

        b_s22 = b_m22 + b_m24;
        b_s23 = b_m30 - b_m26;
        b_s24 = b_s22 + b_s23;
        b_s25 = b_s24 - b_m28;
        b_s26 = b_m21 + b_m29;
        b_s27 = b_m27 - b_m23;
        b_s28 = b_s26 + b_s27;
        b_s29 = b_m25 + b_in10;
        b_s30 = b_s28 - b_s29;

        // Output point 8: x(7)
        out[out_strides[7]] = b_s30 - b_s25;
        // Output point 18: x(17)
        out[out_strides[17]] = -(b_s30 + b_s25);

        b_m31 = CRTM_11_3 * b_in0;
        b_m32 = CRTM_11_4 * b_in1;
        b_m33 = CRTM_11_9 * b_in2;
        b_m34 = CRTM_11_10 * b_in3;
        b_m35 = CRTM_11_1 * b_in4;
        b_m36 = CRTM_11_2 * b_in5;
        b_m37 = CRTM_11_5 * b_in6;
        b_m38 = CRTM_11_6 * b_in7;
        b_m39 = CRTM_11_7 * b_in8;
        b_m40 = CRTM_11_8 * b_in9;

        b_s31 = b_m32 - b_m34;
        b_s32 = b_m38 - b_m40;
        b_s33 = b_s31 + b_s32;
        b_s34 = b_s33 - b_m36;
        b_s35 = b_m31 + b_m35;
        b_s36 = b_m37 + b_m39;
        b_s37 = b_s35 - b_s36;
        b_s38 = b_in10 - b_m33;
        b_s39 = b_s37 + b_s38;

        // Output point 10: x(9)
        out[out_strides[9]] = b_s39 - b_s34;
        // Output point 16: x(15)
        out[out_strides[15]] = -(b_s39 + b_s34);

        b_m41 = CRTM_11_5 * b_in0;
        b_m42 = CRTM_11_6 * b_in1;
        b_m43 = CRTM_11_3 * b_in2;
        b_m44 = CRTM_11_4 * b_in3;
        b_m45 = CRTM_11_7 * b_in4;
        b_m46 = CRTM_11_8 * b_in5;
        b_m47 = CRTM_11_1 * b_in6;
        b_m48 = CRTM_11_2 * b_in7;
        b_m49 = CRTM_11_9 * b_in8;
        b_m50 = CRTM_11_10 * b_in9;

        b_s40 = b_m42 - b_m44;
        b_s41 = b_m46 + b_m50;
        b_s42 = b_s40 + b_s41;
        b_s43 = b_s42 - b_m48;
        b_s44 = b_m41 + b_m45;
        b_s45 = b_m43 + b_m47;
        b_s46 = b_s44 - b_s45;
        b_s47 = b_m49 - b_in10;
        b_s48 = b_s46 + b_s47;

        // Output point 12: x(11)
        out[out_strides[11]] = b_s48 - b_s43;
        // Output point 14: x(13)
        out[out_strides[13]] = -(b_s48 + b_s43);

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft11c_fp64_fwd(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                        FFTZ_VOID *out_real,
                                        FFTZ_VOID *out_imag, FFTZ_INTP n,
                                        aoclfftz_strides_t *strides,
                                        FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");

    const FFTZ_DOUBLE CRTM_11_1 =
        0.841253532831181168861811648919367717513292498;
    const FFTZ_DOUBLE CRTM_11_2 =
        0.540640817455597582107635954318691695431770608;
    const FFTZ_DOUBLE CRTM_11_3 =
        0.415415013001886425529274149229623203524004910;
    const FFTZ_DOUBLE CRTM_11_4 =
        0.909631995354518371411715383079028460060241051;
    const FFTZ_DOUBLE CRTM_11_5 =
        0.142314838273285140443792668616369668791051361;
    const FFTZ_DOUBLE CRTM_11_6 =
        0.989821441880932732376092037776718787376519372;
    const FFTZ_DOUBLE CRTM_11_7 =
        0.654860733945285064056925072466293553183791199;
    const FFTZ_DOUBLE CRTM_11_8 =
        0.755749574354258283774035843972344420179717445;
    const FFTZ_DOUBLE CRTM_11_9 =
        0.959492973614497389890368057066327699062454848;
    const FFTZ_DOUBLE CRTM_11_10 =
        0.281732556841429697711417915346616899035777899;

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
        /* Standard DFT */
        FFTZ_DOUBLE a_in0, a_in1, a_in2, a_in3, a_in4, a_in5, a_in6, a_in7,
            a_in8, a_in9, a_in10;
        FFTZ_DOUBLE a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_s9,
               a_s10, a_s11, a_s12, a_s13, a_s14, a_s15, a_s16, a_s17, a_s18,
               a_s19, a_s20, a_s21, a_s22, a_s23, a_s24, a_s25, a_s26, a_s27,
               a_s28, a_s29, a_s30, a_s31, a_s32, a_s33, a_s34, a_s35, a_s36,
               a_s37, a_s38, a_s39, a_s40, a_s41, a_s42, a_s43, a_s44, a_s45,
               a_s46, a_s47, a_s48;
        FFTZ_DOUBLE a_m0, a_m1, a_m2, a_m3, a_m4, a_m5, a_m6, a_m7, a_m8, a_m9,
               a_m10, a_m11, a_m12, a_m13, a_m14, a_m15, a_m16, a_m17, a_m18,
               a_m19, a_m20, a_m21, a_m22, a_m23, a_m24, a_m25, a_m26, a_m27,
               a_m28, a_m29, a_m30, a_m31, a_m32, a_m33, a_m34, a_m35, a_m36,
               a_m37, a_m38, a_m39, a_m40, a_m41, a_m42, a_m43, a_m44, a_m45,
               a_m46, a_m47, a_m48, a_m49;

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

        a_s0 = a_in1 + a_in10;
        a_s1 = a_in2 + a_in9;
        a_s2 = a_in3 + a_in8;
        a_s3 = a_in4 + a_in7;
        a_s4 = a_in5 + a_in6;
        a_s5 = a_in1 - a_in10;
        a_s6 = a_in2 - a_in9;
        a_s7 = a_in3 - a_in8;
        a_s8 = a_in4 - a_in7;
        a_s9 = a_in5 - a_in6;

        a_s10 = a_s0 + a_s1;
        a_s11 = a_s2 + a_s3;
        a_s12 = a_s4 + a_in0;
        a_s13 = a_s10 + a_s11;

        // Output point 1: X(0)
        *out = a_s12 + a_s13;

        a_m0 = CRTM_11_1 * a_s0;
        a_m1 = CRTM_11_3 * a_s1;
        a_m2 = CRTM_11_5 * a_s2;
        a_m3 = CRTM_11_7 * a_s3;
        a_m4 = CRTM_11_9 * a_s4;

        a_s14 = a_m0 + a_m1;
        a_s15 = a_m2 + a_m3;
        a_s16 = a_in0 - a_m4;
        a_s17 = a_s14 - a_s15;

        // Output point 4: X(3)
        out[out_strides[3]] = a_s16 + a_s17;

        a_m5 = CRTM_11_2 * a_s5;
        a_m6 = CRTM_11_4 * a_s6;
        a_m7 = CRTM_11_6 * a_s7;
        a_m8 = CRTM_11_8 * a_s8;
        a_m9 = CRTM_11_10 * a_s9;

        a_s18 = a_m5 + a_m6;
        a_s19 = a_m7 + a_m8;
        a_s20 = a_s19 + a_m9;

        // Output point 5: X(4)
        out[out_strides[4]] = -(a_s18 + a_s20);

        a_m10 = CRTM_11_1 * a_s4;
        a_m11 = CRTM_11_3 * a_s0;
        a_m12 = CRTM_11_5 * a_s3;
        a_m13 = CRTM_11_7 * a_s1;
        a_m14 = CRTM_11_9 * a_s2;

        a_s21 = a_m10 + a_m11;
        a_s22 = a_m12 + a_m13;
        a_s23 = a_in0 - a_m14;
        a_s24 = a_s21 - a_s22;

        // Output point 8: X(7)
        out[out_strides[7]] = a_s23 + a_s24;

        a_m15 = CRTM_11_2 * a_s9;
        a_m16 = CRTM_11_4 * a_s5;
        a_m17 = CRTM_11_6 * a_s8;
        a_m18 = CRTM_11_8 * a_s6;
        a_m19 = CRTM_11_10 * a_s7;

        a_s25 = a_m15 - a_m16;
        a_s26 = a_m17 - a_m18;
        a_s27 = a_s26 + a_m19;

        // Output point 9: X(8)
        out[out_strides[8]] = a_s25 + a_s27;

        a_m20 = CRTM_11_1 * a_s3;
        a_m21 = CRTM_11_3 * a_s2;
        a_m22 = CRTM_11_5 * a_s0;
        a_m23 = CRTM_11_7 * a_s4;
        a_m24 = CRTM_11_9 * a_s1;

        a_s28 = a_m20 + a_m21;
        a_s29 = a_m22 + a_m23;
        a_s30 = a_in0 - a_m24;
        a_s31 = a_s30 - a_s29;

        // Output point 12: X(11)
        out[out_strides[11]] = a_s28 + a_s31;

        a_m25 = CRTM_11_2 * a_s8;
        a_m26 = CRTM_11_4 * a_s7;
        a_m27 = CRTM_11_6 * a_s5;
        a_m28 = CRTM_11_8 * a_s9;
        a_m29 = CRTM_11_10 * a_s6;

        a_s32 = a_m26 - a_m25;
        a_s33 = a_m27 + a_m28;
        a_s34 = a_s32 - a_s33;

        // Output point 13: X(12)
        out[out_strides[12]] = a_s34 + a_m29;

        a_m30 = CRTM_11_1 * a_s2;
        a_m31 = CRTM_11_3 * a_s4;
        a_m32 = CRTM_11_5 * a_s1;
        a_m33 = CRTM_11_7 * a_s0;
        a_m34 = CRTM_11_9 * a_s3;

        a_s35 = a_m30 + a_m31;
        a_s36 = a_m32 + a_m33;
        a_s37 = a_in0 - a_m34;
        a_s38 = a_s35 - a_s36;

        // Output point 16: X(15)
        out[out_strides[15]] = a_s38 + a_s37;

        a_m35 = CRTM_11_2 * a_s7;
        a_m36 = CRTM_11_4 * a_s9;
        a_m37 = CRTM_11_6 * a_s6;
        a_m38 = CRTM_11_8 * a_s5;
        a_m39 = CRTM_11_10 * a_s8;

        a_s39 = a_m36 - a_m35;
        a_s40 = a_m37 - a_m38;
        a_s41 = a_s40 - a_m39;

        // Output point 17: X(16)
        out[out_strides[16]] = a_s39 + a_s41;

        a_m40 = CRTM_11_1 * a_s1;
        a_m41 = CRTM_11_3 * a_s3;
        a_m42 = CRTM_11_5 * a_s4;
        a_m43 = CRTM_11_7 * a_s2;
        a_m44 = CRTM_11_9 * a_s0;

        a_s42 = a_m40 + a_m41;
        a_s43 = a_m42 + a_m43;
        a_s44 = a_in0 - a_m44;
        a_s45 = a_s42 - a_s43;

        // Output point 20: X(19)
        out[out_strides[19]] = a_s44 + a_s45;

        a_m45 = CRTM_11_2 * a_s6;
        a_m46 = CRTM_11_4 * a_s8;
        a_m47 = CRTM_11_6 * a_s9;
        a_m48 = CRTM_11_8 * a_s7;
        a_m49 = CRTM_11_10 * a_s5;

        a_s46 = a_m45 + a_m46;
        a_s47 = a_m47 + a_m48;
        a_s48 = a_s46 - a_s47;

        // Output point 21: X(20)
        out[out_strides[20]] = a_s48 - a_m49;

        /* Shifted DFT */
        FFTZ_DOUBLE b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7,
            b_in8, b_in9, b_in10;
        FFTZ_DOUBLE b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9,
               b_s10, b_s11, b_s12, b_s13, b_s14, b_s15, b_s16, b_s17, b_s18,
               b_s19, b_s20, b_s21, b_s22, b_s23, b_s24, b_s25, b_s26, b_s27,
               b_s28, b_s29, b_s30, b_s31, b_s32, b_s33, b_s34, b_s35, b_s36,
               b_s37, b_s38, b_s39, b_s40, b_s41, b_s42, b_s43, b_s44, b_s45,
               b_s46, b_s47, b_s48, b_s49, b_s50, b_s51, b_s52, b_s53, b_s54;
        FFTZ_DOUBLE b_m0, b_m1, b_m2, b_m3, b_m4, b_m5, b_m6, b_m7, b_m8, b_m9,
               b_m10, b_m11, b_m12, b_m13, b_m14, b_m15, b_m16, b_m17, b_m18,
               b_m19, b_m20, b_m21, b_m22, b_m23, b_m24, b_m25, b_m26, b_m27,
               b_m28, b_m29, b_m30, b_m31, b_m32, b_m33, b_m34, b_m35, b_m36,
               b_m37, b_m38, b_m39, b_m40, b_m41, b_m42, b_m43, b_m44, b_m45,
               b_m46, b_m47, b_m48, b_m49;

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

        b_s0 = b_in1 + b_in10;
        b_s1 = b_in2 + b_in9;
        b_s2 = b_in3 + b_in8;
        b_s3 = b_in4 + b_in7;
        b_s4 = b_in5 + b_in6;
        b_s5 = b_in1 - b_in10;
        b_s6 = b_in2 - b_in9;
        b_s7 = b_in3 - b_in8;
        b_s8 = b_in4 - b_in7;
        b_s9 = b_in5 - b_in6;

        b_m0 = CRTM_11_9 * b_s5;
        b_m1 = CRTM_11_1 * b_s6;
        b_s10 = b_m0 + b_m1;
        b_m2 = CRTM_11_7 * b_s7;
        b_s11 = b_s10 + b_m2;
        b_m3 = CRTM_11_3 * b_s8;
        b_s12 = b_s11 + b_m3;
        b_m4 = CRTM_11_5 * b_s9;
        b_s13 = b_s12 + b_m4;

        // Output point 2: X(1)
        out[out_strides[1]] = b_in0 + b_s13;

        b_m5 = CRTM_11_10 * b_s0;
        b_m6 = CRTM_11_2 * b_s1;
        b_s14 = b_m5 + b_m6;
        b_m7 = CRTM_11_8 * b_s2;
        b_s15 = b_s14 + b_m7;
        b_m8 = CRTM_11_4 * b_s3;
        b_s16 = b_s15 + b_m8;
        b_m9 = CRTM_11_6 * b_s4;
        b_s17 = b_s16 + b_m9;

        // Output point 3: X(2)
        out[out_strides[2]] = -b_s17;

        b_m10 = CRTM_11_7 * b_s5;
        b_m11 = CRTM_11_5 * b_s6;
        b_s18 = b_m10 - b_m11;
        b_m12 = CRTM_11_1 * b_s7;
        b_s19 = b_s18 - b_m12;
        b_m13 = CRTM_11_9 * b_s8;
        b_s20 = b_s19 - b_m13;
        b_m14 = CRTM_11_3 * b_s9;
        b_s21 = b_s20 - b_m14;

        // Output point 6: X(5)
        out[out_strides[5]] = b_in0 + b_s21;

        b_m15 = CRTM_11_8 * b_s0;
        b_m16 = CRTM_11_6 * b_s1;
        b_s22 = b_m15 + b_m16;
        b_m17 = CRTM_11_2 * b_s2;
        b_s23 = b_s22 + b_m17;
        b_m18 = CRTM_11_10 * b_s3;
        b_s24 = b_s23 - b_m18;
        b_m19 = CRTM_11_4 * b_s4;
        b_s25 = b_s24 - b_m19;

        // Output point 7: X(6)
        out[out_strides[6]] = -b_s25;

        b_m20 = CRTM_11_5 * b_s5;
        b_m21 = CRTM_11_9 * b_s6;
        b_s26 = b_m20 - b_m21;
        b_m22 = CRTM_11_3 * b_s7;
        b_s27 = b_s26 - b_m22;
        b_m23 = CRTM_11_1 * b_s8;
        b_s28 = b_s27 + b_m23;
        b_m24 = CRTM_11_7 * b_s9;
        b_s29 = b_s28 + b_m24;

        // Output point 10: X(9)
        out[out_strides[9]] = b_in0 + b_s29;

        b_m25 = CRTM_11_6 * b_s0;
        b_m26 = CRTM_11_10 * b_s1;
        b_s30 = b_m25 + b_m26;
        b_m27 = CRTM_11_4 * b_s2;
        b_s31 = b_s30 - b_m27;
        b_m28 = CRTM_11_2 * b_s3;
        b_s32 = b_s31 - b_m28;
        b_m29 = CRTM_11_8 * b_s4;
        b_s33 = b_s32 + b_m29;

        // Output point 11: X(10)
        out[out_strides[10]] = -b_s33;

        b_m30 = CRTM_11_3 * b_s5;
        b_m31 = CRTM_11_7 * b_s6;
        b_s34 = b_m30 + b_m31;
        b_m32 = CRTM_11_9 * b_s7;
        b_s35 = b_m32 - b_s34;
        b_m33 = CRTM_11_5 * b_s8;
        b_s36 = b_s35 - b_m33;
        b_m34 = CRTM_11_1 * b_s9;
        b_s37 = b_s36 - b_m34;

        // Output point 14: X(13)
        out[out_strides[13]] = b_in0 + b_s37;

        b_m35 = CRTM_11_4 * b_s0;
        b_m36 = CRTM_11_8 * b_s1;
        b_s38 = b_m35 - b_m36;
        b_m37 = CRTM_11_10 * b_s2;
        b_s39 = b_s38 - b_m37;
        b_m38 = CRTM_11_6 * b_s3;
        b_s40 = b_s39 + b_m38;
        b_m39 = CRTM_11_2 * b_s4;
        b_s41 = b_s40 - b_m39;

        // Output point 15: X(14)
        out[out_strides[14]] = -b_s41;

        b_m40 = CRTM_11_1 * b_s5;
        b_m41 = CRTM_11_3 * b_s6;
        b_s42 = b_m41 - b_m40;
        b_m42 = CRTM_11_5 * b_s7;
        b_s43 = b_s42 + b_m42;
        b_m43 = CRTM_11_7 * b_s8;
        b_s44 = b_s43 - b_m43;
        b_m44 = CRTM_11_9 * b_s9;
        b_s45 = b_s44 + b_m44;

        // Output point 18: X(17)
        out[out_strides[17]] = b_in0 + b_s45;

        b_m45 = CRTM_11_2 * b_s0;
        b_m46 = CRTM_11_4 * b_s1;
        b_s46 = b_m45 - b_m46;
        b_m47 = CRTM_11_6 * b_s2;
        b_s47 = b_s46 + b_m47;
        b_m48 = CRTM_11_8 * b_s3;
        b_s48 = b_s47 - b_m48;
        b_m49 = CRTM_11_10 * b_s4;
        b_s49 = b_s48 + b_m49;

        // Output point 19: X(18)
        out[out_strides[18]] = -b_s49;

        b_s50 = b_in0 - b_s5;
        b_s51 = b_s50 + b_s6;
        b_s52 = b_s51 - b_s7;
        b_s53 = b_s52 + b_s8;
        b_s54 = b_s53 - b_s9;

        // Output point 22: X(21)
        out[out_strides[21]] = b_s54;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft11c_fp64_bwd(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                        FFTZ_VOID *out_real,
                                        FFTZ_VOID *out_imag, FFTZ_INTP n,
                                        aoclfftz_strides_t *strides,
                                        FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");

    const FFTZ_DOUBLE CRTM_11_1 =
        1.682507065662362337723623297838735435026584997;
    const FFTZ_DOUBLE CRTM_11_2 =
        1.081281634911195164215271908637383390863541216;
    const FFTZ_DOUBLE CRTM_11_3 =
        0.830830026003772851058548298459246407048009821;
    const FFTZ_DOUBLE CRTM_11_4 =
        1.819263990709036742823430766158056920120482102;
    const FFTZ_DOUBLE CRTM_11_5 =
        0.284629676546570280887585337232739337582102722;
    const FFTZ_DOUBLE CRTM_11_6 =
        1.979642883761865464752184075553437574753038744;
    const FFTZ_DOUBLE CRTM_11_7 =
        1.309721467890570128113850144932587106367582399;
    const FFTZ_DOUBLE CRTM_11_8 =
        1.511499148708516567548071687944688840359434890;
    const FFTZ_DOUBLE CRTM_11_9 =
        1.918985947228994779780736114132655398124909697;
    const FFTZ_DOUBLE CRTM_11_10 =
        0.563465113682859395422835830693233798071555798;
    const FFTZ_DOUBLE CRTM_11_11 =
        2.000000000000000000000000000000000000000000000;

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
            a_in8, a_in9, a_in10;
        FFTZ_DOUBLE a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_s9,
               a_s10, a_s11, a_s12, a_s13, a_s14, a_s15, a_s16, a_s17, a_s18,
               a_s19, a_s20, a_s21, a_s22, a_s23, a_s24, a_s25, a_s26, a_s27,
               a_s28, a_s29, a_s30, a_s31, a_s32, a_s33, a_s34, a_s35, a_s36,
               a_s37, a_s38, a_s39, a_s40, a_s41, a_s42, a_s43, a_s44, a_s45,
               a_s46, a_s47, a_s48;
        FFTZ_DOUBLE a_m0, a_m1, a_m2, a_m3, a_m4, a_m5, a_m6, a_m7, a_m8, a_m9,
               a_m10, a_m11, a_m12, a_m13, a_m14, a_m15, a_m16, a_m17, a_m18,
               a_m19, a_m20, a_m21, a_m22, a_m23, a_m24, a_m25, a_m26, a_m27,
               a_m28, a_m29, a_m30, a_m31, a_m32, a_m33, a_m34, a_m35, a_m36,
               a_m37, a_m38, a_m39, a_m40, a_m41, a_m42, a_m43, a_m44, a_m45,
               a_m46, a_m47, a_m48, a_m49, a_m50;

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
        // Input point 20: X(19)
        a_in9 = in[in_strides[19]];
        // Input point 21: X(20)
        a_in10 = in[in_strides[20]];

        a_s0 = a_in1 + a_in3;
        a_s1 = a_in5 + a_in7;
        a_s2 = a_s0 + a_s1;
        a_s3 = a_s2 + a_in9;
        a_m0 = CRTM_11_11 * a_s3;

        // Output point 1: x(0)
        *out = a_in0 + a_m0;

        a_m1 = CRTM_11_1 * a_in1;
        a_m2 = CRTM_11_2 * a_in2;
        a_m3 = CRTM_11_3 * a_in3;
        a_m4 = CRTM_11_4 * a_in4;
        a_m5 = CRTM_11_5 * a_in5;
        a_m6 = CRTM_11_6 * a_in6;
        a_m7 = CRTM_11_7 * a_in7;
        a_m8 = CRTM_11_8 * a_in8;
        a_m9 = CRTM_11_9 * a_in9;
        a_m10 = CRTM_11_10 * a_in10;

        a_s4 = a_m1 + a_m3;
        a_s5 = a_s4 + a_in0;
        a_s6 = a_m5 + a_m7;
        a_s7 = a_s6 + a_m9;
        a_s8 = a_s5 - a_s7;

        a_s9 = a_m2 + a_m4;
        a_s10 = a_m6 + a_m8;
        a_s11 = a_s9 + a_m10;
        a_s12 = a_s10 + a_s11;

        // Output point 3: x(2)
        out[out_strides[2]] = a_s8 - a_s12;
        // Output point 21: x(20)
        out[out_strides[20]] = a_s8 + a_s12;

        a_m11 = CRTM_11_1 * a_in9;
        a_m12 = CRTM_11_2 * a_in10;
        a_m13 = CRTM_11_3 * a_in1;
        a_m14 = CRTM_11_4 * a_in2;
        a_m15 = CRTM_11_5 * a_in7;
        a_m16 = CRTM_11_6 * a_in8;
        a_m17 = CRTM_11_7 * a_in3;
        a_m18 = CRTM_11_8 * a_in4;
        a_m19 = CRTM_11_9 * a_in5;
        a_m20 = CRTM_11_10 * a_in6;

        a_s13 = a_m11 + a_m13;
        a_s14 = a_s13 + a_in0;
        a_s15 = a_m15 + a_m17;
        a_s16 = a_s15 + a_m19;
        a_s17 = a_s14 - a_s16;

        a_s18 = a_m12 - a_m14;
        a_s19 = a_m16 - a_m18;
        a_s20 = a_s19 + a_m20;
        a_s21 = a_s18 + a_s20;

        // Output point 5: x(4)
        out[out_strides[4]] = a_s17 + a_s21;
        // Output point 19: x(18)
        out[out_strides[18]] = a_s17 - a_s21;

        a_m21 = CRTM_11_1 * a_in7;
        a_m22 = CRTM_11_2 * a_in8;
        a_m23 = CRTM_11_3 * a_in5;
        a_m24 = CRTM_11_4 * a_in6;
        a_m25 = CRTM_11_5 * a_in1;
        a_m26 = CRTM_11_6 * a_in2;
        a_m27 = CRTM_11_7 * a_in9;
        a_m28 = CRTM_11_8 * a_in10;
        a_m29 = CRTM_11_9 * a_in3;
        a_m30 = CRTM_11_10 * a_in4;

        a_s22 = a_m21 + a_m23;
        a_s23 = a_s22 + a_in0;
        a_s24 = a_m25 + a_m27;
        a_s25 = a_s24 + a_m29;
        a_s26 = a_s23 - a_s25;

        a_s27 = a_m22 - a_m24;
        a_s28 = a_m26 + a_m28;
        a_s29 = a_s28 - a_m30;
        a_s30 = a_s27 + a_s29;

        // Output point 7: x(6)
        out[out_strides[6]] = a_s26 - a_s30;
        // Output point 17: x(16)
        out[out_strides[16]] = a_s26 + a_s30;

        a_m31 = CRTM_11_1 * a_in5;
        a_m32 = CRTM_11_2 * a_in6;
        a_m33 = CRTM_11_3 * a_in9;
        a_m34 = CRTM_11_4 * a_in10;
        a_m35 = CRTM_11_5 * a_in3;
        a_m36 = CRTM_11_6 * a_in4;
        a_m37 = CRTM_11_7 * a_in1;
        a_m38 = CRTM_11_8 * a_in2;
        a_m39 = CRTM_11_9 * a_in7;
        a_m40 = CRTM_11_10 * a_in8;

        a_s31 = a_m31 + a_m33;
        a_s32 = a_s31 + a_in0;
        a_s33 = a_m35 + a_m37;
        a_s34 = a_s33 + a_m39;
        a_s35 = a_s32 - a_s34;

        a_s36 = a_m32 - a_m34;
        a_s37 = a_m38 - a_m36;
        a_s38 = a_s37 + a_m40;
        a_s39 = a_s36 + a_s38;

        // Output point 9: x(8)
        out[out_strides[8]] = a_s35 - a_s39;
        // Output point 15: x(14)
        out[out_strides[14]] = a_s35 + a_s39;

        a_m41 = CRTM_11_1 * a_in3;
        a_m42 = CRTM_11_2 * a_in4;
        a_m43 = CRTM_11_3 * a_in7;
        a_m44 = CRTM_11_4 * a_in8;
        a_m45 = CRTM_11_5 * a_in9;
        a_m46 = CRTM_11_6 * a_in10;
        a_m47 = CRTM_11_7 * a_in5;
        a_m48 = CRTM_11_8 * a_in6;
        a_m49 = CRTM_11_9 * a_in1;
        a_m50 = CRTM_11_10 * a_in2;

        a_s40 = a_m41 + a_m43;
        a_s41 = a_s40 + a_in0;
        a_s42 = a_m45 + a_m47;
        a_s43 = a_s42 + a_m49;
        a_s44 = a_s41 - a_s43;

        a_s45 = a_m42 + a_m44;
        a_s46 = a_m46 + a_m48;
        a_s47 = a_s45 - a_s46;
        a_s48 = a_s47 - a_m50;

        // Output point 11: x(10)
        out[out_strides[10]] = a_s44 + a_s48;
        // Output point 13: x(12)
        out[out_strides[12]] = a_s44 - a_s48;

        /* Shifted DFT */
        FFTZ_DOUBLE b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7,
            b_in8, b_in9, b_in10;
        FFTZ_DOUBLE b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9,
               b_s10, b_s11, b_s12, b_s13, b_s14, b_s15, b_s16, b_s17, b_s18,
               b_s19, b_s20, b_s21, b_s22, b_s23, b_s24, b_s25, b_s26, b_s27,
               b_s28, b_s29, b_s30, b_s31, b_s32, b_s33, b_s34, b_s35, b_s36,
               b_s37, b_s38, b_s39, b_s40, b_s41, b_s42, b_s43, b_s44, b_s45,
               b_s46, b_s47, b_s48;
        FFTZ_DOUBLE b_m0, b_m1, b_m2, b_m3, b_m4, b_m5, b_m6, b_m7, b_m8, b_m9,
               b_m10, b_m11, b_m12, b_m13, b_m14, b_m15, b_m16, b_m17, b_m18,
               b_m19, b_m20, b_m21, b_m22, b_m23, b_m24, b_m25, b_m26, b_m27,
               b_m28, b_m29, b_m30, b_m31, b_m32, b_m33, b_m34, b_m35, b_m36,
               b_m37, b_m38, b_m39, b_m40, b_m41, b_m42, b_m43, b_m44, b_m45,
               b_m46, b_m47, b_m48, b_m49, b_m50;

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
        // Input point 19: X(18)
        b_in9 = in[in_strides[18]];
        // Input point 22: X(21)
        b_in10 = in[in_strides[21]];

        b_s0 = b_in0 + b_in2;
        b_s1 = b_in4 + b_in6;
        b_s2 = b_s0 + b_s1;
        b_s3 = b_s2 + b_in8;
        b_m0 = CRTM_11_11 * b_s3;

        // Output point 2: x(1)
        out[out_strides[1]] = b_m0 + b_in10;

        b_m1 = CRTM_11_9 * b_in0;
        b_m2 = CRTM_11_10 * b_in1;
        b_m3 = CRTM_11_1 * b_in8;
        b_m4 = CRTM_11_2 * b_in9;
        b_m5 = CRTM_11_7 * b_in2;
        b_m6 = CRTM_11_8 * b_in3;
        b_m7 = CRTM_11_3 * b_in6;
        b_m8 = CRTM_11_4 * b_in7;
        b_m9 = CRTM_11_5 * b_in4;
        b_m10 = CRTM_11_6 * b_in5;

        b_s4 = b_m2 + b_m4;
        b_s5 = b_m6 + b_m8;
        b_s6 = b_s4 + b_s5;
        b_s7 = b_s6 + b_m10;
        b_s8 = b_m1 - b_m3;
        b_s9 = b_m5 - b_m7;
        b_s10 = b_s8 + b_s9;
        b_s11 = b_m9 - b_in10;
        b_s12 = b_s10 + b_s11;

        // Output point 4: x(3)
        out[out_strides[3]] = b_s12 - b_s7;
        // Output point 22: x(21)
        out[out_strides[21]] = -(b_s12 + b_s7);

        b_m11 = CRTM_11_1 * b_in0;
        b_m12 = CRTM_11_2 * b_in1;
        b_m13 = CRTM_11_5 * b_in2;
        b_m14 = CRTM_11_6 * b_in3;
        b_m15 = CRTM_11_9 * b_in4;
        b_m16 = CRTM_11_10 * b_in5;
        b_m17 = CRTM_11_7 * b_in6;
        b_m18 = CRTM_11_8 * b_in7;
        b_m19 = CRTM_11_3 * b_in8;
        b_m20 = CRTM_11_4 * b_in9;

        b_s13 = b_m12 + b_m16;
        b_s14 = b_m14 - b_m20;
        b_s15 = b_s13 + b_s14;
        b_s16 = b_s15 - b_m18;
        b_s17 = b_m11 + b_m19;
        b_s18 = b_m13 + b_m17;
        b_s19 = b_s17 - b_s18;
        b_s20 = b_in10 - b_m15;
        b_s21 = b_s19 + b_s20;

        // Output point 6: x(5)
        out[out_strides[5]] = b_s21 - b_s16;
        // Output point 20: x(19)
        out[out_strides[19]] = -(b_s21 + b_s16);

        b_m21 = CRTM_11_7 * b_in0;
        b_m22 = CRTM_11_8 * b_in1;
        b_m23 = CRTM_11_1 * b_in2;
        b_m24 = CRTM_11_2 * b_in3;
        b_m25 = CRTM_11_3 * b_in4;
        b_m26 = CRTM_11_4 * b_in5;
        b_m27 = CRTM_11_9 * b_in6;
        b_m28 = CRTM_11_10 * b_in7;
        b_m29 = CRTM_11_5 * b_in8;
        b_m30 = CRTM_11_6 * b_in9;

        b_s22 = b_m22 + b_m24;
        b_s23 = b_m30 - b_m26;
        b_s24 = b_s22 + b_s23;
        b_s25 = b_s24 - b_m28;
        b_s26 = b_m21 + b_m29;
        b_s27 = b_m27 - b_m23;
        b_s28 = b_s26 + b_s27;
        b_s29 = b_m25 + b_in10;
        b_s30 = b_s28 - b_s29;

        // Output point 8: x(7)
        out[out_strides[7]] = b_s30 - b_s25;
        // Output point 18: x(17)
        out[out_strides[17]] = -(b_s30 + b_s25);

        b_m31 = CRTM_11_3 * b_in0;
        b_m32 = CRTM_11_4 * b_in1;
        b_m33 = CRTM_11_9 * b_in2;
        b_m34 = CRTM_11_10 * b_in3;
        b_m35 = CRTM_11_1 * b_in4;
        b_m36 = CRTM_11_2 * b_in5;
        b_m37 = CRTM_11_5 * b_in6;
        b_m38 = CRTM_11_6 * b_in7;
        b_m39 = CRTM_11_7 * b_in8;
        b_m40 = CRTM_11_8 * b_in9;

        b_s31 = b_m32 - b_m34;
        b_s32 = b_m38 - b_m40;
        b_s33 = b_s31 + b_s32;
        b_s34 = b_s33 - b_m36;
        b_s35 = b_m31 + b_m35;
        b_s36 = b_m37 + b_m39;
        b_s37 = b_s35 - b_s36;
        b_s38 = b_in10 - b_m33;
        b_s39 = b_s37 + b_s38;

        // Output point 10: x(9)
        out[out_strides[9]] = b_s39 - b_s34;
        // Output point 16: x(15)
        out[out_strides[15]] = -(b_s39 + b_s34);

        b_m41 = CRTM_11_5 * b_in0;
        b_m42 = CRTM_11_6 * b_in1;
        b_m43 = CRTM_11_3 * b_in2;
        b_m44 = CRTM_11_4 * b_in3;
        b_m45 = CRTM_11_7 * b_in4;
        b_m46 = CRTM_11_8 * b_in5;
        b_m47 = CRTM_11_1 * b_in6;
        b_m48 = CRTM_11_2 * b_in7;
        b_m49 = CRTM_11_9 * b_in8;
        b_m50 = CRTM_11_10 * b_in9;

        b_s40 = b_m42 - b_m44;
        b_s41 = b_m46 + b_m50;
        b_s42 = b_s40 + b_s41;
        b_s43 = b_s42 - b_m48;
        b_s44 = b_m41 + b_m45;
        b_s45 = b_m43 + b_m47;
        b_s46 = b_s44 - b_s45;
        b_s47 = b_m49 - b_in10;
        b_s48 = b_s46 + b_s47;

        // Output point 12: x(11)
        out[out_strides[11]] = b_s48 - b_s43;
        // Output point 14: x(13)
        out[out_strides[13]] = -(b_s48 + b_s43);

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hcf_rfft11c(FFTZ_UINT8 precision, FFTZ_UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hcf_rfft11c_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft11c_fp64_fwd;
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
            return r2hcf_rfft11c_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft11c_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
