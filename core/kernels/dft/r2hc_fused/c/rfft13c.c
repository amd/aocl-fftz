// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft13c.c
 *
 *  @brief Radix-13 r2hc_fused Real-FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-13 real-to-halfcomplex fused of two
 *  different implementations (Standard DFT and Shifted DFT that differs in DFT
 *  weight matrix) using scalar operations for single-precision
 *  and double-precision inputs.
 *
 *  @author Amrin Fathima
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] = {
                                                   {{0, 106, 166, 26, 0, 0},
                                                    {0, 107, 167, 26, 0, 0}},
                                                   {{0, 106, 166, 26, 0, 0},
                                                    {0, 107, 167, 26, 0, 0}}};

ops_cycles_t get_ops_cnt_r2hcf_rfft13c(FFTZ_UINT8 precision,
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

static FFTZ_VOID r2hcf_rfft13c_fp32_fwd(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                        FFTZ_VOID *out_real,
                                        FFTZ_VOID *out_imag, FFTZ_INTP n,
                                        aoclfftz_strides_t *strides,
                                        FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_13_1 =
        1.73205080756887719317660412343684583902359008789060f;
    const FFTZ_FLOAT CRTM_13_2 =
        0.38739058546761714712838513245787843298017865366345f;
    const FFTZ_FLOAT CRTM_13_3 =
        0.13298312460741867056777367279491668244720145803175f;
    const FFTZ_FLOAT CRTM_13_4 =
        0.11385447905579067614456922127652058891458054833493f;
    const FFTZ_FLOAT CRTM_13_5 =
        0.25176851643188325619942540855595308576560112726050f;
    const FFTZ_FLOAT CRTM_13_6 =
        0.86602540378443864676372317075293618347140262700000f;
    const FFTZ_FLOAT CRTM_13_7 =
        0.50000000000000000000000000000000000000000000000000f;
    const FFTZ_FLOAT CRTM_13_8 =
        2.00000000000000000000000000000000000000000000000000f;
    const FFTZ_FLOAT R13_DGC_1 =
        0.08333333333333341693177688718872429763576971755296f;
    const FFTZ_FLOAT R13_DGC_2 =
        0.25624767158293663769405689230781736686616059890980f;
    const FFTZ_FLOAT R13_DGC_3 =
        0.15689139105158457352993666120753768234094124525375f;
    const FFTZ_FLOAT R13_DGC_4 =
        0.25826039031174479484752691274654956353060547073611f;
    const FFTZ_FLOAT R13_DGC_5 =
        0.26596624921483734113554734558983336489440291606351f;
    const FFTZ_FLOAT R13_DGC_6 =
        0.57514072947400312136838554745545338846100160800000f;
    const FFTZ_FLOAT R13_DGC_7 =
        0.17413860115213590500566079492926474261696467600000f;
    const FFTZ_FLOAT R13_DGC_8 =
        0.07590298603719379320004091411468632874055721777845f;
    const FFTZ_FLOAT R13_DGC_9 =
        0.50353703286376651239885081711190617153120225452101f;
    const FFTZ_FLOAT R13_DGC_10 =
        0.3002386359663325979287571057309917466781880617299f;
    const FFTZ_FLOAT R13_DGC_11 =
        0.0115991056057681999237624507209650002461566663358f;
    const FFTZ_FLOAT R13_DGC_12 =
        0.3004626062886657229721584869768537486323029264720f;
    const FFTZ_FLOAT R13_DFT_C1 =
        0.97094181742605202715698227629378922724986510573900f;
    const FFTZ_FLOAT R13_DFT_C2 =
        0.88545602565320989590037552201509887860549841634750f;
    const FFTZ_FLOAT R13_DFT_C3 =
        0.74851074817110109863463059970135138384645159017580f;
    const FFTZ_FLOAT R13_DFT_C4 =
        0.56806474673115580251180755912751662453349255245350f;
    const FFTZ_FLOAT R13_DFT_C5 =
        0.35460488704253562596963789260001847431635543211380f;
    const FFTZ_FLOAT R13_DFT_C6 =
        0.12053668025532305334906768745254358227368115922760f;
    const FFTZ_FLOAT R13_DFT_S1 =
        0.23931566428755776714875372626021189520317302273830f;
    const FFTZ_FLOAT R13_DFT_S2 =
        0.46472317204376854565601533513310477755773586533250f;
    const FFTZ_FLOAT R13_DFT_S3 =
        0.66312265824079520237678549266676627952476410704410f;
    const FFTZ_FLOAT R13_DFT_S4 =
        0.82298386589365639457961742343938199065506769308760f;
    const FFTZ_FLOAT R13_DFT_S5 =
        0.93501624268541482343978459983783072905051746957840f;
    const FFTZ_FLOAT R13_DFT_S6 =
        0.99270887409805399280075164949252017934367563297020f;

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
        // Standard DFT
        FFTZ_FLOAT a_in0, a_in1, a_in2, a_in3, a_in4, a_in5, a_in6, a_in7,
                   a_in8, a_in9, a_in10, a_in11, a_in12;
        FFTZ_FLOAT a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_s9,
                   a_s10, a_s11, a_s12, a_s13, a_s14, a_s15, a_s16, a_s17,
                   a_s18, a_s19, a_s20, a_s21, a_s22, a_s23, a_s24, a_s25,
                   a_s26, a_s27, a_s28, a_s29, a_s30, a_s31, a_s32, a_s33,
                   a_s34, a_s35, a_s36, a_s37, a_s38, a_s39, a_s40, a_s41,
                   a_s42, a_s43, a_s44, a_s45, a_s46, a_s47, a_s48, a_s49,
                   a_s50, a_s51, a_s52, a_s53, a_s54, a_s55, a_s56, a_s57,
                   a_s58, a_s59, a_s60, a_s61, a_s62, a_s63, a_s64;
        FFTZ_FLOAT a_m0, a_m1, a_m2, a_m3, a_m4, a_m5, a_m6, a_m7, a_m8, a_m9,
                   a_m10, a_m11, a_m12, a_m13, a_m14, a_m15, a_m16, a_m17,
                   a_m18, a_m19, a_m20, a_m21, a_m22, a_m23, a_m24, a_m25,
                   a_m26, a_m27, a_m28, a_m29, a_m30, a_m31, a_m32, a_m33;

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

        a_s0 = a_in2 + a_in7;
        a_s1 = a_in7 - a_in2;
        a_s2 = a_in6 + a_in11;
        a_s3 = a_in11 - a_in6;
        a_s4 = a_s0 + a_s2;
        a_s5 = a_s0 - a_s2;
        a_m0 = CRTM_13_6 * a_s5;
        a_s6 = a_s1 + a_s3;
        a_s7 = a_s1 - a_s3;
        a_s8 = a_in4 + a_in10;
        a_s9 = a_in10 - a_in4;
        a_s10 = a_in3 + a_in9;
        a_s11 = a_in9 - a_in3;
        a_s12 = a_s8 + a_s10;
        a_s13 = a_s8 - a_s10;
        a_s14 = a_s9 - a_s11;
        a_s15 = a_s9 + a_s11;
        a_m1 = CRTM_13_6 * a_s15;
        a_m2 = CRTM_13_7 * a_s13;
        a_s16 = a_s4 + a_s12;
        a_s17 = a_s4 - a_s12;
        a_s18 = a_in8 + a_in5;
        a_s19 = a_in5 - a_in8;
        a_m3 = CRTM_13_7 * a_s6;
        a_s42 = a_m3 + a_s19;
        a_s20 = a_in1 - a_in12;
        a_s21 = a_in1 + a_in12;
        a_s36 = a_s20 + a_m2;
        a_s22 = a_s20 - a_s13;
        a_s23 = a_s6 - a_s19;
        a_m4 = R13_DGC_6 * a_s22;
        a_m5 = R13_DGC_7 * a_s23;
        a_s43 = a_m4 + a_m5;
        a_m6 = R13_DGC_6 * a_s23;
        a_m7 = R13_DGC_7 * a_s22;
        a_s45 = a_m6 - a_m7;
        a_s47 = a_s21 + a_s18;
        a_s48 = a_s21 - a_s18;
        a_m8 = CRTM_13_7 * a_s16;
        a_s32 = a_s47 - a_m8;
        a_s28 = a_s47 + a_s16;
        a_m9 = CRTM_13_7 * a_s17;
        a_s33 = a_s48 + a_m9;
        a_s39 = a_s48 - a_s17;
        a_m10 = a_s39 * R13_DGC_12;
        // Output point 1: X(0)
        *out = a_s28 + a_in0;

        a_m11 = -(a_s28 * R13_DGC_1);
        a_s63 = a_m11 + a_in0;
        a_s24 = a_s63 + a_m10;
        a_s25 = a_s63 - a_m10;

        a_s61 = a_s36 + a_m0;
        a_s62 = a_s36 - a_m0;
        a_s46 = a_s42 + a_m1;
        a_s29 = a_s42 - a_m1;

        a_m12 = R13_DGC_2 * a_s61;
        a_m13 = R13_DGC_3 * a_s46;
        a_s40 = -(a_m12 + a_m13);
        a_m14 = R13_DGC_2 * a_s46;
        a_m15 = R13_DGC_3 * a_s61;
        a_s41 = a_m14 - a_m15;

        a_m16 = R13_DGC_10 * a_s29;
        a_m17 = R13_DGC_11 * a_s62;
        a_s34 = a_m17 - a_m16;
        a_m18 = R13_DGC_10 * a_s62;
        a_m19 = R13_DGC_11 * a_s29;
        a_s35 = a_m18 + a_m19;

        a_s26 = a_s41 + a_s34;
        a_s44 = a_s41 - a_s34;
        a_m20 = CRTM_13_1 * a_s44;
        a_s27 = a_s40 + a_s35;
        a_s64 = a_s40 - a_s35;
        a_m21 = CRTM_13_1 * a_s64;

        a_s30 = a_s7 + a_s14;
        a_m22 = R13_DGC_4 * a_s33;
        a_m23 = CRTM_13_3 * a_s30;
        a_s49 = a_m22 - a_m23;
        a_m24 = CRTM_13_2 * a_s30;
        a_m25 = R13_DGC_5 * a_s33;
        a_s37 = -(a_m24 + a_m25);

        a_s31 = a_s7 - a_s14;
        a_m26 = R13_DGC_8 * a_s32;
        a_m27 = CRTM_13_5 * a_s31;
        a_s50 = a_m26 - a_m27;
        a_m28 = CRTM_13_4 * a_s31;
        a_m29 = R13_DGC_9 * a_s32;
        a_s38 = -(a_m28 + a_m29);

        a_s51 = a_s37 + a_s38;
        a_s59 = a_s37 - a_s38;
        a_s52 = a_s49 + a_s50;
        a_s53 = a_s49 - a_s50;

        a_m30 = CRTM_13_8 * a_s52;
        // Output point 4: X(3)
        out[out_strides[3]] = a_s24 + a_m30;

        a_m31 = CRTM_13_8 * a_s26;
        // Output point 5: X(4)
        out[out_strides[4]] = a_s45 + a_m31;

        a_s54 = a_s24 - a_s52;
        a_s55 = a_s45 - a_s26;
        // Output point 12: X(11)
        out[out_strides[11]] = a_s54 + a_s59;
        // Output point 13: X(12)
        out[out_strides[12]] = a_s55 + a_m21;
        // Output point 16: X(15)
        out[out_strides[15]] = a_s54 - a_s59;
        // Output point 17: X(16)
        out[out_strides[16]] = a_m21 - a_s55;

        a_s56 = a_s27 - a_s43;
        a_m32 = CRTM_13_8 * a_s27;
        a_s57 = -(a_m32 + a_s43);
        a_m33 = CRTM_13_8 * a_s53;
        a_s58 = a_s25 - a_m33;
        a_s60 = a_s25 + a_s53;
        // Output point 8: X(7)
        out[out_strides[7]] = a_s60 - a_s51;
        // Output point 24: X(23)
        out[out_strides[23]] = a_s60 + a_s51;

        // Output point 20: X(19)
        out[out_strides[19]] = a_s58;
        // Output point 21: X(20)
        out[out_strides[20]] = a_s57;

        // Output point 9: X(8)
        out[out_strides[8]] = a_m20 + a_s56;
        // Output point 25: X(24)
        out[out_strides[24]] = a_s56 - a_m20;

        // Shifted DFT
        FFTZ_FLOAT b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7,
                   b_in8, b_in9, b_in10, b_in11, b_in12;
        FFTZ_FLOAT b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9,
                   b_s10, b_s11, b_s12, b_s13, b_s14, b_s15, b_s16, b_s17,
                   b_s18, b_s19, b_s20, b_s21, b_s22, b_s23, b_s24, b_s25,
                   b_s26, b_s27, b_s28, b_s29, b_s30, b_s31, b_s32, b_s33,
                   b_s34, b_s35, b_s36, b_s37, b_s38, b_s39, b_s40, b_s41,
                   b_s42, b_s43, b_s44, b_s45, b_s46, b_s47, b_s48, b_s49,
                   b_s50, b_s51, b_s52, b_s53, b_s54, b_s55, b_s56, b_s57,
                   b_s58, b_s59, b_s60, b_s61, b_s62, b_s63, b_s64, b_s65,
                   b_s66, b_s67, b_s68, b_s69, b_s70, b_s71, b_s72, b_s73,
                   b_s74, b_s75, b_s76;
        FFTZ_FLOAT b_m0, b_m1, b_m2, b_m3, b_m4, b_m5, b_m6, b_m7, b_m8, b_m9,
                   b_m10, b_m11, b_m12, b_m13, b_m14, b_m15, b_m16, b_m17,
                   b_m18, b_m19, b_m20, b_m21, b_m22, b_m23, b_m24, b_m25,
                   b_m26, b_m27, b_m28, b_m29, b_m30, b_m31, b_m32, b_m33,
                   b_m34, b_m35, b_m36, b_m37, b_m38, b_m39, b_m40, b_m41,
                   b_m42, b_m43, b_m44, b_m45, b_m46, b_m47, b_m48, b_m49,
                   b_m50, b_m51, b_m52, b_m53, b_m54, b_m55, b_m56, b_m57,
                   b_m58, b_m59, b_m60, b_m61, b_m62, b_m63, b_m64, b_m65,
                   b_m66, b_m67, b_m68, b_m69, b_m70, b_m71;

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

        b_s0 = b_in1 + b_in12;
        b_s1 = b_in1 - b_in12;
        b_s2 = b_in2 + b_in11;
        b_s3 = b_in2 - b_in11;
        b_s4 = b_in3 + b_in10;
        b_s5 = b_in3 - b_in10;
        b_s6 = b_in4 + b_in9;
        b_s7 = b_in4 - b_in9;
        b_s8 = b_in5 + b_in8;
        b_s9 = b_in5 - b_in8;
        b_s10 = b_in6 + b_in7;
        b_s11 = b_in6 - b_in7;

        b_m0 = R13_DFT_C1 * b_s1;
        b_m1 = R13_DFT_C2 * b_s3;
        b_s12 = b_m0 + b_m1;
        b_m2 = R13_DFT_C3 * b_s5;
        b_m3 = R13_DFT_C4 * b_s7;
        b_s13 = b_m2 + b_m3;
        b_m4 = R13_DFT_C5 * b_s9;
        b_m5 = R13_DFT_C6 * b_s11;
        b_s14 = b_m4 + b_m5;
        b_s15 = b_s12 + b_s13;
        b_s16 = b_s15 + b_s14;
        // Output point 2: X(1)
        out[out_strides[1]] = b_in0 + b_s16;

        b_m6 = R13_DFT_S1 * b_s0;
        b_m7 = R13_DFT_S2 * b_s2;
        b_s17 = b_m6 + b_m7;
        b_m8 = R13_DFT_S3 * b_s4;
        b_m9 = R13_DFT_S4 * b_s6;
        b_s18 = b_m8 + b_m9;
        b_m10 = R13_DFT_S5 * b_s8;
        b_m11 = R13_DFT_S6 * b_s10;
        b_s19 = b_m10 + b_m11;
        b_s20 = b_s17 + b_s18;
        b_s21 = b_s20 + b_s19;
        // Output point 3: X(2)
        out[out_strides[2]] = -b_s21;

        b_m12 = R13_DFT_C3 * b_s1;
        b_m13 = R13_DFT_C6 * b_s3;
        b_s22 = b_m12 + b_m13;
        b_m14 = R13_DFT_C4 * b_s5;
        b_m15 = R13_DFT_C1 * b_s7;
        b_s23 = b_m14 + b_m15;
        b_m16 = R13_DFT_C2 * b_s9;
        b_m17 = R13_DFT_C5 * b_s11;
        b_s24 = b_m16 + b_m17;
        b_s25 = b_s23 + b_s24;
        b_s26 = b_s22 - b_s25;
        // Output point 6: X(5)
        out[out_strides[5]] = b_in0 + b_s26;

        b_m18 = R13_DFT_S3 * b_s0;
        b_m19 = R13_DFT_S6 * b_s2;
        b_s27 = b_m18 + b_m19;
        b_m20 = R13_DFT_S4 * b_s4;
        b_m21 = R13_DFT_S1 * b_s6;
        b_s28 = b_m20 + b_m21;
        b_m22 = R13_DFT_S2 * b_s8;
        b_m23 = R13_DFT_S5 * b_s10;
        b_s29 = b_m22 + b_m23;
        b_s30 = b_s27 + b_s28;
        b_s31 = b_s29 - b_s30;
        // Output point 7: X(6)
        out[out_strides[6]] = b_s31;

        b_m24 = R13_DFT_C5 * b_s1;
        b_m25 = R13_DFT_C3 * b_s3;
        b_s32 = b_m24 - b_m25;
        b_m26 = R13_DFT_C2 * b_s5;
        b_m27 = R13_DFT_C6 * b_s7;
        b_s33 = b_m27 - b_m26;
        b_m28 = R13_DFT_C1 * b_s9;
        b_m29 = R13_DFT_C4 * b_s11;
        b_s34 = b_m28 + b_m29;
        b_s35 = b_s32 + b_s33;
        b_s36 = b_s35 + b_s34;
        // Output point 10: X(9)
        out[out_strides[9]] = b_in0 + b_s36;

        b_m30 = R13_DFT_S5 * b_s0;
        b_m31 = R13_DFT_S3 * b_s2;
        b_s37 = b_m30 + b_m31;
        b_m32 = R13_DFT_S2 * b_s4;
        b_m33 = R13_DFT_S6 * b_s6;
        b_s38 = b_m32 + b_m33;
        b_m34 = R13_DFT_S1 * b_s8;
        b_m35 = R13_DFT_S4 * b_s10;
        b_s39 = b_m34 - b_m35;
        b_s40 = b_s38 + b_s39;
        b_s41 = b_s40 - b_s37;
        // Output point 11: X(10)
        out[out_strides[10]] = b_s41;

        b_m36 = R13_DFT_C6 * b_s1;
        b_m37 = R13_DFT_C1 * b_s3;
        b_s42 = b_m36 + b_m37;
        b_m38 = R13_DFT_C5 * b_s5;
        b_m39 = R13_DFT_C2 * b_s7;
        b_s43 = b_m38 + b_m39;
        b_m40 = R13_DFT_C4 * b_s9;
        b_m41 = R13_DFT_C3 * b_s11;
        b_s44 = b_m40 + b_m41;
        b_s45 = b_s42 + b_s44;
        b_s46 = b_s43 - b_s45;
        // Output point 14: X(13)
        out[out_strides[13]] = b_in0 + b_s46;

        b_m42 = R13_DFT_S6 * b_s0;
        b_m43 = R13_DFT_S1 * b_s2;
        b_s47 = b_m43 - b_m42;
        b_m44 = R13_DFT_S5 * b_s4;

        b_m45 = R13_DFT_S2 * b_s6;
        b_s48 = b_m44 - b_m45;
        b_m46 = R13_DFT_S4 * b_s8;
        b_m47 = R13_DFT_S3 * b_s10;
        b_s49 = b_m47 - b_m46;
        b_s50 = b_s47 + b_s48;
        b_s51 = b_s50 + b_s49;
        // Output point 15: X(14)
        out[out_strides[14]] = b_s51;

        b_m48 = R13_DFT_C4 * b_s1;

        b_m49 = R13_DFT_C5 * b_s3;
        b_s52 = b_m48 + b_m49;
        b_m50 = R13_DFT_C1 * b_s5;

        b_m51 = R13_DFT_C3 * b_s7;
        b_s53 = b_m50 - b_m51;
        b_m52 = R13_DFT_C6 * b_s9;

        b_m53 = R13_DFT_C2 * b_s11;
        b_s54 = b_m53 - b_m52;
        b_s55 = b_s53 + b_s54;
        b_s56 = b_s55 - b_s52;
        // Output point 18: X(17)
        out[out_strides[17]] = b_in0 + b_s56;

        b_m54 = R13_DFT_S4 * b_s0;

        b_m55 = R13_DFT_S5 * b_s2;
        b_s57 = b_m55 - b_m54;
        b_m56 = R13_DFT_S1 * b_s4;

        b_m57 = R13_DFT_S3 * b_s6;
        b_s58 = b_m56 + b_m57;
        b_m58 = R13_DFT_S6 * b_s8;

        b_m59 = R13_DFT_S2 * b_s10;
        b_s59 = b_m58 - b_m59;
        b_s60 = b_s57 + b_s59;
        b_s61 = b_s60 - b_s58;
        // Output point 19: X(18)
        out[out_strides[18]] = b_s61;

        b_m60 = R13_DFT_C2 * b_s1;
        b_m61 = R13_DFT_C4 * b_s3;
        b_s62 = b_m61 - b_m60;
        b_m62 = R13_DFT_C6 * b_s5;
        b_m63 = R13_DFT_C5 * b_s7;
        b_s63 = b_m62 + b_m63;
        b_m64 = R13_DFT_C3 * b_s9;
        b_m65 = R13_DFT_C1 * b_s11;
        b_s64 = b_m64 - b_m65;
        b_s65 = b_s62 + b_s64;
        b_s66 = b_s65 - b_s63;
        // Output point 22: X(21)
        out[out_strides[21]] = b_in0 + b_s66;

        b_m66 = R13_DFT_S2 * b_s0;
        b_m67 = R13_DFT_S4 * b_s2;
        b_s67 = b_m67 - b_m66;
        b_m68 = R13_DFT_S6 * b_s4;
        b_m69 = R13_DFT_S5 * b_s6;
        b_s68 = b_m69 - b_m68;
        b_m70 = R13_DFT_S3 * b_s8;
        b_m71 = R13_DFT_S1 * b_s10;
        b_s69 = b_m71 - b_m70;
        b_s70 = b_s67 + b_s68;
        b_s71 = b_s70 + b_s69;
        // Output point 23: X(22)
        out[out_strides[22]] = b_s71;

        b_s72 = b_s3 - b_s1;
        b_s73 = b_s7 - b_s5;
        b_s74 = b_s11 - b_s9;
        b_s75 = b_s72 + b_s73;
        b_s76 = b_s75 + b_s74;
        // Output point 26: X(25)
        out[out_strides[25]] = b_in0 + b_s76;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft13c_fp32_bwd(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                        FFTZ_VOID *out_real,
                                        FFTZ_VOID *out_imag, FFTZ_INTP n,
                                        aoclfftz_strides_t *strides,
                                        FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_13_1 =
        1.732050807568877293527446341505872366942805254f;
    const FFTZ_FLOAT CRTM_13_2 =
        1.007074065727533254493747707736933954186697125f;
    const FFTZ_FLOAT CRTM_13_3 =
        0.531932498429674575175042127684371897596660533f;
    const FFTZ_FLOAT CRTM_13_4 =
        1.150281458948006242736771094910906776922003215f;
    const FFTZ_FLOAT CRTM_13_5 =
        0.348277202304271810011321589858529485233929352f;
    const FFTZ_FLOAT CRTM_13_6 =
        0.500000000000000000000000000000000000000000000f;
    const FFTZ_FLOAT CRTM_13_7 =
        2.000000000000000000000000000000000000000000000f;
    const FFTZ_FLOAT R13_DGC_1 =
        0.166666666666666666666666666666666666666666667f;
    const FFTZ_FLOAT R13_DGC_2 =
        0.256247671582936600958684654061725059144125175f;
    const FFTZ_FLOAT R13_DGC_3 =
        0.156891391051584611046832726756003269660212636f;
    const FFTZ_FLOAT R13_DGC_4 =
        0.774781170935234584261351932853525703557550433f;
    const FFTZ_FLOAT R13_DGC_5 =
        0.265966249214837287587521063842185948798330267f;
    const FFTZ_FLOAT R13_DGC_6 =
        0.600925212577331548853203544578415991041882762f;
    const FFTZ_FLOAT R13_DGC_7 =
        0.151805972074387731966205794490207080712856746f;
    const FFTZ_FLOAT R13_DGC_8 =
        0.227708958111581597949308691735310621069285120f;
    const FFTZ_FLOAT R13_DGC_9 =
        0.503537032863766627246873853868466977093348562f;
    const FFTZ_FLOAT R13_DGC_10 =
        0.300238635966332641462884626667381504676006424f;
    const FFTZ_FLOAT R13_DGC_11 =
        0.011599105605768290721655456654083252189827041f;
    const FFTZ_FLOAT R13_DGC_12 =
        0.516520780623489722840901288569017135705033622f;
    const FFTZ_FLOAT R13_DFT_C1 =
        1.94188363485210405431396455258757845449973021147800f;
    const FFTZ_FLOAT R13_DFT_C2 =
        1.77091205130641979180075104403019775721099683269510f;
    const FFTZ_FLOAT R13_DFT_C3 =
        1.49702149634220219726926119940270276769290318035170f;
    const FFTZ_FLOAT R13_DFT_C4 =
        1.13612949346231160502361511825503324906698510490700f;
    const FFTZ_FLOAT R13_DFT_C5 =
        0.70920977408507125193927578520003694863271086422760f;
    const FFTZ_FLOAT R13_DFT_C6 =
        0.24107336051064610669813537490508716454736231845510f;
    const FFTZ_FLOAT R13_DFT_S1 =
        0.47863132857511553429750745252042379040634604547660f;
    const FFTZ_FLOAT R13_DFT_S2 =
        0.92944634408753709131203067026620955511547173066490f;
    const FFTZ_FLOAT R13_DFT_S3 =
        1.32624531648159040475357098533353255904952821408820f;
    const FFTZ_FLOAT R13_DFT_S4 =
        1.64596773178731278915923484687876398131013538617510f;
    const FFTZ_FLOAT R13_DFT_S5 =
        1.87003248537082964687956919967566145810103493915690f;
    const FFTZ_FLOAT R13_DFT_S6 =
        1.98541774819610798560150329898504035868735126594030f;

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
        // Standard DFT
        FFTZ_FLOAT a_in0, a_in1, a_in2, a_in3, a_in4, a_in5, a_in6, a_in7,
                   a_in8, a_in9, a_in10, a_in11, a_in12;
        FFTZ_FLOAT a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_s9,
                   a_s10, a_s11, a_s12, a_s13, a_s14, a_s15, a_s16, a_s17,
                   a_s18, a_s19, a_s20, a_s21, a_s22, a_s23, a_s24, a_s25,
                   a_s26, a_s27, a_s28, a_s29, a_s30, a_s31, a_s32, a_s33,
                   a_s34, a_s35, a_s36, a_s37, a_s38, a_s39, a_s40, a_s41,
                   a_s42, a_s43, a_s44, a_s45, a_s46, a_s47, a_s48, a_s49,
                   a_s50, a_s51, a_s52, a_s53, a_s54, a_s55, a_s56, a_s57,
                   a_s58, a_s59, a_s60, a_s61, a_s62;
        FFTZ_FLOAT a_m0, a_m1, a_m2, a_m3, a_m4, a_m5, a_m6, a_m7, a_m8, a_m9,
                   a_m10, a_m11, a_m12, a_m13, a_m14, a_m15, a_m16, a_m17,
                   a_m18, a_m19, a_m20, a_m21, a_m22, a_m23, a_m24, a_m25,
                   a_m26, a_m27, a_m28, a_m29, a_m30, a_m31, a_m32, a_m33,
                   a_m34;

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
        // Input point 24: X(23)
        a_in11 = in[in_strides[23]];
        // Input point 25: X(24)
        a_in12 = in[in_strides[24]];
        a_s0 = a_in6 - a_in8;
        a_m0 = CRTM_13_7 * a_in2;
        a_s1 = a_m0 - a_s0;
        a_s59 = a_in6 + a_in8;
        a_m1 = CRTM_13_1 * a_s59;
        a_s2 = a_in12 + a_in4;
        a_s60 = a_in12 - a_in4;
        a_m2 = CRTM_13_1 * a_s60;
        a_m3 = CRTM_13_7 * a_in10;
        a_s3 = a_s2 - a_m3;
        a_s4 = a_s1 + a_m2;
        a_s5 = a_s3 - a_m1;
        a_m4 = R13_DGC_11 * a_s4;
        a_m5 = R13_DGC_10 * a_s5;
        a_s6 = a_m4 + a_m5;
        a_m6 = R13_DGC_10 * a_s4;
        a_m7 = R13_DGC_11 * a_s5;
        a_s7 = a_m6 - a_m7;
        a_s8 = a_in2 + a_s0;
        a_s9 = a_s2 + a_in10;
        a_m8 = CRTM_13_4 * a_s8;
        a_m9 = CRTM_13_5 * a_s9;
        a_s10 = a_m8 - a_m9;
        a_m10 = CRTM_13_5 * a_s8;
        a_m11 = CRTM_13_4 * a_s9;
        a_s11 = a_m10 + a_m11;
        a_s12 = a_s1 - a_m2;
        a_s13 = a_m1 + a_s3;
        a_m12 = R13_DGC_3 * a_s12;
        a_m13 = R13_DGC_2 * a_s13;
        a_s14 = a_m12 + a_m13;
        a_m14 = R13_DGC_3 * a_s13;
        a_m15 = R13_DGC_2 * a_s12;
        a_s15 = a_m14 - a_m15;
        a_s16 = a_in3 + a_in11;
        a_s17 = a_in9 + a_s16;
        a_m16 = CRTM_13_6 * a_s16;
        a_s18 = a_in9 - a_m16;
        a_s19 = a_in3 - a_in11;
        a_s20 = a_in5 + a_in7;
        a_s21 = a_in1 + a_s20;
        a_m17 = CRTM_13_6 * a_s20;
        a_s22 = a_in1 - a_m17;
        a_s23 = a_in5 - a_in7;
        a_s24 = a_s21 - a_s17;
        a_m18 = R13_DGC_6 * a_s24;
        a_s25 = a_s21 + a_s17;
        a_m33 = CRTM_13_7 * a_s25;
        // Output point 1: x(0)
        *out = a_m33 + a_in0;

        a_m34 = R13_DGC_1 * a_s25;
        a_s26 = a_in0 - a_m34;
        a_s27 = a_s23 + a_s19;
        a_s28 = a_s22 + a_s18;
        a_m19 = R13_DGC_9 * a_s27;
        a_m20 = R13_DGC_7 * a_s28;
        a_s29 = a_m19 + a_m20;
        a_s30 = a_s22 - a_s18;
        a_s31 = a_s23 - a_s19;
        a_m21 = R13_DGC_12 * a_s30;
        a_m22 = R13_DGC_5 * a_s31;
        a_s32 = a_m21 - a_m22;
        a_s61 = a_s6 + a_s14;
        a_m23 = CRTM_13_1 * a_s61;
        a_s62 = a_s7 - a_s15;
        a_m24 = CRTM_13_1 * a_s62;
        a_s33 = a_s7 + a_s15;
        a_s34 = a_s10 - a_s33;
        a_m25 = CRTM_13_7 * a_s33;
        a_s35 = a_m25 + a_s10;
        a_s36 = a_s6 - a_s14;
        a_m26 = CRTM_13_7 * a_s36;
        a_s37 = a_m26 - a_s11;
        a_s38 = a_s36 + a_s11;
        a_m27 = R13_DGC_4 * a_s31;
        a_m28 = CRTM_13_3 * a_s30;
        a_s39 = a_m27 + a_m28;
        a_m29 = R13_DGC_8 * a_s27;
        a_m30 = CRTM_13_2 * a_s28;
        a_s40 = a_m29 - a_m30;
        a_s41 = a_s39 - a_s40;
        a_s42 = a_s39 + a_s40;
        a_s43 = a_s26 - a_s29;
        a_s44 = a_m18 - a_s32;
        a_s45 = a_s43 - a_s44;
        a_s46 = a_s44 + a_s43;
        a_m31 = CRTM_13_7 * a_s29;
        a_s47 = a_m31 + a_s26;
        a_m32 = CRTM_13_7 * a_s32;
        a_s48 = a_m32 + a_m18;
        a_s49 = a_s47 - a_s48;
        // Output point 17: x(16)
        out[out_strides[16]] = a_s49 + a_s35;
        // Output point 11: x(10)
        out[out_strides[10]] = a_s49 - a_s35;

        a_s50 = a_s48 + a_s47;
        // Output point 25: x(24)
        out[out_strides[24]] = a_s50 - a_s37;
        // Output point 3: x(2)
        out[out_strides[2]] = a_s50 + a_s37;

        a_s51 = a_s45 - a_m23;
        a_s52 = a_s41 - a_s34;
        // Output point 5: x(4)
        out[out_strides[4]] = a_s51 + a_s52;
        // Output point 15: x(14)
        out[out_strides[14]] = a_s51 - a_s52;

        a_s53 = a_s46 - a_s38;
        a_s54 = a_s42 + a_m24;
        // Output point 19: x(18)
        out[out_strides[18]] = a_s54 + a_s53;
        // Output point 7: x(6)
        out[out_strides[6]] = a_s53 - a_s54;

        a_s55 = a_s42 - a_m24;
        a_s56 = a_s46 + a_s38;
        // Output point 9: x(8)
        out[out_strides[8]] = a_s55 + a_s56;
        // Output point 21: x(20)
        out[out_strides[20]] = a_s56 - a_s55;

        a_s57 = a_s45 + a_m23;
        a_s58 = a_s41 + a_s34;
        // Output point 13: x(12)
        out[out_strides[12]] = a_s57 - a_s58;
        // Output point 23: x(22)
        out[out_strides[22]] = a_s57 + a_s58;

        // Shifted DFT
        FFTZ_FLOAT b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7,
                   b_in8, b_in9, b_in10, b_in11, b_in12;
        FFTZ_FLOAT b_m0, b_m1, b_m2, b_m3, b_m4, b_m5, b_m6, b_m7, b_m8, b_m9,
                   b_m10, b_m11, b_m12, b_m13, b_m14, b_m15, b_m16, b_m17,
                   b_m18, b_m19, b_m20, b_m21, b_m22, b_m23, b_m24, b_m25,
                   b_m26, b_m27, b_m28, b_m29, b_m30, b_m31, b_m32, b_m33,
                   b_m34, b_m35, b_m36, b_m37, b_m38, b_m39, b_m40, b_m41,
                   b_m42, b_m43, b_m44, b_m45, b_m46, b_m47, b_m48, b_m49,
                   b_m50, b_m51, b_m52, b_m53, b_m54, b_m55, b_m56, b_m57,
                   b_m58, b_m59, b_m60, b_m61, b_m62, b_m63, b_m64, b_m65,
                   b_m66, b_m67, b_m68, b_m69, b_m70, b_m71;
        FFTZ_FLOAT b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9,
                   b_s10, b_s11, b_s12, b_s13, b_s14, b_s15, b_s16, b_s17,
                   b_s18, b_s19, b_s20, b_s21, b_s22, b_s23, b_s24, b_s25,
                   b_s26, b_s27, b_s28, b_s29, b_s30, b_s31, b_s32, b_s33,
                   b_s34, b_s35, b_s36, b_s37, b_s38, b_s39, b_s40, b_s41,
                   b_s42, b_s43, b_s44, b_s45, b_s46, b_s47, b_s48, b_s49,
                   b_s50, b_s51, b_s52, b_s53, b_s54, b_s55, b_s56, b_s57,
                   b_s58, b_s59, b_s60, b_s61, b_s62, b_s63, b_s64, b_s65,
                   b_s66, b_s67, b_s68, b_s69, b_s70, b_s71;

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
        // Input point 23: X(22)
        b_in11 = in[in_strides[22]];
        // Input point 26: X(25)
        b_in12 = in[in_strides[25]];

        b_s0 = b_in0 + b_in2;
        b_s1 = b_in4 + b_in6;
        b_s2 = b_in8 + b_in10;
        b_s3 = b_s0 + b_s1;
        b_s4 = b_s3 + b_s2;
        b_s71 = b_s4 + b_s4;
        // Output point 2: x(1)
        out[out_strides[1]] = b_s71 + b_in12;

        b_m0 = R13_DFT_C1 * b_in0;
        b_m1 = R13_DFT_C3 * b_in2;
        b_s5 = b_m0 + b_m1;
        b_m2 = R13_DFT_C5 * b_in4;
        b_m3 = R13_DFT_C6 * b_in6;
        b_s6 = b_m2 - b_m3;
        b_m4 = R13_DFT_C4 * b_in8;
        b_m5 = R13_DFT_C2 * b_in10;
        b_s7 = b_m4 + b_m5;
        b_s8 = b_s5 + b_s6;
        b_s9 = b_s8 - b_s7;
        b_s10 = b_s9 - b_in12;

        b_m6 = R13_DFT_S1 * b_in1;
        b_m7 = R13_DFT_S3 * b_in3;
        b_s11 = b_m6 + b_m7;
        b_m8 = R13_DFT_S5 * b_in5;
        b_m9 = R13_DFT_S6 * b_in7;
        b_s12 = b_m8 + b_m9;
        b_m10 = R13_DFT_S4 * b_in9;
        b_m11 = R13_DFT_S2 * b_in11;
        b_s13 = b_m10 + b_m11;
        b_s14 = b_s11 + b_s12;
        b_s15 = b_s14 + b_s13;

        // Output point 4: x(3)
        out[out_strides[3]] = b_s10 - b_s15;
        // Output point 26: x(25)
        out[out_strides[25]] = -b_s10 - b_s15;

        b_m12 = R13_DFT_C2 * b_in0;
        b_m13 = R13_DFT_C6 * b_in2;
        b_s16 = b_m12 + b_m13;
        b_m14 = R13_DFT_C3 * b_in4;
        b_m15 = R13_DFT_C1 * b_in6;
        b_s17 = b_m14 + b_m15;
        b_m16 = R13_DFT_C5 * b_in8;
        b_m17 = R13_DFT_C4 * b_in10;
        b_s18 = b_m17 - b_m16;
        b_s19 = b_s16 - b_s17;
        b_s20 = b_s19 + b_s18;
        b_s21 = b_s20 + b_in12;

        b_m18 = R13_DFT_S2 * b_in1;
        b_m19 = R13_DFT_S6 * b_in3;
        b_s22 = b_m18 + b_m19;
        b_m20 = R13_DFT_S3 * b_in5;
        b_m21 = R13_DFT_S1 * b_in7;
        b_s23 = b_m20 - b_m21;
        b_m22 = R13_DFT_S5 * b_in9;
        b_m23 = R13_DFT_S4 * b_in11;
        b_s24 = b_m22 + b_m23;
        b_s25 = b_s22 + b_s23;
        b_s26 = b_s25 - b_s24;

        // Output point 6: x(5)
        out[out_strides[5]] = b_s21 - b_s26;
        // Output point 24: x(23)
        out[out_strides[23]] = -b_s21 - b_s26;

        b_m24 = R13_DFT_C3 * b_in0;
        b_m25 = R13_DFT_C4 * b_in2;
        b_s27 = b_m24 - b_m25;
        b_m26 = R13_DFT_C2 * b_in4;
        b_m27 = R13_DFT_C5 * b_in6;
        b_s28 = b_m27 - b_m26;
        b_m28 = R13_DFT_C1 * b_in8;
        b_m29 = R13_DFT_C6 * b_in10;
        b_s29 = b_m28 - b_m29;
        b_s30 = b_s27 + b_s28;
        b_s31 = b_s30 + b_s29;
        b_s32 = b_s31 - b_in12;

        b_m30 = R13_DFT_S3 * b_in1;
        b_m31 = R13_DFT_S4 * b_in3;
        b_s33 = b_m30 + b_m31;
        b_m32 = R13_DFT_S2 * b_in5;
        b_m33 = R13_DFT_S5 * b_in7;
        b_s34 = b_m32 + b_m33;
        b_m34 = R13_DFT_S1 * b_in9;
        b_m35 = R13_DFT_S6 * b_in11;
        b_s35 = b_m34 + b_m35;
        b_s36 = b_s33 - b_s34;
        b_s37 = b_s36 + b_s35;

        // Output point 8: x(7)
        out[out_strides[7]] = b_s32 - b_s37;
        // Output point 22: x(21)
        out[out_strides[21]] = -b_s32 - b_s37;

        b_m36 = R13_DFT_C4 * b_in0;
        b_m37 = R13_DFT_C1 * b_in2;
        b_s38 = b_m36 - b_m37;
        b_m38 = R13_DFT_C6 * b_in4;
        b_m39 = R13_DFT_C2 * b_in6;
        b_s39 = b_m38 + b_m39;
        b_m40 = R13_DFT_C3 * b_in8;
        b_m41 = R13_DFT_C5 * b_in10;
        b_s40 = b_m40 + b_m41;
        b_s41 = b_s38 + b_s39;
        b_s42 = b_s41 - b_s40;
        b_s43 = b_s42 + b_in12;

        b_m42 = R13_DFT_S4 * b_in1;
        b_m43 = R13_DFT_S1 * b_in3;
        b_s44 = b_m42 + b_m43;
        b_m44 = R13_DFT_S6 * b_in5;
        b_m45 = R13_DFT_S2 * b_in7;
        b_s45 = b_m45 - b_m44;
        b_m46 = R13_DFT_S3 * b_in9;
        b_m47 = R13_DFT_S5 * b_in11;
        b_s46 = b_m46 - b_m47;
        b_s47 = b_s44 + b_s45;
        b_s48 = b_s47 + b_s46;

        // Output point 10: x(9)
        out[out_strides[9]] = b_s43 - b_s48;
        // Output point 20: x(19)
        out[out_strides[19]] = -b_s43 - b_s48;

        b_m48 = R13_DFT_C5 * b_in0;
        b_m49 = R13_DFT_C2 * b_in2;
        b_s49 = b_m48 - b_m49;
        b_m50 = R13_DFT_C1 * b_in4;
        b_m51 = R13_DFT_C4 * b_in6;
        b_s50 = b_m50 - b_m51;
        b_m52 = R13_DFT_C6 * b_in8;
        b_m53 = R13_DFT_C3 * b_in10;
        b_s51 = b_m53 - b_m52;
        b_s52 = b_s49 + b_s50;
        b_s53 = b_s52 + b_s51;
        b_s54 = b_s53 - b_in12;

        b_m54 = R13_DFT_S5 * b_in1;
        b_m55 = R13_DFT_S2 * b_in3;
        b_s55 = b_m54 - b_m55;
        b_m56 = R13_DFT_S1 * b_in5;
        b_m57 = R13_DFT_S4 * b_in7;
        b_s56 = b_m57 - b_m56;
        b_m58 = R13_DFT_S6 * b_in9;
        b_m59 = R13_DFT_S3 * b_in11;
        b_s57 = b_m59 - b_m58;
        b_s58 = b_s55 + b_s56;
        b_s59 = b_s58 + b_s57;

        // Output point 12: x(11)
        out[out_strides[11]] = b_s54 - b_s59;
        // Output point 18: x(17)
        out[out_strides[17]] = -b_s54 - b_s59;

        b_m60 = R13_DFT_C6 * b_in0;
        b_m61 = R13_DFT_C5 * b_in2;
        b_s60 = b_m60 - b_m61;
        b_m62 = R13_DFT_C4 * b_in4;
        b_m63 = R13_DFT_C3 * b_in6;
        b_s61 = b_m62 - b_m63;
        b_m64 = R13_DFT_C2 * b_in8;
        b_m65 = R13_DFT_C1 * b_in10;
        b_s62 = b_m64 - b_m65;
        b_s63 = b_s60 + b_s61;
        b_s64 = b_s63 + b_s62;
        b_s65 = b_s64 + b_in12;

        b_m66 = R13_DFT_S6 * b_in1;
        b_m67 = R13_DFT_S5 * b_in3;
        b_s66 = b_m66 - b_m67;
        b_m68 = R13_DFT_S4 * b_in5;
        b_m69 = R13_DFT_S3 * b_in7;
        b_s67 = b_m68 - b_m69;
        b_m70 = R13_DFT_S2 * b_in9;
        b_m71 = R13_DFT_S1 * b_in11;
        b_s68 = b_m70 - b_m71;
        b_s69 = b_s66 + b_s67;
        b_s70 = b_s69 + b_s68;

        // Output point 14: x(13)
        out[out_strides[13]] = b_s65 - b_s70;
        // Output point 16: x(15)
        out[out_strides[15]] = -b_s65 - b_s70;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft13c_fp64_fwd(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                        FFTZ_VOID *out_real,
                                        FFTZ_VOID *out_imag, FFTZ_INTP n,
                                        aoclfftz_strides_t *strides,
                                        FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_13_1 =
        1.73205080756887719317660412343684583902359008789060;
    const FFTZ_DOUBLE CRTM_13_2 =
        0.38739058546761714712838513245787843298017865366345;
    const FFTZ_DOUBLE CRTM_13_3 =
        0.13298312460741867056777367279491668244720145803175;
    const FFTZ_DOUBLE CRTM_13_4 =
        0.11385447905579067614456922127652058891458054833493;
    const FFTZ_DOUBLE CRTM_13_5 =
        0.25176851643188325619942540855595308576560112726050;
    const FFTZ_DOUBLE CRTM_13_6 =
        0.86602540378443864676372317075293618347140262700000;
    const FFTZ_DOUBLE CRTM_13_7 =
        0.50000000000000000000000000000000000000000000000000;
    const FFTZ_DOUBLE CRTM_13_8 =
        2.00000000000000000000000000000000000000000000000000;
    const FFTZ_DOUBLE R13_DGC_1 =
        0.08333333333333341693177688718872429763576971755296;
    const FFTZ_DOUBLE R13_DGC_2 =
        0.25624767158293663769405689230781736686616059890980;
    const FFTZ_DOUBLE R13_DGC_3 =
        0.15689139105158457352993666120753768234094124525375;
    const FFTZ_DOUBLE R13_DGC_4 =
        0.25826039031174479484752691274654956353060547073611;
    const FFTZ_DOUBLE R13_DGC_5 =
        0.26596624921483734113554734558983336489440291606351;
    const FFTZ_DOUBLE R13_DGC_6 =
        0.57514072947400312136838554745545338846100160800000;
    const FFTZ_DOUBLE R13_DGC_7 =
        0.17413860115213590500566079492926474261696467600000;
    const FFTZ_DOUBLE R13_DGC_8 =
        0.07590298603719379320004091411468632874055721777845;
    const FFTZ_DOUBLE R13_DGC_9 =
        0.50353703286376651239885081711190617153120225452101;
    const FFTZ_DOUBLE R13_DGC_10 =
        0.3002386359663325979287571057309917466781880617299;
    const FFTZ_DOUBLE R13_DGC_11 =
        0.0115991056057681999237624507209650002461566663358;
    const FFTZ_DOUBLE R13_DGC_12 =
        0.3004626062886657229721584869768537486323029264720;
    const FFTZ_DOUBLE R13_DFT_C1 =
        0.97094181742605202715698227629378922724986510573900;
    const FFTZ_DOUBLE R13_DFT_C2 =
        0.88545602565320989590037552201509887860549841634750;
    const FFTZ_DOUBLE R13_DFT_C3 =
        0.74851074817110109863463059970135138384645159017580;
    const FFTZ_DOUBLE R13_DFT_C4 =
        0.56806474673115580251180755912751662453349255245350;
    const FFTZ_DOUBLE R13_DFT_C5 =
        0.35460488704253562596963789260001847431635543211380;
    const FFTZ_DOUBLE R13_DFT_C6 =
        0.12053668025532305334906768745254358227368115922760;
    const FFTZ_DOUBLE R13_DFT_S1 =
        0.23931566428755776714875372626021189520317302273830;
    const FFTZ_DOUBLE R13_DFT_S2 =
        0.46472317204376854565601533513310477755773586533250;
    const FFTZ_DOUBLE R13_DFT_S3 =
        0.66312265824079520237678549266676627952476410704410;
    const FFTZ_DOUBLE R13_DFT_S4 =
        0.82298386589365639457961742343938199065506769308760;
    const FFTZ_DOUBLE R13_DFT_S5 =
        0.93501624268541482343978459983783072905051746957840;
    const FFTZ_DOUBLE R13_DFT_S6 =
        0.99270887409805399280075164949252017934367563297020;

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
        // Standard DFT
        FFTZ_DOUBLE a_in0, a_in1, a_in2, a_in3, a_in4, a_in5, a_in6, a_in7,
                    a_in8, a_in9, a_in10, a_in11, a_in12;
        FFTZ_DOUBLE a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_s9,
                    a_s10, a_s11, a_s12, a_s13, a_s14, a_s15, a_s16, a_s17,
                    a_s18, a_s19, a_s20, a_s21, a_s22, a_s23, a_s24, a_s25,
                    a_s26, a_s27, a_s28, a_s29, a_s30, a_s31, a_s32, a_s33,
                    a_s34, a_s35, a_s36, a_s37, a_s38, a_s39, a_s40, a_s41,
                    a_s42, a_s43, a_s44, a_s45, a_s46, a_s47, a_s48, a_s49, 
                    a_s50, a_s51, a_s52, a_s53, a_s54, a_s55, a_s56, a_s57,
                    a_s58, a_s59, a_s60, a_s61, a_s62, a_s63, a_s64;
        FFTZ_DOUBLE a_m0, a_m1, a_m2, a_m3, a_m4, a_m5, a_m6, a_m7, a_m8, a_m9,
                    a_m10, a_m11, a_m12, a_m13, a_m14, a_m15, a_m16, a_m17,
                    a_m18, a_m19, a_m20, a_m21, a_m22, a_m23, a_m24, a_m25,
                    a_m26, a_m27, a_m28, a_m29, a_m30, a_m31, a_m32, a_m33;

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

        a_s0 = a_in2 + a_in7;
        a_s1 = a_in7 - a_in2;
        a_s2 = a_in6 + a_in11;
        a_s3 = a_in11 - a_in6;
        a_s4 = a_s0 + a_s2;
        a_s5 = a_s0 - a_s2;
        a_m0 = CRTM_13_6 * a_s5;
        a_s6 = a_s1 + a_s3;
        a_s7 = a_s1 - a_s3;
        a_s8 = a_in4 + a_in10;
        a_s9 = a_in10 - a_in4;
        a_s10 = a_in3 + a_in9;
        a_s11 = a_in9 - a_in3;
        a_s12 = a_s8 + a_s10;
        a_s13 = a_s8 - a_s10;
        a_s14 = a_s9 - a_s11;
        a_s15 = a_s9 + a_s11;
        a_m1 = CRTM_13_6 * a_s15;
        a_m2 = CRTM_13_7 * a_s13;
        a_s16 = a_s4 + a_s12;
        a_s17 = a_s4 - a_s12;
        a_s18 = a_in8 + a_in5;
        a_s19 = a_in5 - a_in8;
        a_m3 = CRTM_13_7 * a_s6;
        a_s42 = a_m3 + a_s19;
        a_s20 = a_in1 - a_in12;
        a_s21 = a_in1 + a_in12;
        a_s36 = a_s20 + a_m2;
        a_s22 = a_s20 - a_s13;
        a_s23 = a_s6 - a_s19;
        a_m4 = R13_DGC_6 * a_s22;
        a_m5 = R13_DGC_7 * a_s23;
        a_s43 = a_m4 + a_m5;
        a_m6 = R13_DGC_6 * a_s23;
        a_m7 = R13_DGC_7 * a_s22;
        a_s45 = a_m6 - a_m7;
        a_s47 = a_s21 + a_s18;
        a_s48 = a_s21 - a_s18;
        a_m8 = CRTM_13_7 * a_s16;
        a_s32 = a_s47 - a_m8;
        a_s28 = a_s47 + a_s16;
        a_m9 = CRTM_13_7 * a_s17;
        a_s33 = a_s48 + a_m9;
        a_s39 = a_s48 - a_s17;
        a_m10 = a_s39 * R13_DGC_12;
        // Output point 1: X(0)
        *out = a_s28 + a_in0;

        a_m11 = -(a_s28 * R13_DGC_1);
        a_s63 = a_m11 + a_in0;
        a_s24 = a_s63 + a_m10;
        a_s25 = a_s63 - a_m10;

        a_s61 = a_s36 + a_m0;
        a_s62 = a_s36 - a_m0;
        a_s46 = a_s42 + a_m1;
        a_s29 = a_s42 - a_m1;

        a_m12 = R13_DGC_2 * a_s61;
        a_m13 = R13_DGC_3 * a_s46;
        a_s40 = -(a_m12 + a_m13);
        a_m14 = R13_DGC_2 * a_s46;
        a_m15 = R13_DGC_3 * a_s61;
        a_s41 = a_m14 - a_m15;

        a_m16 = R13_DGC_10 * a_s29;
        a_m17 = R13_DGC_11 * a_s62;
        a_s34 = a_m17 - a_m16;
        a_m18 = R13_DGC_10 * a_s62;
        a_m19 = R13_DGC_11 * a_s29;
        a_s35 = a_m18 + a_m19;

        a_s26 = a_s41 + a_s34;
        a_s44 = a_s41 - a_s34;
        a_m20 = CRTM_13_1 * a_s44;
        a_s27 = a_s40 + a_s35;
        a_s64 = a_s40 - a_s35;
        a_m21 = CRTM_13_1 * a_s64;

        a_s30 = a_s7 + a_s14;
        a_m22 = R13_DGC_4 * a_s33;
        a_m23 = CRTM_13_3 * a_s30;
        a_s49 = a_m22 - a_m23;
        a_m24 = CRTM_13_2 * a_s30;
        a_m25 = R13_DGC_5 * a_s33;
        a_s37 = -(a_m24 + a_m25);

        a_s31 = a_s7 - a_s14;
        a_m26 = R13_DGC_8 * a_s32;
        a_m27 = CRTM_13_5 * a_s31;
        a_s50 = a_m26 - a_m27;
        a_m28 = CRTM_13_4 * a_s31;
        a_m29 = R13_DGC_9 * a_s32;
        a_s38 = -(a_m28 + a_m29);

        a_s51 = a_s37 + a_s38;
        a_s59 = a_s37 - a_s38;
        a_s52 = a_s49 + a_s50;
        a_s53 = a_s49 - a_s50;

        a_m30 = CRTM_13_8 * a_s52;
        // Output point 4: X(3)
        out[out_strides[3]] = a_s24 + a_m30;
        a_m31 = CRTM_13_8 * a_s26;
        // Output point 5: X(4)
        out[out_strides[4]] = a_s45 + a_m31;

        a_s54 = a_s24 - a_s52;
        a_s55 = a_s45 - a_s26;
        // Output point 12: X(11)
        out[out_strides[11]] = a_s54 + a_s59;
        // Output point 13: X(12)
        out[out_strides[12]] = a_s55 + a_m21;
        // Output point 16: X(15)
        out[out_strides[15]] = a_s54 - a_s59;
        // Output point 17: X(16)
        out[out_strides[16]] = a_m21 - a_s55;

        a_s56 = a_s27 - a_s43;
        a_m32 = CRTM_13_8 * a_s27;
        a_s57 = -(a_m32 + a_s43);
        a_m33 = CRTM_13_8 * a_s53;
        a_s58 = a_s25 - a_m33;
        a_s60 = a_s25 + a_s53;
        // Output point 8: X(7)
        out[out_strides[7]] = a_s60 - a_s51;
        // Output point 24: X(23)
        out[out_strides[23]] = a_s60 + a_s51;

        // Output point 20: X(19)
        out[out_strides[19]] = a_s58;
        // Output point 21: X(20)
        out[out_strides[20]] = a_s57;

        // Output point 9: X(8)
        out[out_strides[8]] = a_m20 + a_s56;
        // Output point 25: X(24)
        out[out_strides[24]] = a_s56 - a_m20;

        // Shifted DFT
        FFTZ_DOUBLE b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7,
                    b_in8, b_in9, b_in10, b_in11, b_in12;
        FFTZ_DOUBLE b_m0, b_m1, b_m2, b_m3, b_m4, b_m5, b_m6, b_m7, b_m8, b_m9,
                    b_m10, b_m11, b_m12, b_m13, b_m14, b_m15, b_m16, b_m17,
                    b_m18, b_m19, b_m20, b_m21, b_m22, b_m23, b_m24, b_m25,
                    b_m26,
                    b_m27, b_m28, b_m29, b_m30, b_m31, b_m32, b_m33, b_m34,
                    b_m35, b_m36, b_m37, b_m38, b_m39, b_m40, b_m41, b_m42,
                    b_m43, b_m44, b_m45, b_m46, b_m47, b_m48, b_m49, b_m50,
                    b_m51, b_m52, b_m53, b_m54, b_m55, b_m56, b_m57, b_m58,
                    b_m59, b_m60, b_m61, b_m62, b_m63, b_m64, b_m65, b_m66,
                    b_m67, b_m68, b_m69, b_m70, b_m71;
        FFTZ_DOUBLE b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9,
                    b_s10, b_s11, b_s12, b_s13, b_s14, b_s15, b_s16, b_s17,
                    b_s18, b_s19, b_s20, b_s21, b_s22, b_s23, b_s24, b_s25,
                    b_s26, b_s27, b_s28, b_s29, b_s30, b_s31, b_s32, b_s33,
                    b_s34, b_s35, b_s36, b_s37, b_s38, b_s39, b_s40, b_s41,
                    b_s42, b_s43, b_s44, b_s45, b_s46, b_s47, b_s48, b_s49,
                    b_s50, b_s51, b_s52, b_s53, b_s54, b_s55, b_s56, b_s57,
                    b_s58, b_s59, b_s60, b_s61, b_s62, b_s63, b_s64, b_s65,
                    b_s66, b_s67, b_s68, b_s69, b_s70, b_s71, b_s72, b_s73,
                    b_s74, b_s75, b_s76;

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

        b_s0 = b_in1 + b_in12;
        b_s1 = b_in1 - b_in12;
        b_s2 = b_in2 + b_in11;
        b_s3 = b_in2 - b_in11;
        b_s4 = b_in3 + b_in10;
        b_s5 = b_in3 - b_in10;
        b_s6 = b_in4 + b_in9;
        b_s7 = b_in4 - b_in9;
        b_s8 = b_in5 + b_in8;
        b_s9 = b_in5 - b_in8;
        b_s10 = b_in6 + b_in7;
        b_s11 = b_in6 - b_in7;

        b_m0 = R13_DFT_C1 * b_s1;
        b_m1 = R13_DFT_C2 * b_s3;
        b_s12 = b_m0 + b_m1;
        b_m2 = R13_DFT_C3 * b_s5;
        b_m3 = R13_DFT_C4 * b_s7;
        b_s13 = b_m2 + b_m3;
        b_m4 = R13_DFT_C5 * b_s9;
        b_m5 = R13_DFT_C6 * b_s11;
        b_s14 = b_m4 + b_m5;
        b_s15 = b_s12 + b_s13;
        b_s16 = b_s15 + b_s14;
        // Output point 2: X(1)
        out[out_strides[1]] = b_in0 + b_s16;

        b_m6 = R13_DFT_S1 * b_s0;
        b_m7 = R13_DFT_S2 * b_s2;
        b_s17 = b_m6 + b_m7;
        b_m8 = R13_DFT_S3 * b_s4;
        b_m9 = R13_DFT_S4 * b_s6;
        b_s18 = b_m8 + b_m9;
        b_m10 = R13_DFT_S5 * b_s8;
        b_m11 = R13_DFT_S6 * b_s10;
        b_s19 = b_m10 + b_m11;
        b_s20 = b_s17 + b_s18;
        b_s21 = b_s20 + b_s19;
        // Output point 3: X(2)
        out[out_strides[2]] = -b_s21;

        b_m12 = R13_DFT_C3 * b_s1;
        b_m13 = R13_DFT_C6 * b_s3;
        b_s22 = b_m12 + b_m13;
        b_m14 = R13_DFT_C4 * b_s5;
        b_m15 = R13_DFT_C1 * b_s7;
        b_s23 = b_m14 + b_m15;
        b_m16 = R13_DFT_C2 * b_s9;
        b_m17 = R13_DFT_C5 * b_s11;
        b_s24 = b_m16 + b_m17;
        b_s25 = b_s23 + b_s24;
        b_s26 = b_s22 - b_s25;
        // Output point 6: X(5)
        out[out_strides[5]] = b_in0 + b_s26;

        b_m18 = R13_DFT_S3 * b_s0;
        b_m19 = R13_DFT_S6 * b_s2;
        b_s27 = b_m18 + b_m19;
        b_m20 = R13_DFT_S4 * b_s4;
        b_m21 = R13_DFT_S1 * b_s6;
        b_s28 = b_m20 + b_m21;
        b_m22 = R13_DFT_S2 * b_s8;
        b_m23 = R13_DFT_S5 * b_s10;
        b_s29 = b_m22 + b_m23;
        b_s30 = b_s27 + b_s28;
        b_s31 = b_s29 - b_s30;
        // Output point 7: X(6)
        out[out_strides[6]] = b_s31;

        b_m24 = R13_DFT_C5 * b_s1;
        b_m25 = R13_DFT_C3 * b_s3;
        b_s32 = b_m24 - b_m25;
        b_m26 = R13_DFT_C2 * b_s5;
        b_m27 = R13_DFT_C6 * b_s7;
        b_s33 = b_m27 - b_m26;
        b_m28 = R13_DFT_C1 * b_s9;
        b_m29 = R13_DFT_C4 * b_s11;
        b_s34 = b_m28 + b_m29;
        b_s35 = b_s32 + b_s33;
        b_s36 = b_s35 + b_s34;
        // Output point 10: X(9)
        out[out_strides[9]] = b_in0 + b_s36;

        b_m30 = R13_DFT_S5 * b_s0;
        b_m31 = R13_DFT_S3 * b_s2;
        b_s37 = b_m30 + b_m31;
        b_m32 = R13_DFT_S2 * b_s4;
        b_m33 = R13_DFT_S6 * b_s6;
        b_s38 = b_m32 + b_m33;
        b_m34 = R13_DFT_S1 * b_s8;
        b_m35 = R13_DFT_S4 * b_s10;
        b_s39 = b_m34 - b_m35;
        b_s40 = b_s38 + b_s39;
        b_s41 = b_s40 - b_s37;
        // Output point 11: X(10)
        out[out_strides[10]] = b_s41;

        b_m36 = R13_DFT_C6 * b_s1;
        b_m37 = R13_DFT_C1 * b_s3;
        b_s42 = b_m36 + b_m37;
        b_m38 = R13_DFT_C5 * b_s5;
        b_m39 = R13_DFT_C2 * b_s7;
        b_s43 = b_m38 + b_m39;
        b_m40 = R13_DFT_C4 * b_s9;
        b_m41 = R13_DFT_C3 * b_s11;
        b_s44 = b_m40 + b_m41;
        b_s45 = b_s42 + b_s44;
        b_s46 = b_s43 - b_s45;
        // Output point 14: X(13)
        out[out_strides[13]] = b_in0 + b_s46;

        b_m42 = R13_DFT_S6 * b_s0;
        b_m43 = R13_DFT_S1 * b_s2;
        b_s47 = b_m43 - b_m42;
        b_m44 = R13_DFT_S5 * b_s4;
        b_m45 = R13_DFT_S2 * b_s6;
        b_s48 = b_m44 - b_m45;
        b_m46 = R13_DFT_S4 * b_s8;
        b_m47 = R13_DFT_S3 * b_s10;
        b_s49 = b_m47 - b_m46;
        b_s50 = b_s47 + b_s48;
        b_s51 = b_s50 + b_s49;
        // Output point 15: X(14)
        out[out_strides[14]] = b_s51;

        b_m48 = R13_DFT_C4 * b_s1;
        b_m49 = R13_DFT_C5 * b_s3;
        b_s52 = b_m48 + b_m49;
        b_m50 = R13_DFT_C1 * b_s5;
        b_m51 = R13_DFT_C3 * b_s7;
        b_s53 = b_m50 - b_m51;
        b_m52 = R13_DFT_C6 * b_s9;
        b_m53 = R13_DFT_C2 * b_s11;
        b_s54 = b_m53 - b_m52;
        b_s55 = b_s53 + b_s54;
        b_s56 = b_s55 - b_s52;
        // Output point 18: X(17)
        out[out_strides[17]] = b_in0 + b_s56;

        b_m54 = R13_DFT_S4 * b_s0;
        b_m55 = R13_DFT_S5 * b_s2;
        b_s57 = b_m55 - b_m54;
        b_m56 = R13_DFT_S1 * b_s4;
        b_m57 = R13_DFT_S3 * b_s6;
        b_s58 = b_m56 + b_m57;
        b_m58 = R13_DFT_S6 * b_s8;
        b_m59 = R13_DFT_S2 * b_s10;
        b_s59 = b_m58 - b_m59;
        b_s60 = b_s57 + b_s59;
        b_s61 = b_s60 - b_s58;
        // Output point 19: X(18)
        out[out_strides[18]] = b_s61;

        b_m60 = R13_DFT_C2 * b_s1;
        b_m61 = R13_DFT_C4 * b_s3;
        b_s62 = b_m61 - b_m60;
        b_m62 = R13_DFT_C6 * b_s5;
        b_m63 = R13_DFT_C5 * b_s7;
        b_s63 = b_m62 + b_m63;
        b_m64 = R13_DFT_C3 * b_s9;
        b_m65 = R13_DFT_C1 * b_s11;
        b_s64 = b_m64 - b_m65;
        b_s65 = b_s62 + b_s64;
        b_s66 = b_s65 - b_s63;
        // Output point 22: X(21)
        out[out_strides[21]] = b_in0 + b_s66;

        b_m66 = R13_DFT_S2 * b_s0;
        b_m67 = R13_DFT_S4 * b_s2;
        b_s67 = b_m67 - b_m66;
        b_m68 = R13_DFT_S6 * b_s4;
        b_m69 = R13_DFT_S5 * b_s6;
        b_s68 = b_m69 - b_m68;
        b_m70 = R13_DFT_S3 * b_s8;
        b_m71 = R13_DFT_S1 * b_s10;
        b_s69 = b_m71 - b_m70;
        b_s70 = b_s67 + b_s68;
        b_s71 = b_s70 + b_s69;
        // Output point 23: X(22)
        out[out_strides[22]] = b_s71;

        b_s72 = b_s3 - b_s1;
        b_s73 = b_s7 - b_s5;
        b_s74 = b_s11 - b_s9;
        b_s75 = b_s72 + b_s73;
        b_s76 = b_s75 + b_s74;
        // Output point 26: X(25)
        out[out_strides[25]] = b_in0 + b_s76;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft13c_fp64_bwd(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                        FFTZ_VOID *out_real,
                                        FFTZ_VOID *out_imag, FFTZ_INTP n,
                                        aoclfftz_strides_t *strides,
                                        FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_13_1 =
        1.732050807568877293527446341505872366942805254;
    const FFTZ_DOUBLE CRTM_13_2 =
        1.007074065727533254493747707736933954186697125;
    const FFTZ_DOUBLE CRTM_13_3 =
        0.531932498429674575175042127684371897596660533;
    const FFTZ_DOUBLE CRTM_13_4 =
        1.150281458948006242736771094910906776922003215;
    const FFTZ_DOUBLE CRTM_13_5 =
        0.348277202304271810011321589858529485233929352;
    const FFTZ_DOUBLE CRTM_13_6 =
        0.500000000000000000000000000000000000000000000;
    const FFTZ_DOUBLE CRTM_13_7 =
        2.000000000000000000000000000000000000000000000;
    const FFTZ_DOUBLE R13_DGC_1 =
        0.166666666666666666666666666666666666666666667;
    const FFTZ_DOUBLE R13_DGC_2 =
        0.256247671582936600958684654061725059144125175;
    const FFTZ_DOUBLE R13_DGC_3 =
        0.156891391051584611046832726756003269660212636;
    const FFTZ_DOUBLE R13_DGC_4 =
        0.774781170935234584261351932853525703557550433;
    const FFTZ_DOUBLE R13_DGC_5 =
        0.265966249214837287587521063842185948798330267;
    const FFTZ_DOUBLE R13_DGC_6 =
        0.600925212577331548853203544578415991041882762;
    const FFTZ_DOUBLE R13_DGC_7 =
        0.151805972074387731966205794490207080712856746;
    const FFTZ_DOUBLE R13_DGC_8 =
        0.227708958111581597949308691735310621069285120;
    const FFTZ_DOUBLE R13_DGC_9 =
        0.503537032863766627246873853868466977093348562;
    const FFTZ_DOUBLE R13_DGC_10 =
        0.300238635966332641462884626667381504676006424;
    const FFTZ_DOUBLE R13_DGC_11 =
        0.011599105605768290721655456654083252189827041;
    const FFTZ_DOUBLE R13_DGC_12 =
        0.516520780623489722840901288569017135705033622;
    const FFTZ_DOUBLE R13_DFT_C1 =
        1.94188363485210405431396455258757845449973021147800;
    const FFTZ_DOUBLE R13_DFT_C2 =
        1.77091205130641979180075104403019775721099683269510;
    const FFTZ_DOUBLE R13_DFT_C3 =
        1.49702149634220219726926119940270276769290318035170;
    const FFTZ_DOUBLE R13_DFT_C4 =
        1.13612949346231160502361511825503324906698510490700;
    const FFTZ_DOUBLE R13_DFT_C5 =
        0.70920977408507125193927578520003694863271086422760;
    const FFTZ_DOUBLE R13_DFT_C6 =
        0.24107336051064610669813537490508716454736231845510;
    const FFTZ_DOUBLE R13_DFT_S1 =
        0.47863132857511553429750745252042379040634604547660;
    const FFTZ_DOUBLE R13_DFT_S2 =
        0.92944634408753709131203067026620955511547173066490;
    const FFTZ_DOUBLE R13_DFT_S3 =
        1.32624531648159040475357098533353255904952821408820;
    const FFTZ_DOUBLE R13_DFT_S4 =
        1.64596773178731278915923484687876398131013538617510;
    const FFTZ_DOUBLE R13_DFT_S5 =
        1.87003248537082964687956919967566145810103493915690;
    const FFTZ_DOUBLE R13_DFT_S6 =
        1.98541774819610798560150329898504035868735126594030;

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
        // Standard DFT
        FFTZ_DOUBLE a_in0, a_in1, a_in2, a_in3, a_in4, a_in5, a_in6, a_in7,
                    a_in8, a_in9, a_in10, a_in11, a_in12;
        FFTZ_DOUBLE a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_s9,
                    a_s10, a_s11, a_s12, a_s13, a_s14, a_s15, a_s16, a_s17,
                    a_s18, a_s19, a_s20, a_s21, a_s22, a_s23, a_s24, a_s25,
                    a_s26, a_s27, a_s28, a_s29, a_s30, a_s31, a_s32, a_s33,
                    a_s34, a_s35, a_s36, a_s37, a_s38, a_s39, a_s40, a_s41,
                    a_s42, a_s43, a_s44, a_s45, a_s46, a_s47, a_s48, a_s49,
                    a_s50, a_s51, a_s52, a_s53, a_s54, a_s55, a_s56, a_s57,
                    a_s58, a_s59, a_s60, a_s61, a_s62;
        FFTZ_DOUBLE a_m0, a_m1, a_m2, a_m3, a_m4, a_m5, a_m6, a_m7, a_m8, a_m9,
                    a_m10, a_m11, a_m12, a_m13, a_m14, a_m15, a_m16, a_m17,
                    a_m18, a_m19, a_m20, a_m21, a_m22, a_m23, a_m24, a_m25,
                    a_m26, a_m27, a_m28, a_m29, a_m30, a_m31, a_m32, a_m33,
                    a_m34;

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
        // Input point 24: X(23)
        a_in11 = in[in_strides[23]];
        // Input point 25: X(24)
        a_in12 = in[in_strides[24]];
        a_s0 = a_in6 - a_in8;
        a_m0 = CRTM_13_7 * a_in2;
        a_s1 = a_m0 - a_s0;
        a_s59 = a_in6 + a_in8;
        a_m1 = CRTM_13_1 * a_s59;
        a_s2 = a_in12 + a_in4;
        a_s60 = a_in12 - a_in4;
        a_m2 = CRTM_13_1 * a_s60;
        a_m3 = CRTM_13_7 * a_in10;
        a_s3 = a_s2 - a_m3;
        a_s4 = a_s1 + a_m2;
        a_s5 = a_s3 - a_m1;
        a_m4 = R13_DGC_11 * a_s4;
        a_m5 = R13_DGC_10 * a_s5;
        a_s6 = a_m4 + a_m5;
        a_m6 = R13_DGC_10 * a_s4;
        a_m7 = R13_DGC_11 * a_s5;
        a_s7 = a_m6 - a_m7;
        a_s8 = a_in2 + a_s0;
        a_s9 = a_s2 + a_in10;
        a_m8 = CRTM_13_4 * a_s8;
        a_m9 = CRTM_13_5 * a_s9;
        a_s10 = a_m8 - a_m9;
        a_m10 = CRTM_13_5 * a_s8;
        a_m11 = CRTM_13_4 * a_s9;
        a_s11 = a_m10 + a_m11;
        a_s12 = a_s1 - a_m2;
        a_s13 = a_m1 + a_s3;
        a_m12 = R13_DGC_3 * a_s12;
        a_m13 = R13_DGC_2 * a_s13;
        a_s14 = a_m12 + a_m13;
        a_m14 = R13_DGC_3 * a_s13;
        a_m15 = R13_DGC_2 * a_s12;
        a_s15 = a_m14 - a_m15;
        a_s16 = a_in3 + a_in11;
        a_s17 = a_in9 + a_s16;
        a_m16 = CRTM_13_6 * a_s16;
        a_s18 = a_in9 - a_m16;
        a_s19 = a_in3 - a_in11;
        a_s20 = a_in5 + a_in7;
        a_s21 = a_in1 + a_s20;
        a_m17 = CRTM_13_6 * a_s20;
        a_s22 = a_in1 - a_m17;
        a_s23 = a_in5 - a_in7;
        a_s24 = a_s21 - a_s17;
        a_m18 = R13_DGC_6 * a_s24;
        a_s25 = a_s21 + a_s17;
        a_m33 = CRTM_13_7 * a_s25;
        // Output point 1: x(0)
        *out = a_m33 + a_in0;

        a_m34 = R13_DGC_1 * a_s25;
        a_s26 = a_in0 - a_m34;
        a_s27 = a_s23 + a_s19;
        a_s28 = a_s22 + a_s18;
        a_m19 = R13_DGC_9 * a_s27;
        a_m20 = R13_DGC_7 * a_s28;
        a_s29 = a_m19 + a_m20;
        a_s30 = a_s22 - a_s18;
        a_s31 = a_s23 - a_s19;
        a_m21 = R13_DGC_12 * a_s30;
        a_m22 = R13_DGC_5 * a_s31;
        a_s32 = a_m21 - a_m22;
        a_s61 = a_s6 + a_s14;
        a_m23 = CRTM_13_1 * a_s61;
        a_s62 = a_s7 - a_s15;
        a_m24 = CRTM_13_1 * a_s62;
        a_s33 = a_s7 + a_s15;
        a_s34 = a_s10 - a_s33;
        a_m25 = CRTM_13_7 * a_s33;
        a_s35 = a_m25 + a_s10;
        a_s36 = a_s6 - a_s14;
        a_m26 = CRTM_13_7 * a_s36;
        a_s37 = a_m26 - a_s11;
        a_s38 = a_s36 + a_s11;
        a_m27 = R13_DGC_4 * a_s31;
        a_m28 = CRTM_13_3 * a_s30;
        a_s39 = a_m27 + a_m28;
        a_m29 = R13_DGC_8 * a_s27;
        a_m30 = CRTM_13_2 * a_s28;
        a_s40 = a_m29 - a_m30;
        a_s41 = a_s39 - a_s40;
        a_s42 = a_s39 + a_s40;
        a_s43 = a_s26 - a_s29;
        a_s44 = a_m18 - a_s32;
        a_s45 = a_s43 - a_s44;
        a_s46 = a_s44 + a_s43;
        a_m31 = CRTM_13_7 * a_s29;
        a_s47 = a_m31 + a_s26;
        a_m32 = CRTM_13_7 * a_s32;
        a_s48 = a_m32 + a_m18;
        a_s49 = a_s47 - a_s48;
        // Output point 17: x(16)
        out[out_strides[16]] = a_s49 + a_s35;
        // Output point 11: x(10)
        out[out_strides[10]] = a_s49 - a_s35;

        a_s50 = a_s48 + a_s47;
        // Output point 25: x(24)
        out[out_strides[24]] = a_s50 - a_s37;
        // Output point 3: x(2)
        out[out_strides[2]] = a_s50 + a_s37;

        a_s51 = a_s45 - a_m23;
        a_s52 = a_s41 - a_s34;
        // Output point 5: x(4)
        out[out_strides[4]] = a_s51 + a_s52;
        // Output point 15: x(14)
        out[out_strides[14]] = a_s51 - a_s52;

        a_s53 = a_s46 - a_s38;
        a_s54 = a_s42 + a_m24;
        // Output point 7: x(6)
        out[out_strides[6]] = a_s53 - a_s54;
        // Output point 19: x(18)
        out[out_strides[18]] = a_s54 + a_s53;

        a_s55 = a_s42 - a_m24;
        a_s56 = a_s46 + a_s38;
        // Output point 9: x(8)
        out[out_strides[8]] = a_s55 + a_s56;
        // Output point 21: x(20)
        out[out_strides[20]] = a_s56 - a_s55;

        a_s57 = a_s45 + a_m23;
        a_s58 = a_s41 + a_s34;
        // Output point 13: x(12)
        out[out_strides[12]] = a_s57 - a_s58;
        // Output point 23: x(22)
        out[out_strides[22]] = a_s57 + a_s58;

        // Shifted DFT
        FFTZ_DOUBLE b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7,
                    b_in8, b_in9, b_in10, b_in11, b_in12;
        FFTZ_DOUBLE b_m0, b_m1, b_m2, b_m3, b_m4, b_m5, b_m6, b_m7, b_m8, b_m9,
                    b_m10, b_m11, b_m12, b_m13, b_m14, b_m15, b_m16, b_m17,
                    b_m18, b_m19, b_m20, b_m21, b_m22, b_m23, b_m24, b_m25,
                    b_m26, b_m27, b_m28, b_m29, b_m30, b_m31, b_m32, b_m33,
                    b_m34, b_m35, b_m36, b_m37, b_m38, b_m39, b_m40, b_m41,
                    b_m42, b_m43, b_m44, b_m45, b_m46, b_m47, b_m48, b_m49,
                    b_m50, b_m51, b_m52, b_m53, b_m54, b_m55, b_m56, b_m57,
                    b_m58, b_m59, b_m60, b_m61, b_m62, b_m63, b_m64, b_m65,
                    b_m66, b_m67, b_m68, b_m69, b_m70, b_m71;
        FFTZ_DOUBLE b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9,
                    b_s10, b_s11, b_s12, b_s13, b_s14, b_s15, b_s16, b_s17,
                    b_s18, b_s19, b_s20, b_s21, b_s22, b_s23, b_s24, b_s25,
                    b_s26, b_s27, b_s28, b_s29, b_s30, b_s31, b_s32, b_s33,
                    b_s34, b_s35, b_s36, b_s37, b_s38, b_s39, b_s40, b_s41,
                    b_s42, b_s43, b_s44, b_s45, b_s46, b_s47, b_s48, b_s49,
                    b_s50, b_s51, b_s52, b_s53, b_s54, b_s55, b_s56, b_s57,
                    b_s58, b_s59, b_s60, b_s61, b_s62, b_s63, b_s64, b_s65,
                    b_s66, b_s67, b_s68, b_s69, b_s70, b_s71;

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
        // Input point 23: X(22)
        b_in11 = in[in_strides[22]];
        // Input point 26: X(25)
        b_in12 = in[in_strides[25]];

        // DC output (n=0)
        b_s0 = b_in0 + b_in2;
        b_s1 = b_in4 + b_in6;
        b_s2 = b_in8 + b_in10;
        b_s3 = b_s0 + b_s1;
        b_s4 = b_s3 + b_s2;
        b_s71 = b_s4 + b_s4;
        // Output point 2: x(1)
        out[out_strides[1]] = b_s71 + b_in12;

        b_m0 = R13_DFT_C1 * b_in0;
        b_m1 = R13_DFT_C3 * b_in2;
        b_s5 = b_m0 + b_m1;
        b_m2 = R13_DFT_C5 * b_in4;
        b_m3 = R13_DFT_C6 * b_in6;
        b_s6 = b_m2 - b_m3;
        b_m4 = R13_DFT_C4 * b_in8;
        b_m5 = R13_DFT_C2 * b_in10;
        b_s7 = b_m4 + b_m5;
        b_s8 = b_s5 + b_s6;
        b_s9 = b_s8 - b_s7;
        b_s10 = b_s9 - b_in12;

        b_m6 = R13_DFT_S1 * b_in1;
        b_m7 = R13_DFT_S3 * b_in3;
        b_s11 = b_m6 + b_m7;
        b_m8 = R13_DFT_S5 * b_in5;
        b_m9 = R13_DFT_S6 * b_in7;
        b_s12 = b_m8 + b_m9;
        b_m10 = R13_DFT_S4 * b_in9;
        b_m11 = R13_DFT_S2 * b_in11;
        b_s13 = b_m10 + b_m11;
        b_s14 = b_s11 + b_s12;
        b_s15 = b_s14 + b_s13;

        // Output point 4: x(3)
        out[out_strides[3]] = b_s10 - b_s15;
        // Output point 26: x(25)
        out[out_strides[25]] = -b_s10 - b_s15;

        b_m12 = R13_DFT_C2 * b_in0;
        b_m13 = R13_DFT_C6 * b_in2;
        b_s16 = b_m12 + b_m13;
        b_m14 = R13_DFT_C3 * b_in4;
        b_m15 = R13_DFT_C1 * b_in6;
        b_s17 = b_m14 + b_m15;
        b_m16 = R13_DFT_C5 * b_in8;
        b_m17 = R13_DFT_C4 * b_in10;
        b_s18 = b_m17 - b_m16;
        b_s19 = b_s16 - b_s17;
        b_s20 = b_s19 + b_s18;
        b_s21 = b_s20 + b_in12;

        b_m18 = R13_DFT_S2 * b_in1;
        b_m19 = R13_DFT_S6 * b_in3;
        b_s22 = b_m18 + b_m19;
        b_m20 = R13_DFT_S3 * b_in5;
        b_m21 = R13_DFT_S1 * b_in7;
        b_s23 = b_m20 - b_m21;
        b_m22 = R13_DFT_S5 * b_in9;
        b_m23 = R13_DFT_S4 * b_in11;
        b_s24 = b_m22 + b_m23;
        b_s25 = b_s22 + b_s23;
        b_s26 = b_s25 - b_s24;

        // Output point 6: x(5)
        out[out_strides[5]] = b_s21 - b_s26;
        // Output point 24: x(23)
        out[out_strides[23]] = -b_s21 - b_s26;

        b_m24 = R13_DFT_C3 * b_in0;
        b_m25 = R13_DFT_C4 * b_in2;
        b_s27 = b_m24 - b_m25;
        b_m26 = R13_DFT_C2 * b_in4;
        b_m27 = R13_DFT_C5 * b_in6;
        b_s28 = b_m27 - b_m26;
        b_m28 = R13_DFT_C1 * b_in8;
        b_m29 = R13_DFT_C6 * b_in10;
        b_s29 = b_m28 - b_m29;
        b_s30 = b_s27 + b_s28;
        b_s31 = b_s30 + b_s29;
        b_s32 = b_s31 - b_in12;

        b_m30 = R13_DFT_S3 * b_in1;
        b_m31 = R13_DFT_S4 * b_in3;
        b_s33 = b_m30 + b_m31;
        b_m32 = R13_DFT_S2 * b_in5;
        b_m33 = R13_DFT_S5 * b_in7;
        b_s34 = b_m32 + b_m33;
        b_m34 = R13_DFT_S1 * b_in9;
        b_m35 = R13_DFT_S6 * b_in11;
        b_s35 = b_m34 + b_m35;
        b_s36 = b_s33 - b_s34;
        b_s37 = b_s36 + b_s35;

        // Output point 8: x(7)
        out[out_strides[7]] = b_s32 - b_s37;
        // Output point 22: x(21)
        out[out_strides[21]] = -b_s32 - b_s37;

        b_m36 = R13_DFT_C4 * b_in0;
        b_m37 = R13_DFT_C1 * b_in2;
        b_s38 = b_m36 - b_m37;
        b_m38 = R13_DFT_C6 * b_in4;
        b_m39 = R13_DFT_C2 * b_in6;
        b_s39 = b_m38 + b_m39;
        b_m40 = R13_DFT_C3 * b_in8;
        b_m41 = R13_DFT_C5 * b_in10;
        b_s40 = b_m40 + b_m41;
        b_s41 = b_s38 + b_s39;
        b_s42 = b_s41 - b_s40;
        b_s43 = b_s42 + b_in12;

        b_m42 = R13_DFT_S4 * b_in1;
        b_m43 = R13_DFT_S1 * b_in3;
        b_s44 = b_m42 + b_m43;
        b_m44 = R13_DFT_S6 * b_in5;
        b_m45 = R13_DFT_S2 * b_in7;
        b_s45 = b_m45 - b_m44;
        b_m46 = R13_DFT_S3 * b_in9;
        b_m47 = R13_DFT_S5 * b_in11;
        b_s46 = b_m46 - b_m47;
        b_s47 = b_s44 + b_s45;
        b_s48 = b_s47 + b_s46;

        // Output point 10: x(9)
        out[out_strides[9]] = b_s43 - b_s48;
        // Output point 20: x(19)
        out[out_strides[19]] = -b_s43 - b_s48;

        b_m48 = R13_DFT_C5 * b_in0;
        b_m49 = R13_DFT_C2 * b_in2;
        b_s49 = b_m48 - b_m49;
        b_m50 = R13_DFT_C1 * b_in4;
        b_m51 = R13_DFT_C4 * b_in6;
        b_s50 = b_m50 - b_m51;
        b_m52 = R13_DFT_C6 * b_in8;
        b_m53 = R13_DFT_C3 * b_in10;
        b_s51 = b_m53 - b_m52;
        b_s52 = b_s49 + b_s50;
        b_s53 = b_s52 + b_s51;
        b_s54 = b_s53 - b_in12;

        b_m54 = R13_DFT_S5 * b_in1;
        b_m55 = R13_DFT_S2 * b_in3;
        b_s55 = b_m54 - b_m55;
        b_m56 = R13_DFT_S1 * b_in5;
        b_m57 = R13_DFT_S4 * b_in7;
        b_s56 = b_m57 - b_m56;
        b_m58 = R13_DFT_S6 * b_in9;
        b_m59 = R13_DFT_S3 * b_in11;
        b_s57 = b_m59 - b_m58;
        b_s58 = b_s55 + b_s56;
        b_s59 = b_s58 + b_s57;

        // Output point 12: x(11)
        out[out_strides[11]] = b_s54 - b_s59;
        // Output point 18: x(17)
        out[out_strides[17]] = -b_s54 - b_s59;

        b_m60 = R13_DFT_C6 * b_in0;
        b_m61 = R13_DFT_C5 * b_in2;
        b_s60 = b_m60 - b_m61;
        b_m62 = R13_DFT_C4 * b_in4;
        b_m63 = R13_DFT_C3 * b_in6;
        b_s61 = b_m62 - b_m63;
        b_m64 = R13_DFT_C2 * b_in8;
        b_m65 = R13_DFT_C1 * b_in10;
        b_s62 = b_m64 - b_m65;
        b_s63 = b_s60 + b_s61;
        b_s64 = b_s63 + b_s62;
        b_s65 = b_s64 + b_in12;

        b_m66 = R13_DFT_S6 * b_in1;
        b_m67 = R13_DFT_S5 * b_in3;
        b_s66 = b_m66 - b_m67;
        b_m68 = R13_DFT_S4 * b_in5;
        b_m69 = R13_DFT_S3 * b_in7;
        b_s67 = b_m68 - b_m69;
        b_m70 = R13_DFT_S2 * b_in9;
        b_m71 = R13_DFT_S1 * b_in11;
        b_s68 = b_m70 - b_m71;
        b_s69 = b_s66 + b_s67;
        b_s70 = b_s69 + b_s68;

        // Output point 14: x(13)
        out[out_strides[13]] = b_s65 - b_s70;
        // Output point 16: x(15)
        out[out_strides[15]] = -b_s65 - b_s70;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hcf_rfft13c(FFTZ_UINT8 precision, FFTZ_UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hcf_rfft13c_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft13c_fp64_fwd;
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
            return r2hcf_rfft13c_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft13c_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
