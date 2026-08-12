// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft13avx128.c
 *
 *  @brief Radix-13 r2hc_fused Real-FFT kernel with AVX-128 operations using
 *  x86 SIMD intrinsics.
 *
 *  This file contains the DIT radix-13 real-to-halfcomplex fused of two
 *  different implementations (Standard DFT and Shifted DFT that differs in DFT
 *  weight matrix) using x86 SIMD intrinsics for single-precision and
 *  double-precision inputs
 *
 *  @author Amrin Fathima
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/r2hc_simd_avx128.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] = {
                                                    {{0, 106, 166, 52, 0, 0},
                                                     {0, 107, 167, 52, 0, 0}},
                                                    {{0, 106, 166, 52, 0, 0},
                                                     {0, 107, 167, 52, 0, 0}}};

ops_cycles_t get_ops_cnt_r2hcf_rfft13avx128(FFTZ_UINT8 precision,
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

static FFTZ_VOID r2hcf_rfft13avx128_fp32_fwd(FFTZ_VOID *in_real,
                                             FFTZ_VOID *in_imag,
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
    FFTZ_FLOAT *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_128_S;
    FFTZ_INTP remaining_sets = n % NUM_SETS_REAL_128_S;

    __m128 v_CRTM_13_1 = _mm_broadcast_ss(&CRTM_13_1);
    __m128 v_CRTM_13_2 = _mm_broadcast_ss(&CRTM_13_2);
    __m128 v_CRTM_13_3 = _mm_broadcast_ss(&CRTM_13_3);
    __m128 v_CRTM_13_4 = _mm_broadcast_ss(&CRTM_13_4);
    __m128 v_CRTM_13_5 = _mm_broadcast_ss(&CRTM_13_5);
    __m128 v_CRTM_13_6 = _mm_broadcast_ss(&CRTM_13_6);
    __m128 v_CRTM_13_7 = _mm_broadcast_ss(&CRTM_13_7);
    __m128 v_CRTM_13_8 = _mm_broadcast_ss(&CRTM_13_8);
    __m128 v_R13_DGC_1 = _mm_broadcast_ss(&R13_DGC_1);
    __m128 v_R13_DGC_2 = _mm_broadcast_ss(&R13_DGC_2);
    __m128 v_R13_DGC_3 = _mm_broadcast_ss(&R13_DGC_3);
    __m128 v_R13_DGC_4 = _mm_broadcast_ss(&R13_DGC_4);
    __m128 v_R13_DGC_5 = _mm_broadcast_ss(&R13_DGC_5);
    __m128 v_R13_DGC_6 = _mm_broadcast_ss(&R13_DGC_6);
    __m128 v_R13_DGC_7 = _mm_broadcast_ss(&R13_DGC_7);
    __m128 v_R13_DGC_8 = _mm_broadcast_ss(&R13_DGC_8);
    __m128 v_R13_DGC_9 = _mm_broadcast_ss(&R13_DGC_9);
    __m128 v_R13_DGC_10 = _mm_broadcast_ss(&R13_DGC_10);
    __m128 v_R13_DGC_11 = _mm_broadcast_ss(&R13_DGC_11);
    __m128 v_R13_DGC_12 = _mm_broadcast_ss(&R13_DGC_12);
    __m128 v_R13_DFT_C1 = _mm_broadcast_ss(&R13_DFT_C1);
    __m128 v_R13_DFT_C2 = _mm_broadcast_ss(&R13_DFT_C2);
    __m128 v_R13_DFT_C3 = _mm_broadcast_ss(&R13_DFT_C3);
    __m128 v_R13_DFT_C4 = _mm_broadcast_ss(&R13_DFT_C4);
    __m128 v_R13_DFT_C5 = _mm_broadcast_ss(&R13_DFT_C5);
    __m128 v_R13_DFT_C6 = _mm_broadcast_ss(&R13_DFT_C6);
    __m128 v_R13_DFT_S1 = _mm_broadcast_ss(&R13_DFT_S1);
    __m128 v_R13_DFT_S2 = _mm_broadcast_ss(&R13_DFT_S2);
    __m128 v_R13_DFT_S3 = _mm_broadcast_ss(&R13_DFT_S3);
    __m128 v_R13_DFT_S4 = _mm_broadcast_ss(&R13_DFT_S4);
    __m128 v_R13_DFT_S5 = _mm_broadcast_ss(&R13_DFT_S5);
    __m128 v_R13_DFT_S6 = _mm_broadcast_ss(&R13_DFT_S6);

    for (cnt = 0; cnt < N; cnt++)
    {
        // Standard DFT
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
               av_in8, av_in9, av_in10, av_in11, av_in12;
        __m128 av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
               av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16,
               av_s17, av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24,
               av_s25, av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32,
               av_s33, av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40,
               av_s41, av_s42, av_s43, av_s44, av_s45, av_s46, av_s47, av_s48,
               av_s49, av_s50, av_s51, av_s52, av_s53, av_s54, av_s55, av_s56,
               av_s57, av_s58, av_s59, av_s60, av_s61, av_s62, av_s63, av_s64;
        __m128 av_m0, av_m1, av_m2, av_m3, av_m4, av_m5, av_m6, av_m7, av_m8,
               av_m9, av_m10, av_m11, av_m12, av_m13, av_m14, av_m15, av_m16,
               av_m17, av_m18, av_m19, av_m20, av_m21, av_m22, av_m23, av_m24,
               av_m25, av_m26, av_m27, av_m28, av_m29, av_m30, av_m31, av_m32,
               av_m33;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21,
               v_out22, v_out23, v_out24, v_out25;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_stride, av_in0);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_128_S(curr_in, v_in_stride, av_in1);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_128_S(curr_in, v_in_stride, av_in2);
        // Input point 7: x(6)
        curr_in = in + in_strides[6];
        LDR_128_S(curr_in, v_in_stride, av_in3);
        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDR_128_S(curr_in, v_in_stride, av_in4);
        // Input point 11: x(10)
        curr_in = in + in_strides[10];
        LDR_128_S(curr_in, v_in_stride, av_in5);
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDR_128_S(curr_in, v_in_stride, av_in6);
        // Input point 15: x(14)
        curr_in = in + in_strides[14];
        LDR_128_S(curr_in, v_in_stride, av_in7);
        // Input point 17: x(16)
        curr_in = in + in_strides[16];
        LDR_128_S(curr_in, v_in_stride, av_in8);
        // Input point 19: x(18)
        curr_in = in + in_strides[18];
        LDR_128_S(curr_in, v_in_stride, av_in9);
        // Input point 21: x(20)
        curr_in = in + in_strides[20];
        LDR_128_S(curr_in, v_in_stride, av_in10);
        // Input point 23: x(22)
        curr_in = in + in_strides[22];
        LDR_128_S(curr_in, v_in_stride, av_in11);
        // Input point 25: x(24)
        curr_in = in + in_strides[24];
        LDR_128_S(curr_in, v_in_stride, av_in12);

        av_s0 = _mm_add_ps(av_in2, av_in7);
        av_s1 = _mm_sub_ps(av_in7, av_in2);
        av_s2 = _mm_add_ps(av_in6, av_in11);
        av_s3 = _mm_sub_ps(av_in11, av_in6);
        av_s4 = _mm_add_ps(av_s0, av_s2);
        av_s5 = _mm_sub_ps(av_s0, av_s2);
        av_m0 = _mm_mul_ps(v_CRTM_13_6, av_s5);
        av_s6 = _mm_add_ps(av_s1, av_s3);
        av_s7 = _mm_sub_ps(av_s1, av_s3);
        av_s8 = _mm_add_ps(av_in4, av_in10);
        av_s9 = _mm_sub_ps(av_in10, av_in4);
        av_s10 = _mm_add_ps(av_in3, av_in9);
        av_s11 = _mm_sub_ps(av_in9, av_in3);
        av_s12 = _mm_add_ps(av_s8, av_s10);
        av_s13 = _mm_sub_ps(av_s8, av_s10);
        av_s14 = _mm_sub_ps(av_s9, av_s11);
        av_s15 = _mm_add_ps(av_s9, av_s11);
        av_m1 = _mm_mul_ps(v_CRTM_13_6, av_s15);
        av_m2 = _mm_mul_ps(v_CRTM_13_7, av_s13);

        av_s16 = _mm_add_ps(av_s4, av_s12);
        av_s17 = _mm_sub_ps(av_s4, av_s12);
        av_s18 = _mm_add_ps(av_in8, av_in5);
        av_s19 = _mm_sub_ps(av_in5, av_in8);
        av_m3 = _mm_mul_ps(v_CRTM_13_7, av_s6);
        av_s42 = _mm_add_ps(av_m3, av_s19);
        av_s20 = _mm_sub_ps(av_in1, av_in12);
        av_s21 = _mm_add_ps(av_in1, av_in12);
        av_s36 = _mm_add_ps(av_s20, av_m2);
        av_s22 = _mm_sub_ps(av_s20, av_s13);
        av_s23 = _mm_sub_ps(av_s6, av_s19);
        av_m4 = _mm_mul_ps(v_R13_DGC_6, av_s22);
        av_m5 = _mm_mul_ps(v_R13_DGC_7, av_s23);
        av_s43 = _mm_add_ps(av_m4, av_m5);
        av_m6 = _mm_mul_ps(v_R13_DGC_6, av_s23);
        av_m7 = _mm_mul_ps(v_R13_DGC_7, av_s22);
        av_s45 = _mm_sub_ps(av_m6, av_m7);
        av_s47 = _mm_add_ps(av_s21, av_s18);
        av_s48 = _mm_sub_ps(av_s21, av_s18);
        av_m8 = _mm_mul_ps(v_CRTM_13_7, av_s16);
        av_s32 = _mm_sub_ps(av_s47, av_m8);
        av_s28 = _mm_add_ps(av_s47, av_s16);
        av_m9 = _mm_mul_ps(v_CRTM_13_7, av_s17);
        av_s33 = _mm_add_ps(av_s48, av_m9);
        av_s39 = _mm_sub_ps(av_s48, av_s17);
        av_m10 = _mm_mul_ps(av_s39, v_R13_DGC_12);
        // Output point 1: X(0)
        v_out0 = _mm_add_ps(av_s28, av_in0);
        STR_128_S(curr_out, v_out_stride, v_out0);
        av_m11 = NEGATE_128_S(_mm_mul_ps(av_s28, v_R13_DGC_1));

        av_s63 = _mm_add_ps(av_m11, av_in0);
        av_s24 = _mm_add_ps(av_s63, av_m10);
        av_s25 = _mm_sub_ps(av_s63, av_m10);
        av_s61 = _mm_add_ps(av_s36, av_m0);
        av_s62 = _mm_sub_ps(av_s36, av_m0);
        av_s46 = _mm_add_ps(av_s42, av_m1);
        av_s29 = _mm_sub_ps(av_s42, av_m1);
        av_m12 = _mm_mul_ps(v_R13_DGC_2, av_s61);
        av_m13 = _mm_mul_ps(v_R13_DGC_3, av_s46);
        av_s40 = NEGATE_128_S(_mm_add_ps(av_m12, av_m13));
        av_m14 = _mm_mul_ps(v_R13_DGC_2, av_s46);
        av_m15 = _mm_mul_ps(v_R13_DGC_3, av_s61);
        av_s41 = _mm_sub_ps(av_m14, av_m15);
        av_m16 = _mm_mul_ps(v_R13_DGC_10, av_s29);
        av_m17 = _mm_mul_ps(v_R13_DGC_11, av_s62);
        av_s34 = _mm_sub_ps(av_m17, av_m16);
        av_m18 = _mm_mul_ps(v_R13_DGC_10, av_s62);
        av_m19 = _mm_mul_ps(v_R13_DGC_11, av_s29);
        av_s35 = _mm_add_ps(av_m18, av_m19);
        av_s26 = _mm_add_ps(av_s41, av_s34);
        av_s44 = _mm_sub_ps(av_s41, av_s34);
        av_m20 = _mm_mul_ps(v_CRTM_13_1, av_s44);
        av_s27 = _mm_add_ps(av_s40, av_s35);
        av_s64 = _mm_sub_ps(av_s40, av_s35);
        av_m21 = _mm_mul_ps(v_CRTM_13_1, av_s64);
        av_s30 = _mm_add_ps(av_s7, av_s14);
        av_m22 = _mm_mul_ps(v_R13_DGC_4, av_s33);
        av_m23 = _mm_mul_ps(v_CRTM_13_3, av_s30);
        av_s49 = _mm_sub_ps(av_m22, av_m23);
        av_m24 = _mm_mul_ps(v_CRTM_13_2, av_s30);
        av_m25 = _mm_mul_ps(v_R13_DGC_5, av_s33);
        av_s37 = NEGATE_128_S(_mm_add_ps(av_m24, av_m25));

        av_s31 = _mm_sub_ps(av_s7, av_s14);
        av_m26 = _mm_mul_ps(v_R13_DGC_8, av_s32);
        av_m27 = _mm_mul_ps(v_CRTM_13_5, av_s31);
        av_s50 = _mm_sub_ps(av_m26, av_m27);
        av_m28 = _mm_mul_ps(v_CRTM_13_4, av_s31);
        av_m29 = _mm_mul_ps(v_R13_DGC_9, av_s32);
        av_s38 = NEGATE_128_S(_mm_add_ps(av_m28, av_m29));

        av_s51 = _mm_add_ps(av_s37, av_s38);
        av_s59 = _mm_sub_ps(av_s37, av_s38);
        av_s52 = _mm_add_ps(av_s49, av_s50);
        av_s53 = _mm_sub_ps(av_s49, av_s50);
        av_m30 = _mm_mul_ps(v_CRTM_13_8, av_s52);
        // Output point 4: X(3)
        v_out3 = _mm_add_ps(av_s24, av_m30);
        av_m31 = _mm_mul_ps(v_CRTM_13_8, av_s26);
        // Output point 5: X(4)
        v_out4 = _mm_add_ps(av_s45, av_m31);
        curr_out = out + out_strides[3];
        STRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);

        av_s54 = _mm_sub_ps(av_s24, av_s52);
        av_s55 = _mm_sub_ps(av_s45, av_s26);
        // Output point 12: X(11)
        v_out11 = _mm_add_ps(av_s54, av_s59);
        // Output point 13: X(12)
        v_out12 = _mm_add_ps(av_s55, av_m21);
        curr_out = out + out_strides[11];
        STRI_2x128_S(curr_out, v_out_stride, v_out11, v_out12);

        // Output point 16: X(15)
        v_out15 = _mm_sub_ps(av_s54, av_s59);
        // Output point 17: X(16)
        v_out16 = _mm_sub_ps(av_m21, av_s55);
        curr_out = out + out_strides[15];
        STRI_2x128_S(curr_out, v_out_stride, v_out15, v_out16);

        av_s56 = _mm_sub_ps(av_s27, av_s43);
        av_m32 = _mm_mul_ps(v_CRTM_13_8, av_s27);
        av_s57 = NEGATE_128_S(_mm_add_ps(av_m32, av_s43));
        av_m33 = _mm_mul_ps(v_CRTM_13_8, av_s53);
        av_s58 = _mm_sub_ps(av_s25, av_m33);
        av_s60 = _mm_add_ps(av_s25, av_s53);
        // Output point 8: X(7)
        v_out7 = _mm_sub_ps(av_s60, av_s51);
        // Output point 9: X(8)
        v_out8 = _mm_add_ps(av_m20, av_s56);
        curr_out = out + out_strides[7];
        STRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);

        // Output point 20: X(19)
        v_out19 = av_s58;
        // Output point 21: X(20)
        v_out20 = av_s57;
        curr_out = out + out_strides[19];
        STRI_2x128_S(curr_out, v_out_stride, v_out19, v_out20);

        // Output point 24: X(23)
        v_out23 = _mm_add_ps(av_s60, av_s51);
        // Output point 25: X(24)
        v_out24 = _mm_sub_ps(av_s56, av_m20);

        curr_out = out + out_strides[23];
        STRI_2x128_S(curr_out, v_out_stride, v_out23, v_out24);

        // Shifted DFT
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8, bv_in9, bv_in10, bv_in11, bv_in12;
        __m128 bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
               bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16,
               bv_s17, bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24,
               bv_s25, bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32,
               bv_s33, bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40,
               bv_s41, bv_s42, bv_s43, bv_s44, bv_s45, bv_s46, bv_s47, bv_s48,
               bv_s49, bv_s50, bv_s51, bv_s52, bv_s53, bv_s54, bv_s55, bv_s56,
               bv_s57, bv_s58, bv_s59, bv_s60, bv_s61, bv_s62, bv_s63, bv_s64,
               bv_s65, bv_s66, bv_s67, bv_s68, bv_s69, bv_s70, bv_s71, bv_s72,
               bv_s73, bv_s74, bv_s75, bv_s76;
        __m128 bv_m0, bv_m1, bv_m2, bv_m3, bv_m4, bv_m5, bv_m6, bv_m7, bv_m8,
               bv_m9, bv_m10, bv_m11, bv_m12, bv_m13, bv_m14, bv_m15, bv_m16,
               bv_m17, bv_m18, bv_m19, bv_m20, bv_m21, bv_m22, bv_m23, bv_m24,
               bv_m25, bv_m26, bv_m27, bv_m28, bv_m29, bv_m30, bv_m31, bv_m32,
               bv_m33, bv_m34, bv_m35, bv_m36, bv_m37, bv_m38, bv_m39, bv_m40,
               bv_m41, bv_m42, bv_m43, bv_m44, bv_m45, bv_m46, bv_m47, bv_m48,
               bv_m49, bv_m50, bv_m51, bv_m52, bv_m53, bv_m54, bv_m55, bv_m56,
               bv_m57, bv_m58, bv_m59, bv_m60, bv_m61, bv_m62, bv_m63, bv_m64,
               bv_m65, bv_m66, bv_m67, bv_m68, bv_m69, bv_m70, bv_m71;

        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_128_S(curr_in, v_in_stride, bv_in0);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_128_S(curr_in, v_in_stride, bv_in1);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_128_S(curr_in, v_in_stride, bv_in2);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_128_S(curr_in, v_in_stride, bv_in3);
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDR_128_S(curr_in, v_in_stride, bv_in4);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_128_S(curr_in, v_in_stride, bv_in5);
        // Input point 14: x(13)
        curr_in = in + in_strides[13];
        LDR_128_S(curr_in, v_in_stride, bv_in6);
        // Input point 16: x(15)
        curr_in = in + in_strides[15];
        LDR_128_S(curr_in, v_in_stride, bv_in7);
        // Input point 18: x(17)
        curr_in = in + in_strides[17];
        LDR_128_S(curr_in, v_in_stride, bv_in8);
        // Input point 20: x(19)
        curr_in = in + in_strides[19];
        LDR_128_S(curr_in, v_in_stride, bv_in9);
        // Input point 22: x(21)
        curr_in = in + in_strides[21];
        LDR_128_S(curr_in, v_in_stride, bv_in10);
        // Input point 24: x(23)
        curr_in = in + in_strides[23];
        LDR_128_S(curr_in, v_in_stride, bv_in11);
        // Input point 26: x(25)
        curr_in = in + in_strides[25];
        LDR_128_S(curr_in, v_in_stride, bv_in12);

        bv_s0 = _mm_add_ps(bv_in1, bv_in12);
        bv_s1 = _mm_sub_ps(bv_in1, bv_in12);
        bv_s2 = _mm_add_ps(bv_in2, bv_in11);
        bv_s3 = _mm_sub_ps(bv_in2, bv_in11);
        bv_s4 = _mm_add_ps(bv_in3, bv_in10);
        bv_s5 = _mm_sub_ps(bv_in3, bv_in10);
        bv_s6 = _mm_add_ps(bv_in4, bv_in9);
        bv_s7 = _mm_sub_ps(bv_in4, bv_in9);
        bv_s8 = _mm_add_ps(bv_in5, bv_in8);
        bv_s9 = _mm_sub_ps(bv_in5, bv_in8);
        bv_s10 = _mm_add_ps(bv_in6, bv_in7);
        bv_s11 = _mm_sub_ps(bv_in6, bv_in7);

        bv_m0 = _mm_mul_ps(v_R13_DFT_C1, bv_s1);
        bv_m1 = _mm_mul_ps(v_R13_DFT_C2, bv_s3);
        bv_s12 = _mm_add_ps(bv_m0, bv_m1);
        bv_m2 = _mm_mul_ps(v_R13_DFT_C3, bv_s5);
        bv_m3 = _mm_mul_ps(v_R13_DFT_C4, bv_s7);
        bv_s13 = _mm_add_ps(bv_m2, bv_m3);
        bv_m4 = _mm_mul_ps(v_R13_DFT_C5, bv_s9);
        bv_m5 = _mm_mul_ps(v_R13_DFT_C6, bv_s11);
        bv_s14 = _mm_add_ps(bv_m4, bv_m5);
        bv_s15 = _mm_add_ps(bv_s12, bv_s13);
        bv_s16 = _mm_add_ps(bv_s15, bv_s14);
        // Output point 2: X(1)
        v_out1 = _mm_add_ps(bv_in0, bv_s16);

        bv_m6 = _mm_mul_ps(v_R13_DFT_S1, bv_s0);
        bv_m7 = _mm_mul_ps(v_R13_DFT_S2, bv_s2);
        bv_s17 = _mm_add_ps(bv_m6, bv_m7);
        bv_m8 = _mm_mul_ps(v_R13_DFT_S3, bv_s4);
        bv_m9 = _mm_mul_ps(v_R13_DFT_S4, bv_s6);
        bv_s18 = _mm_add_ps(bv_m8, bv_m9);
        bv_m10 = _mm_mul_ps(v_R13_DFT_S5, bv_s8);
        bv_m11 = _mm_mul_ps(v_R13_DFT_S6, bv_s10);
        bv_s19 = _mm_add_ps(bv_m10, bv_m11);
        bv_s20 = _mm_add_ps(bv_s17, bv_s18);
        bv_s21 = _mm_add_ps(bv_s20, bv_s19);
        // Output point 3: X(2)
        v_out2 = NEGATE_128_S(bv_s21);
        curr_out = out + out_strides[1];
        STRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);

        bv_m12 = _mm_mul_ps(v_R13_DFT_C3, bv_s1);
        bv_m13 = _mm_mul_ps(v_R13_DFT_C6, bv_s3);
        bv_s22 = _mm_add_ps(bv_m12, bv_m13);
        bv_m14 = _mm_mul_ps(v_R13_DFT_C4, bv_s5);
        bv_m15 = _mm_mul_ps(v_R13_DFT_C1, bv_s7);
        bv_s23 = _mm_add_ps(bv_m14, bv_m15);
        bv_m16 = _mm_mul_ps(v_R13_DFT_C2, bv_s9);
        bv_m17 = _mm_mul_ps(v_R13_DFT_C5, bv_s11);
        bv_s24 = _mm_add_ps(bv_m16, bv_m17);
        bv_s25 = _mm_add_ps(bv_s23, bv_s24);
        bv_s26 = _mm_sub_ps(bv_s22, bv_s25);
        // Output point 6: X(5)
        v_out5 = _mm_add_ps(bv_in0, bv_s26);

        bv_m18 = _mm_mul_ps(v_R13_DFT_S3, bv_s0);
        bv_m19 = _mm_mul_ps(v_R13_DFT_S6, bv_s2);
        bv_s27 = _mm_add_ps(bv_m18, bv_m19);
        bv_m20 = _mm_mul_ps(v_R13_DFT_S4, bv_s4);
        bv_m21 = _mm_mul_ps(v_R13_DFT_S1, bv_s6);
        bv_s28 = _mm_add_ps(bv_m20, bv_m21);
        bv_m22 = _mm_mul_ps(v_R13_DFT_S2, bv_s8);
        bv_m23 = _mm_mul_ps(v_R13_DFT_S5, bv_s10);
        bv_s29 = _mm_add_ps(bv_m22, bv_m23);
        bv_s30 = _mm_add_ps(bv_s27, bv_s28);
        bv_s31 = _mm_sub_ps(bv_s29, bv_s30);
        // Output point 7: X(6)
        v_out6 = bv_s31;
        curr_out = out + out_strides[5];
        STRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);

        bv_m24 = _mm_mul_ps(v_R13_DFT_C5, bv_s1);
        bv_m25 = _mm_mul_ps(v_R13_DFT_C3, bv_s3);
        bv_s32 = _mm_sub_ps(bv_m24, bv_m25);
        bv_m26 = _mm_mul_ps(v_R13_DFT_C2, bv_s5);
        bv_m27 = _mm_mul_ps(v_R13_DFT_C6, bv_s7);
        bv_s33 = _mm_sub_ps(bv_m27, bv_m26);
        bv_m28 = _mm_mul_ps(v_R13_DFT_C1, bv_s9);
        bv_m29 = _mm_mul_ps(v_R13_DFT_C4, bv_s11);
        bv_s34 = _mm_add_ps(bv_m28, bv_m29);
        bv_s35 = _mm_add_ps(bv_s32, bv_s33);
        bv_s36 = _mm_add_ps(bv_s35, bv_s34);
        // Output point 10: X(9)
        v_out9 = _mm_add_ps(bv_in0, bv_s36);

        bv_m30 = _mm_mul_ps(v_R13_DFT_S5, bv_s0);
        bv_m31 = _mm_mul_ps(v_R13_DFT_S3, bv_s2);
        bv_s37 = _mm_add_ps(bv_m30, bv_m31);
        bv_m32 = _mm_mul_ps(v_R13_DFT_S2, bv_s4);
        bv_m33 = _mm_mul_ps(v_R13_DFT_S6, bv_s6);
        bv_s38 = _mm_add_ps(bv_m32, bv_m33);
        bv_m34 = _mm_mul_ps(v_R13_DFT_S1, bv_s8);
        bv_m35 = _mm_mul_ps(v_R13_DFT_S4, bv_s10);
        bv_s39 = _mm_sub_ps(bv_m34, bv_m35);
        bv_s40 = _mm_add_ps(bv_s38, bv_s39);
        bv_s41 = _mm_sub_ps(bv_s40, bv_s37);
        // Output point 11: X(10)
        v_out10 = bv_s41;
        curr_out = out + out_strides[9];
        STRI_2x128_S(curr_out, v_out_stride, v_out9, v_out10);

        bv_m36 = _mm_mul_ps(v_R13_DFT_C6, bv_s1);
        bv_m37 = _mm_mul_ps(v_R13_DFT_C1, bv_s3);
        bv_s42 = _mm_add_ps(bv_m36, bv_m37);
        bv_m38 = _mm_mul_ps(v_R13_DFT_C5, bv_s5);
        bv_m39 = _mm_mul_ps(v_R13_DFT_C2, bv_s7);
        bv_s43 = _mm_add_ps(bv_m38, bv_m39);
        bv_m40 = _mm_mul_ps(v_R13_DFT_C4, bv_s9);
        bv_m41 = _mm_mul_ps(v_R13_DFT_C3, bv_s11);
        bv_s44 = _mm_add_ps(bv_m40, bv_m41);
        bv_s45 = _mm_add_ps(bv_s42, bv_s44);
        bv_s46 = _mm_sub_ps(bv_s43, bv_s45);
        // Output point 14: X(13)
        v_out13 = _mm_add_ps(bv_in0, bv_s46);

        bv_m42 = _mm_mul_ps(v_R13_DFT_S6, bv_s0);
        bv_m43 = _mm_mul_ps(v_R13_DFT_S1, bv_s2);
        bv_s47 = _mm_sub_ps(bv_m43, bv_m42);
        bv_m44 = _mm_mul_ps(v_R13_DFT_S5, bv_s4);
        bv_m45 = _mm_mul_ps(v_R13_DFT_S2, bv_s6);
        bv_s48 = _mm_sub_ps(bv_m44, bv_m45);
        bv_m46 = _mm_mul_ps(v_R13_DFT_S4, bv_s8);
        bv_m47 = _mm_mul_ps(v_R13_DFT_S3, bv_s10);
        bv_s49 = _mm_sub_ps(bv_m47, bv_m46);
        bv_s50 = _mm_add_ps(bv_s47, bv_s48);
        bv_s51 = _mm_add_ps(bv_s50, bv_s49);
        // Output point 15: X(14)
        v_out14 = bv_s51;
        curr_out = out + out_strides[13];
        STRI_2x128_S(curr_out, v_out_stride, v_out13, v_out14);

        bv_m48 = _mm_mul_ps(v_R13_DFT_C4, bv_s1);
        bv_m49 = _mm_mul_ps(v_R13_DFT_C5, bv_s3);
        bv_s52 = _mm_add_ps(bv_m48, bv_m49);
        bv_m50 = _mm_mul_ps(v_R13_DFT_C1, bv_s5);
        bv_m51 = _mm_mul_ps(v_R13_DFT_C3, bv_s7);
        bv_s53 = _mm_sub_ps(bv_m50, bv_m51);
        bv_m52 = _mm_mul_ps(v_R13_DFT_C6, bv_s9);
        bv_m53 = _mm_mul_ps(v_R13_DFT_C2, bv_s11);
        bv_s54 = _mm_sub_ps(bv_m53, bv_m52);
        bv_s55 = _mm_add_ps(bv_s53, bv_s54);
        bv_s56 = _mm_sub_ps(bv_s55, bv_s52);
        // Output point 18: X(17)
        v_out17 = _mm_add_ps(bv_in0, bv_s56);

        bv_m54 = _mm_mul_ps(v_R13_DFT_S4, bv_s0);
        bv_m55 = _mm_mul_ps(v_R13_DFT_S5, bv_s2);
        bv_s57 = _mm_sub_ps(bv_m55, bv_m54);
        bv_m56 = _mm_mul_ps(v_R13_DFT_S1, bv_s4);
        bv_m57 = _mm_mul_ps(v_R13_DFT_S3, bv_s6);
        bv_s58 = _mm_add_ps(bv_m56, bv_m57);
        bv_m58 = _mm_mul_ps(v_R13_DFT_S6, bv_s8);
        bv_m59 = _mm_mul_ps(v_R13_DFT_S2, bv_s10);
        bv_s59 = _mm_sub_ps(bv_m58, bv_m59);
        bv_s60 = _mm_add_ps(bv_s57, bv_s59);
        bv_s61 = _mm_sub_ps(bv_s60, bv_s58);
        // Output point 19: X(18)
        v_out18 = bv_s61;
        curr_out = out + out_strides[17];
        STRI_2x128_S(curr_out, v_out_stride, v_out17, v_out18);

        bv_m60 = _mm_mul_ps(v_R13_DFT_C2, bv_s1);
        bv_m61 = _mm_mul_ps(v_R13_DFT_C4, bv_s3);
        bv_s62 = _mm_sub_ps(bv_m61, bv_m60);
        bv_m62 = _mm_mul_ps(v_R13_DFT_C6, bv_s5);
        bv_m63 = _mm_mul_ps(v_R13_DFT_C5, bv_s7);
        bv_s63 = _mm_add_ps(bv_m62, bv_m63);
        bv_m64 = _mm_mul_ps(v_R13_DFT_C3, bv_s9);
        bv_m65 = _mm_mul_ps(v_R13_DFT_C1, bv_s11);
        bv_s64 = _mm_sub_ps(bv_m64, bv_m65);
        bv_s65 = _mm_add_ps(bv_s62, bv_s64);
        bv_s66 = _mm_sub_ps(bv_s65, bv_s63);
        // Output point 22: X(21)
        v_out21 = _mm_add_ps(bv_in0, bv_s66);

        bv_m66 = _mm_mul_ps(v_R13_DFT_S2, bv_s0);
        bv_m67 = _mm_mul_ps(v_R13_DFT_S4, bv_s2);
        bv_s67 = _mm_sub_ps(bv_m67, bv_m66);
        bv_m68 = _mm_mul_ps(v_R13_DFT_S6, bv_s4);
        bv_m69 = _mm_mul_ps(v_R13_DFT_S5, bv_s6);
        bv_s68 = _mm_sub_ps(bv_m69, bv_m68);
        bv_m70 = _mm_mul_ps(v_R13_DFT_S3, bv_s8);
        bv_m71 = _mm_mul_ps(v_R13_DFT_S1, bv_s10);
        bv_s69 = _mm_sub_ps(bv_m71, bv_m70);
        bv_s70 = _mm_add_ps(bv_s67, bv_s68);
        bv_s71 = _mm_add_ps(bv_s70, bv_s69);
        // Output point 23: X(22)
        v_out22 = bv_s71;
        curr_out = out + out_strides[21];
        STRI_2x128_S(curr_out, v_out_stride, v_out21, v_out22);

        bv_s72 = _mm_sub_ps(bv_s3, bv_s1);
        bv_s73 = _mm_sub_ps(bv_s7, bv_s5);
        bv_s74 = _mm_sub_ps(bv_s11, bv_s9);
        bv_s75 = _mm_add_ps(bv_s72, bv_s73);
        bv_s76 = _mm_add_ps(bv_s75, bv_s74);
        // Output point 26: X(25)
        v_out25 = _mm_add_ps(bv_in0, bv_s76);
        curr_out = out + out_strides[25];
        STR_128_S(curr_out, v_out_stride, v_out25);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & 2)
    {
        // Standard DFT
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
               av_in8, av_in9, av_in10, av_in11, av_in12;
        __m128 av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
               av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16,
               av_s17, av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24,
               av_s25, av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32,
               av_s33, av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40,
               av_s41, av_s42, av_s43, av_s44, av_s45, av_s46, av_s47, av_s48,
               av_s49, av_s50, av_s51, av_s52, av_s53, av_s54, av_s55, av_s56,
               av_s57, av_s58, av_s59, av_s60, av_s61, av_s62, av_s63, av_s64;
        __m128 av_m0, av_m1, av_m2, av_m3, av_m4, av_m5, av_m6, av_m7, av_m8,
               av_m9, av_m10, av_m11, av_m12, av_m13, av_m14, av_m15, av_m16,
               av_m17, av_m18, av_m19, av_m20, av_m21, av_m22, av_m23, av_m24,
               av_m25, av_m26, av_m27, av_m28, av_m29, av_m30, av_m31, av_m32,
               av_m33;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21,
               v_out22, v_out23, v_out24, v_out25;
        
        curr_in = in;
        curr_out = out;
        
        // Input point 1: x(0)
        LDHR_128_S(curr_in, v_in_stride, av_in0);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDHR_128_S(curr_in, v_in_stride, av_in1);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDHR_128_S(curr_in, v_in_stride, av_in2);
        // Input point 7: x(6)
        curr_in = in + in_strides[6];
        LDHR_128_S(curr_in, v_in_stride, av_in3);
        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDHR_128_S(curr_in, v_in_stride, av_in4);
        // Input point 11: x(10)
        curr_in = in + in_strides[10];
        LDHR_128_S(curr_in, v_in_stride, av_in5);
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDHR_128_S(curr_in, v_in_stride, av_in6);
        // Input point 15: x(14)
        curr_in = in + in_strides[14];
        LDHR_128_S(curr_in, v_in_stride, av_in7);
        // Input point 17: x(16)
        curr_in = in + in_strides[16];
        LDHR_128_S(curr_in, v_in_stride, av_in8);
        // Input point 19: x(18)
        curr_in = in + in_strides[18];
        LDHR_128_S(curr_in, v_in_stride, av_in9);
        // Input point 21: x(20)
        curr_in = in + in_strides[20];
        LDHR_128_S(curr_in, v_in_stride, av_in10);
        // Input point 23: x(22)
        curr_in = in + in_strides[22];
        LDHR_128_S(curr_in, v_in_stride, av_in11);
        // Input point 25: x(24)
        curr_in = in + in_strides[24];
        LDHR_128_S(curr_in, v_in_stride, av_in12);

        av_s0 = _mm_add_ps(av_in2, av_in7);
        av_s1 = _mm_sub_ps(av_in7, av_in2);
        av_s2 = _mm_add_ps(av_in6, av_in11);
        av_s3 = _mm_sub_ps(av_in11, av_in6);
        av_s4 = _mm_add_ps(av_s0, av_s2);
        av_s5 = _mm_sub_ps(av_s0, av_s2);
        av_m0 = _mm_mul_ps(v_CRTM_13_6, av_s5);
        av_s6 = _mm_add_ps(av_s1, av_s3);
        av_s7 = _mm_sub_ps(av_s1, av_s3);
        av_s8 = _mm_add_ps(av_in4, av_in10);
        av_s9 = _mm_sub_ps(av_in10, av_in4);
        av_s10 = _mm_add_ps(av_in3, av_in9);
        av_s11 = _mm_sub_ps(av_in9, av_in3);
        av_s12 = _mm_add_ps(av_s8, av_s10);
        av_s13 = _mm_sub_ps(av_s8, av_s10);
        av_s14 = _mm_sub_ps(av_s9, av_s11);
        av_s15 = _mm_add_ps(av_s9, av_s11);
        av_m1 = _mm_mul_ps(v_CRTM_13_6, av_s15);
        av_m2 = _mm_mul_ps(v_CRTM_13_7, av_s13);

        av_s16 = _mm_add_ps(av_s4, av_s12);
        av_s17 = _mm_sub_ps(av_s4, av_s12);
        av_s18 = _mm_add_ps(av_in8, av_in5);
        av_s19 = _mm_sub_ps(av_in5, av_in8);
        av_m3 = _mm_mul_ps(v_CRTM_13_7, av_s6);
        av_s42 = _mm_add_ps(av_m3, av_s19);
        av_s20 = _mm_sub_ps(av_in1, av_in12);
        av_s21 = _mm_add_ps(av_in1, av_in12);
        av_s36 = _mm_add_ps(av_s20, av_m2);
        av_s22 = _mm_sub_ps(av_s20, av_s13);
        av_s23 = _mm_sub_ps(av_s6, av_s19);
        av_m4 = _mm_mul_ps(v_R13_DGC_6, av_s22);
        av_m5 = _mm_mul_ps(v_R13_DGC_7, av_s23);
        av_s43 = _mm_add_ps(av_m4, av_m5);
        av_m6 = _mm_mul_ps(v_R13_DGC_6, av_s23);
        av_m7 = _mm_mul_ps(v_R13_DGC_7, av_s22);
        av_s45 = _mm_sub_ps(av_m6, av_m7);
        av_s47 = _mm_add_ps(av_s21, av_s18);
        av_s48 = _mm_sub_ps(av_s21, av_s18);
        av_m8 = _mm_mul_ps(v_CRTM_13_7, av_s16);
        av_s32 = _mm_sub_ps(av_s47, av_m8);
        av_s28 = _mm_add_ps(av_s47, av_s16);
        av_m9 = _mm_mul_ps(v_CRTM_13_7, av_s17);
        av_s33 = _mm_add_ps(av_s48, av_m9);
        av_s39 = _mm_sub_ps(av_s48, av_s17);
        av_m10 = _mm_mul_ps(av_s39, v_R13_DGC_12);
        // Output point 1: X(0)
        v_out0 = _mm_add_ps(av_s28, av_in0);
        STHR_128_S(curr_out, v_out_stride, v_out0);

        av_m11 = NEGATE_128_S(_mm_mul_ps(av_s28, v_R13_DGC_1));
        av_s63 = _mm_add_ps(av_m11, av_in0);
        av_s24 = _mm_add_ps(av_s63, av_m10);
        av_s25 = _mm_sub_ps(av_s63, av_m10);
        av_s61 = _mm_add_ps(av_s36, av_m0);
        av_s62 = _mm_sub_ps(av_s36, av_m0);
        av_s46 = _mm_add_ps(av_s42, av_m1);
        av_s29 = _mm_sub_ps(av_s42, av_m1);
        av_m12 = _mm_mul_ps(v_R13_DGC_2, av_s61);
        av_m13 = _mm_mul_ps(v_R13_DGC_3, av_s46);
        av_s40 = NEGATE_128_S(_mm_add_ps(av_m12, av_m13));

        av_m14 = _mm_mul_ps(v_R13_DGC_2, av_s46);
        av_m15 = _mm_mul_ps(v_R13_DGC_3, av_s61);
        av_s41 = _mm_sub_ps(av_m14, av_m15);
        av_m16 = _mm_mul_ps(v_R13_DGC_10, av_s29);
        av_m17 = _mm_mul_ps(v_R13_DGC_11, av_s62);
        av_s34 = _mm_sub_ps(av_m17, av_m16);
        av_m18 = _mm_mul_ps(v_R13_DGC_10, av_s62);
        av_m19 = _mm_mul_ps(v_R13_DGC_11, av_s29);
        av_s35 = _mm_add_ps(av_m18, av_m19);
        av_s26 = _mm_add_ps(av_s41, av_s34);
        av_s44 = _mm_sub_ps(av_s41, av_s34);
        av_m20 = _mm_mul_ps(v_CRTM_13_1, av_s44);
        av_s27 = _mm_add_ps(av_s40, av_s35);
        av_s64 = _mm_sub_ps(av_s40, av_s35);
        av_m21 = _mm_mul_ps(v_CRTM_13_1, av_s64);
        av_s30 = _mm_add_ps(av_s7, av_s14);
        av_m22 = _mm_mul_ps(v_R13_DGC_4, av_s33);
        av_m23 = _mm_mul_ps(v_CRTM_13_3, av_s30);
        av_s49 = _mm_sub_ps(av_m22, av_m23);
        av_m24 = _mm_mul_ps(v_CRTM_13_2, av_s30);
        av_m25 = _mm_mul_ps(v_R13_DGC_5, av_s33);
        av_s37 = NEGATE_128_S(_mm_add_ps(av_m24, av_m25));

        av_s31 = _mm_sub_ps(av_s7, av_s14);
        av_m26 = _mm_mul_ps(v_R13_DGC_8, av_s32);
        av_m27 = _mm_mul_ps(v_CRTM_13_5, av_s31);
        av_s50 = _mm_sub_ps(av_m26, av_m27);
        av_m28 = _mm_mul_ps(v_CRTM_13_4, av_s31);
        av_m29 = _mm_mul_ps(v_R13_DGC_9, av_s32);
        av_s38 = NEGATE_128_S(_mm_add_ps(av_m28, av_m29));

        av_s51 = _mm_add_ps(av_s37, av_s38);
        av_s59 = _mm_sub_ps(av_s37, av_s38);
        av_s52 = _mm_add_ps(av_s49, av_s50);
        av_s53 = _mm_sub_ps(av_s49, av_s50);
        av_m30 = _mm_mul_ps(v_CRTM_13_8, av_s52);
        // Output point 4: X(3)
        v_out3 = _mm_add_ps(av_s24, av_m30);
        av_m31 = _mm_mul_ps(v_CRTM_13_8, av_s26);
        // Output point 5: X(4)
        v_out4 = _mm_add_ps(av_s45, av_m31);
        curr_out = out + out_strides[3];
        STHRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);

        av_s54 = _mm_sub_ps(av_s24, av_s52);
        av_s55 = _mm_sub_ps(av_s45, av_s26);
        // Output point 12: X(11)
        v_out11 = _mm_add_ps(av_s54, av_s59);
        // Output point 13: X(12)
        v_out12 = _mm_add_ps(av_s55, av_m21);
        curr_out = out + out_strides[11];
        STHRI_2x128_S(curr_out, v_out_stride, v_out11, v_out12);

        // Output point 16: X(15)
        v_out15 = _mm_sub_ps(av_s54, av_s59);
        // Output point 17: X(16)
        v_out16 = _mm_sub_ps(av_m21, av_s55);
        curr_out = out + out_strides[15];
        STHRI_2x128_S(curr_out, v_out_stride, v_out15, v_out16);

        av_s56 = _mm_sub_ps(av_s27, av_s43);
        av_m32 = _mm_mul_ps(v_CRTM_13_8, av_s27);
        av_s57 = NEGATE_128_S(_mm_add_ps(av_m32, av_s43));
        av_m33 = _mm_mul_ps(v_CRTM_13_8, av_s53);
        av_s58 = _mm_sub_ps(av_s25, av_m33);
        av_s60 = _mm_add_ps(av_s25, av_s53);
        // Output point 8: X(7)
        v_out7 = _mm_sub_ps(av_s60, av_s51);
        // Output point 9: X(8)
        v_out8 = _mm_add_ps(av_m20, av_s56);
        curr_out = out + out_strides[7];
        STHRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);
        
        // Output point 20: X(19)
        v_out19 = av_s58;
        // Output point 21: X(20)
        v_out20 = av_s57;
        curr_out = out + out_strides[19];
        STHRI_2x128_S(curr_out, v_out_stride, v_out19, v_out20);

        // Output point 24: X(23)
        v_out23 = _mm_add_ps(av_s60, av_s51);
        // Output point 25: X(24)
        v_out24 = _mm_sub_ps(av_s56, av_m20);
        curr_out = out + out_strides[23];
        STHRI_2x128_S(curr_out, v_out_stride, v_out23, v_out24);

        // Shifted DFT
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8, bv_in9, bv_in10, bv_in11, bv_in12;
        __m128 bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
               bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16,
               bv_s17, bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24,
               bv_s25, bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32,
               bv_s33, bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40,
               bv_s41, bv_s42, bv_s43, bv_s44, bv_s45, bv_s46, bv_s47, bv_s48,
               bv_s49, bv_s50, bv_s51, bv_s52, bv_s53, bv_s54, bv_s55, bv_s56,
               bv_s57, bv_s58, bv_s59, bv_s60, bv_s61, bv_s62, bv_s63, bv_s64,
               bv_s65, bv_s66, bv_s67, bv_s68, bv_s69, bv_s70, bv_s71, bv_s72,
               bv_s73, bv_s74, bv_s75, bv_s76;
        __m128 bv_m0, bv_m1, bv_m2, bv_m3, bv_m4, bv_m5, bv_m6, bv_m7, bv_m8,
               bv_m9, bv_m10, bv_m11, bv_m12, bv_m13, bv_m14, bv_m15, bv_m16,
               bv_m17, bv_m18, bv_m19, bv_m20, bv_m21, bv_m22, bv_m23, bv_m24,
               bv_m25, bv_m26, bv_m27, bv_m28, bv_m29, bv_m30, bv_m31, bv_m32,
               bv_m33, bv_m34, bv_m35, bv_m36, bv_m37, bv_m38, bv_m39, bv_m40,
               bv_m41, bv_m42, bv_m43, bv_m44, bv_m45, bv_m46, bv_m47, bv_m48,
               bv_m49, bv_m50, bv_m51, bv_m52, bv_m53, bv_m54, bv_m55, bv_m56,
               bv_m57, bv_m58, bv_m59, bv_m60, bv_m61, bv_m62, bv_m63, bv_m64,
               bv_m65, bv_m66, bv_m67, bv_m68, bv_m69, bv_m70, bv_m71;

        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDHR_128_S(curr_in, v_in_stride, bv_in0);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDHR_128_S(curr_in, v_in_stride, bv_in1);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDHR_128_S(curr_in, v_in_stride, bv_in2);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDHR_128_S(curr_in, v_in_stride, bv_in3);
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDHR_128_S(curr_in, v_in_stride, bv_in4);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDHR_128_S(curr_in, v_in_stride, bv_in5);
        // Input point 14: x(13)
        curr_in = in + in_strides[13];
        LDHR_128_S(curr_in, v_in_stride, bv_in6);
        // Input point 16: x(15)
        curr_in = in + in_strides[15];
        LDHR_128_S(curr_in, v_in_stride, bv_in7);
        // Input point 18: x(17)
        curr_in = in + in_strides[17];
        LDHR_128_S(curr_in, v_in_stride, bv_in8);
        // Input point 20: x(19)
        curr_in = in + in_strides[19];
        LDHR_128_S(curr_in, v_in_stride, bv_in9);
        // Input point 22: x(21)
        curr_in = in + in_strides[21];
        LDHR_128_S(curr_in, v_in_stride, bv_in10);
        // Input point 24: x(23)
        curr_in = in + in_strides[23];
        LDHR_128_S(curr_in, v_in_stride, bv_in11);
        // Input point 26: x(25)
        curr_in = in + in_strides[25];
        LDHR_128_S(curr_in, v_in_stride, bv_in12);

        bv_s0 = _mm_add_ps(bv_in1, bv_in12);
        bv_s1 = _mm_sub_ps(bv_in1, bv_in12);
        bv_s2 = _mm_add_ps(bv_in2, bv_in11);
        bv_s3 = _mm_sub_ps(bv_in2, bv_in11);
        bv_s4 = _mm_add_ps(bv_in3, bv_in10);
        bv_s5 = _mm_sub_ps(bv_in3, bv_in10);
        bv_s6 = _mm_add_ps(bv_in4, bv_in9);
        bv_s7 = _mm_sub_ps(bv_in4, bv_in9);
        bv_s8 = _mm_add_ps(bv_in5, bv_in8);
        bv_s9 = _mm_sub_ps(bv_in5, bv_in8);
        bv_s10 = _mm_add_ps(bv_in6, bv_in7);
        bv_s11 = _mm_sub_ps(bv_in6, bv_in7);

        bv_m0 = _mm_mul_ps(v_R13_DFT_C1, bv_s1);
        bv_m1 = _mm_mul_ps(v_R13_DFT_C2, bv_s3);
        bv_s12 = _mm_add_ps(bv_m0, bv_m1);
        bv_m2 = _mm_mul_ps(v_R13_DFT_C3, bv_s5);
        bv_m3 = _mm_mul_ps(v_R13_DFT_C4, bv_s7);
        bv_s13 = _mm_add_ps(bv_m2, bv_m3);
        bv_m4 = _mm_mul_ps(v_R13_DFT_C5, bv_s9);
        bv_m5 = _mm_mul_ps(v_R13_DFT_C6, bv_s11);
        bv_s14 = _mm_add_ps(bv_m4, bv_m5);
        bv_s15 = _mm_add_ps(bv_s12, bv_s13);
        bv_s16 = _mm_add_ps(bv_s15, bv_s14);
        // Output point 2: X(1)
        v_out1 = _mm_add_ps(bv_in0, bv_s16);
        curr_out = out + out_strides[1];

        bv_m6 = _mm_mul_ps(v_R13_DFT_S1, bv_s0);
        bv_m7 = _mm_mul_ps(v_R13_DFT_S2, bv_s2);
        bv_s17 = _mm_add_ps(bv_m6, bv_m7);
        bv_m8 = _mm_mul_ps(v_R13_DFT_S3, bv_s4);
        bv_m9 = _mm_mul_ps(v_R13_DFT_S4, bv_s6);
        bv_s18 = _mm_add_ps(bv_m8, bv_m9);
        bv_m10 = _mm_mul_ps(v_R13_DFT_S5, bv_s8);
        bv_m11 = _mm_mul_ps(v_R13_DFT_S6, bv_s10);
        bv_s19 = _mm_add_ps(bv_m10, bv_m11);
        bv_s20 = _mm_add_ps(bv_s17, bv_s18);
        bv_s21 = _mm_add_ps(bv_s20, bv_s19);
        // Output point 3: X(2)
        v_out2 = NEGATE_128_S(bv_s21);
        curr_out = out + out_strides[1];
        STHRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);

        bv_m12 = _mm_mul_ps(v_R13_DFT_C3, bv_s1);
        bv_m13 = _mm_mul_ps(v_R13_DFT_C6, bv_s3);
        bv_s22 = _mm_add_ps(bv_m12, bv_m13);
        bv_m14 = _mm_mul_ps(v_R13_DFT_C4, bv_s5);
        bv_m15 = _mm_mul_ps(v_R13_DFT_C1, bv_s7);
        bv_s23 = _mm_add_ps(bv_m14, bv_m15);
        bv_m16 = _mm_mul_ps(v_R13_DFT_C2, bv_s9);
        bv_m17 = _mm_mul_ps(v_R13_DFT_C5, bv_s11);
        bv_s24 = _mm_add_ps(bv_m16, bv_m17);
        bv_s25 = _mm_add_ps(bv_s23, bv_s24);
        bv_s26 = _mm_sub_ps(bv_s22, bv_s25);
        // Output point 6: X(5)
        v_out5 = _mm_add_ps(bv_in0, bv_s26);

        bv_m18 = _mm_mul_ps(v_R13_DFT_S3, bv_s0);
        bv_m19 = _mm_mul_ps(v_R13_DFT_S6, bv_s2);
        bv_s27 = _mm_add_ps(bv_m18, bv_m19);
        bv_m20 = _mm_mul_ps(v_R13_DFT_S4, bv_s4);
        bv_m21 = _mm_mul_ps(v_R13_DFT_S1, bv_s6);
        bv_s28 = _mm_add_ps(bv_m20, bv_m21);
        bv_m22 = _mm_mul_ps(v_R13_DFT_S2, bv_s8);
        bv_m23 = _mm_mul_ps(v_R13_DFT_S5, bv_s10);
        bv_s29 = _mm_add_ps(bv_m22, bv_m23);
        bv_s30 = _mm_add_ps(bv_s27, bv_s28);
        bv_s31 = _mm_sub_ps(bv_s29, bv_s30);
        // Output point 7: X(6)
        v_out6 = bv_s31;
        curr_out = out + out_strides[5];
        STHRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);

        bv_m24 = _mm_mul_ps(v_R13_DFT_C5, bv_s1);
        bv_m25 = _mm_mul_ps(v_R13_DFT_C3, bv_s3);
        bv_s32 = _mm_sub_ps(bv_m24, bv_m25);
        bv_m26 = _mm_mul_ps(v_R13_DFT_C2, bv_s5);
        bv_m27 = _mm_mul_ps(v_R13_DFT_C6, bv_s7);
        bv_s33 = _mm_sub_ps(bv_m27, bv_m26);
        bv_m28 = _mm_mul_ps(v_R13_DFT_C1, bv_s9);
        bv_m29 = _mm_mul_ps(v_R13_DFT_C4, bv_s11);
        bv_s34 = _mm_add_ps(bv_m28, bv_m29);
        bv_s35 = _mm_add_ps(bv_s32, bv_s33);
        bv_s36 = _mm_add_ps(bv_s35, bv_s34);
        // Output point 10: X(9)
        v_out9 = _mm_add_ps(bv_in0, bv_s36);

        bv_m30 = _mm_mul_ps(v_R13_DFT_S5, bv_s0);
        bv_m31 = _mm_mul_ps(v_R13_DFT_S3, bv_s2);
        bv_s37 = _mm_add_ps(bv_m30, bv_m31);
        bv_m32 = _mm_mul_ps(v_R13_DFT_S2, bv_s4);
        bv_m33 = _mm_mul_ps(v_R13_DFT_S6, bv_s6);
        bv_s38 = _mm_add_ps(bv_m32, bv_m33);
        bv_m34 = _mm_mul_ps(v_R13_DFT_S1, bv_s8);
        bv_m35 = _mm_mul_ps(v_R13_DFT_S4, bv_s10);
        bv_s39 = _mm_sub_ps(bv_m34, bv_m35);
        bv_s40 = _mm_add_ps(bv_s38, bv_s39);
        bv_s41 = _mm_sub_ps(bv_s40, bv_s37);
        // Output point 11: X(10)
        v_out10 = bv_s41;
        curr_out = out + out_strides[9];
        STHRI_2x128_S(curr_out, v_out_stride, v_out9, v_out10);

        bv_m36 = _mm_mul_ps(v_R13_DFT_C6, bv_s1);
        bv_m37 = _mm_mul_ps(v_R13_DFT_C1, bv_s3);
        bv_s42 = _mm_add_ps(bv_m36, bv_m37);
        bv_m38 = _mm_mul_ps(v_R13_DFT_C5, bv_s5);
        bv_m39 = _mm_mul_ps(v_R13_DFT_C2, bv_s7);
        bv_s43 = _mm_add_ps(bv_m38, bv_m39);
        bv_m40 = _mm_mul_ps(v_R13_DFT_C4, bv_s9);
        bv_m41 = _mm_mul_ps(v_R13_DFT_C3, bv_s11);
        bv_s44 = _mm_add_ps(bv_m40, bv_m41);
        bv_s45 = _mm_add_ps(bv_s42, bv_s44);
        bv_s46 = _mm_sub_ps(bv_s43, bv_s45);
        // Output point 14: X(13)
        v_out13 = _mm_add_ps(bv_in0, bv_s46);

        bv_m42 = _mm_mul_ps(v_R13_DFT_S6, bv_s0);
        bv_m43 = _mm_mul_ps(v_R13_DFT_S1, bv_s2);
        bv_s47 = _mm_sub_ps(bv_m43, bv_m42);
        bv_m44 = _mm_mul_ps(v_R13_DFT_S5, bv_s4);
        bv_m45 = _mm_mul_ps(v_R13_DFT_S2, bv_s6);
        bv_s48 = _mm_sub_ps(bv_m44, bv_m45);
        bv_m46 = _mm_mul_ps(v_R13_DFT_S4, bv_s8);
        bv_m47 = _mm_mul_ps(v_R13_DFT_S3, bv_s10);
        bv_s49 = _mm_sub_ps(bv_m47, bv_m46);
        bv_s50 = _mm_add_ps(bv_s47, bv_s48);
        bv_s51 = _mm_add_ps(bv_s50, bv_s49);
        // Output point 15: X(14)
        v_out14 = bv_s51;
        curr_out = out + out_strides[13];
        STHRI_2x128_S(curr_out, v_out_stride, v_out13, v_out14);

        bv_m48 = _mm_mul_ps(v_R13_DFT_C4, bv_s1);
        bv_m49 = _mm_mul_ps(v_R13_DFT_C5, bv_s3);
        bv_s52 = _mm_add_ps(bv_m48, bv_m49);
        bv_m50 = _mm_mul_ps(v_R13_DFT_C1, bv_s5);
        bv_m51 = _mm_mul_ps(v_R13_DFT_C3, bv_s7);
        bv_s53 = _mm_sub_ps(bv_m50, bv_m51);
        bv_m52 = _mm_mul_ps(v_R13_DFT_C6, bv_s9);
        bv_m53 = _mm_mul_ps(v_R13_DFT_C2, bv_s11);
        bv_s54 = _mm_sub_ps(bv_m53, bv_m52);
        bv_s55 = _mm_add_ps(bv_s53, bv_s54);
        bv_s56 = _mm_sub_ps(bv_s55, bv_s52);
        // Output point 18: X(17)
        v_out17 = _mm_add_ps(bv_in0, bv_s56);

        bv_m54 = _mm_mul_ps(v_R13_DFT_S4, bv_s0);
        bv_m55 = _mm_mul_ps(v_R13_DFT_S5, bv_s2);
        bv_s57 = _mm_sub_ps(bv_m55, bv_m54);
        bv_m56 = _mm_mul_ps(v_R13_DFT_S1, bv_s4);
        bv_m57 = _mm_mul_ps(v_R13_DFT_S3, bv_s6);
        bv_s58 = _mm_add_ps(bv_m56, bv_m57);
        bv_m58 = _mm_mul_ps(v_R13_DFT_S6, bv_s8);
        bv_m59 = _mm_mul_ps(v_R13_DFT_S2, bv_s10);
        bv_s59 = _mm_sub_ps(bv_m58, bv_m59);
        bv_s60 = _mm_add_ps(bv_s57, bv_s59);
        bv_s61 = _mm_sub_ps(bv_s60, bv_s58);
        // Output point 19: X(18)
        v_out18 = bv_s61;
        curr_out = out + out_strides[17];
        STHRI_2x128_S(curr_out, v_out_stride, v_out17, v_out18);

        bv_m60 = _mm_mul_ps(v_R13_DFT_C2, bv_s1);
        bv_m61 = _mm_mul_ps(v_R13_DFT_C4, bv_s3);
        bv_s62 = _mm_sub_ps(bv_m61, bv_m60);
        bv_m62 = _mm_mul_ps(v_R13_DFT_C6, bv_s5);
        bv_m63 = _mm_mul_ps(v_R13_DFT_C5, bv_s7);
        bv_s63 = _mm_add_ps(bv_m62, bv_m63);
        bv_m64 = _mm_mul_ps(v_R13_DFT_C3, bv_s9);
        bv_m65 = _mm_mul_ps(v_R13_DFT_C1, bv_s11);
        bv_s64 = _mm_sub_ps(bv_m64, bv_m65);
        bv_s65 = _mm_add_ps(bv_s62, bv_s64);
        bv_s66 = _mm_sub_ps(bv_s65, bv_s63);
        // Output point 22: X(21)
        v_out21 = _mm_add_ps(bv_in0, bv_s66);

        bv_m66 = _mm_mul_ps(v_R13_DFT_S2, bv_s0);
        bv_m67 = _mm_mul_ps(v_R13_DFT_S4, bv_s2);
        bv_s67 = _mm_sub_ps(bv_m67, bv_m66);
        bv_m68 = _mm_mul_ps(v_R13_DFT_S6, bv_s4);
        bv_m69 = _mm_mul_ps(v_R13_DFT_S5, bv_s6);
        bv_s68 = _mm_sub_ps(bv_m69, bv_m68);
        bv_m70 = _mm_mul_ps(v_R13_DFT_S3, bv_s8);
        bv_m71 = _mm_mul_ps(v_R13_DFT_S1, bv_s10);
        bv_s69 = _mm_sub_ps(bv_m71, bv_m70);
        bv_s70 = _mm_add_ps(bv_s67, bv_s68);
        bv_s71 = _mm_add_ps(bv_s70, bv_s69);
        // Output point 23: X(22)
        v_out22 = bv_s71;
        curr_out = out + out_strides[21];
        STHRI_2x128_S(curr_out, v_out_stride, v_out21, v_out22);

        bv_s72 = _mm_sub_ps(bv_s3, bv_s1);
        bv_s73 = _mm_sub_ps(bv_s7, bv_s5);
        bv_s74 = _mm_sub_ps(bv_s11, bv_s9);
        bv_s75 = _mm_add_ps(bv_s72, bv_s73);
        bv_s76 = _mm_add_ps(bv_s75, bv_s74);
        // Output point 26: X(25)
        v_out25 = _mm_add_ps(bv_in0, bv_s76);
        curr_out = out + out_strides[25];
        STHR_128_S(curr_out, v_out_stride, v_out25);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }

    if (remaining_sets & 1)
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

        a_s56 = a_s27 - a_s43;
        // Output point 9: X(8)
        out[out_strides[8]] = a_m20 + a_s56;
        // Output point 25: X(24)
        out[out_strides[24]] = a_s56 - a_m20;

        // Output point 12: X(11)
        out[out_strides[11]] = a_s54 + a_s59;
        // Output point 16: X(15)
        out[out_strides[15]] = a_s54 - a_s59;

        // Output point 13: X(12)
        out[out_strides[12]] = a_s55 + a_m21;
        // Output point 17: X(16)
        out[out_strides[16]] = a_m21 - a_s55;

        a_m32 = CRTM_13_8 * a_s27;
        a_s57 = -(a_m32 + a_s43);
        // Output point 21: X(20)
        out[out_strides[20]] = a_s57;

        a_m33 = CRTM_13_8 * a_s53;
        a_s58 = a_s25 - a_m33;
        // Output point 20: X(19)
        out[out_strides[19]] = a_s58;

        a_s60 = a_s25 + a_s53;
        // Output point 8: X(7)
        out[out_strides[7]] = a_s60 - a_s51;
        // Output point 24: X(23)
        out[out_strides[23]] = a_s60 + a_s51;

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
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft13avx128_fp32_bwd(FFTZ_VOID *in_real,
                                             FFTZ_VOID *in_imag,
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
    FFTZ_FLOAT *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_128_S;
    FFTZ_INTP remaining_sets = n % NUM_SETS_REAL_128_S;

    __m128 v_CRTM_13_1 = _mm_set1_ps(CRTM_13_1);
    __m128 v_CRTM_13_2 = _mm_set1_ps(CRTM_13_2);
    __m128 v_CRTM_13_3 = _mm_set1_ps(CRTM_13_3);
    __m128 v_CRTM_13_4 = _mm_set1_ps(CRTM_13_4);
    __m128 v_CRTM_13_5 = _mm_set1_ps(CRTM_13_5);
    __m128 v_CRTM_13_6 = _mm_set1_ps(CRTM_13_6);
    __m128 v_CRTM_13_7 = _mm_set1_ps(CRTM_13_7);
    __m128 v_R13_DGC_1 = _mm_set1_ps(R13_DGC_1);
    __m128 v_R13_DGC_2 = _mm_set1_ps(R13_DGC_2);
    __m128 v_R13_DGC_3 = _mm_set1_ps(R13_DGC_3);
    __m128 v_R13_DGC_4 = _mm_set1_ps(R13_DGC_4);
    __m128 v_R13_DGC_5 = _mm_set1_ps(R13_DGC_5);
    __m128 v_R13_DGC_6 = _mm_set1_ps(R13_DGC_6);
    __m128 v_R13_DGC_7 = _mm_set1_ps(R13_DGC_7);
    __m128 v_R13_DGC_8 = _mm_set1_ps(R13_DGC_8);
    __m128 v_R13_DGC_9 = _mm_set1_ps(R13_DGC_9);
    __m128 v_R13_DGC_10 = _mm_set1_ps(R13_DGC_10);
    __m128 v_R13_DGC_11 = _mm_set1_ps(R13_DGC_11);
    __m128 v_R13_DGC_12 = _mm_set1_ps(R13_DGC_12);
    __m128 v_R13_DFT_C1 = _mm_set1_ps(R13_DFT_C1);
    __m128 v_R13_DFT_C2 = _mm_set1_ps(R13_DFT_C2);
    __m128 v_R13_DFT_C3 = _mm_set1_ps(R13_DFT_C3);
    __m128 v_R13_DFT_C4 = _mm_set1_ps(R13_DFT_C4);
    __m128 v_R13_DFT_C5 = _mm_set1_ps(R13_DFT_C5);
    __m128 v_R13_DFT_C6 = _mm_set1_ps(R13_DFT_C6);
    __m128 v_R13_DFT_S1 = _mm_set1_ps(R13_DFT_S1);
    __m128 v_R13_DFT_S2 = _mm_set1_ps(R13_DFT_S2);
    __m128 v_R13_DFT_S3 = _mm_set1_ps(R13_DFT_S3);
    __m128 v_R13_DFT_S4 = _mm_set1_ps(R13_DFT_S4);
    __m128 v_R13_DFT_S5 = _mm_set1_ps(R13_DFT_S5);
    __m128 v_R13_DFT_S6 = _mm_set1_ps(R13_DFT_S6);

    for (cnt = 0; cnt < N; cnt++)
    {
        // Standard DFT
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
               av_in8, av_in9, av_in10, av_in11, av_in12;
        __m128 av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
               av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16,
               av_s17, av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24,
               av_s25, av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32,
               av_s33, av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40,
               av_s41, av_s42, av_s43, av_s44, av_s45, av_s46, av_s47, av_s48,
               av_s49, av_s50, av_s51, av_s52, av_s53, av_s54, av_s55, av_s56,
               av_s57, av_s58, av_s59, av_s60, av_s61, av_s62;
        __m128 av_m0, av_m1, av_m2, av_m3, av_m4, av_m5, av_m6, av_m7, av_m8,
               av_m9, av_m10, av_m11, av_m12, av_m13, av_m14, av_m15, av_m16,
               av_m17, av_m18, av_m19, av_m20, av_m21, av_m22, av_m23, av_m24,
               av_m25, av_m26, av_m27, av_m28, av_m29, av_m30, av_m31, av_m32,
               av_m33, av_m34;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21,
               v_out22, v_out23, v_out24, v_out25;
   
        curr_in = in;
        curr_out = out;

        // Input point 1: X(0)
        LDR_128_S(curr_in, v_in_stride, av_in0);
        // Input point 4: X(3) & Input point 5: X(4)
        curr_in = in + in_strides[3];
        LDRI_2x128_S(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: X(7) & Input point 9: X(8)
        curr_in = in + in_strides[7];
        LDRI_2x128_S(curr_in, v_in_stride, av_in3, av_in4);
        // Input point 12: X(11) & Input point 13: X(12)
        curr_in = in + in_strides[11];
        LDRI_2x128_S(curr_in, v_in_stride, av_in5, av_in6);
        // Input point 16: X(15) & Input point 17: X(16)
        curr_in = in + in_strides[15];
        LDRI_2x128_S(curr_in, v_in_stride, av_in7, av_in8);
        // Input point 20: X(19) & Input point 21: X(20)
        curr_in = in + in_strides[19];
        LDRI_2x128_S(curr_in, v_in_stride, av_in9, av_in10);
        // Input point 24: X(23) & Input point 25: X(24)
        curr_in = in + in_strides[23];
        LDRI_2x128_S(curr_in, v_in_stride, av_in11, av_in12);

        av_s0 = _mm_sub_ps(av_in6, av_in8);
        av_m0 = _mm_mul_ps(v_CRTM_13_7, av_in2);
        av_s1 = _mm_sub_ps(av_m0, av_s0);
        av_s59 = _mm_add_ps(av_in6, av_in8);
        av_m1 = _mm_mul_ps(v_CRTM_13_1, av_s59);
        av_s2 = _mm_add_ps(av_in12, av_in4);
        av_s60 = _mm_sub_ps(av_in12, av_in4);
        av_m2 = _mm_mul_ps(v_CRTM_13_1, av_s60);
        av_m3 = _mm_mul_ps(v_CRTM_13_7, av_in10);
        av_s3 = _mm_sub_ps(av_s2, av_m3);
        av_s4 = _mm_add_ps(av_s1, av_m2);
        av_s5 = _mm_sub_ps(av_s3, av_m1);
        av_m4 = _mm_mul_ps(v_R13_DGC_11, av_s4);
        av_m5 = _mm_mul_ps(v_R13_DGC_10, av_s5);
        av_s6 = _mm_add_ps(av_m4, av_m5);
        av_m6 = _mm_mul_ps(v_R13_DGC_10, av_s4);
        av_m7 = _mm_mul_ps(v_R13_DGC_11, av_s5);
        av_s7 = _mm_sub_ps(av_m6, av_m7);
        av_s8 = _mm_add_ps(av_in2, av_s0);
        av_s9 = _mm_add_ps(av_s2, av_in10);
        av_m8 = _mm_mul_ps(v_CRTM_13_4, av_s8);
        av_m9 = _mm_mul_ps(v_CRTM_13_5, av_s9);

        av_s10 = _mm_sub_ps(av_m8, av_m9);
        av_m10 = _mm_mul_ps(v_CRTM_13_5, av_s8);
        av_m11 = _mm_mul_ps(v_CRTM_13_4, av_s9);
        av_s11 = _mm_add_ps(av_m10, av_m11);
        av_s12 = _mm_sub_ps(av_s1, av_m2);
        av_s13 = _mm_add_ps(av_m1, av_s3);
        av_m12 = _mm_mul_ps(v_R13_DGC_3, av_s12);
        av_m13 = _mm_mul_ps(v_R13_DGC_2, av_s13);
        av_s14 = _mm_add_ps(av_m12, av_m13);
        av_m14 = _mm_mul_ps(v_R13_DGC_3, av_s13);
        av_m15 = _mm_mul_ps(v_R13_DGC_2, av_s12);
        av_s15 = _mm_sub_ps(av_m14, av_m15);
        av_s16 = _mm_add_ps(av_in3, av_in11);
        av_s17 = _mm_add_ps(av_in9, av_s16);
        av_m16 = _mm_mul_ps(v_CRTM_13_6, av_s16);
        av_s18 = _mm_sub_ps(av_in9, av_m16);
        av_s19 = _mm_sub_ps(av_in3, av_in11);
        av_s20 = _mm_add_ps(av_in5, av_in7);
        av_s21 = _mm_add_ps(av_in1, av_s20);
        av_m17 = _mm_mul_ps(v_CRTM_13_6, av_s20);
        av_s22 = _mm_sub_ps(av_in1, av_m17);
        av_s23 = _mm_sub_ps(av_in5, av_in7);
        av_s24 = _mm_sub_ps(av_s21, av_s17);
        av_m18 = _mm_mul_ps(v_R13_DGC_6, av_s24);
        av_s25 = _mm_add_ps(av_s21, av_s17);
        av_m33 = _mm_mul_ps(v_CRTM_13_7, av_s25);
        // Output point 1: x(0)
        v_out0 = _mm_add_ps(av_m33, av_in0);
        STR_128_S(curr_out, v_out_stride, v_out0);

        av_m34 = _mm_mul_ps(v_R13_DGC_1, av_s25);
        av_s26 = _mm_sub_ps(av_in0, av_m34);
        av_s27 = _mm_add_ps(av_s23, av_s19);
        av_s28 = _mm_add_ps(av_s22, av_s18);
        av_m19 = _mm_mul_ps(v_R13_DGC_9, av_s27);
        av_m20 = _mm_mul_ps(v_R13_DGC_7, av_s28);
        av_s29 = _mm_add_ps(av_m19, av_m20);
        av_s30 = _mm_sub_ps(av_s22, av_s18);
        av_s31 = _mm_sub_ps(av_s23, av_s19);
        av_m21 = _mm_mul_ps(v_R13_DGC_12, av_s30);
        av_m22 = _mm_mul_ps(v_R13_DGC_5, av_s31);
        av_s32 = _mm_sub_ps(av_m21, av_m22);
        av_s61 = _mm_add_ps(av_s6, av_s14);
        av_m23 = _mm_mul_ps(v_CRTM_13_1, av_s61);
        av_s62 = _mm_sub_ps(av_s7, av_s15);
        av_m24 = _mm_mul_ps(v_CRTM_13_1, av_s62);
        av_s33 = _mm_add_ps(av_s7, av_s15);
        av_s34 = _mm_sub_ps(av_s10, av_s33);
        av_m25 = _mm_mul_ps(v_CRTM_13_7, av_s33);
        av_s35 = _mm_add_ps(av_m25, av_s10);
        av_s36 = _mm_sub_ps(av_s6, av_s14);
        av_m26 = _mm_mul_ps(v_CRTM_13_7, av_s36);
        av_s37 = _mm_sub_ps(av_m26, av_s11);
        av_s38 = _mm_add_ps(av_s36, av_s11);
        av_m27 = _mm_mul_ps(v_R13_DGC_4, av_s31);
        av_m28 = _mm_mul_ps(v_CRTM_13_3, av_s30);
        av_s39 = _mm_add_ps(av_m27, av_m28);
        av_m29 = _mm_mul_ps(v_R13_DGC_8, av_s27);
        av_m30 = _mm_mul_ps(v_CRTM_13_2, av_s28);
        av_s40 = _mm_sub_ps(av_m29, av_m30);

        av_s41 = _mm_sub_ps(av_s39, av_s40);
        av_s42 = _mm_add_ps(av_s39, av_s40);
        av_s43 = _mm_sub_ps(av_s26, av_s29);
        av_s44 = _mm_sub_ps(av_m18, av_s32);
        av_s45 = _mm_sub_ps(av_s43, av_s44);
        av_s46 = _mm_add_ps(av_s44, av_s43);
        av_m31 = _mm_mul_ps(v_CRTM_13_7, av_s29);
        av_s47 = _mm_add_ps(av_m31, av_s26);
        av_m32 = _mm_mul_ps(v_CRTM_13_7, av_s32);
        av_s48 = _mm_add_ps(av_m32, av_m18);
        av_s49 = _mm_sub_ps(av_s47, av_s48);
        // Output point 17: x(16)
        v_out16 = _mm_add_ps(av_s49, av_s35);
        curr_out = out + out_strides[16];
        STR_128_S(curr_out, v_out_stride, v_out16);

        // Output point 11: x(10)
        v_out10 = _mm_sub_ps(av_s49, av_s35);
        curr_out = out + out_strides[10];
        STR_128_S(curr_out, v_out_stride, v_out10);

        av_s50 = _mm_add_ps(av_s48, av_s47);
        // Output point 25: x(24)
        v_out24 = _mm_sub_ps(av_s50, av_s37);
        curr_out = out + out_strides[24];
        STR_128_S(curr_out, v_out_stride, v_out24);

        // Output point 3: x(2)
        v_out2 = _mm_add_ps(av_s50, av_s37);
        curr_out = out + out_strides[2];
        STR_128_S(curr_out, v_out_stride, v_out2);

        av_s51 = _mm_sub_ps(av_s45, av_m23);
        av_s52 = _mm_sub_ps(av_s41, av_s34);
        // Output point 5: x(4)
        v_out4 = _mm_add_ps(av_s51, av_s52);
        curr_out = out + out_strides[4];
        STR_128_S(curr_out, v_out_stride, v_out4);

        // Output point 15: x(14)
        v_out14 = _mm_sub_ps(av_s51, av_s52);
        curr_out = out + out_strides[14];
        STR_128_S(curr_out, v_out_stride, v_out14);

        av_s53 = _mm_sub_ps(av_s46, av_s38);
        av_s54 = _mm_add_ps(av_s42, av_m24);
        // Output point 19: x(18)
        v_out18 = _mm_add_ps(av_s54, av_s53);
        curr_out = out + out_strides[18];
        STR_128_S(curr_out, v_out_stride, v_out18);

        // Output point 7: x(6)
        v_out6 = _mm_sub_ps(av_s53, av_s54);
        curr_out = out + out_strides[6];
        STR_128_S(curr_out, v_out_stride, v_out6);

        av_s55 = _mm_sub_ps(av_s42, av_m24);
        av_s56 = _mm_add_ps(av_s46, av_s38);
        // Output point 9: x(8)
        v_out8 = _mm_add_ps(av_s55, av_s56);
        curr_out = out + out_strides[8];
        STR_128_S(curr_out, v_out_stride, v_out8);

        // Output point 21: x(20)
        v_out20 = _mm_sub_ps(av_s56, av_s55);
        curr_out = out + out_strides[20];
        STR_128_S(curr_out, v_out_stride, v_out20);

        av_s57 = _mm_add_ps(av_s45, av_m23);
        av_s58 = _mm_add_ps(av_s41, av_s34);
        // Output point 13: x(12)
        v_out12 = _mm_sub_ps(av_s57, av_s58);
        curr_out = out + out_strides[12];
        STR_128_S(curr_out, v_out_stride, v_out12);

        // Output point 23: x(22)
        v_out22 = _mm_add_ps(av_s57, av_s58);
        curr_out = out + out_strides[22];
        STR_128_S(curr_out, v_out_stride, v_out22);

        // Shifted DFT
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8, bv_in9, bv_in10, bv_in11, bv_in12;
        __m128 bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
               bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16,
               bv_s17, bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24,
               bv_s25, bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32,
               bv_s33, bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40,
               bv_s41, bv_s42, bv_s43, bv_s44, bv_s45, bv_s46, bv_s47, bv_s48,
               bv_s49, bv_s50, bv_s51, bv_s52, bv_s53, bv_s54, bv_s55, bv_s56,
               bv_s57, bv_s58, bv_s59, bv_s60, bv_s61, bv_s62, bv_s63, bv_s64,
               bv_s65, bv_s66, bv_s67, bv_s68, bv_s69, bv_s70;
        __m128 bv_m0, bv_m1, bv_m2, bv_m3, bv_m4, bv_m5, bv_m6, bv_m7, bv_m8,
               bv_m9, bv_m10, bv_m11, bv_m12, bv_m13, bv_m14, bv_m15, bv_m16,
               bv_m17, bv_m18, bv_m19, bv_m20, bv_m21, bv_m22, bv_m23, bv_m24,
               bv_m25, bv_m26, bv_m27, bv_m28, bv_m29, bv_m30, bv_m31, bv_m32,
               bv_m33, bv_m34, bv_m35, bv_m36, bv_m37, bv_m38, bv_m39, bv_m40,
               bv_m41, bv_m42, bv_m43, bv_m44, bv_m45, bv_m46, bv_m47, bv_m48,
               bv_m49, bv_m50, bv_m51, bv_m52, bv_m53, bv_m54, bv_m55, bv_m56,
               bv_m57, bv_m58, bv_m59, bv_m60, bv_m61, bv_m62, bv_m63, bv_m64,
               bv_m65, bv_m66, bv_m67, bv_m68, bv_m69, bv_m70, bv_m71;

        // Input point 2: X(1) & Input point 3: X(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: X(5) & Input point 7: X(6)
        curr_in = in + in_strides[5];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: X(9) & Input point 11: X(10)
        curr_in = in + in_strides[9];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in4, bv_in5);
        // Input point 14: X(13) & Input point 15: X(14)
        curr_in = in + in_strides[13];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in6, bv_in7);
        // Input point 18: X(17) & Input point 19: X(18)
        curr_in = in + in_strides[17];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in8, bv_in9);
        // Input point 22: X(21) & Input point 23: X(22)
        curr_in = in + in_strides[21];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in10, bv_in11);
        // Input point 26: X(25)
        curr_in = in + in_strides[25];
        LDR_128_S(curr_in, v_in_stride, bv_in12);

        bv_s0 = _mm_add_ps(bv_in0, bv_in2);
        bv_s1 = _mm_add_ps(bv_in4, bv_in6);
        bv_s2 = _mm_add_ps(bv_in8, bv_in10);
        bv_s3 = _mm_add_ps(bv_s0, bv_s1);
        bv_s4 = _mm_add_ps(bv_s3, bv_s2);
        // Output point 2: x(1)
        v_out1 = _mm_add_ps(bv_s4, bv_s4);
        v_out1 = _mm_add_ps(v_out1, bv_in12);
        curr_out = out + out_strides[1];
        STR_128_S(curr_out, v_out_stride, v_out1);

        bv_m0 = _mm_mul_ps(v_R13_DFT_C1, bv_in0);
        bv_m1 = _mm_mul_ps(v_R13_DFT_C3, bv_in2);
        bv_s5 = _mm_add_ps(bv_m0, bv_m1);
        bv_m2 = _mm_mul_ps(v_R13_DFT_C5, bv_in4);
        bv_m3 = _mm_mul_ps(v_R13_DFT_C6, bv_in6);
        bv_s6 = _mm_sub_ps(bv_m2, bv_m3);
        bv_m4 = _mm_mul_ps(v_R13_DFT_C4, bv_in8);
        bv_m5 = _mm_mul_ps(v_R13_DFT_C2, bv_in10);
        bv_s7 = _mm_add_ps(bv_m4, bv_m5);
        bv_s8 = _mm_add_ps(bv_s5, bv_s6);
        bv_s9 = _mm_sub_ps(bv_s8, bv_s7);
        bv_s10 = _mm_sub_ps(bv_s9, bv_in12);

        bv_m6 = _mm_mul_ps(v_R13_DFT_S1, bv_in1);
        bv_m7 = _mm_mul_ps(v_R13_DFT_S3, bv_in3);
        bv_s11 = _mm_add_ps(bv_m6, bv_m7);
        bv_m8 = _mm_mul_ps(v_R13_DFT_S5, bv_in5);
        bv_m9 = _mm_mul_ps(v_R13_DFT_S6, bv_in7);
        bv_s12 = _mm_add_ps(bv_m8, bv_m9);
        bv_m10 = _mm_mul_ps(v_R13_DFT_S4, bv_in9);
        bv_m11 = _mm_mul_ps(v_R13_DFT_S2, bv_in11);
        bv_s13 = _mm_add_ps(bv_m10, bv_m11);
        bv_s14 = _mm_add_ps(bv_s11, bv_s12);
        bv_s15 = _mm_add_ps(bv_s14, bv_s13);

        // Output point 4: x(3)
        v_out3 = _mm_sub_ps(bv_s10, bv_s15);
        curr_out = out + out_strides[3];
        STR_128_S(curr_out, v_out_stride, v_out3);

        // Output point 26: x(25)
        v_out25 = _mm_sub_ps(NEGATE_128_S(bv_s10), bv_s15);
        curr_out = out + out_strides[25];
        STR_128_S(curr_out, v_out_stride, v_out25);

        bv_m12 = _mm_mul_ps(v_R13_DFT_C2, bv_in0);
        bv_m13 = _mm_mul_ps(v_R13_DFT_C6, bv_in2);
        bv_s16 = _mm_add_ps(bv_m12, bv_m13);
        bv_m14 = _mm_mul_ps(v_R13_DFT_C3, bv_in4);
        bv_m15 = _mm_mul_ps(v_R13_DFT_C1, bv_in6);
        bv_s17 = _mm_add_ps(bv_m14, bv_m15);
        bv_m16 = _mm_mul_ps(v_R13_DFT_C5, bv_in8);
        bv_m17 = _mm_mul_ps(v_R13_DFT_C4, bv_in10);
        bv_s18 = _mm_sub_ps(bv_m17, bv_m16);
        bv_s19 = _mm_sub_ps(bv_s16, bv_s17);
        bv_s20 = _mm_add_ps(bv_s19, bv_s18);
        bv_s21 = _mm_add_ps(bv_s20, bv_in12);

        bv_m18 = _mm_mul_ps(v_R13_DFT_S2, bv_in1);
        bv_m19 = _mm_mul_ps(v_R13_DFT_S6, bv_in3);
        bv_s22 = _mm_add_ps(bv_m18, bv_m19);
        bv_m20 = _mm_mul_ps(v_R13_DFT_S3, bv_in5);
        bv_m21 = _mm_mul_ps(v_R13_DFT_S1, bv_in7);
        bv_s23 = _mm_sub_ps(bv_m20, bv_m21);
        bv_m22 = _mm_mul_ps(v_R13_DFT_S5, bv_in9);
        bv_m23 = _mm_mul_ps(v_R13_DFT_S4, bv_in11);
        bv_s24 = _mm_add_ps(bv_m22, bv_m23);
        bv_s25 = _mm_add_ps(bv_s22, bv_s23);
        bv_s26 = _mm_sub_ps(bv_s25, bv_s24);

        // Output point 6: x(5)
        v_out5 = _mm_sub_ps(bv_s21, bv_s26);
        curr_out = out + out_strides[5];
        STR_128_S(curr_out, v_out_stride, v_out5);

        // Output point 24: x(23)
        v_out23 = _mm_sub_ps(NEGATE_128_S(bv_s21), bv_s26);
        curr_out = out + out_strides[23];
        STR_128_S(curr_out, v_out_stride, v_out23);

        bv_m24 = _mm_mul_ps(v_R13_DFT_C3, bv_in0);
        bv_m25 = _mm_mul_ps(v_R13_DFT_C4, bv_in2);
        bv_s27 = _mm_sub_ps(bv_m24, bv_m25);
        bv_m26 = _mm_mul_ps(v_R13_DFT_C2, bv_in4);
        bv_m27 = _mm_mul_ps(v_R13_DFT_C5, bv_in6);
        bv_s28 = _mm_sub_ps(bv_m27, bv_m26);
        bv_m28 = _mm_mul_ps(v_R13_DFT_C1, bv_in8);
        bv_m29 = _mm_mul_ps(v_R13_DFT_C6, bv_in10);
        bv_s29 = _mm_sub_ps(bv_m28, bv_m29);
        bv_s30 = _mm_add_ps(bv_s27, bv_s28);
        bv_s31 = _mm_add_ps(bv_s30, bv_s29);
        bv_s32 = _mm_sub_ps(bv_s31, bv_in12);

        bv_m30 = _mm_mul_ps(v_R13_DFT_S3, bv_in1);
        bv_m31 = _mm_mul_ps(v_R13_DFT_S4, bv_in3);
        bv_s33 = _mm_add_ps(bv_m30, bv_m31);
        bv_m32 = _mm_mul_ps(v_R13_DFT_S2, bv_in5);
        bv_m33 = _mm_mul_ps(v_R13_DFT_S5, bv_in7);
        bv_s34 = _mm_add_ps(bv_m32, bv_m33);
        bv_m34 = _mm_mul_ps(v_R13_DFT_S1, bv_in9);
        bv_m35 = _mm_mul_ps(v_R13_DFT_S6, bv_in11);
        bv_s35 = _mm_add_ps(bv_m34, bv_m35);
        bv_s36 = _mm_sub_ps(bv_s33, bv_s34);
        bv_s37 = _mm_add_ps(bv_s36, bv_s35);

        // Output point 8: x(7)
        v_out7 = _mm_sub_ps(bv_s32, bv_s37);
        curr_out = out + out_strides[7];
        STR_128_S(curr_out, v_out_stride, v_out7);

        // Output point 22: x(21)
        v_out21 = _mm_sub_ps(NEGATE_128_S(bv_s32), bv_s37);
        curr_out = out + out_strides[21];
        STR_128_S(curr_out, v_out_stride, v_out21);

        bv_m36 = _mm_mul_ps(v_R13_DFT_C4, bv_in0);
        bv_m37 = _mm_mul_ps(v_R13_DFT_C1, bv_in2);
        bv_s38 = _mm_sub_ps(bv_m36, bv_m37);
        bv_m38 = _mm_mul_ps(v_R13_DFT_C6, bv_in4);
        bv_m39 = _mm_mul_ps(v_R13_DFT_C2, bv_in6);
        bv_s39 = _mm_add_ps(bv_m38, bv_m39);
        bv_m40 = _mm_mul_ps(v_R13_DFT_C3, bv_in8);
        bv_m41 = _mm_mul_ps(v_R13_DFT_C5, bv_in10);
        bv_s40 = _mm_add_ps(bv_m40, bv_m41);
        bv_s41 = _mm_add_ps(bv_s38, bv_s39);
        bv_s42 = _mm_sub_ps(bv_s41, bv_s40);
        bv_s43 = _mm_add_ps(bv_s42, bv_in12);

        bv_m42 = _mm_mul_ps(v_R13_DFT_S4, bv_in1);
        bv_m43 = _mm_mul_ps(v_R13_DFT_S1, bv_in3);
        bv_s44 = _mm_add_ps(bv_m42, bv_m43);
        bv_m44 = _mm_mul_ps(v_R13_DFT_S6, bv_in5);
        bv_m45 = _mm_mul_ps(v_R13_DFT_S2, bv_in7);
        bv_s45 = _mm_sub_ps(bv_m45, bv_m44);
        bv_m46 = _mm_mul_ps(v_R13_DFT_S3, bv_in9);
        bv_m47 = _mm_mul_ps(v_R13_DFT_S5, bv_in11);
        bv_s46 = _mm_sub_ps(bv_m46, bv_m47);
        bv_s47 = _mm_add_ps(bv_s44, bv_s45);
        bv_s48 = _mm_add_ps(bv_s47, bv_s46);

        // Output point 10: x(9)
        v_out9 = _mm_sub_ps(bv_s43, bv_s48);
        curr_out = out + out_strides[9];
        STR_128_S(curr_out, v_out_stride, v_out9);

        // Output point 20: x(19)
        v_out19 = _mm_sub_ps(NEGATE_128_S(bv_s43), bv_s48);
        curr_out = out + out_strides[19];
        STR_128_S(curr_out, v_out_stride, v_out19);

        bv_m48 = _mm_mul_ps(v_R13_DFT_C5, bv_in0);
        bv_m49 = _mm_mul_ps(v_R13_DFT_C2, bv_in2);
        bv_s49 = _mm_sub_ps(bv_m48, bv_m49);
        bv_m50 = _mm_mul_ps(v_R13_DFT_C1, bv_in4);
        bv_m51 = _mm_mul_ps(v_R13_DFT_C4, bv_in6);
        bv_s50 = _mm_sub_ps(bv_m50, bv_m51);
        bv_m52 = _mm_mul_ps(v_R13_DFT_C6, bv_in8);
        bv_m53 = _mm_mul_ps(v_R13_DFT_C3, bv_in10);
        bv_s51 = _mm_sub_ps(bv_m53, bv_m52);
        bv_s52 = _mm_add_ps(bv_s49, bv_s50);
        bv_s53 = _mm_add_ps(bv_s52, bv_s51);
        bv_s54 = _mm_sub_ps(bv_s53, bv_in12);

        bv_m54 = _mm_mul_ps(v_R13_DFT_S5, bv_in1);
        bv_m55 = _mm_mul_ps(v_R13_DFT_S2, bv_in3);
        bv_s55 = _mm_sub_ps(bv_m54, bv_m55);
        bv_m56 = _mm_mul_ps(v_R13_DFT_S1, bv_in5);
        bv_m57 = _mm_mul_ps(v_R13_DFT_S4, bv_in7);
        bv_s56 = _mm_sub_ps(bv_m57, bv_m56);
        bv_m58 = _mm_mul_ps(v_R13_DFT_S6, bv_in9);
        bv_m59 = _mm_mul_ps(v_R13_DFT_S3, bv_in11);
        bv_s57 = _mm_sub_ps(bv_m59, bv_m58);
        bv_s58 = _mm_add_ps(bv_s55, bv_s56);
        bv_s59 = _mm_add_ps(bv_s58, bv_s57);

        // Output point 12: x(11)
        v_out11 = _mm_sub_ps(bv_s54, bv_s59);
        curr_out = out + out_strides[11];
        STR_128_S(curr_out, v_out_stride, v_out11);

        // Output point 18: x(17)
        v_out17 = _mm_sub_ps(NEGATE_128_S(bv_s54), bv_s59);
        curr_out = out + out_strides[17];
        STR_128_S(curr_out, v_out_stride, v_out17);

        bv_m60 = _mm_mul_ps(v_R13_DFT_C6, bv_in0);
        bv_m61 = _mm_mul_ps(v_R13_DFT_C5, bv_in2);
        bv_s60 = _mm_sub_ps(bv_m60, bv_m61);
        bv_m62 = _mm_mul_ps(v_R13_DFT_C4, bv_in4);
        bv_m63 = _mm_mul_ps(v_R13_DFT_C3, bv_in6);
        bv_s61 = _mm_sub_ps(bv_m62, bv_m63);
        bv_m64 = _mm_mul_ps(v_R13_DFT_C2, bv_in8);
        bv_m65 = _mm_mul_ps(v_R13_DFT_C1, bv_in10);
        bv_s62 = _mm_sub_ps(bv_m64, bv_m65);
        bv_s63 = _mm_add_ps(bv_s60, bv_s61);
        bv_s64 = _mm_add_ps(bv_s63, bv_s62);
        bv_s65 = _mm_add_ps(bv_s64, bv_in12);

        bv_m66 = _mm_mul_ps(v_R13_DFT_S6, bv_in1);
        bv_m67 = _mm_mul_ps(v_R13_DFT_S5, bv_in3);
        bv_s66 = _mm_sub_ps(bv_m66, bv_m67);
        bv_m68 = _mm_mul_ps(v_R13_DFT_S4, bv_in5);
        bv_m69 = _mm_mul_ps(v_R13_DFT_S3, bv_in7);
        bv_s67 = _mm_sub_ps(bv_m68, bv_m69);
        bv_m70 = _mm_mul_ps(v_R13_DFT_S2, bv_in9);
        bv_m71 = _mm_mul_ps(v_R13_DFT_S1, bv_in11);
        bv_s68 = _mm_sub_ps(bv_m70, bv_m71);
        bv_s69 = _mm_add_ps(bv_s66, bv_s67);
        bv_s70 = _mm_add_ps(bv_s69, bv_s68);

        // Output point 14: x(13)
        v_out13 = _mm_sub_ps(bv_s65, bv_s70);
        curr_out = out + out_strides[13];
        STR_128_S(curr_out, v_out_stride, v_out13);

        // Output point 16: x(15)
        v_out15 = _mm_sub_ps(NEGATE_128_S(bv_s65), bv_s70);
        curr_out = out + out_strides[15];
        STR_128_S(curr_out, v_out_stride, v_out15);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & 2)
    {
        // Standard DFT
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
                av_in8, av_in9, av_in10, av_in11, av_in12;
        __m128 av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
                av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16,
                av_s17, av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24,
                av_s25, av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32,
                av_s33, av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40,
                av_s41, av_s42, av_s43, av_s44, av_s45, av_s46, av_s47, av_s48,
                av_s49, av_s50, av_s51, av_s52, av_s53, av_s54, av_s55, av_s56,
                av_s57, av_s58, av_s59, av_s60, av_s61, av_s62;
        __m128 av_m0, av_m1, av_m2, av_m3, av_m4, av_m5, av_m6, av_m7, av_m8,
               av_m9, av_m10, av_m11, av_m12, av_m13, av_m14, av_m15, av_m16,
               av_m17, av_m18, av_m19, av_m20, av_m21, av_m22, av_m23, av_m24,
               av_m25, av_m26, av_m27, av_m28, av_m29, av_m30, av_m31, av_m32,
               av_m33, av_m34;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21,
               v_out22, v_out23, v_out24, v_out25;

        curr_in = in;
        curr_out = out;

        // Input point 1: X(0)
        LDHR_128_S(curr_in, v_in_stride, av_in0);
        curr_in = in + in_strides[3];
        // Input point 4: X(3) & Input point 5: X(4)
        LDHRI_2x128_S(curr_in, v_in_stride, av_in1, av_in2);
        curr_in = in + in_strides[7];
        // Input point 8: X(7) & Input point 9: X(8)
        LDHRI_2x128_S(curr_in, v_in_stride, av_in3, av_in4);
        curr_in = in + in_strides[11];
        // Input point 12: X(11) & Input point 13: X(12)
        LDHRI_2x128_S(curr_in, v_in_stride, av_in5, av_in6);
        curr_in = in + in_strides[15];
        // Input point 16: X(15) & Input point 17: X(16)
        LDHRI_2x128_S(curr_in, v_in_stride, av_in7, av_in8);
        curr_in = in + in_strides[19];
        // Input point 20: X(19) & Input point 21: X(20)
        LDHRI_2x128_S(curr_in, v_in_stride, av_in9, av_in10);
        curr_in = in + in_strides[23];
        // Input point 24: X(23) & Input point 25: X(24)
        LDHRI_2x128_S(curr_in, v_in_stride, av_in11, av_in12);

        av_s0 = _mm_sub_ps(av_in6, av_in8);
        av_m0 = _mm_mul_ps(v_CRTM_13_7, av_in2);
        av_s1 = _mm_sub_ps(av_m0, av_s0);
        av_s59 = _mm_add_ps(av_in6, av_in8);
        av_m1 = _mm_mul_ps(v_CRTM_13_1, av_s59);
        av_s2 = _mm_add_ps(av_in12, av_in4);
        av_s60 = _mm_sub_ps(av_in12, av_in4);
        av_m2 = _mm_mul_ps(v_CRTM_13_1, av_s60);
        av_m3 = _mm_mul_ps(v_CRTM_13_7, av_in10);
        av_s3 = _mm_sub_ps(av_s2, av_m3);
        av_s4 = _mm_add_ps(av_s1, av_m2);
        av_s5 = _mm_sub_ps(av_s3, av_m1);
        av_m4 = _mm_mul_ps(v_R13_DGC_11, av_s4);
        av_m5 = _mm_mul_ps(v_R13_DGC_10, av_s5);
        av_s6 = _mm_add_ps(av_m4, av_m5);
        av_m6 = _mm_mul_ps(v_R13_DGC_10, av_s4);
        av_m7 = _mm_mul_ps(v_R13_DGC_11, av_s5);
        av_s7 = _mm_sub_ps(av_m6, av_m7);
        av_s8 = _mm_add_ps(av_in2, av_s0);
        av_s9 = _mm_add_ps(av_s2, av_in10);
        av_m8 = _mm_mul_ps(v_CRTM_13_4, av_s8);
        av_m9 = _mm_mul_ps(v_CRTM_13_5, av_s9);
        av_s10 = _mm_sub_ps(av_m8, av_m9);
        av_m10 = _mm_mul_ps(v_CRTM_13_5, av_s8);
        av_m11 = _mm_mul_ps(v_CRTM_13_4, av_s9);
        av_s11 = _mm_add_ps(av_m10, av_m11);
        av_s12 = _mm_sub_ps(av_s1, av_m2);
        av_s13 = _mm_add_ps(av_m1, av_s3);
        av_m12 = _mm_mul_ps(v_R13_DGC_3, av_s12);
        av_m13 = _mm_mul_ps(v_R13_DGC_2, av_s13);
        av_s14 = _mm_add_ps(av_m12, av_m13);
        av_m14 = _mm_mul_ps(v_R13_DGC_3, av_s13);
        av_m15 = _mm_mul_ps(v_R13_DGC_2, av_s12);
        av_s15 = _mm_sub_ps(av_m14, av_m15);
        av_s16 = _mm_add_ps(av_in3, av_in11);
        av_s17 = _mm_add_ps(av_in9, av_s16);
        av_m16 = _mm_mul_ps(v_CRTM_13_6, av_s16);
        av_s18 = _mm_sub_ps(av_in9, av_m16);
        av_s19 = _mm_sub_ps(av_in3, av_in11);
        av_s20 = _mm_add_ps(av_in5, av_in7);
        av_s21 = _mm_add_ps(av_in1, av_s20);
        av_m17 = _mm_mul_ps(v_CRTM_13_6, av_s20);
        av_s22 = _mm_sub_ps(av_in1, av_m17);
        av_s23 = _mm_sub_ps(av_in5, av_in7);
        av_s24 = _mm_sub_ps(av_s21, av_s17);
        av_m18 = _mm_mul_ps(v_R13_DGC_6, av_s24);
        av_s25 = _mm_add_ps(av_s21, av_s17);
        av_m33 = _mm_mul_ps(v_CRTM_13_7, av_s25);
        // Output point 1: x(0)
        v_out0 = _mm_add_ps(av_m33, av_in0);
        STHR_128_S(curr_out, v_out_stride, v_out0);

        av_m34 = _mm_mul_ps(v_R13_DGC_1, av_s25);
        av_s26 = _mm_sub_ps(av_in0, av_m34);
        av_s27 = _mm_add_ps(av_s23, av_s19);
        av_s28 = _mm_add_ps(av_s22, av_s18);
        av_m19 = _mm_mul_ps(v_R13_DGC_9, av_s27);
        av_m20 = _mm_mul_ps(v_R13_DGC_7, av_s28);
        av_s29 = _mm_add_ps(av_m19, av_m20);
        av_s30 = _mm_sub_ps(av_s22, av_s18);
        av_s31 = _mm_sub_ps(av_s23, av_s19);
        av_m21 = _mm_mul_ps(v_R13_DGC_12, av_s30);
        av_m22 = _mm_mul_ps(v_R13_DGC_5, av_s31);
        av_s32 = _mm_sub_ps(av_m21, av_m22);
        av_s61 = _mm_add_ps(av_s6, av_s14);
        av_m23 = _mm_mul_ps(v_CRTM_13_1, av_s61);
        av_s62 = _mm_sub_ps(av_s7, av_s15);
        av_m24 = _mm_mul_ps(v_CRTM_13_1, av_s62);
        av_s33 = _mm_add_ps(av_s7, av_s15);
        av_s34 = _mm_sub_ps(av_s10, av_s33);
        av_m25 = _mm_mul_ps(v_CRTM_13_7, av_s33);
        av_s35 = _mm_add_ps(av_m25, av_s10);
        av_s36 = _mm_sub_ps(av_s6, av_s14);
        av_m26 = _mm_mul_ps(v_CRTM_13_7, av_s36);
        av_s37 = _mm_sub_ps(av_m26, av_s11);
        av_s38 = _mm_add_ps(av_s36, av_s11);
        av_m27 = _mm_mul_ps(v_R13_DGC_4, av_s31);
        av_m28 = _mm_mul_ps(v_CRTM_13_3, av_s30);
        av_s39 = _mm_add_ps(av_m27, av_m28);
        av_m29 = _mm_mul_ps(v_R13_DGC_8, av_s27);
        av_m30 = _mm_mul_ps(v_CRTM_13_2, av_s28);
        av_s40 = _mm_sub_ps(av_m29, av_m30);
        av_s41 = _mm_sub_ps(av_s39, av_s40);
        av_s42 = _mm_add_ps(av_s39, av_s40);
        av_s43 = _mm_sub_ps(av_s26, av_s29);
        av_s44 = _mm_sub_ps(av_m18, av_s32);
        av_s45 = _mm_sub_ps(av_s43, av_s44);
        av_s46 = _mm_add_ps(av_s44, av_s43);
        av_m31 = _mm_mul_ps(v_CRTM_13_7, av_s29);
        av_s47 = _mm_add_ps(av_m31, av_s26);
        av_m32 = _mm_mul_ps(v_CRTM_13_7, av_s32);
        av_s48 = _mm_add_ps(av_m32, av_m18);
        av_s49 = _mm_sub_ps(av_s47, av_s48);
        // Output point 17: x(16)
        v_out16 = _mm_add_ps(av_s49, av_s35);
        curr_out = out + out_strides[16];
        STHR_128_S(curr_out, v_out_stride, v_out16);

        // Output point 11: x(10)
        v_out10 = _mm_sub_ps(av_s49, av_s35);
        curr_out = out + out_strides[10];
        STHR_128_S(curr_out, v_out_stride, v_out10);

        av_s50 = _mm_add_ps(av_s48, av_s47);
        // Output point 25: x(24)
        v_out24 = _mm_sub_ps(av_s50, av_s37);
        curr_out = out + out_strides[24];
        STHR_128_S(curr_out, v_out_stride, v_out24);

        // Output point 3: x(2)
        v_out2 = _mm_add_ps(av_s50, av_s37);
        curr_out = out + out_strides[2];
        STHR_128_S(curr_out, v_out_stride, v_out2);

        av_s51 = _mm_sub_ps(av_s45, av_m23);
        av_s52 = _mm_sub_ps(av_s41, av_s34);
        // Output point 5: x(4)
        v_out4 = _mm_add_ps(av_s51, av_s52);
        curr_out = out + out_strides[4];
        STHR_128_S(curr_out, v_out_stride, v_out4);

        // Output point 15: x(14)
        v_out14 = _mm_sub_ps(av_s51, av_s52);
        curr_out = out + out_strides[14];
        STHR_128_S(curr_out, v_out_stride, v_out14);

        av_s53 = _mm_sub_ps(av_s46, av_s38);
        av_s54 = _mm_add_ps(av_s42, av_m24);
        // Output point 19: x(18)
        v_out18 = _mm_add_ps(av_s54, av_s53);
        curr_out = out + out_strides[18];
        STHR_128_S(curr_out, v_out_stride, v_out18);

        // Output point 7: x(6)
        v_out6 = _mm_sub_ps(av_s53, av_s54);
        curr_out = out + out_strides[6];
        STHR_128_S(curr_out, v_out_stride, v_out6);

        av_s55 = _mm_sub_ps(av_s42, av_m24);
        av_s56 = _mm_add_ps(av_s46, av_s38);
        // Output point 9: x(8)
        v_out8 = _mm_add_ps(av_s55, av_s56);
        curr_out = out + out_strides[8];
        STHR_128_S(curr_out, v_out_stride, v_out8);

        // Output point 21: x(20)
        v_out20 = _mm_sub_ps(av_s56, av_s55);
        curr_out = out + out_strides[20];
        STHR_128_S(curr_out, v_out_stride, v_out20);

        av_s57 = _mm_add_ps(av_s45, av_m23);
        av_s58 = _mm_add_ps(av_s41, av_s34);
        // Output point 13: x(12)
        v_out12 = _mm_sub_ps(av_s57, av_s58);
        curr_out = out + out_strides[12];
        STHR_128_S(curr_out, v_out_stride, v_out12);

        // Output point 23: x(22)
        v_out22 = _mm_add_ps(av_s57, av_s58);
        curr_out = out + out_strides[22];
        STHR_128_S(curr_out, v_out_stride, v_out22);

        // Shifted DFT
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8, bv_in9, bv_in10, bv_in11, bv_in12;
        __m128 bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
               bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16,
               bv_s17, bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24,
               bv_s25, bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32,
               bv_s33, bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40,
               bv_s41, bv_s42, bv_s43, bv_s44, bv_s45, bv_s46, bv_s47, bv_s48,
               bv_s49, bv_s50, bv_s51, bv_s52, bv_s53, bv_s54, bv_s55, bv_s56,
               bv_s57, bv_s58, bv_s59, bv_s60, bv_s61, bv_s62, bv_s63, bv_s64,
               bv_s65, bv_s66, bv_s67, bv_s68, bv_s69, bv_s70;
        __m128 bv_m0, bv_m1, bv_m2, bv_m3, bv_m4, bv_m5, bv_m6, bv_m7, bv_m8,
               bv_m9, bv_m10, bv_m11, bv_m12, bv_m13, bv_m14, bv_m15, bv_m16,
               bv_m17, bv_m18, bv_m19, bv_m20, bv_m21, bv_m22, bv_m23, bv_m24,
               bv_m25, bv_m26, bv_m27, bv_m28, bv_m29, bv_m30, bv_m31, bv_m32,
               bv_m33, bv_m34, bv_m35, bv_m36, bv_m37, bv_m38, bv_m39, bv_m40,
               bv_m41, bv_m42, bv_m43, bv_m44, bv_m45, bv_m46, bv_m47, bv_m48,
               bv_m49, bv_m50, bv_m51, bv_m52, bv_m53, bv_m54, bv_m55, bv_m56,
               bv_m57, bv_m58, bv_m59, bv_m60, bv_m61, bv_m62, bv_m63, bv_m64,
               bv_m65, bv_m66, bv_m67, bv_m68, bv_m69, bv_m70, bv_m71;

        // Input point 2: X(1) & Input point 3: X(2)
        curr_in = in + in_strides[1];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: X(5) & Input point 7: X(6)
        curr_in = in + in_strides[5];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: X(9) & Input point 11: X(10)
        curr_in = in + in_strides[9];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in4, bv_in5);
        // Input point 14: X(13) & Input point 15: X(14)
        curr_in = in + in_strides[13];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in6, bv_in7);
        // Input point 18: X(17) & Input point 19: X(18)
        curr_in = in + in_strides[17];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in8, bv_in9);
        // Input point 22: X(21) & Input point 23: X(22)
        curr_in = in + in_strides[21];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in10, bv_in11);
        // Input point 26: X(25)
        curr_in = in + in_strides[25];
        LDHR_128_S(curr_in, v_in_stride, bv_in12);

        bv_s0 = _mm_add_ps(bv_in0, bv_in2);
        bv_s1 = _mm_add_ps(bv_in4, bv_in6);
        bv_s2 = _mm_add_ps(bv_in8, bv_in10);
        bv_s3 = _mm_add_ps(bv_s0, bv_s1);
        bv_s4 = _mm_add_ps(bv_s3, bv_s2);
        // Output point 2: x(1)
        v_out1 = _mm_add_ps(bv_s4, bv_s4);
        v_out1 = _mm_add_ps(v_out1, bv_in12);
        curr_out = out + out_strides[1];
        STHR_128_S(curr_out, v_out_stride, v_out1);

        bv_m0 = _mm_mul_ps(v_R13_DFT_C1, bv_in0);
        bv_m1 = _mm_mul_ps(v_R13_DFT_C3, bv_in2);
        bv_s5 = _mm_add_ps(bv_m0, bv_m1);
        bv_m2 = _mm_mul_ps(v_R13_DFT_C5, bv_in4);
        bv_m3 = _mm_mul_ps(v_R13_DFT_C6, bv_in6);
        bv_s6 = _mm_sub_ps(bv_m2, bv_m3);
        bv_m4 = _mm_mul_ps(v_R13_DFT_C4, bv_in8);
        bv_m5 = _mm_mul_ps(v_R13_DFT_C2, bv_in10);
        bv_s7 = _mm_add_ps(bv_m4, bv_m5);
        bv_s8 = _mm_add_ps(bv_s5, bv_s6);
        bv_s9 = _mm_sub_ps(bv_s8, bv_s7);
        bv_s10 = _mm_sub_ps(bv_s9, bv_in12);

        bv_m6 = _mm_mul_ps(v_R13_DFT_S1, bv_in1);
        bv_m7 = _mm_mul_ps(v_R13_DFT_S3, bv_in3);
        bv_s11 = _mm_add_ps(bv_m6, bv_m7);
        bv_m8 = _mm_mul_ps(v_R13_DFT_S5, bv_in5);
        bv_m9 = _mm_mul_ps(v_R13_DFT_S6, bv_in7);
        bv_s12 = _mm_add_ps(bv_m8, bv_m9);
        bv_m10 = _mm_mul_ps(v_R13_DFT_S4, bv_in9);
        bv_m11 = _mm_mul_ps(v_R13_DFT_S2, bv_in11);
        bv_s13 = _mm_add_ps(bv_m10, bv_m11);
        bv_s14 = _mm_add_ps(bv_s11, bv_s12);
        bv_s15 = _mm_add_ps(bv_s14, bv_s13);

        // Output point 4: x(3)
        v_out3 = _mm_sub_ps(bv_s10, bv_s15);
        curr_out = out + out_strides[3];
        STHR_128_S(curr_out, v_out_stride, v_out3);

        // Output point 26: x(25)
        v_out25 = _mm_sub_ps(NEGATE_128_S(bv_s10), bv_s15);
        curr_out = out + out_strides[25];
        STHR_128_S(curr_out, v_out_stride, v_out25);

        bv_m12 = _mm_mul_ps(v_R13_DFT_C2, bv_in0);
        bv_m13 = _mm_mul_ps(v_R13_DFT_C6, bv_in2);
        bv_s16 = _mm_add_ps(bv_m12, bv_m13);
        bv_m14 = _mm_mul_ps(v_R13_DFT_C3, bv_in4);
        bv_m15 = _mm_mul_ps(v_R13_DFT_C1, bv_in6);
        bv_s17 = _mm_add_ps(bv_m14, bv_m15);
        bv_m16 = _mm_mul_ps(v_R13_DFT_C5, bv_in8);
        bv_m17 = _mm_mul_ps(v_R13_DFT_C4, bv_in10);
        bv_s18 = _mm_sub_ps(bv_m17, bv_m16);
        bv_s19 = _mm_sub_ps(bv_s16, bv_s17);
        bv_s20 = _mm_add_ps(bv_s19, bv_s18);
        bv_s21 = _mm_add_ps(bv_s20, bv_in12);

        bv_m18 = _mm_mul_ps(v_R13_DFT_S2, bv_in1);
        bv_m19 = _mm_mul_ps(v_R13_DFT_S6, bv_in3);
        bv_s22 = _mm_add_ps(bv_m18, bv_m19);
        bv_m20 = _mm_mul_ps(v_R13_DFT_S3, bv_in5);
        bv_m21 = _mm_mul_ps(v_R13_DFT_S1, bv_in7);
        bv_s23 = _mm_sub_ps(bv_m20, bv_m21);
        bv_m22 = _mm_mul_ps(v_R13_DFT_S5, bv_in9);
        bv_m23 = _mm_mul_ps(v_R13_DFT_S4, bv_in11);
        bv_s24 = _mm_add_ps(bv_m22, bv_m23);
        bv_s25 = _mm_add_ps(bv_s22, bv_s23);
        bv_s26 = _mm_sub_ps(bv_s25, bv_s24);

        // Output point 6: x(5)
        v_out5 = _mm_sub_ps(bv_s21, bv_s26);
        curr_out = out + out_strides[5];
        STHR_128_S(curr_out, v_out_stride, v_out5);

        // Output point 24: x(23)
        v_out23 = _mm_sub_ps(NEGATE_128_S(bv_s21), bv_s26);
        curr_out = out + out_strides[23];
        STHR_128_S(curr_out, v_out_stride, v_out23);

        bv_m24 = _mm_mul_ps(v_R13_DFT_C3, bv_in0);
        bv_m25 = _mm_mul_ps(v_R13_DFT_C4, bv_in2);
        bv_s27 = _mm_sub_ps(bv_m24, bv_m25);
        bv_m26 = _mm_mul_ps(v_R13_DFT_C2, bv_in4);
        bv_m27 = _mm_mul_ps(v_R13_DFT_C5, bv_in6);
        bv_s28 = _mm_sub_ps(bv_m27, bv_m26);
        bv_m28 = _mm_mul_ps(v_R13_DFT_C1, bv_in8);
        bv_m29 = _mm_mul_ps(v_R13_DFT_C6, bv_in10);
        bv_s29 = _mm_sub_ps(bv_m28, bv_m29);
        bv_s30 = _mm_add_ps(bv_s27, bv_s28);
        bv_s31 = _mm_add_ps(bv_s30, bv_s29);
        bv_s32 = _mm_sub_ps(bv_s31, bv_in12);

        bv_m30 = _mm_mul_ps(v_R13_DFT_S3, bv_in1);
        bv_m31 = _mm_mul_ps(v_R13_DFT_S4, bv_in3);
        bv_s33 = _mm_add_ps(bv_m30, bv_m31);
        bv_m32 = _mm_mul_ps(v_R13_DFT_S2, bv_in5);
        bv_m33 = _mm_mul_ps(v_R13_DFT_S5, bv_in7);
        bv_s34 = _mm_add_ps(bv_m32, bv_m33);
        bv_m34 = _mm_mul_ps(v_R13_DFT_S1, bv_in9);
        bv_m35 = _mm_mul_ps(v_R13_DFT_S6, bv_in11);
        bv_s35 = _mm_add_ps(bv_m34, bv_m35);
        bv_s36 = _mm_sub_ps(bv_s33, bv_s34);
        bv_s37 = _mm_add_ps(bv_s36, bv_s35);

        // Output point 8: x(7)
        v_out7 = _mm_sub_ps(bv_s32, bv_s37);
        curr_out = out + out_strides[7];
        STHR_128_S(curr_out, v_out_stride, v_out7);

        // Output point 22: x(21)
        v_out21 = _mm_sub_ps(NEGATE_128_S(bv_s32), bv_s37);
        curr_out = out + out_strides[21];
        STHR_128_S(curr_out, v_out_stride, v_out21);

        bv_m36 = _mm_mul_ps(v_R13_DFT_C4, bv_in0);
        bv_m37 = _mm_mul_ps(v_R13_DFT_C1, bv_in2);
        bv_s38 = _mm_sub_ps(bv_m36, bv_m37);
        bv_m38 = _mm_mul_ps(v_R13_DFT_C6, bv_in4);
        bv_m39 = _mm_mul_ps(v_R13_DFT_C2, bv_in6);
        bv_s39 = _mm_add_ps(bv_m38, bv_m39);
        bv_m40 = _mm_mul_ps(v_R13_DFT_C3, bv_in8);
        bv_m41 = _mm_mul_ps(v_R13_DFT_C5, bv_in10);
        bv_s40 = _mm_add_ps(bv_m40, bv_m41);
        bv_s41 = _mm_add_ps(bv_s38, bv_s39);
        bv_s42 = _mm_sub_ps(bv_s41, bv_s40);
        bv_s43 = _mm_add_ps(bv_s42, bv_in12);

        bv_m42 = _mm_mul_ps(v_R13_DFT_S4, bv_in1);
        bv_m43 = _mm_mul_ps(v_R13_DFT_S1, bv_in3);
        bv_s44 = _mm_add_ps(bv_m42, bv_m43);
        bv_m44 = _mm_mul_ps(v_R13_DFT_S6, bv_in5);
        bv_m45 = _mm_mul_ps(v_R13_DFT_S2, bv_in7);
        bv_s45 = _mm_sub_ps(bv_m45, bv_m44);
        bv_m46 = _mm_mul_ps(v_R13_DFT_S3, bv_in9);
        bv_m47 = _mm_mul_ps(v_R13_DFT_S5, bv_in11);
        bv_s46 = _mm_sub_ps(bv_m46, bv_m47);
        bv_s47 = _mm_add_ps(bv_s44, bv_s45);
        bv_s48 = _mm_add_ps(bv_s47, bv_s46);

        // Output point 10: x(9)
        v_out9 = _mm_sub_ps(bv_s43, bv_s48);
        curr_out = out + out_strides[9];
        STHR_128_S(curr_out, v_out_stride, v_out9);

        // Output point 20: x(19)
        v_out19 = _mm_sub_ps(NEGATE_128_S(bv_s43), bv_s48);
        curr_out = out + out_strides[19];
        STHR_128_S(curr_out, v_out_stride, v_out19);

        bv_m48 = _mm_mul_ps(v_R13_DFT_C5, bv_in0);
        bv_m49 = _mm_mul_ps(v_R13_DFT_C2, bv_in2);
        bv_s49 = _mm_sub_ps(bv_m48, bv_m49);
        bv_m50 = _mm_mul_ps(v_R13_DFT_C1, bv_in4);
        bv_m51 = _mm_mul_ps(v_R13_DFT_C4, bv_in6);
        bv_s50 = _mm_sub_ps(bv_m50, bv_m51);
        bv_m52 = _mm_mul_ps(v_R13_DFT_C6, bv_in8);
        bv_m53 = _mm_mul_ps(v_R13_DFT_C3, bv_in10);
        bv_s51 = _mm_sub_ps(bv_m53, bv_m52);
        bv_s52 = _mm_add_ps(bv_s49, bv_s50);
        bv_s53 = _mm_add_ps(bv_s52, bv_s51);
        bv_s54 = _mm_sub_ps(bv_s53, bv_in12);

        bv_m54 = _mm_mul_ps(v_R13_DFT_S5, bv_in1);
        bv_m55 = _mm_mul_ps(v_R13_DFT_S2, bv_in3);
        bv_s55 = _mm_sub_ps(bv_m54, bv_m55);
        bv_m56 = _mm_mul_ps(v_R13_DFT_S1, bv_in5);
        bv_m57 = _mm_mul_ps(v_R13_DFT_S4, bv_in7);
        bv_s56 = _mm_sub_ps(bv_m57, bv_m56);
        bv_m58 = _mm_mul_ps(v_R13_DFT_S6, bv_in9);
        bv_m59 = _mm_mul_ps(v_R13_DFT_S3, bv_in11);
        bv_s57 = _mm_sub_ps(bv_m59, bv_m58);
        bv_s58 = _mm_add_ps(bv_s55, bv_s56);
        bv_s59 = _mm_add_ps(bv_s58, bv_s57);

        // Output point 12: x(11)
        v_out11 = _mm_sub_ps(bv_s54, bv_s59);
        curr_out = out + out_strides[11];
        STHR_128_S(curr_out, v_out_stride, v_out11);

        // Output point 18: x(17)
        v_out17 = _mm_sub_ps(NEGATE_128_S(bv_s54), bv_s59);
        curr_out = out + out_strides[17];
        STHR_128_S(curr_out, v_out_stride, v_out17);

        bv_m60 = _mm_mul_ps(v_R13_DFT_C6, bv_in0);
        bv_m61 = _mm_mul_ps(v_R13_DFT_C5, bv_in2);
        bv_s60 = _mm_sub_ps(bv_m60, bv_m61);
        bv_m62 = _mm_mul_ps(v_R13_DFT_C4, bv_in4);
        bv_m63 = _mm_mul_ps(v_R13_DFT_C3, bv_in6);
        bv_s61 = _mm_sub_ps(bv_m62, bv_m63);
        bv_m64 = _mm_mul_ps(v_R13_DFT_C2, bv_in8);
        bv_m65 = _mm_mul_ps(v_R13_DFT_C1, bv_in10);
        bv_s62 = _mm_sub_ps(bv_m64, bv_m65);
        bv_s63 = _mm_add_ps(bv_s60, bv_s61);
        bv_s64 = _mm_add_ps(bv_s63, bv_s62);
        bv_s65 = _mm_add_ps(bv_s64, bv_in12);

        bv_m66 = _mm_mul_ps(v_R13_DFT_S6, bv_in1);
        bv_m67 = _mm_mul_ps(v_R13_DFT_S5, bv_in3);
        bv_s66 = _mm_sub_ps(bv_m66, bv_m67);
        bv_m68 = _mm_mul_ps(v_R13_DFT_S4, bv_in5);
        bv_m69 = _mm_mul_ps(v_R13_DFT_S3, bv_in7);
        bv_s67 = _mm_sub_ps(bv_m68, bv_m69);
        bv_m70 = _mm_mul_ps(v_R13_DFT_S2, bv_in9);
        bv_m71 = _mm_mul_ps(v_R13_DFT_S1, bv_in11);
        bv_s68 = _mm_sub_ps(bv_m70, bv_m71);
        bv_s69 = _mm_add_ps(bv_s66, bv_s67);
        bv_s70 = _mm_add_ps(bv_s69, bv_s68);

        // Output point 14: x(13)
        v_out13 = _mm_sub_ps(bv_s65, bv_s70);
        curr_out = out + out_strides[13];
        STHR_128_S(curr_out, v_out_stride, v_out13);

        // Output point 16: x(15)
        v_out15 = _mm_sub_ps(NEGATE_128_S(bv_s65), bv_s70);
        curr_out = out + out_strides[15];
        STHR_128_S(curr_out, v_out_stride, v_out15);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }

    if (remaining_sets & 1)
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
        // Input point 7: X(6)
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
        // Output point 5: x4)
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
        // Input point 16: X(25)
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
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft13avx128_fp64_fwd(FFTZ_VOID *in_real,
                                             FFTZ_VOID *in_imag,
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
    FFTZ_DOUBLE *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_128_D;
    FFTZ_INTP remaining_sets = n % NUM_SETS_REAL_128_D;

    __m128d v_CRTM_13_1 = _mm_set1_pd(CRTM_13_1);
    __m128d v_CRTM_13_2 = _mm_set1_pd(CRTM_13_2);
    __m128d v_CRTM_13_3 = _mm_set1_pd(CRTM_13_3);
    __m128d v_CRTM_13_4 = _mm_set1_pd(CRTM_13_4);
    __m128d v_CRTM_13_5 = _mm_set1_pd(CRTM_13_5);
    __m128d v_CRTM_13_6 = _mm_set1_pd(CRTM_13_6);
    __m128d v_CRTM_13_7 = _mm_set1_pd(CRTM_13_7);
    __m128d v_CRTM_13_8 = _mm_set1_pd(CRTM_13_8);
    __m128d v_R13_DGC_1 = _mm_set1_pd(R13_DGC_1);
    __m128d v_R13_DGC_2 = _mm_set1_pd(R13_DGC_2);
    __m128d v_R13_DGC_3 = _mm_set1_pd(R13_DGC_3);
    __m128d v_R13_DGC_4 = _mm_set1_pd(R13_DGC_4);
    __m128d v_R13_DGC_5 = _mm_set1_pd(R13_DGC_5);
    __m128d v_R13_DGC_6 = _mm_set1_pd(R13_DGC_6);
    __m128d v_R13_DGC_7 = _mm_set1_pd(R13_DGC_7);
    __m128d v_R13_DGC_8 = _mm_set1_pd(R13_DGC_8);
    __m128d v_R13_DGC_9 = _mm_set1_pd(R13_DGC_9);
    __m128d v_R13_DGC_10 = _mm_set1_pd(R13_DGC_10);
    __m128d v_R13_DGC_11 = _mm_set1_pd(R13_DGC_11);
    __m128d v_R13_DGC_12 = _mm_set1_pd(R13_DGC_12);
    __m128d v_R13_DFT_C1 = _mm_set1_pd(R13_DFT_C1);
    __m128d v_R13_DFT_C2 = _mm_set1_pd(R13_DFT_C2);
    __m128d v_R13_DFT_C3 = _mm_set1_pd(R13_DFT_C3);
    __m128d v_R13_DFT_C4 = _mm_set1_pd(R13_DFT_C4);
    __m128d v_R13_DFT_C5 = _mm_set1_pd(R13_DFT_C5);
    __m128d v_R13_DFT_C6 = _mm_set1_pd(R13_DFT_C6);
    __m128d v_R13_DFT_S1 = _mm_set1_pd(R13_DFT_S1);
    __m128d v_R13_DFT_S2 = _mm_set1_pd(R13_DFT_S2);
    __m128d v_R13_DFT_S3 = _mm_set1_pd(R13_DFT_S3);
    __m128d v_R13_DFT_S4 = _mm_set1_pd(R13_DFT_S4);
    __m128d v_R13_DFT_S5 = _mm_set1_pd(R13_DFT_S5);
    __m128d v_R13_DFT_S6 = _mm_set1_pd(R13_DFT_S6);

    for (cnt = 0; cnt < N; cnt++)
    {
        // Standard DFT
        __m128d av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
                av_in8, av_in9, av_in10, av_in11, av_in12;
        __m128d av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
                av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16,
                av_s17, av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24,
                av_s25, av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32,
                av_s33, av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40,
                av_s41, av_s42, av_s43, av_s44, av_s45, av_s46, av_s47, av_s48,
                av_s49, av_s50, av_s51, av_s52, av_s53, av_s54, av_s55, av_s56,
                av_s57, av_s58, av_s59, av_s60, av_s61, av_s62, av_s63, av_s64;
        __m128d av_m0, av_m1, av_m2, av_m3, av_m4, av_m5, av_m6, av_m7, av_m8,
                av_m9, av_m10, av_m11, av_m12, av_m13, av_m14, av_m15, av_m16,
                av_m17, av_m18, av_m19, av_m20, av_m21, av_m22, av_m23, av_m24,
                av_m25, av_m26, av_m27, av_m28, av_m29, av_m30, av_m31, av_m32,
                av_m33;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
                v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21,
                v_out22, v_out23, v_out24, v_out25;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_stride, av_in0);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_128_D(curr_in, v_in_stride, av_in1);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_128_D(curr_in, v_in_stride, av_in2);
        // Input point 7: x(6)
        curr_in = in + in_strides[6];
        LDR_128_D(curr_in, v_in_stride, av_in3);
        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDR_128_D(curr_in, v_in_stride, av_in4);
        // Input point 11: x(10)
        curr_in = in + in_strides[10];
        LDR_128_D(curr_in, v_in_stride, av_in5);
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDR_128_D(curr_in, v_in_stride, av_in6);
        // Input point 15: x(14)
        curr_in = in + in_strides[14];
        LDR_128_D(curr_in, v_in_stride, av_in7);
        // Input point 17: x(16)
        curr_in = in + in_strides[16];
        LDR_128_D(curr_in, v_in_stride, av_in8);
        // Input point 19: x(18)
        curr_in = in + in_strides[18];
        LDR_128_D(curr_in, v_in_stride, av_in9);
        // Input point 21: x(20)
        curr_in = in + in_strides[20];
        LDR_128_D(curr_in, v_in_stride, av_in10);
        // Input point 23: x(22)
        curr_in = in + in_strides[22];
        LDR_128_D(curr_in, v_in_stride, av_in11);
        // Input point 25: x(24)
        curr_in = in + in_strides[24];
        LDR_128_D(curr_in, v_in_stride, av_in12);

        av_s0 = _mm_add_pd(av_in2, av_in7);
        av_s1 = _mm_sub_pd(av_in7, av_in2);
        av_s2 = _mm_add_pd(av_in6, av_in11);
        av_s3 = _mm_sub_pd(av_in11, av_in6);
        av_s4 = _mm_add_pd(av_s0, av_s2);
        av_s5 = _mm_sub_pd(av_s0, av_s2);
        av_m0 = _mm_mul_pd(v_CRTM_13_6, av_s5);
        av_s6 = _mm_add_pd(av_s1, av_s3);
        av_s7 = _mm_sub_pd(av_s1, av_s3);
        av_s8 = _mm_add_pd(av_in4, av_in10);
        av_s9 = _mm_sub_pd(av_in10, av_in4);
        av_s10 = _mm_add_pd(av_in3, av_in9);
        av_s11 = _mm_sub_pd(av_in9, av_in3);
        av_s12 = _mm_add_pd(av_s8, av_s10);
        av_s13 = _mm_sub_pd(av_s8, av_s10);
        av_s14 = _mm_sub_pd(av_s9, av_s11);
        av_s15 = _mm_add_pd(av_s9, av_s11);
        av_m1 = _mm_mul_pd(v_CRTM_13_6, av_s15);
        av_m2 = _mm_mul_pd(v_CRTM_13_7, av_s13);

        av_s16 = _mm_add_pd(av_s4, av_s12);
        av_s17 = _mm_sub_pd(av_s4, av_s12);
        av_s18 = _mm_add_pd(av_in8, av_in5);
        av_s19 = _mm_sub_pd(av_in5, av_in8);
        av_m3 = _mm_mul_pd(v_CRTM_13_7, av_s6);
        av_s42 = _mm_add_pd(av_m3, av_s19);
        av_s20 = _mm_sub_pd(av_in1, av_in12);
        av_s21 = _mm_add_pd(av_in1, av_in12);
        av_s36 = _mm_add_pd(av_s20, av_m2);
        av_s22 = _mm_sub_pd(av_s20, av_s13);
        av_s23 = _mm_sub_pd(av_s6, av_s19);
        av_m4 = _mm_mul_pd(v_R13_DGC_6, av_s22);
        av_m5 = _mm_mul_pd(v_R13_DGC_7, av_s23);
        av_s43 = _mm_add_pd(av_m4, av_m5);
        av_m6 = _mm_mul_pd(v_R13_DGC_6, av_s23);
        av_m7 = _mm_mul_pd(v_R13_DGC_7, av_s22);
        av_s45 = _mm_sub_pd(av_m6, av_m7);
        av_s47 = _mm_add_pd(av_s21, av_s18);
        av_s48 = _mm_sub_pd(av_s21, av_s18);
        av_m8 = _mm_mul_pd(v_CRTM_13_7, av_s16);
        av_s32 = _mm_sub_pd(av_s47, av_m8);
        av_s28 = _mm_add_pd(av_s47, av_s16);
        av_m9 = _mm_mul_pd(v_CRTM_13_7, av_s17);
        av_s33 = _mm_add_pd(av_s48, av_m9);
        av_s39 = _mm_sub_pd(av_s48, av_s17);
        av_m10 = _mm_mul_pd(av_s39, v_R13_DGC_12);
        // Output point 1: X(0)
        v_out0 = _mm_add_pd(av_s28, av_in0);
        STR_128_D(curr_out, v_out_stride, v_out0);

        av_m11 = NEGATE_128_D(_mm_mul_pd(av_s28, v_R13_DGC_1));
        av_s63 = _mm_add_pd(av_m11, av_in0);
        av_s24 = _mm_add_pd(av_s63, av_m10);
        av_s25 = _mm_sub_pd(av_s63, av_m10);
        av_s61 = _mm_add_pd(av_s36, av_m0);
        av_s62 = _mm_sub_pd(av_s36, av_m0);
        av_s46 = _mm_add_pd(av_s42, av_m1);
        av_s29 = _mm_sub_pd(av_s42, av_m1);
        av_m12 = _mm_mul_pd(v_R13_DGC_2, av_s61);
        av_m13 = _mm_mul_pd(v_R13_DGC_3, av_s46);
        av_s40 = NEGATE_128_D(_mm_add_pd(av_m12, av_m13));

        av_m14 = _mm_mul_pd(v_R13_DGC_2, av_s46);
        av_m15 = _mm_mul_pd(v_R13_DGC_3, av_s61);
        av_s41 = _mm_sub_pd(av_m14, av_m15);
        av_m16 = _mm_mul_pd(v_R13_DGC_10, av_s29);
        av_m17 = _mm_mul_pd(v_R13_DGC_11, av_s62);
        av_s34 = _mm_sub_pd(av_m17, av_m16);
        av_m18 = _mm_mul_pd(v_R13_DGC_10, av_s62);
        av_m19 = _mm_mul_pd(v_R13_DGC_11, av_s29);
        av_s35 = _mm_add_pd(av_m18, av_m19);
        av_s26 = _mm_add_pd(av_s41, av_s34);
        av_s44 = _mm_sub_pd(av_s41, av_s34);
        av_m20 = _mm_mul_pd(v_CRTM_13_1, av_s44);
        av_s27 = _mm_add_pd(av_s40, av_s35);
        av_s64 = _mm_sub_pd(av_s40, av_s35);
        av_m21 = _mm_mul_pd(v_CRTM_13_1, av_s64);
        av_s30 = _mm_add_pd(av_s7, av_s14);
        av_m22 = _mm_mul_pd(v_R13_DGC_4, av_s33);
        av_m23 = _mm_mul_pd(v_CRTM_13_3, av_s30);
        av_s49 = _mm_sub_pd(av_m22, av_m23);
        av_m24 = _mm_mul_pd(v_CRTM_13_2, av_s30);
        av_m25 = _mm_mul_pd(v_R13_DGC_5, av_s33);
        av_s37 = NEGATE_128_D(_mm_add_pd(av_m24, av_m25));

        av_s31 = _mm_sub_pd(av_s7, av_s14);
        av_m26 = _mm_mul_pd(v_R13_DGC_8, av_s32);
        av_m27 = _mm_mul_pd(v_CRTM_13_5, av_s31);
        av_s50 = _mm_sub_pd(av_m26, av_m27);
        av_m28 = _mm_mul_pd(v_CRTM_13_4, av_s31);
        av_m29 = _mm_mul_pd(v_R13_DGC_9, av_s32);
        av_s38 = NEGATE_128_D(_mm_add_pd(av_m28, av_m29));

        av_s51 = _mm_add_pd(av_s37, av_s38);
        av_s59 = _mm_sub_pd(av_s37, av_s38);
        av_s52 = _mm_add_pd(av_s49, av_s50);
        av_s53 = _mm_sub_pd(av_s49, av_s50);
        av_m30 = _mm_mul_pd(v_CRTM_13_8, av_s52);
        // Output point 4: X(3)
        v_out3 = _mm_add_pd(av_s24, av_m30);
        av_m31 = _mm_mul_pd(v_CRTM_13_8, av_s26);
        // Output point 5: X(4)
        v_out4 = _mm_add_pd(av_s45, av_m31);
        curr_out = out + out_strides[3];
        STRI_2x128_D(curr_out, v_out_stride, v_out3, v_out4);

        av_s54 = _mm_sub_pd(av_s24, av_s52);
        av_s55 = _mm_sub_pd(av_s45, av_s26);
        // Output point 12: X(11)
        v_out11 = _mm_add_pd(av_s54, av_s59);
        // Output point 13: X(12)
        v_out12 = _mm_add_pd(av_s55, av_m21);
        curr_out = out + out_strides[11];
        STRI_2x128_D(curr_out, v_out_stride, v_out11, v_out12);

        // Output point 16: X(15)
        v_out15 = _mm_sub_pd(av_s54, av_s59);
        // Output point 17: X(16)
        v_out16 = _mm_sub_pd(av_m21, av_s55);
        curr_out = out + out_strides[15];
        STRI_2x128_D(curr_out, v_out_stride, v_out15, v_out16);

        av_s56 = _mm_sub_pd(av_s27, av_s43);
        av_m32 = _mm_mul_pd(v_CRTM_13_8, av_s27);
        av_s57 = NEGATE_128_D(_mm_add_pd(av_m32, av_s43));
        av_m33 = _mm_mul_pd(v_CRTM_13_8, av_s53);
        av_s58 = _mm_sub_pd(av_s25, av_m33);
        av_s60 = _mm_add_pd(av_s25, av_s53);
        // Output point 8: X(7)
        v_out7 = _mm_sub_pd(av_s60, av_s51);
        // Output point 9: X(8)
        v_out8 = _mm_add_pd(av_m20, av_s56);
        curr_out = out + out_strides[7];
        STRI_2x128_D(curr_out, v_out_stride, v_out7, v_out8);
       
        // Output point 20: X(19)
        v_out19 = av_s58;
        // Output point 21: X(20)
        v_out20 = av_s57;
        curr_out = out + out_strides[19];
        STRI_2x128_D(curr_out, v_out_stride, v_out19, v_out20);

        // Output point 24: X(23)
        v_out23 = _mm_add_pd(av_s60, av_s51);
        // Output point 25: X(24)
        v_out24 = _mm_sub_pd(av_s56, av_m20);
        curr_out = out + out_strides[23];
        STRI_2x128_D(curr_out, v_out_stride, v_out23, v_out24);

        // Shifted DFT
        __m128d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
                bv_in8, bv_in9, bv_in10, bv_in11, bv_in12;
        __m128d bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
                bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16,
                bv_s17, bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24,
                bv_s25, bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32,
                bv_s33, bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40,
                bv_s41, bv_s42, bv_s43, bv_s44, bv_s45, bv_s46, bv_s47, bv_s48,
                bv_s49, bv_s50, bv_s51, bv_s52, bv_s53, bv_s54, bv_s55, bv_s56,
                bv_s57, bv_s58, bv_s59, bv_s60, bv_s61, bv_s62, bv_s63, bv_s64,
                bv_s65, bv_s66, bv_s67, bv_s68, bv_s69, bv_s70, bv_s71, bv_s72,
                bv_s73, bv_s74, bv_s75, bv_s76;
        __m128d bv_m0, bv_m1, bv_m2, bv_m3, bv_m4, bv_m5, bv_m6, bv_m7, bv_m8,
                bv_m9, bv_m10, bv_m11, bv_m12, bv_m13, bv_m14, bv_m15, bv_m16,
                bv_m17, bv_m18, bv_m19, bv_m20, bv_m21, bv_m22, bv_m23, bv_m24,
                bv_m25, bv_m26, bv_m27, bv_m28, bv_m29, bv_m30, bv_m31, bv_m32,
                bv_m33, bv_m34, bv_m35, bv_m36, bv_m37, bv_m38, bv_m39, bv_m40,
                bv_m41, bv_m42, bv_m43, bv_m44, bv_m45, bv_m46, bv_m47, bv_m48,
                bv_m49, bv_m50, bv_m51, bv_m52, bv_m53, bv_m54, bv_m55, bv_m56,
                bv_m57, bv_m58, bv_m59, bv_m60, bv_m61, bv_m62, bv_m63, bv_m64,
                bv_m65, bv_m66, bv_m67, bv_m68, bv_m69, bv_m70, bv_m71;

        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_128_D(curr_in, v_in_stride, bv_in0);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_128_D(curr_in, v_in_stride, bv_in1);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_128_D(curr_in, v_in_stride, bv_in2);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_128_D(curr_in, v_in_stride, bv_in3);
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDR_128_D(curr_in, v_in_stride, bv_in4);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_128_D(curr_in, v_in_stride, bv_in5);
        // Input point 14: x(13)
        curr_in = in + in_strides[13];
        LDR_128_D(curr_in, v_in_stride, bv_in6);
        // Input point 16: x(15)
        curr_in = in + in_strides[15];
        LDR_128_D(curr_in, v_in_stride, bv_in7);
        // Input point 18: x(17)
        curr_in = in + in_strides[17];
        LDR_128_D(curr_in, v_in_stride, bv_in8);
        // Input point 20: x(19)
        curr_in = in + in_strides[19];
        LDR_128_D(curr_in, v_in_stride, bv_in9);
        // Input point 22: x(21)
        curr_in = in + in_strides[21];
        LDR_128_D(curr_in, v_in_stride, bv_in10);
        // Input point 24: x(23)
        curr_in = in + in_strides[23];
        LDR_128_D(curr_in, v_in_stride, bv_in11);
        // Input point 26: x(25)
        curr_in = in + in_strides[25];
        LDR_128_D(curr_in, v_in_stride, bv_in12);

        bv_s0 = _mm_add_pd(bv_in1, bv_in12);
        bv_s1 = _mm_sub_pd(bv_in1, bv_in12);
        bv_s2 = _mm_add_pd(bv_in2, bv_in11);
        bv_s3 = _mm_sub_pd(bv_in2, bv_in11);
        bv_s4 = _mm_add_pd(bv_in3, bv_in10);
        bv_s5 = _mm_sub_pd(bv_in3, bv_in10);
        bv_s6 = _mm_add_pd(bv_in4, bv_in9);
        bv_s7 = _mm_sub_pd(bv_in4, bv_in9);
        bv_s8 = _mm_add_pd(bv_in5, bv_in8);
        bv_s9 = _mm_sub_pd(bv_in5, bv_in8);
        bv_s10 = _mm_add_pd(bv_in6, bv_in7);
        bv_s11 = _mm_sub_pd(bv_in6, bv_in7);

        bv_m0 = _mm_mul_pd(v_R13_DFT_C1, bv_s1);
        bv_m1 = _mm_mul_pd(v_R13_DFT_C2, bv_s3);
        bv_s12 = _mm_add_pd(bv_m0, bv_m1);
        bv_m2 = _mm_mul_pd(v_R13_DFT_C3, bv_s5);
        bv_m3 = _mm_mul_pd(v_R13_DFT_C4, bv_s7);
        bv_s13 = _mm_add_pd(bv_m2, bv_m3);
        bv_m4 = _mm_mul_pd(v_R13_DFT_C5, bv_s9);
        bv_m5 = _mm_mul_pd(v_R13_DFT_C6, bv_s11);
        bv_s14 = _mm_add_pd(bv_m4, bv_m5);
        bv_s15 = _mm_add_pd(bv_s12, bv_s13);
        bv_s16 = _mm_add_pd(bv_s15, bv_s14);
        // Output point 2: X(1)
        v_out1 = _mm_add_pd(bv_in0, bv_s16);

        bv_m6 = _mm_mul_pd(v_R13_DFT_S1, bv_s0);
        bv_m7 = _mm_mul_pd(v_R13_DFT_S2, bv_s2);
        bv_s17 = _mm_add_pd(bv_m6, bv_m7);
        bv_m8 = _mm_mul_pd(v_R13_DFT_S3, bv_s4);
        bv_m9 = _mm_mul_pd(v_R13_DFT_S4, bv_s6);
        bv_s18 = _mm_add_pd(bv_m8, bv_m9);
        bv_m10 = _mm_mul_pd(v_R13_DFT_S5, bv_s8);
        bv_m11 = _mm_mul_pd(v_R13_DFT_S6, bv_s10);
        bv_s19 = _mm_add_pd(bv_m10, bv_m11);
        bv_s20 = _mm_add_pd(bv_s17, bv_s18);
        bv_s21 = _mm_add_pd(bv_s20, bv_s19);
        // Output point 3: X(2)
        v_out2 = NEGATE_128_D(bv_s21);
        curr_out = out + out_strides[1];
        STRI_2x128_D(curr_out, v_out_stride, v_out1, v_out2);

        bv_m12 = _mm_mul_pd(v_R13_DFT_C3, bv_s1);
        bv_m13 = _mm_mul_pd(v_R13_DFT_C6, bv_s3);
        bv_s22 = _mm_add_pd(bv_m12, bv_m13);
        bv_m14 = _mm_mul_pd(v_R13_DFT_C4, bv_s5);
        bv_m15 = _mm_mul_pd(v_R13_DFT_C1, bv_s7);
        bv_s23 = _mm_add_pd(bv_m14, bv_m15);
        bv_m16 = _mm_mul_pd(v_R13_DFT_C2, bv_s9);
        bv_m17 = _mm_mul_pd(v_R13_DFT_C5, bv_s11);
        bv_s24 = _mm_add_pd(bv_m16, bv_m17);
        bv_s25 = _mm_add_pd(bv_s23, bv_s24);
        bv_s26 = _mm_sub_pd(bv_s22, bv_s25);
        // Output point 6: X(5)
        v_out5 = _mm_add_pd(bv_in0, bv_s26);

        bv_m18 = _mm_mul_pd(v_R13_DFT_S3, bv_s0);
        bv_m19 = _mm_mul_pd(v_R13_DFT_S6, bv_s2);
        bv_s27 = _mm_add_pd(bv_m18, bv_m19);
        bv_m20 = _mm_mul_pd(v_R13_DFT_S4, bv_s4);
        bv_m21 = _mm_mul_pd(v_R13_DFT_S1, bv_s6);
        bv_s28 = _mm_add_pd(bv_m20, bv_m21);
        bv_m22 = _mm_mul_pd(v_R13_DFT_S2, bv_s8);
        bv_m23 = _mm_mul_pd(v_R13_DFT_S5, bv_s10);
        bv_s29 = _mm_add_pd(bv_m22, bv_m23);
        bv_s30 = _mm_add_pd(bv_s27, bv_s28);
        bv_s31 = _mm_sub_pd(bv_s29, bv_s30);
        // Output point 7: X(6)
        v_out6 = bv_s31;
        curr_out = out + out_strides[5];
        STRI_2x128_D(curr_out, v_out_stride, v_out5, v_out6);

        bv_m24 = _mm_mul_pd(v_R13_DFT_C5, bv_s1);
        bv_m25 = _mm_mul_pd(v_R13_DFT_C3, bv_s3);
        bv_s32 = _mm_sub_pd(bv_m24, bv_m25);
        bv_m26 = _mm_mul_pd(v_R13_DFT_C2, bv_s5);
        bv_m27 = _mm_mul_pd(v_R13_DFT_C6, bv_s7);
        bv_s33 = _mm_sub_pd(bv_m27, bv_m26);
        bv_m28 = _mm_mul_pd(v_R13_DFT_C1, bv_s9);
        bv_m29 = _mm_mul_pd(v_R13_DFT_C4, bv_s11);
        bv_s34 = _mm_add_pd(bv_m28, bv_m29);
        bv_s35 = _mm_add_pd(bv_s32, bv_s33);
        bv_s36 = _mm_add_pd(bv_s35, bv_s34);
        // Output point 10: X(9)
        v_out9 = _mm_add_pd(bv_in0, bv_s36);

        bv_m30 = _mm_mul_pd(v_R13_DFT_S5, bv_s0);
        bv_m31 = _mm_mul_pd(v_R13_DFT_S3, bv_s2);
        bv_s37 = _mm_add_pd(bv_m30, bv_m31);
        bv_m32 = _mm_mul_pd(v_R13_DFT_S2, bv_s4);
        bv_m33 = _mm_mul_pd(v_R13_DFT_S6, bv_s6);
        bv_s38 = _mm_add_pd(bv_m32, bv_m33);
        bv_m34 = _mm_mul_pd(v_R13_DFT_S1, bv_s8);
        bv_m35 = _mm_mul_pd(v_R13_DFT_S4, bv_s10);
        bv_s39 = _mm_sub_pd(bv_m34, bv_m35);
        bv_s40 = _mm_add_pd(bv_s38, bv_s39);
        bv_s41 = _mm_sub_pd(bv_s40, bv_s37);
        // Output point 11: X(10)
        v_out10 = bv_s41;
        curr_out = out + out_strides[9];
        STRI_2x128_D(curr_out, v_out_stride, v_out9, v_out10);

        bv_m36 = _mm_mul_pd(v_R13_DFT_C6, bv_s1);
        bv_m37 = _mm_mul_pd(v_R13_DFT_C1, bv_s3);
        bv_s42 = _mm_add_pd(bv_m36, bv_m37);
        bv_m38 = _mm_mul_pd(v_R13_DFT_C5, bv_s5);
        bv_m39 = _mm_mul_pd(v_R13_DFT_C2, bv_s7);
        bv_s43 = _mm_add_pd(bv_m38, bv_m39);
        bv_m40 = _mm_mul_pd(v_R13_DFT_C4, bv_s9);
        bv_m41 = _mm_mul_pd(v_R13_DFT_C3, bv_s11);
        bv_s44 = _mm_add_pd(bv_m40, bv_m41);
        bv_s45 = _mm_add_pd(bv_s42, bv_s44);
        bv_s46 = _mm_sub_pd(bv_s43, bv_s45);
        // Output point 14: X(13)
        v_out13 = _mm_add_pd(bv_in0, bv_s46);

        bv_m42 = _mm_mul_pd(v_R13_DFT_S6, bv_s0);
        bv_m43 = _mm_mul_pd(v_R13_DFT_S1, bv_s2);
        bv_s47 = _mm_sub_pd(bv_m43, bv_m42);
        bv_m44 = _mm_mul_pd(v_R13_DFT_S5, bv_s4);
        bv_m45 = _mm_mul_pd(v_R13_DFT_S2, bv_s6);
        bv_s48 = _mm_sub_pd(bv_m44, bv_m45);
        bv_m46 = _mm_mul_pd(v_R13_DFT_S4, bv_s8);
        bv_m47 = _mm_mul_pd(v_R13_DFT_S3, bv_s10);
        bv_s49 = _mm_sub_pd(bv_m47, bv_m46);
        bv_s50 = _mm_add_pd(bv_s47, bv_s48);
        bv_s51 = _mm_add_pd(bv_s50, bv_s49);
        // Output point 15: X(14)
        v_out14 = bv_s51;
        curr_out = out + out_strides[13];
        STRI_2x128_D(curr_out, v_out_stride, v_out13, v_out14);

        bv_m48 = _mm_mul_pd(v_R13_DFT_C4, bv_s1);
        bv_m49 = _mm_mul_pd(v_R13_DFT_C5, bv_s3);
        bv_s52 = _mm_add_pd(bv_m48, bv_m49);
        bv_m50 = _mm_mul_pd(v_R13_DFT_C1, bv_s5);
        bv_m51 = _mm_mul_pd(v_R13_DFT_C3, bv_s7);
        bv_s53 = _mm_sub_pd(bv_m50, bv_m51);
        bv_m52 = _mm_mul_pd(v_R13_DFT_C6, bv_s9);
        bv_m53 = _mm_mul_pd(v_R13_DFT_C2, bv_s11);
        bv_s54 = _mm_sub_pd(bv_m53, bv_m52);
        bv_s55 = _mm_add_pd(bv_s53, bv_s54);
        bv_s56 = _mm_sub_pd(bv_s55, bv_s52);
        // Output point 18: X(17)
        v_out17 = _mm_add_pd(bv_in0, bv_s56);

        bv_m54 = _mm_mul_pd(v_R13_DFT_S4, bv_s0);
        bv_m55 = _mm_mul_pd(v_R13_DFT_S5, bv_s2);
        bv_s57 = _mm_sub_pd(bv_m55, bv_m54);
        bv_m56 = _mm_mul_pd(v_R13_DFT_S1, bv_s4);
        bv_m57 = _mm_mul_pd(v_R13_DFT_S3, bv_s6);
        bv_s58 = _mm_add_pd(bv_m56, bv_m57);
        bv_m58 = _mm_mul_pd(v_R13_DFT_S6, bv_s8);
        bv_m59 = _mm_mul_pd(v_R13_DFT_S2, bv_s10);
        bv_s59 = _mm_sub_pd(bv_m58, bv_m59);
        bv_s60 = _mm_add_pd(bv_s57, bv_s59);
        bv_s61 = _mm_sub_pd(bv_s60, bv_s58);
        // Output point 19: X(18)
        v_out18 = bv_s61;
        curr_out = out + out_strides[17];
        STRI_2x128_D(curr_out, v_out_stride, v_out17, v_out18);

        bv_m60 = _mm_mul_pd(v_R13_DFT_C2, bv_s1);
        bv_m61 = _mm_mul_pd(v_R13_DFT_C4, bv_s3);
        bv_s62 = _mm_sub_pd(bv_m61, bv_m60);
        bv_m62 = _mm_mul_pd(v_R13_DFT_C6, bv_s5);
        bv_m63 = _mm_mul_pd(v_R13_DFT_C5, bv_s7);
        bv_s63 = _mm_add_pd(bv_m62, bv_m63);
        bv_m64 = _mm_mul_pd(v_R13_DFT_C3, bv_s9);
        bv_m65 = _mm_mul_pd(v_R13_DFT_C1, bv_s11);
        bv_s64 = _mm_sub_pd(bv_m64, bv_m65);
        bv_s65 = _mm_add_pd(bv_s62, bv_s64);
        bv_s66 = _mm_sub_pd(bv_s65, bv_s63);
        // Output point 22: X(21)
        v_out21 = _mm_add_pd(bv_in0, bv_s66);

        bv_m66 = _mm_mul_pd(v_R13_DFT_S2, bv_s0);
        bv_m67 = _mm_mul_pd(v_R13_DFT_S4, bv_s2);
        bv_s67 = _mm_sub_pd(bv_m67, bv_m66);
        bv_m68 = _mm_mul_pd(v_R13_DFT_S6, bv_s4);
        bv_m69 = _mm_mul_pd(v_R13_DFT_S5, bv_s6);
        bv_s68 = _mm_sub_pd(bv_m69, bv_m68);
        bv_m70 = _mm_mul_pd(v_R13_DFT_S3, bv_s8);
        bv_m71 = _mm_mul_pd(v_R13_DFT_S1, bv_s10);
        bv_s69 = _mm_sub_pd(bv_m71, bv_m70);
        bv_s70 = _mm_add_pd(bv_s67, bv_s68);
        bv_s71 = _mm_add_pd(bv_s70, bv_s69);
        // Output point 23: X(22)
        v_out22 = bv_s71;
        curr_out = out + out_strides[21];
        STRI_2x128_D(curr_out, v_out_stride, v_out21, v_out22);

        bv_s72 = _mm_sub_pd(bv_s3, bv_s1);
        bv_s73 = _mm_sub_pd(bv_s7, bv_s5);
        bv_s74 = _mm_sub_pd(bv_s11, bv_s9);
        bv_s75 = _mm_add_pd(bv_s72, bv_s73);
        bv_s76 = _mm_add_pd(bv_s75, bv_s74);
        // Output point 26: X(25)
        v_out25 = _mm_add_pd(bv_in0, bv_s76);
        curr_out = out + out_strides[25];
        STR_128_D(curr_out, v_out_stride, v_out25);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
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
        a_s56 = a_s27 - a_s43;
        // Output point 9: X(8)
        out[out_strides[8]] = a_m20 + a_s56;
        // Output point 25: X(24)
        out[out_strides[24]] = a_s56 - a_m20;

        // Output point 12: X(11)
        out[out_strides[11]] = a_s54 + a_s59;
        // Output point 16: X(15)
        out[out_strides[15]] = a_s54 - a_s59;

        // Output point 13: X(12)
        out[out_strides[12]] = a_s55 + a_m21;
        // Output point 17: X(16)
        out[out_strides[16]] = a_m21 - a_s55;

        a_m32 = CRTM_13_8 * a_s27;
        a_s57 = -(a_m32 + a_s43);
        // Output point 21: X(20)
        out[out_strides[20]] = a_s57;
        a_m33 = CRTM_13_8 * a_s53;
        a_s58 = a_s25 - a_m33;
        // Output point 20: X(19)
        out[out_strides[19]] = a_s58;
        a_s60 = a_s25 + a_s53;
        // Output point 8: X(7)
        out[out_strides[7]] = a_s60 - a_s51;
        // Output point 24: X(23)
        out[out_strides[23]] = a_s60 + a_s51;

        // Shifted DFT
        FFTZ_DOUBLE b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7,
            b_in8, b_in9, b_in10, b_in11, b_in12;
        FFTZ_DOUBLE b_m0, b_m1, b_m2, b_m3, b_m4, b_m5, b_m6, b_m7, b_m8, b_m9,
            b_m10, b_m11, b_m12, b_m13, b_m14, b_m15, b_m16, b_m17, b_m18,
            b_m19, b_m20, b_m21, b_m22, b_m23, b_m24, b_m25, b_m26, b_m27,
            b_m28, b_m29, b_m30, b_m31, b_m32, b_m33, b_m34, b_m35, b_m36,
            b_m37, b_m38, b_m39, b_m40, b_m41, b_m42, b_m43, b_m44, b_m45,
            b_m46, b_m47, b_m48, b_m49, b_m50, b_m51, b_m52, b_m53, b_m54,
            b_m55, b_m56, b_m57, b_m58, b_m59, b_m60, b_m61, b_m62, b_m63,
            b_m64, b_m65, b_m66, b_m67, b_m68, b_m69, b_m70, b_m71;
        FFTZ_DOUBLE b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9,
            b_s10, b_s11, b_s12, b_s13, b_s14, b_s15, b_s16, b_s17, b_s18,
            b_s19, b_s20, b_s21, b_s22, b_s23, b_s24, b_s25, b_s26, b_s27,
            b_s28, b_s29, b_s30, b_s31, b_s32, b_s33, b_s34, b_s35, b_s36,
            b_s37, b_s38, b_s39, b_s40, b_s41, b_s42, b_s43, b_s44, b_s45,
            b_s46, b_s47, b_s48, b_s49, b_s50, b_s51, b_s52, b_s53, b_s54,
            b_s55, b_s56, b_s57, b_s58, b_s59, b_s60, b_s61, b_s62, b_s63,
            b_s64, b_s65, b_s66, b_s67, b_s68, b_s69, b_s70, b_s71, b_s72,
            b_s73, b_s74, b_s75, b_s76;

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
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft13avx128_fp64_bwd(FFTZ_VOID *in_real,
                                             FFTZ_VOID *in_imag,
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
    FFTZ_DOUBLE *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_128_D;
    FFTZ_INTP remaining_sets = n % NUM_SETS_REAL_128_D;

    __m128d v_CRTM_13_1 = _mm_set1_pd(CRTM_13_1);
    __m128d v_CRTM_13_2 = _mm_set1_pd(CRTM_13_2);
    __m128d v_CRTM_13_3 = _mm_set1_pd(CRTM_13_3);
    __m128d v_CRTM_13_4 = _mm_set1_pd(CRTM_13_4);
    __m128d v_CRTM_13_5 = _mm_set1_pd(CRTM_13_5);
    __m128d v_CRTM_13_6 = _mm_set1_pd(CRTM_13_6);
    __m128d v_CRTM_13_7 = _mm_set1_pd(CRTM_13_7);
    __m128d v_R13_DGC_1 = _mm_set1_pd(R13_DGC_1);
    __m128d v_R13_DGC_2 = _mm_set1_pd(R13_DGC_2);
    __m128d v_R13_DGC_3 = _mm_set1_pd(R13_DGC_3);
    __m128d v_R13_DGC_4 = _mm_set1_pd(R13_DGC_4);
    __m128d v_R13_DGC_5 = _mm_set1_pd(R13_DGC_5);
    __m128d v_R13_DGC_6 = _mm_set1_pd(R13_DGC_6);
    __m128d v_R13_DGC_7 = _mm_set1_pd(R13_DGC_7);
    __m128d v_R13_DGC_8 = _mm_set1_pd(R13_DGC_8);
    __m128d v_R13_DGC_9 = _mm_set1_pd(R13_DGC_9);
    __m128d v_R13_DGC_10 = _mm_set1_pd(R13_DGC_10);
    __m128d v_R13_DGC_11 = _mm_set1_pd(R13_DGC_11);
    __m128d v_R13_DGC_12 = _mm_set1_pd(R13_DGC_12);
    __m128d v_R13_DFT_C1 = _mm_set1_pd(R13_DFT_C1);
    __m128d v_R13_DFT_C2 = _mm_set1_pd(R13_DFT_C2);
    __m128d v_R13_DFT_C3 = _mm_set1_pd(R13_DFT_C3);
    __m128d v_R13_DFT_C4 = _mm_set1_pd(R13_DFT_C4);
    __m128d v_R13_DFT_C5 = _mm_set1_pd(R13_DFT_C5);
    __m128d v_R13_DFT_C6 = _mm_set1_pd(R13_DFT_C6);
    __m128d v_R13_DFT_S1 = _mm_set1_pd(R13_DFT_S1);
    __m128d v_R13_DFT_S2 = _mm_set1_pd(R13_DFT_S2);
    __m128d v_R13_DFT_S3 = _mm_set1_pd(R13_DFT_S3);
    __m128d v_R13_DFT_S4 = _mm_set1_pd(R13_DFT_S4);
    __m128d v_R13_DFT_S5 = _mm_set1_pd(R13_DFT_S5);
    __m128d v_R13_DFT_S6 = _mm_set1_pd(R13_DFT_S6);

    for (cnt = 0; cnt < N; cnt++)
    {
        // Standard DFT
        __m128d av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
                av_in8, av_in9, av_in10, av_in11, av_in12;
        __m128d av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
                av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16,
                av_s17, av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24,
                av_s25, av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32,
                av_s33, av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40,
                av_s41, av_s42, av_s43, av_s44, av_s45, av_s46, av_s47, av_s48,
                av_s49, av_s50, av_s51, av_s52, av_s53, av_s54, av_s55, av_s56,
                av_s57, av_s58, av_s59, av_s60, av_s61, av_s62;
        __m128d av_m0, av_m1, av_m2, av_m3, av_m4, av_m5, av_m6, av_m7, av_m8,
                av_m9, av_m10, av_m11, av_m12, av_m13, av_m14, av_m15, av_m16,
                av_m17, av_m18, av_m19, av_m20, av_m21, av_m22, av_m23, av_m24,
                av_m25, av_m26, av_m27, av_m28, av_m29, av_m30, av_m31, av_m32,
                av_m33, av_m34;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
                v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21,
                v_out22, v_out23, v_out24, v_out25;

        curr_in = in;
        curr_out = out;

        // Input point 1: X(0)
        LDR_128_D(curr_in, v_in_stride, av_in0);
        // Input point 4: X(3) & Input point 5: X(4)
        curr_in = in + in_strides[3];
        LDRI_2x128_D(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: X(7) & Input point 9: X(8)
        curr_in = in + in_strides[7];
        LDRI_2x128_D(curr_in, v_in_stride, av_in3, av_in4);
        // Input point 12: X(11) & Input point 13: X(12)
        curr_in = in + in_strides[11];
        LDRI_2x128_D(curr_in, v_in_stride, av_in5, av_in6);
        // Input point 16: X(15) & Input point 17: X(16)
        curr_in = in + in_strides[15];
        LDRI_2x128_D(curr_in, v_in_stride, av_in7, av_in8);
        // Input point 20: X(19) & Input point 21: X(20)
        curr_in = in + in_strides[19];
        LDRI_2x128_D(curr_in, v_in_stride, av_in9, av_in10);
        // Input point 24: X(23) & Input point 25: X(24)
        curr_in = in + in_strides[23];
        LDRI_2x128_D(curr_in, v_in_stride, av_in11, av_in12);

        av_s0 = _mm_sub_pd(av_in6, av_in8);
        av_m0 = _mm_mul_pd(v_CRTM_13_7, av_in2);
        av_s1 = _mm_sub_pd(av_m0, av_s0);
        av_s59 = _mm_add_pd(av_in6, av_in8);
        av_m1 = _mm_mul_pd(v_CRTM_13_1, av_s59);
        av_s2 = _mm_add_pd(av_in12, av_in4);
        av_s60 = _mm_sub_pd(av_in12, av_in4);
        av_m2 = _mm_mul_pd(v_CRTM_13_1, av_s60);
        av_m3 = _mm_mul_pd(v_CRTM_13_7, av_in10);
        av_s3 = _mm_sub_pd(av_s2, av_m3);
        av_s4 = _mm_add_pd(av_s1, av_m2);
        av_s5 = _mm_sub_pd(av_s3, av_m1);
        av_m4 = _mm_mul_pd(v_R13_DGC_11, av_s4);
        av_m5 = _mm_mul_pd(v_R13_DGC_10, av_s5);
        av_s6 = _mm_add_pd(av_m4, av_m5);
        av_m6 = _mm_mul_pd(v_R13_DGC_10, av_s4);
        av_m7 = _mm_mul_pd(v_R13_DGC_11, av_s5);
        av_s7 = _mm_sub_pd(av_m6, av_m7);
        av_s8 = _mm_add_pd(av_in2, av_s0);
        av_s9 = _mm_add_pd(av_s2, av_in10);
        av_m8 = _mm_mul_pd(v_CRTM_13_4, av_s8);
        av_m9 = _mm_mul_pd(v_CRTM_13_5, av_s9);

        av_s10 = _mm_sub_pd(av_m8, av_m9);
        av_m10 = _mm_mul_pd(v_CRTM_13_5, av_s8);
        av_m11 = _mm_mul_pd(v_CRTM_13_4, av_s9);
        av_s11 = _mm_add_pd(av_m10, av_m11);
        av_s12 = _mm_sub_pd(av_s1, av_m2);
        av_s13 = _mm_add_pd(av_m1, av_s3);
        av_m12 = _mm_mul_pd(v_R13_DGC_3, av_s12);
        av_m13 = _mm_mul_pd(v_R13_DGC_2, av_s13);
        av_s14 = _mm_add_pd(av_m12, av_m13);
        av_m14 = _mm_mul_pd(v_R13_DGC_3, av_s13);
        av_m15 = _mm_mul_pd(v_R13_DGC_2, av_s12);
        av_s15 = _mm_sub_pd(av_m14, av_m15);
        av_s16 = _mm_add_pd(av_in3, av_in11);
        av_s17 = _mm_add_pd(av_in9, av_s16);
        av_m16 = _mm_mul_pd(v_CRTM_13_6, av_s16);
        av_s18 = _mm_sub_pd(av_in9, av_m16);
        av_s19 = _mm_sub_pd(av_in3, av_in11);
        av_s20 = _mm_add_pd(av_in5, av_in7);
        av_s21 = _mm_add_pd(av_in1, av_s20);
        av_m17 = _mm_mul_pd(v_CRTM_13_6, av_s20);
        av_s22 = _mm_sub_pd(av_in1, av_m17);
        av_s23 = _mm_sub_pd(av_in5, av_in7);
        av_s24 = _mm_sub_pd(av_s21, av_s17);
        av_m18 = _mm_mul_pd(v_R13_DGC_6, av_s24);
        av_s25 = _mm_add_pd(av_s21, av_s17);
        av_m33 = _mm_mul_pd(v_CRTM_13_7, av_s25);
        // Output point 1: x(0)
        v_out0 = _mm_add_pd(av_m33, av_in0);
        STR_128_D(curr_out, v_out_stride, v_out0);

        av_m34 = _mm_mul_pd(v_R13_DGC_1, av_s25);
        av_s26 = _mm_sub_pd(av_in0, av_m34);
        av_s27 = _mm_add_pd(av_s23, av_s19);
        av_s28 = _mm_add_pd(av_s22, av_s18);
        av_m19 = _mm_mul_pd(v_R13_DGC_9, av_s27);
        av_m20 = _mm_mul_pd(v_R13_DGC_7, av_s28);
        av_s29 = _mm_add_pd(av_m19, av_m20);
        av_s30 = _mm_sub_pd(av_s22, av_s18);
        av_s31 = _mm_sub_pd(av_s23, av_s19);
        av_m21 = _mm_mul_pd(v_R13_DGC_12, av_s30);
        av_m22 = _mm_mul_pd(v_R13_DGC_5, av_s31);
        av_s32 = _mm_sub_pd(av_m21, av_m22);
        av_s61 = _mm_add_pd(av_s6, av_s14);
        av_m23 = _mm_mul_pd(v_CRTM_13_1, av_s61);
        av_s62 = _mm_sub_pd(av_s7, av_s15);
        av_m24 = _mm_mul_pd(v_CRTM_13_1, av_s62);
        av_s33 = _mm_add_pd(av_s7, av_s15);
        av_s34 = _mm_sub_pd(av_s10, av_s33);
        av_m25 = _mm_mul_pd(v_CRTM_13_7, av_s33);
        av_s35 = _mm_add_pd(av_m25, av_s10);
        av_s36 = _mm_sub_pd(av_s6, av_s14);
        av_m26 = _mm_mul_pd(v_CRTM_13_7, av_s36);
        av_s37 = _mm_sub_pd(av_m26, av_s11);
        av_s38 = _mm_add_pd(av_s36, av_s11);
        av_m27 = _mm_mul_pd(v_R13_DGC_4, av_s31);
        av_m28 = _mm_mul_pd(v_CRTM_13_3, av_s30);
        av_s39 = _mm_add_pd(av_m27, av_m28);
        av_m29 = _mm_mul_pd(v_R13_DGC_8, av_s27);
        av_m30 = _mm_mul_pd(v_CRTM_13_2, av_s28);
        av_s40 = _mm_sub_pd(av_m29, av_m30);

        av_s41 = _mm_sub_pd(av_s39, av_s40);
        av_s42 = _mm_add_pd(av_s39, av_s40);
        av_s43 = _mm_sub_pd(av_s26, av_s29);
        av_s44 = _mm_sub_pd(av_m18, av_s32);
        av_s45 = _mm_sub_pd(av_s43, av_s44);
        av_s46 = _mm_add_pd(av_s44, av_s43);
        av_m31 = _mm_mul_pd(v_CRTM_13_7, av_s29);
        av_s47 = _mm_add_pd(av_m31, av_s26);
        av_m32 = _mm_mul_pd(v_CRTM_13_7, av_s32);
        av_s48 = _mm_add_pd(av_m32, av_m18);
        av_s49 = _mm_sub_pd(av_s47, av_s48);
        // Output point 17: x(16)
        v_out16 = _mm_add_pd(av_s49, av_s35);
        curr_out = out + out_strides[16];
        STR_128_D(curr_out, v_out_stride, v_out16);

        // Output point 11: x(10)
        v_out10 = _mm_sub_pd(av_s49, av_s35);
        curr_out = out + out_strides[10];
        STR_128_D(curr_out, v_out_stride, v_out10);

        av_s50 = _mm_add_pd(av_s48, av_s47);
        // Output point 25: x(24)
        v_out24 = _mm_sub_pd(av_s50, av_s37);
        curr_out = out + out_strides[24];
        STR_128_D(curr_out, v_out_stride, v_out24);

        // Output point 3: x(2)
        v_out2 = _mm_add_pd(av_s50, av_s37);
        curr_out = out + out_strides[2];
        STR_128_D(curr_out, v_out_stride, v_out2);

        av_s51 = _mm_sub_pd(av_s45, av_m23);
        av_s52 = _mm_sub_pd(av_s41, av_s34);
        // Output point 5: x(4)
        v_out4 = _mm_add_pd(av_s51, av_s52);
        curr_out = out + out_strides[4];
        STR_128_D(curr_out, v_out_stride, v_out4);

        // Output point 15: x(14)
        v_out14 = _mm_sub_pd(av_s51, av_s52);
        curr_out = out + out_strides[14];
        STR_128_D(curr_out, v_out_stride, v_out14);

        av_s53 = _mm_sub_pd(av_s46, av_s38);
        av_s54 = _mm_add_pd(av_s42, av_m24);
        // Output point 7: x(6)
        v_out6 = _mm_sub_pd(av_s53, av_s54);
        curr_out = out + out_strides[6];
        STR_128_D(curr_out, v_out_stride, v_out6);

        // Output point 19: x(18)
        v_out18 = _mm_add_pd(av_s54, av_s53);
        curr_out = out + out_strides[18];
        STR_128_D(curr_out, v_out_stride, v_out18);

        av_s55 = _mm_sub_pd(av_s42, av_m24);
        av_s56 = _mm_add_pd(av_s46, av_s38);
        // Output point 9: x(8)
        v_out8 = _mm_add_pd(av_s55, av_s56);
        curr_out = out + out_strides[8];
        STR_128_D(curr_out, v_out_stride, v_out8);

        // Output point 21: x(20)
        v_out20 = _mm_sub_pd(av_s56, av_s55);
        curr_out = out + out_strides[20];
        STR_128_D(curr_out, v_out_stride, v_out20);

        av_s57 = _mm_add_pd(av_s45, av_m23);
        av_s58 = _mm_add_pd(av_s41, av_s34);
        // Output point 13: x(12)
        v_out12 = _mm_sub_pd(av_s57, av_s58);
        curr_out = out + out_strides[12];
        STR_128_D(curr_out, v_out_stride, v_out12);

        // Output point 23: x(22)
        v_out22 = _mm_add_pd(av_s57, av_s58);
        curr_out = out + out_strides[22];
        STR_128_D(curr_out, v_out_stride, v_out22);

        // Shifted DFT
        __m128d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
                 bv_in8, bv_in9, bv_in10, bv_in11, bv_in12;
        __m128d bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
                bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16,
                bv_s17, bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24,
                bv_s25, bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32,
                bv_s33, bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40,
                bv_s41, bv_s42, bv_s43, bv_s44, bv_s45, bv_s46, bv_s47, bv_s48,
                bv_s49, bv_s50, bv_s51, bv_s52, bv_s53, bv_s54, bv_s55, bv_s56,
                bv_s57, bv_s58, bv_s59, bv_s60, bv_s61, bv_s62, bv_s63, bv_s64,
                bv_s65, bv_s66, bv_s67, bv_s68, bv_s69, bv_s70;
        __m128d bv_m0, bv_m1, bv_m2, bv_m3, bv_m4, bv_m5, bv_m6, bv_m7, bv_m8,
                bv_m9, bv_m10, bv_m11, bv_m12, bv_m13, bv_m14, bv_m15, bv_m16,
                bv_m17, bv_m18, bv_m19, bv_m20, bv_m21, bv_m22, bv_m23, bv_m24,
                bv_m25, bv_m26, bv_m27, bv_m28, bv_m29, bv_m30, bv_m31, bv_m32,
                bv_m33, bv_m34, bv_m35, bv_m36, bv_m37, bv_m38, bv_m39, bv_m40,
                bv_m41, bv_m42, bv_m43, bv_m44, bv_m45, bv_m46, bv_m47, bv_m48,
                bv_m49, bv_m50, bv_m51, bv_m52, bv_m53, bv_m54, bv_m55, bv_m56,
                bv_m57, bv_m58, bv_m59, bv_m60, bv_m61, bv_m62, bv_m63, bv_m64,
                bv_m65, bv_m66, bv_m67, bv_m68, bv_m69, bv_m70, bv_m71;

        // Input point 2: X(1) & Input point 3: X(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: X(5) & Input point 7: X(6)
        curr_in = in + in_strides[5];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: X(9) & Input point 11: X(10)
        curr_in = in + in_strides[9];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in4, bv_in5);
        // Input point 14: X(13) & Input point 15: X(14)
        curr_in = in + in_strides[13];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in6, bv_in7);
        // Input point 18: X(17) & Input point 19: X(18)
        curr_in = in + in_strides[17];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in8, bv_in9);
        // Input point 22: X(21) & Input point 23: X(22)
        curr_in = in + in_strides[21];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in10, bv_in11);
        // Input point 26: X(25)
        curr_in = in + in_strides[25];
        LDR_128_D(curr_in, v_in_stride, bv_in12);

        bv_s0 = _mm_add_pd(bv_in0, bv_in2);
        bv_s1 = _mm_add_pd(bv_in4, bv_in6);
        bv_s2 = _mm_add_pd(bv_in8, bv_in10);
        bv_s3 = _mm_add_pd(bv_s0, bv_s1);
        bv_s4 = _mm_add_pd(bv_s3, bv_s2);
        // Output point 2: x(1)
        v_out1 = _mm_add_pd(bv_s4, bv_s4);
        v_out1 = _mm_add_pd(v_out1, bv_in12);
        curr_out = out + out_strides[1];
        STR_128_D(curr_out, v_out_stride, v_out1);

        bv_m0 = _mm_mul_pd(v_R13_DFT_C1, bv_in0);
        bv_m1 = _mm_mul_pd(v_R13_DFT_C3, bv_in2);
        bv_s5 = _mm_add_pd(bv_m0, bv_m1);
        bv_m2 = _mm_mul_pd(v_R13_DFT_C5, bv_in4);
        bv_m3 = _mm_mul_pd(v_R13_DFT_C6, bv_in6);
        bv_s6 = _mm_sub_pd(bv_m2, bv_m3);
        bv_m4 = _mm_mul_pd(v_R13_DFT_C4, bv_in8);
        bv_m5 = _mm_mul_pd(v_R13_DFT_C2, bv_in10);
        bv_s7 = _mm_add_pd(bv_m4, bv_m5);
        bv_s8 = _mm_add_pd(bv_s5, bv_s6);
        bv_s9 = _mm_sub_pd(bv_s8, bv_s7);
        bv_s10 = _mm_sub_pd(bv_s9, bv_in12);
        bv_m6 = _mm_mul_pd(v_R13_DFT_S1, bv_in1);
        bv_m7 = _mm_mul_pd(v_R13_DFT_S3, bv_in3);
        bv_s11 = _mm_add_pd(bv_m6, bv_m7);
        bv_m8 = _mm_mul_pd(v_R13_DFT_S5, bv_in5);
        bv_m9 = _mm_mul_pd(v_R13_DFT_S6, bv_in7);
        bv_s12 = _mm_add_pd(bv_m8, bv_m9);
        bv_m10 = _mm_mul_pd(v_R13_DFT_S4, bv_in9);
        bv_m11 = _mm_mul_pd(v_R13_DFT_S2, bv_in11);
        bv_s13 = _mm_add_pd(bv_m10, bv_m11);
        bv_s14 = _mm_add_pd(bv_s11, bv_s12);
        bv_s15 = _mm_add_pd(bv_s14, bv_s13);
        // Output point 4: x(3)
        v_out3 = _mm_sub_pd(bv_s10, bv_s15);
        curr_out = out + out_strides[3];
        STR_128_D(curr_out, v_out_stride, v_out3);

        // Output point 26: x(25)
        v_out25 = _mm_sub_pd(NEGATE_128_D(bv_s10), bv_s15);
        curr_out = out + out_strides[25];
        STR_128_D(curr_out, v_out_stride, v_out25);

        bv_m12 = _mm_mul_pd(v_R13_DFT_C2, bv_in0);
        bv_m13 = _mm_mul_pd(v_R13_DFT_C6, bv_in2);
        bv_s16 = _mm_add_pd(bv_m12, bv_m13);
        bv_m14 = _mm_mul_pd(v_R13_DFT_C3, bv_in4);
        bv_m15 = _mm_mul_pd(v_R13_DFT_C1, bv_in6);
        bv_s17 = _mm_add_pd(bv_m14, bv_m15);
        bv_m16 = _mm_mul_pd(v_R13_DFT_C5, bv_in8);
        bv_m17 = _mm_mul_pd(v_R13_DFT_C4, bv_in10);
        bv_s18 = _mm_sub_pd(bv_m17, bv_m16);
        bv_s19 = _mm_sub_pd(bv_s16, bv_s17);
        bv_s20 = _mm_add_pd(bv_s19, bv_s18);
        bv_s21 = _mm_add_pd(bv_s20, bv_in12);
        bv_m18 = _mm_mul_pd(v_R13_DFT_S2, bv_in1);
        bv_m19 = _mm_mul_pd(v_R13_DFT_S6, bv_in3);
        bv_s22 = _mm_add_pd(bv_m18, bv_m19);
        bv_m20 = _mm_mul_pd(v_R13_DFT_S3, bv_in5);
        bv_m21 = _mm_mul_pd(v_R13_DFT_S1, bv_in7);
        bv_s23 = _mm_sub_pd(bv_m20, bv_m21);
        bv_m22 = _mm_mul_pd(v_R13_DFT_S5, bv_in9);
        bv_m23 = _mm_mul_pd(v_R13_DFT_S4, bv_in11);
        bv_s24 = _mm_add_pd(bv_m22, bv_m23);
        bv_s25 = _mm_add_pd(bv_s22, bv_s23);
        bv_s26 = _mm_sub_pd(bv_s25, bv_s24);
        // Output point 6: x(5)
        v_out5 = _mm_sub_pd(bv_s21, bv_s26);
        curr_out = out + out_strides[5];
        STR_128_D(curr_out, v_out_stride, v_out5);

        // Output point 24: x(23)
        v_out23 = _mm_sub_pd(NEGATE_128_D(bv_s21), bv_s26);
        curr_out = out + out_strides[23];
        STR_128_D(curr_out, v_out_stride, v_out23);

        bv_m24 = _mm_mul_pd(v_R13_DFT_C3, bv_in0);
        bv_m25 = _mm_mul_pd(v_R13_DFT_C4, bv_in2);
        bv_s27 = _mm_sub_pd(bv_m24, bv_m25);
        bv_m26 = _mm_mul_pd(v_R13_DFT_C2, bv_in4);
        bv_m27 = _mm_mul_pd(v_R13_DFT_C5, bv_in6);
        bv_s28 = _mm_sub_pd(bv_m27, bv_m26);
        bv_m28 = _mm_mul_pd(v_R13_DFT_C1, bv_in8);
        bv_m29 = _mm_mul_pd(v_R13_DFT_C6, bv_in10);
        bv_s29 = _mm_sub_pd(bv_m28, bv_m29);
        bv_s30 = _mm_add_pd(bv_s27, bv_s28);
        bv_s31 = _mm_add_pd(bv_s30, bv_s29);
        bv_s32 = _mm_sub_pd(bv_s31, bv_in12);
        bv_m30 = _mm_mul_pd(v_R13_DFT_S3, bv_in1);
        bv_m31 = _mm_mul_pd(v_R13_DFT_S4, bv_in3);
        bv_s33 = _mm_add_pd(bv_m30, bv_m31);
        bv_m32 = _mm_mul_pd(v_R13_DFT_S2, bv_in5);
        bv_m33 = _mm_mul_pd(v_R13_DFT_S5, bv_in7);
        bv_s34 = _mm_add_pd(bv_m32, bv_m33);
        bv_m34 = _mm_mul_pd(v_R13_DFT_S1, bv_in9);
        bv_m35 = _mm_mul_pd(v_R13_DFT_S6, bv_in11);
        bv_s35 = _mm_add_pd(bv_m34, bv_m35);
        bv_s36 = _mm_sub_pd(bv_s33, bv_s34);
        bv_s37 = _mm_add_pd(bv_s36, bv_s35);
        // Output point 8: x(7)
        v_out7 = _mm_sub_pd(bv_s32, bv_s37);
        curr_out = out + out_strides[7];
        STR_128_D(curr_out, v_out_stride, v_out7);

        // Output point 22: x(21)
        v_out21 = _mm_sub_pd(NEGATE_128_D(bv_s32), bv_s37);
        curr_out = out + out_strides[21];
        STR_128_D(curr_out, v_out_stride, v_out21);

        bv_m36 = _mm_mul_pd(v_R13_DFT_C4, bv_in0);
        bv_m37 = _mm_mul_pd(v_R13_DFT_C1, bv_in2);
        bv_s38 = _mm_sub_pd(bv_m36, bv_m37);
        bv_m38 = _mm_mul_pd(v_R13_DFT_C6, bv_in4);
        bv_m39 = _mm_mul_pd(v_R13_DFT_C2, bv_in6);
        bv_s39 = _mm_add_pd(bv_m38, bv_m39);
        bv_m40 = _mm_mul_pd(v_R13_DFT_C3, bv_in8);
        bv_m41 = _mm_mul_pd(v_R13_DFT_C5, bv_in10);
        bv_s40 = _mm_add_pd(bv_m40, bv_m41);
        bv_s41 = _mm_add_pd(bv_s38, bv_s39);
        bv_s42 = _mm_sub_pd(bv_s41, bv_s40);
        bv_s43 = _mm_add_pd(bv_s42, bv_in12);
        bv_m42 = _mm_mul_pd(v_R13_DFT_S4, bv_in1);
        bv_m43 = _mm_mul_pd(v_R13_DFT_S1, bv_in3);
        bv_s44 = _mm_add_pd(bv_m42, bv_m43);
        bv_m44 = _mm_mul_pd(v_R13_DFT_S6, bv_in5);
        bv_m45 = _mm_mul_pd(v_R13_DFT_S2, bv_in7);
        bv_s45 = _mm_sub_pd(bv_m45, bv_m44);
        bv_m46 = _mm_mul_pd(v_R13_DFT_S3, bv_in9);
        bv_m47 = _mm_mul_pd(v_R13_DFT_S5, bv_in11);
        bv_s46 = _mm_sub_pd(bv_m46, bv_m47);
        bv_s47 = _mm_add_pd(bv_s44, bv_s45);
        bv_s48 = _mm_add_pd(bv_s47, bv_s46);
        // Output point 10: x(9)
        v_out9 = _mm_sub_pd(bv_s43, bv_s48);
        curr_out = out + out_strides[9];
        STR_128_D(curr_out, v_out_stride, v_out9);

        // Output point 20: x(19)
        v_out19 = _mm_sub_pd(NEGATE_128_D(bv_s43), bv_s48);
        curr_out = out + out_strides[19];
        STR_128_D(curr_out, v_out_stride, v_out19);

        bv_m48 = _mm_mul_pd(v_R13_DFT_C5, bv_in0);
        bv_m49 = _mm_mul_pd(v_R13_DFT_C2, bv_in2);
        bv_s49 = _mm_sub_pd(bv_m48, bv_m49);
        bv_m50 = _mm_mul_pd(v_R13_DFT_C1, bv_in4);
        bv_m51 = _mm_mul_pd(v_R13_DFT_C4, bv_in6);
        bv_s50 = _mm_sub_pd(bv_m50, bv_m51);
        bv_m52 = _mm_mul_pd(v_R13_DFT_C6, bv_in8);
        bv_m53 = _mm_mul_pd(v_R13_DFT_C3, bv_in10);
        bv_s51 = _mm_sub_pd(bv_m53, bv_m52);
        bv_s52 = _mm_add_pd(bv_s49, bv_s50);
        bv_s53 = _mm_add_pd(bv_s52, bv_s51);
        bv_s54 = _mm_sub_pd(bv_s53, bv_in12);
        bv_m54 = _mm_mul_pd(v_R13_DFT_S5, bv_in1);
        bv_m55 = _mm_mul_pd(v_R13_DFT_S2, bv_in3);
        bv_s55 = _mm_sub_pd(bv_m54, bv_m55);
        bv_m56 = _mm_mul_pd(v_R13_DFT_S1, bv_in5);
        bv_m57 = _mm_mul_pd(v_R13_DFT_S4, bv_in7);
        bv_s56 = _mm_sub_pd(bv_m57, bv_m56);
        bv_m58 = _mm_mul_pd(v_R13_DFT_S6, bv_in9);
        bv_m59 = _mm_mul_pd(v_R13_DFT_S3, bv_in11);
        bv_s57 = _mm_sub_pd(bv_m59, bv_m58);
        bv_s58 = _mm_add_pd(bv_s55, bv_s56);
        bv_s59 = _mm_add_pd(bv_s58, bv_s57);
        // Output point 12: x(11)
        v_out11 = _mm_sub_pd(bv_s54, bv_s59);
        curr_out = out + out_strides[11];
        STR_128_D(curr_out, v_out_stride, v_out11);

        // Output point 18: x(17)
        v_out17 = _mm_sub_pd(NEGATE_128_D(bv_s54), bv_s59);
        curr_out = out + out_strides[17];
        STR_128_D(curr_out, v_out_stride, v_out17);

        bv_m60 = _mm_mul_pd(v_R13_DFT_C6, bv_in0);
        bv_m61 = _mm_mul_pd(v_R13_DFT_C5, bv_in2);
        bv_s60 = _mm_sub_pd(bv_m60, bv_m61);
        bv_m62 = _mm_mul_pd(v_R13_DFT_C4, bv_in4);
        bv_m63 = _mm_mul_pd(v_R13_DFT_C3, bv_in6);
        bv_s61 = _mm_sub_pd(bv_m62, bv_m63);
        bv_m64 = _mm_mul_pd(v_R13_DFT_C2, bv_in8);
        bv_m65 = _mm_mul_pd(v_R13_DFT_C1, bv_in10);
        bv_s62 = _mm_sub_pd(bv_m64, bv_m65);
        bv_s63 = _mm_add_pd(bv_s60, bv_s61);
        bv_s64 = _mm_add_pd(bv_s63, bv_s62);
        bv_s65 = _mm_add_pd(bv_s64, bv_in12);
        bv_m66 = _mm_mul_pd(v_R13_DFT_S6, bv_in1);
        bv_m67 = _mm_mul_pd(v_R13_DFT_S5, bv_in3);
        bv_s66 = _mm_sub_pd(bv_m66, bv_m67);
        bv_m68 = _mm_mul_pd(v_R13_DFT_S4, bv_in5);
        bv_m69 = _mm_mul_pd(v_R13_DFT_S3, bv_in7);
        bv_s67 = _mm_sub_pd(bv_m68, bv_m69);
        bv_m70 = _mm_mul_pd(v_R13_DFT_S2, bv_in9);
        bv_m71 = _mm_mul_pd(v_R13_DFT_S1, bv_in11);
        bv_s68 = _mm_sub_pd(bv_m70, bv_m71);
        bv_s69 = _mm_add_pd(bv_s66, bv_s67);
        bv_s70 = _mm_add_pd(bv_s69, bv_s68);
        // Output point 14: x(13)
        v_out13 = _mm_sub_pd(bv_s65, bv_s70);
        curr_out = out + out_strides[13];
        STR_128_D(curr_out, v_out_stride, v_out13);

        // Output point 16: x(15)
        v_out15 = _mm_sub_pd(NEGATE_128_D(bv_s65), bv_s70);
        curr_out = out + out_strides[15];
        STR_128_D(curr_out, v_out_stride, v_out15);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
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
        // Input point 23: X(23)
        a_in11 = in[in_strides[23]];
        // Input point 24: X(23)
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
                    b_m10, b_m11, b_m12, b_m13, b_m14, b_m15, b_m16, b_m17, b_m18,
                    b_m19, b_m20, b_m21, b_m22, b_m23, b_m24, b_m25, b_m26, b_m27,
                    b_m28, b_m29, b_m30, b_m31, b_m32, b_m33, b_m34, b_m35, b_m36,
                    b_m37, b_m38, b_m39, b_m40, b_m41, b_m42, b_m43, b_m44, b_m45,
                    b_m46, b_m47, b_m48, b_m49, b_m50, b_m51, b_m52, b_m53, b_m54,
                    b_m55, b_m56, b_m57, b_m58, b_m59, b_m60, b_m61, b_m62, b_m63,
                    b_m64, b_m65, b_m66, b_m67, b_m68, b_m69, b_m70, b_m71;
        FFTZ_DOUBLE b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9,
                    b_s10, b_s11, b_s12, b_s13, b_s14, b_s15, b_s16, b_s17, b_s18,
                    b_s19, b_s20, b_s21, b_s22, b_s23, b_s24, b_s25, b_s26, b_s27,
                    b_s28, b_s29, b_s30, b_s31, b_s32, b_s33, b_s34, b_s35, b_s36,
                    b_s37, b_s38, b_s39, b_s40, b_s41, b_s42, b_s43, b_s44, b_s45,
                    b_s46, b_s47, b_s48, b_s49, b_s50, b_s51, b_s52, b_s53, b_s54,
                    b_s55, b_s56, b_s57, b_s58, b_s59, b_s60, b_s61, b_s62, b_s63,
                    b_s64, b_s65, b_s66, b_s67, b_s68, b_s69, b_s70, b_s71;

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
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hcf_rfft13avx128(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hcf_rfft13avx128_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft13avx128_fp64_fwd;
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
            return r2hcf_rfft13avx128_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft13avx128_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
