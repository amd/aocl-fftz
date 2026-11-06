// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft16avx128.c
 *
 *  @brief Radix-16 r2hc_fused Real-FFT kernel with with AVX-128 operations
 *  using x86 SIMD intrinsics.
 *
 *  This file contains the DIT radix-16 real-to-halfcomplex fused of two
 *  different implementations (Standard DFT and Shifted DFT that differs in DFT
 *  weight matrix) using x86 SIMD intrinsics for single-precision  and
 *  double-precision inputs
 *
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/r2hc_simd_avx128.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                {{{0, 42, 124, 196, 132, 3},
                                                  {0, 50, 124, 196, 147, 1}},
                                                 {{0, 42, 124, 98,  30,  3},
                                                  {0, 50, 124, 98,  30,  1}}};

ops_cycles_t get_ops_cnt_r2hcf_rfft16avx128(FFTZ_UINT8 precision,
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

static FFTZ_VOID r2hcf_rfft16avx128_fp32_fwd(FFTZ_VOID *in_real,
                                             FFTZ_VOID *in_complex,
                                             FFTZ_VOID *out_real,
                                             FFTZ_VOID *out_complex,
                                             FFTZ_INTP n,
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
    FFTZ_UINT8 is_contiguous_in = (v_in_stride == 1);
    FFTZ_INTP v_out_dc_nyq_stride = strides->v_out_sym_stride;
    FFTZ_UINT8 is_contiguous_out_dc_nyq = (v_out_dc_nyq_stride == 1);

    FFTZ_INTP cnt;
    FFTZ_FLOAT *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_128_S;

    __m128 v_CRTM_16_1 = _mm_broadcast_ss(&CRTM_16_1);
    __m128 v_CRTM_16_2 = _mm_broadcast_ss(&CRTM_16_2);
    __m128 v_CRTM_16_3 = _mm_broadcast_ss(&CRTM_16_3);
    __m128 v_CRTM_16_4 = _mm_broadcast_ss(&CRTM_16_4);
    __m128 v_CRTM_16_5 = _mm_broadcast_ss(&CRTM_16_5);
    __m128 v_CRTM_16_6 = _mm_broadcast_ss(&CRTM_16_6);
    __m128 v_CRTM_16_7 = _mm_broadcast_ss(&CRTM_16_7);

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
               av_in8, av_in9, av_in10, av_in11, av_in12, av_in13, av_in14,
               av_in15;
        __m128 av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8, av_s9,
               av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16, av_s17,
               av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24, av_s25,
               av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32, av_s33,
               av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40, av_s41,
               av_s42;
        __m128 av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8, av_t9,
               av_t10, av_t11, av_t12;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21,
               v_out22, v_out23, v_out24, v_out25, v_out26, v_out27, v_out28,
               v_out29, v_out30, v_out31;

        curr_in = in_r;
        curr_out = out_cp;

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_stride, av_in0, is_contiguous_in);
        // Input point 3: x(2)
        curr_in = in_r + in_strides[2];
        LDR_128_S(curr_in, v_in_stride, av_in1, is_contiguous_in);
        // Input point 5: x(4)
        curr_in = in_r + in_strides[4];
        LDR_128_S(curr_in, v_in_stride, av_in2, is_contiguous_in);
        // Input point 7: x(6)
        curr_in = in_r + in_strides[6];
        LDR_128_S(curr_in, v_in_stride, av_in3, is_contiguous_in);
        // Input point 9: x(8)
        curr_in = in_r + in_strides[8];
        LDR_128_S(curr_in, v_in_stride, av_in4, is_contiguous_in);
        // Input point 11: x(10)
        curr_in = in_r + in_strides[10];
        LDR_128_S(curr_in, v_in_stride, av_in5, is_contiguous_in);
        // Input point 13: x(12)
        curr_in = in_r + in_strides[12];
        LDR_128_S(curr_in, v_in_stride, av_in6, is_contiguous_in);
        // Input point 15: x(14)
        curr_in = in_r + in_strides[14];
        LDR_128_S(curr_in, v_in_stride, av_in7, is_contiguous_in);
        // Input point 17: x(16)
        curr_in = in_r + in_strides[16];
        LDR_128_S(curr_in, v_in_stride, av_in8, is_contiguous_in);
        // Input point 19: x(18)
        curr_in = in_r + in_strides[18];
        LDR_128_S(curr_in, v_in_stride, av_in9, is_contiguous_in);
        // Input point 21: x(20)
        curr_in = in_r + in_strides[20];
        LDR_128_S(curr_in, v_in_stride, av_in10, is_contiguous_in);
        // Input point 23: x(22)
        curr_in = in_r + in_strides[22];
        LDR_128_S(curr_in, v_in_stride, av_in11, is_contiguous_in);
        // Input point 25: x(24)
        curr_in = in_r + in_strides[24];
        LDR_128_S(curr_in, v_in_stride, av_in12, is_contiguous_in);
        // Input point 27: x(26)
        curr_in = in_r + in_strides[26];
        LDR_128_S(curr_in, v_in_stride, av_in13, is_contiguous_in);
        // Input point 29: x(28)
        curr_in = in_r + in_strides[28];
        LDR_128_S(curr_in, v_in_stride, av_in14, is_contiguous_in);
        // Input point 31: x(30)
        curr_in = in_r + in_strides[30];
        LDR_128_S(curr_in, v_in_stride, av_in15, is_contiguous_in);

        av_s1 = _mm_add_ps(av_in0, av_in8);
        av_s2 = _mm_sub_ps(av_in0, av_in8);
        av_s3 = _mm_add_ps(av_in1, av_in15);
        av_s4 = _mm_sub_ps(av_in1, av_in15);
        av_s5 = _mm_add_ps(av_in2, av_in6);
        av_s6 = _mm_sub_ps(av_in2, av_in6);
        av_s7 = _mm_add_ps(av_in3, av_in5);
        av_s8 = _mm_sub_ps(av_in3, av_in5);
        av_s9 = _mm_add_ps(av_in4, av_in12);
        av_s10 = _mm_sub_ps(av_in4, av_in12);
        av_s11 = _mm_add_ps(av_in7, av_in9);
        av_s12 = _mm_sub_ps(av_in7, av_in9);
        av_s13 = _mm_add_ps(av_in10, av_in14);
        av_s14 = _mm_sub_ps(av_in10, av_in14);
        av_s15 = _mm_add_ps(av_in11, av_in13);
        av_s16 = _mm_sub_ps(av_in11, av_in13);

        av_s17 = _mm_add_ps(av_s1, av_s9);
        av_s18 = _mm_sub_ps(av_s1, av_s9);
        av_s19 = _mm_add_ps(av_s3, av_s11);
        av_s20 = _mm_sub_ps(av_s3, av_s11);
        av_s21 = _mm_add_ps(av_s4, av_s12);
        av_s22 = _mm_sub_ps(av_s4, av_s12);
        av_s23 = _mm_add_ps(av_s5, av_s13);
        av_s24 = _mm_sub_ps(av_s5, av_s13);
        // Output point 16: X(15)
        v_out15 = _mm_sub_ps(av_s17, av_s23);

        av_s25 = _mm_add_ps(av_s6, av_s14);
        av_s26 = _mm_sub_ps(av_s6, av_s14);
        av_s27 = _mm_add_ps(av_s7, av_s15);
        av_s28 = _mm_sub_ps(av_s7, av_s15);
        av_s29 = _mm_add_ps(av_s8, av_s16);
        av_s30 = _mm_sub_ps(av_s8, av_s16);
        // Output point 17: X(16)
        v_out16 = _mm_sub_ps(av_s29, av_s22);
        curr_out = out_cp + out_strides[15];
        STRI_2x128_S(curr_out, v_out_stride, v_out15, v_out16);

        av_s31 = _mm_add_ps(av_s19, av_s27);
        av_s32 = _mm_sub_ps(av_s19, av_s27);
        av_s33 = _mm_add_ps(av_s17, av_s23);
        av_s34 = _mm_add_ps(av_s22, av_s29);
        // Output point 1: X(0)
        v_out0 = _mm_add_ps(av_s31, av_s33);
        curr_out = out_r + out_strides[0];
        STR_128_S(curr_out, v_out_dc_nyq_stride, v_out0, is_contiguous_out_dc_nyq);
        // Output pt 32: X(31)
        curr_out = out_r + out_strides[31];
        v_out31 = _mm_sub_ps(av_s33, av_s31);
        STR_128_S(curr_out, v_out_dc_nyq_stride, v_out31, is_contiguous_out_dc_nyq);

        av_t1 = _mm_mul_ps(av_s20, v_CRTM_16_1);
        av_t2 = _mm_mul_ps(av_s21, v_CRTM_16_1);
        av_t3 = _mm_mul_ps(av_s28, v_CRTM_16_1);
        av_t4 = _mm_mul_ps(av_s30, v_CRTM_16_1);
        av_t5 = _mm_mul_ps(av_s20, v_CRTM_16_2);
        av_t6 = _mm_mul_ps(av_s21, v_CRTM_16_2);
        av_t7 = _mm_mul_ps(av_s28, v_CRTM_16_2);
        av_t8 = _mm_mul_ps(av_s30, v_CRTM_16_2);
        av_t9 = _mm_mul_ps(av_s24, v_CRTM_16_3);
        av_t10 = _mm_mul_ps(av_s26, v_CRTM_16_3);
        av_t11 = _mm_mul_ps(av_s32, v_CRTM_16_3);
        // Output point 8: X(7)
        v_out7 = _mm_add_ps(av_s18, av_t11);
        // Output point 24: X(23)
        v_out23 = _mm_sub_ps(av_s18, av_t11);

        av_t12 = _mm_mul_ps(av_s34, v_CRTM_16_3);
        // Output point 9: X(8)
        v_out8 = NEGATE_128_S(_mm_add_ps(av_s25, av_t12));
        curr_out = out_cp + out_strides[7];
        STRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);
        // Output point 25: X(24)
        v_out24 = _mm_sub_ps(av_s25, av_t12);
        curr_out = out_cp + out_strides[23];
        STRI_2x128_S(curr_out, v_out_stride, v_out23, v_out24);

        av_s35 = _mm_add_ps(av_t1, av_t8);
        av_s36 = _mm_add_ps(av_t10, av_s2);
        // Output point 4: X(3)
        v_out3 = _mm_add_ps(av_s35, av_s36);
        // Output point 28: X(27)
        v_out27 = _mm_sub_ps(av_s36, av_s35);

        av_s37 = _mm_add_ps(av_t3, av_t6);
        av_s38 = _mm_add_ps(av_t9, av_s10);
        // Output point 5: X(4)
        v_out4 = NEGATE_128_S(_mm_add_ps(av_s37, av_s38));
        curr_out = out_cp + out_strides[3];
        STRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);
        // Output point 29: X(28)
        v_out28 = _mm_sub_ps(av_s38, av_s37);
        curr_out = out_cp + out_strides[27];
        STRI_2x128_S(curr_out, v_out_stride, v_out27, v_out28);

        av_s39 = _mm_sub_ps(av_t4, av_t5);
        av_s40 = _mm_sub_ps(av_s2, av_t10);
        // Output point 12: X(11)
        v_out11 = _mm_sub_ps(av_s40, av_s39);
        // Output point 20: X(19)
        v_out19 = _mm_add_ps(av_s39, av_s40);

        av_s41 = _mm_sub_ps(av_t7, av_t2);
        av_s42 = _mm_sub_ps(av_s10, av_t9);
        // Output point 13: X(12)
        v_out12 = _mm_add_ps(av_s41, av_s42);
        curr_out = out_cp + out_strides[11];
        STRI_2x128_S(curr_out, v_out_stride, v_out11, v_out12);
        // Output point 21: X(20)
        v_out20 = _mm_sub_ps(av_s41, av_s42);
        curr_out = out_cp + out_strides[19];
        STRI_2x128_S(curr_out, v_out_stride, v_out19, v_out20);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8, bv_in9, bv_in10, bv_in11, bv_in12, bv_in13, bv_in14,
               bv_in15;
        __m128 bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
               bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17,
               bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24, bv_s25,
               bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32, bv_s33,
               bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40, bv_s41,
               bv_s42, bv_s43, bv_s44, bv_s45, bv_s46, bv_s47, bv_s48, bv_s49,
               bv_s50;
        __m128 bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
               bv_t10, bv_t11, bv_t12, bv_t13, bv_t14, bv_t15, bv_t16, bv_t17,
               bv_t18, bv_t19, bv_t20, bv_t21, bv_t22, bv_t23, bv_t24, bv_t25,
               bv_t26, bv_t27, bv_t28, bv_t29, bv_t30;

        // Input point 2: x(1)
        curr_in = in_r + in_strides[1];
        LDR_128_S(curr_in, v_in_stride, bv_in0, is_contiguous_in);
        // Input point 4: x(3)
        curr_in = in_r + in_strides[3];
        LDR_128_S(curr_in, v_in_stride, bv_in1, is_contiguous_in);
        // Input point 6: x(5)
        curr_in = in_r + in_strides[5];
        LDR_128_S(curr_in, v_in_stride, bv_in2, is_contiguous_in);
        // Input point 8: x(7)
        curr_in = in_r + in_strides[7];
        LDR_128_S(curr_in, v_in_stride, bv_in3, is_contiguous_in);
        // Input point 10: x(9)
        curr_in = in_r + in_strides[9];
        LDR_128_S(curr_in, v_in_stride, bv_in4, is_contiguous_in);
        // Input point 12: x(11)
        curr_in = in_r + in_strides[11];
        LDR_128_S(curr_in, v_in_stride, bv_in5, is_contiguous_in);
        // Input point 14: x(13)
        curr_in = in_r + in_strides[13];
        LDR_128_S(curr_in, v_in_stride, bv_in6, is_contiguous_in);
        // Input point 16: x(15)
        curr_in = in_r + in_strides[15];
        LDR_128_S(curr_in, v_in_stride, bv_in7, is_contiguous_in);
        // Input point 18: x(17)
        curr_in = in_r + in_strides[17];
        LDR_128_S(curr_in, v_in_stride, bv_in8, is_contiguous_in);
        // Input point 20: x(19)
        curr_in = in_r + in_strides[19];
        LDR_128_S(curr_in, v_in_stride, bv_in9, is_contiguous_in);
        // Input point 22: x(21)
        curr_in = in_r + in_strides[21];
        LDR_128_S(curr_in, v_in_stride, bv_in10, is_contiguous_in);
        // Input point 24: x(23)
        curr_in = in_r + in_strides[23];
        LDR_128_S(curr_in, v_in_stride, bv_in11, is_contiguous_in);
        // Input point 26: x(25)
        curr_in = in_r + in_strides[25];
        LDR_128_S(curr_in, v_in_stride, bv_in12, is_contiguous_in);
        // Input point 28: x(27)
        curr_in = in_r + in_strides[27];
        LDR_128_S(curr_in, v_in_stride, bv_in13, is_contiguous_in);
        // Input point 30: x(29)
        curr_in = in_r + in_strides[29];
        LDR_128_S(curr_in, v_in_stride, bv_in14, is_contiguous_in);
        // Input point 32: x(31)
        curr_in = in_r + in_strides[31];
        LDR_128_S(curr_in, v_in_stride, bv_in15, is_contiguous_in);

        bv_s1 = _mm_sub_ps(bv_in3, bv_in11);
        bv_s2 = _mm_add_ps(bv_in3, bv_in11);
        bv_s3 = _mm_sub_ps(bv_in4, bv_in12);
        bv_s4 = _mm_add_ps(bv_in4, bv_in12);
        bv_s5 = _mm_sub_ps(bv_in5, bv_in13);
        bv_s6 = _mm_add_ps(bv_in5, bv_in13);

        bv_t1 = _mm_mul_ps(v_CRTM_16_3, bv_s3);
        bv_s7 = _mm_add_ps(bv_t1, bv_in0);
        bv_t7 = _mm_mul_ps(v_CRTM_16_1, bv_in2);
        bv_t8 = _mm_mul_ps(v_CRTM_16_2, bv_in10);
        bv_t9 = _mm_mul_ps(v_CRTM_16_2, bv_in6);
        bv_t10 = _mm_mul_ps(v_CRTM_16_1, bv_in14);
        bv_s8 = _mm_sub_ps(bv_t7, bv_t8);
        bv_s9 = _mm_sub_ps(bv_t9, bv_t10);
        bv_s10 = _mm_add_ps(bv_s8, bv_s9);
        bv_s11 = _mm_sub_ps(bv_s7, bv_s10);
        bv_s12 = _mm_add_ps(bv_s7, bv_s10);

        bv_t2 = _mm_mul_ps(v_CRTM_16_3, bv_s5);
        bv_s13 = _mm_add_ps(bv_t2, bv_in1);
        bv_t3 = _mm_mul_ps(v_CRTM_16_3, bv_s6);
        bv_s14 = _mm_add_ps(bv_t3, bv_in9);

        bv_t11 = _mm_mul_ps(v_CRTM_16_6, bv_s13);
        bv_t12 = _mm_mul_ps(v_CRTM_16_7, bv_s14);
        bv_s15 = _mm_sub_ps(bv_t11, bv_t12);

        bv_t4 = _mm_mul_ps(v_CRTM_16_3, bv_s2);
        bv_s16 = _mm_add_ps(bv_t4, bv_in7);
        bv_t5 = _mm_mul_ps(v_CRTM_16_3, bv_s1);
        bv_s17 = _mm_sub_ps(bv_t5, bv_in15);

        bv_t13 = _mm_mul_ps(v_CRTM_16_7, bv_s16);
        bv_t14 = _mm_mul_ps(v_CRTM_16_6, bv_s17);
        bv_s18 = _mm_add_ps(bv_t13, bv_t14);

        bv_s19 = _mm_add_ps(bv_s15, bv_s18);
        bv_s20 = _mm_sub_ps(bv_s18, bv_s15);

        // Output point 2: X(1)
        v_out1 = _mm_add_ps(bv_s12, bv_s19);
        // Output point 30: X(29)
        v_out29 = _mm_sub_ps(bv_s12, bv_s19);

        bv_t6 = _mm_mul_ps(v_CRTM_16_3, bv_s4);
        bv_s21 = _mm_add_ps(bv_t6, bv_in8);

        bv_t15 = _mm_mul_ps(v_CRTM_16_2, bv_in2);
        bv_t16 = _mm_mul_ps(v_CRTM_16_1, bv_in10);
        bv_t17 = _mm_mul_ps(v_CRTM_16_1, bv_in6);
        bv_t18 = _mm_mul_ps(v_CRTM_16_2, bv_in14);
        bv_s22 = _mm_add_ps(bv_t15, bv_t16);
        bv_s23 = _mm_add_ps(bv_t17, bv_t18);
        bv_s24 = _mm_add_ps(bv_s22, bv_s23);
        bv_s25 = _mm_sub_ps(bv_s21, bv_s24);
        bv_s26 = _mm_add_ps(bv_s21, bv_s24);
        // Output point 15: X(14)
        v_out14 = _mm_add_ps(bv_s25, bv_s20);
        // Output point 19: X(18)
        v_out18 = _mm_sub_ps(bv_s20, bv_s25);

        bv_t19 = _mm_mul_ps(v_CRTM_16_6, bv_s14);
        bv_t20 = _mm_mul_ps(v_CRTM_16_7, bv_s13);
        bv_t21 = _mm_mul_ps(v_CRTM_16_7, bv_s17);
        bv_t22 = _mm_mul_ps(v_CRTM_16_6, bv_s16);
        bv_s27 = _mm_add_ps(bv_t19, bv_t20);
        bv_s28 = _mm_sub_ps(bv_t21, bv_t22);

        bv_s29 = _mm_add_ps(bv_s28, bv_s27);
        bv_s30 = _mm_sub_ps(bv_s28, bv_s27);
        // Output point 3: X(2)
        v_out2 = _mm_sub_ps(bv_s30, bv_s26);
        curr_out = out_cp + out_strides[1];
        STRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 31: X(30)
        v_out30 = _mm_add_ps(bv_s30, bv_s26);
        curr_out = out_cp + out_strides[29];
        STRI_2x128_S(curr_out, v_out_stride, v_out29, v_out30);

        // Output point 18: X(17)
        v_out17 = _mm_sub_ps(bv_s11, bv_s29);
        curr_out = out_cp + out_strides[17];
        STRI_2x128_S(curr_out, v_out_stride, v_out17, v_out18);
        // Output point 14: X(13)
        v_out13 = _mm_add_ps(bv_s11, bv_s29);
        curr_out = out_cp + out_strides[13];
        STRI_2x128_S(curr_out, v_out_stride, v_out13, v_out14);

        bv_s31 = _mm_sub_ps(bv_in0, bv_t1);
        bv_s32 = _mm_sub_ps(bv_s22, bv_s23);
        bv_s33 = _mm_add_ps(bv_s31, bv_s32);
        bv_s34 = _mm_sub_ps(bv_s31, bv_s32);

        bv_s35 = _mm_sub_ps(bv_in9, bv_t3);
        bv_s36 = _mm_sub_ps(bv_in1, bv_t2);

        bv_t23 = _mm_mul_ps(v_CRTM_16_4, bv_s35);
        bv_t24 = _mm_mul_ps(v_CRTM_16_5, bv_s36);
        bv_s37 = _mm_add_ps(bv_t23, bv_t24);

        bv_s38 = _mm_sub_ps(bv_in7, bv_t4);
        bv_s39 = _mm_add_ps(bv_in15, bv_t5);

        bv_t25 = _mm_mul_ps(v_CRTM_16_4, bv_s38);
        bv_t26 = _mm_mul_ps(v_CRTM_16_5, bv_s39);
        bv_s40 = _mm_add_ps(bv_t25, bv_t26);

        bv_s41 = _mm_sub_ps(bv_s37, bv_s40);
        bv_s42 = _mm_add_ps(bv_s37, bv_s40);

        // Output point 6: X(5)
        v_out5 = _mm_add_ps(bv_s33, bv_s41);
        // Output point 26: X(25)
        v_out25 = _mm_sub_ps(bv_s33, bv_s41);

        bv_s43 = _mm_sub_ps(bv_in8, bv_t6);
        bv_s44 = _mm_sub_ps(bv_s9, bv_s8);
        bv_s45 = _mm_sub_ps(bv_s44, bv_s43);
        bv_s46 = _mm_add_ps(bv_s44, bv_s43);
        // Output point 11: X(10)
        v_out10 = _mm_sub_ps(bv_s45, bv_s42);
        // Output point 23: X(22)
        v_out22 = NEGATE_128_S(_mm_add_ps(bv_s45, bv_s42));

        bv_t27 = _mm_mul_ps(v_CRTM_16_5, bv_s38);
        bv_t28 = _mm_mul_ps(v_CRTM_16_4, bv_s39);
        bv_t29 = _mm_mul_ps(v_CRTM_16_5, bv_s35);
        bv_t30 = _mm_mul_ps(v_CRTM_16_4, bv_s36);
        bv_s47 = _mm_sub_ps(bv_t27, bv_t28);
        bv_s48 = _mm_sub_ps(bv_t29, bv_t30);

        bv_s49 = _mm_sub_ps(bv_s47, bv_s48);
        bv_s50 = _mm_add_ps(bv_s48, bv_s47);

        // Output point 22: X(21)
        v_out21 = _mm_sub_ps(bv_s34, bv_s49);
        curr_out = out_cp + out_strides[21];
        STRI_2x128_S(curr_out, v_out_stride, v_out21, v_out22);
        // Output point 10: X(9)
        v_out9 = _mm_add_ps(bv_s34, bv_s49);
        curr_out = out_cp + out_strides[9];
        STRI_2x128_S(curr_out, v_out_stride, v_out9, v_out10);

        // Output point 27: X(26)
        v_out26 = _mm_sub_ps(bv_s50, bv_s46);
        curr_out = out_cp + out_strides[25];
        STRI_2x128_S(curr_out, v_out_stride, v_out25, v_out26);
        // Output point 7: X(6)
        v_out6 = _mm_add_ps(bv_s50, bv_s46);
        curr_out = out_cp + out_strides[5];
        STRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);

        in_r += v_in_stride * NUM_SETS_REAL_128_S;
        out_cp += v_out_stride * NUM_SETS_REAL_128_S;
        out_r += v_out_dc_nyq_stride * NUM_SETS_REAL_128_S;
    }
    // tail cases
    if (n & 2)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
               av_in8, av_in9, av_in10, av_in11, av_in12, av_in13, av_in14,
               av_in15;
        __m128 av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8, av_s9,
               av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16, av_s17,
               av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24, av_s25,
               av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32, av_s33,
               av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40, av_s41,
               av_s42;
        __m128 av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8, av_t9,
               av_t10, av_t11, av_t12;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21,
               v_out22, v_out23, v_out24, v_out25, v_out26, v_out27, v_out28,
               v_out29, v_out30, v_out31;

        curr_in = in_r;
        curr_out = out_cp;

        // Input point 1: x(0)
        LDHR_128_S(curr_in, v_in_stride, av_in0);
        // Input point 3: x(2)
        curr_in = in_r + in_strides[2];
        LDHR_128_S(curr_in, v_in_stride, av_in1);
        // Input point 5: x(4)
        curr_in = in_r + in_strides[4];
        LDHR_128_S(curr_in, v_in_stride, av_in2);
        // Input point 7: x(6)
        curr_in = in_r + in_strides[6];
        LDHR_128_S(curr_in, v_in_stride, av_in3);
        // Input point 9: x(8)
        curr_in = in_r + in_strides[8];
        LDHR_128_S(curr_in, v_in_stride, av_in4);
        // Input point 11: x(10)
        curr_in = in_r + in_strides[10];
        LDHR_128_S(curr_in, v_in_stride, av_in5);
        // Input point 13: x(12)
        curr_in = in_r + in_strides[12];
        LDHR_128_S(curr_in, v_in_stride, av_in6);
        // Input point 15: x(14)
        curr_in = in_r + in_strides[14];
        LDHR_128_S(curr_in, v_in_stride, av_in7);
        // Input point 17: x(16)
        curr_in = in_r + in_strides[16];
        LDHR_128_S(curr_in, v_in_stride, av_in8);
        // Input point 19: x(18)
        curr_in = in_r + in_strides[18];
        LDHR_128_S(curr_in, v_in_stride, av_in9);
        // Input point 21: x(20)
        curr_in = in_r + in_strides[20];
        LDHR_128_S(curr_in, v_in_stride, av_in10);
        // Input point 23: x(22)
        curr_in = in_r + in_strides[22];
        LDHR_128_S(curr_in, v_in_stride, av_in11);
        // Input point 25: x(24)
        curr_in = in_r + in_strides[24];
        LDHR_128_S(curr_in, v_in_stride, av_in12);
        // Input point 27: x(26)
        curr_in = in_r + in_strides[26];
        LDHR_128_S(curr_in, v_in_stride, av_in13);
        // Input point 29: x(28)
        curr_in = in_r + in_strides[28];
        LDHR_128_S(curr_in, v_in_stride, av_in14);
        // Input point 31: x(30)
        curr_in = in_r + in_strides[30];
        LDHR_128_S(curr_in, v_in_stride, av_in15);

        av_s1 = _mm_add_ps(av_in0, av_in8);
        av_s2 = _mm_sub_ps(av_in0, av_in8);
        av_s3 = _mm_add_ps(av_in1, av_in15);
        av_s4 = _mm_sub_ps(av_in1, av_in15);
        av_s5 = _mm_add_ps(av_in2, av_in6);
        av_s6 = _mm_sub_ps(av_in2, av_in6);
        av_s7 = _mm_add_ps(av_in3, av_in5);
        av_s8 = _mm_sub_ps(av_in3, av_in5);
        av_s9 = _mm_add_ps(av_in4, av_in12);
        av_s10 = _mm_sub_ps(av_in4, av_in12);
        av_s11 = _mm_add_ps(av_in7, av_in9);
        av_s12 = _mm_sub_ps(av_in7, av_in9);
        av_s13 = _mm_add_ps(av_in10, av_in14);
        av_s14 = _mm_sub_ps(av_in10, av_in14);
        av_s15 = _mm_add_ps(av_in11, av_in13);
        av_s16 = _mm_sub_ps(av_in11, av_in13);

        av_s17 = _mm_add_ps(av_s1, av_s9);
        av_s18 = _mm_sub_ps(av_s1, av_s9);
        av_s19 = _mm_add_ps(av_s3, av_s11);
        av_s20 = _mm_sub_ps(av_s3, av_s11);
        av_s21 = _mm_add_ps(av_s4, av_s12);
        av_s22 = _mm_sub_ps(av_s4, av_s12);
        av_s23 = _mm_add_ps(av_s5, av_s13);
        av_s24 = _mm_sub_ps(av_s5, av_s13);
        // Output point 16: X(15)
        v_out15 = _mm_sub_ps(av_s17, av_s23);

        av_s25 = _mm_add_ps(av_s6, av_s14);
        av_s26 = _mm_sub_ps(av_s6, av_s14);
        av_s27 = _mm_add_ps(av_s7, av_s15);
        av_s28 = _mm_sub_ps(av_s7, av_s15);
        av_s29 = _mm_add_ps(av_s8, av_s16);
        av_s30 = _mm_sub_ps(av_s8, av_s16);
        // Output point 17: X(16)
        v_out16 = _mm_sub_ps(av_s29, av_s22);
        curr_out = out_cp + out_strides[15];
        STHRI_2x128_S(curr_out, v_out_stride, v_out15, v_out16);

        av_s31 = _mm_add_ps(av_s19, av_s27);
        av_s32 = _mm_sub_ps(av_s19, av_s27);
        av_s33 = _mm_add_ps(av_s17, av_s23);
        av_s34 = _mm_add_ps(av_s22, av_s29);
        // Output point 1: X(0)
        v_out0 = _mm_add_ps(av_s31, av_s33);
        curr_out = out_r + out_strides[0];
        STHR_128_S(curr_out, v_out_dc_nyq_stride, v_out0);
        // Output pt 32: X(31)
        curr_out = out_r + out_strides[31];
        v_out31 = _mm_sub_ps(av_s33, av_s31);
        STHR_128_S(curr_out, v_out_dc_nyq_stride, v_out31);

        av_t1 = _mm_mul_ps(av_s20, v_CRTM_16_1);
        av_t2 = _mm_mul_ps(av_s21, v_CRTM_16_1);
        av_t3 = _mm_mul_ps(av_s28, v_CRTM_16_1);
        av_t4 = _mm_mul_ps(av_s30, v_CRTM_16_1);
        av_t5 = _mm_mul_ps(av_s20, v_CRTM_16_2);
        av_t6 = _mm_mul_ps(av_s21, v_CRTM_16_2);
        av_t7 = _mm_mul_ps(av_s28, v_CRTM_16_2);
        av_t8 = _mm_mul_ps(av_s30, v_CRTM_16_2);
        av_t9 = _mm_mul_ps(av_s24, v_CRTM_16_3);
        av_t10 = _mm_mul_ps(av_s26, v_CRTM_16_3);
        av_t11 = _mm_mul_ps(av_s32, v_CRTM_16_3);
        // Output point 8: X(7)
        v_out7 = _mm_add_ps(av_s18, av_t11);
        // Output point 24: X(23)
        v_out23 = _mm_sub_ps(av_s18, av_t11);

        av_t12 = _mm_mul_ps(av_s34, v_CRTM_16_3);
        // Output point 9: X(8)
        v_out8 = NEGATE_128_S(_mm_add_ps(av_s25, av_t12));
        curr_out = out_cp + out_strides[7];
        STHRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);
        // Output point 25: X(24)
        v_out24 = _mm_sub_ps(av_s25, av_t12);
        curr_out = out_cp + out_strides[23];
        STHRI_2x128_S(curr_out, v_out_stride, v_out23, v_out24);

        av_s35 = _mm_add_ps(av_t1, av_t8);
        av_s36 = _mm_add_ps(av_t10, av_s2);
        // Output point 4: X(3)
        v_out3 = _mm_add_ps(av_s35, av_s36);
        // Output point 28: X(27)
        v_out27 = _mm_sub_ps(av_s36, av_s35);

        av_s37 = _mm_add_ps(av_t3, av_t6);
        av_s38 = _mm_add_ps(av_t9, av_s10);
        // Output point 5: X(4)
        v_out4 = NEGATE_128_S(_mm_add_ps(av_s37, av_s38));
        curr_out = out_cp + out_strides[3];
        STHRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);
        // Output point 29: X(28)
        v_out28 = _mm_sub_ps(av_s38, av_s37);
        curr_out = out_cp + out_strides[27];
        STHRI_2x128_S(curr_out, v_out_stride, v_out27, v_out28);

        av_s39 = _mm_sub_ps(av_t4, av_t5);
        av_s40 = _mm_sub_ps(av_s2, av_t10);
        // Output point 12: X(11)
        v_out11 = _mm_sub_ps(av_s40, av_s39);
        // Output point 20: X(19)
        v_out19 = _mm_add_ps(av_s39, av_s40);

        av_s41 = _mm_sub_ps(av_t7, av_t2);
        av_s42 = _mm_sub_ps(av_s10, av_t9);
        // Output point 13: X(12)
        v_out12 = _mm_add_ps(av_s41, av_s42);
        curr_out = out_cp + out_strides[11];
        STHRI_2x128_S(curr_out, v_out_stride, v_out11, v_out12);
        // Output point 21: X(20)
        v_out20 = _mm_sub_ps(av_s41, av_s42);
        curr_out = out_cp + out_strides[19];
        STHRI_2x128_S(curr_out, v_out_stride, v_out19, v_out20);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8, bv_in9, bv_in10, bv_in11, bv_in12, bv_in13, bv_in14,
               bv_in15;
        __m128 bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
               bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17,
               bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24, bv_s25,
               bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32, bv_s33,
               bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40, bv_s41,
               bv_s42, bv_s43, bv_s44, bv_s45, bv_s46, bv_s47, bv_s48, bv_s49,
               bv_s50;
        __m128 bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
               bv_t10, bv_t11, bv_t12, bv_t13, bv_t14, bv_t15, bv_t16, bv_t17,
               bv_t18, bv_t19, bv_t20, bv_t21, bv_t22, bv_t23, bv_t24, bv_t25,
               bv_t26, bv_t27, bv_t28, bv_t29, bv_t30;

        // Input point 2: x(1)
        curr_in = in_r + in_strides[1];
        LDHR_128_S(curr_in, v_in_stride, bv_in0);
        // Input point 4: x(3)
        curr_in = in_r + in_strides[3];
        LDHR_128_S(curr_in, v_in_stride, bv_in1);
        // Input point 6: x(5)
        curr_in = in_r + in_strides[5];
        LDHR_128_S(curr_in, v_in_stride, bv_in2);
        // Input point 8: x(7)
        curr_in = in_r + in_strides[7];
        LDHR_128_S(curr_in, v_in_stride, bv_in3);
        // Input point 10: x(9)
        curr_in = in_r + in_strides[9];
        LDHR_128_S(curr_in, v_in_stride, bv_in4);
        // Input point 12: x(11)
        curr_in = in_r + in_strides[11];
        LDHR_128_S(curr_in, v_in_stride, bv_in5);
        // Input point 14: x(13)
        curr_in = in_r + in_strides[13];
        LDHR_128_S(curr_in, v_in_stride, bv_in6);
        // Input point 16: x(15)
        curr_in = in_r + in_strides[15];
        LDHR_128_S(curr_in, v_in_stride, bv_in7);
        // Input point 18: x(17)
        curr_in = in_r + in_strides[17];
        LDHR_128_S(curr_in, v_in_stride, bv_in8);
        // Input point 20: x(19)
        curr_in = in_r + in_strides[19];
        LDHR_128_S(curr_in, v_in_stride, bv_in9);
        // Input point 22: x(21)
        curr_in = in_r + in_strides[21];
        LDHR_128_S(curr_in, v_in_stride, bv_in10);
        // Input point 24: x(23)
        curr_in = in_r + in_strides[23];
        LDHR_128_S(curr_in, v_in_stride, bv_in11);
        // Input point 26: x(25)
        curr_in = in_r + in_strides[25];
        LDHR_128_S(curr_in, v_in_stride, bv_in12);
        // Input point 28: x(27)
        curr_in = in_r + in_strides[27];
        LDHR_128_S(curr_in, v_in_stride, bv_in13);
        // Input point 30: x(29)
        curr_in = in_r + in_strides[29];
        LDHR_128_S(curr_in, v_in_stride, bv_in14);
        // Input point 32: x(31)
        curr_in = in_r + in_strides[31];
        LDHR_128_S(curr_in, v_in_stride, bv_in15);

        bv_s1 = _mm_sub_ps(bv_in3, bv_in11);
        bv_s2 = _mm_add_ps(bv_in3, bv_in11);
        bv_s3 = _mm_sub_ps(bv_in4, bv_in12);
        bv_s4 = _mm_add_ps(bv_in4, bv_in12);
        bv_s5 = _mm_sub_ps(bv_in5, bv_in13);
        bv_s6 = _mm_add_ps(bv_in5, bv_in13);

        bv_t1 = _mm_mul_ps(v_CRTM_16_3, bv_s3);
        bv_s7 = _mm_add_ps(bv_t1, bv_in0);
        bv_t7 = _mm_mul_ps(v_CRTM_16_1, bv_in2);
        bv_t8 = _mm_mul_ps(v_CRTM_16_2, bv_in10);
        bv_t9 = _mm_mul_ps(v_CRTM_16_2, bv_in6);
        bv_t10 = _mm_mul_ps(v_CRTM_16_1, bv_in14);
        bv_s8 = _mm_sub_ps(bv_t7, bv_t8);
        bv_s9 = _mm_sub_ps(bv_t9, bv_t10);
        bv_s10 = _mm_add_ps(bv_s8, bv_s9);
        bv_s11 = _mm_sub_ps(bv_s7, bv_s10);
        bv_s12 = _mm_add_ps(bv_s7, bv_s10);

        bv_t2 = _mm_mul_ps(v_CRTM_16_3, bv_s5);
        bv_s13 = _mm_add_ps(bv_t2, bv_in1);
        bv_t3 = _mm_mul_ps(v_CRTM_16_3, bv_s6);
        bv_s14 = _mm_add_ps(bv_t3, bv_in9);

        bv_t11 = _mm_mul_ps(v_CRTM_16_6, bv_s13);
        bv_t12 = _mm_mul_ps(v_CRTM_16_7, bv_s14);
        bv_s15 = _mm_sub_ps(bv_t11, bv_t12);

        bv_t4 = _mm_mul_ps(v_CRTM_16_3, bv_s2);
        bv_s16 = _mm_add_ps(bv_t4, bv_in7);
        bv_t5 = _mm_mul_ps(v_CRTM_16_3, bv_s1);
        bv_s17 = _mm_sub_ps(bv_t5, bv_in15);

        bv_t13 = _mm_mul_ps(v_CRTM_16_7, bv_s16);
        bv_t14 = _mm_mul_ps(v_CRTM_16_6, bv_s17);
        bv_s18 = _mm_add_ps(bv_t13, bv_t14);

        bv_s19 = _mm_add_ps(bv_s15, bv_s18);
        bv_s20 = _mm_sub_ps(bv_s18, bv_s15);

        // Output point 2: X(1)
        v_out1 = _mm_add_ps(bv_s12, bv_s19);
        // Output point 30: X(29)
        v_out29 = _mm_sub_ps(bv_s12, bv_s19);

        bv_t6 = _mm_mul_ps(v_CRTM_16_3, bv_s4);
        bv_s21 = _mm_add_ps(bv_t6, bv_in8);

        bv_t15 = _mm_mul_ps(v_CRTM_16_2, bv_in2);
        bv_t16 = _mm_mul_ps(v_CRTM_16_1, bv_in10);
        bv_t17 = _mm_mul_ps(v_CRTM_16_1, bv_in6);
        bv_t18 = _mm_mul_ps(v_CRTM_16_2, bv_in14);
        bv_s22 = _mm_add_ps(bv_t15, bv_t16);
        bv_s23 = _mm_add_ps(bv_t17, bv_t18);
        bv_s24 = _mm_add_ps(bv_s22, bv_s23);
        bv_s25 = _mm_sub_ps(bv_s21, bv_s24);
        bv_s26 = _mm_add_ps(bv_s21, bv_s24);
        // Output point 15: X(14)
        v_out14 = _mm_add_ps(bv_s25, bv_s20);
        // Output point 19: X(18)
        v_out18 = _mm_sub_ps(bv_s20, bv_s25);

        bv_t19 = _mm_mul_ps(v_CRTM_16_6, bv_s14);
        bv_t20 = _mm_mul_ps(v_CRTM_16_7, bv_s13);
        bv_t21 = _mm_mul_ps(v_CRTM_16_7, bv_s17);
        bv_t22 = _mm_mul_ps(v_CRTM_16_6, bv_s16);
        bv_s27 = _mm_add_ps(bv_t19, bv_t20);
        bv_s28 = _mm_sub_ps(bv_t21, bv_t22);

        bv_s29 = _mm_add_ps(bv_s28, bv_s27);
        bv_s30 = _mm_sub_ps(bv_s28, bv_s27);
        // Output point 3: X(2)
        v_out2 = _mm_sub_ps(bv_s30, bv_s26);
        curr_out = out_cp + out_strides[1];
        STHRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 31: X(30)
        v_out30 = _mm_add_ps(bv_s30, bv_s26);
        curr_out = out_cp + out_strides[29];
        STHRI_2x128_S(curr_out, v_out_stride, v_out29, v_out30);

        // Output point 18: X(17)
        v_out17 = _mm_sub_ps(bv_s11, bv_s29);
        curr_out = out_cp + out_strides[17];
        STHRI_2x128_S(curr_out, v_out_stride, v_out17, v_out18);
        // Output point 14: X(13)
        v_out13 = _mm_add_ps(bv_s11, bv_s29);
        curr_out = out_cp + out_strides[13];
        STHRI_2x128_S(curr_out, v_out_stride, v_out13, v_out14);

        bv_s31 = _mm_sub_ps(bv_in0, bv_t1);
        bv_s32 = _mm_sub_ps(bv_s22, bv_s23);
        bv_s33 = _mm_add_ps(bv_s31, bv_s32);
        bv_s34 = _mm_sub_ps(bv_s31, bv_s32);

        bv_s35 = _mm_sub_ps(bv_in9, bv_t3);
        bv_s36 = _mm_sub_ps(bv_in1, bv_t2);

        bv_t23 = _mm_mul_ps(v_CRTM_16_4, bv_s35);
        bv_t24 = _mm_mul_ps(v_CRTM_16_5, bv_s36);
        bv_s37 = _mm_add_ps(bv_t23, bv_t24);

        bv_s38 = _mm_sub_ps(bv_in7, bv_t4);
        bv_s39 = _mm_add_ps(bv_in15, bv_t5);

        bv_t25 = _mm_mul_ps(v_CRTM_16_4, bv_s38);
        bv_t26 = _mm_mul_ps(v_CRTM_16_5, bv_s39);
        bv_s40 = _mm_add_ps(bv_t25, bv_t26);

        bv_s41 = _mm_sub_ps(bv_s37, bv_s40);
        bv_s42 = _mm_add_ps(bv_s37, bv_s40);

        // Output point 6: X(5)
        v_out5 = _mm_add_ps(bv_s33, bv_s41);
        // Output point 26: X(25)
        v_out25 = _mm_sub_ps(bv_s33, bv_s41);

        bv_s43 = _mm_sub_ps(bv_in8, bv_t6);
        bv_s44 = _mm_sub_ps(bv_s9, bv_s8);
        bv_s45 = _mm_sub_ps(bv_s44, bv_s43);
        bv_s46 = _mm_add_ps(bv_s44, bv_s43);
        // Output point 11: X(10)
        v_out10 = _mm_sub_ps(bv_s45, bv_s42);
        // Output point 23: X(22)
        v_out22 = NEGATE_128_S(_mm_add_ps(bv_s45, bv_s42));

        bv_t27 = _mm_mul_ps(v_CRTM_16_5, bv_s38);
        bv_t28 = _mm_mul_ps(v_CRTM_16_4, bv_s39);
        bv_t29 = _mm_mul_ps(v_CRTM_16_5, bv_s35);
        bv_t30 = _mm_mul_ps(v_CRTM_16_4, bv_s36);
        bv_s47 = _mm_sub_ps(bv_t27, bv_t28);
        bv_s48 = _mm_sub_ps(bv_t29, bv_t30);

        bv_s49 = _mm_sub_ps(bv_s47, bv_s48);
        bv_s50 = _mm_add_ps(bv_s48, bv_s47);

        // Output point 22: X(21)
        v_out21 = _mm_sub_ps(bv_s34, bv_s49);
        curr_out = out_cp + out_strides[21];
        STHRI_2x128_S(curr_out, v_out_stride, v_out21, v_out22);
        // Output point 10: X(9)
        v_out9 = _mm_add_ps(bv_s34, bv_s49);
        curr_out = out_cp + out_strides[9];
        STHRI_2x128_S(curr_out, v_out_stride, v_out9, v_out10);

        // Output point 27: X(26)
        v_out26 = _mm_sub_ps(bv_s50, bv_s46);
        curr_out = out_cp + out_strides[25];
        STHRI_2x128_S(curr_out, v_out_stride, v_out25, v_out26);
        // Output point 7: X(6)
        v_out6 = _mm_add_ps(bv_s50, bv_s46);
        curr_out = out_cp + out_strides[5];
        STHRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);

        in_r = in_r + (v_in_stride << 1);
        out_cp = out_cp + (v_out_stride << 1);
        out_r = out_r + (v_out_dc_nyq_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        /* Standard DFT */
        FFTZ_FLOAT a_in0, a_in1, a_in2, a_in3, a_in4, a_in5, a_in6, a_in7,
            a_in8, a_in9, a_in10, a_in11, a_in12, a_in13, a_in14, a_in15;
        FFTZ_FLOAT a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_s9,
            a_s10, a_s11, a_s12, a_s13, a_s14, a_s15, a_s16, a_s17, a_s18,
            a_s19, a_s20, a_s21, a_s22, a_s23, a_t12, a_t16, a_t14, a_s27,
            a_t13, a_s29, a_t15, a_s31, a_s32, a_t17, a_t0, a_t1, a_t2, a_t3,
            a_t4, a_t5, a_t6, a_t7, a_t8, a_t9, a_t10, a_t11, a_s46, a_t18,
            a_t20, a_t19, a_s49, a_s50, a_s51, a_s52;

        curr_in = in_r;
        curr_out = out_r;
        // Input point 1: x(0)
        a_in0 = *in_r;
        // Input point 3: x(2)
        a_in1 = in_r[in_strides[2]];
        // Input point 5: x(4)
        a_in2 = in_r[in_strides[4]];
        // Input point 7: x(6)
        a_in3 = in_r[in_strides[6]];
        // Input point 9: x(8)
        a_in4 = in_r[in_strides[8]];
        // Input point 11: x(10)
        a_in5 = in_r[in_strides[10]];
        // Input point 13: x(12)
        a_in6 = in_r[in_strides[12]];
        // Input point 15: x(14)
        a_in7 = in_r[in_strides[14]];
        // Input point 17: x(16)
        a_in8 = in_r[in_strides[16]];
        // Input point 19: x(18)
        a_in9 = in_r[in_strides[18]];
        // Input point 21: x(20)
        a_in10 = in_r[in_strides[20]];
        // Input point 23: x(22)
        a_in11 = in_r[in_strides[22]];
        // Input point 25: x(24)
        a_in12 = in_r[in_strides[24]];
        // Input point 27: x(26)
        a_in13 = in_r[in_strides[26]];
        // Input point 29: x(28)
        a_in14 = in_r[in_strides[28]];
        // Input point 31: x(30)
        a_in15 = in_r[in_strides[30]];

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
        out_cp[out_strides[15]] = a_s16 - a_s22;

        a_t12 = a_s5 + a_s13;
        a_t16 = a_s5 - a_s13;
        a_t14 = a_s6 + a_s14;
        a_s27 = a_s6 - a_s14;
        a_t13 = a_s7 + a_s15;
        a_s29 = a_s7 - a_s15;
        // Output point 17: X(16)
        out_cp[out_strides[16]] = a_t13 - a_s21;

        a_t15 = a_s18 + a_t14;
        a_s31 = a_s18 - a_t14;
        a_s32 = a_s16 + a_s22;
        a_t17 = a_s21 + a_t13;
        // Output point 1: X(0)
        *out_r = a_t15 + a_s32;
        // Output point 32: X(31)
        out_r[out_strides[31]] = a_s32 - a_t15;

        a_t0 = CRTM_16_1 * a_s19;
        a_t1 = CRTM_16_1 * a_s20;
        a_t2 = CRTM_16_1 * a_s27;
        a_t3 = CRTM_16_1 * a_s29;
        a_t4 = CRTM_16_2 * a_s19;
        a_t5 = CRTM_16_2 * a_s20;
        a_t6 = CRTM_16_2 * a_s27;
        a_t7 = CRTM_16_2 * a_s29;
        a_t8 = CRTM_16_3 * a_s23;
        a_t9 = CRTM_16_3 * a_t16;
        a_t10 = CRTM_16_3 * a_s31;
        // Output point 8: X(7)
        out_cp[out_strides[7]] = a_s17 + a_t10;
        // Output point 24: X(23)
        out_cp[out_strides[23]] = a_s17 - a_t10;

        a_t11 = CRTM_16_3 * a_t17;
        // Output point 9: X(8)
        out_cp[out_strides[8]] = -(a_t12 + a_t11);
        // Output point 25: X(24)
        out_cp[out_strides[24]] = a_t12 - a_t11;

        a_s46 = a_t0 + a_t7;
        a_t18 = a_t9 + a_s1;
        // Output point 4: X(3)
        out_cp[out_strides[3]] = a_s46 + a_t18;
        // Output point 28: X(27)
        out_cp[out_strides[27]] = a_t18 - a_s46;

        a_t20 = a_t2 + a_t5;
        a_t19 = a_t8 + a_s9;
        // Output point 5: X(4)
        out_cp[out_strides[4]] = -(a_t20 + a_t19);
        // Output point 29: X(28)
        out_cp[out_strides[28]] = a_t19 - a_t20;

        a_s49 = a_t3 - a_t4;
        a_s50 = a_s1 - a_t9;
        // Output point 12: X(11)
        out_cp[out_strides[11]] = a_s50 - a_s49;
        // Output point 20: X(19)
        out_cp[out_strides[19]] = a_s49 + a_s50;

        a_s51 = a_t6 - a_t1;
        a_s52 = a_s9 - a_t8;
        // Output point 13: X(12)
        out_cp[out_strides[12]] = a_s51 + a_s52;
        // Output point 21: X(20)
        out_cp[out_strides[20]] = a_s51 - a_s52;

        /* Shifted DFT */
        FFTZ_FLOAT b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7,
            b_in8, b_in9, b_in10, b_in11, b_in12, b_in13, b_in14, b_in15;
        FFTZ_FLOAT b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_t0, b_s7, b_t1, b_t2,
            b_s10, b_s11, b_s12, b_t3, b_s14, b_t4, b_s16, b_t5, b_t6, b_s19,
            b_t7, b_s21, b_t8, b_s23, b_s24, b_t9, b_s26, b_t10, b_t11, b_s29,
            b_s30, b_s31, b_t12, b_t13, b_s34, b_s35, b_s36, b_t18, b_t19,
            b_s39, b_s40, b_t20, b_t14, b_s43, b_s44, b_t15, b_s46, b_s47,
            b_s48, b_s49, b_s50, b_s51, b_t16, b_t17, b_s53, b_s54;

        // Input point 2: x(1)
        b_in0 = in_r[in_strides[1]];
        // Input point 4: x(3)
        b_in1 = in_r[in_strides[3]];
        // Input point 6: x(5)
        b_in2 = in_r[in_strides[5]];
        // Input point 8: x(7)
        b_in3 = in_r[in_strides[7]];
        // Input point 10: x(9)
        b_in4 = in_r[in_strides[9]];
        // Input point 12: x(11)
        b_in5 = in_r[in_strides[11]];
        // Input point 14: x(13)
        b_in6 = in_r[in_strides[13]];
        // Input point 16: x(15)
        b_in7 = in_r[in_strides[15]];
        // Input point 18: x(17)
        b_in8 = in_r[in_strides[17]];
        // Input point 20: x(19)
        b_in9 = in_r[in_strides[19]];
        // Input point 22: x(21)
        b_in10 = in_r[in_strides[21]];
        // Input point 24: x(23)
        b_in11 = in_r[in_strides[23]];
        // Input point 26: x(25)
        b_in12 = in_r[in_strides[25]];
        // Input point 28: x(27)
        b_in13 = in_r[in_strides[27]];
        // Input point 30: x(29)
        b_in14 = in_r[in_strides[29]];
        // Input point 32: x(31)
        b_in15 = in_r[in_strides[31]];

        b_s0 = b_in3 - b_in11;
        b_s1 = b_in3 + b_in11;
        b_s2 = b_in4 - b_in12;
        b_s3 = b_in4 + b_in12;
        b_s4 = b_in5 - b_in13;
        b_s5 = b_in5 + b_in13;

        b_t0 = CRTM_16_3 * b_s2;
        b_s7 = b_t0 + b_in0;
        b_t1 = (CRTM_16_1 * b_in2) - (CRTM_16_2 * b_in10);
        b_t2 = (CRTM_16_2 * b_in6) - (CRTM_16_1 * b_in14);
        b_s10 = b_t1 + b_t2;
        b_s11 = b_s7 - b_s10;
        b_s12 = b_s7 + b_s10;

        b_t3 = CRTM_16_3 * b_s4;
        b_s14 = b_t3 + b_in1;
        b_t4 = CRTM_16_3 * b_s5;
        b_s16 = b_t4 + b_in9;

        b_t5 = (CRTM_16_6 * b_s14) - (CRTM_16_7 * b_s16);

        b_t6 = CRTM_16_3 * b_s1;
        b_s19 = b_t6 + b_in7;
        b_t7 = CRTM_16_3 * b_s0;
        b_s21 = b_t7 - b_in15;

        b_t8 = (CRTM_16_7 * b_s19) + (CRTM_16_6 * b_s21);

        b_s23 = b_t5 + b_t8;
        b_s24 = b_t8 - b_t5;

        // Output point 2: X(1)
        out_cp[out_strides[1]] = b_s12 + b_s23;
        // Output point 30: X(29)
        out_cp[out_strides[29]] = b_s12 - b_s23;

        b_t9 = CRTM_16_3 * b_s3;
        b_s26 = b_t9 + b_in8;

        b_t10 = (CRTM_16_2 * b_in2) + (CRTM_16_1 * b_in10);
        b_t11 = (CRTM_16_1 * b_in6) + (CRTM_16_2 * b_in14);
        b_s29 = b_t10 + b_t11;
        b_s30 = b_s26 - b_s29;
        b_s31 = b_s26 + b_s29;
        // Output point 15: X(14)
        out_cp[out_strides[14]] = b_s30 + b_s24;
        // Output point 19: X(18)
        out_cp[out_strides[18]] = b_s24 - b_s30;

        b_t12 = (CRTM_16_6 * b_s16) + (CRTM_16_7 * b_s14);
        b_t13 = (CRTM_16_7 * b_s21) - (CRTM_16_6 * b_s19);

        b_s34 = b_t13 + b_t12;
        b_s35 = b_t13 - b_t12;

        // Output point 3: X(2)
        out_cp[out_strides[2]] = b_s35 - b_s31;
        // Output point 31: X(30)
        out_cp[out_strides[30]] = b_s35 + b_s31;

        // Output point 18: X(17)
        out_cp[out_strides[17]] = b_s11 - b_s34;
        // Output point 14: X(13)
        out_cp[out_strides[13]] = b_s11 + b_s34;

        b_s36 = b_in0 - b_t0;

        b_t18 = b_t10 - b_t11;
        b_t19 = b_s36 + b_t18;
        b_s39 = b_s36 - b_t18;

        b_s40 = b_in9 - b_t4;
        b_t20 = b_in1 - b_t3;

        b_t14 = (CRTM_16_4 * b_s40) + (CRTM_16_5 * b_t20);

        b_s43 = b_in7 - b_t6;
        b_s44 = b_in15 + b_t7;

        b_t15 = (CRTM_16_4 * b_s43) + (CRTM_16_5 * b_s44);

        b_s46 = b_t14 - b_t15;
        b_s47 = b_t14 + b_t15;

        // Output point 6: X(5)
        out_cp[out_strides[5]] = b_t19 + b_s46;
        // Output point 26: X(25)
        out_cp[out_strides[25]] = b_t19 - b_s46;

        b_s48 = b_in8 - b_t9;
        b_s49 = b_t2 - b_t1;
        b_s50 = b_s49 - b_s48;
        b_s51 = b_s49 + b_s48;
        // Output point 11: X(10)
        out_cp[out_strides[10]] = b_s50 - b_s47;
        // Output point 23: X(22)
        out_cp[out_strides[22]] = -(b_s50 + b_s47);

        b_t16 = (CRTM_16_5 * b_s43) - (CRTM_16_4 * b_s44);
        b_t17 = (CRTM_16_5 * b_s40) - (CRTM_16_4 * b_t20);

        b_s53 = b_t16 - b_t17;
        b_s54 = b_t17 + b_t16;

        // Output point 22: X(21)
        out_cp[out_strides[21]] = b_s39 - b_s53;
        // Output point 10: X(9)
        out_cp[out_strides[9]] = b_s39 + b_s53;

        // Output point 27: X(26)
        out_cp[out_strides[26]] = b_s54 - b_s51;
        // Output point 7: X(6)
        out_cp[out_strides[6]] = b_s54 + b_s51;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft16avx128_fp32_bwd(FFTZ_VOID *in_real,
                                             FFTZ_VOID *in_complex,
                                             FFTZ_VOID *out_real,
                                             FFTZ_VOID *out_complex,
                                             FFTZ_INTP n,
                                             aoclfftz_strides_t *strides,
                                             FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_16_1 =
        1.847759065022573256256366378793576573644833252f;
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
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_UINT8 is_contiguous_out = (v_out_stride == 1);
    FFTZ_INTP v_in_dc_nyq_stride = strides->v_in_sym_stride;
    FFTZ_UINT8 is_contiguous_in_dc_nyq = (v_in_dc_nyq_stride == 1);

    FFTZ_INTP cnt;
    FFTZ_FLOAT *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_128_S;

    __m128 v_CRTM_16_1 = _mm_broadcast_ss(&CRTM_16_1);
    __m128 v_CRTM_16_2 = _mm_broadcast_ss(&CRTM_16_2);
    __m128 v_CRTM_16_3 = _mm_broadcast_ss(&CRTM_16_3);
    __m128 v_CRTM_16_4 = _mm_broadcast_ss(&CRTM_16_4);
    __m128 v_CRTM_16_5 = _mm_broadcast_ss(&CRTM_16_5);
    __m128 v_CRTM_16_6 = _mm_broadcast_ss(&CRTM_16_6);
    __m128 v_CRTM_16_7 = _mm_broadcast_ss(&CRTM_16_7);
    __m128 v_CRTM_16_8 = _mm_broadcast_ss(&CRTM_16_8);
    __m128 v_CRTM_16_9 = _mm_broadcast_ss(&CRTM_16_9);

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
               av_in8, av_in9, av_in10, av_in11, av_in12, av_in13, av_in14,
               av_in15;
        __m128 av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8, av_s9,
               av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16, av_s17,
               av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24, av_s25,
               av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32, av_s33,
               av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40, av_s41,
               av_s42;
        __m128 av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8, av_t9,
               av_t10, av_t11, av_t12, av_t13, av_t14, av_t15, av_t16, av_t17,
               av_t18;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21,
               v_out22, v_out23, v_out24, v_out25, v_out26, v_out27, v_out28,
               v_out29, v_out30, v_out31;

        curr_in = in_r;
        curr_out = out_r;

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_dc_nyq_stride, av_in0, is_contiguous_in_dc_nyq);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in_cp + in_strides[3];
        LDRI_2x128_S(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in_cp + in_strides[7];
        LDRI_2x128_S(curr_in, v_in_stride, av_in3, av_in4);
        // Input point 12: x(11) & Input point 13: x(12)
        curr_in = in_cp + in_strides[11];
        LDRI_2x128_S(curr_in, v_in_stride, av_in5, av_in6);
        // Input point 16: x(15) & Input point 17: x(16)
        curr_in = in_cp + in_strides[15];
        LDRI_2x128_S(curr_in, v_in_stride, av_in7, av_in8);
        // Input point 20: x(19) & Input point 21: x(20)
        curr_in = in_cp + in_strides[19];
        LDRI_2x128_S(curr_in, v_in_stride, av_in9, av_in10);
        // Input point 24: x(23) & Input point 25: x(24)
        curr_in = in_cp + in_strides[23];
        LDRI_2x128_S(curr_in, v_in_stride, av_in11, av_in12);
        // Input point 28: x(27) & Input point 29: x(28)
        curr_in = in_cp + in_strides[27];
        LDRI_2x128_S(curr_in, v_in_stride, av_in13, av_in14);
        // Input point 32: x(31)
        curr_in = in_r + in_strides[31];
        LDR_128_S(curr_in, v_in_dc_nyq_stride, av_in15, is_contiguous_in_dc_nyq);

        av_s1 = _mm_add_ps(av_in0, av_in15);
        av_s2 = _mm_sub_ps(av_in0, av_in15);
        av_s3 = _mm_add_ps(av_in1, av_in13);
        av_s4 = _mm_sub_ps(av_in1, av_in13);
        av_s5 = _mm_add_ps(av_in2, av_in14);
        av_s6 = _mm_sub_ps(av_in2, av_in14);
        av_s7 = _mm_add_ps(av_in3, av_in11);
        av_s8 = _mm_sub_ps(av_in3, av_in11);
        av_s9 = _mm_add_ps(av_in4, av_in12);
        av_s10 = _mm_sub_ps(av_in4, av_in12);
        av_s11 = _mm_add_ps(av_in5, av_in9);
        av_s12 = _mm_sub_ps(av_in5, av_in9);
        av_s13 = _mm_add_ps(av_in6, av_in10);
        av_s14 = _mm_sub_ps(av_in6, av_in10);

        av_s15 = _mm_add_ps(av_s3, av_s11);
        av_s16 = _mm_sub_ps(av_s3, av_s11);
        av_s17 = _mm_add_ps(av_s4, av_s13);
        av_s18 = _mm_sub_ps(av_s4, av_s13);
        av_s19 = _mm_add_ps(av_s5, av_s12);
        av_s20 = _mm_sub_ps(av_s5, av_s12);
        av_s21 = _mm_add_ps(av_s6, av_s14);
        av_s22 = _mm_sub_ps(av_s6, av_s14);
        av_s23 = _mm_add_ps(av_s8, av_s9);
        av_s24 = _mm_sub_ps(av_s8, av_s9);

        av_t1 = _mm_mul_ps(av_in7, v_CRTM_16_4);
        av_t4 = _mm_mul_ps(av_s15, v_CRTM_16_4);
        av_t3 = _mm_mul_ps(av_s7, v_CRTM_16_4);
        av_s25 = _mm_add_ps(av_s1, av_t1);
        av_s27 = _mm_add_ps(av_t3, av_s25);
        // output point 1: x(0)
        v_out0 = _mm_add_ps(av_s27, av_t4);
        STR_128_S(curr_out, v_out_stride, v_out0, is_contiguous_out);
        // output point 17: x(16)
        curr_out = out_r + out_strides[16];
        v_out16 = _mm_sub_ps(av_s27, av_t4);
        STR_128_S(curr_out, v_out_stride, v_out16, is_contiguous_out);

        av_t5 = _mm_mul_ps(av_s22, v_CRTM_16_4);
        av_s28 = _mm_sub_ps(av_s25, av_t3);
        // output point 9: x(8)
        curr_out = out_r + out_strides[8];
        v_out8 = _mm_sub_ps(av_s28, av_t5);
        STR_128_S(curr_out, v_out_stride, v_out8, is_contiguous_out);
        // output point 25: x(24)
        curr_out = out_r + out_strides[24];
        v_out24 = _mm_add_ps(av_s28, av_t5);
        STR_128_S(curr_out, v_out_stride, v_out24, is_contiguous_out);

        av_t2 = _mm_mul_ps(av_in8, v_CRTM_16_4);
        av_t6 = _mm_mul_ps(av_s24, v_CRTM_16_3);
        av_t7 = _mm_mul_ps(av_s18, v_CRTM_16_1);
        av_t8 = _mm_mul_ps(av_s20, v_CRTM_16_2);
        av_s30 = _mm_sub_ps(av_s2, av_t2);
        av_s33 = _mm_add_ps(av_t6, av_s30);
        av_s34 = _mm_sub_ps(av_t7, av_t8);
        // output point 3: x(2)
        curr_out = out_r + out_strides[2];
        v_out2 = _mm_add_ps(av_s33, av_s34);
        STR_128_S(curr_out, v_out_stride, v_out2, is_contiguous_out);
        // output point 19: x(18)
        curr_out = out_r + out_strides[18];
        v_out18 = _mm_sub_ps(av_s33, av_s34);
        STR_128_S(curr_out, v_out_stride, v_out18, is_contiguous_out);

        av_s26 = _mm_sub_ps(av_s1, av_t1);
        av_s29 = _mm_add_ps(av_s2, av_t2);
        av_t9 = _mm_mul_ps(av_s23, v_CRTM_16_3);
        av_t10 = _mm_mul_ps(av_s19, v_CRTM_16_1);
        av_t11 = _mm_mul_ps(av_s17, v_CRTM_16_2);
        av_s35 = _mm_sub_ps(av_s29, av_t9);
        av_s36 = _mm_sub_ps(av_t11, av_t10);
        // output point 7: x(6)
        curr_out = out_r + out_strides[6];
        v_out6 = _mm_add_ps(av_s35, av_s36);
        STR_128_S(curr_out, v_out_stride, v_out6, is_contiguous_out);
        // output point 23: x(22)
        curr_out = out_r + out_strides[22];
        v_out22 = _mm_sub_ps(av_s35, av_s36);
        STR_128_S(curr_out, v_out_stride, v_out22, is_contiguous_out);

        av_t14 = _mm_mul_ps(av_s17, v_CRTM_16_1);
        av_t15 = _mm_mul_ps(av_s19, v_CRTM_16_2);
        av_s39 = _mm_add_ps(av_s29, av_t9);
        av_s40 = _mm_add_ps(av_t14, av_t15);
        // output point 15: x(14)
        curr_out = out_r + out_strides[14];
        v_out14 = _mm_sub_ps(av_s39, av_s40);
        STR_128_S(curr_out, v_out_stride, v_out14, is_contiguous_out);
        // output point 31: x(30)
        curr_out = out_r + out_strides[30];
        v_out30 = _mm_add_ps(av_s39, av_s40);
        STR_128_S(curr_out, v_out_stride, v_out30, is_contiguous_out);

        av_t12 = _mm_mul_ps(av_s20, v_CRTM_16_1);
        av_t13 = _mm_mul_ps(av_s18, v_CRTM_16_2);
        av_s37 = _mm_sub_ps(av_s30, av_t6);
        av_s38 = _mm_add_ps(av_t12, av_t13);
        // output point 11: x(10)
        curr_out = out_r + out_strides[10];
        v_out10 = _mm_sub_ps(av_s37, av_s38);
        STR_128_S(curr_out, v_out_stride, v_out10, is_contiguous_out);
        // output point 27: x(26)
        curr_out = out_r + out_strides[26];
        v_out26 = _mm_add_ps(av_s37, av_s38);
        STR_128_S(curr_out, v_out_stride, v_out26, is_contiguous_out);

        av_s32 = _mm_sub_ps(av_s16, av_s21);
        av_t16 = _mm_mul_ps(av_s32, v_CRTM_16_3);
        av_t18 = _mm_mul_ps(av_s10, v_CRTM_16_4);
        av_s41 = _mm_sub_ps(av_s26, av_t18);
        // output point 5: x(4)
        curr_out = out_r + out_strides[4];
        v_out4 = _mm_add_ps(av_s41, av_t16);
        STR_128_S(curr_out, v_out_stride, v_out4, is_contiguous_out);
        // output point 21: x(20)
        curr_out = out_r + out_strides[20];
        v_out20 = _mm_sub_ps(av_s41, av_t16);
        STR_128_S(curr_out, v_out_stride, v_out20, is_contiguous_out);

        av_s31 = _mm_add_ps(av_s16, av_s21);
        av_t17 = _mm_mul_ps(av_s31, v_CRTM_16_3);
        av_s42 = _mm_add_ps(av_s26, av_t18);
        // output point 13: x(12)
        curr_out = out_r + out_strides[12];
        v_out12 = _mm_sub_ps(av_s42, av_t17);
        STR_128_S(curr_out, v_out_stride, v_out12, is_contiguous_out);
        // output point 29: x(28)
        curr_out = out_r + out_strides[28];
        v_out28 = _mm_add_ps(av_s42, av_t17);
        STR_128_S(curr_out, v_out_stride, v_out28, is_contiguous_out);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8, bv_in9, bv_in10, bv_in11, bv_in12, bv_in13, bv_in14,
               bv_in15;
        __m128 bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_t0, bv_s7, bv_t1, bv_t2,
               bv_s10, bv_s11, bv_s12, bv_t3, bv_s14, bv_t4, bv_s16, bv_t5,
               bv_t6, bv_s19, bv_t7, bv_s21, bv_t8, bv_s23, bv_s24, bv_t9,
               bv_s26, bv_t10, bv_t11, bv_s29, bv_s30, bv_s31, bv_t12, bv_t13,
               bv_s34, bv_s35, bv_s36, bv_t18, bv_t19, bv_s39, bv_s40, bv_t20,
               bv_t14, bv_s43, bv_s44, bv_t15, bv_s46, bv_s47, bv_s48, bv_s49,
               bv_s50;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in_cp + in_strides[1];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in_cp + in_strides[5];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in_cp + in_strides[9];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in4, bv_in5);
        // Input point 14: x(13) & Input point 15: x(14)
        curr_in = in_cp + in_strides[13];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in6, bv_in7);
        // Input point 18: x(17) & Input point 19: x(18)
        curr_in = in_cp + in_strides[17];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in8, bv_in9);
        // Input point 22: x(21) & Input point 23: x(22)
        curr_in = in_cp + in_strides[21];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in10, bv_in11);
        // Input point 26: x(25) & Input point 27: x(26)
        curr_in = in_cp + in_strides[25];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in12, bv_in13);
        // Input point 30: x(29) & Input point 31: x(30)
        curr_in = in_cp + in_strides[29];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in14, bv_in15);

        bv_s1 = _mm_add_ps(bv_in14, bv_in0);
        bv_s2 = _mm_sub_ps(bv_in14, bv_in0);
        bv_s3 = _mm_add_ps(bv_in15, bv_in1);
        bv_s4 = _mm_sub_ps(bv_in15, bv_in1);
        bv_s5 = _mm_add_ps(bv_in12, bv_in2);
        bv_t0 = _mm_sub_ps(bv_in12, bv_in2);
        bv_s7 = _mm_add_ps(bv_in13, bv_in3);
        bv_t1 = _mm_sub_ps(bv_in13, bv_in3);
        bv_t2 = _mm_add_ps(bv_in10, bv_in4);
        bv_s10 = _mm_sub_ps(bv_in10, bv_in4);
        bv_s11 = _mm_add_ps(bv_in11, bv_in5);
        bv_s12 = _mm_sub_ps(bv_in11, bv_in5);
        bv_t3 = _mm_add_ps(bv_in8, bv_in6);
        bv_s14 = _mm_sub_ps(bv_in8, bv_in6);
        bv_t4 = _mm_add_ps(bv_in9, bv_in7);
        bv_s16 = _mm_sub_ps(bv_in9, bv_in7);

        bv_t5 = _mm_sub_ps(bv_s1, bv_t3);
        bv_t6 = _mm_add_ps(bv_s12, bv_t1);
        bv_s19 = _mm_add_ps(bv_t5, bv_t6);
        bv_t7 = _mm_sub_ps(bv_t6, bv_t5);

        bv_s21 = _mm_add_ps(bv_s4, bv_s16);
        bv_t8 = _mm_sub_ps(bv_s5, bv_t2);
        bv_s23 = _mm_add_ps(bv_s21, bv_t8);
        bv_s24 = _mm_sub_ps(bv_s21, bv_t8);

        // Output point 6: X(5)
        v_out5 = _mm_add_ps(_mm_mul_ps(v_CRTM_16_1, bv_s19),
                               _mm_mul_ps(v_CRTM_16_2, bv_s23));
        curr_out = out_r + out_strides[5];
        STR_128_S(curr_out, v_out_stride, v_out5, is_contiguous_out);
        // Output point 22: X(21)
        v_out21 = _mm_sub_ps(_mm_mul_ps(v_CRTM_16_1, bv_s23),
                                _mm_mul_ps(v_CRTM_16_2, bv_s19));
        curr_out = out_r + out_strides[21];
        STR_128_S(curr_out, v_out_stride, v_out21, is_contiguous_out);
        // Output point 30: X(29)
        v_out29 = _mm_add_ps(_mm_mul_ps(v_CRTM_16_1, bv_t7),
                                _mm_mul_ps(v_CRTM_16_2, bv_s24));
        curr_out = out_r + out_strides[29];
        STR_128_S(curr_out, v_out_stride, v_out29, is_contiguous_out);
        // Output point 14: X(13)
        v_out13 = _mm_sub_ps(_mm_mul_ps(v_CRTM_16_1, bv_s24),
                                _mm_mul_ps(v_CRTM_16_2, bv_t7));
        curr_out = out_r + out_strides[13];
        STR_128_S(curr_out, v_out_stride, v_out13, is_contiguous_out);

        bv_t9 = _mm_add_ps(bv_s1, bv_t3);
        bv_s26 = _mm_add_ps(bv_s5, bv_t2);
        // Output point 2: X(1)
        v_out1 = _mm_mul_ps(v_CRTM_16_4, _mm_add_ps(bv_t9, bv_s26));
        curr_out = out_r + out_strides[1];
        STR_128_S(curr_out, v_out_stride, v_out1, is_contiguous_out);

        bv_t10 = _mm_sub_ps(bv_s4, bv_s16);
        bv_t11 = _mm_sub_ps(bv_s12, bv_t1);
        // Output point 18: X(17)
        v_out17 = _mm_mul_ps(v_CRTM_16_4, _mm_add_ps(bv_t10, bv_t11));
        curr_out = out_r + out_strides[17];
        STR_128_S(curr_out, v_out_stride, v_out17, is_contiguous_out);

        bv_s29 = _mm_sub_ps(bv_t9, bv_s26);
        bv_s30 = _mm_sub_ps(bv_t10, bv_t11);

        // Output point 10: X(9)
        v_out9 = _mm_mul_ps(v_CRTM_16_3, _mm_add_ps(bv_s29, bv_s30));
        curr_out = out_r + out_strides[9];
        STR_128_S(curr_out, v_out_stride, v_out9, is_contiguous_out);
        // Output point 26: X(25)
        v_out25 = _mm_mul_ps(v_CRTM_16_3, _mm_sub_ps(bv_s30, bv_s29));
        curr_out = out_r + out_strides[25];
        STR_128_S(curr_out, v_out_stride, v_out25, is_contiguous_out);

        bv_s31 = _mm_add_ps(bv_s2, bv_t4);
        bv_t12 = NEGATE_128_S(_mm_add_ps(bv_s3, bv_s14));

        bv_t13 = _mm_sub_ps(bv_s14, bv_s3);
        bv_s34 = _mm_sub_ps(bv_t4, bv_s2);

        bv_s35 = _mm_add_ps(bv_s11, bv_t0);
        bv_s36 = _mm_add_ps(bv_s10, bv_s7);

        bv_t18 = _mm_mul_ps(v_CRTM_16_5, _mm_add_ps(bv_s35, bv_s36));
        bv_t19 = _mm_mul_ps(v_CRTM_16_5, _mm_sub_ps(bv_s36, bv_s35));

        bv_s43 = _mm_add_ps(bv_t18, bv_s31);
        bv_s44 = _mm_add_ps(bv_t12, bv_t19);
        bv_t15 = _mm_sub_ps(bv_s31, bv_t18);
        bv_s46 = _mm_sub_ps(bv_t12, bv_t19);

        // Output point 4: X(3)
        v_out3 = _mm_sub_ps(_mm_mul_ps(v_CRTM_16_7, bv_s44),
                               _mm_mul_ps(v_CRTM_16_6, bv_s43));
        curr_out = out_r + out_strides[3];
        STR_128_S(curr_out, v_out_stride, v_out3, is_contiguous_out);
        // Output point 12: X(11)
        v_out11 = _mm_sub_ps(_mm_mul_ps(v_CRTM_16_9, bv_s46),
                                _mm_mul_ps(v_CRTM_16_8, bv_t15));
        curr_out = out_r + out_strides[11];
        STR_128_S(curr_out, v_out_stride, v_out11, is_contiguous_out);
        // Output point 28: X(27)
        v_out27 = _mm_add_ps(_mm_mul_ps(v_CRTM_16_9, bv_t15),
                                _mm_mul_ps(v_CRTM_16_8, bv_s46));
        curr_out = out_r + out_strides[27];
        STR_128_S(curr_out, v_out_stride, v_out27, is_contiguous_out);
        // Output point 20: X(19)
        v_out19 = _mm_add_ps(_mm_mul_ps(v_CRTM_16_6, bv_s44),
                                _mm_mul_ps(v_CRTM_16_7, bv_s43));
        curr_out = out_r + out_strides[19];
        STR_128_S(curr_out, v_out_stride, v_out19, is_contiguous_out);

        bv_s39 = _mm_sub_ps(bv_t0, bv_s11);
        bv_s40 = _mm_sub_ps(bv_s10, bv_s7);

        bv_t20 = _mm_mul_ps(v_CRTM_16_5, _mm_sub_ps(bv_s39, bv_s40));
        bv_t14 = _mm_mul_ps(v_CRTM_16_5, _mm_add_ps(bv_s39, bv_s40));

        bv_s47 = _mm_add_ps(bv_t13, bv_t20);
        bv_s48 = _mm_add_ps(bv_s34, bv_t14);
        bv_s49 = _mm_sub_ps(bv_t13, bv_t20);
        bv_s50 = _mm_sub_ps(bv_s34, bv_t14);

        // Output point 16: X(15)
        v_out15 = _mm_add_ps(_mm_mul_ps(v_CRTM_16_6, bv_s47),
                                _mm_mul_ps(v_CRTM_16_7, bv_s50));
        curr_out = out_r + out_strides[15];
        STR_128_S(curr_out, v_out_stride, v_out15, is_contiguous_out);
        // Output point 32: X(31)
        v_out31 = _mm_sub_ps(_mm_mul_ps(v_CRTM_16_7, bv_s47),
                                _mm_mul_ps(v_CRTM_16_6, bv_s50));
        curr_out = out_r + out_strides[31];
        STR_128_S(curr_out, v_out_stride, v_out31, is_contiguous_out);
        // Output point 8: X(7)
        v_out7 = _mm_add_ps(_mm_mul_ps(v_CRTM_16_9, bv_s48),
                               _mm_mul_ps(v_CRTM_16_8, bv_s49));
        curr_out = out_r + out_strides[7];
        STR_128_S(curr_out, v_out_stride, v_out7, is_contiguous_out);
        // Output point 24: X(23)
        v_out23 = _mm_sub_ps(_mm_mul_ps(v_CRTM_16_9, bv_s49),
                                _mm_mul_ps(v_CRTM_16_8, bv_s48));
        curr_out = out_r + out_strides[23];
        STR_128_S(curr_out, v_out_stride, v_out23, is_contiguous_out);

        in_cp += v_in_stride * NUM_SETS_REAL_128_S;
        in_r += v_in_dc_nyq_stride * NUM_SETS_REAL_128_S;
        out_r += v_out_stride * NUM_SETS_REAL_128_S;
    }
    // tail cases
    if (n & 2)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
               av_in8, av_in9, av_in10, av_in11, av_in12, av_in13, av_in14,
               av_in15;
        __m128 av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8, av_s9,
               av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16, av_s17,
               av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24, av_s25,
               av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32, av_s33,
               av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40, av_s41,
               av_s42;
        __m128 av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8, av_t9,
               av_t10, av_t11, av_t12, av_t13, av_t14, av_t15, av_t16, av_t17,
               av_t18;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21,
               v_out22, v_out23, v_out24, v_out25, v_out26, v_out27, v_out28,
               v_out29, v_out30, v_out31;

        curr_in = in_r;
        curr_out = out_r;

        // Input point 1: x(0)
        LDHR_128_S(curr_in, v_in_dc_nyq_stride, av_in0);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in_cp + in_strides[3];
        LDHRI_2x128_S(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in_cp + in_strides[7];
        LDHRI_2x128_S(curr_in, v_in_stride, av_in3, av_in4);
        // Input point 12: x(11) & Input point 13: x(12)
        curr_in = in_cp + in_strides[11];
        LDHRI_2x128_S(curr_in, v_in_stride, av_in5, av_in6);
        // Input point 16: x(15) & Input point 17: x(16)
        curr_in = in_cp + in_strides[15];
        LDHRI_2x128_S(curr_in, v_in_stride, av_in7, av_in8);
        // Input point 20: x(19) & Input point 21: x(20)
        curr_in = in_cp + in_strides[19];
        LDHRI_2x128_S(curr_in, v_in_stride, av_in9, av_in10);
        // Input point 24: x(23) & Input point 25: x(24)
        curr_in = in_cp + in_strides[23];
        LDHRI_2x128_S(curr_in, v_in_stride, av_in11, av_in12);
        // Input point 28: x(27) & Input point 29: x(28)
        curr_in = in_cp + in_strides[27];
        LDHRI_2x128_S(curr_in, v_in_stride, av_in13, av_in14);
        // Input point 32: x(31)
        curr_in = in_r + in_strides[31];
        LDHR_128_S(curr_in, v_in_dc_nyq_stride, av_in15);

        av_s1 = _mm_add_ps(av_in0, av_in15);
        av_s2 = _mm_sub_ps(av_in0, av_in15);
        av_s3 = _mm_add_ps(av_in1, av_in13);
        av_s4 = _mm_sub_ps(av_in1, av_in13);
        av_s5 = _mm_add_ps(av_in2, av_in14);
        av_s6 = _mm_sub_ps(av_in2, av_in14);
        av_s7 = _mm_add_ps(av_in3, av_in11);
        av_s8 = _mm_sub_ps(av_in3, av_in11);
        av_s9 = _mm_add_ps(av_in4, av_in12);
        av_s10 = _mm_sub_ps(av_in4, av_in12);
        av_s11 = _mm_add_ps(av_in5, av_in9);
        av_s12 = _mm_sub_ps(av_in5, av_in9);
        av_s13 = _mm_add_ps(av_in6, av_in10);
        av_s14 = _mm_sub_ps(av_in6, av_in10);

        av_s15 = _mm_add_ps(av_s3, av_s11);
        av_s16 = _mm_sub_ps(av_s3, av_s11);
        av_s17 = _mm_add_ps(av_s4, av_s13);
        av_s18 = _mm_sub_ps(av_s4, av_s13);
        av_s19 = _mm_add_ps(av_s5, av_s12);
        av_s20 = _mm_sub_ps(av_s5, av_s12);
        av_s21 = _mm_add_ps(av_s6, av_s14);
        av_s22 = _mm_sub_ps(av_s6, av_s14);
        av_s23 = _mm_add_ps(av_s8, av_s9);
        av_s24 = _mm_sub_ps(av_s8, av_s9);

        av_t1 = _mm_mul_ps(av_in7, v_CRTM_16_4);
        av_t4 = _mm_mul_ps(av_s15, v_CRTM_16_4);
        av_t3 = _mm_mul_ps(av_s7, v_CRTM_16_4);
        av_s25 = _mm_add_ps(av_s1, av_t1);
        av_s27 = _mm_add_ps(av_t3, av_s25);
        // output point 1: x(0)
        v_out0 = _mm_add_ps(av_s27, av_t4);
        STHR_128_S(curr_out, v_out_stride, v_out0);
        // output point 17: x(16)
        curr_out = out_r + out_strides[16];
        v_out16 = _mm_sub_ps(av_s27, av_t4);
        STHR_128_S(curr_out, v_out_stride, v_out16);

        av_t5 = _mm_mul_ps(av_s22, v_CRTM_16_4);
        av_s28 = _mm_sub_ps(av_s25, av_t3);
        // output point 9: x(8)
        curr_out = out_r + out_strides[8];
        v_out8 = _mm_sub_ps(av_s28, av_t5);
        STHR_128_S(curr_out, v_out_stride, v_out8);
        // output point 25: x(24)
        curr_out = out_r + out_strides[24];
        v_out24 = _mm_add_ps(av_s28, av_t5);
        STHR_128_S(curr_out, v_out_stride, v_out24);

        av_t2 = _mm_mul_ps(av_in8, v_CRTM_16_4);
        av_t6 = _mm_mul_ps(av_s24, v_CRTM_16_3);
        av_t7 = _mm_mul_ps(av_s18, v_CRTM_16_1);
        av_t8 = _mm_mul_ps(av_s20, v_CRTM_16_2);
        av_s30 = _mm_sub_ps(av_s2, av_t2);
        av_s33 = _mm_add_ps(av_t6, av_s30);
        av_s34 = _mm_sub_ps(av_t7, av_t8);
        // output point 3: x(2)
        curr_out = out_r + out_strides[2];
        v_out2 = _mm_add_ps(av_s33, av_s34);
        STHR_128_S(curr_out, v_out_stride, v_out2);
        // output point 19: x(18)
        curr_out = out_r + out_strides[18];
        v_out18 = _mm_sub_ps(av_s33, av_s34);
        STHR_128_S(curr_out, v_out_stride, v_out18);

        av_s26 = _mm_sub_ps(av_s1, av_t1);
        av_s29 = _mm_add_ps(av_s2, av_t2);
        av_t9 = _mm_mul_ps(av_s23, v_CRTM_16_3);
        av_t10 = _mm_mul_ps(av_s19, v_CRTM_16_1);
        av_t11 = _mm_mul_ps(av_s17, v_CRTM_16_2);
        av_s35 = _mm_sub_ps(av_s29, av_t9);
        av_s36 = _mm_sub_ps(av_t11, av_t10);
        // output point 7: x(6)
        curr_out = out_r + out_strides[6];
        v_out6 = _mm_add_ps(av_s35, av_s36);
        STHR_128_S(curr_out, v_out_stride, v_out6);
        // output point 23: x(22)
        curr_out = out_r + out_strides[22];
        v_out22 = _mm_sub_ps(av_s35, av_s36);
        STHR_128_S(curr_out, v_out_stride, v_out22);

        av_t14 = _mm_mul_ps(av_s17, v_CRTM_16_1);
        av_t15 = _mm_mul_ps(av_s19, v_CRTM_16_2);
        av_s39 = _mm_add_ps(av_s29, av_t9);
        av_s40 = _mm_add_ps(av_t14, av_t15);
        // output point 15: x(14)
        curr_out = out_r + out_strides[14];
        v_out14 = _mm_sub_ps(av_s39, av_s40);
        STHR_128_S(curr_out, v_out_stride, v_out14);
        // output point 31: x(30)
        curr_out = out_r + out_strides[30];
        v_out30 = _mm_add_ps(av_s39, av_s40);
        STHR_128_S(curr_out, v_out_stride, v_out30);

        av_t12 = _mm_mul_ps(av_s20, v_CRTM_16_1);
        av_t13 = _mm_mul_ps(av_s18, v_CRTM_16_2);
        av_s37 = _mm_sub_ps(av_s30, av_t6);
        av_s38 = _mm_add_ps(av_t12, av_t13);
        // output point 11: x(10)
        curr_out = out_r + out_strides[10];
        v_out10 = _mm_sub_ps(av_s37, av_s38);
        STHR_128_S(curr_out, v_out_stride, v_out10);
        // output point 27: x(26)
        curr_out = out_r + out_strides[26];
        v_out26 = _mm_add_ps(av_s37, av_s38);
        STHR_128_S(curr_out, v_out_stride, v_out26);

        av_s32 = _mm_sub_ps(av_s16, av_s21);
        av_t16 = _mm_mul_ps(av_s32, v_CRTM_16_3);
        av_t18 = _mm_mul_ps(av_s10, v_CRTM_16_4);
        av_s41 = _mm_sub_ps(av_s26, av_t18);
        // output point 5: x(4)
        curr_out = out_r + out_strides[4];
        v_out4 = _mm_add_ps(av_s41, av_t16);
        STHR_128_S(curr_out, v_out_stride, v_out4);
        // output point 21: x(20)
        curr_out = out_r + out_strides[20];
        v_out20 = _mm_sub_ps(av_s41, av_t16);
        STHR_128_S(curr_out, v_out_stride, v_out20);

        av_s31 = _mm_add_ps(av_s16, av_s21);
        av_t17 = _mm_mul_ps(av_s31, v_CRTM_16_3);
        av_s42 = _mm_add_ps(av_s26, av_t18);
        // output point 13: x(12)
        curr_out = out_r + out_strides[12];
        v_out12 = _mm_sub_ps(av_s42, av_t17);
        STHR_128_S(curr_out, v_out_stride, v_out12);
        // output point 29: x(28)
        curr_out = out_r + out_strides[28];
        v_out28 = _mm_add_ps(av_s42, av_t17);
        STHR_128_S(curr_out, v_out_stride, v_out28);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8, bv_in9, bv_in10, bv_in11, bv_in12, bv_in13, bv_in14,
               bv_in15;
        __m128 bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_t0, bv_s7, bv_t1, bv_t2,
               bv_s10, bv_s11, bv_s12, bv_t3, bv_s14, bv_t4, bv_s16, bv_t5,
               bv_t6, bv_s19, bv_t7, bv_s21, bv_t8, bv_s23, bv_s24, bv_t9,
               bv_s26, bv_t10, bv_t11, bv_s29, bv_s30, bv_s31, bv_t12, bv_t13,
               bv_s34, bv_s35, bv_s36, bv_t18, bv_t19, bv_s39, bv_s40, bv_t20,
               bv_t14, bv_s43, bv_s44, bv_t15, bv_s46, bv_s47, bv_s48, bv_s49,
               bv_s50;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in_cp + in_strides[1];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in_cp + in_strides[5];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in_cp + in_strides[9];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in4, bv_in5);
        // Input point 14: x(13) & Input point 15: x(14)
        curr_in = in_cp + in_strides[13];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in6, bv_in7);
        // Input point 18: x(17) & Input point 19: x(18)
        curr_in = in_cp + in_strides[17];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in8, bv_in9);
        // Input point 22: x(21) & Input point 23: x(22)
        curr_in = in_cp + in_strides[21];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in10, bv_in11);
        // Input point 26: x(25) & Input point 27: x(26)
        curr_in = in_cp + in_strides[25];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in12, bv_in13);
        // Input point 30: x(29) & Input point 31: x(30)
        curr_in = in_cp + in_strides[29];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in14, bv_in15);

        bv_s1 = _mm_add_ps(bv_in14, bv_in0);
        bv_s2 = _mm_sub_ps(bv_in14, bv_in0);
        bv_s3 = _mm_add_ps(bv_in15, bv_in1);
        bv_s4 = _mm_sub_ps(bv_in15, bv_in1);
        bv_s5 = _mm_add_ps(bv_in12, bv_in2);
        bv_t0 = _mm_sub_ps(bv_in12, bv_in2);
        bv_s7 = _mm_add_ps(bv_in13, bv_in3);
        bv_t1 = _mm_sub_ps(bv_in13, bv_in3);
        bv_t2 = _mm_add_ps(bv_in10, bv_in4);
        bv_s10 = _mm_sub_ps(bv_in10, bv_in4);
        bv_s11 = _mm_add_ps(bv_in11, bv_in5);
        bv_s12 = _mm_sub_ps(bv_in11, bv_in5);
        bv_t3 = _mm_add_ps(bv_in8, bv_in6);
        bv_s14 = _mm_sub_ps(bv_in8, bv_in6);
        bv_t4 = _mm_add_ps(bv_in9, bv_in7);
        bv_s16 = _mm_sub_ps(bv_in9, bv_in7);

        bv_t5 = _mm_sub_ps(bv_s1, bv_t3);
        bv_t6 = _mm_add_ps(bv_s12, bv_t1);
        bv_s19 = _mm_add_ps(bv_t5, bv_t6);
        bv_t7 = _mm_sub_ps(bv_t6, bv_t5);

        bv_s21 = _mm_add_ps(bv_s4, bv_s16);
        bv_t8 = _mm_sub_ps(bv_s5, bv_t2);
        bv_s23 = _mm_add_ps(bv_s21, bv_t8);
        bv_s24 = _mm_sub_ps(bv_s21, bv_t8);

        // Output point 6: X(5)
        v_out5 = _mm_add_ps(_mm_mul_ps(v_CRTM_16_1, bv_s19),
                            _mm_mul_ps(v_CRTM_16_2, bv_s23));
        curr_out = out_r + out_strides[5];
        STHR_128_S(curr_out, v_out_stride, v_out5);
        // Output point 22: X(21)
        v_out21 = _mm_sub_ps(_mm_mul_ps(v_CRTM_16_1, bv_s23),
                             _mm_mul_ps(v_CRTM_16_2, bv_s19));
        curr_out = out_r + out_strides[21];
        STHR_128_S(curr_out, v_out_stride, v_out21);
        // Output point 30: X(29)
        v_out29 = _mm_add_ps(_mm_mul_ps(v_CRTM_16_1, bv_t7),
                             _mm_mul_ps(v_CRTM_16_2, bv_s24));
        curr_out = out_r + out_strides[29];
        STHR_128_S(curr_out, v_out_stride, v_out29);
        // Output point 14: X(13)
        v_out13 = _mm_sub_ps(_mm_mul_ps(v_CRTM_16_1, bv_s24),
                             _mm_mul_ps(v_CRTM_16_2, bv_t7));
        curr_out = out_r + out_strides[13];
        STHR_128_S(curr_out, v_out_stride, v_out13);

        bv_t9 = _mm_add_ps(bv_s1, bv_t3);
        bv_s26 = _mm_add_ps(bv_s5, bv_t2);
        // Output point 2: X(1)
        v_out1 = _mm_mul_ps(v_CRTM_16_4, _mm_add_ps(bv_t9, bv_s26));
        curr_out = out_r + out_strides[1];
        STHR_128_S(curr_out, v_out_stride, v_out1);

        bv_t10 = _mm_sub_ps(bv_s4, bv_s16);
        bv_t11 = _mm_sub_ps(bv_s12, bv_t1);
        // Output point 18: X(17)
        v_out17 = _mm_mul_ps(v_CRTM_16_4, _mm_add_ps(bv_t10, bv_t11));
        curr_out = out_r + out_strides[17];
        STHR_128_S(curr_out, v_out_stride, v_out17);

        bv_s29 = _mm_sub_ps(bv_t9, bv_s26);
        bv_s30 = _mm_sub_ps(bv_t10, bv_t11);

        // Output point 10: X(9)
        v_out9 = _mm_mul_ps(v_CRTM_16_3, _mm_add_ps(bv_s29, bv_s30));
        curr_out = out_r + out_strides[9];
        STHR_128_S(curr_out, v_out_stride, v_out9);
        // Output point 26: X(25)
        v_out25 = _mm_mul_ps(v_CRTM_16_3, _mm_sub_ps(bv_s30, bv_s29));
        curr_out = out_r + out_strides[25];
        STHR_128_S(curr_out, v_out_stride, v_out25);

        bv_s31 = _mm_add_ps(bv_s2, bv_t4);
        bv_t12 = NEGATE_128_S(_mm_add_ps(bv_s3, bv_s14));

        bv_t13 = _mm_sub_ps(bv_s14, bv_s3);
        bv_s34 = _mm_sub_ps(bv_t4, bv_s2);

        bv_s35 = _mm_add_ps(bv_s11, bv_t0);
        bv_s36 = _mm_add_ps(bv_s10, bv_s7);

        bv_t18 = _mm_mul_ps(v_CRTM_16_5, _mm_add_ps(bv_s35, bv_s36));
        bv_t19 = _mm_mul_ps(v_CRTM_16_5, _mm_sub_ps(bv_s36, bv_s35));

        bv_s43 = _mm_add_ps(bv_t18, bv_s31);
        bv_s44 = _mm_add_ps(bv_t12, bv_t19);
        bv_t15 = _mm_sub_ps(bv_s31, bv_t18);
        bv_s46 = _mm_sub_ps(bv_t12, bv_t19);

        // Output point 4: X(3)
        v_out3 = _mm_sub_ps(_mm_mul_ps(v_CRTM_16_7, bv_s44),
                            _mm_mul_ps(v_CRTM_16_6, bv_s43));
        curr_out = out_r + out_strides[3];
        STHR_128_S(curr_out, v_out_stride, v_out3);
        // Output point 12: X(11)
        v_out11 = _mm_sub_ps(_mm_mul_ps(v_CRTM_16_9, bv_s46),
                             _mm_mul_ps(v_CRTM_16_8, bv_t15));
        curr_out = out_r + out_strides[11];
        STHR_128_S(curr_out, v_out_stride, v_out11);
        // Output point 28: X(27)
        v_out27 = _mm_add_ps(_mm_mul_ps(v_CRTM_16_9, bv_t15),
                             _mm_mul_ps(v_CRTM_16_8, bv_s46));
        curr_out = out_r + out_strides[27];
        STHR_128_S(curr_out, v_out_stride, v_out27);
        // Output point 20: X(19)
        v_out19 = _mm_add_ps(_mm_mul_ps(v_CRTM_16_6, bv_s44),
                             _mm_mul_ps(v_CRTM_16_7, bv_s43));
        curr_out = out_r + out_strides[19];
        STHR_128_S(curr_out, v_out_stride, v_out19);

        bv_s39 = _mm_sub_ps(bv_t0, bv_s11);
        bv_s40 = _mm_sub_ps(bv_s10, bv_s7);

        bv_t20 = _mm_mul_ps(v_CRTM_16_5, _mm_sub_ps(bv_s39, bv_s40));
        bv_t14 = _mm_mul_ps(v_CRTM_16_5, _mm_add_ps(bv_s39, bv_s40));

        bv_s47 = _mm_add_ps(bv_t13, bv_t20);
        bv_s48 = _mm_add_ps(bv_s34, bv_t14);
        bv_s49 = _mm_sub_ps(bv_t13, bv_t20);
        bv_s50 = _mm_sub_ps(bv_s34, bv_t14);

        // Output point 16: X(15)
        v_out15 = _mm_add_ps(_mm_mul_ps(v_CRTM_16_6, bv_s47),
                             _mm_mul_ps(v_CRTM_16_7, bv_s50));
        curr_out = out_r + out_strides[15];
        STHR_128_S(curr_out, v_out_stride, v_out15);
        // Output point 32: X(31)
        v_out31 = _mm_sub_ps(_mm_mul_ps(v_CRTM_16_7, bv_s47),
                             _mm_mul_ps(v_CRTM_16_6, bv_s50));
        curr_out = out_r + out_strides[31];
        STHR_128_S(curr_out, v_out_stride, v_out31);
        // Output point 8: X(7)
        v_out7 = _mm_add_ps(_mm_mul_ps(v_CRTM_16_9, bv_s48),
                            _mm_mul_ps(v_CRTM_16_8, bv_s49));
        curr_out = out_r + out_strides[7];
        STHR_128_S(curr_out, v_out_stride, v_out7);
        // Output point 24: X(23)
        v_out23 = _mm_sub_ps(_mm_mul_ps(v_CRTM_16_9, bv_s49),
                             _mm_mul_ps(v_CRTM_16_8, bv_s48));
        curr_out = out_r + out_strides[23];
        STHR_128_S(curr_out, v_out_stride, v_out23);

        in_cp = in_cp + (v_in_stride << 1);
        in_r = in_r + (v_in_dc_nyq_stride << 1);
        out_r = out_r + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        /* Standard DFT */
        FFTZ_FLOAT a_in0, a_in1, a_in2, a_in3, a_in4, a_in5, a_in6, a_in7,
            a_in8, a_in9, a_in10, a_in11, a_in12, a_in13, a_in14, a_in15;
        FFTZ_FLOAT a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_s9,
            a_s10, a_s11, a_s12, a_s13, a_s14, a_s15, a_s16, a_s17, a_s18,
            a_s19, a_s20, a_s21, a_s22, a_s23, a_t12, a_t16, a_t14, a_s27,
            a_t13, a_s29, a_t15, a_s31, a_s32, a_t17, a_t0, a_t1, a_t2, a_t3,
            a_t4, a_t5, a_t6, a_t7, a_t8, a_t9, a_t10, a_t11, a_s46, a_t18,
            a_t20, a_t19, a_s49, a_s50, a_s51, a_s52, a_s53, a_s54, a_s55,
            a_s56, a_s57, a_s58;

        curr_in = in_r;
        curr_out = out_r;
        // Input point 1: x(0)
        a_in0 = *in_r;
        // Input point 4: x(3)
        a_in1 = in_cp[in_strides[3]];
        // Input point 5: x(4)
        a_in2 = in_cp[in_strides[4]];
        // Input point 8: x(7)
        a_in3 = in_cp[in_strides[7]];
        // Input point 9: x(8)
        a_in4 = in_cp[in_strides[8]];
        // Input point 12: x(11)
        a_in5 = in_cp[in_strides[11]];
        // Input point 13: x(12)
        a_in6 = in_cp[in_strides[12]];
        // Input point 16: x(15)
        a_in7 = in_cp[in_strides[15]];
        // Input point 17: x(16)
        a_in8 = in_cp[in_strides[16]];
        // Input point 20: x(19)
        a_in9 = in_cp[in_strides[19]];
        // Input point 21: x(20)
        a_in10 = in_cp[in_strides[20]];
        // Input point 24: x(23)
        a_in11 = in_cp[in_strides[23]];
        // Input point 25: x(24)
        a_in12 = in_cp[in_strides[24]];
        // Input point 28: x(27)
        a_in13 = in_cp[in_strides[27]];
        // Input point 29: x(28)
        a_in14 = in_cp[in_strides[28]];
        // Input point 32: x(31)
        a_in15 = in_r[in_strides[31]];

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
        a_s27 = a_s0 + a_t12;
        a_s31 = a_t14 + a_s27;
        // Output point 1: X(0)
        *out_r = a_s31 + a_t13;
        // Output point 17: X(16)
        out_r[out_strides[16]] = a_s31 - a_t13;

        a_t15 = CRTM_16_4 * a_s21;
        a_s32 = a_s27 - a_t14;
        // Output point 9: X(8)
        out_r[out_strides[8]] = a_s32 - a_t15;
        // Output point 25: X(24)
        out_r[out_strides[24]] = a_s32 + a_t15;

        a_t16 = CRTM_16_4 * a_in8;
        a_t17 = CRTM_16_3 * a_s23;
        a_t0 = CRTM_16_1 * a_s17;
        a_t1 = CRTM_16_2 * a_s19;
        a_t3 = a_s1 - a_t16;
        a_s49 = a_t17 + a_t3;
        a_s50 = a_t0 - a_t1;
        // Output point 3: X(2)
        out_r[out_strides[2]] = a_s49 + a_s50;
        // Output point 19: X(18)
        out_r[out_strides[18]] = a_s49 - a_s50;

        a_s29 = a_s0 - a_t12;
        a_t2 = a_s1 + a_t16;
        a_t4 = CRTM_16_3 * a_s22;
        a_t5 = CRTM_16_1 * a_s18;
        a_t6 = CRTM_16_2 * a_s16;
        a_s51 = a_t2 - a_t4;
        a_s52 = a_t6 - a_t5;
        // Output point 7: X(6)
        out_r[out_strides[6]] = a_s51 + a_s52;
        // Output point 23: X(22)
        out_r[out_strides[22]] = a_s51 - a_s52;

        a_t9 = CRTM_16_1 * a_s16;
        a_t10 = CRTM_16_2 * a_s18;
        a_s55 = a_t2 + a_t4;
        a_s56 = a_t9 + a_t10;
        // Output point 15: X(14)
        out_r[out_strides[14]] = a_s55 - a_s56;
        // Output point 31: X(30)
        out_r[out_strides[30]] = a_s55 + a_s56;

        a_t7 = CRTM_16_1 * a_s19;
        a_t8 = CRTM_16_2 * a_s17;
        a_s53 = a_t3 - a_t17;
        a_s54 = a_t7 + a_t8;
        // Output point 11: X(10)
        out_r[out_strides[10]] = a_s53 - a_s54;
        // Output point 27: X(26)
        out_r[out_strides[26]] = a_s53 + a_s54;

        a_s46 = a_s15 - a_s20;
        a_t18 = CRTM_16_3 * a_s46;
        a_t19 = CRTM_16_4 * a_s9;
        a_s57 = a_s29 - a_t19;
        // Output point 5: X(4)
        out_r[out_strides[4]] = a_s57 + a_t18;
        // Output point 21: X(20)
        out_r[out_strides[20]] = a_s57 - a_t18;

        a_t11 = a_s15 + a_s20;
        a_t20 = CRTM_16_3 * a_t11;
        a_s58 = a_s29 + a_t19;
        // Output point 13: X(12)
        out_r[out_strides[12]] = a_s58 - a_t20;
        // Output point 29: X(28)
        out_r[out_strides[28]] = a_s58 + a_t20;

        /* Shifted DFT */
        FFTZ_FLOAT b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7,
            b_in8, b_in9, b_in10, b_in11, b_in12, b_in13, b_in14, b_in15;
        FFTZ_FLOAT b_s1, b_s2, b_s3, b_s4, b_s5, b_t0, b_s7, b_t1, b_t2, b_s10,
              b_s11, b_s12, b_t3, b_s14, b_t4, b_s16, b_t5, b_t6, b_s19, b_t7,
              b_s21, b_t8, b_s23, b_s24, b_t9, b_s26, b_t10, b_t11, b_s29,
              b_s30, b_s31, b_t12, b_t13, b_s34, b_s35, b_s36, b_t18, b_t19,
              b_s39, b_s40, b_t20, b_t14, b_s43, b_s44, b_t15, b_s46, b_s47,
              b_s48, b_s49, b_s50;

        //  Input point 2: x(1)
        b_in0 = in_cp[in_strides[1]];
        // Input point 3: x(2)
        b_in1 = in_cp[in_strides[2]];
        // Input point 6: x(5)
        b_in2 = in_cp[in_strides[5]];
        // Input point 7: x(6)
        b_in3 = in_cp[in_strides[6]];
        // Input point 10: x(9)
        b_in4 = in_cp[in_strides[9]];
        // Input point 11: x(10)
        b_in5 = in_cp[in_strides[10]];
        // Input point 14: x(13)
        b_in6 = in_cp[in_strides[13]];
        // Input point 15: x(14)
        b_in7 = in_cp[in_strides[14]];
        // Input point 18: x(17)
        b_in8 = in_cp[in_strides[17]];
        // Input point 19: x(18)
        b_in9 = in_cp[in_strides[18]];
        // Input point 22: x(21)
        b_in10 = in_cp[in_strides[21]];
        // Input point 23: x(22)
        b_in11 = in_cp[in_strides[22]];
        // Input point 26: x(25)
        b_in12 = in_cp[in_strides[25]];
        // Input point 27: x(26)
        b_in13 = in_cp[in_strides[26]];
        // Input point 30: x(29)
        b_in14 = in_cp[in_strides[29]];
        // Input point 31: x(30)
        b_in15 = in_cp[in_strides[30]];

        b_s1 = b_in14 + b_in0;
        b_s2 = b_in14 - b_in0;
        b_s3 = b_in15 + b_in1;
        b_s4 = b_in15 - b_in1;
        b_s5 = b_in12 + b_in2;
        b_t0 = b_in12 - b_in2;
        b_s7 = b_in13 + b_in3;
        b_t1 = b_in13 - b_in3;
        b_t2 = b_in10 + b_in4;
        b_s10 = b_in10 - b_in4;
        b_s11 = b_in11 + b_in5;
        b_s12 = b_in11 - b_in5;
        b_t3 = b_in8 + b_in6;
        b_s14 = b_in8 - b_in6;
        b_t4 = b_in9 + b_in7;
        b_s16 = b_in9 - b_in7;

        b_t5 = b_s1 - b_t3;
        b_t6 = b_s12 + b_t1;
        b_s19 = b_t5 + b_t6;
        b_t7 = b_t6 - b_t5;

        b_s21 = b_s4 + b_s16;
        b_t8 = b_s5 - b_t2;
        b_s23 = b_s21 + b_t8;
        b_s24 = b_s21 - b_t8;

        // Output point 6: X(5)
        out_r[out_strides[5]] = (CRTM_16_1 * b_s19) + (CRTM_16_2 * b_s23);
        // Output point 22: X(21)
        out_r[out_strides[21]] = (CRTM_16_1 * b_s23) - (CRTM_16_2 * b_s19);
        // Output point 30: X(29)
        out_r[out_strides[29]] = (CRTM_16_1 * b_t7) + (CRTM_16_2 * b_s24);
        // Output point 14: X(13)
        out_r[out_strides[13]] = (CRTM_16_1 * b_s24) - (CRTM_16_2 * b_t7);

        b_t9 = b_s1 + b_t3;
        b_s26 = b_s5 + b_t2;
        // Output point 2: X(1)
        out_r[out_strides[1]] = CRTM_16_4 * (b_t9 + b_s26);

        b_t10 = b_s4 - b_s16;
        b_t11 = b_s12 - b_t1;
        // Output point 18: X(17)
        out_r[out_strides[17]] = CRTM_16_4 * (b_t10 + b_t11);

        b_s29 = b_t9 - b_s26;
        b_s30 = b_t10 - b_t11;

        // Output point 10: X(9)
        out_r[out_strides[9]] = CRTM_16_3 * (b_s29 + b_s30);
        // Output point 26: X(25)
        out_r[out_strides[25]] = CRTM_16_3 * (b_s30 - b_s29);

        b_s31 = b_s2 + b_t4;
        b_t12 = -b_s3 - b_s14;

        b_t13 = b_s14 - b_s3;
        b_s34 = b_t4 - b_s2;

        b_s35 = b_s11 + b_t0;
        b_s36 = b_s10 + b_s7;

        b_t18 = CRTM_16_5 * (b_s35 + b_s36);
        b_t19 = CRTM_16_5 * (b_s36 - b_s35);

        b_s43 = b_t18 + b_s31;
        b_s44 = b_t12 + b_t19;
        b_t15 = b_s31 - b_t18;
        b_s46 = b_t12 - b_t19;

        // Output point 4: X(3)
        out_r[out_strides[3]] = (CRTM_16_7 * b_s44) - (CRTM_16_6 * b_s43);
        // Output point 12: X(11)
        out_r[out_strides[11]] = (CRTM_16_9 * b_s46) - (CRTM_16_8 * b_t15);
        // Output point 28: X(27)
        out_r[out_strides[27]] = (CRTM_16_9 * b_t15) + (CRTM_16_8 * b_s46);
        // Output point 20: X(19)
        out_r[out_strides[19]] = (CRTM_16_6 * b_s44) + (CRTM_16_7 * b_s43);

        b_s39 = b_t0 - b_s11;
        b_s40 = b_s10 - b_s7;

        b_t20 = CRTM_16_5 * (b_s39 - b_s40);
        b_t14 = CRTM_16_5 * (b_s39 + b_s40);

        b_s47 = b_t13 + b_t20;
        b_s48 = b_s34 + b_t14;
        b_s49 = b_t13 - b_t20;
        b_s50 = b_s34 - b_t14;

        // Output point 16: X(15)
        out_r[out_strides[15]] = (CRTM_16_6 * b_s47) + (CRTM_16_7 * b_s50);
        // Output point 32: X(31)
        out_r[out_strides[31]] = (CRTM_16_7 * b_s47) - (CRTM_16_6 * b_s50);
        // Output point 8: X(7)
        out_r[out_strides[7]] = (CRTM_16_9 * b_s48) + (CRTM_16_8 * b_s49);
        // Output point 24: X(23)
        out_r[out_strides[23]] = (CRTM_16_9 * b_s49) - (CRTM_16_8 * b_s48);
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft16avx128_fp64_fwd(FFTZ_VOID *in_real,
                                             FFTZ_VOID *in_complex,
                                             FFTZ_VOID *out_real,
                                             FFTZ_VOID *out_complex,
                                             FFTZ_INTP n,
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
    FFTZ_UINT8 is_contiguous_in = (v_in_stride == 1);
    FFTZ_INTP v_out_dc_nyq_stride = strides->v_out_sym_stride;
    FFTZ_UINT8 is_contiguous_out_dc_nyq = (v_out_dc_nyq_stride == 1);

    FFTZ_INTP cnt;
    FFTZ_DOUBLE *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_128_D;

    __m128d v_CRTM_16_1 = _mm_set1_pd(CRTM_16_1);
    __m128d v_CRTM_16_2 = _mm_set1_pd(CRTM_16_2);
    __m128d v_CRTM_16_3 = _mm_set1_pd(CRTM_16_3);
    __m128d v_CRTM_16_4 = _mm_set1_pd(CRTM_16_4);
    __m128d v_CRTM_16_5 = _mm_set1_pd(CRTM_16_5);
    __m128d v_CRTM_16_6 = _mm_set1_pd(CRTM_16_6);
    __m128d v_CRTM_16_7 = _mm_set1_pd(CRTM_16_7);

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m128d av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
                av_in8, av_in9, av_in10, av_in11, av_in12, av_in13, av_in14,
                av_in15;
        __m128d av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8, av_s9,
                av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16, av_s17,
                av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24, av_s25,
                av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32, av_s33,
                av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40, av_s41,
                av_s42;
        __m128d av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8, av_t9,
                av_t10, av_t11, av_t12;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
                v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21,
                v_out22, v_out23, v_out24, v_out25, v_out26, v_out27, v_out28,
                v_out29, v_out30, v_out31;

        curr_in = in_r;
        curr_out = out_cp;

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_stride, av_in0, is_contiguous_in);
        // Input point 3: x(2)
        curr_in = in_r + in_strides[2];
        LDR_128_D(curr_in, v_in_stride, av_in1, is_contiguous_in);
        // Input point 5: x(4)
        curr_in = in_r + in_strides[4];
        LDR_128_D(curr_in, v_in_stride, av_in2, is_contiguous_in);
        // Input point 7: x(6)
        curr_in = in_r + in_strides[6];
        LDR_128_D(curr_in, v_in_stride, av_in3, is_contiguous_in);
        // Input point 9: x(8)
        curr_in = in_r + in_strides[8];
        LDR_128_D(curr_in, v_in_stride, av_in4, is_contiguous_in);
        // Input point 11: x(10)
        curr_in = in_r + in_strides[10];
        LDR_128_D(curr_in, v_in_stride, av_in5, is_contiguous_in);
        // Input point 13: x(12)
        curr_in = in_r + in_strides[12];
        LDR_128_D(curr_in, v_in_stride, av_in6, is_contiguous_in);
        // Input point 15: x(14)
        curr_in = in_r + in_strides[14];
        LDR_128_D(curr_in, v_in_stride, av_in7, is_contiguous_in);
        // Input point 17: x(16)
        curr_in = in_r + in_strides[16];
        LDR_128_D(curr_in, v_in_stride, av_in8, is_contiguous_in);
        // Input point 19: x(18)
        curr_in = in_r + in_strides[18];
        LDR_128_D(curr_in, v_in_stride, av_in9, is_contiguous_in);
        // Input point 21: x(20)
        curr_in = in_r + in_strides[20];
        LDR_128_D(curr_in, v_in_stride, av_in10, is_contiguous_in);
        // Input point 23: x(22)
        curr_in = in_r + in_strides[22];
        LDR_128_D(curr_in, v_in_stride, av_in11, is_contiguous_in);
        // Input point 25: x(24)
        curr_in = in_r + in_strides[24];
        LDR_128_D(curr_in, v_in_stride, av_in12, is_contiguous_in);
        // Input point 27: x(26)
        curr_in = in_r + in_strides[26];
        LDR_128_D(curr_in, v_in_stride, av_in13, is_contiguous_in);
        // Input point 29: x(28)
        curr_in = in_r + in_strides[28];
        LDR_128_D(curr_in, v_in_stride, av_in14, is_contiguous_in);
        // Input point 31: x(30)
        curr_in = in_r + in_strides[30];
        LDR_128_D(curr_in, v_in_stride, av_in15, is_contiguous_in);

        av_s1 = _mm_add_pd(av_in0, av_in8);
        av_s2 = _mm_sub_pd(av_in0, av_in8);
        av_s3 = _mm_add_pd(av_in1, av_in15);
        av_s4 = _mm_sub_pd(av_in1, av_in15);
        av_s5 = _mm_add_pd(av_in2, av_in6);
        av_s6 = _mm_sub_pd(av_in2, av_in6);
        av_s7 = _mm_add_pd(av_in3, av_in5);
        av_s8 = _mm_sub_pd(av_in3, av_in5);
        av_s9 = _mm_add_pd(av_in4, av_in12);
        av_s10 = _mm_sub_pd(av_in4, av_in12);
        av_s11 = _mm_add_pd(av_in7, av_in9);
        av_s12 = _mm_sub_pd(av_in7, av_in9);
        av_s13 = _mm_add_pd(av_in10, av_in14);
        av_s14 = _mm_sub_pd(av_in10, av_in14);
        av_s15 = _mm_add_pd(av_in11, av_in13);
        av_s16 = _mm_sub_pd(av_in11, av_in13);

        av_s17 = _mm_add_pd(av_s1, av_s9);
        av_s18 = _mm_sub_pd(av_s1, av_s9);
        av_s19 = _mm_add_pd(av_s3, av_s11);
        av_s20 = _mm_sub_pd(av_s3, av_s11);
        av_s21 = _mm_add_pd(av_s4, av_s12);
        av_s22 = _mm_sub_pd(av_s4, av_s12);
        av_s23 = _mm_add_pd(av_s5, av_s13);
        av_s24 = _mm_sub_pd(av_s5, av_s13);
        // Output point 16: X(15)
        v_out15 = _mm_sub_pd(av_s17, av_s23);

        av_s25 = _mm_add_pd(av_s6, av_s14);
        av_s26 = _mm_sub_pd(av_s6, av_s14);
        av_s27 = _mm_add_pd(av_s7, av_s15);
        av_s28 = _mm_sub_pd(av_s7, av_s15);
        av_s29 = _mm_add_pd(av_s8, av_s16);
        av_s30 = _mm_sub_pd(av_s8, av_s16);
        // Output point 17: X(16)
        v_out16 = _mm_sub_pd(av_s29, av_s22);
        curr_out = out_cp + out_strides[15];
        STRI_2x128_D(curr_out, v_out_stride, v_out15, v_out16);

        av_s31 = _mm_add_pd(av_s19, av_s27);
        av_s32 = _mm_sub_pd(av_s19, av_s27);
        av_s33 = _mm_add_pd(av_s17, av_s23);
        av_s34 = _mm_add_pd(av_s22, av_s29);
        // Output point 1: X(0)
        v_out0 = _mm_add_pd(av_s31, av_s33);
        curr_out = out_r + out_strides[0];
        STR_128_D(curr_out, v_out_dc_nyq_stride, v_out0, is_contiguous_out_dc_nyq);
        // Output pt 32: X(31)
        curr_out = out_r + out_strides[31];
        v_out31 = _mm_sub_pd(av_s33, av_s31);
        STR_128_D(curr_out, v_out_dc_nyq_stride, v_out31, is_contiguous_out_dc_nyq);

        av_t1 = _mm_mul_pd(av_s20, v_CRTM_16_1);
        av_t2 = _mm_mul_pd(av_s21, v_CRTM_16_1);
        av_t3 = _mm_mul_pd(av_s28, v_CRTM_16_1);
        av_t4 = _mm_mul_pd(av_s30, v_CRTM_16_1);
        av_t5 = _mm_mul_pd(av_s20, v_CRTM_16_2);
        av_t6 = _mm_mul_pd(av_s21, v_CRTM_16_2);
        av_t7 = _mm_mul_pd(av_s28, v_CRTM_16_2);
        av_t8 = _mm_mul_pd(av_s30, v_CRTM_16_2);
        av_t9 = _mm_mul_pd(av_s24, v_CRTM_16_3);
        av_t10 = _mm_mul_pd(av_s26, v_CRTM_16_3);
        av_t11 = _mm_mul_pd(av_s32, v_CRTM_16_3);
        // Output point 8: X(7)
        v_out7 = _mm_add_pd(av_s18, av_t11);
        // Output point 24: X(23)
        v_out23 = _mm_sub_pd(av_s18, av_t11);

        av_t12 = _mm_mul_pd(av_s34, v_CRTM_16_3);
        // Output point 9: X(8)
        v_out8 = NEGATE_128_D(_mm_add_pd(av_s25, av_t12));
        curr_out = out_cp + out_strides[7];
        STRI_2x128_D(curr_out, v_out_stride, v_out7, v_out8);
        // Output point 25: X(24)
        v_out24 = _mm_sub_pd(av_s25, av_t12);
        curr_out = out_cp + out_strides[23];
        STRI_2x128_D(curr_out, v_out_stride, v_out23, v_out24);

        av_s35 = _mm_add_pd(av_t1, av_t8);
        av_s36 = _mm_add_pd(av_t10, av_s2);
        // Output point 4: X(3)
        v_out3 = _mm_add_pd(av_s35, av_s36);
        // Output point 28: X(27)
        v_out27 = _mm_sub_pd(av_s36, av_s35);

        av_s37 = _mm_add_pd(av_t3, av_t6);
        av_s38 = _mm_add_pd(av_t9, av_s10);
        // Output point 5: X(4)
        v_out4 = NEGATE_128_D(_mm_add_pd(av_s37, av_s38));
        curr_out = out_cp + out_strides[3];
        STRI_2x128_D(curr_out, v_out_stride, v_out3, v_out4);
        // Output point 29: X(28)
        v_out28 = _mm_sub_pd(av_s38, av_s37);
        curr_out = out_cp + out_strides[27];
        STRI_2x128_D(curr_out, v_out_stride, v_out27, v_out28);

        av_s39 = _mm_sub_pd(av_t4, av_t5);
        av_s40 = _mm_sub_pd(av_s2, av_t10);
        // Output point 12: X(11)
        v_out11 = _mm_sub_pd(av_s40, av_s39);
        // Output point 20: X(19)
        v_out19 = _mm_add_pd(av_s39, av_s40);

        av_s41 = _mm_sub_pd(av_t7, av_t2);
        av_s42 = _mm_sub_pd(av_s10, av_t9);
        // Output point 13: X(12)
        v_out12 = _mm_add_pd(av_s41, av_s42);
        curr_out = out_cp + out_strides[11];
        STRI_2x128_D(curr_out, v_out_stride, v_out11, v_out12);
        // Output point 21: X(20)
        v_out20 = _mm_sub_pd(av_s41, av_s42);
        curr_out = out_cp + out_strides[19];
        STRI_2x128_D(curr_out, v_out_stride, v_out19, v_out20);

        /* Shifted DFT */
        __m128d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
                bv_in8, bv_in9, bv_in10, bv_in11, bv_in12, bv_in13, bv_in14,
                bv_in15;
        __m128d bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
                bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17,
                bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24, bv_s25,
                bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32, bv_s33,
                bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40, bv_s41,
                bv_s42, bv_s43, bv_s44, bv_s45, bv_s46, bv_s47, bv_s48, bv_s49,
                bv_s50;
        __m128d bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
                bv_t10, bv_t11, bv_t12, bv_t13, bv_t14, bv_t15, bv_t16, bv_t17,
                bv_t18, bv_t19, bv_t20, bv_t21, bv_t22, bv_t23, bv_t24, bv_t25,
                bv_t26, bv_t27, bv_t28, bv_t29, bv_t30;

        // Input point 2: x(1)
        curr_in = in_r + in_strides[1];
        LDR_128_D(curr_in, v_in_stride, bv_in0, is_contiguous_in);
        // Input point 4: x(3)
        curr_in = in_r + in_strides[3];
        LDR_128_D(curr_in, v_in_stride, bv_in1, is_contiguous_in);
        // Input point 6: x(5)
        curr_in = in_r + in_strides[5];
        LDR_128_D(curr_in, v_in_stride, bv_in2, is_contiguous_in);
        // Input point 8: x(7)
        curr_in = in_r + in_strides[7];
        LDR_128_D(curr_in, v_in_stride, bv_in3, is_contiguous_in);
        // Input point 10: x(9)
        curr_in = in_r + in_strides[9];
        LDR_128_D(curr_in, v_in_stride, bv_in4, is_contiguous_in);
        // Input point 12: x(11)
        curr_in = in_r + in_strides[11];
        LDR_128_D(curr_in, v_in_stride, bv_in5, is_contiguous_in);
        // Input point 14: x(13)
        curr_in = in_r + in_strides[13];
        LDR_128_D(curr_in, v_in_stride, bv_in6, is_contiguous_in);
        // Input point 16: x(15)
        curr_in = in_r + in_strides[15];
        LDR_128_D(curr_in, v_in_stride, bv_in7, is_contiguous_in);
        // Input point 18: x(17)
        curr_in = in_r + in_strides[17];
        LDR_128_D(curr_in, v_in_stride, bv_in8, is_contiguous_in);
        // Input point 20: x(19)
        curr_in = in_r + in_strides[19];
        LDR_128_D(curr_in, v_in_stride, bv_in9, is_contiguous_in);
        // Input point 22: x(21)
        curr_in = in_r + in_strides[21];
        LDR_128_D(curr_in, v_in_stride, bv_in10, is_contiguous_in);
        // Input point 24: x(23)
        curr_in = in_r + in_strides[23];
        LDR_128_D(curr_in, v_in_stride, bv_in11, is_contiguous_in);
        // Input point 26: x(25)
        curr_in = in_r + in_strides[25];
        LDR_128_D(curr_in, v_in_stride, bv_in12, is_contiguous_in);
        // Input point 28: x(27)
        curr_in = in_r + in_strides[27];
        LDR_128_D(curr_in, v_in_stride, bv_in13, is_contiguous_in);
        // Input point 30: x(29)
        curr_in = in_r + in_strides[29];
        LDR_128_D(curr_in, v_in_stride, bv_in14, is_contiguous_in);
        // Input point 32: x(31)
        curr_in = in_r + in_strides[31];
        LDR_128_D(curr_in, v_in_stride, bv_in15, is_contiguous_in);

        bv_s1 = _mm_sub_pd(bv_in3, bv_in11);
        bv_s2 = _mm_add_pd(bv_in3, bv_in11);
        bv_s3 = _mm_sub_pd(bv_in4, bv_in12);
        bv_s4 = _mm_add_pd(bv_in4, bv_in12);
        bv_s5 = _mm_sub_pd(bv_in5, bv_in13);
        bv_s6 = _mm_add_pd(bv_in5, bv_in13);

        bv_t1 = _mm_mul_pd(v_CRTM_16_3, bv_s3);
        bv_s7 = _mm_add_pd(bv_t1, bv_in0);
        bv_t7 = _mm_mul_pd(v_CRTM_16_1, bv_in2);
        bv_t8 = _mm_mul_pd(v_CRTM_16_2, bv_in10);
        bv_t9 = _mm_mul_pd(v_CRTM_16_2, bv_in6);
        bv_t10 = _mm_mul_pd(v_CRTM_16_1, bv_in14);
        bv_s8 = _mm_sub_pd(bv_t7, bv_t8);
        bv_s9 = _mm_sub_pd(bv_t9, bv_t10);
        bv_s10 = _mm_add_pd(bv_s8, bv_s9);
        bv_s11 = _mm_sub_pd(bv_s7, bv_s10);
        bv_s12 = _mm_add_pd(bv_s7, bv_s10);

        bv_t2 = _mm_mul_pd(v_CRTM_16_3, bv_s5);
        bv_s13 = _mm_add_pd(bv_t2, bv_in1);
        bv_t3 = _mm_mul_pd(v_CRTM_16_3, bv_s6);
        bv_s14 = _mm_add_pd(bv_t3, bv_in9);

        bv_t11 = _mm_mul_pd(v_CRTM_16_6, bv_s13);
        bv_t12 = _mm_mul_pd(v_CRTM_16_7, bv_s14);
        bv_s15 = _mm_sub_pd(bv_t11, bv_t12);

        bv_t4 = _mm_mul_pd(v_CRTM_16_3, bv_s2);
        bv_s16 = _mm_add_pd(bv_t4, bv_in7);
        bv_t5 = _mm_mul_pd(v_CRTM_16_3, bv_s1);
        bv_s17 = _mm_sub_pd(bv_t5, bv_in15);

        bv_t13 = _mm_mul_pd(v_CRTM_16_7, bv_s16);
        bv_t14 = _mm_mul_pd(v_CRTM_16_6, bv_s17);
        bv_s18 = _mm_add_pd(bv_t13, bv_t14);

        bv_s19 = _mm_add_pd(bv_s15, bv_s18);
        bv_s20 = _mm_sub_pd(bv_s18, bv_s15);

        // Output point 2: X(1)
        v_out1 = _mm_add_pd(bv_s12, bv_s19);
        // Output point 30: X(29)
        v_out29 = _mm_sub_pd(bv_s12, bv_s19);

        bv_t6 = _mm_mul_pd(v_CRTM_16_3, bv_s4);
        bv_s21 = _mm_add_pd(bv_t6, bv_in8);

        bv_t15 = _mm_mul_pd(v_CRTM_16_2, bv_in2);
        bv_t16 = _mm_mul_pd(v_CRTM_16_1, bv_in10);
        bv_t17 = _mm_mul_pd(v_CRTM_16_1, bv_in6);
        bv_t18 = _mm_mul_pd(v_CRTM_16_2, bv_in14);
        bv_s22 = _mm_add_pd(bv_t15, bv_t16);
        bv_s23 = _mm_add_pd(bv_t17, bv_t18);
        bv_s24 = _mm_add_pd(bv_s22, bv_s23);
        bv_s25 = _mm_sub_pd(bv_s21, bv_s24);
        bv_s26 = _mm_add_pd(bv_s21, bv_s24);
        // Output point 15: X(14)
        v_out14 = _mm_add_pd(bv_s25, bv_s20);
        // Output point 19: X(18)
        v_out18 = _mm_sub_pd(bv_s20, bv_s25);

        bv_t19 = _mm_mul_pd(v_CRTM_16_6, bv_s14);
        bv_t20 = _mm_mul_pd(v_CRTM_16_7, bv_s13);
        bv_t21 = _mm_mul_pd(v_CRTM_16_7, bv_s17);
        bv_t22 = _mm_mul_pd(v_CRTM_16_6, bv_s16);
        bv_s27 = _mm_add_pd(bv_t19, bv_t20);
        bv_s28 = _mm_sub_pd(bv_t21, bv_t22);

        bv_s29 = _mm_add_pd(bv_s28, bv_s27);
        bv_s30 = _mm_sub_pd(bv_s28, bv_s27);
        // Output point 3: X(2)
        v_out2 = _mm_sub_pd(bv_s30, bv_s26);
        curr_out = out_cp + out_strides[1];
        STRI_2x128_D(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 31: X(30)
        v_out30 = _mm_add_pd(bv_s30, bv_s26);
        curr_out = out_cp + out_strides[29];
        STRI_2x128_D(curr_out, v_out_stride, v_out29, v_out30);

        // Output point 18: X(17)
        v_out17 = _mm_sub_pd(bv_s11, bv_s29);
        curr_out = out_cp + out_strides[17];
        STRI_2x128_D(curr_out, v_out_stride, v_out17, v_out18);
        // Output point 14: X(13)
        v_out13 = _mm_add_pd(bv_s11, bv_s29);
        curr_out = out_cp + out_strides[13];
        STRI_2x128_D(curr_out, v_out_stride, v_out13, v_out14);

        bv_s31 = _mm_sub_pd(bv_in0, bv_t1);
        bv_s32 = _mm_sub_pd(bv_s22, bv_s23);
        bv_s33 = _mm_add_pd(bv_s31, bv_s32);
        bv_s34 = _mm_sub_pd(bv_s31, bv_s32);

        bv_s35 = _mm_sub_pd(bv_in9, bv_t3);
        bv_s36 = _mm_sub_pd(bv_in1, bv_t2);

        bv_t23 = _mm_mul_pd(v_CRTM_16_4, bv_s35);
        bv_t24 = _mm_mul_pd(v_CRTM_16_5, bv_s36);
        bv_s37 = _mm_add_pd(bv_t23, bv_t24);

        bv_s38 = _mm_sub_pd(bv_in7, bv_t4);
        bv_s39 = _mm_add_pd(bv_in15, bv_t5);

        bv_t25 = _mm_mul_pd(v_CRTM_16_4, bv_s38);
        bv_t26 = _mm_mul_pd(v_CRTM_16_5, bv_s39);
        bv_s40 = _mm_add_pd(bv_t25, bv_t26);

        bv_s41 = _mm_sub_pd(bv_s37, bv_s40);
        bv_s42 = _mm_add_pd(bv_s37, bv_s40);

        // Output point 6: X(5)
        v_out5 = _mm_add_pd(bv_s33, bv_s41);
        // Output point 26: X(25)
        v_out25 = _mm_sub_pd(bv_s33, bv_s41);

        bv_s43 = _mm_sub_pd(bv_in8, bv_t6);
        bv_s44 = _mm_sub_pd(bv_s9, bv_s8);
        bv_s45 = _mm_sub_pd(bv_s44, bv_s43);
        bv_s46 = _mm_add_pd(bv_s44, bv_s43);
        // Output point 11: X(10)
        v_out10 = _mm_sub_pd(bv_s45, bv_s42);
        // Output point 23: X(22)
        v_out22 = NEGATE_128_D(_mm_add_pd(bv_s45, bv_s42));

        bv_t27 = _mm_mul_pd(v_CRTM_16_5, bv_s38);
        bv_t28 = _mm_mul_pd(v_CRTM_16_4, bv_s39);
        bv_t29 = _mm_mul_pd(v_CRTM_16_5, bv_s35);
        bv_t30 = _mm_mul_pd(v_CRTM_16_4, bv_s36);
        bv_s47 = _mm_sub_pd(bv_t27, bv_t28);
        bv_s48 = _mm_sub_pd(bv_t29, bv_t30);

        bv_s49 = _mm_sub_pd(bv_s47, bv_s48);
        bv_s50 = _mm_add_pd(bv_s48, bv_s47);

        // Output point 22: X(21)
        v_out21 = _mm_sub_pd(bv_s34, bv_s49);
        curr_out = out_cp + out_strides[21];
        STRI_2x128_D(curr_out, v_out_stride, v_out21, v_out22);
        // Output point 10: X(9)
        v_out9 = _mm_add_pd(bv_s34, bv_s49);
        curr_out = out_cp + out_strides[9];
        STRI_2x128_D(curr_out, v_out_stride, v_out9, v_out10);

        // Output point 27: X(26)
        v_out26 = _mm_sub_pd(bv_s50, bv_s46);
        curr_out = out_cp + out_strides[25];
        STRI_2x128_D(curr_out, v_out_stride, v_out25, v_out26);
        // Output point 7: X(6)
        v_out6 = _mm_add_pd(bv_s50, bv_s46);
        curr_out = out_cp + out_strides[5];
        STRI_2x128_D(curr_out, v_out_stride, v_out5, v_out6);

        in_r += v_in_stride * NUM_SETS_REAL_128_D;
        out_cp += v_out_stride * NUM_SETS_REAL_128_D;
        out_r += v_out_dc_nyq_stride * NUM_SETS_REAL_128_D;
    }
    // tail cases
    if (n & 1)
    {
        /* Standard DFT */
        FFTZ_DOUBLE a_in0, a_in1, a_in2, a_in3, a_in4, a_in5, a_in6, a_in7,
            a_in8, a_in9, a_in10, a_in11, a_in12, a_in13, a_in14, a_in15;
        FFTZ_DOUBLE a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_s9,
               a_s10, a_s11, a_s12, a_s13, a_s14, a_s15, a_s16, a_s17, a_s18,
               a_s19, a_s20, a_s21, a_s22, a_s23, a_t12, a_t16, a_t14, a_s27,
               a_t13, a_s29, a_t15, a_s31, a_s32, a_t17, a_t0, a_t1, a_t2, a_t3,
               a_t4, a_t5, a_t6, a_t7, a_t8, a_t9, a_t10, a_t11, a_s46, a_t18,
               a_t20, a_t19, a_s49, a_s50, a_s51, a_s52;

        curr_in = in_r;
        curr_out = out_r;
        // Input point 1: x(0)
        a_in0 = *in_r;
        // Input point 3: x(2)
        a_in1 = in_r[in_strides[2]];
        // Input point 5: x(4)
        a_in2 = in_r[in_strides[4]];
        // Input point 7: x(6)
        a_in3 = in_r[in_strides[6]];
        // Input point 9: x(8)
        a_in4 = in_r[in_strides[8]];
        // Input point 11: x(10)
        a_in5 = in_r[in_strides[10]];
        // Input point 13: x(12)
        a_in6 = in_r[in_strides[12]];
        // Input point 15: x(14)
        a_in7 = in_r[in_strides[14]];
        // Input point 17: x(16)
        a_in8 = in_r[in_strides[16]];
        // Input point 19: x(18)
        a_in9 = in_r[in_strides[18]];
        // Input point 21: x(20)
        a_in10 = in_r[in_strides[20]];
        // Input point 23: x(22)
        a_in11 = in_r[in_strides[22]];
        // Input point 25: x(24)
        a_in12 = in_r[in_strides[24]];
        // Input point 27: x(26)
        a_in13 = in_r[in_strides[26]];
        // Input point 29: x(28)
        a_in14 = in_r[in_strides[28]];
        // Input point 31: x(30)
        a_in15 = in_r[in_strides[30]];

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
        out_cp[out_strides[15]] = a_s16 - a_s22;

        a_t12 = a_s5 + a_s13;
        a_t16 = a_s5 - a_s13;
        a_t14 = a_s6 + a_s14;
        a_s27 = a_s6 - a_s14;
        a_t13 = a_s7 + a_s15;
        a_s29 = a_s7 - a_s15;
        // Output point 17: X(16)
        out_cp[out_strides[16]] = a_t13 - a_s21;

        a_t15 = a_s18 + a_t14;
        a_s31 = a_s18 - a_t14;
        a_s32 = a_s16 + a_s22;
        a_t17 = a_s21 + a_t13;
        // Output point 1: X(0)
        *out_r = a_t15 + a_s32;
        // Output point 32: X(31)
        out_r[out_strides[31]] = a_s32 - a_t15;

        a_t0 = CRTM_16_1 * a_s19;
        a_t1 = CRTM_16_1 * a_s20;
        a_t2 = CRTM_16_1 * a_s27;
        a_t3 = CRTM_16_1 * a_s29;
        a_t4 = CRTM_16_2 * a_s19;
        a_t5 = CRTM_16_2 * a_s20;
        a_t6 = CRTM_16_2 * a_s27;
        a_t7 = CRTM_16_2 * a_s29;
        a_t8 = CRTM_16_3 * a_s23;
        a_t9 = CRTM_16_3 * a_t16;
        a_t10 = CRTM_16_3 * a_s31;
        // Output point 8: X(7)
        out_cp[out_strides[7]] = a_s17 + a_t10;
        // Output point 24: X(23)
        out_cp[out_strides[23]] = a_s17 - a_t10;

        a_t11 = CRTM_16_3 * a_t17;
        // Output point 9: X(8)
        out_cp[out_strides[8]] = -(a_t12 + a_t11);
        // Output point 25: X(24)
        out_cp[out_strides[24]] = a_t12 - a_t11;

        a_s46 = a_t0 + a_t7;
        a_t18 = a_t9 + a_s1;
        // Output point 4: X(3)
        out_cp[out_strides[3]] = a_s46 + a_t18;
        // Output point 28: X(27)
        out_cp[out_strides[27]] = a_t18 - a_s46;

        a_t20 = a_t2 + a_t5;
        a_t19 = a_t8 + a_s9;
        // Output point 5: X(4)
        out_cp[out_strides[4]] = -(a_t20 + a_t19);
        // Output point 29: X(28)
        out_cp[out_strides[28]] = a_t19 - a_t20;

        a_s49 = a_t3 - a_t4;
        a_s50 = a_s1 - a_t9;
        // Output point 12: X(11)
        out_cp[out_strides[11]] = a_s50 - a_s49;
        // Output point 20: X(19)
        out_cp[out_strides[19]] = a_s49 + a_s50;

        a_s51 = a_t6 - a_t1;
        a_s52 = a_s9 - a_t8;
        // Output point 13: X(12)
        out_cp[out_strides[12]] = a_s51 + a_s52;
        // Output point 21: X(20)
        out_cp[out_strides[20]] = a_s51 - a_s52;

        /* Shifted DFT */
        FFTZ_DOUBLE b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7,
            b_in8, b_in9, b_in10, b_in11, b_in12, b_in13, b_in14, b_in15;
        FFTZ_DOUBLE b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_t0, b_s7, b_t1, b_t2,
               b_s10, b_s11, b_s12, b_t3, b_s14, b_t4, b_s16, b_t5, b_t6, b_s19,
               b_t7, b_s21, b_t8, b_s23, b_s24, b_t9, b_s26, b_t10, b_t11,
               b_s29, b_s30, b_s31, b_t12, b_t13, b_s34, b_s35, b_s36, b_t18,
               b_t19, b_s39, b_s40, b_t20, b_t14, b_s43, b_s44, b_t15, b_s46,
               b_s47, b_s48, b_s49, b_s50, b_s51, b_t16, b_t17, b_s53, b_s54;

        // Input point 2: x(1)
        b_in0 = in_r[in_strides[1]];
        // Input point 4: x(3)
        b_in1 = in_r[in_strides[3]];
        // Input point 6: x(5)
        b_in2 = in_r[in_strides[5]];
        // Input point 8: x(7)
        b_in3 = in_r[in_strides[7]];
        // Input point 10: x(9)
        b_in4 = in_r[in_strides[9]];
        // Input point 12: x(11)
        b_in5 = in_r[in_strides[11]];
        // Input point 14: x(13)
        b_in6 = in_r[in_strides[13]];
        // Input point 16: x(15)
        b_in7 = in_r[in_strides[15]];
        // Input point 18: x(17)
        b_in8 = in_r[in_strides[17]];
        // Input point 20: x(19)
        b_in9 = in_r[in_strides[19]];
        // Input point 22: x(21)
        b_in10 = in_r[in_strides[21]];
        // Input point 24: x(23)
        b_in11 = in_r[in_strides[23]];
        // Input point 26: x(25)
        b_in12 = in_r[in_strides[25]];
        // Input point 28: x(27)
        b_in13 = in_r[in_strides[27]];
        // Input point 30: x(29)
        b_in14 = in_r[in_strides[29]];
        // Input point 32: x(31)
        b_in15 = in_r[in_strides[31]];

        b_s0 = b_in3 - b_in11;
        b_s1 = b_in3 + b_in11;
        b_s2 = b_in4 - b_in12;
        b_s3 = b_in4 + b_in12;
        b_s4 = b_in5 - b_in13;
        b_s5 = b_in5 + b_in13;

        b_t0 = CRTM_16_3 * b_s2;
        b_s7 = b_t0 + b_in0;
        b_t1 = (CRTM_16_1 * b_in2) - (CRTM_16_2 * b_in10);
        b_t2 = (CRTM_16_2 * b_in6) - (CRTM_16_1 * b_in14);
        b_s10 = b_t1 + b_t2;
        b_s11 = b_s7 - b_s10;
        b_s12 = b_s7 + b_s10;

        b_t3 = CRTM_16_3 * b_s4;
        b_s14 = b_t3 + b_in1;
        b_t4 = CRTM_16_3 * b_s5;
        b_s16 = b_t4 + b_in9;

        b_t5 = (CRTM_16_6 * b_s14) - (CRTM_16_7 * b_s16);

        b_t6 = CRTM_16_3 * b_s1;
        b_s19 = b_t6 + b_in7;
        b_t7 = CRTM_16_3 * b_s0;
        b_s21 = b_t7 - b_in15;

        b_t8 = (CRTM_16_7 * b_s19) + (CRTM_16_6 * b_s21);

        b_s23 = b_t5 + b_t8;
        b_s24 = b_t8 - b_t5;

        // Output point 2: X(1)
        out_cp[out_strides[1]] = b_s12 + b_s23;
        // Output point 30: X(29)
        out_cp[out_strides[29]] = b_s12 - b_s23;

        b_t9 = CRTM_16_3 * b_s3;
        b_s26 = b_t9 + b_in8;

        b_t10 = (CRTM_16_2 * b_in2) + (CRTM_16_1 * b_in10);
        b_t11 = (CRTM_16_1 * b_in6) + (CRTM_16_2 * b_in14);
        b_s29 = b_t10 + b_t11;
        b_s30 = b_s26 - b_s29;
        b_s31 = b_s26 + b_s29;
        // Output point 15: X(14)
        out_cp[out_strides[14]] = b_s30 + b_s24;
        // Output point 19: X(18)
        out_cp[out_strides[18]] = b_s24 - b_s30;

        b_t12 = (CRTM_16_6 * b_s16) + (CRTM_16_7 * b_s14);
        b_t13 = (CRTM_16_7 * b_s21) - (CRTM_16_6 * b_s19);

        b_s34 = b_t13 + b_t12;
        b_s35 = b_t13 - b_t12;

        // Output point 3: X(2)
        out_cp[out_strides[2]] = b_s35 - b_s31;
        // Output point 31: X(30)
        out_cp[out_strides[30]] = b_s35 + b_s31;

        // Output point 18: X(17)
        out_cp[out_strides[17]] = b_s11 - b_s34;
        // Output point 14: X(13)
        out_cp[out_strides[13]] = b_s11 + b_s34;

        b_s36 = b_in0 - b_t0;

        b_t18 = b_t10 - b_t11;
        b_t19 = b_s36 + b_t18;
        b_s39 = b_s36 - b_t18;

        b_s40 = b_in9 - b_t4;
        b_t20 = b_in1 - b_t3;

        b_t14 = (CRTM_16_4 * b_s40) + (CRTM_16_5 * b_t20);

        b_s43 = b_in7 - b_t6;
        b_s44 = b_in15 + b_t7;

        b_t15 = (CRTM_16_4 * b_s43) + (CRTM_16_5 * b_s44);

        b_s46 = b_t14 - b_t15;
        b_s47 = b_t14 + b_t15;

        // Output point 6: X(5)
        out_cp[out_strides[5]] = b_t19 + b_s46;
        // Output point 26: X(25)
        out_cp[out_strides[25]] = b_t19 - b_s46;

        b_s48 = b_in8 - b_t9;
        b_s49 = b_t2 - b_t1;
        b_s50 = b_s49 - b_s48;
        b_s51 = b_s49 + b_s48;
        // Output point 11: X(10)
        out_cp[out_strides[10]] = b_s50 - b_s47;
        // Output point 23: X(22)
        out_cp[out_strides[22]] = -(b_s50 + b_s47);

        b_t16 = (CRTM_16_5 * b_s43) - (CRTM_16_4 * b_s44);
        b_t17 = (CRTM_16_5 * b_s40) - (CRTM_16_4 * b_t20);

        b_s53 = b_t16 - b_t17;
        b_s54 = b_t17 + b_t16;

        // Output point 22: X(21)
        out_cp[out_strides[21]] = b_s39 - b_s53;
        // Output point 10: X(9)
        out_cp[out_strides[9]] = b_s39 + b_s53;

        // Output point 27: X(26)
        out_cp[out_strides[26]] = b_s54 - b_s51;
        // Output point 7: X(6)
        out_cp[out_strides[6]] = b_s54 + b_s51;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft16avx128_fp64_bwd(FFTZ_VOID *in_real,
                                             FFTZ_VOID *in_complex,
                                             FFTZ_VOID *out_real,
                                             FFTZ_VOID *out_complex,
                                             FFTZ_INTP n,
                                             aoclfftz_strides_t *strides,
                                             FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_16_1 =
        1.847759065022573256256366378793576573644833252;
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
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_UINT8 is_contiguous_out = (v_out_stride == 1);
    FFTZ_INTP v_in_dc_nyq_stride = strides->v_in_sym_stride;
    FFTZ_UINT8 is_contiguous_in_dc_nyq = (v_in_dc_nyq_stride == 1);

    FFTZ_INTP cnt;
    FFTZ_DOUBLE *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_128_D;

    __m128d v_CRTM_16_1 = _mm_set1_pd(CRTM_16_1);
    __m128d v_CRTM_16_2 = _mm_set1_pd(CRTM_16_2);
    __m128d v_CRTM_16_3 = _mm_set1_pd(CRTM_16_3);
    __m128d v_CRTM_16_4 = _mm_set1_pd(CRTM_16_4);
    __m128d v_CRTM_16_5 = _mm_set1_pd(CRTM_16_5);
    __m128d v_CRTM_16_6 = _mm_set1_pd(CRTM_16_6);
    __m128d v_CRTM_16_7 = _mm_set1_pd(CRTM_16_7);
    __m128d v_CRTM_16_8 = _mm_set1_pd(CRTM_16_8);
    __m128d v_CRTM_16_9 = _mm_set1_pd(CRTM_16_9);

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m128d av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
                av_in8, av_in9, av_in10, av_in11, av_in12, av_in13, av_in14,
                av_in15;
        __m128d av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8, av_s9,
                av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16, av_s17,
                av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24, av_s25,
                av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32, av_s33,
                av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40, av_s41,
                av_s42;
        __m128d av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8, av_t9,
                av_t10, av_t11, av_t12, av_t13, av_t14, av_t15, av_t16, av_t17,
                av_t18;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
                v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21,
                v_out22, v_out23, v_out24, v_out25, v_out26, v_out27, v_out28,
                v_out29, v_out30, v_out31;

        curr_in = in_r;
        curr_out = out_r;

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_dc_nyq_stride, av_in0, is_contiguous_in_dc_nyq);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in_cp + in_strides[3];
        LDRI_2x128_D(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in_cp + in_strides[7];
        LDRI_2x128_D(curr_in, v_in_stride, av_in3, av_in4);
        // Input point 12: x(11) & Input point 13: x(12)
        curr_in = in_cp + in_strides[11];
        LDRI_2x128_D(curr_in, v_in_stride, av_in5, av_in6);
        // Input point 16: x(15) & Input point 17: x(16)
        curr_in = in_cp + in_strides[15];
        LDRI_2x128_D(curr_in, v_in_stride, av_in7, av_in8);
        // Input point 20: x(19) & Input point 21: x(20)
        curr_in = in_cp + in_strides[19];
        LDRI_2x128_D(curr_in, v_in_stride, av_in9, av_in10);
        // Input point 24: x(23) & Input point 25: x(24)
        curr_in = in_cp + in_strides[23];
        LDRI_2x128_D(curr_in, v_in_stride, av_in11, av_in12);
        // Input point 28: x(27) & Input point 29: x(28)
        curr_in = in_cp + in_strides[27];
        LDRI_2x128_D(curr_in, v_in_stride, av_in13, av_in14);
        // Input point 32: x(31)
        curr_in = in_r + in_strides[31];
        LDR_128_D(curr_in, v_in_dc_nyq_stride, av_in15, is_contiguous_in_dc_nyq);

        av_s1 = _mm_add_pd(av_in0, av_in15);
        av_s2 = _mm_sub_pd(av_in0, av_in15);
        av_s3 = _mm_add_pd(av_in1, av_in13);
        av_s4 = _mm_sub_pd(av_in1, av_in13);
        av_s5 = _mm_add_pd(av_in2, av_in14);
        av_s6 = _mm_sub_pd(av_in2, av_in14);
        av_s7 = _mm_add_pd(av_in3, av_in11);
        av_s8 = _mm_sub_pd(av_in3, av_in11);
        av_s9 = _mm_add_pd(av_in4, av_in12);
        av_s10 = _mm_sub_pd(av_in4, av_in12);
        av_s11 = _mm_add_pd(av_in5, av_in9);
        av_s12 = _mm_sub_pd(av_in5, av_in9);
        av_s13 = _mm_add_pd(av_in6, av_in10);
        av_s14 = _mm_sub_pd(av_in6, av_in10);

        av_s15 = _mm_add_pd(av_s3, av_s11);
        av_s16 = _mm_sub_pd(av_s3, av_s11);
        av_s17 = _mm_add_pd(av_s4, av_s13);
        av_s18 = _mm_sub_pd(av_s4, av_s13);
        av_s19 = _mm_add_pd(av_s5, av_s12);
        av_s20 = _mm_sub_pd(av_s5, av_s12);
        av_s21 = _mm_add_pd(av_s6, av_s14);
        av_s22 = _mm_sub_pd(av_s6, av_s14);
        av_s23 = _mm_add_pd(av_s8, av_s9);
        av_s24 = _mm_sub_pd(av_s8, av_s9);

        av_t1 = _mm_mul_pd(av_in7, v_CRTM_16_4);
        av_t4 = _mm_mul_pd(av_s15, v_CRTM_16_4);
        av_t3 = _mm_mul_pd(av_s7, v_CRTM_16_4);
        av_s25 = _mm_add_pd(av_s1, av_t1);
        av_s27 = _mm_add_pd(av_t3, av_s25);
        // output point 1: x(0)
        v_out0 = _mm_add_pd(av_s27, av_t4);
        STR_128_D(curr_out, v_out_stride, v_out0, is_contiguous_out);
        // output point 17: x(16)
        curr_out = out_r + out_strides[16];
        v_out16 = _mm_sub_pd(av_s27, av_t4);
        STR_128_D(curr_out, v_out_stride, v_out16, is_contiguous_out);

        av_t5 = _mm_mul_pd(av_s22, v_CRTM_16_4);
        av_s28 = _mm_sub_pd(av_s25, av_t3);
        // output point 9: x(8)
        curr_out = out_r + out_strides[8];
        v_out8 = _mm_sub_pd(av_s28, av_t5);
        STR_128_D(curr_out, v_out_stride, v_out8, is_contiguous_out);
        // output point 25: x(24)
        curr_out = out_r + out_strides[24];
        v_out24 = _mm_add_pd(av_s28, av_t5);
        STR_128_D(curr_out, v_out_stride, v_out24, is_contiguous_out);

        av_t2 = _mm_mul_pd(av_in8, v_CRTM_16_4);
        av_t6 = _mm_mul_pd(av_s24, v_CRTM_16_3);
        av_t7 = _mm_mul_pd(av_s18, v_CRTM_16_1);
        av_t8 = _mm_mul_pd(av_s20, v_CRTM_16_2);
        av_s30 = _mm_sub_pd(av_s2, av_t2);
        av_s33 = _mm_add_pd(av_t6, av_s30);
        av_s34 = _mm_sub_pd(av_t7, av_t8);
        // output point 3: x(2)
        curr_out = out_r + out_strides[2];
        v_out2 = _mm_add_pd(av_s33, av_s34);
        STR_128_D(curr_out, v_out_stride, v_out2, is_contiguous_out);
        // output point 19: x(18)
        curr_out = out_r + out_strides[18];
        v_out18 = _mm_sub_pd(av_s33, av_s34);
        STR_128_D(curr_out, v_out_stride, v_out18, is_contiguous_out);

        av_s26 = _mm_sub_pd(av_s1, av_t1);
        av_s29 = _mm_add_pd(av_s2, av_t2);
        av_t9 = _mm_mul_pd(av_s23, v_CRTM_16_3);
        av_t10 = _mm_mul_pd(av_s19, v_CRTM_16_1);
        av_t11 = _mm_mul_pd(av_s17, v_CRTM_16_2);
        av_s35 = _mm_sub_pd(av_s29, av_t9);
        av_s36 = _mm_sub_pd(av_t11, av_t10);
        // output point 7: x(6)
        curr_out = out_r + out_strides[6];
        v_out6 = _mm_add_pd(av_s35, av_s36);
        STR_128_D(curr_out, v_out_stride, v_out6, is_contiguous_out);
        // output point 23: x(22)
        curr_out = out_r + out_strides[22];
        v_out22 = _mm_sub_pd(av_s35, av_s36);
        STR_128_D(curr_out, v_out_stride, v_out22, is_contiguous_out);

        av_t14 = _mm_mul_pd(av_s17, v_CRTM_16_1);
        av_t15 = _mm_mul_pd(av_s19, v_CRTM_16_2);
        av_s39 = _mm_add_pd(av_s29, av_t9);
        av_s40 = _mm_add_pd(av_t14, av_t15);
        // output point 15: x(14)
        curr_out = out_r + out_strides[14];
        v_out14 = _mm_sub_pd(av_s39, av_s40);
        STR_128_D(curr_out, v_out_stride, v_out14, is_contiguous_out);
        // output point 31: x(30)
        curr_out = out_r + out_strides[30];
        v_out30 = _mm_add_pd(av_s39, av_s40);
        STR_128_D(curr_out, v_out_stride, v_out30, is_contiguous_out);

        av_t12 = _mm_mul_pd(av_s20, v_CRTM_16_1);
        av_t13 = _mm_mul_pd(av_s18, v_CRTM_16_2);
        av_s37 = _mm_sub_pd(av_s30, av_t6);
        av_s38 = _mm_add_pd(av_t12, av_t13);
        // output point 11: x(10)
        curr_out = out_r + out_strides[10];
        v_out10 = _mm_sub_pd(av_s37, av_s38);
        STR_128_D(curr_out, v_out_stride, v_out10, is_contiguous_out);
        // output point 27: x(26)
        curr_out = out_r + out_strides[26];
        v_out26 = _mm_add_pd(av_s37, av_s38);
        STR_128_D(curr_out, v_out_stride, v_out26, is_contiguous_out);

        av_s32 = _mm_sub_pd(av_s16, av_s21);
        av_t16 = _mm_mul_pd(av_s32, v_CRTM_16_3);
        av_t18 = _mm_mul_pd(av_s10, v_CRTM_16_4);
        av_s41 = _mm_sub_pd(av_s26, av_t18);
        // output point 5: x(4)
        curr_out = out_r + out_strides[4];
        v_out4 = _mm_add_pd(av_s41, av_t16);
        STR_128_D(curr_out, v_out_stride, v_out4, is_contiguous_out);
        // output point 21: x(20)
        curr_out = out_r + out_strides[20];
        v_out20 = _mm_sub_pd(av_s41, av_t16);
        STR_128_D(curr_out, v_out_stride, v_out20, is_contiguous_out);

        av_s31 = _mm_add_pd(av_s16, av_s21);
        av_t17 = _mm_mul_pd(av_s31, v_CRTM_16_3);
        av_s42 = _mm_add_pd(av_s26, av_t18);
        // output point 13: x(12)
        curr_out = out_r + out_strides[12];
        v_out12 = _mm_sub_pd(av_s42, av_t17);
        STR_128_D(curr_out, v_out_stride, v_out12, is_contiguous_out);
        // output point 29: x(28)
        curr_out = out_r + out_strides[28];
        v_out28 = _mm_add_pd(av_s42, av_t17);
        STR_128_D(curr_out, v_out_stride, v_out28, is_contiguous_out);

        /* Shifted DFT */
        __m128d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
                bv_in8, bv_in9, bv_in10, bv_in11, bv_in12, bv_in13, bv_in14,
                bv_in15;
        __m128d bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_t0, bv_s7, bv_t1, bv_t2,
                bv_s10, bv_s11, bv_s12, bv_t3, bv_s14, bv_t4, bv_s16, bv_t5,
                bv_t6, bv_s19, bv_t7, bv_s21, bv_t8, bv_s23, bv_s24, bv_t9,
                bv_s26, bv_t10, bv_t11, bv_s29, bv_s30, bv_s31, bv_t12, bv_t13,
                bv_s34, bv_s35, bv_s36, bv_t18, bv_t19, bv_s39, bv_s40, bv_t20,
                bv_t14, bv_s43, bv_s44, bv_t15, bv_s46, bv_s47, bv_s48, bv_s49,
                bv_s50;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in_cp + in_strides[1];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in_cp + in_strides[5];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in_cp + in_strides[9];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in4, bv_in5);
        // Input point 14: x(13) & Input point 15: x(14)
        curr_in = in_cp + in_strides[13];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in6, bv_in7);
        // Input point 18: x(17) & Input point 19: x(18)
        curr_in = in_cp + in_strides[17];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in8, bv_in9);
        // Input point 22: x(21) & Input point 23: x(22)
        curr_in = in_cp + in_strides[21];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in10, bv_in11);
        // Input point 26: x(25) & Input point 27: x(26)
        curr_in = in_cp + in_strides[25];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in12, bv_in13);
        // Input point 30: x(29) & Input point 31: x(30)
        curr_in = in_cp + in_strides[29];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in14, bv_in15);

        bv_s1 = _mm_add_pd(bv_in14, bv_in0);
        bv_s2 = _mm_sub_pd(bv_in14, bv_in0);
        bv_s3 = _mm_add_pd(bv_in15, bv_in1);
        bv_s4 = _mm_sub_pd(bv_in15, bv_in1);
        bv_s5 = _mm_add_pd(bv_in12, bv_in2);
        bv_t0 = _mm_sub_pd(bv_in12, bv_in2);
        bv_s7 = _mm_add_pd(bv_in13, bv_in3);
        bv_t1 = _mm_sub_pd(bv_in13, bv_in3);
        bv_t2 = _mm_add_pd(bv_in10, bv_in4);
        bv_s10 = _mm_sub_pd(bv_in10, bv_in4);
        bv_s11 = _mm_add_pd(bv_in11, bv_in5);
        bv_s12 = _mm_sub_pd(bv_in11, bv_in5);
        bv_t3 = _mm_add_pd(bv_in8, bv_in6);
        bv_s14 = _mm_sub_pd(bv_in8, bv_in6);
        bv_t4 = _mm_add_pd(bv_in9, bv_in7);
        bv_s16 = _mm_sub_pd(bv_in9, bv_in7);

        bv_t5 = _mm_sub_pd(bv_s1, bv_t3);
        bv_t6 = _mm_add_pd(bv_s12, bv_t1);
        bv_s19 = _mm_add_pd(bv_t5, bv_t6);
        bv_t7 = _mm_sub_pd(bv_t6, bv_t5);

        bv_s21 = _mm_add_pd(bv_s4, bv_s16);
        bv_t8 = _mm_sub_pd(bv_s5, bv_t2);
        bv_s23 = _mm_add_pd(bv_s21, bv_t8);
        bv_s24 = _mm_sub_pd(bv_s21, bv_t8);

        // Output point 6: X(5)
        v_out5 = _mm_add_pd(_mm_mul_pd(v_CRTM_16_1, bv_s19),
                               _mm_mul_pd(v_CRTM_16_2, bv_s23));
        curr_out = out_r + out_strides[5];
        STR_128_D(curr_out, v_out_stride, v_out5, is_contiguous_out);
        // Output point 22: X(21)
        v_out21 = _mm_sub_pd(_mm_mul_pd(v_CRTM_16_1, bv_s23),
                                _mm_mul_pd(v_CRTM_16_2, bv_s19));
        curr_out = out_r + out_strides[21];
        STR_128_D(curr_out, v_out_stride, v_out21, is_contiguous_out);
        // Output point 30: X(29)
        v_out29 = _mm_add_pd(_mm_mul_pd(v_CRTM_16_1, bv_t7),
                                _mm_mul_pd(v_CRTM_16_2, bv_s24));
        curr_out = out_r + out_strides[29];
        STR_128_D(curr_out, v_out_stride, v_out29, is_contiguous_out);
        // Output point 14: X(13)
        v_out13 = _mm_sub_pd(_mm_mul_pd(v_CRTM_16_1, bv_s24),
                                _mm_mul_pd(v_CRTM_16_2, bv_t7));
        curr_out = out_r + out_strides[13];
        STR_128_D(curr_out, v_out_stride, v_out13, is_contiguous_out);

        bv_t9 = _mm_add_pd(bv_s1, bv_t3);
        bv_s26 = _mm_add_pd(bv_s5, bv_t2);
        // Output point 2: X(1)
        v_out1 = _mm_mul_pd(v_CRTM_16_4, _mm_add_pd(bv_t9, bv_s26));
        curr_out = out_r + out_strides[1];
        STR_128_D(curr_out, v_out_stride, v_out1, is_contiguous_out);

        bv_t10 = _mm_sub_pd(bv_s4, bv_s16);
        bv_t11 = _mm_sub_pd(bv_s12, bv_t1);
        // Output point 18: X(17)
        v_out17 = _mm_mul_pd(v_CRTM_16_4, _mm_add_pd(bv_t10, bv_t11));
        curr_out = out_r + out_strides[17];
        STR_128_D(curr_out, v_out_stride, v_out17, is_contiguous_out);

        bv_s29 = _mm_sub_pd(bv_t9, bv_s26);
        bv_s30 = _mm_sub_pd(bv_t10, bv_t11);

        // Output point 10: X(9)
        v_out9 = _mm_mul_pd(v_CRTM_16_3, _mm_add_pd(bv_s29, bv_s30));
        curr_out = out_r + out_strides[9];
        STR_128_D(curr_out, v_out_stride, v_out9, is_contiguous_out);
        // Output point 26: X(25)
        v_out25 = _mm_mul_pd(v_CRTM_16_3, _mm_sub_pd(bv_s30, bv_s29));
        curr_out = out_r + out_strides[25];
        STR_128_D(curr_out, v_out_stride, v_out25, is_contiguous_out);

        bv_s31 = _mm_add_pd(bv_s2, bv_t4);
        bv_t12 = NEGATE_128_D(_mm_add_pd(bv_s3, bv_s14));

        bv_t13 = _mm_sub_pd(bv_s14, bv_s3);
        bv_s34 = _mm_sub_pd(bv_t4, bv_s2);

        bv_s35 = _mm_add_pd(bv_s11, bv_t0);
        bv_s36 = _mm_add_pd(bv_s10, bv_s7);

        bv_t18 = _mm_mul_pd(v_CRTM_16_5, _mm_add_pd(bv_s35, bv_s36));
        bv_t19 = _mm_mul_pd(v_CRTM_16_5, _mm_sub_pd(bv_s36, bv_s35));

        bv_s43 = _mm_add_pd(bv_t18, bv_s31);
        bv_s44 = _mm_add_pd(bv_t12, bv_t19);
        bv_t15 = _mm_sub_pd(bv_s31, bv_t18);
        bv_s46 = _mm_sub_pd(bv_t12, bv_t19);

        // Output point 4: X(3)
        v_out3 = _mm_sub_pd(_mm_mul_pd(v_CRTM_16_7, bv_s44),
                               _mm_mul_pd(v_CRTM_16_6, bv_s43));
        curr_out = out_r + out_strides[3];
        STR_128_D(curr_out, v_out_stride, v_out3, is_contiguous_out);
        // Output point 12: X(11)
        v_out11 = _mm_sub_pd(_mm_mul_pd(v_CRTM_16_9, bv_s46),
                                _mm_mul_pd(v_CRTM_16_8, bv_t15));
        curr_out = out_r + out_strides[11];
        STR_128_D(curr_out, v_out_stride, v_out11, is_contiguous_out);
        // Output point 28: X(27)
        v_out27 = _mm_add_pd(_mm_mul_pd(v_CRTM_16_9, bv_t15),
                                _mm_mul_pd(v_CRTM_16_8, bv_s46));
        curr_out = out_r + out_strides[27];
        STR_128_D(curr_out, v_out_stride, v_out27, is_contiguous_out);
        // Output point 20: X(19)
        v_out19 = _mm_add_pd(_mm_mul_pd(v_CRTM_16_6, bv_s44),
                                _mm_mul_pd(v_CRTM_16_7, bv_s43));
        curr_out = out_r + out_strides[19];
        STR_128_D(curr_out, v_out_stride, v_out19, is_contiguous_out);

        bv_s39 = _mm_sub_pd(bv_t0, bv_s11);
        bv_s40 = _mm_sub_pd(bv_s10, bv_s7);

        bv_t20 = _mm_mul_pd(v_CRTM_16_5, _mm_sub_pd(bv_s39, bv_s40));
        bv_t14 = _mm_mul_pd(v_CRTM_16_5, _mm_add_pd(bv_s39, bv_s40));

        bv_s47 = _mm_add_pd(bv_t13, bv_t20);
        bv_s48 = _mm_add_pd(bv_s34, bv_t14);
        bv_s49 = _mm_sub_pd(bv_t13, bv_t20);
        bv_s50 = _mm_sub_pd(bv_s34, bv_t14);

        // Output point 16: X(15)
        v_out15 = _mm_add_pd(_mm_mul_pd(v_CRTM_16_6, bv_s47),
                                _mm_mul_pd(v_CRTM_16_7, bv_s50));
        curr_out = out_r + out_strides[15];
        STR_128_D(curr_out, v_out_stride, v_out15, is_contiguous_out);
        // Output point 32: X(31)
        v_out31 = _mm_sub_pd(_mm_mul_pd(v_CRTM_16_7, bv_s47),
                                _mm_mul_pd(v_CRTM_16_6, bv_s50));
        curr_out = out_r + out_strides[31];
        STR_128_D(curr_out, v_out_stride, v_out31, is_contiguous_out);
        // Output point 8: X(7)
        v_out7 = _mm_add_pd(_mm_mul_pd(v_CRTM_16_9, bv_s48),
                               _mm_mul_pd(v_CRTM_16_8, bv_s49));
        curr_out = out_r + out_strides[7];
        STR_128_D(curr_out, v_out_stride, v_out7, is_contiguous_out);
        // Output point 24: X(23)
        v_out23 = _mm_sub_pd(_mm_mul_pd(v_CRTM_16_9, bv_s49),
                                _mm_mul_pd(v_CRTM_16_8, bv_s48));
        curr_out = out_r + out_strides[23];
        STR_128_D(curr_out, v_out_stride, v_out23, is_contiguous_out);

        in_cp += v_in_stride * NUM_SETS_REAL_128_D;
        in_r += v_in_dc_nyq_stride * NUM_SETS_REAL_128_D;
        out_r += v_out_stride * NUM_SETS_REAL_128_D;
    }
    // tail cases
    if (n & 1)
    {
        /* Standard DFT */
        FFTZ_DOUBLE a_in0, a_in1, a_in2, a_in3, a_in4, a_in5, a_in6, a_in7,
            a_in8, a_in9, a_in10, a_in11, a_in12, a_in13, a_in14, a_in15;
        FFTZ_DOUBLE a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_s9,
               a_s10, a_s11, a_s12, a_s13, a_s14, a_s15, a_s16, a_s17, a_s18,
               a_s19, a_s20, a_s21, a_s22, a_s23, a_t12, a_t16, a_t14, a_s27,
               a_t13, a_s29, a_t15, a_s31, a_s32, a_t17, a_t0, a_t1, a_t2, a_t3,
               a_t4, a_t5, a_t6, a_t7, a_t8, a_t9, a_t10, a_t11, a_s46, a_t18,
               a_t20, a_t19, a_s49, a_s50, a_s51, a_s52, a_s53, a_s54, a_s55,
               a_s56, a_s57, a_s58;

        curr_in = in_r;
        curr_out = out_r;
        // Input point 1: x(0)
        a_in0 = *in_r;
        // Input point 4: x(3)
        a_in1 = in_cp[in_strides[3]];
        // Input point 5: x(4)
        a_in2 = in_cp[in_strides[4]];
        // Input point 8: x(7)
        a_in3 = in_cp[in_strides[7]];
        // Input point 9: x(8)
        a_in4 = in_cp[in_strides[8]];
        // Input point 12: x(11)
        a_in5 = in_cp[in_strides[11]];
        // Input point 13: x(12)
        a_in6 = in_cp[in_strides[12]];
        // Input point 16: x(15)
        a_in7 = in_cp[in_strides[15]];
        // Input point 17: x(16)
        a_in8 = in_cp[in_strides[16]];
        // Input point 20: x(19)
        a_in9 = in_cp[in_strides[19]];
        // Input point 21: x(20)
        a_in10 = in_cp[in_strides[20]];
        // Input point 24: x(23)
        a_in11 = in_cp[in_strides[23]];
        // Input point 25: x(24)
        a_in12 = in_cp[in_strides[24]];
        // Input point 28: x(27)
        a_in13 = in_cp[in_strides[27]];
        // Input point 29: x(28)
        a_in14 = in_cp[in_strides[28]];
        // Input point 32: x(31)
        a_in15 = in_r[in_strides[31]];

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
        a_s27 = a_s0 + a_t12;
        a_s31 = a_t14 + a_s27;
        // Output point 1: X(0)
        *out_r = a_s31 + a_t13;
        // Output point 17: X(16)
        out_r[out_strides[16]] = a_s31 - a_t13;

        a_t15 = CRTM_16_4 * a_s21;
        a_s32 = a_s27 - a_t14;
        // Output point 9: X(8)
        out_r[out_strides[8]] = a_s32 - a_t15;
        // Output point 25: X(24)
        out_r[out_strides[24]] = a_s32 + a_t15;

        a_t16 = CRTM_16_4 * a_in8;
        a_t17 = CRTM_16_3 * a_s23;
        a_t0 = CRTM_16_1 * a_s17;
        a_t1 = CRTM_16_2 * a_s19;
        a_t3 = a_s1 - a_t16;
        a_s49 = a_t17 + a_t3;
        a_s50 = a_t0 - a_t1;
        // Output point 3: X(2)
        out_r[out_strides[2]] = a_s49 + a_s50;
        // Output point 19: X(18)
        out_r[out_strides[18]] = a_s49 - a_s50;

        a_s29 = a_s0 - a_t12;
        a_t2 = a_s1 + a_t16;
        a_t4 = CRTM_16_3 * a_s22;
        a_t5 = CRTM_16_1 * a_s18;
        a_t6 = CRTM_16_2 * a_s16;
        a_s51 = a_t2 - a_t4;
        a_s52 = a_t6 - a_t5;
        // Output point 7: X(6)
        out_r[out_strides[6]] = a_s51 + a_s52;
        // Output point 23: X(22)
        out_r[out_strides[22]] = a_s51 - a_s52;

        a_t9 = CRTM_16_1 * a_s16;
        a_t10 = CRTM_16_2 * a_s18;
        a_s55 = a_t2 + a_t4;
        a_s56 = a_t9 + a_t10;
        // Output point 15: X(14)
        out_r[out_strides[14]] = a_s55 - a_s56;
        // Output point 31: X(30)
        out_r[out_strides[30]] = a_s55 + a_s56;

        a_t7 = CRTM_16_1 * a_s19;
        a_t8 = CRTM_16_2 * a_s17;
        a_s53 = a_t3 - a_t17;
        a_s54 = a_t7 + a_t8;
        // Output point 11: X(10)
        out_r[out_strides[10]] = a_s53 - a_s54;
        // Output point 27: X(26)
        out_r[out_strides[26]] = a_s53 + a_s54;

        a_s46 = a_s15 - a_s20;
        a_t18 = CRTM_16_3 * a_s46;
        a_t19 = CRTM_16_4 * a_s9;
        a_s57 = a_s29 - a_t19;
        // Output point 5: X(4)
        out_r[out_strides[4]] = a_s57 + a_t18;
        // Output point 21: X(20)
        out_r[out_strides[20]] = a_s57 - a_t18;

        a_t11 = a_s15 + a_s20;
        a_t20 = CRTM_16_3 * a_t11;
        a_s58 = a_s29 + a_t19;
        // Output point 13: X(12)
        out_r[out_strides[12]] = a_s58 - a_t20;
        // Output point 29: X(28)
        out_r[out_strides[28]] = a_s58 + a_t20;

        /* Shifted DFT */
        FFTZ_DOUBLE b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7,
            b_in8, b_in9, b_in10, b_in11, b_in12, b_in13, b_in14, b_in15;
        FFTZ_DOUBLE b_s1, b_s2, b_s3, b_s4, b_s5, b_t0, b_s7, b_t1, b_t2, b_s10,
               b_s11, b_s12, b_t3, b_s14, b_t4, b_s16, b_t5, b_t6, b_s19, b_t7,
               b_s21, b_t8, b_s23, b_s24, b_t9, b_s26, b_t10, b_t11, b_s29,
               b_s30, b_s31, b_t12, b_t13, b_s34, b_s35, b_s36, b_t18, b_t19,
               b_s39, b_s40, b_t20, b_t14, b_s43, b_s44, b_t15, b_s46, b_s47,
               b_s48, b_s49, b_s50;

        //  Input point 2: x(1)
        b_in0 = in_cp[in_strides[1]];
        // Input point 3: x(2)
        b_in1 = in_cp[in_strides[2]];
        // Input point 6: x(5)
        b_in2 = in_cp[in_strides[5]];
        // Input point 7: x(6)
        b_in3 = in_cp[in_strides[6]];
        // Input point 10: x(9)
        b_in4 = in_cp[in_strides[9]];
        // Input point 11: x(10)
        b_in5 = in_cp[in_strides[10]];
        // Input point 14: x(13)
        b_in6 = in_cp[in_strides[13]];
        // Input point 15: x(14)
        b_in7 = in_cp[in_strides[14]];
        // Input point 18: x(17)
        b_in8 = in_cp[in_strides[17]];
        // Input point 19: x(18)
        b_in9 = in_cp[in_strides[18]];
        // Input point 22: x(21)
        b_in10 = in_cp[in_strides[21]];
        // Input point 23: x(22)
        b_in11 = in_cp[in_strides[22]];
        // Input point 26: x(25)
        b_in12 = in_cp[in_strides[25]];
        // Input point 27: x(26)
        b_in13 = in_cp[in_strides[26]];
        // Input point 30: x(29)
        b_in14 = in_cp[in_strides[29]];
        // Input point 31: x(30)
        b_in15 = in_cp[in_strides[30]];

        b_s1 = b_in14 + b_in0;
        b_s2 = b_in14 - b_in0;
        b_s3 = b_in15 + b_in1;
        b_s4 = b_in15 - b_in1;
        b_s5 = b_in12 + b_in2;
        b_t0 = b_in12 - b_in2;
        b_s7 = b_in13 + b_in3;
        b_t1 = b_in13 - b_in3;
        b_t2 = b_in10 + b_in4;
        b_s10 = b_in10 - b_in4;
        b_s11 = b_in11 + b_in5;
        b_s12 = b_in11 - b_in5;
        b_t3 = b_in8 + b_in6;
        b_s14 = b_in8 - b_in6;
        b_t4 = b_in9 + b_in7;
        b_s16 = b_in9 - b_in7;

        b_t5 = b_s1 - b_t3;
        b_t6 = b_s12 + b_t1;
        b_s19 = b_t5 + b_t6;
        b_t7 = b_t6 - b_t5;

        b_s21 = b_s4 + b_s16;
        b_t8 = b_s5 - b_t2;
        b_s23 = b_s21 + b_t8;
        b_s24 = b_s21 - b_t8;

        // Output point 6: X(5)
        out_r[out_strides[5]] = (CRTM_16_1 * b_s19) + (CRTM_16_2 * b_s23);
        // Output point 22: X(21)
        out_r[out_strides[21]] = (CRTM_16_1 * b_s23) - (CRTM_16_2 * b_s19);
        // Output point 30: X(29)
        out_r[out_strides[29]] = (CRTM_16_1 * b_t7) + (CRTM_16_2 * b_s24);
        // Output point 14: X(13)
        out_r[out_strides[13]] = (CRTM_16_1 * b_s24) - (CRTM_16_2 * b_t7);

        b_t9 = b_s1 + b_t3;
        b_s26 = b_s5 + b_t2;
        // Output point 2: X(1)
        out_r[out_strides[1]] = CRTM_16_4 * (b_t9 + b_s26);

        b_t10 = b_s4 - b_s16;
        b_t11 = b_s12 - b_t1;
        // Output point 18: X(17)
        out_r[out_strides[17]] = CRTM_16_4 * (b_t10 + b_t11);

        b_s29 = b_t9 - b_s26;
        b_s30 = b_t10 - b_t11;

        // Output point 10: X(9)
        out_r[out_strides[9]] = CRTM_16_3 * (b_s29 + b_s30);
        // Output point 26: X(25)
        out_r[out_strides[25]] = CRTM_16_3 * (b_s30 - b_s29);

        b_s31 = b_s2 + b_t4;
        b_t12 = -b_s3 - b_s14;

        b_t13 = b_s14 - b_s3;
        b_s34 = b_t4 - b_s2;

        b_s35 = b_s11 + b_t0;
        b_s36 = b_s10 + b_s7;

        b_t18 = CRTM_16_5 * (b_s35 + b_s36);
        b_t19 = CRTM_16_5 * (b_s36 - b_s35);

        b_s43 = b_t18 + b_s31;
        b_s44 = b_t12 + b_t19;
        b_t15 = b_s31 - b_t18;
        b_s46 = b_t12 - b_t19;

        // Output point 4: X(3)
        out_r[out_strides[3]] = (CRTM_16_7 * b_s44) - (CRTM_16_6 * b_s43);
        // Output point 12: X(11)
        out_r[out_strides[11]] = (CRTM_16_9 * b_s46) - (CRTM_16_8 * b_t15);
        // Output point 28: X(27)
        out_r[out_strides[27]] = (CRTM_16_9 * b_t15) + (CRTM_16_8 * b_s46);
        // Output point 20: X(19)
        out_r[out_strides[19]] = (CRTM_16_6 * b_s44) + (CRTM_16_7 * b_s43);

        b_s39 = b_t0 - b_s11;
        b_s40 = b_s10 - b_s7;

        b_t20 = CRTM_16_5 * (b_s39 - b_s40);
        b_t14 = CRTM_16_5 * (b_s39 + b_s40);

        b_s47 = b_t13 + b_t20;
        b_s48 = b_s34 + b_t14;
        b_s49 = b_t13 - b_t20;
        b_s50 = b_s34 - b_t14;

        // Output point 16: X(15)
        out_r[out_strides[15]] = (CRTM_16_6 * b_s47) + (CRTM_16_7 * b_s50);
        // Output point 32: X(31)
        out_r[out_strides[31]] = (CRTM_16_7 * b_s47) - (CRTM_16_6 * b_s50);
        // Output point 8: X(7)
        out_r[out_strides[7]] = (CRTM_16_9 * b_s48) + (CRTM_16_8 * b_s49);
        // Output point 24: X(23)
        out_r[out_strides[23]] = (CRTM_16_9 * b_s49) - (CRTM_16_8 * b_s48);
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hcf_rfft16avx128(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hcf_rfft16avx128_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft16avx128_fp64_fwd;
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
            return r2hcf_rfft16avx128_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft16avx128_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
