// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft14avx512.c
 *
 *  @brief Radix-14 r2hc_fused Real-FFT kernel with with AVX-512 operations
 *  using x86 SIMD intrinsics.
 *
 *  This file contains the DIT radix-14 real-to-halfcomplex fused of two
 *  different implementations (Standard DFT and Shifted DFT that differs in DFT
 *  weight matrix) using x86 SIMD intrinsics for single-precision  and
 *  double-precision inputs
 *
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/r2hc_simd_avx512.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                {{{0, 72, 122, 688, 386, 169},
                                                  {0, 76, 123, 688, 490, 170}},
                                                 {{0, 72, 122, 344, 26,  169},
                                                  {0, 76, 123, 344, 26,  170}}};

ops_cycles_t get_ops_cnt_r2hcf_rfft14avx512(FFTZ_UINT8 precision,
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

static FFTZ_VOID r2hcf_rfft14avx512_fp32_fwd(FFTZ_VOID *in_real,
                                             FFTZ_VOID *in_imag,
                                             FFTZ_VOID *out_real,
                                             FFTZ_VOID *out_imag, FFTZ_INTP n,
                                             aoclfftz_strides_t *strides,
                                             FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_14_1 =
        0.900968867902419126236102319507445051165919162f;
    const FFTZ_FLOAT CRTM_14_2 =
        0.433883739117558120475768332848358754609990728f;
    const FFTZ_FLOAT CRTM_14_3 =
        0.623489801858733530525004884004239810632274731f;
    const FFTZ_FLOAT CRTM_14_4 =
        0.781831482468029808708444526674057750232334519f;
    const FFTZ_FLOAT CRTM_14_5 =
        0.222520933956314404288902564496794759466355569f;
    const FFTZ_FLOAT CRTM_14_6 =
        0.974927912181823607018131682993931217232785801f;

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
    FFTZ_INTP N = n / NUM_SETS_REAL_512_S;
    FFTZ_INTP remaining_sets = n % NUM_SETS_REAL_512_S;

    __m512 v_CRTM_14_1 = _mm512_set1_ps(CRTM_14_1);
    __m512 v_CRTM_14_2 = _mm512_set1_ps(CRTM_14_2);
    __m512 v_CRTM_14_3 = _mm512_set1_ps(CRTM_14_3);
    __m512 v_CRTM_14_4 = _mm512_set1_ps(CRTM_14_4);
    __m512 v_CRTM_14_5 = _mm512_set1_ps(CRTM_14_5);
    __m512 v_CRTM_14_6 = _mm512_set1_ps(CRTM_14_6);

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m512 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
               av_in8, av_in9, av_in10, av_in11, av_in12, av_in13;
        __m512 av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8, av_s9,
               av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16, av_s17,
               av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24, av_s25,
               av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32, av_s33,
               av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40, av_s41,
               av_s42, av_s43, av_s44, av_s45, av_s46, av_s47, av_s48;
        __m512 av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8, av_t9,
               av_t10, av_t11, av_t12, av_t13, av_t14, av_t15, av_t16, av_t17,
               av_t18, av_t19, av_t20, av_t21, av_t22, av_t23, av_t24, av_t25,
               av_t26, av_t27, av_t28, av_t29, av_t30, av_t31, av_t32, av_t33,
               av_t34, av_t35, av_t36;
        __m512 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21,
               v_out22, v_out23, v_out24, v_out25, v_out26, v_out27;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_512_S(curr_in, v_in_stride, av_in0);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_512_S(curr_in, v_in_stride, av_in1);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_512_S(curr_in, v_in_stride, av_in2);
        // Input point 7: x(6)
        curr_in = in + in_strides[6];
        LDR_512_S(curr_in, v_in_stride, av_in3);
        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDR_512_S(curr_in, v_in_stride, av_in4);
        // Input point 11: x(10)
        curr_in = in + in_strides[10];
        LDR_512_S(curr_in, v_in_stride, av_in5);
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDR_512_S(curr_in, v_in_stride, av_in6);
        // Input point 15: x(14)
        curr_in = in + in_strides[14];
        LDR_512_S(curr_in, v_in_stride, av_in7);
        // Input point 17: x(16)
        curr_in = in + in_strides[16];
        LDR_512_S(curr_in, v_in_stride, av_in8);
        // Input point 19: x(18)
        curr_in = in + in_strides[18];
        LDR_512_S(curr_in, v_in_stride, av_in9);
        // Input point 21: x(20)
        curr_in = in + in_strides[20];
        LDR_512_S(curr_in, v_in_stride, av_in10);
        // Input point 23: x(22)
        curr_in = in + in_strides[22];
        LDR_512_S(curr_in, v_in_stride, av_in11);
        // Input point 25: x(24)
        curr_in = in + in_strides[24];
        LDR_512_S(curr_in, v_in_stride, av_in12);
        // Input point 27: x(26)
        curr_in = in + in_strides[26];
        LDR_512_S(curr_in, v_in_stride, av_in13);

        av_s1 = _mm512_sub_ps(av_in0, av_in7);
        av_s2 = _mm512_add_ps(av_in0, av_in7);
        av_s3 = _mm512_sub_ps(av_in13, av_in1);
        av_s4 = _mm512_add_ps(av_in13, av_in1);
        av_s5 = _mm512_sub_ps(av_in12, av_in2);
        av_s6 = _mm512_add_ps(av_in12, av_in2);
        av_s7 = _mm512_sub_ps(av_in11, av_in3);
        av_s8 = _mm512_add_ps(av_in11, av_in3);
        av_s9 = _mm512_sub_ps(av_in10, av_in4);
        av_s10 = _mm512_add_ps(av_in10, av_in4);
        av_s11 = _mm512_sub_ps(av_in9, av_in5);
        av_s12 = _mm512_add_ps(av_in9, av_in5);
        av_s13 = _mm512_sub_ps(av_in8, av_in6);
        av_s14 = _mm512_add_ps(av_in8, av_in6);

        av_s15 = _mm512_add_ps(av_s4, av_s14);
        av_s16 = _mm512_add_ps(av_s6, av_s12);
        av_s17 = _mm512_add_ps(av_s8, av_s10);

        av_s18 = _mm512_sub_ps(av_s14, av_s4);
        av_s19 = _mm512_sub_ps(av_s6, av_s12);
        av_s20 = _mm512_sub_ps(av_s10, av_s8);
        av_s27 = _mm512_add_ps(av_s2, av_s15);
        av_s28 = _mm512_add_ps(av_s16, av_s17);
        av_s29 = _mm512_add_ps(av_s1, av_s18);
        av_s30 = _mm512_add_ps(av_s19, av_s20);
        // Output pt 1: X(0)
        v_out0 = _mm512_add_ps(av_s27, av_s28);
        STR_512_S(curr_out, v_out_stride, v_out0);
        // Output pt 28: X(27)
        v_out27 = _mm512_add_ps(av_s29, av_s30);
        curr_out = out + out_strides[27];
        STR_512_S(curr_out, v_out_stride, v_out27);

        av_t1 = _mm512_mul_ps(v_CRTM_14_1, av_s18);
        av_t2 = _mm512_mul_ps(v_CRTM_14_3, av_s19);
        av_t3 = _mm512_mul_ps(v_CRTM_14_5, av_s20);
        av_s31 = _mm512_sub_ps(av_s1, av_t1);
        av_s32 = _mm512_sub_ps(av_t2, av_t3);
        // Output point 4: X(3)
        v_out3 = _mm512_add_ps(av_s31, av_s32);

        av_s21 = _mm512_add_ps(av_s3, av_s13);
        av_s22 = _mm512_add_ps(av_s5, av_s11);
        av_s23 = _mm512_add_ps(av_s7, av_s9);

        av_t4 = _mm512_mul_ps(v_CRTM_14_2, av_s21);
        av_t5 = _mm512_mul_ps(v_CRTM_14_4, av_s22);
        av_t6 = _mm512_mul_ps(v_CRTM_14_6, av_s23);
        av_s33 = _mm512_add_ps(av_t4, av_t5);
        // Output point 5: X(4)
        v_out4 = _mm512_add_ps(av_s33, av_t6);
        curr_out = out + out_strides[3];
        STRI_2x512_S(curr_out, v_out_stride, v_out3, v_out4);

        av_t7 = _mm512_mul_ps(v_CRTM_14_1, av_s17);
        av_t8 = _mm512_mul_ps(v_CRTM_14_3, av_s15);
        av_t9 = _mm512_mul_ps(v_CRTM_14_5, av_s16);
        av_s34 = _mm512_sub_ps(av_s2, av_t7);
        av_s35 = _mm512_sub_ps(av_t8, av_t9);
        // Output point 8: X(7)
        v_out7 = _mm512_add_ps(av_s34, av_s35);

        av_s24 = _mm512_sub_ps(av_s3, av_s13);
        av_s25 = _mm512_sub_ps(av_s5, av_s11);
        av_s26 = _mm512_sub_ps(av_s7, av_s9);

        av_t10 = _mm512_mul_ps(v_CRTM_14_2, av_s26);
        av_t11 = _mm512_mul_ps(v_CRTM_14_4, av_s24);
        av_t12 = _mm512_mul_ps(v_CRTM_14_6, av_s25);
        av_s36 = _mm512_add_ps(av_t10, av_t11);
        // Output point 9: X(8)
        v_out8 = _mm512_add_ps(av_s36, av_t12);
        curr_out = out + out_strides[7];
        STRI_2x512_S(curr_out, v_out_stride, v_out7, v_out8);

        av_t13 = _mm512_mul_ps(v_CRTM_14_1, av_s19);
        av_t14 = _mm512_mul_ps(v_CRTM_14_3, av_s20);
        av_t15 = _mm512_mul_ps(v_CRTM_14_5, av_s18);
        av_s37 = _mm512_sub_ps(av_s1, av_t13);
        av_s38 = _mm512_sub_ps(av_t14, av_t15);
        // Output point 12: X(11)
        v_out11 = _mm512_add_ps(av_s37, av_s38);

        av_t16 = _mm512_mul_ps(v_CRTM_14_2, av_s22);
        av_t17 = _mm512_mul_ps(v_CRTM_14_4, av_s23);
        av_t18 = _mm512_mul_ps(v_CRTM_14_6, av_s21);
        av_s39 = _mm512_sub_ps(av_t16, av_t17);
        // Output point 13: X(12)
        v_out12 = _mm512_add_ps(av_s39, av_t18);
        curr_out = out + out_strides[11];
        STRI_2x512_S(curr_out, v_out_stride, v_out11, v_out12);

        av_t19 = _mm512_mul_ps(v_CRTM_14_1, av_s16);
        av_t20 = _mm512_mul_ps(v_CRTM_14_3, av_s17);
        av_t21 = _mm512_mul_ps(v_CRTM_14_5, av_s15);
        av_s40 = _mm512_sub_ps(av_s2, av_t19);
        av_s41 = _mm512_sub_ps(av_t20, av_t21);
        // Output point 16: X(15)
        v_out15 = _mm512_add_ps(av_s40, av_s41);

        av_t22 = _mm512_mul_ps(v_CRTM_14_2, av_s25);
        av_t23 = _mm512_mul_ps(v_CRTM_14_4, av_s26);
        av_t24 = _mm512_mul_ps(v_CRTM_14_6, av_s24);
        av_s42 = _mm512_sub_ps(av_t24, av_t22);
        // Output point 17: X(16)
        v_out16 = _mm512_sub_ps(av_s42, av_t23);
        curr_out = out + out_strides[15];
        STRI_2x512_S(curr_out, v_out_stride, v_out15, v_out16);

        av_t25 = _mm512_mul_ps(v_CRTM_14_1, av_s20);
        av_t26 = _mm512_mul_ps(v_CRTM_14_3, av_s18);
        av_t27 = _mm512_mul_ps(v_CRTM_14_5, av_s19);
        av_s43 = _mm512_sub_ps(av_s1, av_t25);
        av_s44 = _mm512_sub_ps(av_t26, av_t27);
        // Output point 20: X(19)
        v_out19 = _mm512_add_ps(av_s43, av_s44);

        av_t28 = _mm512_mul_ps(v_CRTM_14_2, av_s23);
        av_t29 = _mm512_mul_ps(v_CRTM_14_4, av_s21);
        av_t30 = _mm512_mul_ps(v_CRTM_14_6, av_s22);
        av_s45 = _mm512_add_ps(av_t28, av_t29);
        // Output point 21: X(20)
        v_out20 = _mm512_sub_ps(av_s45, av_t30);
        curr_out = out + out_strides[19];
        STRI_2x512_S(curr_out, v_out_stride, v_out19, v_out20);

        av_t31 = _mm512_mul_ps(v_CRTM_14_1, av_s15);
        av_t32 = _mm512_mul_ps(v_CRTM_14_3, av_s16);
        av_t33 = _mm512_mul_ps(v_CRTM_14_5, av_s17);
        av_s46 = _mm512_sub_ps(av_s2, av_t31);
        av_s47 = _mm512_sub_ps(av_t32, av_t33);
        // Output point 24: X(23)
        v_out23 = _mm512_add_ps(av_s46, av_s47);

        av_t34 = _mm512_mul_ps(v_CRTM_14_2, av_s24);
        av_t35 = _mm512_mul_ps(v_CRTM_14_4, av_s25);
        av_t36 = _mm512_mul_ps(v_CRTM_14_6, av_s26);
        av_s48 = _mm512_sub_ps(av_t34, av_t35);
        // Output point 25: X(24)
        v_out24 = _mm512_add_ps(av_s48, av_t36);
        curr_out = out + out_strides[23];
        STRI_2x512_S(curr_out, v_out_stride, v_out23, v_out24);

        /* Shifted DFT */
        __m512 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8, bv_in9, bv_in10, bv_in11, bv_in12, bv_in13;
        __m512 bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
               bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17,
               bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24, bv_s25,
               bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32, bv_s33,
               bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40, bv_s41,
               bv_s42, bv_s43, bv_s44, bv_s45, bv_s46;
        __m512 bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
               bv_t10, bv_t11, bv_t12, bv_t13, bv_t14, bv_t15, bv_t16, bv_t17,
               bv_t18, bv_t19, bv_t20, bv_t21, bv_t22, bv_t23, bv_t24, bv_t25,
               bv_t26, bv_t27, bv_t28, bv_t29, bv_t30, bv_t31, bv_t32, bv_t33,
               bv_t34, bv_t35, bv_t36;

        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_512_S(curr_in, v_in_stride, bv_in0);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_512_S(curr_in, v_in_stride, bv_in1);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_512_S(curr_in, v_in_stride, bv_in2);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_512_S(curr_in, v_in_stride, bv_in3);
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDR_512_S(curr_in, v_in_stride, bv_in4);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_512_S(curr_in, v_in_stride, bv_in5);
        // Input point 14: x(13)
        curr_in = in + in_strides[13];
        LDR_512_S(curr_in, v_in_stride, bv_in6);
        // Input point 16: x(15)
        curr_in = in + in_strides[15];
        LDR_512_S(curr_in, v_in_stride, bv_in7);
        // Input point 18: x(17)
        curr_in = in + in_strides[17];
        LDR_512_S(curr_in, v_in_stride, bv_in8);
        // Input point 20: x(19)
        curr_in = in + in_strides[19];
        LDR_512_S(curr_in, v_in_stride, bv_in9);
        // Input point 22: x(21)
        curr_in = in + in_strides[21];
        LDR_512_S(curr_in, v_in_stride, bv_in10);
        // Input point 24: x(23)
        curr_in = in + in_strides[23];
        LDR_512_S(curr_in, v_in_stride, bv_in11);
        // Input point 26: x(25)
        curr_in = in + in_strides[25];
        LDR_512_S(curr_in, v_in_stride, bv_in12);
        // Input point 28: x(27)
        curr_in = in + in_strides[27];
        LDR_512_S(curr_in, v_in_stride, bv_in13);

        bv_s1 = _mm512_add_ps(bv_in1, bv_in13);
        bv_s2 = _mm512_sub_ps(bv_in1, bv_in13);
        bv_s3 = _mm512_add_ps(bv_in2, bv_in12);
        bv_s4 = _mm512_sub_ps(bv_in2, bv_in12);
        bv_s5 = _mm512_add_ps(bv_in3, bv_in11);
        bv_s6 = _mm512_sub_ps(bv_in3, bv_in11);
        bv_s7 = _mm512_add_ps(bv_in4, bv_in10);
        bv_s8 = _mm512_sub_ps(bv_in4, bv_in10);
        bv_s9 = _mm512_add_ps(bv_in5, bv_in9);
        bv_s10 = _mm512_sub_ps(bv_in5, bv_in9);
        bv_s11 = _mm512_add_ps(bv_in6, bv_in8);
        bv_s12 = _mm512_sub_ps(bv_in6, bv_in8);

        bv_t1 = _mm512_mul_ps(v_CRTM_14_5, bv_s12);
        bv_t2 = _mm512_mul_ps(v_CRTM_14_1, bv_s4);
        bv_t3 = _mm512_mul_ps(v_CRTM_14_3, bv_s8);
        bv_t4 = _mm512_mul_ps(v_CRTM_14_6, bv_s2);
        bv_t5 = _mm512_mul_ps(v_CRTM_14_2, bv_s10);
        bv_t6 = _mm512_mul_ps(v_CRTM_14_4, bv_s6);

        bv_t7 = _mm512_mul_ps(v_CRTM_14_5, bv_s1);
        bv_t8 = _mm512_mul_ps(v_CRTM_14_1, bv_s9);
        bv_t9 = _mm512_mul_ps(v_CRTM_14_3, bv_s5);
        bv_t10 = _mm512_mul_ps(v_CRTM_14_6, bv_s11);
        bv_t11 = _mm512_mul_ps(v_CRTM_14_2, bv_s3);
        bv_t12 = _mm512_mul_ps(v_CRTM_14_4, bv_s7);

        bv_t13 = _mm512_mul_ps(v_CRTM_14_5, bv_s4);
        bv_t14 = _mm512_mul_ps(v_CRTM_14_1, bv_s8);
        bv_t15 = _mm512_mul_ps(v_CRTM_14_3, bv_s12);
        bv_t16 = _mm512_mul_ps(v_CRTM_14_6, bv_s10);
        bv_t17 = _mm512_mul_ps(v_CRTM_14_2, bv_s6);
        bv_t18 = _mm512_mul_ps(v_CRTM_14_4, bv_s2);

        bv_t19 = _mm512_mul_ps(v_CRTM_14_5, bv_s9);
        bv_t20 = _mm512_mul_ps(v_CRTM_14_1, bv_s5);
        bv_t21 = _mm512_mul_ps(v_CRTM_14_3, bv_s1);
        bv_t22 = _mm512_mul_ps(v_CRTM_14_6, bv_s3);
        bv_t23 = _mm512_mul_ps(v_CRTM_14_2, bv_s7);
        bv_t24 = _mm512_mul_ps(v_CRTM_14_4, bv_s11);

        bv_t25 = _mm512_mul_ps(v_CRTM_14_5, bv_s8);
        bv_t26 = _mm512_mul_ps(v_CRTM_14_1, bv_s12);
        bv_t27 = _mm512_mul_ps(v_CRTM_14_3, bv_s4);
        bv_t28 = _mm512_mul_ps(v_CRTM_14_6, bv_s6);
        bv_t29 = _mm512_mul_ps(v_CRTM_14_2, bv_s2);
        bv_t30 = _mm512_mul_ps(v_CRTM_14_4, bv_s10);

        bv_t31 = _mm512_mul_ps(v_CRTM_14_5, bv_s5);
        bv_t32 = _mm512_mul_ps(v_CRTM_14_1, bv_s1);
        bv_t33 = _mm512_mul_ps(v_CRTM_14_3, bv_s9);
        bv_t34 = _mm512_mul_ps(v_CRTM_14_6, bv_s7);
        bv_t35 = _mm512_mul_ps(v_CRTM_14_2, bv_s11);
        bv_t36 = _mm512_mul_ps(v_CRTM_14_4, bv_s3);

        bv_s13 = _mm512_add_ps(bv_in0, bv_t1);
        bv_s14 = _mm512_add_ps(bv_t2, bv_t3);
        bv_s15 = _mm512_add_ps(bv_s13, bv_s14);
        bv_s16 = _mm512_add_ps(bv_t4, bv_t5);
        bv_s17 = _mm512_add_ps(bv_s16, bv_t6);
        // Output point 2: X(1)
        v_out1 = _mm512_add_ps(bv_s15, bv_s17);
        // Output point 26: X(25)
        v_out25 = _mm512_sub_ps(bv_s15, bv_s17);

        bv_s18 = _mm512_add_ps(bv_in7, bv_t7);
        bv_s19 = _mm512_add_ps(bv_t8, bv_t9);
        bv_s20 = _mm512_add_ps(bv_s18, bv_s19);
        bv_s21 = _mm512_add_ps(bv_t10, bv_t11);
        bv_s22 = _mm512_add_ps(bv_s21, bv_t12);
        // Output point 3: X(2)
        v_out2 = NEGATE_512_S(_mm512_add_ps(bv_s20, bv_s22));
        curr_out = out + out_strides[1];
        STRI_2x512_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 27: X(26)
        v_out26 = _mm512_sub_ps(bv_s22, bv_s20);
        curr_out = out + out_strides[25];
        STRI_2x512_S(curr_out, v_out_stride, v_out25, v_out26);

        bv_s23 = _mm512_add_ps(bv_in0, bv_t13);
        bv_s24 = _mm512_add_ps(bv_t14, bv_t15);
        bv_s25 = _mm512_sub_ps(bv_s23, bv_s24);
        bv_s26 = _mm512_add_ps(bv_t16, bv_t17);
        bv_s27 = _mm512_sub_ps(bv_t18, bv_s26);
        // Output point 6: X(5)
        v_out5 = _mm512_add_ps(bv_s25, bv_s27);
        // Output point 22: X(21)
        v_out21 = _mm512_sub_ps(bv_s25, bv_s27);

        bv_s28 = _mm512_add_ps(bv_in7, bv_t19);
        bv_s29 = _mm512_add_ps(bv_t20, bv_t21);
        bv_s30 = _mm512_sub_ps(bv_s28, bv_s29);
        bv_s31 = _mm512_add_ps(bv_t22, bv_t23);
        bv_s32 = _mm512_sub_ps(bv_t24, bv_s31);
        // Output point 7: X(6)
        v_out6 = _mm512_add_ps(bv_s30, bv_s32);
        curr_out = out + out_strides[5];
        STRI_2x512_S(curr_out, v_out_stride, v_out5, v_out6);
        // Output point 23: X(22)
        v_out22 = _mm512_sub_ps(bv_s30, bv_s32);
        curr_out = out + out_strides[21];
        STRI_2x512_S(curr_out, v_out_stride, v_out21, v_out22);

        bv_s33 = _mm512_sub_ps(bv_in0, bv_t25);
        bv_s34 = _mm512_sub_ps(bv_t26, bv_t27);
        bv_s35 = _mm512_add_ps(bv_s33, bv_s34);
        bv_s36 = _mm512_sub_ps(bv_t29, bv_t28);
        bv_s37 = _mm512_add_ps(bv_s36, bv_t30);
        // Output point 10: X(9)
        v_out9 = _mm512_add_ps(bv_s35, bv_s37);
        // Output point 18: X(17)
        v_out17 = _mm512_sub_ps(bv_s35, bv_s37);

        bv_s38 = _mm512_sub_ps(bv_t31, bv_in7);
        bv_s39 = _mm512_sub_ps(bv_t33, bv_t32);
        bv_s40 = _mm512_add_ps(bv_s38, bv_s39);
        bv_s41 = _mm512_sub_ps(bv_t34, bv_t35);
        bv_s42 = _mm512_sub_ps(bv_s41, bv_t36);
        // Output point 11: X(10)
        v_out10 = _mm512_add_ps(bv_s40, bv_s42);
        curr_out = out + out_strides[9];
        STRI_2x512_S(curr_out, v_out_stride, v_out9, v_out10);
        // Output point 19: X(18)
        v_out18 = _mm512_sub_ps(bv_s40, bv_s42);
        curr_out = out + out_strides[17];
        STRI_2x512_S(curr_out, v_out_stride, v_out17, v_out18);

        bv_s43 = _mm512_add_ps(bv_in0, bv_s8);
        bv_s44 = _mm512_add_ps(bv_s4, bv_s12);
        bv_s45 = _mm512_add_ps(bv_in7, bv_s5);
        bv_s46 = _mm512_add_ps(bv_s1, bv_s9);
        // Output point 14: X(13)
        v_out13 = _mm512_sub_ps(bv_s43, bv_s44);
        // Output point 15: X(14)
        v_out14 = _mm512_sub_ps(bv_s45, bv_s46);
        curr_out = out + out_strides[13];
        STRI_2x512_S(curr_out, v_out_stride, v_out13, v_out14);

        in = in + (v_in_stride << 4);
        out = out + (v_out_stride << 4);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_256_S)
    {
        /* Standard DFT */
        __m256 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
               av_in8, av_in9, av_in10, av_in11, av_in12, av_in13;
        __m256 av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8, av_s9,
               av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16, av_s17,
               av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24, av_s25,
               av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32, av_s33,
               av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40, av_s41,
               av_s42, av_s43, av_s44, av_s45, av_s46, av_s47, av_s48;
        __m256 av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8, av_t9,
               av_t10, av_t11, av_t12, av_t13, av_t14, av_t15, av_t16, av_t17,
               av_t18, av_t19, av_t20, av_t21, av_t22, av_t23, av_t24, av_t25,
               av_t26, av_t27, av_t28, av_t29, av_t30, av_t31, av_t32, av_t33,
               av_t34, av_t35, av_t36;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21,
               v_out22, v_out23, v_out24, v_out25, v_out26, v_out27;

        curr_in = in;
        curr_out = out;

        __m256 v256_CRTM_14_1 = _mm512_castps512_ps256(v_CRTM_14_1);
        __m256 v256_CRTM_14_2 = _mm512_castps512_ps256(v_CRTM_14_2);
        __m256 v256_CRTM_14_3 = _mm512_castps512_ps256(v_CRTM_14_3);
        __m256 v256_CRTM_14_4 = _mm512_castps512_ps256(v_CRTM_14_4);
        __m256 v256_CRTM_14_5 = _mm512_castps512_ps256(v_CRTM_14_5);
        __m256 v256_CRTM_14_6 = _mm512_castps512_ps256(v_CRTM_14_6);

        // Input point 1: x(0)
        LDR_256_S(curr_in, v_in_stride, av_in0);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_256_S(curr_in, v_in_stride, av_in1);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_256_S(curr_in, v_in_stride, av_in2);
        // Input point 7: x(6)
        curr_in = in + in_strides[6];
        LDR_256_S(curr_in, v_in_stride, av_in3);
        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDR_256_S(curr_in, v_in_stride, av_in4);
        // Input point 11: x(10)
        curr_in = in + in_strides[10];
        LDR_256_S(curr_in, v_in_stride, av_in5);
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDR_256_S(curr_in, v_in_stride, av_in6);
        // Input point 15: x(14)
        curr_in = in + in_strides[14];
        LDR_256_S(curr_in, v_in_stride, av_in7);
        // Input point 17: x(16)
        curr_in = in + in_strides[16];
        LDR_256_S(curr_in, v_in_stride, av_in8);
        // Input point 19: x(18)
        curr_in = in + in_strides[18];
        LDR_256_S(curr_in, v_in_stride, av_in9);
        // Input point 21: x(20)
        curr_in = in + in_strides[20];
        LDR_256_S(curr_in, v_in_stride, av_in10);
        // Input point 23: x(22)
        curr_in = in + in_strides[22];
        LDR_256_S(curr_in, v_in_stride, av_in11);
        // Input point 25: x(24)
        curr_in = in + in_strides[24];
        LDR_256_S(curr_in, v_in_stride, av_in12);
        // Input point 27: x(26)
        curr_in = in + in_strides[26];
        LDR_256_S(curr_in, v_in_stride, av_in13);

        av_s1 = _mm256_sub_ps(av_in0, av_in7);
        av_s2 = _mm256_add_ps(av_in0, av_in7);
        av_s3 = _mm256_sub_ps(av_in13, av_in1);
        av_s4 = _mm256_add_ps(av_in13, av_in1);
        av_s5 = _mm256_sub_ps(av_in12, av_in2);
        av_s6 = _mm256_add_ps(av_in12, av_in2);
        av_s7 = _mm256_sub_ps(av_in11, av_in3);
        av_s8 = _mm256_add_ps(av_in11, av_in3);
        av_s9 = _mm256_sub_ps(av_in10, av_in4);
        av_s10 = _mm256_add_ps(av_in10, av_in4);
        av_s11 = _mm256_sub_ps(av_in9, av_in5);
        av_s12 = _mm256_add_ps(av_in9, av_in5);
        av_s13 = _mm256_sub_ps(av_in8, av_in6);
        av_s14 = _mm256_add_ps(av_in8, av_in6);

        av_s15 = _mm256_add_ps(av_s4, av_s14);
        av_s16 = _mm256_add_ps(av_s6, av_s12);
        av_s17 = _mm256_add_ps(av_s8, av_s10);

        av_s18 = _mm256_sub_ps(av_s14, av_s4);
        av_s19 = _mm256_sub_ps(av_s6, av_s12);
        av_s20 = _mm256_sub_ps(av_s10, av_s8);
        av_s27 = _mm256_add_ps(av_s2, av_s15);
        av_s28 = _mm256_add_ps(av_s16, av_s17);
        av_s29 = _mm256_add_ps(av_s1, av_s18);
        av_s30 = _mm256_add_ps(av_s19, av_s20);
        // Output pt 1: X(0)
        v_out0 = _mm256_add_ps(av_s27, av_s28);
        STR_256_S(curr_out, v_out_stride, v_out0);
        // Output pt 28: X(27)
        v_out27 = _mm256_add_ps(av_s29, av_s30);
        curr_out = out + out_strides[27];
        STR_256_S(curr_out, v_out_stride, v_out27);

        av_t1 = _mm256_mul_ps(v256_CRTM_14_1, av_s18);
        av_t2 = _mm256_mul_ps(v256_CRTM_14_3, av_s19);
        av_t3 = _mm256_mul_ps(v256_CRTM_14_5, av_s20);
        av_s31 = _mm256_sub_ps(av_s1, av_t1);
        av_s32 = _mm256_sub_ps(av_t2, av_t3);
        // Output point 4: X(3)
        v_out3 = _mm256_add_ps(av_s31, av_s32);

        av_s21 = _mm256_add_ps(av_s3, av_s13);
        av_s22 = _mm256_add_ps(av_s5, av_s11);
        av_s23 = _mm256_add_ps(av_s7, av_s9);

        av_t4 = _mm256_mul_ps(v256_CRTM_14_2, av_s21);
        av_t5 = _mm256_mul_ps(v256_CRTM_14_4, av_s22);
        av_t6 = _mm256_mul_ps(v256_CRTM_14_6, av_s23);
        av_s33 = _mm256_add_ps(av_t4, av_t5);
        // Output point 5: X(4)
        v_out4 = _mm256_add_ps(av_s33, av_t6);
        curr_out = out + out_strides[3];
        STRI_2x256_S(curr_out, v_out_stride, v_out3, v_out4);

        av_t7 = _mm256_mul_ps(v256_CRTM_14_1, av_s17);
        av_t8 = _mm256_mul_ps(v256_CRTM_14_3, av_s15);
        av_t9 = _mm256_mul_ps(v256_CRTM_14_5, av_s16);
        av_s34 = _mm256_sub_ps(av_s2, av_t7);
        av_s35 = _mm256_sub_ps(av_t8, av_t9);
        // Output point 8: X(7)
        v_out7 = _mm256_add_ps(av_s34, av_s35);

        av_s24 = _mm256_sub_ps(av_s3, av_s13);
        av_s25 = _mm256_sub_ps(av_s5, av_s11);
        av_s26 = _mm256_sub_ps(av_s7, av_s9);

        av_t10 = _mm256_mul_ps(v256_CRTM_14_2, av_s26);
        av_t11 = _mm256_mul_ps(v256_CRTM_14_4, av_s24);
        av_t12 = _mm256_mul_ps(v256_CRTM_14_6, av_s25);
        av_s36 = _mm256_add_ps(av_t10, av_t11);
        // Output point 9: X(8)
        v_out8 = _mm256_add_ps(av_s36, av_t12);
        curr_out = out + out_strides[7];
        STRI_2x256_S(curr_out, v_out_stride, v_out7, v_out8);

        av_t13 = _mm256_mul_ps(v256_CRTM_14_1, av_s19);
        av_t14 = _mm256_mul_ps(v256_CRTM_14_3, av_s20);
        av_t15 = _mm256_mul_ps(v256_CRTM_14_5, av_s18);
        av_s37 = _mm256_sub_ps(av_s1, av_t13);
        av_s38 = _mm256_sub_ps(av_t14, av_t15);
        // Output point 12: X(11)
        v_out11 = _mm256_add_ps(av_s37, av_s38);

        av_t16 = _mm256_mul_ps(v256_CRTM_14_2, av_s22);
        av_t17 = _mm256_mul_ps(v256_CRTM_14_4, av_s23);
        av_t18 = _mm256_mul_ps(v256_CRTM_14_6, av_s21);
        av_s39 = _mm256_sub_ps(av_t16, av_t17);
        // Output point 13: X(12)
        v_out12 = _mm256_add_ps(av_s39, av_t18);
        curr_out = out + out_strides[11];
        STRI_2x256_S(curr_out, v_out_stride, v_out11, v_out12);

        av_t19 = _mm256_mul_ps(v256_CRTM_14_1, av_s16);
        av_t20 = _mm256_mul_ps(v256_CRTM_14_3, av_s17);
        av_t21 = _mm256_mul_ps(v256_CRTM_14_5, av_s15);
        av_s40 = _mm256_sub_ps(av_s2, av_t19);
        av_s41 = _mm256_sub_ps(av_t20, av_t21);
        // Output point 16: X(15)
        v_out15 = _mm256_add_ps(av_s40, av_s41);

        av_t22 = _mm256_mul_ps(v256_CRTM_14_2, av_s25);
        av_t23 = _mm256_mul_ps(v256_CRTM_14_4, av_s26);
        av_t24 = _mm256_mul_ps(v256_CRTM_14_6, av_s24);
        av_s42 = _mm256_sub_ps(av_t24, av_t22);
        // Output point 17: X(16)
        v_out16 = _mm256_sub_ps(av_s42, av_t23);
        curr_out = out + out_strides[15];
        STRI_2x256_S(curr_out, v_out_stride, v_out15, v_out16);

        av_t25 = _mm256_mul_ps(v256_CRTM_14_1, av_s20);
        av_t26 = _mm256_mul_ps(v256_CRTM_14_3, av_s18);
        av_t27 = _mm256_mul_ps(v256_CRTM_14_5, av_s19);
        av_s43 = _mm256_sub_ps(av_s1, av_t25);
        av_s44 = _mm256_sub_ps(av_t26, av_t27);
        // Output point 20: X(19)
        v_out19 = _mm256_add_ps(av_s43, av_s44);

        av_t28 = _mm256_mul_ps(v256_CRTM_14_2, av_s23);
        av_t29 = _mm256_mul_ps(v256_CRTM_14_4, av_s21);
        av_t30 = _mm256_mul_ps(v256_CRTM_14_6, av_s22);
        av_s45 = _mm256_add_ps(av_t28, av_t29);
        // Output point 21: X(20)
        v_out20 = _mm256_sub_ps(av_s45, av_t30);
        curr_out = out + out_strides[19];
        STRI_2x256_S(curr_out, v_out_stride, v_out19, v_out20);

        av_t31 = _mm256_mul_ps(v256_CRTM_14_1, av_s15);
        av_t32 = _mm256_mul_ps(v256_CRTM_14_3, av_s16);
        av_t33 = _mm256_mul_ps(v256_CRTM_14_5, av_s17);
        av_s46 = _mm256_sub_ps(av_s2, av_t31);
        av_s47 = _mm256_sub_ps(av_t32, av_t33);
        // Output point 24: X(23)
        v_out23 = _mm256_add_ps(av_s46, av_s47);

        av_t34 = _mm256_mul_ps(v256_CRTM_14_2, av_s24);
        av_t35 = _mm256_mul_ps(v256_CRTM_14_4, av_s25);
        av_t36 = _mm256_mul_ps(v256_CRTM_14_6, av_s26);
        av_s48 = _mm256_sub_ps(av_t34, av_t35);
        // Output point 25: X(24)
        v_out24 = _mm256_add_ps(av_s48, av_t36);
        curr_out = out + out_strides[23];
        STRI_2x256_S(curr_out, v_out_stride, v_out23, v_out24);

        /* Shifted DFT */
        __m256 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8, bv_in9, bv_in10, bv_in11, bv_in12, bv_in13;
        __m256 bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
               bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17,
               bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24, bv_s25,
               bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32, bv_s33,
               bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40, bv_s41,
               bv_s42, bv_s43, bv_s44, bv_s45, bv_s46;
        __m256 bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
               bv_t10, bv_t11, bv_t12, bv_t13, bv_t14, bv_t15, bv_t16, bv_t17,
               bv_t18, bv_t19, bv_t20, bv_t21, bv_t22, bv_t23, bv_t24, bv_t25,
               bv_t26, bv_t27, bv_t28, bv_t29, bv_t30, bv_t31, bv_t32, bv_t33,
               bv_t34, bv_t35, bv_t36;

        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_256_S(curr_in, v_in_stride, bv_in0);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_256_S(curr_in, v_in_stride, bv_in1);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_256_S(curr_in, v_in_stride, bv_in2);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_256_S(curr_in, v_in_stride, bv_in3);
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDR_256_S(curr_in, v_in_stride, bv_in4);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_256_S(curr_in, v_in_stride, bv_in5);
        // Input point 14: x(13)
        curr_in = in + in_strides[13];
        LDR_256_S(curr_in, v_in_stride, bv_in6);
        // Input point 16: x(15)
        curr_in = in + in_strides[15];
        LDR_256_S(curr_in, v_in_stride, bv_in7);
        // Input point 18: x(17)
        curr_in = in + in_strides[17];
        LDR_256_S(curr_in, v_in_stride, bv_in8);
        // Input point 20: x(19)
        curr_in = in + in_strides[19];
        LDR_256_S(curr_in, v_in_stride, bv_in9);
        // Input point 22: x(21)
        curr_in = in + in_strides[21];
        LDR_256_S(curr_in, v_in_stride, bv_in10);
        // Input point 24: x(23)
        curr_in = in + in_strides[23];
        LDR_256_S(curr_in, v_in_stride, bv_in11);
        // Input point 26: x(25)
        curr_in = in + in_strides[25];
        LDR_256_S(curr_in, v_in_stride, bv_in12);
        // Input point 28: x(27)
        curr_in = in + in_strides[27];
        LDR_256_S(curr_in, v_in_stride, bv_in13);

        bv_s1 = _mm256_add_ps(bv_in1, bv_in13);
        bv_s2 = _mm256_sub_ps(bv_in1, bv_in13);
        bv_s3 = _mm256_add_ps(bv_in2, bv_in12);
        bv_s4 = _mm256_sub_ps(bv_in2, bv_in12);
        bv_s5 = _mm256_add_ps(bv_in3, bv_in11);
        bv_s6 = _mm256_sub_ps(bv_in3, bv_in11);
        bv_s7 = _mm256_add_ps(bv_in4, bv_in10);
        bv_s8 = _mm256_sub_ps(bv_in4, bv_in10);
        bv_s9 = _mm256_add_ps(bv_in5, bv_in9);
        bv_s10 = _mm256_sub_ps(bv_in5, bv_in9);
        bv_s11 = _mm256_add_ps(bv_in6, bv_in8);
        bv_s12 = _mm256_sub_ps(bv_in6, bv_in8);

        bv_t1 = _mm256_mul_ps(v256_CRTM_14_5, bv_s12);
        bv_t2 = _mm256_mul_ps(v256_CRTM_14_1, bv_s4);
        bv_t3 = _mm256_mul_ps(v256_CRTM_14_3, bv_s8);
        bv_t4 = _mm256_mul_ps(v256_CRTM_14_6, bv_s2);
        bv_t5 = _mm256_mul_ps(v256_CRTM_14_2, bv_s10);
        bv_t6 = _mm256_mul_ps(v256_CRTM_14_4, bv_s6);

        bv_t7 = _mm256_mul_ps(v256_CRTM_14_5, bv_s1);
        bv_t8 = _mm256_mul_ps(v256_CRTM_14_1, bv_s9);
        bv_t9 = _mm256_mul_ps(v256_CRTM_14_3, bv_s5);
        bv_t10 = _mm256_mul_ps(v256_CRTM_14_6, bv_s11);
        bv_t11 = _mm256_mul_ps(v256_CRTM_14_2, bv_s3);
        bv_t12 = _mm256_mul_ps(v256_CRTM_14_4, bv_s7);

        bv_t13 = _mm256_mul_ps(v256_CRTM_14_5, bv_s4);
        bv_t14 = _mm256_mul_ps(v256_CRTM_14_1, bv_s8);
        bv_t15 = _mm256_mul_ps(v256_CRTM_14_3, bv_s12);
        bv_t16 = _mm256_mul_ps(v256_CRTM_14_6, bv_s10);
        bv_t17 = _mm256_mul_ps(v256_CRTM_14_2, bv_s6);
        bv_t18 = _mm256_mul_ps(v256_CRTM_14_4, bv_s2);

        bv_t19 = _mm256_mul_ps(v256_CRTM_14_5, bv_s9);
        bv_t20 = _mm256_mul_ps(v256_CRTM_14_1, bv_s5);
        bv_t21 = _mm256_mul_ps(v256_CRTM_14_3, bv_s1);
        bv_t22 = _mm256_mul_ps(v256_CRTM_14_6, bv_s3);
        bv_t23 = _mm256_mul_ps(v256_CRTM_14_2, bv_s7);
        bv_t24 = _mm256_mul_ps(v256_CRTM_14_4, bv_s11);

        bv_t25 = _mm256_mul_ps(v256_CRTM_14_5, bv_s8);
        bv_t26 = _mm256_mul_ps(v256_CRTM_14_1, bv_s12);
        bv_t27 = _mm256_mul_ps(v256_CRTM_14_3, bv_s4);
        bv_t28 = _mm256_mul_ps(v256_CRTM_14_6, bv_s6);
        bv_t29 = _mm256_mul_ps(v256_CRTM_14_2, bv_s2);
        bv_t30 = _mm256_mul_ps(v256_CRTM_14_4, bv_s10);

        bv_t31 = _mm256_mul_ps(v256_CRTM_14_5, bv_s5);
        bv_t32 = _mm256_mul_ps(v256_CRTM_14_1, bv_s1);
        bv_t33 = _mm256_mul_ps(v256_CRTM_14_3, bv_s9);
        bv_t34 = _mm256_mul_ps(v256_CRTM_14_6, bv_s7);
        bv_t35 = _mm256_mul_ps(v256_CRTM_14_2, bv_s11);
        bv_t36 = _mm256_mul_ps(v256_CRTM_14_4, bv_s3);

        bv_s13 = _mm256_add_ps(bv_in0, bv_t1);
        bv_s14 = _mm256_add_ps(bv_t2, bv_t3);
        bv_s15 = _mm256_add_ps(bv_s13, bv_s14);
        bv_s16 = _mm256_add_ps(bv_t4, bv_t5);
        bv_s17 = _mm256_add_ps(bv_s16, bv_t6);
        // Output point 2: X(1)
        v_out1 = _mm256_add_ps(bv_s15, bv_s17);
        // Output point 26: X(25)
        v_out25 = _mm256_sub_ps(bv_s15, bv_s17);

        bv_s18 = _mm256_add_ps(bv_in7, bv_t7);
        bv_s19 = _mm256_add_ps(bv_t8, bv_t9);
        bv_s20 = _mm256_add_ps(bv_s18, bv_s19);
        bv_s21 = _mm256_add_ps(bv_t10, bv_t11);
        bv_s22 = _mm256_add_ps(bv_s21, bv_t12);
        // Output point 3: X(2)
        v_out2 = NEGATE_256_S(_mm256_add_ps(bv_s20, bv_s22));
        curr_out = out + out_strides[1];
        STRI_2x256_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 27: X(26)
        v_out26 = _mm256_sub_ps(bv_s22, bv_s20);
        curr_out = out + out_strides[25];
        STRI_2x256_S(curr_out, v_out_stride, v_out25, v_out26);

        bv_s23 = _mm256_add_ps(bv_in0, bv_t13);
        bv_s24 = _mm256_add_ps(bv_t14, bv_t15);
        bv_s25 = _mm256_sub_ps(bv_s23, bv_s24);
        bv_s26 = _mm256_add_ps(bv_t16, bv_t17);
        bv_s27 = _mm256_sub_ps(bv_t18, bv_s26);
        // Output point 6: X(5)
        v_out5 = _mm256_add_ps(bv_s25, bv_s27);
        // Output point 22: X(21)
        v_out21 = _mm256_sub_ps(bv_s25, bv_s27);

        bv_s28 = _mm256_add_ps(bv_in7, bv_t19);
        bv_s29 = _mm256_add_ps(bv_t20, bv_t21);
        bv_s30 = _mm256_sub_ps(bv_s28, bv_s29);
        bv_s31 = _mm256_add_ps(bv_t22, bv_t23);
        bv_s32 = _mm256_sub_ps(bv_t24, bv_s31);
        // Output point 7: X(6)
        v_out6 = _mm256_add_ps(bv_s30, bv_s32);
        curr_out = out + out_strides[5];
        STRI_2x256_S(curr_out, v_out_stride, v_out5, v_out6);
        // Output point 23: X(22)
        v_out22 = _mm256_sub_ps(bv_s30, bv_s32);
        curr_out = out + out_strides[21];
        STRI_2x256_S(curr_out, v_out_stride, v_out21, v_out22);

        bv_s33 = _mm256_sub_ps(bv_in0, bv_t25);
        bv_s34 = _mm256_sub_ps(bv_t26, bv_t27);
        bv_s35 = _mm256_add_ps(bv_s33, bv_s34);
        bv_s36 = _mm256_sub_ps(bv_t29, bv_t28);
        bv_s37 = _mm256_add_ps(bv_s36, bv_t30);
        // Output point 10: X(9)
        v_out9 = _mm256_add_ps(bv_s35, bv_s37);
        // Output point 18: X(17)
        v_out17 = _mm256_sub_ps(bv_s35, bv_s37);

        bv_s38 = _mm256_sub_ps(bv_t31, bv_in7);
        bv_s39 = _mm256_sub_ps(bv_t33, bv_t32);
        bv_s40 = _mm256_add_ps(bv_s38, bv_s39);
        bv_s41 = _mm256_sub_ps(bv_t34, bv_t35);
        bv_s42 = _mm256_sub_ps(bv_s41, bv_t36);
        // Output point 11: X(10)
        v_out10 = _mm256_add_ps(bv_s40, bv_s42);
        curr_out = out + out_strides[9];
        STRI_2x256_S(curr_out, v_out_stride, v_out9, v_out10);
        // Output point 19: X(18)
        v_out18 = _mm256_sub_ps(bv_s40, bv_s42);
        curr_out = out + out_strides[17];
        STRI_2x256_S(curr_out, v_out_stride, v_out17, v_out18);

        bv_s43 = _mm256_add_ps(bv_in0, bv_s8);
        bv_s44 = _mm256_add_ps(bv_s4, bv_s12);
        bv_s45 = _mm256_add_ps(bv_in7, bv_s5);
        bv_s46 = _mm256_add_ps(bv_s1, bv_s9);
        // Output point 14: X(13)
        v_out13 = _mm256_sub_ps(bv_s43, bv_s44);
        // Output point 15: X(14)
        v_out14 = _mm256_sub_ps(bv_s45, bv_s46);
        curr_out = out + out_strides[13];
        STRI_2x256_S(curr_out, v_out_stride, v_out13, v_out14);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_S)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
               av_in8, av_in9, av_in10, av_in11, av_in12, av_in13;
        __m128 av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8, av_s9,
               av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16, av_s17,
               av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24, av_s25,
               av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32, av_s33,
               av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40, av_s41,
               av_s42, av_s43, av_s44, av_s45, av_s46, av_s47, av_s48;
        __m128 av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8, av_t9,
               av_t10, av_t11, av_t12, av_t13, av_t14, av_t15, av_t16, av_t17,
               av_t18, av_t19, av_t20, av_t21, av_t22, av_t23, av_t24, av_t25,
               av_t26, av_t27, av_t28, av_t29, av_t30, av_t31, av_t32, av_t33,
               av_t34, av_t35, av_t36;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21,
               v_out22, v_out23, v_out24, v_out25, v_out26, v_out27;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_14_1 = _mm512_castps512_ps128(v_CRTM_14_1);
        __m128 v128_CRTM_14_2 = _mm512_castps512_ps128(v_CRTM_14_2);
        __m128 v128_CRTM_14_3 = _mm512_castps512_ps128(v_CRTM_14_3);
        __m128 v128_CRTM_14_4 = _mm512_castps512_ps128(v_CRTM_14_4);
        __m128 v128_CRTM_14_5 = _mm512_castps512_ps128(v_CRTM_14_5);
        __m128 v128_CRTM_14_6 = _mm512_castps512_ps128(v_CRTM_14_6);

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
        // Input point 27: x(26)
        curr_in = in + in_strides[26];
        LDR_128_S(curr_in, v_in_stride, av_in13);

        av_s1 = _mm_sub_ps(av_in0, av_in7);
        av_s2 = _mm_add_ps(av_in0, av_in7);
        av_s3 = _mm_sub_ps(av_in13, av_in1);
        av_s4 = _mm_add_ps(av_in13, av_in1);
        av_s5 = _mm_sub_ps(av_in12, av_in2);
        av_s6 = _mm_add_ps(av_in12, av_in2);
        av_s7 = _mm_sub_ps(av_in11, av_in3);
        av_s8 = _mm_add_ps(av_in11, av_in3);
        av_s9 = _mm_sub_ps(av_in10, av_in4);
        av_s10 = _mm_add_ps(av_in10, av_in4);
        av_s11 = _mm_sub_ps(av_in9, av_in5);
        av_s12 = _mm_add_ps(av_in9, av_in5);
        av_s13 = _mm_sub_ps(av_in8, av_in6);
        av_s14 = _mm_add_ps(av_in8, av_in6);

        av_s15 = _mm_add_ps(av_s4, av_s14);
        av_s16 = _mm_add_ps(av_s6, av_s12);
        av_s17 = _mm_add_ps(av_s8, av_s10);

        av_s18 = _mm_sub_ps(av_s14, av_s4);
        av_s19 = _mm_sub_ps(av_s6, av_s12);
        av_s20 = _mm_sub_ps(av_s10, av_s8);
        av_s27 = _mm_add_ps(av_s2, av_s15);
        av_s28 = _mm_add_ps(av_s16, av_s17);
        av_s29 = _mm_add_ps(av_s1, av_s18);
        av_s30 = _mm_add_ps(av_s19, av_s20);
        // Output pt 1: X(0)
        v_out0 = _mm_add_ps(av_s27, av_s28);
        STR_128_S(curr_out, v_out_stride, v_out0);
        // Output pt 28: X(27)
        v_out27 = _mm_add_ps(av_s29, av_s30);
        curr_out = out + out_strides[27];
        STR_128_S(curr_out, v_out_stride, v_out27);

        av_t1 = _mm_mul_ps(v128_CRTM_14_1, av_s18);
        av_t2 = _mm_mul_ps(v128_CRTM_14_3, av_s19);
        av_t3 = _mm_mul_ps(v128_CRTM_14_5, av_s20);
        av_s31 = _mm_sub_ps(av_s1, av_t1);
        av_s32 = _mm_sub_ps(av_t2, av_t3);
        // Output point 4: X(3)
        v_out3 = _mm_add_ps(av_s31, av_s32);

        av_s21 = _mm_add_ps(av_s3, av_s13);
        av_s22 = _mm_add_ps(av_s5, av_s11);
        av_s23 = _mm_add_ps(av_s7, av_s9);

        av_t4 = _mm_mul_ps(v128_CRTM_14_2, av_s21);
        av_t5 = _mm_mul_ps(v128_CRTM_14_4, av_s22);
        av_t6 = _mm_mul_ps(v128_CRTM_14_6, av_s23);
        av_s33 = _mm_add_ps(av_t4, av_t5);
        // Output point 5: X(4)
        v_out4 = _mm_add_ps(av_s33, av_t6);
        curr_out = out + out_strides[3];
        STRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);

        av_t7 = _mm_mul_ps(v128_CRTM_14_1, av_s17);
        av_t8 = _mm_mul_ps(v128_CRTM_14_3, av_s15);
        av_t9 = _mm_mul_ps(v128_CRTM_14_5, av_s16);
        av_s34 = _mm_sub_ps(av_s2, av_t7);
        av_s35 = _mm_sub_ps(av_t8, av_t9);
        // Output point 8: X(7)
        v_out7 = _mm_add_ps(av_s34, av_s35);

        av_s24 = _mm_sub_ps(av_s3, av_s13);
        av_s25 = _mm_sub_ps(av_s5, av_s11);
        av_s26 = _mm_sub_ps(av_s7, av_s9);

        av_t10 = _mm_mul_ps(v128_CRTM_14_2, av_s26);
        av_t11 = _mm_mul_ps(v128_CRTM_14_4, av_s24);
        av_t12 = _mm_mul_ps(v128_CRTM_14_6, av_s25);
        av_s36 = _mm_add_ps(av_t10, av_t11);
        // Output point 9: X(8)
        v_out8 = _mm_add_ps(av_s36, av_t12);
        curr_out = out + out_strides[7];
        STRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);

        av_t13 = _mm_mul_ps(v128_CRTM_14_1, av_s19);
        av_t14 = _mm_mul_ps(v128_CRTM_14_3, av_s20);
        av_t15 = _mm_mul_ps(v128_CRTM_14_5, av_s18);
        av_s37 = _mm_sub_ps(av_s1, av_t13);
        av_s38 = _mm_sub_ps(av_t14, av_t15);
        // Output point 12: X(11)
        v_out11 = _mm_add_ps(av_s37, av_s38);

        av_t16 = _mm_mul_ps(v128_CRTM_14_2, av_s22);
        av_t17 = _mm_mul_ps(v128_CRTM_14_4, av_s23);
        av_t18 = _mm_mul_ps(v128_CRTM_14_6, av_s21);
        av_s39 = _mm_sub_ps(av_t16, av_t17);
        // Output point 13: X(12)
        v_out12 = _mm_add_ps(av_s39, av_t18);
        curr_out = out + out_strides[11];
        STRI_2x128_S(curr_out, v_out_stride, v_out11, v_out12);

        av_t19 = _mm_mul_ps(v128_CRTM_14_1, av_s16);
        av_t20 = _mm_mul_ps(v128_CRTM_14_3, av_s17);
        av_t21 = _mm_mul_ps(v128_CRTM_14_5, av_s15);
        av_s40 = _mm_sub_ps(av_s2, av_t19);
        av_s41 = _mm_sub_ps(av_t20, av_t21);
        // Output point 16: X(15)
        v_out15 = _mm_add_ps(av_s40, av_s41);

        av_t22 = _mm_mul_ps(v128_CRTM_14_2, av_s25);
        av_t23 = _mm_mul_ps(v128_CRTM_14_4, av_s26);
        av_t24 = _mm_mul_ps(v128_CRTM_14_6, av_s24);
        av_s42 = _mm_sub_ps(av_t24, av_t22);
        // Output point 17: X(16)
        v_out16 = _mm_sub_ps(av_s42, av_t23);
        curr_out = out + out_strides[15];
        STRI_2x128_S(curr_out, v_out_stride, v_out15, v_out16);

        av_t25 = _mm_mul_ps(v128_CRTM_14_1, av_s20);
        av_t26 = _mm_mul_ps(v128_CRTM_14_3, av_s18);
        av_t27 = _mm_mul_ps(v128_CRTM_14_5, av_s19);
        av_s43 = _mm_sub_ps(av_s1, av_t25);
        av_s44 = _mm_sub_ps(av_t26, av_t27);
        // Output point 20: X(19)
        v_out19 = _mm_add_ps(av_s43, av_s44);

        av_t28 = _mm_mul_ps(v128_CRTM_14_2, av_s23);
        av_t29 = _mm_mul_ps(v128_CRTM_14_4, av_s21);
        av_t30 = _mm_mul_ps(v128_CRTM_14_6, av_s22);
        av_s45 = _mm_add_ps(av_t28, av_t29);
        // Output point 21: X(20)
        v_out20 = _mm_sub_ps(av_s45, av_t30);
        curr_out = out + out_strides[19];
        STRI_2x128_S(curr_out, v_out_stride, v_out19, v_out20);

        av_t31 = _mm_mul_ps(v128_CRTM_14_1, av_s15);
        av_t32 = _mm_mul_ps(v128_CRTM_14_3, av_s16);
        av_t33 = _mm_mul_ps(v128_CRTM_14_5, av_s17);
        av_s46 = _mm_sub_ps(av_s2, av_t31);
        av_s47 = _mm_sub_ps(av_t32, av_t33);
        // Output point 24: X(23)
        v_out23 = _mm_add_ps(av_s46, av_s47);

        av_t34 = _mm_mul_ps(v128_CRTM_14_2, av_s24);
        av_t35 = _mm_mul_ps(v128_CRTM_14_4, av_s25);
        av_t36 = _mm_mul_ps(v128_CRTM_14_6, av_s26);
        av_s48 = _mm_sub_ps(av_t34, av_t35);
        // Output point 25: X(24)
        v_out24 = _mm_add_ps(av_s48, av_t36);
        curr_out = out + out_strides[23];
        STRI_2x128_S(curr_out, v_out_stride, v_out23, v_out24);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8, bv_in9, bv_in10, bv_in11, bv_in12, bv_in13;
        __m128 bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
               bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17,
               bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24, bv_s25,
               bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32, bv_s33,
               bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40, bv_s41,
               bv_s42, bv_s43, bv_s44, bv_s45, bv_s46;
        __m128 bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
               bv_t10, bv_t11, bv_t12, bv_t13, bv_t14, bv_t15, bv_t16, bv_t17,
               bv_t18, bv_t19, bv_t20, bv_t21, bv_t22, bv_t23, bv_t24, bv_t25,
               bv_t26, bv_t27, bv_t28, bv_t29, bv_t30, bv_t31, bv_t32, bv_t33,
               bv_t34, bv_t35, bv_t36;

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
        // Input point 28: x(27)
        curr_in = in + in_strides[27];
        LDR_128_S(curr_in, v_in_stride, bv_in13);

        bv_s1 = _mm_add_ps(bv_in1, bv_in13);
        bv_s2 = _mm_sub_ps(bv_in1, bv_in13);
        bv_s3 = _mm_add_ps(bv_in2, bv_in12);
        bv_s4 = _mm_sub_ps(bv_in2, bv_in12);
        bv_s5 = _mm_add_ps(bv_in3, bv_in11);
        bv_s6 = _mm_sub_ps(bv_in3, bv_in11);
        bv_s7 = _mm_add_ps(bv_in4, bv_in10);
        bv_s8 = _mm_sub_ps(bv_in4, bv_in10);
        bv_s9 = _mm_add_ps(bv_in5, bv_in9);
        bv_s10 = _mm_sub_ps(bv_in5, bv_in9);
        bv_s11 = _mm_add_ps(bv_in6, bv_in8);
        bv_s12 = _mm_sub_ps(bv_in6, bv_in8);

        bv_t1 = _mm_mul_ps(v128_CRTM_14_5, bv_s12);
        bv_t2 = _mm_mul_ps(v128_CRTM_14_1, bv_s4);
        bv_t3 = _mm_mul_ps(v128_CRTM_14_3, bv_s8);
        bv_t4 = _mm_mul_ps(v128_CRTM_14_6, bv_s2);
        bv_t5 = _mm_mul_ps(v128_CRTM_14_2, bv_s10);
        bv_t6 = _mm_mul_ps(v128_CRTM_14_4, bv_s6);

        bv_t7 = _mm_mul_ps(v128_CRTM_14_5, bv_s1);
        bv_t8 = _mm_mul_ps(v128_CRTM_14_1, bv_s9);
        bv_t9 = _mm_mul_ps(v128_CRTM_14_3, bv_s5);
        bv_t10 = _mm_mul_ps(v128_CRTM_14_6, bv_s11);
        bv_t11 = _mm_mul_ps(v128_CRTM_14_2, bv_s3);
        bv_t12 = _mm_mul_ps(v128_CRTM_14_4, bv_s7);

        bv_t13 = _mm_mul_ps(v128_CRTM_14_5, bv_s4);
        bv_t14 = _mm_mul_ps(v128_CRTM_14_1, bv_s8);
        bv_t15 = _mm_mul_ps(v128_CRTM_14_3, bv_s12);
        bv_t16 = _mm_mul_ps(v128_CRTM_14_6, bv_s10);
        bv_t17 = _mm_mul_ps(v128_CRTM_14_2, bv_s6);
        bv_t18 = _mm_mul_ps(v128_CRTM_14_4, bv_s2);

        bv_t19 = _mm_mul_ps(v128_CRTM_14_5, bv_s9);
        bv_t20 = _mm_mul_ps(v128_CRTM_14_1, bv_s5);
        bv_t21 = _mm_mul_ps(v128_CRTM_14_3, bv_s1);
        bv_t22 = _mm_mul_ps(v128_CRTM_14_6, bv_s3);
        bv_t23 = _mm_mul_ps(v128_CRTM_14_2, bv_s7);
        bv_t24 = _mm_mul_ps(v128_CRTM_14_4, bv_s11);

        bv_t25 = _mm_mul_ps(v128_CRTM_14_5, bv_s8);
        bv_t26 = _mm_mul_ps(v128_CRTM_14_1, bv_s12);
        bv_t27 = _mm_mul_ps(v128_CRTM_14_3, bv_s4);
        bv_t28 = _mm_mul_ps(v128_CRTM_14_6, bv_s6);
        bv_t29 = _mm_mul_ps(v128_CRTM_14_2, bv_s2);
        bv_t30 = _mm_mul_ps(v128_CRTM_14_4, bv_s10);

        bv_t31 = _mm_mul_ps(v128_CRTM_14_5, bv_s5);
        bv_t32 = _mm_mul_ps(v128_CRTM_14_1, bv_s1);
        bv_t33 = _mm_mul_ps(v128_CRTM_14_3, bv_s9);
        bv_t34 = _mm_mul_ps(v128_CRTM_14_6, bv_s7);
        bv_t35 = _mm_mul_ps(v128_CRTM_14_2, bv_s11);
        bv_t36 = _mm_mul_ps(v128_CRTM_14_4, bv_s3);

        bv_s13 = _mm_add_ps(bv_in0, bv_t1);
        bv_s14 = _mm_add_ps(bv_t2, bv_t3);
        bv_s15 = _mm_add_ps(bv_s13, bv_s14);
        bv_s16 = _mm_add_ps(bv_t4, bv_t5);
        bv_s17 = _mm_add_ps(bv_s16, bv_t6);
        // Output point 2: X(1)
        v_out1 = _mm_add_ps(bv_s15, bv_s17);
        // Output point 26: X(25)
        v_out25 = _mm_sub_ps(bv_s15, bv_s17);

        bv_s18 = _mm_add_ps(bv_in7, bv_t7);
        bv_s19 = _mm_add_ps(bv_t8, bv_t9);
        bv_s20 = _mm_add_ps(bv_s18, bv_s19);
        bv_s21 = _mm_add_ps(bv_t10, bv_t11);
        bv_s22 = _mm_add_ps(bv_s21, bv_t12);
        // Output point 3: X(2)
        v_out2 = NEGATE_128_S(_mm_add_ps(bv_s20, bv_s22));
        curr_out = out + out_strides[1];
        STRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 27: X(26)
        v_out26 = _mm_sub_ps(bv_s22, bv_s20);
        curr_out = out + out_strides[25];
        STRI_2x128_S(curr_out, v_out_stride, v_out25, v_out26);

        bv_s23 = _mm_add_ps(bv_in0, bv_t13);
        bv_s24 = _mm_add_ps(bv_t14, bv_t15);
        bv_s25 = _mm_sub_ps(bv_s23, bv_s24);
        bv_s26 = _mm_add_ps(bv_t16, bv_t17);
        bv_s27 = _mm_sub_ps(bv_t18, bv_s26);
        // Output point 6: X(5)
        v_out5 = _mm_add_ps(bv_s25, bv_s27);
        // Output point 22: X(21)
        v_out21 = _mm_sub_ps(bv_s25, bv_s27);

        bv_s28 = _mm_add_ps(bv_in7, bv_t19);
        bv_s29 = _mm_add_ps(bv_t20, bv_t21);
        bv_s30 = _mm_sub_ps(bv_s28, bv_s29);
        bv_s31 = _mm_add_ps(bv_t22, bv_t23);
        bv_s32 = _mm_sub_ps(bv_t24, bv_s31);
        // Output point 7: X(6)
        v_out6 = _mm_add_ps(bv_s30, bv_s32);
        curr_out = out + out_strides[5];
        STRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);
        // Output point 23: X(22)
        v_out22 = _mm_sub_ps(bv_s30, bv_s32);
        curr_out = out + out_strides[21];
        STRI_2x128_S(curr_out, v_out_stride, v_out21, v_out22);

        bv_s33 = _mm_sub_ps(bv_in0, bv_t25);
        bv_s34 = _mm_sub_ps(bv_t26, bv_t27);
        bv_s35 = _mm_add_ps(bv_s33, bv_s34);
        bv_s36 = _mm_sub_ps(bv_t29, bv_t28);
        bv_s37 = _mm_add_ps(bv_s36, bv_t30);
        // Output point 10: X(9)
        v_out9 = _mm_add_ps(bv_s35, bv_s37);
        // Output point 18: X(17)
        v_out17 = _mm_sub_ps(bv_s35, bv_s37);

        bv_s38 = _mm_sub_ps(bv_t31, bv_in7);
        bv_s39 = _mm_sub_ps(bv_t33, bv_t32);
        bv_s40 = _mm_add_ps(bv_s38, bv_s39);
        bv_s41 = _mm_sub_ps(bv_t34, bv_t35);
        bv_s42 = _mm_sub_ps(bv_s41, bv_t36);
        // Output point 11: X(10)
        v_out10 = _mm_add_ps(bv_s40, bv_s42);
        curr_out = out + out_strides[9];
        STRI_2x128_S(curr_out, v_out_stride, v_out9, v_out10);
        // Output point 19: X(18)
        v_out18 = _mm_sub_ps(bv_s40, bv_s42);
        curr_out = out + out_strides[17];
        STRI_2x128_S(curr_out, v_out_stride, v_out17, v_out18);

        bv_s43 = _mm_add_ps(bv_in0, bv_s8);
        bv_s44 = _mm_add_ps(bv_s4, bv_s12);
        bv_s45 = _mm_add_ps(bv_in7, bv_s5);
        bv_s46 = _mm_add_ps(bv_s1, bv_s9);
        // Output point 14: X(13)
        v_out13 = _mm_sub_ps(bv_s43, bv_s44);
        // Output point 15: X(14)
        v_out14 = _mm_sub_ps(bv_s45, bv_s46);
        curr_out = out + out_strides[13];
        STRI_2x128_S(curr_out, v_out_stride, v_out13, v_out14);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & 2)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
               av_in8, av_in9, av_in10, av_in11, av_in12, av_in13;
        __m128 av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8, av_s9,
               av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16, av_s17,
               av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24, av_s25,
               av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32, av_s33,
               av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40, av_s41,
               av_s42, av_s43, av_s44, av_s45, av_s46, av_s47, av_s48;
        __m128 av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8, av_t9,
               av_t10, av_t11, av_t12, av_t13, av_t14, av_t15, av_t16, av_t17,
               av_t18, av_t19, av_t20, av_t21, av_t22, av_t23, av_t24, av_t25,
               av_t26, av_t27, av_t28, av_t29, av_t30, av_t31, av_t32, av_t33,
               av_t34, av_t35, av_t36;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21,
               v_out22, v_out23, v_out24, v_out25, v_out26, v_out27;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_14_1 = _mm512_castps512_ps128(v_CRTM_14_1);
        __m128 v128_CRTM_14_2 = _mm512_castps512_ps128(v_CRTM_14_2);
        __m128 v128_CRTM_14_3 = _mm512_castps512_ps128(v_CRTM_14_3);
        __m128 v128_CRTM_14_4 = _mm512_castps512_ps128(v_CRTM_14_4);
        __m128 v128_CRTM_14_5 = _mm512_castps512_ps128(v_CRTM_14_5);
        __m128 v128_CRTM_14_6 = _mm512_castps512_ps128(v_CRTM_14_6);

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
        // Input point 27: x(26)
        curr_in = in + in_strides[26];
        LDHR_128_S(curr_in, v_in_stride, av_in13);

        av_s1 = _mm_sub_ps(av_in0, av_in7);
        av_s2 = _mm_add_ps(av_in0, av_in7);
        av_s3 = _mm_sub_ps(av_in13, av_in1);
        av_s4 = _mm_add_ps(av_in13, av_in1);
        av_s5 = _mm_sub_ps(av_in12, av_in2);
        av_s6 = _mm_add_ps(av_in12, av_in2);
        av_s7 = _mm_sub_ps(av_in11, av_in3);
        av_s8 = _mm_add_ps(av_in11, av_in3);
        av_s9 = _mm_sub_ps(av_in10, av_in4);
        av_s10 = _mm_add_ps(av_in10, av_in4);
        av_s11 = _mm_sub_ps(av_in9, av_in5);
        av_s12 = _mm_add_ps(av_in9, av_in5);
        av_s13 = _mm_sub_ps(av_in8, av_in6);
        av_s14 = _mm_add_ps(av_in8, av_in6);

        av_s15 = _mm_add_ps(av_s4, av_s14);
        av_s16 = _mm_add_ps(av_s6, av_s12);
        av_s17 = _mm_add_ps(av_s8, av_s10);

        av_s18 = _mm_sub_ps(av_s14, av_s4);
        av_s19 = _mm_sub_ps(av_s6, av_s12);
        av_s20 = _mm_sub_ps(av_s10, av_s8);
        av_s27 = _mm_add_ps(av_s2, av_s15);
        av_s28 = _mm_add_ps(av_s16, av_s17);
        av_s29 = _mm_add_ps(av_s1, av_s18);
        av_s30 = _mm_add_ps(av_s19, av_s20);
        // Output pt 1: X(0)
        v_out0 = _mm_add_ps(av_s27, av_s28);
        STHR_128_S(curr_out, v_out_stride, v_out0);
        // Output pt 28: X(27)
        v_out27 = _mm_add_ps(av_s29, av_s30);
        curr_out = out + out_strides[27];
        STHR_128_S(curr_out, v_out_stride, v_out27);

        av_t1 = _mm_mul_ps(v128_CRTM_14_1, av_s18);
        av_t2 = _mm_mul_ps(v128_CRTM_14_3, av_s19);
        av_t3 = _mm_mul_ps(v128_CRTM_14_5, av_s20);
        av_s31 = _mm_sub_ps(av_s1, av_t1);
        av_s32 = _mm_sub_ps(av_t2, av_t3);
        // Output point 4: X(3)
        v_out3 = _mm_add_ps(av_s31, av_s32);

        av_s21 = _mm_add_ps(av_s3, av_s13);
        av_s22 = _mm_add_ps(av_s5, av_s11);
        av_s23 = _mm_add_ps(av_s7, av_s9);

        av_t4 = _mm_mul_ps(v128_CRTM_14_2, av_s21);
        av_t5 = _mm_mul_ps(v128_CRTM_14_4, av_s22);
        av_t6 = _mm_mul_ps(v128_CRTM_14_6, av_s23);
        av_s33 = _mm_add_ps(av_t4, av_t5);
        // Output point 5: X(4)
        v_out4 = _mm_add_ps(av_s33, av_t6);
        curr_out = out + out_strides[3];
        STHRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);

        av_t7 = _mm_mul_ps(v128_CRTM_14_1, av_s17);
        av_t8 = _mm_mul_ps(v128_CRTM_14_3, av_s15);
        av_t9 = _mm_mul_ps(v128_CRTM_14_5, av_s16);
        av_s34 = _mm_sub_ps(av_s2, av_t7);
        av_s35 = _mm_sub_ps(av_t8, av_t9);
        // Output point 8: X(7)
        v_out7 = _mm_add_ps(av_s34, av_s35);

        av_s24 = _mm_sub_ps(av_s3, av_s13);
        av_s25 = _mm_sub_ps(av_s5, av_s11);
        av_s26 = _mm_sub_ps(av_s7, av_s9);

        av_t10 = _mm_mul_ps(v128_CRTM_14_2, av_s26);
        av_t11 = _mm_mul_ps(v128_CRTM_14_4, av_s24);
        av_t12 = _mm_mul_ps(v128_CRTM_14_6, av_s25);
        av_s36 = _mm_add_ps(av_t10, av_t11);
        // Output point 9: X(8)
        v_out8 = _mm_add_ps(av_s36, av_t12);
        curr_out = out + out_strides[7];
        STHRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);

        av_t13 = _mm_mul_ps(v128_CRTM_14_1, av_s19);
        av_t14 = _mm_mul_ps(v128_CRTM_14_3, av_s20);
        av_t15 = _mm_mul_ps(v128_CRTM_14_5, av_s18);
        av_s37 = _mm_sub_ps(av_s1, av_t13);
        av_s38 = _mm_sub_ps(av_t14, av_t15);
        // Output point 12: X(11)
        v_out11 = _mm_add_ps(av_s37, av_s38);

        av_t16 = _mm_mul_ps(v128_CRTM_14_2, av_s22);
        av_t17 = _mm_mul_ps(v128_CRTM_14_4, av_s23);
        av_t18 = _mm_mul_ps(v128_CRTM_14_6, av_s21);
        av_s39 = _mm_sub_ps(av_t16, av_t17);
        // Output point 13: X(12)
        v_out12 = _mm_add_ps(av_s39, av_t18);
        curr_out = out + out_strides[11];
        STHRI_2x128_S(curr_out, v_out_stride, v_out11, v_out12);

        av_t19 = _mm_mul_ps(v128_CRTM_14_1, av_s16);
        av_t20 = _mm_mul_ps(v128_CRTM_14_3, av_s17);
        av_t21 = _mm_mul_ps(v128_CRTM_14_5, av_s15);
        av_s40 = _mm_sub_ps(av_s2, av_t19);
        av_s41 = _mm_sub_ps(av_t20, av_t21);
        // Output point 16: X(15)
        v_out15 = _mm_add_ps(av_s40, av_s41);

        av_t22 = _mm_mul_ps(v128_CRTM_14_2, av_s25);
        av_t23 = _mm_mul_ps(v128_CRTM_14_4, av_s26);
        av_t24 = _mm_mul_ps(v128_CRTM_14_6, av_s24);
        av_s42 = _mm_sub_ps(av_t24, av_t22);
        // Output point 17: X(16)
        v_out16 = _mm_sub_ps(av_s42, av_t23);
        curr_out = out + out_strides[15];
        STHRI_2x128_S(curr_out, v_out_stride, v_out15, v_out16);

        av_t25 = _mm_mul_ps(v128_CRTM_14_1, av_s20);
        av_t26 = _mm_mul_ps(v128_CRTM_14_3, av_s18);
        av_t27 = _mm_mul_ps(v128_CRTM_14_5, av_s19);
        av_s43 = _mm_sub_ps(av_s1, av_t25);
        av_s44 = _mm_sub_ps(av_t26, av_t27);
        // Output point 20: X(19)
        v_out19 = _mm_add_ps(av_s43, av_s44);

        av_t28 = _mm_mul_ps(v128_CRTM_14_2, av_s23);
        av_t29 = _mm_mul_ps(v128_CRTM_14_4, av_s21);
        av_t30 = _mm_mul_ps(v128_CRTM_14_6, av_s22);
        av_s45 = _mm_add_ps(av_t28, av_t29);
        // Output point 21: X(20)
        v_out20 = _mm_sub_ps(av_s45, av_t30);
        curr_out = out + out_strides[19];
        STHRI_2x128_S(curr_out, v_out_stride, v_out19, v_out20);

        av_t31 = _mm_mul_ps(v128_CRTM_14_1, av_s15);
        av_t32 = _mm_mul_ps(v128_CRTM_14_3, av_s16);
        av_t33 = _mm_mul_ps(v128_CRTM_14_5, av_s17);
        av_s46 = _mm_sub_ps(av_s2, av_t31);
        av_s47 = _mm_sub_ps(av_t32, av_t33);
        // Output point 24: X(23)
        v_out23 = _mm_add_ps(av_s46, av_s47);

        av_t34 = _mm_mul_ps(v128_CRTM_14_2, av_s24);
        av_t35 = _mm_mul_ps(v128_CRTM_14_4, av_s25);
        av_t36 = _mm_mul_ps(v128_CRTM_14_6, av_s26);
        av_s48 = _mm_sub_ps(av_t34, av_t35);
        // Output point 25: X(24)
        v_out24 = _mm_add_ps(av_s48, av_t36);
        curr_out = out + out_strides[23];
        STHRI_2x128_S(curr_out, v_out_stride, v_out23, v_out24);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8, bv_in9, bv_in10, bv_in11, bv_in12, bv_in13;
        __m128 bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
               bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17,
               bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24, bv_s25,
               bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32, bv_s33,
               bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40, bv_s41,
               bv_s42, bv_s43, bv_s44, bv_s45, bv_s46;
        __m128 bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
               bv_t10, bv_t11, bv_t12, bv_t13, bv_t14, bv_t15, bv_t16, bv_t17,
               bv_t18, bv_t19, bv_t20, bv_t21, bv_t22, bv_t23, bv_t24, bv_t25,
               bv_t26, bv_t27, bv_t28, bv_t29, bv_t30, bv_t31, bv_t32, bv_t33,
               bv_t34, bv_t35, bv_t36;

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
        // Input point 28: x(27)
        curr_in = in + in_strides[27];
        LDHR_128_S(curr_in, v_in_stride, bv_in13);

        bv_s1 = _mm_add_ps(bv_in1, bv_in13);
        bv_s2 = _mm_sub_ps(bv_in1, bv_in13);
        bv_s3 = _mm_add_ps(bv_in2, bv_in12);
        bv_s4 = _mm_sub_ps(bv_in2, bv_in12);
        bv_s5 = _mm_add_ps(bv_in3, bv_in11);
        bv_s6 = _mm_sub_ps(bv_in3, bv_in11);
        bv_s7 = _mm_add_ps(bv_in4, bv_in10);
        bv_s8 = _mm_sub_ps(bv_in4, bv_in10);
        bv_s9 = _mm_add_ps(bv_in5, bv_in9);
        bv_s10 = _mm_sub_ps(bv_in5, bv_in9);
        bv_s11 = _mm_add_ps(bv_in6, bv_in8);
        bv_s12 = _mm_sub_ps(bv_in6, bv_in8);

        bv_t1 = _mm_mul_ps(v128_CRTM_14_5, bv_s12);
        bv_t2 = _mm_mul_ps(v128_CRTM_14_1, bv_s4);
        bv_t3 = _mm_mul_ps(v128_CRTM_14_3, bv_s8);
        bv_t4 = _mm_mul_ps(v128_CRTM_14_6, bv_s2);
        bv_t5 = _mm_mul_ps(v128_CRTM_14_2, bv_s10);
        bv_t6 = _mm_mul_ps(v128_CRTM_14_4, bv_s6);

        bv_t7 = _mm_mul_ps(v128_CRTM_14_5, bv_s1);
        bv_t8 = _mm_mul_ps(v128_CRTM_14_1, bv_s9);
        bv_t9 = _mm_mul_ps(v128_CRTM_14_3, bv_s5);
        bv_t10 = _mm_mul_ps(v128_CRTM_14_6, bv_s11);
        bv_t11 = _mm_mul_ps(v128_CRTM_14_2, bv_s3);
        bv_t12 = _mm_mul_ps(v128_CRTM_14_4, bv_s7);

        bv_t13 = _mm_mul_ps(v128_CRTM_14_5, bv_s4);
        bv_t14 = _mm_mul_ps(v128_CRTM_14_1, bv_s8);
        bv_t15 = _mm_mul_ps(v128_CRTM_14_3, bv_s12);
        bv_t16 = _mm_mul_ps(v128_CRTM_14_6, bv_s10);
        bv_t17 = _mm_mul_ps(v128_CRTM_14_2, bv_s6);
        bv_t18 = _mm_mul_ps(v128_CRTM_14_4, bv_s2);

        bv_t19 = _mm_mul_ps(v128_CRTM_14_5, bv_s9);
        bv_t20 = _mm_mul_ps(v128_CRTM_14_1, bv_s5);
        bv_t21 = _mm_mul_ps(v128_CRTM_14_3, bv_s1);
        bv_t22 = _mm_mul_ps(v128_CRTM_14_6, bv_s3);
        bv_t23 = _mm_mul_ps(v128_CRTM_14_2, bv_s7);
        bv_t24 = _mm_mul_ps(v128_CRTM_14_4, bv_s11);

        bv_t25 = _mm_mul_ps(v128_CRTM_14_5, bv_s8);
        bv_t26 = _mm_mul_ps(v128_CRTM_14_1, bv_s12);
        bv_t27 = _mm_mul_ps(v128_CRTM_14_3, bv_s4);
        bv_t28 = _mm_mul_ps(v128_CRTM_14_6, bv_s6);
        bv_t29 = _mm_mul_ps(v128_CRTM_14_2, bv_s2);
        bv_t30 = _mm_mul_ps(v128_CRTM_14_4, bv_s10);

        bv_t31 = _mm_mul_ps(v128_CRTM_14_5, bv_s5);
        bv_t32 = _mm_mul_ps(v128_CRTM_14_1, bv_s1);
        bv_t33 = _mm_mul_ps(v128_CRTM_14_3, bv_s9);
        bv_t34 = _mm_mul_ps(v128_CRTM_14_6, bv_s7);
        bv_t35 = _mm_mul_ps(v128_CRTM_14_2, bv_s11);
        bv_t36 = _mm_mul_ps(v128_CRTM_14_4, bv_s3);

        bv_s13 = _mm_add_ps(bv_in0, bv_t1);
        bv_s14 = _mm_add_ps(bv_t2, bv_t3);
        bv_s15 = _mm_add_ps(bv_s13, bv_s14);
        bv_s16 = _mm_add_ps(bv_t4, bv_t5);
        bv_s17 = _mm_add_ps(bv_s16, bv_t6);
        // Output point 2: X(1)
        v_out1 = _mm_add_ps(bv_s15, bv_s17);
        // Output point 26: X(25)
        v_out25 = _mm_sub_ps(bv_s15, bv_s17);

        bv_s18 = _mm_add_ps(bv_in7, bv_t7);
        bv_s19 = _mm_add_ps(bv_t8, bv_t9);
        bv_s20 = _mm_add_ps(bv_s18, bv_s19);
        bv_s21 = _mm_add_ps(bv_t10, bv_t11);
        bv_s22 = _mm_add_ps(bv_s21, bv_t12);
        // Output point 3: X(2)
        v_out2 = NEGATE_128_S(_mm_add_ps(bv_s20, bv_s22));
        curr_out = out + out_strides[1];
        STHRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 27: X(26)
        v_out26 = _mm_sub_ps(bv_s22, bv_s20);
        curr_out = out + out_strides[25];
        STHRI_2x128_S(curr_out, v_out_stride, v_out25, v_out26);

        bv_s23 = _mm_add_ps(bv_in0, bv_t13);
        bv_s24 = _mm_add_ps(bv_t14, bv_t15);
        bv_s25 = _mm_sub_ps(bv_s23, bv_s24);
        bv_s26 = _mm_add_ps(bv_t16, bv_t17);
        bv_s27 = _mm_sub_ps(bv_t18, bv_s26);
        // Output point 6: X(5)
        v_out5 = _mm_add_ps(bv_s25, bv_s27);
        // Output point 22: X(21)
        v_out21 = _mm_sub_ps(bv_s25, bv_s27);

        bv_s28 = _mm_add_ps(bv_in7, bv_t19);
        bv_s29 = _mm_add_ps(bv_t20, bv_t21);
        bv_s30 = _mm_sub_ps(bv_s28, bv_s29);
        bv_s31 = _mm_add_ps(bv_t22, bv_t23);
        bv_s32 = _mm_sub_ps(bv_t24, bv_s31);
        // Output point 7: X(6)
        v_out6 = _mm_add_ps(bv_s30, bv_s32);
        curr_out = out + out_strides[5];
        STHRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);
        // Output point 23: X(22)
        v_out22 = _mm_sub_ps(bv_s30, bv_s32);
        curr_out = out + out_strides[21];
        STHRI_2x128_S(curr_out, v_out_stride, v_out21, v_out22);

        bv_s33 = _mm_sub_ps(bv_in0, bv_t25);
        bv_s34 = _mm_sub_ps(bv_t26, bv_t27);
        bv_s35 = _mm_add_ps(bv_s33, bv_s34);
        bv_s36 = _mm_sub_ps(bv_t29, bv_t28);
        bv_s37 = _mm_add_ps(bv_s36, bv_t30);
        // Output point 10: X(9)
        v_out9 = _mm_add_ps(bv_s35, bv_s37);
        // Output point 18: X(17)
        v_out17 = _mm_sub_ps(bv_s35, bv_s37);

        bv_s38 = _mm_sub_ps(bv_t31, bv_in7);
        bv_s39 = _mm_sub_ps(bv_t33, bv_t32);
        bv_s40 = _mm_add_ps(bv_s38, bv_s39);
        bv_s41 = _mm_sub_ps(bv_t34, bv_t35);
        bv_s42 = _mm_sub_ps(bv_s41, bv_t36);
        // Output point 11: X(10)
        v_out10 = _mm_add_ps(bv_s40, bv_s42);
        curr_out = out + out_strides[9];
        STHRI_2x128_S(curr_out, v_out_stride, v_out9, v_out10);
        // Output point 19: X(18)
        v_out18 = _mm_sub_ps(bv_s40, bv_s42);
        curr_out = out + out_strides[17];
        STHRI_2x128_S(curr_out, v_out_stride, v_out17, v_out18);

        bv_s43 = _mm_add_ps(bv_in0, bv_s8);
        bv_s44 = _mm_add_ps(bv_s4, bv_s12);
        bv_s45 = _mm_add_ps(bv_in7, bv_s5);
        bv_s46 = _mm_add_ps(bv_s1, bv_s9);
        // Output point 14: X(13)
        v_out13 = _mm_sub_ps(bv_s43, bv_s44);
        // Output point 15: X(14)
        v_out14 = _mm_sub_ps(bv_s45, bv_s46);
        curr_out = out + out_strides[13];
        STHRI_2x128_S(curr_out, v_out_stride, v_out13, v_out14);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        /* Standard DFT */
        FFTZ_FLOAT a_in0, a_in1, a_in2, a_in3, a_in4, a_in5, a_in6, a_in7,
            a_in8, a_in9, a_in10, a_in11, a_in12, a_in13;
        FFTZ_FLOAT a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_s9,
            a_s10, a_s11, a_s12, a_s13, a_s14, a_s15, a_s16, a_s17, a_s18,
            a_s19, a_s20, a_s21, a_s22, a_s23, a_s24, a_s25, a_s26, a_s27,
            a_s28, a_s29, a_s30, a_s31, a_s32, a_s33, a_s34, a_s35, a_s36,
            a_s37, a_s38, a_s39, a_s40, a_s41, a_s42, a_s43, a_s44, a_s45,
            a_s46, a_s47;
        FFTZ_FLOAT a_t0, a_t1, a_t2, a_t3, a_t4, a_t5, a_t6, a_t7, a_t8, a_t9,
            a_t10, a_t11, a_t12, a_t13, a_t14, a_t15, a_t16, a_t17, a_t18,
            a_t19, a_t20, a_t21, a_t22, a_t23, a_t24, a_t25, a_t26, a_t27,
            a_t28, a_t29, a_t30, a_t31, a_t32, a_t33, a_t34, a_t35;

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

        a_s0 = a_in0 - a_in7;
        a_s1 = a_in0 + a_in7;
        a_s2 = a_in13 - a_in1;
        a_s3 = a_in13 + a_in1;
        a_s4 = a_in12 - a_in2;
        a_s5 = a_in12 + a_in2;
        a_s6 = a_in11 - a_in3;
        a_s7 = a_in11 + a_in3;
        a_s8 = a_in10 - a_in4;
        a_s9 = a_in10 + a_in4;
        a_s10 = a_in9 - a_in5;
        a_s11 = a_in9 + a_in5;
        a_s12 = a_in8 - a_in6;
        a_s13 = a_in8 + a_in6;

        a_s14 = a_s3 + a_s13;
        a_s15 = a_s5 + a_s11;
        a_s16 = a_s7 + a_s9;

        a_s17 = a_s13 - a_s3;
        a_s18 = a_s5 - a_s11;
        a_s19 = a_s9 - a_s7;
        a_s26 = a_s1 + a_s14;
        a_s27 = a_s15 + a_s16;
        a_s28 = a_s0 + a_s17;
        a_s29 = a_s18 + a_s19;
        // Output point 1: X(0)
        *out = a_s26 + a_s27;
        // Output point 28: X(27)
        out[out_strides[27]] = a_s28 + a_s29;

        a_t0 = CRTM_14_1 * a_s17;
        a_t1 = CRTM_14_3 * a_s18;
        a_t2 = CRTM_14_5 * a_s19;
        a_s30 = a_s0 - a_t0;
        a_s31 = a_t1 - a_t2;
        // Output point 4: X(3)
        out[out_strides[3]] = a_s30 + a_s31;

        a_s20 = a_s2 + a_s12;
        a_s21 = a_s4 + a_s10;
        a_s22 = a_s6 + a_s8;
        a_t3 = CRTM_14_2 * a_s20;
        a_t4 = CRTM_14_4 * a_s21;
        a_t5 = CRTM_14_6 * a_s22;
        a_s32 = a_t3 + a_t4;
        // Output point 5: X(4)
        out[out_strides[4]] = a_s32 + a_t5;

        a_t6 = CRTM_14_1 * a_s16;
        a_t7 = CRTM_14_3 * a_s14;
        a_t8 = CRTM_14_5 * a_s15;
        a_s33 = a_s1 - a_t6;
        a_s34 = a_t7 - a_t8;
        // Output point 8: X(7)
        out[out_strides[7]] = a_s33 + a_s34;

        a_s23 = a_s2 - a_s12;
        a_s24 = a_s4 - a_s10;
        a_s25 = a_s6 - a_s8;
        a_t9 = CRTM_14_2 * a_s25;
        a_t10 = CRTM_14_4 * a_s23;
        a_t11 = CRTM_14_6 * a_s24;
        a_s35 = a_t9 + a_t10;
        // Output point 9: X(8)
        out[out_strides[8]] = a_s35 + a_t11;

        a_t12 = CRTM_14_1 * a_s18;
        a_t13 = CRTM_14_3 * a_s19;
        a_t14 = CRTM_14_5 * a_s17;
        a_s36 = a_s0 - a_t12;
        a_s37 = a_t13 - a_t14;
        // Output point 12: X(11)
        out[out_strides[11]] = a_s36 + a_s37;

        a_t15 = CRTM_14_2 * a_s21;
        a_t16 = CRTM_14_4 * a_s22;
        a_t17 = CRTM_14_6 * a_s20;
        a_s38 = a_t15 - a_t16;
        // Output point 13: X(12)
        out[out_strides[12]] = a_s38 + a_t17;

        a_t18 = CRTM_14_1 * a_s15;
        a_t19 = CRTM_14_3 * a_s16;
        a_t20 = CRTM_14_5 * a_s14;
        a_s39 = a_s1 - a_t18;
        a_s40 = a_t19 - a_t20;
        // Output point 16: X(15)
        out[out_strides[15]] = a_s39 + a_s40;

        a_t21 = CRTM_14_2 * a_s24;
        a_t22 = CRTM_14_4 * a_s25;
        a_t23 = CRTM_14_6 * a_s23;
        a_s41 = a_t23 - a_t21;
        // Output point 17: X(16)
        out[out_strides[16]] = a_s41 - a_t22;

        a_t24 = CRTM_14_1 * a_s19;
        a_t25 = CRTM_14_3 * a_s17;
        a_t26 = CRTM_14_5 * a_s18;
        a_s42 = a_s0 - a_t24;
        a_s43 = a_t25 - a_t26;
        // Output point 20: X(19)
        out[out_strides[19]] = a_s42 + a_s43;

        a_t27 = CRTM_14_2 * a_s22;
        a_t28 = CRTM_14_4 * a_s20;
        a_t29 = CRTM_14_6 * a_s21;
        a_s44 = a_t27 + a_t28;
        // Output point 21: X(20)
        out[out_strides[20]] = a_s44 - a_t29;

        a_t30 = CRTM_14_1 * a_s14;
        a_t31 = CRTM_14_3 * a_s15;
        a_t32 = CRTM_14_5 * a_s16;
        a_s45 = a_s1 - a_t30;
        a_s46 = a_t31 - a_t32;
        // Output point 24: X(23)
        out[out_strides[23]] = a_s45 + a_s46;

        a_t33 = CRTM_14_2 * a_s23;
        a_t34 = CRTM_14_4 * a_s24;
        a_t35 = CRTM_14_6 * a_s25;
        a_s47 = a_t33 - a_t34;
        // Output point 25: X(24)
        out[out_strides[24]] = a_s47 + a_t35;

        /* Shifted DFT */
        FFTZ_FLOAT b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7,
            b_in8, b_in9, b_in10, b_in11, b_in12, b_in13;
        FFTZ_FLOAT b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9,
            b_s10, b_s11, b_s12, b_s13, b_s14, b_s15, b_s16, b_s17, b_s18,
            b_s19, b_s20, b_s21, b_s22, b_s23, b_s24, b_s25, b_s26, b_s27,
            b_s28, b_s29, b_s30, b_s31, b_s32, b_s33, b_s34, b_s35, b_s36,
            b_s37, b_s38, b_s39, b_s40, b_s41, b_s42, b_s43, b_s44, b_s45;
        FFTZ_FLOAT b_t0, b_t1, b_t2, b_t3, b_t4, b_t5, b_t6, b_t7, b_t8, b_t9,
            b_t10, b_t11, b_t12, b_t13, b_t14, b_t15, b_t16, b_t17, b_t18,
            b_t19, b_t20, b_t21, b_t22, b_t23, b_t24, b_t25, b_t26, b_t27,
            b_t28, b_t29, b_t30, b_t31, b_t32, b_t33, b_t34, b_t35;

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

        b_s0 = b_in1 + b_in13;
        b_s1 = b_in1 - b_in13;
        b_s2 = b_in2 + b_in12;
        b_s3 = b_in2 - b_in12;
        b_s4 = b_in3 + b_in11;
        b_s5 = b_in3 - b_in11;
        b_s6 = b_in4 + b_in10;
        b_s7 = b_in4 - b_in10;
        b_s8 = b_in5 + b_in9;
        b_s9 = b_in5 - b_in9;
        b_s10 = b_in6 + b_in8;
        b_s11 = b_in6 - b_in8;

        b_t0 = CRTM_14_5 * b_s11;
        b_t1 = CRTM_14_1 * b_s3;
        b_t2 = CRTM_14_3 * b_s7;
        b_t3 = CRTM_14_6 * b_s1;
        b_t4 = CRTM_14_2 * b_s9;
        b_t5 = CRTM_14_4 * b_s5;

        b_t6 = CRTM_14_5 * b_s0;
        b_t7 = CRTM_14_1 * b_s8;
        b_t8 = CRTM_14_3 * b_s4;
        b_t9 = CRTM_14_6 * b_s10;
        b_t10 = CRTM_14_2 * b_s2;
        b_t11 = CRTM_14_4 * b_s6;

        b_t12 = CRTM_14_5 * b_s3;
        b_t13 = CRTM_14_1 * b_s7;
        b_t14 = CRTM_14_3 * b_s11;
        b_t15 = CRTM_14_6 * b_s9;
        b_t16 = CRTM_14_2 * b_s5;
        b_t17 = CRTM_14_4 * b_s1;

        b_t18 = CRTM_14_5 * b_s8;
        b_t19 = CRTM_14_1 * b_s4;
        b_t20 = CRTM_14_3 * b_s0;
        b_t21 = CRTM_14_6 * b_s2;
        b_t22 = CRTM_14_2 * b_s6;
        b_t23 = CRTM_14_4 * b_s10;

        b_t24 = CRTM_14_5 * b_s7;
        b_t25 = CRTM_14_1 * b_s11;
        b_t26 = CRTM_14_3 * b_s3;
        b_t27 = CRTM_14_6 * b_s5;
        b_t28 = CRTM_14_2 * b_s1;
        b_t29 = CRTM_14_4 * b_s9;

        b_t30 = CRTM_14_5 * b_s4;
        b_t31 = CRTM_14_1 * b_s0;
        b_t32 = CRTM_14_3 * b_s8;
        b_t33 = CRTM_14_6 * b_s6;
        b_t34 = CRTM_14_2 * b_s10;
        b_t35 = CRTM_14_4 * b_s2;

        b_s12 = b_in0 + b_t0;
        b_s13 = b_t1 + b_t2;
        b_s14 = b_s12 + b_s13;
        b_s15 = b_t3 + b_t4;
        b_s16 = b_s15 + b_t5;
        // Output point 2: X(1)
        out[out_strides[1]] = b_s14 + b_s16;
        // Output point 26: X(25)
        out[out_strides[25]] = b_s14 - b_s16;

        b_s17 = b_in7 + b_t6;
        b_s18 = b_t7 + b_t8;
        b_s19 = b_s17 + b_s18;
        b_s20 = b_t9 + b_t10;
        b_s21 = b_s20 + b_t11;
        // Output point 3: X(2)
        out[out_strides[2]] = -(b_s19 + b_s21);
        // Output point 27: X(26)
        out[out_strides[26]] = b_s21 - b_s19;

        b_s22 = b_in0 + b_t12;
        b_s23 = b_t13 + b_t14;
        b_s24 = b_s22 - b_s23;
        b_s25 = b_t15 + b_t16;
        b_s26 = b_t17 - b_s25;
        // Output point 6: X(5)
        out[out_strides[5]] = b_s24 + b_s26;
        // Output point 22: X(21)
        out[out_strides[21]] = b_s24 - b_s26;

        b_s27 = b_in7 + b_t18;
        b_s28 = b_t19 + b_t20;
        b_s29 = b_s27 - b_s28;
        b_s30 = b_t21 + b_t22;
        b_s31 = b_t23 - b_s30;
        // Output point 7: X(6)
        out[out_strides[6]] = b_s29 + b_s31;
        // Output point 23: X(22)
        out[out_strides[22]] = b_s29 - b_s31;

        b_s32 = b_in0 - b_t24;
        b_s33 = b_t25 - b_t26;
        b_s34 = b_s32 + b_s33;
        b_s35 = b_t28 - b_t27;
        b_s36 = b_s35 + b_t29;
        // Output point 10: X(9)
        out[out_strides[9]] = b_s34 + b_s36;
        // Output point 18: X(17)
        out[out_strides[17]] = b_s34 - b_s36;

        b_s37 = b_t30 - b_in7;
        b_s38 = b_t32 - b_t31;
        b_s39 = b_s37 + b_s38;
        b_s40 = b_t33 - b_t34;
        b_s41 = b_s40 - b_t35;
        // Output point 11: X(10)
        out[out_strides[10]] = b_s39 + b_s41;
        // Output point 19: X(18)
        out[out_strides[18]] = b_s39 - b_s41;

        b_s42 = b_in0 + b_s7;
        b_s43 = b_s3 + b_s11;
        b_s44 = b_in7 + b_s4;
        b_s45 = b_s0 + b_s8;
        // Output point 14: X(13)
        out[out_strides[13]] = b_s42 - b_s43;
        // Output point 15: X(14)
        out[out_strides[14]] = b_s44 - b_s45;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft14avx512_fp32_bwd(FFTZ_VOID *in_real,
                                             FFTZ_VOID *in_imag,
                                             FFTZ_VOID *out_real,
                                             FFTZ_VOID *out_imag, FFTZ_INTP n,
                                             aoclfftz_strides_t *strides,
                                             FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_14_1 =
        0.867767478235116240951536665696717509219981456f;
    const FFTZ_FLOAT CRTM_14_2 =
        1.801937735804838252472204639014890102331838324f;
    const FFTZ_FLOAT CRTM_14_3 =
        1.563662964936059617416889053348115500464669037f;
    const FFTZ_FLOAT CRTM_14_4 =
        1.246979603717467061050009768008479621264549462f;
    const FFTZ_FLOAT CRTM_14_5 =
        1.949855824363647214036263365987862434465571601f;
    const FFTZ_FLOAT CRTM_14_6 =
        0.445041867912628808577805128993589518932711138f;
    const FFTZ_FLOAT CRTM_14_7 =
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
    FFTZ_FLOAT *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_512_S;
    FFTZ_INTP remaining_sets = n % NUM_SETS_REAL_512_S;

    __m512 v_CRTM_14_1 = _mm512_set1_ps(CRTM_14_1);
    __m512 v_CRTM_14_2 = _mm512_set1_ps(CRTM_14_2);
    __m512 v_CRTM_14_3 = _mm512_set1_ps(CRTM_14_3);
    __m512 v_CRTM_14_4 = _mm512_set1_ps(CRTM_14_4);
    __m512 v_CRTM_14_5 = _mm512_set1_ps(CRTM_14_5);
    __m512 v_CRTM_14_6 = _mm512_set1_ps(CRTM_14_6);
    __m512 v_CRTM_14_7 = _mm512_set1_ps(CRTM_14_7);

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m512 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
               av_in8, av_in9, av_in10, av_in11, av_in12, av_in13;
        __m512 av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8, av_s9,
               av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16, av_s17,
               av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24, av_s25,
               av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32, av_s33,
               av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40, av_s41,
               av_s42, av_s43, av_s44, av_s45, av_s46, av_s47, av_s48;
        __m512 av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8, av_t9,
               av_t10, av_t11, av_t12, av_t13, av_t14, av_t15, av_t16, av_t17,
               av_t18, av_t19, av_t20, av_t21, av_t22, av_t23, av_t24, av_t25,
               av_t26, av_t27, av_t28, av_t29, av_t30, av_t31, av_t32, av_t33,
               av_t34, av_t35, av_t36, av_t37, av_t38;
        __m512 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21,
               v_out22, v_out23, v_out24, v_out25, v_out26, v_out27;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_512_S(curr_in, v_in_stride, av_in0);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x512_S(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in + in_strides[7];
        LDRI_2x512_S(curr_in, v_in_stride, av_in3, av_in4);
        // Input point 12: x(11) & Input point 13: x(12)
        curr_in = in + in_strides[11];
        LDRI_2x512_S(curr_in, v_in_stride, av_in5, av_in6);
        // Input point 16: x(15) & Input point 17: x(16)
        curr_in = in + in_strides[15];
        LDRI_2x512_S(curr_in, v_in_stride, av_in7, av_in8);
        // Input point 20: x(19) & Input point 21: x(20)
        curr_in = in + in_strides[19];
        LDRI_2x512_S(curr_in, v_in_stride, av_in9, av_in10);
        // Input point 24: x(23) & Input point 25: x(24)
        curr_in = in + in_strides[23];
        LDRI_2x512_S(curr_in, v_in_stride, av_in11, av_in12);
        // Input point 28: x(27)
        curr_in = in + in_strides[27];
        LDR_512_S(curr_in, v_in_stride, av_in13);

        av_s1 = _mm512_sub_ps(av_in0, av_in13);
        av_s2 = _mm512_add_ps(av_in0, av_in13);
        av_s3 = _mm512_sub_ps(av_in11, av_in1);
        av_s4 = _mm512_add_ps(av_in1, av_in11);
        av_s5 = _mm512_sub_ps(av_in2, av_in12);
        av_s6 = _mm512_add_ps(av_in2, av_in12);
        av_s7 = _mm512_sub_ps(av_in3, av_in9);
        av_s8 = _mm512_add_ps(av_in3, av_in9);
        av_s9 = _mm512_sub_ps(av_in4, av_in10);
        av_s10 = _mm512_add_ps(av_in4, av_in10);
        av_s11 = _mm512_sub_ps(av_in7, av_in5);
        av_s12 = _mm512_add_ps(av_in5, av_in7);
        av_s13 = _mm512_sub_ps(av_in6, av_in8);
        av_s14 = _mm512_add_ps(av_in6, av_in8);

        av_s27 = _mm512_add_ps(av_s12, av_s4);
        av_s28 = _mm512_add_ps(av_s27, av_s8);
        av_t37 = _mm512_mul_ps(v_CRTM_14_7, av_s28);
        av_s29 = _mm512_add_ps(av_s3, av_s7);
        av_s30 = _mm512_add_ps(av_s29, av_s11);
        av_t38 = _mm512_mul_ps(v_CRTM_14_7, av_s30);
        // Output pt 1: X(0)
        v_out0 = _mm512_add_ps(av_t37, av_s2);
        STR_512_S(curr_out, v_out_stride, v_out0);
        // Output pt 15: X(14)
        curr_out = out + out_strides[14];
        v_out14 = _mm512_add_ps(av_t38, av_s1);
        STR_512_S(curr_out, v_out_stride, v_out14);

        av_t1 = _mm512_mul_ps(v_CRTM_14_1, av_s6);
        av_t2 = _mm512_mul_ps(v_CRTM_14_3, av_s10);
        av_t3 = _mm512_mul_ps(v_CRTM_14_5, av_s14);
        av_t4 = _mm512_mul_ps(v_CRTM_14_2, av_s3);
        av_t5 = _mm512_mul_ps(v_CRTM_14_4, av_s7);
        av_t6 = _mm512_mul_ps(v_CRTM_14_6, av_s11);

        av_s31 = _mm512_sub_ps(av_t5, av_t6);
        av_s32 = _mm512_sub_ps(av_s1, av_t4);
        av_s33 = _mm512_add_ps(av_t1, av_t2);

        av_s15 = _mm512_add_ps(av_s31, av_s32);
        av_s16 = _mm512_add_ps(av_s33, av_t3);
        // Output point 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm512_sub_ps(av_s15, av_s16);
        STR_512_S(curr_out, v_out_stride, v_out2);
        // Output point 27: X(26)
        curr_out = out + out_strides[26];
        v_out26 = _mm512_add_ps(av_s15, av_s16);
        STR_512_S(curr_out, v_out_stride, v_out26);

        av_t7 = _mm512_mul_ps(v_CRTM_14_1, av_s13);
        av_t8 = _mm512_mul_ps(v_CRTM_14_3, av_s5);
        av_t9 = _mm512_mul_ps(v_CRTM_14_5, av_s9);

        av_t10 = _mm512_mul_ps(v_CRTM_14_2, av_s12);
        av_t11 = _mm512_mul_ps(v_CRTM_14_4, av_s4);
        av_t12 = _mm512_mul_ps(v_CRTM_14_6, av_s8);

        av_s34 = _mm512_sub_ps(av_s2, av_t10);
        av_s35 = _mm512_sub_ps(av_t11, av_t12);
        av_s36 = _mm512_add_ps(av_t7, av_t8);

        av_s17 = av_s34 + av_s35;
        av_s18 = _mm512_add_ps(av_s36, av_t9);
        // Output point 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm512_sub_ps(av_s17, av_s18);
        STR_512_S(curr_out, v_out_stride, v_out4);
        // Output point 25: X(24)
        curr_out = out + out_strides[24];
        v_out24 = _mm512_add_ps(av_s17, av_s18);
        STR_512_S(curr_out, v_out_stride, v_out24);

        av_t13 = _mm512_mul_ps(v_CRTM_14_1, av_s10);
        av_t14 = _mm512_mul_ps(v_CRTM_14_3, av_s14);
        av_t15 = _mm512_mul_ps(v_CRTM_14_5, av_s6);
        av_t16 = _mm512_mul_ps(v_CRTM_14_2, av_s7);
        av_t17 = _mm512_mul_ps(v_CRTM_14_4, av_s11);
        av_t18 = _mm512_mul_ps(v_CRTM_14_6, av_s3);

        av_s37 = _mm512_sub_ps(av_s1, av_t16);
        av_s38 = _mm512_sub_ps(av_t17, av_t18);
        av_s39 = _mm512_sub_ps(av_t14, av_t15);
        av_s19 = _mm512_add_ps(av_s37, av_s38);
        av_s20 = _mm512_sub_ps(av_s39, av_t13);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm512_add_ps(av_s19, av_s20);
        STR_512_S(curr_out, v_out_stride, v_out6);
        // Output point 23: X(22)
        curr_out = out + out_strides[22];
        v_out22 = _mm512_sub_ps(av_s19, av_s20);
        STR_512_S(curr_out, v_out_stride, v_out22);

        av_t19 = _mm512_mul_ps(v_CRTM_14_1, av_s9);
        av_t20 = _mm512_mul_ps(v_CRTM_14_3, av_s13);
        av_t21 = _mm512_mul_ps(v_CRTM_14_5, av_s5);
        av_t22 = _mm512_mul_ps(v_CRTM_14_2, av_s8);
        av_t23 = _mm512_mul_ps(v_CRTM_14_4, av_s12);
        av_t24 = _mm512_mul_ps(v_CRTM_14_6, av_s4);

        av_s40 = _mm512_sub_ps(av_s2, av_t22);
        av_s41 = _mm512_sub_ps(av_t23, av_t24);
        av_s42 = _mm512_add_ps(av_t19, av_t20);
        av_s21 = _mm512_add_ps(av_s40, av_s41);
        av_s22 = _mm512_sub_ps(av_s42, av_t21);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm512_add_ps(av_s21, av_s22);
        STR_512_S(curr_out, v_out_stride, v_out8);
        // Output point 21: X(20)
        curr_out = out + out_strides[20];
        v_out20 = _mm512_sub_ps(av_s21, av_s22);
        STR_512_S(curr_out, v_out_stride, v_out20);

        av_t25 = _mm512_mul_ps(v_CRTM_14_1, av_s14);
        av_t26 = _mm512_mul_ps(v_CRTM_14_3, av_s6);
        av_t27 = _mm512_mul_ps(v_CRTM_14_5, av_s10);
        av_t28 = _mm512_mul_ps(v_CRTM_14_2, av_s11);
        av_t29 = _mm512_mul_ps(v_CRTM_14_4, av_s3);
        av_t30 = _mm512_mul_ps(v_CRTM_14_6, av_s7);

        av_s43 = _mm512_sub_ps(av_s1, av_t28);
        av_s44 = _mm512_sub_ps(av_t29, av_t30);
        av_s45 = _mm512_sub_ps(av_t27, av_t25);
        av_s23 = _mm512_add_ps(av_s43, av_s44);
        av_s24 = _mm512_sub_ps(av_s45, av_t26);
        // Output pt 11: X(10)
        curr_out = out + out_strides[10];
        v_out10 = _mm512_add_ps(av_s23, av_s24);
        STR_512_S(curr_out, v_out_stride, v_out10);
        // Output pt 19: X(18)
        curr_out = out + out_strides[18];
        v_out18 = _mm512_sub_ps(av_s23, av_s24);
        STR_512_S(curr_out, v_out_stride, v_out18);

        av_t31 = _mm512_mul_ps(v_CRTM_14_1, av_s5);
        av_t32 = _mm512_mul_ps(v_CRTM_14_3, av_s9);
        av_t33 = _mm512_mul_ps(v_CRTM_14_5, av_s13);
        av_t34 = _mm512_mul_ps(v_CRTM_14_2, av_s4);
        av_t35 = _mm512_mul_ps(v_CRTM_14_4, av_s8);
        av_t36 = _mm512_mul_ps(v_CRTM_14_6, av_s12);

        av_s46 = _mm512_sub_ps(av_s2, av_t34);
        av_s47 = _mm512_sub_ps(av_t35, av_t36);
        av_s48 = _mm512_sub_ps(av_t32, av_t31);
        av_s25 = _mm512_add_ps(av_s46, av_s47);
        av_s26 = _mm512_sub_ps(av_s48, av_t33);
        // Output pt 13: X(12)
        curr_out = out + out_strides[12];
        v_out12 = _mm512_add_ps(av_s25, av_s26);
        STR_512_S(curr_out, v_out_stride, v_out12);
        // Output pt 17: X(16)
        curr_out = out + out_strides[16];
        v_out16 = _mm512_sub_ps(av_s25, av_s26);
        STR_512_S(curr_out, v_out_stride, v_out16);

        /* Shifted DFT */
        __m512 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8, bv_in9, bv_in10, bv_in11, bv_in12, bv_in13;
        __m512 bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
               bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17,
               bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24, bv_s25,
               bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32, bv_s33,
               bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40, bv_s41,
               bv_s42, bv_s43, bv_s44, bv_s45, bv_s46, bv_s47, bv_s48, bv_s49,
               bv_s50;
        __m512 bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
               bv_t10, bv_t11, bv_t12, bv_t13, bv_t14, bv_t15, bv_t16, bv_t17,
               bv_t18, bv_t19, bv_t20, bv_t21, bv_t22, bv_t23, bv_t24, bv_t25,
               bv_t26, bv_t27, bv_t28, bv_t29, bv_t30, bv_t31, bv_t32, bv_t33,
               bv_t34, bv_t35, bv_t36;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x512_S(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDRI_2x512_S(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDRI_2x512_S(curr_in, v_in_stride, bv_in4, bv_in5);
        // Input point 14: x(13) & Input point 15: x(14)
        curr_in = in + in_strides[13];
        LDRI_2x512_S(curr_in, v_in_stride, bv_in6, bv_in7);
        // Input point 18: x(17) & Input point 19: x(18)
        curr_in = in + in_strides[17];
        LDRI_2x512_S(curr_in, v_in_stride, bv_in8, bv_in9);
        // Input point 22: x(21) & Input point 23: x(22)
        curr_in = in + in_strides[21];
        LDRI_2x512_S(curr_in, v_in_stride, bv_in10, bv_in11);
        // Input point 26: x(25) & Input point 27: x(26)
        curr_in = in + in_strides[25];
        LDRI_2x512_S(curr_in, v_in_stride, bv_in12, bv_in13);

        bv_s1 = _mm512_add_ps(bv_in0, bv_in12);
        bv_s2 = _mm512_sub_ps(bv_in0, bv_in12);
        bv_s3 = _mm512_add_ps(bv_in1, bv_in13);
        bv_s4 = _mm512_sub_ps(bv_in1, bv_in13);
        bv_s5 = _mm512_add_ps(bv_in2, bv_in10);
        bv_s6 = _mm512_sub_ps(bv_in2, bv_in10);
        bv_s7 = _mm512_add_ps(bv_in3, bv_in11);
        bv_s8 = _mm512_sub_ps(bv_in3, bv_in11);
        bv_s9 = _mm512_add_ps(bv_in4, bv_in8);
        bv_s10 = _mm512_sub_ps(bv_in4, bv_in8);
        bv_s11 = _mm512_add_ps(bv_in5, bv_in9);
        bv_s12 = _mm512_sub_ps(bv_in5, bv_in9);

        bv_t1 = _mm512_mul_ps(v_CRTM_14_6, bv_s3);
        bv_t2 = _mm512_mul_ps(v_CRTM_14_2, bv_s11);
        bv_t3 = _mm512_mul_ps(v_CRTM_14_4, bv_s7);
        bv_t4 = _mm512_mul_ps(v_CRTM_14_5, bv_s2);
        bv_t5 = _mm512_mul_ps(v_CRTM_14_1, bv_s10);
        bv_t6 = _mm512_mul_ps(v_CRTM_14_3, bv_s6);

        bv_t7 = _mm512_mul_ps(v_CRTM_14_6, bv_s5);
        bv_t8 = _mm512_mul_ps(v_CRTM_14_2, bv_s1);
        bv_t9 = _mm512_mul_ps(v_CRTM_14_4, bv_s9);
        bv_t10 = _mm512_mul_ps(v_CRTM_14_5, bv_s8);
        bv_t11 = _mm512_mul_ps(v_CRTM_14_1, bv_s4);
        bv_t12 = _mm512_mul_ps(v_CRTM_14_3, bv_s12);

        bv_t13 = _mm512_mul_ps(v_CRTM_14_6, bv_s11);
        bv_t14 = _mm512_mul_ps(v_CRTM_14_2, bv_s7);
        bv_t15 = _mm512_mul_ps(v_CRTM_14_4, bv_s3);
        bv_t16 = _mm512_mul_ps(v_CRTM_14_5, bv_s10);
        bv_t17 = _mm512_mul_ps(v_CRTM_14_1, bv_s6);
        bv_t18 = _mm512_mul_ps(v_CRTM_14_3, bv_s2);

        bv_t19 = _mm512_mul_ps(v_CRTM_14_6, bv_s9);
        bv_t20 = _mm512_mul_ps(v_CRTM_14_2, bv_s5);
        bv_t21 = _mm512_mul_ps(v_CRTM_14_4, bv_s1);
        bv_t22 = _mm512_mul_ps(v_CRTM_14_5, bv_s12);
        bv_t23 = _mm512_mul_ps(v_CRTM_14_1, bv_s8);
        bv_t24 = _mm512_mul_ps(v_CRTM_14_3, bv_s4);

        bv_t25 = _mm512_mul_ps(v_CRTM_14_6, bv_s7);
        bv_t26 = _mm512_mul_ps(v_CRTM_14_2, bv_s3);
        bv_t27 = _mm512_mul_ps(v_CRTM_14_4, bv_s11);
        bv_t28 = _mm512_mul_ps(v_CRTM_14_5, bv_s6);
        bv_t29 = _mm512_mul_ps(v_CRTM_14_1, bv_s2);
        bv_t30 = _mm512_mul_ps(v_CRTM_14_3, bv_s10);

        bv_t31 = _mm512_mul_ps(v_CRTM_14_6, bv_s1);
        bv_t32 = _mm512_mul_ps(v_CRTM_14_2, bv_s9);
        bv_t33 = _mm512_mul_ps(v_CRTM_14_4, bv_s5);
        bv_t34 = _mm512_mul_ps(v_CRTM_14_5, bv_s4);
        bv_t35 = _mm512_mul_ps(v_CRTM_14_1, bv_s12);
        bv_t36 = _mm512_mul_ps(v_CRTM_14_3, bv_s8);

        bv_s13 = _mm512_add_ps(bv_in6, bv_in6);
        bv_s14 = _mm512_add_ps(bv_in7, bv_in7);

        bv_s15 = _mm512_add_ps(bv_t1, bv_t2);
        bv_s16 = _mm512_add_ps(bv_t3, bv_s14);
        bv_s17 = _mm512_add_ps(bv_s15, bv_s16);
        bv_s18 = _mm512_add_ps(bv_t4, bv_t5);
        bv_s19 = _mm512_add_ps(bv_t6, bv_s18);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm512_sub_ps(bv_s19, bv_s17);
        STR_512_S(curr_out, v_out_stride, v_out3);
        // Output pt 28: X(27)
        curr_out = out + out_strides[27];
        v_out27 = NEGATE_512_S(_mm512_add_ps(bv_s17, bv_s19));
        STR_512_S(curr_out, v_out_stride, v_out27);

        bv_s20 = _mm512_add_ps(bv_t10, bv_t11);
        bv_s21 = _mm512_add_ps(bv_t12, bv_s20);
        bv_s22 = _mm512_add_ps(bv_t7, bv_t8);
        bv_s23 = _mm512_add_ps(bv_t9, bv_s13);
        bv_s24 = _mm512_sub_ps(bv_s22, bv_s23);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm512_sub_ps(bv_s24, bv_s21);
        STR_512_S(curr_out, v_out_stride, v_out5);
        // Output pt 26: X(25)
        curr_out = out + out_strides[25];
        v_out25 = NEGATE_512_S(_mm512_add_ps(bv_s21, bv_s24));
        STR_512_S(curr_out, v_out_stride, v_out25);

        bv_s25 = _mm512_sub_ps(bv_t13, bv_t14);
        bv_s26 = _mm512_sub_ps(bv_s14, bv_t15);
        bv_s27 = _mm512_add_ps(bv_s25, bv_s26);
        bv_s28 = _mm512_add_ps(bv_t16, bv_t17);
        bv_s29 = _mm512_sub_ps(bv_t18, bv_s28);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = _mm512_add_ps(bv_s27, bv_s29);
        STR_512_S(curr_out, v_out_stride, v_out7);
        // Output pt 24: X(23)
        curr_out = out + out_strides[23];
        v_out23 = _mm512_sub_ps(bv_s27, bv_s29);
        STR_512_S(curr_out, v_out_stride, v_out23);

        bv_s30 = _mm512_sub_ps(bv_t22, bv_t23);
        bv_s31 = _mm512_sub_ps(bv_s30, bv_t24);
        bv_s32 = _mm512_add_ps(bv_t19, bv_t20);
        bv_s33 = _mm512_add_ps(bv_t21, bv_s13);
        bv_s34 = _mm512_sub_ps(bv_s33, bv_s32);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm512_add_ps(bv_s31, bv_s34);
        STR_512_S(curr_out, v_out_stride, v_out9);
        // Output pt 22: X(21)
        curr_out = out + out_strides[21];
        v_out21 = _mm512_sub_ps(bv_s31, bv_s34);
        STR_512_S(curr_out, v_out_stride, v_out21);

        bv_s35 = _mm512_sub_ps(bv_t25, bv_t26);
        bv_s36 = _mm512_sub_ps(bv_t27, bv_s14);
        bv_s37 = _mm512_add_ps(bv_s35, bv_s36);
        bv_s38 = _mm512_sub_ps(bv_t29, bv_t28);
        bv_s39 = _mm512_add_ps(bv_t30, bv_s38);
        // Output pt 12: X(11)
        curr_out = out + out_strides[11];
        v_out11 = _mm512_add_ps(bv_s37, bv_s39);
        STR_512_S(curr_out, v_out_stride, v_out11);
        // Output pt 20: X(19)
        curr_out = out + out_strides[19];
        v_out19 = _mm512_sub_ps(bv_s37, bv_s39);
        STR_512_S(curr_out, v_out_stride, v_out19);

        bv_s40 = _mm512_add_ps(bv_t34, bv_t35);
        bv_s41 = _mm512_sub_ps(bv_t36, bv_s40);
        bv_s42 = _mm512_add_ps(bv_t31, bv_t32);
        bv_s43 = _mm512_add_ps(bv_t33, bv_s13);
        bv_s44 = _mm512_sub_ps(bv_s42, bv_s43);
        // Output pt 14: X(13)
        curr_out = out + out_strides[13];
        v_out13 = _mm512_add_ps(bv_s41, bv_s44);
        STR_512_S(curr_out, v_out_stride, v_out13);
        // Output pt 18: X(17)
        curr_out = out + out_strides[17];
        v_out17 = _mm512_sub_ps(bv_s41, bv_s44);
        STR_512_S(curr_out, v_out_stride, v_out17);

        bv_s45 = _mm512_add_ps(bv_s1, bv_s5);
        bv_s46 = _mm512_add_ps(bv_in6, bv_s9);
        bv_s47 = _mm512_add_ps(bv_s45, bv_s46);
        // Output pt 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm512_mul_ps(bv_s47, v_CRTM_14_7);
        STR_512_S(curr_out, v_out_stride, v_out1);

        bv_s48 = _mm512_add_ps(bv_s3, bv_s11);
        bv_s49 = _mm512_add_ps(bv_in7, bv_s7);
        bv_s50 = _mm512_sub_ps(bv_s49, bv_s48);
        // Output pt 16: X(15)
        curr_out = out + out_strides[15];
        v_out15 = _mm512_mul_ps(bv_s50, v_CRTM_14_7);
        STR_512_S(curr_out, v_out_stride, v_out15);

        in = in + (v_in_stride << 4);
        out = out + (v_out_stride << 4);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_256_S)
    {
        /* Standard DFT */
        __m256 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
               av_in8, av_in9, av_in10, av_in11, av_in12, av_in13;
        __m256 av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8, av_s9,
               av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16, av_s17,
               av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24, av_s25,
               av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32, av_s33,
               av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40, av_s41,
               av_s42, av_s43, av_s44, av_s45, av_s46, av_s47, av_s48;
        __m256 av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8, av_t9,
               av_t10, av_t11, av_t12, av_t13, av_t14, av_t15, av_t16, av_t17,
               av_t18, av_t19, av_t20, av_t21, av_t22, av_t23, av_t24, av_t25,
               av_t26, av_t27, av_t28, av_t29, av_t30, av_t31, av_t32, av_t33,
               av_t34, av_t35, av_t36, av_t37, av_t38;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21,
               v_out22, v_out23, v_out24, v_out25, v_out26, v_out27;

        curr_in = in;
        curr_out = out;

        __m256 v256_CRTM_14_1 = _mm512_castps512_ps256(v_CRTM_14_1);
        __m256 v256_CRTM_14_2 = _mm512_castps512_ps256(v_CRTM_14_2);
        __m256 v256_CRTM_14_3 = _mm512_castps512_ps256(v_CRTM_14_3);
        __m256 v256_CRTM_14_4 = _mm512_castps512_ps256(v_CRTM_14_4);
        __m256 v256_CRTM_14_5 = _mm512_castps512_ps256(v_CRTM_14_5);
        __m256 v256_CRTM_14_6 = _mm512_castps512_ps256(v_CRTM_14_6);
        __m256 v256_CRTM_14_7 = _mm512_castps512_ps256(v_CRTM_14_7);

        // Input point 1: x(0)
        LDR_256_S(curr_in, v_in_stride, av_in0);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x256_S(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in + in_strides[7];
        LDRI_2x256_S(curr_in, v_in_stride, av_in3, av_in4);
        // Input point 12: x(11) & Input point 13: x(12)
        curr_in = in + in_strides[11];
        LDRI_2x256_S(curr_in, v_in_stride, av_in5, av_in6);
        // Input point 16: x(15) & Input point 17: x(16)
        curr_in = in + in_strides[15];
        LDRI_2x256_S(curr_in, v_in_stride, av_in7, av_in8);
        // Input point 20: x(19) & Input point 21: x(20)
        curr_in = in + in_strides[19];
        LDRI_2x256_S(curr_in, v_in_stride, av_in9, av_in10);
        // Input point 24: x(23) & Input point 25: x(24)
        curr_in = in + in_strides[23];
        LDRI_2x256_S(curr_in, v_in_stride, av_in11, av_in12);
        // Input point 28: x(27)
        curr_in = in + in_strides[27];
        LDR_256_S(curr_in, v_in_stride, av_in13);

        av_s1 = _mm256_sub_ps(av_in0, av_in13);
        av_s2 = _mm256_add_ps(av_in0, av_in13);
        av_s3 = _mm256_sub_ps(av_in11, av_in1);
        av_s4 = _mm256_add_ps(av_in1, av_in11);
        av_s5 = _mm256_sub_ps(av_in2, av_in12);
        av_s6 = _mm256_add_ps(av_in2, av_in12);
        av_s7 = _mm256_sub_ps(av_in3, av_in9);
        av_s8 = _mm256_add_ps(av_in3, av_in9);
        av_s9 = _mm256_sub_ps(av_in4, av_in10);
        av_s10 = _mm256_add_ps(av_in4, av_in10);
        av_s11 = _mm256_sub_ps(av_in7, av_in5);
        av_s12 = _mm256_add_ps(av_in5, av_in7);
        av_s13 = _mm256_sub_ps(av_in6, av_in8);
        av_s14 = _mm256_add_ps(av_in6, av_in8);

        av_s27 = _mm256_add_ps(av_s12, av_s4);
        av_s28 = _mm256_add_ps(av_s27, av_s8);
        av_t37 = _mm256_mul_ps(v256_CRTM_14_7, av_s28);
        av_s29 = _mm256_add_ps(av_s3, av_s7);
        av_s30 = _mm256_add_ps(av_s29, av_s11);
        av_t38 = _mm256_mul_ps(v256_CRTM_14_7, av_s30);
        // Output pt 1: X(0)
        v_out0 = _mm256_add_ps(av_t37, av_s2);
        STR_256_S(curr_out, v_out_stride, v_out0);
        // Output pt 15: X(14)
        curr_out = out + out_strides[14];
        v_out14 = _mm256_add_ps(av_t38, av_s1);
        STR_256_S(curr_out, v_out_stride, v_out14);

        av_t1 = _mm256_mul_ps(v256_CRTM_14_1, av_s6);
        av_t2 = _mm256_mul_ps(v256_CRTM_14_3, av_s10);
        av_t3 = _mm256_mul_ps(v256_CRTM_14_5, av_s14);
        av_t4 = _mm256_mul_ps(v256_CRTM_14_2, av_s3);
        av_t5 = _mm256_mul_ps(v256_CRTM_14_4, av_s7);
        av_t6 = _mm256_mul_ps(v256_CRTM_14_6, av_s11);

        av_s31 = _mm256_sub_ps(av_t5, av_t6);
        av_s32 = _mm256_sub_ps(av_s1, av_t4);
        av_s33 = _mm256_add_ps(av_t1, av_t2);

        av_s15 = _mm256_add_ps(av_s31, av_s32);
        av_s16 = _mm256_add_ps(av_s33, av_t3);
        // Output point 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm256_sub_ps(av_s15, av_s16);
        STR_256_S(curr_out, v_out_stride, v_out2);
        // Output point 27: X(26)
        curr_out = out + out_strides[26];
        v_out26 = _mm256_add_ps(av_s15, av_s16);
        STR_256_S(curr_out, v_out_stride, v_out26);

        av_t7 = _mm256_mul_ps(v256_CRTM_14_1, av_s13);
        av_t8 = _mm256_mul_ps(v256_CRTM_14_3, av_s5);
        av_t9 = _mm256_mul_ps(v256_CRTM_14_5, av_s9);

        av_t10 = _mm256_mul_ps(v256_CRTM_14_2, av_s12);
        av_t11 = _mm256_mul_ps(v256_CRTM_14_4, av_s4);
        av_t12 = _mm256_mul_ps(v256_CRTM_14_6, av_s8);

        av_s34 = _mm256_sub_ps(av_s2, av_t10);
        av_s35 = _mm256_sub_ps(av_t11, av_t12);
        av_s36 = _mm256_add_ps(av_t7, av_t8);

        av_s17 = av_s34 + av_s35;
        av_s18 = _mm256_add_ps(av_s36, av_t9);
        // Output point 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm256_sub_ps(av_s17, av_s18);
        STR_256_S(curr_out, v_out_stride, v_out4);
        // Output point 25: X(24)
        curr_out = out + out_strides[24];
        v_out24 = _mm256_add_ps(av_s17, av_s18);
        STR_256_S(curr_out, v_out_stride, v_out24);

        av_t13 = _mm256_mul_ps(v256_CRTM_14_1, av_s10);
        av_t14 = _mm256_mul_ps(v256_CRTM_14_3, av_s14);
        av_t15 = _mm256_mul_ps(v256_CRTM_14_5, av_s6);
        av_t16 = _mm256_mul_ps(v256_CRTM_14_2, av_s7);
        av_t17 = _mm256_mul_ps(v256_CRTM_14_4, av_s11);
        av_t18 = _mm256_mul_ps(v256_CRTM_14_6, av_s3);

        av_s37 = _mm256_sub_ps(av_s1, av_t16);
        av_s38 = _mm256_sub_ps(av_t17, av_t18);
        av_s39 = _mm256_sub_ps(av_t14, av_t15);
        av_s19 = _mm256_add_ps(av_s37, av_s38);
        av_s20 = _mm256_sub_ps(av_s39, av_t13);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm256_add_ps(av_s19, av_s20);
        STR_256_S(curr_out, v_out_stride, v_out6);
        // Output point 23: X(22)
        curr_out = out + out_strides[22];
        v_out22 = _mm256_sub_ps(av_s19, av_s20);
        STR_256_S(curr_out, v_out_stride, v_out22);

        av_t19 = _mm256_mul_ps(v256_CRTM_14_1, av_s9);
        av_t20 = _mm256_mul_ps(v256_CRTM_14_3, av_s13);
        av_t21 = _mm256_mul_ps(v256_CRTM_14_5, av_s5);
        av_t22 = _mm256_mul_ps(v256_CRTM_14_2, av_s8);
        av_t23 = _mm256_mul_ps(v256_CRTM_14_4, av_s12);
        av_t24 = _mm256_mul_ps(v256_CRTM_14_6, av_s4);

        av_s40 = _mm256_sub_ps(av_s2, av_t22);
        av_s41 = _mm256_sub_ps(av_t23, av_t24);
        av_s42 = _mm256_add_ps(av_t19, av_t20);
        av_s21 = _mm256_add_ps(av_s40, av_s41);
        av_s22 = _mm256_sub_ps(av_s42, av_t21);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm256_add_ps(av_s21, av_s22);
        STR_256_S(curr_out, v_out_stride, v_out8);
        // Output point 21: X(20)
        curr_out = out + out_strides[20];
        v_out20 = _mm256_sub_ps(av_s21, av_s22);
        STR_256_S(curr_out, v_out_stride, v_out20);

        av_t25 = _mm256_mul_ps(v256_CRTM_14_1, av_s14);
        av_t26 = _mm256_mul_ps(v256_CRTM_14_3, av_s6);
        av_t27 = _mm256_mul_ps(v256_CRTM_14_5, av_s10);
        av_t28 = _mm256_mul_ps(v256_CRTM_14_2, av_s11);
        av_t29 = _mm256_mul_ps(v256_CRTM_14_4, av_s3);
        av_t30 = _mm256_mul_ps(v256_CRTM_14_6, av_s7);

        av_s43 = _mm256_sub_ps(av_s1, av_t28);
        av_s44 = _mm256_sub_ps(av_t29, av_t30);
        av_s45 = _mm256_sub_ps(av_t27, av_t25);
        av_s23 = _mm256_add_ps(av_s43, av_s44);
        av_s24 = _mm256_sub_ps(av_s45, av_t26);
        // Output pt 11: X(10)
        curr_out = out + out_strides[10];
        v_out10 = _mm256_add_ps(av_s23, av_s24);
        STR_256_S(curr_out, v_out_stride, v_out10);
        // Output pt 19: X(18)
        curr_out = out + out_strides[18];
        v_out18 = _mm256_sub_ps(av_s23, av_s24);
        STR_256_S(curr_out, v_out_stride, v_out18);

        av_t31 = _mm256_mul_ps(v256_CRTM_14_1, av_s5);
        av_t32 = _mm256_mul_ps(v256_CRTM_14_3, av_s9);
        av_t33 = _mm256_mul_ps(v256_CRTM_14_5, av_s13);
        av_t34 = _mm256_mul_ps(v256_CRTM_14_2, av_s4);
        av_t35 = _mm256_mul_ps(v256_CRTM_14_4, av_s8);
        av_t36 = _mm256_mul_ps(v256_CRTM_14_6, av_s12);

        av_s46 = _mm256_sub_ps(av_s2, av_t34);
        av_s47 = _mm256_sub_ps(av_t35, av_t36);
        av_s48 = _mm256_sub_ps(av_t32, av_t31);
        av_s25 = _mm256_add_ps(av_s46, av_s47);
        av_s26 = _mm256_sub_ps(av_s48, av_t33);
        // Output pt 13: X(12)
        curr_out = out + out_strides[12];
        v_out12 = _mm256_add_ps(av_s25, av_s26);
        STR_256_S(curr_out, v_out_stride, v_out12);
        // Output pt 17: X(16)
        curr_out = out + out_strides[16];
        v_out16 = _mm256_sub_ps(av_s25, av_s26);
        STR_256_S(curr_out, v_out_stride, v_out16);

        /* Shifted DFT */
        __m256 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8, bv_in9, bv_in10, bv_in11, bv_in12, bv_in13;
        __m256 bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
               bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17,
               bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24, bv_s25,
               bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32, bv_s33,
               bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40, bv_s41,
               bv_s42, bv_s43, bv_s44, bv_s45, bv_s46, bv_s47, bv_s48, bv_s49,
               bv_s50;
        __m256 bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
               bv_t10, bv_t11, bv_t12, bv_t13, bv_t14, bv_t15, bv_t16, bv_t17,
               bv_t18, bv_t19, bv_t20, bv_t21, bv_t22, bv_t23, bv_t24, bv_t25,
               bv_t26, bv_t27, bv_t28, bv_t29, bv_t30, bv_t31, bv_t32, bv_t33,
               bv_t34, bv_t35, bv_t36;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x256_S(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDRI_2x256_S(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDRI_2x256_S(curr_in, v_in_stride, bv_in4, bv_in5);
        // Input point 14: x(13) & Input point 15: x(14)
        curr_in = in + in_strides[13];
        LDRI_2x256_S(curr_in, v_in_stride, bv_in6, bv_in7);
        // Input point 18: x(17) & Input point 19: x(18)
        curr_in = in + in_strides[17];
        LDRI_2x256_S(curr_in, v_in_stride, bv_in8, bv_in9);
        // Input point 22: x(21) & Input point 23: x(22)
        curr_in = in + in_strides[21];
        LDRI_2x256_S(curr_in, v_in_stride, bv_in10, bv_in11);
        // Input point 26: x(25) & Input point 27: x(26)
        curr_in = in + in_strides[25];
        LDRI_2x256_S(curr_in, v_in_stride, bv_in12, bv_in13);

        bv_s1 = _mm256_add_ps(bv_in0, bv_in12);
        bv_s2 = _mm256_sub_ps(bv_in0, bv_in12);
        bv_s3 = _mm256_add_ps(bv_in1, bv_in13);
        bv_s4 = _mm256_sub_ps(bv_in1, bv_in13);
        bv_s5 = _mm256_add_ps(bv_in2, bv_in10);
        bv_s6 = _mm256_sub_ps(bv_in2, bv_in10);
        bv_s7 = _mm256_add_ps(bv_in3, bv_in11);
        bv_s8 = _mm256_sub_ps(bv_in3, bv_in11);
        bv_s9 = _mm256_add_ps(bv_in4, bv_in8);
        bv_s10 = _mm256_sub_ps(bv_in4, bv_in8);
        bv_s11 = _mm256_add_ps(bv_in5, bv_in9);
        bv_s12 = _mm256_sub_ps(bv_in5, bv_in9);

        bv_t1 = _mm256_mul_ps(v256_CRTM_14_6, bv_s3);
        bv_t2 = _mm256_mul_ps(v256_CRTM_14_2, bv_s11);
        bv_t3 = _mm256_mul_ps(v256_CRTM_14_4, bv_s7);
        bv_t4 = _mm256_mul_ps(v256_CRTM_14_5, bv_s2);
        bv_t5 = _mm256_mul_ps(v256_CRTM_14_1, bv_s10);
        bv_t6 = _mm256_mul_ps(v256_CRTM_14_3, bv_s6);

        bv_t7 = _mm256_mul_ps(v256_CRTM_14_6, bv_s5);
        bv_t8 = _mm256_mul_ps(v256_CRTM_14_2, bv_s1);
        bv_t9 = _mm256_mul_ps(v256_CRTM_14_4, bv_s9);
        bv_t10 = _mm256_mul_ps(v256_CRTM_14_5, bv_s8);
        bv_t11 = _mm256_mul_ps(v256_CRTM_14_1, bv_s4);
        bv_t12 = _mm256_mul_ps(v256_CRTM_14_3, bv_s12);

        bv_t13 = _mm256_mul_ps(v256_CRTM_14_6, bv_s11);
        bv_t14 = _mm256_mul_ps(v256_CRTM_14_2, bv_s7);
        bv_t15 = _mm256_mul_ps(v256_CRTM_14_4, bv_s3);
        bv_t16 = _mm256_mul_ps(v256_CRTM_14_5, bv_s10);
        bv_t17 = _mm256_mul_ps(v256_CRTM_14_1, bv_s6);
        bv_t18 = _mm256_mul_ps(v256_CRTM_14_3, bv_s2);

        bv_t19 = _mm256_mul_ps(v256_CRTM_14_6, bv_s9);
        bv_t20 = _mm256_mul_ps(v256_CRTM_14_2, bv_s5);
        bv_t21 = _mm256_mul_ps(v256_CRTM_14_4, bv_s1);
        bv_t22 = _mm256_mul_ps(v256_CRTM_14_5, bv_s12);
        bv_t23 = _mm256_mul_ps(v256_CRTM_14_1, bv_s8);
        bv_t24 = _mm256_mul_ps(v256_CRTM_14_3, bv_s4);

        bv_t25 = _mm256_mul_ps(v256_CRTM_14_6, bv_s7);
        bv_t26 = _mm256_mul_ps(v256_CRTM_14_2, bv_s3);
        bv_t27 = _mm256_mul_ps(v256_CRTM_14_4, bv_s11);
        bv_t28 = _mm256_mul_ps(v256_CRTM_14_5, bv_s6);
        bv_t29 = _mm256_mul_ps(v256_CRTM_14_1, bv_s2);
        bv_t30 = _mm256_mul_ps(v256_CRTM_14_3, bv_s10);

        bv_t31 = _mm256_mul_ps(v256_CRTM_14_6, bv_s1);
        bv_t32 = _mm256_mul_ps(v256_CRTM_14_2, bv_s9);
        bv_t33 = _mm256_mul_ps(v256_CRTM_14_4, bv_s5);
        bv_t34 = _mm256_mul_ps(v256_CRTM_14_5, bv_s4);
        bv_t35 = _mm256_mul_ps(v256_CRTM_14_1, bv_s12);
        bv_t36 = _mm256_mul_ps(v256_CRTM_14_3, bv_s8);

        bv_s13 = _mm256_add_ps(bv_in6, bv_in6);
        bv_s14 = _mm256_add_ps(bv_in7, bv_in7);

        bv_s15 = _mm256_add_ps(bv_t1, bv_t2);
        bv_s16 = _mm256_add_ps(bv_t3, bv_s14);
        bv_s17 = _mm256_add_ps(bv_s15, bv_s16);
        bv_s18 = _mm256_add_ps(bv_t4, bv_t5);
        bv_s19 = _mm256_add_ps(bv_t6, bv_s18);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm256_sub_ps(bv_s19, bv_s17);
        STR_256_S(curr_out, v_out_stride, v_out3);
        // Output pt 28: X(27)
        curr_out = out + out_strides[27];
        v_out27 = NEGATE_256_S(_mm256_add_ps(bv_s17, bv_s19));
        STR_256_S(curr_out, v_out_stride, v_out27);

        bv_s20 = _mm256_add_ps(bv_t10, bv_t11);
        bv_s21 = _mm256_add_ps(bv_t12, bv_s20);
        bv_s22 = _mm256_add_ps(bv_t7, bv_t8);
        bv_s23 = _mm256_add_ps(bv_t9, bv_s13);
        bv_s24 = _mm256_sub_ps(bv_s22, bv_s23);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm256_sub_ps(bv_s24, bv_s21);
        STR_256_S(curr_out, v_out_stride, v_out5);
        // Output pt 26: X(25)
        curr_out = out + out_strides[25];
        v_out25 = NEGATE_256_S(_mm256_add_ps(bv_s21, bv_s24));
        STR_256_S(curr_out, v_out_stride, v_out25);

        bv_s25 = _mm256_sub_ps(bv_t13, bv_t14);
        bv_s26 = _mm256_sub_ps(bv_s14, bv_t15);
        bv_s27 = _mm256_add_ps(bv_s25, bv_s26);
        bv_s28 = _mm256_add_ps(bv_t16, bv_t17);
        bv_s29 = _mm256_sub_ps(bv_t18, bv_s28);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = _mm256_add_ps(bv_s27, bv_s29);
        STR_256_S(curr_out, v_out_stride, v_out7);
        // Output pt 24: X(23)
        curr_out = out + out_strides[23];
        v_out23 = _mm256_sub_ps(bv_s27, bv_s29);
        STR_256_S(curr_out, v_out_stride, v_out23);

        bv_s30 = _mm256_sub_ps(bv_t22, bv_t23);
        bv_s31 = _mm256_sub_ps(bv_s30, bv_t24);
        bv_s32 = _mm256_add_ps(bv_t19, bv_t20);
        bv_s33 = _mm256_add_ps(bv_t21, bv_s13);
        bv_s34 = _mm256_sub_ps(bv_s33, bv_s32);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm256_add_ps(bv_s31, bv_s34);
        STR_256_S(curr_out, v_out_stride, v_out9);
        // Output pt 22: X(21)
        curr_out = out + out_strides[21];
        v_out21 = _mm256_sub_ps(bv_s31, bv_s34);
        STR_256_S(curr_out, v_out_stride, v_out21);

        bv_s35 = _mm256_sub_ps(bv_t25, bv_t26);
        bv_s36 = _mm256_sub_ps(bv_t27, bv_s14);
        bv_s37 = _mm256_add_ps(bv_s35, bv_s36);
        bv_s38 = _mm256_sub_ps(bv_t29, bv_t28);
        bv_s39 = _mm256_add_ps(bv_t30, bv_s38);
        // Output pt 12: X(11)
        curr_out = out + out_strides[11];
        v_out11 = _mm256_add_ps(bv_s37, bv_s39);
        STR_256_S(curr_out, v_out_stride, v_out11);
        // Output pt 20: X(19)
        curr_out = out + out_strides[19];
        v_out19 = _mm256_sub_ps(bv_s37, bv_s39);
        STR_256_S(curr_out, v_out_stride, v_out19);

        bv_s40 = _mm256_add_ps(bv_t34, bv_t35);
        bv_s41 = _mm256_sub_ps(bv_t36, bv_s40);
        bv_s42 = _mm256_add_ps(bv_t31, bv_t32);
        bv_s43 = _mm256_add_ps(bv_t33, bv_s13);
        bv_s44 = _mm256_sub_ps(bv_s42, bv_s43);
        // Output pt 14: X(13)
        curr_out = out + out_strides[13];
        v_out13 = _mm256_add_ps(bv_s41, bv_s44);
        STR_256_S(curr_out, v_out_stride, v_out13);
        // Output pt 18: X(17)
        curr_out = out + out_strides[17];
        v_out17 = _mm256_sub_ps(bv_s41, bv_s44);
        STR_256_S(curr_out, v_out_stride, v_out17);

        bv_s45 = _mm256_add_ps(bv_s1, bv_s5);
        bv_s46 = _mm256_add_ps(bv_in6, bv_s9);
        bv_s47 = _mm256_add_ps(bv_s45, bv_s46);
        // Output pt 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm256_mul_ps(bv_s47, v256_CRTM_14_7);
        STR_256_S(curr_out, v_out_stride, v_out1);

        bv_s48 = _mm256_add_ps(bv_s3, bv_s11);
        bv_s49 = _mm256_add_ps(bv_in7, bv_s7);
        bv_s50 = _mm256_sub_ps(bv_s49, bv_s48);
        // Output pt 16: X(15)
        curr_out = out + out_strides[15];
        v_out15 = _mm256_mul_ps(bv_s50, v256_CRTM_14_7);
        STR_256_S(curr_out, v_out_stride, v_out15);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_S)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
               av_in8, av_in9, av_in10, av_in11, av_in12, av_in13;
        __m128 av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8, av_s9,
               av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16, av_s17,
               av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24, av_s25,
               av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32, av_s33,
               av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40, av_s41,
               av_s42, av_s43, av_s44, av_s45, av_s46, av_s47, av_s48;
        __m128 av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8, av_t9,
               av_t10, av_t11, av_t12, av_t13, av_t14, av_t15, av_t16, av_t17,
               av_t18, av_t19, av_t20, av_t21, av_t22, av_t23, av_t24, av_t25,
               av_t26, av_t27, av_t28, av_t29, av_t30, av_t31, av_t32, av_t33,
               av_t34, av_t35, av_t36, av_t37, av_t38;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21,
               v_out22, v_out23, v_out24, v_out25, v_out26, v_out27;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_14_1 = _mm512_castps512_ps128(v_CRTM_14_1);
        __m128 v128_CRTM_14_2 = _mm512_castps512_ps128(v_CRTM_14_2);
        __m128 v128_CRTM_14_3 = _mm512_castps512_ps128(v_CRTM_14_3);
        __m128 v128_CRTM_14_4 = _mm512_castps512_ps128(v_CRTM_14_4);
        __m128 v128_CRTM_14_5 = _mm512_castps512_ps128(v_CRTM_14_5);
        __m128 v128_CRTM_14_6 = _mm512_castps512_ps128(v_CRTM_14_6);
        __m128 v128_CRTM_14_7 = _mm512_castps512_ps128(v_CRTM_14_7);

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_stride, av_in0);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x128_S(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in + in_strides[7];
        LDRI_2x128_S(curr_in, v_in_stride, av_in3, av_in4);
        // Input point 12: x(11) & Input point 13: x(12)
        curr_in = in + in_strides[11];
        LDRI_2x128_S(curr_in, v_in_stride, av_in5, av_in6);
        // Input point 16: x(15) & Input point 17: x(16)
        curr_in = in + in_strides[15];
        LDRI_2x128_S(curr_in, v_in_stride, av_in7, av_in8);
        // Input point 20: x(19) & Input point 21: x(20)
        curr_in = in + in_strides[19];
        LDRI_2x128_S(curr_in, v_in_stride, av_in9, av_in10);
        // Input point 24: x(23) & Input point 25: x(24)
        curr_in = in + in_strides[23];
        LDRI_2x128_S(curr_in, v_in_stride, av_in11, av_in12);
        // Input point 28: x(27)
        curr_in = in + in_strides[27];
        LDR_128_S(curr_in, v_in_stride, av_in13);

        av_s1 = _mm_sub_ps(av_in0, av_in13);
        av_s2 = _mm_add_ps(av_in0, av_in13);
        av_s3 = _mm_sub_ps(av_in11, av_in1);
        av_s4 = _mm_add_ps(av_in1, av_in11);
        av_s5 = _mm_sub_ps(av_in2, av_in12);
        av_s6 = _mm_add_ps(av_in2, av_in12);
        av_s7 = _mm_sub_ps(av_in3, av_in9);
        av_s8 = _mm_add_ps(av_in3, av_in9);
        av_s9 = _mm_sub_ps(av_in4, av_in10);
        av_s10 = _mm_add_ps(av_in4, av_in10);
        av_s11 = _mm_sub_ps(av_in7, av_in5);
        av_s12 = _mm_add_ps(av_in5, av_in7);
        av_s13 = _mm_sub_ps(av_in6, av_in8);
        av_s14 = _mm_add_ps(av_in6, av_in8);

        av_s27 = _mm_add_ps(av_s12, av_s4);
        av_s28 = _mm_add_ps(av_s27, av_s8);
        av_t37 = _mm_mul_ps(v128_CRTM_14_7, av_s28);
        av_s29 = _mm_add_ps(av_s3, av_s7);
        av_s30 = _mm_add_ps(av_s29, av_s11);
        av_t38 = _mm_mul_ps(v128_CRTM_14_7, av_s30);
        // Output pt 1: X(0)
        v_out0 = _mm_add_ps(av_t37, av_s2);
        STR_128_S(curr_out, v_out_stride, v_out0);
        // Output pt 15: X(14)
        curr_out = out + out_strides[14];
        v_out14 = _mm_add_ps(av_t38, av_s1);
        STR_128_S(curr_out, v_out_stride, v_out14);

        av_t1 = _mm_mul_ps(v128_CRTM_14_1, av_s6);
        av_t2 = _mm_mul_ps(v128_CRTM_14_3, av_s10);
        av_t3 = _mm_mul_ps(v128_CRTM_14_5, av_s14);
        av_t4 = _mm_mul_ps(v128_CRTM_14_2, av_s3);
        av_t5 = _mm_mul_ps(v128_CRTM_14_4, av_s7);
        av_t6 = _mm_mul_ps(v128_CRTM_14_6, av_s11);

        av_s31 = _mm_sub_ps(av_t5, av_t6);
        av_s32 = _mm_sub_ps(av_s1, av_t4);
        av_s33 = _mm_add_ps(av_t1, av_t2);

        av_s15 = _mm_add_ps(av_s31, av_s32);
        av_s16 = _mm_add_ps(av_s33, av_t3);
        // Output point 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm_sub_ps(av_s15, av_s16);
        STR_128_S(curr_out, v_out_stride, v_out2);
        // Output point 27: X(26)
        curr_out = out + out_strides[26];
        v_out26 = _mm_add_ps(av_s15, av_s16);
        STR_128_S(curr_out, v_out_stride, v_out26);

        av_t7 = _mm_mul_ps(v128_CRTM_14_1, av_s13);
        av_t8 = _mm_mul_ps(v128_CRTM_14_3, av_s5);
        av_t9 = _mm_mul_ps(v128_CRTM_14_5, av_s9);

        av_t10 = _mm_mul_ps(v128_CRTM_14_2, av_s12);
        av_t11 = _mm_mul_ps(v128_CRTM_14_4, av_s4);
        av_t12 = _mm_mul_ps(v128_CRTM_14_6, av_s8);

        av_s34 = _mm_sub_ps(av_s2, av_t10);
        av_s35 = _mm_sub_ps(av_t11, av_t12);
        av_s36 = _mm_add_ps(av_t7, av_t8);

        av_s17 = av_s34 + av_s35;
        av_s18 = _mm_add_ps(av_s36, av_t9);
        // Output point 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm_sub_ps(av_s17, av_s18);
        STR_128_S(curr_out, v_out_stride, v_out4);
        // Output point 25: X(24)
        curr_out = out + out_strides[24];
        v_out24 = _mm_add_ps(av_s17, av_s18);
        STR_128_S(curr_out, v_out_stride, v_out24);

        av_t13 = _mm_mul_ps(v128_CRTM_14_1, av_s10);
        av_t14 = _mm_mul_ps(v128_CRTM_14_3, av_s14);
        av_t15 = _mm_mul_ps(v128_CRTM_14_5, av_s6);
        av_t16 = _mm_mul_ps(v128_CRTM_14_2, av_s7);
        av_t17 = _mm_mul_ps(v128_CRTM_14_4, av_s11);
        av_t18 = _mm_mul_ps(v128_CRTM_14_6, av_s3);

        av_s37 = _mm_sub_ps(av_s1, av_t16);
        av_s38 = _mm_sub_ps(av_t17, av_t18);
        av_s39 = _mm_sub_ps(av_t14, av_t15);
        av_s19 = _mm_add_ps(av_s37, av_s38);
        av_s20 = _mm_sub_ps(av_s39, av_t13);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm_add_ps(av_s19, av_s20);
        STR_128_S(curr_out, v_out_stride, v_out6);
        // Output point 23: X(22)
        curr_out = out + out_strides[22];
        v_out22 = _mm_sub_ps(av_s19, av_s20);
        STR_128_S(curr_out, v_out_stride, v_out22);

        av_t19 = _mm_mul_ps(v128_CRTM_14_1, av_s9);
        av_t20 = _mm_mul_ps(v128_CRTM_14_3, av_s13);
        av_t21 = _mm_mul_ps(v128_CRTM_14_5, av_s5);
        av_t22 = _mm_mul_ps(v128_CRTM_14_2, av_s8);
        av_t23 = _mm_mul_ps(v128_CRTM_14_4, av_s12);
        av_t24 = _mm_mul_ps(v128_CRTM_14_6, av_s4);

        av_s40 = _mm_sub_ps(av_s2, av_t22);
        av_s41 = _mm_sub_ps(av_t23, av_t24);
        av_s42 = _mm_add_ps(av_t19, av_t20);
        av_s21 = _mm_add_ps(av_s40, av_s41);
        av_s22 = _mm_sub_ps(av_s42, av_t21);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm_add_ps(av_s21, av_s22);
        STR_128_S(curr_out, v_out_stride, v_out8);
        // Output point 21: X(20)
        curr_out = out + out_strides[20];
        v_out20 = _mm_sub_ps(av_s21, av_s22);
        STR_128_S(curr_out, v_out_stride, v_out20);

        av_t25 = _mm_mul_ps(v128_CRTM_14_1, av_s14);
        av_t26 = _mm_mul_ps(v128_CRTM_14_3, av_s6);
        av_t27 = _mm_mul_ps(v128_CRTM_14_5, av_s10);
        av_t28 = _mm_mul_ps(v128_CRTM_14_2, av_s11);
        av_t29 = _mm_mul_ps(v128_CRTM_14_4, av_s3);
        av_t30 = _mm_mul_ps(v128_CRTM_14_6, av_s7);

        av_s43 = _mm_sub_ps(av_s1, av_t28);
        av_s44 = _mm_sub_ps(av_t29, av_t30);
        av_s45 = _mm_sub_ps(av_t27, av_t25);
        av_s23 = _mm_add_ps(av_s43, av_s44);
        av_s24 = _mm_sub_ps(av_s45, av_t26);
        // Output pt 11: X(10)
        curr_out = out + out_strides[10];
        v_out10 = _mm_add_ps(av_s23, av_s24);
        STR_128_S(curr_out, v_out_stride, v_out10);
        // Output pt 19: X(18)
        curr_out = out + out_strides[18];
        v_out18 = _mm_sub_ps(av_s23, av_s24);
        STR_128_S(curr_out, v_out_stride, v_out18);

        av_t31 = _mm_mul_ps(v128_CRTM_14_1, av_s5);
        av_t32 = _mm_mul_ps(v128_CRTM_14_3, av_s9);
        av_t33 = _mm_mul_ps(v128_CRTM_14_5, av_s13);
        av_t34 = _mm_mul_ps(v128_CRTM_14_2, av_s4);
        av_t35 = _mm_mul_ps(v128_CRTM_14_4, av_s8);
        av_t36 = _mm_mul_ps(v128_CRTM_14_6, av_s12);

        av_s46 = _mm_sub_ps(av_s2, av_t34);
        av_s47 = _mm_sub_ps(av_t35, av_t36);
        av_s48 = _mm_sub_ps(av_t32, av_t31);
        av_s25 = _mm_add_ps(av_s46, av_s47);
        av_s26 = _mm_sub_ps(av_s48, av_t33);
        // Output pt 13: X(12)
        curr_out = out + out_strides[12];
        v_out12 = _mm_add_ps(av_s25, av_s26);
        STR_128_S(curr_out, v_out_stride, v_out12);
        // Output pt 17: X(16)
        curr_out = out + out_strides[16];
        v_out16 = _mm_sub_ps(av_s25, av_s26);
        STR_128_S(curr_out, v_out_stride, v_out16);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8, bv_in9, bv_in10, bv_in11, bv_in12, bv_in13;
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
               bv_t26, bv_t27, bv_t28, bv_t29, bv_t30, bv_t31, bv_t32, bv_t33,
               bv_t34, bv_t35, bv_t36;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in4, bv_in5);
        // Input point 14: x(13) & Input point 15: x(14)
        curr_in = in + in_strides[13];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in6, bv_in7);
        // Input point 18: x(17) & Input point 19: x(18)
        curr_in = in + in_strides[17];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in8, bv_in9);
        // Input point 22: x(21) & Input point 23: x(22)
        curr_in = in + in_strides[21];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in10, bv_in11);
        // Input point 26: x(25) & Input point 27: x(26)
        curr_in = in + in_strides[25];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in12, bv_in13);

        bv_s1 = _mm_add_ps(bv_in0, bv_in12);
        bv_s2 = _mm_sub_ps(bv_in0, bv_in12);
        bv_s3 = _mm_add_ps(bv_in1, bv_in13);
        bv_s4 = _mm_sub_ps(bv_in1, bv_in13);
        bv_s5 = _mm_add_ps(bv_in2, bv_in10);
        bv_s6 = _mm_sub_ps(bv_in2, bv_in10);
        bv_s7 = _mm_add_ps(bv_in3, bv_in11);
        bv_s8 = _mm_sub_ps(bv_in3, bv_in11);
        bv_s9 = _mm_add_ps(bv_in4, bv_in8);
        bv_s10 = _mm_sub_ps(bv_in4, bv_in8);
        bv_s11 = _mm_add_ps(bv_in5, bv_in9);
        bv_s12 = _mm_sub_ps(bv_in5, bv_in9);

        bv_t1 = _mm_mul_ps(v128_CRTM_14_6, bv_s3);
        bv_t2 = _mm_mul_ps(v128_CRTM_14_2, bv_s11);
        bv_t3 = _mm_mul_ps(v128_CRTM_14_4, bv_s7);
        bv_t4 = _mm_mul_ps(v128_CRTM_14_5, bv_s2);
        bv_t5 = _mm_mul_ps(v128_CRTM_14_1, bv_s10);
        bv_t6 = _mm_mul_ps(v128_CRTM_14_3, bv_s6);

        bv_t7 = _mm_mul_ps(v128_CRTM_14_6, bv_s5);
        bv_t8 = _mm_mul_ps(v128_CRTM_14_2, bv_s1);
        bv_t9 = _mm_mul_ps(v128_CRTM_14_4, bv_s9);
        bv_t10 = _mm_mul_ps(v128_CRTM_14_5, bv_s8);
        bv_t11 = _mm_mul_ps(v128_CRTM_14_1, bv_s4);
        bv_t12 = _mm_mul_ps(v128_CRTM_14_3, bv_s12);

        bv_t13 = _mm_mul_ps(v128_CRTM_14_6, bv_s11);
        bv_t14 = _mm_mul_ps(v128_CRTM_14_2, bv_s7);
        bv_t15 = _mm_mul_ps(v128_CRTM_14_4, bv_s3);
        bv_t16 = _mm_mul_ps(v128_CRTM_14_5, bv_s10);
        bv_t17 = _mm_mul_ps(v128_CRTM_14_1, bv_s6);
        bv_t18 = _mm_mul_ps(v128_CRTM_14_3, bv_s2);

        bv_t19 = _mm_mul_ps(v128_CRTM_14_6, bv_s9);
        bv_t20 = _mm_mul_ps(v128_CRTM_14_2, bv_s5);
        bv_t21 = _mm_mul_ps(v128_CRTM_14_4, bv_s1);
        bv_t22 = _mm_mul_ps(v128_CRTM_14_5, bv_s12);
        bv_t23 = _mm_mul_ps(v128_CRTM_14_1, bv_s8);
        bv_t24 = _mm_mul_ps(v128_CRTM_14_3, bv_s4);

        bv_t25 = _mm_mul_ps(v128_CRTM_14_6, bv_s7);
        bv_t26 = _mm_mul_ps(v128_CRTM_14_2, bv_s3);
        bv_t27 = _mm_mul_ps(v128_CRTM_14_4, bv_s11);
        bv_t28 = _mm_mul_ps(v128_CRTM_14_5, bv_s6);
        bv_t29 = _mm_mul_ps(v128_CRTM_14_1, bv_s2);
        bv_t30 = _mm_mul_ps(v128_CRTM_14_3, bv_s10);

        bv_t31 = _mm_mul_ps(v128_CRTM_14_6, bv_s1);
        bv_t32 = _mm_mul_ps(v128_CRTM_14_2, bv_s9);
        bv_t33 = _mm_mul_ps(v128_CRTM_14_4, bv_s5);
        bv_t34 = _mm_mul_ps(v128_CRTM_14_5, bv_s4);
        bv_t35 = _mm_mul_ps(v128_CRTM_14_1, bv_s12);
        bv_t36 = _mm_mul_ps(v128_CRTM_14_3, bv_s8);

        bv_s13 = _mm_add_ps(bv_in6, bv_in6);
        bv_s14 = _mm_add_ps(bv_in7, bv_in7);

        bv_s15 = _mm_add_ps(bv_t1, bv_t2);
        bv_s16 = _mm_add_ps(bv_t3, bv_s14);
        bv_s17 = _mm_add_ps(bv_s15, bv_s16);
        bv_s18 = _mm_add_ps(bv_t4, bv_t5);
        bv_s19 = _mm_add_ps(bv_t6, bv_s18);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm_sub_ps(bv_s19, bv_s17);
        STR_128_S(curr_out, v_out_stride, v_out3);
        // Output pt 28: X(27)
        curr_out = out + out_strides[27];
        v_out27 = NEGATE_128_S(_mm_add_ps(bv_s17, bv_s19));
        STR_128_S(curr_out, v_out_stride, v_out27);

        bv_s20 = _mm_add_ps(bv_t10, bv_t11);
        bv_s21 = _mm_add_ps(bv_t12, bv_s20);
        bv_s22 = _mm_add_ps(bv_t7, bv_t8);
        bv_s23 = _mm_add_ps(bv_t9, bv_s13);
        bv_s24 = _mm_sub_ps(bv_s22, bv_s23);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm_sub_ps(bv_s24, bv_s21);
        STR_128_S(curr_out, v_out_stride, v_out5);
        // Output pt 26: X(25)
        curr_out = out + out_strides[25];
        v_out25 = NEGATE_128_S(_mm_add_ps(bv_s21, bv_s24));
        STR_128_S(curr_out, v_out_stride, v_out25);

        bv_s25 = _mm_sub_ps(bv_t13, bv_t14);
        bv_s26 = _mm_sub_ps(bv_s14, bv_t15);
        bv_s27 = _mm_add_ps(bv_s25, bv_s26);
        bv_s28 = _mm_add_ps(bv_t16, bv_t17);
        bv_s29 = _mm_sub_ps(bv_t18, bv_s28);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = _mm_add_ps(bv_s27, bv_s29);
        STR_128_S(curr_out, v_out_stride, v_out7);
        // Output pt 24: X(23)
        curr_out = out + out_strides[23];
        v_out23 = _mm_sub_ps(bv_s27, bv_s29);
        STR_128_S(curr_out, v_out_stride, v_out23);

        bv_s30 = _mm_sub_ps(bv_t22, bv_t23);
        bv_s31 = _mm_sub_ps(bv_s30, bv_t24);
        bv_s32 = _mm_add_ps(bv_t19, bv_t20);
        bv_s33 = _mm_add_ps(bv_t21, bv_s13);
        bv_s34 = _mm_sub_ps(bv_s33, bv_s32);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm_add_ps(bv_s31, bv_s34);
        STR_128_S(curr_out, v_out_stride, v_out9);
        // Output pt 22: X(21)
        curr_out = out + out_strides[21];
        v_out21 = _mm_sub_ps(bv_s31, bv_s34);
        STR_128_S(curr_out, v_out_stride, v_out21);

        bv_s35 = _mm_sub_ps(bv_t25, bv_t26);
        bv_s36 = _mm_sub_ps(bv_t27, bv_s14);
        bv_s37 = _mm_add_ps(bv_s35, bv_s36);
        bv_s38 = _mm_sub_ps(bv_t29, bv_t28);
        bv_s39 = _mm_add_ps(bv_t30, bv_s38);
        // Output pt 12: X(11)
        curr_out = out + out_strides[11];
        v_out11 = _mm_add_ps(bv_s37, bv_s39);
        STR_128_S(curr_out, v_out_stride, v_out11);
        // Output pt 20: X(19)
        curr_out = out + out_strides[19];
        v_out19 = _mm_sub_ps(bv_s37, bv_s39);
        STR_128_S(curr_out, v_out_stride, v_out19);

        bv_s40 = _mm_add_ps(bv_t34, bv_t35);
        bv_s41 = _mm_sub_ps(bv_t36, bv_s40);
        bv_s42 = _mm_add_ps(bv_t31, bv_t32);
        bv_s43 = _mm_add_ps(bv_t33, bv_s13);
        bv_s44 = _mm_sub_ps(bv_s42, bv_s43);
        // Output pt 14: X(13)
        curr_out = out + out_strides[13];
        v_out13 = _mm_add_ps(bv_s41, bv_s44);
        STR_128_S(curr_out, v_out_stride, v_out13);
        // Output pt 18: X(17)
        curr_out = out + out_strides[17];
        v_out17 = _mm_sub_ps(bv_s41, bv_s44);
        STR_128_S(curr_out, v_out_stride, v_out17);

        bv_s45 = _mm_add_ps(bv_s1, bv_s5);
        bv_s46 = _mm_add_ps(bv_in6, bv_s9);
        bv_s47 = _mm_add_ps(bv_s45, bv_s46);
        // Output pt 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm_mul_ps(bv_s47, v128_CRTM_14_7);
        STR_128_S(curr_out, v_out_stride, v_out1);

        bv_s48 = _mm_add_ps(bv_s3, bv_s11);
        bv_s49 = _mm_add_ps(bv_in7, bv_s7);
        bv_s50 = _mm_sub_ps(bv_s49, bv_s48);
        // Output pt 16: X(15)
        curr_out = out + out_strides[15];
        v_out15 = _mm_mul_ps(bv_s50, v128_CRTM_14_7);
        STR_128_S(curr_out, v_out_stride, v_out15);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & 2)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
               av_in8, av_in9, av_in10, av_in11, av_in12, av_in13;
        __m128 av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8, av_s9,
               av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16, av_s17,
               av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24, av_s25,
               av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32, av_s33,
               av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40, av_s41,
               av_s42, av_s43, av_s44, av_s45, av_s46, av_s47, av_s48;
        __m128 av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8, av_t9,
               av_t10, av_t11, av_t12, av_t13, av_t14, av_t15, av_t16, av_t17,
               av_t18, av_t19, av_t20, av_t21, av_t22, av_t23, av_t24, av_t25,
               av_t26, av_t27, av_t28, av_t29, av_t30, av_t31, av_t32, av_t33,
               av_t34, av_t35, av_t36, av_t37, av_t38;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21,
               v_out22, v_out23, v_out24, v_out25, v_out26, v_out27;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_14_1 = _mm512_castps512_ps128(v_CRTM_14_1);
        __m128 v128_CRTM_14_2 = _mm512_castps512_ps128(v_CRTM_14_2);
        __m128 v128_CRTM_14_3 = _mm512_castps512_ps128(v_CRTM_14_3);
        __m128 v128_CRTM_14_4 = _mm512_castps512_ps128(v_CRTM_14_4);
        __m128 v128_CRTM_14_5 = _mm512_castps512_ps128(v_CRTM_14_5);
        __m128 v128_CRTM_14_6 = _mm512_castps512_ps128(v_CRTM_14_6);
        __m128 v128_CRTM_14_7 = _mm512_castps512_ps128(v_CRTM_14_7);

        // Input point 1: x(0)
        LDHR_128_S(curr_in, v_in_stride, av_in0);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDHRI_2x128_S(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in + in_strides[7];
        LDHRI_2x128_S(curr_in, v_in_stride, av_in3, av_in4);
        // Input point 12: x(11) & Input point 13: x(12)
        curr_in = in + in_strides[11];
        LDHRI_2x128_S(curr_in, v_in_stride, av_in5, av_in6);
        // Input point 16: x(15) & Input point 17: x(16)
        curr_in = in + in_strides[15];
        LDHRI_2x128_S(curr_in, v_in_stride, av_in7, av_in8);
        // Input point 20: x(19) & Input point 21: x(20)
        curr_in = in + in_strides[19];
        LDHRI_2x128_S(curr_in, v_in_stride, av_in9, av_in10);
        // Input point 24: x(23) & Input point 25: x(24)
        curr_in = in + in_strides[23];
        LDHRI_2x128_S(curr_in, v_in_stride, av_in11, av_in12);
        // Input point 28: x(27)
        curr_in = in + in_strides[27];
        LDHR_128_S(curr_in, v_in_stride, av_in13);

        av_s1 = _mm_sub_ps(av_in0, av_in13);
        av_s2 = _mm_add_ps(av_in0, av_in13);
        av_s3 = _mm_sub_ps(av_in11, av_in1);
        av_s4 = _mm_add_ps(av_in1, av_in11);
        av_s5 = _mm_sub_ps(av_in2, av_in12);
        av_s6 = _mm_add_ps(av_in2, av_in12);
        av_s7 = _mm_sub_ps(av_in3, av_in9);
        av_s8 = _mm_add_ps(av_in3, av_in9);
        av_s9 = _mm_sub_ps(av_in4, av_in10);
        av_s10 = _mm_add_ps(av_in4, av_in10);
        av_s11 = _mm_sub_ps(av_in7, av_in5);
        av_s12 = _mm_add_ps(av_in5, av_in7);
        av_s13 = _mm_sub_ps(av_in6, av_in8);
        av_s14 = _mm_add_ps(av_in6, av_in8);

        av_s27 = _mm_add_ps(av_s12, av_s4);
        av_s28 = _mm_add_ps(av_s27, av_s8);
        av_t37 = _mm_mul_ps(v128_CRTM_14_7, av_s28);
        av_s29 = _mm_add_ps(av_s3, av_s7);
        av_s30 = _mm_add_ps(av_s29, av_s11);
        av_t38 = _mm_mul_ps(v128_CRTM_14_7, av_s30);
        // Output pt 1: X(0)
        v_out0 = _mm_add_ps(av_t37, av_s2);
        STHR_128_S(curr_out, v_out_stride, v_out0);
        // Output pt 15: X(14)
        curr_out = out + out_strides[14];
        v_out14 = _mm_add_ps(av_t38, av_s1);
        STHR_128_S(curr_out, v_out_stride, v_out14);

        av_t1 = _mm_mul_ps(v128_CRTM_14_1, av_s6);
        av_t2 = _mm_mul_ps(v128_CRTM_14_3, av_s10);
        av_t3 = _mm_mul_ps(v128_CRTM_14_5, av_s14);
        av_t4 = _mm_mul_ps(v128_CRTM_14_2, av_s3);
        av_t5 = _mm_mul_ps(v128_CRTM_14_4, av_s7);
        av_t6 = _mm_mul_ps(v128_CRTM_14_6, av_s11);

        av_s31 = _mm_sub_ps(av_t5, av_t6);
        av_s32 = _mm_sub_ps(av_s1, av_t4);
        av_s33 = _mm_add_ps(av_t1, av_t2);

        av_s15 = _mm_add_ps(av_s31, av_s32);
        av_s16 = _mm_add_ps(av_s33, av_t3);
        // Output point 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm_sub_ps(av_s15, av_s16);
        STHR_128_S(curr_out, v_out_stride, v_out2);
        // Output point 27: X(26)
        curr_out = out + out_strides[26];
        v_out26 = _mm_add_ps(av_s15, av_s16);
        STHR_128_S(curr_out, v_out_stride, v_out26);

        av_t7 = _mm_mul_ps(v128_CRTM_14_1, av_s13);
        av_t8 = _mm_mul_ps(v128_CRTM_14_3, av_s5);
        av_t9 = _mm_mul_ps(v128_CRTM_14_5, av_s9);

        av_t10 = _mm_mul_ps(v128_CRTM_14_2, av_s12);
        av_t11 = _mm_mul_ps(v128_CRTM_14_4, av_s4);
        av_t12 = _mm_mul_ps(v128_CRTM_14_6, av_s8);

        av_s34 = _mm_sub_ps(av_s2, av_t10);
        av_s35 = _mm_sub_ps(av_t11, av_t12);
        av_s36 = _mm_add_ps(av_t7, av_t8);

        av_s17 = av_s34 + av_s35;
        av_s18 = _mm_add_ps(av_s36, av_t9);
        // Output point 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm_sub_ps(av_s17, av_s18);
        STHR_128_S(curr_out, v_out_stride, v_out4);
        // Output point 25: X(24)
        curr_out = out + out_strides[24];
        v_out24 = _mm_add_ps(av_s17, av_s18);
        STHR_128_S(curr_out, v_out_stride, v_out24);

        av_t13 = _mm_mul_ps(v128_CRTM_14_1, av_s10);
        av_t14 = _mm_mul_ps(v128_CRTM_14_3, av_s14);
        av_t15 = _mm_mul_ps(v128_CRTM_14_5, av_s6);
        av_t16 = _mm_mul_ps(v128_CRTM_14_2, av_s7);
        av_t17 = _mm_mul_ps(v128_CRTM_14_4, av_s11);
        av_t18 = _mm_mul_ps(v128_CRTM_14_6, av_s3);

        av_s37 = _mm_sub_ps(av_s1, av_t16);
        av_s38 = _mm_sub_ps(av_t17, av_t18);
        av_s39 = _mm_sub_ps(av_t14, av_t15);
        av_s19 = _mm_add_ps(av_s37, av_s38);
        av_s20 = _mm_sub_ps(av_s39, av_t13);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm_add_ps(av_s19, av_s20);
        STHR_128_S(curr_out, v_out_stride, v_out6);
        // Output point 23: X(22)
        curr_out = out + out_strides[22];
        v_out22 = _mm_sub_ps(av_s19, av_s20);
        STHR_128_S(curr_out, v_out_stride, v_out22);

        av_t19 = _mm_mul_ps(v128_CRTM_14_1, av_s9);
        av_t20 = _mm_mul_ps(v128_CRTM_14_3, av_s13);
        av_t21 = _mm_mul_ps(v128_CRTM_14_5, av_s5);
        av_t22 = _mm_mul_ps(v128_CRTM_14_2, av_s8);
        av_t23 = _mm_mul_ps(v128_CRTM_14_4, av_s12);
        av_t24 = _mm_mul_ps(v128_CRTM_14_6, av_s4);

        av_s40 = _mm_sub_ps(av_s2, av_t22);
        av_s41 = _mm_sub_ps(av_t23, av_t24);
        av_s42 = _mm_add_ps(av_t19, av_t20);
        av_s21 = _mm_add_ps(av_s40, av_s41);
        av_s22 = _mm_sub_ps(av_s42, av_t21);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm_add_ps(av_s21, av_s22);
        STHR_128_S(curr_out, v_out_stride, v_out8);
        // Output point 21: X(20)
        curr_out = out + out_strides[20];
        v_out20 = _mm_sub_ps(av_s21, av_s22);
        STHR_128_S(curr_out, v_out_stride, v_out20);

        av_t25 = _mm_mul_ps(v128_CRTM_14_1, av_s14);
        av_t26 = _mm_mul_ps(v128_CRTM_14_3, av_s6);
        av_t27 = _mm_mul_ps(v128_CRTM_14_5, av_s10);
        av_t28 = _mm_mul_ps(v128_CRTM_14_2, av_s11);
        av_t29 = _mm_mul_ps(v128_CRTM_14_4, av_s3);
        av_t30 = _mm_mul_ps(v128_CRTM_14_6, av_s7);

        av_s43 = _mm_sub_ps(av_s1, av_t28);
        av_s44 = _mm_sub_ps(av_t29, av_t30);
        av_s45 = _mm_sub_ps(av_t27, av_t25);
        av_s23 = _mm_add_ps(av_s43, av_s44);
        av_s24 = _mm_sub_ps(av_s45, av_t26);
        // Output pt 11: X(10)
        curr_out = out + out_strides[10];
        v_out10 = _mm_add_ps(av_s23, av_s24);
        STHR_128_S(curr_out, v_out_stride, v_out10);
        // Output pt 19: X(18)
        curr_out = out + out_strides[18];
        v_out18 = _mm_sub_ps(av_s23, av_s24);
        STHR_128_S(curr_out, v_out_stride, v_out18);

        av_t31 = _mm_mul_ps(v128_CRTM_14_1, av_s5);
        av_t32 = _mm_mul_ps(v128_CRTM_14_3, av_s9);
        av_t33 = _mm_mul_ps(v128_CRTM_14_5, av_s13);
        av_t34 = _mm_mul_ps(v128_CRTM_14_2, av_s4);
        av_t35 = _mm_mul_ps(v128_CRTM_14_4, av_s8);
        av_t36 = _mm_mul_ps(v128_CRTM_14_6, av_s12);

        av_s46 = _mm_sub_ps(av_s2, av_t34);
        av_s47 = _mm_sub_ps(av_t35, av_t36);
        av_s48 = _mm_sub_ps(av_t32, av_t31);
        av_s25 = _mm_add_ps(av_s46, av_s47);
        av_s26 = _mm_sub_ps(av_s48, av_t33);
        // Output pt 13: X(12)
        curr_out = out + out_strides[12];
        v_out12 = _mm_add_ps(av_s25, av_s26);
        STHR_128_S(curr_out, v_out_stride, v_out12);
        // Output pt 17: X(16)
        curr_out = out + out_strides[16];
        v_out16 = _mm_sub_ps(av_s25, av_s26);
        STHR_128_S(curr_out, v_out_stride, v_out16);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8, bv_in9, bv_in10, bv_in11, bv_in12, bv_in13;
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
               bv_t26, bv_t27, bv_t28, bv_t29, bv_t30, bv_t31, bv_t32, bv_t33,
               bv_t34, bv_t35, bv_t36;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in4, bv_in5);
        // Input point 14: x(13) & Input point 15: x(14)
        curr_in = in + in_strides[13];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in6, bv_in7);
        // Input point 18: x(17) & Input point 19: x(18)
        curr_in = in + in_strides[17];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in8, bv_in9);
        // Input point 22: x(21) & Input point 23: x(22)
        curr_in = in + in_strides[21];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in10, bv_in11);
        // Input point 26: x(25) & Input point 27: x(26)
        curr_in = in + in_strides[25];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in12, bv_in13);

        bv_s1 = _mm_add_ps(bv_in0, bv_in12);
        bv_s2 = _mm_sub_ps(bv_in0, bv_in12);
        bv_s3 = _mm_add_ps(bv_in1, bv_in13);
        bv_s4 = _mm_sub_ps(bv_in1, bv_in13);
        bv_s5 = _mm_add_ps(bv_in2, bv_in10);
        bv_s6 = _mm_sub_ps(bv_in2, bv_in10);
        bv_s7 = _mm_add_ps(bv_in3, bv_in11);
        bv_s8 = _mm_sub_ps(bv_in3, bv_in11);
        bv_s9 = _mm_add_ps(bv_in4, bv_in8);
        bv_s10 = _mm_sub_ps(bv_in4, bv_in8);
        bv_s11 = _mm_add_ps(bv_in5, bv_in9);
        bv_s12 = _mm_sub_ps(bv_in5, bv_in9);

        bv_t1 = _mm_mul_ps(v128_CRTM_14_6, bv_s3);
        bv_t2 = _mm_mul_ps(v128_CRTM_14_2, bv_s11);
        bv_t3 = _mm_mul_ps(v128_CRTM_14_4, bv_s7);
        bv_t4 = _mm_mul_ps(v128_CRTM_14_5, bv_s2);
        bv_t5 = _mm_mul_ps(v128_CRTM_14_1, bv_s10);
        bv_t6 = _mm_mul_ps(v128_CRTM_14_3, bv_s6);

        bv_t7 = _mm_mul_ps(v128_CRTM_14_6, bv_s5);
        bv_t8 = _mm_mul_ps(v128_CRTM_14_2, bv_s1);
        bv_t9 = _mm_mul_ps(v128_CRTM_14_4, bv_s9);
        bv_t10 = _mm_mul_ps(v128_CRTM_14_5, bv_s8);
        bv_t11 = _mm_mul_ps(v128_CRTM_14_1, bv_s4);
        bv_t12 = _mm_mul_ps(v128_CRTM_14_3, bv_s12);

        bv_t13 = _mm_mul_ps(v128_CRTM_14_6, bv_s11);
        bv_t14 = _mm_mul_ps(v128_CRTM_14_2, bv_s7);
        bv_t15 = _mm_mul_ps(v128_CRTM_14_4, bv_s3);
        bv_t16 = _mm_mul_ps(v128_CRTM_14_5, bv_s10);
        bv_t17 = _mm_mul_ps(v128_CRTM_14_1, bv_s6);
        bv_t18 = _mm_mul_ps(v128_CRTM_14_3, bv_s2);

        bv_t19 = _mm_mul_ps(v128_CRTM_14_6, bv_s9);
        bv_t20 = _mm_mul_ps(v128_CRTM_14_2, bv_s5);
        bv_t21 = _mm_mul_ps(v128_CRTM_14_4, bv_s1);
        bv_t22 = _mm_mul_ps(v128_CRTM_14_5, bv_s12);
        bv_t23 = _mm_mul_ps(v128_CRTM_14_1, bv_s8);
        bv_t24 = _mm_mul_ps(v128_CRTM_14_3, bv_s4);

        bv_t25 = _mm_mul_ps(v128_CRTM_14_6, bv_s7);
        bv_t26 = _mm_mul_ps(v128_CRTM_14_2, bv_s3);
        bv_t27 = _mm_mul_ps(v128_CRTM_14_4, bv_s11);
        bv_t28 = _mm_mul_ps(v128_CRTM_14_5, bv_s6);
        bv_t29 = _mm_mul_ps(v128_CRTM_14_1, bv_s2);
        bv_t30 = _mm_mul_ps(v128_CRTM_14_3, bv_s10);

        bv_t31 = _mm_mul_ps(v128_CRTM_14_6, bv_s1);
        bv_t32 = _mm_mul_ps(v128_CRTM_14_2, bv_s9);
        bv_t33 = _mm_mul_ps(v128_CRTM_14_4, bv_s5);
        bv_t34 = _mm_mul_ps(v128_CRTM_14_5, bv_s4);
        bv_t35 = _mm_mul_ps(v128_CRTM_14_1, bv_s12);
        bv_t36 = _mm_mul_ps(v128_CRTM_14_3, bv_s8);

        bv_s13 = _mm_add_ps(bv_in6, bv_in6);
        bv_s14 = _mm_add_ps(bv_in7, bv_in7);

        bv_s15 = _mm_add_ps(bv_t1, bv_t2);
        bv_s16 = _mm_add_ps(bv_t3, bv_s14);
        bv_s17 = _mm_add_ps(bv_s15, bv_s16);
        bv_s18 = _mm_add_ps(bv_t4, bv_t5);
        bv_s19 = _mm_add_ps(bv_t6, bv_s18);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm_sub_ps(bv_s19, bv_s17);
        STHR_128_S(curr_out, v_out_stride, v_out3);
        // Output pt 28: X(27)
        curr_out = out + out_strides[27];
        v_out27 = NEGATE_128_S(_mm_add_ps(bv_s17, bv_s19));
        STHR_128_S(curr_out, v_out_stride, v_out27);

        bv_s20 = _mm_add_ps(bv_t10, bv_t11);
        bv_s21 = _mm_add_ps(bv_t12, bv_s20);
        bv_s22 = _mm_add_ps(bv_t7, bv_t8);
        bv_s23 = _mm_add_ps(bv_t9, bv_s13);
        bv_s24 = _mm_sub_ps(bv_s22, bv_s23);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm_sub_ps(bv_s24, bv_s21);
        STHR_128_S(curr_out, v_out_stride, v_out5);
        // Output pt 26: X(25)
        curr_out = out + out_strides[25];
        v_out25 = NEGATE_128_S(_mm_add_ps(bv_s21, bv_s24));
        STHR_128_S(curr_out, v_out_stride, v_out25);

        bv_s25 = _mm_sub_ps(bv_t13, bv_t14);
        bv_s26 = _mm_sub_ps(bv_s14, bv_t15);
        bv_s27 = _mm_add_ps(bv_s25, bv_s26);
        bv_s28 = _mm_add_ps(bv_t16, bv_t17);
        bv_s29 = _mm_sub_ps(bv_t18, bv_s28);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = _mm_add_ps(bv_s27, bv_s29);
        STHR_128_S(curr_out, v_out_stride, v_out7);
        // Output pt 24: X(23)
        curr_out = out + out_strides[23];
        v_out23 = _mm_sub_ps(bv_s27, bv_s29);
        STHR_128_S(curr_out, v_out_stride, v_out23);

        bv_s30 = _mm_sub_ps(bv_t22, bv_t23);
        bv_s31 = _mm_sub_ps(bv_s30, bv_t24);
        bv_s32 = _mm_add_ps(bv_t19, bv_t20);
        bv_s33 = _mm_add_ps(bv_t21, bv_s13);
        bv_s34 = _mm_sub_ps(bv_s33, bv_s32);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm_add_ps(bv_s31, bv_s34);
        STHR_128_S(curr_out, v_out_stride, v_out9);
        // Output pt 22: X(21)
        curr_out = out + out_strides[21];
        v_out21 = _mm_sub_ps(bv_s31, bv_s34);
        STHR_128_S(curr_out, v_out_stride, v_out21);

        bv_s35 = _mm_sub_ps(bv_t25, bv_t26);
        bv_s36 = _mm_sub_ps(bv_t27, bv_s14);
        bv_s37 = _mm_add_ps(bv_s35, bv_s36);
        bv_s38 = _mm_sub_ps(bv_t29, bv_t28);
        bv_s39 = _mm_add_ps(bv_t30, bv_s38);
        // Output pt 12: X(11)
        curr_out = out + out_strides[11];
        v_out11 = _mm_add_ps(bv_s37, bv_s39);
        STHR_128_S(curr_out, v_out_stride, v_out11);
        // Output pt 20: X(19)
        curr_out = out + out_strides[19];
        v_out19 = _mm_sub_ps(bv_s37, bv_s39);
        STHR_128_S(curr_out, v_out_stride, v_out19);

        bv_s40 = _mm_add_ps(bv_t34, bv_t35);
        bv_s41 = _mm_sub_ps(bv_t36, bv_s40);
        bv_s42 = _mm_add_ps(bv_t31, bv_t32);
        bv_s43 = _mm_add_ps(bv_t33, bv_s13);
        bv_s44 = _mm_sub_ps(bv_s42, bv_s43);
        // Output pt 14: X(13)
        curr_out = out + out_strides[13];
        v_out13 = _mm_add_ps(bv_s41, bv_s44);
        STHR_128_S(curr_out, v_out_stride, v_out13);
        // Output pt 18: X(17)
        curr_out = out + out_strides[17];
        v_out17 = _mm_sub_ps(bv_s41, bv_s44);
        STHR_128_S(curr_out, v_out_stride, v_out17);

        bv_s45 = _mm_add_ps(bv_s1, bv_s5);
        bv_s46 = _mm_add_ps(bv_in6, bv_s9);
        bv_s47 = _mm_add_ps(bv_s45, bv_s46);
        // Output pt 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm_mul_ps(bv_s47, v128_CRTM_14_7);
        STHR_128_S(curr_out, v_out_stride, v_out1);

        bv_s48 = _mm_add_ps(bv_s3, bv_s11);
        bv_s49 = _mm_add_ps(bv_in7, bv_s7);
        bv_s50 = _mm_sub_ps(bv_s49, bv_s48);
        // Output pt 16: X(15)
        curr_out = out + out_strides[15];
        v_out15 = _mm_mul_ps(bv_s50, v128_CRTM_14_7);
        STHR_128_S(curr_out, v_out_stride, v_out15);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        /* Standard DFT */
        FFTZ_FLOAT a_in0, a_in1, a_in2, a_in3, a_in4, a_in5, a_in6, a_in7,
            a_in8, a_in9, a_in10, a_in11, a_in12, a_in13;
        FFTZ_FLOAT a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_s9,
            a_s10, a_s11, a_s12, a_s13, a_s14, a_s15, a_s16, a_s17, a_s18,
            a_s19, a_s20, a_s21, a_s22, a_s23, a_s24, a_s25, a_s26, a_s27,
            a_s28, a_s29, a_s30, a_s31, a_s32, a_s33, a_s34, a_s35, a_s36,
            a_s37, a_s38, a_s39, a_s40, a_s41, a_s42, a_s43, a_s44, a_s45,
            a_s46, a_s47;
        FFTZ_FLOAT a_t0, a_t1, a_t2, a_t3, a_t4, a_t5, a_t6, a_t7, a_t8, a_t9,
            a_t10, a_t11, a_t12, a_t13, a_t14, a_t15, a_t16, a_t17, a_t18,
            a_t19, a_t20, a_t21, a_t22, a_t23, a_t24, a_t25, a_t26, a_t27,
            a_t28, a_t29, a_t30, a_t31, a_t32, a_t33, a_t34, a_t35, a_t36,
            a_t37;

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

        a_s0 = a_in0 - a_in13;
        a_s1 = a_in0 + a_in13;
        a_s2 = a_in11 - a_in1;
        a_s3 = a_in1 + a_in11;
        a_s4 = a_in2 - a_in12;
        a_s5 = a_in2 + a_in12;
        a_s6 = a_in3 - a_in9;
        a_s7 = a_in3 + a_in9;
        a_s8 = a_in4 - a_in10;
        a_s9 = a_in4 + a_in10;
        a_s10 = a_in7 - a_in5;
        a_s11 = a_in5 + a_in7;
        a_s12 = a_in6 - a_in8;
        a_s13 = a_in6 + a_in8;

        a_s26 = a_s11 + a_s3;
        a_s27 = a_s26 + a_s7;
        a_t0 = CRTM_14_7 * a_s27;
        a_s28 = a_s2 + a_s6;
        a_s29 = a_s28 + a_s10;
        a_t1 = CRTM_14_7 * a_s29;
        // Output point 1: X(0)
        *out = a_t0 + a_s1;
        // Output point 15: X(14)
        out[out_strides[14]] = a_t1 + a_s0;

        a_t2 = CRTM_14_1 * a_s5;
        a_t3 = CRTM_14_3 * a_s9;
        a_t4 = CRTM_14_5 * a_s13;
        a_t5 = CRTM_14_2 * a_s2;
        a_t6 = CRTM_14_4 * a_s6;
        a_t7 = CRTM_14_6 * a_s10;

        a_s30 = a_t6 - a_t7;
        a_s31 = a_s0 - a_t5;
        a_s32 = a_t2 + a_t3;
        a_s14 = a_s30 + a_s31;
        a_s15 = a_s32 + a_t4;
        // Output point 3: X(2)
        out[out_strides[2]] = a_s14 - a_s15;
        // Output point 27: X(26)
        out[out_strides[26]] = a_s14 + a_s15;

        a_t8 = CRTM_14_1 * a_s12;
        a_t9 = CRTM_14_3 * a_s4;
        a_t10 = CRTM_14_5 * a_s8;
        a_t11 = CRTM_14_2 * a_s11;
        a_t12 = CRTM_14_4 * a_s3;
        a_t13 = CRTM_14_6 * a_s7;

        a_s33 = a_s1 - a_t11;
        a_s34 = a_t12 - a_t13;
        a_s35 = a_t8 + a_t9;
        a_s16 = a_s33 + a_s34;
        a_s17 = a_s35 + a_t10;
        // Output point 5: X(4)
        out[out_strides[4]] = a_s16 - a_s17;
        // Output point 25: X(24)
        out[out_strides[24]] = a_s16 + a_s17;

        a_t14 = CRTM_14_1 * a_s9;
        a_t15 = CRTM_14_3 * a_s13;
        a_t16 = CRTM_14_5 * a_s5;
        a_t17 = CRTM_14_2 * a_s6;
        a_t18 = CRTM_14_4 * a_s10;
        a_t19 = CRTM_14_6 * a_s2;

        a_s36 = a_s0 - a_t17;
        a_s37 = a_t18 - a_t19;
        a_s38 = a_t15 - a_t16;
        a_s18 = a_s36 + a_s37;
        a_s19 = a_s38 - a_t14;
        // Output point 7: X(6)
        out[out_strides[6]] = a_s18 + a_s19;
        // Output point 23: X(22)
        out[out_strides[22]] = a_s18 - a_s19;

        a_t20 = CRTM_14_1 * a_s8;
        a_t21 = CRTM_14_3 * a_s12;
        a_t22 = CRTM_14_5 * a_s4;
        a_t23 = CRTM_14_2 * a_s7;
        a_t24 = CRTM_14_4 * a_s11;
        a_t25 = CRTM_14_6 * a_s3;

        a_s39 = a_s1 - a_t23;
        a_s40 = a_t24 - a_t25;
        a_s41 = a_t20 + a_t21;
        a_s20 = a_s39 + a_s40;
        a_s21 = a_s41 - a_t22;
        // Output point 9: X(8)
        out[out_strides[8]] = a_s20 + a_s21;
        // Output point 21: X(20)
        out[out_strides[20]] = a_s20 - a_s21;

        a_t26 = CRTM_14_1 * a_s13;
        a_t27 = CRTM_14_3 * a_s5;
        a_t28 = CRTM_14_5 * a_s9;
        a_t29 = CRTM_14_2 * a_s10;
        a_t30 = CRTM_14_4 * a_s2;
        a_t31 = CRTM_14_6 * a_s6;

        a_s42 = a_s0 - a_t29;
        a_s43 = a_t30 - a_t31;
        a_s44 = a_t28 - a_t26;
        a_s22 = a_s42 + a_s43;
        a_s23 = a_s44 - a_t27;
        // Output point 11: X(10)
        out[out_strides[10]] = a_s22 + a_s23;
        // Output point 19: X(18)
        out[out_strides[18]] = a_s22 - a_s23;

        a_t32 = CRTM_14_1 * a_s4;
        a_t33 = CRTM_14_3 * a_s8;
        a_t34 = CRTM_14_5 * a_s12;
        a_t35 = CRTM_14_2 * a_s3;
        a_t36 = CRTM_14_4 * a_s7;
        a_t37 = CRTM_14_6 * a_s11;

        a_s45 = a_s1 - a_t35;
        a_s46 = a_t36 - a_t37;
        a_s47 = a_t33 - a_t32;
        a_s24 = a_s45 + a_s46;
        a_s25 = a_s47 - a_t34;
        // Output point 13: X(12)
        out[out_strides[12]] = a_s24 + a_s25;
        // Output point 17: X(16)
        out[out_strides[16]] = a_s24 - a_s25;

        /* Shifted DFT */
        FFTZ_FLOAT b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7,
            b_in8, b_in9, b_in10, b_in11, b_in12, b_in13;
        FFTZ_FLOAT b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9,
            b_s10, b_s11, b_s12, b_s13, b_s14, b_s15, b_s16, b_s17, b_s18,
            b_s19, b_s20, b_s21, b_s22, b_s23, b_s24, b_s25, b_s26, b_s27,
            b_s28, b_s29, b_s30, b_s31, b_s32, b_s33, b_s34, b_s35, b_s36,
            b_s37, b_s38, b_s39, b_s40, b_s41, b_s42, b_s43, b_s44, b_s45,
            b_s46, b_s47, b_s48, b_s49;
        FFTZ_FLOAT b_t0, b_t1, b_t2, b_t3, b_t4, b_t5, b_t6, b_t7, b_t8, b_t9,
            b_t10, b_t11, b_t12, b_t13, b_t14, b_t15, b_t16, b_t17, b_t18,
            b_t19, b_t20, b_t21, b_t22, b_t23, b_t24, b_t25, b_t26, b_t27,
            b_t28, b_t29, b_t30, b_t31, b_t32, b_t33, b_t34, b_t35;

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

        b_s0 = b_in0 + b_in12;
        b_s1 = b_in0 - b_in12;
        b_s2 = b_in1 + b_in13;
        b_s3 = b_in1 - b_in13;
        b_s4 = b_in2 + b_in10;
        b_s5 = b_in2 - b_in10;
        b_s6 = b_in3 + b_in11;
        b_s7 = b_in3 - b_in11;
        b_s8 = b_in4 + b_in8;
        b_s9 = b_in4 - b_in8;
        b_s10 = b_in5 + b_in9;
        b_s11 = b_in5 - b_in9;

        b_t0 = CRTM_14_6 * b_s2;
        b_t1 = CRTM_14_2 * b_s10;
        b_t2 = CRTM_14_4 * b_s6;
        b_t3 = CRTM_14_5 * b_s1;
        b_t4 = CRTM_14_1 * b_s9;
        b_t5 = CRTM_14_3 * b_s5;

        b_t6 = CRTM_14_6 * b_s4;
        b_t7 = CRTM_14_2 * b_s0;
        b_t8 = CRTM_14_4 * b_s8;
        b_t9 = CRTM_14_5 * b_s7;
        b_t10 = CRTM_14_1 * b_s3;
        b_t11 = CRTM_14_3 * b_s11;

        b_t12 = CRTM_14_6 * b_s10;
        b_t13 = CRTM_14_2 * b_s6;
        b_t14 = CRTM_14_4 * b_s2;
        b_t15 = CRTM_14_5 * b_s9;
        b_t16 = CRTM_14_1 * b_s5;
        b_t17 = CRTM_14_3 * b_s1;

        b_t18 = CRTM_14_6 * b_s8;
        b_t19 = CRTM_14_2 * b_s4;
        b_t20 = CRTM_14_4 * b_s0;
        b_t21 = CRTM_14_5 * b_s11;
        b_t22 = CRTM_14_1 * b_s7;
        b_t23 = CRTM_14_3 * b_s3;

        b_t24 = CRTM_14_6 * b_s6;
        b_t25 = CRTM_14_2 * b_s2;
        b_t26 = CRTM_14_4 * b_s10;
        b_t27 = CRTM_14_5 * b_s5;
        b_t28 = CRTM_14_1 * b_s1;
        b_t29 = CRTM_14_3 * b_s9;

        b_t30 = CRTM_14_6 * b_s0;
        b_t31 = CRTM_14_2 * b_s8;
        b_t32 = CRTM_14_4 * b_s4;
        b_t33 = CRTM_14_5 * b_s3;
        b_t34 = CRTM_14_1 * b_s11;
        b_t35 = CRTM_14_3 * b_s7;

        b_s12 = b_in6 + b_in6;
        b_s13 = b_in7 + b_in7;

        b_s14 = b_t0 + b_t1;
        b_s15 = b_t2 + b_s13;
        b_s16 = b_s14 + b_s15;
        b_s17 = b_t3 + b_t4;
        b_s18 = b_t5 + b_s17;
        // Output point 4: X(3)
        out[out_strides[3]] = b_s18 - b_s16;
        // Output point 28: X(27)
        out[out_strides[27]] = -(b_s16 + b_s18);

        b_s19 = b_t9 + b_t10;
        b_s20 = b_t11 + b_s19;
        b_s21 = b_t6 + b_t7;
        b_s22 = b_t8 + b_s12;
        b_s23 = b_s21 - b_s22;
        // Output point 6: X(5)
        out[out_strides[5]] = b_s23 - b_s20;
        // Output point 26: X(25)
        out[out_strides[25]] = -(b_s20 + b_s23);

        b_s24 = b_t12 - b_t13;
        b_s25 = b_s13 - b_t14;
        b_s26 = b_s24 + b_s25;
        b_s27 = b_t15 + b_t16;
        b_s28 = b_t17 - b_s27;
        // Output point 8: X(7)
        out[out_strides[7]] = b_s26 + b_s28;
        // Output point 24: X(23)
        out[out_strides[23]] = b_s26 - b_s28;

        b_s29 = b_t21 - b_t22;
        b_s30 = b_s29 - b_t23;
        b_s31 = b_t18 + b_t19;
        b_s32 = b_t20 + b_s12;
        b_s33 = b_s32 - b_s31;
        // Output point 10: X(9)
        out[out_strides[9]] = b_s30 + b_s33;
        // Output point 22: X(21)
        out[out_strides[21]] = b_s30 - b_s33;

        b_s34 = b_t24 - b_t25;
        b_s35 = b_t26 - b_s13;
        b_s36 = b_s34 + b_s35;
        b_s37 = b_t28 - b_t27;
        b_s38 = b_t29 + b_s37;
        // Output point 12: X(11)
        out[out_strides[11]] = b_s36 + b_s38;
        // Output point 20: X(19)
        out[out_strides[19]] = b_s36 - b_s38;

        b_s39 = b_t33 + b_t34;
        b_s40 = b_t35 - b_s39;
        b_s41 = b_t30 + b_t31;
        b_s42 = b_t32 + b_s12;
        b_s43 = b_s41 - b_s42;
        // Output point 14: X(13)
        out[out_strides[13]] = b_s40 + b_s43;
        // Output point 18: X(17)
        out[out_strides[17]] = b_s40 - b_s43;

        b_s44 = b_s0 + b_s4;
        b_s45 = b_in6 + b_s8;
        b_s46 = b_s44 + b_s45;
        // Output point 2: X(1)
        out[out_strides[1]] = b_s46 * CRTM_14_7;

        b_s47 = b_s2 + b_s10;
        b_s48 = b_in7 + b_s6;
        b_s49 = b_s48 - b_s47;
        // Output point 16: X(15)
        out[out_strides[15]] = b_s49 * CRTM_14_7;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft14avx512_fp64_fwd(FFTZ_VOID *in_real,
                                             FFTZ_VOID *in_imag,
                                             FFTZ_VOID *out_real,
                                             FFTZ_VOID *out_imag, FFTZ_INTP n,
                                             aoclfftz_strides_t *strides,
                                             FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_14_1 =
        0.900968867902419126236102319507445051165919162;
    const FFTZ_DOUBLE CRTM_14_2 =
        0.433883739117558120475768332848358754609990728;
    const FFTZ_DOUBLE CRTM_14_3 =
        0.623489801858733530525004884004239810632274731;
    const FFTZ_DOUBLE CRTM_14_4 =
        0.781831482468029808708444526674057750232334519;
    const FFTZ_DOUBLE CRTM_14_5 =
        0.222520933956314404288902564496794759466355569;
    const FFTZ_DOUBLE CRTM_14_6 =
        0.974927912181823607018131682993931217232785801;

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
    FFTZ_INTP N = n / NUM_SETS_REAL_256_S;
    FFTZ_INTP remaining_sets = n % NUM_SETS_REAL_256_S;

    __m512d v_CRTM_14_1 = _mm512_set1_pd(CRTM_14_1);
    __m512d v_CRTM_14_2 = _mm512_set1_pd(CRTM_14_2);
    __m512d v_CRTM_14_3 = _mm512_set1_pd(CRTM_14_3);
    __m512d v_CRTM_14_4 = _mm512_set1_pd(CRTM_14_4);
    __m512d v_CRTM_14_5 = _mm512_set1_pd(CRTM_14_5);
    __m512d v_CRTM_14_6 = _mm512_set1_pd(CRTM_14_6);

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m512d av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
                av_in8, av_in9, av_in10, av_in11, av_in12, av_in13;
        __m512d av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8, av_s9,
                av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16, av_s17,
                av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24, av_s25,
                av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32, av_s33,
                av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40, av_s41,
                av_s42, av_s43, av_s44, av_s45, av_s46, av_s47, av_s48;
        __m512d av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8, av_t9,
                av_t10, av_t11, av_t12, av_t13, av_t14, av_t15, av_t16, av_t17,
                av_t18, av_t19, av_t20, av_t21, av_t22, av_t23, av_t24, av_t25,
                av_t26, av_t27, av_t28, av_t29, av_t30, av_t31, av_t32, av_t33,
                av_t34, av_t35, av_t36;
        __m512d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
                v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21,
                v_out22, v_out23, v_out24, v_out25, v_out26, v_out27;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_512_D(curr_in, v_in_stride, av_in0);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_512_D(curr_in, v_in_stride, av_in1);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_512_D(curr_in, v_in_stride, av_in2);
        // Input point 7: x(6)
        curr_in = in + in_strides[6];
        LDR_512_D(curr_in, v_in_stride, av_in3);
        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDR_512_D(curr_in, v_in_stride, av_in4);
        // Input point 11: x(10)
        curr_in = in + in_strides[10];
        LDR_512_D(curr_in, v_in_stride, av_in5);
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDR_512_D(curr_in, v_in_stride, av_in6);
        // Input point 15: x(14)
        curr_in = in + in_strides[14];
        LDR_512_D(curr_in, v_in_stride, av_in7);
        // Input point 17: x(16)
        curr_in = in + in_strides[16];
        LDR_512_D(curr_in, v_in_stride, av_in8);
        // Input point 19: x(18)
        curr_in = in + in_strides[18];
        LDR_512_D(curr_in, v_in_stride, av_in9);
        // Input point 21: x(20)
        curr_in = in + in_strides[20];
        LDR_512_D(curr_in, v_in_stride, av_in10);
        // Input point 23: x(22)
        curr_in = in + in_strides[22];
        LDR_512_D(curr_in, v_in_stride, av_in11);
        // Input point 25: x(24)
        curr_in = in + in_strides[24];
        LDR_512_D(curr_in, v_in_stride, av_in12);
        // Input point 27: x(26)
        curr_in = in + in_strides[26];
        LDR_512_D(curr_in, v_in_stride, av_in13);

        av_s1 = _mm512_sub_pd(av_in0, av_in7);
        av_s2 = _mm512_add_pd(av_in0, av_in7);
        av_s3 = _mm512_sub_pd(av_in13, av_in1);
        av_s4 = _mm512_add_pd(av_in13, av_in1);
        av_s5 = _mm512_sub_pd(av_in12, av_in2);
        av_s6 = _mm512_add_pd(av_in12, av_in2);
        av_s7 = _mm512_sub_pd(av_in11, av_in3);
        av_s8 = _mm512_add_pd(av_in11, av_in3);
        av_s9 = _mm512_sub_pd(av_in10, av_in4);
        av_s10 = _mm512_add_pd(av_in10, av_in4);
        av_s11 = _mm512_sub_pd(av_in9, av_in5);
        av_s12 = _mm512_add_pd(av_in9, av_in5);
        av_s13 = _mm512_sub_pd(av_in8, av_in6);
        av_s14 = _mm512_add_pd(av_in8, av_in6);

        av_s15 = _mm512_add_pd(av_s4, av_s14);
        av_s16 = _mm512_add_pd(av_s6, av_s12);
        av_s17 = _mm512_add_pd(av_s8, av_s10);

        av_s18 = _mm512_sub_pd(av_s14, av_s4);
        av_s19 = _mm512_sub_pd(av_s6, av_s12);
        av_s20 = _mm512_sub_pd(av_s10, av_s8);
        av_s27 = _mm512_add_pd(av_s2, av_s15);
        av_s28 = _mm512_add_pd(av_s16, av_s17);
        av_s29 = _mm512_add_pd(av_s1, av_s18);
        av_s30 = _mm512_add_pd(av_s19, av_s20);
        // Output pt 1: X(0)
        v_out0 = _mm512_add_pd(av_s27, av_s28);
        STR_512_D(curr_out, v_out_stride, v_out0);
        // Output pt 28: X(27)
        v_out27 = _mm512_add_pd(av_s29, av_s30);
        curr_out = out + out_strides[27];
        STR_512_D(curr_out, v_out_stride, v_out27);

        av_t1 = _mm512_mul_pd(v_CRTM_14_1, av_s18);
        av_t2 = _mm512_mul_pd(v_CRTM_14_3, av_s19);
        av_t3 = _mm512_mul_pd(v_CRTM_14_5, av_s20);
        av_s31 = _mm512_sub_pd(av_s1, av_t1);
        av_s32 = _mm512_sub_pd(av_t2, av_t3);
        // Output point 4: X(3)
        v_out3 = _mm512_add_pd(av_s31, av_s32);

        av_s21 = _mm512_add_pd(av_s3, av_s13);
        av_s22 = _mm512_add_pd(av_s5, av_s11);
        av_s23 = _mm512_add_pd(av_s7, av_s9);

        av_t4 = _mm512_mul_pd(v_CRTM_14_2, av_s21);
        av_t5 = _mm512_mul_pd(v_CRTM_14_4, av_s22);
        av_t6 = _mm512_mul_pd(v_CRTM_14_6, av_s23);
        av_s33 = _mm512_add_pd(av_t4, av_t5);
        // Output point 5: X(4)
        v_out4 = _mm512_add_pd(av_s33, av_t6);
        curr_out = out + out_strides[3];
        STRI_2x512_D(curr_out, v_out_stride, v_out3, v_out4);

        av_t7 = _mm512_mul_pd(v_CRTM_14_1, av_s17);
        av_t8 = _mm512_mul_pd(v_CRTM_14_3, av_s15);
        av_t9 = _mm512_mul_pd(v_CRTM_14_5, av_s16);
        av_s34 = _mm512_sub_pd(av_s2, av_t7);
        av_s35 = _mm512_sub_pd(av_t8, av_t9);
        // Output point 8: X(7)
        v_out7 = _mm512_add_pd(av_s34, av_s35);

        av_s24 = _mm512_sub_pd(av_s3, av_s13);
        av_s25 = _mm512_sub_pd(av_s5, av_s11);
        av_s26 = _mm512_sub_pd(av_s7, av_s9);

        av_t10 = _mm512_mul_pd(v_CRTM_14_2, av_s26);
        av_t11 = _mm512_mul_pd(v_CRTM_14_4, av_s24);
        av_t12 = _mm512_mul_pd(v_CRTM_14_6, av_s25);
        av_s36 = _mm512_add_pd(av_t10, av_t11);
        // Output point 9: X(8)
        v_out8 = _mm512_add_pd(av_s36, av_t12);
        curr_out = out + out_strides[7];
        STRI_2x512_D(curr_out, v_out_stride, v_out7, v_out8);

        av_t13 = _mm512_mul_pd(v_CRTM_14_1, av_s19);
        av_t14 = _mm512_mul_pd(v_CRTM_14_3, av_s20);
        av_t15 = _mm512_mul_pd(v_CRTM_14_5, av_s18);
        av_s37 = _mm512_sub_pd(av_s1, av_t13);
        av_s38 = _mm512_sub_pd(av_t14, av_t15);
        // Output point 12: X(11)
        v_out11 = _mm512_add_pd(av_s37, av_s38);

        av_t16 = _mm512_mul_pd(v_CRTM_14_2, av_s22);
        av_t17 = _mm512_mul_pd(v_CRTM_14_4, av_s23);
        av_t18 = _mm512_mul_pd(v_CRTM_14_6, av_s21);
        av_s39 = _mm512_sub_pd(av_t16, av_t17);
        // Output point 13: X(12)
        v_out12 = _mm512_add_pd(av_s39, av_t18);
        curr_out = out + out_strides[11];
        STRI_2x512_D(curr_out, v_out_stride, v_out11, v_out12);

        av_t19 = _mm512_mul_pd(v_CRTM_14_1, av_s16);
        av_t20 = _mm512_mul_pd(v_CRTM_14_3, av_s17);
        av_t21 = _mm512_mul_pd(v_CRTM_14_5, av_s15);
        av_s40 = _mm512_sub_pd(av_s2, av_t19);
        av_s41 = _mm512_sub_pd(av_t20, av_t21);
        // Output point 16: X(15)
        v_out15 = _mm512_add_pd(av_s40, av_s41);

        av_t22 = _mm512_mul_pd(v_CRTM_14_2, av_s25);
        av_t23 = _mm512_mul_pd(v_CRTM_14_4, av_s26);
        av_t24 = _mm512_mul_pd(v_CRTM_14_6, av_s24);
        av_s42 = _mm512_sub_pd(av_t24, av_t22);
        // Output point 17: X(16)
        v_out16 = _mm512_sub_pd(av_s42, av_t23);
        curr_out = out + out_strides[15];
        STRI_2x512_D(curr_out, v_out_stride, v_out15, v_out16);

        av_t25 = _mm512_mul_pd(v_CRTM_14_1, av_s20);
        av_t26 = _mm512_mul_pd(v_CRTM_14_3, av_s18);
        av_t27 = _mm512_mul_pd(v_CRTM_14_5, av_s19);
        av_s43 = _mm512_sub_pd(av_s1, av_t25);
        av_s44 = _mm512_sub_pd(av_t26, av_t27);
        // Output point 20: X(19)
        v_out19 = _mm512_add_pd(av_s43, av_s44);

        av_t28 = _mm512_mul_pd(v_CRTM_14_2, av_s23);
        av_t29 = _mm512_mul_pd(v_CRTM_14_4, av_s21);
        av_t30 = _mm512_mul_pd(v_CRTM_14_6, av_s22);
        av_s45 = _mm512_add_pd(av_t28, av_t29);
        // Output point 21: X(20)
        v_out20 = _mm512_sub_pd(av_s45, av_t30);
        curr_out = out + out_strides[19];
        STRI_2x512_D(curr_out, v_out_stride, v_out19, v_out20);

        av_t31 = _mm512_mul_pd(v_CRTM_14_1, av_s15);
        av_t32 = _mm512_mul_pd(v_CRTM_14_3, av_s16);
        av_t33 = _mm512_mul_pd(v_CRTM_14_5, av_s17);
        av_s46 = _mm512_sub_pd(av_s2, av_t31);
        av_s47 = _mm512_sub_pd(av_t32, av_t33);
        // Output point 24: X(23)
        v_out23 = _mm512_add_pd(av_s46, av_s47);

        av_t34 = _mm512_mul_pd(v_CRTM_14_2, av_s24);
        av_t35 = _mm512_mul_pd(v_CRTM_14_4, av_s25);
        av_t36 = _mm512_mul_pd(v_CRTM_14_6, av_s26);
        av_s48 = _mm512_sub_pd(av_t34, av_t35);
        // Output point 25: X(24)
        v_out24 = _mm512_add_pd(av_s48, av_t36);
        curr_out = out + out_strides[23];
        STRI_2x512_D(curr_out, v_out_stride, v_out23, v_out24);

        /* Shifted DFT */
        __m512d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
                bv_in8, bv_in9, bv_in10, bv_in11, bv_in12, bv_in13;
        __m512d bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
                bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17,
                bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24, bv_s25,
                bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32, bv_s33,
                bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40, bv_s41,
                bv_s42, bv_s43, bv_s44, bv_s45, bv_s46;
        __m512d bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
                bv_t10, bv_t11, bv_t12, bv_t13, bv_t14, bv_t15, bv_t16, bv_t17,
                bv_t18, bv_t19, bv_t20, bv_t21, bv_t22, bv_t23, bv_t24, bv_t25,
                bv_t26, bv_t27, bv_t28, bv_t29, bv_t30, bv_t31, bv_t32, bv_t33,
                bv_t34, bv_t35, bv_t36;

        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_512_D(curr_in, v_in_stride, bv_in0);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_512_D(curr_in, v_in_stride, bv_in1);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_512_D(curr_in, v_in_stride, bv_in2);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_512_D(curr_in, v_in_stride, bv_in3);
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDR_512_D(curr_in, v_in_stride, bv_in4);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_512_D(curr_in, v_in_stride, bv_in5);
        // Input point 14: x(13)
        curr_in = in + in_strides[13];
        LDR_512_D(curr_in, v_in_stride, bv_in6);
        // Input point 16: x(15)
        curr_in = in + in_strides[15];
        LDR_512_D(curr_in, v_in_stride, bv_in7);
        // Input point 18: x(17)
        curr_in = in + in_strides[17];
        LDR_512_D(curr_in, v_in_stride, bv_in8);
        // Input point 20: x(19)
        curr_in = in + in_strides[19];
        LDR_512_D(curr_in, v_in_stride, bv_in9);
        // Input point 22: x(21)
        curr_in = in + in_strides[21];
        LDR_512_D(curr_in, v_in_stride, bv_in10);
        // Input point 24: x(23)
        curr_in = in + in_strides[23];
        LDR_512_D(curr_in, v_in_stride, bv_in11);
        // Input point 26: x(25)
        curr_in = in + in_strides[25];
        LDR_512_D(curr_in, v_in_stride, bv_in12);
        // Input point 28: x(27)
        curr_in = in + in_strides[27];
        LDR_512_D(curr_in, v_in_stride, bv_in13);

        bv_s1 = _mm512_add_pd(bv_in1, bv_in13);
        bv_s2 = _mm512_sub_pd(bv_in1, bv_in13);
        bv_s3 = _mm512_add_pd(bv_in2, bv_in12);
        bv_s4 = _mm512_sub_pd(bv_in2, bv_in12);
        bv_s5 = _mm512_add_pd(bv_in3, bv_in11);
        bv_s6 = _mm512_sub_pd(bv_in3, bv_in11);
        bv_s7 = _mm512_add_pd(bv_in4, bv_in10);
        bv_s8 = _mm512_sub_pd(bv_in4, bv_in10);
        bv_s9 = _mm512_add_pd(bv_in5, bv_in9);
        bv_s10 = _mm512_sub_pd(bv_in5, bv_in9);
        bv_s11 = _mm512_add_pd(bv_in6, bv_in8);
        bv_s12 = _mm512_sub_pd(bv_in6, bv_in8);

        bv_t1 = _mm512_mul_pd(v_CRTM_14_5, bv_s12);
        bv_t2 = _mm512_mul_pd(v_CRTM_14_1, bv_s4);
        bv_t3 = _mm512_mul_pd(v_CRTM_14_3, bv_s8);
        bv_t4 = _mm512_mul_pd(v_CRTM_14_6, bv_s2);
        bv_t5 = _mm512_mul_pd(v_CRTM_14_2, bv_s10);
        bv_t6 = _mm512_mul_pd(v_CRTM_14_4, bv_s6);

        bv_t7 = _mm512_mul_pd(v_CRTM_14_5, bv_s1);
        bv_t8 = _mm512_mul_pd(v_CRTM_14_1, bv_s9);
        bv_t9 = _mm512_mul_pd(v_CRTM_14_3, bv_s5);
        bv_t10 = _mm512_mul_pd(v_CRTM_14_6, bv_s11);
        bv_t11 = _mm512_mul_pd(v_CRTM_14_2, bv_s3);
        bv_t12 = _mm512_mul_pd(v_CRTM_14_4, bv_s7);

        bv_t13 = _mm512_mul_pd(v_CRTM_14_5, bv_s4);
        bv_t14 = _mm512_mul_pd(v_CRTM_14_1, bv_s8);
        bv_t15 = _mm512_mul_pd(v_CRTM_14_3, bv_s12);
        bv_t16 = _mm512_mul_pd(v_CRTM_14_6, bv_s10);
        bv_t17 = _mm512_mul_pd(v_CRTM_14_2, bv_s6);
        bv_t18 = _mm512_mul_pd(v_CRTM_14_4, bv_s2);

        bv_t19 = _mm512_mul_pd(v_CRTM_14_5, bv_s9);
        bv_t20 = _mm512_mul_pd(v_CRTM_14_1, bv_s5);
        bv_t21 = _mm512_mul_pd(v_CRTM_14_3, bv_s1);
        bv_t22 = _mm512_mul_pd(v_CRTM_14_6, bv_s3);
        bv_t23 = _mm512_mul_pd(v_CRTM_14_2, bv_s7);
        bv_t24 = _mm512_mul_pd(v_CRTM_14_4, bv_s11);

        bv_t25 = _mm512_mul_pd(v_CRTM_14_5, bv_s8);
        bv_t26 = _mm512_mul_pd(v_CRTM_14_1, bv_s12);
        bv_t27 = _mm512_mul_pd(v_CRTM_14_3, bv_s4);
        bv_t28 = _mm512_mul_pd(v_CRTM_14_6, bv_s6);
        bv_t29 = _mm512_mul_pd(v_CRTM_14_2, bv_s2);
        bv_t30 = _mm512_mul_pd(v_CRTM_14_4, bv_s10);

        bv_t31 = _mm512_mul_pd(v_CRTM_14_5, bv_s5);
        bv_t32 = _mm512_mul_pd(v_CRTM_14_1, bv_s1);
        bv_t33 = _mm512_mul_pd(v_CRTM_14_3, bv_s9);
        bv_t34 = _mm512_mul_pd(v_CRTM_14_6, bv_s7);
        bv_t35 = _mm512_mul_pd(v_CRTM_14_2, bv_s11);
        bv_t36 = _mm512_mul_pd(v_CRTM_14_4, bv_s3);

        bv_s13 = _mm512_add_pd(bv_in0, bv_t1);
        bv_s14 = _mm512_add_pd(bv_t2, bv_t3);
        bv_s15 = _mm512_add_pd(bv_s13, bv_s14);
        bv_s16 = _mm512_add_pd(bv_t4, bv_t5);
        bv_s17 = _mm512_add_pd(bv_s16, bv_t6);
        // Output point 2: X(1)
        v_out1 = _mm512_add_pd(bv_s15, bv_s17);
        // Output point 26: X(25)
        v_out25 = _mm512_sub_pd(bv_s15, bv_s17);

        bv_s18 = _mm512_add_pd(bv_in7, bv_t7);
        bv_s19 = _mm512_add_pd(bv_t8, bv_t9);
        bv_s20 = _mm512_add_pd(bv_s18, bv_s19);
        bv_s21 = _mm512_add_pd(bv_t10, bv_t11);
        bv_s22 = _mm512_add_pd(bv_s21, bv_t12);
        // Output point 3: X(2)
        v_out2 = NEGATE_512_D(_mm512_add_pd(bv_s20, bv_s22));
        curr_out = out + out_strides[1];
        STRI_2x512_D(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 27: X(26)
        v_out26 = _mm512_sub_pd(bv_s22, bv_s20);
        curr_out = out + out_strides[25];
        STRI_2x512_D(curr_out, v_out_stride, v_out25, v_out26);

        bv_s23 = _mm512_add_pd(bv_in0, bv_t13);
        bv_s24 = _mm512_add_pd(bv_t14, bv_t15);
        bv_s25 = _mm512_sub_pd(bv_s23, bv_s24);
        bv_s26 = _mm512_add_pd(bv_t16, bv_t17);
        bv_s27 = _mm512_sub_pd(bv_t18, bv_s26);
        // Output point 6: X(5)
        v_out5 = _mm512_add_pd(bv_s25, bv_s27);
        // Output point 22: X(21)
        v_out21 = _mm512_sub_pd(bv_s25, bv_s27);

        bv_s28 = _mm512_add_pd(bv_in7, bv_t19);
        bv_s29 = _mm512_add_pd(bv_t20, bv_t21);
        bv_s30 = _mm512_sub_pd(bv_s28, bv_s29);
        bv_s31 = _mm512_add_pd(bv_t22, bv_t23);
        bv_s32 = _mm512_sub_pd(bv_t24, bv_s31);
        // Output point 7: X(6)
        v_out6 = _mm512_add_pd(bv_s30, bv_s32);
        curr_out = out + out_strides[5];
        STRI_2x512_D(curr_out, v_out_stride, v_out5, v_out6);
        // Output point 23: X(22)
        v_out22 = _mm512_sub_pd(bv_s30, bv_s32);
        curr_out = out + out_strides[21];
        STRI_2x512_D(curr_out, v_out_stride, v_out21, v_out22);

        bv_s33 = _mm512_sub_pd(bv_in0, bv_t25);
        bv_s34 = _mm512_sub_pd(bv_t26, bv_t27);
        bv_s35 = _mm512_add_pd(bv_s33, bv_s34);
        bv_s36 = _mm512_sub_pd(bv_t29, bv_t28);
        bv_s37 = _mm512_add_pd(bv_s36, bv_t30);
        // Output point 10: X(9)
        v_out9 = _mm512_add_pd(bv_s35, bv_s37);
        // Output point 18: X(17)
        v_out17 = _mm512_sub_pd(bv_s35, bv_s37);

        bv_s38 = _mm512_sub_pd(bv_t31, bv_in7);
        bv_s39 = _mm512_sub_pd(bv_t33, bv_t32);
        bv_s40 = _mm512_add_pd(bv_s38, bv_s39);
        bv_s41 = _mm512_sub_pd(bv_t34, bv_t35);
        bv_s42 = _mm512_sub_pd(bv_s41, bv_t36);
        // Output point 11: X(10)
        v_out10 = _mm512_add_pd(bv_s40, bv_s42);
        curr_out = out + out_strides[9];
        STRI_2x512_D(curr_out, v_out_stride, v_out9, v_out10);
        // Output point 19: X(18)
        v_out18 = _mm512_sub_pd(bv_s40, bv_s42);
        curr_out = out + out_strides[17];
        STRI_2x512_D(curr_out, v_out_stride, v_out17, v_out18);

        bv_s43 = _mm512_add_pd(bv_in0, bv_s8);
        bv_s44 = _mm512_add_pd(bv_s4, bv_s12);
        bv_s45 = _mm512_add_pd(bv_in7, bv_s5);
        bv_s46 = _mm512_add_pd(bv_s1, bv_s9);
        // Output point 14: X(13)
        v_out13 = _mm512_sub_pd(bv_s43, bv_s44);
        // Output point 15: X(14)
        v_out14 = _mm512_sub_pd(bv_s45, bv_s46);
        curr_out = out + out_strides[13];
        STRI_2x512_D(curr_out, v_out_stride, v_out13, v_out14);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_256_D)
    {
        /* Standard DFT */
        __m256d av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
                av_in8, av_in9, av_in10, av_in11, av_in12, av_in13;
        __m256d av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8, av_s9,
                av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16, av_s17,
                av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24, av_s25,
                av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32, av_s33,
                av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40, av_s41,
                av_s42, av_s43, av_s44, av_s45, av_s46, av_s47, av_s48;
        __m256d av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8, av_t9,
                av_t10, av_t11, av_t12, av_t13, av_t14, av_t15, av_t16, av_t17,
                av_t18, av_t19, av_t20, av_t21, av_t22, av_t23, av_t24, av_t25,
                av_t26, av_t27, av_t28, av_t29, av_t30, av_t31, av_t32, av_t33,
                av_t34, av_t35, av_t36;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
                v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21,
                v_out22, v_out23, v_out24, v_out25, v_out26, v_out27;

        curr_in = in;
        curr_out = out;

        __m256d v256_CRTM_14_1 = _mm512_castpd512_pd256(v_CRTM_14_1);
        __m256d v256_CRTM_14_2 = _mm512_castpd512_pd256(v_CRTM_14_2);
        __m256d v256_CRTM_14_3 = _mm512_castpd512_pd256(v_CRTM_14_3);
        __m256d v256_CRTM_14_4 = _mm512_castpd512_pd256(v_CRTM_14_4);
        __m256d v256_CRTM_14_5 = _mm512_castpd512_pd256(v_CRTM_14_5);
        __m256d v256_CRTM_14_6 = _mm512_castpd512_pd256(v_CRTM_14_6);

        // Input point 1: x(0)
        LDR_256_D(curr_in, v_in_stride, av_in0);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_256_D(curr_in, v_in_stride, av_in1);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_256_D(curr_in, v_in_stride, av_in2);
        // Input point 7: x(6)
        curr_in = in + in_strides[6];
        LDR_256_D(curr_in, v_in_stride, av_in3);
        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDR_256_D(curr_in, v_in_stride, av_in4);
        // Input point 11: x(10)
        curr_in = in + in_strides[10];
        LDR_256_D(curr_in, v_in_stride, av_in5);
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDR_256_D(curr_in, v_in_stride, av_in6);
        // Input point 15: x(14)
        curr_in = in + in_strides[14];
        LDR_256_D(curr_in, v_in_stride, av_in7);
        // Input point 17: x(16)
        curr_in = in + in_strides[16];
        LDR_256_D(curr_in, v_in_stride, av_in8);
        // Input point 19: x(18)
        curr_in = in + in_strides[18];
        LDR_256_D(curr_in, v_in_stride, av_in9);
        // Input point 21: x(20)
        curr_in = in + in_strides[20];
        LDR_256_D(curr_in, v_in_stride, av_in10);
        // Input point 23: x(22)
        curr_in = in + in_strides[22];
        LDR_256_D(curr_in, v_in_stride, av_in11);
        // Input point 25: x(24)
        curr_in = in + in_strides[24];
        LDR_256_D(curr_in, v_in_stride, av_in12);
        // Input point 27: x(26)
        curr_in = in + in_strides[26];
        LDR_256_D(curr_in, v_in_stride, av_in13);

        av_s1 = _mm256_sub_pd(av_in0, av_in7);
        av_s2 = _mm256_add_pd(av_in0, av_in7);
        av_s3 = _mm256_sub_pd(av_in13, av_in1);
        av_s4 = _mm256_add_pd(av_in13, av_in1);
        av_s5 = _mm256_sub_pd(av_in12, av_in2);
        av_s6 = _mm256_add_pd(av_in12, av_in2);
        av_s7 = _mm256_sub_pd(av_in11, av_in3);
        av_s8 = _mm256_add_pd(av_in11, av_in3);
        av_s9 = _mm256_sub_pd(av_in10, av_in4);
        av_s10 = _mm256_add_pd(av_in10, av_in4);
        av_s11 = _mm256_sub_pd(av_in9, av_in5);
        av_s12 = _mm256_add_pd(av_in9, av_in5);
        av_s13 = _mm256_sub_pd(av_in8, av_in6);
        av_s14 = _mm256_add_pd(av_in8, av_in6);

        av_s15 = _mm256_add_pd(av_s4, av_s14);
        av_s16 = _mm256_add_pd(av_s6, av_s12);
        av_s17 = _mm256_add_pd(av_s8, av_s10);

        av_s18 = _mm256_sub_pd(av_s14, av_s4);
        av_s19 = _mm256_sub_pd(av_s6, av_s12);
        av_s20 = _mm256_sub_pd(av_s10, av_s8);
        av_s27 = _mm256_add_pd(av_s2, av_s15);
        av_s28 = _mm256_add_pd(av_s16, av_s17);
        av_s29 = _mm256_add_pd(av_s1, av_s18);
        av_s30 = _mm256_add_pd(av_s19, av_s20);
        // Output pt 1: X(0)
        v_out0 = _mm256_add_pd(av_s27, av_s28);
        STR_256_D(curr_out, v_out_stride, v_out0);
        // Output pt 28: X(27)
        v_out27 = _mm256_add_pd(av_s29, av_s30);
        curr_out = out + out_strides[27];
        STR_256_D(curr_out, v_out_stride, v_out27);

        av_t1 = _mm256_mul_pd(v256_CRTM_14_1, av_s18);
        av_t2 = _mm256_mul_pd(v256_CRTM_14_3, av_s19);
        av_t3 = _mm256_mul_pd(v256_CRTM_14_5, av_s20);
        av_s31 = _mm256_sub_pd(av_s1, av_t1);
        av_s32 = _mm256_sub_pd(av_t2, av_t3);
        // Output point 4: X(3)
        v_out3 = _mm256_add_pd(av_s31, av_s32);

        av_s21 = _mm256_add_pd(av_s3, av_s13);
        av_s22 = _mm256_add_pd(av_s5, av_s11);
        av_s23 = _mm256_add_pd(av_s7, av_s9);

        av_t4 = _mm256_mul_pd(v256_CRTM_14_2, av_s21);
        av_t5 = _mm256_mul_pd(v256_CRTM_14_4, av_s22);
        av_t6 = _mm256_mul_pd(v256_CRTM_14_6, av_s23);
        av_s33 = _mm256_add_pd(av_t4, av_t5);
        // Output point 5: X(4)
        v_out4 = _mm256_add_pd(av_s33, av_t6);
        curr_out = out + out_strides[3];
        STRI_2x256_D(curr_out, v_out_stride, v_out3, v_out4);

        av_t7 = _mm256_mul_pd(v256_CRTM_14_1, av_s17);
        av_t8 = _mm256_mul_pd(v256_CRTM_14_3, av_s15);
        av_t9 = _mm256_mul_pd(v256_CRTM_14_5, av_s16);
        av_s34 = _mm256_sub_pd(av_s2, av_t7);
        av_s35 = _mm256_sub_pd(av_t8, av_t9);
        // Output point 8: X(7)
        v_out7 = _mm256_add_pd(av_s34, av_s35);

        av_s24 = _mm256_sub_pd(av_s3, av_s13);
        av_s25 = _mm256_sub_pd(av_s5, av_s11);
        av_s26 = _mm256_sub_pd(av_s7, av_s9);

        av_t10 = _mm256_mul_pd(v256_CRTM_14_2, av_s26);
        av_t11 = _mm256_mul_pd(v256_CRTM_14_4, av_s24);
        av_t12 = _mm256_mul_pd(v256_CRTM_14_6, av_s25);
        av_s36 = _mm256_add_pd(av_t10, av_t11);
        // Output point 9: X(8)
        v_out8 = _mm256_add_pd(av_s36, av_t12);
        curr_out = out + out_strides[7];
        STRI_2x256_D(curr_out, v_out_stride, v_out7, v_out8);

        av_t13 = _mm256_mul_pd(v256_CRTM_14_1, av_s19);
        av_t14 = _mm256_mul_pd(v256_CRTM_14_3, av_s20);
        av_t15 = _mm256_mul_pd(v256_CRTM_14_5, av_s18);
        av_s37 = _mm256_sub_pd(av_s1, av_t13);
        av_s38 = _mm256_sub_pd(av_t14, av_t15);
        // Output point 12: X(11)
        v_out11 = _mm256_add_pd(av_s37, av_s38);

        av_t16 = _mm256_mul_pd(v256_CRTM_14_2, av_s22);
        av_t17 = _mm256_mul_pd(v256_CRTM_14_4, av_s23);
        av_t18 = _mm256_mul_pd(v256_CRTM_14_6, av_s21);
        av_s39 = _mm256_sub_pd(av_t16, av_t17);
        // Output point 13: X(12)
        v_out12 = _mm256_add_pd(av_s39, av_t18);
        curr_out = out + out_strides[11];
        STRI_2x256_D(curr_out, v_out_stride, v_out11, v_out12);

        av_t19 = _mm256_mul_pd(v256_CRTM_14_1, av_s16);
        av_t20 = _mm256_mul_pd(v256_CRTM_14_3, av_s17);
        av_t21 = _mm256_mul_pd(v256_CRTM_14_5, av_s15);
        av_s40 = _mm256_sub_pd(av_s2, av_t19);
        av_s41 = _mm256_sub_pd(av_t20, av_t21);
        // Output point 16: X(15)
        v_out15 = _mm256_add_pd(av_s40, av_s41);

        av_t22 = _mm256_mul_pd(v256_CRTM_14_2, av_s25);
        av_t23 = _mm256_mul_pd(v256_CRTM_14_4, av_s26);
        av_t24 = _mm256_mul_pd(v256_CRTM_14_6, av_s24);
        av_s42 = _mm256_sub_pd(av_t24, av_t22);
        // Output point 17: X(16)
        v_out16 = _mm256_sub_pd(av_s42, av_t23);
        curr_out = out + out_strides[15];
        STRI_2x256_D(curr_out, v_out_stride, v_out15, v_out16);

        av_t25 = _mm256_mul_pd(v256_CRTM_14_1, av_s20);
        av_t26 = _mm256_mul_pd(v256_CRTM_14_3, av_s18);
        av_t27 = _mm256_mul_pd(v256_CRTM_14_5, av_s19);
        av_s43 = _mm256_sub_pd(av_s1, av_t25);
        av_s44 = _mm256_sub_pd(av_t26, av_t27);
        // Output point 20: X(19)
        v_out19 = _mm256_add_pd(av_s43, av_s44);

        av_t28 = _mm256_mul_pd(v256_CRTM_14_2, av_s23);
        av_t29 = _mm256_mul_pd(v256_CRTM_14_4, av_s21);
        av_t30 = _mm256_mul_pd(v256_CRTM_14_6, av_s22);
        av_s45 = _mm256_add_pd(av_t28, av_t29);
        // Output point 21: X(20)
        v_out20 = _mm256_sub_pd(av_s45, av_t30);
        curr_out = out + out_strides[19];
        STRI_2x256_D(curr_out, v_out_stride, v_out19, v_out20);

        av_t31 = _mm256_mul_pd(v256_CRTM_14_1, av_s15);
        av_t32 = _mm256_mul_pd(v256_CRTM_14_3, av_s16);
        av_t33 = _mm256_mul_pd(v256_CRTM_14_5, av_s17);
        av_s46 = _mm256_sub_pd(av_s2, av_t31);
        av_s47 = _mm256_sub_pd(av_t32, av_t33);
        // Output point 24: X(23)
        v_out23 = _mm256_add_pd(av_s46, av_s47);

        av_t34 = _mm256_mul_pd(v256_CRTM_14_2, av_s24);
        av_t35 = _mm256_mul_pd(v256_CRTM_14_4, av_s25);
        av_t36 = _mm256_mul_pd(v256_CRTM_14_6, av_s26);
        av_s48 = _mm256_sub_pd(av_t34, av_t35);
        // Output point 25: X(24)
        v_out24 = _mm256_add_pd(av_s48, av_t36);
        curr_out = out + out_strides[23];
        STRI_2x256_D(curr_out, v_out_stride, v_out23, v_out24);

        /* Shifted DFT */
        __m256d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
                bv_in8, bv_in9, bv_in10, bv_in11, bv_in12, bv_in13;
        __m256d bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
                bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17,
                bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24, bv_s25,
                bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32, bv_s33,
                bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40, bv_s41,
                bv_s42, bv_s43, bv_s44, bv_s45, bv_s46;
        __m256d bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
                bv_t10, bv_t11, bv_t12, bv_t13, bv_t14, bv_t15, bv_t16, bv_t17,
                bv_t18, bv_t19, bv_t20, bv_t21, bv_t22, bv_t23, bv_t24, bv_t25,
                bv_t26, bv_t27, bv_t28, bv_t29, bv_t30, bv_t31, bv_t32, bv_t33,
                bv_t34, bv_t35, bv_t36;

        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_256_D(curr_in, v_in_stride, bv_in0);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_256_D(curr_in, v_in_stride, bv_in1);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_256_D(curr_in, v_in_stride, bv_in2);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_256_D(curr_in, v_in_stride, bv_in3);
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDR_256_D(curr_in, v_in_stride, bv_in4);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_256_D(curr_in, v_in_stride, bv_in5);
        // Input point 14: x(13)
        curr_in = in + in_strides[13];
        LDR_256_D(curr_in, v_in_stride, bv_in6);
        // Input point 16: x(15)
        curr_in = in + in_strides[15];
        LDR_256_D(curr_in, v_in_stride, bv_in7);
        // Input point 18: x(17)
        curr_in = in + in_strides[17];
        LDR_256_D(curr_in, v_in_stride, bv_in8);
        // Input point 20: x(19)
        curr_in = in + in_strides[19];
        LDR_256_D(curr_in, v_in_stride, bv_in9);
        // Input point 22: x(21)
        curr_in = in + in_strides[21];
        LDR_256_D(curr_in, v_in_stride, bv_in10);
        // Input point 24: x(23)
        curr_in = in + in_strides[23];
        LDR_256_D(curr_in, v_in_stride, bv_in11);
        // Input point 26: x(25)
        curr_in = in + in_strides[25];
        LDR_256_D(curr_in, v_in_stride, bv_in12);
        // Input point 28: x(27)
        curr_in = in + in_strides[27];
        LDR_256_D(curr_in, v_in_stride, bv_in13);

        bv_s1 = _mm256_add_pd(bv_in1, bv_in13);
        bv_s2 = _mm256_sub_pd(bv_in1, bv_in13);
        bv_s3 = _mm256_add_pd(bv_in2, bv_in12);
        bv_s4 = _mm256_sub_pd(bv_in2, bv_in12);
        bv_s5 = _mm256_add_pd(bv_in3, bv_in11);
        bv_s6 = _mm256_sub_pd(bv_in3, bv_in11);
        bv_s7 = _mm256_add_pd(bv_in4, bv_in10);
        bv_s8 = _mm256_sub_pd(bv_in4, bv_in10);
        bv_s9 = _mm256_add_pd(bv_in5, bv_in9);
        bv_s10 = _mm256_sub_pd(bv_in5, bv_in9);
        bv_s11 = _mm256_add_pd(bv_in6, bv_in8);
        bv_s12 = _mm256_sub_pd(bv_in6, bv_in8);

        bv_t1 = _mm256_mul_pd(v256_CRTM_14_5, bv_s12);
        bv_t2 = _mm256_mul_pd(v256_CRTM_14_1, bv_s4);
        bv_t3 = _mm256_mul_pd(v256_CRTM_14_3, bv_s8);
        bv_t4 = _mm256_mul_pd(v256_CRTM_14_6, bv_s2);
        bv_t5 = _mm256_mul_pd(v256_CRTM_14_2, bv_s10);
        bv_t6 = _mm256_mul_pd(v256_CRTM_14_4, bv_s6);

        bv_t7 = _mm256_mul_pd(v256_CRTM_14_5, bv_s1);
        bv_t8 = _mm256_mul_pd(v256_CRTM_14_1, bv_s9);
        bv_t9 = _mm256_mul_pd(v256_CRTM_14_3, bv_s5);
        bv_t10 = _mm256_mul_pd(v256_CRTM_14_6, bv_s11);
        bv_t11 = _mm256_mul_pd(v256_CRTM_14_2, bv_s3);
        bv_t12 = _mm256_mul_pd(v256_CRTM_14_4, bv_s7);

        bv_t13 = _mm256_mul_pd(v256_CRTM_14_5, bv_s4);
        bv_t14 = _mm256_mul_pd(v256_CRTM_14_1, bv_s8);
        bv_t15 = _mm256_mul_pd(v256_CRTM_14_3, bv_s12);
        bv_t16 = _mm256_mul_pd(v256_CRTM_14_6, bv_s10);
        bv_t17 = _mm256_mul_pd(v256_CRTM_14_2, bv_s6);
        bv_t18 = _mm256_mul_pd(v256_CRTM_14_4, bv_s2);

        bv_t19 = _mm256_mul_pd(v256_CRTM_14_5, bv_s9);
        bv_t20 = _mm256_mul_pd(v256_CRTM_14_1, bv_s5);
        bv_t21 = _mm256_mul_pd(v256_CRTM_14_3, bv_s1);
        bv_t22 = _mm256_mul_pd(v256_CRTM_14_6, bv_s3);
        bv_t23 = _mm256_mul_pd(v256_CRTM_14_2, bv_s7);
        bv_t24 = _mm256_mul_pd(v256_CRTM_14_4, bv_s11);

        bv_t25 = _mm256_mul_pd(v256_CRTM_14_5, bv_s8);
        bv_t26 = _mm256_mul_pd(v256_CRTM_14_1, bv_s12);
        bv_t27 = _mm256_mul_pd(v256_CRTM_14_3, bv_s4);
        bv_t28 = _mm256_mul_pd(v256_CRTM_14_6, bv_s6);
        bv_t29 = _mm256_mul_pd(v256_CRTM_14_2, bv_s2);
        bv_t30 = _mm256_mul_pd(v256_CRTM_14_4, bv_s10);

        bv_t31 = _mm256_mul_pd(v256_CRTM_14_5, bv_s5);
        bv_t32 = _mm256_mul_pd(v256_CRTM_14_1, bv_s1);
        bv_t33 = _mm256_mul_pd(v256_CRTM_14_3, bv_s9);
        bv_t34 = _mm256_mul_pd(v256_CRTM_14_6, bv_s7);
        bv_t35 = _mm256_mul_pd(v256_CRTM_14_2, bv_s11);
        bv_t36 = _mm256_mul_pd(v256_CRTM_14_4, bv_s3);

        bv_s13 = _mm256_add_pd(bv_in0, bv_t1);
        bv_s14 = _mm256_add_pd(bv_t2, bv_t3);
        bv_s15 = _mm256_add_pd(bv_s13, bv_s14);
        bv_s16 = _mm256_add_pd(bv_t4, bv_t5);
        bv_s17 = _mm256_add_pd(bv_s16, bv_t6);
        // Output point 2: X(1)
        v_out1 = _mm256_add_pd(bv_s15, bv_s17);
        // Output point 26: X(25)
        v_out25 = _mm256_sub_pd(bv_s15, bv_s17);

        bv_s18 = _mm256_add_pd(bv_in7, bv_t7);
        bv_s19 = _mm256_add_pd(bv_t8, bv_t9);
        bv_s20 = _mm256_add_pd(bv_s18, bv_s19);
        bv_s21 = _mm256_add_pd(bv_t10, bv_t11);
        bv_s22 = _mm256_add_pd(bv_s21, bv_t12);
        // Output point 3: X(2)
        v_out2 = NEGATE_256_D(_mm256_add_pd(bv_s20, bv_s22));
        curr_out = out + out_strides[1];
        STRI_2x256_D(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 27: X(26)
        v_out26 = _mm256_sub_pd(bv_s22, bv_s20);
        curr_out = out + out_strides[25];
        STRI_2x256_D(curr_out, v_out_stride, v_out25, v_out26);

        bv_s23 = _mm256_add_pd(bv_in0, bv_t13);
        bv_s24 = _mm256_add_pd(bv_t14, bv_t15);
        bv_s25 = _mm256_sub_pd(bv_s23, bv_s24);
        bv_s26 = _mm256_add_pd(bv_t16, bv_t17);
        bv_s27 = _mm256_sub_pd(bv_t18, bv_s26);
        // Output point 6: X(5)
        v_out5 = _mm256_add_pd(bv_s25, bv_s27);
        // Output point 22: X(21)
        v_out21 = _mm256_sub_pd(bv_s25, bv_s27);

        bv_s28 = _mm256_add_pd(bv_in7, bv_t19);
        bv_s29 = _mm256_add_pd(bv_t20, bv_t21);
        bv_s30 = _mm256_sub_pd(bv_s28, bv_s29);
        bv_s31 = _mm256_add_pd(bv_t22, bv_t23);
        bv_s32 = _mm256_sub_pd(bv_t24, bv_s31);
        // Output point 7: X(6)
        v_out6 = _mm256_add_pd(bv_s30, bv_s32);
        curr_out = out + out_strides[5];
        STRI_2x256_D(curr_out, v_out_stride, v_out5, v_out6);
        // Output point 23: X(22)
        v_out22 = _mm256_sub_pd(bv_s30, bv_s32);
        curr_out = out + out_strides[21];
        STRI_2x256_D(curr_out, v_out_stride, v_out21, v_out22);

        bv_s33 = _mm256_sub_pd(bv_in0, bv_t25);
        bv_s34 = _mm256_sub_pd(bv_t26, bv_t27);
        bv_s35 = _mm256_add_pd(bv_s33, bv_s34);
        bv_s36 = _mm256_sub_pd(bv_t29, bv_t28);
        bv_s37 = _mm256_add_pd(bv_s36, bv_t30);
        // Output point 10: X(9)
        v_out9 = _mm256_add_pd(bv_s35, bv_s37);
        // Output point 18: X(17)
        v_out17 = _mm256_sub_pd(bv_s35, bv_s37);

        bv_s38 = _mm256_sub_pd(bv_t31, bv_in7);
        bv_s39 = _mm256_sub_pd(bv_t33, bv_t32);
        bv_s40 = _mm256_add_pd(bv_s38, bv_s39);
        bv_s41 = _mm256_sub_pd(bv_t34, bv_t35);
        bv_s42 = _mm256_sub_pd(bv_s41, bv_t36);
        // Output point 11: X(10)
        v_out10 = _mm256_add_pd(bv_s40, bv_s42);
        curr_out = out + out_strides[9];
        STRI_2x256_D(curr_out, v_out_stride, v_out9, v_out10);
        // Output point 19: X(18)
        v_out18 = _mm256_sub_pd(bv_s40, bv_s42);
        curr_out = out + out_strides[17];
        STRI_2x256_D(curr_out, v_out_stride, v_out17, v_out18);

        bv_s43 = _mm256_add_pd(bv_in0, bv_s8);
        bv_s44 = _mm256_add_pd(bv_s4, bv_s12);
        bv_s45 = _mm256_add_pd(bv_in7, bv_s5);
        bv_s46 = _mm256_add_pd(bv_s1, bv_s9);
        // Output point 14: X(13)
        v_out13 = _mm256_sub_pd(bv_s43, bv_s44);
        // Output point 15: X(14)
        v_out14 = _mm256_sub_pd(bv_s45, bv_s46);
        curr_out = out + out_strides[13];
        STRI_2x256_D(curr_out, v_out_stride, v_out13, v_out14);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_D)
    {
        /* Standard DFT */
        __m128d av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
                av_in8, av_in9, av_in10, av_in11, av_in12, av_in13;
        __m128d av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8, av_s9,
                av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16, av_s17,
                av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24, av_s25,
                av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32, av_s33,
                av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40, av_s41,
                av_s42, av_s43, av_s44, av_s45, av_s46, av_s47, av_s48;
        __m128d av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8, av_t9,
                av_t10, av_t11, av_t12, av_t13, av_t14, av_t15, av_t16, av_t17,
                av_t18, av_t19, av_t20, av_t21, av_t22, av_t23, av_t24, av_t25,
                av_t26, av_t27, av_t28, av_t29, av_t30, av_t31, av_t32, av_t33,
                av_t34, av_t35, av_t36;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
                v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21,
                v_out22, v_out23, v_out24, v_out25, v_out26, v_out27;

        curr_in = in;
        curr_out = out;

        __m128d v128_CRTM_14_1 = _mm512_castpd512_pd128(v_CRTM_14_1);
        __m128d v128_CRTM_14_2 = _mm512_castpd512_pd128(v_CRTM_14_2);
        __m128d v128_CRTM_14_3 = _mm512_castpd512_pd128(v_CRTM_14_3);
        __m128d v128_CRTM_14_4 = _mm512_castpd512_pd128(v_CRTM_14_4);
        __m128d v128_CRTM_14_5 = _mm512_castpd512_pd128(v_CRTM_14_5);
        __m128d v128_CRTM_14_6 = _mm512_castpd512_pd128(v_CRTM_14_6);

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
        // Input point 27: x(26)
        curr_in = in + in_strides[26];
        LDR_128_D(curr_in, v_in_stride, av_in13);

        av_s1 = _mm_sub_pd(av_in0, av_in7);
        av_s2 = _mm_add_pd(av_in0, av_in7);
        av_s3 = _mm_sub_pd(av_in13, av_in1);
        av_s4 = _mm_add_pd(av_in13, av_in1);
        av_s5 = _mm_sub_pd(av_in12, av_in2);
        av_s6 = _mm_add_pd(av_in12, av_in2);
        av_s7 = _mm_sub_pd(av_in11, av_in3);
        av_s8 = _mm_add_pd(av_in11, av_in3);
        av_s9 = _mm_sub_pd(av_in10, av_in4);
        av_s10 = _mm_add_pd(av_in10, av_in4);
        av_s11 = _mm_sub_pd(av_in9, av_in5);
        av_s12 = _mm_add_pd(av_in9, av_in5);
        av_s13 = _mm_sub_pd(av_in8, av_in6);
        av_s14 = _mm_add_pd(av_in8, av_in6);

        av_s15 = _mm_add_pd(av_s4, av_s14);
        av_s16 = _mm_add_pd(av_s6, av_s12);
        av_s17 = _mm_add_pd(av_s8, av_s10);

        av_s18 = _mm_sub_pd(av_s14, av_s4);
        av_s19 = _mm_sub_pd(av_s6, av_s12);
        av_s20 = _mm_sub_pd(av_s10, av_s8);
        av_s27 = _mm_add_pd(av_s2, av_s15);
        av_s28 = _mm_add_pd(av_s16, av_s17);
        av_s29 = _mm_add_pd(av_s1, av_s18);
        av_s30 = _mm_add_pd(av_s19, av_s20);
        // Output pt 1: X(0)
        v_out0 = _mm_add_pd(av_s27, av_s28);
        STR_128_D(curr_out, v_out_stride, v_out0);
        // Output pt 28: X(27)
        v_out27 = _mm_add_pd(av_s29, av_s30);
        curr_out = out + out_strides[27];
        STR_128_D(curr_out, v_out_stride, v_out27);

        av_t1 = _mm_mul_pd(v128_CRTM_14_1, av_s18);
        av_t2 = _mm_mul_pd(v128_CRTM_14_3, av_s19);
        av_t3 = _mm_mul_pd(v128_CRTM_14_5, av_s20);
        av_s31 = _mm_sub_pd(av_s1, av_t1);
        av_s32 = _mm_sub_pd(av_t2, av_t3);
        // Output point 4: X(3)
        v_out3 = _mm_add_pd(av_s31, av_s32);

        av_s21 = _mm_add_pd(av_s3, av_s13);
        av_s22 = _mm_add_pd(av_s5, av_s11);
        av_s23 = _mm_add_pd(av_s7, av_s9);

        av_t4 = _mm_mul_pd(v128_CRTM_14_2, av_s21);
        av_t5 = _mm_mul_pd(v128_CRTM_14_4, av_s22);
        av_t6 = _mm_mul_pd(v128_CRTM_14_6, av_s23);
        av_s33 = _mm_add_pd(av_t4, av_t5);
        // Output point 5: X(4)
        v_out4 = _mm_add_pd(av_s33, av_t6);
        curr_out = out + out_strides[3];
        STRI_2x128_D(curr_out, v_out_stride, v_out3, v_out4);

        av_t7 = _mm_mul_pd(v128_CRTM_14_1, av_s17);
        av_t8 = _mm_mul_pd(v128_CRTM_14_3, av_s15);
        av_t9 = _mm_mul_pd(v128_CRTM_14_5, av_s16);
        av_s34 = _mm_sub_pd(av_s2, av_t7);
        av_s35 = _mm_sub_pd(av_t8, av_t9);
        // Output point 8: X(7)
        v_out7 = _mm_add_pd(av_s34, av_s35);

        av_s24 = _mm_sub_pd(av_s3, av_s13);
        av_s25 = _mm_sub_pd(av_s5, av_s11);
        av_s26 = _mm_sub_pd(av_s7, av_s9);

        av_t10 = _mm_mul_pd(v128_CRTM_14_2, av_s26);
        av_t11 = _mm_mul_pd(v128_CRTM_14_4, av_s24);
        av_t12 = _mm_mul_pd(v128_CRTM_14_6, av_s25);
        av_s36 = _mm_add_pd(av_t10, av_t11);
        // Output point 9: X(8)
        v_out8 = _mm_add_pd(av_s36, av_t12);
        curr_out = out + out_strides[7];
        STRI_2x128_D(curr_out, v_out_stride, v_out7, v_out8);

        av_t13 = _mm_mul_pd(v128_CRTM_14_1, av_s19);
        av_t14 = _mm_mul_pd(v128_CRTM_14_3, av_s20);
        av_t15 = _mm_mul_pd(v128_CRTM_14_5, av_s18);
        av_s37 = _mm_sub_pd(av_s1, av_t13);
        av_s38 = _mm_sub_pd(av_t14, av_t15);
        // Output point 12: X(11)
        v_out11 = _mm_add_pd(av_s37, av_s38);

        av_t16 = _mm_mul_pd(v128_CRTM_14_2, av_s22);
        av_t17 = _mm_mul_pd(v128_CRTM_14_4, av_s23);
        av_t18 = _mm_mul_pd(v128_CRTM_14_6, av_s21);
        av_s39 = _mm_sub_pd(av_t16, av_t17);
        // Output point 13: X(12)
        v_out12 = _mm_add_pd(av_s39, av_t18);
        curr_out = out + out_strides[11];
        STRI_2x128_D(curr_out, v_out_stride, v_out11, v_out12);

        av_t19 = _mm_mul_pd(v128_CRTM_14_1, av_s16);
        av_t20 = _mm_mul_pd(v128_CRTM_14_3, av_s17);
        av_t21 = _mm_mul_pd(v128_CRTM_14_5, av_s15);
        av_s40 = _mm_sub_pd(av_s2, av_t19);
        av_s41 = _mm_sub_pd(av_t20, av_t21);
        // Output point 16: X(15)
        v_out15 = _mm_add_pd(av_s40, av_s41);

        av_t22 = _mm_mul_pd(v128_CRTM_14_2, av_s25);
        av_t23 = _mm_mul_pd(v128_CRTM_14_4, av_s26);
        av_t24 = _mm_mul_pd(v128_CRTM_14_6, av_s24);
        av_s42 = _mm_sub_pd(av_t24, av_t22);
        // Output point 17: X(16)
        v_out16 = _mm_sub_pd(av_s42, av_t23);
        curr_out = out + out_strides[15];
        STRI_2x128_D(curr_out, v_out_stride, v_out15, v_out16);

        av_t25 = _mm_mul_pd(v128_CRTM_14_1, av_s20);
        av_t26 = _mm_mul_pd(v128_CRTM_14_3, av_s18);
        av_t27 = _mm_mul_pd(v128_CRTM_14_5, av_s19);
        av_s43 = _mm_sub_pd(av_s1, av_t25);
        av_s44 = _mm_sub_pd(av_t26, av_t27);
        // Output point 20: X(19)
        v_out19 = _mm_add_pd(av_s43, av_s44);

        av_t28 = _mm_mul_pd(v128_CRTM_14_2, av_s23);
        av_t29 = _mm_mul_pd(v128_CRTM_14_4, av_s21);
        av_t30 = _mm_mul_pd(v128_CRTM_14_6, av_s22);
        av_s45 = _mm_add_pd(av_t28, av_t29);
        // Output point 21: X(20)
        v_out20 = _mm_sub_pd(av_s45, av_t30);
        curr_out = out + out_strides[19];
        STRI_2x128_D(curr_out, v_out_stride, v_out19, v_out20);

        av_t31 = _mm_mul_pd(v128_CRTM_14_1, av_s15);
        av_t32 = _mm_mul_pd(v128_CRTM_14_3, av_s16);
        av_t33 = _mm_mul_pd(v128_CRTM_14_5, av_s17);
        av_s46 = _mm_sub_pd(av_s2, av_t31);
        av_s47 = _mm_sub_pd(av_t32, av_t33);
        // Output point 24: X(23)
        v_out23 = _mm_add_pd(av_s46, av_s47);

        av_t34 = _mm_mul_pd(v128_CRTM_14_2, av_s24);
        av_t35 = _mm_mul_pd(v128_CRTM_14_4, av_s25);
        av_t36 = _mm_mul_pd(v128_CRTM_14_6, av_s26);
        av_s48 = _mm_sub_pd(av_t34, av_t35);
        // Output point 25: X(24)
        v_out24 = _mm_add_pd(av_s48, av_t36);
        curr_out = out + out_strides[23];
        STRI_2x128_D(curr_out, v_out_stride, v_out23, v_out24);

        /* Shifted DFT */
        __m128d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
                bv_in8, bv_in9, bv_in10, bv_in11, bv_in12, bv_in13;
        __m128d bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
                bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17,
                bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24, bv_s25,
                bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32, bv_s33,
                bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40, bv_s41,
                bv_s42, bv_s43, bv_s44, bv_s45, bv_s46;
        __m128d bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
                bv_t10, bv_t11, bv_t12, bv_t13, bv_t14, bv_t15, bv_t16, bv_t17,
                bv_t18, bv_t19, bv_t20, bv_t21, bv_t22, bv_t23, bv_t24, bv_t25,
                bv_t26, bv_t27, bv_t28, bv_t29, bv_t30, bv_t31, bv_t32, bv_t33,
                bv_t34, bv_t35, bv_t36;

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
        // Input point 28: x(27)
        curr_in = in + in_strides[27];
        LDR_128_D(curr_in, v_in_stride, bv_in13);

        bv_s1 = _mm_add_pd(bv_in1, bv_in13);
        bv_s2 = _mm_sub_pd(bv_in1, bv_in13);
        bv_s3 = _mm_add_pd(bv_in2, bv_in12);
        bv_s4 = _mm_sub_pd(bv_in2, bv_in12);
        bv_s5 = _mm_add_pd(bv_in3, bv_in11);
        bv_s6 = _mm_sub_pd(bv_in3, bv_in11);
        bv_s7 = _mm_add_pd(bv_in4, bv_in10);
        bv_s8 = _mm_sub_pd(bv_in4, bv_in10);
        bv_s9 = _mm_add_pd(bv_in5, bv_in9);
        bv_s10 = _mm_sub_pd(bv_in5, bv_in9);
        bv_s11 = _mm_add_pd(bv_in6, bv_in8);
        bv_s12 = _mm_sub_pd(bv_in6, bv_in8);

        bv_t1 = _mm_mul_pd(v128_CRTM_14_5, bv_s12);
        bv_t2 = _mm_mul_pd(v128_CRTM_14_1, bv_s4);
        bv_t3 = _mm_mul_pd(v128_CRTM_14_3, bv_s8);
        bv_t4 = _mm_mul_pd(v128_CRTM_14_6, bv_s2);
        bv_t5 = _mm_mul_pd(v128_CRTM_14_2, bv_s10);
        bv_t6 = _mm_mul_pd(v128_CRTM_14_4, bv_s6);

        bv_t7 = _mm_mul_pd(v128_CRTM_14_5, bv_s1);
        bv_t8 = _mm_mul_pd(v128_CRTM_14_1, bv_s9);
        bv_t9 = _mm_mul_pd(v128_CRTM_14_3, bv_s5);
        bv_t10 = _mm_mul_pd(v128_CRTM_14_6, bv_s11);
        bv_t11 = _mm_mul_pd(v128_CRTM_14_2, bv_s3);
        bv_t12 = _mm_mul_pd(v128_CRTM_14_4, bv_s7);

        bv_t13 = _mm_mul_pd(v128_CRTM_14_5, bv_s4);
        bv_t14 = _mm_mul_pd(v128_CRTM_14_1, bv_s8);
        bv_t15 = _mm_mul_pd(v128_CRTM_14_3, bv_s12);
        bv_t16 = _mm_mul_pd(v128_CRTM_14_6, bv_s10);
        bv_t17 = _mm_mul_pd(v128_CRTM_14_2, bv_s6);
        bv_t18 = _mm_mul_pd(v128_CRTM_14_4, bv_s2);

        bv_t19 = _mm_mul_pd(v128_CRTM_14_5, bv_s9);
        bv_t20 = _mm_mul_pd(v128_CRTM_14_1, bv_s5);
        bv_t21 = _mm_mul_pd(v128_CRTM_14_3, bv_s1);
        bv_t22 = _mm_mul_pd(v128_CRTM_14_6, bv_s3);
        bv_t23 = _mm_mul_pd(v128_CRTM_14_2, bv_s7);
        bv_t24 = _mm_mul_pd(v128_CRTM_14_4, bv_s11);

        bv_t25 = _mm_mul_pd(v128_CRTM_14_5, bv_s8);
        bv_t26 = _mm_mul_pd(v128_CRTM_14_1, bv_s12);
        bv_t27 = _mm_mul_pd(v128_CRTM_14_3, bv_s4);
        bv_t28 = _mm_mul_pd(v128_CRTM_14_6, bv_s6);
        bv_t29 = _mm_mul_pd(v128_CRTM_14_2, bv_s2);
        bv_t30 = _mm_mul_pd(v128_CRTM_14_4, bv_s10);

        bv_t31 = _mm_mul_pd(v128_CRTM_14_5, bv_s5);
        bv_t32 = _mm_mul_pd(v128_CRTM_14_1, bv_s1);
        bv_t33 = _mm_mul_pd(v128_CRTM_14_3, bv_s9);
        bv_t34 = _mm_mul_pd(v128_CRTM_14_6, bv_s7);
        bv_t35 = _mm_mul_pd(v128_CRTM_14_2, bv_s11);
        bv_t36 = _mm_mul_pd(v128_CRTM_14_4, bv_s3);

        bv_s13 = _mm_add_pd(bv_in0, bv_t1);
        bv_s14 = _mm_add_pd(bv_t2, bv_t3);
        bv_s15 = _mm_add_pd(bv_s13, bv_s14);
        bv_s16 = _mm_add_pd(bv_t4, bv_t5);
        bv_s17 = _mm_add_pd(bv_s16, bv_t6);
        // Output point 2: X(1)
        v_out1 = _mm_add_pd(bv_s15, bv_s17);
        // Output point 26: X(25)
        v_out25 = _mm_sub_pd(bv_s15, bv_s17);

        bv_s18 = _mm_add_pd(bv_in7, bv_t7);
        bv_s19 = _mm_add_pd(bv_t8, bv_t9);
        bv_s20 = _mm_add_pd(bv_s18, bv_s19);
        bv_s21 = _mm_add_pd(bv_t10, bv_t11);
        bv_s22 = _mm_add_pd(bv_s21, bv_t12);
        // Output point 3: X(2)
        v_out2 = NEGATE_128_D(_mm_add_pd(bv_s20, bv_s22));
        curr_out = out + out_strides[1];
        STRI_2x128_D(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 27: X(26)
        v_out26 = _mm_sub_pd(bv_s22, bv_s20);
        curr_out = out + out_strides[25];
        STRI_2x128_D(curr_out, v_out_stride, v_out25, v_out26);

        bv_s23 = _mm_add_pd(bv_in0, bv_t13);
        bv_s24 = _mm_add_pd(bv_t14, bv_t15);
        bv_s25 = _mm_sub_pd(bv_s23, bv_s24);
        bv_s26 = _mm_add_pd(bv_t16, bv_t17);
        bv_s27 = _mm_sub_pd(bv_t18, bv_s26);
        // Output point 6: X(5)
        v_out5 = _mm_add_pd(bv_s25, bv_s27);
        // Output point 22: X(21)
        v_out21 = _mm_sub_pd(bv_s25, bv_s27);

        bv_s28 = _mm_add_pd(bv_in7, bv_t19);
        bv_s29 = _mm_add_pd(bv_t20, bv_t21);
        bv_s30 = _mm_sub_pd(bv_s28, bv_s29);
        bv_s31 = _mm_add_pd(bv_t22, bv_t23);
        bv_s32 = _mm_sub_pd(bv_t24, bv_s31);
        // Output point 7: X(6)
        v_out6 = _mm_add_pd(bv_s30, bv_s32);
        curr_out = out + out_strides[5];
        STRI_2x128_D(curr_out, v_out_stride, v_out5, v_out6);
        // Output point 23: X(22)
        v_out22 = _mm_sub_pd(bv_s30, bv_s32);
        curr_out = out + out_strides[21];
        STRI_2x128_D(curr_out, v_out_stride, v_out21, v_out22);

        bv_s33 = _mm_sub_pd(bv_in0, bv_t25);
        bv_s34 = _mm_sub_pd(bv_t26, bv_t27);
        bv_s35 = _mm_add_pd(bv_s33, bv_s34);
        bv_s36 = _mm_sub_pd(bv_t29, bv_t28);
        bv_s37 = _mm_add_pd(bv_s36, bv_t30);
        // Output point 10: X(9)
        v_out9 = _mm_add_pd(bv_s35, bv_s37);
        // Output point 18: X(17)
        v_out17 = _mm_sub_pd(bv_s35, bv_s37);

        bv_s38 = _mm_sub_pd(bv_t31, bv_in7);
        bv_s39 = _mm_sub_pd(bv_t33, bv_t32);
        bv_s40 = _mm_add_pd(bv_s38, bv_s39);
        bv_s41 = _mm_sub_pd(bv_t34, bv_t35);
        bv_s42 = _mm_sub_pd(bv_s41, bv_t36);
        // Output point 11: X(10)
        v_out10 = _mm_add_pd(bv_s40, bv_s42);
        curr_out = out + out_strides[9];
        STRI_2x128_D(curr_out, v_out_stride, v_out9, v_out10);
        // Output point 19: X(18)
        v_out18 = _mm_sub_pd(bv_s40, bv_s42);
        curr_out = out + out_strides[17];
        STRI_2x128_D(curr_out, v_out_stride, v_out17, v_out18);

        bv_s43 = _mm_add_pd(bv_in0, bv_s8);
        bv_s44 = _mm_add_pd(bv_s4, bv_s12);
        bv_s45 = _mm_add_pd(bv_in7, bv_s5);
        bv_s46 = _mm_add_pd(bv_s1, bv_s9);
        // Output point 14: X(13)
        v_out13 = _mm_sub_pd(bv_s43, bv_s44);
        // Output point 15: X(14)
        v_out14 = _mm_sub_pd(bv_s45, bv_s46);
        curr_out = out + out_strides[13];
        STRI_2x128_D(curr_out, v_out_stride, v_out13, v_out14);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        /* Standard DFT */
        FFTZ_DOUBLE a_in0, a_in1, a_in2, a_in3, a_in4, a_in5, a_in6, a_in7,
            a_in8, a_in9, a_in10, a_in11, a_in12, a_in13;
        FFTZ_DOUBLE a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_s9,
               a_s10, a_s11, a_s12, a_s13, a_s14, a_s15, a_s16, a_s17, a_s18,
               a_s19, a_s20, a_s21, a_s22, a_s23, a_s24, a_s25, a_s26, a_s27,
               a_s28, a_s29, a_s30, a_s31, a_s32, a_s33, a_s34, a_s35, a_s36,
               a_s37, a_s38, a_s39, a_s40, a_s41, a_s42, a_s43, a_s44, a_s45,
               a_s46, a_s47;
        FFTZ_DOUBLE a_t0, a_t1, a_t2, a_t3, a_t4, a_t5, a_t6, a_t7, a_t8, a_t9,
               a_t10, a_t11, a_t12, a_t13, a_t14, a_t15, a_t16, a_t17, a_t18,
               a_t19, a_t20, a_t21, a_t22, a_t23, a_t24, a_t25, a_t26, a_t27,
               a_t28, a_t29, a_t30, a_t31, a_t32, a_t33, a_t34, a_t35;

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

        a_s0 = a_in0 - a_in7;
        a_s1 = a_in0 + a_in7;
        a_s2 = a_in13 - a_in1;
        a_s3 = a_in13 + a_in1;
        a_s4 = a_in12 - a_in2;
        a_s5 = a_in12 + a_in2;
        a_s6 = a_in11 - a_in3;
        a_s7 = a_in11 + a_in3;
        a_s8 = a_in10 - a_in4;
        a_s9 = a_in10 + a_in4;
        a_s10 = a_in9 - a_in5;
        a_s11 = a_in9 + a_in5;
        a_s12 = a_in8 - a_in6;
        a_s13 = a_in8 + a_in6;

        a_s14 = a_s3 + a_s13;
        a_s15 = a_s5 + a_s11;
        a_s16 = a_s7 + a_s9;

        a_s17 = a_s13 - a_s3;
        a_s18 = a_s5 - a_s11;
        a_s19 = a_s9 - a_s7;
        a_s26 = a_s1 + a_s14;
        a_s27 = a_s15 + a_s16;
        a_s28 = a_s0 + a_s17;
        a_s29 = a_s18 + a_s19;
        // Output point 1: X(0)
        *out = a_s26 + a_s27;
        // Output point 28: X(27)
        out[out_strides[27]] = a_s28 + a_s29;

        a_t0 = CRTM_14_1 * a_s17;
        a_t1 = CRTM_14_3 * a_s18;
        a_t2 = CRTM_14_5 * a_s19;
        a_s30 = a_s0 - a_t0;
        a_s31 = a_t1 - a_t2;
        // Output point 4: X(3)
        out[out_strides[3]] = a_s30 + a_s31;

        a_s20 = a_s2 + a_s12;
        a_s21 = a_s4 + a_s10;
        a_s22 = a_s6 + a_s8;
        a_t3 = CRTM_14_2 * a_s20;
        a_t4 = CRTM_14_4 * a_s21;
        a_t5 = CRTM_14_6 * a_s22;
        a_s32 = a_t3 + a_t4;
        // Output point 5: X(4)
        out[out_strides[4]] = a_s32 + a_t5;

        a_t6 = CRTM_14_1 * a_s16;
        a_t7 = CRTM_14_3 * a_s14;
        a_t8 = CRTM_14_5 * a_s15;
        a_s33 = a_s1 - a_t6;
        a_s34 = a_t7 - a_t8;
        // Output point 8: X(7)
        out[out_strides[7]] = a_s33 + a_s34;

        a_s23 = a_s2 - a_s12;
        a_s24 = a_s4 - a_s10;
        a_s25 = a_s6 - a_s8;
        a_t9 = CRTM_14_2 * a_s25;
        a_t10 = CRTM_14_4 * a_s23;
        a_t11 = CRTM_14_6 * a_s24;
        a_s35 = a_t9 + a_t10;
        // Output point 9: X(8)
        out[out_strides[8]] = a_s35 + a_t11;

        a_t12 = CRTM_14_1 * a_s18;
        a_t13 = CRTM_14_3 * a_s19;
        a_t14 = CRTM_14_5 * a_s17;
        a_s36 = a_s0 - a_t12;
        a_s37 = a_t13 - a_t14;
        // Output point 12: X(11)
        out[out_strides[11]] = a_s36 + a_s37;

        a_t15 = CRTM_14_2 * a_s21;
        a_t16 = CRTM_14_4 * a_s22;
        a_t17 = CRTM_14_6 * a_s20;
        a_s38 = a_t15 - a_t16;
        // Output point 13: X(12)
        out[out_strides[12]] = a_s38 + a_t17;

        a_t18 = CRTM_14_1 * a_s15;
        a_t19 = CRTM_14_3 * a_s16;
        a_t20 = CRTM_14_5 * a_s14;
        a_s39 = a_s1 - a_t18;
        a_s40 = a_t19 - a_t20;
        // Output point 16: X(15)
        out[out_strides[15]] = a_s39 + a_s40;

        a_t21 = CRTM_14_2 * a_s24;
        a_t22 = CRTM_14_4 * a_s25;
        a_t23 = CRTM_14_6 * a_s23;
        a_s41 = a_t23 - a_t21;
        // Output point 17: X(16)
        out[out_strides[16]] = a_s41 - a_t22;

        a_t24 = CRTM_14_1 * a_s19;
        a_t25 = CRTM_14_3 * a_s17;
        a_t26 = CRTM_14_5 * a_s18;
        a_s42 = a_s0 - a_t24;
        a_s43 = a_t25 - a_t26;
        // Output point 20: X(19)
        out[out_strides[19]] = a_s42 + a_s43;

        a_t27 = CRTM_14_2 * a_s22;
        a_t28 = CRTM_14_4 * a_s20;
        a_t29 = CRTM_14_6 * a_s21;
        a_s44 = a_t27 + a_t28;
        // Output point 21: X(20)
        out[out_strides[20]] = a_s44 - a_t29;

        a_t30 = CRTM_14_1 * a_s14;
        a_t31 = CRTM_14_3 * a_s15;
        a_t32 = CRTM_14_5 * a_s16;
        a_s45 = a_s1 - a_t30;
        a_s46 = a_t31 - a_t32;
        // Output point 24: X(23)
        out[out_strides[23]] = a_s45 + a_s46;

        a_t33 = CRTM_14_2 * a_s23;
        a_t34 = CRTM_14_4 * a_s24;
        a_t35 = CRTM_14_6 * a_s25;
        a_s47 = a_t33 - a_t34;
        // Output point 25: X(24)
        out[out_strides[24]] = a_s47 + a_t35;

        /* Shifted DFT */
        FFTZ_DOUBLE b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7,
            b_in8, b_in9, b_in10, b_in11, b_in12, b_in13;
        FFTZ_DOUBLE b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9,
               b_s10, b_s11, b_s12, b_s13, b_s14, b_s15, b_s16, b_s17, b_s18,
               b_s19, b_s20, b_s21, b_s22, b_s23, b_s24, b_s25, b_s26, b_s27,
               b_s28, b_s29, b_s30, b_s31, b_s32, b_s33, b_s34, b_s35, b_s36,
               b_s37, b_s38, b_s39, b_s40, b_s41, b_s42, b_s43, b_s44, b_s45;
        FFTZ_DOUBLE b_t0, b_t1, b_t2, b_t3, b_t4, b_t5, b_t6, b_t7, b_t8, b_t9,
               b_t10, b_t11, b_t12, b_t13, b_t14, b_t15, b_t16, b_t17, b_t18,
               b_t19, b_t20, b_t21, b_t22, b_t23, b_t24, b_t25, b_t26, b_t27,
               b_t28, b_t29, b_t30, b_t31, b_t32, b_t33, b_t34, b_t35;

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

        b_s0 = b_in1 + b_in13;
        b_s1 = b_in1 - b_in13;
        b_s2 = b_in2 + b_in12;
        b_s3 = b_in2 - b_in12;
        b_s4 = b_in3 + b_in11;
        b_s5 = b_in3 - b_in11;
        b_s6 = b_in4 + b_in10;
        b_s7 = b_in4 - b_in10;
        b_s8 = b_in5 + b_in9;
        b_s9 = b_in5 - b_in9;
        b_s10 = b_in6 + b_in8;
        b_s11 = b_in6 - b_in8;

        b_t0 = CRTM_14_5 * b_s11;
        b_t1 = CRTM_14_1 * b_s3;
        b_t2 = CRTM_14_3 * b_s7;
        b_t3 = CRTM_14_6 * b_s1;
        b_t4 = CRTM_14_2 * b_s9;
        b_t5 = CRTM_14_4 * b_s5;

        b_t6 = CRTM_14_5 * b_s0;
        b_t7 = CRTM_14_1 * b_s8;
        b_t8 = CRTM_14_3 * b_s4;
        b_t9 = CRTM_14_6 * b_s10;
        b_t10 = CRTM_14_2 * b_s2;
        b_t11 = CRTM_14_4 * b_s6;

        b_t12 = CRTM_14_5 * b_s3;
        b_t13 = CRTM_14_1 * b_s7;
        b_t14 = CRTM_14_3 * b_s11;
        b_t15 = CRTM_14_6 * b_s9;
        b_t16 = CRTM_14_2 * b_s5;
        b_t17 = CRTM_14_4 * b_s1;

        b_t18 = CRTM_14_5 * b_s8;
        b_t19 = CRTM_14_1 * b_s4;
        b_t20 = CRTM_14_3 * b_s0;
        b_t21 = CRTM_14_6 * b_s2;
        b_t22 = CRTM_14_2 * b_s6;
        b_t23 = CRTM_14_4 * b_s10;

        b_t24 = CRTM_14_5 * b_s7;
        b_t25 = CRTM_14_1 * b_s11;
        b_t26 = CRTM_14_3 * b_s3;
        b_t27 = CRTM_14_6 * b_s5;
        b_t28 = CRTM_14_2 * b_s1;
        b_t29 = CRTM_14_4 * b_s9;

        b_t30 = CRTM_14_5 * b_s4;
        b_t31 = CRTM_14_1 * b_s0;
        b_t32 = CRTM_14_3 * b_s8;
        b_t33 = CRTM_14_6 * b_s6;
        b_t34 = CRTM_14_2 * b_s10;
        b_t35 = CRTM_14_4 * b_s2;

        b_s12 = b_in0 + b_t0;
        b_s13 = b_t1 + b_t2;
        b_s14 = b_s12 + b_s13;
        b_s15 = b_t3 + b_t4;
        b_s16 = b_s15 + b_t5;
        // Output point 2: X(1)
        out[out_strides[1]] = b_s14 + b_s16;
        // Output point 26: X(25)
        out[out_strides[25]] = b_s14 - b_s16;

        b_s17 = b_in7 + b_t6;
        b_s18 = b_t7 + b_t8;
        b_s19 = b_s17 + b_s18;
        b_s20 = b_t9 + b_t10;
        b_s21 = b_s20 + b_t11;
        // Output point 3: X(2)
        out[out_strides[2]] = -(b_s19 + b_s21);
        // Output point 27: X(26)
        out[out_strides[26]] = b_s21 - b_s19;

        b_s22 = b_in0 + b_t12;
        b_s23 = b_t13 + b_t14;
        b_s24 = b_s22 - b_s23;
        b_s25 = b_t15 + b_t16;
        b_s26 = b_t17 - b_s25;
        // Output point 6: X(5)
        out[out_strides[5]] = b_s24 + b_s26;
        // Output point 22: X(21)
        out[out_strides[21]] = b_s24 - b_s26;

        b_s27 = b_in7 + b_t18;
        b_s28 = b_t19 + b_t20;
        b_s29 = b_s27 - b_s28;
        b_s30 = b_t21 + b_t22;
        b_s31 = b_t23 - b_s30;
        // Output point 7: X(6)
        out[out_strides[6]] = b_s29 + b_s31;
        // Output point 23: X(22)
        out[out_strides[22]] = b_s29 - b_s31;

        b_s32 = b_in0 - b_t24;
        b_s33 = b_t25 - b_t26;
        b_s34 = b_s32 + b_s33;
        b_s35 = b_t28 - b_t27;
        b_s36 = b_s35 + b_t29;
        // Output point 10: X(9)
        out[out_strides[9]] = b_s34 + b_s36;
        // Output point 18: X(17)
        out[out_strides[17]] = b_s34 - b_s36;

        b_s37 = b_t30 - b_in7;
        b_s38 = b_t32 - b_t31;
        b_s39 = b_s37 + b_s38;
        b_s40 = b_t33 - b_t34;
        b_s41 = b_s40 - b_t35;
        // Output point 11: X(10)
        out[out_strides[10]] = b_s39 + b_s41;
        // Output point 19: X(18)
        out[out_strides[18]] = b_s39 - b_s41;

        b_s42 = b_in0 + b_s7;
        b_s43 = b_s3 + b_s11;
        b_s44 = b_in7 + b_s4;
        b_s45 = b_s0 + b_s8;
        // Output point 14: X(13)
        out[out_strides[13]] = b_s42 - b_s43;
        // Output point 15: X(14)
        out[out_strides[14]] = b_s44 - b_s45;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft14avx512_fp64_bwd(FFTZ_VOID *in_real,
                                             FFTZ_VOID *in_imag,
                                             FFTZ_VOID *out_real,
                                             FFTZ_VOID *out_imag, FFTZ_INTP n,
                                             aoclfftz_strides_t *strides,
                                             FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_14_1 =
        0.867767478235116240951536665696717509219981456;
    const FFTZ_DOUBLE CRTM_14_2 =
        1.801937735804838252472204639014890102331838324;
    const FFTZ_DOUBLE CRTM_14_3 =
        1.563662964936059617416889053348115500464669037;
    const FFTZ_DOUBLE CRTM_14_4 =
        1.246979603717467061050009768008479621264549462;
    const FFTZ_DOUBLE CRTM_14_5 =
        1.949855824363647214036263365987862434465571601;
    const FFTZ_DOUBLE CRTM_14_6 =
        0.445041867912628808577805128993589518932711138;
    const FFTZ_DOUBLE CRTM_14_7 =
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
    FFTZ_DOUBLE *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_256_S;
    FFTZ_INTP remaining_sets = n % NUM_SETS_REAL_256_S;

    __m512d v_CRTM_14_1 = _mm512_set1_pd(CRTM_14_1);
    __m512d v_CRTM_14_2 = _mm512_set1_pd(CRTM_14_2);
    __m512d v_CRTM_14_3 = _mm512_set1_pd(CRTM_14_3);
    __m512d v_CRTM_14_4 = _mm512_set1_pd(CRTM_14_4);
    __m512d v_CRTM_14_5 = _mm512_set1_pd(CRTM_14_5);
    __m512d v_CRTM_14_6 = _mm512_set1_pd(CRTM_14_6);
    __m512d v_CRTM_14_7 = _mm512_set1_pd(CRTM_14_7);

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m512d av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
                av_in8, av_in9, av_in10, av_in11, av_in12, av_in13;
        __m512d av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8, av_s9,
                av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16, av_s17,
                av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24, av_s25,
                av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32, av_s33,
                av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40, av_s41,
                av_s42, av_s43, av_s44, av_s45, av_s46, av_s47, av_s48;
        __m512d av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8, av_t9,
                av_t10, av_t11, av_t12, av_t13, av_t14, av_t15, av_t16, av_t17,
                av_t18, av_t19, av_t20, av_t21, av_t22, av_t23, av_t24, av_t25,
                av_t26, av_t27, av_t28, av_t29, av_t30, av_t31, av_t32, av_t33,
                av_t34, av_t35, av_t36, av_t37, av_t38;
        __m512d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
                v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21,
                v_out22, v_out23, v_out24, v_out25, v_out26, v_out27;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_512_D(curr_in, v_in_stride, av_in0);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x512_D(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in + in_strides[7];
        LDRI_2x512_D(curr_in, v_in_stride, av_in3, av_in4);
        // Input point 12: x(11) & Input point 13: x(12)
        curr_in = in + in_strides[11];
        LDRI_2x512_D(curr_in, v_in_stride, av_in5, av_in6);
        // Input point 16: x(15) & Input point 17: x(16)
        curr_in = in + in_strides[15];
        LDRI_2x512_D(curr_in, v_in_stride, av_in7, av_in8);
        // Input point 20: x(19) & Input point 21: x(20)
        curr_in = in + in_strides[19];
        LDRI_2x512_D(curr_in, v_in_stride, av_in9, av_in10);
        // Input point 24: x(23) & Input point 25: x(24)
        curr_in = in + in_strides[23];
        LDRI_2x512_D(curr_in, v_in_stride, av_in11, av_in12);
        // Input point 28: x(27)
        curr_in = in + in_strides[27];
        LDR_512_D(curr_in, v_in_stride, av_in13);

        av_s1 = _mm512_sub_pd(av_in0, av_in13);
        av_s2 = _mm512_add_pd(av_in0, av_in13);
        av_s3 = _mm512_sub_pd(av_in11, av_in1);
        av_s4 = _mm512_add_pd(av_in1, av_in11);
        av_s5 = _mm512_sub_pd(av_in2, av_in12);
        av_s6 = _mm512_add_pd(av_in2, av_in12);
        av_s7 = _mm512_sub_pd(av_in3, av_in9);
        av_s8 = _mm512_add_pd(av_in3, av_in9);
        av_s9 = _mm512_sub_pd(av_in4, av_in10);
        av_s10 = _mm512_add_pd(av_in4, av_in10);
        av_s11 = _mm512_sub_pd(av_in7, av_in5);
        av_s12 = _mm512_add_pd(av_in5, av_in7);
        av_s13 = _mm512_sub_pd(av_in6, av_in8);
        av_s14 = _mm512_add_pd(av_in6, av_in8);

        av_s27 = _mm512_add_pd(av_s12, av_s4);
        av_s28 = _mm512_add_pd(av_s27, av_s8);
        av_t37 = _mm512_mul_pd(v_CRTM_14_7, av_s28);
        av_s29 = _mm512_add_pd(av_s3, av_s7);
        av_s30 = _mm512_add_pd(av_s29, av_s11);
        av_t38 = _mm512_mul_pd(v_CRTM_14_7, av_s30);
        // Output pt 1: X(0)
        v_out0 = _mm512_add_pd(av_t37, av_s2);
        STR_512_D(curr_out, v_out_stride, v_out0);
        // Output pt 15: X(14)
        curr_out = out + out_strides[14];
        v_out14 = _mm512_add_pd(av_t38, av_s1);
        STR_512_D(curr_out, v_out_stride, v_out14);

        av_t1 = _mm512_mul_pd(v_CRTM_14_1, av_s6);
        av_t2 = _mm512_mul_pd(v_CRTM_14_3, av_s10);
        av_t3 = _mm512_mul_pd(v_CRTM_14_5, av_s14);
        av_t4 = _mm512_mul_pd(v_CRTM_14_2, av_s3);
        av_t5 = _mm512_mul_pd(v_CRTM_14_4, av_s7);
        av_t6 = _mm512_mul_pd(v_CRTM_14_6, av_s11);

        av_s31 = _mm512_sub_pd(av_t5, av_t6);
        av_s32 = _mm512_sub_pd(av_s1, av_t4);
        av_s33 = _mm512_add_pd(av_t1, av_t2);

        av_s15 = _mm512_add_pd(av_s31, av_s32);
        av_s16 = _mm512_add_pd(av_s33, av_t3);
        // Output point 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm512_sub_pd(av_s15, av_s16);
        STR_512_D(curr_out, v_out_stride, v_out2);
        // Output point 27: X(26)
        curr_out = out + out_strides[26];
        v_out26 = _mm512_add_pd(av_s15, av_s16);
        STR_512_D(curr_out, v_out_stride, v_out26);

        av_t7 = _mm512_mul_pd(v_CRTM_14_1, av_s13);
        av_t8 = _mm512_mul_pd(v_CRTM_14_3, av_s5);
        av_t9 = _mm512_mul_pd(v_CRTM_14_5, av_s9);

        av_t10 = _mm512_mul_pd(v_CRTM_14_2, av_s12);
        av_t11 = _mm512_mul_pd(v_CRTM_14_4, av_s4);
        av_t12 = _mm512_mul_pd(v_CRTM_14_6, av_s8);

        av_s34 = _mm512_sub_pd(av_s2, av_t10);
        av_s35 = _mm512_sub_pd(av_t11, av_t12);
        av_s36 = _mm512_add_pd(av_t7, av_t8);

        av_s17 = av_s34 + av_s35;
        av_s18 = _mm512_add_pd(av_s36, av_t9);
        // Output point 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm512_sub_pd(av_s17, av_s18);
        STR_512_D(curr_out, v_out_stride, v_out4);
        // Output point 25: X(24)
        curr_out = out + out_strides[24];
        v_out24 = _mm512_add_pd(av_s17, av_s18);
        STR_512_D(curr_out, v_out_stride, v_out24);

        av_t13 = _mm512_mul_pd(v_CRTM_14_1, av_s10);
        av_t14 = _mm512_mul_pd(v_CRTM_14_3, av_s14);
        av_t15 = _mm512_mul_pd(v_CRTM_14_5, av_s6);
        av_t16 = _mm512_mul_pd(v_CRTM_14_2, av_s7);
        av_t17 = _mm512_mul_pd(v_CRTM_14_4, av_s11);
        av_t18 = _mm512_mul_pd(v_CRTM_14_6, av_s3);

        av_s37 = _mm512_sub_pd(av_s1, av_t16);
        av_s38 = _mm512_sub_pd(av_t17, av_t18);
        av_s39 = _mm512_sub_pd(av_t14, av_t15);
        av_s19 = _mm512_add_pd(av_s37, av_s38);
        av_s20 = _mm512_sub_pd(av_s39, av_t13);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm512_add_pd(av_s19, av_s20);
        STR_512_D(curr_out, v_out_stride, v_out6);
        // Output point 23: X(22)
        curr_out = out + out_strides[22];
        v_out22 = _mm512_sub_pd(av_s19, av_s20);
        STR_512_D(curr_out, v_out_stride, v_out22);

        av_t19 = _mm512_mul_pd(v_CRTM_14_1, av_s9);
        av_t20 = _mm512_mul_pd(v_CRTM_14_3, av_s13);
        av_t21 = _mm512_mul_pd(v_CRTM_14_5, av_s5);
        av_t22 = _mm512_mul_pd(v_CRTM_14_2, av_s8);
        av_t23 = _mm512_mul_pd(v_CRTM_14_4, av_s12);
        av_t24 = _mm512_mul_pd(v_CRTM_14_6, av_s4);

        av_s40 = _mm512_sub_pd(av_s2, av_t22);
        av_s41 = _mm512_sub_pd(av_t23, av_t24);
        av_s42 = _mm512_add_pd(av_t19, av_t20);
        av_s21 = _mm512_add_pd(av_s40, av_s41);
        av_s22 = _mm512_sub_pd(av_s42, av_t21);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm512_add_pd(av_s21, av_s22);
        STR_512_D(curr_out, v_out_stride, v_out8);
        // Output point 21: X(20)
        curr_out = out + out_strides[20];
        v_out20 = _mm512_sub_pd(av_s21, av_s22);
        STR_512_D(curr_out, v_out_stride, v_out20);

        av_t25 = _mm512_mul_pd(v_CRTM_14_1, av_s14);
        av_t26 = _mm512_mul_pd(v_CRTM_14_3, av_s6);
        av_t27 = _mm512_mul_pd(v_CRTM_14_5, av_s10);
        av_t28 = _mm512_mul_pd(v_CRTM_14_2, av_s11);
        av_t29 = _mm512_mul_pd(v_CRTM_14_4, av_s3);
        av_t30 = _mm512_mul_pd(v_CRTM_14_6, av_s7);

        av_s43 = _mm512_sub_pd(av_s1, av_t28);
        av_s44 = _mm512_sub_pd(av_t29, av_t30);
        av_s45 = _mm512_sub_pd(av_t27, av_t25);
        av_s23 = _mm512_add_pd(av_s43, av_s44);
        av_s24 = _mm512_sub_pd(av_s45, av_t26);
        // Output pt 11: X(10)
        curr_out = out + out_strides[10];
        v_out10 = _mm512_add_pd(av_s23, av_s24);
        STR_512_D(curr_out, v_out_stride, v_out10);
        // Output pt 19: X(18)
        curr_out = out + out_strides[18];
        v_out18 = _mm512_sub_pd(av_s23, av_s24);
        STR_512_D(curr_out, v_out_stride, v_out18);

        av_t31 = _mm512_mul_pd(v_CRTM_14_1, av_s5);
        av_t32 = _mm512_mul_pd(v_CRTM_14_3, av_s9);
        av_t33 = _mm512_mul_pd(v_CRTM_14_5, av_s13);
        av_t34 = _mm512_mul_pd(v_CRTM_14_2, av_s4);
        av_t35 = _mm512_mul_pd(v_CRTM_14_4, av_s8);
        av_t36 = _mm512_mul_pd(v_CRTM_14_6, av_s12);

        av_s46 = _mm512_sub_pd(av_s2, av_t34);
        av_s47 = _mm512_sub_pd(av_t35, av_t36);
        av_s48 = _mm512_sub_pd(av_t32, av_t31);
        av_s25 = _mm512_add_pd(av_s46, av_s47);
        av_s26 = _mm512_sub_pd(av_s48, av_t33);
        // Output pt 13: X(12)
        curr_out = out + out_strides[12];
        v_out12 = _mm512_add_pd(av_s25, av_s26);
        STR_512_D(curr_out, v_out_stride, v_out12);
        // Output pt 17: X(16)
        curr_out = out + out_strides[16];
        v_out16 = _mm512_sub_pd(av_s25, av_s26);
        STR_512_D(curr_out, v_out_stride, v_out16);

        /* Shifted DFT */
        __m512d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
                bv_in8, bv_in9, bv_in10, bv_in11, bv_in12, bv_in13;
        __m512d bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
                bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17,
                bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24, bv_s25,
                bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32, bv_s33,
                bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40, bv_s41,
                bv_s42, bv_s43, bv_s44, bv_s45, bv_s46, bv_s47, bv_s48, bv_s49,
                bv_s50;
        __m512d bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
                bv_t10, bv_t11, bv_t12, bv_t13, bv_t14, bv_t15, bv_t16, bv_t17,
                bv_t18, bv_t19, bv_t20, bv_t21, bv_t22, bv_t23, bv_t24, bv_t25,
                bv_t26, bv_t27, bv_t28, bv_t29, bv_t30, bv_t31, bv_t32, bv_t33,
                bv_t34, bv_t35, bv_t36;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x512_D(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDRI_2x512_D(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDRI_2x512_D(curr_in, v_in_stride, bv_in4, bv_in5);
        // Input point 14: x(13) & Input point 15: x(14)
        curr_in = in + in_strides[13];
        LDRI_2x512_D(curr_in, v_in_stride, bv_in6, bv_in7);
        // Input point 18: x(17) & Input point 19: x(18)
        curr_in = in + in_strides[17];
        LDRI_2x512_D(curr_in, v_in_stride, bv_in8, bv_in9);
        // Input point 22: x(21) & Input point 23: x(22)
        curr_in = in + in_strides[21];
        LDRI_2x512_D(curr_in, v_in_stride, bv_in10, bv_in11);
        // Input point 26: x(25) & Input point 27: x(26)
        curr_in = in + in_strides[25];
        LDRI_2x512_D(curr_in, v_in_stride, bv_in12, bv_in13);

        bv_s1 = _mm512_add_pd(bv_in0, bv_in12);
        bv_s2 = _mm512_sub_pd(bv_in0, bv_in12);
        bv_s3 = _mm512_add_pd(bv_in1, bv_in13);
        bv_s4 = _mm512_sub_pd(bv_in1, bv_in13);
        bv_s5 = _mm512_add_pd(bv_in2, bv_in10);
        bv_s6 = _mm512_sub_pd(bv_in2, bv_in10);
        bv_s7 = _mm512_add_pd(bv_in3, bv_in11);
        bv_s8 = _mm512_sub_pd(bv_in3, bv_in11);
        bv_s9 = _mm512_add_pd(bv_in4, bv_in8);
        bv_s10 = _mm512_sub_pd(bv_in4, bv_in8);
        bv_s11 = _mm512_add_pd(bv_in5, bv_in9);
        bv_s12 = _mm512_sub_pd(bv_in5, bv_in9);

        bv_t1 = _mm512_mul_pd(v_CRTM_14_6, bv_s3);
        bv_t2 = _mm512_mul_pd(v_CRTM_14_2, bv_s11);
        bv_t3 = _mm512_mul_pd(v_CRTM_14_4, bv_s7);
        bv_t4 = _mm512_mul_pd(v_CRTM_14_5, bv_s2);
        bv_t5 = _mm512_mul_pd(v_CRTM_14_1, bv_s10);
        bv_t6 = _mm512_mul_pd(v_CRTM_14_3, bv_s6);

        bv_t7 = _mm512_mul_pd(v_CRTM_14_6, bv_s5);
        bv_t8 = _mm512_mul_pd(v_CRTM_14_2, bv_s1);
        bv_t9 = _mm512_mul_pd(v_CRTM_14_4, bv_s9);
        bv_t10 = _mm512_mul_pd(v_CRTM_14_5, bv_s8);
        bv_t11 = _mm512_mul_pd(v_CRTM_14_1, bv_s4);
        bv_t12 = _mm512_mul_pd(v_CRTM_14_3, bv_s12);

        bv_t13 = _mm512_mul_pd(v_CRTM_14_6, bv_s11);
        bv_t14 = _mm512_mul_pd(v_CRTM_14_2, bv_s7);
        bv_t15 = _mm512_mul_pd(v_CRTM_14_4, bv_s3);
        bv_t16 = _mm512_mul_pd(v_CRTM_14_5, bv_s10);
        bv_t17 = _mm512_mul_pd(v_CRTM_14_1, bv_s6);
        bv_t18 = _mm512_mul_pd(v_CRTM_14_3, bv_s2);

        bv_t19 = _mm512_mul_pd(v_CRTM_14_6, bv_s9);
        bv_t20 = _mm512_mul_pd(v_CRTM_14_2, bv_s5);
        bv_t21 = _mm512_mul_pd(v_CRTM_14_4, bv_s1);
        bv_t22 = _mm512_mul_pd(v_CRTM_14_5, bv_s12);
        bv_t23 = _mm512_mul_pd(v_CRTM_14_1, bv_s8);
        bv_t24 = _mm512_mul_pd(v_CRTM_14_3, bv_s4);

        bv_t25 = _mm512_mul_pd(v_CRTM_14_6, bv_s7);
        bv_t26 = _mm512_mul_pd(v_CRTM_14_2, bv_s3);
        bv_t27 = _mm512_mul_pd(v_CRTM_14_4, bv_s11);
        bv_t28 = _mm512_mul_pd(v_CRTM_14_5, bv_s6);
        bv_t29 = _mm512_mul_pd(v_CRTM_14_1, bv_s2);
        bv_t30 = _mm512_mul_pd(v_CRTM_14_3, bv_s10);

        bv_t31 = _mm512_mul_pd(v_CRTM_14_6, bv_s1);
        bv_t32 = _mm512_mul_pd(v_CRTM_14_2, bv_s9);
        bv_t33 = _mm512_mul_pd(v_CRTM_14_4, bv_s5);
        bv_t34 = _mm512_mul_pd(v_CRTM_14_5, bv_s4);
        bv_t35 = _mm512_mul_pd(v_CRTM_14_1, bv_s12);
        bv_t36 = _mm512_mul_pd(v_CRTM_14_3, bv_s8);

        bv_s13 = _mm512_add_pd(bv_in6, bv_in6);
        bv_s14 = _mm512_add_pd(bv_in7, bv_in7);

        bv_s15 = _mm512_add_pd(bv_t1, bv_t2);
        bv_s16 = _mm512_add_pd(bv_t3, bv_s14);
        bv_s17 = _mm512_add_pd(bv_s15, bv_s16);
        bv_s18 = _mm512_add_pd(bv_t4, bv_t5);
        bv_s19 = _mm512_add_pd(bv_t6, bv_s18);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm512_sub_pd(bv_s19, bv_s17);
        STR_512_D(curr_out, v_out_stride, v_out3);
        // Output pt 28: X(27)
        curr_out = out + out_strides[27];
        v_out27 = NEGATE_512_D(_mm512_add_pd(bv_s17, bv_s19));
        STR_512_D(curr_out, v_out_stride, v_out27);

        bv_s20 = _mm512_add_pd(bv_t10, bv_t11);
        bv_s21 = _mm512_add_pd(bv_t12, bv_s20);
        bv_s22 = _mm512_add_pd(bv_t7, bv_t8);
        bv_s23 = _mm512_add_pd(bv_t9, bv_s13);
        bv_s24 = _mm512_sub_pd(bv_s22, bv_s23);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm512_sub_pd(bv_s24, bv_s21);
        STR_512_D(curr_out, v_out_stride, v_out5);
        // Output pt 26: X(25)
        curr_out = out + out_strides[25];
        v_out25 = NEGATE_512_D(_mm512_add_pd(bv_s21, bv_s24));
        STR_512_D(curr_out, v_out_stride, v_out25);

        bv_s25 = _mm512_sub_pd(bv_t13, bv_t14);
        bv_s26 = _mm512_sub_pd(bv_s14, bv_t15);
        bv_s27 = _mm512_add_pd(bv_s25, bv_s26);
        bv_s28 = _mm512_add_pd(bv_t16, bv_t17);
        bv_s29 = _mm512_sub_pd(bv_t18, bv_s28);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = _mm512_add_pd(bv_s27, bv_s29);
        STR_512_D(curr_out, v_out_stride, v_out7);
        // Output pt 24: X(23)
        curr_out = out + out_strides[23];
        v_out23 = _mm512_sub_pd(bv_s27, bv_s29);
        STR_512_D(curr_out, v_out_stride, v_out23);

        bv_s30 = _mm512_sub_pd(bv_t22, bv_t23);
        bv_s31 = _mm512_sub_pd(bv_s30, bv_t24);
        bv_s32 = _mm512_add_pd(bv_t19, bv_t20);
        bv_s33 = _mm512_add_pd(bv_t21, bv_s13);
        bv_s34 = _mm512_sub_pd(bv_s33, bv_s32);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm512_add_pd(bv_s31, bv_s34);
        STR_512_D(curr_out, v_out_stride, v_out9);
        // Output pt 22: X(21)
        curr_out = out + out_strides[21];
        v_out21 = _mm512_sub_pd(bv_s31, bv_s34);
        STR_512_D(curr_out, v_out_stride, v_out21);

        bv_s35 = _mm512_sub_pd(bv_t25, bv_t26);
        bv_s36 = _mm512_sub_pd(bv_t27, bv_s14);
        bv_s37 = _mm512_add_pd(bv_s35, bv_s36);
        bv_s38 = _mm512_sub_pd(bv_t29, bv_t28);
        bv_s39 = _mm512_add_pd(bv_t30, bv_s38);
        // Output pt 12: X(11)
        curr_out = out + out_strides[11];
        v_out11 = _mm512_add_pd(bv_s37, bv_s39);
        STR_512_D(curr_out, v_out_stride, v_out11);
        // Output pt 20: X(19)
        curr_out = out + out_strides[19];
        v_out19 = _mm512_sub_pd(bv_s37, bv_s39);
        STR_512_D(curr_out, v_out_stride, v_out19);

        bv_s40 = _mm512_add_pd(bv_t34, bv_t35);
        bv_s41 = _mm512_sub_pd(bv_t36, bv_s40);
        bv_s42 = _mm512_add_pd(bv_t31, bv_t32);
        bv_s43 = _mm512_add_pd(bv_t33, bv_s13);
        bv_s44 = _mm512_sub_pd(bv_s42, bv_s43);
        // Output pt 14: X(13)
        curr_out = out + out_strides[13];
        v_out13 = _mm512_add_pd(bv_s41, bv_s44);
        STR_512_D(curr_out, v_out_stride, v_out13);
        // Output pt 18: X(17)
        curr_out = out + out_strides[17];
        v_out17 = _mm512_sub_pd(bv_s41, bv_s44);
        STR_512_D(curr_out, v_out_stride, v_out17);

        bv_s45 = _mm512_add_pd(bv_s1, bv_s5);
        bv_s46 = _mm512_add_pd(bv_in6, bv_s9);
        bv_s47 = _mm512_add_pd(bv_s45, bv_s46);
        // Output pt 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm512_mul_pd(bv_s47, v_CRTM_14_7);
        STR_512_D(curr_out, v_out_stride, v_out1);

        bv_s48 = _mm512_add_pd(bv_s3, bv_s11);
        bv_s49 = _mm512_add_pd(bv_in7, bv_s7);
        bv_s50 = _mm512_sub_pd(bv_s49, bv_s48);
        // Output pt 16: X(15)
        curr_out = out + out_strides[15];
        v_out15 = _mm512_mul_pd(bv_s50, v_CRTM_14_7);
        STR_512_D(curr_out, v_out_stride, v_out15);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_256_D)
    {
        /* Standard DFT */
        __m256d av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
                av_in8, av_in9, av_in10, av_in11, av_in12, av_in13;
        __m256d av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8, av_s9,
                av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16, av_s17,
                av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24, av_s25,
                av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32, av_s33,
                av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40, av_s41,
                av_s42, av_s43, av_s44, av_s45, av_s46, av_s47, av_s48;
        __m256d av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8, av_t9,
                av_t10, av_t11, av_t12, av_t13, av_t14, av_t15, av_t16, av_t17,
                av_t18, av_t19, av_t20, av_t21, av_t22, av_t23, av_t24, av_t25,
                av_t26, av_t27, av_t28, av_t29, av_t30, av_t31, av_t32, av_t33,
                av_t34, av_t35, av_t36, av_t37, av_t38;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
                v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21,
                v_out22, v_out23, v_out24, v_out25, v_out26, v_out27;

        curr_in = in;
        curr_out = out;

        __m256d v256_CRTM_14_1 = _mm512_castpd512_pd256(v_CRTM_14_1);
        __m256d v256_CRTM_14_2 = _mm512_castpd512_pd256(v_CRTM_14_2);
        __m256d v256_CRTM_14_3 = _mm512_castpd512_pd256(v_CRTM_14_3);
        __m256d v256_CRTM_14_4 = _mm512_castpd512_pd256(v_CRTM_14_4);
        __m256d v256_CRTM_14_5 = _mm512_castpd512_pd256(v_CRTM_14_5);
        __m256d v256_CRTM_14_6 = _mm512_castpd512_pd256(v_CRTM_14_6);
        __m256d v256_CRTM_14_7 = _mm512_castpd512_pd256(v_CRTM_14_7);

        // Input point 1: x(0)
        LDR_256_D(curr_in, v_in_stride, av_in0);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x256_D(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in + in_strides[7];
        LDRI_2x256_D(curr_in, v_in_stride, av_in3, av_in4);
        // Input point 12: x(11) & Input point 13: x(12)
        curr_in = in + in_strides[11];
        LDRI_2x256_D(curr_in, v_in_stride, av_in5, av_in6);
        // Input point 16: x(15) & Input point 17: x(16)
        curr_in = in + in_strides[15];
        LDRI_2x256_D(curr_in, v_in_stride, av_in7, av_in8);
        // Input point 20: x(19) & Input point 21: x(20)
        curr_in = in + in_strides[19];
        LDRI_2x256_D(curr_in, v_in_stride, av_in9, av_in10);
        // Input point 24: x(23) & Input point 25: x(24)
        curr_in = in + in_strides[23];
        LDRI_2x256_D(curr_in, v_in_stride, av_in11, av_in12);
        // Input point 28: x(27)
        curr_in = in + in_strides[27];
        LDR_256_D(curr_in, v_in_stride, av_in13);

        av_s1 = _mm256_sub_pd(av_in0, av_in13);
        av_s2 = _mm256_add_pd(av_in0, av_in13);
        av_s3 = _mm256_sub_pd(av_in11, av_in1);
        av_s4 = _mm256_add_pd(av_in1, av_in11);
        av_s5 = _mm256_sub_pd(av_in2, av_in12);
        av_s6 = _mm256_add_pd(av_in2, av_in12);
        av_s7 = _mm256_sub_pd(av_in3, av_in9);
        av_s8 = _mm256_add_pd(av_in3, av_in9);
        av_s9 = _mm256_sub_pd(av_in4, av_in10);
        av_s10 = _mm256_add_pd(av_in4, av_in10);
        av_s11 = _mm256_sub_pd(av_in7, av_in5);
        av_s12 = _mm256_add_pd(av_in5, av_in7);
        av_s13 = _mm256_sub_pd(av_in6, av_in8);
        av_s14 = _mm256_add_pd(av_in6, av_in8);

        av_s27 = _mm256_add_pd(av_s12, av_s4);
        av_s28 = _mm256_add_pd(av_s27, av_s8);
        av_t37 = _mm256_mul_pd(v256_CRTM_14_7, av_s28);
        av_s29 = _mm256_add_pd(av_s3, av_s7);
        av_s30 = _mm256_add_pd(av_s29, av_s11);
        av_t38 = _mm256_mul_pd(v256_CRTM_14_7, av_s30);
        // Output pt 1: X(0)
        v_out0 = _mm256_add_pd(av_t37, av_s2);
        STR_256_D(curr_out, v_out_stride, v_out0);
        // Output pt 15: X(14)
        curr_out = out + out_strides[14];
        v_out14 = _mm256_add_pd(av_t38, av_s1);
        STR_256_D(curr_out, v_out_stride, v_out14);

        av_t1 = _mm256_mul_pd(v256_CRTM_14_1, av_s6);
        av_t2 = _mm256_mul_pd(v256_CRTM_14_3, av_s10);
        av_t3 = _mm256_mul_pd(v256_CRTM_14_5, av_s14);
        av_t4 = _mm256_mul_pd(v256_CRTM_14_2, av_s3);
        av_t5 = _mm256_mul_pd(v256_CRTM_14_4, av_s7);
        av_t6 = _mm256_mul_pd(v256_CRTM_14_6, av_s11);

        av_s31 = _mm256_sub_pd(av_t5, av_t6);
        av_s32 = _mm256_sub_pd(av_s1, av_t4);
        av_s33 = _mm256_add_pd(av_t1, av_t2);

        av_s15 = _mm256_add_pd(av_s31, av_s32);
        av_s16 = _mm256_add_pd(av_s33, av_t3);
        // Output point 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm256_sub_pd(av_s15, av_s16);
        STR_256_D(curr_out, v_out_stride, v_out2);
        // Output point 27: X(26)
        curr_out = out + out_strides[26];
        v_out26 = _mm256_add_pd(av_s15, av_s16);
        STR_256_D(curr_out, v_out_stride, v_out26);

        av_t7 = _mm256_mul_pd(v256_CRTM_14_1, av_s13);
        av_t8 = _mm256_mul_pd(v256_CRTM_14_3, av_s5);
        av_t9 = _mm256_mul_pd(v256_CRTM_14_5, av_s9);

        av_t10 = _mm256_mul_pd(v256_CRTM_14_2, av_s12);
        av_t11 = _mm256_mul_pd(v256_CRTM_14_4, av_s4);
        av_t12 = _mm256_mul_pd(v256_CRTM_14_6, av_s8);

        av_s34 = _mm256_sub_pd(av_s2, av_t10);
        av_s35 = _mm256_sub_pd(av_t11, av_t12);
        av_s36 = _mm256_add_pd(av_t7, av_t8);

        av_s17 = av_s34 + av_s35;
        av_s18 = _mm256_add_pd(av_s36, av_t9);
        // Output point 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm256_sub_pd(av_s17, av_s18);
        STR_256_D(curr_out, v_out_stride, v_out4);
        // Output point 25: X(24)
        curr_out = out + out_strides[24];
        v_out24 = _mm256_add_pd(av_s17, av_s18);
        STR_256_D(curr_out, v_out_stride, v_out24);

        av_t13 = _mm256_mul_pd(v256_CRTM_14_1, av_s10);
        av_t14 = _mm256_mul_pd(v256_CRTM_14_3, av_s14);
        av_t15 = _mm256_mul_pd(v256_CRTM_14_5, av_s6);
        av_t16 = _mm256_mul_pd(v256_CRTM_14_2, av_s7);
        av_t17 = _mm256_mul_pd(v256_CRTM_14_4, av_s11);
        av_t18 = _mm256_mul_pd(v256_CRTM_14_6, av_s3);

        av_s37 = _mm256_sub_pd(av_s1, av_t16);
        av_s38 = _mm256_sub_pd(av_t17, av_t18);
        av_s39 = _mm256_sub_pd(av_t14, av_t15);
        av_s19 = _mm256_add_pd(av_s37, av_s38);
        av_s20 = _mm256_sub_pd(av_s39, av_t13);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm256_add_pd(av_s19, av_s20);
        STR_256_D(curr_out, v_out_stride, v_out6);
        // Output point 23: X(22)
        curr_out = out + out_strides[22];
        v_out22 = _mm256_sub_pd(av_s19, av_s20);
        STR_256_D(curr_out, v_out_stride, v_out22);

        av_t19 = _mm256_mul_pd(v256_CRTM_14_1, av_s9);
        av_t20 = _mm256_mul_pd(v256_CRTM_14_3, av_s13);
        av_t21 = _mm256_mul_pd(v256_CRTM_14_5, av_s5);
        av_t22 = _mm256_mul_pd(v256_CRTM_14_2, av_s8);
        av_t23 = _mm256_mul_pd(v256_CRTM_14_4, av_s12);
        av_t24 = _mm256_mul_pd(v256_CRTM_14_6, av_s4);

        av_s40 = _mm256_sub_pd(av_s2, av_t22);
        av_s41 = _mm256_sub_pd(av_t23, av_t24);
        av_s42 = _mm256_add_pd(av_t19, av_t20);
        av_s21 = _mm256_add_pd(av_s40, av_s41);
        av_s22 = _mm256_sub_pd(av_s42, av_t21);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm256_add_pd(av_s21, av_s22);
        STR_256_D(curr_out, v_out_stride, v_out8);
        // Output point 21: X(20)
        curr_out = out + out_strides[20];
        v_out20 = _mm256_sub_pd(av_s21, av_s22);
        STR_256_D(curr_out, v_out_stride, v_out20);

        av_t25 = _mm256_mul_pd(v256_CRTM_14_1, av_s14);
        av_t26 = _mm256_mul_pd(v256_CRTM_14_3, av_s6);
        av_t27 = _mm256_mul_pd(v256_CRTM_14_5, av_s10);
        av_t28 = _mm256_mul_pd(v256_CRTM_14_2, av_s11);
        av_t29 = _mm256_mul_pd(v256_CRTM_14_4, av_s3);
        av_t30 = _mm256_mul_pd(v256_CRTM_14_6, av_s7);

        av_s43 = _mm256_sub_pd(av_s1, av_t28);
        av_s44 = _mm256_sub_pd(av_t29, av_t30);
        av_s45 = _mm256_sub_pd(av_t27, av_t25);
        av_s23 = _mm256_add_pd(av_s43, av_s44);
        av_s24 = _mm256_sub_pd(av_s45, av_t26);
        // Output pt 11: X(10)
        curr_out = out + out_strides[10];
        v_out10 = _mm256_add_pd(av_s23, av_s24);
        STR_256_D(curr_out, v_out_stride, v_out10);
        // Output pt 19: X(18)
        curr_out = out + out_strides[18];
        v_out18 = _mm256_sub_pd(av_s23, av_s24);
        STR_256_D(curr_out, v_out_stride, v_out18);

        av_t31 = _mm256_mul_pd(v256_CRTM_14_1, av_s5);
        av_t32 = _mm256_mul_pd(v256_CRTM_14_3, av_s9);
        av_t33 = _mm256_mul_pd(v256_CRTM_14_5, av_s13);
        av_t34 = _mm256_mul_pd(v256_CRTM_14_2, av_s4);
        av_t35 = _mm256_mul_pd(v256_CRTM_14_4, av_s8);
        av_t36 = _mm256_mul_pd(v256_CRTM_14_6, av_s12);

        av_s46 = _mm256_sub_pd(av_s2, av_t34);
        av_s47 = _mm256_sub_pd(av_t35, av_t36);
        av_s48 = _mm256_sub_pd(av_t32, av_t31);
        av_s25 = _mm256_add_pd(av_s46, av_s47);
        av_s26 = _mm256_sub_pd(av_s48, av_t33);
        // Output pt 13: X(12)
        curr_out = out + out_strides[12];
        v_out12 = _mm256_add_pd(av_s25, av_s26);
        STR_256_D(curr_out, v_out_stride, v_out12);
        // Output pt 17: X(16)
        curr_out = out + out_strides[16];
        v_out16 = _mm256_sub_pd(av_s25, av_s26);
        STR_256_D(curr_out, v_out_stride, v_out16);

        /* Shifted DFT */
        __m256d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
                bv_in8, bv_in9, bv_in10, bv_in11, bv_in12, bv_in13;
        __m256d bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
                bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17,
                bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24, bv_s25,
                bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32, bv_s33,
                bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40, bv_s41,
                bv_s42, bv_s43, bv_s44, bv_s45, bv_s46, bv_s47, bv_s48, bv_s49,
                bv_s50;
        __m256d bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
                bv_t10, bv_t11, bv_t12, bv_t13, bv_t14, bv_t15, bv_t16, bv_t17,
                bv_t18, bv_t19, bv_t20, bv_t21, bv_t22, bv_t23, bv_t24, bv_t25,
                bv_t26, bv_t27, bv_t28, bv_t29, bv_t30, bv_t31, bv_t32, bv_t33,
                bv_t34, bv_t35, bv_t36;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x256_D(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDRI_2x256_D(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDRI_2x256_D(curr_in, v_in_stride, bv_in4, bv_in5);
        // Input point 14: x(13) & Input point 15: x(14)
        curr_in = in + in_strides[13];
        LDRI_2x256_D(curr_in, v_in_stride, bv_in6, bv_in7);
        // Input point 18: x(17) & Input point 19: x(18)
        curr_in = in + in_strides[17];
        LDRI_2x256_D(curr_in, v_in_stride, bv_in8, bv_in9);
        // Input point 22: x(21) & Input point 23: x(22)
        curr_in = in + in_strides[21];
        LDRI_2x256_D(curr_in, v_in_stride, bv_in10, bv_in11);
        // Input point 26: x(25) & Input point 27: x(26)
        curr_in = in + in_strides[25];
        LDRI_2x256_D(curr_in, v_in_stride, bv_in12, bv_in13);

        bv_s1 = _mm256_add_pd(bv_in0, bv_in12);
        bv_s2 = _mm256_sub_pd(bv_in0, bv_in12);
        bv_s3 = _mm256_add_pd(bv_in1, bv_in13);
        bv_s4 = _mm256_sub_pd(bv_in1, bv_in13);
        bv_s5 = _mm256_add_pd(bv_in2, bv_in10);
        bv_s6 = _mm256_sub_pd(bv_in2, bv_in10);
        bv_s7 = _mm256_add_pd(bv_in3, bv_in11);
        bv_s8 = _mm256_sub_pd(bv_in3, bv_in11);
        bv_s9 = _mm256_add_pd(bv_in4, bv_in8);
        bv_s10 = _mm256_sub_pd(bv_in4, bv_in8);
        bv_s11 = _mm256_add_pd(bv_in5, bv_in9);
        bv_s12 = _mm256_sub_pd(bv_in5, bv_in9);

        bv_t1 = _mm256_mul_pd(v256_CRTM_14_6, bv_s3);
        bv_t2 = _mm256_mul_pd(v256_CRTM_14_2, bv_s11);
        bv_t3 = _mm256_mul_pd(v256_CRTM_14_4, bv_s7);
        bv_t4 = _mm256_mul_pd(v256_CRTM_14_5, bv_s2);
        bv_t5 = _mm256_mul_pd(v256_CRTM_14_1, bv_s10);
        bv_t6 = _mm256_mul_pd(v256_CRTM_14_3, bv_s6);

        bv_t7 = _mm256_mul_pd(v256_CRTM_14_6, bv_s5);
        bv_t8 = _mm256_mul_pd(v256_CRTM_14_2, bv_s1);
        bv_t9 = _mm256_mul_pd(v256_CRTM_14_4, bv_s9);
        bv_t10 = _mm256_mul_pd(v256_CRTM_14_5, bv_s8);
        bv_t11 = _mm256_mul_pd(v256_CRTM_14_1, bv_s4);
        bv_t12 = _mm256_mul_pd(v256_CRTM_14_3, bv_s12);

        bv_t13 = _mm256_mul_pd(v256_CRTM_14_6, bv_s11);
        bv_t14 = _mm256_mul_pd(v256_CRTM_14_2, bv_s7);
        bv_t15 = _mm256_mul_pd(v256_CRTM_14_4, bv_s3);
        bv_t16 = _mm256_mul_pd(v256_CRTM_14_5, bv_s10);
        bv_t17 = _mm256_mul_pd(v256_CRTM_14_1, bv_s6);
        bv_t18 = _mm256_mul_pd(v256_CRTM_14_3, bv_s2);

        bv_t19 = _mm256_mul_pd(v256_CRTM_14_6, bv_s9);
        bv_t20 = _mm256_mul_pd(v256_CRTM_14_2, bv_s5);
        bv_t21 = _mm256_mul_pd(v256_CRTM_14_4, bv_s1);
        bv_t22 = _mm256_mul_pd(v256_CRTM_14_5, bv_s12);
        bv_t23 = _mm256_mul_pd(v256_CRTM_14_1, bv_s8);
        bv_t24 = _mm256_mul_pd(v256_CRTM_14_3, bv_s4);

        bv_t25 = _mm256_mul_pd(v256_CRTM_14_6, bv_s7);
        bv_t26 = _mm256_mul_pd(v256_CRTM_14_2, bv_s3);
        bv_t27 = _mm256_mul_pd(v256_CRTM_14_4, bv_s11);
        bv_t28 = _mm256_mul_pd(v256_CRTM_14_5, bv_s6);
        bv_t29 = _mm256_mul_pd(v256_CRTM_14_1, bv_s2);
        bv_t30 = _mm256_mul_pd(v256_CRTM_14_3, bv_s10);

        bv_t31 = _mm256_mul_pd(v256_CRTM_14_6, bv_s1);
        bv_t32 = _mm256_mul_pd(v256_CRTM_14_2, bv_s9);
        bv_t33 = _mm256_mul_pd(v256_CRTM_14_4, bv_s5);
        bv_t34 = _mm256_mul_pd(v256_CRTM_14_5, bv_s4);
        bv_t35 = _mm256_mul_pd(v256_CRTM_14_1, bv_s12);
        bv_t36 = _mm256_mul_pd(v256_CRTM_14_3, bv_s8);

        bv_s13 = _mm256_add_pd(bv_in6, bv_in6);
        bv_s14 = _mm256_add_pd(bv_in7, bv_in7);

        bv_s15 = _mm256_add_pd(bv_t1, bv_t2);
        bv_s16 = _mm256_add_pd(bv_t3, bv_s14);
        bv_s17 = _mm256_add_pd(bv_s15, bv_s16);
        bv_s18 = _mm256_add_pd(bv_t4, bv_t5);
        bv_s19 = _mm256_add_pd(bv_t6, bv_s18);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm256_sub_pd(bv_s19, bv_s17);
        STR_256_D(curr_out, v_out_stride, v_out3);
        // Output pt 28: X(27)
        curr_out = out + out_strides[27];
        v_out27 = NEGATE_256_D(_mm256_add_pd(bv_s17, bv_s19));
        STR_256_D(curr_out, v_out_stride, v_out27);

        bv_s20 = _mm256_add_pd(bv_t10, bv_t11);
        bv_s21 = _mm256_add_pd(bv_t12, bv_s20);
        bv_s22 = _mm256_add_pd(bv_t7, bv_t8);
        bv_s23 = _mm256_add_pd(bv_t9, bv_s13);
        bv_s24 = _mm256_sub_pd(bv_s22, bv_s23);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm256_sub_pd(bv_s24, bv_s21);
        STR_256_D(curr_out, v_out_stride, v_out5);
        // Output pt 26: X(25)
        curr_out = out + out_strides[25];
        v_out25 = NEGATE_256_D(_mm256_add_pd(bv_s21, bv_s24));
        STR_256_D(curr_out, v_out_stride, v_out25);

        bv_s25 = _mm256_sub_pd(bv_t13, bv_t14);
        bv_s26 = _mm256_sub_pd(bv_s14, bv_t15);
        bv_s27 = _mm256_add_pd(bv_s25, bv_s26);
        bv_s28 = _mm256_add_pd(bv_t16, bv_t17);
        bv_s29 = _mm256_sub_pd(bv_t18, bv_s28);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = _mm256_add_pd(bv_s27, bv_s29);
        STR_256_D(curr_out, v_out_stride, v_out7);
        // Output pt 24: X(23)
        curr_out = out + out_strides[23];
        v_out23 = _mm256_sub_pd(bv_s27, bv_s29);
        STR_256_D(curr_out, v_out_stride, v_out23);

        bv_s30 = _mm256_sub_pd(bv_t22, bv_t23);
        bv_s31 = _mm256_sub_pd(bv_s30, bv_t24);
        bv_s32 = _mm256_add_pd(bv_t19, bv_t20);
        bv_s33 = _mm256_add_pd(bv_t21, bv_s13);
        bv_s34 = _mm256_sub_pd(bv_s33, bv_s32);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm256_add_pd(bv_s31, bv_s34);
        STR_256_D(curr_out, v_out_stride, v_out9);
        // Output pt 22: X(21)
        curr_out = out + out_strides[21];
        v_out21 = _mm256_sub_pd(bv_s31, bv_s34);
        STR_256_D(curr_out, v_out_stride, v_out21);

        bv_s35 = _mm256_sub_pd(bv_t25, bv_t26);
        bv_s36 = _mm256_sub_pd(bv_t27, bv_s14);
        bv_s37 = _mm256_add_pd(bv_s35, bv_s36);
        bv_s38 = _mm256_sub_pd(bv_t29, bv_t28);
        bv_s39 = _mm256_add_pd(bv_t30, bv_s38);
        // Output pt 12: X(11)
        curr_out = out + out_strides[11];
        v_out11 = _mm256_add_pd(bv_s37, bv_s39);
        STR_256_D(curr_out, v_out_stride, v_out11);
        // Output pt 20: X(19)
        curr_out = out + out_strides[19];
        v_out19 = _mm256_sub_pd(bv_s37, bv_s39);
        STR_256_D(curr_out, v_out_stride, v_out19);

        bv_s40 = _mm256_add_pd(bv_t34, bv_t35);
        bv_s41 = _mm256_sub_pd(bv_t36, bv_s40);
        bv_s42 = _mm256_add_pd(bv_t31, bv_t32);
        bv_s43 = _mm256_add_pd(bv_t33, bv_s13);
        bv_s44 = _mm256_sub_pd(bv_s42, bv_s43);
        // Output pt 14: X(13)
        curr_out = out + out_strides[13];
        v_out13 = _mm256_add_pd(bv_s41, bv_s44);
        STR_256_D(curr_out, v_out_stride, v_out13);
        // Output pt 18: X(17)
        curr_out = out + out_strides[17];
        v_out17 = _mm256_sub_pd(bv_s41, bv_s44);
        STR_256_D(curr_out, v_out_stride, v_out17);

        bv_s45 = _mm256_add_pd(bv_s1, bv_s5);
        bv_s46 = _mm256_add_pd(bv_in6, bv_s9);
        bv_s47 = _mm256_add_pd(bv_s45, bv_s46);
        // Output pt 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm256_mul_pd(bv_s47, v256_CRTM_14_7);
        STR_256_D(curr_out, v_out_stride, v_out1);

        bv_s48 = _mm256_add_pd(bv_s3, bv_s11);
        bv_s49 = _mm256_add_pd(bv_in7, bv_s7);
        bv_s50 = _mm256_sub_pd(bv_s49, bv_s48);
        // Output pt 16: X(15)
        curr_out = out + out_strides[15];
        v_out15 = _mm256_mul_pd(bv_s50, v256_CRTM_14_7);
        STR_256_D(curr_out, v_out_stride, v_out15);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_D)
    {
        /* Standard DFT */
        __m128d av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
                av_in8, av_in9, av_in10, av_in11, av_in12, av_in13;
        __m128d av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8, av_s9,
                av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16, av_s17,
                av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24, av_s25,
                av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32, av_s33,
                av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40, av_s41,
                av_s42, av_s43, av_s44, av_s45, av_s46, av_s47, av_s48;
        __m128d av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8, av_t9,
                av_t10, av_t11, av_t12, av_t13, av_t14, av_t15, av_t16, av_t17,
                av_t18, av_t19, av_t20, av_t21, av_t22, av_t23, av_t24, av_t25,
                av_t26, av_t27, av_t28, av_t29, av_t30, av_t31, av_t32, av_t33,
                av_t34, av_t35, av_t36, av_t37, av_t38;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
                v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21,
                v_out22, v_out23, v_out24, v_out25, v_out26, v_out27;

        curr_in = in;
        curr_out = out;

        __m128d v128_CRTM_14_1 = _mm512_castpd512_pd128(v_CRTM_14_1);
        __m128d v128_CRTM_14_2 = _mm512_castpd512_pd128(v_CRTM_14_2);
        __m128d v128_CRTM_14_3 = _mm512_castpd512_pd128(v_CRTM_14_3);
        __m128d v128_CRTM_14_4 = _mm512_castpd512_pd128(v_CRTM_14_4);
        __m128d v128_CRTM_14_5 = _mm512_castpd512_pd128(v_CRTM_14_5);
        __m128d v128_CRTM_14_6 = _mm512_castpd512_pd128(v_CRTM_14_6);
        __m128d v128_CRTM_14_7 = _mm512_castpd512_pd128(v_CRTM_14_7);

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_stride, av_in0);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x128_D(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in + in_strides[7];
        LDRI_2x128_D(curr_in, v_in_stride, av_in3, av_in4);
        // Input point 12: x(11) & Input point 13: x(12)
        curr_in = in + in_strides[11];
        LDRI_2x128_D(curr_in, v_in_stride, av_in5, av_in6);
        // Input point 16: x(15) & Input point 17: x(16)
        curr_in = in + in_strides[15];
        LDRI_2x128_D(curr_in, v_in_stride, av_in7, av_in8);
        // Input point 20: x(19) & Input point 21: x(20)
        curr_in = in + in_strides[19];
        LDRI_2x128_D(curr_in, v_in_stride, av_in9, av_in10);
        // Input point 24: x(23) & Input point 25: x(24)
        curr_in = in + in_strides[23];
        LDRI_2x128_D(curr_in, v_in_stride, av_in11, av_in12);
        // Input point 28: x(27)
        curr_in = in + in_strides[27];
        LDR_128_D(curr_in, v_in_stride, av_in13);

        av_s1 = _mm_sub_pd(av_in0, av_in13);
        av_s2 = _mm_add_pd(av_in0, av_in13);
        av_s3 = _mm_sub_pd(av_in11, av_in1);
        av_s4 = _mm_add_pd(av_in1, av_in11);
        av_s5 = _mm_sub_pd(av_in2, av_in12);
        av_s6 = _mm_add_pd(av_in2, av_in12);
        av_s7 = _mm_sub_pd(av_in3, av_in9);
        av_s8 = _mm_add_pd(av_in3, av_in9);
        av_s9 = _mm_sub_pd(av_in4, av_in10);
        av_s10 = _mm_add_pd(av_in4, av_in10);
        av_s11 = _mm_sub_pd(av_in7, av_in5);
        av_s12 = _mm_add_pd(av_in5, av_in7);
        av_s13 = _mm_sub_pd(av_in6, av_in8);
        av_s14 = _mm_add_pd(av_in6, av_in8);

        av_s27 = _mm_add_pd(av_s12, av_s4);
        av_s28 = _mm_add_pd(av_s27, av_s8);
        av_t37 = _mm_mul_pd(v128_CRTM_14_7, av_s28);
        av_s29 = _mm_add_pd(av_s3, av_s7);
        av_s30 = _mm_add_pd(av_s29, av_s11);
        av_t38 = _mm_mul_pd(v128_CRTM_14_7, av_s30);
        // Output pt 1: X(0)
        v_out0 = _mm_add_pd(av_t37, av_s2);
        STR_128_D(curr_out, v_out_stride, v_out0);
        // Output pt 15: X(14)
        curr_out = out + out_strides[14];
        v_out14 = _mm_add_pd(av_t38, av_s1);
        STR_128_D(curr_out, v_out_stride, v_out14);

        av_t1 = _mm_mul_pd(v128_CRTM_14_1, av_s6);
        av_t2 = _mm_mul_pd(v128_CRTM_14_3, av_s10);
        av_t3 = _mm_mul_pd(v128_CRTM_14_5, av_s14);
        av_t4 = _mm_mul_pd(v128_CRTM_14_2, av_s3);
        av_t5 = _mm_mul_pd(v128_CRTM_14_4, av_s7);
        av_t6 = _mm_mul_pd(v128_CRTM_14_6, av_s11);

        av_s31 = _mm_sub_pd(av_t5, av_t6);
        av_s32 = _mm_sub_pd(av_s1, av_t4);
        av_s33 = _mm_add_pd(av_t1, av_t2);

        av_s15 = _mm_add_pd(av_s31, av_s32);
        av_s16 = _mm_add_pd(av_s33, av_t3);
        // Output point 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm_sub_pd(av_s15, av_s16);
        STR_128_D(curr_out, v_out_stride, v_out2);
        // Output point 27: X(26)
        curr_out = out + out_strides[26];
        v_out26 = _mm_add_pd(av_s15, av_s16);
        STR_128_D(curr_out, v_out_stride, v_out26);

        av_t7 = _mm_mul_pd(v128_CRTM_14_1, av_s13);
        av_t8 = _mm_mul_pd(v128_CRTM_14_3, av_s5);
        av_t9 = _mm_mul_pd(v128_CRTM_14_5, av_s9);

        av_t10 = _mm_mul_pd(v128_CRTM_14_2, av_s12);
        av_t11 = _mm_mul_pd(v128_CRTM_14_4, av_s4);
        av_t12 = _mm_mul_pd(v128_CRTM_14_6, av_s8);

        av_s34 = _mm_sub_pd(av_s2, av_t10);
        av_s35 = _mm_sub_pd(av_t11, av_t12);
        av_s36 = _mm_add_pd(av_t7, av_t8);

        av_s17 = av_s34 + av_s35;
        av_s18 = _mm_add_pd(av_s36, av_t9);
        // Output point 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm_sub_pd(av_s17, av_s18);
        STR_128_D(curr_out, v_out_stride, v_out4);
        // Output point 25: X(24)
        curr_out = out + out_strides[24];
        v_out24 = _mm_add_pd(av_s17, av_s18);
        STR_128_D(curr_out, v_out_stride, v_out24);

        av_t13 = _mm_mul_pd(v128_CRTM_14_1, av_s10);
        av_t14 = _mm_mul_pd(v128_CRTM_14_3, av_s14);
        av_t15 = _mm_mul_pd(v128_CRTM_14_5, av_s6);
        av_t16 = _mm_mul_pd(v128_CRTM_14_2, av_s7);
        av_t17 = _mm_mul_pd(v128_CRTM_14_4, av_s11);
        av_t18 = _mm_mul_pd(v128_CRTM_14_6, av_s3);

        av_s37 = _mm_sub_pd(av_s1, av_t16);
        av_s38 = _mm_sub_pd(av_t17, av_t18);
        av_s39 = _mm_sub_pd(av_t14, av_t15);
        av_s19 = _mm_add_pd(av_s37, av_s38);
        av_s20 = _mm_sub_pd(av_s39, av_t13);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm_add_pd(av_s19, av_s20);
        STR_128_D(curr_out, v_out_stride, v_out6);
        // Output point 23: X(22)
        curr_out = out + out_strides[22];
        v_out22 = _mm_sub_pd(av_s19, av_s20);
        STR_128_D(curr_out, v_out_stride, v_out22);

        av_t19 = _mm_mul_pd(v128_CRTM_14_1, av_s9);
        av_t20 = _mm_mul_pd(v128_CRTM_14_3, av_s13);
        av_t21 = _mm_mul_pd(v128_CRTM_14_5, av_s5);
        av_t22 = _mm_mul_pd(v128_CRTM_14_2, av_s8);
        av_t23 = _mm_mul_pd(v128_CRTM_14_4, av_s12);
        av_t24 = _mm_mul_pd(v128_CRTM_14_6, av_s4);

        av_s40 = _mm_sub_pd(av_s2, av_t22);
        av_s41 = _mm_sub_pd(av_t23, av_t24);
        av_s42 = _mm_add_pd(av_t19, av_t20);
        av_s21 = _mm_add_pd(av_s40, av_s41);
        av_s22 = _mm_sub_pd(av_s42, av_t21);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm_add_pd(av_s21, av_s22);
        STR_128_D(curr_out, v_out_stride, v_out8);
        // Output point 21: X(20)
        curr_out = out + out_strides[20];
        v_out20 = _mm_sub_pd(av_s21, av_s22);
        STR_128_D(curr_out, v_out_stride, v_out20);

        av_t25 = _mm_mul_pd(v128_CRTM_14_1, av_s14);
        av_t26 = _mm_mul_pd(v128_CRTM_14_3, av_s6);
        av_t27 = _mm_mul_pd(v128_CRTM_14_5, av_s10);
        av_t28 = _mm_mul_pd(v128_CRTM_14_2, av_s11);
        av_t29 = _mm_mul_pd(v128_CRTM_14_4, av_s3);
        av_t30 = _mm_mul_pd(v128_CRTM_14_6, av_s7);

        av_s43 = _mm_sub_pd(av_s1, av_t28);
        av_s44 = _mm_sub_pd(av_t29, av_t30);
        av_s45 = _mm_sub_pd(av_t27, av_t25);
        av_s23 = _mm_add_pd(av_s43, av_s44);
        av_s24 = _mm_sub_pd(av_s45, av_t26);
        // Output pt 11: X(10)
        curr_out = out + out_strides[10];
        v_out10 = _mm_add_pd(av_s23, av_s24);
        STR_128_D(curr_out, v_out_stride, v_out10);
        // Output pt 19: X(18)
        curr_out = out + out_strides[18];
        v_out18 = _mm_sub_pd(av_s23, av_s24);
        STR_128_D(curr_out, v_out_stride, v_out18);

        av_t31 = _mm_mul_pd(v128_CRTM_14_1, av_s5);
        av_t32 = _mm_mul_pd(v128_CRTM_14_3, av_s9);
        av_t33 = _mm_mul_pd(v128_CRTM_14_5, av_s13);
        av_t34 = _mm_mul_pd(v128_CRTM_14_2, av_s4);
        av_t35 = _mm_mul_pd(v128_CRTM_14_4, av_s8);
        av_t36 = _mm_mul_pd(v128_CRTM_14_6, av_s12);

        av_s46 = _mm_sub_pd(av_s2, av_t34);
        av_s47 = _mm_sub_pd(av_t35, av_t36);
        av_s48 = _mm_sub_pd(av_t32, av_t31);
        av_s25 = _mm_add_pd(av_s46, av_s47);
        av_s26 = _mm_sub_pd(av_s48, av_t33);
        // Output pt 13: X(12)
        curr_out = out + out_strides[12];
        v_out12 = _mm_add_pd(av_s25, av_s26);
        STR_128_D(curr_out, v_out_stride, v_out12);
        // Output pt 17: X(16)
        curr_out = out + out_strides[16];
        v_out16 = _mm_sub_pd(av_s25, av_s26);
        STR_128_D(curr_out, v_out_stride, v_out16);

        /* Shifted DFT */
        __m128d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
                bv_in8, bv_in9, bv_in10, bv_in11, bv_in12, bv_in13;
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
                bv_t26, bv_t27, bv_t28, bv_t29, bv_t30, bv_t31, bv_t32, bv_t33,
                bv_t34, bv_t35, bv_t36;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in4, bv_in5);
        // Input point 14: x(13) & Input point 15: x(14)
        curr_in = in + in_strides[13];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in6, bv_in7);
        // Input point 18: x(17) & Input point 19: x(18)
        curr_in = in + in_strides[17];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in8, bv_in9);
        // Input point 22: x(21) & Input point 23: x(22)
        curr_in = in + in_strides[21];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in10, bv_in11);
        // Input point 26: x(25) & Input point 27: x(26)
        curr_in = in + in_strides[25];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in12, bv_in13);

        bv_s1 = _mm_add_pd(bv_in0, bv_in12);
        bv_s2 = _mm_sub_pd(bv_in0, bv_in12);
        bv_s3 = _mm_add_pd(bv_in1, bv_in13);
        bv_s4 = _mm_sub_pd(bv_in1, bv_in13);
        bv_s5 = _mm_add_pd(bv_in2, bv_in10);
        bv_s6 = _mm_sub_pd(bv_in2, bv_in10);
        bv_s7 = _mm_add_pd(bv_in3, bv_in11);
        bv_s8 = _mm_sub_pd(bv_in3, bv_in11);
        bv_s9 = _mm_add_pd(bv_in4, bv_in8);
        bv_s10 = _mm_sub_pd(bv_in4, bv_in8);
        bv_s11 = _mm_add_pd(bv_in5, bv_in9);
        bv_s12 = _mm_sub_pd(bv_in5, bv_in9);

        bv_t1 = _mm_mul_pd(v128_CRTM_14_6, bv_s3);
        bv_t2 = _mm_mul_pd(v128_CRTM_14_2, bv_s11);
        bv_t3 = _mm_mul_pd(v128_CRTM_14_4, bv_s7);
        bv_t4 = _mm_mul_pd(v128_CRTM_14_5, bv_s2);
        bv_t5 = _mm_mul_pd(v128_CRTM_14_1, bv_s10);
        bv_t6 = _mm_mul_pd(v128_CRTM_14_3, bv_s6);

        bv_t7 = _mm_mul_pd(v128_CRTM_14_6, bv_s5);
        bv_t8 = _mm_mul_pd(v128_CRTM_14_2, bv_s1);
        bv_t9 = _mm_mul_pd(v128_CRTM_14_4, bv_s9);
        bv_t10 = _mm_mul_pd(v128_CRTM_14_5, bv_s8);
        bv_t11 = _mm_mul_pd(v128_CRTM_14_1, bv_s4);
        bv_t12 = _mm_mul_pd(v128_CRTM_14_3, bv_s12);

        bv_t13 = _mm_mul_pd(v128_CRTM_14_6, bv_s11);
        bv_t14 = _mm_mul_pd(v128_CRTM_14_2, bv_s7);
        bv_t15 = _mm_mul_pd(v128_CRTM_14_4, bv_s3);
        bv_t16 = _mm_mul_pd(v128_CRTM_14_5, bv_s10);
        bv_t17 = _mm_mul_pd(v128_CRTM_14_1, bv_s6);
        bv_t18 = _mm_mul_pd(v128_CRTM_14_3, bv_s2);

        bv_t19 = _mm_mul_pd(v128_CRTM_14_6, bv_s9);
        bv_t20 = _mm_mul_pd(v128_CRTM_14_2, bv_s5);
        bv_t21 = _mm_mul_pd(v128_CRTM_14_4, bv_s1);
        bv_t22 = _mm_mul_pd(v128_CRTM_14_5, bv_s12);
        bv_t23 = _mm_mul_pd(v128_CRTM_14_1, bv_s8);
        bv_t24 = _mm_mul_pd(v128_CRTM_14_3, bv_s4);

        bv_t25 = _mm_mul_pd(v128_CRTM_14_6, bv_s7);
        bv_t26 = _mm_mul_pd(v128_CRTM_14_2, bv_s3);
        bv_t27 = _mm_mul_pd(v128_CRTM_14_4, bv_s11);
        bv_t28 = _mm_mul_pd(v128_CRTM_14_5, bv_s6);
        bv_t29 = _mm_mul_pd(v128_CRTM_14_1, bv_s2);
        bv_t30 = _mm_mul_pd(v128_CRTM_14_3, bv_s10);

        bv_t31 = _mm_mul_pd(v128_CRTM_14_6, bv_s1);
        bv_t32 = _mm_mul_pd(v128_CRTM_14_2, bv_s9);
        bv_t33 = _mm_mul_pd(v128_CRTM_14_4, bv_s5);
        bv_t34 = _mm_mul_pd(v128_CRTM_14_5, bv_s4);
        bv_t35 = _mm_mul_pd(v128_CRTM_14_1, bv_s12);
        bv_t36 = _mm_mul_pd(v128_CRTM_14_3, bv_s8);

        bv_s13 = _mm_add_pd(bv_in6, bv_in6);
        bv_s14 = _mm_add_pd(bv_in7, bv_in7);

        bv_s15 = _mm_add_pd(bv_t1, bv_t2);
        bv_s16 = _mm_add_pd(bv_t3, bv_s14);
        bv_s17 = _mm_add_pd(bv_s15, bv_s16);
        bv_s18 = _mm_add_pd(bv_t4, bv_t5);
        bv_s19 = _mm_add_pd(bv_t6, bv_s18);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm_sub_pd(bv_s19, bv_s17);
        STR_128_D(curr_out, v_out_stride, v_out3);
        // Output pt 28: X(27)
        curr_out = out + out_strides[27];
        v_out27 = NEGATE_128_D(_mm_add_pd(bv_s17, bv_s19));
        STR_128_D(curr_out, v_out_stride, v_out27);

        bv_s20 = _mm_add_pd(bv_t10, bv_t11);
        bv_s21 = _mm_add_pd(bv_t12, bv_s20);
        bv_s22 = _mm_add_pd(bv_t7, bv_t8);
        bv_s23 = _mm_add_pd(bv_t9, bv_s13);
        bv_s24 = _mm_sub_pd(bv_s22, bv_s23);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm_sub_pd(bv_s24, bv_s21);
        STR_128_D(curr_out, v_out_stride, v_out5);
        // Output pt 26: X(25)
        curr_out = out + out_strides[25];
        v_out25 = NEGATE_128_D(_mm_add_pd(bv_s21, bv_s24));
        STR_128_D(curr_out, v_out_stride, v_out25);

        bv_s25 = _mm_sub_pd(bv_t13, bv_t14);
        bv_s26 = _mm_sub_pd(bv_s14, bv_t15);
        bv_s27 = _mm_add_pd(bv_s25, bv_s26);
        bv_s28 = _mm_add_pd(bv_t16, bv_t17);
        bv_s29 = _mm_sub_pd(bv_t18, bv_s28);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = _mm_add_pd(bv_s27, bv_s29);
        STR_128_D(curr_out, v_out_stride, v_out7);
        // Output pt 24: X(23)
        curr_out = out + out_strides[23];
        v_out23 = _mm_sub_pd(bv_s27, bv_s29);
        STR_128_D(curr_out, v_out_stride, v_out23);

        bv_s30 = _mm_sub_pd(bv_t22, bv_t23);
        bv_s31 = _mm_sub_pd(bv_s30, bv_t24);
        bv_s32 = _mm_add_pd(bv_t19, bv_t20);
        bv_s33 = _mm_add_pd(bv_t21, bv_s13);
        bv_s34 = _mm_sub_pd(bv_s33, bv_s32);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm_add_pd(bv_s31, bv_s34);
        STR_128_D(curr_out, v_out_stride, v_out9);
        // Output pt 22: X(21)
        curr_out = out + out_strides[21];
        v_out21 = _mm_sub_pd(bv_s31, bv_s34);
        STR_128_D(curr_out, v_out_stride, v_out21);

        bv_s35 = _mm_sub_pd(bv_t25, bv_t26);
        bv_s36 = _mm_sub_pd(bv_t27, bv_s14);
        bv_s37 = _mm_add_pd(bv_s35, bv_s36);
        bv_s38 = _mm_sub_pd(bv_t29, bv_t28);
        bv_s39 = _mm_add_pd(bv_t30, bv_s38);
        // Output pt 12: X(11)
        curr_out = out + out_strides[11];
        v_out11 = _mm_add_pd(bv_s37, bv_s39);
        STR_128_D(curr_out, v_out_stride, v_out11);
        // Output pt 20: X(19)
        curr_out = out + out_strides[19];
        v_out19 = _mm_sub_pd(bv_s37, bv_s39);
        STR_128_D(curr_out, v_out_stride, v_out19);

        bv_s40 = _mm_add_pd(bv_t34, bv_t35);
        bv_s41 = _mm_sub_pd(bv_t36, bv_s40);
        bv_s42 = _mm_add_pd(bv_t31, bv_t32);
        bv_s43 = _mm_add_pd(bv_t33, bv_s13);
        bv_s44 = _mm_sub_pd(bv_s42, bv_s43);
        // Output pt 14: X(13)
        curr_out = out + out_strides[13];
        v_out13 = _mm_add_pd(bv_s41, bv_s44);
        STR_128_D(curr_out, v_out_stride, v_out13);
        // Output pt 18: X(17)
        curr_out = out + out_strides[17];
        v_out17 = _mm_sub_pd(bv_s41, bv_s44);
        STR_128_D(curr_out, v_out_stride, v_out17);

        bv_s45 = _mm_add_pd(bv_s1, bv_s5);
        bv_s46 = _mm_add_pd(bv_in6, bv_s9);
        bv_s47 = _mm_add_pd(bv_s45, bv_s46);
        // Output pt 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm_mul_pd(bv_s47, v128_CRTM_14_7);
        STR_128_D(curr_out, v_out_stride, v_out1);

        bv_s48 = _mm_add_pd(bv_s3, bv_s11);
        bv_s49 = _mm_add_pd(bv_in7, bv_s7);
        bv_s50 = _mm_sub_pd(bv_s49, bv_s48);
        // Output pt 16: X(15)
        curr_out = out + out_strides[15];
        v_out15 = _mm_mul_pd(bv_s50, v128_CRTM_14_7);
        STR_128_D(curr_out, v_out_stride, v_out15);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        /* Standard DFT */
        FFTZ_DOUBLE a_in0, a_in1, a_in2, a_in3, a_in4, a_in5, a_in6, a_in7,
            a_in8, a_in9, a_in10, a_in11, a_in12, a_in13;
        FFTZ_DOUBLE a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_s9,
               a_s10, a_s11, a_s12, a_s13, a_s14, a_s15, a_s16, a_s17, a_s18,
               a_s19, a_s20, a_s21, a_s22, a_s23, a_s24, a_s25, a_s26, a_s27,
               a_s28, a_s29, a_s30, a_s31, a_s32, a_s33, a_s34, a_s35, a_s36,
               a_s37, a_s38, a_s39, a_s40, a_s41, a_s42, a_s43, a_s44, a_s45,
               a_s46, a_s47;
        FFTZ_DOUBLE a_t0, a_t1, a_t2, a_t3, a_t4, a_t5, a_t6, a_t7, a_t8, a_t9,
               a_t10, a_t11, a_t12, a_t13, a_t14, a_t15, a_t16, a_t17, a_t18,
               a_t19, a_t20, a_t21, a_t22, a_t23, a_t24, a_t25, a_t26, a_t27,
               a_t28, a_t29, a_t30, a_t31, a_t32, a_t33, a_t34, a_t35, a_t36,
               a_t37;

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

        a_s0 = a_in0 - a_in13;
        a_s1 = a_in0 + a_in13;
        a_s2 = a_in11 - a_in1;
        a_s3 = a_in1 + a_in11;
        a_s4 = a_in2 - a_in12;
        a_s5 = a_in2 + a_in12;
        a_s6 = a_in3 - a_in9;
        a_s7 = a_in3 + a_in9;
        a_s8 = a_in4 - a_in10;
        a_s9 = a_in4 + a_in10;
        a_s10 = a_in7 - a_in5;
        a_s11 = a_in5 + a_in7;
        a_s12 = a_in6 - a_in8;
        a_s13 = a_in6 + a_in8;

        a_s26 = a_s11 + a_s3;
        a_s27 = a_s26 + a_s7;
        a_t0 = CRTM_14_7 * a_s27;
        a_s28 = a_s2 + a_s6;
        a_s29 = a_s28 + a_s10;
        a_t1 = CRTM_14_7 * a_s29;
        // Output point 1: X(0)
        *out = a_t0 + a_s1;
        // Output point 15: X(14)
        out[out_strides[14]] = a_t1 + a_s0;

        a_t2 = CRTM_14_1 * a_s5;
        a_t3 = CRTM_14_3 * a_s9;
        a_t4 = CRTM_14_5 * a_s13;
        a_t5 = CRTM_14_2 * a_s2;
        a_t6 = CRTM_14_4 * a_s6;
        a_t7 = CRTM_14_6 * a_s10;

        a_s30 = a_t6 - a_t7;
        a_s31 = a_s0 - a_t5;
        a_s32 = a_t2 + a_t3;
        a_s14 = a_s30 + a_s31;
        a_s15 = a_s32 + a_t4;
        // Output point 3: X(2)
        out[out_strides[2]] = a_s14 - a_s15;
        // Output point 27: X(26)
        out[out_strides[26]] = a_s14 + a_s15;

        a_t8 = CRTM_14_1 * a_s12;
        a_t9 = CRTM_14_3 * a_s4;
        a_t10 = CRTM_14_5 * a_s8;
        a_t11 = CRTM_14_2 * a_s11;
        a_t12 = CRTM_14_4 * a_s3;
        a_t13 = CRTM_14_6 * a_s7;

        a_s33 = a_s1 - a_t11;
        a_s34 = a_t12 - a_t13;
        a_s35 = a_t8 + a_t9;
        a_s16 = a_s33 + a_s34;
        a_s17 = a_s35 + a_t10;
        // Output point 5: X(4)
        out[out_strides[4]] = a_s16 - a_s17;
        // Output point 25: X(24)
        out[out_strides[24]] = a_s16 + a_s17;

        a_t14 = CRTM_14_1 * a_s9;
        a_t15 = CRTM_14_3 * a_s13;
        a_t16 = CRTM_14_5 * a_s5;
        a_t17 = CRTM_14_2 * a_s6;
        a_t18 = CRTM_14_4 * a_s10;
        a_t19 = CRTM_14_6 * a_s2;

        a_s36 = a_s0 - a_t17;
        a_s37 = a_t18 - a_t19;
        a_s38 = a_t15 - a_t16;
        a_s18 = a_s36 + a_s37;
        a_s19 = a_s38 - a_t14;
        // Output point 7: X(6)
        out[out_strides[6]] = a_s18 + a_s19;
        // Output point 23: X(22)
        out[out_strides[22]] = a_s18 - a_s19;

        a_t20 = CRTM_14_1 * a_s8;
        a_t21 = CRTM_14_3 * a_s12;
        a_t22 = CRTM_14_5 * a_s4;
        a_t23 = CRTM_14_2 * a_s7;
        a_t24 = CRTM_14_4 * a_s11;
        a_t25 = CRTM_14_6 * a_s3;

        a_s39 = a_s1 - a_t23;
        a_s40 = a_t24 - a_t25;
        a_s41 = a_t20 + a_t21;
        a_s20 = a_s39 + a_s40;
        a_s21 = a_s41 - a_t22;
        // Output point 9: X(8)
        out[out_strides[8]] = a_s20 + a_s21;
        // Output point 21: X(20)
        out[out_strides[20]] = a_s20 - a_s21;

        a_t26 = CRTM_14_1 * a_s13;
        a_t27 = CRTM_14_3 * a_s5;
        a_t28 = CRTM_14_5 * a_s9;
        a_t29 = CRTM_14_2 * a_s10;
        a_t30 = CRTM_14_4 * a_s2;
        a_t31 = CRTM_14_6 * a_s6;

        a_s42 = a_s0 - a_t29;
        a_s43 = a_t30 - a_t31;
        a_s44 = a_t28 - a_t26;
        a_s22 = a_s42 + a_s43;
        a_s23 = a_s44 - a_t27;
        // Output point 11: X(10)
        out[out_strides[10]] = a_s22 + a_s23;
        // Output point 19: X(18)
        out[out_strides[18]] = a_s22 - a_s23;

        a_t32 = CRTM_14_1 * a_s4;
        a_t33 = CRTM_14_3 * a_s8;
        a_t34 = CRTM_14_5 * a_s12;
        a_t35 = CRTM_14_2 * a_s3;
        a_t36 = CRTM_14_4 * a_s7;
        a_t37 = CRTM_14_6 * a_s11;

        a_s45 = a_s1 - a_t35;
        a_s46 = a_t36 - a_t37;
        a_s47 = a_t33 - a_t32;
        a_s24 = a_s45 + a_s46;
        a_s25 = a_s47 - a_t34;
        // Output point 13: X(12)
        out[out_strides[12]] = a_s24 + a_s25;
        // Output point 17: X(16)
        out[out_strides[16]] = a_s24 - a_s25;

        /* Shifted DFT */
        FFTZ_DOUBLE b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7,
            b_in8, b_in9, b_in10, b_in11, b_in12, b_in13;
        FFTZ_DOUBLE b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9,
               b_s10, b_s11, b_s12, b_s13, b_s14, b_s15, b_s16, b_s17, b_s18,
               b_s19, b_s20, b_s21, b_s22, b_s23, b_s24, b_s25, b_s26, b_s27,
               b_s28, b_s29, b_s30, b_s31, b_s32, b_s33, b_s34, b_s35, b_s36,
               b_s37, b_s38, b_s39, b_s40, b_s41, b_s42, b_s43, b_s44, b_s45,
               b_s46, b_s47, b_s48, b_s49;
        FFTZ_DOUBLE b_t0, b_t1, b_t2, b_t3, b_t4, b_t5, b_t6, b_t7, b_t8, b_t9,
               b_t10, b_t11, b_t12, b_t13, b_t14, b_t15, b_t16, b_t17, b_t18,
               b_t19, b_t20, b_t21, b_t22, b_t23, b_t24, b_t25, b_t26, b_t27,
               b_t28, b_t29, b_t30, b_t31, b_t32, b_t33, b_t34, b_t35;

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

        b_s0 = b_in0 + b_in12;
        b_s1 = b_in0 - b_in12;
        b_s2 = b_in1 + b_in13;
        b_s3 = b_in1 - b_in13;
        b_s4 = b_in2 + b_in10;
        b_s5 = b_in2 - b_in10;
        b_s6 = b_in3 + b_in11;
        b_s7 = b_in3 - b_in11;
        b_s8 = b_in4 + b_in8;
        b_s9 = b_in4 - b_in8;
        b_s10 = b_in5 + b_in9;
        b_s11 = b_in5 - b_in9;

        b_t0 = CRTM_14_6 * b_s2;
        b_t1 = CRTM_14_2 * b_s10;
        b_t2 = CRTM_14_4 * b_s6;
        b_t3 = CRTM_14_5 * b_s1;
        b_t4 = CRTM_14_1 * b_s9;
        b_t5 = CRTM_14_3 * b_s5;

        b_t6 = CRTM_14_6 * b_s4;
        b_t7 = CRTM_14_2 * b_s0;
        b_t8 = CRTM_14_4 * b_s8;
        b_t9 = CRTM_14_5 * b_s7;
        b_t10 = CRTM_14_1 * b_s3;
        b_t11 = CRTM_14_3 * b_s11;

        b_t12 = CRTM_14_6 * b_s10;
        b_t13 = CRTM_14_2 * b_s6;
        b_t14 = CRTM_14_4 * b_s2;
        b_t15 = CRTM_14_5 * b_s9;
        b_t16 = CRTM_14_1 * b_s5;
        b_t17 = CRTM_14_3 * b_s1;

        b_t18 = CRTM_14_6 * b_s8;
        b_t19 = CRTM_14_2 * b_s4;
        b_t20 = CRTM_14_4 * b_s0;
        b_t21 = CRTM_14_5 * b_s11;
        b_t22 = CRTM_14_1 * b_s7;
        b_t23 = CRTM_14_3 * b_s3;

        b_t24 = CRTM_14_6 * b_s6;
        b_t25 = CRTM_14_2 * b_s2;
        b_t26 = CRTM_14_4 * b_s10;
        b_t27 = CRTM_14_5 * b_s5;
        b_t28 = CRTM_14_1 * b_s1;
        b_t29 = CRTM_14_3 * b_s9;

        b_t30 = CRTM_14_6 * b_s0;
        b_t31 = CRTM_14_2 * b_s8;
        b_t32 = CRTM_14_4 * b_s4;
        b_t33 = CRTM_14_5 * b_s3;
        b_t34 = CRTM_14_1 * b_s11;
        b_t35 = CRTM_14_3 * b_s7;

        b_s12 = b_in6 + b_in6;
        b_s13 = b_in7 + b_in7;

        b_s14 = b_t0 + b_t1;
        b_s15 = b_t2 + b_s13;
        b_s16 = b_s14 + b_s15;
        b_s17 = b_t3 + b_t4;
        b_s18 = b_t5 + b_s17;
        // Output point 4: X(3)
        out[out_strides[3]] = b_s18 - b_s16;
        // Output point 28: X(27)
        out[out_strides[27]] = -(b_s16 + b_s18);

        b_s19 = b_t9 + b_t10;
        b_s20 = b_t11 + b_s19;
        b_s21 = b_t6 + b_t7;
        b_s22 = b_t8 + b_s12;
        b_s23 = b_s21 - b_s22;
        // Output point 6: X(5)
        out[out_strides[5]] = b_s23 - b_s20;
        // Output point 26: X(25)
        out[out_strides[25]] = -(b_s20 + b_s23);

        b_s24 = b_t12 - b_t13;
        b_s25 = b_s13 - b_t14;
        b_s26 = b_s24 + b_s25;
        b_s27 = b_t15 + b_t16;
        b_s28 = b_t17 - b_s27;
        // Output point 8: X(7)
        out[out_strides[7]] = b_s26 + b_s28;
        // Output point 24: X(23)
        out[out_strides[23]] = b_s26 - b_s28;

        b_s29 = b_t21 - b_t22;
        b_s30 = b_s29 - b_t23;
        b_s31 = b_t18 + b_t19;
        b_s32 = b_t20 + b_s12;
        b_s33 = b_s32 - b_s31;
        // Output point 10: X(9)
        out[out_strides[9]] = b_s30 + b_s33;
        // Output point 22: X(21)
        out[out_strides[21]] = b_s30 - b_s33;

        b_s34 = b_t24 - b_t25;
        b_s35 = b_t26 - b_s13;
        b_s36 = b_s34 + b_s35;
        b_s37 = b_t28 - b_t27;
        b_s38 = b_t29 + b_s37;
        // Output point 12: X(11)
        out[out_strides[11]] = b_s36 + b_s38;
        // Output point 20: X(19)
        out[out_strides[19]] = b_s36 - b_s38;

        b_s39 = b_t33 + b_t34;
        b_s40 = b_t35 - b_s39;
        b_s41 = b_t30 + b_t31;
        b_s42 = b_t32 + b_s12;
        b_s43 = b_s41 - b_s42;
        // Output point 14: X(13)
        out[out_strides[13]] = b_s40 + b_s43;
        // Output point 18: X(17)
        out[out_strides[17]] = b_s40 - b_s43;

        b_s44 = b_s0 + b_s4;
        b_s45 = b_in6 + b_s8;
        b_s46 = b_s44 + b_s45;
        // Output point 2: X(1)
        out[out_strides[1]] = b_s46 * CRTM_14_7;

        b_s47 = b_s2 + b_s10;
        b_s48 = b_in7 + b_s6;
        b_s49 = b_s48 - b_s47;
        // Output point 16: X(15)
        out[out_strides[15]] = b_s49 * CRTM_14_7;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hcf_rfft14avx512(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hcf_rfft14avx512_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft14avx512_fp64_fwd;
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
            return r2hcf_rfft14avx512_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft14avx512_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
