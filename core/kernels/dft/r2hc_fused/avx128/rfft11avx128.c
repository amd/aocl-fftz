// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft11avx128.c
 *
 *  @brief Radix-11 r2hc_fused Real-FFT kernel with AVX-128 operations
 *  using x86 SIMD intrinsics
 *
 *  This file contains the DIT radix-11 real-to-halfcomplex fused of two
 *  different implementations (Standard DFT and Shifted DFT that differs in DFT
 *  weight matrix) using x86 SIMD intrinsics for single-precision and
 *  double-precision inputs
 *
 *  @author Amrin Fathima
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/r2hc_simd_avx128.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                   {{{0, 100, 124, 48, 0, 0},
                                                     {0, 102, 124, 48, 0, 0}},
                                                    {{0, 100, 124, 48, 0, 0},
                                                     {0, 102, 124, 48, 0, 0}}};

ops_cycles_t get_ops_cnt_r2hcf_rfft11avx128(FFTZ_UINT8 precision,
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

static FFTZ_VOID r2hcf_rfft11avx128_fp32_fwd(FFTZ_VOID *in_real,
                                             FFTZ_VOID *in_imag,
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
    FFTZ_UINT8 is_contiguous_in = (v_in_stride == 1);
    FFTZ_UINT8 is_contiguous_out = (v_out_stride == 1);

    FFTZ_INTP cnt;
    FFTZ_FLOAT *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_128_S;

    __m128 v_CRTM_11_1 = _mm_broadcast_ss(&CRTM_11_1);
    __m128 v_CRTM_11_2 = _mm_broadcast_ss(&CRTM_11_2);
    __m128 v_CRTM_11_3 = _mm_broadcast_ss(&CRTM_11_3);
    __m128 v_CRTM_11_4 = _mm_broadcast_ss(&CRTM_11_4);
    __m128 v_CRTM_11_5 = _mm_broadcast_ss(&CRTM_11_5);
    __m128 v_CRTM_11_6 = _mm_broadcast_ss(&CRTM_11_6);
    __m128 v_CRTM_11_7 = _mm_broadcast_ss(&CRTM_11_7);
    __m128 v_CRTM_11_8 = _mm_broadcast_ss(&CRTM_11_8);
    __m128 v_CRTM_11_9 = _mm_broadcast_ss(&CRTM_11_9);
    __m128 v_CRTM_11_10 = _mm_broadcast_ss(&CRTM_11_10);

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
               av_in8, av_in9, av_in10;
        __m128 av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
               av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16,
               av_s17, av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24,
               av_s25, av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32,
               av_s33, av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40,
               av_s41, av_s42, av_s43, av_s44, av_s45, av_s46, av_s47, av_s48;
        __m128 av_m0, av_m1, av_m2, av_m3, av_m4, av_m5, av_m6, av_m7, av_m8,
               av_m9, av_m10, av_m11, av_m12, av_m13, av_m14, av_m15, av_m16,
               av_m17, av_m18, av_m19, av_m20, av_m21, av_m22, av_m23, av_m24,
               av_m25, av_m26, av_m27, av_m28, av_m29, av_m30, av_m31, av_m32,
               av_m33, av_m34, av_m35, av_m36, av_m37, av_m38, av_m39, av_m40,
               av_m41, av_m42, av_m43, av_m44, av_m45, av_m46, av_m47, av_m48,
               av_m49;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_stride, av_in0, is_contiguous_in);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_128_S(curr_in, v_in_stride, av_in1, is_contiguous_in);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_128_S(curr_in, v_in_stride, av_in2, is_contiguous_in);
        // Input point 7: x(6)
        curr_in = in + in_strides[6];
        LDR_128_S(curr_in, v_in_stride, av_in3, is_contiguous_in);
        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDR_128_S(curr_in, v_in_stride, av_in4, is_contiguous_in);
        // Input point 11: x(10)
        curr_in = in + in_strides[10];
        LDR_128_S(curr_in, v_in_stride, av_in5, is_contiguous_in);
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDR_128_S(curr_in, v_in_stride, av_in6, is_contiguous_in);
        // Input point 15: x(14)
        curr_in = in + in_strides[14];
        LDR_128_S(curr_in, v_in_stride, av_in7, is_contiguous_in);
        // Input point 17: x(16)
        curr_in = in + in_strides[16];
        LDR_128_S(curr_in, v_in_stride, av_in8, is_contiguous_in);
        // Input point 19: x(18)
        curr_in = in + in_strides[18];
        LDR_128_S(curr_in, v_in_stride, av_in9, is_contiguous_in);
        // Input point 21: x(20)
        curr_in = in + in_strides[20];
        LDR_128_S(curr_in, v_in_stride, av_in10, is_contiguous_in);

        av_s0 = _mm_add_ps(av_in1, av_in10);
        av_s1 = _mm_add_ps(av_in2, av_in9);
        av_s2 = _mm_add_ps(av_in3, av_in8);
        av_s3 = _mm_add_ps(av_in4, av_in7);
        av_s4 = _mm_add_ps(av_in5, av_in6);
        av_s5 = _mm_sub_ps(av_in1, av_in10);
        av_s6 = _mm_sub_ps(av_in2, av_in9);
        av_s7 = _mm_sub_ps(av_in3, av_in8);
        av_s8 = _mm_sub_ps(av_in4, av_in7);
        av_s9 = _mm_sub_ps(av_in5, av_in6);

        av_s10 = _mm_add_ps(av_s0, av_s1);
        av_s11 = _mm_add_ps(av_s2, av_s3);
        av_s12 = _mm_add_ps(av_s4, av_in0);
        av_s13 = _mm_add_ps(av_s10, av_s11);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(av_s12, av_s13);
        STR_128_S(curr_out, v_out_stride, v_out0, is_contiguous_out);

        av_m0 = _mm_mul_ps(v_CRTM_11_1, av_s0);
        av_m1 = _mm_mul_ps(v_CRTM_11_3, av_s1);
        av_m2 = _mm_mul_ps(v_CRTM_11_5, av_s2);
        av_m3 = _mm_mul_ps(v_CRTM_11_7, av_s3);
        av_m4 = _mm_mul_ps(v_CRTM_11_9, av_s4);

        av_s14 = _mm_add_ps(av_m0, av_m1);
        av_s15 = _mm_add_ps(av_m2, av_m3);
        av_s16 = _mm_sub_ps(av_in0, av_m4);
        av_s17 = _mm_sub_ps(av_s14, av_s15);

        // Output point 4: X(3)
        v_out3 = _mm_add_ps(av_s16, av_s17);
        curr_out = out + out_strides[3];
        STR_128_S(curr_out, v_out_stride, v_out3, is_contiguous_out);

        av_m5 = _mm_mul_ps(v_CRTM_11_2, av_s5);
        av_m6 = _mm_mul_ps(v_CRTM_11_4, av_s6);
        av_m7 = _mm_mul_ps(v_CRTM_11_6, av_s7);
        av_m8 = _mm_mul_ps(v_CRTM_11_8, av_s8);
        av_m9 = _mm_mul_ps(v_CRTM_11_10, av_s9);

        av_s18 = _mm_add_ps(av_m5, av_m6);
        av_s19 = _mm_add_ps(av_m7, av_m8);
        av_s20 = _mm_add_ps(av_s19, av_m9);

        // Output point 5: X(4)
        v_out4 = NEGATE_128_S(_mm_add_ps(av_s18, av_s20));
        curr_out = out + out_strides[4];
        STR_128_S(curr_out, v_out_stride, v_out4, is_contiguous_out);

        av_m10 = _mm_mul_ps(v_CRTM_11_1, av_s4);
        av_m11 = _mm_mul_ps(v_CRTM_11_3, av_s0);
        av_m12 = _mm_mul_ps(v_CRTM_11_5, av_s3);
        av_m13 = _mm_mul_ps(v_CRTM_11_7, av_s1);
        av_m14 = _mm_mul_ps(v_CRTM_11_9, av_s2);

        av_s21 = _mm_add_ps(av_m10, av_m11);
        av_s22 = _mm_add_ps(av_m12, av_m13);
        av_s23 = _mm_sub_ps(av_in0, av_m14);
        av_s24 = _mm_sub_ps(av_s21, av_s22);

        // Output point 8: X(7)
        v_out7 = _mm_add_ps(av_s23, av_s24);
        curr_out = out + out_strides[7];
        STR_128_S(curr_out, v_out_stride, v_out7, is_contiguous_out);

        av_m15 = _mm_mul_ps(v_CRTM_11_2, av_s9);
        av_m16 = _mm_mul_ps(v_CRTM_11_4, av_s5);
        av_m17 = _mm_mul_ps(v_CRTM_11_6, av_s8);
        av_m18 = _mm_mul_ps(v_CRTM_11_8, av_s6);
        av_m19 = _mm_mul_ps(v_CRTM_11_10, av_s7);

        av_s25 = _mm_sub_ps(av_m15, av_m16);
        av_s26 = _mm_sub_ps(av_m17, av_m18);
        av_s27 = _mm_add_ps(av_s26, av_m19);

        // Output point 9: X(8)
        v_out8 = _mm_add_ps(av_s25, av_s27);
        curr_out = out + out_strides[8];
        STR_128_S(curr_out, v_out_stride, v_out8, is_contiguous_out);

        av_m20 = _mm_mul_ps(v_CRTM_11_1, av_s3);
        av_m21 = _mm_mul_ps(v_CRTM_11_3, av_s2);
        av_m22 = _mm_mul_ps(v_CRTM_11_5, av_s0);
        av_m23 = _mm_mul_ps(v_CRTM_11_7, av_s4);
        av_m24 = _mm_mul_ps(v_CRTM_11_9, av_s1);

        av_s28 = _mm_add_ps(av_m20, av_m21);
        av_s29 = _mm_add_ps(av_m22, av_m23);
        av_s30 = _mm_sub_ps(av_in0, av_m24);
        av_s31 = _mm_sub_ps(av_s30, av_s29);

        // Output point 12: X(11)
        v_out11 = _mm_add_ps(av_s28, av_s31);
        curr_out = out + out_strides[11];
        STR_128_S(curr_out, v_out_stride, v_out11, is_contiguous_out);

        av_m25 = _mm_mul_ps(v_CRTM_11_2, av_s8);
        av_m26 = _mm_mul_ps(v_CRTM_11_4, av_s7);
        av_m27 = _mm_mul_ps(v_CRTM_11_6, av_s5);
        av_m28 = _mm_mul_ps(v_CRTM_11_8, av_s9);
        av_m29 = _mm_mul_ps(v_CRTM_11_10, av_s6);

        av_s32 = _mm_sub_ps(av_m26, av_m25);
        av_s33 = _mm_add_ps(av_m27, av_m28);
        av_s34 = _mm_sub_ps(av_s32, av_s33);

        // Output point 13: X(12)
        v_out12 = _mm_add_ps(av_s34, av_m29);
        curr_out = out + out_strides[12];
        STR_128_S(curr_out, v_out_stride, v_out12, is_contiguous_out);

        av_m30 = _mm_mul_ps(v_CRTM_11_1, av_s2);
        av_m31 = _mm_mul_ps(v_CRTM_11_3, av_s4);
        av_m32 = _mm_mul_ps(v_CRTM_11_5, av_s1);
        av_m33 = _mm_mul_ps(v_CRTM_11_7, av_s0);
        av_m34 = _mm_mul_ps(v_CRTM_11_9, av_s3);

        av_s35 = _mm_add_ps(av_m30, av_m31);
        av_s36 = _mm_add_ps(av_m32, av_m33);
        av_s37 = _mm_sub_ps(av_in0, av_m34);
        av_s38 = _mm_sub_ps(av_s35, av_s36);

        // Output point 16: X(15)
        v_out15 = _mm_add_ps(av_s38, av_s37);
        curr_out = out + out_strides[15];
        STR_128_S(curr_out, v_out_stride, v_out15, is_contiguous_out);

        av_m35 = _mm_mul_ps(v_CRTM_11_2, av_s7);
        av_m36 = _mm_mul_ps(v_CRTM_11_4, av_s9);
        av_m37 = _mm_mul_ps(v_CRTM_11_6, av_s6);
        av_m38 = _mm_mul_ps(v_CRTM_11_8, av_s5);
        av_m39 = _mm_mul_ps(v_CRTM_11_10, av_s8);

        av_s39 = _mm_sub_ps(av_m36, av_m35);
        av_s40 = _mm_sub_ps(av_m37, av_m38);
        av_s41 = _mm_sub_ps(av_s40, av_m39);

        // Output point 17: X(16)
        v_out16 = _mm_add_ps(av_s39, av_s41);
        curr_out = out + out_strides[16];
        STR_128_S(curr_out, v_out_stride, v_out16, is_contiguous_out);

        av_m40 = _mm_mul_ps(v_CRTM_11_1, av_s1);
        av_m41 = _mm_mul_ps(v_CRTM_11_3, av_s3);
        av_m42 = _mm_mul_ps(v_CRTM_11_5, av_s4);
        av_m43 = _mm_mul_ps(v_CRTM_11_7, av_s2);
        av_m44 = _mm_mul_ps(v_CRTM_11_9, av_s0);

        av_s42 = _mm_add_ps(av_m40, av_m41);
        av_s43 = _mm_add_ps(av_m42, av_m43);
        av_s44 = _mm_sub_ps(av_in0, av_m44);
        av_s45 = _mm_sub_ps(av_s42, av_s43);

        // Output point 20: X(19)
        v_out19 = _mm_add_ps(av_s44, av_s45);
        curr_out = out + out_strides[19];
        STR_128_S(curr_out, v_out_stride, v_out19, is_contiguous_out);

        av_m45 = _mm_mul_ps(v_CRTM_11_2, av_s6);
        av_m46 = _mm_mul_ps(v_CRTM_11_4, av_s8);
        av_m47 = _mm_mul_ps(v_CRTM_11_6, av_s9);
        av_m48 = _mm_mul_ps(v_CRTM_11_8, av_s7);
        av_m49 = _mm_mul_ps(v_CRTM_11_10, av_s5);

        av_s46 = _mm_add_ps(av_m45, av_m46);
        av_s47 = _mm_add_ps(av_m47, av_m48);
        av_s48 = _mm_sub_ps(av_s46, av_s47);

        // Output point 21: X(20)
        v_out20 = _mm_sub_ps(av_s48, av_m49);
        curr_out = out + out_strides[20];
        STR_128_S(curr_out, v_out_stride, v_out20, is_contiguous_out);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8, bv_in9, bv_in10;
        __m128 bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
               bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16,
               bv_s17, bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24,
               bv_s25, bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32,
               bv_s33, bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40,
               bv_s41, bv_s42, bv_s43, bv_s44, bv_s45, bv_s46, bv_s47, bv_s48,
               bv_s49, bv_s50, bv_s51, bv_s52, bv_s53, bv_s54;
        __m128 bv_m0, bv_m1, bv_m2, bv_m3, bv_m4, bv_m5, bv_m6, bv_m7, bv_m8,
               bv_m9, bv_m10, bv_m11, bv_m12, bv_m13, bv_m14, bv_m15, bv_m16,
               bv_m17, bv_m18, bv_m19, bv_m20, bv_m21, bv_m22, bv_m23, bv_m24,
               bv_m25, bv_m26, bv_m27, bv_m28, bv_m29, bv_m30, bv_m31, bv_m32,
               bv_m33, bv_m34, bv_m35, bv_m36, bv_m37, bv_m38, bv_m39, bv_m40,
               bv_m41, bv_m42, bv_m43, bv_m44, bv_m45, bv_m46, bv_m47, bv_m48,
               bv_m49;

        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_128_S(curr_in, v_in_stride, bv_in0, is_contiguous_in);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_128_S(curr_in, v_in_stride, bv_in1, is_contiguous_in);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_128_S(curr_in, v_in_stride, bv_in2, is_contiguous_in);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_128_S(curr_in, v_in_stride, bv_in3, is_contiguous_in);
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDR_128_S(curr_in, v_in_stride, bv_in4, is_contiguous_in);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_128_S(curr_in, v_in_stride, bv_in5, is_contiguous_in);
        // Input point 14: x(13)
        curr_in = in + in_strides[13];
        LDR_128_S(curr_in, v_in_stride, bv_in6, is_contiguous_in);
        // Input point 16: x(15)
        curr_in = in + in_strides[15];
        LDR_128_S(curr_in, v_in_stride, bv_in7, is_contiguous_in);
        // Input point 18: x(17)
        curr_in = in + in_strides[17];
        LDR_128_S(curr_in, v_in_stride, bv_in8, is_contiguous_in);
        // Input point 20: x(19)
        curr_in = in + in_strides[19];
        LDR_128_S(curr_in, v_in_stride, bv_in9, is_contiguous_in);
        // Input point 22: x(21)
        curr_in = in + in_strides[21];
        LDR_128_S(curr_in, v_in_stride, bv_in10, is_contiguous_in);

        bv_s0 = _mm_add_ps(bv_in1, bv_in10);
        bv_s1 = _mm_add_ps(bv_in2, bv_in9);
        bv_s2 = _mm_add_ps(bv_in3, bv_in8);
        bv_s3 = _mm_add_ps(bv_in4, bv_in7);
        bv_s4 = _mm_add_ps(bv_in5, bv_in6);
        bv_s5 = _mm_sub_ps(bv_in1, bv_in10);
        bv_s6 = _mm_sub_ps(bv_in2, bv_in9);
        bv_s7 = _mm_sub_ps(bv_in3, bv_in8);
        bv_s8 = _mm_sub_ps(bv_in4, bv_in7);
        bv_s9 = _mm_sub_ps(bv_in5, bv_in6);

        bv_m0 = _mm_mul_ps(v_CRTM_11_9, bv_s5);
        bv_m1 = _mm_mul_ps(v_CRTM_11_1, bv_s6);
        bv_s10 = _mm_add_ps(bv_m0, bv_m1);
        bv_m2 = _mm_mul_ps(v_CRTM_11_7, bv_s7);
        bv_s11 = _mm_add_ps(bv_s10, bv_m2);
        bv_m3 = _mm_mul_ps(v_CRTM_11_3, bv_s8);
        bv_s12 = _mm_add_ps(bv_s11, bv_m3);
        bv_m4 = _mm_mul_ps(v_CRTM_11_5, bv_s9);
        bv_s13 = _mm_add_ps(bv_s12, bv_m4);

        // Output point 2: X(1)
        v_out1 = _mm_add_ps(bv_in0, bv_s13);

        bv_m5 = _mm_mul_ps(v_CRTM_11_10, bv_s0);
        bv_m6 = _mm_mul_ps(v_CRTM_11_2, bv_s1);
        bv_s14 = _mm_add_ps(bv_m5, bv_m6);
        bv_m7 = _mm_mul_ps(v_CRTM_11_8, bv_s2);
        bv_s15 = _mm_add_ps(bv_s14, bv_m7);
        bv_m8 = _mm_mul_ps(v_CRTM_11_4, bv_s3);
        bv_s16 = _mm_add_ps(bv_s15, bv_m8);
        bv_m9 = _mm_mul_ps(v_CRTM_11_6, bv_s4);
        bv_s17 = _mm_add_ps(bv_s16, bv_m9);

        // Output point 3: X(2)
        v_out2 = NEGATE_128_S(bv_s17);
        curr_out = out + out_strides[1];
        STRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);

        bv_m10 = _mm_mul_ps(v_CRTM_11_7, bv_s5);
        bv_m11 = _mm_mul_ps(v_CRTM_11_5, bv_s6);
        bv_s18 = _mm_sub_ps(bv_m10, bv_m11);
        bv_m12 = _mm_mul_ps(v_CRTM_11_1, bv_s7);
        bv_s19 = _mm_sub_ps(bv_s18, bv_m12);
        bv_m13 = _mm_mul_ps(v_CRTM_11_9, bv_s8);
        bv_s20 = _mm_sub_ps(bv_s19, bv_m13);
        bv_m14 = _mm_mul_ps(v_CRTM_11_3, bv_s9);
        bv_s21 = _mm_sub_ps(bv_s20, bv_m14);

        // Output point 6: X(5)
        v_out5 = _mm_add_ps(bv_in0, bv_s21);

        bv_m15 = _mm_mul_ps(v_CRTM_11_8, bv_s0);
        bv_m16 = _mm_mul_ps(v_CRTM_11_6, bv_s1);
        bv_s22 = _mm_add_ps(bv_m15, bv_m16);
        bv_m17 = _mm_mul_ps(v_CRTM_11_2, bv_s2);
        bv_s23 = _mm_add_ps(bv_s22, bv_m17);
        bv_m18 = _mm_mul_ps(v_CRTM_11_10, bv_s3);
        bv_s24 = _mm_sub_ps(bv_s23, bv_m18);
        bv_m19 = _mm_mul_ps(v_CRTM_11_4, bv_s4);
        bv_s25 = _mm_sub_ps(bv_s24, bv_m19);

        // Output point 7: X(6)
        v_out6 = NEGATE_128_S(bv_s25);
        curr_out = out + out_strides[5];
        STRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);

        bv_m20 = _mm_mul_ps(v_CRTM_11_5, bv_s5);
        bv_m21 = _mm_mul_ps(v_CRTM_11_9, bv_s6);
        bv_s26 = _mm_sub_ps(bv_m20, bv_m21);
        bv_m22 = _mm_mul_ps(v_CRTM_11_3, bv_s7);
        bv_s27 = _mm_sub_ps(bv_s26, bv_m22);
        bv_m23 = _mm_mul_ps(v_CRTM_11_1, bv_s8);
        bv_s28 = _mm_add_ps(bv_s27, bv_m23);
        bv_m24 = _mm_mul_ps(v_CRTM_11_7, bv_s9);
        bv_s29 = _mm_add_ps(bv_s28, bv_m24);

        // Output point 10: X(9)
        v_out9 = _mm_add_ps(bv_in0, bv_s29);

        bv_m25 = _mm_mul_ps(v_CRTM_11_6, bv_s0);
        bv_m26 = _mm_mul_ps(v_CRTM_11_10, bv_s1);
        bv_s30 = _mm_add_ps(bv_m25, bv_m26);
        bv_m27 = _mm_mul_ps(v_CRTM_11_4, bv_s2);
        bv_s31 = _mm_sub_ps(bv_s30, bv_m27);
        bv_m28 = _mm_mul_ps(v_CRTM_11_2, bv_s3);
        bv_s32 = _mm_sub_ps(bv_s31, bv_m28);
        bv_m29 = _mm_mul_ps(v_CRTM_11_8, bv_s4);
        bv_s33 = _mm_add_ps(bv_s32, bv_m29);

        // Output point 11: X(10)
        v_out10 = NEGATE_128_S(bv_s33);
        curr_out = out + out_strides[9];
        STRI_2x128_S(curr_out, v_out_stride, v_out9, v_out10);

        bv_m30 = _mm_mul_ps(v_CRTM_11_3, bv_s5);
        bv_m31 = _mm_mul_ps(v_CRTM_11_7, bv_s6);
        bv_s34 = _mm_add_ps(bv_m30, bv_m31);
        bv_m32 = _mm_mul_ps(v_CRTM_11_9, bv_s7);
        bv_s35 = _mm_sub_ps(bv_m32, bv_s34);
        bv_m33 = _mm_mul_ps(v_CRTM_11_5, bv_s8);
        bv_s36 = _mm_sub_ps(bv_s35, bv_m33);
        bv_m34 = _mm_mul_ps(v_CRTM_11_1, bv_s9);
        bv_s37 = _mm_sub_ps(bv_s36, bv_m34);

        // Output point 14: X(13)
        v_out13 = _mm_add_ps(bv_in0, bv_s37);

        bv_m35 = _mm_mul_ps(v_CRTM_11_4, bv_s0);
        bv_m36 = _mm_mul_ps(v_CRTM_11_8, bv_s1);
        bv_s38 = _mm_sub_ps(bv_m35, bv_m36);
        bv_m37 = _mm_mul_ps(v_CRTM_11_10, bv_s2);
        bv_s39 = _mm_sub_ps(bv_s38, bv_m37);
        bv_m38 = _mm_mul_ps(v_CRTM_11_6, bv_s3);
        bv_s40 = _mm_add_ps(bv_s39, bv_m38);
        bv_m39 = _mm_mul_ps(v_CRTM_11_2, bv_s4);
        bv_s41 = _mm_sub_ps(bv_s40, bv_m39);

        // Output point 15: X(14)
        v_out14 = NEGATE_128_S(bv_s41);
        curr_out = out + out_strides[13];
        STRI_2x128_S(curr_out, v_out_stride, v_out13, v_out14);

        bv_m40 = _mm_mul_ps(v_CRTM_11_1, bv_s5);
        bv_m41 = _mm_mul_ps(v_CRTM_11_3, bv_s6);
        bv_s42 = _mm_sub_ps(bv_m41, bv_m40);
        bv_m42 = _mm_mul_ps(v_CRTM_11_5, bv_s7);
        bv_s43 = _mm_add_ps(bv_s42, bv_m42);
        bv_m43 = _mm_mul_ps(v_CRTM_11_7, bv_s8);
        bv_s44 = _mm_sub_ps(bv_s43, bv_m43);
        bv_m44 = _mm_mul_ps(v_CRTM_11_9, bv_s9);
        bv_s45 = _mm_add_ps(bv_s44, bv_m44);

        // Output point 18: X(17)
        v_out17 = _mm_add_ps(bv_in0, bv_s45);

        bv_m45 = _mm_mul_ps(v_CRTM_11_2, bv_s0);
        bv_m46 = _mm_mul_ps(v_CRTM_11_4, bv_s1);
        bv_s46 = _mm_sub_ps(bv_m45, bv_m46);
        bv_m47 = _mm_mul_ps(v_CRTM_11_6, bv_s2);
        bv_s47 = _mm_add_ps(bv_s46, bv_m47);
        bv_m48 = _mm_mul_ps(v_CRTM_11_8, bv_s3);
        bv_s48 = _mm_sub_ps(bv_s47, bv_m48);
        bv_m49 = _mm_mul_ps(v_CRTM_11_10, bv_s4);
        bv_s49 = _mm_add_ps(bv_s48, bv_m49);

        // Output point 19: X(18)
        v_out18 = NEGATE_128_S(bv_s49);
        curr_out = out + out_strides[17];
        STRI_2x128_S(curr_out, v_out_stride, v_out17, v_out18);

        bv_s50 = _mm_sub_ps(bv_in0, bv_s5);
        bv_s51 = _mm_add_ps(bv_s50, bv_s6);
        bv_s52 = _mm_sub_ps(bv_s51, bv_s7);
        bv_s53 = _mm_add_ps(bv_s52, bv_s8);
        bv_s54 = _mm_sub_ps(bv_s53, bv_s9);

        // Output point 21: X(20)
        v_out21 = bv_s54;
        curr_out = out + out_strides[21];
        STR_128_S(curr_out, v_out_stride, v_out21, is_contiguous_out);

        in += v_in_stride * NUM_SETS_REAL_128_S;
        out += v_out_stride * NUM_SETS_REAL_128_S;
    }
    // tail cases
    if (n & 2)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
               av_in8, av_in9, av_in10;
        __m128 av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
               av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16,
               av_s17, av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24,
               av_s25, av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32,
               av_s33, av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40,
               av_s41, av_s42, av_s43, av_s44, av_s45, av_s46, av_s47, av_s48;
        __m128 av_m0, av_m1, av_m2, av_m3, av_m4, av_m5, av_m6, av_m7, av_m8,
               av_m9, av_m10, av_m11, av_m12, av_m13, av_m14, av_m15, av_m16,
               av_m17, av_m18, av_m19, av_m20, av_m21, av_m22, av_m23, av_m24,
               av_m25, av_m26, av_m27, av_m28, av_m29, av_m30, av_m31, av_m32,
               av_m33, av_m34, av_m35, av_m36, av_m37, av_m38, av_m39, av_m40,
               av_m41, av_m42, av_m43, av_m44, av_m45, av_m46, av_m47, av_m48,
               av_m49;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21;

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

        av_s0 = _mm_add_ps(av_in1, av_in10);
        av_s1 = _mm_add_ps(av_in2, av_in9);
        av_s2 = _mm_add_ps(av_in3, av_in8);
        av_s3 = _mm_add_ps(av_in4, av_in7);
        av_s4 = _mm_add_ps(av_in5, av_in6);
        av_s5 = _mm_sub_ps(av_in1, av_in10);
        av_s6 = _mm_sub_ps(av_in2, av_in9);
        av_s7 = _mm_sub_ps(av_in3, av_in8);
        av_s8 = _mm_sub_ps(av_in4, av_in7);
        av_s9 = _mm_sub_ps(av_in5, av_in6);

        av_s10 = _mm_add_ps(av_s0, av_s1);
        av_s11 = _mm_add_ps(av_s2, av_s3);
        av_s12 = _mm_add_ps(av_s4, av_in0);
        av_s13 = _mm_add_ps(av_s10, av_s11);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(av_s12, av_s13);
        STHR_128_S(curr_out, v_out_stride, v_out0);

        av_m0 = _mm_mul_ps(v_CRTM_11_1, av_s0);
        av_m1 = _mm_mul_ps(v_CRTM_11_3, av_s1);
        av_m2 = _mm_mul_ps(v_CRTM_11_5, av_s2);
        av_m3 = _mm_mul_ps(v_CRTM_11_7, av_s3);
        av_m4 = _mm_mul_ps(v_CRTM_11_9, av_s4);

        av_s14 = _mm_add_ps(av_m0, av_m1);
        av_s15 = _mm_add_ps(av_m2, av_m3);
        av_s16 = _mm_sub_ps(av_in0, av_m4);
        av_s17 = _mm_sub_ps(av_s14, av_s15);

        // Output point 4: X(3)
        v_out3 = _mm_add_ps(av_s16, av_s17);

        av_m5 = _mm_mul_ps(v_CRTM_11_2, av_s5);
        av_m6 = _mm_mul_ps(v_CRTM_11_4, av_s6);
        av_m7 = _mm_mul_ps(v_CRTM_11_6, av_s7);
        av_m8 = _mm_mul_ps(v_CRTM_11_8, av_s8);
        av_m9 = _mm_mul_ps(v_CRTM_11_10, av_s9);

        av_s18 = _mm_add_ps(av_m5, av_m6);
        av_s19 = _mm_add_ps(av_m7, av_m8);
        av_s20 = _mm_add_ps(av_s19, av_m9);

        // Output point 5: X(4)
        v_out4 = NEGATE_128_S(_mm_add_ps(av_s18, av_s20));
        curr_out = out + out_strides[3];
        STHRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);

        av_m10 = _mm_mul_ps(v_CRTM_11_1, av_s4);
        av_m11 = _mm_mul_ps(v_CRTM_11_3, av_s0);
        av_m12 = _mm_mul_ps(v_CRTM_11_5, av_s3);
        av_m13 = _mm_mul_ps(v_CRTM_11_7, av_s1);
        av_m14 = _mm_mul_ps(v_CRTM_11_9, av_s2);

        av_s21 = _mm_add_ps(av_m10, av_m11);
        av_s22 = _mm_add_ps(av_m12, av_m13);
        av_s23 = _mm_sub_ps(av_in0, av_m14);
        av_s24 = _mm_sub_ps(av_s21, av_s22);

        // Output point 8: X(7)
        v_out7 = _mm_add_ps(av_s23, av_s24);

        av_m15 = _mm_mul_ps(v_CRTM_11_2, av_s9);
        av_m16 = _mm_mul_ps(v_CRTM_11_4, av_s5);
        av_m17 = _mm_mul_ps(v_CRTM_11_6, av_s8);
        av_m18 = _mm_mul_ps(v_CRTM_11_8, av_s6);
        av_m19 = _mm_mul_ps(v_CRTM_11_10, av_s7);

        av_s25 = _mm_sub_ps(av_m15, av_m16);
        av_s26 = _mm_sub_ps(av_m17, av_m18);
        av_s27 = _mm_add_ps(av_s26, av_m19);

        // Output point 9: X(8)
        v_out8 = _mm_add_ps(av_s25, av_s27);
        curr_out = out + out_strides[7];
        STHRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);

        av_m20 = _mm_mul_ps(v_CRTM_11_1, av_s3);
        av_m21 = _mm_mul_ps(v_CRTM_11_3, av_s2);
        av_m22 = _mm_mul_ps(v_CRTM_11_5, av_s0);
        av_m23 = _mm_mul_ps(v_CRTM_11_7, av_s4);
        av_m24 = _mm_mul_ps(v_CRTM_11_9, av_s1);

        av_s28 = _mm_add_ps(av_m20, av_m21);
        av_s29 = _mm_add_ps(av_m22, av_m23);
        av_s30 = _mm_sub_ps(av_in0, av_m24);
        av_s31 = _mm_sub_ps(av_s30, av_s29);

        // Output point 12: X(11)
        v_out11 = _mm_add_ps(av_s28, av_s31);

        av_m25 = _mm_mul_ps(v_CRTM_11_2, av_s8);
        av_m26 = _mm_mul_ps(v_CRTM_11_4, av_s7);
        av_m27 = _mm_mul_ps(v_CRTM_11_6, av_s5);
        av_m28 = _mm_mul_ps(v_CRTM_11_8, av_s9);
        av_m29 = _mm_mul_ps(v_CRTM_11_10, av_s6);

        av_s32 = _mm_sub_ps(av_m26, av_m25);
        av_s33 = _mm_add_ps(av_m27, av_m28);
        av_s34 = _mm_sub_ps(av_s32, av_s33);

        // Output point 13: X(12)
        v_out12 = _mm_add_ps(av_s34, av_m29);
        curr_out = out + out_strides[11];
        STHRI_2x128_S(curr_out, v_out_stride, v_out11, v_out12);

        av_m30 = _mm_mul_ps(v_CRTM_11_1, av_s2);
        av_m31 = _mm_mul_ps(v_CRTM_11_3, av_s4);
        av_m32 = _mm_mul_ps(v_CRTM_11_5, av_s1);
        av_m33 = _mm_mul_ps(v_CRTM_11_7, av_s0);
        av_m34 = _mm_mul_ps(v_CRTM_11_9, av_s3);

        av_s35 = _mm_add_ps(av_m30, av_m31);
        av_s36 = _mm_add_ps(av_m32, av_m33);
        av_s37 = _mm_sub_ps(av_in0, av_m34);
        av_s38 = _mm_sub_ps(av_s35, av_s36);

        // Output point 16: X(15)
        v_out15 = _mm_add_ps(av_s38, av_s37);

        av_m35 = _mm_mul_ps(v_CRTM_11_2, av_s7);
        av_m36 = _mm_mul_ps(v_CRTM_11_4, av_s9);
        av_m37 = _mm_mul_ps(v_CRTM_11_6, av_s6);
        av_m38 = _mm_mul_ps(v_CRTM_11_8, av_s5);
        av_m39 = _mm_mul_ps(v_CRTM_11_10, av_s8);

        av_s39 = _mm_sub_ps(av_m36, av_m35);
        av_s40 = _mm_sub_ps(av_m37, av_m38);
        av_s41 = _mm_sub_ps(av_s40, av_m39);

        // Output point 17: X(16)
        v_out16 = _mm_add_ps(av_s39, av_s41);
        curr_out = out + out_strides[15];
        STHRI_2x128_S(curr_out, v_out_stride, v_out15, v_out16);

        av_m40 = _mm_mul_ps(v_CRTM_11_1, av_s1);
        av_m41 = _mm_mul_ps(v_CRTM_11_3, av_s3);
        av_m42 = _mm_mul_ps(v_CRTM_11_5, av_s4);
        av_m43 = _mm_mul_ps(v_CRTM_11_7, av_s2);
        av_m44 = _mm_mul_ps(v_CRTM_11_9, av_s0);

        av_s42 = _mm_add_ps(av_m40, av_m41);
        av_s43 = _mm_add_ps(av_m42, av_m43);
        av_s44 = _mm_sub_ps(av_in0, av_m44);
        av_s45 = _mm_sub_ps(av_s42, av_s43);

        // Output point 20: X(19)
        v_out19 = _mm_add_ps(av_s44, av_s45);

        av_m45 = _mm_mul_ps(v_CRTM_11_2, av_s6);
        av_m46 = _mm_mul_ps(v_CRTM_11_4, av_s8);
        av_m47 = _mm_mul_ps(v_CRTM_11_6, av_s9);
        av_m48 = _mm_mul_ps(v_CRTM_11_8, av_s7);
        av_m49 = _mm_mul_ps(v_CRTM_11_10, av_s5);

        av_s46 = _mm_add_ps(av_m45, av_m46);
        av_s47 = _mm_add_ps(av_m47, av_m48);
        av_s48 = _mm_sub_ps(av_s46, av_s47);

        // Output point 21: X(20)
        v_out20 = _mm_sub_ps(av_s48, av_m49);
        curr_out = out + out_strides[19];
        STHRI_2x128_S(curr_out, v_out_stride, v_out19, v_out20);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8, bv_in9, bv_in10;
        __m128 bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
               bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16,
               bv_s17, bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24,
               bv_s25, bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32,
               bv_s33, bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40,
               bv_s41, bv_s42, bv_s43, bv_s44, bv_s45, bv_s46, bv_s47, bv_s48,
               bv_s49, bv_s50, bv_s51, bv_s52, bv_s53, bv_s54;
        __m128 bv_m0, bv_m1, bv_m2, bv_m3, bv_m4, bv_m5, bv_m6, bv_m7, bv_m8,
               bv_m9, bv_m10, bv_m11, bv_m12, bv_m13, bv_m14, bv_m15, bv_m16,
               bv_m17, bv_m18, bv_m19, bv_m20, bv_m21, bv_m22, bv_m23, bv_m24,
               bv_m25, bv_m26, bv_m27, bv_m28, bv_m29, bv_m30, bv_m31, bv_m32,
               bv_m33, bv_m34, bv_m35, bv_m36, bv_m37, bv_m38, bv_m39, bv_m40,
               bv_m41, bv_m42, bv_m43, bv_m44, bv_m45, bv_m46, bv_m47, bv_m48,
               bv_m49;

        curr_in = in + in_strides[1];
        LDHR_128_S(curr_in, v_in_stride, bv_in0);
        curr_in = in + in_strides[3];
        LDHR_128_S(curr_in, v_in_stride, bv_in1);
        curr_in = in + in_strides[5];
        LDHR_128_S(curr_in, v_in_stride, bv_in2);
        curr_in = in + in_strides[7];
        LDHR_128_S(curr_in, v_in_stride, bv_in3);
        curr_in = in + in_strides[9];
        LDHR_128_S(curr_in, v_in_stride, bv_in4);
        curr_in = in + in_strides[11];
        LDHR_128_S(curr_in, v_in_stride, bv_in5);
        curr_in = in + in_strides[13];
        LDHR_128_S(curr_in, v_in_stride, bv_in6);
        curr_in = in + in_strides[15];
        LDHR_128_S(curr_in, v_in_stride, bv_in7);
        curr_in = in + in_strides[17];
        LDHR_128_S(curr_in, v_in_stride, bv_in8);
        curr_in = in + in_strides[19];
        LDHR_128_S(curr_in, v_in_stride, bv_in9);
        curr_in = in + in_strides[21];
        LDHR_128_S(curr_in, v_in_stride, bv_in10);

        bv_s0 = _mm_add_ps(bv_in1, bv_in10);
        bv_s1 = _mm_add_ps(bv_in2, bv_in9);
        bv_s2 = _mm_add_ps(bv_in3, bv_in8);
        bv_s3 = _mm_add_ps(bv_in4, bv_in7);
        bv_s4 = _mm_add_ps(bv_in5, bv_in6);
        bv_s5 = _mm_sub_ps(bv_in1, bv_in10);
        bv_s6 = _mm_sub_ps(bv_in2, bv_in9);
        bv_s7 = _mm_sub_ps(bv_in3, bv_in8);
        bv_s8 = _mm_sub_ps(bv_in4, bv_in7);
        bv_s9 = _mm_sub_ps(bv_in5, bv_in6);

        bv_m0 = _mm_mul_ps(v_CRTM_11_9, bv_s5);
        bv_m1 = _mm_mul_ps(v_CRTM_11_1, bv_s6);
        bv_s10 = _mm_add_ps(bv_m0, bv_m1);
        bv_m2 = _mm_mul_ps(v_CRTM_11_7, bv_s7);
        bv_s11 = _mm_add_ps(bv_s10, bv_m2);
        bv_m3 = _mm_mul_ps(v_CRTM_11_3, bv_s8);
        bv_s12 = _mm_add_ps(bv_s11, bv_m3);
        bv_m4 = _mm_mul_ps(v_CRTM_11_5, bv_s9);
        bv_s13 = _mm_add_ps(bv_s12, bv_m4);

        // Output point 2: X(1)
        v_out1 = _mm_add_ps(bv_in0, bv_s13);

        bv_m5 = _mm_mul_ps(v_CRTM_11_10, bv_s0);
        bv_m6 = _mm_mul_ps(v_CRTM_11_2, bv_s1);
        bv_s14 = _mm_add_ps(bv_m5, bv_m6);
        bv_m7 = _mm_mul_ps(v_CRTM_11_8, bv_s2);
        bv_s15 = _mm_add_ps(bv_s14, bv_m7);
        bv_m8 = _mm_mul_ps(v_CRTM_11_4, bv_s3);
        bv_s16 = _mm_add_ps(bv_s15, bv_m8);
        bv_m9 = _mm_mul_ps(v_CRTM_11_6, bv_s4);
        bv_s17 = _mm_add_ps(bv_s16, bv_m9);

        // Output point 3: X(2)
        v_out2 = NEGATE_128_S(bv_s17);
        curr_out = out + out_strides[1];
        STHRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);

        bv_m10 = _mm_mul_ps(v_CRTM_11_7, bv_s5);
        bv_m11 = _mm_mul_ps(v_CRTM_11_5, bv_s6);
        bv_s18 = _mm_sub_ps(bv_m10, bv_m11);
        bv_m12 = _mm_mul_ps(v_CRTM_11_1, bv_s7);
        bv_s19 = _mm_sub_ps(bv_s18, bv_m12);
        bv_m13 = _mm_mul_ps(v_CRTM_11_9, bv_s8);
        bv_s20 = _mm_sub_ps(bv_s19, bv_m13);
        bv_m14 = _mm_mul_ps(v_CRTM_11_3, bv_s9);
        bv_s21 = _mm_sub_ps(bv_s20, bv_m14);

        // Output point 6: X(5)
        v_out5 = _mm_add_ps(bv_in0, bv_s21);

        bv_m15 = _mm_mul_ps(v_CRTM_11_8, bv_s0);
        bv_m16 = _mm_mul_ps(v_CRTM_11_6, bv_s1);
        bv_s22 = _mm_add_ps(bv_m15, bv_m16);
        bv_m17 = _mm_mul_ps(v_CRTM_11_2, bv_s2);
        bv_s23 = _mm_add_ps(bv_s22, bv_m17);
        bv_m18 = _mm_mul_ps(v_CRTM_11_10, bv_s3);
        bv_s24 = _mm_sub_ps(bv_s23, bv_m18);
        bv_m19 = _mm_mul_ps(v_CRTM_11_4, bv_s4);
        bv_s25 = _mm_sub_ps(bv_s24, bv_m19);

        // Output point 7: X(6)
        v_out6 = NEGATE_128_S(bv_s25);
        curr_out = out + out_strides[5];
        STHRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);

        bv_m20 = _mm_mul_ps(v_CRTM_11_5, bv_s5);
        bv_m21 = _mm_mul_ps(v_CRTM_11_9, bv_s6);
        bv_s26 = _mm_sub_ps(bv_m20, bv_m21);
        bv_m22 = _mm_mul_ps(v_CRTM_11_3, bv_s7);
        bv_s27 = _mm_sub_ps(bv_s26, bv_m22);
        bv_m23 = _mm_mul_ps(v_CRTM_11_1, bv_s8);
        bv_s28 = _mm_add_ps(bv_s27, bv_m23);
        bv_m24 = _mm_mul_ps(v_CRTM_11_7, bv_s9);
        bv_s29 = _mm_add_ps(bv_s28, bv_m24);

        // Output point 10: X(9)
        v_out9 = _mm_add_ps(bv_in0, bv_s29);

        bv_m25 = _mm_mul_ps(v_CRTM_11_6, bv_s0);
        bv_m26 = _mm_mul_ps(v_CRTM_11_10, bv_s1);
        bv_s30 = _mm_add_ps(bv_m25, bv_m26);
        bv_m27 = _mm_mul_ps(v_CRTM_11_4, bv_s2);
        bv_s31 = _mm_sub_ps(bv_s30, bv_m27);
        bv_m28 = _mm_mul_ps(v_CRTM_11_2, bv_s3);
        bv_s32 = _mm_sub_ps(bv_s31, bv_m28);
        bv_m29 = _mm_mul_ps(v_CRTM_11_8, bv_s4);
        bv_s33 = _mm_add_ps(bv_s32, bv_m29);

        // Output point 11: X(10)
        v_out10 = NEGATE_128_S(bv_s33);
        curr_out = out + out_strides[9];
        STHRI_2x128_S(curr_out, v_out_stride, v_out9, v_out10);

        bv_m30 = _mm_mul_ps(v_CRTM_11_3, bv_s5);
        bv_m31 = _mm_mul_ps(v_CRTM_11_7, bv_s6);
        bv_s34 = _mm_add_ps(bv_m30, bv_m31);
        bv_m32 = _mm_mul_ps(v_CRTM_11_9, bv_s7);
        bv_s35 = _mm_sub_ps(bv_m32, bv_s34);
        bv_m33 = _mm_mul_ps(v_CRTM_11_5, bv_s8);
        bv_s36 = _mm_sub_ps(bv_s35, bv_m33);
        bv_m34 = _mm_mul_ps(v_CRTM_11_1, bv_s9);
        bv_s37 = _mm_sub_ps(bv_s36, bv_m34);

        // Output point 14: X(13)
        v_out13 = _mm_add_ps(bv_in0, bv_s37);

        bv_m35 = _mm_mul_ps(v_CRTM_11_4, bv_s0);
        bv_m36 = _mm_mul_ps(v_CRTM_11_8, bv_s1);
        bv_s38 = _mm_sub_ps(bv_m35, bv_m36);
        bv_m37 = _mm_mul_ps(v_CRTM_11_10, bv_s2);
        bv_s39 = _mm_sub_ps(bv_s38, bv_m37);
        bv_m38 = _mm_mul_ps(v_CRTM_11_6, bv_s3);
        bv_s40 = _mm_add_ps(bv_s39, bv_m38);
        bv_m39 = _mm_mul_ps(v_CRTM_11_2, bv_s4);
        bv_s41 = _mm_sub_ps(bv_s40, bv_m39);

        // Output point 15: X(14)
        v_out14 = NEGATE_128_S(bv_s41);
        curr_out = out + out_strides[13];
        STHRI_2x128_S(curr_out, v_out_stride, v_out13, v_out14);

        bv_m40 = _mm_mul_ps(v_CRTM_11_1, bv_s5);
        bv_m41 = _mm_mul_ps(v_CRTM_11_3, bv_s6);
        bv_s42 = _mm_sub_ps(bv_m41, bv_m40);
        bv_m42 = _mm_mul_ps(v_CRTM_11_5, bv_s7);
        bv_s43 = _mm_add_ps(bv_s42, bv_m42);
        bv_m43 = _mm_mul_ps(v_CRTM_11_7, bv_s8);
        bv_s44 = _mm_sub_ps(bv_s43, bv_m43);
        bv_m44 = _mm_mul_ps(v_CRTM_11_9, bv_s9);
        bv_s45 = _mm_add_ps(bv_s44, bv_m44);

        // Output point 18: X(17)
        v_out17 = _mm_add_ps(bv_in0, bv_s45);

        bv_m45 = _mm_mul_ps(v_CRTM_11_2, bv_s0);
        bv_m46 = _mm_mul_ps(v_CRTM_11_4, bv_s1);
        bv_s46 = _mm_sub_ps(bv_m45, bv_m46);
        bv_m47 = _mm_mul_ps(v_CRTM_11_6, bv_s2);
        bv_s47 = _mm_add_ps(bv_s46, bv_m47);
        bv_m48 = _mm_mul_ps(v_CRTM_11_8, bv_s3);
        bv_s48 = _mm_sub_ps(bv_s47, bv_m48);
        bv_m49 = _mm_mul_ps(v_CRTM_11_10, bv_s4);
        bv_s49 = _mm_add_ps(bv_s48, bv_m49);

        // Output point 19: X(18)
        v_out18 = NEGATE_128_S(bv_s49);
        curr_out = out + out_strides[17];
        STHRI_2x128_S(curr_out, v_out_stride, v_out17, v_out18);

        bv_s50 = _mm_sub_ps(bv_in0, bv_s5);
        bv_s51 = _mm_add_ps(bv_s50, bv_s6);
        bv_s52 = _mm_sub_ps(bv_s51, bv_s7);
        bv_s53 = _mm_add_ps(bv_s52, bv_s8);
        bv_s54 = _mm_sub_ps(bv_s53, bv_s9);

        // Output point 22: X(21)
        v_out21 = bv_s54;
        curr_out = out + out_strides[21];
        STHR_128_S(curr_out, v_out_stride, v_out21);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
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

        in += v_in_stride;
        out += v_out_stride;

    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft11avx128_fp32_bwd(FFTZ_VOID *in_real,
                                             FFTZ_VOID *in_imag,
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
    FFTZ_UINT8 is_contiguous_in = (v_in_stride == 1);
    FFTZ_UINT8 is_contiguous_out = (v_out_stride == 1);

    FFTZ_INTP cnt;
    FFTZ_FLOAT *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_128_S;

    __m128 v_CRTM_11_1 = _mm_broadcast_ss(&CRTM_11_1);
    __m128 v_CRTM_11_2 = _mm_broadcast_ss(&CRTM_11_2);
    __m128 v_CRTM_11_3 = _mm_broadcast_ss(&CRTM_11_3);
    __m128 v_CRTM_11_4 = _mm_broadcast_ss(&CRTM_11_4);
    __m128 v_CRTM_11_5 = _mm_broadcast_ss(&CRTM_11_5);
    __m128 v_CRTM_11_6 = _mm_broadcast_ss(&CRTM_11_6);
    __m128 v_CRTM_11_7 = _mm_broadcast_ss(&CRTM_11_7);
    __m128 v_CRTM_11_8 = _mm_broadcast_ss(&CRTM_11_8);
    __m128 v_CRTM_11_9 = _mm_broadcast_ss(&CRTM_11_9);
    __m128 v_CRTM_11_10 = _mm_broadcast_ss(&CRTM_11_10);
    __m128 v_CRTM_11_11 = _mm_broadcast_ss(&CRTM_11_11);

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
               av_in8, av_in9, av_in10;
        __m128 av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
               av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16,
               av_s17, av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24,
               av_s25, av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32,
               av_s33, av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40,
               av_s41, av_s42, av_s43, av_s44, av_s45, av_s46, av_s47, av_s48;
        __m128 av_m0, av_m1, av_m2, av_m3, av_m4, av_m5, av_m6, av_m7, av_m8,
               av_m9, av_m10, av_m11, av_m12, av_m13, av_m14, av_m15, av_m16,
               av_m17, av_m18, av_m19, av_m20, av_m21, av_m22, av_m23, av_m24,
               av_m25, av_m26, av_m27, av_m28, av_m29, av_m30, av_m31, av_m32,
               av_m33, av_m34, av_m35, av_m36, av_m37, av_m38, av_m39, av_m40,
               av_m41, av_m42, av_m43, av_m44, av_m45, av_m46, av_m47, av_m48,
               av_m49, av_m50;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21;

        curr_in = in;
        curr_out = out;

        // Input point 1: X(0)
        LDR_128_S(curr_in, v_in_stride, av_in0, is_contiguous_in);
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

        av_s0 = _mm_add_ps(av_in1, av_in3);
        av_s1 = _mm_add_ps(av_in5, av_in7);
        av_s2 = _mm_add_ps(av_s0, av_s1);
        av_s3 = _mm_add_ps(av_s2, av_in9);
        av_m0 = _mm_mul_ps(v_CRTM_11_11, av_s3);

        // Output point 1: x(0)
        v_out0 = _mm_add_ps(av_in0, av_m0);
        STR_128_S(curr_out, v_out_stride, v_out0, is_contiguous_out);

        av_m1 = _mm_mul_ps(v_CRTM_11_1, av_in1);
        av_m2 = _mm_mul_ps(v_CRTM_11_2, av_in2);
        av_m3 = _mm_mul_ps(v_CRTM_11_3, av_in3);
        av_m4 = _mm_mul_ps(v_CRTM_11_4, av_in4);
        av_m5 = _mm_mul_ps(v_CRTM_11_5, av_in5);
        av_m6 = _mm_mul_ps(v_CRTM_11_6, av_in6);
        av_m7 = _mm_mul_ps(v_CRTM_11_7, av_in7);
        av_m8 = _mm_mul_ps(v_CRTM_11_8, av_in8);
        av_m9 = _mm_mul_ps(v_CRTM_11_9, av_in9);
        av_m10 = _mm_mul_ps(v_CRTM_11_10, av_in10);

        av_s4 = _mm_add_ps(av_m1, av_m3);
        av_s5 = _mm_add_ps(av_s4, av_in0);
        av_s6 = _mm_add_ps(av_m5, av_m7);
        av_s7 = _mm_add_ps(av_s6, av_m9);
        av_s8 = _mm_sub_ps(av_s5, av_s7);
        av_s9 = _mm_add_ps(av_m2, av_m4);
        av_s10 = _mm_add_ps(av_m6, av_m8);
        av_s11 = _mm_add_ps(av_s9, av_m10);
        av_s12 = _mm_add_ps(av_s10, av_s11);

        // Output point 3: x(2)
        v_out2 = _mm_sub_ps(av_s8, av_s12);
        curr_out = out + out_strides[2];
        STR_128_S(curr_out, v_out_stride, v_out2, is_contiguous_out);
        // Output point 21: x(20)
        v_out20 = _mm_add_ps(av_s8, av_s12);
        curr_out = out + out_strides[20];
        STR_128_S(curr_out, v_out_stride, v_out20, is_contiguous_out);

        av_m11 = _mm_mul_ps(v_CRTM_11_1, av_in9);
        av_m12 = _mm_mul_ps(v_CRTM_11_2, av_in10);
        av_m13 = _mm_mul_ps(v_CRTM_11_3, av_in1);
        av_m14 = _mm_mul_ps(v_CRTM_11_4, av_in2);
        av_m15 = _mm_mul_ps(v_CRTM_11_5, av_in7);
        av_m16 = _mm_mul_ps(v_CRTM_11_6, av_in8);
        av_m17 = _mm_mul_ps(v_CRTM_11_7, av_in3);
        av_m18 = _mm_mul_ps(v_CRTM_11_8, av_in4);
        av_m19 = _mm_mul_ps(v_CRTM_11_9, av_in5);
        av_m20 = _mm_mul_ps(v_CRTM_11_10, av_in6);

        av_s13 = _mm_add_ps(av_m11, av_m13);
        av_s14 = _mm_add_ps(av_s13, av_in0);
        av_s15 = _mm_add_ps(av_m15, av_m17);
        av_s16 = _mm_add_ps(av_s15, av_m19);
        av_s17 = _mm_sub_ps(av_s14, av_s16);
        av_s18 = _mm_sub_ps(av_m12, av_m14);
        av_s19 = _mm_sub_ps(av_m16, av_m18);
        av_s20 = _mm_add_ps(av_s19, av_m20);
        av_s21 = _mm_add_ps(av_s18, av_s20);

        // Output point 5: x(4)
        v_out4 = _mm_add_ps(av_s17, av_s21);
        curr_out = out + out_strides[4];
        STR_128_S(curr_out, v_out_stride, v_out4, is_contiguous_out);
        // Output point 19: x(18)
        v_out18 = _mm_sub_ps(av_s17, av_s21);
        curr_out = out + out_strides[18];
        STR_128_S(curr_out, v_out_stride, v_out18, is_contiguous_out);

        av_m21 = _mm_mul_ps(v_CRTM_11_1, av_in7);
        av_m22 = _mm_mul_ps(v_CRTM_11_2, av_in8);
        av_m23 = _mm_mul_ps(v_CRTM_11_3, av_in5);
        av_m24 = _mm_mul_ps(v_CRTM_11_4, av_in6);
        av_m25 = _mm_mul_ps(v_CRTM_11_5, av_in1);
        av_m26 = _mm_mul_ps(v_CRTM_11_6, av_in2);
        av_m27 = _mm_mul_ps(v_CRTM_11_7, av_in9);
        av_m28 = _mm_mul_ps(v_CRTM_11_8, av_in10);
        av_m29 = _mm_mul_ps(v_CRTM_11_9, av_in3);
        av_m30 = _mm_mul_ps(v_CRTM_11_10, av_in4);

        av_s22 = _mm_add_ps(av_m21, av_m23);
        av_s23 = _mm_add_ps(av_s22, av_in0);
        av_s24 = _mm_add_ps(av_m25, av_m27);
        av_s25 = _mm_add_ps(av_s24, av_m29);
        av_s26 = _mm_sub_ps(av_s23, av_s25);
        av_s27 = _mm_sub_ps(av_m22, av_m24);
        av_s28 = _mm_add_ps(av_m26, av_m28);
        av_s29 = _mm_sub_ps(av_s28, av_m30);
        av_s30 = _mm_add_ps(av_s27, av_s29);

        // Output point 7: x(6)
        v_out6 = _mm_sub_ps(av_s26, av_s30);
        curr_out = out + out_strides[6];
        STR_128_S(curr_out, v_out_stride, v_out6, is_contiguous_out);
        // Output point 17: x(16)
        v_out16 = _mm_add_ps(av_s26, av_s30);
        curr_out = out + out_strides[16];
        STR_128_S(curr_out, v_out_stride, v_out16, is_contiguous_out);

        av_m31 = _mm_mul_ps(v_CRTM_11_1, av_in5);
        av_m32 = _mm_mul_ps(v_CRTM_11_2, av_in6);
        av_m33 = _mm_mul_ps(v_CRTM_11_3, av_in9);
        av_m34 = _mm_mul_ps(v_CRTM_11_4, av_in10);
        av_m35 = _mm_mul_ps(v_CRTM_11_5, av_in3);
        av_m36 = _mm_mul_ps(v_CRTM_11_6, av_in4);
        av_m37 = _mm_mul_ps(v_CRTM_11_7, av_in1);
        av_m38 = _mm_mul_ps(v_CRTM_11_8, av_in2);
        av_m39 = _mm_mul_ps(v_CRTM_11_9, av_in7);
        av_m40 = _mm_mul_ps(v_CRTM_11_10, av_in8);

        av_s31 = _mm_add_ps(av_m31, av_m33);
        av_s32 = _mm_add_ps(av_s31, av_in0);
        av_s33 = _mm_add_ps(av_m35, av_m37);
        av_s34 = _mm_add_ps(av_s33, av_m39);
        av_s35 = _mm_sub_ps(av_s32, av_s34);
        av_s36 = _mm_sub_ps(av_m32, av_m34);
        av_s37 = _mm_sub_ps(av_m38, av_m36);
        av_s38 = _mm_add_ps(av_s37, av_m40);
        av_s39 = _mm_add_ps(av_s36, av_s38);

        // Output point 9: x(8)
        v_out8 = _mm_sub_ps(av_s35, av_s39);
        curr_out = out + out_strides[8];
        STR_128_S(curr_out, v_out_stride, v_out8, is_contiguous_out);
        // Output point 15: x(14)
        v_out14 = _mm_add_ps(av_s35, av_s39);
        curr_out = out + out_strides[14];
        STR_128_S(curr_out, v_out_stride, v_out14, is_contiguous_out);

        av_m41 = _mm_mul_ps(v_CRTM_11_1, av_in3);
        av_m42 = _mm_mul_ps(v_CRTM_11_2, av_in4);
        av_m43 = _mm_mul_ps(v_CRTM_11_3, av_in7);
        av_m44 = _mm_mul_ps(v_CRTM_11_4, av_in8);
        av_m45 = _mm_mul_ps(v_CRTM_11_5, av_in9);
        av_m46 = _mm_mul_ps(v_CRTM_11_6, av_in10);
        av_m47 = _mm_mul_ps(v_CRTM_11_7, av_in5);
        av_m48 = _mm_mul_ps(v_CRTM_11_8, av_in6);
        av_m49 = _mm_mul_ps(v_CRTM_11_9, av_in1);
        av_m50 = _mm_mul_ps(v_CRTM_11_10, av_in2);

        av_s40 = _mm_add_ps(av_m41, av_m43);
        av_s41 = _mm_add_ps(av_s40, av_in0);
        av_s42 = _mm_add_ps(av_m45, av_m47);
        av_s43 = _mm_add_ps(av_s42, av_m49);
        av_s44 = _mm_sub_ps(av_s41, av_s43);
        av_s45 = _mm_add_ps(av_m42, av_m44);
        av_s46 = _mm_add_ps(av_m46, av_m48);
        av_s47 = _mm_sub_ps(av_s45, av_s46);
        av_s48 = _mm_sub_ps(av_s47, av_m50);

        // Output point 11: x(10)
        v_out10 = _mm_add_ps(av_s44, av_s48);
        curr_out = out + out_strides[10];
        STR_128_S(curr_out, v_out_stride, v_out10, is_contiguous_out);
        // Output point 13: x(12)
        v_out12 = _mm_sub_ps(av_s44, av_s48);
        curr_out = out + out_strides[12];
        STR_128_S(curr_out, v_out_stride, v_out12, is_contiguous_out);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8, bv_in9, bv_in10;
        __m128 bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
               bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16,
               bv_s17, bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24,
               bv_s25, bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32,
               bv_s33, bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40,
               bv_s41, bv_s42, bv_s43, bv_s44, bv_s45, bv_s46, bv_s47, bv_s48;
        __m128 bv_m0, bv_m1, bv_m2, bv_m3, bv_m4, bv_m5, bv_m6, bv_m7, bv_m8,
               bv_m9, bv_m10, bv_m11, bv_m12, bv_m13, bv_m14, bv_m15, bv_m16,
               bv_m17, bv_m18, bv_m19, bv_m20, bv_m21, bv_m22, bv_m23, bv_m24,
               bv_m25, bv_m26, bv_m27, bv_m28, bv_m29, bv_m30, bv_m31, bv_m32,
               bv_m33, bv_m34, bv_m35, bv_m36, bv_m37, bv_m38, bv_m39, bv_m40,
               bv_m41, bv_m42, bv_m43, bv_m44, bv_m45, bv_m46, bv_m47, bv_m48,
               bv_m49, bv_m50;

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
        // Input point 22: X(21)
        curr_in = in + in_strides[21];
        LDR_128_S(curr_in, v_in_stride, bv_in10, is_contiguous_in);

        bv_s0 = _mm_add_ps(bv_in0, bv_in2);
        bv_s1 = _mm_add_ps(bv_in4, bv_in6);
        bv_s2 = _mm_add_ps(bv_s0, bv_s1);
        bv_s3 = _mm_add_ps(bv_s2, bv_in8);
        bv_m0 = _mm_mul_ps(v_CRTM_11_11, bv_s3);

        // Output point 2: x(1)
        v_out1 = _mm_add_ps(bv_m0, bv_in10);
        curr_out = out + out_strides[1];
        STR_128_S(curr_out, v_out_stride, v_out1, is_contiguous_out);

        bv_m1 = _mm_mul_ps(v_CRTM_11_9, bv_in0);
        bv_m2 = _mm_mul_ps(v_CRTM_11_10, bv_in1);
        bv_m3 = _mm_mul_ps(v_CRTM_11_1, bv_in8);
        bv_m4 = _mm_mul_ps(v_CRTM_11_2, bv_in9);
        bv_m5 = _mm_mul_ps(v_CRTM_11_7, bv_in2);
        bv_m6 = _mm_mul_ps(v_CRTM_11_8, bv_in3);
        bv_m7 = _mm_mul_ps(v_CRTM_11_3, bv_in6);
        bv_m8 = _mm_mul_ps(v_CRTM_11_4, bv_in7);
        bv_m9 = _mm_mul_ps(v_CRTM_11_5, bv_in4);
        bv_m10 = _mm_mul_ps(v_CRTM_11_6, bv_in5);

        bv_s4 = _mm_add_ps(bv_m2, bv_m4);
        bv_s5 = _mm_add_ps(bv_m6, bv_m8);
        bv_s6 = _mm_add_ps(bv_s4, bv_s5);
        bv_s7 = _mm_add_ps(bv_s6, bv_m10);
        bv_s8 = _mm_sub_ps(bv_m1, bv_m3);
        bv_s9 = _mm_sub_ps(bv_m5, bv_m7);
        bv_s10 = _mm_add_ps(bv_s8, bv_s9);
        bv_s11 = _mm_sub_ps(bv_m9, bv_in10);
        bv_s12 = _mm_add_ps(bv_s10, bv_s11);

        // Output point 4: x(3)
        v_out3 = _mm_sub_ps(bv_s12, bv_s7);
        curr_out = out + out_strides[3];
        STR_128_S(curr_out, v_out_stride, v_out3, is_contiguous_out);
        // Output point 22: x(21)
        v_out21 = NEGATE_128_S(_mm_add_ps(bv_s12, bv_s7));
        curr_out = out + out_strides[21];
        STR_128_S(curr_out, v_out_stride, v_out21, is_contiguous_out);

        bv_m11 = _mm_mul_ps(v_CRTM_11_1, bv_in0);
        bv_m12 = _mm_mul_ps(v_CRTM_11_2, bv_in1);
        bv_m13 = _mm_mul_ps(v_CRTM_11_5, bv_in2);
        bv_m14 = _mm_mul_ps(v_CRTM_11_6, bv_in3);
        bv_m15 = _mm_mul_ps(v_CRTM_11_9, bv_in4);
        bv_m16 = _mm_mul_ps(v_CRTM_11_10, bv_in5);
        bv_m17 = _mm_mul_ps(v_CRTM_11_7, bv_in6);
        bv_m18 = _mm_mul_ps(v_CRTM_11_8, bv_in7);
        bv_m19 = _mm_mul_ps(v_CRTM_11_3, bv_in8);
        bv_m20 = _mm_mul_ps(v_CRTM_11_4, bv_in9);

        bv_s13 = _mm_add_ps(bv_m12, bv_m16);
        bv_s14 = _mm_sub_ps(bv_m14, bv_m20);
        bv_s15 = _mm_add_ps(bv_s13, bv_s14);
        bv_s16 = _mm_sub_ps(bv_s15, bv_m18);
        bv_s17 = _mm_add_ps(bv_m11, bv_m19);
        bv_s18 = _mm_add_ps(bv_m13, bv_m17);
        bv_s19 = _mm_sub_ps(bv_s17, bv_s18);
        bv_s20 = _mm_sub_ps(bv_in10, bv_m15);
        bv_s21 = _mm_add_ps(bv_s19, bv_s20);

        // Output point 6: x(5)
        v_out5 = _mm_sub_ps(bv_s21, bv_s16);
        curr_out = out + out_strides[5];
        STR_128_S(curr_out, v_out_stride, v_out5, is_contiguous_out);
        // Output point 20: x(19)
        v_out19 = NEGATE_128_S(_mm_add_ps(bv_s21, bv_s16));
        curr_out = out + out_strides[19];
        STR_128_S(curr_out, v_out_stride, v_out19, is_contiguous_out);

        bv_m21 = _mm_mul_ps(v_CRTM_11_7, bv_in0);
        bv_m22 = _mm_mul_ps(v_CRTM_11_8, bv_in1);
        bv_m23 = _mm_mul_ps(v_CRTM_11_1, bv_in2);
        bv_m24 = _mm_mul_ps(v_CRTM_11_2, bv_in3);
        bv_m25 = _mm_mul_ps(v_CRTM_11_3, bv_in4);
        bv_m26 = _mm_mul_ps(v_CRTM_11_4, bv_in5);
        bv_m27 = _mm_mul_ps(v_CRTM_11_9, bv_in6);
        bv_m28 = _mm_mul_ps(v_CRTM_11_10, bv_in7);
        bv_m29 = _mm_mul_ps(v_CRTM_11_5, bv_in8);
        bv_m30 = _mm_mul_ps(v_CRTM_11_6, bv_in9);

        bv_s22 = _mm_add_ps(bv_m22, bv_m24);
        bv_s23 = _mm_sub_ps(bv_m30, bv_m26);
        bv_s24 = _mm_add_ps(bv_s22, bv_s23);
        bv_s25 = _mm_sub_ps(bv_s24, bv_m28);
        bv_s26 = _mm_add_ps(bv_m21, bv_m29);
        bv_s27 = _mm_sub_ps(bv_m27, bv_m23);
        bv_s28 = _mm_add_ps(bv_s26, bv_s27);
        bv_s29 = _mm_add_ps(bv_m25, bv_in10);
        bv_s30 = _mm_sub_ps(bv_s28, bv_s29);

        // Output point 8: x(7)
        v_out7 = _mm_sub_ps(bv_s30, bv_s25);
        curr_out = out + out_strides[7];
        STR_128_S(curr_out, v_out_stride, v_out7, is_contiguous_out);
        // Output point 18: x(17)
        v_out17 = NEGATE_128_S(_mm_add_ps(bv_s30, bv_s25));
        curr_out = out + out_strides[17];
        STR_128_S(curr_out, v_out_stride, v_out17, is_contiguous_out);

        bv_m31 = _mm_mul_ps(v_CRTM_11_3, bv_in0);
        bv_m32 = _mm_mul_ps(v_CRTM_11_4, bv_in1);
        bv_m33 = _mm_mul_ps(v_CRTM_11_9, bv_in2);
        bv_m34 = _mm_mul_ps(v_CRTM_11_10, bv_in3);
        bv_m35 = _mm_mul_ps(v_CRTM_11_1, bv_in4);
        bv_m36 = _mm_mul_ps(v_CRTM_11_2, bv_in5);
        bv_m37 = _mm_mul_ps(v_CRTM_11_5, bv_in6);
        bv_m38 = _mm_mul_ps(v_CRTM_11_6, bv_in7);
        bv_m39 = _mm_mul_ps(v_CRTM_11_7, bv_in8);
        bv_m40 = _mm_mul_ps(v_CRTM_11_8, bv_in9);

        bv_s31 = _mm_sub_ps(bv_m32, bv_m34);
        bv_s32 = _mm_sub_ps(bv_m38, bv_m40);
        bv_s33 = _mm_add_ps(bv_s31, bv_s32);
        bv_s34 = _mm_sub_ps(bv_s33, bv_m36);
        bv_s35 = _mm_add_ps(bv_m31, bv_m35);
        bv_s36 = _mm_add_ps(bv_m37, bv_m39);
        bv_s37 = _mm_sub_ps(bv_s35, bv_s36);
        bv_s38 = _mm_sub_ps(bv_in10, bv_m33);
        bv_s39 = _mm_add_ps(bv_s37, bv_s38);

        // Output point 10: x(9)
        v_out9 = _mm_sub_ps(bv_s39, bv_s34);
        curr_out = out + out_strides[9];
        STR_128_S(curr_out, v_out_stride, v_out9, is_contiguous_out);
        // Output point 16: x(15)
        v_out15 = NEGATE_128_S(_mm_add_ps(bv_s39, bv_s34));
        curr_out = out + out_strides[15];
        STR_128_S(curr_out, v_out_stride, v_out15, is_contiguous_out);

        bv_m41 = _mm_mul_ps(v_CRTM_11_5, bv_in0);
        bv_m42 = _mm_mul_ps(v_CRTM_11_6, bv_in1);
        bv_m43 = _mm_mul_ps(v_CRTM_11_3, bv_in2);
        bv_m44 = _mm_mul_ps(v_CRTM_11_4, bv_in3);
        bv_m45 = _mm_mul_ps(v_CRTM_11_7, bv_in4);
        bv_m46 = _mm_mul_ps(v_CRTM_11_8, bv_in5);
        bv_m47 = _mm_mul_ps(v_CRTM_11_1, bv_in6);
        bv_m48 = _mm_mul_ps(v_CRTM_11_2, bv_in7);
        bv_m49 = _mm_mul_ps(v_CRTM_11_9, bv_in8);
        bv_m50 = _mm_mul_ps(v_CRTM_11_10, bv_in9);

        bv_s40 = _mm_sub_ps(bv_m42, bv_m44);
        bv_s41 = _mm_add_ps(bv_m46, bv_m50);
        bv_s42 = _mm_add_ps(bv_s40, bv_s41);
        bv_s43 = _mm_sub_ps(bv_s42, bv_m48);
        bv_s44 = _mm_add_ps(bv_m41, bv_m45);
        bv_s45 = _mm_add_ps(bv_m43, bv_m47);
        bv_s46 = _mm_sub_ps(bv_s44, bv_s45);
        bv_s47 = _mm_sub_ps(bv_m49, bv_in10);
        bv_s48 = _mm_add_ps(bv_s46, bv_s47);

        // Output point 12: x(11)
        v_out11 = _mm_sub_ps(bv_s48, bv_s43);
        curr_out = out + out_strides[11];
        STR_128_S(curr_out, v_out_stride, v_out11, is_contiguous_out);
        // Output point 14: x(13)
        v_out13 = NEGATE_128_S(_mm_add_ps(bv_s48, bv_s43));
        curr_out = out + out_strides[13];
        STR_128_S(curr_out, v_out_stride, v_out13, is_contiguous_out);

        in += v_in_stride * NUM_SETS_REAL_128_S;
        out += v_out_stride * NUM_SETS_REAL_128_S;
    }
    // tail cases
    if (n & 2)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
               av_in8, av_in9, av_in10;
        __m128 av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
               av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16,
               av_s17, av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24,
               av_s25, av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32,
               av_s33, av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40,
               av_s41, av_s42, av_s43, av_s44, av_s45, av_s46, av_s47, av_s48;
        __m128 av_m0, av_m1, av_m2, av_m3, av_m4, av_m5, av_m6, av_m7, av_m8,
               av_m9, av_m10, av_m11, av_m12, av_m13, av_m14, av_m15, av_m16,
               av_m17, av_m18, av_m19, av_m20, av_m21, av_m22, av_m23, av_m24,
               av_m25, av_m26, av_m27, av_m28, av_m29, av_m30, av_m31, av_m32,
               av_m33, av_m34, av_m35, av_m36, av_m37, av_m38, av_m39, av_m40,
               av_m41, av_m42, av_m43, av_m44, av_m45, av_m46, av_m47, av_m48,
               av_m49, av_m50;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21;

        curr_in = in;
        curr_out = out;

        // Input point 1: X(0)
        LDHR_128_S(curr_in, v_in_stride, av_in0);
        // Input point 4: X(3) & Input point 5: X(4)
        curr_in = in + in_strides[3];
        LDHRI_2x128_S(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: X(7) & Input point 9: X(8)
        curr_in = in + in_strides[7];
        LDHRI_2x128_S(curr_in, v_in_stride, av_in3, av_in4);
        // Input point 12: X(11) & Input point 13: X(12)
        curr_in = in + in_strides[11];
        LDHRI_2x128_S(curr_in, v_in_stride, av_in5, av_in6);
        // Input point 16: X(15) & Input point 17: X(16)
        curr_in = in + in_strides[15];
        LDHRI_2x128_S(curr_in, v_in_stride, av_in7, av_in8);
        // Input point 20: X(19) & Input point 21: X(20)
        curr_in = in + in_strides[19];
        LDHRI_2x128_S(curr_in, v_in_stride, av_in9, av_in10);

        av_s0 = _mm_add_ps(av_in1, av_in3);
        av_s1 = _mm_add_ps(av_in5, av_in7);
        av_s2 = _mm_add_ps(av_s0, av_s1);
        av_s3 = _mm_add_ps(av_s2, av_in9);
        av_m0 = _mm_mul_ps(v_CRTM_11_11, av_s3);

        // Output point 1: x(0)
        v_out0 = _mm_add_ps(av_in0, av_m0);
        STHR_128_S(curr_out, v_out_stride, v_out0);

        av_m1 = _mm_mul_ps(v_CRTM_11_1, av_in1);
        av_m2 = _mm_mul_ps(v_CRTM_11_2, av_in2);
        av_m3 = _mm_mul_ps(v_CRTM_11_3, av_in3);
        av_m4 = _mm_mul_ps(v_CRTM_11_4, av_in4);
        av_m5 = _mm_mul_ps(v_CRTM_11_5, av_in5);
        av_m6 = _mm_mul_ps(v_CRTM_11_6, av_in6);
        av_m7 = _mm_mul_ps(v_CRTM_11_7, av_in7);
        av_m8 = _mm_mul_ps(v_CRTM_11_8, av_in8);
        av_m9 = _mm_mul_ps(v_CRTM_11_9, av_in9);
        av_m10 = _mm_mul_ps(v_CRTM_11_10, av_in10);

        av_s4 = _mm_add_ps(av_m1, av_m3);
        av_s5 = _mm_add_ps(av_s4, av_in0);
        av_s6 = _mm_add_ps(av_m5, av_m7);
        av_s7 = _mm_add_ps(av_s6, av_m9);
        av_s8 = _mm_sub_ps(av_s5, av_s7);
        av_s9 = _mm_add_ps(av_m2, av_m4);
        av_s10 = _mm_add_ps(av_m6, av_m8);
        av_s11 = _mm_add_ps(av_s9, av_m10);
        av_s12 = _mm_add_ps(av_s10, av_s11);

        // Output point 3: x(2)
        v_out2 = _mm_sub_ps(av_s8, av_s12);
        curr_out = out + out_strides[2];
        STHR_128_S(curr_out, v_out_stride, v_out2);
        // Output point 21: x(20)
        v_out20 = _mm_add_ps(av_s8, av_s12);
        curr_out = out + out_strides[20];
        STHR_128_S(curr_out, v_out_stride, v_out20);

        av_m11 = _mm_mul_ps(v_CRTM_11_1, av_in9);
        av_m12 = _mm_mul_ps(v_CRTM_11_2, av_in10);
        av_m13 = _mm_mul_ps(v_CRTM_11_3, av_in1);
        av_m14 = _mm_mul_ps(v_CRTM_11_4, av_in2);
        av_m15 = _mm_mul_ps(v_CRTM_11_5, av_in7);
        av_m16 = _mm_mul_ps(v_CRTM_11_6, av_in8);
        av_m17 = _mm_mul_ps(v_CRTM_11_7, av_in3);
        av_m18 = _mm_mul_ps(v_CRTM_11_8, av_in4);
        av_m19 = _mm_mul_ps(v_CRTM_11_9, av_in5);
        av_m20 = _mm_mul_ps(v_CRTM_11_10, av_in6);

        av_s13 = _mm_add_ps(av_m11, av_m13);
        av_s14 = _mm_add_ps(av_s13, av_in0);
        av_s15 = _mm_add_ps(av_m15, av_m17);
        av_s16 = _mm_add_ps(av_s15, av_m19);
        av_s17 = _mm_sub_ps(av_s14, av_s16);
        av_s18 = _mm_sub_ps(av_m12, av_m14);
        av_s19 = _mm_sub_ps(av_m16, av_m18);
        av_s20 = _mm_add_ps(av_s19, av_m20);
        av_s21 = _mm_add_ps(av_s18, av_s20);

        // Output point 5: x(4)
        v_out4 = _mm_add_ps(av_s17, av_s21);
        curr_out = out + out_strides[4];
        STHR_128_S(curr_out, v_out_stride, v_out4);
        // Output point 19: x(18)
        v_out18 = _mm_sub_ps(av_s17, av_s21);
        curr_out = out + out_strides[18];
        STHR_128_S(curr_out, v_out_stride, v_out18);

        av_m21 = _mm_mul_ps(v_CRTM_11_1, av_in7);
        av_m22 = _mm_mul_ps(v_CRTM_11_2, av_in8);
        av_m23 = _mm_mul_ps(v_CRTM_11_3, av_in5);
        av_m24 = _mm_mul_ps(v_CRTM_11_4, av_in6);
        av_m25 = _mm_mul_ps(v_CRTM_11_5, av_in1);
        av_m26 = _mm_mul_ps(v_CRTM_11_6, av_in2);
        av_m27 = _mm_mul_ps(v_CRTM_11_7, av_in9);
        av_m28 = _mm_mul_ps(v_CRTM_11_8, av_in10);
        av_m29 = _mm_mul_ps(v_CRTM_11_9, av_in3);
        av_m30 = _mm_mul_ps(v_CRTM_11_10, av_in4);

        av_s22 = _mm_add_ps(av_m21, av_m23);
        av_s23 = _mm_add_ps(av_s22, av_in0);
        av_s24 = _mm_add_ps(av_m25, av_m27);
        av_s25 = _mm_add_ps(av_s24, av_m29);
        av_s26 = _mm_sub_ps(av_s23, av_s25);
        av_s27 = _mm_sub_ps(av_m22, av_m24);
        av_s28 = _mm_add_ps(av_m26, av_m28);
        av_s29 = _mm_sub_ps(av_s28, av_m30);
        av_s30 = _mm_add_ps(av_s27, av_s29);

        // Output point 7: x(6)
        v_out6 = _mm_sub_ps(av_s26, av_s30);
        curr_out = out + out_strides[6];
        STHR_128_S(curr_out, v_out_stride, v_out6);
        // Output point 17: x(16)
        v_out16 = _mm_add_ps(av_s26, av_s30);
        curr_out = out + out_strides[16];
        STHR_128_S(curr_out, v_out_stride, v_out16);

        av_m31 = _mm_mul_ps(v_CRTM_11_1, av_in5);
        av_m32 = _mm_mul_ps(v_CRTM_11_2, av_in6);
        av_m33 = _mm_mul_ps(v_CRTM_11_3, av_in9);
        av_m34 = _mm_mul_ps(v_CRTM_11_4, av_in10);
        av_m35 = _mm_mul_ps(v_CRTM_11_5, av_in3);
        av_m36 = _mm_mul_ps(v_CRTM_11_6, av_in4);
        av_m37 = _mm_mul_ps(v_CRTM_11_7, av_in1);
        av_m38 = _mm_mul_ps(v_CRTM_11_8, av_in2);
        av_m39 = _mm_mul_ps(v_CRTM_11_9, av_in7);
        av_m40 = _mm_mul_ps(v_CRTM_11_10, av_in8);

        av_s31 = _mm_add_ps(av_m31, av_m33);
        av_s32 = _mm_add_ps(av_s31, av_in0);
        av_s33 = _mm_add_ps(av_m35, av_m37);
        av_s34 = _mm_add_ps(av_s33, av_m39);
        av_s35 = _mm_sub_ps(av_s32, av_s34);
        av_s36 = _mm_sub_ps(av_m32, av_m34);
        av_s37 = _mm_sub_ps(av_m38, av_m36);
        av_s38 = _mm_add_ps(av_s37, av_m40);
        av_s39 = _mm_add_ps(av_s36, av_s38);

        // Output point 9: x(8)
        v_out8 = _mm_sub_ps(av_s35, av_s39);
        curr_out = out + out_strides[8];
        STHR_128_S(curr_out, v_out_stride, v_out8);
        // Output point 15: x(14)
        v_out14 = _mm_add_ps(av_s35, av_s39);
        curr_out = out + out_strides[14];
        STHR_128_S(curr_out, v_out_stride, v_out14);

        av_m41 = _mm_mul_ps(v_CRTM_11_1, av_in3);
        av_m42 = _mm_mul_ps(v_CRTM_11_2, av_in4);
        av_m43 = _mm_mul_ps(v_CRTM_11_3, av_in7);
        av_m44 = _mm_mul_ps(v_CRTM_11_4, av_in8);
        av_m45 = _mm_mul_ps(v_CRTM_11_5, av_in9);
        av_m46 = _mm_mul_ps(v_CRTM_11_6, av_in10);
        av_m47 = _mm_mul_ps(v_CRTM_11_7, av_in5);
        av_m48 = _mm_mul_ps(v_CRTM_11_8, av_in6);
        av_m49 = _mm_mul_ps(v_CRTM_11_9, av_in1);
        av_m50 = _mm_mul_ps(v_CRTM_11_10, av_in2);

        av_s40 = _mm_add_ps(av_m41, av_m43);
        av_s41 = _mm_add_ps(av_s40, av_in0);
        av_s42 = _mm_add_ps(av_m45, av_m47);
        av_s43 = _mm_add_ps(av_s42, av_m49);
        av_s44 = _mm_sub_ps(av_s41, av_s43);
        av_s45 = _mm_add_ps(av_m42, av_m44);
        av_s46 = _mm_add_ps(av_m46, av_m48);
        av_s47 = _mm_sub_ps(av_s45, av_s46);
        av_s48 = _mm_sub_ps(av_s47, av_m50);

        // Output point 11: x(10)
        v_out10 = _mm_add_ps(av_s44, av_s48);
        curr_out = out + out_strides[10];
        STHR_128_S(curr_out, v_out_stride, v_out10);
        // Output point 13: x(12)
        v_out12 = _mm_sub_ps(av_s44, av_s48);
        curr_out = out + out_strides[12];
        STHR_128_S(curr_out, v_out_stride, v_out12);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8, bv_in9, bv_in10;
        __m128 bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
               bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16,
               bv_s17, bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24,
               bv_s25, bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32,
               bv_s33, bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40,
               bv_s41, bv_s42, bv_s43, bv_s44, bv_s45, bv_s46, bv_s47, bv_s48;
        __m128 bv_m0, bv_m1, bv_m2, bv_m3, bv_m4, bv_m5, bv_m6, bv_m7, bv_m8,
               bv_m9, bv_m10, bv_m11, bv_m12, bv_m13, bv_m14, bv_m15, bv_m16,
               bv_m17, bv_m18, bv_m19, bv_m20, bv_m21, bv_m22, bv_m23, bv_m24,
               bv_m25, bv_m26, bv_m27, bv_m28, bv_m29, bv_m30, bv_m31, bv_m32,
               bv_m33, bv_m34, bv_m35, bv_m36, bv_m37, bv_m38, bv_m39, bv_m40,
               bv_m41, bv_m42, bv_m43, bv_m44, bv_m45, bv_m46, bv_m47, bv_m48,
               bv_m49, bv_m50;

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
        // Input point 22: X(21)
        curr_in = in + in_strides[21];
        LDHR_128_S(curr_in, v_in_stride, bv_in10);

        bv_s0 = _mm_add_ps(bv_in0, bv_in2);
        bv_s1 = _mm_add_ps(bv_in4, bv_in6);
        bv_s2 = _mm_add_ps(bv_s0, bv_s1);
        bv_s3 = _mm_add_ps(bv_s2, bv_in8);
        bv_m0 = _mm_mul_ps(v_CRTM_11_11, bv_s3);

        // Output point 2: x(1)
        v_out1 = _mm_add_ps(bv_m0, bv_in10);
        curr_out = out + out_strides[1];
        STHR_128_S(curr_out, v_out_stride, v_out1);

        bv_m1 = _mm_mul_ps(v_CRTM_11_9, bv_in0);
        bv_m2 = _mm_mul_ps(v_CRTM_11_10, bv_in1);
        bv_m3 = _mm_mul_ps(v_CRTM_11_1, bv_in8);
        bv_m4 = _mm_mul_ps(v_CRTM_11_2, bv_in9);
        bv_m5 = _mm_mul_ps(v_CRTM_11_7, bv_in2);
        bv_m6 = _mm_mul_ps(v_CRTM_11_8, bv_in3);
        bv_m7 = _mm_mul_ps(v_CRTM_11_3, bv_in6);
        bv_m8 = _mm_mul_ps(v_CRTM_11_4, bv_in7);
        bv_m9 = _mm_mul_ps(v_CRTM_11_5, bv_in4);
        bv_m10 = _mm_mul_ps(v_CRTM_11_6, bv_in5);

        bv_s4 = _mm_add_ps(bv_m2, bv_m4);
        bv_s5 = _mm_add_ps(bv_m6, bv_m8);
        bv_s6 = _mm_add_ps(bv_s4, bv_s5);
        bv_s7 = _mm_add_ps(bv_s6, bv_m10);
        bv_s8 = _mm_sub_ps(bv_m1, bv_m3);
        bv_s9 = _mm_sub_ps(bv_m5, bv_m7);
        bv_s10 = _mm_add_ps(bv_s8, bv_s9);
        bv_s11 = _mm_sub_ps(bv_m9, bv_in10);
        bv_s12 = _mm_add_ps(bv_s10, bv_s11);

        // Output point 4: x(3)
        v_out3 = _mm_sub_ps(bv_s12, bv_s7);
        curr_out = out + out_strides[3];
        STHR_128_S(curr_out, v_out_stride, v_out3);

        // Output point 22: x(21)
        v_out21 = NEGATE_128_S(_mm_add_ps(bv_s12, bv_s7));
        curr_out = out + out_strides[21];
        STHR_128_S(curr_out, v_out_stride, v_out21);

        bv_m11 = _mm_mul_ps(v_CRTM_11_1, bv_in0);
        bv_m12 = _mm_mul_ps(v_CRTM_11_2, bv_in1);
        bv_m13 = _mm_mul_ps(v_CRTM_11_5, bv_in2);
        bv_m14 = _mm_mul_ps(v_CRTM_11_6, bv_in3);
        bv_m15 = _mm_mul_ps(v_CRTM_11_9, bv_in4);
        bv_m16 = _mm_mul_ps(v_CRTM_11_10, bv_in5);
        bv_m17 = _mm_mul_ps(v_CRTM_11_7, bv_in6);
        bv_m18 = _mm_mul_ps(v_CRTM_11_8, bv_in7);
        bv_m19 = _mm_mul_ps(v_CRTM_11_3, bv_in8);
        bv_m20 = _mm_mul_ps(v_CRTM_11_4, bv_in9);

        bv_s13 = _mm_add_ps(bv_m12, bv_m16);
        bv_s14 = _mm_sub_ps(bv_m14, bv_m20);
        bv_s15 = _mm_add_ps(bv_s13, bv_s14);
        bv_s16 = _mm_sub_ps(bv_s15, bv_m18);
        bv_s17 = _mm_add_ps(bv_m11, bv_m19);
        bv_s18 = _mm_add_ps(bv_m13, bv_m17);
        bv_s19 = _mm_sub_ps(bv_s17, bv_s18);
        bv_s20 = _mm_sub_ps(bv_in10, bv_m15);
        bv_s21 = _mm_add_ps(bv_s19, bv_s20);

        // Output point 6: x(5)
        v_out5 = _mm_sub_ps(bv_s21, bv_s16);
        curr_out = out + out_strides[5];
        STHR_128_S(curr_out, v_out_stride, v_out5);
        // Output point 20: x(19)
        v_out19 = NEGATE_128_S(_mm_add_ps(bv_s21, bv_s16));
        curr_out = out + out_strides[19];
        STHR_128_S(curr_out, v_out_stride, v_out19);

        bv_m21 = _mm_mul_ps(v_CRTM_11_7, bv_in0);
        bv_m22 = _mm_mul_ps(v_CRTM_11_8, bv_in1);
        bv_m23 = _mm_mul_ps(v_CRTM_11_1, bv_in2);
        bv_m24 = _mm_mul_ps(v_CRTM_11_2, bv_in3);
        bv_m25 = _mm_mul_ps(v_CRTM_11_3, bv_in4);
        bv_m26 = _mm_mul_ps(v_CRTM_11_4, bv_in5);
        bv_m27 = _mm_mul_ps(v_CRTM_11_9, bv_in6);
        bv_m28 = _mm_mul_ps(v_CRTM_11_10, bv_in7);
        bv_m29 = _mm_mul_ps(v_CRTM_11_5, bv_in8);
        bv_m30 = _mm_mul_ps(v_CRTM_11_6, bv_in9);

        bv_s22 = _mm_add_ps(bv_m22, bv_m24);
        bv_s23 = _mm_sub_ps(bv_m30, bv_m26);
        bv_s24 = _mm_add_ps(bv_s22, bv_s23);
        bv_s25 = _mm_sub_ps(bv_s24, bv_m28);
        bv_s26 = _mm_add_ps(bv_m21, bv_m29);
        bv_s27 = _mm_sub_ps(bv_m27, bv_m23);
        bv_s28 = _mm_add_ps(bv_s26, bv_s27);
        bv_s29 = _mm_add_ps(bv_m25, bv_in10);
        bv_s30 = _mm_sub_ps(bv_s28, bv_s29);

        // Output point 8: x(7)
        v_out7 = _mm_sub_ps(bv_s30, bv_s25);
        curr_out = out + out_strides[7];
        STHR_128_S(curr_out, v_out_stride, v_out7);
        // Output point 18: x(17)
        v_out17 = NEGATE_128_S(_mm_add_ps(bv_s30, bv_s25));
        curr_out = out + out_strides[17];
        STHR_128_S(curr_out, v_out_stride, v_out17);

        bv_m31 = _mm_mul_ps(v_CRTM_11_3, bv_in0);
        bv_m32 = _mm_mul_ps(v_CRTM_11_4, bv_in1);
        bv_m33 = _mm_mul_ps(v_CRTM_11_9, bv_in2);
        bv_m34 = _mm_mul_ps(v_CRTM_11_10, bv_in3);
        bv_m35 = _mm_mul_ps(v_CRTM_11_1, bv_in4);
        bv_m36 = _mm_mul_ps(v_CRTM_11_2, bv_in5);
        bv_m37 = _mm_mul_ps(v_CRTM_11_5, bv_in6);
        bv_m38 = _mm_mul_ps(v_CRTM_11_6, bv_in7);
        bv_m39 = _mm_mul_ps(v_CRTM_11_7, bv_in8);
        bv_m40 = _mm_mul_ps(v_CRTM_11_8, bv_in9);

        bv_s31 = _mm_sub_ps(bv_m32, bv_m34);
        bv_s32 = _mm_sub_ps(bv_m38, bv_m40);
        bv_s33 = _mm_add_ps(bv_s31, bv_s32);
        bv_s34 = _mm_sub_ps(bv_s33, bv_m36);
        bv_s35 = _mm_add_ps(bv_m31, bv_m35);
        bv_s36 = _mm_add_ps(bv_m37, bv_m39);
        bv_s37 = _mm_sub_ps(bv_s35, bv_s36);
        bv_s38 = _mm_sub_ps(bv_in10, bv_m33);
        bv_s39 = _mm_add_ps(bv_s37, bv_s38);

        // Output point 10: x(9)
        v_out9 = _mm_sub_ps(bv_s39, bv_s34);
        curr_out = out + out_strides[9];
        STHR_128_S(curr_out, v_out_stride, v_out9);
        // Output point 16: x(15)
        v_out15 = NEGATE_128_S(_mm_add_ps(bv_s39, bv_s34));
        curr_out = out + out_strides[15];
        STHR_128_S(curr_out, v_out_stride, v_out15);

        bv_m41 = _mm_mul_ps(v_CRTM_11_5, bv_in0);
        bv_m42 = _mm_mul_ps(v_CRTM_11_6, bv_in1);
        bv_m43 = _mm_mul_ps(v_CRTM_11_3, bv_in2);
        bv_m44 = _mm_mul_ps(v_CRTM_11_4, bv_in3);
        bv_m45 = _mm_mul_ps(v_CRTM_11_7, bv_in4);
        bv_m46 = _mm_mul_ps(v_CRTM_11_8, bv_in5);
        bv_m47 = _mm_mul_ps(v_CRTM_11_1, bv_in6);
        bv_m48 = _mm_mul_ps(v_CRTM_11_2, bv_in7);
        bv_m49 = _mm_mul_ps(v_CRTM_11_9, bv_in8);
        bv_m50 = _mm_mul_ps(v_CRTM_11_10, bv_in9);

        bv_s40 = _mm_sub_ps(bv_m42, bv_m44);
        bv_s41 = _mm_add_ps(bv_m46, bv_m50);
        bv_s42 = _mm_add_ps(bv_s40, bv_s41);
        bv_s43 = _mm_sub_ps(bv_s42, bv_m48);
        bv_s44 = _mm_add_ps(bv_m41, bv_m45);
        bv_s45 = _mm_add_ps(bv_m43, bv_m47);
        bv_s46 = _mm_sub_ps(bv_s44, bv_s45);
        bv_s47 = _mm_sub_ps(bv_m49, bv_in10);
        bv_s48 = _mm_add_ps(bv_s46, bv_s47);

        // Output point 12: x(11)
        v_out11 = _mm_sub_ps(bv_s48, bv_s43);
        curr_out = out + out_strides[11];
        STHR_128_S(curr_out, v_out_stride, v_out11);
        // Output point 14: x(13)
        v_out13 = NEGATE_128_S(_mm_add_ps(bv_s48, bv_s43));
        curr_out = out + out_strides[13];
        STHR_128_S(curr_out, v_out_stride, v_out13);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
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
        FFTZ_FLOAT  b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9,
               b_s10, b_s11, b_s12, b_s13, b_s14, b_s15, b_s16, b_s17, b_s18,
               b_s19, b_s20, b_s21, b_s22, b_s23, b_s24, b_s25, b_s26, b_s27,
               b_s28, b_s29, b_s30, b_s31, b_s32, b_s33, b_s34, b_s35, b_s36,
               b_s37, b_s38, b_s39, b_s40, b_s41, b_s42, b_s43, b_s44, b_s45,
               b_s46, b_s47, b_s48;
        FFTZ_FLOAT  b_m0, b_m1, b_m2, b_m3, b_m4, b_m5, b_m6, b_m7, b_m8, b_m9,
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
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft11avx128_fp64_fwd(FFTZ_VOID *in_real,
                                             FFTZ_VOID *in_imag,
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
    FFTZ_UINT8 is_contiguous_in = (v_in_stride == 1);
    FFTZ_UINT8 is_contiguous_out = (v_out_stride == 1);

    FFTZ_INTP cnt;
    FFTZ_DOUBLE *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_128_D;

    __m128d v_CRTM_11_1 = _mm_set1_pd(CRTM_11_1);
    __m128d v_CRTM_11_2 = _mm_set1_pd(CRTM_11_2);
    __m128d v_CRTM_11_3 = _mm_set1_pd(CRTM_11_3);
    __m128d v_CRTM_11_4 = _mm_set1_pd(CRTM_11_4);
    __m128d v_CRTM_11_5 = _mm_set1_pd(CRTM_11_5);
    __m128d v_CRTM_11_6 = _mm_set1_pd(CRTM_11_6);
    __m128d v_CRTM_11_7 = _mm_set1_pd(CRTM_11_7);
    __m128d v_CRTM_11_8 = _mm_set1_pd(CRTM_11_8);
    __m128d v_CRTM_11_9 = _mm_set1_pd(CRTM_11_9);
    __m128d v_CRTM_11_10 = _mm_set1_pd(CRTM_11_10);

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m128d av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
                av_in8, av_in9, av_in10;
        __m128d av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
                av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16,
                av_s17, av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24,
                av_s25, av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32,
                av_s33, av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40,
                av_s41, av_s42, av_s43, av_s44, av_s45, av_s46, av_s47, av_s48;
        __m128d av_m0, av_m1, av_m2, av_m3, av_m4, av_m5, av_m6, av_m7, av_m8,
                av_m9, av_m10, av_m11, av_m12, av_m13, av_m14, av_m15, av_m16,
                av_m17, av_m18, av_m19, av_m20, av_m21, av_m22, av_m23, av_m24,
                av_m25, av_m26, av_m27, av_m28, av_m29, av_m30, av_m31, av_m32,
                av_m33, av_m34, av_m35, av_m36, av_m37, av_m38, av_m39, av_m40,
                av_m41, av_m42, av_m43, av_m44, av_m45, av_m46, av_m47, av_m48,
                av_m49;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
                v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_stride, av_in0, is_contiguous_in);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_128_D(curr_in, v_in_stride, av_in1, is_contiguous_in);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_128_D(curr_in, v_in_stride, av_in2, is_contiguous_in);
        // Input point 7: x(6)
        curr_in = in + in_strides[6];
        LDR_128_D(curr_in, v_in_stride, av_in3, is_contiguous_in);
        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDR_128_D(curr_in, v_in_stride, av_in4, is_contiguous_in);
        // Input point 11: x(10)
        curr_in = in + in_strides[10];
        LDR_128_D(curr_in, v_in_stride, av_in5, is_contiguous_in);
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDR_128_D(curr_in, v_in_stride, av_in6, is_contiguous_in);
        // Input point 15: x(14)
        curr_in = in + in_strides[14];
        LDR_128_D(curr_in, v_in_stride, av_in7, is_contiguous_in);
        // Input point 17: x(16)
        curr_in = in + in_strides[16];
        LDR_128_D(curr_in, v_in_stride, av_in8, is_contiguous_in);
        // Input point 19: x(18)
        curr_in = in + in_strides[18];
        LDR_128_D(curr_in, v_in_stride, av_in9, is_contiguous_in);
        // Input point 21: x(20)
        curr_in = in + in_strides[20];
        LDR_128_D(curr_in, v_in_stride, av_in10, is_contiguous_in);

        av_s0 = _mm_add_pd(av_in1, av_in10);
        av_s1 = _mm_add_pd(av_in2, av_in9);
        av_s2 = _mm_add_pd(av_in3, av_in8);
        av_s3 = _mm_add_pd(av_in4, av_in7);
        av_s4 = _mm_add_pd(av_in5, av_in6);
        av_s5 = _mm_sub_pd(av_in1, av_in10);
        av_s6 = _mm_sub_pd(av_in2, av_in9);
        av_s7 = _mm_sub_pd(av_in3, av_in8);
        av_s8 = _mm_sub_pd(av_in4, av_in7);
        av_s9 = _mm_sub_pd(av_in5, av_in6);

        av_s10 = _mm_add_pd(av_s0, av_s1);
        av_s11 = _mm_add_pd(av_s2, av_s3);
        av_s12 = _mm_add_pd(av_s4, av_in0);
        av_s13 = _mm_add_pd(av_s10, av_s11);

        // Output point 1: X(0)
        v_out0 = _mm_add_pd(av_s12, av_s13);
        STR_128_D(curr_out, v_out_stride, v_out0, is_contiguous_out);

        av_m0 = _mm_mul_pd(v_CRTM_11_1, av_s0);
        av_m1 = _mm_mul_pd(v_CRTM_11_3, av_s1);
        av_m2 = _mm_mul_pd(v_CRTM_11_5, av_s2);
        av_m3 = _mm_mul_pd(v_CRTM_11_7, av_s3);
        av_m4 = _mm_mul_pd(v_CRTM_11_9, av_s4);

        av_s14 = _mm_add_pd(av_m0, av_m1);
        av_s15 = _mm_add_pd(av_m2, av_m3);
        av_s16 = _mm_sub_pd(av_in0, av_m4);
        av_s17 = _mm_sub_pd(av_s14, av_s15);

        // Output point 4: X(3)
        v_out3 = _mm_add_pd(av_s16, av_s17);

        av_m5 = _mm_mul_pd(v_CRTM_11_2, av_s5);
        av_m6 = _mm_mul_pd(v_CRTM_11_4, av_s6);
        av_m7 = _mm_mul_pd(v_CRTM_11_6, av_s7);
        av_m8 = _mm_mul_pd(v_CRTM_11_8, av_s8);
        av_m9 = _mm_mul_pd(v_CRTM_11_10, av_s9);

        av_s18 = _mm_add_pd(av_m5, av_m6);
        av_s19 = _mm_add_pd(av_m7, av_m8);
        av_s20 = _mm_add_pd(av_s19, av_m9);

        // Output point 5: X(4)
        v_out4 = NEGATE_128_D(_mm_add_pd(av_s18, av_s20));
        curr_out = out + out_strides[3];
        STRI_2x128_D(curr_out, v_out_stride, v_out3, v_out4);

        av_m10 = _mm_mul_pd(v_CRTM_11_1, av_s4);
        av_m11 = _mm_mul_pd(v_CRTM_11_3, av_s0);
        av_m12 = _mm_mul_pd(v_CRTM_11_5, av_s3);
        av_m13 = _mm_mul_pd(v_CRTM_11_7, av_s1);
        av_m14 = _mm_mul_pd(v_CRTM_11_9, av_s2);

        av_s21 = _mm_add_pd(av_m10, av_m11);
        av_s22 = _mm_add_pd(av_m12, av_m13);
        av_s23 = _mm_sub_pd(av_in0, av_m14);
        av_s24 = _mm_sub_pd(av_s21, av_s22);

        // Output point 8: X(7)
        v_out7 = _mm_add_pd(av_s23, av_s24);

        av_m15 = _mm_mul_pd(v_CRTM_11_2, av_s9);
        av_m16 = _mm_mul_pd(v_CRTM_11_4, av_s5);
        av_m17 = _mm_mul_pd(v_CRTM_11_6, av_s8);
        av_m18 = _mm_mul_pd(v_CRTM_11_8, av_s6);
        av_m19 = _mm_mul_pd(v_CRTM_11_10, av_s7);

        av_s25 = _mm_sub_pd(av_m15, av_m16);
        av_s26 = _mm_sub_pd(av_m17, av_m18);
        av_s27 = _mm_add_pd(av_s26, av_m19);

        // Output point 9: X(8)
        v_out8 = _mm_add_pd(av_s25, av_s27);
        curr_out = out + out_strides[7];
        STRI_2x128_D(curr_out, v_out_stride, v_out7, v_out8);

        av_m20 = _mm_mul_pd(v_CRTM_11_1, av_s3);
        av_m21 = _mm_mul_pd(v_CRTM_11_3, av_s2);
        av_m22 = _mm_mul_pd(v_CRTM_11_5, av_s0);
        av_m23 = _mm_mul_pd(v_CRTM_11_7, av_s4);
        av_m24 = _mm_mul_pd(v_CRTM_11_9, av_s1);

        av_s28 = _mm_add_pd(av_m20, av_m21);
        av_s29 = _mm_add_pd(av_m22, av_m23);
        av_s30 = _mm_sub_pd(av_in0, av_m24);
        av_s31 = _mm_sub_pd(av_s30, av_s29);

        // Output point 12: X(11)
        v_out11 = _mm_add_pd(av_s28, av_s31);

        av_m25 = _mm_mul_pd(v_CRTM_11_2, av_s8);
        av_m26 = _mm_mul_pd(v_CRTM_11_4, av_s7);
        av_m27 = _mm_mul_pd(v_CRTM_11_6, av_s5);
        av_m28 = _mm_mul_pd(v_CRTM_11_8, av_s9);
        av_m29 = _mm_mul_pd(v_CRTM_11_10, av_s6);

        av_s32 = _mm_sub_pd(av_m26, av_m25);
        av_s33 = _mm_add_pd(av_m27, av_m28);
        av_s34 = _mm_sub_pd(av_s32, av_s33);

        // Output point 13: X(12)
        v_out12 = _mm_add_pd(av_s34, av_m29);
        curr_out = out + out_strides[11];
        STRI_2x128_D(curr_out, v_out_stride, v_out11, v_out12);

        av_m30 = _mm_mul_pd(v_CRTM_11_1, av_s2);
        av_m31 = _mm_mul_pd(v_CRTM_11_3, av_s4);
        av_m32 = _mm_mul_pd(v_CRTM_11_5, av_s1);
        av_m33 = _mm_mul_pd(v_CRTM_11_7, av_s0);
        av_m34 = _mm_mul_pd(v_CRTM_11_9, av_s3);

        av_s35 = _mm_add_pd(av_m30, av_m31);
        av_s36 = _mm_add_pd(av_m32, av_m33);
        av_s37 = _mm_sub_pd(av_in0, av_m34);
        av_s38 = _mm_sub_pd(av_s35, av_s36);

        // Output point 16: X(15)
        v_out15 = _mm_add_pd(av_s38, av_s37);

        av_m35 = _mm_mul_pd(v_CRTM_11_2, av_s7);
        av_m36 = _mm_mul_pd(v_CRTM_11_4, av_s9);
        av_m37 = _mm_mul_pd(v_CRTM_11_6, av_s6);
        av_m38 = _mm_mul_pd(v_CRTM_11_8, av_s5);
        av_m39 = _mm_mul_pd(v_CRTM_11_10, av_s8);

        av_s39 = _mm_sub_pd(av_m36, av_m35);
        av_s40 = _mm_sub_pd(av_m37, av_m38);
        av_s41 = _mm_sub_pd(av_s40, av_m39);

        // Output point 17: X(16)
        v_out16 = _mm_add_pd(av_s39, av_s41);
        curr_out = out + out_strides[15];
        STRI_2x128_D(curr_out, v_out_stride, v_out15, v_out16);

        av_m40 = _mm_mul_pd(v_CRTM_11_1, av_s1);
        av_m41 = _mm_mul_pd(v_CRTM_11_3, av_s3);
        av_m42 = _mm_mul_pd(v_CRTM_11_5, av_s4);
        av_m43 = _mm_mul_pd(v_CRTM_11_7, av_s2);
        av_m44 = _mm_mul_pd(v_CRTM_11_9, av_s0);

        av_s42 = _mm_add_pd(av_m40, av_m41);
        av_s43 = _mm_add_pd(av_m42, av_m43);
        av_s44 = _mm_sub_pd(av_in0, av_m44);
        av_s45 = _mm_sub_pd(av_s42, av_s43);

        // Output point 20: X(19)
        v_out19 = _mm_add_pd(av_s44, av_s45);

        av_m45 = _mm_mul_pd(v_CRTM_11_2, av_s6);
        av_m46 = _mm_mul_pd(v_CRTM_11_4, av_s8);
        av_m47 = _mm_mul_pd(v_CRTM_11_6, av_s9);
        av_m48 = _mm_mul_pd(v_CRTM_11_8, av_s7);
        av_m49 = _mm_mul_pd(v_CRTM_11_10, av_s5);

        av_s46 = _mm_add_pd(av_m45, av_m46);
        av_s47 = _mm_add_pd(av_m47, av_m48);
        av_s48 = _mm_sub_pd(av_s46, av_s47);

        // Output point 21: X(20)
        v_out20 = _mm_sub_pd(av_s48, av_m49);
        curr_out = out + out_strides[19];
        STRI_2x128_D(curr_out, v_out_stride, v_out19, v_out20);

        /* Shifted DFT */
        __m128d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
                bv_in8, bv_in9, bv_in10;
        __m128d bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
                bv_s9;
        __m128d bv_m0, bv_m1, bv_m2, bv_m3, bv_m4, bv_m5, bv_m6, bv_m7, bv_m8,
                bv_m9, bv_m10, bv_m11, bv_m12, bv_m13, bv_m14, bv_m15, bv_m16,
                bv_m17, bv_m18, bv_m19, bv_m20, bv_m21, bv_m22, bv_m23, bv_m24,
                bv_m25, bv_m26, bv_m27, bv_m28, bv_m29, bv_m30, bv_m31, bv_m32,
                bv_m33, bv_m34, bv_m35, bv_m36, bv_m37, bv_m38, bv_m39, bv_m40,
                bv_m41, bv_m42, bv_m43, bv_m44, bv_m45, bv_m46, bv_m47, bv_m48,
                bv_m49;

        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_128_D(curr_in, v_in_stride, bv_in0, is_contiguous_in);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_128_D(curr_in, v_in_stride, bv_in1, is_contiguous_in);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_128_D(curr_in, v_in_stride, bv_in2, is_contiguous_in);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_128_D(curr_in, v_in_stride, bv_in3, is_contiguous_in);
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDR_128_D(curr_in, v_in_stride, bv_in4, is_contiguous_in);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_128_D(curr_in, v_in_stride, bv_in5, is_contiguous_in);
        // Input point 14: x(13)
        curr_in = in + in_strides[13];
        LDR_128_D(curr_in, v_in_stride, bv_in6, is_contiguous_in);
        // Input point 16: x(15)
        curr_in = in + in_strides[15];
        LDR_128_D(curr_in, v_in_stride, bv_in7, is_contiguous_in);
        // Input point 18: x(17)
        curr_in = in + in_strides[17];
        LDR_128_D(curr_in, v_in_stride, bv_in8, is_contiguous_in);
        // Input point 20: x(19)
        curr_in = in + in_strides[19];
        LDR_128_D(curr_in, v_in_stride, bv_in9, is_contiguous_in);
        // Input point 22: x(21)
        curr_in = in + in_strides[21];
        LDR_128_D(curr_in, v_in_stride, bv_in10, is_contiguous_in);

        bv_s0 = _mm_add_pd(bv_in1, bv_in10);
        bv_s1 = _mm_add_pd(bv_in2, bv_in9);
        bv_s2 = _mm_add_pd(bv_in3, bv_in8);
        bv_s3 = _mm_add_pd(bv_in4, bv_in7);
        bv_s4 = _mm_add_pd(bv_in5, bv_in6);
        bv_s5 = _mm_sub_pd(bv_in1, bv_in10);
        bv_s6 = _mm_sub_pd(bv_in2, bv_in9);
        bv_s7 = _mm_sub_pd(bv_in3, bv_in8);
        bv_s8 = _mm_sub_pd(bv_in4, bv_in7);
        bv_s9 = _mm_sub_pd(bv_in5, bv_in6);

        bv_m0 = _mm_mul_pd(v_CRTM_11_9, bv_s5);
        bv_m1 = _mm_mul_pd(v_CRTM_11_1, bv_s6);
        bv_m2 = _mm_add_pd(bv_m0, bv_m1);
        bv_m3 = _mm_mul_pd(v_CRTM_11_7, bv_s7);
        bv_m4 = _mm_add_pd(bv_m2, bv_m3);
        bv_m5 = _mm_mul_pd(v_CRTM_11_3, bv_s8);
        bv_m6 = _mm_add_pd(bv_m4, bv_m5);
        bv_m7 = _mm_mul_pd(v_CRTM_11_5, bv_s9);
        bv_m8 = _mm_add_pd(bv_m6, bv_m7);

        // Output point 2: X(1)
        v_out1 = _mm_add_pd(bv_in0, bv_m8);

        bv_m9 = _mm_mul_pd(v_CRTM_11_10, bv_s0);
        bv_m10 = _mm_mul_pd(v_CRTM_11_2, bv_s1);
        bv_m11 = _mm_add_pd(bv_m9, bv_m10);
        bv_m12 = _mm_mul_pd(v_CRTM_11_8, bv_s2);
        bv_m13 = _mm_add_pd(bv_m11, bv_m12);
        bv_m14 = _mm_mul_pd(v_CRTM_11_4, bv_s3);
        bv_m15 = _mm_add_pd(bv_m13, bv_m14);
        bv_m16 = _mm_mul_pd(v_CRTM_11_6, bv_s4);
        bv_m17 = _mm_add_pd(bv_m15, bv_m16);

        // Output point 3: X(2)
        v_out2 = NEGATE_128_D(bv_m17);
        curr_out = out + out_strides[1];
        STRI_2x128_D(curr_out, v_out_stride, v_out1, v_out2);

        bv_m18 = _mm_mul_pd(v_CRTM_11_7, bv_s5);
        bv_m19 = _mm_mul_pd(v_CRTM_11_5, bv_s6);
        bv_m20 = _mm_sub_pd(bv_m18, bv_m19);
        bv_m21 = _mm_mul_pd(v_CRTM_11_1, bv_s7);
        bv_m22 = _mm_sub_pd(bv_m20, bv_m21);
        bv_m23 = _mm_mul_pd(v_CRTM_11_9, bv_s8);
        bv_m24 = _mm_sub_pd(bv_m22, bv_m23);
        bv_m25 = _mm_mul_pd(v_CRTM_11_3, bv_s9);
        bv_m26 = _mm_sub_pd(bv_m24, bv_m25);

        // Output point 6: X(5)
        v_out5 = _mm_add_pd(bv_in0, bv_m26);

        bv_m27 = _mm_mul_pd(v_CRTM_11_8, bv_s0);
        bv_m28 = _mm_mul_pd(v_CRTM_11_6, bv_s1);
        bv_m29 = _mm_add_pd(bv_m27, bv_m28);
        bv_m30 = _mm_mul_pd(v_CRTM_11_2, bv_s2);
        bv_m31 = _mm_add_pd(bv_m29, bv_m30);
        bv_m32 = _mm_mul_pd(v_CRTM_11_10, bv_s3);
        bv_m33 = _mm_sub_pd(bv_m31, bv_m32);
        bv_m34 = _mm_mul_pd(v_CRTM_11_4, bv_s4);
        bv_m35 = _mm_sub_pd(bv_m33, bv_m34);

        // Output point 7: X(6)
        v_out6 = NEGATE_128_D(bv_m35);
        curr_out = out + out_strides[5];
        STRI_2x128_D(curr_out, v_out_stride, v_out5, v_out6);

        bv_m36 = _mm_mul_pd(v_CRTM_11_5, bv_s5);
        bv_m37 = _mm_mul_pd(v_CRTM_11_9, bv_s6);
        bv_m38 = _mm_sub_pd(bv_m36, bv_m37);
        bv_m39 = _mm_mul_pd(v_CRTM_11_3, bv_s7);
        bv_m40 = _mm_sub_pd(bv_m38, bv_m39);
        bv_m41 = _mm_mul_pd(v_CRTM_11_1, bv_s8);
        bv_m42 = _mm_add_pd(bv_m40, bv_m41);
        bv_m43 = _mm_mul_pd(v_CRTM_11_7, bv_s9);
        bv_m44 = _mm_add_pd(bv_m42, bv_m43);

        // Output point 10: X(9)
        v_out9 = _mm_add_pd(bv_in0, bv_m44);

        bv_m45 = _mm_mul_pd(v_CRTM_11_6, bv_s0);
        bv_m46 = _mm_mul_pd(v_CRTM_11_10, bv_s1);
        bv_m47 = _mm_add_pd(bv_m45, bv_m46);
        bv_m48 = _mm_mul_pd(v_CRTM_11_4, bv_s2);
        bv_m49 = _mm_sub_pd(bv_m47, bv_m48);
        bv_m0 = _mm_mul_pd(v_CRTM_11_2, bv_s3);
        bv_m1 = _mm_sub_pd(bv_m49, bv_m0);
        bv_m2 = _mm_mul_pd(v_CRTM_11_8, bv_s4);
        bv_m3 = _mm_add_pd(bv_m1, bv_m2);

        // Output point 11: X(10)
        v_out10 = NEGATE_128_D(bv_m3);
        curr_out = out + out_strides[9];
        STRI_2x128_D(curr_out, v_out_stride, v_out9, v_out10);

        bv_m4 = _mm_mul_pd(v_CRTM_11_3, bv_s5);
        bv_m5 = _mm_mul_pd(v_CRTM_11_7, bv_s6);
        bv_m6 = _mm_add_pd(bv_m4, bv_m5);
        bv_m7 = _mm_mul_pd(v_CRTM_11_9, bv_s7);
        bv_m8 = _mm_sub_pd(bv_m7, bv_m6);
        bv_m9 = _mm_mul_pd(v_CRTM_11_5, bv_s8);
        bv_m10 = _mm_sub_pd(bv_m8, bv_m9);
        bv_m11 = _mm_mul_pd(v_CRTM_11_1, bv_s9);
        bv_m12 = _mm_sub_pd(bv_m10, bv_m11);

        // Output point 14: X(13)
        v_out13 = _mm_add_pd(bv_in0, bv_m12);

        bv_m13 = _mm_mul_pd(v_CRTM_11_4, bv_s0);
        bv_m14 = _mm_mul_pd(v_CRTM_11_8, bv_s1);
        bv_m15 = _mm_sub_pd(bv_m13, bv_m14);
        bv_m16 = _mm_mul_pd(v_CRTM_11_10, bv_s2);
        bv_m17 = _mm_sub_pd(bv_m15, bv_m16);
        bv_m18 = _mm_mul_pd(v_CRTM_11_6, bv_s3);
        bv_m19 = _mm_add_pd(bv_m17, bv_m18);
        bv_m20 = _mm_mul_pd(v_CRTM_11_2, bv_s4);
        bv_m21 = _mm_sub_pd(bv_m19, bv_m20);

        // Output point 15: X(14)
        v_out14 = NEGATE_128_D(bv_m21);
        curr_out = out + out_strides[13];
        STRI_2x128_D(curr_out, v_out_stride, v_out13, v_out14);

        bv_m22 = _mm_mul_pd(v_CRTM_11_1, bv_s5);
        bv_m23 = _mm_mul_pd(v_CRTM_11_3, bv_s6);
        bv_m24 = _mm_sub_pd(bv_m23, bv_m22);
        bv_m25 = _mm_mul_pd(v_CRTM_11_5, bv_s7);
        bv_m26 = _mm_add_pd(bv_m24, bv_m25);
        bv_m27 = _mm_mul_pd(v_CRTM_11_7, bv_s8);
        bv_m28 = _mm_sub_pd(bv_m26, bv_m27);
        bv_m29 = _mm_mul_pd(v_CRTM_11_9, bv_s9);
        bv_m30 = _mm_add_pd(bv_m28, bv_m29);

        // Output point 18: X(17)
        v_out17 = _mm_add_pd(bv_in0, bv_m30);

        bv_m31 = _mm_mul_pd(v_CRTM_11_2, bv_s0);
        bv_m32 = _mm_mul_pd(v_CRTM_11_4, bv_s1);
        bv_m33 = _mm_sub_pd(bv_m31, bv_m32);
        bv_m34 = _mm_mul_pd(v_CRTM_11_6, bv_s2);
        bv_m35 = _mm_add_pd(bv_m33, bv_m34);
        bv_m36 = _mm_mul_pd(v_CRTM_11_8, bv_s3);
        bv_m37 = _mm_sub_pd(bv_m35, bv_m36);
        bv_m38 = _mm_mul_pd(v_CRTM_11_10, bv_s4);
        bv_m39 = _mm_add_pd(bv_m37, bv_m38);

        // Output point 19: X(18)
        v_out18 = NEGATE_128_D(bv_m39);
        curr_out = out + out_strides[17];
        STRI_2x128_D(curr_out, v_out_stride, v_out17, v_out18);

        bv_m40 = _mm_sub_pd(bv_in0, bv_s5);
        bv_m41 = _mm_add_pd(bv_m40, bv_s6);
        bv_m42 = _mm_sub_pd(bv_m41, bv_s7);
        bv_m43 = _mm_add_pd(bv_m42, bv_s8);
        bv_m44 = _mm_sub_pd(bv_m43, bv_s9);

        // Output point 22: X(21)
        v_out21 = bv_m44;
        curr_out = out + out_strides[21];
        STR_128_D(curr_out, v_out_stride, v_out21, is_contiguous_out);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
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

        // Input point 0: x(0)
        a_in0 = *in;
        // Input point 3: x(2)
        a_in1 = in[in_strides[2]];
        // Input point 4: x(3)
        a_in2 = in[in_strides[4]];
        // Input point 6: x(5)
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
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft11avx128_fp64_bwd(FFTZ_VOID *in_real,
                                             FFTZ_VOID *in_imag,
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
    FFTZ_UINT8 is_contiguous_in = (v_in_stride == 1);
    FFTZ_UINT8 is_contiguous_out = (v_out_stride == 1);

    FFTZ_INTP cnt;
    FFTZ_DOUBLE *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_128_D;

    __m128d v_CRTM_11_1 = _mm_set1_pd(CRTM_11_1);
    __m128d v_CRTM_11_2 = _mm_set1_pd(CRTM_11_2);
    __m128d v_CRTM_11_3 = _mm_set1_pd(CRTM_11_3);
    __m128d v_CRTM_11_4 = _mm_set1_pd(CRTM_11_4);
    __m128d v_CRTM_11_5 = _mm_set1_pd(CRTM_11_5);
    __m128d v_CRTM_11_6 = _mm_set1_pd(CRTM_11_6);
    __m128d v_CRTM_11_7 = _mm_set1_pd(CRTM_11_7);
    __m128d v_CRTM_11_8 = _mm_set1_pd(CRTM_11_8);
    __m128d v_CRTM_11_9 = _mm_set1_pd(CRTM_11_9);
    __m128d v_CRTM_11_10 = _mm_set1_pd(CRTM_11_10);
    __m128d v_CRTM_11_11 = _mm_set1_pd(CRTM_11_11);

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m128d av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
                av_in8, av_in9, av_in10;
        __m128d av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
                av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16,
                av_s17, av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24,
                av_s25, av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32,
                av_s33, av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40,
                av_s41, av_s42, av_s43, av_s44, av_s45, av_s46, av_s47, av_s48;
        __m128d av_m0, av_m1, av_m2, av_m3, av_m4, av_m5, av_m6, av_m7, av_m8,
                av_m9, av_m10, av_m11, av_m12, av_m13, av_m14, av_m15, av_m16,
                av_m17, av_m18, av_m19, av_m20, av_m21, av_m22, av_m23, av_m24,
                av_m25, av_m26, av_m27, av_m28, av_m29, av_m30, av_m31, av_m32,
                av_m33, av_m34, av_m35, av_m36, av_m37, av_m38, av_m39, av_m40,
                av_m41, av_m42, av_m43, av_m44, av_m45, av_m46, av_m47, av_m48,
                av_m49, av_m50;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
                v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21;

        curr_in = in;
        curr_out = out;

        // Input point 1: X(0)
        LDR_128_D(curr_in, v_in_stride, av_in0, is_contiguous_in);
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

        av_s0 = _mm_add_pd(av_in1, av_in3);
        av_s1 = _mm_add_pd(av_in5, av_in7);
        av_s2 = _mm_add_pd(av_s0, av_s1);
        av_s3 = _mm_add_pd(av_s2, av_in9);
        av_m0 = _mm_mul_pd(v_CRTM_11_11, av_s3);

        // Output point 1: x(0)
        v_out0 = _mm_add_pd(av_in0, av_m0);
        STR_128_D(curr_out, v_out_stride, v_out0, is_contiguous_out);

        av_m1 = _mm_mul_pd(v_CRTM_11_1, av_in1);
        av_m2 = _mm_mul_pd(v_CRTM_11_2, av_in2);
        av_m3 = _mm_mul_pd(v_CRTM_11_3, av_in3);
        av_m4 = _mm_mul_pd(v_CRTM_11_4, av_in4);
        av_m5 = _mm_mul_pd(v_CRTM_11_5, av_in5);
        av_m6 = _mm_mul_pd(v_CRTM_11_6, av_in6);
        av_m7 = _mm_mul_pd(v_CRTM_11_7, av_in7);
        av_m8 = _mm_mul_pd(v_CRTM_11_8, av_in8);
        av_m9 = _mm_mul_pd(v_CRTM_11_9, av_in9);
        av_m10 = _mm_mul_pd(v_CRTM_11_10, av_in10);

        av_s4 = _mm_add_pd(av_m1, av_m3);
        av_s5 = _mm_add_pd(av_s4, av_in0);
        av_s6 = _mm_add_pd(av_m5, av_m7);
        av_s7 = _mm_add_pd(av_s6, av_m9);
        av_s8 = _mm_sub_pd(av_s5, av_s7);
        av_s9 = _mm_add_pd(av_m2, av_m4);
        av_s10 = _mm_add_pd(av_m6, av_m8);
        av_s11 = _mm_add_pd(av_s9, av_m10);
        av_s12 = _mm_add_pd(av_s10, av_s11);

        // Output point 3: x(2)
        v_out2 = _mm_sub_pd(av_s8, av_s12);
        curr_out = out + out_strides[2];
        STR_128_D(curr_out, v_out_stride, v_out2, is_contiguous_out);
        // Output point 21: x(20)
        v_out20 = _mm_add_pd(av_s8, av_s12);
        curr_out = out + out_strides[20];
        STR_128_D(curr_out, v_out_stride, v_out20, is_contiguous_out);

        av_m11 = _mm_mul_pd(v_CRTM_11_1, av_in9);
        av_m12 = _mm_mul_pd(v_CRTM_11_2, av_in10);
        av_m13 = _mm_mul_pd(v_CRTM_11_3, av_in1);
        av_m14 = _mm_mul_pd(v_CRTM_11_4, av_in2);
        av_m15 = _mm_mul_pd(v_CRTM_11_5, av_in7);
        av_m16 = _mm_mul_pd(v_CRTM_11_6, av_in8);
        av_m17 = _mm_mul_pd(v_CRTM_11_7, av_in3);
        av_m18 = _mm_mul_pd(v_CRTM_11_8, av_in4);
        av_m19 = _mm_mul_pd(v_CRTM_11_9, av_in5);
        av_m20 = _mm_mul_pd(v_CRTM_11_10, av_in6);

        av_s13 = _mm_add_pd(av_m11, av_m13);
        av_s14 = _mm_add_pd(av_s13, av_in0);
        av_s15 = _mm_add_pd(av_m15, av_m17);
        av_s16 = _mm_add_pd(av_s15, av_m19);
        av_s17 = _mm_sub_pd(av_s14, av_s16);
        av_s18 = _mm_sub_pd(av_m12, av_m14);
        av_s19 = _mm_sub_pd(av_m16, av_m18);
        av_s20 = _mm_add_pd(av_s19, av_m20);
        av_s21 = _mm_add_pd(av_s18, av_s20);

        // Output point 5: x(4)
        v_out4 = _mm_add_pd(av_s17, av_s21);
        curr_out = out + out_strides[4];
        STR_128_D(curr_out, v_out_stride, v_out4, is_contiguous_out);
        // Output point 19: x(18)
        v_out18 = _mm_sub_pd(av_s17, av_s21);
        curr_out = out + out_strides[18];
        STR_128_D(curr_out, v_out_stride, v_out18, is_contiguous_out);

        av_m21 = _mm_mul_pd(v_CRTM_11_1, av_in7);
        av_m22 = _mm_mul_pd(v_CRTM_11_2, av_in8);
        av_m23 = _mm_mul_pd(v_CRTM_11_3, av_in5);
        av_m24 = _mm_mul_pd(v_CRTM_11_4, av_in6);
        av_m25 = _mm_mul_pd(v_CRTM_11_5, av_in1);
        av_m26 = _mm_mul_pd(v_CRTM_11_6, av_in2);
        av_m27 = _mm_mul_pd(v_CRTM_11_7, av_in9);
        av_m28 = _mm_mul_pd(v_CRTM_11_8, av_in10);
        av_m29 = _mm_mul_pd(v_CRTM_11_9, av_in3);
        av_m30 = _mm_mul_pd(v_CRTM_11_10, av_in4);

        av_s22 = _mm_add_pd(av_m21, av_m23);
        av_s23 = _mm_add_pd(av_s22, av_in0);
        av_s24 = _mm_add_pd(av_m25, av_m27);
        av_s25 = _mm_add_pd(av_s24, av_m29);
        av_s26 = _mm_sub_pd(av_s23, av_s25);
        av_s27 = _mm_sub_pd(av_m22, av_m24);
        av_s28 = _mm_add_pd(av_m26, av_m28);
        av_s29 = _mm_sub_pd(av_s28, av_m30);
        av_s30 = _mm_add_pd(av_s27, av_s29);

        // Output point 7: x(6)
        v_out6 = _mm_sub_pd(av_s26, av_s30);
        curr_out = out + out_strides[6];
        STR_128_D(curr_out, v_out_stride, v_out6, is_contiguous_out);
        // Output point 17: x(16)
        v_out16 = _mm_add_pd(av_s26, av_s30);
        curr_out = out + out_strides[16];
        STR_128_D(curr_out, v_out_stride, v_out16, is_contiguous_out);

        av_m31 = _mm_mul_pd(v_CRTM_11_1, av_in5);
        av_m32 = _mm_mul_pd(v_CRTM_11_2, av_in6);
        av_m33 = _mm_mul_pd(v_CRTM_11_3, av_in9);
        av_m34 = _mm_mul_pd(v_CRTM_11_4, av_in10);
        av_m35 = _mm_mul_pd(v_CRTM_11_5, av_in3);
        av_m36 = _mm_mul_pd(v_CRTM_11_6, av_in4);
        av_m37 = _mm_mul_pd(v_CRTM_11_7, av_in1);
        av_m38 = _mm_mul_pd(v_CRTM_11_8, av_in2);
        av_m39 = _mm_mul_pd(v_CRTM_11_9, av_in7);
        av_m40 = _mm_mul_pd(v_CRTM_11_10, av_in8);

        av_s31 = _mm_add_pd(av_m31, av_m33);
        av_s32 = _mm_add_pd(av_s31, av_in0);
        av_s33 = _mm_add_pd(av_m35, av_m37);
        av_s34 = _mm_add_pd(av_s33, av_m39);
        av_s35 = _mm_sub_pd(av_s32, av_s34);
        av_s36 = _mm_sub_pd(av_m32, av_m34);
        av_s37 = _mm_sub_pd(av_m38, av_m36);
        av_s38 = _mm_add_pd(av_s37, av_m40);
        av_s39 = _mm_add_pd(av_s36, av_s38);

        // Output point 9: x(8)
        v_out8 = _mm_sub_pd(av_s35, av_s39);
        curr_out = out + out_strides[8];
        STR_128_D(curr_out, v_out_stride, v_out8, is_contiguous_out);
        // Output point 15: x(14)
        v_out14 = _mm_add_pd(av_s35, av_s39);
        curr_out = out + out_strides[14];
        STR_128_D(curr_out, v_out_stride, v_out14, is_contiguous_out);

        av_m41 = _mm_mul_pd(v_CRTM_11_1, av_in3);
        av_m42 = _mm_mul_pd(v_CRTM_11_2, av_in4);
        av_m43 = _mm_mul_pd(v_CRTM_11_3, av_in7);
        av_m44 = _mm_mul_pd(v_CRTM_11_4, av_in8);
        av_m45 = _mm_mul_pd(v_CRTM_11_5, av_in9);
        av_m46 = _mm_mul_pd(v_CRTM_11_6, av_in10);
        av_m47 = _mm_mul_pd(v_CRTM_11_7, av_in5);
        av_m48 = _mm_mul_pd(v_CRTM_11_8, av_in6);
        av_m49 = _mm_mul_pd(v_CRTM_11_9, av_in1);
        av_m50 = _mm_mul_pd(v_CRTM_11_10, av_in2);

        av_s40 = _mm_add_pd(av_m41, av_m43);
        av_s41 = _mm_add_pd(av_s40, av_in0);
        av_s42 = _mm_add_pd(av_m45, av_m47);
        av_s43 = _mm_add_pd(av_s42, av_m49);
        av_s44 = _mm_sub_pd(av_s41, av_s43);
        av_s45 = _mm_add_pd(av_m42, av_m44);
        av_s46 = _mm_add_pd(av_m46, av_m48);
        av_s47 = _mm_sub_pd(av_s45, av_s46);
        av_s48 = _mm_sub_pd(av_s47, av_m50);

        // Output point 11: x(10)
        v_out10 = _mm_add_pd(av_s44, av_s48);
        curr_out = out + out_strides[10];
        STR_128_D(curr_out, v_out_stride, v_out10, is_contiguous_out);
        // Output point 13: x(12)
        v_out12 = _mm_sub_pd(av_s44, av_s48);
        curr_out = out + out_strides[12];
        STR_128_D(curr_out, v_out_stride, v_out12, is_contiguous_out);

        /* Shifted DFT */
        __m128d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
                bv_in8, bv_in9, bv_in10;
        __m128d bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
                bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16,
                bv_s17, bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24,
                bv_s25, bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32,
                bv_s33, bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40,
                bv_s41, bv_s42, bv_s43, bv_s44, bv_s45, bv_s46, bv_s47, bv_s48;
        __m128d bv_m0, bv_m1, bv_m2, bv_m3, bv_m4, bv_m5, bv_m6, bv_m7, bv_m8,
                bv_m9, bv_m10, bv_m11, bv_m12, bv_m13, bv_m14, bv_m15, bv_m16,
                bv_m17, bv_m18, bv_m19, bv_m20, bv_m21, bv_m22, bv_m23, bv_m24,
                bv_m25, bv_m26, bv_m27, bv_m28, bv_m29, bv_m30, bv_m31, bv_m32,
                bv_m33, bv_m34, bv_m35, bv_m36, bv_m37, bv_m38, bv_m39, bv_m40,
                bv_m41, bv_m42, bv_m43, bv_m44, bv_m45, bv_m46, bv_m47, bv_m48,
                bv_m49, bv_m50;

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
        // Input point 22: X(21)
        curr_in = in + in_strides[21];
        LDR_128_D(curr_in, v_in_stride, bv_in10, is_contiguous_in);

        bv_s0 = _mm_add_pd(bv_in0, bv_in2);
        bv_s1 = _mm_add_pd(bv_in4, bv_in6);
        bv_s2 = _mm_add_pd(bv_s0, bv_s1);
        bv_s3 = _mm_add_pd(bv_s2, bv_in8);
        bv_m0 = _mm_mul_pd(v_CRTM_11_11, bv_s3);

        // Output point 2: x(1)
        v_out1 = _mm_add_pd(bv_m0, bv_in10);
        curr_out = out + out_strides[1];
        STR_128_D(curr_out, v_out_stride, v_out1, is_contiguous_out);

        bv_m1 = _mm_mul_pd(v_CRTM_11_9, bv_in0);
        bv_m2 = _mm_mul_pd(v_CRTM_11_10, bv_in1);
        bv_m3 = _mm_mul_pd(v_CRTM_11_1, bv_in8);
        bv_m4 = _mm_mul_pd(v_CRTM_11_2, bv_in9);
        bv_m5 = _mm_mul_pd(v_CRTM_11_7, bv_in2);
        bv_m6 = _mm_mul_pd(v_CRTM_11_8, bv_in3);
        bv_m7 = _mm_mul_pd(v_CRTM_11_3, bv_in6);
        bv_m8 = _mm_mul_pd(v_CRTM_11_4, bv_in7);
        bv_m9 = _mm_mul_pd(v_CRTM_11_5, bv_in4);
        bv_m10 = _mm_mul_pd(v_CRTM_11_6, bv_in5);

        bv_s4 = _mm_add_pd(bv_m2, bv_m4);
        bv_s5 = _mm_add_pd(bv_m6, bv_m8);
        bv_s6 = _mm_add_pd(bv_s4, bv_s5);
        bv_s7 = _mm_add_pd(bv_s6, bv_m10);
        bv_s8 = _mm_sub_pd(bv_m1, bv_m3);
        bv_s9 = _mm_sub_pd(bv_m5, bv_m7);
        bv_s10 = _mm_add_pd(bv_s8, bv_s9);
        bv_s11 = _mm_sub_pd(bv_m9, bv_in10);
        bv_s12 = _mm_add_pd(bv_s10, bv_s11);

        // Output point 4: x(3)
        v_out3 = _mm_sub_pd(bv_s12, bv_s7);
        curr_out = out + out_strides[3];
        STR_128_D(curr_out, v_out_stride, v_out3, is_contiguous_out);
        // Output point 22: x(21)
        v_out21 = NEGATE_128_D(_mm_add_pd(bv_s12, bv_s7));
        curr_out = out + out_strides[21];
        STR_128_D(curr_out, v_out_stride, v_out21, is_contiguous_out);

        bv_m11 = _mm_mul_pd(v_CRTM_11_1, bv_in0);
        bv_m12 = _mm_mul_pd(v_CRTM_11_2, bv_in1);
        bv_m13 = _mm_mul_pd(v_CRTM_11_5, bv_in2);
        bv_m14 = _mm_mul_pd(v_CRTM_11_6, bv_in3);
        bv_m15 = _mm_mul_pd(v_CRTM_11_9, bv_in4);
        bv_m16 = _mm_mul_pd(v_CRTM_11_10, bv_in5);
        bv_m17 = _mm_mul_pd(v_CRTM_11_7, bv_in6);
        bv_m18 = _mm_mul_pd(v_CRTM_11_8, bv_in7);
        bv_m19 = _mm_mul_pd(v_CRTM_11_3, bv_in8);
        bv_m20 = _mm_mul_pd(v_CRTM_11_4, bv_in9);

        bv_s13 = _mm_add_pd(bv_m12, bv_m16);
        bv_s14 = _mm_sub_pd(bv_m14, bv_m20);
        bv_s15 = _mm_add_pd(bv_s13, bv_s14);
        bv_s16 = _mm_sub_pd(bv_s15, bv_m18);
        bv_s17 = _mm_add_pd(bv_m11, bv_m19);
        bv_s18 = _mm_add_pd(bv_m13, bv_m17);
        bv_s19 = _mm_sub_pd(bv_s17, bv_s18);
        bv_s20 = _mm_sub_pd(bv_in10, bv_m15);
        bv_s21 = _mm_add_pd(bv_s19, bv_s20);

        // Output point 6: x(5)
        v_out5 = _mm_sub_pd(bv_s21, bv_s16);
        curr_out = out + out_strides[5];
        STR_128_D(curr_out, v_out_stride, v_out5, is_contiguous_out);
        // Output point 20: x(19)
        v_out19 = NEGATE_128_D(_mm_add_pd(bv_s21, bv_s16));
        curr_out = out + out_strides[19];
        STR_128_D(curr_out, v_out_stride, v_out19, is_contiguous_out);

        bv_m21 = _mm_mul_pd(v_CRTM_11_7, bv_in0);
        bv_m22 = _mm_mul_pd(v_CRTM_11_8, bv_in1);
        bv_m23 = _mm_mul_pd(v_CRTM_11_1, bv_in2);
        bv_m24 = _mm_mul_pd(v_CRTM_11_2, bv_in3);
        bv_m25 = _mm_mul_pd(v_CRTM_11_3, bv_in4);
        bv_m26 = _mm_mul_pd(v_CRTM_11_4, bv_in5);
        bv_m27 = _mm_mul_pd(v_CRTM_11_9, bv_in6);
        bv_m28 = _mm_mul_pd(v_CRTM_11_10, bv_in7);
        bv_m29 = _mm_mul_pd(v_CRTM_11_5, bv_in8);
        bv_m30 = _mm_mul_pd(v_CRTM_11_6, bv_in9);

        bv_s22 = _mm_add_pd(bv_m22, bv_m24);
        bv_s23 = _mm_sub_pd(bv_m30, bv_m26);
        bv_s24 = _mm_add_pd(bv_s22, bv_s23);
        bv_s25 = _mm_sub_pd(bv_s24, bv_m28);
        bv_s26 = _mm_add_pd(bv_m21, bv_m29);
        bv_s27 = _mm_sub_pd(bv_m27, bv_m23);
        bv_s28 = _mm_add_pd(bv_s26, bv_s27);
        bv_s29 = _mm_add_pd(bv_m25, bv_in10);
        bv_s30 = _mm_sub_pd(bv_s28, bv_s29);

        // Output point 8: x(7)
        v_out7 = _mm_sub_pd(bv_s30, bv_s25);
        curr_out = out + out_strides[7];
        STR_128_D(curr_out, v_out_stride, v_out7, is_contiguous_out);
        // Output point 18: x(17)
        v_out17 = NEGATE_128_D(_mm_add_pd(bv_s30, bv_s25));
        curr_out = out + out_strides[17];
        STR_128_D(curr_out, v_out_stride, v_out17, is_contiguous_out);

        bv_m31 = _mm_mul_pd(v_CRTM_11_3, bv_in0);
        bv_m32 = _mm_mul_pd(v_CRTM_11_4, bv_in1);
        bv_m33 = _mm_mul_pd(v_CRTM_11_9, bv_in2);
        bv_m34 = _mm_mul_pd(v_CRTM_11_10, bv_in3);
        bv_m35 = _mm_mul_pd(v_CRTM_11_1, bv_in4);
        bv_m36 = _mm_mul_pd(v_CRTM_11_2, bv_in5);
        bv_m37 = _mm_mul_pd(v_CRTM_11_5, bv_in6);
        bv_m38 = _mm_mul_pd(v_CRTM_11_6, bv_in7);
        bv_m39 = _mm_mul_pd(v_CRTM_11_7, bv_in8);
        bv_m40 = _mm_mul_pd(v_CRTM_11_8, bv_in9);

        bv_s31 = _mm_sub_pd(bv_m32, bv_m34);
        bv_s32 = _mm_sub_pd(bv_m38, bv_m40);
        bv_s33 = _mm_add_pd(bv_s31, bv_s32);
        bv_s34 = _mm_sub_pd(bv_s33, bv_m36);
        bv_s35 = _mm_add_pd(bv_m31, bv_m35);
        bv_s36 = _mm_add_pd(bv_m37, bv_m39);
        bv_s37 = _mm_sub_pd(bv_s35, bv_s36);
        bv_s38 = _mm_sub_pd(bv_in10, bv_m33);
        bv_s39 = _mm_add_pd(bv_s37, bv_s38);

        // Output point 10: x(9)
        v_out9 = _mm_sub_pd(bv_s39, bv_s34);
        curr_out = out + out_strides[9];
        STR_128_D(curr_out, v_out_stride, v_out9, is_contiguous_out);
        // Output point 16: x(15)
        v_out15 = NEGATE_128_D(_mm_add_pd(bv_s39, bv_s34));
        curr_out = out + out_strides[15];
        STR_128_D(curr_out, v_out_stride, v_out15, is_contiguous_out);

        bv_m41 = _mm_mul_pd(v_CRTM_11_5, bv_in0);
        bv_m42 = _mm_mul_pd(v_CRTM_11_6, bv_in1);
        bv_m43 = _mm_mul_pd(v_CRTM_11_3, bv_in2);
        bv_m44 = _mm_mul_pd(v_CRTM_11_4, bv_in3);
        bv_m45 = _mm_mul_pd(v_CRTM_11_7, bv_in4);
        bv_m46 = _mm_mul_pd(v_CRTM_11_8, bv_in5);
        bv_m47 = _mm_mul_pd(v_CRTM_11_1, bv_in6);
        bv_m48 = _mm_mul_pd(v_CRTM_11_2, bv_in7);
        bv_m49 = _mm_mul_pd(v_CRTM_11_9, bv_in8);
        bv_m50 = _mm_mul_pd(v_CRTM_11_10, bv_in9);

        bv_s40 = _mm_sub_pd(bv_m42, bv_m44);
        bv_s41 = _mm_add_pd(bv_m46, bv_m50);
        bv_s42 = _mm_add_pd(bv_s40, bv_s41);
        bv_s43 = _mm_sub_pd(bv_s42, bv_m48);
        bv_s44 = _mm_add_pd(bv_m41, bv_m45);
        bv_s45 = _mm_add_pd(bv_m43, bv_m47);
        bv_s46 = _mm_sub_pd(bv_s44, bv_s45);
        bv_s47 = _mm_sub_pd(bv_m49, bv_in10);
        bv_s48 = _mm_add_pd(bv_s46, bv_s47);

        // Output point 12: x(11)
        v_out11 = _mm_sub_pd(bv_s48, bv_s43);
        curr_out = out + out_strides[11];
        STR_128_D(curr_out, v_out_stride, v_out11, is_contiguous_out);
        // Output point 14: x(13)
        v_out13 = NEGATE_128_D(_mm_add_pd(bv_s48, bv_s43));
        curr_out = out + out_strides[13];
        STR_128_D(curr_out, v_out_stride, v_out13, is_contiguous_out);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
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
               a_m46, a_m47, a_m48, a_m49, a_m50;

        // Input point 1: X(0)
        a_in0 = *in;
        // Input point 4: X(3)
        a_in1 = in[in_strides[3]];
        // Input point 5: X(4)
        a_in2 = in[in_strides[4]];
        // Input point 6: X(7)
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
        // Input point 19: X(18)
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
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hcf_rfft11avx128(FFTZ_UINT8 precision,
                                         FFTZ_UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hcf_rfft11avx128_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft11avx128_fp64_fwd;
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
            return r2hcf_rfft11avx128_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft11avx128_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
