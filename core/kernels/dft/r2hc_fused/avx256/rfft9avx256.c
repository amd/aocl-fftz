// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft9avx256.c
 *
 *  @brief Radix-9 r2hc_fused Real-FFT kernel with AVX-256 operations using
 *  x86 SIMD intrinsics.
 *
 *  This file contains the DIT radix-9 real-to-halfcomplex fused of
 *  two different implementations (Standard DFT and Shifted DFT that
 *  differs in DFT weight matrix) using x86 SIMD intrinsics for
 *  single-precision and double-precision inputs
 *
 *  @author Amrin Fathima
 */
#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/r2hc_simd_avx256.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] = {
                                                    {{0, 44, 76, 200, 122, 33},
                                                     {0, 46, 75, 200, 143, 33}},
                                                    {{0, 44, 76, 100, 14,  33},
                                                     {0, 46, 75, 100, 14,  33}}};

ops_cycles_t get_ops_cnt_r2hcf_rfft9avx256(UINT8 precision, UINT8 direction)
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

static VOID r2hcf_rfft9avx256_fp32_fwd(VOID *in_real, VOID *in_imag,
                                       VOID *out_real, VOID *out_imag, INTP n,
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
    FLOAT *curr_in, *curr_out;
    INTP N = n / NUM_SETS_REAL_256_S;
    INTP remaining_sets = n % NUM_SETS_REAL_256_S;

    __m256 v_CRTM_9_0 = _mm256_broadcast_ss(&CRTM_9_0);
    __m256 v_CRTM_9_1 = _mm256_broadcast_ss(&CRTM_9_1);
    __m256 v_CRTM_9_2 = _mm256_broadcast_ss(&CRTM_9_2);
    __m256 v_CRTM_9_3 = _mm256_broadcast_ss(&CRTM_9_3);
    __m256 v_CRTM_9_4 = _mm256_broadcast_ss(&CRTM_9_4);
    __m256 v_CRTM_9_5 = _mm256_broadcast_ss(&CRTM_9_5);
    __m256 v_CRTM_9_6 = _mm256_broadcast_ss(&CRTM_9_6);
    __m256 v_CRTM_9_7 = _mm256_broadcast_ss(&CRTM_9_7);

    for (cnt = 0; cnt < N; cnt++)
    {
        // Standard DFT
        __m256 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
               av_in8;
        __m256 av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
               av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16, 
               av_s17, av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24, 
               av_s25, av_s26, av_s27, av_s28;
        __m256 av_m0, av_m1, av_m2, av_m3, av_m4, av_m5, av_m6, av_m7, av_m8,
               av_m9, av_m10, av_m11, av_m12, av_m13, av_m14, av_m15, av_m16, 
               av_m17, av_m18, av_m19, av_m20, av_m21;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15,v_out16, v_out17;

        curr_in = in;
        curr_out = out;

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

        av_s0 = _mm256_add_ps(av_in1, av_in8);
        av_s1 = _mm256_sub_ps(av_in1, av_in8);
        av_s2 = _mm256_add_ps(av_in2, av_in7);
        av_s3 = _mm256_sub_ps(av_in2, av_in7);
        av_s4 = _mm256_add_ps(av_in3, av_in6);
        av_s5 = _mm256_sub_ps(av_in3, av_in6);
        av_s6 = _mm256_add_ps(av_in4, av_in5);
        av_s7 = _mm256_sub_ps(av_in4, av_in5);

        av_s8 = _mm256_add_ps(av_s0, av_s6);
        av_s9 = _mm256_add_ps(av_s8, av_s2);
        av_s10 = _mm256_sub_ps(av_s1, av_s3);
        av_s11 = _mm256_add_ps(av_s10, av_s7);
        av_s12 = _mm256_add_ps(av_s9, av_s4);
        av_s13 = _mm256_add_ps(av_s12, av_in0);
        av_m0 = _mm256_mul_ps(v_CRTM_9_4, av_s4);
        av_m1 = _mm256_mul_ps(v_CRTM_9_5, av_s5);
        av_s14 = _mm256_sub_ps(av_in0, av_m0);

        av_m2 = _mm256_mul_ps(v_CRTM_9_0, av_s0);
        av_m3 = _mm256_mul_ps(v_CRTM_9_2, av_s2);
        av_m4 = _mm256_mul_ps(v_CRTM_9_6, av_s6);
        av_m5 = _mm256_mul_ps(v_CRTM_9_1, av_s1);
        av_m6 = _mm256_mul_ps(v_CRTM_9_3, av_s3);
        av_m7 = _mm256_mul_ps(v_CRTM_9_7, av_s7);
        av_m8 = _mm256_mul_ps(v_CRTM_9_0, av_s2);
        av_m9 = _mm256_mul_ps(v_CRTM_9_2, av_s0);
        av_m10 = _mm256_mul_ps(v_CRTM_9_6, av_s2);
        av_m11 = _mm256_mul_ps(v_CRTM_9_0, av_s6);
        av_m12 = _mm256_mul_ps(v_CRTM_9_1, av_s7);
        av_m13 = _mm256_mul_ps(v_CRTM_9_3, av_s1);
        av_m14 = _mm256_mul_ps(v_CRTM_9_7, av_s3);

        // Output point 1: X(0)
        v_out0 = av_s13;
        STR_256_S(curr_out, v_out_stride, v_out0);

        av_s15 = _mm256_add_ps(av_s14, av_m2);
        av_s16 = _mm256_sub_ps(av_m3, av_m4);
        // Output point 4: X(3)
        v_out3 = _mm256_add_ps(av_s15, av_s16);

        av_s17 = _mm256_add_ps(av_m5, av_m6);
        av_s18 = _mm256_add_ps(av_m7, av_m1);
        av_s19 = NEGATE_256_S(_mm256_add_ps(av_s17, av_s18));
        // Output point 5: X(4)
        v_out4 = av_s19;
        curr_out = out + out_strides[3];
        STRI_2x256_S(curr_out, v_out_stride, v_out3, v_out4);

        av_s20 = _mm256_add_ps(av_m11, av_m9);
        av_s21 = _mm256_sub_ps(av_s20, av_m10);
        // Output point 8: X(7)
        v_out7 = _mm256_add_ps(av_s21, av_s14);

        av_s22 = _mm256_add_ps(av_m13, av_m14);
        av_s23 = _mm256_sub_ps(av_m1, av_s22);
        // Output point 9: X(8)
        v_out8 = _mm256_add_ps(av_s23, av_m12);
        curr_out = out + out_strides[7];
        STRI_2x256_S(curr_out, v_out_stride, v_out7, v_out8);

        av_m15 = _mm256_mul_ps(v_CRTM_9_4, av_s9);
        av_s24 = _mm256_sub_ps(av_s4, av_m15);
        // Output point 12: X(11)
        v_out11 = _mm256_add_ps(av_in0, av_s24);

        av_m16 = NEGATE_256_S(_mm256_mul_ps(v_CRTM_9_5, av_s11));
        // Output point 13: X(12)
        v_out12 = av_m16;
        curr_out = out + out_strides[11];
        STRI_2x256_S(curr_out, v_out_stride, v_out11, v_out12);

        av_s25 = _mm256_add_ps(av_s14, av_m8);
        av_m17 = _mm256_mul_ps(v_CRTM_9_6, av_s0);
        av_s26 = _mm256_sub_ps(av_s25, av_m17);
        av_m18 = _mm256_mul_ps(v_CRTM_9_2, av_s6);
        // Output point 16: X(15)
        v_out15 = _mm256_add_ps(av_s26, av_m18);

        av_m19 = _mm256_mul_ps(v_CRTM_9_1, av_s3);
        av_m20 = _mm256_mul_ps(v_CRTM_9_3, av_s7);
        av_s27 = _mm256_add_ps(av_m19, av_m20);
        av_s28 = _mm256_sub_ps(av_s27, av_m1);
        av_m21 = _mm256_mul_ps(v_CRTM_9_7, av_s1);
        // Output point 17: X(16)
        v_out16 = _mm256_sub_ps(av_s28, av_m21);
        curr_out = out + out_strides[15];
        STRI_2x256_S(curr_out, v_out_stride, v_out15, v_out16);
 
         // Shifted DFT
        __m256 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8;
        __m256 bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
               bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, 
               bv_s17, bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24, 
               bv_s25, bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32, 
               bv_s33, bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40;
        __m256 bv_m0, bv_m1, bv_m2, bv_m3, bv_m4, bv_m5, bv_m6, bv_m7, bv_m8,
               bv_m9, bv_m10, bv_m11, bv_m12, bv_m13, bv_m14, bv_m15, bv_m16, 
               bv_m17, bv_m18, bv_m19;

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

        bv_s0 = _mm256_add_ps(bv_in1, bv_in8);
        bv_s1 = _mm256_sub_ps(bv_in1, bv_in8);
        bv_s2 = _mm256_add_ps(bv_in2, bv_in7);
        bv_s3 = _mm256_sub_ps(bv_in2, bv_in7);
        bv_s4 = _mm256_add_ps(bv_in3, bv_in6);
        bv_s5 = _mm256_sub_ps(bv_in3, bv_in6);
        bv_s6 = _mm256_add_ps(bv_in4, bv_in5);
        bv_s7 = _mm256_sub_ps(bv_in4, bv_in5);

        bv_m0 = _mm256_mul_ps(v_CRTM_9_6, bv_s1);
        bv_m1 = _mm256_mul_ps(v_CRTM_9_6, bv_s3);
        bv_m2 = _mm256_mul_ps(v_CRTM_9_6, bv_s7);
        bv_m3 = _mm256_mul_ps(v_CRTM_9_0, bv_s3);
        bv_m4 = _mm256_mul_ps(v_CRTM_9_0, bv_s7);
        bv_m5 = _mm256_mul_ps(v_CRTM_9_0, bv_s1);
        bv_m6 = _mm256_mul_ps(v_CRTM_9_4, bv_s5);
        bv_m7 = _mm256_mul_ps(v_CRTM_9_2, bv_s7);
        bv_m8 = _mm256_mul_ps(v_CRTM_9_2, bv_s1);
        bv_m9 = _mm256_mul_ps(v_CRTM_9_2, bv_s3);

        bv_m10 = _mm256_mul_ps(v_CRTM_9_7, bv_s0);
        bv_m11 = _mm256_mul_ps(v_CRTM_9_7, bv_s2);
        bv_m12 = _mm256_mul_ps(v_CRTM_9_7, bv_s6);
        bv_m13 = _mm256_mul_ps(v_CRTM_9_1, bv_s2);
        bv_m14 = _mm256_mul_ps(v_CRTM_9_1, bv_s6);
        bv_m15 = _mm256_mul_ps(v_CRTM_9_1, bv_s0);
        bv_m16 = _mm256_mul_ps(v_CRTM_9_3, bv_s0);
        bv_m17 = _mm256_mul_ps(v_CRTM_9_3, bv_s2);
        bv_m18 = _mm256_mul_ps(v_CRTM_9_3, bv_s6);
        bv_m19 = _mm256_mul_ps(v_CRTM_9_5, bv_s4);

        bv_s8 = _mm256_add_ps(bv_m0, bv_m3);
        bv_s9 = _mm256_add_ps(bv_s8, bv_m6);
        bv_s10 = _mm256_add_ps(bv_s9, bv_m7);
        bv_s11 = _mm256_add_ps(bv_s10, bv_in0);
        bv_s12 = _mm256_add_ps(bv_m10, bv_m13);
        bv_s13 = _mm256_add_ps(bv_s12, bv_m19);
        bv_s14 = NEGATE_256_S(_mm256_add_ps(bv_s13, bv_m18));
        // Output point 2: X(1)
        v_out1 = bv_s11;
        // Output point 3: X(2)
        v_out2 = bv_s14;
        curr_out = out + out_strides[1];
        STRI_2x256_S(curr_out, v_out_stride, v_out1, v_out2);

        bv_s15 = _mm256_sub_ps(bv_s1, bv_s3);
        bv_s16 = _mm256_sub_ps(bv_s15, bv_s7);
        bv_s17 = _mm256_mul_ps(v_CRTM_9_4, bv_s16);
        bv_s18 = _mm256_sub_ps(bv_s6, bv_s0);
        bv_s19 = _mm256_sub_ps(bv_s18, bv_s2);
        bv_s20 = _mm256_mul_ps(v_CRTM_9_5, bv_s19);
        bv_s21 = _mm256_add_ps(bv_s17, bv_in0);
        bv_s22 = _mm256_sub_ps(bv_s21, bv_s5);
        // Output point 6: X(5)
        v_out5 = bv_s22;
        // Output point 7: X(6)
        v_out6 = bv_s20;
        curr_out = out + out_strides[5];
        STRI_2x256_S(curr_out, v_out_stride, v_out5, v_out6);

        bv_s23 = _mm256_sub_ps(bv_m4, bv_m1);
        bv_s24 = _mm256_add_ps(bv_s23, bv_m6);
        bv_s25 = _mm256_sub_ps(bv_s24, bv_m8);
        bv_s26 = _mm256_add_ps(bv_s25, bv_in0);
        bv_s27 = _mm256_sub_ps(bv_m11, bv_m14);
        bv_s28 = _mm256_add_ps(bv_s27, bv_m19);
        bv_s29 = _mm256_sub_ps(bv_s28, bv_m16);
        // Output point 10: X(9)
        v_out9 = bv_s26;
        // Output point 11: X(10)
        v_out10 = bv_s29;
        curr_out = out + out_strides[9];
        STRI_2x256_S(curr_out, v_out_stride, v_out9, v_out10);

        bv_s30 = _mm256_sub_ps(bv_m6, bv_m2);
        bv_s31 = _mm256_sub_ps(bv_s30, bv_m5);
        bv_s32 = _mm256_add_ps(bv_s31, bv_m9);
        bv_s33 = _mm256_add_ps(bv_s32, bv_in0);
        bv_s34 = _mm256_sub_ps(bv_m12, bv_m15);
        bv_s35 = _mm256_sub_ps(bv_s34, bv_m19);
        bv_s36 = _mm256_add_ps(bv_s35, bv_m17);
        // Output point 14: X(13)
        v_out13 = bv_s33;
        // Output point 15: X(14)
        v_out14 = bv_s36;
        curr_out = out + out_strides[13];
        STRI_2x256_S(curr_out, v_out_stride, v_out13, v_out14);

        bv_s37 = _mm256_sub_ps(bv_s3, bv_s5);
        bv_s38 = _mm256_sub_ps(bv_s37, bv_s1);
        bv_s39 = _mm256_add_ps(bv_s38, bv_in0);
        bv_s40 = _mm256_add_ps(bv_s39, bv_s7);
        // Output point 18: X(17)
        v_out17 = bv_s40;
        curr_out = out + out_strides[17];
        STR_256_S(curr_out, v_out_stride, v_out17);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_S)
    {
        // Standard DFT
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
               av_in8;
        __m128 av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
               av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16,
               av_s17, av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24, 
               av_s25, av_s26, av_s27, av_s28;
        __m128 av_m0, av_m1, av_m2, av_m3, av_m4, av_m5, av_m6, av_m7, av_m8,
               av_m9, av_m10, av_m11, av_m12, av_m13, av_m14, av_m15, av_m16, 
               av_m17, av_m18, av_m19, av_m20, av_m21;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_9_0 = _mm256_castps256_ps128(v_CRTM_9_0);
        __m128 v128_CRTM_9_1 = _mm256_castps256_ps128(v_CRTM_9_1);
        __m128 v128_CRTM_9_2 = _mm256_castps256_ps128(v_CRTM_9_2);
        __m128 v128_CRTM_9_3 = _mm256_castps256_ps128(v_CRTM_9_3);
        __m128 v128_CRTM_9_4 = _mm256_castps256_ps128(v_CRTM_9_4);
        __m128 v128_CRTM_9_5 = _mm256_castps256_ps128(v_CRTM_9_5);
        __m128 v128_CRTM_9_6 = _mm256_castps256_ps128(v_CRTM_9_6);
        __m128 v128_CRTM_9_7 = _mm256_castps256_ps128(v_CRTM_9_7);

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

        av_s0 = _mm_add_ps(av_in1, av_in8);
        av_s1 = _mm_sub_ps(av_in1, av_in8);
        av_s2 = _mm_add_ps(av_in2, av_in7);
        av_s3 = _mm_sub_ps(av_in2, av_in7);
        av_s4 = _mm_add_ps(av_in3, av_in6);
        av_s5 = _mm_sub_ps(av_in3, av_in6);
        av_s6 = _mm_add_ps(av_in4, av_in5);
        av_s7 = _mm_sub_ps(av_in4, av_in5);

        av_s8 = _mm_add_ps(av_s0, av_s6);
        av_s9 = _mm_add_ps(av_s8, av_s2);
        av_s10 = _mm_sub_ps(av_s1, av_s3);
        av_s11 = _mm_add_ps(av_s10, av_s7);
        av_s12 = _mm_add_ps(av_s9, av_s4);
        av_s13 = _mm_add_ps(av_s12, av_in0);
        av_m0 = _mm_mul_ps(v128_CRTM_9_4, av_s4);
        av_m1 = _mm_mul_ps(v128_CRTM_9_5, av_s5);
        av_s14 = _mm_sub_ps(av_in0, av_m0);

        av_m2 = _mm_mul_ps(v128_CRTM_9_0, av_s0);
        av_m3 = _mm_mul_ps(v128_CRTM_9_2, av_s2);
        av_m4 = _mm_mul_ps(v128_CRTM_9_6, av_s6);
        av_m5 = _mm_mul_ps(v128_CRTM_9_1, av_s1);
        av_m6 = _mm_mul_ps(v128_CRTM_9_3, av_s3);
        av_m7 = _mm_mul_ps(v128_CRTM_9_7, av_s7);
        av_m8 = _mm_mul_ps(v128_CRTM_9_0, av_s2);
        av_m9 = _mm_mul_ps(v128_CRTM_9_2, av_s0);
        av_m10 = _mm_mul_ps(v128_CRTM_9_6, av_s2);
        av_m11 = _mm_mul_ps(v128_CRTM_9_0, av_s6);
        av_m12 = _mm_mul_ps(v128_CRTM_9_1, av_s7);
        av_m13 = _mm_mul_ps(v128_CRTM_9_3, av_s1);
        av_m14 = _mm_mul_ps(v128_CRTM_9_7, av_s3);

        // Output point 1: X(0)
        v_out0 = av_s13;
        STR_128_S(curr_out, v_out_stride, v_out0);

        av_s15 = _mm_add_ps(av_s14, av_m2);
        av_s16 = _mm_sub_ps(av_m3, av_m4);
        // Output point 4: X(3)
        v_out3 = _mm_add_ps(av_s15, av_s16);

        av_s17 = _mm_add_ps(av_m5, av_m6);
        av_s18 = _mm_add_ps(av_m7, av_m1);
        av_s19 = NEGATE_128_S(_mm_add_ps(av_s17, av_s18));
        // Output point 5: X(4)
        v_out4 = av_s19;
        curr_out = out + out_strides[3];
        STRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);

        av_s20 = _mm_add_ps(av_m11, av_m9);
        av_s21 = _mm_sub_ps(av_s20, av_m10);
        // Output point 8: X(7)
        v_out7 = _mm_add_ps(av_s21, av_s14);

        av_s22 = _mm_add_ps(av_m13, av_m14);
        av_s23 = _mm_sub_ps(av_m1, av_s22);
        // Output point 9: X(8)
        v_out8 = _mm_add_ps(av_s23, av_m12);
        curr_out = out + out_strides[7];
        STRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);

        av_m15 = _mm_mul_ps(v128_CRTM_9_4, av_s9);
        av_s24 = _mm_sub_ps(av_s4, av_m15);
        // Output point 12: X(11)
        v_out11 = _mm_add_ps(av_in0, av_s24);

        av_m16 = NEGATE_128_S(_mm_mul_ps(v128_CRTM_9_5, av_s11));
        // Output point 13: X(12)
        v_out12 = av_m16;
        curr_out = out + out_strides[11];
        STRI_2x128_S(curr_out, v_out_stride, v_out11, v_out12);

        av_s25 = _mm_add_ps(av_s14, av_m8);
        av_m17 = _mm_mul_ps(v128_CRTM_9_6, av_s0);
        av_s26 = _mm_sub_ps(av_s25, av_m17);
        av_m18 = _mm_mul_ps(v128_CRTM_9_2, av_s6);
        // Output point 16: X(15)
        v_out15 = _mm_add_ps(av_s26, av_m18);

        av_m19 = _mm_mul_ps(v128_CRTM_9_1, av_s3);
        av_m20 = _mm_mul_ps(v128_CRTM_9_3, av_s7);
        av_s27 = _mm_add_ps(av_m19, av_m20);
        av_s28 = _mm_sub_ps(av_s27, av_m1);
        av_m21 = _mm_mul_ps(v128_CRTM_9_7, av_s1);
        // Output point 17: X(16)
        v_out16 = _mm_sub_ps(av_s28, av_m21);
        curr_out = out + out_strides[15];
        STRI_2x128_S(curr_out, v_out_stride, v_out15, v_out16);

        // Shifted DFT
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8;
        __m128 bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
               bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, 
               bv_s17, bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24, 
               bv_s25, bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32, 
               bv_s33, bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40;
        __m128 bv_m0, bv_m1, bv_m2, bv_m3, bv_m4, bv_m5, bv_m6, bv_m7, bv_m8,
               bv_m9, bv_m10, bv_m11, bv_m12, bv_m13, bv_m14, bv_m15, bv_m16, 
               bv_m17, bv_m18, bv_m19;

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

        bv_s0 = _mm_add_ps(bv_in1, bv_in8);
        bv_s1 = _mm_sub_ps(bv_in1, bv_in8);
        bv_s2 = _mm_add_ps(bv_in2, bv_in7);
        bv_s3 = _mm_sub_ps(bv_in2, bv_in7);
        bv_s4 = _mm_add_ps(bv_in3, bv_in6);
        bv_s5 = _mm_sub_ps(bv_in3, bv_in6);
        bv_s6 = _mm_add_ps(bv_in4, bv_in5);
        bv_s7 = _mm_sub_ps(bv_in4, bv_in5);

        bv_m0 = _mm_mul_ps(v128_CRTM_9_6, bv_s1);
        bv_m1 = _mm_mul_ps(v128_CRTM_9_6, bv_s3);
        bv_m2 = _mm_mul_ps(v128_CRTM_9_6, bv_s7);
        bv_m3 = _mm_mul_ps(v128_CRTM_9_0, bv_s3);
        bv_m4 = _mm_mul_ps(v128_CRTM_9_0, bv_s7);
        bv_m5 = _mm_mul_ps(v128_CRTM_9_0, bv_s1);
        bv_m6 = _mm_mul_ps(v128_CRTM_9_4, bv_s5);
        bv_m7 = _mm_mul_ps(v128_CRTM_9_2, bv_s7);
        bv_m8 = _mm_mul_ps(v128_CRTM_9_2, bv_s1);
        bv_m9 = _mm_mul_ps(v128_CRTM_9_2, bv_s3);

        bv_m10 = _mm_mul_ps(v128_CRTM_9_7, bv_s0);
        bv_m11 = _mm_mul_ps(v128_CRTM_9_7, bv_s2);
        bv_m12 = _mm_mul_ps(v128_CRTM_9_7, bv_s6);
        bv_m13 = _mm_mul_ps(v128_CRTM_9_1, bv_s2);
        bv_m14 = _mm_mul_ps(v128_CRTM_9_1, bv_s6);
        bv_m15 = _mm_mul_ps(v128_CRTM_9_1, bv_s0);
        bv_m16 = _mm_mul_ps(v128_CRTM_9_3, bv_s0);
        bv_m17 = _mm_mul_ps(v128_CRTM_9_3, bv_s2);
        bv_m18 = _mm_mul_ps(v128_CRTM_9_3, bv_s6);
        bv_m19 = _mm_mul_ps(v128_CRTM_9_5, bv_s4);

        bv_s8 = _mm_add_ps(bv_m0, bv_m3);
        bv_s9 = _mm_add_ps(bv_s8, bv_m6);
        bv_s10 = _mm_add_ps(bv_s9, bv_m7);
        bv_s11 = _mm_add_ps(bv_s10, bv_in0);
        bv_s12 = _mm_add_ps(bv_m10, bv_m13);
        bv_s13 = _mm_add_ps(bv_s12, bv_m19);
        bv_s14 = NEGATE_128_S(_mm_add_ps(bv_s13, bv_m18));
        // Output point 2: X(1)
        v_out1 = bv_s11;
        // Output point 3: X(2)
        v_out2 = bv_s14;
        curr_out = out + out_strides[1];
        STRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);

        bv_s15 = _mm_sub_ps(bv_s1, bv_s3);
        bv_s16 = _mm_sub_ps(bv_s15, bv_s7);
        bv_s17 = _mm_mul_ps(v128_CRTM_9_4, bv_s16);
        bv_s18 = _mm_sub_ps(bv_s6, bv_s0);
        bv_s19 = _mm_sub_ps(bv_s18, bv_s2);
        bv_s20 = _mm_mul_ps(v128_CRTM_9_5, bv_s19);
        bv_s21 = _mm_add_ps(bv_s17, bv_in0);
        bv_s22 = _mm_sub_ps(bv_s21, bv_s5);
        // Output point 6: X(5)
        v_out5 = bv_s22;
        // Output point 7: X(6)
        v_out6 = bv_s20;
        curr_out = out + out_strides[5];
        STRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);

        bv_s23 = _mm_sub_ps(bv_m4, bv_m1);
        bv_s24 = _mm_add_ps(bv_s23, bv_m6);
        bv_s25 = _mm_sub_ps(bv_s24, bv_m8);
        bv_s26 = _mm_add_ps(bv_s25, bv_in0);
        bv_s27 = _mm_sub_ps(bv_m11, bv_m14);
        bv_s28 = _mm_add_ps(bv_s27, bv_m19);
        bv_s29 = _mm_sub_ps(bv_s28, bv_m16);
        // Output point 10: X(9)
        v_out9 = bv_s26;
        // Output point 11: X(10)
        v_out10 = bv_s29;
        curr_out = out + out_strides[9];
        STRI_2x128_S(curr_out, v_out_stride, v_out9, v_out10);

        bv_s30 = _mm_sub_ps(bv_m6, bv_m2);
        bv_s31 = _mm_sub_ps(bv_s30, bv_m5);
        bv_s32 = _mm_add_ps(bv_s31, bv_m9);
        bv_s33 = _mm_add_ps(bv_s32, bv_in0);
        bv_s34 = _mm_sub_ps(bv_m12, bv_m15);
        bv_s35 = _mm_sub_ps(bv_s34, bv_m19);
        bv_s36 = _mm_add_ps(bv_s35, bv_m17);
        // Output point 14: X(13)
        v_out13 = bv_s33;
        // Output point 15: X(14)
        v_out14 = bv_s36;
        curr_out = out + out_strides[13];
        STRI_2x128_S(curr_out, v_out_stride, v_out13, v_out14);

        bv_s37 = _mm_sub_ps(bv_s3, bv_s5);
        bv_s38 = _mm_sub_ps(bv_s37, bv_s1);
        bv_s39 = _mm_add_ps(bv_s38, bv_in0);
        bv_s40 = _mm_add_ps(bv_s39, bv_s7);
        // Output point 18: X(17)
        v_out17 = bv_s40;
        curr_out = out + out_strides[17];
        STR_128_S(curr_out, v_out_stride, v_out17);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & 2)
    {
        // Standard DFT
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
               av_in8;
        __m128 av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
               av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16,
               av_s17, av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24, 
               av_s25, av_s26, av_s27, av_s28;
        __m128 av_m0, av_m1, av_m2, av_m3, av_m4, av_m5, av_m6, av_m7, av_m8,
               av_m9, av_m10, av_m11, av_m12, av_m13, av_m14, av_m15, av_m16,
               av_m17, av_m18, av_m19, av_m20, av_m21;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_9_0 = _mm256_castps256_ps128(v_CRTM_9_0);
        __m128 v128_CRTM_9_1 = _mm256_castps256_ps128(v_CRTM_9_1);
        __m128 v128_CRTM_9_2 = _mm256_castps256_ps128(v_CRTM_9_2);
        __m128 v128_CRTM_9_3 = _mm256_castps256_ps128(v_CRTM_9_3);
        __m128 v128_CRTM_9_4 = _mm256_castps256_ps128(v_CRTM_9_4);
        __m128 v128_CRTM_9_5 = _mm256_castps256_ps128(v_CRTM_9_5);
        __m128 v128_CRTM_9_6 = _mm256_castps256_ps128(v_CRTM_9_6);
        __m128 v128_CRTM_9_7 = _mm256_castps256_ps128(v_CRTM_9_7);

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

        av_s0 = _mm_add_ps(av_in1, av_in8);
        av_s1 = _mm_sub_ps(av_in1, av_in8);
        av_s2 = _mm_add_ps(av_in2, av_in7);
        av_s3 = _mm_sub_ps(av_in2, av_in7);
        av_s4 = _mm_add_ps(av_in3, av_in6);
        av_s5 = _mm_sub_ps(av_in3, av_in6);
        av_s6 = _mm_add_ps(av_in4, av_in5);
        av_s7 = _mm_sub_ps(av_in4, av_in5);

        av_s8 = _mm_add_ps(av_s0, av_s6);
        av_s9 = _mm_add_ps(av_s8, av_s2);
        av_s10 = _mm_sub_ps(av_s1, av_s3);
        av_s11 = _mm_add_ps(av_s10, av_s7);
        av_s12 = _mm_add_ps(av_s9, av_s4);
        av_s13 = _mm_add_ps(av_s12, av_in0);
        av_m0 = _mm_mul_ps(v128_CRTM_9_4, av_s4);
        av_m1 = _mm_mul_ps(v128_CRTM_9_5, av_s5);
        av_s14 = _mm_sub_ps(av_in0, av_m0);

        av_m2 = _mm_mul_ps(v128_CRTM_9_0, av_s0);
        av_m3 = _mm_mul_ps(v128_CRTM_9_2, av_s2);
        av_m4 = _mm_mul_ps(v128_CRTM_9_6, av_s6);
        av_m5 = _mm_mul_ps(v128_CRTM_9_1, av_s1);
        av_m6 = _mm_mul_ps(v128_CRTM_9_3, av_s3);
        av_m7 = _mm_mul_ps(v128_CRTM_9_7, av_s7);
        av_m8 = _mm_mul_ps(v128_CRTM_9_0, av_s2);
        av_m9 = _mm_mul_ps(v128_CRTM_9_2, av_s0);
        av_m10 = _mm_mul_ps(v128_CRTM_9_6, av_s2);
        av_m11 = _mm_mul_ps(v128_CRTM_9_0, av_s6);
        av_m12 = _mm_mul_ps(v128_CRTM_9_1, av_s7);
        av_m13 = _mm_mul_ps(v128_CRTM_9_3, av_s1);
        av_m14 = _mm_mul_ps(v128_CRTM_9_7, av_s3);

        // Output point 1: X(0)
        v_out0 = av_s13;
        STHR_128_S(curr_out, v_out_stride, v_out0);

        av_s15 = _mm_add_ps(av_s14, av_m2);
        av_s16 = _mm_sub_ps(av_m3, av_m4);
        // Output point 4: X(3)
        v_out3 = _mm_add_ps(av_s15, av_s16);

        av_s17 = _mm_add_ps(av_m5, av_m6);
        av_s18 = _mm_add_ps(av_m7, av_m1);
        av_s19 = NEGATE_128_S(_mm_add_ps(av_s17, av_s18));
        // Output point 5: X(4)
        v_out4 = av_s19;
        curr_out = out + out_strides[3];
        STHRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);

        av_s20 = _mm_add_ps(av_m11, av_m9);
        av_s21 = _mm_sub_ps(av_s20, av_m10);
        // Output point 8: X(7)
        v_out7 = _mm_add_ps(av_s21, av_s14);

        av_s22 = _mm_add_ps(av_m13, av_m14);
        av_s23 = _mm_sub_ps(av_m1, av_s22);
        // Output point 9: X(8)
        v_out8 = _mm_add_ps(av_s23, av_m12);
        curr_out = out + out_strides[7];
        STHRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);

        av_m15 = _mm_mul_ps(v128_CRTM_9_4, av_s9);
        av_s24 = _mm_sub_ps(av_s4, av_m15);
        // Output point 12: X(11)
        v_out11 = _mm_add_ps(av_in0, av_s24);

        av_m16 = NEGATE_128_S(_mm_mul_ps(v128_CRTM_9_5, av_s11));
        // Output point 13: X(12)
        v_out12 = av_m16;
        curr_out = out + out_strides[11];
        STHRI_2x128_S(curr_out, v_out_stride, v_out11, v_out12);

        av_s25 = _mm_add_ps(av_s14, av_m8);
        av_m17 = _mm_mul_ps(v128_CRTM_9_6, av_s0);
        av_s26 = _mm_sub_ps(av_s25, av_m17);
        av_m18 = _mm_mul_ps(v128_CRTM_9_2, av_s6);
        // Output point 16: X(15)
        v_out15 = _mm_add_ps(av_s26, av_m18);

        av_m19 = _mm_mul_ps(v128_CRTM_9_1, av_s3);
        av_m20 = _mm_mul_ps(v128_CRTM_9_3, av_s7);
        av_s27 = _mm_add_ps(av_m19, av_m20);
        av_s28 = _mm_sub_ps(av_s27, av_m1);
        av_m21 = _mm_mul_ps(v128_CRTM_9_7, av_s1);
        // Output point 17: X(16)
        v_out16 = _mm_sub_ps(av_s28, av_m21);
        curr_out = out + out_strides[15];
        STHRI_2x128_S(curr_out, v_out_stride, v_out15, v_out16);
 
         // Shifted DFT
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8;
        __m128 bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
               bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16,
               bv_s17, bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24,
               bv_s25, bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32,
               bv_s33, bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40;
        __m128 bv_m0, bv_m1, bv_m2, bv_m3, bv_m4, bv_m5, bv_m6, bv_m7, bv_m8,
               bv_m9, bv_m10, bv_m11, bv_m12, bv_m13, bv_m14, bv_m15, bv_m16,
               bv_m17, bv_m18, bv_m19;

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

        bv_s0 = _mm_add_ps(bv_in1, bv_in8);
        bv_s1 = _mm_sub_ps(bv_in1, bv_in8);
        bv_s2 = _mm_add_ps(bv_in2, bv_in7);
        bv_s3 = _mm_sub_ps(bv_in2, bv_in7);
        bv_s4 = _mm_add_ps(bv_in3, bv_in6);
        bv_s5 = _mm_sub_ps(bv_in3, bv_in6);
        bv_s6 = _mm_add_ps(bv_in4, bv_in5);
        bv_s7 = _mm_sub_ps(bv_in4, bv_in5);

        bv_m0 = _mm_mul_ps(v128_CRTM_9_6, bv_s1);
        bv_m1 = _mm_mul_ps(v128_CRTM_9_6, bv_s3);
        bv_m2 = _mm_mul_ps(v128_CRTM_9_6, bv_s7);
        bv_m3 = _mm_mul_ps(v128_CRTM_9_0, bv_s3);
        bv_m4 = _mm_mul_ps(v128_CRTM_9_0, bv_s7);
        bv_m5 = _mm_mul_ps(v128_CRTM_9_0, bv_s1);
        bv_m6 = _mm_mul_ps(v128_CRTM_9_4, bv_s5);
        bv_m7 = _mm_mul_ps(v128_CRTM_9_2, bv_s7);
        bv_m8 = _mm_mul_ps(v128_CRTM_9_2, bv_s1);
        bv_m9 = _mm_mul_ps(v128_CRTM_9_2, bv_s3);

        bv_m10 = _mm_mul_ps(v128_CRTM_9_7, bv_s0);
        bv_m11 = _mm_mul_ps(v128_CRTM_9_7, bv_s2);
        bv_m12 = _mm_mul_ps(v128_CRTM_9_7, bv_s6);
        bv_m13 = _mm_mul_ps(v128_CRTM_9_1, bv_s2);
        bv_m14 = _mm_mul_ps(v128_CRTM_9_1, bv_s6);
        bv_m15 = _mm_mul_ps(v128_CRTM_9_1, bv_s0);
        bv_m16 = _mm_mul_ps(v128_CRTM_9_3, bv_s0);
        bv_m17 = _mm_mul_ps(v128_CRTM_9_3, bv_s2);
        bv_m18 = _mm_mul_ps(v128_CRTM_9_3, bv_s6);
        bv_m19 = _mm_mul_ps(v128_CRTM_9_5, bv_s4);

        bv_s8 = _mm_add_ps(bv_m0, bv_m3);
        bv_s9 = _mm_add_ps(bv_s8, bv_m6);
        bv_s10 = _mm_add_ps(bv_s9, bv_m7);
        bv_s11 = _mm_add_ps(bv_s10, bv_in0);
        bv_s12 = _mm_add_ps(bv_m10, bv_m13);
        bv_s13 = _mm_add_ps(bv_s12, bv_m19);
        bv_s14 = NEGATE_128_S(_mm_add_ps(bv_s13, bv_m18));
        // Output point 2: X(1)
        v_out1 = bv_s11;
        // Output point 3: X(2)
        v_out2 = bv_s14;
        curr_out = out + out_strides[1];
        STHRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);

        bv_s15 = _mm_sub_ps(bv_s1, bv_s3);
        bv_s16 = _mm_sub_ps(bv_s15, bv_s7);
        bv_s17 = _mm_mul_ps(v128_CRTM_9_4, bv_s16);
        bv_s18 = _mm_sub_ps(bv_s6, bv_s0);
        bv_s19 = _mm_sub_ps(bv_s18, bv_s2);
        bv_s20 = _mm_mul_ps(v128_CRTM_9_5, bv_s19);
        bv_s21 = _mm_add_ps(bv_s17, bv_in0);
        bv_s22 = _mm_sub_ps(bv_s21, bv_s5);
        // Output point 6: X(5)
        v_out5 = bv_s22;
        // Output point 7: X(6)
        v_out6 = bv_s20;
        curr_out = out + out_strides[5];
        STHRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);

        bv_s23 = _mm_sub_ps(bv_m4, bv_m1);
        bv_s24 = _mm_add_ps(bv_s23, bv_m6);
        bv_s25 = _mm_sub_ps(bv_s24, bv_m8);
        bv_s26 = _mm_add_ps(bv_s25, bv_in0);
        bv_s27 = _mm_sub_ps(bv_m11, bv_m14);
        bv_s28 = _mm_add_ps(bv_s27, bv_m19);
        bv_s29 = _mm_sub_ps(bv_s28, bv_m16);
        // Output point 10: X(9)
        v_out9 = bv_s26;
        // Output point 11: X(10)
        v_out10 = bv_s29;
        curr_out = out + out_strides[9];
        STHRI_2x128_S(curr_out, v_out_stride, v_out9, v_out10);

        bv_s30 = _mm_sub_ps(bv_m6, bv_m2);
        bv_s31 = _mm_sub_ps(bv_s30, bv_m5);
        bv_s32 = _mm_add_ps(bv_s31, bv_m9);
        bv_s33 = _mm_add_ps(bv_s32, bv_in0);
        bv_s34 = _mm_sub_ps(bv_m12, bv_m15);
        bv_s35 = _mm_sub_ps(bv_s34, bv_m19);
        bv_s36 = _mm_add_ps(bv_s35, bv_m17);
        // Output point 14: X(13)
        v_out13 = bv_s33;
        // Output point 15: X(14)
        v_out14 = bv_s36;
        curr_out = out + out_strides[13];
        STHRI_2x128_S(curr_out, v_out_stride, v_out13, v_out14);

        bv_s37 = _mm_sub_ps(bv_s3, bv_s5);
        bv_s38 = _mm_sub_ps(bv_s37, bv_s1);
        bv_s39 = _mm_add_ps(bv_s38, bv_in0);
        bv_s40 = _mm_add_ps(bv_s39, bv_s7);
        // Output point 18: X(17)
        v_out17 = bv_s40;
        curr_out = out + out_strides[17];
        STHR_128_S(curr_out, v_out_stride, v_out17);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        // Standard DFT
        FLOAT av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
              av_in8;
        FLOAT av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
              av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16,
              av_s17, av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24,
              av_s25, av_s26, av_s27, av_s28;
        FLOAT av_m0, av_m1, av_m2, av_m3, av_m4, av_m5, av_m6, av_m7, av_m8,
              av_m9, av_m10, av_m11, av_m12, av_m13, av_m14, av_m15, av_m16, 
              av_m17, av_m18, av_m19, av_m20, av_m21;

        // Input point 1: x(0)
        av_in0 = *in;
        // Input point 3: x(2)
        av_in1 = in[in_strides[2]];
        // Input point 5: x(4)
        av_in2 = in[in_strides[4]];
        // Input point 7: x(6)
        av_in3 = in[in_strides[6]];
        // Input point 9: x(8)
        av_in4 = in[in_strides[8]];
        // Input point 11: x(10)
        av_in5 = in[in_strides[10]];
        // Input point 13: x(12)
        av_in6 = in[in_strides[12]];
        // Input point 15: x(14)
        av_in7 = in[in_strides[14]];
        // Input point 17: x(16)
        av_in8 = in[in_strides[16]];

        av_s0 = av_in1 + av_in8;
        av_s1 = av_in1 - av_in8;
        av_s2 = av_in2 + av_in7;
        av_s3 = av_in2 - av_in7;
        av_s4 = av_in3 + av_in6;
        av_s5 = av_in3 - av_in6;
        av_s6 = av_in4 + av_in5;
        av_s7 = av_in4 - av_in5;

        av_s8 = av_s0 + av_s6;
        av_s9 = av_s8 + av_s2;
        av_s10 = av_s1 - av_s3;
        av_s11 = av_s10 + av_s7;
        av_s12 = av_s9 + av_s4;
        av_s13 = av_s12 + av_in0;

        av_m0 = CRTM_9_4 * av_s4;
        av_m1 = CRTM_9_5 * av_s5;
        av_s14 = av_in0 - av_m0;

        // Output point 1: X(0)
        *out = av_s13;

        av_m2 = CRTM_9_0 * av_s0;
        av_m3 = CRTM_9_2 * av_s2;
        av_m4 = CRTM_9_6 * av_s6;
        av_s15 = av_s14 + av_m2;
        av_s16 = av_s15 + av_m3;
        // Output point 4: X(3)
        out[out_strides[3]] = av_s16 - av_m4;

        av_m5 = CRTM_9_1 * av_s1;
        av_m6 = CRTM_9_3 * av_s3;
        av_m7 = CRTM_9_7 * av_s7;
        av_s17 = av_m5 + av_m6;
        av_s18 = av_s17 + av_m1;
        av_s19 = -(av_s18 + av_m7);
        // Output point 5: X(4)
        out[out_strides[4]] = av_s19;

        av_m8 = CRTM_9_0 * av_s6;
        av_m9 = CRTM_9_2 * av_s0;
        av_m10 = CRTM_9_6 * av_s2;
        av_s20 = av_s14 + av_m8;
        av_s21 = av_s20 + av_m9;
        // Output point 8: X(7)
        out[out_strides[7]] = av_s21 - av_m10;

        av_m11 = CRTM_9_1 * av_s7;
        av_m12 = CRTM_9_3 * av_s1;
        av_m13 = CRTM_9_7 * av_s3;
        av_s22 = av_m11 - av_m12;
        av_s23 = av_s22 + av_m1;
        // Output point 9: X(8)
        out[out_strides[8]] = av_s23 - av_m13;

        av_m14 = CRTM_9_4 * av_s9;
        av_s24 = av_in0 + av_s4;
        // Output point 12: X(11)
        out[out_strides[11]] = av_s24 - av_m14;

        av_m15 = -CRTM_9_5 * av_s11;
        // Output point 13: X(12)
        out[out_strides[12]] = av_m15;

        av_m16 = CRTM_9_0 * av_s2;
        av_m17 = CRTM_9_2 * av_s6;
        av_m18 = CRTM_9_6 * av_s0;
        av_s25 = av_s14 + av_m16;
        av_s26 = av_s25 + av_m17;
        // Output point 16: X(15)
        out[out_strides[15]] = av_s26 - av_m18;

        av_m19 = CRTM_9_1 * av_s3;
        av_m20 = CRTM_9_3 * av_s7;
        av_m21 = CRTM_9_7 * av_s1;
        av_s27 = av_m19 + av_m20;
        av_s28 = av_s27 - av_m1;
        // Output point 17: X(16)
        out[out_strides[16]] = av_s28 - av_m21;

        // Shifted DFT
        FLOAT bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
            bv_in8;
        FLOAT bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
              bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16,
              bv_s17, bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24,
              bv_s25, bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32,
              bv_s33, bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40;
        FLOAT bv_m0, bv_m1, bv_m2, bv_m3, bv_m4, bv_m5, bv_m6, bv_m7, bv_m8,
              bv_m9, bv_m10, bv_m11, bv_m12, bv_m13, bv_m14, bv_m15, bv_m16,
              bv_m17, bv_m18, bv_m19;

        // Input point 2: x(1)
        bv_in0 = in[in_strides[1]];
        // Input point 4: x(3)
        bv_in1 = in[in_strides[3]];
        // Input point 6: x(5)
        bv_in2 = in[in_strides[5]];
        // Input point 8: x(7)
        bv_in3 = in[in_strides[7]];
        // Input point 10: x(9)
        bv_in4 = in[in_strides[9]];
        // Input point 12: x(11)
        bv_in5 = in[in_strides[11]];
        // Input point 14: x(13)
        bv_in6 = in[in_strides[13]];
        // Input point 16: x(15)
        bv_in7 = in[in_strides[15]];
        // Input point 18: x(17)
        bv_in8 = in[in_strides[17]];

        bv_s0 = bv_in1 + bv_in8;
        bv_s1 = bv_in1 - bv_in8;
        bv_s2 = bv_in2 + bv_in7;
        bv_s3 = bv_in2 - bv_in7;
        bv_s4 = bv_in3 + bv_in6;
        bv_s5 = bv_in3 - bv_in6;
        bv_s6 = bv_in4 + bv_in5;
        bv_s7 = bv_in4 - bv_in5;

        bv_m0 = CRTM_9_6 * bv_s1;
        bv_m1 = CRTM_9_6 * bv_s3;
        bv_m2 = CRTM_9_6 * bv_s7;
        bv_m3 = CRTM_9_0 * bv_s3;
        bv_m4 = CRTM_9_0 * bv_s7;
        bv_m5 = CRTM_9_0 * bv_s1;
        bv_m6 = CRTM_9_4 * bv_s5;
        bv_m7 = CRTM_9_2 * bv_s7;
        bv_m8 = CRTM_9_2 * bv_s1;
        bv_m9 = CRTM_9_2 * bv_s3;

        bv_m10 = CRTM_9_7 * bv_s0;
        bv_m11 = CRTM_9_7 * bv_s2;
        bv_m12 = CRTM_9_7 * bv_s6;
        bv_m13 = CRTM_9_1 * bv_s2;
        bv_m14 = CRTM_9_1 * bv_s6;
        bv_m15 = CRTM_9_1 * bv_s0;
        bv_m16 = CRTM_9_3 * bv_s0;
        bv_m17 = CRTM_9_3 * bv_s2;
        bv_m18 = CRTM_9_3 * bv_s6;
        bv_m19 = CRTM_9_5 * bv_s4;

        bv_s8 = bv_m0 + bv_m3;
        bv_s9 = bv_s8 + bv_m6;
        bv_s10 = bv_s9 + bv_m7;
        bv_s11 = bv_s10 + bv_in0;
        bv_s12 = bv_m10 + bv_m13;
        bv_s13 = bv_s12 + bv_m19;
        bv_s14 = -(bv_s13 + bv_m18);
        // Output point 2: X(1)
        out[out_strides[1]] = bv_s11;
        // Output point 3: X(2)
        out[out_strides[2]] = bv_s14;

        bv_s15 = bv_s1 - bv_s3;
        bv_s16 = bv_s15 - bv_s7;
        bv_s17 = CRTM_9_4 * bv_s16;
        bv_s18 = bv_s6 - bv_s0;
        bv_s19 = bv_s18 - bv_s2;
        bv_s20 = CRTM_9_5 * bv_s19;
        bv_s21 = bv_s17 + bv_in0;
        bv_s22 = bv_s21 - bv_s5;
        // Output point 6: X(5)
        out[out_strides[5]] = bv_s22;
        // Output point 7: X(6)
        out[out_strides[6]] = bv_s20;

        bv_s23 = bv_m4 - bv_m1;
        bv_s24 = bv_s23 + bv_m6;
        bv_s25 = bv_s24 - bv_m8;
        bv_s26 = bv_s25 + bv_in0;
        bv_s27 = bv_m11 - bv_m14;
        bv_s28 = bv_s27 + bv_m19;
        bv_s29 = bv_s28 - bv_m16;
        // Output point 10: X(9)
        out[out_strides[9]] = bv_s26;
        // Output point 11: X(10)
        out[out_strides[10]] = bv_s29;

        bv_s30 = bv_m6 - bv_m2;
        bv_s31 = bv_s30 - bv_m5;
        bv_s32 = bv_s31 + bv_m9;
        bv_s33 = bv_s32 + bv_in0;
        bv_s34 = bv_m12 - bv_m15;
        bv_s35 = bv_s34 - bv_m19;
        bv_s36 = bv_s35 + bv_m17;
        // Output point 14: X(13)
        out[out_strides[13]] = bv_s33;
        // Output point 15: X(14)
        out[out_strides[14]] = bv_s36;

        bv_s37 = bv_s3 - bv_s5;
        bv_s38 = bv_s37 - bv_s1;
        bv_s39 = bv_s38 + bv_in0;
        bv_s40 = bv_s39 + bv_s7;
        // Output point 18: X(17)
        out[out_strides[17]] = bv_s40;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hcf_rfft9avx256_fp32_bwd(VOID *in_real, VOID *in_imag,
                                       VOID *out_real, VOID *out_imag, INTP n,
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
    FLOAT *curr_in, *curr_out;
    INTP N = n / NUM_SETS_REAL_256_S;
    INTP remaining_sets = n % NUM_SETS_REAL_256_S;

    __m256 v_CRTM_9_0 = _mm256_broadcast_ss(&CRTM_9_0);
    __m256 v_CRTM_9_1 = _mm256_broadcast_ss(&CRTM_9_1);
    __m256 v_CRTM_9_2 = _mm256_broadcast_ss(&CRTM_9_2);
    __m256 v_CRTM_9_3 = _mm256_broadcast_ss(&CRTM_9_3);
    __m256 v_CRTM_9_4 = _mm256_broadcast_ss(&CRTM_9_4);
    __m256 v_CRTM_9_5 = _mm256_broadcast_ss(&CRTM_9_5);
    __m256 v_CRTM_9_6 = _mm256_broadcast_ss(&CRTM_9_6);
    __m256 v_CRTM_9_7 = _mm256_broadcast_ss(&CRTM_9_7);
    __m256 v_CRTM_9_8 = _mm256_broadcast_ss(&CRTM_9_8);
    __m256 v_CRTM_9_9 = _mm256_broadcast_ss(&CRTM_9_9);
    __m256 v_CRTM_9_10 = _mm256_broadcast_ss(&CRTM_9_10);
    __m256 v_CRTM_9_11 = _mm256_broadcast_ss(&CRTM_9_11);

    for (cnt = 0; cnt < N; cnt++)
    {
        // Standard DFT
        __m256 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
            av_in8;
        __m256 av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
               av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16,
               av_s17, av_s18, av_s19, av_s20, av_s21, av_s22;
        __m256 av_m0, av_m1, av_m2, av_m3, av_m4, av_m5, av_m6, av_m7, av_m8,
               av_m9, av_m10, av_m11, av_m12, av_m13, av_m14, av_m15, av_m16,
               av_m17;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17;

        curr_in = in;
        // Input point 1: X(0)
        LDR_256_S(curr_in, v_in_stride, av_in0);
        // Input point 4: X(3) & Input point 5: X(4)
        curr_in = in + in_strides[3];
        LDRI_2x256_S(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: X(7) & Input point 9: X(8)
        curr_in = in + in_strides[7];
        LDRI_2x256_S(curr_in, v_in_stride, av_in3, av_in4);
        // Input point 12: X(11) & Input point 13: X(12)
        curr_in = in + in_strides[11];
        LDRI_2x256_S(curr_in, v_in_stride, av_in5, av_in6);
        // Input point 16: X(15) & Input point 17: X(16)
        curr_in = in + in_strides[15];
        LDRI_2x256_S(curr_in, v_in_stride, av_in7, av_in8);

        av_m0  = _mm256_mul_ps(v_CRTM_9_7, av_in6);
        av_s0  = _mm256_sub_ps(av_in0, av_in5);
        av_m1  = _mm256_mul_ps(v_CRTM_9_6, av_in5);
        av_s1  = _mm256_add_ps(av_in0, av_m1);
        av_s2  = _mm256_sub_ps(av_s0, av_m0);
        av_s3  = _mm256_add_ps(av_s0, av_m0);
        av_s4  = _mm256_add_ps(av_in7, av_in3);
        av_s5  = _mm256_sub_ps(av_in7, av_in3);
        av_m2  = _mm256_mul_ps(v_CRTM_9_5, av_s5);
        av_s6  = _mm256_add_ps(av_in8, av_in4);
        av_m3  = _mm256_mul_ps(v_CRTM_9_5, av_s6);
        av_s7  = _mm256_sub_ps(av_in4, av_in8);
        av_s8  = _mm256_add_ps(av_in1, av_s4);
        av_m4  = _mm256_mul_ps(v_CRTM_9_4, av_s7);
        av_s9  = _mm256_add_ps(av_in2, av_m4);
        av_s10 = _mm256_add_ps(av_m2, av_s9);
        av_s11 = _mm256_sub_ps(av_s9, av_m2);
        av_m5  = _mm256_mul_ps(v_CRTM_9_4, av_s4);
        av_s12 = _mm256_sub_ps(av_in1, av_m5);
        av_s13 = _mm256_sub_ps(av_s12, av_m3);
        av_s14 = _mm256_add_ps(av_s12, av_m3);
        av_m6  = _mm256_mul_ps(v_CRTM_9_6, av_s8);
        // Output point 1: x(0)
        v_out0 = _mm256_add_ps(av_s1, av_m6);
        curr_out = out;
        STR_256_S(curr_out, v_out_stride, v_out0);

        av_s15 = _mm256_sub_ps(av_s1, av_s8);
        av_s16 = _mm256_sub_ps(av_in2, av_s7);
        av_m7  = _mm256_mul_ps(v_CRTM_9_7, av_s16);
        // Output point 7: x(6)
        v_out6 = _mm256_sub_ps(av_s15, av_m7);
        curr_out = out + out_strides[6];
        STR_256_S(curr_out, v_out_stride, v_out6);
        // Output point 13: x(12)
        v_out12 = _mm256_add_ps(av_s15, av_m7);
        curr_out = out + out_strides[12];
        STR_256_S(curr_out, v_out_stride, v_out12);

        av_m8  = _mm256_mul_ps(v_CRTM_9_0, av_s13);
        av_m9  = _mm256_mul_ps(v_CRTM_9_1, av_s10);
        av_s17 = _mm256_sub_ps(av_m8, av_m9);
        av_m10 = _mm256_mul_ps(v_CRTM_9_11, av_s10);
        av_m11 = _mm256_mul_ps(v_CRTM_9_10, av_s13);
        av_s18 = _mm256_add_ps(av_m11, av_m10);
        av_s19 = _mm256_sub_ps(av_s2, av_s17);
        av_m12 = _mm256_mul_ps(v_CRTM_9_6, av_s17);
        // Output point 3: x(2)
        v_out2 = _mm256_add_ps(av_s2, av_m12);
        curr_out = out + out_strides[2];
        STR_256_S(curr_out, v_out_stride, v_out2);
        // Output point 15: x(14)
        v_out14 = _mm256_add_ps(av_s19, av_s18);
        curr_out = out + out_strides[14];
        STR_256_S(curr_out, v_out_stride, v_out14);
        // Output point 9: x(8)
        v_out8 = _mm256_sub_ps(av_s19, av_s18);
        curr_out = out + out_strides[8];
        STR_256_S(curr_out, v_out_stride, v_out8);

        av_m13 = _mm256_mul_ps(v_CRTM_9_8, av_s14);
        av_m14 = _mm256_mul_ps(v_CRTM_9_9, av_s11);
        av_s20 = _mm256_add_ps(av_m13, av_m14);
        av_m15 = _mm256_mul_ps(v_CRTM_9_2, av_s14);
        av_m16 = _mm256_mul_ps(v_CRTM_9_3, av_s11);
        av_s21 = _mm256_sub_ps(av_m15, av_m16);
        av_s22 = _mm256_sub_ps(av_s3, av_s21);
        av_m17 = _mm256_mul_ps(v_CRTM_9_6, av_s21);
        // Output point 5: x(4)
        v_out4 = _mm256_add_ps(av_s3, av_m17);
        curr_out = out + out_strides[4];
        STR_256_S(curr_out, v_out_stride, v_out4);
        // Output point 17: x(16)
        v_out16 = _mm256_add_ps(av_s22, av_s20);
        curr_out = out + out_strides[16];
        STR_256_S(curr_out, v_out_stride, v_out16);
        // Output point 11: x(10)
        v_out10 = _mm256_sub_ps(av_s22, av_s20);
        curr_out = out + out_strides[10];
        STR_256_S(curr_out, v_out_stride, v_out10);

        // Shifted DFT
        __m256 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8;
        __m256 bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
               bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16,
               bv_s17, bv_s18, bv_s19, bv_s20, bv_s21, bv_s22,
               bv_s23;
        __m256 bv_m0, bv_m1, bv_m2, bv_m3, bv_m4, bv_m5, bv_m6, bv_m7, bv_m8,
               bv_m9, bv_m10, bv_m11, bv_m12, bv_m13, bv_m14, bv_m15, bv_m16,
               bv_m17;

        // Input point 2: X(1) & Input point 3: X(2)
        curr_in = in + in_strides[1];
        LDRI_2x256_S(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: X(5) & Input point 7: X(6)
        curr_in = in + in_strides[5];
        LDRI_2x256_S(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: X(9) & Input point 11: X(10)
        curr_in = in + in_strides[9];
        LDRI_2x256_S(curr_in, v_in_stride, bv_in4, bv_in5);
        // Input point 14: X(13) & Input point 15: X(14)
        curr_in = in + in_strides[13];
        LDRI_2x256_S(curr_in, v_in_stride, bv_in6, bv_in7);
        curr_in = in + in_strides[17];
        // Input point 18: X(17)
        LDR_256_S(curr_in, v_in_stride, bv_in8);

        bv_m0  = _mm256_mul_ps(v_CRTM_9_7, bv_in3);
        bv_s0  = _mm256_sub_ps(bv_in2, bv_in8);
        bv_m1  = _mm256_mul_ps(v_CRTM_9_6, bv_in2);
        bv_s1  = _mm256_add_ps(bv_m1, bv_in8);
        bv_s2  = _mm256_sub_ps(bv_s0, bv_m0);
        bv_s3  = _mm256_add_ps(bv_s0, bv_m0);
        bv_s4  = _mm256_add_ps(bv_in0, bv_in4);
        bv_s5  = _mm256_sub_ps(bv_in4, bv_in0);
        bv_m2  = _mm256_mul_ps(v_CRTM_9_5, bv_s5);
        bv_s6  = _mm256_sub_ps(bv_in5, bv_in1);
        bv_s7  = _mm256_add_ps(bv_in1, bv_in5);
        bv_m3  = _mm256_mul_ps(v_CRTM_9_5, bv_s7);
        bv_s8  = _mm256_add_ps(bv_in6, bv_s4);
        bv_m4  = _mm256_mul_ps(v_CRTM_9_4, bv_s6);
        bv_s9  = _mm256_add_ps(bv_m4, bv_in7);
        bv_s10 = _mm256_sub_ps(bv_m2, bv_s9);
        bv_s11 = _mm256_add_ps(bv_m2, bv_s9);
        bv_m5  = _mm256_mul_ps(v_CRTM_9_4, bv_s4);
        bv_s12 = _mm256_sub_ps(bv_m5, bv_in6);
        bv_s13 = _mm256_add_ps(bv_s12, bv_m3);
        bv_s14 = _mm256_sub_ps(bv_s12, bv_m3);
 
        bv_s15 = _mm256_sub_ps(bv_s8, bv_s1);
        bv_s16 = _mm256_sub_ps(bv_s6, bv_in7);
        bv_m6  = _mm256_mul_ps(v_CRTM_9_7, bv_s16);
        // Output point 8: x(7)
        v_out7 = _mm256_add_ps(bv_s15, bv_m6);
        curr_out = out + out_strides[7];
        STR_256_S(curr_out, v_out_stride, v_out7);
        // Output point 14: x(13)
        v_out13 = _mm256_sub_ps(bv_m6, bv_s15);
        curr_out = out + out_strides[13];
        STR_256_S(curr_out, v_out_stride, v_out13);

        bv_m7  = _mm256_mul_ps(v_CRTM_9_9, bv_s10);
        bv_m8  = _mm256_mul_ps(v_CRTM_9_8, bv_s13);
        bv_s17 = _mm256_sub_ps(bv_m7, bv_m8);
        bv_m9  = _mm256_mul_ps(v_CRTM_9_2, bv_s13);
        bv_m10 = _mm256_mul_ps(v_CRTM_9_3, bv_s10);
        bv_s18 = _mm256_add_ps(bv_m9, bv_m10);
        bv_s19 = _mm256_sub_ps(bv_s3, bv_s18);
        // Output point 12: x(11)
        v_out11 = _mm256_add_ps(bv_s19, bv_s17);
        curr_out = out + out_strides[11];
        STR_256_S(curr_out, v_out_stride, v_out11);
        // Output point 18: x(17)
        v_out17 = _mm256_sub_ps(bv_s17, bv_s19);
        curr_out = out + out_strides[17];
        STR_256_S(curr_out, v_out_stride, v_out17);

        bv_m11 = _mm256_mul_ps(v_CRTM_9_10, bv_s14);
        bv_m12 = _mm256_mul_ps(v_CRTM_9_11, bv_s11);
        bv_s20 = _mm256_add_ps(bv_m11, bv_m12);
        bv_m13 = _mm256_mul_ps(v_CRTM_9_0, bv_s14);
        bv_m14 = _mm256_mul_ps(v_CRTM_9_1, bv_s11);
        bv_s21 = _mm256_sub_ps(bv_m13, bv_m14);
        bv_s22 = _mm256_sub_ps(bv_s21, bv_s2);
        // Output point 10: x(9)
        v_out9 = _mm256_add_ps(bv_s22, bv_s20);
        curr_out = out + out_strides[9];
        STR_256_S(curr_out, v_out_stride, v_out9);
        // Output point 16: x(15)
        v_out15 = _mm256_sub_ps(bv_s20, bv_s22);
        curr_out = out + out_strides[15];
        STR_256_S(curr_out, v_out_stride, v_out15);

        bv_m15 = _mm256_mul_ps(v_CRTM_9_6, bv_s8);
        // Output point 2: x(1)
        v_out1 = _mm256_add_ps(bv_m15, bv_s1);
        curr_out = out + out_strides[1];
        STR_256_S(curr_out, v_out_stride, v_out1);

        bv_m16 = _mm256_mul_ps(v_CRTM_9_6, bv_s21);
        // Output point 4: x(3)
        v_out3 = _mm256_add_ps(bv_m16, bv_s2);
        curr_out = out + out_strides[3];
        STR_256_S(curr_out, v_out_stride, v_out3);

        bv_m17 = _mm256_mul_ps(v_CRTM_9_6, bv_s18);
        bv_s23 = NEGATE_256_S(_mm256_add_ps(bv_m17, bv_s3));
        // Output point 6: x(5)
        v_out5 = bv_s23;
        curr_out = out + out_strides[5];
        STR_256_S(curr_out, v_out_stride, v_out5);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_S)
    { 
        // Standard DFT
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
               av_in8;
        __m128 av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
               av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16,
               av_s17, av_s18, av_s19, av_s20, av_s21, av_s22;
        __m128 av_m0, av_m1, av_m2, av_m3, av_m4, av_m5, av_m6, av_m7, av_m8,
               av_m9, av_m10, av_m11, av_m12, av_m13, av_m14, av_m15, av_m16,
               av_m17;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17;
 
        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_9_0  = _mm256_castps256_ps128(v_CRTM_9_0);
        __m128 v128_CRTM_9_1  = _mm256_castps256_ps128(v_CRTM_9_1);
        __m128 v128_CRTM_9_2  = _mm256_castps256_ps128(v_CRTM_9_2);
        __m128 v128_CRTM_9_3  = _mm256_castps256_ps128(v_CRTM_9_3);
        __m128 v128_CRTM_9_4  = _mm256_castps256_ps128(v_CRTM_9_4);
        __m128 v128_CRTM_9_5  = _mm256_castps256_ps128(v_CRTM_9_5);
        __m128 v128_CRTM_9_6  = _mm256_castps256_ps128(v_CRTM_9_6);
        __m128 v128_CRTM_9_7  = _mm256_castps256_ps128(v_CRTM_9_7);
        __m128 v128_CRTM_9_8  = _mm256_castps256_ps128(v_CRTM_9_8);
        __m128 v128_CRTM_9_9  = _mm256_castps256_ps128(v_CRTM_9_9);
        __m128 v128_CRTM_9_10 = _mm256_castps256_ps128(v_CRTM_9_10);
        __m128 v128_CRTM_9_11 = _mm256_castps256_ps128(v_CRTM_9_11);

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

        av_m0  = _mm_mul_ps(v128_CRTM_9_7, av_in6);
        av_s0  = _mm_sub_ps(av_in0, av_in5);
        av_m1  = _mm_mul_ps(v128_CRTM_9_6, av_in5);
        av_s1  = _mm_add_ps(av_in0, av_m1);
        av_s2  = _mm_sub_ps(av_s0, av_m0);
        av_s3  = _mm_add_ps(av_s0, av_m0);
        av_s4  = _mm_add_ps(av_in7, av_in3);
        av_s5  = _mm_sub_ps(av_in7, av_in3);
        av_m2  = _mm_mul_ps(v128_CRTM_9_5, av_s5);
        av_s6  = _mm_add_ps(av_in8, av_in4);
        av_m3  = _mm_mul_ps(v128_CRTM_9_5, av_s6);
        av_s7  = _mm_sub_ps(av_in4, av_in8);
        av_s8  = _mm_add_ps(av_in1, av_s4);
        av_m4  = _mm_mul_ps(v128_CRTM_9_4, av_s7);
        av_s9  = _mm_add_ps(av_in2, av_m4);
        av_s10 = _mm_add_ps(av_m2, av_s9);
        av_s11 = _mm_sub_ps(av_s9, av_m2);
        av_m5  = _mm_mul_ps(v128_CRTM_9_4, av_s4);
        av_s12 = _mm_sub_ps(av_in1, av_m5);
        av_s13 = _mm_sub_ps(av_s12, av_m3);
        av_s14 = _mm_add_ps(av_s12, av_m3);
        av_m6  = _mm_mul_ps(v128_CRTM_9_6, av_s8);
        // Output point 1: x(0)
        v_out0 = _mm_add_ps(av_s1, av_m6);
        STR_128_S(curr_out, v_out_stride, v_out0);

        av_s15 = _mm_sub_ps(av_s1, av_s8);
        av_s16 = _mm_sub_ps(av_in2, av_s7);
        av_m7  = _mm_mul_ps(v128_CRTM_9_7, av_s16);
        // Output point 7: x(6)
        v_out6 = _mm_sub_ps(av_s15, av_m7);
        curr_out = out + out_strides[6];
        STR_128_S(curr_out, v_out_stride, v_out6);
        // Output point 13: x(12)
        v_out12 = _mm_add_ps(av_s15, av_m7);
        curr_out = out + out_strides[12];
        STR_128_S(curr_out, v_out_stride, v_out12);

        av_m8  = _mm_mul_ps(v128_CRTM_9_0, av_s13);
        av_m9  = _mm_mul_ps(v128_CRTM_9_1, av_s10);
        av_s17 = _mm_sub_ps(av_m8, av_m9);
        av_m10 = _mm_mul_ps(v128_CRTM_9_11, av_s10);
        av_m11 = _mm_mul_ps(v128_CRTM_9_10, av_s13);
        av_s18 = _mm_add_ps(av_m11, av_m10);
        av_s19 = _mm_sub_ps(av_s2, av_s17);
        av_m12 = _mm_mul_ps(v128_CRTM_9_6, av_s17);
        // Output point 3: x(2)
        v_out2 = _mm_add_ps(av_s2, av_m12);
        curr_out = out + out_strides[2];
        STR_128_S(curr_out, v_out_stride, v_out2);
        // Output point 15: x(14)
        v_out14 = _mm_add_ps(av_s19, av_s18);
        curr_out = out + out_strides[14];
        STR_128_S(curr_out, v_out_stride, v_out14);
        // Output point 9: x(8)
        v_out8 = _mm_sub_ps(av_s19, av_s18);
        curr_out = out + out_strides[8];
        STR_128_S(curr_out, v_out_stride, v_out8);

        av_m13 = _mm_mul_ps(v128_CRTM_9_8, av_s14);
        av_m14 = _mm_mul_ps(v128_CRTM_9_9, av_s11);
        av_s20 = _mm_add_ps(av_m13, av_m14);
        av_m15 = _mm_mul_ps(v128_CRTM_9_2, av_s14);
        av_m16 = _mm_mul_ps(v128_CRTM_9_3, av_s11);
        av_s21 = _mm_sub_ps(av_m15, av_m16);
        av_s22 = _mm_sub_ps(av_s3, av_s21);
        av_m17 = _mm_mul_ps(v128_CRTM_9_6, av_s21);
        // Output point 5: x(4)
        v_out4 = _mm_add_ps(av_s3, av_m17);
        curr_out = out + out_strides[4];
        STR_128_S(curr_out, v_out_stride, v_out4);
        // Output point 17: x(16)
        v_out16 = _mm_add_ps(av_s22, av_s20);
        curr_out = out + out_strides[16];
        STR_128_S(curr_out, v_out_stride, v_out16);
        // Output point 11: x(10)
        v_out10 = _mm_sub_ps(av_s22, av_s20);
        curr_out = out + out_strides[10];
        STR_128_S(curr_out, v_out_stride, v_out10);

        // Shifted DFT
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8;
        __m128 bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
               bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, 
               bv_s17, bv_s18, bv_s19, bv_s20, bv_s21, bv_s22,
               bv_s23;
        __m128 bv_m0, bv_m1, bv_m2, bv_m3, bv_m4, bv_m5, bv_m6, bv_m7, bv_m8,
               bv_m9, bv_m10, bv_m11, bv_m12, bv_m13, bv_m14, bv_m15, bv_m16,
               bv_m17;

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
        curr_in = in + in_strides[17];
        // Input point 18: X(17)
        LDR_128_S(curr_in, v_in_stride, bv_in8);

        bv_m0  = _mm_mul_ps(v128_CRTM_9_7, bv_in3);
        bv_s0  = _mm_sub_ps(bv_in2, bv_in8);
        bv_s2  = _mm_sub_ps(bv_s0, bv_m0);
        bv_m1  = _mm_mul_ps(v128_CRTM_9_6, bv_in2);
        bv_s1  = _mm_add_ps(bv_m1, bv_in8);
        bv_s3  = _mm_add_ps(bv_s0, bv_m0);
        bv_s4  = _mm_add_ps(bv_in0, bv_in4);
        bv_s5  = _mm_sub_ps(bv_in4, bv_in0);
        bv_m2  = _mm_mul_ps(v128_CRTM_9_5, bv_s5);
        bv_s6  = _mm_sub_ps(bv_in5, bv_in1);
        bv_s7  = _mm_add_ps(bv_in1, bv_in5);
        bv_m3  = _mm_mul_ps(v128_CRTM_9_5, bv_s7);
        bv_s8  = _mm_add_ps(bv_in6, bv_s4);
        bv_m4  = _mm_mul_ps(v128_CRTM_9_4, bv_s6);
        bv_s9  = _mm_add_ps(bv_m4, bv_in7);
        bv_s10 = _mm_sub_ps(bv_m2, bv_s9);
        bv_s11 = _mm_add_ps(bv_m2, bv_s9);
        bv_m5  = _mm_mul_ps(v128_CRTM_9_4, bv_s4);
        bv_s12 = _mm_sub_ps(bv_m5, bv_in6);
        bv_s13 = _mm_add_ps(bv_s12, bv_m3);
        bv_s14 = _mm_sub_ps(bv_s12, bv_m3);
 
        bv_s15 = _mm_sub_ps(bv_s8, bv_s1);
        bv_s16 = _mm_sub_ps(bv_s6, bv_in7);
        bv_m6  = _mm_mul_ps(v128_CRTM_9_7, bv_s16);
        // Output point 8: x(7)
        v_out7 = _mm_add_ps(bv_s15, bv_m6);
        curr_out = out + out_strides[7];
        STR_128_S(curr_out, v_out_stride, v_out7);
        // Output point 14: x(13)
        v_out13 = _mm_sub_ps(bv_m6, bv_s15);
        curr_out = out + out_strides[13];
        STR_128_S(curr_out, v_out_stride, v_out13);

        bv_m7  = _mm_mul_ps(v128_CRTM_9_9, bv_s10);
        bv_m8  = _mm_mul_ps(v128_CRTM_9_8, bv_s13);
        bv_s17 = _mm_sub_ps(bv_m7, bv_m8);
        bv_m9  = _mm_mul_ps(v128_CRTM_9_2, bv_s13);
        bv_m10 = _mm_mul_ps(v128_CRTM_9_3, bv_s10);
        bv_s18 = _mm_add_ps(bv_m9, bv_m10);
        bv_s19 = _mm_sub_ps(bv_s3, bv_s18);
        // Output point 12: x(11)
        v_out11 = _mm_add_ps(bv_s19, bv_s17);
        curr_out = out + out_strides[11];
        STR_128_S(curr_out, v_out_stride, v_out11);
        // Output point 18: x(17)
        v_out17 = _mm_sub_ps(bv_s17, bv_s19);
        curr_out = out + out_strides[17];
        STR_128_S(curr_out, v_out_stride, v_out17);

        bv_m11 = _mm_mul_ps(v128_CRTM_9_10, bv_s14);
        bv_m12 = _mm_mul_ps(v128_CRTM_9_11, bv_s11);
        bv_s20 = _mm_add_ps(bv_m11, bv_m12);
        bv_m13 = _mm_mul_ps(v128_CRTM_9_0, bv_s14);
        bv_m14 = _mm_mul_ps(v128_CRTM_9_1, bv_s11);
        bv_s21 = _mm_sub_ps(bv_m13, bv_m14);
        bv_s22 = _mm_sub_ps(bv_s21, bv_s2);
        // Output point 10: x(9)
        v_out9 = _mm_add_ps(bv_s22, bv_s20);
        curr_out = out + out_strides[9];
        STR_128_S(curr_out, v_out_stride, v_out9);
        // Output point 16: x(15)
        v_out15 = _mm_sub_ps(bv_s20, bv_s22);
        curr_out = out + out_strides[15];
        STR_128_S(curr_out, v_out_stride, v_out15);

        bv_m15 = _mm_mul_ps(v128_CRTM_9_6, bv_s8);
        // Output point 2: x(1)
        v_out1 = _mm_add_ps(bv_m15, bv_s1);
        curr_out = out + out_strides[1];
        STR_128_S(curr_out, v_out_stride, v_out1);

        bv_m16 = _mm_mul_ps(v128_CRTM_9_6, bv_s21);
        // Output point 4: x(3)
        v_out3 = _mm_add_ps(bv_m16, bv_s2);
        curr_out = out + out_strides[3];
        STR_128_S(curr_out, v_out_stride, v_out3);

        bv_m17 = _mm_mul_ps(v128_CRTM_9_6, bv_s18);
        bv_s23 = NEGATE_128_S(_mm_add_ps(bv_m17, bv_s3));
        // Output point 6: x(5)
        v_out5 = bv_s23;
        curr_out = out + out_strides[5];
        STR_128_S(curr_out, v_out_stride, v_out5);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & 2)
     {
        // Standard DFT
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
               av_in8;
        __m128 av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
               av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16,
               av_s17, av_s18, av_s19, av_s20, av_s21, av_s22;
        __m128 av_m0, av_m1, av_m2, av_m3, av_m4, av_m5, av_m6, av_m7, av_m8,
               av_m9, av_m10, av_m11, av_m12, av_m13, av_m14, av_m15, av_m16,
               av_m17;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17;
 
        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_9_0  = _mm256_castps256_ps128(v_CRTM_9_0);
        __m128 v128_CRTM_9_1  = _mm256_castps256_ps128(v_CRTM_9_1);
        __m128 v128_CRTM_9_2  = _mm256_castps256_ps128(v_CRTM_9_2);
        __m128 v128_CRTM_9_3  = _mm256_castps256_ps128(v_CRTM_9_3);
        __m128 v128_CRTM_9_4  = _mm256_castps256_ps128(v_CRTM_9_4);
        __m128 v128_CRTM_9_5  = _mm256_castps256_ps128(v_CRTM_9_5);
        __m128 v128_CRTM_9_6  = _mm256_castps256_ps128(v_CRTM_9_6);
        __m128 v128_CRTM_9_7  = _mm256_castps256_ps128(v_CRTM_9_7);
        __m128 v128_CRTM_9_8 = _mm256_castps256_ps128(v_CRTM_9_8);
        __m128 v128_CRTM_9_9 = _mm256_castps256_ps128(v_CRTM_9_9);
        __m128 v128_CRTM_9_10 = _mm256_castps256_ps128(v_CRTM_9_10);
        __m128 v128_CRTM_9_11 = _mm256_castps256_ps128(v_CRTM_9_11);

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

        av_m0  = _mm_mul_ps(v128_CRTM_9_7, av_in6);
        av_s0  = _mm_sub_ps(av_in0, av_in5);
        av_m1  = _mm_mul_ps(v128_CRTM_9_6, av_in5);
        av_s1  = _mm_add_ps(av_in0, av_m1);
        av_s2  = _mm_sub_ps(av_s0, av_m0);
        av_s3  = _mm_add_ps(av_s0, av_m0);
        av_s4  = _mm_add_ps(av_in7, av_in3);
        av_s5  = _mm_sub_ps(av_in7, av_in3);
        av_m2  = _mm_mul_ps(v128_CRTM_9_5, av_s5);
        av_s6  = _mm_add_ps(av_in8, av_in4);
        av_m3  = _mm_mul_ps(v128_CRTM_9_5, av_s6);
        av_s7  = _mm_sub_ps(av_in4, av_in8);
        av_s8  = _mm_add_ps(av_in1, av_s4);
        av_m4  = _mm_mul_ps(v128_CRTM_9_4, av_s7);
        av_s9  = _mm_add_ps(av_in2, av_m4);
        av_s10 = _mm_add_ps(av_m2, av_s9);
        av_s11 = _mm_sub_ps(av_s9, av_m2);
        av_m5  = _mm_mul_ps(v128_CRTM_9_4, av_s4);
        av_s12 = _mm_sub_ps(av_in1, av_m5);
        av_s13 = _mm_sub_ps(av_s12, av_m3);
        av_s14 = _mm_add_ps(av_s12, av_m3);
        av_m6  = _mm_mul_ps(v128_CRTM_9_6, av_s8);
        // Output point 1: x(0)
        v_out0 = _mm_add_ps(av_s1, av_m6);
        curr_out = out;
        STHR_128_S(curr_out, v_out_stride, v_out0);

        av_s15 = _mm_sub_ps(av_s1, av_s8);
        av_s16 = _mm_sub_ps(av_in2, av_s7);
        av_m7  = _mm_mul_ps(v128_CRTM_9_7, av_s16);
        // Output point 7: x(6)
        v_out6 = _mm_sub_ps(av_s15, av_m7);
        curr_out = out + out_strides[6];
        STHR_128_S(curr_out, v_out_stride, v_out6);
        // Output point 13: x(12)
        v_out12 = _mm_add_ps(av_s15, av_m7);
        curr_out = out + out_strides[12];
        STHR_128_S(curr_out, v_out_stride, v_out12);

        av_m8  = _mm_mul_ps(v128_CRTM_9_0, av_s13);
        av_m9  = _mm_mul_ps(v128_CRTM_9_1, av_s10);
        av_s17 = _mm_sub_ps(av_m8, av_m9);
        av_m10 = _mm_mul_ps(v128_CRTM_9_11, av_s10);
        av_m11 = _mm_mul_ps(v128_CRTM_9_10, av_s13);
        av_s18 = _mm_add_ps(av_m11, av_m10);
        av_s19 = _mm_sub_ps(av_s2, av_s17);
        av_m12 = _mm_mul_ps(v128_CRTM_9_6, av_s17);
        // Output point 3: x(2)
        v_out2 = _mm_add_ps(av_s2, av_m12);
        curr_out = out + out_strides[2];
        STHR_128_S(curr_out, v_out_stride, v_out2);
        // Output point 15: x(14)
        v_out14 = _mm_add_ps(av_s19, av_s18);
        curr_out = out + out_strides[14];
        STHR_128_S(curr_out, v_out_stride, v_out14);
        // Output point 9: x(8)
        v_out8 = _mm_sub_ps(av_s19, av_s18);
        curr_out = out + out_strides[8];
        STHR_128_S(curr_out, v_out_stride, v_out8);

        av_m13 = _mm_mul_ps(v128_CRTM_9_8, av_s14);
        av_m14 = _mm_mul_ps(v128_CRTM_9_9, av_s11);
        av_s20 = _mm_add_ps(av_m13, av_m14);
        av_m15 = _mm_mul_ps(v128_CRTM_9_2, av_s14);
        av_m16 = _mm_mul_ps(v128_CRTM_9_3, av_s11);
        av_s21 = _mm_sub_ps(av_m15, av_m16);
        av_s22 = _mm_sub_ps(av_s3, av_s21);
        av_m17 = _mm_mul_ps(v128_CRTM_9_6, av_s21);
        // Output point 5: x(4)
        v_out4 = _mm_add_ps(av_s3, av_m17);
        curr_out = out + out_strides[4];
        STHR_128_S(curr_out, v_out_stride, v_out4);
        // Output point 17: x(16)
        v_out16 = _mm_add_ps(av_s22, av_s20);
        curr_out = out + out_strides[16];
        STHR_128_S(curr_out, v_out_stride, v_out16);
        // Output point 11: x(10)
        v_out10 = _mm_sub_ps(av_s22, av_s20);
        curr_out = out + out_strides[10];
        STHR_128_S(curr_out, v_out_stride, v_out10);

        // Shifted DFT
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8;
        __m128 bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
               bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, 
               bv_s17, bv_s18, bv_s19, bv_s20, bv_s21, bv_s22,
               bv_s23;
        __m128 bv_m0, bv_m1, bv_m2, bv_m3, bv_m4, bv_m5, bv_m6, bv_m7, bv_m8,
               bv_m9, bv_m10, bv_m11, bv_m12, bv_m13, bv_m14, bv_m15, bv_m16,
               bv_m17;

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
        // Input point 18: X(17)
        curr_in = in + in_strides[17];
        LDHR_128_S(curr_in, v_in_stride, bv_in8);
 
        bv_m0  = _mm_mul_ps(v128_CRTM_9_7, bv_in3);
        bv_s0  = _mm_sub_ps(bv_in2, bv_in8);
        bv_m1  = _mm_mul_ps(v128_CRTM_9_6, bv_in2);
        bv_s1  = _mm_add_ps(bv_m1, bv_in8);
        bv_s2  = _mm_sub_ps(bv_s0, bv_m0);
        bv_s3  = _mm_add_ps(bv_s0, bv_m0);
        bv_s4  = _mm_add_ps(bv_in0, bv_in4);
        bv_s5  = _mm_sub_ps(bv_in4, bv_in0);
        bv_m2  = _mm_mul_ps(v128_CRTM_9_5, bv_s5);
        bv_s6  = _mm_sub_ps(bv_in5, bv_in1);
        bv_s7  = _mm_add_ps(bv_in1, bv_in5);
        bv_m3  = _mm_mul_ps(v128_CRTM_9_5, bv_s7);
        bv_s8  = _mm_add_ps(bv_in6, bv_s4);
        bv_m4  = _mm_mul_ps(v128_CRTM_9_4, bv_s6);
        bv_s9  = _mm_add_ps(bv_m4, bv_in7);
        bv_s10 = _mm_sub_ps(bv_m2, bv_s9);
        bv_s11 = _mm_add_ps(bv_m2, bv_s9);
        bv_m5  = _mm_mul_ps(v128_CRTM_9_4, bv_s4);
        bv_s12 = _mm_sub_ps(bv_m5, bv_in6);
        bv_s13 = _mm_add_ps(bv_s12, bv_m3);
        bv_s14 = _mm_sub_ps(bv_s12, bv_m3);
 
        bv_s15 = _mm_sub_ps(bv_s8, bv_s1);
        bv_s16 = _mm_sub_ps(bv_s6, bv_in7);
        bv_m6  = _mm_mul_ps(v128_CRTM_9_7, bv_s16);
        // Output point 8: x(7)
        v_out7 = _mm_add_ps(bv_s15, bv_m6);
        curr_out = out + out_strides[7];
        STHR_128_S(curr_out, v_out_stride, v_out7);
        // Output point 14: x(13)
        v_out13 = _mm_sub_ps(bv_m6, bv_s15);
        curr_out = out + out_strides[13];
        STHR_128_S(curr_out, v_out_stride, v_out13);

        bv_m7  = _mm_mul_ps(v128_CRTM_9_9, bv_s10);
        bv_m8  = _mm_mul_ps(v128_CRTM_9_8, bv_s13);
        bv_s17 = _mm_sub_ps(bv_m7, bv_m8);
        bv_m9  = _mm_mul_ps(v128_CRTM_9_2, bv_s13);
        bv_m10 = _mm_mul_ps(v128_CRTM_9_3, bv_s10);
        bv_s18 = _mm_add_ps(bv_m9, bv_m10);
        bv_s19 = _mm_sub_ps(bv_s3, bv_s18);
        // Output point 12: x(11)
        v_out11 = _mm_add_ps(bv_s19, bv_s17);
        curr_out = out + out_strides[11];
        STHR_128_S(curr_out, v_out_stride, v_out11);
        // Output point 18: x(17)
        v_out17 = _mm_sub_ps(bv_s17, bv_s19);
        curr_out = out + out_strides[17];
        STHR_128_S(curr_out, v_out_stride, v_out17);

        bv_m11 = _mm_mul_ps(v128_CRTM_9_10, bv_s14);
        bv_m12 = _mm_mul_ps(v128_CRTM_9_11, bv_s11);
        bv_s20 = _mm_add_ps(bv_m11, bv_m12);
        bv_m13 = _mm_mul_ps(v128_CRTM_9_0, bv_s14);
        bv_m14 = _mm_mul_ps(v128_CRTM_9_1, bv_s11);
        bv_s21 = _mm_sub_ps(bv_m13, bv_m14);
        bv_s22 = _mm_sub_ps(bv_s21, bv_s2);
        // Output point 10: x(9)
        v_out9 = _mm_add_ps(bv_s22, bv_s20);
        curr_out = out + out_strides[9];
        STHR_128_S(curr_out, v_out_stride, v_out9);
        // Output point 16: x(15)
        v_out15 = _mm_sub_ps(bv_s20, bv_s22);
        curr_out = out + out_strides[15];
        STHR_128_S(curr_out, v_out_stride, v_out15);

        bv_m15 = _mm_mul_ps(v128_CRTM_9_6, bv_s8);
        // Output point 2: x(1)
        v_out1 = _mm_add_ps(bv_m15, bv_s1);
        curr_out = out + out_strides[1];
        STHR_128_S(curr_out, v_out_stride, v_out1);

        bv_m16 = _mm_mul_ps(v128_CRTM_9_6, bv_s21);
        // Output point 4: x(3)
        v_out3 = _mm_add_ps(bv_m16, bv_s2);
        curr_out = out + out_strides[3];
        STHR_128_S(curr_out, v_out_stride, v_out3);

        bv_m17 = _mm_mul_ps(v128_CRTM_9_6, bv_s18);
        bv_s23 = NEGATE_128_S(_mm_add_ps(bv_m17, bv_s3));
        // Output point 6: x(5)
        v_out5 = bv_s23;
        curr_out = out + out_strides[5];
        STHR_128_S(curr_out, v_out_stride, v_out5);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
     // tail cases
     if (remaining_sets & 1)
     {
        // Standard DFT
        FLOAT av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
              av_in8;
        FLOAT av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
              av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16, 
              av_s17, av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24, 
              av_s25, av_s26, av_s27, av_s28, av_s29, av_s30, av_s31;
        FLOAT av_m0, av_m1, av_m2, av_m3, av_m4, av_m5, av_m6, av_m7, av_m8,
              av_m9, av_m10, av_m11, av_m12, av_m13, av_m14, av_m15, av_m16,
              av_m17;
 
        // Input point 1: X(0)
        av_in0 = *in;
        // Input point 4: X(3)
        av_in1 = in[in_strides[3]];
        // Input point 5: X(4)
        av_in2 = in[in_strides[4]];
        // Input point 8: X(7)
        av_in3 = in[in_strides[7]];
        // Input point 9: X(8)
        av_in4 = in[in_strides[8]];
        // Input point 12: X(11)
        av_in5 = in[in_strides[11]];
        // Input point 13: X(12)
        av_in6 = in[in_strides[12]];
        // Input point 16: X(15)
        av_in7 = in[in_strides[15]];
        // Input point 17: X(16)
        av_in8 = in[in_strides[16]];

        av_m0  = CRTM_9_7 * av_in6;
        av_s0  = av_in0 - av_in5;
        av_m1  = CRTM_9_6 * av_in5;
        av_s1  = av_in0 + av_m1;
        av_s2  = av_s0 - av_m0;
        av_s3  = av_s0 + av_m0;
        av_s4  = av_in7 + av_in3;
        av_s5  = av_in7 - av_in3;
        av_m2  = CRTM_9_5 * av_s5;
        av_s6  = av_in8 + av_in4;
        av_m3  = CRTM_9_5 * av_s6;
        av_s7  = av_in4 - av_in8;
        av_s8  = av_in1 + av_s4;
        av_m4  = CRTM_9_4 * av_s7;
        av_s9  = av_in2 + av_m4;
        av_s10 = av_m2 + av_s9;
        av_s11 = av_s9 - av_m2;
        av_m5  = CRTM_9_4 * av_s4;
        av_s12 = av_in1 - av_m5;
        av_s13 = av_s12 - av_m3;
        av_s14 = av_s12 + av_m3;
        av_m6  = CRTM_9_6 * av_s8;
        av_s15 = av_s1 + av_m6;
        // Output point 1: x(0)
        *out = av_s15;

        av_s16 = av_s1 - av_s8;
        av_s17 = av_in2 - av_s7;
        av_m7  = CRTM_9_7 * av_s17;
        av_s18 = av_s16 - av_m7;
        av_s19 = av_s16 + av_m7;
        // Output point 7: x(6)
        out[out_strides[6]] = av_s18;
        // Output point 13: x(12)
        out[out_strides[12]] = av_s19;

        av_m8  = CRTM_9_0 * av_s13;
        av_m9  = CRTM_9_1 * av_s10;
        av_s20 = av_m8 - av_m9;
        av_m10 = CRTM_9_11 * av_s10;
        av_m11 = CRTM_9_10 * av_s13;
        av_s21 = av_m11 + av_m10;
        av_s22 = av_s2 - av_s20;
        av_m12 = CRTM_9_6 * av_s20;
        av_s23 = av_s2 + av_m12;
        // Output point 3: x(2)
        out[out_strides[2]] = av_s23;
        av_s24 = av_s22 + av_s21;
        av_s25 = av_s22 - av_s21;
        // Output point 15: x(14)
        out[out_strides[14]] = av_s24;
        // Output point 9: x(8)
        out[out_strides[8]] = av_s25;

        av_m13 = CRTM_9_8 * av_s14;
        av_m14 = CRTM_9_9 * av_s11;
        av_s26 = av_m13 + av_m14;
        av_m15 = CRTM_9_2 * av_s14;
        av_m16 = CRTM_9_3 * av_s11;
        av_s27 = av_m15 - av_m16;
        av_s28 = av_s3 - av_s27;
        av_m17 = CRTM_9_6 * av_s27;
        av_s29 = av_s3 + av_m17;
        // Output point 5: x(4)
        out[out_strides[4]] = av_s29;
        av_s30 = av_s28 + av_s26;
        av_s31 = av_s28 - av_s26;
        // Output point 17: x(16)
        out[out_strides[16]] = av_s30;
        // Output point 11: x(10)
        out[out_strides[10]] = av_s31;

        // Shifted DFT
        FLOAT bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
              bv_in8;
        FLOAT bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
              bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, 
              bv_s17, bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24, 
              bv_s25, bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31;
        FLOAT bv_m0, bv_m1, bv_m2, bv_m3, bv_m4, bv_m5, bv_m6, bv_m7, bv_m8,
              bv_m9, bv_m10, bv_m11, bv_m12, bv_m13, bv_m14, bv_m15, bv_m16,
              bv_m17;

        // Input point 2: X(1)
        bv_in0 = in[in_strides[1]];
        // Input point 3: X(2)
        bv_in1 = in[in_strides[2]];
        // Input point 6: X(5)
        bv_in2 = in[in_strides[5]];
        // Input point 7: X(6)
        bv_in3 = in[in_strides[6]];
        // Input point 10: X(9)
        bv_in4 = in[in_strides[9]];
        // Input point 11: X(10)
        bv_in5 = in[in_strides[10]];
        // Input point 14: X(13)
        bv_in6 = in[in_strides[13]];
        // Input point 15: X(14)
        bv_in7 = in[in_strides[14]];
        // Input point 18: X(17)
        bv_in8 = in[in_strides[17]];
 
        bv_m0  = CRTM_9_7 * bv_in3;
        bv_s0  = bv_in2 - bv_in8;
        bv_m1  = CRTM_9_6 * bv_in2;
        bv_s1  = bv_m1 + bv_in8;
        bv_s2  = bv_s0 - bv_m0;
        bv_s3  = bv_s0 + bv_m0;
        bv_s4  = bv_in0 + bv_in4;
        bv_s5  = bv_in4 - bv_in0;
        bv_m2  = CRTM_9_5 * bv_s5;
        bv_s6  = bv_in5 - bv_in1;
        bv_s7  = bv_in1 + bv_in5;
        bv_m3  = CRTM_9_5 * bv_s7;
        bv_s8  = bv_in6 + bv_s4;
        bv_m4  = CRTM_9_4 * bv_s6;
        bv_s9  = bv_m4 + bv_in7;
        bv_s10 = bv_m2 - bv_s9;
        bv_s11 = bv_m2 + bv_s9;
        bv_m5  = CRTM_9_4 * bv_s4;
        bv_s12 = bv_m5 - bv_in6;
        bv_s13 = bv_s12 + bv_m3;
        bv_s14 = bv_s12 - bv_m3;
 
        bv_s15 = bv_s8 - bv_s1;
        bv_s16 = bv_s6 - bv_in7;
        bv_m6  = CRTM_9_7 * bv_s16;
        bv_m7  = CRTM_9_9 * bv_s10;
        bv_m8  = CRTM_9_8 * bv_s13;
        bv_s17 = bv_m7 - bv_m8;
        bv_m9  = CRTM_9_2 * bv_s13;
        bv_m10 = CRTM_9_3 * bv_s10;
        bv_s18 = bv_m9 + bv_m10;
        bv_s19 = bv_s3 - bv_s18;
        bv_m11 = CRTM_9_10 * bv_s14;
        bv_m12 = CRTM_9_11 * bv_s11;
        bv_s20 = bv_m11 + bv_m12;
        bv_m13 = CRTM_9_0 * bv_s14;
        bv_m14 = CRTM_9_1 * bv_s11;
        bv_s21 = bv_m13 - bv_m14;
        bv_s22 = bv_s21 - bv_s2;
 
        bv_m15 = CRTM_9_6 * bv_s8;
        bv_s23 = bv_m15 + bv_s1;
        // Output point 2: x(1)
        out[out_strides[1]] = bv_s23;

        bv_m16 = CRTM_9_6 * bv_s21;
        bv_s24 = bv_m16 + bv_s2;
        // Output point 4: x(3)
        out[out_strides[3]] = bv_s24;

        bv_m17 = CRTM_9_6 * bv_s18;
        bv_s25 = -(bv_m17 + bv_s3);
        // Output point 6: x(5)
        out[out_strides[5]] = bv_s25;

        bv_s26 = bv_s15 + bv_m6;
        // Output point 8: x(7)
        out[out_strides[7]] = bv_s26;

        bv_s27 = bv_s22 + bv_s20;
        // Output point 10: x(9)
        out[out_strides[9]] = bv_s27;

        bv_s28 = bv_s19 + bv_s17;
        // Output point 12: x(11)
        out[out_strides[11]] = bv_s28;

        bv_s29 = bv_m6 - bv_s15;
        // Output point 14: x(13)
        out[out_strides[13]] = bv_s29;

        bv_s30 = bv_s20 - bv_s22;
        // Output point 16: x(15)
        out[out_strides[15]] = bv_s30;

        bv_s31 = bv_s17 - bv_s19;
        // Output point 18: x(17)
        out[out_strides[17]] = bv_s31;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hcf_rfft9avx256_fp64_fwd(VOID *in_real, VOID *in_imag,
                                       VOID *out_real, VOID *out_imag, INTP n,
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
    DOUBLE *curr_in, *curr_out;
    INTP N = n / NUM_SETS_REAL_256_D;
    INTP remaining_sets = n % NUM_SETS_REAL_256_D;

    __m256d v_CRTM_9_0 = _mm256_broadcast_sd(&CRTM_9_0);
    __m256d v_CRTM_9_1 = _mm256_broadcast_sd(&CRTM_9_1);
    __m256d v_CRTM_9_2 = _mm256_broadcast_sd(&CRTM_9_2);
    __m256d v_CRTM_9_3 = _mm256_broadcast_sd(&CRTM_9_3);
    __m256d v_CRTM_9_4 = _mm256_broadcast_sd(&CRTM_9_4);
    __m256d v_CRTM_9_5 = _mm256_broadcast_sd(&CRTM_9_5);
    __m256d v_CRTM_9_6 = _mm256_broadcast_sd(&CRTM_9_6);
    __m256d v_CRTM_9_7 = _mm256_broadcast_sd(&CRTM_9_7);

    for (cnt = 0; cnt < N; cnt++)
    {
        // Standard DFT
        __m256d av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
                av_in8;
        __m256d av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
                av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16,
                av_s17, av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24,
                av_s25, av_s26, av_s27, av_s28;
        __m256d av_m0, av_m1, av_m2, av_m3, av_m4, av_m5, av_m6, av_m7, av_m8,
                av_m9, av_m10, av_m11, av_m12, av_m13, av_m14, av_m15, av_m16,
                av_m17, av_m18, av_m19, av_m20, av_m21;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
                v_out15, v_out16, v_out17;

        curr_in = in;
        curr_out = out;

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

        av_s0 = _mm256_add_pd(av_in1, av_in8);
        av_s1 = _mm256_sub_pd(av_in1, av_in8);
        av_s2 = _mm256_add_pd(av_in2, av_in7);
        av_s3 = _mm256_sub_pd(av_in2, av_in7);
        av_s4 = _mm256_add_pd(av_in3, av_in6);
        av_s5 = _mm256_sub_pd(av_in3, av_in6);
        av_s6 = _mm256_add_pd(av_in4, av_in5);
        av_s7 = _mm256_sub_pd(av_in4, av_in5);

        av_s8 = _mm256_add_pd(av_s0, av_s6);
        av_s9 = _mm256_add_pd(av_s8, av_s2);
        av_s10 = _mm256_sub_pd(av_s1, av_s3);
        av_s11 = _mm256_add_pd(av_s10, av_s7);
        av_s12 = _mm256_add_pd(av_s9, av_s4);
        av_s13 = _mm256_add_pd(av_s12, av_in0);
        av_m0 = _mm256_mul_pd(v_CRTM_9_4, av_s4);
        av_m1 = _mm256_mul_pd(v_CRTM_9_5, av_s5);
        av_s14 = _mm256_sub_pd(av_in0, av_m0);

        av_m2 = _mm256_mul_pd(v_CRTM_9_0, av_s0);
        av_m3 = _mm256_mul_pd(v_CRTM_9_2, av_s2);
        av_m4 = _mm256_mul_pd(v_CRTM_9_6, av_s6);
        av_m5 = _mm256_mul_pd(v_CRTM_9_1, av_s1);
        av_m6 = _mm256_mul_pd(v_CRTM_9_3, av_s3);
        av_m7 = _mm256_mul_pd(v_CRTM_9_7, av_s7);
        av_m8 = _mm256_mul_pd(v_CRTM_9_0, av_s2);
        av_m9 = _mm256_mul_pd(v_CRTM_9_2, av_s0);
        av_m10 = _mm256_mul_pd(v_CRTM_9_6, av_s2);
        av_m11 = _mm256_mul_pd(v_CRTM_9_0, av_s6);
        av_m12 = _mm256_mul_pd(v_CRTM_9_1, av_s7);
        av_m13 = _mm256_mul_pd(v_CRTM_9_3, av_s1);
        av_m14 = _mm256_mul_pd(v_CRTM_9_7, av_s3);

        // Output point 1: X(0)
        v_out0 = av_s13;
        STR_256_D(curr_out, v_out_stride, v_out0);

        av_s15 = _mm256_add_pd(av_s14, av_m2);
        av_s16 = _mm256_sub_pd(av_m3, av_m4);
        // Output point 4: X(3)
        v_out3 = _mm256_add_pd(av_s15, av_s16);

        av_s17 = _mm256_add_pd(av_m5, av_m6);
        av_s18 = _mm256_add_pd(av_m7, av_m1);
        av_s19 = NEGATE_256_D(_mm256_add_pd(av_s17, av_s18));
        // Output point 5: X(4)
        v_out4 = av_s19;
        curr_out = out + out_strides[3];
        STRI_2x256_D(curr_out, v_out_stride, v_out3, v_out4);

        av_s20 = _mm256_add_pd(av_m11, av_m9);
        av_s21 = _mm256_sub_pd(av_s20, av_m10);
        // Output point 8: X(7)
        v_out7 = _mm256_add_pd(av_s21, av_s14);

        av_s22 = _mm256_add_pd(av_m13, av_m14);
        av_s23 = _mm256_sub_pd(av_m1, av_s22);
        // Output point 9: X(8)
        v_out8 = _mm256_add_pd(av_s23, av_m12);
        curr_out = out + out_strides[7];
        STRI_2x256_D(curr_out, v_out_stride, v_out7, v_out8);

        av_m15 = _mm256_mul_pd(v_CRTM_9_4, av_s9);
        av_s24 = _mm256_sub_pd(av_s4, av_m15);
        // Output point 12: X(11)
        v_out11 = _mm256_add_pd(av_in0, av_s24);

        av_m16 = NEGATE_256_D(_mm256_mul_pd(v_CRTM_9_5, av_s11));
        // Output point 13: X(12)
        v_out12 = av_m16;
        curr_out = out + out_strides[11];
        STRI_2x256_D(curr_out, v_out_stride, v_out11, v_out12);

        av_s25 = _mm256_add_pd(av_s14, av_m8);
        av_m17 = _mm256_mul_pd(v_CRTM_9_6, av_s0);
        av_s26 = _mm256_sub_pd(av_s25, av_m17);
        av_m18 = _mm256_mul_pd(v_CRTM_9_2, av_s6);
        // Output point 16: X(15)
        v_out15 = _mm256_add_pd(av_s26, av_m18);

        av_m19 = _mm256_mul_pd(v_CRTM_9_1, av_s3);
        av_m20 = _mm256_mul_pd(v_CRTM_9_3, av_s7);
        av_s27 = _mm256_add_pd(av_m19, av_m20);
        av_s28 = _mm256_sub_pd(av_s27, av_m1);
        av_m21 = _mm256_mul_pd(v_CRTM_9_7, av_s1);
        // Output point 17: X(16)
        v_out16 = _mm256_sub_pd(av_s28, av_m21);
        curr_out = out + out_strides[15];
        STRI_2x256_D(curr_out, v_out_stride, v_out15, v_out16);

        // Shifted DFT
        __m256d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
                bv_in8;
        __m256d bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
                bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16,
                bv_s17, bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24,
                bv_s25, bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32,
                bv_s33, bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40;
        __m256d bv_m0, bv_m1, bv_m2, bv_m3, bv_m4, bv_m5, bv_m6, bv_m7, bv_m8,
                bv_m9, bv_m10, bv_m11, bv_m12, bv_m13, bv_m14, bv_m15, bv_m16,
                bv_m17, bv_m18, bv_m19;

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

        bv_s0 = _mm256_add_pd(bv_in1, bv_in8);
        bv_s1 = _mm256_sub_pd(bv_in1, bv_in8);
        bv_s2 = _mm256_add_pd(bv_in2, bv_in7);
        bv_s3 = _mm256_sub_pd(bv_in2, bv_in7);
        bv_s4 = _mm256_add_pd(bv_in3, bv_in6);
        bv_s5 = _mm256_sub_pd(bv_in3, bv_in6);
        bv_s6 = _mm256_add_pd(bv_in4, bv_in5);
        bv_s7 = _mm256_sub_pd(bv_in4, bv_in5);

        bv_m0 = _mm256_mul_pd(v_CRTM_9_6, bv_s1);
        bv_m1 = _mm256_mul_pd(v_CRTM_9_6, bv_s3);
        bv_m2 = _mm256_mul_pd(v_CRTM_9_6, bv_s7);
        bv_m3 = _mm256_mul_pd(v_CRTM_9_0, bv_s3);
        bv_m4 = _mm256_mul_pd(v_CRTM_9_0, bv_s7);
        bv_m5 = _mm256_mul_pd(v_CRTM_9_0, bv_s1);
        bv_m6 = _mm256_mul_pd(v_CRTM_9_4, bv_s5);
        bv_m7 = _mm256_mul_pd(v_CRTM_9_2, bv_s7);
        bv_m8 = _mm256_mul_pd(v_CRTM_9_2, bv_s1);
        bv_m9 = _mm256_mul_pd(v_CRTM_9_2, bv_s3);

        bv_m10 = _mm256_mul_pd(v_CRTM_9_7, bv_s0);
        bv_m11 = _mm256_mul_pd(v_CRTM_9_7, bv_s2);
        bv_m12 = _mm256_mul_pd(v_CRTM_9_7, bv_s6);
        bv_m13 = _mm256_mul_pd(v_CRTM_9_1, bv_s2);
        bv_m14 = _mm256_mul_pd(v_CRTM_9_1, bv_s6);
        bv_m15 = _mm256_mul_pd(v_CRTM_9_1, bv_s0);
        bv_m16 = _mm256_mul_pd(v_CRTM_9_3, bv_s0);
        bv_m17 = _mm256_mul_pd(v_CRTM_9_3, bv_s2);
        bv_m18 = _mm256_mul_pd(v_CRTM_9_3, bv_s6);
        bv_m19 = _mm256_mul_pd(v_CRTM_9_5, bv_s4);

        bv_s8 = _mm256_add_pd(bv_m0, bv_m3);
        bv_s9 = _mm256_add_pd(bv_s8, bv_m6);
        bv_s10 = _mm256_add_pd(bv_s9, bv_m7);
        bv_s11 = _mm256_add_pd(bv_s10, bv_in0);
        bv_s12 = _mm256_add_pd(bv_m10, bv_m13);
        bv_s13 = _mm256_add_pd(bv_s12, bv_m19);
        bv_s14 = NEGATE_256_D(_mm256_add_pd(bv_s13, bv_m18));
        // Output point 2: X(1)
        v_out1 = bv_s11;
        // Output point 3: X(2)
        v_out2 = bv_s14;
        curr_out = out + out_strides[1];
        STRI_2x256_D(curr_out, v_out_stride, v_out1, v_out2);

        bv_s15 = _mm256_sub_pd(bv_s1, bv_s3);
        bv_s16 = _mm256_sub_pd(bv_s15, bv_s7);
        bv_s17 = _mm256_mul_pd(v_CRTM_9_4, bv_s16);
        bv_s18 = _mm256_sub_pd(bv_s6, bv_s0);
        bv_s19 = _mm256_sub_pd(bv_s18, bv_s2);
        bv_s20 = _mm256_mul_pd(v_CRTM_9_5, bv_s19);
        bv_s21 = _mm256_add_pd(bv_s17, bv_in0);
        bv_s22 = _mm256_sub_pd(bv_s21, bv_s5);
        // Output point 6: X(5)
        v_out5 = bv_s22;
        // Output point 7: X(6)
        v_out6 = bv_s20;
        curr_out = out + out_strides[5];
        STRI_2x256_D(curr_out, v_out_stride, v_out5, v_out6);

        bv_s23 = _mm256_sub_pd(bv_m4, bv_m1);
        bv_s24 = _mm256_add_pd(bv_s23, bv_m6);
        bv_s25 = _mm256_sub_pd(bv_s24, bv_m8);
        bv_s26 = _mm256_add_pd(bv_s25, bv_in0);
        bv_s27 = _mm256_sub_pd(bv_m11, bv_m14);
        bv_s28 = _mm256_add_pd(bv_s27, bv_m19);
        bv_s29 = _mm256_sub_pd(bv_s28, bv_m16);
        // Output point 10: X(9)
        v_out9 = bv_s26;
        // Output point 11: X(10)
        v_out10 = bv_s29;
        curr_out = out + out_strides[9];
        STRI_2x256_D(curr_out, v_out_stride, v_out9, v_out10);

        bv_s30 = _mm256_sub_pd(bv_m6, bv_m2);
        bv_s31 = _mm256_sub_pd(bv_s30, bv_m5);
        bv_s32 = _mm256_add_pd(bv_s31, bv_m9);
        bv_s33 = _mm256_add_pd(bv_s32, bv_in0);
        bv_s34 = _mm256_sub_pd(bv_m12, bv_m15);
        bv_s35 = _mm256_sub_pd(bv_s34, bv_m19);
        bv_s36 = _mm256_add_pd(bv_s35, bv_m17);
        // Output point 14: X(13)
        v_out13 = bv_s33;
        // Output point 15: X(14)
        v_out14 = bv_s36;
        curr_out = out + out_strides[13];
        STRI_2x256_D(curr_out, v_out_stride, v_out13, v_out14);

        bv_s37 = _mm256_sub_pd(bv_s3, bv_s5);
        bv_s38 = _mm256_sub_pd(bv_s37, bv_s1);
        bv_s39 = _mm256_add_pd(bv_s38, bv_in0);
        bv_s40 = _mm256_add_pd(bv_s39, bv_s7);
        // Output point 18: X(17)
        v_out17 = bv_s40;
        curr_out = out + out_strides[17];
        STR_256_D(curr_out, v_out_stride, v_out17);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_D)
    {
        // Standard DFT
        __m128d av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
                av_in8;
        __m128d av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
                av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16,
                av_s17, av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24,
                av_s25, av_s26, av_s27, av_s28;
        __m128d av_m0, av_m1, av_m2, av_m3, av_m4, av_m5, av_m6, av_m7, av_m8,
                av_m9, av_m10,
                av_m11, av_m12, av_m13, av_m14, av_m15, av_m16, av_m17, av_m18,
                av_m19, av_m20, av_m21;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
                v_out15,
                v_out16, v_out17;
 
        curr_in = in;
        curr_out = out;

        __m128d v128_CRTM_9_0 = _mm256_castpd256_pd128(v_CRTM_9_0);
        __m128d v128_CRTM_9_1 = _mm256_castpd256_pd128(v_CRTM_9_1);
        __m128d v128_CRTM_9_2 = _mm256_castpd256_pd128(v_CRTM_9_2);
        __m128d v128_CRTM_9_3 = _mm256_castpd256_pd128(v_CRTM_9_3);
        __m128d v128_CRTM_9_4 = _mm256_castpd256_pd128(v_CRTM_9_4);
        __m128d v128_CRTM_9_5 = _mm256_castpd256_pd128(v_CRTM_9_5);
        __m128d v128_CRTM_9_6 = _mm256_castpd256_pd128(v_CRTM_9_6);
        __m128d v128_CRTM_9_7 = _mm256_castpd256_pd128(v_CRTM_9_7);

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
 
        av_s0 = _mm_add_pd(av_in1, av_in8);
        av_s1 = _mm_sub_pd(av_in1, av_in8);
        av_s2 = _mm_add_pd(av_in2, av_in7);
        av_s3 = _mm_sub_pd(av_in2, av_in7);
        av_s4 = _mm_add_pd(av_in3, av_in6);
        av_s5 = _mm_sub_pd(av_in3, av_in6);
        av_s6 = _mm_add_pd(av_in4, av_in5);
        av_s7 = _mm_sub_pd(av_in4, av_in5);
 
        av_s8 = _mm_add_pd(av_s0, av_s6);
        av_s9 = _mm_add_pd(av_s8, av_s2);
        av_s10 = _mm_sub_pd(av_s1, av_s3);
        av_s11 = _mm_add_pd(av_s10, av_s7);
        av_s12 = _mm_add_pd(av_s9, av_s4);
        av_s13 = _mm_add_pd(av_s12, av_in0);
        av_m0 = _mm_mul_pd(v128_CRTM_9_4, av_s4);
        av_m1 = _mm_mul_pd(v128_CRTM_9_5, av_s5);
        av_s14 = _mm_sub_pd(av_in0, av_m0);
 
        av_m2 = _mm_mul_pd(v128_CRTM_9_0, av_s0);
        av_m3 = _mm_mul_pd(v128_CRTM_9_2, av_s2);
        av_m4 = _mm_mul_pd(v128_CRTM_9_6, av_s6);
        av_m5 = _mm_mul_pd(v128_CRTM_9_1, av_s1);
        av_m6 = _mm_mul_pd(v128_CRTM_9_3, av_s3);
        av_m7 = _mm_mul_pd(v128_CRTM_9_7, av_s7);
        av_m8 = _mm_mul_pd(v128_CRTM_9_0, av_s2);
        av_m9 = _mm_mul_pd(v128_CRTM_9_2, av_s0);
        av_m10 = _mm_mul_pd(v128_CRTM_9_6, av_s2);
        av_m11 = _mm_mul_pd(v128_CRTM_9_0, av_s6);
        av_m12 = _mm_mul_pd(v128_CRTM_9_1, av_s7);
        av_m13 = _mm_mul_pd(v128_CRTM_9_3, av_s1);
        av_m14 = _mm_mul_pd(v128_CRTM_9_7, av_s3);
 
        // Output point 1: X(0)
        v_out0 = av_s13;
        STR_128_D(curr_out, v_out_stride, v_out0);

        av_s15 = _mm_add_pd(av_s14, av_m2);
        av_s16 = _mm_sub_pd(av_m3, av_m4);
        // Output point 4: X(3)
        v_out3 = _mm_add_pd(av_s15, av_s16);

        av_s17 = _mm_add_pd(av_m5, av_m6);
        av_s18 = _mm_add_pd(av_m7, av_m1);
        av_s19 = NEGATE_128_D(_mm_add_pd(av_s17, av_s18));
        // Output point 5: X(4)
        v_out4 = av_s19;
        curr_out = out + out_strides[3];
        STRI_2x128_D(curr_out, v_out_stride, v_out3, v_out4);

        av_s20 = _mm_add_pd(av_m11, av_m9);
        av_s21 = _mm_sub_pd(av_s20, av_m10);
        // Output point 8: X(7)
        v_out7 = _mm_add_pd(av_s21, av_s14);

        av_s22 = _mm_add_pd(av_m13, av_m14);
        av_s23 = _mm_sub_pd(av_m1, av_s22);
        // Output point 9: X(8)
        v_out8 = _mm_add_pd(av_s23, av_m12);
        curr_out = out + out_strides[7];
        STRI_2x128_D(curr_out, v_out_stride, v_out7, v_out8);

        av_m15 = _mm_mul_pd(v128_CRTM_9_4, av_s9);
        av_s24 = _mm_sub_pd(av_s4, av_m15);
        // Output point 12: X(11)
        v_out11 = _mm_add_pd(av_in0, av_s24);

        av_m16 = NEGATE_128_D(_mm_mul_pd(v128_CRTM_9_5, av_s11));
        // Output point 13: X(12)
        v_out12 = av_m16;
        curr_out = out + out_strides[11];
        STRI_2x128_D(curr_out, v_out_stride, v_out11, v_out12);

        av_s25 = _mm_add_pd(av_s14, av_m8);
        av_m17 = _mm_mul_pd(v128_CRTM_9_6, av_s0);
        av_s26 = _mm_sub_pd(av_s25, av_m17);
        av_m18 = _mm_mul_pd(v128_CRTM_9_2, av_s6);
        // Output point 16: X(15)
        v_out15 = _mm_add_pd(av_s26, av_m18);

        av_m19 = _mm_mul_pd(v128_CRTM_9_1, av_s3);
        av_m20 = _mm_mul_pd(v128_CRTM_9_3, av_s7);
        av_s27 = _mm_add_pd(av_m19, av_m20);
        av_s28 = _mm_sub_pd(av_s27, av_m1);
        av_m21 = _mm_mul_pd(v128_CRTM_9_7, av_s1);
        // Output point 17: X(16)
        v_out16 = _mm_sub_pd(av_s28, av_m21);
        curr_out = out + out_strides[15];
        STRI_2x128_D(curr_out, v_out_stride, v_out15, v_out16);

        // Shifted DFT
        __m128d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
                bv_in8;
        __m128d bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
                bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16,
                bv_s17, bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24,
                bv_s25, bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32,
                bv_s33, bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40;
        __m128d bv_m0, bv_m1, bv_m2, bv_m3, bv_m4, bv_m5, bv_m6, bv_m7, bv_m8,
                bv_m9, bv_m10, bv_m11, bv_m12, bv_m13, bv_m14, bv_m15, bv_m16,
                bv_m17, bv_m18, bv_m19;

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

        bv_s0 = _mm_add_pd(bv_in1, bv_in8);
        bv_s1 = _mm_sub_pd(bv_in1, bv_in8);
        bv_s2 = _mm_add_pd(bv_in2, bv_in7);
        bv_s3 = _mm_sub_pd(bv_in2, bv_in7);
        bv_s4 = _mm_add_pd(bv_in3, bv_in6);
        bv_s5 = _mm_sub_pd(bv_in3, bv_in6);
        bv_s6 = _mm_add_pd(bv_in4, bv_in5);
        bv_s7 = _mm_sub_pd(bv_in4, bv_in5);

        bv_m0 = _mm_mul_pd(v128_CRTM_9_6, bv_s1);
        bv_m1 = _mm_mul_pd(v128_CRTM_9_6, bv_s3);
        bv_m2 = _mm_mul_pd(v128_CRTM_9_6, bv_s7);
        bv_m3 = _mm_mul_pd(v128_CRTM_9_0, bv_s3);
        bv_m4 = _mm_mul_pd(v128_CRTM_9_0, bv_s7);
        bv_m5 = _mm_mul_pd(v128_CRTM_9_0, bv_s1);
        bv_m6 = _mm_mul_pd(v128_CRTM_9_4, bv_s5);
        bv_m7 = _mm_mul_pd(v128_CRTM_9_2, bv_s7);
        bv_m8 = _mm_mul_pd(v128_CRTM_9_2, bv_s1);
        bv_m9 = _mm_mul_pd(v128_CRTM_9_2, bv_s3);

        bv_m10 = _mm_mul_pd(v128_CRTM_9_7, bv_s0);
        bv_m11 = _mm_mul_pd(v128_CRTM_9_7, bv_s2);
        bv_m12 = _mm_mul_pd(v128_CRTM_9_7, bv_s6);
        bv_m13 = _mm_mul_pd(v128_CRTM_9_1, bv_s2);
        bv_m14 = _mm_mul_pd(v128_CRTM_9_1, bv_s6);
        bv_m15 = _mm_mul_pd(v128_CRTM_9_1, bv_s0);
        bv_m16 = _mm_mul_pd(v128_CRTM_9_3, bv_s0);
        bv_m17 = _mm_mul_pd(v128_CRTM_9_3, bv_s2);
        bv_m18 = _mm_mul_pd(v128_CRTM_9_3, bv_s6);
        bv_m19 = _mm_mul_pd(v128_CRTM_9_5, bv_s4);

        bv_s8 = _mm_add_pd(bv_m0, bv_m3);
        bv_s9 = _mm_add_pd(bv_s8, bv_m6);
        bv_s10 = _mm_add_pd(bv_s9, bv_m7);
        bv_s11 = _mm_add_pd(bv_s10, bv_in0);
        bv_s12 = _mm_add_pd(bv_m10, bv_m13);
        bv_s13 = _mm_add_pd(bv_s12, bv_m19);
        bv_s14 = NEGATE_128_D(_mm_add_pd(bv_s13, bv_m18));
        // Output point 2: X(1)
        v_out1 = bv_s11;
        // Output point 3: X(2)
        v_out2 = bv_s14;
        curr_out = out + out_strides[1];
        STRI_2x128_D(curr_out, v_out_stride, v_out1, v_out2);

        bv_s15 = _mm_sub_pd(bv_s1, bv_s3);
        bv_s16 = _mm_sub_pd(bv_s15, bv_s7);
        bv_s17 = _mm_mul_pd(v128_CRTM_9_4, bv_s16);
        bv_s18 = _mm_sub_pd(bv_s6, bv_s0);
        bv_s19 = _mm_sub_pd(bv_s18, bv_s2);
        bv_s20 = _mm_mul_pd(v128_CRTM_9_5, bv_s19);
        bv_s21 = _mm_add_pd(bv_s17, bv_in0);
        bv_s22 = _mm_sub_pd(bv_s21, bv_s5);
        // Output point 6: X(5)
        v_out5 = bv_s22;
        // Output point 7: X(6)
        v_out6 = bv_s20;
        curr_out = out + out_strides[5];
        STRI_2x128_D(curr_out, v_out_stride, v_out5, v_out6);

        bv_s23 = _mm_sub_pd(bv_m4, bv_m1);
        bv_s24 = _mm_add_pd(bv_s23, bv_m6);
        bv_s25 = _mm_sub_pd(bv_s24, bv_m8);
        bv_s26 = _mm_add_pd(bv_s25, bv_in0);
        bv_s27 = _mm_sub_pd(bv_m11, bv_m14);
        bv_s28 = _mm_add_pd(bv_s27, bv_m19);
        bv_s29 = _mm_sub_pd(bv_s28, bv_m16);
        // Output point 10: X(9)
        v_out9 = bv_s26;
        // Output point 11: X(10)
        v_out10 = bv_s29;
        curr_out = out + out_strides[9];
        STRI_2x128_D(curr_out, v_out_stride, v_out9, v_out10);

        bv_s30 = _mm_sub_pd(bv_m6, bv_m2);
        bv_s31 = _mm_sub_pd(bv_s30, bv_m5);
        bv_s32 = _mm_add_pd(bv_s31, bv_m9);
        bv_s33 = _mm_add_pd(bv_s32, bv_in0);
        bv_s34 = _mm_sub_pd(bv_m12, bv_m15);
        bv_s35 = _mm_sub_pd(bv_s34, bv_m19);
        bv_s36 = _mm_add_pd(bv_s35, bv_m17);
        // Output point 14: X(13)
        v_out13 = bv_s33;
        // Output point 15: X(14)
        v_out14 = bv_s36;
        curr_out = out + out_strides[13];
        STRI_2x128_D(curr_out, v_out_stride, v_out13, v_out14);

        bv_s37 = _mm_sub_pd(bv_s3, bv_s5);
        bv_s38 = _mm_sub_pd(bv_s37, bv_s1);
        bv_s39 = _mm_add_pd(bv_s38, bv_in0);
        bv_s40 = _mm_add_pd(bv_s39, bv_s7);
        // Output point 18: X(17)
        v_out17 = bv_s40;
        curr_out = out + out_strides[17];
        STR_128_D(curr_out, v_out_stride, v_out17);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        // Standard DFT
        DOUBLE av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
               av_in8;
        DOUBLE av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
               av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16,
               av_s17, av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24,
               av_s25, av_s26, av_s27, av_s28;
        DOUBLE av_m0, av_m1, av_m2, av_m3, av_m4, av_m5, av_m6, av_m7, av_m8,
               av_m9, av_m10, av_m11, av_m12, av_m13, av_m14, av_m15, av_m16, 
               av_m17, av_m18, av_m19, av_m20, av_m21;

        // Input point 1: x(0)
        av_in0 = *in;
        // Input point 3: x(2)
        av_in1 = in[in_strides[2]];
        // Input point 5: x(4)
        av_in2 = in[in_strides[4]];
        // Input point 7: x(6)
        av_in3 = in[in_strides[6]];
        // Input point 9: x(8)
        av_in4 = in[in_strides[8]];
        // Input point 11: x(10)
        av_in5 = in[in_strides[10]];
        // Input point 13: x(12)
        av_in6 = in[in_strides[12]];
        // Input point 15: x(14)
        av_in7 = in[in_strides[14]];
        // Input point 17: x(16)
        av_in8 = in[in_strides[16]];

        av_s0 = av_in1 + av_in8;
        av_s1 = av_in1 - av_in8;
        av_s2 = av_in2 + av_in7;
        av_s3 = av_in2 - av_in7;
        av_s4 = av_in3 + av_in6;
        av_s5 = av_in3 - av_in6;
        av_s6 = av_in4 + av_in5;
        av_s7 = av_in4 - av_in5;

        av_s8 = av_s0 + av_s6;
        av_s9 = av_s8 + av_s2;
        av_s10 = av_s1 - av_s3;
        av_s11 = av_s10 + av_s7;
        av_s12 = av_s9 + av_s4;
        av_s13 = av_s12 + av_in0;

        av_m0 = CRTM_9_4 * av_s4;
        av_m1 = CRTM_9_5 * av_s5;
        av_s14 = av_in0 - av_m0;

        // Output point 1: X(0)
        *out = av_s13;

        av_m2 = CRTM_9_0 * av_s0;
        av_m3 = CRTM_9_2 * av_s2;
        av_m4 = CRTM_9_6 * av_s6;
        av_s15 = av_s14 + av_m2;
        av_s16 = av_s15 + av_m3;
        // Output point 4: X(3)
        out[out_strides[3]] = av_s16 - av_m4;

        av_m5 = CRTM_9_1 * av_s1;
        av_m6 = CRTM_9_3 * av_s3;
        av_m7 = CRTM_9_7 * av_s7;
        av_s17 = av_m5 + av_m6;
        av_s18 = av_s17 + av_m1;
        av_s19 = -(av_s18 + av_m7);
        // Output point 5: X(4)
        out[out_strides[4]] = av_s19;

        av_m8 = CRTM_9_0 * av_s6;
        av_m9 = CRTM_9_2 * av_s0;
        av_m10 = CRTM_9_6 * av_s2;
        av_s20 = av_s14 + av_m8;
        av_s21 = av_s20 + av_m9;
        // Output point 8: X(7)
        out[out_strides[7]] = av_s21 - av_m10;

        av_m11 = CRTM_9_1 * av_s7;
        av_m12 = CRTM_9_3 * av_s1;
        av_m13 = CRTM_9_7 * av_s3;
        av_s22 = av_m11 - av_m12;
        av_s23 = av_s22 + av_m1;
        // Output point 9: X(8)
        out[out_strides[8]] = av_s23 - av_m13;

        av_m14 = CRTM_9_4 * av_s9;
        av_s24 = av_in0 + av_s4;
        // Output point 12: X(11)
        out[out_strides[11]] = av_s24 - av_m14;

        av_m15 = -CRTM_9_5 * av_s11;
        // Output point 13: X(12)
        out[out_strides[12]] = av_m15;

        av_m16 = CRTM_9_0 * av_s2;
        av_m17 = CRTM_9_2 * av_s6;
        av_m18 = CRTM_9_6 * av_s0;
        av_s25 = av_s14 + av_m16;
        av_s26 = av_s25 + av_m17;
        // Output point 16: X(15)
        out[out_strides[15]] = av_s26 - av_m18;

        av_m19 = CRTM_9_1 * av_s3;
        av_m20 = CRTM_9_3 * av_s7;
        av_m21 = CRTM_9_7 * av_s1;
        av_s27 = av_m19 + av_m20;
        av_s28 = av_s27 - av_m1;
        // Output point 17: X(16)
        out[out_strides[16]] = av_s28 - av_m21;

        // Shifted DFT
        DOUBLE bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8;
        DOUBLE bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
               bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16,
               bv_s17, bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24,
               bv_s25, bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32,
               bv_s33, bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40;
        DOUBLE bv_m0, bv_m1, bv_m2, bv_m3, bv_m4, bv_m5, bv_m6, bv_m7, bv_m8,
               bv_m9, bv_m10, bv_m11, bv_m12, bv_m13, bv_m14, bv_m15, bv_m16,
               bv_m17, bv_m18, bv_m19;

        // Input point 2: x(1)
        bv_in0 = in[in_strides[1]];
        // Input point 4: x(3)
        bv_in1 = in[in_strides[3]];
        // Input point 6: x(5)
        bv_in2 = in[in_strides[5]];
        // Input point 8: x(7)
        bv_in3 = in[in_strides[7]];
        // Input point 10: x(9)
        bv_in4 = in[in_strides[9]];
        // Input point 12: x(11)
        bv_in5 = in[in_strides[11]];
        // Input point 14: x(13)
        bv_in6 = in[in_strides[13]];
        // Input point 16: x(15)
        bv_in7 = in[in_strides[15]];
        // Input point 18: x(17)
        bv_in8 = in[in_strides[17]];

        bv_s0 = bv_in1 + bv_in8;
        bv_s1 = bv_in1 - bv_in8;
        bv_s2 = bv_in2 + bv_in7;
        bv_s3 = bv_in2 - bv_in7;
        bv_s4 = bv_in3 + bv_in6;
        bv_s5 = bv_in3 - bv_in6;
        bv_s6 = bv_in4 + bv_in5;
        bv_s7 = bv_in4 - bv_in5;

        bv_m0 = CRTM_9_6 * bv_s1;
        bv_m1 = CRTM_9_6 * bv_s3;
        bv_m2 = CRTM_9_6 * bv_s7;
        bv_m3 = CRTM_9_0 * bv_s3;
        bv_m4 = CRTM_9_0 * bv_s7;
        bv_m5 = CRTM_9_0 * bv_s1;
        bv_m6 = CRTM_9_4 * bv_s5;
        bv_m7 = CRTM_9_2 * bv_s7;
        bv_m8 = CRTM_9_2 * bv_s1;
        bv_m9 = CRTM_9_2 * bv_s3;

        bv_m10 = CRTM_9_7 * bv_s0;
        bv_m11 = CRTM_9_7 * bv_s2;
        bv_m12 = CRTM_9_7 * bv_s6;
        bv_m13 = CRTM_9_1 * bv_s2;
        bv_m14 = CRTM_9_1 * bv_s6;
        bv_m15 = CRTM_9_1 * bv_s0;
        bv_m16 = CRTM_9_3 * bv_s0;
        bv_m17 = CRTM_9_3 * bv_s2;
        bv_m18 = CRTM_9_3 * bv_s6;
        bv_m19 = CRTM_9_5 * bv_s4;

        bv_s8 = bv_m0 + bv_m3;
        bv_s9 = bv_s8 + bv_m6;
        bv_s10 = bv_s9 + bv_m7;
        bv_s11 = bv_s10 + bv_in0;
        bv_s12 = bv_m10 + bv_m13;
        bv_s13 = bv_s12 + bv_m19;
        bv_s14 = -(bv_s13 + bv_m18);
        // Output point 2: X(1)
        out[out_strides[1]] = bv_s11;
        // Output point 3: X(2)
        out[out_strides[2]] = bv_s14;

        bv_s15 = bv_s1 - bv_s3;
        bv_s16 = bv_s15 - bv_s7;
        bv_s17 = CRTM_9_4 * bv_s16;
        bv_s18 = bv_s6 - bv_s0;
        bv_s19 = bv_s18 - bv_s2;
        bv_s20 = CRTM_9_5 * bv_s19;
        bv_s21 = bv_s17 + bv_in0;
        bv_s22 = bv_s21 - bv_s5;
        // Output point 6: X(5)
        out[out_strides[5]] = bv_s22;
        // Output point 7: X(6)
        out[out_strides[6]] = bv_s20;

        bv_s23 = bv_m4 - bv_m1;
        bv_s24 = bv_s23 + bv_m6;
        bv_s25 = bv_s24 - bv_m8;
        bv_s26 = bv_s25 + bv_in0;
        bv_s27 = bv_m11 - bv_m14;
        bv_s28 = bv_s27 + bv_m19;
        bv_s29 = bv_s28 - bv_m16;
        // Output point 10: X(9)
        out[out_strides[9]] = bv_s26;
        // Output point 11: X(10)
        out[out_strides[10]] = bv_s29;

        bv_s30 = bv_m6 - bv_m2;
        bv_s31 = bv_s30 - bv_m5;
        bv_s32 = bv_s31 + bv_m9;
        bv_s33 = bv_s32 + bv_in0;
        bv_s34 = bv_m12 - bv_m15;
        bv_s35 = bv_s34 - bv_m19;
        bv_s36 = bv_s35 + bv_m17;
        // Output point 14: X(13)
        out[out_strides[13]] = bv_s33;
        // Output point 15: X(14)
        out[out_strides[14]] = bv_s36;

        bv_s37 = bv_s3 - bv_s5;
        bv_s38 = bv_s37 - bv_s1;
        bv_s39 = bv_s38 + bv_in0;
        bv_s40 = bv_s39 + bv_s7;
        // Output point 18: X(17)
        out[out_strides[17]] = bv_s40;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hcf_rfft9avx256_fp64_bwd(VOID *in_real, VOID *in_imag,
                                       VOID *out_real, VOID *out_imag, INTP n,
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
    DOUBLE *curr_in, *curr_out;
    INTP N = n / NUM_SETS_REAL_256_D;
    INTP remaining_sets = n % NUM_SETS_REAL_256_D;

    __m256d v_CRTM_9_0 = _mm256_broadcast_sd(&CRTM_9_0);
    __m256d v_CRTM_9_1 = _mm256_broadcast_sd(&CRTM_9_1);
    __m256d v_CRTM_9_2 = _mm256_broadcast_sd(&CRTM_9_2);
    __m256d v_CRTM_9_3 = _mm256_broadcast_sd(&CRTM_9_3);
    __m256d v_CRTM_9_4 = _mm256_broadcast_sd(&CRTM_9_4);
    __m256d v_CRTM_9_5 = _mm256_broadcast_sd(&CRTM_9_5);
    __m256d v_CRTM_9_6 = _mm256_broadcast_sd(&CRTM_9_6);
    __m256d v_CRTM_9_7 = _mm256_broadcast_sd(&CRTM_9_7);
    __m256d v_CRTM_9_8 = _mm256_broadcast_sd(&CRTM_9_8);
    __m256d v_CRTM_9_9 = _mm256_broadcast_sd(&CRTM_9_9);
    __m256d v_CRTM_9_10 = _mm256_broadcast_sd(&CRTM_9_10);
    __m256d v_CRTM_9_11 = _mm256_broadcast_sd(&CRTM_9_11);

    for (cnt = 0; cnt < N; cnt++)
    {
        // Standard DFT
        __m256d av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
                av_in8;
        __m256d av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
                av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16,
                av_s17, av_s18, av_s19, av_s20, av_s21, av_s22;
        __m256d av_m0, av_m1, av_m2, av_m3, av_m4, av_m5, av_m6, av_m7, av_m8,
                av_m9, av_m10, av_m11, av_m12, av_m13, av_m14, av_m15, av_m16,
                av_m17;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
                v_out15, v_out16, v_out17;

        curr_in = in;
        // Input point 1: X(0)
        LDR_256_D(curr_in, v_in_stride, av_in0);
        // Input point 4: X(3) & Input point 5: X(4)
        curr_in = in + in_strides[3];
        LDRI_2x256_D(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: X(7) & Input point 9: X(8)
        curr_in = in + in_strides[7];
        LDRI_2x256_D(curr_in, v_in_stride, av_in3, av_in4);
        // Input point 12: X(11) & Input point 13: X(12)
        curr_in = in + in_strides[11];
        LDRI_2x256_D(curr_in, v_in_stride, av_in5, av_in6);
        // Input point 16: X(15) & Input point 17: X(16)
        curr_in = in + in_strides[15];
        LDRI_2x256_D(curr_in, v_in_stride, av_in7, av_in8);

        av_m0  = _mm256_mul_pd(v_CRTM_9_7, av_in6);
        av_s0  = _mm256_sub_pd(av_in0, av_in5);
        av_m1  = _mm256_mul_pd(v_CRTM_9_6, av_in5);
        av_s1  = _mm256_add_pd(av_in0, av_m1);
        av_s2  = _mm256_sub_pd(av_s0, av_m0);
        av_s3  = _mm256_add_pd(av_s0, av_m0);
        av_s4  = _mm256_add_pd(av_in7, av_in3);
        av_s5  = _mm256_sub_pd(av_in7, av_in3);
        av_m2  = _mm256_mul_pd(v_CRTM_9_5, av_s5);
        av_s6  = _mm256_add_pd(av_in8, av_in4);
        av_m3  = _mm256_mul_pd(v_CRTM_9_5, av_s6);
        av_s7  = _mm256_sub_pd(av_in4, av_in8);
        av_s8  = _mm256_add_pd(av_in1, av_s4);
        av_m4  = _mm256_mul_pd(v_CRTM_9_4, av_s7);
        av_s9  = _mm256_add_pd(av_in2, av_m4);
        av_s10 = _mm256_add_pd(av_m2, av_s9);
        av_s11 = _mm256_sub_pd(av_s9, av_m2);
        av_m5  = _mm256_mul_pd(v_CRTM_9_4, av_s4);
        av_s12 = _mm256_sub_pd(av_in1, av_m5);
        av_s13 = _mm256_sub_pd(av_s12, av_m3);
        av_s14 = _mm256_add_pd(av_s12, av_m3);
        av_m6  = _mm256_mul_pd(v_CRTM_9_6, av_s8);
        // Output point 1: x(0)
        v_out0 = _mm256_add_pd(av_s1, av_m6);
        curr_out = out;
        STR_256_D(curr_out, v_out_stride, v_out0);

        av_s15 = _mm256_sub_pd(av_s1, av_s8);
        av_s16 = _mm256_sub_pd(av_in2, av_s7);
        av_m7  = _mm256_mul_pd(v_CRTM_9_7, av_s16);
        // Output point 7: x(6)
        v_out6 = _mm256_sub_pd(av_s15, av_m7);
        curr_out = out + out_strides[6];
        STR_256_D(curr_out, v_out_stride, v_out6);
        // Output point 13: x(12)
        v_out12 = _mm256_add_pd(av_s15, av_m7);
        curr_out = out + out_strides[12];
        STR_256_D(curr_out, v_out_stride, v_out12);

        av_m8  = _mm256_mul_pd(v_CRTM_9_0, av_s13);
        av_m9  = _mm256_mul_pd(v_CRTM_9_1, av_s10);
        av_s17 = _mm256_sub_pd(av_m8, av_m9);
        av_m10 = _mm256_mul_pd(v_CRTM_9_11, av_s10);
        av_m11 = _mm256_mul_pd(v_CRTM_9_10, av_s13);
        av_s18 = _mm256_add_pd(av_m11, av_m10);
        av_s19 = _mm256_sub_pd(av_s2, av_s17);
        av_m12 = _mm256_mul_pd(v_CRTM_9_6, av_s17);
        // Output point 3: x(2)
        v_out2 = _mm256_add_pd(av_s2, av_m12);
        curr_out = out + out_strides[2];
        STR_256_D(curr_out, v_out_stride, v_out2);
        // Output point 15: x(14)
        v_out14 = _mm256_add_pd(av_s19, av_s18);
        curr_out = out + out_strides[14];
        STR_256_D(curr_out, v_out_stride, v_out14);
        // Output point 9: x(8)
        v_out8 = _mm256_sub_pd(av_s19, av_s18);
        curr_out = out + out_strides[8];
        STR_256_D(curr_out, v_out_stride, v_out8);

        av_m13 = _mm256_mul_pd(v_CRTM_9_8, av_s14);
        av_m14 = _mm256_mul_pd(v_CRTM_9_9, av_s11);
        av_s20 = _mm256_add_pd(av_m13, av_m14);
        av_m15 = _mm256_mul_pd(v_CRTM_9_2, av_s14);
        av_m16 = _mm256_mul_pd(v_CRTM_9_3, av_s11);
        av_s21 = _mm256_sub_pd(av_m15, av_m16);
        av_s22 = _mm256_sub_pd(av_s3, av_s21);
        av_m17 = _mm256_mul_pd(v_CRTM_9_6, av_s21);
        // Output point 5: x(4)
        v_out4 = _mm256_add_pd(av_s3, av_m17);
        curr_out = out + out_strides[4];
        STR_256_D(curr_out, v_out_stride, v_out4);
        // Output point 17: x(16)
        v_out16 = _mm256_add_pd(av_s22, av_s20);
        curr_out = out + out_strides[16];
        STR_256_D(curr_out, v_out_stride, v_out16);
        // Output point 11: x(10)
        v_out10 = _mm256_sub_pd(av_s22, av_s20);
        curr_out = out + out_strides[10];
        STR_256_D(curr_out, v_out_stride, v_out10);

        // Shifted DFT
        __m256d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
                bv_in8;
        __m256d bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
                bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16,
                bv_s17, bv_s18, bv_s19, bv_s20, bv_s21, bv_s22,
               bv_s23;
        __m256d bv_m0, bv_m1, bv_m2, bv_m3, bv_m4, bv_m5, bv_m6, bv_m7, bv_m8,
                bv_m9, bv_m10, bv_m11, bv_m12, bv_m13, bv_m14, bv_m15, bv_m16,
                bv_m17;

        // Input point 2: X(1) & Input point 3: X(2)
        curr_in = in + in_strides[1];
        LDRI_2x256_D(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: X(5) & Input point 7: X(6)
        curr_in = in + in_strides[5];
        LDRI_2x256_D(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: X(9) & Input point 11: X(10)
        curr_in = in + in_strides[9];
        LDRI_2x256_D(curr_in, v_in_stride, bv_in4, bv_in5);
        // Input point 14: X(13) & Input point 15: X(14)
        curr_in = in + in_strides[13];
        LDRI_2x256_D(curr_in, v_in_stride, bv_in6, bv_in7);
        curr_in = in + in_strides[17];
        // Input point 18: X(17)
        LDR_256_D(curr_in, v_in_stride, bv_in8);

        bv_m0 = _mm256_mul_pd(v_CRTM_9_7, bv_in3);
        bv_s0 = _mm256_sub_pd(bv_in2, bv_in8);
        bv_m1 = _mm256_mul_pd(v_CRTM_9_6, bv_in2);
        bv_s1 = _mm256_add_pd(bv_m1, bv_in8);
        bv_s2 = _mm256_sub_pd(bv_s0, bv_m0);
        bv_s3 = _mm256_add_pd(bv_s0, bv_m0);
        bv_s4 = _mm256_add_pd(bv_in0, bv_in4);
        bv_s5 = _mm256_sub_pd(bv_in4, bv_in0);
        bv_m2 = _mm256_mul_pd(v_CRTM_9_5, bv_s5);
        bv_s6 = _mm256_sub_pd(bv_in5, bv_in1);
        bv_s7 = _mm256_add_pd(bv_in1, bv_in5);
        bv_m3 = _mm256_mul_pd(v_CRTM_9_5, bv_s7);
        bv_s8 = _mm256_add_pd(bv_in6, bv_s4);
        bv_m4 = _mm256_mul_pd(v_CRTM_9_4, bv_s6);
        bv_s9 = _mm256_add_pd(bv_m4, bv_in7);
        bv_s10 = _mm256_sub_pd(bv_m2, bv_s9);
        bv_s11 = _mm256_add_pd(bv_m2, bv_s9);
        bv_m5 = _mm256_mul_pd(v_CRTM_9_4, bv_s4);
        bv_s12 = _mm256_sub_pd(bv_m5, bv_in6);
        bv_s13 = _mm256_add_pd(bv_s12, bv_m3);
        bv_s14 = _mm256_sub_pd(bv_s12, bv_m3);

        bv_s15 = _mm256_sub_pd(bv_s8, bv_s1);
        bv_s16 = _mm256_sub_pd(bv_s6, bv_in7);
        bv_m6  = _mm256_mul_pd(v_CRTM_9_7, bv_s16);
        // Output point 8: x(7)
        v_out7 = _mm256_add_pd(bv_s15, bv_m6);
        curr_out = out + out_strides[7];
        STR_256_D(curr_out, v_out_stride, v_out7);
        // Output point 14: x(13)
        v_out13 = _mm256_sub_pd(bv_m6, bv_s15);
        curr_out = out + out_strides[13];
        STR_256_D(curr_out, v_out_stride, v_out13);

        bv_m7  = _mm256_mul_pd(v_CRTM_9_9, bv_s10);
        bv_m8  = _mm256_mul_pd(v_CRTM_9_8, bv_s13);
        bv_s17 = _mm256_sub_pd(bv_m7, bv_m8);
        bv_m9  = _mm256_mul_pd(v_CRTM_9_2, bv_s13);
        bv_m10 = _mm256_mul_pd(v_CRTM_9_3, bv_s10);
        bv_s18 = _mm256_add_pd(bv_m9, bv_m10);
        bv_s19 = _mm256_sub_pd(bv_s3, bv_s18);
        // Output point 12: x(11)
        v_out11 = _mm256_add_pd(bv_s19, bv_s17);
        curr_out = out + out_strides[11];
        STR_256_D(curr_out, v_out_stride, v_out11);
        // Output point 18: x(17)
        v_out17 = _mm256_sub_pd(bv_s17, bv_s19);
        curr_out = out + out_strides[17];
        STR_256_D(curr_out, v_out_stride, v_out17);

        bv_m11 = _mm256_mul_pd(v_CRTM_9_10, bv_s14);
        bv_m12 = _mm256_mul_pd(v_CRTM_9_11, bv_s11);
        bv_s20 = _mm256_add_pd(bv_m11, bv_m12);
        bv_m13 = _mm256_mul_pd(v_CRTM_9_0, bv_s14);
        bv_m14 = _mm256_mul_pd(v_CRTM_9_1, bv_s11);
        bv_s21 = _mm256_sub_pd(bv_m13, bv_m14);
        bv_s22 = _mm256_sub_pd(bv_s21, bv_s2);
        // Output point 10: x(9)
        v_out9 = _mm256_add_pd(bv_s22, bv_s20);
        curr_out = out + out_strides[9];
        STR_256_D(curr_out, v_out_stride, v_out9);
        // Output point 16: x(15)
        v_out15 = _mm256_sub_pd(bv_s20, bv_s22);
        curr_out = out + out_strides[15];
        STR_256_D(curr_out, v_out_stride, v_out15);

        bv_m15 = _mm256_mul_pd(v_CRTM_9_6, bv_s8);
        // Output point 2: x(1)
        v_out1 = _mm256_add_pd(bv_m15, bv_s1);
        curr_out = out + out_strides[1];
        STR_256_D(curr_out, v_out_stride, v_out1);

        bv_m16 = _mm256_mul_pd(v_CRTM_9_6, bv_s21);
        // Output point 4: x(3)
        v_out3 = _mm256_add_pd(bv_m16, bv_s2);
        curr_out = out + out_strides[3];
        STR_256_D(curr_out, v_out_stride, v_out3);

        bv_m17 = _mm256_mul_pd(v_CRTM_9_6, bv_s18);
        bv_s23 = NEGATE_256_D(_mm256_add_pd(bv_m17, bv_s3));
        // Output point 6: x(5)
        v_out5 = bv_s23;
        curr_out = out + out_strides[5];
        STR_256_D(curr_out, v_out_stride, v_out5);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_D)
    {
        // Standard DFT
        __m128d av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
                av_in8;
        __m128d av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
                av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16,
                av_s17, av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24,
                av_s25, av_s26, av_s27, av_s28, av_s29, av_s30, av_s31;
        __m128d av_m0, av_m1, av_m2, av_m3, av_m4, av_m5, av_m6, av_m7, av_m8,
                av_m9, av_m10, av_m11, av_m12, av_m13, av_m14, av_m15, av_m16,
                av_m17;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
                v_out15, v_out16, v_out17;
 
        curr_in = in;
        curr_out = out;

        __m128d v128_CRTM_9_0 = _mm256_castpd256_pd128(v_CRTM_9_0);
        __m128d v128_CRTM_9_1 = _mm256_castpd256_pd128(v_CRTM_9_1);
        __m128d v128_CRTM_9_2 = _mm256_castpd256_pd128(v_CRTM_9_2);
        __m128d v128_CRTM_9_3 = _mm256_castpd256_pd128(v_CRTM_9_3);
        __m128d v128_CRTM_9_4 = _mm256_castpd256_pd128(v_CRTM_9_4);
        __m128d v128_CRTM_9_5 = _mm256_castpd256_pd128(v_CRTM_9_5);
        __m128d v128_CRTM_9_6 = _mm256_castpd256_pd128(v_CRTM_9_6);
        __m128d v128_CRTM_9_7 = _mm256_castpd256_pd128(v_CRTM_9_7);
        __m128d v128_CRTM_9_8 = _mm256_castpd256_pd128(v_CRTM_9_8);
        __m128d v128_CRTM_9_9 = _mm256_castpd256_pd128(v_CRTM_9_9);
        __m128d v128_CRTM_9_10 = _mm256_castpd256_pd128(v_CRTM_9_10);
        __m128d v128_CRTM_9_11 = _mm256_castpd256_pd128(v_CRTM_9_11);

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

        av_m0  = _mm_mul_pd(v128_CRTM_9_7, av_in6);
        av_s0  = _mm_sub_pd(av_in0, av_in5);
        av_m1  = _mm_mul_pd(v128_CRTM_9_6, av_in5);
        av_s1  = _mm_add_pd(av_in0, av_m1);
        av_s2  = _mm_sub_pd(av_s0, av_m0);
        av_s3  = _mm_add_pd(av_s0, av_m0);
        av_s4  = _mm_add_pd(av_in7, av_in3);
        av_s5  = _mm_sub_pd(av_in7, av_in3);
        av_m2  = _mm_mul_pd(v128_CRTM_9_5, av_s5);
        av_s6  = _mm_add_pd(av_in8, av_in4);
        av_m3  = _mm_mul_pd(v128_CRTM_9_5, av_s6);
        av_s7  = _mm_sub_pd(av_in4, av_in8);
        av_s8  = _mm_add_pd(av_in1, av_s4);
        av_m4  = _mm_mul_pd(v128_CRTM_9_4, av_s7);
        av_s9  = _mm_add_pd(av_in2, av_m4);
        av_s10 = _mm_add_pd(av_m2, av_s9);
        av_s11 = _mm_sub_pd(av_s9, av_m2);
        av_m5  = _mm_mul_pd(v128_CRTM_9_4, av_s4);
        av_s12 = _mm_sub_pd(av_in1, av_m5);
        av_s13 = _mm_sub_pd(av_s12, av_m3);
        av_s14 = _mm_add_pd(av_s12, av_m3);
        av_m6  = _mm_mul_pd(v128_CRTM_9_6, av_s8);
        av_s15 = _mm_add_pd(av_s1, av_m6);
        // Output point 1: x(0)
        v_out0 = av_s15;
        STR_128_D(curr_out, v_out_stride, v_out0);

        av_s16 = _mm_sub_pd(av_s1, av_s8);
        av_s17 = _mm_sub_pd(av_in2, av_s7);
        av_m7  = _mm_mul_pd(v128_CRTM_9_7, av_s17);
        av_s18 = _mm_sub_pd(av_s16, av_m7);
        av_s19 = _mm_add_pd(av_s16, av_m7);
        // Output point 7: x(6)
        v_out6 = av_s18;
        curr_out = out + out_strides[6];
        STR_128_D(curr_out, v_out_stride, v_out6);
        // Output point 13: x(12)
        v_out12 = av_s19;
        curr_out = out + out_strides[12];
        STR_128_D(curr_out, v_out_stride, v_out12);

        av_m8  = _mm_mul_pd(v128_CRTM_9_0, av_s13);
        av_m9  = _mm_mul_pd(v128_CRTM_9_1, av_s10);
        av_s20 = _mm_sub_pd(av_m8, av_m9);
        av_m10 = _mm_mul_pd(v128_CRTM_9_11, av_s10);
        av_m11 = _mm_mul_pd(v128_CRTM_9_10, av_s13);
        av_s21 = _mm_add_pd(av_m11, av_m10);
        av_s22 = _mm_sub_pd(av_s2, av_s20);
        av_m12 = _mm_mul_pd(v128_CRTM_9_6, av_s20);
        av_s23 = _mm_add_pd(av_s2, av_m12);
        // Output point 3: x(2)
        v_out2 = av_s23;
        curr_out = out + out_strides[2];
        STR_128_D(curr_out, v_out_stride, v_out2);

        av_s24 = _mm_add_pd(av_s22, av_s21);
        av_s25 = _mm_sub_pd(av_s22, av_s21);
        // Output point 15: x(14)
        v_out14 = av_s24;
        curr_out = out + out_strides[14];
        STR_128_D(curr_out, v_out_stride, v_out14);
        // Output point 9: x(8)
        v_out8 = av_s25;
        curr_out = out + out_strides[8];
        STR_128_D(curr_out, v_out_stride, v_out8);

        av_m13 = _mm_mul_pd(v128_CRTM_9_8, av_s14);
        av_m14 = _mm_mul_pd(v128_CRTM_9_9, av_s11);
        av_s26 = _mm_add_pd(av_m13, av_m14);
        av_m15 = _mm_mul_pd(v128_CRTM_9_2, av_s14);
        av_m16 = _mm_mul_pd(v128_CRTM_9_3, av_s11);
        av_s27 = _mm_sub_pd(av_m15, av_m16);
        av_s28 = _mm_sub_pd(av_s3, av_s27);
        av_m17 = _mm_mul_pd(v128_CRTM_9_6, av_s27);
        av_s29 = _mm_add_pd(av_s3, av_m17);
        // Output point 5: x(4)
        v_out4 = av_s29;
        curr_out = out + out_strides[4];
        STR_128_D(curr_out, v_out_stride, v_out4);

        av_s30 = _mm_add_pd(av_s28, av_s26);
        av_s31 = _mm_sub_pd(av_s28, av_s26);
        // Output point 17: x(16)
        v_out16 = av_s30;
        curr_out = out + out_strides[16];
        STR_128_D(curr_out, v_out_stride, v_out16);
        // Output point 11: x(10)
        v_out10 = av_s31;
        curr_out = out + out_strides[10];
        STR_128_D(curr_out, v_out_stride, v_out10);

        // Shifted DFT
        __m128d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
                bv_in8;
        __m128d bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
                bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16,
                bv_s17, bv_s18, bv_s19, bv_s20, bv_s21, bv_s22,
               bv_s23;
        __m128d bv_m0, bv_m1, bv_m2, bv_m3, bv_m4, bv_m5, bv_m6, bv_m7, bv_m8,
                bv_m9, bv_m10, bv_m11, bv_m12, bv_m13, bv_m14, bv_m15, bv_m16,
                bv_m17;

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
        curr_in = in + in_strides[17];
        // Input point 18: X(17)
        LDR_128_D(curr_in, v_in_stride, bv_in8);
 
        bv_m0  = _mm_mul_pd(v128_CRTM_9_7, bv_in3);
        bv_s0  = _mm_sub_pd(bv_in2, bv_in8);
        bv_m1  = _mm_mul_pd(v128_CRTM_9_6, bv_in2);
        bv_s1  = _mm_add_pd(bv_m1, bv_in8);
        bv_s2  = _mm_sub_pd(bv_s0, bv_m0);
        bv_s3  = _mm_add_pd(bv_s0, bv_m0);
        bv_s4  = _mm_add_pd(bv_in0, bv_in4);
        bv_s5  = _mm_sub_pd(bv_in4, bv_in0);
        bv_m2  = _mm_mul_pd(v128_CRTM_9_5, bv_s5);
        bv_s6  = _mm_sub_pd(bv_in5, bv_in1);
        bv_s7  = _mm_add_pd(bv_in1, bv_in5);
        bv_m3  = _mm_mul_pd(v128_CRTM_9_5, bv_s7);
        bv_s8  = _mm_add_pd(bv_in6, bv_s4);
        bv_m4  = _mm_mul_pd(v128_CRTM_9_4, bv_s6);
        bv_s9  = _mm_add_pd(bv_m4, bv_in7);
        bv_s10 = _mm_sub_pd(bv_m2, bv_s9);
        bv_s11 = _mm_add_pd(bv_m2, bv_s9);
        bv_m5  = _mm_mul_pd(v128_CRTM_9_4, bv_s4);
        bv_s12 = _mm_sub_pd(bv_m5, bv_in6);
        bv_s13 = _mm_add_pd(bv_s12, bv_m3);
        bv_s14 = _mm_sub_pd(bv_s12, bv_m3);
 
        bv_s15 = _mm_sub_pd(bv_s8, bv_s1);
        bv_s16 = _mm_sub_pd(bv_s6, bv_in7);
        bv_m6  = _mm_mul_pd(v128_CRTM_9_7, bv_s16);
        bv_m7  = _mm_mul_pd(v128_CRTM_9_9, bv_s10);
        bv_m8  = _mm_mul_pd(v128_CRTM_9_8, bv_s13);
        bv_s17 = _mm_sub_pd(bv_m7, bv_m8);
        bv_m9  = _mm_mul_pd(v128_CRTM_9_2, bv_s13);
        bv_m10 = _mm_mul_pd(v128_CRTM_9_3, bv_s10);
        bv_s18 = _mm_add_pd(bv_m9, bv_m10);
        bv_s19 = _mm_sub_pd(bv_s3, bv_s18);
        bv_m11 = _mm_mul_pd(v128_CRTM_9_10, bv_s14);
        bv_m12 = _mm_mul_pd(v128_CRTM_9_11, bv_s11);
        bv_s20 = _mm_add_pd(bv_m11, bv_m12);
        bv_m13 = _mm_mul_pd(v128_CRTM_9_0, bv_s14);
        bv_m14 = _mm_mul_pd(v128_CRTM_9_1, bv_s11);
        bv_s21 = _mm_sub_pd(bv_m13, bv_m14);
        bv_s22 = _mm_sub_pd(bv_s21, bv_s2);
 
        bv_m15 = _mm_mul_pd(v128_CRTM_9_6, bv_s8);
        // Output point 2: x(1)
        v_out1 = _mm_add_pd(bv_m15, bv_s1);
        curr_out = out + out_strides[1];
        STR_128_D(curr_out, v_out_stride, v_out1);

        bv_m16 = _mm_mul_pd(v128_CRTM_9_6, bv_s21);
        // Output point 4: x(3)
        v_out3 = _mm_add_pd(bv_m16, bv_s2);
        curr_out = out + out_strides[3];
        STR_128_D(curr_out, v_out_stride, v_out3);

        bv_m17 = _mm_mul_pd(v128_CRTM_9_6, bv_s18);
        bv_s23 = NEGATE_128_D(_mm_add_pd(bv_m17, bv_s3));
        // Output point 6: x(5)
        v_out5 = bv_s23;
        curr_out = out + out_strides[5];
        STR_128_D(curr_out, v_out_stride, v_out5);

        // Output point 8: x(7)
        v_out7 = _mm_add_pd(bv_s15, bv_m6);
        curr_out = out + out_strides[7];
        STR_128_D(curr_out, v_out_stride, v_out7);
        // Output point 10: x(9)
        v_out9 = _mm_add_pd(bv_s22, bv_s20);
        curr_out = out + out_strides[9];
        STR_128_D(curr_out, v_out_stride, v_out9);
        // Output point 12: x(11)
        v_out11 = _mm_add_pd(bv_s19, bv_s17);
        curr_out = out + out_strides[11];
        STR_128_D(curr_out, v_out_stride, v_out11);
        // Output point 14: x(13)
        v_out13 = _mm_sub_pd(bv_m6, bv_s15);
        curr_out = out + out_strides[13];
        STR_128_D(curr_out, v_out_stride, v_out13);
        // Output point 16: x(15)
        v_out15 = _mm_sub_pd(bv_s20, bv_s22);
        curr_out = out + out_strides[15];
        STR_128_D(curr_out, v_out_stride, v_out15);
        // Output point 18: x(17)
        v_out17 = _mm_sub_pd(bv_s17, bv_s19);
        curr_out = out + out_strides[17];
        STR_128_D(curr_out, v_out_stride, v_out17);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        // Standard DFT
        DOUBLE av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
               av_in8;
        DOUBLE av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
               av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16, 
               av_s17, av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24,
               av_s25, av_s26, av_s27, av_s28, av_s29, av_s30, av_s31;
        DOUBLE av_m0, av_m1, av_m2, av_m3, av_m4, av_m5, av_m6, av_m7, av_m8,
               av_m9, av_m10, av_m11, av_m12, av_m13, av_m14, av_m15, av_m16,
               av_m17;
 
        // Input point 1: X(0)
        av_in0 = *in;
        // Input point 4: X(3)
        av_in1 = in[in_strides[3]];
        // Input point 5: X(4)
        av_in2 = in[in_strides[4]];
        // Input point 8: X(7)
        av_in3 = in[in_strides[7]];
        // Input point 9: X(8)
        av_in4 = in[in_strides[8]];
        // Input point 12: X(11)
        av_in5 = in[in_strides[11]];
        // Input point 13: X(12)
        av_in6 = in[in_strides[12]];
        // Input point 16: X(15)
        av_in7 = in[in_strides[15]];
        // Input point 17: X(16)
        av_in8 = in[in_strides[16]];

        av_m0  = CRTM_9_7 * av_in6;
        av_s0  = av_in0 - av_in5;
        av_m1  = CRTM_9_6 * av_in5;
        av_s1  = av_in0 + av_m1;
        av_s2  = av_s0 - av_m0;
        av_s3  = av_s0 + av_m0;
        av_s4  = av_in7 + av_in3;
        av_s5  = av_in7 - av_in3;
        av_m2  = CRTM_9_5 * av_s5;
        av_s6  = av_in8 + av_in4;
        av_m3  = CRTM_9_5 * av_s6;
        av_s7  = av_in4 - av_in8;
        av_s8  = av_in1 + av_s4;
        av_m4  = CRTM_9_4 * av_s7;
        av_s9  = av_in2 + av_m4;
        av_s10 = av_m2 + av_s9;
        av_s11 = av_s9 - av_m2;
        av_m5  = CRTM_9_4 * av_s4;
        av_s12 = av_in1 - av_m5;
        av_s13 = av_s12 - av_m3;
        av_s14 = av_s12 + av_m3;
        av_m6  = CRTM_9_6 * av_s8;
        av_s15 = av_s1 + av_m6;
        // Output point 1: x(0)
        *out = av_s15;

        av_s16 = av_s1 - av_s8;
        av_s17 = av_in2 - av_s7;
        av_m7  = CRTM_9_7 * av_s17;
        av_s18 = av_s16 - av_m7;
        av_s19 = av_s16 + av_m7;
        // Output point 7: x(6)
        out[out_strides[6]] = av_s18;
        // Output point 13: x(12)
        out[out_strides[12]] = av_s19;

        av_m8  = CRTM_9_0 * av_s13;
        av_m9  = CRTM_9_1 * av_s10;
        av_s20 = av_m8 - av_m9;
        av_m10 = CRTM_9_11 * av_s10;
        av_m11 = CRTM_9_10 * av_s13;
        av_s21 = av_m11 + av_m10;
        av_s22 = av_s2 - av_s20;
        av_m12 = CRTM_9_6 * av_s20;
        av_s23 = av_s2 + av_m12;
        // Output point 3: x(2)
        out[out_strides[2]] = av_s23;
        av_s24 = av_s22 + av_s21;
        av_s25 = av_s22 - av_s21;
        // Output point 15: x(14)
        out[out_strides[14]] = av_s24;
        // Output point 9: x(8)
        out[out_strides[8]] = av_s25;

        av_m13 = CRTM_9_8 * av_s14;
        av_m14 = CRTM_9_9 * av_s11;
        av_s26 = av_m13 + av_m14;
        av_m15 = CRTM_9_2 * av_s14;
        av_m16 = CRTM_9_3 * av_s11;
        av_s27 = av_m15 - av_m16;
        av_s28 = av_s3 - av_s27;
        av_m17 = CRTM_9_6 * av_s27;
        av_s29 = av_s3 + av_m17;
        // Output point 5: x(4)
        out[out_strides[4]] = av_s29;
        av_s30 = av_s28 + av_s26;
        av_s31 = av_s28 - av_s26;
        // Output point 17: x(16)
        out[out_strides[16]] = av_s30;
        // Output point 11: x(10)
        out[out_strides[10]] = av_s31;

        // Shifted DFT
        DOUBLE bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8;
        DOUBLE bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
               bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16,
               bv_s17, bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24,
               bv_s25, bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31;
        DOUBLE bv_m0, bv_m1, bv_m2, bv_m3, bv_m4, bv_m5, bv_m6, bv_m7, bv_m8,
               bv_m9, bv_m10, bv_m11, bv_m12, bv_m13, bv_m14, bv_m15, bv_m16,
               bv_m17;

        // Input point 2: X(1)
        bv_in0 = in[in_strides[1]];
        // Input point 3: X(2)
        bv_in1 = in[in_strides[2]];
        // Input point 6: X(5)
        bv_in2 = in[in_strides[5]];
        // Input point 7: X(6)
        bv_in3 = in[in_strides[6]];
        // Input point 10: X(9)
        bv_in4 = in[in_strides[9]];
        // Input point 11: X(10)
        bv_in5 = in[in_strides[10]];
        // Input point 14: X(13)
        bv_in6 = in[in_strides[13]];
        // Input point 15: X(14)
        bv_in7 = in[in_strides[14]];
        // Input point 18: X(17)
        bv_in8 = in[in_strides[17]];
 
        bv_m0  = CRTM_9_7 * bv_in3;
        bv_s0  = bv_in2 - bv_in8;
        bv_m1  = CRTM_9_6 * bv_in2;
        bv_s1  = bv_m1 + bv_in8;
        bv_s2  = bv_s0 - bv_m0;
        bv_s3  = bv_s0 + bv_m0;
        bv_s4  = bv_in0 + bv_in4;
        bv_s5  = bv_in4 - bv_in0;
        bv_m2  = CRTM_9_5 * bv_s5;
        bv_s6  = bv_in5 - bv_in1;
        bv_s7  = bv_in1 + bv_in5;
        bv_m3  = CRTM_9_5 * bv_s7;
        bv_s8  = bv_in6 + bv_s4;
        bv_m4  = CRTM_9_4 * bv_s6;
        bv_s9  = bv_m4 + bv_in7;
        bv_s10 = bv_m2 - bv_s9;
        bv_s11 = bv_m2 + bv_s9;
        bv_m5  = CRTM_9_4 * bv_s4;
        bv_s12 = bv_m5 - bv_in6;
        bv_s13 = bv_s12 + bv_m3;
        bv_s14 = bv_s12 - bv_m3;
 
        bv_s15 = bv_s8 - bv_s1;
        bv_s16 = bv_s6 - bv_in7;
        bv_m6  = CRTM_9_7 * bv_s16;
        bv_m7  = CRTM_9_9 * bv_s10;
        bv_m8  = CRTM_9_8 * bv_s13;
        bv_s17 = bv_m7 - bv_m8;
        bv_m9  = CRTM_9_2 * bv_s13;
        bv_m10 = CRTM_9_3 * bv_s10;
        bv_s18 = bv_m9 + bv_m10;
        bv_s19 = bv_s3 - bv_s18;
        bv_m11 = CRTM_9_10 * bv_s14;
        bv_m12 = CRTM_9_11 * bv_s11;
        bv_s20 = bv_m11 + bv_m12;
        bv_m13 = CRTM_9_0 * bv_s14;
        bv_m14 = CRTM_9_1 * bv_s11;
        bv_s21 = bv_m13 - bv_m14;
        bv_s22 = bv_s21 - bv_s2;
 
        bv_m15 = CRTM_9_6 * bv_s8;
        bv_s23 = bv_m15 + bv_s1;
        // Output point 2: x(1)
        out[out_strides[1]] = bv_s23;

        bv_m16 = CRTM_9_6 * bv_s21;
        bv_s24 = bv_m16 + bv_s2;
        // Output point 4: x(3)
        out[out_strides[3]] = bv_s24;

        bv_m17 = CRTM_9_6 * bv_s18;
        bv_s25 = -(bv_m17 + bv_s3);
        // Output point 6: x(5)
        out[out_strides[5]] = bv_s25;

        bv_s26 = bv_s15 + bv_m6;
        // Output point 8: x(7)
        out[out_strides[7]] = bv_s26;

        bv_s27 = bv_s22 + bv_s20;
        // Output point 10: x(9)
        out[out_strides[9]] = bv_s27;

        bv_s28 = bv_s19 + bv_s17;
        // Output point 12: x(11)
        out[out_strides[11]] = bv_s28;

        bv_s29 = bv_m6 - bv_s15;
        // Output point 14: x(13)
        out[out_strides[13]] = bv_s29;

        bv_s30 = bv_s20 - bv_s22;
        // Output point 16: x(15)
        out[out_strides[15]] = bv_s30;

        bv_s31 = bv_s17 - bv_s19;
        // Output point 18: x(17)
        out[out_strides[17]] = bv_s31;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hcf_rfft9avx256(UINT8 precision, UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hcf_rfft9avx256_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft9avx256_fp64_fwd;
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
            return r2hcf_rfft9avx256_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft9avx256_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
