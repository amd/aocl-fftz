// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft11avx128.c
 *
 *  @brief Radix-11 r2hc Real-FFT kernel with AVX-128 operations using x86 SIMD
 *  intrinsics
 *
 *  This file contains the DIT radix-11 real-to-halfcomplex implementations
 *  using AVX128 SIMD operations for single-precision and double-precision
 *  inputs.
 *
 *  @author Amrin Fathima
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/r2hc_simd_avx128.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                    {{{0, 50, 60, 64, 44, 1},
                                                      {0, 51, 60, 64, 48, 0}},
                                                     {{0, 50, 60, 32, 8,  1},
                                                      {0, 51, 60, 32, 8,  0}}};

ops_cycles_t get_ops_cnt_r2hc_rfft11avx128(UINT8 precision, UINT8 direction)
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

static VOID rfft11avx128_fp32_fwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                  VOID *out_imag, INTP n,
                                  aoclfftz_strides_t *strides, VOID *twd,
                                  UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");

    const FLOAT CRTM_11_1 = 0.841253532831181168861811648919367717513292498f;
    const FLOAT CRTM_11_2 = 0.540640817455597582107635954318691695431770608f;
    const FLOAT CRTM_11_3 = 0.415415013001886425529274149229623203524004910f;
    const FLOAT CRTM_11_4 = 0.909631995354518371411715383079028460060241051f;
    const FLOAT CRTM_11_5 = 0.142314838273285140443792668616369668791051361f;
    const FLOAT CRTM_11_6 = 0.989821441880932732376092037776718787376519372f;
    const FLOAT CRTM_11_7 = 0.654860733945285064056925072466293553183791199f;
    const FLOAT CRTM_11_8 = 0.755749574354258283774035843972344420179717445f;
    const FLOAT CRTM_11_9 = 0.959492973614497389890368057066327699062454848f;
    const FLOAT CRTM_11_10 = 0.281732556841429697711417915346616899035777899f;

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
    INTP N = n / NUM_SETS_REAL_128_S;

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
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9, v_in10;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
               v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27,
               v_s28, v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36,
               v_s37, v_s38, v_s39, v_s40, v_s41, v_s42, v_s43, v_s44, v_s45,
               v_s46, v_s47, v_s48;
        __m128 v_m0, v_m1, v_m2, v_m3, v_m4, v_m5, v_m6, v_m7, v_m8, v_m9,
               v_m10, v_m11, v_m12, v_m13, v_m14, v_m15, v_m16, v_m17, v_m18,
               v_m19, v_m20, v_m21, v_m22, v_m23, v_m24, v_m25, v_m26, v_m27,
               v_m28, v_m29, v_m30, v_m31, v_m32, v_m33, v_m34, v_m35, v_m36,
               v_m37, v_m38, v_m39, v_m40, v_m41, v_m42, v_m43, v_m44, v_m45,
               v_m46, v_m47, v_m48, v_m49;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_128_S(curr_in, v_in_stride, v_in1);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_128_S(curr_in, v_in_stride, v_in2);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_128_S(curr_in, v_in_stride, v_in3);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_128_S(curr_in, v_in_stride, v_in4);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_128_S(curr_in, v_in_stride, v_in5);
        // Input point 7: x(6)
        curr_in = in + in_strides[6];
        LDR_128_S(curr_in, v_in_stride, v_in6);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_128_S(curr_in, v_in_stride, v_in7);
        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDR_128_S(curr_in, v_in_stride, v_in8);
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDR_128_S(curr_in, v_in_stride, v_in9);
        // Input point 11: x(10)
        curr_in = in + in_strides[10];
        LDR_128_S(curr_in, v_in_stride, v_in10);

        v_s0 = _mm_add_ps(v_in1, v_in10);
        v_s1 = _mm_add_ps(v_in2, v_in9);
        v_s2 = _mm_add_ps(v_in3, v_in8);
        v_s3 = _mm_add_ps(v_in4, v_in7);
        v_s4 = _mm_add_ps(v_in5, v_in6);
        v_s5 = _mm_sub_ps(v_in1, v_in10);
        v_s6 = _mm_sub_ps(v_in2, v_in9);
        v_s7 = _mm_sub_ps(v_in3, v_in8);
        v_s8 = _mm_sub_ps(v_in4, v_in7);
        v_s9 = _mm_sub_ps(v_in5, v_in6);

        v_s10 = _mm_add_ps(v_s0, v_s1);
        v_s11 = _mm_add_ps(v_s2, v_s3);
        v_s12 = _mm_add_ps(v_s4, v_in0);
        v_s13 = _mm_add_ps(v_s10, v_s11);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_s12, v_s13);
        STR_128_S(curr_out, v_out_stride, v_out0);

        v_m0 = _mm_mul_ps(v_CRTM_11_1, v_s0);
        v_m1 = _mm_mul_ps(v_CRTM_11_3, v_s1);
        v_m2 = _mm_mul_ps(v_CRTM_11_5, v_s2);
        v_m3 = _mm_mul_ps(v_CRTM_11_7, v_s3);
        v_m4 = _mm_mul_ps(v_CRTM_11_9, v_s4);

        v_s14 = _mm_add_ps(v_m0, v_m1);
        v_s15 = _mm_add_ps(v_m2, v_m3);
        v_s16 = _mm_sub_ps(v_in0, v_m4);
        v_s17 = _mm_sub_ps(v_s14, v_s15);

        // Output point 2: X(1)
        v_out1 = _mm_add_ps(v_s16, v_s17);

        v_m5 = _mm_mul_ps(v_CRTM_11_2, v_s5);
        v_m6 = _mm_mul_ps(v_CRTM_11_4, v_s6);
        v_m7 = _mm_mul_ps(v_CRTM_11_6, v_s7);
        v_m8 = _mm_mul_ps(v_CRTM_11_8, v_s8);
        v_m9 = _mm_mul_ps(v_CRTM_11_10, v_s9);

        v_s18 = _mm_add_ps(v_m5, v_m6);
        v_s19 = _mm_add_ps(v_m7, v_m8);
        v_s20 = _mm_add_ps(v_s19, v_m9);

        // Output point 3: X(2)
        v_out2 = NEGATE_128_S(_mm_add_ps(v_s18, v_s20));
        curr_out = out + out_strides[1];
        STRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);

        v_m10 = _mm_mul_ps(v_CRTM_11_1, v_s4);
        v_m11 = _mm_mul_ps(v_CRTM_11_3, v_s0);
        v_m12 = _mm_mul_ps(v_CRTM_11_5, v_s3);
        v_m13 = _mm_mul_ps(v_CRTM_11_7, v_s1);
        v_m14 = _mm_mul_ps(v_CRTM_11_9, v_s2);

        v_s21 = _mm_add_ps(v_m10, v_m11);
        v_s22 = _mm_add_ps(v_m12, v_m13);
        v_s23 = _mm_sub_ps(v_in0, v_m14);
        v_s24 = _mm_sub_ps(v_s21, v_s22);

        // Output point 4: X(3)
        v_out3 = _mm_add_ps(v_s23, v_s24);

        v_m15 = _mm_mul_ps(v_CRTM_11_2, v_s9);
        v_m16 = _mm_mul_ps(v_CRTM_11_4, v_s5);
        v_m17 = _mm_mul_ps(v_CRTM_11_6, v_s8);
        v_m18 = _mm_mul_ps(v_CRTM_11_8, v_s6);
        v_m19 = _mm_mul_ps(v_CRTM_11_10, v_s7);

        v_s25 = _mm_sub_ps(v_m15, v_m16);
        v_s26 = _mm_sub_ps(v_m17, v_m18);
        v_s27 = _mm_add_ps(v_s26, v_m19);

        // Output point 5: X(4)
        v_out4 = _mm_add_ps(v_s25, v_s27);
        curr_out = out + out_strides[3];
        STRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);

        v_m20 = _mm_mul_ps(v_CRTM_11_1, v_s3);
        v_m21 = _mm_mul_ps(v_CRTM_11_3, v_s2);
        v_m22 = _mm_mul_ps(v_CRTM_11_5, v_s0);
        v_m23 = _mm_mul_ps(v_CRTM_11_7, v_s4);
        v_m24 = _mm_mul_ps(v_CRTM_11_9, v_s1);

        v_s28 = _mm_add_ps(v_m20, v_m21);
        v_s29 = _mm_add_ps(v_m22, v_m23);
        v_s30 = _mm_sub_ps(v_in0, v_m24);
        v_s31 = _mm_sub_ps(v_s30, v_s29);

        // Output point 6: X(5)
        v_out5 = _mm_add_ps(v_s28, v_s31);

        v_m25 = _mm_mul_ps(v_CRTM_11_2, v_s8);
        v_m26 = _mm_mul_ps(v_CRTM_11_4, v_s7);
        v_m27 = _mm_mul_ps(v_CRTM_11_6, v_s5);
        v_m28 = _mm_mul_ps(v_CRTM_11_8, v_s9);
        v_m29 = _mm_mul_ps(v_CRTM_11_10, v_s6);

        v_s32 = _mm_sub_ps(v_m26, v_m25);
        v_s33 = _mm_add_ps(v_m27, v_m28);
        v_s34 = _mm_sub_ps(v_s32, v_s33);

        // Output point 7: X(6)
        v_out6 = _mm_add_ps(v_s34, v_m29);
        curr_out = out + out_strides[5];
        STRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);

        v_m30 = _mm_mul_ps(v_CRTM_11_1, v_s2);
        v_m31 = _mm_mul_ps(v_CRTM_11_3, v_s4);
        v_m32 = _mm_mul_ps(v_CRTM_11_5, v_s1);
        v_m33 = _mm_mul_ps(v_CRTM_11_7, v_s0);
        v_m34 = _mm_mul_ps(v_CRTM_11_9, v_s3);

        v_s35 = _mm_add_ps(v_m30, v_m31);
        v_s36 = _mm_add_ps(v_m32, v_m33);
        v_s37 = _mm_sub_ps(v_in0, v_m34);
        v_s38 = _mm_sub_ps(v_s35, v_s36);

        // Output point 8: X(7)
        v_out7 = _mm_add_ps(v_s38, v_s37);

        v_m35 = _mm_mul_ps(v_CRTM_11_2, v_s7);
        v_m36 = _mm_mul_ps(v_CRTM_11_4, v_s9);
        v_m37 = _mm_mul_ps(v_CRTM_11_6, v_s6);
        v_m38 = _mm_mul_ps(v_CRTM_11_8, v_s5);
        v_m39 = _mm_mul_ps(v_CRTM_11_10, v_s8);

        v_s39 = _mm_sub_ps(v_m36, v_m35);
        v_s40 = _mm_sub_ps(v_m37, v_m38);
        v_s41 = _mm_sub_ps(v_s40, v_m39);

        // Output point 9: X(8)
        v_out8 = _mm_add_ps(v_s39, v_s41);
        curr_out = out + out_strides[7];
        STRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);

        v_m40 = _mm_mul_ps(v_CRTM_11_1, v_s1);
        v_m41 = _mm_mul_ps(v_CRTM_11_3, v_s3);
        v_m42 = _mm_mul_ps(v_CRTM_11_5, v_s4);
        v_m43 = _mm_mul_ps(v_CRTM_11_7, v_s2);
        v_m44 = _mm_mul_ps(v_CRTM_11_9, v_s0);

        v_s42 = _mm_add_ps(v_m40, v_m41);
        v_s43 = _mm_add_ps(v_m42, v_m43);
        v_s44 = _mm_sub_ps(v_in0, v_m44);
        v_s45 = _mm_sub_ps(v_s42, v_s43);

        // Output point 10: X(9)
        v_out9 = _mm_add_ps(v_s44, v_s45);

        v_m45 = _mm_mul_ps(v_CRTM_11_2, v_s6);
        v_m46 = _mm_mul_ps(v_CRTM_11_4, v_s8);
        v_m47 = _mm_mul_ps(v_CRTM_11_6, v_s9);
        v_m48 = _mm_mul_ps(v_CRTM_11_8, v_s7);
        v_m49 = _mm_mul_ps(v_CRTM_11_10, v_s5);

        v_s46 = _mm_add_ps(v_m45, v_m46);
        v_s47 = _mm_add_ps(v_m47, v_m48);
        v_s48 = _mm_sub_ps(v_s46, v_s47);

        // Output point 11: X(10)
        v_out10 = _mm_sub_ps(v_s48, v_m49);
        curr_out = out + out_strides[9];
        STRI_2x128_S(curr_out, v_out_stride, v_out9, v_out10);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (n & 2)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9, v_in10;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
               v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27,
               v_s28, v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36,
               v_s37, v_s38, v_s39, v_s40, v_s41, v_s42, v_s43, v_s44, v_s45,
               v_s46, v_s47, v_s48;
        __m128 v_m0, v_m1, v_m2, v_m3, v_m4, v_m5, v_m6, v_m7, v_m8, v_m9,
               v_m10, v_m11, v_m12, v_m13, v_m14, v_m15, v_m16, v_m17, v_m18,
               v_m19, v_m20, v_m21, v_m22, v_m23, v_m24, v_m25, v_m26, v_m27,
               v_m28, v_m29, v_m30, v_m31, v_m32, v_m33, v_m34, v_m35, v_m36,
               v_m37, v_m38, v_m39, v_m40, v_m41, v_m42, v_m43, v_m44, v_m45,
               v_m46, v_m47, v_m48, v_m49;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDHR_128_S(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDHR_128_S(curr_in, v_in_stride, v_in1);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDHR_128_S(curr_in, v_in_stride, v_in2);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDHR_128_S(curr_in, v_in_stride, v_in3);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDHR_128_S(curr_in, v_in_stride, v_in4);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDHR_128_S(curr_in, v_in_stride, v_in5);
        // Input point 7: x(6)
        curr_in = in + in_strides[6];
        LDHR_128_S(curr_in, v_in_stride, v_in6);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDHR_128_S(curr_in, v_in_stride, v_in7);
        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDHR_128_S(curr_in, v_in_stride, v_in8);
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDHR_128_S(curr_in, v_in_stride, v_in9);
        // Input point 11: x(10)
        curr_in = in + in_strides[10];
        LDHR_128_S(curr_in, v_in_stride, v_in10);

        v_s0 = _mm_add_ps(v_in1, v_in10);
        v_s1 = _mm_add_ps(v_in2, v_in9);
        v_s2 = _mm_add_ps(v_in3, v_in8);
        v_s3 = _mm_add_ps(v_in4, v_in7);
        v_s4 = _mm_add_ps(v_in5, v_in6);
        v_s5 = _mm_sub_ps(v_in1, v_in10);
        v_s6 = _mm_sub_ps(v_in2, v_in9);
        v_s7 = _mm_sub_ps(v_in3, v_in8);
        v_s8 = _mm_sub_ps(v_in4, v_in7);
        v_s9 = _mm_sub_ps(v_in5, v_in6);

        v_s10 = _mm_add_ps(v_s0, v_s1);
        v_s11 = _mm_add_ps(v_s2, v_s3);
        v_s12 = _mm_add_ps(v_s4, v_in0);
        v_s13 = _mm_add_ps(v_s10, v_s11);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_s12, v_s13);
        STHR_128_S(curr_out, v_out_stride, v_out0);

        v_m0 = _mm_mul_ps(v_CRTM_11_1, v_s0);
        v_m1 = _mm_mul_ps(v_CRTM_11_3, v_s1);
        v_m2 = _mm_mul_ps(v_CRTM_11_5, v_s2);
        v_m3 = _mm_mul_ps(v_CRTM_11_7, v_s3);
        v_m4 = _mm_mul_ps(v_CRTM_11_9, v_s4);

        v_s14 = _mm_add_ps(v_m0, v_m1);
        v_s15 = _mm_add_ps(v_m2, v_m3);
        v_s16 = _mm_sub_ps(v_in0, v_m4);
        v_s17 = _mm_sub_ps(v_s14, v_s15);

        // Output point 2: X(1)
        v_out1 = _mm_add_ps(v_s16, v_s17);

        v_m5 = _mm_mul_ps(v_CRTM_11_2, v_s5);
        v_m6 = _mm_mul_ps(v_CRTM_11_4, v_s6);
        v_m7 = _mm_mul_ps(v_CRTM_11_6, v_s7);
        v_m8 = _mm_mul_ps(v_CRTM_11_8, v_s8);
        v_m9 = _mm_mul_ps(v_CRTM_11_10, v_s9);

        v_s18 = _mm_add_ps(v_m5, v_m6);
        v_s19 = _mm_add_ps(v_m7, v_m8);
        v_s20 = _mm_add_ps(v_s19, v_m9);

        // Output point 3: X(2)
        v_out2 = NEGATE_128_S(_mm_add_ps(v_s18, v_s20));
        curr_out = out + out_strides[1];
        STHRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);

        v_m10 = _mm_mul_ps(v_CRTM_11_1, v_s4);
        v_m11 = _mm_mul_ps(v_CRTM_11_3, v_s0);
        v_m12 = _mm_mul_ps(v_CRTM_11_5, v_s3);
        v_m13 = _mm_mul_ps(v_CRTM_11_7, v_s1);
        v_m14 = _mm_mul_ps(v_CRTM_11_9, v_s2);

        v_s21 = _mm_add_ps(v_m10, v_m11);
        v_s22 = _mm_add_ps(v_m12, v_m13);
        v_s23 = _mm_sub_ps(v_in0, v_m14);
        v_s24 = _mm_sub_ps(v_s21, v_s22);

        // Output point 4: X(3)
        v_out3 = _mm_add_ps(v_s23, v_s24);

        v_m15 = _mm_mul_ps(v_CRTM_11_2, v_s9);
        v_m16 = _mm_mul_ps(v_CRTM_11_4, v_s5);
        v_m17 = _mm_mul_ps(v_CRTM_11_6, v_s8);
        v_m18 = _mm_mul_ps(v_CRTM_11_8, v_s6);
        v_m19 = _mm_mul_ps(v_CRTM_11_10, v_s7);

        v_s25 = _mm_sub_ps(v_m15, v_m16);
        v_s26 = _mm_sub_ps(v_m17, v_m18);
        v_s27 = _mm_add_ps(v_s26, v_m19);

        // Output point 5: X(4)
        v_out4 = _mm_add_ps(v_s25, v_s27);
        curr_out = out + out_strides[3];
        STHRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);

        v_m20 = _mm_mul_ps(v_CRTM_11_1, v_s3);
        v_m21 = _mm_mul_ps(v_CRTM_11_3, v_s2);
        v_m22 = _mm_mul_ps(v_CRTM_11_5, v_s0);
        v_m23 = _mm_mul_ps(v_CRTM_11_7, v_s4);
        v_m24 = _mm_mul_ps(v_CRTM_11_9, v_s1);

        v_s28 = _mm_add_ps(v_m20, v_m21);
        v_s29 = _mm_add_ps(v_m22, v_m23);
        v_s30 = _mm_sub_ps(v_in0, v_m24);
        v_s31 = _mm_sub_ps(v_s30, v_s29);

        // Output point 6: X(5)
        v_out5 = _mm_add_ps(v_s28, v_s31);

        v_m25 = _mm_mul_ps(v_CRTM_11_2, v_s8);
        v_m26 = _mm_mul_ps(v_CRTM_11_4, v_s7);
        v_m27 = _mm_mul_ps(v_CRTM_11_6, v_s5);
        v_m28 = _mm_mul_ps(v_CRTM_11_8, v_s9);
        v_m29 = _mm_mul_ps(v_CRTM_11_10, v_s6);

        v_s32 = _mm_sub_ps(v_m26, v_m25);
        v_s33 = _mm_add_ps(v_m27, v_m28);
        v_s34 = _mm_sub_ps(v_s32, v_s33);

        // Output point 7: X(6)
        v_out6 = _mm_add_ps(v_s34, v_m29);
        curr_out = out + out_strides[5];
        STHRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);

        v_m30 = _mm_mul_ps(v_CRTM_11_1, v_s2);
        v_m31 = _mm_mul_ps(v_CRTM_11_3, v_s4);
        v_m32 = _mm_mul_ps(v_CRTM_11_5, v_s1);
        v_m33 = _mm_mul_ps(v_CRTM_11_7, v_s0);
        v_m34 = _mm_mul_ps(v_CRTM_11_9, v_s3);

        v_s35 = _mm_add_ps(v_m30, v_m31);
        v_s36 = _mm_add_ps(v_m32, v_m33);
        v_s37 = _mm_sub_ps(v_in0, v_m34);
        v_s38 = _mm_sub_ps(v_s35, v_s36);

        // Output point 8: X(7)
        v_out7 = _mm_add_ps(v_s38, v_s37);

        v_m35 = _mm_mul_ps(v_CRTM_11_2, v_s7);
        v_m36 = _mm_mul_ps(v_CRTM_11_4, v_s9);
        v_m37 = _mm_mul_ps(v_CRTM_11_6, v_s6);
        v_m38 = _mm_mul_ps(v_CRTM_11_8, v_s5);
        v_m39 = _mm_mul_ps(v_CRTM_11_10, v_s8);

        v_s39 = _mm_sub_ps(v_m36, v_m35);
        v_s40 = _mm_sub_ps(v_m37, v_m38);
        v_s41 = _mm_sub_ps(v_s40, v_m39);

        // Output point 9: X(8)
        v_out8 = _mm_add_ps(v_s39, v_s41);
        curr_out = out + out_strides[7];
        STHRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);

        v_m40 = _mm_mul_ps(v_CRTM_11_1, v_s1);
        v_m41 = _mm_mul_ps(v_CRTM_11_3, v_s3);
        v_m42 = _mm_mul_ps(v_CRTM_11_5, v_s4);
        v_m43 = _mm_mul_ps(v_CRTM_11_7, v_s2);
        v_m44 = _mm_mul_ps(v_CRTM_11_9, v_s0);

        v_s42 = _mm_add_ps(v_m40, v_m41);
        v_s43 = _mm_add_ps(v_m42, v_m43);
        v_s44 = _mm_sub_ps(v_in0, v_m44);
        v_s45 = _mm_sub_ps(v_s42, v_s43);

        // Output point 10: X(9)
        v_out9 = _mm_add_ps(v_s44, v_s45);

        v_m45 = _mm_mul_ps(v_CRTM_11_2, v_s6);
        v_m46 = _mm_mul_ps(v_CRTM_11_4, v_s8);
        v_m47 = _mm_mul_ps(v_CRTM_11_6, v_s9);
        v_m48 = _mm_mul_ps(v_CRTM_11_8, v_s7);
        v_m49 = _mm_mul_ps(v_CRTM_11_10, v_s5);

        v_s46 = _mm_add_ps(v_m45, v_m46);
        v_s47 = _mm_add_ps(v_m47, v_m48);
        v_s48 = _mm_sub_ps(v_s46, v_s47);

        // Output point 11: X(10)
        v_out10 = _mm_sub_ps(v_s48, v_m49);
        curr_out = out + out_strides[9];
        STHRI_2x128_S(curr_out, v_out_stride, v_out9, v_out10);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        FLOAT in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10;
        FLOAT s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14,
              s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27,
              s28, s29, s30, s31, s32, s33, s34, s35, s36, s37, s38, s39, s40,
              s41, s42, s43, s44, s45, s46, s47, s48;
        FLOAT m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14,
              m15, m16, m17, m18, m19, m20, m21, m22, m23, m24, m25, m26, m27,
              m28, m29, m30, m31, m32, m33, m34, m35, m36, m37, m38, m39, m40,
              m41, m42, m43, m44, m45, m46, m47, m48, m49;

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
        // Input point 10: x(9)
        in9 = in[in_strides[9]];
        // Input point 11: x(10)
        in10 = in[in_strides[10]];

        s0 = in1 + in10;
        s1 = in2 + in9;
        s2 = in3 + in8;
        s3 = in4 + in7;
        s4 = in5 + in6;
        s5 = in1 - in10;
        s6 = in2 - in9;
        s7 = in3 - in8;
        s8 = in4 - in7;
        s9 = in5 - in6;

        s10 = s0 + s1;
        s11 = s2 + s3;
        s12 = s4 + in0;
        s13 = s10 + s11;

        // Output point 1: X(0)
        *out = s12 + s13;

        m0 = CRTM_11_1 * s0;
        m1 = CRTM_11_3 * s1;
        m2 = CRTM_11_5 * s2;
        m3 = CRTM_11_7 * s3;
        m4 = CRTM_11_9 * s4;

        s14 = m0 + m1;
        s15 = m2 + m3;
        s16 = in0 - m4;
        s17 = s14 - s15;

        // Output point 2: X(1)
        out[out_strides[1]] = s16 + s17;

        m5 = CRTM_11_2 * s5;
        m6 = CRTM_11_4 * s6;
        m7 = CRTM_11_6 * s7;
        m8 = CRTM_11_8 * s8;
        m9 = CRTM_11_10 * s9;

        s18 = m5 + m6;
        s19 = m7 + m8;
        s20 = s19 + m9;

        // Output point 3: X(2)
        out[out_strides[2]] = -(s18 + s20);

        m10 = CRTM_11_1 * s4;
        m11 = CRTM_11_3 * s0;
        m12 = CRTM_11_5 * s3;
        m13 = CRTM_11_7 * s1;
        m14 = CRTM_11_9 * s2;

        s21 = m10 + m11;
        s22 = m12 + m13;
        s23 = in0 - m14;
        s24 = s21 - s22;

        // Output point 4: X(3)
        out[out_strides[3]] = s23 + s24;

        m15 = CRTM_11_2 * s9;
        m16 = CRTM_11_4 * s5;
        m17 = CRTM_11_6 * s8;
        m18 = CRTM_11_8 * s6;
        m19 = CRTM_11_10 * s7;

        s25 = m15 - m16;
        s26 = m17 - m18;
        s27 = s26 + m19;

        // Output point 5: X(4)
        out[out_strides[4]] = s25 + s27;

        m20 = CRTM_11_1 * s3;
        m21 = CRTM_11_3 * s2;
        m22 = CRTM_11_5 * s0;
        m23 = CRTM_11_7 * s4;
        m24 = CRTM_11_9 * s1;

        s28 = m20 + m21;
        s29 = m22 + m23;
        s30 = in0 - m24;
        s31 = s30 - s29;

        // Output point 6: X(5)
        out[out_strides[5]] = s28 + s31;

        m25 = CRTM_11_2 * s8;
        m26 = CRTM_11_4 * s7;
        m27 = CRTM_11_6 * s5;
        m28 = CRTM_11_8 * s9;
        m29 = CRTM_11_10 * s6;

        s32 = m26 - m25;
        s33 = m27 + m28;
        s34 = s32 - s33;

        // Output point 7: X(6)
        out[out_strides[6]] = s34 + m29;

        m30 = CRTM_11_1 * s2;
        m31 = CRTM_11_3 * s4;
        m32 = CRTM_11_5 * s1;
        m33 = CRTM_11_7 * s0;
        m34 = CRTM_11_9 * s3;

        s35 = m30 + m31;
        s36 = m32 + m33;
        s37 = in0 - m34;
        s38 = s35 - s36;

        // Output point 8: X(7)
        out[out_strides[7]] = s38 + s37;

        m35 = CRTM_11_2 * s7;
        m36 = CRTM_11_4 * s9;
        m37 = CRTM_11_6 * s6;
        m38 = CRTM_11_8 * s5;
        m39 = CRTM_11_10 * s8;

        s39 = m36 - m35;
        s40 = m37 - m38;
        s41 = s40 - m39;

        // Output point 9: X(8)
        out[out_strides[8]] = s39 + s41;

        m40 = CRTM_11_1 * s1;
        m41 = CRTM_11_3 * s3;
        m42 = CRTM_11_5 * s4;
        m43 = CRTM_11_7 * s2;
        m44 = CRTM_11_9 * s0;

        s42 = m40 + m41;
        s43 = m42 + m43;
        s44 = in0 - m44;
        s45 = s42 - s43;

        // Output point 10: X(9)
        out[out_strides[9]] = s44 + s45;

        m45 = CRTM_11_2 * s6;
        m46 = CRTM_11_4 * s8;
        m47 = CRTM_11_6 * s9;
        m48 = CRTM_11_8 * s7;
        m49 = CRTM_11_10 * s5;

        s46 = m45 + m46;
        s47 = m47 + m48;
        s48 = s46 - s47;

        // Output point 11: X(10)
        out[out_strides[10]] = s48 - m49;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID rfft11avx128_fp32_bwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                  VOID *out_imag, INTP n,
                                  aoclfftz_strides_t *strides, VOID *twd,
                                  UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");

    const FLOAT CRTM_11_1 = 1.682507065662362337723623297838735435026584997f;
    const FLOAT CRTM_11_2 = 1.081281634911195164215271908637383390863541216f;
    const FLOAT CRTM_11_3 = 0.830830026003772851058548298459246407048009821f;
    const FLOAT CRTM_11_4 = 1.819263990709036742823430766158056920120482102f;
    const FLOAT CRTM_11_5 = 0.284629676546570280887585337232739337582102722f;
    const FLOAT CRTM_11_6 = 1.979642883761865464752184075553437574753038744f;
    const FLOAT CRTM_11_7 = 1.309721467890570128113850144932587106367582399f;
    const FLOAT CRTM_11_8 = 1.511499148708516567548071687944688840359434890f;
    const FLOAT CRTM_11_9 = 1.918985947228994779780736114132655398124909697f;
    const FLOAT CRTM_11_10 = 0.563465113682859395422835830693233798071555798f;
    const FLOAT CRTM_11_11 = 2.000000000000000000000000000000000000000000000f;

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
    INTP N = n / NUM_SETS_REAL_128_S;

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
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9, v_in10;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
               v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27,
               v_s28, v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36,
               v_s37, v_s38, v_s39, v_s40, v_s41, v_s42, v_s43, v_s44, v_s45,
               v_s46, v_s47, v_s48;
        __m128 v_m0, v_m1, v_m2, v_m3, v_m4, v_m5, v_m6, v_m7, v_m8, v_m9,
               v_m10, v_m11, v_m12, v_m13, v_m14, v_m15, v_m16, v_m17, v_m18,
               v_m19, v_m20, v_m21, v_m22, v_m23, v_m24, v_m25, v_m26, v_m27,
               v_m28, v_m29, v_m30, v_m31, v_m32, v_m33, v_m34, v_m35, v_m36,
               v_m37, v_m38, v_m39, v_m40, v_m41, v_m42, v_m43, v_m44, v_m45,
               v_m46, v_m47, v_m48, v_m49, v_m50;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10;

        curr_in = in;
        curr_out = out;

        // Input point 1: X(0)
        LDR_128_S(curr_in, v_in_stride, v_in0);
        // Input point 2: X(1) & Input point 3: X(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_S(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: X(3) & Input point 5: X(4)
        curr_in = in + in_strides[3];
        LDRI_2x128_S(curr_in, v_in_stride, v_in3, v_in4);
        // Input point 6: X(5) & Input point 7: X(6)
        curr_in = in + in_strides[5];
        LDRI_2x128_S(curr_in, v_in_stride, v_in5, v_in6);
        // Input point 8: X(7) & Input point 9: X(8)
        curr_in = in + in_strides[7];
        LDRI_2x128_S(curr_in, v_in_stride, v_in7, v_in8);
        // Input point 10: X(9) & Input point 11: X(10)
        curr_in = in + in_strides[9];
        LDRI_2x128_S(curr_in, v_in_stride, v_in9, v_in10);

        v_s0 = _mm_add_ps(v_in1, v_in3);
        v_s1 = _mm_add_ps(v_in5, v_in7);
        v_s2 = _mm_add_ps(v_s0, v_s1);
        v_s3 = _mm_add_ps(v_s2, v_in9);
        v_m0 = _mm_mul_ps(v_CRTM_11_11, v_s3);

        // Output point 1: x(0)
        v_out0 = _mm_add_ps(v_in0, v_m0);
        STR_128_S(curr_out, v_out_stride, v_out0);

        v_m1 = _mm_mul_ps(v_CRTM_11_1, v_in1);
        v_m2 = _mm_mul_ps(v_CRTM_11_2, v_in2);
        v_m3 = _mm_mul_ps(v_CRTM_11_3, v_in3);
        v_m4 = _mm_mul_ps(v_CRTM_11_4, v_in4);
        v_m5 = _mm_mul_ps(v_CRTM_11_5, v_in5);
        v_m6 = _mm_mul_ps(v_CRTM_11_6, v_in6);
        v_m7 = _mm_mul_ps(v_CRTM_11_7, v_in7);
        v_m8 = _mm_mul_ps(v_CRTM_11_8, v_in8);
        v_m9 = _mm_mul_ps(v_CRTM_11_9, v_in9);
        v_m10 = _mm_mul_ps(v_CRTM_11_10, v_in10);

        v_s4 = _mm_add_ps(v_m1, v_m3);
        v_s5 = _mm_add_ps(v_s4, v_in0);
        v_s6 = _mm_add_ps(v_m5, v_m7);
        v_s7 = _mm_add_ps(v_s6, v_m9);
        v_s8 = _mm_sub_ps(v_s5, v_s7);
        v_s9 = _mm_add_ps(v_m2, v_m4);
        v_s10 = _mm_add_ps(v_m6, v_m8);
        v_s11 = _mm_add_ps(v_s9, v_m10);
        v_s12 = _mm_add_ps(v_s10, v_s11);

        // Output point 2: x(1)
        v_out1 = _mm_sub_ps(v_s8, v_s12);
        curr_out = out + out_strides[1];
        STR_128_S(curr_out, v_out_stride, v_out1);
        // Output point 11: x(10)
        v_out10 = _mm_add_ps(v_s8, v_s12);
        curr_out = out + out_strides[10];
        STR_128_S(curr_out, v_out_stride, v_out10);

        v_m11 = _mm_mul_ps(v_CRTM_11_1, v_in9);
        v_m12 = _mm_mul_ps(v_CRTM_11_2, v_in10);
        v_m13 = _mm_mul_ps(v_CRTM_11_3, v_in1);
        v_m14 = _mm_mul_ps(v_CRTM_11_4, v_in2);
        v_m15 = _mm_mul_ps(v_CRTM_11_5, v_in7);
        v_m16 = _mm_mul_ps(v_CRTM_11_6, v_in8);
        v_m17 = _mm_mul_ps(v_CRTM_11_7, v_in3);
        v_m18 = _mm_mul_ps(v_CRTM_11_8, v_in4);
        v_m19 = _mm_mul_ps(v_CRTM_11_9, v_in5);
        v_m20 = _mm_mul_ps(v_CRTM_11_10, v_in6);

        v_s13 = _mm_add_ps(v_m11, v_m13);
        v_s14 = _mm_add_ps(v_s13, v_in0);
        v_s15 = _mm_add_ps(v_m15, v_m17);
        v_s16 = _mm_add_ps(v_s15, v_m19);
        v_s17 = _mm_sub_ps(v_s14, v_s16);
        v_s18 = _mm_sub_ps(v_m12, v_m14);
        v_s19 = _mm_sub_ps(v_m16, v_m18);
        v_s20 = _mm_add_ps(v_s19, v_m20);
        v_s21 = _mm_add_ps(v_s18, v_s20);

        // Output point 3: x(2)
        v_out2 = _mm_add_ps(v_s17, v_s21);
        curr_out = out + out_strides[2];
        STR_128_S(curr_out, v_out_stride, v_out2);
        // Output point 10: x(9)
        v_out9 = _mm_sub_ps(v_s17, v_s21);
        curr_out = out + out_strides[9];
        STR_128_S(curr_out, v_out_stride, v_out9);

        v_m21 = _mm_mul_ps(v_CRTM_11_1, v_in7);
        v_m22 = _mm_mul_ps(v_CRTM_11_2, v_in8);
        v_m23 = _mm_mul_ps(v_CRTM_11_3, v_in5);
        v_m24 = _mm_mul_ps(v_CRTM_11_4, v_in6);
        v_m25 = _mm_mul_ps(v_CRTM_11_5, v_in1);
        v_m26 = _mm_mul_ps(v_CRTM_11_6, v_in2);
        v_m27 = _mm_mul_ps(v_CRTM_11_7, v_in9);
        v_m28 = _mm_mul_ps(v_CRTM_11_8, v_in10);
        v_m29 = _mm_mul_ps(v_CRTM_11_9, v_in3);
        v_m30 = _mm_mul_ps(v_CRTM_11_10, v_in4);

        v_s22 = _mm_add_ps(v_m21, v_m23);
        v_s23 = _mm_add_ps(v_s22, v_in0);
        v_s24 = _mm_add_ps(v_m25, v_m27);
        v_s25 = _mm_add_ps(v_s24, v_m29);
        v_s26 = _mm_sub_ps(v_s23, v_s25);
        v_s27 = _mm_sub_ps(v_m22, v_m24);
        v_s28 = _mm_add_ps(v_m26, v_m28);
        v_s29 = _mm_sub_ps(v_s28, v_m30);
        v_s30 = _mm_add_ps(v_s27, v_s29);

        // Output point 4: x(3)
        v_out3 = _mm_sub_ps(v_s26, v_s30);
        curr_out = out + out_strides[3];
        STR_128_S(curr_out, v_out_stride, v_out3);
        // Output point 9: x(8)
        v_out8 = _mm_add_ps(v_s26, v_s30);
        curr_out = out + out_strides[8];
        STR_128_S(curr_out, v_out_stride, v_out8);

        v_m31 = _mm_mul_ps(v_CRTM_11_1, v_in5);
        v_m32 = _mm_mul_ps(v_CRTM_11_2, v_in6);
        v_m33 = _mm_mul_ps(v_CRTM_11_3, v_in9);
        v_m34 = _mm_mul_ps(v_CRTM_11_4, v_in10);
        v_m35 = _mm_mul_ps(v_CRTM_11_5, v_in3);
        v_m36 = _mm_mul_ps(v_CRTM_11_6, v_in4);
        v_m37 = _mm_mul_ps(v_CRTM_11_7, v_in1);
        v_m38 = _mm_mul_ps(v_CRTM_11_8, v_in2);
        v_m39 = _mm_mul_ps(v_CRTM_11_9, v_in7);
        v_m40 = _mm_mul_ps(v_CRTM_11_10, v_in8);

        v_s31 = _mm_add_ps(v_m31, v_m33);
        v_s32 = _mm_add_ps(v_s31, v_in0);
        v_s33 = _mm_add_ps(v_m35, v_m37);
        v_s34 = _mm_add_ps(v_s33, v_m39);
        v_s35 = _mm_sub_ps(v_s32, v_s34);
        v_s36 = _mm_sub_ps(v_m32, v_m34);
        v_s37 = _mm_sub_ps(v_m38, v_m36);
        v_s38 = _mm_add_ps(v_s37, v_m40);
        v_s39 = _mm_add_ps(v_s36, v_s38);

        // Output point 5: x(4)
        v_out4 = _mm_sub_ps(v_s35, v_s39);
        curr_out = out + out_strides[4];
        STR_128_S(curr_out, v_out_stride, v_out4);
        // Output point 8: x(7)
        v_out7 = _mm_add_ps(v_s35, v_s39);
        curr_out = out + out_strides[7];
        STR_128_S(curr_out, v_out_stride, v_out7);

        v_m41 = _mm_mul_ps(v_CRTM_11_1, v_in3);
        v_m42 = _mm_mul_ps(v_CRTM_11_2, v_in4);
        v_m43 = _mm_mul_ps(v_CRTM_11_3, v_in7);
        v_m44 = _mm_mul_ps(v_CRTM_11_4, v_in8);
        v_m45 = _mm_mul_ps(v_CRTM_11_5, v_in9);
        v_m46 = _mm_mul_ps(v_CRTM_11_6, v_in10);
        v_m47 = _mm_mul_ps(v_CRTM_11_7, v_in5);
        v_m48 = _mm_mul_ps(v_CRTM_11_8, v_in6);
        v_m49 = _mm_mul_ps(v_CRTM_11_9, v_in1);
        v_m50 = _mm_mul_ps(v_CRTM_11_10, v_in2);

        v_s40 = _mm_add_ps(v_m41, v_m43);
        v_s41 = _mm_add_ps(v_s40, v_in0);
        v_s42 = _mm_add_ps(v_m45, v_m47);
        v_s43 = _mm_add_ps(v_s42, v_m49);
        v_s44 = _mm_sub_ps(v_s41, v_s43);
        v_s45 = _mm_add_ps(v_m42, v_m44);
        v_s46 = _mm_add_ps(v_m46, v_m48);
        v_s47 = _mm_sub_ps(v_s45, v_s46);
        v_s48 = _mm_sub_ps(v_s47, v_m50);

        // Output point 6: x(5)
        v_out5 = _mm_add_ps(v_s44, v_s48);
        curr_out = out + out_strides[5];
        STR_128_S(curr_out, v_out_stride, v_out5);
        // Output point 7: x(6)
        v_out6 = _mm_sub_ps(v_s44, v_s48);
        curr_out = out + out_strides[6];
        STR_128_S(curr_out, v_out_stride, v_out6);

        in += v_in_stride * NUM_SETS_REAL_128_S;
        out += v_out_stride * NUM_SETS_REAL_128_S;
    }
    // tail cases
    if (n & 2)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9, v_in10;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
               v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27,
               v_s28, v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36,
               v_s37, v_s38, v_s39, v_s40, v_s41, v_s42, v_s43, v_s44, v_s45,
               v_s46, v_s47, v_s48;
        __m128 v_m0, v_m1, v_m2, v_m3, v_m4, v_m5, v_m6, v_m7, v_m8, v_m9,
               v_m10, v_m11, v_m12, v_m13, v_m14, v_m15, v_m16, v_m17, v_m18,
               v_m19, v_m20, v_m21, v_m22, v_m23, v_m24, v_m25, v_m26, v_m27,
               v_m28, v_m29, v_m30, v_m31, v_m32, v_m33, v_m34, v_m35, v_m36,
               v_m37, v_m38, v_m39, v_m40, v_m41, v_m42, v_m43, v_m44, v_m45,
               v_m46, v_m47, v_m48, v_m49, v_m50;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10;

        curr_in = in;
        curr_out = out;

        // Input point 1: X(0)
        LDHR_128_S(curr_in, v_in_stride, v_in0);
        // Input point 2: X(1) & Input point 3: X(2)
        curr_in = in + in_strides[1];
        LDHRI_2x128_S(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: X(3) & Input point 5: X(4)
        curr_in = in + in_strides[3];
        LDHRI_2x128_S(curr_in, v_in_stride, v_in3, v_in4);
        // Input point 6: X(5) & Input point 7: X(6)
        curr_in = in + in_strides[5];
        LDHRI_2x128_S(curr_in, v_in_stride, v_in5, v_in6);
        // Input point 8: X(7) & Input point 9: X(8)
        curr_in = in + in_strides[7];
        LDHRI_2x128_S(curr_in, v_in_stride, v_in7, v_in8);
        // Input point 10: X(9) & Input point 11: X(10)
        curr_in = in + in_strides[9];
        LDHRI_2x128_S(curr_in, v_in_stride, v_in9, v_in10);

        v_s0 = _mm_add_ps(v_in1, v_in3);
        v_s1 = _mm_add_ps(v_in5, v_in7);
        v_s2 = _mm_add_ps(v_s0, v_s1);
        v_s3 = _mm_add_ps(v_s2, v_in9);
        v_m0 = _mm_mul_ps(v_CRTM_11_11, v_s3);

        // Output point 1: x(0)
        v_out0 = _mm_add_ps(v_in0, v_m0);
        STHR_128_S(curr_out, v_out_stride, v_out0);

        v_m1 = _mm_mul_ps(v_CRTM_11_1, v_in1);
        v_m2 = _mm_mul_ps(v_CRTM_11_2, v_in2);
        v_m3 = _mm_mul_ps(v_CRTM_11_3, v_in3);
        v_m4 = _mm_mul_ps(v_CRTM_11_4, v_in4);
        v_m5 = _mm_mul_ps(v_CRTM_11_5, v_in5);
        v_m6 = _mm_mul_ps(v_CRTM_11_6, v_in6);
        v_m7 = _mm_mul_ps(v_CRTM_11_7, v_in7);
        v_m8 = _mm_mul_ps(v_CRTM_11_8, v_in8);
        v_m9 = _mm_mul_ps(v_CRTM_11_9, v_in9);
        v_m10 = _mm_mul_ps(v_CRTM_11_10, v_in10);

        v_s4 = _mm_add_ps(v_m1, v_m3);
        v_s5 = _mm_add_ps(v_s4, v_in0);
        v_s6 = _mm_add_ps(v_m5, v_m7);
        v_s7 = _mm_add_ps(v_s6, v_m9);
        v_s8 = _mm_sub_ps(v_s5, v_s7);
        v_s9 = _mm_add_ps(v_m2, v_m4);
        v_s10 = _mm_add_ps(v_m6, v_m8);
        v_s11 = _mm_add_ps(v_s9, v_m10);
        v_s12 = _mm_add_ps(v_s10, v_s11);

        // Output point 2: x(1)
        v_out1 = _mm_sub_ps(v_s8, v_s12);
        curr_out = out + out_strides[1];
        STHR_128_S(curr_out, v_out_stride, v_out1);
        // Output point 11: x(10)
        v_out10 = _mm_add_ps(v_s8, v_s12);
        curr_out = out + out_strides[10];
        STHR_128_S(curr_out, v_out_stride, v_out10);

        v_m11 = _mm_mul_ps(v_CRTM_11_1, v_in9);
        v_m12 = _mm_mul_ps(v_CRTM_11_2, v_in10);
        v_m13 = _mm_mul_ps(v_CRTM_11_3, v_in1);
        v_m14 = _mm_mul_ps(v_CRTM_11_4, v_in2);
        v_m15 = _mm_mul_ps(v_CRTM_11_5, v_in7);
        v_m16 = _mm_mul_ps(v_CRTM_11_6, v_in8);
        v_m17 = _mm_mul_ps(v_CRTM_11_7, v_in3);
        v_m18 = _mm_mul_ps(v_CRTM_11_8, v_in4);
        v_m19 = _mm_mul_ps(v_CRTM_11_9, v_in5);
        v_m20 = _mm_mul_ps(v_CRTM_11_10, v_in6);

        v_s13 = _mm_add_ps(v_m11, v_m13);
        v_s14 = _mm_add_ps(v_s13, v_in0);
        v_s15 = _mm_add_ps(v_m15, v_m17);
        v_s16 = _mm_add_ps(v_s15, v_m19);
        v_s17 = _mm_sub_ps(v_s14, v_s16);
        v_s18 = _mm_sub_ps(v_m12, v_m14);
        v_s19 = _mm_sub_ps(v_m16, v_m18);
        v_s20 = _mm_add_ps(v_s19, v_m20);
        v_s21 = _mm_add_ps(v_s18, v_s20);

        // Output point 3: x(2)
        v_out2 = _mm_add_ps(v_s17, v_s21);
        curr_out = out + out_strides[2];
        STHR_128_S(curr_out, v_out_stride, v_out2);
        // Output point 10: x(9)
        v_out9 = _mm_sub_ps(v_s17, v_s21);
        curr_out = out + out_strides[9];
        STHR_128_S(curr_out, v_out_stride, v_out9);

        v_m21 = _mm_mul_ps(v_CRTM_11_1, v_in7);
        v_m22 = _mm_mul_ps(v_CRTM_11_2, v_in8);
        v_m23 = _mm_mul_ps(v_CRTM_11_3, v_in5);
        v_m24 = _mm_mul_ps(v_CRTM_11_4, v_in6);
        v_m25 = _mm_mul_ps(v_CRTM_11_5, v_in1);
        v_m26 = _mm_mul_ps(v_CRTM_11_6, v_in2);
        v_m27 = _mm_mul_ps(v_CRTM_11_7, v_in9);
        v_m28 = _mm_mul_ps(v_CRTM_11_8, v_in10);
        v_m29 = _mm_mul_ps(v_CRTM_11_9, v_in3);
        v_m30 = _mm_mul_ps(v_CRTM_11_10, v_in4);

        v_s22 = _mm_add_ps(v_m21, v_m23);
        v_s23 = _mm_add_ps(v_s22, v_in0);
        v_s24 = _mm_add_ps(v_m25, v_m27);
        v_s25 = _mm_add_ps(v_s24, v_m29);
        v_s26 = _mm_sub_ps(v_s23, v_s25);
        v_s27 = _mm_sub_ps(v_m22, v_m24);
        v_s28 = _mm_add_ps(v_m26, v_m28);
        v_s29 = _mm_sub_ps(v_s28, v_m30);
        v_s30 = _mm_add_ps(v_s27, v_s29);

        // Output point 4: x(3)
        v_out3 = _mm_sub_ps(v_s26, v_s30);
        curr_out = out + out_strides[3];
        STHR_128_S(curr_out, v_out_stride, v_out3);
        // Output point 9: x(8)
        v_out8 = _mm_add_ps(v_s26, v_s30);
        curr_out = out + out_strides[8];
        STHR_128_S(curr_out, v_out_stride, v_out8);

        v_m31 = _mm_mul_ps(v_CRTM_11_1, v_in5);
        v_m32 = _mm_mul_ps(v_CRTM_11_2, v_in6);
        v_m33 = _mm_mul_ps(v_CRTM_11_3, v_in9);
        v_m34 = _mm_mul_ps(v_CRTM_11_4, v_in10);
        v_m35 = _mm_mul_ps(v_CRTM_11_5, v_in3);
        v_m36 = _mm_mul_ps(v_CRTM_11_6, v_in4);
        v_m37 = _mm_mul_ps(v_CRTM_11_7, v_in1);
        v_m38 = _mm_mul_ps(v_CRTM_11_8, v_in2);
        v_m39 = _mm_mul_ps(v_CRTM_11_9, v_in7);
        v_m40 = _mm_mul_ps(v_CRTM_11_10, v_in8);

        v_s31 = _mm_add_ps(v_m31, v_m33);
        v_s32 = _mm_add_ps(v_s31, v_in0);
        v_s33 = _mm_add_ps(v_m35, v_m37);
        v_s34 = _mm_add_ps(v_s33, v_m39);
        v_s35 = _mm_sub_ps(v_s32, v_s34);
        v_s36 = _mm_sub_ps(v_m32, v_m34);
        v_s37 = _mm_sub_ps(v_m38, v_m36);
        v_s38 = _mm_add_ps(v_s37, v_m40);
        v_s39 = _mm_add_ps(v_s36, v_s38);

        // Output point 5: x(4)
        v_out4 = _mm_sub_ps(v_s35, v_s39);
        curr_out = out + out_strides[4];
        STHR_128_S(curr_out, v_out_stride, v_out4);
        // Output point 8: x(7)
        v_out7 = _mm_add_ps(v_s35, v_s39);
        curr_out = out + out_strides[7];
        STHR_128_S(curr_out, v_out_stride, v_out7);

        v_m41 = _mm_mul_ps(v_CRTM_11_1, v_in3);
        v_m42 = _mm_mul_ps(v_CRTM_11_2, v_in4);
        v_m43 = _mm_mul_ps(v_CRTM_11_3, v_in7);
        v_m44 = _mm_mul_ps(v_CRTM_11_4, v_in8);
        v_m45 = _mm_mul_ps(v_CRTM_11_5, v_in9);
        v_m46 = _mm_mul_ps(v_CRTM_11_6, v_in10);
        v_m47 = _mm_mul_ps(v_CRTM_11_7, v_in5);
        v_m48 = _mm_mul_ps(v_CRTM_11_8, v_in6);
        v_m49 = _mm_mul_ps(v_CRTM_11_9, v_in1);
        v_m50 = _mm_mul_ps(v_CRTM_11_10, v_in2);

        v_s40 = _mm_add_ps(v_m41, v_m43);
        v_s41 = _mm_add_ps(v_s40, v_in0);
        v_s42 = _mm_add_ps(v_m45, v_m47);
        v_s43 = _mm_add_ps(v_s42, v_m49);
        v_s44 = _mm_sub_ps(v_s41, v_s43);
        v_s45 = _mm_add_ps(v_m42, v_m44);
        v_s46 = _mm_add_ps(v_m46, v_m48);
        v_s47 = _mm_sub_ps(v_s45, v_s46);
        v_s48 = _mm_sub_ps(v_s47, v_m50);

        // Output point 6: x(5)
        v_out5 = _mm_add_ps(v_s44, v_s48);
        curr_out = out + out_strides[5];
        STHR_128_S(curr_out, v_out_stride, v_out5);
        // Output point 7: x(6)
        v_out6 = _mm_sub_ps(v_s44, v_s48);
        curr_out = out + out_strides[6];
        STHR_128_S(curr_out, v_out_stride, v_out6);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        FLOAT in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10;
        FLOAT s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14,
              s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27,
              s28, s29, s30, s31, s32, s33, s34, s35, s36, s37, s38, s39, s40,
              s41, s42, s43, s44, s45, s46, s47, s48;
        FLOAT m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14,
              m15, m16, m17, m18, m19, m20, m21, m22, m23, m24, m25, m26, m27,
              m28, m29, m30, m31, m32, m33, m34, m35, m36, m37, m38, m39, m40,
              m41, m42, m43, m44, m45, m46, m47, m48, m49, m50;

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
        // Input point 10: X(9)
        in9 = in[in_strides[9]];
        // Input point 11: X(10)
        in10 = in[in_strides[10]];

        s0 = in1 + in3;
        s1 = in5 + in7;
        s2 = s0 + s1;
        s3 = s2 + in9;
        m0 = CRTM_11_11 * s3;

        // Output point 1: x(0)
        *out = in0 + m0;

        m1 = CRTM_11_1 * in1;
        m2 = CRTM_11_2 * in2;
        m3 = CRTM_11_3 * in3;
        m4 = CRTM_11_4 * in4;
        m5 = CRTM_11_5 * in5;
        m6 = CRTM_11_6 * in6;
        m7 = CRTM_11_7 * in7;
        m8 = CRTM_11_8 * in8;
        m9 = CRTM_11_9 * in9;
        m10 = CRTM_11_10 * in10;

        s4 = m1 + m3;
        s5 = s4 + in0;
        s6 = m5 + m7;
        s7 = s6 + m9;
        s8 = s5 - s7;

        s9 = m2 + m4;
        s10 = m6 + m8;
        s11 = s9 + m10;
        s12 = s10 + s11;

        // Output point 2: x(1)
        out[out_strides[1]] = s8 - s12;
        // Output point 11: x(10)
        out[out_strides[10]] = s8 + s12;

        m11 = CRTM_11_1 * in9;
        m12 = CRTM_11_2 * in10;
        m13 = CRTM_11_3 * in1;
        m14 = CRTM_11_4 * in2;
        m15 = CRTM_11_5 * in7;
        m16 = CRTM_11_6 * in8;
        m17 = CRTM_11_7 * in3;
        m18 = CRTM_11_8 * in4;
        m19 = CRTM_11_9 * in5;
        m20 = CRTM_11_10 * in6;

        s13 = m11 + m13;
        s14 = s13 + in0;
        s15 = m15 + m17;
        s16 = s15 + m19;
        s17 = s14 - s16;

        s18 = m12 - m14;
        s19 = m16 - m18;
        s20 = s19 + m20;
        s21 = s18 + s20;

        // Output point 3: x(2)
        out[out_strides[2]] = s17 + s21;
        // Output point 10: x(9)
        out[out_strides[9]] = s17 - s21;

        m21 = CRTM_11_1 * in7;
        m22 = CRTM_11_2 * in8;
        m23 = CRTM_11_3 * in5;
        m24 = CRTM_11_4 * in6;
        m25 = CRTM_11_5 * in1;
        m26 = CRTM_11_6 * in2;
        m27 = CRTM_11_7 * in9;
        m28 = CRTM_11_8 * in10;
        m29 = CRTM_11_9 * in3;
        m30 = CRTM_11_10 * in4;

        s22 = m21 + m23;
        s23 = s22 + in0;
        s24 = m25 + m27;
        s25 = s24 + m29;
        s26 = s23 - s25;

        s27 = m22 - m24;
        s28 = m26 + m28;
        s29 = s28 - m30;
        s30 = s27 + s29;

        // Output point 4: x(3)
        out[out_strides[3]] = s26 - s30;
        // Output point 9: x(8)
        out[out_strides[8]] = s26 + s30;

        m31 = CRTM_11_1 * in5;
        m32 = CRTM_11_2 * in6;
        m33 = CRTM_11_3 * in9;
        m34 = CRTM_11_4 * in10;
        m35 = CRTM_11_5 * in3;
        m36 = CRTM_11_6 * in4;
        m37 = CRTM_11_7 * in1;
        m38 = CRTM_11_8 * in2;
        m39 = CRTM_11_9 * in7;
        m40 = CRTM_11_10 * in8;

        s31 = m31 + m33;
        s32 = s31 + in0;
        s33 = m35 + m37;
        s34 = s33 + m39;
        s35 = s32 - s34;

        s36 = m32 - m34;
        s37 = m38 - m36;
        s38 = s37 + m40;
        s39 = s36 + s38;

        // Output point 5: x(4)
        out[out_strides[4]] = s35 - s39;
        // Output point 8: x(7)
        out[out_strides[7]] = s35 + s39;

        m41 = CRTM_11_1 * in3;
        m42 = CRTM_11_2 * in4;
        m43 = CRTM_11_3 * in7;
        m44 = CRTM_11_4 * in8;
        m45 = CRTM_11_5 * in9;
        m46 = CRTM_11_6 * in10;
        m47 = CRTM_11_7 * in5;
        m48 = CRTM_11_8 * in6;
        m49 = CRTM_11_9 * in1;
        m50 = CRTM_11_10 * in2;

        s40 = m41 + m43;
        s41 = s40 + in0;
        s42 = m45 + m47;
        s43 = s42 + m49;
        s44 = s41 - s43;

        s45 = m42 + m44;
        s46 = m46 + m48;
        s47 = s45 - s46;
        s48 = s47 - m50;

        // Output point 6: x(5)
        out[out_strides[5]] = s44 + s48;
        // Output point 7: x(6)
        out[out_strides[6]] = s44 - s48;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID rfft11avx128_fp64_fwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                  VOID *out_imag, INTP n,
                                  aoclfftz_strides_t *strides, VOID *twd,
                                  UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");

    const DOUBLE CRTM_11_1 = 0.841253532831181168861811648919367717513292498;
    const DOUBLE CRTM_11_2 = 0.540640817455597582107635954318691695431770608;
    const DOUBLE CRTM_11_3 = 0.415415013001886425529274149229623203524004910;
    const DOUBLE CRTM_11_4 = 0.909631995354518371411715383079028460060241051;
    const DOUBLE CRTM_11_5 = 0.142314838273285140443792668616369668791051361;
    const DOUBLE CRTM_11_6 = 0.989821441880932732376092037776718787376519372;
    const DOUBLE CRTM_11_7 = 0.654860733945285064056925072466293553183791199;
    const DOUBLE CRTM_11_8 = 0.755749574354258283774035843972344420179717445;
    const DOUBLE CRTM_11_9 = 0.959492973614497389890368057066327699062454848;
    const DOUBLE CRTM_11_10 = 0.281732556841429697711417915346616899035777899;

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
    INTP N = n / NUM_SETS_REAL_128_D;

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
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
                v_in9, v_in10;
        __m128d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
                v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
                v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27,
                v_s28, v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36,
                v_s37, v_s38, v_s39, v_s40, v_s41, v_s42, v_s43, v_s44, v_s45,
                v_s46, v_s47, v_s48;
        __m128d v_m0, v_m1, v_m2, v_m3, v_m4, v_m5, v_m6, v_m7, v_m8, v_m9,
                v_m10, v_m11, v_m12, v_m13, v_m14, v_m15, v_m16, v_m17, v_m18,
                v_m19, v_m20, v_m21, v_m22, v_m23, v_m24, v_m25, v_m26, v_m27,
                v_m28, v_m29, v_m30, v_m31, v_m32, v_m33, v_m34, v_m35, v_m36,
                v_m37, v_m38, v_m39, v_m40, v_m41, v_m42, v_m43, v_m44, v_m45,
                v_m46, v_m47, v_m48, v_m49;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_128_D(curr_in, v_in_stride, v_in1);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_128_D(curr_in, v_in_stride, v_in2);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_128_D(curr_in, v_in_stride, v_in3);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_128_D(curr_in, v_in_stride, v_in4);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_128_D(curr_in, v_in_stride, v_in5);
        // Input point 7: x(6)
        curr_in = in + in_strides[6];
        LDR_128_D(curr_in, v_in_stride, v_in6);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_128_D(curr_in, v_in_stride, v_in7);
        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDR_128_D(curr_in, v_in_stride, v_in8);
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDR_128_D(curr_in, v_in_stride, v_in9);
        // Input point 11: x(10)
        curr_in = in + in_strides[10];
        LDR_128_D(curr_in, v_in_stride, v_in10);

        v_s0 = _mm_add_pd(v_in1, v_in10);
        v_s1 = _mm_add_pd(v_in2, v_in9);
        v_s2 = _mm_add_pd(v_in3, v_in8);
        v_s3 = _mm_add_pd(v_in4, v_in7);
        v_s4 = _mm_add_pd(v_in5, v_in6);
        v_s5 = _mm_sub_pd(v_in1, v_in10);
        v_s6 = _mm_sub_pd(v_in2, v_in9);
        v_s7 = _mm_sub_pd(v_in3, v_in8);
        v_s8 = _mm_sub_pd(v_in4, v_in7);
        v_s9 = _mm_sub_pd(v_in5, v_in6);

        v_s10 = _mm_add_pd(v_s0, v_s1);
        v_s11 = _mm_add_pd(v_s2, v_s3);
        v_s12 = _mm_add_pd(v_s4, v_in0);
        v_s13 = _mm_add_pd(v_s10, v_s11);

        // Output point 1: X(0)
        v_out0 = _mm_add_pd(v_s12, v_s13);
        STR_128_D(curr_out, v_out_stride, v_out0);

        v_m0 = _mm_mul_pd(v_CRTM_11_1, v_s0);
        v_m1 = _mm_mul_pd(v_CRTM_11_3, v_s1);
        v_m2 = _mm_mul_pd(v_CRTM_11_5, v_s2);
        v_m3 = _mm_mul_pd(v_CRTM_11_7, v_s3);
        v_m4 = _mm_mul_pd(v_CRTM_11_9, v_s4);

        v_s14 = _mm_add_pd(v_m0, v_m1);
        v_s15 = _mm_add_pd(v_m2, v_m3);
        v_s16 = _mm_sub_pd(v_in0, v_m4);
        v_s17 = _mm_sub_pd(v_s14, v_s15);

        // Output point 2: X(1)
        v_out1 = _mm_add_pd(v_s16, v_s17);

        v_m5 = _mm_mul_pd(v_CRTM_11_2, v_s5);
        v_m6 = _mm_mul_pd(v_CRTM_11_4, v_s6);
        v_m7 = _mm_mul_pd(v_CRTM_11_6, v_s7);
        v_m8 = _mm_mul_pd(v_CRTM_11_8, v_s8);
        v_m9 = _mm_mul_pd(v_CRTM_11_10, v_s9);

        v_s18 = _mm_add_pd(v_m5, v_m6);
        v_s19 = _mm_add_pd(v_m7, v_m8);
        v_s20 = _mm_add_pd(v_s19, v_m9);

        // Output point 3: X(2)
        v_out2 = NEGATE_128_D(_mm_add_pd(v_s18, v_s20));
        curr_out = out + out_strides[1];
        STRI_2x128_D(curr_out, v_out_stride, v_out1, v_out2);

        v_m10 = _mm_mul_pd(v_CRTM_11_1, v_s4);
        v_m11 = _mm_mul_pd(v_CRTM_11_3, v_s0);
        v_m12 = _mm_mul_pd(v_CRTM_11_5, v_s3);
        v_m13 = _mm_mul_pd(v_CRTM_11_7, v_s1);
        v_m14 = _mm_mul_pd(v_CRTM_11_9, v_s2);

        v_s21 = _mm_add_pd(v_m10, v_m11);
        v_s22 = _mm_add_pd(v_m12, v_m13);
        v_s23 = _mm_sub_pd(v_in0, v_m14);
        v_s24 = _mm_sub_pd(v_s21, v_s22);

        // Output point 4: X(3)
        v_out3 = _mm_add_pd(v_s23, v_s24);

        v_m15 = _mm_mul_pd(v_CRTM_11_2, v_s9);
        v_m16 = _mm_mul_pd(v_CRTM_11_4, v_s5);
        v_m17 = _mm_mul_pd(v_CRTM_11_6, v_s8);
        v_m18 = _mm_mul_pd(v_CRTM_11_8, v_s6);
        v_m19 = _mm_mul_pd(v_CRTM_11_10, v_s7);

        v_s25 = _mm_sub_pd(v_m15, v_m16);
        v_s26 = _mm_sub_pd(v_m17, v_m18);
        v_s27 = _mm_add_pd(v_s26, v_m19);

        // Output point 5: X(4)
        v_out4 = _mm_add_pd(v_s25, v_s27);
        curr_out = out + out_strides[3];
        STRI_2x128_D(curr_out, v_out_stride, v_out3, v_out4);

        v_m20 = _mm_mul_pd(v_CRTM_11_1, v_s3);
        v_m21 = _mm_mul_pd(v_CRTM_11_3, v_s2);
        v_m22 = _mm_mul_pd(v_CRTM_11_5, v_s0);
        v_m23 = _mm_mul_pd(v_CRTM_11_7, v_s4);
        v_m24 = _mm_mul_pd(v_CRTM_11_9, v_s1);

        v_s28 = _mm_add_pd(v_m20, v_m21);
        v_s29 = _mm_add_pd(v_m22, v_m23);
        v_s30 = _mm_sub_pd(v_in0, v_m24);
        v_s31 = _mm_sub_pd(v_s30, v_s29);

        // Output point 6: X(5)
        v_out5 = _mm_add_pd(v_s28, v_s31);

        v_m25 = _mm_mul_pd(v_CRTM_11_2, v_s8);
        v_m26 = _mm_mul_pd(v_CRTM_11_4, v_s7);
        v_m27 = _mm_mul_pd(v_CRTM_11_6, v_s5);
        v_m28 = _mm_mul_pd(v_CRTM_11_8, v_s9);
        v_m29 = _mm_mul_pd(v_CRTM_11_10, v_s6);

        v_s32 = _mm_sub_pd(v_m26, v_m25);
        v_s33 = _mm_add_pd(v_m27, v_m28);
        v_s34 = _mm_sub_pd(v_s32, v_s33);

        // Output point 7: X(6)
        v_out6 = _mm_add_pd(v_s34, v_m29);
        curr_out = out + out_strides[5];
        STRI_2x128_D(curr_out, v_out_stride, v_out5, v_out6);

        v_m30 = _mm_mul_pd(v_CRTM_11_1, v_s2);
        v_m31 = _mm_mul_pd(v_CRTM_11_3, v_s4);
        v_m32 = _mm_mul_pd(v_CRTM_11_5, v_s1);
        v_m33 = _mm_mul_pd(v_CRTM_11_7, v_s0);
        v_m34 = _mm_mul_pd(v_CRTM_11_9, v_s3);

        v_s35 = _mm_add_pd(v_m30, v_m31);
        v_s36 = _mm_add_pd(v_m32, v_m33);
        v_s37 = _mm_sub_pd(v_in0, v_m34);
        v_s38 = _mm_sub_pd(v_s35, v_s36);

        // Output point 8: X(7)
        v_out7 = _mm_add_pd(v_s38, v_s37);

        v_m35 = _mm_mul_pd(v_CRTM_11_2, v_s7);
        v_m36 = _mm_mul_pd(v_CRTM_11_4, v_s9);
        v_m37 = _mm_mul_pd(v_CRTM_11_6, v_s6);
        v_m38 = _mm_mul_pd(v_CRTM_11_8, v_s5);
        v_m39 = _mm_mul_pd(v_CRTM_11_10, v_s8);

        v_s39 = _mm_sub_pd(v_m36, v_m35);
        v_s40 = _mm_sub_pd(v_m37, v_m38);
        v_s41 = _mm_sub_pd(v_s40, v_m39);

        // Output point 9: X(8)
        v_out8 = _mm_add_pd(v_s39, v_s41);
        curr_out = out + out_strides[7];
        STRI_2x128_D(curr_out, v_out_stride, v_out7, v_out8);

        v_m40 = _mm_mul_pd(v_CRTM_11_1, v_s1);
        v_m41 = _mm_mul_pd(v_CRTM_11_3, v_s3);
        v_m42 = _mm_mul_pd(v_CRTM_11_5, v_s4);
        v_m43 = _mm_mul_pd(v_CRTM_11_7, v_s2);
        v_m44 = _mm_mul_pd(v_CRTM_11_9, v_s0);

        v_s42 = _mm_add_pd(v_m40, v_m41);
        v_s43 = _mm_add_pd(v_m42, v_m43);
        v_s44 = _mm_sub_pd(v_in0, v_m44);
        v_s45 = _mm_sub_pd(v_s42, v_s43);

        // Output point 10: X(9)
        v_out9 = _mm_add_pd(v_s44, v_s45);
        curr_out = out + out_strides[9];
        STR_128_D(curr_out, v_out_stride, v_out9);

        v_m45 = _mm_mul_pd(v_CRTM_11_2, v_s6);
        v_m46 = _mm_mul_pd(v_CRTM_11_4, v_s8);
        v_m47 = _mm_mul_pd(v_CRTM_11_6, v_s9);
        v_m48 = _mm_mul_pd(v_CRTM_11_8, v_s7);
        v_m49 = _mm_mul_pd(v_CRTM_11_10, v_s5);

        v_s46 = _mm_add_pd(v_m45, v_m46);
        v_s47 = _mm_add_pd(v_m47, v_m48);
        v_s48 = _mm_sub_pd(v_s46, v_s47);

        // Output point 11: X(10)
        v_out10 = _mm_sub_pd(v_s48, v_m49);
        curr_out = out + out_strides[10];
        STR_128_D(curr_out, v_out_stride, v_out10);

        in += v_in_stride * NUM_SETS_REAL_128_D;
        out += v_out_stride * NUM_SETS_REAL_128_D;
    }
    // tail cases
    if (n & 1)
    {
        DOUBLE in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10;
        DOUBLE s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14,
               s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27,
               s28, s29, s30, s31, s32, s33, s34, s35, s36, s37, s38, s39, s40,
               s41, s42, s43, s44, s45, s46, s47, s48;
        DOUBLE m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14,
               m15, m16, m17, m18, m19, m20, m21, m22, m23, m24, m25, m26, m27,
               m28, m29, m30, m31, m32, m33, m34, m35, m36, m37, m38, m39, m40,
               m41, m42, m43, m44, m45, m46, m47, m48, m49;

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
        // Input point 10: x(9)
        in9 = in[in_strides[9]];
        // Input point 11: x(10)
        in10 = in[in_strides[10]];

        s0 = in1 + in10;
        s1 = in2 + in9;
        s2 = in3 + in8;
        s3 = in4 + in7;
        s4 = in5 + in6;
        s5 = in1 - in10;
        s6 = in2 - in9;
        s7 = in3 - in8;
        s8 = in4 - in7;
        s9 = in5 - in6;

        s10 = s0 + s1;
        s11 = s2 + s3;
        s12 = s4 + in0;
        s13 = s10 + s11;

        // Output point 1: X(0)
        *out = s12 + s13;

        m0 = CRTM_11_1 * s0;
        m1 = CRTM_11_3 * s1;
        m2 = CRTM_11_5 * s2;
        m3 = CRTM_11_7 * s3;
        m4 = CRTM_11_9 * s4;

        s14 = m0 + m1;
        s15 = m2 + m3;
        s16 = in0 - m4;
        s17 = s14 - s15;

        // Output point 2: X(1)
        out[out_strides[1]] = s16 + s17;

        m5 = CRTM_11_2 * s5;
        m6 = CRTM_11_4 * s6;
        m7 = CRTM_11_6 * s7;
        m8 = CRTM_11_8 * s8;
        m9 = CRTM_11_10 * s9;

        s18 = m5 + m6;
        s19 = m7 + m8;
        s20 = s19 + m9;

        // Output point 3: X(2)
        out[out_strides[2]] = -(s18 + s20);

        m10 = CRTM_11_1 * s4;
        m11 = CRTM_11_3 * s0;
        m12 = CRTM_11_5 * s3;
        m13 = CRTM_11_7 * s1;
        m14 = CRTM_11_9 * s2;

        s21 = m10 + m11;
        s22 = m12 + m13;
        s23 = in0 - m14;
        s24 = s21 - s22;

        // Output point 4: X(3)
        out[out_strides[3]] = s23 + s24;

        m15 = CRTM_11_2 * s9;
        m16 = CRTM_11_4 * s5;
        m17 = CRTM_11_6 * s8;
        m18 = CRTM_11_8 * s6;
        m19 = CRTM_11_10 * s7;

        s25 = m15 - m16;
        s26 = m17 - m18;
        s27 = s26 + m19;

        // Output point 5: X(4)
        out[out_strides[4]] = s25 + s27;

        m20 = CRTM_11_1 * s3;
        m21 = CRTM_11_3 * s2;
        m22 = CRTM_11_5 * s0;
        m23 = CRTM_11_7 * s4;
        m24 = CRTM_11_9 * s1;

        s28 = m20 + m21;
        s29 = m22 + m23;
        s30 = in0 - m24;
        s31 = s30 - s29;

        // Output point 6: X(5)
        out[out_strides[5]] = s28 + s31;

        m25 = CRTM_11_2 * s8;
        m26 = CRTM_11_4 * s7;
        m27 = CRTM_11_6 * s5;
        m28 = CRTM_11_8 * s9;
        m29 = CRTM_11_10 * s6;

        s32 = m26 - m25;
        s33 = m27 + m28;
        s34 = s32 - s33;

        // Output point 7: X(6)
        out[out_strides[6]] = s34 + m29;

        m30 = CRTM_11_1 * s2;
        m31 = CRTM_11_3 * s4;
        m32 = CRTM_11_5 * s1;
        m33 = CRTM_11_7 * s0;
        m34 = CRTM_11_9 * s3;

        s35 = m30 + m31;
        s36 = m32 + m33;
        s37 = in0 - m34;
        s38 = s35 - s36;

        // Output point 8: X(7)
        out[out_strides[7]] = s38 + s37;

        m35 = CRTM_11_2 * s7;
        m36 = CRTM_11_4 * s9;
        m37 = CRTM_11_6 * s6;
        m38 = CRTM_11_8 * s5;
        m39 = CRTM_11_10 * s8;

        s39 = m36 - m35;
        s40 = m37 - m38;
        s41 = s40 - m39;

        // Output point 9: X(8)
        out[out_strides[8]] = s39 + s41;

        m40 = CRTM_11_1 * s1;
        m41 = CRTM_11_3 * s3;
        m42 = CRTM_11_5 * s4;
        m43 = CRTM_11_7 * s2;
        m44 = CRTM_11_9 * s0;

        s42 = m40 + m41;
        s43 = m42 + m43;
        s44 = in0 - m44;
        s45 = s42 - s43;

        // Output point 10: X(9)
        out[out_strides[9]] = s44 + s45;

        m45 = CRTM_11_2 * s6;
        m46 = CRTM_11_4 * s8;
        m47 = CRTM_11_6 * s9;
        m48 = CRTM_11_8 * s7;
        m49 = CRTM_11_10 * s5;

        s46 = m45 + m46;
        s47 = m47 + m48;
        s48 = s46 - s47;

        // Output point 11: X(10)
        out[out_strides[10]] = s48 - m49;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID rfft11avx128_fp64_bwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                                  VOID *out_imag, INTP n,
                                  aoclfftz_strides_t *strides, VOID *twd,
                                  UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");

    const DOUBLE CRTM_11_1 = 1.682507065662362337723623297838735435026584997;
    const DOUBLE CRTM_11_2 = 1.081281634911195164215271908637383390863541216;
    const DOUBLE CRTM_11_3 = 0.830830026003772851058548298459246407048009821;
    const DOUBLE CRTM_11_4 = 1.819263990709036742823430766158056920120482102;
    const DOUBLE CRTM_11_5 = 0.284629676546570280887585337232739337582102722;
    const DOUBLE CRTM_11_6 = 1.979642883761865464752184075553437574753038744;
    const DOUBLE CRTM_11_7 = 1.309721467890570128113850144932587106367582399;
    const DOUBLE CRTM_11_8 = 1.511499148708516567548071687944688840359434890;
    const DOUBLE CRTM_11_9 = 1.918985947228994779780736114132655398124909697;
    const DOUBLE CRTM_11_10 = 0.563465113682859395422835830693233798071555798;
    const DOUBLE CRTM_11_11 = 2.000000000000000000000000000000000000000000000;

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
    INTP N = n / NUM_SETS_REAL_128_D;

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
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
                v_in9, v_in10;
        __m128d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
                v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
                v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27,
                v_s28, v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36,
                v_s37, v_s38, v_s39, v_s40, v_s41, v_s42, v_s43, v_s44, v_s45,
                v_s46, v_s47, v_s48;
        __m128d v_m0, v_m1, v_m2, v_m3, v_m4, v_m5, v_m6, v_m7, v_m8, v_m9,
                v_m10, v_m11, v_m12, v_m13, v_m14, v_m15, v_m16, v_m17, v_m18,
                v_m19, v_m20, v_m21, v_m22, v_m23, v_m24, v_m25, v_m26, v_m27,
                v_m28, v_m29, v_m30, v_m31, v_m32, v_m33, v_m34, v_m35, v_m36,
                v_m37, v_m38, v_m39, v_m40, v_m41, v_m42, v_m43, v_m44, v_m45,
                v_m46, v_m47, v_m48, v_m49, v_m50;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10;

        curr_in = in;
        curr_out = out;

        // Input point 1: X(0)
        LDR_128_D(curr_in, v_in_stride, v_in0);
        // Input point 2: X(1) & Input point 3: X(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_D(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: X(3) & Input point 5: X(4)
        curr_in = in + in_strides[3];
        LDRI_2x128_D(curr_in, v_in_stride, v_in3, v_in4);
        // Input point 6: X(5) & Input point 7: X(6)
        curr_in = in + in_strides[5];
        LDRI_2x128_D(curr_in, v_in_stride, v_in5, v_in6);
        // Input point 8: X(7) & Input point 9: X(8)
        curr_in = in + in_strides[7];
        LDRI_2x128_D(curr_in, v_in_stride, v_in7, v_in8);
        // Input point 10: X(9) & Input point 11: X(10)
        curr_in = in + in_strides[9];
        LDRI_2x128_D(curr_in, v_in_stride, v_in9, v_in10);

        v_s0 = _mm_add_pd(v_in1, v_in3);
        v_s1 = _mm_add_pd(v_in5, v_in7);
        v_s2 = _mm_add_pd(v_s0, v_s1);
        v_s3 = _mm_add_pd(v_s2, v_in9);
        v_m0 = _mm_mul_pd(v_CRTM_11_11, v_s3);

        // Output point 1: x(0)
        v_out0 = _mm_add_pd(v_in0, v_m0);
        STR_128_D(curr_out, v_out_stride, v_out0);

        v_m1 = _mm_mul_pd(v_CRTM_11_1, v_in1);
        v_m2 = _mm_mul_pd(v_CRTM_11_2, v_in2);
        v_m3 = _mm_mul_pd(v_CRTM_11_3, v_in3);
        v_m4 = _mm_mul_pd(v_CRTM_11_4, v_in4);
        v_m5 = _mm_mul_pd(v_CRTM_11_5, v_in5);
        v_m6 = _mm_mul_pd(v_CRTM_11_6, v_in6);
        v_m7 = _mm_mul_pd(v_CRTM_11_7, v_in7);
        v_m8 = _mm_mul_pd(v_CRTM_11_8, v_in8);
        v_m9 = _mm_mul_pd(v_CRTM_11_9, v_in9);
        v_m10 = _mm_mul_pd(v_CRTM_11_10, v_in10);

        v_s4 = _mm_add_pd(v_m1, v_m3);
        v_s5 = _mm_add_pd(v_s4, v_in0);
        v_s6 = _mm_add_pd(v_m5, v_m7);
        v_s7 = _mm_add_pd(v_s6, v_m9);
        v_s8 = _mm_sub_pd(v_s5, v_s7);
        v_s9 = _mm_add_pd(v_m2, v_m4);
        v_s10 = _mm_add_pd(v_m6, v_m8);
        v_s11 = _mm_add_pd(v_s9, v_m10);
        v_s12 = _mm_add_pd(v_s10, v_s11);

        // Output point 2: x(1)
        v_out1 = _mm_sub_pd(v_s8, v_s12);
        curr_out = out + out_strides[1];
        STR_128_D(curr_out, v_out_stride, v_out1);
        // Output point 11: x(10)
        v_out10 = _mm_add_pd(v_s8, v_s12);
        curr_out = out + out_strides[10];
        STR_128_D(curr_out, v_out_stride, v_out10);

        v_m11 = _mm_mul_pd(v_CRTM_11_1, v_in9);
        v_m12 = _mm_mul_pd(v_CRTM_11_2, v_in10);
        v_m13 = _mm_mul_pd(v_CRTM_11_3, v_in1);
        v_m14 = _mm_mul_pd(v_CRTM_11_4, v_in2);
        v_m15 = _mm_mul_pd(v_CRTM_11_5, v_in7);
        v_m16 = _mm_mul_pd(v_CRTM_11_6, v_in8);
        v_m17 = _mm_mul_pd(v_CRTM_11_7, v_in3);
        v_m18 = _mm_mul_pd(v_CRTM_11_8, v_in4);
        v_m19 = _mm_mul_pd(v_CRTM_11_9, v_in5);
        v_m20 = _mm_mul_pd(v_CRTM_11_10, v_in6);

        v_s13 = _mm_add_pd(v_m11, v_m13);
        v_s14 = _mm_add_pd(v_s13, v_in0);
        v_s15 = _mm_add_pd(v_m15, v_m17);
        v_s16 = _mm_add_pd(v_s15, v_m19);
        v_s17 = _mm_sub_pd(v_s14, v_s16);
        v_s18 = _mm_sub_pd(v_m12, v_m14);
        v_s19 = _mm_sub_pd(v_m16, v_m18);
        v_s20 = _mm_add_pd(v_s19, v_m20);
        v_s21 = _mm_add_pd(v_s18, v_s20);

        // Output point 3: x(2)
        v_out2 = _mm_add_pd(v_s17, v_s21);
        curr_out = out + out_strides[2];
        STR_128_D(curr_out, v_out_stride, v_out2);
        // Output point 10: x(9)
        v_out9 = _mm_sub_pd(v_s17, v_s21);
        curr_out = out + out_strides[9];
        STR_128_D(curr_out, v_out_stride, v_out9);

        v_m21 = _mm_mul_pd(v_CRTM_11_1, v_in7);
        v_m22 = _mm_mul_pd(v_CRTM_11_2, v_in8);
        v_m23 = _mm_mul_pd(v_CRTM_11_3, v_in5);
        v_m24 = _mm_mul_pd(v_CRTM_11_4, v_in6);
        v_m25 = _mm_mul_pd(v_CRTM_11_5, v_in1);
        v_m26 = _mm_mul_pd(v_CRTM_11_6, v_in2);
        v_m27 = _mm_mul_pd(v_CRTM_11_7, v_in9);
        v_m28 = _mm_mul_pd(v_CRTM_11_8, v_in10);
        v_m29 = _mm_mul_pd(v_CRTM_11_9, v_in3);
        v_m30 = _mm_mul_pd(v_CRTM_11_10, v_in4);

        v_s22 = _mm_add_pd(v_m21, v_m23);
        v_s23 = _mm_add_pd(v_s22, v_in0);
        v_s24 = _mm_add_pd(v_m25, v_m27);
        v_s25 = _mm_add_pd(v_s24, v_m29);
        v_s26 = _mm_sub_pd(v_s23, v_s25);
        v_s27 = _mm_sub_pd(v_m22, v_m24);
        v_s28 = _mm_add_pd(v_m26, v_m28);
        v_s29 = _mm_sub_pd(v_s28, v_m30);
        v_s30 = _mm_add_pd(v_s27, v_s29);

        // Output point 4: x(3)
        v_out3 = _mm_sub_pd(v_s26, v_s30);
        curr_out = out + out_strides[3];
        STR_128_D(curr_out, v_out_stride, v_out3);
        // Output point 9: x(8)
        v_out8 = _mm_add_pd(v_s26, v_s30);
        curr_out = out + out_strides[8];
        STR_128_D(curr_out, v_out_stride, v_out8);

        v_m31 = _mm_mul_pd(v_CRTM_11_1, v_in5);
        v_m32 = _mm_mul_pd(v_CRTM_11_2, v_in6);
        v_m33 = _mm_mul_pd(v_CRTM_11_3, v_in9);
        v_m34 = _mm_mul_pd(v_CRTM_11_4, v_in10);
        v_m35 = _mm_mul_pd(v_CRTM_11_5, v_in3);
        v_m36 = _mm_mul_pd(v_CRTM_11_6, v_in4);
        v_m37 = _mm_mul_pd(v_CRTM_11_7, v_in1);
        v_m38 = _mm_mul_pd(v_CRTM_11_8, v_in2);
        v_m39 = _mm_mul_pd(v_CRTM_11_9, v_in7);
        v_m40 = _mm_mul_pd(v_CRTM_11_10, v_in8);

        v_s31 = _mm_add_pd(v_m31, v_m33);
        v_s32 = _mm_add_pd(v_s31, v_in0);
        v_s33 = _mm_add_pd(v_m35, v_m37);
        v_s34 = _mm_add_pd(v_s33, v_m39);
        v_s35 = _mm_sub_pd(v_s32, v_s34);
        v_s36 = _mm_sub_pd(v_m32, v_m34);
        v_s37 = _mm_sub_pd(v_m38, v_m36);
        v_s38 = _mm_add_pd(v_s37, v_m40);
        v_s39 = _mm_add_pd(v_s36, v_s38);

        // Output point 5: x(4)
        v_out4 = _mm_sub_pd(v_s35, v_s39);
        curr_out = out + out_strides[4];
        STR_128_D(curr_out, v_out_stride, v_out4);
        // Output point 8: x(7)
        v_out7 = _mm_add_pd(v_s35, v_s39);
        curr_out = out + out_strides[7];
        STR_128_D(curr_out, v_out_stride, v_out7);

        v_m41 = _mm_mul_pd(v_CRTM_11_1, v_in3);
        v_m42 = _mm_mul_pd(v_CRTM_11_2, v_in4);
        v_m43 = _mm_mul_pd(v_CRTM_11_3, v_in7);
        v_m44 = _mm_mul_pd(v_CRTM_11_4, v_in8);
        v_m45 = _mm_mul_pd(v_CRTM_11_5, v_in9);
        v_m46 = _mm_mul_pd(v_CRTM_11_6, v_in10);
        v_m47 = _mm_mul_pd(v_CRTM_11_7, v_in5);
        v_m48 = _mm_mul_pd(v_CRTM_11_8, v_in6);
        v_m49 = _mm_mul_pd(v_CRTM_11_9, v_in1);
        v_m50 = _mm_mul_pd(v_CRTM_11_10, v_in2);

        v_s40 = _mm_add_pd(v_m41, v_m43);
        v_s41 = _mm_add_pd(v_s40, v_in0);
        v_s42 = _mm_add_pd(v_m45, v_m47);
        v_s43 = _mm_add_pd(v_s42, v_m49);
        v_s44 = _mm_sub_pd(v_s41, v_s43);
        v_s45 = _mm_add_pd(v_m42, v_m44);
        v_s46 = _mm_add_pd(v_m46, v_m48);
        v_s47 = _mm_sub_pd(v_s45, v_s46);
        v_s48 = _mm_sub_pd(v_s47, v_m50);

        // Output point 6: x(5)
        v_out5 = _mm_add_pd(v_s44, v_s48);
        curr_out = out + out_strides[5];
        STR_128_D(curr_out, v_out_stride, v_out5);
        // Output point 7: x(6)
        v_out6 = _mm_sub_pd(v_s44, v_s48);
        curr_out = out + out_strides[6];
        STR_128_D(curr_out, v_out_stride, v_out6);

        in += v_in_stride * NUM_SETS_REAL_128_D;
        out += v_out_stride * NUM_SETS_REAL_128_D;
    }
    // tail cases
    if (n & 1)
    {
        DOUBLE in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10;
        DOUBLE s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14,
               s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27,
               s28, s29, s30, s31, s32, s33, s34, s35, s36, s37, s38, s39, s40,
               s41, s42, s43, s44, s45, s46, s47, s48;
        DOUBLE m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14,
               m15, m16, m17, m18, m19, m20, m21, m22, m23, m24, m25, m26, m27,
               m28, m29, m30, m31, m32, m33, m34, m35, m36, m37, m38, m39, m40,
               m41, m42, m43, m44, m45, m46, m47, m48, m49, m50;

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
        // Input point 10: X(9)
        in9 = in[in_strides[9]];
        // Input point 11: X(10)
        in10 = in[in_strides[10]];

        s0 = in1 + in3;
        s1 = in5 + in7;
        s2 = s0 + s1;
        s3 = s2 + in9;
        m0 = CRTM_11_11 * s3;

        // Output point 1: x(0)
        *out = in0 + m0;

        m1 = CRTM_11_1 * in1;
        m2 = CRTM_11_2 * in2;
        m3 = CRTM_11_3 * in3;
        m4 = CRTM_11_4 * in4;
        m5 = CRTM_11_5 * in5;
        m6 = CRTM_11_6 * in6;
        m7 = CRTM_11_7 * in7;
        m8 = CRTM_11_8 * in8;
        m9 = CRTM_11_9 * in9;
        m10 = CRTM_11_10 * in10;

        s4 = m1 + m3;
        s5 = s4 + in0;
        s6 = m5 + m7;
        s7 = s6 + m9;
        s8 = s5 - s7;

        s9 = m2 + m4;
        s10 = m6 + m8;
        s11 = s9 + m10;
        s12 = s10 + s11;

        // Output point 2: x(1)
        out[out_strides[1]] = s8 - s12;
        // Output point 11: x(10)
        out[out_strides[10]] = s8 + s12;

        m11 = CRTM_11_1 * in9;
        m12 = CRTM_11_2 * in10;
        m13 = CRTM_11_3 * in1;
        m14 = CRTM_11_4 * in2;
        m15 = CRTM_11_5 * in7;
        m16 = CRTM_11_6 * in8;
        m17 = CRTM_11_7 * in3;
        m18 = CRTM_11_8 * in4;
        m19 = CRTM_11_9 * in5;
        m20 = CRTM_11_10 * in6;

        s13 = m11 + m13;
        s14 = s13 + in0;
        s15 = m15 + m17;
        s16 = s15 + m19;
        s17 = s14 - s16;

        s18 = m12 - m14;
        s19 = m16 - m18;
        s20 = s19 + m20;
        s21 = s18 + s20;

        // Output point 3: x(2)
        out[out_strides[2]] = s17 + s21;
        // Output point 10: x(9)
        out[out_strides[9]] = s17 - s21;

        m21 = CRTM_11_1 * in7;
        m22 = CRTM_11_2 * in8;
        m23 = CRTM_11_3 * in5;
        m24 = CRTM_11_4 * in6;
        m25 = CRTM_11_5 * in1;
        m26 = CRTM_11_6 * in2;
        m27 = CRTM_11_7 * in9;
        m28 = CRTM_11_8 * in10;
        m29 = CRTM_11_9 * in3;
        m30 = CRTM_11_10 * in4;

        s22 = m21 + m23;
        s23 = s22 + in0;
        s24 = m25 + m27;
        s25 = s24 + m29;
        s26 = s23 - s25;

        s27 = m22 - m24;
        s28 = m26 + m28;
        s29 = s28 - m30;
        s30 = s27 + s29;

        // Output point 4: x(3)
        out[out_strides[3]] = s26 - s30;
        // Output point 9: x(8)
        out[out_strides[8]] = s26 + s30;

        m31 = CRTM_11_1 * in5;
        m32 = CRTM_11_2 * in6;
        m33 = CRTM_11_3 * in9;
        m34 = CRTM_11_4 * in10;
        m35 = CRTM_11_5 * in3;
        m36 = CRTM_11_6 * in4;
        m37 = CRTM_11_7 * in1;
        m38 = CRTM_11_8 * in2;
        m39 = CRTM_11_9 * in7;
        m40 = CRTM_11_10 * in8;

        s31 = m31 + m33;
        s32 = s31 + in0;
        s33 = m35 + m37;
        s34 = s33 + m39;
        s35 = s32 - s34;

        s36 = m32 - m34;
        s37 = m38 - m36;
        s38 = s37 + m40;
        s39 = s36 + s38;

        // Output point 5: x(4)
        out[out_strides[4]] = s35 - s39;
        // Output point 8: x(7)
        out[out_strides[7]] = s35 + s39;

        m41 = CRTM_11_1 * in3;
        m42 = CRTM_11_2 * in4;
        m43 = CRTM_11_3 * in7;
        m44 = CRTM_11_4 * in8;
        m45 = CRTM_11_5 * in9;
        m46 = CRTM_11_6 * in10;
        m47 = CRTM_11_7 * in5;
        m48 = CRTM_11_8 * in6;
        m49 = CRTM_11_9 * in1;
        m50 = CRTM_11_10 * in2;

        s40 = m41 + m43;
        s41 = s40 + in0;
        s42 = m45 + m47;
        s43 = s42 + m49;
        s44 = s41 - s43;

        s45 = m42 + m44;
        s46 = m46 + m48;
        s47 = s45 - s46;
        s48 = s47 - m50;

        // Output point 6: x(5)
        out[out_strides[5]] = s44 + s48;
        // Output point 7: x(6)
        out[out_strides[6]] = s44 - s48;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hc_rfft11avx128(UINT8 precision, UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return rfft11avx128_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return rfft11avx128_fp64_fwd;
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
            return rfft11avx128_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return rfft11avx128_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
