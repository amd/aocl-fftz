// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft9avx128.c
 *
 *  @brief Radix-9 r2hc Real-FFT kernel with AVX-128 operations using x86 SIMD
 *  intrinsics
 *
 *  This file contains the DIT radix-9 real-to-halfcomplex implementations using
 *  AVX128 SIMD operations for single-precision and double-precision inputs.
 *
 *  @author Amrin Fathima
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/r2hc_simd_avx128.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] = {
                                                    {{0, 22, 37, 58, 40, 1},
                                                     {0, 18, 32, 58, 44, 1}},
                                                    {{0, 22, 37, 29, 6,  1},
                                                     {0, 18, 32, 29, 6,  1}}};

ops_cycles_t get_ops_cnt_r2hc_rfft9avx128(FFTZ_UINT8 precision,
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

static FFTZ_VOID r2hc_rfft9avx128_fp32_fwd(FFTZ_VOID *in_real,
                                           FFTZ_VOID *in_imag,
                                           FFTZ_VOID *out_real,
                                           FFTZ_VOID *out_imag, FFTZ_INTP n,
                                           aoclfftz_strides_t *strides,
                                           FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_9_0 =
        0.766044443118978035202392650555416673935832457f;
    const FFTZ_FLOAT CRTM_9_1 =
        0.642787609686539326322643409907263432907559884f;
    const FFTZ_FLOAT CRTM_9_2 =
        0.173648177666930348851716626769314796000375677f;
    const FFTZ_FLOAT CRTM_9_3 =
        0.984807753012208059366743024589523013670643252f;
    const FFTZ_FLOAT CRTM_9_4 =
        0.500000000000000000000000000000000000000000000f;
    const FFTZ_FLOAT CRTM_9_5 =
        0.866025403784438646763723170752936183471402627f;
    const FFTZ_FLOAT CRTM_9_6 =
        0.939692620785908384054109277324975766871890789f;
    const FFTZ_FLOAT CRTM_9_7 =
        0.342020143325668733044099614682259580763083320f;

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

    __m128 v_CRTM_9_0 = _mm_broadcast_ss(&CRTM_9_0);
    __m128 v_CRTM_9_1 = _mm_broadcast_ss(&CRTM_9_1);
    __m128 v_CRTM_9_2 = _mm_broadcast_ss(&CRTM_9_2);
    __m128 v_CRTM_9_3 = _mm_broadcast_ss(&CRTM_9_3);
    __m128 v_CRTM_9_4 = _mm_broadcast_ss(&CRTM_9_4);
    __m128 v_CRTM_9_5 = _mm_broadcast_ss(&CRTM_9_5);
    __m128 v_CRTM_9_6 = _mm_broadcast_ss(&CRTM_9_6);
    __m128 v_CRTM_9_7 = _mm_broadcast_ss(&CRTM_9_7);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
               v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26;
        __m128 v_m0, v_m1, v_m2, v_m3, v_m4, v_m5, v_m6, v_m7, v_m8, v_m9,
               v_m10, v_m11, v_m12, v_m13, v_m14, v_m15, v_m16, v_m17, v_m18,
               v_m19, v_m20, v_m21;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_stride, v_in0, is_contiguous_in);
        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_128_S(curr_in, v_in_stride, v_in1, is_contiguous_in);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_128_S(curr_in, v_in_stride, v_in2, is_contiguous_in);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_128_S(curr_in, v_in_stride, v_in3, is_contiguous_in);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_128_S(curr_in, v_in_stride, v_in4, is_contiguous_in);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_128_S(curr_in, v_in_stride, v_in5, is_contiguous_in);
        // Input point 7: x(6)
        curr_in = in + in_strides[6];
        LDR_128_S(curr_in, v_in_stride, v_in6, is_contiguous_in);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_128_S(curr_in, v_in_stride, v_in7, is_contiguous_in);
        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDR_128_S(curr_in, v_in_stride, v_in8, is_contiguous_in);

        v_s0 = _mm_add_ps(v_in1, v_in8);
        v_s1 = _mm_sub_ps(v_in1, v_in8);
        v_s2 = _mm_add_ps(v_in2, v_in7);
        v_s3 = _mm_sub_ps(v_in2, v_in7);
        v_s4 = _mm_add_ps(v_in3, v_in6);
        v_s5 = _mm_sub_ps(v_in3, v_in6);
        v_s6 = _mm_add_ps(v_in4, v_in5);
        v_s7 = _mm_sub_ps(v_in4, v_in5);

        v_s8 = _mm_add_ps(v_s0, v_s6);
        v_s9 = _mm_add_ps(v_s8, v_s2);
        v_s10 = _mm_sub_ps(v_s1, v_s3);
        v_s11 = _mm_add_ps(v_s10, v_s7);
        v_s12 = _mm_add_ps(v_s9, v_s4);
        v_s13 = _mm_add_ps(v_s12, v_in0);
        v_m0 = _mm_mul_ps(v_CRTM_9_4, v_s4);
        v_m1 = _mm_mul_ps(v_CRTM_9_5, v_s5);
        v_s14 = _mm_sub_ps(v_in0, v_m0);

        v_m2 = _mm_mul_ps(v_CRTM_9_0, v_s0);
        v_m3 = _mm_mul_ps(v_CRTM_9_2, v_s2);
        v_m4 = _mm_mul_ps(v_CRTM_9_6, v_s6);
        v_m5 = _mm_mul_ps(v_CRTM_9_1, v_s1);
        v_m6 = _mm_mul_ps(v_CRTM_9_3, v_s3);
        v_m7 = _mm_mul_ps(v_CRTM_9_7, v_s7);
        v_m8 = _mm_mul_ps(v_CRTM_9_0, v_s2);
        v_m9 = _mm_mul_ps(v_CRTM_9_2, v_s0);
        v_m10 = _mm_mul_ps(v_CRTM_9_6, v_s2);
        v_m11 = _mm_mul_ps(v_CRTM_9_0, v_s6);
        v_m12 = _mm_mul_ps(v_CRTM_9_1, v_s7);
        v_m13 = _mm_mul_ps(v_CRTM_9_3, v_s1);
        v_m14 = _mm_mul_ps(v_CRTM_9_7, v_s3);

        // Output point 1: X(0)
        v_out0 = v_s13;
        STR_128_S(curr_out, v_out_stride, v_out0, is_contiguous_out);

        v_s15 = _mm_add_ps(v_s14, v_m2);
        v_s16 = _mm_sub_ps(v_m3, v_m4);
        // Output point 2: X(1)
        v_out1 = _mm_add_ps(v_s15, v_s16);

        v_s17 = _mm_add_ps(v_m5, v_m6);
        v_s18 = _mm_add_ps(v_m7, v_m1);
        v_s19 = NEGATE_128_S(_mm_add_ps(v_s17, v_s18));
        // Output point 3: X(2)
        v_out2 = v_s19;
        curr_out = out + out_strides[1];
        STRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);

        v_s20 = _mm_add_ps(v_m11, v_m9);
        v_s21 = _mm_sub_ps(v_s20, v_m10);
        // Output point 4: X(3)
        v_out3 = _mm_add_ps(v_s21, v_s14);

        v_s22 = _mm_add_ps(v_m13, v_m14);
        v_s23 = _mm_sub_ps(v_m1, v_s22);
        // Output point 5: X(4)
        v_out4 = _mm_add_ps(v_s23, v_m12);
        curr_out = out + out_strides[3];
        STRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);

        v_m15 = _mm_mul_ps(v_CRTM_9_4, v_s9);
        v_s24 = _mm_sub_ps(v_s4, v_m15);
        // Output point 6: X(5)
        v_out5 = _mm_add_ps(v_in0, v_s24);

        v_m16 = NEGATE_128_S(_mm_mul_ps(v_CRTM_9_5, v_s11));
        // Output point 7: X(6)
        v_out6 = v_m16;
        curr_out = out + out_strides[5];
        STRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);

        v_s25 = _mm_add_ps(v_s14, v_m8);
        v_m17 = _mm_mul_ps(v_CRTM_9_6, v_s0);
        v_s26 = _mm_sub_ps(v_s25, v_m17);
        v_m18 = _mm_mul_ps(v_CRTM_9_2, v_s6);
        // Output point 8: X(7)
        v_out7 = _mm_add_ps(v_s26, v_m18);

        v_m19 = _mm_mul_ps(v_CRTM_9_1, v_s3);
        v_m20 = _mm_mul_ps(v_CRTM_9_3, v_s7);
        v_s15 = _mm_add_ps(v_m19, v_m20);
        v_s16 = _mm_sub_ps(v_s15, v_m1);
        v_m21 = _mm_mul_ps(v_CRTM_9_7, v_s1);
        // Output point 9: X(8)
        v_out8 = _mm_sub_ps(v_s16, v_m21);
        curr_out = out + out_strides[7];
        STRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);

        in += v_in_stride * NUM_SETS_REAL_128_S;
        out += v_out_stride * NUM_SETS_REAL_128_S;
    }
    // tail cases
    if (n & 2)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
               v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26;
        __m128 v_m0, v_m1, v_m2, v_m3, v_m4, v_m5, v_m6, v_m7, v_m8, v_m9,
               v_m10, v_m11, v_m12, v_m13, v_m14, v_m15, v_m16, v_m17, v_m18,
               v_m19, v_m20, v_m21;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8;

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

        v_s0 = _mm_add_ps(v_in1, v_in8);
        v_s1 = _mm_sub_ps(v_in1, v_in8);
        v_s2 = _mm_add_ps(v_in2, v_in7);
        v_s3 = _mm_sub_ps(v_in2, v_in7);
        v_s4 = _mm_add_ps(v_in3, v_in6);
        v_s5 = _mm_sub_ps(v_in3, v_in6);
        v_s6 = _mm_add_ps(v_in4, v_in5);
        v_s7 = _mm_sub_ps(v_in4, v_in5);

        v_s8 = _mm_add_ps(v_s0, v_s6);
        v_s9 = _mm_add_ps(v_s8, v_s2);
        v_s10 = _mm_sub_ps(v_s1, v_s3);
        v_s11 = _mm_add_ps(v_s10, v_s7);
        v_s12 = _mm_add_ps(v_s9, v_s4);
        v_s13 = _mm_add_ps(v_s12, v_in0);
        v_m0 = _mm_mul_ps(v_CRTM_9_4, v_s4);
        v_m1 = _mm_mul_ps(v_CRTM_9_5, v_s5);
        v_s14 = _mm_sub_ps(v_in0, v_m0);

        v_m2 = _mm_mul_ps(v_CRTM_9_0, v_s0);
        v_m3 = _mm_mul_ps(v_CRTM_9_2, v_s2);
        v_m4 = _mm_mul_ps(v_CRTM_9_6, v_s6);
        v_m5 = _mm_mul_ps(v_CRTM_9_1, v_s1);
        v_m6 = _mm_mul_ps(v_CRTM_9_3, v_s3);
        v_m7 = _mm_mul_ps(v_CRTM_9_7, v_s7);
        v_m8 = _mm_mul_ps(v_CRTM_9_0, v_s2);
        v_m9 = _mm_mul_ps(v_CRTM_9_2, v_s0);
        v_m10 = _mm_mul_ps(v_CRTM_9_6, v_s2);
        v_m11 = _mm_mul_ps(v_CRTM_9_0, v_s6);
        v_m12 = _mm_mul_ps(v_CRTM_9_1, v_s7);
        v_m13 = _mm_mul_ps(v_CRTM_9_3, v_s1);
        v_m14 = _mm_mul_ps(v_CRTM_9_7, v_s3);

        // Output point 1: X(0)
        v_out0 = v_s13;
        STHR_128_S(curr_out, v_out_stride, v_out0);

        v_s15 = _mm_add_ps(v_s14, v_m2);
        v_s16 = _mm_sub_ps(v_m3, v_m4);
        // Output point 2: X(1)
        v_out1 = _mm_add_ps(v_s15, v_s16);

        v_s17 = _mm_add_ps(v_m5, v_m6);
        v_s18 = _mm_add_ps(v_m7, v_m1);
        v_s19 = NEGATE_128_S(_mm_add_ps(v_s17, v_s18));
        // Output point 3: X(2)
        v_out2 = v_s19;
        curr_out = out + out_strides[1];
        STHRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);

        v_s20 = _mm_add_ps(v_m11, v_m9);
        v_s21 = _mm_sub_ps(v_s20, v_m10);
        // Output point 4: X(3)
        v_out3 = _mm_add_ps(v_s21, v_s14);

        v_s22 = _mm_add_ps(v_m13, v_m14);
        v_s23 = _mm_sub_ps(v_m1, v_s22);
        // Output point 5: X(4)
        v_out4 = _mm_add_ps(v_s23, v_m12);
        curr_out = out + out_strides[3];
        STHRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);

        v_m15 = _mm_mul_ps(v_CRTM_9_4, v_s9);
        v_s24 = _mm_sub_ps(v_s4, v_m15);
        // Output point 6: X(5)
        v_out5 = _mm_add_ps(v_in0, v_s24);

        v_m16 = NEGATE_128_S(_mm_mul_ps(v_CRTM_9_5, v_s11));
        // Output point 7: X(6)
        v_out6 = v_m16;
        curr_out = out + out_strides[5];
        STHRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);

        v_s25 = _mm_add_ps(v_s14, v_m8);
        v_m17 = _mm_mul_ps(v_CRTM_9_6, v_s0);
        v_s26 = _mm_sub_ps(v_s25, v_m17);
        v_m18 = _mm_mul_ps(v_CRTM_9_2, v_s6);
        // Output point 8: X(7)
        v_out7 = _mm_add_ps(v_s26, v_m18);

        v_m19 = _mm_mul_ps(v_CRTM_9_1, v_s3);
        v_m20 = _mm_mul_ps(v_CRTM_9_3, v_s7);
        v_s15 = _mm_add_ps(v_m19, v_m20);
        v_s16 = _mm_sub_ps(v_s15, v_m1);
        v_m21 = _mm_mul_ps(v_CRTM_9_7, v_s1);
        // Output point 9: X(8)
        v_out8 = _mm_sub_ps(v_s16, v_m21);
        curr_out = out + out_strides[7];
        STHRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        FFTZ_FLOAT in0, in1, in2, in3, in4, in5, in6, in7, in8;
        FFTZ_FLOAT s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13,
            s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26,
            s27, s28;
        FFTZ_FLOAT m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13,
            m14, m15, m16, m17, m18, m19, m20, m21;

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
        s13 = s1 - s3;
        s10 = s13 + s7;
        s14 = s9 + s4;
        s11 = s14 + in0;

        m0 = CRTM_9_4 * s4;
        m1 = CRTM_9_5 * s5;
        s12 = in0 - m0;

        // Output point 1: X(0)
        *out = s11;

        m2 = CRTM_9_0 * s0;
        m3 = CRTM_9_2 * s2;
        m4 = CRTM_9_6 * s6;
        s15 = s12 + m2;
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
        s20 = s12 + m8;
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

        m15 = -CRTM_9_5 * s10;
        // Output point 7: X(6)
        out[out_strides[6]] = m15;

        m16 = CRTM_9_0 * s2;
        m17 = CRTM_9_2 * s6;
        m18 = CRTM_9_6 * s0;
        s25 = s12 + m16;
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
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft9avx128_fp32_bwd(FFTZ_VOID *in_real,
                                           FFTZ_VOID *in_imag,
                                           FFTZ_VOID *out_real,
                                           FFTZ_VOID *out_imag, FFTZ_INTP n,
                                           aoclfftz_strides_t *strides,
                                           FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_9_0 =
        0.766044443118978035202392650555416673935832457f;
    const FFTZ_FLOAT CRTM_9_1 =
        0.642787609686539326322643409907263432907559884f;
    const FFTZ_FLOAT CRTM_9_2 =
        0.173648177666930348851716626769314796000375677f;
    const FFTZ_FLOAT CRTM_9_3 =
        0.984807753012208059366743024589523013670643252f;
    const FFTZ_FLOAT CRTM_9_4 =
        0.500000000000000000000000000000000000000000000f;
    const FFTZ_FLOAT CRTM_9_5 =
        0.866025403784438646763723170752936183471402627f;
    const FFTZ_FLOAT CRTM_9_6 =
        2.000000000000000000000000000000000000000000000f;
    // Below CRTMs are the product or sum of the above CRTMs, Precomputed
    // to save multiplications on the fly.
    // CRTM_9_7 = CRTM_9_6 * CRTM_9_5
    const FFTZ_FLOAT CRTM_9_7 =
        1.732050807568877293527446341505872366942805254f;
    // CRTM_9_8 = CRTM_9_6 * CRTM_9_5 * CRTM_9_3
    const FFTZ_FLOAT CRTM_9_8 =
        1.705737063904886419256501927880148143872040592f;
    // CRTM_9_9 = CRTM_9_7 * CRTM_9_2
    const FFTZ_FLOAT CRTM_9_9 =
        0.300767466360870593278543795225003852144476516f;
    // CRTM_9_10 = CRTM_9_8 - CRTM_9_0 + CRTM_9_2
    const FFTZ_FLOAT CRTM_9_10 =
        1.113340798452838732905825904094046265936583812f;
    // CRTM_9_11 = CRTM_9_3 + CRTM_9_6 * CRTM_9_2 * CRTM_9_3
    const FFTZ_FLOAT CRTM_9_11 =
        1.326827896337876792410842639271782594433726619f;

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

    __m128 v128_CRTM_9_0  = _mm_broadcast_ss(&CRTM_9_0);
    __m128 v128_CRTM_9_1  = _mm_broadcast_ss(&CRTM_9_1);
    __m128 v128_CRTM_9_2  = _mm_broadcast_ss(&CRTM_9_2);
    __m128 v128_CRTM_9_3  = _mm_broadcast_ss(&CRTM_9_3);
    __m128 v128_CRTM_9_4  = _mm_broadcast_ss(&CRTM_9_4);
    __m128 v128_CRTM_9_5  = _mm_broadcast_ss(&CRTM_9_5);
    __m128 v128_CRTM_9_6  = _mm_broadcast_ss(&CRTM_9_6);
    __m128 v128_CRTM_9_7  = _mm_broadcast_ss(&CRTM_9_7);
    __m128 v128_CRTM_9_8  = _mm_broadcast_ss(&CRTM_9_8);
    __m128 v128_CRTM_9_9  = _mm_broadcast_ss(&CRTM_9_9);
    __m128 v128_CRTM_9_10 = _mm_broadcast_ss(&CRTM_9_10);
    __m128 v128_CRTM_9_11 = _mm_broadcast_ss(&CRTM_9_11);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8;
        __m128 v_m0, v_m1, v_m2, v_m3, v_m4, v_m5, v_m6, v_m7, v_m8, v_m9,
               v_m10, v_m11, v_m12, v_m13, v_m14, v_m15, v_m16, v_m17;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
               v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27,
               v_s28, v_s29, v_s30, v_s31;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8;

        curr_in = in;
        curr_out = out;

        // Input point 1: X(0)
        LDR_128_S(curr_in, v_in_stride, v_in0, is_contiguous_in);
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

        v_m0 = _mm_mul_ps(v128_CRTM_9_7, v_in6);
        v_s0 = _mm_sub_ps(v_in0, v_in5);
        v_m1 = _mm_mul_ps(v128_CRTM_9_6, v_in5);
        v_s1 = _mm_add_ps(v_in0, v_m1);
        v_s2 = _mm_sub_ps(v_s0, v_m0);
        v_s3 = _mm_add_ps(v_s0, v_m0);
        v_s4 = _mm_add_ps(v_in7, v_in3);
        v_s5 = _mm_sub_ps(v_in7, v_in3);
        v_m2 = _mm_mul_ps(v128_CRTM_9_5, v_s5);
        v_s6 = _mm_add_ps(v_in8, v_in4);
        v_m3 = _mm_mul_ps(v128_CRTM_9_5, v_s6);
        v_s7 = _mm_sub_ps(v_in4, v_in8);
        v_s8 = _mm_add_ps(v_in1, v_s4);
        v_m4 = _mm_mul_ps(v128_CRTM_9_4, v_s7);
        v_s9 = _mm_add_ps(v_in2, v_m4);
        v_s10 = _mm_add_ps(v_m2, v_s9);
        v_s11 = _mm_sub_ps(v_s9, v_m2);
        v_m5 = _mm_mul_ps(v128_CRTM_9_4, v_s4);
        v_s12 = _mm_sub_ps(v_in1, v_m5);
        v_s13 = _mm_sub_ps(v_s12, v_m3);
        v_s14 = _mm_add_ps(v_s12, v_m3);
        v_m6 = _mm_mul_ps(v128_CRTM_9_6, v_s8);
        v_s15 = _mm_add_ps(v_s1, v_m6);
        // Output point 1: x(0)
        v_out0 = v_s15;
        STR_128_S(curr_out, v_out_stride, v_out0, is_contiguous_out);

        v_s16 = _mm_sub_ps(v_s1, v_s8);
        v_s17 = _mm_sub_ps(v_in2, v_s7);
        v_m7 = _mm_mul_ps(v128_CRTM_9_7, v_s17);
        v_s18 = _mm_sub_ps(v_s16, v_m7);
        v_s19 = _mm_add_ps(v_s16, v_m7);
        // Output point 4: x(3)
        v_out3 = v_s18;
        curr_out = out + out_strides[3];
        STR_128_S(curr_out, v_out_stride, v_out3, is_contiguous_out);
        // Output point 7: x(6)
        v_out6 = v_s19;
        curr_out = out + out_strides[6];
        STR_128_S(curr_out, v_out_stride, v_out6, is_contiguous_out);

        v_m8 = _mm_mul_ps(v128_CRTM_9_0, v_s13);
        v_m9 = _mm_mul_ps(v128_CRTM_9_1, v_s10);
        v_s20 = _mm_sub_ps(v_m8, v_m9);
        v_m10 = _mm_mul_ps(v128_CRTM_9_11, v_s10);
        v_m11 = _mm_mul_ps(v128_CRTM_9_10, v_s13);
        v_s21 = _mm_add_ps(v_m11, v_m10);
        v_s22 = _mm_sub_ps(v_s2, v_s20);
        v_m12 = _mm_mul_ps(v128_CRTM_9_6, v_s20);
        v_s23 = _mm_add_ps(v_s2, v_m12);
        // Output point 2: x(1)
        v_out1 = v_s23;
        curr_out = out + out_strides[1];
        STR_128_S(curr_out, v_out_stride, v_out1, is_contiguous_out);

        v_s24 = _mm_add_ps(v_s22, v_s21);
        v_s25 = _mm_sub_ps(v_s22, v_s21);
        // Output point 8: x(7)
        v_out7 = v_s24;
        curr_out = out + out_strides[7];
        STR_128_S(curr_out, v_out_stride, v_out7, is_contiguous_out);
        // Output point 5: x(4)
        v_out4 = v_s25;
        curr_out = out + out_strides[4];
        STR_128_S(curr_out, v_out_stride, v_out4, is_contiguous_out);

        v_m13 = _mm_mul_ps(v128_CRTM_9_8, v_s14);
        v_m14 = _mm_mul_ps(v128_CRTM_9_9, v_s11);
        v_s26 = _mm_add_ps(v_m13, v_m14);
        v_m15 = _mm_mul_ps(v128_CRTM_9_2, v_s14);
        v_m16 = _mm_mul_ps(v128_CRTM_9_3, v_s11);
        v_s27 = _mm_sub_ps(v_m15, v_m16);
        v_s28 = _mm_sub_ps(v_s3, v_s27);
        v_m17 = _mm_mul_ps(v128_CRTM_9_6, v_s27);
        v_s29 = _mm_add_ps(v_s3, v_m17);
        // Output point 3: x(2)
        v_out2 = v_s29;
        curr_out = out + out_strides[2];
        STR_128_S(curr_out, v_out_stride, v_out2, is_contiguous_out);

        v_s30 = _mm_add_ps(v_s28, v_s26);
        v_s31 = _mm_sub_ps(v_s28, v_s26);
        // Output point 9: x(8)
        v_out8 = v_s30;
        curr_out = out + out_strides[8];
        STR_128_S(curr_out, v_out_stride, v_out8, is_contiguous_out);
        // Output point 6: x(5)
        v_out5 = v_s31;
        curr_out = out + out_strides[5];
        STR_128_S(curr_out, v_out_stride, v_out5, is_contiguous_out);

        in += v_in_stride * NUM_SETS_REAL_128_S;
        out += v_out_stride * NUM_SETS_REAL_128_S;
    }
    // tail cases
    if (n & 2)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8;
        __m128 v_m0, v_m1, v_m2, v_m3, v_m4, v_m5, v_m6, v_m7, v_m8, v_m9,
               v_m10, v_m11, v_m12, v_m13, v_m14, v_m15, v_m16, v_m17;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
               v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27,
               v_s28, v_s29, v_s30, v_s31;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8;

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

        v_m0 = _mm_mul_ps(v128_CRTM_9_7, v_in6);
        v_s0 = _mm_sub_ps(v_in0, v_in5);
        v_m1 = _mm_mul_ps(v128_CRTM_9_6, v_in5);
        v_s1 = _mm_add_ps(v_in0, v_m1);
        v_s2 = _mm_sub_ps(v_s0, v_m0);
        v_s3 = _mm_add_ps(v_s0, v_m0);
        v_s4 = _mm_add_ps(v_in7, v_in3);
        v_s5 = _mm_sub_ps(v_in7, v_in3);
        v_m2 = _mm_mul_ps(v128_CRTM_9_5, v_s5);
        v_s6 = _mm_add_ps(v_in8, v_in4);
        v_m3 = _mm_mul_ps(v128_CRTM_9_5, v_s6);
        v_s7 = _mm_sub_ps(v_in4, v_in8);
        v_s8 = _mm_add_ps(v_in1, v_s4);
        v_m4 = _mm_mul_ps(v128_CRTM_9_4, v_s7);
        v_s9 = _mm_add_ps(v_in2, v_m4);
        v_s10 = _mm_add_ps(v_m2, v_s9);
        v_s11 = _mm_sub_ps(v_s9, v_m2);
        v_m5 = _mm_mul_ps(v128_CRTM_9_4, v_s4);
        v_s12 = _mm_sub_ps(v_in1, v_m5);
        v_s13 = _mm_sub_ps(v_s12, v_m3);
        v_s14 = _mm_add_ps(v_s12, v_m3);
        v_m6 = _mm_mul_ps(v128_CRTM_9_6, v_s8);
        v_s15 = _mm_add_ps(v_s1, v_m6);
        // Output point 1: x(0)
        v_out0 = v_s15;
        STHR_128_S(curr_out, v_out_stride, v_out0);

        v_s16 = _mm_sub_ps(v_s1, v_s8);
        v_s17 = _mm_sub_ps(v_in2, v_s7);
        v_m7 = _mm_mul_ps(v128_CRTM_9_7, v_s17);
        v_s18 = _mm_sub_ps(v_s16, v_m7);
        v_s19 = _mm_add_ps(v_s16, v_m7);
        // Output point 4: x(3)
        v_out3 = v_s18;
        curr_out = out + out_strides[3];
        STHR_128_S(curr_out, v_out_stride, v_out3);
        // Output point 7: x(6)
        v_out6 = v_s19;
        curr_out = out + out_strides[6];
        STHR_128_S(curr_out, v_out_stride, v_out6);

        v_m8 = _mm_mul_ps(v128_CRTM_9_0, v_s13);
        v_m9 = _mm_mul_ps(v128_CRTM_9_1, v_s10);
        v_s20 = _mm_sub_ps(v_m8, v_m9);
        v_m10 = _mm_mul_ps(v128_CRTM_9_11, v_s10);
        v_m11 = _mm_mul_ps(v128_CRTM_9_10, v_s13);
        v_s21 = _mm_add_ps(v_m11, v_m10);
        v_s22 = _mm_sub_ps(v_s2, v_s20);
        v_m12 = _mm_mul_ps(v128_CRTM_9_6, v_s20);
        v_s23 = _mm_add_ps(v_s2, v_m12);
        // Output point 2: x(1)
        v_out1 = v_s23;
        curr_out = out + out_strides[1];
        STHR_128_S(curr_out, v_out_stride, v_out1);

        v_s24 = _mm_add_ps(v_s22, v_s21);
        v_s25 = _mm_sub_ps(v_s22, v_s21);
        // Output point 8: x(7)
        v_out7 = v_s24;
        curr_out = out + out_strides[7];
        STHR_128_S(curr_out, v_out_stride, v_out7);
        // Output point 5: x(4)
        v_out4 = v_s25;
        curr_out = out + out_strides[4];
        STHR_128_S(curr_out, v_out_stride, v_out4);

        v_m13 = _mm_mul_ps(v128_CRTM_9_8, v_s14);
        v_m14 = _mm_mul_ps(v128_CRTM_9_9, v_s11);
        v_s26 = _mm_add_ps(v_m13, v_m14);
        v_m15 = _mm_mul_ps(v128_CRTM_9_2, v_s14);
        v_m16 = _mm_mul_ps(v128_CRTM_9_3, v_s11);
        v_s27 = _mm_sub_ps(v_m15, v_m16);
        v_s28 = _mm_sub_ps(v_s3, v_s27);
        v_m17 = _mm_mul_ps(v128_CRTM_9_6, v_s27);
        v_s29 = _mm_add_ps(v_s3, v_m17);
        // Output point 3: x(2)
        v_out2 = v_s29;
        curr_out = out + out_strides[2];
        STHR_128_S(curr_out, v_out_stride, v_out2);

        v_s30 = _mm_add_ps(v_s28, v_s26);
        v_s31 = _mm_sub_ps(v_s28, v_s26);
        // Output point 9: x(8)
        v_out8 = v_s30;
        curr_out = out + out_strides[8];
        STHR_128_S(curr_out, v_out_stride, v_out8);
        // Output point 6: x(5)
        v_out5 = v_s31;
        curr_out = out + out_strides[5];
        STHR_128_S(curr_out, v_out_stride, v_out5);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        FFTZ_FLOAT v0, v1, v2, v3, v4, v5, v6, v7, v8;
        FFTZ_FLOAT s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13,
            s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26,
            s27, s28, s29, s30, s31;
        FFTZ_FLOAT m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13,
            m14, m15, m16, m17;

        // Input point 1: X(0)
        v0 = *in;
        // Input point 2: X(1)
        v1 = in[in_strides[1]];
        // Input point 3: X(2)
        v2 = in[in_strides[2]];
        // Input point 4: X(3)
        v3 = in[in_strides[3]];
        // Input point 5: X(4)
        v4 = in[in_strides[4]];
        // Input point 6: X(5)
        v5 = in[in_strides[5]];
        // Input point 7: X(6)
        v6 = in[in_strides[6]];
        // Input point 8: X(7)
        v7 = in[in_strides[7]];
        // Input point 9: X(8)
        v8 = in[in_strides[8]];

        m0 = CRTM_9_7 * v6;
        s0 = v0 - v5;
        m1 = CRTM_9_6 * v5;
        s1 = v0 + m1;
        s2 = s0 - m0;
        s3 = s0 + m0;
        s4 = v7 + v3;
        s5 = v7 - v3;
        m2 = CRTM_9_5 * s5;
        s6 = v8 + v4;
        m3 = CRTM_9_5 * s6;
        s7 = v4 - v8;
        s8 = v1 + s4;
        m4 = CRTM_9_4 * s7;
        s9 = v2 + m4;
        s10 = m2 + s9;
        s11 = s9 - m2;
        m5 = CRTM_9_4 * s4;
        s12 = v1 - m5;
        s13 = s12 - m3;
        s14 = s12 + m3;
        m6 = CRTM_9_6 * s8;
        s15 = s1 + m6;
        // Output point 1: x(0)
        *out = s15;

        s16 = s1 - s8;
        s17 = v2 - s7;
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
    }

    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft9avx128_fp64_fwd(FFTZ_VOID *in_real,
                                           FFTZ_VOID *in_imag,
                                           FFTZ_VOID *out_real,
                                           FFTZ_VOID *out_imag, FFTZ_INTP n,
                                           aoclfftz_strides_t *strides,
                                           FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_9_0 =
        0.766044443118978035202392650555416673935832457;
    const FFTZ_DOUBLE CRTM_9_1 =
        0.642787609686539326322643409907263432907559884;
    const FFTZ_DOUBLE CRTM_9_2 =
        0.173648177666930348851716626769314796000375677;
    const FFTZ_DOUBLE CRTM_9_3 =
        0.984807753012208059366743024589523013670643252;
    const FFTZ_DOUBLE CRTM_9_4 =
        0.500000000000000000000000000000000000000000000;
    const FFTZ_DOUBLE CRTM_9_5 =
        0.866025403784438646763723170752936183471402627;
    const FFTZ_DOUBLE CRTM_9_6 =
        0.939692620785908384054109277324975766871890789;
    const FFTZ_DOUBLE CRTM_9_7 =
        0.342020143325668733044099614682259580763083320;

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

    __m128d v_CRTM_9_0 = _mm_set1_pd(CRTM_9_0);
    __m128d v_CRTM_9_1 = _mm_set1_pd(CRTM_9_1);
    __m128d v_CRTM_9_2 = _mm_set1_pd(CRTM_9_2);
    __m128d v_CRTM_9_3 = _mm_set1_pd(CRTM_9_3);
    __m128d v_CRTM_9_4 = _mm_set1_pd(CRTM_9_4);
    __m128d v_CRTM_9_5 = _mm_set1_pd(CRTM_9_5);
    __m128d v_CRTM_9_6 = _mm_set1_pd(CRTM_9_6);
    __m128d v_CRTM_9_7 = _mm_set1_pd(CRTM_9_7);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8;
        __m128d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
                v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
                v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26;
        __m128d v_m0, v_m1, v_m2, v_m3, v_m4, v_m5, v_m6, v_m7, v_m8, v_m9,
                v_m10, v_m11, v_m12, v_m13, v_m14, v_m15, v_m16, v_m17, v_m18,
                v_m19, v_m20, v_m21;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_stride, v_in0, is_contiguous_in);
        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_128_D(curr_in, v_in_stride, v_in1, is_contiguous_in);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_128_D(curr_in, v_in_stride, v_in2, is_contiguous_in);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_128_D(curr_in, v_in_stride, v_in3, is_contiguous_in);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_128_D(curr_in, v_in_stride, v_in4, is_contiguous_in);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_128_D(curr_in, v_in_stride, v_in5, is_contiguous_in);
        // Input point 7: x(6)
        curr_in = in + in_strides[6];
        LDR_128_D(curr_in, v_in_stride, v_in6, is_contiguous_in);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_128_D(curr_in, v_in_stride, v_in7, is_contiguous_in);
        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDR_128_D(curr_in, v_in_stride, v_in8, is_contiguous_in);

        v_s0 = _mm_add_pd(v_in1, v_in8);
        v_s1 = _mm_sub_pd(v_in1, v_in8);
        v_s2 = _mm_add_pd(v_in2, v_in7);
        v_s3 = _mm_sub_pd(v_in2, v_in7);
        v_s4 = _mm_add_pd(v_in3, v_in6);
        v_s5 = _mm_sub_pd(v_in3, v_in6);
        v_s6 = _mm_add_pd(v_in4, v_in5);
        v_s7 = _mm_sub_pd(v_in4, v_in5);

        v_s8 = _mm_add_pd(v_s0, v_s6);
        v_s9 = _mm_add_pd(v_s8, v_s2);
        v_s10 = _mm_sub_pd(v_s1, v_s3);
        v_s11 = _mm_add_pd(v_s10, v_s7);
        v_s12 = _mm_add_pd(v_s9, v_s4);
        v_s13 = _mm_add_pd(v_s12, v_in0);
        v_m0 = _mm_mul_pd(v_CRTM_9_4, v_s4);
        v_m1 = _mm_mul_pd(v_CRTM_9_5, v_s5);
        v_s14 = _mm_sub_pd(v_in0, v_m0);

        v_m2 = _mm_mul_pd(v_CRTM_9_0, v_s0);
        v_m3 = _mm_mul_pd(v_CRTM_9_2, v_s2);
        v_m4 = _mm_mul_pd(v_CRTM_9_6, v_s6);
        v_m5 = _mm_mul_pd(v_CRTM_9_1, v_s1);
        v_m6 = _mm_mul_pd(v_CRTM_9_3, v_s3);
        v_m7 = _mm_mul_pd(v_CRTM_9_7, v_s7);
        v_m8 = _mm_mul_pd(v_CRTM_9_0, v_s2);
        v_m9 = _mm_mul_pd(v_CRTM_9_2, v_s0);
        v_m10 = _mm_mul_pd(v_CRTM_9_6, v_s2);
        v_m11 = _mm_mul_pd(v_CRTM_9_0, v_s6);
        v_m12 = _mm_mul_pd(v_CRTM_9_1, v_s7);
        v_m13 = _mm_mul_pd(v_CRTM_9_3, v_s1);
        v_m14 = _mm_mul_pd(v_CRTM_9_7, v_s3);
        // Output point 1: X(0)
        v_out0 = v_s13;
        STR_128_D(curr_out, v_out_stride, v_out0, is_contiguous_out);

        v_s15 = _mm_add_pd(v_s14, v_m2);
        v_s16 = _mm_sub_pd(v_m3, v_m4);
        // Output point 2: X(1)
        v_out1 = _mm_add_pd(v_s15, v_s16);

        v_s17 = _mm_add_pd(v_m5, v_m6);
        v_s18 = _mm_add_pd(v_m7, v_m1);
        v_s19 = NEGATE_128_D(_mm_add_pd(v_s17, v_s18));
        // Output point 3: X(2)
        v_out2 = v_s19;
        curr_out = out + out_strides[1];
        STRI_2x128_D(curr_out, v_out_stride, v_out1, v_out2);

        v_s20 = _mm_add_pd(v_m11, v_m9);
        v_s21 = _mm_sub_pd(v_s20, v_m10);
        // Output point 4: X(3)
        v_out3 = _mm_add_pd(v_s21, v_s14);

        v_s22 = _mm_add_pd(v_m13, v_m14);
        v_s23 = _mm_sub_pd(v_m1, v_s22);
        // Output point 5: X(4)
        v_out4 = _mm_add_pd(v_s23, v_m12);
        curr_out = out + out_strides[3];
        STRI_2x128_D(curr_out, v_out_stride, v_out3, v_out4);

        v_m15 = _mm_mul_pd(v_CRTM_9_4, v_s9);
        v_s24 = _mm_sub_pd(v_s4, v_m15);
        // Output point 6: X(5)
        v_out5 = _mm_add_pd(v_in0, v_s24);

        // Output point 7: X(6)
        v_m16 = NEGATE_128_D(_mm_mul_pd(v_CRTM_9_5, v_s11));
        v_out6 = v_m16;
        curr_out = out + out_strides[5];
        STRI_2x128_D(curr_out, v_out_stride, v_out5, v_out6);

        v_s25 = _mm_add_pd(v_s14, v_m8);
        v_m17 = _mm_mul_pd(v_CRTM_9_6, v_s0);
        v_s26 = _mm_sub_pd(v_s25, v_m17);
        v_m18 = _mm_mul_pd(v_CRTM_9_2, v_s6);
        // Output point 8: X(7)
        v_out7 = _mm_add_pd(v_s26, v_m18);

        v_m19 = _mm_mul_pd(v_CRTM_9_1, v_s3);
        v_m20 = _mm_mul_pd(v_CRTM_9_3, v_s7);
        v_s15 = _mm_add_pd(v_m19, v_m20);
        v_s16 = _mm_sub_pd(v_s15, v_m1);
        v_m21 = _mm_mul_pd(v_CRTM_9_7, v_s1);
        // Output point 9: X(8)
        v_out8 = _mm_sub_pd(v_s16, v_m21);
        curr_out = out + out_strides[7];
        STRI_2x128_D(curr_out, v_out_stride, v_out7, v_out8);

        in += v_in_stride * NUM_SETS_REAL_128_D;
        out += v_out_stride * NUM_SETS_REAL_128_D;
    }
    // tail cases
    if (n & 1)
    {
        FFTZ_DOUBLE v0, v1, v2, v3, v4, v5, v6, v7, v8;
        FFTZ_DOUBLE s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13,
            s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26,
            s27, s28;
        FFTZ_DOUBLE m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13,
            m14, m15, m16, m17, m18, m19, m20, m21;

        // Input point 1: x(0)
        v0 = *in;
        // Input point 2: x(1)
        v1 = in[in_strides[1]];
        // Input point 3: x(2)
        v2 = in[in_strides[2]];
        // Input point 4: x(3)
        v3 = in[in_strides[3]];
        // Input point 5: x(4)
        v4 = in[in_strides[4]];
        // Input point 6: x(5)
        v5 = in[in_strides[5]];
        // Input point 7: x(6)
        v6 = in[in_strides[6]];
        // Input point 8: x(7)
        v7 = in[in_strides[7]];
        // Input point 9: x(8)
        v8 = in[in_strides[8]];

        s0 = v1 + v8;
        s1 = v1 - v8;
        s2 = v2 + v7;
        s3 = v2 - v7;
        s4 = v3 + v6;
        s5 = v3 - v6;
        s6 = v4 + v5;
        s7 = v4 - v5;

        s8 = s0 + s6;
        s9 = s8 + s2;
        s13 = s1 - s3;
        s10 = s13 + s7;
        s14 = s9 + s4;
        s11 = s14 + v0;

        m0 = CRTM_9_4 * s4;
        m1 = CRTM_9_5 * s5;
        s12 = v0 - m0;

        // Output point 1: X(0)
        *out = s11;

        m2 = CRTM_9_0 * s0;
        m3 = CRTM_9_2 * s2;
        m4 = CRTM_9_6 * s6;
        s15 = s12 + m2;
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
        s20 = s12 + m8;
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
        s24 = v0 + s4;
        // Output point 6: X(5)
        out[out_strides[5]] = s24 - m14;

        m15 = -CRTM_9_5 * s10;
        // Output point 7: X(6)
        out[out_strides[6]] = m15;

        m16 = CRTM_9_0 * s2;
        m17 = CRTM_9_2 * s6;
        m18 = CRTM_9_6 * s0;
        s25 = s12 + m16;
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
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft9avx128_fp64_bwd(FFTZ_VOID *in_real,
                                           FFTZ_VOID *in_imag,
                                           FFTZ_VOID *out_real,
                                           FFTZ_VOID *out_imag, FFTZ_INTP n,
                                           aoclfftz_strides_t *strides,
                                           FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_9_0 =
        0.766044443118978035202392650555416673935832457;
    const FFTZ_DOUBLE CRTM_9_1 =
        0.642787609686539326322643409907263432907559884;
    const FFTZ_DOUBLE CRTM_9_2 =
        0.173648177666930348851716626769314796000375677;
    const FFTZ_DOUBLE CRTM_9_3 =
        0.984807753012208059366743024589523013670643252;
    const FFTZ_DOUBLE CRTM_9_4 =
        0.500000000000000000000000000000000000000000000;
    const FFTZ_DOUBLE CRTM_9_5 =
        0.866025403784438646763723170752936183471402627;
    const FFTZ_DOUBLE CRTM_9_6 =
        2.000000000000000000000000000000000000000000000;
    // Below CRTMs are the product or sum of the above CRTMs, Precomputed
    // to save multiplications on the fly.
    // CRTM_9_7 = CRTM_9_6 * CRTM_9_5
    const FFTZ_DOUBLE CRTM_9_7 =
        1.732050807568877293527446341505872366942805254;
    // CRTM_9_8 = CRTM_9_6 * CRTM_9_5 * CRTM_9_3
    const FFTZ_DOUBLE CRTM_9_8 =
        1.705737063904886419256501927880148143872040592;
    // CRTM_9_9 = CRTM_9_7 * CRTM_9_2
    const FFTZ_DOUBLE CRTM_9_9 =
        0.300767466360870593278543795225003852144476516;
    // CRTM_9_10 = CRTM_9_8 - CRTM_9_0 + CRTM_9_2
    const FFTZ_DOUBLE CRTM_9_10 =
        1.113340798452838732905825904094046265936583812;
    // CRTM_9_11 = CRTM_9_3 + CRTM_9_6 * CRTM_9_2 * CRTM_9_3
    const FFTZ_DOUBLE CRTM_9_11 =
        1.326827896337876792410842639271782594433726619;

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

    __m128d v128_CRTM_9_0  = _mm_set1_pd(CRTM_9_0);
    __m128d v128_CRTM_9_1  = _mm_set1_pd(CRTM_9_1);
    __m128d v128_CRTM_9_2  = _mm_set1_pd(CRTM_9_2);
    __m128d v128_CRTM_9_3  = _mm_set1_pd(CRTM_9_3);
    __m128d v128_CRTM_9_4  = _mm_set1_pd(CRTM_9_4);
    __m128d v128_CRTM_9_5  = _mm_set1_pd(CRTM_9_5);
    __m128d v128_CRTM_9_6  = _mm_set1_pd(CRTM_9_6);
    __m128d v128_CRTM_9_7  = _mm_set1_pd(CRTM_9_7);
    __m128d v128_CRTM_9_8  = _mm_set1_pd(CRTM_9_8);
    __m128d v128_CRTM_9_9  = _mm_set1_pd(CRTM_9_9);
    __m128d v128_CRTM_9_10 = _mm_set1_pd(CRTM_9_10);
    __m128d v128_CRTM_9_11 = _mm_set1_pd(CRTM_9_11);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8;
        __m128d v_m0, v_m1, v_m2, v_m3, v_m4, v_m5, v_m6, v_m7, v_m8, v_m9,
                v_m10, v_m11, v_m12, v_m13, v_m14, v_m15, v_m16, v_m17;
        __m128d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
                v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
                v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27,
                v_s28, v_s29, v_s30, v_s31;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8;

        curr_in = in;
        curr_out = out;

        // Input point 1: X(0)
        LDR_128_D(curr_in, v_in_stride, v_in0, is_contiguous_in);
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

        v_m0 = _mm_mul_pd(v128_CRTM_9_7, v_in6);
        v_s0 = _mm_sub_pd(v_in0, v_in5);
        v_m1 = _mm_mul_pd(v128_CRTM_9_6, v_in5);
        v_s1 = _mm_add_pd(v_in0, v_m1);
        v_s2 = _mm_sub_pd(v_s0, v_m0);
        v_s3 = _mm_add_pd(v_s0, v_m0);
        v_s4 = _mm_add_pd(v_in7, v_in3);
        v_s5 = _mm_sub_pd(v_in7, v_in3);
        v_m2 = _mm_mul_pd(v128_CRTM_9_5, v_s5);
        v_s6 = _mm_add_pd(v_in8, v_in4);
        v_m3 = _mm_mul_pd(v128_CRTM_9_5, v_s6);
        v_s7 = _mm_sub_pd(v_in4, v_in8);
        v_s8 = _mm_add_pd(v_in1, v_s4);
        v_m4 = _mm_mul_pd(v128_CRTM_9_4, v_s7);
        v_s9 = _mm_add_pd(v_in2, v_m4);
        v_s10 = _mm_add_pd(v_m2, v_s9);
        v_s11 = _mm_sub_pd(v_s9, v_m2);
        v_m5 = _mm_mul_pd(v128_CRTM_9_4, v_s4);
        v_s12 = _mm_sub_pd(v_in1, v_m5);
        v_s13 = _mm_sub_pd(v_s12, v_m3);
        v_s14 = _mm_add_pd(v_s12, v_m3);
        v_m6 = _mm_mul_pd(v128_CRTM_9_6, v_s8);
        v_s15 = _mm_add_pd(v_s1, v_m6);
        // Output point 1: x(0)
        v_out0 = v_s15;
        STR_128_D(curr_out, v_out_stride, v_out0, is_contiguous_out);

        v_s16 = _mm_sub_pd(v_s1, v_s8);
        v_s17 = _mm_sub_pd(v_in2, v_s7);
        v_m7 = _mm_mul_pd(v128_CRTM_9_7, v_s17);
        v_s18 = _mm_sub_pd(v_s16, v_m7);
        v_s19 = _mm_add_pd(v_s16, v_m7);
        // Output point 4: x(3)
        v_out3 = v_s18;
        curr_out = out + out_strides[3];
        STR_128_D(curr_out, v_out_stride, v_out3, is_contiguous_out);
        // Output point 7: x(6)
        v_out6 = v_s19;
        curr_out = out + out_strides[6];
        STR_128_D(curr_out, v_out_stride, v_out6, is_contiguous_out);

        v_m8 = _mm_mul_pd(v128_CRTM_9_0, v_s13);
        v_m9 = _mm_mul_pd(v128_CRTM_9_1, v_s10);
        v_s20 = _mm_sub_pd(v_m8, v_m9);
        v_m10 = _mm_mul_pd(v128_CRTM_9_11, v_s10);
        v_m11 = _mm_mul_pd(v128_CRTM_9_10, v_s13);
        v_s21 = _mm_add_pd(v_m11, v_m10);
        v_s22 = _mm_sub_pd(v_s2, v_s20);
        v_m12 = _mm_mul_pd(v128_CRTM_9_6, v_s20);
        v_s23 = _mm_add_pd(v_s2, v_m12);
        // Output point 2: x(1)
        v_out1 = v_s23;
        curr_out = out + out_strides[1];
        STR_128_D(curr_out, v_out_stride, v_out1, is_contiguous_out);

        v_s24 = _mm_add_pd(v_s22, v_s21);
        v_s25 = _mm_sub_pd(v_s22, v_s21);
        // Output point 8: x(7)
        v_out7 = v_s24;
        curr_out = out + out_strides[7];
        STR_128_D(curr_out, v_out_stride, v_out7, is_contiguous_out);
        // Output point 5: x(4)
        v_out4 = v_s25;
        curr_out = out + out_strides[4];
        STR_128_D(curr_out, v_out_stride, v_out4, is_contiguous_out);

        v_m13 = _mm_mul_pd(v128_CRTM_9_8, v_s14);
        v_m14 = _mm_mul_pd(v128_CRTM_9_9, v_s11);
        v_s26 = _mm_add_pd(v_m13, v_m14);
        v_m15 = _mm_mul_pd(v128_CRTM_9_2, v_s14);
        v_m16 = _mm_mul_pd(v128_CRTM_9_3, v_s11);
        v_s27 = _mm_sub_pd(v_m15, v_m16);
        v_s28 = _mm_sub_pd(v_s3, v_s27);
        v_m17 = _mm_mul_pd(v128_CRTM_9_6, v_s27);
        v_s29 = _mm_add_pd(v_s3, v_m17);
        // Output point 3: x(2)
        v_out2 = v_s29;
        curr_out = out + out_strides[2];
        STR_128_D(curr_out, v_out_stride, v_out2, is_contiguous_out);

        v_s30 = _mm_add_pd(v_s28, v_s26);
        v_s31 = _mm_sub_pd(v_s28, v_s26);
        // Output point 9: x(8)
        v_out8 = v_s30;
        curr_out = out + out_strides[8];
        STR_128_D(curr_out, v_out_stride, v_out8, is_contiguous_out);
        // Output point 6: x(5)
        v_out5 = v_s31;
        curr_out = out + out_strides[5];
        STR_128_D(curr_out, v_out_stride, v_out5, is_contiguous_out);

        in += v_in_stride * NUM_SETS_REAL_128_D;
        out += v_out_stride * NUM_SETS_REAL_128_D;
    }
    // tail cases
    if (n & 1)
    {
        FFTZ_DOUBLE v0, v1, v2, v3, v4, v5, v6, v7, v8;
        FFTZ_DOUBLE s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13,
            s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26,
            s27, s28, s29, s30, s31;
        FFTZ_DOUBLE m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13,
            m14, m15, m16, m17;

        // Input point 1: X(0)
        v0 = *in;
        // Input point 2: X(1)
        v1 = in[in_strides[1]];
        // Input point 3: X(2)
        v2 = in[in_strides[2]];
        // Input point 4: X(3)
        v3 = in[in_strides[3]];
        // Input point 5: X(4)
        v4 = in[in_strides[4]];
        // Input point 6: X(5)
        v5 = in[in_strides[5]];
        // Input point 7: X(6)
        v6 = in[in_strides[6]];
        // Input point 8: X(7)
        v7 = in[in_strides[7]];
        // Input point 9: X(8)
        v8 = in[in_strides[8]];

        m0 = CRTM_9_7 * v6;
        s0 = v0 - v5;
        m1 = CRTM_9_6 * v5;
        s1 = v0 + m1;
        s2 = s0 - m0;
        s3 = s0 + m0;
        s4 = v7 + v3;
        s5 = v7 - v3;
        m2 = CRTM_9_5 * s5;
        s6 = v8 + v4;
        m3 = CRTM_9_5 * s6;
        s7 = v4 - v8;
        s8 = v1 + s4;
        m4 = CRTM_9_4 * s7;
        s9 = v2 + m4;
        s10 = m2 + s9;
        s11 = s9 - m2;
        m5 = CRTM_9_4 * s4;
        s12 = v1 - m5;
        s13 = s12 - m3;
        s14 = s12 + m3;
        m6 = CRTM_9_6 * s8;
        s15 = s1 + m6;
        // Output point 1: x(0)
        *out = s15;

        s16 = s1 - s8;
        s17 = v2 - s7;
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
    }

    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hc_rfft9avx128(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hc_rfft9avx128_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft9avx128_fp64_fwd;
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
            return r2hc_rfft9avx128_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft9avx128_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
