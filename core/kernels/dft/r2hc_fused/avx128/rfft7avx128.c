// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft7avx128.c
 *
 *  @brief Radix-7 r2hc_fused Real-FFT kernel with AVX-128 operations using x86
 *  SIMD intrinsics
 *
 *  This file contains the DIT radix-7 real-to-halfcomplex fused of two
 *  different implementations (Standard DFT and Shifted DFT that differs in DFT
 *  weight matrix) using x86 SIMD intrinsics for single-precision and
 *  double-precision inputs
 *
 *  @author Jeya R
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/r2hc_simd_avx128.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                {{{0, 36, 48, 88, 60, 1},
                                                  {0, 38, 48, 88, 66, 1}},
                                                 {{0, 36, 48, 44, 12, 1},
                                                  {0, 38, 48, 44, 12, 1}}};

ops_cycles_t get_ops_cnt_r2hcf_rfft7avx128(FFTZ_UINT8 precision,
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

static FFTZ_VOID r2hcf_rfft7avx128_fp32_fwd(FFTZ_VOID *in_real,
                                            FFTZ_VOID *in_imag,
                                            FFTZ_VOID *out_real,
                                            FFTZ_VOID *out_imag, FFTZ_INTP n,
                                            aoclfftz_strides_t *strides,
                                            FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_7_1 =
        0.900968867902419126236102319507445051165919162f;
    const FFTZ_FLOAT CRTM_7_2 =
        0.433883739117558120475768332848358754609990728f;
    const FFTZ_FLOAT CRTM_7_3 =
        0.623489801858733530525004884004239810632274731f;
    const FFTZ_FLOAT CRTM_7_4 =
        0.781831482468029808708444526674057750232334519f;
    const FFTZ_FLOAT CRTM_7_5 =
        0.222520933956314404288902564496794759466355569f;
    const FFTZ_FLOAT CRTM_7_6 =
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
    FFTZ_INTP N = n / NUM_SETS_REAL_128_S;
    FFTZ_FLOAT *curr_in, *curr_out;

    __m128 v_CRTM_7_1 = _mm_broadcast_ss(&CRTM_7_1);
    __m128 v_CRTM_7_2 = _mm_broadcast_ss(&CRTM_7_2);
    __m128 v_CRTM_7_3 = _mm_broadcast_ss(&CRTM_7_3);
    __m128 v_CRTM_7_4 = _mm_broadcast_ss(&CRTM_7_4);
    __m128 v_CRTM_7_5 = _mm_broadcast_ss(&CRTM_7_5);
    __m128 v_CRTM_7_6 = _mm_broadcast_ss(&CRTM_7_6);

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6;
        __m128 av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8, av_s9,
               av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16, av_s17;
        __m128 av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8, av_t9,
               av_t10, av_t11, av_t12, av_t13, av_t14, av_t15, av_t16, av_t17,
               av_t18;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13;

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

        av_s1 = _mm_add_ps(av_in6, av_in1);
        av_s2 = _mm_sub_ps(av_in6, av_in1);
        av_s3 = _mm_add_ps(av_in5, av_in2);
        av_s4 = _mm_sub_ps(av_in5, av_in2);
        av_s5 = _mm_add_ps(av_in4, av_in3);
        av_s6 = _mm_sub_ps(av_in4, av_in3);

        av_s7 = _mm_add_ps(av_in0, av_s1);
        av_s8 = _mm_add_ps(av_s3, av_s5);

        av_t1 = _mm_mul_ps(v_CRTM_7_1, av_s5);
        av_t2 = _mm_mul_ps(v_CRTM_7_3, av_s1);
        av_t3 = _mm_mul_ps(v_CRTM_7_5, av_s3);
        av_s9 = _mm_sub_ps(av_in0, av_t1);

        av_s10 = _mm_sub_ps(av_t2, av_t3);
        av_t4 = _mm_mul_ps(v_CRTM_7_2, av_s6);
        av_t5 = _mm_mul_ps(v_CRTM_7_4, av_s2);

        av_t6 = _mm_mul_ps(v_CRTM_7_6, av_s4);
        av_s11 = _mm_add_ps(av_t4, av_t5);

        av_t7 = _mm_mul_ps(v_CRTM_7_1, av_s3);
        av_t8 = _mm_mul_ps(v_CRTM_7_3, av_s5);
        av_t9 = _mm_mul_ps(v_CRTM_7_5, av_s1);

        av_s12 = _mm_sub_ps(av_in0, av_t7);
        av_s13 = _mm_sub_ps(av_t8, av_t9);
        av_t10 = _mm_mul_ps(v_CRTM_7_2, av_s4);
        av_t11 = _mm_mul_ps(v_CRTM_7_4, av_s6);

        av_t12 = _mm_mul_ps(v_CRTM_7_6, av_s2);
        av_s14 = _mm_add_ps(av_t10, av_t11);
        av_t13 = _mm_mul_ps(v_CRTM_7_1, av_s1);
        av_t14 = _mm_mul_ps(v_CRTM_7_3, av_s3);
        av_t15 = _mm_mul_ps(v_CRTM_7_5, av_s5);

        av_s15 = _mm_sub_ps(av_in0, av_t13);
        av_s16 = _mm_sub_ps(av_t14, av_t15);
        av_t16 = _mm_mul_ps(v_CRTM_7_2, av_s2);
        av_t17 = _mm_mul_ps(v_CRTM_7_4, av_s4);
        av_t18 = _mm_mul_ps(v_CRTM_7_6, av_s6);
        av_s17 = _mm_sub_ps(av_t16, av_t17);

        // Output pt 1: X(0)
        v_out0 = _mm_add_ps(av_s7, av_s8);
        STR_128_S(curr_out, v_out_stride, v_out0);
        // Output pt 4: X(3) & Output pt 5: X(4)
        curr_out = out + out_strides[3];
        v_out3 = _mm_add_ps(av_s9, av_s10);
        v_out4 = _mm_add_ps(av_t6, av_s11);
        STRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);
        // Output pt 8: X(7) & Output pt 9: X(8)
        curr_out = out + out_strides[7];
        v_out7 = _mm_add_ps(av_s12, av_s13);
        v_out8 = _mm_sub_ps(av_t12, av_s14);
        STRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);
        // Output pt 12: X(11) & Output pt 13: X(12)
        curr_out = out + out_strides[11];
        v_out11 = _mm_add_ps(av_s15, av_s16);
        v_out12 = _mm_add_ps(av_s17, av_t18);
        STRI_2x128_S(curr_out, v_out_stride, v_out11, v_out12);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6;
        __m128 bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
               bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17;
        __m128 bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
               bv_t10, bv_t11, bv_t12, bv_t13, bv_t14, bv_t15, bv_t16, bv_t17,
               bv_t18;

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

        bv_s1 = _mm_add_ps(bv_in6, bv_in1);
        bv_s2 = _mm_sub_ps(bv_in6, bv_in1);
        bv_s3 = _mm_add_ps(bv_in5, bv_in2);
        bv_s4 = _mm_sub_ps(bv_in5, bv_in2);
        bv_s5 = _mm_add_ps(bv_in4, bv_in3);
        bv_s6 = _mm_sub_ps(bv_in4, bv_in3);

        bv_t1 = _mm_mul_ps(v_CRTM_7_1, bv_s2);
        bv_t2 = _mm_mul_ps(v_CRTM_7_3, bv_s4);

        bv_t3 = _mm_mul_ps(v_CRTM_7_5, bv_s6);
        bv_s7 = _mm_sub_ps(bv_in0, bv_t1);
        bv_s8 = _mm_add_ps(bv_t2, bv_t3);
        bv_t4 = _mm_mul_ps(v_CRTM_7_2, bv_s1);

        bv_t5 = _mm_mul_ps(v_CRTM_7_4, bv_s3);
        bv_t6 = _mm_mul_ps(v_CRTM_7_6, bv_s5);
        bv_s9 = NEGATE_128_S(_mm_add_ps(bv_t4, bv_t5));

        bv_t7 = _mm_mul_ps(v_CRTM_7_1, bv_s4);
        bv_t8 = _mm_mul_ps(v_CRTM_7_3, bv_s6);
        bv_t9 = _mm_mul_ps(v_CRTM_7_5, bv_s2);
        bv_s10 = _mm_add_ps(bv_in0, bv_t7);
        bv_s11 = _mm_sub_ps(bv_t8, bv_t9);

        bv_t10 = _mm_mul_ps(v_CRTM_7_2, bv_s3);
        bv_t11 = _mm_mul_ps(v_CRTM_7_4, bv_s5);

        bv_t12 = _mm_mul_ps(v_CRTM_7_6, bv_s1);
        bv_s12 = _mm_sub_ps(bv_t11, bv_t10);

        bv_t13 = _mm_mul_ps(v_CRTM_7_1, bv_s6);
        bv_t14 = _mm_mul_ps(v_CRTM_7_3, bv_s2);
        bv_t15 = _mm_mul_ps(v_CRTM_7_5, bv_s4);
        bv_s13 = _mm_sub_ps(bv_in0, bv_t13);
        bv_s14 = _mm_add_ps(bv_t14, bv_t15);

        bv_t16 = _mm_mul_ps(v_CRTM_7_2, bv_s5);
        bv_t17 = _mm_mul_ps(v_CRTM_7_4, bv_s1);
        bv_t18 = _mm_mul_ps(v_CRTM_7_6, bv_s3);
        bv_s15 = _mm_add_ps(bv_t16, bv_t17);
        bv_s16 = _mm_add_ps(bv_in0, bv_s2);
        bv_s17 = _mm_sub_ps(bv_s6, bv_s4);

        // Output pt 2: X(1) & Output pt 3: X(2)
        curr_out = out + out_strides[1];
        v_out1 = _mm_sub_ps(bv_s7, bv_s8);
        v_out2 = _mm_sub_ps(bv_s9, bv_t6);
        STRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output pt 6: X(5) & Output pt 7: X(6)
        curr_out = out + out_strides[5];
        v_out5 = _mm_add_ps(bv_s10, bv_s11);
        v_out6 = _mm_sub_ps(bv_s12, bv_t12);
        STRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);
        // Output pt 10: X(9) & Output pt 11: X(10)
        curr_out = out + out_strides[9];
        v_out9 = _mm_add_ps(bv_s13, bv_s14);
        v_out10 = _mm_sub_ps(bv_t18, bv_s15);
        STRI_2x128_S(curr_out, v_out_stride, v_out9, v_out10);
        // Output pt 14: X(13)
        curr_out = out + out_strides[13];
        v_out13 = _mm_add_ps(bv_s16, bv_s17);
        STR_128_S(curr_out, v_out_stride, v_out13);

        in += v_in_stride * NUM_SETS_REAL_128_S;
        out += v_out_stride * NUM_SETS_REAL_128_S;
    }
    // tail cases
    if (n & 2)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6;
        __m128 av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8, av_s9,
               av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16, av_s17;
        __m128 av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8, av_t9,
               av_t10, av_t11, av_t12, av_t13, av_t14, av_t15, av_t16, av_t17,
               av_t18;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13;

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

        av_s1 = _mm_add_ps(av_in6, av_in1);
        av_s2 = _mm_sub_ps(av_in6, av_in1);
        av_s3 = _mm_add_ps(av_in5, av_in2);
        av_s4 = _mm_sub_ps(av_in5, av_in2);
        av_s5 = _mm_add_ps(av_in4, av_in3);
        av_s6 = _mm_sub_ps(av_in4, av_in3);

        av_s7 = _mm_add_ps(av_in0, av_s1);
        av_s8 = _mm_add_ps(av_s3, av_s5);

        av_t1 = _mm_mul_ps(v_CRTM_7_1, av_s5);
        av_t2 = _mm_mul_ps(v_CRTM_7_3, av_s1);
        av_t3 = _mm_mul_ps(v_CRTM_7_5, av_s3);
        av_s9 = _mm_sub_ps(av_in0, av_t1);

        av_s10 = _mm_sub_ps(av_t2, av_t3);
        av_t4 = _mm_mul_ps(v_CRTM_7_2, av_s6);
        av_t5 = _mm_mul_ps(v_CRTM_7_4, av_s2);

        av_t6 = _mm_mul_ps(v_CRTM_7_6, av_s4);
        av_s11 = _mm_add_ps(av_t4, av_t5);

        av_t7 = _mm_mul_ps(v_CRTM_7_1, av_s3);
        av_t8 = _mm_mul_ps(v_CRTM_7_3, av_s5);
        av_t9 = _mm_mul_ps(v_CRTM_7_5, av_s1);

        av_s12 = _mm_sub_ps(av_in0, av_t7);
        av_s13 = _mm_sub_ps(av_t8, av_t9);
        av_t10 = _mm_mul_ps(v_CRTM_7_2, av_s4);
        av_t11 = _mm_mul_ps(v_CRTM_7_4, av_s6);

        av_t12 = _mm_mul_ps(v_CRTM_7_6, av_s2);
        av_s14 = _mm_add_ps(av_t10, av_t11);
        av_t13 = _mm_mul_ps(v_CRTM_7_1, av_s1);
        av_t14 = _mm_mul_ps(v_CRTM_7_3, av_s3);
        av_t15 = _mm_mul_ps(v_CRTM_7_5, av_s5);

        av_s15 = _mm_sub_ps(av_in0, av_t13);
        av_s16 = _mm_sub_ps(av_t14, av_t15);
        av_t16 = _mm_mul_ps(v_CRTM_7_2, av_s2);
        av_t17 = _mm_mul_ps(v_CRTM_7_4, av_s4);
        av_t18 = _mm_mul_ps(v_CRTM_7_6, av_s6);
        av_s17 = _mm_sub_ps(av_t16, av_t17);

        // Output pt 1: X(0)
        v_out0 = _mm_add_ps(av_s7, av_s8);
        STHR_128_S(curr_out, v_out_stride, v_out0);
        // Output pt 4: X(3) & Output pt 5: X(4)
        curr_out = out + out_strides[3];
        v_out3 = _mm_add_ps(av_s9, av_s10);
        v_out4 = _mm_add_ps(av_t6, av_s11);
        STHRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);
        // Output pt 8: X(7) & Output pt 9: X(8)
        curr_out = out + out_strides[7];
        v_out7 = _mm_add_ps(av_s12, av_s13);
        v_out8 = _mm_sub_ps(av_t12, av_s14);
        STHRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);
        // Output pt 12: X(11) & Output pt 13: X(12)
        curr_out = out + out_strides[11];
        v_out11 = _mm_add_ps(av_s15, av_s16);
        v_out12 = _mm_add_ps(av_s17, av_t18);
        STHRI_2x128_S(curr_out, v_out_stride, v_out11, v_out12);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6;
        __m128 bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
               bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17;
        __m128 bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
               bv_t10, bv_t11, bv_t12, bv_t13, bv_t14, bv_t15, bv_t16, bv_t17,
               bv_t18;

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

        bv_s1 = _mm_add_ps(bv_in6, bv_in1);
        bv_s2 = _mm_sub_ps(bv_in6, bv_in1);
        bv_s3 = _mm_add_ps(bv_in5, bv_in2);
        bv_s4 = _mm_sub_ps(bv_in5, bv_in2);
        bv_s5 = _mm_add_ps(bv_in4, bv_in3);
        bv_s6 = _mm_sub_ps(bv_in4, bv_in3);

        bv_t1 = _mm_mul_ps(v_CRTM_7_1, bv_s2);
        bv_t2 = _mm_mul_ps(v_CRTM_7_3, bv_s4);

        bv_t3 = _mm_mul_ps(v_CRTM_7_5, bv_s6);
        bv_s7 = _mm_sub_ps(bv_in0, bv_t1);
        bv_s8 = _mm_add_ps(bv_t2, bv_t3);
        bv_t4 = _mm_mul_ps(v_CRTM_7_2, bv_s1);

        bv_t5 = _mm_mul_ps(v_CRTM_7_4, bv_s3);
        bv_t6 = _mm_mul_ps(v_CRTM_7_6, bv_s5);
        bv_s9 = NEGATE_128_S(_mm_add_ps(bv_t4, bv_t5));

        bv_t7 = _mm_mul_ps(v_CRTM_7_1, bv_s4);
        bv_t8 = _mm_mul_ps(v_CRTM_7_3, bv_s6);
        bv_t9 = _mm_mul_ps(v_CRTM_7_5, bv_s2);
        bv_s10 = _mm_add_ps(bv_in0, bv_t7);
        bv_s11 = _mm_sub_ps(bv_t8, bv_t9);

        bv_t10 = _mm_mul_ps(v_CRTM_7_2, bv_s3);
        bv_t11 = _mm_mul_ps(v_CRTM_7_4, bv_s5);

        bv_t12 = _mm_mul_ps(v_CRTM_7_6, bv_s1);
        bv_s12 = _mm_sub_ps(bv_t11, bv_t10);

        bv_t13 = _mm_mul_ps(v_CRTM_7_1, bv_s6);
        bv_t14 = _mm_mul_ps(v_CRTM_7_3, bv_s2);
        bv_t15 = _mm_mul_ps(v_CRTM_7_5, bv_s4);
        bv_s13 = _mm_sub_ps(bv_in0, bv_t13);
        bv_s14 = _mm_add_ps(bv_t14, bv_t15);

        bv_t16 = _mm_mul_ps(v_CRTM_7_2, bv_s5);
        bv_t17 = _mm_mul_ps(v_CRTM_7_4, bv_s1);
        bv_t18 = _mm_mul_ps(v_CRTM_7_6, bv_s3);
        bv_s15 = _mm_add_ps(bv_t16, bv_t17);
        bv_s16 = _mm_add_ps(bv_in0, bv_s2);
        bv_s17 = _mm_sub_ps(bv_s6, bv_s4);

        // Output pt 2: X(1) & Output pt 3: X(2)
        curr_out = out + out_strides[1];
        v_out1 = _mm_sub_ps(bv_s7, bv_s8);
        v_out2 = _mm_sub_ps(bv_s9, bv_t6);
        STHRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output pt 6: X(5) & Output pt 7: X(6)
        curr_out = out + out_strides[5];
        v_out5 = _mm_add_ps(bv_s10, bv_s11);
        v_out6 = _mm_sub_ps(bv_s12, bv_t12);
        STHRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);
        // Output pt 10: X(9) & Output pt 11: X(10)
        curr_out = out + out_strides[9];
        v_out9 = _mm_add_ps(bv_s13, bv_s14);
        v_out10 = _mm_sub_ps(bv_t18, bv_s15);
        STHRI_2x128_S(curr_out, v_out_stride, v_out9, v_out10);
        // Output pt 14: X(13)
        curr_out = out + out_strides[13];
        v_out13 = _mm_add_ps(bv_s16, bv_s17);
        STHR_128_S(curr_out, v_out_stride, v_out13);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        /* Standard DFT */
        FFTZ_FLOAT a_in0, a_in1, a_in2, a_in3, a_in4, a_in5, a_in6;
        FFTZ_FLOAT a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_t1, a_t2,
            a_t3, a_t4, a_t5, a_s9, a_s10, a_t6, a_s11, a_t7, a_t8, a_t9, a_s12,
            a_s13, a_t10, a_t11, a_t12, a_t13, a_t14, a_t15, a_s14, a_s15,
            a_s16, a_t16, a_t17, a_t18, a_s17;

        a_in0 = *in;                  // Input point 1: x(0)
        a_in1 = in[in_strides[2]];    // Input point 3: x(2)
        a_in2 = in[in_strides[4]];    // Input point 5: x(4)
        a_in3 = in[in_strides[6]];    // Input point 7: x(6)
        a_in4 = in[in_strides[8]];    // Input point 9: x(8)
        a_in5 = in[in_strides[10]];   // Input point 11: x(10)
        a_in6 = in[in_strides[12]];   // Input point 13: x(12)

        a_s1 = a_in6 + a_in1;
        a_s2 = a_in6 - a_in1;
        a_s3 = a_in5 + a_in2;
        a_s4 = a_in5 - a_in2;
        a_s5 = a_in4 + a_in3;
        a_s6 = a_in4 - a_in3;
        a_s7 = a_in0 + a_s1;
        a_s8 = a_s3 + a_s5;

        a_t1 = CRTM_7_1 * a_s5;
        a_t2 = CRTM_7_3 * a_s1;
        a_t3 = CRTM_7_5 * a_s3;
        a_t4 = CRTM_7_2 * a_s6;
        a_t5 = CRTM_7_4 * a_s2;
        a_s9 = a_in0 - a_t1;
        a_s10 = a_t2 - a_t3;

        a_t6 = CRTM_7_6 * a_s4;
        a_s11 = a_t4 + a_t5;
        a_t7 = CRTM_7_1 * a_s3;
        a_t8 = CRTM_7_3 * a_s5;
        a_t9 = CRTM_7_5 * a_s1;

        a_s12 = a_in0 - a_t7;
        a_s13 = a_t8 - a_t9;
        a_t10 = CRTM_7_2 * a_s4;
        a_t11 = CRTM_7_4 * a_s6;

        a_t12 = CRTM_7_6 * a_s2;
        a_t13 = CRTM_7_1 * a_s1;
        a_t14 = CRTM_7_3 * a_s3;
        a_t15 = CRTM_7_5 * a_s5;
        a_s14 = a_t10 + a_t11;

        a_s15 = a_in0 - a_t13;
        a_s16 = a_t14 - a_t15;
        a_t16 = CRTM_7_2 * a_s2;
        a_t17 = CRTM_7_4 * a_s4;
        a_t18 = CRTM_7_6 * a_s6;
        a_s17 = a_t16 - a_t17;

        *out = a_s7 + a_s8;                       // Output pt 1: X(0)
        out[out_strides[3]]  = a_s9 + a_s10;      // Output pt 4: X(3)
        out[out_strides[4]]  = a_t6 + a_s11;      // Output pt 5: X(4)
        out[out_strides[7]]  = a_s12 + a_s13;     // Output pt 8: X(7)
        out[out_strides[8]]  = a_t12 - a_s14;     // Output pt 9: X(8)
        out[out_strides[11]] = a_s15 + a_s16;     // Output pt 12: X(11)
        out[out_strides[12]] = a_s17 + a_t18;     // Output pt 13: X(12)

        /* Shifted DFT */
        FFTZ_FLOAT b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6;
        FFTZ_FLOAT b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_t1, b_t2, b_t3, b_s7,
              b_s8, b_t4, b_t5, b_t6, b_s9, b_t7, b_t8, b_t9, b_s10, b_s11,
              b_t10, b_t11, b_t12, b_s12, b_t13, b_t14, b_t15, b_s13, b_s14,
              b_t16, b_t17, b_t18, b_s15, b_s16, b_s17;

        b_in0 = in[in_strides[1]];    // Input point 2: x(1)
        b_in1 = in[in_strides[3]];    // Input point 4: x(3)
        b_in2 = in[in_strides[5]];    // Input point 6: x(5)
        b_in3 = in[in_strides[7]];    // Input point 8: x(7)
        b_in4 = in[in_strides[9]];    // Input point 10: x(9)
        b_in5 = in[in_strides[11]];   // Input point 12: x(11)
        b_in6 = in[in_strides[13]];   // Input point 14: x(13)

        b_s1 = b_in6 + b_in1;
        b_s2 = b_in6 - b_in1;
        b_s3 = b_in5 + b_in2;
        b_s4 = b_in5 - b_in2;
        b_s5 = b_in4 + b_in3;
        b_s6 = b_in4 - b_in3;

        b_t1 = CRTM_7_1 * b_s2;
        b_t2 = CRTM_7_3 * b_s4;

        b_t3 = CRTM_7_5 * b_s6;
        b_s7  = b_in0 - b_t1;
        b_s8 = b_t2 + b_t3;
        b_t4 = CRTM_7_2 * b_s1;
        b_t5 = CRTM_7_4 * b_s3;
        b_t6 = CRTM_7_6 * b_s5;
        b_s9 = -b_t4 - b_t5;

        b_t7 = CRTM_7_1 * b_s4;
        b_t8 = CRTM_7_3 * b_s6;

        b_t9 = CRTM_7_5 * b_s2;
        b_s10 = b_in0 + b_t7;
        b_t10 = CRTM_7_2 * b_s3;
        b_s11 = b_t8 - b_t9;
        b_t11 = CRTM_7_4 * b_s5;
        b_t12 = CRTM_7_6 * b_s1;
        b_s12 = b_t11 - b_t10;

        b_t13 = CRTM_7_1 * b_s6;
        b_t14 = CRTM_7_3 * b_s2;
        b_t15 = CRTM_7_5 * b_s4;
        b_s13 = b_in0 - b_t13;
        b_s14 = b_t14 + b_t15;

        b_t16 = CRTM_7_2 * b_s5;
        b_t17 = CRTM_7_4 * b_s1;
        b_t18 = CRTM_7_6 * b_s3;
        b_s15 = b_t16 + b_t17;
        b_s16 = b_in0 + b_s2;
        b_s17 = b_s6 - b_s4;

        out[out_strides[1]]  = b_s7 - b_s8;      // Output pt 2: X(1)
        out[out_strides[2]]  = b_s9 - b_t6;      // Output pt 3: X(2)
        out[out_strides[5]]  = b_s10 + b_s11;    // Output pt 6: X(5)
        out[out_strides[6]]  = b_s12 - b_t12;    // Output pt 7: X(6)
        out[out_strides[9]]  = b_s13 + b_s14;    // Output pt 10: X(9)
        out[out_strides[10]] = b_t18 - b_s15;    // Output pt 11: X(10)
        out[out_strides[13]] = b_s16 + b_s17;    // Output pt 14: X(13)
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft7avx128_fp32_bwd(FFTZ_VOID *in_real,
                                            FFTZ_VOID *in_imag,
                                            FFTZ_VOID *out_real,
                                            FFTZ_VOID *out_imag, FFTZ_INTP n,
                                            aoclfftz_strides_t *strides,
                                            FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_7_1 =
        1.801937735804838252472204639014890102331838324f;
    const FFTZ_FLOAT CRTM_7_2 =
        0.867767478235116240951536665696717509219981456f;
    const FFTZ_FLOAT CRTM_7_3 =
        1.246979603717467061050009768008479621264549462f;
    const FFTZ_FLOAT CRTM_7_4 =
        1.563662964936059617416889053348115500464669038f;
    const FFTZ_FLOAT CRTM_7_5 =
        0.445041867912628808577805128993589518932711138f;
    const FFTZ_FLOAT CRTM_7_6 =
        1.949855824363647214036263365987862434465571602f;
    const FFTZ_FLOAT CRTM_7_7 =
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
    FFTZ_INTP N = n / NUM_SETS_REAL_128_S;
    FFTZ_FLOAT *curr_in, *curr_out;

    __m128 v_CRTM_7_1 = _mm_broadcast_ss(&CRTM_7_1);
    __m128 v_CRTM_7_2 = _mm_broadcast_ss(&CRTM_7_2);
    __m128 v_CRTM_7_3 = _mm_broadcast_ss(&CRTM_7_3);
    __m128 v_CRTM_7_4 = _mm_broadcast_ss(&CRTM_7_4);
    __m128 v_CRTM_7_5 = _mm_broadcast_ss(&CRTM_7_5);
    __m128 v_CRTM_7_6 = _mm_broadcast_ss(&CRTM_7_6);
    __m128 v_CRTM_7_7 = _mm_broadcast_ss(&CRTM_7_7);

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6;
        __m128 av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8, av_s9,
               av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16, av_s17;
        __m128 av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8, av_t9,
               av_t10, av_t11, av_t12, av_t13, av_t14, av_t15, av_t16, av_t17,
               av_t18, av_t19;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13;

        curr_in = in;
        curr_out = out;

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

        av_t1 = _mm_mul_ps(v_CRTM_7_1, av_in5);
        av_t2 = _mm_mul_ps(v_CRTM_7_3, av_in1);
        av_t3 = _mm_mul_ps(v_CRTM_7_5, av_in3);
        av_s1 = _mm_sub_ps(av_in0, av_t1);
        av_s2 = _mm_sub_ps(av_t2, av_t3);
        av_s3 = _mm_add_ps(av_s1, av_s2);

        av_t4 = _mm_mul_ps(v_CRTM_7_2, av_in6);
        av_t5 = _mm_mul_ps(v_CRTM_7_4, av_in2);
        av_t6 = _mm_mul_ps(v_CRTM_7_6, av_in4);
        av_s4 = _mm_add_ps(av_t4, av_t5);
        av_s5 = _mm_add_ps(av_s4, av_t6);

        av_t7 = _mm_mul_ps(v_CRTM_7_1, av_in3);
        av_t8 = _mm_mul_ps(v_CRTM_7_3, av_in5);
        av_t9 = _mm_mul_ps(v_CRTM_7_5, av_in1);
        av_s6 = _mm_sub_ps(av_in0, av_t7);
        av_s7 = _mm_sub_ps(av_t8, av_t9);
        av_s8 = _mm_add_ps(av_s6, av_s7);

        av_t10 = _mm_mul_ps(v_CRTM_7_2, av_in4);
        av_t11 = _mm_mul_ps(v_CRTM_7_4, av_in6);
        av_t12 = _mm_mul_ps(v_CRTM_7_6, av_in2);
        av_s9 = _mm_add_ps(av_t10, av_t11);
        av_s10 = _mm_sub_ps(av_t12, av_s9);

        av_t13 = _mm_mul_ps(v_CRTM_7_1, av_in1);
        av_t14 = _mm_mul_ps(v_CRTM_7_3, av_in3);
        av_t15 = _mm_mul_ps(v_CRTM_7_5, av_in5);
        av_s11 = _mm_sub_ps(av_in0, av_t13);
        av_s12 = _mm_sub_ps(av_t14, av_t15);
        av_s13 = _mm_add_ps(av_s11, av_s12);

        av_t16 = _mm_mul_ps(v_CRTM_7_2, av_in2);
        av_t17 = _mm_mul_ps(v_CRTM_7_4, av_in4);
        av_t18 = _mm_mul_ps(v_CRTM_7_6, av_in6);
        av_s14 = _mm_sub_ps(av_t16, av_t17);
        av_s15 = _mm_add_ps(av_s14, av_t18);

        av_s16 = _mm_add_ps(av_in1, av_in3);
        av_s17 = _mm_add_ps(av_s16, av_in5);
        av_t19 = _mm_mul_ps(v_CRTM_7_7, av_s17);

        // Output pt 1: X(0)
        v_out0 = _mm_add_ps(av_in0, av_t19);
        STR_128_S(curr_out, v_out_stride, v_out0);
        // Output pt 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm_sub_ps(av_s3, av_s5);
        STR_128_S(curr_out, v_out_stride, v_out2);
        // Output pt 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm_sub_ps(av_s8, av_s10);
        STR_128_S(curr_out, v_out_stride, v_out4);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm_sub_ps(av_s13, av_s15);
        STR_128_S(curr_out, v_out_stride, v_out6);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm_add_ps(av_s13, av_s15);
        STR_128_S(curr_out, v_out_stride, v_out8);
        // Output pt 11: X(10)
        curr_out = out + out_strides[10];
        v_out10 = _mm_add_ps(av_s8, av_s10);
        STR_128_S(curr_out, v_out_stride, v_out10);
        // Output pt 13: X(12)
        curr_out = out + out_strides[12];
        v_out12 = _mm_add_ps(av_s3, av_s5);
        STR_128_S(curr_out, v_out_stride, v_out12);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6;
        __m128 bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
               bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17;
        __m128 bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
               bv_t10, bv_t11, bv_t12, bv_t13, bv_t14, bv_t15, bv_t16, bv_t17,
               bv_t18, bv_t19;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in4, bv_in5);
        // Input point 14: x(13)
        curr_in = in + in_strides[13];
        LDR_128_S(curr_in, v_in_stride, bv_in6);

        bv_t1 = _mm_mul_ps(v_CRTM_7_1, bv_in0);
        bv_t2 = _mm_mul_ps(v_CRTM_7_3, bv_in4);
        bv_t3 = _mm_mul_ps(v_CRTM_7_5, bv_in2);
        bv_s1 = _mm_sub_ps(bv_in6, bv_t1);
        bv_s2 = _mm_sub_ps(bv_t2, bv_t3);
        bv_s3 = _mm_add_ps(bv_s1, bv_s2);

        bv_t4 = _mm_mul_ps(v_CRTM_7_2, bv_in1);
        bv_t5 = _mm_mul_ps(v_CRTM_7_4, bv_in5);
        bv_t6 = _mm_mul_ps(v_CRTM_7_6, bv_in3);
        bv_s4 = _mm_add_ps(bv_t4, bv_t5);
        bv_s5 = _mm_add_ps(bv_s4, bv_t6);

        bv_t7 = _mm_mul_ps(v_CRTM_7_1, bv_in2);
        bv_t8 = _mm_mul_ps(v_CRTM_7_3, bv_in0);
        bv_t9 = _mm_mul_ps(v_CRTM_7_5, bv_in4);
        bv_s6 = _mm_sub_ps(bv_in6, bv_t7);
        bv_s7 = _mm_sub_ps(bv_t8, bv_t9);
        bv_s8 = _mm_add_ps(bv_s6, bv_s7);

        bv_t10 = _mm_mul_ps(v_CRTM_7_2, bv_in3);
        bv_t11 = _mm_mul_ps(v_CRTM_7_4, bv_in1);
        bv_t12 = _mm_mul_ps(v_CRTM_7_6, bv_in5);
        bv_s9 = _mm_add_ps(bv_t10, bv_t11);
        bv_s10 = _mm_sub_ps(bv_t12, bv_s9);

        bv_t13 = _mm_mul_ps(v_CRTM_7_1, bv_in4);
        bv_t14 = _mm_mul_ps(v_CRTM_7_3, bv_in2);
        bv_t15 = _mm_mul_ps(v_CRTM_7_5, bv_in0);
        bv_s11 = _mm_sub_ps(bv_in6, bv_t13);
        bv_s12 = _mm_sub_ps(bv_t14, bv_t15);
        bv_s13 = _mm_add_ps(bv_s11, bv_s12);

        bv_t16 = _mm_mul_ps(v_CRTM_7_2, bv_in5);
        bv_t17 = _mm_mul_ps(v_CRTM_7_4, bv_in3);
        bv_t18 = _mm_mul_ps(v_CRTM_7_6, bv_in1);
        bv_s14 = _mm_sub_ps(bv_t17, bv_t16);
        bv_s15 = _mm_sub_ps(bv_s14, bv_t18);

        bv_s16 = _mm_add_ps(bv_in0, bv_in2);
        bv_s17 = _mm_add_ps(bv_s16, bv_in4);
        bv_t19 = _mm_mul_ps(v_CRTM_7_7, bv_s17);
        // Output pt 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm_add_ps(bv_in6, bv_t19);
        STR_128_S(curr_out, v_out_stride, v_out1);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = NEGATE_128_S(_mm_add_ps(bv_s3, bv_s5));
        STR_128_S(curr_out, v_out_stride, v_out3);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm_add_ps(bv_s8, bv_s10);
        STR_128_S(curr_out, v_out_stride, v_out5);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = _mm_sub_ps(bv_s15, bv_s13);
        STR_128_S(curr_out, v_out_stride, v_out7);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm_add_ps(bv_s13, bv_s15);
        STR_128_S(curr_out, v_out_stride, v_out9);
        // Output pt 12: X(11)
        curr_out = out + out_strides[11];
        v_out11 = _mm_sub_ps(bv_s10, bv_s8);
        STR_128_S(curr_out, v_out_stride, v_out11);
        // Output pt 14: X(13)
        curr_out = out + out_strides[13];
        v_out13 = _mm_sub_ps(bv_s3, bv_s5);
        STR_128_S(curr_out, v_out_stride, v_out13);

        in += v_in_stride * NUM_SETS_REAL_128_S;
        out += v_out_stride * NUM_SETS_REAL_128_S;
    }
    // tail cases
    if (n & 2)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6;
        __m128 av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8, av_s9,
               av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16, av_s17;
        __m128 av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8, av_t9,
               av_t10, av_t11, av_t12, av_t13, av_t14, av_t15, av_t16, av_t17,
               av_t18, av_t19;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13;

        curr_in = in;
        curr_out = out;

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

        av_t1 = _mm_mul_ps(v_CRTM_7_1, av_in5);
        av_t2 = _mm_mul_ps(v_CRTM_7_3, av_in1);
        av_t3 = _mm_mul_ps(v_CRTM_7_5, av_in3);
        av_s1 = _mm_sub_ps(av_in0, av_t1);
        av_s2 = _mm_sub_ps(av_t2, av_t3);
        av_s3 = _mm_add_ps(av_s1, av_s2);

        av_t4 = _mm_mul_ps(v_CRTM_7_2, av_in6);
        av_t5 = _mm_mul_ps(v_CRTM_7_4, av_in2);
        av_t6 = _mm_mul_ps(v_CRTM_7_6, av_in4);
        av_s4 = _mm_add_ps(av_t4, av_t5);
        av_s5 = _mm_add_ps(av_s4, av_t6);

        av_t7 = _mm_mul_ps(v_CRTM_7_1, av_in3);
        av_t8 = _mm_mul_ps(v_CRTM_7_3, av_in5);
        av_t9 = _mm_mul_ps(v_CRTM_7_5, av_in1);
        av_s6 = _mm_sub_ps(av_in0, av_t7);
        av_s7 = _mm_sub_ps(av_t8, av_t9);
        av_s8 = _mm_add_ps(av_s6, av_s7);

        av_t10 = _mm_mul_ps(v_CRTM_7_2, av_in4);
        av_t11 = _mm_mul_ps(v_CRTM_7_4, av_in6);
        av_t12 = _mm_mul_ps(v_CRTM_7_6, av_in2);
        av_s9 = _mm_add_ps(av_t10, av_t11);
        av_s10 = _mm_sub_ps(av_t12, av_s9);

        av_t13 = _mm_mul_ps(v_CRTM_7_1, av_in1);
        av_t14 = _mm_mul_ps(v_CRTM_7_3, av_in3);
        av_t15 = _mm_mul_ps(v_CRTM_7_5, av_in5);
        av_s11 = _mm_sub_ps(av_in0, av_t13);
        av_s12 = _mm_sub_ps(av_t14, av_t15);
        av_s13 = _mm_add_ps(av_s11, av_s12);

        av_t16 = _mm_mul_ps(v_CRTM_7_2, av_in2);
        av_t17 = _mm_mul_ps(v_CRTM_7_4, av_in4);
        av_t18 = _mm_mul_ps(v_CRTM_7_6, av_in6);
        av_s14 = _mm_sub_ps(av_t16, av_t17);
        av_s15 = _mm_add_ps(av_s14, av_t18);

        av_s16 = _mm_add_ps(av_in1, av_in3);
        av_s17 = _mm_add_ps(av_s16, av_in5);
        av_t19 = _mm_mul_ps(v_CRTM_7_7, av_s17);

        // Output pt 1: X(0)
        v_out0 = _mm_add_ps(av_in0, av_t19);
        STHR_128_S(curr_out, v_out_stride, v_out0);
        // Output pt 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm_sub_ps(av_s3, av_s5);
        STHR_128_S(curr_out, v_out_stride, v_out2);
        // Output pt 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm_sub_ps(av_s8, av_s10);
        STHR_128_S(curr_out, v_out_stride, v_out4);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm_sub_ps(av_s13, av_s15);
        STHR_128_S(curr_out, v_out_stride, v_out6);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm_add_ps(av_s13, av_s15);
        STHR_128_S(curr_out, v_out_stride, v_out8);
        // Output pt 11: X(10)
        curr_out = out + out_strides[10];
        v_out10 = _mm_add_ps(av_s8, av_s10);
        STHR_128_S(curr_out, v_out_stride, v_out10);
        // Output pt 13: X(12)
        curr_out = out + out_strides[12];
        v_out12 = _mm_add_ps(av_s3, av_s5);
        STHR_128_S(curr_out, v_out_stride, v_out12);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6;
        __m128 bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
               bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17;
        __m128 bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
               bv_t10, bv_t11, bv_t12, bv_t13, bv_t14, bv_t15, bv_t16, bv_t17,
               bv_t18, bv_t19;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in4, bv_in5);
        // Input point 14: x(13)
        curr_in = in + in_strides[13];
        LDHR_128_S(curr_in, v_in_stride, bv_in6);

        bv_t1 = _mm_mul_ps(v_CRTM_7_1, bv_in0);
        bv_t2 = _mm_mul_ps(v_CRTM_7_3, bv_in4);
        bv_t3 = _mm_mul_ps(v_CRTM_7_5, bv_in2);
        bv_s1 = _mm_sub_ps(bv_in6, bv_t1);
        bv_s2 = _mm_sub_ps(bv_t2, bv_t3);
        bv_s3 = _mm_add_ps(bv_s1, bv_s2);

        bv_t4 = _mm_mul_ps(v_CRTM_7_2, bv_in1);
        bv_t5 = _mm_mul_ps(v_CRTM_7_4, bv_in5);
        bv_t6 = _mm_mul_ps(v_CRTM_7_6, bv_in3);
        bv_s4 = _mm_add_ps(bv_t4, bv_t5);
        bv_s5 = _mm_add_ps(bv_s4, bv_t6);

        bv_t7 = _mm_mul_ps(v_CRTM_7_1, bv_in2);
        bv_t8 = _mm_mul_ps(v_CRTM_7_3, bv_in0);
        bv_t9 = _mm_mul_ps(v_CRTM_7_5, bv_in4);
        bv_s6 = _mm_sub_ps(bv_in6, bv_t7);
        bv_s7 = _mm_sub_ps(bv_t8, bv_t9);
        bv_s8 = _mm_add_ps(bv_s6, bv_s7);

        bv_t10 = _mm_mul_ps(v_CRTM_7_2, bv_in3);
        bv_t11 = _mm_mul_ps(v_CRTM_7_4, bv_in1);
        bv_t12 = _mm_mul_ps(v_CRTM_7_6, bv_in5);
        bv_s9 = _mm_add_ps(bv_t10, bv_t11);
        bv_s10 = _mm_sub_ps(bv_t12, bv_s9);

        bv_t13 = _mm_mul_ps(v_CRTM_7_1, bv_in4);
        bv_t14 = _mm_mul_ps(v_CRTM_7_3, bv_in2);
        bv_t15 = _mm_mul_ps(v_CRTM_7_5, bv_in0);
        bv_s11 = _mm_sub_ps(bv_in6, bv_t13);
        bv_s12 = _mm_sub_ps(bv_t14, bv_t15);
        bv_s13 = _mm_add_ps(bv_s11, bv_s12);

        bv_t16 = _mm_mul_ps(v_CRTM_7_2, bv_in5);
        bv_t17 = _mm_mul_ps(v_CRTM_7_4, bv_in3);
        bv_t18 = _mm_mul_ps(v_CRTM_7_6, bv_in1);
        bv_s14 = _mm_sub_ps(bv_t17, bv_t16);
        bv_s15 = _mm_sub_ps(bv_s14, bv_t18);

        bv_s16 = _mm_add_ps(bv_in0, bv_in2);
        bv_s17 = _mm_add_ps(bv_s16, bv_in4);
        bv_t19 = _mm_mul_ps(v_CRTM_7_7, bv_s17);
        // Output pt 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm_add_ps(bv_in6, bv_t19);
        STHR_128_S(curr_out, v_out_stride, v_out1);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = NEGATE_128_S(_mm_add_ps(bv_s3, bv_s5));
        STHR_128_S(curr_out, v_out_stride, v_out3);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm_add_ps(bv_s8, bv_s10);
        STHR_128_S(curr_out, v_out_stride, v_out5);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = _mm_sub_ps(bv_s15, bv_s13);
        STHR_128_S(curr_out, v_out_stride, v_out7);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm_add_ps(bv_s13, bv_s15);
        STHR_128_S(curr_out, v_out_stride, v_out9);
        // Output pt 12: X(11)
        curr_out = out + out_strides[11];
        v_out11 = _mm_sub_ps(bv_s10, bv_s8);
        STHR_128_S(curr_out, v_out_stride, v_out11);
        // Output pt 14: X(13)
        curr_out = out + out_strides[13];
        v_out13 = _mm_sub_ps(bv_s3, bv_s5);
        STHR_128_S(curr_out, v_out_stride, v_out13);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        /* Standard DFT */
        FFTZ_FLOAT a_in0, a_in1, a_in2, a_in3, a_in4, a_in5, a_in6;
        FFTZ_FLOAT a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_s9, a_s10,
              a_s11, a_s12, a_s13, a_s14, a_s15, a_s16, a_s17, a_t1, a_t2,
              a_t3, a_t4, a_t5, a_t6, a_t7, a_t8, a_t9, a_t10, a_t11, a_t12,
              a_t13, a_t14, a_t15, a_t16, a_t17, a_t18, a_t19;

        a_in0 = *in;                  // Input point 1: x(0)
        a_in1 = in[in_strides[3]];    // Input point 4: x(3)
        a_in2 = in[in_strides[4]];    // Input point 5: x(4)
        a_in3 = in[in_strides[7]];    // Input point 8: x(7)
        a_in4 = in[in_strides[8]];    // Input point 9: x(8)
        a_in5 = in[in_strides[11]];   // Input point 12: x(11)
        a_in6 = in[in_strides[12]];   // Input point 13: x(12)

        a_t1 = CRTM_7_1 * a_in5;
        a_t2 = CRTM_7_3 * a_in1;
        a_t3 = CRTM_7_5 * a_in3;
        a_s1 = a_in0 - a_t1;
        a_s2 = a_t2 - a_t3;

        a_t4 = CRTM_7_2 * a_in6;
        a_t5 = CRTM_7_4 * a_in2;
        a_t6 = CRTM_7_6 * a_in4;
        a_s3 = a_t4 + a_t5;
        a_s4 = a_s1 + a_s2;
        a_s5 = a_s3 + a_t6;

        a_t7 = CRTM_7_1 * a_in3;
        a_t8 = CRTM_7_3 * a_in5;
        a_t9 = CRTM_7_5 * a_in1;
        a_s6 = a_in0 - a_t7;
        a_s7 = a_t8 - a_t9;

        a_t10 = CRTM_7_2 * a_in4;
        a_t11 = CRTM_7_4 * a_in6;
        a_t12 = CRTM_7_6 * a_in2;
        a_s8 = a_t10 + a_t11;

        a_s9 = a_s6 + a_s7;
        a_s10 = a_t12 - a_s8;

        a_t13 = CRTM_7_1 * a_in1;
        a_t14 = CRTM_7_3 * a_in3;
        a_t15 = CRTM_7_5 * a_in5;
        a_s11 = a_in0 - a_t13;
        a_s12 = a_t14 - a_t15;

        a_t16 = CRTM_7_2 * a_in2;
        a_t17 = CRTM_7_4 * a_in4;
        a_t18 = CRTM_7_6 * a_in6;
        a_s13 = a_t16 - a_t17;

        a_s14 = a_in1 + a_in3;
        a_s15 = a_s11 + a_s12;
        a_s16 = a_s14 + a_in5;
        a_s17 = a_s13 + a_t18;
        a_t19 = CRTM_7_7 * a_s16;

        *out = a_in0 + a_t19;                   // Output pt 1: X(0)
        out[out_strides[2]]  = a_s4 - a_s5;     // Output pt 3: X(2)
        out[out_strides[4]]  = a_s9 - a_s10;    // Output pt 5: X(4)
        out[out_strides[6]]  = a_s15 - a_s17;   // Output pt 7: X(6)
        out[out_strides[8]]  = a_s15 + a_s17;   // Output pt 9: X(8)
        out[out_strides[10]] = a_s9 + a_s10;    // Output pt 11: X(10)
        out[out_strides[12]] = a_s4 + a_s5;     // Output pt 13: X(12)

        /* Shifted DFT */
        FFTZ_FLOAT b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6;
        FFTZ_FLOAT b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9, b_s10,
              b_s11, b_s12, b_s13, b_s14, b_s15, b_s16, b_s17, b_t1, b_t2,
              b_t3, b_t4, b_t5, b_t6, b_t7, b_t8, b_t9, b_t10, b_t11, b_t12,
              b_t13, b_t14, b_t15, b_t16, b_t17, b_t18, b_t19;

        b_in0 = in[in_strides[1]];    // Input point 2: x(1)
        b_in1 = in[in_strides[2]];    // Input point 3: x(2)
        b_in2 = in[in_strides[5]];    // Input point 6: x(5)
        b_in3 = in[in_strides[6]];    // Input point 7: x(6)
        b_in4 = in[in_strides[9]];    // Input point 10: x(9)
        b_in5 = in[in_strides[10]];   // Input point 11: x(10)
        b_in6 = in[in_strides[13]];   // Input point 14: x(13)

        b_t1 = CRTM_7_1 * b_in0;
        b_t2 = CRTM_7_3 * b_in4;
        b_t3 = CRTM_7_5 * b_in2;
        b_s1 = b_in6 - b_t1;
        b_s2 = b_t2 - b_t3;
        b_s3 = b_s1 + b_s2;

        b_t4 = CRTM_7_2 * b_in1;
        b_t5 = CRTM_7_4 * b_in5;
        b_t6 = CRTM_7_6 * b_in3;
        b_s4 = b_t4 + b_t5;

        b_s5 = b_s4 + b_t6;
        b_t7 = CRTM_7_1 * b_in2;
        b_t8 = CRTM_7_3 * b_in0;
        b_t9 = CRTM_7_5 * b_in4;
        b_s6 = b_in6 - b_t7;
        b_s7 = b_t8 - b_t9;

        b_t10 = CRTM_7_2 * b_in3;
        b_t11 = CRTM_7_4 * b_in1;
        b_t12 = CRTM_7_6 * b_in5;
        b_s8 = b_t10 + b_t11;

        b_s9 = b_s6 + b_s7;
        b_s10 = b_t12 - b_s8;

        b_t13 = CRTM_7_1 * b_in4;
        b_t14 = CRTM_7_3 * b_in2;
        b_t15 = CRTM_7_5 * b_in0;
        b_s11 = b_in6 - b_t13;
        b_s12 = b_t14 - b_t15;

        b_t16 = CRTM_7_2 * b_in5;
        b_t17 = CRTM_7_4 * b_in3;
        b_t18 = CRTM_7_6 * b_in1;
        b_s13 = b_t17 - b_t16;

        b_s14 = b_in0 + b_in2;
        b_s15 = b_s11 + b_s12;
        b_s16 = b_s14 + b_in4;
        b_s17 = b_s13 - b_t18;
        b_t19 = CRTM_7_7 * b_s16;

        out[out_strides[1]]  = b_in6 + b_t19;   // Output pt 2: X(1)
        out[out_strides[3]]  = -b_s3 - b_s5;    // Output pt 4: X(3)
        out[out_strides[5]]  = b_s9 + b_s10;    // Output pt 6: X(5)
        out[out_strides[7]]  = b_s17 - b_s15;   // Output pt 8: X(7)
        out[out_strides[9]]  = b_s15 + b_s17;   // Output pt 10: X(9)
        out[out_strides[11]] = b_s10 - b_s9;    // Output pt 12: X(11)
        out[out_strides[13]] = b_s3 - b_s5;     // Output pt 14: X(13)
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}
static FFTZ_VOID r2hcf_rfft7avx128_fp64_fwd(FFTZ_VOID *in_real,
                                            FFTZ_VOID *in_imag,
                                            FFTZ_VOID *out_real,
                                            FFTZ_VOID *out_imag, FFTZ_INTP n,
                                            aoclfftz_strides_t *strides,
                                            FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_7_1 =
        0.900968867902419126236102319507445051165919162;
    const FFTZ_DOUBLE CRTM_7_2 =
        0.433883739117558120475768332848358754609990728;
    const FFTZ_DOUBLE CRTM_7_3 =
        0.623489801858733530525004884004239810632274731;
    const FFTZ_DOUBLE CRTM_7_4 =
        0.781831482468029808708444526674057750232334519;
    const FFTZ_DOUBLE CRTM_7_5 =
        0.222520933956314404288902564496794759466355569;
    const FFTZ_DOUBLE CRTM_7_6 =
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
    FFTZ_INTP N = n / NUM_SETS_REAL_128_D;
    FFTZ_DOUBLE *curr_in, *curr_out;

    __m128d v_CRTM_7_1 = _mm_set1_pd(CRTM_7_1);
    __m128d v_CRTM_7_2 = _mm_set1_pd(CRTM_7_2);
    __m128d v_CRTM_7_3 = _mm_set1_pd(CRTM_7_3);
    __m128d v_CRTM_7_4 = _mm_set1_pd(CRTM_7_4);
    __m128d v_CRTM_7_5 = _mm_set1_pd(CRTM_7_5);
    __m128d v_CRTM_7_6 = _mm_set1_pd(CRTM_7_6);

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m128d av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6;
        __m128d av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8, av_s9,
                av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16, av_s17;
        __m128d av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8, av_t9,
                av_t10, av_t11, av_t12, av_t13, av_t14, av_t15, av_t16, av_t17,
                av_t18;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13;

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

        av_s1 = _mm_add_pd(av_in6, av_in1);
        av_s2 = _mm_sub_pd(av_in6, av_in1);
        av_s3 = _mm_add_pd(av_in5, av_in2);
        av_s4 = _mm_sub_pd(av_in5, av_in2);
        av_s5 = _mm_add_pd(av_in4, av_in3);
        av_s6 = _mm_sub_pd(av_in4, av_in3);

        av_s7 = _mm_add_pd(av_in0, av_s1);
        av_s8 = _mm_add_pd(av_s3, av_s5);

        av_t1 = _mm_mul_pd(v_CRTM_7_1, av_s5);
        av_t2 = _mm_mul_pd(v_CRTM_7_3, av_s1);
        av_t3 = _mm_mul_pd(v_CRTM_7_5, av_s3);
        av_s9 = _mm_sub_pd(av_in0, av_t1);

        av_s10 = _mm_sub_pd(av_t2, av_t3);
        av_t4 = _mm_mul_pd(v_CRTM_7_2, av_s6);
        av_t5 = _mm_mul_pd(v_CRTM_7_4, av_s2);

        av_t6 = _mm_mul_pd(v_CRTM_7_6, av_s4);
        av_s11 = _mm_add_pd(av_t4, av_t5);

        av_t7 = _mm_mul_pd(v_CRTM_7_1, av_s3);
        av_t8 = _mm_mul_pd(v_CRTM_7_3, av_s5);
        av_t9 = _mm_mul_pd(v_CRTM_7_5, av_s1);

        av_s12 = _mm_sub_pd(av_in0, av_t7);
        av_s13 = _mm_sub_pd(av_t8, av_t9);
        av_t10 = _mm_mul_pd(v_CRTM_7_2, av_s4);
        av_t11 = _mm_mul_pd(v_CRTM_7_4, av_s6);

        av_t12 = _mm_mul_pd(v_CRTM_7_6, av_s2);
        av_s14 = _mm_add_pd(av_t10, av_t11);
        av_t13 = _mm_mul_pd(v_CRTM_7_1, av_s1);
        av_t14 = _mm_mul_pd(v_CRTM_7_3, av_s3);
        av_t15 = _mm_mul_pd(v_CRTM_7_5, av_s5);

        av_s15 = _mm_sub_pd(av_in0, av_t13);
        av_s16 = _mm_sub_pd(av_t14, av_t15);
        av_t16 = _mm_mul_pd(v_CRTM_7_2, av_s2);
        av_t17 = _mm_mul_pd(v_CRTM_7_4, av_s4);
        av_t18 = _mm_mul_pd(v_CRTM_7_6, av_s6);
        av_s17 = _mm_sub_pd(av_t16, av_t17);

        // Output pt 1: X(0)
        v_out0 = _mm_add_pd(av_s7, av_s8);
        STR_128_D(curr_out, v_out_stride, v_out0);
        // Output pt 4: X(3) & Output pt 5: X(4)
        curr_out = out + out_strides[3];
        v_out3 = _mm_add_pd(av_s9, av_s10);
        v_out4 = _mm_add_pd(av_t6, av_s11);
        STRI_2x128_D(curr_out, v_out_stride, v_out3, v_out4);
        // Output pt 8: X(7) & Output pt 9: X(8)
        curr_out = out + out_strides[7];
        v_out7 = _mm_add_pd(av_s12, av_s13);
        v_out8 = _mm_sub_pd(av_t12, av_s14);
        STRI_2x128_D(curr_out, v_out_stride, v_out7, v_out8);
        // Output pt 12: X(11) & Output pt 13: X(12)
        curr_out = out + out_strides[11];
        v_out11 = _mm_add_pd(av_s15, av_s16);
        v_out12 = _mm_add_pd(av_s17, av_t18);
        STRI_2x128_D(curr_out, v_out_stride, v_out11, v_out12);

        /* Shifted DFT */
        __m128d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6;
        __m128d bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
                bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17;
        __m128d bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
                bv_t10, bv_t11, bv_t12, bv_t13, bv_t14, bv_t15, bv_t16, bv_t17,
                bv_t18;

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

        bv_s1 = _mm_add_pd(bv_in6, bv_in1);
        bv_s2 = _mm_sub_pd(bv_in6, bv_in1);
        bv_s3 = _mm_add_pd(bv_in5, bv_in2);
        bv_s4 = _mm_sub_pd(bv_in5, bv_in2);
        bv_s5 = _mm_add_pd(bv_in4, bv_in3);
        bv_s6 = _mm_sub_pd(bv_in4, bv_in3);

        bv_t1 = _mm_mul_pd(v_CRTM_7_1, bv_s2);
        bv_t2 = _mm_mul_pd(v_CRTM_7_3, bv_s4);

        bv_t3 = _mm_mul_pd(v_CRTM_7_5, bv_s6);
        bv_s7 = _mm_sub_pd(bv_in0, bv_t1);
        bv_s8 = _mm_add_pd(bv_t2, bv_t3);
        bv_t4 = _mm_mul_pd(v_CRTM_7_2, bv_s1);

        bv_t5 = _mm_mul_pd(v_CRTM_7_4, bv_s3);
        bv_t6 = _mm_mul_pd(v_CRTM_7_6, bv_s5);
        bv_s9 = NEGATE_128_D(_mm_add_pd(bv_t4, bv_t5));

        bv_t7 = _mm_mul_pd(v_CRTM_7_1, bv_s4);
        bv_t8 = _mm_mul_pd(v_CRTM_7_3, bv_s6);
        bv_t9 = _mm_mul_pd(v_CRTM_7_5, bv_s2);
        bv_s10 = _mm_add_pd(bv_in0, bv_t7);
        bv_s11 = _mm_sub_pd(bv_t8, bv_t9);

        bv_t10 = _mm_mul_pd(v_CRTM_7_2, bv_s3);
        bv_t11 = _mm_mul_pd(v_CRTM_7_4, bv_s5);

        bv_t12 = _mm_mul_pd(v_CRTM_7_6, bv_s1);
        bv_s12 = _mm_sub_pd(bv_t11, bv_t10);

        bv_t13 = _mm_mul_pd(v_CRTM_7_1, bv_s6);
        bv_t14 = _mm_mul_pd(v_CRTM_7_3, bv_s2);
        bv_t15 = _mm_mul_pd(v_CRTM_7_5, bv_s4);
        bv_s13 = _mm_sub_pd(bv_in0, bv_t13);
        bv_s14 = _mm_add_pd(bv_t14, bv_t15);

        bv_t16 = _mm_mul_pd(v_CRTM_7_2, bv_s5);
        bv_t17 = _mm_mul_pd(v_CRTM_7_4, bv_s1);
        bv_t18 = _mm_mul_pd(v_CRTM_7_6, bv_s3);
        bv_s15 = _mm_add_pd(bv_t16, bv_t17);
        bv_s16 = _mm_add_pd(bv_in0, bv_s2);
        bv_s17 = _mm_sub_pd(bv_s6, bv_s4);

        // Output pt 2: X(1) & Output pt 3: X(2)
        curr_out = out + out_strides[1];
        v_out1 = _mm_sub_pd(bv_s7, bv_s8);
        v_out2 = _mm_sub_pd(bv_s9, bv_t6);
        STRI_2x128_D(curr_out, v_out_stride, v_out1, v_out2);
        // Output pt 6: X(5) & Output pt 7: X(6)
        curr_out = out + out_strides[5];
        v_out5 = _mm_add_pd(bv_s10, bv_s11);
        v_out6 = _mm_sub_pd(bv_s12, bv_t12);
        STRI_2x128_D(curr_out, v_out_stride, v_out5, v_out6);
        // Output pt 10: X(9) & Output pt 11: X(10)
        curr_out = out + out_strides[9];
        v_out9 = _mm_add_pd(bv_s13, bv_s14);
        v_out10 = _mm_sub_pd(bv_t18, bv_s15);
        STRI_2x128_D(curr_out, v_out_stride, v_out9, v_out10);
        // Output pt 14: X(13)
        curr_out = out + out_strides[13];
        v_out13 = _mm_add_pd(bv_s16, bv_s17);
        STR_128_D(curr_out, v_out_stride, v_out13);

        in += v_in_stride * NUM_SETS_REAL_128_D;
        out += v_out_stride * NUM_SETS_REAL_128_D;
    }
    // tail cases
    if (n & 1)
    {
        /* Standard DFT */
        FFTZ_DOUBLE a_in0, a_in1, a_in2, a_in3, a_in4, a_in5, a_in6;
        FFTZ_DOUBLE a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_t1, a_t2,
            a_t3, a_t4, a_t5, a_s9, a_s10, a_t6, a_s11, a_t7, a_t8, a_t9, a_s12,
            a_s13, a_t10, a_t11, a_t12, a_t13, a_t14, a_t15, a_s14, a_s15,
            a_s16, a_t16, a_t17, a_t18, a_s17;

        a_in0 = *in;                  // Input point 1: x(0)
        a_in1 = in[in_strides[2]];    // Input point 3: x(2)
        a_in2 = in[in_strides[4]];    // Input point 5: x(4)
        a_in3 = in[in_strides[6]];    // Input point 7: x(6)
        a_in4 = in[in_strides[8]];    // Input point 9: x(8)
        a_in5 = in[in_strides[10]];   // Input point 11: x(10)
        a_in6 = in[in_strides[12]];   // Input point 13: x(12)

        a_s1 = a_in6 + a_in1;
        a_s2 = a_in6 - a_in1;
        a_s3 = a_in5 + a_in2;
        a_s4 = a_in5 - a_in2;
        a_s5 = a_in4 + a_in3;
        a_s6 = a_in4 - a_in3;
        a_s7 = a_in0 + a_s1;
        a_s8 = a_s3 + a_s5;

        a_t1 = CRTM_7_1 * a_s5;
        a_t2 = CRTM_7_3 * a_s1;
        a_t3 = CRTM_7_5 * a_s3;
        a_t4 = CRTM_7_2 * a_s6;
        a_t5 = CRTM_7_4 * a_s2;
        a_s9 = a_in0 - a_t1;
        a_s10 = a_t2 - a_t3;

        a_t6 = CRTM_7_6 * a_s4;
        a_s11 = a_t4 + a_t5;
        a_t7 = CRTM_7_1 * a_s3;
        a_t8 = CRTM_7_3 * a_s5;
        a_t9 = CRTM_7_5 * a_s1;

        a_s12 = a_in0 - a_t7;
        a_s13 = a_t8 - a_t9;
        a_t10 = CRTM_7_2 * a_s4;
        a_t11 = CRTM_7_4 * a_s6;

        a_t12 = CRTM_7_6 * a_s2;
        a_t13 = CRTM_7_1 * a_s1;
        a_t14 = CRTM_7_3 * a_s3;
        a_t15 = CRTM_7_5 * a_s5;
        a_s14 = a_t10 + a_t11;

        a_s15 = a_in0 - a_t13;
        a_s16 = a_t14 - a_t15;
        a_t16 = CRTM_7_2 * a_s2;
        a_t17 = CRTM_7_4 * a_s4;
        a_t18 = CRTM_7_6 * a_s6;
        a_s17 = a_t16 - a_t17;

        *out = a_s7 + a_s8;                       // Output pt 1: X(0)
        out[out_strides[3]]  = a_s9 + a_s10;      // Output pt 4: X(3)
        out[out_strides[4]]  = a_t6 + a_s11;      // Output pt 5: X(4)
        out[out_strides[7]]  = a_s12 + a_s13;     // Output pt 8: X(7)
        out[out_strides[8]]  = a_t12 - a_s14;     // Output pt 9: X(8)
        out[out_strides[11]] = a_s15 + a_s16;     // Output pt 12: X(11)
        out[out_strides[12]] = a_s17 + a_t18;     // Output pt 13: X(12)

        /* Shifted DFT */
        FFTZ_DOUBLE b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6;
        FFTZ_DOUBLE b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_t1, b_t2, b_t3, b_s7,
               b_s8, b_t4, b_t5, b_t6, b_s9, b_t7, b_t8, b_t9, b_s10, b_s11,
               b_t10, b_t11, b_t12, b_s12, b_t13, b_t14, b_t15, b_s13, b_s14,
               b_t16, b_t17, b_t18, b_s15, b_s16, b_s17;

        b_in0 = in[in_strides[1]];    // Input point 2: x(1)
        b_in1 = in[in_strides[3]];    // Input point 4: x(3)
        b_in2 = in[in_strides[5]];    // Input point 6: x(5)
        b_in3 = in[in_strides[7]];    // Input point 8: x(7)
        b_in4 = in[in_strides[9]];    // Input point 10: x(9)
        b_in5 = in[in_strides[11]];   // Input point 12: x(11)
        b_in6 = in[in_strides[13]];   // Input point 14: x(13)

        b_s1 = b_in6 + b_in1;
        b_s2 = b_in6 - b_in1;
        b_s3 = b_in5 + b_in2;
        b_s4 = b_in5 - b_in2;
        b_s5 = b_in4 + b_in3;
        b_s6 = b_in4 - b_in3;

        b_t1 = CRTM_7_1 * b_s2;
        b_t2 = CRTM_7_3 * b_s4;

        b_t3 = CRTM_7_5 * b_s6;
        b_s7  = b_in0 - b_t1;
        b_s8 = b_t2 + b_t3;
        b_t4 = CRTM_7_2 * b_s1;
        b_t5 = CRTM_7_4 * b_s3;
        b_t6 = CRTM_7_6 * b_s5;
        b_s9 = -b_t4 - b_t5;

        b_t7 = CRTM_7_1 * b_s4;
        b_t8 = CRTM_7_3 * b_s6;

        b_t9 = CRTM_7_5 * b_s2;
        b_s10 = b_in0 + b_t7;
        b_t10 = CRTM_7_2 * b_s3;
        b_s11 = b_t8 - b_t9;
        b_t11 = CRTM_7_4 * b_s5;
        b_t12 = CRTM_7_6 * b_s1;
        b_s12 = b_t11 - b_t10;

        b_t13 = CRTM_7_1 * b_s6;
        b_t14 = CRTM_7_3 * b_s2;
        b_t15 = CRTM_7_5 * b_s4;
        b_s13 = b_in0 - b_t13;
        b_s14 = b_t14 + b_t15;

        b_t16 = CRTM_7_2 * b_s5;
        b_t17 = CRTM_7_4 * b_s1;
        b_t18 = CRTM_7_6 * b_s3;
        b_s15 = b_t16 + b_t17;
        b_s16 = b_in0 + b_s2;
        b_s17 = b_s6 - b_s4;

        out[out_strides[1]]  = b_s7 - b_s8;      // Output pt 2: X(1)
        out[out_strides[2]]  = b_s9 - b_t6;      // Output pt 3: X(2)
        out[out_strides[5]]  = b_s10 + b_s11;    // Output pt 6: X(5)
        out[out_strides[6]]  = b_s12 - b_t12;    // Output pt 7: X(6)
        out[out_strides[9]]  = b_s13 + b_s14;    // Output pt 10: X(9)
        out[out_strides[10]] = b_t18 - b_s15;    // Output pt 11: X(10)
        out[out_strides[13]] = b_s16 + b_s17;    // Output pt 14: X(13)
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hcf_rfft7avx128_fp64_bwd(FFTZ_VOID *in_real,
                                            FFTZ_VOID *in_imag,
                                            FFTZ_VOID *out_real,
                                            FFTZ_VOID *out_imag, FFTZ_INTP n,
                                            aoclfftz_strides_t *strides,
                                            FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_7_1 =
        1.801937735804838252472204639014890102331838324;
    const FFTZ_DOUBLE CRTM_7_2 =
        0.867767478235116240951536665696717509219981456;
    const FFTZ_DOUBLE CRTM_7_3 =
        1.246979603717467061050009768008479621264549462;
    const FFTZ_DOUBLE CRTM_7_4 =
        1.563662964936059617416889053348115500464669038;
    const FFTZ_DOUBLE CRTM_7_5 =
        0.445041867912628808577805128993589518932711138;
    const FFTZ_DOUBLE CRTM_7_6 =
        1.949855824363647214036263365987862434465571602;
    const FFTZ_DOUBLE CRTM_7_7 =
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
    FFTZ_INTP N = n / NUM_SETS_REAL_128_D;
    FFTZ_DOUBLE *curr_in, *curr_out;

    __m128d v_CRTM_7_1 = _mm_set1_pd(CRTM_7_1);
    __m128d v_CRTM_7_2 = _mm_set1_pd(CRTM_7_2);
    __m128d v_CRTM_7_3 = _mm_set1_pd(CRTM_7_3);
    __m128d v_CRTM_7_4 = _mm_set1_pd(CRTM_7_4);
    __m128d v_CRTM_7_5 = _mm_set1_pd(CRTM_7_5);
    __m128d v_CRTM_7_6 = _mm_set1_pd(CRTM_7_6);
    __m128d v_CRTM_7_7 = _mm_set1_pd(CRTM_7_7);

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m128d av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6;
        __m128d av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8, av_s9,
                av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16, av_s17;
        __m128d av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8, av_t9,
                av_t10, av_t11, av_t12, av_t13, av_t14, av_t15, av_t16, av_t17,
                av_t18, av_t19;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13;

        curr_in = in;
        curr_out = out;

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

        av_t1 = _mm_mul_pd(v_CRTM_7_1, av_in5);
        av_t2 = _mm_mul_pd(v_CRTM_7_3, av_in1);
        av_t3 = _mm_mul_pd(v_CRTM_7_5, av_in3);
        av_s1 = _mm_sub_pd(av_in0, av_t1);
        av_s2 = _mm_sub_pd(av_t2, av_t3);
        av_s3 = _mm_add_pd(av_s1, av_s2);

        av_t4 = _mm_mul_pd(v_CRTM_7_2, av_in6);
        av_t5 = _mm_mul_pd(v_CRTM_7_4, av_in2);
        av_t6 = _mm_mul_pd(v_CRTM_7_6, av_in4);
        av_s4 = _mm_add_pd(av_t4, av_t5);
        av_s5 = _mm_add_pd(av_s4, av_t6);

        av_t7 = _mm_mul_pd(v_CRTM_7_1, av_in3);
        av_t8 = _mm_mul_pd(v_CRTM_7_3, av_in5);
        av_t9 = _mm_mul_pd(v_CRTM_7_5, av_in1);
        av_s6 = _mm_sub_pd(av_in0, av_t7);
        av_s7 = _mm_sub_pd(av_t8, av_t9);
        av_s8 = _mm_add_pd(av_s6, av_s7);

        av_t10 = _mm_mul_pd(v_CRTM_7_2, av_in4);
        av_t11 = _mm_mul_pd(v_CRTM_7_4, av_in6);
        av_t12 = _mm_mul_pd(v_CRTM_7_6, av_in2);
        av_s9 = _mm_add_pd(av_t10, av_t11);
        av_s10 = _mm_sub_pd(av_t12, av_s9);

        av_t13 = _mm_mul_pd(v_CRTM_7_1, av_in1);
        av_t14 = _mm_mul_pd(v_CRTM_7_3, av_in3);
        av_t15 = _mm_mul_pd(v_CRTM_7_5, av_in5);
        av_s11 = _mm_sub_pd(av_in0, av_t13);
        av_s12 = _mm_sub_pd(av_t14, av_t15);
        av_s13 = _mm_add_pd(av_s11, av_s12);

        av_t16 = _mm_mul_pd(v_CRTM_7_2, av_in2);
        av_t17 = _mm_mul_pd(v_CRTM_7_4, av_in4);
        av_t18 = _mm_mul_pd(v_CRTM_7_6, av_in6);
        av_s14 = _mm_sub_pd(av_t16, av_t17);
        av_s15 = _mm_add_pd(av_s14, av_t18);

        av_s16 = _mm_add_pd(av_in1, av_in3);
        av_s17 = _mm_add_pd(av_s16, av_in5);
        av_t19 = _mm_mul_pd(v_CRTM_7_7, av_s17);

        // Output pt 1: X(0)
        v_out0 = _mm_add_pd(av_in0, av_t19);
        STR_128_D(curr_out, v_out_stride, v_out0);
        // Output pt 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm_sub_pd(av_s3, av_s5);
        STR_128_D(curr_out, v_out_stride, v_out2);
        // Output pt 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm_sub_pd(av_s8, av_s10);
        STR_128_D(curr_out, v_out_stride, v_out4);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm_sub_pd(av_s13, av_s15);
        STR_128_D(curr_out, v_out_stride, v_out6);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm_add_pd(av_s13, av_s15);
        STR_128_D(curr_out, v_out_stride, v_out8);
        // Output pt 11: X(10)
        curr_out = out + out_strides[10];
        v_out10 = _mm_add_pd(av_s8, av_s10);
        STR_128_D(curr_out, v_out_stride, v_out10);
        // Output pt 13: X(12)
        curr_out = out + out_strides[12];
        v_out12 = _mm_add_pd(av_s3, av_s5);
        STR_128_D(curr_out, v_out_stride, v_out12);

        /* Shifted DFT */
        __m128d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6;
        __m128d bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
                bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17;
        __m128d bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
                bv_t10, bv_t11, bv_t12, bv_t13, bv_t14, bv_t15, bv_t16, bv_t17,
                bv_t18, bv_t19;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in4, bv_in5);
        // Input point 14: x(13)
        curr_in = in + in_strides[13];
        LDR_128_D(curr_in, v_in_stride, bv_in6);

        bv_t1 = _mm_mul_pd(v_CRTM_7_1, bv_in0);
        bv_t2 = _mm_mul_pd(v_CRTM_7_3, bv_in4);
        bv_t3 = _mm_mul_pd(v_CRTM_7_5, bv_in2);
        bv_s1 = _mm_sub_pd(bv_in6, bv_t1);
        bv_s2 = _mm_sub_pd(bv_t2, bv_t3);
        bv_s3 = _mm_add_pd(bv_s1, bv_s2);

        bv_t4 = _mm_mul_pd(v_CRTM_7_2, bv_in1);
        bv_t5 = _mm_mul_pd(v_CRTM_7_4, bv_in5);
        bv_t6 = _mm_mul_pd(v_CRTM_7_6, bv_in3);
        bv_s4 = _mm_add_pd(bv_t4, bv_t5);
        bv_s5 = _mm_add_pd(bv_s4, bv_t6);

        bv_t7 = _mm_mul_pd(v_CRTM_7_1, bv_in2);
        bv_t8 = _mm_mul_pd(v_CRTM_7_3, bv_in0);
        bv_t9 = _mm_mul_pd(v_CRTM_7_5, bv_in4);
        bv_s6 = _mm_sub_pd(bv_in6, bv_t7);
        bv_s7 = _mm_sub_pd(bv_t8, bv_t9);
        bv_s8 = _mm_add_pd(bv_s6, bv_s7);

        bv_t10 = _mm_mul_pd(v_CRTM_7_2, bv_in3);
        bv_t11 = _mm_mul_pd(v_CRTM_7_4, bv_in1);
        bv_t12 = _mm_mul_pd(v_CRTM_7_6, bv_in5);
        bv_s9 = _mm_add_pd(bv_t10, bv_t11);
        bv_s10 = _mm_sub_pd(bv_t12, bv_s9);

        bv_t13 = _mm_mul_pd(v_CRTM_7_1, bv_in4);
        bv_t14 = _mm_mul_pd(v_CRTM_7_3, bv_in2);
        bv_t15 = _mm_mul_pd(v_CRTM_7_5, bv_in0);
        bv_s11 = _mm_sub_pd(bv_in6, bv_t13);
        bv_s12 = _mm_sub_pd(bv_t14, bv_t15);
        bv_s13 = _mm_add_pd(bv_s11, bv_s12);

        bv_t16 = _mm_mul_pd(v_CRTM_7_2, bv_in5);
        bv_t17 = _mm_mul_pd(v_CRTM_7_4, bv_in3);
        bv_t18 = _mm_mul_pd(v_CRTM_7_6, bv_in1);
        bv_s14 = _mm_sub_pd(bv_t17, bv_t16);
        bv_s15 = _mm_sub_pd(bv_s14, bv_t18);

        bv_s16 = _mm_add_pd(bv_in0, bv_in2);
        bv_s17 = _mm_add_pd(bv_s16, bv_in4);
        bv_t19 = _mm_mul_pd(v_CRTM_7_7, bv_s17);

        // Output pt 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm_add_pd(bv_in6, bv_t19);
        STR_128_D(curr_out, v_out_stride, v_out1);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = NEGATE_128_D(_mm_add_pd(bv_s3, bv_s5));
        STR_128_D(curr_out, v_out_stride, v_out3);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm_add_pd(bv_s8, bv_s10);
        STR_128_D(curr_out, v_out_stride, v_out5);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = _mm_sub_pd(bv_s15, bv_s13);
        STR_128_D(curr_out, v_out_stride, v_out7);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm_add_pd(bv_s13, bv_s15);
        STR_128_D(curr_out, v_out_stride, v_out9);
        // Output pt 12: X(11)
        curr_out = out + out_strides[11];
        v_out11 = _mm_sub_pd(bv_s10, bv_s8);
        STR_128_D(curr_out, v_out_stride, v_out11);
        // Output pt 14: X(13)
        curr_out = out + out_strides[13];
        v_out13 = _mm_sub_pd(bv_s3, bv_s5);
        STR_128_D(curr_out, v_out_stride, v_out13);

        in += v_in_stride * NUM_SETS_REAL_128_D;
        out += v_out_stride * NUM_SETS_REAL_128_D;
    }
    // tail cases
    if (n & 1)
    {
        /* Standard DFT */
        FFTZ_DOUBLE a_in0, a_in1, a_in2, a_in3, a_in4, a_in5, a_in6;
        FFTZ_DOUBLE a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_s9, a_s10,
               a_s11, a_s12, a_s13, a_s14, a_s15, a_s16, a_s17, a_t1, a_t2,
               a_t3, a_t4, a_t5, a_t6, a_t7, a_t8, a_t9, a_t10, a_t11, a_t12,
               a_t13, a_t14, a_t15, a_t16, a_t17, a_t18, a_t19;

        a_in0 = *in;                  // Input point 1: x(0)
        a_in1 = in[in_strides[3]];    // Input point 4: x(3)
        a_in2 = in[in_strides[4]];    // Input point 5: x(4)
        a_in3 = in[in_strides[7]];    // Input point 8: x(7)
        a_in4 = in[in_strides[8]];    // Input point 9: x(8)
        a_in5 = in[in_strides[11]];   // Input point 12: x(11)
        a_in6 = in[in_strides[12]];   // Input point 13: x(12)

        a_t1 = CRTM_7_1 * a_in5;
        a_t2 = CRTM_7_3 * a_in1;
        a_t3 = CRTM_7_5 * a_in3;
        a_s1 = a_in0 - a_t1;
        a_s2 = a_t2 - a_t3;

        a_t4 = CRTM_7_2 * a_in6;
        a_t5 = CRTM_7_4 * a_in2;
        a_t6 = CRTM_7_6 * a_in4;
        a_s3 = a_t4 + a_t5;
        a_s4 = a_s1 + a_s2;
        a_s5 = a_s3 + a_t6;

        a_t7 = CRTM_7_1 * a_in3;
        a_t8 = CRTM_7_3 * a_in5;
        a_t9 = CRTM_7_5 * a_in1;
        a_s6 = a_in0 - a_t7;
        a_s7 = a_t8 - a_t9;

        a_t10 = CRTM_7_2 * a_in4;
        a_t11 = CRTM_7_4 * a_in6;
        a_t12 = CRTM_7_6 * a_in2;
        a_s8 = a_t10 + a_t11;

        a_s9 = a_s6 + a_s7;
        a_s10 = a_t12 - a_s8;

        a_t13 = CRTM_7_1 * a_in1;
        a_t14 = CRTM_7_3 * a_in3;
        a_t15 = CRTM_7_5 * a_in5;
        a_s11 = a_in0 - a_t13;
        a_s12 = a_t14 - a_t15;

        a_t16 = CRTM_7_2 * a_in2;
        a_t17 = CRTM_7_4 * a_in4;
        a_t18 = CRTM_7_6 * a_in6;
        a_s13 = a_t16 - a_t17;

        a_s14 = a_in1 + a_in3;
        a_s15 = a_s11 + a_s12;
        a_s16 = a_s14 + a_in5;
        a_s17 = a_s13 + a_t18;
        a_t19 = CRTM_7_7 * a_s16;

        *out = a_in0 + a_t19;                   // Output pt 1: X(0)
        out[out_strides[2]]  = a_s4 - a_s5;     // Output pt 3: X(2)
        out[out_strides[4]]  = a_s9 - a_s10;    // Output pt 5: X(4)
        out[out_strides[6]]  = a_s15 - a_s17;   // Output pt 7: X(6)
        out[out_strides[8]]  = a_s15 + a_s17;   // Output pt 9: X(8)
        out[out_strides[10]] = a_s9 + a_s10;    // Output pt 11: X(10)
        out[out_strides[12]] = a_s4 + a_s5;     // Output pt 13: X(12)

        /* Shifted DFT */
        FFTZ_DOUBLE b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6;
        FFTZ_DOUBLE b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9, b_s10,
               b_s11, b_s12, b_s13, b_s14, b_s15, b_s16, b_s17, b_t1, b_t2,
               b_t3, b_t4, b_t5, b_t6, b_t7, b_t8, b_t9, b_t10, b_t11, b_t12,
               b_t13, b_t14, b_t15, b_t16, b_t17, b_t18, b_t19;

        b_in0 = in[in_strides[1]];    // Input point 2: x(1)
        b_in1 = in[in_strides[2]];    // Input point 3: x(2)
        b_in2 = in[in_strides[5]];    // Input point 6: x(5)
        b_in3 = in[in_strides[6]];    // Input point 7: x(6)
        b_in4 = in[in_strides[9]];    // Input point 10: x(9)
        b_in5 = in[in_strides[10]];   // Input point 11: x(10)
        b_in6 = in[in_strides[13]];   // Input point 14: x(13)

        b_t1 = CRTM_7_1 * b_in0;
        b_t2 = CRTM_7_3 * b_in4;
        b_t3 = CRTM_7_5 * b_in2;
        b_s1 = b_in6 - b_t1;
        b_s2 = b_t2 - b_t3;
        b_s3 = b_s1 + b_s2;

        b_t4 = CRTM_7_2 * b_in1;
        b_t5 = CRTM_7_4 * b_in5;
        b_t6 = CRTM_7_6 * b_in3;
        b_s4 = b_t4 + b_t5;

        b_s5 = b_s4 + b_t6;
        b_t7 = CRTM_7_1 * b_in2;
        b_t8 = CRTM_7_3 * b_in0;
        b_t9 = CRTM_7_5 * b_in4;
        b_s6 = b_in6 - b_t7;
        b_s7 = b_t8 - b_t9;

        b_t10 = CRTM_7_2 * b_in3;
        b_t11 = CRTM_7_4 * b_in1;
        b_t12 = CRTM_7_6 * b_in5;
        b_s8 = b_t10 + b_t11;

        b_s9 = b_s6 + b_s7;
        b_s10 = b_t12 - b_s8;

        b_t13 = CRTM_7_1 * b_in4;
        b_t14 = CRTM_7_3 * b_in2;
        b_t15 = CRTM_7_5 * b_in0;
        b_s11 = b_in6 - b_t13;
        b_s12 = b_t14 - b_t15;

        b_t16 = CRTM_7_2 * b_in5;
        b_t17 = CRTM_7_4 * b_in3;
        b_t18 = CRTM_7_6 * b_in1;
        b_s13 = b_t17 - b_t16;

        b_s14 = b_in0 + b_in2;
        b_s15 = b_s11 + b_s12;
        b_s16 = b_s14 + b_in4;
        b_s17 = b_s13 - b_t18;
        b_t19 = CRTM_7_7 * b_s16;

        out[out_strides[1]]  = b_in6 + b_t19;   // Output pt 2: X(1)
        out[out_strides[3]]  = -b_s3 - b_s5;    // Output pt 4: X(3)
        out[out_strides[5]]  = b_s9 + b_s10;    // Output pt 6: X(5)
        out[out_strides[7]]  = b_s17 - b_s15;   // Output pt 8: X(7)
        out[out_strides[9]]  = b_s15 + b_s17;   // Output pt 10: X(9)
        out[out_strides[11]] = b_s10 - b_s9;    // Output pt 12: X(11)
        out[out_strides[13]] = b_s3 - b_s5;     // Output pt 14: X(13)
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hcf_rfft7avx128(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hcf_rfft7avx128_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft7avx128_fp64_fwd;
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
            return r2hcf_rfft7avx128_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft7avx128_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
