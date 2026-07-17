// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft16avx128.c
 *
 *  @brief Radix-16 r2hc Real-FFT kernel with AVX-128 operations using x86 SIMD
 *  intrinsics
 *
 *  This file contains the DIT radix-16 real-to-halfcomplex implementations
 *  using AVX128 SIMD operations for single-precision and double-precision
 *  inputs.
 *
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/r2hc_simd_avx128.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                {{{0, 12, 58, 100, 68, 2},
                                                  {0, 18, 58, 100, 75, 0}},
                                                 {{0, 12, 58,  50, 14, 2},
                                                  {0, 18, 58,  50, 14, 0}}};

ops_cycles_t get_ops_cnt_r2hc_rfft16avx128(FFTZ_UINT8 precision,
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

static FFTZ_VOID r2hc_rfft16avx128_fp32_fwd(FFTZ_VOID *in_real,
                                            FFTZ_VOID *in_imag,
                                            FFTZ_VOID *out_real,
                                            FFTZ_VOID *out_imag, FFTZ_INTP n,
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

    __m128 v_CRTM_16_1 = _mm_broadcast_ss(&CRTM_16_1);
    __m128 v_CRTM_16_2 = _mm_broadcast_ss(&CRTM_16_2);
    __m128 v_CRTM_16_3 = _mm_broadcast_ss(&CRTM_16_3);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9, v_in10, v_in11, v_in12, v_in13, v_in14, v_in15;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
               v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27,
               v_s28, v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36,
               v_s37, v_s38, v_s39, v_s40, v_s41;
        __m128 v_tv0, v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7, v_tv8,
               v_tv9, v_tv10, v_tv11;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15;

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
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDR_128_S(curr_in, v_in_stride, v_in9, is_contiguous_in);
        // Input point 11: x(10)
        curr_in = in + in_strides[10];
        LDR_128_S(curr_in, v_in_stride, v_in10, is_contiguous_in);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_128_S(curr_in, v_in_stride, v_in11, is_contiguous_in);
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDR_128_S(curr_in, v_in_stride, v_in12, is_contiguous_in);
        // Input point 14: x(13)
        curr_in = in + in_strides[13];
        LDR_128_S(curr_in, v_in_stride, v_in13, is_contiguous_in);
        // Input point 15: x(14)
        curr_in = in + in_strides[14];
        LDR_128_S(curr_in, v_in_stride, v_in14, is_contiguous_in);
        // Input point 16: x(15)
        curr_in = in + in_strides[15];
        LDR_128_S(curr_in, v_in_stride, v_in15, is_contiguous_in);

        v_s0 = _mm_add_ps(v_in0, v_in8);
        v_s1 = _mm_sub_ps(v_in0, v_in8);
        v_s2 = _mm_add_ps(v_in1, v_in15);
        v_s3 = _mm_sub_ps(v_in1, v_in15);
        v_s4 = _mm_add_ps(v_in2, v_in6);
        v_s5 = _mm_sub_ps(v_in2, v_in6);
        v_s6 = _mm_add_ps(v_in3, v_in5);
        v_s7 = _mm_sub_ps(v_in3, v_in5);
        v_s8 = _mm_add_ps(v_in4, v_in12);
        v_s9 = _mm_sub_ps(v_in4, v_in12);
        v_s10 = _mm_add_ps(v_in7, v_in9);
        v_s11 = _mm_sub_ps(v_in7, v_in9);
        v_s12 = _mm_add_ps(v_in10, v_in14);
        v_s13 = _mm_sub_ps(v_in10, v_in14);
        v_s14 = _mm_add_ps(v_in11, v_in13);
        v_s15 = _mm_sub_ps(v_in11, v_in13);

        v_s16 = _mm_add_ps(v_s0, v_s8);
        v_s17 = _mm_sub_ps(v_s0, v_s8);
        v_s18 = _mm_add_ps(v_s2, v_s10);
        v_s19 = _mm_sub_ps(v_s2, v_s10);
        v_s20 = _mm_add_ps(v_s3, v_s11);
        v_s21 = _mm_sub_ps(v_s11, v_s3);
        v_s22 = _mm_add_ps(v_s4, v_s12);
        v_s23 = _mm_sub_ps(v_s4, v_s12);
        // Output point 8: X(7)
        v_out7 = _mm_sub_ps(v_s16, v_s22);

        v_s24 = _mm_add_ps(v_s5, v_s13);
        v_s25 = _mm_sub_ps(v_s5, v_s13);
        v_s26 = _mm_add_ps(v_s6, v_s14);
        v_s27 = _mm_sub_ps(v_s6, v_s14);
        v_s28 = _mm_add_ps(v_s7, v_s15);
        v_s29 = _mm_sub_ps(v_s7, v_s15);
        // Output point 9: X(8)
        v_out8 = _mm_add_ps(v_s28, v_s21);

        v_s30 = _mm_add_ps(v_s18, v_s26);
        v_s31 = _mm_sub_ps(v_s18, v_s26);
        v_s32 = _mm_add_ps(v_s16, v_s22);
        v_s33 = _mm_sub_ps(v_s21, v_s28);
        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_s30, v_s32);
        // Output point 16: X(15)
        v_out15 = _mm_sub_ps(v_s32, v_s30);

        v_tv0 = _mm_mul_ps(v_s19, v_CRTM_16_1);
        v_tv1 = _mm_mul_ps(v_s20, v_CRTM_16_1);
        v_tv2 = _mm_mul_ps(v_s27, v_CRTM_16_1);
        v_tv3 = _mm_mul_ps(v_s29, v_CRTM_16_1);
        v_tv4 = _mm_mul_ps(v_s19, v_CRTM_16_2);
        v_tv5 = _mm_mul_ps(v_s20, v_CRTM_16_2);
        v_tv6 = _mm_mul_ps(v_s27, v_CRTM_16_2);
        v_tv7 = _mm_mul_ps(v_s29, v_CRTM_16_2);
        v_tv8 = _mm_mul_ps(v_s23, v_CRTM_16_3);
        v_tv9 = _mm_mul_ps(v_s25, v_CRTM_16_3);
        v_tv10 = _mm_mul_ps(v_s31, v_CRTM_16_3);
        // Output point 4: X(3)
        v_out3 = _mm_add_ps(v_s17, v_tv10);
        // Output point 12: X(11)
        v_out11 = _mm_sub_ps(v_s17, v_tv10);

        v_tv11 = _mm_mul_ps(v_s33, v_CRTM_16_3);
        // Output point 5: X(4)
        v_out4 = _mm_sub_ps(v_tv11, v_s24);
        // Output point 13: X(12)
        v_out12 = _mm_add_ps(v_s24, v_tv11);

        v_s34 = _mm_add_ps(v_tv0, v_tv7);
        v_s35 = _mm_add_ps(v_tv9, v_s1);
        // Output point 2: X(1)
        v_out1 = _mm_add_ps(v_s34, v_s35);
        // Output point 14: X(13)
        v_out13 = _mm_sub_ps(v_s35, v_s34);

        v_s36 = _mm_add_ps(v_tv2, v_tv5);
        v_s37 = _mm_add_ps(v_tv8, v_s9);
        // Output point 3: X(2)
        v_out2 = NEGATE_128_S(_mm_add_ps(v_s36, v_s37));
        // Output point 15: X(14)
        v_out14 = _mm_sub_ps(v_s37, v_s36);

        v_s38 = _mm_sub_ps(v_tv3, v_tv4);
        v_s39 = _mm_sub_ps(v_s1, v_tv9);
        // Output point 6: X(5)
        v_out5 = _mm_sub_ps(v_s39, v_s38);
        // Output point 10: X(9)
        v_out9 = _mm_add_ps(v_s38, v_s39);

        v_s40 = _mm_sub_ps(v_tv6, v_tv1);
        v_s41 = _mm_sub_ps(v_s9, v_tv8);
        // Output point 7: X(6)
        v_out6 = _mm_add_ps(v_s40, v_s41);
        // Output point 11: X(10)
        v_out10 = _mm_sub_ps(v_s40, v_s41);

        STR_128_S(curr_out, v_out_stride, v_out0, is_contiguous_out);
        curr_out = out + out_strides[1];
        STRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out + out_strides[3];
        STRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);
        curr_out = out + out_strides[5];
        STRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);
        curr_out = out + out_strides[7];
        STRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);
        curr_out = out + out_strides[9];
        STRI_2x128_S(curr_out, v_out_stride, v_out9, v_out10);
        curr_out = out + out_strides[11];
        STRI_2x128_S(curr_out, v_out_stride, v_out11, v_out12);
        curr_out = out + out_strides[13];
        STRI_2x128_S(curr_out, v_out_stride, v_out13, v_out14);
        curr_out = out + out_strides[15];
        STR_128_S(curr_out, v_out_stride, v_out15, is_contiguous_out);

        in += v_in_stride * NUM_SETS_REAL_128_S;
        out += v_out_stride * NUM_SETS_REAL_128_S;
    }
    // tail cases
    if (n & 2)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9, v_in10, v_in11, v_in12, v_in13, v_in14, v_in15;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
               v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27,
               v_s28, v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36,
               v_s37, v_s38, v_s39, v_s40, v_s41;
        __m128 v_tv0, v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7, v_tv8,
               v_tv9, v_tv10, v_tv11;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15;

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
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDHR_128_S(curr_in, v_in_stride, v_in11);
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDHR_128_S(curr_in, v_in_stride, v_in12);
        // Input point 14: x(13)
        curr_in = in + in_strides[13];
        LDHR_128_S(curr_in, v_in_stride, v_in13);
        // Input point 15: x(14)
        curr_in = in + in_strides[14];
        LDHR_128_S(curr_in, v_in_stride, v_in14);
        // Input point 16: x(15)
        curr_in = in + in_strides[15];
        LDHR_128_S(curr_in, v_in_stride, v_in15);

        v_s0 = _mm_add_ps(v_in0, v_in8);
        v_s1 = _mm_sub_ps(v_in0, v_in8);
        v_s2 = _mm_add_ps(v_in1, v_in15);
        v_s3 = _mm_sub_ps(v_in1, v_in15);
        v_s4 = _mm_add_ps(v_in2, v_in6);
        v_s5 = _mm_sub_ps(v_in2, v_in6);
        v_s6 = _mm_add_ps(v_in3, v_in5);
        v_s7 = _mm_sub_ps(v_in3, v_in5);
        v_s8 = _mm_add_ps(v_in4, v_in12);
        v_s9 = _mm_sub_ps(v_in4, v_in12);
        v_s10 = _mm_add_ps(v_in7, v_in9);
        v_s11 = _mm_sub_ps(v_in7, v_in9);
        v_s12 = _mm_add_ps(v_in10, v_in14);
        v_s13 = _mm_sub_ps(v_in10, v_in14);
        v_s14 = _mm_add_ps(v_in11, v_in13);
        v_s15 = _mm_sub_ps(v_in11, v_in13);

        v_s16 = _mm_add_ps(v_s0, v_s8);
        v_s17 = _mm_sub_ps(v_s0, v_s8);
        v_s18 = _mm_add_ps(v_s2, v_s10);
        v_s19 = _mm_sub_ps(v_s2, v_s10);
        v_s20 = _mm_add_ps(v_s3, v_s11);
        v_s21 = _mm_sub_ps(v_s11, v_s3);
        v_s22 = _mm_add_ps(v_s4, v_s12);
        v_s23 = _mm_sub_ps(v_s4, v_s12);
        // Output point 8: X(7)
        v_out7 = _mm_sub_ps(v_s16, v_s22);

        v_s24 = _mm_add_ps(v_s5, v_s13);
        v_s25 = _mm_sub_ps(v_s5, v_s13);
        v_s26 = _mm_add_ps(v_s6, v_s14);
        v_s27 = _mm_sub_ps(v_s6, v_s14);
        v_s28 = _mm_add_ps(v_s7, v_s15);
        v_s29 = _mm_sub_ps(v_s7, v_s15);
        // Output point 9: X(8)
        v_out8 = _mm_add_ps(v_s28, v_s21);

        v_s30 = _mm_add_ps(v_s18, v_s26);
        v_s31 = _mm_sub_ps(v_s18, v_s26);
        v_s32 = _mm_add_ps(v_s16, v_s22);
        v_s33 = _mm_sub_ps(v_s21, v_s28);
        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_s30, v_s32);
        // Output point 16: X(15)
        v_out15 = _mm_sub_ps(v_s32, v_s30);

        v_tv0 = _mm_mul_ps(v_s19, v_CRTM_16_1);
        v_tv1 = _mm_mul_ps(v_s20, v_CRTM_16_1);
        v_tv2 = _mm_mul_ps(v_s27, v_CRTM_16_1);
        v_tv3 = _mm_mul_ps(v_s29, v_CRTM_16_1);
        v_tv4 = _mm_mul_ps(v_s19, v_CRTM_16_2);
        v_tv5 = _mm_mul_ps(v_s20, v_CRTM_16_2);
        v_tv6 = _mm_mul_ps(v_s27, v_CRTM_16_2);
        v_tv7 = _mm_mul_ps(v_s29, v_CRTM_16_2);
        v_tv8 = _mm_mul_ps(v_s23, v_CRTM_16_3);
        v_tv9 = _mm_mul_ps(v_s25, v_CRTM_16_3);
        v_tv10 = _mm_mul_ps(v_s31, v_CRTM_16_3);
        // Output point 4: X(3)
        v_out3 = _mm_add_ps(v_s17, v_tv10);
        // Output point 12: X(11)
        v_out11 = _mm_sub_ps(v_s17, v_tv10);

        v_tv11 = _mm_mul_ps(v_s33, v_CRTM_16_3);
        // Output point 5: X(4)
        v_out4 = _mm_sub_ps(v_tv11, v_s24);
        // Output point 13: X(12)
        v_out12 = _mm_add_ps(v_s24, v_tv11);

        v_s34 = _mm_add_ps(v_tv0, v_tv7);
        v_s35 = _mm_add_ps(v_tv9, v_s1);
        // Output point 2: X(1)
        v_out1 = _mm_add_ps(v_s34, v_s35);
        // Output point 14: X(13)
        v_out13 = _mm_sub_ps(v_s35, v_s34);

        v_s36 = _mm_add_ps(v_tv2, v_tv5);
        v_s37 = _mm_add_ps(v_tv8, v_s9);
        // Output point 3: X(2)
        v_out2 = NEGATE_128_S(_mm_add_ps(v_s36, v_s37));
        // Output point 15: X(14)
        v_out14 = _mm_sub_ps(v_s37, v_s36);

        v_s38 = _mm_sub_ps(v_tv3, v_tv4);
        v_s39 = _mm_sub_ps(v_s1, v_tv9);
        // Output point 6: X(5)
        v_out5 = _mm_sub_ps(v_s39, v_s38);
        // Output point 10: X(9)
        v_out9 = _mm_add_ps(v_s38, v_s39);

        v_s40 = _mm_sub_ps(v_tv6, v_tv1);
        v_s41 = _mm_sub_ps(v_s9, v_tv8);
        // Output point 7: X(6)
        v_out6 = _mm_add_ps(v_s40, v_s41);
        // Output point 11: X(10)
        v_out10 = _mm_sub_ps(v_s40, v_s41);

        STHR_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STHRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out + out_strides[3];
        STHRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);
        curr_out = out + out_strides[5];
        STHRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);
        curr_out = out + out_strides[7];
        STHRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);
        curr_out = out + out_strides[9];
        STHRI_2x128_S(curr_out, v_out_stride, v_out9, v_out10);
        curr_out = out + out_strides[11];
        STHRI_2x128_S(curr_out, v_out_stride, v_out11, v_out12);
        curr_out = out + out_strides[13];
        STHRI_2x128_S(curr_out, v_out_stride, v_out13, v_out14);
        curr_out = out + out_strides[15];
        STHR_128_S(curr_out, v_out_stride, v_out15);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        FFTZ_FLOAT in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11,
              in12, in13, in14, in15;
        FFTZ_FLOAT s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13,
            s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26,
            s27, s28, s29, s30, s31, s32, s33, s34, s35, s36, s37, s38, s39,
            s40, s41;
        FFTZ_FLOAT t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11;

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
        // Input point 12: x(11)
        in11 = in[in_strides[11]];
        // Input point 13: x(12)
        in12 = in[in_strides[12]];
        // Input point 14: x(13)
        in13 = in[in_strides[13]];
        // Input point 15: x(14)
        in14 = in[in_strides[14]];
        // Input point 16: x(15)
        in15 = in[in_strides[15]];

        s0 = in0 + in8;
        s1 = in0 - in8;
        s2 = in1 + in15;
        s3 = in1 - in15;
        s4 = in2 + in6;
        s5 = in2 - in6;
        s6 = in3 + in5;
        s7 = in3 - in5;
        s8 = in4 + in12;
        s9 = in4 - in12;
        s10 = in7 + in9;
        s11 = in7 - in9;
        s12 = in10 + in14;
        s13 = in10 - in14;
        s14 = in11 + in13;
        s15 = in11 - in13;

        s16 = s0 + s8;
        s17 = s0 - s8;
        s18 = s2 + s10;
        s19 = s2 - s10;
        s20 = s3 + s11;
        s21 = s11 - s3;
        s22 = s4 + s12;
        s23 = s4 - s12;
        // Output point 8: X(7)
        out[out_strides[7]] = s16 - s22;

        s24 = s5 + s13;
        s25 = s5 - s13;
        s26 = s6 + s14;
        s27 = s6 - s14;
        s28 = s7 + s15;
        s29 = s7 - s15;
        // Output point 9: X(8)
        out[out_strides[8]] = s28 + s21;

        s30 = s18 + s26;
        s31 = s18 - s26;
        s32 = s16 + s22;
        s33 = s21 - s28;
        // Output point 1: X(0)
        *out = s30 + s32;
        // Output point 16: X(15)
        out[out_strides[15]] = s32 - s30;

        t0 = s19 * CRTM_16_1;
        t1 = s20 * CRTM_16_1;
        t2 = s27 * CRTM_16_1;
        t3 = s29 * CRTM_16_1;
        t4 = s19 * CRTM_16_2;
        t5 = s20 * CRTM_16_2;
        t6 = s27 * CRTM_16_2;
        t7 = s29 * CRTM_16_2;
        t8 = s23 * CRTM_16_3;
        t9 = s25 * CRTM_16_3;
        t10 = s31 * CRTM_16_3;
        // Output point 4: X(3)
        out[out_strides[3]] = s17 + t10;
        // Output point 12: X(11)
        out[out_strides[11]] = s17 - t10;

        t11 = s33 * CRTM_16_3;
        // Output point 5: X(4)
        out[out_strides[4]] = t11 - s24;
        // Output point 13: X(12)
        out[out_strides[12]] = s24 + t11;

        s34 = t0 + t7;
        s35 = t9 + s1;
        // Output point 2: X(1)
        out[out_strides[1]] = s34 + s35;
        // Output point 14: X(13)
        out[out_strides[13]] = s35 - s34;

        s36 = t2 + t5;
        s37 = t8 + s9;
        // Output point 3: X(2)
        out[out_strides[2]] = -(s36 + s37);
        // Output point 15: X(14)
        out[out_strides[14]] = s37 - s36;

        s38 = t3 - t4;
        s39 = s1 - t9;
        // Output point 6: X(5)
        out[out_strides[5]] = s39 - s38;
        // Output point 10: X(9)
        out[out_strides[9]] = s38 + s39;

        s40 = t6 - t1;
        s41 = s9 - t8;
        // Output point 7: X(6)
        out[out_strides[6]] = s40 + s41;
        // Output point 11: X(10)
        out[out_strides[10]] = s40 - s41;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft16avx128_fp32_bwd(FFTZ_VOID *in_real,
                                            FFTZ_VOID *in_imag,
                                            FFTZ_VOID *out_real,
                                            FFTZ_VOID *out_imag, FFTZ_INTP n,
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

    __m128 v_CRTM_16_1 = _mm_broadcast_ss(&CRTM_16_1);
    __m128 v_CRTM_16_2 = _mm_broadcast_ss(&CRTM_16_2);
    __m128 v_CRTM_16_3 = _mm_broadcast_ss(&CRTM_16_3);
    __m128 v_CRTM_16_4 = _mm_broadcast_ss(&CRTM_16_4);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9, v_in10, v_in11, v_in12, v_in13, v_in14, v_in15;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
               v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27,
               v_s28, v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36,
               v_s37, v_s38, v_s39, v_s40, v_s41;
        __m128 v_tv0, v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7, v_tv8,
               v_tv9, v_tv10, v_tv11, v_tv12, v_tv13, v_tv14, v_tv15, v_tv16,
               v_tv17;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_stride, v_in0, is_contiguous_in);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_S(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x128_S(curr_in, v_in_stride, v_in3, v_in4);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDRI_2x128_S(curr_in, v_in_stride, v_in5, v_in6);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in + in_strides[7];
        LDRI_2x128_S(curr_in, v_in_stride, v_in7, v_in8);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDRI_2x128_S(curr_in, v_in_stride, v_in9, v_in10);
        // Input point 12: x(11) & Input point 13: x(12)
        curr_in = in + in_strides[11];
        LDRI_2x128_S(curr_in, v_in_stride, v_in11, v_in12);
        // Input point 14: x(13) & Input point 15: x(14)
        curr_in = in + in_strides[13];
        LDRI_2x128_S(curr_in, v_in_stride, v_in13, v_in14);
        // Input point 16: x(15)
        curr_in = in + in_strides[15];
        LDR_128_S(curr_in, v_in_stride, v_in15, is_contiguous_in);

        v_s0 = _mm_add_ps(v_in0, v_in15);
        v_s1 = _mm_sub_ps(v_in0, v_in15);
        v_s2 = _mm_add_ps(v_in1, v_in13);
        v_s3 = _mm_sub_ps(v_in1, v_in13);
        v_s4 = _mm_add_ps(v_in2, v_in14);
        v_s5 = _mm_sub_ps(v_in2, v_in14);
        v_s6 = _mm_add_ps(v_in3, v_in11);
        v_s7 = _mm_sub_ps(v_in3, v_in11);
        v_s8 = _mm_add_ps(v_in4, v_in12);
        v_s9 = _mm_sub_ps(v_in4, v_in12);
        v_s10 = _mm_add_ps(v_in5, v_in9);
        v_s11 = _mm_sub_ps(v_in5, v_in9);
        v_s12 = _mm_add_ps(v_in6, v_in10);
        v_s13 = _mm_sub_ps(v_in6, v_in10);

        v_s14 = _mm_add_ps(v_s2, v_s10);
        v_s15 = _mm_sub_ps(v_s2, v_s10);
        v_s16 = _mm_add_ps(v_s3, v_s12);
        v_s17 = _mm_sub_ps(v_s3, v_s12);
        v_s18 = _mm_add_ps(v_s4, v_s11);
        v_s19 = _mm_sub_ps(v_s4, v_s11);
        v_s20 = _mm_add_ps(v_s5, v_s13);
        v_s21 = _mm_sub_ps(v_s5, v_s13);
        v_s22 = _mm_add_ps(v_s7, v_s8);
        v_s23 = _mm_sub_ps(v_s7, v_s8);

        v_tv0 = _mm_mul_ps(v_in7, v_CRTM_16_4);
        v_tv3 = _mm_mul_ps(v_s14, v_CRTM_16_4);
        v_tv2 = _mm_mul_ps(v_s6, v_CRTM_16_4);
        v_s24 = _mm_add_ps(v_s0, v_tv0);
        v_s26 = _mm_add_ps(v_tv2, v_s24);
        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_s26, v_tv3);
        STR_128_S(curr_out, v_out_stride, v_out0, is_contiguous_out);
        // Output point 9: X(8)
        v_out8 = _mm_sub_ps(v_s26, v_tv3);
        curr_out = out + out_strides[8];
        STR_128_S(curr_out, v_out_stride, v_out8, is_contiguous_out);

        v_tv4 = _mm_mul_ps(v_s21, v_CRTM_16_4);
        v_s27 = _mm_sub_ps(v_s24, v_tv2);
        // Output point 5: X(4)
        v_out4 = _mm_sub_ps(v_s27, v_tv4);
        curr_out = out + out_strides[4];
        STR_128_S(curr_out, v_out_stride, v_out4, is_contiguous_out);
        // Output point 13: X(12)
        v_out12 = _mm_add_ps(v_s27, v_tv4);
        curr_out = out + out_strides[12];
        STR_128_S(curr_out, v_out_stride, v_out12, is_contiguous_out);

        v_tv1 = _mm_mul_ps(v_in8, v_CRTM_16_4);
        v_tv5 = _mm_mul_ps(v_s23, v_CRTM_16_3);
        v_tv6 = _mm_mul_ps(v_s17, v_CRTM_16_1);
        v_tv7 = _mm_mul_ps(v_s19, v_CRTM_16_2);
        v_s29 = _mm_sub_ps(v_s1, v_tv1);
        v_s32 = _mm_add_ps(v_tv5, v_s29);
        v_s33 = _mm_sub_ps(v_tv6, v_tv7);
        // Output point 2: X(1)
        v_out1 = _mm_add_ps(v_s32, v_s33);
        curr_out = out + out_strides[1];
        STR_128_S(curr_out, v_out_stride, v_out1, is_contiguous_out);
        // Output point 10: X(9)
        v_out9 = _mm_sub_ps(v_s32, v_s33);
        curr_out = out + out_strides[9];
        STR_128_S(curr_out, v_out_stride, v_out9, is_contiguous_out);

        v_s25 = _mm_sub_ps(v_s0, v_tv0);
        v_s28 = _mm_add_ps(v_s1, v_tv1);
        v_tv8 = _mm_mul_ps(v_s22, v_CRTM_16_3);
        v_tv9 = _mm_mul_ps(v_s18, v_CRTM_16_1);
        v_tv10 = _mm_mul_ps(v_s16, v_CRTM_16_2);
        v_s34 = _mm_sub_ps(v_s28, v_tv8);
        v_s35 = _mm_sub_ps(v_tv10, v_tv9);
        // Output point 4: X(3)
        v_out3 = _mm_add_ps(v_s34, v_s35);
        curr_out = out + out_strides[3];
        STR_128_S(curr_out, v_out_stride, v_out3, is_contiguous_out);
        // Output point 12: X(11)
        v_out11 = _mm_sub_ps(v_s34, v_s35);
        curr_out = out + out_strides[11];
        STR_128_S(curr_out, v_out_stride, v_out11, is_contiguous_out);

        v_tv13 = _mm_mul_ps(v_s16, v_CRTM_16_1);
        v_tv14 = _mm_mul_ps(v_s18, v_CRTM_16_2);
        v_s38 = _mm_add_ps(v_s28, v_tv8);
        v_s39 = _mm_add_ps(v_tv13, v_tv14);
        // Output point 8: X(7)
        v_out7 = _mm_sub_ps(v_s38, v_s39);
        curr_out = out + out_strides[7];
        STR_128_S(curr_out, v_out_stride, v_out7, is_contiguous_out);
        // Output point 16: X(15)
        v_out15 = _mm_add_ps(v_s38, v_s39);
        curr_out = out + out_strides[15];
        STR_128_S(curr_out, v_out_stride, v_out15, is_contiguous_out);

        v_tv11 = _mm_mul_ps(v_s19, v_CRTM_16_1);
        v_tv12 = _mm_mul_ps(v_s17, v_CRTM_16_2);
        v_s36 = _mm_sub_ps(v_s29, v_tv5);
        v_s37 = _mm_add_ps(v_tv11, v_tv12);
        // Output point 6: X(5)
        v_out5 = _mm_sub_ps(v_s36, v_s37);
        curr_out = out + out_strides[5];
        STR_128_S(curr_out, v_out_stride, v_out5, is_contiguous_out);
        // Output point 14: X(13)
        v_out13 = _mm_add_ps(v_s36, v_s37);
        curr_out = out + out_strides[13];
        STR_128_S(curr_out, v_out_stride, v_out13, is_contiguous_out);

        v_s31 = _mm_sub_ps(v_s15, v_s20);
        v_tv15 = _mm_mul_ps(v_s31, v_CRTM_16_3);
        v_tv17 = _mm_mul_ps(v_s9, v_CRTM_16_4);
        v_s40 = _mm_sub_ps(v_s25, v_tv17);
        // Output point 3: X(2)
        v_out2 = _mm_add_ps(v_s40, v_tv15);
        curr_out = out + out_strides[2];
        STR_128_S(curr_out, v_out_stride, v_out2, is_contiguous_out);
        // Output point 11: X(10)
        v_out10 = _mm_sub_ps(v_s40, v_tv15);
        curr_out = out + out_strides[10];
        STR_128_S(curr_out, v_out_stride, v_out10, is_contiguous_out);

        v_s30 = _mm_add_ps(v_s15, v_s20);
        v_tv16 = _mm_mul_ps(v_s30, v_CRTM_16_3);
        v_s41 = _mm_add_ps(v_s25, v_tv17);
        // Output point 7: X(6)
        v_out6 = _mm_sub_ps(v_s41, v_tv16);
        curr_out = out + out_strides[6];
        STR_128_S(curr_out, v_out_stride, v_out6, is_contiguous_out);
        // Output point 15: X(14)
        v_out14 = _mm_add_ps(v_s41, v_tv16);
        curr_out = out + out_strides[14];
        STR_128_S(curr_out, v_out_stride, v_out14, is_contiguous_out);

        in += v_in_stride * NUM_SETS_REAL_128_S;
        out += v_out_stride * NUM_SETS_REAL_128_S;
    }
    // tail cases
    if (n & 2)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9, v_in10, v_in11, v_in12, v_in13, v_in14, v_in15;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
               v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27,
               v_s28, v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36,
               v_s37, v_s38, v_s39, v_s40, v_s41;
        __m128 v_tv0, v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7, v_tv8,
               v_tv9, v_tv10, v_tv11, v_tv12, v_tv13, v_tv14, v_tv15, v_tv16,
               v_tv17;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDHR_128_S(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDHRI_2x128_S(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDHRI_2x128_S(curr_in, v_in_stride, v_in3, v_in4);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDHRI_2x128_S(curr_in, v_in_stride, v_in5, v_in6);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in + in_strides[7];
        LDHRI_2x128_S(curr_in, v_in_stride, v_in7, v_in8);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDHRI_2x128_S(curr_in, v_in_stride, v_in9, v_in10);
        // Input point 12: x(11) & Input point 13: x(12)
        curr_in = in + in_strides[11];
        LDHRI_2x128_S(curr_in, v_in_stride, v_in11, v_in12);
        // Input point 14: x(13) & Input point 15: x(14)
        curr_in = in + in_strides[13];
        LDHRI_2x128_S(curr_in, v_in_stride, v_in13, v_in14);
        // Input point 16: x(15)
        curr_in = in + in_strides[15];
        LDHR_128_S(curr_in, v_in_stride, v_in15);

        v_s0 = _mm_add_ps(v_in0, v_in15);
        v_s1 = _mm_sub_ps(v_in0, v_in15);
        v_s2 = _mm_add_ps(v_in1, v_in13);
        v_s3 = _mm_sub_ps(v_in1, v_in13);
        v_s4 = _mm_add_ps(v_in2, v_in14);
        v_s5 = _mm_sub_ps(v_in2, v_in14);
        v_s6 = _mm_add_ps(v_in3, v_in11);
        v_s7 = _mm_sub_ps(v_in3, v_in11);
        v_s8 = _mm_add_ps(v_in4, v_in12);
        v_s9 = _mm_sub_ps(v_in4, v_in12);
        v_s10 = _mm_add_ps(v_in5, v_in9);
        v_s11 = _mm_sub_ps(v_in5, v_in9);
        v_s12 = _mm_add_ps(v_in6, v_in10);
        v_s13 = _mm_sub_ps(v_in6, v_in10);

        v_s14 = _mm_add_ps(v_s2, v_s10);
        v_s15 = _mm_sub_ps(v_s2, v_s10);
        v_s16 = _mm_add_ps(v_s3, v_s12);
        v_s17 = _mm_sub_ps(v_s3, v_s12);
        v_s18 = _mm_add_ps(v_s4, v_s11);
        v_s19 = _mm_sub_ps(v_s4, v_s11);
        v_s20 = _mm_add_ps(v_s5, v_s13);
        v_s21 = _mm_sub_ps(v_s5, v_s13);
        v_s22 = _mm_add_ps(v_s7, v_s8);
        v_s23 = _mm_sub_ps(v_s7, v_s8);

        v_tv0 = _mm_mul_ps(v_in7, v_CRTM_16_4);
        v_tv3 = _mm_mul_ps(v_s14, v_CRTM_16_4);
        v_tv2 = _mm_mul_ps(v_s6, v_CRTM_16_4);
        v_s24 = _mm_add_ps(v_s0, v_tv0);
        v_s26 = _mm_add_ps(v_tv2, v_s24);
        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_s26, v_tv3);
        STHR_128_S(curr_out, v_out_stride, v_out0);
        // Output point 9: X(8)
        v_out8 = _mm_sub_ps(v_s26, v_tv3);
        curr_out = out + out_strides[8];
        STHR_128_S(curr_out, v_out_stride, v_out8);

        v_tv4 = _mm_mul_ps(v_s21, v_CRTM_16_4);
        v_s27 = _mm_sub_ps(v_s24, v_tv2);
        // Output point 5: X(4)
        v_out4 = _mm_sub_ps(v_s27, v_tv4);
        curr_out = out + out_strides[4];
        STHR_128_S(curr_out, v_out_stride, v_out4);
        // Output point 13: X(12)
        v_out12 = _mm_add_ps(v_s27, v_tv4);
        curr_out = out + out_strides[12];
        STHR_128_S(curr_out, v_out_stride, v_out12);

        v_tv1 = _mm_mul_ps(v_in8, v_CRTM_16_4);
        v_tv5 = _mm_mul_ps(v_s23, v_CRTM_16_3);
        v_tv6 = _mm_mul_ps(v_s17, v_CRTM_16_1);
        v_tv7 = _mm_mul_ps(v_s19, v_CRTM_16_2);
        v_s29 = _mm_sub_ps(v_s1, v_tv1);
        v_s32 = _mm_add_ps(v_tv5, v_s29);
        v_s33 = _mm_sub_ps(v_tv6, v_tv7);
        // Output point 2: X(1)
        v_out1 = _mm_add_ps(v_s32, v_s33);
        curr_out = out + out_strides[1];
        STHR_128_S(curr_out, v_out_stride, v_out1);
        // Output point 10: X(9)
        v_out9 = _mm_sub_ps(v_s32, v_s33);
        curr_out = out + out_strides[9];
        STHR_128_S(curr_out, v_out_stride, v_out9);

        v_s25 = _mm_sub_ps(v_s0, v_tv0);
        v_s28 = _mm_add_ps(v_s1, v_tv1);
        v_tv8 = _mm_mul_ps(v_s22, v_CRTM_16_3);
        v_tv9 = _mm_mul_ps(v_s18, v_CRTM_16_1);
        v_tv10 = _mm_mul_ps(v_s16, v_CRTM_16_2);
        v_s34 = _mm_sub_ps(v_s28, v_tv8);
        v_s35 = _mm_sub_ps(v_tv10, v_tv9);
        // Output point 4: X(3)
        v_out3 = _mm_add_ps(v_s34, v_s35);
        curr_out = out + out_strides[3];
        STHR_128_S(curr_out, v_out_stride, v_out3);
        // Output point 12: X(11)
        v_out11 = _mm_sub_ps(v_s34, v_s35);
        curr_out = out + out_strides[11];
        STHR_128_S(curr_out, v_out_stride, v_out11);

        v_tv13 = _mm_mul_ps(v_s16, v_CRTM_16_1);
        v_tv14 = _mm_mul_ps(v_s18, v_CRTM_16_2);
        v_s38 = _mm_add_ps(v_s28, v_tv8);
        v_s39 = _mm_add_ps(v_tv13, v_tv14);
        // Output point 8: X(7)
        v_out7 = _mm_sub_ps(v_s38, v_s39);
        curr_out = out + out_strides[7];
        STHR_128_S(curr_out, v_out_stride, v_out7);
        // Output point 16: X(15)
        v_out15 = _mm_add_ps(v_s38, v_s39);
        curr_out = out + out_strides[15];
        STHR_128_S(curr_out, v_out_stride, v_out15);

        v_tv11 = _mm_mul_ps(v_s19, v_CRTM_16_1);
        v_tv12 = _mm_mul_ps(v_s17, v_CRTM_16_2);
        v_s36 = _mm_sub_ps(v_s29, v_tv5);
        v_s37 = _mm_add_ps(v_tv11, v_tv12);
        // Output point 6: X(5)
        v_out5 = _mm_sub_ps(v_s36, v_s37);
        curr_out = out + out_strides[5];
        STHR_128_S(curr_out, v_out_stride, v_out5);
        // Output point 14: X(13)
        v_out13 = _mm_add_ps(v_s36, v_s37);
        curr_out = out + out_strides[13];
        STHR_128_S(curr_out, v_out_stride, v_out13);

        v_s31 = _mm_sub_ps(v_s15, v_s20);
        v_tv15 = _mm_mul_ps(v_s31, v_CRTM_16_3);
        v_tv17 = _mm_mul_ps(v_s9, v_CRTM_16_4);
        v_s40 = _mm_sub_ps(v_s25, v_tv17);
        // Output point 3: X(2)
        v_out2 = _mm_add_ps(v_s40, v_tv15);
        curr_out = out + out_strides[2];
        STHR_128_S(curr_out, v_out_stride, v_out2);
        // Output point 11: X(10)
        v_out10 = _mm_sub_ps(v_s40, v_tv15);
        curr_out = out + out_strides[10];
        STHR_128_S(curr_out, v_out_stride, v_out10);

        v_s30 = _mm_add_ps(v_s15, v_s20);
        v_tv16 = _mm_mul_ps(v_s30, v_CRTM_16_3);
        v_s41 = _mm_add_ps(v_s25, v_tv17);
        // Output point 7: X(6)
        v_out6 = _mm_sub_ps(v_s41, v_tv16);
        curr_out = out + out_strides[6];
        STHR_128_S(curr_out, v_out_stride, v_out6);
        // Output point 15: X(14)
        v_out14 = _mm_add_ps(v_s41, v_tv16);
        curr_out = out + out_strides[14];
        STHR_128_S(curr_out, v_out_stride, v_out14);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        FFTZ_FLOAT in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11,
              in12, in13, in14, in15;
        FFTZ_FLOAT s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13,
            s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26,
            s27, s28, s29, s30, s31, s32, s33, s34, s35, s36, s37, s38, s39,
            s40, s41;
        FFTZ_FLOAT t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13,
            t14, t15, t16, t17;

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
        // Input point 12: x(11)
        in11 = in[in_strides[11]];
        // Input point 13: x(12)
        in12 = in[in_strides[12]];
        // Input point 14: x(13)
        in13 = in[in_strides[13]];
        // Input point 15: x(14)
        in14 = in[in_strides[14]];
        // Input point 16: x(15)
        in15 = in[in_strides[15]];

        s0 = in0 + in15;
        s1 = in0 - in15;
        s2 = in1 + in13;
        s3 = in1 - in13;
        s4 = in2 + in14;
        s5 = in2 - in14;
        s6 = in3 + in11;
        s7 = in3 - in11;
        s8 = in4 + in12;
        s9 = in4 - in12;
        s10 = in5 + in9;
        s11 = in5 - in9;
        s12 = in6 + in10;
        s13 = in6 - in10;

        s14 = s2 + s10;
        s15 = s2 - s10;
        s16 = s3 + s12;
        s17 = s3 - s12;
        s18 = s4 + s11;
        s19 = s4 - s11;
        s20 = s5 + s13;
        s21 = s5 - s13;
        s22 = s7 + s8;
        s23 = s7 - s8;

        t0 = in7 * CRTM_16_4;
        t3 = s14 * CRTM_16_4;
        t2 = s6 * CRTM_16_4;
        s24 = s0 + t0;
        s26 = t2 + s24;
        // Output point 1: X(0)
        *out = s26 + t3;
        // Output point 9: X(8)
        out[out_strides[8]] = s26 - t3;

        t4 = s21 * CRTM_16_4;
        s27 = s24 - t2;
        // Output point 5: X(4)
        out[out_strides[4]] = s27 - t4;
        // Output point 13: X(12)
        out[out_strides[12]] = s27 + t4;

        t1 = in8 * CRTM_16_4;
        t5 = s23 * CRTM_16_3;
        t6 = s17 * CRTM_16_1;
        t7 = s19 * CRTM_16_2;
        s29 = s1 - t1;
        s32 = t5 + s29;
        s33 = t6 - t7;
        // Output point 2: X(1)
        out[out_strides[1]] = s32 + s33;
        // Output point 10: X(9)
        out[out_strides[9]] = s32 - s33;

        s25 = s0 - t0;
        s28 = s1 + t1;
        t8 = s22 * CRTM_16_3;
        t9 = s18 * CRTM_16_1;
        t10 = s16 * CRTM_16_2;
        s34 = s28 - t8;
        s35 = t10 - t9;
        // Output point 4: X(3)
        out[out_strides[3]] = s34 + s35;
        // Output point 12: X(11)
        out[out_strides[11]] = s34 - s35;

        t13 = s16 * CRTM_16_1;
        t14 = s18 * CRTM_16_2;
        s38 = s28 + t8;
        s39 = t13 + t14;
        // Output point 8: X(7)
        out[out_strides[7]] = s38 - s39;
        // Output point 16: X(15)
        out[out_strides[15]] = s38 + s39;

        t11 = s19 * CRTM_16_1;
        t12 = s17 * CRTM_16_2;
        s36 = s29 - t5;
        s37 = t11 + t12;
        // Output point 6: X(5)
        out[out_strides[5]] = s36 - s37;
        // Output point 14: X(13)
        out[out_strides[13]] = s36 + s37;

        s31 = s15 - s20;
        t15 = s31 * CRTM_16_3;
        t17 = s9 * CRTM_16_4;
        s40 = s25 - t17;
        // Output point 3: X(2)
        out[out_strides[2]] = s40 + t15;
        // Output point 11: X(10)
        out[out_strides[10]] = s40 - t15;

        s30 = s15 + s20;
        t16 = s30 * CRTM_16_3;
        s41 = s25 + t17;
        // Output point 7: X(6)
        out[out_strides[6]] = s41 - t16;
        // Output point 15: X(14)
        out[out_strides[14]] = s41 + t16;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft16avx128_fp64_fwd(FFTZ_VOID *in_real,
                                            FFTZ_VOID *in_imag,
                                            FFTZ_VOID *out_real,
                                            FFTZ_VOID *out_imag, FFTZ_INTP n,
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

    __m128d v_CRTM_16_1 = _mm_set1_pd(CRTM_16_1);
    __m128d v_CRTM_16_2 = _mm_set1_pd(CRTM_16_2);
    __m128d v_CRTM_16_3 = _mm_set1_pd(CRTM_16_3);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
                v_in9, v_in10, v_in11, v_in12, v_in13, v_in14, v_in15;
        __m128d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
                v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
                v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27,
                v_s28, v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36,
                v_s37, v_s38, v_s39, v_s40, v_s41;
        __m128d v_tv0, v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7, v_tv8,
                v_tv9, v_tv10, v_tv11;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
                v_out15;

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
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDR_128_D(curr_in, v_in_stride, v_in9, is_contiguous_in);
        // Input point 11: x(10)
        curr_in = in + in_strides[10];
        LDR_128_D(curr_in, v_in_stride, v_in10, is_contiguous_in);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_128_D(curr_in, v_in_stride, v_in11, is_contiguous_in);
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDR_128_D(curr_in, v_in_stride, v_in12, is_contiguous_in);
        // Input point 14: x(13)
        curr_in = in + in_strides[13];
        LDR_128_D(curr_in, v_in_stride, v_in13, is_contiguous_in);
        // Input point 15: x(14)
        curr_in = in + in_strides[14];
        LDR_128_D(curr_in, v_in_stride, v_in14, is_contiguous_in);
        // Input point 16: x(15)
        curr_in = in + in_strides[15];
        LDR_128_D(curr_in, v_in_stride, v_in15, is_contiguous_in);

        v_s0 = _mm_add_pd(v_in0, v_in8);
        v_s1 = _mm_sub_pd(v_in0, v_in8);
        v_s2 = _mm_add_pd(v_in1, v_in15);
        v_s3 = _mm_sub_pd(v_in1, v_in15);
        v_s4 = _mm_add_pd(v_in2, v_in6);
        v_s5 = _mm_sub_pd(v_in2, v_in6);
        v_s6 = _mm_add_pd(v_in3, v_in5);
        v_s7 = _mm_sub_pd(v_in3, v_in5);
        v_s8 = _mm_add_pd(v_in4, v_in12);
        v_s9 = _mm_sub_pd(v_in4, v_in12);
        v_s10 = _mm_add_pd(v_in7, v_in9);
        v_s11 = _mm_sub_pd(v_in7, v_in9);
        v_s12 = _mm_add_pd(v_in10, v_in14);
        v_s13 = _mm_sub_pd(v_in10, v_in14);
        v_s14 = _mm_add_pd(v_in11, v_in13);
        v_s15 = _mm_sub_pd(v_in11, v_in13);

        v_s16 = _mm_add_pd(v_s0, v_s8);
        v_s17 = _mm_sub_pd(v_s0, v_s8);
        v_s18 = _mm_add_pd(v_s2, v_s10);
        v_s19 = _mm_sub_pd(v_s2, v_s10);
        v_s20 = _mm_add_pd(v_s3, v_s11);
        v_s21 = _mm_sub_pd(v_s11, v_s3);
        v_s22 = _mm_add_pd(v_s4, v_s12);
        v_s23 = _mm_sub_pd(v_s4, v_s12);
        // Output point 8: X(7)
        v_out7 = _mm_sub_pd(v_s16, v_s22);

        v_s24 = _mm_add_pd(v_s5, v_s13);
        v_s25 = _mm_sub_pd(v_s5, v_s13);
        v_s26 = _mm_add_pd(v_s6, v_s14);
        v_s27 = _mm_sub_pd(v_s6, v_s14);
        v_s28 = _mm_add_pd(v_s7, v_s15);
        v_s29 = _mm_sub_pd(v_s7, v_s15);
        // Output point 9: X(8)
        v_out8 = _mm_add_pd(v_s28, v_s21);

        v_s30 = _mm_add_pd(v_s18, v_s26);
        v_s31 = _mm_sub_pd(v_s18, v_s26);
        v_s32 = _mm_add_pd(v_s16, v_s22);
        v_s33 = _mm_sub_pd(v_s21, v_s28);
        // Output point 1: X(0)
        v_out0 = _mm_add_pd(v_s30, v_s32);
        // Output point 16: X(15)
        v_out15 = _mm_sub_pd(v_s32, v_s30);

        v_tv0 = _mm_mul_pd(v_s19, v_CRTM_16_1);
        v_tv1 = _mm_mul_pd(v_s20, v_CRTM_16_1);
        v_tv2 = _mm_mul_pd(v_s27, v_CRTM_16_1);
        v_tv3 = _mm_mul_pd(v_s29, v_CRTM_16_1);
        v_tv4 = _mm_mul_pd(v_s19, v_CRTM_16_2);
        v_tv5 = _mm_mul_pd(v_s20, v_CRTM_16_2);
        v_tv6 = _mm_mul_pd(v_s27, v_CRTM_16_2);
        v_tv7 = _mm_mul_pd(v_s29, v_CRTM_16_2);
        v_tv8 = _mm_mul_pd(v_s23, v_CRTM_16_3);
        v_tv9 = _mm_mul_pd(v_s25, v_CRTM_16_3);
        v_tv10 = _mm_mul_pd(v_s31, v_CRTM_16_3);
        // Output point 4: X(3)
        v_out3 = _mm_add_pd(v_s17, v_tv10);
        // Output point 12: X(11)
        v_out11 = _mm_sub_pd(v_s17, v_tv10);

        v_tv11 = _mm_mul_pd(v_s33, v_CRTM_16_3);
        // Output point 5: X(4)
        v_out4 = _mm_sub_pd(v_tv11, v_s24);
        // Output point 13: X(12)
        v_out12 = _mm_add_pd(v_s24, v_tv11);

        v_s34 = _mm_add_pd(v_tv0, v_tv7);
        v_s35 = _mm_add_pd(v_tv9, v_s1);
        // Output point 2: X(1)
        v_out1 = _mm_add_pd(v_s34, v_s35);
        // Output point 14: X(13)
        v_out13 = _mm_sub_pd(v_s35, v_s34);

        v_s36 = _mm_add_pd(v_tv2, v_tv5);
        v_s37 = _mm_add_pd(v_tv8, v_s9);
        // Output point 3: X(2)
        v_out2 = NEGATE_128_D(_mm_add_pd(v_s36, v_s37));
        // Output point 15: X(14)
        v_out14 = _mm_sub_pd(v_s37, v_s36);

        v_s38 = _mm_sub_pd(v_tv3, v_tv4);
        v_s39 = _mm_sub_pd(v_s1, v_tv9);
        // Output point 6: X(5)
        v_out5 = _mm_sub_pd(v_s39, v_s38);
        // Output point 10: X(9)
        v_out9 = _mm_add_pd(v_s38, v_s39);

        v_s40 = _mm_sub_pd(v_tv6, v_tv1);
        v_s41 = _mm_sub_pd(v_s9, v_tv8);
        // Output point 7: X(6)
        v_out6 = _mm_add_pd(v_s40, v_s41);
        // Output point 11: X(10)
        v_out10 = _mm_sub_pd(v_s40, v_s41);

        STR_128_D(curr_out, v_out_stride, v_out0, is_contiguous_out);
        curr_out = out + out_strides[1];
        STRI_2x128_D(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out + out_strides[3];
        STRI_2x128_D(curr_out, v_out_stride, v_out3, v_out4);
        curr_out = out + out_strides[5];
        STRI_2x128_D(curr_out, v_out_stride, v_out5, v_out6);
        curr_out = out + out_strides[7];
        STRI_2x128_D(curr_out, v_out_stride, v_out7, v_out8);
        curr_out = out + out_strides[9];
        STRI_2x128_D(curr_out, v_out_stride, v_out9, v_out10);
        curr_out = out + out_strides[11];
        STRI_2x128_D(curr_out, v_out_stride, v_out11, v_out12);
        curr_out = out + out_strides[13];
        STRI_2x128_D(curr_out, v_out_stride, v_out13, v_out14);
        curr_out = out + out_strides[15];
        STR_128_D(curr_out, v_out_stride, v_out15, is_contiguous_out);

        in += v_in_stride * NUM_SETS_REAL_128_D;
        out += v_out_stride * NUM_SETS_REAL_128_D;
    }
    // tail cases
    if (n & 1)
    {
        FFTZ_DOUBLE in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10,
            in11, in12, in13, in14, in15;
        FFTZ_DOUBLE s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13,
            s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26,
            s27, s28, s29, s30, s31, s32, s33, s34, s35, s36, s37, s38, s39,
            s40, s41;
        FFTZ_DOUBLE t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11;

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
        // Input point 12: x(11)
        in11 = in[in_strides[11]];
        // Input point 13: x(12)
        in12 = in[in_strides[12]];
        // Input point 14: x(13)
        in13 = in[in_strides[13]];
        // Input point 15: x(14)
        in14 = in[in_strides[14]];
        // Input point 16: x(15)
        in15 = in[in_strides[15]];

        s0 = in0 + in8;
        s1 = in0 - in8;
        s2 = in1 + in15;
        s3 = in1 - in15;
        s4 = in2 + in6;
        s5 = in2 - in6;
        s6 = in3 + in5;
        s7 = in3 - in5;
        s8 = in4 + in12;
        s9 = in4 - in12;
        s10 = in7 + in9;
        s11 = in7 - in9;
        s12 = in10 + in14;
        s13 = in10 - in14;
        s14 = in11 + in13;
        s15 = in11 - in13;

        s16 = s0 + s8;
        s17 = s0 - s8;
        s18 = s2 + s10;
        s19 = s2 - s10;
        s20 = s3 + s11;
        s21 = s11 - s3;
        s22 = s4 + s12;
        s23 = s4 - s12;
        // Output point 8: X(7)
        out[out_strides[7]] = s16 - s22;

        s24 = s5 + s13;
        s25 = s5 - s13;
        s26 = s6 + s14;
        s27 = s6 - s14;
        s28 = s7 + s15;
        s29 = s7 - s15;
        // Output point 9: X(8)
        out[out_strides[8]] = s28 + s21;

        s30 = s18 + s26;
        s31 = s18 - s26;
        s32 = s16 + s22;
        s33 = s21 - s28;
        // Output point 1: X(0)
        *out = s30 + s32;
        // Output point 16: X(15)
        out[out_strides[15]] = s32 - s30;

        t0 = s19 * CRTM_16_1;
        t1 = s20 * CRTM_16_1;
        t2 = s27 * CRTM_16_1;
        t3 = s29 * CRTM_16_1;
        t4 = s19 * CRTM_16_2;
        t5 = s20 * CRTM_16_2;
        t6 = s27 * CRTM_16_2;
        t7 = s29 * CRTM_16_2;
        t8 = s23 * CRTM_16_3;
        t9 = s25 * CRTM_16_3;
        t10 = s31 * CRTM_16_3;
        // Output point 4: X(3)
        out[out_strides[3]] = s17 + t10;
        // Output point 12: X(11)
        out[out_strides[11]] = s17 - t10;

        t11 = s33 * CRTM_16_3;
        // Output point 5: X(4)
        out[out_strides[4]] = t11 - s24;
        // Output point 13: X(12)
        out[out_strides[12]] = s24 + t11;

        s34 = t0 + t7;
        s35 = t9 + s1;
        // Output point 2: X(1)
        out[out_strides[1]] = s34 + s35;
        // Output point 14: X(13)
        out[out_strides[13]] = s35 - s34;

        s36 = t2 + t5;
        s37 = t8 + s9;
        // Output point 3: X(2)
        out[out_strides[2]] = -(s36 + s37);
        // Output point 15: X(14)
        out[out_strides[14]] = s37 - s36;

        s38 = t3 - t4;
        s39 = s1 - t9;
        // Output point 6: X(5)
        out[out_strides[5]] = s39 - s38;
        // Output point 10: X(9)
        out[out_strides[9]] = s38 + s39;

        s40 = t6 - t1;
        s41 = s9 - t8;
        // Output point 7: X(6)
        out[out_strides[6]] = s40 + s41;
        // Output point 11: X(10)
        out[out_strides[10]] = s40 - s41;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft16avx128_fp64_bwd(FFTZ_VOID *in_real,
                                            FFTZ_VOID *in_imag,
                                            FFTZ_VOID *out_real,
                                            FFTZ_VOID *out_imag, FFTZ_INTP n,
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

    __m128d v_CRTM_16_1 = _mm_set1_pd(CRTM_16_1);
    __m128d v_CRTM_16_2 = _mm_set1_pd(CRTM_16_2);
    __m128d v_CRTM_16_3 = _mm_set1_pd(CRTM_16_3);
    __m128d v_CRTM_16_4 = _mm_set1_pd(CRTM_16_4);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
                v_in9, v_in10, v_in11, v_in12, v_in13, v_in14, v_in15;
        __m128d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
                v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
                v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27,
                v_s28, v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36,
                v_s37, v_s38, v_s39, v_s40, v_s41;
        __m128d v_tv0, v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6, v_tv7, v_tv8,
                v_tv9, v_tv10, v_tv11, v_tv12, v_tv13, v_tv14, v_tv15, v_tv16,
                v_tv17;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
                v_out15;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_stride, v_in0, is_contiguous_in);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_D(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x128_D(curr_in, v_in_stride, v_in3, v_in4);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDRI_2x128_D(curr_in, v_in_stride, v_in5, v_in6);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in + in_strides[7];
        LDRI_2x128_D(curr_in, v_in_stride, v_in7, v_in8);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDRI_2x128_D(curr_in, v_in_stride, v_in9, v_in10);
        // Input point 12: x(11) & Input point 13: x(12)
        curr_in = in + in_strides[11];
        LDRI_2x128_D(curr_in, v_in_stride, v_in11, v_in12);
        // Input point 14: x(13) & Input point 15: x(14)
        curr_in = in + in_strides[13];
        LDRI_2x128_D(curr_in, v_in_stride, v_in13, v_in14);
        // Input point 16: x(15)
        curr_in = in + in_strides[15];
        LDR_128_D(curr_in, v_in_stride, v_in15, is_contiguous_in);

        v_s0 = _mm_add_pd(v_in0, v_in15);
        v_s1 = _mm_sub_pd(v_in0, v_in15);
        v_s2 = _mm_add_pd(v_in1, v_in13);
        v_s3 = _mm_sub_pd(v_in1, v_in13);
        v_s4 = _mm_add_pd(v_in2, v_in14);
        v_s5 = _mm_sub_pd(v_in2, v_in14);
        v_s6 = _mm_add_pd(v_in3, v_in11);
        v_s7 = _mm_sub_pd(v_in3, v_in11);
        v_s8 = _mm_add_pd(v_in4, v_in12);
        v_s9 = _mm_sub_pd(v_in4, v_in12);
        v_s10 = _mm_add_pd(v_in5, v_in9);
        v_s11 = _mm_sub_pd(v_in5, v_in9);
        v_s12 = _mm_add_pd(v_in6, v_in10);
        v_s13 = _mm_sub_pd(v_in6, v_in10);

        v_s14 = _mm_add_pd(v_s2, v_s10);
        v_s15 = _mm_sub_pd(v_s2, v_s10);
        v_s16 = _mm_add_pd(v_s3, v_s12);
        v_s17 = _mm_sub_pd(v_s3, v_s12);
        v_s18 = _mm_add_pd(v_s4, v_s11);
        v_s19 = _mm_sub_pd(v_s4, v_s11);
        v_s20 = _mm_add_pd(v_s5, v_s13);
        v_s21 = _mm_sub_pd(v_s5, v_s13);
        v_s22 = _mm_add_pd(v_s7, v_s8);
        v_s23 = _mm_sub_pd(v_s7, v_s8);

        v_tv0 = _mm_mul_pd(v_in7, v_CRTM_16_4);
        v_tv3 = _mm_mul_pd(v_s14, v_CRTM_16_4);
        v_tv2 = _mm_mul_pd(v_s6, v_CRTM_16_4);
        v_s24 = _mm_add_pd(v_s0, v_tv0);
        v_s26 = _mm_add_pd(v_tv2, v_s24);
        // Output point 1: X(0)
        v_out0 = _mm_add_pd(v_s26, v_tv3);
        STR_128_D(curr_out, v_out_stride, v_out0, is_contiguous_out);
        // Output point 9: X(8)
        v_out8 = _mm_sub_pd(v_s26, v_tv3);
        curr_out = out + out_strides[8];
        STR_128_D(curr_out, v_out_stride, v_out8, is_contiguous_out);

        v_tv4 = _mm_mul_pd(v_s21, v_CRTM_16_4);
        v_s27 = _mm_sub_pd(v_s24, v_tv2);
        // Output point 5: X(4)
        v_out4 = _mm_sub_pd(v_s27, v_tv4);
        curr_out = out + out_strides[4];
        STR_128_D(curr_out, v_out_stride, v_out4, is_contiguous_out);
        // Output point 13: X(12)
        v_out12 = _mm_add_pd(v_s27, v_tv4);
        curr_out = out + out_strides[12];
        STR_128_D(curr_out, v_out_stride, v_out12, is_contiguous_out);

        v_tv1 = _mm_mul_pd(v_in8, v_CRTM_16_4);
        v_tv5 = _mm_mul_pd(v_s23, v_CRTM_16_3);
        v_tv6 = _mm_mul_pd(v_s17, v_CRTM_16_1);
        v_tv7 = _mm_mul_pd(v_s19, v_CRTM_16_2);
        v_s29 = _mm_sub_pd(v_s1, v_tv1);
        v_s32 = _mm_add_pd(v_tv5, v_s29);
        v_s33 = _mm_sub_pd(v_tv6, v_tv7);
        // Output point 2: X(1)
        v_out1 = _mm_add_pd(v_s32, v_s33);
        curr_out = out + out_strides[1];
        STR_128_D(curr_out, v_out_stride, v_out1, is_contiguous_out);
        // Output point 10: X(9)
        v_out9 = _mm_sub_pd(v_s32, v_s33);
        curr_out = out + out_strides[9];
        STR_128_D(curr_out, v_out_stride, v_out9, is_contiguous_out);

        v_s25 = _mm_sub_pd(v_s0, v_tv0);
        v_s28 = _mm_add_pd(v_s1, v_tv1);
        v_tv8 = _mm_mul_pd(v_s22, v_CRTM_16_3);
        v_tv9 = _mm_mul_pd(v_s18, v_CRTM_16_1);
        v_tv10 = _mm_mul_pd(v_s16, v_CRTM_16_2);
        v_s34 = _mm_sub_pd(v_s28, v_tv8);
        v_s35 = _mm_sub_pd(v_tv10, v_tv9);
        // Output point 4: X(3)
        v_out3 = _mm_add_pd(v_s34, v_s35);
        curr_out = out + out_strides[3];
        STR_128_D(curr_out, v_out_stride, v_out3, is_contiguous_out);
        // Output point 12: X(11)
        v_out11 = _mm_sub_pd(v_s34, v_s35);
        curr_out = out + out_strides[11];
        STR_128_D(curr_out, v_out_stride, v_out11, is_contiguous_out);

        v_tv13 = _mm_mul_pd(v_s16, v_CRTM_16_1);
        v_tv14 = _mm_mul_pd(v_s18, v_CRTM_16_2);
        v_s38 = _mm_add_pd(v_s28, v_tv8);
        v_s39 = _mm_add_pd(v_tv13, v_tv14);
        // Output point 8: X(7)
        v_out7 = _mm_sub_pd(v_s38, v_s39);
        curr_out = out + out_strides[7];
        STR_128_D(curr_out, v_out_stride, v_out7, is_contiguous_out);
        // Output point 16: X(15)
        v_out15 = _mm_add_pd(v_s38, v_s39);
        curr_out = out + out_strides[15];
        STR_128_D(curr_out, v_out_stride, v_out15, is_contiguous_out);

        v_tv11 = _mm_mul_pd(v_s19, v_CRTM_16_1);
        v_tv12 = _mm_mul_pd(v_s17, v_CRTM_16_2);
        v_s36 = _mm_sub_pd(v_s29, v_tv5);
        v_s37 = _mm_add_pd(v_tv11, v_tv12);
        // Output point 6: X(5)
        v_out5 = _mm_sub_pd(v_s36, v_s37);
        curr_out = out + out_strides[5];
        STR_128_D(curr_out, v_out_stride, v_out5, is_contiguous_out);
        // Output point 14: X(13)
        v_out13 = _mm_add_pd(v_s36, v_s37);
        curr_out = out + out_strides[13];
        STR_128_D(curr_out, v_out_stride, v_out13, is_contiguous_out);

        v_s31 = _mm_sub_pd(v_s15, v_s20);
        v_tv15 = _mm_mul_pd(v_s31, v_CRTM_16_3);
        v_tv17 = _mm_mul_pd(v_s9, v_CRTM_16_4);
        v_s40 = _mm_sub_pd(v_s25, v_tv17);
        // Output point 3: X(2)
        v_out2 = _mm_add_pd(v_s40, v_tv15);
        curr_out = out + out_strides[2];
        STR_128_D(curr_out, v_out_stride, v_out2, is_contiguous_out);
        // Output point 11: X(10)
        v_out10 = _mm_sub_pd(v_s40, v_tv15);
        curr_out = out + out_strides[10];
        STR_128_D(curr_out, v_out_stride, v_out10, is_contiguous_out);

        v_s30 = _mm_add_pd(v_s15, v_s20);
        v_tv16 = _mm_mul_pd(v_s30, v_CRTM_16_3);
        v_s41 = _mm_add_pd(v_s25, v_tv17);
        // Output point 7: X(6)
        v_out6 = _mm_sub_pd(v_s41, v_tv16);
        curr_out = out + out_strides[6];
        STR_128_D(curr_out, v_out_stride, v_out6, is_contiguous_out);
        // Output point 15: X(14)
        v_out14 = _mm_add_pd(v_s41, v_tv16);
        curr_out = out + out_strides[14];
        STR_128_D(curr_out, v_out_stride, v_out14, is_contiguous_out);

        in += v_in_stride * NUM_SETS_REAL_128_D;
        out += v_out_stride * NUM_SETS_REAL_128_D;
    }
    // tail cases
    if (n & 1)
    {
        FFTZ_DOUBLE in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10,
            in11, in12, in13, in14, in15;
        FFTZ_DOUBLE s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13,
            s14, s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26,
            s27, s28, s29, s30, s31, s32, s33, s34, s35, s36, s37, s38, s39,
            s40, s41;
        FFTZ_DOUBLE t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13,
            t14, t15, t16, t17;

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
        // Input point 12: x(11)
        in11 = in[in_strides[11]];
        // Input point 13: x(12)
        in12 = in[in_strides[12]];
        // Input point 14: x(13)
        in13 = in[in_strides[13]];
        // Input point 15: x(14)
        in14 = in[in_strides[14]];
        // Input point 16: x(15)
        in15 = in[in_strides[15]];

        s0 = in0 + in15;
        s1 = in0 - in15;
        s2 = in1 + in13;
        s3 = in1 - in13;
        s4 = in2 + in14;
        s5 = in2 - in14;
        s6 = in3 + in11;
        s7 = in3 - in11;
        s8 = in4 + in12;
        s9 = in4 - in12;
        s10 = in5 + in9;
        s11 = in5 - in9;
        s12 = in6 + in10;
        s13 = in6 - in10;

        s14 = s2 + s10;
        s15 = s2 - s10;
        s16 = s3 + s12;
        s17 = s3 - s12;
        s18 = s4 + s11;
        s19 = s4 - s11;
        s20 = s5 + s13;
        s21 = s5 - s13;
        s22 = s7 + s8;
        s23 = s7 - s8;

        t0 = in7 * CRTM_16_4;
        t3 = s14 * CRTM_16_4;
        t2 = s6 * CRTM_16_4;
        s24 = s0 + t0;
        s26 = t2 + s24;
        // Output point 1: X(0)
        *out = s26 + t3;
        // Output point 9: X(8)
        out[out_strides[8]] = s26 - t3;

        t4 = s21 * CRTM_16_4;
        s27 = s24 - t2;
        // Output point 5: X(4)
        out[out_strides[4]] = s27 - t4;
        // Output point 13: X(12)
        out[out_strides[12]] = s27 + t4;

        t1 = in8 * CRTM_16_4;
        t5 = s23 * CRTM_16_3;
        t6 = s17 * CRTM_16_1;
        t7 = s19 * CRTM_16_2;
        s29 = s1 - t1;
        s32 = t5 + s29;
        s33 = t6 - t7;
        // Output point 2: X(1)
        out[out_strides[1]] = s32 + s33;
        // Output point 10: X(9)
        out[out_strides[9]] = s32 - s33;

        s25 = s0 - t0;
        s28 = s1 + t1;
        t8 = s22 * CRTM_16_3;
        t9 = s18 * CRTM_16_1;
        t10 = s16 * CRTM_16_2;
        s34 = s28 - t8;
        s35 = t10 - t9;
        // Output point 4: X(3)
        out[out_strides[3]] = s34 + s35;
        // Output point 12: X(11)
        out[out_strides[11]] = s34 - s35;

        t13 = s16 * CRTM_16_1;
        t14 = s18 * CRTM_16_2;
        s38 = s28 + t8;
        s39 = t13 + t14;
        // Output point 8: X(7)
        out[out_strides[7]] = s38 - s39;
        // Output point 16: X(15)
        out[out_strides[15]] = s38 + s39;

        t11 = s19 * CRTM_16_1;
        t12 = s17 * CRTM_16_2;
        s36 = s29 - t5;
        s37 = t11 + t12;
        // Output point 6: X(5)
        out[out_strides[5]] = s36 - s37;
        // Output point 14: X(13)
        out[out_strides[13]] = s36 + s37;

        s31 = s15 - s20;
        t15 = s31 * CRTM_16_3;
        t17 = s9 * CRTM_16_4;
        s40 = s25 - t17;
        // Output point 3: X(2)
        out[out_strides[2]] = s40 + t15;
        // Output point 11: X(10)
        out[out_strides[10]] = s40 - t15;

        s30 = s15 + s20;
        t16 = s30 * CRTM_16_3;
        s41 = s25 + t17;
        // Output point 7: X(6)
        out[out_strides[6]] = s41 - t16;
        // Output point 15: X(14)
        out[out_strides[14]] = s41 + t16;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hc_rfft16avx128(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hc_rfft16avx128_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft16avx128_fp64_fwd;
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
            return r2hc_rfft16avx128_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft16avx128_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
