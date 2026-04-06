// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fft20avx128.c
 *
 *  @brief Radix-20 FFT kernel with AVX-128 operations using x86 SIMD intrinsics
 *
 *  This file contains the DIT radix-20 FFT implementations using AVX-128
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Jeevanantham N
 *  @author Jeya R
 *
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 32, 94, 80, 18, 9},
                                                     {0, 32, 94, 40, 18, 9}};

ops_cycles_t get_ops_cnt_fft20avx128(UINT8 precision, UINT8 direction)
{
    if (precision == DT_FLOAT)
    {
        return ops_cnt[0];
    }
    else
    {
        return ops_cnt[1];
    }
}

static VOID fft20avx128fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                            VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                            VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");

    FLOAT *in_r = (FLOAT *)in_real;
    FLOAT *out_r = (FLOAT *)out_real;
    FLOAT *curr_in = NULL;
    FLOAT *curr_out = NULL;

#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;

    INTP N = n / NUM_SETS_128_S;
    INTP remaining_sets = n % NUM_SETS_128_S;
    INTP count = 0;

    /* Radix-20 twiddle constants */
    const FLOAT CRTM_20[4] = {
        0.55901699437494742410229341718281905886015458990288,
        0.61803398874989484820458683436563811772030917980576,
        0.95105651629515357211643933337938214340569863400000,
        0.25000000000000000000000000000000000000000000000000};
    __m128 v_K1 = _mm_broadcast_ss(&CRTM_20[0]);
    __m128 v_K2 = _mm_broadcast_ss(&CRTM_20[1]);
    __m128 v_K3 = _mm_broadcast_ss(&CRTM_20[2]);
    __m128 v_K4 = _mm_broadcast_ss(&CRTM_20[3]);
    __m128 v_neg_zero = _mm_set1_ps(-0.0f);

    /*
     * Phase shift macros for complex numbers in interleaved format:
     *   PS_P90(b, c) = c + j*b  (+90 degree phase shift on b, then add to c)
     *   PS_N90(b, c) = c - j*b  (-90 degree phase shift on b, then add to c)
     */
    #define PS_P90_128(b, c) _mm_addsub_ps(c, SWAP_RI_128_S(b))
    #define PS_N90_128(b, c) \
        _mm_addsub_ps(c, _mm_xor_ps(SWAP_RI_128_S(b), v_neg_zero))

    for (count = 0; count < N; count++)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4;
        __m128 v_in5, v_in6, v_in7, v_in8, v_in9;
        __m128 v_in10, v_in11, v_in12, v_in13, v_in14;
        __m128 v_in15, v_in16, v_in17, v_in18, v_in19;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4;
        __m128 v_out5, v_out6, v_out7, v_out8, v_out9;
        __m128 v_out10, v_out11, v_out12, v_out13, v_out14;
        __m128 v_out15, v_out16, v_out17, v_out18, v_out19;

        curr_in = in_r;
        curr_out = out_r;

        GATHER2_128_S(curr_in, v_in_stride, v_in0);
        GATHER2_128_S(in_r + in_strides[1], v_in_stride, v_in1);
        GATHER2_128_S(in_r + in_strides[2], v_in_stride, v_in2);
        GATHER2_128_S(in_r + in_strides[3], v_in_stride, v_in3);
        GATHER2_128_S(in_r + in_strides[4], v_in_stride, v_in4);
        GATHER2_128_S(in_r + in_strides[5], v_in_stride, v_in5);
        GATHER2_128_S(in_r + in_strides[6], v_in_stride, v_in6);
        GATHER2_128_S(in_r + in_strides[7], v_in_stride, v_in7);
        GATHER2_128_S(in_r + in_strides[8], v_in_stride, v_in8);
        GATHER2_128_S(in_r + in_strides[9], v_in_stride, v_in9);
        GATHER2_128_S(in_r + in_strides[10], v_in_stride, v_in10);
        GATHER2_128_S(in_r + in_strides[11], v_in_stride, v_in11);
        GATHER2_128_S(in_r + in_strides[12], v_in_stride, v_in12);
        GATHER2_128_S(in_r + in_strides[13], v_in_stride, v_in13);
        GATHER2_128_S(in_r + in_strides[14], v_in_stride, v_in14);
        GATHER2_128_S(in_r + in_strides[15], v_in_stride, v_in15);
        GATHER2_128_S(in_r + in_strides[16], v_in_stride, v_in16);
        GATHER2_128_S(in_r + in_strides[17], v_in_stride, v_in17);
        GATHER2_128_S(in_r + in_strides[18], v_in_stride, v_in18);
        GATHER2_128_S(in_r + in_strides[19], v_in_stride, v_in19);

        __m128 v_s1 = _mm_add_ps(v_in0, v_in10);
        __m128 v_s2 = _mm_add_ps(v_in5, v_in15);
        __m128 v_d1 = _mm_sub_ps(v_in0, v_in10);
        __m128 v_cv1 = _mm_add_ps(v_s1, v_s2);
        __m128 v_d2 = _mm_sub_ps(v_in5, v_in15);
        __m128 v_cv2 = _mm_sub_ps(v_s1, v_s2);

        __m128 v_d3 = _mm_sub_ps(v_in4, v_in14);
        __m128 v_s3 = _mm_add_ps(v_in4, v_in14);
        __m128 v_d4 = _mm_sub_ps(v_in13, v_in3);
        __m128 v_s4 = _mm_add_ps(v_in13, v_in3);
        __m128 v_d5 = _mm_sub_ps(v_in17, v_in7);
        __m128 v_s5 = _mm_add_ps(v_in17, v_in7);
        __m128 v_d6 = _mm_sub_ps(v_in16, v_in6);
        __m128 v_s6 = _mm_add_ps(v_in16, v_in6);
        __m128 v_d7 = _mm_sub_ps(v_in8, v_in18);
        __m128 v_s7 = _mm_add_ps(v_in8, v_in18);
        __m128 v_d8 = _mm_sub_ps(v_in9, v_in19);
        __m128 v_s8 = _mm_add_ps(v_in9, v_in19);
        __m128 v_d9 = _mm_sub_ps(v_in1, v_in11);
        __m128 v_s9 = _mm_add_ps(v_in1, v_in11);
        __m128 v_d10 = _mm_sub_ps(v_in12, v_in2);
        __m128 v_s10 = _mm_add_ps(v_in12, v_in2);

        __m128 v_cv3 = _mm_sub_ps(v_d9, v_d8);
        __m128 v_cv4 = _mm_sub_ps(v_d3, v_d6);
        __m128 v_cv5 = _mm_sub_ps(v_d7, v_d10);
        __m128 v_cv6 = _mm_sub_ps(v_d5, v_d4);
        __m128 v_cv7 = _mm_sub_ps(v_s3, v_s8);
        __m128 v_cv8 = _mm_sub_ps(v_s6, v_s9);
        __m128 v_cv9 = _mm_add_ps(v_cv7, v_cv8);
        __m128 v_cv10 = _mm_add_ps(v_s7, v_s4);
        __m128 v_cv11 = _mm_add_ps(v_s10, v_s5);
        __m128 v_cv12 = _mm_add_ps(v_cv10, v_cv11);
        __m128 v_cv13 = _mm_add_ps(v_s3, v_s8);
        __m128 v_cv14 = _mm_add_ps(v_s6, v_s9);
        __m128 v_cv15 = _mm_add_ps(v_cv13, v_cv14);
        __m128 v_cv16 = _mm_sub_ps(v_s7, v_s4);
        __m128 v_cv17 = _mm_sub_ps(v_s10, v_s5);
        __m128 v_cv18 = _mm_add_ps(v_cv16, v_cv17);

        __m128 v_cv19 = _mm_add_ps(v_d3, v_d6);
        __m128 v_cv20 = _mm_add_ps(v_d7, v_d10);
        __m128 v_cv21 = _mm_add_ps(v_cv19, v_cv20);
        __m128 v_cv22 = _mm_sub_ps(v_cv19, v_cv20);
        __m128 v_cv23 = _mm_add_ps(v_d8, v_d9);
        __m128 v_cv24 = _mm_add_ps(v_d4, v_d5);
        __m128 v_cv25 = _mm_add_ps(v_cv23, v_cv24);
        __m128 v_cv26 = _mm_sub_ps(v_cv24, v_cv23);

        __m128 v_cv27 = _mm_add_ps(v_d1, v_cv21);
        __m128 v_cv28 = _mm_add_ps(v_d2, v_cv25);

        /* Output point 6 : X[5] */
        /* Output point 16 : X[15] */
        if (flag == 0)
        {
            v_out5 = PS_N90_128(v_cv28, v_cv27);
            v_out15 = PS_P90_128(v_cv28, v_cv27);
        }
        else
        {
            v_out5 = PS_P90_128(v_cv28, v_cv27);
            v_out15 = PS_N90_128(v_cv28, v_cv27);
        }

        __m128 v_av1 = _mm_sub_ps(v_cv15, v_cv12);
        __m128 v_av2 = _mm_add_ps(v_cv15, v_cv12);
        __m128 v_av3 = _mm_sub_ps(v_cv1, _mm_mul_ps(v_K4, v_av2));
        __m128 v_av4 = _mm_sub_ps(v_cv13, v_cv14);
        __m128 v_av5 = _mm_sub_ps(v_cv10, v_cv11);
        __m128 v_av6 = _mm_mul_ps(v_K3,
            _mm_add_ps(_mm_mul_ps(v_K2, v_av5), v_av4));
        __m128 v_av7 = _mm_mul_ps(v_K3,
            _mm_sub_ps(v_av5, _mm_mul_ps(v_K2, v_av4)));

        /* Output point 1 : X[0] */
        v_out0 = _mm_add_ps(v_cv1, v_av2);
        __m128 v_av8 = _mm_sub_ps(v_av3, _mm_mul_ps(v_K1, v_av1));
        __m128 v_av9 = _mm_add_ps(v_av3, _mm_mul_ps(v_K1, v_av1));

        /* Output point 9 : X[8], Output point 13 : X[12] */
        /* Output point 5 : X[4], Output point 17 : X[16] */
        if (flag == 0)
        {
            v_out8 = PS_N90_128(v_av7, v_av8);
            v_out12 = PS_P90_128(v_av7, v_av8);
            v_out4 = PS_P90_128(v_av6, v_av9);
            v_out16 = PS_N90_128(v_av6, v_av9);
        }
        else
        {
            v_out8 = PS_P90_128(v_av7, v_av8);
            v_out12 = PS_N90_128(v_av7, v_av8);
            v_out4 = PS_N90_128(v_av6, v_av9);
            v_out16 = PS_P90_128(v_av6, v_av9);
        }

        __m128 v_av10 = _mm_sub_ps(v_cv9, v_cv18);
        __m128 v_av11 = _mm_add_ps(v_cv9, v_cv18);
        __m128 v_av12 = _mm_sub_ps(v_cv2, _mm_mul_ps(v_K4, v_av11));
        __m128 v_av13 = _mm_sub_ps(v_cv16, v_cv17);
        __m128 v_av14 = _mm_sub_ps(v_cv7, v_cv8);
        __m128 v_av15 = _mm_mul_ps(v_K3,
            _mm_sub_ps(v_av13, _mm_mul_ps(v_K2, v_av14)));
        __m128 v_av16 = _mm_mul_ps(v_K3,
            _mm_add_ps(_mm_mul_ps(v_K2, v_av13), v_av14));

        /* Output point 11 : X[10] */
        v_out10 = _mm_add_ps(v_cv2, v_av11);
        __m128 v_av17 = _mm_add_ps(v_av12, _mm_mul_ps(v_K1, v_av10));
        __m128 v_av18 = _mm_sub_ps(v_av12, _mm_mul_ps(v_K1, v_av10));

        /* Output point 7 : X[6], Output point 15 : X[14] */
        /* Output point 3 : X[2], Output point 19 : X[18] */
        if (flag == 0)
        {
            v_out6 = PS_N90_128(v_av16, v_av17);
            v_out14 = PS_P90_128(v_av16, v_av17);
            v_out2 = PS_P90_128(v_av15, v_av18);
            v_out18 = PS_N90_128(v_av15, v_av18);
        }
        else
        {
            v_out6 = PS_P90_128(v_av16, v_av17);
            v_out14 = PS_N90_128(v_av16, v_av17);
            v_out2 = PS_N90_128(v_av15, v_av18);
            v_out18 = PS_P90_128(v_av15, v_av18);
        }

        __m128 v_av19 = _mm_add_ps(_mm_mul_ps(v_K2, v_cv6), v_cv3);
        __m128 v_av20 = _mm_add_ps(_mm_mul_ps(v_K2, v_cv5), v_cv4);
        __m128 v_av21 = _mm_sub_ps(v_cv5, _mm_mul_ps(v_K2, v_cv4));
        __m128 v_av22 = _mm_sub_ps(v_cv6, _mm_mul_ps(v_K2, v_cv3));
        __m128 v_av23 = _mm_sub_ps(v_d2, _mm_mul_ps(v_K4, v_cv25));
        __m128 v_av24 = _mm_sub_ps(v_av23, _mm_mul_ps(v_K1, v_cv26));
        __m128 v_av25 = _mm_add_ps(v_av23, _mm_mul_ps(v_K1, v_cv26));
        __m128 v_av26 = _mm_sub_ps(v_d1, _mm_mul_ps(v_K4, v_cv21));
        __m128 v_av27 = _mm_add_ps(v_av26, _mm_mul_ps(v_K1, v_cv22));
        __m128 v_av28 = _mm_sub_ps(v_av26, _mm_mul_ps(v_K1, v_cv22));

        __m128 v_av29 = _mm_add_ps(v_av27, _mm_mul_ps(v_K3, v_av19));
        __m128 v_av30 = _mm_add_ps(v_av24, _mm_mul_ps(v_K3, v_av20));
        __m128 v_av31 = _mm_add_ps(v_av28, _mm_mul_ps(v_K3, v_av22));
        __m128 v_av32 = _mm_add_ps(v_av25, _mm_mul_ps(v_K3, v_av21));
        __m128 v_av33 = _mm_sub_ps(v_av27, _mm_mul_ps(v_K3, v_av19));
        __m128 v_av34 = _mm_sub_ps(v_av24, _mm_mul_ps(v_K3, v_av20));
        __m128 v_av35 = _mm_sub_ps(v_av28, _mm_mul_ps(v_K3, v_av22));
        __m128 v_av36 = _mm_sub_ps(v_av25, _mm_mul_ps(v_K3, v_av21));

        /* Output point 2 : X[1], Output point 20 : X[19] */
        /* Output point 14 : X[13], Output point 8 : X[7] */
        /* Output point 10 : X[9], Output point 12 : X[11] */
        /* Output point 18 : X[17], Output point 4 : X[3] */
        if (flag == 0)
        {
            v_out1 = PS_N90_128(v_av30, v_av29);
            v_out19 = PS_P90_128(v_av30, v_av29);
            v_out13 = PS_N90_128(v_av32, v_av31);
            v_out7 = PS_P90_128(v_av32, v_av31);
            v_out9 = PS_N90_128(v_av34, v_av33);
            v_out11 = PS_P90_128(v_av34, v_av33);
            v_out17 = PS_N90_128(v_av36, v_av35);
            v_out3 = PS_P90_128(v_av36, v_av35);
        }
        else
        {
            v_out1 = PS_P90_128(v_av30, v_av29);
            v_out19 = PS_N90_128(v_av30, v_av29);
            v_out13 = PS_P90_128(v_av32, v_av31);
            v_out7 = PS_N90_128(v_av32, v_av31);
            v_out9 = PS_P90_128(v_av34, v_av33);
            v_out11 = PS_N90_128(v_av34, v_av33);
            v_out17 = PS_P90_128(v_av36, v_av35);
            v_out3 = PS_N90_128(v_av36, v_av35);
        }

        SCATTER2_128_S(curr_out, v_out_stride, v_out0);
        SCATTER2_128_S(out_r + out_strides[1], v_out_stride, v_out1);
        SCATTER2_128_S(out_r + out_strides[2], v_out_stride, v_out2);
        SCATTER2_128_S(out_r + out_strides[3], v_out_stride, v_out3);
        SCATTER2_128_S(out_r + out_strides[4], v_out_stride, v_out4);
        SCATTER2_128_S(out_r + out_strides[5], v_out_stride, v_out5);
        SCATTER2_128_S(out_r + out_strides[6], v_out_stride, v_out6);
        SCATTER2_128_S(out_r + out_strides[7], v_out_stride, v_out7);
        SCATTER2_128_S(out_r + out_strides[8], v_out_stride, v_out8);
        SCATTER2_128_S(out_r + out_strides[9], v_out_stride, v_out9);
        SCATTER2_128_S(out_r + out_strides[10], v_out_stride, v_out10);
        SCATTER2_128_S(out_r + out_strides[11], v_out_stride, v_out11);
        SCATTER2_128_S(out_r + out_strides[12], v_out_stride, v_out12);
        SCATTER2_128_S(out_r + out_strides[13], v_out_stride, v_out13);
        SCATTER2_128_S(out_r + out_strides[14], v_out_stride, v_out14);
        SCATTER2_128_S(out_r + out_strides[15], v_out_stride, v_out15);
        SCATTER2_128_S(out_r + out_strides[16], v_out_stride, v_out16);
        SCATTER2_128_S(out_r + out_strides[17], v_out_stride, v_out17);
        SCATTER2_128_S(out_r + out_strides[18], v_out_stride, v_out18);
        SCATTER2_128_S(out_r + out_strides[19], v_out_stride, v_out19);

        in_r += NUM_SETS_128_S * v_in_stride;
        out_r += NUM_SETS_128_S * v_out_stride;
        remaining_sets = remaining_sets % NUM_SETS_128_S;
    }

    /* Single element tail case (1 complex number) */
    if (remaining_sets & 1)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4;
        __m128 v_in5, v_in6, v_in7, v_in8, v_in9;
        __m128 v_in10, v_in11, v_in12, v_in13, v_in14;
        __m128 v_in15, v_in16, v_in17, v_in18, v_in19;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4;
        __m128 v_out5, v_out6, v_out7, v_out8, v_out9;
        __m128 v_out10, v_out11, v_out12, v_out13, v_out14;
        __m128 v_out15, v_out16, v_out17, v_out18, v_out19;

        curr_in = in_r;
        curr_out = out_r;

        LD_LOW_128_S(curr_in, v_in0);
        LD_LOW_128_S(in_r + in_strides[1], v_in1);
        LD_LOW_128_S(in_r + in_strides[2], v_in2);
        LD_LOW_128_S(in_r + in_strides[3], v_in3);
        LD_LOW_128_S(in_r + in_strides[4], v_in4);
        LD_LOW_128_S(in_r + in_strides[5], v_in5);
        LD_LOW_128_S(in_r + in_strides[6], v_in6);
        LD_LOW_128_S(in_r + in_strides[7], v_in7);
        LD_LOW_128_S(in_r + in_strides[8], v_in8);
        LD_LOW_128_S(in_r + in_strides[9], v_in9);
        LD_LOW_128_S(in_r + in_strides[10], v_in10);
        LD_LOW_128_S(in_r + in_strides[11], v_in11);
        LD_LOW_128_S(in_r + in_strides[12], v_in12);
        LD_LOW_128_S(in_r + in_strides[13], v_in13);
        LD_LOW_128_S(in_r + in_strides[14], v_in14);
        LD_LOW_128_S(in_r + in_strides[15], v_in15);
        LD_LOW_128_S(in_r + in_strides[16], v_in16);
        LD_LOW_128_S(in_r + in_strides[17], v_in17);
        LD_LOW_128_S(in_r + in_strides[18], v_in18);
        LD_LOW_128_S(in_r + in_strides[19], v_in19);

        __m128 v_s1 = _mm_add_ps(v_in0, v_in10);
        __m128 v_s2 = _mm_add_ps(v_in5, v_in15);
        __m128 v_d1 = _mm_sub_ps(v_in0, v_in10);
        __m128 v_cv1 = _mm_add_ps(v_s1, v_s2);
        __m128 v_d2 = _mm_sub_ps(v_in5, v_in15);
        __m128 v_cv2 = _mm_sub_ps(v_s1, v_s2);

        __m128 v_d3 = _mm_sub_ps(v_in4, v_in14);
        __m128 v_s3 = _mm_add_ps(v_in4, v_in14);
        __m128 v_d4 = _mm_sub_ps(v_in13, v_in3);
        __m128 v_s4 = _mm_add_ps(v_in13, v_in3);
        __m128 v_d5 = _mm_sub_ps(v_in17, v_in7);
        __m128 v_s5 = _mm_add_ps(v_in17, v_in7);
        __m128 v_d6 = _mm_sub_ps(v_in16, v_in6);
        __m128 v_s6 = _mm_add_ps(v_in16, v_in6);
        __m128 v_d7 = _mm_sub_ps(v_in8, v_in18);
        __m128 v_s7 = _mm_add_ps(v_in8, v_in18);
        __m128 v_d8 = _mm_sub_ps(v_in9, v_in19);
        __m128 v_s8 = _mm_add_ps(v_in9, v_in19);
        __m128 v_d9 = _mm_sub_ps(v_in1, v_in11);
        __m128 v_s9 = _mm_add_ps(v_in1, v_in11);
        __m128 v_d10 = _mm_sub_ps(v_in12, v_in2);
        __m128 v_s10 = _mm_add_ps(v_in12, v_in2);

        __m128 v_cv3 = _mm_sub_ps(v_d9, v_d8);
        __m128 v_cv4 = _mm_sub_ps(v_d3, v_d6);
        __m128 v_cv5 = _mm_sub_ps(v_d7, v_d10);
        __m128 v_cv6 = _mm_sub_ps(v_d5, v_d4);
        __m128 v_cv7 = _mm_sub_ps(v_s3, v_s8);
        __m128 v_cv8 = _mm_sub_ps(v_s6, v_s9);
        __m128 v_cv9 = _mm_add_ps(v_cv7, v_cv8);
        __m128 v_cv10 = _mm_add_ps(v_s7, v_s4);
        __m128 v_cv11 = _mm_add_ps(v_s10, v_s5);
        __m128 v_cv12 = _mm_add_ps(v_cv10, v_cv11);
        __m128 v_cv13 = _mm_add_ps(v_s3, v_s8);
        __m128 v_cv14 = _mm_add_ps(v_s6, v_s9);
        __m128 v_cv15 = _mm_add_ps(v_cv13, v_cv14);
        __m128 v_cv16 = _mm_sub_ps(v_s7, v_s4);
        __m128 v_cv17 = _mm_sub_ps(v_s10, v_s5);
        __m128 v_cv18 = _mm_add_ps(v_cv16, v_cv17);

        __m128 v_cv19 = _mm_add_ps(v_d3, v_d6);
        __m128 v_cv20 = _mm_add_ps(v_d7, v_d10);
        __m128 v_cv21 = _mm_add_ps(v_cv19, v_cv20);
        __m128 v_cv22 = _mm_sub_ps(v_cv19, v_cv20);
        __m128 v_cv23 = _mm_add_ps(v_d8, v_d9);
        __m128 v_cv24 = _mm_add_ps(v_d4, v_d5);
        __m128 v_cv25 = _mm_add_ps(v_cv23, v_cv24);
        __m128 v_cv26 = _mm_sub_ps(v_cv24, v_cv23);

        __m128 v_cv27 = _mm_add_ps(v_d1, v_cv21);
        __m128 v_cv28 = _mm_add_ps(v_d2, v_cv25);

        /* Output point 6 : X[5] */
        /* Output point 16 : X[15] */
        if (flag == 0)
        {
            v_out5 = PS_N90_128(v_cv28, v_cv27);
            v_out15 = PS_P90_128(v_cv28, v_cv27);
        }
        else
        {
            v_out5 = PS_P90_128(v_cv28, v_cv27);
            v_out15 = PS_N90_128(v_cv28, v_cv27);
        }

        __m128 v_av1 = _mm_sub_ps(v_cv15, v_cv12);
        __m128 v_av2 = _mm_add_ps(v_cv15, v_cv12);
        __m128 v_av3 = _mm_sub_ps(v_cv1, _mm_mul_ps(v_K4, v_av2));
        __m128 v_av4 = _mm_sub_ps(v_cv13, v_cv14);
        __m128 v_av5 = _mm_sub_ps(v_cv10, v_cv11);
        __m128 v_av6 = _mm_mul_ps(v_K3,
            _mm_add_ps(_mm_mul_ps(v_K2, v_av5), v_av4));
        __m128 v_av7 = _mm_mul_ps(v_K3,
            _mm_sub_ps(v_av5, _mm_mul_ps(v_K2, v_av4)));

        /* Output point 1 : X[0] */
        v_out0 = _mm_add_ps(v_cv1, v_av2);
        __m128 v_av8 = _mm_sub_ps(v_av3, _mm_mul_ps(v_K1, v_av1));
        __m128 v_av9 = _mm_add_ps(v_av3, _mm_mul_ps(v_K1, v_av1));

        /* Output point 9 : X[8], Output point 13 : X[12] */
        /* Output point 5 : X[4], Output point 17 : X[16] */
        if (flag == 0)
        {
            v_out8 = PS_N90_128(v_av7, v_av8);
            v_out12 = PS_P90_128(v_av7, v_av8);
            v_out4 = PS_P90_128(v_av6, v_av9);
            v_out16 = PS_N90_128(v_av6, v_av9);
        }
        else
        {
            v_out8 = PS_P90_128(v_av7, v_av8);
            v_out12 = PS_N90_128(v_av7, v_av8);
            v_out4 = PS_N90_128(v_av6, v_av9);
            v_out16 = PS_P90_128(v_av6, v_av9);
        }

        __m128 v_av10 = _mm_sub_ps(v_cv9, v_cv18);
        __m128 v_av11 = _mm_add_ps(v_cv9, v_cv18);
        __m128 v_av12 = _mm_sub_ps(v_cv2, _mm_mul_ps(v_K4, v_av11));
        __m128 v_av13 = _mm_sub_ps(v_cv16, v_cv17);
        __m128 v_av14 = _mm_sub_ps(v_cv7, v_cv8);
        __m128 v_av15 = _mm_mul_ps(v_K3,
            _mm_sub_ps(v_av13, _mm_mul_ps(v_K2, v_av14)));
        __m128 v_av16 = _mm_mul_ps(v_K3,
            _mm_add_ps(_mm_mul_ps(v_K2, v_av13), v_av14));

        /* Output point 11 : X[10] */
        v_out10 = _mm_add_ps(v_cv2, v_av11);
        __m128 v_av17 = _mm_add_ps(v_av12, _mm_mul_ps(v_K1, v_av10));
        __m128 v_av18 = _mm_sub_ps(v_av12, _mm_mul_ps(v_K1, v_av10));

        /* Output point 7 : X[6], Output point 15 : X[14] */
        /* Output point 3 : X[2], Output point 19 : X[18] */
        if (flag == 0)
        {
            v_out6 = PS_N90_128(v_av16, v_av17);
            v_out14 = PS_P90_128(v_av16, v_av17);
            v_out2 = PS_P90_128(v_av15, v_av18);
            v_out18 = PS_N90_128(v_av15, v_av18);
        }
        else
        {
            v_out6 = PS_P90_128(v_av16, v_av17);
            v_out14 = PS_N90_128(v_av16, v_av17);
            v_out2 = PS_N90_128(v_av15, v_av18);
            v_out18 = PS_P90_128(v_av15, v_av18);
        }

        __m128 v_av19 = _mm_add_ps(_mm_mul_ps(v_K2, v_cv6), v_cv3);
        __m128 v_av20 = _mm_add_ps(_mm_mul_ps(v_K2, v_cv5), v_cv4);
        __m128 v_av21 = _mm_sub_ps(v_cv5, _mm_mul_ps(v_K2, v_cv4));
        __m128 v_av22 = _mm_sub_ps(v_cv6, _mm_mul_ps(v_K2, v_cv3));
        __m128 v_av23 = _mm_sub_ps(v_d2, _mm_mul_ps(v_K4, v_cv25));
        __m128 v_av24 = _mm_sub_ps(v_av23, _mm_mul_ps(v_K1, v_cv26));
        __m128 v_av25 = _mm_add_ps(v_av23, _mm_mul_ps(v_K1, v_cv26));
        __m128 v_av26 = _mm_sub_ps(v_d1, _mm_mul_ps(v_K4, v_cv21));
        __m128 v_av27 = _mm_add_ps(v_av26, _mm_mul_ps(v_K1, v_cv22));
        __m128 v_av28 = _mm_sub_ps(v_av26, _mm_mul_ps(v_K1, v_cv22));

        __m128 v_av29 = _mm_add_ps(v_av27, _mm_mul_ps(v_K3, v_av19));
        __m128 v_av30 = _mm_add_ps(v_av24, _mm_mul_ps(v_K3, v_av20));
        __m128 v_av31 = _mm_add_ps(v_av28, _mm_mul_ps(v_K3, v_av22));
        __m128 v_av32 = _mm_add_ps(v_av25, _mm_mul_ps(v_K3, v_av21));
        __m128 v_av33 = _mm_sub_ps(v_av27, _mm_mul_ps(v_K3, v_av19));
        __m128 v_av34 = _mm_sub_ps(v_av24, _mm_mul_ps(v_K3, v_av20));
        __m128 v_av35 = _mm_sub_ps(v_av28, _mm_mul_ps(v_K3, v_av22));
        __m128 v_av36 = _mm_sub_ps(v_av25, _mm_mul_ps(v_K3, v_av21));

        /* Output point 2 : X[1], Output point 20 : X[19] */
        /* Output point 14 : X[13], Output point 8 : X[7] */
        /* Output point 10 : X[9], Output point 12 : X[11] */
        /* Output point 18 : X[17], Output point 4 : X[3] */
        if (flag == 0)
        {
            v_out1 = PS_N90_128(v_av30, v_av29);
            v_out19 = PS_P90_128(v_av30, v_av29);
            v_out13 = PS_N90_128(v_av32, v_av31);
            v_out7 = PS_P90_128(v_av32, v_av31);
            v_out9 = PS_N90_128(v_av34, v_av33);
            v_out11 = PS_P90_128(v_av34, v_av33);
            v_out17 = PS_N90_128(v_av36, v_av35);
            v_out3 = PS_P90_128(v_av36, v_av35);
        }
        else
        {
            v_out1 = PS_P90_128(v_av30, v_av29);
            v_out19 = PS_N90_128(v_av30, v_av29);
            v_out13 = PS_P90_128(v_av32, v_av31);
            v_out7 = PS_N90_128(v_av32, v_av31);
            v_out9 = PS_P90_128(v_av34, v_av33);
            v_out11 = PS_N90_128(v_av34, v_av33);
            v_out17 = PS_P90_128(v_av36, v_av35);
            v_out3 = PS_N90_128(v_av36, v_av35);
        }

        ST_LOW_128_S(curr_out, v_out0);
        ST_LOW_128_S(out_r + out_strides[1], v_out1);
        ST_LOW_128_S(out_r + out_strides[2], v_out2);
        ST_LOW_128_S(out_r + out_strides[3], v_out3);
        ST_LOW_128_S(out_r + out_strides[4], v_out4);
        ST_LOW_128_S(out_r + out_strides[5], v_out5);
        ST_LOW_128_S(out_r + out_strides[6], v_out6);
        ST_LOW_128_S(out_r + out_strides[7], v_out7);
        ST_LOW_128_S(out_r + out_strides[8], v_out8);
        ST_LOW_128_S(out_r + out_strides[9], v_out9);
        ST_LOW_128_S(out_r + out_strides[10], v_out10);
        ST_LOW_128_S(out_r + out_strides[11], v_out11);
        ST_LOW_128_S(out_r + out_strides[12], v_out12);
        ST_LOW_128_S(out_r + out_strides[13], v_out13);
        ST_LOW_128_S(out_r + out_strides[14], v_out14);
        ST_LOW_128_S(out_r + out_strides[15], v_out15);
        ST_LOW_128_S(out_r + out_strides[16], v_out16);
        ST_LOW_128_S(out_r + out_strides[17], v_out17);
        ST_LOW_128_S(out_r + out_strides[18], v_out18);
        ST_LOW_128_S(out_r + out_strides[19], v_out19);
    }

    #undef PS_P90_128
    #undef PS_N90_128

    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID fft20avx128fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                            VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                            VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");

    DOUBLE *in_r = (DOUBLE *)in_real;
    DOUBLE *out_r = (DOUBLE *)out_real;
    DOUBLE *curr_in, *curr_out;

#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;

    INTP N = n / NUM_SETS_128_D;
    INTP count;

    // Radix-20 twiddle constants
    __m128d v_C1 =
        _mm_set1_pd(0.559016994374947424102293417182819058860154590);
    __m128d v_C2 =
        _mm_set1_pd(0.618033988749894848204586834365638117720309180);
    __m128d v_C3 =
        _mm_set1_pd(0.951056516295153572116439333379382143405698634);
    __m128d v_C4 =
        _mm_set1_pd(0.250000000000000000000000000000000000000000000);
    __m128d v_neg_zero = _mm_set1_pd(-0.0);

    /* Complex rotation macros for forward and backward transforms */
    #define PS_P90_128(b, c) _mm_addsub_pd(c, SWAP_RI_128_D(b))
    #define PS_N90_128(b, c) \
        _mm_addsub_pd(c, _mm_xor_pd(SWAP_RI_128_D(b), v_neg_zero))

    for(count = 0; count < N; count++)
    {
        __m128d v_K1 = v_C1;
        __m128d v_K2 = v_C2;
        __m128d v_K3 = v_C3;
        __m128d v_K4 = v_C4;

        __m128d v_in0, v_in1, v_in2, v_in3, v_in4;
        __m128d v_in5, v_in6, v_in7, v_in8, v_in9;
        __m128d v_in10, v_in11, v_in12, v_in13, v_in14;
        __m128d v_in15, v_in16, v_in17, v_in18, v_in19;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4;
        __m128d v_out5, v_out6, v_out7, v_out8, v_out9;
        __m128d v_out10, v_out11, v_out12, v_out13, v_out14;
        __m128d v_out15, v_out16, v_out17, v_out18, v_out19;

        curr_in = in_r;
        curr_out = out_r;

        LD_128_D(curr_in, v_in0);
        LD_128_D(in_r + in_strides[1], v_in1);
        LD_128_D(in_r + in_strides[2], v_in2);
        LD_128_D(in_r + in_strides[3], v_in3);
        LD_128_D(in_r + in_strides[4], v_in4);
        LD_128_D(in_r + in_strides[5], v_in5);
        LD_128_D(in_r + in_strides[6], v_in6);
        LD_128_D(in_r + in_strides[7], v_in7);
        LD_128_D(in_r + in_strides[8], v_in8);
        LD_128_D(in_r + in_strides[9], v_in9);
        LD_128_D(in_r + in_strides[10], v_in10);
        LD_128_D(in_r + in_strides[11], v_in11);
        LD_128_D(in_r + in_strides[12], v_in12);
        LD_128_D(in_r + in_strides[13], v_in13);
        LD_128_D(in_r + in_strides[14], v_in14);
        LD_128_D(in_r + in_strides[15], v_in15);
        LD_128_D(in_r + in_strides[16], v_in16);
        LD_128_D(in_r + in_strides[17], v_in17);
        LD_128_D(in_r + in_strides[18], v_in18);
        LD_128_D(in_r + in_strides[19], v_in19);

        __m128d v_s1 = _mm_add_pd(v_in0, v_in10);
        __m128d v_s2 = _mm_add_pd(v_in5, v_in15);
        __m128d v_d1 = _mm_sub_pd(v_in0, v_in10);
        __m128d v_cv1 = _mm_add_pd(v_s1, v_s2);
        __m128d v_d2 = _mm_sub_pd(v_in5, v_in15);
        __m128d v_cv2 = _mm_sub_pd(v_s1, v_s2);

        __m128d v_d3 = _mm_sub_pd(v_in4, v_in14);
        __m128d v_s3 = _mm_add_pd(v_in4, v_in14);
        __m128d v_d4 = _mm_sub_pd(v_in13, v_in3);
        __m128d v_s4 = _mm_add_pd(v_in13, v_in3);
        __m128d v_d5 = _mm_sub_pd(v_in17, v_in7);
        __m128d v_s5 = _mm_add_pd(v_in17, v_in7);
        __m128d v_d6 = _mm_sub_pd(v_in16, v_in6);
        __m128d v_s6 = _mm_add_pd(v_in16, v_in6);
        __m128d v_d7 = _mm_sub_pd(v_in8, v_in18);
        __m128d v_s7 = _mm_add_pd(v_in8, v_in18);
        __m128d v_d8 = _mm_sub_pd(v_in9, v_in19);
        __m128d v_s8 = _mm_add_pd(v_in9, v_in19);
        __m128d v_d9 = _mm_sub_pd(v_in1, v_in11);
        __m128d v_s9 = _mm_add_pd(v_in1, v_in11);
        __m128d v_d10 = _mm_sub_pd(v_in12, v_in2);
        __m128d v_s10 = _mm_add_pd(v_in12, v_in2);

        __m128d v_cv3 = _mm_sub_pd(v_d9, v_d8);
        __m128d v_cv4 = _mm_sub_pd(v_d3, v_d6);
        __m128d v_cv5 = _mm_sub_pd(v_d7, v_d10);
        __m128d v_cv6 = _mm_sub_pd(v_d5, v_d4);
        __m128d v_cv7 = _mm_sub_pd(v_s3, v_s8);
        __m128d v_cv8 = _mm_sub_pd(v_s6, v_s9);
        __m128d v_cv9 = _mm_add_pd(v_cv7, v_cv8);
        __m128d v_cv10 = _mm_add_pd(v_s7, v_s4);
        __m128d v_cv11 = _mm_add_pd(v_s10, v_s5);
        __m128d v_cv12 = _mm_add_pd(v_cv10, v_cv11);
        __m128d v_cv13 = _mm_add_pd(v_s3, v_s8);
        __m128d v_cv14 = _mm_add_pd(v_s6, v_s9);
        __m128d v_cv15 = _mm_add_pd(v_cv13, v_cv14);
        __m128d v_cv16 = _mm_sub_pd(v_s7, v_s4);
        __m128d v_cv17 = _mm_sub_pd(v_s10, v_s5);
        __m128d v_cv18 = _mm_add_pd(v_cv16, v_cv17);

        __m128d v_cv19 = _mm_add_pd(v_d3, v_d6);
        __m128d v_cv20 = _mm_add_pd(v_d7, v_d10);
        __m128d v_cv21 = _mm_add_pd(v_cv19, v_cv20);
        __m128d v_cv22 = _mm_sub_pd(v_cv19, v_cv20);
        __m128d v_cv23 = _mm_add_pd(v_d8, v_d9);
        __m128d v_cv24 = _mm_add_pd(v_d4, v_d5);
        __m128d v_cv25 = _mm_add_pd(v_cv23, v_cv24);
        __m128d v_cv26 = _mm_sub_pd(v_cv24, v_cv23);

        __m128d v_cv27 = _mm_add_pd(v_d1, v_cv21);
        __m128d v_cv28 = _mm_add_pd(v_d2, v_cv25);

        // Output point 6 : X[5]
        // Output point 16 : X[15]
        if (flag == 0)
        {
            v_out5 = PS_N90_128(v_cv28, v_cv27);
            v_out15 = PS_P90_128(v_cv28, v_cv27);
        }
        else
        {
            v_out5 = PS_P90_128(v_cv28, v_cv27);
            v_out15 = PS_N90_128(v_cv28, v_cv27);
        }

        __m128d v_av1 = _mm_sub_pd(v_cv15, v_cv12);
        __m128d v_av2 = _mm_add_pd(v_cv15, v_cv12);
        __m128d v_av3 = _mm_sub_pd(v_cv1, _mm_mul_pd(v_K4, v_av2));
        __m128d v_av4 = _mm_sub_pd(v_cv13, v_cv14);
        __m128d v_av5 = _mm_sub_pd(v_cv10, v_cv11);
        __m128d v_av6 = _mm_mul_pd(v_K3,
            _mm_add_pd(_mm_mul_pd(v_K2, v_av5), v_av4));
        __m128d v_av7 = _mm_mul_pd(v_K3,
            _mm_sub_pd(v_av5, _mm_mul_pd(v_K2, v_av4)));
        // Output point 1 : X[0]
        v_out0 = _mm_add_pd(v_cv1, v_av2);
        __m128d v_av8 = _mm_sub_pd(v_av3, _mm_mul_pd(v_K1, v_av1));
        __m128d v_av9 = _mm_add_pd(v_av3, _mm_mul_pd(v_K1, v_av1));

        // Output point 9 : X[8], Output point 13 : X[12]
        // Output point 5 : X[4], Output point 17 : X[16]
        if (flag == 0)
        {
            v_out8 = PS_N90_128(v_av7, v_av8);
            v_out12 = PS_P90_128(v_av7, v_av8);
            v_out4 = PS_P90_128(v_av6, v_av9);
            v_out16 = PS_N90_128(v_av6, v_av9);
        }
        else
        {
            v_out8 = PS_P90_128(v_av7, v_av8);
            v_out12 = PS_N90_128(v_av7, v_av8);
            v_out4 = PS_N90_128(v_av6, v_av9);
            v_out16 = PS_P90_128(v_av6, v_av9);
        }

        __m128d v_av10 = _mm_sub_pd(v_cv9, v_cv18);
        __m128d v_av11 = _mm_add_pd(v_cv9, v_cv18);
        __m128d v_av12 = _mm_sub_pd(v_cv2, _mm_mul_pd(v_K4, v_av11));
        __m128d v_av13 = _mm_sub_pd(v_cv16, v_cv17);
        __m128d v_av14 = _mm_sub_pd(v_cv7, v_cv8);
        __m128d v_av15 = _mm_mul_pd(v_K3,
            _mm_sub_pd(v_av13, _mm_mul_pd(v_K2, v_av14)));
        __m128d v_av16 = _mm_mul_pd(v_K3,
            _mm_add_pd(_mm_mul_pd(v_K2, v_av13), v_av14));
        // Output point 11 : X[10]
        v_out10 = _mm_add_pd(v_cv2, v_av11);
        __m128d v_av17 = _mm_add_pd(v_av12, _mm_mul_pd(v_K1, v_av10));
        __m128d v_av18 = _mm_sub_pd(v_av12, _mm_mul_pd(v_K1, v_av10));

        // Output point 7 : X[6], Output point 15 : X[14]
        // Output point 3 : X[2], Output point 19 : X[18]
        if (flag == 0)
        {
            v_out6 = PS_N90_128(v_av16, v_av17);
            v_out14 = PS_P90_128(v_av16, v_av17);
            v_out2 = PS_P90_128(v_av15, v_av18);
            v_out18 = PS_N90_128(v_av15, v_av18);
        }
        else
        {
            v_out6 = PS_P90_128(v_av16, v_av17);
            v_out14 = PS_N90_128(v_av16, v_av17);
            v_out2 = PS_N90_128(v_av15, v_av18);
            v_out18 = PS_P90_128(v_av15, v_av18);
        }

        __m128d v_av19 = _mm_add_pd(_mm_mul_pd(v_K2, v_cv6), v_cv3);
        __m128d v_av20 = _mm_add_pd(_mm_mul_pd(v_K2, v_cv5), v_cv4);
        __m128d v_av21 = _mm_sub_pd(v_cv5, _mm_mul_pd(v_K2, v_cv4));
        __m128d v_av22 = _mm_sub_pd(v_cv6, _mm_mul_pd(v_K2, v_cv3));
        __m128d v_av23 = _mm_sub_pd(v_d2, _mm_mul_pd(v_K4, v_cv25));
        __m128d v_av24 = _mm_sub_pd(v_av23, _mm_mul_pd(v_K1, v_cv26));
        __m128d v_av25 = _mm_add_pd(v_av23, _mm_mul_pd(v_K1, v_cv26));
        __m128d v_av26 = _mm_sub_pd(v_d1, _mm_mul_pd(v_K4, v_cv21));
        __m128d v_av27 = _mm_add_pd(v_av26, _mm_mul_pd(v_K1, v_cv22));
        __m128d v_av28 = _mm_sub_pd(v_av26, _mm_mul_pd(v_K1, v_cv22));

        __m128d v_av29 = _mm_add_pd(v_av27, _mm_mul_pd(v_K3, v_av19));
        __m128d v_av30 = _mm_add_pd(v_av24, _mm_mul_pd(v_K3, v_av20));
        __m128d v_av31 = _mm_add_pd(v_av28, _mm_mul_pd(v_K3, v_av22));
        __m128d v_av32 = _mm_add_pd(v_av25, _mm_mul_pd(v_K3, v_av21));
        __m128d v_av33 = _mm_sub_pd(v_av27, _mm_mul_pd(v_K3, v_av19));
        __m128d v_av34 = _mm_sub_pd(v_av24, _mm_mul_pd(v_K3, v_av20));
        __m128d v_av35 = _mm_sub_pd(v_av28, _mm_mul_pd(v_K3, v_av22));
        __m128d v_av36 = _mm_sub_pd(v_av25, _mm_mul_pd(v_K3, v_av21));

        // Output point 2 : X[1], Output point 20 : X[19]
        // Output point 14 : X[13], Output point 8 : X[7]
        // Output point 10 : X[9], Output point 12 : X[11]
        // Output point 18 : X[17], Output point 4 : X[3]
        if (flag == 0)
        {
            v_out1 = PS_N90_128(v_av30, v_av29);
            v_out19 = PS_P90_128(v_av30, v_av29);
            v_out13 = PS_N90_128(v_av32, v_av31);
            v_out7 = PS_P90_128(v_av32, v_av31);
            v_out9 = PS_N90_128(v_av34, v_av33);
            v_out11 = PS_P90_128(v_av34, v_av33);
            v_out17 = PS_N90_128(v_av36, v_av35);
            v_out3 = PS_P90_128(v_av36, v_av35);
        }
        else
        {
            v_out1 = PS_P90_128(v_av30, v_av29);
            v_out19 = PS_N90_128(v_av30, v_av29);
            v_out13 = PS_P90_128(v_av32, v_av31);
            v_out7 = PS_N90_128(v_av32, v_av31);
            v_out9 = PS_P90_128(v_av34, v_av33);
            v_out11 = PS_N90_128(v_av34, v_av33);
            v_out17 = PS_P90_128(v_av36, v_av35);
            v_out3 = PS_N90_128(v_av36, v_av35);
        }

        ST_128_D(curr_out, v_out0);
        ST_128_D(out_r + out_strides[1], v_out1);
        ST_128_D(out_r + out_strides[2], v_out2);
        ST_128_D(out_r + out_strides[3], v_out3);
        ST_128_D(out_r + out_strides[4], v_out4);
        ST_128_D(out_r + out_strides[5], v_out5);
        ST_128_D(out_r + out_strides[6], v_out6);
        ST_128_D(out_r + out_strides[7], v_out7);
        ST_128_D(out_r + out_strides[8], v_out8);
        ST_128_D(out_r + out_strides[9], v_out9);
        ST_128_D(out_r + out_strides[10], v_out10);
        ST_128_D(out_r + out_strides[11], v_out11);
        ST_128_D(out_r + out_strides[12], v_out12);
        ST_128_D(out_r + out_strides[13], v_out13);
        ST_128_D(out_r + out_strides[14], v_out14);
        ST_128_D(out_r + out_strides[15], v_out15);
        ST_128_D(out_r + out_strides[16], v_out16);
        ST_128_D(out_r + out_strides[17], v_out17);
        ST_128_D(out_r + out_strides[18], v_out18);
        ST_128_D(out_r + out_strides[19], v_out19);

        in_r += NUM_SETS_128_D * v_in_stride;
        out_r += NUM_SETS_128_D * v_out_stride;
    }

    #undef PS_P90_128
    #undef PS_N90_128

    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_fft20avx128(UINT8 precision, UINT8 direction)
{
    if (precision == DT_FLOAT)
    {
        return fft20avx128fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft20avx128fp64;
    }
    else
    {
        return NULL;
    }
}
