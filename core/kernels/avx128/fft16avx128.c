/**
 * Copyright (C) 2024, Advanced Micro Devices. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/** @file fft16avx128.c
 *
 *  @brief Radix-16 FFT kernel with avx128 operations using x86 SIMD intrinsics
 *
 *  This file contains the DIT radix-16 FFT implementations using avx128
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Varun Sanjay
 *
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_common.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 15, 72, 64, 7, 7},
                                                     {0, 15, 72, 32, 7, 7}};
ops_cycles_t get_ops_cnt_fft16avx128(INT32 precision)
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

static VOID fft16avx128fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                            VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                            UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const FLOAT CRTM_16[7] = {
        0.92387953251128675612818318939678828682241662586364,
        0.38268343236508977172845998403039886676134456248563,
        0.70710678118654752440084436210484903928483593768847,
        0.70710678118654752440084436210484903928483593768847,
        0.38268343236508977172845998403039886676134456248563,
        0.92387953251128675612818318939678828682241662586364,
        1.0};

    FLOAT *in_r = (FLOAT *)in_real;
    FLOAT *out_r = (FLOAT *)out_real;
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
    INTP count;
    FLOAT *curr_in, *curr_out;

    __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
    __m128 v_in6, v_in7, v_in8, v_in9, v_in10;
    __m128 v_in11, v_in12, v_in13, v_in14, v_in15;
    __m128 v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6;
    __m128 v_tv7, v_tv9, v_tv11, v_tv12;
    __m128 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6;
    __m128 v_av7, v_av8, v_av9, v_av10, v_av12, v_av14;
    __m128 v_av15, v_av16, v_av17, v_av18, v_av19, v_av21;
    __m128 v_av22, v_av23, v_av24, v_av25, v_av20, v_av26;
    __m128 v_av27, v_av28, v_av29, v_av30, v_av31;
    __m128 v_av32, v_av33, v_av34, v_av35, v_av36;
    __m128 v_av39, v_av40, v_av41, v_av42, v_av43;
    __m128 v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7;
    __m128 v_cv8, v_cv9, v_cv10, v_cv11, v_cv12, v_cv13;
    __m128 v_cv14, v_cv15, v_cv16;
    __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6;
    __m128 v_out7, v_out8, v_out9, v_out10, v_out11, v_out12;
    __m128 v_out13, v_out14, v_out15;

    __m128 v_K1 = _mm_broadcast_ss(&CRTM_16[0]);
    __m128 v_K2 = _mm_broadcast_ss(&CRTM_16[1]);
    __m128 v_K3 = _mm_broadcast_ss(&CRTM_16[2]);
    __m128 v_K4 = _mm_broadcast_ss(&CRTM_16[3]);
    __m128 v_K5 = _mm_broadcast_ss(&CRTM_16[4]);
    __m128 v_K6 = _mm_broadcast_ss(&CRTM_16[5]);
    __m128 v_K7 = _mm_broadcast_ss(&CRTM_16[6]);

    if (flag)
    {
        in_r = in_imag;
        out_r = out_imag;
        v_K2 = -v_K2;
        v_K4 = -v_K4;
        v_K6 = -v_K6;
        v_K7 = -v_K7;
    }

    for (count = 0; count < N; count++)
    {
        curr_in = in_r;
        curr_out = out_r;

        GATHER2_128_S(curr_in, v_in_stride, v_in0);
        curr_in = in_r + in_strides[1];
        GATHER2_128_S(curr_in, v_in_stride, v_in1);
        curr_in = in_r + in_strides[2];
        GATHER2_128_S(curr_in, v_in_stride, v_in2);
        curr_in = in_r + in_strides[3];
        GATHER2_128_S(curr_in, v_in_stride, v_in3);
        curr_in = in_r + in_strides[4];
        GATHER2_128_S(curr_in, v_in_stride, v_in4);
        curr_in = in_r + in_strides[5];
        GATHER2_128_S(curr_in, v_in_stride, v_in5);
        curr_in = in_r + in_strides[6];
        GATHER2_128_S(curr_in, v_in_stride, v_in6);
        curr_in = in_r + in_strides[7];
        GATHER2_128_S(curr_in, v_in_stride, v_in7);
        curr_in = in_r + in_strides[8];
        GATHER2_128_S(curr_in, v_in_stride, v_in8);
        curr_in = in_r + in_strides[9];
        GATHER2_128_S(curr_in, v_in_stride, v_in9);
        curr_in = in_r + in_strides[10];
        GATHER2_128_S(curr_in, v_in_stride, v_in10);
        curr_in = in_r + in_strides[11];
        GATHER2_128_S(curr_in, v_in_stride, v_in11);
        curr_in = in_r + in_strides[12];
        GATHER2_128_S(curr_in, v_in_stride, v_in12);
        curr_in = in_r + in_strides[13];
        GATHER2_128_S(curr_in, v_in_stride, v_in13);
        curr_in = in_r + in_strides[14];
        GATHER2_128_S(curr_in, v_in_stride, v_in14);
        curr_in = in_r + in_strides[15];
        GATHER2_128_S(curr_in, v_in_stride, v_in15);

        // common calculations
        v_cv1 = _mm_add_ps(v_in0, v_in8);
        v_cv2 = _mm_add_ps(v_in1, v_in15);
        v_cv3 = _mm_add_ps(v_in2, v_in14);
        v_cv4 = _mm_add_ps(v_in3, v_in13);
        v_cv5 = _mm_add_ps(v_in4, v_in12);
        v_cv6 = _mm_add_ps(v_in5, v_in11);
        v_cv7 = _mm_add_ps(v_in6, v_in10);
        v_cv8 = _mm_add_ps(v_in7, v_in9);

        v_cv9 = _mm_sub_ps(v_in0, v_in8);
        v_cv10 = _mm_sub_ps(v_in1, v_in15);
        v_cv11 = _mm_sub_ps(v_in2, v_in14);
        v_cv12 = _mm_sub_ps(v_in3, v_in13);
        v_cv13 = _mm_sub_ps(v_in4, v_in12);
        v_cv14 = _mm_sub_ps(v_in5, v_in11);
        v_cv15 = _mm_sub_ps(v_in6, v_in10);
        v_cv16 = _mm_sub_ps(v_in7, v_in9);

        v_av1 = _mm_sub_ps(v_cv8, v_cv2);
        v_tv1 = _mm_mul_ps(v_K1, v_av1);

        v_av2 = _mm_sub_ps(v_cv7, v_cv3);
        v_tv2 = _mm_mul_ps(v_K3, v_av2);

        v_av3 = _mm_sub_ps(v_cv6, v_cv4);
        v_tv3 = _mm_mul_ps(v_K5, v_av3);

        v_av4 = _mm_sub_ps(v_cv9, v_tv2);
        v_av5 = _mm_add_ps(v_tv3, v_tv1);
        v_av15 = _mm_sub_ps(v_av4, v_av5);

        v_av6 = _mm_add_ps(v_cv16, v_cv10);
        v_tv4 = _mm_mul_ps(v_K2, v_av6);

        v_av7 = _mm_add_ps(v_cv11, v_cv15);
        v_tv5 = _mm_mul_ps(v_K4, v_av7);

        v_av8 = _mm_add_ps(v_cv12, v_cv14);
        v_tv6 = _mm_mul_ps(v_K6, v_av8);

        v_tv7 = _mm_mul_ps(v_K7, v_cv13);

        v_av9 = _mm_add_ps(v_tv5, v_tv7);
        v_av41 = _mm_add_ps(v_tv4, v_tv6);
        v_av10 = _mm_add_ps(v_av41, v_av9);

        v_av10 = SWAP_RI_128_S(CONJ_128_S(v_av10));

        // Output point 2 : X[1]
        v_out1 = _mm_sub_ps(v_av15, v_av10);
        // Output point 16 : X[15]
        v_out15 = _mm_add_ps(v_av15, v_av10);

        v_av12 = _mm_add_ps(v_av4, v_av5);

        v_av14 = _mm_sub_ps(v_av41, v_av9);

        v_av14 = SWAP_RI_128_S(CONJ_128_S(v_av14));

        // Output point 8 : X[7]
        v_out7 = _mm_sub_ps(v_av12, v_av14);
        // Output point 10 : X[9]
        v_out9 = _mm_add_ps(v_av12, v_av14);

        v_av16 = _mm_sub_ps(v_cv1, v_cv5);
        v_av17 = _mm_add_ps(v_cv2, v_cv8);
        v_av18 = _mm_add_ps(v_cv4, v_cv6);
        v_av19 = _mm_sub_ps(v_av17, v_av18);
        v_av20 = _mm_mul_ps(v_K3, v_av19);
        v_av21 = _mm_add_ps(v_av16, v_av20);

        v_av22 = _mm_sub_ps(v_cv14, v_cv12);
        v_av23 = _mm_sub_ps(v_cv16, v_cv10);
        v_av24 = _mm_mul_ps(v_K4, _mm_add_ps(v_av22, v_av23));

        v_av25 = _mm_sub_ps(v_cv15, v_cv11);
        v_av26 = _mm_mul_ps(v_K7, v_av25);

        v_av27 = _mm_add_ps(v_av24, v_av26);

        v_av27 = SWAP_RI_128_S(CONJ_128_S(v_av27));

        // Output point 3 : X[2]
        v_out2 = _mm_add_ps(v_av21, v_av27);
        // Output point 15 : X[14]
        v_out14 = _mm_sub_ps(v_av21, v_av27);

        v_av21 = _mm_sub_ps(v_av16, v_av20);

        v_av27 = _mm_sub_ps(v_av24, v_av26);

        v_av27 = SWAP_RI_128_S(CONJ_128_S(v_av27));

        // Output point 7 : X[6]
        v_out6 = _mm_add_ps(v_av21, v_av27);
        // Output point 11 : X[10]
        v_out10 = _mm_sub_ps(v_av21, v_av27);

        v_av28 = _mm_sub_ps(_mm_mul_ps(v_K1, v_av3), _mm_mul_ps(v_K5, v_av1));

        v_av29 = _mm_add_ps(v_cv9, v_tv2);
        v_av30 = _mm_add_ps(v_av29, v_av28);

        v_tv9 = _mm_mul_ps(v_K2, v_av8);
        v_tv11 = _mm_mul_ps(v_K6, v_av6);

        v_av31 = _mm_sub_ps(v_tv11, v_tv9);
        v_av32 = _mm_sub_ps(v_tv5, v_tv7);
        v_av33 = _mm_add_ps(v_av31, v_av32);

        v_av33 = SWAP_RI_128_S(CONJ_128_S(v_av33));

        // Output point 4 : X[3]
        v_out3 = _mm_sub_ps(v_av30, v_av33);
        // Output point 14 : X[13]
        v_out13 = _mm_add_ps(v_av30, v_av33);

        v_av42 = _mm_sub_ps(v_av29, v_av28);
        v_av43 = _mm_sub_ps(v_av31, v_av32);
        v_av43 = SWAP_RI_128_S(CONJ_128_S(v_av43));

        // Output point 6 : X[5]
        v_out5 = _mm_sub_ps(v_av42, v_av43);
        // Output point 12 : X[11]
        v_out11 = _mm_add_ps(v_av42, v_av43);

        v_av34 = _mm_add_ps(v_cv1, v_cv5);
        v_av35 = _mm_add_ps(v_cv3, v_cv7);

        v_av36 = _mm_sub_ps(v_av34, v_av35);

        v_tv12 = _mm_mul_ps(v_K7, _mm_sub_ps(v_av22, v_av23));

        v_tv12 = SWAP_RI_128_S(CONJ_128_S(v_tv12));

        // Output point 5 : X[4]
        v_out4 = _mm_sub_ps(v_av36, v_tv12);
        // Output point 13 : X[12]
        v_out12 = _mm_add_ps(v_av36, v_tv12);

        v_av39 = _mm_add_ps(v_av34, v_av35);

        v_av40 = _mm_add_ps(v_av17, v_av18);

        // Output point 1 : X[0]
        v_out0 = _mm_add_ps(v_av39, v_av40);
        // Output point 9 : X[8]
        v_out8 = _mm_sub_ps(v_av39, v_av40);

        SCATTER2_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out_r + out_strides[1];
        SCATTER2_128_S(curr_out, v_out_stride, v_out1);
        curr_out = out_r + out_strides[2];
        SCATTER2_128_S(curr_out, v_out_stride, v_out2);
        curr_out = out_r + out_strides[3];
        SCATTER2_128_S(curr_out, v_out_stride, v_out3);
        curr_out = out_r + out_strides[4];
        SCATTER2_128_S(curr_out, v_out_stride, v_out4);
        curr_out = out_r + out_strides[5];
        SCATTER2_128_S(curr_out, v_out_stride, v_out5);
        curr_out = out_r + out_strides[6];
        SCATTER2_128_S(curr_out, v_out_stride, v_out6);
        curr_out = out_r + out_strides[7];
        SCATTER2_128_S(curr_out, v_out_stride, v_out7);
        curr_out = out_r + out_strides[8];
        SCATTER2_128_S(curr_out, v_out_stride, v_out8);
        curr_out = out_r + out_strides[9];
        SCATTER2_128_S(curr_out, v_out_stride, v_out9);
        curr_out = out_r + out_strides[10];
        SCATTER2_128_S(curr_out, v_out_stride, v_out10);
        curr_out = out_r + out_strides[11];
        SCATTER2_128_S(curr_out, v_out_stride, v_out11);
        curr_out = out_r + out_strides[12];
        SCATTER2_128_S(curr_out, v_out_stride, v_out12);
        curr_out = out_r + out_strides[13];
        SCATTER2_128_S(curr_out, v_out_stride, v_out13);
        curr_out = out_r + out_strides[14];
        SCATTER2_128_S(curr_out, v_out_stride, v_out14);
        curr_out = out_r + out_strides[15];
        SCATTER2_128_S(curr_out, v_out_stride, v_out15);

        in_r += NUM_SETS_128_S * v_in_stride;
        out_r += NUM_SETS_128_S * v_out_stride;
    }
    // tail case
    if (n & 1)
    {
        curr_in = in_r;
        curr_out = out_r;

        LD_LOW_128_S(curr_in, v_in0);
        curr_in = in_r + in_strides[1];
        LD_LOW_128_S(curr_in, v_in1);
        curr_in = in_r + in_strides[2];
        LD_LOW_128_S(curr_in, v_in2);
        curr_in = in_r + in_strides[3];
        LD_LOW_128_S(curr_in, v_in3);
        curr_in = in_r + in_strides[4];
        LD_LOW_128_S(curr_in, v_in4);
        curr_in = in_r + in_strides[5];
        LD_LOW_128_S(curr_in, v_in5);
        curr_in = in_r + in_strides[6];
        LD_LOW_128_S(curr_in, v_in6);
        curr_in = in_r + in_strides[7];
        LD_LOW_128_S(curr_in, v_in7);
        curr_in = in_r + in_strides[8];
        LD_LOW_128_S(curr_in, v_in8);
        curr_in = in_r + in_strides[9];
        LD_LOW_128_S(curr_in, v_in9);
        curr_in = in_r + in_strides[10];
        LD_LOW_128_S(curr_in, v_in10);
        curr_in = in_r + in_strides[11];
        LD_LOW_128_S(curr_in, v_in11);
        curr_in = in_r + in_strides[12];
        LD_LOW_128_S(curr_in, v_in12);
        curr_in = in_r + in_strides[13];
        LD_LOW_128_S(curr_in, v_in13);
        curr_in = in_r + in_strides[14];
        LD_LOW_128_S(curr_in, v_in14);
        curr_in = in_r + in_strides[15];
        LD_LOW_128_S(curr_in, v_in15);

        // common calculations
        v_cv1 = _mm_add_ps(v_in0, v_in8);
        v_cv2 = _mm_add_ps(v_in1, v_in15);
        v_cv3 = _mm_add_ps(v_in2, v_in14);
        v_cv4 = _mm_add_ps(v_in3, v_in13);
        v_cv5 = _mm_add_ps(v_in4, v_in12);
        v_cv6 = _mm_add_ps(v_in5, v_in11);
        v_cv7 = _mm_add_ps(v_in6, v_in10);
        v_cv8 = _mm_add_ps(v_in7, v_in9);

        v_cv9 = _mm_sub_ps(v_in0, v_in8);
        v_cv10 = _mm_sub_ps(v_in1, v_in15);
        v_cv11 = _mm_sub_ps(v_in2, v_in14);
        v_cv12 = _mm_sub_ps(v_in3, v_in13);
        v_cv13 = _mm_sub_ps(v_in4, v_in12);
        v_cv14 = _mm_sub_ps(v_in5, v_in11);
        v_cv15 = _mm_sub_ps(v_in6, v_in10);
        v_cv16 = _mm_sub_ps(v_in7, v_in9);

        v_av1 = _mm_sub_ps(v_cv8, v_cv2);
        v_tv1 = _mm_mul_ps(v_K1, v_av1);

        v_av2 = _mm_sub_ps(v_cv7, v_cv3);
        v_tv2 = _mm_mul_ps(v_K3, v_av2);

        v_av3 = _mm_sub_ps(v_cv6, v_cv4);
        v_tv3 = _mm_mul_ps(v_K5, v_av3);

        v_av4 = _mm_sub_ps(v_cv9, v_tv2);
        v_av5 = _mm_add_ps(v_tv3, v_tv1);
        v_av15 = _mm_sub_ps(v_av4, v_av5);

        v_av6 = _mm_add_ps(v_cv16, v_cv10);
        v_tv4 = _mm_mul_ps(v_K2, v_av6);

        v_av7 = _mm_add_ps(v_cv11, v_cv15);
        v_tv5 = _mm_mul_ps(v_K4, v_av7);

        v_av8 = _mm_add_ps(v_cv12, v_cv14);
        v_tv6 = _mm_mul_ps(v_K6, v_av8);

        v_tv7 = _mm_mul_ps(v_K7, v_cv13);

        v_av9 = _mm_add_ps(v_tv5, v_tv7);
        v_av41 = _mm_add_ps(v_tv4, v_tv6);
        v_av10 = _mm_add_ps(v_av41, v_av9);

        v_av10 = SWAP_RI_128_S(CONJ_128_S(v_av10));

        // Output point 2 : X[1]
        v_out1 = _mm_sub_ps(v_av15, v_av10);
        // Output point 16 : X[15]
        v_out15 = _mm_add_ps(v_av15, v_av10);

        v_av12 = _mm_add_ps(v_av4, v_av5);

        v_av14 = _mm_sub_ps(v_av41, v_av9);

        v_av14 = SWAP_RI_128_S(CONJ_128_S(v_av14));

        // Output point 8 : X[7]
        v_out7 = _mm_sub_ps(v_av12, v_av14);
        // Output point 10 : X[9]
        v_out9 = _mm_add_ps(v_av12, v_av14);

        v_av16 = _mm_sub_ps(v_cv1, v_cv5);
        v_av17 = _mm_add_ps(v_cv2, v_cv8);
        v_av18 = _mm_add_ps(v_cv4, v_cv6);
        v_av19 = _mm_sub_ps(v_av17, v_av18);
        v_av20 = _mm_mul_ps(v_K3, v_av19);
        v_av21 = _mm_add_ps(v_av16, v_av20);

        v_av22 = _mm_sub_ps(v_cv14, v_cv12);
        v_av23 = _mm_sub_ps(v_cv16, v_cv10);
        v_av24 = _mm_mul_ps(v_K4, _mm_add_ps(v_av22, v_av23));

        v_av25 = _mm_sub_ps(v_cv15, v_cv11);
        v_av26 = _mm_mul_ps(v_K7, v_av25);

        v_av27 = _mm_add_ps(v_av24, v_av26);

        v_av27 = SWAP_RI_128_S(CONJ_128_S(v_av27));

        // Output point 3 : X[2]
        v_out2 = _mm_add_ps(v_av21, v_av27);
        // Output point 15 : X[14]
        v_out14 = _mm_sub_ps(v_av21, v_av27);

        v_av21 = _mm_sub_ps(v_av16, v_av20);

        v_av27 = _mm_sub_ps(v_av24, v_av26);

        v_av27 = SWAP_RI_128_S(CONJ_128_S(v_av27));

        // Output point 7 : X[6]
        v_out6 = _mm_add_ps(v_av21, v_av27);
        // Output point 11 : X[10]
        v_out10 = _mm_sub_ps(v_av21, v_av27);

        v_av28 = _mm_sub_ps(_mm_mul_ps(v_K1, v_av3), _mm_mul_ps(v_K5, v_av1));

        v_av29 = _mm_add_ps(v_cv9, v_tv2);
        v_av30 = _mm_add_ps(v_av29, v_av28);

        v_tv9 = _mm_mul_ps(v_K2, v_av8);
        v_tv11 = _mm_mul_ps(v_K6, v_av6);

        v_av31 = _mm_sub_ps(v_tv11, v_tv9);
        v_av32 = _mm_sub_ps(v_tv5, v_tv7);
        v_av33 = _mm_add_ps(v_av31, v_av32);

        v_av33 = SWAP_RI_128_S(CONJ_128_S(v_av33));

        // Output point 4 : X[3]
        v_out3 = _mm_sub_ps(v_av30, v_av33);
        // Output point 14 : X[13]
        v_out13 = _mm_add_ps(v_av30, v_av33);

        v_av42 = _mm_sub_ps(v_av29, v_av28);
        v_av43 = _mm_sub_ps(v_av31, v_av32);
        v_av43 = SWAP_RI_128_S(CONJ_128_S(v_av43));

        // Output point 6 : X[5]
        v_out5 = _mm_sub_ps(v_av42, v_av43);
        // Output point 12 : X[11]
        v_out11 = _mm_add_ps(v_av42, v_av43);

        v_av34 = _mm_add_ps(v_cv1, v_cv5);
        v_av35 = _mm_add_ps(v_cv3, v_cv7);

        v_av36 = _mm_sub_ps(v_av34, v_av35);

        v_tv12 = _mm_mul_ps(v_K7, _mm_sub_ps(v_av22, v_av23));

        v_tv12 = SWAP_RI_128_S(CONJ_128_S(v_tv12));

        // Output point 5 : X[4]
        v_out4 = _mm_sub_ps(v_av36, v_tv12);
        // Output point 13 : X[12]
        v_out12 = _mm_add_ps(v_av36, v_tv12);

        v_av39 = _mm_add_ps(v_av34, v_av35);

        v_av40 = _mm_add_ps(v_av17, v_av18);

        // Output point 1 : X[0]
        v_out0 = _mm_add_ps(v_av39, v_av40);
        // Output point 9 : X[8]
        v_out8 = _mm_sub_ps(v_av39, v_av40);

        ST_LOW_128_S(curr_out, v_out0);
        curr_out = out_r + out_strides[1];
        ST_LOW_128_S(curr_out, v_out1);
        curr_out = out_r + out_strides[2];
        ST_LOW_128_S(curr_out, v_out2);
        curr_out = out_r + out_strides[3];
        ST_LOW_128_S(curr_out, v_out3);
        curr_out = out_r + out_strides[4];
        ST_LOW_128_S(curr_out, v_out4);
        curr_out = out_r + out_strides[5];
        ST_LOW_128_S(curr_out, v_out5);
        curr_out = out_r + out_strides[6];
        ST_LOW_128_S(curr_out, v_out6);
        curr_out = out_r + out_strides[7];
        ST_LOW_128_S(curr_out, v_out7);
        curr_out = out_r + out_strides[8];
        ST_LOW_128_S(curr_out, v_out8);
        curr_out = out_r + out_strides[9];
        ST_LOW_128_S(curr_out, v_out9);
        curr_out = out_r + out_strides[10];
        ST_LOW_128_S(curr_out, v_out10);
        curr_out = out_r + out_strides[11];
        ST_LOW_128_S(curr_out, v_out11);
        curr_out = out_r + out_strides[12];
        ST_LOW_128_S(curr_out, v_out12);
        curr_out = out_r + out_strides[13];
        ST_LOW_128_S(curr_out, v_out13);
        curr_out = out_r + out_strides[14];
        ST_LOW_128_S(curr_out, v_out14);
        curr_out = out_r + out_strides[15];
        ST_LOW_128_S(curr_out, v_out15);
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID fft16avx128fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                            VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                            UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const DOUBLE CRTM_16[7] = {
        0.92387953251128675612818318939678828682241662586364,
        0.38268343236508977172845998403039886676134456248563,
        0.70710678118654752440084436210484903928483593768847,
        0.70710678118654752440084436210484903928483593768847,
        0.38268343236508977172845998403039886676134456248563,
        0.92387953251128675612818318939678828682241662586364,
        1.0};

    DOUBLE *in_r = (DOUBLE *)in_real;
    DOUBLE *out_r = (DOUBLE *)out_real;
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
    DOUBLE *curr_in, *curr_out;

    __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
    __m128d v_in6, v_in7, v_in8, v_in9, v_in10;
    __m128d v_in11, v_in12, v_in13, v_in14, v_in15;
    __m128d v_tv1, v_tv2, v_tv3, v_tv4, v_tv5, v_tv6;
    __m128d v_tv7, v_tv9, v_tv11, v_tv12;
    __m128d v_av1, v_av2, v_av3, v_av4, v_av5, v_av6;
    __m128d v_av7, v_av8, v_av9, v_av10, v_av12, v_av14;
    __m128d v_av15, v_av16, v_av17, v_av18, v_av19, v_av21;
    __m128d v_av22, v_av23, v_av24, v_av25, v_av20, v_av26;
    __m128d v_av27, v_av28, v_av29, v_av30, v_av31;
    __m128d v_av32, v_av33, v_av34, v_av35, v_av36;
    __m128d v_av39, v_av40, v_av41, v_av42, v_av43;
    __m128d v_cv1, v_cv2, v_cv3, v_cv4, v_cv5, v_cv6, v_cv7;
    __m128d v_cv8, v_cv9, v_cv10, v_cv11, v_cv12, v_cv13;
    __m128d v_cv14, v_cv15, v_cv16;
    __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6;
    __m128d v_out7, v_out8, v_out9, v_out10, v_out11, v_out12;
    __m128d v_out13, v_out14, v_out15;

    __m128d v_K1 = _mm_set1_pd(CRTM_16[0]);
    __m128d v_K2 = _mm_set1_pd(CRTM_16[1]);
    __m128d v_K3 = _mm_set1_pd(CRTM_16[2]);
    __m128d v_K4 = _mm_set1_pd(CRTM_16[3]);
    __m128d v_K5 = _mm_set1_pd(CRTM_16[4]);
    __m128d v_K6 = _mm_set1_pd(CRTM_16[5]);
    __m128d v_K7 = _mm_set1_pd(CRTM_16[6]);

    if (flag)
    {
        in_r = in_imag;
        out_r = out_imag;
        v_K2 = -v_K2;
        v_K4 = -v_K4;
        v_K6 = -v_K6;
        v_K7 = -v_K7;
    }

    for (count = 0; count < N; count++)
    {
        curr_in = in_r;
        curr_out = out_r;

        LD_128_D(curr_in, v_in0);
        curr_in = in_r + in_strides[1];
        LD_128_D(curr_in, v_in1);
        curr_in = in_r + in_strides[2];
        LD_128_D(curr_in, v_in2);
        curr_in = in_r + in_strides[3];
        LD_128_D(curr_in, v_in3);
        curr_in = in_r + in_strides[4];
        LD_128_D(curr_in, v_in4);
        curr_in = in_r + in_strides[5];
        LD_128_D(curr_in, v_in5);
        curr_in = in_r + in_strides[6];
        LD_128_D(curr_in, v_in6);
        curr_in = in_r + in_strides[7];
        LD_128_D(curr_in, v_in7);
        curr_in = in_r + in_strides[8];
        LD_128_D(curr_in, v_in8);
        curr_in = in_r + in_strides[9];
        LD_128_D(curr_in, v_in9);
        curr_in = in_r + in_strides[10];
        LD_128_D(curr_in, v_in10);
        curr_in = in_r + in_strides[11];
        LD_128_D(curr_in, v_in11);
        curr_in = in_r + in_strides[12];
        LD_128_D(curr_in, v_in12);
        curr_in = in_r + in_strides[13];
        LD_128_D(curr_in, v_in13);
        curr_in = in_r + in_strides[14];
        LD_128_D(curr_in, v_in14);
        curr_in = in_r + in_strides[15];
        LD_128_D(curr_in, v_in15);

        // common calculations
        v_cv1 = _mm_add_pd(v_in0, v_in8);
        v_cv2 = _mm_add_pd(v_in1, v_in15);
        v_cv3 = _mm_add_pd(v_in2, v_in14);
        v_cv4 = _mm_add_pd(v_in3, v_in13);
        v_cv5 = _mm_add_pd(v_in4, v_in12);
        v_cv6 = _mm_add_pd(v_in5, v_in11);
        v_cv7 = _mm_add_pd(v_in6, v_in10);
        v_cv8 = _mm_add_pd(v_in7, v_in9);

        v_cv9 = _mm_sub_pd(v_in0, v_in8);
        v_cv10 = _mm_sub_pd(v_in1, v_in15);
        v_cv11 = _mm_sub_pd(v_in2, v_in14);
        v_cv12 = _mm_sub_pd(v_in3, v_in13);
        v_cv13 = _mm_sub_pd(v_in4, v_in12);
        v_cv14 = _mm_sub_pd(v_in5, v_in11);
        v_cv15 = _mm_sub_pd(v_in6, v_in10);
        v_cv16 = _mm_sub_pd(v_in7, v_in9);

        v_av1 = _mm_sub_pd(v_cv8, v_cv2);
        v_tv1 = _mm_mul_pd(v_K1, v_av1);

        v_av2 = _mm_sub_pd(v_cv7, v_cv3);
        v_tv2 = _mm_mul_pd(v_K3, v_av2);

        v_av3 = _mm_sub_pd(v_cv6, v_cv4);
        v_tv3 = _mm_mul_pd(v_K5, v_av3);

        v_av4 = _mm_sub_pd(v_cv9, v_tv2);
        v_av5 = _mm_add_pd(v_tv3, v_tv1);
        v_av15 = _mm_sub_pd(v_av4, v_av5);

        v_av6 = _mm_add_pd(v_cv16, v_cv10);
        v_tv4 = _mm_mul_pd(v_K2, v_av6);

        v_av7 = _mm_add_pd(v_cv11, v_cv15);
        v_tv5 = _mm_mul_pd(v_K4, v_av7);

        v_av8 = _mm_add_pd(v_cv12, v_cv14);
        v_tv6 = _mm_mul_pd(v_K6, v_av8);

        v_tv7 = _mm_mul_pd(v_K7, v_cv13);

        v_av9 = _mm_add_pd(v_tv5, v_tv7);
        v_av41 = _mm_add_pd(v_tv4, v_tv6);
        v_av10 = _mm_add_pd(v_av41, v_av9);

        v_av10 = SWAP_RI_128_D(CONJ_128_D(v_av10));

        // Output point 2 : X[1]
        v_out1 = _mm_sub_pd(v_av15, v_av10);
        // Output point 16 : X[15]
        v_out15 = _mm_add_pd(v_av15, v_av10);

        v_av12 = _mm_add_pd(v_av4, v_av5);

        v_av14 = _mm_sub_pd(v_av41, v_av9);

        v_av14 = SWAP_RI_128_D(CONJ_128_D(v_av14));

        // Output point 8 : X[7]
        v_out7 = _mm_sub_pd(v_av12, v_av14);
        // Output point 10 : X[9]
        v_out9 = _mm_add_pd(v_av12, v_av14);

        v_av16 = _mm_sub_pd(v_cv1, v_cv5);
        v_av17 = _mm_add_pd(v_cv2, v_cv8);
        v_av18 = _mm_add_pd(v_cv4, v_cv6);
        v_av19 = _mm_sub_pd(v_av17, v_av18);
        v_av20 = _mm_mul_pd(v_K3, v_av19);
        v_av21 = _mm_add_pd(v_av16, v_av20);

        v_av22 = _mm_sub_pd(v_cv14, v_cv12);
        v_av23 = _mm_sub_pd(v_cv16, v_cv10);
        v_av24 = _mm_mul_pd(v_K4, _mm_add_pd(v_av22, v_av23));

        v_av25 = _mm_sub_pd(v_cv15, v_cv11);
        v_av26 = _mm_mul_pd(v_K7, v_av25);

        v_av27 = _mm_add_pd(v_av24, v_av26);

        v_av27 = SWAP_RI_128_D(CONJ_128_D(v_av27));

        // Output point 3 : X[2]
        v_out2 = _mm_add_pd(v_av21, v_av27);
        // Output point 15 : X[14]
        v_out14 = _mm_sub_pd(v_av21, v_av27);

        v_av21 = _mm_sub_pd(v_av16, v_av20);

        v_av27 = _mm_sub_pd(v_av24, v_av26);

        v_av27 = SWAP_RI_128_D(CONJ_128_D(v_av27));

        // Output point 7 : X[6]
        v_out6 = _mm_add_pd(v_av21, v_av27);
        // Output point 11 : X[10]
        v_out10 = _mm_sub_pd(v_av21, v_av27);

        //_mm_mul_pd(v_K1, v_av3)

        v_av28 = _mm_sub_pd(_mm_mul_pd(v_K1, v_av3), _mm_mul_pd(v_K5, v_av1));

        v_av29 = _mm_add_pd(v_cv9, v_tv2);
        v_av30 = _mm_add_pd(v_av29, v_av28);

        v_tv9 = _mm_mul_pd(v_K2, v_av8);
        v_tv11 = _mm_mul_pd(v_K6, v_av6);

        v_av31 = _mm_sub_pd(v_tv11, v_tv9);
        v_av32 = _mm_sub_pd(v_tv5, v_tv7);
        v_av33 = _mm_add_pd(v_av31, v_av32);

        v_av33 = SWAP_RI_128_D(CONJ_128_D(v_av33));

        // Output point 4 : X[3]
        v_out3 = _mm_sub_pd(v_av30, v_av33);
        // Output point 14 : X[13]
        v_out13 = _mm_add_pd(v_av30, v_av33);

        v_av42 = _mm_sub_pd(v_av29, v_av28);
        v_av43 = _mm_sub_pd(v_av31, v_av32);
        v_av43 = SWAP_RI_128_D(CONJ_128_D(v_av43));

        // Output point 6 : X[5]
        v_out5 = _mm_sub_pd(v_av42, v_av43);
        // Output point 12 : X[11]
        v_out11 = _mm_add_pd(v_av42, v_av43);

        v_av34 = _mm_add_pd(v_cv1, v_cv5);
        v_av35 = _mm_add_pd(v_cv3, v_cv7);

        v_av36 = _mm_sub_pd(v_av34, v_av35);

        v_tv12 = _mm_mul_pd(v_K7, _mm_sub_pd(v_av22, v_av23));

        v_tv12 = SWAP_RI_128_D(CONJ_128_D(v_tv12));

        // Output point 5 : X[4]
        v_out4 = _mm_sub_pd(v_av36, v_tv12);
        // Output point 13 : X[12]
        v_out12 = _mm_add_pd(v_av36, v_tv12);

        v_av39 = _mm_add_pd(v_av34, v_av35);

        v_av40 = _mm_add_pd(v_av17, v_av18);

        // Output point 1 : X[0]
        v_out0 = _mm_add_pd(v_av39, v_av40);
        // Output point 9 : X[8]
        v_out8 = _mm_sub_pd(v_av39, v_av40);

        ST_128_D(curr_out, v_out0);
        curr_out = out_r + out_strides[1];
        ST_128_D(curr_out, v_out1);
        curr_out = out_r + out_strides[2];
        ST_128_D(curr_out, v_out2);
        curr_out = out_r + out_strides[3];
        ST_128_D(curr_out, v_out3);
        curr_out = out_r + out_strides[4];
        ST_128_D(curr_out, v_out4);
        curr_out = out_r + out_strides[5];
        ST_128_D(curr_out, v_out5);
        curr_out = out_r + out_strides[6];
        ST_128_D(curr_out, v_out6);
        curr_out = out_r + out_strides[7];
        ST_128_D(curr_out, v_out7);
        curr_out = out_r + out_strides[8];
        ST_128_D(curr_out, v_out8);
        curr_out = out_r + out_strides[9];
        ST_128_D(curr_out, v_out9);
        curr_out = out_r + out_strides[10];
        ST_128_D(curr_out, v_out10);
        curr_out = out_r + out_strides[11];
        ST_128_D(curr_out, v_out11);
        curr_out = out_r + out_strides[12];
        ST_128_D(curr_out, v_out12);
        curr_out = out_r + out_strides[13];
        ST_128_D(curr_out, v_out13);
        curr_out = out_r + out_strides[14];
        ST_128_D(curr_out, v_out14);
        curr_out = out_r + out_strides[15];
        ST_128_D(curr_out, v_out15);

        in_r += NUM_SETS_128_D * v_in_stride;
        out_r += NUM_SETS_128_D * v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

kfft_ register_kernel_fft16avx128(INT32 precision)
{
    if (precision == DT_FLOAT)
    {
        return fft16avx128fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft16avx128fp64;
    }
    else
    {
        return NULL;
    }
}
