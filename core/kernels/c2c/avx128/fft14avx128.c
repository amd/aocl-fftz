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

/** @file fft14avx128.c
 *
 *  @brief Radix-14 FFT kernel with avx128 operations using x86 SIMD intrinsics
 *
 *  This file contains the DIT radix-14 FFT implementations using avx128
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Varun Sanjay
 *
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_common.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 36, 74, 56, 6, 6},
                                                     {0, 36, 74, 28, 6, 6}};

ops_cycles_t get_ops_cnt_fft14avx128(UINT8 precision, UINT8 direction)
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

static VOID fft14avx128fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                            VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                            UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const FLOAT CRTM_14[6] = {
        0.90096886790241912623610231950744505116591916200000,
        0.43388373911755809802961881825301518357930603231829,
        0.62348980185873356948108200474179836074227404291372,
        0.78183148246802977764200968763519351412805665195327,
        0.22252093395631447715505298010340457043006139348720,
        0.97492791218182360701813168299393121723278580100000};

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

    __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8;
    __m128 v_in9, v_in10, v_in11, v_in12, v_in13;
    __m128 v_av1, v_av2, v_av3, v_tv1, v_tv2, v_tv3, v_av4, v_av5, v_av6,
           v_av7, v_tv4;
    __m128 v_tv5, v_tv6, v_av8, v_av9, v_av10, v_av11, v_av12, v_tv7, v_tv8;
    __m128 v_tv9, v_av13, v_av14, v_av15, v_av16, v_tv10, v_tv11, v_tv12,
           v_av17, v_av18;
    __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7;
    __m128 v_out8, v_out9, v_out10, v_out11, v_out12, v_out13;

    __m128 v_K1 = _mm_broadcast_ss(&CRTM_14[0]);
    __m128 v_K2 = _mm_broadcast_ss(&CRTM_14[1]);
    __m128 v_K3 = _mm_broadcast_ss(&CRTM_14[2]);
    __m128 v_K4 = _mm_broadcast_ss(&CRTM_14[3]);
    __m128 v_K5 = _mm_broadcast_ss(&CRTM_14[4]);
    __m128 v_K6 = _mm_broadcast_ss(&CRTM_14[5]);

    if (flag)
    {
        in_r = in_imag;
        out_r = out_imag;
        v_K2 = -v_K2;
        v_K4 = -v_K4;
        v_K6 = -v_K6;
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

        v_av1 = _mm_add_ps(v_in2, v_in12);
        v_av2 = _mm_add_ps(v_in4, v_in10);
        v_av3 = _mm_add_ps(v_in6, v_in8);
        v_tv1 = _mm_mul_ps(v_K3, v_av1);
        v_tv2 = _mm_mul_ps(v_K5, v_av2);
        v_tv3 = _mm_mul_ps(v_K1, v_av3);
        v_av4 = _mm_sub_ps(v_tv1, v_tv2);
        v_av4 = _mm_sub_ps(v_av4, v_tv3);
        v_av4 = _mm_add_ps(v_av4, v_in0);

        v_av5 = _mm_add_ps(v_in1, v_in13);
        v_av6 = _mm_add_ps(v_in3, v_in11);
        v_av7 = _mm_add_ps(v_in5, v_in9);

        v_tv4 = _mm_mul_ps(v_K1, v_av5);
        v_tv5 = _mm_mul_ps(v_K5, v_av6);
        v_tv6 = _mm_mul_ps(v_K3, v_av7);
        v_av8 = _mm_add_ps(v_tv4, v_tv5);
        v_av8 = _mm_sub_ps(v_av8, v_tv6);
        v_av8 = _mm_sub_ps(v_av8, v_in7);
        v_av9 = _mm_add_ps(v_av4, v_av8);

        v_av10 = _mm_sub_ps(v_in13, v_in1);
        v_av11 = _mm_sub_ps(v_in11, v_in3);
        v_av12 = _mm_sub_ps(v_in9, v_in5);
        v_tv7 = _mm_mul_ps(v_K2, v_av10);
        v_tv8 = _mm_mul_ps(v_K6, v_av11);
        v_tv9 = _mm_mul_ps(v_K4, v_av12);

        v_av13 = _mm_add_ps(v_tv7, v_tv8);
        v_av13 = _mm_add_ps(v_av13, v_tv9);

        v_av14 = _mm_sub_ps(v_in12, v_in2);
        v_av15 = _mm_sub_ps(v_in4, v_in10);
        v_av16 = _mm_sub_ps(v_in8, v_in6);
        v_tv10 = _mm_mul_ps(v_K4, v_av14);
        v_tv11 = _mm_mul_ps(v_K6, v_av15);
        v_tv12 = _mm_mul_ps(v_K2, v_av16);

        v_av17 = _mm_sub_ps(v_tv10, v_tv11);
        v_av17 = _mm_add_ps(v_av17, v_tv12);
        v_av18 = _mm_add_ps(v_av13, v_av17);

        v_av18 = SWAP_RI_128_S(CONJ_128_S(v_av18));

        // Output point 2:X[1]
        v_out1 = _mm_add_ps(v_av9, v_av18);
        // Output point 14:X[13]
        v_out13 = _mm_sub_ps(v_av9, v_av18);

        v_av9 = _mm_sub_ps(v_av4, v_av8);
        v_av18 = _mm_sub_ps(v_av13, v_av17);
        v_av18 = SWAP_RI_128_S(CONJ_128_S(v_av18));

        // Output point 7:X[6]
        v_out6 = _mm_add_ps(v_av9, v_av18);
        // Output point 9:X[8]
        v_out8 = _mm_sub_ps(v_av9, v_av18);

        v_tv1 = _mm_mul_ps(v_K3, v_av3);
        v_tv2 = _mm_mul_ps(v_K5, v_av1);
        v_tv3 = _mm_mul_ps(v_K1, v_av2);
        v_av4 = _mm_sub_ps(v_tv1, v_tv2);
        v_av4 = _mm_sub_ps(v_av4, v_tv3);
        v_av4 = _mm_add_ps(v_av4, v_in0);

        v_tv4 = _mm_mul_ps(v_K3, v_av5);
        v_tv5 = _mm_mul_ps(v_K1, v_av6);
        v_tv6 = _mm_mul_ps(v_K5, v_av7);
        v_av8 = _mm_sub_ps(v_tv4, v_tv5);
        v_av8 = _mm_sub_ps(v_av8, v_tv6);
        v_av8 = _mm_add_ps(v_av8, v_in7);
        v_av9 = _mm_add_ps(v_av4, v_av8);

        v_tv7 = _mm_mul_ps(v_K4, v_av10);
        v_tv8 = _mm_mul_ps(v_K2, v_av11);
        v_tv9 = _mm_mul_ps(v_K6, v_av12);

        v_av13 = _mm_add_ps(v_tv7, v_tv8);
        v_av13 = _mm_sub_ps(v_av13, v_tv9);

        v_tv10 = _mm_mul_ps(v_K6, v_av14);
        v_tv11 = _mm_mul_ps(v_K2, v_av15);
        v_tv12 = _mm_mul_ps(v_K4, v_av16);

        v_av17 = _mm_add_ps(v_tv10, v_tv11);
        v_av17 = _mm_sub_ps(v_av17, v_tv12);
        v_av18 = _mm_add_ps(v_av13, v_av17);
        v_av18 = SWAP_RI_128_S(CONJ_128_S(v_av18));

        // Output point 3:X[2]
        v_out2 = _mm_add_ps(v_av9, v_av18);
        // Output point 13:X[12]
        v_out12 = _mm_sub_ps(v_av9, v_av18);

        v_av9 = _mm_sub_ps(v_av4, v_av8);
        v_av18 = _mm_sub_ps(v_av13, v_av17);
        v_av18 = SWAP_RI_128_S(CONJ_128_S(v_av18));

        // Output point 6:X[5]
        v_out5 = _mm_add_ps(v_av9, v_av18);
        // Output point 10:X[9]
        v_out9 = _mm_sub_ps(v_av9, v_av18);

        v_tv1 = _mm_mul_ps(v_K3, v_av2);
        v_tv2 = _mm_mul_ps(v_K1, v_av1);
        v_tv3 = _mm_mul_ps(v_K5, v_av3);
        v_av4 = _mm_sub_ps(v_tv1, v_tv2);
        v_av4 = _mm_sub_ps(v_av4, v_tv3);
        v_av4 = _mm_add_ps(v_av4, v_in0);

        v_tv4 = _mm_mul_ps(v_K5, v_av5);
        v_tv5 = _mm_mul_ps(v_K3, v_av6);
        v_tv6 = _mm_mul_ps(v_K1, v_av7);
        v_av8 = _mm_sub_ps(v_tv4, v_tv5);
        v_av8 = _mm_add_ps(v_av8, v_tv6);
        v_av8 = _mm_sub_ps(v_av8, v_in7);
        v_av9 = _mm_add_ps(v_av4, v_av8);

        v_tv7 = _mm_mul_ps(v_K6, v_av10);
        v_tv8 = _mm_mul_ps(v_K4, v_av11);
        v_tv9 = _mm_mul_ps(v_K2, v_av12);

        v_av13 = _mm_sub_ps(v_tv7, v_tv8);
        v_av13 = _mm_add_ps(v_av13, v_tv9);

        v_tv10 = _mm_mul_ps(v_K2, v_av14);
        v_tv11 = _mm_mul_ps(v_K4, v_av15);
        v_tv12 = _mm_mul_ps(v_K6, v_av16);

        v_av17 = _mm_add_ps(v_tv10, v_tv11);
        v_av17 = _mm_add_ps(v_av17, v_tv12);
        v_av18 = _mm_add_ps(v_av13, v_av17);
        v_av18 = SWAP_RI_128_S(CONJ_128_S(v_av18));

        // Output point 4:X[3]
        v_out3 = _mm_add_ps(v_av9, v_av18);
        // Output point 12:X[11]
        v_out11 = _mm_sub_ps(v_av9, v_av18);

        v_av9 = _mm_sub_ps(v_av4, v_av8);
        v_av18 = _mm_sub_ps(v_av13, v_av17);
        v_av18 = SWAP_RI_128_S(CONJ_128_S(v_av18));

        // Output point 5:X[4]
        v_out4 = _mm_add_ps(v_av9, v_av18);
        // Output point 11:X[10]
        v_out10 = _mm_sub_ps(v_av9, v_av18);

        v_av4 = _mm_add_ps(v_av5, v_av6);
        v_av4 = _mm_add_ps(v_av4, v_av7);
        v_av4 = _mm_add_ps(v_av4, v_in7);

        v_av5 = _mm_add_ps(v_in0, v_av1);
        v_av5 = _mm_add_ps(v_av5, v_av2);
        v_av5 = _mm_add_ps(v_av5, v_av3);

        // Output point 1:X[0]
        v_out0 = _mm_add_ps(v_av5, v_av4);
        // Output point 8:X[7]
        v_out7 = _mm_sub_ps(v_av5, v_av4);

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

        v_av1 = _mm_add_ps(v_in2, v_in12);
        v_av2 = _mm_add_ps(v_in4, v_in10);
        v_av3 = _mm_add_ps(v_in6, v_in8);
        v_tv1 = _mm_mul_ps(v_K3, v_av1);
        v_tv2 = _mm_mul_ps(v_K5, v_av2);
        v_tv3 = _mm_mul_ps(v_K1, v_av3);
        v_av4 = _mm_sub_ps(v_tv1, v_tv2);
        v_av4 = _mm_sub_ps(v_av4, v_tv3);
        v_av4 = _mm_add_ps(v_av4, v_in0);

        v_av5 = _mm_add_ps(v_in1, v_in13);
        v_av6 = _mm_add_ps(v_in3, v_in11);
        v_av7 = _mm_add_ps(v_in5, v_in9);

        v_tv4 = _mm_mul_ps(v_K1, v_av5);
        v_tv5 = _mm_mul_ps(v_K5, v_av6);
        v_tv6 = _mm_mul_ps(v_K3, v_av7);
        v_av8 = _mm_add_ps(v_tv4, v_tv5);
        v_av8 = _mm_sub_ps(v_av8, v_tv6);
        v_av8 = _mm_sub_ps(v_av8, v_in7);
        v_av9 = _mm_add_ps(v_av4, v_av8);

        v_av10 = _mm_sub_ps(v_in13, v_in1);
        v_av11 = _mm_sub_ps(v_in11, v_in3);
        v_av12 = _mm_sub_ps(v_in9, v_in5);
        v_tv7 = _mm_mul_ps(v_K2, v_av10);
        v_tv8 = _mm_mul_ps(v_K6, v_av11);
        v_tv9 = _mm_mul_ps(v_K4, v_av12);

        v_av13 = _mm_add_ps(v_tv7, v_tv8);
        v_av13 = _mm_add_ps(v_av13, v_tv9);

        v_av14 = _mm_sub_ps(v_in12, v_in2);
        v_av15 = _mm_sub_ps(v_in4, v_in10);
        v_av16 = _mm_sub_ps(v_in8, v_in6);
        v_tv10 = _mm_mul_ps(v_K4, v_av14);
        v_tv11 = _mm_mul_ps(v_K6, v_av15);
        v_tv12 = _mm_mul_ps(v_K2, v_av16);

        v_av17 = _mm_sub_ps(v_tv10, v_tv11);
        v_av17 = _mm_add_ps(v_av17, v_tv12);
        v_av18 = _mm_add_ps(v_av13, v_av17);

        v_av18 = SWAP_RI_128_S(CONJ_128_S(v_av18));

        // Output point 2:X[1]
        v_out1 = _mm_add_ps(v_av9, v_av18);
        // Output point 14:X[13]
        v_out13 = _mm_sub_ps(v_av9, v_av18);

        v_av9 = _mm_sub_ps(v_av4, v_av8);
        v_av18 = _mm_sub_ps(v_av13, v_av17);
        v_av18 = SWAP_RI_128_S(CONJ_128_S(v_av18));

        // Output point 7:X[6]
        v_out6 = _mm_add_ps(v_av9, v_av18);
        // Output point 9:X[8]
        v_out8 = _mm_sub_ps(v_av9, v_av18);

        v_tv1 = _mm_mul_ps(v_K3, v_av3);
        v_tv2 = _mm_mul_ps(v_K5, v_av1);
        v_tv3 = _mm_mul_ps(v_K1, v_av2);
        v_av4 = _mm_sub_ps(v_tv1, v_tv2);
        v_av4 = _mm_sub_ps(v_av4, v_tv3);
        v_av4 = _mm_add_ps(v_av4, v_in0);

        v_tv4 = _mm_mul_ps(v_K3, v_av5);
        v_tv5 = _mm_mul_ps(v_K1, v_av6);
        v_tv6 = _mm_mul_ps(v_K5, v_av7);
        v_av8 = _mm_sub_ps(v_tv4, v_tv5);
        v_av8 = _mm_sub_ps(v_av8, v_tv6);
        v_av8 = _mm_add_ps(v_av8, v_in7);
        v_av9 = _mm_add_ps(v_av4, v_av8);

        v_tv7 = _mm_mul_ps(v_K4, v_av10);
        v_tv8 = _mm_mul_ps(v_K2, v_av11);
        v_tv9 = _mm_mul_ps(v_K6, v_av12);

        v_av13 = _mm_add_ps(v_tv7, v_tv8);
        v_av13 = _mm_sub_ps(v_av13, v_tv9);

        v_tv10 = _mm_mul_ps(v_K6, v_av14);
        v_tv11 = _mm_mul_ps(v_K2, v_av15);
        v_tv12 = _mm_mul_ps(v_K4, v_av16);

        v_av17 = _mm_add_ps(v_tv10, v_tv11);
        v_av17 = _mm_sub_ps(v_av17, v_tv12);
        v_av18 = _mm_add_ps(v_av13, v_av17);
        v_av18 = SWAP_RI_128_S(CONJ_128_S(v_av18));

        // Output point 3:X[2]
        v_out2 = _mm_add_ps(v_av9, v_av18);
        // Output point 13:X[12]
        v_out12 = _mm_sub_ps(v_av9, v_av18);

        v_av9 = _mm_sub_ps(v_av4, v_av8);
        v_av18 = _mm_sub_ps(v_av13, v_av17);
        v_av18 = SWAP_RI_128_S(CONJ_128_S(v_av18));

        // Output point 6:X[5]
        v_out5 = _mm_add_ps(v_av9, v_av18);
        // Output point 10:X[9]
        v_out9 = _mm_sub_ps(v_av9, v_av18);

        v_tv1 = _mm_mul_ps(v_K3, v_av2);
        v_tv2 = _mm_mul_ps(v_K1, v_av1);
        v_tv3 = _mm_mul_ps(v_K5, v_av3);
        v_av4 = _mm_sub_ps(v_tv1, v_tv2);
        v_av4 = _mm_sub_ps(v_av4, v_tv3);
        v_av4 = _mm_add_ps(v_av4, v_in0);

        v_tv4 = _mm_mul_ps(v_K5, v_av5);
        v_tv5 = _mm_mul_ps(v_K3, v_av6);
        v_tv6 = _mm_mul_ps(v_K1, v_av7);
        v_av8 = _mm_sub_ps(v_tv4, v_tv5);
        v_av8 = _mm_add_ps(v_av8, v_tv6);
        v_av8 = _mm_sub_ps(v_av8, v_in7);
        v_av9 = _mm_add_ps(v_av4, v_av8);

        v_tv7 = _mm_mul_ps(v_K6, v_av10);
        v_tv8 = _mm_mul_ps(v_K4, v_av11);
        v_tv9 = _mm_mul_ps(v_K2, v_av12);

        v_av13 = _mm_sub_ps(v_tv7, v_tv8);
        v_av13 = _mm_add_ps(v_av13, v_tv9);

        v_tv10 = _mm_mul_ps(v_K2, v_av14);
        v_tv11 = _mm_mul_ps(v_K4, v_av15);
        v_tv12 = _mm_mul_ps(v_K6, v_av16);

        v_av17 = _mm_add_ps(v_tv10, v_tv11);
        v_av17 = _mm_add_ps(v_av17, v_tv12);
        v_av18 = _mm_add_ps(v_av13, v_av17);
        v_av18 = SWAP_RI_128_S(CONJ_128_S(v_av18));

        // Output point 4:X[3]
        v_out3 = _mm_add_ps(v_av9, v_av18);
        // Output point 12:X[11]
        v_out11 = _mm_sub_ps(v_av9, v_av18);

        v_av9 = _mm_sub_ps(v_av4, v_av8);
        v_av18 = _mm_sub_ps(v_av13, v_av17);
        v_av18 = SWAP_RI_128_S(CONJ_128_S(v_av18));

        // Output point 5:X[4]
        v_out4 = _mm_add_ps(v_av9, v_av18);
        // Output point 11:X[10]
        v_out10 = _mm_sub_ps(v_av9, v_av18);

        v_av4 = _mm_add_ps(v_av5, v_av6);
        v_av4 = _mm_add_ps(v_av4, v_av7);
        v_av4 = _mm_add_ps(v_av4, v_in7);

        v_av5 = _mm_add_ps(v_in0, v_av1);
        v_av5 = _mm_add_ps(v_av5, v_av2);
        v_av5 = _mm_add_ps(v_av5, v_av3);

        // Output point 1:X[0]
        v_out0 = _mm_add_ps(v_av5, v_av4);
        // Output point 8:X[7]
        v_out7 = _mm_sub_ps(v_av5, v_av4);

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
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID fft14avx128fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                            VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                            UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const DOUBLE CRTM_14[6] = {
        0.90096886790241912623610231950744505116591916200000,
        0.43388373911755809802961881825301518357930603231829,
        0.62348980185873356948108200474179836074227404291372,
        0.78183148246802977764200968763519351412805665195327,
        0.22252093395631447715505298010340457043006139348720,
        0.97492791218182360701813168299393121723278580100000};

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

    __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8;
    __m128d v_in9, v_in10, v_in11, v_in12, v_in13;
    __m128d v_av1, v_av2, v_av3, v_tv1, v_tv2, v_tv3, v_av4, v_av5, v_av6,
            v_av7, v_tv4;
    __m128d v_tv5, v_tv6, v_av8, v_av9, v_av10, v_av11, v_av12, v_tv7, v_tv8;
    __m128d v_tv9, v_av13, v_av14, v_av15, v_av16, v_tv10, v_tv11, v_tv12,
            v_av17, v_av18;
    __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7;
    __m128d v_out8, v_out9, v_out10, v_out11, v_out12, v_out13;

    __m128d v_K1 = _mm_set1_pd(CRTM_14[0]);
    __m128d v_K2 = _mm_set1_pd(CRTM_14[1]);
    __m128d v_K3 = _mm_set1_pd(CRTM_14[2]);
    __m128d v_K4 = _mm_set1_pd(CRTM_14[3]);
    __m128d v_K5 = _mm_set1_pd(CRTM_14[4]);
    __m128d v_K6 = _mm_set1_pd(CRTM_14[5]);

    if (flag)
    {
        in_r = in_imag;
        out_r = out_imag;
        v_K2 = -v_K2;
        v_K4 = -v_K4;
        v_K6 = -v_K6;
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

        v_av1 = _mm_add_pd(v_in2, v_in12);
        v_av2 = _mm_add_pd(v_in4, v_in10);
        v_av3 = _mm_add_pd(v_in6, v_in8);
        v_tv1 = _mm_mul_pd(v_K3, v_av1);
        v_tv2 = _mm_mul_pd(v_K5, v_av2);
        v_tv3 = _mm_mul_pd(v_K1, v_av3);
        v_av4 = _mm_sub_pd(v_tv1, v_tv2);
        v_av4 = _mm_sub_pd(v_av4, v_tv3);
        v_av4 = _mm_add_pd(v_av4, v_in0);

        v_av5 = _mm_add_pd(v_in1, v_in13);
        v_av6 = _mm_add_pd(v_in3, v_in11);
        v_av7 = _mm_add_pd(v_in5, v_in9);

        v_tv4 = _mm_mul_pd(v_K1, v_av5);
        v_tv5 = _mm_mul_pd(v_K5, v_av6);
        v_tv6 = _mm_mul_pd(v_K3, v_av7);
        v_av8 = _mm_add_pd(v_tv4, v_tv5);
        v_av8 = _mm_sub_pd(v_av8, v_tv6);
        v_av8 = _mm_sub_pd(v_av8, v_in7);
        v_av9 = _mm_add_pd(v_av4, v_av8);

        v_av10 = _mm_sub_pd(v_in13, v_in1);
        v_av11 = _mm_sub_pd(v_in11, v_in3);
        v_av12 = _mm_sub_pd(v_in9, v_in5);
        v_tv7 = _mm_mul_pd(v_K2, v_av10);
        v_tv8 = _mm_mul_pd(v_K6, v_av11);
        v_tv9 = _mm_mul_pd(v_K4, v_av12);

        v_av13 = _mm_add_pd(v_tv7, v_tv8);
        v_av13 = _mm_add_pd(v_av13, v_tv9);

        v_av14 = _mm_sub_pd(v_in12, v_in2);
        v_av15 = _mm_sub_pd(v_in4, v_in10);
        v_av16 = _mm_sub_pd(v_in8, v_in6);
        v_tv10 = _mm_mul_pd(v_K4, v_av14);
        v_tv11 = _mm_mul_pd(v_K6, v_av15);
        v_tv12 = _mm_mul_pd(v_K2, v_av16);

        v_av17 = _mm_sub_pd(v_tv10, v_tv11);
        v_av17 = _mm_add_pd(v_av17, v_tv12);
        v_av18 = _mm_add_pd(v_av13, v_av17);

        v_av18 = SWAP_RI_128_D(CONJ_128_D(v_av18));

        // Output point 2:X[1]
        v_out1 = _mm_add_pd(v_av9, v_av18);
        // Output point 14:X[13]
        v_out13 = _mm_sub_pd(v_av9, v_av18);

        v_av9 = _mm_sub_pd(v_av4, v_av8);
        v_av18 = _mm_sub_pd(v_av13, v_av17);
        v_av18 = SWAP_RI_128_D(CONJ_128_D(v_av18));

        // Output point 7:X[6]
        v_out6 = _mm_add_pd(v_av9, v_av18);
        // Output point 9:X[8]
        v_out8 = _mm_sub_pd(v_av9, v_av18);

        v_tv1 = _mm_mul_pd(v_K3, v_av3);
        v_tv2 = _mm_mul_pd(v_K5, v_av1);
        v_tv3 = _mm_mul_pd(v_K1, v_av2);
        v_av4 = _mm_sub_pd(v_tv1, v_tv2);
        v_av4 = _mm_sub_pd(v_av4, v_tv3);
        v_av4 = _mm_add_pd(v_av4, v_in0);

        v_tv4 = _mm_mul_pd(v_K3, v_av5);
        v_tv5 = _mm_mul_pd(v_K1, v_av6);
        v_tv6 = _mm_mul_pd(v_K5, v_av7);
        v_av8 = _mm_sub_pd(v_tv4, v_tv5);
        v_av8 = _mm_sub_pd(v_av8, v_tv6);
        v_av8 = _mm_add_pd(v_av8, v_in7);
        v_av9 = _mm_add_pd(v_av4, v_av8);

        v_tv7 = _mm_mul_pd(v_K4, v_av10);
        v_tv8 = _mm_mul_pd(v_K2, v_av11);
        v_tv9 = _mm_mul_pd(v_K6, v_av12);

        v_av13 = _mm_add_pd(v_tv7, v_tv8);
        v_av13 = _mm_sub_pd(v_av13, v_tv9);

        v_tv10 = _mm_mul_pd(v_K6, v_av14);
        v_tv11 = _mm_mul_pd(v_K2, v_av15);
        v_tv12 = _mm_mul_pd(v_K4, v_av16);

        v_av17 = _mm_add_pd(v_tv10, v_tv11);
        v_av17 = _mm_sub_pd(v_av17, v_tv12);
        v_av18 = _mm_add_pd(v_av13, v_av17);
        v_av18 = SWAP_RI_128_D(CONJ_128_D(v_av18));

        // Output point 3:X[2]
        v_out2 = _mm_add_pd(v_av9, v_av18);
        // Output point 13:X[12]
        v_out12 = _mm_sub_pd(v_av9, v_av18);

        v_av9 = _mm_sub_pd(v_av4, v_av8);
        v_av18 = _mm_sub_pd(v_av13, v_av17);
        v_av18 = SWAP_RI_128_D(CONJ_128_D(v_av18));

        // Output point 6:X[5]
        v_out5 = _mm_add_pd(v_av9, v_av18);
        // Output point 10:X[9]
        v_out9 = _mm_sub_pd(v_av9, v_av18);

        v_tv1 = _mm_mul_pd(v_K3, v_av2);
        v_tv2 = _mm_mul_pd(v_K1, v_av1);
        v_tv3 = _mm_mul_pd(v_K5, v_av3);
        v_av4 = _mm_sub_pd(v_tv1, v_tv2);
        v_av4 = _mm_sub_pd(v_av4, v_tv3);
        v_av4 = _mm_add_pd(v_av4, v_in0);

        v_tv4 = _mm_mul_pd(v_K5, v_av5);
        v_tv5 = _mm_mul_pd(v_K3, v_av6);
        v_tv6 = _mm_mul_pd(v_K1, v_av7);
        v_av8 = _mm_sub_pd(v_tv4, v_tv5);
        v_av8 = _mm_add_pd(v_av8, v_tv6);
        v_av8 = _mm_sub_pd(v_av8, v_in7);
        v_av9 = _mm_add_pd(v_av4, v_av8);

        v_tv7 = _mm_mul_pd(v_K6, v_av10);
        v_tv8 = _mm_mul_pd(v_K4, v_av11);
        v_tv9 = _mm_mul_pd(v_K2, v_av12);

        v_av13 = _mm_sub_pd(v_tv7, v_tv8);
        v_av13 = _mm_add_pd(v_av13, v_tv9);

        v_tv10 = _mm_mul_pd(v_K2, v_av14);
        v_tv11 = _mm_mul_pd(v_K4, v_av15);
        v_tv12 = _mm_mul_pd(v_K6, v_av16);

        v_av17 = _mm_add_pd(v_tv10, v_tv11);
        v_av17 = _mm_add_pd(v_av17, v_tv12);
        v_av18 = _mm_add_pd(v_av13, v_av17);
        v_av18 = SWAP_RI_128_D(CONJ_128_D(v_av18));

        // Output point 4:X[3]
        v_out3 = _mm_add_pd(v_av9, v_av18);
        // Output point 12:X[11]
        v_out11 = _mm_sub_pd(v_av9, v_av18);

        v_av9 = _mm_sub_pd(v_av4, v_av8);
        v_av18 = _mm_sub_pd(v_av13, v_av17);
        v_av18 = SWAP_RI_128_D(CONJ_128_D(v_av18));

        // Output point 5:X[4]
        v_out4 = _mm_add_pd(v_av9, v_av18);
        // Output point 11:X[10]
        v_out10 = _mm_sub_pd(v_av9, v_av18);

        v_av4 = _mm_add_pd(v_av5, v_av6);
        v_av4 = _mm_add_pd(v_av4, v_av7);
        v_av4 = _mm_add_pd(v_av4, v_in7);

        v_av5 = _mm_add_pd(v_in0, v_av1);
        v_av5 = _mm_add_pd(v_av5, v_av2);
        v_av5 = _mm_add_pd(v_av5, v_av3);

        // Output point 1:X[0]
        v_out0 = _mm_add_pd(v_av5, v_av4);
        // Output point 8:X[7]
        v_out7 = _mm_sub_pd(v_av5, v_av4);

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

        in_r += NUM_SETS_128_D * v_in_stride;
        out_r += NUM_SETS_128_D * v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

kfft_ register_kernel_fft14avx128(UINT8 precision, UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return fft14avx128fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft14avx128fp64;
    }
    else
    {
        return NULL;
    }
}
