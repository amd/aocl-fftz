/**
 * Copyright (C) 2025, Advanced Micro Devices. All rights reserved.
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

/** @file rfft12avx512.c
 *
 *  @brief Radix-12 r2hc Real-FFT kernel with AVX-512 operations using x86 SIMD
 *  intrinsics
 *
 *  This file contains the DIT radix-12 real-to-halfcomplex implementations
 *  using AVX512 SIMD operations for single-precision and double-precision
 *  inputs.
 *
 *  @author Jeya R
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/r2hc_simd_avx512.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                {{{0, 8 , 38, 304, 178, 72},
                                                  {0, 10, 38, 304, 178, 72}},
                                                 {{0, 8 , 38, 152, 10, 72},
                                                  {0, 10, 38, 152, 10, 72}}};

ops_cycles_t get_ops_cnt_r2hc_rfft12avx512(UINT8 precision, UINT8 direction)
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

static VOID r2hc_rfft12avx512_fp32_fwd(VOID *in_real, VOID *in_imag,
                                       VOID *out_real, VOID *out_imag, INTP n,
                                       aoclfftz_strides_t *strides, VOID *twd,
                                       UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_12_1 = 0.866025403784438646763723170752936183471402627f;
    const FLOAT CRTM_12_2 = 0.500000000000000000000000000000000000000000000f;

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
    INTP N = n >> 4;

    __m512 v_CRTM_12_1 = _mm512_set1_ps(CRTM_12_1);
    __m512 v_CRTM_12_2 = _mm512_set1_ps(CRTM_12_2);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m512 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9, v_in10, v_in11;
        __m512 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
               v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27;
        __m512 v_t0, v_t1, v_t2, v_t3, v_t4, v_t5;
        __m512 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_512_S(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_512_S(curr_in, v_in_stride, v_in1);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_512_S(curr_in, v_in_stride, v_in2);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_512_S(curr_in, v_in_stride, v_in3);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_512_S(curr_in, v_in_stride, v_in4);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_512_S(curr_in, v_in_stride, v_in5);
        // Input point 7: x(6)
        curr_in = in + in_strides[6];
        LDR_512_S(curr_in, v_in_stride, v_in6);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_512_S(curr_in, v_in_stride, v_in7);
        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDR_512_S(curr_in, v_in_stride, v_in8);
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDR_512_S(curr_in, v_in_stride, v_in9);
        // Input point 11: x(10)
        curr_in = in + in_strides[10];
        LDR_512_S(curr_in, v_in_stride, v_in10);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_512_S(curr_in, v_in_stride, v_in11);

        v_s0 = _mm512_add_ps(v_in11, v_in1);
        v_s1 = _mm512_sub_ps(v_in11, v_in1);
        v_s2 = _mm512_add_ps(v_in5, v_in7);
        v_s3 = _mm512_sub_ps(v_in5, v_in7);
        v_s4 = _mm512_add_ps(v_in0, v_in6);
        v_s5 = _mm512_sub_ps(v_in0, v_in6);
        v_s6 = _mm512_add_ps(v_in10, v_in2);
        v_s7 = _mm512_sub_ps(v_in10, v_in2);
        v_s8 = _mm512_add_ps(v_in4, v_in8);
        v_s9 = _mm512_sub_ps(v_in4, v_in8);
        v_s10 = _mm512_add_ps(v_in9, v_in3);
        v_s11 = _mm512_sub_ps(v_in9, v_in3);

        v_s12 = _mm512_add_ps(v_s0, v_s2);
        v_s13 = _mm512_sub_ps(v_s0, v_s2);
        v_s14 = _mm512_add_ps(v_s6, v_s8);
        v_s15 = _mm512_sub_ps(v_s6, v_s8);
        v_s16 = _mm512_add_ps(v_s1, v_s3);
        v_s17 = _mm512_sub_ps(v_s1, v_s3);
        v_s18 = _mm512_add_ps(v_s4, v_s10);
        v_s19 = _mm512_sub_ps(v_s4, v_s10);
        v_s20 = _mm512_add_ps(v_s12, v_s14);
        v_s21 = _mm512_sub_ps(v_s12, v_s14);

        v_t0 = _mm512_mul_ps(v_CRTM_12_2, v_s15);
        v_t1 = _mm512_mul_ps(v_CRTM_12_2, v_s17);
        v_s22 = _mm512_add_ps(v_s5, v_t0);
        v_s23 = _mm512_add_ps(v_t1, v_s11);
        v_s24 = _mm512_add_ps(v_s7, v_s9);
        v_s25 = _mm512_sub_ps(v_s7, v_s9);
        v_s26 = _mm512_add_ps(v_s16, v_s24);
        v_s27 = _mm512_sub_ps(v_s16, v_s24);

        v_t2 = _mm512_mul_ps(v_CRTM_12_1, v_s13);
        v_t3 = _mm512_mul_ps(v_CRTM_12_2, v_s20);
        v_t4 = _mm512_mul_ps(v_CRTM_12_2, v_s21);
        v_t5 = _mm512_mul_ps(v_CRTM_12_1, v_s25);

        v_out0 = _mm512_add_ps(v_s20, v_s18);
        v_out1 = _mm512_add_ps(v_s22, v_t2);
        v_out2 = _mm512_add_ps(v_s23, v_t5);
        v_out3 = _mm512_add_ps(v_t4, v_s19);
        v_out4 = _mm512_mul_ps(v_CRTM_12_1, v_s26);
        v_out5 = _mm512_sub_ps(v_s5, v_s15);
        v_out6 = _mm512_sub_ps(v_s17, v_s11);
        v_out7 = _mm512_sub_ps(v_s18, v_t3);
        v_out8 = _mm512_mul_ps(v_CRTM_12_1, v_s27);
        v_out9 = _mm512_sub_ps(v_s22, v_t2);
        v_out10 = _mm512_sub_ps(v_s23, v_t5);
        v_out11 = _mm512_sub_ps(v_s19, v_s21);

        // Output point 1: X(0)
        STR_512_S(curr_out, v_out_stride, v_out0);
        // Output point 2: x(1) & Output point 3: x(2)
        curr_out = out + out_strides[1];
        STRI_2x512_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 4: x(3) & Output point 5: x(4)
        curr_out = out + out_strides[3];
        STRI_2x512_S(curr_out, v_out_stride, v_out3, v_out4);
        // Output point 6: x(5) & Output point 7: x(6)
        curr_out = out + out_strides[5];
        STRI_2x512_S(curr_out, v_out_stride, v_out5, v_out6);
        // Output point 8: x(7) & Output point 9: x(8)
        curr_out = out + out_strides[7];
        STRI_2x512_S(curr_out, v_out_stride, v_out7, v_out8);
        // Output point 10: x(9) & Output point 11: x(10)
        curr_out = out + out_strides[9];
        STRI_2x512_S(curr_out, v_out_stride, v_out9, v_out10);
        // Output point 12: x(11)
        curr_out = out + out_strides[11];
        STR_512_S(curr_out, v_out_stride, v_out11);

        in = in + (v_in_stride << 4);
        out = out + (v_out_stride << 4);
    }
    // tail cases
    if (n & 8)
    {
        __m256 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9, v_in10, v_in11;
        __m256 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
               v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27;
        __m256 v_t0, v_t1, v_t2, v_t3, v_t4, v_t5;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11;

        __m256 v256_CRTM_12_1 = _mm512_castps512_ps256(v_CRTM_12_1);
        __m256 v256_CRTM_12_2 = _mm512_castps512_ps256(v_CRTM_12_2);

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_256_S(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_256_S(curr_in, v_in_stride, v_in1);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_256_S(curr_in, v_in_stride, v_in2);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_256_S(curr_in, v_in_stride, v_in3);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_256_S(curr_in, v_in_stride, v_in4);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_256_S(curr_in, v_in_stride, v_in5);
        // Input point 7: x(6)
        curr_in = in + in_strides[6];
        LDR_256_S(curr_in, v_in_stride, v_in6);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_256_S(curr_in, v_in_stride, v_in7);
        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDR_256_S(curr_in, v_in_stride, v_in8);
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDR_256_S(curr_in, v_in_stride, v_in9);
        // Input point 11: x(10)
        curr_in = in + in_strides[10];
        LDR_256_S(curr_in, v_in_stride, v_in10);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_256_S(curr_in, v_in_stride, v_in11);

        v_s0 = _mm256_add_ps(v_in11, v_in1);
        v_s1 = _mm256_sub_ps(v_in11, v_in1);
        v_s2 = _mm256_add_ps(v_in5, v_in7);
        v_s3 = _mm256_sub_ps(v_in5, v_in7);
        v_s4 = _mm256_add_ps(v_in0, v_in6);
        v_s5 = _mm256_sub_ps(v_in0, v_in6);
        v_s6 = _mm256_add_ps(v_in10, v_in2);
        v_s7 = _mm256_sub_ps(v_in10, v_in2);
        v_s8 = _mm256_add_ps(v_in4, v_in8);
        v_s9 = _mm256_sub_ps(v_in4, v_in8);
        v_s10 = _mm256_add_ps(v_in9, v_in3);
        v_s11 = _mm256_sub_ps(v_in9, v_in3);

        v_s12 = _mm256_add_ps(v_s0, v_s2);
        v_s13 = _mm256_sub_ps(v_s0, v_s2);
        v_s14 = _mm256_add_ps(v_s6, v_s8);
        v_s15 = _mm256_sub_ps(v_s6, v_s8);
        v_s16 = _mm256_add_ps(v_s1, v_s3);
        v_s17 = _mm256_sub_ps(v_s1, v_s3);
        v_s18 = _mm256_add_ps(v_s4, v_s10);
        v_s19 = _mm256_sub_ps(v_s4, v_s10);
        v_s20 = _mm256_add_ps(v_s12, v_s14);
        v_s21 = _mm256_sub_ps(v_s12, v_s14);

        v_t0 = _mm256_mul_ps(v256_CRTM_12_2, v_s15);
        v_t1 = _mm256_mul_ps(v256_CRTM_12_2, v_s17);
        v_s22 = _mm256_add_ps(v_s5, v_t0);
        v_s23 = _mm256_add_ps(v_t1, v_s11);
        v_s24 = _mm256_add_ps(v_s7, v_s9);
        v_s25 = _mm256_sub_ps(v_s7, v_s9);
        v_s26 = _mm256_add_ps(v_s16, v_s24);
        v_s27 = _mm256_sub_ps(v_s16, v_s24);

        v_t2 = _mm256_mul_ps(v256_CRTM_12_1, v_s13);
        v_t3 = _mm256_mul_ps(v256_CRTM_12_2, v_s20);
        v_t4 = _mm256_mul_ps(v256_CRTM_12_2, v_s21);
        v_t5 = _mm256_mul_ps(v256_CRTM_12_1, v_s25);

        v_out0 = _mm256_add_ps(v_s20, v_s18);
        v_out1 = _mm256_add_ps(v_s22, v_t2);
        v_out2 = _mm256_add_ps(v_s23, v_t5);
        v_out3 = _mm256_add_ps(v_t4, v_s19);
        v_out4 = _mm256_mul_ps(v256_CRTM_12_1, v_s26);
        v_out5 = _mm256_sub_ps(v_s5, v_s15);
        v_out6 = _mm256_sub_ps(v_s17, v_s11);
        v_out7 = _mm256_sub_ps(v_s18, v_t3);
        v_out8 = _mm256_mul_ps(v256_CRTM_12_1, v_s27);
        v_out9 = _mm256_sub_ps(v_s22, v_t2);
        v_out10 = _mm256_sub_ps(v_s23, v_t5);
        v_out11 = _mm256_sub_ps(v_s19, v_s21);

        // Output point 1: X(0)
        STR_256_S(curr_out, v_out_stride, v_out0);
        // Output point 2: x(1) & Output point 3: x(2)
        curr_out = out + out_strides[1];
        STRI_2x256_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 4: x(3) & Output point 5: x(4)
        curr_out = out + out_strides[3];
        STRI_2x256_S(curr_out, v_out_stride, v_out3, v_out4);
        // Output point 6: x(5) & Output point 7: x(6)
        curr_out = out + out_strides[5];
        STRI_2x256_S(curr_out, v_out_stride, v_out5, v_out6);
        // Output point 8: x(7) & Output point 9: x(8)
        curr_out = out + out_strides[7];
        STRI_2x256_S(curr_out, v_out_stride, v_out7, v_out8);
        // Output point 10: x(9) & Output point 11: x(10)
        curr_out = out + out_strides[9];
        STRI_2x256_S(curr_out, v_out_stride, v_out9, v_out10);
        // Output point 12: x(11)
        curr_out = out + out_strides[11];
        STR_256_S(curr_out, v_out_stride, v_out11);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (n & 4)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9, v_in10, v_in11;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
               v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27;
        __m128 v_t0, v_t1, v_t2, v_t3, v_t4, v_t5;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11;

        __m128 v128_CRTM_12_1 = _mm512_castps512_ps128(v_CRTM_12_1);
        __m128 v128_CRTM_12_2 = _mm512_castps512_ps128(v_CRTM_12_2);

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
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_128_S(curr_in, v_in_stride, v_in11);

        v_s0 = _mm_add_ps(v_in11, v_in1);
        v_s1 = _mm_sub_ps(v_in11, v_in1);
        v_s2 = _mm_add_ps(v_in5, v_in7);
        v_s3 = _mm_sub_ps(v_in5, v_in7);
        v_s4 = _mm_add_ps(v_in0, v_in6);
        v_s5 = _mm_sub_ps(v_in0, v_in6);
        v_s6 = _mm_add_ps(v_in10, v_in2);
        v_s7 = _mm_sub_ps(v_in10, v_in2);
        v_s8 = _mm_add_ps(v_in4, v_in8);
        v_s9 = _mm_sub_ps(v_in4, v_in8);
        v_s10 = _mm_add_ps(v_in9, v_in3);
        v_s11 = _mm_sub_ps(v_in9, v_in3);

        v_s12 = _mm_add_ps(v_s0, v_s2);
        v_s13 = _mm_sub_ps(v_s0, v_s2);
        v_s14 = _mm_add_ps(v_s6, v_s8);
        v_s15 = _mm_sub_ps(v_s6, v_s8);
        v_s16 = _mm_add_ps(v_s1, v_s3);
        v_s17 = _mm_sub_ps(v_s1, v_s3);
        v_s18 = _mm_add_ps(v_s4, v_s10);
        v_s19 = _mm_sub_ps(v_s4, v_s10);
        v_s20 = _mm_add_ps(v_s12, v_s14);
        v_s21 = _mm_sub_ps(v_s12, v_s14);

        v_t0 = _mm_mul_ps(v128_CRTM_12_2, v_s15);
        v_t1 = _mm_mul_ps(v128_CRTM_12_2, v_s17);
        v_s22 = _mm_add_ps(v_s5, v_t0);
        v_s23 = _mm_add_ps(v_t1, v_s11);
        v_s24 = _mm_add_ps(v_s7, v_s9);
        v_s25 = _mm_sub_ps(v_s7, v_s9);
        v_s26 = _mm_add_ps(v_s16, v_s24);
        v_s27 = _mm_sub_ps(v_s16, v_s24);

        v_t2 = _mm_mul_ps(v128_CRTM_12_1, v_s13);
        v_t3 = _mm_mul_ps(v128_CRTM_12_2, v_s20);
        v_t4 = _mm_mul_ps(v128_CRTM_12_2, v_s21);
        v_t5 = _mm_mul_ps(v128_CRTM_12_1, v_s25);

        v_out0 = _mm_add_ps(v_s20, v_s18);
        v_out1 = _mm_add_ps(v_s22, v_t2);
        v_out2 = _mm_add_ps(v_s23, v_t5);
        v_out3 = _mm_add_ps(v_t4, v_s19);
        v_out4 = _mm_mul_ps(v128_CRTM_12_1, v_s26);
        v_out5 = _mm_sub_ps(v_s5, v_s15);
        v_out6 = _mm_sub_ps(v_s17, v_s11);
        v_out7 = _mm_sub_ps(v_s18, v_t3);
        v_out8 = _mm_mul_ps(v128_CRTM_12_1, v_s27);
        v_out9 = _mm_sub_ps(v_s22, v_t2);
        v_out10 = _mm_sub_ps(v_s23, v_t5);
        v_out11 = _mm_sub_ps(v_s19, v_s21);

        // Output point 1: X(0)
        STR_128_S(curr_out, v_out_stride, v_out0);
        // Output point 2: x(1) & Output point 3: x(2)
        curr_out = out + out_strides[1];
        STRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 4: x(3) & Output point 5: x(4)
        curr_out = out + out_strides[3];
        STRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);
        // Output point 6: x(5) & Output point 7: x(6)
        curr_out = out + out_strides[5];
        STRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);
        // Output point 8: x(7) & Output point 9: x(8)
        curr_out = out + out_strides[7];
        STRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);
        // Output point 10: x(9) & Output point 11: x(10)
        curr_out = out + out_strides[9];
        STRI_2x128_S(curr_out, v_out_stride, v_out9, v_out10);
        // Output point 12: x(11)
        curr_out = out + out_strides[11];
        STR_128_S(curr_out, v_out_stride, v_out11);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (n & 2)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9, v_in10, v_in11;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
               v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27;
        __m128 v_t0, v_t1, v_t2, v_t3, v_t4, v_t5;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_12_1 = _mm512_castps512_ps128(v_CRTM_12_1);
        __m128 v128_CRTM_12_2 = _mm512_castps512_ps128(v_CRTM_12_2);

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

        v_s0 = _mm_add_ps(v_in11, v_in1);
        v_s1 = _mm_sub_ps(v_in11, v_in1);
        v_s2 = _mm_add_ps(v_in5, v_in7);
        v_s3 = _mm_sub_ps(v_in5, v_in7);
        v_s4 = _mm_add_ps(v_in0, v_in6);
        v_s5 = _mm_sub_ps(v_in0, v_in6);
        v_s6 = _mm_add_ps(v_in10, v_in2);
        v_s7 = _mm_sub_ps(v_in10, v_in2);
        v_s8 = _mm_add_ps(v_in4, v_in8);
        v_s9 = _mm_sub_ps(v_in4, v_in8);
        v_s10 = _mm_add_ps(v_in9, v_in3);
        v_s11 = _mm_sub_ps(v_in9, v_in3);

        v_s12 = _mm_add_ps(v_s0, v_s2);
        v_s13 = _mm_sub_ps(v_s0, v_s2);
        v_s14 = _mm_add_ps(v_s6, v_s8);
        v_s15 = _mm_sub_ps(v_s6, v_s8);
        v_s16 = _mm_add_ps(v_s1, v_s3);
        v_s17 = _mm_sub_ps(v_s1, v_s3);
        v_s18 = _mm_add_ps(v_s4, v_s10);
        v_s19 = _mm_sub_ps(v_s4, v_s10);
        v_s20 = _mm_add_ps(v_s12, v_s14);
        v_s21 = _mm_sub_ps(v_s12, v_s14);

        v_t0 = _mm_mul_ps(v128_CRTM_12_2, v_s15);
        v_t1 = _mm_mul_ps(v128_CRTM_12_2, v_s17);
        v_s22 = _mm_add_ps(v_s5, v_t0);
        v_s23 = _mm_add_ps(v_t1, v_s11);
        v_s24 = _mm_add_ps(v_s7, v_s9);
        v_s25 = _mm_sub_ps(v_s7, v_s9);
        v_s26 = _mm_add_ps(v_s16, v_s24);
        v_s27 = _mm_sub_ps(v_s16, v_s24);

        v_t2 = _mm_mul_ps(v128_CRTM_12_1, v_s13);
        v_t3 = _mm_mul_ps(v128_CRTM_12_2, v_s20);
        v_t4 = _mm_mul_ps(v128_CRTM_12_2, v_s21);
        v_t5 = _mm_mul_ps(v128_CRTM_12_1, v_s25);

        v_out0 = _mm_add_ps(v_s20, v_s18);
        v_out1 = _mm_add_ps(v_s22, v_t2);
        v_out2 = _mm_add_ps(v_s23, v_t5);
        v_out3 = _mm_add_ps(v_t4, v_s19);
        v_out4 = _mm_mul_ps(v128_CRTM_12_1, v_s26);
        v_out5 = _mm_sub_ps(v_s5, v_s15);
        v_out6 = _mm_sub_ps(v_s17, v_s11);
        v_out7 = _mm_sub_ps(v_s18, v_t3);
        v_out8 = _mm_mul_ps(v128_CRTM_12_1, v_s27);
        v_out9 = _mm_sub_ps(v_s22, v_t2);
        v_out10 = _mm_sub_ps(v_s23, v_t5);
        v_out11 = _mm_sub_ps(v_s19, v_s21);

        // Output point 1: X(0)
        STHR_128_S(curr_out, v_out_stride, v_out0);
        // Output point 2: x(1) & Output point 3: x(2)
        curr_out = out + out_strides[1];
        STHRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 4: x(3) & Output point 5: x(4)
        curr_out = out + out_strides[3];
        STHRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);
        // Output point 6: x(5) & Output point 7: x(6)
        curr_out = out + out_strides[5];
        STHRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);
        // Output point 8: x(7) & Output point 9: x(8)
        curr_out = out + out_strides[7];
        STHRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);
        // Output point 10: x(9) & Output point 11: x(10)
        curr_out = out + out_strides[9];
        STHRI_2x128_S(curr_out, v_out_stride, v_out9, v_out10);
        // Output point 12: x(11)
        curr_out = out + out_strides[11];
        STHR_128_S(curr_out, v_out_stride, v_out11);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        FLOAT in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11;
        FLOAT s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14,
              s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27;
        FLOAT t0, t1, t2, t3, t4, t5;

        in0 = *in;                 // Input point 1: x(0)
        in1 = in[in_strides[1]];   // Input point 2: x(1)
        in2 = in[in_strides[2]];   // Input point 3: x(2)
        in3 = in[in_strides[3]];   // Input point 4: x(3)
        in4 = in[in_strides[4]];   // Input point 5: x(4)
        in5 = in[in_strides[5]];   // Input point 6: x(5)
        in6 = in[in_strides[6]];   // Input point 7: x(6)
        in7 = in[in_strides[7]];   // Input point 8: x(7)
        in8 = in[in_strides[8]];   // Input point 9: x(8)
        in9 = in[in_strides[9]];   // Input point 10: x(9)
        in10 = in[in_strides[10]]; // Input point 11: x(10)
        in11 = in[in_strides[11]]; // Input point 12: x(11)

        s0 = in11 + in1;
        s1 = in11 - in1;
        s2 = in5 + in7;
        s3 = in5 - in7;
        s4 = in0 + in6;
        s5 = in0 - in6;
        s6 = in10 + in2;
        s7 = in10 - in2;
        s8 = in4 + in8;
        s9 = in4 - in8;
        s10 = in9 + in3;
        s11 = in9 - in3;

        s12 = s0 + s2;
        s13 = s0 - s2;
        s14 = s6 + s8;
        s15 = s6 - s8;
        s16 = s1 + s3;
        s17 = s1 - s3;
        s18 = s4 + s10;
        s19 = s4 - s10;
        s20 = s12 + s14;
        s21 = s12 - s14;

        t0 = CRTM_12_2 * s15;
        t1 = CRTM_12_2 * s17;
        s22 = s5 + t0;
        s23 = t1 + s11;
        s24 = s7 + s9;
        s25 = s7 - s9;
        s26 = s16 + s24;
        s27 = s16 - s24;

        t2 = CRTM_12_1 * s13;
        t3 = CRTM_12_2 * s20;
        t4 = CRTM_12_2 * s21;
        t5 = CRTM_12_1 * s25;

        *out = s20 + s18;                      // output pt 1: X(0)
        out[out_strides[1]] = s22 + t2;        // output pt 2: X(1)
        out[out_strides[2]] = s23 + t5;        // output pt 2: X(1)
        out[out_strides[3]] = t4 + s19;        // output pt 4: X(3)
        out[out_strides[4]] = CRTM_12_1 * s26; // output pt 5: X(4)
        out[out_strides[5]] = s5 - s15;        // output pt 6: X(5)
        out[out_strides[6]] = s17 - s11;       // output pt 7: X(6)
        out[out_strides[7]] = s18 - t3;        // output pt 8: X(7)
        out[out_strides[8]] = CRTM_12_1 * s27; // output pt 9: X(8)
        out[out_strides[9]] = s22 - t2;        // output pt 10: X(9)
        out[out_strides[10]] = s23 - t5;       // output pt 11: X(10)
        out[out_strides[11]] = s19 - s21;      // output pt 12: X(11)
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hc_rfft12avx512_fp32_bwd(VOID *in_real, VOID *in_imag,
                                       VOID *out_real, VOID *out_imag, INTP n,
                                       aoclfftz_strides_t *strides, VOID *twd,
                                       UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_12_1 = 1.732050807568877293527446341505872366942805254f;
    const FLOAT CRTM_12_2 = 2.000000000000000000000000000000000000000000000f;

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
    INTP N = n >> 4;

    __m512 v_CRTM_12_1 = _mm512_set1_ps(CRTM_12_1);
    __m512 v_CRTM_12_2 = _mm512_set1_ps(CRTM_12_2);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m512 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9, v_in10, v_in11;
        __m512 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
               v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26;
        __m512 v_t0, v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8;
        __m512 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_512_S(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x512_S(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x512_S(curr_in, v_in_stride, v_in3, v_in4);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDRI_2x512_S(curr_in, v_in_stride, v_in5, v_in6);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in + in_strides[7];
        LDRI_2x512_S(curr_in, v_in_stride, v_in7, v_in8);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDRI_2x512_S(curr_in, v_in_stride, v_in9, v_in10);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_512_S(curr_in, v_in_stride, v_in11);

        v_s0 = _mm512_add_ps(v_in0, v_in11);
        v_s1 = _mm512_sub_ps(v_in0, v_in11);
        v_s2 = _mm512_add_ps(v_in1, v_in9);
        v_s3 = _mm512_sub_ps(v_in1, v_in9);
        v_s4 = _mm512_add_ps(v_in3, v_in7);
        v_s5 = _mm512_sub_ps(v_in3, v_in7);
        v_s6 = _mm512_add_ps(v_in4, v_in8);
        v_s7 = _mm512_sub_ps(v_in4, v_in8);
        v_s8 = _mm512_add_ps(v_in10, v_in2);
        v_s9 = _mm512_sub_ps(v_in10, v_in2);

        v_t0 = _mm512_mul_ps(v_CRTM_12_2, v_in5);
        v_t1 = _mm512_mul_ps(v_CRTM_12_2, v_in6);

        v_s10 = _mm512_add_ps(v_s2, v_s4);
        v_s11 = _mm512_add_ps(v_s0, v_t0);
        v_s12 = _mm512_sub_ps(v_s0, v_t0);
        v_s13 = _mm512_mul_ps(v_CRTM_12_2, v_s10);

        v_s14 = _mm512_add_ps(v_s3, v_s6);
        v_s15 = _mm512_sub_ps(v_s3, v_s6);
        v_t2 = _mm512_mul_ps(v_CRTM_12_1, v_s14);
        v_t3 = _mm512_mul_ps(v_CRTM_12_1, v_s15);

        v_s16 = _mm512_add_ps(v_s9, v_s7);
        v_s17 = _mm512_sub_ps(v_s9, v_s7);
        v_t4 = _mm512_mul_ps(v_CRTM_12_1, v_s16);
        v_t5 = _mm512_mul_ps(v_CRTM_12_1, v_s17);

        v_s18 = _mm512_add_ps(v_s5, v_s8);
        v_s19 = _mm512_sub_ps(v_s5, v_s8);
        v_s20 = _mm512_sub_ps(v_s2, v_s4);
        v_t6 = _mm512_mul_ps(v_CRTM_12_2, v_s18);
        v_t7 = _mm512_mul_ps(v_CRTM_12_2, v_s19);
        v_t8 = _mm512_mul_ps(v_CRTM_12_2, v_s20);

        v_s21 = _mm512_sub_ps(v_s1, v_t1);
        v_s22 = _mm512_add_ps(v_s1, v_t1);
        v_s23 = _mm512_add_ps(v_s19, v_s21);
        v_s24 = _mm512_add_ps(v_s22, v_s18);
        v_s25 = _mm512_sub_ps(v_s11, v_s10);
        v_s26 = _mm512_add_ps(v_s12, v_s20);

        v_out0 = _mm512_add_ps(v_s11, v_s13);
        v_out1 = _mm512_add_ps(v_t3, v_s23);
        v_out2 = _mm512_add_ps(v_t5, v_s26);
        v_out3 = _mm512_sub_ps(v_s22, v_t6);
        v_out4 = _mm512_add_ps(v_s25, v_t4);
        v_out5 = _mm512_sub_ps(v_s23, v_t3);
        v_out6 = _mm512_sub_ps(v_s12, v_t8);
        v_out7 = _mm512_sub_ps(v_s24, v_t2);
        v_out8 = _mm512_sub_ps(v_s25, v_t4);
        v_out9 = _mm512_sub_ps(v_s21, v_t7);
        v_out10 = _mm512_sub_ps(v_s26, v_t5);
        v_out11 = _mm512_add_ps(v_s24, v_t2);

        // Output point 1: x(0)
        STR_512_S(curr_out, v_out_stride, v_out0);
        // Output point 2: x(1)
        curr_out = out + out_strides[1];
        STR_512_S(curr_out, v_out_stride, v_out1);
        // Output point 3: x(2)
        curr_out = out + out_strides[2];
        STR_512_S(curr_out, v_out_stride, v_out2);
        // Output point 4: x(3)
        curr_out = out + out_strides[3];
        STR_512_S(curr_out, v_out_stride, v_out3);
        // Output point 5: x(4)
        curr_out = out + out_strides[4];
        STR_512_S(curr_out, v_out_stride, v_out4);
        // Output point 6: x(5)
        curr_out = out + out_strides[5];
        STR_512_S(curr_out, v_out_stride, v_out5);
        // Output point 7: x(6)
        curr_out = out + out_strides[6];
        STR_512_S(curr_out, v_out_stride, v_out6);
        // Output point 8: x(7)
        curr_out = out + out_strides[7];
        STR_512_S(curr_out, v_out_stride, v_out7);
        // Output point 9: x(8)
        curr_out = out + out_strides[8];
        STR_512_S(curr_out, v_out_stride, v_out8);
        // Output point 10: x(9)
        curr_out = out + out_strides[9];
        STR_512_S(curr_out, v_out_stride, v_out9);
        // Output point 11: x(10)
        curr_out = out + out_strides[10];
        STR_512_S(curr_out, v_out_stride, v_out10);
        // Output point 12: x(11)
        curr_out = out + out_strides[11];
        STR_512_S(curr_out, v_out_stride, v_out11);

        in = in + (v_in_stride << 4);
        out = out + (v_out_stride << 4);
    }
    // tail cases
    if (n & 8)
    {
        __m256 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9, v_in10, v_in11;
        __m256 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
               v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26;
        __m256 v_t0, v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11;

        curr_in = in;
        curr_out = out;

        __m256 v256_CRTM_12_1 = _mm512_castps512_ps256(v_CRTM_12_1);
        __m256 v256_CRTM_12_2 = _mm512_castps512_ps256(v_CRTM_12_2);

        // Input point 1: x(0)
        LDR_256_S(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x256_S(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x256_S(curr_in, v_in_stride, v_in3, v_in4);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDRI_2x256_S(curr_in, v_in_stride, v_in5, v_in6);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in + in_strides[7];
        LDRI_2x256_S(curr_in, v_in_stride, v_in7, v_in8);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDRI_2x256_S(curr_in, v_in_stride, v_in9, v_in10);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_256_S(curr_in, v_in_stride, v_in11);

        v_s0 = _mm256_add_ps(v_in0, v_in11);
        v_s1 = _mm256_sub_ps(v_in0, v_in11);
        v_s2 = _mm256_add_ps(v_in1, v_in9);
        v_s3 = _mm256_sub_ps(v_in1, v_in9);
        v_s4 = _mm256_add_ps(v_in3, v_in7);
        v_s5 = _mm256_sub_ps(v_in3, v_in7);
        v_s6 = _mm256_add_ps(v_in4, v_in8);
        v_s7 = _mm256_sub_ps(v_in4, v_in8);
        v_s8 = _mm256_add_ps(v_in10, v_in2);
        v_s9 = _mm256_sub_ps(v_in10, v_in2);

        v_t0 = _mm256_mul_ps(v256_CRTM_12_2, v_in5);
        v_t1 = _mm256_mul_ps(v256_CRTM_12_2, v_in6);

        v_s10 = _mm256_add_ps(v_s2, v_s4);
        v_s11 = _mm256_add_ps(v_s0, v_t0);
        v_s12 = _mm256_sub_ps(v_s0, v_t0);
        v_s13 = _mm256_mul_ps(v256_CRTM_12_2, v_s10);

        v_s14 = _mm256_add_ps(v_s3, v_s6);
        v_s15 = _mm256_sub_ps(v_s3, v_s6);
        v_t2 = _mm256_mul_ps(v256_CRTM_12_1, v_s14);
        v_t3 = _mm256_mul_ps(v256_CRTM_12_1, v_s15);

        v_s16 = _mm256_add_ps(v_s9, v_s7);
        v_s17 = _mm256_sub_ps(v_s9, v_s7);
        v_t4 = _mm256_mul_ps(v256_CRTM_12_1, v_s16);
        v_t5 = _mm256_mul_ps(v256_CRTM_12_1, v_s17);

        v_s18 = _mm256_add_ps(v_s5, v_s8);
        v_s19 = _mm256_sub_ps(v_s5, v_s8);
        v_s20 = _mm256_sub_ps(v_s2, v_s4);
        v_t6 = _mm256_mul_ps(v256_CRTM_12_2, v_s18);
        v_t7 = _mm256_mul_ps(v256_CRTM_12_2, v_s19);
        v_t8 = _mm256_mul_ps(v256_CRTM_12_2, v_s20);

        v_s21 = _mm256_sub_ps(v_s1, v_t1);
        v_s22 = _mm256_add_ps(v_s1, v_t1);
        v_s23 = _mm256_add_ps(v_s19, v_s21);
        v_s24 = _mm256_add_ps(v_s22, v_s18);
        v_s25 = _mm256_sub_ps(v_s11, v_s10);
        v_s26 = _mm256_add_ps(v_s12, v_s20);

        v_out0 = _mm256_add_ps(v_s11, v_s13);
        v_out1 = _mm256_add_ps(v_t3, v_s23);
        v_out2 = _mm256_add_ps(v_t5, v_s26);
        v_out3 = _mm256_sub_ps(v_s22, v_t6);
        v_out4 = _mm256_add_ps(v_s25, v_t4);
        v_out5 = _mm256_sub_ps(v_s23, v_t3);
        v_out6 = _mm256_sub_ps(v_s12, v_t8);
        v_out7 = _mm256_sub_ps(v_s24, v_t2);
        v_out8 = _mm256_sub_ps(v_s25, v_t4);
        v_out9 = _mm256_sub_ps(v_s21, v_t7);
        v_out10 = _mm256_sub_ps(v_s26, v_t5);
        v_out11 = _mm256_add_ps(v_s24, v_t2);

        // Output point 1: x(0)
        STR_256_S(curr_out, v_out_stride, v_out0);
        // Output point 2: x(1)
        curr_out = out + out_strides[1];
        STR_256_S(curr_out, v_out_stride, v_out1);
        // Output point 3: x(2)
        curr_out = out + out_strides[2];
        STR_256_S(curr_out, v_out_stride, v_out2);
        // Output point 4: x(3)
        curr_out = out + out_strides[3];
        STR_256_S(curr_out, v_out_stride, v_out3);
        // Output point 5: x(4)
        curr_out = out + out_strides[4];
        STR_256_S(curr_out, v_out_stride, v_out4);
        // Output point 6: x(5)
        curr_out = out + out_strides[5];
        STR_256_S(curr_out, v_out_stride, v_out5);
        // Output point 7: x(6)
        curr_out = out + out_strides[6];
        STR_256_S(curr_out, v_out_stride, v_out6);
        // Output point 8: x(7)
        curr_out = out + out_strides[7];
        STR_256_S(curr_out, v_out_stride, v_out7);
        // Output point 9: x(8)
        curr_out = out + out_strides[8];
        STR_256_S(curr_out, v_out_stride, v_out8);
        // Output point 10: x(9)
        curr_out = out + out_strides[9];
        STR_256_S(curr_out, v_out_stride, v_out9);
        // Output point 11: x(10)
        curr_out = out + out_strides[10];
        STR_256_S(curr_out, v_out_stride, v_out10);
        // Output point 12: x(11)
        curr_out = out + out_strides[11];
        STR_256_S(curr_out, v_out_stride, v_out11);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (n & 4)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9, v_in10, v_in11;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
               v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26;
        __m128 v_t0, v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_12_1 = _mm512_castps512_ps128(v_CRTM_12_1);
        __m128 v128_CRTM_12_2 = _mm512_castps512_ps128(v_CRTM_12_2);

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_stride, v_in0);
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
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_128_S(curr_in, v_in_stride, v_in11);

        v_s0 = _mm_add_ps(v_in0, v_in11);
        v_s1 = _mm_sub_ps(v_in0, v_in11);
        v_s2 = _mm_add_ps(v_in1, v_in9);
        v_s3 = _mm_sub_ps(v_in1, v_in9);
        v_s4 = _mm_add_ps(v_in3, v_in7);
        v_s5 = _mm_sub_ps(v_in3, v_in7);
        v_s6 = _mm_add_ps(v_in4, v_in8);
        v_s7 = _mm_sub_ps(v_in4, v_in8);
        v_s8 = _mm_add_ps(v_in10, v_in2);
        v_s9 = _mm_sub_ps(v_in10, v_in2);

        v_t0 = _mm_mul_ps(v128_CRTM_12_2, v_in5);
        v_t1 = _mm_mul_ps(v128_CRTM_12_2, v_in6);

        v_s10 = _mm_add_ps(v_s2, v_s4);
        v_s11 = _mm_add_ps(v_s0, v_t0);
        v_s12 = _mm_sub_ps(v_s0, v_t0);
        v_s13 = _mm_mul_ps(v128_CRTM_12_2, v_s10);

        v_s14 = _mm_add_ps(v_s3, v_s6);
        v_s15 = _mm_sub_ps(v_s3, v_s6);
        v_t2 = _mm_mul_ps(v128_CRTM_12_1, v_s14);
        v_t3 = _mm_mul_ps(v128_CRTM_12_1, v_s15);

        v_s16 = _mm_add_ps(v_s9, v_s7);
        v_s17 = _mm_sub_ps(v_s9, v_s7);
        v_t4 = _mm_mul_ps(v128_CRTM_12_1, v_s16);
        v_t5 = _mm_mul_ps(v128_CRTM_12_1, v_s17);

        v_s18 = _mm_add_ps(v_s5, v_s8);
        v_s19 = _mm_sub_ps(v_s5, v_s8);
        v_s20 = _mm_sub_ps(v_s2, v_s4);
        v_t6 = _mm_mul_ps(v128_CRTM_12_2, v_s18);
        v_t7 = _mm_mul_ps(v128_CRTM_12_2, v_s19);
        v_t8 = _mm_mul_ps(v128_CRTM_12_2, v_s20);

        v_s21 = _mm_sub_ps(v_s1, v_t1);
        v_s22 = _mm_add_ps(v_s1, v_t1);
        v_s23 = _mm_add_ps(v_s19, v_s21);
        v_s24 = _mm_add_ps(v_s22, v_s18);
        v_s25 = _mm_sub_ps(v_s11, v_s10);
        v_s26 = _mm_add_ps(v_s12, v_s20);

        v_out0 = _mm_add_ps(v_s11, v_s13);
        v_out1 = _mm_add_ps(v_t3, v_s23);
        v_out2 = _mm_add_ps(v_t5, v_s26);
        v_out3 = _mm_sub_ps(v_s22, v_t6);
        v_out4 = _mm_add_ps(v_s25, v_t4);
        v_out5 = _mm_sub_ps(v_s23, v_t3);
        v_out6 = _mm_sub_ps(v_s12, v_t8);
        v_out7 = _mm_sub_ps(v_s24, v_t2);
        v_out8 = _mm_sub_ps(v_s25, v_t4);
        v_out9 = _mm_sub_ps(v_s21, v_t7);
        v_out10 = _mm_sub_ps(v_s26, v_t5);
        v_out11 = _mm_add_ps(v_s24, v_t2);

        // Output point 1: x(0)
        STR_128_S(curr_out, v_out_stride, v_out0);
        // Output point 2: x(1)
        curr_out = out + out_strides[1];
        STR_128_S(curr_out, v_out_stride, v_out1);
        // Output point 2: x(1)
        curr_out = out + out_strides[2];
        STR_128_S(curr_out, v_out_stride, v_out2);
        // Output point 4: x(3)
        curr_out = out + out_strides[3];
        STR_128_S(curr_out, v_out_stride, v_out3);
        // Output point 5: x(4)
        curr_out = out + out_strides[4];
        STR_128_S(curr_out, v_out_stride, v_out4);
        // Output point 6: x(5)
        curr_out = out + out_strides[5];
        STR_128_S(curr_out, v_out_stride, v_out5);
        // Output point 7: x(6)
        curr_out = out + out_strides[6];
        STR_128_S(curr_out, v_out_stride, v_out6);
        // Output point 8: x(7)
        curr_out = out + out_strides[7];
        STR_128_S(curr_out, v_out_stride, v_out7);
        // Output point 9: x(8)
        curr_out = out + out_strides[8];
        STR_128_S(curr_out, v_out_stride, v_out8);
        // Output point 10: x(9)
        curr_out = out + out_strides[9];
        STR_128_S(curr_out, v_out_stride, v_out9);
        // Output point 11: x(10)
        curr_out = out + out_strides[10];
        STR_128_S(curr_out, v_out_stride, v_out10);
        // Output point 12: x(11)
        curr_out = out + out_strides[11];
        STR_128_S(curr_out, v_out_stride, v_out11);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (n & 2)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9, v_in10, v_in11;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
               v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26;
        __m128 v_t0, v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_12_1 = _mm512_castps512_ps128(v_CRTM_12_1);
        __m128 v128_CRTM_12_2 = _mm512_castps512_ps128(v_CRTM_12_2);

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
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDHR_128_S(curr_in, v_in_stride, v_in11);

        v_s0 = _mm_add_ps(v_in0, v_in11);
        v_s1 = _mm_sub_ps(v_in0, v_in11);
        v_s2 = _mm_add_ps(v_in1, v_in9);
        v_s3 = _mm_sub_ps(v_in1, v_in9);
        v_s4 = _mm_add_ps(v_in3, v_in7);
        v_s5 = _mm_sub_ps(v_in3, v_in7);
        v_s6 = _mm_add_ps(v_in4, v_in8);
        v_s7 = _mm_sub_ps(v_in4, v_in8);
        v_s8 = _mm_add_ps(v_in10, v_in2);
        v_s9 = _mm_sub_ps(v_in10, v_in2);

        v_t0 = _mm_mul_ps(v128_CRTM_12_2, v_in5);
        v_t1 = _mm_mul_ps(v128_CRTM_12_2, v_in6);

        v_s10 = _mm_add_ps(v_s2, v_s4);
        v_s11 = _mm_add_ps(v_s0, v_t0);
        v_s12 = _mm_sub_ps(v_s0, v_t0);
        v_s13 = _mm_mul_ps(v128_CRTM_12_2, v_s10);

        v_s14 = _mm_add_ps(v_s3, v_s6);
        v_s15 = _mm_sub_ps(v_s3, v_s6);
        v_t2 = _mm_mul_ps(v128_CRTM_12_1, v_s14);
        v_t3 = _mm_mul_ps(v128_CRTM_12_1, v_s15);

        v_s16 = _mm_add_ps(v_s9, v_s7);
        v_s17 = _mm_sub_ps(v_s9, v_s7);
        v_t4 = _mm_mul_ps(v128_CRTM_12_1, v_s16);
        v_t5 = _mm_mul_ps(v128_CRTM_12_1, v_s17);

        v_s18 = _mm_add_ps(v_s5, v_s8);
        v_s19 = _mm_sub_ps(v_s5, v_s8);
        v_s20 = _mm_sub_ps(v_s2, v_s4);
        v_t6 = _mm_mul_ps(v128_CRTM_12_2, v_s18);
        v_t7 = _mm_mul_ps(v128_CRTM_12_2, v_s19);
        v_t8 = _mm_mul_ps(v128_CRTM_12_2, v_s20);

        v_s21 = _mm_sub_ps(v_s1, v_t1);
        v_s22 = _mm_add_ps(v_s1, v_t1);
        v_s23 = _mm_add_ps(v_s19, v_s21);
        v_s24 = _mm_add_ps(v_s22, v_s18);
        v_s25 = _mm_sub_ps(v_s11, v_s10);
        v_s26 = _mm_add_ps(v_s12, v_s20);

        v_out0 = _mm_add_ps(v_s11, v_s13);
        v_out1 = _mm_add_ps(v_t3, v_s23);
        v_out2 = _mm_add_ps(v_t5, v_s26);
        v_out3 = _mm_sub_ps(v_s22, v_t6);
        v_out4 = _mm_add_ps(v_s25, v_t4);
        v_out5 = _mm_sub_ps(v_s23, v_t3);
        v_out6 = _mm_sub_ps(v_s12, v_t8);
        v_out7 = _mm_sub_ps(v_s24, v_t2);
        v_out8 = _mm_sub_ps(v_s25, v_t4);
        v_out9 = _mm_sub_ps(v_s21, v_t7);
        v_out10 = _mm_sub_ps(v_s26, v_t5);
        v_out11 = _mm_add_ps(v_s24, v_t2);

        // Output point 1: x(0)
        STHR_128_S(curr_out, v_out_stride, v_out0);
        // Output point 2: x(1)
        curr_out = out + out_strides[1];
        STHR_128_S(curr_out, v_out_stride, v_out1);
        // Output point 3: x(2)
        curr_out = out + out_strides[2];
        STHR_128_S(curr_out, v_out_stride, v_out2);
        // Output point 4: x(3)
        curr_out = out + out_strides[3];
        STHR_128_S(curr_out, v_out_stride, v_out3);
        // Output point 5: x(4)
        curr_out = out + out_strides[4];
        STHR_128_S(curr_out, v_out_stride, v_out4);
        // Output point 6: x(5)
        curr_out = out + out_strides[5];
        STHR_128_S(curr_out, v_out_stride, v_out5);
        // Output point 7: x(6)
        curr_out = out + out_strides[6];
        STHR_128_S(curr_out, v_out_stride, v_out6);
        // Output point 8: x(7)
        curr_out = out + out_strides[7];
        STHR_128_S(curr_out, v_out_stride, v_out7);
        // Output point 9: x(8)
        curr_out = out + out_strides[8];
        STHR_128_S(curr_out, v_out_stride, v_out8);
        // Output point 10: x(9)
        curr_out = out + out_strides[9];
        STHR_128_S(curr_out, v_out_stride, v_out9);
        // Output point 11: x(10)
        curr_out = out + out_strides[10];
        STHR_128_S(curr_out, v_out_stride, v_out10);
        // Output point 12: x(11)
        curr_out = out + out_strides[11];
        STHR_128_S(curr_out, v_out_stride, v_out11);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        FLOAT in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11;
        FLOAT s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14,
              s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26;
        FLOAT t0, t1, t2, t3, t4, t5, t6, t7, t8;

        in0 = *in;                 // Input point 1: X(0)
        in1 = in[in_strides[1]];   // Input point 2: X(1)
        in2 = in[in_strides[2]];   // Input point 3: X(2)
        in3 = in[in_strides[3]];   // Input point 4: X(3)
        in4 = in[in_strides[4]];   // Input point 5: X(4)
        in5 = in[in_strides[5]];   // Input point 6: X(5)
        in6 = in[in_strides[6]];   // Input point 7: X(6)
        in7 = in[in_strides[7]];   // Input point 8: X(7)
        in8 = in[in_strides[8]];   // Input point 9: X(8)
        in9 = in[in_strides[9]];   // Input point 10: X(9)
        in10 = in[in_strides[10]]; // Input point 11: X(10)
        in11 = in[in_strides[11]]; // Input point 12: X(11)

        s0 = in0 + in11;
        s1 = in0 - in11;
        s2 = in1 + in9;
        s3 = in1 - in9;
        s4 = in3 + in7;
        s5 = in3 - in7;
        s6 = in4 + in8;
        s7 = in4 - in8;
        s8 = in10 + in2;
        s9 = in10 - in2;

        t0 = CRTM_12_2 * in5;
        t1 = CRTM_12_2 * in6;

        s10 = s2 + s4;
        s11 = s0 + t0;
        s12 = s0 - t0;
        s13 = CRTM_12_2 * s10;

        s14 = s3 + s6;
        s15 = s3 - s6;
        t2 = CRTM_12_1 * s14;
        t3 = CRTM_12_1 * s15;

        s16 = s9 + s7;
        s17 = s9 - s7;
        t4 = CRTM_12_1 * s16;
        t5 = CRTM_12_1 * s17;

        s18 = s5 + s8;
        s19 = s5 - s8;
        s20 = s2 - s4;
        t6 = CRTM_12_2 * s18;
        t7 = CRTM_12_2 * s19;
        t8 = CRTM_12_2 * s20;

        s21 = s1 - t1;
        s22 = s1 + t1;
        s23 = s19 + s21;
        s24 = s22 + s18;
        s25 = s11 - s10;
        s26 = s12 + s20;

        *out = s11 + s13;                // output pt 1: x(0)
        out[out_strides[1]] = t3 + s23;  // output pt 2: x(1)
        out[out_strides[2]] = t5 + s26;  // output pt 3: x(2)
        out[out_strides[3]] = s22 - t6;  // output pt 4: x(3)
        out[out_strides[4]] = s25 + t4;  // output pt 5: x(4)
        out[out_strides[5]] = s23 - t3;  // output pt 6: x(5)
        out[out_strides[6]] = s12 - t8;  // output pt 7: x(6)
        out[out_strides[7]] = s24 - t2;  // output pt 8: x(7)
        out[out_strides[8]] = s25 - t4;  // output pt 9: x(8)
        out[out_strides[9]] = s21 - t7;  // output pt 10: x(9)
        out[out_strides[10]] = s26 - t5; // output pt 11: x(10)
        out[out_strides[11]] = s24 + t2; // output pt 12: x(11)
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hc_rfft12avx512_fp64_fwd(VOID *in_real, VOID *in_imag,
                                       VOID *out_real, VOID *out_imag, INTP n,
                                       aoclfftz_strides_t *strides, VOID *twd,
                                       UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_12_1 = 0.866025403784438646763723170752936183471402627;
    const DOUBLE CRTM_12_2 = 0.500000000000000000000000000000000000000000000;

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
    INTP N = n >> 3;

    __m512d v_CRTM_12_1 = _mm512_set1_pd(CRTM_12_1);
    __m512d v_CRTM_12_2 = _mm512_set1_pd(CRTM_12_2);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m512d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
                v_in9, v_in10, v_in11;
        __m512d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
                v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
                v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27;
        __m512d v_t0, v_t1, v_t2, v_t3, v_t4, v_t5;
        __m512d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_512_D(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_512_D(curr_in, v_in_stride, v_in1);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_512_D(curr_in, v_in_stride, v_in2);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_512_D(curr_in, v_in_stride, v_in3);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_512_D(curr_in, v_in_stride, v_in4);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_512_D(curr_in, v_in_stride, v_in5);
        // Input point 7: x(6)
        curr_in = in + in_strides[6];
        LDR_512_D(curr_in, v_in_stride, v_in6);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_512_D(curr_in, v_in_stride, v_in7);
        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDR_512_D(curr_in, v_in_stride, v_in8);
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDR_512_D(curr_in, v_in_stride, v_in9);
        // Input point 11: x(10)
        curr_in = in + in_strides[10];
        LDR_512_D(curr_in, v_in_stride, v_in10);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_512_D(curr_in, v_in_stride, v_in11);

        v_s0 = _mm512_add_pd(v_in11, v_in1);
        v_s1 = _mm512_sub_pd(v_in11, v_in1);
        v_s2 = _mm512_add_pd(v_in5, v_in7);
        v_s3 = _mm512_sub_pd(v_in5, v_in7);
        v_s4 = _mm512_add_pd(v_in0, v_in6);
        v_s5 = _mm512_sub_pd(v_in0, v_in6);
        v_s6 = _mm512_add_pd(v_in10, v_in2);
        v_s7 = _mm512_sub_pd(v_in10, v_in2);
        v_s8 = _mm512_add_pd(v_in4, v_in8);
        v_s9 = _mm512_sub_pd(v_in4, v_in8);
        v_s10 = _mm512_add_pd(v_in9, v_in3);
        v_s11 = _mm512_sub_pd(v_in9, v_in3);

        v_s12 = _mm512_add_pd(v_s0, v_s2);
        v_s13 = _mm512_sub_pd(v_s0, v_s2);
        v_s14 = _mm512_add_pd(v_s6, v_s8);
        v_s15 = _mm512_sub_pd(v_s6, v_s8);
        v_s16 = _mm512_add_pd(v_s1, v_s3);
        v_s17 = _mm512_sub_pd(v_s1, v_s3);
        v_s18 = _mm512_add_pd(v_s4, v_s10);
        v_s19 = _mm512_sub_pd(v_s4, v_s10);
        v_s20 = _mm512_add_pd(v_s12, v_s14);
        v_s21 = _mm512_sub_pd(v_s12, v_s14);

        v_t0 = _mm512_mul_pd(v_CRTM_12_2, v_s15);
        v_t1 = _mm512_mul_pd(v_CRTM_12_2, v_s17);
        v_s22 = _mm512_add_pd(v_s5, v_t0);
        v_s23 = _mm512_add_pd(v_t1, v_s11);
        v_s24 = _mm512_add_pd(v_s7, v_s9);
        v_s25 = _mm512_sub_pd(v_s7, v_s9);
        v_s26 = _mm512_add_pd(v_s16, v_s24);
        v_s27 = _mm512_sub_pd(v_s16, v_s24);

        v_t2 = _mm512_mul_pd(v_CRTM_12_1, v_s13);
        v_t3 = _mm512_mul_pd(v_CRTM_12_2, v_s20);
        v_t4 = _mm512_mul_pd(v_CRTM_12_2, v_s21);
        v_t5 = _mm512_mul_pd(v_CRTM_12_1, v_s25);

        v_out0 = _mm512_add_pd(v_s20, v_s18);
        v_out1 = _mm512_add_pd(v_s22, v_t2);
        v_out2 = _mm512_add_pd(v_s23, v_t5);
        v_out3 = _mm512_add_pd(v_t4, v_s19);
        v_out4 = _mm512_mul_pd(v_CRTM_12_1, v_s26);
        v_out5 = _mm512_sub_pd(v_s5, v_s15);
        v_out6 = _mm512_sub_pd(v_s17, v_s11);
        v_out7 = _mm512_sub_pd(v_s18, v_t3);
        v_out8 = _mm512_mul_pd(v_CRTM_12_1, v_s27);
        v_out9 = _mm512_sub_pd(v_s22, v_t2);
        v_out10 = _mm512_sub_pd(v_s23, v_t5);
        v_out11 = _mm512_sub_pd(v_s19, v_s21);

        // Output point 1: X(0)
        STR_512_D(curr_out, v_out_stride, v_out0);
        // Output point 2: x(1) & Output point 3: x(2)
        curr_out = out + out_strides[1];
        STRI_2x512_D(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 4: x(3) & Output point 5: x(4)
        curr_out = out + out_strides[3];
        STRI_2x512_D(curr_out, v_out_stride, v_out3, v_out4);
        // Output point 6: x(5) & Output point 7: x(6)
        curr_out = out + out_strides[5];
        STRI_2x512_D(curr_out, v_out_stride, v_out5, v_out6);
        // Output point 8: x(7) & Output point 9: x(8)
        curr_out = out + out_strides[7];
        STRI_2x512_D(curr_out, v_out_stride, v_out7, v_out8);
        // Output point 10: x(9) & Output point 11: x(10)
        curr_out = out + out_strides[9];
        STRI_2x512_D(curr_out, v_out_stride, v_out9, v_out10);
        // Output point 12: x(11)
        curr_out = out + out_strides[11];
        STR_512_D(curr_out, v_out_stride, v_out11);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (n & 4)
    {
        __m256d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
                v_in9, v_in10, v_in11;
        __m256d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
                v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
                v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27;
        __m256d v_t0, v_t1, v_t2, v_t3, v_t4, v_t5;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11;

        __m256d v256_CRTM_12_1 = _mm512_castpd512_pd256(v_CRTM_12_1);
        __m256d v256_CRTM_12_2 = _mm512_castpd512_pd256(v_CRTM_12_2);

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_256_D(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_256_D(curr_in, v_in_stride, v_in1);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_256_D(curr_in, v_in_stride, v_in2);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_256_D(curr_in, v_in_stride, v_in3);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_256_D(curr_in, v_in_stride, v_in4);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_256_D(curr_in, v_in_stride, v_in5);
        // Input point 7: x(6)
        curr_in = in + in_strides[6];
        LDR_256_D(curr_in, v_in_stride, v_in6);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_256_D(curr_in, v_in_stride, v_in7);
        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDR_256_D(curr_in, v_in_stride, v_in8);
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDR_256_D(curr_in, v_in_stride, v_in9);
        // Input point 11: x(10)
        curr_in = in + in_strides[10];
        LDR_256_D(curr_in, v_in_stride, v_in10);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_256_D(curr_in, v_in_stride, v_in11);

        v_s0 = _mm256_add_pd(v_in11, v_in1);
        v_s1 = _mm256_sub_pd(v_in11, v_in1);
        v_s2 = _mm256_add_pd(v_in5, v_in7);
        v_s3 = _mm256_sub_pd(v_in5, v_in7);
        v_s4 = _mm256_add_pd(v_in0, v_in6);
        v_s5 = _mm256_sub_pd(v_in0, v_in6);
        v_s6 = _mm256_add_pd(v_in10, v_in2);
        v_s7 = _mm256_sub_pd(v_in10, v_in2);
        v_s8 = _mm256_add_pd(v_in4, v_in8);
        v_s9 = _mm256_sub_pd(v_in4, v_in8);
        v_s10 = _mm256_add_pd(v_in9, v_in3);
        v_s11 = _mm256_sub_pd(v_in9, v_in3);

        v_s12 = _mm256_add_pd(v_s0, v_s2);
        v_s13 = _mm256_sub_pd(v_s0, v_s2);
        v_s14 = _mm256_add_pd(v_s6, v_s8);
        v_s15 = _mm256_sub_pd(v_s6, v_s8);
        v_s16 = _mm256_add_pd(v_s1, v_s3);
        v_s17 = _mm256_sub_pd(v_s1, v_s3);
        v_s18 = _mm256_add_pd(v_s4, v_s10);
        v_s19 = _mm256_sub_pd(v_s4, v_s10);
        v_s20 = _mm256_add_pd(v_s12, v_s14);
        v_s21 = _mm256_sub_pd(v_s12, v_s14);

        v_t0 = _mm256_mul_pd(v256_CRTM_12_2, v_s15);
        v_t1 = _mm256_mul_pd(v256_CRTM_12_2, v_s17);
        v_s22 = _mm256_add_pd(v_s5, v_t0);
        v_s23 = _mm256_add_pd(v_t1, v_s11);
        v_s24 = _mm256_add_pd(v_s7, v_s9);
        v_s25 = _mm256_sub_pd(v_s7, v_s9);
        v_s26 = _mm256_add_pd(v_s16, v_s24);
        v_s27 = _mm256_sub_pd(v_s16, v_s24);

        v_t2 = _mm256_mul_pd(v256_CRTM_12_1, v_s13);
        v_t3 = _mm256_mul_pd(v256_CRTM_12_2, v_s20);
        v_t4 = _mm256_mul_pd(v256_CRTM_12_2, v_s21);
        v_t5 = _mm256_mul_pd(v256_CRTM_12_1, v_s25);

        v_out0 = _mm256_add_pd(v_s20, v_s18);
        v_out1 = _mm256_add_pd(v_s22, v_t2);
        v_out2 = _mm256_add_pd(v_s23, v_t5);
        v_out3 = _mm256_add_pd(v_t4, v_s19);
        v_out4 = _mm256_mul_pd(v256_CRTM_12_1, v_s26);
        v_out5 = _mm256_sub_pd(v_s5, v_s15);
        v_out6 = _mm256_sub_pd(v_s17, v_s11);
        v_out7 = _mm256_sub_pd(v_s18, v_t3);
        v_out8 = _mm256_mul_pd(v256_CRTM_12_1, v_s27);
        v_out9 = _mm256_sub_pd(v_s22, v_t2);
        v_out10 = _mm256_sub_pd(v_s23, v_t5);
        v_out11 = _mm256_sub_pd(v_s19, v_s21);

        // Output point 1: X(0)
        STR_256_D(curr_out, v_out_stride, v_out0);
        // Output point 2: x(1) & Output point 3: x(2)
        curr_out = out + out_strides[1];
        STRI_2x256_D(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 4: x(3) & Output point 5: x(4)
        curr_out = out + out_strides[3];
        STRI_2x256_D(curr_out, v_out_stride, v_out3, v_out4);
        // Output point 6: x(5) & Output point 7: x(6)
        curr_out = out + out_strides[5];
        STRI_2x256_D(curr_out, v_out_stride, v_out5, v_out6);
        // Output point 8: x(7) & Output point 9: x(8)
        curr_out = out + out_strides[7];
        STRI_2x256_D(curr_out, v_out_stride, v_out7, v_out8);
        // Output point 10: x(9) & Output point 11: x(10)
        curr_out = out + out_strides[9];
        STRI_2x256_D(curr_out, v_out_stride, v_out9, v_out10);
        // Output point 12: x(11)
        curr_out = out + out_strides[11];
        STR_256_D(curr_out, v_out_stride, v_out11);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (n & 2)
    {
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
                v_in9, v_in10, v_in11;
        __m128d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
                v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
                v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27;
        __m128d v_t0, v_t1, v_t2, v_t3, v_t4, v_t5;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11;

        __m128d v128_CRTM_12_1 = _mm512_castpd512_pd128(v_CRTM_12_1);
        __m128d v128_CRTM_12_2 = _mm512_castpd512_pd128(v_CRTM_12_2);

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
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_128_D(curr_in, v_in_stride, v_in11);

        v_s0 = _mm_add_pd(v_in11, v_in1);
        v_s1 = _mm_sub_pd(v_in11, v_in1);
        v_s2 = _mm_add_pd(v_in5, v_in7);
        v_s3 = _mm_sub_pd(v_in5, v_in7);
        v_s4 = _mm_add_pd(v_in0, v_in6);
        v_s5 = _mm_sub_pd(v_in0, v_in6);
        v_s6 = _mm_add_pd(v_in10, v_in2);
        v_s7 = _mm_sub_pd(v_in10, v_in2);
        v_s8 = _mm_add_pd(v_in4, v_in8);
        v_s9 = _mm_sub_pd(v_in4, v_in8);
        v_s10 = _mm_add_pd(v_in9, v_in3);
        v_s11 = _mm_sub_pd(v_in9, v_in3);

        v_s12 = _mm_add_pd(v_s0, v_s2);
        v_s13 = _mm_sub_pd(v_s0, v_s2);
        v_s14 = _mm_add_pd(v_s6, v_s8);
        v_s15 = _mm_sub_pd(v_s6, v_s8);
        v_s16 = _mm_add_pd(v_s1, v_s3);
        v_s17 = _mm_sub_pd(v_s1, v_s3);
        v_s18 = _mm_add_pd(v_s4, v_s10);
        v_s19 = _mm_sub_pd(v_s4, v_s10);
        v_s20 = _mm_add_pd(v_s12, v_s14);
        v_s21 = _mm_sub_pd(v_s12, v_s14);

        v_t0 = _mm_mul_pd(v128_CRTM_12_2, v_s15);
        v_t1 = _mm_mul_pd(v128_CRTM_12_2, v_s17);
        v_s22 = _mm_add_pd(v_s5, v_t0);
        v_s23 = _mm_add_pd(v_t1, v_s11);
        v_s24 = _mm_add_pd(v_s7, v_s9);
        v_s25 = _mm_sub_pd(v_s7, v_s9);
        v_s26 = _mm_add_pd(v_s16, v_s24);
        v_s27 = _mm_sub_pd(v_s16, v_s24);

        v_t2 = _mm_mul_pd(v128_CRTM_12_1, v_s13);
        v_t3 = _mm_mul_pd(v128_CRTM_12_2, v_s20);
        v_t4 = _mm_mul_pd(v128_CRTM_12_2, v_s21);
        v_t5 = _mm_mul_pd(v128_CRTM_12_1, v_s25);

        v_out0 = _mm_add_pd(v_s20, v_s18);
        v_out1 = _mm_add_pd(v_s22, v_t2);
        v_out2 = _mm_add_pd(v_s23, v_t5);
        v_out3 = _mm_add_pd(v_t4, v_s19);
        v_out4 = _mm_mul_pd(v128_CRTM_12_1, v_s26);
        v_out5 = _mm_sub_pd(v_s5, v_s15);
        v_out6 = _mm_sub_pd(v_s17, v_s11);
        v_out7 = _mm_sub_pd(v_s18, v_t3);
        v_out8 = _mm_mul_pd(v128_CRTM_12_1, v_s27);
        v_out9 = _mm_sub_pd(v_s22, v_t2);
        v_out10 = _mm_sub_pd(v_s23, v_t5);
        v_out11 = _mm_sub_pd(v_s19, v_s21);

        // Output point 1: X(0)
        STR_128_D(curr_out, v_out_stride, v_out0);
        // Output point 2: x(1) & Output point 3: x(2)
        curr_out = out + out_strides[1];
        STRI_2x128_D(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 4: x(3) & Output point 5: x(4)
        curr_out = out + out_strides[3];
        STRI_2x128_D(curr_out, v_out_stride, v_out3, v_out4);
        // Output point 6: x(5) & Output point 7: x(6)
        curr_out = out + out_strides[5];
        STRI_2x128_D(curr_out, v_out_stride, v_out5, v_out6);
        // Output point 8: x(7) & Output point 9: x(8)
        curr_out = out + out_strides[7];
        STRI_2x128_D(curr_out, v_out_stride, v_out7, v_out8);
        // Output point 10: x(9) & Output point 11: x(10)
        curr_out = out + out_strides[9];
        STRI_2x128_D(curr_out, v_out_stride, v_out9, v_out10);
        // Output point 12: x(11)
        curr_out = out + out_strides[11];
        STR_128_D(curr_out, v_out_stride, v_out11);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        DOUBLE in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11;
        DOUBLE s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14,
               s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26, s27;
        DOUBLE t0, t1, t2, t3, t4, t5;

        in0 = *in;                 // Input point 1: x(0)
        in1 = in[in_strides[1]];   // Input point 2: x(1)
        in2 = in[in_strides[2]];   // Input point 3: x(2)
        in3 = in[in_strides[3]];   // Input point 4: x(3)
        in4 = in[in_strides[4]];   // Input point 5: x(4)
        in5 = in[in_strides[5]];   // Input point 6: x(5)
        in6 = in[in_strides[6]];   // Input point 7: x(6)
        in7 = in[in_strides[7]];   // Input point 8: x(7)
        in8 = in[in_strides[8]];   // Input point 9: x(8)
        in9 = in[in_strides[9]];   // Input point 10: x(9)
        in10 = in[in_strides[10]]; // Input point 11: x(10)
        in11 = in[in_strides[11]]; // Input point 12: x(11)

        s0 = in11 + in1;
        s1 = in11 - in1;
        s2 = in5 + in7;
        s3 = in5 - in7;
        s4 = in0 + in6;
        s5 = in0 - in6;
        s6 = in10 + in2;
        s7 = in10 - in2;
        s8 = in4 + in8;
        s9 = in4 - in8;
        s10 = in9 + in3;
        s11 = in9 - in3;

        s12 = s0 + s2;
        s13 = s0 - s2;
        s14 = s6 + s8;
        s15 = s6 - s8;
        s16 = s1 + s3;
        s17 = s1 - s3;
        s18 = s4 + s10;
        s19 = s4 - s10;
        s20 = s12 + s14;
        s21 = s12 - s14;

        t0 = CRTM_12_2 * s15;
        t1 = CRTM_12_2 * s17;
        s22 = s5 + t0;
        s23 = t1 + s11;
        s24 = s7 + s9;
        s25 = s7 - s9;
        s26 = s16 + s24;
        s27 = s16 - s24;

        t2 = CRTM_12_1 * s13;
        t3 = CRTM_12_2 * s20;
        t4 = CRTM_12_2 * s21;
        t5 = CRTM_12_1 * s25;

        *out = s20 + s18;                      // output pt 1: X(0)
        out[out_strides[1]] = s22 + t2;        // output pt 2: X(1)
        out[out_strides[2]] = s23 + t5;        // output pt 2: X(1)
        out[out_strides[3]] = t4 + s19;        // output pt 4: X(3)
        out[out_strides[4]] = CRTM_12_1 * s26; // output pt 5: X(4)
        out[out_strides[5]] = s5 - s15;        // output pt 6: X(5)
        out[out_strides[6]] = s17 - s11;       // output pt 7: X(6)
        out[out_strides[7]] = s18 - t3;        // output pt 8: X(7)
        out[out_strides[8]] = CRTM_12_1 * s27; // output pt 9: X(8)
        out[out_strides[9]] = s22 - t2;        // output pt 10: X(9)
        out[out_strides[10]] = s23 - t5;       // output pt 11: X(10)
        out[out_strides[11]] = s19 - s21;      // output pt 12: X(11)
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hc_rfft12avx512_fp64_bwd(VOID *in_real, VOID *in_imag,
                                       VOID *out_real, VOID *out_imag, INTP n,
                                       aoclfftz_strides_t *strides, VOID *twd,
                                       UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_12_1 = 1.732050807568877293527446341505872366942805254;
    const DOUBLE CRTM_12_2 = 2.000000000000000000000000000000000000000000000;

    DOUBLE *in = (DOUBLE *)in_real;
    DOUBLE *out = (DOUBLE *)out_real;
#ifdef VOLATILE_DTRIDE_ARRAY
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
    INTP N = n >> 3;

    __m512d v_CRTM_12_1 = _mm512_set1_pd(CRTM_12_1);
    __m512d v_CRTM_12_2 = _mm512_set1_pd(CRTM_12_2);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m512d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
                v_in9, v_in10, v_in11;
        __m512d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
                v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
                v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26;
        __m512d v_t0, v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8;
        __m512d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_512_D(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x512_D(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x512_D(curr_in, v_in_stride, v_in3, v_in4);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDRI_2x512_D(curr_in, v_in_stride, v_in5, v_in6);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in + in_strides[7];
        LDRI_2x512_D(curr_in, v_in_stride, v_in7, v_in8);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDRI_2x512_D(curr_in, v_in_stride, v_in9, v_in10);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_512_D(curr_in, v_in_stride, v_in11);

        v_s0 = _mm512_add_pd(v_in0, v_in11);
        v_s1 = _mm512_sub_pd(v_in0, v_in11);
        v_s2 = _mm512_add_pd(v_in1, v_in9);
        v_s3 = _mm512_sub_pd(v_in1, v_in9);
        v_s4 = _mm512_add_pd(v_in3, v_in7);
        v_s5 = _mm512_sub_pd(v_in3, v_in7);
        v_s6 = _mm512_add_pd(v_in4, v_in8);
        v_s7 = _mm512_sub_pd(v_in4, v_in8);
        v_s8 = _mm512_add_pd(v_in10, v_in2);
        v_s9 = _mm512_sub_pd(v_in10, v_in2);

        v_t0 = _mm512_mul_pd(v_CRTM_12_2, v_in5);
        v_t1 = _mm512_mul_pd(v_CRTM_12_2, v_in6);

        v_s10 = _mm512_add_pd(v_s2, v_s4);
        v_s11 = _mm512_add_pd(v_s0, v_t0);
        v_s12 = _mm512_sub_pd(v_s0, v_t0);
        v_s13 = _mm512_mul_pd(v_CRTM_12_2, v_s10);

        v_s14 = _mm512_add_pd(v_s3, v_s6);
        v_s15 = _mm512_sub_pd(v_s3, v_s6);
        v_t2 = _mm512_mul_pd(v_CRTM_12_1, v_s14);
        v_t3 = _mm512_mul_pd(v_CRTM_12_1, v_s15);

        v_s16 = _mm512_add_pd(v_s9, v_s7);
        v_s17 = _mm512_sub_pd(v_s9, v_s7);
        v_t4 = _mm512_mul_pd(v_CRTM_12_1, v_s16);
        v_t5 = _mm512_mul_pd(v_CRTM_12_1, v_s17);

        v_s18 = _mm512_add_pd(v_s5, v_s8);
        v_s19 = _mm512_sub_pd(v_s5, v_s8);
        v_s20 = _mm512_sub_pd(v_s2, v_s4);
        v_t6 = _mm512_mul_pd(v_CRTM_12_2, v_s18);
        v_t7 = _mm512_mul_pd(v_CRTM_12_2, v_s19);
        v_t8 = _mm512_mul_pd(v_CRTM_12_2, v_s20);

        v_s21 = _mm512_sub_pd(v_s1, v_t1);
        v_s22 = _mm512_add_pd(v_s1, v_t1);
        v_s23 = _mm512_add_pd(v_s19, v_s21);
        v_s24 = _mm512_add_pd(v_s22, v_s18);
        v_s25 = _mm512_sub_pd(v_s11, v_s10);
        v_s26 = _mm512_add_pd(v_s12, v_s20);

        v_out0 = _mm512_add_pd(v_s11, v_s13);
        v_out1 = _mm512_add_pd(v_t3, v_s23);
        v_out2 = _mm512_add_pd(v_t5, v_s26);
        v_out3 = _mm512_sub_pd(v_s22, v_t6);
        v_out4 = _mm512_add_pd(v_s25, v_t4);
        v_out5 = _mm512_sub_pd(v_s23, v_t3);
        v_out6 = _mm512_sub_pd(v_s12, v_t8);
        v_out7 = _mm512_sub_pd(v_s24, v_t2);
        v_out8 = _mm512_sub_pd(v_s25, v_t4);
        v_out9 = _mm512_sub_pd(v_s21, v_t7);
        v_out10 = _mm512_sub_pd(v_s26, v_t5);
        v_out11 = _mm512_add_pd(v_s24, v_t2);

        // Output point 1: x(0)
        STR_512_D(curr_out, v_out_stride, v_out0);
        // Output point 2: x(1)
        curr_out = out + out_strides[1];
        STR_512_D(curr_out, v_out_stride, v_out1);
        // Output point 3: x(2)
        curr_out = out + out_strides[2];
        STR_512_D(curr_out, v_out_stride, v_out2);
        // Output point 4: x(3)
        curr_out = out + out_strides[3];
        STR_512_D(curr_out, v_out_stride, v_out3);
        // Output point 5: x(4)
        curr_out = out + out_strides[4];
        STR_512_D(curr_out, v_out_stride, v_out4);
        // Output point 6: x(5)
        curr_out = out + out_strides[5];
        STR_512_D(curr_out, v_out_stride, v_out5);
        // Output point 7: x(6)
        curr_out = out + out_strides[6];
        STR_512_D(curr_out, v_out_stride, v_out6);
        // Output point 8: x(7)
        curr_out = out + out_strides[7];
        STR_512_D(curr_out, v_out_stride, v_out7);
        // Output point 9: x(8)
        curr_out = out + out_strides[8];
        STR_512_D(curr_out, v_out_stride, v_out8);
        // Output point 10: x(9)
        curr_out = out + out_strides[9];
        STR_512_D(curr_out, v_out_stride, v_out9);
        // Output point 11: x(10)
        curr_out = out + out_strides[10];
        STR_512_D(curr_out, v_out_stride, v_out10);
        // Output point 12: x(11)
        curr_out = out + out_strides[11];
        STR_512_D(curr_out, v_out_stride, v_out11);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (n & 4)
    {
        __m256d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
                v_in9, v_in10, v_in11;
        __m256d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
                v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
                v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26;
        __m256d v_t0, v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11;

        curr_in = in;
        curr_out = out;

        __m256d v256_CRTM_12_1 = _mm512_castpd512_pd256(v_CRTM_12_1);
        __m256d v256_CRTM_12_2 = _mm512_castpd512_pd256(v_CRTM_12_2);

        // Input point 1: x(0)
        LDR_256_D(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x256_D(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x256_D(curr_in, v_in_stride, v_in3, v_in4);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDRI_2x256_D(curr_in, v_in_stride, v_in5, v_in6);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in + in_strides[7];
        LDRI_2x256_D(curr_in, v_in_stride, v_in7, v_in8);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDRI_2x256_D(curr_in, v_in_stride, v_in9, v_in10);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_256_D(curr_in, v_in_stride, v_in11);

        v_s0 = _mm256_add_pd(v_in0, v_in11);
        v_s1 = _mm256_sub_pd(v_in0, v_in11);
        v_s2 = _mm256_add_pd(v_in1, v_in9);
        v_s3 = _mm256_sub_pd(v_in1, v_in9);
        v_s4 = _mm256_add_pd(v_in3, v_in7);
        v_s5 = _mm256_sub_pd(v_in3, v_in7);
        v_s6 = _mm256_add_pd(v_in4, v_in8);
        v_s7 = _mm256_sub_pd(v_in4, v_in8);
        v_s8 = _mm256_add_pd(v_in10, v_in2);
        v_s9 = _mm256_sub_pd(v_in10, v_in2);

        v_t0 = _mm256_mul_pd(v256_CRTM_12_2, v_in5);
        v_t1 = _mm256_mul_pd(v256_CRTM_12_2, v_in6);

        v_s10 = _mm256_add_pd(v_s2, v_s4);
        v_s11 = _mm256_add_pd(v_s0, v_t0);
        v_s12 = _mm256_sub_pd(v_s0, v_t0);
        v_s13 = _mm256_mul_pd(v256_CRTM_12_2, v_s10);

        v_s14 = _mm256_add_pd(v_s3, v_s6);
        v_s15 = _mm256_sub_pd(v_s3, v_s6);
        v_t2 = _mm256_mul_pd(v256_CRTM_12_1, v_s14);
        v_t3 = _mm256_mul_pd(v256_CRTM_12_1, v_s15);

        v_s16 = _mm256_add_pd(v_s9, v_s7);
        v_s17 = _mm256_sub_pd(v_s9, v_s7);
        v_t4 = _mm256_mul_pd(v256_CRTM_12_1, v_s16);
        v_t5 = _mm256_mul_pd(v256_CRTM_12_1, v_s17);

        v_s18 = _mm256_add_pd(v_s5, v_s8);
        v_s19 = _mm256_sub_pd(v_s5, v_s8);
        v_s20 = _mm256_sub_pd(v_s2, v_s4);
        v_t6 = _mm256_mul_pd(v256_CRTM_12_2, v_s18);
        v_t7 = _mm256_mul_pd(v256_CRTM_12_2, v_s19);
        v_t8 = _mm256_mul_pd(v256_CRTM_12_2, v_s20);

        v_s21 = _mm256_sub_pd(v_s1, v_t1);
        v_s22 = _mm256_add_pd(v_s1, v_t1);
        v_s23 = _mm256_add_pd(v_s19, v_s21);
        v_s24 = _mm256_add_pd(v_s22, v_s18);
        v_s25 = _mm256_sub_pd(v_s11, v_s10);
        v_s26 = _mm256_add_pd(v_s12, v_s20);

        v_out0 = _mm256_add_pd(v_s11, v_s13);
        v_out1 = _mm256_add_pd(v_t3, v_s23);
        v_out2 = _mm256_add_pd(v_t5, v_s26);
        v_out3 = _mm256_sub_pd(v_s22, v_t6);
        v_out4 = _mm256_add_pd(v_s25, v_t4);
        v_out5 = _mm256_sub_pd(v_s23, v_t3);
        v_out6 = _mm256_sub_pd(v_s12, v_t8);
        v_out7 = _mm256_sub_pd(v_s24, v_t2);
        v_out8 = _mm256_sub_pd(v_s25, v_t4);
        v_out9 = _mm256_sub_pd(v_s21, v_t7);
        v_out10 = _mm256_sub_pd(v_s26, v_t5);
        v_out11 = _mm256_add_pd(v_s24, v_t2);

        // Output point 1: x(0)
        STR_256_D(curr_out, v_out_stride, v_out0);
        // Output point 2: x(1)
        curr_out = out + out_strides[1];
        STR_256_D(curr_out, v_out_stride, v_out1);
        // Output point 3: x(2)
        curr_out = out + out_strides[2];
        STR_256_D(curr_out, v_out_stride, v_out2);
        // Output point 4: x(3)
        curr_out = out + out_strides[3];
        STR_256_D(curr_out, v_out_stride, v_out3);
        // Output point 5: x(4)
        curr_out = out + out_strides[4];
        STR_256_D(curr_out, v_out_stride, v_out4);
        // Output point 6: x(5)
        curr_out = out + out_strides[5];
        STR_256_D(curr_out, v_out_stride, v_out5);
        // Output point 7: x(6)
        curr_out = out + out_strides[6];
        STR_256_D(curr_out, v_out_stride, v_out6);
        // Output point 8: x(7)
        curr_out = out + out_strides[7];
        STR_256_D(curr_out, v_out_stride, v_out7);
        // Output point 9: x(8)
        curr_out = out + out_strides[8];
        STR_256_D(curr_out, v_out_stride, v_out8);
        // Output point 10: x(9)
        curr_out = out + out_strides[9];
        STR_256_D(curr_out, v_out_stride, v_out9);
        // Output point 11: x(10)
        curr_out = out + out_strides[10];
        STR_256_D(curr_out, v_out_stride, v_out10);
        // Output point 12: x(11)
        curr_out = out + out_strides[11];
        STR_256_D(curr_out, v_out_stride, v_out11);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (n & 2)
    {
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
                v_in9, v_in10, v_in11;
        __m128d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
                v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
                v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26;
        __m128d v_t0, v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11;

        curr_in = in;
        curr_out = out;

        __m128d v128_CRTM_12_1 = _mm512_castpd512_pd128(v_CRTM_12_1);
        __m128d v128_CRTM_12_2 = _mm512_castpd512_pd128(v_CRTM_12_2);

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_stride, v_in0);
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
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_128_D(curr_in, v_in_stride, v_in11);

        v_s0 = _mm_add_pd(v_in0, v_in11);
        v_s1 = _mm_sub_pd(v_in0, v_in11);
        v_s2 = _mm_add_pd(v_in1, v_in9);
        v_s3 = _mm_sub_pd(v_in1, v_in9);
        v_s4 = _mm_add_pd(v_in3, v_in7);
        v_s5 = _mm_sub_pd(v_in3, v_in7);
        v_s6 = _mm_add_pd(v_in4, v_in8);
        v_s7 = _mm_sub_pd(v_in4, v_in8);
        v_s8 = _mm_add_pd(v_in10, v_in2);
        v_s9 = _mm_sub_pd(v_in10, v_in2);

        v_t0 = _mm_mul_pd(v128_CRTM_12_2, v_in5);
        v_t1 = _mm_mul_pd(v128_CRTM_12_2, v_in6);

        v_s10 = _mm_add_pd(v_s2, v_s4);
        v_s11 = _mm_add_pd(v_s0, v_t0);
        v_s12 = _mm_sub_pd(v_s0, v_t0);
        v_s13 = _mm_mul_pd(v128_CRTM_12_2, v_s10);

        v_s14 = _mm_add_pd(v_s3, v_s6);
        v_s15 = _mm_sub_pd(v_s3, v_s6);
        v_t2 = _mm_mul_pd(v128_CRTM_12_1, v_s14);
        v_t3 = _mm_mul_pd(v128_CRTM_12_1, v_s15);

        v_s16 = _mm_add_pd(v_s9, v_s7);
        v_s17 = _mm_sub_pd(v_s9, v_s7);
        v_t4 = _mm_mul_pd(v128_CRTM_12_1, v_s16);
        v_t5 = _mm_mul_pd(v128_CRTM_12_1, v_s17);

        v_s18 = _mm_add_pd(v_s5, v_s8);
        v_s19 = _mm_sub_pd(v_s5, v_s8);
        v_s20 = _mm_sub_pd(v_s2, v_s4);
        v_t6 = _mm_mul_pd(v128_CRTM_12_2, v_s18);
        v_t7 = _mm_mul_pd(v128_CRTM_12_2, v_s19);
        v_t8 = _mm_mul_pd(v128_CRTM_12_2, v_s20);

        v_s21 = _mm_sub_pd(v_s1, v_t1);
        v_s22 = _mm_add_pd(v_s1, v_t1);
        v_s23 = _mm_add_pd(v_s19, v_s21);
        v_s24 = _mm_add_pd(v_s22, v_s18);
        v_s25 = _mm_sub_pd(v_s11, v_s10);
        v_s26 = _mm_add_pd(v_s12, v_s20);

        v_out0 = _mm_add_pd(v_s11, v_s13);
        v_out1 = _mm_add_pd(v_t3, v_s23);
        v_out2 = _mm_add_pd(v_t5, v_s26);
        v_out3 = _mm_sub_pd(v_s22, v_t6);
        v_out4 = _mm_add_pd(v_s25, v_t4);
        v_out5 = _mm_sub_pd(v_s23, v_t3);
        v_out6 = _mm_sub_pd(v_s12, v_t8);
        v_out7 = _mm_sub_pd(v_s24, v_t2);
        v_out8 = _mm_sub_pd(v_s25, v_t4);
        v_out9 = _mm_sub_pd(v_s21, v_t7);
        v_out10 = _mm_sub_pd(v_s26, v_t5);
        v_out11 = _mm_add_pd(v_s24, v_t2);

        // Output point 1: x(0)
        STR_128_D(curr_out, v_out_stride, v_out0);
        // Output point 2: x(1)
        curr_out = out + out_strides[1];
        STR_128_D(curr_out, v_out_stride, v_out1);
        // Output point 2: x(1)
        curr_out = out + out_strides[2];
        STR_128_D(curr_out, v_out_stride, v_out2);
        // Output point 4: x(3)
        curr_out = out + out_strides[3];
        STR_128_D(curr_out, v_out_stride, v_out3);
        // Output point 5: x(4)
        curr_out = out + out_strides[4];
        STR_128_D(curr_out, v_out_stride, v_out4);
        // Output point 6: x(5)
        curr_out = out + out_strides[5];
        STR_128_D(curr_out, v_out_stride, v_out5);
        // Output point 7: x(6)
        curr_out = out + out_strides[6];
        STR_128_D(curr_out, v_out_stride, v_out6);
        // Output point 8: x(7)
        curr_out = out + out_strides[7];
        STR_128_D(curr_out, v_out_stride, v_out7);
        // Output point 9: x(8)
        curr_out = out + out_strides[8];
        STR_128_D(curr_out, v_out_stride, v_out8);
        // Output point 10: x(9)
        curr_out = out + out_strides[9];
        STR_128_D(curr_out, v_out_stride, v_out9);
        // Output point 11: x(10)
        curr_out = out + out_strides[10];
        STR_128_D(curr_out, v_out_stride, v_out10);
        // Output point 12: x(11)
        curr_out = out + out_strides[11];
        STR_128_D(curr_out, v_out_stride, v_out11);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        DOUBLE in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11;
        DOUBLE s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14,
               s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25, s26;
        DOUBLE t0, t1, t2, t3, t4, t5, t6, t7, t8;

        in0 = *in;                 // Input point 1: X(0)
        in1 = in[in_strides[1]];   // Input point 2: X(1)
        in2 = in[in_strides[2]];   // Input point 3: X(2)
        in3 = in[in_strides[3]];   // Input point 4: X(3)
        in4 = in[in_strides[4]];   // Input point 5: X(4)
        in5 = in[in_strides[5]];   // Input point 6: X(5)
        in6 = in[in_strides[6]];   // Input point 7: X(6)
        in7 = in[in_strides[7]];   // Input point 8: X(7)
        in8 = in[in_strides[8]];   // Input point 9: X(8)
        in9 = in[in_strides[9]];   // Input point 10: X(9)
        in10 = in[in_strides[10]]; // Input point 11: X(10)
        in11 = in[in_strides[11]]; // Input point 12: X(11)

        s0 = in0 + in11;
        s1 = in0 - in11;
        s2 = in1 + in9;
        s3 = in1 - in9;
        s4 = in3 + in7;
        s5 = in3 - in7;
        s6 = in4 + in8;
        s7 = in4 - in8;
        s8 = in10 + in2;
        s9 = in10 - in2;

        t0 = CRTM_12_2 * in5;
        t1 = CRTM_12_2 * in6;

        s10 = s2 + s4;
        s11 = s0 + t0;
        s12 = s0 - t0;
        s13 = CRTM_12_2 * s10;

        s14 = s3 + s6;
        s15 = s3 - s6;
        t2 = CRTM_12_1 * s14;
        t3 = CRTM_12_1 * s15;

        s16 = s9 + s7;
        s17 = s9 - s7;
        t4 = CRTM_12_1 * s16;
        t5 = CRTM_12_1 * s17;

        s18 = s5 + s8;
        s19 = s5 - s8;
        s20 = s2 - s4;
        t6 = CRTM_12_2 * s18;
        t7 = CRTM_12_2 * s19;
        t8 = CRTM_12_2 * s20;

        s21 = s1 - t1;
        s22 = s1 + t1;
        s23 = s19 + s21;
        s24 = s22 + s18;
        s25 = s11 - s10;
        s26 = s12 + s20;

        *out = s11 + s13;                // output pt 1: x(0)
        out[out_strides[1]] = t3 + s23;  // output pt 2: x(1)
        out[out_strides[2]] = t5 + s26;  // output pt 3: x(2)
        out[out_strides[3]] = s22 - t6;  // output pt 4: x(3)
        out[out_strides[4]] = s25 + t4;  // output pt 5: x(4)
        out[out_strides[5]] = s23 - t3;  // output pt 6: x(5)
        out[out_strides[6]] = s12 - t8;  // output pt 7: x(6)
        out[out_strides[7]] = s24 - t2;  // output pt 8: x(7)
        out[out_strides[8]] = s25 - t4;  // output pt 9: x(8)
        out[out_strides[9]] = s21 - t7;  // output pt 10: x(9)
        out[out_strides[10]] = s26 - t5; // output pt 11: x(10)
        out[out_strides[11]] = s24 + t2; // output pt 12: x(11)
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hc_rfft12avx512(UINT8 precision, UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hc_rfft12avx512_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft12avx512_fp64_fwd;
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
            return r2hc_rfft12avx512_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft12avx512_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
