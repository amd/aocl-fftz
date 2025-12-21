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

/** @file rfft6avx256.c
 *
 *  @brief Radix-6 r2hc_fused Real-FFT kernel with with AVX-256 operations using
 *  x86 SIMD intrinsics.
 *
 *  This file contains the DIT radix-6 real-to-halfcomplex fused of two
 *  different implementations (Standard DFT and Shifted DFT that differs in DFT
 *  weight matrix) using x86 SIMD intrinsics for single-precision and
 *  double-precision inputs
 *
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/r2hc_simd_avx256.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                {{{0,  8, 26, 152,  94, 25},
                                                  {0, 10, 26, 152, 109, 25}},
                                                 {{0,  8, 26,  76,  10, 25},
                                                  {0, 10, 26,  76,  10, 25}}};

ops_cycles_t get_ops_cnt_r2hcf_rfft6avx256(UINT8 precision, UINT8 direction)
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

static VOID r2hcf_rfft6avx256_fp32_fwd(VOID *in_real, VOID *in_imag,
                                       VOID *out_real, VOID *out_imag, INTP n,
                                       aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_6_1 = 0.500000000000000000000000000000000000000000000f;
    const FLOAT CRTM_6_2 = 0.866025403784438646763723170752936183471402627f;

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

    __m256 v_CRTM_6_1 = _mm256_broadcast_ss(&CRTM_6_1);
    __m256 v_CRTM_6_2 = _mm256_broadcast_ss(&CRTM_6_2);

    for (cnt = 0; cnt < N; cnt++)
    {
        // Standard DFT
        __m256 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5;
        __m256 av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
               av_s9;
        __m256 av_t0, av_t1;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11;

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

        av_s0 = _mm256_add_ps(av_in0, av_in3);
        av_s1 = _mm256_sub_ps(av_in0, av_in3);
        av_s2 = _mm256_add_ps(av_in1, av_in2);
        av_s3 = _mm256_sub_ps(av_in2, av_in1);

        av_s4 = _mm256_add_ps(av_in4, av_in5);
        av_s5 = _mm256_sub_ps(av_in5, av_in4);
        av_s6 = _mm256_add_ps(av_s2, av_s4);
        av_s7 = _mm256_sub_ps(av_s5, av_s3);

        av_t0 = _mm256_mul_ps(v_CRTM_6_1, av_s7);
        av_s8 = _mm256_sub_ps(av_s4, av_s2);

        av_t1 = _mm256_mul_ps(v_CRTM_6_1, av_s6);
        av_s9 = _mm256_add_ps(av_s3, av_s5);

        // Output pt 1: X(0)
        v_out0 = _mm256_add_ps(av_s0, av_s6);
        STR_256_S(curr_out, v_out_stride, v_out0);
        // Output pt 4: X(3) & Output pt 5: X(4)
        curr_out = out + out_strides[3];
        v_out3 = _mm256_add_ps(av_s1, av_t0);
        v_out4 = _mm256_mul_ps(v_CRTM_6_2, av_s8);
        STRI_2x256_S(curr_out, v_out_stride, v_out3, v_out4);
        // Output pt 8: X(7) & Output pt 9: X(8)
        curr_out = out + out_strides[7];
        v_out7 = _mm256_sub_ps(av_s0, av_t1);
        v_out8 = _mm256_mul_ps(v_CRTM_6_2, av_s9);
        STRI_2x256_S(curr_out, v_out_stride, v_out7, v_out8);
        // Output pt 12: X(11)
        curr_out = out + out_strides[11];
        v_out11 = _mm256_sub_ps(av_s1, av_s7);
        STR_256_S(curr_out, v_out_stride, v_out11);

        // Shifted DFT
        __m256 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5;
        __m256 bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5;
        __m256 bv_t0, bv_t1, bv_t2, bv_t3;

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

        bv_s0 = _mm256_sub_ps(bv_in1, bv_in5);
        bv_s1 = _mm256_add_ps(bv_in1, bv_in5);
        bv_s2 = _mm256_sub_ps(bv_in2, bv_in4);
        bv_s3 = _mm256_add_ps(bv_in2, bv_in4);

        bv_t0 = _mm256_mul_ps(v_CRTM_6_2, bv_s0);
        bv_t1 = _mm256_mul_ps(v_CRTM_6_1, bv_s1);
        bv_t2 = _mm256_mul_ps(v_CRTM_6_1, bv_s2);
        bv_t3 = _mm256_mul_ps(v_CRTM_6_2, bv_s3);

        bv_s4 = _mm256_add_ps(bv_in0, bv_t2);
        bv_s5 = _mm256_add_ps(bv_in3, bv_t1);

        // Output point 2: X(1) & Output point 3: X(2)
        curr_out = out + out_strides[1];
        v_out1 = _mm256_add_ps(bv_s4, bv_t0);
        v_out2 = NEGATE_256_S(_mm256_add_ps(bv_t3, bv_s5));
        STRI_2x256_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 6: X(5) & Output point 7: X(6)
        curr_out = out + out_strides[5];
        v_out5 = _mm256_sub_ps(bv_in0, bv_s2);
        v_out6 = _mm256_sub_ps(bv_in3, bv_s1);
        STRI_2x256_S(curr_out, v_out_stride, v_out5, v_out6);
        // Output point 10: X(9) & Output point 11: X(10)
        curr_out = out + out_strides[9];
        v_out9 = _mm256_sub_ps(bv_s4, bv_t0);
        v_out10 = _mm256_sub_ps(bv_t3, bv_s5);
        STRI_2x256_S(curr_out, v_out_stride, v_out9, v_out10);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_S)
    {
        // Standard DFT
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5;
        __m128 av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
               av_s9;
        __m128 av_t0, av_t1;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_6_1 = _mm256_castps256_ps128(v_CRTM_6_1);
        __m128 v128_CRTM_6_2 = _mm256_castps256_ps128(v_CRTM_6_2);

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

        av_s0 = _mm_add_ps(av_in0, av_in3);
        av_s1 = _mm_sub_ps(av_in0, av_in3);
        av_s2 = _mm_add_ps(av_in1, av_in2);
        av_s3 = _mm_sub_ps(av_in2, av_in1);

        av_s4 = _mm_add_ps(av_in4, av_in5);
        av_s5 = _mm_sub_ps(av_in5, av_in4);
        av_s6 = _mm_add_ps(av_s2, av_s4);
        av_s7 = _mm_sub_ps(av_s5, av_s3);

        av_t0 = _mm_mul_ps(v128_CRTM_6_1, av_s7);
        av_s8 = _mm_sub_ps(av_s4, av_s2);

        av_t1 = _mm_mul_ps(v128_CRTM_6_1, av_s6);
        av_s9 = _mm_add_ps(av_s3, av_s5);

        // Output pt 1: X(0)
        v_out0 = _mm_add_ps(av_s0, av_s6);
        STR_128_S(curr_out, v_out_stride, v_out0);
        // Output pt 4: X(3) & Output pt 5: X(4)
        curr_out = out + out_strides[3];
        v_out3 = _mm_add_ps(av_s1, av_t0);
        v_out4 = _mm_mul_ps(v128_CRTM_6_2, av_s8);
        STRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);
        // Output pt 8: X(7) & Output pt 9: X(8)
        curr_out = out + out_strides[7];
        v_out7 = _mm_sub_ps(av_s0, av_t1);
        v_out8 = _mm_mul_ps(v128_CRTM_6_2, av_s9);
        STRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);
        // Output pt 12: X(11)
        curr_out = out + out_strides[11];
        v_out11 = _mm_sub_ps(av_s1, av_s7);
        STR_128_S(curr_out, v_out_stride, v_out11);

        // Shifted DFT
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5;
        __m128 bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5;
        __m128 bv_t0, bv_t1, bv_t2, bv_t3;

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

        bv_s0 = _mm_sub_ps(bv_in1, bv_in5);
        bv_s1 = _mm_add_ps(bv_in1, bv_in5);
        bv_s2 = _mm_sub_ps(bv_in2, bv_in4);
        bv_s3 = _mm_add_ps(bv_in2, bv_in4);

        bv_t0 = _mm_mul_ps(v128_CRTM_6_2, bv_s0);
        bv_t1 = _mm_mul_ps(v128_CRTM_6_1, bv_s1);
        bv_t2 = _mm_mul_ps(v128_CRTM_6_1, bv_s2);
        bv_t3 = _mm_mul_ps(v128_CRTM_6_2, bv_s3);

        bv_s4 = _mm_add_ps(bv_in0, bv_t2);
        bv_s5 = _mm_add_ps(bv_in3, bv_t1);

        // Output point 2: X(1) & Output point 3: X(2)
        curr_out = out + out_strides[1];
        v_out1 = _mm_add_ps(bv_s4, bv_t0);
        v_out2 = NEGATE_128_S(_mm_add_ps(bv_t3, bv_s5));
        STRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 6: X(5) & Output point 7: X(6)
        curr_out = out + out_strides[5];
        v_out5 = _mm_sub_ps(bv_in0, bv_s2);
        v_out6 = _mm_sub_ps(bv_in3, bv_s1);
        STRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);
        // Output point 10: X(9) & Output point 11: X(10)
        curr_out = out + out_strides[9];
        v_out9 = _mm_sub_ps(bv_s4, bv_t0);
        v_out10 = _mm_sub_ps(bv_t3, bv_s5);
        STRI_2x128_S(curr_out, v_out_stride, v_out9, v_out10);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & 2)
    {
        // Standard DFT
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5;
        __m128 av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
               av_s9;
        __m128 av_t0, av_t1;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_6_1 = _mm256_castps256_ps128(v_CRTM_6_1);
        __m128 v128_CRTM_6_2 = _mm256_castps256_ps128(v_CRTM_6_2);

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

        av_s0 = _mm_add_ps(av_in0, av_in3);
        av_s1 = _mm_sub_ps(av_in0, av_in3);
        av_s2 = _mm_add_ps(av_in1, av_in2);
        av_s3 = _mm_sub_ps(av_in2, av_in1);

        av_s4 = _mm_add_ps(av_in4, av_in5);
        av_s5 = _mm_sub_ps(av_in5, av_in4);
        av_s6 = _mm_add_ps(av_s2, av_s4);
        av_s7 = _mm_sub_ps(av_s5, av_s3);

        av_t0 = _mm_mul_ps(v128_CRTM_6_1, av_s7);
        av_s8 = _mm_sub_ps(av_s4, av_s2);

        av_t1 = _mm_mul_ps(v128_CRTM_6_1, av_s6);
        av_s9 = _mm_add_ps(av_s3, av_s5);

        // Output pt 1: X(0)
        v_out0 = _mm_add_ps(av_s0, av_s6);
        STHR_128_S(curr_out, v_out_stride, v_out0);
        // Output pt 4: X(3) & Output pt 5: X(4)
        curr_out = out + out_strides[3];
        v_out3 = _mm_add_ps(av_s1, av_t0);
        v_out4 = _mm_mul_ps(v128_CRTM_6_2, av_s8);
        STHRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);
        // Output pt 8: X(7) & Output pt 9: X(8)
        curr_out = out + out_strides[7];
        v_out7 = _mm_sub_ps(av_s0, av_t1);
        v_out8 = _mm_mul_ps(v128_CRTM_6_2, av_s9);
        STHRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);
        // Output pt 12: X(11)
        curr_out = out + out_strides[11];
        v_out11 = _mm_sub_ps(av_s1, av_s7);
        STHR_128_S(curr_out, v_out_stride, v_out11);

        // Shifted DFT
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5;
        __m128 bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5;
        __m128 bv_t0, bv_t1, bv_t2, bv_t3;

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

        bv_s0 = _mm_sub_ps(bv_in1, bv_in5);
        bv_s1 = _mm_add_ps(bv_in1, bv_in5);
        bv_s2 = _mm_sub_ps(bv_in2, bv_in4);
        bv_s3 = _mm_add_ps(bv_in2, bv_in4);

        bv_t0 = _mm_mul_ps(v128_CRTM_6_2, bv_s0);
        bv_t1 = _mm_mul_ps(v128_CRTM_6_1, bv_s1);
        bv_t2 = _mm_mul_ps(v128_CRTM_6_1, bv_s2);
        bv_t3 = _mm_mul_ps(v128_CRTM_6_2, bv_s3);

        bv_s4 = _mm_add_ps(bv_in0, bv_t2);
        bv_s5 = _mm_add_ps(bv_in3, bv_t1);

        // Output point 2: X(1) & Output point 3: X(2)
        curr_out = out + out_strides[1];
        v_out1 = _mm_add_ps(bv_s4, bv_t0);
        v_out2 = NEGATE_128_S(_mm_add_ps(bv_t3, bv_s5));
        STHRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 6: X(5) & Output point 7: X(6)
        curr_out = out + out_strides[5];
        v_out5 = _mm_sub_ps(bv_in0, bv_s2);
        v_out6 = _mm_sub_ps(bv_in3, bv_s1);
        STHRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);
        // Output point 10: X(9) & Output point 11: X(10)
        curr_out = out + out_strides[9];
        v_out9 = _mm_sub_ps(bv_s4, bv_t0);
        v_out10 = _mm_sub_ps(bv_t3, bv_s5);
        STHRI_2x128_S(curr_out, v_out_stride, v_out9, v_out10);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        // Standard DFT
        FLOAT a_in0, a_in1, a_in2, a_in3, a_in4, a_in5;
        FLOAT a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_s9;
        FLOAT a_t0, a_t1;

        a_in0 = *in;                // Input point 1: x(0)
        a_in1 = in[in_strides[2]];  // Input point 3: x(2)
        a_in2 = in[in_strides[4]];  // Input point 5: x(4)
        a_in3 = in[in_strides[6]];  // Input point 7: x(6)
        a_in4 = in[in_strides[8]];  // Input point 9: x(8)
        a_in5 = in[in_strides[10]]; // Input point 11: x(10)

        a_s0 = a_in0 + a_in3;
        a_s1 = a_in0 - a_in3;
        a_s2 = a_in1 + a_in2;
        a_s3 = a_in1 - a_in2;
        a_s4 = a_in5 + a_in4;
        a_s5 = a_in5 - a_in4;
        a_s6 = a_s2 + a_s4;
        a_s7 = a_s3 + a_s5;

        a_t0 = CRTM_6_1 * a_s7;
        a_s8 = a_s4 - a_s2;
        a_t1 = CRTM_6_1 * a_s6;
        a_s9 = a_s5 - a_s3;

        *out = a_s0 + a_s6;                    // Output pt 1: X(0)
        out[out_strides[3]] = a_s1 + a_t0;     // Output pt 4: X(3)
        out[out_strides[4]] = CRTM_6_2 * a_s8; // Output pt 5: X(4)
        out[out_strides[7]] = a_s0 - a_t1;     // Output pt 8: X(7)
        out[out_strides[8]] = CRTM_6_2 * a_s9; // Output pt 9: X(8)
        out[out_strides[11]] = a_s1 - a_s7;    // Output pt 12: X(11)

        // Shifted DFT
        FLOAT b_in0, b_in1, b_in2, b_in3, b_in4, b_in5;
        FLOAT b_s0, b_s1, b_s2, b_s3, b_s4, b_s5;
        FLOAT b_t0, b_t1, b_t2, b_t3;

        b_in0 = in[in_strides[1]];  // Input point 2: x(1)
        b_in1 = in[in_strides[3]];  // Input point 4: x(3)
        b_in2 = in[in_strides[5]];  // Input point 6: x(5)
        b_in3 = in[in_strides[7]];  // Input point 8: x(7)
        b_in4 = in[in_strides[9]];  // Input point 10: x(9)
        b_in5 = in[in_strides[11]]; // Input point 12: x(11)

        b_s0 = b_in1 - b_in5;
        b_s1 = b_in1 + b_in5;
        b_s2 = b_in2 - b_in4;
        b_s3 = b_in2 + b_in4;

        b_t0 = CRTM_6_2 * b_s0;
        b_t1 = CRTM_6_1 * b_s1;
        b_t2 = CRTM_6_1 * b_s2;
        b_t3 = CRTM_6_2 * b_s3;

        b_s4 = b_in0 + b_t2;
        b_s5 = b_in3 + b_t1;

        out[out_strides[1]] = b_s4 + b_t0;  // Output pt 2: X(1)
        out[out_strides[2]] = -b_t3 - b_s5; // Output pt 3: X(2)
        out[out_strides[5]] = b_in0 - b_s2; // Output pt 6: X(5)
        out[out_strides[6]] = b_in3 - b_s1; // Output pt 7: X(6)
        out[out_strides[9]] = b_s4 - b_t0;  // Output pt 10: X(9)
        out[out_strides[10]] = b_t3 - b_s5; // Output pt 11: X(10)
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hcf_rfft6avx256_fp32_bwd(VOID *in_real, VOID *in_imag,
                                       VOID *out_real, VOID *out_imag, INTP n,
                                       aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_6_1 = 1.732050807568877293527446341505872366942805253f;
    const FLOAT CRTM_6_2 = 2.000000000000000000000000000000000000000000000f;

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

    __m256 v_CRTM_6_1 = _mm256_broadcast_ss(&CRTM_6_1);
    __m256 v_CRTM_6_2 = _mm256_broadcast_ss(&CRTM_6_2);

    for (cnt = 0; cnt < N; cnt++)
    {
        // Standard DFT
        __m256 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5;
        __m256 av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7;
        __m256 av_t0, av_t1, av_t2, av_t3;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_256_S(curr_in, v_in_stride, av_in0);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDRI_2x256_S(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDRI_2x256_S(curr_in, v_in_stride, av_in3, av_in4);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_256_S(curr_in, v_in_stride, av_in5);

        av_s0 = _mm256_add_ps(av_in0, av_in5);
        av_s1 = _mm256_sub_ps(av_in0, av_in5);
        av_s2 = _mm256_add_ps(av_in1, av_in3);

        av_s3 = _mm256_sub_ps(av_in1, av_in3);
        av_s4 = _mm256_add_ps(av_in2, av_in4);
        av_s5 = _mm256_sub_ps(av_in2, av_in4);

        av_t0 = _mm256_mul_ps(v_CRTM_6_1, av_s4);
        av_t1 = _mm256_mul_ps(v_CRTM_6_1, av_s5);
        av_s6 = _mm256_add_ps(av_s1, av_s3);

        av_s7 = _mm256_sub_ps(av_s0, av_s2);
        av_t2 = _mm256_mul_ps(v_CRTM_6_2, av_s2);
        av_t3 = _mm256_mul_ps(v_CRTM_6_2, av_s3);

        // Output pt 1: X(0)
        v_out0 = _mm256_add_ps(av_s0, av_t2);
        STR_256_S(curr_out, v_out_stride, v_out0);
        // Output point 3: X(2)--> type-0
        curr_out = out + out_strides[2];
        v_out2 = _mm256_sub_ps(av_s6, av_t0);
        STR_256_S(curr_out, v_out_stride, v_out2);
        // Output pt 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm256_sub_ps(av_s7, av_t1);
        STR_256_S(curr_out, v_out_stride, v_out4);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm256_sub_ps(av_s1, av_t3);
        STR_256_S(curr_out, v_out_stride, v_out6);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm256_add_ps(av_s7, av_t1);
        STR_256_S(curr_out, v_out_stride, v_out8);
        // Output pt 11: X(10)
        curr_out = out + out_strides[10];
        v_out10 = _mm256_add_ps(av_s6, av_t0);
        STR_256_S(curr_out, v_out_stride, v_out10);

        // Shifted DFT
        __m256 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5;
        __m256 bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5;
        __m256 bv_t0, bv_t1, bv_t2, bv_t3, bv_t4, bv_t5;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x256_S(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)--> type-2
        curr_in = in + in_strides[5];
        LDRI_2x256_S(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDRI_2x256_S(curr_in, v_in_stride, bv_in4, bv_in5);

        bv_s0 = _mm256_add_ps(bv_in0, bv_in4);
        bv_s1 = _mm256_sub_ps(bv_in0, bv_in4);
        bv_t0 = _mm256_mul_ps(v_CRTM_6_1, bv_s1);

        bv_s2 = _mm256_add_ps(bv_in1, bv_in5);
        bv_s3 = _mm256_sub_ps(bv_in5, bv_in1);
        bv_t1 = _mm256_mul_ps(v_CRTM_6_1, bv_s3);

        bv_t2 = _mm256_mul_ps(v_CRTM_6_2, bv_s0);
        bv_t3 = _mm256_mul_ps(v_CRTM_6_2, bv_s2);
        bv_t4 = _mm256_mul_ps(v_CRTM_6_2, bv_in2);

        bv_t5 = _mm256_mul_ps(v_CRTM_6_2, bv_in3);
        bv_s4 = _mm256_add_ps(bv_s2, bv_t5);
        bv_s5 = _mm256_sub_ps(bv_s0, bv_t4);

        // Output pt 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm256_add_ps(bv_t4, bv_t2);
        STR_256_S(curr_out, v_out_stride, v_out1);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm256_sub_ps(bv_t0, bv_s4);
        STR_256_S(curr_out, v_out_stride, v_out3);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm256_add_ps(bv_s5, bv_t1);
        STR_256_S(curr_out, v_out_stride, v_out5);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = _mm256_sub_ps(bv_t5, bv_t3);
        STR_256_S(curr_out, v_out_stride, v_out7);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm256_sub_ps(bv_t1, bv_s5);
        STR_256_S(curr_out, v_out_stride, v_out9);
        // Output pt 12: X(11)
        curr_out = out + out_strides[11];
        v_out11 = NEGATE_256_S(_mm256_add_ps(bv_t0, bv_s4));
        STR_256_S(curr_out, v_out_stride, v_out11);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_S)
    {
        // Standard DFT
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5;
        __m128 av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7;
        __m128 av_t0, av_t1, av_t2, av_t3;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_6_1 = _mm256_castps256_ps128(v_CRTM_6_1);
        __m128 v128_CRTM_6_2 = _mm256_castps256_ps128(v_CRTM_6_2);

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_stride, av_in0);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDRI_2x128_S(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDRI_2x128_S(curr_in, v_in_stride, av_in3, av_in4);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_128_S(curr_in, v_in_stride, av_in5);

        av_s0 = _mm_add_ps(av_in0, av_in5);
        av_s1 = _mm_sub_ps(av_in0, av_in5);
        av_s2 = _mm_add_ps(av_in1, av_in3);

        av_s3 = _mm_sub_ps(av_in1, av_in3);
        av_s4 = _mm_add_ps(av_in2, av_in4);
        av_s5 = _mm_sub_ps(av_in2, av_in4);

        av_t0 = _mm_mul_ps(v128_CRTM_6_1, av_s4);
        av_t1 = _mm_mul_ps(v128_CRTM_6_1, av_s5);
        av_s6 = _mm_add_ps(av_s1, av_s3);

        av_s7 = _mm_sub_ps(av_s0, av_s2);
        av_t2 = _mm_mul_ps(v128_CRTM_6_2, av_s2);
        av_t3 = _mm_mul_ps(v128_CRTM_6_2, av_s3);

        // Output pt 1: X(0)
        v_out0 = _mm_add_ps(av_s0, av_t2);
        STR_128_S(curr_out, v_out_stride, v_out0);
        // Output point 3: X(2)--> type-0
        curr_out = out + out_strides[2];
        v_out2 = _mm_sub_ps(av_s6, av_t0);
        STR_128_S(curr_out, v_out_stride, v_out2);
        // Output pt 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm_sub_ps(av_s7, av_t1);
        STR_128_S(curr_out, v_out_stride, v_out4);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm_sub_ps(av_s1, av_t3);
        STR_128_S(curr_out, v_out_stride, v_out6);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm_add_ps(av_s7, av_t1);
        STR_128_S(curr_out, v_out_stride, v_out8);
        // Output pt 11: X(10)
        curr_out = out + out_strides[10];
        v_out10 = _mm_add_ps(av_s6, av_t0);
        STR_128_S(curr_out, v_out_stride, v_out10);

        // Shifted DFT
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5;
        __m128 bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5;
        __m128 bv_t0, bv_t1, bv_t2, bv_t3, bv_t4, bv_t5;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)--> type-2
        curr_in = in + in_strides[5];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in4, bv_in5);

        bv_s0 = _mm_add_ps(bv_in0, bv_in4);
        bv_s1 = _mm_sub_ps(bv_in0, bv_in4);
        bv_t0 = _mm_mul_ps(v128_CRTM_6_1, bv_s1);

        bv_s2 = _mm_add_ps(bv_in1, bv_in5);
        bv_s3 = _mm_sub_ps(bv_in5, bv_in1);
        bv_t1 = _mm_mul_ps(v128_CRTM_6_1, bv_s3);

        bv_t2 = _mm_mul_ps(v128_CRTM_6_2, bv_s0);
        bv_t3 = _mm_mul_ps(v128_CRTM_6_2, bv_s2);
        bv_t4 = _mm_mul_ps(v128_CRTM_6_2, bv_in2);

        bv_t5 = _mm_mul_ps(v128_CRTM_6_2, bv_in3);
        bv_s4 = _mm_add_ps(bv_s2, bv_t5);
        bv_s5 = _mm_sub_ps(bv_s0, bv_t4);

        // Output pt 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm_add_ps(bv_t4, bv_t2);
        STR_128_S(curr_out, v_out_stride, v_out1);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm_sub_ps(bv_t0, bv_s4);
        STR_128_S(curr_out, v_out_stride, v_out3);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm_add_ps(bv_s5, bv_t1);
        STR_128_S(curr_out, v_out_stride, v_out5);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = _mm_sub_ps(bv_t5, bv_t3);
        STR_128_S(curr_out, v_out_stride, v_out7);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm_sub_ps(bv_t1, bv_s5);
        STR_128_S(curr_out, v_out_stride, v_out9);
        // Output pt 12: X(11)
        curr_out = out + out_strides[11];
        v_out11 = NEGATE_128_S(_mm_add_ps(bv_t0, bv_s4));
        STR_128_S(curr_out, v_out_stride, v_out11);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & 2)
    {
        // Standard DFT
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5;
        __m128 av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7;
        __m128 av_t0, av_t1, av_t2, av_t3;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_6_1 = _mm256_castps256_ps128(v_CRTM_6_1);
        __m128 v128_CRTM_6_2 = _mm256_castps256_ps128(v_CRTM_6_2);

        // Input point 1: x(0)
        LDHR_128_S(curr_in, v_in_stride, av_in0);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDHRI_2x128_S(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDHRI_2x128_S(curr_in, v_in_stride, av_in3, av_in4);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDHR_128_S(curr_in, v_in_stride, av_in5);

        av_s0 = _mm_add_ps(av_in0, av_in5);
        av_s1 = _mm_sub_ps(av_in0, av_in5);
        av_s2 = _mm_add_ps(av_in1, av_in3);

        av_s3 = _mm_sub_ps(av_in1, av_in3);
        av_s4 = _mm_add_ps(av_in2, av_in4);
        av_s5 = _mm_sub_ps(av_in2, av_in4);

        av_t0 = _mm_mul_ps(v128_CRTM_6_1, av_s4);
        av_t1 = _mm_mul_ps(v128_CRTM_6_1, av_s5);
        av_s6 = _mm_add_ps(av_s1, av_s3);

        av_s7 = _mm_sub_ps(av_s0, av_s2);
        av_t2 = _mm_mul_ps(v128_CRTM_6_2, av_s2);
        av_t3 = _mm_mul_ps(v128_CRTM_6_2, av_s3);

        // Output pt 1: X(0)
        v_out0 = _mm_add_ps(av_s0, av_t2);
        STHR_128_S(curr_out, v_out_stride, v_out0);
        // Output point 3: X(2)--> type-0
        curr_out = out + out_strides[2];
        v_out2 = _mm_sub_ps(av_s6, av_t0);
        STHR_128_S(curr_out, v_out_stride, v_out2);
        // Output pt 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm_sub_ps(av_s7, av_t1);
        STHR_128_S(curr_out, v_out_stride, v_out4);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm_sub_ps(av_s1, av_t3);
        STHR_128_S(curr_out, v_out_stride, v_out6);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm_add_ps(av_s7, av_t1);
        STHR_128_S(curr_out, v_out_stride, v_out8);
        // Output pt 11: X(10)
        curr_out = out + out_strides[10];
        v_out10 = _mm_add_ps(av_s6, av_t0);
        STHR_128_S(curr_out, v_out_stride, v_out10);

        // Shifted DFT
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5;
        __m128 bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5;
        __m128 bv_t0, bv_t1, bv_t2, bv_t3, bv_t4, bv_t5;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)--> type-2
        curr_in = in + in_strides[5];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in4, bv_in5);

        bv_s0 = _mm_add_ps(bv_in0, bv_in4);
        bv_s1 = _mm_sub_ps(bv_in0, bv_in4);
        bv_t0 = _mm_mul_ps(v128_CRTM_6_1, bv_s1);

        bv_s2 = _mm_add_ps(bv_in1, bv_in5);
        bv_s3 = _mm_sub_ps(bv_in5, bv_in1);
        bv_t1 = _mm_mul_ps(v128_CRTM_6_1, bv_s3);

        bv_t2 = _mm_mul_ps(v128_CRTM_6_2, bv_s0);
        bv_t3 = _mm_mul_ps(v128_CRTM_6_2, bv_s2);
        bv_t4 = _mm_mul_ps(v128_CRTM_6_2, bv_in2);

        bv_t5 = _mm_mul_ps(v128_CRTM_6_2, bv_in3);
        bv_s4 = _mm_add_ps(bv_s2, bv_t5);
        bv_s5 = _mm_sub_ps(bv_s0, bv_t4);

        // Output pt 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm_add_ps(bv_t4, bv_t2);
        STHR_128_S(curr_out, v_out_stride, v_out1);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm_sub_ps(bv_t0, bv_s4);
        STHR_128_S(curr_out, v_out_stride, v_out3);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm_add_ps(bv_s5, bv_t1);
        STHR_128_S(curr_out, v_out_stride, v_out5);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = _mm_sub_ps(bv_t5, bv_t3);
        STHR_128_S(curr_out, v_out_stride, v_out7);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm_sub_ps(bv_t1, bv_s5);
        STHR_128_S(curr_out, v_out_stride, v_out9);
        // Output pt 12: X(11)
        curr_out = out + out_strides[11];
        v_out11 = NEGATE_128_S(_mm_add_ps(bv_t0, bv_s4));
        STHR_128_S(curr_out, v_out_stride, v_out11);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        // Standard DFT
        FLOAT a_in0, a_in1, a_in2, a_in3, a_in4, a_in5;
        FLOAT a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s8, a_s9;
        FLOAT a_t0, a_t1, a_t2, a_t3;

        a_in0 = *in;                // Input point 1: x(0)
        a_in1 = in[in_strides[3]];  // Input point 3: x(2)
        a_in2 = in[in_strides[4]];  // Input point 5: x(4)
        a_in3 = in[in_strides[7]];  // Input point 7: x(6)
        a_in4 = in[in_strides[8]];  // Input point 9: x(8)
        a_in5 = in[in_strides[11]]; // Input point 11: x(10)

        a_s0 = a_in0 + a_in5;
        a_s1 = a_in0 - a_in5;
        a_s2 = a_in1 + a_in3;
        a_s3 = a_in1 - a_in3;
        a_s4 = a_in2 + a_in4;
        a_s5 = a_in2 - a_in4;

        a_t0 = CRTM_6_1 * a_s4;
        a_t1 = CRTM_6_1 * a_s5;
        a_s8 = a_s1 + a_s3;
        a_s9 = a_s0 - a_s2;

        a_t2 = CRTM_6_2 * a_s2;
        a_t3 = CRTM_6_2 * a_s3;

        *out = a_s0 + a_t2;                 // Output pt 1: X(0)
        out[out_strides[2]] = a_s8 - a_t0;  // Output pt 3: X(2)
        out[out_strides[4]] = a_s9 - a_t1;  // Output pt 5: X(4)
        out[out_strides[6]] = a_s1 - a_t3;  // Output pt 7: X(6)
        out[out_strides[8]] = a_s9 + a_t1;  // Output pt 9: X(8)
        out[out_strides[10]] = a_s8 + a_t0; // Output pt 11: X(10)

        // Shifted DFT
        FLOAT b_in0, b_in1, b_in2, b_in3, b_in4, b_in5;
        FLOAT b_s0, b_s1, b_s3, b_s4, b_s5, b_s6;
        FLOAT b_t0, b_t1, b_t2, b_t3, b_t4, b_t5;

        b_in0 = in[in_strides[1]];  // Input point 2: x(1)
        b_in1 = in[in_strides[2]];  // Input point 4: x(3)
        b_in2 = in[in_strides[5]];  // Input point 6: x(5)
        b_in3 = in[in_strides[6]];  // Input point 8: x(7)
        b_in4 = in[in_strides[9]];  // Input point 10: x(9)
        b_in5 = in[in_strides[10]]; // Input point 12: x(11)

        b_s0 = b_in0 + b_in4;
        b_s1 = b_in0 - b_in4;
        b_t0 = CRTM_6_1 * b_s1;
        b_s3 = b_in1 + b_in5;
        b_s4 = b_in5 - b_in1;
        b_t1 = CRTM_6_1 * b_s4;

        b_t2 = CRTM_6_2 * b_s0;
        b_t3 = CRTM_6_2 * b_s3;
        b_t4 = CRTM_6_2 * b_in2;
        b_t5 = CRTM_6_2 * b_in3;

        b_s5 = b_s3 + b_t5;
        b_s6 = b_s0 - b_t4;

        out[out_strides[1]] = b_t4 + b_t2;   // Output pt 2: X(1)
        out[out_strides[3]] = b_t0 - b_s5;   // Output pt 4: X(3)
        out[out_strides[5]] = b_s6 + b_t1;   // Output pt 6: X(5)
        out[out_strides[7]] = b_t5 - b_t3;   // Output pt 8: X(7)
        out[out_strides[9]] = b_t1 - b_s6;   // Output pt 10: X(9)
        out[out_strides[11]] = -b_t0 - b_s5; // Output pt 12: X(11)
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hcf_rfft6avx256_fp64_fwd(VOID *in_real, VOID *in_imag,
                                       VOID *out_real, VOID *out_imag, INTP n,
                                       aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_6_1 = 0.500000000000000000000000000000000000000000000;
    const DOUBLE CRTM_6_2 = 0.866025403784438646763723170752936183471402627;

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

    __m256d v_CRTM_6_1 = _mm256_broadcast_sd(&CRTM_6_1);
    __m256d v_CRTM_6_2 = _mm256_broadcast_sd(&CRTM_6_2);

    for (cnt = 0; cnt < N; cnt++)
    {
        // Standard DFT
        __m256d av_in0, av_in1, av_in2, av_in3, av_in4, av_in5;
        __m256d av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
                av_s9;
        __m256d av_t0, av_t1;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11;

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

        av_s0 = _mm256_add_pd(av_in0, av_in3);
        av_s1 = _mm256_sub_pd(av_in0, av_in3);
        av_s2 = _mm256_add_pd(av_in1, av_in2);
        av_s3 = _mm256_sub_pd(av_in2, av_in1);

        av_s4 = _mm256_add_pd(av_in4, av_in5);
        av_s5 = _mm256_sub_pd(av_in5, av_in4);
        av_s6 = _mm256_add_pd(av_s2, av_s4);
        av_s7 = _mm256_sub_pd(av_s5, av_s3);

        av_t0 = _mm256_mul_pd(v_CRTM_6_1, av_s7);
        av_s8 = _mm256_sub_pd(av_s4, av_s2);

        av_t1 = _mm256_mul_pd(v_CRTM_6_1, av_s6);
        av_s9 = _mm256_add_pd(av_s3, av_s5);

        // Output pt 1: X(0)
        v_out0 = _mm256_add_pd(av_s0, av_s6);
        STR_256_D(curr_out, v_out_stride, v_out0);
        // Output pt 4: X(3) & Output pt 5: X(4)
        curr_out = out + out_strides[3];
        v_out3 = _mm256_add_pd(av_s1, av_t0);
        v_out4 = _mm256_mul_pd(v_CRTM_6_2, av_s8);
        STRI_2x256_D(curr_out, v_out_stride, v_out3, v_out4);
        // Output pt 8: X(7) & Output pt 9: X(8)
        curr_out = out + out_strides[7];
        v_out7 = _mm256_sub_pd(av_s0, av_t1);
        v_out8 = _mm256_mul_pd(v_CRTM_6_2, av_s9);
        STRI_2x256_D(curr_out, v_out_stride, v_out7, v_out8);
        // Output pt 12: X(11)
        curr_out = out + out_strides[11];
        v_out11 = _mm256_sub_pd(av_s1, av_s7);
        STR_256_D(curr_out, v_out_stride, v_out11);

        // Shifted DFT
        __m256d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5;
        __m256d bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5;
        __m256d bv_t0, bv_t1, bv_t2, bv_t3;

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

        bv_s0 = _mm256_sub_pd(bv_in1, bv_in5);
        bv_s1 = _mm256_add_pd(bv_in1, bv_in5);
        bv_s2 = _mm256_sub_pd(bv_in2, bv_in4);
        bv_s3 = _mm256_add_pd(bv_in2, bv_in4);

        bv_t0 = _mm256_mul_pd(v_CRTM_6_2, bv_s0);
        bv_t1 = _mm256_mul_pd(v_CRTM_6_1, bv_s1);
        bv_t2 = _mm256_mul_pd(v_CRTM_6_1, bv_s2);
        bv_t3 = _mm256_mul_pd(v_CRTM_6_2, bv_s3);

        bv_s4 = _mm256_add_pd(bv_in0, bv_t2);
        bv_s5 = _mm256_add_pd(bv_in3, bv_t1);

        // Output point 2: X(1) & Output point 3: X(2)
        curr_out = out + out_strides[1];
        v_out1 = _mm256_add_pd(bv_s4, bv_t0);
        v_out2 = NEGATE_256_D(_mm256_add_pd(bv_t3, bv_s5));
        STRI_2x256_D(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 6: X(5) & Output point 7: X(6)
        curr_out = out + out_strides[5];
        v_out5 = _mm256_sub_pd(bv_in0, bv_s2);
        v_out6 = _mm256_sub_pd(bv_in3, bv_s1);
        STRI_2x256_D(curr_out, v_out_stride, v_out5, v_out6);
        // Output point 10: X(9) & Output point 11: X(10)
        curr_out = out + out_strides[9];
        v_out9 = _mm256_sub_pd(bv_s4, bv_t0);
        v_out10 = _mm256_sub_pd(bv_t3, bv_s5);
        STRI_2x256_D(curr_out, v_out_stride, v_out9, v_out10);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_D)
    {
        // Standard DFT
        __m128d av_in0, av_in1, av_in2, av_in3, av_in4, av_in5;
        __m128d av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
                av_s9;
        __m128d av_t0, av_t1;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11;

        curr_in = in;
        curr_out = out;

        __m128d v128_CRTM_6_1 = _mm256_castpd256_pd128(v_CRTM_6_1);
        __m128d v128_CRTM_6_2 = _mm256_castpd256_pd128(v_CRTM_6_2);

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

        av_s0 = _mm_add_pd(av_in0, av_in3);
        av_s1 = _mm_sub_pd(av_in0, av_in3);
        av_s2 = _mm_add_pd(av_in1, av_in2);
        av_s3 = _mm_sub_pd(av_in2, av_in1);

        av_s4 = _mm_add_pd(av_in4, av_in5);
        av_s5 = _mm_sub_pd(av_in5, av_in4);
        av_s6 = _mm_add_pd(av_s2, av_s4);
        av_s7 = _mm_sub_pd(av_s5, av_s3);

        av_t0 = _mm_mul_pd(v128_CRTM_6_1, av_s7);
        av_s8 = _mm_sub_pd(av_s4, av_s2);

        av_t1 = _mm_mul_pd(v128_CRTM_6_1, av_s6);
        av_s9 = _mm_add_pd(av_s3, av_s5);

        // Output pt 1: X(0)
        v_out0 = _mm_add_pd(av_s0, av_s6);
        STR_128_D(curr_out, v_out_stride, v_out0);
        // Output pt 4: X(3) & Output pt 5: X(4)
        curr_out = out + out_strides[3];
        v_out3 = _mm_add_pd(av_s1, av_t0);
        v_out4 = _mm_mul_pd(v128_CRTM_6_2, av_s8);
        STRI_2x128_D(curr_out, v_out_stride, v_out3, v_out4);
        // Output pt 8: X(7) & Output pt 9: X(8)
        curr_out = out + out_strides[7];
        v_out7 = _mm_sub_pd(av_s0, av_t1);
        v_out8 = _mm_mul_pd(v128_CRTM_6_2, av_s9);
        STRI_2x128_D(curr_out, v_out_stride, v_out7, v_out8);
        // Output pt 12: X(11)
        curr_out = out + out_strides[11];
        v_out11 = _mm_sub_pd(av_s1, av_s7);
        STR_128_D(curr_out, v_out_stride, v_out11);

        // Shifted DFT
        __m128d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5;
        __m128d bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5;
        __m128d bv_t0, bv_t1, bv_t2, bv_t3;

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

        bv_s0 = _mm_sub_pd(bv_in1, bv_in5);
        bv_s1 = _mm_add_pd(bv_in1, bv_in5);
        bv_s2 = _mm_sub_pd(bv_in2, bv_in4);
        bv_s3 = _mm_add_pd(bv_in2, bv_in4);

        bv_t0 = _mm_mul_pd(v128_CRTM_6_2, bv_s0);
        bv_t1 = _mm_mul_pd(v128_CRTM_6_1, bv_s1);
        bv_t2 = _mm_mul_pd(v128_CRTM_6_1, bv_s2);
        bv_t3 = _mm_mul_pd(v128_CRTM_6_2, bv_s3);

        bv_s4 = _mm_add_pd(bv_in0, bv_t2);
        bv_s5 = _mm_add_pd(bv_in3, bv_t1);

        // Output point 2: X(1) & Output point 3: X(2)
        curr_out = out + out_strides[1];
        v_out1 = _mm_add_pd(bv_s4, bv_t0);
        v_out2 = NEGATE_128_D(_mm_add_pd(bv_t3, bv_s5));
        STRI_2x128_D(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 6: X(5) & Output point 7: X(6)
        curr_out = out + out_strides[5];
        v_out5 = _mm_sub_pd(bv_in0, bv_s2);
        v_out6 = _mm_sub_pd(bv_in3, bv_s1);
        STRI_2x128_D(curr_out, v_out_stride, v_out5, v_out6);
        // Output point 10: X(9) & Output point 11: X(10)
        curr_out = out + out_strides[9];
        v_out9 = _mm_sub_pd(bv_s4, bv_t0);
        v_out10 = _mm_sub_pd(bv_t3, bv_s5);
        STRI_2x128_D(curr_out, v_out_stride, v_out9, v_out10);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        // Standard DFT
        DOUBLE a_in0, a_in1, a_in2, a_in3, a_in4, a_in5;
        DOUBLE a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_s9;
        DOUBLE a_t0, a_t1;

        a_in0 = *in;                // Input point 1: x(0)
        a_in1 = in[in_strides[2]];  // Input point 3: x(2)
        a_in2 = in[in_strides[4]];  // Input point 5: x(4)
        a_in3 = in[in_strides[6]];  // Input point 7: x(6)
        a_in4 = in[in_strides[8]];  // Input point 9: x(8)
        a_in5 = in[in_strides[10]]; // Input point 11: x(10)

        a_s0 = a_in0 + a_in3;
        a_s1 = a_in0 - a_in3;
        a_s2 = a_in1 + a_in2;
        a_s3 = a_in1 - a_in2;
        a_s4 = a_in5 + a_in4;
        a_s5 = a_in5 - a_in4;
        a_s6 = a_s2 + a_s4;
        a_s7 = a_s3 + a_s5;

        a_t0 = CRTM_6_1 * a_s7;
        a_s8 = a_s4 - a_s2;
        a_t1 = CRTM_6_1 * a_s6;
        a_s9 = a_s5 - a_s3;

        *out = a_s0 + a_s6;                    // Output pt 1: X(0)
        out[out_strides[3]] = a_s1 + a_t0;     // Output pt 4: X(3)
        out[out_strides[4]] = CRTM_6_2 * a_s8; // Output pt 5: X(4)
        out[out_strides[7]] = a_s0 - a_t1;     // Output pt 8: X(7)
        out[out_strides[8]] = CRTM_6_2 * a_s9; // Output pt 9: X(8)
        out[out_strides[11]] = a_s1 - a_s7;    // Output pt 12: X(11)

        // Shifted DFT
        DOUBLE b_in0, b_in1, b_in2, b_in3, b_in4, b_in5;
        DOUBLE b_s0, b_s1, b_s2, b_s3, b_s4, b_s5;
        DOUBLE b_t0, b_t1, b_t2, b_t3;

        b_in0 = in[in_strides[1]];  // Input point 2: x(1)
        b_in1 = in[in_strides[3]];  // Input point 4: x(3)
        b_in2 = in[in_strides[5]];  // Input point 6: x(5)
        b_in3 = in[in_strides[7]];  // Input point 8: x(7)
        b_in4 = in[in_strides[9]];  // Input point 10: x(9)
        b_in5 = in[in_strides[11]]; // Input point 12: x(11)

        b_s0 = b_in1 - b_in5;
        b_s1 = b_in1 + b_in5;
        b_s2 = b_in2 - b_in4;
        b_s3 = b_in2 + b_in4;

        b_t0 = CRTM_6_2 * b_s0;
        b_t1 = CRTM_6_1 * b_s1;
        b_t2 = CRTM_6_1 * b_s2;
        b_t3 = CRTM_6_2 * b_s3;

        b_s4 = b_in0 + b_t2;
        b_s5 = b_in3 + b_t1;

        out[out_strides[1]] = b_s4 + b_t0;  // Output pt 2: X(1)
        out[out_strides[2]] = -b_t3 - b_s5; // Output pt 3: X(2)
        out[out_strides[5]] = b_in0 - b_s2; // Output pt 6: X(5)
        out[out_strides[6]] = b_in3 - b_s1; // Output pt 7: X(6)
        out[out_strides[9]] = b_s4 - b_t0;  // Output pt 10: X(9)
        out[out_strides[10]] = b_t3 - b_s5; // Output pt 11: X(10)
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hcf_rfft6avx256_fp64_bwd(VOID *in_real, VOID *in_imag,
                                       VOID *out_real, VOID *out_imag, INTP n,
                                       aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_6_1 = 1.732050807568877293527446341505872366942805253;
    const DOUBLE CRTM_6_2 = 2.000000000000000000000000000000000000000000000;

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

    __m256d v_CRTM_6_1 = _mm256_broadcast_sd(&CRTM_6_1);
    __m256d v_CRTM_6_2 = _mm256_broadcast_sd(&CRTM_6_2);

    for (cnt = 0; cnt < N; cnt++)
    {
        // Standard DFT
        __m256d av_in0, av_in1, av_in2, av_in3, av_in4, av_in5;
        __m256d av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7;
        __m256d av_t0, av_t1, av_t2, av_t3;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_256_D(curr_in, v_in_stride, av_in0);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDRI_2x256_D(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDRI_2x256_D(curr_in, v_in_stride, av_in3, av_in4);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_256_D(curr_in, v_in_stride, av_in5);

        av_s0 = _mm256_add_pd(av_in0, av_in5);
        av_s1 = _mm256_sub_pd(av_in0, av_in5);
        av_s2 = _mm256_add_pd(av_in1, av_in3);

        av_s3 = _mm256_sub_pd(av_in1, av_in3);
        av_s4 = _mm256_add_pd(av_in2, av_in4);
        av_s5 = _mm256_sub_pd(av_in2, av_in4);

        av_t0 = _mm256_mul_pd(v_CRTM_6_1, av_s4);
        av_t1 = _mm256_mul_pd(v_CRTM_6_1, av_s5);
        av_s6 = _mm256_add_pd(av_s1, av_s3);

        av_s7 = _mm256_sub_pd(av_s0, av_s2);
        av_t2 = _mm256_mul_pd(v_CRTM_6_2, av_s2);
        av_t3 = _mm256_mul_pd(v_CRTM_6_2, av_s3);

        // Output pt 1: X(0)
        v_out0 = _mm256_add_pd(av_s0, av_t2);
        STR_256_D(curr_out, v_out_stride, v_out0);
        // Output point 3: X(2)--> type-0
        curr_out = out + out_strides[2];
        v_out2 = _mm256_sub_pd(av_s6, av_t0);
        STR_256_D(curr_out, v_out_stride, v_out2);
        // Output pt 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm256_sub_pd(av_s7, av_t1);
        STR_256_D(curr_out, v_out_stride, v_out4);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm256_sub_pd(av_s1, av_t3);
        STR_256_D(curr_out, v_out_stride, v_out6);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm256_add_pd(av_s7, av_t1);
        STR_256_D(curr_out, v_out_stride, v_out8);
        // Output pt 11: X(10)
        curr_out = out + out_strides[10];
        v_out10 = _mm256_add_pd(av_s6, av_t0);
        STR_256_D(curr_out, v_out_stride, v_out10);

        // Shifted DFT
        __m256d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5;
        __m256d bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5;
        __m256d bv_t0, bv_t1, bv_t2, bv_t3, bv_t4, bv_t5;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x256_D(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)--> type-2
        curr_in = in + in_strides[5];
        LDRI_2x256_D(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDRI_2x256_D(curr_in, v_in_stride, bv_in4, bv_in5);

        bv_s0 = _mm256_add_pd(bv_in0, bv_in4);
        bv_s1 = _mm256_sub_pd(bv_in0, bv_in4);
        bv_t0 = _mm256_mul_pd(v_CRTM_6_1, bv_s1);

        bv_s2 = _mm256_add_pd(bv_in1, bv_in5);
        bv_s3 = _mm256_sub_pd(bv_in5, bv_in1);
        bv_t1 = _mm256_mul_pd(v_CRTM_6_1, bv_s3);

        bv_t2 = _mm256_mul_pd(v_CRTM_6_2, bv_s0);
        bv_t3 = _mm256_mul_pd(v_CRTM_6_2, bv_s2);
        bv_t4 = _mm256_mul_pd(v_CRTM_6_2, bv_in2);

        bv_t5 = _mm256_mul_pd(v_CRTM_6_2, bv_in3);
        bv_s4 = _mm256_add_pd(bv_s2, bv_t5);
        bv_s5 = _mm256_sub_pd(bv_s0, bv_t4);

        // Output pt 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm256_add_pd(bv_t4, bv_t2);
        STR_256_D(curr_out, v_out_stride, v_out1);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm256_sub_pd(bv_t0, bv_s4);
        STR_256_D(curr_out, v_out_stride, v_out3);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm256_add_pd(bv_s5, bv_t1);
        STR_256_D(curr_out, v_out_stride, v_out5);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = _mm256_sub_pd(bv_t5, bv_t3);
        STR_256_D(curr_out, v_out_stride, v_out7);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm256_sub_pd(bv_t1, bv_s5);
        STR_256_D(curr_out, v_out_stride, v_out9);
        // Output pt 12: X(11)
        curr_out = out + out_strides[11];
        v_out11 = NEGATE_256_D(_mm256_add_pd(bv_t0, bv_s4));
        STR_256_D(curr_out, v_out_stride, v_out11);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_D)
    {
        // Standard DFT
        __m128d av_in0, av_in1, av_in2, av_in3, av_in4, av_in5;
        __m128d av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7;
        __m128d av_t0, av_t1, av_t2, av_t3;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11;

        curr_in = in;
        curr_out = out;

        __m128d v128_CRTM_6_1 = _mm256_castpd256_pd128(v_CRTM_6_1);
        __m128d v128_CRTM_6_2 = _mm256_castpd256_pd128(v_CRTM_6_2);

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_stride, av_in0);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDRI_2x128_D(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDRI_2x128_D(curr_in, v_in_stride, av_in3, av_in4);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_128_D(curr_in, v_in_stride, av_in5);

        av_s0 = _mm_add_pd(av_in0, av_in5);
        av_s1 = _mm_sub_pd(av_in0, av_in5);
        av_s2 = _mm_add_pd(av_in1, av_in3);

        av_s3 = _mm_sub_pd(av_in1, av_in3);
        av_s4 = _mm_add_pd(av_in2, av_in4);
        av_s5 = _mm_sub_pd(av_in2, av_in4);

        av_t0 = _mm_mul_pd(v128_CRTM_6_1, av_s4);
        av_t1 = _mm_mul_pd(v128_CRTM_6_1, av_s5);
        av_s6 = _mm_add_pd(av_s1, av_s3);

        av_s7 = _mm_sub_pd(av_s0, av_s2);
        av_t2 = _mm_mul_pd(v128_CRTM_6_2, av_s2);
        av_t3 = _mm_mul_pd(v128_CRTM_6_2, av_s3);

        // Output pt 1: X(0)
        v_out0 = _mm_add_pd(av_s0, av_t2);
        STR_128_D(curr_out, v_out_stride, v_out0);
        // Output point 3: X(2)--> type-0
        curr_out = out + out_strides[2];
        v_out2 = _mm_sub_pd(av_s6, av_t0);
        STR_128_D(curr_out, v_out_stride, v_out2);
        // Output pt 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm_sub_pd(av_s7, av_t1);
        STR_128_D(curr_out, v_out_stride, v_out4);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm_sub_pd(av_s1, av_t3);
        STR_128_D(curr_out, v_out_stride, v_out6);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm_add_pd(av_s7, av_t1);
        STR_128_D(curr_out, v_out_stride, v_out8);
        // Output pt 11: X(10)
        curr_out = out + out_strides[10];
        v_out10 = _mm_add_pd(av_s6, av_t0);
        STR_128_D(curr_out, v_out_stride, v_out10);

        // Shifted DFT
        __m128d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5;
        __m128d bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5;
        __m128d bv_t0, bv_t1, bv_t2, bv_t3, bv_t4, bv_t5;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)--> type-2
        curr_in = in + in_strides[5];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in4, bv_in5);

        bv_s0 = _mm_add_pd(bv_in0, bv_in4);
        bv_s1 = _mm_sub_pd(bv_in0, bv_in4);
        bv_t0 = _mm_mul_pd(v128_CRTM_6_1, bv_s1);

        bv_s2 = _mm_add_pd(bv_in1, bv_in5);
        bv_s3 = _mm_sub_pd(bv_in5, bv_in1);
        bv_t1 = _mm_mul_pd(v128_CRTM_6_1, bv_s3);

        bv_t2 = _mm_mul_pd(v128_CRTM_6_2, bv_s0);
        bv_t3 = _mm_mul_pd(v128_CRTM_6_2, bv_s2);
        bv_t4 = _mm_mul_pd(v128_CRTM_6_2, bv_in2);

        bv_t5 = _mm_mul_pd(v128_CRTM_6_2, bv_in3);
        bv_s4 = _mm_add_pd(bv_s2, bv_t5);
        bv_s5 = _mm_sub_pd(bv_s0, bv_t4);

        // Output pt 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm_add_pd(bv_t4, bv_t2);
        STR_128_D(curr_out, v_out_stride, v_out1);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm_sub_pd(bv_t0, bv_s4);
        STR_128_D(curr_out, v_out_stride, v_out3);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm_add_pd(bv_s5, bv_t1);
        STR_128_D(curr_out, v_out_stride, v_out5);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = _mm_sub_pd(bv_t5, bv_t3);
        STR_128_D(curr_out, v_out_stride, v_out7);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm_sub_pd(bv_t1, bv_s5);
        STR_128_D(curr_out, v_out_stride, v_out9);
        // Output pt 12: X(11)
        curr_out = out + out_strides[11];
        v_out11 = NEGATE_128_D(_mm_add_pd(bv_t0, bv_s4));
        STR_128_D(curr_out, v_out_stride, v_out11);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        // Standard DFT
        DOUBLE a_in0, a_in1, a_in2, a_in3, a_in4, a_in5;
        DOUBLE a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s8, a_s9;
        DOUBLE a_t0, a_t1, a_t2, a_t3;

        a_in0 = *in;                // Input point 1: x(0)
        a_in1 = in[in_strides[3]];  // Input point 3: x(2)
        a_in2 = in[in_strides[4]];  // Input point 5: x(4)
        a_in3 = in[in_strides[7]];  // Input point 7: x(6)
        a_in4 = in[in_strides[8]];  // Input point 9: x(8)
        a_in5 = in[in_strides[11]]; // Input point 11: x(10)

        a_s0 = a_in0 + a_in5;
        a_s1 = a_in0 - a_in5;
        a_s2 = a_in1 + a_in3;
        a_s3 = a_in1 - a_in3;
        a_s4 = a_in2 + a_in4;
        a_s5 = a_in2 - a_in4;

        a_t0 = CRTM_6_1 * a_s4;
        a_t1 = CRTM_6_1 * a_s5;
        a_s8 = a_s1 + a_s3;
        a_s9 = a_s0 - a_s2;

        a_t2 = CRTM_6_2 * a_s2;
        a_t3 = CRTM_6_2 * a_s3;

        *out = a_s0 + a_t2;                 // Output pt 1: X(0)
        out[out_strides[2]] = a_s8 - a_t0;  // Output pt 3: X(2)
        out[out_strides[4]] = a_s9 - a_t1;  // Output pt 5: X(4)
        out[out_strides[6]] = a_s1 - a_t3;  // Output pt 7: X(6)
        out[out_strides[8]] = a_s9 + a_t1;  // Output pt 9: X(8)
        out[out_strides[10]] = a_s8 + a_t0; // Output pt 11: X(10)

        // Shifted DFT
        DOUBLE b_in0, b_in1, b_in2, b_in3, b_in4, b_in5;
        DOUBLE b_s0, b_s1, b_s3, b_s4, b_s5, b_s6;
        DOUBLE b_t0, b_t1, b_t2, b_t3, b_t4, b_t5;

        b_in0 = in[in_strides[1]];  // Input point 2: x(1)
        b_in1 = in[in_strides[2]];  // Input point 4: x(3)
        b_in2 = in[in_strides[5]];  // Input point 6: x(5)
        b_in3 = in[in_strides[6]];  // Input point 8: x(7)
        b_in4 = in[in_strides[9]];  // Input point 10: x(9)
        b_in5 = in[in_strides[10]]; // Input point 12: x(11)

        b_s0 = b_in0 + b_in4;
        b_s1 = b_in0 - b_in4;
        b_t0 = CRTM_6_1 * b_s1;
        b_s3 = b_in1 + b_in5;
        b_s4 = b_in5 - b_in1;
        b_t1 = CRTM_6_1 * b_s4;

        b_t2 = CRTM_6_2 * b_s0;
        b_t3 = CRTM_6_2 * b_s3;
        b_t4 = CRTM_6_2 * b_in2;
        b_t5 = CRTM_6_2 * b_in3;

        b_s5 = b_s3 + b_t5;
        b_s6 = b_s0 - b_t4;

        out[out_strides[1]] = b_t4 + b_t2;   // Output pt 2: X(1)
        out[out_strides[3]] = b_t0 - b_s5;   // Output pt 4: X(3)
        out[out_strides[5]] = b_s6 + b_t1;   // Output pt 6: X(5)
        out[out_strides[7]] = b_t5 - b_t3;   // Output pt 8: X(7)
        out[out_strides[9]] = b_t1 - b_s6;   // Output pt 10: X(9)
        out[out_strides[11]] = -b_t0 - b_s5; // Output pt 12: X(11)
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hcf_rfft6avx256(UINT8 precision, UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hcf_rfft6avx256_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft6avx256_fp64_fwd;
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
            return r2hcf_rfft6avx256_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft6avx256_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
