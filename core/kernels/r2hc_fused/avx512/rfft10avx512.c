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

/** @file rfft10avx512.c
 *
 *  @brief Radix-10 r2hc_fused Real-FFT kernel with with AVX-512 operations
 *  using x86 SIMD intrinsics.
 *
 *  This file contains the DIT radix-10 real-to-halfcomplex fused of two
 *  different implementations (Standard DFT and Shifted DFT that differs in DFT
 *  weight matrix) using x86 SIMD intrinsics for single-precision and
 *  double-precision inputs
 *
 *  @author Jeya R
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/r2hc_simd_avx512.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                {{{0, 24, 66, 496, 282, 123},
                                                  {0, 28, 68, 496, 354, 123}},
                                                 {{0, 24, 66, 248, 18,  123},
                                                  {0, 28, 68, 248, 18,  123}}};

ops_cycles_t get_ops_cnt_r2hcf_rfft10avx512(UINT8 precision, UINT8 direction)
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

static VOID r2hcf_rfft10avx512_fp32_fwd(VOID *in_real, VOID *in_imag,
                                        VOID *out_real, VOID *out_imag, INTP n,
                                        aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const FLOAT CRTM_10_1 =
        0.55901699437494742410229341718281905886015458990288f;
    const FLOAT CRTM_10_2 =
        0.25000000000000000000000000000000000000000000000000f;
    const FLOAT CRTM_10_3 =
        0.58778525229247315738615484497912915412138427663885f;
    const FLOAT CRTM_10_4 =
        0.95105651629515357211643933337938214340569863400000f;

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

    __m512 v_CRTM_10_1 = _mm512_set1_ps(CRTM_10_1);
    __m512 v_CRTM_10_2 = _mm512_set1_ps(CRTM_10_2);
    __m512 v_CRTM_10_3 = _mm512_set1_ps(CRTM_10_3);
    __m512 v_CRTM_10_4 = _mm512_set1_ps(CRTM_10_4);

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m512 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
               av_in8, av_in9;
        __m512 av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
               av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16,
               av_s17, av_s18, av_s19, av_s20, av_s21, av_s22, av_s23;
        __m512 av_t0, av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8,
               av_t9, av_t10, av_t11;
        __m512 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17, v_out18, v_out19;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_512_S(curr_in, v_in_stride, av_in0);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_512_S(curr_in, v_in_stride, av_in1);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_512_S(curr_in, v_in_stride, av_in2);
        // Input point 7: x(6)
        curr_in = in + in_strides[6];
        LDR_512_S(curr_in, v_in_stride, av_in3);
        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDR_512_S(curr_in, v_in_stride, av_in4);
        // Input point 11: x(10)
        curr_in = in + in_strides[10];
        LDR_512_S(curr_in, v_in_stride, av_in5);
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDR_512_S(curr_in, v_in_stride, av_in6);
        // Input point 15: x(14)
        curr_in = in + in_strides[14];
        LDR_512_S(curr_in, v_in_stride, av_in7);
        // Input point 17: x(16)
        curr_in = in + in_strides[16];
        LDR_512_S(curr_in, v_in_stride, av_in8);
        // Input point 19: x(18)
        curr_in = in + in_strides[18];
        LDR_512_S(curr_in, v_in_stride, av_in9);

        av_s0 = _mm512_add_ps(av_in0, av_in5);
        av_s1 = _mm512_sub_ps(av_in0, av_in5);
        av_s2 = _mm512_add_ps(av_in1, av_in9);
        av_s3 = _mm512_sub_ps(av_in1, av_in9);
        av_s4 = _mm512_add_ps(av_in2, av_in3);
        av_s5 = _mm512_sub_ps(av_in2, av_in3);
        av_s6 = _mm512_add_ps(av_in4, av_in6);
        av_s7 = _mm512_sub_ps(av_in4, av_in6);
        av_s8 = _mm512_add_ps(av_in7, av_in8);
        av_s9 = _mm512_sub_ps(av_in7, av_in8);

        av_s10 = _mm512_add_ps(av_s2, av_s6);
        av_s14 = _mm512_add_ps(av_s4, av_s8);
        av_s18 = _mm512_add_ps(av_s10,av_s14);
        // Output pt 1: X(0)
        v_out0 = _mm512_add_ps(av_s0, av_s18);
        STR_512_S(curr_out, v_out_stride, v_out0);

        av_s11 = _mm512_sub_ps(av_s2, av_s6);
        av_s17 = _mm512_sub_ps(av_s5, av_s9);
        av_s20 = _mm512_add_ps(av_s11, av_s17);
        av_s21 = _mm512_sub_ps(av_s11, av_s17);
        // Output point 20: X(19)
        v_out19 = _mm512_sub_ps(av_s1, av_s21);
        curr_out = out + out_strides[19];
        STR_512_S(curr_out, v_out_stride, v_out19);

        av_t0  = _mm512_mul_ps(v_CRTM_10_2, av_s21);
        av_t1  = _mm512_mul_ps(v_CRTM_10_1, av_s20);
        av_s22 = _mm512_add_ps(av_t0, av_s1);
        // Output point 4: X(3)
        v_out3 = _mm512_add_ps(av_s22, av_t1);
        // Output point 12: X(11)
        v_out11 = _mm512_sub_ps(av_s22, av_t1);

        av_s12 = _mm512_add_ps(av_s3, av_s7);
        av_s15 = _mm512_sub_ps(av_s4, av_s8);
        av_t4  = _mm512_mul_ps(v_CRTM_10_3, av_s12);
        av_t10 = _mm512_mul_ps(v_CRTM_10_4, av_s15);
        // Output point 5: X(4)
        v_out4 = NEGATE_512_S(_mm512_add_ps(av_t4, av_t10));
        curr_out = out + out_strides[3];
        STRI_2x512_S(curr_out, v_out_stride, v_out3, v_out4);

        av_t6 = _mm512_mul_ps(v_CRTM_10_3, av_s15);
        av_t8 = _mm512_mul_ps(v_CRTM_10_4, av_s12);
        // Output point 13: X(12)
        v_out12 = _mm512_sub_ps(av_t6, av_t8);
        curr_out = out + out_strides[11];
        STRI_2x512_S(curr_out, v_out_stride, v_out11, v_out12);

        av_s13 = _mm512_sub_ps(av_s7, av_s3);
        av_s16 = _mm512_add_ps(av_s5, av_s9);
        av_t5  = _mm512_mul_ps(v_CRTM_10_3, av_s13);
        av_t11 = _mm512_mul_ps(v_CRTM_10_4, av_s16);
        // Output point 17: X(16)
        v_out16 = _mm512_add_ps(av_t11, av_t5);

        av_t7 = _mm512_mul_ps(v_CRTM_10_3, av_s16);
        av_t9 = _mm512_mul_ps(v_CRTM_10_4, av_s13);
        // Output point 9: X(8)
        v_out8 = _mm512_sub_ps(av_t9, av_t7);

        av_s19 = _mm512_sub_ps(av_s10, av_s14);
        av_t2  = _mm512_mul_ps(v_CRTM_10_2, av_s18);
        av_t3  = _mm512_mul_ps(v_CRTM_10_1, av_s19);
        av_s23 = _mm512_sub_ps(av_s0, av_t2);
        // Output point 8: X(7)
        v_out7 = _mm512_add_ps(av_s23, av_t3);
        curr_out = out + out_strides[7];
        STRI_2x512_S(curr_out, v_out_stride, v_out7, v_out8);
        // Output point 16: X(15)
        v_out15 = _mm512_sub_ps(av_s23, av_t3);
        curr_out = out + out_strides[15];
        STRI_2x512_S(curr_out, v_out_stride, v_out15, v_out16);

        /* Shifted DFT */
        __m512 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8, bv_in9;
        __m512 bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
               bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16,
               bv_s17, bv_s18, bv_s19, bv_s20, bv_s21;
        __m512 bv_t0, bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8,
               bv_t9, bv_t10, bv_t11;

        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_512_S(curr_in, v_in_stride, bv_in0);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_512_S(curr_in, v_in_stride, bv_in1);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_512_S(curr_in, v_in_stride, bv_in2);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_512_S(curr_in, v_in_stride, bv_in3);
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDR_512_S(curr_in, v_in_stride, bv_in4);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_512_S(curr_in, v_in_stride, bv_in5);
        // Input point 14: x(13)
        curr_in = in + in_strides[13];
        LDR_512_S(curr_in, v_in_stride, bv_in6);
        // Input point 16: x(15)
        curr_in = in + in_strides[15];
        LDR_512_S(curr_in, v_in_stride, bv_in7);
        // Input point 18: x(17)
        curr_in = in + in_strides[17];
        LDR_512_S(curr_in, v_in_stride, bv_in8);
        // Input point 20: x(19)
        curr_in = in + in_strides[19];
        LDR_512_S(curr_in, v_in_stride, bv_in9);

        bv_s0 = _mm512_add_ps(bv_in1, bv_in9);
        bv_s1 = _mm512_sub_ps(bv_in1, bv_in9);
        bv_s2 = _mm512_add_ps(bv_in2, bv_in8);
        bv_s3 = _mm512_sub_ps(bv_in2, bv_in8);
        bv_s4 = _mm512_add_ps(bv_in3, bv_in7);
        bv_s5 = _mm512_sub_ps(bv_in3, bv_in7);
        bv_s6 = _mm512_add_ps(bv_in4, bv_in6);
        bv_s7 = _mm512_sub_ps(bv_in4, bv_in6);

        bv_s8  = _mm512_add_ps(bv_s0, bv_s4);
        bv_s9  = _mm512_sub_ps(bv_s0, bv_s4);
        bv_s10 = _mm512_add_ps(bv_s3, bv_s7);
        bv_s11 = _mm512_sub_ps(bv_s3, bv_s7);
        // Output point 10: X(9)
        v_out9 = _mm512_sub_ps(bv_in0, bv_s11);
        // Output point 11: X(10)
        v_out10 = NEGATE_512_S(_mm512_add_ps(bv_s9, bv_in5));
        curr_out = out + out_strides[9];
        STRI_2x512_S(curr_out, v_out_stride, v_out9, v_out10);

        bv_t0 = _mm512_mul_ps(v_CRTM_10_2, bv_s11);
        bv_t1 = _mm512_mul_ps(v_CRTM_10_1, bv_s10);
        bv_t2 = _mm512_mul_ps(v_CRTM_10_1, bv_s8);
        bv_t3 = _mm512_mul_ps(v_CRTM_10_2, bv_s9);

        bv_s12 = _mm512_add_ps(bv_in0, bv_t0);
        bv_s13 = _mm512_sub_ps(bv_t3, bv_in5);

        bv_t4  = _mm512_mul_ps(v_CRTM_10_4, bv_s1);
        bv_t5  = _mm512_mul_ps(v_CRTM_10_3, bv_s5);
        bv_t6  = _mm512_mul_ps(v_CRTM_10_4, bv_s6);
        bv_t7  = _mm512_mul_ps(v_CRTM_10_3, bv_s2);
        bv_t8  = _mm512_mul_ps(v_CRTM_10_4, bv_s5);
        bv_t9  = _mm512_mul_ps(v_CRTM_10_3, bv_s1);
        bv_t10 = _mm512_mul_ps(v_CRTM_10_4, bv_s2);
        bv_t11 = _mm512_mul_ps(v_CRTM_10_3, bv_s6);

        bv_s14 = _mm512_add_ps(bv_s12, bv_t1);
        bv_s15 = _mm512_add_ps(bv_t4, bv_t5);
        // Output point 2: X(1)
        v_out1 = _mm512_add_ps(bv_s14, bv_s15);
        // Output point 18: X(17)
        v_out17 = _mm512_sub_ps(bv_s14, bv_s15);

        bv_s16 = _mm512_sub_ps(bv_s13, bv_t2);
        bv_s17 = _mm512_add_ps(bv_t6, bv_t7);
        // Output point 3: X(2)
        v_out2 = _mm512_sub_ps(bv_s16, bv_s17);
        curr_out = out + out_strides[1];
        STRI_2x512_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 19: X(18)
        v_out18 = _mm512_add_ps(bv_s16, bv_s17);
        curr_out = out + out_strides[17];
        STRI_2x512_S(curr_out, v_out_stride, v_out17, v_out18);

        bv_s18 = _mm512_sub_ps(bv_s12, bv_t1);
        bv_s19 = _mm512_sub_ps(bv_t9, bv_t8);
        // Output point 6: X(5)
        v_out5 = _mm512_add_ps(bv_s18, bv_s19);
        // Output point 14: X(13)
        v_out13 = _mm512_sub_ps(bv_s18, bv_s19);

        bv_s20 = _mm512_add_ps(bv_s13, bv_t2);
        bv_s21 = _mm512_sub_ps(bv_t11, bv_t10);
        // Output point 7: X(6)
        v_out6 = _mm512_sub_ps(bv_s21, bv_s20);
        curr_out = out + out_strides[5];
        STRI_2x512_S(curr_out, v_out_stride, v_out5, v_out6);
        // Output point 15: X(14)
        v_out14 = NEGATE_512_S(_mm512_add_ps(bv_s21, bv_s20));
        curr_out = out + out_strides[13];
        STRI_2x512_S(curr_out, v_out_stride, v_out13, v_out14);

        in = in + (v_in_stride << 4);
        out = out + (v_out_stride << 4);
    }
    // tail cases
    if (n & 8)
    {
        /* Standard DFT */
        __m256 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
               av_in8, av_in9;
        __m256 av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
               av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16,
               av_s17, av_s18, av_s19, av_s20, av_s21, av_s22, av_s23;
        __m256 av_t0, av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8,
               av_t9, av_t10, av_t11;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17, v_out18, v_out19;

        curr_in = in;
        curr_out = out;

        __m256 v256_CRTM_10_1 = _mm512_castps512_ps256(v_CRTM_10_1);
        __m256 v256_CRTM_10_2 = _mm512_castps512_ps256(v_CRTM_10_2);
        __m256 v256_CRTM_10_3 = _mm512_castps512_ps256(v_CRTM_10_3);
        __m256 v256_CRTM_10_4 = _mm512_castps512_ps256(v_CRTM_10_4);

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
        // Input point 19: x(18)
        curr_in = in + in_strides[18];
        LDR_256_S(curr_in, v_in_stride, av_in9);

        av_s0 = _mm256_add_ps(av_in0, av_in5);
        av_s1 = _mm256_sub_ps(av_in0, av_in5);
        av_s2 = _mm256_add_ps(av_in1, av_in9);
        av_s3 = _mm256_sub_ps(av_in1, av_in9);
        av_s4 = _mm256_add_ps(av_in2, av_in3);
        av_s5 = _mm256_sub_ps(av_in2, av_in3);
        av_s6 = _mm256_add_ps(av_in4, av_in6);
        av_s7 = _mm256_sub_ps(av_in4, av_in6);
        av_s8 = _mm256_add_ps(av_in7, av_in8);
        av_s9 = _mm256_sub_ps(av_in7, av_in8);

        av_s10 = _mm256_add_ps(av_s2, av_s6);
        av_s14 = _mm256_add_ps(av_s4, av_s8);
        av_s18 = _mm256_add_ps(av_s10,av_s14);
        // Output pt 1: X(0)
        v_out0 = _mm256_add_ps(av_s0, av_s18);
        STR_256_S(curr_out, v_out_stride, v_out0);

        av_s11 = _mm256_sub_ps(av_s2, av_s6);
        av_s17 = _mm256_sub_ps(av_s5, av_s9);
        av_s20 = _mm256_add_ps(av_s11, av_s17);
        av_s21 = _mm256_sub_ps(av_s11, av_s17);
        // Output point 20: X(19)
        v_out19 = _mm256_sub_ps(av_s1, av_s21);
        curr_out = out + out_strides[19];
        STR_256_S(curr_out, v_out_stride, v_out19);

        av_t0  = _mm256_mul_ps(v256_CRTM_10_2, av_s21);
        av_t1  = _mm256_mul_ps(v256_CRTM_10_1, av_s20);
        av_s22 = _mm256_add_ps(av_t0, av_s1);
        // Output point 4: X(3)
        v_out3 = _mm256_add_ps(av_s22, av_t1);
        // Output point 12: X(11)
        v_out11 = _mm256_sub_ps(av_s22, av_t1);

        av_s12 = _mm256_add_ps(av_s3, av_s7);
        av_s15 = _mm256_sub_ps(av_s4, av_s8);
        av_t4  = _mm256_mul_ps(v256_CRTM_10_3, av_s12);
        av_t10 = _mm256_mul_ps(v256_CRTM_10_4, av_s15);
        // Output point 5: X(4)
        v_out4 = NEGATE_256_S(_mm256_add_ps(av_t4, av_t10));
        curr_out = out + out_strides[3];
        STRI_2x256_S(curr_out, v_out_stride, v_out3, v_out4);

        av_t6 = _mm256_mul_ps(v256_CRTM_10_3, av_s15);
        av_t8 = _mm256_mul_ps(v256_CRTM_10_4, av_s12);
        // Output point 13: X(12)
        v_out12 = _mm256_sub_ps(av_t6, av_t8);
        curr_out = out + out_strides[11];
        STRI_2x256_S(curr_out, v_out_stride, v_out11, v_out12);

        av_s13 = _mm256_sub_ps(av_s7, av_s3);
        av_s16 = _mm256_add_ps(av_s5, av_s9);
        av_t5  = _mm256_mul_ps(v256_CRTM_10_3, av_s13);
        av_t11 = _mm256_mul_ps(v256_CRTM_10_4, av_s16);
        // Output point 17: X(16)
        v_out16 = _mm256_add_ps(av_t11, av_t5);

        av_t7 = _mm256_mul_ps(v256_CRTM_10_3, av_s16);
        av_t9 = _mm256_mul_ps(v256_CRTM_10_4, av_s13);
        // Output point 9: X(8)
        v_out8 = _mm256_sub_ps(av_t9, av_t7);

        av_s19 = _mm256_sub_ps(av_s10, av_s14);
        av_t2  = _mm256_mul_ps(v256_CRTM_10_2, av_s18);
        av_t3  = _mm256_mul_ps(v256_CRTM_10_1, av_s19);
        av_s23 = _mm256_sub_ps(av_s0, av_t2);
        // Output point 8: X(7)
        v_out7 = _mm256_add_ps(av_s23, av_t3);
        curr_out = out + out_strides[7];
        STRI_2x256_S(curr_out, v_out_stride, v_out7, v_out8);
        // Output point 16: X(15)
        v_out15 = _mm256_sub_ps(av_s23, av_t3);
        curr_out = out + out_strides[15];
        STRI_2x256_S(curr_out, v_out_stride, v_out15, v_out16);

        /* Shifted DFT */
        __m256 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8, bv_in9;
        __m256 bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
               bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16,
               bv_s17, bv_s18, bv_s19, bv_s20, bv_s21;
        __m256 bv_t0, bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8,
               bv_t9, bv_t10, bv_t11;

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
        // Input point 20: x(19)
        curr_in = in + in_strides[19];
        LDR_256_S(curr_in, v_in_stride, bv_in9);

        bv_s0 = _mm256_add_ps(bv_in1, bv_in9);
        bv_s1 = _mm256_sub_ps(bv_in1, bv_in9);
        bv_s2 = _mm256_add_ps(bv_in2, bv_in8);
        bv_s3 = _mm256_sub_ps(bv_in2, bv_in8);
        bv_s4 = _mm256_add_ps(bv_in3, bv_in7);
        bv_s5 = _mm256_sub_ps(bv_in3, bv_in7);
        bv_s6 = _mm256_add_ps(bv_in4, bv_in6);
        bv_s7 = _mm256_sub_ps(bv_in4, bv_in6);

        bv_s8  = _mm256_add_ps(bv_s0, bv_s4);
        bv_s9  = _mm256_sub_ps(bv_s0, bv_s4);
        bv_s10 = _mm256_add_ps(bv_s3, bv_s7);
        bv_s11 = _mm256_sub_ps(bv_s3, bv_s7);
        // Output point 10: X(9)
        v_out9 = _mm256_sub_ps(bv_in0, bv_s11);
        // Output point 11: X(10)
        v_out10 = NEGATE_256_S(_mm256_add_ps(bv_s9, bv_in5));
        curr_out = out + out_strides[9];
        STRI_2x256_S(curr_out, v_out_stride, v_out9, v_out10);

        bv_t0 = _mm256_mul_ps(v256_CRTM_10_2, bv_s11);
        bv_t1 = _mm256_mul_ps(v256_CRTM_10_1, bv_s10);
        bv_t2 = _mm256_mul_ps(v256_CRTM_10_1, bv_s8);
        bv_t3 = _mm256_mul_ps(v256_CRTM_10_2, bv_s9);

        bv_s12 = _mm256_add_ps(bv_in0, bv_t0);
        bv_s13 = _mm256_sub_ps(bv_t3, bv_in5);

        bv_t4  = _mm256_mul_ps(v256_CRTM_10_4, bv_s1);
        bv_t5  = _mm256_mul_ps(v256_CRTM_10_3, bv_s5);
        bv_t6  = _mm256_mul_ps(v256_CRTM_10_4, bv_s6);
        bv_t7  = _mm256_mul_ps(v256_CRTM_10_3, bv_s2);
        bv_t8  = _mm256_mul_ps(v256_CRTM_10_4, bv_s5);
        bv_t9  = _mm256_mul_ps(v256_CRTM_10_3, bv_s1);
        bv_t10 = _mm256_mul_ps(v256_CRTM_10_4, bv_s2);
        bv_t11 = _mm256_mul_ps(v256_CRTM_10_3, bv_s6);

        bv_s14 = _mm256_add_ps(bv_s12, bv_t1);
        bv_s15 = _mm256_add_ps(bv_t4, bv_t5);
        // Output point 2: X(1)
        v_out1 = _mm256_add_ps(bv_s14, bv_s15);
        // Output point 18: X(17)
        v_out17 = _mm256_sub_ps(bv_s14, bv_s15);

        bv_s16 = _mm256_sub_ps(bv_s13, bv_t2);
        bv_s17 = _mm256_add_ps(bv_t6, bv_t7);
        // Output point 3: X(2)
        v_out2 = _mm256_sub_ps(bv_s16, bv_s17);
        curr_out = out + out_strides[1];
        STRI_2x256_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 19: X(18)
        v_out18 = _mm256_add_ps(bv_s16, bv_s17);
        curr_out = out + out_strides[17];
        STRI_2x256_S(curr_out, v_out_stride, v_out17, v_out18);

        bv_s18 = _mm256_sub_ps(bv_s12, bv_t1);
        bv_s19 = _mm256_sub_ps(bv_t9, bv_t8);
        // Output point 6: X(5)
        v_out5 = _mm256_add_ps(bv_s18, bv_s19);
        // Output point 14: X(13)
        v_out13 = _mm256_sub_ps(bv_s18, bv_s19);

        bv_s20 = _mm256_add_ps(bv_s13, bv_t2);
        bv_s21 = _mm256_sub_ps(bv_t11, bv_t10);
        // Output point 7: X(6)
        v_out6 = _mm256_sub_ps(bv_s21, bv_s20);
        curr_out = out + out_strides[5];
        STRI_2x256_S(curr_out, v_out_stride, v_out5, v_out6);
        // Output point 15: X(14)
        v_out14 = NEGATE_256_S(_mm256_add_ps(bv_s21, bv_s20));
        curr_out = out + out_strides[13];
        STRI_2x256_S(curr_out, v_out_stride, v_out13, v_out14);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (n & 4)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
               av_in8, av_in9;
        __m128 av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
               av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16,
               av_s17, av_s18, av_s19, av_s20, av_s21, av_s22, av_s23;
        __m128 av_t0, av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8,
               av_t9, av_t10, av_t11;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17, v_out18, v_out19;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_10_1 = _mm512_castps512_ps128(v_CRTM_10_1);
        __m128 v128_CRTM_10_2 = _mm512_castps512_ps128(v_CRTM_10_2);
        __m128 v128_CRTM_10_3 = _mm512_castps512_ps128(v_CRTM_10_3);
        __m128 v128_CRTM_10_4 = _mm512_castps512_ps128(v_CRTM_10_4);

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
        // Input point 19: x(18)
        curr_in = in + in_strides[18];
        LDR_128_S(curr_in, v_in_stride, av_in9);

        av_s0 = _mm_add_ps(av_in0, av_in5);
        av_s1 = _mm_sub_ps(av_in0, av_in5);
        av_s2 = _mm_add_ps(av_in1, av_in9);
        av_s3 = _mm_sub_ps(av_in1, av_in9);
        av_s4 = _mm_add_ps(av_in2, av_in3);
        av_s5 = _mm_sub_ps(av_in2, av_in3);
        av_s6 = _mm_add_ps(av_in4, av_in6);
        av_s7 = _mm_sub_ps(av_in4, av_in6);
        av_s8 = _mm_add_ps(av_in7, av_in8);
        av_s9 = _mm_sub_ps(av_in7, av_in8);

        av_s10 = _mm_add_ps(av_s2, av_s6);
        av_s14 = _mm_add_ps(av_s4, av_s8);
        av_s18 = _mm_add_ps(av_s10,av_s14);
        // Output pt 1: X(0)
        v_out0 = _mm_add_ps(av_s0, av_s18);
        STR_128_S(curr_out, v_out_stride, v_out0);

        av_s11 = _mm_sub_ps(av_s2, av_s6);
        av_s17 = _mm_sub_ps(av_s5, av_s9);
        av_s20 = _mm_add_ps(av_s11, av_s17);
        av_s21 = _mm_sub_ps(av_s11, av_s17);
        // Output point 20: X(19)
        v_out19 = _mm_sub_ps(av_s1, av_s21);
        curr_out = out + out_strides[19];
        STR_128_S(curr_out, v_out_stride, v_out19);

        av_t0  = _mm_mul_ps(v128_CRTM_10_2, av_s21);
        av_t1  = _mm_mul_ps(v128_CRTM_10_1, av_s20);
        av_s22 = _mm_add_ps(av_t0, av_s1);
        // Output point 4: X(3)
        v_out3 = _mm_add_ps(av_s22, av_t1);
        // Output point 12: X(11)
        v_out11 = _mm_sub_ps(av_s22, av_t1);

        av_s12 = _mm_add_ps(av_s3, av_s7);
        av_s15 = _mm_sub_ps(av_s4, av_s8);
        av_t4  = _mm_mul_ps(v128_CRTM_10_3, av_s12);
        av_t10 = _mm_mul_ps(v128_CRTM_10_4, av_s15);
        // Output point 5: X(4)
        v_out4 = NEGATE_128_S(_mm_add_ps(av_t4, av_t10));
        curr_out = out + out_strides[3];
        STRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);

        av_t6 = _mm_mul_ps(v128_CRTM_10_3, av_s15);
        av_t8 = _mm_mul_ps(v128_CRTM_10_4, av_s12);
        // Output point 13: X(12)
        v_out12 = _mm_sub_ps(av_t6, av_t8);
        curr_out = out + out_strides[11];
        STRI_2x128_S(curr_out, v_out_stride, v_out11, v_out12);

        av_s13 = _mm_sub_ps(av_s7, av_s3);
        av_s16 = _mm_add_ps(av_s5, av_s9);
        av_t5  = _mm_mul_ps(v128_CRTM_10_3, av_s13);
        av_t11 = _mm_mul_ps(v128_CRTM_10_4, av_s16);
        // Output point 17: X(16)
        v_out16 = _mm_add_ps(av_t11, av_t5);

        av_t7 = _mm_mul_ps(v128_CRTM_10_3, av_s16);
        av_t9 = _mm_mul_ps(v128_CRTM_10_4, av_s13);
        // Output point 9: X(8)
        v_out8 = _mm_sub_ps(av_t9, av_t7);

        av_s19 = _mm_sub_ps(av_s10, av_s14);
        av_t2  = _mm_mul_ps(v128_CRTM_10_2, av_s18);
        av_t3  = _mm_mul_ps(v128_CRTM_10_1, av_s19);
        av_s23 = _mm_sub_ps(av_s0, av_t2);
        // Output point 8: X(7)
        v_out7 = _mm_add_ps(av_s23, av_t3);
        curr_out = out + out_strides[7];
        STRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);
        // Output point 16: X(15)
        v_out15 = _mm_sub_ps(av_s23, av_t3);
        curr_out = out + out_strides[15];
        STRI_2x128_S(curr_out, v_out_stride, v_out15, v_out16);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8, bv_in9;
        __m128 bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
               bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16,
               bv_s17, bv_s18, bv_s19, bv_s20, bv_s21;
        __m128 bv_t0, bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8,
               bv_t9, bv_t10, bv_t11;

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
        // Input point 20: x(19)
        curr_in = in + in_strides[19];
        LDR_128_S(curr_in, v_in_stride, bv_in9);

        bv_s0 = _mm_add_ps(bv_in1, bv_in9);
        bv_s1 = _mm_sub_ps(bv_in1, bv_in9);
        bv_s2 = _mm_add_ps(bv_in2, bv_in8);
        bv_s3 = _mm_sub_ps(bv_in2, bv_in8);
        bv_s4 = _mm_add_ps(bv_in3, bv_in7);
        bv_s5 = _mm_sub_ps(bv_in3, bv_in7);
        bv_s6 = _mm_add_ps(bv_in4, bv_in6);
        bv_s7 = _mm_sub_ps(bv_in4, bv_in6);

        bv_s8  = _mm_add_ps(bv_s0, bv_s4);
        bv_s9  = _mm_sub_ps(bv_s0, bv_s4);
        bv_s10 = _mm_add_ps(bv_s3, bv_s7);
        bv_s11 = _mm_sub_ps(bv_s3, bv_s7);
        // Output point 10: X(9)
        v_out9 = _mm_sub_ps(bv_in0, bv_s11);
        // Output point 11: X(10)
        v_out10 = NEGATE_128_S(_mm_add_ps(bv_s9, bv_in5));
        curr_out = out + out_strides[9];
        STRI_2x128_S(curr_out, v_out_stride, v_out9, v_out10);

        bv_t0 = _mm_mul_ps(v128_CRTM_10_2, bv_s11);
        bv_t1 = _mm_mul_ps(v128_CRTM_10_1, bv_s10);
        bv_t2 = _mm_mul_ps(v128_CRTM_10_1, bv_s8);
        bv_t3 = _mm_mul_ps(v128_CRTM_10_2, bv_s9);

        bv_s12 = _mm_add_ps(bv_in0, bv_t0);
        bv_s13 = _mm_sub_ps(bv_t3, bv_in5);

        bv_t4  = _mm_mul_ps(v128_CRTM_10_4, bv_s1);
        bv_t5  = _mm_mul_ps(v128_CRTM_10_3, bv_s5);
        bv_t6  = _mm_mul_ps(v128_CRTM_10_4, bv_s6);
        bv_t7  = _mm_mul_ps(v128_CRTM_10_3, bv_s2);
        bv_t8  = _mm_mul_ps(v128_CRTM_10_4, bv_s5);
        bv_t9  = _mm_mul_ps(v128_CRTM_10_3, bv_s1);
        bv_t10 = _mm_mul_ps(v128_CRTM_10_4, bv_s2);
        bv_t11 = _mm_mul_ps(v128_CRTM_10_3, bv_s6);

        bv_s14 = _mm_add_ps(bv_s12, bv_t1);
        bv_s15 = _mm_add_ps(bv_t4, bv_t5);
        // Output point 2: X(1)
        v_out1 = _mm_add_ps(bv_s14, bv_s15);
        // Output point 18: X(17)
        v_out17 = _mm_sub_ps(bv_s14, bv_s15);

        bv_s16 = _mm_sub_ps(bv_s13, bv_t2);
        bv_s17 = _mm_add_ps(bv_t6, bv_t7);
        // Output point 3: X(2)
        v_out2 = _mm_sub_ps(bv_s16, bv_s17);
        curr_out = out + out_strides[1];
        STRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 19: X(18)
        v_out18 = _mm_add_ps(bv_s16, bv_s17);
        curr_out = out + out_strides[17];
        STRI_2x128_S(curr_out, v_out_stride, v_out17, v_out18);

        bv_s18 = _mm_sub_ps(bv_s12, bv_t1);
        bv_s19 = _mm_sub_ps(bv_t9, bv_t8);
        // Output point 6: X(5)
        v_out5 = _mm_add_ps(bv_s18, bv_s19);
        // Output point 14: X(13)
        v_out13 = _mm_sub_ps(bv_s18, bv_s19);

        bv_s20 = _mm_add_ps(bv_s13, bv_t2);
        bv_s21 = _mm_sub_ps(bv_t11, bv_t10);
        // Output point 7: X(6)
        v_out6 = _mm_sub_ps(bv_s21, bv_s20);
        curr_out = out + out_strides[5];
        STRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);
        // Output point 15: X(14)
        v_out14 = NEGATE_128_S(_mm_add_ps(bv_s21, bv_s20));
        curr_out = out + out_strides[13];
        STRI_2x128_S(curr_out, v_out_stride, v_out13, v_out14);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (n & 2)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
               av_in8, av_in9;
        __m128 av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
               av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16,
               av_s17, av_s18, av_s19, av_s20, av_s21, av_s22, av_s23;
        __m128 av_t0, av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8,
               av_t9, av_t10, av_t11;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17, v_out18, v_out19;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_10_1 = _mm512_castps512_ps128(v_CRTM_10_1);
        __m128 v128_CRTM_10_2 = _mm512_castps512_ps128(v_CRTM_10_2);
        __m128 v128_CRTM_10_3 = _mm512_castps512_ps128(v_CRTM_10_3);
        __m128 v128_CRTM_10_4 = _mm512_castps512_ps128(v_CRTM_10_4);

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
        // Input point 19: x(18)
        curr_in = in + in_strides[18];
        LDHR_128_S(curr_in, v_in_stride, av_in9);

        av_s0 = _mm_add_ps(av_in0, av_in5);
        av_s1 = _mm_sub_ps(av_in0, av_in5);
        av_s2 = _mm_add_ps(av_in1, av_in9);
        av_s3 = _mm_sub_ps(av_in1, av_in9);
        av_s4 = _mm_add_ps(av_in2, av_in3);
        av_s5 = _mm_sub_ps(av_in2, av_in3);
        av_s6 = _mm_add_ps(av_in4, av_in6);
        av_s7 = _mm_sub_ps(av_in4, av_in6);
        av_s8 = _mm_add_ps(av_in7, av_in8);
        av_s9 = _mm_sub_ps(av_in7, av_in8);

        av_s10 = _mm_add_ps(av_s2, av_s6);
        av_s14 = _mm_add_ps(av_s4, av_s8);
        av_s18 = _mm_add_ps(av_s10,av_s14);
        // Output pt 1: X(0)
        v_out0 = _mm_add_ps(av_s0, av_s18);
        STHR_128_S(curr_out, v_out_stride, v_out0);

        av_s11 = _mm_sub_ps(av_s2, av_s6);
        av_s17 = _mm_sub_ps(av_s5, av_s9);
        av_s20 = _mm_add_ps(av_s11, av_s17);
        av_s21 = _mm_sub_ps(av_s11, av_s17);
        // Output point 20: X(19)
        v_out19 = _mm_sub_ps(av_s1, av_s21);
        curr_out = out + out_strides[19];
        STHR_128_S(curr_out, v_out_stride, v_out19);

        av_t0  = _mm_mul_ps(v128_CRTM_10_2, av_s21);
        av_t1  = _mm_mul_ps(v128_CRTM_10_1, av_s20);
        av_s22 = _mm_add_ps(av_t0, av_s1);
        // Output point 4: X(3)
        v_out3 = _mm_add_ps(av_s22, av_t1);
        // Output point 12: X(11)
        v_out11 = _mm_sub_ps(av_s22, av_t1);

        av_s12 = _mm_add_ps(av_s3, av_s7);
        av_s15 = _mm_sub_ps(av_s4, av_s8);
        av_t4  = _mm_mul_ps(v128_CRTM_10_3, av_s12);
        av_t10 = _mm_mul_ps(v128_CRTM_10_4, av_s15);
        // Output point 5: X(4)
        v_out4 = NEGATE_128_S(_mm_add_ps(av_t4, av_t10));
        curr_out = out + out_strides[3];
        STHRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);

        av_t6 = _mm_mul_ps(v128_CRTM_10_3, av_s15);
        av_t8 = _mm_mul_ps(v128_CRTM_10_4, av_s12);
        // Output point 13: X(12)
        v_out12 = _mm_sub_ps(av_t6, av_t8);
        curr_out = out + out_strides[11];
        STHRI_2x128_S(curr_out, v_out_stride, v_out11, v_out12);

        av_s13 = _mm_sub_ps(av_s7, av_s3);
        av_s16 = _mm_add_ps(av_s5, av_s9);
        av_t5  = _mm_mul_ps(v128_CRTM_10_3, av_s13);
        av_t11 = _mm_mul_ps(v128_CRTM_10_4, av_s16);
        // Output point 17: X(16)
        v_out16 = _mm_add_ps(av_t11, av_t5);

        av_t7 = _mm_mul_ps(v128_CRTM_10_3, av_s16);
        av_t9 = _mm_mul_ps(v128_CRTM_10_4, av_s13);
        // Output point 9: X(8)
        v_out8 = _mm_sub_ps(av_t9, av_t7);

        av_s19 = _mm_sub_ps(av_s10, av_s14);
        av_t2  = _mm_mul_ps(v128_CRTM_10_2, av_s18);
        av_t3  = _mm_mul_ps(v128_CRTM_10_1, av_s19);
        av_s23 = _mm_sub_ps(av_s0, av_t2);
        // Output point 8: X(7)
        v_out7 = _mm_add_ps(av_s23, av_t3);
        curr_out = out + out_strides[7];
        STHRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);
        // Output point 16: X(15)
        v_out15 = _mm_sub_ps(av_s23, av_t3);
        curr_out = out + out_strides[15];
        STHRI_2x128_S(curr_out, v_out_stride, v_out15, v_out16);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8, bv_in9;
        __m128 bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
               bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16,
               bv_s17, bv_s18, bv_s19, bv_s20, bv_s21;
        __m128 bv_t0, bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8,
               bv_t9, bv_t10, bv_t11;

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
        // Input point 20: x(19)
        curr_in = in + in_strides[19];
        LDHR_128_S(curr_in, v_in_stride, bv_in9);

        bv_s0 = _mm_add_ps(bv_in1, bv_in9);
        bv_s1 = _mm_sub_ps(bv_in1, bv_in9);
        bv_s2 = _mm_add_ps(bv_in2, bv_in8);
        bv_s3 = _mm_sub_ps(bv_in2, bv_in8);
        bv_s4 = _mm_add_ps(bv_in3, bv_in7);
        bv_s5 = _mm_sub_ps(bv_in3, bv_in7);
        bv_s6 = _mm_add_ps(bv_in4, bv_in6);
        bv_s7 = _mm_sub_ps(bv_in4, bv_in6);

        bv_s8  = _mm_add_ps(bv_s0, bv_s4);
        bv_s9  = _mm_sub_ps(bv_s0, bv_s4);
        bv_s10 = _mm_add_ps(bv_s3, bv_s7);
        bv_s11 = _mm_sub_ps(bv_s3, bv_s7);
        // Output point 10: X(9)
        v_out9 = _mm_sub_ps(bv_in0, bv_s11);
        // Output point 11: X(10)
        v_out10 = NEGATE_128_S(_mm_add_ps(bv_s9, bv_in5));
        curr_out = out + out_strides[9];
        STHRI_2x128_S(curr_out, v_out_stride, v_out9, v_out10);

        bv_t0 = _mm_mul_ps(v128_CRTM_10_2, bv_s11);
        bv_t1 = _mm_mul_ps(v128_CRTM_10_1, bv_s10);
        bv_t2 = _mm_mul_ps(v128_CRTM_10_1, bv_s8);
        bv_t3 = _mm_mul_ps(v128_CRTM_10_2, bv_s9);

        bv_s12 = _mm_add_ps(bv_in0, bv_t0);
        bv_s13 = _mm_sub_ps(bv_t3, bv_in5);

        bv_t4  = _mm_mul_ps(v128_CRTM_10_4, bv_s1);
        bv_t5  = _mm_mul_ps(v128_CRTM_10_3, bv_s5);
        bv_t6  = _mm_mul_ps(v128_CRTM_10_4, bv_s6);
        bv_t7  = _mm_mul_ps(v128_CRTM_10_3, bv_s2);
        bv_t8  = _mm_mul_ps(v128_CRTM_10_4, bv_s5);
        bv_t9  = _mm_mul_ps(v128_CRTM_10_3, bv_s1);
        bv_t10 = _mm_mul_ps(v128_CRTM_10_4, bv_s2);
        bv_t11 = _mm_mul_ps(v128_CRTM_10_3, bv_s6);

        bv_s14 = _mm_add_ps(bv_s12, bv_t1);
        bv_s15 = _mm_add_ps(bv_t4, bv_t5);
        // Output point 2: X(1)
        v_out1 = _mm_add_ps(bv_s14, bv_s15);
        // Output point 18: X(17)
        v_out17 = _mm_sub_ps(bv_s14, bv_s15);

        bv_s16 = _mm_sub_ps(bv_s13, bv_t2);
        bv_s17 = _mm_add_ps(bv_t6, bv_t7);
        // Output point 3: X(2)
        v_out2 = _mm_sub_ps(bv_s16, bv_s17);
        curr_out = out + out_strides[1];
        STHRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 19: X(18)
        v_out18 = _mm_add_ps(bv_s16, bv_s17);
        curr_out = out + out_strides[17];
        STHRI_2x128_S(curr_out, v_out_stride, v_out17, v_out18);

        bv_s18 = _mm_sub_ps(bv_s12, bv_t1);
        bv_s19 = _mm_sub_ps(bv_t9, bv_t8);
        // Output point 6: X(5)
        v_out5 = _mm_add_ps(bv_s18, bv_s19);
        // Output point 14: X(13)
        v_out13 = _mm_sub_ps(bv_s18, bv_s19);

        bv_s20 = _mm_add_ps(bv_s13, bv_t2);
        bv_s21 = _mm_sub_ps(bv_t11, bv_t10);
        // Output point 7: X(6)
        v_out6 = _mm_sub_ps(bv_s21, bv_s20);
        curr_out = out + out_strides[5];
        STHRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);
        // Output point 15: X(14)
        v_out14 = NEGATE_128_S(_mm_add_ps(bv_s21, bv_s20));
        curr_out = out + out_strides[13];
        STHRI_2x128_S(curr_out, v_out_stride, v_out13, v_out14);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        /* Standard DFT */
        FLOAT a_in0, a_in1, a_in2, a_in3, a_in4, a_in5, a_in6, a_in7, a_in8,
              a_in9;
        FLOAT a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_s9, a_s10,
              a_s11, a_s12, a_s13, a_s14, a_s15, a_s16, a_s17, a_s18, a_s19,
              a_s20, a_s21, a_s22, a_s23, a_t0, a_t1, a_t2, a_t3, a_t4, a_t5,
              a_t6, a_t7, a_t8, a_t9, a_t10, a_t11;

        // Input point 1: x(0)
        a_in0 = *in;
        // Input point 3: x(2)
        a_in1 = in[in_strides[2]];
        // Input point 5: x(4)
        a_in2 = in[in_strides[4]];
        // Input point 7: x(6)
        a_in3 = in[in_strides[6]];
        // Input point 9: x(8)
        a_in4 = in[in_strides[8]];
        // Input point 11: x(10)
        a_in5 = in[in_strides[10]];
        // Input point 13: x(12)
        a_in6 = in[in_strides[12]];
        // Input point 15: x(14)
        a_in7 = in[in_strides[14]];
        // Input point 17: x(16)
        a_in8 = in[in_strides[16]];
        // Input point 19: x(18)
        a_in9 = in[in_strides[18]];

        a_s0 = a_in0 + a_in5;
        a_s1 = a_in0 - a_in5;
        a_s2 = a_in1 + a_in9;
        a_s3 = a_in1 - a_in9;
        a_s4 = a_in2 + a_in3;
        a_s5 = a_in2 - a_in3;
        a_s6 = a_in4 + a_in6;
        a_s7 = a_in4 - a_in6;
        a_s8 = a_in7 + a_in8;
        a_s9 = a_in7 - a_in8;

        a_s10 = a_s2 + a_s6;
        a_s14 = a_s4 + a_s8;
        a_s18 = a_s10 + a_s14;
        // Output point 1: X(0)
        *out = a_s0 + a_s18;

        a_s11 = a_s2 - a_s6;
        a_s17 = a_s5 - a_s9;
        a_s20 = a_s11 + a_s17;
        a_s21 = a_s11 - a_s17;
        // Output point 20: X(19)
        out[out_strides[19]] = a_s1 - a_s21;

        a_t0 = CRTM_10_2 * a_s21;
        a_t1 = CRTM_10_1 * a_s20;
        a_s22 = a_t0 + a_s1;
        // Output point 4: X(3)
        out[out_strides[3]] = a_s22 + a_t1;
        // Output point 12: X(11)
        out[out_strides[11]] = a_s22 - a_t1;

        a_s12 = a_s3 + a_s7;
        a_s15 = a_s4 - a_s8;
        a_t2 = CRTM_10_3 * a_s12;
        a_t3 = CRTM_10_4 * a_s15;
        // Output point 5: X(4)
        out[out_strides[4]] = -(a_t2 + a_t3);

        a_t4 = CRTM_10_3 * a_s15;
        a_t5 = CRTM_10_4 * a_s12;
        // Output point 13: X(12)
        out[out_strides[12]] = a_t4 - a_t5;

        a_s13 = a_s7 - a_s3;
        a_s16 = a_s5 + a_s9;
        a_t6 = CRTM_10_3 * a_s13;
        a_t7 = CRTM_10_4 * a_s16;
        // Output point 17: X(16)
        out[out_strides[16]] = a_t7 + a_t6;

        a_t8 = CRTM_10_3 * a_s16;
        a_t9 = CRTM_10_4 * a_s13;
        // Output point 9: X(8)
        out[out_strides[8]] = a_t9 - a_t8;

        a_s19 = a_s10 - a_s14;
        a_t10 = CRTM_10_2 * a_s18;
        a_t11 = CRTM_10_1 * a_s19;
        a_s23 = a_s0 - a_t10;
        // Output point 8: X(7)
        out[out_strides[7]] = a_s23 + a_t11;
        // Output point 16: X(15)
        out[out_strides[15]] = a_s23 - a_t11;

        /* Shifted DFT */
        FLOAT b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7, b_in8,
              b_in9;
        FLOAT b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9, b_s10,
              b_s11, b_s12, b_s13,b_s14, b_s15, b_s16, b_s17, b_s18, b_s19,
              b_s20, b_s21;
        FLOAT b_t0, b_t1, b_t2, b_t3, b_t4, b_t5, b_t6, b_t7, b_t8, b_t9, b_t10,
              b_t11;

        // Input point 2: x(1)
        b_in0 = in[in_strides[1]];
        // Input point 4: x(3)
        b_in1 = in[in_strides[3]];
        // Input point 6: x(5)
        b_in2 = in[in_strides[5]];
        // Input point 8: x(7)
        b_in3 = in[in_strides[7]];
        // Input point 10: x(9)
        b_in4 = in[in_strides[9]];
        // Input point 12: x(11)
        b_in5 = in[in_strides[11]];
        // Input point 14: x(13)
        b_in6 = in[in_strides[13]];
        // Input point 16: x(15)
        b_in7 = in[in_strides[15]];
        // Input point 18: x(17)
        b_in8 = in[in_strides[17]];
        // Input point 20: x(19)
        b_in9 = in[in_strides[19]];

        b_s0 = b_in1 + b_in9;
        b_s1 = b_in1 - b_in9;
        b_s2 = b_in2 + b_in8;
        b_s3 = b_in2 - b_in8;
        b_s4 = b_in3 + b_in7;
        b_s5 = b_in3 - b_in7;
        b_s6 = b_in4 + b_in6;
        b_s7 = b_in4 - b_in6;

        b_s8 = b_s0 + b_s4;
        b_s9 = b_s0 - b_s4;
        b_s10 = b_s3 + b_s7;
        b_s11 = b_s3 - b_s7;
        // Output point 11: X(10)
        out[out_strides[10]] = -(b_s9 + b_in5);
        // Output point 10: X(9)
        out[out_strides[9]] = b_in0 - b_s11;

        b_t0 = CRTM_10_2 * b_s11;
        b_t1 = CRTM_10_1 * b_s10;
        b_t2 = CRTM_10_1 * b_s8;
        b_t3 = CRTM_10_2 * b_s9;

        b_s12 = b_in0 + b_t0;
        b_s13 = b_t3 - b_in5;

        b_t4 = CRTM_10_4 * b_s1;
        b_t5 = CRTM_10_3 * b_s5;
        b_t6 = CRTM_10_4 * b_s6;
        b_t7 = CRTM_10_3 * b_s2;
        b_t8 = CRTM_10_4 * b_s5;
        b_t9 = CRTM_10_3 * b_s1;
        b_t10 = CRTM_10_4 * b_s2;
        b_t11 = CRTM_10_3 * b_s6;

        b_s14 = b_s12 + b_t1;
        b_s15 = b_t4 + b_t5;
        // Output point 2: X(1)
        out[out_strides[1]] = b_s14 + b_s15;
        // Output point 18: X(17)
        out[out_strides[17]] = b_s14 - b_s15;

        b_s16 = b_s13 - b_t2;
        b_s17 = b_t6 + b_t7;
        // Output point 3: X(2)
        out[out_strides[2]] = b_s16 - b_s17;
        // Output point 19: X(18)
        out[out_strides[18]] = b_s16 + b_s17;

        b_s18 = b_s12 - b_t1;
        b_s19 = b_t9 - b_t8;
        // Output point 6: X(5)
        out[out_strides[5]] = b_s18 + b_s19;
        // Output point 14: X(13)
        out[out_strides[13]] = b_s18 - b_s19;

        b_s20 = b_s13 + b_t2;
        b_s21 = b_t11 - b_t10;
        // Output point 7: X(6)
        out[out_strides[6]] = b_s21 - b_s20;
        // Output point 15: X(14)
        out[out_strides[14]] = -(b_s21 + b_s20);
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID r2hcf_rfft10avx512_fp32_bwd(VOID *in_real, VOID *in_imag,
                                        VOID *out_real, VOID *out_imag, INTP n,
                                        aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const FLOAT CRTM_10_1 = 1.118033988749894848204586834365638117720309180f;
    const FLOAT CRTM_10_2 = 0.500000000000000000000000000000000000000000000f;
    const FLOAT CRTM_10_3 = 2.000000000000000000000000000000000000000000000f;
    const FLOAT CRTM_10_4 = 1.175570504584946258337411909278145537195304875f;
    const FLOAT CRTM_10_5 = 1.902113032590307144232878666758764286811397268f;

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

    __m512 v_CRTM_10_1 = _mm512_set1_ps(CRTM_10_1);
    __m512 v_CRTM_10_2 = _mm512_set1_ps(CRTM_10_2);
    __m512 v_CRTM_10_3 = _mm512_set1_ps(CRTM_10_3);
    __m512 v_CRTM_10_4 = _mm512_set1_ps(CRTM_10_4);
    __m512 v_CRTM_10_5 = _mm512_set1_ps(CRTM_10_5);

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m512 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
               av_in8, av_in9;
        __m512 av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8, av_s9,
               av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16, av_s17,
               av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24;
        __m512 av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8, av_t9,
               av_t10, av_t11, av_t12, av_t13, av_t14;
        __m512 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17, v_out18, v_out19;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_512_S(curr_in, v_in_stride, av_in0);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x512_S(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in + in_strides[7];
        LDRI_2x512_S(curr_in, v_in_stride, av_in3, av_in4);
        // Input point 12: x(11) & Input point 13: x(12)
        curr_in = in + in_strides[11];
        LDRI_2x512_S(curr_in, v_in_stride, av_in5, av_in6);
        // Input point 16: x(15) & Input point 17: x(16)
        curr_in = in + in_strides[15];
        LDRI_2x512_S(curr_in, v_in_stride, av_in7, av_in8);
        // Input point 20: x(19)
        curr_in = in + in_strides[19];
        LDR_512_S(curr_in, v_in_stride, av_in9);

        av_s1  = _mm512_add_ps(av_in0, av_in9);
        av_s2  = _mm512_sub_ps(av_in0, av_in9);
        av_s3  = _mm512_add_ps(av_in1, av_in7);
        av_s4  = _mm512_sub_ps(av_in1, av_in7);
        av_s5  = _mm512_add_ps(av_in2, av_in8);
        av_s6  = _mm512_sub_ps(av_in2, av_in8);
        av_t11 = _mm512_mul_ps(v_CRTM_10_4, av_s6);
        av_s9  = _mm512_add_ps(av_in4, av_in6);

        av_t9  = _mm512_mul_ps(v_CRTM_10_4, av_s9);
        av_t10 = _mm512_mul_ps(v_CRTM_10_5, av_s5);
        av_s23 = _mm512_sub_ps(av_t9, av_t10);
        av_s10 = _mm512_sub_ps(av_in4, av_in6);
        av_t12 = _mm512_mul_ps(v_CRTM_10_5, av_s10);
        av_s24 = _mm512_sub_ps(av_t12, av_t11);

        av_s7  = _mm512_add_ps(av_in3, av_in5);
        av_s8  = _mm512_sub_ps(av_in3, av_in5);
        av_s14 = _mm512_sub_ps(av_s4, av_s8);
        av_t14 = _mm512_mul_ps(v_CRTM_10_3, av_s14);
        // Output point 11: X(10)
        v_out10 = _mm512_sub_ps(av_s2, av_t14);
        curr_out = out + out_strides[10];
        STR_512_S(curr_out, v_out_stride, v_out10);

        av_s11 = _mm512_add_ps(av_s3, av_s7);
        av_t13 = _mm512_mul_ps(v_CRTM_10_3, av_s11);
        // Output point 1: X(0)
        v_out0 = _mm512_add_ps(av_s1, av_t13);
        curr_out = out + out_strides[0];
        STR_512_S(curr_out, v_out_stride, v_out0);

        av_t1  = _mm512_mul_ps(v_CRTM_10_2, av_s11);
        av_s16 = _mm512_sub_ps(av_s1, av_t1);
        av_s12 = _mm512_sub_ps(av_s3, av_s7);
        av_t3  = _mm512_mul_ps(v_CRTM_10_1, av_s12);
        av_s19 = _mm512_add_ps(av_s16, av_t3);
        av_s20 = _mm512_sub_ps(av_s16, av_t3);
        // Output point 9: X(8)
        v_out8 = _mm512_add_ps(av_s20, av_s24);
        curr_out = out + out_strides[8];
        STR_512_S(curr_out, v_out_stride, v_out8);
        // Output point 13: X(12)
        v_out12 = _mm512_sub_ps(av_s20, av_s24);
        curr_out = out + out_strides[12];
        STR_512_S(curr_out, v_out_stride, v_out12);

        av_s13 = _mm512_add_ps(av_s4, av_s8);
        av_t4  = _mm512_mul_ps(v_CRTM_10_1, av_s13);
        av_t2  = _mm512_mul_ps(v_CRTM_10_2, av_s14);
        av_s15 = _mm512_add_ps(av_s2, av_t2);
        av_s17 = _mm512_add_ps(av_s15, av_t4);
        av_s18 = _mm512_sub_ps(av_s15, av_t4);
        // Output point 7: X(6)
        v_out6 = _mm512_add_ps(av_s18, av_s23);
        curr_out = out + out_strides[6];
        STR_512_S(curr_out, v_out_stride, v_out6);
        // Output point 15: X(14)
        v_out14 = _mm512_sub_ps(av_s18, av_s23);
        curr_out = out + out_strides[14];
        STR_512_S(curr_out, v_out_stride, v_out14);

        av_t5  = _mm512_mul_ps(v_CRTM_10_4, av_s5);
        av_t6  = _mm512_mul_ps(v_CRTM_10_5, av_s9);
        av_s21 = _mm512_add_ps(av_t5, av_t6);
        // Output point 3: X(2)
        v_out2 = _mm512_sub_ps(av_s17, av_s21);
        curr_out = out + out_strides[2];
        STR_512_S(curr_out, v_out_stride, v_out2);
        // Output point 19: X(18)
        v_out18 = _mm512_add_ps(av_s17, av_s21);
        curr_out = out + out_strides[18];
        STR_512_S(curr_out, v_out_stride, v_out18);

        av_t7  = _mm512_mul_ps(v_CRTM_10_4, av_s10);
        av_t8  = _mm512_mul_ps(v_CRTM_10_5, av_s6);
        av_s22 = _mm512_add_ps(av_t7, av_t8);
        // Output point 5: X(4)
        v_out4 = _mm512_sub_ps(av_s19, av_s22);
        curr_out = out + out_strides[4];
        STR_512_S(curr_out, v_out_stride, v_out4);
        // Output point 17: X(16)
        v_out16 = _mm512_add_ps(av_s19, av_s22);
        curr_out = out + out_strides[16];
        STR_512_S(curr_out, v_out_stride, v_out16);

        /* Shifted DFT */
        __m512 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8, bv_in9;
        __m512 bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
               bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17,
               bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24;
        __m512 bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
               bv_t10, bv_t11, bv_t12, bv_t13, bv_t14;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x512_S(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDRI_2x512_S(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDRI_2x512_S(curr_in, v_in_stride, bv_in4, bv_in5);
        // Input point 14: x(13) & Input point 15: x(14)
        curr_in = in + in_strides[13];
        LDRI_2x512_S(curr_in, v_in_stride, bv_in6, bv_in7);
        // Input point 18: x(17) & Input point 19: x(18)
        curr_in = in + in_strides[17];
        LDRI_2x512_S(curr_in, v_in_stride, bv_in8, bv_in9);

        bv_s1 = _mm512_add_ps(bv_in0, bv_in8);
        bv_s2 = _mm512_sub_ps(bv_in0, bv_in8);
        bv_s3 = _mm512_add_ps(bv_in1, bv_in9);
        bv_s4 = _mm512_sub_ps(bv_in1, bv_in9);
        bv_s5 = _mm512_add_ps(bv_in2, bv_in6);
        bv_s6 = _mm512_sub_ps(bv_in2, bv_in6);
        bv_s7 = _mm512_add_ps(bv_in3, bv_in7);
        bv_s8 = _mm512_sub_ps(bv_in3, bv_in7);

        bv_s9  = _mm512_add_ps(bv_s1, bv_s5);
        bv_s10 = _mm512_sub_ps(bv_s1, bv_s5);
        bv_s11 = _mm512_add_ps(bv_s3, bv_s7);
        bv_s12 = _mm512_sub_ps(bv_s3, bv_s7);

        bv_t1  = _mm512_mul_ps(v_CRTM_10_1, bv_s11);
        bv_t2  = _mm512_mul_ps(v_CRTM_10_2, bv_s12);
        bv_t3  = _mm512_mul_ps(v_CRTM_10_1, bv_s10);
        bv_t4  = _mm512_mul_ps(v_CRTM_10_2, bv_s9);
        bv_t5  = _mm512_mul_ps(v_CRTM_10_3, bv_in4);
        bv_t6  = _mm512_mul_ps(v_CRTM_10_3, bv_in5);
        bv_s23 = _mm512_add_ps(bv_s9, bv_s9);
        // Output point 2: X(1)
        v_out1 = _mm512_add_ps(bv_t5, bv_s23);
        curr_out = out + out_strides[1];
        STR_512_S(curr_out, v_out_stride, v_out1);

        bv_s24 = _mm512_add_ps(bv_s12, bv_s12);
        // Output point 12: X(11)
        v_out11 = NEGATE_512_S(_mm512_add_ps(bv_t6, bv_s24));
        curr_out = out + out_strides[11];
        STR_512_S(curr_out, v_out_stride, v_out11);

        bv_s13 = _mm512_sub_ps(bv_t2, bv_t6);
        bv_s14 = _mm512_sub_ps(bv_t4, bv_t5);

        bv_s15 = _mm512_sub_ps(bv_s13, bv_t1);
        bv_s16 = _mm512_add_ps(bv_s13, bv_t1);
        bv_s17 = _mm512_add_ps(bv_t3, bv_s14);
        bv_s18 = _mm512_sub_ps(bv_t3, bv_s14);

        bv_t7  = _mm512_mul_ps(v_CRTM_10_5, bv_s2);
        bv_t8  = _mm512_mul_ps(v_CRTM_10_5, bv_s4);
        bv_t9  = _mm512_mul_ps(v_CRTM_10_5, bv_s6);
        bv_t10 = _mm512_mul_ps(v_CRTM_10_5, bv_s8);
        bv_t11 = _mm512_mul_ps(v_CRTM_10_4, bv_s2);
        bv_t12 = _mm512_mul_ps(v_CRTM_10_4, bv_s4);
        bv_t13 = _mm512_mul_ps(v_CRTM_10_4, bv_s6);
        bv_t14 = _mm512_mul_ps(v_CRTM_10_4, bv_s8);

        bv_s19 = _mm512_add_ps(bv_t7, bv_t13);
        // Output point 4: X(3)
        v_out3 = _mm512_add_ps(bv_s15, bv_s19);
        curr_out = out + out_strides[3];
        STR_512_S(curr_out, v_out_stride, v_out3);
        // Output point 20: X(19)
        v_out19 = _mm512_sub_ps(bv_s15, bv_s19);
        curr_out = out + out_strides[19];
        STR_512_S(curr_out, v_out_stride, v_out19);

        bv_s20 = _mm512_add_ps(bv_t10, bv_t12);
        // Output point 6: X(5)
        v_out5 = _mm512_sub_ps(bv_s17, bv_s20);
        curr_out = out + out_strides[5];
        STR_512_S(curr_out, v_out_stride, v_out5);
        // Output point 18: X(17)
        v_out17 = NEGATE_512_S(_mm512_add_ps(bv_s17, bv_s20));
        curr_out = out + out_strides[17];
        STR_512_S(curr_out, v_out_stride, v_out17);

        bv_s21 = _mm512_sub_ps(bv_t11, bv_t9);
        // Output point 8: X(7)
        v_out7 = _mm512_sub_ps(bv_s21, bv_s16);
        curr_out = out + out_strides[7];
        STR_512_S(curr_out, v_out_stride, v_out7);
        // Output point 16: X(15)
        v_out15 = NEGATE_512_S(_mm512_add_ps(bv_s16, bv_s21));
        curr_out = out + out_strides[15];
        STR_512_S(curr_out, v_out_stride, v_out15);

        bv_s22 = _mm512_sub_ps(bv_t14, bv_t8);
        // Output point 10: X(9)
        v_out9 = _mm512_add_ps(bv_s22, bv_s18);
        curr_out = out + out_strides[9];
        STR_512_S(curr_out, v_out_stride, v_out9);
        // Output point 14: X(13)
        v_out13 = _mm512_sub_ps(bv_s22, bv_s18);
        curr_out = out + out_strides[13];
        STR_512_S(curr_out, v_out_stride, v_out13);

        in = in + (v_in_stride << 4);
        out = out + (v_out_stride << 4);
    }
    // tail cases
    if (n & 8)
    {
        /* Standard DFT */
        __m256 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
               av_in8, av_in9;
        __m256 av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8, av_s9,
               av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16, av_s17,
               av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24;
        __m256 av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8, av_t9,
               av_t10, av_t11, av_t12, av_t13, av_t14;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17, v_out18, v_out19;

        curr_in = in;
        curr_out = out;

        __m256 v256_CRTM_10_1 = _mm512_castps512_ps256(v_CRTM_10_1);
        __m256 v256_CRTM_10_2 = _mm512_castps512_ps256(v_CRTM_10_2);
        __m256 v256_CRTM_10_3 = _mm512_castps512_ps256(v_CRTM_10_3);
        __m256 v256_CRTM_10_4 = _mm512_castps512_ps256(v_CRTM_10_4);
        __m256 v256_CRTM_10_5 = _mm512_castps512_ps256(v_CRTM_10_5);

        // Input point 1: x(0)
        LDR_256_S(curr_in, v_in_stride, av_in0);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x256_S(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in + in_strides[7];
        LDRI_2x256_S(curr_in, v_in_stride, av_in3, av_in4);
        // Input point 12: x(11) & Input point 13: x(12)
        curr_in = in + in_strides[11];
        LDRI_2x256_S(curr_in, v_in_stride, av_in5, av_in6);
        // Input point 16: x(15) & Input point 17: x(16)
        curr_in = in + in_strides[15];
        LDRI_2x256_S(curr_in, v_in_stride, av_in7, av_in8);
        // Input point 20: x(19)
        curr_in = in + in_strides[19];
        LDR_256_S(curr_in, v_in_stride, av_in9);

        av_s1  = _mm256_add_ps(av_in0, av_in9);
        av_s2  = _mm256_sub_ps(av_in0, av_in9);
        av_s3  = _mm256_add_ps(av_in1, av_in7);
        av_s4  = _mm256_sub_ps(av_in1, av_in7);
        av_s5  = _mm256_add_ps(av_in2, av_in8);
        av_s6  = _mm256_sub_ps(av_in2, av_in8);
        av_t11 = _mm256_mul_ps(v256_CRTM_10_4, av_s6);
        av_s9  = _mm256_add_ps(av_in4, av_in6);

        av_t9  = _mm256_mul_ps(v256_CRTM_10_4, av_s9);
        av_t10 = _mm256_mul_ps(v256_CRTM_10_5, av_s5);
        av_s23 = _mm256_sub_ps(av_t9, av_t10);
        av_s10 = _mm256_sub_ps(av_in4, av_in6);
        av_t12 = _mm256_mul_ps(v256_CRTM_10_5, av_s10);
        av_s24 = _mm256_sub_ps(av_t12, av_t11);

        av_s7  = _mm256_add_ps(av_in3, av_in5);
        av_s8  = _mm256_sub_ps(av_in3, av_in5);
        av_s14 = _mm256_sub_ps(av_s4, av_s8);
        av_t14 = _mm256_mul_ps(v256_CRTM_10_3, av_s14);
        // Output point 11: X(10)
        v_out10 = _mm256_sub_ps(av_s2, av_t14);
        curr_out = out + out_strides[10];
        STR_256_S(curr_out, v_out_stride, v_out10);

        av_s11 = _mm256_add_ps(av_s3, av_s7);
        av_t13 = _mm256_mul_ps(v256_CRTM_10_3, av_s11);
        // Output point 1: X(0)
        v_out0 = _mm256_add_ps(av_s1, av_t13);
        curr_out = out + out_strides[0];
        STR_256_S(curr_out, v_out_stride, v_out0);

        av_t1  = _mm256_mul_ps(v256_CRTM_10_2, av_s11);
        av_s16 = _mm256_sub_ps(av_s1, av_t1);
        av_s12 = _mm256_sub_ps(av_s3, av_s7);
        av_t3  = _mm256_mul_ps(v256_CRTM_10_1, av_s12);
        av_s19 = _mm256_add_ps(av_s16, av_t3);
        av_s20 = _mm256_sub_ps(av_s16, av_t3);
        // Output point 9: X(8)
        v_out8 = _mm256_add_ps(av_s20, av_s24);
        curr_out = out + out_strides[8];
        STR_256_S(curr_out, v_out_stride, v_out8);
        // Output point 13: X(12)
        v_out12 = _mm256_sub_ps(av_s20, av_s24);
        curr_out = out + out_strides[12];
        STR_256_S(curr_out, v_out_stride, v_out12);

        av_s13 = _mm256_add_ps(av_s4, av_s8);
        av_t4  = _mm256_mul_ps(v256_CRTM_10_1, av_s13);
        av_t2  = _mm256_mul_ps(v256_CRTM_10_2, av_s14);
        av_s15 = _mm256_add_ps(av_s2, av_t2);
        av_s17 = _mm256_add_ps(av_s15, av_t4);
        av_s18 = _mm256_sub_ps(av_s15, av_t4);
        // Output point 7: X(6)
        v_out6 = _mm256_add_ps(av_s18, av_s23);
        curr_out = out + out_strides[6];
        STR_256_S(curr_out, v_out_stride, v_out6);
        // Output point 15: X(14)
        v_out14 = _mm256_sub_ps(av_s18, av_s23);
        curr_out = out + out_strides[14];
        STR_256_S(curr_out, v_out_stride, v_out14);

        av_t5  = _mm256_mul_ps(v256_CRTM_10_4, av_s5);
        av_t6  = _mm256_mul_ps(v256_CRTM_10_5, av_s9);
        av_s21 = _mm256_add_ps(av_t5, av_t6);
        // Output point 3: X(2)
        v_out2 = _mm256_sub_ps(av_s17, av_s21);
        curr_out = out + out_strides[2];
        STR_256_S(curr_out, v_out_stride, v_out2);
        // Output point 19: X(18)
        v_out18 = _mm256_add_ps(av_s17, av_s21);
        curr_out = out + out_strides[18];
        STR_256_S(curr_out, v_out_stride, v_out18);

        av_t7  = _mm256_mul_ps(v256_CRTM_10_4, av_s10);
        av_t8  = _mm256_mul_ps(v256_CRTM_10_5, av_s6);
        av_s22 = _mm256_add_ps(av_t7, av_t8);
        // Output point 5: X(4)
        v_out4 = _mm256_sub_ps(av_s19, av_s22);
        curr_out = out + out_strides[4];
        STR_256_S(curr_out, v_out_stride, v_out4);
        // Output point 17: X(16)
        v_out16 = _mm256_add_ps(av_s19, av_s22);
        curr_out = out + out_strides[16];
        STR_256_S(curr_out, v_out_stride, v_out16);

        /* Shifted DFT */
        __m256 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8, bv_in9;
        __m256 bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
               bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17,
               bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24;
        __m256 bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
               bv_t10, bv_t11, bv_t12, bv_t13, bv_t14;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x256_S(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDRI_2x256_S(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDRI_2x256_S(curr_in, v_in_stride, bv_in4, bv_in5);
        // Input point 14: x(13) & Input point 15: x(14)
        curr_in = in + in_strides[13];
        LDRI_2x256_S(curr_in, v_in_stride, bv_in6, bv_in7);
        // Input point 18: x(17) & Input point 19: x(18)
        curr_in = in + in_strides[17];
        LDRI_2x256_S(curr_in, v_in_stride, bv_in8, bv_in9);

        bv_s1 = _mm256_add_ps(bv_in0, bv_in8);
        bv_s2 = _mm256_sub_ps(bv_in0, bv_in8);
        bv_s3 = _mm256_add_ps(bv_in1, bv_in9);
        bv_s4 = _mm256_sub_ps(bv_in1, bv_in9);
        bv_s5 = _mm256_add_ps(bv_in2, bv_in6);
        bv_s6 = _mm256_sub_ps(bv_in2, bv_in6);
        bv_s7 = _mm256_add_ps(bv_in3, bv_in7);
        bv_s8 = _mm256_sub_ps(bv_in3, bv_in7);

        bv_s9  = _mm256_add_ps(bv_s1, bv_s5);
        bv_s10 = _mm256_sub_ps(bv_s1, bv_s5);
        bv_s11 = _mm256_add_ps(bv_s3, bv_s7);
        bv_s12 = _mm256_sub_ps(bv_s3, bv_s7);

        bv_t1  = _mm256_mul_ps(v256_CRTM_10_1, bv_s11);
        bv_t2  = _mm256_mul_ps(v256_CRTM_10_2, bv_s12);
        bv_t3  = _mm256_mul_ps(v256_CRTM_10_1, bv_s10);
        bv_t4  = _mm256_mul_ps(v256_CRTM_10_2, bv_s9);
        bv_t5  = _mm256_mul_ps(v256_CRTM_10_3, bv_in4);
        bv_t6  = _mm256_mul_ps(v256_CRTM_10_3, bv_in5);
        bv_s23 = _mm256_add_ps(bv_s9, bv_s9);
        // Output point 2: X(1)
        v_out1 = _mm256_add_ps(bv_t5, bv_s23);
        curr_out = out + out_strides[1];
        STR_256_S(curr_out, v_out_stride, v_out1);

        bv_s24 = _mm256_add_ps(bv_s12, bv_s12);
        // Output point 12: X(11)
        v_out11 = NEGATE_256_S(_mm256_add_ps(bv_t6, bv_s24));
        curr_out = out + out_strides[11];
        STR_256_S(curr_out, v_out_stride, v_out11);

        bv_s13 = _mm256_sub_ps(bv_t2, bv_t6);
        bv_s14 = _mm256_sub_ps(bv_t4, bv_t5);

        bv_s15 = _mm256_sub_ps(bv_s13, bv_t1);
        bv_s16 = _mm256_add_ps(bv_s13, bv_t1);
        bv_s17 = _mm256_add_ps(bv_t3, bv_s14);
        bv_s18 = _mm256_sub_ps(bv_t3, bv_s14);

        bv_t7  = _mm256_mul_ps(v256_CRTM_10_5, bv_s2);
        bv_t8  = _mm256_mul_ps(v256_CRTM_10_5, bv_s4);
        bv_t9  = _mm256_mul_ps(v256_CRTM_10_5, bv_s6);
        bv_t10 = _mm256_mul_ps(v256_CRTM_10_5, bv_s8);
        bv_t11 = _mm256_mul_ps(v256_CRTM_10_4, bv_s2);
        bv_t12 = _mm256_mul_ps(v256_CRTM_10_4, bv_s4);
        bv_t13 = _mm256_mul_ps(v256_CRTM_10_4, bv_s6);
        bv_t14 = _mm256_mul_ps(v256_CRTM_10_4, bv_s8);

        bv_s19 = _mm256_add_ps(bv_t7, bv_t13);
        // Output point 4: X(3)
        v_out3 = _mm256_add_ps(bv_s15, bv_s19);
        curr_out = out + out_strides[3];
        STR_256_S(curr_out, v_out_stride, v_out3);
        // Output point 20: X(19)
        v_out19 = _mm256_sub_ps(bv_s15, bv_s19);
        curr_out = out + out_strides[19];
        STR_256_S(curr_out, v_out_stride, v_out19);

        bv_s20 = _mm256_add_ps(bv_t10, bv_t12);
        // Output point 6: X(5)
        v_out5 = _mm256_sub_ps(bv_s17, bv_s20);
        curr_out = out + out_strides[5];
        STR_256_S(curr_out, v_out_stride, v_out5);
        // Output point 18: X(17)
        v_out17 = NEGATE_256_S(_mm256_add_ps(bv_s17, bv_s20));
        curr_out = out + out_strides[17];
        STR_256_S(curr_out, v_out_stride, v_out17);

        bv_s21 = _mm256_sub_ps(bv_t11, bv_t9);
        // Output point 8: X(7)
        v_out7 = _mm256_sub_ps(bv_s21, bv_s16);
        curr_out = out + out_strides[7];
        STR_256_S(curr_out, v_out_stride, v_out7);
        // Output point 16: X(15)
        v_out15 = NEGATE_256_S(_mm256_add_ps(bv_s16, bv_s21));
        curr_out = out + out_strides[15];
        STR_256_S(curr_out, v_out_stride, v_out15);

        bv_s22 = _mm256_sub_ps(bv_t14, bv_t8);
        // Output point 10: X(9)
        v_out9 = _mm256_add_ps(bv_s22, bv_s18);
        curr_out = out + out_strides[9];
        STR_256_S(curr_out, v_out_stride, v_out9);
        // Output point 14: X(13)
        v_out13 = _mm256_sub_ps(bv_s22, bv_s18);
        curr_out = out + out_strides[13];
        STR_256_S(curr_out, v_out_stride, v_out13);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (n & 4)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
               av_in8, av_in9;
        __m128 av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8, av_s9,
               av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16, av_s17,
               av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24;
        __m128 av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8, av_t9,
               av_t10, av_t11, av_t12, av_t13, av_t14;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17, v_out18, v_out19;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_10_1 = _mm512_castps512_ps128(v_CRTM_10_1);
        __m128 v128_CRTM_10_2 = _mm512_castps512_ps128(v_CRTM_10_2);
        __m128 v128_CRTM_10_3 = _mm512_castps512_ps128(v_CRTM_10_3);
        __m128 v128_CRTM_10_4 = _mm512_castps512_ps128(v_CRTM_10_4);
        __m128 v128_CRTM_10_5 = _mm512_castps512_ps128(v_CRTM_10_5);

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
        // Input point 16: x(15) & Input point 17: x(16)
        curr_in = in + in_strides[15];
        LDRI_2x128_S(curr_in, v_in_stride, av_in7, av_in8);
        // Input point 20: x(19)
        curr_in = in + in_strides[19];
        LDR_128_S(curr_in, v_in_stride, av_in9);

        av_s1  = _mm_add_ps(av_in0, av_in9);
        av_s2  = _mm_sub_ps(av_in0, av_in9);
        av_s3  = _mm_add_ps(av_in1, av_in7);
        av_s4  = _mm_sub_ps(av_in1, av_in7);
        av_s5  = _mm_add_ps(av_in2, av_in8);
        av_s6  = _mm_sub_ps(av_in2, av_in8);
        av_t11 = _mm_mul_ps(v128_CRTM_10_4, av_s6);
        av_s9  = _mm_add_ps(av_in4, av_in6);

        av_t9  = _mm_mul_ps(v128_CRTM_10_4, av_s9);
        av_t10 = _mm_mul_ps(v128_CRTM_10_5, av_s5);
        av_s23 = _mm_sub_ps(av_t9, av_t10);
        av_s10 = _mm_sub_ps(av_in4, av_in6);
        av_t12 = _mm_mul_ps(v128_CRTM_10_5, av_s10);
        av_s24 = _mm_sub_ps(av_t12, av_t11);

        av_s7  = _mm_add_ps(av_in3, av_in5);
        av_s8  = _mm_sub_ps(av_in3, av_in5);
        av_s14 = _mm_sub_ps(av_s4, av_s8);
        av_t14 = _mm_mul_ps(v128_CRTM_10_3, av_s14);
        // Output point 11: X(10)
        v_out10 = _mm_sub_ps(av_s2, av_t14);
        curr_out = out + out_strides[10];
        STR_128_S(curr_out, v_out_stride, v_out10);

        av_s11 = _mm_add_ps(av_s3, av_s7);
        av_t13 = _mm_mul_ps(v128_CRTM_10_3, av_s11);
        // Output point 1: X(0)
        v_out0 = _mm_add_ps(av_s1, av_t13);
        curr_out = out + out_strides[0];
        STR_128_S(curr_out, v_out_stride, v_out0);

        av_t1  = _mm_mul_ps(v128_CRTM_10_2, av_s11);
        av_s16 = _mm_sub_ps(av_s1, av_t1);
        av_s12 = _mm_sub_ps(av_s3, av_s7);
        av_t3  = _mm_mul_ps(v128_CRTM_10_1, av_s12);
        av_s19 = _mm_add_ps(av_s16, av_t3);
        av_s20 = _mm_sub_ps(av_s16, av_t3);
        // Output point 9: X(8)
        v_out8 = _mm_add_ps(av_s20, av_s24);
        curr_out = out + out_strides[8];
        STR_128_S(curr_out, v_out_stride, v_out8);
        // Output point 13: X(12)
        v_out12 = _mm_sub_ps(av_s20, av_s24);
        curr_out = out + out_strides[12];
        STR_128_S(curr_out, v_out_stride, v_out12);

        av_s13 = _mm_add_ps(av_s4, av_s8);
        av_t4  = _mm_mul_ps(v128_CRTM_10_1, av_s13);
        av_t2  = _mm_mul_ps(v128_CRTM_10_2, av_s14);
        av_s15 = _mm_add_ps(av_s2, av_t2);
        av_s17 = _mm_add_ps(av_s15, av_t4);
        av_s18 = _mm_sub_ps(av_s15, av_t4);
        // Output point 7: X(6)
        v_out6 = _mm_add_ps(av_s18, av_s23);
        curr_out = out + out_strides[6];
        STR_128_S(curr_out, v_out_stride, v_out6);
        // Output point 15: X(14)
        v_out14 = _mm_sub_ps(av_s18, av_s23);
        curr_out = out + out_strides[14];
        STR_128_S(curr_out, v_out_stride, v_out14);

        av_t5  = _mm_mul_ps(v128_CRTM_10_4, av_s5);
        av_t6  = _mm_mul_ps(v128_CRTM_10_5, av_s9);
        av_s21 = _mm_add_ps(av_t5, av_t6);
        // Output point 3: X(2)
        v_out2 = _mm_sub_ps(av_s17, av_s21);
        curr_out = out + out_strides[2];
        STR_128_S(curr_out, v_out_stride, v_out2);
        // Output point 19: X(18)
        v_out18 = _mm_add_ps(av_s17, av_s21);
        curr_out = out + out_strides[18];
        STR_128_S(curr_out, v_out_stride, v_out18);

        av_t7  = _mm_mul_ps(v128_CRTM_10_4, av_s10);
        av_t8  = _mm_mul_ps(v128_CRTM_10_5, av_s6);
        av_s22 = _mm_add_ps(av_t7, av_t8);
        // Output point 5: X(4)
        v_out4 = _mm_sub_ps(av_s19, av_s22);
        curr_out = out + out_strides[4];
        STR_128_S(curr_out, v_out_stride, v_out4);
        // Output point 17: X(16)
        v_out16 = _mm_add_ps(av_s19, av_s22);
        curr_out = out + out_strides[16];
        STR_128_S(curr_out, v_out_stride, v_out16);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8, bv_in9;
        __m128 bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
               bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17,
               bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24;
        __m128 bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
               bv_t10, bv_t11, bv_t12, bv_t13, bv_t14;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in4, bv_in5);
        // Input point 14: x(13) & Input point 15: x(14)
        curr_in = in + in_strides[13];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in6, bv_in7);
        // Input point 18: x(17) & Input point 19: x(18)
        curr_in = in + in_strides[17];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in8, bv_in9);

        bv_s1 = _mm_add_ps(bv_in0, bv_in8);
        bv_s2 = _mm_sub_ps(bv_in0, bv_in8);
        bv_s3 = _mm_add_ps(bv_in1, bv_in9);
        bv_s4 = _mm_sub_ps(bv_in1, bv_in9);
        bv_s5 = _mm_add_ps(bv_in2, bv_in6);
        bv_s6 = _mm_sub_ps(bv_in2, bv_in6);
        bv_s7 = _mm_add_ps(bv_in3, bv_in7);
        bv_s8 = _mm_sub_ps(bv_in3, bv_in7);

        bv_s9  = _mm_add_ps(bv_s1, bv_s5);
        bv_s10 = _mm_sub_ps(bv_s1, bv_s5);
        bv_s11 = _mm_add_ps(bv_s3, bv_s7);
        bv_s12 = _mm_sub_ps(bv_s3, bv_s7);

        bv_t1  = _mm_mul_ps(v128_CRTM_10_1, bv_s11);
        bv_t2  = _mm_mul_ps(v128_CRTM_10_2, bv_s12);
        bv_t3  = _mm_mul_ps(v128_CRTM_10_1, bv_s10);
        bv_t4  = _mm_mul_ps(v128_CRTM_10_2, bv_s9);
        bv_t5  = _mm_mul_ps(v128_CRTM_10_3, bv_in4);
        bv_t6  = _mm_mul_ps(v128_CRTM_10_3, bv_in5);
        bv_s23 = _mm_add_ps(bv_s9, bv_s9);
        // Output point 2: X(1)
        v_out1 = _mm_add_ps(bv_t5, bv_s23);
        curr_out = out + out_strides[1];
        STR_128_S(curr_out, v_out_stride, v_out1);

        bv_s24 = _mm_add_ps(bv_s12, bv_s12);
        // Output point 12: X(11)
        v_out11 = NEGATE_128_S(_mm_add_ps(bv_t6, bv_s24));
        curr_out = out + out_strides[11];
        STR_128_S(curr_out, v_out_stride, v_out11);

        bv_s13 = _mm_sub_ps(bv_t2, bv_t6);
        bv_s14 = _mm_sub_ps(bv_t4, bv_t5);

        bv_s15 = _mm_sub_ps(bv_s13, bv_t1);
        bv_s16 = _mm_add_ps(bv_s13, bv_t1);
        bv_s17 = _mm_add_ps(bv_t3, bv_s14);
        bv_s18 = _mm_sub_ps(bv_t3, bv_s14);

        bv_t7  = _mm_mul_ps(v128_CRTM_10_5, bv_s2);
        bv_t8  = _mm_mul_ps(v128_CRTM_10_5, bv_s4);
        bv_t9  = _mm_mul_ps(v128_CRTM_10_5, bv_s6);
        bv_t10 = _mm_mul_ps(v128_CRTM_10_5, bv_s8);
        bv_t11 = _mm_mul_ps(v128_CRTM_10_4, bv_s2);
        bv_t12 = _mm_mul_ps(v128_CRTM_10_4, bv_s4);
        bv_t13 = _mm_mul_ps(v128_CRTM_10_4, bv_s6);
        bv_t14 = _mm_mul_ps(v128_CRTM_10_4, bv_s8);

        bv_s19 = _mm_add_ps(bv_t7, bv_t13);
        // Output point 4: X(3)
        v_out3 = _mm_add_ps(bv_s15, bv_s19);
        curr_out = out + out_strides[3];
        STR_128_S(curr_out, v_out_stride, v_out3);
        // Output point 20: X(19)
        v_out19 = _mm_sub_ps(bv_s15, bv_s19);
        curr_out = out + out_strides[19];
        STR_128_S(curr_out, v_out_stride, v_out19);

        bv_s20 = _mm_add_ps(bv_t10, bv_t12);
        // Output point 6: X(5)
        v_out5 = _mm_sub_ps(bv_s17, bv_s20);
        curr_out = out + out_strides[5];
        STR_128_S(curr_out, v_out_stride, v_out5);
        // Output point 18: X(17)
        v_out17 = NEGATE_128_S(_mm_add_ps(bv_s17, bv_s20));
        curr_out = out + out_strides[17];
        STR_128_S(curr_out, v_out_stride, v_out17);

        bv_s21 = _mm_sub_ps(bv_t11, bv_t9);
        // Output point 8: X(7)
        v_out7 = _mm_sub_ps(bv_s21, bv_s16);
        curr_out = out + out_strides[7];
        STR_128_S(curr_out, v_out_stride, v_out7);
        // Output point 16: X(15)
        v_out15 = NEGATE_128_S(_mm_add_ps(bv_s16, bv_s21));
        curr_out = out + out_strides[15];
        STR_128_S(curr_out, v_out_stride, v_out15);

        bv_s22 = _mm_sub_ps(bv_t14, bv_t8);
        // Output point 10: X(9)
        v_out9 = _mm_add_ps(bv_s22, bv_s18);
        curr_out = out + out_strides[9];
        STR_128_S(curr_out, v_out_stride, v_out9);
        // Output point 14: X(13)
        v_out13 = _mm_sub_ps(bv_s22, bv_s18);
        curr_out = out + out_strides[13];
        STR_128_S(curr_out, v_out_stride, v_out13);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (n & 2)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
               av_in8, av_in9;
        __m128 av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8, av_s9,
               av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16, av_s17,
               av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24;
        __m128 av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8, av_t9,
               av_t10, av_t11, av_t12, av_t13, av_t14;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17, v_out18, v_out19;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_10_1 = _mm512_castps512_ps128(v_CRTM_10_1);
        __m128 v128_CRTM_10_2 = _mm512_castps512_ps128(v_CRTM_10_2);
        __m128 v128_CRTM_10_3 = _mm512_castps512_ps128(v_CRTM_10_3);
        __m128 v128_CRTM_10_4 = _mm512_castps512_ps128(v_CRTM_10_4);
        __m128 v128_CRTM_10_5 = _mm512_castps512_ps128(v_CRTM_10_5);

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
        // Input point 16: x(15) & Input point 17: x(16)
        curr_in = in + in_strides[15];
        LDHRI_2x128_S(curr_in, v_in_stride, av_in7, av_in8);
        // Input point 20: x(19)
        curr_in = in + in_strides[19];
        LDHR_128_S(curr_in, v_in_stride, av_in9);

        av_s1  = _mm_add_ps(av_in0, av_in9);
        av_s2  = _mm_sub_ps(av_in0, av_in9);
        av_s3  = _mm_add_ps(av_in1, av_in7);
        av_s4  = _mm_sub_ps(av_in1, av_in7);
        av_s5  = _mm_add_ps(av_in2, av_in8);
        av_s6  = _mm_sub_ps(av_in2, av_in8);
        av_t11 = _mm_mul_ps(v128_CRTM_10_4, av_s6);
        av_s9  = _mm_add_ps(av_in4, av_in6);

        av_t9  = _mm_mul_ps(v128_CRTM_10_4, av_s9);
        av_t10 = _mm_mul_ps(v128_CRTM_10_5, av_s5);
        av_s23 = _mm_sub_ps(av_t9, av_t10);
        av_s10 = _mm_sub_ps(av_in4, av_in6);
        av_t12 = _mm_mul_ps(v128_CRTM_10_5, av_s10);
        av_s24 = _mm_sub_ps(av_t12, av_t11);

        av_s7  = _mm_add_ps(av_in3, av_in5);
        av_s8  = _mm_sub_ps(av_in3, av_in5);
        av_s14 = _mm_sub_ps(av_s4, av_s8);
        av_t14 = _mm_mul_ps(v128_CRTM_10_3, av_s14);
        // Output point 11: X(10)
        v_out10 = _mm_sub_ps(av_s2, av_t14);
        curr_out = out + out_strides[10];
        STHR_128_S(curr_out, v_out_stride, v_out10);

        av_s11 = _mm_add_ps(av_s3, av_s7);
        av_t13 = _mm_mul_ps(v128_CRTM_10_3, av_s11);
        // Output point 1: X(0)
        v_out0 = _mm_add_ps(av_s1, av_t13);
        curr_out = out + out_strides[0];
        STHR_128_S(curr_out, v_out_stride, v_out0);

        av_t1  = _mm_mul_ps(v128_CRTM_10_2, av_s11);
        av_s16 = _mm_sub_ps(av_s1, av_t1);
        av_s12 = _mm_sub_ps(av_s3, av_s7);
        av_t3  = _mm_mul_ps(v128_CRTM_10_1, av_s12);
        av_s19 = _mm_add_ps(av_s16, av_t3);
        av_s20 = _mm_sub_ps(av_s16, av_t3);
        // Output point 9: X(8)
        v_out8 = _mm_add_ps(av_s20, av_s24);
        curr_out = out + out_strides[8];
        STHR_128_S(curr_out, v_out_stride, v_out8);
        // Output point 13: X(12)
        v_out12 = _mm_sub_ps(av_s20, av_s24);
        curr_out = out + out_strides[12];
        STHR_128_S(curr_out, v_out_stride, v_out12);

        av_s13 = _mm_add_ps(av_s4, av_s8);
        av_t4  = _mm_mul_ps(v128_CRTM_10_1, av_s13);
        av_t2  = _mm_mul_ps(v128_CRTM_10_2, av_s14);
        av_s15 = _mm_add_ps(av_s2, av_t2);
        av_s17 = _mm_add_ps(av_s15, av_t4);
        av_s18 = _mm_sub_ps(av_s15, av_t4);
        // Output point 7: X(6)
        v_out6 = _mm_add_ps(av_s18, av_s23);
        curr_out = out + out_strides[6];
        STHR_128_S(curr_out, v_out_stride, v_out6);
        // Output point 15: X(14)
        v_out14 = _mm_sub_ps(av_s18, av_s23);
        curr_out = out + out_strides[14];
        STHR_128_S(curr_out, v_out_stride, v_out14);

        av_t5  = _mm_mul_ps(v128_CRTM_10_4, av_s5);
        av_t6  = _mm_mul_ps(v128_CRTM_10_5, av_s9);
        av_s21 = _mm_add_ps(av_t5, av_t6);
        // Output point 3: X(2)
        v_out2 = _mm_sub_ps(av_s17, av_s21);
        curr_out = out + out_strides[2];
        STHR_128_S(curr_out, v_out_stride, v_out2);
        // Output point 19: X(18)
        v_out18 = _mm_add_ps(av_s17, av_s21);
        curr_out = out + out_strides[18];
        STHR_128_S(curr_out, v_out_stride, v_out18);

        av_t7  = _mm_mul_ps(v128_CRTM_10_4, av_s10);
        av_t8  = _mm_mul_ps(v128_CRTM_10_5, av_s6);
        av_s22 = _mm_add_ps(av_t7, av_t8);
        // Output point 5: X(4)
        v_out4 = _mm_sub_ps(av_s19, av_s22);
        curr_out = out + out_strides[4];
        STHR_128_S(curr_out, v_out_stride, v_out4);
        // Output point 17: X(16)
        v_out16 = _mm_add_ps(av_s19, av_s22);
        curr_out = out + out_strides[16];
        STHR_128_S(curr_out, v_out_stride, v_out16);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8, bv_in9;
        __m128 bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
               bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17,
               bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24;
        __m128 bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
               bv_t10, bv_t11, bv_t12, bv_t13, bv_t14;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in4, bv_in5);
        // Input point 14: x(13) & Input point 15: x(14)
        curr_in = in + in_strides[13];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in6, bv_in7);
        // Input point 18: x(17) & Input point 19: x(18)
        curr_in = in + in_strides[17];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in8, bv_in9);

        bv_s1 = _mm_add_ps(bv_in0, bv_in8);
        bv_s2 = _mm_sub_ps(bv_in0, bv_in8);
        bv_s3 = _mm_add_ps(bv_in1, bv_in9);
        bv_s4 = _mm_sub_ps(bv_in1, bv_in9);
        bv_s5 = _mm_add_ps(bv_in2, bv_in6);
        bv_s6 = _mm_sub_ps(bv_in2, bv_in6);
        bv_s7 = _mm_add_ps(bv_in3, bv_in7);
        bv_s8 = _mm_sub_ps(bv_in3, bv_in7);

        bv_s9  = _mm_add_ps(bv_s1, bv_s5);
        bv_s10 = _mm_sub_ps(bv_s1, bv_s5);
        bv_s11 = _mm_add_ps(bv_s3, bv_s7);
        bv_s12 = _mm_sub_ps(bv_s3, bv_s7);

        bv_t1  = _mm_mul_ps(v128_CRTM_10_1, bv_s11);
        bv_t2  = _mm_mul_ps(v128_CRTM_10_2, bv_s12);
        bv_t3  = _mm_mul_ps(v128_CRTM_10_1, bv_s10);
        bv_t4  = _mm_mul_ps(v128_CRTM_10_2, bv_s9);
        bv_t5  = _mm_mul_ps(v128_CRTM_10_3, bv_in4);
        bv_t6  = _mm_mul_ps(v128_CRTM_10_3, bv_in5);
        bv_s23 = _mm_add_ps(bv_s9, bv_s9);
        // Output point 2: X(1)
        v_out1 = _mm_add_ps(bv_t5, bv_s23);
        curr_out = out + out_strides[1];
        STHR_128_S(curr_out, v_out_stride, v_out1);

        bv_s24 = _mm_add_ps(bv_s12, bv_s12);
        // Output point 12: X(11)
        v_out11 = NEGATE_128_S(_mm_add_ps(bv_t6, bv_s24));
        curr_out = out + out_strides[11];
        STHR_128_S(curr_out, v_out_stride, v_out11);

        bv_s13 = _mm_sub_ps(bv_t2, bv_t6);
        bv_s14 = _mm_sub_ps(bv_t4, bv_t5);

        bv_s15 = _mm_sub_ps(bv_s13, bv_t1);
        bv_s16 = _mm_add_ps(bv_s13, bv_t1);
        bv_s17 = _mm_add_ps(bv_t3, bv_s14);
        bv_s18 = _mm_sub_ps(bv_t3, bv_s14);

        bv_t7  = _mm_mul_ps(v128_CRTM_10_5, bv_s2);
        bv_t8  = _mm_mul_ps(v128_CRTM_10_5, bv_s4);
        bv_t9  = _mm_mul_ps(v128_CRTM_10_5, bv_s6);
        bv_t10 = _mm_mul_ps(v128_CRTM_10_5, bv_s8);
        bv_t11 = _mm_mul_ps(v128_CRTM_10_4, bv_s2);
        bv_t12 = _mm_mul_ps(v128_CRTM_10_4, bv_s4);
        bv_t13 = _mm_mul_ps(v128_CRTM_10_4, bv_s6);
        bv_t14 = _mm_mul_ps(v128_CRTM_10_4, bv_s8);

        bv_s19 = _mm_add_ps(bv_t7, bv_t13);
        // Output point 4: X(3)
        v_out3 = _mm_add_ps(bv_s15, bv_s19);
        curr_out = out + out_strides[3];
        STHR_128_S(curr_out, v_out_stride, v_out3);
        // Output point 20: X(19)
        v_out19 = _mm_sub_ps(bv_s15, bv_s19);
        curr_out = out + out_strides[19];
        STHR_128_S(curr_out, v_out_stride, v_out19);

        bv_s20 = _mm_add_ps(bv_t10, bv_t12);
        // Output point 6: X(5)
        v_out5 = _mm_sub_ps(bv_s17, bv_s20);
        curr_out = out + out_strides[5];
        STHR_128_S(curr_out, v_out_stride, v_out5);
        // Output point 18: X(17)
        v_out17 = NEGATE_128_S(_mm_add_ps(bv_s17, bv_s20));
        curr_out = out + out_strides[17];
        STHR_128_S(curr_out, v_out_stride, v_out17);

        bv_s21 = _mm_sub_ps(bv_t11, bv_t9);
        // Output point 8: X(7)
        v_out7 = _mm_sub_ps(bv_s21, bv_s16);
        curr_out = out + out_strides[7];
        STHR_128_S(curr_out, v_out_stride, v_out7);
        // Output point 16: X(15)
        v_out15 = NEGATE_128_S(_mm_add_ps(bv_s16, bv_s21));
        curr_out = out + out_strides[15];
        STHR_128_S(curr_out, v_out_stride, v_out15);

        bv_s22 = _mm_sub_ps(bv_t14, bv_t8);
        // Output point 10: X(9)
        v_out9 = _mm_add_ps(bv_s22, bv_s18);
        curr_out = out + out_strides[9];
        STHR_128_S(curr_out, v_out_stride, v_out9);
        // Output point 14: X(13)
        v_out13 = _mm_sub_ps(bv_s22, bv_s18);
        curr_out = out + out_strides[13];
        STHR_128_S(curr_out, v_out_stride, v_out13);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        /* Standard DFT */
        FLOAT a_in0, a_in1, a_in2, a_in3, a_in4, a_in5, a_in6, a_in7, a_in8,
              a_in9;
        FLOAT a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_s9, a_s10,
              a_s11, a_s12, a_s13, a_s14, a_s15, a_s16, a_s17, a_s18, a_s19,
              a_s20, a_s21, a_s22, a_s23;
        FLOAT a_t0, a_t1, a_t2, a_t3, a_t4, a_t5, a_t6, a_t7, a_t8, a_t9, a_t10,
              a_t11, a_t12, a_t13;

        //  Input point 1: x(0)
        a_in0 = *in;
        // Input point 4: x(3)
        a_in1 = in[in_strides[3]];
        // Input point 5: x(4)
        a_in2 = in[in_strides[4]];
        // Input point 8: x(7)
        a_in3 = in[in_strides[7]];
        // Input point 9: x(8)
        a_in4 = in[in_strides[8]];
        // Input point 12: x(11)
        a_in5 = in[in_strides[11]];
        // Input point 13: x(12)
        a_in6 = in[in_strides[12]];
        // Input point 16: x(15)
        a_in7 = in[in_strides[15]];
        // Input point 17: x(16)
        a_in8 = in[in_strides[16]];
        // Input point 20: x(19)
        a_in9 = in[in_strides[19]];

        a_s0 = a_in0 + a_in9;
        a_s1 = a_in0 - a_in9;
        a_s2 = a_in1 + a_in7;
        a_s3 = a_in1 - a_in7;
        a_s4 = a_in2 + a_in8;
        a_s5 = a_in2 - a_in8;
        a_t0 = CRTM_10_4 * a_s5;
        a_s8 = a_in4 + a_in6;

        a_t1 = CRTM_10_4 * a_s8;
        a_t2 = CRTM_10_5 * a_s4;
        a_s22 = a_t1 - a_t2;
        a_s9 = a_in4 - a_in6;
        a_t3 = CRTM_10_5 * a_s9;
        a_s23 = a_t3 - a_t0;

        a_s6 = a_in3 + a_in5;
        a_s7 = a_in3 - a_in5;
        a_s13 = a_s3 - a_s7;
        a_t4 = CRTM_10_3 * a_s13;
        // Output point 11: X(10)
        out[out_strides[10]] = a_s1 - a_t4;

        a_s10 = a_s2 + a_s6;
        a_t5 = CRTM_10_3 * a_s10;
        // Output point 1: X(0)
        *out = a_s0 + a_t5;

        a_t6 = CRTM_10_2 * a_s10;
        a_s15 = a_s0 - a_t6;
        a_s11 = a_s2 - a_s6;
        a_t7 = CRTM_10_1 * a_s11;
        a_s18 = a_s15 + a_t7;
        a_s19 = a_s15 - a_t7;
        // Output point 9: X(8)
        out[out_strides[8]] = a_s19 + a_s23;
        // Output point 13: X(12)
        out[out_strides[12]] = a_s19 - a_s23;

        a_s12 = a_s3 + a_s7;
        a_t8 = CRTM_10_1 * a_s12;
        a_t9 = CRTM_10_2 * a_s13;
        a_s14 = a_s1 + a_t9;
        a_s16 = a_s14 + a_t8;
        a_s17 = a_s14 - a_t8;
        // Output point 7: X(6)
        out[out_strides[6]] = a_s17 + a_s22;
        // Output point 15: X(14)
        out[out_strides[14]] = a_s17 - a_s22;

        a_t10 = CRTM_10_4 * a_s4;
        a_t11 = CRTM_10_5 * a_s8;
        a_s20 = a_t10 + a_t11;
        // Output point 3: X(2)
        out[out_strides[2]] = a_s16 - a_s20;
        // Output point 19: X(18)
        out[out_strides[18]] = a_s16 + a_s20;

        a_t12 = CRTM_10_4 * a_s9;
        a_t13 = CRTM_10_5 * a_s5;
        a_s21 = a_t12 + a_t13;
        // Output point 5: X(4)
        out[out_strides[4]] = a_s18 - a_s21;
        // Output point 17: X(16)
        out[out_strides[16]] = a_s18 + a_s21;

        /* Shifted DFT */
        FLOAT b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7, b_in8,
              b_in9;
        FLOAT b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9, b_s10,
              b_s11, b_s12, b_s13, b_s14, b_s15, b_s16, b_s17, b_s18, b_s19,
              b_s20, b_s21, b_s22, b_s23;
        FLOAT b_t0, b_t1, b_t2, b_t3, b_t4, b_t5, b_t6, b_t7, b_t8, b_t9, b_t10,
              b_t11, b_t12, b_t13;

        // Input point 2: x(1)
        b_in0 = in[in_strides[1]];
        // Input point 3: x(2)
        b_in1 = in[in_strides[2]];
        // Input point 6: x(5)
        b_in2 = in[in_strides[5]];
        // Input point 7: x(6)
        b_in3 = in[in_strides[6]];
        // Input point 10: x(9)
        b_in4 = in[in_strides[9]];
        // Input point 11: x(10)
        b_in5 = in[in_strides[10]];
        // Input point 14: x(13)
        b_in6 = in[in_strides[13]];
        // Input point 15: x(14)
        b_in7 = in[in_strides[14]];
        // Input point 18: x(17)
        b_in8 = in[in_strides[17]];
        // Input point 19: x(18)
        b_in9 = in[in_strides[18]];

        b_s0 = b_in0 + b_in8;
        b_s1 = b_in0 - b_in8;
        b_s2 = b_in1 + b_in9;
        b_s3 = b_in1 - b_in9;
        b_s4 = b_in2 + b_in6;
        b_s5 = b_in2 - b_in6;
        b_s6 = b_in3 + b_in7;
        b_s7 = b_in3 - b_in7;

        b_s8 = b_s0 + b_s4;
        b_s9 = b_s0 - b_s4;
        b_s10 = b_s2 + b_s6;
        b_s11 = b_s2 - b_s6;

        b_t0 = CRTM_10_1 * b_s10;
        b_t1 = CRTM_10_2 * b_s11;
        b_t2 = CRTM_10_1 * b_s9;
        b_t3 = CRTM_10_2 * b_s8;
        b_t4 = CRTM_10_3 * b_in4;
        b_t5 = CRTM_10_3 * b_in5;
        b_s22 = b_s8 + b_s8;
        // Output point 2: X(1)
        out[out_strides[1]] = b_t4 + b_s22;
        b_s23 = b_s11 + b_s11;
        // Output point 12: X(11)
        out[out_strides[11]] = -(b_t5 + b_s23);

        b_s12 = b_t1 - b_t5;
        b_s13 = b_t3 - b_t4;

        b_s14 = b_s12 - b_t0;
        b_s15 = b_s12 + b_t0;
        b_s16 = b_t2 + b_s13;
        b_s17 = b_t2 - b_s13;

        b_t6 = CRTM_10_5 * b_s1;
        b_t7 = CRTM_10_5 * b_s3;
        b_t8 = CRTM_10_5 * b_s5;
        b_t9 = CRTM_10_5 * b_s7;
        b_t10 = CRTM_10_4 * b_s1;
        b_t11 = CRTM_10_4 * b_s3;
        b_t12 = CRTM_10_4 * b_s5;
        b_t13 = CRTM_10_4 * b_s7;

        b_s18 = b_t6 + b_t12;
        // Output point 4: X(3)
        out[out_strides[3]] = b_s14 + b_s18;
        // Output point 20: X(19)
        out[out_strides[19]] = b_s14 - b_s18;

        b_s19 = b_t9 + b_t11;
        // Output point 6: X(5)
        out[out_strides[5]] = b_s16 - b_s19;
        // Output point 18: X(17)
        out[out_strides[17]] = -(b_s16 + b_s19);

        b_s20 = b_t10 - b_t8;
        // Output point 8: X(7)
        out[out_strides[7]] = b_s20 - b_s15;
        // Output point 16: X(15)
        out[out_strides[15]] = -(b_s15 + b_s20);

        b_s21 = b_t13 - b_t7;
        // Output point 10: X(9)
        out[out_strides[9]] = b_s21 + b_s17;
        // Output point 14: X(13)
        out[out_strides[13]] = b_s21 - b_s17;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID r2hcf_rfft10avx512_fp64_fwd(VOID *in_real, VOID *in_imag,
                                        VOID *out_real, VOID *out_imag, INTP n,
                                        aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const DOUBLE CRTM_10_1 =
        0.55901699437494742410229341718281905886015458990288;
    const DOUBLE CRTM_10_2 =
        0.25000000000000000000000000000000000000000000000000;
    const DOUBLE CRTM_10_3 =
        0.58778525229247315738615484497912915412138427663885;
    const DOUBLE CRTM_10_4 =
        0.95105651629515357211643933337938214340569863400000;

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

    __m512d v_CRTM_10_1 = _mm512_set1_pd(CRTM_10_1);
    __m512d v_CRTM_10_2 = _mm512_set1_pd(CRTM_10_2);
    __m512d v_CRTM_10_3 = _mm512_set1_pd(CRTM_10_3);
    __m512d v_CRTM_10_4 = _mm512_set1_pd(CRTM_10_4);

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m512d av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
                av_in8, av_in9;
        __m512d av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
                av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16,
                av_s17, av_s18, av_s19, av_s20, av_s21, av_s22, av_s23;
        __m512d av_t0, av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8,
                av_t9, av_t10, av_t11;
        __m512d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
                v_out15, v_out16, v_out17, v_out18, v_out19;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_512_D(curr_in, v_in_stride, av_in0);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_512_D(curr_in, v_in_stride, av_in1);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_512_D(curr_in, v_in_stride, av_in2);
        // Input point 7: x(6)
        curr_in = in + in_strides[6];
        LDR_512_D(curr_in, v_in_stride, av_in3);
        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDR_512_D(curr_in, v_in_stride, av_in4);
        // Input point 11: x(10)
        curr_in = in + in_strides[10];
        LDR_512_D(curr_in, v_in_stride, av_in5);
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDR_512_D(curr_in, v_in_stride, av_in6);
        // Input point 15: x(14)
        curr_in = in + in_strides[14];
        LDR_512_D(curr_in, v_in_stride, av_in7);
        // Input point 17: x(16)
        curr_in = in + in_strides[16];
        LDR_512_D(curr_in, v_in_stride, av_in8);
        // Input point 19: x(18)
        curr_in = in + in_strides[18];
        LDR_512_D(curr_in, v_in_stride, av_in9);

        av_s0 = _mm512_add_pd(av_in0, av_in5);
        av_s1 = _mm512_sub_pd(av_in0, av_in5);
        av_s2 = _mm512_add_pd(av_in1, av_in9);
        av_s3 = _mm512_sub_pd(av_in1, av_in9);
        av_s4 = _mm512_add_pd(av_in2, av_in3);
        av_s5 = _mm512_sub_pd(av_in2, av_in3);
        av_s6 = _mm512_add_pd(av_in4, av_in6);
        av_s7 = _mm512_sub_pd(av_in4, av_in6);
        av_s8 = _mm512_add_pd(av_in7, av_in8);
        av_s9 = _mm512_sub_pd(av_in7, av_in8);

        av_s10 = _mm512_add_pd(av_s2, av_s6);
        av_s14 = _mm512_add_pd(av_s4, av_s8);
        av_s18 = _mm512_add_pd(av_s10,av_s14);
        // Output pt 1: X(0)
        v_out0 = _mm512_add_pd(av_s0, av_s18);
        STR_512_D(curr_out, v_out_stride, v_out0);

        av_s11 = _mm512_sub_pd(av_s2, av_s6);
        av_s17 = _mm512_sub_pd(av_s5, av_s9);
        av_s20 = _mm512_add_pd(av_s11, av_s17);
        av_s21 = _mm512_sub_pd(av_s11, av_s17);
        // Output point 20: X(19)
        v_out19 = _mm512_sub_pd(av_s1, av_s21);
        curr_out = out + out_strides[19];
        STR_512_D(curr_out, v_out_stride, v_out19);

        av_t0  = _mm512_mul_pd(v_CRTM_10_2, av_s21);
        av_t1  = _mm512_mul_pd(v_CRTM_10_1, av_s20);
        av_s22 = _mm512_add_pd(av_t0, av_s1);
        // Output point 4: X(3)
        v_out3 = _mm512_add_pd(av_s22, av_t1);
        // Output point 12: X(11)
        v_out11 = _mm512_sub_pd(av_s22, av_t1);

        av_s12 = _mm512_add_pd(av_s3, av_s7);
        av_s15 = _mm512_sub_pd(av_s4, av_s8);
        av_t4  = _mm512_mul_pd(v_CRTM_10_3, av_s12);
        av_t10 = _mm512_mul_pd(v_CRTM_10_4, av_s15);
        // Output point 5: X(4)
        v_out4 = NEGATE_512_D(_mm512_add_pd(av_t4, av_t10));
        curr_out = out + out_strides[3];
        STRI_2x512_D(curr_out, v_out_stride, v_out3, v_out4);

        av_t6 = _mm512_mul_pd(v_CRTM_10_3, av_s15);
        av_t8 = _mm512_mul_pd(v_CRTM_10_4, av_s12);
        // Output point 13: X(12)
        v_out12 = _mm512_sub_pd(av_t6, av_t8);
        curr_out = out + out_strides[11];
        STRI_2x512_D(curr_out, v_out_stride, v_out11, v_out12);

        av_s13 = _mm512_sub_pd(av_s7, av_s3);
        av_s16 = _mm512_add_pd(av_s5, av_s9);
        av_t5  = _mm512_mul_pd(v_CRTM_10_3, av_s13);
        av_t11 = _mm512_mul_pd(v_CRTM_10_4, av_s16);
        // Output point 17: X(16)
        v_out16 = _mm512_add_pd(av_t11, av_t5);

        av_t7 = _mm512_mul_pd(v_CRTM_10_3, av_s16);
        av_t9 = _mm512_mul_pd(v_CRTM_10_4, av_s13);
        // Output point 9: X(8)
        v_out8 = _mm512_sub_pd(av_t9, av_t7);

        av_s19 = _mm512_sub_pd(av_s10, av_s14);
        av_t2  = _mm512_mul_pd(v_CRTM_10_2, av_s18);
        av_t3  = _mm512_mul_pd(v_CRTM_10_1, av_s19);
        av_s23 = _mm512_sub_pd(av_s0, av_t2);
        // Output point 8: X(7)
        v_out7 = _mm512_add_pd(av_s23, av_t3);
        curr_out = out + out_strides[7];
        STRI_2x512_D(curr_out, v_out_stride, v_out7, v_out8);
        // Output point 16: X(15)
        v_out15 = _mm512_sub_pd(av_s23, av_t3);
        curr_out = out + out_strides[15];
        STRI_2x512_D(curr_out, v_out_stride, v_out15, v_out16);

        /* Shifted DFT */
        __m512d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
                bv_in8, bv_in9;
        __m512d bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
                bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16,
                bv_s17, bv_s18, bv_s19, bv_s20, bv_s21;
        __m512d bv_t0, bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8,
                bv_t9, bv_t10, bv_t11;

        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_512_D(curr_in, v_in_stride, bv_in0);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_512_D(curr_in, v_in_stride, bv_in1);
        // Input point 6: x(5)
        curr_in = in + in_strides[5];
        LDR_512_D(curr_in, v_in_stride, bv_in2);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_512_D(curr_in, v_in_stride, bv_in3);
        // Input point 10: x(9)
        curr_in = in + in_strides[9];
        LDR_512_D(curr_in, v_in_stride, bv_in4);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_512_D(curr_in, v_in_stride, bv_in5);
        // Input point 14: x(13)
        curr_in = in + in_strides[13];
        LDR_512_D(curr_in, v_in_stride, bv_in6);
        // Input point 16: x(15)
        curr_in = in + in_strides[15];
        LDR_512_D(curr_in, v_in_stride, bv_in7);
        // Input point 18: x(17)
        curr_in = in + in_strides[17];
        LDR_512_D(curr_in, v_in_stride, bv_in8);
        // Input point 20: x(19)
        curr_in = in + in_strides[19];
        LDR_512_D(curr_in, v_in_stride, bv_in9);

        bv_s0 = _mm512_add_pd(bv_in1, bv_in9);
        bv_s1 = _mm512_sub_pd(bv_in1, bv_in9);
        bv_s2 = _mm512_add_pd(bv_in2, bv_in8);
        bv_s3 = _mm512_sub_pd(bv_in2, bv_in8);
        bv_s4 = _mm512_add_pd(bv_in3, bv_in7);
        bv_s5 = _mm512_sub_pd(bv_in3, bv_in7);
        bv_s6 = _mm512_add_pd(bv_in4, bv_in6);
        bv_s7 = _mm512_sub_pd(bv_in4, bv_in6);

        bv_s8  = _mm512_add_pd(bv_s0, bv_s4);
        bv_s9  = _mm512_sub_pd(bv_s0, bv_s4);
        bv_s10 = _mm512_add_pd(bv_s3, bv_s7);
        bv_s11 = _mm512_sub_pd(bv_s3, bv_s7);
        // Output point 10: X(9)
        v_out9 = _mm512_sub_pd(bv_in0, bv_s11);
        // Output point 11: X(10)
        v_out10 = NEGATE_512_D(_mm512_add_pd(bv_s9, bv_in5));
        curr_out = out + out_strides[9];
        STRI_2x512_D(curr_out, v_out_stride, v_out9, v_out10);

        bv_t0 = _mm512_mul_pd(v_CRTM_10_2, bv_s11);
        bv_t1 = _mm512_mul_pd(v_CRTM_10_1, bv_s10);
        bv_t2 = _mm512_mul_pd(v_CRTM_10_1, bv_s8);
        bv_t3 = _mm512_mul_pd(v_CRTM_10_2, bv_s9);

        bv_s12 = _mm512_add_pd(bv_in0, bv_t0);
        bv_s13 = _mm512_sub_pd(bv_t3, bv_in5);

        bv_t4  = _mm512_mul_pd(v_CRTM_10_4, bv_s1);
        bv_t5  = _mm512_mul_pd(v_CRTM_10_3, bv_s5);
        bv_t6  = _mm512_mul_pd(v_CRTM_10_4, bv_s6);
        bv_t7  = _mm512_mul_pd(v_CRTM_10_3, bv_s2);
        bv_t8  = _mm512_mul_pd(v_CRTM_10_4, bv_s5);
        bv_t9  = _mm512_mul_pd(v_CRTM_10_3, bv_s1);
        bv_t10 = _mm512_mul_pd(v_CRTM_10_4, bv_s2);
        bv_t11 = _mm512_mul_pd(v_CRTM_10_3, bv_s6);

        bv_s14 = _mm512_add_pd(bv_s12, bv_t1);
        bv_s15 = _mm512_add_pd(bv_t4, bv_t5);
        // Output point 2: X(1)
        v_out1 = _mm512_add_pd(bv_s14, bv_s15);
        // Output point 18: X(17)
        v_out17 = _mm512_sub_pd(bv_s14, bv_s15);

        bv_s16 = _mm512_sub_pd(bv_s13, bv_t2);
        bv_s17 = _mm512_add_pd(bv_t6, bv_t7);
        // Output point 3: X(2)
        v_out2 = _mm512_sub_pd(bv_s16, bv_s17);
        curr_out = out + out_strides[1];
        STRI_2x512_D(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 19: X(18)
        v_out18 = _mm512_add_pd(bv_s16, bv_s17);
        curr_out = out + out_strides[17];
        STRI_2x512_D(curr_out, v_out_stride, v_out17, v_out18);

        bv_s18 = _mm512_sub_pd(bv_s12, bv_t1);
        bv_s19 = _mm512_sub_pd(bv_t9, bv_t8);
        // Output point 6: X(5)
        v_out5 = _mm512_add_pd(bv_s18, bv_s19);
        // Output point 14: X(13)
        v_out13 = _mm512_sub_pd(bv_s18, bv_s19);

        bv_s20 = _mm512_add_pd(bv_s13, bv_t2);
        bv_s21 = _mm512_sub_pd(bv_t11, bv_t10);
        // Output point 7: X(6)
        v_out6 = _mm512_sub_pd(bv_s21, bv_s20);
        curr_out = out + out_strides[5];
        STRI_2x512_D(curr_out, v_out_stride, v_out5, v_out6);
        // Output point 15: X(14)
        v_out14 = NEGATE_512_D(_mm512_add_pd(bv_s21, bv_s20));
        curr_out = out + out_strides[13];
        STRI_2x512_D(curr_out, v_out_stride, v_out13, v_out14);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (n & 4)
    {
        /* Standard DFT */
        __m256d av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
                av_in8, av_in9;
        __m256d av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
                av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16,
                av_s17, av_s18, av_s19, av_s20, av_s21, av_s22, av_s23;
        __m256d av_t0, av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8,
                av_t9, av_t10, av_t11;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
                v_out15, v_out16, v_out17, v_out18, v_out19;

        curr_in = in;
        curr_out = out;

        __m256d v256_CRTM_10_1 = _mm512_castpd512_pd256(v_CRTM_10_1);
        __m256d v256_CRTM_10_2 = _mm512_castpd512_pd256(v_CRTM_10_2);
        __m256d v256_CRTM_10_3 = _mm512_castpd512_pd256(v_CRTM_10_3);
        __m256d v256_CRTM_10_4 = _mm512_castpd512_pd256(v_CRTM_10_4);

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
        // Input point 19: x(18)
        curr_in = in + in_strides[18];
        LDR_256_D(curr_in, v_in_stride, av_in9);

        av_s0 = _mm256_add_pd(av_in0, av_in5);
        av_s1 = _mm256_sub_pd(av_in0, av_in5);
        av_s2 = _mm256_add_pd(av_in1, av_in9);
        av_s3 = _mm256_sub_pd(av_in1, av_in9);
        av_s4 = _mm256_add_pd(av_in2, av_in3);
        av_s5 = _mm256_sub_pd(av_in2, av_in3);
        av_s6 = _mm256_add_pd(av_in4, av_in6);
        av_s7 = _mm256_sub_pd(av_in4, av_in6);
        av_s8 = _mm256_add_pd(av_in7, av_in8);
        av_s9 = _mm256_sub_pd(av_in7, av_in8);

        av_s10 = _mm256_add_pd(av_s2, av_s6);
        av_s14 = _mm256_add_pd(av_s4, av_s8);
        av_s18 = _mm256_add_pd(av_s10,av_s14);
        // Output pt 1: X(0)
        v_out0 = _mm256_add_pd(av_s0, av_s18);
        STR_256_D(curr_out, v_out_stride, v_out0);

        av_s11 = _mm256_sub_pd(av_s2, av_s6);
        av_s17 = _mm256_sub_pd(av_s5, av_s9);
        av_s20 = _mm256_add_pd(av_s11, av_s17);
        av_s21 = _mm256_sub_pd(av_s11, av_s17);
        // Output point 20: X(19)
        v_out19 = _mm256_sub_pd(av_s1, av_s21);
        curr_out = out + out_strides[19];
        STR_256_D(curr_out, v_out_stride, v_out19);

        av_t0  = _mm256_mul_pd(v256_CRTM_10_2, av_s21);
        av_t1  = _mm256_mul_pd(v256_CRTM_10_1, av_s20);
        av_s22 = _mm256_add_pd(av_t0, av_s1);
        // Output point 4: X(3)
        v_out3 = _mm256_add_pd(av_s22, av_t1);
        // Output point 12: X(11)
        v_out11 = _mm256_sub_pd(av_s22, av_t1);

        av_s12 = _mm256_add_pd(av_s3, av_s7);
        av_s15 = _mm256_sub_pd(av_s4, av_s8);
        av_t4  = _mm256_mul_pd(v256_CRTM_10_3, av_s12);
        av_t10 = _mm256_mul_pd(v256_CRTM_10_4, av_s15);
        // Output point 5: X(4)
        v_out4 = NEGATE_256_D(_mm256_add_pd(av_t4, av_t10));
        curr_out = out + out_strides[3];
        STRI_2x256_D(curr_out, v_out_stride, v_out3, v_out4);

        av_t6 = _mm256_mul_pd(v256_CRTM_10_3, av_s15);
        av_t8 = _mm256_mul_pd(v256_CRTM_10_4, av_s12);
        // Output point 13: X(12)
        v_out12 = _mm256_sub_pd(av_t6, av_t8);
        curr_out = out + out_strides[11];
        STRI_2x256_D(curr_out, v_out_stride, v_out11, v_out12);

        av_s13 = _mm256_sub_pd(av_s7, av_s3);
        av_s16 = _mm256_add_pd(av_s5, av_s9);
        av_t5  = _mm256_mul_pd(v256_CRTM_10_3, av_s13);
        av_t11 = _mm256_mul_pd(v256_CRTM_10_4, av_s16);
        // Output point 17: X(16)
        v_out16 = _mm256_add_pd(av_t11, av_t5);

        av_t7 = _mm256_mul_pd(v256_CRTM_10_3, av_s16);
        av_t9 = _mm256_mul_pd(v256_CRTM_10_4, av_s13);
        // Output point 9: X(8)
        v_out8 = _mm256_sub_pd(av_t9, av_t7);

        av_s19 = _mm256_sub_pd(av_s10, av_s14);
        av_t2  = _mm256_mul_pd(v256_CRTM_10_2, av_s18);
        av_t3  = _mm256_mul_pd(v256_CRTM_10_1, av_s19);
        av_s23 = _mm256_sub_pd(av_s0, av_t2);
        // Output point 8: X(7)
        v_out7 = _mm256_add_pd(av_s23, av_t3);
        curr_out = out + out_strides[7];
        STRI_2x256_D(curr_out, v_out_stride, v_out7, v_out8);
        // Output point 16: X(15)
        v_out15 = _mm256_sub_pd(av_s23, av_t3);
        curr_out = out + out_strides[15];
        STRI_2x256_D(curr_out, v_out_stride, v_out15, v_out16);

        /* Shifted DFT */
        __m256d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
                bv_in8, bv_in9;
        __m256d bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
                bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16,
                bv_s17, bv_s18, bv_s19, bv_s20, bv_s21;
        __m256d bv_t0, bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8,
                bv_t9, bv_t10, bv_t11;

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
        // Input point 20: x(19)
        curr_in = in + in_strides[19];
        LDR_256_D(curr_in, v_in_stride, bv_in9);

        bv_s0 = _mm256_add_pd(bv_in1, bv_in9);
        bv_s1 = _mm256_sub_pd(bv_in1, bv_in9);
        bv_s2 = _mm256_add_pd(bv_in2, bv_in8);
        bv_s3 = _mm256_sub_pd(bv_in2, bv_in8);
        bv_s4 = _mm256_add_pd(bv_in3, bv_in7);
        bv_s5 = _mm256_sub_pd(bv_in3, bv_in7);
        bv_s6 = _mm256_add_pd(bv_in4, bv_in6);
        bv_s7 = _mm256_sub_pd(bv_in4, bv_in6);

        bv_s8  = _mm256_add_pd(bv_s0, bv_s4);
        bv_s9  = _mm256_sub_pd(bv_s0, bv_s4);
        bv_s10 = _mm256_add_pd(bv_s3, bv_s7);
        bv_s11 = _mm256_sub_pd(bv_s3, bv_s7);
        // Output point 10: X(9)
        v_out9 = _mm256_sub_pd(bv_in0, bv_s11);
        // Output point 11: X(10)
        v_out10 = NEGATE_256_D(_mm256_add_pd(bv_s9, bv_in5));
        curr_out = out + out_strides[9];
        STRI_2x256_D(curr_out, v_out_stride, v_out9, v_out10);

        bv_t0 = _mm256_mul_pd(v256_CRTM_10_2, bv_s11);
        bv_t1 = _mm256_mul_pd(v256_CRTM_10_1, bv_s10);
        bv_t2 = _mm256_mul_pd(v256_CRTM_10_1, bv_s8);
        bv_t3 = _mm256_mul_pd(v256_CRTM_10_2, bv_s9);

        bv_s12 = _mm256_add_pd(bv_in0, bv_t0);
        bv_s13 = _mm256_sub_pd(bv_t3, bv_in5);

        bv_t4  = _mm256_mul_pd(v256_CRTM_10_4, bv_s1);
        bv_t5  = _mm256_mul_pd(v256_CRTM_10_3, bv_s5);
        bv_t6  = _mm256_mul_pd(v256_CRTM_10_4, bv_s6);
        bv_t7  = _mm256_mul_pd(v256_CRTM_10_3, bv_s2);
        bv_t8  = _mm256_mul_pd(v256_CRTM_10_4, bv_s5);
        bv_t9  = _mm256_mul_pd(v256_CRTM_10_3, bv_s1);
        bv_t10 = _mm256_mul_pd(v256_CRTM_10_4, bv_s2);
        bv_t11 = _mm256_mul_pd(v256_CRTM_10_3, bv_s6);

        bv_s14 = _mm256_add_pd(bv_s12, bv_t1);
        bv_s15 = _mm256_add_pd(bv_t4, bv_t5);
        // Output point 2: X(1)
        v_out1 = _mm256_add_pd(bv_s14, bv_s15);
        // Output point 18: X(17)
        v_out17 = _mm256_sub_pd(bv_s14, bv_s15);

        bv_s16 = _mm256_sub_pd(bv_s13, bv_t2);
        bv_s17 = _mm256_add_pd(bv_t6, bv_t7);
        // Output point 3: X(2)
        v_out2 = _mm256_sub_pd(bv_s16, bv_s17);
        curr_out = out + out_strides[1];
        STRI_2x256_D(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 19: X(18)
        v_out18 = _mm256_add_pd(bv_s16, bv_s17);
        curr_out = out + out_strides[17];
        STRI_2x256_D(curr_out, v_out_stride, v_out17, v_out18);

        bv_s18 = _mm256_sub_pd(bv_s12, bv_t1);
        bv_s19 = _mm256_sub_pd(bv_t9, bv_t8);
        // Output point 6: X(5)
        v_out5 = _mm256_add_pd(bv_s18, bv_s19);
        // Output point 14: X(13)
        v_out13 = _mm256_sub_pd(bv_s18, bv_s19);

        bv_s20 = _mm256_add_pd(bv_s13, bv_t2);
        bv_s21 = _mm256_sub_pd(bv_t11, bv_t10);
        // Output point 7: X(6)
        v_out6 = _mm256_sub_pd(bv_s21, bv_s20);
        curr_out = out + out_strides[5];
        STRI_2x256_D(curr_out, v_out_stride, v_out5, v_out6);
        // Output point 15: X(14)
        v_out14 = NEGATE_256_D(_mm256_add_pd(bv_s21, bv_s20));
        curr_out = out + out_strides[13];
        STRI_2x256_D(curr_out, v_out_stride, v_out13, v_out14);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (n & 2)
    {
        /* Standard DFT */
        __m128d av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
                av_in8, av_in9;
        __m128d av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
                av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16,
                av_s17, av_s18, av_s19, av_s20, av_s21, av_s22, av_s23;
        __m128d av_t0, av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8,
                av_t9, av_t10, av_t11;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
                v_out15, v_out16, v_out17, v_out18, v_out19;

        curr_in = in;
        curr_out = out;

        __m128d v128_CRTM_10_1 = _mm512_castpd512_pd128(v_CRTM_10_1);
        __m128d v128_CRTM_10_2 = _mm512_castpd512_pd128(v_CRTM_10_2);
        __m128d v128_CRTM_10_3 = _mm512_castpd512_pd128(v_CRTM_10_3);
        __m128d v128_CRTM_10_4 = _mm512_castpd512_pd128(v_CRTM_10_4);

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
        // Input point 19: x(18)
        curr_in = in + in_strides[18];
        LDR_128_D(curr_in, v_in_stride, av_in9);

        av_s0 = _mm_add_pd(av_in0, av_in5);
        av_s1 = _mm_sub_pd(av_in0, av_in5);
        av_s2 = _mm_add_pd(av_in1, av_in9);
        av_s3 = _mm_sub_pd(av_in1, av_in9);
        av_s4 = _mm_add_pd(av_in2, av_in3);
        av_s5 = _mm_sub_pd(av_in2, av_in3);
        av_s6 = _mm_add_pd(av_in4, av_in6);
        av_s7 = _mm_sub_pd(av_in4, av_in6);
        av_s8 = _mm_add_pd(av_in7, av_in8);
        av_s9 = _mm_sub_pd(av_in7, av_in8);

        av_s10 = _mm_add_pd(av_s2, av_s6);
        av_s14 = _mm_add_pd(av_s4, av_s8);
        av_s18 = _mm_add_pd(av_s10,av_s14);
        // Output pt 1: X(0)
        v_out0 = _mm_add_pd(av_s0, av_s18);
        STR_128_D(curr_out, v_out_stride, v_out0);

        av_s11 = _mm_sub_pd(av_s2, av_s6);
        av_s17 = _mm_sub_pd(av_s5, av_s9);
        av_s20 = _mm_add_pd(av_s11, av_s17);
        av_s21 = _mm_sub_pd(av_s11, av_s17);
        // Output point 20: X(19)
        v_out19 = _mm_sub_pd(av_s1, av_s21);
        curr_out = out + out_strides[19];
        STR_128_D(curr_out, v_out_stride, v_out19);

        av_t0  = _mm_mul_pd(v128_CRTM_10_2, av_s21);
        av_t1  = _mm_mul_pd(v128_CRTM_10_1, av_s20);
        av_s22 = _mm_add_pd(av_t0, av_s1);
        // Output point 4: X(3)
        v_out3 = _mm_add_pd(av_s22, av_t1);
        // Output point 12: X(11)
        v_out11 = _mm_sub_pd(av_s22, av_t1);

        av_s12 = _mm_add_pd(av_s3, av_s7);
        av_s15 = _mm_sub_pd(av_s4, av_s8);
        av_t4  = _mm_mul_pd(v128_CRTM_10_3, av_s12);
        av_t10 = _mm_mul_pd(v128_CRTM_10_4, av_s15);
        // Output point 5: X(4)
        v_out4 = NEGATE_128_D(_mm_add_pd(av_t4, av_t10));
        curr_out = out + out_strides[3];
        STRI_2x128_D(curr_out, v_out_stride, v_out3, v_out4);

        av_t6 = _mm_mul_pd(v128_CRTM_10_3, av_s15);
        av_t8 = _mm_mul_pd(v128_CRTM_10_4, av_s12);
        // Output point 13: X(12)
        v_out12 = _mm_sub_pd(av_t6, av_t8);
        curr_out = out + out_strides[11];
        STRI_2x128_D(curr_out, v_out_stride, v_out11, v_out12);

        av_s13 = _mm_sub_pd(av_s7, av_s3);
        av_s16 = _mm_add_pd(av_s5, av_s9);
        av_t5  = _mm_mul_pd(v128_CRTM_10_3, av_s13);
        av_t11 = _mm_mul_pd(v128_CRTM_10_4, av_s16);
        // Output point 17: X(16)
        v_out16 = _mm_add_pd(av_t11, av_t5);

        av_t7 = _mm_mul_pd(v128_CRTM_10_3, av_s16);
        av_t9 = _mm_mul_pd(v128_CRTM_10_4, av_s13);
        // Output point 9: X(8)
        v_out8 = _mm_sub_pd(av_t9, av_t7);

        av_s19 = _mm_sub_pd(av_s10, av_s14);
        av_t2  = _mm_mul_pd(v128_CRTM_10_2, av_s18);
        av_t3  = _mm_mul_pd(v128_CRTM_10_1, av_s19);
        av_s23 = _mm_sub_pd(av_s0, av_t2);
        // Output point 8: X(7)
        v_out7 = _mm_add_pd(av_s23, av_t3);
        curr_out = out + out_strides[7];
        STRI_2x128_D(curr_out, v_out_stride, v_out7, v_out8);
        // Output point 16: X(15)
        v_out15 = _mm_sub_pd(av_s23, av_t3);
        curr_out = out + out_strides[15];
        STRI_2x128_D(curr_out, v_out_stride, v_out15, v_out16);

        /* Shifted DFT */
        __m128d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
                bv_in8, bv_in9;
        __m128d bv_s0, bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8,
                bv_s9, bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16,
                bv_s17, bv_s18, bv_s19, bv_s20, bv_s21;
        __m128d bv_t0, bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8,
                bv_t9, bv_t10, bv_t11;

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
        // Input point 20: x(19)
        curr_in = in + in_strides[19];
        LDR_128_D(curr_in, v_in_stride, bv_in9);

        bv_s0 = _mm_add_pd(bv_in1, bv_in9);
        bv_s1 = _mm_sub_pd(bv_in1, bv_in9);
        bv_s2 = _mm_add_pd(bv_in2, bv_in8);
        bv_s3 = _mm_sub_pd(bv_in2, bv_in8);
        bv_s4 = _mm_add_pd(bv_in3, bv_in7);
        bv_s5 = _mm_sub_pd(bv_in3, bv_in7);
        bv_s6 = _mm_add_pd(bv_in4, bv_in6);
        bv_s7 = _mm_sub_pd(bv_in4, bv_in6);

        bv_s8  = _mm_add_pd(bv_s0, bv_s4);
        bv_s9  = _mm_sub_pd(bv_s0, bv_s4);
        bv_s10 = _mm_add_pd(bv_s3, bv_s7);
        bv_s11 = _mm_sub_pd(bv_s3, bv_s7);
        // Output point 10: X(9)
        v_out9 = _mm_sub_pd(bv_in0, bv_s11);
        // Output point 11: X(10)
        v_out10 = NEGATE_128_D(_mm_add_pd(bv_s9, bv_in5));
        curr_out = out + out_strides[9];
        STRI_2x128_D(curr_out, v_out_stride, v_out9, v_out10);

        bv_t0 = _mm_mul_pd(v128_CRTM_10_2, bv_s11);
        bv_t1 = _mm_mul_pd(v128_CRTM_10_1, bv_s10);
        bv_t2 = _mm_mul_pd(v128_CRTM_10_1, bv_s8);
        bv_t3 = _mm_mul_pd(v128_CRTM_10_2, bv_s9);

        bv_s12 = _mm_add_pd(bv_in0, bv_t0);
        bv_s13 = _mm_sub_pd(bv_t3, bv_in5);

        bv_t4  = _mm_mul_pd(v128_CRTM_10_4, bv_s1);
        bv_t5  = _mm_mul_pd(v128_CRTM_10_3, bv_s5);
        bv_t6  = _mm_mul_pd(v128_CRTM_10_4, bv_s6);
        bv_t7  = _mm_mul_pd(v128_CRTM_10_3, bv_s2);
        bv_t8  = _mm_mul_pd(v128_CRTM_10_4, bv_s5);
        bv_t9  = _mm_mul_pd(v128_CRTM_10_3, bv_s1);
        bv_t10 = _mm_mul_pd(v128_CRTM_10_4, bv_s2);
        bv_t11 = _mm_mul_pd(v128_CRTM_10_3, bv_s6);

        bv_s14 = _mm_add_pd(bv_s12, bv_t1);
        bv_s15 = _mm_add_pd(bv_t4, bv_t5);
        // Output point 2: X(1)
        v_out1 = _mm_add_pd(bv_s14, bv_s15);
        // Output point 18: X(17)
        v_out17 = _mm_sub_pd(bv_s14, bv_s15);

        bv_s16 = _mm_sub_pd(bv_s13, bv_t2);
        bv_s17 = _mm_add_pd(bv_t6, bv_t7);
        // Output point 3: X(2)
        v_out2 = _mm_sub_pd(bv_s16, bv_s17);
        curr_out = out + out_strides[1];
        STRI_2x128_D(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 19: X(18)
        v_out18 = _mm_add_pd(bv_s16, bv_s17);
        curr_out = out + out_strides[17];
        STRI_2x128_D(curr_out, v_out_stride, v_out17, v_out18);

        bv_s18 = _mm_sub_pd(bv_s12, bv_t1);
        bv_s19 = _mm_sub_pd(bv_t9, bv_t8);
        // Output point 6: X(5)
        v_out5 = _mm_add_pd(bv_s18, bv_s19);
        // Output point 14: X(13)
        v_out13 = _mm_sub_pd(bv_s18, bv_s19);

        bv_s20 = _mm_add_pd(bv_s13, bv_t2);
        bv_s21 = _mm_sub_pd(bv_t11, bv_t10);
        // Output point 7: X(6)
        v_out6 = _mm_sub_pd(bv_s21, bv_s20);
        curr_out = out + out_strides[5];
        STRI_2x128_D(curr_out, v_out_stride, v_out5, v_out6);
        // Output point 15: X(14)
        v_out14 = NEGATE_128_D(_mm_add_pd(bv_s21, bv_s20));
        curr_out = out + out_strides[13];
        STRI_2x128_D(curr_out, v_out_stride, v_out13, v_out14);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        /* Standard DFT */
        DOUBLE a_in0, a_in1, a_in2, a_in3, a_in4, a_in5, a_in6, a_in7, a_in8,
               a_in9;
        DOUBLE a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_s9,
               a_s10, a_s11, a_s12, a_s13, a_s14, a_s15, a_s16, a_s17, a_s18,
               a_s19, a_s20, a_s21, a_s22, a_s23, a_t0, a_t1, a_t2, a_t3, a_t4,
               a_t5, a_t6, a_t7, a_t8, a_t9, a_t10, a_t11;

        // Input point 1: x(0)
        a_in0 = *in;
        // Input point 3: x(2)
        a_in1 = in[in_strides[2]];
        // Input point 5: x(4)
        a_in2 = in[in_strides[4]];
        // Input point 7: x(6)
        a_in3 = in[in_strides[6]];
        // Input point 9: x(8)
        a_in4 = in[in_strides[8]];
        // Input point 11: x(10)
        a_in5 = in[in_strides[10]];
        // Input point 13: x(12)
        a_in6 = in[in_strides[12]];
        // Input point 15: x(14)
        a_in7 = in[in_strides[14]];
        // Input point 17: x(16)
        a_in8 = in[in_strides[16]];
        // Input point 19: x(18)
        a_in9 = in[in_strides[18]];

        a_s0 = a_in0 + a_in5;
        a_s1 = a_in0 - a_in5;
        a_s2 = a_in1 + a_in9;
        a_s3 = a_in1 - a_in9;
        a_s4 = a_in2 + a_in3;
        a_s5 = a_in2 - a_in3;
        a_s6 = a_in4 + a_in6;
        a_s7 = a_in4 - a_in6;
        a_s8 = a_in7 + a_in8;
        a_s9 = a_in7 - a_in8;

        a_s10 = a_s2 + a_s6;
        a_s14 = a_s4 + a_s8;
        a_s18 = a_s10 + a_s14;
        // Output point 1: X(0)
        *out = a_s0 + a_s18;

        a_s11 = a_s2 - a_s6;
        a_s17 = a_s5 - a_s9;
        a_s20 = a_s11 + a_s17;
        a_s21 = a_s11 - a_s17;
        // Output point 20: X(19)
        out[out_strides[19]] = a_s1 - a_s21;

        a_t0 = CRTM_10_2 * a_s21;
        a_t1 = CRTM_10_1 * a_s20;
        a_s22 = a_t0 + a_s1;
        // Output point 4: X(3)
        out[out_strides[3]] = a_s22 + a_t1;
        // Output point 12: X(11)
        out[out_strides[11]] = a_s22 - a_t1;

        a_s12 = a_s3 + a_s7;
        a_s15 = a_s4 - a_s8;
        a_t2 = CRTM_10_3 * a_s12;
        a_t3 = CRTM_10_4 * a_s15;
        // Output point 5: X(4)
        out[out_strides[4]] = -(a_t2 + a_t3);

        a_t4 = CRTM_10_3 * a_s15;
        a_t5 = CRTM_10_4 * a_s12;
        // Output point 13: X(12)
        out[out_strides[12]] = a_t4 - a_t5;

        a_s13 = a_s7 - a_s3;
        a_s16 = a_s5 + a_s9;
        a_t6 = CRTM_10_3 * a_s13;
        a_t7 = CRTM_10_4 * a_s16;
        // Output point 17: X(16)
        out[out_strides[16]] = a_t7 + a_t6;

        a_t8 = CRTM_10_3 * a_s16;
        a_t9 = CRTM_10_4 * a_s13;
        // Output point 9: X(8)
        out[out_strides[8]] = a_t9 - a_t8;

        a_s19 = a_s10 - a_s14;
        a_t10 = CRTM_10_2 * a_s18;
        a_t11 = CRTM_10_1 * a_s19;
        a_s23 = a_s0 - a_t10;
        // Output point 8: X(7)
        out[out_strides[7]] = a_s23 + a_t11;
        // Output point 16: X(15)
        out[out_strides[15]] = a_s23 - a_t11;

        /* Shifted DFT */
        DOUBLE b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7, b_in8,
               b_in9;
        DOUBLE b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9,
               b_s10, b_s11, b_s12, b_s13,b_s14, b_s15, b_s16, b_s17, b_s18,
               b_s19, b_s20, b_s21;
        DOUBLE b_t0, b_t1, b_t2, b_t3, b_t4, b_t5, b_t6, b_t7, b_t8, b_t9,
               b_t10, b_t11;

        // Input point 2: x(1)
        b_in0 = in[in_strides[1]];
        // Input point 4: x(3)
        b_in1 = in[in_strides[3]];
        // Input point 6: x(5)
        b_in2 = in[in_strides[5]];
        // Input point 8: x(7)
        b_in3 = in[in_strides[7]];
        // Input point 10: x(9)
        b_in4 = in[in_strides[9]];
        // Input point 12: x(11)
        b_in5 = in[in_strides[11]];
        // Input point 14: x(13)
        b_in6 = in[in_strides[13]];
        // Input point 16: x(15)
        b_in7 = in[in_strides[15]];
        // Input point 18: x(17)
        b_in8 = in[in_strides[17]];
        // Input point 20: x(19)
        b_in9 = in[in_strides[19]];

        b_s0 = b_in1 + b_in9;
        b_s1 = b_in1 - b_in9;
        b_s2 = b_in2 + b_in8;
        b_s3 = b_in2 - b_in8;
        b_s4 = b_in3 + b_in7;
        b_s5 = b_in3 - b_in7;
        b_s6 = b_in4 + b_in6;
        b_s7 = b_in4 - b_in6;

        b_s8 = b_s0 + b_s4;
        b_s9 = b_s0 - b_s4;
        b_s10 = b_s3 + b_s7;
        b_s11 = b_s3 - b_s7;
        // Output point 11: X(10)
        out[out_strides[10]] = -(b_s9 + b_in5);
        // Output point 10: X(9)
        out[out_strides[9]] = b_in0 - b_s11;

        b_t0 = CRTM_10_2 * b_s11;
        b_t1 = CRTM_10_1 * b_s10;
        b_t2 = CRTM_10_1 * b_s8;
        b_t3 = CRTM_10_2 * b_s9;

        b_s12 = b_in0 + b_t0;
        b_s13 = b_t3 - b_in5;

        b_t4 = CRTM_10_4 * b_s1;
        b_t5 = CRTM_10_3 * b_s5;
        b_t6 = CRTM_10_4 * b_s6;
        b_t7 = CRTM_10_3 * b_s2;
        b_t8 = CRTM_10_4 * b_s5;
        b_t9 = CRTM_10_3 * b_s1;
        b_t10 = CRTM_10_4 * b_s2;
        b_t11 = CRTM_10_3 * b_s6;

        b_s14 = b_s12 + b_t1;
        b_s15 = b_t4 + b_t5;
        // Output point 2: X(1)
        out[out_strides[1]] = b_s14 + b_s15;
        // Output point 18: X(17)
        out[out_strides[17]] = b_s14 - b_s15;

        b_s16 = b_s13 - b_t2;
        b_s17 = b_t6 + b_t7;
        // Output point 3: X(2)
        out[out_strides[2]] = b_s16 - b_s17;
        // Output point 19: X(18)
        out[out_strides[18]] = b_s16 + b_s17;

        b_s18 = b_s12 - b_t1;
        b_s19 = b_t9 - b_t8;
        // Output point 6: X(5)
        out[out_strides[5]] = b_s18 + b_s19;
        // Output point 14: X(13)
        out[out_strides[13]] = b_s18 - b_s19;

        b_s20 = b_s13 + b_t2;
        b_s21 = b_t11 - b_t10;
        // Output point 7: X(6)
        out[out_strides[6]] = b_s21 - b_s20;
        // Output point 15: X(14)
        out[out_strides[14]] = -(b_s21 + b_s20);
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID r2hcf_rfft10avx512_fp64_bwd(VOID *in_real, VOID *in_imag,
                                        VOID *out_real, VOID *out_imag, INTP n,
                                        aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const DOUBLE CRTM_10_1 = 1.118033988749894848204586834365638117720309180;
    const DOUBLE CRTM_10_2 = 0.500000000000000000000000000000000000000000000;
    const DOUBLE CRTM_10_3 = 2.000000000000000000000000000000000000000000000;
    const DOUBLE CRTM_10_4 = 1.175570504584946258337411909278145537195304875;
    const DOUBLE CRTM_10_5 = 1.902113032590307144232878666758764286811397268;

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

    __m512d v_CRTM_10_1 = _mm512_set1_pd(CRTM_10_1);
    __m512d v_CRTM_10_2 = _mm512_set1_pd(CRTM_10_2);
    __m512d v_CRTM_10_3 = _mm512_set1_pd(CRTM_10_3);
    __m512d v_CRTM_10_4 = _mm512_set1_pd(CRTM_10_4);
    __m512d v_CRTM_10_5 = _mm512_set1_pd(CRTM_10_5);

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m512d av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
                av_in8, av_in9;
        __m512d av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8, av_s9,
                av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16, av_s17,
                av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24;
        __m512d av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8, av_t9,
                av_t10, av_t11, av_t12, av_t13, av_t14;
        __m512d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
                v_out15, v_out16, v_out17, v_out18, v_out19;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_512_D(curr_in, v_in_stride, av_in0);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x512_D(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in + in_strides[7];
        LDRI_2x512_D(curr_in, v_in_stride, av_in3, av_in4);
        // Input point 12: x(11) & Input point 13: x(12)
        curr_in = in + in_strides[11];
        LDRI_2x512_D(curr_in, v_in_stride, av_in5, av_in6);
        // Input point 16: x(15) & Input point 17: x(16)
        curr_in = in + in_strides[15];
        LDRI_2x512_D(curr_in, v_in_stride, av_in7, av_in8);
        // Input point 20: x(19)
        curr_in = in + in_strides[19];
        LDR_512_D(curr_in, v_in_stride, av_in9);

        av_s1  = _mm512_add_pd(av_in0, av_in9);
        av_s2  = _mm512_sub_pd(av_in0, av_in9);
        av_s3  = _mm512_add_pd(av_in1, av_in7);
        av_s4  = _mm512_sub_pd(av_in1, av_in7);
        av_s5  = _mm512_add_pd(av_in2, av_in8);
        av_s6  = _mm512_sub_pd(av_in2, av_in8);
        av_t11 = _mm512_mul_pd(v_CRTM_10_4, av_s6);
        av_s9  = _mm512_add_pd(av_in4, av_in6);

        av_t9  = _mm512_mul_pd(v_CRTM_10_4, av_s9);
        av_t10 = _mm512_mul_pd(v_CRTM_10_5, av_s5);
        av_s23 = _mm512_sub_pd(av_t9, av_t10);
        av_s10 = _mm512_sub_pd(av_in4, av_in6);
        av_t12 = _mm512_mul_pd(v_CRTM_10_5, av_s10);
        av_s24 = _mm512_sub_pd(av_t12, av_t11);

        av_s7  = _mm512_add_pd(av_in3, av_in5);
        av_s8  = _mm512_sub_pd(av_in3, av_in5);
        av_s14 = _mm512_sub_pd(av_s4, av_s8);
        av_t14 = _mm512_mul_pd(v_CRTM_10_3, av_s14);
        // Output point 11: X(10)
        v_out10 = _mm512_sub_pd(av_s2, av_t14);
        curr_out = out + out_strides[10];
        STR_512_D(curr_out, v_out_stride, v_out10);

        av_s11 = _mm512_add_pd(av_s3, av_s7);
        av_t13 = _mm512_mul_pd(v_CRTM_10_3, av_s11);
        // Output point 1: X(0)
        v_out0 = _mm512_add_pd(av_s1, av_t13);
        curr_out = out + out_strides[0];
        STR_512_D(curr_out, v_out_stride, v_out0);

        av_t1  = _mm512_mul_pd(v_CRTM_10_2, av_s11);
        av_s16 = _mm512_sub_pd(av_s1, av_t1);
        av_s12 = _mm512_sub_pd(av_s3, av_s7);
        av_t3  = _mm512_mul_pd(v_CRTM_10_1, av_s12);
        av_s19 = _mm512_add_pd(av_s16, av_t3);
        av_s20 = _mm512_sub_pd(av_s16, av_t3);
        // Output point 9: X(8)
        v_out8 = _mm512_add_pd(av_s20, av_s24);
        curr_out = out + out_strides[8];
        STR_512_D(curr_out, v_out_stride, v_out8);
        // Output point 13: X(12)
        v_out12 = _mm512_sub_pd(av_s20, av_s24);
        curr_out = out + out_strides[12];
        STR_512_D(curr_out, v_out_stride, v_out12);

        av_s13 = _mm512_add_pd(av_s4, av_s8);
        av_t4  = _mm512_mul_pd(v_CRTM_10_1, av_s13);
        av_t2  = _mm512_mul_pd(v_CRTM_10_2, av_s14);
        av_s15 = _mm512_add_pd(av_s2, av_t2);
        av_s17 = _mm512_add_pd(av_s15, av_t4);
        av_s18 = _mm512_sub_pd(av_s15, av_t4);
        // Output point 7: X(6)
        v_out6 = _mm512_add_pd(av_s18, av_s23);
        curr_out = out + out_strides[6];
        STR_512_D(curr_out, v_out_stride, v_out6);
        // Output point 15: X(14)
        v_out14 = _mm512_sub_pd(av_s18, av_s23);
        curr_out = out + out_strides[14];
        STR_512_D(curr_out, v_out_stride, v_out14);

        av_t5  = _mm512_mul_pd(v_CRTM_10_4, av_s5);
        av_t6  = _mm512_mul_pd(v_CRTM_10_5, av_s9);
        av_s21 = _mm512_add_pd(av_t5, av_t6);
        // Output point 3: X(2)
        v_out2 = _mm512_sub_pd(av_s17, av_s21);
        curr_out = out + out_strides[2];
        STR_512_D(curr_out, v_out_stride, v_out2);
        // Output point 19: X(18)
        v_out18 = _mm512_add_pd(av_s17, av_s21);
        curr_out = out + out_strides[18];
        STR_512_D(curr_out, v_out_stride, v_out18);

        av_t7  = _mm512_mul_pd(v_CRTM_10_4, av_s10);
        av_t8  = _mm512_mul_pd(v_CRTM_10_5, av_s6);
        av_s22 = _mm512_add_pd(av_t7, av_t8);
        // Output point 5: X(4)
        v_out4 = _mm512_sub_pd(av_s19, av_s22);
        curr_out = out + out_strides[4];
        STR_512_D(curr_out, v_out_stride, v_out4);
        // Output point 17: X(16)
        v_out16 = _mm512_add_pd(av_s19, av_s22);
        curr_out = out + out_strides[16];
        STR_512_D(curr_out, v_out_stride, v_out16);

        /* Shifted DFT */
        __m512d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
                bv_in8, bv_in9;
        __m512d bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
                bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17,
                bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24;
        __m512d bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
                bv_t10, bv_t11, bv_t12, bv_t13, bv_t14;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x512_D(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDRI_2x512_D(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDRI_2x512_D(curr_in, v_in_stride, bv_in4, bv_in5);
        // Input point 14: x(13) & Input point 15: x(14)
        curr_in = in + in_strides[13];
        LDRI_2x512_D(curr_in, v_in_stride, bv_in6, bv_in7);
        // Input point 18: x(17) & Input point 19: x(18)
        curr_in = in + in_strides[17];
        LDRI_2x512_D(curr_in, v_in_stride, bv_in8, bv_in9);

        bv_s1 = _mm512_add_pd(bv_in0, bv_in8);
        bv_s2 = _mm512_sub_pd(bv_in0, bv_in8);
        bv_s3 = _mm512_add_pd(bv_in1, bv_in9);
        bv_s4 = _mm512_sub_pd(bv_in1, bv_in9);
        bv_s5 = _mm512_add_pd(bv_in2, bv_in6);
        bv_s6 = _mm512_sub_pd(bv_in2, bv_in6);
        bv_s7 = _mm512_add_pd(bv_in3, bv_in7);
        bv_s8 = _mm512_sub_pd(bv_in3, bv_in7);

        bv_s9  = _mm512_add_pd(bv_s1, bv_s5);
        bv_s10 = _mm512_sub_pd(bv_s1, bv_s5);
        bv_s11 = _mm512_add_pd(bv_s3, bv_s7);
        bv_s12 = _mm512_sub_pd(bv_s3, bv_s7);

        bv_t1  = _mm512_mul_pd(v_CRTM_10_1, bv_s11);
        bv_t2  = _mm512_mul_pd(v_CRTM_10_2, bv_s12);
        bv_t3  = _mm512_mul_pd(v_CRTM_10_1, bv_s10);
        bv_t4  = _mm512_mul_pd(v_CRTM_10_2, bv_s9);
        bv_t5  = _mm512_mul_pd(v_CRTM_10_3, bv_in4);
        bv_t6  = _mm512_mul_pd(v_CRTM_10_3, bv_in5);
        bv_s23 = _mm512_add_pd(bv_s9, bv_s9);
        // Output point 2: X(1)
        v_out1 = _mm512_add_pd(bv_t5, bv_s23);
        curr_out = out + out_strides[1];
        STR_512_D(curr_out, v_out_stride, v_out1);

        bv_s24 = _mm512_add_pd(bv_s12, bv_s12);
        // Output point 12: X(11)
        v_out11 = NEGATE_512_D(_mm512_add_pd(bv_t6, bv_s24));
        curr_out = out + out_strides[11];
        STR_512_D(curr_out, v_out_stride, v_out11);

        bv_s13 = _mm512_sub_pd(bv_t2, bv_t6);
        bv_s14 = _mm512_sub_pd(bv_t4, bv_t5);

        bv_s15 = _mm512_sub_pd(bv_s13, bv_t1);
        bv_s16 = _mm512_add_pd(bv_s13, bv_t1);
        bv_s17 = _mm512_add_pd(bv_t3, bv_s14);
        bv_s18 = _mm512_sub_pd(bv_t3, bv_s14);

        bv_t7  = _mm512_mul_pd(v_CRTM_10_5, bv_s2);
        bv_t8  = _mm512_mul_pd(v_CRTM_10_5, bv_s4);
        bv_t9  = _mm512_mul_pd(v_CRTM_10_5, bv_s6);
        bv_t10 = _mm512_mul_pd(v_CRTM_10_5, bv_s8);
        bv_t11 = _mm512_mul_pd(v_CRTM_10_4, bv_s2);
        bv_t12 = _mm512_mul_pd(v_CRTM_10_4, bv_s4);
        bv_t13 = _mm512_mul_pd(v_CRTM_10_4, bv_s6);
        bv_t14 = _mm512_mul_pd(v_CRTM_10_4, bv_s8);

        bv_s19 = _mm512_add_pd(bv_t7, bv_t13);
        // Output point 4: X(3)
        v_out3 = _mm512_add_pd(bv_s15, bv_s19);
        curr_out = out + out_strides[3];
        STR_512_D(curr_out, v_out_stride, v_out3);
        // Output point 20: X(19)
        v_out19 = _mm512_sub_pd(bv_s15, bv_s19);
        curr_out = out + out_strides[19];
        STR_512_D(curr_out, v_out_stride, v_out19);

        bv_s20 = _mm512_add_pd(bv_t10, bv_t12);
        // Output point 6: X(5)
        v_out5 = _mm512_sub_pd(bv_s17, bv_s20);
        curr_out = out + out_strides[5];
        STR_512_D(curr_out, v_out_stride, v_out5);
        // Output point 18: X(17)
        v_out17 = NEGATE_512_D(_mm512_add_pd(bv_s17, bv_s20));
        curr_out = out + out_strides[17];
        STR_512_D(curr_out, v_out_stride, v_out17);

        bv_s21 = _mm512_sub_pd(bv_t11, bv_t9);
        // Output point 8: X(7)
        v_out7 = _mm512_sub_pd(bv_s21, bv_s16);
        curr_out = out + out_strides[7];
        STR_512_D(curr_out, v_out_stride, v_out7);
        // Output point 16: X(15)
        v_out15 = NEGATE_512_D(_mm512_add_pd(bv_s16, bv_s21));
        curr_out = out + out_strides[15];
        STR_512_D(curr_out, v_out_stride, v_out15);

        bv_s22 = _mm512_sub_pd(bv_t14, bv_t8);
        // Output point 10: X(9)
        v_out9 = _mm512_add_pd(bv_s22, bv_s18);
        curr_out = out + out_strides[9];
        STR_512_D(curr_out, v_out_stride, v_out9);
        // Output point 14: X(13)
        v_out13 = _mm512_sub_pd(bv_s22, bv_s18);
        curr_out = out + out_strides[13];
        STR_512_D(curr_out, v_out_stride, v_out13);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (n & 4)
    {
        /* Standard DFT */
        __m256d av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
                av_in8, av_in9;
        __m256d av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8, av_s9,
                av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16, av_s17,
                av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24;
        __m256d av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8, av_t9,
                av_t10, av_t11, av_t12, av_t13, av_t14;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
                v_out15, v_out16, v_out17, v_out18, v_out19;

        curr_in = in;
        curr_out = out;

        __m256d v256_CRTM_10_1 = _mm512_castpd512_pd256(v_CRTM_10_1);
        __m256d v256_CRTM_10_2 = _mm512_castpd512_pd256(v_CRTM_10_2);
        __m256d v256_CRTM_10_3 = _mm512_castpd512_pd256(v_CRTM_10_3);
        __m256d v256_CRTM_10_4 = _mm512_castpd512_pd256(v_CRTM_10_4);
        __m256d v256_CRTM_10_5 = _mm512_castpd512_pd256(v_CRTM_10_5);

        // Input point 1: x(0)
        LDR_256_D(curr_in, v_in_stride, av_in0);
        // Input point 4: x(3) & Input point 5: x(4)
        curr_in = in + in_strides[3];
        LDRI_2x256_D(curr_in, v_in_stride, av_in1, av_in2);
        // Input point 8: x(7) & Input point 9: x(8)
        curr_in = in + in_strides[7];
        LDRI_2x256_D(curr_in, v_in_stride, av_in3, av_in4);
        // Input point 12: x(11) & Input point 13: x(12)
        curr_in = in + in_strides[11];
        LDRI_2x256_D(curr_in, v_in_stride, av_in5, av_in6);
        // Input point 16: x(15) & Input point 17: x(16)
        curr_in = in + in_strides[15];
        LDRI_2x256_D(curr_in, v_in_stride, av_in7, av_in8);
        // Input point 20: x(19)
        curr_in = in + in_strides[19];
        LDR_256_D(curr_in, v_in_stride, av_in9);

        av_s1  = _mm256_add_pd(av_in0, av_in9);
        av_s2  = _mm256_sub_pd(av_in0, av_in9);
        av_s3  = _mm256_add_pd(av_in1, av_in7);
        av_s4  = _mm256_sub_pd(av_in1, av_in7);
        av_s5  = _mm256_add_pd(av_in2, av_in8);
        av_s6  = _mm256_sub_pd(av_in2, av_in8);
        av_t11 = _mm256_mul_pd(v256_CRTM_10_4, av_s6);
        av_s9  = _mm256_add_pd(av_in4, av_in6);

        av_t9  = _mm256_mul_pd(v256_CRTM_10_4, av_s9);
        av_t10 = _mm256_mul_pd(v256_CRTM_10_5, av_s5);
        av_s23 = _mm256_sub_pd(av_t9, av_t10);
        av_s10 = _mm256_sub_pd(av_in4, av_in6);
        av_t12 = _mm256_mul_pd(v256_CRTM_10_5, av_s10);
        av_s24 = _mm256_sub_pd(av_t12, av_t11);

        av_s7  = _mm256_add_pd(av_in3, av_in5);
        av_s8  = _mm256_sub_pd(av_in3, av_in5);
        av_s14 = _mm256_sub_pd(av_s4, av_s8);
        av_t14 = _mm256_mul_pd(v256_CRTM_10_3, av_s14);
        // Output point 11: X(10)
        v_out10 = _mm256_sub_pd(av_s2, av_t14);
        curr_out = out + out_strides[10];
        STR_256_D(curr_out, v_out_stride, v_out10);

        av_s11 = _mm256_add_pd(av_s3, av_s7);
        av_t13 = _mm256_mul_pd(v256_CRTM_10_3, av_s11);
        // Output point 1: X(0)
        v_out0 = _mm256_add_pd(av_s1, av_t13);
        curr_out = out + out_strides[0];
        STR_256_D(curr_out, v_out_stride, v_out0);

        av_t1  = _mm256_mul_pd(v256_CRTM_10_2, av_s11);
        av_s16 = _mm256_sub_pd(av_s1, av_t1);
        av_s12 = _mm256_sub_pd(av_s3, av_s7);
        av_t3  = _mm256_mul_pd(v256_CRTM_10_1, av_s12);
        av_s19 = _mm256_add_pd(av_s16, av_t3);
        av_s20 = _mm256_sub_pd(av_s16, av_t3);
        // Output point 9: X(8)
        v_out8 = _mm256_add_pd(av_s20, av_s24);
        curr_out = out + out_strides[8];
        STR_256_D(curr_out, v_out_stride, v_out8);
        // Output point 13: X(12)
        v_out12 = _mm256_sub_pd(av_s20, av_s24);
        curr_out = out + out_strides[12];
        STR_256_D(curr_out, v_out_stride, v_out12);

        av_s13 = _mm256_add_pd(av_s4, av_s8);
        av_t4  = _mm256_mul_pd(v256_CRTM_10_1, av_s13);
        av_t2  = _mm256_mul_pd(v256_CRTM_10_2, av_s14);
        av_s15 = _mm256_add_pd(av_s2, av_t2);
        av_s17 = _mm256_add_pd(av_s15, av_t4);
        av_s18 = _mm256_sub_pd(av_s15, av_t4);
        // Output point 7: X(6)
        v_out6 = _mm256_add_pd(av_s18, av_s23);
        curr_out = out + out_strides[6];
        STR_256_D(curr_out, v_out_stride, v_out6);
        // Output point 15: X(14)
        v_out14 = _mm256_sub_pd(av_s18, av_s23);
        curr_out = out + out_strides[14];
        STR_256_D(curr_out, v_out_stride, v_out14);

        av_t5  = _mm256_mul_pd(v256_CRTM_10_4, av_s5);
        av_t6  = _mm256_mul_pd(v256_CRTM_10_5, av_s9);
        av_s21 = _mm256_add_pd(av_t5, av_t6);
        // Output point 3: X(2)
        v_out2 = _mm256_sub_pd(av_s17, av_s21);
        curr_out = out + out_strides[2];
        STR_256_D(curr_out, v_out_stride, v_out2);
        // Output point 19: X(18)
        v_out18 = _mm256_add_pd(av_s17, av_s21);
        curr_out = out + out_strides[18];
        STR_256_D(curr_out, v_out_stride, v_out18);

        av_t7  = _mm256_mul_pd(v256_CRTM_10_4, av_s10);
        av_t8  = _mm256_mul_pd(v256_CRTM_10_5, av_s6);
        av_s22 = _mm256_add_pd(av_t7, av_t8);
        // Output point 5: X(4)
        v_out4 = _mm256_sub_pd(av_s19, av_s22);
        curr_out = out + out_strides[4];
        STR_256_D(curr_out, v_out_stride, v_out4);
        // Output point 17: X(16)
        v_out16 = _mm256_add_pd(av_s19, av_s22);
        curr_out = out + out_strides[16];
        STR_256_D(curr_out, v_out_stride, v_out16);

        /* Shifted DFT */
        __m256d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
                bv_in8, bv_in9;
        __m256d bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
                bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17,
                bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24;
        __m256d bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
                bv_t10, bv_t11, bv_t12, bv_t13, bv_t14;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x256_D(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDRI_2x256_D(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDRI_2x256_D(curr_in, v_in_stride, bv_in4, bv_in5);
        // Input point 14: x(13) & Input point 15: x(14)
        curr_in = in + in_strides[13];
        LDRI_2x256_D(curr_in, v_in_stride, bv_in6, bv_in7);
        // Input point 18: x(17) & Input point 19: x(18)
        curr_in = in + in_strides[17];
        LDRI_2x256_D(curr_in, v_in_stride, bv_in8, bv_in9);

        bv_s1 = _mm256_add_pd(bv_in0, bv_in8);
        bv_s2 = _mm256_sub_pd(bv_in0, bv_in8);
        bv_s3 = _mm256_add_pd(bv_in1, bv_in9);
        bv_s4 = _mm256_sub_pd(bv_in1, bv_in9);
        bv_s5 = _mm256_add_pd(bv_in2, bv_in6);
        bv_s6 = _mm256_sub_pd(bv_in2, bv_in6);
        bv_s7 = _mm256_add_pd(bv_in3, bv_in7);
        bv_s8 = _mm256_sub_pd(bv_in3, bv_in7);

        bv_s9  = _mm256_add_pd(bv_s1, bv_s5);
        bv_s10 = _mm256_sub_pd(bv_s1, bv_s5);
        bv_s11 = _mm256_add_pd(bv_s3, bv_s7);
        bv_s12 = _mm256_sub_pd(bv_s3, bv_s7);

        bv_t1  = _mm256_mul_pd(v256_CRTM_10_1, bv_s11);
        bv_t2  = _mm256_mul_pd(v256_CRTM_10_2, bv_s12);
        bv_t3  = _mm256_mul_pd(v256_CRTM_10_1, bv_s10);
        bv_t4  = _mm256_mul_pd(v256_CRTM_10_2, bv_s9);
        bv_t5  = _mm256_mul_pd(v256_CRTM_10_3, bv_in4);
        bv_t6  = _mm256_mul_pd(v256_CRTM_10_3, bv_in5);
        bv_s23 = _mm256_add_pd(bv_s9, bv_s9);
        // Output point 2: X(1)
        v_out1 = _mm256_add_pd(bv_t5, bv_s23);
        curr_out = out + out_strides[1];
        STR_256_D(curr_out, v_out_stride, v_out1);

        bv_s24 = _mm256_add_pd(bv_s12, bv_s12);
        // Output point 12: X(11)
        v_out11 = NEGATE_256_D(_mm256_add_pd(bv_t6, bv_s24));
        curr_out = out + out_strides[11];
        STR_256_D(curr_out, v_out_stride, v_out11);

        bv_s13 = _mm256_sub_pd(bv_t2, bv_t6);
        bv_s14 = _mm256_sub_pd(bv_t4, bv_t5);

        bv_s15 = _mm256_sub_pd(bv_s13, bv_t1);
        bv_s16 = _mm256_add_pd(bv_s13, bv_t1);
        bv_s17 = _mm256_add_pd(bv_t3, bv_s14);
        bv_s18 = _mm256_sub_pd(bv_t3, bv_s14);

        bv_t7  = _mm256_mul_pd(v256_CRTM_10_5, bv_s2);
        bv_t8  = _mm256_mul_pd(v256_CRTM_10_5, bv_s4);
        bv_t9  = _mm256_mul_pd(v256_CRTM_10_5, bv_s6);
        bv_t10 = _mm256_mul_pd(v256_CRTM_10_5, bv_s8);
        bv_t11 = _mm256_mul_pd(v256_CRTM_10_4, bv_s2);
        bv_t12 = _mm256_mul_pd(v256_CRTM_10_4, bv_s4);
        bv_t13 = _mm256_mul_pd(v256_CRTM_10_4, bv_s6);
        bv_t14 = _mm256_mul_pd(v256_CRTM_10_4, bv_s8);

        bv_s19 = _mm256_add_pd(bv_t7, bv_t13);
        // Output point 4: X(3)
        v_out3 = _mm256_add_pd(bv_s15, bv_s19);
        curr_out = out + out_strides[3];
        STR_256_D(curr_out, v_out_stride, v_out3);
        // Output point 20: X(19)
        v_out19 = _mm256_sub_pd(bv_s15, bv_s19);
        curr_out = out + out_strides[19];
        STR_256_D(curr_out, v_out_stride, v_out19);

        bv_s20 = _mm256_add_pd(bv_t10, bv_t12);
        // Output point 6: X(5)
        v_out5 = _mm256_sub_pd(bv_s17, bv_s20);
        curr_out = out + out_strides[5];
        STR_256_D(curr_out, v_out_stride, v_out5);
        // Output point 18: X(17)
        v_out17 = NEGATE_256_D(_mm256_add_pd(bv_s17, bv_s20));
        curr_out = out + out_strides[17];
        STR_256_D(curr_out, v_out_stride, v_out17);

        bv_s21 = _mm256_sub_pd(bv_t11, bv_t9);
        // Output point 8: X(7)
        v_out7 = _mm256_sub_pd(bv_s21, bv_s16);
        curr_out = out + out_strides[7];
        STR_256_D(curr_out, v_out_stride, v_out7);
        // Output point 16: X(15)
        v_out15 = NEGATE_256_D(_mm256_add_pd(bv_s16, bv_s21));
        curr_out = out + out_strides[15];
        STR_256_D(curr_out, v_out_stride, v_out15);

        bv_s22 = _mm256_sub_pd(bv_t14, bv_t8);
        // Output point 10: X(9)
        v_out9 = _mm256_add_pd(bv_s22, bv_s18);
        curr_out = out + out_strides[9];
        STR_256_D(curr_out, v_out_stride, v_out9);
        // Output point 14: X(13)
        v_out13 = _mm256_sub_pd(bv_s22, bv_s18);
        curr_out = out + out_strides[13];
        STR_256_D(curr_out, v_out_stride, v_out13);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (n & 2)
    {
        /* Standard DFT */
        __m128d av_in0, av_in1, av_in2, av_in3, av_in4, av_in5, av_in6, av_in7,
                av_in8, av_in9;
        __m128d av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8, av_s9,
                av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16, av_s17,
                av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24;
        __m128d av_t1, av_t2, av_t3, av_t4, av_t5, av_t6, av_t7, av_t8, av_t9,
                av_t10, av_t11, av_t12, av_t13, av_t14;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
                v_out15, v_out16, v_out17, v_out18, v_out19;

        curr_in = in;
        curr_out = out;

        __m128d v128_CRTM_10_1 = _mm512_castpd512_pd128(v_CRTM_10_1);
        __m128d v128_CRTM_10_2 = _mm512_castpd512_pd128(v_CRTM_10_2);
        __m128d v128_CRTM_10_3 = _mm512_castpd512_pd128(v_CRTM_10_3);
        __m128d v128_CRTM_10_4 = _mm512_castpd512_pd128(v_CRTM_10_4);
        __m128d v128_CRTM_10_5 = _mm512_castpd512_pd128(v_CRTM_10_5);

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
        // Input point 16: x(15) & Input point 17: x(16)
        curr_in = in + in_strides[15];
        LDRI_2x128_D(curr_in, v_in_stride, av_in7, av_in8);
        // Input point 20: x(19)
        curr_in = in + in_strides[19];
        LDR_128_D(curr_in, v_in_stride, av_in9);

        av_s1  = _mm_add_pd(av_in0, av_in9);
        av_s2  = _mm_sub_pd(av_in0, av_in9);
        av_s3  = _mm_add_pd(av_in1, av_in7);
        av_s4  = _mm_sub_pd(av_in1, av_in7);
        av_s5  = _mm_add_pd(av_in2, av_in8);
        av_s6  = _mm_sub_pd(av_in2, av_in8);
        av_t11 = _mm_mul_pd(v128_CRTM_10_4, av_s6);
        av_s9  = _mm_add_pd(av_in4, av_in6);

        av_t9  = _mm_mul_pd(v128_CRTM_10_4, av_s9);
        av_t10 = _mm_mul_pd(v128_CRTM_10_5, av_s5);
        av_s23 = _mm_sub_pd(av_t9, av_t10);
        av_s10 = _mm_sub_pd(av_in4, av_in6);
        av_t12 = _mm_mul_pd(v128_CRTM_10_5, av_s10);
        av_s24 = _mm_sub_pd(av_t12, av_t11);

        av_s7  = _mm_add_pd(av_in3, av_in5);
        av_s8  = _mm_sub_pd(av_in3, av_in5);
        av_s14 = _mm_sub_pd(av_s4, av_s8);
        av_t14 = _mm_mul_pd(v128_CRTM_10_3, av_s14);
        // Output point 11: X(10)
        v_out10 = _mm_sub_pd(av_s2, av_t14);
        curr_out = out + out_strides[10];
        STR_128_D(curr_out, v_out_stride, v_out10);

        av_s11 = _mm_add_pd(av_s3, av_s7);
        av_t13 = _mm_mul_pd(v128_CRTM_10_3, av_s11);
        // Output point 1: X(0)
        v_out0 = _mm_add_pd(av_s1, av_t13);
        curr_out = out + out_strides[0];
        STR_128_D(curr_out, v_out_stride, v_out0);

        av_t1  = _mm_mul_pd(v128_CRTM_10_2, av_s11);
        av_s16 = _mm_sub_pd(av_s1, av_t1);
        av_s12 = _mm_sub_pd(av_s3, av_s7);
        av_t3  = _mm_mul_pd(v128_CRTM_10_1, av_s12);
        av_s19 = _mm_add_pd(av_s16, av_t3);
        av_s20 = _mm_sub_pd(av_s16, av_t3);
        // Output point 9: X(8)
        v_out8 = _mm_add_pd(av_s20, av_s24);
        curr_out = out + out_strides[8];
        STR_128_D(curr_out, v_out_stride, v_out8);
        // Output point 13: X(12)
        v_out12 = _mm_sub_pd(av_s20, av_s24);
        curr_out = out + out_strides[12];
        STR_128_D(curr_out, v_out_stride, v_out12);

        av_s13 = _mm_add_pd(av_s4, av_s8);
        av_t4  = _mm_mul_pd(v128_CRTM_10_1, av_s13);
        av_t2  = _mm_mul_pd(v128_CRTM_10_2, av_s14);
        av_s15 = _mm_add_pd(av_s2, av_t2);
        av_s17 = _mm_add_pd(av_s15, av_t4);
        av_s18 = _mm_sub_pd(av_s15, av_t4);
        // Output point 7: X(6)
        v_out6 = _mm_add_pd(av_s18, av_s23);
        curr_out = out + out_strides[6];
        STR_128_D(curr_out, v_out_stride, v_out6);
        // Output point 15: X(14)
        v_out14 = _mm_sub_pd(av_s18, av_s23);
        curr_out = out + out_strides[14];
        STR_128_D(curr_out, v_out_stride, v_out14);

        av_t5  = _mm_mul_pd(v128_CRTM_10_4, av_s5);
        av_t6  = _mm_mul_pd(v128_CRTM_10_5, av_s9);
        av_s21 = _mm_add_pd(av_t5, av_t6);
        // Output point 3: X(2)
        v_out2 = _mm_sub_pd(av_s17, av_s21);
        curr_out = out + out_strides[2];
        STR_128_D(curr_out, v_out_stride, v_out2);
        // Output point 19: X(18)
        v_out18 = _mm_add_pd(av_s17, av_s21);
        curr_out = out + out_strides[18];
        STR_128_D(curr_out, v_out_stride, v_out18);

        av_t7  = _mm_mul_pd(v128_CRTM_10_4, av_s10);
        av_t8  = _mm_mul_pd(v128_CRTM_10_5, av_s6);
        av_s22 = _mm_add_pd(av_t7, av_t8);
        // Output point 5: X(4)
        v_out4 = _mm_sub_pd(av_s19, av_s22);
        curr_out = out + out_strides[4];
        STR_128_D(curr_out, v_out_stride, v_out4);
        // Output point 17: X(16)
        v_out16 = _mm_add_pd(av_s19, av_s22);
        curr_out = out + out_strides[16];
        STR_128_D(curr_out, v_out_stride, v_out16);

        /* Shifted DFT */
        __m128d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
                bv_in8, bv_in9;
        __m128d bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
                bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17,
                bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24;
        __m128d bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
                bv_t10, bv_t11, bv_t12, bv_t13, bv_t14;

        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in0, bv_in1);
        // Input point 6: x(5) & Input point 7: x(6)
        curr_in = in + in_strides[5];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in2, bv_in3);
        // Input point 10: x(9) & Input point 11: x(10)
        curr_in = in + in_strides[9];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in4, bv_in5);
        // Input point 14: x(13) & Input point 15: x(14)
        curr_in = in + in_strides[13];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in6, bv_in7);
        // Input point 18: x(17) & Input point 19: x(18)
        curr_in = in + in_strides[17];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in8, bv_in9);

        bv_s1 = _mm_add_pd(bv_in0, bv_in8);
        bv_s2 = _mm_sub_pd(bv_in0, bv_in8);
        bv_s3 = _mm_add_pd(bv_in1, bv_in9);
        bv_s4 = _mm_sub_pd(bv_in1, bv_in9);
        bv_s5 = _mm_add_pd(bv_in2, bv_in6);
        bv_s6 = _mm_sub_pd(bv_in2, bv_in6);
        bv_s7 = _mm_add_pd(bv_in3, bv_in7);
        bv_s8 = _mm_sub_pd(bv_in3, bv_in7);

        bv_s9  = _mm_add_pd(bv_s1, bv_s5);
        bv_s10 = _mm_sub_pd(bv_s1, bv_s5);
        bv_s11 = _mm_add_pd(bv_s3, bv_s7);
        bv_s12 = _mm_sub_pd(bv_s3, bv_s7);

        bv_t1  = _mm_mul_pd(v128_CRTM_10_1, bv_s11);
        bv_t2  = _mm_mul_pd(v128_CRTM_10_2, bv_s12);
        bv_t3  = _mm_mul_pd(v128_CRTM_10_1, bv_s10);
        bv_t4  = _mm_mul_pd(v128_CRTM_10_2, bv_s9);
        bv_t5  = _mm_mul_pd(v128_CRTM_10_3, bv_in4);
        bv_t6  = _mm_mul_pd(v128_CRTM_10_3, bv_in5);
        bv_s23 = _mm_add_pd(bv_s9, bv_s9);
        // Output point 2: X(1)
        v_out1 = _mm_add_pd(bv_t5, bv_s23);
        curr_out = out + out_strides[1];
        STR_128_D(curr_out, v_out_stride, v_out1);

        bv_s24 = _mm_add_pd(bv_s12, bv_s12);
        // Output point 12: X(11)
        v_out11 = NEGATE_128_D(_mm_add_pd(bv_t6, bv_s24));
        curr_out = out + out_strides[11];
        STR_128_D(curr_out, v_out_stride, v_out11);

        bv_s13 = _mm_sub_pd(bv_t2, bv_t6);
        bv_s14 = _mm_sub_pd(bv_t4, bv_t5);

        bv_s15 = _mm_sub_pd(bv_s13, bv_t1);
        bv_s16 = _mm_add_pd(bv_s13, bv_t1);
        bv_s17 = _mm_add_pd(bv_t3, bv_s14);
        bv_s18 = _mm_sub_pd(bv_t3, bv_s14);

        bv_t7  = _mm_mul_pd(v128_CRTM_10_5, bv_s2);
        bv_t8  = _mm_mul_pd(v128_CRTM_10_5, bv_s4);
        bv_t9  = _mm_mul_pd(v128_CRTM_10_5, bv_s6);
        bv_t10 = _mm_mul_pd(v128_CRTM_10_5, bv_s8);
        bv_t11 = _mm_mul_pd(v128_CRTM_10_4, bv_s2);
        bv_t12 = _mm_mul_pd(v128_CRTM_10_4, bv_s4);
        bv_t13 = _mm_mul_pd(v128_CRTM_10_4, bv_s6);
        bv_t14 = _mm_mul_pd(v128_CRTM_10_4, bv_s8);

        bv_s19 = _mm_add_pd(bv_t7, bv_t13);
        // Output point 4: X(3)
        v_out3 = _mm_add_pd(bv_s15, bv_s19);
        curr_out = out + out_strides[3];
        STR_128_D(curr_out, v_out_stride, v_out3);
        // Output point 20: X(19)
        v_out19 = _mm_sub_pd(bv_s15, bv_s19);
        curr_out = out + out_strides[19];
        STR_128_D(curr_out, v_out_stride, v_out19);

        bv_s20 = _mm_add_pd(bv_t10, bv_t12);
        // Output point 6: X(5)
        v_out5 = _mm_sub_pd(bv_s17, bv_s20);
        curr_out = out + out_strides[5];
        STR_128_D(curr_out, v_out_stride, v_out5);
        // Output point 18: X(17)
        v_out17 = NEGATE_128_D(_mm_add_pd(bv_s17, bv_s20));
        curr_out = out + out_strides[17];
        STR_128_D(curr_out, v_out_stride, v_out17);

        bv_s21 = _mm_sub_pd(bv_t11, bv_t9);
        // Output point 8: X(7)
        v_out7 = _mm_sub_pd(bv_s21, bv_s16);
        curr_out = out + out_strides[7];
        STR_128_D(curr_out, v_out_stride, v_out7);
        // Output point 16: X(15)
        v_out15 = NEGATE_128_D(_mm_add_pd(bv_s16, bv_s21));
        curr_out = out + out_strides[15];
        STR_128_D(curr_out, v_out_stride, v_out15);

        bv_s22 = _mm_sub_pd(bv_t14, bv_t8);
        // Output point 10: X(9)
        v_out9 = _mm_add_pd(bv_s22, bv_s18);
        curr_out = out + out_strides[9];
        STR_128_D(curr_out, v_out_stride, v_out9);
        // Output point 14: X(13)
        v_out13 = _mm_sub_pd(bv_s22, bv_s18);
        curr_out = out + out_strides[13];
        STR_128_D(curr_out, v_out_stride, v_out13);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        /* Standard DFT */
        DOUBLE a_in0, a_in1, a_in2, a_in3, a_in4, a_in5, a_in6, a_in7, a_in8,
               a_in9;
        DOUBLE a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7, a_s8, a_s9,
               a_s10, a_s11, a_s12, a_s13, a_s14, a_s15, a_s16, a_s17, a_s18,
               a_s19, a_s20, a_s21, a_s22, a_s23;
        DOUBLE a_t0, a_t1, a_t2, a_t3, a_t4, a_t5, a_t6, a_t7, a_t8, a_t9,
               a_t10, a_t11, a_t12, a_t13;

        //  Input point 1: x(0)
        a_in0 = *in;
        // Input point 4: x(3)
        a_in1 = in[in_strides[3]];
        // Input point 5: x(4)
        a_in2 = in[in_strides[4]];
        // Input point 8: x(7)
        a_in3 = in[in_strides[7]];
        // Input point 9: x(8)
        a_in4 = in[in_strides[8]];
        // Input point 12: x(11)
        a_in5 = in[in_strides[11]];
        // Input point 13: x(12)
        a_in6 = in[in_strides[12]];
        // Input point 16: x(15)
        a_in7 = in[in_strides[15]];
        // Input point 17: x(16)
        a_in8 = in[in_strides[16]];
        // Input point 20: x(19)
        a_in9 = in[in_strides[19]];

        a_s0 = a_in0 + a_in9;
        a_s1 = a_in0 - a_in9;
        a_s2 = a_in1 + a_in7;
        a_s3 = a_in1 - a_in7;
        a_s4 = a_in2 + a_in8;
        a_s5 = a_in2 - a_in8;
        a_t0 = CRTM_10_4 * a_s5;
        a_s8 = a_in4 + a_in6;

        a_t1 = CRTM_10_4 * a_s8;
        a_t2 = CRTM_10_5 * a_s4;
        a_s22 = a_t1 - a_t2;
        a_s9 = a_in4 - a_in6;
        a_t3 = CRTM_10_5 * a_s9;
        a_s23 = a_t3 - a_t0;

        a_s6 = a_in3 + a_in5;
        a_s7 = a_in3 - a_in5;
        a_s13 = a_s3 - a_s7;
        a_t4 = CRTM_10_3 * a_s13;
        // Output point 11: X(10)
        out[out_strides[10]] = a_s1 - a_t4;

        a_s10 = a_s2 + a_s6;
        a_t5 = CRTM_10_3 * a_s10;
        // Output point 1: X(0)
        *out = a_s0 + a_t5;

        a_t6 = CRTM_10_2 * a_s10;
        a_s15 = a_s0 - a_t6;
        a_s11 = a_s2 - a_s6;
        a_t7 = CRTM_10_1 * a_s11;
        a_s18 = a_s15 + a_t7;
        a_s19 = a_s15 - a_t7;
        // Output point 9: X(8)
        out[out_strides[8]] = a_s19 + a_s23;
        // Output point 13: X(12)
        out[out_strides[12]] = a_s19 - a_s23;

        a_s12 = a_s3 + a_s7;
        a_t8 = CRTM_10_1 * a_s12;
        a_t9 = CRTM_10_2 * a_s13;
        a_s14 = a_s1 + a_t9;
        a_s16 = a_s14 + a_t8;
        a_s17 = a_s14 - a_t8;
        // Output point 7: X(6)
        out[out_strides[6]] = a_s17 + a_s22;
        // Output point 15: X(14)
        out[out_strides[14]] = a_s17 - a_s22;

        a_t10 = CRTM_10_4 * a_s4;
        a_t11 = CRTM_10_5 * a_s8;
        a_s20 = a_t10 + a_t11;
        // Output point 3: X(2)
        out[out_strides[2]] = a_s16 - a_s20;
        // Output point 19: X(18)
        out[out_strides[18]] = a_s16 + a_s20;

        a_t12 = CRTM_10_4 * a_s9;
        a_t13 = CRTM_10_5 * a_s5;
        a_s21 = a_t12 + a_t13;
        // Output point 5: X(4)
        out[out_strides[4]] = a_s18 - a_s21;
        // Output point 17: X(16)
        out[out_strides[16]] = a_s18 + a_s21;

        /* Shifted DFT */
        DOUBLE b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7, b_in8,
               b_in9;
        DOUBLE b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9,
               b_s10, b_s11, b_s12, b_s13, b_s14, b_s15, b_s16, b_s17, b_s18,
               b_s19, b_s20, b_s21, b_s22, b_s23;
        DOUBLE b_t0, b_t1, b_t2, b_t3, b_t4, b_t5, b_t6, b_t7, b_t8, b_t9,
               b_t10, b_t11, b_t12, b_t13;

        // Input point 2: x(1)
        b_in0 = in[in_strides[1]];
        // Input point 3: x(2)
        b_in1 = in[in_strides[2]];
        // Input point 6: x(5)
        b_in2 = in[in_strides[5]];
        // Input point 7: x(6)
        b_in3 = in[in_strides[6]];
        // Input point 10: x(9)
        b_in4 = in[in_strides[9]];
        // Input point 11: x(10)
        b_in5 = in[in_strides[10]];
        // Input point 14: x(13)
        b_in6 = in[in_strides[13]];
        // Input point 15: x(14)
        b_in7 = in[in_strides[14]];
        // Input point 18: x(17)
        b_in8 = in[in_strides[17]];
        // Input point 19: x(18)
        b_in9 = in[in_strides[18]];

        b_s0 = b_in0 + b_in8;
        b_s1 = b_in0 - b_in8;
        b_s2 = b_in1 + b_in9;
        b_s3 = b_in1 - b_in9;
        b_s4 = b_in2 + b_in6;
        b_s5 = b_in2 - b_in6;
        b_s6 = b_in3 + b_in7;
        b_s7 = b_in3 - b_in7;

        b_s8 = b_s0 + b_s4;
        b_s9 = b_s0 - b_s4;
        b_s10 = b_s2 + b_s6;
        b_s11 = b_s2 - b_s6;

        b_t0 = CRTM_10_1 * b_s10;
        b_t1 = CRTM_10_2 * b_s11;
        b_t2 = CRTM_10_1 * b_s9;
        b_t3 = CRTM_10_2 * b_s8;
        b_t4 = CRTM_10_3 * b_in4;
        b_t5 = CRTM_10_3 * b_in5;
        b_s22 = b_s8 + b_s8;
        // Output point 2: X(1)
        out[out_strides[1]] = b_t4 + b_s22;
        b_s23 = b_s11 + b_s11;
        // Output point 12: X(11)
        out[out_strides[11]] = -(b_t5 + b_s23);

        b_s12 = b_t1 - b_t5;
        b_s13 = b_t3 - b_t4;

        b_s14 = b_s12 - b_t0;
        b_s15 = b_s12 + b_t0;
        b_s16 = b_t2 + b_s13;
        b_s17 = b_t2 - b_s13;

        b_t6 = CRTM_10_5 * b_s1;
        b_t7 = CRTM_10_5 * b_s3;
        b_t8 = CRTM_10_5 * b_s5;
        b_t9 = CRTM_10_5 * b_s7;
        b_t10 = CRTM_10_4 * b_s1;
        b_t11 = CRTM_10_4 * b_s3;
        b_t12 = CRTM_10_4 * b_s5;
        b_t13 = CRTM_10_4 * b_s7;

        b_s18 = b_t6 + b_t12;
        // Output point 4: X(3)
        out[out_strides[3]] = b_s14 + b_s18;
        // Output point 20: X(19)
        out[out_strides[19]] = b_s14 - b_s18;

        b_s19 = b_t9 + b_t11;
        // Output point 6: X(5)
        out[out_strides[5]] = b_s16 - b_s19;
        // Output point 18: X(17)
        out[out_strides[17]] = -(b_s16 + b_s19);

        b_s20 = b_t10 - b_t8;
        // Output point 8: X(7)
        out[out_strides[7]] = b_s20 - b_s15;
        // Output point 16: X(15)
        out[out_strides[15]] = -(b_s15 + b_s20);

        b_s21 = b_t13 - b_t7;
        // Output point 10: X(9)
        out[out_strides[9]] = b_s21 + b_s17;
        // Output point 14: X(13)
        out[out_strides[13]] = b_s21 - b_s17;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

kfft_ register_kernel_r2hcf_rfft10avx512(UINT8 precision, UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hcf_rfft10avx512_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft10avx512_fp64_fwd;
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
            return r2hcf_rfft10avx512_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft10avx512_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
