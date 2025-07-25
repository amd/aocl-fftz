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

/** @file rfft14avx512.c
 *
 *  @brief Radix-14 r2hc Real-FFT kernel with AVX-512 operations using x86 SIMD
 *  intrinsics
 *
 *  This file contains the DIT radix-14 real-to-halfcomplex implementations
 *  using AVX512 SIMD operations for single-precision and double-precision
 *  inputs.
 *
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/r2hc_simd_avx512.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                {{{0, 36, 62, 352, 204, 84},
                                                  {0, 38, 61, 352, 252, 84}},
                                                 {{0, 36, 62, 176, 12,  84},
                                                  {0, 38, 61, 176, 12,  84}}};

ops_cycles_t get_ops_cnt_r2hc_rfft14avx512(UINT8 precision, UINT8 direction)
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

static VOID r2hc_rfft14avx512_fp32_fwd(VOID *in_real, VOID *in_imag,
                                       VOID *out_real, VOID *out_imag, INTP n,
                                       aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const FLOAT CRTM_14_1 = 0.900968867902419126236102319507445051165919162f;
    const FLOAT CRTM_14_2 = 0.433883739117558120475768332848358754609990728f;
    const FLOAT CRTM_14_3 = 0.623489801858733530525004884004239810632274731f;
    const FLOAT CRTM_14_4 = 0.781831482468029808708444526674057750232334519f;
    const FLOAT CRTM_14_5 = 0.222520933956314404288902564496794759466355569f;
    const FLOAT CRTM_14_6 = 0.974927912181823607018131682993931217232785801f;

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

    __m512 v_CRTM_14_1 = _mm512_set1_ps(CRTM_14_1);
    __m512 v_CRTM_14_2 = _mm512_set1_ps(CRTM_14_2);
    __m512 v_CRTM_14_3 = _mm512_set1_ps(CRTM_14_3);
    __m512 v_CRTM_14_4 = _mm512_set1_ps(CRTM_14_4);
    __m512 v_CRTM_14_5 = _mm512_set1_ps(CRTM_14_5);
    __m512 v_CRTM_14_6 = _mm512_set1_ps(CRTM_14_6);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m512 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9, v_in10, v_in11, v_in12, v_in13;
        __m512 v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9, v_s10,
               v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18, v_s19,
               v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27, v_s28,
               v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36, v_s37,
               v_s38, v_s39, v_s40, v_s41, v_s42, v_s43, v_s44, v_s45, v_s46,
               v_s47, v_s48;
        __m512 v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9, v_t10,
               v_t11, v_t12, v_t13, v_t14, v_t15, v_t16, v_t17, v_t18, v_t19,
               v_t20, v_t21, v_t22, v_t23, v_t24, v_t25, v_t26, v_t27, v_t28,
               v_t29, v_t30, v_t31, v_t32, v_t33, v_t34, v_t35, v_t36;
        __m512 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13;

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
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDR_512_S(curr_in, v_in_stride, v_in12);
        // Input point 14: x(13)
        curr_in = in + in_strides[13];
        LDR_512_S(curr_in, v_in_stride, v_in13);

        v_s1 = _mm512_sub_ps(v_in0, v_in7);
        v_s2 = _mm512_add_ps(v_in0, v_in7);
        v_s3 = _mm512_sub_ps(v_in13, v_in1);
        v_s4 = _mm512_add_ps(v_in13, v_in1);
        v_s5 = _mm512_sub_ps(v_in12, v_in2);
        v_s6 = _mm512_add_ps(v_in12, v_in2);
        v_s7 = _mm512_sub_ps(v_in11, v_in3);
        v_s8 = _mm512_add_ps(v_in11, v_in3);
        v_s9 = _mm512_sub_ps(v_in10, v_in4);
        v_s10 = _mm512_add_ps(v_in10, v_in4);
        v_s11 = _mm512_sub_ps(v_in9, v_in5);
        v_s12 = _mm512_add_ps(v_in9, v_in5);
        v_s13 = _mm512_sub_ps(v_in8, v_in6);
        v_s14 = _mm512_add_ps(v_in8, v_in6);

        v_s15 = _mm512_add_ps(v_s4, v_s14);
        v_s16 = _mm512_add_ps(v_s6, v_s12);
        v_s17 = _mm512_add_ps(v_s8, v_s10);

        v_s18 = _mm512_sub_ps(v_s14, v_s4);
        v_s19 = _mm512_sub_ps(v_s6, v_s12);
        v_s20 = _mm512_sub_ps(v_s10, v_s8);
        v_s27 = _mm512_add_ps(v_s2, v_s15);
        v_s28 = _mm512_add_ps(v_s16, v_s17);
        v_s29 = _mm512_add_ps(v_s1, v_s18);
        v_s30 = _mm512_add_ps(v_s19, v_s20);
        // Output pt 1: X(0)
        v_out0 = _mm512_add_ps(v_s27, v_s28);
        STR_512_S(curr_out, v_out_stride, v_out0);
        // Output pt 14: X(13)
        v_out13 = _mm512_add_ps(v_s29, v_s30);
        curr_out = out + out_strides[13];
        STR_512_S(curr_out, v_out_stride, v_out13);

        v_t1 = _mm512_mul_ps(v_CRTM_14_1, v_s18);
        v_t2 = _mm512_mul_ps(v_CRTM_14_3, v_s19);
        v_t3 = _mm512_mul_ps(v_CRTM_14_5, v_s20);
        v_s31 = _mm512_sub_ps(v_s1, v_t1);
        v_s32 = _mm512_sub_ps(v_t2, v_t3);
        // Output point 2: X(1)
        v_out1 = _mm512_add_ps(v_s31, v_s32);

        v_s21 = _mm512_add_ps(v_s3, v_s13);
        v_s22 = _mm512_add_ps(v_s5, v_s11);
        v_s23 = _mm512_add_ps(v_s7, v_s9);

        v_t4 = _mm512_mul_ps(v_CRTM_14_2, v_s21);
        v_t5 = _mm512_mul_ps(v_CRTM_14_4, v_s22);
        v_t6 = _mm512_mul_ps(v_CRTM_14_6, v_s23);
        v_s33 = _mm512_add_ps(v_t4, v_t5);
        // Output point 3: X(2)
        v_out2 = _mm512_add_ps(v_s33, v_t6);
        curr_out = out + out_strides[1];
        STRI_2x512_S(curr_out, v_out_stride, v_out1, v_out2);

        v_t7 = _mm512_mul_ps(v_CRTM_14_1, v_s17);
        v_t8 = _mm512_mul_ps(v_CRTM_14_3, v_s15);
        v_t9 = _mm512_mul_ps(v_CRTM_14_5, v_s16);
        v_s34 = _mm512_sub_ps(v_s2, v_t7);
        v_s35 = _mm512_sub_ps(v_t8, v_t9);
        // Output point 4: X(3)
        v_out3 = _mm512_add_ps(v_s34, v_s35);

        v_s24 = _mm512_sub_ps(v_s3, v_s13);
        v_s25 = _mm512_sub_ps(v_s5, v_s11);
        v_s26 = _mm512_sub_ps(v_s7, v_s9);

        v_t10 = _mm512_mul_ps(v_CRTM_14_2, v_s26);
        v_t11 = _mm512_mul_ps(v_CRTM_14_4, v_s24);
        v_t12 = _mm512_mul_ps(v_CRTM_14_6, v_s25);
        v_s36 = _mm512_add_ps(v_t10, v_t11);
        // Output point 5: X(4)
        v_out4 = _mm512_add_ps(v_s36, v_t12);
        curr_out = out + out_strides[3];
        STRI_2x512_S(curr_out, v_out_stride, v_out3, v_out4);

        v_t13 = _mm512_mul_ps(v_CRTM_14_1, v_s19);
        v_t14 = _mm512_mul_ps(v_CRTM_14_3, v_s20);
        v_t15 = _mm512_mul_ps(v_CRTM_14_5, v_s18);
        v_s37 = _mm512_sub_ps(v_s1, v_t13);
        v_s38 = _mm512_sub_ps(v_t14, v_t15);
        // Output point 6: X(5)
        v_out5 = _mm512_add_ps(v_s37, v_s38);

        v_t16 = _mm512_mul_ps(v_CRTM_14_2, v_s22);
        v_t17 = _mm512_mul_ps(v_CRTM_14_4, v_s23);
        v_t18 = _mm512_mul_ps(v_CRTM_14_6, v_s21);
        v_s39 = _mm512_sub_ps(v_t16, v_t17);
        // Output point 7: X(6)
        v_out6 = _mm512_add_ps(v_s39, v_t18);
        curr_out = out + out_strides[5];
        STRI_2x512_S(curr_out, v_out_stride, v_out5, v_out6);

        v_t19 = _mm512_mul_ps(v_CRTM_14_1, v_s16);
        v_t20 = _mm512_mul_ps(v_CRTM_14_3, v_s17);
        v_t21 = _mm512_mul_ps(v_CRTM_14_5, v_s15);
        v_s40 = _mm512_sub_ps(v_s2, v_t19);
        v_s41 = _mm512_sub_ps(v_t20, v_t21);
        // Output point 8: X(7)
        v_out7 = _mm512_add_ps(v_s40, v_s41);

        v_t22 = _mm512_mul_ps(v_CRTM_14_2, v_s25);
        v_t23 = _mm512_mul_ps(v_CRTM_14_4, v_s26);
        v_t24 = _mm512_mul_ps(v_CRTM_14_6, v_s24);
        v_s42 = _mm512_sub_ps(v_t24, v_t22);
        // Output point 9: X(8)
        v_out8 = _mm512_sub_ps(v_s42, v_t23);
        curr_out = out + out_strides[7];
        STRI_2x512_S(curr_out, v_out_stride, v_out7, v_out8);

        v_t25 = _mm512_mul_ps(v_CRTM_14_1, v_s20);
        v_t26 = _mm512_mul_ps(v_CRTM_14_3, v_s18);
        v_t27 = _mm512_mul_ps(v_CRTM_14_5, v_s19);
        v_s43 = _mm512_sub_ps(v_s1, v_t25);
        v_s44 = _mm512_sub_ps(v_t26, v_t27);
        // Output point 10: X(9)
        v_out9 = _mm512_add_ps(v_s43, v_s44);

        v_t28 = _mm512_mul_ps(v_CRTM_14_2, v_s23);
        v_t29 = _mm512_mul_ps(v_CRTM_14_4, v_s21);
        v_t30 = _mm512_mul_ps(v_CRTM_14_6, v_s22);
        v_s45 = _mm512_add_ps(v_t28, v_t29);
        // Output point 11: X(10)
        v_out10 = _mm512_sub_ps(v_s45, v_t30);
        curr_out = out + out_strides[9];
        STRI_2x512_S(curr_out, v_out_stride, v_out9, v_out10);

        v_t31 = _mm512_mul_ps(v_CRTM_14_1, v_s15);
        v_t32 = _mm512_mul_ps(v_CRTM_14_3, v_s16);
        v_t33 = _mm512_mul_ps(v_CRTM_14_5, v_s17);
        v_s46 = _mm512_sub_ps(v_s2, v_t31);
        v_s47 = _mm512_sub_ps(v_t32, v_t33);
        // Output point 12: X(11)
        v_out11 = _mm512_add_ps(v_s46, v_s47);

        v_t34 = _mm512_mul_ps(v_CRTM_14_2, v_s24);
        v_t35 = _mm512_mul_ps(v_CRTM_14_4, v_s25);
        v_t36 = _mm512_mul_ps(v_CRTM_14_6, v_s26);
        v_s48 = _mm512_sub_ps(v_t34, v_t35);
        // Output point 13: X(12)
        v_out12 = _mm512_add_ps(v_s48, v_t36);
        curr_out = out + out_strides[11];
        STRI_2x512_S(curr_out, v_out_stride, v_out11, v_out12);

        in = in + (v_in_stride << 4);
        out = out + (v_out_stride << 4);
    }
    // tail cases
    if (n & 8)
    {
        __m256 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9, v_in10, v_in11, v_in12, v_in13;
        __m256 v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9, v_s10,
               v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18, v_s19,
               v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27, v_s28,
               v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36, v_s37,
               v_s38, v_s39, v_s40, v_s41, v_s42, v_s43, v_s44, v_s45, v_s46,
               v_s47, v_s48;
        __m256 v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9, v_t10,
               v_t11, v_t12, v_t13, v_t14, v_t15, v_t16, v_t17, v_t18, v_t19,
               v_t20, v_t21, v_t22, v_t23, v_t24, v_t25, v_t26, v_t27, v_t28,
               v_t29, v_t30, v_t31, v_t32, v_t33, v_t34, v_t35, v_t36;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13;

        curr_in = in;
        curr_out = out;

        __m256 v256_CRTM_14_1 = _mm512_castps512_ps256(v_CRTM_14_1);
        __m256 v256_CRTM_14_2 = _mm512_castps512_ps256(v_CRTM_14_2);
        __m256 v256_CRTM_14_3 = _mm512_castps512_ps256(v_CRTM_14_3);
        __m256 v256_CRTM_14_4 = _mm512_castps512_ps256(v_CRTM_14_4);
        __m256 v256_CRTM_14_5 = _mm512_castps512_ps256(v_CRTM_14_5);
        __m256 v256_CRTM_14_6 = _mm512_castps512_ps256(v_CRTM_14_6);

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
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDR_256_S(curr_in, v_in_stride, v_in12);
        // Input point 14: x(13)
        curr_in = in + in_strides[13];
        LDR_256_S(curr_in, v_in_stride, v_in13);

        v_s1 = _mm256_sub_ps(v_in0, v_in7);
        v_s2 = _mm256_add_ps(v_in0, v_in7);
        v_s3 = _mm256_sub_ps(v_in13, v_in1);
        v_s4 = _mm256_add_ps(v_in13, v_in1);
        v_s5 = _mm256_sub_ps(v_in12, v_in2);
        v_s6 = _mm256_add_ps(v_in12, v_in2);
        v_s7 = _mm256_sub_ps(v_in11, v_in3);
        v_s8 = _mm256_add_ps(v_in11, v_in3);
        v_s9 = _mm256_sub_ps(v_in10, v_in4);
        v_s10 = _mm256_add_ps(v_in10, v_in4);
        v_s11 = _mm256_sub_ps(v_in9, v_in5);
        v_s12 = _mm256_add_ps(v_in9, v_in5);
        v_s13 = _mm256_sub_ps(v_in8, v_in6);
        v_s14 = _mm256_add_ps(v_in8, v_in6);

        v_s15 = _mm256_add_ps(v_s4, v_s14);
        v_s16 = _mm256_add_ps(v_s6, v_s12);
        v_s17 = _mm256_add_ps(v_s8, v_s10);

        v_s18 = _mm256_sub_ps(v_s14, v_s4);
        v_s19 = _mm256_sub_ps(v_s6, v_s12);
        v_s20 = _mm256_sub_ps(v_s10, v_s8);
        v_s27 = _mm256_add_ps(v_s2, v_s15);
        v_s28 = _mm256_add_ps(v_s16, v_s17);
        v_s29 = _mm256_add_ps(v_s1, v_s18);
        v_s30 = _mm256_add_ps(v_s19, v_s20);
        // Output pt 1: X(0)
        v_out0 = _mm256_add_ps(v_s27, v_s28);
        STR_256_S(curr_out, v_out_stride, v_out0);
        // Output pt 14: X(13)
        v_out13 = _mm256_add_ps(v_s29, v_s30);
        curr_out = out + out_strides[13];
        STR_256_S(curr_out, v_out_stride, v_out13);

        v_t1 = _mm256_mul_ps(v256_CRTM_14_1, v_s18);
        v_t2 = _mm256_mul_ps(v256_CRTM_14_3, v_s19);
        v_t3 = _mm256_mul_ps(v256_CRTM_14_5, v_s20);
        v_s31 = _mm256_sub_ps(v_s1, v_t1);
        v_s32 = _mm256_sub_ps(v_t2, v_t3);
        // Output point 2: X(1)
        v_out1 = _mm256_add_ps(v_s31, v_s32);

        v_s21 = _mm256_add_ps(v_s3, v_s13);
        v_s22 = _mm256_add_ps(v_s5, v_s11);
        v_s23 = _mm256_add_ps(v_s7, v_s9);

        v_t4 = _mm256_mul_ps(v256_CRTM_14_2, v_s21);
        v_t5 = _mm256_mul_ps(v256_CRTM_14_4, v_s22);
        v_t6 = _mm256_mul_ps(v256_CRTM_14_6, v_s23);
        v_s33 = _mm256_add_ps(v_t4, v_t5);
        // Output point 3: X(2)
        v_out2 = _mm256_add_ps(v_s33, v_t6);
        curr_out = out + out_strides[1];
        STRI_2x256_S(curr_out, v_out_stride, v_out1, v_out2);

        v_t7 = _mm256_mul_ps(v256_CRTM_14_1, v_s17);
        v_t8 = _mm256_mul_ps(v256_CRTM_14_3, v_s15);
        v_t9 = _mm256_mul_ps(v256_CRTM_14_5, v_s16);
        v_s34 = _mm256_sub_ps(v_s2, v_t7);
        v_s35 = _mm256_sub_ps(v_t8, v_t9);
        // Output point 4: X(3)
        v_out3 = _mm256_add_ps(v_s34, v_s35);

        v_s24 = _mm256_sub_ps(v_s3, v_s13);
        v_s25 = _mm256_sub_ps(v_s5, v_s11);
        v_s26 = _mm256_sub_ps(v_s7, v_s9);

        v_t10 = _mm256_mul_ps(v256_CRTM_14_2, v_s26);
        v_t11 = _mm256_mul_ps(v256_CRTM_14_4, v_s24);
        v_t12 = _mm256_mul_ps(v256_CRTM_14_6, v_s25);
        v_s36 = _mm256_add_ps(v_t10, v_t11);
        // Output point 5: X(4)
        v_out4 = _mm256_add_ps(v_s36, v_t12);
        curr_out = out + out_strides[3];
        STRI_2x256_S(curr_out, v_out_stride, v_out3, v_out4);

        v_t13 = _mm256_mul_ps(v256_CRTM_14_1, v_s19);
        v_t14 = _mm256_mul_ps(v256_CRTM_14_3, v_s20);
        v_t15 = _mm256_mul_ps(v256_CRTM_14_5, v_s18);
        v_s37 = _mm256_sub_ps(v_s1, v_t13);
        v_s38 = _mm256_sub_ps(v_t14, v_t15);
        // Output point 6: X(5)
        v_out5 = _mm256_add_ps(v_s37, v_s38);

        v_t16 = _mm256_mul_ps(v256_CRTM_14_2, v_s22);
        v_t17 = _mm256_mul_ps(v256_CRTM_14_4, v_s23);
        v_t18 = _mm256_mul_ps(v256_CRTM_14_6, v_s21);
        v_s39 = _mm256_sub_ps(v_t16, v_t17);
        // Output point 7: X(6)
        v_out6 = _mm256_add_ps(v_s39, v_t18);
        curr_out = out + out_strides[5];
        STRI_2x256_S(curr_out, v_out_stride, v_out5, v_out6);

        v_t19 = _mm256_mul_ps(v256_CRTM_14_1, v_s16);
        v_t20 = _mm256_mul_ps(v256_CRTM_14_3, v_s17);
        v_t21 = _mm256_mul_ps(v256_CRTM_14_5, v_s15);
        v_s40 = _mm256_sub_ps(v_s2, v_t19);
        v_s41 = _mm256_sub_ps(v_t20, v_t21);
        // Output point 8: X(7)
        v_out7 = _mm256_add_ps(v_s40, v_s41);

        v_t22 = _mm256_mul_ps(v256_CRTM_14_2, v_s25);
        v_t23 = _mm256_mul_ps(v256_CRTM_14_4, v_s26);
        v_t24 = _mm256_mul_ps(v256_CRTM_14_6, v_s24);
        v_s42 = _mm256_sub_ps(v_t24, v_t22);
        // Output point 9: X(8)
        v_out8 = _mm256_sub_ps(v_s42, v_t23);
        curr_out = out + out_strides[7];
        STRI_2x256_S(curr_out, v_out_stride, v_out7, v_out8);

        v_t25 = _mm256_mul_ps(v256_CRTM_14_1, v_s20);
        v_t26 = _mm256_mul_ps(v256_CRTM_14_3, v_s18);
        v_t27 = _mm256_mul_ps(v256_CRTM_14_5, v_s19);
        v_s43 = _mm256_sub_ps(v_s1, v_t25);
        v_s44 = _mm256_sub_ps(v_t26, v_t27);
        // Output point 10: X(9)
        v_out9 = _mm256_add_ps(v_s43, v_s44);

        v_t28 = _mm256_mul_ps(v256_CRTM_14_2, v_s23);
        v_t29 = _mm256_mul_ps(v256_CRTM_14_4, v_s21);
        v_t30 = _mm256_mul_ps(v256_CRTM_14_6, v_s22);
        v_s45 = _mm256_add_ps(v_t28, v_t29);
        // Output point 11: X(10)
        v_out10 = _mm256_sub_ps(v_s45, v_t30);
        curr_out = out + out_strides[9];
        STRI_2x256_S(curr_out, v_out_stride, v_out9, v_out10);

        v_t31 = _mm256_mul_ps(v256_CRTM_14_1, v_s15);
        v_t32 = _mm256_mul_ps(v256_CRTM_14_3, v_s16);
        v_t33 = _mm256_mul_ps(v256_CRTM_14_5, v_s17);
        v_s46 = _mm256_sub_ps(v_s2, v_t31);
        v_s47 = _mm256_sub_ps(v_t32, v_t33);
        // Output point 12: X(11)
        v_out11 = _mm256_add_ps(v_s46, v_s47);

        v_t34 = _mm256_mul_ps(v256_CRTM_14_2, v_s24);
        v_t35 = _mm256_mul_ps(v256_CRTM_14_4, v_s25);
        v_t36 = _mm256_mul_ps(v256_CRTM_14_6, v_s26);
        v_s48 = _mm256_sub_ps(v_t34, v_t35);
        // Output point 13: X(12)
        v_out12 = _mm256_add_ps(v_s48, v_t36);
        curr_out = out + out_strides[11];
        STRI_2x256_S(curr_out, v_out_stride, v_out11, v_out12);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (n & 4)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9, v_in10, v_in11, v_in12, v_in13;
        __m128 v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9, v_s10,
               v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18, v_s19,
               v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27, v_s28,
               v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36, v_s37,
               v_s38, v_s39, v_s40, v_s41, v_s42, v_s43, v_s44, v_s45, v_s46,
               v_s47, v_s48;
        __m128 v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9, v_t10,
               v_t11, v_t12, v_t13, v_t14, v_t15, v_t16, v_t17, v_t18, v_t19,
               v_t20, v_t21, v_t22, v_t23, v_t24, v_t25, v_t26, v_t27, v_t28,
               v_t29, v_t30, v_t31, v_t32, v_t33, v_t34, v_t35, v_t36;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_14_1 = _mm512_castps512_ps128(v_CRTM_14_1);
        __m128 v128_CRTM_14_2 = _mm512_castps512_ps128(v_CRTM_14_2);
        __m128 v128_CRTM_14_3 = _mm512_castps512_ps128(v_CRTM_14_3);
        __m128 v128_CRTM_14_4 = _mm512_castps512_ps128(v_CRTM_14_4);
        __m128 v128_CRTM_14_5 = _mm512_castps512_ps128(v_CRTM_14_5);
        __m128 v128_CRTM_14_6 = _mm512_castps512_ps128(v_CRTM_14_6);

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
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDR_128_S(curr_in, v_in_stride, v_in12);
        // Input point 14: x(13)
        curr_in = in + in_strides[13];
        LDR_128_S(curr_in, v_in_stride, v_in13);

        v_s1 = _mm_sub_ps(v_in0, v_in7);
        v_s2 = _mm_add_ps(v_in0, v_in7);
        v_s3 = _mm_sub_ps(v_in13, v_in1);
        v_s4 = _mm_add_ps(v_in13, v_in1);
        v_s5 = _mm_sub_ps(v_in12, v_in2);
        v_s6 = _mm_add_ps(v_in12, v_in2);
        v_s7 = _mm_sub_ps(v_in11, v_in3);
        v_s8 = _mm_add_ps(v_in11, v_in3);
        v_s9 = _mm_sub_ps(v_in10, v_in4);
        v_s10 = _mm_add_ps(v_in10, v_in4);
        v_s11 = _mm_sub_ps(v_in9, v_in5);
        v_s12 = _mm_add_ps(v_in9, v_in5);
        v_s13 = _mm_sub_ps(v_in8, v_in6);
        v_s14 = _mm_add_ps(v_in8, v_in6);

        v_s15 = _mm_add_ps(v_s4, v_s14);
        v_s16 = _mm_add_ps(v_s6, v_s12);
        v_s17 = _mm_add_ps(v_s8, v_s10);

        v_s18 = _mm_sub_ps(v_s14, v_s4);
        v_s19 = _mm_sub_ps(v_s6, v_s12);
        v_s20 = _mm_sub_ps(v_s10, v_s8);
        v_s27 = _mm_add_ps(v_s2, v_s15);
        v_s28 = _mm_add_ps(v_s16, v_s17);
        v_s29 = _mm_add_ps(v_s1, v_s18);
        v_s30 = _mm_add_ps(v_s19, v_s20);
        // Output pt 1: X(0)
        v_out0 = _mm_add_ps(v_s27, v_s28);
        STR_128_S(curr_out, v_out_stride, v_out0);
        // Output pt 14: X(13)
        v_out13 = _mm_add_ps(v_s29, v_s30);
        curr_out = out + out_strides[13];
        STR_128_S(curr_out, v_out_stride, v_out13);

        v_t1 = _mm_mul_ps(v128_CRTM_14_1, v_s18);
        v_t2 = _mm_mul_ps(v128_CRTM_14_3, v_s19);
        v_t3 = _mm_mul_ps(v128_CRTM_14_5, v_s20);
        v_s31 = _mm_sub_ps(v_s1, v_t1);
        v_s32 = _mm_sub_ps(v_t2, v_t3);
        // Output point 2: X(1)
        v_out1 = _mm_add_ps(v_s31, v_s32);

        v_s21 = _mm_add_ps(v_s3, v_s13);
        v_s22 = _mm_add_ps(v_s5, v_s11);
        v_s23 = _mm_add_ps(v_s7, v_s9);

        v_t4 = _mm_mul_ps(v128_CRTM_14_2, v_s21);
        v_t5 = _mm_mul_ps(v128_CRTM_14_4, v_s22);
        v_t6 = _mm_mul_ps(v128_CRTM_14_6, v_s23);
        v_s33 = _mm_add_ps(v_t4, v_t5);
        // Output point 3: X(2)
        v_out2 = _mm_add_ps(v_s33, v_t6);
        curr_out = out + out_strides[1];
        STRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);

        v_t7 = _mm_mul_ps(v128_CRTM_14_1, v_s17);
        v_t8 = _mm_mul_ps(v128_CRTM_14_3, v_s15);
        v_t9 = _mm_mul_ps(v128_CRTM_14_5, v_s16);
        v_s34 = _mm_sub_ps(v_s2, v_t7);
        v_s35 = _mm_sub_ps(v_t8, v_t9);
        // Output point 4: X(3)
        v_out3 = _mm_add_ps(v_s34, v_s35);

        v_s24 = _mm_sub_ps(v_s3, v_s13);
        v_s25 = _mm_sub_ps(v_s5, v_s11);
        v_s26 = _mm_sub_ps(v_s7, v_s9);

        v_t10 = _mm_mul_ps(v128_CRTM_14_2, v_s26);
        v_t11 = _mm_mul_ps(v128_CRTM_14_4, v_s24);
        v_t12 = _mm_mul_ps(v128_CRTM_14_6, v_s25);
        v_s36 = _mm_add_ps(v_t10, v_t11);
        // Output point 5: X(4)
        v_out4 = _mm_add_ps(v_s36, v_t12);
        curr_out = out + out_strides[3];
        STRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);

        v_t13 = _mm_mul_ps(v128_CRTM_14_1, v_s19);
        v_t14 = _mm_mul_ps(v128_CRTM_14_3, v_s20);
        v_t15 = _mm_mul_ps(v128_CRTM_14_5, v_s18);
        v_s37 = _mm_sub_ps(v_s1, v_t13);
        v_s38 = _mm_sub_ps(v_t14, v_t15);
        // Output point 6: X(5)
        v_out5 = _mm_add_ps(v_s37, v_s38);

        v_t16 = _mm_mul_ps(v128_CRTM_14_2, v_s22);
        v_t17 = _mm_mul_ps(v128_CRTM_14_4, v_s23);
        v_t18 = _mm_mul_ps(v128_CRTM_14_6, v_s21);
        v_s39 = _mm_sub_ps(v_t16, v_t17);
        // Output point 7: X(6)
        v_out6 = _mm_add_ps(v_s39, v_t18);
        curr_out = out + out_strides[5];
        STRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);

        v_t19 = _mm_mul_ps(v128_CRTM_14_1, v_s16);
        v_t20 = _mm_mul_ps(v128_CRTM_14_3, v_s17);
        v_t21 = _mm_mul_ps(v128_CRTM_14_5, v_s15);
        v_s40 = _mm_sub_ps(v_s2, v_t19);
        v_s41 = _mm_sub_ps(v_t20, v_t21);
        // Output point 8: X(7)
        v_out7 = _mm_add_ps(v_s40, v_s41);

        v_t22 = _mm_mul_ps(v128_CRTM_14_2, v_s25);
        v_t23 = _mm_mul_ps(v128_CRTM_14_4, v_s26);
        v_t24 = _mm_mul_ps(v128_CRTM_14_6, v_s24);
        v_s42 = _mm_sub_ps(v_t24, v_t22);
        // Output point 9: X(8)
        v_out8 = _mm_sub_ps(v_s42, v_t23);
        curr_out = out + out_strides[7];
        STRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);

        v_t25 = _mm_mul_ps(v128_CRTM_14_1, v_s20);
        v_t26 = _mm_mul_ps(v128_CRTM_14_3, v_s18);
        v_t27 = _mm_mul_ps(v128_CRTM_14_5, v_s19);
        v_s43 = _mm_sub_ps(v_s1, v_t25);
        v_s44 = _mm_sub_ps(v_t26, v_t27);
        // Output point 10: X(9)
        v_out9 = _mm_add_ps(v_s43, v_s44);

        v_t28 = _mm_mul_ps(v128_CRTM_14_2, v_s23);
        v_t29 = _mm_mul_ps(v128_CRTM_14_4, v_s21);
        v_t30 = _mm_mul_ps(v128_CRTM_14_6, v_s22);
        v_s45 = _mm_add_ps(v_t28, v_t29);
        // Output point 11: X(10)
        v_out10 = _mm_sub_ps(v_s45, v_t30);
        curr_out = out + out_strides[9];
        STRI_2x128_S(curr_out, v_out_stride, v_out9, v_out10);

        v_t31 = _mm_mul_ps(v128_CRTM_14_1, v_s15);
        v_t32 = _mm_mul_ps(v128_CRTM_14_3, v_s16);
        v_t33 = _mm_mul_ps(v128_CRTM_14_5, v_s17);
        v_s46 = _mm_sub_ps(v_s2, v_t31);
        v_s47 = _mm_sub_ps(v_t32, v_t33);
        // Output point 12: X(11)
        v_out11 = _mm_add_ps(v_s46, v_s47);

        v_t34 = _mm_mul_ps(v128_CRTM_14_2, v_s24);
        v_t35 = _mm_mul_ps(v128_CRTM_14_4, v_s25);
        v_t36 = _mm_mul_ps(v128_CRTM_14_6, v_s26);
        v_s48 = _mm_sub_ps(v_t34, v_t35);
        // Output point 13: X(12)
        v_out12 = _mm_add_ps(v_s48, v_t36);
        curr_out = out + out_strides[11];
        STRI_2x128_S(curr_out, v_out_stride, v_out11, v_out12);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (n & 2)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9, v_in10, v_in11, v_in12, v_in13;
        __m128 v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9, v_s10,
               v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18, v_s19,
               v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27, v_s28,
               v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36, v_s37,
               v_s38, v_s39, v_s40, v_s41, v_s42, v_s43, v_s44, v_s45, v_s46,
               v_s47, v_s48;
        __m128 v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9, v_t10,
               v_t11, v_t12, v_t13, v_t14, v_t15, v_t16, v_t17, v_t18, v_t19,
               v_t20, v_t21, v_t22, v_t23, v_t24, v_t25, v_t26, v_t27, v_t28,
               v_t29, v_t30, v_t31, v_t32, v_t33, v_t34, v_t35, v_t36;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_14_1 = _mm512_castps512_ps128(v_CRTM_14_1);
        __m128 v128_CRTM_14_2 = _mm512_castps512_ps128(v_CRTM_14_2);
        __m128 v128_CRTM_14_3 = _mm512_castps512_ps128(v_CRTM_14_3);
        __m128 v128_CRTM_14_4 = _mm512_castps512_ps128(v_CRTM_14_4);
        __m128 v128_CRTM_14_5 = _mm512_castps512_ps128(v_CRTM_14_5);
        __m128 v128_CRTM_14_6 = _mm512_castps512_ps128(v_CRTM_14_6);

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

        v_s1 = _mm_sub_ps(v_in0, v_in7);
        v_s2 = _mm_add_ps(v_in0, v_in7);
        v_s3 = _mm_sub_ps(v_in13, v_in1);
        v_s4 = _mm_add_ps(v_in13, v_in1);
        v_s5 = _mm_sub_ps(v_in12, v_in2);
        v_s6 = _mm_add_ps(v_in12, v_in2);
        v_s7 = _mm_sub_ps(v_in11, v_in3);
        v_s8 = _mm_add_ps(v_in11, v_in3);
        v_s9 = _mm_sub_ps(v_in10, v_in4);
        v_s10 = _mm_add_ps(v_in10, v_in4);
        v_s11 = _mm_sub_ps(v_in9, v_in5);
        v_s12 = _mm_add_ps(v_in9, v_in5);
        v_s13 = _mm_sub_ps(v_in8, v_in6);
        v_s14 = _mm_add_ps(v_in8, v_in6);

        v_s15 = _mm_add_ps(v_s4, v_s14);
        v_s16 = _mm_add_ps(v_s6, v_s12);
        v_s17 = _mm_add_ps(v_s8, v_s10);

        v_s18 = _mm_sub_ps(v_s14, v_s4);
        v_s19 = _mm_sub_ps(v_s6, v_s12);
        v_s20 = _mm_sub_ps(v_s10, v_s8);
        v_s27 = _mm_add_ps(v_s2, v_s15);
        v_s28 = _mm_add_ps(v_s16, v_s17);
        v_s29 = _mm_add_ps(v_s1, v_s18);
        v_s30 = _mm_add_ps(v_s19, v_s20);
        // Output pt 1: X(0)
        v_out0 = _mm_add_ps(v_s27, v_s28);
        STHR_128_S(curr_out, v_out_stride, v_out0);
        // Output pt 14: X(13)
        v_out13 = _mm_add_ps(v_s29, v_s30);
        curr_out = out + out_strides[13];
        STHR_128_S(curr_out, v_out_stride, v_out13);

        v_t1 = _mm_mul_ps(v128_CRTM_14_1, v_s18);
        v_t2 = _mm_mul_ps(v128_CRTM_14_3, v_s19);
        v_t3 = _mm_mul_ps(v128_CRTM_14_5, v_s20);
        v_s31 = _mm_sub_ps(v_s1, v_t1);
        v_s32 = _mm_sub_ps(v_t2, v_t3);
        // Output point 2: X(1)
        v_out1 = _mm_add_ps(v_s31, v_s32);

        v_s21 = _mm_add_ps(v_s3, v_s13);
        v_s22 = _mm_add_ps(v_s5, v_s11);
        v_s23 = _mm_add_ps(v_s7, v_s9);

        v_t4 = _mm_mul_ps(v128_CRTM_14_2, v_s21);
        v_t5 = _mm_mul_ps(v128_CRTM_14_4, v_s22);
        v_t6 = _mm_mul_ps(v128_CRTM_14_6, v_s23);
        v_s33 = _mm_add_ps(v_t4, v_t5);
        // Output point 3: X(2)
        v_out2 = _mm_add_ps(v_s33, v_t6);
        curr_out = out + out_strides[1];
        STHRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);

        v_t7 = _mm_mul_ps(v128_CRTM_14_1, v_s17);
        v_t8 = _mm_mul_ps(v128_CRTM_14_3, v_s15);
        v_t9 = _mm_mul_ps(v128_CRTM_14_5, v_s16);
        v_s34 = _mm_sub_ps(v_s2, v_t7);
        v_s35 = _mm_sub_ps(v_t8, v_t9);
        // Output point 4: X(3)
        v_out3 = _mm_add_ps(v_s34, v_s35);

        v_s24 = _mm_sub_ps(v_s3, v_s13);
        v_s25 = _mm_sub_ps(v_s5, v_s11);
        v_s26 = _mm_sub_ps(v_s7, v_s9);

        v_t10 = _mm_mul_ps(v128_CRTM_14_2, v_s26);
        v_t11 = _mm_mul_ps(v128_CRTM_14_4, v_s24);
        v_t12 = _mm_mul_ps(v128_CRTM_14_6, v_s25);
        v_s36 = _mm_add_ps(v_t10, v_t11);
        // Output point 5: X(4)
        v_out4 = _mm_add_ps(v_s36, v_t12);
        curr_out = out + out_strides[3];
        STHRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);

        v_t13 = _mm_mul_ps(v128_CRTM_14_1, v_s19);
        v_t14 = _mm_mul_ps(v128_CRTM_14_3, v_s20);
        v_t15 = _mm_mul_ps(v128_CRTM_14_5, v_s18);
        v_s37 = _mm_sub_ps(v_s1, v_t13);
        v_s38 = _mm_sub_ps(v_t14, v_t15);
        // Output point 6: X(5)
        v_out5 = _mm_add_ps(v_s37, v_s38);

        v_t16 = _mm_mul_ps(v128_CRTM_14_2, v_s22);
        v_t17 = _mm_mul_ps(v128_CRTM_14_4, v_s23);
        v_t18 = _mm_mul_ps(v128_CRTM_14_6, v_s21);
        v_s39 = _mm_sub_ps(v_t16, v_t17);
        // Output point 7: X(6)
        v_out6 = _mm_add_ps(v_s39, v_t18);
        curr_out = out + out_strides[5];
        STHRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);

        v_t19 = _mm_mul_ps(v128_CRTM_14_1, v_s16);
        v_t20 = _mm_mul_ps(v128_CRTM_14_3, v_s17);
        v_t21 = _mm_mul_ps(v128_CRTM_14_5, v_s15);
        v_s40 = _mm_sub_ps(v_s2, v_t19);
        v_s41 = _mm_sub_ps(v_t20, v_t21);
        // Output point 8: X(7)
        v_out7 = _mm_add_ps(v_s40, v_s41);

        v_t22 = _mm_mul_ps(v128_CRTM_14_2, v_s25);
        v_t23 = _mm_mul_ps(v128_CRTM_14_4, v_s26);
        v_t24 = _mm_mul_ps(v128_CRTM_14_6, v_s24);
        v_s42 = _mm_sub_ps(v_t24, v_t22);
        // Output point 9: X(8)
        v_out8 = _mm_sub_ps(v_s42, v_t23);
        curr_out = out + out_strides[7];
        STHRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);

        v_t25 = _mm_mul_ps(v128_CRTM_14_1, v_s20);
        v_t26 = _mm_mul_ps(v128_CRTM_14_3, v_s18);
        v_t27 = _mm_mul_ps(v128_CRTM_14_5, v_s19);
        v_s43 = _mm_sub_ps(v_s1, v_t25);
        v_s44 = _mm_sub_ps(v_t26, v_t27);
        // Output point 10: X(9)
        v_out9 = _mm_add_ps(v_s43, v_s44);

        v_t28 = _mm_mul_ps(v128_CRTM_14_2, v_s23);
        v_t29 = _mm_mul_ps(v128_CRTM_14_4, v_s21);
        v_t30 = _mm_mul_ps(v128_CRTM_14_6, v_s22);
        v_s45 = _mm_add_ps(v_t28, v_t29);
        // Output point 11: X(10)
        v_out10 = _mm_sub_ps(v_s45, v_t30);
        curr_out = out + out_strides[9];
        STHRI_2x128_S(curr_out, v_out_stride, v_out9, v_out10);

        v_t31 = _mm_mul_ps(v128_CRTM_14_1, v_s15);
        v_t32 = _mm_mul_ps(v128_CRTM_14_3, v_s16);
        v_t33 = _mm_mul_ps(v128_CRTM_14_5, v_s17);
        v_s46 = _mm_sub_ps(v_s2, v_t31);
        v_s47 = _mm_sub_ps(v_t32, v_t33);
        // Output point 12: X(11)
        v_out11 = _mm_add_ps(v_s46, v_s47);

        v_t34 = _mm_mul_ps(v128_CRTM_14_2, v_s24);
        v_t35 = _mm_mul_ps(v128_CRTM_14_4, v_s25);
        v_t36 = _mm_mul_ps(v128_CRTM_14_6, v_s26);
        v_s48 = _mm_sub_ps(v_t34, v_t35);
        // Output point 13: X(12)
        v_out12 = _mm_add_ps(v_s48, v_t36);
        curr_out = out + out_strides[11];
        STHRI_2x128_S(curr_out, v_out_stride, v_out11, v_out12);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        FLOAT in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11,
              in12, in13;
        FLOAT s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14,
              s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25;
        FLOAT t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14,
              t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27,
              t28, t29, t30, t31, t32, t33, t34, t35;

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

        s0 = in0 - in7;
        s1 = in0 + in7;
        s2 = in13 - in1;
        s3 = in13 + in1;
        s4 = in12 - in2;
        s5 = in12 + in2;
        s6 = in11 - in3;
        s7 = in11 + in3;
        s8 = in10 - in4;
        s9 = in10 + in4;
        s10 = in9 - in5;
        s11 = in9 + in5;
        s12 = in8 - in6;
        s13 = in8 + in6;

        s14 = s3 + s13;
        s15 = s5 + s11;
        s16 = s7 + s9;

        s17 = s13 - s3;
        s18 = s5 - s11;
        s19 = s9 - s7;

        // Output point 1: X(0)
        *out = s1 + s14 + s15 + s16;
        // Output point 13: X(14)
        out[out_strides[13]] = s0 + s17 + s18 + s19;

        t0 = CRTM_14_1 * s17;
        t1 = CRTM_14_3 * s18;
        t2 = CRTM_14_5 * s19;

        // Output point 2: X(1)
        out[out_strides[1]] = s0 - t0 + t1 - t2;

        s20 = s2 + s12;
        s21 = s4 + s10;
        s22 = s6 + s8;

        t3 = CRTM_14_2 * s20;
        t4 = CRTM_14_4 * s21;
        t5 = CRTM_14_6 * s22;

        // Output point 3: X(2)
        out[out_strides[2]] = t3 + t4 + t5;

        t6 = CRTM_14_1 * s16;
        t7 = CRTM_14_3 * s14;
        t8 = CRTM_14_5 * s15;

        // Output point 4: X(3)
        out[out_strides[3]] = s1 - t6 + t7 - t8;

        s23 = s2 - s12;
        s24 = s4 - s10;
        s25 = s6 - s8;

        t9 = CRTM_14_2 * s25;
        t10 = CRTM_14_4 * s23;
        t11 = CRTM_14_6 * s24;

        // Output point 5: X(4)
        out[out_strides[4]] = t9 + t10 + t11;

        t12 = CRTM_14_1 * s18;
        t13 = CRTM_14_3 * s19;
        t14 = CRTM_14_5 * s17;
        // Output point 6: X(5)
        out[out_strides[5]] = s0 - t12 + t13 - t14;
        t15 = CRTM_14_2 * s21;
        t16 = CRTM_14_4 * s22;
        t17 = CRTM_14_6 * s20;
        // Output point 7: X(6)
        out[out_strides[6]] = t15 - t16 + t17;

        t18 = CRTM_14_1 * s15;
        t19 = CRTM_14_3 * s16;
        t20 = CRTM_14_5 * s14;
        // Output point 8: X(7)
        out[out_strides[7]] = s1 - t18 + t19 - t20;
        t21 = CRTM_14_2 * s24;
        t22 = CRTM_14_4 * s25;
        t23 = CRTM_14_6 * s23;
        // Output point 9: X(8)
        out[out_strides[8]] = t23 - t22 - t21;

        t24 = CRTM_14_1 * s19;
        t25 = CRTM_14_3 * s17;
        t26 = CRTM_14_5 * s18;
        // Output point 10: X(9)
        out[out_strides[9]] = s0 - t24 + t25 - t26;
        t27 = CRTM_14_2 * s22;
        t28 = CRTM_14_4 * s20;
        t29 = CRTM_14_6 * s21;
        // Output point 11: X(10)
        out[out_strides[10]] = t27 + t28 - t29;

        t30 = CRTM_14_1 * s14;
        t31 = CRTM_14_3 * s15;
        t32 = CRTM_14_5 * s16;
        // Output point 5: X(4)
        out[out_strides[11]] = s1 - t30 + t31 - t32;
        t33 = CRTM_14_2 * s23;
        t34 = CRTM_14_4 * s24;
        t35 = CRTM_14_6 * s25;
        // Output point 5: X(4)
        out[out_strides[12]] = t33 - t34 + t35;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID r2hc_rfft14avx512_fp32_bwd(VOID *in_real, VOID *in_imag,
                                       VOID *out_real, VOID *out_imag, INTP n,
                                       aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const FLOAT CRTM_14_1 = 0.867767478235116240951536665696717509219981456f;
    const FLOAT CRTM_14_2 = 1.801937735804838252472204639014890102331838324f;
    const FLOAT CRTM_14_3 = 1.563662964936059617416889053348115500464669038f;
    const FLOAT CRTM_14_4 = 1.246979603717467061050009768008479621264549462f;
    const FLOAT CRTM_14_5 = 1.949855824363647214036263365987862434465571602f;
    const FLOAT CRTM_14_6 = 0.445041867912628808577805128993589518932711138f;
    const FLOAT CRTM_14_7 = 2.000000000000000000000000000000000000000000000f;

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

    __m512 v_CRTM_14_1 = _mm512_set1_ps(CRTM_14_1);
    __m512 v_CRTM_14_2 = _mm512_set1_ps(CRTM_14_2);
    __m512 v_CRTM_14_3 = _mm512_set1_ps(CRTM_14_3);
    __m512 v_CRTM_14_4 = _mm512_set1_ps(CRTM_14_4);
    __m512 v_CRTM_14_5 = _mm512_set1_ps(CRTM_14_5);
    __m512 v_CRTM_14_6 = _mm512_set1_ps(CRTM_14_6);
    __m512 v_CRTM_14_7 = _mm512_set1_ps(CRTM_14_7);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m512 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9, v_in10, v_in11, v_in12, v_in13;
        __m512 v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9, v_s10,
               v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18, v_s19,
               v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27, v_s28,
               v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36, v_s37,
               v_s38, v_s39, v_s40, v_s41, v_s42, v_s43, v_s44, v_s45, v_s46,
               v_s47, v_s48;
        __m512 v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9, v_t10,
               v_t11, v_t12, v_t13, v_t14, v_t15, v_t16, v_t17, v_t18, v_t19,
               v_t20, v_t21, v_t22, v_t23, v_t24, v_t25, v_t26, v_t27, v_t28,
               v_t29, v_t30, v_t31, v_t32, v_t33, v_t34, v_t35, v_t36, v_t37,
               v_t38;
        __m512 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13;

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
        // Input point 12: x(11) & Input point 13: x(12)
        curr_in = in + in_strides[11];
        LDRI_2x512_S(curr_in, v_in_stride, v_in11, v_in12);
        // Input point 14: x(13) & Input point 15: x(14)
        curr_in = in + in_strides[13];
        LDR_512_S(curr_in, v_in_stride, v_in13);

        v_s1 = _mm512_sub_ps(v_in0, v_in13);
        v_s2 = _mm512_add_ps(v_in0, v_in13);
        v_s3 = _mm512_sub_ps(v_in11, v_in1);
        v_s4 = _mm512_add_ps(v_in1, v_in11);
        v_s5 = _mm512_sub_ps(v_in2, v_in12);
        v_s6 = _mm512_add_ps(v_in2, v_in12);
        v_s7 = _mm512_sub_ps(v_in3, v_in9);
        v_s8 = _mm512_add_ps(v_in3, v_in9);
        v_s9 = _mm512_sub_ps(v_in4, v_in10);
        v_s10 = _mm512_add_ps(v_in4, v_in10);
        v_s11 = _mm512_sub_ps(v_in7, v_in5);
        v_s12 = _mm512_add_ps(v_in5, v_in7);
        v_s13 = _mm512_sub_ps(v_in6, v_in8);
        v_s14 = _mm512_add_ps(v_in6, v_in8);

        v_s27 = _mm512_add_ps(v_s12, v_s4);
        v_s28 = _mm512_add_ps(v_s27, v_s8);
        v_t37 = _mm512_mul_ps(v_CRTM_14_7, v_s28);
        v_s29 = _mm512_add_ps(v_s3, v_s7);
        v_s30 = _mm512_add_ps(v_s29, v_s11);
        v_t38 = _mm512_mul_ps(v_CRTM_14_7, v_s30);
        // Output pt 1: X(0)
        v_out0 = _mm512_add_ps(v_t37, v_s2);
        STR_512_S(curr_out, v_out_stride, v_out0);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = _mm512_add_ps(v_t38, v_s1);
        STR_512_S(curr_out, v_out_stride, v_out7);

        v_t1 = _mm512_mul_ps(v_CRTM_14_1, v_s6);
        v_t2 = _mm512_mul_ps(v_CRTM_14_3, v_s10);
        v_t3 = _mm512_mul_ps(v_CRTM_14_5, v_s14);
        v_t4 = _mm512_mul_ps(v_CRTM_14_2, v_s3);
        v_t5 = _mm512_mul_ps(v_CRTM_14_4, v_s7);
        v_t6 = _mm512_mul_ps(v_CRTM_14_6, v_s11);

        v_s31 = _mm512_sub_ps(v_t5, v_t6);
        v_s32 = _mm512_sub_ps(v_s1, v_t4);
        v_s33 = _mm512_add_ps(v_t1, v_t2);

        v_s15 = _mm512_add_ps(v_s31, v_s32);
        v_s16 = _mm512_add_ps(v_s33, v_t3);
        // Output point 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm512_sub_ps(v_s15, v_s16);
        STR_512_S(curr_out, v_out_stride, v_out1);
        // Output point 14: X(13)
        curr_out = out + out_strides[13];
        v_out13 = _mm512_add_ps(v_s15, v_s16);
        STR_512_S(curr_out, v_out_stride, v_out13);

        v_t7 = _mm512_mul_ps(v_CRTM_14_1, v_s13);
        v_t8 = _mm512_mul_ps(v_CRTM_14_3, v_s5);
        v_t9 = _mm512_mul_ps(v_CRTM_14_5, v_s9);

        v_t10 = _mm512_mul_ps(v_CRTM_14_2, v_s12);
        v_t11 = _mm512_mul_ps(v_CRTM_14_4, v_s4);
        v_t12 = _mm512_mul_ps(v_CRTM_14_6, v_s8);

        v_s34 = _mm512_sub_ps(v_s2, v_t10);
        v_s35 = _mm512_sub_ps(v_t11, v_t12);
        v_s36 = _mm512_add_ps(v_t7, v_t8);

        v_s17 = v_s34 + v_s35;
        v_s18 = _mm512_add_ps(v_s36, v_t9);
        // Output point 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm512_sub_ps(v_s17, v_s18);
        STR_512_S(curr_out, v_out_stride, v_out2);
        // Output point 13: X(12)
        curr_out = out + out_strides[12];
        v_out12 = _mm512_add_ps(v_s17, v_s18);
        STR_512_S(curr_out, v_out_stride, v_out12);

        v_t13 = _mm512_mul_ps(v_CRTM_14_1, v_s10);
        v_t14 = _mm512_mul_ps(v_CRTM_14_3, v_s14);
        v_t15 = _mm512_mul_ps(v_CRTM_14_5, v_s6);
        v_t16 = _mm512_mul_ps(v_CRTM_14_2, v_s7);
        v_t17 = _mm512_mul_ps(v_CRTM_14_4, v_s11);
        v_t18 = _mm512_mul_ps(v_CRTM_14_6, v_s3);

        v_s37 = _mm512_sub_ps(v_s1, v_t16);
        v_s38 = _mm512_sub_ps(v_t17, v_t18);
        v_s39 = _mm512_sub_ps(v_t14, v_t15);
        v_s19 = _mm512_add_ps(v_s37, v_s38);
        v_s20 = _mm512_sub_ps(v_s39, v_t13);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm512_add_ps(v_s19, v_s20);
        STR_512_S(curr_out, v_out_stride, v_out3);
        // Output point 12: X(11)
        curr_out = out + out_strides[11];
        v_out11 = _mm512_sub_ps(v_s19, v_s20);
        STR_512_S(curr_out, v_out_stride, v_out11);

        v_t19 = _mm512_mul_ps(v_CRTM_14_1, v_s9);
        v_t20 = _mm512_mul_ps(v_CRTM_14_3, v_s13);
        v_t21 = _mm512_mul_ps(v_CRTM_14_5, v_s5);
        v_t22 = _mm512_mul_ps(v_CRTM_14_2, v_s8);
        v_t23 = _mm512_mul_ps(v_CRTM_14_4, v_s12);
        v_t24 = _mm512_mul_ps(v_CRTM_14_6, v_s4);

        v_s40 = _mm512_sub_ps(v_s2, v_t22);
        v_s41 = _mm512_sub_ps(v_t23, v_t24);
        v_s42 = _mm512_add_ps(v_t19, v_t20);
        v_s21 = _mm512_add_ps(v_s40, v_s41);
        v_s22 = _mm512_sub_ps(v_s42, v_t21);
        // Output pt 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm512_add_ps(v_s21, v_s22);
        STR_512_S(curr_out, v_out_stride, v_out4);
        // Output point 11: X(10)
        curr_out = out + out_strides[10];
        v_out10 = _mm512_sub_ps(v_s21, v_s22);
        STR_512_S(curr_out, v_out_stride, v_out10);

        v_t25 = _mm512_mul_ps(v_CRTM_14_1, v_s14);
        v_t26 = _mm512_mul_ps(v_CRTM_14_3, v_s6);
        v_t27 = _mm512_mul_ps(v_CRTM_14_5, v_s10);
        v_t28 = _mm512_mul_ps(v_CRTM_14_2, v_s11);
        v_t29 = _mm512_mul_ps(v_CRTM_14_4, v_s3);
        v_t30 = _mm512_mul_ps(v_CRTM_14_6, v_s7);

        v_s43 = _mm512_sub_ps(v_s1, v_t28);
        v_s44 = _mm512_sub_ps(v_t29, v_t30);
        v_s45 = _mm512_sub_ps(v_t27, v_t25);
        v_s23 = _mm512_add_ps(v_s43, v_s44);
        v_s24 = _mm512_sub_ps(v_s45, v_t26);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm512_add_ps(v_s23, v_s24);
        STR_512_S(curr_out, v_out_stride, v_out5);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm512_sub_ps(v_s23, v_s24);
        STR_512_S(curr_out, v_out_stride, v_out9);

        v_t31 = _mm512_mul_ps(v_CRTM_14_1, v_s5);
        v_t32 = _mm512_mul_ps(v_CRTM_14_3, v_s9);
        v_t33 = _mm512_mul_ps(v_CRTM_14_5, v_s13);
        v_t34 = _mm512_mul_ps(v_CRTM_14_2, v_s4);
        v_t35 = _mm512_mul_ps(v_CRTM_14_4, v_s8);
        v_t36 = _mm512_mul_ps(v_CRTM_14_6, v_s12);

        v_s46 = _mm512_sub_ps(v_s2, v_t34);
        v_s47 = _mm512_sub_ps(v_t35, v_t36);
        v_s48 = _mm512_sub_ps(v_t32, v_t31);
        v_s25 = _mm512_add_ps(v_s46, v_s47);
        v_s26 = _mm512_sub_ps(v_s48, v_t33);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm512_add_ps(v_s25, v_s26);
        STR_512_S(curr_out, v_out_stride, v_out6);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm512_sub_ps(v_s25, v_s26);
        STR_512_S(curr_out, v_out_stride, v_out8);

        in = in + (v_in_stride << 4);
        out = out + (v_out_stride << 4);
    }
    // tail cases
    if (n & 8)
    {
        __m256 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9, v_in10, v_in11, v_in12, v_in13;
        __m256 v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9, v_s10,
               v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18, v_s19,
               v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27, v_s28,
               v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36, v_s37,
               v_s38, v_s39, v_s40, v_s41, v_s42, v_s43, v_s44, v_s45, v_s46,
               v_s47, v_s48;
        __m256 v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9, v_t10,
               v_t11, v_t12, v_t13, v_t14, v_t15, v_t16, v_t17, v_t18, v_t19,
               v_t20, v_t21, v_t22, v_t23, v_t24, v_t25, v_t26, v_t27, v_t28,
               v_t29, v_t30, v_t31, v_t32, v_t33, v_t34, v_t35, v_t36, v_t37,
               v_t38;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13;

        curr_in = in;
        curr_out = out;

        __m256 v256_CRTM_14_1 = _mm512_castps512_ps256(v_CRTM_14_1);
        __m256 v256_CRTM_14_2 = _mm512_castps512_ps256(v_CRTM_14_2);
        __m256 v256_CRTM_14_3 = _mm512_castps512_ps256(v_CRTM_14_3);
        __m256 v256_CRTM_14_4 = _mm512_castps512_ps256(v_CRTM_14_4);
        __m256 v256_CRTM_14_5 = _mm512_castps512_ps256(v_CRTM_14_5);
        __m256 v256_CRTM_14_6 = _mm512_castps512_ps256(v_CRTM_14_6);
        __m256 v256_CRTM_14_7 = _mm512_castps512_ps256(v_CRTM_14_7);

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
        // Input point 12: x(11) & Input point 13: x(12)
        curr_in = in + in_strides[11];
        LDRI_2x256_S(curr_in, v_in_stride, v_in11, v_in12);
        // Input point 14: x(13) & Input point 15: x(14)
        curr_in = in + in_strides[13];
        LDR_256_S(curr_in, v_in_stride, v_in13);

        v_s1 = _mm256_sub_ps(v_in0, v_in13);
        v_s2 = _mm256_add_ps(v_in0, v_in13);
        v_s3 = _mm256_sub_ps(v_in11, v_in1);
        v_s4 = _mm256_add_ps(v_in1, v_in11);
        v_s5 = _mm256_sub_ps(v_in2, v_in12);
        v_s6 = _mm256_add_ps(v_in2, v_in12);
        v_s7 = _mm256_sub_ps(v_in3, v_in9);
        v_s8 = _mm256_add_ps(v_in3, v_in9);
        v_s9 = _mm256_sub_ps(v_in4, v_in10);
        v_s10 = _mm256_add_ps(v_in4, v_in10);
        v_s11 = _mm256_sub_ps(v_in7, v_in5);
        v_s12 = _mm256_add_ps(v_in5, v_in7);
        v_s13 = _mm256_sub_ps(v_in6, v_in8);
        v_s14 = _mm256_add_ps(v_in6, v_in8);

        v_s27 = _mm256_add_ps(v_s12, v_s4);
        v_s28 = _mm256_add_ps(v_s27, v_s8);
        v_t37 = _mm256_mul_ps(v256_CRTM_14_7, v_s28);
        v_s29 = _mm256_add_ps(v_s3, v_s7);
        v_s30 = _mm256_add_ps(v_s29, v_s11);
        v_t38 = _mm256_mul_ps(v256_CRTM_14_7, v_s30);
        // Output pt 1: X(0)
        v_out0 = _mm256_add_ps(v_t37, v_s2);
        STR_256_S(curr_out, v_out_stride, v_out0);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = _mm256_add_ps(v_t38, v_s1);
        STR_256_S(curr_out, v_out_stride, v_out7);

        v_t1 = _mm256_mul_ps(v256_CRTM_14_1, v_s6);
        v_t2 = _mm256_mul_ps(v256_CRTM_14_3, v_s10);
        v_t3 = _mm256_mul_ps(v256_CRTM_14_5, v_s14);
        v_t4 = _mm256_mul_ps(v256_CRTM_14_2, v_s3);
        v_t5 = _mm256_mul_ps(v256_CRTM_14_4, v_s7);
        v_t6 = _mm256_mul_ps(v256_CRTM_14_6, v_s11);

        v_s31 = _mm256_sub_ps(v_t5, v_t6);
        v_s32 = _mm256_sub_ps(v_s1, v_t4);
        v_s33 = _mm256_add_ps(v_t1, v_t2);

        v_s15 = _mm256_add_ps(v_s31, v_s32);
        v_s16 = _mm256_add_ps(v_s33, v_t3);
        // Output point 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm256_sub_ps(v_s15, v_s16);
        STR_256_S(curr_out, v_out_stride, v_out1);
        // Output point 14: X(13)
        curr_out = out + out_strides[13];
        v_out13 = _mm256_add_ps(v_s15, v_s16);
        STR_256_S(curr_out, v_out_stride, v_out13);

        v_t7 = _mm256_mul_ps(v256_CRTM_14_1, v_s13);
        v_t8 = _mm256_mul_ps(v256_CRTM_14_3, v_s5);
        v_t9 = _mm256_mul_ps(v256_CRTM_14_5, v_s9);

        v_t10 = _mm256_mul_ps(v256_CRTM_14_2, v_s12);
        v_t11 = _mm256_mul_ps(v256_CRTM_14_4, v_s4);
        v_t12 = _mm256_mul_ps(v256_CRTM_14_6, v_s8);

        v_s34 = _mm256_sub_ps(v_s2, v_t10);
        v_s35 = _mm256_sub_ps(v_t11, v_t12);
        v_s36 = _mm256_add_ps(v_t7, v_t8);

        v_s17 = v_s34 + v_s35;
        v_s18 = _mm256_add_ps(v_s36, v_t9);
        // Output point 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm256_sub_ps(v_s17, v_s18);
        STR_256_S(curr_out, v_out_stride, v_out2);
        // Output point 13: X(12)
        curr_out = out + out_strides[12];
        v_out12 = _mm256_add_ps(v_s17, v_s18);
        STR_256_S(curr_out, v_out_stride, v_out12);

        v_t13 = _mm256_mul_ps(v256_CRTM_14_1, v_s10);
        v_t14 = _mm256_mul_ps(v256_CRTM_14_3, v_s14);
        v_t15 = _mm256_mul_ps(v256_CRTM_14_5, v_s6);
        v_t16 = _mm256_mul_ps(v256_CRTM_14_2, v_s7);
        v_t17 = _mm256_mul_ps(v256_CRTM_14_4, v_s11);
        v_t18 = _mm256_mul_ps(v256_CRTM_14_6, v_s3);

        v_s37 = _mm256_sub_ps(v_s1, v_t16);
        v_s38 = _mm256_sub_ps(v_t17, v_t18);
        v_s39 = _mm256_sub_ps(v_t14, v_t15);
        v_s19 = _mm256_add_ps(v_s37, v_s38);
        v_s20 = _mm256_sub_ps(v_s39, v_t13);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm256_add_ps(v_s19, v_s20);
        STR_256_S(curr_out, v_out_stride, v_out3);
        // Output point 12: X(11)
        curr_out = out + out_strides[11];
        v_out11 = _mm256_sub_ps(v_s19, v_s20);
        STR_256_S(curr_out, v_out_stride, v_out11);

        v_t19 = _mm256_mul_ps(v256_CRTM_14_1, v_s9);
        v_t20 = _mm256_mul_ps(v256_CRTM_14_3, v_s13);
        v_t21 = _mm256_mul_ps(v256_CRTM_14_5, v_s5);
        v_t22 = _mm256_mul_ps(v256_CRTM_14_2, v_s8);
        v_t23 = _mm256_mul_ps(v256_CRTM_14_4, v_s12);
        v_t24 = _mm256_mul_ps(v256_CRTM_14_6, v_s4);

        v_s40 = _mm256_sub_ps(v_s2, v_t22);
        v_s41 = _mm256_sub_ps(v_t23, v_t24);
        v_s42 = _mm256_add_ps(v_t19, v_t20);
        v_s21 = _mm256_add_ps(v_s40, v_s41);
        v_s22 = _mm256_sub_ps(v_s42, v_t21);
        // Output pt 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm256_add_ps(v_s21, v_s22);
        STR_256_S(curr_out, v_out_stride, v_out4);
        // Output point 11: X(10)
        curr_out = out + out_strides[10];
        v_out10 = _mm256_sub_ps(v_s21, v_s22);
        STR_256_S(curr_out, v_out_stride, v_out10);

        v_t25 = _mm256_mul_ps(v256_CRTM_14_1, v_s14);
        v_t26 = _mm256_mul_ps(v256_CRTM_14_3, v_s6);
        v_t27 = _mm256_mul_ps(v256_CRTM_14_5, v_s10);
        v_t28 = _mm256_mul_ps(v256_CRTM_14_2, v_s11);
        v_t29 = _mm256_mul_ps(v256_CRTM_14_4, v_s3);
        v_t30 = _mm256_mul_ps(v256_CRTM_14_6, v_s7);

        v_s43 = _mm256_sub_ps(v_s1, v_t28);
        v_s44 = _mm256_sub_ps(v_t29, v_t30);
        v_s45 = _mm256_sub_ps(v_t27, v_t25);
        v_s23 = _mm256_add_ps(v_s43, v_s44);
        v_s24 = _mm256_sub_ps(v_s45, v_t26);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm256_add_ps(v_s23, v_s24);
        STR_256_S(curr_out, v_out_stride, v_out5);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm256_sub_ps(v_s23, v_s24);
        STR_256_S(curr_out, v_out_stride, v_out9);

        v_t31 = _mm256_mul_ps(v256_CRTM_14_1, v_s5);
        v_t32 = _mm256_mul_ps(v256_CRTM_14_3, v_s9);
        v_t33 = _mm256_mul_ps(v256_CRTM_14_5, v_s13);
        v_t34 = _mm256_mul_ps(v256_CRTM_14_2, v_s4);
        v_t35 = _mm256_mul_ps(v256_CRTM_14_4, v_s8);
        v_t36 = _mm256_mul_ps(v256_CRTM_14_6, v_s12);

        v_s46 = _mm256_sub_ps(v_s2, v_t34);
        v_s47 = _mm256_sub_ps(v_t35, v_t36);
        v_s48 = _mm256_sub_ps(v_t32, v_t31);
        v_s25 = _mm256_add_ps(v_s46, v_s47);
        v_s26 = _mm256_sub_ps(v_s48, v_t33);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm256_add_ps(v_s25, v_s26);
        STR_256_S(curr_out, v_out_stride, v_out6);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm256_sub_ps(v_s25, v_s26);
        STR_256_S(curr_out, v_out_stride, v_out8);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (n & 4)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9, v_in10, v_in11, v_in12, v_in13;
        __m128 v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9, v_s10,
               v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18, v_s19,
               v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27, v_s28,
               v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36, v_s37,
               v_s38, v_s39, v_s40, v_s41, v_s42, v_s43, v_s44, v_s45, v_s46,
               v_s47, v_s48;
        __m128 v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9, v_t10,
               v_t11, v_t12, v_t13, v_t14, v_t15, v_t16, v_t17, v_t18, v_t19,
               v_t20, v_t21, v_t22, v_t23, v_t24, v_t25, v_t26, v_t27, v_t28,
               v_t29, v_t30, v_t31, v_t32, v_t33, v_t34, v_t35, v_t36, v_t37,
               v_t38;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_14_1 = _mm512_castps512_ps128(v_CRTM_14_1);
        __m128 v128_CRTM_14_2 = _mm512_castps512_ps128(v_CRTM_14_2);
        __m128 v128_CRTM_14_3 = _mm512_castps512_ps128(v_CRTM_14_3);
        __m128 v128_CRTM_14_4 = _mm512_castps512_ps128(v_CRTM_14_4);
        __m128 v128_CRTM_14_5 = _mm512_castps512_ps128(v_CRTM_14_5);
        __m128 v128_CRTM_14_6 = _mm512_castps512_ps128(v_CRTM_14_6);
        __m128 v128_CRTM_14_7 = _mm512_castps512_ps128(v_CRTM_14_7);

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
        // Input point 12: x(11) & Input point 13: x(12)
        curr_in = in + in_strides[11];
        LDRI_2x128_S(curr_in, v_in_stride, v_in11, v_in12);
        // Input point 14: x(13) & Input point 15: x(14)
        curr_in = in + in_strides[13];
        LDR_128_S(curr_in, v_in_stride, v_in13);

        v_s1 = _mm_sub_ps(v_in0, v_in13);
        v_s2 = _mm_add_ps(v_in0, v_in13);
        v_s3 = _mm_sub_ps(v_in11, v_in1);
        v_s4 = _mm_add_ps(v_in1, v_in11);
        v_s5 = _mm_sub_ps(v_in2, v_in12);
        v_s6 = _mm_add_ps(v_in2, v_in12);
        v_s7 = _mm_sub_ps(v_in3, v_in9);
        v_s8 = _mm_add_ps(v_in3, v_in9);
        v_s9 = _mm_sub_ps(v_in4, v_in10);
        v_s10 = _mm_add_ps(v_in4, v_in10);
        v_s11 = _mm_sub_ps(v_in7, v_in5);
        v_s12 = _mm_add_ps(v_in5, v_in7);
        v_s13 = _mm_sub_ps(v_in6, v_in8);
        v_s14 = _mm_add_ps(v_in6, v_in8);

        v_s27 = _mm_add_ps(v_s12, v_s4);
        v_s28 = _mm_add_ps(v_s27, v_s8);
        v_t37 = _mm_mul_ps(v128_CRTM_14_7, v_s28);
        v_s29 = _mm_add_ps(v_s3, v_s7);
        v_s30 = _mm_add_ps(v_s29, v_s11);
        v_t38 = _mm_mul_ps(v128_CRTM_14_7, v_s30);
        // Output pt 1: X(0)
        v_out0 = _mm_add_ps(v_t37, v_s2);
        STR_128_S(curr_out, v_out_stride, v_out0);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = _mm_add_ps(v_t38, v_s1);
        STR_128_S(curr_out, v_out_stride, v_out7);

        v_t1 = _mm_mul_ps(v128_CRTM_14_1, v_s6);
        v_t2 = _mm_mul_ps(v128_CRTM_14_3, v_s10);
        v_t3 = _mm_mul_ps(v128_CRTM_14_5, v_s14);
        v_t4 = _mm_mul_ps(v128_CRTM_14_2, v_s3);
        v_t5 = _mm_mul_ps(v128_CRTM_14_4, v_s7);
        v_t6 = _mm_mul_ps(v128_CRTM_14_6, v_s11);

        v_s31 = _mm_sub_ps(v_t5, v_t6);
        v_s32 = _mm_sub_ps(v_s1, v_t4);
        v_s33 = _mm_add_ps(v_t1, v_t2);

        v_s15 = _mm_add_ps(v_s31, v_s32);
        v_s16 = _mm_add_ps(v_s33, v_t3);
        // Output point 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm_sub_ps(v_s15, v_s16);
        STR_128_S(curr_out, v_out_stride, v_out1);
        // Output point 14: X(13)
        curr_out = out + out_strides[13];
        v_out13 = _mm_add_ps(v_s15, v_s16);
        STR_128_S(curr_out, v_out_stride, v_out13);

        v_t7 = _mm_mul_ps(v128_CRTM_14_1, v_s13);
        v_t8 = _mm_mul_ps(v128_CRTM_14_3, v_s5);
        v_t9 = _mm_mul_ps(v128_CRTM_14_5, v_s9);

        v_t10 = _mm_mul_ps(v128_CRTM_14_2, v_s12);
        v_t11 = _mm_mul_ps(v128_CRTM_14_4, v_s4);
        v_t12 = _mm_mul_ps(v128_CRTM_14_6, v_s8);

        v_s34 = _mm_sub_ps(v_s2, v_t10);
        v_s35 = _mm_sub_ps(v_t11, v_t12);
        v_s36 = _mm_add_ps(v_t7, v_t8);

        v_s17 = v_s34 + v_s35;
        v_s18 = _mm_add_ps(v_s36, v_t9);
        // Output point 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm_sub_ps(v_s17, v_s18);
        STR_128_S(curr_out, v_out_stride, v_out2);
        // Output point 13: X(12)
        curr_out = out + out_strides[12];
        v_out12 = _mm_add_ps(v_s17, v_s18);
        STR_128_S(curr_out, v_out_stride, v_out12);

        v_t13 = _mm_mul_ps(v128_CRTM_14_1, v_s10);
        v_t14 = _mm_mul_ps(v128_CRTM_14_3, v_s14);
        v_t15 = _mm_mul_ps(v128_CRTM_14_5, v_s6);
        v_t16 = _mm_mul_ps(v128_CRTM_14_2, v_s7);
        v_t17 = _mm_mul_ps(v128_CRTM_14_4, v_s11);
        v_t18 = _mm_mul_ps(v128_CRTM_14_6, v_s3);

        v_s37 = _mm_sub_ps(v_s1, v_t16);
        v_s38 = _mm_sub_ps(v_t17, v_t18);
        v_s39 = _mm_sub_ps(v_t14, v_t15);
        v_s19 = _mm_add_ps(v_s37, v_s38);
        v_s20 = _mm_sub_ps(v_s39, v_t13);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm_add_ps(v_s19, v_s20);
        STR_128_S(curr_out, v_out_stride, v_out3);
        // Output point 12: X(11)
        curr_out = out + out_strides[11];
        v_out11 = _mm_sub_ps(v_s19, v_s20);
        STR_128_S(curr_out, v_out_stride, v_out11);

        v_t19 = _mm_mul_ps(v128_CRTM_14_1, v_s9);
        v_t20 = _mm_mul_ps(v128_CRTM_14_3, v_s13);
        v_t21 = _mm_mul_ps(v128_CRTM_14_5, v_s5);
        v_t22 = _mm_mul_ps(v128_CRTM_14_2, v_s8);
        v_t23 = _mm_mul_ps(v128_CRTM_14_4, v_s12);
        v_t24 = _mm_mul_ps(v128_CRTM_14_6, v_s4);

        v_s40 = _mm_sub_ps(v_s2, v_t22);
        v_s41 = _mm_sub_ps(v_t23, v_t24);
        v_s42 = _mm_add_ps(v_t19, v_t20);
        v_s21 = _mm_add_ps(v_s40, v_s41);
        v_s22 = _mm_sub_ps(v_s42, v_t21);
        // Output pt 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm_add_ps(v_s21, v_s22);
        STR_128_S(curr_out, v_out_stride, v_out4);
        // Output point 11: X(10)
        curr_out = out + out_strides[10];
        v_out10 = _mm_sub_ps(v_s21, v_s22);
        STR_128_S(curr_out, v_out_stride, v_out10);

        v_t25 = _mm_mul_ps(v128_CRTM_14_1, v_s14);
        v_t26 = _mm_mul_ps(v128_CRTM_14_3, v_s6);
        v_t27 = _mm_mul_ps(v128_CRTM_14_5, v_s10);
        v_t28 = _mm_mul_ps(v128_CRTM_14_2, v_s11);
        v_t29 = _mm_mul_ps(v128_CRTM_14_4, v_s3);
        v_t30 = _mm_mul_ps(v128_CRTM_14_6, v_s7);

        v_s43 = _mm_sub_ps(v_s1, v_t28);
        v_s44 = _mm_sub_ps(v_t29, v_t30);
        v_s45 = _mm_sub_ps(v_t27, v_t25);
        v_s23 = _mm_add_ps(v_s43, v_s44);
        v_s24 = _mm_sub_ps(v_s45, v_t26);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm_add_ps(v_s23, v_s24);
        STR_128_S(curr_out, v_out_stride, v_out5);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm_sub_ps(v_s23, v_s24);
        STR_128_S(curr_out, v_out_stride, v_out9);

        v_t31 = _mm_mul_ps(v128_CRTM_14_1, v_s5);
        v_t32 = _mm_mul_ps(v128_CRTM_14_3, v_s9);
        v_t33 = _mm_mul_ps(v128_CRTM_14_5, v_s13);
        v_t34 = _mm_mul_ps(v128_CRTM_14_2, v_s4);
        v_t35 = _mm_mul_ps(v128_CRTM_14_4, v_s8);
        v_t36 = _mm_mul_ps(v128_CRTM_14_6, v_s12);

        v_s46 = _mm_sub_ps(v_s2, v_t34);
        v_s47 = _mm_sub_ps(v_t35, v_t36);
        v_s48 = _mm_sub_ps(v_t32, v_t31);
        v_s25 = _mm_add_ps(v_s46, v_s47);
        v_s26 = _mm_sub_ps(v_s48, v_t33);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm_add_ps(v_s25, v_s26);
        STR_128_S(curr_out, v_out_stride, v_out6);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm_sub_ps(v_s25, v_s26);
        STR_128_S(curr_out, v_out_stride, v_out8);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (n & 2)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9, v_in10, v_in11, v_in12, v_in13;
        __m128 v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9, v_s10,
               v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18, v_s19,
               v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27, v_s28,
               v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36, v_s37,
               v_s38, v_s39, v_s40, v_s41, v_s42, v_s43, v_s44, v_s45, v_s46,
               v_s47, v_s48;
        __m128 v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9, v_t10,
               v_t11, v_t12, v_t13, v_t14, v_t15, v_t16, v_t17, v_t18, v_t19,
               v_t20, v_t21, v_t22, v_t23, v_t24, v_t25, v_t26, v_t27, v_t28,
               v_t29, v_t30, v_t31, v_t32, v_t33, v_t34, v_t35, v_t36, v_t37,
               v_t38;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_14_1 = _mm512_castps512_ps128(v_CRTM_14_1);
        __m128 v128_CRTM_14_2 = _mm512_castps512_ps128(v_CRTM_14_2);
        __m128 v128_CRTM_14_3 = _mm512_castps512_ps128(v_CRTM_14_3);
        __m128 v128_CRTM_14_4 = _mm512_castps512_ps128(v_CRTM_14_4);
        __m128 v128_CRTM_14_5 = _mm512_castps512_ps128(v_CRTM_14_5);
        __m128 v128_CRTM_14_6 = _mm512_castps512_ps128(v_CRTM_14_6);
        __m128 v128_CRTM_14_7 = _mm512_castps512_ps128(v_CRTM_14_7);

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
        LDHR_128_S(curr_in, v_in_stride, v_in13);

        v_s1 = _mm_sub_ps(v_in0, v_in13);
        v_s2 = _mm_add_ps(v_in0, v_in13);
        v_s3 = _mm_sub_ps(v_in11, v_in1);
        v_s4 = _mm_add_ps(v_in1, v_in11);
        v_s5 = _mm_sub_ps(v_in2, v_in12);
        v_s6 = _mm_add_ps(v_in2, v_in12);
        v_s7 = _mm_sub_ps(v_in3, v_in9);
        v_s8 = _mm_add_ps(v_in3, v_in9);
        v_s9 = _mm_sub_ps(v_in4, v_in10);
        v_s10 = _mm_add_ps(v_in4, v_in10);
        v_s11 = _mm_sub_ps(v_in7, v_in5);
        v_s12 = _mm_add_ps(v_in5, v_in7);
        v_s13 = _mm_sub_ps(v_in6, v_in8);
        v_s14 = _mm_add_ps(v_in6, v_in8);

        v_s27 = _mm_add_ps(v_s12, v_s4);
        v_s28 = _mm_add_ps(v_s27, v_s8);
        v_t37 = _mm_mul_ps(v128_CRTM_14_7, v_s28);
        v_s29 = _mm_add_ps(v_s3, v_s7);
        v_s30 = _mm_add_ps(v_s29, v_s11);
        v_t38 = _mm_mul_ps(v128_CRTM_14_7, v_s30);
        // Output pt 1: X(0)
        v_out0 = _mm_add_ps(v_t37, v_s2);
        STHR_128_S(curr_out, v_out_stride, v_out0);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = _mm_add_ps(v_t38, v_s1);
        STHR_128_S(curr_out, v_out_stride, v_out7);

        v_t1 = _mm_mul_ps(v128_CRTM_14_1, v_s6);
        v_t2 = _mm_mul_ps(v128_CRTM_14_3, v_s10);
        v_t3 = _mm_mul_ps(v128_CRTM_14_5, v_s14);
        v_t4 = _mm_mul_ps(v128_CRTM_14_2, v_s3);
        v_t5 = _mm_mul_ps(v128_CRTM_14_4, v_s7);
        v_t6 = _mm_mul_ps(v128_CRTM_14_6, v_s11);

        v_s31 = _mm_sub_ps(v_t5, v_t6);
        v_s32 = _mm_sub_ps(v_s1, v_t4);
        v_s33 = _mm_add_ps(v_t1, v_t2);

        v_s15 = _mm_add_ps(v_s31, v_s32);
        v_s16 = _mm_add_ps(v_s33, v_t3);
        // Output point 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm_sub_ps(v_s15, v_s16);
        STHR_128_S(curr_out, v_out_stride, v_out1);
        // Output point 14: X(13)
        curr_out = out + out_strides[13];
        v_out13 = _mm_add_ps(v_s15, v_s16);
        STHR_128_S(curr_out, v_out_stride, v_out13);

        v_t7 = _mm_mul_ps(v128_CRTM_14_1, v_s13);
        v_t8 = _mm_mul_ps(v128_CRTM_14_3, v_s5);
        v_t9 = _mm_mul_ps(v128_CRTM_14_5, v_s9);

        v_t10 = _mm_mul_ps(v128_CRTM_14_2, v_s12);
        v_t11 = _mm_mul_ps(v128_CRTM_14_4, v_s4);
        v_t12 = _mm_mul_ps(v128_CRTM_14_6, v_s8);

        v_s34 = _mm_sub_ps(v_s2, v_t10);
        v_s35 = _mm_sub_ps(v_t11, v_t12);
        v_s36 = _mm_add_ps(v_t7, v_t8);

        v_s17 = v_s34 + v_s35;
        v_s18 = _mm_add_ps(v_s36, v_t9);
        // Output point 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm_sub_ps(v_s17, v_s18);
        STHR_128_S(curr_out, v_out_stride, v_out2);
        // Output point 13: X(12)
        curr_out = out + out_strides[12];
        v_out12 = _mm_add_ps(v_s17, v_s18);
        STHR_128_S(curr_out, v_out_stride, v_out12);

        v_t13 = _mm_mul_ps(v128_CRTM_14_1, v_s10);
        v_t14 = _mm_mul_ps(v128_CRTM_14_3, v_s14);
        v_t15 = _mm_mul_ps(v128_CRTM_14_5, v_s6);
        v_t16 = _mm_mul_ps(v128_CRTM_14_2, v_s7);
        v_t17 = _mm_mul_ps(v128_CRTM_14_4, v_s11);
        v_t18 = _mm_mul_ps(v128_CRTM_14_6, v_s3);

        v_s37 = _mm_sub_ps(v_s1, v_t16);
        v_s38 = _mm_sub_ps(v_t17, v_t18);
        v_s39 = _mm_sub_ps(v_t14, v_t15);
        v_s19 = _mm_add_ps(v_s37, v_s38);
        v_s20 = _mm_sub_ps(v_s39, v_t13);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm_add_ps(v_s19, v_s20);
        STHR_128_S(curr_out, v_out_stride, v_out3);
        // Output point 12: X(11)
        curr_out = out + out_strides[11];
        v_out11 = _mm_sub_ps(v_s19, v_s20);
        STHR_128_S(curr_out, v_out_stride, v_out11);

        v_t19 = _mm_mul_ps(v128_CRTM_14_1, v_s9);
        v_t20 = _mm_mul_ps(v128_CRTM_14_3, v_s13);
        v_t21 = _mm_mul_ps(v128_CRTM_14_5, v_s5);
        v_t22 = _mm_mul_ps(v128_CRTM_14_2, v_s8);
        v_t23 = _mm_mul_ps(v128_CRTM_14_4, v_s12);
        v_t24 = _mm_mul_ps(v128_CRTM_14_6, v_s4);

        v_s40 = _mm_sub_ps(v_s2, v_t22);
        v_s41 = _mm_sub_ps(v_t23, v_t24);
        v_s42 = _mm_add_ps(v_t19, v_t20);
        v_s21 = _mm_add_ps(v_s40, v_s41);
        v_s22 = _mm_sub_ps(v_s42, v_t21);
        // Output pt 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm_add_ps(v_s21, v_s22);
        STHR_128_S(curr_out, v_out_stride, v_out4);
        // Output point 11: X(10)
        curr_out = out + out_strides[10];
        v_out10 = _mm_sub_ps(v_s21, v_s22);
        STHR_128_S(curr_out, v_out_stride, v_out10);

        v_t25 = _mm_mul_ps(v128_CRTM_14_1, v_s14);
        v_t26 = _mm_mul_ps(v128_CRTM_14_3, v_s6);
        v_t27 = _mm_mul_ps(v128_CRTM_14_5, v_s10);
        v_t28 = _mm_mul_ps(v128_CRTM_14_2, v_s11);
        v_t29 = _mm_mul_ps(v128_CRTM_14_4, v_s3);
        v_t30 = _mm_mul_ps(v128_CRTM_14_6, v_s7);

        v_s43 = _mm_sub_ps(v_s1, v_t28);
        v_s44 = _mm_sub_ps(v_t29, v_t30);
        v_s45 = _mm_sub_ps(v_t27, v_t25);
        v_s23 = _mm_add_ps(v_s43, v_s44);
        v_s24 = _mm_sub_ps(v_s45, v_t26);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm_add_ps(v_s23, v_s24);
        STHR_128_S(curr_out, v_out_stride, v_out5);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm_sub_ps(v_s23, v_s24);
        STHR_128_S(curr_out, v_out_stride, v_out9);

        v_t31 = _mm_mul_ps(v128_CRTM_14_1, v_s5);
        v_t32 = _mm_mul_ps(v128_CRTM_14_3, v_s9);
        v_t33 = _mm_mul_ps(v128_CRTM_14_5, v_s13);
        v_t34 = _mm_mul_ps(v128_CRTM_14_2, v_s4);
        v_t35 = _mm_mul_ps(v128_CRTM_14_4, v_s8);
        v_t36 = _mm_mul_ps(v128_CRTM_14_6, v_s12);

        v_s46 = _mm_sub_ps(v_s2, v_t34);
        v_s47 = _mm_sub_ps(v_t35, v_t36);
        v_s48 = _mm_sub_ps(v_t32, v_t31);
        v_s25 = _mm_add_ps(v_s46, v_s47);
        v_s26 = _mm_sub_ps(v_s48, v_t33);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm_add_ps(v_s25, v_s26);
        STHR_128_S(curr_out, v_out_stride, v_out6);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm_sub_ps(v_s25, v_s26);
        STHR_128_S(curr_out, v_out_stride, v_out8);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        FLOAT in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11,
              in12, in13;
        FLOAT s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14,
              s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25;
        FLOAT t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14,
              t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27,
              t28, t29, t30, t31, t32, t33, t34, t35;

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

        s0 = in0 - in13;
        s1 = in0 + in13;
        s2 = in11 - in1;
        s3 = in1 + in11;
        s4 = in2 - in12;
        s5 = in2 + in12;
        s6 = in3 - in9;
        s7 = in3 + in9;
        s8 = in4 - in10;
        s9 = in4 + in10;
        s10 = in7 - in5;
        s11 = in5 + in7;
        s12 = in6 - in8;
        s13 = in6 + in8;

        *out = CRTM_14_7 * (s11 + s3 + s7) + s1;
        out[out_strides[7]] = CRTM_14_7 * (s2 + s6 + s10) + s0;

        t0 = CRTM_14_1 * s5;
        t1 = CRTM_14_3 * s9;
        t2 = CRTM_14_5 * s13;
        t3 = CRTM_14_2 * s2;
        t4 = CRTM_14_4 * s6;
        t5 = CRTM_14_6 * s10;

        s14 = t4 - t5 - t3 + s0;
        s15 = t0 + t1 + t2;
        out[out_strides[1]] = s14 - s15;
        out[out_strides[13]] = s14 + s15;

        t6 = CRTM_14_1 * s12;
        t7 = CRTM_14_3 * s4;
        t8 = CRTM_14_5 * s8;

        t9 = CRTM_14_2 * s11;
        t10 = CRTM_14_4 * s3;
        t11 = CRTM_14_6 * s7;

        s16 = t10 - t9 - t11 + s1;
        s17 = t6 + t7 + t8;
        out[out_strides[2]] = s16 - s17;
        out[out_strides[12]] = s16 + s17;

        t12 = CRTM_14_1 * s9;
        t13 = CRTM_14_3 * s13;
        t14 = CRTM_14_5 * s5;
        t15 = CRTM_14_2 * s6;
        t16 = CRTM_14_4 * s10;
        t17 = CRTM_14_6 * s2;

        s18 = t16 - t15 - t17 + s0;
        s19 = t13 - t12 - t14;
        out[out_strides[3]] = s18 + s19;
        out[out_strides[11]] = s18 - s19;

        t18 = CRTM_14_1 * s8;
        t19 = CRTM_14_3 * s12;
        t20 = CRTM_14_5 * s4;
        t21 = CRTM_14_2 * s7;
        t22 = CRTM_14_4 * s11;
        t23 = CRTM_14_6 * -s3;

        s20 = t22 + t23 - t21 + s1;
        s21 = t18 + t19 - t20;
        out[out_strides[4]] = s20 + s21;
        out[out_strides[10]] = s20 - s21;

        t24 = CRTM_14_1 * s13;
        t25 = CRTM_14_3 * s5;
        t26 = CRTM_14_5 * s9;
        t27 = CRTM_14_2 * s10;
        t28 = CRTM_14_4 * s2;
        t29 = CRTM_14_6 * s6;

        s22 = t28 - t27 - t29 + s0;
        s23 = t26 - t24 - t25;
        out[out_strides[5]] = s22 + s23;
        out[out_strides[9]] = s22 - s23;

        t30 = CRTM_14_1 * s4;
        t31 = CRTM_14_3 * s8;
        t32 = CRTM_14_5 * s12;
        t33 = CRTM_14_2 * s3;
        t34 = CRTM_14_4 * s7;
        t35 = CRTM_14_6 * s11;

        s24 = t34 - t33 - t35 + s1;
        s25 = t31 - t30 - t32;
        out[out_strides[6]] = s24 + s25;
        out[out_strides[8]] = s24 - s25;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID r2hc_rfft14avx512_fp64_fwd(VOID *in_real, VOID *in_imag,
                                       VOID *out_real, VOID *out_imag, INTP n,
                                       aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const DOUBLE CRTM_14_1 = 0.900968867902419126236102319507445051165919162;
    const DOUBLE CRTM_14_2 = 0.433883739117558120475768332848358754609990728;
    const DOUBLE CRTM_14_3 = 0.623489801858733530525004884004239810632274731;
    const DOUBLE CRTM_14_4 = 0.781831482468029808708444526674057750232334519;
    const DOUBLE CRTM_14_5 = 0.222520933956314404288902564496794759466355569;
    const DOUBLE CRTM_14_6 = 0.974927912181823607018131682993931217232785801;

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

    __m512d v_CRTM_14_1 = _mm512_set1_pd(CRTM_14_1);
    __m512d v_CRTM_14_2 = _mm512_set1_pd(CRTM_14_2);
    __m512d v_CRTM_14_3 = _mm512_set1_pd(CRTM_14_3);
    __m512d v_CRTM_14_4 = _mm512_set1_pd(CRTM_14_4);
    __m512d v_CRTM_14_5 = _mm512_set1_pd(CRTM_14_5);
    __m512d v_CRTM_14_6 = _mm512_set1_pd(CRTM_14_6);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m512d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9, v_in10, v_in11, v_in12, v_in13;
        __m512d v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9, v_s10,
               v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18, v_s19,
               v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27, v_s28,
               v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36, v_s37,
               v_s38, v_s39, v_s40, v_s41, v_s42, v_s43, v_s44, v_s45, v_s46,
               v_s47, v_s48;
        __m512d v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9, v_t10,
               v_t11, v_t12, v_t13, v_t14, v_t15, v_t16, v_t17, v_t18, v_t19,
               v_t20, v_t21, v_t22, v_t23, v_t24, v_t25, v_t26, v_t27, v_t28,
               v_t29, v_t30, v_t31, v_t32, v_t33, v_t34, v_t35, v_t36;
        __m512d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13;

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
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDR_512_D(curr_in, v_in_stride, v_in12);
        // Input point 14: x(13)
        curr_in = in + in_strides[13];
        LDR_512_D(curr_in, v_in_stride, v_in13);

        v_s1 = _mm512_sub_pd(v_in0, v_in7);
        v_s2 = _mm512_add_pd(v_in0, v_in7);
        v_s3 = _mm512_sub_pd(v_in13, v_in1);
        v_s4 = _mm512_add_pd(v_in13, v_in1);
        v_s5 = _mm512_sub_pd(v_in12, v_in2);
        v_s6 = _mm512_add_pd(v_in12, v_in2);
        v_s7 = _mm512_sub_pd(v_in11, v_in3);
        v_s8 = _mm512_add_pd(v_in11, v_in3);
        v_s9 = _mm512_sub_pd(v_in10, v_in4);
        v_s10 = _mm512_add_pd(v_in10, v_in4);
        v_s11 = _mm512_sub_pd(v_in9, v_in5);
        v_s12 = _mm512_add_pd(v_in9, v_in5);
        v_s13 = _mm512_sub_pd(v_in8, v_in6);
        v_s14 = _mm512_add_pd(v_in8, v_in6);

        v_s15 = _mm512_add_pd(v_s4, v_s14);
        v_s16 = _mm512_add_pd(v_s6, v_s12);
        v_s17 = _mm512_add_pd(v_s8, v_s10);

        v_s18 = _mm512_sub_pd(v_s14, v_s4);
        v_s19 = _mm512_sub_pd(v_s6, v_s12);
        v_s20 = _mm512_sub_pd(v_s10, v_s8);
        v_s27 = _mm512_add_pd(v_s2, v_s15);
        v_s28 = _mm512_add_pd(v_s16, v_s17);
        v_s29 = _mm512_add_pd(v_s1, v_s18);
        v_s30 = _mm512_add_pd(v_s19, v_s20);
        // Output pt 1: X(0)
        v_out0 = _mm512_add_pd(v_s27, v_s28);
        STR_512_D(curr_out, v_out_stride, v_out0);
        // Output pt 14: X(13)
        v_out13 = _mm512_add_pd(v_s29, v_s30);
        curr_out = out + out_strides[13];
        STR_512_D(curr_out, v_out_stride, v_out13);

        v_t1 = _mm512_mul_pd(v_CRTM_14_1, v_s18);
        v_t2 = _mm512_mul_pd(v_CRTM_14_3, v_s19);
        v_t3 = _mm512_mul_pd(v_CRTM_14_5, v_s20);
        v_s31 = _mm512_sub_pd(v_s1, v_t1);
        v_s32 = _mm512_sub_pd(v_t2, v_t3);
        // Output point 2: X(1)
        v_out1 = _mm512_add_pd(v_s31, v_s32);

        v_s21 = _mm512_add_pd(v_s3, v_s13);
        v_s22 = _mm512_add_pd(v_s5, v_s11);
        v_s23 = _mm512_add_pd(v_s7, v_s9);

        v_t4 = _mm512_mul_pd(v_CRTM_14_2, v_s21);
        v_t5 = _mm512_mul_pd(v_CRTM_14_4, v_s22);
        v_t6 = _mm512_mul_pd(v_CRTM_14_6, v_s23);
        v_s33 = _mm512_add_pd(v_t4, v_t5);
        // Output point 3: X(2)
        v_out2 = _mm512_add_pd(v_s33, v_t6);
        curr_out = out + out_strides[1];
        STRI_2x512_D(curr_out, v_out_stride, v_out1, v_out2);

        v_t7 = _mm512_mul_pd(v_CRTM_14_1, v_s17);
        v_t8 = _mm512_mul_pd(v_CRTM_14_3, v_s15);
        v_t9 = _mm512_mul_pd(v_CRTM_14_5, v_s16);
        v_s34 = _mm512_sub_pd(v_s2, v_t7);
        v_s35 = _mm512_sub_pd(v_t8, v_t9);
        // Output point 4: X(3)
        v_out3 = _mm512_add_pd(v_s34, v_s35);

        v_s24 = _mm512_sub_pd(v_s3, v_s13);
        v_s25 = _mm512_sub_pd(v_s5, v_s11);
        v_s26 = _mm512_sub_pd(v_s7, v_s9);

        v_t10 = _mm512_mul_pd(v_CRTM_14_2, v_s26);
        v_t11 = _mm512_mul_pd(v_CRTM_14_4, v_s24);
        v_t12 = _mm512_mul_pd(v_CRTM_14_6, v_s25);
        v_s36 = _mm512_add_pd(v_t10, v_t11);
        // Output point 5: X(4)
        v_out4 = _mm512_add_pd(v_s36, v_t12);
        curr_out = out + out_strides[3];
        STRI_2x512_D(curr_out, v_out_stride, v_out3, v_out4);

        v_t13 = _mm512_mul_pd(v_CRTM_14_1, v_s19);
        v_t14 = _mm512_mul_pd(v_CRTM_14_3, v_s20);
        v_t15 = _mm512_mul_pd(v_CRTM_14_5, v_s18);
        v_s37 = _mm512_sub_pd(v_s1, v_t13);
        v_s38 = _mm512_sub_pd(v_t14, v_t15);
        // Output point 6: X(5)
        v_out5 = _mm512_add_pd(v_s37, v_s38);

        v_t16 = _mm512_mul_pd(v_CRTM_14_2, v_s22);
        v_t17 = _mm512_mul_pd(v_CRTM_14_4, v_s23);
        v_t18 = _mm512_mul_pd(v_CRTM_14_6, v_s21);
        v_s39 = _mm512_sub_pd(v_t16, v_t17);
        // Output point 7: X(6)
        v_out6 = _mm512_add_pd(v_s39, v_t18);
        curr_out = out + out_strides[5];
        STRI_2x512_D(curr_out, v_out_stride, v_out5, v_out6);

        v_t19 = _mm512_mul_pd(v_CRTM_14_1, v_s16);
        v_t20 = _mm512_mul_pd(v_CRTM_14_3, v_s17);
        v_t21 = _mm512_mul_pd(v_CRTM_14_5, v_s15);
        v_s40 = _mm512_sub_pd(v_s2, v_t19);
        v_s41 = _mm512_sub_pd(v_t20, v_t21);
        // Output point 8: X(7)
        v_out7 = _mm512_add_pd(v_s40, v_s41);

        v_t22 = _mm512_mul_pd(v_CRTM_14_2, v_s25);
        v_t23 = _mm512_mul_pd(v_CRTM_14_4, v_s26);
        v_t24 = _mm512_mul_pd(v_CRTM_14_6, v_s24);
        v_s42 = _mm512_sub_pd(v_t24, v_t22);
        // Output point 9: X(8)
        v_out8 = _mm512_sub_pd(v_s42, v_t23);
        curr_out = out + out_strides[7];
        STRI_2x512_D(curr_out, v_out_stride, v_out7, v_out8);

        v_t25 = _mm512_mul_pd(v_CRTM_14_1, v_s20);
        v_t26 = _mm512_mul_pd(v_CRTM_14_3, v_s18);
        v_t27 = _mm512_mul_pd(v_CRTM_14_5, v_s19);
        v_s43 = _mm512_sub_pd(v_s1, v_t25);
        v_s44 = _mm512_sub_pd(v_t26, v_t27);
        // Output point 10: X(9)
        v_out9 = _mm512_add_pd(v_s43, v_s44);

        v_t28 = _mm512_mul_pd(v_CRTM_14_2, v_s23);
        v_t29 = _mm512_mul_pd(v_CRTM_14_4, v_s21);
        v_t30 = _mm512_mul_pd(v_CRTM_14_6, v_s22);
        v_s45 = _mm512_add_pd(v_t28, v_t29);
        // Output point 11: X(10)
        v_out10 = _mm512_sub_pd(v_s45, v_t30);
        curr_out = out + out_strides[9];
        STRI_2x512_D(curr_out, v_out_stride, v_out9, v_out10);

        v_t31 = _mm512_mul_pd(v_CRTM_14_1, v_s15);
        v_t32 = _mm512_mul_pd(v_CRTM_14_3, v_s16);
        v_t33 = _mm512_mul_pd(v_CRTM_14_5, v_s17);
        v_s46 = _mm512_sub_pd(v_s2, v_t31);
        v_s47 = _mm512_sub_pd(v_t32, v_t33);
        // Output point 12: X(11)
        v_out11 = _mm512_add_pd(v_s46, v_s47);

        v_t34 = _mm512_mul_pd(v_CRTM_14_2, v_s24);
        v_t35 = _mm512_mul_pd(v_CRTM_14_4, v_s25);
        v_t36 = _mm512_mul_pd(v_CRTM_14_6, v_s26);
        v_s48 = _mm512_sub_pd(v_t34, v_t35);
        // Output point 13: X(12)
        v_out12 = _mm512_add_pd(v_s48, v_t36);
        curr_out = out + out_strides[11];
        STRI_2x512_D(curr_out, v_out_stride, v_out11, v_out12);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (n & 4)
    {
        __m256d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9, v_in10, v_in11, v_in12, v_in13;
        __m256d v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9, v_s10,
               v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18, v_s19,
               v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27, v_s28,
               v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36, v_s37,
               v_s38, v_s39, v_s40, v_s41, v_s42, v_s43, v_s44, v_s45, v_s46,
               v_s47, v_s48;
        __m256d v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9, v_t10,
               v_t11, v_t12, v_t13, v_t14, v_t15, v_t16, v_t17, v_t18, v_t19,
               v_t20, v_t21, v_t22, v_t23, v_t24, v_t25, v_t26, v_t27, v_t28,
               v_t29, v_t30, v_t31, v_t32, v_t33, v_t34, v_t35, v_t36;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13;

        curr_in = in;
        curr_out = out;

        __m256d v256_CRTM_14_1 = _mm512_castpd512_pd256(v_CRTM_14_1);
        __m256d v256_CRTM_14_2 = _mm512_castpd512_pd256(v_CRTM_14_2);
        __m256d v256_CRTM_14_3 = _mm512_castpd512_pd256(v_CRTM_14_3);
        __m256d v256_CRTM_14_4 = _mm512_castpd512_pd256(v_CRTM_14_4);
        __m256d v256_CRTM_14_5 = _mm512_castpd512_pd256(v_CRTM_14_5);
        __m256d v256_CRTM_14_6 = _mm512_castpd512_pd256(v_CRTM_14_6);

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
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDR_256_D(curr_in, v_in_stride, v_in12);
        // Input point 14: x(13)
        curr_in = in + in_strides[13];
        LDR_256_D(curr_in, v_in_stride, v_in13);

        v_s1 = _mm256_sub_pd(v_in0, v_in7);
        v_s2 = _mm256_add_pd(v_in0, v_in7);
        v_s3 = _mm256_sub_pd(v_in13, v_in1);
        v_s4 = _mm256_add_pd(v_in13, v_in1);
        v_s5 = _mm256_sub_pd(v_in12, v_in2);
        v_s6 = _mm256_add_pd(v_in12, v_in2);
        v_s7 = _mm256_sub_pd(v_in11, v_in3);
        v_s8 = _mm256_add_pd(v_in11, v_in3);
        v_s9 = _mm256_sub_pd(v_in10, v_in4);
        v_s10 = _mm256_add_pd(v_in10, v_in4);
        v_s11 = _mm256_sub_pd(v_in9, v_in5);
        v_s12 = _mm256_add_pd(v_in9, v_in5);
        v_s13 = _mm256_sub_pd(v_in8, v_in6);
        v_s14 = _mm256_add_pd(v_in8, v_in6);

        v_s15 = _mm256_add_pd(v_s4, v_s14);
        v_s16 = _mm256_add_pd(v_s6, v_s12);
        v_s17 = _mm256_add_pd(v_s8, v_s10);

        v_s18 = _mm256_sub_pd(v_s14, v_s4);
        v_s19 = _mm256_sub_pd(v_s6, v_s12);
        v_s20 = _mm256_sub_pd(v_s10, v_s8);
        v_s27 = _mm256_add_pd(v_s2, v_s15);
        v_s28 = _mm256_add_pd(v_s16, v_s17);
        v_s29 = _mm256_add_pd(v_s1, v_s18);
        v_s30 = _mm256_add_pd(v_s19, v_s20);
        // Output pt 1: X(0)
        v_out0 = _mm256_add_pd(v_s27, v_s28);
        STR_256_D(curr_out, v_out_stride, v_out0);
        // Output pt 14: X(13)
        v_out13 = _mm256_add_pd(v_s29, v_s30);
        curr_out = out + out_strides[13];
        STR_256_D(curr_out, v_out_stride, v_out13);

        v_t1 = _mm256_mul_pd(v256_CRTM_14_1, v_s18);
        v_t2 = _mm256_mul_pd(v256_CRTM_14_3, v_s19);
        v_t3 = _mm256_mul_pd(v256_CRTM_14_5, v_s20);
        v_s31 = _mm256_sub_pd(v_s1, v_t1);
        v_s32 = _mm256_sub_pd(v_t2, v_t3);
        // Output point 2: X(1)
        v_out1 = _mm256_add_pd(v_s31, v_s32);

        v_s21 = _mm256_add_pd(v_s3, v_s13);
        v_s22 = _mm256_add_pd(v_s5, v_s11);
        v_s23 = _mm256_add_pd(v_s7, v_s9);

        v_t4 = _mm256_mul_pd(v256_CRTM_14_2, v_s21);
        v_t5 = _mm256_mul_pd(v256_CRTM_14_4, v_s22);
        v_t6 = _mm256_mul_pd(v256_CRTM_14_6, v_s23);
        v_s33 = _mm256_add_pd(v_t4, v_t5);
        // Output point 3: X(2)
        v_out2 = _mm256_add_pd(v_s33, v_t6);
        curr_out = out + out_strides[1];
        STRI_2x256_D(curr_out, v_out_stride, v_out1, v_out2);

        v_t7 = _mm256_mul_pd(v256_CRTM_14_1, v_s17);
        v_t8 = _mm256_mul_pd(v256_CRTM_14_3, v_s15);
        v_t9 = _mm256_mul_pd(v256_CRTM_14_5, v_s16);
        v_s34 = _mm256_sub_pd(v_s2, v_t7);
        v_s35 = _mm256_sub_pd(v_t8, v_t9);
        // Output point 4: X(3)
        v_out3 = _mm256_add_pd(v_s34, v_s35);

        v_s24 = _mm256_sub_pd(v_s3, v_s13);
        v_s25 = _mm256_sub_pd(v_s5, v_s11);
        v_s26 = _mm256_sub_pd(v_s7, v_s9);

        v_t10 = _mm256_mul_pd(v256_CRTM_14_2, v_s26);
        v_t11 = _mm256_mul_pd(v256_CRTM_14_4, v_s24);
        v_t12 = _mm256_mul_pd(v256_CRTM_14_6, v_s25);
        v_s36 = _mm256_add_pd(v_t10, v_t11);
        // Output point 5: X(4)
        v_out4 = _mm256_add_pd(v_s36, v_t12);
        curr_out = out + out_strides[3];
        STRI_2x256_D(curr_out, v_out_stride, v_out3, v_out4);

        v_t13 = _mm256_mul_pd(v256_CRTM_14_1, v_s19);
        v_t14 = _mm256_mul_pd(v256_CRTM_14_3, v_s20);
        v_t15 = _mm256_mul_pd(v256_CRTM_14_5, v_s18);
        v_s37 = _mm256_sub_pd(v_s1, v_t13);
        v_s38 = _mm256_sub_pd(v_t14, v_t15);
        // Output point 6: X(5)
        v_out5 = _mm256_add_pd(v_s37, v_s38);

        v_t16 = _mm256_mul_pd(v256_CRTM_14_2, v_s22);
        v_t17 = _mm256_mul_pd(v256_CRTM_14_4, v_s23);
        v_t18 = _mm256_mul_pd(v256_CRTM_14_6, v_s21);
        v_s39 = _mm256_sub_pd(v_t16, v_t17);
        // Output point 7: X(6)
        v_out6 = _mm256_add_pd(v_s39, v_t18);
        curr_out = out + out_strides[5];
        STRI_2x256_D(curr_out, v_out_stride, v_out5, v_out6);

        v_t19 = _mm256_mul_pd(v256_CRTM_14_1, v_s16);
        v_t20 = _mm256_mul_pd(v256_CRTM_14_3, v_s17);
        v_t21 = _mm256_mul_pd(v256_CRTM_14_5, v_s15);
        v_s40 = _mm256_sub_pd(v_s2, v_t19);
        v_s41 = _mm256_sub_pd(v_t20, v_t21);
        // Output point 8: X(7)
        v_out7 = _mm256_add_pd(v_s40, v_s41);

        v_t22 = _mm256_mul_pd(v256_CRTM_14_2, v_s25);
        v_t23 = _mm256_mul_pd(v256_CRTM_14_4, v_s26);
        v_t24 = _mm256_mul_pd(v256_CRTM_14_6, v_s24);
        v_s42 = _mm256_sub_pd(v_t24, v_t22);
        // Output point 9: X(8)
        v_out8 = _mm256_sub_pd(v_s42, v_t23);
        curr_out = out + out_strides[7];
        STRI_2x256_D(curr_out, v_out_stride, v_out7, v_out8);

        v_t25 = _mm256_mul_pd(v256_CRTM_14_1, v_s20);
        v_t26 = _mm256_mul_pd(v256_CRTM_14_3, v_s18);
        v_t27 = _mm256_mul_pd(v256_CRTM_14_5, v_s19);
        v_s43 = _mm256_sub_pd(v_s1, v_t25);
        v_s44 = _mm256_sub_pd(v_t26, v_t27);
        // Output point 10: X(9)
        v_out9 = _mm256_add_pd(v_s43, v_s44);

        v_t28 = _mm256_mul_pd(v256_CRTM_14_2, v_s23);
        v_t29 = _mm256_mul_pd(v256_CRTM_14_4, v_s21);
        v_t30 = _mm256_mul_pd(v256_CRTM_14_6, v_s22);
        v_s45 = _mm256_add_pd(v_t28, v_t29);
        // Output point 11: X(10)
        v_out10 = _mm256_sub_pd(v_s45, v_t30);
        curr_out = out + out_strides[9];
        STRI_2x256_D(curr_out, v_out_stride, v_out9, v_out10);

        v_t31 = _mm256_mul_pd(v256_CRTM_14_1, v_s15);
        v_t32 = _mm256_mul_pd(v256_CRTM_14_3, v_s16);
        v_t33 = _mm256_mul_pd(v256_CRTM_14_5, v_s17);
        v_s46 = _mm256_sub_pd(v_s2, v_t31);
        v_s47 = _mm256_sub_pd(v_t32, v_t33);
        // Output point 12: X(11)
        v_out11 = _mm256_add_pd(v_s46, v_s47);

        v_t34 = _mm256_mul_pd(v256_CRTM_14_2, v_s24);
        v_t35 = _mm256_mul_pd(v256_CRTM_14_4, v_s25);
        v_t36 = _mm256_mul_pd(v256_CRTM_14_6, v_s26);
        v_s48 = _mm256_sub_pd(v_t34, v_t35);
        // Output point 13: X(12)
        v_out12 = _mm256_add_pd(v_s48, v_t36);
        curr_out = out + out_strides[11];
        STRI_2x256_D(curr_out, v_out_stride, v_out11, v_out12);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (n & 2)
    {
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
               v_in9, v_in10, v_in11, v_in12, v_in13;
        __m128d v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9, v_s10,
               v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18, v_s19,
               v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27, v_s28,
               v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36, v_s37,
               v_s38, v_s39, v_s40, v_s41, v_s42, v_s43, v_s44, v_s45, v_s46,
               v_s47, v_s48;
        __m128d v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9, v_t10,
               v_t11, v_t12, v_t13, v_t14, v_t15, v_t16, v_t17, v_t18, v_t19,
               v_t20, v_t21, v_t22, v_t23, v_t24, v_t25, v_t26, v_t27, v_t28,
               v_t29, v_t30, v_t31, v_t32, v_t33, v_t34, v_t35, v_t36;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13;

        curr_in = in;
        curr_out = out;

        __m128d v128_CRTM_14_1 = _mm512_castpd512_pd128(v_CRTM_14_1);
        __m128d v128_CRTM_14_2 = _mm512_castpd512_pd128(v_CRTM_14_2);
        __m128d v128_CRTM_14_3 = _mm512_castpd512_pd128(v_CRTM_14_3);
        __m128d v128_CRTM_14_4 = _mm512_castpd512_pd128(v_CRTM_14_4);
        __m128d v128_CRTM_14_5 = _mm512_castpd512_pd128(v_CRTM_14_5);
        __m128d v128_CRTM_14_6 = _mm512_castpd512_pd128(v_CRTM_14_6);

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
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDR_128_D(curr_in, v_in_stride, v_in12);
        // Input point 14: x(13)
        curr_in = in + in_strides[13];
        LDR_128_D(curr_in, v_in_stride, v_in13);

        v_s1 = _mm_sub_pd(v_in0, v_in7);
        v_s2 = _mm_add_pd(v_in0, v_in7);
        v_s3 = _mm_sub_pd(v_in13, v_in1);
        v_s4 = _mm_add_pd(v_in13, v_in1);
        v_s5 = _mm_sub_pd(v_in12, v_in2);
        v_s6 = _mm_add_pd(v_in12, v_in2);
        v_s7 = _mm_sub_pd(v_in11, v_in3);
        v_s8 = _mm_add_pd(v_in11, v_in3);
        v_s9 = _mm_sub_pd(v_in10, v_in4);
        v_s10 = _mm_add_pd(v_in10, v_in4);
        v_s11 = _mm_sub_pd(v_in9, v_in5);
        v_s12 = _mm_add_pd(v_in9, v_in5);
        v_s13 = _mm_sub_pd(v_in8, v_in6);
        v_s14 = _mm_add_pd(v_in8, v_in6);

        v_s15 = _mm_add_pd(v_s4, v_s14);
        v_s16 = _mm_add_pd(v_s6, v_s12);
        v_s17 = _mm_add_pd(v_s8, v_s10);

        v_s18 = _mm_sub_pd(v_s14, v_s4);
        v_s19 = _mm_sub_pd(v_s6, v_s12);
        v_s20 = _mm_sub_pd(v_s10, v_s8);
        v_s27 = _mm_add_pd(v_s2, v_s15);
        v_s28 = _mm_add_pd(v_s16, v_s17);
        v_s29 = _mm_add_pd(v_s1, v_s18);
        v_s30 = _mm_add_pd(v_s19, v_s20);
        // Output pt 1: X(0)
        v_out0 = _mm_add_pd(v_s27, v_s28);
        STR_128_D(curr_out, v_out_stride, v_out0);
        // Output pt 14: X(13)
        v_out13 = _mm_add_pd(v_s29, v_s30);
        curr_out = out + out_strides[13];
        STR_128_D(curr_out, v_out_stride, v_out13);

        v_t1 = _mm_mul_pd(v128_CRTM_14_1, v_s18);
        v_t2 = _mm_mul_pd(v128_CRTM_14_3, v_s19);
        v_t3 = _mm_mul_pd(v128_CRTM_14_5, v_s20);
        v_s31 = _mm_sub_pd(v_s1, v_t1);
        v_s32 = _mm_sub_pd(v_t2, v_t3);
        // Output point 2: X(1)
        v_out1 = _mm_add_pd(v_s31, v_s32);

        v_s21 = _mm_add_pd(v_s3, v_s13);
        v_s22 = _mm_add_pd(v_s5, v_s11);
        v_s23 = _mm_add_pd(v_s7, v_s9);

        v_t4 = _mm_mul_pd(v128_CRTM_14_2, v_s21);
        v_t5 = _mm_mul_pd(v128_CRTM_14_4, v_s22);
        v_t6 = _mm_mul_pd(v128_CRTM_14_6, v_s23);
        v_s33 = _mm_add_pd(v_t4, v_t5);
        // Output point 3: X(2)
        v_out2 = _mm_add_pd(v_s33, v_t6);
        curr_out = out + out_strides[1];
        STRI_2x128_D(curr_out, v_out_stride, v_out1, v_out2);

        v_t7 = _mm_mul_pd(v128_CRTM_14_1, v_s17);
        v_t8 = _mm_mul_pd(v128_CRTM_14_3, v_s15);
        v_t9 = _mm_mul_pd(v128_CRTM_14_5, v_s16);
        v_s34 = _mm_sub_pd(v_s2, v_t7);
        v_s35 = _mm_sub_pd(v_t8, v_t9);
        // Output point 4: X(3)
        v_out3 = _mm_add_pd(v_s34, v_s35);

        v_s24 = _mm_sub_pd(v_s3, v_s13);
        v_s25 = _mm_sub_pd(v_s5, v_s11);
        v_s26 = _mm_sub_pd(v_s7, v_s9);

        v_t10 = _mm_mul_pd(v128_CRTM_14_2, v_s26);
        v_t11 = _mm_mul_pd(v128_CRTM_14_4, v_s24);
        v_t12 = _mm_mul_pd(v128_CRTM_14_6, v_s25);
        v_s36 = _mm_add_pd(v_t10, v_t11);
        // Output point 5: X(4)
        v_out4 = _mm_add_pd(v_s36, v_t12);
        curr_out = out + out_strides[3];
        STRI_2x128_D(curr_out, v_out_stride, v_out3, v_out4);

        v_t13 = _mm_mul_pd(v128_CRTM_14_1, v_s19);
        v_t14 = _mm_mul_pd(v128_CRTM_14_3, v_s20);
        v_t15 = _mm_mul_pd(v128_CRTM_14_5, v_s18);
        v_s37 = _mm_sub_pd(v_s1, v_t13);
        v_s38 = _mm_sub_pd(v_t14, v_t15);
        // Output point 6: X(5)
        v_out5 = _mm_add_pd(v_s37, v_s38);

        v_t16 = _mm_mul_pd(v128_CRTM_14_2, v_s22);
        v_t17 = _mm_mul_pd(v128_CRTM_14_4, v_s23);
        v_t18 = _mm_mul_pd(v128_CRTM_14_6, v_s21);
        v_s39 = _mm_sub_pd(v_t16, v_t17);
        // Output point 7: X(6)
        v_out6 = _mm_add_pd(v_s39, v_t18);
        curr_out = out + out_strides[5];
        STRI_2x128_D(curr_out, v_out_stride, v_out5, v_out6);

        v_t19 = _mm_mul_pd(v128_CRTM_14_1, v_s16);
        v_t20 = _mm_mul_pd(v128_CRTM_14_3, v_s17);
        v_t21 = _mm_mul_pd(v128_CRTM_14_5, v_s15);
        v_s40 = _mm_sub_pd(v_s2, v_t19);
        v_s41 = _mm_sub_pd(v_t20, v_t21);
        // Output point 8: X(7)
        v_out7 = _mm_add_pd(v_s40, v_s41);

        v_t22 = _mm_mul_pd(v128_CRTM_14_2, v_s25);
        v_t23 = _mm_mul_pd(v128_CRTM_14_4, v_s26);
        v_t24 = _mm_mul_pd(v128_CRTM_14_6, v_s24);
        v_s42 = _mm_sub_pd(v_t24, v_t22);
        // Output point 9: X(8)
        v_out8 = _mm_sub_pd(v_s42, v_t23);
        curr_out = out + out_strides[7];
        STRI_2x128_D(curr_out, v_out_stride, v_out7, v_out8);

        v_t25 = _mm_mul_pd(v128_CRTM_14_1, v_s20);
        v_t26 = _mm_mul_pd(v128_CRTM_14_3, v_s18);
        v_t27 = _mm_mul_pd(v128_CRTM_14_5, v_s19);
        v_s43 = _mm_sub_pd(v_s1, v_t25);
        v_s44 = _mm_sub_pd(v_t26, v_t27);
        // Output point 10: X(9)
        v_out9 = _mm_add_pd(v_s43, v_s44);

        v_t28 = _mm_mul_pd(v128_CRTM_14_2, v_s23);
        v_t29 = _mm_mul_pd(v128_CRTM_14_4, v_s21);
        v_t30 = _mm_mul_pd(v128_CRTM_14_6, v_s22);
        v_s45 = _mm_add_pd(v_t28, v_t29);
        // Output point 11: X(10)
        v_out10 = _mm_sub_pd(v_s45, v_t30);
        curr_out = out + out_strides[9];
        STRI_2x128_D(curr_out, v_out_stride, v_out9, v_out10);

        v_t31 = _mm_mul_pd(v128_CRTM_14_1, v_s15);
        v_t32 = _mm_mul_pd(v128_CRTM_14_3, v_s16);
        v_t33 = _mm_mul_pd(v128_CRTM_14_5, v_s17);
        v_s46 = _mm_sub_pd(v_s2, v_t31);
        v_s47 = _mm_sub_pd(v_t32, v_t33);
        // Output point 12: X(11)
        v_out11 = _mm_add_pd(v_s46, v_s47);

        v_t34 = _mm_mul_pd(v128_CRTM_14_2, v_s24);
        v_t35 = _mm_mul_pd(v128_CRTM_14_4, v_s25);
        v_t36 = _mm_mul_pd(v128_CRTM_14_6, v_s26);
        v_s48 = _mm_sub_pd(v_t34, v_t35);
        // Output point 13: X(12)
        v_out12 = _mm_add_pd(v_s48, v_t36);
        curr_out = out + out_strides[11];
        STRI_2x128_D(curr_out, v_out_stride, v_out11, v_out12);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        DOUBLE in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11,
              in12, in13;
        DOUBLE s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14,
              s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25;
        DOUBLE t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14,
              t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27,
              t28, t29, t30, t31, t32, t33, t34, t35;

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

        s0 = in0 - in7;
        s1 = in0 + in7;
        s2 = in13 - in1;
        s3 = in13 + in1;
        s4 = in12 - in2;
        s5 = in12 + in2;
        s6 = in11 - in3;
        s7 = in11 + in3;
        s8 = in10 - in4;
        s9 = in10 + in4;
        s10 = in9 - in5;
        s11 = in9 + in5;
        s12 = in8 - in6;
        s13 = in8 + in6;

        s14 = s3 + s13;
        s15 = s5 + s11;
        s16 = s7 + s9;

        s17 = s13 - s3;
        s18 = s5 - s11;
        s19 = s9 - s7;

        // Output point 1: X(0)
        *out = s1 + s14 + s15 + s16;
        // Output point 13: X(14)
        out[out_strides[13]] = s0 + s17 + s18 + s19;

        t0 = CRTM_14_1 * s17;
        t1 = CRTM_14_3 * s18;
        t2 = CRTM_14_5 * s19;

        // Output point 2: X(1)
        out[out_strides[1]] = s0 - t0 + t1 - t2;

        s20 = s2 + s12;
        s21 = s4 + s10;
        s22 = s6 + s8;

        t3 = CRTM_14_2 * s20;
        t4 = CRTM_14_4 * s21;
        t5 = CRTM_14_6 * s22;

        // Output point 3: X(2)
        out[out_strides[2]] = t3 + t4 + t5;

        t6 = CRTM_14_1 * s16;
        t7 = CRTM_14_3 * s14;
        t8 = CRTM_14_5 * s15;

        // Output point 4: X(3)
        out[out_strides[3]] = s1 - t6 + t7 - t8;

        s23 = s2 - s12;
        s24 = s4 - s10;
        s25 = s6 - s8;

        t9 = CRTM_14_2 * s25;
        t10 = CRTM_14_4 * s23;
        t11 = CRTM_14_6 * s24;

        // Output point 5: X(4)
        out[out_strides[4]] = t9 + t10 + t11;

        t12 = CRTM_14_1 * s18;
        t13 = CRTM_14_3 * s19;
        t14 = CRTM_14_5 * s17;
        // Output point 6: X(5)
        out[out_strides[5]] = s0 - t12 + t13 - t14;
        t15 = CRTM_14_2 * s21;
        t16 = CRTM_14_4 * s22;
        t17 = CRTM_14_6 * s20;
        // Output point 7: X(6)
        out[out_strides[6]] = t15 - t16 + t17;

        t18 = CRTM_14_1 * s15;
        t19 = CRTM_14_3 * s16;
        t20 = CRTM_14_5 * s14;
        // Output point 8: X(7)
        out[out_strides[7]] = s1 - t18 + t19 - t20;
        t21 = CRTM_14_2 * s24;
        t22 = CRTM_14_4 * s25;
        t23 = CRTM_14_6 * s23;
        // Output point 9: X(8)
        out[out_strides[8]] = t23 - t22 - t21;

        t24 = CRTM_14_1 * s19;
        t25 = CRTM_14_3 * s17;
        t26 = CRTM_14_5 * s18;
        // Output point 10: X(9)
        out[out_strides[9]] = s0 - t24 + t25 - t26;
        t27 = CRTM_14_2 * s22;
        t28 = CRTM_14_4 * s20;
        t29 = CRTM_14_6 * s21;
        // Output point 11: X(10)
        out[out_strides[10]] = t27 + t28 - t29;

        t30 = CRTM_14_1 * s14;
        t31 = CRTM_14_3 * s15;
        t32 = CRTM_14_5 * s16;
        // Output point 5: X(4)
        out[out_strides[11]] = s1 - t30 + t31 - t32;
        t33 = CRTM_14_2 * s23;
        t34 = CRTM_14_4 * s24;
        t35 = CRTM_14_6 * s25;
        // Output point 5: X(4)
        out[out_strides[12]] = t33 - t34 + t35;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID r2hc_rfft14avx512_fp64_bwd(VOID *in_real, VOID *in_imag,
                                       VOID *out_real, VOID *out_imag, INTP n,
                                       aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const DOUBLE CRTM_14_1 = 0.867767478235116240951536665696717509219981456;
    const DOUBLE CRTM_14_2 = 1.801937735804838252472204639014890102331838324;
    const DOUBLE CRTM_14_3 = 1.563662964936059617416889053348115500464669038;
    const DOUBLE CRTM_14_4 = 1.246979603717467061050009768008479621264549462;
    const DOUBLE CRTM_14_5 = 1.949855824363647214036263365987862434465571602;
    const DOUBLE CRTM_14_6 = 0.445041867912628808577805128993589518932711138;
    const DOUBLE CRTM_14_7 = 2.000000000000000000000000000000000000000000000;

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

    __m512d v_CRTM_14_1 = _mm512_set1_pd(CRTM_14_1);
    __m512d v_CRTM_14_2 = _mm512_set1_pd(CRTM_14_2);
    __m512d v_CRTM_14_3 = _mm512_set1_pd(CRTM_14_3);
    __m512d v_CRTM_14_4 = _mm512_set1_pd(CRTM_14_4);
    __m512d v_CRTM_14_5 = _mm512_set1_pd(CRTM_14_5);
    __m512d v_CRTM_14_6 = _mm512_set1_pd(CRTM_14_6);
    __m512d v_CRTM_14_7 = _mm512_set1_pd(CRTM_14_7);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m512d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
                v_in9, v_in10, v_in11, v_in12, v_in13;
        __m512d v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9, v_s10,
                v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18, v_s19,
                v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27, v_s28,
                v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36, v_s37,
                v_s38, v_s39, v_s40, v_s41, v_s42, v_s43, v_s44, v_s45, v_s46,
                v_s47, v_s48;
        __m512d v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9, v_t10,
                v_t11, v_t12, v_t13, v_t14, v_t15, v_t16, v_t17, v_t18, v_t19,
                v_t20, v_t21, v_t22, v_t23, v_t24, v_t25, v_t26, v_t27, v_t28,
                v_t29, v_t30, v_t31, v_t32, v_t33, v_t34, v_t35, v_t36, v_t37,
                v_t38;
        __m512d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13;

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
        // Input point 12: x(11) & Input point 13: x(12)
        curr_in = in + in_strides[11];
        LDRI_2x512_D(curr_in, v_in_stride, v_in11, v_in12);
        // Input point 14: x(13) & Input point 15: x(14)
        curr_in = in + in_strides[13];
        LDR_512_D(curr_in, v_in_stride, v_in13);

        v_s1 = _mm512_sub_pd(v_in0, v_in13);
        v_s2 = _mm512_add_pd(v_in0, v_in13);
        v_s3 = _mm512_sub_pd(v_in11, v_in1);
        v_s4 = _mm512_add_pd(v_in1, v_in11);
        v_s5 = _mm512_sub_pd(v_in2, v_in12);
        v_s6 = _mm512_add_pd(v_in2, v_in12);
        v_s7 = _mm512_sub_pd(v_in3, v_in9);
        v_s8 = _mm512_add_pd(v_in3, v_in9);
        v_s9 = _mm512_sub_pd(v_in4, v_in10);
        v_s10 = _mm512_add_pd(v_in4, v_in10);
        v_s11 = _mm512_sub_pd(v_in7, v_in5);
        v_s12 = _mm512_add_pd(v_in5, v_in7);
        v_s13 = _mm512_sub_pd(v_in6, v_in8);
        v_s14 = _mm512_add_pd(v_in6, v_in8);

        v_s27 = _mm512_add_pd(v_s12, v_s4);
        v_s28 = _mm512_add_pd(v_s27, v_s8);
        v_t37 = _mm512_mul_pd(v_CRTM_14_7, v_s28);
        v_s29 = _mm512_add_pd(v_s3, v_s7);
        v_s30 = _mm512_add_pd(v_s29, v_s11);
        v_t38 = _mm512_mul_pd(v_CRTM_14_7, v_s30);
        // Output pt 1: X(0)
        v_out0 = _mm512_add_pd(v_t37, v_s2);
        STR_512_D(curr_out, v_out_stride, v_out0);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = _mm512_add_pd(v_t38, v_s1);
        STR_512_D(curr_out, v_out_stride, v_out7);

        v_t1 = _mm512_mul_pd(v_CRTM_14_1, v_s6);
        v_t2 = _mm512_mul_pd(v_CRTM_14_3, v_s10);
        v_t3 = _mm512_mul_pd(v_CRTM_14_5, v_s14);
        v_t4 = _mm512_mul_pd(v_CRTM_14_2, v_s3);
        v_t5 = _mm512_mul_pd(v_CRTM_14_4, v_s7);
        v_t6 = _mm512_mul_pd(v_CRTM_14_6, v_s11);

        v_s31 = _mm512_sub_pd(v_t5, v_t6);
        v_s32 = _mm512_sub_pd(v_s1, v_t4);
        v_s33 = _mm512_add_pd(v_t1, v_t2);

        v_s15 = _mm512_add_pd(v_s31, v_s32);
        v_s16 = _mm512_add_pd(v_s33, v_t3);
        // Output point 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm512_sub_pd(v_s15, v_s16);
        STR_512_D(curr_out, v_out_stride, v_out1);
        // Output point 14: X(13)
        curr_out = out + out_strides[13];
        v_out13 = _mm512_add_pd(v_s15, v_s16);
        STR_512_D(curr_out, v_out_stride, v_out13);

        v_t7 = _mm512_mul_pd(v_CRTM_14_1, v_s13);
        v_t8 = _mm512_mul_pd(v_CRTM_14_3, v_s5);
        v_t9 = _mm512_mul_pd(v_CRTM_14_5, v_s9);

        v_t10 = _mm512_mul_pd(v_CRTM_14_2, v_s12);
        v_t11 = _mm512_mul_pd(v_CRTM_14_4, v_s4);
        v_t12 = _mm512_mul_pd(v_CRTM_14_6, v_s8);

        v_s34 = _mm512_sub_pd(v_s2, v_t10);
        v_s35 = _mm512_sub_pd(v_t11, v_t12);
        v_s36 = _mm512_add_pd(v_t7, v_t8);

        v_s17 = v_s34 + v_s35;
        v_s18 = _mm512_add_pd(v_s36, v_t9);
        // Output point 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm512_sub_pd(v_s17, v_s18);
        STR_512_D(curr_out, v_out_stride, v_out2);
        // Output point 13: X(12)
        curr_out = out + out_strides[12];
        v_out12 = _mm512_add_pd(v_s17, v_s18);
        STR_512_D(curr_out, v_out_stride, v_out12);

        v_t13 = _mm512_mul_pd(v_CRTM_14_1, v_s10);
        v_t14 = _mm512_mul_pd(v_CRTM_14_3, v_s14);
        v_t15 = _mm512_mul_pd(v_CRTM_14_5, v_s6);
        v_t16 = _mm512_mul_pd(v_CRTM_14_2, v_s7);
        v_t17 = _mm512_mul_pd(v_CRTM_14_4, v_s11);
        v_t18 = _mm512_mul_pd(v_CRTM_14_6, v_s3);

        v_s37 = _mm512_sub_pd(v_s1, v_t16);
        v_s38 = _mm512_sub_pd(v_t17, v_t18);
        v_s39 = _mm512_sub_pd(v_t14, v_t15);
        v_s19 = _mm512_add_pd(v_s37, v_s38);
        v_s20 = _mm512_sub_pd(v_s39, v_t13);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm512_add_pd(v_s19, v_s20);
        STR_512_D(curr_out, v_out_stride, v_out3);
        // Output point 12: X(11)
        curr_out = out + out_strides[11];
        v_out11 = _mm512_sub_pd(v_s19, v_s20);
        STR_512_D(curr_out, v_out_stride, v_out11);

        v_t19 = _mm512_mul_pd(v_CRTM_14_1, v_s9);
        v_t20 = _mm512_mul_pd(v_CRTM_14_3, v_s13);
        v_t21 = _mm512_mul_pd(v_CRTM_14_5, v_s5);
        v_t22 = _mm512_mul_pd(v_CRTM_14_2, v_s8);
        v_t23 = _mm512_mul_pd(v_CRTM_14_4, v_s12);
        v_t24 = _mm512_mul_pd(v_CRTM_14_6, v_s4);

        v_s40 = _mm512_sub_pd(v_s2, v_t22);
        v_s41 = _mm512_sub_pd(v_t23, v_t24);
        v_s42 = _mm512_add_pd(v_t19, v_t20);
        v_s21 = _mm512_add_pd(v_s40, v_s41);
        v_s22 = _mm512_sub_pd(v_s42, v_t21);
        // Output pt 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm512_add_pd(v_s21, v_s22);
        STR_512_D(curr_out, v_out_stride, v_out4);
        // Output point 11: X(10)
        curr_out = out + out_strides[10];
        v_out10 = _mm512_sub_pd(v_s21, v_s22);
        STR_512_D(curr_out, v_out_stride, v_out10);

        v_t25 = _mm512_mul_pd(v_CRTM_14_1, v_s14);
        v_t26 = _mm512_mul_pd(v_CRTM_14_3, v_s6);
        v_t27 = _mm512_mul_pd(v_CRTM_14_5, v_s10);
        v_t28 = _mm512_mul_pd(v_CRTM_14_2, v_s11);
        v_t29 = _mm512_mul_pd(v_CRTM_14_4, v_s3);
        v_t30 = _mm512_mul_pd(v_CRTM_14_6, v_s7);

        v_s43 = _mm512_sub_pd(v_s1, v_t28);
        v_s44 = _mm512_sub_pd(v_t29, v_t30);
        v_s45 = _mm512_sub_pd(v_t27, v_t25);
        v_s23 = _mm512_add_pd(v_s43, v_s44);
        v_s24 = _mm512_sub_pd(v_s45, v_t26);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm512_add_pd(v_s23, v_s24);
        STR_512_D(curr_out, v_out_stride, v_out5);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm512_sub_pd(v_s23, v_s24);
        STR_512_D(curr_out, v_out_stride, v_out9);

        v_t31 = _mm512_mul_pd(v_CRTM_14_1, v_s5);
        v_t32 = _mm512_mul_pd(v_CRTM_14_3, v_s9);
        v_t33 = _mm512_mul_pd(v_CRTM_14_5, v_s13);
        v_t34 = _mm512_mul_pd(v_CRTM_14_2, v_s4);
        v_t35 = _mm512_mul_pd(v_CRTM_14_4, v_s8);
        v_t36 = _mm512_mul_pd(v_CRTM_14_6, v_s12);

        v_s46 = _mm512_sub_pd(v_s2, v_t34);
        v_s47 = _mm512_sub_pd(v_t35, v_t36);
        v_s48 = _mm512_sub_pd(v_t32, v_t31);
        v_s25 = _mm512_add_pd(v_s46, v_s47);
        v_s26 = _mm512_sub_pd(v_s48, v_t33);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm512_add_pd(v_s25, v_s26);
        STR_512_D(curr_out, v_out_stride, v_out6);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm512_sub_pd(v_s25, v_s26);
        STR_512_D(curr_out, v_out_stride, v_out8);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (n & 4)
    {
        __m256d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
                v_in9, v_in10, v_in11, v_in12, v_in13;
        __m256d v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9, v_s10,
                v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18, v_s19,
                v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27, v_s28,
                v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36, v_s37,
                v_s38, v_s39, v_s40, v_s41, v_s42, v_s43, v_s44, v_s45, v_s46,
                v_s47, v_s48;
        __m256d v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9, v_t10,
                v_t11, v_t12, v_t13, v_t14, v_t15, v_t16, v_t17, v_t18, v_t19,
                v_t20, v_t21, v_t22, v_t23, v_t24, v_t25, v_t26, v_t27, v_t28,
                v_t29, v_t30, v_t31, v_t32, v_t33, v_t34, v_t35, v_t36, v_t37,
                v_t38;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13;

        curr_in = in;
        curr_out = out;

        __m256d v256_CRTM_14_1 = _mm512_castpd512_pd256(v_CRTM_14_1);
        __m256d v256_CRTM_14_2 = _mm512_castpd512_pd256(v_CRTM_14_2);
        __m256d v256_CRTM_14_3 = _mm512_castpd512_pd256(v_CRTM_14_3);
        __m256d v256_CRTM_14_4 = _mm512_castpd512_pd256(v_CRTM_14_4);
        __m256d v256_CRTM_14_5 = _mm512_castpd512_pd256(v_CRTM_14_5);
        __m256d v256_CRTM_14_6 = _mm512_castpd512_pd256(v_CRTM_14_6);
        __m256d v256_CRTM_14_7 = _mm512_castpd512_pd256(v_CRTM_14_7);

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
        // Input point 12: x(11) & Input point 13: x(12)
        curr_in = in + in_strides[11];
        LDRI_2x256_D(curr_in, v_in_stride, v_in11, v_in12);
        // Input point 14: x(13) & Input point 15: x(14)
        curr_in = in + in_strides[13];
        LDR_256_D(curr_in, v_in_stride, v_in13);

        v_s1 = _mm256_sub_pd(v_in0, v_in13);
        v_s2 = _mm256_add_pd(v_in0, v_in13);
        v_s3 = _mm256_sub_pd(v_in11, v_in1);
        v_s4 = _mm256_add_pd(v_in1, v_in11);
        v_s5 = _mm256_sub_pd(v_in2, v_in12);
        v_s6 = _mm256_add_pd(v_in2, v_in12);
        v_s7 = _mm256_sub_pd(v_in3, v_in9);
        v_s8 = _mm256_add_pd(v_in3, v_in9);
        v_s9 = _mm256_sub_pd(v_in4, v_in10);
        v_s10 = _mm256_add_pd(v_in4, v_in10);
        v_s11 = _mm256_sub_pd(v_in7, v_in5);
        v_s12 = _mm256_add_pd(v_in5, v_in7);
        v_s13 = _mm256_sub_pd(v_in6, v_in8);
        v_s14 = _mm256_add_pd(v_in6, v_in8);

        v_s27 = _mm256_add_pd(v_s12, v_s4);
        v_s28 = _mm256_add_pd(v_s27, v_s8);
        v_t37 = _mm256_mul_pd(v256_CRTM_14_7, v_s28);
        v_s29 = _mm256_add_pd(v_s3, v_s7);
        v_s30 = _mm256_add_pd(v_s29, v_s11);
        v_t38 = _mm256_mul_pd(v256_CRTM_14_7, v_s30);
        // Output pt 1: X(0)
        v_out0 = _mm256_add_pd(v_t37, v_s2);
        STR_256_D(curr_out, v_out_stride, v_out0);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = _mm256_add_pd(v_t38, v_s1);
        STR_256_D(curr_out, v_out_stride, v_out7);

        v_t1 = _mm256_mul_pd(v256_CRTM_14_1, v_s6);
        v_t2 = _mm256_mul_pd(v256_CRTM_14_3, v_s10);
        v_t3 = _mm256_mul_pd(v256_CRTM_14_5, v_s14);
        v_t4 = _mm256_mul_pd(v256_CRTM_14_2, v_s3);
        v_t5 = _mm256_mul_pd(v256_CRTM_14_4, v_s7);
        v_t6 = _mm256_mul_pd(v256_CRTM_14_6, v_s11);

        v_s31 = _mm256_sub_pd(v_t5, v_t6);
        v_s32 = _mm256_sub_pd(v_s1, v_t4);
        v_s33 = _mm256_add_pd(v_t1, v_t2);

        v_s15 = _mm256_add_pd(v_s31, v_s32);
        v_s16 = _mm256_add_pd(v_s33, v_t3);
        // Output point 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm256_sub_pd(v_s15, v_s16);
        STR_256_D(curr_out, v_out_stride, v_out1);
        // Output point 14: X(13)
        curr_out = out + out_strides[13];
        v_out13 = _mm256_add_pd(v_s15, v_s16);
        STR_256_D(curr_out, v_out_stride, v_out13);

        v_t7 = _mm256_mul_pd(v256_CRTM_14_1, v_s13);
        v_t8 = _mm256_mul_pd(v256_CRTM_14_3, v_s5);
        v_t9 = _mm256_mul_pd(v256_CRTM_14_5, v_s9);

        v_t10 = _mm256_mul_pd(v256_CRTM_14_2, v_s12);
        v_t11 = _mm256_mul_pd(v256_CRTM_14_4, v_s4);
        v_t12 = _mm256_mul_pd(v256_CRTM_14_6, v_s8);

        v_s34 = _mm256_sub_pd(v_s2, v_t10);
        v_s35 = _mm256_sub_pd(v_t11, v_t12);
        v_s36 = _mm256_add_pd(v_t7, v_t8);

        v_s17 = v_s34 + v_s35;
        v_s18 = _mm256_add_pd(v_s36, v_t9);
        // Output point 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm256_sub_pd(v_s17, v_s18);
        STR_256_D(curr_out, v_out_stride, v_out2);
        // Output point 13: X(12)
        curr_out = out + out_strides[12];
        v_out12 = _mm256_add_pd(v_s17, v_s18);
        STR_256_D(curr_out, v_out_stride, v_out12);

        v_t13 = _mm256_mul_pd(v256_CRTM_14_1, v_s10);
        v_t14 = _mm256_mul_pd(v256_CRTM_14_3, v_s14);
        v_t15 = _mm256_mul_pd(v256_CRTM_14_5, v_s6);
        v_t16 = _mm256_mul_pd(v256_CRTM_14_2, v_s7);
        v_t17 = _mm256_mul_pd(v256_CRTM_14_4, v_s11);
        v_t18 = _mm256_mul_pd(v256_CRTM_14_6, v_s3);

        v_s37 = _mm256_sub_pd(v_s1, v_t16);
        v_s38 = _mm256_sub_pd(v_t17, v_t18);
        v_s39 = _mm256_sub_pd(v_t14, v_t15);
        v_s19 = _mm256_add_pd(v_s37, v_s38);
        v_s20 = _mm256_sub_pd(v_s39, v_t13);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm256_add_pd(v_s19, v_s20);
        STR_256_D(curr_out, v_out_stride, v_out3);
        // Output point 12: X(11)
        curr_out = out + out_strides[11];
        v_out11 = _mm256_sub_pd(v_s19, v_s20);
        STR_256_D(curr_out, v_out_stride, v_out11);

        v_t19 = _mm256_mul_pd(v256_CRTM_14_1, v_s9);
        v_t20 = _mm256_mul_pd(v256_CRTM_14_3, v_s13);
        v_t21 = _mm256_mul_pd(v256_CRTM_14_5, v_s5);
        v_t22 = _mm256_mul_pd(v256_CRTM_14_2, v_s8);
        v_t23 = _mm256_mul_pd(v256_CRTM_14_4, v_s12);
        v_t24 = _mm256_mul_pd(v256_CRTM_14_6, v_s4);

        v_s40 = _mm256_sub_pd(v_s2, v_t22);
        v_s41 = _mm256_sub_pd(v_t23, v_t24);
        v_s42 = _mm256_add_pd(v_t19, v_t20);
        v_s21 = _mm256_add_pd(v_s40, v_s41);
        v_s22 = _mm256_sub_pd(v_s42, v_t21);
        // Output pt 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm256_add_pd(v_s21, v_s22);
        STR_256_D(curr_out, v_out_stride, v_out4);
        // Output point 11: X(10)
        curr_out = out + out_strides[10];
        v_out10 = _mm256_sub_pd(v_s21, v_s22);
        STR_256_D(curr_out, v_out_stride, v_out10);

        v_t25 = _mm256_mul_pd(v256_CRTM_14_1, v_s14);
        v_t26 = _mm256_mul_pd(v256_CRTM_14_3, v_s6);
        v_t27 = _mm256_mul_pd(v256_CRTM_14_5, v_s10);
        v_t28 = _mm256_mul_pd(v256_CRTM_14_2, v_s11);
        v_t29 = _mm256_mul_pd(v256_CRTM_14_4, v_s3);
        v_t30 = _mm256_mul_pd(v256_CRTM_14_6, v_s7);

        v_s43 = _mm256_sub_pd(v_s1, v_t28);
        v_s44 = _mm256_sub_pd(v_t29, v_t30);
        v_s45 = _mm256_sub_pd(v_t27, v_t25);
        v_s23 = _mm256_add_pd(v_s43, v_s44);
        v_s24 = _mm256_sub_pd(v_s45, v_t26);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm256_add_pd(v_s23, v_s24);
        STR_256_D(curr_out, v_out_stride, v_out5);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm256_sub_pd(v_s23, v_s24);
        STR_256_D(curr_out, v_out_stride, v_out9);

        v_t31 = _mm256_mul_pd(v256_CRTM_14_1, v_s5);
        v_t32 = _mm256_mul_pd(v256_CRTM_14_3, v_s9);
        v_t33 = _mm256_mul_pd(v256_CRTM_14_5, v_s13);
        v_t34 = _mm256_mul_pd(v256_CRTM_14_2, v_s4);
        v_t35 = _mm256_mul_pd(v256_CRTM_14_4, v_s8);
        v_t36 = _mm256_mul_pd(v256_CRTM_14_6, v_s12);

        v_s46 = _mm256_sub_pd(v_s2, v_t34);
        v_s47 = _mm256_sub_pd(v_t35, v_t36);
        v_s48 = _mm256_sub_pd(v_t32, v_t31);
        v_s25 = _mm256_add_pd(v_s46, v_s47);
        v_s26 = _mm256_sub_pd(v_s48, v_t33);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm256_add_pd(v_s25, v_s26);
        STR_256_D(curr_out, v_out_stride, v_out6);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm256_sub_pd(v_s25, v_s26);
        STR_256_D(curr_out, v_out_stride, v_out8);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (n & 2)
    {
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7, v_in8,
                v_in9, v_in10, v_in11, v_in12, v_in13;
        __m128d v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9, v_s10,
                v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18, v_s19,
                v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27, v_s28,
                v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36, v_s37,
                v_s38, v_s39, v_s40, v_s41, v_s42, v_s43, v_s44, v_s45, v_s46,
                v_s47, v_s48;
        __m128d v_t1, v_t2, v_t3, v_t4, v_t5, v_t6, v_t7, v_t8, v_t9, v_t10,
                v_t11, v_t12, v_t13, v_t14, v_t15, v_t16, v_t17, v_t18, v_t19,
                v_t20, v_t21, v_t22, v_t23, v_t24, v_t25, v_t26, v_t27, v_t28,
                v_t29, v_t30, v_t31, v_t32, v_t33, v_t34, v_t35, v_t36, v_t37,
                v_t38;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13;

        curr_in = in;
        curr_out = out;

        __m128d v128_CRTM_14_1 = _mm512_castpd512_pd128(v_CRTM_14_1);
        __m128d v128_CRTM_14_2 = _mm512_castpd512_pd128(v_CRTM_14_2);
        __m128d v128_CRTM_14_3 = _mm512_castpd512_pd128(v_CRTM_14_3);
        __m128d v128_CRTM_14_4 = _mm512_castpd512_pd128(v_CRTM_14_4);
        __m128d v128_CRTM_14_5 = _mm512_castpd512_pd128(v_CRTM_14_5);
        __m128d v128_CRTM_14_6 = _mm512_castpd512_pd128(v_CRTM_14_6);
        __m128d v128_CRTM_14_7 = _mm512_castpd512_pd128(v_CRTM_14_7);

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
        // Input point 12: x(11) & Input point 13: x(12)
        curr_in = in + in_strides[11];
        LDRI_2x128_D(curr_in, v_in_stride, v_in11, v_in12);
        // Input point 14: x(13) & Input point 15: x(14)
        curr_in = in + in_strides[13];
        LDR_128_D(curr_in, v_in_stride, v_in13);

        v_s1 = _mm_sub_pd(v_in0, v_in13);
        v_s2 = _mm_add_pd(v_in0, v_in13);
        v_s3 = _mm_sub_pd(v_in11, v_in1);
        v_s4 = _mm_add_pd(v_in1, v_in11);
        v_s5 = _mm_sub_pd(v_in2, v_in12);
        v_s6 = _mm_add_pd(v_in2, v_in12);
        v_s7 = _mm_sub_pd(v_in3, v_in9);
        v_s8 = _mm_add_pd(v_in3, v_in9);
        v_s9 = _mm_sub_pd(v_in4, v_in10);
        v_s10 = _mm_add_pd(v_in4, v_in10);
        v_s11 = _mm_sub_pd(v_in7, v_in5);
        v_s12 = _mm_add_pd(v_in5, v_in7);
        v_s13 = _mm_sub_pd(v_in6, v_in8);
        v_s14 = _mm_add_pd(v_in6, v_in8);

        v_s27 = _mm_add_pd(v_s12, v_s4);
        v_s28 = _mm_add_pd(v_s27, v_s8);
        v_t37 = _mm_mul_pd(v128_CRTM_14_7, v_s28);
        v_s29 = _mm_add_pd(v_s3, v_s7);
        v_s30 = _mm_add_pd(v_s29, v_s11);
        v_t38 = _mm_mul_pd(v128_CRTM_14_7, v_s30);
        // Output pt 1: X(0)
        v_out0 = _mm_add_pd(v_t37, v_s2);
        STR_128_D(curr_out, v_out_stride, v_out0);
        // Output pt 8: X(7)
        curr_out = out + out_strides[7];
        v_out7 = _mm_add_pd(v_t38, v_s1);
        STR_128_D(curr_out, v_out_stride, v_out7);

        v_t1 = _mm_mul_pd(v128_CRTM_14_1, v_s6);
        v_t2 = _mm_mul_pd(v128_CRTM_14_3, v_s10);
        v_t3 = _mm_mul_pd(v128_CRTM_14_5, v_s14);
        v_t4 = _mm_mul_pd(v128_CRTM_14_2, v_s3);
        v_t5 = _mm_mul_pd(v128_CRTM_14_4, v_s7);
        v_t6 = _mm_mul_pd(v128_CRTM_14_6, v_s11);

        v_s31 = _mm_sub_pd(v_t5, v_t6);
        v_s32 = _mm_sub_pd(v_s1, v_t4);
        v_s33 = _mm_add_pd(v_t1, v_t2);

        v_s15 = _mm_add_pd(v_s31, v_s32);
        v_s16 = _mm_add_pd(v_s33, v_t3);
        // Output point 2: X(1)
        curr_out = out + out_strides[1];
        v_out1 = _mm_sub_pd(v_s15, v_s16);
        STR_128_D(curr_out, v_out_stride, v_out1);
        // Output point 14: X(13)
        curr_out = out + out_strides[13];
        v_out13 = _mm_add_pd(v_s15, v_s16);
        STR_128_D(curr_out, v_out_stride, v_out13);

        v_t7 = _mm_mul_pd(v128_CRTM_14_1, v_s13);
        v_t8 = _mm_mul_pd(v128_CRTM_14_3, v_s5);
        v_t9 = _mm_mul_pd(v128_CRTM_14_5, v_s9);

        v_t10 = _mm_mul_pd(v128_CRTM_14_2, v_s12);
        v_t11 = _mm_mul_pd(v128_CRTM_14_4, v_s4);
        v_t12 = _mm_mul_pd(v128_CRTM_14_6, v_s8);

        v_s34 = _mm_sub_pd(v_s2, v_t10);
        v_s35 = _mm_sub_pd(v_t11, v_t12);
        v_s36 = _mm_add_pd(v_t7, v_t8);

        v_s17 = v_s34 + v_s35;
        v_s18 = _mm_add_pd(v_s36, v_t9);
        // Output point 3: X(2)
        curr_out = out + out_strides[2];
        v_out2 = _mm_sub_pd(v_s17, v_s18);
        STR_128_D(curr_out, v_out_stride, v_out2);
        // Output point 13: X(12)
        curr_out = out + out_strides[12];
        v_out12 = _mm_add_pd(v_s17, v_s18);
        STR_128_D(curr_out, v_out_stride, v_out12);

        v_t13 = _mm_mul_pd(v128_CRTM_14_1, v_s10);
        v_t14 = _mm_mul_pd(v128_CRTM_14_3, v_s14);
        v_t15 = _mm_mul_pd(v128_CRTM_14_5, v_s6);
        v_t16 = _mm_mul_pd(v128_CRTM_14_2, v_s7);
        v_t17 = _mm_mul_pd(v128_CRTM_14_4, v_s11);
        v_t18 = _mm_mul_pd(v128_CRTM_14_6, v_s3);

        v_s37 = _mm_sub_pd(v_s1, v_t16);
        v_s38 = _mm_sub_pd(v_t17, v_t18);
        v_s39 = _mm_sub_pd(v_t14, v_t15);
        v_s19 = _mm_add_pd(v_s37, v_s38);
        v_s20 = _mm_sub_pd(v_s39, v_t13);
        // Output pt 4: X(3)
        curr_out = out + out_strides[3];
        v_out3 = _mm_add_pd(v_s19, v_s20);
        STR_128_D(curr_out, v_out_stride, v_out3);
        // Output point 12: X(11)
        curr_out = out + out_strides[11];
        v_out11 = _mm_sub_pd(v_s19, v_s20);
        STR_128_D(curr_out, v_out_stride, v_out11);

        v_t19 = _mm_mul_pd(v128_CRTM_14_1, v_s9);
        v_t20 = _mm_mul_pd(v128_CRTM_14_3, v_s13);
        v_t21 = _mm_mul_pd(v128_CRTM_14_5, v_s5);
        v_t22 = _mm_mul_pd(v128_CRTM_14_2, v_s8);
        v_t23 = _mm_mul_pd(v128_CRTM_14_4, v_s12);
        v_t24 = _mm_mul_pd(v128_CRTM_14_6, v_s4);

        v_s40 = _mm_sub_pd(v_s2, v_t22);
        v_s41 = _mm_sub_pd(v_t23, v_t24);
        v_s42 = _mm_add_pd(v_t19, v_t20);
        v_s21 = _mm_add_pd(v_s40, v_s41);
        v_s22 = _mm_sub_pd(v_s42, v_t21);
        // Output pt 5: X(4)
        curr_out = out + out_strides[4];
        v_out4 = _mm_add_pd(v_s21, v_s22);
        STR_128_D(curr_out, v_out_stride, v_out4);
        // Output point 11: X(10)
        curr_out = out + out_strides[10];
        v_out10 = _mm_sub_pd(v_s21, v_s22);
        STR_128_D(curr_out, v_out_stride, v_out10);

        v_t25 = _mm_mul_pd(v128_CRTM_14_1, v_s14);
        v_t26 = _mm_mul_pd(v128_CRTM_14_3, v_s6);
        v_t27 = _mm_mul_pd(v128_CRTM_14_5, v_s10);
        v_t28 = _mm_mul_pd(v128_CRTM_14_2, v_s11);
        v_t29 = _mm_mul_pd(v128_CRTM_14_4, v_s3);
        v_t30 = _mm_mul_pd(v128_CRTM_14_6, v_s7);

        v_s43 = _mm_sub_pd(v_s1, v_t28);
        v_s44 = _mm_sub_pd(v_t29, v_t30);
        v_s45 = _mm_sub_pd(v_t27, v_t25);
        v_s23 = _mm_add_pd(v_s43, v_s44);
        v_s24 = _mm_sub_pd(v_s45, v_t26);
        // Output pt 6: X(5)
        curr_out = out + out_strides[5];
        v_out5 = _mm_add_pd(v_s23, v_s24);
        STR_128_D(curr_out, v_out_stride, v_out5);
        // Output pt 10: X(9)
        curr_out = out + out_strides[9];
        v_out9 = _mm_sub_pd(v_s23, v_s24);
        STR_128_D(curr_out, v_out_stride, v_out9);

        v_t31 = _mm_mul_pd(v128_CRTM_14_1, v_s5);
        v_t32 = _mm_mul_pd(v128_CRTM_14_3, v_s9);
        v_t33 = _mm_mul_pd(v128_CRTM_14_5, v_s13);
        v_t34 = _mm_mul_pd(v128_CRTM_14_2, v_s4);
        v_t35 = _mm_mul_pd(v128_CRTM_14_4, v_s8);
        v_t36 = _mm_mul_pd(v128_CRTM_14_6, v_s12);

        v_s46 = _mm_sub_pd(v_s2, v_t34);
        v_s47 = _mm_sub_pd(v_t35, v_t36);
        v_s48 = _mm_sub_pd(v_t32, v_t31);
        v_s25 = _mm_add_pd(v_s46, v_s47);
        v_s26 = _mm_sub_pd(v_s48, v_t33);
        // Output pt 7: X(6)
        curr_out = out + out_strides[6];
        v_out6 = _mm_add_pd(v_s25, v_s26);
        STR_128_D(curr_out, v_out_stride, v_out6);
        // Output pt 9: X(8)
        curr_out = out + out_strides[8];
        v_out8 = _mm_sub_pd(v_s25, v_s26);
        STR_128_D(curr_out, v_out_stride, v_out8);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        DOUBLE in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11,
               in12, in13;
        DOUBLE s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14,
               s15, s16, s17, s18, s19, s20, s21, s22, s23, s24, s25;
        DOUBLE t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14,
               t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27,
               t28, t29, t30, t31, t32, t33, t34, t35;

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

        s0 = in0 - in13;
        s1 = in0 + in13;
        s2 = in11 - in1;
        s3 = in1 + in11;
        s4 = in2 - in12;
        s5 = in2 + in12;
        s6 = in3 - in9;
        s7 = in3 + in9;
        s8 = in4 - in10;
        s9 = in4 + in10;
        s10 = in7 - in5;
        s11 = in5 + in7;
        s12 = in6 - in8;
        s13 = in6 + in8;

        *out = CRTM_14_7 * (s11 + s3 + s7) + s1;
        out[out_strides[7]] = CRTM_14_7 * (s2 + s6 + s10) + s0;

        t0 = CRTM_14_1 * s5;
        t1 = CRTM_14_3 * s9;
        t2 = CRTM_14_5 * s13;
        t3 = CRTM_14_2 * s2;
        t4 = CRTM_14_4 * s6;
        t5 = CRTM_14_6 * s10;

        s14 = t4 - t5 - t3 + s0;
        s15 = t0 + t1 + t2;
        out[out_strides[1]] = s14 - s15;
        out[out_strides[13]] = s14 + s15;

        t6 = CRTM_14_1 * s12;
        t7 = CRTM_14_3 * s4;
        t8 = CRTM_14_5 * s8;

        t9 = CRTM_14_2 * s11;
        t10 = CRTM_14_4 * s3;
        t11 = CRTM_14_6 * s7;

        s16 = t10 - t9 - t11 + s1;
        s17 = t6 + t7 + t8;
        out[out_strides[2]] = s16 - s17;
        out[out_strides[12]] = s16 + s17;

        t12 = CRTM_14_1 * s9;
        t13 = CRTM_14_3 * s13;
        t14 = CRTM_14_5 * s5;
        t15 = CRTM_14_2 * s6;
        t16 = CRTM_14_4 * s10;
        t17 = CRTM_14_6 * s2;

        s18 = t16 - t15 - t17 + s0;
        s19 = t13 - t12 - t14;
        out[out_strides[3]] = s18 + s19;
        out[out_strides[11]] = s18 - s19;

        t18 = CRTM_14_1 * s8;
        t19 = CRTM_14_3 * s12;
        t20 = CRTM_14_5 * s4;
        t21 = CRTM_14_2 * s7;
        t22 = CRTM_14_4 * s11;
        t23 = CRTM_14_6 * -s3;

        s20 = t22 + t23 - t21 + s1;
        s21 = t18 + t19 - t20;
        out[out_strides[4]] = s20 + s21;
        out[out_strides[10]] = s20 - s21;

        t24 = CRTM_14_1 * s13;
        t25 = CRTM_14_3 * s5;
        t26 = CRTM_14_5 * s9;
        t27 = CRTM_14_2 * s10;
        t28 = CRTM_14_4 * s2;
        t29 = CRTM_14_6 * s6;

        s22 = t28 - t27 - t29 + s0;
        s23 = t26 - t24 - t25;
        out[out_strides[5]] = s22 + s23;
        out[out_strides[9]] = s22 - s23;

        t30 = CRTM_14_1 * s4;
        t31 = CRTM_14_3 * s8;
        t32 = CRTM_14_5 * s12;
        t33 = CRTM_14_2 * s3;
        t34 = CRTM_14_4 * s7;
        t35 = CRTM_14_6 * s11;

        s24 = t34 - t33 - t35 + s1;
        s25 = t31 - t30 - t32;
        out[out_strides[6]] = s24 + s25;
        out[out_strides[8]] = s24 - s25;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

kfft_ register_kernel_r2hc_rfft14avx512(UINT8 precision, UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hc_rfft14avx512_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft14avx512_fp64_fwd;
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
            return r2hc_rfft14avx512_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft14avx512_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
