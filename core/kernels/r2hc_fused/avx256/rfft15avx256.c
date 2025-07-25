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

/** @file rfft15avx256.c
 *
 *  @brief Radix-15 r2hc_fused Real-FFT kernel with with AVX-256 operations
 *  using x86 SIMD intrinsics.
 *
 *  This file contains the DIT radix-15 real-to-halfcomplex fused of two
 *  different implementations (Standard DFT and Shifted DFT that differs in DFT
 *  weight matrix) using x86 SIMD intrinsics for single-precision  and
 *  double-precision inputs
 *
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/r2hc_simd_avx256.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                {{{0, 58, 136, 368, 220, 60},
                                                  {0, 53, 128, 424, 311, 62}},
                                                 {{0, 58, 136, 184, 28,  60},
                                                  {0, 53, 128, 212, 14,  62}}};

ops_cycles_t get_ops_cnt_r2hcf_rfft15avx256(UINT8 precision, UINT8 direction)
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

static VOID r2hcf_rfft15avx256_fp32_fwd(VOID *in_real, VOID *in_imag,
                                        VOID *out_real, VOID *out_imag, INTP n,
                                        aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const FLOAT CRTM_15_1 =
        0.55901699437494742410229341718281905886015458990288f;
    const FLOAT CRTM_15_2 =
        0.25000000000000000000000000000000000000000000000000f;
    const FLOAT CRTM_15_3 =
        0.95105651629515357211643933337938214340569863400000f;
    const FLOAT CRTM_15_4 =
        0.58778525229247301629891039327884007596190389052978f;
    const FLOAT CRTM_15_5 =
        0.50000000000000000000000000000000000000000000000000f;
    const FLOAT CRTM_15_6 =
        0.86602540378443864676372317075293618347140262690519f;
    const FLOAT CRTM_15_11 = 0.309016994374947424102293417182819058860154590f;
    const FLOAT CRTM_15_12 = 0.809016994374947424102293417182819058860154590f;
    // Below CRTMs are the product of the above CRTMs, Precomputed to save
    // multiplications on the fly.
    // CRTM_15_7 = CRTM_15_6 * CRTM_15_4
    const FLOAT CRTM_15_7 =
        0.50903696045256706468216979248996715975105181034577f;
    // CRTM_15_8 = CRTM_15_6 * CRTM_15_3
    const FLOAT CRTM_15_8 =
        0.82363910354633184270744116161596601637855195182647f;
    // CRTM_15_9 = CRTM_15_6 * CRTM_15_1
    const FLOAT CRTM_15_9 =
        0.48412291827592710612024388657479988457787393064252f;
    // CRTM_15_10 = CRTM_15_6 * CRTM_15_2
    const FLOAT CRTM_15_10 =
        0.21650635094610964914707551542960572987794876098633f;

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
    INTP N = n >> 3;

    __m256 v_CRTM_15_1 = _mm256_broadcast_ss(&CRTM_15_1);
    __m256 v_CRTM_15_2 = _mm256_broadcast_ss(&CRTM_15_2);
    __m256 v_CRTM_15_3 = _mm256_broadcast_ss(&CRTM_15_3);
    __m256 v_CRTM_15_4 = _mm256_broadcast_ss(&CRTM_15_4);
    __m256 v_CRTM_15_5 = _mm256_broadcast_ss(&CRTM_15_5);
    __m256 v_CRTM_15_6 = _mm256_broadcast_ss(&CRTM_15_6);
    __m256 v_CRTM_15_7 = _mm256_broadcast_ss(&CRTM_15_7);
    __m256 v_CRTM_15_8 = _mm256_broadcast_ss(&CRTM_15_8);
    __m256 v_CRTM_15_9 = _mm256_broadcast_ss(&CRTM_15_9);
    __m256 v_CRTM_15_10 = _mm256_broadcast_ss(&CRTM_15_10);
    __m256 v_CRTM_15_11 = _mm256_broadcast_ss(&CRTM_15_11);
    __m256 v_CRTM_15_12 = _mm256_broadcast_ss(&CRTM_15_12);

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m256 av_in0, av_in1, av_in2, av_in3;
        __m256 av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
               av_s9, av_s10, av_s11, av_s12, av_s13, av_s15, av_s16, av_s17,
               av_s19, av_s20, av_s21, av_s22, av_s23, av_s24, av_s25, av_s26,
               av_s27, av_s28, av_s29, av_s30, av_s31;
        __m256 av_t0, av_t1, av_t2, av_t3;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21,
               v_out22, v_out23, v_out24, v_out25, v_out26, v_out27, v_out28,
               v_out29;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        curr_in = in + in_strides[0];
        LDR_256_S(curr_in, v_in_stride, av_in0);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_256_S(curr_in, v_in_stride, av_in1);
        // Input point 15: x(14)
        curr_in = in + in_strides[14];
        LDR_256_S(curr_in, v_in_stride, av_in2);
        // Input point 25: x(24)
        curr_in = in + in_strides[24];
        LDR_256_S(curr_in, v_in_stride, av_in3);

        av_s0 = _mm256_add_ps(av_in1, av_in2);
        av_s1 = _mm256_sub_ps(av_in1, av_in2);
        av_s2 = _mm256_add_ps(av_in3, av_s0);
        av_s3 = _mm256_sub_ps(av_in3, _mm256_mul_ps(v_CRTM_15_5, av_s0));

        // Input point 17: x(16)
        curr_in = in + in_strides[16];
        LDR_256_S(curr_in, v_in_stride, av_in1);
        // Input point 27: x(26)
        curr_in = in + in_strides[26];
        LDR_256_S(curr_in, v_in_stride, av_in2);
        // Input point 7: x(6)
        curr_in = in + in_strides[6];
        LDR_256_S(curr_in, v_in_stride, av_in3);

        av_s0 = _mm256_add_ps(av_in1, av_in2);
        av_s4 = _mm256_sub_ps(av_in2, av_in1);
        av_s5 = _mm256_add_ps(av_in3, av_s0);
        av_s6 = _mm256_sub_ps(av_in3, _mm256_mul_ps(v_CRTM_15_5, av_s0));

        av_s23 = _mm256_add_ps(av_s6, av_s3);
        av_s24 = _mm256_sub_ps(av_s6, av_s3);
        av_s26 = _mm256_add_ps(av_s4, av_s1);
        av_s27 = _mm256_sub_ps(av_s4, av_s1);

        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_256_S(curr_in, v_in_stride, av_in1);
        // Input point 23: x(22)
        curr_in = in + in_strides[22];
        LDR_256_S(curr_in, v_in_stride, av_in2);
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDR_256_S(curr_in, v_in_stride, av_in3);

        av_s0 = _mm256_add_ps(av_in1, av_in2);
        av_s1 = _mm256_sub_ps(av_in2, av_in1);
        av_s7 = _mm256_add_ps(av_in3, av_s0);
        av_s3 = _mm256_sub_ps(av_in3, _mm256_mul_ps(v_CRTM_15_5, av_s0));

        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDR_256_S(curr_in, v_in_stride, av_in1);
        // Input point 29: x(28)
        curr_in = in + in_strides[28];
        LDR_256_S(curr_in, v_in_stride, av_in2);
        // Input point 19: x(18)
        curr_in = in + in_strides[18];
        LDR_256_S(curr_in, v_in_stride, av_in3);

        av_s0 = _mm256_add_ps(av_in1, av_in2);
        av_s4 = _mm256_sub_ps(av_in2, av_in1);
        av_s8 = _mm256_add_ps(av_in3, av_s0);
        av_s6 = _mm256_sub_ps(av_in3, _mm256_mul_ps(v_CRTM_15_5, av_s0));

        av_s28 = _mm256_add_ps(av_s6, av_s3);
        av_s29 = _mm256_sub_ps(av_s3, av_s6);
        av_s30 = _mm256_add_ps(av_s4, av_s1);
        av_s31 = _mm256_sub_ps(av_s1, av_s4);

        // Input point 11: x(10)
        curr_in = in + in_strides[10];
        LDR_256_S(curr_in, v_in_stride, av_in1);
        // Input point 21: x(20)
        curr_in = in + in_strides[20];
        LDR_256_S(curr_in, v_in_stride, av_in2);

        av_s0 = _mm256_add_ps(av_in1, av_in2);
        av_t0 = _mm256_mul_ps(v_CRTM_15_6, _mm256_sub_ps(av_in2, av_in1));
        av_s9 = _mm256_add_ps(av_in0, av_s0);
        av_s3 = _mm256_sub_ps(av_in0, _mm256_mul_ps(v_CRTM_15_5, av_s0));

        av_s11 = _mm256_add_ps(av_s8, av_s7);
        av_s12 = _mm256_add_ps(av_s2, av_s5);
        av_s19 = _mm256_sub_ps(av_s5, av_s2);
        av_s20 = _mm256_sub_ps(av_s8, av_s7);
        av_s13 = _mm256_add_ps(av_s11, av_s12);
        av_t1 = _mm256_mul_ps(v_CRTM_15_1, _mm256_sub_ps(av_s11, av_s12));
        av_s15 = _mm256_sub_ps(av_s9, _mm256_mul_ps(v_CRTM_15_2, av_s13));

        // Output point 1: X(0)
        v_out0 = _mm256_add_ps(av_s9, av_s13);
        STR_256_S(curr_out, v_out_stride, v_out0);

        // Output point 12: X(11)
        v_out11 = _mm256_add_ps(av_s15, av_t1);
        // Output point 13: X(12)
        v_out12 = _mm256_add_ps(_mm256_mul_ps(v_CRTM_15_3, av_s20),
                                _mm256_mul_ps(v_CRTM_15_4, av_s19));
        curr_out = out + out_strides[11];
        STRI_2x256_S(curr_out, v_out_stride, v_out11, v_out12);

        // Output point 24: X(23)
        v_out23 = _mm256_sub_ps(av_s15, av_t1);
        // Output point 25: X(24)
        v_out24 = _mm256_sub_ps(_mm256_mul_ps(v_CRTM_15_4, av_s20),
                                _mm256_mul_ps(v_CRTM_15_3, av_s19));
        curr_out = out + out_strides[23];
        STRI_2x256_S(curr_out, v_out_stride, v_out23, v_out24);

        av_t2 = _mm256_mul_ps(v_CRTM_15_1, _mm256_sub_ps(av_s28, av_s23));
        av_s13 = _mm256_add_ps(av_s28, av_s23);

        // Output point 20: X(19)
        v_out19 = _mm256_add_ps(av_s13, av_s3);

        av_t3 = _mm256_mul_ps(v_CRTM_15_9, _mm256_add_ps(av_s30, av_s27));
        av_s17 = _mm256_sub_ps(av_s30, av_s27);

        // Output point 21: X(20)
        v_out20 = _mm256_sub_ps(_mm256_mul_ps(v_CRTM_15_6, av_s17), av_t0);
        curr_out = out + out_strides[19];
        STRI_2x256_S(curr_out, v_out_stride, v_out19, v_out20);

        av_s15 = _mm256_sub_ps(av_s3, _mm256_mul_ps(v_CRTM_15_2, av_s13));
        av_s20 = _mm256_add_ps(av_t0, _mm256_mul_ps(v_CRTM_15_10, av_s17));
        av_s21 = _mm256_sub_ps(av_s15, av_t2);
        av_s25 = _mm256_sub_ps(_mm256_mul_ps(v_CRTM_15_8, av_s26),
                               _mm256_mul_ps(v_CRTM_15_7, av_s31));

        // Output point 4: X(3)
        v_out3 = _mm256_add_ps(av_s21, av_s25);

        // Output point 16: X(15)
        v_out15 = _mm256_sub_ps(av_s21, av_s25);

        av_s21 = _mm256_add_ps(av_s15, av_t2);
        av_s23 = _mm256_add_ps(_mm256_mul_ps(v_CRTM_15_8, av_s31),
                               _mm256_mul_ps(v_CRTM_15_7, av_s26));

        // Output point 28: X(27)
        v_out27 = _mm256_add_ps(av_s21, av_s23);

        av_s22 = _mm256_add_ps(av_s20, av_t3);
        av_s16 = _mm256_add_ps(_mm256_mul_ps(v_CRTM_15_4, av_s29),
                               _mm256_mul_ps(v_CRTM_15_3, av_s24));

        // Output point 5: X(4)
        v_out4 = _mm256_sub_ps(av_s22, av_s16);
        curr_out = out + out_strides[3];
        STRI_2x256_S(curr_out, v_out_stride, v_out3, v_out4);
        // Output point 17: X(16)
        v_out16 = _mm256_add_ps(av_s22, av_s16);
        curr_out = out + out_strides[15];
        STRI_2x256_S(curr_out, v_out_stride, v_out15, v_out16);

        av_s22 = _mm256_sub_ps(av_s20, av_t3);
        av_s10 = _mm256_sub_ps(_mm256_mul_ps(v_CRTM_15_3, av_s29),
                               _mm256_mul_ps(v_CRTM_15_4, av_s24));

        // Output point 29: X(28)
        v_out28 = _mm256_add_ps(av_s22, av_s10);
        curr_out = out + out_strides[27];
        STRI_2x256_S(curr_out, v_out_stride, v_out27, v_out28);

        // Output point 8: X(7)
        v_out7 = _mm256_sub_ps(av_s21, av_s23);

        // Output point 9: X(8)
        v_out8 = _mm256_sub_ps(av_s10, av_s22);
        curr_out = out + out_strides[7];
        STRI_2x256_S(curr_out, v_out_stride, v_out7, v_out8);

        /* Shifted DFT */
        __m256 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8, bv_in9, bv_in10, bv_in11, bv_in12, bv_in13, bv_in14;
        __m256 bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
               bv_t10, bv_t11, bv_t12;
        __m256 bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
               bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17,
               bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24, bv_s25,
               bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32, bv_s33,
               bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40, bv_s41,
               bv_s42, bv_s43, bv_s44, bv_s45, bv_s46, bv_s47, bv_s48, bv_s49,
               bv_s50, bv_s51;

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
        // Input point 22: x(21)
        curr_in = in + in_strides[21];
        LDR_256_S(curr_in, v_in_stride, bv_in10);
        // Input point 24: x(23)
        curr_in = in + in_strides[23];
        LDR_256_S(curr_in, v_in_stride, bv_in11);
        // Input point 26: x(25)
        curr_in = in + in_strides[25];
        LDR_256_S(curr_in, v_in_stride, bv_in12);
        // Input point 28: x(27)
        curr_in = in + in_strides[27];
        LDR_256_S(curr_in, v_in_stride, bv_in13);
        // Input point 30: x(29)
        curr_in = in + in_strides[29];
        LDR_256_S(curr_in, v_in_stride, bv_in14);

        bv_s1 = _mm256_add_ps(bv_in1, bv_in4);
        bv_s2 = _mm256_add_ps(bv_in7, bv_in13);
        bv_s3 = _mm256_sub_ps(bv_in7, bv_in13);
        bv_s4 = _mm256_add_ps(bv_in6, bv_in9);
        bv_s5 = _mm256_add_ps(bv_in12, bv_in3);
        bv_s42 = _mm256_add_ps(_mm256_mul_ps(v_CRTM_15_4, bv_s5),
                               _mm256_mul_ps(v_CRTM_15_3, bv_s4));
        bv_s28 = _mm256_sub_ps(_mm256_mul_ps(v_CRTM_15_4, bv_s4),
                               _mm256_mul_ps(v_CRTM_15_3, bv_s5));

        bv_s6 = _mm256_add_ps(bv_in14, bv_in11);
        bv_s7 = _mm256_add_ps(bv_in2, bv_in8);
        bv_t2  = _mm256_mul_ps(v_CRTM_15_1, _mm256_sub_ps(bv_in14, bv_s7));
        bv_s47 = _mm256_sub_ps(bv_t2, _mm256_mul_ps(v_CRTM_15_11, bv_in11));
        bv_s30 = _mm256_sub_ps(_mm256_mul_ps(v_CRTM_15_12, bv_in11), bv_in5);

        bv_s8 = _mm256_sub_ps(bv_in2, bv_in8);
        bv_s9 = _mm256_add_ps(bv_in9, bv_in3);
        bv_s10 = _mm256_add_ps(bv_in6, bv_in12);
        bv_s26 = _mm256_add_ps(_mm256_mul_ps(v_CRTM_15_4, bv_s6),
                               _mm256_mul_ps(v_CRTM_15_3, bv_s8));

        bv_s11 = _mm256_sub_ps(bv_s9, bv_s10);
        bv_s12 = _mm256_add_ps(bv_in1, bv_s2);
        bv_s13 = _mm256_add_ps(bv_in14, bv_s7);
        bv_s48 = _mm256_add_ps(_mm256_mul_ps(v_CRTM_15_2, bv_s13), bv_in5);
        bv_s14 = _mm256_sub_ps(bv_in0, bv_s11);

        bv_s15 = _mm256_add_ps(bv_in10, bv_in4);
        bv_s16 = _mm256_sub_ps(bv_s15, bv_s12);
        bv_s17 = _mm256_add_ps(bv_in11, bv_in5);
        bv_s18 = _mm256_sub_ps(bv_s13, bv_s17);
        bv_s19 = _mm256_add_ps(bv_s16, bv_s18);
        bv_s20 = _mm256_add_ps(bv_in0, _mm256_mul_ps(v_CRTM_15_2, bv_s11));
        // Output pt 30: X(29)
        v_out29 = _mm256_add_ps(bv_s19, bv_s14);
        curr_out = out + out_strides[29];
        STR_256_S(curr_out, v_out_stride, v_out29);
        // Output pt 10: X(9) & Output pt 11: X(10)
        v_out9 = _mm256_sub_ps(bv_s14, _mm256_mul_ps(v_CRTM_15_5, bv_s19));
        v_out10 = _mm256_mul_ps(v_CRTM_15_6, _mm256_sub_ps(bv_s16, bv_s18));
        curr_out = out + out_strides[9];
        STRI_2x256_S(curr_out, v_out_stride, v_out9, v_out10);

        bv_s21 = _mm256_add_ps(bv_in9, bv_in12);
        bv_s22 = _mm256_add_ps(bv_in6, bv_in3);
        bv_s23 = _mm256_sub_ps(bv_s21, bv_s22);
        bv_t1  = _mm256_mul_ps(v_CRTM_15_1, bv_s23);
        bv_s24 = _mm256_add_ps(bv_s20, bv_t1);
        bv_s25 = _mm256_sub_ps(_mm256_mul_ps(v_CRTM_15_3, bv_s3),
                               _mm256_mul_ps(v_CRTM_15_4, bv_s1));

        bv_t6  = _mm256_mul_ps(v_CRTM_15_6, _mm256_add_ps(bv_s25, bv_s26));
        bv_s27 = _mm256_sub_ps(bv_s25, bv_s26);

        bv_s29 = _mm256_sub_ps(_mm256_mul_ps(v_CRTM_15_5, bv_s27), bv_s28);
        bv_s31 = _mm256_add_ps(_mm256_mul_ps(v_CRTM_15_2, bv_s13), bv_t2);
        bv_s32 = _mm256_sub_ps(bv_s30, bv_s31);

        bv_t3  = _mm256_mul_ps(v_CRTM_15_1, _mm256_sub_ps(bv_in1, bv_s2));
        bv_s33 = _mm256_add_ps(bv_in10, bv_t3);
        bv_t4  = _mm256_mul_ps(v_CRTM_15_2, bv_s12);
        bv_s34 = _mm256_sub_ps(bv_t4, _mm256_mul_ps(v_CRTM_15_12, bv_in4));
        bv_s35 = _mm256_add_ps(bv_s33, bv_s34);
        bv_s36 = _mm256_add_ps(bv_s32, bv_s35);
        bv_t5  = _mm256_mul_ps(v_CRTM_15_6, _mm256_sub_ps(bv_s32, bv_s35));
        // Output pt 6: X(5) & Output pt 7: X(6)
        v_out5 = _mm256_add_ps(bv_s24, bv_s36);
        v_out6 = _mm256_add_ps(bv_s28, bv_s27);
        curr_out = out + out_strides[5];
        STRI_2x256_S(curr_out, v_out_stride, v_out5, v_out6);

        bv_s37 = _mm256_sub_ps(bv_s24, _mm256_mul_ps(v_CRTM_15_5, bv_s36));
        // Output pt 14: X(13)& Output pt 15: X(14)
        v_out13 = _mm256_sub_ps(bv_s37, bv_t6);
        v_out14 = _mm256_add_ps(bv_t5, bv_s29);
        curr_out = out + out_strides[13];
        STRI_2x256_S(curr_out, v_out_stride, v_out13, v_out14);
        // Output pt 26: X(25) & Output pt 27: X(26)
        v_out25 = _mm256_add_ps(bv_s37, bv_t6);
        v_out26 = _mm256_sub_ps(bv_t5, bv_s29);
        curr_out = out + out_strides[25];
        STRI_2x256_S(curr_out, v_out_stride, v_out25, v_out26);

        bv_t7  = _mm256_mul_ps(v_CRTM_15_4, bv_s3);
        bv_t8  = _mm256_mul_ps(v_CRTM_15_3, bv_s1);
        bv_s38 = _mm256_add_ps(bv_t8, bv_t7);
        bv_t9  = _mm256_mul_ps(v_CRTM_15_4, bv_s8);
        bv_t10 = _mm256_mul_ps(v_CRTM_15_3, bv_s6);
        bv_s39 = _mm256_sub_ps(bv_t9, bv_t10);

        bv_t11 = _mm256_mul_ps(v_CRTM_15_6, _mm256_add_ps(bv_s38, bv_s39));
        bv_s40 = _mm256_sub_ps(bv_s20, bv_t1);
        bv_s41 = _mm256_sub_ps(bv_s39, bv_s38);
        bv_s43 = _mm256_sub_ps(_mm256_mul_ps(v_CRTM_15_5, bv_s41), bv_s42);

        bv_s44 = _mm256_sub_ps(bv_t4, bv_t3);
        bv_s45 = _mm256_add_ps(bv_in10, _mm256_mul_ps(v_CRTM_15_11, bv_in4));
        bv_s46 = _mm256_add_ps(bv_s45, bv_s44);
        bv_s49 = _mm256_sub_ps(bv_s47, bv_s48);
        bv_s50 = _mm256_add_ps(bv_s46, bv_s49);
        bv_t12 = _mm256_mul_ps(v_CRTM_15_6, _mm256_sub_ps(bv_s49, bv_s46));
        bv_s51 = _mm256_sub_ps(bv_s40, _mm256_mul_ps(v_CRTM_15_5, bv_s50));

        // Output pt 2: X(1) & Output pt 3: X(2)
        v_out1 = _mm256_add_ps(bv_s51, bv_t11);
        v_out2 = _mm256_add_ps(bv_s43, bv_t12);
        curr_out = out + out_strides[1];
        STRI_2x256_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output pt 18: X(17) & Output pt 19: X(18)
        v_out17 = _mm256_add_ps(bv_s40, bv_s50);
        v_out18 = _mm256_add_ps(bv_s42, bv_s41);
        curr_out = out + out_strides[17];
        STRI_2x256_S(curr_out, v_out_stride, v_out17, v_out18);
        // Output pt 22: X(21) & Output pt 23: X(22)
        v_out21 = _mm256_sub_ps(bv_s51, bv_t11);
        v_out22 = _mm256_sub_ps(bv_s43, bv_t12);
        curr_out = out + out_strides[21];
        STRI_2x256_S(curr_out, v_out_stride, v_out21, v_out22);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (n & 4)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1, av_in2, av_in3;
        __m128 av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
               av_s9, av_s10, av_s11, av_s12, av_s13, av_s15, av_s16, av_s17,
               av_s19, av_s20, av_s21, av_s22, av_s23, av_s24, av_s25, av_s26,
               av_s27, av_s28, av_s29, av_s30, av_s31;
        __m128 av_t0, av_t1, av_t2, av_t3;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21,
               v_out22, v_out23, v_out24, v_out25, v_out26, v_out27, v_out28,
               v_out29;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_15_1 = _mm256_castps256_ps128(v_CRTM_15_1);
        __m128 v128_CRTM_15_2 = _mm256_castps256_ps128(v_CRTM_15_2);
        __m128 v128_CRTM_15_3 = _mm256_castps256_ps128(v_CRTM_15_3);
        __m128 v128_CRTM_15_4 = _mm256_castps256_ps128(v_CRTM_15_4);
        __m128 v128_CRTM_15_5 = _mm256_castps256_ps128(v_CRTM_15_5);
        __m128 v128_CRTM_15_6 = _mm256_castps256_ps128(v_CRTM_15_6);
        __m128 v128_CRTM_15_7 = _mm256_castps256_ps128(v_CRTM_15_7);
        __m128 v128_CRTM_15_8 = _mm256_castps256_ps128(v_CRTM_15_8);
        __m128 v128_CRTM_15_9 = _mm256_castps256_ps128(v_CRTM_15_9);
        __m128 v128_CRTM_15_10 = _mm256_castps256_ps128(v_CRTM_15_10);
        __m128 v128_CRTM_15_11 = _mm256_castps256_ps128(v_CRTM_15_11);
        __m128 v128_CRTM_15_12 = _mm256_castps256_ps128(v_CRTM_15_12);

        // Input point 1: x(0)
        curr_in = in + in_strides[0];
        LDR_128_S(curr_in, v_in_stride, av_in0);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_128_S(curr_in, v_in_stride, av_in1);
        // Input point 15: x(14)
        curr_in = in + in_strides[14];
        LDR_128_S(curr_in, v_in_stride, av_in2);
        // Input point 25: x(24)
        curr_in = in + in_strides[24];
        LDR_128_S(curr_in, v_in_stride, av_in3);

        av_s0 = _mm_add_ps(av_in1, av_in2);
        av_s1 = _mm_sub_ps(av_in1, av_in2);
        av_s2 = _mm_add_ps(av_in3, av_s0);
        av_s3 = _mm_sub_ps(av_in3, _mm_mul_ps(v128_CRTM_15_5, av_s0));

        // Input point 17: x(16)
        curr_in = in + in_strides[16];
        LDR_128_S(curr_in, v_in_stride, av_in1);
        // Input point 27: x(26)
        curr_in = in + in_strides[26];
        LDR_128_S(curr_in, v_in_stride, av_in2);
        // Input point 7: x(6)
        curr_in = in + in_strides[6];
        LDR_128_S(curr_in, v_in_stride, av_in3);

        av_s0 = _mm_add_ps(av_in1, av_in2);
        av_s4 = _mm_sub_ps(av_in2, av_in1);
        av_s5 = _mm_add_ps(av_in3, av_s0);
        av_s6 = _mm_sub_ps(av_in3, _mm_mul_ps(v128_CRTM_15_5, av_s0));

        av_s23 = _mm_add_ps(av_s6, av_s3);
        av_s24 = _mm_sub_ps(av_s6, av_s3);
        av_s26 = _mm_add_ps(av_s4, av_s1);
        av_s27 = _mm_sub_ps(av_s4, av_s1);

        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_128_S(curr_in, v_in_stride, av_in1);
        // Input point 23: x(22)
        curr_in = in + in_strides[22];
        LDR_128_S(curr_in, v_in_stride, av_in2);
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDR_128_S(curr_in, v_in_stride, av_in3);

        av_s0 = _mm_add_ps(av_in1, av_in2);
        av_s1 = _mm_sub_ps(av_in2, av_in1);
        av_s7 = _mm_add_ps(av_in3, av_s0);
        av_s3 = _mm_sub_ps(av_in3, _mm_mul_ps(v128_CRTM_15_5, av_s0));

        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDR_128_S(curr_in, v_in_stride, av_in1);
        // Input point 29: x(28)
        curr_in = in + in_strides[28];
        LDR_128_S(curr_in, v_in_stride, av_in2);
        // Input point 19: x(18)
        curr_in = in + in_strides[18];
        LDR_128_S(curr_in, v_in_stride, av_in3);

        av_s0 = _mm_add_ps(av_in1, av_in2);
        av_s4 = _mm_sub_ps(av_in2, av_in1);
        av_s8 = _mm_add_ps(av_in3, av_s0);
        av_s6 = _mm_sub_ps(av_in3, _mm_mul_ps(v128_CRTM_15_5, av_s0));

        av_s28 = _mm_add_ps(av_s6, av_s3);
        av_s29 = _mm_sub_ps(av_s3, av_s6);
        av_s30 = _mm_add_ps(av_s4, av_s1);
        av_s31 = _mm_sub_ps(av_s1, av_s4);

        // Input point 11: x(10)
        curr_in = in + in_strides[10];
        LDR_128_S(curr_in, v_in_stride, av_in1);
        // Input point 21: x(20)
        curr_in = in + in_strides[20];
        LDR_128_S(curr_in, v_in_stride, av_in2);

        av_s0 = _mm_add_ps(av_in1, av_in2);
        av_t0 = _mm_mul_ps(v128_CRTM_15_6, _mm_sub_ps(av_in2, av_in1));
        av_s9 = _mm_add_ps(av_in0, av_s0);
        av_s3 = _mm_sub_ps(av_in0, _mm_mul_ps(v128_CRTM_15_5, av_s0));

        av_s11 = _mm_add_ps(av_s8, av_s7);
        av_s12 = _mm_add_ps(av_s2, av_s5);
        av_s19 = _mm_sub_ps(av_s5, av_s2);
        av_s20 = _mm_sub_ps(av_s8, av_s7);
        av_s13 = _mm_add_ps(av_s11, av_s12);
        av_t1 = _mm_mul_ps(v128_CRTM_15_1, _mm_sub_ps(av_s11, av_s12));
        av_s15 = _mm_sub_ps(av_s9, _mm_mul_ps(v128_CRTM_15_2, av_s13));

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(av_s9, av_s13);
        STR_128_S(curr_out, v_out_stride, v_out0);

        // Output point 12: X(11)
        v_out11 = _mm_add_ps(av_s15, av_t1);
        // Output point 13: X(12)
        v_out12 = _mm_add_ps(_mm_mul_ps(v128_CRTM_15_3, av_s20),
                             _mm_mul_ps(v128_CRTM_15_4, av_s19));
        curr_out = out + out_strides[11];
        STRI_2x128_S(curr_out, v_out_stride, v_out11, v_out12);

        // Output point 24: X(23)
        v_out23 = _mm_sub_ps(av_s15, av_t1);
        // Output point 25: X(24)
        v_out24 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_4, av_s20),
                             _mm_mul_ps(v128_CRTM_15_3, av_s19));
        curr_out = out + out_strides[23];
        STRI_2x128_S(curr_out, v_out_stride, v_out23, v_out24);

        av_t2 = _mm_mul_ps(v128_CRTM_15_1, _mm_sub_ps(av_s28, av_s23));
        av_s13 = _mm_add_ps(av_s28, av_s23);

        // Output point 20: X(19)
        v_out19 = _mm_add_ps(av_s13, av_s3);

        av_t3 = _mm_mul_ps(v128_CRTM_15_9, _mm_add_ps(av_s30, av_s27));
        av_s17 = _mm_sub_ps(av_s30, av_s27);

        // Output point 21: X(20)
        v_out20 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_6, av_s17), av_t0);
        curr_out = out + out_strides[19];
        STRI_2x128_S(curr_out, v_out_stride, v_out19, v_out20);

        av_s15 = _mm_sub_ps(av_s3, _mm_mul_ps(v128_CRTM_15_2, av_s13));
        av_s20 = _mm_add_ps(av_t0, _mm_mul_ps(v128_CRTM_15_10, av_s17));
        av_s21 = _mm_sub_ps(av_s15, av_t2);
        av_s25 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_8, av_s26),
                            _mm_mul_ps(v128_CRTM_15_7, av_s31));

        // Output point 4: X(3)
        v_out3 = _mm_add_ps(av_s21, av_s25);

        // Output point 16: X(15)
        v_out15 = _mm_sub_ps(av_s21, av_s25);

        av_s21 = _mm_add_ps(av_s15, av_t2);
        av_s23 = _mm_add_ps(_mm_mul_ps(v128_CRTM_15_8, av_s31),
                            _mm_mul_ps(v128_CRTM_15_7, av_s26));

        // Output point 28: X(27)
        v_out27 = _mm_add_ps(av_s21, av_s23);

        av_s22 = _mm_add_ps(av_s20, av_t3);
        av_s16 = _mm_add_ps(_mm_mul_ps(v128_CRTM_15_4, av_s29),
                            _mm_mul_ps(v128_CRTM_15_3, av_s24));

        // Output point 5: X(4)
        v_out4 = _mm_sub_ps(av_s22, av_s16);
        curr_out = out + out_strides[3];
        STRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);
        // Output point 17: X(16)
        v_out16 = _mm_add_ps(av_s22, av_s16);
        curr_out = out + out_strides[15];
        STRI_2x128_S(curr_out, v_out_stride, v_out15, v_out16);

        av_s22 = _mm_sub_ps(av_s20, av_t3);
        av_s10 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_3, av_s29),
                            _mm_mul_ps(v128_CRTM_15_4, av_s24));

        // Output point 29: X(28)
        v_out28 = _mm_add_ps(av_s22, av_s10);
        curr_out = out + out_strides[27];
        STRI_2x128_S(curr_out, v_out_stride, v_out27, v_out28);

        // Output point 8: X(7)
        v_out7 = _mm_sub_ps(av_s21, av_s23);

        // Output point 9: X(8)
        v_out8 = _mm_sub_ps(av_s10, av_s22);
        curr_out = out + out_strides[7];
        STRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8, bv_in9, bv_in10, bv_in11, bv_in12, bv_in13, bv_in14;
        __m128 bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
               bv_t10, bv_t11, bv_t12;
        __m128 bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
               bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17,
               bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24, bv_s25,
               bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32, bv_s33,
               bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40, bv_s41,
               bv_s42, bv_s43, bv_s44, bv_s45, bv_s46, bv_s47, bv_s48, bv_s49,
               bv_s50, bv_s51;

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
        // Input point 22: x(21)
        curr_in = in + in_strides[21];
        LDR_128_S(curr_in, v_in_stride, bv_in10);
        // Input point 24: x(23)
        curr_in = in + in_strides[23];
        LDR_128_S(curr_in, v_in_stride, bv_in11);
        // Input point 26: x(25)
        curr_in = in + in_strides[25];
        LDR_128_S(curr_in, v_in_stride, bv_in12);
        // Input point 28: x(27)
        curr_in = in + in_strides[27];
        LDR_128_S(curr_in, v_in_stride, bv_in13);
        // Input point 30: x(29)
        curr_in = in + in_strides[29];
        LDR_128_S(curr_in, v_in_stride, bv_in14);

        bv_s1 = _mm_add_ps(bv_in1, bv_in4);
        bv_s2 = _mm_add_ps(bv_in7, bv_in13);
        bv_s3 = _mm_sub_ps(bv_in7, bv_in13);
        bv_s4 = _mm_add_ps(bv_in6, bv_in9);
        bv_s5 = _mm_add_ps(bv_in12, bv_in3);
        bv_s42 = _mm_add_ps(_mm_mul_ps(v128_CRTM_15_4, bv_s5),
                            _mm_mul_ps(v128_CRTM_15_3, bv_s4));
        bv_s28 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_4, bv_s4),
                            _mm_mul_ps(v128_CRTM_15_3, bv_s5));

        bv_s6 = _mm_add_ps(bv_in14, bv_in11);
        bv_s7 = _mm_add_ps(bv_in2, bv_in8);
        bv_t2  = _mm_mul_ps(v128_CRTM_15_1, _mm_sub_ps(bv_in14, bv_s7));
        bv_s47 = _mm_sub_ps(bv_t2, _mm_mul_ps(v128_CRTM_15_11, bv_in11));
        bv_s30 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_12, bv_in11), bv_in5);

        bv_s8 = _mm_sub_ps(bv_in2, bv_in8);
        bv_s9 = _mm_add_ps(bv_in9, bv_in3);
        bv_s10 = _mm_add_ps(bv_in6, bv_in12);
        bv_s26 = _mm_add_ps(_mm_mul_ps(v128_CRTM_15_4, bv_s6),
                            _mm_mul_ps(v128_CRTM_15_3, bv_s8));

        bv_s11 = _mm_sub_ps(bv_s9, bv_s10);
        bv_s12 = _mm_add_ps(bv_in1, bv_s2);
        bv_s13 = _mm_add_ps(bv_in14, bv_s7);
        bv_s48 = _mm_add_ps(_mm_mul_ps(v128_CRTM_15_2, bv_s13), bv_in5);
        bv_s14 = _mm_sub_ps(bv_in0, bv_s11);

        bv_s15 = _mm_add_ps(bv_in10, bv_in4);
        bv_s16 = _mm_sub_ps(bv_s15, bv_s12);
        bv_s17 = _mm_add_ps(bv_in11, bv_in5);
        bv_s18 = _mm_sub_ps(bv_s13, bv_s17);
        bv_s19 = _mm_add_ps(bv_s16, bv_s18);
        bv_s20 = _mm_add_ps(bv_in0, _mm_mul_ps(v128_CRTM_15_2, bv_s11));
        // Output pt 30: X(29)
        v_out29 = _mm_add_ps(bv_s19, bv_s14);
        curr_out = out + out_strides[29];
        STR_128_S(curr_out, v_out_stride, v_out29);
        // Output pt 10: X(9) & Output pt 11: X(10)
        v_out9 = _mm_sub_ps(bv_s14, _mm_mul_ps(v128_CRTM_15_5, bv_s19));
        v_out10 = _mm_mul_ps(v128_CRTM_15_6, _mm_sub_ps(bv_s16, bv_s18));
        curr_out = out + out_strides[9];
        STRI_2x128_S(curr_out, v_out_stride, v_out9, v_out10);

        bv_s21 = _mm_add_ps(bv_in9, bv_in12);
        bv_s22 = _mm_add_ps(bv_in6, bv_in3);
        bv_s23 = _mm_sub_ps(bv_s21, bv_s22);
        bv_t1  = _mm_mul_ps(v128_CRTM_15_1, bv_s23);
        bv_s24 = _mm_add_ps(bv_s20, bv_t1);
        bv_s25 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_3, bv_s3),
                            _mm_mul_ps(v128_CRTM_15_4, bv_s1));

        bv_t6  = _mm_mul_ps(v128_CRTM_15_6, _mm_add_ps(bv_s25, bv_s26));
        bv_s27 = _mm_sub_ps(bv_s25, bv_s26);

        bv_s29 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_5, bv_s27), bv_s28);
        bv_s31 = _mm_add_ps(_mm_mul_ps(v128_CRTM_15_2, bv_s13), bv_t2);
        bv_s32 = _mm_sub_ps(bv_s30, bv_s31);

        bv_t3  = _mm_mul_ps(v128_CRTM_15_1, _mm_sub_ps(bv_in1, bv_s2));
        bv_s33 = _mm_add_ps(bv_in10, bv_t3);
        bv_t4  = _mm_mul_ps(v128_CRTM_15_2, bv_s12);
        bv_s34 = _mm_sub_ps(bv_t4, _mm_mul_ps(v128_CRTM_15_12, bv_in4));
        bv_s35 = _mm_add_ps(bv_s33, bv_s34);
        bv_s36 = _mm_add_ps(bv_s32, bv_s35);
        bv_t5  = _mm_mul_ps(v128_CRTM_15_6, _mm_sub_ps(bv_s32, bv_s35));
        // Output pt 6: X(5) & Output pt 7: X(6)
        v_out5 = _mm_add_ps(bv_s24, bv_s36);
        v_out6 = _mm_add_ps(bv_s28, bv_s27);
        curr_out = out + out_strides[5];
        STRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);

        bv_s37 = _mm_sub_ps(bv_s24, _mm_mul_ps(v128_CRTM_15_5, bv_s36));
        // Output pt 14: X(13)& Output pt 15: X(14)
        v_out13 = _mm_sub_ps(bv_s37, bv_t6);
        v_out14 = _mm_add_ps(bv_t5, bv_s29);
        curr_out = out + out_strides[13];
        STRI_2x128_S(curr_out, v_out_stride, v_out13, v_out14);
        // Output pt 26: X(25) & Output pt 27: X(26)
        v_out25 = _mm_add_ps(bv_s37, bv_t6);
        v_out26 = _mm_sub_ps(bv_t5, bv_s29);
        curr_out = out + out_strides[25];
        STRI_2x128_S(curr_out, v_out_stride, v_out25, v_out26);

        bv_t7  = _mm_mul_ps(v128_CRTM_15_4, bv_s3);
        bv_t8  = _mm_mul_ps(v128_CRTM_15_3, bv_s1);
        bv_s38 = _mm_add_ps(bv_t8, bv_t7);
        bv_t9  = _mm_mul_ps(v128_CRTM_15_4, bv_s8);
        bv_t10 = _mm_mul_ps(v128_CRTM_15_3, bv_s6);
        bv_s39 = _mm_sub_ps(bv_t9, bv_t10);

        bv_t11 = _mm_mul_ps(v128_CRTM_15_6, _mm_add_ps(bv_s38, bv_s39));
        bv_s40 = _mm_sub_ps(bv_s20, bv_t1);
        bv_s41 = _mm_sub_ps(bv_s39, bv_s38);
        bv_s43 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_5, bv_s41), bv_s42);

        bv_s44 = _mm_sub_ps(bv_t4, bv_t3);
        bv_s45 = _mm_add_ps(bv_in10, _mm_mul_ps(v128_CRTM_15_11, bv_in4));
        bv_s46 = _mm_add_ps(bv_s45, bv_s44);
        bv_s49 = _mm_sub_ps(bv_s47, bv_s48);
        bv_s50 = _mm_add_ps(bv_s46, bv_s49);
        bv_t12 = _mm_mul_ps(v128_CRTM_15_6, _mm_sub_ps(bv_s49, bv_s46));
        bv_s51 = _mm_sub_ps(bv_s40, _mm_mul_ps(v128_CRTM_15_5, bv_s50));

        // Output pt 2: X(1) & Output pt 3: X(2)
        v_out1 = _mm_add_ps(bv_s51, bv_t11);
        v_out2 = _mm_add_ps(bv_s43, bv_t12);
        curr_out = out + out_strides[1];
        STRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output pt 18: X(17) & Output pt 19: X(18)
        v_out17 = _mm_add_ps(bv_s40, bv_s50);
        v_out18 = _mm_add_ps(bv_s42, bv_s41);
        curr_out = out + out_strides[17];
        STRI_2x128_S(curr_out, v_out_stride, v_out17, v_out18);
        // Output pt 22: X(21) & Output pt 23: X(22)
        v_out21 = _mm_sub_ps(bv_s51, bv_t11);
        v_out22 = _mm_sub_ps(bv_s43, bv_t12);
        curr_out = out + out_strides[21];
        STRI_2x128_S(curr_out, v_out_stride, v_out21, v_out22);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (n & 2)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1, av_in2, av_in3;
        __m128 av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
               av_s9, av_s10, av_s11, av_s12, av_s13, av_s15, av_s16, av_s17,
               av_s19, av_s20, av_s21, av_s22, av_s23, av_s24, av_s25, av_s26,
               av_s27, av_s28, av_s29, av_s30, av_s31;
        __m128 av_t0, av_t1, av_t2, av_t3;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21,
               v_out22, v_out23, v_out24, v_out25, v_out26, v_out27, v_out28,
               v_out29;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_15_1 = _mm256_castps256_ps128(v_CRTM_15_1);
        __m128 v128_CRTM_15_2 = _mm256_castps256_ps128(v_CRTM_15_2);
        __m128 v128_CRTM_15_3 = _mm256_castps256_ps128(v_CRTM_15_3);
        __m128 v128_CRTM_15_4 = _mm256_castps256_ps128(v_CRTM_15_4);
        __m128 v128_CRTM_15_5 = _mm256_castps256_ps128(v_CRTM_15_5);
        __m128 v128_CRTM_15_6 = _mm256_castps256_ps128(v_CRTM_15_6);
        __m128 v128_CRTM_15_7 = _mm256_castps256_ps128(v_CRTM_15_7);
        __m128 v128_CRTM_15_8 = _mm256_castps256_ps128(v_CRTM_15_8);
        __m128 v128_CRTM_15_9 = _mm256_castps256_ps128(v_CRTM_15_9);
        __m128 v128_CRTM_15_10 = _mm256_castps256_ps128(v_CRTM_15_10);
        __m128 v128_CRTM_15_11 = _mm256_castps256_ps128(v_CRTM_15_11);
        __m128 v128_CRTM_15_12 = _mm256_castps256_ps128(v_CRTM_15_12);

        // Input point 1: x(0)
        curr_in = in + in_strides[0];
        LDHR_128_S(curr_in, v_in_stride, av_in0);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDHR_128_S(curr_in, v_in_stride, av_in1);
        // Input point 15: x(14)
        curr_in = in + in_strides[14];
        LDHR_128_S(curr_in, v_in_stride, av_in2);
        // Input point 25: x(24)
        curr_in = in + in_strides[24];
        LDHR_128_S(curr_in, v_in_stride, av_in3);

        av_s0 = _mm_add_ps(av_in1, av_in2);
        av_s1 = _mm_sub_ps(av_in1, av_in2);
        av_s2 = _mm_add_ps(av_in3, av_s0);
        av_s3 = _mm_sub_ps(av_in3, _mm_mul_ps(v128_CRTM_15_5, av_s0));

        // Input point 17: x(16)
        curr_in = in + in_strides[16];
        LDHR_128_S(curr_in, v_in_stride, av_in1);
        // Input point 27: x(26)
        curr_in = in + in_strides[26];
        LDHR_128_S(curr_in, v_in_stride, av_in2);
        // Input point 7: x(6)
        curr_in = in + in_strides[6];
        LDHR_128_S(curr_in, v_in_stride, av_in3);

        av_s0 = _mm_add_ps(av_in1, av_in2);
        av_s4 = _mm_sub_ps(av_in2, av_in1);
        av_s5 = _mm_add_ps(av_in3, av_s0);
        av_s6 = _mm_sub_ps(av_in3, _mm_mul_ps(v128_CRTM_15_5, av_s0));

        av_s23 = _mm_add_ps(av_s6, av_s3);
        av_s24 = _mm_sub_ps(av_s6, av_s3);
        av_s26 = _mm_add_ps(av_s4, av_s1);
        av_s27 = _mm_sub_ps(av_s4, av_s1);

        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDHR_128_S(curr_in, v_in_stride, av_in1);
        // Input point 23: x(22)
        curr_in = in + in_strides[22];
        LDHR_128_S(curr_in, v_in_stride, av_in2);
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDHR_128_S(curr_in, v_in_stride, av_in3);

        av_s0 = _mm_add_ps(av_in1, av_in2);
        av_s1 = _mm_sub_ps(av_in2, av_in1);
        av_s7 = _mm_add_ps(av_in3, av_s0);
        av_s3 = _mm_sub_ps(av_in3, _mm_mul_ps(v128_CRTM_15_5, av_s0));

        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDHR_128_S(curr_in, v_in_stride, av_in1);
        // Input point 29: x(28)
        curr_in = in + in_strides[28];
        LDHR_128_S(curr_in, v_in_stride, av_in2);
        // Input point 19: x(18)
        curr_in = in + in_strides[18];
        LDHR_128_S(curr_in, v_in_stride, av_in3);

        av_s0 = _mm_add_ps(av_in1, av_in2);
        av_s4 = _mm_sub_ps(av_in2, av_in1);
        av_s8 = _mm_add_ps(av_in3, av_s0);
        av_s6 = _mm_sub_ps(av_in3, _mm_mul_ps(v128_CRTM_15_5, av_s0));

        av_s28 = _mm_add_ps(av_s6, av_s3);
        av_s29 = _mm_sub_ps(av_s3, av_s6);
        av_s30 = _mm_add_ps(av_s4, av_s1);
        av_s31 = _mm_sub_ps(av_s1, av_s4);

        // Input point 11: x(10)
        curr_in = in + in_strides[10];
        LDHR_128_S(curr_in, v_in_stride, av_in1);
        // Input point 21: x(20)
        curr_in = in + in_strides[20];
        LDHR_128_S(curr_in, v_in_stride, av_in2);

        av_s0 = _mm_add_ps(av_in1, av_in2);
        av_t0 = _mm_mul_ps(v128_CRTM_15_6, _mm_sub_ps(av_in2, av_in1));
        av_s9 = _mm_add_ps(av_in0, av_s0);
        av_s3 = _mm_sub_ps(av_in0, _mm_mul_ps(v128_CRTM_15_5, av_s0));

        av_s11 = _mm_add_ps(av_s8, av_s7);
        av_s12 = _mm_add_ps(av_s2, av_s5);
        av_s19 = _mm_sub_ps(av_s5, av_s2);
        av_s20 = _mm_sub_ps(av_s8, av_s7);
        av_s13 = _mm_add_ps(av_s11, av_s12);
        av_t1 = _mm_mul_ps(v128_CRTM_15_1, _mm_sub_ps(av_s11, av_s12));
        av_s15 = _mm_sub_ps(av_s9, _mm_mul_ps(v128_CRTM_15_2, av_s13));

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(av_s9, av_s13);
        STHR_128_S(curr_out, v_out_stride, v_out0);

        // Output point 12: X(11)
        v_out11 = _mm_add_ps(av_s15, av_t1);
        // Output point 13: X(12)
        v_out12 = _mm_add_ps(_mm_mul_ps(v128_CRTM_15_3, av_s20),
                             _mm_mul_ps(v128_CRTM_15_4, av_s19));
        curr_out = out + out_strides[11];
        STHRI_2x128_S(curr_out, v_out_stride, v_out11, v_out12);

        // Output point 24: X(23)
        v_out23 = _mm_sub_ps(av_s15, av_t1);
        // Output point 25: X(24)
        v_out24 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_4, av_s20),
                             _mm_mul_ps(v128_CRTM_15_3, av_s19));
        curr_out = out + out_strides[23];
        STHRI_2x128_S(curr_out, v_out_stride, v_out23, v_out24);

        av_t2 = _mm_mul_ps(v128_CRTM_15_1, _mm_sub_ps(av_s28, av_s23));
        av_s13 = _mm_add_ps(av_s28, av_s23);

        // Output point 20: X(19)
        v_out19 = _mm_add_ps(av_s13, av_s3);

        av_t3 = _mm_mul_ps(v128_CRTM_15_9, _mm_add_ps(av_s30, av_s27));
        av_s17 = _mm_sub_ps(av_s30, av_s27);

        // Output point 21: X(20)
        v_out20 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_6, av_s17), av_t0);
        curr_out = out + out_strides[19];
        STHRI_2x128_S(curr_out, v_out_stride, v_out19, v_out20);

        av_s15 = _mm_sub_ps(av_s3, _mm_mul_ps(v128_CRTM_15_2, av_s13));
        av_s20 = _mm_add_ps(av_t0, _mm_mul_ps(v128_CRTM_15_10, av_s17));
        av_s21 = _mm_sub_ps(av_s15, av_t2);
        av_s25 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_8, av_s26),
                            _mm_mul_ps(v128_CRTM_15_7, av_s31));

        // Output point 4: X(3)
        v_out3 = _mm_add_ps(av_s21, av_s25);

        // Output point 16: X(15)
        v_out15 = _mm_sub_ps(av_s21, av_s25);

        av_s21 = _mm_add_ps(av_s15, av_t2);
        av_s23 = _mm_add_ps(_mm_mul_ps(v128_CRTM_15_8, av_s31),
                            _mm_mul_ps(v128_CRTM_15_7, av_s26));

        // Output point 28: X(27)
        v_out27 = _mm_add_ps(av_s21, av_s23);

        av_s22 = _mm_add_ps(av_s20, av_t3);
        av_s16 = _mm_add_ps(_mm_mul_ps(v128_CRTM_15_4, av_s29),
                            _mm_mul_ps(v128_CRTM_15_3, av_s24));

        // Output point 5: X(4)
        v_out4 = _mm_sub_ps(av_s22, av_s16);
        curr_out = out + out_strides[3];
        STHRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);
        // Output point 17: X(16)
        v_out16 = _mm_add_ps(av_s22, av_s16);
        curr_out = out + out_strides[15];
        STHRI_2x128_S(curr_out, v_out_stride, v_out15, v_out16);

        av_s22 = _mm_sub_ps(av_s20, av_t3);
        av_s10 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_3, av_s29),
                            _mm_mul_ps(v128_CRTM_15_4, av_s24));

        // Output point 29: X(28)
        v_out28 = _mm_add_ps(av_s22, av_s10);
        curr_out = out + out_strides[27];
        STHRI_2x128_S(curr_out, v_out_stride, v_out27, v_out28);

        // Output point 8: X(7)
        v_out7 = _mm_sub_ps(av_s21, av_s23);

        // Output point 9: X(8)
        v_out8 = _mm_sub_ps(av_s10, av_s22);
        curr_out = out + out_strides[7];
        STHRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8, bv_in9, bv_in10, bv_in11, bv_in12, bv_in13, bv_in14;
        __m128 bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
               bv_t10, bv_t11, bv_t12;
        __m128 bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
               bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17,
               bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24, bv_s25,
               bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32, bv_s33,
               bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40, bv_s41,
               bv_s42, bv_s43, bv_s44, bv_s45, bv_s46, bv_s47, bv_s48, bv_s49,
               bv_s50, bv_s51;

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
        // Input point 22: x(21)
        curr_in = in + in_strides[21];
        LDHR_128_S(curr_in, v_in_stride, bv_in10);
        // Input point 24: x(23)
        curr_in = in + in_strides[23];
        LDHR_128_S(curr_in, v_in_stride, bv_in11);
        // Input point 26: x(25)
        curr_in = in + in_strides[25];
        LDHR_128_S(curr_in, v_in_stride, bv_in12);
        // Input point 28: x(27)
        curr_in = in + in_strides[27];
        LDHR_128_S(curr_in, v_in_stride, bv_in13);
        // Input point 30: x(29)
        curr_in = in + in_strides[29];
        LDHR_128_S(curr_in, v_in_stride, bv_in14);

        bv_s1 = _mm_add_ps(bv_in1, bv_in4);
        bv_s2 = _mm_add_ps(bv_in7, bv_in13);
        bv_s3 = _mm_sub_ps(bv_in7, bv_in13);
        bv_s4 = _mm_add_ps(bv_in6, bv_in9);
        bv_s5 = _mm_add_ps(bv_in12, bv_in3);
        bv_s42 = _mm_add_ps(_mm_mul_ps(v128_CRTM_15_4, bv_s5),
                            _mm_mul_ps(v128_CRTM_15_3, bv_s4));
        bv_s28 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_4, bv_s4),
                            _mm_mul_ps(v128_CRTM_15_3, bv_s5));

        bv_s6 = _mm_add_ps(bv_in14, bv_in11);
        bv_s7 = _mm_add_ps(bv_in2, bv_in8);
        bv_t2  = _mm_mul_ps(v128_CRTM_15_1, _mm_sub_ps(bv_in14, bv_s7));
        bv_s47 = _mm_sub_ps(bv_t2, _mm_mul_ps(v128_CRTM_15_11, bv_in11));
        bv_s30 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_12, bv_in11), bv_in5);

        bv_s8 = _mm_sub_ps(bv_in2, bv_in8);
        bv_s9 = _mm_add_ps(bv_in9, bv_in3);
        bv_s10 = _mm_add_ps(bv_in6, bv_in12);
        bv_s26 = _mm_add_ps(_mm_mul_ps(v128_CRTM_15_4, bv_s6),
                            _mm_mul_ps(v128_CRTM_15_3, bv_s8));

        bv_s11 = _mm_sub_ps(bv_s9, bv_s10);
        bv_s12 = _mm_add_ps(bv_in1, bv_s2);
        bv_s13 = _mm_add_ps(bv_in14, bv_s7);
        bv_s48 = _mm_add_ps(_mm_mul_ps(v128_CRTM_15_2, bv_s13), bv_in5);
        bv_s14 = _mm_sub_ps(bv_in0, bv_s11);

        bv_s15 = _mm_add_ps(bv_in10, bv_in4);
        bv_s16 = _mm_sub_ps(bv_s15, bv_s12);
        bv_s17 = _mm_add_ps(bv_in11, bv_in5);
        bv_s18 = _mm_sub_ps(bv_s13, bv_s17);
        bv_s19 = _mm_add_ps(bv_s16, bv_s18);
        bv_s20 = _mm_add_ps(bv_in0, _mm_mul_ps(v128_CRTM_15_2, bv_s11));
        // Output pt 30: X(29)
        v_out29 = _mm_add_ps(bv_s19, bv_s14);
        curr_out = out + out_strides[29];
        STHR_128_S(curr_out, v_out_stride, v_out29);
        // Output pt 10: X(9) & Output pt 11: X(10)
        v_out9 = _mm_sub_ps(bv_s14, _mm_mul_ps(v128_CRTM_15_5, bv_s19));
        v_out10 = _mm_mul_ps(v128_CRTM_15_6, _mm_sub_ps(bv_s16, bv_s18));
        curr_out = out + out_strides[9];
        STHRI_2x128_S(curr_out, v_out_stride, v_out9, v_out10);

        bv_s21 = _mm_add_ps(bv_in9, bv_in12);
        bv_s22 = _mm_add_ps(bv_in6, bv_in3);
        bv_s23 = _mm_sub_ps(bv_s21, bv_s22);
        bv_t1  = _mm_mul_ps(v128_CRTM_15_1, bv_s23);
        bv_s24 = _mm_add_ps(bv_s20, bv_t1);
        bv_s25 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_3, bv_s3),
                            _mm_mul_ps(v128_CRTM_15_4, bv_s1));

        bv_t6  = _mm_mul_ps(v128_CRTM_15_6, _mm_add_ps(bv_s25, bv_s26));
        bv_s27 = _mm_sub_ps(bv_s25, bv_s26);

        bv_s29 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_5, bv_s27), bv_s28);
        bv_s31 = _mm_add_ps(_mm_mul_ps(v128_CRTM_15_2, bv_s13), bv_t2);
        bv_s32 = _mm_sub_ps(bv_s30, bv_s31);

        bv_t3  = _mm_mul_ps(v128_CRTM_15_1, _mm_sub_ps(bv_in1, bv_s2));
        bv_s33 = _mm_add_ps(bv_in10, bv_t3);
        bv_t4  = _mm_mul_ps(v128_CRTM_15_2, bv_s12);
        bv_s34 = _mm_sub_ps(bv_t4, _mm_mul_ps(v128_CRTM_15_12, bv_in4));
        bv_s35 = _mm_add_ps(bv_s33, bv_s34);
        bv_s36 = _mm_add_ps(bv_s32, bv_s35);
        bv_t5  = _mm_mul_ps(v128_CRTM_15_6, _mm_sub_ps(bv_s32, bv_s35));
        // Output pt 6: X(5) & Output pt 7: X(6)
        v_out5 = _mm_add_ps(bv_s24, bv_s36);
        v_out6 = _mm_add_ps(bv_s28, bv_s27);
        curr_out = out + out_strides[5];
        STHRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);

        bv_s37 = _mm_sub_ps(bv_s24, _mm_mul_ps(v128_CRTM_15_5, bv_s36));
        // Output pt 14: X(13)& Output pt 15: X(14)
        v_out13 = _mm_sub_ps(bv_s37, bv_t6);
        v_out14 = _mm_add_ps(bv_t5, bv_s29);
        curr_out = out + out_strides[13];
        STHRI_2x128_S(curr_out, v_out_stride, v_out13, v_out14);
        // Output pt 26: X(25) & Output pt 27: X(26)
        v_out25 = _mm_add_ps(bv_s37, bv_t6);
        v_out26 = _mm_sub_ps(bv_t5, bv_s29);
        curr_out = out + out_strides[25];
        STHRI_2x128_S(curr_out, v_out_stride, v_out25, v_out26);

        bv_t7  = _mm_mul_ps(v128_CRTM_15_4, bv_s3);
        bv_t8  = _mm_mul_ps(v128_CRTM_15_3, bv_s1);
        bv_s38 = _mm_add_ps(bv_t8, bv_t7);
        bv_t9  = _mm_mul_ps(v128_CRTM_15_4, bv_s8);
        bv_t10 = _mm_mul_ps(v128_CRTM_15_3, bv_s6);
        bv_s39 = _mm_sub_ps(bv_t9, bv_t10);

        bv_t11 = _mm_mul_ps(v128_CRTM_15_6, _mm_add_ps(bv_s38, bv_s39));
        bv_s40 = _mm_sub_ps(bv_s20, bv_t1);
        bv_s41 = _mm_sub_ps(bv_s39, bv_s38);
        bv_s43 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_5, bv_s41), bv_s42);

        bv_s44 = _mm_sub_ps(bv_t4, bv_t3);
        bv_s45 = _mm_add_ps(bv_in10, _mm_mul_ps(v128_CRTM_15_11, bv_in4));
        bv_s46 = _mm_add_ps(bv_s45, bv_s44);
        bv_s49 = _mm_sub_ps(bv_s47, bv_s48);
        bv_s50 = _mm_add_ps(bv_s46, bv_s49);
        bv_t12 = _mm_mul_ps(v128_CRTM_15_6, _mm_sub_ps(bv_s49, bv_s46));
        bv_s51 = _mm_sub_ps(bv_s40, _mm_mul_ps(v128_CRTM_15_5, bv_s50));

        // Output pt 2: X(1) & Output pt 3: X(2)
        v_out1 = _mm_add_ps(bv_s51, bv_t11);
        v_out2 = _mm_add_ps(bv_s43, bv_t12);
        curr_out = out + out_strides[1];
        STHRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output pt 18: X(17) & Output pt 19: X(18)
        v_out17 = _mm_add_ps(bv_s40, bv_s50);
        v_out18 = _mm_add_ps(bv_s42, bv_s41);
        curr_out = out + out_strides[17];
        STHRI_2x128_S(curr_out, v_out_stride, v_out17, v_out18);
        // Output pt 22: X(21) & Output pt 23: X(22)
        v_out21 = _mm_sub_ps(bv_s51, bv_t11);
        v_out22 = _mm_sub_ps(bv_s43, bv_t12);
        curr_out = out + out_strides[21];
        STHRI_2x128_S(curr_out, v_out_stride, v_out21, v_out22);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        /* Standard DFT*/
        FLOAT a_in0, a_in1, a_in2, a_in3, a_s0, a_s1, a_s2, a_s3, a_s4, a_s5,
              a_s6, a_s7, a_s8, a_s9, a_s10, a_s11, a_s12, a_s13, a_s15, a_s16,
              a_s17, a_s19, a_s20, a_s21, a_s22, a_s23, a_s24, a_s25, a_s26,
              a_s27, a_s28, a_s29, a_s30, a_s31, a_t0, a_t1, a_t2, a_t3;

        a_in0 = *in;
        a_in1 = in[in_strides[4]];
        a_in2 = in[in_strides[14]];
        a_in3 = in[in_strides[24]];

        a_s0 = a_in1 + a_in2;
        a_s1 = a_in1 - a_in2;
        a_s2 = a_in3 + a_s0;
        a_s3 = a_in3 - (CRTM_15_5 * a_s0);

        a_in1 = in[in_strides[16]];
        a_in2 = in[in_strides[26]];
        a_in3 = in[in_strides[6]];

        a_s0 = a_in1 + a_in2;
        a_s4 = a_in2 - a_in1;
        a_s5 = a_in3 + a_s0;
        a_s6 = a_in3 - (CRTM_15_5 * a_s0);

        a_s23 = a_s6 + a_s3;
        a_s24 = a_s6 - a_s3;
        a_s26 = a_s4 + a_s1;
        a_s27 = a_s4 - a_s1;

        a_in1 = in[in_strides[2]];
        a_in2 = in[in_strides[22]];
        a_in3 = in[in_strides[12]];

        a_s0 = a_in1 + a_in2;
        a_s1 = a_in2 - a_in1;
        a_s7 = a_in3 + a_s0;
        a_s3 = a_in3 - (CRTM_15_5 * a_s0);

        a_in1 = in[in_strides[8]];
        a_in2 = in[in_strides[28]];
        a_in3 = in[in_strides[18]];

        a_s0 = a_in1 + a_in2;
        a_s4 = a_in2 - a_in1;
        a_s8 = a_in3 + a_s0;
        a_s6 = a_in3 - (CRTM_15_5 * a_s0);

        a_s28 = a_s6 + a_s3;
        a_s29 = a_s3 - a_s6;
        a_s30 = a_s4 + a_s1;
        a_s31 = a_s1 - a_s4;

        a_in1 = in[in_strides[10]];
        a_in2 = in[in_strides[20]];

        a_s0 = a_in1 + a_in2;
        a_t0 = CRTM_15_6 * (a_in2 - a_in1);
        a_s9 = a_in0 + a_s0;
        a_s3 = a_in0 - (CRTM_15_5 * a_s0);

        a_s11 = a_s8 + a_s7;
        a_s12 = a_s2 + a_s5;
        a_s19 = a_s5 - a_s2;
        a_s20 = a_s8 - a_s7;
        a_s13 = a_s11 + a_s12;
        a_t1 = CRTM_15_1 * (a_s11 - a_s12);
        a_s15 = a_s9 - (CRTM_15_2 * a_s13);

        *out = a_s9 + a_s13;

        out[out_strides[11]] = a_s15 + a_t1;
        out[out_strides[12]] = (CRTM_15_3 * a_s20) + (CRTM_15_4 * a_s19);

        out[out_strides[23]] = a_s15 - a_t1;
        out[out_strides[24]] = (CRTM_15_4 * a_s20) - (CRTM_15_3 * a_s19);

        a_t2 = CRTM_15_1 * (a_s28 - a_s23);
        a_s13 = a_s28 + a_s23;

        out[out_strides[19]] = a_s13 + a_s3;

        a_t3 = CRTM_15_9 * (a_s30 + a_s27);
        a_s17 = a_s30 - a_s27;

        out[out_strides[20]] = CRTM_15_6 * a_s17 - a_t0;

        a_s15 = a_s3 - (CRTM_15_2 * a_s13);
        a_s20 = a_t0 + (CRTM_15_10 * a_s17);
        a_s21 = a_s15 - a_t2;
        a_s25 = (CRTM_15_8 * a_s26) - (CRTM_15_7 * a_s31);

        out[out_strides[3]] = a_s21 + a_s25;
        out[out_strides[15]] = a_s21 - a_s25;

        a_s21 = a_s15 + a_t2;
        a_s23 = (CRTM_15_8 * a_s31) + (CRTM_15_7 * a_s26);

        out[out_strides[27]] = a_s21 + a_s23;

        a_s22 = a_s20 + a_t3;
        a_s16 = (CRTM_15_4 * a_s29) + (CRTM_15_3 * a_s24);

        out[out_strides[4]] = a_s22 - a_s16;
        out[out_strides[16]] = a_s22 + a_s16;

        a_s22 = a_s20 - a_t3;
        a_s10 = (CRTM_15_3 * a_s29) - (CRTM_15_4 * a_s24);
        out[out_strides[28]] = a_s22 + a_s10;

        out[out_strides[7]] = a_s21 - a_s23;
        out[out_strides[8]] = a_s10 - a_s22;

        /* Shifted DFT */
        FLOAT b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7, b_in8,
              b_in9, b_in10, b_in11, b_in12, b_in13, b_in14;
        FLOAT b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9, b_s10,
              b_s11, b_s12, b_s13, b_s14, b_s15, b_s16, b_s17, b_s18, b_s19,
              b_s20, b_s21, b_s22, b_s23, b_s24, b_s25, b_s26, b_s27, b_s28,
              b_s29, b_s30, b_s31, b_s32, b_s33, b_s34, b_s35, b_s36, b_s37,
              b_s38, b_s39, b_s40, b_s41, b_s42;
        FLOAT b_t0, b_t1, b_t2, b_t3, b_t4, b_t5, b_t6, b_t7, b_t8, b_t9, b_t10,
              b_t11, b_t12, b_t13, b_t14, b_t15, b_t16, b_t17, b_t18, b_t19;

        b_in0 = in[in_strides[1]];
        b_in1 = in[in_strides[3]];
        b_in2 = in[in_strides[5]];
        b_in3 = in[in_strides[7]];
        b_in4 = in[in_strides[9]];
        b_in5 = in[in_strides[11]];
        b_in6 = in[in_strides[13]];
        b_in7 = in[in_strides[15]];
        b_in8 = in[in_strides[17]];
        b_in9 = in[in_strides[19]];
        b_in10 = in[in_strides[21]];
        b_in11 = in[in_strides[23]];
        b_in12 = in[in_strides[25]];
        b_in13 = in[in_strides[27]];
        b_in14 = in[in_strides[29]];

        b_s0 = b_in1 + b_in4;
        b_s1 = b_in7 + b_in13;
        b_s2 = b_in7 - b_in13;
        b_s3 = b_in6 + b_in9;
        b_s4 = b_in12 + b_in3;
        b_s5 = b_in14 + b_in11;
        b_s6 = b_in2 + b_in8;
        b_s7 = b_in2 - b_in8;
        b_s8 = b_in9 + b_in3;
        b_s9 = b_in6 + b_in12;

        b_s10 = b_s8 - b_s9;
        b_s11 = b_in1 + b_s1;
        b_s12 = b_in14 + b_s6;
        b_s13 = b_in0 - b_s10;

        b_s14 = b_in10 + b_in4;
        b_s15 = b_s14 - b_s11;
        b_s16 = b_in11 + b_in5;
        b_s17 = b_s12 - b_s16;
        b_s18 = b_s15 + b_s17;
        b_s19 = b_in0 + CRTM_15_2 * (b_s10);
        b_s20 = b_in9 + b_in12;
        b_s21 = b_in6 + b_in3;
        b_s22 = b_s20 - b_s21;
        b_t0 = CRTM_15_1 * (b_s22);
        b_s23 = b_s19 + b_t0;
        b_t1 = CRTM_15_3 * b_s2 - CRTM_15_4 * b_s0;
        b_t2 = CRTM_15_4 * b_s5 + CRTM_15_3 * b_s7;
        b_s24 = b_t1 - b_t2;
        b_t3 = CRTM_15_4 * b_s3 - CRTM_15_3 * b_s4;

        b_t4 = CRTM_15_5 * b_s24 - b_t3;
        b_t5 = CRTM_15_12 * b_in11 - b_in5;
        b_t6 = CRTM_15_1 * (b_in14 - b_s6);
        b_t7 = CRTM_15_2 * b_s12 + b_t6;
        b_s25 = b_t5 - b_t7;
        b_t8 = CRTM_15_1 * (b_in1 - b_s1);
        b_s26 = b_in10 + b_t8;
        b_t9 = CRTM_15_2 * b_s11;
        b_s27 = b_t9 - CRTM_15_12 * b_in4;
        b_s28  = b_s26 + b_s27;
        b_s29 = b_s25 + b_s28;
        b_t10 = CRTM_15_6 * (b_s25 - b_s28);

        b_s30 = b_s23 - CRTM_15_5 * b_s29;
        b_t11 = CRTM_15_6 * (b_t1 + b_t2);

        b_t12 = CRTM_15_4 * b_s2;
        b_t13 = CRTM_15_3 * b_s0;
        b_s31 = b_t13 + b_t12;
        b_t14 = CRTM_15_4 * b_s7;
        b_t15 = CRTM_15_3 * b_s5;
        b_s32 = b_t14 - b_t15;
        b_t16 = CRTM_15_6 * (b_s31 + b_s32);
        b_s33 = b_s19 - b_t0;
        b_s34 = b_s32 - b_s31;
        b_s35 = (CRTM_15_4 * b_s4) + (CRTM_15_3 * b_s3);
        b_t17 = CRTM_15_5 * b_s34 - b_s35;
        b_s36 = b_t9 - b_t8;
        b_s37 = b_in10 + CRTM_15_11 * b_in4;
        b_s38 = b_s37 + b_s36;
        b_s39 = b_t6 - CRTM_15_11 * b_in11;
        b_t18 = CRTM_15_2 * b_s12 + b_in5;
        b_s40 = b_s39 - b_t18;
        b_s41 = b_s38 + b_s40;
        b_t19 = CRTM_15_6 * (b_s40 - b_s38);
        b_s42 = b_s33 - CRTM_15_5 * b_s41;

        out[out_strides[1]]  = b_s42 + b_t16;
        out[out_strides[2]]  = b_t17 + b_t19;
        out[out_strides[5]]  = b_s23 + b_s29;
        out[out_strides[6]]  = b_t3 + b_s24;
        out[out_strides[9]]  = b_s13 - CRTM_15_5 * b_s18;
        out[out_strides[10]] = CRTM_15_6 * (b_s15 - b_s17);
        out[out_strides[13]] = b_s30 - b_t11;
        out[out_strides[14]] = b_t10 + b_t4;
        out[out_strides[17]] = b_s33 + b_s41;
        out[out_strides[18]] = b_s35 + b_s34;
        out[out_strides[21]] = b_s42 - b_t16;
        out[out_strides[22]] = b_t17 - b_t19;
        out[out_strides[25]] = b_s30 + b_t11;
        out[out_strides[26]] = b_t10 - b_t4;
        out[out_strides[29]] = b_s18 + b_s13;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID r2hcf_rfft15avx256_fp32_bwd(VOID *in_real, VOID *in_imag,
                                        VOID *out_real, VOID *out_imag, INTP n,
                                        aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const FLOAT CRTM_15_1 = 1.118033988749894848204586834365638117720309180f;
    const FLOAT CRTM_15_2 = 0.500000000000000000000000000000000000000000000f;
    const FLOAT CRTM_15_3 = 1.902113032590307144232878666758764286811397268f;
    const FLOAT CRTM_15_4 = 1.175570504584946258337411909278145537195304875f;
    const FLOAT CRTM_15_5 = 2.000000000000000000000000000000000000000000000f;
    const FLOAT CRTM_15_6 = 0.250000000000000000000000000000000000000000000f;
    const FLOAT CRTM_15_7 =
        1.01807392091025412936433958497993431950210362069154f;
    const FLOAT CRTM_15_8 =
        1.64727820709266368541488232323193203275710390365294f;
    const FLOAT CRTM_15_9 =
        0.96824583655185421224048777314959976915574786128504f;
    const FLOAT CRTM_15_10 = 1.732050807568877293527446341505872366942805254f;
    const FLOAT CRTM_15_11 = 0.433012701892219323381861585376468091735701313f;
    const FLOAT CRTM_15_12 = 0.587785252292473129168705954639072768597652438f;
    const FLOAT CRTM_15_13 = 0.951056516295153572116439333379382143405698634f;
    const FLOAT CRTM_15_14 = 0.559016994374947424102293417182819058860154590f;

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
    INTP N = n >> 3;

    __m256 v_CRTM_15_1 = _mm256_broadcast_ss(&CRTM_15_1);
    __m256 v_CRTM_15_2 = _mm256_broadcast_ss(&CRTM_15_2);
    __m256 v_CRTM_15_3 = _mm256_broadcast_ss(&CRTM_15_3);
    __m256 v_CRTM_15_4 = _mm256_broadcast_ss(&CRTM_15_4);
    __m256 v_CRTM_15_5 = _mm256_broadcast_ss(&CRTM_15_5);
    __m256 v_CRTM_15_6 = _mm256_broadcast_ss(&CRTM_15_6);
    __m256 v_CRTM_15_7 = _mm256_broadcast_ss(&CRTM_15_7);
    __m256 v_CRTM_15_8 = _mm256_broadcast_ss(&CRTM_15_8);
    __m256 v_CRTM_15_9 = _mm256_broadcast_ss(&CRTM_15_9);
    __m256 v_CRTM_15_10 = _mm256_broadcast_ss(&CRTM_15_10);
    __m256 v_CRTM_15_11 = _mm256_broadcast_ss(&CRTM_15_11);
    __m256 v_CRTM_15_12 = _mm256_broadcast_ss(&CRTM_15_12);
    __m256 v_CRTM_15_13 = _mm256_broadcast_ss(&CRTM_15_13);
    __m256 v_CRTM_15_14 = _mm256_broadcast_ss(&CRTM_15_14);

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m256 av_in0, av_in1, av_in2, av_in3;
        __m256 av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
               av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16,
               av_s17, av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24,
               av_s25, av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32,
               av_s33, av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40;
        __m256 av_t0, av_t1, av_t2, av_t4, av_t5;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21,
               v_out22, v_out23, v_out24, v_out25, v_out26, v_out27, v_out28,
               v_out29;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        curr_in = in + in_strides[0];
        LDR_256_S(curr_in, v_in_stride, av_in0);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_256_S(curr_in, v_in_stride, av_in1);
        // Input point 28: x(27)
        curr_in = in + in_strides[27];
        LDR_256_S(curr_in, v_in_stride, av_in2);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_256_S(curr_in, v_in_stride, av_in3);

        av_s0 = _mm256_add_ps(av_in1, av_in2);
        av_s1 = _mm256_sub_ps(av_in1, av_in2);
        av_s2 = _mm256_add_ps(av_in3, av_s0);
        av_s3 = _mm256_sub_ps(av_in3, _mm256_mul_ps(v_CRTM_15_2, av_s0));

        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDR_256_S(curr_in, v_in_stride, av_in1);
        // Input point 29: x(28)
        curr_in = in + in_strides[28];
        LDR_256_S(curr_in, v_in_stride, av_in2);
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDR_256_S(curr_in, v_in_stride, av_in3);

        av_s0 = _mm256_add_ps(av_in1, av_in2);
        av_s4 = _mm256_sub_ps(av_in1, av_in2);
        av_s5 = _mm256_sub_ps(av_s0, av_in3);
        av_s6 = _mm256_add_ps(av_in3, _mm256_mul_ps(v_CRTM_15_2, av_s0));

        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_256_S(curr_in, v_in_stride, av_in1);
        // Input point 16: x(15)
        curr_in = in + in_strides[15];
        LDR_256_S(curr_in, v_in_stride, av_in2);
        // Input point 24: x(23)
        curr_in = in + in_strides[23];
        LDR_256_S(curr_in, v_in_stride, av_in3);

        av_s0 = _mm256_add_ps(av_in1, av_in2);
        av_s7 = _mm256_sub_ps(av_in2, av_in1);
        av_s8 = _mm256_add_ps(av_in3, av_s0);
        av_s9 = _mm256_sub_ps(av_in3, _mm256_mul_ps(v_CRTM_15_2, av_s0));

        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_256_S(curr_in, v_in_stride, av_in1);
        // Input point 17: x(16)
        curr_in = in + in_strides[16];
        LDR_256_S(curr_in, v_in_stride, av_in2);
        // Input point 25: x(24)
        curr_in = in + in_strides[24];
        LDR_256_S(curr_in, v_in_stride, av_in3);

        av_s0 = _mm256_sub_ps(av_in1, av_in2);
        av_s10 = _mm256_add_ps(av_in2, av_in1);
        av_s11 = _mm256_add_ps(av_in3, av_s0);
        av_s12 = _mm256_sub_ps(av_in3, _mm256_mul_ps(v_CRTM_15_2, av_s0));

        // Input point 20: x(19)
        curr_in = in + in_strides[19];
        LDR_256_S(curr_in, v_in_stride, av_in1);
        // Input point 21: x(20)
        curr_in = in + in_strides[20];
        LDR_256_S(curr_in, v_in_stride, av_in2);

        av_s13 = _mm256_add_ps(av_in0, _mm256_mul_ps(v_CRTM_15_5, av_in1));
        av_s14 = _mm256_sub_ps(av_in0, av_in1);
        av_s15 = _mm256_add_ps(av_s8, av_s2);
        av_t0 = _mm256_mul_ps(v_CRTM_15_1, _mm256_sub_ps(av_s8, av_s2));
        av_s16 = _mm256_sub_ps(av_s13, _mm256_mul_ps(v_CRTM_15_2, av_s15));

        // Output point 1: X(0)
        v_out0 = _mm256_add_ps(av_s13, _mm256_mul_ps(v_CRTM_15_5, av_s15));
        STR_256_S(curr_out, v_out_stride, v_out0);

        av_s17 = _mm256_add_ps(_mm256_mul_ps(v_CRTM_15_3, av_s11),
                               _mm256_mul_ps(v_CRTM_15_4, av_s5));
        av_s18 = _mm256_add_ps(av_s16, av_t0);

        // Output point 7: X(6)
        v_out6 = _mm256_sub_ps(av_s18, av_s17);
        curr_out = out + out_strides[6];
        STR_256_S(curr_out, v_out_stride, v_out6);

        // Output point 25: X(24)
        v_out24 = _mm256_add_ps(av_s18, av_s17);
        curr_out = out + out_strides[24];
        STR_256_S(curr_out, v_out_stride, v_out24);

        av_s19 = _mm256_sub_ps(av_s16, av_t0);
        av_s20 = _mm256_sub_ps(_mm256_mul_ps(v_CRTM_15_4, av_s11),
                               _mm256_mul_ps(v_CRTM_15_3, av_s5));

        // Output point 13: X(12)
        v_out12 = _mm256_sub_ps(av_s19, av_s20);
        curr_out = out + out_strides[12];
        STR_256_S(curr_out, v_out_stride, v_out12);

        // Output point 19: X(18)
        v_out18 = _mm256_add_ps(av_s19, av_s20);
        curr_out = out + out_strides[18];
        STR_256_S(curr_out, v_out_stride, v_out18);

        av_t1 = _mm256_mul_ps(v_CRTM_15_1, _mm256_sub_ps(av_s9, av_s3));
        av_t2 = _mm256_mul_ps(v_CRTM_15_9, _mm256_add_ps(av_s4, av_s10));
        av_s21 = _mm256_add_ps(av_s9, av_s3);
        av_s22 = _mm256_sub_ps(av_s4, av_s10);
        av_s40 = _mm256_add_ps(_mm256_mul_ps(v_CRTM_15_5, av_s21), av_s14);
        av_t4 = _mm256_mul_ps(v_CRTM_15_10, _mm256_add_ps(av_s22, av_in2));

        // Output point 11: X(10)
        v_out10 = _mm256_add_ps(av_s40, av_t4);
        curr_out = out + out_strides[10];
        STR_256_S(curr_out, v_out_stride, v_out10);

        // Output point 21: X(20)
        v_out20 = _mm256_sub_ps(av_s40, av_t4);
        curr_out = out + out_strides[20];
        STR_256_S(curr_out, v_out_stride, v_out20);

        av_s23 = _mm256_sub_ps(av_s14, _mm256_mul_ps(v_CRTM_15_2, av_s21));
        av_t5 = _mm256_mul_ps(v_CRTM_15_10,
                _mm256_sub_ps( _mm256_mul_ps(v_CRTM_15_6, av_s22), av_in2));
        av_s24 = _mm256_add_ps(av_s23, av_t5);
        av_s25 = _mm256_sub_ps(av_s23, av_t5);
        av_s26 = _mm256_sub_ps(av_t2, av_t1);
        av_s27 = _mm256_add_ps(av_t1, av_t2);

        av_s28 = _mm256_sub_ps(av_s24, av_s27);
        av_s29 = _mm256_add_ps(_mm256_mul_ps(v_CRTM_15_3, av_s6),
                               _mm256_mul_ps(v_CRTM_15_4, av_s12));
        av_s30 = _mm256_sub_ps(_mm256_mul_ps(v_CRTM_15_8, av_s1),
                               _mm256_mul_ps(v_CRTM_15_7, av_s7));
        av_s31 = _mm256_sub_ps(av_s30, av_s29);
        av_s32 = _mm256_add_ps(av_s29, av_s30);

        // Output point 3: X(2)
        v_out2 = _mm256_add_ps(av_s28, av_s31);
        curr_out = out + out_strides[2];
        STR_256_S(curr_out, v_out_stride, v_out2);

        // Output point 9: X(8)
        v_out8 = _mm256_sub_ps(av_s28, av_s31);
        curr_out = out + out_strides[8];
        STR_256_S(curr_out, v_out_stride, v_out8);

        av_s33 = _mm256_add_ps(av_s24, av_s27);
        av_s34 = _mm256_add_ps(_mm256_mul_ps(v_CRTM_15_8, av_s7),
                               _mm256_mul_ps(v_CRTM_15_7, av_s1));
        av_s35 = _mm256_sub_ps(_mm256_mul_ps(v_CRTM_15_3, av_s12),
                               _mm256_mul_ps(v_CRTM_15_4, av_s6));
        av_s36 = _mm256_add_ps(av_s35, av_s34);

        // Output point 15: X(14)
        v_out14 = _mm256_add_ps(av_s33, av_s36);
        curr_out = out + out_strides[14];
        STR_256_S(curr_out, v_out_stride, v_out14);

        // Output point 27: X(26)
        v_out26 = _mm256_sub_ps(av_s33, av_s36);
        curr_out = out + out_strides[26];
        STR_256_S(curr_out, v_out_stride, v_out26);

        av_s37 = _mm256_sub_ps(av_s25, av_s26);
        av_s38 = _mm256_sub_ps(av_s35, av_s34);

        // Output point 5: X(4)
        v_out4 = _mm256_add_ps(av_s37, av_s38);
        curr_out = out + out_strides[4];
        STR_256_S(curr_out, v_out_stride, v_out4);

        // Output point 17: X(16)
        v_out16 = _mm256_sub_ps(av_s37, av_s38);
        curr_out = out + out_strides[16];
        STR_256_S(curr_out, v_out_stride, v_out16);

        av_s39 = _mm256_add_ps(av_s25, av_s26);

        // Output point 23: X(22)
        v_out22 = _mm256_sub_ps(av_s39, av_s32);
        curr_out = out + out_strides[22];
        STR_256_S(curr_out, v_out_stride, v_out22);

        // Output point 29: X(28)
        v_out28 = _mm256_add_ps(av_s39, av_s32);
        curr_out = out + out_strides[28];
        STR_256_S(curr_out, v_out_stride, v_out28);

        /* Shifted DFT */
        __m256 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8, bv_in9, bv_in10, bv_in11, bv_in12, bv_in13, bv_in14;
        __m256 bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
               bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17,
               bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24, bv_s25,
               bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32, bv_s33,
               bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40, bv_s41,
               bv_s42, bv_s43, bv_s44, bv_s45, bv_s46, bv_s47, bv_s48, bv_s49;
        __m256 bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
               bv_t10, bv_t11, bv_t12, bv_t13, bv_t14, bv_t15, bv_t16, bv_t17,
               bv_t18, bv_t19, bv_t20, bv_t21, bv_t22, bv_t23, bv_t24, bv_t25,
               bv_t26;

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
        // Input point 22: x(21) & Input point 23: x(22)
        curr_in = in + in_strides[21];
        LDRI_2x256_S(curr_in, v_in_stride, bv_in10, bv_in11);
        // Input point 26: x(25) & Input point 27: x(26)
        curr_in = in + in_strides[25];
        LDRI_2x256_S(curr_in, v_in_stride, bv_in12, bv_in13);
        // Input point 30: x(29)
        curr_in = in + in_strides[29];
        LDR_256_S(curr_in, v_in_stride, bv_in14);

        bv_s1 = _mm256_add_ps(bv_in0, bv_in10);
        bv_s2 = _mm256_sub_ps(bv_in0, bv_in10);
        bv_s3 = _mm256_add_ps(bv_in1, bv_in11);
        bv_s4 = _mm256_sub_ps(bv_in1, bv_in11);
        bv_s5 = _mm256_add_ps(bv_in6, bv_in12);
        bv_s6 = _mm256_sub_ps(bv_in6, bv_in12);
        bv_s7 = _mm256_add_ps(bv_in7, bv_in13);
        bv_s8 = _mm256_sub_ps(bv_in7, bv_in13);
        bv_s9 = _mm256_add_ps(bv_in2, bv_in8);
        bv_s10 = _mm256_sub_ps(bv_in2, bv_in8);

        bv_s11 = _mm256_sub_ps(bv_s1, bv_s5);
        bv_t1 = _mm256_mul_ps(v_CRTM_15_14, bv_s11);
        bv_t2 = _mm256_mul_ps(v_CRTM_15_1, bv_s10);
        bv_t3 = _mm256_mul_ps(v_CRTM_15_5, bv_t1);
        bv_s12 = _mm256_sub_ps(bv_t3, bv_t2);
        bv_t4 = _mm256_mul_ps(v_CRTM_15_2, bv_s9);
        bv_s13 = _mm256_sub_ps(bv_in14, bv_t4);
        bv_s14 = _mm256_add_ps(bv_s1, bv_s5);

        bv_t5 = _mm256_mul_ps(v_CRTM_15_6, bv_s14);
        bv_s15 = _mm256_sub_ps(bv_in4, bv_t5);
        bv_t6 = _mm256_mul_ps(v_CRTM_15_5, bv_s15);
        bv_s16 = _mm256_add_ps(bv_s13, bv_t6);
        bv_s17 = _mm256_sub_ps(bv_s12, bv_s16);
        bv_s18 = _mm256_add_ps(bv_s12, bv_s16);

        bv_t7 = _mm256_mul_ps(v_CRTM_15_13, bv_s8);
        bv_t8 = _mm256_mul_ps(v_CRTM_15_12, bv_s3);
        bv_t9 = _mm256_mul_ps(v_CRTM_15_4, bv_in9);
        bv_t10 = _mm256_mul_ps(v_CRTM_15_3, bv_in3);
        bv_s19 = _mm256_sub_ps(bv_t7, bv_t8);
        bv_t11 = _mm256_mul_ps(v_CRTM_15_5, bv_s19);
        bv_s20 = _mm256_sub_ps(bv_t9, bv_t10);
        bv_s21 = _mm256_add_ps(bv_t11, bv_s20);

        // Output pt 8: X(7)
        v_out7 = _mm256_add_ps(bv_s21, bv_s17);
        curr_out = out + out_strides[7];
        STR_256_S(curr_out, v_out_stride, v_out7);

        // Output pt 26: X(25)
        v_out25 = _mm256_sub_ps(bv_s21, bv_s17);
        curr_out = out + out_strides[25];
        STR_256_S(curr_out, v_out_stride, v_out25);

        bv_s22 = _mm256_add_ps(bv_s4, bv_s7);
        bv_s23 = _mm256_sub_ps(bv_in5, bv_s22);
        bv_t12 = _mm256_mul_ps(v_CRTM_15_10, bv_s23);
        bv_s24 = _mm256_add_ps(bv_s14, bv_in4);
        bv_t13 = _mm256_mul_ps(v_CRTM_15_5, bv_s9);
        bv_s25 = _mm256_add_ps(bv_in14, bv_t13);
        bv_s26 = _mm256_sub_ps(bv_s24, bv_s25);
        // Output pt 12: X(11)
        v_out11 = _mm256_add_ps(bv_t12, bv_s26);
        curr_out = out + out_strides[11];
        STR_256_S(curr_out, v_out_stride, v_out11);
        // Output pt 22: X(21)
        v_out21 = _mm256_sub_ps(bv_t12, bv_s26);
        curr_out = out + out_strides[21];
        STR_256_S(curr_out, v_out_stride, v_out21);
        bv_t19 = _mm256_mul_ps(v_CRTM_15_5, bv_s24);
        // Output pt 2: X(1)
        v_out1 = _mm256_add_ps(bv_t19, bv_s25);
        curr_out = out + out_strides[1];
        STR_256_S(curr_out, v_out_stride, v_out1);

        bv_t14 = _mm256_mul_ps(v_CRTM_15_4, bv_in3);
        bv_t15 = _mm256_mul_ps(v_CRTM_15_3, bv_in9);
        bv_s27 = _mm256_add_ps(bv_t14, bv_t15);

        bv_t16 = _mm256_mul_ps(v_CRTM_15_13, bv_s3);
        bv_t17 = _mm256_mul_ps(v_CRTM_15_12, bv_s8);
        bv_s28 = _mm256_add_ps(bv_t16, bv_t17);
        bv_t18 = _mm256_mul_ps(v_CRTM_15_5, bv_s28);
        bv_s29 = _mm256_sub_ps(bv_s27, bv_t18);
        // Output pt 14: X(13)
        v_out13 = _mm256_add_ps(bv_s29, bv_s18);
        curr_out = out + out_strides[13];
        STR_256_S(curr_out, v_out_stride, v_out13);
        // Output pt 20: X(19)
        v_out19 = _mm256_sub_ps(bv_s29, bv_s18);
        curr_out = out + out_strides[19];
        STR_256_S(curr_out, v_out_stride, v_out19);

        bv_t20 = _mm256_mul_ps(v_CRTM_15_11, bv_s22);
        bv_t21 = _mm256_mul_ps(v_CRTM_15_10, bv_in5);
        bv_s30 = _mm256_add_ps(bv_t20, bv_t21);
        bv_s31 = _mm256_sub_ps(bv_s4, bv_s7);
        bv_t22 = _mm256_mul_ps(v_CRTM_15_9, bv_s31);
        bv_s32 = _mm256_sub_ps(bv_t22, bv_s30);
        bv_s33 = _mm256_add_ps(bv_s30, bv_t22);
        bv_s34 = _mm256_add_ps(bv_t1, bv_t2);
        bv_s35 = _mm256_sub_ps(bv_s13, bv_s15);
        bv_s36 = _mm256_add_ps(bv_s34, bv_s35);
        bv_s37 = _mm256_sub_ps(bv_s34, bv_s35);
        bv_s38 = _mm256_add_ps(bv_s28, bv_s27);

        bv_t23 = _mm256_mul_ps(v_CRTM_15_8, bv_s2);
        bv_t24 = _mm256_mul_ps(v_CRTM_15_7, bv_s6);
        bv_s39 = _mm256_add_ps(bv_t23, bv_t24);
        bv_t25 = _mm256_mul_ps(v_CRTM_15_8, bv_s6);
        bv_t26 = _mm256_mul_ps(v_CRTM_15_7, bv_s2);
        bv_s40 = _mm256_sub_ps(bv_t25, bv_t26);
        bv_s41 = _mm256_sub_ps(bv_s19, bv_s20);
        bv_s42 = _mm256_sub_ps(bv_s37, bv_s38);
        bv_s43 = _mm256_add_ps(bv_s37, bv_s38);
        bv_s44 = _mm256_add_ps(bv_s39, bv_s32);
        bv_s45 = _mm256_sub_ps(bv_s39, bv_s32);
        bv_s46 = _mm256_sub_ps(bv_s40, bv_s33);
        bv_s47 = _mm256_add_ps(bv_s33, bv_s40);
        bv_s48 = _mm256_add_ps(bv_s36, bv_s41);
        bv_s49 = _mm256_sub_ps(bv_s36, bv_s41);

        // Output pt 4: X(3)
        v_out3 = _mm256_add_ps(bv_s42, bv_s44);
        curr_out = out + out_strides[3];
        STR_256_S(curr_out, v_out_stride, v_out3);
        // Output pt 24: X(23)
        v_out23 = _mm256_sub_ps(bv_s42, bv_s44);
        curr_out = out + out_strides[23];
        STR_256_S(curr_out, v_out_stride, v_out23);
        // Output pt 18: X(17)
        v_out17 = _mm256_add_ps(bv_s46, bv_s48);
        curr_out = out + out_strides[17];
        STR_256_S(curr_out, v_out_stride, v_out17);
        // Output pt 28: X(27)
        v_out27 = _mm256_sub_ps(bv_s46, bv_s48);
        curr_out = out + out_strides[27];
        STR_256_S(curr_out, v_out_stride, v_out27);
        // Output pt 6: X(5)
        v_out5 = _mm256_sub_ps(bv_s49, bv_s47);
        curr_out = out + out_strides[5];
        STR_256_S(curr_out, v_out_stride, v_out5);
        // Output pt 16: X(15)
        v_out15 = NEGATE_256_S(_mm256_add_ps(bv_s47, bv_s49));
        curr_out = out + out_strides[15];
        STR_256_S(curr_out, v_out_stride, v_out15);
        // Output pt 10: X(9)
        v_out9 = _mm256_sub_ps(bv_s45, bv_s43);
        curr_out = out + out_strides[9];
        STR_256_S(curr_out, v_out_stride, v_out9);
        // Output pt 30: X(29)
        v_out29 = NEGATE_256_S(_mm256_add_ps(bv_s43, bv_s45));
        curr_out = out + out_strides[29];
        STR_256_S(curr_out, v_out_stride, v_out29);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (n & 4)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1, av_in2, av_in3;
        __m128 av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
               av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16,
               av_s17, av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24,
               av_s25, av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32,
               av_s33, av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40;
        __m128 av_t0, av_t1, av_t2, av_t4, av_t5;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21,
               v_out22, v_out23, v_out24, v_out25, v_out26, v_out27, v_out28,
               v_out29;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_15_1 = _mm256_castps256_ps128(v_CRTM_15_1);
        __m128 v128_CRTM_15_2 = _mm256_castps256_ps128(v_CRTM_15_2);
        __m128 v128_CRTM_15_3 = _mm256_castps256_ps128(v_CRTM_15_3);
        __m128 v128_CRTM_15_4 = _mm256_castps256_ps128(v_CRTM_15_4);
        __m128 v128_CRTM_15_5 = _mm256_castps256_ps128(v_CRTM_15_5);
        __m128 v128_CRTM_15_6 = _mm256_castps256_ps128(v_CRTM_15_6);
        __m128 v128_CRTM_15_7 = _mm256_castps256_ps128(v_CRTM_15_7);
        __m128 v128_CRTM_15_8 = _mm256_castps256_ps128(v_CRTM_15_8);
        __m128 v128_CRTM_15_9 = _mm256_castps256_ps128(v_CRTM_15_9);
        __m128 v128_CRTM_15_10 = _mm256_castps256_ps128(v_CRTM_15_10);
        __m128 v128_CRTM_15_11 = _mm256_castps256_ps128(v_CRTM_15_11);
        __m128 v128_CRTM_15_12 = _mm256_castps256_ps128(v_CRTM_15_12);
        __m128 v128_CRTM_15_13 = _mm256_castps256_ps128(v_CRTM_15_13);
        __m128 v128_CRTM_15_14 = _mm256_castps256_ps128(v_CRTM_15_14);

        // Input point 1: x(0)
        curr_in = in + in_strides[0];
        LDR_128_S(curr_in, v_in_stride, av_in0);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_128_S(curr_in, v_in_stride, av_in1);
        // Input point 28: x(27)
        curr_in = in + in_strides[27];
        LDR_128_S(curr_in, v_in_stride, av_in2);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_128_S(curr_in, v_in_stride, av_in3);

        av_s0 = _mm_add_ps(av_in1, av_in2);
        av_s1 = _mm_sub_ps(av_in1, av_in2);
        av_s2 = _mm_add_ps(av_in3, av_s0);
        av_s3 = _mm_sub_ps(av_in3, _mm_mul_ps(v128_CRTM_15_2, av_s0));

        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDR_128_S(curr_in, v_in_stride, av_in1);
        // Input point 29: x(28)
        curr_in = in + in_strides[28];
        LDR_128_S(curr_in, v_in_stride, av_in2);
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDR_128_S(curr_in, v_in_stride, av_in3);

        av_s0 = _mm_add_ps(av_in1, av_in2);
        av_s4 = _mm_sub_ps(av_in1, av_in2);
        av_s5 = _mm_sub_ps(av_s0, av_in3);
        av_s6 = _mm_add_ps(av_in3, _mm_mul_ps(v128_CRTM_15_2, av_s0));

        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_128_S(curr_in, v_in_stride, av_in1);
        // Input point 16: x(15)
        curr_in = in + in_strides[15];
        LDR_128_S(curr_in, v_in_stride, av_in2);
        // Input point 24: x(23)
        curr_in = in + in_strides[23];
        LDR_128_S(curr_in, v_in_stride, av_in3);

        av_s0 = _mm_add_ps(av_in1, av_in2);
        av_s7 = _mm_sub_ps(av_in2, av_in1);
        av_s8 = _mm_add_ps(av_in3, av_s0);
        av_s9 = _mm_sub_ps(av_in3, _mm_mul_ps(v128_CRTM_15_2, av_s0));

        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_128_S(curr_in, v_in_stride, av_in1);
        // Input point 17: x(16)
        curr_in = in + in_strides[16];
        LDR_128_S(curr_in, v_in_stride, av_in2);
        // Input point 25: x(24)
        curr_in = in + in_strides[24];
        LDR_128_S(curr_in, v_in_stride, av_in3);

        av_s0 = _mm_sub_ps(av_in1, av_in2);
        av_s10 = _mm_add_ps(av_in2, av_in1);
        av_s11 = _mm_add_ps(av_in3, av_s0);
        av_s12 = _mm_sub_ps(av_in3, _mm_mul_ps(v128_CRTM_15_2, av_s0));

        // Input point 20: x(19)
        curr_in = in + in_strides[19];
        LDR_128_S(curr_in, v_in_stride, av_in1);
        // Input point 21: x(20)
        curr_in = in + in_strides[20];
        LDR_128_S(curr_in, v_in_stride, av_in2);

        av_s13 = _mm_add_ps(av_in0, _mm_mul_ps(v128_CRTM_15_5, av_in1));
        av_s14 = _mm_sub_ps(av_in0, av_in1);
        av_s15 = _mm_add_ps(av_s8, av_s2);
        av_t0 = _mm_mul_ps(v128_CRTM_15_1, _mm_sub_ps(av_s8, av_s2));
        av_s16 = _mm_sub_ps(av_s13, _mm_mul_ps(v128_CRTM_15_2, av_s15));

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(av_s13, _mm_mul_ps(v128_CRTM_15_5, av_s15));
        STR_128_S(curr_out, v_out_stride, v_out0);

        av_s17 = _mm_add_ps(_mm_mul_ps(v128_CRTM_15_3, av_s11),
                            _mm_mul_ps(v128_CRTM_15_4, av_s5));
        av_s18 = _mm_add_ps(av_s16, av_t0);

        // Output point 7: X(6)
        v_out6 = _mm_sub_ps(av_s18, av_s17);
        curr_out = out + out_strides[6];
        STR_128_S(curr_out, v_out_stride, v_out6);

        // Output point 25: X(24)
        v_out24 = _mm_add_ps(av_s18, av_s17);
        curr_out = out + out_strides[24];
        STR_128_S(curr_out, v_out_stride, v_out24);

        av_s19 = _mm_sub_ps(av_s16, av_t0);
        av_s20 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_4, av_s11),
                            _mm_mul_ps(v128_CRTM_15_3, av_s5));

        // Output point 13: X(12)
        v_out12 = _mm_sub_ps(av_s19, av_s20);
        curr_out = out + out_strides[12];
        STR_128_S(curr_out, v_out_stride, v_out12);

        // Output point 19: X(18)
        v_out18 = _mm_add_ps(av_s19, av_s20);
        curr_out = out + out_strides[18];
        STR_128_S(curr_out, v_out_stride, v_out18);

        av_t1 = _mm_mul_ps(v128_CRTM_15_1, _mm_sub_ps(av_s9, av_s3));
        av_t2 = _mm_mul_ps(v128_CRTM_15_9, _mm_add_ps(av_s4, av_s10));
        av_s21 = _mm_add_ps(av_s9, av_s3);
        av_s22 = _mm_sub_ps(av_s4, av_s10);
        av_s40 = _mm_add_ps(_mm_mul_ps(v128_CRTM_15_5, av_s21), av_s14);
        av_t4 = _mm_mul_ps(v128_CRTM_15_10, _mm_add_ps(av_s22, av_in2));

        // Output point 11: X(10)
        v_out10 = _mm_add_ps(av_s40, av_t4);
        curr_out = out + out_strides[10];
        STR_128_S(curr_out, v_out_stride, v_out10);

        // Output point 21: X(20)
        v_out20 = _mm_sub_ps(av_s40, av_t4);
        curr_out = out + out_strides[20];
        STR_128_S(curr_out, v_out_stride, v_out20);

        av_s23 = _mm_sub_ps(av_s14, _mm_mul_ps(v128_CRTM_15_2, av_s21));
        av_t5 = _mm_mul_ps(v128_CRTM_15_10,
                _mm_sub_ps( _mm_mul_ps(v128_CRTM_15_6, av_s22), av_in2));
        av_s24 = _mm_add_ps(av_s23, av_t5);
        av_s25 = _mm_sub_ps(av_s23, av_t5);
        av_s26 = _mm_sub_ps(av_t2, av_t1);
        av_s27 = _mm_add_ps(av_t1, av_t2);

        av_s28 = _mm_sub_ps(av_s24, av_s27);
        av_s29 = _mm_add_ps(_mm_mul_ps(v128_CRTM_15_3, av_s6),
                            _mm_mul_ps(v128_CRTM_15_4, av_s12));
        av_s30 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_8, av_s1),
                            _mm_mul_ps(v128_CRTM_15_7, av_s7));
        av_s31 = _mm_sub_ps(av_s30, av_s29);
        av_s32 = _mm_add_ps(av_s29, av_s30);

        // Output point 3: X(2)
        v_out2 = _mm_add_ps(av_s28, av_s31);
        curr_out = out + out_strides[2];
        STR_128_S(curr_out, v_out_stride, v_out2);

        // Output point 9: X(8)
        v_out8 = _mm_sub_ps(av_s28, av_s31);
        curr_out = out + out_strides[8];
        STR_128_S(curr_out, v_out_stride, v_out8);

        av_s33 = _mm_add_ps(av_s24, av_s27);
        av_s34 = _mm_add_ps(_mm_mul_ps(v128_CRTM_15_8, av_s7),
                            _mm_mul_ps(v128_CRTM_15_7, av_s1));
        av_s35 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_3, av_s12),
                            _mm_mul_ps(v128_CRTM_15_4, av_s6));
        av_s36 = _mm_add_ps(av_s35, av_s34);

        // Output point 15: X(14)
        v_out14 = _mm_add_ps(av_s33, av_s36);
        curr_out = out + out_strides[14];
        STR_128_S(curr_out, v_out_stride, v_out14);

        // Output point 27: X(26)
        v_out26 = _mm_sub_ps(av_s33, av_s36);
        curr_out = out + out_strides[26];
        STR_128_S(curr_out, v_out_stride, v_out26);

        av_s37 = _mm_sub_ps(av_s25, av_s26);
        av_s38 = _mm_sub_ps(av_s35, av_s34);

        // Output point 5: X(4)
        v_out4 = _mm_add_ps(av_s37, av_s38);
        curr_out = out + out_strides[4];
        STR_128_S(curr_out, v_out_stride, v_out4);

        // Output point 17: X(16)
        v_out16 = _mm_sub_ps(av_s37, av_s38);
        curr_out = out + out_strides[16];
        STR_128_S(curr_out, v_out_stride, v_out16);

        av_s39 = _mm_add_ps(av_s25, av_s26);

        // Output point 23: X(22)
        v_out22 = _mm_sub_ps(av_s39, av_s32);
        curr_out = out + out_strides[22];
        STR_128_S(curr_out, v_out_stride, v_out22);

        // Output point 29: X(28)
        v_out28 = _mm_add_ps(av_s39, av_s32);
        curr_out = out + out_strides[28];
        STR_128_S(curr_out, v_out_stride, v_out28);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8, bv_in9, bv_in10, bv_in11, bv_in12, bv_in13, bv_in14;
        __m128 bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
               bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17,
               bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24, bv_s25,
               bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32, bv_s33,
               bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40, bv_s41,
               bv_s42, bv_s43, bv_s44, bv_s45, bv_s46, bv_s47, bv_s48, bv_s49;
        __m128 bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
               bv_t10, bv_t11, bv_t12, bv_t13, bv_t14, bv_t15, bv_t16, bv_t17,
               bv_t18, bv_t19, bv_t20, bv_t21, bv_t22, bv_t23, bv_t24, bv_t25,
               bv_t26;

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
        // Input point 22: x(21) & Input point 23: x(22)
        curr_in = in + in_strides[21];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in10, bv_in11);
        // Input point 26: x(25) & Input point 27: x(26)
        curr_in = in + in_strides[25];
        LDRI_2x128_S(curr_in, v_in_stride, bv_in12, bv_in13);
        // Input point 30: x(29)
        curr_in = in + in_strides[29];
        LDR_128_S(curr_in, v_in_stride, bv_in14);

        bv_s1 = _mm_add_ps(bv_in0, bv_in10);
        bv_s2 = _mm_sub_ps(bv_in0, bv_in10);
        bv_s3 = _mm_add_ps(bv_in1, bv_in11);
        bv_s4 = _mm_sub_ps(bv_in1, bv_in11);
        bv_s5 = _mm_add_ps(bv_in6, bv_in12);
        bv_s6 = _mm_sub_ps(bv_in6, bv_in12);
        bv_s7 = _mm_add_ps(bv_in7, bv_in13);
        bv_s8 = _mm_sub_ps(bv_in7, bv_in13);
        bv_s9 = _mm_add_ps(bv_in2, bv_in8);
        bv_s10 = _mm_sub_ps(bv_in2, bv_in8);

        bv_s11 = _mm_sub_ps(bv_s1, bv_s5);
        bv_t1 = _mm_mul_ps(v128_CRTM_15_14, bv_s11);
        bv_t2 = _mm_mul_ps(v128_CRTM_15_1, bv_s10);
        bv_t3 = _mm_mul_ps(v128_CRTM_15_5, bv_t1);
        bv_s12 = _mm_sub_ps(bv_t3, bv_t2);
        bv_t4 = _mm_mul_ps(v128_CRTM_15_2, bv_s9);
        bv_s13 = _mm_sub_ps(bv_in14, bv_t4);
        bv_s14 = _mm_add_ps(bv_s1, bv_s5);

        bv_t5 = _mm_mul_ps(v128_CRTM_15_6, bv_s14);
        bv_s15 = _mm_sub_ps(bv_in4, bv_t5);
        bv_t6 = _mm_mul_ps(v128_CRTM_15_5, bv_s15);
        bv_s16 = _mm_add_ps(bv_s13, bv_t6);
        bv_s17 = _mm_sub_ps(bv_s12, bv_s16);
        bv_s18 = _mm_add_ps(bv_s12, bv_s16);

        bv_t7 = _mm_mul_ps(v128_CRTM_15_13, bv_s8);
        bv_t8 = _mm_mul_ps(v128_CRTM_15_12, bv_s3);
        bv_t9 = _mm_mul_ps(v128_CRTM_15_4, bv_in9);
        bv_t10 = _mm_mul_ps(v128_CRTM_15_3, bv_in3);
        bv_s19 = _mm_sub_ps(bv_t7, bv_t8);
        bv_t11 = _mm_mul_ps(v128_CRTM_15_5, bv_s19);
        bv_s20 = _mm_sub_ps(bv_t9, bv_t10);
        bv_s21 = _mm_add_ps(bv_t11, bv_s20);

        // Output pt 8: X(7)
        v_out7 = _mm_add_ps(bv_s21, bv_s17);
        curr_out = out + out_strides[7];
        STR_128_S(curr_out, v_out_stride, v_out7);

        // Output pt 26: X(25)
        v_out25 = _mm_sub_ps(bv_s21, bv_s17);
        curr_out = out + out_strides[25];
        STR_128_S(curr_out, v_out_stride, v_out25);

        bv_s22 = _mm_add_ps(bv_s4, bv_s7);
        bv_s23 = _mm_sub_ps(bv_in5, bv_s22);
        bv_t12 = _mm_mul_ps(v128_CRTM_15_10, bv_s23);
        bv_s24 = _mm_add_ps(bv_s14, bv_in4);
        bv_t13 = _mm_mul_ps(v128_CRTM_15_5, bv_s9);
        bv_s25 = _mm_add_ps(bv_in14, bv_t13);
        bv_s26 = _mm_sub_ps(bv_s24, bv_s25);
        // Output pt 12: X(11)
        v_out11 = _mm_add_ps(bv_t12, bv_s26);
        curr_out = out + out_strides[11];
        STR_128_S(curr_out, v_out_stride, v_out11);
        // Output pt 22: X(21)
        v_out21 = _mm_sub_ps(bv_t12, bv_s26);
        curr_out = out + out_strides[21];
        STR_128_S(curr_out, v_out_stride, v_out21);
        bv_t19 = _mm_mul_ps(v128_CRTM_15_5, bv_s24);
        // Output pt 2: X(1)
        v_out1 = _mm_add_ps(bv_t19, bv_s25);
        curr_out = out + out_strides[1];
        STR_128_S(curr_out, v_out_stride, v_out1);

        bv_t14 = _mm_mul_ps(v128_CRTM_15_4, bv_in3);
        bv_t15 = _mm_mul_ps(v128_CRTM_15_3, bv_in9);
        bv_s27 = _mm_add_ps(bv_t14, bv_t15);

        bv_t16 = _mm_mul_ps(v128_CRTM_15_13, bv_s3);
        bv_t17 = _mm_mul_ps(v128_CRTM_15_12, bv_s8);
        bv_s28 = _mm_add_ps(bv_t16, bv_t17);
        bv_t18 = _mm_mul_ps(v128_CRTM_15_5, bv_s28);
        bv_s29 = _mm_sub_ps(bv_s27, bv_t18);
        // Output pt 14: X(13)
        v_out13 = _mm_add_ps(bv_s29, bv_s18);
        curr_out = out + out_strides[13];
        STR_128_S(curr_out, v_out_stride, v_out13);
        // Output pt 20: X(19)
        v_out19 = _mm_sub_ps(bv_s29, bv_s18);
        curr_out = out + out_strides[19];
        STR_128_S(curr_out, v_out_stride, v_out19);

        bv_t20 = _mm_mul_ps(v128_CRTM_15_11, bv_s22);
        bv_t21 = _mm_mul_ps(v128_CRTM_15_10, bv_in5);
        bv_s30 = _mm_add_ps(bv_t20, bv_t21);
        bv_s31 = _mm_sub_ps(bv_s4, bv_s7);
        bv_t22 = _mm_mul_ps(v128_CRTM_15_9, bv_s31);
        bv_s32 = _mm_sub_ps(bv_t22, bv_s30);
        bv_s33 = _mm_add_ps(bv_s30, bv_t22);
        bv_s34 = _mm_add_ps(bv_t1, bv_t2);
        bv_s35 = _mm_sub_ps(bv_s13, bv_s15);
        bv_s36 = _mm_add_ps(bv_s34, bv_s35);
        bv_s37 = _mm_sub_ps(bv_s34, bv_s35);
        bv_s38 = _mm_add_ps(bv_s28, bv_s27);

        bv_t23 = _mm_mul_ps(v128_CRTM_15_8, bv_s2);
        bv_t24 = _mm_mul_ps(v128_CRTM_15_7, bv_s6);
        bv_s39 = _mm_add_ps(bv_t23, bv_t24);
        bv_t25 = _mm_mul_ps(v128_CRTM_15_8, bv_s6);
        bv_t26 = _mm_mul_ps(v128_CRTM_15_7, bv_s2);
        bv_s40 = _mm_sub_ps(bv_t25, bv_t26);
        bv_s41 = _mm_sub_ps(bv_s19, bv_s20);
        bv_s42 = _mm_sub_ps(bv_s37, bv_s38);
        bv_s43 = _mm_add_ps(bv_s37, bv_s38);
        bv_s44 = _mm_add_ps(bv_s39, bv_s32);
        bv_s45 = _mm_sub_ps(bv_s39, bv_s32);
        bv_s46 = _mm_sub_ps(bv_s40, bv_s33);
        bv_s47 = _mm_add_ps(bv_s33, bv_s40);
        bv_s48 = _mm_add_ps(bv_s36, bv_s41);
        bv_s49 = _mm_sub_ps(bv_s36, bv_s41);

        // Output pt 4: X(3)
        v_out3 = _mm_add_ps(bv_s42, bv_s44);
        curr_out = out + out_strides[3];
        STR_128_S(curr_out, v_out_stride, v_out3);
        // Output pt 24: X(23)
        v_out23 = _mm_sub_ps(bv_s42, bv_s44);
        curr_out = out + out_strides[23];
        STR_128_S(curr_out, v_out_stride, v_out23);
        // Output pt 18: X(17)
        v_out17 = _mm_add_ps(bv_s46, bv_s48);
        curr_out = out + out_strides[17];
        STR_128_S(curr_out, v_out_stride, v_out17);
        // Output pt 28: X(27)
        v_out27 = _mm_sub_ps(bv_s46, bv_s48);
        curr_out = out + out_strides[27];
        STR_128_S(curr_out, v_out_stride, v_out27);
        // Output pt 6: X(5)
        v_out5 = _mm_sub_ps(bv_s49, bv_s47);
        curr_out = out + out_strides[5];
        STR_128_S(curr_out, v_out_stride, v_out5);
        // Output pt 16: X(15)
        v_out15 = NEGATE_128_S(_mm_add_ps(bv_s47, bv_s49));
        curr_out = out + out_strides[15];
        STR_128_S(curr_out, v_out_stride, v_out15);
        // Output pt 10: X(9)
        v_out9 = _mm_sub_ps(bv_s45, bv_s43);
        curr_out = out + out_strides[9];
        STR_128_S(curr_out, v_out_stride, v_out9);
        // Output pt 30: X(29)
        v_out29 = NEGATE_128_S(_mm_add_ps(bv_s43, bv_s45));
        curr_out = out + out_strides[29];
        STR_128_S(curr_out, v_out_stride, v_out29);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (n & 2)
    {
        /* Standard DFT */
        __m128 av_in0, av_in1, av_in2, av_in3;
        __m128 av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
               av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16,
               av_s17, av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24,
               av_s25, av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32,
               av_s33, av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40;
        __m128 av_t0, av_t1, av_t2, av_t4, av_t5;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
               v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21,
               v_out22, v_out23, v_out24, v_out25, v_out26, v_out27, v_out28,
               v_out29;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_15_1 = _mm256_castps256_ps128(v_CRTM_15_1);
        __m128 v128_CRTM_15_2 = _mm256_castps256_ps128(v_CRTM_15_2);
        __m128 v128_CRTM_15_3 = _mm256_castps256_ps128(v_CRTM_15_3);
        __m128 v128_CRTM_15_4 = _mm256_castps256_ps128(v_CRTM_15_4);
        __m128 v128_CRTM_15_5 = _mm256_castps256_ps128(v_CRTM_15_5);
        __m128 v128_CRTM_15_6 = _mm256_castps256_ps128(v_CRTM_15_6);
        __m128 v128_CRTM_15_7 = _mm256_castps256_ps128(v_CRTM_15_7);
        __m128 v128_CRTM_15_8 = _mm256_castps256_ps128(v_CRTM_15_8);
        __m128 v128_CRTM_15_9 = _mm256_castps256_ps128(v_CRTM_15_9);
        __m128 v128_CRTM_15_10 = _mm256_castps256_ps128(v_CRTM_15_10);
        __m128 v128_CRTM_15_11 = _mm256_castps256_ps128(v_CRTM_15_11);
        __m128 v128_CRTM_15_12 = _mm256_castps256_ps128(v_CRTM_15_12);
        __m128 v128_CRTM_15_13 = _mm256_castps256_ps128(v_CRTM_15_13);
        __m128 v128_CRTM_15_14 = _mm256_castps256_ps128(v_CRTM_15_14);

        // Input point 1: x(0)
        curr_in = in + in_strides[0];
        LDHR_128_S(curr_in, v_in_stride, av_in0);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDHR_128_S(curr_in, v_in_stride, av_in1);
        // Input point 28: x(27)
        curr_in = in + in_strides[27];
        LDHR_128_S(curr_in, v_in_stride, av_in2);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDHR_128_S(curr_in, v_in_stride, av_in3);

        av_s0 = _mm_add_ps(av_in1, av_in2);
        av_s1 = _mm_sub_ps(av_in1, av_in2);
        av_s2 = _mm_add_ps(av_in3, av_s0);
        av_s3 = _mm_sub_ps(av_in3, _mm_mul_ps(v128_CRTM_15_2, av_s0));

        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDHR_128_S(curr_in, v_in_stride, av_in1);
        // Input point 29: x(28)
        curr_in = in + in_strides[28];
        LDHR_128_S(curr_in, v_in_stride, av_in2);
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDHR_128_S(curr_in, v_in_stride, av_in3);

        av_s0 = _mm_add_ps(av_in1, av_in2);
        av_s4 = _mm_sub_ps(av_in1, av_in2);
        av_s5 = _mm_sub_ps(av_s0, av_in3);
        av_s6 = _mm_add_ps(av_in3, _mm_mul_ps(v128_CRTM_15_2, av_s0));

        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDHR_128_S(curr_in, v_in_stride, av_in1);
        // Input point 16: x(15)
        curr_in = in + in_strides[15];
        LDHR_128_S(curr_in, v_in_stride, av_in2);
        // Input point 24: x(23)
        curr_in = in + in_strides[23];
        LDHR_128_S(curr_in, v_in_stride, av_in3);

        av_s0 = _mm_add_ps(av_in1, av_in2);
        av_s7 = _mm_sub_ps(av_in2, av_in1);
        av_s8 = _mm_add_ps(av_in3, av_s0);
        av_s9 = _mm_sub_ps(av_in3, _mm_mul_ps(v128_CRTM_15_2, av_s0));

        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDHR_128_S(curr_in, v_in_stride, av_in1);
        // Input point 17: x(16)
        curr_in = in + in_strides[16];
        LDHR_128_S(curr_in, v_in_stride, av_in2);
        // Input point 25: x(24)
        curr_in = in + in_strides[24];
        LDHR_128_S(curr_in, v_in_stride, av_in3);

        av_s0 = _mm_sub_ps(av_in1, av_in2);
        av_s10 = _mm_add_ps(av_in2, av_in1);
        av_s11 = _mm_add_ps(av_in3, av_s0);
        av_s12 = _mm_sub_ps(av_in3, _mm_mul_ps(v128_CRTM_15_2, av_s0));

        // Input point 20: x(19)
        curr_in = in + in_strides[19];
        LDHR_128_S(curr_in, v_in_stride, av_in1);
        // Input point 21: x(20)
        curr_in = in + in_strides[20];
        LDHR_128_S(curr_in, v_in_stride, av_in2);

        av_s13 = _mm_add_ps(av_in0, _mm_mul_ps(v128_CRTM_15_5, av_in1));
        av_s14 = _mm_sub_ps(av_in0, av_in1);
        av_s15 = _mm_add_ps(av_s8, av_s2);
        av_t0 = _mm_mul_ps(v128_CRTM_15_1, _mm_sub_ps(av_s8, av_s2));
        av_s16 = _mm_sub_ps(av_s13, _mm_mul_ps(v128_CRTM_15_2, av_s15));

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(av_s13, _mm_mul_ps(v128_CRTM_15_5, av_s15));
        STHR_128_S(curr_out, v_out_stride, v_out0);

        av_s17 = _mm_add_ps(_mm_mul_ps(v128_CRTM_15_3, av_s11),
                            _mm_mul_ps(v128_CRTM_15_4, av_s5));
        av_s18 = _mm_add_ps(av_s16, av_t0);

        // Output point 7: X(6)
        v_out6 = _mm_sub_ps(av_s18, av_s17);
        curr_out = out + out_strides[6];
        STHR_128_S(curr_out, v_out_stride, v_out6);

        // Output point 25: X(24)
        v_out24 = _mm_add_ps(av_s18, av_s17);
        curr_out = out + out_strides[24];
        STHR_128_S(curr_out, v_out_stride, v_out24);

        av_s19 = _mm_sub_ps(av_s16, av_t0);
        av_s20 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_4, av_s11),
                            _mm_mul_ps(v128_CRTM_15_3, av_s5));

        // Output point 13: X(12)
        v_out12 = _mm_sub_ps(av_s19, av_s20);
        curr_out = out + out_strides[12];
        STHR_128_S(curr_out, v_out_stride, v_out12);

        // Output point 19: X(18)
        v_out18 = _mm_add_ps(av_s19, av_s20);
        curr_out = out + out_strides[18];
        STHR_128_S(curr_out, v_out_stride, v_out18);

        av_t1 = _mm_mul_ps(v128_CRTM_15_1, _mm_sub_ps(av_s9, av_s3));
        av_t2 = _mm_mul_ps(v128_CRTM_15_9, _mm_add_ps(av_s4, av_s10));
        av_s21 = _mm_add_ps(av_s9, av_s3);
        av_s22 = _mm_sub_ps(av_s4, av_s10);
        av_s40 = _mm_add_ps(_mm_mul_ps(v128_CRTM_15_5, av_s21), av_s14);
        av_t4 = _mm_mul_ps(v128_CRTM_15_10, _mm_add_ps(av_s22, av_in2));

        // Output point 11: X(10)
        v_out10 = _mm_add_ps(av_s40, av_t4);
        curr_out = out + out_strides[10];
        STHR_128_S(curr_out, v_out_stride, v_out10);

        // Output point 21: X(20)
        v_out20 = _mm_sub_ps(av_s40, av_t4);
        curr_out = out + out_strides[20];
        STHR_128_S(curr_out, v_out_stride, v_out20);

        av_s23 = _mm_sub_ps(av_s14, _mm_mul_ps(v128_CRTM_15_2, av_s21));
        av_t5 = _mm_mul_ps(v128_CRTM_15_10,
                _mm_sub_ps( _mm_mul_ps(v128_CRTM_15_6, av_s22), av_in2));
        av_s24 = _mm_add_ps(av_s23, av_t5);
        av_s25 = _mm_sub_ps(av_s23, av_t5);
        av_s26 = _mm_sub_ps(av_t2, av_t1);
        av_s27 = _mm_add_ps(av_t1, av_t2);

        av_s28 = _mm_sub_ps(av_s24, av_s27);
        av_s29 = _mm_add_ps(_mm_mul_ps(v128_CRTM_15_3, av_s6),
                            _mm_mul_ps(v128_CRTM_15_4, av_s12));
        av_s30 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_8, av_s1),
                            _mm_mul_ps(v128_CRTM_15_7, av_s7));
        av_s31 = _mm_sub_ps(av_s30, av_s29);
        av_s32 = _mm_add_ps(av_s29, av_s30);

        // Output point 3: X(2)
        v_out2 = _mm_add_ps(av_s28, av_s31);
        curr_out = out + out_strides[2];
        STHR_128_S(curr_out, v_out_stride, v_out2);

        // Output point 9: X(8)
        v_out8 = _mm_sub_ps(av_s28, av_s31);
        curr_out = out + out_strides[8];
        STHR_128_S(curr_out, v_out_stride, v_out8);

        av_s33 = _mm_add_ps(av_s24, av_s27);
        av_s34 = _mm_add_ps(_mm_mul_ps(v128_CRTM_15_8, av_s7),
                            _mm_mul_ps(v128_CRTM_15_7, av_s1));
        av_s35 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_3, av_s12),
                            _mm_mul_ps(v128_CRTM_15_4, av_s6));
        av_s36 = _mm_add_ps(av_s35, av_s34);

        // Output point 15: X(14)
        v_out14 = _mm_add_ps(av_s33, av_s36);
        curr_out = out + out_strides[14];
        STHR_128_S(curr_out, v_out_stride, v_out14);

        // Output point 27: X(26)
        v_out26 = _mm_sub_ps(av_s33, av_s36);
        curr_out = out + out_strides[26];
        STHR_128_S(curr_out, v_out_stride, v_out26);

        av_s37 = _mm_sub_ps(av_s25, av_s26);
        av_s38 = _mm_sub_ps(av_s35, av_s34);

        // Output point 5: X(4)
        v_out4 = _mm_add_ps(av_s37, av_s38);
        curr_out = out + out_strides[4];
        STHR_128_S(curr_out, v_out_stride, v_out4);

        // Output point 17: X(16)
        v_out16 = _mm_sub_ps(av_s37, av_s38);
        curr_out = out + out_strides[16];
        STHR_128_S(curr_out, v_out_stride, v_out16);

        av_s39 = _mm_add_ps(av_s25, av_s26);

        // Output point 23: X(22)
        v_out22 = _mm_sub_ps(av_s39, av_s32);
        curr_out = out + out_strides[22];
        STHR_128_S(curr_out, v_out_stride, v_out22);

        // Output point 29: X(28)
        v_out28 = _mm_add_ps(av_s39, av_s32);
        curr_out = out + out_strides[28];
        STHR_128_S(curr_out, v_out_stride, v_out28);

        /* Shifted DFT */
        __m128 bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
               bv_in8, bv_in9, bv_in10, bv_in11, bv_in12, bv_in13, bv_in14;
        __m128 bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
               bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17,
               bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24, bv_s25,
               bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32, bv_s33,
               bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40, bv_s41,
               bv_s42, bv_s43, bv_s44, bv_s45, bv_s46, bv_s47, bv_s48, bv_s49;
        __m128 bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
               bv_t10, bv_t11, bv_t12, bv_t13, bv_t14, bv_t15, bv_t16, bv_t17,
               bv_t18, bv_t19, bv_t20, bv_t21, bv_t22, bv_t23, bv_t24, bv_t25,
               bv_t26;

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
        // Input point 22: x(21) & Input point 23: x(22)
        curr_in = in + in_strides[21];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in10, bv_in11);
        // Input point 26: x(25) & Input point 27: x(26)
        curr_in = in + in_strides[25];
        LDHRI_2x128_S(curr_in, v_in_stride, bv_in12, bv_in13);
        // Input point 30: x(29)
        curr_in = in + in_strides[29];
        LDHR_128_S(curr_in, v_in_stride, bv_in14);

        bv_s1 = _mm_add_ps(bv_in0, bv_in10);
        bv_s2 = _mm_sub_ps(bv_in0, bv_in10);
        bv_s3 = _mm_add_ps(bv_in1, bv_in11);
        bv_s4 = _mm_sub_ps(bv_in1, bv_in11);
        bv_s5 = _mm_add_ps(bv_in6, bv_in12);
        bv_s6 = _mm_sub_ps(bv_in6, bv_in12);
        bv_s7 = _mm_add_ps(bv_in7, bv_in13);
        bv_s8 = _mm_sub_ps(bv_in7, bv_in13);
        bv_s9 = _mm_add_ps(bv_in2, bv_in8);
        bv_s10 = _mm_sub_ps(bv_in2, bv_in8);

        bv_s11 = _mm_sub_ps(bv_s1, bv_s5);
        bv_t1 = _mm_mul_ps(v128_CRTM_15_14, bv_s11);
        bv_t2 = _mm_mul_ps(v128_CRTM_15_1, bv_s10);
        bv_t3 = _mm_mul_ps(v128_CRTM_15_5, bv_t1);
        bv_s12 = _mm_sub_ps(bv_t3, bv_t2);
        bv_t4 = _mm_mul_ps(v128_CRTM_15_2, bv_s9);
        bv_s13 = _mm_sub_ps(bv_in14, bv_t4);
        bv_s14 = _mm_add_ps(bv_s1, bv_s5);

        bv_t5 = _mm_mul_ps(v128_CRTM_15_6, bv_s14);
        bv_s15 = _mm_sub_ps(bv_in4, bv_t5);
        bv_t6 = _mm_mul_ps(v128_CRTM_15_5, bv_s15);
        bv_s16 = _mm_add_ps(bv_s13, bv_t6);
        bv_s17 = _mm_sub_ps(bv_s12, bv_s16);
        bv_s18 = _mm_add_ps(bv_s12, bv_s16);

        bv_t7 = _mm_mul_ps(v128_CRTM_15_13, bv_s8);
        bv_t8 = _mm_mul_ps(v128_CRTM_15_12, bv_s3);
        bv_t9 = _mm_mul_ps(v128_CRTM_15_4, bv_in9);
        bv_t10 = _mm_mul_ps(v128_CRTM_15_3, bv_in3);
        bv_s19 = _mm_sub_ps(bv_t7, bv_t8);
        bv_t11 = _mm_mul_ps(v128_CRTM_15_5, bv_s19);
        bv_s20 = _mm_sub_ps(bv_t9, bv_t10);
        bv_s21 = _mm_add_ps(bv_t11, bv_s20);

        // Output pt 8: X(7)
        v_out7 = _mm_add_ps(bv_s21, bv_s17);
        curr_out = out + out_strides[7];
        STHR_128_S(curr_out, v_out_stride, v_out7);

        // Output pt 26: X(25)
        v_out25 = _mm_sub_ps(bv_s21, bv_s17);
        curr_out = out + out_strides[25];
        STHR_128_S(curr_out, v_out_stride, v_out25);

        bv_s22 = _mm_add_ps(bv_s4, bv_s7);
        bv_s23 = _mm_sub_ps(bv_in5, bv_s22);
        bv_t12 = _mm_mul_ps(v128_CRTM_15_10, bv_s23);
        bv_s24 = _mm_add_ps(bv_s14, bv_in4);
        bv_t13 = _mm_mul_ps(v128_CRTM_15_5, bv_s9);
        bv_s25 = _mm_add_ps(bv_in14, bv_t13);
        bv_s26 = _mm_sub_ps(bv_s24, bv_s25);
        // Output pt 12: X(11)
        v_out11 = _mm_add_ps(bv_t12, bv_s26);
        curr_out = out + out_strides[11];
        STHR_128_S(curr_out, v_out_stride, v_out11);
        // Output pt 22: X(21)
        v_out21 = _mm_sub_ps(bv_t12, bv_s26);
        curr_out = out + out_strides[21];
        STHR_128_S(curr_out, v_out_stride, v_out21);
        bv_t19 = _mm_mul_ps(v128_CRTM_15_5, bv_s24);
        // Output pt 2: X(1)
        v_out1 = _mm_add_ps(bv_t19, bv_s25);
        curr_out = out + out_strides[1];
        STHR_128_S(curr_out, v_out_stride, v_out1);

        bv_t14 = _mm_mul_ps(v128_CRTM_15_4, bv_in3);
        bv_t15 = _mm_mul_ps(v128_CRTM_15_3, bv_in9);
        bv_s27 = _mm_add_ps(bv_t14, bv_t15);

        bv_t16 = _mm_mul_ps(v128_CRTM_15_13, bv_s3);
        bv_t17 = _mm_mul_ps(v128_CRTM_15_12, bv_s8);
        bv_s28 = _mm_add_ps(bv_t16, bv_t17);
        bv_t18 = _mm_mul_ps(v128_CRTM_15_5, bv_s28);
        bv_s29 = _mm_sub_ps(bv_s27, bv_t18);
        // Output pt 14: X(13)
        v_out13 = _mm_add_ps(bv_s29, bv_s18);
        curr_out = out + out_strides[13];
        STHR_128_S(curr_out, v_out_stride, v_out13);
        // Output pt 20: X(19)
        v_out19 = _mm_sub_ps(bv_s29, bv_s18);
        curr_out = out + out_strides[19];
        STHR_128_S(curr_out, v_out_stride, v_out19);

        bv_t20 = _mm_mul_ps(v128_CRTM_15_11, bv_s22);
        bv_t21 = _mm_mul_ps(v128_CRTM_15_10, bv_in5);
        bv_s30 = _mm_add_ps(bv_t20, bv_t21);
        bv_s31 = _mm_sub_ps(bv_s4, bv_s7);
        bv_t22 = _mm_mul_ps(v128_CRTM_15_9, bv_s31);
        bv_s32 = _mm_sub_ps(bv_t22, bv_s30);
        bv_s33 = _mm_add_ps(bv_s30, bv_t22);
        bv_s34 = _mm_add_ps(bv_t1, bv_t2);
        bv_s35 = _mm_sub_ps(bv_s13, bv_s15);
        bv_s36 = _mm_add_ps(bv_s34, bv_s35);
        bv_s37 = _mm_sub_ps(bv_s34, bv_s35);
        bv_s38 = _mm_add_ps(bv_s28, bv_s27);

        bv_t23 = _mm_mul_ps(v128_CRTM_15_8, bv_s2);
        bv_t24 = _mm_mul_ps(v128_CRTM_15_7, bv_s6);
        bv_s39 = _mm_add_ps(bv_t23, bv_t24);
        bv_t25 = _mm_mul_ps(v128_CRTM_15_8, bv_s6);
        bv_t26 = _mm_mul_ps(v128_CRTM_15_7, bv_s2);
        bv_s40 = _mm_sub_ps(bv_t25, bv_t26);
        bv_s41 = _mm_sub_ps(bv_s19, bv_s20);
        bv_s42 = _mm_sub_ps(bv_s37, bv_s38);
        bv_s43 = _mm_add_ps(bv_s37, bv_s38);
        bv_s44 = _mm_add_ps(bv_s39, bv_s32);
        bv_s45 = _mm_sub_ps(bv_s39, bv_s32);
        bv_s46 = _mm_sub_ps(bv_s40, bv_s33);
        bv_s47 = _mm_add_ps(bv_s33, bv_s40);
        bv_s48 = _mm_add_ps(bv_s36, bv_s41);
        bv_s49 = _mm_sub_ps(bv_s36, bv_s41);

        // Output pt 4: X(3)
        v_out3 = _mm_add_ps(bv_s42, bv_s44);
        curr_out = out + out_strides[3];
        STHR_128_S(curr_out, v_out_stride, v_out3);
        // Output pt 24: X(23)
        v_out23 = _mm_sub_ps(bv_s42, bv_s44);
        curr_out = out + out_strides[23];
        STHR_128_S(curr_out, v_out_stride, v_out23);
        // Output pt 18: X(17)
        v_out17 = _mm_add_ps(bv_s46, bv_s48);
        curr_out = out + out_strides[17];
        STHR_128_S(curr_out, v_out_stride, v_out17);
        // Output pt 28: X(27)
        v_out27 = _mm_sub_ps(bv_s46, bv_s48);
        curr_out = out + out_strides[27];
        STHR_128_S(curr_out, v_out_stride, v_out27);
        // Output pt 6: X(5)
        v_out5 = _mm_sub_ps(bv_s49, bv_s47);
        curr_out = out + out_strides[5];
        STHR_128_S(curr_out, v_out_stride, v_out5);
        // Output pt 16: X(15)
        v_out15 = NEGATE_128_S(_mm_add_ps(bv_s47, bv_s49));
        curr_out = out + out_strides[15];
        STHR_128_S(curr_out, v_out_stride, v_out15);
        // Output pt 10: X(9)
        v_out9 = _mm_sub_ps(bv_s45, bv_s43);
        curr_out = out + out_strides[9];
        STHR_128_S(curr_out, v_out_stride, v_out9);
        // Output pt 30: X(29)
        v_out29 = NEGATE_128_S(_mm_add_ps(bv_s43, bv_s45));
        curr_out = out + out_strides[29];
        STHR_128_S(curr_out, v_out_stride, v_out29);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        /* Standard DFT */
        FLOAT a_in0, a_in1, a_in2, a_in3, a_s0, a_s1, a_s2, a_s3, a_s4, a_s5,
              a_s6, a_s7, a_s8, a_s9, a_s10, a_s11, a_s12, a_s13, a_s14, a_s15,
              a_s16, a_s17, a_s18, a_s19, a_s20, a_s21, a_s22, a_s23, a_s24,
              a_s25, a_s26, a_s27, a_s28, a_s29, a_s30, a_s31, a_s32, a_s33,
              a_s34, a_s35, a_s36, a_s37, a_s38, a_s39, a_t0, a_t1, a_t2, a_t3,
              a_t4, a_t5;

        //  Input point 1: x(0)
        a_in0 = *in;
        //  Input point 8: x(7)
        a_in1 = in[in_strides[7]];
        //  Input point 28: x(27)
        a_in2 = in[in_strides[27]];
        //  Input point 12: x(11)
        a_in3 = in[in_strides[11]];

        a_s0 = a_in1 + a_in2;
        a_s1 = a_in1 - a_in2;
        a_s2 = a_in3 + a_s0;
        a_s3 = a_in3 - (CRTM_15_2 * a_s0);

        //  Input point 9: x(8)
        a_in1 = in[in_strides[8]];
        //  Input point 29: x(28)
        a_in2 = in[in_strides[28]];
        //  Input point 13: x(12)
        a_in3 = in[in_strides[12]];

        a_s0 = a_in1 + a_in2;
        a_s4 = a_in1 - a_in2;
        a_s5 = a_s0 - a_in3;
        a_s6 = a_in3 + (CRTM_15_2 * a_s0);

        //  Input point 4: x(3)
        a_in1 = in[in_strides[3]];
        //  Input point 16: x(15)
        a_in2 = in[in_strides[15]];
        //  Input point 24: x(23)
        a_in3 = in[in_strides[23]];

        a_s0 = a_in1 + a_in2;
        a_s7 = a_in2 - a_in1;
        a_s8 = a_in3 + a_s0;
        a_s9 = a_in3 - (CRTM_15_2 * a_s0);

        //  Input point 5: x(4)
        a_in1 = in[in_strides[4]];
        //  Input point 17: x(16)
        a_in2 = in[in_strides[16]];
        //  Input point 25: x(24)
        a_in3 = in[in_strides[24]];
        a_s0 = a_in1 - a_in2;
        a_s10 = a_in2 + a_in1;
        a_s11 = a_in3 + a_s0;
        a_s12 = a_in3 - (CRTM_15_2 * a_s0);

        //  Input point 20: x(19)
        a_in1 = in[in_strides[19]];
        //  Input point 21: x(20)
        a_in2 = in[in_strides[20]];

        a_s13 = a_in0 + CRTM_15_5 * a_in1;
        a_s14 = a_in0 - a_in1;
        a_s15 = a_s8 + a_s2;
        a_t0 = CRTM_15_1 * (a_s8 - a_s2);
        a_s16 = a_s13 - (CRTM_15_2 * a_s15);
        // Output point 1: X(0)
        *out = a_s13 + CRTM_15_5 * a_s15;

        a_s17 = (CRTM_15_3 * a_s11) + (CRTM_15_4 * a_s5);
        a_s18 = a_s16 + a_t0;

        // Output point 7: X(6)
        out[out_strides[6]] = a_s18 - a_s17;
        // Output point 25: X(24)
        out[out_strides[24]] = a_s18 + a_s17;

        a_s19 = a_s16 - a_t0;
        a_s20 = (CRTM_15_4 * a_s11) - (CRTM_15_3 * a_s5);

        // Output point 13: X(12)
        out[out_strides[12]] = a_s19 - a_s20;
        // Output point 19: X(18)
        out[out_strides[18]] = a_s19 + a_s20;

        a_t1 = CRTM_15_1 * (a_s9 - a_s3);
        a_t2 = CRTM_15_9 * (a_s10 + a_s4);
        a_s21 = (a_s9 + a_s3);
        a_s22 = a_s4 - a_s10;
        a_t3 = CRTM_15_5 * a_s21 + a_s14;
        a_t4 = CRTM_15_10 * (a_s22 + a_in2);

        // Output point 11: X(10)
        out[out_strides[10]] = a_t3 + a_t4;
        // Output point 21: X(20)
        out[out_strides[20]] = a_t3 - a_t4;

        a_s23 = a_s14 - (CRTM_15_2 * a_s21);
        a_t5 = CRTM_15_10 * ((CRTM_15_6 * a_s22) - a_in2);
        a_s24 = a_t5 + a_s23;
        a_s25 = a_s23 - a_t5;
        a_s26 = a_t2 - a_t1;
        a_s27 = a_t1 + a_t2;

        a_s28 = a_s24 - a_s27;
        a_s29 = (CRTM_15_4 * a_s12) + (CRTM_15_3 * a_s6);
        a_s30 = (CRTM_15_8 * a_s1) - (CRTM_15_7 * a_s7);
        a_s31 = a_s30 - a_s29;
        a_s32 = a_s29 + a_s30;

        // Output point 3: X(2)
        out[out_strides[2]] = a_s28 + a_s31;
        // Output point 9: X(8)
        out[out_strides[8]] = a_s28 - a_s31;

        a_s33 = a_s24 + a_s27;
        a_s34 = (CRTM_15_8 * a_s7) + (CRTM_15_7 * a_s1);
        a_s35 = (CRTM_15_3 * a_s12) - (CRTM_15_4 * a_s6);
        a_s36 = a_s35 + a_s34;

        // Output point 15: X(14)
        out[out_strides[14]] = a_s33 + a_s36;
        // Output point 27: X(26)
        out[out_strides[26]] = a_s33 - a_s36;

        a_s37 = a_s25 - a_s26;
        a_s38 = a_s35 - a_s34;
        // Output point 5: X(4)
        out[out_strides[4]] = a_s37 + a_s38;
        // Output point 17: X(16)
        out[out_strides[16]] = a_s37 - a_s38;

        a_s39 = a_s25 + a_s26;

        // Output point 23: X(22)
        out[out_strides[22]] = a_s39 - a_s32;
        // Output point 29: X(28)
        out[out_strides[28]] = a_s39 + a_s32;

        /* Shifted DFT */
        FLOAT b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7, b_in8,
              b_in9, b_in10, b_in11, b_in12, b_in13, b_in14;
        FLOAT b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9, b_s10,
              b_s11, b_s12, b_s13, b_s14, b_s15, b_s16, b_s17, b_s18, b_s19,
              b_s20, b_s21, b_s22, b_s23, b_s24, b_s25, b_s26, b_s27, b_s28,
              b_s29, b_s30, b_s31, b_s32, b_s33, b_s34, b_s35, b_s36, b_s37,
              b_s38, b_s39, b_s40, b_s41, b_s42, b_s43, b_s44, b_s45, b_s46,
              b_s47, b_s48, b_t0, b_t1, b_t2, b_t3, b_t4, b_t5, b_t6, b_t7,
              b_t8, b_t9, b_t10, b_t11, b_t12, b_t13, b_t14, b_t15, b_t16,
              b_t17, b_t18, b_t19, b_t20, b_t21, b_t22, b_t23, b_t24, b_t25;

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
        // Input point 22: x(21)
        b_in10 = in[in_strides[21]];
        // Input point 23: x(22)
        b_in11 = in[in_strides[22]];
        // Input point 26: x(25)
        b_in12 = in[in_strides[25]];
        // Input point 27: x(26)
        b_in13 = in[in_strides[26]];
        // Input point 30: x(29)
        b_in14 = in[in_strides[29]];

        b_s0 = b_in0 + b_in10;
        b_s1 = b_in0 - b_in10;
        b_s2 = b_in1 + b_in11;
        b_s3 = b_in1 - b_in11;
        b_s4 = b_in6 + b_in12;
        b_s5 = b_in6 - b_in12;
        b_s6 = b_in7 + b_in13;
        b_s7 = b_in7 - b_in13;
        b_s8 = b_in2 + b_in8;
        b_s9 = b_in2 - b_in8;

        b_s10 = b_s0 - b_s4;
        b_t0 = CRTM_15_14 * b_s10;
        b_t1 = CRTM_15_1 * b_s9;
        b_t2 = CRTM_15_5 * b_t0;
        b_s11 = b_t2 - b_t1;
        b_t3 = CRTM_15_2 * b_s8;
        b_s12 = b_in14 - b_t3;
        b_s13 = b_s0 + b_s4;

        b_t4 = CRTM_15_6 * b_s13;
        b_s14 = b_in4 - b_t4;
        b_t5 = CRTM_15_5 * b_s14;
        b_s15 = b_s12 + b_t5;
        b_s16 = b_s11 - b_s15;
        b_s17 = b_s11 + b_s15;

        b_t6 = CRTM_15_13 * b_s7;
        b_t7 = CRTM_15_12 * b_s2;
        b_t8 = CRTM_15_4 * b_in9;
        b_t9 = CRTM_15_3 * b_in3;
        b_s18 = b_t6 - b_t7;
        b_t10 = CRTM_15_5 * b_s18;
        b_s19 = b_t8 - b_t9;
        b_s20 = b_t10 + b_s19;
        // Output pt 8: X(7)
        out[out_strides[7]] = b_s20 + b_s16;
        // Output pt 26: X(25)
        out[out_strides[25]] = b_s20 - b_s16;

        b_s21 = b_s3 + b_s6;
        b_s22 = b_in5 - b_s21;
        b_t11 = CRTM_15_10 * b_s22;
        b_s23 = b_s13 + b_in4;
        b_t12 = CRTM_15_5 * b_s8;
        b_s24 = b_in14 + b_t12;
        b_s25 = b_s23 - b_s24;
        // Output pt 12: X(11)
        out[out_strides[11]] = b_t11 + b_s25;
        // Output pt 22: X(21)
        out[out_strides[21]] = b_t11 - b_s25;

        b_t13 = CRTM_15_5 * b_s23;
        // Output pt 2: X(1)
        out[out_strides[1]] = b_t13 + b_s24;

        b_t14 = CRTM_15_4 * b_in3;
        b_t15 = CRTM_15_3 * b_in9;
        b_s26 = b_t14 + b_t15;

        b_t16 = CRTM_15_13 * b_s2;
        b_t17 = CRTM_15_12 * b_s7;
        b_s27 = b_t16 + b_t17;
        b_t18 = CRTM_15_5 * b_s27;
        b_s28 = b_s26 - b_t18;
        // Output pt 14: X(13)
        out[out_strides[13]] = b_s28 + b_s17;
        // Output pt 20: X(19)
        out[out_strides[19]] = b_s28 - b_s17;

        b_t19 = CRTM_15_11 * b_s21;
        b_t20 = CRTM_15_10 * b_in5;
        b_s29 = b_t19 + b_t20;
        b_s30 = b_s3 - b_s6;
        b_t21 = CRTM_15_9 * b_s30;
        b_s31 = b_t21 - b_s29;
        b_s32 = b_s29 + b_t21;
        b_s33 = b_t0 + b_t1;
        b_s34 = b_s12 - b_s14;
        b_s35 = b_s33 + b_s34;
        b_s36 = b_s33 - b_s34;
        b_s37 = b_s27 + b_s26;

        b_t22 = CRTM_15_8 * b_s1;
        b_t23 = CRTM_15_7 * b_s5;
        b_s38 = b_t22 + b_t23;
        b_t24 = CRTM_15_8 * b_s5;
        b_t25 = CRTM_15_7 * b_s1;
        b_s39 = b_t24 - b_t25;
        b_s40 = b_s18 - b_s19;
        b_s41 = b_s36 - b_s37;
        b_s42 = b_s36 + b_s37;
        b_s43 = b_s38 + b_s31;
        // Output pt 4: X(3)
        out[out_strides[3]] = b_s41 + b_s43;
        // Output pt 24: X(23)
        out[out_strides[23]] = b_s41 - b_s43;

        b_s44 = b_s38 - b_s31;
        // Output pt 10: X(9)
        out[out_strides[9]] = b_s44 - b_s42;
        // Output pt 30: X(29)
        out[out_strides[29]] = -(b_s42 + b_s44);

        b_s45 = b_s39 - b_s32;
        b_s46 = b_s32 + b_s39;
        b_s47 = b_s35 + b_s40;
        // Output pt 18: X(17)
        out[out_strides[17]] = b_s45 + b_s47;
        // Output pt 28: X(27)
        out[out_strides[27]] = b_s45 - b_s47;

        b_s48 = b_s35 - b_s40;
        // Output pt 6: X(5)
        out[out_strides[5]] = b_s48 - b_s46;
        // Output pt 16: X(15)
        out[out_strides[15]] = -(b_s46 + b_s48);
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID r2hcf_rfft15avx256_fp64_fwd(VOID *in_real, VOID *in_imag,
                                        VOID *out_real, VOID *out_imag, INTP n,
                                        aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const DOUBLE CRTM_15_1 =
        0.55901699437494742410229341718281905886015458990288;
    const DOUBLE CRTM_15_2 =
        0.25000000000000000000000000000000000000000000000000;
    const DOUBLE CRTM_15_3 =
        0.95105651629515357211643933337938214340569863400000;
    const DOUBLE CRTM_15_4 =
        0.58778525229247301629891039327884007596190389052978;
    const DOUBLE CRTM_15_5 =
        0.50000000000000000000000000000000000000000000000000;
    const DOUBLE CRTM_15_6 =
        0.86602540378443864676372317075293618347140262690519;
    const DOUBLE CRTM_15_11 = 0.309016994374947424102293417182819058860154590;
    const DOUBLE CRTM_15_12 = 0.809016994374947424102293417182819058860154590;
    // Below CRTMs are the product of the above CRTMs, Precomputed to save
    // multiplications on the fly.
    // CRTM_15_7 = CRTM_15_6 * CRTM_15_4
    const DOUBLE CRTM_15_7 =
        0.50903696045256706468216979248996715975105181034577;
    // CRTM_15_8 = CRTM_15_6 * CRTM_15_3
    const DOUBLE CRTM_15_8 =
        0.82363910354633184270744116161596601637855195182647;
    // CRTM_15_9 = CRTM_15_6 * CRTM_15_1
    const DOUBLE CRTM_15_9 =
        0.48412291827592710612024388657479988457787393064252;
    // CRTM_15_10 = CRTM_15_6 * CRTM_15_2
    const DOUBLE CRTM_15_10 =
        0.21650635094610964914707551542960572987794876098633;

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
    INTP N = n >> 2;

    __m256d v_CRTM_15_1 = _mm256_broadcast_sd(&CRTM_15_1);
    __m256d v_CRTM_15_2 = _mm256_broadcast_sd(&CRTM_15_2);
    __m256d v_CRTM_15_3 = _mm256_broadcast_sd(&CRTM_15_3);
    __m256d v_CRTM_15_4 = _mm256_broadcast_sd(&CRTM_15_4);
    __m256d v_CRTM_15_5 = _mm256_broadcast_sd(&CRTM_15_5);
    __m256d v_CRTM_15_6 = _mm256_broadcast_sd(&CRTM_15_6);
    __m256d v_CRTM_15_7 = _mm256_broadcast_sd(&CRTM_15_7);
    __m256d v_CRTM_15_8 = _mm256_broadcast_sd(&CRTM_15_8);
    __m256d v_CRTM_15_9 = _mm256_broadcast_sd(&CRTM_15_9);
    __m256d v_CRTM_15_10 = _mm256_broadcast_sd(&CRTM_15_10);
    __m256d v_CRTM_15_11 = _mm256_broadcast_sd(&CRTM_15_11);
    __m256d v_CRTM_15_12 = _mm256_broadcast_sd(&CRTM_15_12);

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m256d av_in0, av_in1, av_in2, av_in3;
        __m256d av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
                av_s9, av_s10, av_s11, av_s12, av_s13, av_s15, av_s16, av_s17,
                av_s19, av_s20, av_s21, av_s22, av_s23, av_s24, av_s25, av_s26,
                av_s27, av_s28, av_s29, av_s30, av_s31;
        __m256d av_t0, av_t1, av_t2, av_t3;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
                v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21,
                v_out22, v_out23, v_out24, v_out25, v_out26, v_out27, v_out28,
                v_out29;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        curr_in = in + in_strides[0];
        LDR_256_D(curr_in, v_in_stride, av_in0);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_256_D(curr_in, v_in_stride, av_in1);
        // Input point 15: x(14)
        curr_in = in + in_strides[14];
        LDR_256_D(curr_in, v_in_stride, av_in2);
        // Input point 25: x(24)
        curr_in = in + in_strides[24];
        LDR_256_D(curr_in, v_in_stride, av_in3);

        av_s0 = _mm256_add_pd(av_in1, av_in2);
        av_s1 = _mm256_sub_pd(av_in1, av_in2);
        av_s2 = _mm256_add_pd(av_in3, av_s0);
        av_s3 = _mm256_sub_pd(av_in3, _mm256_mul_pd(v_CRTM_15_5, av_s0));

        // Input point 17: x(16)
        curr_in = in + in_strides[16];
        LDR_256_D(curr_in, v_in_stride, av_in1);
        // Input point 27: x(26)
        curr_in = in + in_strides[26];
        LDR_256_D(curr_in, v_in_stride, av_in2);
        // Input point 7: x(6)
        curr_in = in + in_strides[6];
        LDR_256_D(curr_in, v_in_stride, av_in3);

        av_s0 = _mm256_add_pd(av_in1, av_in2);
        av_s4 = _mm256_sub_pd(av_in2, av_in1);
        av_s5 = _mm256_add_pd(av_in3, av_s0);
        av_s6 = _mm256_sub_pd(av_in3, _mm256_mul_pd(v_CRTM_15_5, av_s0));

        av_s23 = _mm256_add_pd(av_s6, av_s3);
        av_s24 = _mm256_sub_pd(av_s6, av_s3);
        av_s26 = _mm256_add_pd(av_s4, av_s1);
        av_s27 = _mm256_sub_pd(av_s4, av_s1);

        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_256_D(curr_in, v_in_stride, av_in1);
        // Input point 23: x(22)
        curr_in = in + in_strides[22];
        LDR_256_D(curr_in, v_in_stride, av_in2);
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDR_256_D(curr_in, v_in_stride, av_in3);

        av_s0 = _mm256_add_pd(av_in1, av_in2);
        av_s1 = _mm256_sub_pd(av_in2, av_in1);
        av_s7 = _mm256_add_pd(av_in3, av_s0);
        av_s3 = _mm256_sub_pd(av_in3, _mm256_mul_pd(v_CRTM_15_5, av_s0));

        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDR_256_D(curr_in, v_in_stride, av_in1);
        // Input point 29: x(28)
        curr_in = in + in_strides[28];
        LDR_256_D(curr_in, v_in_stride, av_in2);
        // Input point 19: x(18)
        curr_in = in + in_strides[18];
        LDR_256_D(curr_in, v_in_stride, av_in3);

        av_s0 = _mm256_add_pd(av_in1, av_in2);
        av_s4 = _mm256_sub_pd(av_in2, av_in1);
        av_s8 = _mm256_add_pd(av_in3, av_s0);
        av_s6 = _mm256_sub_pd(av_in3, _mm256_mul_pd(v_CRTM_15_5, av_s0));

        av_s28 = _mm256_add_pd(av_s6, av_s3);
        av_s29 = _mm256_sub_pd(av_s3, av_s6);
        av_s30 = _mm256_add_pd(av_s4, av_s1);
        av_s31 = _mm256_sub_pd(av_s1, av_s4);

        // Input point 11: x(10)
        curr_in = in + in_strides[10];
        LDR_256_D(curr_in, v_in_stride, av_in1);
        // Input point 21: x(20)
        curr_in = in + in_strides[20];
        LDR_256_D(curr_in, v_in_stride, av_in2);

        av_s0 = _mm256_add_pd(av_in1, av_in2);
        av_t0 = _mm256_mul_pd(v_CRTM_15_6, _mm256_sub_pd(av_in2, av_in1));
        av_s9 = _mm256_add_pd(av_in0, av_s0);
        av_s3 = _mm256_sub_pd(av_in0, _mm256_mul_pd(v_CRTM_15_5, av_s0));

        av_s11 = _mm256_add_pd(av_s8, av_s7);
        av_s12 = _mm256_add_pd(av_s2, av_s5);
        av_s19 = _mm256_sub_pd(av_s5, av_s2);
        av_s20 = _mm256_sub_pd(av_s8, av_s7);
        av_s13 = _mm256_add_pd(av_s11, av_s12);
        av_t1 = _mm256_mul_pd(v_CRTM_15_1, _mm256_sub_pd(av_s11, av_s12));
        av_s15 = _mm256_sub_pd(av_s9, _mm256_mul_pd(v_CRTM_15_2, av_s13));

        // Output point 1: X(0)
        v_out0 = _mm256_add_pd(av_s9, av_s13);
        STR_256_D(curr_out, v_out_stride, v_out0);

        // Output point 12: X(11)
        v_out11 = _mm256_add_pd(av_s15, av_t1);
        // Output point 13: X(12)
        v_out12 = _mm256_add_pd(_mm256_mul_pd(v_CRTM_15_3, av_s20),
                                _mm256_mul_pd(v_CRTM_15_4, av_s19));
        curr_out = out + out_strides[11];
        STRI_2x256_D(curr_out, v_out_stride, v_out11, v_out12);

        // Output point 24: X(23)
        v_out23 = _mm256_sub_pd(av_s15, av_t1);
        // Output point 25: X(24)
        v_out24 = _mm256_sub_pd(_mm256_mul_pd(v_CRTM_15_4, av_s20),
                                _mm256_mul_pd(v_CRTM_15_3, av_s19));
        curr_out = out + out_strides[23];
        STRI_2x256_D(curr_out, v_out_stride, v_out23, v_out24);

        av_t2 = _mm256_mul_pd(v_CRTM_15_1, _mm256_sub_pd(av_s28, av_s23));
        av_s13 = _mm256_add_pd(av_s28, av_s23);

        // Output point 20: X(19)
        v_out19 = _mm256_add_pd(av_s13, av_s3);

        av_t3 = _mm256_mul_pd(v_CRTM_15_9, _mm256_add_pd(av_s30, av_s27));
        av_s17 = _mm256_sub_pd(av_s30, av_s27);

        // Output point 21: X(20)
        v_out20 = _mm256_sub_pd(_mm256_mul_pd(v_CRTM_15_6, av_s17), av_t0);
        curr_out = out + out_strides[19];
        STRI_2x256_D(curr_out, v_out_stride, v_out19, v_out20);

        av_s15 = _mm256_sub_pd(av_s3, _mm256_mul_pd(v_CRTM_15_2, av_s13));
        av_s20 = _mm256_add_pd(av_t0, _mm256_mul_pd(v_CRTM_15_10, av_s17));
        av_s21 = _mm256_sub_pd(av_s15, av_t2);
        av_s25 = _mm256_sub_pd(_mm256_mul_pd(v_CRTM_15_8, av_s26),
                               _mm256_mul_pd(v_CRTM_15_7, av_s31));

        // Output point 4: X(3)
        v_out3 = _mm256_add_pd(av_s21, av_s25);

        // Output point 16: X(15)
        v_out15 = _mm256_sub_pd(av_s21, av_s25);

        av_s21 = _mm256_add_pd(av_s15, av_t2);
        av_s23 = _mm256_add_pd(_mm256_mul_pd(v_CRTM_15_8, av_s31),
                               _mm256_mul_pd(v_CRTM_15_7, av_s26));

        // Output point 28: X(27)
        v_out27 = _mm256_add_pd(av_s21, av_s23);

        av_s22 = _mm256_add_pd(av_s20, av_t3);
        av_s16 = _mm256_add_pd(_mm256_mul_pd(v_CRTM_15_4, av_s29),
                               _mm256_mul_pd(v_CRTM_15_3, av_s24));

        // Output point 5: X(4)
        v_out4 = _mm256_sub_pd(av_s22, av_s16);
        curr_out = out + out_strides[3];
        STRI_2x256_D(curr_out, v_out_stride, v_out3, v_out4);
        // Output point 17: X(16)
        v_out16 = _mm256_add_pd(av_s22, av_s16);
        curr_out = out + out_strides[15];
        STRI_2x256_D(curr_out, v_out_stride, v_out15, v_out16);

        av_s22 = _mm256_sub_pd(av_s20, av_t3);
        av_s10 = _mm256_sub_pd(_mm256_mul_pd(v_CRTM_15_3, av_s29),
                               _mm256_mul_pd(v_CRTM_15_4, av_s24));

        // Output point 29: X(28)
        v_out28 = _mm256_add_pd(av_s22, av_s10);
        curr_out = out + out_strides[27];
        STRI_2x256_D(curr_out, v_out_stride, v_out27, v_out28);

        // Output point 8: X(7)
        v_out7 = _mm256_sub_pd(av_s21, av_s23);

        // Output point 9: X(8)
        v_out8 = _mm256_sub_pd(av_s10, av_s22);
        curr_out = out + out_strides[7];
        STRI_2x256_D(curr_out, v_out_stride, v_out7, v_out8);

        /* Shifted DFT */
        __m256d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
                bv_in8, bv_in9, bv_in10, bv_in11, bv_in12, bv_in13, bv_in14;
        __m256d bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
                bv_t10, bv_t11, bv_t12;
        __m256d bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
                bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17,
                bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24, bv_s25,
                bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32, bv_s33,
                bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40, bv_s41,
                bv_s42, bv_s43, bv_s44, bv_s45, bv_s46, bv_s47, bv_s48, bv_s49,
                bv_s50, bv_s51;

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
        // Input point 22: x(21)
        curr_in = in + in_strides[21];
        LDR_256_D(curr_in, v_in_stride, bv_in10);
        // Input point 24: x(23)
        curr_in = in + in_strides[23];
        LDR_256_D(curr_in, v_in_stride, bv_in11);
        // Input point 26: x(25)
        curr_in = in + in_strides[25];
        LDR_256_D(curr_in, v_in_stride, bv_in12);
        // Input point 28: x(27)
        curr_in = in + in_strides[27];
        LDR_256_D(curr_in, v_in_stride, bv_in13);
        // Input point 30: x(29)
        curr_in = in + in_strides[29];
        LDR_256_D(curr_in, v_in_stride, bv_in14);

        bv_s1 = _mm256_add_pd(bv_in1, bv_in4);
        bv_s2 = _mm256_add_pd(bv_in7, bv_in13);
        bv_s3 = _mm256_sub_pd(bv_in7, bv_in13);
        bv_s4 = _mm256_add_pd(bv_in6, bv_in9);
        bv_s5 = _mm256_add_pd(bv_in12, bv_in3);
        bv_s42 = _mm256_add_pd(_mm256_mul_pd(v_CRTM_15_4, bv_s5),
                               _mm256_mul_pd(v_CRTM_15_3, bv_s4));
        bv_s28 = _mm256_sub_pd(_mm256_mul_pd(v_CRTM_15_4, bv_s4),
                               _mm256_mul_pd(v_CRTM_15_3, bv_s5));

        bv_s6 = _mm256_add_pd(bv_in14, bv_in11);
        bv_s7 = _mm256_add_pd(bv_in2, bv_in8);
        bv_t2  = _mm256_mul_pd(v_CRTM_15_1, _mm256_sub_pd(bv_in14, bv_s7));
        bv_s47 = _mm256_sub_pd(bv_t2, _mm256_mul_pd(v_CRTM_15_11, bv_in11));
        bv_s30 = _mm256_sub_pd(_mm256_mul_pd(v_CRTM_15_12, bv_in11), bv_in5);

        bv_s8 = _mm256_sub_pd(bv_in2, bv_in8);
        bv_s9 = _mm256_add_pd(bv_in9, bv_in3);
        bv_s10 = _mm256_add_pd(bv_in6, bv_in12);
        bv_s26 = _mm256_add_pd(_mm256_mul_pd(v_CRTM_15_4, bv_s6),
                               _mm256_mul_pd(v_CRTM_15_3, bv_s8));

        bv_s11 = _mm256_sub_pd(bv_s9, bv_s10);
        bv_s12 = _mm256_add_pd(bv_in1, bv_s2);
        bv_s13 = _mm256_add_pd(bv_in14, bv_s7);
        bv_s48 = _mm256_add_pd(_mm256_mul_pd(v_CRTM_15_2, bv_s13), bv_in5);
        bv_s14 = _mm256_sub_pd(bv_in0, bv_s11);

        bv_s15 = _mm256_add_pd(bv_in10, bv_in4);
        bv_s16 = _mm256_sub_pd(bv_s15, bv_s12);
        bv_s17 = _mm256_add_pd(bv_in11, bv_in5);
        bv_s18 = _mm256_sub_pd(bv_s13, bv_s17);
        bv_s19 = _mm256_add_pd(bv_s16, bv_s18);
        bv_s20 = _mm256_add_pd(bv_in0, _mm256_mul_pd(v_CRTM_15_2, bv_s11));
        // Output pt 30: X(29)
        v_out29 = _mm256_add_pd(bv_s19, bv_s14);
        curr_out = out + out_strides[29];
        STR_256_D(curr_out, v_out_stride, v_out29);
        // Output pt 10: X(9) & Output pt 11: X(10)
        v_out9 = _mm256_sub_pd(bv_s14, _mm256_mul_pd(v_CRTM_15_5, bv_s19));
        v_out10 = _mm256_mul_pd(v_CRTM_15_6, _mm256_sub_pd(bv_s16, bv_s18));
        curr_out = out + out_strides[9];
        STRI_2x256_D(curr_out, v_out_stride, v_out9, v_out10);

        bv_s21 = _mm256_add_pd(bv_in9, bv_in12);
        bv_s22 = _mm256_add_pd(bv_in6, bv_in3);
        bv_s23 = _mm256_sub_pd(bv_s21, bv_s22);
        bv_t1  = _mm256_mul_pd(v_CRTM_15_1, bv_s23);
        bv_s24 = _mm256_add_pd(bv_s20, bv_t1);
        bv_s25 = _mm256_sub_pd(_mm256_mul_pd(v_CRTM_15_3, bv_s3),
                               _mm256_mul_pd(v_CRTM_15_4, bv_s1));

        bv_t6  = _mm256_mul_pd(v_CRTM_15_6, _mm256_add_pd(bv_s25, bv_s26));
        bv_s27 = _mm256_sub_pd(bv_s25, bv_s26);

        bv_s29 = _mm256_sub_pd(_mm256_mul_pd(v_CRTM_15_5, bv_s27), bv_s28);
        bv_s31 = _mm256_add_pd(_mm256_mul_pd(v_CRTM_15_2, bv_s13), bv_t2);
        bv_s32 = _mm256_sub_pd(bv_s30, bv_s31);

        bv_t3  = _mm256_mul_pd(v_CRTM_15_1, _mm256_sub_pd(bv_in1, bv_s2));
        bv_s33 = _mm256_add_pd(bv_in10, bv_t3);
        bv_t4  = _mm256_mul_pd(v_CRTM_15_2, bv_s12);
        bv_s34 = _mm256_sub_pd(bv_t4, _mm256_mul_pd(v_CRTM_15_12, bv_in4));
        bv_s35 = _mm256_add_pd(bv_s33, bv_s34);
        bv_s36 = _mm256_add_pd(bv_s32, bv_s35);
        bv_t5  = _mm256_mul_pd(v_CRTM_15_6, _mm256_sub_pd(bv_s32, bv_s35));
        // Output pt 6: X(5) & Output pt 7: X(6)
        v_out5 = _mm256_add_pd(bv_s24, bv_s36);
        v_out6 = _mm256_add_pd(bv_s28, bv_s27);
        curr_out = out + out_strides[5];
        STRI_2x256_D(curr_out, v_out_stride, v_out5, v_out6);

        bv_s37 = _mm256_sub_pd(bv_s24, _mm256_mul_pd(v_CRTM_15_5, bv_s36));
        // Output pt 14: X(13)& Output pt 15: X(14)
        v_out13 = _mm256_sub_pd(bv_s37, bv_t6);
        v_out14 = _mm256_add_pd(bv_t5, bv_s29);
        curr_out = out + out_strides[13];
        STRI_2x256_D(curr_out, v_out_stride, v_out13, v_out14);
        // Output pt 26: X(25) & Output pt 27: X(26)
        v_out25 = _mm256_add_pd(bv_s37, bv_t6);
        v_out26 = _mm256_sub_pd(bv_t5, bv_s29);
        curr_out = out + out_strides[25];
        STRI_2x256_D(curr_out, v_out_stride, v_out25, v_out26);

        bv_t7  = _mm256_mul_pd(v_CRTM_15_4, bv_s3);
        bv_t8  = _mm256_mul_pd(v_CRTM_15_3, bv_s1);
        bv_s38 = _mm256_add_pd(bv_t8, bv_t7);
        bv_t9  = _mm256_mul_pd(v_CRTM_15_4, bv_s8);
        bv_t10 = _mm256_mul_pd(v_CRTM_15_3, bv_s6);
        bv_s39 = _mm256_sub_pd(bv_t9, bv_t10);

        bv_t11 = _mm256_mul_pd(v_CRTM_15_6, _mm256_add_pd(bv_s38, bv_s39));
        bv_s40 = _mm256_sub_pd(bv_s20, bv_t1);
        bv_s41 = _mm256_sub_pd(bv_s39, bv_s38);
        bv_s43 = _mm256_sub_pd(_mm256_mul_pd(v_CRTM_15_5, bv_s41), bv_s42);

        bv_s44 = _mm256_sub_pd(bv_t4, bv_t3);
        bv_s45 = _mm256_add_pd(bv_in10, _mm256_mul_pd(v_CRTM_15_11, bv_in4));
        bv_s46 = _mm256_add_pd(bv_s45, bv_s44);
        bv_s49 = _mm256_sub_pd(bv_s47, bv_s48);
        bv_s50 = _mm256_add_pd(bv_s46, bv_s49);
        bv_t12 = _mm256_mul_pd(v_CRTM_15_6, _mm256_sub_pd(bv_s49, bv_s46));
        bv_s51 = _mm256_sub_pd(bv_s40, _mm256_mul_pd(v_CRTM_15_5, bv_s50));

        // Output pt 2: X(1) & Output pt 3: X(2)
        v_out1 = _mm256_add_pd(bv_s51, bv_t11);
        v_out2 = _mm256_add_pd(bv_s43, bv_t12);
        curr_out = out + out_strides[1];
        STRI_2x256_D(curr_out, v_out_stride, v_out1, v_out2);
        // Output pt 18: X(17) & Output pt 19: X(18)
        v_out17 = _mm256_add_pd(bv_s40, bv_s50);
        v_out18 = _mm256_add_pd(bv_s42, bv_s41);
        curr_out = out + out_strides[17];
        STRI_2x256_D(curr_out, v_out_stride, v_out17, v_out18);
        // Output pt 22: X(21) & Output pt 23: X(22)
        v_out21 = _mm256_sub_pd(bv_s51, bv_t11);
        v_out22 = _mm256_sub_pd(bv_s43, bv_t12);
        curr_out = out + out_strides[21];
        STRI_2x256_D(curr_out, v_out_stride, v_out21, v_out22);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (n & 2)
    {
        /* Standard DFT */
        __m128d av_in0, av_in1, av_in2, av_in3;
        __m128d av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
                av_s9, av_s10, av_s11, av_s12, av_s13, av_s15, av_s16, av_s17,
                av_s19, av_s20, av_s21, av_s22, av_s23, av_s24, av_s25, av_s26,
                av_s27, av_s28, av_s29, av_s30, av_s31;
        __m128d av_t0, av_t1, av_t2, av_t3;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
                v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21,
                v_out22, v_out23, v_out24, v_out25, v_out26, v_out27, v_out28,
                v_out29;

        curr_in = in;
        curr_out = out;

        __m128d v128_CRTM_15_1 = _mm256_castpd256_pd128(v_CRTM_15_1);
        __m128d v128_CRTM_15_2 = _mm256_castpd256_pd128(v_CRTM_15_2);
        __m128d v128_CRTM_15_3 = _mm256_castpd256_pd128(v_CRTM_15_3);
        __m128d v128_CRTM_15_4 = _mm256_castpd256_pd128(v_CRTM_15_4);
        __m128d v128_CRTM_15_5 = _mm256_castpd256_pd128(v_CRTM_15_5);
        __m128d v128_CRTM_15_6 = _mm256_castpd256_pd128(v_CRTM_15_6);
        __m128d v128_CRTM_15_7 = _mm256_castpd256_pd128(v_CRTM_15_7);
        __m128d v128_CRTM_15_8 = _mm256_castpd256_pd128(v_CRTM_15_8);
        __m128d v128_CRTM_15_9 = _mm256_castpd256_pd128(v_CRTM_15_9);
        __m128d v128_CRTM_15_10 = _mm256_castpd256_pd128(v_CRTM_15_10);
        __m128d v128_CRTM_15_11 = _mm256_castpd256_pd128(v_CRTM_15_11);
        __m128d v128_CRTM_15_12 = _mm256_castpd256_pd128(v_CRTM_15_12);

        // Input point 1: x(0)
        curr_in = in + in_strides[0];
        LDR_128_D(curr_in, v_in_stride, av_in0);
        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_128_D(curr_in, v_in_stride, av_in1);
        // Input point 15: x(14)
        curr_in = in + in_strides[14];
        LDR_128_D(curr_in, v_in_stride, av_in2);
        // Input point 25: x(24)
        curr_in = in + in_strides[24];
        LDR_128_D(curr_in, v_in_stride, av_in3);

        av_s0 = _mm_add_pd(av_in1, av_in2);
        av_s1 = _mm_sub_pd(av_in1, av_in2);
        av_s2 = _mm_add_pd(av_in3, av_s0);
        av_s3 = _mm_sub_pd(av_in3, _mm_mul_pd(v128_CRTM_15_5, av_s0));

        // Input point 17: x(16)
        curr_in = in + in_strides[16];
        LDR_128_D(curr_in, v_in_stride, av_in1);
        // Input point 27: x(26)
        curr_in = in + in_strides[26];
        LDR_128_D(curr_in, v_in_stride, av_in2);
        // Input point 7: x(6)
        curr_in = in + in_strides[6];
        LDR_128_D(curr_in, v_in_stride, av_in3);

        av_s0 = _mm_add_pd(av_in1, av_in2);
        av_s4 = _mm_sub_pd(av_in2, av_in1);
        av_s5 = _mm_add_pd(av_in3, av_s0);
        av_s6 = _mm_sub_pd(av_in3, _mm_mul_pd(v128_CRTM_15_5, av_s0));

        av_s23 = _mm_add_pd(av_s6, av_s3);
        av_s24 = _mm_sub_pd(av_s6, av_s3);
        av_s26 = _mm_add_pd(av_s4, av_s1);
        av_s27 = _mm_sub_pd(av_s4, av_s1);

        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_128_D(curr_in, v_in_stride, av_in1);
        // Input point 23: x(22)
        curr_in = in + in_strides[22];
        LDR_128_D(curr_in, v_in_stride, av_in2);
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDR_128_D(curr_in, v_in_stride, av_in3);

        av_s0 = _mm_add_pd(av_in1, av_in2);
        av_s1 = _mm_sub_pd(av_in2, av_in1);
        av_s7 = _mm_add_pd(av_in3, av_s0);
        av_s3 = _mm_sub_pd(av_in3, _mm_mul_pd(v128_CRTM_15_5, av_s0));

        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDR_128_D(curr_in, v_in_stride, av_in1);
        // Input point 29: x(28)
        curr_in = in + in_strides[28];
        LDR_128_D(curr_in, v_in_stride, av_in2);
        // Input point 19: x(18)
        curr_in = in + in_strides[18];
        LDR_128_D(curr_in, v_in_stride, av_in3);

        av_s0 = _mm_add_pd(av_in1, av_in2);
        av_s4 = _mm_sub_pd(av_in2, av_in1);
        av_s8 = _mm_add_pd(av_in3, av_s0);
        av_s6 = _mm_sub_pd(av_in3, _mm_mul_pd(v128_CRTM_15_5, av_s0));

        av_s28 = _mm_add_pd(av_s6, av_s3);
        av_s29 = _mm_sub_pd(av_s3, av_s6);
        av_s30 = _mm_add_pd(av_s4, av_s1);
        av_s31 = _mm_sub_pd(av_s1, av_s4);

        // Input point 11: x(10)
        curr_in = in + in_strides[10];
        LDR_128_D(curr_in, v_in_stride, av_in1);
        // Input point 21: x(20)
        curr_in = in + in_strides[20];
        LDR_128_D(curr_in, v_in_stride, av_in2);

        av_s0 = _mm_add_pd(av_in1, av_in2);
        av_t0 = _mm_mul_pd(v128_CRTM_15_6, _mm_sub_pd(av_in2, av_in1));
        av_s9 = _mm_add_pd(av_in0, av_s0);
        av_s3 = _mm_sub_pd(av_in0, _mm_mul_pd(v128_CRTM_15_5, av_s0));

        av_s11 = _mm_add_pd(av_s8, av_s7);
        av_s12 = _mm_add_pd(av_s2, av_s5);
        av_s19 = _mm_sub_pd(av_s5, av_s2);
        av_s20 = _mm_sub_pd(av_s8, av_s7);
        av_s13 = _mm_add_pd(av_s11, av_s12);
        av_t1 = _mm_mul_pd(v128_CRTM_15_1, _mm_sub_pd(av_s11, av_s12));
        av_s15 = _mm_sub_pd(av_s9, _mm_mul_pd(v128_CRTM_15_2, av_s13));

        // Output point 1: X(0)
        v_out0 = _mm_add_pd(av_s9, av_s13);
        STR_128_D(curr_out, v_out_stride, v_out0);

        // Output point 12: X(11)
        v_out11 = _mm_add_pd(av_s15, av_t1);
        // Output point 13: X(12)
        v_out12 = _mm_add_pd(_mm_mul_pd(v128_CRTM_15_3, av_s20),
                             _mm_mul_pd(v128_CRTM_15_4, av_s19));
        curr_out = out + out_strides[11];
        STRI_2x128_D(curr_out, v_out_stride, v_out11, v_out12);

        // Output point 24: X(23)
        v_out23 = _mm_sub_pd(av_s15, av_t1);
        // Output point 25: X(24)
        v_out24 = _mm_sub_pd(_mm_mul_pd(v128_CRTM_15_4, av_s20),
                             _mm_mul_pd(v128_CRTM_15_3, av_s19));
        curr_out = out + out_strides[23];
        STRI_2x128_D(curr_out, v_out_stride, v_out23, v_out24);

        av_t2 = _mm_mul_pd(v128_CRTM_15_1, _mm_sub_pd(av_s28, av_s23));
        av_s13 = _mm_add_pd(av_s28, av_s23);

        // Output point 20: X(19)
        v_out19 = _mm_add_pd(av_s13, av_s3);

        av_t3 = _mm_mul_pd(v128_CRTM_15_9, _mm_add_pd(av_s30, av_s27));
        av_s17 = _mm_sub_pd(av_s30, av_s27);

        // Output point 21: X(20)
        v_out20 = _mm_sub_pd(_mm_mul_pd(v128_CRTM_15_6, av_s17), av_t0);
        curr_out = out + out_strides[19];
        STRI_2x128_D(curr_out, v_out_stride, v_out19, v_out20);

        av_s15 = _mm_sub_pd(av_s3, _mm_mul_pd(v128_CRTM_15_2, av_s13));
        av_s20 = _mm_add_pd(av_t0, _mm_mul_pd(v128_CRTM_15_10, av_s17));
        av_s21 = _mm_sub_pd(av_s15, av_t2);
        av_s25 = _mm_sub_pd(_mm_mul_pd(v128_CRTM_15_8, av_s26),
                            _mm_mul_pd(v128_CRTM_15_7, av_s31));

        // Output point 4: X(3)
        v_out3 = _mm_add_pd(av_s21, av_s25);

        // Output point 16: X(15)
        v_out15 = _mm_sub_pd(av_s21, av_s25);

        av_s21 = _mm_add_pd(av_s15, av_t2);
        av_s23 = _mm_add_pd(_mm_mul_pd(v128_CRTM_15_8, av_s31),
                            _mm_mul_pd(v128_CRTM_15_7, av_s26));

        // Output point 28: X(27)
        v_out27 = _mm_add_pd(av_s21, av_s23);

        av_s22 = _mm_add_pd(av_s20, av_t3);
        av_s16 = _mm_add_pd(_mm_mul_pd(v128_CRTM_15_4, av_s29),
                            _mm_mul_pd(v128_CRTM_15_3, av_s24));

        // Output point 5: X(4)
        v_out4 = _mm_sub_pd(av_s22, av_s16);
        curr_out = out + out_strides[3];
        STRI_2x128_D(curr_out, v_out_stride, v_out3, v_out4);
        // Output point 17: X(16)
        v_out16 = _mm_add_pd(av_s22, av_s16);
        curr_out = out + out_strides[15];
        STRI_2x128_D(curr_out, v_out_stride, v_out15, v_out16);

        av_s22 = _mm_sub_pd(av_s20, av_t3);
        av_s10 = _mm_sub_pd(_mm_mul_pd(v128_CRTM_15_3, av_s29),
                            _mm_mul_pd(v128_CRTM_15_4, av_s24));

        // Output point 29: X(28)
        v_out28 = _mm_add_pd(av_s22, av_s10);
        curr_out = out + out_strides[27];
        STRI_2x128_D(curr_out, v_out_stride, v_out27, v_out28);

        // Output point 8: X(7)
        v_out7 = _mm_sub_pd(av_s21, av_s23);

        // Output point 9: X(8)
        v_out8 = _mm_sub_pd(av_s10, av_s22);
        curr_out = out + out_strides[7];
        STRI_2x128_D(curr_out, v_out_stride, v_out7, v_out8);

        /* Shifted DFT */
        __m128d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
                bv_in8, bv_in9, bv_in10, bv_in11, bv_in12, bv_in13, bv_in14;
        __m128d bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
                bv_t10, bv_t11, bv_t12;
        __m128d bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
                bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17,
                bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24, bv_s25,
                bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32, bv_s33,
                bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40, bv_s41,
                bv_s42, bv_s43, bv_s44, bv_s45, bv_s46, bv_s47, bv_s48, bv_s49,
                bv_s50, bv_s51;

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
        // Input point 22: x(21)
        curr_in = in + in_strides[21];
        LDR_128_D(curr_in, v_in_stride, bv_in10);
        // Input point 24: x(23)
        curr_in = in + in_strides[23];
        LDR_128_D(curr_in, v_in_stride, bv_in11);
        // Input point 26: x(25)
        curr_in = in + in_strides[25];
        LDR_128_D(curr_in, v_in_stride, bv_in12);
        // Input point 28: x(27)
        curr_in = in + in_strides[27];
        LDR_128_D(curr_in, v_in_stride, bv_in13);
        // Input point 30: x(29)
        curr_in = in + in_strides[29];
        LDR_128_D(curr_in, v_in_stride, bv_in14);

        bv_s1 = _mm_add_pd(bv_in1, bv_in4);
        bv_s2 = _mm_add_pd(bv_in7, bv_in13);
        bv_s3 = _mm_sub_pd(bv_in7, bv_in13);
        bv_s4 = _mm_add_pd(bv_in6, bv_in9);
        bv_s5 = _mm_add_pd(bv_in12, bv_in3);
        bv_s42 = _mm_add_pd(_mm_mul_pd(v128_CRTM_15_4, bv_s5),
                            _mm_mul_pd(v128_CRTM_15_3, bv_s4));
        bv_s28 = _mm_sub_pd(_mm_mul_pd(v128_CRTM_15_4, bv_s4),
                            _mm_mul_pd(v128_CRTM_15_3, bv_s5));

        bv_s6 = _mm_add_pd(bv_in14, bv_in11);
        bv_s7 = _mm_add_pd(bv_in2, bv_in8);
        bv_t2  = _mm_mul_pd(v128_CRTM_15_1, _mm_sub_pd(bv_in14, bv_s7));
        bv_s47 = _mm_sub_pd(bv_t2, _mm_mul_pd(v128_CRTM_15_11, bv_in11));
        bv_s30 = _mm_sub_pd(_mm_mul_pd(v128_CRTM_15_12, bv_in11), bv_in5);

        bv_s8 = _mm_sub_pd(bv_in2, bv_in8);
        bv_s9 = _mm_add_pd(bv_in9, bv_in3);
        bv_s10 = _mm_add_pd(bv_in6, bv_in12);
        bv_s26 = _mm_add_pd(_mm_mul_pd(v128_CRTM_15_4, bv_s6),
                            _mm_mul_pd(v128_CRTM_15_3, bv_s8));

        bv_s11 = _mm_sub_pd(bv_s9, bv_s10);
        bv_s12 = _mm_add_pd(bv_in1, bv_s2);
        bv_s13 = _mm_add_pd(bv_in14, bv_s7);
        bv_s48 = _mm_add_pd(_mm_mul_pd(v128_CRTM_15_2, bv_s13), bv_in5);
        bv_s14 = _mm_sub_pd(bv_in0, bv_s11);

        bv_s15 = _mm_add_pd(bv_in10, bv_in4);
        bv_s16 = _mm_sub_pd(bv_s15, bv_s12);
        bv_s17 = _mm_add_pd(bv_in11, bv_in5);
        bv_s18 = _mm_sub_pd(bv_s13, bv_s17);
        bv_s19 = _mm_add_pd(bv_s16, bv_s18);
        bv_s20 = _mm_add_pd(bv_in0, _mm_mul_pd(v128_CRTM_15_2, bv_s11));
        // Output pt 30: X(29)
        v_out29 = _mm_add_pd(bv_s19, bv_s14);
        curr_out = out + out_strides[29];
        STR_128_D(curr_out, v_out_stride, v_out29);
        // Output pt 10: X(9) & Output pt 11: X(10)
        v_out9 = _mm_sub_pd(bv_s14, _mm_mul_pd(v128_CRTM_15_5, bv_s19));
        v_out10 = _mm_mul_pd(v128_CRTM_15_6, _mm_sub_pd(bv_s16, bv_s18));
        curr_out = out + out_strides[9];
        STRI_2x128_D(curr_out, v_out_stride, v_out9, v_out10);

        bv_s21 = _mm_add_pd(bv_in9, bv_in12);
        bv_s22 = _mm_add_pd(bv_in6, bv_in3);
        bv_s23 = _mm_sub_pd(bv_s21, bv_s22);
        bv_t1  = _mm_mul_pd(v128_CRTM_15_1, bv_s23);
        bv_s24 = _mm_add_pd(bv_s20, bv_t1);
        bv_s25 = _mm_sub_pd(_mm_mul_pd(v128_CRTM_15_3, bv_s3),
                            _mm_mul_pd(v128_CRTM_15_4, bv_s1));

        bv_t6  = _mm_mul_pd(v128_CRTM_15_6, _mm_add_pd(bv_s25, bv_s26));
        bv_s27 = _mm_sub_pd(bv_s25, bv_s26);

        bv_s29 = _mm_sub_pd(_mm_mul_pd(v128_CRTM_15_5, bv_s27), bv_s28);
        bv_s31 = _mm_add_pd(_mm_mul_pd(v128_CRTM_15_2, bv_s13), bv_t2);
        bv_s32 = _mm_sub_pd(bv_s30, bv_s31);

        bv_t3  = _mm_mul_pd(v128_CRTM_15_1, _mm_sub_pd(bv_in1, bv_s2));
        bv_s33 = _mm_add_pd(bv_in10, bv_t3);
        bv_t4  = _mm_mul_pd(v128_CRTM_15_2, bv_s12);
        bv_s34 = _mm_sub_pd(bv_t4, _mm_mul_pd(v128_CRTM_15_12, bv_in4));
        bv_s35 = _mm_add_pd(bv_s33, bv_s34);
        bv_s36 = _mm_add_pd(bv_s32, bv_s35);
        bv_t5  = _mm_mul_pd(v128_CRTM_15_6, _mm_sub_pd(bv_s32, bv_s35));
        // Output pt 6: X(5) & Output pt 7: X(6)
        v_out5 = _mm_add_pd(bv_s24, bv_s36);
        v_out6 = _mm_add_pd(bv_s28, bv_s27);
        curr_out = out + out_strides[5];
        STRI_2x128_D(curr_out, v_out_stride, v_out5, v_out6);

        bv_s37 = _mm_sub_pd(bv_s24, _mm_mul_pd(v128_CRTM_15_5, bv_s36));
        // Output pt 14: X(13)& Output pt 15: X(14)
        v_out13 = _mm_sub_pd(bv_s37, bv_t6);
        v_out14 = _mm_add_pd(bv_t5, bv_s29);
        curr_out = out + out_strides[13];
        STRI_2x128_D(curr_out, v_out_stride, v_out13, v_out14);
        // Output pt 26: X(25) & Output pt 27: X(26)
        v_out25 = _mm_add_pd(bv_s37, bv_t6);
        v_out26 = _mm_sub_pd(bv_t5, bv_s29);
        curr_out = out + out_strides[25];
        STRI_2x128_D(curr_out, v_out_stride, v_out25, v_out26);

        bv_t7  = _mm_mul_pd(v128_CRTM_15_4, bv_s3);
        bv_t8  = _mm_mul_pd(v128_CRTM_15_3, bv_s1);
        bv_s38 = _mm_add_pd(bv_t8, bv_t7);
        bv_t9  = _mm_mul_pd(v128_CRTM_15_4, bv_s8);
        bv_t10 = _mm_mul_pd(v128_CRTM_15_3, bv_s6);
        bv_s39 = _mm_sub_pd(bv_t9, bv_t10);

        bv_t11 = _mm_mul_pd(v128_CRTM_15_6, _mm_add_pd(bv_s38, bv_s39));
        bv_s40 = _mm_sub_pd(bv_s20, bv_t1);
        bv_s41 = _mm_sub_pd(bv_s39, bv_s38);
        bv_s43 = _mm_sub_pd(_mm_mul_pd(v128_CRTM_15_5, bv_s41), bv_s42);

        bv_s44 = _mm_sub_pd(bv_t4, bv_t3);
        bv_s45 = _mm_add_pd(bv_in10, _mm_mul_pd(v128_CRTM_15_11, bv_in4));
        bv_s46 = _mm_add_pd(bv_s45, bv_s44);
        bv_s49 = _mm_sub_pd(bv_s47, bv_s48);
        bv_s50 = _mm_add_pd(bv_s46, bv_s49);
        bv_t12 = _mm_mul_pd(v128_CRTM_15_6, _mm_sub_pd(bv_s49, bv_s46));
        bv_s51 = _mm_sub_pd(bv_s40, _mm_mul_pd(v128_CRTM_15_5, bv_s50));

        // Output pt 2: X(1) & Output pt 3: X(2)
        v_out1 = _mm_add_pd(bv_s51, bv_t11);
        v_out2 = _mm_add_pd(bv_s43, bv_t12);
        curr_out = out + out_strides[1];
        STRI_2x128_D(curr_out, v_out_stride, v_out1, v_out2);
        // Output pt 18: X(17) & Output pt 19: X(18)
        v_out17 = _mm_add_pd(bv_s40, bv_s50);
        v_out18 = _mm_add_pd(bv_s42, bv_s41);
        curr_out = out + out_strides[17];
        STRI_2x128_D(curr_out, v_out_stride, v_out17, v_out18);
        // Output pt 22: X(21) & Output pt 23: X(22)
        v_out21 = _mm_sub_pd(bv_s51, bv_t11);
        v_out22 = _mm_sub_pd(bv_s43, bv_t12);
        curr_out = out + out_strides[21];
        STRI_2x128_D(curr_out, v_out_stride, v_out21, v_out22);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        /* Standard DFT*/
        DOUBLE a_in0, a_in1, a_in2, a_in3, a_s0, a_s1, a_s2, a_s3, a_s4, a_s5,
               a_s6, a_s7, a_s8, a_s9, a_s10, a_s11, a_s12, a_s13, a_s15, a_s16,
               a_s17, a_s19, a_s20, a_s21, a_s22, a_s23, a_s24, a_s25, a_s26,
               a_s27, a_s28, a_s29, a_s30, a_s31, a_t0, a_t1, a_t2, a_t3;

        a_in0 = *in;
        a_in1 = in[in_strides[4]];
        a_in2 = in[in_strides[14]];
        a_in3 = in[in_strides[24]];

        a_s0 = a_in1 + a_in2;
        a_s1 = a_in1 - a_in2;
        a_s2 = a_in3 + a_s0;
        a_s3 = a_in3 - (CRTM_15_5 * a_s0);

        a_in1 = in[in_strides[16]];
        a_in2 = in[in_strides[26]];
        a_in3 = in[in_strides[6]];

        a_s0 = a_in1 + a_in2;
        a_s4 = a_in2 - a_in1;
        a_s5 = a_in3 + a_s0;
        a_s6 = a_in3 - (CRTM_15_5 * a_s0);

        a_s23 = a_s6 + a_s3;
        a_s24 = a_s6 - a_s3;
        a_s26 = a_s4 + a_s1;
        a_s27 = a_s4 - a_s1;

        a_in1 = in[in_strides[2]];
        a_in2 = in[in_strides[22]];
        a_in3 = in[in_strides[12]];

        a_s0 = a_in1 + a_in2;
        a_s1 = a_in2 - a_in1;
        a_s7 = a_in3 + a_s0;
        a_s3 = a_in3 - (CRTM_15_5 * a_s0);

        a_in1 = in[in_strides[8]];
        a_in2 = in[in_strides[28]];
        a_in3 = in[in_strides[18]];

        a_s0 = a_in1 + a_in2;
        a_s4 = a_in2 - a_in1;
        a_s8 = a_in3 + a_s0;
        a_s6 = a_in3 - (CRTM_15_5 * a_s0);

        a_s28 = a_s6 + a_s3;
        a_s29 = a_s3 - a_s6;
        a_s30 = a_s4 + a_s1;
        a_s31 = a_s1 - a_s4;

        a_in1 = in[in_strides[10]];
        a_in2 = in[in_strides[20]];

        a_s0 = a_in1 + a_in2;
        a_t0 = CRTM_15_6 * (a_in2 - a_in1);
        a_s9 = a_in0 + a_s0;
        a_s3 = a_in0 - (CRTM_15_5 * a_s0);

        a_s11 = a_s8 + a_s7;
        a_s12 = a_s2 + a_s5;
        a_s19 = a_s5 - a_s2;
        a_s20 = a_s8 - a_s7;
        a_s13 = a_s11 + a_s12;
        a_t1 = CRTM_15_1 * (a_s11 - a_s12);
        a_s15 = a_s9 - (CRTM_15_2 * a_s13);

        *out = a_s9 + a_s13;

        out[out_strides[11]] = a_s15 + a_t1;
        out[out_strides[12]] = (CRTM_15_3 * a_s20) + (CRTM_15_4 * a_s19);

        out[out_strides[23]] = a_s15 - a_t1;
        out[out_strides[24]] = (CRTM_15_4 * a_s20) - (CRTM_15_3 * a_s19);

        a_t2 = CRTM_15_1 * (a_s28 - a_s23);
        a_s13 = a_s28 + a_s23;

        out[out_strides[19]] = a_s13 + a_s3;

        a_t3 = CRTM_15_9 * (a_s30 + a_s27);
        a_s17 = a_s30 - a_s27;

        out[out_strides[20]] = CRTM_15_6 * a_s17 - a_t0;

        a_s15 = a_s3 - (CRTM_15_2 * a_s13);
        a_s20 = a_t0 + (CRTM_15_10 * a_s17);
        a_s21 = a_s15 - a_t2;
        a_s25 = (CRTM_15_8 * a_s26) - (CRTM_15_7 * a_s31);

        out[out_strides[3]] = a_s21 + a_s25;
        out[out_strides[15]] = a_s21 - a_s25;

        a_s21 = a_s15 + a_t2;
        a_s23 = (CRTM_15_8 * a_s31) + (CRTM_15_7 * a_s26);

        out[out_strides[27]] = a_s21 + a_s23;

        a_s22 = a_s20 + a_t3;
        a_s16 = (CRTM_15_4 * a_s29) + (CRTM_15_3 * a_s24);

        out[out_strides[4]] = a_s22 - a_s16;
        out[out_strides[16]] = a_s22 + a_s16;

        a_s22 = a_s20 - a_t3;
        a_s10 = (CRTM_15_3 * a_s29) - (CRTM_15_4 * a_s24);
        out[out_strides[28]] = a_s22 + a_s10;

        out[out_strides[7]] = a_s21 - a_s23;
        out[out_strides[8]] = a_s10 - a_s22;

        /* Shifted DFT */
        DOUBLE b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7, b_in8,
               b_in9, b_in10, b_in11, b_in12, b_in13, b_in14;
        DOUBLE b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9,
               b_s10, b_s11, b_s12, b_s13, b_s14, b_s15, b_s16, b_s17, b_s18,
               b_s19, b_s20, b_s21, b_s22, b_s23, b_s24, b_s25, b_s26, b_s27,
               b_s28, b_s29, b_s30, b_s31, b_s32, b_s33, b_s34, b_s35, b_s36,
               b_s37, b_s38, b_s39, b_s40, b_s41, b_s42;
        DOUBLE b_t0, b_t1, b_t2, b_t3, b_t4, b_t5, b_t6, b_t7, b_t8, b_t9,
               b_t10, b_t11, b_t12, b_t13, b_t14, b_t15, b_t16, b_t17, b_t18,
               b_t19;

        b_in0 = in[in_strides[1]];
        b_in1 = in[in_strides[3]];
        b_in2 = in[in_strides[5]];
        b_in3 = in[in_strides[7]];
        b_in4 = in[in_strides[9]];
        b_in5 = in[in_strides[11]];
        b_in6 = in[in_strides[13]];
        b_in7 = in[in_strides[15]];
        b_in8 = in[in_strides[17]];
        b_in9 = in[in_strides[19]];
        b_in10 = in[in_strides[21]];
        b_in11 = in[in_strides[23]];
        b_in12 = in[in_strides[25]];
        b_in13 = in[in_strides[27]];
        b_in14 = in[in_strides[29]];

        b_s0 = b_in1 + b_in4;
        b_s1 = b_in7 + b_in13;
        b_s2 = b_in7 - b_in13;
        b_s3 = b_in6 + b_in9;
        b_s4 = b_in12 + b_in3;
        b_s5 = b_in14 + b_in11;
        b_s6 = b_in2 + b_in8;
        b_s7 = b_in2 - b_in8;
        b_s8 = b_in9 + b_in3;
        b_s9 = b_in6 + b_in12;

        b_s10 = b_s8 - b_s9;
        b_s11 = b_in1 + b_s1;
        b_s12 = b_in14 + b_s6;
        b_s13 = b_in0 - b_s10;

        b_s14 = b_in10 + b_in4;
        b_s15 = b_s14 - b_s11;
        b_s16 = b_in11 + b_in5;
        b_s17 = b_s12 - b_s16;
        b_s18 = b_s15 + b_s17;
        b_s19 = b_in0 + CRTM_15_2 * (b_s10);
        b_s20 = b_in9 + b_in12;
        b_s21 = b_in6 + b_in3;
        b_s22 = b_s20 - b_s21;
        b_t0 = CRTM_15_1 * (b_s22);
        b_s23 = b_s19 + b_t0;
        b_t1 = CRTM_15_3 * b_s2 - CRTM_15_4 * b_s0;
        b_t2 = CRTM_15_4 * b_s5 + CRTM_15_3 * b_s7;
        b_s24 = b_t1 - b_t2;
        b_t3 = CRTM_15_4 * b_s3 - CRTM_15_3 * b_s4;

        b_t4 = CRTM_15_5 * b_s24 - b_t3;
        b_t5 = CRTM_15_12 * b_in11 - b_in5;
        b_t6 = CRTM_15_1 * (b_in14 - b_s6);
        b_t7 = CRTM_15_2 * b_s12 + b_t6;
        b_s25 = b_t5 - b_t7;
        b_t8 = CRTM_15_1 * (b_in1 - b_s1);
        b_s26 = b_in10 + b_t8;
        b_t9 = CRTM_15_2 * b_s11;
        b_s27 = b_t9 - CRTM_15_12 * b_in4;
        b_s28  = b_s26 + b_s27;
        b_s29 = b_s25 + b_s28;
        b_t10 = CRTM_15_6 * (b_s25 - b_s28);

        b_s30 = b_s23 - CRTM_15_5 * b_s29;
        b_t11 = CRTM_15_6 * (b_t1 + b_t2);

        b_t12 = CRTM_15_4 * b_s2;
        b_t13 = CRTM_15_3 * b_s0;
        b_s31 = b_t13 + b_t12;
        b_t14 = CRTM_15_4 * b_s7;
        b_t15 = CRTM_15_3 * b_s5;
        b_s32 = b_t14 - b_t15;
        b_t16 = CRTM_15_6 * (b_s31 + b_s32);
        b_s33 = b_s19 - b_t0;
        b_s34 = b_s32 - b_s31;
        b_s35 = (CRTM_15_4 * b_s4) + (CRTM_15_3 * b_s3);
        b_t17 = CRTM_15_5 * b_s34 - b_s35;
        b_s36 = b_t9 - b_t8;
        b_s37 = b_in10 + CRTM_15_11 * b_in4;
        b_s38 = b_s37 + b_s36;
        b_s39 = b_t6 - CRTM_15_11 * b_in11;
        b_t18 = CRTM_15_2 * b_s12 + b_in5;
        b_s40 = b_s39 - b_t18;
        b_s41 = b_s38 + b_s40;
        b_t19 = CRTM_15_6 * (b_s40 - b_s38);
        b_s42 = b_s33 - CRTM_15_5 * b_s41;

        out[out_strides[1]]  = b_s42 + b_t16;
        out[out_strides[2]]  = b_t17 + b_t19;
        out[out_strides[5]]  = b_s23 + b_s29;
        out[out_strides[6]]  = b_t3 + b_s24;
        out[out_strides[9]]  = b_s13 - CRTM_15_5 * b_s18;
        out[out_strides[10]] = CRTM_15_6 * (b_s15 - b_s17);
        out[out_strides[13]] = b_s30 - b_t11;
        out[out_strides[14]] = b_t10 + b_t4;
        out[out_strides[17]] = b_s33 + b_s41;
        out[out_strides[18]] = b_s35 + b_s34;
        out[out_strides[21]] = b_s42 - b_t16;
        out[out_strides[22]] = b_t17 - b_t19;
        out[out_strides[25]] = b_s30 + b_t11;
        out[out_strides[26]] = b_t10 - b_t4;
        out[out_strides[29]] = b_s18 + b_s13;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID r2hcf_rfft15avx256_fp64_bwd(VOID *in_real, VOID *in_imag,
                                        VOID *out_real, VOID *out_imag, INTP n,
                                        aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const DOUBLE CRTM_15_1 = 1.118033988749894848204586834365638117720309180;
    const DOUBLE CRTM_15_2 = 0.500000000000000000000000000000000000000000000;
    const DOUBLE CRTM_15_3 = 1.902113032590307144232878666758764286811397268;
    const DOUBLE CRTM_15_4 = 1.175570504584946258337411909278145537195304875;
    const DOUBLE CRTM_15_5 = 2.000000000000000000000000000000000000000000000;
    const DOUBLE CRTM_15_6 = 0.250000000000000000000000000000000000000000000;
    const DOUBLE CRTM_15_7 =
        1.01807392091025412936433958497993431950210362069154;
    const DOUBLE CRTM_15_8 =
        1.64727820709266368541488232323193203275710390365294;
    const DOUBLE CRTM_15_9 =
        0.96824583655185421224048777314959976915574786128504;
    const DOUBLE CRTM_15_10 = 1.732050807568877293527446341505872366942805254;
    const DOUBLE CRTM_15_11 = 0.433012701892219323381861585376468091735701313;
    const DOUBLE CRTM_15_12 = 0.587785252292473129168705954639072768597652438;
    const DOUBLE CRTM_15_13 = 0.951056516295153572116439333379382143405698634;
    const DOUBLE CRTM_15_14 = 0.559016994374947424102293417182819058860154590;

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
    INTP N = n >> 2;

    __m256d v_CRTM_15_1 = _mm256_broadcast_sd(&CRTM_15_1);
    __m256d v_CRTM_15_2 = _mm256_broadcast_sd(&CRTM_15_2);
    __m256d v_CRTM_15_3 = _mm256_broadcast_sd(&CRTM_15_3);
    __m256d v_CRTM_15_4 = _mm256_broadcast_sd(&CRTM_15_4);
    __m256d v_CRTM_15_5 = _mm256_broadcast_sd(&CRTM_15_5);
    __m256d v_CRTM_15_6 = _mm256_broadcast_sd(&CRTM_15_6);
    __m256d v_CRTM_15_7 = _mm256_broadcast_sd(&CRTM_15_7);
    __m256d v_CRTM_15_8 = _mm256_broadcast_sd(&CRTM_15_8);
    __m256d v_CRTM_15_9 = _mm256_broadcast_sd(&CRTM_15_9);
    __m256d v_CRTM_15_10 = _mm256_broadcast_sd(&CRTM_15_10);
    __m256d v_CRTM_15_11 = _mm256_broadcast_sd(&CRTM_15_11);
    __m256d v_CRTM_15_12 = _mm256_broadcast_sd(&CRTM_15_12);
    __m256d v_CRTM_15_13 = _mm256_broadcast_sd(&CRTM_15_13);
    __m256d v_CRTM_15_14 = _mm256_broadcast_sd(&CRTM_15_14);

    for (cnt = 0; cnt < N; cnt++)
    {
        /* Standard DFT */
        __m256d av_in0, av_in1, av_in2, av_in3;
        __m256d av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
                av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16,
                av_s17, av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24,
                av_s25, av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32,
                av_s33, av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40;
        __m256d av_t0, av_t1, av_t2, av_t4, av_t5;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
                v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21,
                v_out22, v_out23, v_out24, v_out25, v_out26, v_out27, v_out28,
                v_out29;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        curr_in = in + in_strides[0];
        LDR_256_D(curr_in, v_in_stride, av_in0);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_256_D(curr_in, v_in_stride, av_in1);
        // Input point 28: x(27)
        curr_in = in + in_strides[27];
        LDR_256_D(curr_in, v_in_stride, av_in2);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_256_D(curr_in, v_in_stride, av_in3);

        av_s0 = _mm256_add_pd(av_in1, av_in2);
        av_s1 = _mm256_sub_pd(av_in1, av_in2);
        av_s2 = _mm256_add_pd(av_in3, av_s0);
        av_s3 = _mm256_sub_pd(av_in3, _mm256_mul_pd(v_CRTM_15_2, av_s0));

        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDR_256_D(curr_in, v_in_stride, av_in1);
        // Input point 29: x(28)
        curr_in = in + in_strides[28];
        LDR_256_D(curr_in, v_in_stride, av_in2);
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDR_256_D(curr_in, v_in_stride, av_in3);

        av_s0 = _mm256_add_pd(av_in1, av_in2);
        av_s4 = _mm256_sub_pd(av_in1, av_in2);
        av_s5 = _mm256_sub_pd(av_s0, av_in3);
        av_s6 = _mm256_add_pd(av_in3, _mm256_mul_pd(v_CRTM_15_2, av_s0));

        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_256_D(curr_in, v_in_stride, av_in1);
        // Input point 16: x(15)
        curr_in = in + in_strides[15];
        LDR_256_D(curr_in, v_in_stride, av_in2);
        // Input point 24: x(23)
        curr_in = in + in_strides[23];
        LDR_256_D(curr_in, v_in_stride, av_in3);

        av_s0 = _mm256_add_pd(av_in1, av_in2);
        av_s7 = _mm256_sub_pd(av_in2, av_in1);
        av_s8 = _mm256_add_pd(av_in3, av_s0);
        av_s9 = _mm256_sub_pd(av_in3, _mm256_mul_pd(v_CRTM_15_2, av_s0));

        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_256_D(curr_in, v_in_stride, av_in1);
        // Input point 17: x(16)
        curr_in = in + in_strides[16];
        LDR_256_D(curr_in, v_in_stride, av_in2);
        // Input point 25: x(24)
        curr_in = in + in_strides[24];
        LDR_256_D(curr_in, v_in_stride, av_in3);

        av_s0 = _mm256_sub_pd(av_in1, av_in2);
        av_s10 = _mm256_add_pd(av_in2, av_in1);
        av_s11 = _mm256_add_pd(av_in3, av_s0);
        av_s12 = _mm256_sub_pd(av_in3, _mm256_mul_pd(v_CRTM_15_2, av_s0));

        // Input point 20: x(19)
        curr_in = in + in_strides[19];
        LDR_256_D(curr_in, v_in_stride, av_in1);
        // Input point 21: x(20)
        curr_in = in + in_strides[20];
        LDR_256_D(curr_in, v_in_stride, av_in2);

        av_s13 = _mm256_add_pd(av_in0, _mm256_mul_pd(v_CRTM_15_5, av_in1));
        av_s14 = _mm256_sub_pd(av_in0, av_in1);
        av_s15 = _mm256_add_pd(av_s8, av_s2);
        av_t0 = _mm256_mul_pd(v_CRTM_15_1, _mm256_sub_pd(av_s8, av_s2));
        av_s16 = _mm256_sub_pd(av_s13, _mm256_mul_pd(v_CRTM_15_2, av_s15));

        // Output point 1: X(0)
        v_out0 = _mm256_add_pd(av_s13, _mm256_mul_pd(v_CRTM_15_5, av_s15));
        STR_256_D(curr_out, v_out_stride, v_out0);

        av_s17 = _mm256_add_pd(_mm256_mul_pd(v_CRTM_15_3, av_s11),
                               _mm256_mul_pd(v_CRTM_15_4, av_s5));
        av_s18 = _mm256_add_pd(av_s16, av_t0);

        // Output point 7: X(6)
        v_out6 = _mm256_sub_pd(av_s18, av_s17);
        curr_out = out + out_strides[6];
        STR_256_D(curr_out, v_out_stride, v_out6);

        // Output point 25: X(24)
        v_out24 = _mm256_add_pd(av_s18, av_s17);
        curr_out = out + out_strides[24];
        STR_256_D(curr_out, v_out_stride, v_out24);

        av_s19 = _mm256_sub_pd(av_s16, av_t0);
        av_s20 = _mm256_sub_pd(_mm256_mul_pd(v_CRTM_15_4, av_s11),
                               _mm256_mul_pd(v_CRTM_15_3, av_s5));

        // Output point 13: X(12)
        v_out12 = _mm256_sub_pd(av_s19, av_s20);
        curr_out = out + out_strides[12];
        STR_256_D(curr_out, v_out_stride, v_out12);

        // Output point 19: X(18)
        v_out18 = _mm256_add_pd(av_s19, av_s20);
        curr_out = out + out_strides[18];
        STR_256_D(curr_out, v_out_stride, v_out18);

        av_t1 = _mm256_mul_pd(v_CRTM_15_1, _mm256_sub_pd(av_s9, av_s3));
        av_t2 = _mm256_mul_pd(v_CRTM_15_9, _mm256_add_pd(av_s4, av_s10));
        av_s21 = _mm256_add_pd(av_s9, av_s3);
        av_s22 = _mm256_sub_pd(av_s4, av_s10);
        av_s40 = _mm256_add_pd(_mm256_mul_pd(v_CRTM_15_5, av_s21), av_s14);
        av_t4 = _mm256_mul_pd(v_CRTM_15_10, _mm256_add_pd(av_s22, av_in2));

        // Output point 11: X(10)
        v_out10 = _mm256_add_pd(av_s40, av_t4);
        curr_out = out + out_strides[10];
        STR_256_D(curr_out, v_out_stride, v_out10);

        // Output point 21: X(20)
        v_out20 = _mm256_sub_pd(av_s40, av_t4);
        curr_out = out + out_strides[20];
        STR_256_D(curr_out, v_out_stride, v_out20);

        av_s23 = _mm256_sub_pd(av_s14, _mm256_mul_pd(v_CRTM_15_2, av_s21));
        av_t5 = _mm256_mul_pd(v_CRTM_15_10,
                _mm256_sub_pd( _mm256_mul_pd(v_CRTM_15_6, av_s22), av_in2));
        av_s24 = _mm256_add_pd(av_s23, av_t5);
        av_s25 = _mm256_sub_pd(av_s23, av_t5);
        av_s26 = _mm256_sub_pd(av_t2, av_t1);
        av_s27 = _mm256_add_pd(av_t1, av_t2);

        av_s28 = _mm256_sub_pd(av_s24, av_s27);
        av_s29 = _mm256_add_pd(_mm256_mul_pd(v_CRTM_15_3, av_s6),
                               _mm256_mul_pd(v_CRTM_15_4, av_s12));
        av_s30 = _mm256_sub_pd(_mm256_mul_pd(v_CRTM_15_8, av_s1),
                               _mm256_mul_pd(v_CRTM_15_7, av_s7));
        av_s31 = _mm256_sub_pd(av_s30, av_s29);
        av_s32 = _mm256_add_pd(av_s29, av_s30);

        // Output point 3: X(2)
        v_out2 = _mm256_add_pd(av_s28, av_s31);
        curr_out = out + out_strides[2];
        STR_256_D(curr_out, v_out_stride, v_out2);

        // Output point 9: X(8)
        v_out8 = _mm256_sub_pd(av_s28, av_s31);
        curr_out = out + out_strides[8];
        STR_256_D(curr_out, v_out_stride, v_out8);

        av_s33 = _mm256_add_pd(av_s24, av_s27);
        av_s34 = _mm256_add_pd(_mm256_mul_pd(v_CRTM_15_8, av_s7),
                               _mm256_mul_pd(v_CRTM_15_7, av_s1));
        av_s35 = _mm256_sub_pd(_mm256_mul_pd(v_CRTM_15_3, av_s12),
                               _mm256_mul_pd(v_CRTM_15_4, av_s6));
        av_s36 = _mm256_add_pd(av_s35, av_s34);

        // Output point 15: X(14)
        v_out14 = _mm256_add_pd(av_s33, av_s36);
        curr_out = out + out_strides[14];
        STR_256_D(curr_out, v_out_stride, v_out14);

        // Output point 27: X(26)
        v_out26 = _mm256_sub_pd(av_s33, av_s36);
        curr_out = out + out_strides[26];
        STR_256_D(curr_out, v_out_stride, v_out26);

        av_s37 = _mm256_sub_pd(av_s25, av_s26);
        av_s38 = _mm256_sub_pd(av_s35, av_s34);

        // Output point 5: X(4)
        v_out4 = _mm256_add_pd(av_s37, av_s38);
        curr_out = out + out_strides[4];
        STR_256_D(curr_out, v_out_stride, v_out4);

        // Output point 17: X(16)
        v_out16 = _mm256_sub_pd(av_s37, av_s38);
        curr_out = out + out_strides[16];
        STR_256_D(curr_out, v_out_stride, v_out16);

        av_s39 = _mm256_add_pd(av_s25, av_s26);

        // Output point 23: X(22)
        v_out22 = _mm256_sub_pd(av_s39, av_s32);
        curr_out = out + out_strides[22];
        STR_256_D(curr_out, v_out_stride, v_out22);

        // Output point 29: X(28)
        v_out28 = _mm256_add_pd(av_s39, av_s32);
        curr_out = out + out_strides[28];
        STR_256_D(curr_out, v_out_stride, v_out28);

        /* Shifted DFT */
        __m256d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
                bv_in8, bv_in9, bv_in10, bv_in11, bv_in12, bv_in13, bv_in14;
        __m256d bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
                bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17,
                bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24, bv_s25,
                bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32, bv_s33,
                bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40, bv_s41,
                bv_s42, bv_s43, bv_s44, bv_s45, bv_s46, bv_s47, bv_s48, bv_s49;
        __m256d bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
                bv_t10, bv_t11, bv_t12, bv_t13, bv_t14, bv_t15, bv_t16, bv_t17,
                bv_t18, bv_t19, bv_t20, bv_t21, bv_t22, bv_t23, bv_t24, bv_t25,
                bv_t26;

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
        // Input point 22: x(21) & Input point 23: x(22)
        curr_in = in + in_strides[21];
        LDRI_2x256_D(curr_in, v_in_stride, bv_in10, bv_in11);
        // Input point 26: x(25) & Input point 27: x(26)
        curr_in = in + in_strides[25];
        LDRI_2x256_D(curr_in, v_in_stride, bv_in12, bv_in13);
        // Input point 30: x(29)
        curr_in = in + in_strides[29];
        LDR_256_D(curr_in, v_in_stride, bv_in14);

        bv_s1 = _mm256_add_pd(bv_in0, bv_in10);
        bv_s2 = _mm256_sub_pd(bv_in0, bv_in10);
        bv_s3 = _mm256_add_pd(bv_in1, bv_in11);
        bv_s4 = _mm256_sub_pd(bv_in1, bv_in11);
        bv_s5 = _mm256_add_pd(bv_in6, bv_in12);
        bv_s6 = _mm256_sub_pd(bv_in6, bv_in12);
        bv_s7 = _mm256_add_pd(bv_in7, bv_in13);
        bv_s8 = _mm256_sub_pd(bv_in7, bv_in13);
        bv_s9 = _mm256_add_pd(bv_in2, bv_in8);
        bv_s10 = _mm256_sub_pd(bv_in2, bv_in8);

        bv_s11 = _mm256_sub_pd(bv_s1, bv_s5);
        bv_t1 = _mm256_mul_pd(v_CRTM_15_14, bv_s11);
        bv_t2 = _mm256_mul_pd(v_CRTM_15_1, bv_s10);
        bv_t3 = _mm256_mul_pd(v_CRTM_15_5, bv_t1);
        bv_s12 = _mm256_sub_pd(bv_t3, bv_t2);
        bv_t4 = _mm256_mul_pd(v_CRTM_15_2, bv_s9);
        bv_s13 = _mm256_sub_pd(bv_in14, bv_t4);
        bv_s14 = _mm256_add_pd(bv_s1, bv_s5);

        bv_t5 = _mm256_mul_pd(v_CRTM_15_6, bv_s14);
        bv_s15 = _mm256_sub_pd(bv_in4, bv_t5);
        bv_t6 = _mm256_mul_pd(v_CRTM_15_5, bv_s15);
        bv_s16 = _mm256_add_pd(bv_s13, bv_t6);
        bv_s17 = _mm256_sub_pd(bv_s12, bv_s16);
        bv_s18 = _mm256_add_pd(bv_s12, bv_s16);

        bv_t7 = _mm256_mul_pd(v_CRTM_15_13, bv_s8);
        bv_t8 = _mm256_mul_pd(v_CRTM_15_12, bv_s3);
        bv_t9 = _mm256_mul_pd(v_CRTM_15_4, bv_in9);
        bv_t10 = _mm256_mul_pd(v_CRTM_15_3, bv_in3);
        bv_s19 = _mm256_sub_pd(bv_t7, bv_t8);
        bv_t11 = _mm256_mul_pd(v_CRTM_15_5, bv_s19);
        bv_s20 = _mm256_sub_pd(bv_t9, bv_t10);
        bv_s21 = _mm256_add_pd(bv_t11, bv_s20);

        // Output pt 8: X(7)
        v_out7 = _mm256_add_pd(bv_s21, bv_s17);
        curr_out = out + out_strides[7];
        STR_256_D(curr_out, v_out_stride, v_out7);

        // Output pt 26: X(25)
        v_out25 = _mm256_sub_pd(bv_s21, bv_s17);
        curr_out = out + out_strides[25];
        STR_256_D(curr_out, v_out_stride, v_out25);

        bv_s22 = _mm256_add_pd(bv_s4, bv_s7);
        bv_s23 = _mm256_sub_pd(bv_in5, bv_s22);
        bv_t12 = _mm256_mul_pd(v_CRTM_15_10, bv_s23);
        bv_s24 = _mm256_add_pd(bv_s14, bv_in4);
        bv_t13 = _mm256_mul_pd(v_CRTM_15_5, bv_s9);
        bv_s25 = _mm256_add_pd(bv_in14, bv_t13);
        bv_s26 = _mm256_sub_pd(bv_s24, bv_s25);
        // Output pt 12: X(11)
        v_out11 = _mm256_add_pd(bv_t12, bv_s26);
        curr_out = out + out_strides[11];
        STR_256_D(curr_out, v_out_stride, v_out11);
        // Output pt 22: X(21)
        v_out21 = _mm256_sub_pd(bv_t12, bv_s26);
        curr_out = out + out_strides[21];
        STR_256_D(curr_out, v_out_stride, v_out21);
        bv_t19 = _mm256_mul_pd(v_CRTM_15_5, bv_s24);
        // Output pt 2: X(1)
        v_out1 = _mm256_add_pd(bv_t19, bv_s25);
        curr_out = out + out_strides[1];
        STR_256_D(curr_out, v_out_stride, v_out1);

        bv_t14 = _mm256_mul_pd(v_CRTM_15_4, bv_in3);
        bv_t15 = _mm256_mul_pd(v_CRTM_15_3, bv_in9);
        bv_s27 = _mm256_add_pd(bv_t14, bv_t15);

        bv_t16 = _mm256_mul_pd(v_CRTM_15_13, bv_s3);
        bv_t17 = _mm256_mul_pd(v_CRTM_15_12, bv_s8);
        bv_s28 = _mm256_add_pd(bv_t16, bv_t17);
        bv_t18 = _mm256_mul_pd(v_CRTM_15_5, bv_s28);
        bv_s29 = _mm256_sub_pd(bv_s27, bv_t18);
        // Output pt 14: X(13)
        v_out13 = _mm256_add_pd(bv_s29, bv_s18);
        curr_out = out + out_strides[13];
        STR_256_D(curr_out, v_out_stride, v_out13);
        // Output pt 20: X(19)
        v_out19 = _mm256_sub_pd(bv_s29, bv_s18);
        curr_out = out + out_strides[19];
        STR_256_D(curr_out, v_out_stride, v_out19);

        bv_t20 = _mm256_mul_pd(v_CRTM_15_11, bv_s22);
        bv_t21 = _mm256_mul_pd(v_CRTM_15_10, bv_in5);
        bv_s30 = _mm256_add_pd(bv_t20, bv_t21);
        bv_s31 = _mm256_sub_pd(bv_s4, bv_s7);
        bv_t22 = _mm256_mul_pd(v_CRTM_15_9, bv_s31);
        bv_s32 = _mm256_sub_pd(bv_t22, bv_s30);
        bv_s33 = _mm256_add_pd(bv_s30, bv_t22);
        bv_s34 = _mm256_add_pd(bv_t1, bv_t2);
        bv_s35 = _mm256_sub_pd(bv_s13, bv_s15);
        bv_s36 = _mm256_add_pd(bv_s34, bv_s35);
        bv_s37 = _mm256_sub_pd(bv_s34, bv_s35);
        bv_s38 = _mm256_add_pd(bv_s28, bv_s27);

        bv_t23 = _mm256_mul_pd(v_CRTM_15_8, bv_s2);
        bv_t24 = _mm256_mul_pd(v_CRTM_15_7, bv_s6);
        bv_s39 = _mm256_add_pd(bv_t23, bv_t24);
        bv_t25 = _mm256_mul_pd(v_CRTM_15_8, bv_s6);
        bv_t26 = _mm256_mul_pd(v_CRTM_15_7, bv_s2);
        bv_s40 = _mm256_sub_pd(bv_t25, bv_t26);
        bv_s41 = _mm256_sub_pd(bv_s19, bv_s20);
        bv_s42 = _mm256_sub_pd(bv_s37, bv_s38);
        bv_s43 = _mm256_add_pd(bv_s37, bv_s38);
        bv_s44 = _mm256_add_pd(bv_s39, bv_s32);
        bv_s45 = _mm256_sub_pd(bv_s39, bv_s32);
        bv_s46 = _mm256_sub_pd(bv_s40, bv_s33);
        bv_s47 = _mm256_add_pd(bv_s33, bv_s40);
        bv_s48 = _mm256_add_pd(bv_s36, bv_s41);
        bv_s49 = _mm256_sub_pd(bv_s36, bv_s41);

        // Output pt 4: X(3)
        v_out3 = _mm256_add_pd(bv_s42, bv_s44);
        curr_out = out + out_strides[3];
        STR_256_D(curr_out, v_out_stride, v_out3);
        // Output pt 24: X(23)
        v_out23 = _mm256_sub_pd(bv_s42, bv_s44);
        curr_out = out + out_strides[23];
        STR_256_D(curr_out, v_out_stride, v_out23);
        // Output pt 18: X(17)
        v_out17 = _mm256_add_pd(bv_s46, bv_s48);
        curr_out = out + out_strides[17];
        STR_256_D(curr_out, v_out_stride, v_out17);
        // Output pt 28: X(27)
        v_out27 = _mm256_sub_pd(bv_s46, bv_s48);
        curr_out = out + out_strides[27];
        STR_256_D(curr_out, v_out_stride, v_out27);
        // Output pt 6: X(5)
        v_out5 = _mm256_sub_pd(bv_s49, bv_s47);
        curr_out = out + out_strides[5];
        STR_256_D(curr_out, v_out_stride, v_out5);
        // Output pt 16: X(15)
        v_out15 = NEGATE_256_D(_mm256_add_pd(bv_s47, bv_s49));
        curr_out = out + out_strides[15];
        STR_256_D(curr_out, v_out_stride, v_out15);
        // Output pt 10: X(9)
        v_out9 = _mm256_sub_pd(bv_s45, bv_s43);
        curr_out = out + out_strides[9];
        STR_256_D(curr_out, v_out_stride, v_out9);
        // Output pt 30: X(29)
        v_out29 = NEGATE_256_D(_mm256_add_pd(bv_s43, bv_s45));
        curr_out = out + out_strides[29];
        STR_256_D(curr_out, v_out_stride, v_out29);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (n & 2)
    {
        /* Standard DFT */
        __m128d av_in0, av_in1, av_in2, av_in3;
        __m128d av_s0, av_s1, av_s2, av_s3, av_s4, av_s5, av_s6, av_s7, av_s8,
                av_s9, av_s10, av_s11, av_s12, av_s13, av_s14, av_s15, av_s16,
                av_s17, av_s18, av_s19, av_s20, av_s21, av_s22, av_s23, av_s24,
                av_s25, av_s26, av_s27, av_s28, av_s29, av_s30, av_s31, av_s32,
                av_s33, av_s34, av_s35, av_s36, av_s37, av_s38, av_s39, av_s40;
        __m128d av_t0, av_t1, av_t2, av_t4, av_t5;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14,
                v_out15, v_out16, v_out17, v_out18, v_out19, v_out20, v_out21,
                v_out22, v_out23, v_out24, v_out25, v_out26, v_out27, v_out28,
                v_out29;

        curr_in = in;
        curr_out = out;

        __m128d v128_CRTM_15_1 = _mm256_castpd256_pd128(v_CRTM_15_1);
        __m128d v128_CRTM_15_2 = _mm256_castpd256_pd128(v_CRTM_15_2);
        __m128d v128_CRTM_15_3 = _mm256_castpd256_pd128(v_CRTM_15_3);
        __m128d v128_CRTM_15_4 = _mm256_castpd256_pd128(v_CRTM_15_4);
        __m128d v128_CRTM_15_5 = _mm256_castpd256_pd128(v_CRTM_15_5);
        __m128d v128_CRTM_15_6 = _mm256_castpd256_pd128(v_CRTM_15_6);
        __m128d v128_CRTM_15_7 = _mm256_castpd256_pd128(v_CRTM_15_7);
        __m128d v128_CRTM_15_8 = _mm256_castpd256_pd128(v_CRTM_15_8);
        __m128d v128_CRTM_15_9 = _mm256_castpd256_pd128(v_CRTM_15_9);
        __m128d v128_CRTM_15_10 = _mm256_castpd256_pd128(v_CRTM_15_10);
        __m128d v128_CRTM_15_11 = _mm256_castpd256_pd128(v_CRTM_15_11);
        __m128d v128_CRTM_15_12 = _mm256_castpd256_pd128(v_CRTM_15_12);
        __m128d v128_CRTM_15_13 = _mm256_castpd256_pd128(v_CRTM_15_13);
        __m128d v128_CRTM_15_14 = _mm256_castpd256_pd128(v_CRTM_15_14);

        // Input point 1: x(0)
        curr_in = in + in_strides[0];
        LDR_128_D(curr_in, v_in_stride, av_in0);
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_128_D(curr_in, v_in_stride, av_in1);
        // Input point 28: x(27)
        curr_in = in + in_strides[27];
        LDR_128_D(curr_in, v_in_stride, av_in2);
        // Input point 12: x(11)
        curr_in = in + in_strides[11];
        LDR_128_D(curr_in, v_in_stride, av_in3);

        av_s0 = _mm_add_pd(av_in1, av_in2);
        av_s1 = _mm_sub_pd(av_in1, av_in2);
        av_s2 = _mm_add_pd(av_in3, av_s0);
        av_s3 = _mm_sub_pd(av_in3, _mm_mul_pd(v128_CRTM_15_2, av_s0));

        // Input point 9: x(8)
        curr_in = in + in_strides[8];
        LDR_128_D(curr_in, v_in_stride, av_in1);
        // Input point 29: x(28)
        curr_in = in + in_strides[28];
        LDR_128_D(curr_in, v_in_stride, av_in2);
        // Input point 13: x(12)
        curr_in = in + in_strides[12];
        LDR_128_D(curr_in, v_in_stride, av_in3);

        av_s0 = _mm_add_pd(av_in1, av_in2);
        av_s4 = _mm_sub_pd(av_in1, av_in2);
        av_s5 = _mm_sub_pd(av_s0, av_in3);
        av_s6 = _mm_add_pd(av_in3, _mm_mul_pd(v128_CRTM_15_2, av_s0));

        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_128_D(curr_in, v_in_stride, av_in1);
        // Input point 16: x(15)
        curr_in = in + in_strides[15];
        LDR_128_D(curr_in, v_in_stride, av_in2);
        // Input point 24: x(23)
        curr_in = in + in_strides[23];
        LDR_128_D(curr_in, v_in_stride, av_in3);

        av_s0 = _mm_add_pd(av_in1, av_in2);
        av_s7 = _mm_sub_pd(av_in2, av_in1);
        av_s8 = _mm_add_pd(av_in3, av_s0);
        av_s9 = _mm_sub_pd(av_in3, _mm_mul_pd(v128_CRTM_15_2, av_s0));

        // Input point 5: x(4)
        curr_in = in + in_strides[4];
        LDR_128_D(curr_in, v_in_stride, av_in1);
        // Input point 17: x(16)
        curr_in = in + in_strides[16];
        LDR_128_D(curr_in, v_in_stride, av_in2);
        // Input point 25: x(24)
        curr_in = in + in_strides[24];
        LDR_128_D(curr_in, v_in_stride, av_in3);

        av_s0 = _mm_sub_pd(av_in1, av_in2);
        av_s10 = _mm_add_pd(av_in2, av_in1);
        av_s11 = _mm_add_pd(av_in3, av_s0);
        av_s12 = _mm_sub_pd(av_in3, _mm_mul_pd(v128_CRTM_15_2, av_s0));

        // Input point 20: x(19)
        curr_in = in + in_strides[19];
        LDR_128_D(curr_in, v_in_stride, av_in1);
        // Input point 21: x(20)
        curr_in = in + in_strides[20];
        LDR_128_D(curr_in, v_in_stride, av_in2);

        av_s13 = _mm_add_pd(av_in0, _mm_mul_pd(v128_CRTM_15_5, av_in1));
        av_s14 = _mm_sub_pd(av_in0, av_in1);
        av_s15 = _mm_add_pd(av_s8, av_s2);
        av_t0 = _mm_mul_pd(v128_CRTM_15_1, _mm_sub_pd(av_s8, av_s2));
        av_s16 = _mm_sub_pd(av_s13, _mm_mul_pd(v128_CRTM_15_2, av_s15));

        // Output point 1: X(0)
        v_out0 = _mm_add_pd(av_s13, _mm_mul_pd(v128_CRTM_15_5, av_s15));
        STR_128_D(curr_out, v_out_stride, v_out0);

        av_s17 = _mm_add_pd(_mm_mul_pd(v128_CRTM_15_3, av_s11),
                            _mm_mul_pd(v128_CRTM_15_4, av_s5));
        av_s18 = _mm_add_pd(av_s16, av_t0);

        // Output point 7: X(6)
        v_out6 = _mm_sub_pd(av_s18, av_s17);
        curr_out = out + out_strides[6];
        STR_128_D(curr_out, v_out_stride, v_out6);

        // Output point 25: X(24)
        v_out24 = _mm_add_pd(av_s18, av_s17);
        curr_out = out + out_strides[24];
        STR_128_D(curr_out, v_out_stride, v_out24);

        av_s19 = _mm_sub_pd(av_s16, av_t0);
        av_s20 = _mm_sub_pd(_mm_mul_pd(v128_CRTM_15_4, av_s11),
                            _mm_mul_pd(v128_CRTM_15_3, av_s5));

        // Output point 13: X(12)
        v_out12 = _mm_sub_pd(av_s19, av_s20);
        curr_out = out + out_strides[12];
        STR_128_D(curr_out, v_out_stride, v_out12);

        // Output point 19: X(18)
        v_out18 = _mm_add_pd(av_s19, av_s20);
        curr_out = out + out_strides[18];
        STR_128_D(curr_out, v_out_stride, v_out18);

        av_t1 = _mm_mul_pd(v128_CRTM_15_1, _mm_sub_pd(av_s9, av_s3));
        av_t2 = _mm_mul_pd(v128_CRTM_15_9, _mm_add_pd(av_s4, av_s10));
        av_s21 = _mm_add_pd(av_s9, av_s3);
        av_s22 = _mm_sub_pd(av_s4, av_s10);
        av_s40 = _mm_add_pd(_mm_mul_pd(v128_CRTM_15_5, av_s21), av_s14);
        av_t4 = _mm_mul_pd(v128_CRTM_15_10, _mm_add_pd(av_s22, av_in2));

        // Output point 11: X(10)
        v_out10 = _mm_add_pd(av_s40, av_t4);
        curr_out = out + out_strides[10];
        STR_128_D(curr_out, v_out_stride, v_out10);

        // Output point 21: X(20)
        v_out20 = _mm_sub_pd(av_s40, av_t4);
        curr_out = out + out_strides[20];
        STR_128_D(curr_out, v_out_stride, v_out20);

        av_s23 = _mm_sub_pd(av_s14, _mm_mul_pd(v128_CRTM_15_2, av_s21));
        av_t5 = _mm_mul_pd(v128_CRTM_15_10,
                _mm_sub_pd( _mm_mul_pd(v128_CRTM_15_6, av_s22), av_in2));
        av_s24 = _mm_add_pd(av_s23, av_t5);
        av_s25 = _mm_sub_pd(av_s23, av_t5);
        av_s26 = _mm_sub_pd(av_t2, av_t1);
        av_s27 = _mm_add_pd(av_t1, av_t2);

        av_s28 = _mm_sub_pd(av_s24, av_s27);
        av_s29 = _mm_add_pd(_mm_mul_pd(v128_CRTM_15_3, av_s6),
                            _mm_mul_pd(v128_CRTM_15_4, av_s12));
        av_s30 = _mm_sub_pd(_mm_mul_pd(v128_CRTM_15_8, av_s1),
                            _mm_mul_pd(v128_CRTM_15_7, av_s7));
        av_s31 = _mm_sub_pd(av_s30, av_s29);
        av_s32 = _mm_add_pd(av_s29, av_s30);

        // Output point 3: X(2)
        v_out2 = _mm_add_pd(av_s28, av_s31);
        curr_out = out + out_strides[2];
        STR_128_D(curr_out, v_out_stride, v_out2);

        // Output point 9: X(8)
        v_out8 = _mm_sub_pd(av_s28, av_s31);
        curr_out = out + out_strides[8];
        STR_128_D(curr_out, v_out_stride, v_out8);

        av_s33 = _mm_add_pd(av_s24, av_s27);
        av_s34 = _mm_add_pd(_mm_mul_pd(v128_CRTM_15_8, av_s7),
                            _mm_mul_pd(v128_CRTM_15_7, av_s1));
        av_s35 = _mm_sub_pd(_mm_mul_pd(v128_CRTM_15_3, av_s12),
                            _mm_mul_pd(v128_CRTM_15_4, av_s6));
        av_s36 = _mm_add_pd(av_s35, av_s34);

        // Output point 15: X(14)
        v_out14 = _mm_add_pd(av_s33, av_s36);
        curr_out = out + out_strides[14];
        STR_128_D(curr_out, v_out_stride, v_out14);

        // Output point 27: X(26)
        v_out26 = _mm_sub_pd(av_s33, av_s36);
        curr_out = out + out_strides[26];
        STR_128_D(curr_out, v_out_stride, v_out26);

        av_s37 = _mm_sub_pd(av_s25, av_s26);
        av_s38 = _mm_sub_pd(av_s35, av_s34);

        // Output point 5: X(4)
        v_out4 = _mm_add_pd(av_s37, av_s38);
        curr_out = out + out_strides[4];
        STR_128_D(curr_out, v_out_stride, v_out4);

        // Output point 17: X(16)
        v_out16 = _mm_sub_pd(av_s37, av_s38);
        curr_out = out + out_strides[16];
        STR_128_D(curr_out, v_out_stride, v_out16);

        av_s39 = _mm_add_pd(av_s25, av_s26);

        // Output point 23: X(22)
        v_out22 = _mm_sub_pd(av_s39, av_s32);
        curr_out = out + out_strides[22];
        STR_128_D(curr_out, v_out_stride, v_out22);

        // Output point 29: X(28)
        v_out28 = _mm_add_pd(av_s39, av_s32);
        curr_out = out + out_strides[28];
        STR_128_D(curr_out, v_out_stride, v_out28);

        /* Shifted DFT */
        __m128d bv_in0, bv_in1, bv_in2, bv_in3, bv_in4, bv_in5, bv_in6, bv_in7,
                bv_in8, bv_in9, bv_in10, bv_in11, bv_in12, bv_in13, bv_in14;
        __m128d bv_s1, bv_s2, bv_s3, bv_s4, bv_s5, bv_s6, bv_s7, bv_s8, bv_s9,
                bv_s10, bv_s11, bv_s12, bv_s13, bv_s14, bv_s15, bv_s16, bv_s17,
                bv_s18, bv_s19, bv_s20, bv_s21, bv_s22, bv_s23, bv_s24, bv_s25,
                bv_s26, bv_s27, bv_s28, bv_s29, bv_s30, bv_s31, bv_s32, bv_s33,
                bv_s34, bv_s35, bv_s36, bv_s37, bv_s38, bv_s39, bv_s40, bv_s41,
                bv_s42, bv_s43, bv_s44, bv_s45, bv_s46, bv_s47, bv_s48, bv_s49;
        __m128d bv_t1, bv_t2, bv_t3, bv_t4, bv_t5, bv_t6, bv_t7, bv_t8, bv_t9,
                bv_t10, bv_t11, bv_t12, bv_t13, bv_t14, bv_t15, bv_t16, bv_t17,
                bv_t18, bv_t19, bv_t20, bv_t21, bv_t22, bv_t23, bv_t24, bv_t25,
                bv_t26;

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
        // Input point 22: x(21) & Input point 23: x(22)
        curr_in = in + in_strides[21];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in10, bv_in11);
        // Input point 26: x(25) & Input point 27: x(26)
        curr_in = in + in_strides[25];
        LDRI_2x128_D(curr_in, v_in_stride, bv_in12, bv_in13);
        // Input point 30: x(29)
        curr_in = in + in_strides[29];
        LDR_128_D(curr_in, v_in_stride, bv_in14);

        bv_s1 = _mm_add_pd(bv_in0, bv_in10);
        bv_s2 = _mm_sub_pd(bv_in0, bv_in10);
        bv_s3 = _mm_add_pd(bv_in1, bv_in11);
        bv_s4 = _mm_sub_pd(bv_in1, bv_in11);
        bv_s5 = _mm_add_pd(bv_in6, bv_in12);
        bv_s6 = _mm_sub_pd(bv_in6, bv_in12);
        bv_s7 = _mm_add_pd(bv_in7, bv_in13);
        bv_s8 = _mm_sub_pd(bv_in7, bv_in13);
        bv_s9 = _mm_add_pd(bv_in2, bv_in8);
        bv_s10 = _mm_sub_pd(bv_in2, bv_in8);

        bv_s11 = _mm_sub_pd(bv_s1, bv_s5);
        bv_t1 = _mm_mul_pd(v128_CRTM_15_14, bv_s11);
        bv_t2 = _mm_mul_pd(v128_CRTM_15_1, bv_s10);
        bv_t3 = _mm_mul_pd(v128_CRTM_15_5, bv_t1);
        bv_s12 = _mm_sub_pd(bv_t3, bv_t2);
        bv_t4 = _mm_mul_pd(v128_CRTM_15_2, bv_s9);
        bv_s13 = _mm_sub_pd(bv_in14, bv_t4);
        bv_s14 = _mm_add_pd(bv_s1, bv_s5);

        bv_t5 = _mm_mul_pd(v128_CRTM_15_6, bv_s14);
        bv_s15 = _mm_sub_pd(bv_in4, bv_t5);
        bv_t6 = _mm_mul_pd(v128_CRTM_15_5, bv_s15);
        bv_s16 = _mm_add_pd(bv_s13, bv_t6);
        bv_s17 = _mm_sub_pd(bv_s12, bv_s16);
        bv_s18 = _mm_add_pd(bv_s12, bv_s16);

        bv_t7 = _mm_mul_pd(v128_CRTM_15_13, bv_s8);
        bv_t8 = _mm_mul_pd(v128_CRTM_15_12, bv_s3);
        bv_t9 = _mm_mul_pd(v128_CRTM_15_4, bv_in9);
        bv_t10 = _mm_mul_pd(v128_CRTM_15_3, bv_in3);
        bv_s19 = _mm_sub_pd(bv_t7, bv_t8);
        bv_t11 = _mm_mul_pd(v128_CRTM_15_5, bv_s19);
        bv_s20 = _mm_sub_pd(bv_t9, bv_t10);
        bv_s21 = _mm_add_pd(bv_t11, bv_s20);

        // Output pt 8: X(7)
        v_out7 = _mm_add_pd(bv_s21, bv_s17);
        curr_out = out + out_strides[7];
        STR_128_D(curr_out, v_out_stride, v_out7);

        // Output pt 26: X(25)
        v_out25 = _mm_sub_pd(bv_s21, bv_s17);
        curr_out = out + out_strides[25];
        STR_128_D(curr_out, v_out_stride, v_out25);

        bv_s22 = _mm_add_pd(bv_s4, bv_s7);
        bv_s23 = _mm_sub_pd(bv_in5, bv_s22);
        bv_t12 = _mm_mul_pd(v128_CRTM_15_10, bv_s23);
        bv_s24 = _mm_add_pd(bv_s14, bv_in4);
        bv_t13 = _mm_mul_pd(v128_CRTM_15_5, bv_s9);
        bv_s25 = _mm_add_pd(bv_in14, bv_t13);
        bv_s26 = _mm_sub_pd(bv_s24, bv_s25);
        // Output pt 12: X(11)
        v_out11 = _mm_add_pd(bv_t12, bv_s26);
        curr_out = out + out_strides[11];
        STR_128_D(curr_out, v_out_stride, v_out11);
        // Output pt 22: X(21)
        v_out21 = _mm_sub_pd(bv_t12, bv_s26);
        curr_out = out + out_strides[21];
        STR_128_D(curr_out, v_out_stride, v_out21);
        bv_t19 = _mm_mul_pd(v128_CRTM_15_5, bv_s24);
        // Output pt 2: X(1)
        v_out1 = _mm_add_pd(bv_t19, bv_s25);
        curr_out = out + out_strides[1];
        STR_128_D(curr_out, v_out_stride, v_out1);

        bv_t14 = _mm_mul_pd(v128_CRTM_15_4, bv_in3);
        bv_t15 = _mm_mul_pd(v128_CRTM_15_3, bv_in9);
        bv_s27 = _mm_add_pd(bv_t14, bv_t15);

        bv_t16 = _mm_mul_pd(v128_CRTM_15_13, bv_s3);
        bv_t17 = _mm_mul_pd(v128_CRTM_15_12, bv_s8);
        bv_s28 = _mm_add_pd(bv_t16, bv_t17);
        bv_t18 = _mm_mul_pd(v128_CRTM_15_5, bv_s28);
        bv_s29 = _mm_sub_pd(bv_s27, bv_t18);
        // Output pt 14: X(13)
        v_out13 = _mm_add_pd(bv_s29, bv_s18);
        curr_out = out + out_strides[13];
        STR_128_D(curr_out, v_out_stride, v_out13);
        // Output pt 20: X(19)
        v_out19 = _mm_sub_pd(bv_s29, bv_s18);
        curr_out = out + out_strides[19];
        STR_128_D(curr_out, v_out_stride, v_out19);

        bv_t20 = _mm_mul_pd(v128_CRTM_15_11, bv_s22);
        bv_t21 = _mm_mul_pd(v128_CRTM_15_10, bv_in5);
        bv_s30 = _mm_add_pd(bv_t20, bv_t21);
        bv_s31 = _mm_sub_pd(bv_s4, bv_s7);
        bv_t22 = _mm_mul_pd(v128_CRTM_15_9, bv_s31);
        bv_s32 = _mm_sub_pd(bv_t22, bv_s30);
        bv_s33 = _mm_add_pd(bv_s30, bv_t22);
        bv_s34 = _mm_add_pd(bv_t1, bv_t2);
        bv_s35 = _mm_sub_pd(bv_s13, bv_s15);
        bv_s36 = _mm_add_pd(bv_s34, bv_s35);
        bv_s37 = _mm_sub_pd(bv_s34, bv_s35);
        bv_s38 = _mm_add_pd(bv_s28, bv_s27);

        bv_t23 = _mm_mul_pd(v128_CRTM_15_8, bv_s2);
        bv_t24 = _mm_mul_pd(v128_CRTM_15_7, bv_s6);
        bv_s39 = _mm_add_pd(bv_t23, bv_t24);
        bv_t25 = _mm_mul_pd(v128_CRTM_15_8, bv_s6);
        bv_t26 = _mm_mul_pd(v128_CRTM_15_7, bv_s2);
        bv_s40 = _mm_sub_pd(bv_t25, bv_t26);
        bv_s41 = _mm_sub_pd(bv_s19, bv_s20);
        bv_s42 = _mm_sub_pd(bv_s37, bv_s38);
        bv_s43 = _mm_add_pd(bv_s37, bv_s38);
        bv_s44 = _mm_add_pd(bv_s39, bv_s32);
        bv_s45 = _mm_sub_pd(bv_s39, bv_s32);
        bv_s46 = _mm_sub_pd(bv_s40, bv_s33);
        bv_s47 = _mm_add_pd(bv_s33, bv_s40);
        bv_s48 = _mm_add_pd(bv_s36, bv_s41);
        bv_s49 = _mm_sub_pd(bv_s36, bv_s41);

        // Output pt 4: X(3)
        v_out3 = _mm_add_pd(bv_s42, bv_s44);
        curr_out = out + out_strides[3];
        STR_128_D(curr_out, v_out_stride, v_out3);
        // Output pt 24: X(23)
        v_out23 = _mm_sub_pd(bv_s42, bv_s44);
        curr_out = out + out_strides[23];
        STR_128_D(curr_out, v_out_stride, v_out23);
        // Output pt 18: X(17)
        v_out17 = _mm_add_pd(bv_s46, bv_s48);
        curr_out = out + out_strides[17];
        STR_128_D(curr_out, v_out_stride, v_out17);
        // Output pt 28: X(27)
        v_out27 = _mm_sub_pd(bv_s46, bv_s48);
        curr_out = out + out_strides[27];
        STR_128_D(curr_out, v_out_stride, v_out27);
        // Output pt 6: X(5)
        v_out5 = _mm_sub_pd(bv_s49, bv_s47);
        curr_out = out + out_strides[5];
        STR_128_D(curr_out, v_out_stride, v_out5);
        // Output pt 16: X(15)
        v_out15 = NEGATE_128_D(_mm_add_pd(bv_s47, bv_s49));
        curr_out = out + out_strides[15];
        STR_128_D(curr_out, v_out_stride, v_out15);
        // Output pt 10: X(9)
        v_out9 = _mm_sub_pd(bv_s45, bv_s43);
        curr_out = out + out_strides[9];
        STR_128_D(curr_out, v_out_stride, v_out9);
        // Output pt 30: X(29)
        v_out29 = NEGATE_128_D(_mm_add_pd(bv_s43, bv_s45));
        curr_out = out + out_strides[29];
        STR_128_D(curr_out, v_out_stride, v_out29);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        /* Standard DFT */
        DOUBLE a_in0, a_in1, a_in2, a_in3, a_s0, a_s1, a_s2, a_s3, a_s4, a_s5,
               a_s6, a_s7, a_s8, a_s9, a_s10, a_s11, a_s12, a_s13, a_s14, a_s15,
               a_s16, a_s17, a_s18, a_s19, a_s20, a_s21, a_s22, a_s23, a_s24,
               a_s25, a_s26, a_s27, a_s28, a_s29, a_s30, a_s31, a_s32, a_s33,
               a_s34, a_s35, a_s36, a_s37, a_s38, a_s39, a_t0, a_t1, a_t2, a_t3,
               a_t4, a_t5;

        //  Input point 1: x(0)
        a_in0 = *in;
        //  Input point 8: x(7)
        a_in1 = in[in_strides[7]];
        //  Input point 28: x(27)
        a_in2 = in[in_strides[27]];
        //  Input point 12: x(11)
        a_in3 = in[in_strides[11]];

        a_s0 = a_in1 + a_in2;
        a_s1 = a_in1 - a_in2;
        a_s2 = a_in3 + a_s0;
        a_s3 = a_in3 - (CRTM_15_2 * a_s0);

        //  Input point 9: x(8)
        a_in1 = in[in_strides[8]];
        //  Input point 29: x(28)
        a_in2 = in[in_strides[28]];
        //  Input point 13: x(12)
        a_in3 = in[in_strides[12]];

        a_s0 = a_in1 + a_in2;
        a_s4 = a_in1 - a_in2;
        a_s5 = a_s0 - a_in3;
        a_s6 = a_in3 + (CRTM_15_2 * a_s0);

        //  Input point 4: x(3)
        a_in1 = in[in_strides[3]];
        //  Input point 16: x(15)
        a_in2 = in[in_strides[15]];
        //  Input point 24: x(23)
        a_in3 = in[in_strides[23]];

        a_s0 = a_in1 + a_in2;
        a_s7 = a_in2 - a_in1;
        a_s8 = a_in3 + a_s0;
        a_s9 = a_in3 - (CRTM_15_2 * a_s0);

        //  Input point 5: x(4)
        a_in1 = in[in_strides[4]];
        //  Input point 17: x(16)
        a_in2 = in[in_strides[16]];
        //  Input point 25: x(24)
        a_in3 = in[in_strides[24]];
        a_s0 = a_in1 - a_in2;
        a_s10 = a_in2 + a_in1;
        a_s11 = a_in3 + a_s0;
        a_s12 = a_in3 - (CRTM_15_2 * a_s0);

        //  Input point 20: x(19)
        a_in1 = in[in_strides[19]];
        //  Input point 21: x(20)
        a_in2 = in[in_strides[20]];

        a_s13 = a_in0 + CRTM_15_5 * a_in1;
        a_s14 = a_in0 - a_in1;
        a_s15 = a_s8 + a_s2;
        a_t0 = CRTM_15_1 * (a_s8 - a_s2);
        a_s16 = a_s13 - (CRTM_15_2 * a_s15);
        // Output point 1: X(0)
        *out = a_s13 + CRTM_15_5 * a_s15;

        a_s17 = (CRTM_15_3 * a_s11) + (CRTM_15_4 * a_s5);
        a_s18 = a_s16 + a_t0;

        // Output point 7: X(6)
        out[out_strides[6]] = a_s18 - a_s17;
        // Output point 25: X(24)
        out[out_strides[24]] = a_s18 + a_s17;

        a_s19 = a_s16 - a_t0;
        a_s20 = (CRTM_15_4 * a_s11) - (CRTM_15_3 * a_s5);

        // Output point 13: X(12)
        out[out_strides[12]] = a_s19 - a_s20;
        // Output point 19: X(18)
        out[out_strides[18]] = a_s19 + a_s20;

        a_t1 = CRTM_15_1 * (a_s9 - a_s3);
        a_t2 = CRTM_15_9 * (a_s10 + a_s4);
        a_s21 = (a_s9 + a_s3);
        a_s22 = a_s4 - a_s10;
        a_t3 = CRTM_15_5 * a_s21 + a_s14;
        a_t4 = CRTM_15_10 * (a_s22 + a_in2);

        // Output point 11: X(10)
        out[out_strides[10]] = a_t3 + a_t4;
        // Output point 21: X(20)
        out[out_strides[20]] = a_t3 - a_t4;

        a_s23 = a_s14 - (CRTM_15_2 * a_s21);
        a_t5 = CRTM_15_10 * ((CRTM_15_6 * a_s22) - a_in2);
        a_s24 = a_t5 + a_s23;
        a_s25 = a_s23 - a_t5;
        a_s26 = a_t2 - a_t1;
        a_s27 = a_t1 + a_t2;

        a_s28 = a_s24 - a_s27;
        a_s29 = (CRTM_15_4 * a_s12) + (CRTM_15_3 * a_s6);
        a_s30 = (CRTM_15_8 * a_s1) - (CRTM_15_7 * a_s7);
        a_s31 = a_s30 - a_s29;
        a_s32 = a_s29 + a_s30;

        // Output point 3: X(2)
        out[out_strides[2]] = a_s28 + a_s31;
        // Output point 9: X(8)
        out[out_strides[8]] = a_s28 - a_s31;

        a_s33 = a_s24 + a_s27;
        a_s34 = (CRTM_15_8 * a_s7) + (CRTM_15_7 * a_s1);
        a_s35 = (CRTM_15_3 * a_s12) - (CRTM_15_4 * a_s6);
        a_s36 = a_s35 + a_s34;

        // Output point 15: X(14)
        out[out_strides[14]] = a_s33 + a_s36;
        // Output point 27: X(26)
        out[out_strides[26]] = a_s33 - a_s36;

        a_s37 = a_s25 - a_s26;
        a_s38 = a_s35 - a_s34;
        // Output point 5: X(4)
        out[out_strides[4]] = a_s37 + a_s38;
        // Output point 17: X(16)
        out[out_strides[16]] = a_s37 - a_s38;

        a_s39 = a_s25 + a_s26;

        // Output point 23: X(22)
        out[out_strides[22]] = a_s39 - a_s32;
        // Output point 29: X(28)
        out[out_strides[28]] = a_s39 + a_s32;

        /* Shifted DFT */
        DOUBLE b_in0, b_in1, b_in2, b_in3, b_in4, b_in5, b_in6, b_in7, b_in8,
               b_in9, b_in10, b_in11, b_in12, b_in13, b_in14;
        DOUBLE b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7, b_s8, b_s9,
               b_s10, b_s11, b_s12, b_s13, b_s14, b_s15, b_s16, b_s17, b_s18,
               b_s19, b_s20, b_s21, b_s22, b_s23, b_s24, b_s25, b_s26, b_s27,
               b_s28, b_s29, b_s30, b_s31, b_s32, b_s33, b_s34, b_s35, b_s36,
               b_s37, b_s38, b_s39, b_s40, b_s41, b_s42, b_s43, b_s44, b_s45,
               b_s46, b_s47, b_s48, b_t0, b_t1, b_t2, b_t3, b_t4, b_t5, b_t6,
               b_t7, b_t8, b_t9, b_t10, b_t11, b_t12, b_t13, b_t14, b_t15,
               b_t16, b_t17, b_t18, b_t19, b_t20, b_t21, b_t22, b_t23, b_t24,
               b_t25;

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
        // Input point 22: x(21)
        b_in10 = in[in_strides[21]];
        // Input point 23: x(22)
        b_in11 = in[in_strides[22]];
        // Input point 26: x(25)
        b_in12 = in[in_strides[25]];
        // Input point 27: x(26)
        b_in13 = in[in_strides[26]];
        // Input point 30: x(29)
        b_in14 = in[in_strides[29]];

        b_s0 = b_in0 + b_in10;
        b_s1 = b_in0 - b_in10;
        b_s2 = b_in1 + b_in11;
        b_s3 = b_in1 - b_in11;
        b_s4 = b_in6 + b_in12;
        b_s5 = b_in6 - b_in12;
        b_s6 = b_in7 + b_in13;
        b_s7 = b_in7 - b_in13;
        b_s8 = b_in2 + b_in8;
        b_s9 = b_in2 - b_in8;

        b_s10 = b_s0 - b_s4;
        b_t0 = CRTM_15_14 * b_s10;
        b_t1 = CRTM_15_1 * b_s9;
        b_t2 = CRTM_15_5 * b_t0;
        b_s11 = b_t2 - b_t1;
        b_t3 = CRTM_15_2 * b_s8;
        b_s12 = b_in14 - b_t3;
        b_s13 = b_s0 + b_s4;

        b_t4 = CRTM_15_6 * b_s13;
        b_s14 = b_in4 - b_t4;
        b_t5 = CRTM_15_5 * b_s14;
        b_s15 = b_s12 + b_t5;
        b_s16 = b_s11 - b_s15;
        b_s17 = b_s11 + b_s15;

        b_t6 = CRTM_15_13 * b_s7;
        b_t7 = CRTM_15_12 * b_s2;
        b_t8 = CRTM_15_4 * b_in9;
        b_t9 = CRTM_15_3 * b_in3;
        b_s18 = b_t6 - b_t7;
        b_t10 = CRTM_15_5 * b_s18;
        b_s19 = b_t8 - b_t9;
        b_s20 = b_t10 + b_s19;
        // Output pt 8: X(7)
        out[out_strides[7]] = b_s20 + b_s16;
        // Output pt 26: X(25)
        out[out_strides[25]] = b_s20 - b_s16;

        b_s21 = b_s3 + b_s6;
        b_s22 = b_in5 - b_s21;
        b_t11 = CRTM_15_10 * b_s22;
        b_s23 = b_s13 + b_in4;
        b_t12 = CRTM_15_5 * b_s8;
        b_s24 = b_in14 + b_t12;
        b_s25 = b_s23 - b_s24;
        // Output pt 12: X(11)
        out[out_strides[11]] = b_t11 + b_s25;
        // Output pt 22: X(21)
        out[out_strides[21]] = b_t11 - b_s25;

        b_t13 = CRTM_15_5 * b_s23;
        // Output pt 2: X(1)
        out[out_strides[1]] = b_t13 + b_s24;

        b_t14 = CRTM_15_4 * b_in3;
        b_t15 = CRTM_15_3 * b_in9;
        b_s26 = b_t14 + b_t15;

        b_t16 = CRTM_15_13 * b_s2;
        b_t17 = CRTM_15_12 * b_s7;
        b_s27 = b_t16 + b_t17;
        b_t18 = CRTM_15_5 * b_s27;
        b_s28 = b_s26 - b_t18;
        // Output pt 14: X(13)
        out[out_strides[13]] = b_s28 + b_s17;
        // Output pt 20: X(19)
        out[out_strides[19]] = b_s28 - b_s17;

        b_t19 = CRTM_15_11 * b_s21;
        b_t20 = CRTM_15_10 * b_in5;
        b_s29 = b_t19 + b_t20;
        b_s30 = b_s3 - b_s6;
        b_t21 = CRTM_15_9 * b_s30;
        b_s31 = b_t21 - b_s29;
        b_s32 = b_s29 + b_t21;
        b_s33 = b_t0 + b_t1;
        b_s34 = b_s12 - b_s14;
        b_s35 = b_s33 + b_s34;
        b_s36 = b_s33 - b_s34;
        b_s37 = b_s27 + b_s26;

        b_t22 = CRTM_15_8 * b_s1;
        b_t23 = CRTM_15_7 * b_s5;
        b_s38 = b_t22 + b_t23;
        b_t24 = CRTM_15_8 * b_s5;
        b_t25 = CRTM_15_7 * b_s1;
        b_s39 = b_t24 - b_t25;
        b_s40 = b_s18 - b_s19;
        b_s41 = b_s36 - b_s37;
        b_s42 = b_s36 + b_s37;
        b_s43 = b_s38 + b_s31;
        // Output pt 4: X(3)
        out[out_strides[3]] = b_s41 + b_s43;
        // Output pt 24: X(23)
        out[out_strides[23]] = b_s41 - b_s43;

        b_s44 = b_s38 - b_s31;
        // Output pt 10: X(9)
        out[out_strides[9]] = b_s44 - b_s42;
        // Output pt 30: X(29)
        out[out_strides[29]] = -(b_s42 + b_s44);

        b_s45 = b_s39 - b_s32;
        b_s46 = b_s32 + b_s39;
        b_s47 = b_s35 + b_s40;
        // Output pt 18: X(17)
        out[out_strides[17]] = b_s45 + b_s47;
        // Output pt 28: X(27)
        out[out_strides[27]] = b_s45 - b_s47;

        b_s48 = b_s35 - b_s40;
        // Output pt 6: X(5)
        out[out_strides[5]] = b_s48 - b_s46;
        // Output pt 16: X(15)
        out[out_strides[15]] = -(b_s46 + b_s48);
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

kfft_ register_kernel_r2hcf_rfft15avx256(UINT8 precision, UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hcf_rfft15avx256_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft15avx256_fp64_fwd;
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
            return r2hcf_rfft15avx256_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft15avx256_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
