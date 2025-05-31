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
 *  @brief Radix-15 r2hc Real-FFT kernel with AVX-256 operations using x86 SIMD
 *  intrinsics
 *
 *  This file contains the DIT radix-15 real-to-halfcomplex implementations
 *  using AVX256 SIMD operations for single-precision and double-precision
 *  inputs.
 *
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/r2hc_simd_avx256.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                {{{0, 25, 64, 184, 110, 30},
                                                  {0, 27, 64, 240, 180, 30}},
                                                 {{0, 25, 64, 92,  14,  30},
                                                  {0, 27, 64, 120, 0,   30}}};

ops_cycles_t get_ops_cnt_r2hc_rfft15avx256(UINT8 precision, UINT8 direction)
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

static VOID r2hc_rfft15avx256_fp32_fwd(VOID *in_real, VOID *in_imag,
                                       VOID *out_real, VOID *out_imag, INTP n,
                                       aoclfftz_strides_t *strides, UINT8 flag)
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

    for (cnt = 0; cnt < N; cnt++)
    {
        __m256 v_in0, v_in1, v_in2, v_in3;
        __m256 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s15, v_s16, v_s17, v_s19, v_s20,
               v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27, v_s28, v_s29,
               v_s30, v_s31;
        __m256 v_t0, v_t1, v_t2, v_t3;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14;

        curr_in = in;
        curr_out = out;

        // Input point 1: X(0)
        curr_in = in + in_strides[0];
        LDR_256_S(curr_in, v_in_stride, v_in0);
        // Input point 3: X(2)
        curr_in = in + in_strides[2];
        LDR_256_S(curr_in, v_in_stride, v_in1);
        // Input point 8: X(7)
        curr_in = in + in_strides[7];
        LDR_256_S(curr_in, v_in_stride, v_in2);
        // Input point 13: X(12)
        curr_in = in + in_strides[12];
        LDR_256_S(curr_in, v_in_stride, v_in3);

        v_s0 = _mm256_add_ps(v_in1, v_in2);
        v_s1 = _mm256_sub_ps(v_in1, v_in2);
        v_s2 = _mm256_add_ps(v_in3, v_s0);
        v_s3 = _mm256_sub_ps(v_in3, _mm256_mul_ps(v_CRTM_15_5, v_s0));

        // Input point 9: X(8)
        curr_in = in + in_strides[8];
        LDR_256_S(curr_in, v_in_stride, v_in1);
        // Input point 14: X(13)
        curr_in = in + in_strides[13];
        LDR_256_S(curr_in, v_in_stride, v_in2);
        // Input point 4: X(3)
        curr_in = in + in_strides[3];
        LDR_256_S(curr_in, v_in_stride, v_in3);

        v_s0 = _mm256_add_ps(v_in1, v_in2);
        v_s4 = _mm256_sub_ps(v_in2, v_in1);
        v_s5 = _mm256_add_ps(v_in3, v_s0);
        v_s6 = _mm256_sub_ps(v_in3, _mm256_mul_ps(v_CRTM_15_5, v_s0));

        v_s23 = _mm256_add_ps(v_s6, v_s3);
        v_s24 = _mm256_sub_ps(v_s6, v_s3);
        v_s26 = _mm256_add_ps(v_s4, v_s1);
        v_s27 = _mm256_sub_ps(v_s4, v_s1);

        // Input point 2: X(1)
        curr_in = in + in_strides[1];
        LDR_256_S(curr_in, v_in_stride, v_in1);
        // Input point 12: X(11)
        curr_in = in + in_strides[11];
        LDR_256_S(curr_in, v_in_stride, v_in2);
        // Input point 7: X(6)
        curr_in = in + in_strides[6];
        LDR_256_S(curr_in, v_in_stride, v_in3);

        v_s0 = _mm256_add_ps(v_in1, v_in2);
        v_s1 = _mm256_sub_ps(v_in2, v_in1);
        v_s7 = _mm256_add_ps(v_in3, v_s0);
        v_s3 = _mm256_sub_ps(v_in3, _mm256_mul_ps(v_CRTM_15_5, v_s0));

        // Input point 5: X(4)
        curr_in = in + in_strides[4];
        LDR_256_S(curr_in, v_in_stride, v_in1);
        // Input point 15: X(14)
        curr_in = in + in_strides[14];
        LDR_256_S(curr_in, v_in_stride, v_in2);
        // Input point 10: X(9)
        curr_in = in + in_strides[9];
        LDR_256_S(curr_in, v_in_stride, v_in3);

        v_s0 = _mm256_add_ps(v_in1, v_in2);
        v_s4 = _mm256_sub_ps(v_in2, v_in1);
        v_s8 = _mm256_add_ps(v_in3, v_s0);
        v_s6 = _mm256_sub_ps(v_in3, _mm256_mul_ps(v_CRTM_15_5, v_s0));

        v_s28 = _mm256_add_ps(v_s6, v_s3);
        v_s29 = _mm256_sub_ps(v_s3, v_s6);
        v_s30 = _mm256_add_ps(v_s4, v_s1);
        v_s31 = _mm256_sub_ps(v_s1, v_s4);

        // Input point 6: X(5)
        curr_in = in + in_strides[5];
        LDR_256_S(curr_in, v_in_stride, v_in1);
        // Input point 11: X(10)
        curr_in = in + in_strides[10];
        LDR_256_S(curr_in, v_in_stride, v_in2);

        v_s0 = _mm256_add_ps(v_in1, v_in2);
        v_t0 = _mm256_mul_ps(v_CRTM_15_6, _mm256_sub_ps(v_in2, v_in1));
        v_s9 = _mm256_add_ps(v_in0, v_s0);
        v_s3 = _mm256_sub_ps(v_in0, _mm256_mul_ps(v_CRTM_15_5, v_s0));

        v_s11 = _mm256_add_ps(v_s8, v_s7);
        v_s12 = _mm256_add_ps(v_s2, v_s5);
        v_s19 = _mm256_sub_ps(v_s5, v_s2);
        v_s20 = _mm256_sub_ps(v_s8, v_s7);
        v_s13 = _mm256_add_ps(v_s11, v_s12);
        v_t1 = _mm256_mul_ps(v_CRTM_15_1, _mm256_sub_ps(v_s11, v_s12));
        v_s15 = _mm256_sub_ps(v_s9, _mm256_mul_ps(v_CRTM_15_2, v_s13));

        // Output point 1: X(0)
        v_out0 = _mm256_add_ps(v_s9, v_s13);
        STR_256_S(curr_out, v_out_stride, v_out0);

        // Output point 3: X(2)
        v_out5 = _mm256_add_ps(v_s15, v_t1);
        // Output point 4: X(3)
        v_out6 = _mm256_add_ps(_mm256_mul_ps(v_CRTM_15_3, v_s20),
                               _mm256_mul_ps(v_CRTM_15_4, v_s19));
        curr_out = out + out_strides[5];
        STRI_2x256_S(curr_out, v_out_stride, v_out5, v_out6);

        // Output point 6: X(5)
        v_out11 = _mm256_sub_ps(v_s15, v_t1);
        // Output point 7: X(6)
        v_out12 = _mm256_sub_ps(_mm256_mul_ps(v_CRTM_15_4, v_s20),
                                _mm256_mul_ps(v_CRTM_15_3, v_s19));
        curr_out = out + out_strides[11];
        STRI_2x256_S(curr_out, v_out_stride, v_out11, v_out12);

        v_t2 = _mm256_mul_ps(v_CRTM_15_1, _mm256_sub_ps(v_s28, v_s23));
        v_s13 = _mm256_add_ps(v_s28, v_s23);

        // Output point 5: X(4)
        v_out9 = _mm256_add_ps(v_s13, v_s3);

        v_t3 = _mm256_mul_ps(v_CRTM_15_9, _mm256_add_ps(v_s30, v_s27));
        v_s17 = _mm256_sub_ps(v_s30, v_s27);

        // Output point 6: X(5)
        v_out10 = _mm256_sub_ps(_mm256_mul_ps(v_CRTM_15_6, v_s17), v_t0);
        curr_out = out + out_strides[9];
        STRI_2x256_S(curr_out, v_out_stride, v_out9, v_out10);

        v_s15 = _mm256_sub_ps(v_s3, _mm256_mul_ps(v_CRTM_15_2, v_s13));
        v_s20 = _mm256_add_ps(v_t0, _mm256_mul_ps(v_CRTM_15_10, v_s17));
        v_s21 = _mm256_sub_ps(v_s15, v_t2);
        v_s25 = _mm256_sub_ps(_mm256_mul_ps(v_CRTM_15_8, v_s26),
                              _mm256_mul_ps(v_CRTM_15_7, v_s31));

        // Output point 1: X(0)
        v_out1 = _mm256_add_ps(v_s21, v_s25);

        // Output point 4: X(3)
        v_out7 = _mm256_sub_ps(v_s21, v_s25);

        v_s21 = _mm256_add_ps(v_s15, v_t2);
        v_s23 = _mm256_add_ps(_mm256_mul_ps(v_CRTM_15_8, v_s31),
                              _mm256_mul_ps(v_CRTM_15_7, v_s26));

        // Output point 7: X(6)
        v_out13 = _mm256_add_ps(v_s21, v_s23);

        v_s22 = _mm256_add_ps(v_s20, v_t3);
        v_s16 = _mm256_add_ps(_mm256_mul_ps(v_CRTM_15_4, v_s29),
                              _mm256_mul_ps(v_CRTM_15_3, v_s24));

        // Output point 2: X(1)
        v_out2 = _mm256_sub_ps(v_s22, v_s16);
        curr_out = out + out_strides[1];
        STRI_2x256_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 5: X(4)
        v_out8 = _mm256_add_ps(v_s22, v_s16);
        curr_out = out + out_strides[7];
        STRI_2x256_S(curr_out, v_out_stride, v_out7, v_out8);

        v_s22 = _mm256_sub_ps(v_s20, v_t3);
        v_s10 = _mm256_sub_ps(_mm256_mul_ps(v_CRTM_15_3, v_s29),
                              _mm256_mul_ps(v_CRTM_15_4, v_s24));

        // Output point 8: X(7)
        v_out14 = _mm256_add_ps(v_s22, v_s10);
        curr_out = out + out_strides[13];
        STRI_2x256_S(curr_out, v_out_stride, v_out13, v_out14);

        // Output point 2: X(1)
        v_out3 = _mm256_sub_ps(v_s21, v_s23);

        // Output point 3: X(2)
        v_out4 = _mm256_sub_ps(v_s10, v_s22);
        curr_out = out + out_strides[3];
        STRI_2x256_S(curr_out, v_out_stride, v_out3, v_out4);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (n & 4)
    {
        __m128 v_in0, v_in1, v_in2, v_in3;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s15, v_s16, v_s17, v_s19, v_s20,
               v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27, v_s28, v_s29,
               v_s30, v_s31;
        __m128 v_t0, v_t1, v_t2, v_t3;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14;

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

        // Input point 1: X(0)
        curr_in = in + in_strides[0];
        LDR_128_S(curr_in, v_in_stride, v_in0);
        // Input point 3: X(2)
        curr_in = in + in_strides[2];
        LDR_128_S(curr_in, v_in_stride, v_in1);
        // Input point 8: X(7)
        curr_in = in + in_strides[7];
        LDR_128_S(curr_in, v_in_stride, v_in2);
        // Input point 13: X(12)
        curr_in = in + in_strides[12];
        LDR_128_S(curr_in, v_in_stride, v_in3);

        v_s0 = _mm_add_ps(v_in1, v_in2);
        v_s1 = _mm_sub_ps(v_in1, v_in2);
        v_s2 = _mm_add_ps(v_in3, v_s0);
        v_s3 = _mm_sub_ps(v_in3, _mm_mul_ps(v128_CRTM_15_5, v_s0));

        // Input point 9: X(8)
        curr_in = in + in_strides[8];
        LDR_128_S(curr_in, v_in_stride, v_in1);
        // Input point 14: X(13)
        curr_in = in + in_strides[13];
        LDR_128_S(curr_in, v_in_stride, v_in2);
        // Input point 4: X(3)
        curr_in = in + in_strides[3];
        LDR_128_S(curr_in, v_in_stride, v_in3);

        v_s0 = _mm_add_ps(v_in1, v_in2);
        v_s4 = _mm_sub_ps(v_in2, v_in1);
        v_s5 = _mm_add_ps(v_in3, v_s0);
        v_s6 = _mm_sub_ps(v_in3, _mm_mul_ps(v128_CRTM_15_5, v_s0));

        v_s23 = _mm_add_ps(v_s6, v_s3);
        v_s24 = _mm_sub_ps(v_s6, v_s3);
        v_s26 = _mm_add_ps(v_s4, v_s1);
        v_s27 = _mm_sub_ps(v_s4, v_s1);

        // Input point 2: X(1)
        curr_in = in + in_strides[1];
        LDR_128_S(curr_in, v_in_stride, v_in1);
        // Input point 12: X(11)
        curr_in = in + in_strides[11];
        LDR_128_S(curr_in, v_in_stride, v_in2);
        // Input point 7: X(6)
        curr_in = in + in_strides[6];
        LDR_128_S(curr_in, v_in_stride, v_in3);

        v_s0 = _mm_add_ps(v_in1, v_in2);
        v_s1 = _mm_sub_ps(v_in2, v_in1);
        v_s7 = _mm_add_ps(v_in3, v_s0);
        v_s3 = _mm_sub_ps(v_in3, _mm_mul_ps(v128_CRTM_15_5, v_s0));

        // Input point 5: X(4)
        curr_in = in + in_strides[4];
        LDR_128_S(curr_in, v_in_stride, v_in1);
        // Input point 15: X(14)
        curr_in = in + in_strides[14];
        LDR_128_S(curr_in, v_in_stride, v_in2);
        // Input point 10: X(9)
        curr_in = in + in_strides[9];
        LDR_128_S(curr_in, v_in_stride, v_in3);

        v_s0 = _mm_add_ps(v_in1, v_in2);
        v_s4 = _mm_sub_ps(v_in2, v_in1);
        v_s8 = _mm_add_ps(v_in3, v_s0);
        v_s6 = _mm_sub_ps(v_in3, _mm_mul_ps(v128_CRTM_15_5, v_s0));

        v_s28 = _mm_add_ps(v_s6, v_s3);
        v_s29 = _mm_sub_ps(v_s3, v_s6);
        v_s30 = _mm_add_ps(v_s4, v_s1);
        v_s31 = _mm_sub_ps(v_s1, v_s4);

        // Input point 6: X(5)
        curr_in = in + in_strides[5];
        LDR_128_S(curr_in, v_in_stride, v_in1);
        // Input point 11: X(10)
        curr_in = in + in_strides[10];
        LDR_128_S(curr_in, v_in_stride, v_in2);

        v_s0 = _mm_add_ps(v_in1, v_in2);
        v_t0 = _mm_mul_ps(v128_CRTM_15_6, _mm_sub_ps(v_in2, v_in1));
        v_s9 = _mm_add_ps(v_in0, v_s0);
        v_s3 = _mm_sub_ps(v_in0, _mm_mul_ps(v128_CRTM_15_5, v_s0));

        v_s11 = _mm_add_ps(v_s8, v_s7);
        v_s12 = _mm_add_ps(v_s2, v_s5);
        v_s19 = _mm_sub_ps(v_s5, v_s2);
        v_s20 = _mm_sub_ps(v_s8, v_s7);
        v_s13 = _mm_add_ps(v_s11, v_s12);
        v_t1 = _mm_mul_ps(v128_CRTM_15_1, _mm_sub_ps(v_s11, v_s12));
        v_s15 = _mm_sub_ps(v_s9, _mm_mul_ps(v128_CRTM_15_2, v_s13));

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_s9, v_s13);
        STR_128_S(curr_out, v_out_stride, v_out0);

        // Output point 3: X(2)
        v_out5 = _mm_add_ps(v_s15, v_t1);
        // Output point 4: X(3)
        v_out6 = _mm_add_ps(_mm_mul_ps(v128_CRTM_15_3, v_s20),
                            _mm_mul_ps(v128_CRTM_15_4, v_s19));
        curr_out = out + out_strides[5];
        STRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);

        // Output point 6: X(5)
        v_out11 = _mm_sub_ps(v_s15, v_t1);
        // Output point 7: X(6)
        v_out12 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_4, v_s20),
                             _mm_mul_ps(v128_CRTM_15_3, v_s19));
        curr_out = out + out_strides[11];
        STRI_2x128_S(curr_out, v_out_stride, v_out11, v_out12);

        v_t2 = _mm_mul_ps(v128_CRTM_15_1, _mm_sub_ps(v_s28, v_s23));
        v_s13 = _mm_add_ps(v_s28, v_s23);

        // Output point 5: X(4)
        v_out9 = _mm_add_ps(v_s13, v_s3);

        v_t3 = _mm_mul_ps(v128_CRTM_15_9, _mm_add_ps(v_s30, v_s27));
        v_s17 = _mm_sub_ps(v_s30, v_s27);

        // Output point 6: X(5)
        v_out10 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_6, v_s17), v_t0);
        curr_out = out + out_strides[9];
        STRI_2x128_S(curr_out, v_out_stride, v_out9, v_out10);

        v_s15 = _mm_sub_ps(v_s3, _mm_mul_ps(v128_CRTM_15_2, v_s13));
        v_s20 = _mm_add_ps(v_t0, _mm_mul_ps(v128_CRTM_15_10, v_s17));
        v_s21 = _mm_sub_ps(v_s15, v_t2);
        v_s25 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_8, v_s26),
                           _mm_mul_ps(v128_CRTM_15_7, v_s31));

        // Output point 1: X(0)
        v_out1 = _mm_add_ps(v_s21, v_s25);

        // Output point 4: X(3)
        v_out7 = _mm_sub_ps(v_s21, v_s25);

        v_s21 = _mm_add_ps(v_s15, v_t2);
        v_s23 = _mm_add_ps(_mm_mul_ps(v128_CRTM_15_8, v_s31),
                           _mm_mul_ps(v128_CRTM_15_7, v_s26));

        // Output point 7: X(6)
        v_out13 = _mm_add_ps(v_s21, v_s23);

        v_s22 = _mm_add_ps(v_s20, v_t3);
        v_s16 = _mm_add_ps(_mm_mul_ps(v128_CRTM_15_4, v_s29),
                           _mm_mul_ps(v128_CRTM_15_3, v_s24));

        // Output point 2: X(1)
        v_out2 = _mm_sub_ps(v_s22, v_s16);
        curr_out = out + out_strides[1];
        STRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 5: X(4)
        v_out8 = _mm_add_ps(v_s22, v_s16);
        curr_out = out + out_strides[7];
        STRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);

        v_s22 = _mm_sub_ps(v_s20, v_t3);
        v_s10 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_3, v_s29),
                           _mm_mul_ps(v128_CRTM_15_4, v_s24));

        // Output point 8: X(7)
        v_out14 = _mm_add_ps(v_s22, v_s10);
        curr_out = out + out_strides[13];
        STRI_2x128_S(curr_out, v_out_stride, v_out13, v_out14);

        // Output point 2: X(1)
        v_out3 = _mm_sub_ps(v_s21, v_s23);

        // Output point 3: X(2)
        v_out4 = _mm_sub_ps(v_s10, v_s22);
        curr_out = out + out_strides[3];
        STRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (n & 2)
    {
        __m128 v_in0, v_in1, v_in2, v_in3;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s15, v_s16, v_s17, v_s19, v_s20,
               v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27, v_s28, v_s29,
               v_s30, v_s31;
        __m128 v_t0, v_t1, v_t2, v_t3;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14;

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

        // Input point 1: X(0)
        curr_in = in + in_strides[0];
        LDHR_128_S(curr_in, v_in_stride, v_in0);
        // Input point 3: X(2)
        curr_in = in + in_strides[2];
        LDHR_128_S(curr_in, v_in_stride, v_in1);
        // Input point 8: X(7)
        curr_in = in + in_strides[7];
        LDHR_128_S(curr_in, v_in_stride, v_in2);
        // Input point 13: X(12)
        curr_in = in + in_strides[12];
        LDHR_128_S(curr_in, v_in_stride, v_in3);

        v_s0 = _mm_add_ps(v_in1, v_in2);
        v_s1 = _mm_sub_ps(v_in1, v_in2);
        v_s2 = _mm_add_ps(v_in3, v_s0);
        v_s3 = _mm_sub_ps(v_in3, _mm_mul_ps(v128_CRTM_15_5, v_s0));

        // Input point 9: X(8)
        curr_in = in + in_strides[8];
        LDHR_128_S(curr_in, v_in_stride, v_in1);
        // Input point 14: X(13)
        curr_in = in + in_strides[13];
        LDHR_128_S(curr_in, v_in_stride, v_in2);
        // Input point 4: X(3)
        curr_in = in + in_strides[3];
        LDHR_128_S(curr_in, v_in_stride, v_in3);

        v_s0 = _mm_add_ps(v_in1, v_in2);
        v_s4 = _mm_sub_ps(v_in2, v_in1);
        v_s5 = _mm_add_ps(v_in3, v_s0);
        v_s6 = _mm_sub_ps(v_in3, _mm_mul_ps(v128_CRTM_15_5, v_s0));

        v_s23 = _mm_add_ps(v_s6, v_s3);
        v_s24 = _mm_sub_ps(v_s6, v_s3);
        v_s26 = _mm_add_ps(v_s4, v_s1);
        v_s27 = _mm_sub_ps(v_s4, v_s1);

        // Input point 2: X(1)
        curr_in = in + in_strides[1];
        LDHR_128_S(curr_in, v_in_stride, v_in1);
        // Input point 12: X(11)
        curr_in = in + in_strides[11];
        LDHR_128_S(curr_in, v_in_stride, v_in2);
        // Input point 7: X(6)
        curr_in = in + in_strides[6];
        LDHR_128_S(curr_in, v_in_stride, v_in3);

        v_s0 = _mm_add_ps(v_in1, v_in2);
        v_s1 = _mm_sub_ps(v_in2, v_in1);
        v_s7 = _mm_add_ps(v_in3, v_s0);
        v_s3 = _mm_sub_ps(v_in3, _mm_mul_ps(v128_CRTM_15_5, v_s0));

        // Input point 5: X(4)
        curr_in = in + in_strides[4];
        LDHR_128_S(curr_in, v_in_stride, v_in1);
        // Input point 15: X(14)
        curr_in = in + in_strides[14];
        LDHR_128_S(curr_in, v_in_stride, v_in2);
        // Input point 10: X(9)
        curr_in = in + in_strides[9];
        LDHR_128_S(curr_in, v_in_stride, v_in3);

        v_s0 = _mm_add_ps(v_in1, v_in2);
        v_s4 = _mm_sub_ps(v_in2, v_in1);
        v_s8 = _mm_add_ps(v_in3, v_s0);
        v_s6 = _mm_sub_ps(v_in3, _mm_mul_ps(v128_CRTM_15_5, v_s0));

        v_s28 = _mm_add_ps(v_s6, v_s3);
        v_s29 = _mm_sub_ps(v_s3, v_s6);
        v_s30 = _mm_add_ps(v_s4, v_s1);
        v_s31 = _mm_sub_ps(v_s1, v_s4);

        // Input point 6: X(5)
        curr_in = in + in_strides[5];
        LDHR_128_S(curr_in, v_in_stride, v_in1);
        // Input point 11: X(10)
        curr_in = in + in_strides[10];
        LDHR_128_S(curr_in, v_in_stride, v_in2);

        v_s0 = _mm_add_ps(v_in1, v_in2);
        v_t0 = _mm_mul_ps(v128_CRTM_15_6, _mm_sub_ps(v_in2, v_in1));
        v_s9 = _mm_add_ps(v_in0, v_s0);
        v_s3 = _mm_sub_ps(v_in0, _mm_mul_ps(v128_CRTM_15_5, v_s0));

        v_s11 = _mm_add_ps(v_s8, v_s7);
        v_s12 = _mm_add_ps(v_s2, v_s5);
        v_s19 = _mm_sub_ps(v_s5, v_s2);
        v_s20 = _mm_sub_ps(v_s8, v_s7);
        v_s13 = _mm_add_ps(v_s11, v_s12);
        v_t1 = _mm_mul_ps(v128_CRTM_15_1, _mm_sub_ps(v_s11, v_s12));
        v_s15 = _mm_sub_ps(v_s9, _mm_mul_ps(v128_CRTM_15_2, v_s13));

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_s9, v_s13);
        STHR_128_S(curr_out, v_out_stride, v_out0);

        // Output point 3: X(2)
        v_out5 = _mm_add_ps(v_s15, v_t1);
        // Output point 4: X(3)
        v_out6 = _mm_add_ps(_mm_mul_ps(v128_CRTM_15_3, v_s20),
                            _mm_mul_ps(v128_CRTM_15_4, v_s19));
        curr_out = out + out_strides[5];
        STHRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);

        // Output point 6: X(5)
        v_out11 = _mm_sub_ps(v_s15, v_t1);
        // Output point 7: X(6)
        v_out12 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_4, v_s20),
                             _mm_mul_ps(v128_CRTM_15_3, v_s19));
        curr_out = out + out_strides[11];
        STHRI_2x128_S(curr_out, v_out_stride, v_out11, v_out12);

        v_t2 = _mm_mul_ps(v128_CRTM_15_1, _mm_sub_ps(v_s28, v_s23));
        v_s13 = _mm_add_ps(v_s28, v_s23);

        // Output point 5: X(4)
        v_out9 = _mm_add_ps(v_s13, v_s3);

        v_t3 = _mm_mul_ps(v128_CRTM_15_9, _mm_add_ps(v_s30, v_s27));
        v_s17 = _mm_sub_ps(v_s30, v_s27);

        // Output point 6: X(5)
        v_out10 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_6, v_s17), v_t0);
        curr_out = out + out_strides[9];
        STHRI_2x128_S(curr_out, v_out_stride, v_out9, v_out10);

        v_s15 = _mm_sub_ps(v_s3, _mm_mul_ps(v128_CRTM_15_2, v_s13));
        v_s20 = _mm_add_ps(v_t0, _mm_mul_ps(v128_CRTM_15_10, v_s17));
        v_s21 = _mm_sub_ps(v_s15, v_t2);
        v_s25 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_8, v_s26),
                           _mm_mul_ps(v128_CRTM_15_7, v_s31));

        // Output point 1: X(0)
        v_out1 = _mm_add_ps(v_s21, v_s25);

        // Output point 4: X(3)
        v_out7 = _mm_sub_ps(v_s21, v_s25);

        v_s21 = _mm_add_ps(v_s15, v_t2);
        v_s23 = _mm_add_ps(_mm_mul_ps(v128_CRTM_15_8, v_s31),
                           _mm_mul_ps(v128_CRTM_15_7, v_s26));

        // Output point 7: X(6)
        v_out13 = _mm_add_ps(v_s21, v_s23);

        v_s22 = _mm_add_ps(v_s20, v_t3);
        v_s16 = _mm_add_ps(_mm_mul_ps(v128_CRTM_15_4, v_s29),
                           _mm_mul_ps(v128_CRTM_15_3, v_s24));

        // Output point 2: X(1)
        v_out2 = _mm_sub_ps(v_s22, v_s16);
        curr_out = out + out_strides[1];
        STHRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 5: X(4)
        v_out8 = _mm_add_ps(v_s22, v_s16);
        curr_out = out + out_strides[7];
        STHRI_2x128_S(curr_out, v_out_stride, v_out7, v_out8);

        v_s22 = _mm_sub_ps(v_s20, v_t3);
        v_s10 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_3, v_s29),
                           _mm_mul_ps(v128_CRTM_15_4, v_s24));

        // Output point 8: X(7)
        v_out14 = _mm_add_ps(v_s22, v_s10);
        curr_out = out + out_strides[13];
        STHRI_2x128_S(curr_out, v_out_stride, v_out13, v_out14);

        // Output point 2: X(1)
        v_out3 = _mm_sub_ps(v_s21, v_s23);

        // Output point 3: X(2)
        v_out4 = _mm_sub_ps(v_s10, v_s22);
        curr_out = out + out_strides[3];
        STHRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        FLOAT in0, in1, in2, in3, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10,
              s11, s12, s13, s15, s16, s17, s19, s20, s21, s22, s23, s24,
              s25, s26, s27, s28, s29, s30, s31, t0, t1, t2, t3;

        in0 = *in;
        in1 = in[in_strides[2]];
        in2 = in[in_strides[7]];
        in3 = in[in_strides[12]];

        s0 = in1 + in2;
        s1 = in1 - in2;
        s2 = in3 + s0;
        s3 = in3 - (CRTM_15_5 * s0);

        in1 = in[in_strides[8]];
        in2 = in[in_strides[13]];
        in3 = in[in_strides[3]];

        s0 = in1 + in2;
        s4 = in2 - in1;
        s5 = in3 + s0;
        s6 = in3 - (CRTM_15_5 * s0);

        s23 = s6 + s3;
        s24 = s6 - s3;
        s26 = s4 + s1;
        s27 = s4 - s1;

        in1 = in[in_strides[1]];
        in2 = in[in_strides[11]];
        in3 = in[in_strides[6]];

        s0 = in1 + in2;
        s1 = in2 - in1;
        s7 = in3 + s0;
        s3 = in3 - (CRTM_15_5 * s0);

        in1 = in[in_strides[4]];
        in2 = in[in_strides[14]];
        in3 = in[in_strides[9]];

        s0 = in1 + in2;
        s4 = in2 - in1;
        s8 = in3 + s0;
        s6 = in3 - (CRTM_15_5 * s0);

        s28 = s6 + s3;
        s29 = s3 - s6;
        s30 = s4 + s1;
        s31 = s1 - s4;

        in1 = in[in_strides[5]];
        in2 = in[in_strides[10]];

        s0 = in1 + in2;
        t0 = CRTM_15_6 * (in2 - in1);
        s9 = in0 + s0;
        s3 = in0 - (CRTM_15_5 * s0);

        s11 = s8 + s7;
        s12 = s2 + s5;
        s19 = s5 - s2;
        s20 = s8 - s7;
        s13 = s11 + s12;
        t1 = CRTM_15_1 * (s11 - s12);
        s15 = s9 - (CRTM_15_2 * s13);

        *out = s9 + s13;

        out[out_strides[5]] = s15 + t1;
        out[out_strides[6]] = (CRTM_15_3 * s20) + (CRTM_15_4 * s19);

        out[out_strides[11]] = s15 - t1;
        out[out_strides[12]] = (CRTM_15_4 * s20) - (CRTM_15_3 * s19);

        t2 = CRTM_15_1 * (s28 - s23);
        s13 = s28 + s23;

        out[out_strides[9]] = s13 + s3;

        t3 = CRTM_15_9 * (s30 + s27);
        s17 = s30 - s27;

        out[out_strides[10]] = CRTM_15_6 * s17 - t0;

        s15 = s3 - (CRTM_15_2 * s13);
        s20 = t0 + (CRTM_15_10 * s17);
        s21 = s15 - t2;
        s25 = (CRTM_15_8 * s26) - (CRTM_15_7 * s31);

        out[out_strides[1]] = s21 + s25;
        out[out_strides[7]] = s21 - s25;

        s21 = s15 + t2;
        s23 = (CRTM_15_8 * s31) + (CRTM_15_7 * s26);

        out[out_strides[13]] = s21 + s23;

        s22 = s20 + t3;
        s16 = (CRTM_15_4 * s29) + (CRTM_15_3 * s24);

        out[out_strides[2]] = s22 - s16;
        out[out_strides[8]] = s22 + s16;

        s22 = s20 - t3;
        s10 = (CRTM_15_3 * s29) - (CRTM_15_4 * s24);
        out[out_strides[14]] = s22 + s10;

        out[out_strides[3]] = s21 - s23;
        out[out_strides[4]] = s10 - s22;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID r2hc_rfft15avx256_fp32_bwd(VOID *in_real, VOID *in_imag,
                                       VOID *out_real, VOID *out_imag, INTP n,
                                       aoclfftz_strides_t *strides, UINT8 flag)
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
    // Below CRTMs are the product of the above CRTMs, Precomputed to save
    // multiplications on the fly.
    // CRTM_15_7 = CRTM_15_6 * CRTM_15_4
    const FLOAT CRTM_15_7 =
        1.01807392091025412936433958497993431950210362069154f;
    // CRTM_15_8 = CRTM_15_6 * CRTM_15_3
    const FLOAT CRTM_15_8 =
        1.64727820709266368541488232323193203275710390365294f;
    // CRTM_15_9 = CRTM_15_6 * CRTM_15_1
    const FLOAT CRTM_15_9 =
        0.96824583655185421224048777314959976915574786128504f;
    // CRTM_15_10 = CRTM_15_6 * CRTM_15_5
    const FLOAT CRTM_15_10 = 1.732050807568877293527446341505872366942805254f;

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

    for (cnt = 0; cnt < N; cnt++)
    {
        __m256 v_in0, v_in1, v_in2, v_in3;
        __m256 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
               v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27,
               v_s28, v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36,
               v_s37, v_s38, v_s39, v_s40;
        __m256 v_t0, v_t1, v_t2, v_t4, v_t5;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14;

        curr_in = in;
        curr_out = out;

        // Input point 1: X(0)
        curr_in = in + in_strides[0];
        LDR_256_S(curr_in, v_in_stride, v_in0);
        // Input point 4: X(3)
        curr_in = in + in_strides[3];
        LDR_256_S(curr_in, v_in_stride, v_in1);
        // Input point 14: X(13)
        curr_in = in + in_strides[13];
        LDR_256_S(curr_in, v_in_stride, v_in2);
        // Input point 6: X(5)
        curr_in = in + in_strides[5];
        LDR_256_S(curr_in, v_in_stride, v_in3);

        v_s0 = _mm256_add_ps(v_in1, v_in2);
        v_s1 = _mm256_sub_ps(v_in1, v_in2);
        v_s2 = _mm256_add_ps(v_in3, v_s0);
        v_s3 = _mm256_sub_ps(v_in3, _mm256_mul_ps(v_CRTM_15_2, v_s0));

        // Input point 5: X(4)
        curr_in = in + in_strides[4];
        LDR_256_S(curr_in, v_in_stride, v_in1);
        // Input point 15: X(14)
        curr_in = in + in_strides[14];
        LDR_256_S(curr_in, v_in_stride, v_in2);
        // Input point 7: X(6)
        curr_in = in + in_strides[6];
        LDR_256_S(curr_in, v_in_stride, v_in3);

        v_s0 = _mm256_add_ps(v_in1, v_in2);
        v_s4 = _mm256_sub_ps(v_in1, v_in2);
        v_s5 = _mm256_sub_ps(v_s0, v_in3);
        v_s6 = _mm256_add_ps(v_in3, _mm256_mul_ps(v_CRTM_15_2, v_s0));

        // Input point 2: X(1)
        curr_in = in + in_strides[1];
        LDR_256_S(curr_in, v_in_stride, v_in1);
        // Input point 8: X(7)
        curr_in = in + in_strides[7];
        LDR_256_S(curr_in, v_in_stride, v_in2);
        // Input point 12: X(11)
        curr_in = in + in_strides[11];
        LDR_256_S(curr_in, v_in_stride, v_in3);

        v_s0 = _mm256_add_ps(v_in1, v_in2);
        v_s7 = _mm256_sub_ps(v_in2, v_in1);
        v_s8 = _mm256_add_ps(v_in3, v_s0);
        v_s9 = _mm256_sub_ps(v_in3, _mm256_mul_ps(v_CRTM_15_2, v_s0));

        // Input point 3: X(2)
        curr_in = in + in_strides[2];
        LDR_256_S(curr_in, v_in_stride, v_in1);
        // Input point 9: X(8)
        curr_in = in + in_strides[8];
        LDR_256_S(curr_in, v_in_stride, v_in2);
        // Input point 13: X(12)
        curr_in = in + in_strides[12];
        LDR_256_S(curr_in, v_in_stride, v_in3);

        v_s0 = _mm256_sub_ps(v_in1, v_in2);
        v_s10 = _mm256_add_ps(v_in2, v_in1);
        v_s11 = _mm256_add_ps(v_in3, v_s0);
        v_s12 = _mm256_sub_ps(v_in3, _mm256_mul_ps(v_CRTM_15_2, v_s0));

        // Input point 10: X(9)
        curr_in = in + in_strides[9];
        LDR_256_S(curr_in, v_in_stride, v_in1);
        // Input point 11: X(10)
        curr_in = in + in_strides[10];
        LDR_256_S(curr_in, v_in_stride, v_in2);

        v_s13 = _mm256_add_ps(v_in0, _mm256_mul_ps(v_CRTM_15_5, v_in1));
        v_s14 = _mm256_sub_ps(v_in0, v_in1);
        v_s15 = _mm256_add_ps(v_s8, v_s2);
        v_t0 = _mm256_mul_ps(v_CRTM_15_1, _mm256_sub_ps(v_s8, v_s2));
        v_s16 = _mm256_sub_ps(v_s13, _mm256_mul_ps(v_CRTM_15_2, v_s15));

        // Output point 1: X(0)
        v_out0 = _mm256_add_ps(v_s13, _mm256_mul_ps(v_CRTM_15_5, v_s15));
        STR_256_S(curr_out, v_out_stride, v_out0);

        v_s17 = _mm256_add_ps(_mm256_mul_ps(v_CRTM_15_3, v_s11),
                              _mm256_mul_ps(v_CRTM_15_4, v_s5));
        v_s18 = _mm256_add_ps(v_s16, v_t0);

        // Output point 2: X(1)
        v_out3 = _mm256_sub_ps(v_s18, v_s17);
        curr_out = out + out_strides[3];
        STR_256_S(curr_out, v_out_stride, v_out3);

        // Output point 7: X(6)
        v_out12 = _mm256_add_ps(v_s18, v_s17);
        curr_out = out + out_strides[12];
        STR_256_S(curr_out, v_out_stride, v_out12);

        v_s19 = _mm256_sub_ps(v_s16, v_t0);
        v_s20 = _mm256_sub_ps(_mm256_mul_ps(v_CRTM_15_4, v_s11),
                              _mm256_mul_ps(v_CRTM_15_3, v_s5));

        // Output point 4: X(3)
        v_out6 = _mm256_sub_ps(v_s19, v_s20);
        curr_out = out + out_strides[6];
        STR_256_S(curr_out, v_out_stride, v_out6);

        // Output point 5: X(4)
        v_out9 = _mm256_add_ps(v_s19, v_s20);
        curr_out = out + out_strides[9];
        STR_256_S(curr_out, v_out_stride, v_out9);

        v_t1 = _mm256_mul_ps(v_CRTM_15_1, _mm256_sub_ps(v_s9, v_s3));
        v_t2 = _mm256_mul_ps(v_CRTM_15_9, _mm256_add_ps(v_s4, v_s10));
        v_s21 = _mm256_add_ps(v_s9, v_s3);
        v_s22 = _mm256_sub_ps(v_s4, v_s10);
        v_s40 = _mm256_add_ps(_mm256_mul_ps(v_CRTM_15_5, v_s21), v_s14);
        v_t4 = _mm256_mul_ps(v_CRTM_15_10, _mm256_add_ps(v_s22, v_in2));

        // Output point 3: X(2)
        v_out5 = _mm256_add_ps(v_s40, v_t4);
        curr_out = out + out_strides[5];
        STR_256_S(curr_out, v_out_stride, v_out5);

        // Output point 6: X(5)
        v_out10 = _mm256_sub_ps(v_s40, v_t4);
        curr_out = out + out_strides[10];
        STR_256_S(curr_out, v_out_stride, v_out10);

        v_s23 = _mm256_sub_ps(v_s14, _mm256_mul_ps(v_CRTM_15_2, v_s21));
        v_t5 = _mm256_mul_ps(v_CRTM_15_10,
                _mm256_sub_ps( _mm256_mul_ps(v_CRTM_15_6, v_s22), v_in2));
        v_s24 = _mm256_add_ps(v_s23, v_t5);
        v_s25 = _mm256_sub_ps(v_s23, v_t5);
        v_s26 = _mm256_sub_ps(v_t2, v_t1);
        v_s27 = _mm256_add_ps(v_t1, v_t2);

        v_s28 = _mm256_sub_ps(v_s24, v_s27);
        v_s29 = _mm256_add_ps(_mm256_mul_ps(v_CRTM_15_3, v_s6),
                              _mm256_mul_ps(v_CRTM_15_4, v_s12));
        v_s30 = _mm256_sub_ps(_mm256_mul_ps(v_CRTM_15_8, v_s1),
                              _mm256_mul_ps(v_CRTM_15_7, v_s7));
        v_s31 = _mm256_sub_ps(v_s30, v_s29);
        v_s32 = _mm256_add_ps(v_s29, v_s30);

        // Output point 1: X(0)
        v_out1 = _mm256_add_ps(v_s28, v_s31);
        curr_out = out + out_strides[1];
        STR_256_S(curr_out, v_out_stride, v_out1);

        // Output point 3: X(2)
        v_out4 = _mm256_sub_ps(v_s28, v_s31);
        curr_out = out + out_strides[4];
        STR_256_S(curr_out, v_out_stride, v_out4);

        v_s33 = _mm256_add_ps(v_s24, v_s27);
        v_s34 = _mm256_add_ps(_mm256_mul_ps(v_CRTM_15_8, v_s7),
                              _mm256_mul_ps(v_CRTM_15_7, v_s1));
        v_s35 = _mm256_sub_ps(_mm256_mul_ps(v_CRTM_15_3, v_s12),
                              _mm256_mul_ps(v_CRTM_15_4, v_s6));
        v_s36 = _mm256_add_ps(v_s35, v_s34);

        // Output point 4: X(3)
        v_out7 = _mm256_add_ps(v_s33, v_s36);
        curr_out = out + out_strides[7];
        STR_256_S(curr_out, v_out_stride, v_out7);

        // Output point 7: X(6)
        v_out13 = _mm256_sub_ps(v_s33, v_s36);
        curr_out = out + out_strides[13];
        STR_256_S(curr_out, v_out_stride, v_out13);

        v_s37 = _mm256_sub_ps(v_s25, v_s26);
        v_s38 = _mm256_sub_ps(v_s35, v_s34);

        // Output point 2: X(1)
        v_out2 = _mm256_add_ps(v_s37, v_s38);
        curr_out = out + out_strides[2];
        STR_256_S(curr_out, v_out_stride, v_out2);

        // Output point 5: X(4)
        v_out8 = _mm256_sub_ps(v_s37, v_s38);
        curr_out = out + out_strides[8];
        STR_256_S(curr_out, v_out_stride, v_out8);

        v_s39 = _mm256_add_ps(v_s25, v_s26);

        // Output point 6: X(5)
        v_out11 = _mm256_sub_ps(v_s39, v_s32);
        curr_out = out + out_strides[11];
        STR_256_S(curr_out, v_out_stride, v_out11);

        // Output point 8: X(7)
        v_out14 = _mm256_add_ps(v_s39, v_s32);
        curr_out = out + out_strides[14];
        STR_256_S(curr_out, v_out_stride, v_out14);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (n & 4)
    {
        __m128 v_in0, v_in1, v_in2, v_in3;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
               v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27,
               v_s28, v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36,
               v_s37, v_s38, v_s39, v_s40;
        __m128 v_t0, v_t1, v_t2, v_t4, v_t5;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14;

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

        // Input point 1: X(0)
        curr_in = in + in_strides[0];
        LDR_128_S(curr_in, v_in_stride, v_in0);
        // Input point 4: X(3)
        curr_in = in + in_strides[3];
        LDR_128_S(curr_in, v_in_stride, v_in1);
        // Input point 14: X(13)
        curr_in = in + in_strides[13];
        LDR_128_S(curr_in, v_in_stride, v_in2);
        // Input point 6: X(5)
        curr_in = in + in_strides[5];
        LDR_128_S(curr_in, v_in_stride, v_in3);

        v_s0 = _mm_add_ps(v_in1, v_in2);
        v_s1 = _mm_sub_ps(v_in1, v_in2);
        v_s2 = _mm_add_ps(v_in3, v_s0);
        v_s3 = _mm_sub_ps(v_in3, _mm_mul_ps(v128_CRTM_15_2, v_s0));

        // Input point 5: X(4)
        curr_in = in + in_strides[4];
        LDR_128_S(curr_in, v_in_stride, v_in1);
        // Input point 15: X(14)
        curr_in = in + in_strides[14];
        LDR_128_S(curr_in, v_in_stride, v_in2);
        // Input point 7: X(6)
        curr_in = in + in_strides[6];
        LDR_128_S(curr_in, v_in_stride, v_in3);

        v_s0 = _mm_add_ps(v_in1, v_in2);
        v_s4 = _mm_sub_ps(v_in1, v_in2);
        v_s5 = _mm_sub_ps(v_s0, v_in3);
        v_s6 = _mm_add_ps(v_in3, _mm_mul_ps(v128_CRTM_15_2, v_s0));

        // Input point 2: X(1)
        curr_in = in + in_strides[1];
        LDR_128_S(curr_in, v_in_stride, v_in1);
        // Input point 8: X(7)
        curr_in = in + in_strides[7];
        LDR_128_S(curr_in, v_in_stride, v_in2);
        // Input point 12: X(11)
        curr_in = in + in_strides[11];
        LDR_128_S(curr_in, v_in_stride, v_in3);

        v_s0 = _mm_add_ps(v_in1, v_in2);
        v_s7 = _mm_sub_ps(v_in2, v_in1);
        v_s8 = _mm_add_ps(v_in3, v_s0);
        v_s9 = _mm_sub_ps(v_in3, _mm_mul_ps(v128_CRTM_15_2, v_s0));

        // Input point 3: X(2)
        curr_in = in + in_strides[2];
        LDR_128_S(curr_in, v_in_stride, v_in1);
        // Input point 9: X(8)
        curr_in = in + in_strides[8];
        LDR_128_S(curr_in, v_in_stride, v_in2);
        // Input point 13: X(12)
        curr_in = in + in_strides[12];
        LDR_128_S(curr_in, v_in_stride, v_in3);

        v_s0 = _mm_sub_ps(v_in1, v_in2);
        v_s10 = _mm_add_ps(v_in2, v_in1);
        v_s11 = _mm_add_ps(v_in3, v_s0);
        v_s12 = _mm_sub_ps(v_in3, _mm_mul_ps(v128_CRTM_15_2, v_s0));

        // Input point 10: X(9)
        curr_in = in + in_strides[9];
        LDR_128_S(curr_in, v_in_stride, v_in1);
        // Input point 11: X(10)
        curr_in = in + in_strides[10];
        LDR_128_S(curr_in, v_in_stride, v_in2);

        v_s13 = _mm_add_ps(v_in0, _mm_mul_ps(v128_CRTM_15_5, v_in1));
        v_s14 = _mm_sub_ps(v_in0, v_in1);
        v_s15 = _mm_add_ps(v_s8, v_s2);
        v_t0 = _mm_mul_ps(v128_CRTM_15_1, _mm_sub_ps(v_s8, v_s2));
        v_s16 = _mm_sub_ps(v_s13, _mm_mul_ps(v128_CRTM_15_2, v_s15));

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_s13, _mm_mul_ps(v128_CRTM_15_5, v_s15));
        STR_128_S(curr_out, v_out_stride, v_out0);

        v_s17 = _mm_add_ps(_mm_mul_ps(v128_CRTM_15_3, v_s11),
                           _mm_mul_ps(v128_CRTM_15_4, v_s5));
        v_s18 = _mm_add_ps(v_s16, v_t0);

        // Output point 2: X(1)
        v_out3 = _mm_sub_ps(v_s18, v_s17);
        curr_out = out + out_strides[3];
        STR_128_S(curr_out, v_out_stride, v_out3);

        // Output point 7: X(6)
        v_out12 = _mm_add_ps(v_s18, v_s17);
        curr_out = out + out_strides[12];
        STR_128_S(curr_out, v_out_stride, v_out12);

        v_s19 = _mm_sub_ps(v_s16, v_t0);
        v_s20 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_4, v_s11),
                           _mm_mul_ps(v128_CRTM_15_3, v_s5));

        // Output point 4: X(3)
        v_out6 = _mm_sub_ps(v_s19, v_s20);
        curr_out = out + out_strides[6];
        STR_128_S(curr_out, v_out_stride, v_out6);

        // Output point 5: X(4)
        v_out9 = _mm_add_ps(v_s19, v_s20);
        curr_out = out + out_strides[9];
        STR_128_S(curr_out, v_out_stride, v_out9);

        v_t1 = _mm_mul_ps(v128_CRTM_15_1, _mm_sub_ps(v_s9, v_s3));
        v_t2 = _mm_mul_ps(v128_CRTM_15_9, _mm_add_ps(v_s4, v_s10));
        v_s21 = _mm_add_ps(v_s9, v_s3);
        v_s22 = _mm_sub_ps(v_s4, v_s10);
        v_s40 = _mm_add_ps(_mm_mul_ps(v128_CRTM_15_5, v_s21), v_s14);
        v_t4 = _mm_mul_ps(v128_CRTM_15_10, _mm_add_ps(v_s22, v_in2));

        // Output point 3: X(2)
        v_out5 = _mm_add_ps(v_s40, v_t4);
        curr_out = out + out_strides[5];
        STR_128_S(curr_out, v_out_stride, v_out5);

        // Output point 6: X(5)
        v_out10 = _mm_sub_ps(v_s40, v_t4);
        curr_out = out + out_strides[10];
        STR_128_S(curr_out, v_out_stride, v_out10);

        v_s23 = _mm_sub_ps(v_s14, _mm_mul_ps(v128_CRTM_15_2, v_s21));
        v_t5 = _mm_mul_ps(v128_CRTM_15_10,
                _mm_sub_ps( _mm_mul_ps(v128_CRTM_15_6, v_s22), v_in2));
        v_s24 = _mm_add_ps(v_s23, v_t5);
        v_s25 = _mm_sub_ps(v_s23, v_t5);
        v_s26 = _mm_sub_ps(v_t2, v_t1);
        v_s27 = _mm_add_ps(v_t1, v_t2);

        v_s28 = _mm_sub_ps(v_s24, v_s27);
        v_s29 = _mm_add_ps(_mm_mul_ps(v128_CRTM_15_3, v_s6),
                           _mm_mul_ps(v128_CRTM_15_4, v_s12));
        v_s30 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_8, v_s1),
                           _mm_mul_ps(v128_CRTM_15_7, v_s7));
        v_s31 = _mm_sub_ps(v_s30, v_s29);
        v_s32 = _mm_add_ps(v_s29, v_s30);

        // Output point 1: X(0)
        v_out1 = _mm_add_ps(v_s28, v_s31);
        curr_out = out + out_strides[1];
        STR_128_S(curr_out, v_out_stride, v_out1);

        // Output point 3: X(2)
        v_out4 = _mm_sub_ps(v_s28, v_s31);
        curr_out = out + out_strides[4];
        STR_128_S(curr_out, v_out_stride, v_out4);

        v_s33 = _mm_add_ps(v_s24, v_s27);
        v_s34 = _mm_add_ps(_mm_mul_ps(v128_CRTM_15_8, v_s7),
                           _mm_mul_ps(v128_CRTM_15_7, v_s1));
        v_s35 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_3, v_s12),
                           _mm_mul_ps(v128_CRTM_15_4, v_s6));
        v_s36 = _mm_add_ps(v_s35, v_s34);

        // Output point 4: X(3)
        v_out7 = _mm_add_ps(v_s33, v_s36);
        curr_out = out + out_strides[7];
        STR_128_S(curr_out, v_out_stride, v_out7);

        // Output point 7: X(6)
        v_out13 = _mm_sub_ps(v_s33, v_s36);
        curr_out = out + out_strides[13];
        STR_128_S(curr_out, v_out_stride, v_out13);

        v_s37 = _mm_sub_ps(v_s25, v_s26);
        v_s38 = _mm_sub_ps(v_s35, v_s34);

        // Output point 2: X(1)
        v_out2 = _mm_add_ps(v_s37, v_s38);
        curr_out = out + out_strides[2];
        STR_128_S(curr_out, v_out_stride, v_out2);

        // Output point 5: X(4)
        v_out8 = _mm_sub_ps(v_s37, v_s38);
        curr_out = out + out_strides[8];
        STR_128_S(curr_out, v_out_stride, v_out8);

        v_s39 = _mm_add_ps(v_s25, v_s26);

        // Output point 6: X(5)
        v_out11 = _mm_sub_ps(v_s39, v_s32);
        curr_out = out + out_strides[11];
        STR_128_S(curr_out, v_out_stride, v_out11);

        // Output point 8: X(7)
        v_out14 = _mm_add_ps(v_s39, v_s32);
        curr_out = out + out_strides[14];
        STR_128_S(curr_out, v_out_stride, v_out14);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (n & 2)
    {
        __m128 v_in0, v_in1, v_in2, v_in3;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
               v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27,
               v_s28, v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36,
               v_s37, v_s38, v_s39, v_s40;
        __m128 v_t0, v_t1, v_t2, v_t4, v_t5;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
               v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14;

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

        // Input point 1: X(0)
        curr_in = in + in_strides[0];
        LDHR_128_S(curr_in, v_in_stride, v_in0);
        // Input point 4: X(3)
        curr_in = in + in_strides[3];
        LDHR_128_S(curr_in, v_in_stride, v_in1);
        // Input point 14: X(13)
        curr_in = in + in_strides[13];
        LDHR_128_S(curr_in, v_in_stride, v_in2);
        // Input point 6: X(5)
        curr_in = in + in_strides[5];
        LDHR_128_S(curr_in, v_in_stride, v_in3);

        v_s0 = _mm_add_ps(v_in1, v_in2);
        v_s1 = _mm_sub_ps(v_in1, v_in2);
        v_s2 = _mm_add_ps(v_in3, v_s0);
        v_s3 = _mm_sub_ps(v_in3, _mm_mul_ps(v128_CRTM_15_2, v_s0));

        // Input point 5: X(4)
        curr_in = in + in_strides[4];
        LDHR_128_S(curr_in, v_in_stride, v_in1);
        // Input point 15: X(14)
        curr_in = in + in_strides[14];
        LDHR_128_S(curr_in, v_in_stride, v_in2);
        // Input point 7: X(6)
        curr_in = in + in_strides[6];
        LDHR_128_S(curr_in, v_in_stride, v_in3);

        v_s0 = _mm_add_ps(v_in1, v_in2);
        v_s4 = _mm_sub_ps(v_in1, v_in2);
        v_s5 = _mm_sub_ps(v_s0, v_in3);
        v_s6 = _mm_add_ps(v_in3, _mm_mul_ps(v128_CRTM_15_2, v_s0));

        // Input point 2: X(1)
        curr_in = in + in_strides[1];
        LDHR_128_S(curr_in, v_in_stride, v_in1);
        // Input point 8: X(7)
        curr_in = in + in_strides[7];
        LDHR_128_S(curr_in, v_in_stride, v_in2);
        // Input point 12: X(11)
        curr_in = in + in_strides[11];
        LDHR_128_S(curr_in, v_in_stride, v_in3);

        v_s0 = _mm_add_ps(v_in1, v_in2);
        v_s7 = _mm_sub_ps(v_in2, v_in1);
        v_s8 = _mm_add_ps(v_in3, v_s0);
        v_s9 = _mm_sub_ps(v_in3, _mm_mul_ps(v128_CRTM_15_2, v_s0));

        // Input point 3: X(2)
        curr_in = in + in_strides[2];
        LDHR_128_S(curr_in, v_in_stride, v_in1);
        // Input point 9: X(8)
        curr_in = in + in_strides[8];
        LDHR_128_S(curr_in, v_in_stride, v_in2);
        // Input point 13: X(12)
        curr_in = in + in_strides[12];
        LDHR_128_S(curr_in, v_in_stride, v_in3);

        v_s0 = _mm_sub_ps(v_in1, v_in2);
        v_s10 = _mm_add_ps(v_in2, v_in1);
        v_s11 = _mm_add_ps(v_in3, v_s0);
        v_s12 = _mm_sub_ps(v_in3, _mm_mul_ps(v128_CRTM_15_2, v_s0));

        // Input point 10: X(9)
        curr_in = in + in_strides[9];
        LDHR_128_S(curr_in, v_in_stride, v_in1);
        // Input point 11: X(10)
        curr_in = in + in_strides[10];
        LDHR_128_S(curr_in, v_in_stride, v_in2);

        v_s13 = _mm_add_ps(v_in0, _mm_mul_ps(v128_CRTM_15_5, v_in1));
        v_s14 = _mm_sub_ps(v_in0, v_in1);
        v_s15 = _mm_add_ps(v_s8, v_s2);
        v_t0 = _mm_mul_ps(v128_CRTM_15_1, _mm_sub_ps(v_s8, v_s2));
        v_s16 = _mm_sub_ps(v_s13, _mm_mul_ps(v128_CRTM_15_2, v_s15));

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_s13, _mm_mul_ps(v128_CRTM_15_5, v_s15));
        STHR_128_S(curr_out, v_out_stride, v_out0);

        v_s17 = _mm_add_ps(_mm_mul_ps(v128_CRTM_15_3, v_s11),
                           _mm_mul_ps(v128_CRTM_15_4, v_s5));
        v_s18 = _mm_add_ps(v_s16, v_t0);

        // Output point 2: X(1)
        v_out3 = _mm_sub_ps(v_s18, v_s17);
        curr_out = out + out_strides[3];
        STHR_128_S(curr_out, v_out_stride, v_out3);

        // Output point 7: X(6)
        v_out12 = _mm_add_ps(v_s18, v_s17);
        curr_out = out + out_strides[12];
        STHR_128_S(curr_out, v_out_stride, v_out12);

        v_s19 = _mm_sub_ps(v_s16, v_t0);
        v_s20 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_4, v_s11),
                           _mm_mul_ps(v128_CRTM_15_3, v_s5));

        // Output point 4: X(3)
        v_out6 = _mm_sub_ps(v_s19, v_s20);
        curr_out = out + out_strides[6];
        STHR_128_S(curr_out, v_out_stride, v_out6);

        // Output point 5: X(4)
        v_out9 = _mm_add_ps(v_s19, v_s20);
        curr_out = out + out_strides[9];
        STHR_128_S(curr_out, v_out_stride, v_out9);

        v_t1 = _mm_mul_ps(v128_CRTM_15_1, _mm_sub_ps(v_s9, v_s3));
        v_t2 = _mm_mul_ps(v128_CRTM_15_9, _mm_add_ps(v_s4, v_s10));
        v_s21 = _mm_add_ps(v_s9, v_s3);
        v_s22 = _mm_sub_ps(v_s4, v_s10);
        v_s40 = _mm_add_ps(_mm_mul_ps(v128_CRTM_15_5, v_s21), v_s14);
        v_t4 = _mm_mul_ps(v128_CRTM_15_10, _mm_add_ps(v_s22, v_in2));

        // Output point 3: X(2)
        v_out5 = _mm_add_ps(v_s40, v_t4);
        curr_out = out + out_strides[5];
        STHR_128_S(curr_out, v_out_stride, v_out5);

        // Output point 6: X(5)
        v_out10 = _mm_sub_ps(v_s40, v_t4);
        curr_out = out + out_strides[10];
        STHR_128_S(curr_out, v_out_stride, v_out10);

        v_s23 = _mm_sub_ps(v_s14, _mm_mul_ps(v128_CRTM_15_2, v_s21));
        v_t5  = _mm_mul_ps(v128_CRTM_15_10,
                _mm_sub_ps( _mm_mul_ps(v128_CRTM_15_6, v_s22), v_in2));
        v_s24 = _mm_add_ps(v_s23, v_t5);
        v_s25 = _mm_sub_ps(v_s23, v_t5);
        v_s26 = _mm_sub_ps(v_t2, v_t1);
        v_s27 = _mm_add_ps(v_t1, v_t2);

        v_s28 = _mm_sub_ps(v_s24, v_s27);
        v_s29 = _mm_add_ps(_mm_mul_ps(v128_CRTM_15_3, v_s6),
                           _mm_mul_ps(v128_CRTM_15_4, v_s12));
        v_s30 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_8, v_s1),
                           _mm_mul_ps(v128_CRTM_15_7, v_s7));
        v_s31 = _mm_sub_ps(v_s30, v_s29);
        v_s32 = _mm_add_ps(v_s29, v_s30);

        // Output point 1: X(0)
        v_out1 = _mm_add_ps(v_s28, v_s31);
        curr_out = out + out_strides[1];
        STHR_128_S(curr_out, v_out_stride, v_out1);

        // Output point 3: X(2)
        v_out4 = _mm_sub_ps(v_s28, v_s31);
        curr_out = out + out_strides[4];
        STHR_128_S(curr_out, v_out_stride, v_out4);

        v_s33 = _mm_add_ps(v_s24, v_s27);
        v_s34 = _mm_add_ps(_mm_mul_ps(v128_CRTM_15_8, v_s7),
                           _mm_mul_ps(v128_CRTM_15_7, v_s1));
        v_s35 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_15_3, v_s12),
                           _mm_mul_ps(v128_CRTM_15_4, v_s6));
        v_s36 = _mm_add_ps(v_s35, v_s34);

        // Output point 4: X(3)
        v_out7 = _mm_add_ps(v_s33, v_s36);
        curr_out = out + out_strides[7];
        STHR_128_S(curr_out, v_out_stride, v_out7);

        // Output point 7: X(6)
        v_out13 = _mm_sub_ps(v_s33, v_s36);
        curr_out = out + out_strides[13];
        STHR_128_S(curr_out, v_out_stride, v_out13);

        v_s37 = _mm_sub_ps(v_s25, v_s26);
        v_s38 = _mm_sub_ps(v_s35, v_s34);

        // Output point 2: X(1)
        v_out2 = _mm_add_ps(v_s37, v_s38);
        curr_out = out + out_strides[2];
        STHR_128_S(curr_out, v_out_stride, v_out2);

        // Output point 5: X(4)
        v_out8 = _mm_sub_ps(v_s37, v_s38);
        curr_out = out + out_strides[8];
        STHR_128_S(curr_out, v_out_stride, v_out8);

        v_s39 = _mm_add_ps(v_s25, v_s26);

        // Output point 6: X(5)
        v_out11 = _mm_sub_ps(v_s39, v_s32);
        curr_out = out + out_strides[11];
        STHR_128_S(curr_out, v_out_stride, v_out11);

        // Output point 8: X(7)
        v_out14 = _mm_add_ps(v_s39, v_s32);
        curr_out = out + out_strides[14];
        STHR_128_S(curr_out, v_out_stride, v_out14);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        FLOAT in0, in1, in2, in3, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10,
              s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23,
              s24, s25, s26, s27, s28, s29, s30, s31, s32, s33, s34, s35, s36,
              s37, s38, s39, t0, t1, t2, t3, t4, t5;

        in0 = *in;
        in1 = in[in_strides[3]];
        in2 = in[in_strides[13]];
        in3 = in[in_strides[5]];

        s0 = in1 + in2;
        s1 = in1 - in2;
        s2 = in3 + s0;
        s3 = in3 - (CRTM_15_2 * s0);

        in1 = in[in_strides[4]];
        in2 = in[in_strides[14]];
        in3 = in[in_strides[6]];

        s0 = in1 + in2;
        s4 = in1 - in2;
        s5 = s0 - in3;
        s6 = in3 + (CRTM_15_2 * s0);

        in1 = in[in_strides[1]];
        in2 = in[in_strides[7]];
        in3 = in[in_strides[11]];

        s0 = in1 + in2;
        s7 = in2 - in1;
        s8 = in3 + s0;
        s9 = in3 - (CRTM_15_2 * s0);

        in1 = in[in_strides[2]];
        in2 = in[in_strides[8]];
        in3 = in[in_strides[12]];

        s0 = in1 - in2;
        s10 = in2 + in1;
        s11 = in3 + s0;
        s12 = in3 - (CRTM_15_2 * s0);

        in1 = in[in_strides[9]];
        in2 = in[in_strides[10]];

        s13 = in0 + CRTM_15_5 * in1;
        s14 = in0 - in1;
        s15 = s8 + s2;
        t0 = CRTM_15_1 * (s8 - s2);
        s16 = s13 - (CRTM_15_2 * s15);

        *out = s13 + CRTM_15_5 * s15;

        s17 = (CRTM_15_3 * s11) + (CRTM_15_4 * s5);
        s18 = s16 + t0;

        out[out_strides[3]] = s18 - s17;
        out[out_strides[12]] = s18 + s17;

        s19 = s16 - t0;
        s20 = (CRTM_15_4 * s11) - (CRTM_15_3 * s5);

        out[out_strides[6]] = s19 - s20;
        out[out_strides[9]] = s19 + s20;

        t1 = CRTM_15_1 * (s9 - s3);
        t2 = CRTM_15_9 * (s10 + s4);
        s21 = (s9 + s3);
        s22 = s4 - s10;
        t3 = CRTM_15_5 * s21 + s14;
        t4 = CRTM_15_10 * (s22 + in2);

        out[out_strides[5]] = t3 + t4;
        out[out_strides[10]] = t3 - t4;

        s23 = s14 - (CRTM_15_2 * s21);
        t5 = CRTM_15_10 * ((CRTM_15_6 * s22) - in2);
        s24 = t5 + s23;
        s25 = s23 - t5;
        s26 = t2 - t1;
        s27 = t1 + t2;

        s28 = s24 - s27;
        s29 = (CRTM_15_4 * s12) + (CRTM_15_3 * s6);
        s30 = (CRTM_15_8 * s1) - (CRTM_15_7 * s7);
        s31 = s30 - s29;
        s32 = s29 + s30;

        out[out_strides[1]] = s28 + s31;
        out[out_strides[4]] = s28 - s31;

        s33 = s24 + s27;
        s34 = (CRTM_15_8 * s7) + (CRTM_15_7 * s1);
        s35 = (CRTM_15_3 * s12) - (CRTM_15_4 * s6);
        s36 = s35 + s34;

        out[out_strides[7]] = s33 + s36;
        out[out_strides[13]] = s33 - s36;

        s37 = s25 - s26;
        s38 = s35 - s34;
        out[out_strides[2]] = s37 + s38;
        out[out_strides[8]] = s37 - s38;

        s39 = s25 + s26;

        out[out_strides[11]] = s39 - s32;
        out[out_strides[14]] = s39 + s32;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID r2hc_rfft15avx256_fp64_fwd(VOID *in_real, VOID *in_imag,
                                       VOID *out_real, VOID *out_imag, INTP n,
                                       aoclfftz_strides_t *strides, UINT8 flag)
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

    for (cnt = 0; cnt < N; cnt++)
    {
        __m256d v_in0, v_in1, v_in2, v_in3;
        __m256d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
                v_s10, v_s11, v_s12, v_s13, v_s15, v_s16, v_s17, v_s19, v_s20,
                v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27, v_s28, v_s29,
                v_s30, v_s31;
        __m256d v_t0, v_t1, v_t2, v_t3;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14;

        curr_in = in;
        curr_out = out;

        // Input point 1: X(0)
        curr_in = in + in_strides[0];
        LDR_256_D(curr_in, v_in_stride, v_in0);
        // Input point 3: X(2)
        curr_in = in + in_strides[2];
        LDR_256_D(curr_in, v_in_stride, v_in1);
        // Input point 8: X(7)
        curr_in = in + in_strides[7];
        LDR_256_D(curr_in, v_in_stride, v_in2);
        // Input point 13: X(12)
        curr_in = in + in_strides[12];
        LDR_256_D(curr_in, v_in_stride, v_in3);

        v_s0 = _mm256_add_pd(v_in1, v_in2);
        v_s1 = _mm256_sub_pd(v_in1, v_in2);
        v_s2 = _mm256_add_pd(v_in3, v_s0);
        v_s3 = _mm256_sub_pd(v_in3, _mm256_mul_pd(v_CRTM_15_5, v_s0));

        // Input point 9: X(8)
        curr_in = in + in_strides[8];
        LDR_256_D(curr_in, v_in_stride, v_in1);
        // Input point 14: X(13)
        curr_in = in + in_strides[13];
        LDR_256_D(curr_in, v_in_stride, v_in2);
        // Input point 4: X(3)
        curr_in = in + in_strides[3];
        LDR_256_D(curr_in, v_in_stride, v_in3);

        v_s0 = _mm256_add_pd(v_in1, v_in2);
        v_s4 = _mm256_sub_pd(v_in2, v_in1);
        v_s5 = _mm256_add_pd(v_in3, v_s0);
        v_s6 = _mm256_sub_pd(v_in3, _mm256_mul_pd(v_CRTM_15_5, v_s0));

        v_s23 = _mm256_add_pd(v_s6, v_s3);
        v_s24 = _mm256_sub_pd(v_s6, v_s3);
        v_s26 = _mm256_add_pd(v_s4, v_s1);
        v_s27 = _mm256_sub_pd(v_s4, v_s1);

        // Input point 2: X(1)
        curr_in = in + in_strides[1];
        LDR_256_D(curr_in, v_in_stride, v_in1);
        // Input point 12: X(11)
        curr_in = in + in_strides[11];
        LDR_256_D(curr_in, v_in_stride, v_in2);
        // Input point 7: X(6)
        curr_in = in + in_strides[6];
        LDR_256_D(curr_in, v_in_stride, v_in3);

        v_s0 = _mm256_add_pd(v_in1, v_in2);
        v_s1 = _mm256_sub_pd(v_in2, v_in1);
        v_s7 = _mm256_add_pd(v_in3, v_s0);
        v_s3 = _mm256_sub_pd(v_in3, _mm256_mul_pd(v_CRTM_15_5, v_s0));

        // Input point 5: X(4)
        curr_in = in + in_strides[4];
        LDR_256_D(curr_in, v_in_stride, v_in1);
        // Input point 15: X(14)
        curr_in = in + in_strides[14];
        LDR_256_D(curr_in, v_in_stride, v_in2);
        // Input point 10: X(9)
        curr_in = in + in_strides[9];
        LDR_256_D(curr_in, v_in_stride, v_in3);

        v_s0 = _mm256_add_pd(v_in1, v_in2);
        v_s4 = _mm256_sub_pd(v_in2, v_in1);
        v_s8 = _mm256_add_pd(v_in3, v_s0);
        v_s6 = _mm256_sub_pd(v_in3, _mm256_mul_pd(v_CRTM_15_5, v_s0));

        v_s28 = _mm256_add_pd(v_s6, v_s3);
        v_s29 = _mm256_sub_pd(v_s3, v_s6);
        v_s30 = _mm256_add_pd(v_s4, v_s1);
        v_s31 = _mm256_sub_pd(v_s1, v_s4);

        // Input point 6: X(5)
        curr_in = in + in_strides[5];
        LDR_256_D(curr_in, v_in_stride, v_in1);
        // Input point 11: X(10)
        curr_in = in + in_strides[10];
        LDR_256_D(curr_in, v_in_stride, v_in2);

        v_s0 = _mm256_add_pd(v_in1, v_in2);
        v_t0 = _mm256_mul_pd(v_CRTM_15_6, _mm256_sub_pd(v_in2, v_in1));
        v_s9 = _mm256_add_pd(v_in0, v_s0);
        v_s3 = _mm256_sub_pd(v_in0, _mm256_mul_pd(v_CRTM_15_5, v_s0));

        v_s11 = _mm256_add_pd(v_s8, v_s7);
        v_s12 = _mm256_add_pd(v_s2, v_s5);
        v_s19 = _mm256_sub_pd(v_s5, v_s2);
        v_s20 = _mm256_sub_pd(v_s8, v_s7);
        v_s13 = _mm256_add_pd(v_s11, v_s12);
        v_t1 = _mm256_mul_pd(v_CRTM_15_1, _mm256_sub_pd(v_s11, v_s12));
        v_s15 = _mm256_sub_pd(v_s9, _mm256_mul_pd(v_CRTM_15_2, v_s13));

        // Output point 1: X(0)
        v_out0 = _mm256_add_pd(v_s9, v_s13);
        STR_256_D(curr_out, v_out_stride, v_out0);

        // Output point 3: X(2)
        v_out5 = _mm256_add_pd(v_s15, v_t1);
        // Output point 4: X(3)
        v_out6 = _mm256_add_pd(_mm256_mul_pd(v_CRTM_15_3, v_s20),
                               _mm256_mul_pd(v_CRTM_15_4, v_s19));
        curr_out = out + out_strides[5];
        STRI_2x256_D(curr_out, v_out_stride, v_out5, v_out6);

        // Output point 6: X(5)
        v_out11 = _mm256_sub_pd(v_s15, v_t1);
        // Output point 7: X(6)
        v_out12 = _mm256_sub_pd(_mm256_mul_pd(v_CRTM_15_4, v_s20),
                                _mm256_mul_pd(v_CRTM_15_3, v_s19));
        curr_out = out + out_strides[11];
        STRI_2x256_D(curr_out, v_out_stride, v_out11, v_out12);

        v_t2 = _mm256_mul_pd(v_CRTM_15_1, _mm256_sub_pd(v_s28, v_s23));
        v_s13 = _mm256_add_pd(v_s28, v_s23);

        // Output point 5: X(4)
        v_out9 = _mm256_add_pd(v_s13, v_s3);

        v_t3 = _mm256_mul_pd(v_CRTM_15_9, _mm256_add_pd(v_s30, v_s27));
        v_s17 = _mm256_sub_pd(v_s30, v_s27);

        // Output point 6: X(5)
        v_out10 = _mm256_sub_pd(_mm256_mul_pd(v_CRTM_15_6, v_s17), v_t0);
        curr_out = out + out_strides[9];
        STRI_2x256_D(curr_out, v_out_stride, v_out9, v_out10);

        v_s15 = _mm256_sub_pd(v_s3, _mm256_mul_pd(v_CRTM_15_2, v_s13));
        v_s20 = _mm256_add_pd(v_t0, _mm256_mul_pd(v_CRTM_15_10, v_s17));
        v_s21 = _mm256_sub_pd(v_s15, v_t2);
        v_s25 = _mm256_sub_pd(_mm256_mul_pd(v_CRTM_15_8, v_s26),
                              _mm256_mul_pd(v_CRTM_15_7, v_s31));

        // Output point 1: X(0)
        v_out1 = _mm256_add_pd(v_s21, v_s25);

        // Output point 4: X(3)
        v_out7 = _mm256_sub_pd(v_s21, v_s25);

        v_s21 = _mm256_add_pd(v_s15, v_t2);
        v_s23 = _mm256_add_pd(_mm256_mul_pd(v_CRTM_15_8, v_s31),
                              _mm256_mul_pd(v_CRTM_15_7, v_s26));

        // Output point 7: X(6)
        v_out13 = _mm256_add_pd(v_s21, v_s23);

        v_s22 = _mm256_add_pd(v_s20, v_t3);
        v_s16 = _mm256_add_pd(_mm256_mul_pd(v_CRTM_15_4, v_s29),
                              _mm256_mul_pd(v_CRTM_15_3, v_s24));

        // Output point 2: X(1)
        v_out2 = _mm256_sub_pd(v_s22, v_s16);
        curr_out = out + out_strides[1];
        STRI_2x256_D(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 5: X(4)
        v_out8 = _mm256_add_pd(v_s22, v_s16);
        curr_out = out + out_strides[7];
        STRI_2x256_D(curr_out, v_out_stride, v_out7, v_out8);

        v_s22 = _mm256_sub_pd(v_s20, v_t3);
        v_s10 = _mm256_sub_pd(_mm256_mul_pd(v_CRTM_15_3, v_s29),
                              _mm256_mul_pd(v_CRTM_15_4, v_s24));

        // Output point 8: X(7)
        v_out14 = _mm256_add_pd(v_s22, v_s10);
        curr_out = out + out_strides[13];
        STRI_2x256_D(curr_out, v_out_stride, v_out13, v_out14);

        // Output point 2: X(1)
        v_out3 = _mm256_sub_pd(v_s21, v_s23);

        // Output point 3: X(2)
        v_out4 = _mm256_sub_pd(v_s10, v_s22);
        curr_out = out + out_strides[3];
        STRI_2x256_D(curr_out, v_out_stride, v_out3, v_out4);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (n & 2)
    {
        __m128d v_in0, v_in1, v_in2, v_in3;
        __m128d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
                v_s10, v_s11, v_s12, v_s13, v_s15, v_s16, v_s17, v_s19, v_s20,
                v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27, v_s28, v_s29,
                v_s30, v_s31;
        __m128d v_t0, v_t1, v_t2, v_t3;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14;

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

        // Input point 1: X(0)
        curr_in = in + in_strides[0];
        LDR_128_D(curr_in, v_in_stride, v_in0);
        // Input point 3: X(2)
        curr_in = in + in_strides[2];
        LDR_128_D(curr_in, v_in_stride, v_in1);
        // Input point 8: X(7)
        curr_in = in + in_strides[7];
        LDR_128_D(curr_in, v_in_stride, v_in2);
        // Input point 13: X(12)
        curr_in = in + in_strides[12];
        LDR_128_D(curr_in, v_in_stride, v_in3);

        v_s0 = _mm_add_pd(v_in1, v_in2);
        v_s1 = _mm_sub_pd(v_in1, v_in2);
        v_s2 = _mm_add_pd(v_in3, v_s0);
        v_s3 = _mm_sub_pd(v_in3, _mm_mul_pd(v128_CRTM_15_5, v_s0));

        // Input point 9: X(8)
        curr_in = in + in_strides[8];
        LDR_128_D(curr_in, v_in_stride, v_in1);
        // Input point 14: X(13)
        curr_in = in + in_strides[13];
        LDR_128_D(curr_in, v_in_stride, v_in2);
        // Input point 4: X(3)
        curr_in = in + in_strides[3];
        LDR_128_D(curr_in, v_in_stride, v_in3);

        v_s0 = _mm_add_pd(v_in1, v_in2);
        v_s4 = _mm_sub_pd(v_in2, v_in1);
        v_s5 = _mm_add_pd(v_in3, v_s0);
        v_s6 = _mm_sub_pd(v_in3, _mm_mul_pd(v128_CRTM_15_5, v_s0));

        v_s23 = _mm_add_pd(v_s6, v_s3);
        v_s24 = _mm_sub_pd(v_s6, v_s3);
        v_s26 = _mm_add_pd(v_s4, v_s1);
        v_s27 = _mm_sub_pd(v_s4, v_s1);

        // Input point 2: X(1)
        curr_in = in + in_strides[1];
        LDR_128_D(curr_in, v_in_stride, v_in1);
        // Input point 12: X(11)
        curr_in = in + in_strides[11];
        LDR_128_D(curr_in, v_in_stride, v_in2);
        // Input point 7: X(6)
        curr_in = in + in_strides[6];
        LDR_128_D(curr_in, v_in_stride, v_in3);

        v_s0 = _mm_add_pd(v_in1, v_in2);
        v_s1 = _mm_sub_pd(v_in2, v_in1);
        v_s7 = _mm_add_pd(v_in3, v_s0);
        v_s3 = _mm_sub_pd(v_in3, _mm_mul_pd(v128_CRTM_15_5, v_s0));

        // Input point 5: X(4)
        curr_in = in + in_strides[4];
        LDR_128_D(curr_in, v_in_stride, v_in1);
        // Input point 15: X(14)
        curr_in = in + in_strides[14];
        LDR_128_D(curr_in, v_in_stride, v_in2);
        // Input point 10: X(9)
        curr_in = in + in_strides[9];
        LDR_128_D(curr_in, v_in_stride, v_in3);

        v_s0 = _mm_add_pd(v_in1, v_in2);
        v_s4 = _mm_sub_pd(v_in2, v_in1);
        v_s8 = _mm_add_pd(v_in3, v_s0);
        v_s6 = _mm_sub_pd(v_in3, _mm_mul_pd(v128_CRTM_15_5, v_s0));

        v_s28 = _mm_add_pd(v_s6, v_s3);
        v_s29 = _mm_sub_pd(v_s3, v_s6);
        v_s30 = _mm_add_pd(v_s4, v_s1);
        v_s31 = _mm_sub_pd(v_s1, v_s4);

        // Input point 6: X(5)
        curr_in = in + in_strides[5];
        LDR_128_D(curr_in, v_in_stride, v_in1);
        // Input point 11: X(10)
        curr_in = in + in_strides[10];
        LDR_128_D(curr_in, v_in_stride, v_in2);

        v_s0 = _mm_add_pd(v_in1, v_in2);
        v_t0 = _mm_mul_pd(v128_CRTM_15_6, _mm_sub_pd(v_in2, v_in1));
        v_s9 = _mm_add_pd(v_in0, v_s0);
        v_s3 = _mm_sub_pd(v_in0, _mm_mul_pd(v128_CRTM_15_5, v_s0));

        v_s11 = _mm_add_pd(v_s8, v_s7);
        v_s12 = _mm_add_pd(v_s2, v_s5);
        v_s19 = _mm_sub_pd(v_s5, v_s2);
        v_s20 = _mm_sub_pd(v_s8, v_s7);
        v_s13 = _mm_add_pd(v_s11, v_s12);
        v_t1 = _mm_mul_pd(v128_CRTM_15_1, _mm_sub_pd(v_s11, v_s12));
        v_s15 = _mm_sub_pd(v_s9, _mm_mul_pd(v128_CRTM_15_2, v_s13));

        // Output point 1: X(0)
        v_out0 = _mm_add_pd(v_s9, v_s13);
        STR_128_D(curr_out, v_out_stride, v_out0);

        // Output point 3: X(2)
        v_out5 = _mm_add_pd(v_s15, v_t1);
        // Output point 4: X(3)
        v_out6 = _mm_add_pd(_mm_mul_pd(v128_CRTM_15_3, v_s20),
                            _mm_mul_pd(v128_CRTM_15_4, v_s19));
        curr_out = out + out_strides[5];
        STRI_2x128_D(curr_out, v_out_stride, v_out5, v_out6);

        // Output point 6: X(5)
        v_out11 = _mm_sub_pd(v_s15, v_t1);
        // Output point 7: X(6)
        v_out12 = _mm_sub_pd(_mm_mul_pd(v128_CRTM_15_4, v_s20),
                             _mm_mul_pd(v128_CRTM_15_3, v_s19));
        curr_out = out + out_strides[11];
        STRI_2x128_D(curr_out, v_out_stride, v_out11, v_out12);

        v_t2 = _mm_mul_pd(v128_CRTM_15_1, _mm_sub_pd(v_s28, v_s23));
        v_s13 = _mm_add_pd(v_s28, v_s23);

        // Output point 5: X(4)
        v_out9 = _mm_add_pd(v_s13, v_s3);

        v_t3 = _mm_mul_pd(v128_CRTM_15_9, _mm_add_pd(v_s30, v_s27));
        v_s17 = _mm_sub_pd(v_s30, v_s27);

        // Output point 6: X(5)
        v_out10 = _mm_sub_pd(_mm_mul_pd(v128_CRTM_15_6, v_s17), v_t0);
        curr_out = out + out_strides[9];
        STRI_2x128_D(curr_out, v_out_stride, v_out9, v_out10);

        v_s15 = _mm_sub_pd(v_s3, _mm_mul_pd(v128_CRTM_15_2, v_s13));
        v_s20 = _mm_add_pd(v_t0, _mm_mul_pd(v128_CRTM_15_10, v_s17));
        v_s21 = _mm_sub_pd(v_s15, v_t2);
        v_s25 = _mm_sub_pd(_mm_mul_pd(v128_CRTM_15_8, v_s26),
                           _mm_mul_pd(v128_CRTM_15_7, v_s31));

        // Output point 1: X(0)
        v_out1 = _mm_add_pd(v_s21, v_s25);

        // Output point 4: X(3)
        v_out7 = _mm_sub_pd(v_s21, v_s25);

        v_s21 = _mm_add_pd(v_s15, v_t2);
        v_s23 = _mm_add_pd(_mm_mul_pd(v128_CRTM_15_8, v_s31),
                           _mm_mul_pd(v128_CRTM_15_7, v_s26));

        // Output point 7: X(6)
        v_out13 = _mm_add_pd(v_s21, v_s23);

        v_s22 = _mm_add_pd(v_s20, v_t3);
        v_s16 = _mm_add_pd(_mm_mul_pd(v128_CRTM_15_4, v_s29),
                           _mm_mul_pd(v128_CRTM_15_3, v_s24));

        // Output point 2: X(1)
        v_out2 = _mm_sub_pd(v_s22, v_s16);
        curr_out = out + out_strides[1];
        STRI_2x128_D(curr_out, v_out_stride, v_out1, v_out2);
        // Output point 5: X(4)
        v_out8 = _mm_add_pd(v_s22, v_s16);
        curr_out = out + out_strides[7];
        STRI_2x128_D(curr_out, v_out_stride, v_out7, v_out8);

        v_s22 = _mm_sub_pd(v_s20, v_t3);
        v_s10 = _mm_sub_pd(_mm_mul_pd(v128_CRTM_15_3, v_s29),
                           _mm_mul_pd(v128_CRTM_15_4, v_s24));

        // Output point 8: X(7)
        v_out14 = _mm_add_pd(v_s22, v_s10);
        curr_out = out + out_strides[13];
        STRI_2x128_D(curr_out, v_out_stride, v_out13, v_out14);

        // Output point 2: X(1)
        v_out3 = _mm_sub_pd(v_s21, v_s23);

        // Output point 3: X(2)
        v_out4 = _mm_sub_pd(v_s10, v_s22);
        curr_out = out + out_strides[3];
        STRI_2x128_D(curr_out, v_out_stride, v_out3, v_out4);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        DOUBLE in0, in1, in2, in3, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10,
               s11, s12, s13, s15, s16, s17, s19, s20, s21, s22, s23, s24,
               s25, s26, s27, s28, s29, s30, s31, t0, t1, t2, t3;

        in0 = *in;
        in1 = in[in_strides[2]];
        in2 = in[in_strides[7]];
        in3 = in[in_strides[12]];

        s0 = in1 + in2;
        s1 = in1 - in2;
        s2 = in3 + s0;
        s3 = in3 - (CRTM_15_5 * s0);

        in1 = in[in_strides[8]];
        in2 = in[in_strides[13]];
        in3 = in[in_strides[3]];

        s0 = in1 + in2;
        s4 = in2 - in1;
        s5 = in3 + s0;
        s6 = in3 - (CRTM_15_5 * s0);

        s23 = s6 + s3;
        s24 = s6 - s3;
        s26 = s4 + s1;
        s27 = s4 - s1;

        in1 = in[in_strides[1]];
        in2 = in[in_strides[11]];
        in3 = in[in_strides[6]];

        s0 = in1 + in2;
        s1 = in2 - in1;
        s7 = in3 + s0;
        s3 = in3 - (CRTM_15_5 * s0);

        in1 = in[in_strides[4]];
        in2 = in[in_strides[14]];
        in3 = in[in_strides[9]];

        s0 = in1 + in2;
        s4 = in2 - in1;
        s8 = in3 + s0;
        s6 = in3 - (CRTM_15_5 * s0);

        s28 = s6 + s3;
        s29 = s3 - s6;
        s30 = s4 + s1;
        s31 = s1 - s4;

        in1 = in[in_strides[5]];
        in2 = in[in_strides[10]];

        s0 = in1 + in2;
        t0 = CRTM_15_6 * (in2 - in1);
        s9 = in0 + s0;
        s3 = in0 - (CRTM_15_5 * s0);

        s11 = s8 + s7;
        s12 = s2 + s5;
        s19 = s5 - s2;
        s20 = s8 - s7;
        s13 = s11 + s12;
        t1 = CRTM_15_1 * (s11 - s12);
        s15 = s9 - (CRTM_15_2 * s13);

        *out = s9 + s13;

        out[out_strides[5]] = s15 + t1;
        out[out_strides[6]] = (CRTM_15_3 * s20) + (CRTM_15_4 * s19);

        out[out_strides[11]] = s15 - t1;
        out[out_strides[12]] = (CRTM_15_4 * s20) - (CRTM_15_3 * s19);

        t2 = CRTM_15_1 * (s28 - s23);
        s13 = s28 + s23;

        out[out_strides[9]] = s13 + s3;

        t3 = CRTM_15_9 * (s30 + s27);
        s17 = s30 - s27;

        out[out_strides[10]] = CRTM_15_6 * s17 - t0;

        s15 = s3 - (CRTM_15_2 * s13);
        s20 = t0 + (CRTM_15_10 * s17);
        s21 = s15 - t2;
        s25 = (CRTM_15_8 * s26) - (CRTM_15_7 * s31);

        out[out_strides[1]] = s21 + s25;
        out[out_strides[7]] = s21 - s25;

        s21 = s15 + t2;
        s23 = (CRTM_15_8 * s31) + (CRTM_15_7 * s26);

        out[out_strides[13]] = s21 + s23;

        s22 = s20 + t3;
        s16 = (CRTM_15_4 * s29) + (CRTM_15_3 * s24);

        out[out_strides[2]] = s22 - s16;
        out[out_strides[8]] = s22 + s16;

        s22 = s20 - t3;
        s10 = (CRTM_15_3 * s29) - (CRTM_15_4 * s24);
        out[out_strides[14]] = s22 + s10;

        out[out_strides[3]] = s21 - s23;
        out[out_strides[4]] = s10 - s22;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID r2hc_rfft15avx256_fp64_bwd(VOID *in_real, VOID *in_imag,
                                       VOID *out_real, VOID *out_imag, INTP n,
                                       aoclfftz_strides_t *strides, UINT8 flag)
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
    // Below CRTMs are the product of the above CRTMs, Precomputed to save
    // multiplications on the fly.
    // CRTM_15_7 = CRTM_15_6 * CRTM_15_4
    const DOUBLE CRTM_15_7 =
        1.01807392091025412936433958497993431950210362069154;
    // CRTM_15_8 = CRTM_15_6 * CRTM_15_3
    const DOUBLE CRTM_15_8 =
        1.64727820709266368541488232323193203275710390365294;
    // CRTM_15_9 = CRTM_15_6 * CRTM_15_1
    const DOUBLE CRTM_15_9 =
        0.96824583655185421224048777314959976915574786128504;
    // CRTM_15_10 = CRTM_15_6 * CRTM_15_5
    const DOUBLE CRTM_15_10 = 1.732050807568877293527446341505872366942805254;

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

    for (cnt = 0; cnt < N; cnt++)
    {
        __m256d v_in0, v_in1, v_in2, v_in3;
        __m256d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
                v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
                v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27,
                v_s28, v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36,
                v_s37, v_s38, v_s39, v_s40;
        __m256d v_t0, v_t1, v_t2, v_t4, v_t5;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14;

        curr_in = in;
        curr_out = out;

        // Input point 1: X(0)
        curr_in = in + in_strides[0];
        LDR_256_D(curr_in, v_in_stride, v_in0);
        // Input point 4: X(3)
        curr_in = in + in_strides[3];
        LDR_256_D(curr_in, v_in_stride, v_in1);
        // Input point 14: X(13)
        curr_in = in + in_strides[13];
        LDR_256_D(curr_in, v_in_stride, v_in2);
        // Input point 6: X(5)
        curr_in = in + in_strides[5];
        LDR_256_D(curr_in, v_in_stride, v_in3);

        v_s0 = _mm256_add_pd(v_in1, v_in2);
        v_s1 = _mm256_sub_pd(v_in1, v_in2);
        v_s2 = _mm256_add_pd(v_in3, v_s0);
        v_s3 = _mm256_sub_pd(v_in3, _mm256_mul_pd(v_CRTM_15_2, v_s0));

        // Input point 5: X(4)
        curr_in = in + in_strides[4];
        LDR_256_D(curr_in, v_in_stride, v_in1);
        // Input point 15: X(14)
        curr_in = in + in_strides[14];
        LDR_256_D(curr_in, v_in_stride, v_in2);
        // Input point 7: X(6)
        curr_in = in + in_strides[6];
        LDR_256_D(curr_in, v_in_stride, v_in3);

        v_s0 = _mm256_add_pd(v_in1, v_in2);
        v_s4 = _mm256_sub_pd(v_in1, v_in2);
        v_s5 = _mm256_sub_pd(v_s0, v_in3);
        v_s6 = _mm256_add_pd(v_in3, _mm256_mul_pd(v_CRTM_15_2, v_s0));

        // Input point 2: X(1)
        curr_in = in + in_strides[1];
        LDR_256_D(curr_in, v_in_stride, v_in1);
        // Input point 8: X(7)
        curr_in = in + in_strides[7];
        LDR_256_D(curr_in, v_in_stride, v_in2);
        // Input point 12: X(11)
        curr_in = in + in_strides[11];
        LDR_256_D(curr_in, v_in_stride, v_in3);

        v_s0 = _mm256_add_pd(v_in1, v_in2);
        v_s7 = _mm256_sub_pd(v_in2, v_in1);
        v_s8 = _mm256_add_pd(v_in3, v_s0);
        v_s9 = _mm256_sub_pd(v_in3, _mm256_mul_pd(v_CRTM_15_2, v_s0));

        // Input point 3: X(2)
        curr_in = in + in_strides[2];
        LDR_256_D(curr_in, v_in_stride, v_in1);
        // Input point 9: X(8)
        curr_in = in + in_strides[8];
        LDR_256_D(curr_in, v_in_stride, v_in2);
        // Input point 13: X(12)
        curr_in = in + in_strides[12];
        LDR_256_D(curr_in, v_in_stride, v_in3);

        v_s0 = _mm256_sub_pd(v_in1, v_in2);
        v_s10 = _mm256_add_pd(v_in2, v_in1);
        v_s11 = _mm256_add_pd(v_in3, v_s0);
        v_s12 = _mm256_sub_pd(v_in3, _mm256_mul_pd(v_CRTM_15_2, v_s0));

        // Input point 10: X(9)
        curr_in = in + in_strides[9];
        LDR_256_D(curr_in, v_in_stride, v_in1);
        // Input point 11: X(10)
        curr_in = in + in_strides[10];
        LDR_256_D(curr_in, v_in_stride, v_in2);

        v_s13 = _mm256_add_pd(v_in0, _mm256_mul_pd(v_CRTM_15_5, v_in1));
        v_s14 = _mm256_sub_pd(v_in0, v_in1);
        v_s15 = _mm256_add_pd(v_s8, v_s2);
        v_t0 = _mm256_mul_pd(v_CRTM_15_1, _mm256_sub_pd(v_s8, v_s2));
        v_s16 = _mm256_sub_pd(v_s13, _mm256_mul_pd(v_CRTM_15_2, v_s15));

        // Output point 1: X(0)
        v_out0 = _mm256_add_pd(v_s13, _mm256_mul_pd(v_CRTM_15_5, v_s15));
        STR_256_D(curr_out, v_out_stride, v_out0);

        v_s17 = _mm256_add_pd(_mm256_mul_pd(v_CRTM_15_3, v_s11),
                              _mm256_mul_pd(v_CRTM_15_4, v_s5));
        v_s18 = _mm256_add_pd(v_s16, v_t0);

        // Output point 2: X(1)
        v_out3 = _mm256_sub_pd(v_s18, v_s17);
        curr_out = out + out_strides[3];
        STR_256_D(curr_out, v_out_stride, v_out3);

        // Output point 7: X(6)
        v_out12 = _mm256_add_pd(v_s18, v_s17);
        curr_out = out + out_strides[12];
        STR_256_D(curr_out, v_out_stride, v_out12);

        v_s19 = _mm256_sub_pd(v_s16, v_t0);
        v_s20 = _mm256_sub_pd(_mm256_mul_pd(v_CRTM_15_4, v_s11),
                              _mm256_mul_pd(v_CRTM_15_3, v_s5));

        // Output point 4: X(3)
        v_out6 = _mm256_sub_pd(v_s19, v_s20);
        curr_out = out + out_strides[6];
        STR_256_D(curr_out, v_out_stride, v_out6);

        // Output point 5: X(4)
        v_out9 = _mm256_add_pd(v_s19, v_s20);
        curr_out = out + out_strides[9];
        STR_256_D(curr_out, v_out_stride, v_out9);

        v_t1 = _mm256_mul_pd(v_CRTM_15_1, _mm256_sub_pd(v_s9, v_s3));
        v_t2 = _mm256_mul_pd(v_CRTM_15_9, _mm256_add_pd(v_s4, v_s10));
        v_s21 = _mm256_add_pd(v_s9, v_s3);
        v_s22 = _mm256_sub_pd(v_s4, v_s10);
        v_s40 = _mm256_add_pd(_mm256_mul_pd(v_CRTM_15_5, v_s21), v_s14);
        v_t4 = _mm256_mul_pd(v_CRTM_15_10, _mm256_add_pd(v_s22, v_in2));

        // Output point 3: X(2)
        v_out5 = _mm256_add_pd(v_s40, v_t4);
        curr_out = out + out_strides[5];
        STR_256_D(curr_out, v_out_stride, v_out5);

        // Output point 6: X(5)
        v_out10 = _mm256_sub_pd(v_s40, v_t4);
        curr_out = out + out_strides[10];
        STR_256_D(curr_out, v_out_stride, v_out10);

        v_s23 = _mm256_sub_pd(v_s14, _mm256_mul_pd(v_CRTM_15_2, v_s21));
        v_t5 = _mm256_mul_pd(v_CRTM_15_10,
                _mm256_sub_pd( _mm256_mul_pd(v_CRTM_15_6, v_s22), v_in2));
        v_s24 = _mm256_add_pd(v_s23, v_t5);
        v_s25 = _mm256_sub_pd(v_s23, v_t5);
        v_s26 = _mm256_sub_pd(v_t2, v_t1);
        v_s27 = _mm256_add_pd(v_t1, v_t2);

        v_s28 = _mm256_sub_pd(v_s24, v_s27);
        v_s29 = _mm256_add_pd(_mm256_mul_pd(v_CRTM_15_3, v_s6),
                              _mm256_mul_pd(v_CRTM_15_4, v_s12));
        v_s30 = _mm256_sub_pd(_mm256_mul_pd(v_CRTM_15_8, v_s1),
                              _mm256_mul_pd(v_CRTM_15_7, v_s7));
        v_s31 = _mm256_sub_pd(v_s30, v_s29);
        v_s32 = _mm256_add_pd(v_s29, v_s30);

        // Output point 1: X(0)
        v_out1 = _mm256_add_pd(v_s28, v_s31);
        curr_out = out + out_strides[1];
        STR_256_D(curr_out, v_out_stride, v_out1);

        // Output point 3: X(2)
        v_out4 = _mm256_sub_pd(v_s28, v_s31);
        curr_out = out + out_strides[4];
        STR_256_D(curr_out, v_out_stride, v_out4);

        v_s33 = _mm256_add_pd(v_s24, v_s27);
        v_s34 = _mm256_add_pd(_mm256_mul_pd(v_CRTM_15_8, v_s7),
                              _mm256_mul_pd(v_CRTM_15_7, v_s1));
        v_s35 = _mm256_sub_pd(_mm256_mul_pd(v_CRTM_15_3, v_s12),
                              _mm256_mul_pd(v_CRTM_15_4, v_s6));
        v_s36 = _mm256_add_pd(v_s35, v_s34);

        // Output point 4: X(3)
        v_out7 = _mm256_add_pd(v_s33, v_s36);
        curr_out = out + out_strides[7];
        STR_256_D(curr_out, v_out_stride, v_out7);

        // Output point 7: X(6)
        v_out13 = _mm256_sub_pd(v_s33, v_s36);
        curr_out = out + out_strides[13];
        STR_256_D(curr_out, v_out_stride, v_out13);

        v_s37 = _mm256_sub_pd(v_s25, v_s26);
        v_s38 = _mm256_sub_pd(v_s35, v_s34);

        // Output point 2: X(1)
        v_out2 = _mm256_add_pd(v_s37, v_s38);
        curr_out = out + out_strides[2];
        STR_256_D(curr_out, v_out_stride, v_out2);

        // Output point 5: X(4)
        v_out8 = _mm256_sub_pd(v_s37, v_s38);
        curr_out = out + out_strides[8];
        STR_256_D(curr_out, v_out_stride, v_out8);

        v_s39 = _mm256_add_pd(v_s25, v_s26);

        // Output point 6: X(5)
        v_out11 = _mm256_sub_pd(v_s39, v_s32);
        curr_out = out + out_strides[11];
        STR_256_D(curr_out, v_out_stride, v_out11);

        // Output point 8: X(7)
        v_out14 = _mm256_add_pd(v_s39, v_s32);
        curr_out = out + out_strides[14];
        STR_256_D(curr_out, v_out_stride, v_out14);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (n & 2)
    {
        __m128d v_in0, v_in1, v_in2, v_in3;
        __m128d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
                v_s10, v_s11, v_s12, v_s13, v_s14, v_s15, v_s16, v_s17, v_s18,
                v_s19, v_s20, v_s21, v_s22, v_s23, v_s24, v_s25, v_s26, v_s27,
                v_s28, v_s29, v_s30, v_s31, v_s32, v_s33, v_s34, v_s35, v_s36,
                v_s37, v_s38, v_s39, v_s40;
        __m128d v_t0, v_t1, v_t2, v_t4, v_t5;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7,
                v_out8, v_out9, v_out10, v_out11, v_out12, v_out13, v_out14;

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

        // Input point 1: X(0)
        curr_in = in + in_strides[0];
        LDR_128_D(curr_in, v_in_stride, v_in0);
        // Input point 4: X(3)
        curr_in = in + in_strides[3];
        LDR_128_D(curr_in, v_in_stride, v_in1);
        // Input point 14: X(13)
        curr_in = in + in_strides[13];
        LDR_128_D(curr_in, v_in_stride, v_in2);
        // Input point 6: X(5)
        curr_in = in + in_strides[5];
        LDR_128_D(curr_in, v_in_stride, v_in3);

        v_s0 = _mm_add_pd(v_in1, v_in2);
        v_s1 = _mm_sub_pd(v_in1, v_in2);
        v_s2 = _mm_add_pd(v_in3, v_s0);
        v_s3 = _mm_sub_pd(v_in3, _mm_mul_pd(v128_CRTM_15_2, v_s0));

        // Input point 5: X(4)
        curr_in = in + in_strides[4];
        LDR_128_D(curr_in, v_in_stride, v_in1);
        // Input point 15: X(14)
        curr_in = in + in_strides[14];
        LDR_128_D(curr_in, v_in_stride, v_in2);
        // Input point 7: X(6)
        curr_in = in + in_strides[6];
        LDR_128_D(curr_in, v_in_stride, v_in3);

        v_s0 = _mm_add_pd(v_in1, v_in2);
        v_s4 = _mm_sub_pd(v_in1, v_in2);
        v_s5 = _mm_sub_pd(v_s0, v_in3);
        v_s6 = _mm_add_pd(v_in3, _mm_mul_pd(v128_CRTM_15_2, v_s0));

        // Input point 2: X(1)
        curr_in = in + in_strides[1];
        LDR_128_D(curr_in, v_in_stride, v_in1);
        // Input point 8: X(7)
        curr_in = in + in_strides[7];
        LDR_128_D(curr_in, v_in_stride, v_in2);
        // Input point 12: X(11)
        curr_in = in + in_strides[11];
        LDR_128_D(curr_in, v_in_stride, v_in3);

        v_s0 = _mm_add_pd(v_in1, v_in2);
        v_s7 = _mm_sub_pd(v_in2, v_in1);
        v_s8 = _mm_add_pd(v_in3, v_s0);
        v_s9 = _mm_sub_pd(v_in3, _mm_mul_pd(v128_CRTM_15_2, v_s0));

        // Input point 3: X(2)
        curr_in = in + in_strides[2];
        LDR_128_D(curr_in, v_in_stride, v_in1);
        // Input point 9: X(8)
        curr_in = in + in_strides[8];
        LDR_128_D(curr_in, v_in_stride, v_in2);
        // Input point 13: X(12)
        curr_in = in + in_strides[12];
        LDR_128_D(curr_in, v_in_stride, v_in3);

        v_s0 = _mm_sub_pd(v_in1, v_in2);
        v_s10 = _mm_add_pd(v_in2, v_in1);
        v_s11 = _mm_add_pd(v_in3, v_s0);
        v_s12 = _mm_sub_pd(v_in3, _mm_mul_pd(v128_CRTM_15_2, v_s0));

        // Input point 10: X(9)
        curr_in = in + in_strides[9];
        LDR_128_D(curr_in, v_in_stride, v_in1);
        // Input point 11: X(10)
        curr_in = in + in_strides[10];
        LDR_128_D(curr_in, v_in_stride, v_in2);

        v_s13 = _mm_add_pd(v_in0, _mm_mul_pd(v128_CRTM_15_5, v_in1));
        v_s14 = _mm_sub_pd(v_in0, v_in1);
        v_s15 = _mm_add_pd(v_s8, v_s2);
        v_t0 = _mm_mul_pd(v128_CRTM_15_1, _mm_sub_pd(v_s8, v_s2));
        v_s16 = _mm_sub_pd(v_s13, _mm_mul_pd(v128_CRTM_15_2, v_s15));

        // Output point 1: X(0)
        v_out0 = _mm_add_pd(v_s13, _mm_mul_pd(v128_CRTM_15_5, v_s15));
        STR_128_D(curr_out, v_out_stride, v_out0);

        v_s17 = _mm_add_pd(_mm_mul_pd(v128_CRTM_15_3, v_s11),
                           _mm_mul_pd(v128_CRTM_15_4, v_s5));
        v_s18 = _mm_add_pd(v_s16, v_t0);

        // Output point 2: X(1)
        v_out3 = _mm_sub_pd(v_s18, v_s17);
        curr_out = out + out_strides[3];
        STR_128_D(curr_out, v_out_stride, v_out3);

        // Output point 7: X(6)
        v_out12 = _mm_add_pd(v_s18, v_s17);
        curr_out = out + out_strides[12];
        STR_128_D(curr_out, v_out_stride, v_out12);

        v_s19 = _mm_sub_pd(v_s16, v_t0);
        v_s20 = _mm_sub_pd(_mm_mul_pd(v128_CRTM_15_4, v_s11),
                           _mm_mul_pd(v128_CRTM_15_3, v_s5));

        // Output point 4: X(3)
        v_out6 = _mm_sub_pd(v_s19, v_s20);
        curr_out = out + out_strides[6];
        STR_128_D(curr_out, v_out_stride, v_out6);

        // Output point 5: X(4)
        v_out9 = _mm_add_pd(v_s19, v_s20);
        curr_out = out + out_strides[9];
        STR_128_D(curr_out, v_out_stride, v_out9);

        v_t1 = _mm_mul_pd(v128_CRTM_15_1, _mm_sub_pd(v_s9, v_s3));
        v_t2 = _mm_mul_pd(v128_CRTM_15_9, _mm_add_pd(v_s4, v_s10));
        v_s21 = _mm_add_pd(v_s9, v_s3);
        v_s22 = _mm_sub_pd(v_s4, v_s10);
        v_s40 = _mm_add_pd(_mm_mul_pd(v128_CRTM_15_5, v_s21), v_s14);
        v_t4 = _mm_mul_pd(v128_CRTM_15_10, _mm_add_pd(v_s22, v_in2));

        // Output point 3: X(2)
        v_out5 = _mm_add_pd(v_s40, v_t4);
        curr_out = out + out_strides[5];
        STR_128_D(curr_out, v_out_stride, v_out5);

        // Output point 6: X(5)
        v_out10 = _mm_sub_pd(v_s40, v_t4);
        curr_out = out + out_strides[10];
        STR_128_D(curr_out, v_out_stride, v_out10);

        v_s23 = _mm_sub_pd(v_s14, _mm_mul_pd(v128_CRTM_15_2, v_s21));
        v_t5 = _mm_mul_pd(v128_CRTM_15_10,
                _mm_sub_pd( _mm_mul_pd(v128_CRTM_15_6, v_s22), v_in2));
        v_s24 = _mm_add_pd(v_s23, v_t5);
        v_s25 = _mm_sub_pd(v_s23, v_t5);
        v_s26 = _mm_sub_pd(v_t2, v_t1);
        v_s27 = _mm_add_pd(v_t1, v_t2);

        v_s28 = _mm_sub_pd(v_s24, v_s27);
        v_s29 = _mm_add_pd(_mm_mul_pd(v128_CRTM_15_3, v_s6),
                           _mm_mul_pd(v128_CRTM_15_4, v_s12));
        v_s30 = _mm_sub_pd(_mm_mul_pd(v128_CRTM_15_8, v_s1),
                           _mm_mul_pd(v128_CRTM_15_7, v_s7));
        v_s31 = _mm_sub_pd(v_s30, v_s29);
        v_s32 = _mm_add_pd(v_s29, v_s30);

        // Output point 1: X(0)
        v_out1 = _mm_add_pd(v_s28, v_s31);
        curr_out = out + out_strides[1];
        STR_128_D(curr_out, v_out_stride, v_out1);

        // Output point 3: X(2)
        v_out4 = _mm_sub_pd(v_s28, v_s31);
        curr_out = out + out_strides[4];
        STR_128_D(curr_out, v_out_stride, v_out4);

        v_s33 = _mm_add_pd(v_s24, v_s27);
        v_s34 = _mm_add_pd(_mm_mul_pd(v128_CRTM_15_8, v_s7),
                           _mm_mul_pd(v128_CRTM_15_7, v_s1));
        v_s35 = _mm_sub_pd(_mm_mul_pd(v128_CRTM_15_3, v_s12),
                           _mm_mul_pd(v128_CRTM_15_4, v_s6));
        v_s36 = _mm_add_pd(v_s35, v_s34);

        // Output point 4: X(3)
        v_out7 = _mm_add_pd(v_s33, v_s36);
        curr_out = out + out_strides[7];
        STR_128_D(curr_out, v_out_stride, v_out7);

        // Output point 7: X(6)
        v_out13 = _mm_sub_pd(v_s33, v_s36);
        curr_out = out + out_strides[13];
        STR_128_D(curr_out, v_out_stride, v_out13);

        v_s37 = _mm_sub_pd(v_s25, v_s26);
        v_s38 = _mm_sub_pd(v_s35, v_s34);

        // Output point 2: X(1)
        v_out2 = _mm_add_pd(v_s37, v_s38);
        curr_out = out + out_strides[2];
        STR_128_D(curr_out, v_out_stride, v_out2);

        // Output point 5: X(4)
        v_out8 = _mm_sub_pd(v_s37, v_s38);
        curr_out = out + out_strides[8];
        STR_128_D(curr_out, v_out_stride, v_out8);

        v_s39 = _mm_add_pd(v_s25, v_s26);

        // Output point 6: X(5)
        v_out11 = _mm_sub_pd(v_s39, v_s32);
        curr_out = out + out_strides[11];
        STR_128_D(curr_out, v_out_stride, v_out11);

        // Output point 8: X(7)
        v_out14 = _mm_add_pd(v_s39, v_s32);
        curr_out = out + out_strides[14];
        STR_128_D(curr_out, v_out_stride, v_out14);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        DOUBLE in0, in1, in2, in3, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10,
               s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23,
               s24, s25, s26, s27, s28, s29, s30, s31, s32, s33, s34, s35, s36,
               s37, s38, s39, t0, t1, t2, t3, t4, t5;

        in0 = *in;
        in1 = in[in_strides[3]];
        in2 = in[in_strides[13]];
        in3 = in[in_strides[5]];

        s0 = in1 + in2;
        s1 = in1 - in2;
        s2 = in3 + s0;
        s3 = in3 - (CRTM_15_2 * s0);

        in1 = in[in_strides[4]];
        in2 = in[in_strides[14]];
        in3 = in[in_strides[6]];

        s0 = in1 + in2;
        s4 = in1 - in2;
        s5 = s0 - in3;
        s6 = in3 + (CRTM_15_2 * s0);

        in1 = in[in_strides[1]];
        in2 = in[in_strides[7]];
        in3 = in[in_strides[11]];

        s0 = in1 + in2;
        s7 = in2 - in1;
        s8 = in3 + s0;
        s9 = in3 - (CRTM_15_2 * s0);

        in1 = in[in_strides[2]];
        in2 = in[in_strides[8]];
        in3 = in[in_strides[12]];

        s0 = in1 - in2;
        s10 = in2 + in1;
        s11 = in3 + s0;
        s12 = in3 - (CRTM_15_2 * s0);

        in1 = in[in_strides[9]];
        in2 = in[in_strides[10]];

        s13 = in0 + CRTM_15_5 * in1;
        s14 = in0 - in1;
        s15 = s8 + s2;
        t0 = CRTM_15_1 * (s8 - s2);
        s16 = s13 - (CRTM_15_2 * s15);

        *out = s13 + CRTM_15_5 * s15;

        s17 = (CRTM_15_3 * s11) + (CRTM_15_4 * s5);
        s18 = s16 + t0;

        out[out_strides[3]] = s18 - s17;
        out[out_strides[12]] = s18 + s17;

        s19 = s16 - t0;
        s20 = (CRTM_15_4 * s11) - (CRTM_15_3 * s5);

        out[out_strides[6]] = s19 - s20;
        out[out_strides[9]] = s19 + s20;

        t1 = CRTM_15_1 * (s9 - s3);
        t2 = CRTM_15_9 * (s10 + s4);
        s21 = (s9 + s3);
        s22 = s4 - s10;
        t3 = CRTM_15_5 * s21 + s14;
        t4 = CRTM_15_10 * (s22 + in2);

        out[out_strides[5]] = t3 + t4;
        out[out_strides[10]] = t3 - t4;

        s23 = s14 - (CRTM_15_2 * s21);
        t5 = CRTM_15_10 * ((CRTM_15_6 * s22) - in2);
        s24 = t5 + s23;
        s25 = s23 - t5;
        s26 = t2 - t1;
        s27 = t1 + t2;

        s28 = s24 - s27;
        s29 = (CRTM_15_4 * s12) + (CRTM_15_3 * s6);
        s30 = (CRTM_15_8 * s1) - (CRTM_15_7 * s7);
        s31 = s30 - s29;
        s32 = s29 + s30;

        out[out_strides[1]] = s28 + s31;
        out[out_strides[4]] = s28 - s31;

        s33 = s24 + s27;
        s34 = (CRTM_15_8 * s7) + (CRTM_15_7 * s1);
        s35 = (CRTM_15_3 * s12) - (CRTM_15_4 * s6);
        s36 = s35 + s34;

        out[out_strides[7]] = s33 + s36;
        out[out_strides[13]] = s33 - s36;

        s37 = s25 - s26;
        s38 = s35 - s34;
        out[out_strides[2]] = s37 + s38;
        out[out_strides[8]] = s37 - s38;

        s39 = s25 + s26;

        out[out_strides[11]] = s39 - s32;
        out[out_strides[14]] = s39 + s32;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

kfft_ register_kernel_r2hc_rfft15avx256(UINT8 precision, UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hc_rfft15avx256_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft15avx256_fp64_fwd;
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
            return r2hc_rfft15avx256_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft15avx256_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
