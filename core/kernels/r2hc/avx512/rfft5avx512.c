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

/** @file rfft5avx512.c
 *
 *  @brief Radix-5 r2hc Real-FFT kernel with AVX-512 operations using x86 SIMD
 *  intrinsics
 *
 *  This file contains the DIT radix-5 real-to-halfcomplex implementations using
 *  AVX512 SIMD operations for single-precision and double-precision inputs.
 *
 *  @author Jeya R
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/r2hc_simd_avx512.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                    {{{0, 6, 12, 128, 76, 30},
                                                      {0, 7, 12, 128, 92, 30}},
                                                     {{0, 6, 12, 64, 4, 30},
                                                      {0, 7, 12, 64, 4, 30}}};

ops_cycles_t get_ops_cnt_r2hc_rfft5avx512(UINT8 precision, UINT8 direction)
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

static VOID r2hc_rfft5avx512_fp32_fwd(VOID *in_real, VOID *in_imag,
                                      VOID *out_real, VOID *out_imag, INTP n,
                                      aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_5_1 = 0.559016994374947424102293417182819058860154590f;
    const FLOAT CRTM_5_2 = 0.951056516295153572116439333379382143405698632f;
    const FLOAT CRTM_5_3 = 0.587785252292473129168705954639072768597652438f;
    const FLOAT CRTM_5_4 = 0.250000000000000000000000000000000000000000000f;

    FLOAT *in = (FLOAT *)in_real;
    FLOAT *out = (FLOAT *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides  = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides  = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride  = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;

    INTP cnt;
    FLOAT *curr_in, *curr_out;
    INTP N = n / NUM_SETS_REAL_512_S;
    INTP remaining_sets = n % NUM_SETS_REAL_512_S;

    __m512 v_CRTM_5_1 = _mm512_set1_ps(CRTM_5_1);
    __m512 v_CRTM_5_2 = _mm512_set1_ps(CRTM_5_2);
    __m512 v_CRTM_5_3 = _mm512_set1_ps(CRTM_5_3);
    __m512 v_CRTM_5_4 = _mm512_set1_ps(CRTM_5_4);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m512 v_in0, v_in1, v_in2, v_in3, v_in4;
        __m512 v_s1, v_s2, v_s3, v_s4, v_s5, v_t1, v_s6;
        __m512 v_out0, v_out1, v_out2, v_out3, v_out4;

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

        v_s1 = _mm512_add_ps(v_in1, v_in4);
        v_s2 = _mm512_sub_ps(v_in4, v_in1);
        v_s3 = _mm512_add_ps(v_in2, v_in3);
        v_s4 = _mm512_sub_ps(v_in2, v_in3);
        v_s6 = _mm512_add_ps(v_s1, v_s3);

        v_s5 = _mm512_sub_ps(v_in0, _mm512_mul_ps(v_CRTM_5_4, v_s6));
        v_t1 = _mm512_mul_ps(v_CRTM_5_1, _mm512_sub_ps(v_s1, v_s3));

        // Output point 1: X(0)
        v_out0 = _mm512_add_ps(v_in0, v_s6);

        // Output point 2: X(1)
        v_out1 = _mm512_add_ps(v_s5, v_t1);

        // Output point 3: X(2)
        v_out2 = _mm512_sub_ps(_mm512_mul_ps(v_CRTM_5_2, v_s2),
                               _mm512_mul_ps(v_CRTM_5_3, v_s4));

        // Output point 4: X(3)
        v_out3 = _mm512_sub_ps(v_s5, v_t1);

        // Output point 5: X(4)
        v_out4 = _mm512_add_ps(_mm512_mul_ps(v_CRTM_5_2, v_s4),
                               _mm512_mul_ps(v_CRTM_5_3, v_s2));

        STR_512_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STRI_2x512_S(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out + out_strides[3];
        STRI_2x512_S(curr_out, v_out_stride, v_out3, v_out4);

        in = in + (v_in_stride << 4);
        out = out + (v_out_stride << 4);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_256_S)
    {
        __m256 v_in0, v_in1, v_in2, v_in3, v_in4;
        __m256 v_s1, v_s2, v_s3, v_s4, v_s5, v_t1, v_s6;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4;

        curr_in = in;
        curr_out = out;

        __m256 v256_CRTM_5_1 = _mm512_castps512_ps256(v_CRTM_5_1);
        __m256 v256_CRTM_5_2 = _mm512_castps512_ps256(v_CRTM_5_2);
        __m256 v256_CRTM_5_3 = _mm512_castps512_ps256(v_CRTM_5_3);
        __m256 v256_CRTM_5_4 = _mm512_castps512_ps256(v_CRTM_5_4);

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

        v_s1 = _mm256_add_ps(v_in1, v_in4);
        v_s2 = _mm256_sub_ps(v_in4, v_in1);
        v_s3 = _mm256_add_ps(v_in2, v_in3);
        v_s4 = _mm256_sub_ps(v_in2, v_in3);
        v_s6 = _mm256_add_ps(v_s1, v_s3);

        v_s5 = _mm256_sub_ps(v_in0, _mm256_mul_ps(v256_CRTM_5_4, v_s6));
        v_t1 = _mm256_mul_ps(v256_CRTM_5_1, _mm256_sub_ps(v_s1, v_s3));

        // Output point 1: X(0)
        v_out0 = _mm256_add_ps(v_in0, v_s6);

        // Output point 2: X(1)
        v_out1 = _mm256_add_ps(v_s5, v_t1);

        // Output point 3: X(2)
        v_out2 = _mm256_sub_ps(_mm256_mul_ps(v256_CRTM_5_2, v_s2),
                               _mm256_mul_ps(v256_CRTM_5_3, v_s4));

        // Output point 4: X(3)
        v_out3 = _mm256_sub_ps(v_s5, v_t1);

        // Output point 5: X(4)
        v_out4 = _mm256_add_ps(_mm256_mul_ps(v256_CRTM_5_2, v_s4),
                               _mm256_mul_ps(v256_CRTM_5_3, v_s2));

        STR_256_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STRI_2x256_S(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out + out_strides[3];
        STRI_2x256_S(curr_out, v_out_stride, v_out3, v_out4);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_S)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4;
        __m128 v_s1, v_s2, v_s3, v_s4, v_s5, v_t1, v_s6;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_5_1 = _mm512_castps512_ps128(v_CRTM_5_1);
        __m128 v128_CRTM_5_2 = _mm512_castps512_ps128(v_CRTM_5_2);
        __m128 v128_CRTM_5_3 = _mm512_castps512_ps128(v_CRTM_5_3);
        __m128 v128_CRTM_5_4 = _mm512_castps512_ps128(v_CRTM_5_4);

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

        v_s1 = _mm_add_ps(v_in1, v_in4);
        v_s2 = _mm_sub_ps(v_in4, v_in1);
        v_s3 = _mm_add_ps(v_in2, v_in3);
        v_s4 = _mm_sub_ps(v_in2, v_in3);
        v_s6 = _mm_add_ps(v_s1, v_s3);

        v_s5 = _mm_sub_ps(v_in0, _mm_mul_ps(v128_CRTM_5_4, v_s6));
        v_t1 = _mm_mul_ps(v128_CRTM_5_1, _mm_sub_ps(v_s1, v_s3));

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_in0, v_s6);

        // Output point 2: X(1)
        v_out1 = _mm_add_ps(v_s5, v_t1);

        // Output point 3: X(2)
        v_out2 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_5_2, v_s2),
                            _mm_mul_ps(v128_CRTM_5_3, v_s4));

        // Output point 4: X(3)
        v_out3 = _mm_sub_ps(v_s5, v_t1);

        // Output point 5: X(4)
        v_out4 = _mm_add_ps(_mm_mul_ps(v128_CRTM_5_2, v_s4),
                            _mm_mul_ps(v128_CRTM_5_3, v_s2));

        STR_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out + out_strides[3];
        STRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & 2)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4;
        __m128 v_s1, v_s2, v_s3, v_s4, v_s5, v_t1, v_s6;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_5_1 = _mm512_castps512_ps128(v_CRTM_5_1);
        __m128 v128_CRTM_5_2 = _mm512_castps512_ps128(v_CRTM_5_2);
        __m128 v128_CRTM_5_3 = _mm512_castps512_ps128(v_CRTM_5_3);
        __m128 v128_CRTM_5_4 = _mm512_castps512_ps128(v_CRTM_5_4);

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

        v_s1 = _mm_add_ps(v_in1, v_in4);
        v_s2 = _mm_sub_ps(v_in4, v_in1);
        v_s3 = _mm_add_ps(v_in2, v_in3);
        v_s4 = _mm_sub_ps(v_in2, v_in3);
        v_s6 = _mm_add_ps(v_s1, v_s3);

        v_s5 = _mm_sub_ps(v_in0, _mm_mul_ps(v128_CRTM_5_4, v_s6));
        v_t1 = _mm_mul_ps(v128_CRTM_5_1, _mm_sub_ps(v_s1, v_s3));

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_in0, v_s6);

        // Output point 2: X(1)
        v_out1 = _mm_add_ps(v_s5, v_t1);

        // Output point 3: X(2)
        v_out2 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_5_2, v_s2),
                            _mm_mul_ps(v128_CRTM_5_3, v_s4));

        // Output point 4: X(3)
        v_out3 = _mm_sub_ps(v_s5, v_t1);

        // Output point 5: X(4)
        v_out4 = _mm_add_ps(_mm_mul_ps(v128_CRTM_5_2, v_s4),
                            _mm_mul_ps(v128_CRTM_5_3, v_s2));

        STHR_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STHRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out + out_strides[3];
        STHRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        FLOAT in0, in1, in2, in3, in4;
        FLOAT s1, s2, s3, s4, s5, s6, t1;

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

        s1 = in1 + in4;
        s2 = in4 - in1;
        s3 = in2 + in3;
        s4 = in2 - in3;
        s6 = s1 + s3;

        s5 = in0 - (CRTM_5_4 * s6);
        t1 = CRTM_5_1 * (s1 - s3);

        // Output point 1: X(0)
        *out = in0 + s6;
        // Output point 2: X(1)
        out[out_strides[1]] = s5 + t1;
        // Output point 3: X(2)
        out[out_strides[2]] = (CRTM_5_2 * s2) - (CRTM_5_3 * s4);
        // Output point 4: X(3)
        out[out_strides[3]] = s5 - t1;
        // Output point 5: X(4)
        out[out_strides[4]] = (CRTM_5_2 * s4) + (CRTM_5_3 * s2);
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hc_rfft5avx512_fp32_bwd(VOID *in_real, VOID *in_imag,
                                      VOID *out_real, VOID *out_imag, INTP n,
                                      aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_5_1 = 1.11803398874989484820458683436563811772030918f;
    const FLOAT CRTM_5_2 = 1.90211303259030714423287866675876428681139726f;
    const FLOAT CRTM_5_3 = 1.17557050458494625833741190927814553719530488f;
    const FLOAT CRTM_5_4 = 0.50000000000000000000000000000000000000000000f;
    const FLOAT CRTM_5_5 = 2.00000000000000000000000000000000000000000000f;

    FLOAT *in = (FLOAT *)in_real;
    FLOAT *out = (FLOAT *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides  = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides  = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride  = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;

    INTP cnt;
    FLOAT *curr_in, *curr_out;
    INTP N = n / NUM_SETS_REAL_512_S;
    INTP remaining_sets = n % NUM_SETS_REAL_512_S;

    __m512 v_CRTM_5_1 = _mm512_set1_ps(CRTM_5_1);
    __m512 v_CRTM_5_2 = _mm512_set1_ps(CRTM_5_2);
    __m512 v_CRTM_5_3 = _mm512_set1_ps(CRTM_5_3);
    __m512 v_CRTM_5_4 = _mm512_set1_ps(CRTM_5_4);
    __m512 v_CRTM_5_5 = _mm512_set1_ps(CRTM_5_5);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m512 v_in0, v_in1, v_in2, v_in3, v_in4;
        __m512 v_s1, v_s2, v_s3, v_s4, v_s5, v_t1, v_s6;
        __m512 v_out0, v_out1, v_out2, v_out3, v_out4;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_512_S(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1) & x(2)
        curr_in = in + in_strides[1];
        LDRI_2x512_S(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3) & x(4)
        curr_in = in + in_strides[3];
        LDRI_2x512_S(curr_in, v_in_stride, v_in3, v_in4);

        v_s6 = _mm512_add_ps(v_in1, v_in3);
        v_t1 = _mm512_mul_ps(v_CRTM_5_1, _mm512_sub_ps(v_in1, v_in3));
        v_s5 = _mm512_sub_ps(v_in0, _mm512_mul_ps(v_CRTM_5_4, v_s6));
        v_s1 = _mm512_add_ps(v_s5, v_t1);
        v_s2 = _mm512_sub_ps(v_s5, v_t1);
        v_s3 = _mm512_add_ps(_mm512_mul_ps(v_CRTM_5_2, v_in2),
                             _mm512_mul_ps(v_CRTM_5_3, v_in4));
        v_s4 = _mm512_sub_ps(_mm512_mul_ps(v_CRTM_5_2, v_in4),
                             _mm512_mul_ps(v_CRTM_5_3, v_in2));

        // Output point 1: X(0)
        v_out0 = _mm512_add_ps(v_in0, _mm512_mul_ps(v_CRTM_5_5, v_s6));

        // Output point 2: X(1)
        v_out1 = _mm512_sub_ps(v_s1, v_s3);

        // Output point 3: X(2)
        v_out2 = _mm512_add_ps(v_s2, v_s4);

        // Output point 4: X(3)
        v_out3 = _mm512_sub_ps(v_s2, v_s4);

        // Output point 5: X(4)
        v_out4 = _mm512_add_ps(v_s1, v_s3);

        STR_512_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STR_512_S(curr_out, v_out_stride, v_out1);
        curr_out = out + out_strides[2];
        STR_512_S(curr_out, v_out_stride, v_out2);
        curr_out = out + out_strides[3];
        STR_512_S(curr_out, v_out_stride, v_out3);
        curr_out = out + out_strides[4];
        STR_512_S(curr_out, v_out_stride, v_out4);

        in = in + (v_in_stride << 4);
        out = out + (v_out_stride << 4);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_256_S)
    {
        __m256 v_in0, v_in1, v_in2, v_in3, v_in4;
        __m256 v_s1, v_s2, v_s3, v_s4, v_s5, v_t1, v_s6;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4;

        curr_in = in;
        curr_out = out;

        __m256 v256_CRTM_5_1 = _mm512_castps512_ps256(v_CRTM_5_1);
        __m256 v256_CRTM_5_2 = _mm512_castps512_ps256(v_CRTM_5_2);
        __m256 v256_CRTM_5_3 = _mm512_castps512_ps256(v_CRTM_5_3);
        __m256 v256_CRTM_5_4 = _mm512_castps512_ps256(v_CRTM_5_4);
        __m256 v256_CRTM_5_5 = _mm512_castps512_ps256(v_CRTM_5_5);

        // Input point 1: x(0)
        LDR_256_S(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1) & x(2)
        curr_in = in + in_strides[1];
        LDRI_2x256_S(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3) & x(4)
        curr_in = in + in_strides[3];
        LDRI_2x256_S(curr_in, v_in_stride, v_in3, v_in4);

        v_s6 = _mm256_add_ps(v_in1, v_in3);
        v_t1 = _mm256_mul_ps(v256_CRTM_5_1, _mm256_sub_ps(v_in1, v_in3));
        v_s5 = _mm256_sub_ps(v_in0, _mm256_mul_ps(v256_CRTM_5_4, v_s6));
        v_s1 = _mm256_add_ps(v_s5, v_t1);
        v_s2 = _mm256_sub_ps(v_s5, v_t1);
        v_s3 = _mm256_add_ps(_mm256_mul_ps(v256_CRTM_5_2, v_in2),
                             _mm256_mul_ps(v256_CRTM_5_3, v_in4));
        v_s4 = _mm256_sub_ps(_mm256_mul_ps(v256_CRTM_5_2, v_in4),
                             _mm256_mul_ps(v256_CRTM_5_3, v_in2));

        // Output point 1: X(0)
        v_out0 = _mm256_add_ps(v_in0, _mm256_mul_ps(v256_CRTM_5_5, v_s6));

        // Output point 2: X(1)
        v_out1 = _mm256_sub_ps(v_s1, v_s3);

        // Output point 3: X(2)
        v_out2 = _mm256_add_ps(v_s2, v_s4);

        // Output point 4: X(3)
        v_out3 = _mm256_sub_ps(v_s2, v_s4);

        // Output point 5: X(4)
        v_out4 = _mm256_add_ps(v_s1, v_s3);

        STR_256_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STR_256_S(curr_out, v_out_stride, v_out1);
        curr_out = out + out_strides[2];
        STR_256_S(curr_out, v_out_stride, v_out2);
        curr_out = out + out_strides[3];
        STR_256_S(curr_out, v_out_stride, v_out3);
        curr_out = out + out_strides[4];
        STR_256_S(curr_out, v_out_stride, v_out4);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_S)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4;
        __m128 v_s1, v_s2, v_s3, v_s4, v_s5, v_t1, v_s6;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_5_1 = _mm512_castps512_ps128(v_CRTM_5_1);
        __m128 v128_CRTM_5_2 = _mm512_castps512_ps128(v_CRTM_5_2);
        __m128 v128_CRTM_5_3 = _mm512_castps512_ps128(v_CRTM_5_3);
        __m128 v128_CRTM_5_4 = _mm512_castps512_ps128(v_CRTM_5_4);
        __m128 v128_CRTM_5_5 = _mm512_castps512_ps128(v_CRTM_5_5);

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1) & x(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_S(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3) & x(4)
        curr_in = in + in_strides[3];
        LDRI_2x128_S(curr_in, v_in_stride, v_in3, v_in4);

        v_s6 = _mm_add_ps(v_in1, v_in3);
        v_t1 = _mm_mul_ps(v128_CRTM_5_1, _mm_sub_ps(v_in1, v_in3));
        v_s5 = _mm_sub_ps(v_in0, _mm_mul_ps(v128_CRTM_5_4, v_s6));
        v_s1 = _mm_add_ps(v_s5, v_t1);
        v_s2 = _mm_sub_ps(v_s5, v_t1);
        v_s3 = _mm_add_ps(_mm_mul_ps(v128_CRTM_5_2, v_in2),
                          _mm_mul_ps(v128_CRTM_5_3, v_in4));
        v_s4 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_5_2, v_in4),
                          _mm_mul_ps(v128_CRTM_5_3, v_in2));

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_in0, _mm_mul_ps(v128_CRTM_5_5, v_s6));

        // Output point 2: X(1)
        v_out1 = _mm_sub_ps(v_s1, v_s3);

        // Output point 3: X(2)
        v_out2 = _mm_add_ps(v_s2, v_s4);

        // Output point 4: X(3)
        v_out3 = _mm_sub_ps(v_s2, v_s4);

        // Output point 5: X(4)
        v_out4 = _mm_add_ps(v_s1, v_s3);

        STR_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STR_128_S(curr_out, v_out_stride, v_out1);
        curr_out = out + out_strides[2];
        STR_128_S(curr_out, v_out_stride, v_out2);
        curr_out = out + out_strides[3];
        STR_128_S(curr_out, v_out_stride, v_out3);
        curr_out = out + out_strides[4];
        STR_128_S(curr_out, v_out_stride, v_out4);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & 2)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4;
        __m128 v_s1, v_s2, v_s3, v_s4, v_s5, v_t1, v_s6;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4;

        __m128 v128_CRTM_5_1 = _mm512_castps512_ps128(v_CRTM_5_1);
        __m128 v128_CRTM_5_2 = _mm512_castps512_ps128(v_CRTM_5_2);
        __m128 v128_CRTM_5_3 = _mm512_castps512_ps128(v_CRTM_5_3);
        __m128 v128_CRTM_5_4 = _mm512_castps512_ps128(v_CRTM_5_4);
        __m128 v128_CRTM_5_5 = _mm512_castps512_ps128(v_CRTM_5_5);

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDHR_128_S(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1) & x(2)
        curr_in = in + in_strides[1];
        LDHRI_2x128_S(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3) & x(4)
        curr_in = in + in_strides[3];
        LDHRI_2x128_S(curr_in, v_in_stride, v_in3, v_in4);

        v_s6 = _mm_add_ps(v_in1, v_in3);
        v_t1 = _mm_mul_ps(v128_CRTM_5_1, _mm_sub_ps(v_in1, v_in3));
        v_s5 = _mm_sub_ps(v_in0, _mm_mul_ps(v128_CRTM_5_4, v_s6));
        v_s1 = _mm_add_ps(v_s5, v_t1);
        v_s2 = _mm_sub_ps(v_s5, v_t1);
        v_s3 = _mm_add_ps(_mm_mul_ps(v128_CRTM_5_2, v_in2),
                          _mm_mul_ps(v128_CRTM_5_3, v_in4));
        v_s4 = _mm_sub_ps(_mm_mul_ps(v128_CRTM_5_2, v_in4),
                          _mm_mul_ps(v128_CRTM_5_3, v_in2));

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_in0, _mm_mul_ps(v128_CRTM_5_5, v_s6));

        // Output point 2: X(1)
        v_out1 = _mm_sub_ps(v_s1, v_s3);

        // Output point 3: X(2)
        v_out2 = _mm_add_ps(v_s2, v_s4);

        // Output point 4: X(3)
        v_out3 = _mm_sub_ps(v_s2, v_s4);

        // Output point 5: X(4)
        v_out4 = _mm_add_ps(v_s1, v_s3);

        STHR_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STHR_128_S(curr_out, v_out_stride, v_out1);
        curr_out = out + out_strides[2];
        STHR_128_S(curr_out, v_out_stride, v_out2);
        curr_out = out + out_strides[3];
        STHR_128_S(curr_out, v_out_stride, v_out3);
        curr_out = out + out_strides[4];
        STHR_128_S(curr_out, v_out_stride, v_out4);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        DOUBLE in0, in1, in2, in3, in4;
        DOUBLE s1, s2, s3, s4, s5, s6, t1;

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

        s6 = in1 + in3;
        t1 = CRTM_5_1 * (in1 - in3);
        s5 = in0 - (CRTM_5_4 * s6);
        s1 = s5 + t1;
        s2 = s5 - t1;
        s3 = (CRTM_5_2 * in2) + (CRTM_5_3 * in4);
        s4 = (CRTM_5_2 * in4) - (CRTM_5_3 * in2);

        // Output point 1: X(0)
        *out = in0 + CRTM_5_5 * s6;

        // Output point 2: X(1)
        out[out_strides[1]] = s1 - s3;

        // Output point 3: X(2)
        out[out_strides[2]] = s2 + s4;

        // Output point 4: X(3)
        out[out_strides[3]] = s2 - s4;

        // Output point 5: X(4)
        out[out_strides[4]] = s1 + s3;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hc_rfft5avx512_fp64_fwd(VOID *in_real, VOID *in_imag,
                                      VOID *out_real, VOID *out_imag, INTP n,
                                      aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_5_1 = 0.559016994374947424102293417182819058860154590;
    const DOUBLE CRTM_5_2 = 0.951056516295153572116439333379382143405698632;
    const DOUBLE CRTM_5_3 = 0.587785252292473129168705954639072768597652438;
    const DOUBLE CRTM_5_4 = 0.250000000000000000000000000000000000000000000;

    DOUBLE *in = (DOUBLE *)in_real;
    DOUBLE *out = (DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides  = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides  = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride  = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;

    INTP cnt;
    DOUBLE *curr_in, *curr_out;
    INTP N = n / NUM_SETS_REAL_512_D;
    INTP remaining_sets = n % NUM_SETS_REAL_512_D;

    __m512d v_CRTM_5_1 = _mm512_set1_pd(CRTM_5_1);
    __m512d v_CRTM_5_2 = _mm512_set1_pd(CRTM_5_2);
    __m512d v_CRTM_5_3 = _mm512_set1_pd(CRTM_5_3);
    __m512d v_CRTM_5_4 = _mm512_set1_pd(CRTM_5_4);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m512d v_in0, v_in1, v_in2, v_in3, v_in4;
        __m512d v_s1, v_s2, v_s3, v_s4, v_s5, v_t1, v_s6;
        __m512d v_out0, v_out1, v_out2, v_out3, v_out4;

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

        v_s1 = _mm512_add_pd(v_in1, v_in4);
        v_s2 = _mm512_sub_pd(v_in4, v_in1);
        v_s3 = _mm512_add_pd(v_in2, v_in3);
        v_s4 = _mm512_sub_pd(v_in2, v_in3);
        v_s6 = _mm512_add_pd(v_s1, v_s3);

        v_s5 = _mm512_sub_pd(v_in0, _mm512_mul_pd(v_CRTM_5_4, v_s6));
        v_t1 = _mm512_mul_pd(v_CRTM_5_1, _mm512_sub_pd(v_s1, v_s3));

        // Output point 1: X(0)
        v_out0 = _mm512_add_pd(v_in0, v_s6);

        // Output point 2: X(1)
        v_out1 = _mm512_add_pd(v_s5, v_t1);

        // Output point 3: X(2)
        v_out2 = _mm512_sub_pd(_mm512_mul_pd(v_CRTM_5_2, v_s2),
                               _mm512_mul_pd(v_CRTM_5_3, v_s4));

        // Output point 4: X(3)
        v_out3 = _mm512_sub_pd(v_s5, v_t1);

        // Output point 5: X(4)
        v_out4 = _mm512_add_pd(_mm512_mul_pd(v_CRTM_5_2, v_s4),
                               _mm512_mul_pd(v_CRTM_5_3, v_s2));

        STR_512_D(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STRI_2x512_D(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out + out_strides[3];
        STRI_2x512_D(curr_out, v_out_stride, v_out3, v_out4);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_256_D)
    {
        __m256d v_in0, v_in1, v_in2, v_in3, v_in4;
        __m256d v_s1, v_s2, v_s3, v_s4, v_s5, v_t1, v_s6;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4;

        curr_in = in;
        curr_out = out;

        __m256d v256_CRTM_5_1 = _mm512_castpd512_pd256(v_CRTM_5_1);
        __m256d v256_CRTM_5_2 = _mm512_castpd512_pd256(v_CRTM_5_2);
        __m256d v256_CRTM_5_3 = _mm512_castpd512_pd256(v_CRTM_5_3);
        __m256d v256_CRTM_5_4 = _mm512_castpd512_pd256(v_CRTM_5_4);

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

        v_s1 = _mm256_add_pd(v_in1, v_in4);
        v_s2 = _mm256_sub_pd(v_in4, v_in1);
        v_s3 = _mm256_add_pd(v_in2, v_in3);
        v_s4 = _mm256_sub_pd(v_in2, v_in3);
        v_s6 = _mm256_add_pd(v_s1, v_s3);

        v_s5 = _mm256_sub_pd(v_in0, _mm256_mul_pd(v256_CRTM_5_4, v_s6));
        v_t1 = _mm256_mul_pd(v256_CRTM_5_1, _mm256_sub_pd(v_s1, v_s3));

        // Output point 1: X(0)
        v_out0 = _mm256_add_pd(v_in0, v_s6);

        // Output point 2: X(1)
        v_out1 = _mm256_add_pd(v_s5, v_t1);

        // Output point 3: X(2)
        v_out2 = _mm256_sub_pd(_mm256_mul_pd(v256_CRTM_5_2, v_s2),
                               _mm256_mul_pd(v256_CRTM_5_3, v_s4));

        // Output point 4: X(3)
        v_out3 = _mm256_sub_pd(v_s5, v_t1);

        // Output point 5: X(4)
        v_out4 = _mm256_add_pd(_mm256_mul_pd(v256_CRTM_5_2, v_s4),
                               _mm256_mul_pd(v256_CRTM_5_3, v_s2));

        STR_256_D(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STRI_2x256_D(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out + out_strides[3];
        STRI_2x256_D(curr_out, v_out_stride, v_out3, v_out4);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & 2)
    {
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4;
        __m128d v_s1, v_s2, v_s3, v_s4, v_s5, v_t1, v_s6;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4;

        curr_in = in;
        curr_out = out;

        __m128d v128_CRTM_5_1 = _mm512_castpd512_pd128(v_CRTM_5_1);
        __m128d v128_CRTM_5_2 = _mm512_castpd512_pd128(v_CRTM_5_2);
        __m128d v128_CRTM_5_3 = _mm512_castpd512_pd128(v_CRTM_5_3);
        __m128d v128_CRTM_5_4 = _mm512_castpd512_pd128(v_CRTM_5_4);

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

        v_s1 = _mm_add_pd(v_in1, v_in4);
        v_s2 = _mm_sub_pd(v_in4, v_in1);
        v_s3 = _mm_add_pd(v_in2, v_in3);
        v_s4 = _mm_sub_pd(v_in2, v_in3);
        v_s6 = _mm_add_pd(v_s1, v_s3);

        v_s5 = _mm_sub_pd(v_in0, _mm_mul_pd(v128_CRTM_5_4, v_s6));
        v_t1 = _mm_mul_pd(v128_CRTM_5_1, _mm_sub_pd(v_s1, v_s3));

        // Output point 1: X(0)
        v_out0 = _mm_add_pd(v_in0, v_s6);

        // Output point 2: X(1)
        v_out1 = _mm_add_pd(v_s5, v_t1);

        // Output point 3: X(2)
        v_out2 = _mm_sub_pd(_mm_mul_pd(v128_CRTM_5_2, v_s2),
                            _mm_mul_pd(v128_CRTM_5_3, v_s4));

        // Output point 4: X(3)
        v_out3 = _mm_sub_pd(v_s5, v_t1);

        // Output point 5: X(4)
        v_out4 = _mm_add_pd(_mm_mul_pd(v128_CRTM_5_2, v_s4),
                            _mm_mul_pd(v128_CRTM_5_3, v_s2));

        STR_128_D(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STRI_2x128_D(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out + out_strides[3];
        STRI_2x128_D(curr_out, v_out_stride, v_out3, v_out4);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        DOUBLE in0, in1, in2, in3, in4;
        DOUBLE s1, s2, s3, s4, s5, s6, t1;

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

        s1 = in1 + in4;
        s2 = in4 - in1;
        s3 = in2 + in3;
        s4 = in2 - in3;
        s6 = s1 + s3;

        s5 = in0 - (CRTM_5_4 * s6);
        t1 = CRTM_5_1 * (s1 - s3);

        // Output point 1: X(0)
        *out = in0 + s6;
        // Output point 2: X(1)
        out[out_strides[1]] = s5 + t1;
        // Output point 3: X(2)
        out[out_strides[2]] = (CRTM_5_2 * s2) - (CRTM_5_3 * s4);
        // Output point 4: X(3)
        out[out_strides[3]] = s5 - t1;
        // Output point 5: X(4)
        out[out_strides[4]] = (CRTM_5_2 * s4) + (CRTM_5_3 * s2);
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hc_rfft5avx512_fp64_bwd(VOID *in_real, VOID *in_imag,
                                      VOID *out_real, VOID *out_imag, INTP n,
                                      aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_5_1 = 1.11803398874989484820458683436563811772030918;
    const DOUBLE CRTM_5_2 = 1.90211303259030714423287866675876428681139726;
    const DOUBLE CRTM_5_3 = 1.17557050458494625833741190927814553719530488;
    const DOUBLE CRTM_5_4 = 0.50000000000000000000000000000000000000000000;
    const DOUBLE CRTM_5_5 = 2.000000000000000000000000000000000000000000000;

    DOUBLE *in = (DOUBLE *)in_real;
    DOUBLE *out = (DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides  = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides  = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride  = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;

    INTP cnt;
    DOUBLE *curr_in, *curr_out;
    INTP N = n / NUM_SETS_REAL_512_D;
    INTP remaining_sets = n % NUM_SETS_REAL_512_D;

    __m512d v_CRTM_5_1 = _mm512_set1_pd(CRTM_5_1);
    __m512d v_CRTM_5_2 = _mm512_set1_pd(CRTM_5_2);
    __m512d v_CRTM_5_3 = _mm512_set1_pd(CRTM_5_3);
    __m512d v_CRTM_5_4 = _mm512_set1_pd(CRTM_5_4);
    __m512d v_CRTM_5_5 = _mm512_set1_pd(CRTM_5_5);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m512d v_in0, v_in1, v_in2, v_in3, v_in4;
        __m512d v_s1, v_s2, v_s3, v_s4, v_s5, v_t1, v_s6;
        __m512d v_out0, v_out1, v_out2, v_out3, v_out4;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_512_D(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1) & x(2)
        curr_in = in + in_strides[1];
        LDRI_2x512_D(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3) & x(4)
        curr_in = in + in_strides[3];
        LDRI_2x512_D(curr_in, v_in_stride, v_in3, v_in4);

        v_s6 = _mm512_add_pd(v_in1, v_in3);
        v_t1 = _mm512_mul_pd(v_CRTM_5_1, _mm512_sub_pd(v_in1, v_in3));
        v_s5 = _mm512_sub_pd(v_in0, _mm512_mul_pd(v_CRTM_5_4, v_s6));
        v_s1 = _mm512_add_pd(v_s5, v_t1);
        v_s2 = _mm512_sub_pd(v_s5, v_t1);
        v_s3 = _mm512_add_pd(_mm512_mul_pd(v_CRTM_5_2, v_in2),
                             _mm512_mul_pd(v_CRTM_5_3, v_in4));
        v_s4 = _mm512_sub_pd(_mm512_mul_pd(v_CRTM_5_2, v_in4),
                             _mm512_mul_pd(v_CRTM_5_3, v_in2));

        // Output point 1: X(0)
        v_out0 = _mm512_add_pd(v_in0, _mm512_mul_pd(v_CRTM_5_5, v_s6));

        // Output point 2: X(1)
        v_out1 = _mm512_sub_pd(v_s1, v_s3);

        // Output point 3: X(2)
        v_out2 = _mm512_add_pd(v_s2, v_s4);

        // Output point 4: X(3)
        v_out3 = _mm512_sub_pd(v_s2, v_s4);

        // Output point 5: X(4)
        v_out4 = _mm512_add_pd(v_s1, v_s3);

        STR_512_D(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STR_512_D(curr_out, v_out_stride, v_out1);
        curr_out = out + out_strides[2];
        STR_512_D(curr_out, v_out_stride, v_out2);
        curr_out = out + out_strides[3];
        STR_512_D(curr_out, v_out_stride, v_out3);
        curr_out = out + out_strides[4];
        STR_512_D(curr_out, v_out_stride, v_out4);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_256_D)
    {
        __m256d v_in0, v_in1, v_in2, v_in3, v_in4;
        __m256d v_s1, v_s2, v_s3, v_s4, v_s5, v_t1, v_s6;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4;

        curr_in = in;
        curr_out = out;

        __m256d v256_CRTM_5_1 = _mm512_castpd512_pd256(v_CRTM_5_1);
        __m256d v256_CRTM_5_2 = _mm512_castpd512_pd256(v_CRTM_5_2);
        __m256d v256_CRTM_5_3 = _mm512_castpd512_pd256(v_CRTM_5_3);
        __m256d v256_CRTM_5_4 = _mm512_castpd512_pd256(v_CRTM_5_4);
        __m256d v256_CRTM_5_5 = _mm512_castpd512_pd256(v_CRTM_5_5);

        // Input point 1: x(0)
        LDR_256_D(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1) & x(2)
        curr_in = in + in_strides[1];
        LDRI_2x256_D(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3) & x(4)
        curr_in = in + in_strides[3];
        LDRI_2x256_D(curr_in, v_in_stride, v_in3, v_in4);

        v_s6 = _mm256_add_pd(v_in1, v_in3);
        v_t1 = _mm256_mul_pd(v256_CRTM_5_1, _mm256_sub_pd(v_in1, v_in3));
        v_s5 = _mm256_sub_pd(v_in0, _mm256_mul_pd(v256_CRTM_5_4, v_s6));
        v_s1 = _mm256_add_pd(v_s5, v_t1);
        v_s2 = _mm256_sub_pd(v_s5, v_t1);
        v_s3 = _mm256_add_pd(_mm256_mul_pd(v256_CRTM_5_2, v_in2),
                             _mm256_mul_pd(v256_CRTM_5_3, v_in4));
        v_s4 = _mm256_sub_pd(_mm256_mul_pd(v256_CRTM_5_2, v_in4),
                             _mm256_mul_pd(v256_CRTM_5_3, v_in2));

        // Output point 1: X(0)
        v_out0 = _mm256_add_pd(v_in0, _mm256_mul_pd(v256_CRTM_5_5, v_s6));

        // Output point 2: X(1)
        v_out1 = _mm256_sub_pd(v_s1, v_s3);

        // Output point 3: X(2)
        v_out2 = _mm256_add_pd(v_s2, v_s4);

        // Output point 4: X(3)
        v_out3 = _mm256_sub_pd(v_s2, v_s4);

        // Output point 5: X(4)
        v_out4 = _mm256_add_pd(v_s1, v_s3);

        STR_256_D(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STR_256_D(curr_out, v_out_stride, v_out1);
        curr_out = out + out_strides[2];
        STR_256_D(curr_out, v_out_stride, v_out2);
        curr_out = out + out_strides[3];
        STR_256_D(curr_out, v_out_stride, v_out3);
        curr_out = out + out_strides[4];
        STR_256_D(curr_out, v_out_stride, v_out4);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & 2)
    {
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4;
        __m128d v_s1, v_s2, v_s3, v_s4, v_s5, v_t1, v_s6;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4;

        curr_in = in;
        curr_out = out;

        __m128d v128_CRTM_5_1 = _mm512_castpd512_pd128(v_CRTM_5_1);
        __m128d v128_CRTM_5_2 = _mm512_castpd512_pd128(v_CRTM_5_2);
        __m128d v128_CRTM_5_3 = _mm512_castpd512_pd128(v_CRTM_5_3);
        __m128d v128_CRTM_5_4 = _mm512_castpd512_pd128(v_CRTM_5_4);
        __m128d v128_CRTM_5_5 = _mm512_castpd512_pd128(v_CRTM_5_5);

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_stride, v_in0);
        // Input points : x(1), x(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_D(curr_in, v_in_stride, v_in1, v_in2);
        // Input points : x(3), x(4)
        curr_in = in + in_strides[3];
        LDRI_2x128_D(curr_in, v_in_stride, v_in3, v_in4);

        v_s6 = _mm_add_pd(v_in1, v_in3);
        v_t1 = _mm_mul_pd(v128_CRTM_5_1, _mm_sub_pd(v_in1, v_in3));
        v_s5 = _mm_sub_pd(v_in0, _mm_mul_pd(v128_CRTM_5_4, v_s6));
        v_s1 = _mm_add_pd(v_s5, v_t1);
        v_s2 = _mm_sub_pd(v_s5, v_t1);
        v_s3 = _mm_add_pd(_mm_mul_pd(v128_CRTM_5_2, v_in2),
                          _mm_mul_pd(v128_CRTM_5_3, v_in4));
        v_s4 = _mm_sub_pd(_mm_mul_pd(v128_CRTM_5_2, v_in4),
                          _mm_mul_pd(v128_CRTM_5_3, v_in2));

        // Output point 1: X(0)
        v_out0 = _mm_add_pd(v_in0, _mm_mul_pd(v128_CRTM_5_5, v_s6));

        // Output point 2: X(1)
        v_out1 = _mm_sub_pd(v_s1, v_s3);

        // Output point 3: X(2)
        v_out2 = _mm_add_pd(v_s2, v_s4);

        // Output point 4: X(3)
        v_out3 = _mm_sub_pd(v_s2, v_s4);

        // Output point 5: X(4)
        v_out4 = _mm_add_pd(v_s1, v_s3);

        STR_128_D(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STR_128_D(curr_out, v_out_stride, v_out1);
        curr_out = out + out_strides[2];
        STR_128_D(curr_out, v_out_stride, v_out2);
        curr_out = out + out_strides[3];
        STR_128_D(curr_out, v_out_stride, v_out3);
        curr_out = out + out_strides[4];
        STR_128_D(curr_out, v_out_stride, v_out4);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        DOUBLE in0, in1, in2, in3, in4;
        DOUBLE s1, s2, s3, s4, s5, s6, t1;

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

        s6 = in1 + in3;
        t1 = CRTM_5_1 * (in1 - in3);
        s5 = in0 - (CRTM_5_4 * s6);
        s1 = s5 + t1;
        s2 = s5 - t1;
        s3 = (CRTM_5_2 * in2) + (CRTM_5_3 * in4);
        s4 = (CRTM_5_2 * in4) - (CRTM_5_3 * in2);

        // Output point 1: X(0)
        *out = in0 + CRTM_5_5 * s6;

        // Output point 2: X(1)
        out[out_strides[1]] = s1 - s3;

        // Output point 3: X(2)
        out[out_strides[2]] = s2 + s4;

        // Output point 4: X(3)
        out[out_strides[3]] = s2 - s4;

        // Output point 5: X(4)
        out[out_strides[4]] = s1 + s3;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hc_rfft5avx512(UINT8 precision, UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hc_rfft5avx512_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft5avx512_fp64_fwd;
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
            return r2hc_rfft5avx512_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft5avx512_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
