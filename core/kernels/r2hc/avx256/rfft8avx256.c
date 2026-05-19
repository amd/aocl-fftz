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

/** @file rfft8avx256.c
 *
 *  @brief Radix-8 r2hc Real-FFT kernel with AVX-256 operations using x86 SIMD
 *  intrinsics
 *
 *  This file contains the DIT radix-8 real-to-halfcomplex implementations using
 *  AVX256 SIMD operations for single-precision and double-precision inputs.
 *
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/r2hc_simd_avx256.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                    {{{0, 2, 20, 104, 66, 16},
                                                      {0, 4, 22, 104, 75, 17}},
                                                     {{0, 2, 20, 52,   6, 16},
                                                      {0, 4, 22, 52,   6, 17}}};

ops_cycles_t get_ops_cnt_r2hc_rfft8avx256(UINT8 precision, UINT8 direction)
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

static VOID r2hc_rfft8avx256_fp32_fwd(VOID *in_real, VOID *in_imag,
                                      VOID *out_real, VOID *out_imag, INTP n,
                                      aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_8_1 = 0.7071067811865475244008443621048490392848359377f;

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

    __m256 v_CRTM_8_1 = _mm256_broadcast_ss(&CRTM_8_1);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m256 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7;
        __m256 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11;
        __m256 v_t0, v_t1;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7;

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

        v_s0 = _mm256_add_ps(v_in7, v_in5);
        v_s1 = _mm256_sub_ps(v_in7, v_in5);
        v_s2 = _mm256_add_ps(v_in6, v_in2);
        v_s3 = _mm256_sub_ps(v_in2, v_in6);
        v_s4 = _mm256_add_ps(v_in4, v_in0);
        v_s5 = _mm256_sub_ps(v_in0, v_in4);
        v_s6 = _mm256_add_ps(v_in3, v_in1);
        v_s7 = _mm256_sub_ps(v_in3, v_in1);

        v_s8 = _mm256_add_ps(v_s6, v_s0);
        v_s9 = _mm256_add_ps(v_s4, v_s2);
        v_s10 = _mm256_sub_ps(v_s1, v_s7);
        v_s11 = _mm256_sub_ps(v_s0, v_s6);

        v_t0 = _mm256_mul_ps(v_CRTM_8_1, v_s10);
        v_t1 = _mm256_mul_ps(v_CRTM_8_1, v_s11);

        // Output point 1: X(0)
        v_out0 = _mm256_add_ps(v_s9, v_s8);

        // Output point 2: X(1)
        v_out1 = _mm256_add_ps(v_s5, v_t0);

        // Output point 3: X(2)
        v_out2 = _mm256_sub_ps(v_t1, v_s3);

        // Output point 4: X(3)
        v_out3 = _mm256_sub_ps(v_s4, v_s2);

        // Output point 5: X(4)
        v_out4 = _mm256_add_ps(v_s7, v_s1);

        // Output point 6: X(5)
        v_out5 = _mm256_sub_ps(v_s5, v_t0);

        // Output point 7: X(6)
        v_out6 = _mm256_add_ps(v_s3, v_t1);

        // Output point 8: X(7)
        v_out7 = _mm256_sub_ps(v_s9, v_s8);

        STR_256_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STRI_2x256_S(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out + out_strides[3];
        STRI_2x256_S(curr_out, v_out_stride, v_out3, v_out4);
        curr_out = out + out_strides[5];
        STRI_2x256_S(curr_out, v_out_stride, v_out5, v_out6);
        curr_out = out + out_strides[7];
        STR_256_S(curr_out, v_out_stride, v_out7);

        in += v_in_stride * NUM_SETS_REAL_256_S;
        out += v_out_stride * NUM_SETS_REAL_256_S;
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_S)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11;
        __m128 v_t0, v_t1;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_8_1 = _mm256_castps256_ps128(v_CRTM_8_1);

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

        v_s0 = _mm_add_ps(v_in7, v_in5);
        v_s1 = _mm_sub_ps(v_in7, v_in5);
        v_s2 = _mm_add_ps(v_in6, v_in2);
        v_s3 = _mm_sub_ps(v_in2, v_in6);
        v_s4 = _mm_add_ps(v_in4, v_in0);
        v_s5 = _mm_sub_ps(v_in0, v_in4);
        v_s6 = _mm_add_ps(v_in3, v_in1);
        v_s7 = _mm_sub_ps(v_in3, v_in1);

        v_s8 = _mm_add_ps(v_s6, v_s0);
        v_s9 = _mm_add_ps(v_s4, v_s2);
        v_s10 = _mm_sub_ps(v_s1, v_s7);
        v_s11 = _mm_sub_ps(v_s0, v_s6);

        v_t0 = _mm_mul_ps(v128_CRTM_8_1, v_s10);
        v_t1 = _mm_mul_ps(v128_CRTM_8_1, v_s11);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_s9, v_s8);

        // Output point 2: X(1)
        v_out1 = _mm_add_ps(v_t0, v_s5);

        // Output point 3: X(2)
        v_out2 = _mm_sub_ps(v_t1, v_s3);

        // Output point 4: X(3)
        v_out3 = _mm_sub_ps(v_s4, v_s2);

        // Output point 5: X(4)
        v_out4 = _mm_add_ps(v_s7, v_s1);

        // Output point 6: X(5)
        v_out5 = _mm_sub_ps(v_s5, v_t0);

        // Output point 7: X(6)
        v_out6 = _mm_add_ps(v_s3, v_t1);

        // Output point 8: X(7)
        v_out7 = _mm_sub_ps(v_s9, v_s8);

        STR_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out + out_strides[3];
        STRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);
        curr_out = out + out_strides[5];
        STRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);
        curr_out = out + out_strides[7];
        STR_128_S(curr_out, v_out_stride, v_out7);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & 2)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
               v_s10, v_s11;
        __m128 v_t0, v_t1;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_8_1 = _mm256_castps256_ps128(v_CRTM_8_1);

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

        v_s0 = _mm_add_ps(v_in7, v_in5);
        v_s1 = _mm_sub_ps(v_in7, v_in5);
        v_s2 = _mm_add_ps(v_in6, v_in2);
        v_s3 = _mm_sub_ps(v_in2, v_in6);
        v_s4 = _mm_add_ps(v_in4, v_in0);
        v_s5 = _mm_sub_ps(v_in0, v_in4);
        v_s6 = _mm_add_ps(v_in3, v_in1);
        v_s7 = _mm_sub_ps(v_in3, v_in1);

        v_s8 = _mm_add_ps(v_s6, v_s0);
        v_s9 = _mm_add_ps(v_s4, v_s2);
        v_s10 = _mm_sub_ps(v_s1, v_s7);
        v_s11 = _mm_sub_ps(v_s0, v_s6);

        v_t0 = _mm_mul_ps(v128_CRTM_8_1, v_s10);
        v_t1 = _mm_mul_ps(v128_CRTM_8_1, v_s11);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_s9, v_s8);

        // Output point 2: X(1)
        v_out1 = _mm_add_ps(v_t0, v_s5);

        // Output point 3: X(2)
        v_out2 = _mm_sub_ps(v_t1, v_s3);

        // Output point 4: X(3)
        v_out3 = _mm_sub_ps(v_s4, v_s2);

        // Output point 5: X(4)
        v_out4 = _mm_add_ps(v_s7, v_s1);

        // Output point 6: X(5)
        v_out5 = _mm_sub_ps(v_s5, v_t0);

        // Output point 7: X(6)
        v_out6 = _mm_add_ps(v_s3, v_t1);

        // Output point 8: X(7)
        v_out7 = _mm_sub_ps(v_s9, v_s8);

        STHR_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STHRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out + out_strides[3];
        STHRI_2x128_S(curr_out, v_out_stride, v_out3, v_out4);
        curr_out = out + out_strides[5];
        STHRI_2x128_S(curr_out, v_out_stride, v_out5, v_out6);
        curr_out = out + out_strides[7];
        STHR_128_S(curr_out, v_out_stride, v_out7);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        FLOAT in0, in1, in2, in3, in4, in5, in6, in7;
        FLOAT s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11;
        FLOAT t0, t1;

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

        s0 = in7 + in5;
        s1 = in7 - in5;
        s2 = in6 + in2;
        s3 = in2 - in6;
        s4 = in4 + in0;
        s5 = in0 - in4;
        s6 = in3 + in1;
        s7 = in3 - in1;

        s8 = s6 + s0;
        s9 = s4 + s2;
        s10 = s1 - s7;
        s11 = s0 - s6;

        t0 = CRTM_8_1 * s10;
        t1 = CRTM_8_1 * s11;

        // Output point 1: X(0)
        *out = s9 + s8;

        // Output point 2: X(1)
        out[out_strides[1]] = s5 + t0;

        // Output point 3: X(2)
        out[out_strides[2]] = t1 - s3;

        // Output point 4: X(3)
        out[out_strides[3]] = s4 - s2;

        // Output point 5: X(4)
        out[out_strides[4]] = s7 + s1;

        // Output point 6: X(5)
        out[out_strides[5]] = s5 - t0;

        // Output point 7: X(6)
        out[out_strides[6]] = s3 + t1;

        // Output point 8: X(7)
        out[out_strides[7]] = s9 - s8;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hc_rfft8avx256_fp32_bwd(VOID *in_real, VOID *in_imag,
                                      VOID *out_real, VOID *out_imag, INTP n,
                                      aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_8_1 = 1.414213562373095048801688724209698078569671875f;
    const FLOAT CRTM_8_2 = 2.000000000000000000000000000000000000000000000f;

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

    __m256 v_CRTM_8_1 = _mm256_broadcast_ss(&CRTM_8_1);
    __m256 v_CRTM_8_2 = _mm256_broadcast_ss(&CRTM_8_2);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m256 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7;
        __m256 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9;
        __m256 v_t0, v_t1, v_t2, v_t3;
        __m256 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7;

        curr_in = in;
        curr_out = out;

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
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_256_S(curr_in, v_in_stride, v_in7);

        v_s0 = _mm256_add_ps(v_in7, v_in0);
        v_s1 = _mm256_sub_ps(v_in7, v_in0);
        v_s2 = _mm256_add_ps(v_in6, v_in2);
        v_t0 = _mm256_mul_ps(v_CRTM_8_2, _mm256_sub_ps(v_in2, v_in6));
        v_t1 = _mm256_mul_ps(v_CRTM_8_2, _mm256_add_ps(v_in5, v_in1));
        v_s3 = _mm256_sub_ps(v_in5, v_in1);
        v_s4 = _mm256_add_ps(v_in4, v_in4);
        v_s5 = _mm256_add_ps(v_in3, v_in3);

        v_t2 = _mm256_mul_ps(v_CRTM_8_1, _mm256_add_ps(v_s3, v_s2));
        v_t3 = _mm256_mul_ps(v_CRTM_8_1, _mm256_sub_ps(v_s3, v_s2));
        v_s6 = _mm256_add_ps(v_s5, v_s0);
        v_s7 = _mm256_sub_ps(v_s0, v_s5);
        v_s8 = _mm256_add_ps(v_s4, v_s1);
        v_s9 = _mm256_sub_ps(v_s4, v_s1);

        // Output point 1: X(0)
        v_out0 = _mm256_add_ps(v_s6, v_t1);

        // Output point 2: X(1)
        v_out1 = NEGATE_256_S(_mm256_add_ps(v_s8, v_t2));

        // Output point 3: X(2)
        v_out2 = _mm256_sub_ps(v_s7, v_t0);

        // Output point 4: X(3)
        v_out3 = _mm256_add_ps(v_s9, v_t3);

        // Output point 5: X(4)
        v_out4 = _mm256_sub_ps(v_s6, v_t1);

        // Output point 6: X(5)
        v_out5 = _mm256_sub_ps(v_t2, v_s8);

        // Output point 7: X(6)
        v_out6 = _mm256_add_ps(v_s7, v_t0);

        // Output point 8: X(7)
        v_out7 = _mm256_sub_ps(v_s9, v_t3);

        STR_256_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STR_256_S(curr_out, v_out_stride, v_out1);
        curr_out = out + out_strides[2];
        STR_256_S(curr_out, v_out_stride, v_out2);
        curr_out = out + out_strides[3];
        STR_256_S(curr_out, v_out_stride, v_out3);
        curr_out = out + out_strides[4];
        STR_256_S(curr_out, v_out_stride, v_out4);
        curr_out = out + out_strides[5];
        STR_256_S(curr_out, v_out_stride, v_out5);
        curr_out = out + out_strides[6];
        STR_256_S(curr_out, v_out_stride, v_out6);
        curr_out = out + out_strides[7];
        STR_256_S(curr_out, v_out_stride, v_out7);

        in += v_in_stride * NUM_SETS_REAL_256_S;
        out += v_out_stride * NUM_SETS_REAL_256_S;
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_S)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9;
        __m128 v_t0, v_t1, v_t2, v_t3;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_8_1 = _mm256_castps256_ps128(v_CRTM_8_1);
        __m128 v128_CRTM_8_2 = _mm256_castps256_ps128(v_CRTM_8_2);

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
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_128_S(curr_in, v_in_stride, v_in7);

        v_s0 = _mm_add_ps(v_in7, v_in0);
        v_s1 = _mm_sub_ps(v_in7, v_in0);
        v_s2 = _mm_add_ps(v_in6, v_in2);
        v_t0 = _mm_mul_ps(v128_CRTM_8_2, _mm_sub_ps(v_in2, v_in6));
        v_t1 = _mm_mul_ps(v128_CRTM_8_2, _mm_add_ps(v_in5, v_in1));
        v_s3 = _mm_sub_ps(v_in5, v_in1);
        v_s4 = _mm_add_ps(v_in4, v_in4);
        v_s5 = _mm_add_ps(v_in3, v_in3);

        v_t2 = _mm_mul_ps(v128_CRTM_8_1, _mm_add_ps(v_s3, v_s2));
        v_t3 = _mm_mul_ps(v128_CRTM_8_1, _mm_sub_ps(v_s3, v_s2));
        v_s6 = _mm_add_ps(v_s5, v_s0);
        v_s7 = _mm_sub_ps(v_s0, v_s5);
        v_s8 = _mm_add_ps(v_s4, v_s1);
        v_s9 = _mm_sub_ps(v_s4, v_s1);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_s6, v_t1);

        // Output point 2: X(1)
        v_out1 = NEGATE_128_S(_mm_add_ps(v_s8, v_t2));

        // Output point 3: X(2)
        v_out2 = _mm_sub_ps(v_s7, v_t0);

        // Output point 4: X(3)
        v_out3 = _mm_add_ps(v_s9, v_t3);

        // Output point 5: X(4)
        v_out4 = _mm_sub_ps(v_s6, v_t1);

        // Output point 6: X(5)
        v_out5 = _mm_sub_ps(v_t2, v_s8);

        // Output point 7: X(6)
        v_out6 = _mm_add_ps(v_s7, v_t0);

        // Output point 8: X(7)
        v_out7 = _mm_sub_ps(v_s9, v_t3);

        STR_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STR_128_S(curr_out, v_out_stride, v_out1);
        curr_out = out + out_strides[2];
        STR_128_S(curr_out, v_out_stride, v_out2);
        curr_out = out + out_strides[3];
        STR_128_S(curr_out, v_out_stride, v_out3);
        curr_out = out + out_strides[4];
        STR_128_S(curr_out, v_out_stride, v_out4);
        curr_out = out + out_strides[5];
        STR_128_S(curr_out, v_out_stride, v_out5);
        curr_out = out + out_strides[6];
        STR_128_S(curr_out, v_out_stride, v_out6);
        curr_out = out + out_strides[7];
        STR_128_S(curr_out, v_out_stride, v_out7);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & 2)
    {
        __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7;
        __m128 v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9;
        __m128 v_t0, v_t1, v_t2, v_t3;
        __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_8_1 = _mm256_castps256_ps128(v_CRTM_8_1);
        __m128 v128_CRTM_8_2 = _mm256_castps256_ps128(v_CRTM_8_2);

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
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDHR_128_S(curr_in, v_in_stride, v_in7);

        v_s0 = _mm_add_ps(v_in7, v_in0);
        v_s1 = _mm_sub_ps(v_in7, v_in0);
        v_s2 = _mm_add_ps(v_in6, v_in2);
        v_t0 = _mm_mul_ps(v128_CRTM_8_2, _mm_sub_ps(v_in2, v_in6));
        v_t1 = _mm_mul_ps(v128_CRTM_8_2, _mm_add_ps(v_in5, v_in1));
        v_s3 = _mm_sub_ps(v_in5, v_in1);
        v_s4 = _mm_add_ps(v_in4, v_in4);
        v_s5 = _mm_add_ps(v_in3, v_in3);

        v_t2 = _mm_mul_ps(v128_CRTM_8_1, _mm_add_ps(v_s3, v_s2));
        v_t3 = _mm_mul_ps(v128_CRTM_8_1, _mm_sub_ps(v_s3, v_s2));
        v_s6 = _mm_add_ps(v_s5, v_s0);
        v_s7 = _mm_sub_ps(v_s0, v_s5);
        v_s8 = _mm_add_ps(v_s4, v_s1);
        v_s9 = _mm_sub_ps(v_s4, v_s1);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_s6, v_t1);

        // Output point 2: X(1)
        v_out1 = NEGATE_128_S(_mm_add_ps(v_s8, v_t2));

        // Output point 3: X(2)
        v_out2 = _mm_sub_ps(v_s7, v_t0);

        // Output point 4: X(3)
        v_out3 = _mm_add_ps(v_s9, v_t3);

        // Output point 5: X(4)
        v_out4 = _mm_sub_ps(v_s6, v_t1);

        // Output point 6: X(5)
        v_out5 = _mm_sub_ps(v_t2, v_s8);

        // Output point 7: X(6)
        v_out6 = _mm_add_ps(v_s7, v_t0);

        // Output point 8: X(7)
        v_out7 = _mm_sub_ps(v_s9, v_t3);

        STHR_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STHR_128_S(curr_out, v_out_stride, v_out1);
        curr_out = out + out_strides[2];
        STHR_128_S(curr_out, v_out_stride, v_out2);
        curr_out = out + out_strides[3];
        STHR_128_S(curr_out, v_out_stride, v_out3);
        curr_out = out + out_strides[4];
        STHR_128_S(curr_out, v_out_stride, v_out4);
        curr_out = out + out_strides[5];
        STHR_128_S(curr_out, v_out_stride, v_out5);
        curr_out = out + out_strides[6];
        STHR_128_S(curr_out, v_out_stride, v_out6);
        curr_out = out + out_strides[7];
        STHR_128_S(curr_out, v_out_stride, v_out7);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        FLOAT in0, in1, in2, in3, in4, in5, in6, in7;
        FLOAT s0, s1, s2, s3, s4, s5, s6, s7, s8, s9;
        FLOAT t0, t1, t2, t3;

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

        s0 = in7 + in0;
        s1 = in7 - in0;
        s2 = in6 + in2;
        t0 = CRTM_8_2 * (in6 - in2);
        t1 = CRTM_8_2 * (in5 + in1);
        s3 = in5 - in1;
        s4 = in4 + in4;
        s5 = in3 + in3;

        t2 = CRTM_8_1 * (s3 + s2);
        t3 = CRTM_8_1 * (s3 - s2);
        s6 = s5 + s0;
        s7 = s5 - s0;
        s8 = s4 + s1;
        s9 = s4 - s1;

        // Output point 1: X(0)
        *out = s6 + t1;

        // Output point 2: X(1)
        out[out_strides[1]] = -s8 - t2;

        // Output point 3: X(2)
        out[out_strides[2]] = -s7 + t0;

        // Output point 4: X(3)
        out[out_strides[3]] = s9 + t3;

        // Output point 5: X(4)
        out[out_strides[4]] = s6 - t1;

        // Output point 6: X(5)
        out[out_strides[5]] = t2 - s8;

        // Output point 7: X(6)
        out[out_strides[6]] = -s7 - t0;

        // Output point 8: X(7)
        out[out_strides[7]] = s9 - t3;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hc_rfft8avx256_fp64_fwd(VOID *in_real, VOID *in_imag,
                                      VOID *out_real, VOID *out_imag, INTP n,
                                      aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_8_1 = 0.7071067811865475244008443621048490392848359377;

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

    __m256d v_CRTM_8_1 = _mm256_broadcast_sd(&CRTM_8_1);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m256d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7;
        __m256d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
                v_s10, v_s11;
        __m256d v_t0, v_t1;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7;

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

        v_s0 = _mm256_add_pd(v_in7, v_in5);
        v_s1 = _mm256_sub_pd(v_in7, v_in5);
        v_s2 = _mm256_add_pd(v_in6, v_in2);
        v_s3 = _mm256_sub_pd(v_in2, v_in6);
        v_s4 = _mm256_add_pd(v_in4, v_in0);
        v_s5 = _mm256_sub_pd(v_in0, v_in4);
        v_s6 = _mm256_add_pd(v_in3, v_in1);
        v_s7 = _mm256_sub_pd(v_in3, v_in1);

        v_s8 = _mm256_add_pd(v_s6, v_s0);
        v_s9 = _mm256_add_pd(v_s4, v_s2);
        v_s10 = _mm256_sub_pd(v_s1, v_s7);
        v_s11 = _mm256_sub_pd(v_s0, v_s6);

        v_t0 = _mm256_mul_pd(v_CRTM_8_1, v_s10);
        v_t1 = _mm256_mul_pd(v_CRTM_8_1, v_s11);

        // Output point 1: X(0)
        v_out0 = _mm256_add_pd(v_s9, v_s8);

        // Output point 2: X(1)
        v_out1 = _mm256_add_pd(v_t0, v_s5);

        // Output point 3: X(2)
        v_out2 = _mm256_sub_pd(v_t1, v_s3);

        // Output point 4: X(3)
        v_out3 = _mm256_sub_pd(v_s4, v_s2);

        // Output point 5: X(4)
        v_out4 = _mm256_add_pd(v_s7, v_s1);

        // Output point 6: X(5)
        v_out5 = _mm256_sub_pd(v_s5, v_t0);

        // Output point 7: X(6)
        v_out6 = _mm256_add_pd(v_s3, v_t1);

        // Output point 8: X(7)
        v_out7 = _mm256_sub_pd(v_s9, v_s8);

        STR_256_D(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STRI_2x256_D(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out + out_strides[3];
        STRI_2x256_D(curr_out, v_out_stride, v_out3, v_out4);
        curr_out = out + out_strides[5];
        STRI_2x256_D(curr_out, v_out_stride, v_out5, v_out6);
        curr_out = out + out_strides[7];
        STR_256_D(curr_out, v_out_stride, v_out7);

        in += v_in_stride * NUM_SETS_REAL_256_D;
        out += v_out_stride * NUM_SETS_REAL_256_D;
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_D)
    {
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7;
        __m128d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9,
                v_s10, v_s11;
        __m128d v_t0, v_t1;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7;

        curr_in = in;
        curr_out = out;

        __m128d v128_CRTM_8_1 = _mm256_castpd256_pd128(v_CRTM_8_1);

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

        v_s0 = _mm_add_pd(v_in7, v_in5);
        v_s1 = _mm_sub_pd(v_in7, v_in5);
        v_s2 = _mm_add_pd(v_in6, v_in2);
        v_s3 = _mm_sub_pd(v_in2, v_in6);
        v_s4 = _mm_add_pd(v_in4, v_in0);
        v_s5 = _mm_sub_pd(v_in0, v_in4);
        v_s6 = _mm_add_pd(v_in3, v_in1);
        v_s7 = _mm_sub_pd(v_in3, v_in1);

        v_s8 = _mm_add_pd(v_s6, v_s0);
        v_s9 = _mm_add_pd(v_s4, v_s2);
        v_s10 = _mm_sub_pd(v_s1, v_s7);
        v_s11 = _mm_sub_pd(v_s0, v_s6);

        v_t0 = _mm_mul_pd(v128_CRTM_8_1, v_s10);
        v_t1 = _mm_mul_pd(v128_CRTM_8_1, v_s11);

        // Output point 1: X(0)
        v_out0 = _mm_add_pd(v_s9, v_s8);

        // Output point 2: X(1)
        v_out1 = _mm_add_pd(v_t0, v_s5);

        // Output point 3: X(2)
        v_out2 = _mm_sub_pd(v_t1, v_s3);

        // Output point 4: X(3)
        v_out3 = _mm_sub_pd(v_s4, v_s2);

        // Output point 5: X(4)
        v_out4 = _mm_add_pd(v_s7, v_s1);

        // Output point 6: X(5)
        v_out5 = _mm_sub_pd(v_s5, v_t0);

        // Output point 7: X(6)
        v_out6 = _mm_add_pd(v_s3, v_t1);

        // Output point 8: X(7)
        v_out7 = _mm_sub_pd(v_s9, v_s8);

        STR_128_D(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STRI_2x128_D(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out + out_strides[3];
        STRI_2x128_D(curr_out, v_out_stride, v_out3, v_out4);
        curr_out = out + out_strides[5];
        STRI_2x128_D(curr_out, v_out_stride, v_out5, v_out6);
        curr_out = out + out_strides[7];
        STR_128_D(curr_out, v_out_stride, v_out7);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        DOUBLE in0, in1, in2, in3, in4, in5, in6, in7;
        DOUBLE s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11;
        DOUBLE t0, t1;

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

        s0 = in7 + in5;
        s1 = in7 - in5;
        s2 = in6 + in2;
        s3 = in2 - in6;
        s4 = in4 + in0;
        s5 = in0 - in4;
        s6 = in3 + in1;
        s7 = in3 - in1;

        s8 = s6 + s0;
        s9 = s4 + s2;
        s10 = s1 - s7;
        s11 = s0 - s6;

        t0 = CRTM_8_1 * s10;
        t1 = CRTM_8_1 * s11;

        // Output point 1: X(0)
        *out = s9 + s8;

        // Output point 2: X(1)
        out[out_strides[1]] = s5 + t0;

        // Output point 3: X(2)
        out[out_strides[2]] = t1 - s3;

        // Output point 4: X(3)
        out[out_strides[3]] = s4 - s2;

        // Output point 5: X(4)
        out[out_strides[4]] = s7 + s1;

        // Output point 6: X(5)
        out[out_strides[5]] = s5 - t0;

        // Output point 7: X(6)
        out[out_strides[6]] = s3 + t1;

        // Output point 8: X(7)
        out[out_strides[7]] = s9 - s8;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hc_rfft8avx256_fp64_bwd(VOID *in_real, VOID *in_imag,
                                      VOID *out_real, VOID *out_imag, INTP n,
                                      aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_8_1 = 1.414213562373095048801688724209698078569671875;
    const DOUBLE CRTM_8_2 = 2.000000000000000000000000000000000000000000000;

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

    __m256d v_CRTM_8_1 = _mm256_broadcast_sd(&CRTM_8_1);
    __m256d v_CRTM_8_2 = _mm256_broadcast_sd(&CRTM_8_2);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m256d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7;
        __m256d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9;
        __m256d v_t0, v_t1, v_t2, v_t3;
        __m256d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7;

        curr_in = in;
        curr_out = out;

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
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_256_D(curr_in, v_in_stride, v_in7);

        v_s0 = _mm256_add_pd(v_in7, v_in0);
        v_s1 = _mm256_sub_pd(v_in7, v_in0);
        v_s2 = _mm256_add_pd(v_in6, v_in2);
        v_t0 = _mm256_mul_pd(v_CRTM_8_2, _mm256_sub_pd(v_in2, v_in6));
        v_t1 = _mm256_mul_pd(v_CRTM_8_2, _mm256_add_pd(v_in5, v_in1));
        v_s3 = _mm256_sub_pd(v_in5, v_in1);
        v_s4 = _mm256_add_pd(v_in4, v_in4);
        v_s5 = _mm256_add_pd(v_in3, v_in3);

        v_t2 = _mm256_mul_pd(v_CRTM_8_1, _mm256_add_pd(v_s3, v_s2));
        v_t3 = _mm256_mul_pd(v_CRTM_8_1, _mm256_sub_pd(v_s3, v_s2));
        v_s6 = _mm256_add_pd(v_s5, v_s0);
        v_s7 = _mm256_sub_pd(v_s0, v_s5);
        v_s8 = _mm256_add_pd(v_s4, v_s1);
        v_s9 = _mm256_sub_pd(v_s4, v_s1);

        // Output point 1: X(0)
        v_out0 = _mm256_add_pd(v_s6, v_t1);

        // Output point 2: X(1)
        v_out1 = NEGATE_256_D(_mm256_add_pd(v_s8, v_t2));

        // Output point 3: X(2)
        v_out2 = _mm256_sub_pd(v_s7, v_t0);

        // Output point 4: X(3)
        v_out3 = _mm256_add_pd(v_s9, v_t3);

        // Output point 5: X(4)
        v_out4 = _mm256_sub_pd(v_s6, v_t1);

        // Output point 6: X(5)
        v_out5 = _mm256_sub_pd(v_t2, v_s8);

        // Output point 7: X(6)
        v_out6 = _mm256_add_pd(v_s7, v_t0);

        // Output point 8: X(7)
        v_out7 = _mm256_sub_pd(v_s9, v_t3);

        STR_256_D(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STR_256_D(curr_out, v_out_stride, v_out1);
        curr_out = out + out_strides[2];
        STR_256_D(curr_out, v_out_stride, v_out2);
        curr_out = out + out_strides[3];
        STR_256_D(curr_out, v_out_stride, v_out3);
        curr_out = out + out_strides[4];
        STR_256_D(curr_out, v_out_stride, v_out4);
        curr_out = out + out_strides[5];
        STR_256_D(curr_out, v_out_stride, v_out5);
        curr_out = out + out_strides[6];
        STR_256_D(curr_out, v_out_stride, v_out6);
        curr_out = out + out_strides[7];
        STR_256_D(curr_out, v_out_stride, v_out7);

        in += v_in_stride * NUM_SETS_REAL_256_D;
        out += v_out_stride * NUM_SETS_REAL_256_D;
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_D)
    {
        __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7;
        __m128d v_s0, v_s1, v_s2, v_s3, v_s4, v_s5, v_s6, v_s7, v_s8, v_s9;
        __m128d v_t0, v_t1, v_t2, v_t3;
        __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7;

        curr_in = in;
        curr_out = out;

        __m128d v128_CRTM_8_1 = _mm256_castpd256_pd128(v_CRTM_8_1);
        __m128d v128_CRTM_8_2 = _mm256_castpd256_pd128(v_CRTM_8_2);

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
        // Input point 8: x(7)
        curr_in = in + in_strides[7];
        LDR_128_D(curr_in, v_in_stride, v_in7);

        v_s0 = _mm_add_pd(v_in7, v_in0);
        v_s1 = _mm_sub_pd(v_in7, v_in0);
        v_s2 = _mm_add_pd(v_in6, v_in2);
        v_t0 = _mm_mul_pd(v128_CRTM_8_2, _mm_sub_pd(v_in2, v_in6));
        v_t1 = _mm_mul_pd(v128_CRTM_8_2, _mm_add_pd(v_in5, v_in1));
        v_s3 = _mm_sub_pd(v_in5, v_in1);
        v_s4 = _mm_add_pd(v_in4, v_in4);
        v_s5 = _mm_add_pd(v_in3, v_in3);

        v_t2 = _mm_mul_pd(v128_CRTM_8_1, _mm_add_pd(v_s3, v_s2));
        v_t3 = _mm_mul_pd(v128_CRTM_8_1, _mm_sub_pd(v_s3, v_s2));
        v_s6 = _mm_add_pd(v_s5, v_s0);
        v_s7 = _mm_sub_pd(v_s0, v_s5);
        v_s8 = _mm_add_pd(v_s4, v_s1);
        v_s9 = _mm_sub_pd(v_s4, v_s1);

        // Output point 1: X(0)
        v_out0 = _mm_add_pd(v_s6, v_t1);

        // Output point 2: X(1)
        v_out1 = NEGATE_128_D(_mm_add_pd(v_s8, v_t2));

        // Output point 3: X(2)
        v_out2 = _mm_sub_pd(v_s7, v_t0);

        // Output point 4: X(3)
        v_out3 = _mm_add_pd(v_s9, v_t3);

        // Output point 5: X(4)
        v_out4 = _mm_sub_pd(v_s6, v_t1);

        // Output point 6: X(5)
        v_out5 = _mm_sub_pd(v_t2, v_s8);

        // Output point 7: X(6)
        v_out6 = _mm_add_pd(v_s7, v_t0);

        // Output point 8: X(7)
        v_out7 = _mm_sub_pd(v_s9, v_t3);

        STR_128_D(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STR_128_D(curr_out, v_out_stride, v_out1);
        curr_out = out + out_strides[2];
        STR_128_D(curr_out, v_out_stride, v_out2);
        curr_out = out + out_strides[3];
        STR_128_D(curr_out, v_out_stride, v_out3);
        curr_out = out + out_strides[4];
        STR_128_D(curr_out, v_out_stride, v_out4);
        curr_out = out + out_strides[5];
        STR_128_D(curr_out, v_out_stride, v_out5);
        curr_out = out + out_strides[6];
        STR_128_D(curr_out, v_out_stride, v_out6);
        curr_out = out + out_strides[7];
        STR_128_D(curr_out, v_out_stride, v_out7);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        DOUBLE in0, in1, in2, in3, in4, in5, in6, in7;
        DOUBLE s0, s1, s2, s3, s4, s5, s6, s7, s8, s9;
        DOUBLE t0, t1, t2, t3;

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

        s0 = in7 + in0;
        s1 = in7 - in0;
        s2 = in6 + in2;
        t0 = CRTM_8_2 * (in6 - in2);
        t1 = CRTM_8_2 * (in5 + in1);
        s3 = in5 - in1;
        s4 = in4 + in4;
        s5 = in3 + in3;

        t2 = CRTM_8_1 * (s3 + s2);
        t3 = CRTM_8_1 * (s3 - s2);
        s6 = s5 + s0;
        s7 = s5 - s0;
        s8 = s4 + s1;
        s9 = s4 - s1;

        // Output point 1: X(0)
        *out = s6 + t1;

        // Output point 2: X(1)
        out[out_strides[1]] = -s8 - t2;

        // Output point 3: X(2)
        out[out_strides[2]] = -s7 + t0;

        // Output point 4: X(3)
        out[out_strides[3]] = s9 + t3;

        // Output point 5: X(4)
        out[out_strides[4]] = s6 - t1;

        // Output point 6: X(5)
        out[out_strides[5]] = t2 - s8;

        // Output point 7: X(6)
        out[out_strides[6]] = -s7 - t0;

        // Output point 8: X(7)
        out[out_strides[7]] = s9 - t3;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hc_rfft8avx256(UINT8 precision, UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hc_rfft8avx256_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft8avx256_fp64_fwd;
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
            return r2hc_rfft8avx256_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft8avx256_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
