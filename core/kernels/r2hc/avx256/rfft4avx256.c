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

/** @file rfft4avx256.c
 *
 *  @brief Radix-4 r2hc Real-FFT kernel with AVX-256 operations using x86 SIMD
 *  intrinsics
 *
 *  This file contains the DIT radix-4 real-to-halfcomplex implementations using
 *  AVX256 SIMD operations for single-precision and double-precision inputs.
 *
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/r2hc_simd_avx256.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                    {{{0, 0, 6, 56, 38, 8},
                                                      {0, 2, 6, 56, 41, 8}},
                                                     {{0, 0, 6, 28,  2, 8},
                                                      {0, 2, 6, 28,  2, 8}}};

ops_cycles_t get_ops_cnt_r2hc_rfft4avx256(UINT8 precision, UINT8 direction)
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

static VOID r2hc_rfft4avx256_fp32_fwd(VOID *in_real, VOID *in_imag,
                                      VOID *out_real, VOID *out_imag, INTP n,
                                      aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif

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

    for (cnt = 0; cnt < N; cnt++)
    {
        __m256 v_in0, v_in1, v_in2, v_in3;
        __m256 v_s0, v_s1;
        __m256 v_out0, v_out1, v_out2, v_out3;

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

        v_s0 = _mm256_add_ps(v_in0, v_in2);
        v_s1 = _mm256_add_ps(v_in1, v_in3);

        // Output point 1: X(0)
        v_out0 = _mm256_add_ps(v_s0, v_s1);

        // Output point 2: X(1)
        v_out1 = _mm256_sub_ps(v_in0, v_in2);

        // Output point 3: X(2)
        v_out2 = _mm256_sub_ps(v_in3, v_in1);

        // Output point 4: X(3)
        v_out3 = _mm256_sub_ps(v_s0, v_s1);

        STR_256_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STRI_2x256_S(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out + out_strides[3];
        STR_256_S(curr_out, v_out_stride, v_out3);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (n & 4)
    {
        __m128 v_in0, v_in1, v_in2, v_in3;
        __m128 v_s0, v_s1;
        __m128 v_out0, v_out1, v_out2, v_out3;

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

        v_s0 = _mm_add_ps(v_in0, v_in2);
        v_s1 = _mm_add_ps(v_in1, v_in3);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_s0, v_s1);

        // Output point 2: X(1)
        v_out1 = _mm_sub_ps(v_in0, v_in2);

        // Output point 3: X(2)
        v_out2 = _mm_sub_ps(v_in3, v_in1);

        // Output point 4: X(3)
        v_out3 = _mm_sub_ps(v_s0, v_s1);

        STR_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out + out_strides[3];
        STR_128_S(curr_out, v_out_stride, v_out3);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (n & 2)
    {
        __m128 v_in0, v_in1, v_in2, v_in3;
        __m128 v_s0, v_s1;
        __m128 v_out0, v_out1, v_out2, v_out3;

        curr_in = in;
        curr_out = out;

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

        v_s0 = _mm_add_ps(v_in0, v_in2);
        v_s1 = _mm_add_ps(v_in1, v_in3);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_s0, v_s1);

        // Output point 2: X(1)
        v_out1 = _mm_sub_ps(v_in0, v_in2);

        // Output point 3: X(2)
        v_out2 = _mm_sub_ps(v_in3, v_in1);

        // Output point 4: X(3)
        v_out3 = _mm_sub_ps(v_s0, v_s1);

        STHR_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STHRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out + out_strides[3];
        STHR_128_S(curr_out, v_out_stride, v_out3);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        FLOAT in0, in1, in2, in3;
        FLOAT s0, s1;

        // Input point 1: x(0)
        in0 = *in;
        // Input point 2: x(1)
        in1 = in[in_strides[1]];
        // Input point 3: x(2)
        in2 = in[in_strides[2]];
        // Input point 4: x(3)
        in3 = in[in_strides[3]];

        s0 = in0 + in2;
        s1 = in1 + in3;

        // Output point 1: X(0)
        *out = s0 + s1;

        // Output point 2: X(1)
        out[out_strides[1]] = in0 - in2;

        // Output point 3: X(2)
        out[out_strides[2]] = in3 - in1;

        // Output point 4: X(3)
        out[out_strides[3]] = s0 - s1;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

VOID r2hc_rfft4avx256_fp32_bwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                               VOID *out_imag, INTP n,
                               aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const FLOAT CRTM_4_1 = 2.000000000000000000000000000000000000000000000f;

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
    INTP N = n >> 3;
    FLOAT *curr_in, *curr_out;

    __m256 v_CRTM_4_1 = _mm256_broadcast_ss(&CRTM_4_1);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m256 v_in0, v_in1, v_in2, v_in3;
        __m256 v_s0, v_s1;
        __m256 v_t0, v_t1;
        __m256 v_out0, v_out1, v_out2, v_out3;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_256_S(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x256_S(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_256_S(curr_in, v_in_stride, v_in3);

        v_s0 = _mm256_add_ps(v_in0, v_in3);
        v_t0 = _mm256_mul_ps(v_CRTM_4_1, v_in1);
        v_s1 = _mm256_sub_ps(v_in0, v_in3);
        v_t1 = _mm256_mul_ps(v_CRTM_4_1, v_in2);

        // Output point 1: X(0)
        v_out0 = _mm256_add_ps(v_s0, v_t0);

        // Output point 2: X(1)
        v_out1 = _mm256_sub_ps(v_s1, v_t1);

        // Output point 3: X(2)
        v_out2 = _mm256_sub_ps(v_s0, v_t0);

        // Output point 4: X(3)
        v_out3 = _mm256_add_ps(v_s1, v_t1);

        STR_256_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STR_256_S(curr_out, v_out_stride, v_out1);
        curr_out = out + out_strides[2];
        STR_256_S(curr_out, v_out_stride, v_out2);
        curr_out = out + out_strides[3];
        STR_256_S(curr_out, v_out_stride, v_out3);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (n & 4)
    {
        __m128 v_in0, v_in1, v_in2, v_in3;
        __m128 v_s0, v_s1;
        __m128 v_t0, v_t1;
        __m128 v_out0, v_out1, v_out2, v_out3;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_4_1 = _mm256_castps256_ps128(v_CRTM_4_1);

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_S(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_128_S(curr_in, v_in_stride, v_in3);

        v_s0 = _mm_add_ps(v_in0, v_in3);
        v_t0 = _mm_mul_ps(v128_CRTM_4_1, v_in1);
        v_s1 = _mm_sub_ps(v_in0, v_in3);
        v_t1 = _mm_mul_ps(v128_CRTM_4_1, v_in2);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_s0, v_t0);

        // Output point 2: X(1)
        v_out1 = _mm_sub_ps(v_s1, v_t1);

        // Output point 3: X(2)
        v_out2 = _mm_sub_ps(v_s0, v_t0);

        // Output point 4: X(3)
        v_out3 = _mm_add_ps(v_s1, v_t1);

        STR_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STR_128_S(curr_out, v_out_stride, v_out1);
        curr_out = out + out_strides[2];
        STR_128_S(curr_out, v_out_stride, v_out2);
        curr_out = out + out_strides[3];
        STR_128_S(curr_out, v_out_stride, v_out3);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (n & 2)
    {
        __m128 v_in0, v_in1, v_in2, v_in3;
        __m128 v_s0, v_s1;
        __m128 v_t0, v_t1;
        __m128 v_out0, v_out1, v_out2, v_out3;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_4_1 = _mm256_castps256_ps128(v_CRTM_4_1);

        // Input point 1: x(0)
        LDHR_128_S(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDHRI_2x128_S(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDHR_128_S(curr_in, v_in_stride, v_in3);

        v_s0 = _mm_add_ps(v_in0, v_in3);
        v_t0 = _mm_mul_ps(v128_CRTM_4_1, v_in1);
        v_s1 = _mm_sub_ps(v_in0, v_in3);
        v_t1 = _mm_mul_ps(v128_CRTM_4_1, v_in2);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_s0, v_t0);

        // Output point 2: X(1)
        v_out1 = _mm_sub_ps(v_s1, v_t1);

        // Output point 3: X(2)
        v_out2 = _mm_sub_ps(v_s0, v_t0);

        // Output point 4: X(3)
        v_out3 = _mm_add_ps(v_s1, v_t1);

        STHR_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STHR_128_S(curr_out, v_out_stride, v_out1);
        curr_out = out + out_strides[2];
        STHR_128_S(curr_out, v_out_stride, v_out2);
        curr_out = out + out_strides[3];
        STHR_128_S(curr_out, v_out_stride, v_out3);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        FLOAT in0, in1, in2, in3;
        FLOAT s0, s1;
        FLOAT t0, t1;

        // Input point 1: x(0)
        in0 = *in;
        // Input point 2: x(1)
        in1 = in[in_strides[1]];
        // Input point 3: x(2)
        in2 = in[in_strides[2]];
        // Input point 4: x(3)
        in3 = in[in_strides[3]];

        s0 = in0 + in3;
        t0 = CRTM_4_1 * in1;
        s1 = in0 - in3;
        t1 = CRTM_4_1 * in2;

        // Output point 1: X(0)
        *out = s0 + t0;

        // Output point 2: X(1)
        out[out_strides[1]] = s1 - t1;

        // Output point 3: X(2)
        out[out_strides[2]] = s0 - t0;

        // Output point 4: X(3)
        out[out_strides[3]] = s1 + t1;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID r2hc_rfft4avx256_fp64_fwd(VOID *in_real, VOID *in_imag,
                                      VOID *out_real, VOID *out_imag, INTP n,
                                      aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif

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

    for (cnt = 0; cnt < N; cnt++)
    {
        __m256d v_in0, v_in1, v_in2, v_in3;
        __m256d v_s0, v_s1;
        __m256d v_out0, v_out1, v_out2, v_out3;

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

        v_s0 = _mm256_add_pd(v_in0, v_in2);
        v_s1 = _mm256_add_pd(v_in1, v_in3);

        // Output point 1: X(0)
        v_out0 = _mm256_add_pd(v_s0, v_s1);

        // Output point 2: X(1)
        v_out1 = _mm256_sub_pd(v_in0, v_in2);

        // Output point 3: X(2)
        v_out2 = _mm256_sub_pd(v_in3, v_in1);

        // Output point 4: X(3)
        v_out3 = _mm256_sub_pd(v_s0, v_s1);

        STR_256_D(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STRI_2x256_D(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out + out_strides[3];
        STR_256_D(curr_out, v_out_stride, v_out3);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (n & 2)
    {
        __m128d v_in0, v_in1, v_in2, v_in3;
        __m128d v_s0, v_s1;
        __m128d v_out0, v_out1, v_out2, v_out3;

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

        v_s0 = _mm_add_pd(v_in0, v_in2);
        v_s1 = _mm_add_pd(v_in1, v_in3);

        // Output point 1: X(0)
        v_out0 = _mm_add_pd(v_s0, v_s1);

        // Output point 2: X(1)
        v_out1 = _mm_sub_pd(v_in0, v_in2);

        // Output point 3: X(2)
        v_out2 = _mm_sub_pd(v_in3, v_in1);

        // Output point 4: X(3)
        v_out3 = _mm_sub_pd(v_s0, v_s1);

        STR_128_D(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STRI_2x128_D(curr_out, v_out_stride, v_out1, v_out2);
        curr_out = out + out_strides[3];
        STR_128_D(curr_out, v_out_stride, v_out3);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        DOUBLE in0, in1, in2, in3;
        DOUBLE s0, s1;

        // Input point 1: x(0)
        in0 = *in;
        // Input point 2: x(1)
        in1 = in[in_strides[1]];
        // Input point 3: x(2)
        in2 = in[in_strides[2]];
        // Input point 4: x(3)
        in3 = in[in_strides[3]];

        s0 = in0 + in2;
        s1 = in1 + in3;

        // Output point 1: X(0)
        *out = s0 + s1;

        // Output point 2: X(1)
        out[out_strides[1]] = in0 - in2;

        // Output point 3: X(2)
        out[out_strides[2]] = in3 - in1;

        // Output point 4: X(3)
        out[out_strides[3]] = s0 - s1;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID r2hc_rfft4avx256_fp64_bwd(VOID *in_real, VOID *in_imag,
                                      VOID *out_real, VOID *out_imag, INTP n,
                                      aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    const DOUBLE CRTM_4_1 = 2.000000000000000000000000000000000000000000000;

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

    __m256d v_CRTM_4_1 = _mm256_broadcast_sd(&CRTM_4_1);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m256d v_in0, v_in1, v_in2, v_in3;
        __m256d v_s0, v_s1;
        __m256d v_t0, v_t1;
        __m256d v_out0, v_out1, v_out2, v_out3;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_256_D(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x256_D(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_256_D(curr_in, v_in_stride, v_in3);

        v_s0 = _mm256_add_pd(v_in0, v_in3);
        v_t0 = _mm256_mul_pd(v_CRTM_4_1, v_in1);
        v_s1 = _mm256_sub_pd(v_in0, v_in3);
        v_t1 = _mm256_mul_pd(v_CRTM_4_1, v_in2);

        // Output point 1: X(0)
        v_out0 = _mm256_add_pd(v_s0, v_t0);

        // Output point 2: X(1)
        v_out1 = _mm256_sub_pd(v_s1, v_t1);

        // Output point 3: X(2)
        v_out2 = _mm256_sub_pd(v_s0, v_t0);

        // Output point 4: X(3)
        v_out3 = _mm256_add_pd(v_s1, v_t1);

        STR_256_D(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STR_256_D(curr_out, v_out_stride, v_out1);
        curr_out = out + out_strides[2];
        STR_256_D(curr_out, v_out_stride, v_out2);
        curr_out = out + out_strides[3];
        STR_256_D(curr_out, v_out_stride, v_out3);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (n & 2)
    {
        __m128d v_in0, v_in1, v_in2, v_in3;
        __m128d v_s0, v_s1;
        __m128d v_t0, v_t1;
        __m128d v_out0, v_out1, v_out2, v_out3;

        curr_in = in;
        curr_out = out;

        __m128d v128_CRTM_4_1 = _mm256_castpd256_pd128(v_CRTM_4_1);

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_D(curr_in, v_in_stride, v_in1, v_in2);
        // Input point 4: x(3)
        curr_in = in + in_strides[3];
        LDR_128_D(curr_in, v_in_stride, v_in3);

        v_s0 = _mm_add_pd(v_in0, v_in3);
        v_t0 = _mm_mul_pd(v128_CRTM_4_1, v_in1);
        v_s1 = _mm_sub_pd(v_in0, v_in3);
        v_t1 = _mm_mul_pd(v128_CRTM_4_1, v_in2);

        // Output point 1: X(0)
        v_out0 = _mm_add_pd(v_s0, v_t0);

        // Output point 2: X(1)
        v_out1 = _mm_sub_pd(v_s1, v_t1);

        // Output point 3: X(2)
        v_out2 = _mm_sub_pd(v_s0, v_t0);

        // Output point 4: X(3)
        v_out3 = _mm_add_pd(v_s1, v_t1);

        STR_128_D(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STR_128_D(curr_out, v_out_stride, v_out1);
        curr_out = out + out_strides[2];
        STR_128_D(curr_out, v_out_stride, v_out2);
        curr_out = out + out_strides[3];
        STR_128_D(curr_out, v_out_stride, v_out3);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        DOUBLE in0, in1, in2, in3;
        DOUBLE s0, s1;
        DOUBLE t0, t1;

        // Input point 1: x(0)
        in0 = *in;
        // Input point 2: x(1)
        in1 = in[in_strides[1]];
        // Input point 3: x(2)
        in2 = in[in_strides[2]];
        // Input point 4: x(3)
        in3 = in[in_strides[3]];

        s0 = in0 + in3;
        t0 = CRTM_4_1 * in1;
        s1 = in0 - in3;
        t1 = CRTM_4_1 * in2;

        // Output point 1: X(0)
        *out = s0 + t0;

        // Output point 2: X(1)
        out[out_strides[1]] = s1 - t1;

        // Output point 3: X(2)
        out[out_strides[2]] = s0 - t0;

        // Output point 4: X(3)
        out[out_strides[3]] = s1 + t1;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

kfft_ register_kernel_r2hc_rfft4avx256(UINT8 precision, UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hc_rfft4avx256_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft4avx256_fp64_fwd;
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
            return r2hc_rfft4avx256_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft4avx256_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
