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

/** @file fft2avx512.c
 *
 *  @brief Radix-2 FFT kernel with AVX-512 operations using x86 SIMD intrinsics
 *
 *  This file contains the DIT radix-2 FFT implementations using AVX-512
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Jeya R
 *
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 0, 2, 32, 6, 12},
                                                     {0, 0, 2, 16, 0, 12}};

ops_cycles_t get_ops_cnt_fft2avx512(UINT8 precision, UINT8 direction)
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

static VOID fft2avx512fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                           VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                           UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    FLOAT *in_r = in_real;
    FLOAT *out_r = out_real;
    FLOAT *curr_in, *curr_out;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;
    INTP N = n / NUM_SETS_512_S;
    INTP remaining_sets = n % NUM_SETS_512_S;
    INTP count;

    if (flag)
    {
        in_r = in_imag;
        out_r = out_imag;
    }

    for (count = 0; count < N; count++)
    {
        __m512 v_in0, v_in1;
        __m512 v_out0, v_out1;

        curr_in = in_r;
        GATHER8_512_S(curr_in, v_in_stride, v_in0);
        curr_in = in_r + in_strides[1];
        GATHER8_512_S(curr_in, v_in_stride, v_in1);

        // Output point 1: X[0]
        v_out0 = _mm512_add_ps(v_in0, v_in1);
        // Output point 2: X[1]
        v_out1 = _mm512_sub_ps(v_in0, v_in1);

        curr_out = out_r;
        SCATTER8_512_S(curr_out, v_out_stride, v_out0);
        curr_out = out_r + out_strides[1];
        SCATTER8_512_S(curr_out, v_out_stride, v_out1);

        in_r += NUM_SETS_512_S * v_in_stride;
        out_r += NUM_SETS_512_S * v_out_stride;
    }
    // tail cases
    if (remaining_sets >= NUM_SETS_256_S)
    {
        __m256 v_in0, v_in1;
        __m256 v_out0, v_out1;
        curr_in = in_r;
        GATHER4_256_S(curr_in, v_in_stride, v_in0);
        curr_in = in_r + in_strides[1];
        GATHER4_256_S(curr_in, v_in_stride, v_in1);

        // Output point 1: X[0]
        v_out0 = _mm256_add_ps(v_in0, v_in1);
        // Output point 2: X[1]
        v_out1 = _mm256_sub_ps(v_in0, v_in1);

        curr_out = out_r;
        SCATTER4_256_S(curr_out, v_out_stride, v_out0);
        curr_out = out_r + out_strides[1];
        SCATTER4_256_S(curr_out, v_out_stride, v_out1);

        in_r += NUM_SETS_256_S * v_in_stride;
        out_r += NUM_SETS_256_S * v_out_stride;
        remaining_sets = remaining_sets - NUM_SETS_256_S;
    }
    // tail case
    if (remaining_sets >= NUM_SETS_128_S)
    {
        __m128 v_in0, v_in1;
        __m128 v_out0, v_out1;
        curr_in = in_r;
        GATHER2_128_S(curr_in, v_in_stride, v_in0);
        curr_in = in_r + in_strides[1];
        GATHER2_128_S(curr_in, v_in_stride, v_in1);

        // Output point 1: X[0]
        v_out0 = _mm_add_ps(v_in0, v_in1);
        // Output point 2: X[1]
        v_out1 = _mm_sub_ps(v_in0, v_in1);

        curr_out = out_r;
        SCATTER2_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out_r + out_strides[1];
        SCATTER2_128_S(curr_out, v_out_stride, v_out1);

        in_r = in_r + (v_in_stride << 1);
        out_r = out_r + (v_out_stride << 1);
    }
    if (remaining_sets & 1)
    {
        __m128 v_in0, v_in1;
        __m128 v_out0, v_out1;

        curr_in = in_r;
        LD_LOW_128_S(curr_in, v_in0);
        curr_in = in_r + in_strides[1];
        LD_LOW_128_S(curr_in, v_in1);

        // Output point 1: X[0]
        v_out0 = _mm_add_ps(v_in0, v_in1);
        // Output point 2: X[1]
        v_out1 = _mm_sub_ps(v_in0, v_in1);

        curr_out = out_r;
        ST_LOW_128_S(curr_out, v_out0);
        curr_out = out_r + out_strides[1];
        ST_LOW_128_S(curr_out, v_out1);
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID fft2avx512fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                           VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                           UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    DOUBLE *in_r = in_real;
    DOUBLE *out_r = out_real;
    DOUBLE *curr_in, *curr_out;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;
    INTP N = n / NUM_SETS_512_D;
    INTP remaining_sets = n % NUM_SETS_512_D;
    INTP count;

    if (flag)
    {
        in_r = in_imag;
        out_r = out_imag;
    }

    for (count = 0; count < N; count++)
    {
        __m512d v_in0, v_in1;
        __m512d v_out0, v_out1;

        curr_in = in_r;
        GATHER4_512_D(curr_in, v_in_stride, v_in0);
        curr_in = in_r + in_strides[1];
        GATHER4_512_D(curr_in, v_in_stride, v_in1);

        // Output point 1: X[0]
        v_out0 = _mm512_add_pd(v_in0, v_in1);
        // Output point 2: X[1]
        v_out1 = _mm512_sub_pd(v_in0, v_in1);

        curr_out = out_r;
        SCATTER4_512_D(curr_out, v_out_stride, v_out0);
        curr_out = out_r + out_strides[1];
        SCATTER4_512_D(curr_out, v_out_stride, v_out1);

        in_r += NUM_SETS_512_D * v_in_stride;
        out_r += NUM_SETS_512_D * v_out_stride;
    }
    // tail cases
    if (remaining_sets >= NUM_SETS_256_D)
    {
        __m256d v_in0, v_in1;
        __m256d v_out0, v_out1;
        curr_in = in_r;
        GATHER2_256_D(curr_in, v_in_stride, v_in0);
        curr_in = in_r + in_strides[1];
        GATHER2_256_D(curr_in, v_in_stride, v_in1);

        // Output point 1: X[0]
        v_out0 = _mm256_add_pd(v_in0, v_in1);
        // Output point 2: X[1]
        v_out1 = _mm256_sub_pd(v_in0, v_in1);

        curr_out = out_r;
        SCATTER2_256_D(curr_out, v_out_stride, v_out0);
        curr_out = out_r + out_strides[1];
        SCATTER2_256_D(curr_out, v_out_stride, v_out1);

        in_r += NUM_SETS_256_D * v_in_stride;
        out_r += NUM_SETS_256_D * v_out_stride;
    }
    // tail case
    if (remaining_sets & 1)
    {
        __m128d v_in0, v_in1;
        __m128d v_out0, v_out1;
        curr_in = in_r;
        LD_128_D(curr_in, v_in0);
        curr_in = in_r + in_strides[1];
        LD_128_D(curr_in, v_in1);

        // Output point 1: X[0]
        v_out0 = _mm_add_pd(v_in0, v_in1);
        // Output point 2: X[1]
        v_out1 = _mm_sub_pd(v_in0, v_in1);

        curr_out = out_r;
        ST_128_D(curr_out, v_out0);
        curr_out = out_r + out_strides[1];
        ST_128_D(curr_out, v_out1);
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

kfft_ register_kernel_fft2avx512(UINT8 precision, UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return fft2avx512fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft2avx512fp64;
    }
    else
    {
        return NULL;
    }
}
