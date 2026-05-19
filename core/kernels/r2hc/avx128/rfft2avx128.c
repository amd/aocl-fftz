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

/** @file rfft2avx128.c
 *
 *  @brief Radix-2 r2hc Real-FFT kernel with AVX-128 operations using x86 SIMD
 *  intrinsics
 *
 *  This file contains the DIT radix-2 real-to-halfcomplex implementations using
 *  AVX128 SIMD operations for single-precision and double-precision inputs.
 *
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/r2hc_simd_avx128.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 0, 2, 16, 12, 0},
                                                     {0, 0, 2, 8, 0, 0}};

ops_cycles_t get_ops_cnt_r2hc_rfft2avx128(UINT8 precision, UINT8 direction)
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

static VOID r2hc_rfft2avx128_fp32(VOID *in_real, VOID *in_imag,
                                  VOID *out_real, VOID *out_imag, INTP n,
                                  aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
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
    INTP N = n / NUM_SETS_REAL_128_S;

    for (cnt = 0; cnt < N; cnt++)
    {
        __m128 v_in0, v_in1;
        __m128 v_out0, v_out1;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_128_S(curr_in, v_in_stride, v_in1);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_in0, v_in1);

        // Output point 2: X(0)
        v_out1 = _mm_sub_ps(v_in0, v_in1);

        STR_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STR_128_S(curr_out, v_out_stride, v_out1);

        in  += v_in_stride * NUM_SETS_REAL_128_S;
        out += v_out_stride * NUM_SETS_REAL_128_S;
    }
    // tail cases
    if (n & 2)
    {
        __m128 v_in0, v_in1;
        __m128 v_out0, v_out1;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDHR_128_S(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDHR_128_S(curr_in, v_in_stride, v_in1);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_in0, v_in1);

        // Output point 2: X(0)
        v_out1 = _mm_sub_ps(v_in0, v_in1);

        STHR_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STHR_128_S(curr_out, v_out_stride, v_out1);

        in  = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        FLOAT v_in0, v_in1;

        // Input point 1: x(0)
        v_in0 = *in;
        // Input point 2: x(1)
        v_in1 = in[in_strides[1]];

        // Output point 1: X(0)
        *out = v_in0 + v_in1;

        // Output point 2: X(0)
        out[out_strides[1]] = v_in0 - v_in1;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID r2hc_rfft2avx128_fp64(VOID *in_real, VOID *in_imag,
                                  VOID *out_real, VOID *out_imag, INTP n,
                                  aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
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
    INTP N = n / NUM_SETS_REAL_128_D;

    for (cnt = 0; cnt < N; cnt++)
    {
        __m128d v_in0, v_in1;
        __m128d v_out0, v_out1;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_128_D(curr_in, v_in_stride, v_in1);

        // Output point 1: X(0)
        v_out0 = _mm_add_pd(v_in0, v_in1);

        // Output point 2: X(0)
        v_out1 = _mm_sub_pd(v_in0, v_in1);

        STR_128_D(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STR_128_D(curr_out, v_out_stride, v_out1);

        in  += v_in_stride * NUM_SETS_REAL_128_D;
        out += v_out_stride * NUM_SETS_REAL_128_D;
    }
    // tail cases
    if (n & 1)
    {
        DOUBLE v_in0, v_in1;

        // Input point 1: x(0)
        v_in0 = *in;
        // Input point 2: x(1)
        v_in1 = in[in_strides[1]];

        // Output point 1: X(0)
        *out = v_in0 + v_in1;

        // Output point 2: X(0)
        out[out_strides[1]] = v_in0 - v_in1;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hc_rfft2avx128(UINT8 precision, UINT8 direction)
{

    if (precision == DT_FLOAT)
    {
        return r2hc_rfft2avx128_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return r2hc_rfft2avx128_fp64;
    }
    else
    {
        return NULL;
    }
}
