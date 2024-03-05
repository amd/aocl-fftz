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

/** @file fft2avx128.c
 *
 *  @brief Radix-2 FFT kernel with avx128 operations using x86 SIMD intrinsics
 *
 *  This file contains the DIT radix-2 FFT implementations using avx128
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Varun Sanjay
 *  @author S. Biplab Raut
 *
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_common.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 0, 2, 8, 0, 0},
                                                     {0, 0, 2, 4, 0, 0}};
ops_cycles_t get_ops_cnt_fft2avx128(INT32 precision)
{
    if (precision == DT_FLOAT)
        return ops_cnt[0];
    else
        return ops_cnt[1];
}

static VOID fft2avx128fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                    VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                    UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    FLOAT *in_r = (FLOAT *)in_real;
    FLOAT *out_r = (FLOAT *)out_real;
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
    INTP N = n / NUM_SETS_128_S;
    INTP count;

    __m128 _in0, _in1;
    __m128 _out0, _out1;

    if (flag)
    {
        in_r = in_imag;
        out_r = out_imag;
    }

    for (count = 0; count < N; count++)
    {
        curr_in = in_r;
        curr_out = out_r;

        GATHER2_128_S(curr_in, v_in_stride, _in0);
        curr_in = in_r + in_strides[1];
        GATHER2_128_S(curr_in, v_in_stride, _in1);

        // Output point 1: X[0]
        _out0 = _mm_add_ps(_in0, _in1);
        // Output point 2: X[1]
        _out1 = _mm_sub_ps(_in0, _in1);

        SCATTER2_128_S(curr_out, v_out_stride, _out0);
        curr_out = out_r + out_strides[1];
        SCATTER2_128_S(curr_out, v_out_stride, _out1);

        in_r += NUM_SETS_128_S * v_in_stride;
        out_r += NUM_SETS_128_S * v_out_stride;
    }
    // tail case
    if (n & 1)
    {
        curr_in = in_r;
        curr_out = out_r;

        LD_LOW_128_S(curr_in, _in0);
        curr_in = in_r + in_strides[1];
        LD_LOW_128_S(curr_in, _in1);

        // Output point 1: X[0]
        _out0 = _mm_add_ps(_in0, _in1);
        // Output point 2: X[1]
        _out1 = _mm_sub_ps(_in0, _in1);

        ST_LOW_128_S(curr_out, _out0);
        curr_out = out_r + out_strides[1];
        ST_LOW_128_S(curr_out, _out1);
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID fft2avx128fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                    VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                    UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    DOUBLE *in_r = (DOUBLE *)in_real;
    DOUBLE *out_r = (DOUBLE *)out_real;
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
    INTP count;
    INTP N = n / NUM_SETS_128_D;

    __m128d _in0, _in1;
    __m128d _out0, _out1;

    if (flag)
    {
        in_r = in_imag;
        out_r = out_imag;
    }
    for (count = 0; count < N; count++)
    {
        curr_in = in_r;
        curr_out = out_r;
        LD_128_D(curr_in, _in0);
        curr_in = in_r + in_strides[1];
        LD_128_D(curr_in, _in1);

        // Output point 1: X[0]
        _out0 = _mm_add_pd(_in0, _in1);
        // Output point 2: X[1]
        _out1 = _mm_sub_pd(_in0, _in1);

        ST_128_D(curr_out, _out0);
        curr_out = out_r + out_strides[1];
        ST_128_D(curr_out, _out1);

        in_r += NUM_SETS_128_D * v_in_stride;
        out_r += NUM_SETS_128_D * v_out_stride;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

kfft_ register_kernel_fft2avx128(INT32 precision)
{
    if (precision == DT_FLOAT)
        return fft2avx128fp32;
    else if (precision == DT_DOUBLE)
        return fft2avx128fp64;
    else
        return NULL;
}
