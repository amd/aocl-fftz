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

/** @file fft4avx128.c
 *
 *  @brief Radix-4 FFT kernel with avx128 operations using x86 SIMD intrinsics
 *
 *  This file contains the DIT radix-4 FFT implementations using avx128
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Varun Sanjay
 *  @author S. Biplab Raut
 *
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_common.h"

kfft_ register_kernel_fft4avx128(INT32 precision)
{
    if (precision == DT_FLOAT)
        return fft4avx128fp32;
    else if (precision == DT_DOUBLE)
        return fft4avx128fp64;
    else
        return NULL;
}

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 1, 8, 16, 1, 1},
                                                     {0, 1, 8, 8, 1, 1}};
ops_cycles_t get_ops_cnt_fft4avx128(INT32 precision)
{
    if (precision == DT_FLOAT)
        return ops_cnt[0];
    else
        return ops_cnt[1];
}

VOID fft4avx128fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                    VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                    UINT8 flag)
{
    const FLOAT CRTM_4_1 = 1.0;

    FLOAT *in_r = (FLOAT *)in_real;
    FLOAT *out_r = (FLOAT *)out_real;
    INTP in_stride = (strides->in_stride << 1);
    INTP out_stride = (strides->out_stride << 1);
    INTP v_in_stride = (strides->v_in_stride << 1);
    INTP v_out_stride = (strides->v_out_stride << 1);
    INTP N = n / NUM_SETS_128_S;
    INTP count;
    FLOAT *curr_in, *curr_out;

    __m128 v_in0, v_in1, v_in2, v_in3;
    __m128 v_out0, v_out1, v_out2, v_out3;
    __m128 v_av1, v_av2;

    __m128 v_K1 = _mm_broadcast_ss(&CRTM_4_1);

    if (flag)
    {
        in_r = in_imag;
        out_r = out_imag;
        v_K1 = -v_K1;
    }

    for (count = 0; count < N; count++)
    {
        curr_in = in_r;
        curr_out = out_r;

        GATHER2_128_S(curr_in, v_in_stride, v_in0);
        curr_in = in_r + in_stride;
        GATHER2_128_S(curr_in, v_in_stride, v_in1);
        curr_in = in_r + (in_stride << 1);
        GATHER2_128_S(curr_in, v_in_stride, v_in2);
        curr_in = in_r + (in_stride * 3);
        GATHER2_128_S(curr_in, v_in_stride, v_in3);

        v_av1 = _mm_add_ps(v_in0, v_in2);
        v_av2 = _mm_add_ps(v_in1, v_in3);

        // Output point 1 : X[0]
        v_out0 = _mm_add_ps(v_av1, v_av2);
        // Output point 3 : X[2]
        v_out2 = _mm_sub_ps(v_av1, v_av2);

        v_av1 = _mm_sub_ps(v_in3, v_in1);
        v_av1 = _mm_mul_ps(v_K1, v_av1);
        v_av1 = SWAP_RI_128_S(CONJ_128_S(v_av1));
        v_av2 = _mm_sub_ps(v_in0, v_in2);

        // Output point 2 : X[1]
        v_out1 = _mm_add_ps(v_av2, v_av1);
        // Output point 4 : X[3]
        v_out3 = _mm_sub_ps(v_av2, v_av1);

        SCATTER2_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out_r + out_stride;
        SCATTER2_128_S(curr_out, v_out_stride, v_out1);
        curr_out = out_r + (out_stride << 1);
        SCATTER2_128_S(curr_out, v_out_stride, v_out2);
        curr_out = out_r + (out_stride * 3);
        SCATTER2_128_S(curr_out, v_out_stride, v_out3);

        in_r += NUM_SETS_128_S  * v_in_stride;
        out_r += NUM_SETS_128_S  * v_out_stride;
    }
    // tail case
    if (n & 1)
    {
        curr_in = in_r;
        curr_out = out_r;

        LD_LOW_128_S(curr_in, v_in0);
        curr_in = in_r + in_stride;
        LD_LOW_128_S(curr_in, v_in1);
        curr_in = in_r + (in_stride << 1);
        LD_LOW_128_S(curr_in, v_in2);
        curr_in = in_r + (in_stride * 3);
        LD_LOW_128_S(curr_in, v_in3);

        v_av1 = _mm_add_ps(v_in0, v_in2);
        v_av2 = _mm_add_ps(v_in1, v_in3);

        // Output point 1 : X[0]
        v_out0 = _mm_add_ps(v_av1, v_av2);
        // Output point 3 : X[2]
        v_out2 = _mm_sub_ps(v_av1, v_av2);

        v_av1 = _mm_sub_ps(v_in3, v_in1);
        v_av1 = _mm_mul_ps(v_K1, v_av1);
        v_av1 = SWAP_RI_128_S(CONJ_128_S(v_av1));
        v_av2 = _mm_sub_ps(v_in0, v_in2);

        // Output point 2 : X[1]
        v_out1 = _mm_add_ps(v_av2, v_av1);
        // Output point 4 : X[3]
        v_out3 = _mm_sub_ps(v_av2, v_av1);

        ST_LOW_128_S(curr_out, v_out0);
        curr_out = out_r + out_stride;
        ST_LOW_128_S(curr_out, v_out1);
        curr_out = out_r + (out_stride << 1);
        ST_LOW_128_S(curr_out, v_out2);
        curr_out = out_r + (out_stride * 3);
        ST_LOW_128_S(curr_out, v_out3);
    }
}

VOID fft4avx128fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                    VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                    UINT8 flag)
{
    const DOUBLE CRTM_4_1 = 1.0;

    DOUBLE *in_r = (DOUBLE *)in_real;
    DOUBLE *out_r = (DOUBLE *)out_real;
    INTP in_stride = (strides->in_stride << 1);
    INTP out_stride = (strides->out_stride << 1);
    INTP v_in_stride = (strides->v_in_stride << 1);
    INTP v_out_stride = (strides->v_out_stride << 1);
    INTP N = n / NUM_SETS_128_D;
    INTP count;
    DOUBLE *curr_in, *curr_out;

    __m128d v_in0, v_in1, v_in2, v_in3;
    __m128d v_out0, v_out1, v_out2, v_out3;
    __m128d v_av1, v_av2;

    __m128d v_K1 = _mm_set1_pd(CRTM_4_1);

    if (flag)
    {
        in_r = in_imag;
        out_r = out_imag;
        v_K1 = -v_K1;
    }

    for (count = 0; count < N; count++)
    {
        curr_in = in_r;
        curr_out = out_r;

        LD_128_D(curr_in, v_in0);
        curr_in = in_r + in_stride;
        LD_128_D(curr_in, v_in1);
        curr_in = in_r + (in_stride << 1);
        LD_128_D(curr_in, v_in2);
        curr_in = in_r + (in_stride * 3);
        LD_128_D(curr_in, v_in3);

        v_av1 = _mm_add_pd(v_in0, v_in2);
        v_av2 = _mm_add_pd(v_in1, v_in3);

        // Output point 1 : X[0]
        v_out0 = _mm_add_pd(v_av1, v_av2);
        // Output point 3 : X[2]
        v_out2 = _mm_sub_pd(v_av1, v_av2);

        v_av1 = _mm_sub_pd(v_in3, v_in1);
        v_av1 = _mm_mul_pd(v_K1, v_av1);
        v_av1 = SWAP_RI_128_D(CONJ_128_D(v_av1));
        v_av2 = _mm_sub_pd(v_in0, v_in2);

        // Output point 2 : X[1]
        v_out1 = _mm_add_pd(v_av2, v_av1);
        // Output point 4 : X[3]
        v_out3 = _mm_sub_pd(v_av2, v_av1);

        ST_128_D(curr_out, v_out0);
        curr_out = out_r + out_stride;
        ST_128_D(curr_out, v_out1);
        curr_out = out_r + (out_stride << 1);
        ST_128_D(curr_out, v_out2);
        curr_out = out_r + (out_stride * 3);
        ST_128_D(curr_out, v_out3);

        in_r += v_in_stride;
        out_r += v_out_stride;
    }
}
