/**
 * Copyright (C) 2024-2025, Advanced Micro Devices. All rights reserved.
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

/** @file fft5avx128.c
 *
 *  @brief Radix-5 FFT kernel with avx128 operations using x86 SIMD intrinsics
 *
 *  This file contains the DIT radix-5 FFT implementations using avx128
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Jeya R
 *  @author Murugan Vairavel
 *  @author S. Biplab Raut
 *
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 6, 16, 20, 2, 2},
                                                     {0, 6, 16, 10, 2, 2}};

ops_cycles_t get_ops_cnt_fft5avx128(UINT8 precision, UINT8 direction)
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

static VOID fft5avx128fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                           VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                           VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_5[4] = {0.559016994374947424102293417182819058860154590,
                             0.250000000000000000000000000000000000000000000,
                             0.951056516295153572116439333379382143405698634,
                             0.587785252292473129168705954639072768597652438};

    FLOAT *in_r = (FLOAT *)in_real;
    FLOAT *out_r = (FLOAT *)out_real;
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
    FLOAT *curr_in, *curr_out;

    __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_av1, v_av2;
    __m128 v_av5, v_av3, v_av4, v_av6;
    __m128 v_cv1, v_tv1, v_tv2, v_tv3, v_tv4;
    __m128 v_out0, v_out1, v_out2, v_out3, v_out4;

    __m128 v_K1 = _mm_broadcast_ss(&CRTM_5[0]);
    __m128 v_K2 = _mm_broadcast_ss(&CRTM_5[1]);
    __m128 v_K3 = _mm_broadcast_ss(&CRTM_5[2]);
    v_K3 = _mm_xor_ps(v_K3, _neg_128_f[flag].s);
    __m128 v_K4 = _mm_broadcast_ss(&CRTM_5[3]);
    v_K4 = _mm_xor_ps(v_K4, _neg_128_f[flag].s);

    for (count = 0; count < N; count++)
    {
        curr_in = in_r;
        curr_out = out_r;

        GATHER2_128_S(curr_in, v_in_stride, v_in0);
        curr_in = in_r + in_strides[1];
        GATHER2_128_S(curr_in, v_in_stride, v_in1);
        curr_in = in_r + in_strides[2];
        GATHER2_128_S(curr_in, v_in_stride, v_in2);
        curr_in = in_r + in_strides[3];
        GATHER2_128_S(curr_in, v_in_stride, v_in3);
        curr_in = in_r + in_strides[4];
        GATHER2_128_S(curr_in, v_in_stride, v_in4);

        // common calculations
        v_av1 = _mm_add_ps(v_in1, v_in4);
        v_av2 = _mm_add_ps(v_in2, v_in3);
        v_av3 = _mm_sub_ps(v_in1, v_in4);
        v_av4 = _mm_sub_ps(v_in2, v_in3);
        v_av5 = _mm_add_ps(v_av1, v_av2);
        v_av6 = _mm_sub_ps(v_av1, v_av2);
        v_tv3 = _mm_mul_ps(v_K3, v_av3);
        v_tv4 = _mm_mul_ps(v_K4, v_av4);

        // Output point 1 : X[0]
        v_out0 = _mm_add_ps(v_in0, v_av5);

        v_tv1 = _mm_mul_ps(v_K2, v_av5);
        v_tv2 = _mm_mul_ps(v_K1, v_av6);
        v_tv1 = _mm_sub_ps(v_in0, v_tv1);
        v_tv4 = _mm_add_ps(v_tv3, v_tv4);
        v_tv4 = SWAP_RI_128_S(CONJ_128_S(v_tv4));
        v_cv1 = _mm_add_ps(v_tv1, v_tv2);

        // Output point 2 : X[1]
        v_out1 = _mm_sub_ps(v_cv1, v_tv4);
        // Output point 5 : X[4]
        v_out4 = _mm_add_ps(v_cv1, v_tv4);

        v_tv3 = _mm_mul_ps(v_K4, v_av3);
        v_tv4 = _mm_mul_ps(v_K3, v_av4);
        v_tv3 = _mm_sub_ps(v_tv3, v_tv4);
        v_tv3 = SWAP_RI_128_S(CONJ_128_S(v_tv3));
        v_cv1 = _mm_sub_ps(v_tv1, v_tv2);

        // Output point 3 : X[2]
        v_out2 = _mm_sub_ps(v_cv1, v_tv3);
        // Output point 4 : X[3]
        v_out3 = _mm_add_ps(v_cv1, v_tv3);

        SCATTER2_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out_r + out_strides[1];
        SCATTER2_128_S(curr_out, v_out_stride, v_out1);
        curr_out = out_r + out_strides[2];
        SCATTER2_128_S(curr_out, v_out_stride, v_out2);
        curr_out = out_r + out_strides[3];
        SCATTER2_128_S(curr_out, v_out_stride, v_out3);
        curr_out = out_r + out_strides[4];
        SCATTER2_128_S(curr_out, v_out_stride, v_out4);

        in_r += NUM_SETS_128_S * v_in_stride;
        out_r += NUM_SETS_128_S * v_out_stride;
    }

    // tail case process one remaining set
    if (n & 1)
    {
        curr_in = in_r;
        curr_out = out_r;

        LD_LOW_128_S(curr_in, v_in0);
        curr_in = in_r + in_strides[1];
        LD_LOW_128_S(curr_in, v_in1);
        curr_in = in_r + in_strides[2];
        LD_LOW_128_S(curr_in, v_in2);
        curr_in = in_r + in_strides[3];
        LD_LOW_128_S(curr_in, v_in3);
        curr_in = in_r + in_strides[4];
        LD_LOW_128_S(curr_in, v_in4);

        // common calculations
        v_av1 = _mm_add_ps(v_in1, v_in4);
        v_av2 = _mm_add_ps(v_in2, v_in3);
        v_av3 = _mm_sub_ps(v_in1, v_in4);
        v_av4 = _mm_sub_ps(v_in2, v_in3);
        v_av5 = _mm_add_ps(v_av1, v_av2);
        v_av6 = _mm_sub_ps(v_av1, v_av2);
        v_tv3 = _mm_mul_ps(v_K3, v_av3);
        v_tv4 = _mm_mul_ps(v_K4, v_av4);

        // Output point 1 : X[0]
        v_out0 = _mm_add_ps(v_in0, v_av5);

        v_tv1 = _mm_mul_ps(v_K2, v_av5);
        v_tv2 = _mm_mul_ps(v_K1, v_av6);
        v_tv1 = _mm_sub_ps(v_in0, v_tv1);
        v_tv4 = _mm_add_ps(v_tv3, v_tv4);
        v_tv4 = SWAP_RI_128_S(CONJ_128_S(v_tv4));
        v_cv1 = _mm_add_ps(v_tv1, v_tv2);

        // Output point 2 : X[1]
        v_out1 = _mm_sub_ps(v_cv1, v_tv4);
        // Output point 5 : X[4]
        v_out4 = _mm_add_ps(v_cv1, v_tv4);

        v_tv3 = _mm_mul_ps(v_K4, v_av3);
        v_tv4 = _mm_mul_ps(v_K3, v_av4);
        v_tv3 = _mm_sub_ps(v_tv3, v_tv4);
        v_tv3 = SWAP_RI_128_S(CONJ_128_S(v_tv3));
        v_cv1 = _mm_sub_ps(v_tv1, v_tv2);

        // Output point 3 : X[2]
        v_out2 = _mm_sub_ps(v_cv1, v_tv3);
        // Output point 4 : X[3]
        v_out3 = _mm_add_ps(v_cv1, v_tv3);

        ST_LOW_128_S(curr_out, v_out0);
        curr_out = out_r + out_strides[1];
        ST_LOW_128_S(curr_out, v_out1);
        curr_out = out_r + out_strides[2];
        ST_LOW_128_S(curr_out, v_out2);
        curr_out = out_r + out_strides[3];
        ST_LOW_128_S(curr_out, v_out3);
        curr_out = out_r + out_strides[4];
        ST_LOW_128_S(curr_out, v_out4);
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID fft5avx128fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                           VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                           VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_5[4] = {0.559016994374947424102293417182819058860154590,
                              0.250000000000000000000000000000000000000000000,
                              0.951056516295153572116439333379382143405698634,
                              0.587785252292473129168705954639072768597652438};

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

    __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_av1, v_av2;
    __m128d v_av5, v_av3, v_av4, v_av6;
    __m128d v_cv1, v_tv1, v_tv2, v_tv3, v_tv4;
    __m128d v_out0, v_out1, v_out2, v_out3, v_out4;

    __m128d v_K1 = _mm_set1_pd(CRTM_5[0]);
    __m128d v_K2 = _mm_set1_pd(CRTM_5[1]);
    __m128d v_K3 = _mm_set1_pd(CRTM_5[2]);
    v_K3 = _mm_xor_pd(v_K3, _neg_128_d[flag].d);
    __m128d v_K4 = _mm_set1_pd(CRTM_5[3]);
    v_K4 = _mm_xor_pd(v_K4, _neg_128_d[flag].d);

    for (count = 0; count < n; count++)
    {
        curr_in = in_r;
        curr_out = out_r;

        LD_128_D(curr_in, v_in0);
        curr_in = in_r + in_strides[1];
        LD_128_D(curr_in, v_in1);
        curr_in = in_r + in_strides[2];
        LD_128_D(curr_in, v_in2);
        curr_in = in_r + in_strides[3];
        LD_128_D(curr_in, v_in3);
        curr_in = in_r + in_strides[4];
        LD_128_D(curr_in, v_in4);

        // common calculations
        v_av1 = _mm_add_pd(v_in1, v_in4);
        v_av2 = _mm_add_pd(v_in2, v_in3);
        v_av3 = _mm_sub_pd(v_in1, v_in4);
        v_av4 = _mm_sub_pd(v_in2, v_in3);
        v_av5 = _mm_add_pd(v_av1, v_av2);
        v_av6 = _mm_sub_pd(v_av1, v_av2);
        v_tv3 = _mm_mul_pd(v_K3, v_av3);
        v_tv4 = _mm_mul_pd(v_K4, v_av4);

        // Output point 1 : X[0]
        v_out0 = _mm_add_pd(v_in0, v_av5);

        v_tv1 = _mm_mul_pd(v_K2, v_av5);
        v_tv2 = _mm_mul_pd(v_K1, v_av6);
        v_tv1 = _mm_sub_pd(v_in0, v_tv1);
        v_tv4 = _mm_add_pd(v_tv3, v_tv4);
        v_tv4 = SWAP_RI_128_D(CONJ_128_D(v_tv4));
        v_cv1 = _mm_add_pd(v_tv1, v_tv2);

        // Output point 2 : X[1]
        v_out1 = _mm_sub_pd(v_cv1, v_tv4);
        // Output point 5 : X[4]
        v_out4 = _mm_add_pd(v_cv1, v_tv4);

        v_tv3 = _mm_mul_pd(v_K4, v_av3);
        v_tv4 = _mm_mul_pd(v_K3, v_av4);
        v_tv3 = _mm_sub_pd(v_tv3, v_tv4);
        v_tv3 = SWAP_RI_128_D(CONJ_128_D(v_tv3));
        v_cv1 = _mm_sub_pd(v_tv1, v_tv2);

        // Output point 3 : X[2]
        v_out2 = _mm_sub_pd(v_cv1, v_tv3);
        // Output point 4 : X[3]
        v_out3 = _mm_add_pd(v_cv1, v_tv3);

        ST_128_D(curr_out, v_out0);
        curr_out = out_r + out_strides[1];
        ST_128_D(curr_out, v_out1);
        curr_out = out_r + out_strides[2];
        ST_128_D(curr_out, v_out2);
        curr_out = out_r + out_strides[3];
        ST_128_D(curr_out, v_out3);
        curr_out = out_r + out_strides[4];
        ST_128_D(curr_out, v_out4);

        in_r += v_in_stride;
        out_r += v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_fft5avx128(UINT8 precision, UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return fft5avx128fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft5avx128fp64;
    }
    else
    {
        return NULL;
    }
}
