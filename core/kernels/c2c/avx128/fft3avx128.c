// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fft3avx128.c
 *
 *  @brief Radix-3 DFT kernel with AVX-128 operations using x86 SIMD intrinsics
 *
 *  This file contains the DIT Radix-3 DFT implementations using AVX128
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Prasandh Sankarankutty
 *  @author Varun Sanjay
 *  @author S. Biplab Raut
 *
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 2, 6, 12, 1, 1},
                                                     {0, 2, 6,  6, 1, 1}};

ops_cycles_t get_ops_cnt_fft3avx128(UINT8 precision, UINT8 direction)
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

static VOID fft3avx128fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                           VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                           VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_3[2] = {0.500000000000000000000000000000000000000000000,
                             0.866025403784438646763723170752936183471402627};

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

    __m128 v_in0, v_in1, v_in2;
    __m128 v_out0, v_out1, v_out2;
    __m128 v_av0, v_av1, v_tv0, v_tv1;

    __m128 v_K1 = _mm_broadcast_ss(&CRTM_3[0]);
    __m128 v_K2 = _mm_broadcast_ss(&CRTM_3[1]);
    v_K2 = _mm_xor_ps(v_K2, _neg_128_f[flag].s);

    for (count = 0; count < N; count++)
    {
        curr_in = in_r;
        curr_out = out_r;

        GATHER2_128_S(curr_in, v_in_stride, v_in0);
        curr_in = in_r + in_strides[1];
        GATHER2_128_S(curr_in, v_in_stride, v_in1);
        curr_in = in_r + in_strides[2];
        GATHER2_128_S(curr_in, v_in_stride, v_in2);

        v_av0 = _mm_add_ps(v_in1, v_in2);
        v_av1 = _mm_sub_ps(v_in1, v_in2);
        v_tv0 = _mm_sub_ps(v_in0, _mm_mul_ps(v_K1, v_av0));

        // Output point 1: X[0]
        v_out0 = _mm_add_ps(v_in0, v_av0);

        v_tv1 = _mm_mul_ps(v_K2, v_av1);
        v_tv1 = SWAP_RI_128_S(CONJ_128_S(v_tv1));

        // Output point 2: X[1]
        v_out1 = _mm_sub_ps(v_tv0, v_tv1);
        // Output point 3: X[2]
        v_out2 = _mm_add_ps(v_tv0, v_tv1);

        SCATTER2_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out_r + out_strides[1];
        SCATTER2_128_S(curr_out, v_out_stride, v_out1);
        curr_out = out_r + out_strides[2];
        SCATTER2_128_S(curr_out, v_out_stride, v_out2);

        in_r += NUM_SETS_128_S * v_in_stride;
        out_r += NUM_SETS_128_S * v_out_stride;
    }
    // tail case
    if (n & 1)
    {
        curr_in = in_r;
        LD_LOW_128_S(curr_in, v_in0);
        curr_in = in_r + in_strides[1];
        LD_LOW_128_S(curr_in, v_in1);
        curr_in = in_r + in_strides[2];
        LD_LOW_128_S(curr_in, v_in2);

        v_av0 = _mm_add_ps(v_in1, v_in2);
        v_av1 = _mm_sub_ps(v_in1, v_in2);
        v_tv0 = _mm_sub_ps(v_in0, _mm_mul_ps(v_K1, v_av0));

        // Output point 1: X[0]
        v_out0 = _mm_add_ps(v_in0, v_av0);

        v_tv1 = _mm_mul_ps(v_K2, v_av1);
        v_tv1 = SWAP_RI_128_S(CONJ_128_S(v_tv1));

        // Output point 2: X[1]
        v_out1 = _mm_sub_ps(v_tv0, v_tv1);
        // Output point 3: X[2]
        v_out2 = _mm_add_ps(v_tv0, v_tv1);

        curr_out = out_r;
        ST_LOW_128_S(curr_out, v_out0);
        curr_out = out_r + out_strides[1];
        ST_LOW_128_S(curr_out, v_out1);
        curr_out = out_r + out_strides[2];
        ST_LOW_128_S(curr_out, v_out2);
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID fft3avx128fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                           VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                           VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_3[2] = {0.500000000000000000000000000000000000000000000,
                              0.866025403784438646763723170752936183471402627};

    DOUBLE *in_r = (DOUBLE *)in_real;
    DOUBLE *out_r = (DOUBLE *)out_real;
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
    DOUBLE *curr_in, *curr_out;

    __m128d v_in0, v_in1, v_in2;
    __m128d v_out0, v_out1, v_out2;
    __m128d v_av0, v_av1, v_tv0, v_tv1;

    __m128d v_K1 = _mm_set1_pd(CRTM_3[0]);
    __m128d v_K2 = _mm_set1_pd(CRTM_3[1]);
    v_K2 = _mm_xor_pd(v_K2, _neg_128_d[flag].d);

    for (count = 0; count < N; count++)
    {
        curr_in = in_r;
        LD_128_D(curr_in, v_in0);
        curr_in = in_r + in_strides[1];
        LD_128_D(curr_in, v_in1);
        curr_in = in_r + in_strides[2];
        LD_128_D(curr_in, v_in2);

        v_av0 = _mm_add_pd(v_in1, v_in2);
        v_av1 = _mm_sub_pd(v_in1, v_in2);
        v_tv0 = _mm_sub_pd(v_in0, _mm_mul_pd(v_K1, v_av0));

        // Output point 1: X[0]
        v_out0 = _mm_add_pd(v_in0, v_av0);

        v_tv1 = _mm_mul_pd(v_K2, v_av1);
        v_tv1 = SWAP_RI_128_D(CONJ_128_D(v_tv1));

        // Output point 2: X[1]
        v_out1 = _mm_sub_pd(v_tv0, v_tv1);
        // Output point 3: X[2]
        v_out2 = _mm_add_pd(v_tv0, v_tv1);

        curr_out = out_r;
        ST_128_D(curr_out, v_out0);
        curr_out = out_r + out_strides[1];
        ST_128_D(curr_out, v_out1);
        curr_out = out_r + out_strides[2];
        ST_128_D(curr_out, v_out2);

        in_r += NUM_SETS_128_D * v_in_stride;
        out_r += NUM_SETS_128_D * v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_fft3avx128(UINT8 precision, UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return fft3avx128fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft3avx128fp64;
    }
    else
    {
        return NULL;
    }
}
