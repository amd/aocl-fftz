// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fft4avx256.c
 *
 *  @brief Radix-4 FFT kernel with AVX-256 operations using x86 SIMD intrinsics
 *
 *  This file contains the DIT radix-4 FFT implementations using AVX-256
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Varun Sanjay
 *
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 1, 8, 32, 5, 9},
                                                     {0, 1, 8, 16, 1, 9}};

ops_cycles_t get_ops_cnt_fft4avx256(UINT8 precision, UINT8 direction)
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

static VOID fft4avx256fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                           VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                           VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_4_1 = 1.0;

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
    UINT8 is_contiguous_in = (v_in_stride == DATA_STRIDE);
    INTP v_out_stride = strides->v_out_stride;
    UINT8 is_contiguous_out = (v_out_stride == DATA_STRIDE);
    INTP N = n / NUM_SETS_256_S;
    INTP remaining_sets = n % NUM_SETS_256_S;
    INTP count;
    FLOAT *curr_in, *curr_out;

    __m256 v_in0, v_in1, v_in2, v_in3;
    __m256 v_av1, v_av2;
    __m256 v_out0, v_out1, v_out2, v_out3;
    __m256 v_C1 = _mm256_broadcast_ss(&CRTM_4_1);
    v_C1 = _mm256_xor_ps(v_C1, _neg_256_f[flag].s);

    for (count = 0; count < N; count++)
    {
        curr_in = in_r;
        curr_out = out_r;

        GATHER4_256_S(curr_in, v_in_stride, v_in0, is_contiguous_in);
        curr_in = in_r + in_strides[1];
        GATHER4_256_S(curr_in, v_in_stride, v_in1, is_contiguous_in);
        curr_in = in_r + in_strides[2];
        GATHER4_256_S(curr_in, v_in_stride, v_in2, is_contiguous_in);
        curr_in = in_r + in_strides[3];
        GATHER4_256_S(curr_in, v_in_stride, v_in3, is_contiguous_in);

        v_av1 = _mm256_add_ps(v_in0, v_in2);
        v_av2 = _mm256_add_ps(v_in1, v_in3);

        // Output point 1 : X[0]
        v_out0 = _mm256_add_ps(v_av1, v_av2);
        // Output point 3 : X[2]
        v_out2 = _mm256_sub_ps(v_av1, v_av2);

        v_av1 = _mm256_sub_ps(v_in3, v_in1);
        v_av1 = _mm256_mul_ps(v_C1, v_av1);
        v_av1 = SWAP_RI_256_S(CONJ_256_S(v_av1));
        v_av2 = _mm256_sub_ps(v_in0, v_in2);

        // Output point 2 : X[1]
        v_out1 = _mm256_add_ps(v_av2, v_av1);
        // Output point 4 : X[3]
        v_out3 = _mm256_sub_ps(v_av2, v_av1);

        SCATTER4_256_S(curr_out, v_out_stride, v_out0, is_contiguous_out);
        curr_out = out_r + out_strides[1];
        SCATTER4_256_S(curr_out, v_out_stride, v_out1, is_contiguous_out);
        curr_out = out_r + out_strides[2];
        SCATTER4_256_S(curr_out, v_out_stride, v_out2, is_contiguous_out);
        curr_out = out_r + out_strides[3];
        SCATTER4_256_S(curr_out, v_out_stride, v_out3, is_contiguous_out);

        in_r += NUM_SETS_256_S * v_in_stride;
        out_r += NUM_SETS_256_S * v_out_stride;
    }
    // tail cases
    if (remaining_sets >= NUM_SETS_128_S)
    {
        __m128 v_in0, v_in1, v_in2, v_in3;
        __m128 v_av1, v_av2;
        __m128 v_out0, v_out1, v_out2, v_out3;

        __m128 v_K1 = _mm256_castps256_ps128(v_C1);

        curr_in = in_r;
        curr_out = out_r;

        GATHER2_128_S(curr_in, v_in_stride, v_in0, is_contiguous_in);
        curr_in = in_r + in_strides[1];
        GATHER2_128_S(curr_in, v_in_stride, v_in1, is_contiguous_in);
        curr_in = in_r + in_strides[2];
        GATHER2_128_S(curr_in, v_in_stride, v_in2, is_contiguous_in);
        curr_in = in_r + in_strides[3];
        GATHER2_128_S(curr_in, v_in_stride, v_in3, is_contiguous_in);

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

        SCATTER2_128_S(curr_out, v_out_stride, v_out0, is_contiguous_out);
        curr_out = out_r + out_strides[1];
        SCATTER2_128_S(curr_out, v_out_stride, v_out1, is_contiguous_out);
        curr_out = out_r + out_strides[2];
        SCATTER2_128_S(curr_out, v_out_stride, v_out2, is_contiguous_out);
        curr_out = out_r + out_strides[3];
        SCATTER2_128_S(curr_out, v_out_stride, v_out3, is_contiguous_out);

        in_r = in_r + (v_in_stride << 1);
        out_r = out_r + (v_out_stride << 1);
    }
    if (remaining_sets & 1)
    {
        __m128 v_in0, v_in1, v_in2, v_in3;
        __m128 v_av1, v_av2;
        __m128 v_out0, v_out1, v_out2, v_out3;

        __m128 v_K1 = _mm256_castps256_ps128(v_C1);

        curr_in = in_r;
        curr_out = out_r;

        LD_LOW_128_S(curr_in, v_in0);
        curr_in = in_r + in_strides[1];
        LD_LOW_128_S(curr_in, v_in1);
        curr_in = in_r + in_strides[2];
        LD_LOW_128_S(curr_in, v_in2);
        curr_in = in_r + in_strides[3];
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
        curr_out = out_r + out_strides[1];
        ST_LOW_128_S(curr_out, v_out1);
        curr_out = out_r + out_strides[2];
        ST_LOW_128_S(curr_out, v_out2);
        curr_out = out_r + out_strides[3];
        ST_LOW_128_S(curr_out, v_out3);
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID fft4avx256fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                           VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                           VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_4_1 = 1.0;

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
    UINT8 is_contiguous_in = (v_in_stride == DATA_STRIDE);
    INTP v_out_stride = strides->v_out_stride;
    UINT8 is_contiguous_out = (v_out_stride == DATA_STRIDE);
    INTP N = n / NUM_SETS_256_D;
    INTP count;
    DOUBLE *curr_in, *curr_out;

    __m256d v_in0, v_in1, v_in2, v_in3;
    __m256d v_av1, v_av2;
    __m256d v_out0, v_out1, v_out2, v_out3;

    __m256d v_C1 = _mm256_broadcast_sd(&CRTM_4_1);
    v_C1 = _mm256_xor_pd(v_C1, _neg_256_d[flag].d);

    for (count = 0; count < N; count++)
    {
        curr_in = in_r;
        curr_out = out_r;

        GATHER2_256_D(curr_in, v_in_stride, v_in0, is_contiguous_in);
        curr_in = in_r + in_strides[1];
        GATHER2_256_D(curr_in, v_in_stride, v_in1, is_contiguous_in);
        curr_in = in_r + in_strides[2];
        GATHER2_256_D(curr_in, v_in_stride, v_in2, is_contiguous_in);
        curr_in = in_r + in_strides[3];
        GATHER2_256_D(curr_in, v_in_stride, v_in3, is_contiguous_in);

        v_av1 = _mm256_add_pd(v_in0, v_in2);
        v_av2 = _mm256_add_pd(v_in1, v_in3);

        // Output point 1 : X[0]
        v_out0 = _mm256_add_pd(v_av1, v_av2);
        // Output point 3 : X[2]
        v_out2 = _mm256_sub_pd(v_av1, v_av2);

        v_av1 = _mm256_sub_pd(v_in3, v_in1);
        v_av1 = _mm256_mul_pd(v_C1, v_av1);
        v_av1 = SWAP_RI_256_D(CONJ_256_D(v_av1));
        v_av2 = _mm256_sub_pd(v_in0, v_in2);

        // Output point 2 : X[1]
        v_out1 = _mm256_add_pd(v_av2, v_av1);
        // Output point 4 : X[3]
        v_out3 = _mm256_sub_pd(v_av2, v_av1);

        SCATTER2_256_D(curr_out, v_out_stride, v_out0, is_contiguous_out);
        curr_out = out_r + out_strides[1];
        SCATTER2_256_D(curr_out, v_out_stride, v_out1, is_contiguous_out);
        curr_out = out_r + out_strides[2];
        SCATTER2_256_D(curr_out, v_out_stride, v_out2, is_contiguous_out);
        curr_out = out_r + out_strides[3];
        SCATTER2_256_D(curr_out, v_out_stride, v_out3, is_contiguous_out);

        in_r += NUM_SETS_256_D * v_in_stride;
        out_r += NUM_SETS_256_D * v_out_stride;
    }
    // tail case
    if (n & 1)
    {
        __m128d v_in0, v_in1, v_in2, v_in3;
        __m128d v_av1, v_av2;
        __m128d v_out0, v_out1, v_out2, v_out3;

        __m128d v_K1 = _mm256_castpd256_pd128(v_C1);

        curr_in = in_r;
        curr_out = out_r;

        LD_128_D(curr_in, v_in0);
        curr_in = in_r + in_strides[1];
        LD_128_D(curr_in, v_in1);
        curr_in = in_r + in_strides[2];
        LD_128_D(curr_in, v_in2);
        curr_in = in_r + in_strides[3];
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
        curr_out = out_r + out_strides[1];
        ST_128_D(curr_out, v_out1);
        curr_out = out_r + out_strides[2];
        ST_128_D(curr_out, v_out2);
        curr_out = out_r + out_strides[3];
        ST_128_D(curr_out, v_out3);
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_fft4avx256(UINT8 precision, UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return fft4avx256fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft4avx256fp64;
    }
    else
    {
        return NULL;
    }
}
