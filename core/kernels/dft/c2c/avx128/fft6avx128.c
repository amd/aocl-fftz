// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fft6avx128.c
 *
 *  @brief Radix-6 FFT kernel with avx128 operations using x86 SIMD intrinsics
 *
 *  This file contains the DIT radix-6 FFT implementations using avx128
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Varun Sanjay
 *  @author S. Biplab Raut
 *
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 4, 18, 24, 2, 2},
                                                     {0, 4, 18, 12, 2, 2}};

ops_cycles_t get_ops_cnt_fft6avx128(UINT8 precision, UINT8 direction)
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

static VOID fft6avx128fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                           VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                           VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_6[2] = {0.500000000000000000000000000000000000000000000,
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
    UINT8 is_contiguous_in = (v_in_stride == DATA_STRIDE);
    INTP v_out_stride = strides->v_out_stride;
    UINT8 is_contiguous_out = (v_out_stride == DATA_STRIDE);
    INTP N = n / NUM_SETS_128_S;
    INTP count;
    FLOAT *curr_in, *curr_out;

    __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
    __m128 v_av1, v_av2, v_av3, v_av4, v_cv1, v_cv2, v_cv3;
    __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

    __m128 v_K1 = _mm_broadcast_ss(&CRTM_6[0]);
    __m128 v_K2 = _mm_broadcast_ss(&CRTM_6[1]);
    v_K2 = _mm_xor_ps(v_K2, _neg_128_f[flag].s);

    for (count = 0; count < N; count++)
    {
        curr_in = in_r;
        curr_out = out_r;

        GATHER2_128_S(curr_in, v_in_stride, v_in0, is_contiguous_in);
        curr_in = in_r + in_strides[1];
        GATHER2_128_S(curr_in, v_in_stride, v_in1, is_contiguous_in);
        curr_in = in_r + in_strides[2];
        GATHER2_128_S(curr_in, v_in_stride, v_in2, is_contiguous_in);
        curr_in = in_r + in_strides[3];
        GATHER2_128_S(curr_in, v_in_stride, v_in3, is_contiguous_in);
        curr_in = in_r + in_strides[4];
        GATHER2_128_S(curr_in, v_in_stride, v_in4, is_contiguous_in);
        curr_in = in_r + in_strides[5];
        GATHER2_128_S(curr_in, v_in_stride, v_in5, is_contiguous_in);

        v_av1 = _mm_add_ps(v_in0, v_in3);
        v_av2 = _mm_add_ps(v_in2, v_in4);
        v_av3 = _mm_add_ps(v_in1, v_in5);
        v_av4 = _mm_add_ps(v_av2, v_av3);
        v_cv1 = _mm_sub_ps(v_av1, _mm_mul_ps(v_K1, v_av4));

        // Output point 1:X[0]
        v_out0 = _mm_add_ps(v_av1, v_av4);

        v_av1 = _mm_sub_ps(v_av2, v_av3);
        v_av2 = _mm_sub_ps(v_in0, v_in3);
        v_cv2 = _mm_sub_ps(v_av2, _mm_mul_ps(v_K1, v_av1));

        // Output point 4:X[3]
        v_out3 = _mm_add_ps(v_av2, v_av1);

        v_av1 = _mm_sub_ps(v_in1, v_in5);
        v_av2 = _mm_sub_ps(v_in2, v_in4);
        v_av3 = _mm_sub_ps(v_av1, v_av2);
        v_cv3 = _mm_mul_ps(v_K2, v_av3);
        v_cv3 = CONJ_128_S(SWAP_RI_128_S(v_cv3));

        // Output point 3:X[2]
        v_out2 = _mm_add_ps(v_cv1, v_cv3);
        // Output point 5:X[4]
        v_out4 = _mm_sub_ps(v_cv1, v_cv3);

        v_av3 = _mm_add_ps(v_av1, v_av2);
        v_cv3 = _mm_mul_ps(v_K2, v_av3);
        v_cv3 = CONJ_128_S(SWAP_RI_128_S(v_cv3));

        // Output point 2:X[1]
        v_out1 = _mm_add_ps(v_cv2, v_cv3);
        // Output point 6:X[5]
        v_out5 = _mm_sub_ps(v_cv2, v_cv3);

        SCATTER2_128_S(curr_out, v_out_stride, v_out0, is_contiguous_out);
        curr_out = out_r + out_strides[1];
        SCATTER2_128_S(curr_out, v_out_stride, v_out1, is_contiguous_out);
        curr_out = out_r + out_strides[2];
        SCATTER2_128_S(curr_out, v_out_stride, v_out2, is_contiguous_out);
        curr_out = out_r + out_strides[3];
        SCATTER2_128_S(curr_out, v_out_stride, v_out3, is_contiguous_out);
        curr_out = out_r + out_strides[4];
        SCATTER2_128_S(curr_out, v_out_stride, v_out4, is_contiguous_out);
        curr_out = out_r + out_strides[5];
        SCATTER2_128_S(curr_out, v_out_stride, v_out5, is_contiguous_out);

        in_r += NUM_SETS_128_S * v_in_stride;
        out_r += NUM_SETS_128_S * v_out_stride;
    }
    // tail case
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
        curr_in = in_r + in_strides[5];
        LD_LOW_128_S(curr_in, v_in5);

        v_av1 = _mm_add_ps(v_in0, v_in3);
        v_av2 = _mm_add_ps(v_in2, v_in4);
        v_av3 = _mm_add_ps(v_in1, v_in5);
        v_av4 = _mm_add_ps(v_av2, v_av3);
        v_cv1 = _mm_sub_ps(v_av1, _mm_mul_ps(v_K1, v_av4));

        // Output point 1:X[0]
        v_out0 = _mm_add_ps(v_av1, v_av4);

        v_av1 = _mm_sub_ps(v_av2, v_av3);
        v_av2 = _mm_sub_ps(v_in0, v_in3);
        v_cv2 = _mm_sub_ps(v_av2, _mm_mul_ps(v_K1, v_av1));

        // Output point 4:X[3]
        v_out3 = _mm_add_ps(v_av2, v_av1);

        v_av1 = _mm_sub_ps(v_in1, v_in5);
        v_av2 = _mm_sub_ps(v_in2, v_in4);
        v_av3 = _mm_sub_ps(v_av1, v_av2);
        v_cv3 = _mm_mul_ps(v_K2, v_av3);
        v_cv3 = CONJ_128_S(SWAP_RI_128_S(v_cv3));

        // Output point 3:X[2]
        v_out2 = _mm_add_ps(v_cv1, v_cv3);
        // Output point 5:X[4]
        v_out4 = _mm_sub_ps(v_cv1, v_cv3);

        v_av3 = _mm_add_ps(v_av1, v_av2);
        v_cv3 = _mm_mul_ps(v_K2, v_av3);
        v_cv3 = CONJ_128_S(SWAP_RI_128_S(v_cv3));

        // Output point 2:X[1]
        v_out1 = _mm_add_ps(v_cv2, v_cv3);
        // Output point 6:X[5]
        v_out5 = _mm_sub_ps(v_cv2, v_cv3);

        ST_LOW_128_S(curr_out, v_out0);
        curr_out = out_r + out_strides[1];
        ST_LOW_128_S(curr_out, v_out1);
        curr_out = out_r + out_strides[2];
        ST_LOW_128_S(curr_out, v_out2);
        curr_out = out_r + out_strides[3];
        ST_LOW_128_S(curr_out, v_out3);
        curr_out = out_r + out_strides[4];
        ST_LOW_128_S(curr_out, v_out4);
        curr_out = out_r + out_strides[5];
        ST_LOW_128_S(curr_out, v_out5);
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID fft6avx128fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                           VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                           VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_6[2] = {0.500000000000000000000000000000000000000000000,
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
    INTP N = n / NUM_SETS_128_D;
    INTP count;
    DOUBLE *curr_in, *curr_out;

    __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5;
    __m128d v_av1, v_av2, v_av3, v_av4, v_cv1, v_cv2, v_cv3;
    __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5;

    __m128d v_K1 = _mm_set1_pd(CRTM_6[0]);
    __m128d v_K2 = _mm_set1_pd(CRTM_6[1]);
    v_K2 = _mm_xor_pd(v_K2, _neg_128_d[flag].d);

    for (count = 0; count < N; count++)
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
        curr_in = in_r + in_strides[5];
        LD_128_D(curr_in, v_in5);

        v_av1 = _mm_add_pd(v_in0, v_in3);
        v_av2 = _mm_add_pd(v_in2, v_in4);
        v_av3 = _mm_add_pd(v_in1, v_in5);
        v_av4 = _mm_add_pd(v_av2, v_av3);
        v_cv1 = _mm_sub_pd(v_av1, _mm_mul_pd(v_K1, v_av4));

        // Output point 1:X[0]
        v_out0 = _mm_add_pd(v_av1, v_av4);

        v_av1 = _mm_sub_pd(v_av2, v_av3);
        v_av2 = _mm_sub_pd(v_in0, v_in3);
        v_cv2 = _mm_sub_pd(v_av2, _mm_mul_pd(v_K1, v_av1));

        // Output point 4:X[3]
        v_out3 = _mm_add_pd(v_av2, v_av1);

        v_av1 = _mm_sub_pd(v_in1, v_in5);
        v_av2 = _mm_sub_pd(v_in2, v_in4);
        v_av3 = _mm_sub_pd(v_av1, v_av2);
        v_cv3 = _mm_mul_pd(v_K2, v_av3);
        v_cv3 = CONJ_128_D(SWAP_RI_128_D(v_cv3));

        // Output point 3:X[2]
        v_out2 = _mm_add_pd(v_cv1, v_cv3);
        // Output point 5:X[4]
        v_out4 = _mm_sub_pd(v_cv1, v_cv3);

        v_av3 = _mm_add_pd(v_av1, v_av2);
        v_cv3 = _mm_mul_pd(v_K2, v_av3);
        v_cv3 = CONJ_128_D(SWAP_RI_128_D(v_cv3));

        // Output point 2:X[1]
        v_out1 = _mm_add_pd(v_cv2, v_cv3);
        // Output point 6:X[5]
        v_out5 = _mm_sub_pd(v_cv2, v_cv3);

        ST_128_D(curr_out, v_out0);
        curr_out = out_r + out_strides[1];
        ST_128_D(curr_out, v_out1);
        curr_out = out_r + out_strides[2];
        ST_128_D(curr_out, v_out2);
        curr_out = out_r + out_strides[3];
        ST_128_D(curr_out, v_out3);
        curr_out = out_r + out_strides[4];
        ST_128_D(curr_out, v_out4);
        curr_out = out_r + out_strides[5];
        ST_128_D(curr_out, v_out5);

        in_r += NUM_SETS_128_D * v_in_stride;
        out_r += NUM_SETS_128_D * v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_fft6avx128(UINT8 precision, UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return fft6avx128fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft6avx128fp64;
    }
    else
    {
        return NULL;
    }
}
