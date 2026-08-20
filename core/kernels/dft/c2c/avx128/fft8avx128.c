// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fft8avx128.c
 *
 *  @brief Radix-8 FFT kernel with AVX-128 operations using x86 SIMD intrinsics
 *
 *  This file contains the DIT radix-8 FFT implementations using AVX128
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Jeya R
 *
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 4, 26, 32, 3, 3},
                                                     {0, 4, 26, 16, 3, 3}};

ops_cycles_t get_ops_cnt_fft8avx128(FFTZ_UINT8 precision, FFTZ_UINT8 direction)
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

static FFTZ_VOID fft8avx128fp32(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                FFTZ_INTP n, aoclfftz_strides_t *strides,
                                FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_8[2] = {1.0,
                             0.707106781186547524400844362104849039284835938};

    FFTZ_FLOAT *in_r = (FFTZ_FLOAT *)in_real;
    FFTZ_FLOAT *out_r = (FFTZ_FLOAT *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_UINT8 is_contiguous_in = (v_in_stride == DATA_STRIDE);
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_UINT8 is_contiguous_out = (v_out_stride == DATA_STRIDE);
    FFTZ_INTP N = n / NUM_SETS_128_S;
    FFTZ_INTP count;
    FFTZ_FLOAT *curr_in, *curr_out;

    __m128 v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7;
    __m128 v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7;
    __m128 v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8,
           v_tv1, v_tv2, v_tv3;
    __m128 v_cv1, v_cv2, v_cv3, v_cv4;

    __m128 v_K1 = _mm_broadcast_ss(&CRTM_8[0]);
    v_K1 = _mm_xor_ps(v_K1, _neg_128_f[flag].s);
    __m128 v_K2 = _mm_broadcast_ss(&CRTM_8[1]);
    __m128 v_K3 = _mm_xor_ps(v_K2, _neg_128_f[flag].s);

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
        curr_in = in_r + in_strides[6];
        GATHER2_128_S(curr_in, v_in_stride, v_in6, is_contiguous_in);
        curr_in = in_r + in_strides[7];
        GATHER2_128_S(curr_in, v_in_stride, v_in7, is_contiguous_in);

        // common operations
        v_av1 = _mm_add_ps(v_in0, v_in4);
        v_av2 = _mm_add_ps(v_in2, v_in6);
        v_av3 = _mm_add_ps(v_in1, v_in5);
        v_av4 = _mm_add_ps(v_in3, v_in7);

        v_av5 = _mm_sub_ps(v_in0, v_in4);
        v_av6 = _mm_sub_ps(v_in2, v_in6);
        v_av7 = _mm_sub_ps(v_in1, v_in5);
        v_av8 = _mm_sub_ps(v_in3, v_in7);

        v_cv1 = _mm_add_ps(v_av1, v_av2);
        v_cv2 = _mm_add_ps(v_av3, v_av4);

        // Output point 1
        v_out0 = _mm_add_ps(v_cv1, v_cv2);
        // Output point 5
        v_out4 = _mm_sub_ps(v_cv1, v_cv2);

        v_cv1 = _mm_sub_ps(v_av3, v_av4);
        v_cv2 = _mm_sub_ps(v_av1, v_av2);

        v_tv1 = _mm_mul_ps(v_K1, v_cv1);
        v_tv1 = CONJ_128_S(v_tv1);
        v_tv1 = SWAP_RI_128_S(v_tv1);

        // Output point 7
        v_out6 = _mm_add_ps(v_cv2, v_tv1);
        // Output point 3
        v_out2 = _mm_sub_ps(v_cv2, v_tv1);

        v_cv1 = _mm_sub_ps(v_av7, v_av8);
        v_tv1 = _mm_mul_ps(v_K2, v_cv1);

        v_cv1 = _mm_add_ps(v_av7, v_av8);
        v_tv2 = _mm_mul_ps(v_K3, v_cv1);
        v_tv3 = _mm_mul_ps(v_K1, v_av6);

        v_cv1 = _mm_sub_ps(v_tv3, v_tv2);
        v_cv2 = _mm_add_ps(v_tv3, v_tv2);

        v_cv1 = CONJ_128_S(v_cv1);
        v_cv1 = SWAP_RI_128_S(v_cv1);
        v_cv2 = CONJ_128_S(v_cv2);
        v_cv2 = SWAP_RI_128_S(v_cv2);

        v_cv3 = _mm_sub_ps(v_av5, v_tv1);
        v_cv4 = _mm_add_ps(v_av5, v_tv1);

        // Output point 2
        v_out1 = _mm_sub_ps(v_cv4, v_cv2);
        // Output point 8
        v_out7 = _mm_add_ps(v_cv4, v_cv2);

        // Output point 6
        v_out5 = _mm_sub_ps(v_cv3, v_cv1);
        // Output point 4
        v_out3 = _mm_add_ps(v_cv3, v_cv1);

        if (out_strides[1] == DATA_STRIDE)
        {
            TRANSPOSE_ST2_128_S(curr_out, v_out_stride, v_out0, v_out1);
            curr_out = curr_out + NUM_SETS_128_S * DATA_STRIDE;
            TRANSPOSE_ST2_128_S(curr_out, v_out_stride, v_out2, v_out3);
            curr_out = curr_out + NUM_SETS_128_S * DATA_STRIDE;
            TRANSPOSE_ST2_128_S(curr_out, v_out_stride, v_out4, v_out5);
            curr_out = curr_out + NUM_SETS_128_S * DATA_STRIDE;
            TRANSPOSE_ST2_128_S(curr_out, v_out_stride, v_out6, v_out7);
        }
        else
        {
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
            curr_out = out_r + out_strides[6];
            SCATTER2_128_S(curr_out, v_out_stride, v_out6, is_contiguous_out);
            curr_out = out_r + out_strides[7];
            SCATTER2_128_S(curr_out, v_out_stride, v_out7, is_contiguous_out);
        }

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
        curr_in = in_r + in_strides[6];
        LD_LOW_128_S(curr_in, v_in6);
        curr_in = in_r + in_strides[7];
        LD_LOW_128_S(curr_in, v_in7);

        // common operations
        v_av1 = _mm_add_ps(v_in0, v_in4);
        v_av2 = _mm_add_ps(v_in2, v_in6);
        v_av3 = _mm_add_ps(v_in1, v_in5);
        v_av4 = _mm_add_ps(v_in3, v_in7);

        v_av5 = _mm_sub_ps(v_in0, v_in4);
        v_av6 = _mm_sub_ps(v_in2, v_in6);
        v_av7 = _mm_sub_ps(v_in1, v_in5);
        v_av8 = _mm_sub_ps(v_in3, v_in7);

        v_cv1 = _mm_add_ps(v_av1, v_av2);
        v_cv2 = _mm_add_ps(v_av3, v_av4);

        // Output point 1
        v_out0 = _mm_add_ps(v_cv1, v_cv2);
        // Output point 5
        v_out4 = _mm_sub_ps(v_cv1, v_cv2);

        v_cv1 = _mm_sub_ps(v_av3, v_av4);
        v_cv2 = _mm_sub_ps(v_av1, v_av2);

        v_tv1 = _mm_mul_ps(v_K1, v_cv1);
        v_tv1 = CONJ_128_S(v_tv1);
        v_tv1 = SWAP_RI_128_S(v_tv1);

        // Output point 7
        v_out6 = _mm_add_ps(v_cv2, v_tv1);
        // Output point 3
        v_out2 = _mm_sub_ps(v_cv2, v_tv1);

        v_cv1 = _mm_sub_ps(v_av7, v_av8);
        v_tv1 = _mm_mul_ps(v_K2, v_cv1);

        v_cv1 = _mm_add_ps(v_av7, v_av8);
        v_tv2 = _mm_mul_ps(v_K3, v_cv1);
        v_tv3 = _mm_mul_ps(v_K1, v_av6);

        v_cv1 = _mm_sub_ps(v_tv3, v_tv2);
        v_cv2 = _mm_add_ps(v_tv3, v_tv2);

        v_cv1 = CONJ_128_S(v_cv1);
        v_cv1 = SWAP_RI_128_S(v_cv1);
        v_cv2 = CONJ_128_S(v_cv2);
        v_cv2 = SWAP_RI_128_S(v_cv2);

        v_cv3 = _mm_sub_ps(v_av5, v_tv1);
        v_cv4 = _mm_add_ps(v_av5, v_tv1);

        // Output point 2
        v_out1 = _mm_sub_ps(v_cv4, v_cv2);
        // Output point 8
        v_out7 = _mm_add_ps(v_cv4, v_cv2);

        // Output point 6
        v_out5 = _mm_sub_ps(v_cv3, v_cv1);
        // Output point 4
        v_out3 = _mm_add_ps(v_cv3, v_cv1);

        if (out_strides[1] == DATA_STRIDE)
        {
            ST_128_S(curr_out, v_out0, v_out1);
            curr_out = out_r + 2 * DATA_STRIDE;
            ST_128_S(curr_out, v_out2, v_out3);
            curr_out = out_r + 4 * DATA_STRIDE;
            ST_128_S(curr_out, v_out4, v_out5);
            curr_out = out_r + 6 * DATA_STRIDE;
            ST_128_S(curr_out, v_out6, v_out7);
        }
        else
        {
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
            curr_out = out_r + out_strides[6];
            ST_LOW_128_S(curr_out, v_out6);
            curr_out = out_r + out_strides[7];
            ST_LOW_128_S(curr_out, v_out7);
        }
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID fft8avx128fp64(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                FFTZ_INTP n, aoclfftz_strides_t *strides,
                                FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_8[2] = {1.0,
                              0.707106781186547524400844362104849039284835938};

    FFTZ_DOUBLE *in_r = (FFTZ_DOUBLE *)in_real;
    FFTZ_DOUBLE *out_r = (FFTZ_DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_INTP count;
    FFTZ_DOUBLE *curr_in, *curr_out;

    __m128d v_in0, v_in1, v_in2, v_in3, v_in4, v_in5, v_in6, v_in7;
    __m128d v_out0, v_out1, v_out2, v_out3, v_out4, v_out5, v_out6, v_out7;
    __m128d v_av1, v_av2, v_av3, v_av4, v_av5, v_av6, v_av7, v_av8,
            v_tv1, v_tv2, v_tv3;
    __m128d v_cv1, v_cv2, v_cv3, v_cv4;

    __m128d v_K1 = _mm_set1_pd(CRTM_8[0]);
    v_K1 = _mm_xor_pd(v_K1, _neg_128_d[flag].d);
    __m128d v_K2 = _mm_set1_pd(CRTM_8[1]);
    __m128d v_K3 = _mm_xor_pd(v_K2, _neg_128_d[flag].d);

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
        curr_in = in_r + in_strides[5];
        LD_128_D(curr_in, v_in5);
        curr_in = in_r + in_strides[6];
        LD_128_D(curr_in, v_in6);
        curr_in = in_r + in_strides[7];
        LD_128_D(curr_in, v_in7);

        // common operations
        v_av1 = _mm_add_pd(v_in0, v_in4);
        v_av2 = _mm_add_pd(v_in2, v_in6);
        v_av3 = _mm_add_pd(v_in1, v_in5);
        v_av4 = _mm_add_pd(v_in3, v_in7);

        v_av5 = _mm_sub_pd(v_in0, v_in4);
        v_av6 = _mm_sub_pd(v_in2, v_in6);
        v_av7 = _mm_sub_pd(v_in1, v_in5);
        v_av8 = _mm_sub_pd(v_in3, v_in7);

        v_cv1 = _mm_add_pd(v_av1, v_av2);
        v_cv2 = _mm_add_pd(v_av3, v_av4);

        // Output point 1
        v_out0 = _mm_add_pd(v_cv1, v_cv2);
        // Output point 5
        v_out4 = _mm_sub_pd(v_cv1, v_cv2);

        v_cv1 = _mm_sub_pd(v_av3, v_av4);
        v_cv2 = _mm_sub_pd(v_av1, v_av2);

        v_tv1 = _mm_mul_pd(v_K1, v_cv1);
        v_tv1 = CONJ_128_D(v_tv1);
        v_tv1 = SWAP_RI_128_D(v_tv1);

        // Output point 7
        v_out6 = _mm_add_pd(v_cv2, v_tv1);
        // Output point 3
        v_out2 = _mm_sub_pd(v_cv2, v_tv1);

        v_cv1 = _mm_sub_pd(v_av7, v_av8);
        v_tv1 = _mm_mul_pd(v_K2, v_cv1);

        v_cv1 = _mm_add_pd(v_av7, v_av8);
        v_tv2 = _mm_mul_pd(v_K3, v_cv1);
        v_tv3 = _mm_mul_pd(v_K1, v_av6);

        v_cv1 = _mm_sub_pd(v_tv3, v_tv2);
        v_cv2 = _mm_add_pd(v_tv3, v_tv2);

        v_cv1 = CONJ_128_D(v_cv1);
        v_cv1 = SWAP_RI_128_D(v_cv1);
        v_cv2 = CONJ_128_D(v_cv2);
        v_cv2 = SWAP_RI_128_D(v_cv2);

        v_cv3 = _mm_sub_pd(v_av5, v_tv1);
        v_cv4 = _mm_add_pd(v_av5, v_tv1);

        // output point 2
        v_out1 = _mm_sub_pd(v_cv4, v_cv2);
        // output point 8
        v_out7 = _mm_add_pd(v_cv4, v_cv2);

        // output point 6
        v_out5 = _mm_sub_pd(v_cv3, v_cv1);
        // output point 4
        v_out3 = _mm_add_pd(v_cv3, v_cv1);

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
        curr_out = out_r + out_strides[6];
        ST_128_D(curr_out, v_out6);
        curr_out = out_r + out_strides[7];
        ST_128_D(curr_out, v_out7);

        in_r += v_in_stride;
        out_r += v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_fft8avx128(FFTZ_UINT8 precision,
                                 FFTZ_UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return fft8avx128fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft8avx128fp64;
    }
    else
    {
        return NULL;
    }
}
