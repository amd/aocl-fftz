// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft3avx128.c
 *
 *  @brief Radix-3 r2hc Real-FFT kernel with AVX-128 operations using x86 SIMD
 *  intrinsics
 *
 *  This file contains the DIT radix-3 real-to-halfcomplex implementations using
 *  AVX128 SIMD operations for single-precision and double-precision inputs.
 *
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/r2hc_simd_avx128.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                    {{{0, 2, 4, 20, 15, 0},
                                                      {0, 1, 5, 20, 15, 0}},
                                                     {{0, 2, 4, 10, 2, 0},
                                                      {0, 1, 5, 10, 2, 0}}};

ops_cycles_t get_ops_cnt_r2hc_rfft3avx128(FFTZ_UINT8 precision,
                                          FFTZ_UINT8 direction)
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

static FFTZ_VOID r2hc_rfft3avx128_fp32_fwd(FFTZ_VOID *in_real,
                                           FFTZ_VOID *in_imag,
                                           FFTZ_VOID *out_real,
                                           FFTZ_VOID *out_imag, FFTZ_INTP n,
                                           aoclfftz_strides_t *strides,
                                           FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_3_1 =
        0.500000000000000000000000000000000000000000000f;
    const FFTZ_FLOAT CRTM_3_2 =
        0.866025403784438646763723170752936183471402627f;

    FFTZ_FLOAT *in = (FFTZ_FLOAT *)in_real;
    FFTZ_FLOAT *out = (FFTZ_FLOAT *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides  = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride  = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_UINT8 is_contiguous_in = (v_in_stride == 1);
    FFTZ_UINT8 is_contiguous_out = (v_out_stride == 1);

    FFTZ_INTP cnt;
    FFTZ_FLOAT *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_128_S;

    __m128 v_CRTM_3_1 = _mm_broadcast_ss(&CRTM_3_1);
    __m128 v_CRTM_3_2 = _mm_broadcast_ss(&CRTM_3_2);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m128 v_in0, v_in1, v_in2;
        __m128 v_av0, v_av1, v_tv0, v_tv1;
        __m128 v_out0, v_out1, v_out2;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_stride, v_in0, is_contiguous_in);
        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_128_S(curr_in, v_in_stride, v_in1, is_contiguous_in);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_128_S(curr_in, v_in_stride, v_in2, is_contiguous_in);

        v_av0 = _mm_add_ps(v_in1, v_in2);
        v_av1 = _mm_sub_ps(v_in2, v_in1);

        v_tv0 = _mm_mul_ps(v_CRTM_3_1, v_av0);
        v_tv1 = _mm_mul_ps(v_CRTM_3_2, v_av1);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_in0, v_av0);

        // Output point 2: X(1)
        v_out1 = _mm_sub_ps(v_in0, v_tv0);

        // Output point 3: X(2)
        v_out2 = v_tv1;

        STR_128_S(curr_out, v_out_stride, v_out0, is_contiguous_out);
        curr_out = out + out_strides[1];
        STRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);

        in  += v_in_stride * NUM_SETS_REAL_128_S;
        out += v_out_stride * NUM_SETS_REAL_128_S;
    }
    // tail cases
    if (n & 2)
    {
        __m128 v_in0, v_in1, v_in2;
        __m128 v_av0, v_av1, v_tv0, v_tv1;
        __m128 v_out0, v_out1, v_out2;

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

        v_av0 = _mm_add_ps(v_in1, v_in2);
        v_av1 = _mm_sub_ps(v_in2, v_in1);

        v_tv0 = _mm_mul_ps(v_CRTM_3_1, v_av0);
        v_tv1 = _mm_mul_ps(v_CRTM_3_2, v_av1);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_in0, v_av0);

        // Output point 2: X(1)
        v_out1 = _mm_sub_ps(v_in0, v_tv0);

        // Output point 3: X(2)
        v_out2 = v_tv1;

        STHR_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STHRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);

        in  = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        FFTZ_FLOAT v_in0, v_in1, v_in2;
        FFTZ_FLOAT v_av0, v_av1, v_tv0, v_tv1;

        // Input point 1: x(0)
        v_in0 = *in;
        // Input point 2: x(1)
        v_in1 = in[in_strides[1]];
        // Input point 3: x(2)
        v_in2 = in[in_strides[2]];

        v_av0 = v_in1 + v_in2;
        v_av1 = v_in2 - v_in1;

        v_tv0 = CRTM_3_1 * v_av0;
        v_tv1 = CRTM_3_2 * v_av1;

        // Output point 1: X(0)
        *out = v_in0 + v_av0;

        // Output point 2: X(1)
        out[out_strides[1]] = v_in0 - v_tv0;

        // Output point 3: X(2)
        out[out_strides[2]] = v_tv1;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft3avx128_fp32_bwd(FFTZ_VOID *in_real,
                                           FFTZ_VOID *in_imag,
                                           FFTZ_VOID *out_real,
                                           FFTZ_VOID *out_imag, FFTZ_INTP n,
                                           aoclfftz_strides_t *strides,
                                           FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_3_1 =
        1.732050807568877293527446341505872366942805254f;

    FFTZ_FLOAT *in = (FFTZ_FLOAT *)in_real;
    FFTZ_FLOAT *out = (FFTZ_FLOAT *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides  = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride  = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_UINT8 is_contiguous_in = (v_in_stride == 1);
    FFTZ_UINT8 is_contiguous_out = (v_out_stride == 1);

    FFTZ_INTP cnt;
    FFTZ_FLOAT *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_128_S;

    __m128 v_CRTM_3_1 = _mm_broadcast_ss(&CRTM_3_1);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m128 v_in0, v_in1, v_in2;
        __m128 v_av0, v_tv0;
        __m128 v_out0, v_out1, v_out2;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_stride, v_in0, is_contiguous_in);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_S(curr_in, v_in_stride, v_in1, v_in2);

        v_av0 = _mm_sub_ps(v_in0, v_in1);
        v_tv0 = _mm_mul_ps(v_CRTM_3_1, v_in2);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_in0, _mm_add_ps(v_in1, v_in1));

        // Output point 2: X(1)
        v_out1 = _mm_sub_ps(v_av0, v_tv0);

        // Output point 3: X(2)
        v_out2 = _mm_add_ps(v_av0, v_tv0);

        STR_128_S(curr_out, v_out_stride, v_out0, is_contiguous_out);
        curr_out = out + out_strides[1];
        STR_128_S(curr_out, v_out_stride, v_out1, is_contiguous_out);
        curr_out = out + out_strides[2];
        STR_128_S(curr_out, v_out_stride, v_out2, is_contiguous_out);

        in  += v_in_stride * NUM_SETS_REAL_128_S;
        out += v_out_stride * NUM_SETS_REAL_128_S;
    }
    // tail cases
    if (n & 2)
    {
        __m128 v_in0, v_in1, v_in2;
        __m128 v_av0, v_tv0;
        __m128 v_out0, v_out1, v_out2;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDHR_128_S(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDHRI_2x128_S(curr_in, v_in_stride, v_in1, v_in2);

        v_av0 = _mm_sub_ps(v_in0, v_in1);
        v_tv0 = _mm_mul_ps(v_CRTM_3_1, v_in2);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_in0 , _mm_add_ps(v_in1, v_in1));

        // Output point 2: X(1)
        v_out1 = _mm_sub_ps(v_av0, v_tv0);

        // Output point 3: X(2)
        v_out2 = _mm_add_ps(v_av0, v_tv0);

        STHR_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STHR_128_S(curr_out, v_out_stride, v_out1);
        curr_out = out + out_strides[2];
        STHR_128_S(curr_out, v_out_stride, v_out2);

        in  = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (n & 1)
    {
        FFTZ_FLOAT v_in0, v_in1, v_in2;
        FFTZ_FLOAT v_av0, v_tv0;

        // Input point 1: x(0)
        v_in0 = *in;
        // Input point 2: x(1)
        v_in1 = in[in_strides[1]];
        // Input point 3: x(2)
        v_in2 = in[in_strides[2]];

        v_av0 = v_in0 - v_in1;
        v_tv0 = CRTM_3_1 * v_in2;

        // Output point 1: X(0)
        *out = v_in0 + 2 * v_in1;

        // Output point 2: X(1)
        out[out_strides[1]] = v_av0 - v_tv0;

        // Output point 3: X(2)
        out[out_strides[2]] = v_av0 + v_tv0;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft3avx128_fp64_fwd(FFTZ_VOID *in_real,
                                           FFTZ_VOID *in_imag,
                                           FFTZ_VOID *out_real,
                                           FFTZ_VOID *out_imag, FFTZ_INTP n,
                                           aoclfftz_strides_t *strides,
                                           FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_3_1 =
        0.500000000000000000000000000000000000000000000;
    const FFTZ_DOUBLE CRTM_3_2 =
        0.866025403784438646763723170752936183471402627;

    FFTZ_DOUBLE *in = (FFTZ_DOUBLE *)in_real;
    FFTZ_DOUBLE *out = (FFTZ_DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides  = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride  = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_UINT8 is_contiguous_in = (v_in_stride == 1);
    FFTZ_UINT8 is_contiguous_out = (v_out_stride == 1);

    FFTZ_INTP cnt;
    FFTZ_DOUBLE *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_128_D;

    __m128d v_CRTM_3_1 = _mm_set1_pd(CRTM_3_1);
    __m128d v_CRTM_3_2 = _mm_set1_pd(CRTM_3_2);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m128d v_in0, v_in1, v_in2;
        __m128d v_av0, v_av1, v_tv0, v_tv1;
        __m128d v_out0, v_out1,v_out2;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_stride, v_in0, is_contiguous_in);
        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_128_D(curr_in, v_in_stride, v_in1, is_contiguous_in);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_128_D(curr_in, v_in_stride, v_in2, is_contiguous_in);

        v_av0 = _mm_add_pd(v_in1, v_in2);
        v_av1 = _mm_sub_pd(v_in2, v_in1);

        v_tv0 = _mm_mul_pd(v_CRTM_3_1, v_av0);
        v_tv1 = _mm_mul_pd(v_CRTM_3_2, v_av1);

        // Output point 1: X(0)
        v_out0 = _mm_add_pd(v_in0, v_av0);

        // Output point 2: X(1)
        v_out1 = _mm_sub_pd(v_in0, v_tv0);

        // Output point 3: X(2)
        v_out2 = v_tv1;

        STR_128_D(curr_out, v_out_stride, v_out0, is_contiguous_out);
        curr_out = out + out_strides[1];
        STRI_2x128_D(curr_out, v_out_stride, v_out1, v_out2);

        in += v_in_stride * NUM_SETS_REAL_128_D;
        out += v_out_stride * NUM_SETS_REAL_128_D;
    }
    // tail cases
    if (n & 1)
    {
        FFTZ_DOUBLE v_in0, v_in1, v_in2;
        FFTZ_DOUBLE v_av0, v_av1, v_tv0, v_tv1;

        // Input point 1: x(0)
        v_in0 = *in;
        // Input point 2: x(1)
        v_in1 = in[in_strides[1]];
        // Input point 3: x(2)
        v_in2 = in[in_strides[2]];

        v_av0 = v_in1 + v_in2;
        v_av1 = v_in2 - v_in1;

        v_tv0 = CRTM_3_1 * v_av0;
        v_tv1 = CRTM_3_2 * v_av1;

        // Output point 1: X(0)
        *out = v_in0 + v_av0;

        // Output point 2: X(1)
        out[out_strides[1]] = v_in0 - v_tv0;

        // Output point 3: X(2)
        out[out_strides[2]] = v_tv1;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft3avx128_fp64_bwd(FFTZ_VOID *in_real,
                                           FFTZ_VOID *in_imag,
                                           FFTZ_VOID *out_real,
                                           FFTZ_VOID *out_imag, FFTZ_INTP n,
                                           aoclfftz_strides_t *strides,
                                           FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_3_1 =
        1.732050807568877293527446341505872366942805254;

    FFTZ_DOUBLE *in = (FFTZ_DOUBLE *)in_real;
    FFTZ_DOUBLE *out = (FFTZ_DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides  = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride  = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_UINT8 is_contiguous_in = (v_in_stride == 1);
    FFTZ_UINT8 is_contiguous_out = (v_out_stride == 1);

    FFTZ_INTP cnt;
    FFTZ_DOUBLE *curr_in, *curr_out;
    FFTZ_INTP N = n / NUM_SETS_REAL_128_D;

    __m128d v_CRTM_3_1 = _mm_set1_pd(CRTM_3_1);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m128d v_in0, v_in1, v_in2;
        __m128d v_av0, v_tv0;
        __m128d v_out0, v_out1,v_out2;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_stride, v_in0, is_contiguous_in);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_D(curr_in, v_in_stride, v_in1, v_in2);

        v_av0 = _mm_sub_pd(v_in0, v_in1);
        v_tv0 = _mm_mul_pd(v_CRTM_3_1, v_in2);

        // Output point 1: X(0)
        v_out0 = _mm_add_pd(v_in0, _mm_add_pd(v_in1, v_in1));

        // Output point 2: X(1)
        v_out1 = _mm_sub_pd(v_av0, v_tv0);

        // Output point 3: X(2)
        v_out2 = _mm_add_pd(v_av0, v_tv0);

        STR_128_D(curr_out, v_out_stride, v_out0, is_contiguous_out);
        curr_out = out + out_strides[1];
        STR_128_D(curr_out, v_out_stride, v_out1, is_contiguous_out);
        curr_out = out + out_strides[2];
        STR_128_D(curr_out, v_out_stride, v_out2, is_contiguous_out);

        in  += v_in_stride * NUM_SETS_REAL_128_D;
        out += v_out_stride * NUM_SETS_REAL_128_D;
    }
    // tail cases
    if (n & 1)
    {
        FFTZ_DOUBLE v_in0, v_in1, v_in2;
        FFTZ_DOUBLE v_av0, v_tv0;

        // Input point 1: x(0)
        v_in0 = *in;
        // Input point 2: x(1)
        v_in1 = in[in_strides[1]];
        // Input point 3: x(2)
        v_in2 = in[in_strides[2]];

        v_av0 = v_in0 - v_in1;
        v_tv0 = CRTM_3_1 * v_in2;

        // Output point 1: X(0)
        *out = v_in0 + 2 * v_in1;

        // Output point 2: X(1)
        out[out_strides[1]] = v_av0 - v_tv0;

        // Output point 3: X(2)
        out[out_strides[2]] = v_av0 + v_tv0;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hc_rfft3avx128(FFTZ_UINT8 precision,
                                       FFTZ_UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hc_rfft3avx128_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft3avx128_fp64_fwd;
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
            return r2hc_rfft3avx128_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft3avx128_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
