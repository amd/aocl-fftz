// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft3avx512.c
 *
 *  @brief Radix-3 r2hc Real-FFT kernel with AVX-512 operations using x86 SIMD
 *  intrinsics
 *
 *  This file contains the DIT radix-3 real-to-halfcomplex implementations using
 *  AVX512 SIMD operations for single-precision and double-precision inputs.
 *
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/r2hc_simd_avx512.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                    {{{0, 2, 4, 80, 50, 18},
                                                      {0, 1, 5, 80, 58, 18}},
                                                     {{0, 2, 4, 40, 2, 18},
                                                      {0, 1, 5, 40, 2, 18}}};

ops_cycles_t get_ops_cnt_r2hc_rfft3avx512(UINT8 precision, UINT8 direction)
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

static VOID r2hc_rfft3avx512_fp32_fwd(VOID *in_real, VOID *in_imag,
                                      VOID *out_real, VOID *out_imag, INTP n,
                                      aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_3_1 = 0.500000000000000000000000000000000000000000000f;
    const FLOAT CRTM_3_2 = 0.866025403784438646763723170752936183471402627f;

    FLOAT *in = (FLOAT *)in_real;
    FLOAT *out = (FLOAT *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides  = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride  = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;

    INTP cnt;
    FLOAT *curr_in, *curr_out;
    INTP N = n / NUM_SETS_REAL_512_S;
    INTP remaining_sets = n % NUM_SETS_REAL_512_S;

    __m512 v_CRTM_3_1 = _mm512_set1_ps(CRTM_3_1);
    __m512 v_CRTM_3_2 = _mm512_set1_ps(CRTM_3_2);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m512 v_in0, v_in1, v_in2;
        __m512 v_av0, v_av1, v_tv0, v_tv1;
        __m512 v_out0, v_out1, v_out2;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_512_S(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_512_S(curr_in, v_in_stride, v_in1);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_512_S(curr_in, v_in_stride, v_in2);

        v_av0 = _mm512_add_ps(v_in1, v_in2);
        v_av1 = _mm512_sub_ps(v_in2, v_in1);

        v_tv0 = _mm512_mul_ps(v_CRTM_3_1, v_av0);
        v_tv1 = _mm512_mul_ps(v_CRTM_3_2, v_av1);

        // Output point 1: X(0)
        v_out0 = _mm512_add_ps(v_in0, v_av0);

        // Output point 2: X(1)
        v_out1 = _mm512_sub_ps(v_in0, v_tv0);

        // Output point 3: X(2)
        v_out2 = v_tv1;

        STR_512_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STRI_2x512_S(curr_out, v_out_stride, v_out1, v_out2);

        in  = in + (v_in_stride << 4);
        out = out + (v_out_stride << 4);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_256_S)
    {
        __m256 v_in0, v_in1, v_in2;
        __m256 v_av0, v_av1, v_tv0, v_tv1;
        __m256 v_out0, v_out1, v_out2;

        curr_in = in;
        curr_out = out;

        __m256 v256_CRTM_3_1 = _mm512_castps512_ps256(v_CRTM_3_1);
        __m256 v256_CRTM_3_2 = _mm512_castps512_ps256(v_CRTM_3_2);

        // Input point 1: x(0)
        LDR_256_S(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_256_S(curr_in, v_in_stride, v_in1);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_256_S(curr_in, v_in_stride, v_in2);

        v_av0 = _mm256_add_ps(v_in1, v_in2);
        v_av1 = _mm256_sub_ps(v_in2, v_in1);

        v_tv0 = _mm256_mul_ps(v256_CRTM_3_1, v_av0);
        v_tv1 = _mm256_mul_ps(v256_CRTM_3_2, v_av1);

        // Output point 1: X(0)
        v_out0 = _mm256_add_ps(v_in0, v_av0);

        // Output point 2: X(1)
        v_out1 = _mm256_sub_ps(v_in0, v_tv0);

        // Output point 3: X(2)
        v_out2 = v_tv1;

        STR_256_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STRI_2x256_S(curr_out, v_out_stride, v_out1, v_out2);

        in  = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_S)
    {
        __m128 v_in0, v_in1, v_in2;
        __m128 v_av0, v_av1, v_tv0, v_tv1;
        __m128 v_out0, v_out1, v_out2;

        curr_in = in;
        curr_out = out;

        __m128 v128_CRTM_3_1 = _mm512_castps512_ps128(v_CRTM_3_1);
        __m128 v128_CRTM_3_2 = _mm512_castps512_ps128(v_CRTM_3_2);

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_128_S(curr_in, v_in_stride, v_in1);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_128_S(curr_in, v_in_stride, v_in2);

        v_av0 = _mm_add_ps(v_in1, v_in2);
        v_av1 = _mm_sub_ps(v_in2, v_in1);

        v_tv0 = _mm_mul_ps(v128_CRTM_3_1, v_av0);
        v_tv1 = _mm_mul_ps(v128_CRTM_3_2, v_av1);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_in0, v_av0);

        // Output point 2: X(1)
        v_out1 = _mm_sub_ps(v_in0, v_tv0);

        // Output point 3: X(2)
        v_out2 = v_tv1;

        STR_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STRI_2x128_S(curr_out, v_out_stride, v_out1, v_out2);

        in  = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & 2)
    {
        __m128 v_in0, v_in1, v_in2;
        __m128 v_av0, v_av1, v_tv0, v_tv1;
        __m128 v_out0, v_out1, v_out2;

        __m128 v128_CRTM_3_1 = _mm512_castps512_ps128(v_CRTM_3_1);
        __m128 v128_CRTM_3_2 = _mm512_castps512_ps128(v_CRTM_3_2);

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

        v_tv0 = _mm_mul_ps(v128_CRTM_3_1, v_av0);
        v_tv1 = _mm_mul_ps(v128_CRTM_3_2, v_av1);

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
    if (remaining_sets & 1)
    {
        FLOAT v_in0, v_in1, v_in2;
        FLOAT v_av0, v_av1, v_tv0, v_tv1;

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

static VOID r2hc_rfft3avx512_fp32_bwd(VOID *in_real, VOID *in_imag,
                                      VOID *out_real, VOID *out_imag, INTP n,
                                      aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_3_1 = 1.732050807568877293527446341505872366942805254f;

    FLOAT *in = (FLOAT *)in_real;
    FLOAT *out = (FLOAT *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides  = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride  = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;

    INTP cnt;
    FLOAT *curr_in, *curr_out;
    INTP N = n / NUM_SETS_REAL_512_S;
    INTP remaining_sets = n % NUM_SETS_REAL_512_S;

    __m512 v_CRTM_3_1 = _mm512_set1_ps(CRTM_3_1);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m512 v_in0, v_in1, v_in2;
        __m512 v_av0, v_tv0;
        __m512 v_out0, v_out1, v_out2;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_512_S(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x512_S(curr_in, v_in_stride, v_in1, v_in2);

        v_av0 = _mm512_sub_ps(v_in0, v_in1);
        v_tv0 = _mm512_mul_ps(v_CRTM_3_1, v_in2);

        // Output point 1: X(0)
        v_out0 = _mm512_add_ps(v_in0, _mm512_add_ps(v_in1, v_in1));

        // Output point 2: X(1)
        v_out1 = _mm512_sub_ps(v_av0, v_tv0);

        // Output point 3: X(2)
        v_out2 = _mm512_add_ps(v_av0, v_tv0);

        STR_512_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STR_512_S(curr_out, v_out_stride, v_out1);
        curr_out = out + out_strides[2];
        STR_512_S(curr_out, v_out_stride, v_out2);

        in  = in + (v_in_stride << 4);
        out = out + (v_out_stride << 4);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_256_S)
    {
        __m256 v_in0, v_in1, v_in2;
        __m256 v_av0, v_tv0;
        __m256 v_out0, v_out1, v_out2;

        curr_in = in;
        curr_out = out;

        __m256 v256_CRTM_3_1 = _mm512_castps512_ps256(v_CRTM_3_1);

        // Input point 1: x(0)
        LDR_256_S(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x256_S(curr_in, v_in_stride, v_in1, v_in2);

        v_av0 = _mm256_sub_ps(v_in0, v_in1);
        v_tv0 = _mm256_mul_ps(v256_CRTM_3_1, v_in2);

        // Output point 1: X(0)
        v_out0 = _mm256_add_ps(v_in0, _mm256_add_ps(v_in1, v_in1));

        // Output point 2: X(1)
        v_out1 = _mm256_sub_ps(v_av0, v_tv0);

        // Output point 3: X(2)
        v_out2 = _mm256_add_ps(v_av0, v_tv0);

        STR_256_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STR_256_S(curr_out, v_out_stride, v_out1);
        curr_out = out + out_strides[2];
        STR_256_S(curr_out, v_out_stride, v_out2);

        in  = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_128_S)
    {
        __m128 v_in0, v_in1, v_in2;
        __m128 v_av0, v_tv0;
        __m128 v_out0, v_out1, v_out2;

        __m128 v128_CRTM_3_1 = _mm512_castps512_ps128(v_CRTM_3_1);

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_128_S(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_S(curr_in, v_in_stride, v_in1, v_in2);

        v_av0 = _mm_sub_ps(v_in0, v_in1);
        v_tv0 = _mm_mul_ps(v128_CRTM_3_1, v_in2);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_in0, _mm_add_ps(v_in1, v_in1));

        // Output point 2: X(1)
        v_out1 = _mm_sub_ps(v_av0, v_tv0);

        // Output point 3: X(2)
        v_out2 = _mm_add_ps(v_av0, v_tv0);

        STR_128_S(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STR_128_S(curr_out, v_out_stride, v_out1);
        curr_out = out + out_strides[2];
        STR_128_S(curr_out, v_out_stride, v_out2);

        in  = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & 2)
    {
        __m128 v_in0, v_in1, v_in2;
        __m128 v_av0, v_tv0;
        __m128 v_out0, v_out1, v_out2;

        __m128 v128_CRTM_3_1 = _mm512_castps512_ps128(v_CRTM_3_1);

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDHR_128_S(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDHRI_2x128_S(curr_in, v_in_stride, v_in1, v_in2);

        v_av0 = _mm_sub_ps(v_in0, v_in1);
        v_tv0 = _mm_mul_ps(v128_CRTM_3_1, v_in2);

        // Output point 1: X(0)
        v_out0 = _mm_add_ps(v_in0, _mm_add_ps(v_in1, v_in1));

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
    if (remaining_sets & 1)
    {
        FLOAT v_in0, v_in1, v_in2;
        FLOAT v_av0, v_tv0;

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

static VOID r2hc_rfft3avx512_fp64_fwd(VOID *in_real, VOID *in_imag,
                                      VOID *out_real, VOID *out_imag, INTP n,
                                      aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_3_1 = 0.500000000000000000000000000000000000000000000;
    const DOUBLE CRTM_3_2 = 0.866025403784438646763723170752936183471402627;

    DOUBLE *in = (DOUBLE *)in_real;
    DOUBLE *out = (DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides  = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride  = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;

    INTP cnt;
    DOUBLE *curr_in, *curr_out;
    INTP N = n / NUM_SETS_REAL_512_D;
    INTP remaining_sets = n % NUM_SETS_REAL_512_D;

    __m512d v_CRTM_3_1 = _mm512_set1_pd(CRTM_3_1);
    __m512d v_CRTM_3_2 = _mm512_set1_pd(CRTM_3_2);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m512d v_in0, v_in1, v_in2;
        __m512d v_av0, v_av1, v_tv0, v_tv1;
        __m512d v_out0, v_out1,v_out2;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_512_D(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_512_D(curr_in, v_in_stride, v_in1);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_512_D(curr_in, v_in_stride, v_in2);

        v_av0 = _mm512_add_pd(v_in1, v_in2);
        v_av1 = _mm512_sub_pd(v_in2, v_in1);

        v_tv0 = _mm512_mul_pd(v_CRTM_3_1, v_av0);
        v_tv1 = _mm512_mul_pd(v_CRTM_3_2, v_av1);

        // Output point 1: X(0)
        v_out0 = _mm512_add_pd(v_in0, v_av0);

        // Output point 2: X(1)
        v_out1 = _mm512_sub_pd(v_in0, v_tv0);

        // Output point 3: X(2)
        v_out2 = v_tv1;

        STR_512_D(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STRI_2x512_D(curr_out, v_out_stride, v_out1, v_out2);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_256_D)
    {
        __m256d v_in0, v_in1, v_in2;
        __m256d v_av0, v_av1, v_tv0, v_tv1;
        __m256d v_out0, v_out1,v_out2;

        curr_in = in;
        curr_out = out;

        __m256d v256_CRTM_3_1 = _mm512_castpd512_pd256(v_CRTM_3_1);
        __m256d v256_CRTM_3_2 = _mm512_castpd512_pd256(v_CRTM_3_2);

        // Input point 1: x(0)
        LDR_256_D(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_256_D(curr_in, v_in_stride, v_in1);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_256_D(curr_in, v_in_stride, v_in2);

        v_av0 = _mm256_add_pd(v_in1, v_in2);
        v_av1 = _mm256_sub_pd(v_in2, v_in1);

        v_tv0 = _mm256_mul_pd(v256_CRTM_3_1, v_av0);
        v_tv1 = _mm256_mul_pd(v256_CRTM_3_2, v_av1);

        // Output point 1: X(0)
        v_out0 = _mm256_add_pd(v_in0, v_av0);

        // Output point 2: X(1)
        v_out1 = _mm256_sub_pd(v_in0, v_tv0);

        // Output point 3: X(2)
        v_out2 = v_tv1;

        STR_256_D(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STRI_2x256_D(curr_out, v_out_stride, v_out1, v_out2);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & 2)
    {
        __m128d v_in0, v_in1, v_in2;
        __m128d v_av0, v_av1, v_tv0, v_tv1;
        __m128d v_out0, v_out1,v_out2;

        curr_in = in;
        curr_out = out;

        __m128d v128_CRTM_3_1 = _mm512_castpd512_pd128(v_CRTM_3_1);
        __m128d v128_CRTM_3_2 = _mm512_castpd512_pd128(v_CRTM_3_2);

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1)
        curr_in = in + in_strides[1];
        LDR_128_D(curr_in, v_in_stride, v_in1);
        // Input point 3: x(2)
        curr_in = in + in_strides[2];
        LDR_128_D(curr_in, v_in_stride, v_in2);

        v_av0 = _mm_add_pd(v_in1, v_in2);
        v_av1 = _mm_sub_pd(v_in2, v_in1);

        v_tv0 = _mm_mul_pd(v128_CRTM_3_1, v_av0);
        v_tv1 = _mm_mul_pd(v128_CRTM_3_2, v_av1);

        // Output point 1: X(0)
        v_out0 = _mm_add_pd(v_in0, v_av0);

        // Output point 2: X(1)
        v_out1 = _mm_sub_pd(v_in0, v_tv0);

        // Output point 3: X(2)
        v_out2 = v_tv1;

        STR_128_D(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STRI_2x128_D(curr_out, v_out_stride, v_out1, v_out2);

        in  = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        DOUBLE v_in0, v_in1, v_in2;
        DOUBLE v_av0, v_av1, v_tv0, v_tv1;

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

static VOID r2hc_rfft3avx512_fp64_bwd(VOID *in_real, VOID *in_imag,
                                      VOID *out_real, VOID *out_imag, INTP n,
                                      aoclfftz_strides_t *strides, VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_3_1 = 1.732050807568877293527446341505872366942805254;

    DOUBLE *in = (DOUBLE *)in_real;
    DOUBLE *out = (DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides  = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride  = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;

    INTP cnt;
    DOUBLE *curr_in, *curr_out;
    INTP N = n / NUM_SETS_REAL_512_D;
    INTP remaining_sets = n % NUM_SETS_REAL_512_D;

    __m512d v_CRTM_3_1 = _mm512_set1_pd(CRTM_3_1);

    for (cnt = 0; cnt < N; cnt++)
    {
        __m512d v_in0, v_in1, v_in2;
        __m512d v_av0, v_tv0;
        __m512d v_out0, v_out1,v_out2;

        curr_in = in;
        curr_out = out;

        // Input point 1: x(0)
        LDR_512_D(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x512_D(curr_in, v_in_stride, v_in1, v_in2);

        v_av0 = _mm512_sub_pd(v_in0, v_in1);
        v_tv0 = _mm512_mul_pd(v_CRTM_3_1, v_in2);

        // Output point 1: X(0)
        v_out0 = _mm512_add_pd(v_in0, _mm512_add_pd(v_in1, v_in1));

        // Output point 2: X(1)
        v_out1 = _mm512_sub_pd(v_av0, v_tv0);

        // Output point 3: X(2)
        v_out2 = _mm512_add_pd(v_av0, v_tv0);

        STR_512_D(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STR_512_D(curr_out, v_out_stride, v_out1);
        curr_out = out + out_strides[2];
        STR_512_D(curr_out, v_out_stride, v_out2);

        in = in + (v_in_stride << 3);
        out = out + (v_out_stride << 3);
    }
    // tail cases
    if (remaining_sets & NUM_SETS_REAL_256_D)
    {
        __m256d v_in0, v_in1, v_in2;
        __m256d v_av0, v_tv0;
        __m256d v_out0, v_out1,v_out2;

        curr_in = in;
        curr_out = out;

        __m256d v256_CRTM_3_1 = _mm512_castpd512_pd256(v_CRTM_3_1);

        // Input point 1: x(0)
        LDR_256_D(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x256_D(curr_in, v_in_stride, v_in1, v_in2);

        v_av0 = _mm256_sub_pd(v_in0, v_in1);
        v_tv0 = _mm256_mul_pd(v256_CRTM_3_1, v_in2);

        // Output point 1: X(0)
        v_out0 = _mm256_add_pd(v_in0, _mm256_add_pd(v_in1, v_in1));

        // Output point 2: X(1)
        v_out1 = _mm256_sub_pd(v_av0, v_tv0);

        // Output point 3: X(2)
        v_out2 = _mm256_add_pd(v_av0, v_tv0);

        STR_256_D(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STR_256_D(curr_out, v_out_stride, v_out1);
        curr_out = out + out_strides[2];
        STR_256_D(curr_out, v_out_stride, v_out2);

        in = in + (v_in_stride << 2);
        out = out + (v_out_stride << 2);
    }
    // tail cases
    if (remaining_sets & 2)
    {
        __m128d v_in0, v_in1, v_in2;
        __m128d v_av0, v_tv0;
        __m128d v_out0, v_out1, v_out2;

        curr_in = in;
        curr_out = out;

        __m128d v128_CRTM_3_1 = _mm512_castpd512_pd128(v_CRTM_3_1);

        // Input point 1: x(0)
        LDR_128_D(curr_in, v_in_stride, v_in0);
        // Input point 2: x(1) & Input point 3: x(2)
        curr_in = in + in_strides[1];
        LDRI_2x128_D(curr_in, v_in_stride, v_in1, v_in2);

        v_av0 = _mm_sub_pd(v_in0, v_in1);
        v_tv0 = _mm_mul_pd(v128_CRTM_3_1, v_in2);

        // Output point 1: X(0)
        v_out0 = _mm_add_pd(v_in0, _mm_add_pd(v_in1, v_in1));

        // Output point 2: X(1)
        v_out1 = _mm_sub_pd(v_av0, v_tv0);

        // Output point 3: X(2)
        v_out2 = _mm_add_pd(v_av0, v_tv0);

        STR_128_D(curr_out, v_out_stride, v_out0);
        curr_out = out + out_strides[1];
        STR_128_D(curr_out, v_out_stride, v_out1);
        curr_out = out + out_strides[2];
        STR_128_D(curr_out, v_out_stride, v_out2);

        in = in + (v_in_stride << 1);
        out = out + (v_out_stride << 1);
    }
    // tail cases
    if (remaining_sets & 1)
    {
        DOUBLE v_in0, v_in1, v_in2;
        DOUBLE v_av0, v_tv0;

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

kfft_ register_kernel_r2hc_rfft3avx512(UINT8 precision, UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hc_rfft3avx512_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft3avx512_fp64_fwd;
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
            return r2hc_rfft3avx512_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft3avx512_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
