// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file complex_mul_avx256.c
 *
 *  @brief Complex elementwise multiplication kernel with AVX-256 operations
 *  using x86 SIMD intrinsics
 *
 *  This file contains the complex elementwise multiplication implementations
 *  using AVX256 SIMD operations for single-precision and double-precision
 *  inputs.
 *
 *  @author Srirammaswamy Srinivasan
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common.h"

static VOID complex_mul_fp32_avx256_fwd(VOID *out, VOID *a, VOID *b, INTP n)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FLOAT *ptr_out = (FLOAT *)out;
    const FLOAT *ptr_a = (const FLOAT *)a;
    const FLOAT *ptr_b = (const FLOAT *)b;
    INTP N = n / NUM_SETS_256_S;
    INTP remaining_sets = n % NUM_SETS_256_S;
    INTP count;

    for (count = 0; count < N; count++)
    {
        __m256 va = _mm256_loadu_ps(ptr_a);
        __m256 vb = _mm256_loadu_ps(ptr_b);

        vb = _mm256_xor_ps(vb, _conj_256_f.s);

        __m256 va_re = BROADCAST_RE_256_S(va);
        __m256 va_im = BROADCAST_IM_256_S(va);
        __m256 vb_swap = SWAP_RI_256_S(vb);

        __m256 t1 = _mm256_mul_ps(va_re, vb);
        __m256 t2 = _mm256_mul_ps(va_im, vb_swap);

        __m256 result = _mm256_addsub_ps(t1, t2);

        _mm256_storeu_ps(ptr_out, result);

        ptr_a += NUM_SETS_256_S * DATA_STRIDE;
        ptr_b += NUM_SETS_256_S * DATA_STRIDE;
        ptr_out += NUM_SETS_256_S * DATA_STRIDE;
    }
    // tail cases
    if (remaining_sets & NUM_SETS_128_S)
    {
        __m128 va = _mm_loadu_ps(ptr_a);
        __m128 vb = _mm_loadu_ps(ptr_b);

        vb = _mm_xor_ps(vb, _conj_128_f.s);

        __m128 va_re = BROADCAST_RE_128_S(va);
        __m128 va_im = BROADCAST_IM_128_S(va);
        __m128 vb_swap = SWAP_RI_128_S(vb);

        __m128 t1 = _mm_mul_ps(va_re, vb);
        __m128 t2 = _mm_mul_ps(va_im, vb_swap);

        __m128 result = _mm_addsub_ps(t1, t2);

        _mm_storeu_ps(ptr_out, result);

        ptr_a += NUM_SETS_128_S * DATA_STRIDE;
        ptr_b += NUM_SETS_128_S * DATA_STRIDE;
        ptr_out += NUM_SETS_128_S * DATA_STRIDE;
    }
    // tail cases
    if (remaining_sets & NUM_SETS_C_S)
    {
        FLOAT a_re = ptr_a[0];
        FLOAT a_im = ptr_a[1];
        FLOAT b_re = ptr_b[0];
        FLOAT b_im = -ptr_b[1];

        ptr_out[0] = (a_re * b_re) - (a_im * b_im);
        ptr_out[1] = (a_re * b_im) + (a_im * b_re);
    }
}

static VOID complex_mul_fp32_avx256_bwd(VOID *out, VOID *a, VOID *b, INTP n)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FLOAT *ptr_out = (FLOAT *)out;
    const FLOAT *ptr_a = (const FLOAT *)a;
    const FLOAT *ptr_b = (const FLOAT *)b;
    INTP N = n / NUM_SETS_256_S;
    INTP remaining_sets = n % NUM_SETS_256_S;
    INTP count;

    for (count = 0; count < N; count++)
    {
        __m256 va = _mm256_loadu_ps(ptr_a);
        __m256 vb = _mm256_loadu_ps(ptr_b);

        __m256 va_re = BROADCAST_RE_256_S(va);
        __m256 va_im = BROADCAST_IM_256_S(va);
        __m256 vb_swap = SWAP_RI_256_S(vb);

        __m256 t1 = _mm256_mul_ps(va_re, vb);
        __m256 t2 = _mm256_mul_ps(va_im, vb_swap);

        __m256 result = _mm256_addsub_ps(t1, t2);

        _mm256_storeu_ps(ptr_out, result);

        ptr_a += NUM_SETS_256_S * DATA_STRIDE;
        ptr_b += NUM_SETS_256_S * DATA_STRIDE;
        ptr_out += NUM_SETS_256_S * DATA_STRIDE;
    }
    // tail cases
    if (remaining_sets & NUM_SETS_128_S)
    {
        __m128 va = _mm_loadu_ps(ptr_a);
        __m128 vb = _mm_loadu_ps(ptr_b);

        __m128 va_re = BROADCAST_RE_128_S(va);
        __m128 va_im = BROADCAST_IM_128_S(va);
        __m128 vb_swap = SWAP_RI_128_S(vb);

        __m128 t1 = _mm_mul_ps(va_re, vb);
        __m128 t2 = _mm_mul_ps(va_im, vb_swap);

        __m128 result = _mm_addsub_ps(t1, t2);

        _mm_storeu_ps(ptr_out, result);

        ptr_a += NUM_SETS_128_S * DATA_STRIDE;
        ptr_b += NUM_SETS_128_S * DATA_STRIDE;
        ptr_out += NUM_SETS_128_S * DATA_STRIDE;
    }
    // tail cases
    if (remaining_sets & NUM_SETS_C_S)
    {
        FLOAT a_re = ptr_a[0];
        FLOAT a_im = ptr_a[1];
        FLOAT b_re = ptr_b[0];
        FLOAT b_im = ptr_b[1];

        ptr_out[0] = (a_re * b_re) - (a_im * b_im);
        ptr_out[1] = (a_re * b_im) + (a_im * b_re);
    }
}

static VOID complex_mul_fp64_avx256_fwd(VOID *out, VOID *a, VOID *b, INTP n)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    DOUBLE *ptr_out = (DOUBLE *)out;
    const DOUBLE *ptr_a = (const DOUBLE *)a;
    const DOUBLE *ptr_b = (const DOUBLE *)b;
    INTP N = n / NUM_SETS_256_D;
    INTP remaining_sets = n % NUM_SETS_256_D;
    INTP count;

    for (count = 0; count < N; count++)
    {
        __m256d va = _mm256_loadu_pd(ptr_a);
        __m256d vb = _mm256_loadu_pd(ptr_b);

        vb = _mm256_xor_pd(vb, _conj_256_d.d);

        __m256d va_re = BROADCAST_RE_256_D(va);
        __m256d va_im = BROADCAST_IM_256_D(va);
        __m256d vb_swap = SWAP_RI_256_D(vb);

        __m256d t1 = _mm256_mul_pd(va_re, vb);
        __m256d t2 = _mm256_mul_pd(va_im, vb_swap);

        __m256d result = _mm256_addsub_pd(t1, t2);

        _mm256_storeu_pd(ptr_out, result);

        ptr_a += NUM_SETS_256_D * DATA_STRIDE;
        ptr_b += NUM_SETS_256_D * DATA_STRIDE;
        ptr_out += NUM_SETS_256_D * DATA_STRIDE;
    }
    // tail cases
    if (remaining_sets & NUM_SETS_C_D)
    {
        DOUBLE a_re = ptr_a[0];
        DOUBLE a_im = ptr_a[1];
        DOUBLE b_re = ptr_b[0];
        DOUBLE b_im = -ptr_b[1];

        ptr_out[0] = (a_re * b_re) - (a_im * b_im);
        ptr_out[1] = (a_re * b_im) + (a_im * b_re);
    }
}

static VOID complex_mul_fp64_avx256_bwd(VOID *out, VOID *a, VOID *b, INTP n)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    DOUBLE *ptr_out = (DOUBLE *)out;
    const DOUBLE *ptr_a = (const DOUBLE *)a;
    const DOUBLE *ptr_b = (const DOUBLE *)b;
    INTP N = n / NUM_SETS_256_D;
    INTP remaining_sets = n % NUM_SETS_256_D;
    INTP count;

    for (count = 0; count < N; count++)
    {
        __m256d va = _mm256_loadu_pd(ptr_a);
        __m256d vb = _mm256_loadu_pd(ptr_b);

        __m256d va_re = BROADCAST_RE_256_D(va);
        __m256d va_im = BROADCAST_IM_256_D(va);
        __m256d vb_swap = SWAP_RI_256_D(vb);

        __m256d t1 = _mm256_mul_pd(va_re, vb);
        __m256d t2 = _mm256_mul_pd(va_im, vb_swap);

        __m256d result = _mm256_addsub_pd(t1, t2);

        _mm256_storeu_pd(ptr_out, result);

        ptr_a += NUM_SETS_256_D * DATA_STRIDE;
        ptr_b += NUM_SETS_256_D * DATA_STRIDE;
        ptr_out += NUM_SETS_256_D * DATA_STRIDE;
    }
    // tail cases
    if (remaining_sets & NUM_SETS_C_D)
    {
        DOUBLE a_re = ptr_a[0];
        DOUBLE a_im = ptr_a[1];
        DOUBLE b_re = ptr_b[0];
        DOUBLE b_im = ptr_b[1];

        ptr_out[0] = (a_re * b_re) - (a_im * b_im);
        ptr_out[1] = (a_re * b_im) + (a_im * b_re);
    }
}

elementwise_mul_ register_elementwise_mul_avx256(UINT8 precision,
                                                 UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return complex_mul_fp32_avx256_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return complex_mul_fp64_avx256_fwd;
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
            return complex_mul_fp32_avx256_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return complex_mul_fp64_avx256_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
