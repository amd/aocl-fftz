// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file ele_mul_avx128.c
 *
 *  @brief Complex elementwise multiplication kernel with AVX-128 operations
 *  using x86 SIMD intrinsics
 *
 *  This file contains the complex elementwise multiplication implementations
 *  using AVX128 SIMD operations for single-precision and double-precision
 *  inputs.
 *
 *  @author Srirammaswamy Srinivasan
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common.h"

static FFTZ_VOID elementwise_mul_fp32_avx128_fwd(FFTZ_VOID *out, FFTZ_VOID *a,
                                                 FFTZ_VOID *b, FFTZ_INTP n,
                                                 FFTZ_INTP start_idx,
                                                 FFTZ_INTP stride)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FFTZ_FLOAT *ptr_out = (FFTZ_FLOAT *)out;
    const FFTZ_FLOAT *ptr_a = (const FFTZ_FLOAT *)a;
    const FFTZ_FLOAT *ptr_b = (const FFTZ_FLOAT *)b;
    FFTZ_INTP N = n / NUM_SETS_128_S;
    FFTZ_INTP count;

    for (count = 0; count < N; count++)
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
    if (n & 1)
    {
        FFTZ_FLOAT a_re = ptr_a[0];
        FFTZ_FLOAT a_im = ptr_a[1];
        FFTZ_FLOAT b_re = ptr_b[0];
        FFTZ_FLOAT b_im = -ptr_b[1];

        ptr_out[0] = (a_re * b_re) - (a_im * b_im);
        ptr_out[1] = (a_re * b_im) + (a_im * b_re);
    }
}

static FFTZ_VOID elementwise_mul_fp32_avx128_bwd(FFTZ_VOID *out, FFTZ_VOID *a,
                                                 FFTZ_VOID *b, FFTZ_INTP n,
                                                 FFTZ_INTP start_idx,
                                                 FFTZ_INTP stride)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FFTZ_FLOAT *ptr_out = (FFTZ_FLOAT *)out;
    const FFTZ_FLOAT *ptr_a = (const FFTZ_FLOAT *)a;
    const FFTZ_FLOAT *ptr_b = (const FFTZ_FLOAT *)b;
    FFTZ_INTP N = n / NUM_SETS_128_S;
    FFTZ_INTP count;

    for (count = 0; count < N; count++)
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
    if (n & 1)
    {
        FFTZ_FLOAT a_re = ptr_a[0];
        FFTZ_FLOAT a_im = ptr_a[1];
        FFTZ_FLOAT b_re = ptr_b[0];
        FFTZ_FLOAT b_im = ptr_b[1];

        ptr_out[0] = (a_re * b_re) - (a_im * b_im);
        ptr_out[1] = (a_re * b_im) + (a_im * b_re);
    }
}

static FFTZ_VOID elementwise_mul_fp64_avx128_fwd(FFTZ_VOID *out, FFTZ_VOID *a,
                                                 FFTZ_VOID *b, FFTZ_INTP n,
                                                 FFTZ_INTP start_idx,
                                                 FFTZ_INTP stride)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FFTZ_DOUBLE *ptr_out = (FFTZ_DOUBLE *)out;
    const FFTZ_DOUBLE *ptr_a = (const FFTZ_DOUBLE *)a;
    const FFTZ_DOUBLE *ptr_b = (const FFTZ_DOUBLE *)b;
    FFTZ_INTP count;

    for (count = 0; count < n; count++)
    {
        __m128d va = _mm_loadu_pd(ptr_a);
        __m128d vb = _mm_loadu_pd(ptr_b);

        vb = _mm_xor_pd(vb, _conj_128_d.d);

        __m128d va_re = BROADCAST_RE_128_D(va);
        __m128d va_im = BROADCAST_IM_128_D(va);
        __m128d vb_swap = SWAP_RI_128_D(vb);

        __m128d t1 = _mm_mul_pd(va_re, vb);
        __m128d t2 = _mm_mul_pd(va_im, vb_swap);

        __m128d result = _mm_addsub_pd(t1, t2);

        _mm_storeu_pd(ptr_out, result);

        ptr_a += NUM_SETS_128_D * DATA_STRIDE;
        ptr_b += NUM_SETS_128_D * DATA_STRIDE;
        ptr_out += NUM_SETS_128_D * DATA_STRIDE;
    }
}

static FFTZ_VOID elementwise_mul_fp64_avx128_bwd(FFTZ_VOID *out, FFTZ_VOID *a,
                                                 FFTZ_VOID *b, FFTZ_INTP n,
                                                 FFTZ_INTP start_idx,
                                                 FFTZ_INTP stride)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FFTZ_DOUBLE *ptr_out = (FFTZ_DOUBLE *)out;
    const FFTZ_DOUBLE *ptr_a = (const FFTZ_DOUBLE *)a;
    const FFTZ_DOUBLE *ptr_b = (const FFTZ_DOUBLE *)b;
    FFTZ_INTP count;

    for (count = 0; count < n; count++)
    {
        __m128d va = _mm_loadu_pd(ptr_a);
        __m128d vb = _mm_loadu_pd(ptr_b);

        __m128d va_re = BROADCAST_RE_128_D(va);
        __m128d va_im = BROADCAST_IM_128_D(va);
        __m128d vb_swap = SWAP_RI_128_D(vb);

        __m128d t1 = _mm_mul_pd(va_re, vb);
        __m128d t2 = _mm_mul_pd(va_im, vb_swap);

        __m128d result = _mm_addsub_pd(t1, t2);

        _mm_storeu_pd(ptr_out, result);

        ptr_a += NUM_SETS_128_D * DATA_STRIDE;
        ptr_b += NUM_SETS_128_D * DATA_STRIDE;
        ptr_out += NUM_SETS_128_D * DATA_STRIDE;
    }
}

static FFTZ_VOID elementwise_mul_strided_in_fp32_avx128_fwd(
    FFTZ_VOID *out, FFTZ_VOID *a, FFTZ_VOID *b, FFTZ_INTP n,
    FFTZ_INTP start_idx, FFTZ_INTP stride)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FFTZ_FLOAT *ptr_out = (FFTZ_FLOAT *)out;
    const FFTZ_FLOAT *ptr_a = (const FFTZ_FLOAT *)a;
    const FFTZ_FLOAT *ptr_b = (const FFTZ_FLOAT *)b;
    FFTZ_INTP in_stride = stride * DATA_STRIDE;
    FFTZ_INTP N = n / NUM_SETS_128_S;
    FFTZ_INTP i = 0;
    FFTZ_INTP count;

    for (count = 0; count < N; count++)
    {
        __m128 vb = _mm_loadu_ps(ptr_b + i * DATA_STRIDE);

        vb = _mm_xor_ps(vb, _conj_128_f.s);

        const FFTZ_FLOAT *va_base = ptr_a + (start_idx + i) * in_stride;
        __m128 va;

        GATHER2_128_S(va_base, in_stride, va, 0);

        __m128 va_re = BROADCAST_RE_128_S(va);
        __m128 va_im = BROADCAST_IM_128_S(va);
        __m128 vb_swap = SWAP_RI_128_S(vb);

        __m128 t1 = _mm_mul_ps(va_re, vb);
        __m128 t2 = _mm_mul_ps(va_im, vb_swap);
        __m128 result = _mm_addsub_ps(t1, t2);

        _mm_storeu_ps(ptr_out + i * DATA_STRIDE, result);
        i += NUM_SETS_128_S;
    }
    // tail cases
    if (n & 1)
    {
        FFTZ_INTP in_idx = (start_idx + i) * in_stride;
        FFTZ_FLOAT a_re = ptr_a[in_idx];
        FFTZ_FLOAT a_im = ptr_a[in_idx + 1];
        FFTZ_FLOAT b_re = ptr_b[i * DATA_STRIDE];
        FFTZ_FLOAT b_im = ptr_b[i * DATA_STRIDE + 1];

        ptr_out[i * DATA_STRIDE] = (a_re * b_re) + (a_im * b_im);
        ptr_out[i * DATA_STRIDE + 1] = (a_im * b_re) - (a_re * b_im);
    }
}

static FFTZ_VOID elementwise_mul_strided_in_fp32_avx128_bwd(
    FFTZ_VOID *out, FFTZ_VOID *a, FFTZ_VOID *b, FFTZ_INTP n,
    FFTZ_INTP start_idx, FFTZ_INTP stride)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FFTZ_FLOAT *ptr_out = (FFTZ_FLOAT *)out;
    const FFTZ_FLOAT *ptr_a = (const FFTZ_FLOAT *)a;
    const FFTZ_FLOAT *ptr_b = (const FFTZ_FLOAT *)b;
    FFTZ_INTP in_stride = stride * DATA_STRIDE;
    FFTZ_INTP N = n / NUM_SETS_128_S;
    FFTZ_INTP i = 0;
    FFTZ_INTP count;

    for (count = 0; count < N; count++)
    {
        __m128 vb = _mm_loadu_ps(ptr_b + i * DATA_STRIDE);

        const FFTZ_FLOAT *va_base = ptr_a + (start_idx + i) * in_stride;
        __m128 va;

        GATHER2_128_S(va_base, in_stride, va, 0);

        __m128 va_re = BROADCAST_RE_128_S(va);
        __m128 va_im = BROADCAST_IM_128_S(va);
        __m128 vb_swap = SWAP_RI_128_S(vb);

        __m128 t1 = _mm_mul_ps(va_re, vb);
        __m128 t2 = _mm_mul_ps(va_im, vb_swap);
        __m128 result = _mm_addsub_ps(t1, t2);

        _mm_storeu_ps(ptr_out + i * DATA_STRIDE, result);
        i += NUM_SETS_128_S;
    }
    // tail cases
    if (n & 1)
    {
        FFTZ_INTP in_idx = (start_idx + i) * in_stride;
        FFTZ_FLOAT a_re = ptr_a[in_idx];
        FFTZ_FLOAT a_im = ptr_a[in_idx + 1];
        FFTZ_FLOAT b_re = ptr_b[i * DATA_STRIDE];
        FFTZ_FLOAT b_im = ptr_b[i * DATA_STRIDE + 1];

        ptr_out[i * DATA_STRIDE] = (a_re * b_re) - (a_im * b_im);
        ptr_out[i * DATA_STRIDE + 1] = (a_re * b_im) + (a_im * b_re);
    }
}

static FFTZ_VOID elementwise_mul_strided_in_fp64_avx128_fwd(
    FFTZ_VOID *out, FFTZ_VOID *a, FFTZ_VOID *b, FFTZ_INTP n,
    FFTZ_INTP start_idx, FFTZ_INTP stride)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FFTZ_DOUBLE *ptr_out = (FFTZ_DOUBLE *)out;
    const FFTZ_DOUBLE *ptr_a = (const FFTZ_DOUBLE *)a;
    const FFTZ_DOUBLE *ptr_b = (const FFTZ_DOUBLE *)b;
    FFTZ_INTP in_stride = stride * DATA_STRIDE;
    FFTZ_INTP count;

    for (count = 0; count < n; count++)
    {
        FFTZ_INTP i = count;
        __m128d vb = _mm_loadu_pd(ptr_b + i * DATA_STRIDE);
        vb = _mm_xor_pd(vb, _conj_128_d.d);

        __m128d va = _mm_loadu_pd(ptr_a + (start_idx + i) * in_stride);

        __m128d va_re = BROADCAST_RE_128_D(va);
        __m128d va_im = BROADCAST_IM_128_D(va);
        __m128d vb_swap = SWAP_RI_128_D(vb);
        __m128d t1 = _mm_mul_pd(va_re, vb);
        __m128d t2 = _mm_mul_pd(va_im, vb_swap);
        __m128d result = _mm_addsub_pd(t1, t2);

        _mm_storeu_pd(ptr_out + i * DATA_STRIDE, result);
    }
}

static FFTZ_VOID elementwise_mul_strided_in_fp64_avx128_bwd(
    FFTZ_VOID *out, FFTZ_VOID *a, FFTZ_VOID *b, FFTZ_INTP n,
    FFTZ_INTP start_idx, FFTZ_INTP stride)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FFTZ_DOUBLE *ptr_out = (FFTZ_DOUBLE *)out;
    const FFTZ_DOUBLE *ptr_a = (const FFTZ_DOUBLE *)a;
    const FFTZ_DOUBLE *ptr_b = (const FFTZ_DOUBLE *)b;
    FFTZ_INTP in_stride = stride * DATA_STRIDE;
    FFTZ_INTP count;

    for (count = 0; count < n; count++)
    {
        FFTZ_INTP i = count;
        __m128d vb = _mm_loadu_pd(ptr_b + i * DATA_STRIDE);
        __m128d va = _mm_loadu_pd(ptr_a + (start_idx + i) * in_stride);

        __m128d va_re = BROADCAST_RE_128_D(va);
        __m128d va_im = BROADCAST_IM_128_D(va);
        __m128d vb_swap = SWAP_RI_128_D(vb);
        __m128d t1 = _mm_mul_pd(va_re, vb);
        __m128d t2 = _mm_mul_pd(va_im, vb_swap);
        __m128d result = _mm_addsub_pd(t1, t2);

        _mm_storeu_pd(ptr_out + i * DATA_STRIDE, result);
    }
}

elementwise_mul_ register_elementwise_mul_avx128(FFTZ_UINT8 precision,
                                                 FFTZ_UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return elementwise_mul_fp32_avx128_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return elementwise_mul_fp64_avx128_fwd;
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
            return elementwise_mul_fp32_avx128_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return elementwise_mul_fp64_avx128_bwd;
        }
        else
        {
            return NULL;
        }
    }
}

elementwise_mul_
register_elementwise_mul_strided_in_avx128(FFTZ_UINT8 precision,
                                           FFTZ_UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return elementwise_mul_strided_in_fp32_avx128_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return elementwise_mul_strided_in_fp64_avx128_fwd;
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
            return elementwise_mul_strided_in_fp32_avx128_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return elementwise_mul_strided_in_fp64_avx128_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
