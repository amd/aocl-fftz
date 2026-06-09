// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file normalize_avx128.c
 *
 * @brief In-place complex buffer normalization with AVX128 SIMD operations.
 *
 * Multiplies all real and imaginary components of a complex buffer by a
 * real scalar factor (the 1/N scaling applied during inverse FFT).
 *
 * @author Jeevanantham N
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common.h"

static VOID normalize_fp32_avx128(VOID *data, INTP n, DOUBLE factor)
{
    FLOAT *ptr_data = (FLOAT *)data;
    INTP N = n / NUM_SETS_128_S;
    INTP remaining_sets = n % NUM_SETS_128_S;
    INTP count;

    __m128 vfactor = _mm_set1_ps((FLOAT)factor);

    for (count = 0; count < N; count++)
    {
        __m128 v = _mm_load_ps(ptr_data);
        v = _mm_mul_ps(v, vfactor);
        _mm_store_ps(ptr_data, v);

        ptr_data += NUM_SETS_128_S * DATA_STRIDE;
    }
    // tail cases
    if (remaining_sets & NUM_SETS_C_S)
    {
        const FLOAT factor_f = (FLOAT)factor;
        ptr_data[0] *= factor_f;
        ptr_data[1] *= factor_f;
    }
}

static VOID normalize_fp64_avx128(VOID *data, INTP n, DOUBLE factor)
{
    DOUBLE *ptr_data = (DOUBLE *)data;
    INTP count;

    __m128d vfactor = _mm_set1_pd(factor);

    for (count = 0; count < n; count++)
    {
        __m128d v = _mm_load_pd(ptr_data);
        v = _mm_mul_pd(v, vfactor);
        _mm_store_pd(ptr_data, v);

        ptr_data += NUM_SETS_128_D * DATA_STRIDE;
    }
}

normalize_ register_normalize_avx128(UINT8 precision)
{
    if (precision == DT_FLOAT)
    {
        return normalize_fp32_avx128;
    }
    else if (precision == DT_DOUBLE)
    {
        return normalize_fp64_avx128;
    }
    else
    {
        return NULL;
    }
}
