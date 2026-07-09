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

static FFTZ_VOID normalize_fp32_avx128(FFTZ_VOID *data, FFTZ_INTP n,
                                       FFTZ_DOUBLE factor)
{
    FFTZ_FLOAT *ptr_data = (FFTZ_FLOAT *)data;
    FFTZ_INTP N = n / NUM_SETS_128_S;
    FFTZ_INTP remaining_sets = n % NUM_SETS_128_S;
    FFTZ_INTP count;

    __m128 vfactor = _mm_set1_ps((FFTZ_FLOAT)factor);

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
        const FFTZ_FLOAT factor_f = (FFTZ_FLOAT)factor;
        ptr_data[0] *= factor_f;
        ptr_data[1] *= factor_f;
    }
}

static FFTZ_VOID normalize_fp64_avx128(FFTZ_VOID *data, FFTZ_INTP n,
                                       FFTZ_DOUBLE factor)
{
    FFTZ_DOUBLE *ptr_data = (FFTZ_DOUBLE *)data;
    FFTZ_INTP count;

    __m128d vfactor = _mm_set1_pd(factor);

    for (count = 0; count < n; count++)
    {
        __m128d v = _mm_load_pd(ptr_data);
        v = _mm_mul_pd(v, vfactor);
        _mm_store_pd(ptr_data, v);

        ptr_data += NUM_SETS_128_D * DATA_STRIDE;
    }
}

normalize_ register_normalize_avx128(FFTZ_UINT8 precision)
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
