// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file normalize_avx256.c
 *
 * @brief In-place complex buffer normalization with AVX256 SIMD operations.
 *
 * Multiplies all real and imaginary components of a complex buffer by a
 * real scalar factor (the 1/N scaling applied during inverse FFT).
 *
 * @author Jeevanantham N
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common.h"

static VOID normalize_fp32_avx256(VOID *data, INTP n, DOUBLE factor)
{
    FLOAT *ptr_data = (FLOAT *)data;
    INTP N = n / NUM_SETS_256_S;
    INTP remaining_sets = n % NUM_SETS_256_S;
    INTP count;

    __m256 vfactor = _mm256_set1_ps((FLOAT)factor);

    for (count = 0; count < N; count++)
    {
        __m256 v = _mm256_load_ps(ptr_data);
        v = _mm256_mul_ps(v, vfactor);
        _mm256_store_ps(ptr_data, v);

        ptr_data += NUM_SETS_256_S * DATA_STRIDE;
    }
    // tail cases
    if (remaining_sets & NUM_SETS_128_S)
    {
        __m128 vfactor_128 = _mm256_castps256_ps128(vfactor);
        __m128 v = _mm_load_ps(ptr_data);
        v = _mm_mul_ps(v, vfactor_128);
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

static VOID normalize_fp64_avx256(VOID *data, INTP n, DOUBLE factor)
{
    DOUBLE *ptr_data = (DOUBLE *)data;
    INTP N = n / NUM_SETS_256_D;
    INTP remaining_sets = n % NUM_SETS_256_D;
    INTP count;

    __m256d vfactor = _mm256_set1_pd(factor);

    for (count = 0; count < N; count++)
    {
        __m256d v = _mm256_load_pd(ptr_data);
        v = _mm256_mul_pd(v, vfactor);
        _mm256_store_pd(ptr_data, v);

        ptr_data += NUM_SETS_256_D * DATA_STRIDE;
    }
    // tail cases
    if (remaining_sets & NUM_SETS_C_D)
    {
        ptr_data[0] *= factor;
        ptr_data[1] *= factor;
    }
}

normalize_ register_normalize_avx256(UINT8 precision)
{
    if (precision == DT_FLOAT)
    {
        return normalize_fp32_avx256;
    }
    else if (precision == DT_DOUBLE)
    {
        return normalize_fp64_avx256;
    }
    else
    {
        return NULL;
    }
}
