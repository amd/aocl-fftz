// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file normalize_avx512.c
 *
 * @brief In-place complex buffer normalization with AVX512 SIMD operations.
 *
 * Multiplies all real and imaginary components of a complex buffer by a
 * real scalar factor (the 1/N scaling applied during inverse FFT).
 *
 * @author Jeevanantham N
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common_avx512.h"

static VOID normalize_fp32_avx512(VOID *data, INTP n, DOUBLE factor)
{
    FLOAT *ptr_data = (FLOAT *)data;
    INTP N = n / NUM_SETS_512_S;
    INTP remaining_sets = n % NUM_SETS_512_S;
    INTP count;

    __m512 vfactor = _mm512_set1_ps((FLOAT)factor);

    for (count = 0; count < N; count++)
    {
        __m512 v = _mm512_load_ps(ptr_data);
        v = _mm512_mul_ps(v, vfactor);
        _mm512_store_ps(ptr_data, v);

        ptr_data += NUM_SETS_512_S * DATA_STRIDE;
    }
    // tail cases
    if (remaining_sets & NUM_SETS_256_S)
    {
        __m256 vfactor_256 = _mm512_castps512_ps256(vfactor);
        __m256 v = _mm256_load_ps(ptr_data);
        v = _mm256_mul_ps(v, vfactor_256);
        _mm256_store_ps(ptr_data, v);

        ptr_data += NUM_SETS_256_S * DATA_STRIDE;
    }
    // tail cases
    if (remaining_sets & NUM_SETS_128_S)
    {
        __m128 vfactor_128 = _mm512_castps512_ps128(vfactor);
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

static VOID normalize_fp64_avx512(VOID *data, INTP n, DOUBLE factor)
{
    DOUBLE *ptr_data = (DOUBLE *)data;
    INTP N = n / NUM_SETS_512_D;
    INTP remaining_sets = n % NUM_SETS_512_D;
    INTP count;

    __m512d vfactor = _mm512_set1_pd(factor);

    for (count = 0; count < N; count++)
    {
        __m512d v = _mm512_load_pd(ptr_data);
        v = _mm512_mul_pd(v, vfactor);
        _mm512_store_pd(ptr_data, v);

        ptr_data += NUM_SETS_512_D * DATA_STRIDE;
    }
    // tail cases
    if (remaining_sets & NUM_SETS_256_D)
    {
        __m256d vfactor_256 = _mm512_castpd512_pd256(vfactor);
        __m256d v = _mm256_load_pd(ptr_data);
        v = _mm256_mul_pd(v, vfactor_256);
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

normalize_ register_normalize_avx512(UINT8 precision)
{
    if (precision == DT_FLOAT)
    {
        return normalize_fp32_avx512;
    }
    else if (precision == DT_DOUBLE)
    {
        return normalize_fp64_avx512;
    }
    else
    {
        return NULL;
    }
}
