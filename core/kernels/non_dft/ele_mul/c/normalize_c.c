// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file normalize_c.c
 *
 * @brief In-place complex buffer normalization with scalar C operations.
 *
 * Multiplies all real and imaginary components of a complex buffer by a
 * real scalar factor (the 1/N scaling applied during inverse FFT).
 *
 * @author Jeevanantham N
 */

#include "core/kernels/kernel.h"

static FFTZ_VOID normalize_fp32_c(FFTZ_VOID *data, FFTZ_INTP n,
                                  FFTZ_DOUBLE factor)
{
    FFTZ_FLOAT *ptr_data = (FFTZ_FLOAT *)data;
    const FFTZ_FLOAT factor_f = (FFTZ_FLOAT)factor;
    FFTZ_INTP total_elements = n * DATA_STRIDE;

    for (FFTZ_INTP i = 0; i < total_elements; i++)
    {
        ptr_data[i] *= factor_f;
    }
}

static FFTZ_VOID normalize_fp64_c(FFTZ_VOID *data, FFTZ_INTP n,
                                  FFTZ_DOUBLE factor)
{
    FFTZ_DOUBLE *ptr_data = (FFTZ_DOUBLE *)data;
    FFTZ_INTP total_elements = n * DATA_STRIDE;

    for (FFTZ_INTP i = 0; i < total_elements; i++)
    {
        ptr_data[i] *= factor;
    }
}

normalize_ register_normalize_c(FFTZ_UINT8 precision)
{
    if (precision == DT_FLOAT)
    {
        return normalize_fp32_c;
    }
    else if (precision == DT_DOUBLE)
    {
        return normalize_fp64_c;
    }
    else
    {
        return NULL;
    }
}
