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

static VOID normalize_fp32_c(VOID *data, INTP n, DOUBLE factor)
{
    FLOAT *ptr_data = (FLOAT *)data;
    const FLOAT factor_f = (FLOAT)factor;
    INTP total_elements = n * DATA_STRIDE;

    for (INTP i = 0; i < total_elements; i++)
    {
        ptr_data[i] *= factor_f;
    }
}

static VOID normalize_fp64_c(VOID *data, INTP n, DOUBLE factor)
{
    DOUBLE *ptr_data = (DOUBLE *)data;
    INTP total_elements = n * DATA_STRIDE;

    for (INTP i = 0; i < total_elements; i++)
    {
        ptr_data[i] *= factor;
    }
}

normalize_ register_normalize_c(UINT8 precision)
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
