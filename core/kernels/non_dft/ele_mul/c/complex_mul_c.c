// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file complex_mul_c.c
 *
 *  @brief Complex elementwise multiplication kernel with scalar C operations
 *
 *  This file contains the complex elementwise multiplication implementations
 *  using scalar C operations for single-precision and double-precision inputs.
 *
 *  @author Srirammaswamy Srinivasan
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"

static VOID complex_mul_fp32_c_fwd(VOID *out, VOID *a, VOID *b, INTP n)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FLOAT *p_out = (FLOAT *)out;
    const FLOAT *p_a = (const FLOAT *)a;
    const FLOAT *p_b = (const FLOAT *)b;
    INTP count;

    for (count = 0; count < n; count++)
    {
        INTP idx = count * DATA_STRIDE;
        FLOAT a_re = p_a[idx];
        FLOAT a_im = p_a[idx + 1];
        FLOAT b_re = p_b[idx];
        FLOAT b_im = p_b[idx + 1];

        p_out[idx] = (a_re * b_re) + (a_im * b_im);
        p_out[idx + 1] = (a_im * b_re) - (a_re * b_im);
    }
}

static VOID complex_mul_fp32_c_bwd(VOID *out, VOID *a, VOID *b, INTP n)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FLOAT *p_out = (FLOAT *)out;
    const FLOAT *p_a = (const FLOAT *)a;
    const FLOAT *p_b = (const FLOAT *)b;
    INTP count;

    for (count = 0; count < n; count++)
    {
        INTP idx = count * DATA_STRIDE;
        FLOAT a_re = p_a[idx];
        FLOAT a_im = p_a[idx + 1];
        FLOAT b_re = p_b[idx];
        FLOAT b_im = p_b[idx + 1];

        p_out[idx] = (a_re * b_re) - (a_im * b_im);
        p_out[idx + 1] = (a_re * b_im) + (a_im * b_re);
    }
}

static VOID complex_mul_fp64_c_fwd(VOID *out, VOID *a, VOID *b, INTP n)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    DOUBLE *p_out = (DOUBLE *)out;
    const DOUBLE *p_a = (const DOUBLE *)a;
    const DOUBLE *p_b = (const DOUBLE *)b;
    INTP count;

    for (count = 0; count < n; count++)
    {
        INTP idx = count * DATA_STRIDE;
        DOUBLE a_re = p_a[idx];
        DOUBLE a_im = p_a[idx + 1];
        DOUBLE b_re = p_b[idx];
        DOUBLE b_im = p_b[idx + 1];

        p_out[idx] = (a_re * b_re) + (a_im * b_im);
        p_out[idx + 1] = (a_im * b_re) - (a_re * b_im);
    }
}

static VOID complex_mul_fp64_c_bwd(VOID *out, VOID *a, VOID *b, INTP n)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    DOUBLE *p_out = (DOUBLE *)out;
    const DOUBLE *p_a = (const DOUBLE *)a;
    const DOUBLE *p_b = (const DOUBLE *)b;
    INTP count;

    for (count = 0; count < n; count++)
    {
        INTP idx = count * DATA_STRIDE;
        DOUBLE a_re = p_a[idx];
        DOUBLE a_im = p_a[idx + 1];
        DOUBLE b_re = p_b[idx];
        DOUBLE b_im = p_b[idx + 1];

        p_out[idx] = (a_re * b_re) - (a_im * b_im);
        p_out[idx + 1] = (a_re * b_im) + (a_im * b_re);
    }
}

elementwise_mul_ register_elementwise_mul_c(UINT8 precision,
                                            UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return complex_mul_fp32_c_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return complex_mul_fp64_c_fwd;
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
            return complex_mul_fp32_c_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return complex_mul_fp64_c_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
