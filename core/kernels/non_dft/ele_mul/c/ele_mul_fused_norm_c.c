// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file ele_mul_fused_norm_c.c
 *
 *  @brief Bluestein step-3 fused normalize-and-multiply kernel (scalar C).
 *
 *  This file contains fused normalize-then-complex-multiply implementations
 *  for single-precision and double-precision inputs, including unit-stride
 *  and strided-out store paths (plan setup selects from out_stride).
 *
 *  @author Amrin Fathima
 */

#include "core/kernels/kernel.h"

static FFTZ_VOID
elementwise_mul_fused_norm_fp32_c_fwd(FFTZ_VOID *out, FFTZ_VOID *a,
                                      FFTZ_VOID *b, FFTZ_INTP n,
                                      FFTZ_DOUBLE factor, FFTZ_INTP out_stride)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FFTZ_FLOAT *p_out = (FFTZ_FLOAT *)out;
    const FFTZ_FLOAT *p_a = (const FFTZ_FLOAT *)a;
    const FFTZ_FLOAT *p_b = (const FFTZ_FLOAT *)b;
    const FFTZ_FLOAT f = (FFTZ_FLOAT)factor;
    out_stride = out_stride * DATA_STRIDE;

    for (FFTZ_INTP i = 0; i < n; i++)
    {
        FFTZ_INTP in_idx = i * DATA_STRIDE;
        FFTZ_FLOAT a_re = p_a[in_idx];
        FFTZ_FLOAT a_im = p_a[in_idx + 1];
        FFTZ_FLOAT b_re = p_b[in_idx] * f;
        FFTZ_FLOAT b_im = p_b[in_idx + 1] * f;

        p_out[i * out_stride] = (a_re * b_re) + (a_im * b_im);
        p_out[i * out_stride + 1] = (a_im * b_re) - (a_re * b_im);
    }
}

static FFTZ_VOID
elementwise_mul_fused_norm_fp32_c_bwd(FFTZ_VOID *out, FFTZ_VOID *a,
                                      FFTZ_VOID *b, FFTZ_INTP n,
                                      FFTZ_DOUBLE factor, FFTZ_INTP out_stride)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FFTZ_FLOAT *p_out = (FFTZ_FLOAT *)out;
    const FFTZ_FLOAT *p_a = (const FFTZ_FLOAT *)a;
    const FFTZ_FLOAT *p_b = (const FFTZ_FLOAT *)b;
    const FFTZ_FLOAT f = (FFTZ_FLOAT)factor;
    out_stride = out_stride * DATA_STRIDE;

    for (FFTZ_INTP i = 0; i < n; i++)
    {
        FFTZ_INTP in_idx = i * DATA_STRIDE;
        FFTZ_FLOAT a_re = p_a[in_idx];
        FFTZ_FLOAT a_im = p_a[in_idx + 1];
        FFTZ_FLOAT b_re = p_b[in_idx] * f;
        FFTZ_FLOAT b_im = p_b[in_idx + 1] * f;

        p_out[i * out_stride] = (a_re * b_re) - (a_im * b_im);
        p_out[i * out_stride + 1] = (a_re * b_im) + (a_im * b_re);
    }
}

static FFTZ_VOID
elementwise_mul_fused_norm_fp64_c_fwd(FFTZ_VOID *out, FFTZ_VOID *a,
                                      FFTZ_VOID *b, FFTZ_INTP n,
                                      FFTZ_DOUBLE factor, FFTZ_INTP out_stride)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FFTZ_DOUBLE *p_out = (FFTZ_DOUBLE *)out;
    const FFTZ_DOUBLE *p_a = (const FFTZ_DOUBLE *)a;
    const FFTZ_DOUBLE *p_b = (const FFTZ_DOUBLE *)b;
    out_stride = out_stride * DATA_STRIDE;

    for (FFTZ_INTP i = 0; i < n; i++)
    {
        FFTZ_INTP in_idx = i * DATA_STRIDE;
        FFTZ_DOUBLE a_re = p_a[in_idx];
        FFTZ_DOUBLE a_im = p_a[in_idx + 1];
        FFTZ_DOUBLE b_re = p_b[in_idx] * factor;
        FFTZ_DOUBLE b_im = p_b[in_idx + 1] * factor;

        p_out[i * out_stride] = (a_re * b_re) + (a_im * b_im);
        p_out[i * out_stride + 1] = (a_im * b_re) - (a_re * b_im);
    }
}

static FFTZ_VOID
elementwise_mul_fused_norm_fp64_c_bwd(FFTZ_VOID *out, FFTZ_VOID *a,
                                      FFTZ_VOID *b, FFTZ_INTP n,
                                      FFTZ_DOUBLE factor, FFTZ_INTP out_stride)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FFTZ_DOUBLE *p_out = (FFTZ_DOUBLE *)out;
    const FFTZ_DOUBLE *p_a = (const FFTZ_DOUBLE *)a;
    const FFTZ_DOUBLE *p_b = (const FFTZ_DOUBLE *)b;
    out_stride = out_stride * DATA_STRIDE;

    for (FFTZ_INTP i = 0; i < n; i++)
    {
        FFTZ_INTP in_idx = i * DATA_STRIDE;
        FFTZ_DOUBLE a_re = p_a[in_idx];
        FFTZ_DOUBLE a_im = p_a[in_idx + 1];
        FFTZ_DOUBLE b_re = p_b[in_idx] * factor;
        FFTZ_DOUBLE b_im = p_b[in_idx + 1] * factor;

        p_out[i * out_stride] = (a_re * b_re) - (a_im * b_im);
        p_out[i * out_stride + 1] = (a_re * b_im) + (a_im * b_re);
    }
}

elementwise_mul_fused_norm_
register_elementwise_mul_fused_norm_c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return elementwise_mul_fused_norm_fp32_c_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return elementwise_mul_fused_norm_fp64_c_fwd;
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
            return elementwise_mul_fused_norm_fp32_c_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return elementwise_mul_fused_norm_fp64_c_bwd;
        }
        else
        {
            return NULL;
        }
    }
}

elementwise_mul_fused_norm_
register_elementwise_mul_fused_norm_strided_out_c(FFTZ_UINT8 precision,
                                                  FFTZ_UINT8 direction)
{
    return register_elementwise_mul_fused_norm_c(precision, direction);
}

