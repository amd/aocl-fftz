// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file ele_mul_c.c
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

static FFTZ_VOID elementwise_mul_fp32_c_fwd(FFTZ_VOID *out, FFTZ_VOID *a,
                                            FFTZ_VOID *b, FFTZ_INTP n,
                                            FFTZ_INTP start_idx,
                                            FFTZ_INTP stride)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FFTZ_FLOAT *p_out = (FFTZ_FLOAT *)out;
    const FFTZ_FLOAT *p_a = (const FFTZ_FLOAT *)a;
    const FFTZ_FLOAT *p_b = (const FFTZ_FLOAT *)b;
    FFTZ_INTP count;

    for (count = 0; count < n; count++)
    {
        FFTZ_INTP idx = count * DATA_STRIDE;
        FFTZ_FLOAT a_re = p_a[idx];
        FFTZ_FLOAT a_im = p_a[idx + 1];
        FFTZ_FLOAT b_re = p_b[idx];
        FFTZ_FLOAT b_im = p_b[idx + 1];

        p_out[idx] = (a_re * b_re) + (a_im * b_im);
        p_out[idx + 1] = (a_im * b_re) - (a_re * b_im);
    }
}

static FFTZ_VOID elementwise_mul_fp32_c_bwd(FFTZ_VOID *out, FFTZ_VOID *a,
                                            FFTZ_VOID *b, FFTZ_INTP n,
                                            FFTZ_INTP start_idx,
                                            FFTZ_INTP stride)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FFTZ_FLOAT *p_out = (FFTZ_FLOAT *)out;
    const FFTZ_FLOAT *p_a = (const FFTZ_FLOAT *)a;
    const FFTZ_FLOAT *p_b = (const FFTZ_FLOAT *)b;
    FFTZ_INTP count;

    for (count = 0; count < n; count++)
    {
        FFTZ_INTP idx = count * DATA_STRIDE;
        FFTZ_FLOAT a_re = p_a[idx];
        FFTZ_FLOAT a_im = p_a[idx + 1];
        FFTZ_FLOAT b_re = p_b[idx];
        FFTZ_FLOAT b_im = p_b[idx + 1];

        p_out[idx] = (a_re * b_re) - (a_im * b_im);
        p_out[idx + 1] = (a_re * b_im) + (a_im * b_re);
    }
}

static FFTZ_VOID elementwise_mul_fp64_c_fwd(FFTZ_VOID *out, FFTZ_VOID *a,
                                            FFTZ_VOID *b, FFTZ_INTP n,
                                            FFTZ_INTP start_idx,
                                            FFTZ_INTP stride)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FFTZ_DOUBLE *p_out = (FFTZ_DOUBLE *)out;
    const FFTZ_DOUBLE *p_a = (const FFTZ_DOUBLE *)a;
    const FFTZ_DOUBLE *p_b = (const FFTZ_DOUBLE *)b;
    FFTZ_INTP count;

    for (count = 0; count < n; count++)
    {
        FFTZ_INTP idx = count * DATA_STRIDE;
        FFTZ_DOUBLE a_re = p_a[idx];
        FFTZ_DOUBLE a_im = p_a[idx + 1];
        FFTZ_DOUBLE b_re = p_b[idx];
        FFTZ_DOUBLE b_im = p_b[idx + 1];

        p_out[idx] = (a_re * b_re) + (a_im * b_im);
        p_out[idx + 1] = (a_im * b_re) - (a_re * b_im);
    }
}

static FFTZ_VOID elementwise_mul_fp64_c_bwd(FFTZ_VOID *out, FFTZ_VOID *a,
                                            FFTZ_VOID *b, FFTZ_INTP n,
                                            FFTZ_INTP start_idx,
                                            FFTZ_INTP stride)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FFTZ_DOUBLE *p_out = (FFTZ_DOUBLE *)out;
    const FFTZ_DOUBLE *p_a = (const FFTZ_DOUBLE *)a;
    const FFTZ_DOUBLE *p_b = (const FFTZ_DOUBLE *)b;
    FFTZ_INTP count;

    for (count = 0; count < n; count++)
    {
        FFTZ_INTP idx = count * DATA_STRIDE;
        FFTZ_DOUBLE a_re = p_a[idx];
        FFTZ_DOUBLE a_im = p_a[idx + 1];
        FFTZ_DOUBLE b_re = p_b[idx];
        FFTZ_DOUBLE b_im = p_b[idx + 1];

        p_out[idx] = (a_re * b_re) - (a_im * b_im);
        p_out[idx + 1] = (a_re * b_im) + (a_im * b_re);
    }
}

static FFTZ_VOID
elementwise_mul_strided_in_fp32_c_fwd(FFTZ_VOID *out, FFTZ_VOID *a,
                                      FFTZ_VOID *b, FFTZ_INTP n,
                                      FFTZ_INTP start_idx, FFTZ_INTP stride)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FFTZ_FLOAT *p_out = (FFTZ_FLOAT *)out;
    const FFTZ_FLOAT *p_a = (const FFTZ_FLOAT *)a;
    const FFTZ_FLOAT *p_b = (const FFTZ_FLOAT *)b;
    FFTZ_INTP in_stride = stride * DATA_STRIDE;

    for (FFTZ_INTP i = 0; i < n; i++)
    {
        FFTZ_INTP in_idx = (start_idx + i) * in_stride;
        FFTZ_INTP out_idx = i * DATA_STRIDE;
        FFTZ_FLOAT a_re = p_a[in_idx];
        FFTZ_FLOAT a_im = p_a[in_idx + 1];
        FFTZ_FLOAT b_re = p_b[out_idx];
        FFTZ_FLOAT b_im = p_b[out_idx + 1];

        p_out[out_idx] = (a_re * b_re) + (a_im * b_im);
        p_out[out_idx + 1] = (a_im * b_re) - (a_re * b_im);
    }
}

static FFTZ_VOID
elementwise_mul_strided_in_fp32_c_bwd(FFTZ_VOID *out, FFTZ_VOID *a,
                                      FFTZ_VOID *b, FFTZ_INTP n,
                                      FFTZ_INTP start_idx, FFTZ_INTP stride)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FFTZ_FLOAT *p_out = (FFTZ_FLOAT *)out;
    const FFTZ_FLOAT *p_a = (const FFTZ_FLOAT *)a;
    const FFTZ_FLOAT *p_b = (const FFTZ_FLOAT *)b;
    FFTZ_INTP in_stride = stride * DATA_STRIDE;

    for (FFTZ_INTP i = 0; i < n; i++)
    {
        FFTZ_INTP in_idx = (start_idx + i) * in_stride;
        FFTZ_INTP out_idx = i * DATA_STRIDE;
        FFTZ_FLOAT a_re = p_a[in_idx];
        FFTZ_FLOAT a_im = p_a[in_idx + 1];
        FFTZ_FLOAT b_re = p_b[out_idx];
        FFTZ_FLOAT b_im = p_b[out_idx + 1];

        p_out[out_idx] = (a_re * b_re) - (a_im * b_im);
        p_out[out_idx + 1] = (a_re * b_im) + (a_im * b_re);
    }
}

static FFTZ_VOID
elementwise_mul_strided_in_fp64_c_fwd(FFTZ_VOID *out, FFTZ_VOID *a,
                                      FFTZ_VOID *b, FFTZ_INTP n,
                                      FFTZ_INTP start_idx, FFTZ_INTP stride)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FFTZ_DOUBLE *p_out = (FFTZ_DOUBLE *)out;
    const FFTZ_DOUBLE *p_a = (const FFTZ_DOUBLE *)a;
    const FFTZ_DOUBLE *p_b = (const FFTZ_DOUBLE *)b;
    FFTZ_INTP in_stride = stride * DATA_STRIDE;

    for (FFTZ_INTP i = 0; i < n; i++)
    {
        FFTZ_INTP in_idx = (start_idx + i) * in_stride;
        FFTZ_INTP out_idx = i * DATA_STRIDE;
        FFTZ_DOUBLE a_re = p_a[in_idx];
        FFTZ_DOUBLE a_im = p_a[in_idx + 1];
        FFTZ_DOUBLE b_re = p_b[out_idx];
        FFTZ_DOUBLE b_im = p_b[out_idx + 1];

        p_out[out_idx] = (a_re * b_re) + (a_im * b_im);
        p_out[out_idx + 1] = (a_im * b_re) - (a_re * b_im);
    }
}

static FFTZ_VOID
elementwise_mul_strided_in_fp64_c_bwd(FFTZ_VOID *out, FFTZ_VOID *a,
                                      FFTZ_VOID *b, FFTZ_INTP n,
                                      FFTZ_INTP start_idx, FFTZ_INTP stride)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FFTZ_DOUBLE *p_out = (FFTZ_DOUBLE *)out;
    const FFTZ_DOUBLE *p_a = (const FFTZ_DOUBLE *)a;
    const FFTZ_DOUBLE *p_b = (const FFTZ_DOUBLE *)b;
    FFTZ_INTP in_stride = stride * DATA_STRIDE;

    for (FFTZ_INTP i = 0; i < n; i++)
    {
        FFTZ_INTP in_idx = (start_idx + i) * in_stride;
        FFTZ_INTP out_idx = i * DATA_STRIDE;
        FFTZ_DOUBLE a_re = p_a[in_idx];
        FFTZ_DOUBLE a_im = p_a[in_idx + 1];
        FFTZ_DOUBLE b_re = p_b[out_idx];
        FFTZ_DOUBLE b_im = p_b[out_idx + 1];

        p_out[out_idx] = (a_re * b_re) - (a_im * b_im);
        p_out[out_idx + 1] = (a_re * b_im) + (a_im * b_re);
    }
}


elementwise_mul_ register_elementwise_mul_c(FFTZ_UINT8 precision,
                                            FFTZ_UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return elementwise_mul_fp32_c_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return elementwise_mul_fp64_c_fwd;
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
            return elementwise_mul_fp32_c_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return elementwise_mul_fp64_c_bwd;
        }
        else
        {
            return NULL;
        }
    }
}

elementwise_mul_ register_elementwise_mul_strided_in_c(FFTZ_UINT8 precision,
                                                       FFTZ_UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return elementwise_mul_strided_in_fp32_c_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return elementwise_mul_strided_in_fp64_c_fwd;
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
            return elementwise_mul_strided_in_fp32_c_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return elementwise_mul_strided_in_fp64_c_bwd;
        }
        else
        {
            return NULL;
        }
    }
}

