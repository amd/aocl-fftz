// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file r2c_c.c
 *
 *  @brief Real to full-complex kernel with scalar C operations
 *
 *  Converts n real values into n interleaved complex values, setting each
 *  imaginary part to zero. The real Bluestein solver applies this to its input
 *  before the forward (R2C) transform. src holds n reals at element stride
 *  `stride`; dst holds n contiguous complex values. Supports single and double
 *  precision.
 *
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"

static FFTZ_VOID
r2c_strided_in_fp32_c(FFTZ_VOID *dst, FFTZ_VOID *src, FFTZ_INTP n,
                      FFTZ_INTP stride)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");
    FFTZ_FLOAT *p_dst = (FFTZ_FLOAT *)dst;
    const FFTZ_FLOAT *p_src = (const FFTZ_FLOAT *)src;
    FFTZ_INTP count;

    for (count = 0; count < n; count++)
    {
        FFTZ_INTP dst_idx = count * DATA_STRIDE;
        FFTZ_INTP src_idx = count * stride;
        p_dst[dst_idx] = p_src[src_idx];
        p_dst[dst_idx + 1] = 0.0f;
    }
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
}

static FFTZ_VOID
r2c_strided_in_fp64_c(FFTZ_VOID *dst, FFTZ_VOID *src, FFTZ_INTP n,
                      FFTZ_INTP stride)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");
    FFTZ_DOUBLE *p_dst = (FFTZ_DOUBLE *)dst;
    const FFTZ_DOUBLE *p_src = (const FFTZ_DOUBLE *)src;
    FFTZ_INTP count;

    for (count = 0; count < n; count++)
    {
        FFTZ_INTP dst_idx = count * DATA_STRIDE;
        FFTZ_INTP src_idx = count * stride;
        p_dst[dst_idx] = p_src[src_idx];
        p_dst[dst_idx + 1] = 0.0;
    }
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
}

type_convert_ register_r2c_type_convert_c(FFTZ_UINT8 precision)
{
    if (precision == DT_FLOAT)
    {
        return r2c_strided_in_fp32_c;
    }
    else if (precision == DT_DOUBLE)
    {
        return r2c_strided_in_fp64_c;
    }
    else
    {
        return NULL;
    }
}
