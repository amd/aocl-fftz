// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file c2r_c.c
 *
 *  @brief Full-complex to real kernel with scalar C operations
 *
 *  Extracts the real part of each of n complex values and discards the
 *  imaginary parts, as a backward real transform yields real output. The real
 *  Bluestein solver applies this to its result after the backward (C2R)
 *  transform. src holds n contiguous complex values; dst holds n reals at
 *  element stride `stride`. Supports single and double precision.
 *
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"

static FFTZ_VOID
c2r_strided_out_fp32_c(FFTZ_VOID *dst, FFTZ_VOID *src, FFTZ_INTP n,
                       FFTZ_INTP stride)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");
    FFTZ_FLOAT *p_dst = (FFTZ_FLOAT *)dst;
    const FFTZ_FLOAT *p_src = (const FFTZ_FLOAT *)src;
    FFTZ_INTP count;

    for (count = 0; count < n; count++)
    {
        p_dst[count * stride] = p_src[count * DATA_STRIDE];
    }
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
}

static FFTZ_VOID
c2r_strided_out_fp64_c(FFTZ_VOID *dst, FFTZ_VOID *src, FFTZ_INTP n,
                       FFTZ_INTP stride)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");
    FFTZ_DOUBLE *p_dst = (FFTZ_DOUBLE *)dst;
    const FFTZ_DOUBLE *p_src = (const FFTZ_DOUBLE *)src;
    FFTZ_INTP count;

    for (count = 0; count < n; count++)
    {
        p_dst[count * stride] = p_src[count * DATA_STRIDE];
    }
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
}

type_convert_ register_c2r_type_convert_c(FFTZ_UINT8 precision)
{
    if (precision == DT_FLOAT)
    {
        return c2r_strided_out_fp32_c;
    }
    else if (precision == DT_DOUBLE)
    {
        return c2r_strided_out_fp64_c;
    }
    else
    {
        return NULL;
    }
}
