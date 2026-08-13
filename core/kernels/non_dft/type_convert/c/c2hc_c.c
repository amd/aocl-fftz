// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file c2hc_c.c
 *
 *  @brief Full-complex to half-complex kernel with scalar C operations
 *
 *  Retains the first n/2+1 points of a complex spectrum and discards the
 *  remainder, which Hermitian symmetry renders redundant for a real transform.
 *  The imaginary parts that a real spectrum defines as zero (DC, and Nyquist
 *  for even n) are cleared rather than copied, so this path yields the same
 *  half-complex output as the direct and CT real paths.
 *  The real Bluestein solver applies this to its result after the forward (R2C)
 *  transform. src holds n contiguous complex values; dst holds the retained
 *  n/2+1 values at element stride `stride`. Supports single and double
 *  precision.
 *
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"

static FFTZ_VOID
c2hc_strided_out_fp32_c(FFTZ_VOID *dst, FFTZ_VOID *src, FFTZ_INTP n,
                        FFTZ_INTP stride)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");
    FFTZ_FLOAT *p_dst = (FFTZ_FLOAT *)dst;
    const FFTZ_FLOAT *p_src = (const FFTZ_FLOAT *)src;
    FFTZ_INTP count;
    FFTZ_INTP n_hc = n / 2 + 1;

    for (count = 0; count < n_hc; count++)
    {
        FFTZ_INTP dst_idx = count * stride * DATA_STRIDE;
        FFTZ_INTP src_idx = count * DATA_STRIDE;
        p_dst[dst_idx] = p_src[src_idx];
        p_dst[dst_idx + 1] = p_src[src_idx + 1];
    }

    // A real spectrum has a real DC term, and a real Nyquist term when n is
    // even; odd n has no Nyquist bin, so its offset folds onto DC, exactly as
    // setup_rdft_dc_nyquist_offsets_ds computes it. Bluestein reaches the
    // spectrum through complex chirp multiplies, so it leaves rounding noise
    // in those imaginary slots where the direct and CT paths emit exact zeros.
    FFTZ_INTP nyquist_im = ((n & 1) == 0) ? n * stride + 1 : 1;
    p_dst[1] = 0.0f;
    p_dst[nyquist_im] = 0.0f;
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
}

static FFTZ_VOID
c2hc_strided_out_fp64_c(FFTZ_VOID *dst, FFTZ_VOID *src, FFTZ_INTP n,
                        FFTZ_INTP stride)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");
    FFTZ_DOUBLE *p_dst = (FFTZ_DOUBLE *)dst;
    const FFTZ_DOUBLE *p_src = (const FFTZ_DOUBLE *)src;
    FFTZ_INTP count;
    FFTZ_INTP n_hc = n / 2 + 1;

    for (count = 0; count < n_hc; count++)
    {
        FFTZ_INTP dst_idx = count * stride * DATA_STRIDE;
        FFTZ_INTP src_idx = count * DATA_STRIDE;
        p_dst[dst_idx] = p_src[src_idx];
        p_dst[dst_idx + 1] = p_src[src_idx + 1];
    }

    // A real spectrum has a real DC term, and a real Nyquist term when n is
    // even; odd n has no Nyquist bin, so its offset folds onto DC, exactly as
    // setup_rdft_dc_nyquist_offsets_ds computes it. Bluestein reaches the
    // spectrum through complex chirp multiplies, so it leaves rounding noise
    // in those imaginary slots where the direct and CT paths emit exact zeros.
    FFTZ_INTP nyquist_im = ((n & 1) == 0) ? n * stride + 1 : 1;
    p_dst[1] = 0.0;
    p_dst[nyquist_im] = 0.0;
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
}

type_convert_ register_c2hc_type_convert_c(FFTZ_UINT8 precision)
{
    if (precision == DT_FLOAT)
    {
        return c2hc_strided_out_fp32_c;
    }
    else if (precision == DT_DOUBLE)
    {
        return c2hc_strided_out_fp64_c;
    }
    else
    {
        return NULL;
    }
}
