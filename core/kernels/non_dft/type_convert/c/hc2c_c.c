// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file hc2c_c.c
 *
 *  @brief Half-complex to full-complex kernel with scalar C operations
 *
 *  Reconstructs a full size-n complex spectrum from its first n/2+1 points,
 *  deriving the remainder from Hermitian symmetry, X[n-k] = conj(X[k]). The
 *  imaginary parts that a real spectrum defines as zero (DC, and Nyquist for
 *  even n) are ignored rather than copied, so a caller that leaves a value
 *  there gets the same result as from the direct and CT real paths.
 *  The real Bluestein solver applies this to its input before the backward
 *  (C2R) transform. src holds n/2+1 points at element stride `stride`; dst
 *  holds n contiguous complex values. Supports single and double precision.
 *
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"

static FFTZ_VOID
hc2c_strided_in_fp32_c(FFTZ_VOID *dst, FFTZ_VOID *src, FFTZ_INTP n,
                       FFTZ_INTP stride)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");
    FFTZ_FLOAT *p_dst = (FFTZ_FLOAT *)dst;
    const FFTZ_FLOAT *p_src = (const FFTZ_FLOAT *)src;
    FFTZ_INTP count;
    FFTZ_INTP n_hc = n / 2 + 1;

    for (count = 0; count < n_hc; count++)
    {
        FFTZ_INTP dst_idx = count * DATA_STRIDE;
        FFTZ_INTP src_idx = count * stride * DATA_STRIDE;
        p_dst[dst_idx] = p_src[src_idx];
        p_dst[dst_idx + 1] = p_src[src_idx + 1];
    }

    // A real spectrum has a real DC term, and a real Nyquist term when n is
    // even; odd n has no Nyquist bin, so its offset folds onto DC. Ignore
    // whatever the caller left in those imaginary slots, so the spectrum
    // rebuilt below is exactly Hermitian as the real paths assume.
    FFTZ_INTP nyquist_im = ((n & 1) == 0) ? n + 1 : 1;
    p_dst[1] = 0.0f;
    p_dst[nyquist_im] = 0.0f;

    for (count = n_hc; count < n; count++)
    {
        FFTZ_INTP dst_idx = count * DATA_STRIDE;
        FFTZ_INTP src_idx = (n - count) * DATA_STRIDE;
        p_dst[dst_idx] = p_dst[src_idx];
        p_dst[dst_idx + 1] = -p_dst[src_idx + 1];
    }
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
}

static FFTZ_VOID
hc2c_strided_in_fp64_c(FFTZ_VOID *dst, FFTZ_VOID *src, FFTZ_INTP n,
                       FFTZ_INTP stride)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");
    FFTZ_DOUBLE *p_dst = (FFTZ_DOUBLE *)dst;
    const FFTZ_DOUBLE *p_src = (const FFTZ_DOUBLE *)src;
    FFTZ_INTP count;
    FFTZ_INTP n_hc = n / 2 + 1;

    for (count = 0; count < n_hc; count++)
    {
        FFTZ_INTP dst_idx = count * DATA_STRIDE;
        FFTZ_INTP src_idx = count * stride * DATA_STRIDE;
        p_dst[dst_idx] = p_src[src_idx];
        p_dst[dst_idx + 1] = p_src[src_idx + 1];
    }

    // A real spectrum has a real DC term, and a real Nyquist term when n is
    // even; odd n has no Nyquist bin, so its offset folds onto DC. Ignore
    // whatever the caller left in those imaginary slots, so the spectrum
    // rebuilt below is exactly Hermitian as the real paths assume.
    FFTZ_INTP nyquist_im = ((n & 1) == 0) ? n + 1 : 1;
    p_dst[1] = 0.0;
    p_dst[nyquist_im] = 0.0;

    for (count = n_hc; count < n; count++)
    {
        FFTZ_INTP dst_idx = count * DATA_STRIDE;
        FFTZ_INTP src_idx = (n - count) * DATA_STRIDE;
        p_dst[dst_idx] = p_dst[src_idx];
        p_dst[dst_idx + 1] = -p_dst[src_idx + 1];
    }
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
}

type_convert_ register_hc2c_type_convert_c(FFTZ_UINT8 precision)
{
    if (precision == DT_FLOAT)
    {
        return hc2c_strided_in_fp32_c;
    }
    else if (precision == DT_DOUBLE)
    {
        return hc2c_strided_in_fp64_c;
    }
    else
    {
        return NULL;
    }
}
