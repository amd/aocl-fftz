// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file c2hc_packed_c.c
 *
 *  @brief Full-complex to packed half-complex kernel (scalar C).
 *
 *  Stores a size-n complex spectrum as the n packed reals a real CT combine
 *  expects, so a Bluestein can stand in for the R2HC stage: a DC term and
 *  (n-1)/2 (Re,Im) pairs. n is always the odd kernel-radix residue, so there
 *  is no Nyquist term (the Hermitian upper half and the zero DC imaginary are
 *  dropped).
 *
 *  The reals are scattered into the regrouped aux layout: DC and Nyquist into
 *  the real band (stepped by v_out_sym_stride), interior pairs into the pairs
 *  band (stepped by v_out_stride); per-point offsets come from the stage's R2HC
 *  out_strides. `group` selects the sub-transform. Single and double precision.
 *
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"

static FFTZ_VOID c2hc_packed_fp32_c(FFTZ_VOID *dst, FFTZ_VOID *src, FFTZ_INTP n,
                                    FFTZ_INTP elem_stride,
                                    aoclfftz_strides_t *strides,
                                    FFTZ_INTP group)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");
    FFTZ_FLOAT *p_dst = (FFTZ_FLOAT *)dst;
    const FFTZ_FLOAT *p_src = (const FFTZ_FLOAT *)src;
    const FFTZ_INTP *p_out = strides->out_strides;
    FFTZ_INTP real_base = group * strides->v_out_sym_stride;
    FFTZ_INTP pairs_base = group * strides->v_out_stride;
    FFTZ_INTP n_pairs = (n - 1) / 2;
    FFTZ_INTP k;

    // DC term is real; its imaginary part is dropped. Lands in the real band.
    p_dst[real_base + p_out[0]] = p_src[0];

    // Interior points: only the lower half is retained, the upper half follows
    // from Hermitian symmetry. Point k lands as an (Re, Im) pair in the pairs
    // band at the out_strides slots (2k-1, 2k).
    for (k = 1; k <= n_pairs; k++)
    {
        p_dst[pairs_base + p_out[2 * k - 1]] = p_src[2 * k];
        p_dst[pairs_base + p_out[2 * k]] = p_src[2 * k + 1];
    }
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
}

static FFTZ_VOID c2hc_packed_fp64_c(FFTZ_VOID *dst, FFTZ_VOID *src, FFTZ_INTP n,
                                    FFTZ_INTP elem_stride,
                                    aoclfftz_strides_t *strides,
                                    FFTZ_INTP group)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");
    FFTZ_DOUBLE *p_dst = (FFTZ_DOUBLE *)dst;
    const FFTZ_DOUBLE *p_src = (const FFTZ_DOUBLE *)src;
    const FFTZ_INTP *p_out = strides->out_strides;
    FFTZ_INTP real_base = group * strides->v_out_sym_stride;
    FFTZ_INTP pairs_base = group * strides->v_out_stride;
    FFTZ_INTP n_pairs = (n - 1) / 2;
    FFTZ_INTP k;

    // DC term is real; its imaginary part is dropped. Lands in the real band.
    p_dst[real_base + p_out[0]] = p_src[0];

    // Interior points: only the lower half is retained, the upper half follows
    // from Hermitian symmetry. Point k lands as an (Re, Im) pair in the pairs
    // band at the out_strides slots (2k-1, 2k).
    for (k = 1; k <= n_pairs; k++)
    {
        p_dst[pairs_base + p_out[2 * k - 1]] = p_src[2 * k];
        p_dst[pairs_base + p_out[2 * k]] = p_src[2 * k + 1];
    }
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
}

type_convert_ register_c2hc_packed_type_convert_c(FFTZ_UINT8 precision)
{
    if (precision == DT_FLOAT)
    {
        return c2hc_packed_fp32_c;
    }
    else if (precision == DT_DOUBLE)
    {
        return c2hc_packed_fp64_c;
    }
    else
    {
        return NULL;
    }
}
