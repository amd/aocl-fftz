// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file hc2c_packed_c.c
 *
 *  @brief Packed half-complex to full-complex kernel (scalar C).
 *
 *  Expands the n packed reals a real CT stage produces into a full size-n
 *  complex spectrum via Hermitian symmetry X[n-k] = conj(X[k]), so a
 *  Bluestein can stand in for the HC2R stage. The reals are gathered from
 *  the regrouped aux layout: the DC real from the real band (stepped by
 *  v_in_sym_stride), interior pairs from the pairs band (stepped by
 *  v_in_stride); per-point offsets come from the stage's R2HC in_strides.
 *  The zero DC imaginary is reinstated here. n is always the odd kernel-radix
 *  residue, so there is no Nyquist term. `group` selects the sub-transform.
 *  Single and double precision.
 *
 *  @author Jeevanantham N
 */

#include "core/kernels/kernel.h"

static FFTZ_VOID hc2c_packed_fp32_c(FFTZ_VOID *dst, FFTZ_VOID *src, FFTZ_INTP n,
                                    FFTZ_INTP elem_stride,
                                    aoclfftz_strides_t *strides,
                                    FFTZ_INTP group)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");
    FFTZ_FLOAT *p_dst = (FFTZ_FLOAT *)dst;
    const FFTZ_FLOAT *p_src = (const FFTZ_FLOAT *)src;
    const FFTZ_INTP *p_in = strides->in_strides;
    FFTZ_INTP real_base = group * strides->v_in_sym_stride;
    FFTZ_INTP pairs_base = group * strides->v_in_stride;
    FFTZ_INTP n_pairs = (n - 1) / 2;
    FFTZ_INTP k;

    // DC term is real; its zero imaginary part is reinstated.
    p_dst[0] = p_src[real_base + p_in[0]];
    p_dst[1] = 0.0f;

    // Each source pair settles both the point it holds and that point's
    // Hermitian mirror, so the upper half costs no reload of the lower half.
    for (k = 1; k <= n_pairs; k++)
    {
        FFTZ_FLOAT re = p_src[pairs_base + p_in[2 * k - 1]];
        FFTZ_FLOAT im = p_src[pairs_base + p_in[2 * k]];

        p_dst[2 * k] = re;
        p_dst[2 * k + 1] = im;
        p_dst[2 * (n - k)] = re;
        p_dst[2 * (n - k) + 1] = -im;
    }
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
}

static FFTZ_VOID hc2c_packed_fp64_c(FFTZ_VOID *dst, FFTZ_VOID *src, FFTZ_INTP n,
                                    FFTZ_INTP elem_stride,
                                    aoclfftz_strides_t *strides,
                                    FFTZ_INTP group)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");
    FFTZ_DOUBLE *p_dst = (FFTZ_DOUBLE *)dst;
    const FFTZ_DOUBLE *p_src = (const FFTZ_DOUBLE *)src;
    const FFTZ_INTP *p_in = strides->in_strides;
    FFTZ_INTP real_base = group * strides->v_in_sym_stride;
    FFTZ_INTP pairs_base = group * strides->v_in_stride;
    FFTZ_INTP n_pairs = (n - 1) / 2;
    FFTZ_INTP k;

    // DC term is real; its zero imaginary part is reinstated.
    p_dst[0] = p_src[real_base + p_in[0]];
    p_dst[1] = 0.0;

    // Each source pair settles both the point it holds and that point's
    // Hermitian mirror, so the upper half costs no reload of the lower half.
    for (k = 1; k <= n_pairs; k++)
    {
        FFTZ_DOUBLE re = p_src[pairs_base + p_in[2 * k - 1]];
        FFTZ_DOUBLE im = p_src[pairs_base + p_in[2 * k]];

        p_dst[2 * k] = re;
        p_dst[2 * k + 1] = im;
        p_dst[2 * (n - k)] = re;
        p_dst[2 * (n - k) + 1] = -im;
    }
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
}

type_convert_ register_hc2c_packed_type_convert_c(FFTZ_UINT8 precision)
{
    if (precision == DT_FLOAT)
    {
        return hc2c_packed_fp32_c;
    }
    else if (precision == DT_DOUBLE)
    {
        return hc2c_packed_fp64_c;
    }
    else
    {
        return NULL;
    }
}
