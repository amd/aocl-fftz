// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file recombine_to_hc_c.c
 *
 * @brief Packed R2C recombine kernel (DIT last stage).
 *
 * Recombines the M-point complex spectrum Z into the N/2 + 1 Hermitian output
 * bins X (N = 2M). The conjugate-pair butterfly and twiddle-table layout are
 * documented in packed_rdft_common.h. DC, Nyquist and the (even M)
 * middle bin are handled separately. All conjugate pairs use the
 * scalar helper.
 *
 * @author Srirammaswamy Srinivasan
 */

#include "core/kernels/kernel.h"
#include "core/kernels/non_dft/real_packed/packed_rdft_common.h"

static FFTZ_VOID recombine_to_hc_fp32_c(FFTZ_VOID *out, const FFTZ_VOID *cout,
                                        const FFTZ_VOID *tw,
                                        const aoclfftz_pack_rdft_t *params)
{
    FFTZ_INTP n0_by2 = params->n0_by2;
    FFTZ_FLOAT *p_out = (FFTZ_FLOAT *)out;
    const FFTZ_FLOAT *p_cout = (const FFTZ_FLOAT *)cout;
    const FFTZ_FLOAT *p_tw = (const FFTZ_FLOAT *)tw;

    const FFTZ_FLOAT *p_tw_re = p_tw;
    const FFTZ_FLOAT *p_tw_im = p_tw + 2 * (n0_by2 / 2 + 1);

    FFTZ_FLOAT cout_dc_re = p_cout[0];
    FFTZ_FLOAT cout_dc_im = p_cout[1];
    p_out[0] = cout_dc_re + cout_dc_im;
    p_out[1] = (FFTZ_FLOAT)0;
    p_out[2 * n0_by2] = cout_dc_re - cout_dc_im;
    p_out[2 * n0_by2 + 1] = (FFTZ_FLOAT)0;

    FFTZ_INTP num_pairs = REAL_PACKED_NUM_PAIRS(n0_by2);
    FFTZ_INTP count;
    for (count = 1; count <= num_pairs; count++)
    {
        recombine_to_hc_pair_fp32(p_out, p_cout, p_tw_re, p_tw_im, n0_by2,
                                  count);
    }

    if ((n0_by2 & 1) == 0)
    {
        FFTZ_INTP middle = n0_by2 / 2;
        p_out[2 * middle] = p_cout[2 * middle];
        p_out[2 * middle + 1] = -p_cout[2 * middle + 1];
    }
}

static FFTZ_VOID recombine_to_hc_fp64_c(FFTZ_VOID *out, const FFTZ_VOID *cout,
                                        const FFTZ_VOID *tw,
                                        const aoclfftz_pack_rdft_t *params)
{
    FFTZ_INTP n0_by2 = params->n0_by2;
    FFTZ_DOUBLE *p_out = (FFTZ_DOUBLE *)out;
    const FFTZ_DOUBLE *p_cout = (const FFTZ_DOUBLE *)cout;
    const FFTZ_DOUBLE *p_tw = (const FFTZ_DOUBLE *)tw;

    const FFTZ_DOUBLE *p_tw_re = p_tw;
    const FFTZ_DOUBLE *p_tw_im = p_tw + 2 * (n0_by2 / 2 + 1);

    FFTZ_DOUBLE cout_dc_re = p_cout[0];
    FFTZ_DOUBLE cout_dc_im = p_cout[1];
    p_out[0] = cout_dc_re + cout_dc_im;
    p_out[1] = (FFTZ_DOUBLE)0;
    p_out[2 * n0_by2] = cout_dc_re - cout_dc_im;
    p_out[2 * n0_by2 + 1] = (FFTZ_DOUBLE)0;

    FFTZ_INTP num_pairs = REAL_PACKED_NUM_PAIRS(n0_by2);
    FFTZ_INTP count;
    for (count = 1; count <= num_pairs; count++)
    {
        recombine_to_hc_pair_fp64(p_out, p_cout, p_tw_re, p_tw_im, n0_by2,
                                  count);
    }

    if ((n0_by2 & 1) == 0)
    {
        FFTZ_INTP middle = n0_by2 / 2;
        p_out[2 * middle] = p_cout[2 * middle];
        p_out[2 * middle + 1] = -p_cout[2 * middle + 1];
    }
}

pack_rdft_ register_recombine_to_hc_c(FFTZ_UINT8 precision)
{
    if (precision == DT_FLOAT)
    {
        return recombine_to_hc_fp32_c;
    }
    else if (precision == DT_DOUBLE)
    {
        return recombine_to_hc_fp64_c;
    }
    else
    {
        return NULL;
    }
}
