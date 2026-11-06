// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file recombine_to_hc_avx512.c
 *
 * @brief Packed R2C recombine kernel (DIT last stage) with AVX-512 intrinsics.
 *
 * Recombines the M-point complex spectrum Z into the N/2 + 1 Hermitian output
 * bins X (N = 2M). The conjugate-pair butterfly and twiddle-table layout are
 * documented in packed_rdft_common.h. DC, Nyquist and the (even M)
 * middle bin are
 * handled separately. AVX-512 processes eight fp32 pairs or four fp64 pairs
 * per vector; remainders descend through AVX-256, AVX-128 and scalar helpers
 * as needed.
 *
 * @author Srirammaswamy Srinivasan
 */

#include "core/kernels/kernel.h"
#include "core/kernels/non_dft/real_packed/packed_rdft_avx256_common.h"
#include "core/kernels/non_dft/real_packed/packed_rdft_avx512_common.h"
#include "core/kernels/non_dft/real_packed/packed_rdft_common.h"

static FFTZ_VOID recombine_to_hc_fp32_avx512(FFTZ_VOID *out,
                                             const FFTZ_VOID *cout,
                                             const FFTZ_VOID *tw,
                                             const aoclfftz_pack_rdft_t *params)
{
    FFTZ_INTP n0_by2 = params->n0_by2;
    FFTZ_FLOAT *p_out = (FFTZ_FLOAT *)out;
    const FFTZ_FLOAT *p_cout = (const FFTZ_FLOAT *)cout;
    const FFTZ_FLOAT *p_tw_re = (const FFTZ_FLOAT *)tw;
    const FFTZ_FLOAT *p_tw_im = p_tw_re + 2 * (n0_by2 / 2 + 1);
    FFTZ_FLOAT cout_dc_re = p_cout[0];
    FFTZ_FLOAT cout_dc_im = p_cout[1];
    p_out[0] = cout_dc_re + cout_dc_im;
    p_out[1] = (FFTZ_FLOAT)0;
    p_out[2 * n0_by2] = cout_dc_re - cout_dc_im;
    p_out[2 * n0_by2 + 1] = (FFTZ_FLOAT)0;

    FFTZ_INTP num_pairs = REAL_PACKED_NUM_PAIRS(n0_by2);
    FFTZ_INTP remaining_pairs = num_pairs % NUM_SETS_512_S;
    FFTZ_INTP count;
    for (count = 1; count <= num_pairs - remaining_pairs;
         count += NUM_SETS_512_S)
    {
        recombine_to_hc_pair_avx512_fp32(p_out, p_cout, p_tw_re, p_tw_im,
                                         n0_by2, count);
    }
    if (remaining_pairs & NUM_SETS_256_S)
    {
        recombine_to_hc_pair_avx256_fp32(p_out, p_cout, p_tw_re, p_tw_im,
                                         n0_by2, count);
        count += NUM_SETS_256_S;
    }
    if (remaining_pairs & NUM_SETS_128_S)
    {
        recombine_to_hc_pair_avx128_fp32(p_out, p_cout, p_tw_re, p_tw_im,
                                         n0_by2, count);
        count += NUM_SETS_128_S;
    }
    if (remaining_pairs & NUM_SETS_C_S)
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

static FFTZ_VOID recombine_to_hc_fp64_avx512(FFTZ_VOID *out,
                                             const FFTZ_VOID *cout,
                                             const FFTZ_VOID *tw,
                                             const aoclfftz_pack_rdft_t *params)
{
    FFTZ_INTP n0_by2 = params->n0_by2;
    FFTZ_DOUBLE *p_out = (FFTZ_DOUBLE *)out;
    const FFTZ_DOUBLE *p_cout = (const FFTZ_DOUBLE *)cout;
    const FFTZ_DOUBLE *p_tw_re = (const FFTZ_DOUBLE *)tw;
    const FFTZ_DOUBLE *p_tw_im = p_tw_re + 2 * (n0_by2 / 2 + 1);
    FFTZ_DOUBLE cout_dc_re = p_cout[0];
    FFTZ_DOUBLE cout_dc_im = p_cout[1];
    p_out[0] = cout_dc_re + cout_dc_im;
    p_out[1] = (FFTZ_DOUBLE)0;
    p_out[2 * n0_by2] = cout_dc_re - cout_dc_im;
    p_out[2 * n0_by2 + 1] = (FFTZ_DOUBLE)0;

    FFTZ_INTP num_pairs = REAL_PACKED_NUM_PAIRS(n0_by2);
    FFTZ_INTP remaining_pairs = num_pairs % NUM_SETS_512_D;
    FFTZ_INTP count;
    for (count = 1; count <= num_pairs - remaining_pairs;
         count += NUM_SETS_512_D)
    {
        recombine_to_hc_pair_avx512_fp64(p_out, p_cout, p_tw_re, p_tw_im,
                                         n0_by2, count);
    }
    if (remaining_pairs & NUM_SETS_256_D)
    {
        recombine_to_hc_pair_avx256_fp64(p_out, p_cout, p_tw_re, p_tw_im,
                                         n0_by2, count);
        count += NUM_SETS_256_D;
    }
    if (remaining_pairs & NUM_SETS_128_D)
    {
        recombine_to_hc_pair_avx128_fp64(p_out, p_cout, p_tw_re, p_tw_im,
                                         n0_by2, count);
    }
    if ((n0_by2 & 1) == 0)
    {
        FFTZ_INTP middle = n0_by2 / 2;
        p_out[2 * middle] = p_cout[2 * middle];
        p_out[2 * middle + 1] = -p_cout[2 * middle + 1];
    }
}

pack_rdft_ register_recombine_to_hc_avx512(FFTZ_UINT8 precision)
{
    if (precision == DT_FLOAT)
    {
        return recombine_to_hc_fp32_avx512;
    }
    else if (precision == DT_DOUBLE)
    {
        return recombine_to_hc_fp64_avx512;
    }
    else
    {
        return NULL;
    }
}
