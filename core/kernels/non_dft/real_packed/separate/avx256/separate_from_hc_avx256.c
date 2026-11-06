// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file separate_from_hc_avx256.c
 *
 * @brief Packed C2R separate kernel (DIF first stage) with AVX-256 intrinsics.
 *
 * Reconstructs the double-length child spectrum consumed by the child
 * inverse FFT, from the N/2 + 1 Hermitian input bins X (N = 2M). Per-pair
 * butterfly and twiddle-table layout are documented in
 * packed_rdft_common.h. The
 * factor of two lets the backward CFFT(M) produce the unnormalized N-point
 * real inverse directly. DC, Nyquist and the (even M) middle bin are handled
 * separately. AVX-256 processes four fp32 pairs or two fp64 pairs per vector;
 * remainders descend through AVX-128 and scalar helpers as needed.
 *
 * @author Srirammaswamy Srinivasan
 */

#include "core/kernels/kernel.h"
#include "core/kernels/non_dft/real_packed/packed_rdft_avx256_common.h"
#include "core/kernels/non_dft/real_packed/packed_rdft_common.h"

static FFTZ_VOID
separate_from_hc_fp32_avx256(FFTZ_VOID *cout, const FFTZ_VOID *in,
                             const FFTZ_VOID *tw,
                             const aoclfftz_pack_rdft_t *params)
{
    FFTZ_INTP n0_by2 = params->n0_by2;
    FFTZ_FLOAT *p_cout = (FFTZ_FLOAT *)cout;
    const FFTZ_FLOAT *p_in = (const FFTZ_FLOAT *)in;
    const FFTZ_FLOAT *p_tw_re = (const FFTZ_FLOAT *)tw;
    const FFTZ_FLOAT *p_tw_im = p_tw_re + 2 * (n0_by2 / 2 + 1);

    p_cout[0] = p_in[0] + p_in[2 * n0_by2];
    p_cout[1] = p_in[0] - p_in[2 * n0_by2];

    FFTZ_INTP num_pairs = REAL_PACKED_NUM_PAIRS(n0_by2);
    FFTZ_INTP remaining_pairs = num_pairs % NUM_SETS_256_S;
    FFTZ_INTP count;
    for (count = 1; count <= num_pairs - remaining_pairs;
         count += NUM_SETS_256_S)
    {
        separate_from_hc_pair_avx256_fp32(p_cout, p_in, p_tw_re, p_tw_im,
                                          n0_by2, count);
    }
    if (remaining_pairs & NUM_SETS_128_S)
    {
        separate_from_hc_pair_avx128_fp32(p_cout, p_in, p_tw_re, p_tw_im,
                                          n0_by2, count);
        count += NUM_SETS_128_S;
    }
    if (remaining_pairs & NUM_SETS_C_S)
    {
        separate_from_hc_pair_fp32(p_cout, p_in, p_tw_re, p_tw_im, n0_by2,
                                   count);
    }
    if ((n0_by2 & 1) == 0)
    {
        FFTZ_INTP middle = n0_by2 / 2;
        p_cout[2 * middle] = (FFTZ_FLOAT)2 * p_in[2 * middle];
        p_cout[2 * middle + 1] = (FFTZ_FLOAT)-2 * p_in[2 * middle + 1];
    }
}

static FFTZ_VOID
separate_from_hc_fp64_avx256(FFTZ_VOID *cout, const FFTZ_VOID *in,
                             const FFTZ_VOID *tw,
                             const aoclfftz_pack_rdft_t *params)
{
    FFTZ_INTP n0_by2 = params->n0_by2;
    FFTZ_DOUBLE *p_cout = (FFTZ_DOUBLE *)cout;
    const FFTZ_DOUBLE *p_in = (const FFTZ_DOUBLE *)in;
    const FFTZ_DOUBLE *p_tw_re = (const FFTZ_DOUBLE *)tw;
    const FFTZ_DOUBLE *p_tw_im = p_tw_re + 2 * (n0_by2 / 2 + 1);

    p_cout[0] = p_in[0] + p_in[2 * n0_by2];
    p_cout[1] = p_in[0] - p_in[2 * n0_by2];

    FFTZ_INTP num_pairs = REAL_PACKED_NUM_PAIRS(n0_by2);
    FFTZ_INTP remaining_pairs = num_pairs % NUM_SETS_256_D;
    FFTZ_INTP count;
    for (count = 1; count <= num_pairs - remaining_pairs;
         count += NUM_SETS_256_D)
    {
        separate_from_hc_pair_avx256_fp64(p_cout, p_in, p_tw_re, p_tw_im,
                                          n0_by2, count);
    }
    if (remaining_pairs & NUM_SETS_128_D)
    {
        separate_from_hc_pair_avx128_fp64(p_cout, p_in, p_tw_re, p_tw_im,
                                          n0_by2, count);
    }
    if ((n0_by2 & 1) == 0)
    {
        FFTZ_INTP middle = n0_by2 / 2;
        p_cout[2 * middle] = (FFTZ_DOUBLE)2 * p_in[2 * middle];
        p_cout[2 * middle + 1] = (FFTZ_DOUBLE)-2 * p_in[2 * middle + 1];
    }
}

pack_rdft_ register_separate_from_hc_avx256(FFTZ_UINT8 precision)
{
    if (precision == DT_FLOAT)
    {
        return separate_from_hc_fp32_avx256;
    }
    else if (precision == DT_DOUBLE)
    {
        return separate_from_hc_fp64_avx256;
    }
    else
    {
        return NULL;
    }
}
