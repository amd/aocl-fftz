// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file packed_rdft_3d_common.h
 *
 * @brief ISA-parameterized 3D packed recombine/separate traversal template.
 *
 * This header intentionally has no include guard. Each packed 3D kernel source
 * defines its function names, row-pair functions, and kernel variant before
 * including this template. Sources under real_packed/recombine/{c,avx*} and
 * real_packed/separate/{c,avx*} provide these definitions.
 *
 * @author Srirammaswamy Srinivasan
 */

#include "core/kernels/non_dft/real_packed/packed_rdft_common.h"

/**
 * Defines typed packed 3D traversals that directly enumerate self planes and
 * conjugate plane pairs.
 *
 * A self plane is plane 0, plus plane n2/2 when n2 is even. It is its own
 * conjugate mirror, so rows within it are paired with their mirrored rows.
 *
 * A conjugate plane pair contains two distinct planes p and n2-p. The traversal
 * handles the pair once, matching each row in p with its mirrored row in n2-p.
 *
 * For an outer coordinate (plane, row), its conjugate coordinate is:
 *
 *   mirror(plane, row) = ((n2 - plane) % n2, (n1 - row) % n1)
 *
 * Example for n2 = 6 planes and n1 = 6 rows:
 *   Plane decomposition:
 *
 *     self planes:  P0, P3
 *     plane pairs:  P1 <-> P5, P2 <-> P4
 *
 *   Row work inside each self plane (P0 and P3):
 *
 *     rows:  r0    r1     r2    r3     r4    r5
 *     work: [r0]  (r1 <-> r5)  (r2 <-> r4)  [r3]
 *            |                                |
 *            +--------- pack_rdft_1d ---------+
 *
 *     Parentheses denote one row-pair kernel call. Brackets denote
 *     self-conjugate rows processed by pack_rdft_1d.
 *
 *   Row work for conjugate planes P1 <-> P5:
 *
 *     P1:r0 <-> P5:r0
 *     P1:r1 <-> P5:r5
 *     P1:r2 <-> P5:r4
 *     P1:r3 <-> P5:r3
 *     P1:r4 <-> P5:r2
 *     P1:r5 <-> P5:r1
 *
 *     P1 pointer: r0, r1, r2, r3, r4, r5
 *     P5 pointer: r0, r5, r4, r3, r2, r1
 *
 *   P2 <-> P4 follows the same row walk. For odd n1 or n2, the corresponding
 *   midpoint self row or plane does not exist.
 *
 * This traversal never visits the duplicate half of the outer coordinates.
 * Pointer increments replace repeated mirror-index and flattened-row
 * calculations.
 */

/**
 * Processes one FP32 self-conjugate plane. Called by PACK_RDFT_KNAME_FP32 for
 * plane 0 and, when n2 is even, plane n2/2. Self-conjugate rows use the 1D
 * kernel; mirrored rows use the FP32 row-pair kernel.
 */
static FFTZ_VOID packed_rdft_3d_self_plane_fp32(
    FFTZ_FLOAT *p_out_plane, const FFTZ_FLOAT *p_in_plane, const FFTZ_VOID *tw,
    const aoclfftz_pack_rdft_t *params, FFTZ_INTP out_row_stride,
    FFTZ_INTP in_row_stride)
{
    FFTZ_INTP n0_by2 = params->n0_by2;
    FFTZ_INTP n1 = params->n1;
    const FFTZ_FLOAT *p_tw_re = (const FFTZ_FLOAT *)tw;
    const FFTZ_FLOAT *p_tw_im = p_tw_re + 2 * (n0_by2 / 2 + 1);
    FFTZ_INTP num_pairs = REAL_PACKED_NUM_PAIRS(n0_by2);
    FFTZ_INTP row_conj_pairs = (n1 - 1) / 2;
    FFTZ_FLOAT *p_out_fwd = p_out_plane + out_row_stride;
    FFTZ_FLOAT *p_out_bwd = p_out_plane + (n1 - 1) * out_row_stride;
    const FFTZ_FLOAT *p_in_fwd = p_in_plane + in_row_stride;
    const FFTZ_FLOAT *p_in_bwd = p_in_plane + (n1 - 1) * in_row_stride;
    pack_rdft_ pack_rdft_1d = params->pack_rdft_1d;

    pack_rdft_1d(p_out_plane, p_in_plane, tw, params);
    for (FFTZ_INTP conj_pair = 0; conj_pair < row_conj_pairs; conj_pair++)
    {
        ROW_CONJ_PAIR_FP32(p_out_fwd, p_out_bwd, p_in_fwd, p_in_bwd, p_tw_re,
                           p_tw_im, n0_by2, num_pairs);
        p_out_fwd += out_row_stride;
        p_out_bwd -= out_row_stride;
        p_in_fwd += in_row_stride;
        p_in_bwd -= in_row_stride;
    }
    if ((n1 & 1) == 0)
    {
        pack_rdft_1d(p_out_fwd, p_in_fwd, tw, params);
    }
}

/**
 * Runs the FP32 packed 3D recombine/separate traversal. Registered through
 * REGISTER_KERNEL and called through params->pack_rdft for each packed 3D
 * transform. Processes self-conjugate planes and each conjugate plane pair
 * once.
 */
static FFTZ_VOID PACK_RDFT_KNAME_FP32(FFTZ_VOID *out, const FFTZ_VOID *in,
                                      const FFTZ_VOID *tw,
                                      const aoclfftz_pack_rdft_t *params)
{
    FFTZ_FLOAT *p_out = (FFTZ_FLOAT *)out;
    const FFTZ_FLOAT *p_in = (const FFTZ_FLOAT *)in;
    const FFTZ_FLOAT *p_tw_re = (const FFTZ_FLOAT *)tw;
    FFTZ_INTP n0_by2 = params->n0_by2;
    const FFTZ_FLOAT *p_tw_im = p_tw_re + 2 * (n0_by2 / 2 + 1);
    FFTZ_INTP num_pairs = REAL_PACKED_NUM_PAIRS(n0_by2);
    FFTZ_INTP n1 = params->n1;
    FFTZ_INTP n2 = params->n2;
#if defined(PACKED_KERNEL_VARIANT_R2C)
    FFTZ_INTP out_row_stride = DATA_STRIDE * (n0_by2 + 1);
    FFTZ_INTP in_row_stride = DATA_STRIDE * params->cout_row_stride;
#else
    FFTZ_INTP out_row_stride = DATA_STRIDE * params->cout_row_stride;
    FFTZ_INTP in_row_stride = DATA_STRIDE * (n0_by2 + 1);
#endif
    FFTZ_INTP out_plane_stride = n1 * out_row_stride;
    FFTZ_INTP in_plane_stride = n1 * in_row_stride;
    FFTZ_INTP out_last_row_offset = (n1 - 1) * out_row_stride;
    FFTZ_INTP in_last_row_offset = (n1 - 1) * in_row_stride;
    FFTZ_INTP plane_conj_pairs = (n2 - 1) / 2;

    FFTZ_FLOAT *p_out_fwd_plane = p_out + out_plane_stride;
    FFTZ_FLOAT *p_out_bwd_plane = p_out + (n2 - 1) * out_plane_stride;
    const FFTZ_FLOAT *p_in_fwd_plane = p_in + in_plane_stride;
    const FFTZ_FLOAT *p_in_bwd_plane = p_in + (n2 - 1) * in_plane_stride;

    packed_rdft_3d_self_plane_fp32(p_out, p_in, tw, params, out_row_stride,
                                   in_row_stride);

    for (FFTZ_INTP conj_pair = 0; conj_pair < plane_conj_pairs; conj_pair++)
    {
        FFTZ_FLOAT *p_out_fwd = p_out_fwd_plane;
        FFTZ_FLOAT *p_out_bwd = p_out_bwd_plane;
        const FFTZ_FLOAT *p_in_fwd = p_in_fwd_plane;
        const FFTZ_FLOAT *p_in_bwd = p_in_bwd_plane;

        ROW_CONJ_PAIR_FP32(p_out_fwd, p_out_bwd, p_in_fwd, p_in_bwd, p_tw_re,
                           p_tw_im, n0_by2, num_pairs);
        p_out_fwd += out_row_stride;
        p_out_bwd += out_last_row_offset;
        p_in_fwd += in_row_stride;
        p_in_bwd += in_last_row_offset;

        for (FFTZ_INTP i1 = 1; i1 < n1; i1++)
        {
            ROW_CONJ_PAIR_FP32(p_out_fwd, p_out_bwd, p_in_fwd, p_in_bwd,
                               p_tw_re, p_tw_im, n0_by2, num_pairs);
            p_out_fwd += out_row_stride;
            p_out_bwd -= out_row_stride;
            p_in_fwd += in_row_stride;
            p_in_bwd -= in_row_stride;
        }
        p_out_fwd_plane += out_plane_stride;
        p_out_bwd_plane -= out_plane_stride;
        p_in_fwd_plane += in_plane_stride;
        p_in_bwd_plane -= in_plane_stride;
    }
    if ((n2 & 1) == 0)
    {
        packed_rdft_3d_self_plane_fp32(p_out_fwd_plane, p_in_fwd_plane, tw,
                                       params, out_row_stride, in_row_stride);
    }
}

/**
 * Processes one FP64 self-conjugate plane. Called by PACK_RDFT_KNAME_FP64 for
 * plane 0 and, when n2 is even, plane n2/2. Self-conjugate rows use the 1D
 * kernel; mirrored rows use the FP64 row-pair kernel.
 */
static FFTZ_VOID packed_rdft_3d_self_plane_fp64(
    FFTZ_DOUBLE *p_out_plane, const FFTZ_DOUBLE *p_in_plane,
    const FFTZ_VOID *tw, const aoclfftz_pack_rdft_t *params,
    FFTZ_INTP out_row_stride, FFTZ_INTP in_row_stride)
{
    FFTZ_INTP n0_by2 = params->n0_by2;
    FFTZ_INTP n1 = params->n1;
    const FFTZ_DOUBLE *p_tw_re = (const FFTZ_DOUBLE *)tw;
    const FFTZ_DOUBLE *p_tw_im = p_tw_re + 2 * (n0_by2 / 2 + 1);
    FFTZ_INTP num_pairs = REAL_PACKED_NUM_PAIRS(n0_by2);
    FFTZ_INTP row_conj_pairs = (n1 - 1) / 2;
    FFTZ_DOUBLE *p_out_fwd = p_out_plane + out_row_stride;
    FFTZ_DOUBLE *p_out_bwd = p_out_plane + (n1 - 1) * out_row_stride;
    const FFTZ_DOUBLE *p_in_fwd = p_in_plane + in_row_stride;
    const FFTZ_DOUBLE *p_in_bwd = p_in_plane + (n1 - 1) * in_row_stride;
    pack_rdft_ pack_rdft_1d = params->pack_rdft_1d;

    pack_rdft_1d(p_out_plane, p_in_plane, tw, params);
    for (FFTZ_INTP conj_pair = 0; conj_pair < row_conj_pairs; conj_pair++)
    {
        ROW_CONJ_PAIR_FP64(p_out_fwd, p_out_bwd, p_in_fwd, p_in_bwd, p_tw_re,
                           p_tw_im, n0_by2, num_pairs);
        p_out_fwd += out_row_stride;
        p_out_bwd -= out_row_stride;
        p_in_fwd += in_row_stride;
        p_in_bwd -= in_row_stride;
    }
    if ((n1 & 1) == 0)
    {
        pack_rdft_1d(p_out_fwd, p_in_fwd, tw, params);
    }
}

/**
 * Runs the FP64 packed 3D recombine/separate traversal. Registered through
 * REGISTER_KERNEL and called through params->pack_rdft for each packed 3D
 * transform. Processes self-conjugate planes and each conjugate plane pair
 * once.
 */
static FFTZ_VOID PACK_RDFT_KNAME_FP64(FFTZ_VOID *out, const FFTZ_VOID *in,
                                      const FFTZ_VOID *tw,
                                      const aoclfftz_pack_rdft_t *params)
{
    FFTZ_DOUBLE *p_out = (FFTZ_DOUBLE *)out;
    const FFTZ_DOUBLE *p_in = (const FFTZ_DOUBLE *)in;
    const FFTZ_DOUBLE *p_tw_re = (const FFTZ_DOUBLE *)tw;
    FFTZ_INTP n0_by2 = params->n0_by2;
    const FFTZ_DOUBLE *p_tw_im = p_tw_re + 2 * (n0_by2 / 2 + 1);
    FFTZ_INTP num_pairs = REAL_PACKED_NUM_PAIRS(n0_by2);
    FFTZ_INTP n1 = params->n1;
    FFTZ_INTP n2 = params->n2;
#if defined(PACKED_KERNEL_VARIANT_R2C)
    FFTZ_INTP out_row_stride = DATA_STRIDE * (n0_by2 + 1);
    FFTZ_INTP in_row_stride = DATA_STRIDE * params->cout_row_stride;
#else
    FFTZ_INTP out_row_stride = DATA_STRIDE * params->cout_row_stride;
    FFTZ_INTP in_row_stride = DATA_STRIDE * (n0_by2 + 1);
#endif
    FFTZ_INTP out_plane_stride = n1 * out_row_stride;
    FFTZ_INTP in_plane_stride = n1 * in_row_stride;
    FFTZ_INTP out_last_row_offset = (n1 - 1) * out_row_stride;
    FFTZ_INTP in_last_row_offset = (n1 - 1) * in_row_stride;
    FFTZ_INTP plane_conj_pairs = (n2 - 1) / 2;

    FFTZ_DOUBLE *p_out_fwd_plane = p_out + out_plane_stride;
    FFTZ_DOUBLE *p_out_bwd_plane = p_out + (n2 - 1) * out_plane_stride;
    const FFTZ_DOUBLE *p_in_fwd_plane = p_in + in_plane_stride;
    const FFTZ_DOUBLE *p_in_bwd_plane = p_in + (n2 - 1) * in_plane_stride;

    packed_rdft_3d_self_plane_fp64(p_out, p_in, tw, params, out_row_stride,
                                   in_row_stride);

    for (FFTZ_INTP conj_pair = 0; conj_pair < plane_conj_pairs; conj_pair++)
    {
        FFTZ_DOUBLE *p_out_fwd = p_out_fwd_plane;
        FFTZ_DOUBLE *p_out_bwd = p_out_bwd_plane;
        const FFTZ_DOUBLE *p_in_fwd = p_in_fwd_plane;
        const FFTZ_DOUBLE *p_in_bwd = p_in_bwd_plane;

        ROW_CONJ_PAIR_FP64(p_out_fwd, p_out_bwd, p_in_fwd, p_in_bwd, p_tw_re,
                           p_tw_im, n0_by2, num_pairs);
        p_out_fwd += out_row_stride;
        p_out_bwd += out_last_row_offset;
        p_in_fwd += in_row_stride;
        p_in_bwd += in_last_row_offset;

        for (FFTZ_INTP i1 = 1; i1 < n1; i1++)
        {
            ROW_CONJ_PAIR_FP64(p_out_fwd, p_out_bwd, p_in_fwd, p_in_bwd,
                               p_tw_re, p_tw_im, n0_by2, num_pairs);
            p_out_fwd += out_row_stride;
            p_out_bwd -= out_row_stride;
            p_in_fwd += in_row_stride;
            p_in_bwd -= in_row_stride;
        }
        p_out_fwd_plane += out_plane_stride;
        p_out_bwd_plane -= out_plane_stride;
        p_in_fwd_plane += in_plane_stride;
        p_in_bwd_plane -= in_plane_stride;
    }
    if ((n2 & 1) == 0)
    {
        packed_rdft_3d_self_plane_fp64(p_out_fwd_plane, p_in_fwd_plane, tw,
                                       params, out_row_stride, in_row_stride);
    }
}

pack_rdft_ REGISTER_KERNEL(FFTZ_UINT8 precision)
{
    if (precision == DT_FLOAT)
    {
        return PACK_RDFT_KNAME_FP32;
    }
    else if (precision == DT_DOUBLE)
    {
        return PACK_RDFT_KNAME_FP64;
    }
    else
    {
        return NULL;
    }
}
