// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file packed_rdft_common.h
 *
 * @brief Common helpers shared by real-packed 1D and 3D kernels.
 *
 * recombine (R2C, DIT last stage) and separate (C2R, DIF first stage) share
 * one conjugate-pair walk over the packed M-point spectrum, so the pair count
 * and the scalar butterflies live here once instead of in every ISA variant.
 *
 * @author Srirammaswamy Srinivasan
 */

#ifndef AOCLFFTZ_PACKED_RDFT_COMMON_H
#define AOCLFFTZ_PACKED_RDFT_COMMON_H

#include "core/kernels/kernel.h"

// Pair count for child length n0_by2 (bins 1 .. floor((n0_by2-1)/2)).
// DC, Nyquist, and the even-n0_by2 middle bin are done outside this loop.
#define REAL_PACKED_NUM_PAIRS(n0_by2) (((n0_by2) - 1) / 2)

// recombine (R2C): DIT last stage. Recombine child pair Z[k], Z[M-k] with one
// twiddle multiply and write Hermitian bins X[k], X[M-k] (N = 2M).
// Twiddle table is pre-broadcast with the 0.5 scale and conjugation baked in
// (tw_re = 0.5*cos, tw_im = -0.5*sin):
//   sum_re     = cout_k_re + cout_mk_re ; diff_im = cout_k_im - cout_mk_im
//   rot_arg_re = cout_k_im + cout_mk_im
//   rot_arg_im = cout_mk_re - cout_k_re
//   rot_re = tw_re*rot_arg_re - tw_im*rot_arg_im
//   rot_im = tw_re*rot_arg_im + tw_im*rot_arg_re
//   X[k]   = (0.5*sum_re + rot_re,  0.5*diff_im + rot_im)
//   X[M-k] = (0.5*sum_re - rot_re,  rot_im - 0.5*diff_im)
//
// `out_k` may alias `cout_k`, and `out_mk` may alias `cout_mk`. All four input
// scalars are loaded before either output is written, so both overwrites are
// safe. `static inline` lets every ISA main loop and SIMD tail share this code
// without call overhead.

static inline FFTZ_VOID recombine_to_hc_pair_bases_fp32(
    FFTZ_FLOAT *out_k, FFTZ_FLOAT *out_mk, const FFTZ_FLOAT *cout_k,
    const FFTZ_FLOAT *cout_mk, const FFTZ_FLOAT *twr, const FFTZ_FLOAT *twi,
    FFTZ_INTP n0_by2, FFTZ_INTP k)
{
    FFTZ_FLOAT cout_k_re = cout_k[2 * k], cout_k_im = cout_k[2 * k + 1];
    FFTZ_FLOAT cout_mk_re = cout_mk[2 * (n0_by2 - k)];
    FFTZ_FLOAT cout_mk_im = cout_mk[2 * (n0_by2 - k) + 1];
    FFTZ_FLOAT sum_re = cout_k_re + cout_mk_re;
    FFTZ_FLOAT diff_im = cout_k_im - cout_mk_im;
    FFTZ_FLOAT rot_arg_re = cout_k_im + cout_mk_im;
    FFTZ_FLOAT rot_arg_im = cout_mk_re - cout_k_re;
    FFTZ_FLOAT tw_re = twr[2 * k], tw_im = twi[2 * k];
    FFTZ_FLOAT rot_re = tw_re * rot_arg_re - tw_im * rot_arg_im;
    FFTZ_FLOAT rot_im = tw_re * rot_arg_im + tw_im * rot_arg_re;
    FFTZ_FLOAT half_re = (FFTZ_FLOAT)0.5 * sum_re;
    FFTZ_FLOAT half_im = (FFTZ_FLOAT)0.5 * diff_im;
    out_k[2 * k] = half_re + rot_re;
    out_k[2 * k + 1] = half_im + rot_im;
    out_mk[2 * (n0_by2 - k)] = half_re - rot_re;
    out_mk[2 * (n0_by2 - k) + 1] = rot_im - half_im;
}

/**
 * @brief Combines one fp64 conjugate pair into Hermitian output bins.
 */
static inline FFTZ_VOID recombine_to_hc_pair_bases_fp64(
    FFTZ_DOUBLE *out_k, FFTZ_DOUBLE *out_mk, const FFTZ_DOUBLE *cout_k,
    const FFTZ_DOUBLE *cout_mk, const FFTZ_DOUBLE *twr, const FFTZ_DOUBLE *twi,
    FFTZ_INTP n0_by2, FFTZ_INTP k)
{
    FFTZ_DOUBLE cout_k_re = cout_k[2 * k], cout_k_im = cout_k[2 * k + 1];
    FFTZ_DOUBLE cout_mk_re = cout_mk[2 * (n0_by2 - k)];
    FFTZ_DOUBLE cout_mk_im = cout_mk[2 * (n0_by2 - k) + 1];
    FFTZ_DOUBLE sum_re = cout_k_re + cout_mk_re;
    FFTZ_DOUBLE diff_im = cout_k_im - cout_mk_im;
    FFTZ_DOUBLE rot_arg_re = cout_k_im + cout_mk_im;
    FFTZ_DOUBLE rot_arg_im = cout_mk_re - cout_k_re;
    FFTZ_DOUBLE tw_re = twr[2 * k], tw_im = twi[2 * k];
    FFTZ_DOUBLE rot_re = tw_re * rot_arg_re - tw_im * rot_arg_im;
    FFTZ_DOUBLE rot_im = tw_re * rot_arg_im + tw_im * rot_arg_re;
    FFTZ_DOUBLE half_re = (FFTZ_DOUBLE)0.5 * sum_re;
    FFTZ_DOUBLE half_im = (FFTZ_DOUBLE)0.5 * diff_im;
    out_k[2 * k] = half_re + rot_re;
    out_k[2 * k + 1] = half_im + rot_im;
    out_mk[2 * (n0_by2 - k)] = half_re - rot_re;
    out_mk[2 * (n0_by2 - k) + 1] = rot_im - half_im;
}

/**
 * @brief Exactly recombines the fp32 middle bin of a conjugate row pair.
 */
static inline FFTZ_VOID
recombine_to_hc_pair_middle_fp32(FFTZ_FLOAT *out_k, FFTZ_FLOAT *out_mk,
                                 const FFTZ_FLOAT *cout_k,
                                 const FFTZ_FLOAT *cout_mk, FFTZ_INTP n0_by2)
{
    FFTZ_INTP middle = n0_by2 / 2;
    FFTZ_FLOAT cout_k_re = cout_k[2 * middle];
    FFTZ_FLOAT cout_k_im = cout_k[2 * middle + 1];
    FFTZ_FLOAT cout_mk_re = cout_mk[2 * middle];
    FFTZ_FLOAT cout_mk_im = cout_mk[2 * middle + 1];
    out_k[2 * middle] = cout_mk_re;
    out_k[2 * middle + 1] = -cout_mk_im;
    out_mk[2 * middle] = cout_k_re;
    out_mk[2 * middle + 1] = -cout_k_im;
}

/**
 * @brief Exactly recombines the fp64 middle bin of a conjugate row pair.
 */
static inline FFTZ_VOID
recombine_to_hc_pair_middle_fp64(FFTZ_DOUBLE *out_k, FFTZ_DOUBLE *out_mk,
                                 const FFTZ_DOUBLE *cout_k,
                                 const FFTZ_DOUBLE *cout_mk, FFTZ_INTP n0_by2)
{
    FFTZ_INTP middle = n0_by2 / 2;
    FFTZ_DOUBLE cout_k_re = cout_k[2 * middle];
    FFTZ_DOUBLE cout_k_im = cout_k[2 * middle + 1];
    FFTZ_DOUBLE cout_mk_re = cout_mk[2 * middle];
    FFTZ_DOUBLE cout_mk_im = cout_mk[2 * middle + 1];
    out_k[2 * middle] = cout_mk_re;
    out_k[2 * middle + 1] = -cout_mk_im;
    out_mk[2 * middle] = cout_k_re;
    out_mk[2 * middle + 1] = -cout_k_im;
}

/**
 * @brief Recombines conjugate pairs into Hermitian output bins.
 *
 * Processes NUM_SETS_C_S fp32 pairs starting at k and mirrored pairs ending at
 * n0_by2-k.
 */
static inline FFTZ_VOID recombine_to_hc_pair_fp32(FFTZ_FLOAT *out,
                                                  const FFTZ_FLOAT *cout,
                                                  const FFTZ_FLOAT *twr,
                                                  const FFTZ_FLOAT *twi,
                                                  FFTZ_INTP n0_by2, FFTZ_INTP k)
{
    recombine_to_hc_pair_bases_fp32(out, out, cout, cout, twr, twi, n0_by2, k);
}

/**
 * @brief Recombines conjugate pairs into Hermitian output bins.
 *
 * Processes NUM_SETS_C_D fp64 pairs starting at k and mirrored pairs ending at
 * n0_by2-k.
 */
static inline FFTZ_VOID recombine_to_hc_pair_fp64(FFTZ_DOUBLE *out,
                                                  const FFTZ_DOUBLE *cout,
                                                  const FFTZ_DOUBLE *twr,
                                                  const FFTZ_DOUBLE *twi,
                                                  FFTZ_INTP n0_by2, FFTZ_INTP k)
{
    recombine_to_hc_pair_bases_fp64(out, out, cout, cout, twr, twi, n0_by2, k);
}

// separate (C2R): one conjugate-pair butterfly reconstructing 2*Z[k], 2*Z[M-k]
// from Hermitian input bins X[k], X[M-k] (N = 2M):
//   tw_re = p_tw_re[2k] ; tw_im = p_tw_im[2k]
//   sum_re   = x_k_re + x_mk_re ; diff_im  = x_k_im - x_mk_im
//   delta_re = x_k_re - x_mk_re ; delta_im = x_k_im + x_mk_im
//   corr_re = 2*( tw_re*delta_re + tw_im*delta_im)
//   corr_im = 2*(-tw_im*delta_re + tw_re*delta_im)
//   Z[k]   = (sum_re - corr_im, corr_re + diff_im)
//   Z[M-k] = (sum_re + corr_im, corr_re - diff_im)
//
// `in` must not alias `cout`: the solver uses a separate scratch spectrum since
// each pair is written only after both `in` entries are read. `static
// inline` so every ISA variant's main loop and SIMD tail share one copy with
// no call overhead.
/**
 * @brief Separates one fp32 Hermitian pair into packed complex values.
 *
 * Reconstructs 2*Z for bin k and mirror bin n0_by2-k, each with its own base
 * pointer so 3D rows can pair across the grid.
 */
static inline FFTZ_VOID separate_from_hc_pair_bases_fp32(
    FFTZ_FLOAT *cout_k, FFTZ_FLOAT *cout_mk, const FFTZ_FLOAT *in_k,
    const FFTZ_FLOAT *in_mk, const FFTZ_FLOAT *twr, const FFTZ_FLOAT *twi,
    FFTZ_INTP n0_by2, FFTZ_INTP k)
{
    FFTZ_FLOAT x_k_re = in_k[2 * k], x_k_im = in_k[2 * k + 1];
    FFTZ_FLOAT x_mk_re = in_mk[2 * (n0_by2 - k)],
               x_mk_im = in_mk[2 * (n0_by2 - k) + 1];
    FFTZ_FLOAT sum_re = x_k_re + x_mk_re, diff_im = x_k_im - x_mk_im;
    FFTZ_FLOAT delta_re = x_k_re - x_mk_re, delta_im = x_k_im + x_mk_im;
    FFTZ_FLOAT tw_re = twr[2 * k], tw_im = twi[2 * k];
    FFTZ_FLOAT corr_re = (FFTZ_FLOAT)2 * (tw_re * delta_re + tw_im * delta_im);
    FFTZ_FLOAT corr_im = (FFTZ_FLOAT)2 * (-tw_im * delta_re + tw_re * delta_im);
    cout_k[2 * k] = sum_re - corr_im;
    cout_k[2 * k + 1] = corr_re + diff_im;
    cout_mk[2 * (n0_by2 - k)] = sum_re + corr_im;
    cout_mk[2 * (n0_by2 - k) + 1] = corr_re - diff_im;
}

/**
 * @brief Separates one fp64 Hermitian pair into packed complex values.
 */
static inline FFTZ_VOID separate_from_hc_pair_bases_fp64(
    FFTZ_DOUBLE *cout_k, FFTZ_DOUBLE *cout_mk, const FFTZ_DOUBLE *in_k,
    const FFTZ_DOUBLE *in_mk, const FFTZ_DOUBLE *twr, const FFTZ_DOUBLE *twi,
    FFTZ_INTP n0_by2, FFTZ_INTP k)
{
    FFTZ_DOUBLE x_k_re = in_k[2 * k], x_k_im = in_k[2 * k + 1];
    FFTZ_DOUBLE x_mk_re = in_mk[2 * (n0_by2 - k)],
                x_mk_im = in_mk[2 * (n0_by2 - k) + 1];
    FFTZ_DOUBLE sum_re = x_k_re + x_mk_re, diff_im = x_k_im - x_mk_im;
    FFTZ_DOUBLE delta_re = x_k_re - x_mk_re, delta_im = x_k_im + x_mk_im;
    FFTZ_DOUBLE tw_re = twr[2 * k], tw_im = twi[2 * k];
    FFTZ_DOUBLE corr_re =
        (FFTZ_DOUBLE)2 * (tw_re * delta_re + tw_im * delta_im);
    FFTZ_DOUBLE corr_im =
        (FFTZ_DOUBLE)2 * (-tw_im * delta_re + tw_re * delta_im);
    cout_k[2 * k] = sum_re - corr_im;
    cout_k[2 * k + 1] = corr_re + diff_im;
    cout_mk[2 * (n0_by2 - k)] = sum_re + corr_im;
    cout_mk[2 * (n0_by2 - k) + 1] = corr_re - diff_im;
}

/**
 * @brief Exactly separates the fp32 middle bin of a conjugate row pair.
 */
static inline FFTZ_VOID
separate_from_hc_pair_middle_fp32(FFTZ_FLOAT *cout_k, FFTZ_FLOAT *cout_mk,
                                  const FFTZ_FLOAT *in_k,
                                  const FFTZ_FLOAT *in_mk, FFTZ_INTP n0_by2)
{
    FFTZ_INTP middle = n0_by2 / 2;
    FFTZ_FLOAT in_k_re = in_k[2 * middle];
    FFTZ_FLOAT in_k_im = in_k[2 * middle + 1];
    FFTZ_FLOAT in_mk_re = in_mk[2 * middle];
    FFTZ_FLOAT in_mk_im = in_mk[2 * middle + 1];
    cout_k[2 * middle] = (FFTZ_FLOAT)2 * in_mk_re;
    cout_k[2 * middle + 1] = (FFTZ_FLOAT)-2 * in_mk_im;
    cout_mk[2 * middle] = (FFTZ_FLOAT)2 * in_k_re;
    cout_mk[2 * middle + 1] = (FFTZ_FLOAT)-2 * in_k_im;
}

/**
 * @brief Exactly separates the fp64 middle bin of a conjugate row pair.
 */
static inline FFTZ_VOID
separate_from_hc_pair_middle_fp64(FFTZ_DOUBLE *cout_k, FFTZ_DOUBLE *cout_mk,
                                  const FFTZ_DOUBLE *in_k,
                                  const FFTZ_DOUBLE *in_mk, FFTZ_INTP n0_by2)
{
    FFTZ_INTP middle = n0_by2 / 2;
    FFTZ_DOUBLE in_k_re = in_k[2 * middle];
    FFTZ_DOUBLE in_k_im = in_k[2 * middle + 1];
    FFTZ_DOUBLE in_mk_re = in_mk[2 * middle];
    FFTZ_DOUBLE in_mk_im = in_mk[2 * middle + 1];
    cout_k[2 * middle] = (FFTZ_DOUBLE)2 * in_mk_re;
    cout_k[2 * middle + 1] = (FFTZ_DOUBLE)-2 * in_mk_im;
    cout_mk[2 * middle] = (FFTZ_DOUBLE)2 * in_k_re;
    cout_mk[2 * middle + 1] = (FFTZ_DOUBLE)-2 * in_k_im;
}

// separate (C2R): DIF first stage. Separate Hermitian bins X[k], X[M-k] into
// the child pair that the M-point CFFT then transforms. Reconstructs
// 2*Z[k], 2*Z[M-k] from X[k], X[M-k] (N = 2M):
//   tw_re = p_tw_re[2k] ; tw_im = p_tw_im[2k]
//   sum_re   = x_k_re + x_mk_re ; diff_im  = x_k_im - x_mk_im
//   delta_re = x_k_re - x_mk_re ; delta_im = x_k_im + x_mk_im
//   corr_re = 2*( tw_re*delta_re + tw_im*delta_im)
//   corr_im = 2*(-tw_im*delta_re + tw_re*delta_im)
//   Z[k]   = (sum_re - corr_im, corr_re + diff_im)
//   Z[M-k] = (sum_re + corr_im, corr_re - diff_im)
//
// `cout` must not overwrite `in`: the solver uses a separate scratch spectrum
// since each pair is written only after both `in` entries are read. `static
// inline` so every ISA variant's main loop and SIMD tail share one copy with
// no call overhead.

/**
 * @brief Separates Hermitian pairs into packed complex values.
 *
 * Reconstructs the double-length child spectrum for NUM_SETS_C_S
 * fp32 pairs starting at k and mirrored pairs ending at n0_by2-k.
 */
static inline FFTZ_VOID
separate_from_hc_pair_fp32(FFTZ_FLOAT *cout, const FFTZ_FLOAT *in,
                           const FFTZ_FLOAT *twr, const FFTZ_FLOAT *twi,
                           FFTZ_INTP n0_by2, FFTZ_INTP k)
{
    separate_from_hc_pair_bases_fp32(cout, cout, in, in, twr, twi, n0_by2, k);
}

/**
 * @brief Separates Hermitian pairs into packed complex values.
 *
 * Reconstructs the double-length child spectrum for NUM_SETS_C_D
 * fp64 pairs starting at k and mirrored pairs ending at n0_by2-k.
 */
static inline FFTZ_VOID
separate_from_hc_pair_fp64(FFTZ_DOUBLE *cout, const FFTZ_DOUBLE *in,
                           const FFTZ_DOUBLE *twr, const FFTZ_DOUBLE *twi,
                           FFTZ_INTP n0_by2, FFTZ_INTP k)
{
    separate_from_hc_pair_bases_fp64(cout, cout, in, in, twr, twi, n0_by2, k);
}

/* k == 0 butterfly of a row pair: the mirror row supplies the conjugate
 * partner, and the row's DC and Nyquist bins fall out of the same even/odd
 * split the interior bins use with a unit twiddle. */
static inline FFTZ_VOID recombine_to_hc_dc_nyq_vals_fp32(
    FFTZ_FLOAT *out, FFTZ_FLOAT cout_row_re, FFTZ_FLOAT cout_row_im,
    FFTZ_FLOAT cout_mirror_re, FFTZ_FLOAT cout_mirror_im, FFTZ_INTP n0_by2)
{
    FFTZ_FLOAT conj_im = -cout_mirror_im;
    FFTZ_FLOAT even_re = (FFTZ_FLOAT)0.5 * (cout_row_re + cout_mirror_re);
    FFTZ_FLOAT even_im = (FFTZ_FLOAT)0.5 * (cout_row_im + conj_im);
    FFTZ_FLOAT odd_re = (FFTZ_FLOAT)0.5 * (cout_row_im - conj_im);
    FFTZ_FLOAT odd_im = (FFTZ_FLOAT)0.5 * (cout_mirror_re - cout_row_re);
    out[0] = even_re + odd_re;
    out[1] = even_im + odd_im;
    out[2 * n0_by2] = even_re - odd_re;
    out[2 * n0_by2 + 1] = even_im - odd_im;
}

/* Snapshot cout[0] of both rows first: combine aliases out==cout, so writing
 * the row's DC would otherwise clobber the partner load for the mirror row. */
static inline FFTZ_VOID recombine_to_hc_dc_nyq_pair_fp32(
    FFTZ_FLOAT *p_out_row, FFTZ_FLOAT *p_out_mirror,
    const FFTZ_FLOAT *p_cout_row, const FFTZ_FLOAT *p_cout_mirror,
    FFTZ_INTP n0_by2, FFTZ_INT32 two_rows)
{
    FFTZ_FLOAT row_re = p_cout_row[0];
    FFTZ_FLOAT row_im = p_cout_row[1];
    FFTZ_FLOAT mirror_re = p_cout_mirror[0];
    FFTZ_FLOAT mirror_im = p_cout_mirror[1];
    recombine_to_hc_dc_nyq_vals_fp32(p_out_row, row_re, row_im, mirror_re,
                                     mirror_im, n0_by2);
    if (two_rows)
    {
        recombine_to_hc_dc_nyq_vals_fp32(p_out_mirror, mirror_re, mirror_im,
                                         row_re, row_im, n0_by2);
    }
}

static inline FFTZ_VOID recombine_to_hc_dc_nyq_vals_fp64(
    FFTZ_DOUBLE *out, FFTZ_DOUBLE cout_row_re, FFTZ_DOUBLE cout_row_im,
    FFTZ_DOUBLE cout_mirror_re, FFTZ_DOUBLE cout_mirror_im, FFTZ_INTP n0_by2)
{
    FFTZ_DOUBLE conj_im = -cout_mirror_im;
    FFTZ_DOUBLE even_re = (FFTZ_DOUBLE)0.5 * (cout_row_re + cout_mirror_re);
    FFTZ_DOUBLE even_im = (FFTZ_DOUBLE)0.5 * (cout_row_im + conj_im);
    FFTZ_DOUBLE odd_re = (FFTZ_DOUBLE)0.5 * (cout_row_im - conj_im);
    FFTZ_DOUBLE odd_im = (FFTZ_DOUBLE)0.5 * (cout_mirror_re - cout_row_re);
    out[0] = even_re + odd_re;
    out[1] = even_im + odd_im;
    out[2 * n0_by2] = even_re - odd_re;
    out[2 * n0_by2 + 1] = even_im - odd_im;
}

static inline FFTZ_VOID recombine_to_hc_dc_nyq_pair_fp64(
    FFTZ_DOUBLE *p_out_row, FFTZ_DOUBLE *p_out_mirror,
    const FFTZ_DOUBLE *p_cout_row, const FFTZ_DOUBLE *p_cout_mirror,
    FFTZ_INTP n0_by2, FFTZ_INT32 two_rows)
{
    FFTZ_DOUBLE row_re = p_cout_row[0];
    FFTZ_DOUBLE row_im = p_cout_row[1];
    FFTZ_DOUBLE mirror_re = p_cout_mirror[0];
    FFTZ_DOUBLE mirror_im = p_cout_mirror[1];
    recombine_to_hc_dc_nyq_vals_fp64(p_out_row, row_re, row_im, mirror_re,
                                     mirror_im, n0_by2);
    if (two_rows)
    {
        recombine_to_hc_dc_nyq_vals_fp64(p_out_mirror, mirror_re, mirror_im,
                                         row_re, row_im, n0_by2);
    }
}

/* Inverse of the k == 0 butterfly: X[row,0] and the conjugate of the mirror
 * row's Nyquist bin rebuild cout[row,0]. */
static inline FFTZ_VOID
separate_from_hc_cout0_fp32(FFTZ_FLOAT *p_cout_row, const FFTZ_FLOAT *p_in_row,
                            const FFTZ_FLOAT *p_in_mirror, FFTZ_INTP n0_by2)
{
    FFTZ_FLOAT row_re = p_in_row[0];
    FFTZ_FLOAT row_im = p_in_row[1];
    FFTZ_FLOAT nyq_re = p_in_mirror[2 * n0_by2];
    FFTZ_FLOAT nyq_im = -p_in_mirror[2 * n0_by2 + 1];
    FFTZ_FLOAT diff_re = row_re - nyq_re;
    FFTZ_FLOAT diff_im = row_im - nyq_im;
    p_cout_row[0] = row_re + nyq_re - diff_im;
    p_cout_row[1] = row_im + nyq_im + diff_re;
}

static inline FFTZ_VOID
separate_from_hc_cout0_fp64(FFTZ_DOUBLE *p_cout_row,
                            const FFTZ_DOUBLE *p_in_row,
                            const FFTZ_DOUBLE *p_in_mirror, FFTZ_INTP n0_by2)
{
    FFTZ_DOUBLE row_re = p_in_row[0];
    FFTZ_DOUBLE row_im = p_in_row[1];
    FFTZ_DOUBLE nyq_re = p_in_mirror[2 * n0_by2];
    FFTZ_DOUBLE nyq_im = -p_in_mirror[2 * n0_by2 + 1];
    FFTZ_DOUBLE diff_re = row_re - nyq_re;
    FFTZ_DOUBLE diff_im = row_im - nyq_im;
    p_cout_row[0] = row_re + nyq_re - diff_im;
    p_cout_row[1] = row_im + nyq_im + diff_re;
}

static inline FFTZ_VOID recombine_to_hc_row_pair_fp32_c(
    FFTZ_FLOAT *p_out_row, FFTZ_FLOAT *p_out_mirror,
    const FFTZ_FLOAT *p_cout_row, const FFTZ_FLOAT *p_cout_mirror,
    const FFTZ_FLOAT *p_tw_re, const FFTZ_FLOAT *p_tw_im, FFTZ_INTP n0_by2,
    FFTZ_INTP num_pairs)
{
    recombine_to_hc_dc_nyq_pair_fp32(p_out_row, p_out_mirror, p_cout_row,
                                     p_cout_mirror, n0_by2, 1);
    FFTZ_INTP count;
    for (count = 1; count <= num_pairs; count++)
    {
        recombine_to_hc_pair_bases_fp32(p_out_row, p_out_mirror, p_cout_row,
                                        p_cout_mirror, p_tw_re, p_tw_im, n0_by2,
                                        count);
    }
    for (count = 1; count <= num_pairs; count++)
    {
        recombine_to_hc_pair_bases_fp32(p_out_mirror, p_out_row, p_cout_mirror,
                                        p_cout_row, p_tw_re, p_tw_im, n0_by2,
                                        count);
    }
    if ((n0_by2 & 1) == 0)
    {
        recombine_to_hc_pair_middle_fp32(p_out_row, p_out_mirror, p_cout_row,
                                         p_cout_mirror, n0_by2);
    }
}

static inline FFTZ_VOID recombine_to_hc_row_pair_fp64_c(
    FFTZ_DOUBLE *p_out_row, FFTZ_DOUBLE *p_out_mirror,
    const FFTZ_DOUBLE *p_cout_row, const FFTZ_DOUBLE *p_cout_mirror,
    const FFTZ_DOUBLE *p_tw_re, const FFTZ_DOUBLE *p_tw_im, FFTZ_INTP n0_by2,
    FFTZ_INTP num_pairs)
{
    recombine_to_hc_dc_nyq_pair_fp64(p_out_row, p_out_mirror, p_cout_row,
                                     p_cout_mirror, n0_by2, 1);
    FFTZ_INTP count;
    for (count = 1; count <= num_pairs; count++)
    {
        recombine_to_hc_pair_bases_fp64(p_out_row, p_out_mirror, p_cout_row,
                                        p_cout_mirror, p_tw_re, p_tw_im, n0_by2,
                                        count);
    }
    for (count = 1; count <= num_pairs; count++)
    {
        recombine_to_hc_pair_bases_fp64(p_out_mirror, p_out_row, p_cout_mirror,
                                        p_cout_row, p_tw_re, p_tw_im, n0_by2,
                                        count);
    }
    if ((n0_by2 & 1) == 0)
    {
        recombine_to_hc_pair_middle_fp64(p_out_row, p_out_mirror, p_cout_row,
                                         p_cout_mirror, n0_by2);
    }
}

static inline FFTZ_VOID separate_from_hc_row_pair_fp32_c(
    FFTZ_FLOAT *p_cout_row, FFTZ_FLOAT *p_cout_mirror,
    const FFTZ_FLOAT *p_in_row, const FFTZ_FLOAT *p_in_mirror,
    const FFTZ_FLOAT *p_tw_re, const FFTZ_FLOAT *p_tw_im, FFTZ_INTP n0_by2,
    FFTZ_INTP num_pairs)
{
    separate_from_hc_cout0_fp32(p_cout_row, p_in_row, p_in_mirror, n0_by2);
    separate_from_hc_cout0_fp32(p_cout_mirror, p_in_mirror, p_in_row, n0_by2);
    FFTZ_INTP count;
    for (count = 1; count <= num_pairs; count++)
    {
        separate_from_hc_pair_bases_fp32(p_cout_row, p_cout_mirror, p_in_row,
                                         p_in_mirror, p_tw_re, p_tw_im, n0_by2,
                                         count);
    }
    for (count = 1; count <= num_pairs; count++)
    {
        separate_from_hc_pair_bases_fp32(p_cout_mirror, p_cout_row, p_in_mirror,
                                         p_in_row, p_tw_re, p_tw_im, n0_by2,
                                         count);
    }
    if ((n0_by2 & 1) == 0)
    {
        separate_from_hc_pair_middle_fp32(p_cout_row, p_cout_mirror, p_in_row,
                                          p_in_mirror, n0_by2);
    }
}

static inline FFTZ_VOID separate_from_hc_row_pair_fp64_c(
    FFTZ_DOUBLE *p_cout_row, FFTZ_DOUBLE *p_cout_mirror,
    const FFTZ_DOUBLE *p_in_row, const FFTZ_DOUBLE *p_in_mirror,
    const FFTZ_DOUBLE *p_tw_re, const FFTZ_DOUBLE *p_tw_im, FFTZ_INTP n0_by2,
    FFTZ_INTP num_pairs)
{
    separate_from_hc_cout0_fp64(p_cout_row, p_in_row, p_in_mirror, n0_by2);
    separate_from_hc_cout0_fp64(p_cout_mirror, p_in_mirror, p_in_row, n0_by2);
    FFTZ_INTP count;
    for (count = 1; count <= num_pairs; count++)
    {
        separate_from_hc_pair_bases_fp64(p_cout_row, p_cout_mirror, p_in_row,
                                         p_in_mirror, p_tw_re, p_tw_im, n0_by2,
                                         count);
    }
    for (count = 1; count <= num_pairs; count++)
    {
        separate_from_hc_pair_bases_fp64(p_cout_mirror, p_cout_row, p_in_mirror,
                                         p_in_row, p_tw_re, p_tw_im, n0_by2,
                                         count);
    }
    if ((n0_by2 & 1) == 0)
    {
        separate_from_hc_pair_middle_fp64(p_cout_row, p_cout_mirror, p_in_row,
                                          p_in_mirror, n0_by2);
    }
}

#endif // AOCLFFTZ_PACKED_RDFT_COMMON_H
