// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file packed_rdft_avx256_common.h
 *
 * @brief 256-bit conjugate-pair butterflies shared by the real-packed
 * recombine and separate kernels.
 *
 * Same bodies as the 128-bit helpers at twice the width; see
 * packed_rdft_avx128_common.h for the algebra. Chains in the 128-bit
 * header so an AVX256 translation unit gets its tail helper from one
 * include.
 *
 * @author Srirammaswamy Srinivasan
 */

#ifndef AOCLFFTZ_PACKED_RDFT_AVX256_COMMON_H
#define AOCLFFTZ_PACKED_RDFT_AVX256_COMMON_H

#include "core/kernels/non_dft/real_packed/packed_rdft_avx128_common.h"

/**
 * @brief Reverses the order of the complex numbers held in a 256-bit register
 * for single precision floating point: [c0..c3] -> [c3..c0].
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 2, other: 0}
#define REV_CPLX_256_S(v)                                                      \
    _mm256_permute_ps(_mm256_permute2f128_ps((v), (v), 0x01), 0x4E)

/**
 * @brief Reverses the order of the complex numbers held in a 256-bit register
 * for double precision floating point: [c0,c1] -> [c1,c0].
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 1, other: 0}
#define REV_CPLX_256_D(v) _mm256_permute2f128_pd((v), (v), 0x01)

static const __m256 v_half_256_s = {0.5f, 0.5f, 0.5f, 0.5f,
                                    0.5f, 0.5f, 0.5f, 0.5f};
static const __m256d v_half_256_d = {0.5, 0.5, 0.5, 0.5};
static const __m256 v_sign_256_s = {-2.0f, 2.0f, -2.0f, 2.0f,
                                    -2.0f, 2.0f, -2.0f, 2.0f};
static const __m256d v_sign_256_d = {-2.0, 2.0, -2.0, 2.0};

/**
 * Packed twiddle: SWAP_RI of operand `d`, then complex mul by (tw_re, tw_im).
 * Direction is conjugation of `d` in the caller, not a sign in this helper.
 */
static inline __m256 apply_packed_twiddle_256_s(__m256 d, __m256 tw_re,
                                                __m256 tw_im)
{
    __m256 rot_arg = SWAP_RI_256_S(d);
    __m256 rot_tmp = _mm256_mul_ps(tw_im, d);
    return _mm256_addsub_ps(_mm256_mul_ps(tw_re, rot_arg), rot_tmp);
}

static inline __m256d apply_packed_twiddle_256_d(__m256d d, __m256d tw_re,
                                                 __m256d tw_im)
{
    __m256d rot_arg = SWAP_RI_256_D(d);
    __m256d rot_tmp = _mm256_mul_pd(tw_im, d);
    return _mm256_addsub_pd(_mm256_mul_pd(tw_re, rot_arg), rot_tmp);
}

static inline FFTZ_VOID recombine_to_hc_pair_bases_avx256_fp32(
    FFTZ_FLOAT *p_out_k, FFTZ_FLOAT *p_out_mk, const FFTZ_FLOAT *p_cout_k,
    const FFTZ_FLOAT *p_cout_mk, const FFTZ_FLOAT *p_tw_re,
    const FFTZ_FLOAT *p_tw_im, FFTZ_INTP n0_by2, FFTZ_INTP k)
{
    const FFTZ_INTP mk_base = 2 * (n0_by2 - k - NUM_SETS_256_S + 1);
    __m256 v_cout_k = _mm256_loadu_ps(p_cout_k + 2 * k);
    __m256 v_cout_mk_raw = _mm256_loadu_ps(p_cout_mk + mk_base);
    __m256 v_cout_mk = REV_CPLX_256_S(v_cout_mk_raw);
    __m256 v_cout_mk_conj = CONJ_256_S(v_cout_mk);
    __m256 v_sum = _mm256_add_ps(v_cout_k, v_cout_mk_conj);
    __m256 v_diff_conj = CONJ_256_S(_mm256_sub_ps(v_cout_mk_conj, v_cout_k));
    __m256 v_tw_re = _mm256_loadu_ps(p_tw_re + 2 * k);
    __m256 v_tw_im = _mm256_loadu_ps(p_tw_im + 2 * k);
    __m256 v_rot = apply_packed_twiddle_256_s(v_diff_conj, v_tw_re, v_tw_im);
    __m256 v_out_k = _mm256_add_ps(_mm256_mul_ps(v_half_256_s, v_sum), v_rot);
    __m256 v_out_mk =
        CONJ_256_S(_mm256_sub_ps(_mm256_mul_ps(v_half_256_s, v_sum), v_rot));
    _mm256_storeu_ps(p_out_k + 2 * k, v_out_k);
    _mm256_storeu_ps(p_out_mk + mk_base, REV_CPLX_256_S(v_out_mk));
}

static inline FFTZ_VOID recombine_to_hc_pair_bases_avx256_fp64(
    FFTZ_DOUBLE *p_out_k, FFTZ_DOUBLE *p_out_mk, const FFTZ_DOUBLE *p_cout_k,
    const FFTZ_DOUBLE *p_cout_mk, const FFTZ_DOUBLE *p_tw_re,
    const FFTZ_DOUBLE *p_tw_im, FFTZ_INTP n0_by2, FFTZ_INTP k)
{
    const FFTZ_INTP mk_base = 2 * (n0_by2 - k - NUM_SETS_256_D + 1);
    __m256d v_cout_k = _mm256_loadu_pd(p_cout_k + 2 * k);
    __m256d v_cout_mk_raw = _mm256_loadu_pd(p_cout_mk + mk_base);
    __m256d v_cout_mk = REV_CPLX_256_D(v_cout_mk_raw);
    __m256d v_cout_mk_conj = CONJ_256_D(v_cout_mk);
    __m256d v_sum = _mm256_add_pd(v_cout_k, v_cout_mk_conj);
    __m256d v_diff_conj = CONJ_256_D(_mm256_sub_pd(v_cout_mk_conj, v_cout_k));
    __m256d v_tw_re = _mm256_loadu_pd(p_tw_re + 2 * k);
    __m256d v_tw_im = _mm256_loadu_pd(p_tw_im + 2 * k);
    __m256d v_rot = apply_packed_twiddle_256_d(v_diff_conj, v_tw_re, v_tw_im);
    __m256d v_out_k = _mm256_add_pd(_mm256_mul_pd(v_half_256_d, v_sum), v_rot);
    __m256d v_out_mk =
        CONJ_256_D(_mm256_sub_pd(_mm256_mul_pd(v_half_256_d, v_sum), v_rot));
    _mm256_storeu_pd(p_out_k + 2 * k, v_out_k);
    _mm256_storeu_pd(p_out_mk + mk_base, REV_CPLX_256_D(v_out_mk));
}

static inline FFTZ_VOID separate_from_hc_pair_bases_avx256_fp32(
    FFTZ_FLOAT *p_cout_k, FFTZ_FLOAT *p_cout_mk, const FFTZ_FLOAT *p_in_k,
    const FFTZ_FLOAT *p_in_mk, const FFTZ_FLOAT *p_tw_re,
    const FFTZ_FLOAT *p_tw_im, FFTZ_INTP n0_by2, FFTZ_INTP k)
{
    const FFTZ_INTP mk_base = 2 * (n0_by2 - k - NUM_SETS_256_S + 1);
    __m256 v_in_k = _mm256_loadu_ps(p_in_k + 2 * k);
    __m256 v_in_mk_raw = _mm256_loadu_ps(p_in_mk + mk_base);
    __m256 v_in_mk = REV_CPLX_256_S(v_in_mk_raw);
    __m256 v_in_mk_conj = CONJ_256_S(v_in_mk);
    __m256 v_base = _mm256_add_ps(v_in_k, v_in_mk_conj);
    __m256 v_delta = _mm256_sub_ps(v_in_k, v_in_mk_conj);
    __m256 v_tw_re = _mm256_loadu_ps(p_tw_re + 2 * k);
    __m256 v_tw_im = _mm256_loadu_ps(p_tw_im + 2 * k);
    __m256 v_corr_half = apply_packed_twiddle_256_s(v_delta, v_tw_re, v_tw_im);
    __m256 v_corr = _mm256_mul_ps(v_corr_half, v_sign_256_s);
    __m256 v_cout_mk = CONJ_256_S(_mm256_sub_ps(v_base, v_corr));
    _mm256_storeu_ps(p_cout_k + 2 * k, _mm256_add_ps(v_base, v_corr));
    _mm256_storeu_ps(p_cout_mk + mk_base, REV_CPLX_256_S(v_cout_mk));
}

static inline FFTZ_VOID separate_from_hc_pair_bases_avx256_fp64(
    FFTZ_DOUBLE *p_cout_k, FFTZ_DOUBLE *p_cout_mk, const FFTZ_DOUBLE *p_in_k,
    const FFTZ_DOUBLE *p_in_mk, const FFTZ_DOUBLE *p_tw_re,
    const FFTZ_DOUBLE *p_tw_im, FFTZ_INTP n0_by2, FFTZ_INTP k)
{
    const FFTZ_INTP mk_base = 2 * (n0_by2 - k - NUM_SETS_256_D + 1);
    __m256d v_in_k = _mm256_loadu_pd(p_in_k + 2 * k);
    __m256d v_in_mk_raw = _mm256_loadu_pd(p_in_mk + mk_base);
    __m256d v_in_mk = REV_CPLX_256_D(v_in_mk_raw);
    __m256d v_in_mk_conj = CONJ_256_D(v_in_mk);
    __m256d v_base = _mm256_add_pd(v_in_k, v_in_mk_conj);
    __m256d v_delta = _mm256_sub_pd(v_in_k, v_in_mk_conj);
    __m256d v_tw_re = _mm256_loadu_pd(p_tw_re + 2 * k);
    __m256d v_tw_im = _mm256_loadu_pd(p_tw_im + 2 * k);
    __m256d v_corr_half = apply_packed_twiddle_256_d(v_delta, v_tw_re, v_tw_im);
    __m256d v_corr = _mm256_mul_pd(v_corr_half, v_sign_256_d);
    __m256d v_cout_mk = CONJ_256_D(_mm256_sub_pd(v_base, v_corr));
    _mm256_storeu_pd(p_cout_k + 2 * k, _mm256_add_pd(v_base, v_corr));
    _mm256_storeu_pd(p_cout_mk + mk_base, REV_CPLX_256_D(v_cout_mk));
}

/**
 * @brief Recombines conjugate pairs into Hermitian output bins.
 *
 * Processes NUM_SETS_256_S fp32 pairs starting at k and mirrored pairs ending
 * at n0_by2-k.
 */
static inline FFTZ_VOID recombine_to_hc_pair_avx256_fp32(
    FFTZ_FLOAT *p_out, const FFTZ_FLOAT *p_cout, const FFTZ_FLOAT *p_tw_re,
    const FFTZ_FLOAT *p_tw_im, FFTZ_INTP n0_by2, FFTZ_INTP k)
{
    recombine_to_hc_pair_bases_avx256_fp32(p_out, p_out, p_cout, p_cout,
                                           p_tw_re, p_tw_im, n0_by2, k);
}

/**
 * @brief Recombines conjugate pairs into Hermitian output bins.
 *
 * Processes NUM_SETS_256_D fp64 pairs starting at k and mirrored pairs ending
 * at n0_by2-k.
 */
static inline FFTZ_VOID recombine_to_hc_pair_avx256_fp64(
    FFTZ_DOUBLE *p_out, const FFTZ_DOUBLE *p_cout, const FFTZ_DOUBLE *p_tw_re,
    const FFTZ_DOUBLE *p_tw_im, FFTZ_INTP n0_by2, FFTZ_INTP k)
{
    recombine_to_hc_pair_bases_avx256_fp64(p_out, p_out, p_cout, p_cout,
                                           p_tw_re, p_tw_im, n0_by2, k);
}

/**
 * @brief Separates Hermitian pairs into packed complex values.
 *
 * Reconstructs the double-length child spectrum for NUM_SETS_256_S
 * fp32 pairs starting at k and mirrored pairs ending at n0_by2-k.
 */
static inline FFTZ_VOID separate_from_hc_pair_avx256_fp32(
    FFTZ_FLOAT *p_cout, const FFTZ_FLOAT *p_in, const FFTZ_FLOAT *p_tw_re,
    const FFTZ_FLOAT *p_tw_im, FFTZ_INTP n0_by2, FFTZ_INTP k)
{
    separate_from_hc_pair_bases_avx256_fp32(p_cout, p_cout, p_in, p_in, p_tw_re,
                                            p_tw_im, n0_by2, k);
}

/**
 * @brief Separates Hermitian pairs into packed complex values.
 *
 * Reconstructs the double-length child spectrum for NUM_SETS_256_D
 * fp64 pairs starting at k and mirrored pairs ending at n0_by2-k.
 */
static inline FFTZ_VOID separate_from_hc_pair_avx256_fp64(
    FFTZ_DOUBLE *p_cout, const FFTZ_DOUBLE *p_in, const FFTZ_DOUBLE *p_tw_re,
    const FFTZ_DOUBLE *p_tw_im, FFTZ_INTP n0_by2, FFTZ_INTP k)
{
    separate_from_hc_pair_bases_avx256_fp64(p_cout, p_cout, p_in, p_in, p_tw_re,
                                            p_tw_im, n0_by2, k);
}

static inline FFTZ_VOID recombine_to_hc_row_pair_fp32_avx256(
    FFTZ_FLOAT *p_out_row, FFTZ_FLOAT *p_out_mirror,
    const FFTZ_FLOAT *p_cout_row, const FFTZ_FLOAT *p_cout_mirror,
    const FFTZ_FLOAT *p_tw_re, const FFTZ_FLOAT *p_tw_im, FFTZ_INTP n0_by2,
    FFTZ_INTP num_pairs)
{
    recombine_to_hc_dc_nyq_pair_fp32(p_out_row, p_out_mirror, p_cout_row,
                                     p_cout_mirror, n0_by2, 1);
    FFTZ_INTP remaining = num_pairs % NUM_SETS_256_S;
    FFTZ_INTP count;
    for (count = 1; count <= num_pairs - remaining; count += NUM_SETS_256_S)
    {
        recombine_to_hc_pair_bases_avx256_fp32(p_out_row, p_out_mirror,
                                               p_cout_row, p_cout_mirror,
                                               p_tw_re, p_tw_im, n0_by2, count);
    }
    if (remaining & NUM_SETS_128_S)
    {
        recombine_to_hc_pair_bases_avx128_fp32(p_out_row, p_out_mirror,
                                               p_cout_row, p_cout_mirror,
                                               p_tw_re, p_tw_im, n0_by2, count);
        count += NUM_SETS_128_S;
    }
    if (remaining & NUM_SETS_C_S)
    {
        recombine_to_hc_pair_bases_fp32(p_out_row, p_out_mirror, p_cout_row,
                                        p_cout_mirror, p_tw_re, p_tw_im, n0_by2,
                                        count);
    }
    for (count = 1; count <= num_pairs - remaining; count += NUM_SETS_256_S)
    {
        recombine_to_hc_pair_bases_avx256_fp32(p_out_mirror, p_out_row,
                                               p_cout_mirror, p_cout_row,
                                               p_tw_re, p_tw_im, n0_by2, count);
    }
    if (remaining & NUM_SETS_128_S)
    {
        recombine_to_hc_pair_bases_avx128_fp32(p_out_mirror, p_out_row,
                                               p_cout_mirror, p_cout_row,
                                               p_tw_re, p_tw_im, n0_by2, count);
        count += NUM_SETS_128_S;
    }
    if (remaining & NUM_SETS_C_S)
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

static inline FFTZ_VOID recombine_to_hc_row_pair_fp64_avx256(
    FFTZ_DOUBLE *p_out_row, FFTZ_DOUBLE *p_out_mirror,
    const FFTZ_DOUBLE *p_cout_row, const FFTZ_DOUBLE *p_cout_mirror,
    const FFTZ_DOUBLE *p_tw_re, const FFTZ_DOUBLE *p_tw_im, FFTZ_INTP n0_by2,
    FFTZ_INTP num_pairs)
{
    recombine_to_hc_dc_nyq_pair_fp64(p_out_row, p_out_mirror, p_cout_row,
                                     p_cout_mirror, n0_by2, 1);
    FFTZ_INTP remaining = num_pairs % NUM_SETS_256_D;
    FFTZ_INTP count;
    for (count = 1; count <= num_pairs - remaining; count += NUM_SETS_256_D)
    {
        recombine_to_hc_pair_bases_avx256_fp64(p_out_row, p_out_mirror,
                                               p_cout_row, p_cout_mirror,
                                               p_tw_re, p_tw_im, n0_by2, count);
    }
    if (remaining & NUM_SETS_128_D)
    {
        recombine_to_hc_pair_bases_avx128_fp64(p_out_row, p_out_mirror,
                                               p_cout_row, p_cout_mirror,
                                               p_tw_re, p_tw_im, n0_by2, count);
    }
    for (count = 1; count <= num_pairs - remaining; count += NUM_SETS_256_D)
    {
        recombine_to_hc_pair_bases_avx256_fp64(p_out_mirror, p_out_row,
                                               p_cout_mirror, p_cout_row,
                                               p_tw_re, p_tw_im, n0_by2, count);
    }
    if (remaining & NUM_SETS_128_D)
    {
        recombine_to_hc_pair_bases_avx128_fp64(p_out_mirror, p_out_row,
                                               p_cout_mirror, p_cout_row,
                                               p_tw_re, p_tw_im, n0_by2, count);
    }
    if ((n0_by2 & 1) == 0)
    {
        recombine_to_hc_pair_middle_fp64(p_out_row, p_out_mirror, p_cout_row,
                                         p_cout_mirror, n0_by2);
    }
}

static inline FFTZ_VOID separate_from_hc_row_pair_fp32_avx256(
    FFTZ_FLOAT *p_cout_row, FFTZ_FLOAT *p_cout_mirror,
    const FFTZ_FLOAT *p_in_row, const FFTZ_FLOAT *p_in_mirror,
    const FFTZ_FLOAT *p_tw_re, const FFTZ_FLOAT *p_tw_im, FFTZ_INTP n0_by2,
    FFTZ_INTP num_pairs)
{
    separate_from_hc_cout0_fp32(p_cout_row, p_in_row, p_in_mirror, n0_by2);
    separate_from_hc_cout0_fp32(p_cout_mirror, p_in_mirror, p_in_row, n0_by2);
    FFTZ_INTP remaining = num_pairs % NUM_SETS_256_S;
    FFTZ_INTP count;
    for (count = 1; count <= num_pairs - remaining; count += NUM_SETS_256_S)
    {
        separate_from_hc_pair_bases_avx256_fp32(p_cout_row, p_cout_mirror,
                                                p_in_row, p_in_mirror, p_tw_re,
                                                p_tw_im, n0_by2, count);
    }
    if (remaining & NUM_SETS_128_S)
    {
        separate_from_hc_pair_bases_avx128_fp32(p_cout_row, p_cout_mirror,
                                                p_in_row, p_in_mirror, p_tw_re,
                                                p_tw_im, n0_by2, count);
        count += NUM_SETS_128_S;
    }
    if (remaining & NUM_SETS_C_S)
    {
        separate_from_hc_pair_bases_fp32(p_cout_row, p_cout_mirror, p_in_row,
                                         p_in_mirror, p_tw_re, p_tw_im, n0_by2,
                                         count);
    }
    for (count = 1; count <= num_pairs - remaining; count += NUM_SETS_256_S)
    {
        separate_from_hc_pair_bases_avx256_fp32(p_cout_mirror, p_cout_row,
                                                p_in_mirror, p_in_row, p_tw_re,
                                                p_tw_im, n0_by2, count);
    }
    if (remaining & NUM_SETS_128_S)
    {
        separate_from_hc_pair_bases_avx128_fp32(p_cout_mirror, p_cout_row,
                                                p_in_mirror, p_in_row, p_tw_re,
                                                p_tw_im, n0_by2, count);
        count += NUM_SETS_128_S;
    }
    if (remaining & NUM_SETS_C_S)
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

static inline FFTZ_VOID separate_from_hc_row_pair_fp64_avx256(
    FFTZ_DOUBLE *p_cout_row, FFTZ_DOUBLE *p_cout_mirror,
    const FFTZ_DOUBLE *p_in_row, const FFTZ_DOUBLE *p_in_mirror,
    const FFTZ_DOUBLE *p_tw_re, const FFTZ_DOUBLE *p_tw_im, FFTZ_INTP n0_by2,
    FFTZ_INTP num_pairs)
{
    separate_from_hc_cout0_fp64(p_cout_row, p_in_row, p_in_mirror, n0_by2);
    separate_from_hc_cout0_fp64(p_cout_mirror, p_in_mirror, p_in_row, n0_by2);
    FFTZ_INTP remaining = num_pairs % NUM_SETS_256_D;
    FFTZ_INTP count;
    for (count = 1; count <= num_pairs - remaining; count += NUM_SETS_256_D)
    {
        separate_from_hc_pair_bases_avx256_fp64(p_cout_row, p_cout_mirror,
                                                p_in_row, p_in_mirror, p_tw_re,
                                                p_tw_im, n0_by2, count);
    }
    if (remaining & NUM_SETS_128_D)
    {
        separate_from_hc_pair_bases_avx128_fp64(p_cout_row, p_cout_mirror,
                                                p_in_row, p_in_mirror, p_tw_re,
                                                p_tw_im, n0_by2, count);
    }
    for (count = 1; count <= num_pairs - remaining; count += NUM_SETS_256_D)
    {
        separate_from_hc_pair_bases_avx256_fp64(p_cout_mirror, p_cout_row,
                                                p_in_mirror, p_in_row, p_tw_re,
                                                p_tw_im, n0_by2, count);
    }
    if (remaining & NUM_SETS_128_D)
    {
        separate_from_hc_pair_bases_avx128_fp64(p_cout_mirror, p_cout_row,
                                                p_in_mirror, p_in_row, p_tw_re,
                                                p_tw_im, n0_by2, count);
    }
    if ((n0_by2 & 1) == 0)
    {
        separate_from_hc_pair_middle_fp64(p_cout_row, p_cout_mirror, p_in_row,
                                          p_in_mirror, n0_by2);
    }
}

#endif // AOCLFFTZ_PACKED_RDFT_AVX256_COMMON_H
