// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file packed_rdft_avx128_common.h
 *
 * @brief 128-bit conjugate-pair butterflies shared by the real-packed
 * recombine and separate kernels.
 *
 * AVX256 tails descend 256 -> 128 -> scalar, so these per-width helpers
 * hold one copy of each butterfly body instead of one per tail step.
 * `static inline` (not macros) so `__m128`/`__m128d` operands keep their
 * types and the compiler inlines with no call overhead.
 *
 * Each helper processes NUM_SETS_128_S/D conjugate pairs starting at k: the
 * ascending block k .. k+W-1 pairs with the descending block n0_by2-k ..
 * n0_by2-k-W+1, loaded contiguously and reversed. For double precision W = 1,
 * so REV_CPLX_128_D is the identity.
 *
 * @author Srirammaswamy Srinivasan
 */

#ifndef AOCLFFTZ_PACKED_RDFT_AVX128_COMMON_H
#define AOCLFFTZ_PACKED_RDFT_AVX128_COMMON_H

#include "core/kernels/non_dft/real_packed/packed_rdft_common.h"
#include "core/kernels/simd_includes/simd_common.h"

/**
 * @brief Reverses the order of the complex numbers held in a 128-bit register
 * for single precision floating point: [c0,c1] -> [c1,c0].
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 1, other: 0}
#define REV_CPLX_128_S(v) _mm_shuffle_ps((v), (v), 0x4E)

/**
 * @brief A 128-bit double precision register holds a single complex number, so
 * reversing it is the identity. Provided so width-generic kernel code can name
 * the operation uniformly.
 */
#define REV_CPLX_128_D(v) (v)

// k-independent broadcast constants. File-scope so the pair loop does not
// rebuild them each call.
static const __m128 v_half_128_s = {0.5f, 0.5f, 0.5f, 0.5f};
static const __m128d v_half_128_d = {0.5, 0.5};
static const __m128 v_sign_128_s = {-2.0f, 2.0f, -2.0f, 2.0f};
static const __m128d v_sign_128_d = {-2.0, 2.0};

/**
 * Packed twiddle: SWAP_RI of operand `d`, then complex mul by (tw_re, tw_im).
 * Direction is conjugation of `d` in the caller, not a sign in this helper.
 */
static inline __m128 apply_packed_twiddle_128_s(__m128 d, __m128 tw_re,
                                                __m128 tw_im)
{
    __m128 rot_arg = SWAP_RI_128_S(d);
    __m128 rot_tmp = _mm_mul_ps(tw_im, d);
    return _mm_addsub_ps(_mm_mul_ps(tw_re, rot_arg), rot_tmp);
}

static inline __m128d apply_packed_twiddle_128_d(__m128d d, __m128d tw_re,
                                                 __m128d tw_im)
{
    __m128d rot_arg = SWAP_RI_128_D(d);
    __m128d rot_tmp = _mm_mul_pd(tw_im, d);
    return _mm_addsub_pd(_mm_mul_pd(tw_re, rot_arg), rot_tmp);
}

static inline FFTZ_VOID recombine_to_hc_pair_bases_avx128_fp32(
    FFTZ_FLOAT *p_out_k, FFTZ_FLOAT *p_out_mk, const FFTZ_FLOAT *p_cout_k,
    const FFTZ_FLOAT *p_cout_mk, const FFTZ_FLOAT *p_tw_re,
    const FFTZ_FLOAT *p_tw_im, FFTZ_INTP n0_by2, FFTZ_INTP k)
{
    const FFTZ_INTP mk_base = 2 * (n0_by2 - k - NUM_SETS_128_S + 1);
    __m128 v_cout_k = _mm_loadu_ps(p_cout_k + 2 * k);
    __m128 v_cout_mk_raw = _mm_loadu_ps(p_cout_mk + mk_base);
    __m128 v_cout_mk = REV_CPLX_128_S(v_cout_mk_raw);
    __m128 v_cout_mk_conj = CONJ_128_S(v_cout_mk);
    __m128 v_sum = _mm_add_ps(v_cout_k, v_cout_mk_conj);
    __m128 v_diff_conj = CONJ_128_S(_mm_sub_ps(v_cout_mk_conj, v_cout_k));
    __m128 v_tw_re = _mm_loadu_ps(p_tw_re + 2 * k);
    __m128 v_tw_im = _mm_loadu_ps(p_tw_im + 2 * k);
    __m128 v_rot = apply_packed_twiddle_128_s(v_diff_conj, v_tw_re, v_tw_im);
    __m128 v_out_k = _mm_add_ps(_mm_mul_ps(v_half_128_s, v_sum), v_rot);
    __m128 v_out_mk =
        CONJ_128_S(_mm_sub_ps(_mm_mul_ps(v_half_128_s, v_sum), v_rot));
    _mm_storeu_ps(p_out_k + 2 * k, v_out_k);
    _mm_storeu_ps(p_out_mk + mk_base, REV_CPLX_128_S(v_out_mk));
}

static inline FFTZ_VOID recombine_to_hc_pair_bases_avx128_fp64(
    FFTZ_DOUBLE *p_out_k, FFTZ_DOUBLE *p_out_mk, const FFTZ_DOUBLE *p_cout_k,
    const FFTZ_DOUBLE *p_cout_mk, const FFTZ_DOUBLE *p_tw_re,
    const FFTZ_DOUBLE *p_tw_im, FFTZ_INTP n0_by2, FFTZ_INTP k)
{
    const FFTZ_INTP mk_base = 2 * (n0_by2 - k - NUM_SETS_128_D + 1);
    __m128d v_cout_k = _mm_loadu_pd(p_cout_k + 2 * k);
    __m128d v_cout_mk_raw = _mm_loadu_pd(p_cout_mk + mk_base);
    __m128d v_cout_mk = REV_CPLX_128_D(v_cout_mk_raw);
    __m128d v_cout_mk_conj = CONJ_128_D(v_cout_mk);
    __m128d v_sum = _mm_add_pd(v_cout_k, v_cout_mk_conj);
    __m128d v_diff_conj = CONJ_128_D(_mm_sub_pd(v_cout_mk_conj, v_cout_k));
    __m128d v_tw_re = _mm_loadu_pd(p_tw_re + 2 * k);
    __m128d v_tw_im = _mm_loadu_pd(p_tw_im + 2 * k);
    __m128d v_rot = apply_packed_twiddle_128_d(v_diff_conj, v_tw_re, v_tw_im);
    __m128d v_out_k = _mm_add_pd(_mm_mul_pd(v_half_128_d, v_sum), v_rot);
    __m128d v_out_mk =
        CONJ_128_D(_mm_sub_pd(_mm_mul_pd(v_half_128_d, v_sum), v_rot));
    _mm_storeu_pd(p_out_k + 2 * k, v_out_k);
    _mm_storeu_pd(p_out_mk + mk_base, REV_CPLX_128_D(v_out_mk));
}

static inline FFTZ_VOID separate_from_hc_pair_bases_avx128_fp32(
    FFTZ_FLOAT *p_cout_k, FFTZ_FLOAT *p_cout_mk, const FFTZ_FLOAT *p_in_k,
    const FFTZ_FLOAT *p_in_mk, const FFTZ_FLOAT *p_tw_re,
    const FFTZ_FLOAT *p_tw_im, FFTZ_INTP n0_by2, FFTZ_INTP k)
{
    const FFTZ_INTP mk_base = 2 * (n0_by2 - k - NUM_SETS_128_S + 1);
    __m128 v_in_k = _mm_loadu_ps(p_in_k + 2 * k);
    __m128 v_in_mk_raw = _mm_loadu_ps(p_in_mk + mk_base);
    __m128 v_in_mk = REV_CPLX_128_S(v_in_mk_raw);
    __m128 v_in_mk_conj = CONJ_128_S(v_in_mk);
    __m128 v_base = _mm_add_ps(v_in_k, v_in_mk_conj);
    __m128 v_delta = _mm_sub_ps(v_in_k, v_in_mk_conj);
    __m128 v_tw_re = _mm_loadu_ps(p_tw_re + 2 * k);
    __m128 v_tw_im = _mm_loadu_ps(p_tw_im + 2 * k);
    __m128 v_corr_half = apply_packed_twiddle_128_s(v_delta, v_tw_re, v_tw_im);
    __m128 v_corr = _mm_mul_ps(v_corr_half, v_sign_128_s);
    __m128 v_cout_mk = CONJ_128_S(_mm_sub_ps(v_base, v_corr));
    _mm_storeu_ps(p_cout_k + 2 * k, _mm_add_ps(v_base, v_corr));
    _mm_storeu_ps(p_cout_mk + mk_base, REV_CPLX_128_S(v_cout_mk));
}

static inline FFTZ_VOID separate_from_hc_pair_bases_avx128_fp64(
    FFTZ_DOUBLE *p_cout_k, FFTZ_DOUBLE *p_cout_mk, const FFTZ_DOUBLE *p_in_k,
    const FFTZ_DOUBLE *p_in_mk, const FFTZ_DOUBLE *p_tw_re,
    const FFTZ_DOUBLE *p_tw_im, FFTZ_INTP n0_by2, FFTZ_INTP k)
{
    const FFTZ_INTP mk_base = 2 * (n0_by2 - k - NUM_SETS_128_D + 1);
    __m128d v_in_k = _mm_loadu_pd(p_in_k + 2 * k);
    __m128d v_in_mk_raw = _mm_loadu_pd(p_in_mk + mk_base);
    __m128d v_in_mk = REV_CPLX_128_D(v_in_mk_raw);
    __m128d v_in_mk_conj = CONJ_128_D(v_in_mk);
    __m128d v_base = _mm_add_pd(v_in_k, v_in_mk_conj);
    __m128d v_delta = _mm_sub_pd(v_in_k, v_in_mk_conj);
    __m128d v_tw_re = _mm_loadu_pd(p_tw_re + 2 * k);
    __m128d v_tw_im = _mm_loadu_pd(p_tw_im + 2 * k);
    __m128d v_corr_half = apply_packed_twiddle_128_d(v_delta, v_tw_re, v_tw_im);
    __m128d v_corr = _mm_mul_pd(v_corr_half, v_sign_128_d);
    __m128d v_cout_mk = CONJ_128_D(_mm_sub_pd(v_base, v_corr));
    _mm_storeu_pd(p_cout_k + 2 * k, _mm_add_pd(v_base, v_corr));
    _mm_storeu_pd(p_cout_mk + mk_base, REV_CPLX_128_D(v_cout_mk));
}

/**
 * @brief Recombines conjugate pairs into Hermitian output bins.
 *
 * Processes NUM_SETS_128_S fp32 pairs starting at k and mirrored pairs ending
 * at n0_by2-k.
 */
static inline FFTZ_VOID recombine_to_hc_pair_avx128_fp32(
    FFTZ_FLOAT *p_out, const FFTZ_FLOAT *p_cout, const FFTZ_FLOAT *p_tw_re,
    const FFTZ_FLOAT *p_tw_im, FFTZ_INTP n0_by2, FFTZ_INTP k)
{
    recombine_to_hc_pair_bases_avx128_fp32(p_out, p_out, p_cout, p_cout,
                                           p_tw_re, p_tw_im, n0_by2, k);
}

/**
 * @brief Recombines conjugate pairs into Hermitian output bins.
 *
 * Processes NUM_SETS_128_D fp64 pairs starting at k and mirrored pairs ending
 * at n0_by2-k.
 */
static inline FFTZ_VOID recombine_to_hc_pair_avx128_fp64(
    FFTZ_DOUBLE *p_out, const FFTZ_DOUBLE *p_cout, const FFTZ_DOUBLE *p_tw_re,
    const FFTZ_DOUBLE *p_tw_im, FFTZ_INTP n0_by2, FFTZ_INTP k)
{
    recombine_to_hc_pair_bases_avx128_fp64(p_out, p_out, p_cout, p_cout,
                                           p_tw_re, p_tw_im, n0_by2, k);
}

/**
 * @brief Separates Hermitian pairs into packed complex values.
 *
 * Reconstructs the double-length child spectrum for NUM_SETS_128_S
 * fp32 pairs starting at k and mirrored pairs ending at n0_by2-k.
 */
static inline FFTZ_VOID separate_from_hc_pair_avx128_fp32(
    FFTZ_FLOAT *p_cout, const FFTZ_FLOAT *p_in, const FFTZ_FLOAT *p_tw_re,
    const FFTZ_FLOAT *p_tw_im, FFTZ_INTP n0_by2, FFTZ_INTP k)
{
    separate_from_hc_pair_bases_avx128_fp32(p_cout, p_cout, p_in, p_in, p_tw_re,
                                            p_tw_im, n0_by2, k);
}

/**
 * @brief Separates Hermitian pairs into packed complex values.
 *
 * Reconstructs the double-length child spectrum for NUM_SETS_128_D
 * fp64 pairs starting at k and mirrored pairs ending at n0_by2-k.
 */
static inline FFTZ_VOID separate_from_hc_pair_avx128_fp64(
    FFTZ_DOUBLE *p_cout, const FFTZ_DOUBLE *p_in, const FFTZ_DOUBLE *p_tw_re,
    const FFTZ_DOUBLE *p_tw_im, FFTZ_INTP n0_by2, FFTZ_INTP k)
{
    separate_from_hc_pair_bases_avx128_fp64(p_cout, p_cout, p_in, p_in, p_tw_re,
                                            p_tw_im, n0_by2, k);
}

static inline FFTZ_VOID recombine_to_hc_row_pair_fp32_avx128(
    FFTZ_FLOAT *p_out_row, FFTZ_FLOAT *p_out_mirror,
    const FFTZ_FLOAT *p_cout_row, const FFTZ_FLOAT *p_cout_mirror,
    const FFTZ_FLOAT *p_tw_re, const FFTZ_FLOAT *p_tw_im, FFTZ_INTP n0_by2,
    FFTZ_INTP num_pairs)
{
    recombine_to_hc_dc_nyq_pair_fp32(p_out_row, p_out_mirror, p_cout_row,
                                     p_cout_mirror, n0_by2, 1);
    FFTZ_INTP remaining = num_pairs % NUM_SETS_128_S;
    FFTZ_INTP count;
    for (count = 1; count <= num_pairs - remaining; count += NUM_SETS_128_S)
    {
        recombine_to_hc_pair_bases_avx128_fp32(p_out_row, p_out_mirror,
                                               p_cout_row, p_cout_mirror,
                                               p_tw_re, p_tw_im, n0_by2, count);
    }
    if (remaining & NUM_SETS_C_S)
    {
        recombine_to_hc_pair_bases_fp32(p_out_row, p_out_mirror, p_cout_row,
                                        p_cout_mirror, p_tw_re, p_tw_im, n0_by2,
                                        count);
    }
    for (count = 1; count <= num_pairs - remaining; count += NUM_SETS_128_S)
    {
        recombine_to_hc_pair_bases_avx128_fp32(p_out_mirror, p_out_row,
                                               p_cout_mirror, p_cout_row,
                                               p_tw_re, p_tw_im, n0_by2, count);
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

static inline FFTZ_VOID recombine_to_hc_row_pair_fp64_avx128(
    FFTZ_DOUBLE *p_out_row, FFTZ_DOUBLE *p_out_mirror,
    const FFTZ_DOUBLE *p_cout_row, const FFTZ_DOUBLE *p_cout_mirror,
    const FFTZ_DOUBLE *p_tw_re, const FFTZ_DOUBLE *p_tw_im, FFTZ_INTP n0_by2,
    FFTZ_INTP num_pairs)
{
    recombine_to_hc_dc_nyq_pair_fp64(p_out_row, p_out_mirror, p_cout_row,
                                     p_cout_mirror, n0_by2, 1);
    FFTZ_INTP remaining = num_pairs % NUM_SETS_128_D;
    FFTZ_INTP count;
    for (count = 1; count <= num_pairs - remaining; count += NUM_SETS_128_D)
    {
        recombine_to_hc_pair_bases_avx128_fp64(p_out_row, p_out_mirror,
                                               p_cout_row, p_cout_mirror,
                                               p_tw_re, p_tw_im, n0_by2, count);
    }
    for (; count <= num_pairs; count++)
    {
        recombine_to_hc_pair_bases_fp64(p_out_row, p_out_mirror, p_cout_row,
                                        p_cout_mirror, p_tw_re, p_tw_im, n0_by2,
                                        count);
    }
    for (count = 1; count <= num_pairs - remaining; count += NUM_SETS_128_D)
    {
        recombine_to_hc_pair_bases_avx128_fp64(p_out_mirror, p_out_row,
                                               p_cout_mirror, p_cout_row,
                                               p_tw_re, p_tw_im, n0_by2, count);
    }
    for (; count <= num_pairs; count++)
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

static inline FFTZ_VOID separate_from_hc_row_pair_fp32_avx128(
    FFTZ_FLOAT *p_cout_row, FFTZ_FLOAT *p_cout_mirror,
    const FFTZ_FLOAT *p_in_row, const FFTZ_FLOAT *p_in_mirror,
    const FFTZ_FLOAT *p_tw_re, const FFTZ_FLOAT *p_tw_im, FFTZ_INTP n0_by2,
    FFTZ_INTP num_pairs)
{
    separate_from_hc_cout0_fp32(p_cout_row, p_in_row, p_in_mirror, n0_by2);
    separate_from_hc_cout0_fp32(p_cout_mirror, p_in_mirror, p_in_row, n0_by2);
    FFTZ_INTP remaining = num_pairs % NUM_SETS_128_S;
    FFTZ_INTP count;
    for (count = 1; count <= num_pairs - remaining; count += NUM_SETS_128_S)
    {
        separate_from_hc_pair_bases_avx128_fp32(p_cout_row, p_cout_mirror,
                                                p_in_row, p_in_mirror, p_tw_re,
                                                p_tw_im, n0_by2, count);
    }
    if (remaining & NUM_SETS_C_S)
    {
        separate_from_hc_pair_bases_fp32(p_cout_row, p_cout_mirror, p_in_row,
                                         p_in_mirror, p_tw_re, p_tw_im, n0_by2,
                                         count);
    }
    for (count = 1; count <= num_pairs - remaining; count += NUM_SETS_128_S)
    {
        separate_from_hc_pair_bases_avx128_fp32(p_cout_mirror, p_cout_row,
                                                p_in_mirror, p_in_row, p_tw_re,
                                                p_tw_im, n0_by2, count);
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

static inline FFTZ_VOID separate_from_hc_row_pair_fp64_avx128(
    FFTZ_DOUBLE *p_cout_row, FFTZ_DOUBLE *p_cout_mirror,
    const FFTZ_DOUBLE *p_in_row, const FFTZ_DOUBLE *p_in_mirror,
    const FFTZ_DOUBLE *p_tw_re, const FFTZ_DOUBLE *p_tw_im, FFTZ_INTP n0_by2,
    FFTZ_INTP num_pairs)
{
    separate_from_hc_cout0_fp64(p_cout_row, p_in_row, p_in_mirror, n0_by2);
    separate_from_hc_cout0_fp64(p_cout_mirror, p_in_mirror, p_in_row, n0_by2);
    FFTZ_INTP remaining = num_pairs % NUM_SETS_128_D;
    FFTZ_INTP count;
    for (count = 1; count <= num_pairs - remaining; count += NUM_SETS_128_D)
    {
        separate_from_hc_pair_bases_avx128_fp64(p_cout_row, p_cout_mirror,
                                                p_in_row, p_in_mirror, p_tw_re,
                                                p_tw_im, n0_by2, count);
    }
    for (; count <= num_pairs; count++)
    {
        separate_from_hc_pair_bases_fp64(p_cout_row, p_cout_mirror, p_in_row,
                                         p_in_mirror, p_tw_re, p_tw_im, n0_by2,
                                         count);
    }
    for (count = 1; count <= num_pairs - remaining; count += NUM_SETS_128_D)
    {
        separate_from_hc_pair_bases_avx128_fp64(p_cout_mirror, p_cout_row,
                                                p_in_mirror, p_in_row, p_tw_re,
                                                p_tw_im, n0_by2, count);
    }
    for (; count <= num_pairs; count++)
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

#endif // AOCLFFTZ_PACKED_RDFT_AVX128_COMMON_H
