// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fused_twiddle_transpose_cmul.h
 *
 *  @brief FMA complex-multiply macros for the fused twiddle+transpose kernels.
 *
 *  CMUL(a, b)      computes a * b.
 *  CMUL_CONJ(a, b) computes a * conj(b).
 *
 *  The fused kernels use CMUL on the forward path (data * twiddle) and
 *  CMUL_CONJ on the inverse path (data * conj(twiddle)), so the inverse path
 *  needs no explicit twiddle conjugation.
 *
 *  Both are built on the broadcast-real/imag, swap-and-cross structure but fuse
 *  the final multiply-add with FMA via FMADDSUB / FMSUBADD (1 mul + 1 FMA per
 *  complex instead of 2 mul + 1 addsub):
 *
 *    CMUL      : a * b       = FMADDSUB(re(a), b,       im(a) * swap(b))
 *    CMUL_CONJ : a * conj(b) = FMSUBADD(im(a), swap(b), re(a) * b)
 *
 *  Each macro requires the matching width's BROADCAST_RE/IM_*, SWAP_RI_*,
 *  FMADDSUB_* / FMSUBADD_* macros (from simd_common.h / simd_common_avx512.h)
 *  to be in scope at the point of use. The _FP32 / _FP64 suffix selects the
 *  precision; the numeric suffix selects the SIMD lane width.
 *
 *  @author Ashwin K. Godbole
 */

#ifndef FUSED_TWIDDLE_TRANSPOSE_CMUL_H
#define FUSED_TWIDDLE_TRANSPOSE_CMUL_H

/* 128-bit */
#define CMUL_128_FP32(a, b)                                                    \
    FMADDSUB_128_S(BROADCAST_RE_128_S(a), (b),                                 \
                   _mm_mul_ps(BROADCAST_IM_128_S(a), SWAP_RI_128_S(b)))
#define CMUL_CONJ_128_FP32(a, b)                                               \
    FMSUBADD_128_S(BROADCAST_IM_128_S(a), SWAP_RI_128_S(b),                    \
                   _mm_mul_ps(BROADCAST_RE_128_S(a), (b)))
#define CMUL_128_FP64(a, b)                                                    \
    FMADDSUB_128_D(BROADCAST_RE_128_D(a), (b),                                 \
                   _mm_mul_pd(BROADCAST_IM_128_D(a), SWAP_RI_128_D(b)))
#define CMUL_CONJ_128_FP64(a, b)                                               \
    FMSUBADD_128_D(BROADCAST_IM_128_D(a), SWAP_RI_128_D(b),                    \
                   _mm_mul_pd(BROADCAST_RE_128_D(a), (b)))

/* 256-bit */
#define CMUL_256_FP32(a, b)                                                    \
    FMADDSUB_256_S(BROADCAST_RE_256_S(a), (b),                                 \
                   _mm256_mul_ps(BROADCAST_IM_256_S(a), SWAP_RI_256_S(b)))
#define CMUL_CONJ_256_FP32(a, b)                                               \
    FMSUBADD_256_S(BROADCAST_IM_256_S(a), SWAP_RI_256_S(b),                    \
                   _mm256_mul_ps(BROADCAST_RE_256_S(a), (b)))
#define CMUL_256_FP64(a, b)                                                    \
    FMADDSUB_256_D(BROADCAST_RE_256_D(a), (b),                                 \
                   _mm256_mul_pd(BROADCAST_IM_256_D(a), SWAP_RI_256_D(b)))
#define CMUL_CONJ_256_FP64(a, b)                                               \
    FMSUBADD_256_D(BROADCAST_IM_256_D(a), SWAP_RI_256_D(b),                    \
                   _mm256_mul_pd(BROADCAST_RE_256_D(a), (b)))

/* 512-bit */
#define CMUL_512_FP32(a, b)                                                    \
    FMADDSUB_512_S(BROADCAST_RE_512_S(a), (b),                                 \
                   _mm512_mul_ps(BROADCAST_IM_512_S(a), SWAP_RI_512_S(b)))
#define CMUL_CONJ_512_FP32(a, b)                                               \
    FMSUBADD_512_S(BROADCAST_IM_512_S(a), SWAP_RI_512_S(b),                    \
                   _mm512_mul_ps(BROADCAST_RE_512_S(a), (b)))
#define CMUL_512_FP64(a, b)                                                    \
    FMADDSUB_512_D(BROADCAST_RE_512_D(a), (b),                                 \
                   _mm512_mul_pd(BROADCAST_IM_512_D(a), SWAP_RI_512_D(b)))
#define CMUL_CONJ_512_FP64(a, b)                                               \
    FMSUBADD_512_D(BROADCAST_IM_512_D(a), SWAP_RI_512_D(b),                    \
                   _mm512_mul_pd(BROADCAST_RE_512_D(a), (b)))

#endif // FUSED_TWIDDLE_TRANSPOSE_CMUL_H

