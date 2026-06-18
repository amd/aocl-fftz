// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file generic_kernels_common.h
 *
 *  @brief Macros required for writing the ISA generic kernels
 *
 *  @author Ashwin K. Godbole
 */

#ifndef GENERIC_KERNELS_COMMON_H
#define GENERIC_KERNELS_COMMON_H

#ifdef KERNEL_USE_AVX128
    #define NUM_SETS_S NUM_SETS_128_S
    #define NUM_SETS_D NUM_SETS_128_D

    #define VREGTYPE_S __m128
    #define VREGTYPE_D __m128d

    // Cost: {fma: 0, mul: 0, add: 0, move: 2, perm: 0, other: 0}
    #define GATHER_S GATHER2_128_S
    // Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 0, other: 0}
    #define GATHER_D LD_128_OFFSET_D

    // Cost: {fma: 0, mul: 2, add: 1, move: 3, perm: 5, other: 0}
    #define TW_GATHER_S TW_GATHER_128_S
    // Cost: {fma: 0, mul: 2, add: 1, move: 3, perm: 1, other: 0}
    #define TW_GATHER_D TW_GATHER_128_D

    // Cost: {fma: 0, mul: 2, add: 1, move: 3, perm: 6, other: 0}
    #define ITW_GATHER_S ITW_GATHER_128_S
    // Cost: {fma: 0, mul: 2, add: 1, move: 3, perm: 2, other: 0}
    #define ITW_GATHER_D ITW_GATHER_128_D

    // Cost: {fma: 0, mul: 0, add: 0, move: 2, perm: 0, other: 0}
    #define SCATTER_S SCATTER2_128_S
    // Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 0, other: 0}
    #define SCATTER_D ST_128_OFFSET_D

    // Cost: {fma: 0, mul: 0, add: 1, move: 0, perm: 0, other: 0}
    #define ADD_S _mm_add_ps
    // Cost: {fma: 0, mul: 0, add: 1, move: 0, perm: 0, other: 0}
    #define ADD_D _mm_add_pd

    // Cost: {fma: 0, mul: 0, add: 1, move: 0, perm: 0, other: 0}
    #define SUB_S _mm_sub_ps
    // Cost: {fma: 0, mul: 0, add: 1, move: 0, perm: 0, other: 0}
    #define SUB_D _mm_sub_pd

    // Cost: {fma: 0, mul: 1, add: 0, move: 0, perm: 0, other: 0}
    #define MUL_S _mm_mul_ps
    // Cost: {fma: 0, mul: 1, add: 0, move: 0, perm: 0, other: 0}
    #define MUL_D _mm_mul_pd

    // Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 0, other: 1}
    #define XOR_S _mm_xor_ps
    // Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 0, other: 1}
    #define XOR_D _mm_xor_pd

    // Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 0, other: 0}
    #define BCAST_S _mm_set1_ps
    // Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 0, other: 0}
    #define BCAST_D _mm_set1_pd

    // Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 0, other: 1}
    #define NEG_S(x, f) _mm_xor_ps(x, _neg_128_f[f].s)
    // Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 0, other: 1}
    #define NEG_D(x, f) _mm_xor_pd(x, _neg_128_d[f].d)

    // No cast needed for AVX128 (it's the lowest level)

    // Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 1, other: 0}
    #define SWAP_RI_S SWAP_RI_128_S
    // Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 1, other: 0}
    #define SWAP_RI_D SWAP_RI_128_D

    // Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 0, other: 1}
    #define CONJ_S CONJ_128_S
    // Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 0, other: 1}
    #define CONJ_D CONJ_128_D

    // Cost: {fma: 0, mul: 0, add: 1, move: 0, perm: 2, other: 0}
    #define SUBADD_SWAPA_S SUBADD_SWAPA_128_S
    // Cost: {fma: 0, mul: 0, add: 1, move: 0, perm: 2, other: 0}
    #define SUBADD_SWAPA_D SUBADD_SWAPA_128_D

    #define GATHER_NOTW_S GATHER_NOTW_128_S
    #define GATHER_NOTW_D GATHER_NOTW_128_D
    #define SCATTER_NOTW_S SCATTER_NOTW_128_S
    #define SCATTER_NOTW_D SCATTER_NOTW_128_D
    #define TW_SCATTER_S TW_SCATTER_128_S
    #define TW_SCATTER_D TW_SCATTER_128_D
    #define ITW_SCATTER_S ITW_SCATTER_128_S
    #define ITW_SCATTER_D ITW_SCATTER_128_D

    #define NEG_ZERO_S(flag) _neg_128_f[flag].s
    #define NEG_ZERO_D(flag) _neg_128_d[flag].d

    // Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 0, other: 0}
    #define LOADU_D _mm_loadu_pd
    // Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 0, other: 0}
    #define BROADCAST_D(ptr) _mm_loadu_pd((ptr))

    // Cost: {fma: 0, mul: 2, add: 1, move: 1, perm: 1, other: 0}
    #define TW_PRELOADED_GATHER_D TW_PRELOADED_GATHER_128_D
    // Cost: {fma: 0, mul: 2, add: 1, move: 1, perm: 2, other: 0}
    #define ITW_PRELOADED_GATHER_D ITW_PRELOADED_GATHER_128_D

    // Variants that take stride value directly (for hoisting optimization)
    #define TW_PRELOADED_GATHER_D_V TW_PRELOADED_GATHER_128_D_V
    #define ITW_PRELOADED_GATHER_D_V ITW_PRELOADED_GATHER_128_D_V

    #define TW_PRELOADED_SCATTER_D TW_PRELOADED_SCATTER_128_D
    #define PRELOADED_GATHER_NOTW_D PRELOADED_GATHER_NOTW_128_D
    #define PRELOADED_SCATTER_NOTW_D PRELOADED_SCATTER_NOTW_128_D
#endif

#ifdef KERNEL_USE_AVX256
    #define NUM_SETS_S NUM_SETS_256_S
    #define NUM_SETS_D NUM_SETS_256_D

    #define VREGTYPE_S __m256
    #define VREGTYPE_D __m256d

    // Cost: {fma: 0, mul: 0, add: 0, move: 4, perm: 1, other: 1}
    #define GATHER_S GATHER4_256_S
    // Cost: {fma: 0, mul: 0, add: 0, move: 2, perm: 0, other: 1}
    #define GATHER_D GATHER2_256_D

    // Cost: {fma: 0, mul: 2, add: 1, move: 5, perm: 6, other: 1}
    #define TW_GATHER_S TW_GATHER_256_S
    // Cost: {fma: 0, mul: 2, add: 1, move: 3, perm: 3, other: 1}
    #define TW_GATHER_D TW_GATHER_256_D

    // Cost: {fma: 0, mul: 2, add: 1, move: 5, perm: 7, other: 1}
    #define ITW_GATHER_S ITW_GATHER_256_S
    // Cost: {fma: 0, mul: 2, add: 1, move: 3, perm: 4, other: 1}
    #define ITW_GATHER_D ITW_GATHER_256_D

    // Cost: {fma: 0, mul: 0, add: 0, move: 4, perm: 0, other: 1}
    #define SCATTER_S SCATTER4_256_S
    // Cost: {fma: 0, mul: 0, add: 0, move: 2, perm: 0, other: 1}
    #define SCATTER_D SCATTER2_256_D

    // Cost: {fma: 0, mul: 0, add: 1, move: 0, perm: 0, other: 0}
    #define ADD_S _mm256_add_ps
    // Cost: {fma: 0, mul: 0, add: 1, move: 0, perm: 0, other: 0}
    #define ADD_D _mm256_add_pd

    // Cost: {fma: 0, mul: 0, add: 1, move: 0, perm: 0, other: 0}
    #define SUB_S _mm256_sub_ps
    // Cost: {fma: 0, mul: 0, add: 1, move: 0, perm: 0, other: 0}
    #define SUB_D _mm256_sub_pd

    // Cost: {fma: 0, mul: 1, add: 0, move: 0, perm: 0, other: 0}
    #define MUL_S _mm256_mul_ps
    // Cost: {fma: 0, mul: 1, add: 0, move: 0, perm: 0, other: 0}
    #define MUL_D _mm256_mul_pd

    // Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 0, other: 1}
    #define XOR_S _mm256_xor_ps
    // Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 0, other: 1}
    #define XOR_D _mm256_xor_pd

    // Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 0, other: 0}
    #define BCAST_S _mm256_set1_ps
    // Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 0, other: 0}
    #define BCAST_D _mm256_set1_pd

    // Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 0, other: 1}
    #define NEG_S(x, f) _mm256_xor_ps(x, _neg_256_f[f].s)
    // Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 0, other: 1}
    #define NEG_D(x, f) _mm256_xor_pd(x, _neg_256_d[f].d)

    // Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 0, other: 0}
    #define CAST_256_TO_128_S _mm256_castps256_ps128
    // Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 0, other: 0}
    #define CAST_256_TO_128_D _mm256_castpd256_pd128

    // Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 1, other: 0}
    #define SWAP_RI_S SWAP_RI_256_S
    // Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 1, other: 0}
    #define SWAP_RI_D SWAP_RI_256_D

    // Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 0, other: 1}
    #define CONJ_S CONJ_256_S
    // Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 0, other: 1}
    #define CONJ_D CONJ_256_D

    // Cost: {fma: 0, mul: 0, add: 1, move: 0, perm: 2, other: 0}
    #define SUBADD_SWAPA_S SUBADD_SWAPA_256_S
    // Cost: {fma: 0, mul: 0, add: 1, move: 0, perm: 2, other: 0}
    #define SUBADD_SWAPA_D SUBADD_SWAPA_256_D

    #define GATHER_NOTW_S GATHER_NOTW_256_S
    #define GATHER_NOTW_D GATHER_NOTW_256_D
    #define SCATTER_NOTW_S SCATTER_NOTW_256_S
    #define SCATTER_NOTW_D SCATTER_NOTW_256_D
    #define TW_SCATTER_S TW_SCATTER_256_S
    #define TW_SCATTER_D TW_SCATTER_256_D
    #define ITW_SCATTER_S ITW_SCATTER_256_S
    #define ITW_SCATTER_D ITW_SCATTER_256_D

    #define NEG_ZERO_S(flag) _neg_256_f[flag].s
    #define NEG_ZERO_D(flag) _neg_256_d[flag].d

    // Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 0, other: 0}
    #define LOADU_D _mm256_loadu_pd
    // Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 0, other: 1}
    #define BROADCAST_D(ptr) _mm256_broadcast_pd((__m128d *)(ptr))

    // Cost: {fma: 0, mul: 2, add: 1, move: 2, perm: 3, other: 1}
    #define TW_PRELOADED_GATHER_D TW_PRELOADED_GATHER_256_D
    // Cost: {fma: 0, mul: 2, add: 1, move: 2, perm: 4, other: 1}
    #define ITW_PRELOADED_GATHER_D ITW_PRELOADED_GATHER_256_D

    // Variants that take stride value directly (for hoisting optimization)
    #define TW_PRELOADED_GATHER_D_V TW_PRELOADED_GATHER_256_D_V
    #define ITW_PRELOADED_GATHER_D_V ITW_PRELOADED_GATHER_256_D_V

    #define TW_PRELOADED_SCATTER_D TW_PRELOADED_SCATTER_256_D
    #define PRELOADED_GATHER_NOTW_D PRELOADED_GATHER_NOTW_256_D
    #define PRELOADED_SCATTER_NOTW_D PRELOADED_SCATTER_NOTW_256_D
#endif

#ifdef KERNEL_USE_AVX512
    #define NUM_SETS_S NUM_SETS_512_S
    #define NUM_SETS_D NUM_SETS_512_D

    #define VREGTYPE_S __m512
    #define VREGTYPE_D __m512d

    // Cost: {fma: 0, mul: 0, add: 0, move: 8, perm: 3, other: 3}
    #define GATHER_S GATHER8_512_S
    // Cost: {fma: 0, mul: 0, add: 0, move: 4, perm: 0, other: 3}
    #define GATHER_D GATHER4_512_D

    // Cost: {fma: 1, mul: 2, add: 0, move: 10, perm: 8, other: 3}
    #define TW_GATHER_S TW_GATHER_512_S
    // Cost: {fma: 1, mul: 2, add: 0, move: 6, perm: 3, other: 3}
    #define TW_GATHER_D TW_GATHER_512_D

    // Cost: {fma: 1, mul: 2, add: 0, move: 10, perm: 9, other: 3}
    #define ITW_GATHER_S ITW_GATHER_512_S
    // Cost: {fma: 1, mul: 2, add: 0, move: 6, perm: 4, other: 3}
    #define ITW_GATHER_D ITW_GATHER_512_D

    // Cost: {fma: 0, mul: 0, add: 0, move: 8, perm: 0, other: 3}
    #define SCATTER_S SCATTER8_512_S
    // Cost: {fma: 0, mul: 0, add: 0, move: 4, perm: 0, other: 3}
    #define SCATTER_D SCATTER4_512_D

    // Cost: {fma: 0, mul: 0, add: 1, move: 0, perm: 0, other: 0}
    #define ADD_S _mm512_add_ps
    // Cost: {fma: 0, mul: 0, add: 1, move: 0, perm: 0, other: 0}
    #define ADD_D _mm512_add_pd

    // Cost: {fma: 0, mul: 0, add: 1, move: 0, perm: 0, other: 0}
    #define SUB_S _mm512_sub_ps
    // Cost: {fma: 0, mul: 0, add: 1, move: 0, perm: 0, other: 0}
    #define SUB_D _mm512_sub_pd

    // Cost: {fma: 0, mul: 1, add: 0, move: 0, perm: 0, other: 0}
    #define MUL_S _mm512_mul_ps
    // Cost: {fma: 0, mul: 1, add: 0, move: 0, perm: 0, other: 0}
    #define MUL_D _mm512_mul_pd

    // Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 0, other: 1}
    #define XOR_S _mm512_xor_ps
    // Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 0, other: 1}
    #define XOR_D _mm512_xor_pd

    // Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 0, other: 0}
    #define BCAST_S _mm512_set1_ps
    // Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 0, other: 0}
    #define BCAST_D _mm512_set1_pd

    // Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 0, other: 1}
    #define NEG_S(x, f) _mm512_xor_ps(x, _neg_512_f[f].s)
    // Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 0, other: 1}
    #define NEG_D(x, f) _mm512_xor_pd(x, _neg_512_d[f].d)

    // Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 0, other: 0}
    #define CAST_512_TO_256_S _mm512_castps512_ps256
    // Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 0, other: 0}
    #define CAST_512_TO_256_D _mm512_castpd512_pd256

    // Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 0, other: 0}
    #define CAST_512_TO_128_S _mm512_castps512_ps128
    // Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 0, other: 0}
    #define CAST_512_TO_128_D _mm512_castpd512_pd128

    // Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 1, other: 0}
    #define SWAP_RI_S SWAP_RI_512_S
    // Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 1, other: 0}
    #define SWAP_RI_D SWAP_RI_512_D

    // Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 0, other: 1}
    #define CONJ_S CONJ_512_S
    // Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 0, other: 1}
    #define CONJ_D CONJ_512_D

    // Cost: {fma: 1, mul: 0, add: 0, move: 0, perm: 2, other: 0}
    #define SUBADD_SWAPA_S SUBADD_SWAPA_512_S
    // Cost: {fma: 1, mul: 0, add: 0, move: 0, perm: 2, other: 0}
    #define SUBADD_SWAPA_D SUBADD_SWAPA_512_D

    #define GATHER_NOTW_S GATHER_NOTW_512_S
    #define GATHER_NOTW_D GATHER_NOTW_512_D
    #define SCATTER_NOTW_S SCATTER_NOTW_512_S
    #define SCATTER_NOTW_D SCATTER_NOTW_512_D
    #define TW_SCATTER_S TW_SCATTER_512_S
    #define TW_SCATTER_D TW_SCATTER_512_D
    #define ITW_SCATTER_S ITW_SCATTER_512_S
    #define ITW_SCATTER_D ITW_SCATTER_512_D

    #define NEG_ZERO_S(flag) _neg_512_f[flag].s
    #define NEG_ZERO_D(flag) _neg_512_d[flag].d

    // Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 0, other: 0}
    #define LOADU_D _mm512_loadu_pd
    // Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 0, other: 0}
    #define BROADCAST_D(ptr) _mm512_broadcast_f64x2(_mm_loadu_pd((ptr)))

    // Cost: {fma: 1, mul: 2, add: 0, move: 5, perm: 3, other: 0}
    #define TW_PRELOADED_GATHER_D TW_PRELOADED_GATHER_512_D
    // Cost: {fma: 1, mul: 2, add: 0, move: 5, perm: 4, other: 0}
    #define ITW_PRELOADED_GATHER_D ITW_PRELOADED_GATHER_512_D

    // Variants that take stride value directly (for hoisting optimization)
    #define TW_PRELOADED_GATHER_D_V TW_PRELOADED_GATHER_512_D_V
    #define ITW_PRELOADED_GATHER_D_V ITW_PRELOADED_GATHER_512_D_V

    #define TW_PRELOADED_SCATTER_D TW_PRELOADED_SCATTER_512_D
    #define PRELOADED_GATHER_NOTW_D PRELOADED_GATHER_NOTW_512_D
    #define PRELOADED_SCATTER_NOTW_D PRELOADED_SCATTER_NOTW_512_D
#endif

// Identity macros for 2nd-half input/output points
// These will be overridden by R2C/C2R twiddle kernel variants
#ifndef OUT_H2_S
#define OUT_H2_S(val) (val)
#endif
#ifndef IN_H2_S
#define IN_H2_S(val) (val)
#endif
#ifndef OUT_H2_D
#define OUT_H2_D(val) (val)
#endif
#ifndef IN_H2_D
#define IN_H2_D(val) (val)
#endif
#ifndef OUT_H2_256_S
#define OUT_H2_256_S(val) (val)
#endif
#ifndef IN_H2_256_S
#define IN_H2_256_S(val) (val)
#endif
#ifndef OUT_H2_128_S
#define OUT_H2_128_S(val) (val)
#endif
#ifndef IN_H2_128_S
#define IN_H2_128_S(val) (val)
#endif
#ifndef OUT_H2_LOW_128_S
#define OUT_H2_LOW_128_S(val) (val)
#endif
#ifndef IN_H2_LOW_128_S
#define IN_H2_LOW_128_S(val) (val)
#endif
#ifndef OUT_H2_256_D
#define OUT_H2_256_D(val) (val)
#endif
#ifndef IN_H2_256_D
#define IN_H2_256_D(val) (val)
#endif
#ifndef OUT_H2_128_D
#define OUT_H2_128_D(val) (val)
#endif
#ifndef IN_H2_128_D
#define IN_H2_128_D(val) (val)
#endif

#endif // GENERIC_KERNELS_COMMON_H
