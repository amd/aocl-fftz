/**
 * Copyright (C) 2025, Advanced Micro Devices. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/** @file generic_kernels_common.h
 *
 *  @brief Macros required for writing the ISA generic kernels
 *
 *  @author Ashwin K. Godbole
 */

#ifndef GENERIC_KERNELS_COMMON_H
#define GENERIC_KERNELS_COMMON_H

/**
 * @brief Generic macro definitions for different SIMD variants
 * 
 * This file maps generic operation names to specific SIMD intrinsics
 * based on compile-time flags (KERNEL_USE_AVX128/256/512).
 * 
 * Each macro includes a cost comment showing the operation complexity
 * in terms of {fma, mul, add, move, perm, other} cycles.
 */

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
    #define NEG_S(x) _mm_xor_ps((x), _mm_set1_ps(-0.0f))
    // Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 0, other: 1}
    #define NEG_D(x) _mm_xor_pd((x), _mm_set1_pd(-0.0))

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

    #define NEG_ZERO_S _neg_zero_128_f.s
    #define NEG_ZERO_D _neg_zero_128_d.d

    #define SET_ZERO_S _mm_setzero_ps
    #define SET_ZERO_D _mm_setzero_pd
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
    #define NEG_S(x) _mm256_xor_ps((x), _mm256_set1_ps(-0.0f))
    // Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 0, other: 1}
    #define NEG_D(x) _mm256_xor_pd((x), _mm256_set1_pd(-0.0))

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

    #define NEG_ZERO_S _neg_zero_256_f.s
    #define NEG_ZERO_D _neg_zero_256_d.d

    #define SET_ZERO_S _mm256_setzero_ps
    #define SET_ZERO_D _mm256_setzero_pd
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
    #define NEG_S(x) _mm512_xor_ps((x), _mm512_set1_ps(-0.0f))
    // Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 0, other: 1}
    #define NEG_D(x) _mm512_xor_pd((x), _mm512_set1_pd(-0.0))

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

    #define NEG_ZERO_S _neg_zero_512_f.s
    #define NEG_ZERO_D _neg_zero_512_d.d

    #define SET_ZERO_S _mm512_setzero_ps
    #define SET_ZERO_D _mm512_setzero_pd
#endif

#endif // GENERIC_KERNELS_COMMON_H
