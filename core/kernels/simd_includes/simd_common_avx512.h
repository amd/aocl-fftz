// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file simd_common_avx512.h
 *
 *  @brief List of common operations and macros used for simd (AVX512) kernels.
 *
 *  This file contains the category-wise macros and common functions used
 *  in the AVX512 kernel variants
 *
 *  @author Murugan Vairavel
 *  @author Srirammaswamy S
 *  @author Jeya R
 *  @author Ashwin K. Godbole
 */

#ifndef AOCLFFTZ_SIMD_COMMON_AVX512_H
#define AOCLFFTZ_SIMD_COMMON_AVX512_H

#include "core/kernels/simd_includes/simd_common.h"

/**
 * @brief interchanges the real and imaginary values in a 512 bit register
 * for single precision floating point using the control value in the last
 * integer argument(b 10 11 00 01)
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 1, other: 0}
#define SWAP_RI_512_S(val) _mm512_permute_ps(val, 177)

/**
 * @brief interchanges the real and imaginary values in a 512 bit register
 * for double precision floating point using the control value in the last
 * integer argument(b 01 01 01 01)
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 1, other: 0}
#define SWAP_RI_512_D(val) _mm512_permute_pd(val, 85)


// FMADD: (a * b) + c
#define FMADD_512_S(a, b, c) _mm512_fmadd_ps((a), (b), (c))
#define FMADD_512_D(a, b, c) _mm512_fmadd_pd((a), (b), (c))

// FMSUB: (a * b) - c
#define FMSUB_512_S(a, b, c) _mm512_fmsub_ps((a), (b), (c))
#define FMSUB_512_D(a, b, c) _mm512_fmsub_pd((a), (b), (c))

// FNMADD: -(a * b) + c
#define FNMADD_512_S(a, b, c) _mm512_fnmadd_ps((a), (b), (c))
#define FNMADD_512_D(a, b, c) _mm512_fnmadd_pd((a), (b), (c))

// FNMSUB: -(a * b) - c
#define FNMSUB_512_S(a, b, c) _mm512_fnmsub_ps((a), (b), (c))
#define FNMSUB_512_D(a, b, c) _mm512_fnmsub_pd((a), (b), (c))

// FMADDSUB: (a * b) +(odd)/-(even) c
#define FMADDSUB_512_S(a, b, c) _mm512_fmaddsub_ps((a), (b), (c))
#define FMADDSUB_512_D(a, b, c) _mm512_fmaddsub_pd((a), (b), (c))

// FMSUBADD: (a * b) -(odd)/+(even) c
#define FMSUBADD_512_S(a, b, c) _mm512_fmsubadd_ps((a), (b), (c))
#define FMSUBADD_512_D(a, b, c) _mm512_fmsubadd_pd((a), (b), (c))

/**
 * @brief Broadcasts real parts of complex numbers in a 512-bit register.
 * Shuffle pattern 0xA0 (10100000b) selects elements [0,0,2,2].
 * Operation : 1 PERM(shuffle)
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 1, other: 0}
#define BROADCAST_RE_512_S(val) _mm512_shuffle_ps(val, val, 0xA0) // 10100000b: [0,0,2,2]

/**
 * @brief Broadcasts imaginary parts of complex numbers in a 512-bit register.
 * Shuffle pattern 0xF5 (11110101b) selects elements [1,1,3,3].
 * Operation : 1 PERM(shuffle)
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 1, other: 0}
#define BROADCAST_IM_512_S(val) _mm512_shuffle_ps(val, val, 0xF5) // 11110101b: [1,1,3,3]

/**
 * @brief Broadcasts real parts of complex numbers in a 512-bit register
 * for double precision floating point.
 * Shuffle pattern 0x00 (00000000b) selects the low element of each 128-bit lane:
 * [0,0,2,2,4,4,6,6].
 * Operation : 1 PERM(shuffle)
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 1, other: 0}
#define BROADCAST_RE_512_D(val) _mm512_shuffle_pd(val, val, 0x00) // 00000000b: [0,0,2,2,4,4,6,6]

/**
 * @brief Broadcasts imaginary parts of complex numbers in a 512-bit register
 * for double precision floating point.
 * Shuffle pattern 0xFF (11111111b) selects the high element of each 128-bit lane:
 * [1,1,3,3,5,5,7,7].
 * Operation : 1 PERM(shuffle)
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 1, other: 0}
#define BROADCAST_IM_512_D(val) _mm512_shuffle_pd(val, val, 0xFF) // 11111111b: [1,1,3,3,5,5,7,7]

/**
 * @brief implies the number of sets that can be processed in parallel.
 * Computed using Register width /(2* sizeof(floating point)
 */
#define NUM_SETS_512_S 8
#define NUM_SETS_512_D 4

union data_union_512
{
    unsigned u[16];
    __m512 s;
    __m512d d;
};

static const union data_union_512
            _conj_512_f = {{ 0x00000000, 0x80000000, 0x00000000, 0x80000000,
                             0x00000000, 0x80000000, 0x00000000, 0x80000000,
                             0x00000000, 0x80000000, 0x00000000, 0x80000000,
                             0x00000000, 0x80000000, 0x00000000, 0x80000000 }};
static const union data_union_512
            _conj_512_d = {{ 0x00000000, 0x00000000, 0x00000000, 0x80000000,
                             0x00000000, 0x00000000, 0x00000000, 0x80000000,
                             0x00000000, 0x00000000, 0x00000000, 0x80000000,
                             0x00000000, 0x00000000, 0x00000000, 0x80000000 }};

static const union data_union_512
    _neg_512_f[2] = {
        {.u = { 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 }},
        {.u = { 0x80000000, 0x80000000, 0x80000000, 0x80000000,
                0x80000000, 0x80000000, 0x80000000, 0x80000000,
                0x80000000, 0x80000000, 0x80000000, 0x80000000,
                0x80000000, 0x80000000, 0x80000000, 0x80000000 }}
    };
static const union data_union_512
    _neg_512_d[2] = {
        {.u = { 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 }},
        {.u = { 0x00000000, 0x80000000, 0x00000000, 0x80000000,
                0x00000000, 0x80000000, 0x00000000, 0x80000000,
                0x00000000, 0x80000000, 0x00000000, 0x80000000,
                0x00000000, 0x80000000, 0x00000000, 0x80000000 }}
    };

/**
 * @brief take conjugate of the complex number A+Bi => A-Bi
 * Operation : 1 OTHERS(xor)
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 0, other: 1}
#define CONJ_512_S(x) _mm512_xor_ps(_conj_512_f.s, x)
// Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 0, other: 1}
#define CONJ_512_D(x) _mm512_xor_pd(_conj_512_d.d, x)

/**
 * @brief alternatively performs addition & subtraction in a 512 bit register
 * for single precision floating point.
 * Operation : 2 PERM, 1 FMA
 */
// Cost: {fma: 1, mul: 0, add: 0, move: 0, perm: 2, other: 0}
#define SUBADD_SWAPA_512_S(a, b, c)                                            \
{                                                                              \
    a = SWAP_RI_512_S(a);                                                      \
    c = SWAP_RI_512_S(_mm512_fmaddsub_ps(_mm512_set1_ps(1.0f), a, b));         \
}

/**
 * @brief alternatively performs addition & subtraction in a 512 bit register
 * for double precision floating point.
 * Operation : 2 PERM, 1 FMA
 */
// Cost: {fma: 1, mul: 0, add: 0, move: 0, perm: 2, other: 0}
#define SUBADD_SWAPA_512_D(a, b, c)                                            \
{                                                                              \
    a = SWAP_RI_512_D(a);                                                      \
    c = SWAP_RI_512_D(_mm512_fmaddsub_pd(_mm512_set1_pd(1.0), a, b));          \
}

/**
 * @brief load eight complex numbers(real,imaginary) of 32 bit single precision
 * floating point number from memory addresses specified by base address
 * and offset, into 512 bit register.
 * Operations : 8 MOV(load), 3 PERM(shuffle), 3 OTHERS(insert).
 * Cast is excluded as it will be a compile time operation.
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 8, perm: 3, other: 3}
#define GATHER8_512_S(base, offset, dest, is_contiguous)                       \
{                                                                              \
    if (is_contiguous)                                                         \
    {                                                                          \
        dest = _mm512_loadu_ps(base);                                          \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        __m128 _low, _high, _tmp;                                              \
        __m256 _256low, _256high;                                              \
        _low = _mm_loadu_ps(base);                                             \
        _tmp = _mm_loadu_ps(base + offset);                                    \
        _low = _mm_shuffle_ps(_low, _tmp, 68);                                 \
        _high = _mm_loadu_ps(base + 2 * offset);                               \
        _tmp = _mm_loadu_ps(base + 3 * offset);                                \
        _high = _mm_shuffle_ps(_high, _tmp, 68);                               \
        _256low = _mm256_insertf128_ps(_mm256_castps128_ps256(_low), _high, 1);\
        _low = _mm_loadu_ps(base + 4 * offset);                                \
        _tmp = _mm_loadu_ps(base + 5 * offset);                                \
        _low = _mm_shuffle_ps(_low, _tmp, 68);                                 \
        _high = _mm_loadu_ps(base + 6 * offset);                               \
        _high = _mm_loadh_pi(_high, (__m64 *)(base + 7 * offset));             \
        _256high = _mm256_insertf128_ps(_mm256_castps128_ps256(_low),          \
            _high, 1);                                                         \
        dest = _mm512_insertf32x8(_mm512_castps256_ps512(_256low),             \
            _256high, 1);                                                      \
    }                                                                          \
}

/**
 * @brief store eight complex numbers(real,imaginary) of 32 bit single precision
 * floating point number from 512 bit register into memory addresses
 * specified by base address and offset.
 * Operations : 8 MOV(store), 3 OTHERS(extract).
 * Cast is excluded as it will be a compile time operation.
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 8, perm: 0, other: 3}
#define SCATTER8_512_S(base, offset, src, is_contiguous)                       \
{                                                                              \
    if (is_contiguous)                                                         \
    {                                                                          \
        _mm512_storeu_ps(base, src);                                           \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        __m256 _256high = _mm512_extractf32x8_ps(src, 1);                      \
        __m256 _256low = _mm512_castps512_ps256(src);                          \
        __m128 _high, _low;                                                    \
        _high = _mm256_extractf128_ps(_256low, 1);                             \
        _low = _mm256_castps256_ps128(_256low);                                \
        _mm_storel_pi((__m64 *)(base), _low);                                  \
        _mm_storeh_pi((__m64 *)(base + offset), _low);                         \
        _mm_storel_pi((__m64 *)(base + 2 * offset), _high);                    \
        _mm_storeh_pi((__m64 *)(base + 3 * offset), _high);                    \
        _high = _mm256_extractf128_ps(_256high, 1);                            \
        _low = _mm256_castps256_ps128(_256high);                               \
        _mm_storel_pi((__m64 *)(base + 4 * offset), _low);                     \
        _mm_storeh_pi((__m64 *)(base + 5 * offset), _low);                     \
        _mm_storel_pi((__m64 *)(base + 6 * offset), _high);                    \
        _mm_storeh_pi((__m64 *)(base + 7 * offset), _high);                    \
    }                                                                          \
}

/**
 * @brief Branchless variant of SCATTER8_512_S to use inside contiguous
 * (out_strides[1] == DATA_STRIDE) branches in kernels, where vector stride
 * `offset` cannot be DATA_STRIDE.
 * Operations : 8 MOV(store), 3 OTHERS(extract)
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 8, perm: 0, other: 3}
#define SCATTER8_512_S_STRIDED(base, offset, src)                              \
{                                                                              \
    __m256 _256high = _mm512_extractf32x8_ps(src, 1);                          \
    __m256 _256low = _mm512_castps512_ps256(src);                              \
    __m128 _high, _low;                                                        \
    _high = _mm256_extractf128_ps(_256low, 1);                                 \
    _low = _mm256_castps256_ps128(_256low);                                    \
    _mm_storel_pi((__m64 *)(base), _low);                                      \
    _mm_storeh_pi((__m64 *)(base + offset), _low);                             \
    _mm_storel_pi((__m64 *)(base + 2 * offset), _high);                        \
    _mm_storeh_pi((__m64 *)(base + 3 * offset), _high);                        \
    _high = _mm256_extractf128_ps(_256high, 1);                                \
    _low = _mm256_castps256_ps128(_256high);                                   \
    _mm_storel_pi((__m64 *)(base + 4 * offset), _low);                         \
    _mm_storeh_pi((__m64 *)(base + 5 * offset), _low);                         \
    _mm_storel_pi((__m64 *)(base + 6 * offset), _high);                        \
    _mm_storeh_pi((__m64 *)(base + 7 * offset), _high);                        \
}

/**
 * @brief load four complex numbers(real,imaginary) of 64 bit double precision
 * floating point number from memory addresses specified by base address
 * and offset into 512 bit register.
 * Operations : 4 MOV(load), 3 OTHERS(insert).
 * Cast is excluded as it will be a compile time operation
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 4, perm: 0, other: 3}
#define GATHER4_512_D(base, offset, dest, is_contiguous)                       \
{                                                                              \
    if (is_contiguous)                                                         \
    {                                                                          \
        dest = _mm512_loadu_pd(base);                                          \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        __m128d _low, _high;                                                   \
        __m256d _256low, _256high;                                             \
        _low = _mm_loadu_pd(base);                                             \
        _high = _mm_loadu_pd(base + offset);                                   \
        _256low =                                                              \
            _mm256_insertf128_pd(_mm256_castpd128_pd256(_low), _high, 1);      \
        _low = _mm_loadu_pd(base + offset * 2);                                \
        _high = _mm_loadu_pd(base + offset * 3);                               \
        _256high =                                                             \
            _mm256_insertf128_pd(_mm256_castpd128_pd256(_low), _high, 1);      \
        dest = _mm512_insertf64x4(_mm512_castpd256_pd512(_256low),             \
               _256high, 1);                                                   \
    }                                                                          \
}

/**
 * @brief store four complex numbers(real,imaginary) of 64 bit double precision
 * floating point number from 512 bit register into memory addresses
 * specified by base address and offset.
 * Operations : 4 MOV(store), 3 OTHERS(extract).
 * Cast is excluded as it will be a compile time operation.
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 4, perm: 0, other: 3}
#define SCATTER4_512_D(base, offset, src, is_contiguous)                       \
{                                                                              \
    if (is_contiguous)                                                         \
    {                                                                          \
        _mm512_storeu_pd(base, src);                                           \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        __m256d _m256high = _mm512_extractf64x4_pd(src, 1);                    \
        __m256d _m256low = _mm512_castpd512_pd256(src);                        \
        __m128d _low, _high;                                                   \
        _high = _mm256_extractf128_pd(_m256low, 1);                            \
        _low = _mm256_castpd256_pd128(_m256low);                               \
        _mm_storeu_pd(base, _low);                                             \
        _mm_storeu_pd(base + offset, _high);                                   \
        _high = _mm256_extractf128_pd(_m256high, 1);                           \
        _low = _mm256_castpd256_pd128(_m256high);                              \
        _mm_storeu_pd(base + 2 * offset, _low);                                \
        _mm_storeu_pd(base + 3 * offset, _high);                               \
    }                                                                          \
}

/**
 * @brief Branchless variant of SCATTER4_512_D to use inside contiguous
 * (out_strides[1] == DATA_STRIDE) branches in kernels, where vector stride
 * `offset` cannot be DATA_STRIDE.
 * Operations : 4 MOV(store), 3 OTHERS(extract)
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 4, perm: 0, other: 3}
#define SCATTER4_512_D_STRIDED(base, offset, src)                              \
{                                                                              \
    __m256d _m256high = _mm512_extractf64x4_pd(src, 1);                        \
    __m256d _m256low = _mm512_castpd512_pd256(src);                            \
    __m128d _low, _high;                                                       \
    _high = _mm256_extractf128_pd(_m256low, 1);                                \
    _low = _mm256_castpd256_pd128(_m256low);                                   \
    _mm_storeu_pd(base, _low);                                                 \
    _mm_storeu_pd(base + offset, _high);                                       \
    _high = _mm256_extractf128_pd(_m256high, 1);                               \
    _low = _mm256_castpd256_pd128(_m256high);                                  \
    _mm_storeu_pd(base + 2 * offset, _low);                                    \
    _mm_storeu_pd(base + 3 * offset, _high);                                   \
}

/**
 * @brief H2 reverse-contiguous gather/scatter helpers.
 *
 * H2 (the conjugate/mirror half of a Hermitian-symmetric r2c/c2r spectrum)
 * is naturally indexed in decreasing memory order, so a unit-stride H2 run
 * has `offset == -DATA_STRIDE` instead of `+DATA_STRIDE`. In that case the
 * points are still one contiguous block in memory, just traversed
 * backwards, so each macro below issues a single wide load/store over that
 * block and then reverses lanes to restore ascending point order. Any other
 * offset falls back to the existing strided gather/scatter macro with
 * `is_contiguous` forced to 0.
 */

/**
 * @brief Load eight H2 32-bit single precision complex points into a
 * 512-bit register, restoring ascending point order.
 *
 * The reverse-contiguous run starts 7 points below `base`. The load is
 * reinterpreted as eight 64-bit (complex) lanes and fully reversed with
 * `_mm512_permutexvar_pd` to turn descending memory order into ascending
 * point order.
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 1, other: 0}
#define GATHER8_H2_512_S(base, offset, dest)                                   \
{                                                                              \
    if ((offset) == -DATA_STRIDE)                                              \
    {                                                                          \
        const __m512i _idx = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);         \
        dest = _mm512_castpd_ps(_mm512_permutexvar_pd(                         \
            _idx, _mm512_castps_pd(_mm512_loadu_ps((base) + 7 * (offset)))));  \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        GATHER8_512_S(base, offset, dest, 0);                                  \
    }                                                                          \
}

/**
 * @brief Store eight H2 32-bit single precision complex points from a
 * 512-bit register, writing them back in descending (reverse-contiguous)
 * order starting at `base + 7 * offset`.
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 1, other: 0}
#define SCATTER8_H2_512_S(base, offset, src)                                   \
{                                                                              \
    if ((offset) == -DATA_STRIDE)                                              \
    {                                                                          \
        const __m512i _idx = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);         \
        _mm512_storeu_ps((base) + 7 * (offset),                                \
                         _mm512_castpd_ps(_mm512_permutexvar_pd(               \
                             _idx, _mm512_castps_pd(src))));                   \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        SCATTER8_512_S(base, offset, src, 0);                                  \
    }                                                                          \
}

/**
 * @brief Load four H2 64-bit double precision complex points into a
 * 512-bit register, restoring ascending point order.
 *
 * The reverse-contiguous run starts 3 points below `base`. Reversing the
 * four 128-bit lanes (`_mm512_shuffle_f64x2`, control 0x1B) turns
 * descending memory order into ascending point order.
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 1, other: 0}
#define GATHER4_H2_512_D(base, offset, dest)                                   \
{                                                                              \
    if ((offset) == -DATA_STRIDE)                                              \
    {                                                                          \
        const __m512d _tmp = _mm512_loadu_pd((base) + 3 * (offset));           \
        dest = _mm512_shuffle_f64x2(_tmp, _tmp, 0x1B);                         \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        GATHER4_512_D(base, offset, dest, 0);                                  \
    }                                                                          \
}

/**
 * @brief Store four H2 64-bit double precision complex points from a
 * 512-bit register, writing them back in descending (reverse-contiguous)
 * order starting at `base + 3 * offset`.
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 1, other: 0}
#define SCATTER4_H2_512_D(base, offset, src)                                   \
{                                                                              \
    if ((offset) == -DATA_STRIDE)                                              \
    {                                                                          \
        _mm512_storeu_pd((base) + 3 * (offset),                                \
                         _mm512_shuffle_f64x2(src, src, 0x1B));                \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        SCATTER4_512_D(base, offset, src, 0);                                  \
    }                                                                          \
}

/**
 * @brief Store four 512-bit double-precision vectors that hold four consecutive
 * complex output points (src0..src3), each laid out as one complex (128-bit)
 * lane per set across NUM_SETS_512_D sets, into contiguous output memory.
 *
 * The four input vectors are point-major (src_k = point k for sets 0..3). When
 * the output points are unit-strided (contiguous), the four points of a given
 * set form a contiguous 512-bit block. This macro performs a 4x4 transpose of
 * the 128-bit complex lanes so that, for each set, the four points are written
 * with a single 512-bit store, avoiding per-point scatter stores.
 *
 * Lane selectors for _mm512_shuffle_f64x2 (imm8): 0x44 = 0b01000100,
 * 0xEE = 0b11101110, 0x88 = 0b10001000, 0xDD = 0b11011101.
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 4, perm: 8, other: 0}
#define TRANSPOSE_ST4_512_D(base, offset, src0, src1, src2, src3)              \
{                                                                              \
    __m512d _ta = _mm512_shuffle_f64x2((src0), (src1), 0x44);                  \
    __m512d _tb = _mm512_shuffle_f64x2((src0), (src1), 0xEE);                  \
    __m512d _tc = _mm512_shuffle_f64x2((src2), (src3), 0x44);                  \
    __m512d _td = _mm512_shuffle_f64x2((src2), (src3), 0xEE);                  \
    _mm512_storeu_pd(base, _mm512_shuffle_f64x2(_ta, _tc, 0x88));              \
    _mm512_storeu_pd(base + offset, _mm512_shuffle_f64x2(_ta, _tc, 0xDD));     \
    _mm512_storeu_pd(base + 2 * offset, _mm512_shuffle_f64x2(_tb, _td, 0x88)); \
    _mm512_storeu_pd(base + 3 * offset, _mm512_shuffle_f64x2(_tb, _td, 0xDD)); \
}

/**
 * @brief Store two 512-bit double-precision vectors that hold two consecutive
 * complex output points (src0, src1), each laid out as one complex (128-bit)
 * lane per set across NUM_SETS_512_D sets, into contiguous output memory.
 *
 * The two input vectors are point-major (src_k = point k for sets 0..3). When
 * the output points are unit-strided (contiguous), the two points of a given
 * set form a contiguous 256-bit block. This macro performs a 2x4 transpose of
 * the 128-bit complex lanes so that, for each set, the two points are written
 * with a single 256-bit store, avoiding per-point scatter stores.
 *
 * Lane selectors for _mm256_permute2f128_pd (imm8): 0x20 = 0b00100000,
 * 0x31 = 0b00110001.
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 4, perm: 4, other: 2}
#define TRANSPOSE_ST2_512_D(base, offset, src0, src1)                          \
{                                                                              \
    __m256d _lo0 = _mm512_castpd512_pd256(src0);                               \
    __m256d _lo1 = _mm512_castpd512_pd256(src1);                               \
    __m256d _hi0 = _mm512_extractf64x4_pd(src0, 1);                            \
    __m256d _hi1 = _mm512_extractf64x4_pd(src1, 1);                            \
    _mm256_storeu_pd(base,                                                     \
        _mm256_permute2f128_pd(_lo0, _lo1, 0x20));                             \
    _mm256_storeu_pd(base + offset,                                            \
        _mm256_permute2f128_pd(_lo0, _lo1, 0x31));                             \
    _mm256_storeu_pd(base + 2 * offset,                                        \
        _mm256_permute2f128_pd(_hi0, _hi1, 0x20));                             \
    _mm256_storeu_pd(base + 3 * offset,                                        \
        _mm256_permute2f128_pd(_hi0, _hi1, 0x31));                             \
}

/**
 * @brief Store eight 512-bit single-precision vectors that hold eight
 * consecutive complex output points (src0..src7), each laid out as one complex
 * (64-bit) lane per set across NUM_SETS_512_S sets, into contiguous output
 * memory.
 *
 * The eight input vectors are point-major (src_k = point k for sets 0..7). When
 * the output points are unit-strided (contiguous), the eight points of a given
 * set form a contiguous 512-bit block. This macro performs an 8x8 transpose of
 * the 64-bit complex lanes so that, for each set, the eight points are written
 * with a single 512-bit store, avoiding per-point scatter stores.
 *
 * Lane selectors for _mm512_shuffle_f64x2 (imm8): 0x88 = 0b10001000,
 * 0xDD = 0b11011101.
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 8, perm: 24, other: 0}
#define TRANSPOSE_ST8_512_S(base, offset, src0, src1, src2, src3, src4, src5,  \
                         src6, src7)                                           \
{                                                                              \
    __m512d _r0 = _mm512_castps_pd(src0);                                      \
    __m512d _r1 = _mm512_castps_pd(src1);                                      \
    __m512d _r2 = _mm512_castps_pd(src2);                                      \
    __m512d _r3 = _mm512_castps_pd(src3);                                      \
    __m512d _r4 = _mm512_castps_pd(src4);                                      \
    __m512d _r5 = _mm512_castps_pd(src5);                                      \
    __m512d _r6 = _mm512_castps_pd(src6);                                      \
    __m512d _r7 = _mm512_castps_pd(src7);                                      \
    __m512d _t0 = _mm512_unpacklo_pd(_r0, _r1);                                \
    __m512d _t1 = _mm512_unpackhi_pd(_r0, _r1);                                \
    __m512d _t2 = _mm512_unpacklo_pd(_r2, _r3);                                \
    __m512d _t3 = _mm512_unpackhi_pd(_r2, _r3);                                \
    __m512d _t4 = _mm512_unpacklo_pd(_r4, _r5);                                \
    __m512d _t5 = _mm512_unpackhi_pd(_r4, _r5);                                \
    __m512d _t6 = _mm512_unpacklo_pd(_r6, _r7);                                \
    __m512d _t7 = _mm512_unpackhi_pd(_r6, _r7);                                \
    __m512d _u0 = _mm512_shuffle_f64x2(_t0, _t2, 0x88);                        \
    __m512d _u1 = _mm512_shuffle_f64x2(_t1, _t3, 0x88);                        \
    __m512d _u2 = _mm512_shuffle_f64x2(_t0, _t2, 0xDD);                        \
    __m512d _u3 = _mm512_shuffle_f64x2(_t1, _t3, 0xDD);                        \
    __m512d _u4 = _mm512_shuffle_f64x2(_t4, _t6, 0x88);                        \
    __m512d _u5 = _mm512_shuffle_f64x2(_t5, _t7, 0x88);                        \
    __m512d _u6 = _mm512_shuffle_f64x2(_t4, _t6, 0xDD);                        \
    __m512d _u7 = _mm512_shuffle_f64x2(_t5, _t7, 0xDD);                        \
    _mm512_storeu_ps(base,                                                     \
        _mm512_castpd_ps(_mm512_shuffle_f64x2(_u0, _u4, 0x88)));               \
    _mm512_storeu_ps(base + offset,                                            \
        _mm512_castpd_ps(_mm512_shuffle_f64x2(_u1, _u5, 0x88)));               \
    _mm512_storeu_ps(base + 2 * offset,                                        \
        _mm512_castpd_ps(_mm512_shuffle_f64x2(_u2, _u6, 0x88)));               \
    _mm512_storeu_ps(base + 3 * offset,                                        \
        _mm512_castpd_ps(_mm512_shuffle_f64x2(_u3, _u7, 0x88)));               \
    _mm512_storeu_ps(base + 4 * offset,                                        \
        _mm512_castpd_ps(_mm512_shuffle_f64x2(_u0, _u4, 0xDD)));               \
    _mm512_storeu_ps(base + 5 * offset,                                        \
        _mm512_castpd_ps(_mm512_shuffle_f64x2(_u1, _u5, 0xDD)));               \
    _mm512_storeu_ps(base + 6 * offset,                                        \
        _mm512_castpd_ps(_mm512_shuffle_f64x2(_u2, _u6, 0xDD)));               \
    _mm512_storeu_ps(base + 7 * offset,                                        \
        _mm512_castpd_ps(_mm512_shuffle_f64x2(_u3, _u7, 0xDD)));               \
}

/**
 * @brief Store four 512-bit single-precision vectors holding four consecutive
 * complex output points (src0..src3), one complex (64-bit) lane per set across
 * NUM_SETS_512_S sets, into contiguous output memory. Performs an 8x4 transpose
 * so that, for each of the eight sets, the four points are written with a
 * single 256-bit store. Used for a leftover group of four points.
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 8, perm: 8, other: 4}
#define TRANSPOSE_ST4_512_S(base, offset, src0, src1, src2, src3)              \
{                                                                              \
    __m512d _a0 = _mm512_castps_pd(src0);                                      \
    __m512d _a1 = _mm512_castps_pd(src1);                                      \
    __m512d _a2 = _mm512_castps_pd(src2);                                      \
    __m512d _a3 = _mm512_castps_pd(src3);                                      \
    __m512d _t0 = _mm512_unpacklo_pd(_a0, _a1);                                \
    __m512d _t1 = _mm512_unpackhi_pd(_a0, _a1);                                \
    __m512d _t2 = _mm512_unpacklo_pd(_a2, _a3);                                \
    __m512d _t3 = _mm512_unpackhi_pd(_a2, _a3);                                \
    const __m512i _idxlo = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);        \
    const __m512i _idxhi = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);      \
    __m512d _e0 = _mm512_permutex2var_pd(_t0, _idxlo, _t2);                    \
    __m512d _e1 = _mm512_permutex2var_pd(_t0, _idxhi, _t2);                    \
    __m512d _e2 = _mm512_permutex2var_pd(_t1, _idxlo, _t3);                    \
    __m512d _e3 = _mm512_permutex2var_pd(_t1, _idxhi, _t3);                    \
    _mm256_storeu_ps(base,                                                     \
        _mm256_castpd_ps(_mm512_castpd512_pd256(_e0)));                        \
    _mm256_storeu_ps(base + offset,                                            \
        _mm256_castpd_ps(_mm512_castpd512_pd256(_e2)));                        \
    _mm256_storeu_ps(base + 2 * offset,                                        \
        _mm256_castpd_ps(_mm512_extractf64x4_pd(_e0, 1)));                     \
    _mm256_storeu_ps(base + 3 * offset,                                        \
        _mm256_castpd_ps(_mm512_extractf64x4_pd(_e2, 1)));                     \
    _mm256_storeu_ps(base + 4 * offset,                                        \
        _mm256_castpd_ps(_mm512_castpd512_pd256(_e1)));                        \
    _mm256_storeu_ps(base + 5 * offset,                                        \
        _mm256_castpd_ps(_mm512_castpd512_pd256(_e3)));                        \
    _mm256_storeu_ps(base + 6 * offset,                                        \
        _mm256_castpd_ps(_mm512_extractf64x4_pd(_e1, 1)));                     \
    _mm256_storeu_ps(base + 7 * offset,                                        \
        _mm256_castpd_ps(_mm512_extractf64x4_pd(_e3, 1)));                     \
}

/**
 * @brief Store two 512-bit single-precision vectors holding two consecutive
 * complex output points (src0, src1), one complex (64-bit) lane per set across
 * NUM_SETS_512_S sets, into contiguous output memory. Performs an 8x2 transpose
 * so that, for each of the eight sets, the two points are written with a single
 * 128-bit store. Used for a leftover group of two points.
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 8, perm: 2, other: 8}
#define TRANSPOSE_ST2_512_S(base, offset, src0, src1)                          \
{                                                                              \
    __m512d _a0 = _mm512_castps_pd(src0);                                      \
    __m512d _a1 = _mm512_castps_pd(src1);                                      \
    __m512 _zlo = _mm512_castpd_ps(_mm512_unpacklo_pd(_a0, _a1));              \
    __m512 _zhi = _mm512_castpd_ps(_mm512_unpackhi_pd(_a0, _a1));              \
    _mm_storeu_ps(base, _mm512_extractf32x4_ps(_zlo, 0));                      \
    _mm_storeu_ps(base + offset, _mm512_extractf32x4_ps(_zhi, 0));             \
    _mm_storeu_ps(base + 2 * offset, _mm512_extractf32x4_ps(_zlo, 1));         \
    _mm_storeu_ps(base + 3 * offset, _mm512_extractf32x4_ps(_zhi, 1));         \
    _mm_storeu_ps(base + 4 * offset, _mm512_extractf32x4_ps(_zlo, 2));         \
    _mm_storeu_ps(base + 5 * offset, _mm512_extractf32x4_ps(_zhi, 2));         \
    _mm_storeu_ps(base + 6 * offset, _mm512_extractf32x4_ps(_zlo, 3));         \
    _mm_storeu_ps(base + 7 * offset, _mm512_extractf32x4_ps(_zhi, 3));         \
}

// Cost: {fma: 1, mul: 1, add: 0, move: 6, perm: 3, other: 3}
#define ITW_GATHER_512_D(gbase, starr, stidx, offset, gdest, twbuf,            \
                         load_multi_cols, is_contiguous)                       \
{                                                                              \
    __m512d twv;                                                               \
    if ((load_multi_cols))                                                     \
    {                                                                          \
        const FFTZ_UINTP addr = DATA_STRIDE * (stidx - 1) * NUM_SETS_512_D;    \
        twv = _mm512_loadu_pd((twbuf) + addr);                                 \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        const FFTZ_UINTP addr = DATA_STRIDE * (stidx - 1);                     \
        twv = _mm512_broadcast_f64x2(_mm_load_pd((twbuf) + addr));             \
    }                                                                          \
    __m512d tmp_in;                                                            \
    GATHER4_512_D((gbase) + starr[stidx], offset, tmp_in, is_contiguous);      \
    const __m512d twr = BROADCAST_RE_512_D(twv);                               \
    const __m512d twi = BROADCAST_IM_512_D(twv);                               \
    const __m512d tmp_i = _mm512_mul_pd(SWAP_RI_512_D(tmp_in), twi);           \
    gdest = FMSUBADD_512_D(tmp_in, twr, tmp_i);                                \
}

// Cost: {fma: 1, mul: 1, add: 0, move: 6, perm: 3, other: 3}
#define TW_GATHER_512_D(gbase, starr, stidx, offset, gdest, twbuf,             \
                        load_multi_cols, is_contiguous)                        \
{                                                                              \
    __m512d twv;                                                               \
    if ((load_multi_cols))                                                     \
    {                                                                          \
        const FFTZ_UINTP addr = DATA_STRIDE * (stidx - 1) * NUM_SETS_512_D;    \
        twv = _mm512_loadu_pd((twbuf) + addr);                                 \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        const FFTZ_UINTP addr = DATA_STRIDE * (stidx - 1);                     \
        twv = _mm512_broadcast_f64x2(_mm_load_pd((twbuf) + addr));             \
    }                                                                          \
    __m512d tmp_in;                                                            \
    GATHER4_512_D((gbase) + starr[stidx], offset, tmp_in, is_contiguous);      \
    const __m512d twr = BROADCAST_RE_512_D(twv);                               \
    const __m512d twi = BROADCAST_IM_512_D(twv);                               \
    const __m512d tmp_i = _mm512_mul_pd(SWAP_RI_512_D(tmp_in), twi);           \
    gdest = FMADDSUB_512_D(tmp_in, twr, tmp_i);                                \
}

// Cost: {fma: 1, mul: 1, add: 0, move: 10, perm: 6, other: 3}
#define ITW_GATHER_512_S(gbase, starr, stidx, offset, gdest, twbuf,            \
                         load_multi_cols, is_contiguous)                       \
{                                                                              \
    __m512 twv;                                                                \
    if ((load_multi_cols))                                                     \
    {                                                                          \
        const FFTZ_UINTP addr = DATA_STRIDE * (stidx - 1) * NUM_SETS_512_S;    \
        twv = _mm512_loadu_ps((twbuf) + addr);                                 \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        const FFTZ_UINTP addr = DATA_STRIDE * (stidx - 1);                     \
        twv = _mm512_broadcast_f32x2(                                          \
            _mm_loadl_pi(_mm_setzero_ps(), (__m64 *)((twbuf) + addr)));        \
    }                                                                          \
    __m512 tmp_in;                                                             \
    GATHER8_512_S((gbase) + starr[stidx], offset, tmp_in, is_contiguous);      \
    const __m512 twr = BROADCAST_RE_512_S(twv);                                \
    const __m512 twi = BROADCAST_IM_512_S(twv);                                \
    const __m512 tmp_i = _mm512_mul_ps(SWAP_RI_512_S(tmp_in), twi);            \
    gdest = FMSUBADD_512_S(tmp_in, twr, tmp_i);                                \
}

// Cost: {fma: 1, mul: 1, add: 0, move: 10, perm: 6, other: 3}
#define TW_GATHER_512_S(gbase, starr, stidx, offset, gdest, twbuf,             \
                        load_multi_cols, is_contiguous)                        \
{                                                                              \
    __m512 twv;                                                                \
    if ((load_multi_cols))                                                     \
    {                                                                          \
        const FFTZ_UINTP addr = DATA_STRIDE * (stidx - 1) * NUM_SETS_512_S;    \
        twv = _mm512_loadu_ps((twbuf) + addr);                                 \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        const FFTZ_UINTP addr = DATA_STRIDE * (stidx - 1);                     \
        twv = _mm512_broadcast_f32x2(                                          \
            _mm_loadl_pi(_mm_setzero_ps(), (__m64 *)((twbuf) + addr)));        \
    }                                                                          \
    __m512 tmp_in;                                                             \
    GATHER8_512_S((gbase) + starr[stidx], offset, tmp_in, is_contiguous);      \
    const __m512 twr = BROADCAST_RE_512_S(twv);                                \
    const __m512 twi = BROADCAST_IM_512_S(twv);                                \
    const __m512 tmp_i = _mm512_mul_ps(SWAP_RI_512_S(tmp_in), twi);            \
    gdest = FMADDSUB_512_S(tmp_in, twr, tmp_i);                                \
}

// Cost: {fma: 1, mul: 2, add: 0, move: 5, perm: 4, other: 0}
#define ITW_PRELOADED_GATHER_512_D(gbase, starr, stidx, offset, gdest, twv,    \
                                   is_contiguous)                              \
{                                                                              \
    __m512d tmp_in;                                                            \
    GATHER4_512_D((gbase) + starr[stidx], offset, tmp_in, is_contiguous);      \
    const __m512d twr = BROADCAST_RE_512_D(twv);                               \
    const __m512d twi = BROADCAST_IM_512_D(twv);                               \
    const __m512d tmp_i = _mm512_mul_pd(SWAP_RI_512_D(tmp_in), twi);           \
    gdest = FMSUBADD_512_D(tmp_in, twr, tmp_i);                                \
}

// Cost: {fma: 1, mul: 2, add: 0, move: 5, perm: 3, other: 0}
#define TW_PRELOADED_GATHER_512_D(gbase, starr, stidx, offset, gdest, twv,     \
                                  is_contiguous)                               \
{                                                                              \
    __m512d tmp_in;                                                            \
    GATHER4_512_D((gbase) + starr[stidx], offset, tmp_in, is_contiguous);      \
    const __m512d twr = BROADCAST_RE_512_D(twv);                               \
    const __m512d twi = BROADCAST_IM_512_D(twv);                               \
    const __m512d tmp_i = _mm512_mul_pd(SWAP_RI_512_D(tmp_in), twi);           \
    gdest = FMADDSUB_512_D(tmp_in, twr, tmp_i);                                \
}

// Cost: {fma: 1, mul: 2, add: 0, move: 5, perm: 4, other: 0}
#define ITW_PRELOADED_GATHER_512_D_V(gbase, stride, offset, gdest, twv,        \
                                     is_contiguous)                            \
{                                                                              \
    __m512d tmp_in;                                                            \
    GATHER4_512_D((gbase) + (stride), offset, tmp_in, is_contiguous);          \
    const __m512d twr = BROADCAST_RE_512_D(twv);                               \
    const __m512d twi = BROADCAST_IM_512_D(twv);                               \
    const __m512d tmp_i = _mm512_mul_pd(SWAP_RI_512_D(tmp_in), twi);           \
    gdest = FMSUBADD_512_D(tmp_in, twr, tmp_i);                                \
}

// Cost: {fma: 1, mul: 2, add: 0, move: 5, perm: 3, other: 0}
#define TW_PRELOADED_GATHER_512_D_V(gbase, stride, offset, gdest, twv,         \
                                    is_contiguous)                             \
{                                                                              \
    __m512d tmp_in;                                                            \
    GATHER4_512_D((gbase) + (stride), offset, tmp_in, is_contiguous);          \
    const __m512d twr = BROADCAST_RE_512_D(twv);                               \
    const __m512d twi = BROADCAST_IM_512_D(twv);                               \
    const __m512d tmp_i = _mm512_mul_pd(SWAP_RI_512_D(tmp_in), twi);           \
    gdest = FMADDSUB_512_D(tmp_in, twr, tmp_i);                                \
}

// Cost: {fma: 1, mul: 1, add: 0, move: 4, perm: 3, other: 3}
#define TW_PRELOADED_SCATTER_512_D(sbase, stride, offset, ssrc, twv,           \
                                   is_contiguous)                              \
{                                                                              \
    const __m512d twr = BROADCAST_RE_512_D(twv);                               \
    const __m512d twi = BROADCAST_IM_512_D(twv);                               \
    const __m512d tmp_i = _mm512_mul_pd(SWAP_RI_512_D(ssrc), twi);             \
    __m512d result = FMADDSUB_512_D(ssrc, twr, tmp_i);                         \
    SCATTER4_512_D((sbase) + (stride), offset, result, is_contiguous);         \
}

// No-twiddle preloaded gather/scatter: signature-compatible with
// TW_PRELOADED_GATHER_512_D_V / TW_PRELOADED_SCATTER_512_D, accepts and
// ignores twv. Used by c2r load and fwd/bwd/r2c store paths so wrapper-local
// TWID_LOAD/STORE_PRELOADED_*_D shims can be pure name-substitution defines.
#define PRELOADED_GATHER_NOTW_512_D(gbase, stride, offset, gdest, twv,         \
                                    is_contiguous)                             \
{                                                                              \
    (FFTZ_VOID)twv;                                                            \
    GATHER4_512_D((gbase) + (stride), offset, (gdest), is_contiguous);         \
}

#define PRELOADED_SCATTER_NOTW_512_D(sbase, stride, offset, ssrc, twv,         \
                                     is_contiguous)                            \
{                                                                              \
    (FFTZ_VOID)twv;                                                            \
    SCATTER4_512_D((sbase) + (stride), offset, (ssrc), is_contiguous);         \
}

/*****************************************************************************
 * GATHER_NOTW / SCATTER_NOTW -- 512-bit variants
 *****************************************************************************/

#define GATHER_NOTW_512_S(gbase, starr, stidx, offset, gdest, twbuf, lmc,      \
                          is_contiguous)                                       \
    GATHER8_512_S((gbase) + starr[(stidx)], offset, gdest, is_contiguous)

#define GATHER_NOTW_512_D(gbase, starr, stidx, offset, gdest, twbuf, lmc,      \
                          is_contiguous)                                       \
    GATHER4_512_D((gbase) + starr[(stidx)], offset, gdest, is_contiguous)

#define SCATTER_NOTW_512_S(sbase, starr, stidx, offset, ssrc, twbuf, lmc,      \
                           is_contiguous)                                      \
    SCATTER8_512_S((sbase) + starr[(stidx)], offset, ssrc, is_contiguous)

#define SCATTER_NOTW_512_D(sbase, starr, stidx, offset, ssrc, twbuf, lmc,      \
                           is_contiguous)                                      \
    SCATTER4_512_D((sbase) + starr[(stidx)], offset, ssrc, is_contiguous)

/**
 * @brief H2 (conjugate-half) counterparts of GATHER_NOTW_512_* /
 * SCATTER_NOTW_512_* above. Same signature-compatible, twiddle-ignoring
 * interface, but route through the reverse-contiguous H2 gather/scatter
 * primitives defined earlier in this file.
 */
#define GATHER_NOTW_H2_512_S(gbase, starr, stidx, offset, gdest, twbuf, lmc,   \
                             is_contiguous)                                    \
    GATHER8_H2_512_S((gbase) + starr[(stidx)], offset, gdest)

#define GATHER_NOTW_H2_512_D(gbase, starr, stidx, offset, gdest, twbuf, lmc,   \
                             is_contiguous)                                    \
    GATHER4_H2_512_D((gbase) + starr[(stidx)], offset, gdest)

#define SCATTER_NOTW_H2_512_S(sbase, starr, stidx, offset, ssrc, twbuf, lmc,   \
                              is_contiguous)                                   \
    SCATTER8_H2_512_S((sbase) + starr[(stidx)], offset, ssrc)

#define SCATTER_NOTW_H2_512_D(sbase, starr, stidx, offset, ssrc, twbuf, lmc,   \
                              is_contiguous)                                   \
    SCATTER4_H2_512_D((sbase) + starr[(stidx)], offset, ssrc)

/*****************************************************************************
 * TW_SCATTER / ITW_SCATTER -- 512-bit variants
 *****************************************************************************/

#define TW_SCATTER_512_D(sbase, starr, stidx, offset, ssrc, twbuf, lmc,        \
                         is_contiguous)                                        \
{                                                                              \
    __m512d twv;                                                               \
    if ((lmc))                                                                 \
    {                                                                          \
        const FFTZ_UINTP addr =                                                \
            DATA_STRIDE * (stidx - 1) * NUM_SETS_512_D;                        \
        twv = _mm512_loadu_pd((twbuf) + addr);                                 \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        const FFTZ_UINTP addr = DATA_STRIDE * (stidx - 1);                     \
        twv = _mm512_broadcast_f64x2(_mm_loadu_pd((twbuf) + addr));            \
    }                                                                          \
    const __m512d twr = BROADCAST_RE_512_D(twv);                               \
    const __m512d twi = BROADCAST_IM_512_D(twv);                               \
    const __m512d tmp_i = _mm512_mul_pd(SWAP_RI_512_D(ssrc), twi);             \
    __m512d _result = FMADDSUB_512_D(ssrc, twr, tmp_i);                        \
    SCATTER4_512_D((sbase) + starr[(stidx)], offset, _result,                  \
        is_contiguous);                                                        \
}

#define ITW_SCATTER_512_D(sbase, starr, stidx, offset, ssrc, twbuf, lmc,       \
                          is_contiguous)                                       \
{                                                                              \
    __m512d twv;                                                               \
    if ((lmc))                                                                 \
    {                                                                          \
        const FFTZ_UINTP addr =                                                \
            DATA_STRIDE * (stidx - 1) * NUM_SETS_512_D;                        \
        twv = _mm512_loadu_pd((twbuf) + addr);                                 \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        const FFTZ_UINTP addr = DATA_STRIDE * (stidx - 1);                     \
        twv = _mm512_broadcast_f64x2(_mm_loadu_pd((twbuf) + addr));            \
    }                                                                          \
    const __m512d twr = BROADCAST_RE_512_D(twv);                               \
    const __m512d twi = BROADCAST_IM_512_D(twv);                               \
    const __m512d tmp_i = _mm512_mul_pd(SWAP_RI_512_D(ssrc), twi);             \
    __m512d _result = FMSUBADD_512_D(ssrc, twr, tmp_i);                        \
    SCATTER4_512_D((sbase) + starr[(stidx)], offset, _result,                  \
        is_contiguous);                                                        \
}

#define TW_SCATTER_512_S(sbase, starr, stidx, offset, ssrc, twbuf, lmc,        \
                         is_contiguous)                                        \
{                                                                              \
    __m512 twv;                                                                \
    if ((lmc))                                                                 \
    {                                                                          \
        const FFTZ_UINTP addr =                                                \
            DATA_STRIDE * (stidx - 1) * NUM_SETS_512_S;                        \
        twv = _mm512_loadu_ps((twbuf) + addr);                                 \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        const FFTZ_UINTP addr = DATA_STRIDE * (stidx - 1);                     \
        twv = _mm512_broadcast_f32x2(                                          \
            _mm_loadl_pi(_mm_setzero_ps(), (__m64 *)((twbuf) + addr)));        \
    }                                                                          \
    const __m512 twr = BROADCAST_RE_512_S(twv);                                \
    const __m512 twi = BROADCAST_IM_512_S(twv);                                \
    const __m512 tmp_i = _mm512_mul_ps(SWAP_RI_512_S(ssrc), twi);              \
    __m512 _result = FMADDSUB_512_S(ssrc, twr, tmp_i);                         \
    SCATTER8_512_S((sbase) + starr[(stidx)], offset, _result,                  \
        is_contiguous);                                                        \
}

#define ITW_SCATTER_512_S(sbase, starr, stidx, offset, ssrc, twbuf, lmc,       \
                          is_contiguous)                                       \
{                                                                              \
    __m512 twv;                                                                \
    if ((lmc))                                                                 \
    {                                                                          \
        const FFTZ_UINTP addr =                                                \
            DATA_STRIDE * (stidx - 1) * NUM_SETS_512_S;                        \
        twv = _mm512_loadu_ps((twbuf) + addr);                                 \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        const FFTZ_UINTP addr = DATA_STRIDE * (stidx - 1);                     \
        twv = _mm512_broadcast_f32x2(                                          \
            _mm_loadl_pi(_mm_setzero_ps(), (__m64 *)((twbuf) + addr)));        \
    }                                                                          \
    const __m512 twr = BROADCAST_RE_512_S(twv);                                \
    const __m512 twi = BROADCAST_IM_512_S(twv);                                \
    const __m512 tmp_i = _mm512_mul_ps(SWAP_RI_512_S(ssrc), twi);              \
    __m512 _result = FMSUBADD_512_S(ssrc, twr, tmp_i);                         \
    SCATTER8_512_S((sbase) + starr[(stidx)], offset, _result,                  \
        is_contiguous);                                                        \
}

// No-twiddle preloaded gather/scatter: signature-compatible with
// TW_PRELOADED_GATHER_512_D_V / TW_PRELOADED_SCATTER_512_D, accepts and
// ignores twv. Used by c2r load and fwd/bwd/r2c store paths so wrapper-local
// TWID_PRELOADED_LOAD/STORE_*_D shims can be pure name-substitution defines.
#define PRELOADED_GATHER_NOTW_512_D(gbase, stride, offset, gdest, twv,         \
                                    is_contiguous)                             \
{                                                                              \
    (FFTZ_VOID)twv;                                                            \
    GATHER4_512_D((gbase) + (stride), offset, (gdest), is_contiguous);         \
}

#define PRELOADED_SCATTER_NOTW_512_D(sbase, stride, offset, ssrc, twv,         \
                                     is_contiguous)                            \
{                                                                              \
    (FFTZ_VOID)twv;                                                            \
    SCATTER4_512_D((sbase) + (stride), offset, (ssrc), is_contiguous);         \
}

/*****************************************************************************
 * GATHER_NOTW / SCATTER_NOTW -- 512-bit variants
 *****************************************************************************/

#define GATHER_NOTW_512_S(gbase, starr, stidx, offset, gdest, twbuf, lmc,      \
                          is_contiguous)                                       \
    GATHER8_512_S((gbase) + starr[(stidx)], offset, gdest, is_contiguous)

#define GATHER_NOTW_512_D(gbase, starr, stidx, offset, gdest, twbuf, lmc,      \
                          is_contiguous)                                       \
    GATHER4_512_D((gbase) + starr[(stidx)], offset, gdest, is_contiguous)

#define SCATTER_NOTW_512_S(sbase, starr, stidx, offset, ssrc, twbuf, lmc,      \
                           is_contiguous)                                      \
    SCATTER8_512_S((sbase) + starr[(stidx)], offset, ssrc, is_contiguous)

#define SCATTER_NOTW_512_D(sbase, starr, stidx, offset, ssrc, twbuf, lmc,      \
                           is_contiguous)                                      \
    SCATTER4_512_D((sbase) + starr[(stidx)], offset, ssrc, is_contiguous)

#endif // AOCLFFTZ_SIMD_COMMON_AVX512_H
