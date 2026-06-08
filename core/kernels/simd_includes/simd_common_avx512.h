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
#define GATHER8_512_S(base, offset, dest)                                      \
    {                                                                          \
        if (offset == 2)                                                       \
        {                                                                      \
            dest = _mm512_loadu_ps(base);                                      \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            __m128 _low, _high, _tmp;                                          \
            __m256 _256low, _256high;                                          \
            _low = _mm_loadu_ps(base);                                         \
            _tmp = _mm_loadu_ps((base) + (offset));                            \
            _low = _mm_shuffle_ps(_low, _tmp, 68);                             \
            _high = _mm_loadu_ps((base) + 2 * (offset));                       \
            _tmp = _mm_loadu_ps((base) + 3 * (offset));                        \
            _high = _mm_shuffle_ps(_high, _tmp, 68);                           \
            _256low =                                                          \
                _mm256_insertf128_ps(_mm256_castps128_ps256(_low), _high, 1);  \
            _low = _mm_loadu_ps((base) + 4 * (offset));                        \
            _tmp = _mm_loadu_ps((base) + 5 * (offset));                        \
            _low = _mm_shuffle_ps(_low, _tmp, 68);                             \
            _high = _mm_loadu_ps((base) + 6 * (offset));                       \
            _high = _mm_loadh_pi(_high, (__m64 *)((base) + 7 * (offset)));     \
            _256high = _mm256_insertf128_ps(_mm256_castps128_ps256(_low),      \
                                             _high, 1);                        \
            dest = _mm512_insertf32x8(_mm512_castps256_ps512(_256low),         \
                                      _256high, 1);                            \
        }                                                                      \
    }

/**
 * @brief store eight complex numbers(real,imaginary) of 32 bit single precision
 * floating point number from 512 bit register into memory addresses
 * specified by base address and offset.
 * Operations : 8 MOV(store), 3 OTHERS(extract).
 * Cast is excluded as it will be a compile time operation.
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 8, perm: 0, other: 3}
#define SCATTER8_512_S(base, offset, src)                                      \
{                                                                              \
    if (offset == 2)                                                           \
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
        _mm_storeh_pi((__m64 *)((base) + (offset)), _low);                     \
        _mm_storel_pi((__m64 *)((base) + 2 * (offset)), _high);                \
        _mm_storeh_pi((__m64 *)((base) + 3 * (offset)), _high);                \
        _high = _mm256_extractf128_ps(_256high, 1);                            \
        _low = _mm256_castps256_ps128(_256high);                               \
        _mm_storel_pi((__m64 *)((base) + 4 * (offset)), _low);                 \
        _mm_storeh_pi((__m64 *)((base) + 5 * (offset)), _low);                 \
        _mm_storel_pi((__m64 *)((base) + 6 * (offset)), _high);                \
        _mm_storeh_pi((__m64 *)((base) + 7 * (offset)), _high);                \
    }                                                                          \
}

/**
 * @brief load four complex numbers(real,imaginary) of 64 bit double precision
 * floating point number from memory addresses specified by base address
 * and offset into 512 bit register.
 * Operations : 4 MOV(load), 3 OTHERS(insert).
 * Cast is excluded as it will be a compile time operation
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 4, perm: 0, other: 3}
#define GATHER4_512_D(base, offset, dest)                                      \
{                                                                              \
    if (offset == 2)                                                           \
    {                                                                          \
        dest = _mm512_loadu_pd(base);                                          \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        __m128d _low, _high;                                                   \
        __m256d _256low, _256high;                                             \
        _low = _mm_loadu_pd(base);                                             \
        _high = _mm_loadu_pd((base) + (offset));                               \
        _256low = _mm256_insertf128_pd(_mm256_castpd128_pd256(_low), _high, 1);\
        _low = _mm_loadu_pd((base) + offset * 2);                              \
        _high = _mm_loadu_pd((base) + offset * 3);                             \
        _256high =                                                             \
            _mm256_insertf128_pd(_mm256_castpd128_pd256(_low), _high, 1);      \
        dest = _mm512_insertf64x4(_mm512_castpd256_pd512(_256low),             \
                                    _256high, 1);                              \
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
#define SCATTER4_512_D(base, offset, src)                                      \
{                                                                              \
    if (offset == 2)                                                           \
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
        _mm_storeu_pd((base) + offset, _high);                                 \
        _high = _mm256_extractf128_pd(_m256high, 1);                           \
        _low = _mm256_castpd256_pd128(_m256high);                              \
        _mm_storeu_pd((base) + 2 * offset, _low);                              \
        _mm_storeu_pd((base) + 3 * offset, _high);                             \
    }                                                                          \
}

// Cost: {fma: 1, mul: 1, add: 0, move: 6, perm: 3, other: 3}
#define ITW_GATHER_512_D(gbase, starr, stidx, offset, gdest, twbuf, n, col,    \
                         load_multi_cols)                                      \
{                                                                              \
    const UINTP addr = DATA_STRIDE * ((stidx) * (n) + (col));                  \
    __m512d twv;                                                               \
    if ((load_multi_cols)) {                                                   \
        twv = _mm512_loadu_pd((twbuf) + addr);                                 \
    }                                                                          \
    else {                                                                     \
        twv = _mm512_broadcast_f64x2(_mm_load_pd((twbuf) + addr));             \
    }                                                                          \
    __m512d tmp_in;                                                            \
    GATHER4_512_D((gbase) + starr[(stidx)], (offset), tmp_in);                 \
    const __m512d twr = BROADCAST_RE_512_D(twv);                               \
    const __m512d twi = BROADCAST_IM_512_D(twv);                               \
    const __m512d tmp_i = _mm512_mul_pd(SWAP_RI_512_D(tmp_in), twi);           \
    gdest = FMSUBADD_512_D(tmp_in, twr, tmp_i);                                \
}

// Cost: {fma: 1, mul: 1, add: 0, move: 6, perm: 3, other: 3}
#define TW_GATHER_512_D(gbase, starr, stidx, offset, gdest, twbuf, n, col,     \
                        load_multi_cols)                                       \
{                                                                              \
    const UINTP addr = DATA_STRIDE * ((stidx) * (n) + (col));                  \
    __m512d twv;                                                               \
    if ((load_multi_cols)) {                                                   \
        twv = _mm512_loadu_pd((twbuf) + addr);                                 \
    }                                                                          \
    else {                                                                     \
        twv = _mm512_broadcast_f64x2(_mm_load_pd((twbuf) + addr));             \
    }                                                                          \
    __m512d tmp_in;                                                            \
    GATHER4_512_D((gbase) + starr[(stidx)], (offset), tmp_in);                 \
    const __m512d twr = BROADCAST_RE_512_D(twv);                               \
    const __m512d twi = BROADCAST_IM_512_D(twv);                               \
    const __m512d tmp_i = _mm512_mul_pd(SWAP_RI_512_D(tmp_in), twi);           \
    gdest = FMADDSUB_512_D(tmp_in, twr, tmp_i);                                \
}

// Cost: {fma: 1, mul: 1, add: 0, move: 10, perm: 6, other: 3}
#define ITW_GATHER_512_S(gbase, starr, stidx, offset, gdest, twbuf, n, col,    \
                         load_multi_cols)                                      \
{                                                                              \
    const UINTP addr = DATA_STRIDE * ((stidx) * (n) + (col));                  \
    __m512 twv;                                                                \
    if ((load_multi_cols)) {                                                   \
        twv = _mm512_loadu_ps((twbuf) + addr);                                 \
    }                                                                          \
    else {                                                                     \
        twv = _mm512_broadcast_f32x2(_mm_loadl_pi(_mm_setzero_ps(),            \
                                                  (__m64 *)((twbuf) + addr))); \
    }                                                                          \
    __m512 tmp_in;                                                             \
    GATHER8_512_S((gbase) + starr[(stidx)], (offset), tmp_in);                 \
    const __m512 twr = BROADCAST_RE_512_S(twv);                                \
    const __m512 twi = BROADCAST_IM_512_S(twv);                                \
    const __m512 tmp_i = _mm512_mul_ps(SWAP_RI_512_S(tmp_in), twi);            \
    gdest = FMSUBADD_512_S(tmp_in, twr, tmp_i);                                \
}

// Cost: {fma: 1, mul: 1, add: 0, move: 10, perm: 6, other: 3}
#define TW_GATHER_512_S(gbase, starr, stidx, offset, gdest, twbuf, n, col,     \
                        load_multi_cols)                                       \
{                                                                              \
    const UINTP addr = DATA_STRIDE * ((stidx) * (n) + (col));                  \
    __m512 twv;                                                                \
    if ((load_multi_cols)) {                                                   \
        twv = _mm512_loadu_ps((twbuf) + addr);                                 \
    }                                                                          \
    else {                                                                     \
        twv = _mm512_broadcast_f32x2(_mm_loadl_pi(_mm_setzero_ps(),            \
                                                  (__m64 *)((twbuf) + addr))); \
    }                                                                          \
    __m512 tmp_in;                                                             \
    GATHER8_512_S((gbase) + starr[(stidx)], (offset), tmp_in);                 \
    const __m512 twr = BROADCAST_RE_512_S(twv);                                \
    const __m512 twi = BROADCAST_IM_512_S(twv);                                \
    const __m512 tmp_i = _mm512_mul_ps(SWAP_RI_512_S(tmp_in), twi);            \
    gdest = FMADDSUB_512_S(tmp_in, twr, tmp_i);                                \
}

// Cost: {fma: 1, mul: 1, add: 0, move: 4, perm: 3, other: 3}
#define ITW_PRELOADED_512_D(gbase, starr, stidx, offset, gdest, twv)           \
{                                                                              \
    __m512d tmp_in;                                                            \
    GATHER4_512_D((gbase) + starr[(stidx)], (offset), tmp_in);                 \
    const __m512d twr = BROADCAST_RE_512_D(twv);                               \
    const __m512d twi = BROADCAST_IM_512_D(twv);                               \
    const __m512d tmp_i = _mm512_mul_pd(SWAP_RI_512_D(tmp_in), twi);           \
    gdest = FMSUBADD_512_D(tmp_in, twr, tmp_i);                                \
}

// Cost: {fma: 1, mul: 1, add: 0, move: 4, perm: 3, other: 3}
#define TW_PRELOADED_512_D(gbase, starr, stidx, offset, gdest, twv)            \
{                                                                              \
    __m512d tmp_in;                                                            \
    GATHER4_512_D((gbase) + starr[(stidx)], (offset), tmp_in);                 \
    const __m512d twr = BROADCAST_RE_512_D(twv);                               \
    const __m512d twi = BROADCAST_IM_512_D(twv);                               \
    const __m512d tmp_i = _mm512_mul_pd(SWAP_RI_512_D(tmp_in), twi);           \
    gdest = FMADDSUB_512_D(tmp_in, twr, tmp_i);                                \
}

// Cost: {fma: 1, mul: 1, add: 0, move: 4, perm: 3, other: 3}
#define ITW_PRELOADED_512_D_V(gbase, stride, offset, gdest, twv)               \
{                                                                              \
    __m512d tmp_in;                                                            \
    GATHER4_512_D((gbase) + (stride), (offset), tmp_in);                       \
    const __m512d twr = BROADCAST_RE_512_D(twv);                               \
    const __m512d twi = BROADCAST_IM_512_D(twv);                               \
    const __m512d tmp_i = _mm512_mul_pd(SWAP_RI_512_D(tmp_in), twi);           \
    gdest = FMSUBADD_512_D(tmp_in, twr, tmp_i);                                \
}

// Cost: {fma: 1, mul: 1, add: 0, move: 4, perm: 3, other: 3}
#define TW_PRELOADED_512_D_V(gbase, stride, offset, gdest, twv)                \
{                                                                              \
    __m512d tmp_in;                                                            \
    GATHER4_512_D((gbase) + (stride), (offset), tmp_in);                       \
    const __m512d twr = BROADCAST_RE_512_D(twv);                               \
    const __m512d twi = BROADCAST_IM_512_D(twv);                               \
    const __m512d tmp_i = _mm512_mul_pd(SWAP_RI_512_D(tmp_in), twi);           \
    gdest = FMADDSUB_512_D(tmp_in, twr, tmp_i);                                \
}

#endif // AOCLFFTZ_SIMD_COMMON_AVX512_H
