// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file simd_common.h
 *
 *  @brief List of common operations and macros used for simd kernels.
 *
 *  This file contains the category-wise macros and common functions used
 *  in the AVX kernel variants
 *
 *  @author Murugan Vairavel
 *  @author Srirammaswamy S
 *  @author Jeya R
 */

#ifndef AOCLFFTZ_SIMD_COMMON_H
#define AOCLFFTZ_SIMD_COMMON_H

#include <immintrin.h>
#include "api/types.h"
/************************ MACRO FUNCTIONS ************************/

/**
 * @brief load two complex numbers(real,imaginary) of 32 bit single precision
 * floating point number from memory addresses specified by base address
 * and offset into 128 bit register.
 * Operations : 2 MOV(load)
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 2, perm: 0, other: 0}
#define GATHER2_128_S(base, offset, dest, is_contiguous)                       \
{                                                                              \
    if (is_contiguous)                                                         \
    {                                                                          \
        dest = _mm_loadu_ps(base);                                             \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        dest = _mm_loadu_ps(base);                                             \
        dest = _mm_loadh_pi(dest, (__m64 *)(base + offset));                   \
    }                                                                          \
}

/**
 * @brief store two complex numbers(real,imaginary) of 32 bit single precision
 * floating point number from 128 bit register into memory addresses
 * specified by base address and offset.
 * Operations : 2 MOV(store)
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 2, perm: 0, other: 0}
#define SCATTER2_128_S(base, offset, src, is_contiguous)                       \
{                                                                              \
    if (is_contiguous)                                                         \
    {                                                                          \
        _mm_storeu_ps(base, src);                                              \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        _mm_storel_pi((__m64 *)(base), src);                                   \
        _mm_storeh_pi((__m64 *)(base + offset), src);                          \
    }                                                                          \
}

/**
 * @brief Branchless variant of SCATTER2_128_S to use inside contiguous
 * (out_strides[1] == DATA_STRIDE) branches in kernels, where vector stride
 * `offset` cannot be DATA_STRIDE.
 * Operation : 2 MOV(store)
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 2, perm: 0, other: 0}
#define SCATTER2_128_S_STRIDED(base, offset, src)                              \
{                                                                              \
    _mm_storel_pi((__m64 *)(base), src);                                       \
    _mm_storeh_pi((__m64 *)(base + offset), src);                              \
}

/**
 * @brief load a complex number(real,imaginary) of 32 bit single precision
 * floating point number from memory addresses specified by base address
 * into 128 bit register.
 * Operation : 1 MOV(load)
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 0, other: 0}
#define LD_LOW_128_S(base, dest)                                               \
{                                                                              \
    dest = _mm_loadl_pi(_mm_setzero_ps(), (__m64 *)(base));                    \
}

/**
 * @brief store a complex number(real,imaginary) of 32 bit single precision
 * floating point number from 128 bit register into memory addresses
 * specified by base address.
 * Operation : 1 MOV(store)
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 0, other: 0}
#define ST_LOW_128_S(base, src)                                                \
{                                                                              \
    _mm_storel_pi((__m64 *)(base), src);                                       \
}

/**
 * @brief store two complex numbers(real,imaginary) of 32 bit single precision
 * floating point, taken from the low 64 bits of two 128 bit registers, into a
 * contiguous 128 bit memory location specified by base address. Used to fuse
 * two single-set point stores into one 128 bit store when the output is
 * contiguous (out_strides[1] == DATA_STRIDE), halving the store count.
 * Operation : 1 PERM(shuffle), 1 MOV(store)
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 1, other: 0}
#define ST_128_S(base, src0, src1)                                             \
{                                                                              \
    _mm_storeu_ps((base), _mm_movelh_ps(src0, src1));                          \
}

/**
 * @brief load a complex number(real,imaginary) of 64 bit double precision
 * floating point number from memory addresses specified by base address
 * into 128 bit register.
 * Operation : 1 MOV(load)
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 0, other: 0}
#define LD_128_D(base, dest)                                                   \
{                                                                              \
    dest = _mm_loadu_pd(base);                                                 \
}

// Generic kernels require a variant of LD_128_D with 3 args. Instead of
// changing the signature of LD_128_D and causing unnecessary edits in all the
// existing functions that use LD_128_D, we create a copy.
// Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 0, other: 0}
#define LD_128_OFFSET_D(base, offset, dest, is_contiguous)                     \
{                                                                              \
    (FFTZ_VOID)is_contiguous;                                                  \
    dest = _mm_loadu_pd(base);                                                 \
}

/**
 * @brief store a complex number(real,imaginary) of 64 bit double precision
 * floating point number from 128 bit register into memory addresses
 * specified by base address.
 * Operation : 1 MOV(store)
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 0, other: 0}
#define ST_128_D(base, src)                                                    \
{                                                                              \
    _mm_storeu_pd(base, src);                                                  \
}

// Generic kernels require a variant of ST_128_D with 3 args. Instead of
// changing the signature of ST_128_D and causing unnecessary edits in all the
// existing functions that use ST_128_D, we create a copy.
// Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 0, other: 0}
#define ST_128_OFFSET_D(base, offset, src, is_contiguous)                      \
{                                                                              \
    (FFTZ_VOID)is_contiguous;                                                  \
    _mm_storeu_pd(base, src);                                                  \
}

/**
 * @brief load four complex numbers(real,imaginary) of 32 bit single precision
 * floating point number from memory addresses specified by base address
 * and offset, into 256 bit register.
 * Operations : 4 MOV(load), 1 PERM(shuffle), 1 OTHERS(insert).Cast is excluded
 * as it will be a compile time operation.
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 4, perm: 1, other: 1}
#define GATHER4_256_S(base, offset, dest, is_contiguous)                       \
{                                                                              \
    if (is_contiguous)                                                         \
    {                                                                          \
        dest = _mm256_loadu_ps(base);                                          \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        __m128 _low, _high, _tmp;                                              \
        _low = _mm_loadu_ps(base);                                             \
        _tmp = _mm_loadu_ps(base + offset);                                    \
        _low = _mm_shuffle_ps(_low, _tmp, 68);                                 \
        _high = _mm_loadu_ps(base + 2 * offset);                               \
        _high = _mm_loadh_pi(_high, (__m64 *)(base + 3 * offset));             \
        dest = _mm256_insertf128_ps(_mm256_castps128_ps256(_low), _high, 1);   \
    }                                                                          \
}

/**
 * @brief store four complex numbers(real,imaginary) of 32 bit single precision
 * floating point number from 256 bit register into memory addresses
 * specified by base address and offset.
 * Operations : 4 MOV(store), 1 OTHERS(extract). Cast is excluded as it
 * will be a compile time operation.
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 4, perm: 0, other: 1}
#define SCATTER4_256_S(base, offset, src, is_contiguous)                       \
{                                                                              \
    if (is_contiguous)                                                         \
    {                                                                          \
        _mm256_storeu_ps(base, src);                                           \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        __m128 _high = _mm256_extractf128_ps(src, 1);                          \
        __m128 _low = _mm256_castps256_ps128(src);                             \
        _mm_storel_pi((__m64 *)(base), _low);                                  \
        _mm_storeh_pi((__m64 *)(base + offset), _low);                         \
        _mm_storel_pi((__m64 *)(base + 2 * offset), _high);                    \
        _mm_storeh_pi((__m64 *)(base + 3 * offset), _high);                    \
    }                                                                          \
}

/**
 * @brief Branchless variant of SCATTER4_256_S to use inside contiguous
 * (out_strides[1] == DATA_STRIDE) branches in kernels, where vector stride
 * `offset` cannot be DATA_STRIDE.
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 4, perm: 0, other: 1}
#define SCATTER4_256_S_STRIDED(base, offset, src)                              \
{                                                                              \
    __m128 _high = _mm256_extractf128_ps(src, 1);                              \
    __m128 _low = _mm256_castps256_ps128(src);                                 \
    _mm_storel_pi((__m64 *)(base), _low);                                      \
    _mm_storeh_pi((__m64 *)(base + offset), _low);                             \
    _mm_storel_pi((__m64 *)(base + 2 * offset), _high);                        \
    _mm_storeh_pi((__m64 *)(base + 3 * offset), _high);                        \
}

/**
 * @brief load two complex numbers(real,imaginary) of 64 bit double precision
 * floating point number from memory addresses specified by base address
 * and offset into 256 bit register.
 * Operations : 2 MOV(load), 1 OTHERS(insert). Cast is excluded as it
 * will be a compile time operation.
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 2, perm: 0, other: 1}
#define GATHER2_256_D(base, offset, dest, is_contiguous)                       \
{                                                                              \
    if (is_contiguous)                                                         \
    {                                                                          \
        dest = _mm256_loadu_pd(base);                                          \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        __m128d _low, _high;                                                   \
        _low = _mm_loadu_pd(base);                                             \
        _high = _mm_loadu_pd(base + offset);                                   \
        dest = _mm256_insertf128_pd(_mm256_castpd128_pd256(_low), _high, 1);   \
    }                                                                          \
}

/**
 * @brief store two complex numbers(real,imaginary) of 64 bit double precision
 * floating point number from 256 bit register into memory addresses
 * specified by base address and offset.
 * Operations : 2 MOV(store), 1 OTHERS(extract). Cast is excluded as it
 * will be a compile time operation.
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 2, perm: 0, other: 1}
#define SCATTER2_256_D(base, offset, src, is_contiguous)                       \
{                                                                              \
    if (is_contiguous)                                                         \
    {                                                                          \
        _mm256_storeu_pd(base, src);                                           \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        __m128d _high = _mm256_extractf128_pd(src, 1);                         \
        __m128d _low = _mm256_castpd256_pd128(src);                            \
        _mm_storeu_pd(base, _low);                                             \
        _mm_storeu_pd(base + offset, _high);                                   \
    }                                                                          \
}

/**
 * @brief Branchless variant of SCATTER2_256_D to use inside contiguous
 * (out_strides[1] == DATA_STRIDE) branches in kernels, where vector stride
 * `offset` cannot be DATA_STRIDE.
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 2, perm: 0, other: 1}
#define SCATTER2_256_D_STRIDED(base, offset, src)                              \
{                                                                              \
    __m128d _high = _mm256_extractf128_pd(src, 1);                             \
    __m128d _low = _mm256_castpd256_pd128(src);                                \
    _mm_storeu_pd(base, _low);                                                 \
    _mm_storeu_pd(base + offset, _high);                                       \
}

/**
 * @brief H2 reverse-contiguous gather/scatter helpers.
 *
 * H2 (the conjugate/mirror half of a Hermitian-symmetric r2c/c2r spectrum)
 * is naturally indexed in decreasing memory order, so a unit-stride H2 run
 * has `offset == -DATA_STRIDE` instead of `+DATA_STRIDE`. In that case the
 * points are still one contiguous block in memory, just traversed
 * backwards, so each macro below issues a single wide load/store over that
 * block and then permutes lanes to swap the memory (descending) order back
 * to the ascending point order the registers must hold. Any other offset
 * falls back to the existing strided gather/scatter macro with
 * `is_contiguous` forced to 0.
 */

/**
 * @brief Load two H2 32-bit single precision complex points into a 128-bit
 * register, restoring ascending point order.
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 1, other: 0}
#define GATHER2_H2_128_S(base, offset, dest)                                   \
{                                                                              \
    if ((offset) == -DATA_STRIDE)                                              \
    {                                                                          \
        dest = _mm_permute_ps(_mm_loadu_ps((base) + (offset)), 0x4E);          \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        GATHER2_128_S(base, offset, dest, 0);                                  \
    }                                                                          \
}

/**
 * @brief Store two H2 32-bit single precision complex points from a 128-bit
 * register, writing them back in descending (reverse-contiguous) order.
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 1, other: 0}
#define SCATTER2_H2_128_S(base, offset, src)                                   \
{                                                                              \
    if ((offset) == -DATA_STRIDE)                                              \
    {                                                                          \
        _mm_storeu_ps((base) + (offset), _mm_permute_ps(src, 0x4E));           \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        SCATTER2_128_S(base, offset, src, 0);                                  \
    }                                                                          \
}

/**
 * @brief Load one H2 64-bit double precision complex point into a 128-bit
 * register.
 *
 * A single point has no lane order to reverse, so this is simply an alias
 * for the plain (non-H2) 128-bit double offset load.
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 0, other: 0}
#define GATHER_H2_128_D(base, offset, dest)                                    \
    LD_128_OFFSET_D(base, offset, dest, 0)

/**
 * @brief Store one H2 64-bit double precision complex point from a 128-bit
 * register.
 *
 * A single point has no lane order to reverse, so this is simply an alias
 * for the plain (non-H2) 128-bit double offset store.
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 0, other: 0}
#define SCATTER_H2_128_D(base, offset, src)                                    \
    ST_128_OFFSET_D(base, offset, src, 0)

/**
 * @brief Load four H2 32-bit single precision complex points into a 256-bit
 * register, restoring ascending point order.
 *
 * The reverse-contiguous run starts 3 points below `base`, so it is loaded
 * from `base + 3 * offset`. The two 128-bit lanes are swapped
 * (permute2f128) and then the two points within each lane are swapped
 * (permute) to turn the descending memory order into ascending point order.
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 2, other: 0}
#define GATHER4_H2_256_S(base, offset, dest)                                   \
{                                                                              \
    if ((offset) == -DATA_STRIDE)                                              \
    {                                                                          \
        const __m256 _load = _mm256_loadu_ps((base) + 3 * (offset));           \
        const __m256 _tmp = _mm256_permute2f128_ps(                            \
            _load, _load, 0x01);                                               \
        dest = _mm256_permute_ps(_tmp, 0x4E);                                  \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        GATHER4_256_S(base, offset, dest, 0);                                  \
    }                                                                          \
}

/**
 * @brief Store four H2 32-bit single precision complex points from a
 * 256-bit register, writing them back in descending (reverse-contiguous)
 * order starting at `base + 3 * offset`.
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 2, other: 0}
#define SCATTER4_H2_256_S(base, offset, src)                                   \
{                                                                              \
    if ((offset) == -DATA_STRIDE)                                              \
    {                                                                          \
        const __m256 _tmp = _mm256_permute2f128_ps(src, src, 0x01);            \
        _mm256_storeu_ps((base) + 3 * (offset),                                \
                         _mm256_permute_ps(_tmp, 0x4E));                       \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        SCATTER4_256_S(base, offset, src, 0);                                  \
    }                                                                          \
}

/**
 * @brief Load two H2 64-bit double precision complex points into a 256-bit
 * register, restoring ascending point order.
 *
 * The two points form one contiguous 256-bit run; a single lane swap
 * (permute2f128_pd) is enough to turn descending memory order into
 * ascending point order.
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 1, other: 0}
#define GATHER2_H2_256_D(base, offset, dest)                                   \
{                                                                              \
    if ((offset) == -DATA_STRIDE)                                              \
    {                                                                          \
        const __m256d _tmp = _mm256_loadu_pd((base) + (offset));               \
        dest = _mm256_permute2f128_pd(_tmp, _tmp, 0x01);                       \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        GATHER2_256_D(base, offset, dest, 0);                                  \
    }                                                                          \
}

/**
 * @brief Store two H2 64-bit double precision complex points from a 256-bit
 * register, writing them back in descending (reverse-contiguous) order.
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 1, perm: 1, other: 0}
#define SCATTER2_H2_256_D(base, offset, src)                                   \
{                                                                              \
    if ((offset) == -DATA_STRIDE)                                              \
    {                                                                          \
        _mm256_storeu_pd((base) + (offset),                                    \
                         _mm256_permute2f128_pd(src, src, 0x01));              \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        SCATTER2_256_D(base, offset, src, 0);                                  \
    }                                                                          \
}

/**
 * @brief Store two 256-bit double-precision vectors that hold two consecutive
 * complex output points (src0, src1), each laid out as one complex (128-bit)
 * lane per set across NUM_SETS_256_D sets, into contiguous output memory.
 *
 * The two input vectors are point-major (src_k = point k for sets 0..1). When
 * the output points are unit-strided (contiguous), the two points of a given
 * set form a contiguous 256-bit block. This macro performs a 2x2 transpose of
 * the 128-bit complex lanes so that, for each set, the two points are written
 * with a single 256-bit store, avoiding per-point scatter stores.
 *
 * Lane selectors for _mm256_permute2f128_pd (imm8): 0x20 = 0b00100000,
 * 0x31 = 0b00110001.
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 2, perm: 2, other: 0}
#define TRANSPOSE_ST2_256_D(base, offset, src0, src1)                          \
{                                                                              \
    _mm256_storeu_pd(base,                                                     \
        _mm256_permute2f128_pd((src0), (src1), 0x20));                         \
    _mm256_storeu_pd(base + offset,                                            \
        _mm256_permute2f128_pd((src0), (src1), 0x31));                         \
}

/**
 * @brief Store four 256-bit single-precision vectors that hold four consecutive
 * complex output points (src0..src3), each laid out as one complex (64-bit)
 * lane per set across NUM_SETS_256_S sets, into contiguous output memory.
 *
 * The four input vectors are point-major (src_k = point k for sets 0..3). When
 * the output points are unit-strided (contiguous), the four points of a given
 * set form a contiguous 256-bit block. This macro performs a 4x4 transpose of
 * the 64-bit complex lanes so that, for each set, the four points are written
 * with a single 256-bit store, avoiding per-point scatter stores.
 *
 * Lane selectors for _mm256_permute2f128_pd (imm8): 0x20 = 0b00100000,
 * 0x31 = 0b00110001.
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 4, perm: 8, other: 0}
#define TRANSPOSE_ST4_256_S(base, offset, src0, src1, src2, src3)              \
{                                                                              \
    __m256d _s0 = _mm256_castps_pd(src0);                                      \
    __m256d _s1 = _mm256_castps_pd(src1);                                      \
    __m256d _s2 = _mm256_castps_pd(src2);                                      \
    __m256d _s3 = _mm256_castps_pd(src3);                                      \
    __m256d _t0 = _mm256_unpacklo_pd(_s0, _s1);                                \
    __m256d _t1 = _mm256_unpackhi_pd(_s0, _s1);                                \
    __m256d _t2 = _mm256_unpacklo_pd(_s2, _s3);                                \
    __m256d _t3 = _mm256_unpackhi_pd(_s2, _s3);                                \
    _mm256_storeu_ps(base,                                                     \
        _mm256_castpd_ps(_mm256_permute2f128_pd(_t0, _t2, 0x20)));             \
    _mm256_storeu_ps(base + offset,                                            \
        _mm256_castpd_ps(_mm256_permute2f128_pd(_t1, _t3, 0x20)));             \
    _mm256_storeu_ps(base + 2 * offset,                                        \
        _mm256_castpd_ps(_mm256_permute2f128_pd(_t0, _t2, 0x31)));             \
    _mm256_storeu_ps(base + 3 * offset,                                        \
        _mm256_castpd_ps(_mm256_permute2f128_pd(_t1, _t3, 0x31)));             \
}

/**
 * @brief Store two 256-bit single-precision vectors holding two consecutive
 * complex output points (src0, src1), one complex (64-bit) lane per set across
 * NUM_SETS_256_S sets, into contiguous output memory. Performs a 4x2 transpose
 * so that, for each of the four sets, the two points are written with a single
 * 128-bit store. Used for a leftover group of two points.
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 4, perm: 2, other: 2}
#define TRANSPOSE_ST2_256_S(base, offset, src0, src1)                          \
{                                                                              \
    __m256d _s0 = _mm256_castps_pd(src0);                                      \
    __m256d _s1 = _mm256_castps_pd(src1);                                      \
    __m256d _zlo = _mm256_unpacklo_pd(_s0, _s1);                               \
    __m256d _zhi = _mm256_unpackhi_pd(_s0, _s1);                               \
    _mm_storeu_ps(base,                                                        \
        _mm_castpd_ps(_mm256_castpd256_pd128(_zlo)));                          \
    _mm_storeu_ps(base + offset,                                               \
        _mm_castpd_ps(_mm256_castpd256_pd128(_zhi)));                          \
    _mm_storeu_ps(base + 2 * offset,                                           \
        _mm_castpd_ps(_mm256_extractf128_pd(_zlo, 1)));                        \
    _mm_storeu_ps(base + 3 * offset,                                           \
        _mm_castpd_ps(_mm256_extractf128_pd(_zhi, 1)));                        \
}

/**
 * @brief Store two 128-bit single-precision vectors that hold two consecutive
 * complex output points (src0, src1), each laid out as one complex (64-bit)
 * lane per set across NUM_SETS_128_S sets, into contiguous output memory.
 *
 * The two input vectors are point-major (src_k = point k for sets 0..1). When
 * the output points are unit-strided (contiguous), the two points of a given
 * set form a contiguous 128-bit block. This macro performs a 2x2 transpose of
 * the 64-bit complex lanes so that, for each set, the two points are written
 * with a single 128-bit store, avoiding per-point scatter stores.
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 2, perm: 2, other: 0}
#define TRANSPOSE_ST2_128_S(base, offset, src0, src1)                          \
{                                                                              \
    __m128d _s0 = _mm_castps_pd(src0);                                         \
    __m128d _s1 = _mm_castps_pd(src1);                                         \
    _mm_storeu_ps(base,                                                        \
        _mm_castpd_ps(_mm_unpacklo_pd(_s0, _s1)));                             \
    _mm_storeu_ps(base + offset,                                               \
        _mm_castpd_ps(_mm_unpackhi_pd(_s0, _s1)));                             \
}

#ifdef ENABLE_FMA

// FMADD: (a * b) + c
#define FMADD_128_S(a, b, c) _mm_fmadd_ps((a), (b), (c))
#define FMADD_256_S(a, b, c) _mm256_fmadd_ps((a), (b), (c))
#define FMADD_128_D(a, b, c) _mm_fmadd_pd((a), (b), (c))
#define FMADD_256_D(a, b, c) _mm256_fmadd_pd((a), (b), (c))

// FMSUB: (a * b) - c
#define FMSUB_128_S(a, b, c) _mm_fmsub_ps((a), (b), (c))
#define FMSUB_256_S(a, b, c) _mm256_fmsub_ps((a), (b), (c))
#define FMSUB_128_D(a, b, c) _mm_fmsub_pd((a), (b), (c))
#define FMSUB_256_D(a, b, c) _mm256_fmsub_pd((a), (b), (c))

// FNMADD: -(a * b) + c
#define FNMADD_128_S(a, b, c) _mm_fnmadd_ps((a), (b), (c))
#define FNMADD_256_S(a, b, c) _mm256_fnmadd_ps((a), (b), (c))
#define FNMADD_128_D(a, b, c) _mm_fnmadd_pd((a), (b), (c))
#define FNMADD_256_D(a, b, c) _mm256_fnmadd_pd((a), (b), (c))

// FNMSUB: -(a * b) - c
#define FNMSUB_128_S(a, b, c) _mm_fnmsub_ps((a), (b), (c))
#define FNMSUB_256_S(a, b, c) _mm256_fnmsub_ps((a), (b), (c))
#define FNMSUB_128_D(a, b, c) _mm_fnmsub_pd((a), (b), (c))
#define FNMSUB_256_D(a, b, c) _mm256_fnmsub_pd((a), (b), (c))

// FMADDSUB: (a * b) +(odd)/-(even) c
#define FMADDSUB_128_S(a, b, c) _mm_fmaddsub_ps((a), (b), (c))
#define FMADDSUB_256_S(a, b, c) _mm256_fmaddsub_ps((a), (b), (c))
#define FMADDSUB_128_D(a, b, c) _mm_fmaddsub_pd((a), (b), (c))
#define FMADDSUB_256_D(a, b, c) _mm256_fmaddsub_pd((a), (b), (c))

// FMSUBADD: (a * b) -(odd)/+(even) c
#define FMSUBADD_128_S(a, b, c) _mm_fmsubadd_ps((a), (b), (c))
#define FMSUBADD_256_S(a, b, c) _mm256_fmsubadd_ps((a), (b), (c))
#define FMSUBADD_128_D(a, b, c) _mm_fmsubadd_pd((a), (b), (c))
#define FMSUBADD_256_D(a, b, c) _mm256_fmsubadd_pd((a), (b), (c))

#else

// FMADD: (a * b) + c
#define FMADD_128_S(a, b, c) _mm_add_ps(_mm_mul_ps((a), (b)), (c))
#define FMADD_256_S(a, b, c) _mm256_add_ps(_mm256_mul_ps((a), (b)), (c))
#define FMADD_128_D(a, b, c) _mm_add_pd(_mm_mul_pd((a), (b)), (c))
#define FMADD_256_D(a, b, c) _mm256_add_pd(_mm256_mul_pd((a), (b)), (c))

// FMSUB: (a * b) - c
#define FMSUB_128_S(a, b, c) _mm_sub_ps(_mm_mul_ps((a), (b)), (c))
#define FMSUB_256_S(a, b, c) _mm256_sub_ps(_mm256_mul_ps((a), (b)), (c))
#define FMSUB_128_D(a, b, c) _mm_sub_pd(_mm_mul_pd((a), (b)), (c))
#define FMSUB_256_D(a, b, c) _mm256_sub_pd(_mm256_mul_pd((a), (b)), (c))

// FNMADD: -(a * b) + c
#define FNMADD_128_S(a, b, c)                                                  \
    _mm_add_ps(                                                                \
        _mm_xor_ps(                                                            \
            _mm_mul_ps((a), (b)),                                              \
            _neg_128_f[1].s                                                    \
        ),                                                                     \
        (c)                                                                    \
    )

#define FNMADD_256_S(a, b, c)                                                  \
    _mm256_add_ps(                                                             \
        _mm256_xor_ps(                                                         \
            _mm256_mul_ps((a), (b)),                                           \
            _neg_256_f[1].s                                                    \
        ),                                                                     \
        (c)                                                                    \
    )

#define FNMADD_128_D(a, b, c)                                                  \
    _mm_add_pd(                                                                \
        _mm_xor_pd(                                                            \
            _mm_mul_pd((a), (b)),                                              \
            _neg_128_d[1].d                                                    \
        ),                                                                     \
        (c)                                                                    \
    )

#define FNMADD_256_D(a, b, c)                                                  \
    _mm256_add_pd(                                                             \
        _mm256_xor_pd(                                                         \
            _mm256_mul_pd((a), (b)),                                           \
            _neg_256_d[1].d                                                    \
        ),                                                                     \
        (c)                                                                    \
    )

// FNMSUB: -(a * b) - c
#define FNMSUB_128_S(a, b, c)                                                  \
    _mm_sub_ps(                                                                \
        _mm_xor_ps(                                                            \
            _mm_mul_ps((a), (b)),                                              \
            _neg_128_f[1].s                                                    \
        ),                                                                     \
        (c)                                                                    \
    )

#define FNMSUB_256_S(a, b, c)                                                  \
    _mm256_sub_ps(                                                             \
        _mm256_xor_ps(                                                         \
            _mm256_mul_ps((a), (b)),                                           \
            _neg_256_f[1].s                                                    \
        ),                                                                     \
        (c)                                                                    \
    )

#define FNMSUB_128_D(a, b, c)                                                  \
    _mm_sub_pd(                                                                \
        _mm_xor_pd(                                                            \
            _mm_mul_pd((a), (b)),                                              \
            _neg_128_d[1].d                                                    \
        ),                                                                     \
        (c)                                                                    \
    )

#define FNMSUB_256_D(a, b, c)                                                  \
    _mm256_sub_pd(                                                             \
        _mm256_xor_pd(                                                         \
            _mm256_mul_pd((a), (b)),                                           \
            _neg_256_d[1].d                                                    \
        ),                                                                     \
        (c)                                                                    \
    )

// FMADDSUB: (a * b) +(odd)/-(even) c
#define FMADDSUB_128_S(a, b, c)                                                \
    _mm_addsub_ps(_mm_mul_ps((a), (b)), (c))

#define FMADDSUB_256_S(a, b, c)                                                \
    _mm256_addsub_ps(_mm256_mul_ps((a), (b)), (c))

#define FMADDSUB_128_D(a, b, c)                                                \
    _mm_addsub_pd(_mm_mul_pd((a), (b)), (c))

#define FMADDSUB_256_D(a, b, c)                                                \
    _mm256_addsub_pd(_mm256_mul_pd((a), (b)), (c))

// FMSUBADD: (a * b) -(odd)/+(even) c
#define FMSUBADD_128_S(a, b, c)                                                \
    _mm_addsub_ps(                                                             \
        _mm_mul_ps((a), (b)),                                                  \
        _mm_xor_ps((c), _neg_128_f[1].s)                                       \
    )

#define FMSUBADD_256_S(a, b, c)                                                \
    _mm256_addsub_ps(                                                          \
        _mm256_mul_ps((a), (b)),                                               \
        _mm256_xor_ps((c), _neg_256_f[1].s)                                    \
    )

#define FMSUBADD_128_D(a, b, c)                                                \
    _mm_addsub_pd(                                                             \
        _mm_mul_pd((a), (b)),                                                  \
        _mm_xor_pd((c), _neg_128_d[1].d)                                       \
    )

#define FMSUBADD_256_D(a, b, c)                                                \
    _mm256_addsub_pd(                                                          \
        _mm256_mul_pd((a), (b)),                                               \
        _mm256_xor_pd((c), _neg_256_d[1].d)                                    \
    )

#endif // ENABLE_FMA

// Cost: {fma: 0, mul: 2, add: 1, move: 3, perm: 2, other: 0}
#define ITW_GATHER_128_D(gbase, starr, stidx, offset, gdest, twbuf,            \
                         load_multi_cols /* unused */, is_contiguous)          \
{                                                                              \
    const FFTZ_UINTP addr = DATA_STRIDE * (stidx - 1);                         \
    const __m128d cd = _mm_load_pd((twbuf) + addr);                            \
    const __m128d bb = _mm_loaddup_pd((gbase) + starr[(stidx)]);               \
    const __m128d aa = _mm_loaddup_pd((gbase) + starr[(stidx)] + 1);           \
    const __m128d ca_da = _mm_mul_pd(cd, aa);                                  \
    const __m128d cb_db = _mm_mul_pd(cd, bb);                                  \
    const __m128d db_cb = SWAP_RI_128_D(cb_db);                                \
    gdest = SWAP_RI_128_D(_mm_addsub_pd(ca_da, db_cb));                        \
}

// Cost: {fma: 0, mul: 2, add: 1, move: 3, perm: 1, other: 0}
#define TW_GATHER_128_D(gbase, starr, stidx, offset, gdest, twbuf,             \
                        load_multi_cols /* unused */, is_contiguous)           \
{                                                                              \
    const FFTZ_UINTP addr = DATA_STRIDE * (stidx - 1);                         \
    const __m128d cd = _mm_load_pd((twbuf) + addr);                            \
    const __m128d aa = _mm_loaddup_pd((gbase) + starr[(stidx)]);               \
    const __m128d bb = _mm_loaddup_pd((gbase) + starr[(stidx)] + 1);           \
    const __m128d ca_da = _mm_mul_pd(cd, aa);                                  \
    const __m128d cb_db = _mm_mul_pd(cd, bb);                                  \
    const __m128d db_cb = SWAP_RI_128_D(cb_db);                                \
    gdest = _mm_addsub_pd(ca_da, db_cb);                                       \
}

// Cost: {fma: 0, mul: 2, add: 1, move: 3, perm: 4, other: 1}
#define ITW_GATHER_256_D(gbase, starr, stidx, offset, gdest, twbuf,            \
                         load_multi_cols, is_contiguous)                       \
{                                                                              \
    __m256d twv;                                                               \
    if ((load_multi_cols))                                                     \
    {                                                                          \
        const FFTZ_UINTP addr =                                                \
            DATA_STRIDE * (stidx - 1) * NUM_SETS_256_D;                        \
        twv = _mm256_loadu_pd((twbuf) + addr);                                 \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        const FFTZ_UINTP addr = DATA_STRIDE * (stidx - 1);                     \
        twv = _mm256_broadcast_pd((__m128d *)((twbuf) + addr));                \
    }                                                                          \
    __m256d tmp_in;                                                            \
    GATHER2_256_D((gbase) + starr[stidx], offset, tmp_in, is_contiguous);      \
    const __m256d tmp_0 = _mm256_mul_pd(tmp_in, twv);                          \
    const __m256d tmp_1 = _mm256_mul_pd(SWAP_RI_256_D(tmp_in), twv);           \
    const __m256d lo_1 = _mm256_unpacklo_pd(tmp_1, tmp_0);                     \
    const __m256d hi_1 = _mm256_unpackhi_pd(tmp_1, tmp_0);                     \
    gdest = SWAP_RI_256_D(_mm256_addsub_pd(lo_1, hi_1));                       \
}

// Cost: {fma: 0, mul: 2, add: 1, move: 3, perm: 3, other: 1}
#define TW_GATHER_256_D(gbase, starr, stidx, offset, gdest, twbuf,             \
                        load_multi_cols, is_contiguous)                        \
{                                                                              \
    __m256d twv;                                                               \
    if ((load_multi_cols))                                                     \
    {                                                                          \
        const FFTZ_UINTP addr =                                                \
            DATA_STRIDE * (stidx - 1) * NUM_SETS_256_D;                        \
        twv = _mm256_loadu_pd((twbuf) + addr);                                 \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        const FFTZ_UINTP addr = DATA_STRIDE * (stidx - 1);                     \
        twv = _mm256_broadcast_pd((__m128d *)((twbuf) + addr));                \
    }                                                                          \
    __m256d tmp_in;                                                            \
    GATHER2_256_D((gbase) + starr[stidx], offset, tmp_in, is_contiguous);      \
    const __m256d tmp_0 = _mm256_mul_pd(tmp_in, twv);                          \
    const __m256d tmp_1 = _mm256_mul_pd(SWAP_RI_256_D(tmp_in), twv);           \
    const __m256d lo_1 = _mm256_unpacklo_pd(tmp_0, tmp_1);                     \
    const __m256d hi_1 = _mm256_unpackhi_pd(tmp_0, tmp_1);                     \
    gdest = _mm256_addsub_pd(lo_1, hi_1);                                      \
}

// Cost: {fma: 0, mul: 2, add: 1, move: 5, perm: 7, other: 1}
#define ITW_GATHER_256_S(gbase, starr, stidx, offset, gdest, twbuf,            \
                         load_multi_cols, is_contiguous)                       \
{                                                                              \
    __m256 twv;                                                                \
    if ((load_multi_cols))                                                     \
    {                                                                          \
        const FFTZ_UINTP addr =                                                \
            DATA_STRIDE * (stidx - 1) * NUM_SETS_256_S;                        \
        twv = _mm256_loadu_ps((twbuf) + addr);                                 \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        const FFTZ_UINTP addr = DATA_STRIDE * (stidx - 1);                     \
        twv = (__m256)(_mm256_broadcast_sd((FFTZ_DOUBLE *)((twbuf) + addr)));  \
    }                                                                          \
    __m256 tmp_in;                                                             \
    GATHER4_256_S((gbase) + starr[stidx], offset, tmp_in, is_contiguous);      \
    __m256 tmp_0 = _mm256_mul_ps(tmp_in, twv);                                 \
    __m256 tmp_1 = _mm256_mul_ps(SWAP_RI_256_S(tmp_in), twv);                  \
    tmp_0 = _mm256_permute_ps(tmp_0, 0xD8 /*0b11011000*/);                     \
    tmp_1 = _mm256_permute_ps(tmp_1, 0xD8 /*0b11011000*/);                     \
    const __m256 lo_1 = _mm256_unpacklo_ps(tmp_1, tmp_0);                      \
    const __m256 hi_1 = _mm256_unpackhi_ps(tmp_1, tmp_0);                      \
    gdest = SWAP_RI_256_S(_mm256_addsub_ps(lo_1, hi_1));                       \
}

// Cost: {fma: 0, mul: 2, add: 1, move: 5, perm: 6, other: 1}
#define TW_GATHER_256_S(gbase, starr, stidx, offset, gdest, twbuf,             \
                        load_multi_cols, is_contiguous)                        \
{                                                                              \
    __m256 twv;                                                                \
    if ((load_multi_cols))                                                     \
    {                                                                          \
        const FFTZ_UINTP addr =                                                \
            DATA_STRIDE * (stidx - 1) * NUM_SETS_256_S;                        \
        twv = _mm256_loadu_ps((twbuf) + addr);                                 \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        const FFTZ_UINTP addr = DATA_STRIDE * (stidx - 1);                     \
        twv = (__m256)(_mm256_broadcast_sd((FFTZ_DOUBLE *)((twbuf) + addr)));  \
    }                                                                          \
    __m256 tmp_in;                                                             \
    GATHER4_256_S((gbase) + starr[stidx], offset, tmp_in, is_contiguous);      \
    __m256 tmp_0 = _mm256_mul_ps(tmp_in, twv);                                 \
    __m256 tmp_1 = _mm256_mul_ps(SWAP_RI_256_S(tmp_in), twv);                  \
    tmp_0 = _mm256_permute_ps(tmp_0, 0xD8 /*0b11011000*/);                     \
    tmp_1 = _mm256_permute_ps(tmp_1, 0xD8 /*0b11011000*/);                     \
    const __m256 lo_1 = _mm256_unpacklo_ps(tmp_0, tmp_1);                      \
    const __m256 hi_1 = _mm256_unpackhi_ps(tmp_0, tmp_1);                      \
    gdest = _mm256_addsub_ps(lo_1, hi_1);                                      \
}

// Cost: {fma: 0, mul: 2, add: 1, move: 3, perm: 6, other: 0}
#define ITW_GATHER_128_S(gbase, starr, stidx, offset, gdest, twbuf,            \
                         load_multi_cols, is_contiguous)                       \
{                                                                              \
    __m128 twv;                                                                \
    if ((load_multi_cols))                                                     \
    {                                                                          \
        const FFTZ_UINTP addr =                                                \
            DATA_STRIDE * (stidx - 1) * NUM_SETS_128_S;                        \
        twv = _mm_loadu_ps((twbuf) + addr);                                    \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        const FFTZ_UINTP addr = DATA_STRIDE * (stidx - 1);                     \
        twv = (__m128)(_mm_loaddup_pd((FFTZ_DOUBLE *)((twbuf) + addr)));       \
    }                                                                          \
    __m128 tmp_in;                                                             \
    GATHER2_128_S((gbase) + starr[(stidx)], offset, tmp_in,                    \
        is_contiguous);                                                        \
    __m128 tmp_0 = _mm_mul_ps(tmp_in, twv);                                    \
    __m128 tmp_1 = _mm_mul_ps(SWAP_RI_128_S(tmp_in), twv);                     \
    tmp_0 = _mm_permute_ps(tmp_0, 0xD8 /*0b11011000*/);                        \
    tmp_1 = _mm_permute_ps(tmp_1, 0xD8 /*0b11011000*/);                        \
    const __m128 lo_1 = _mm_unpacklo_ps(tmp_1, tmp_0);                         \
    const __m128 hi_1 = _mm_unpackhi_ps(tmp_1, tmp_0);                         \
    gdest = SWAP_RI_128_S(_mm_addsub_ps(lo_1, hi_1));                          \
}

// Cost: {fma: 0, mul: 2, add: 1, move: 3, perm: 5, other: 0}
#define TW_GATHER_128_S(gbase, starr, stidx, offset, gdest, twbuf,             \
                        load_multi_cols, is_contiguous)                        \
{                                                                              \
    __m128 twv;                                                                \
    if ((load_multi_cols))                                                     \
    {                                                                          \
        const FFTZ_UINTP addr =                                                \
            DATA_STRIDE * (stidx - 1) * NUM_SETS_128_S;                        \
        twv = _mm_loadu_ps((twbuf) + addr);                                    \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        const FFTZ_UINTP addr = DATA_STRIDE * (stidx - 1);                     \
        twv = (__m128)(_mm_loaddup_pd((FFTZ_DOUBLE *)((twbuf) + addr)));       \
    }                                                                          \
    __m128 tmp_in;                                                             \
    GATHER2_128_S((gbase) + starr[(stidx)], offset, tmp_in,                    \
        is_contiguous);                                                        \
    __m128 tmp_0 = _mm_mul_ps(tmp_in, twv);                                    \
    __m128 tmp_1 = _mm_mul_ps(SWAP_RI_128_S(tmp_in), twv);                     \
    tmp_0 = _mm_permute_ps(tmp_0, 0xD8 /*0b11011000*/);                        \
    tmp_1 = _mm_permute_ps(tmp_1, 0xD8 /*0b11011000*/);                        \
    const __m128 lo_1 = _mm_unpacklo_ps(tmp_0, tmp_1);                         \
    const __m128 hi_1 = _mm_unpackhi_ps(tmp_0, tmp_1);                         \
    gdest = _mm_addsub_ps(lo_1, hi_1);                                         \
}

// Cost: {fma: 0, mul: 2, add: 1, move: 2, perm: 6, other: 0}
#define ITW_GATHER_LOW_128_S(gbase, starr, stidx, gdest, twbuf,                \
                             load_multi_cols /* unused */, is_contiguous)      \
{                                                                              \
    (FFTZ_VOID)(is_contiguous);                                                \
    const FFTZ_UINTP addr = DATA_STRIDE * (stidx - 1);                         \
    __m128 tmp_in, twv;                                                        \
    LD_LOW_128_S((twbuf) + addr, twv);                                         \
    LD_LOW_128_S((gbase) + starr[(stidx)], tmp_in);                            \
    __m128 tmp_0 = _mm_mul_ps(tmp_in, twv);                                    \
    __m128 tmp_1 = _mm_mul_ps(SWAP_RI_128_S(tmp_in), twv);                     \
    tmp_0 = _mm_permute_ps(tmp_0, 0xD8 /*0b11011000*/);                        \
    tmp_1 = _mm_permute_ps(tmp_1, 0xD8 /*0b11011000*/);                        \
    const __m128 lo_1 = _mm_unpacklo_ps(tmp_1, tmp_0);                         \
    const __m128 hi_1 = _mm_unpackhi_ps(tmp_1, tmp_0);                         \
    gdest = SWAP_RI_128_S(_mm_addsub_ps(lo_1, hi_1));                          \
}

// Cost: {fma: 0, mul: 2, add: 1, move: 2, perm: 5, other: 0}
#define TW_GATHER_LOW_128_S(gbase, starr, stidx, gdest, twbuf,                 \
                            load_multi_cols /* unused */, is_contiguous)       \
{                                                                              \
    (FFTZ_VOID)(is_contiguous);                                                \
    const FFTZ_UINTP addr = DATA_STRIDE * (stidx - 1);                         \
    __m128 tmp_in, twv;                                                        \
    LD_LOW_128_S((twbuf) + addr, twv);                                         \
    LD_LOW_128_S((gbase) + starr[(stidx)], tmp_in);                            \
    __m128 tmp_0 = _mm_mul_ps(tmp_in, twv);                                    \
    __m128 tmp_1 = _mm_mul_ps(SWAP_RI_128_S(tmp_in), twv);                     \
    tmp_0 = _mm_permute_ps(tmp_0, 0xD8 /*0b11011000*/);                        \
    tmp_1 = _mm_permute_ps(tmp_1, 0xD8 /*0b11011000*/);                        \
    const __m128 lo_1 = _mm_unpacklo_ps(tmp_0, tmp_1);                         \
    const __m128 hi_1 = _mm_unpackhi_ps(tmp_0, tmp_1);                         \
    gdest = _mm_addsub_ps(lo_1, hi_1);                                         \
}

/**
 * @brief interchanges the real and imaginary values in a 128 bit register
 * for single precision floating point.
 * Operation : 1 PERM(shuffle)
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 1, other: 0}
#define SWAP_RI_128_S(val) _mm_shuffle_ps(val, val, 177)

/**
 * @brief interchanges the real and imaginary values in a 128 bit register
 * for double precision floating point.
 * Operation : 1 PERM(shuffle)
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 1, other: 0}
#define SWAP_RI_128_D(val) _mm_shuffle_pd(val, val, 1)

/**
 * @brief interchanges the real and imaginary values in a 256 bit register
 * for single precision floating point using the control value in the last
 * integer argument(b 10 11 00 01)
 * Operation : 1 PERM
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 1, other: 0}
#define SWAP_RI_256_S(val) _mm256_permute_ps(val, 177)

/**
 * @brief interchanges the real and imaginary values in a 256 bit register
 * for double precision floating point using the control value in the last
 * integer argument(b 01 01)
 * Operation : 1 PERM
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 1, other: 0}
#define SWAP_RI_256_D(val) _mm256_permute_pd(val, 5)

/**
 * @brief Broadcasts real parts of complex numbers in a 128-bit register.
 * Shuffle pattern 0xA0 (10100000b) selects elements [0,0,2,2].
 * Operation : 1 PERM(shuffle)
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 1, other: 0}
#define BROADCAST_RE_128_S(val) _mm_shuffle_ps(val, val, 0xA0) // 10100000b: [0,0,2,2]

/**
 * @brief Broadcasts imaginary parts of complex numbers in a 128-bit register.
 * Shuffle pattern 0xF5 (11110101b) selects elements [1,1,3,3].
 * Operation : 1 PERM(shuffle)
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 1, other: 0}
#define BROADCAST_IM_128_S(val) _mm_shuffle_ps(val, val, 0xF5) // 11110101b: [1,1,3,3]

/**
 * @brief Broadcasts the real part of a complex number in a 128-bit register
 * for double precision floating point.
 * Shuffle pattern 0x0 (00b) selects the low element of each operand: [0,0].
 * Operation : 1 PERM(shuffle)
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 1, other: 0}
#define BROADCAST_RE_128_D(val) _mm_shuffle_pd(val, val, 0x0) // 00b: [0,0]

/**
 * @brief Broadcasts the imaginary part of a complex number in a 128-bit
 * register for double precision floating point. Shuffle pattern 0x3 (11b)
 * selects the high element of each operand: [1,1]. Operation : 1 PERM(shuffle)
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 1, other: 0}
#define BROADCAST_IM_128_D(val) _mm_shuffle_pd(val, val, 0x3) // 11b: [1,1]

/**
 * @brief Broadcasts real parts of complex numbers in a 256-bit register.
 * Shuffle pattern 0xA0 (10100000b) selects elements [0,0,2,2].
 * Operation : 1 PERM(shuffle)
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 1, other: 0}
#define BROADCAST_RE_256_S(val) _mm256_shuffle_ps(val, val, 0xA0) // 10100000b: [0,0,2,2]

/**
 * @brief Broadcasts imaginary parts of complex numbers in a 256-bit register.
 * Shuffle pattern 0xF5 (11110101b) selects elements [1,1,3,3].
 * Operation : 1 PERM(shuffle)
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 1, other: 0}
#define BROADCAST_IM_256_S(val) _mm256_shuffle_ps(val, val, 0xF5) // 11110101b: [1,1,3,3]

/**
 * @brief Broadcasts real parts of complex numbers in a 256-bit register
 * for double precision floating point.
 * Shuffle pattern 0x0 (0000b) selects the low element of each 128-bit lane:
 * [0,0,2,2]. Operation : 1 PERM(shuffle)
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 1, other: 0}
#define BROADCAST_RE_256_D(val) _mm256_shuffle_pd(val, val, 0x0) // 0000b: [0,0,2,2]

/**
 * @brief Broadcasts imaginary parts of complex numbers in a 256-bit register
 * for double precision floating point.
 * Shuffle pattern 0xF (1111b) selects the high element of each 128-bit lane:
 * [1,1,3,3]. Operation : 1 PERM(shuffle)
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 1, other: 0}
#define BROADCAST_IM_256_D(val) _mm256_shuffle_pd(val, val, 0xF) // 1111b: [1,1,3,3]

/**
 * @brief implies the number of sets that can be processed in parallel.
 * Computed using Register width /(2* sizeof(floating point)
 */
#define NUM_SETS_128_S 2
#define NUM_SETS_128_D 1
#define NUM_SETS_256_S 4
#define NUM_SETS_256_D 2

/**
 * @brief prepare -0.0 for complex conjugate.
 */
union data_union_128
{
    unsigned u[4];
    __m128 s;
    __m128d d;
};
union data_union_256
{
    unsigned u[8];
    __m256 s;
    __m256d d;
};

static const union data_union_128 _conj_128_f = {
    {0x00000000, 0x80000000, 0x00000000, 0x80000000}};
static const union data_union_128 _conj_128_d = {
    {0x00000000, 0x00000000, 0x00000000, 0x80000000}};

static const union data_union_128 _neg_128_f[2] = {
    {.u = {0x00000000, 0x00000000, 0x00000000, 0x00000000}},
    {.u = {0x80000000, 0x80000000, 0x80000000, 0x80000000}}};
static const union data_union_128 _neg_128_d[2] = {
    {.u = {0x00000000, 0x00000000, 0x00000000, 0x00000000}},
    {.u = {0x00000000, 0x80000000, 0x00000000, 0x80000000}}};

static const union data_union_256 _conj_256_f = {
    {0x00000000, 0x80000000, 0x00000000, 0x80000000, 0x00000000, 0x80000000,
     0x00000000, 0x80000000}};
static const union data_union_256 _conj_256_d = {
    {0x00000000, 0x00000000, 0x00000000, 0x80000000, 0x00000000, 0x00000000,
     0x00000000, 0x80000000}};

static const union data_union_256 _neg_256_f[2] = {
    {.u = {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
           0x00000000, 0x00000000, 0x00000000}},
    {.u = {0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000,
           0x80000000, 0x80000000, 0x80000000}}};
static const union data_union_256 _neg_256_d[2] = {
    {.u = {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
           0x00000000, 0x00000000, 0x00000000}},
    {.u = {0x00000000, 0x80000000, 0x00000000, 0x80000000, 0x00000000,
           0x80000000, 0x00000000, 0x80000000}}};

/**
 * @brief take conjugate of the complex number A+Bi => A-Bi
 * Operation : 1 OTHERS(xor)
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 0, other: 1}
#define CONJ_128_S(x) _mm_xor_ps(_conj_128_f.s, x)
// Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 0, other: 1}
#define CONJ_128_D(x) _mm_xor_pd(_conj_128_d.d, x)
// Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 0, other: 1}
#define CONJ_256_S(x) _mm256_xor_ps(_conj_256_f.s, x)
// Cost: {fma: 0, mul: 0, add: 0, move: 0, perm: 0, other: 1}
#define CONJ_256_D(x) _mm256_xor_pd(_conj_256_d.d, x)

/**
 * @brief alternatively performs addition & subtraction in a 128 bit register
 * for single precision floating point.
 * Operation : 2 PERM, 1 ADD
 */
// Cost: {fma: 0, mul: 0, add: 1, move: 0, perm: 2, other: 0}
#define SUBADD_SWAPA_128_S(a, b, c)                                            \
{                                                                              \
    a = SWAP_RI_128_S(a);                                                      \
    c = SWAP_RI_128_S(_mm_addsub_ps(a, b));                                    \
}

/**
 * @brief alternatively performs addition & subtraction in a 128 bit register
 * for double precision floating point.
 * Operation : 2 PERM, 1 ADD
 */
// Cost: {fma: 0, mul: 0, add: 1, move: 0, perm: 2, other: 0}
#define SUBADD_SWAPA_128_D(a, b, c)                                            \
{                                                                              \
    a = SWAP_RI_128_D(a);                                                      \
    c = SWAP_RI_128_D(_mm_addsub_pd(a, b));                                    \
}
/**
 * @brief alternatively performs addition & subtraction in a 256 bit register
 * for single precision floating point.
 * Operation : 2 PERM, 1 ADD
 */
// Cost: {fma: 0, mul: 0, add: 1, move: 0, perm: 2, other: 0}
#define SUBADD_SWAPA_256_S(a, b, c)                                            \
{                                                                              \
    a = SWAP_RI_256_S(a);                                                      \
    c = SWAP_RI_256_S(_mm256_addsub_ps(a, b));                                 \
}

/**
 * @brief alternatively performs addition & subtraction in a 256 bit register
 * for double precision floating point.
 * Operation : 2 PERM, 1 ADD
 */
// Cost: {fma: 0, mul: 0, add: 1, move: 0, perm: 2, other: 0}
#define SUBADD_SWAPA_256_D(a, b, c)                                            \
{                                                                              \
    a = SWAP_RI_256_D(a);                                                      \
    c = SWAP_RI_256_D(_mm256_addsub_pd(a, b));                                 \
}

// Cost: {fma: 0, mul: 2, add: 1, move: 2, perm: 4, other: 1}
#define ITW_PRELOADED_GATHER_256_D(gbase, starr, stidx, offset, gdest, twv,    \
                                   is_contiguous)                              \
{                                                                              \
    __m256d tmp_in;                                                            \
    GATHER2_256_D((gbase) + starr[(stidx)], offset, tmp_in, is_contiguous);    \
    const __m256d tmp_0 = _mm256_mul_pd(tmp_in, twv);                          \
    const __m256d tmp_1 = _mm256_mul_pd(SWAP_RI_256_D(tmp_in), twv);           \
    const __m256d lo_1 = _mm256_unpacklo_pd(tmp_1, tmp_0);                     \
    const __m256d hi_1 = _mm256_unpackhi_pd(tmp_1, tmp_0);                     \
    gdest = SWAP_RI_256_D(_mm256_addsub_pd(lo_1, hi_1));                       \
}

// Cost: {fma: 0, mul: 2, add: 1, move: 2, perm: 3, other: 1}
#define TW_PRELOADED_GATHER_256_D(gbase, starr, stidx, offset, gdest, twv,     \
                                  is_contiguous)                               \
{                                                                              \
    __m256d tmp_in;                                                            \
    GATHER2_256_D((gbase) + starr[(stidx)], offset, tmp_in, is_contiguous);    \
    const __m256d tmp_0 = _mm256_mul_pd(tmp_in, twv);                          \
    const __m256d tmp_1 = _mm256_mul_pd(SWAP_RI_256_D(tmp_in), twv);           \
    const __m256d lo_1 = _mm256_unpacklo_pd(tmp_0, tmp_1);                     \
    const __m256d hi_1 = _mm256_unpackhi_pd(tmp_0, tmp_1);                     \
    gdest = _mm256_addsub_pd(lo_1, hi_1);                                      \
}

// Cost: {fma: 0, mul: 2, add: 1, move: 1, perm: 2, other: 0}
#define ITW_PRELOADED_GATHER_128_D(gbase, starr, stidx, offset, gdest, twv,    \
                                   is_contiguous)                              \
{                                                                              \
    (FFTZ_VOID)is_contiguous;                                                  \
    const __m128d bb = _mm_loaddup_pd((gbase) + starr[(stidx)]);               \
    const __m128d aa = _mm_loaddup_pd((gbase) + starr[(stidx)] + 1);           \
    const __m128d ca_da = _mm_mul_pd(twv, aa);                                 \
    const __m128d cb_db = _mm_mul_pd(twv, bb);                                 \
    const __m128d db_cb = SWAP_RI_128_D(cb_db);                                \
    gdest = SWAP_RI_128_D(_mm_addsub_pd(ca_da, db_cb));                        \
}

// Cost: {fma: 0, mul: 2, add: 1, move: 1, perm: 1, other: 0}
#define TW_PRELOADED_GATHER_128_D(gbase, starr, stidx, offset, gdest, twv,     \
                                  is_contiguous)                               \
{                                                                              \
    (FFTZ_VOID)is_contiguous;                                                  \
    const __m128d aa = _mm_loaddup_pd((gbase) + starr[(stidx)]);               \
    const __m128d bb = _mm_loaddup_pd((gbase) + starr[(stidx)] + 1);           \
    const __m128d ca_da = _mm_mul_pd(twv, aa);                                 \
    const __m128d cb_db = _mm_mul_pd(twv, bb);                                 \
    const __m128d db_cb = SWAP_RI_128_D(cb_db);                                \
    gdest = _mm_addsub_pd(ca_da, db_cb);                                       \
}

// Cost: {fma: 0, mul: 2, add: 1, move: 2, perm: 4, other: 1}
#define ITW_PRELOADED_GATHER_256_D_V(gbase, stride, offset, gdest, twv,        \
                                     is_contiguous)                            \
{                                                                              \
    __m256d tmp_in;                                                            \
    GATHER2_256_D((gbase) + (stride), offset, tmp_in, is_contiguous);          \
    const __m256d tmp_0 = _mm256_mul_pd(tmp_in, twv);                          \
    const __m256d tmp_1 = _mm256_mul_pd(SWAP_RI_256_D(tmp_in), twv);           \
    const __m256d lo_1 = _mm256_unpacklo_pd(tmp_1, tmp_0);                     \
    const __m256d hi_1 = _mm256_unpackhi_pd(tmp_1, tmp_0);                     \
    gdest = SWAP_RI_256_D(_mm256_addsub_pd(lo_1, hi_1));                       \
}

// Cost: {fma: 0, mul: 2, add: 1, move: 2, perm: 3, other: 1}
#define TW_PRELOADED_GATHER_256_D_V(gbase, stride, offset, gdest, twv,         \
                                    is_contiguous)                             \
{                                                                              \
    __m256d tmp_in;                                                            \
    GATHER2_256_D((gbase) + (stride), offset, tmp_in, is_contiguous);          \
    const __m256d tmp_0 = _mm256_mul_pd(tmp_in, twv);                          \
    const __m256d tmp_1 = _mm256_mul_pd(SWAP_RI_256_D(tmp_in), twv);           \
    const __m256d lo_1 = _mm256_unpacklo_pd(tmp_0, tmp_1);                     \
    const __m256d hi_1 = _mm256_unpackhi_pd(tmp_0, tmp_1);                     \
    gdest = _mm256_addsub_pd(lo_1, hi_1);                                      \
}

// Cost: {fma: 0, mul: 2, add: 1, move: 1, perm: 2, other: 0}
#define ITW_PRELOADED_GATHER_128_D_V(gbase, stride, offset, gdest, twv,        \
                                     is_contiguous)                            \
{                                                                              \
    (FFTZ_VOID)is_contiguous;                                                  \
    const __m128d bb = _mm_loaddup_pd((gbase) + (stride));                     \
    const __m128d aa = _mm_loaddup_pd((gbase) + (stride) + 1);                 \
    const __m128d ca_da = _mm_mul_pd(twv, aa);                                 \
    const __m128d cb_db = _mm_mul_pd(twv, bb);                                 \
    const __m128d db_cb = SWAP_RI_128_D(cb_db);                                \
    gdest = SWAP_RI_128_D(_mm_addsub_pd(ca_da, db_cb));                        \
}

// Cost: {fma: 0, mul: 2, add: 1, move: 1, perm: 1, other: 0}
#define TW_PRELOADED_GATHER_128_D_V(gbase, stride, offset, gdest, twv,         \
                                    is_contiguous)                             \
{                                                                              \
    (FFTZ_VOID)is_contiguous;                                                  \
    const __m128d aa = _mm_loaddup_pd((gbase) + (stride));                     \
    const __m128d bb = _mm_loaddup_pd((gbase) + (stride) + 1);                 \
    const __m128d ca_da = _mm_mul_pd(twv, aa);                                 \
    const __m128d cb_db = _mm_mul_pd(twv, bb);                                 \
    const __m128d db_cb = SWAP_RI_128_D(cb_db);                                \
    gdest = _mm_addsub_pd(ca_da, db_cb);                                       \
}

// Cost: {fma: 0, mul: 2, add: 1, move: 2, perm: 3, other: 1}
#define TW_PRELOADED_SCATTER_256_D(sbase, stride, offset, ssrc, twv,           \
                                   is_contiguous)                              \
{                                                                              \
    const __m256d tmp_0 = _mm256_mul_pd(ssrc, twv);                            \
    const __m256d tmp_1 = _mm256_mul_pd(SWAP_RI_256_D(ssrc), twv);             \
    const __m256d lo_1 = _mm256_unpacklo_pd(tmp_0, tmp_1);                     \
    const __m256d hi_1 = _mm256_unpackhi_pd(tmp_0, tmp_1);                     \
    __m256d _result = _mm256_addsub_pd(lo_1, hi_1);                            \
    SCATTER2_256_D((sbase) + (stride), offset, _result, is_contiguous);        \
}

// Cost: {fma: 0, mul: 2, add: 1, move: 1, perm: 1, other: 0}
#define TW_PRELOADED_SCATTER_128_D(sbase, stride, offset, ssrc, twv,           \
                                   is_contiguous)                              \
{                                                                              \
    (FFTZ_VOID)is_contiguous;                                                  \
    const __m128d tmp_0 = _mm_mul_pd(ssrc, twv);                               \
    const __m128d tmp_1 = _mm_mul_pd(SWAP_RI_128_D(ssrc), twv);                \
    const __m128d lo_1 = _mm_unpacklo_pd(tmp_0, tmp_1);                        \
    const __m128d hi_1 = _mm_unpackhi_pd(tmp_0, tmp_1);                        \
    __m128d _result = _mm_addsub_pd(lo_1, hi_1);                               \
    ST_128_OFFSET_D((sbase) + (stride), offset, _result, is_contiguous);       \
}

// No-twiddle preloaded gather/scatter: signature-compatible with
// TW_PRELOADED_GATHER_*_D_V / TW_PRELOADED_SCATTER_*_D, accepts and ignores
// twv. Used by c2r load and fwd/bwd/r2c store paths so wrapper-local
// TWID_PRELOADED_LOAD/STORE_*_D shims can be pure name-substitution defines.
#define PRELOADED_GATHER_NOTW_256_D(gbase, stride, offset, gdest, twv,         \
                                    is_contiguous)                             \
{                                                                              \
    (FFTZ_VOID) twv;                                                           \
    GATHER2_256_D((gbase) + (stride), offset, (gdest), is_contiguous);         \
}

#define PRELOADED_SCATTER_NOTW_256_D(sbase, stride, offset, ssrc, twv,         \
                                     is_contiguous)                            \
{                                                                              \
    (FFTZ_VOID) twv;                                                           \
    SCATTER2_256_D((sbase) + (stride), offset, (ssrc), is_contiguous);         \
}

#define PRELOADED_GATHER_NOTW_128_D(gbase, stride, offset, gdest, twv,         \
                                    is_contiguous)                             \
{                                                                              \
    (FFTZ_VOID) twv;                                                           \
    LD_128_OFFSET_D((gbase) + (stride), offset, (gdest), is_contiguous);       \
}

#define PRELOADED_SCATTER_NOTW_128_D(sbase, stride, offset, ssrc, twv,         \
                                     is_contiguous)                            \
{                                                                              \
    (FFTZ_VOID) twv;                                                           \
    ST_128_OFFSET_D((sbase) + (stride), offset, (ssrc), is_contiguous);        \
}

/*****************************************************************************
 * GATHER_NOTW / SCATTER_NOTW -- signature-compatible with TW_GATHER /
 * TW_SCATTER
 * but perform plain load/store, ignoring twiddle arguments.
 *****************************************************************************/

#define GATHER_NOTW_128_D(gbase, starr, stidx, offset, gdest, twbuf, lmc,      \
                          is_contiguous)                                       \
    LD_128_OFFSET_D((gbase) + starr[(stidx)], offset, gdest, is_contiguous)

#define GATHER_NOTW_256_D(gbase, starr, stidx, offset, gdest, twbuf, lmc,      \
                          is_contiguous)                                       \
    GATHER2_256_D((gbase) + starr[(stidx)], offset, gdest, is_contiguous)

#define GATHER_NOTW_256_S(gbase, starr, stidx, offset, gdest, twbuf, lmc,      \
                          is_contiguous)                                       \
    GATHER4_256_S((gbase) + starr[(stidx)], offset, gdest, is_contiguous)

#define GATHER_NOTW_128_S(gbase, starr, stidx, offset, gdest, twbuf, lmc,      \
                          is_contiguous)                                       \
    GATHER2_128_S((gbase) + starr[(stidx)], offset, gdest, is_contiguous)

#define GATHER_NOTW_LOW_128_S(gbase, starr, stidx, gdest, twbuf, lmc,          \
                              is_contiguous)                                   \
{                                                                              \
    (FFTZ_VOID)is_contiguous;                                                  \
    LD_LOW_128_S((gbase) + starr[(stidx)], gdest);                             \
}

/**
 * @brief H2 (conjugate-half) counterparts of GATHER_NOTW_* above.
 * Same signature-compatible, twiddle-ignoring interface, but route through
 * the reverse-contiguous H2 gather primitives defined earlier in this file.
 */
#define GATHER_NOTW_H2_128_D(gbase, starr, stidx, offset, gdest, twbuf, lmc,   \
                             is_contiguous)                                    \
    GATHER_H2_128_D((gbase) + starr[(stidx)], offset, gdest)

#define GATHER_NOTW_H2_256_D(gbase, starr, stidx, offset, gdest, twbuf, lmc,   \
                             is_contiguous)                                    \
    GATHER2_H2_256_D((gbase) + starr[(stidx)], offset, gdest)

#define GATHER_NOTW_H2_256_S(gbase, starr, stidx, offset, gdest, twbuf, lmc,   \
                             is_contiguous)                                    \
    GATHER4_H2_256_S((gbase) + starr[(stidx)], offset, gdest)

#define GATHER_NOTW_H2_128_S(gbase, starr, stidx, offset, gdest, twbuf, lmc,   \
                             is_contiguous)                                    \
    GATHER2_H2_128_S((gbase) + starr[(stidx)], offset, gdest)

/**
 * @brief H2 low-lane leftover-point gather. A single leftover point has no
 * lane order to reverse, so this aliases directly to
 * GATHER_NOTW_LOW_128_S.
 */
#define GATHER_NOTW_H2_LOW_128_S(gbase, starr, stidx, gdest, twbuf, lmc,       \
                                 is_contiguous)                                \
    GATHER_NOTW_LOW_128_S(gbase, starr, stidx, gdest, twbuf, lmc,              \
                          is_contiguous)

#define SCATTER_NOTW_128_D(sbase, starr, stidx, offset, ssrc, twbuf, lmc,      \
                           is_contiguous)                                      \
    ST_128_OFFSET_D((sbase) + starr[(stidx)], offset, ssrc, is_contiguous)

#define SCATTER_NOTW_256_D(sbase, starr, stidx, offset, ssrc, twbuf, lmc,      \
                           is_contiguous)                                      \
    SCATTER2_256_D((sbase) + starr[(stidx)], offset, ssrc, is_contiguous)

#define SCATTER_NOTW_256_S(sbase, starr, stidx, offset, ssrc, twbuf, lmc,      \
                           is_contiguous)                                      \
    SCATTER4_256_S((sbase) + starr[(stidx)], offset, ssrc, is_contiguous)

#define SCATTER_NOTW_128_S(sbase, starr, stidx, offset, ssrc, twbuf, lmc,      \
                           is_contiguous)                                      \
    SCATTER2_128_S((sbase) + starr[(stidx)], offset, ssrc, is_contiguous)

#define SCATTER_NOTW_LOW_128_S(sbase, starr, stidx, ssrc, twbuf, lmc,          \
                               is_contiguous)                                  \
{                                                                              \
    (FFTZ_VOID)is_contiguous;                                                  \
    ST_LOW_128_S((sbase) + starr[(stidx)], ssrc);                              \
}

/**
 * @brief H2 (conjugate-half) counterparts of SCATTER_NOTW_* above.
 * Same signature-compatible, twiddle-ignoring interface, but route through
 * the reverse-contiguous H2 scatter primitives defined earlier in this
 * file.
 */
#define SCATTER_NOTW_H2_128_D(sbase, starr, stidx, offset, ssrc, twbuf, lmc,   \
                              is_contiguous)                                   \
    SCATTER_H2_128_D((sbase) + starr[(stidx)], offset, ssrc)

#define SCATTER_NOTW_H2_256_D(sbase, starr, stidx, offset, ssrc, twbuf, lmc,   \
                              is_contiguous)                                   \
    SCATTER2_H2_256_D((sbase) + starr[(stidx)], offset, ssrc)

#define SCATTER_NOTW_H2_256_S(sbase, starr, stidx, offset, ssrc, twbuf, lmc,   \
                              is_contiguous)                                   \
    SCATTER4_H2_256_S((sbase) + starr[(stidx)], offset, ssrc)

#define SCATTER_NOTW_H2_128_S(sbase, starr, stidx, offset, ssrc, twbuf, lmc,   \
                              is_contiguous)                                   \
    SCATTER2_H2_128_S((sbase) + starr[(stidx)], offset, ssrc)

/**
 * @brief H2 low-lane leftover-point scatter. A single leftover point has no
 * lane order to reverse, so this aliases directly to
 * SCATTER_NOTW_LOW_128_S.
 */
#define SCATTER_NOTW_H2_LOW_128_S(sbase, starr, stidx, ssrc, twbuf, lmc,       \
                                  is_contiguous)                               \
    SCATTER_NOTW_LOW_128_S(sbase, starr, stidx, ssrc, twbuf, lmc,              \
                           is_contiguous)

/*****************************************************************************
 * TW_SCATTER / ITW_SCATTER -- twiddle multiply then store.
 * Mirrors TW_GATHER / ITW_GATHER with same signature but store side.
 *****************************************************************************/

#define TW_SCATTER_128_D(sbase, starr, stidx, offset, ssrc, twbuf, lmc,        \
                         is_contiguous)                                        \
{                                                                              \
    const FFTZ_UINTP addr = DATA_STRIDE * (stidx - 1);                         \
    const __m128d cd = _mm_loadu_pd((twbuf) + addr);                           \
    const __m128d _tw_t0 = _mm_mul_pd(ssrc, cd);                               \
    const __m128d _tw_t1 = _mm_mul_pd(SWAP_RI_128_D(ssrc), cd);                \
    const __m128d _tw_lo = _mm_unpacklo_pd(_tw_t0, _tw_t1);                    \
    const __m128d _tw_hi = _mm_unpackhi_pd(_tw_t0, _tw_t1);                    \
    __m128d _result = _mm_addsub_pd(_tw_lo, _tw_hi);                           \
    ST_128_OFFSET_D((sbase) + starr[stidx], offset, _result, is_contiguous);   \
}

#define ITW_SCATTER_128_D(sbase, starr, stidx, offset, ssrc, twbuf, lmc,       \
                          is_contiguous)                                       \
{                                                                              \
    const FFTZ_UINTP addr = DATA_STRIDE * (stidx - 1);                         \
    const __m128d cd = _mm_loadu_pd((twbuf) + addr);                           \
    const __m128d _tw_t0 = _mm_mul_pd(ssrc, cd);                               \
    const __m128d _tw_t1 = _mm_mul_pd(SWAP_RI_128_D(ssrc), cd);                \
    const __m128d _tw_lo = _mm_unpacklo_pd(_tw_t1, _tw_t0);                    \
    const __m128d _tw_hi = _mm_unpackhi_pd(_tw_t1, _tw_t0);                    \
    __m128d _result = SWAP_RI_128_D(_mm_addsub_pd(_tw_lo, _tw_hi));            \
    ST_128_OFFSET_D((sbase) + starr[stidx], offset, _result, is_contiguous);   \
}

#define TW_SCATTER_256_D(sbase, starr, stidx, offset, ssrc, twbuf, lmc,        \
                         is_contiguous)                                        \
{                                                                              \
    __m256d twv;                                                               \
    if ((lmc))                                                                 \
    {                                                                          \
        const FFTZ_UINTP addr =                                                \
            DATA_STRIDE * (stidx - 1) * NUM_SETS_256_D;                        \
        twv = _mm256_loadu_pd((twbuf) + addr);                                 \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        const FFTZ_UINTP addr = DATA_STRIDE * (stidx - 1);                     \
        twv = _mm256_broadcast_pd((__m128d *)((twbuf) + addr));                \
    }                                                                          \
    const __m256d tmp_0 = _mm256_mul_pd(ssrc, twv);                            \
    const __m256d tmp_1 = _mm256_mul_pd(SWAP_RI_256_D(ssrc), twv);             \
    const __m256d lo_1 = _mm256_unpacklo_pd(tmp_0, tmp_1);                     \
    const __m256d hi_1 = _mm256_unpackhi_pd(tmp_0, tmp_1);                     \
    __m256d _result = _mm256_addsub_pd(lo_1, hi_1);                            \
    SCATTER2_256_D((sbase) + starr[stidx], offset, _result, is_contiguous);    \
}

#define ITW_SCATTER_256_D(sbase, starr, stidx, offset, ssrc, twbuf, lmc,       \
                          is_contiguous)                                       \
{                                                                              \
    __m256d twv;                                                               \
    if ((lmc))                                                                 \
    {                                                                          \
        const FFTZ_UINTP addr =                                                \
            DATA_STRIDE * (stidx - 1) * NUM_SETS_256_D;                        \
        twv = _mm256_loadu_pd((twbuf) + addr);                                 \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        const FFTZ_UINTP addr = DATA_STRIDE * (stidx - 1);                     \
        twv = _mm256_broadcast_pd((__m128d *)((twbuf) + addr));                \
    }                                                                          \
    const __m256d tmp_0 = _mm256_mul_pd(ssrc, twv);                            \
    const __m256d tmp_1 = _mm256_mul_pd(SWAP_RI_256_D(ssrc), twv);             \
    const __m256d lo_1 = _mm256_unpacklo_pd(tmp_1, tmp_0);                     \
    const __m256d hi_1 = _mm256_unpackhi_pd(tmp_1, tmp_0);                     \
    __m256d _result = SWAP_RI_256_D(_mm256_addsub_pd(lo_1, hi_1));             \
    SCATTER2_256_D((sbase) + starr[stidx], offset, _result, is_contiguous);    \
}

#define TW_SCATTER_256_S(sbase, starr, stidx, offset, ssrc, twbuf, lmc,        \
                         is_contiguous)                                        \
{                                                                              \
    __m256 twv;                                                                \
    if ((lmc))                                                                 \
    {                                                                          \
        const FFTZ_UINTP addr =                                                \
            DATA_STRIDE * (stidx - 1) * NUM_SETS_256_S;                        \
        twv = _mm256_loadu_ps((twbuf) + addr);                                 \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        const FFTZ_UINTP addr = DATA_STRIDE * (stidx - 1);                     \
        twv = (__m256)(_mm256_broadcast_sd((FFTZ_DOUBLE *)((twbuf) + addr)));  \
    }                                                                          \
    __m256 tmp_0 = _mm256_mul_ps(ssrc, twv);                                   \
    __m256 tmp_1 = _mm256_mul_ps(SWAP_RI_256_S(ssrc), twv);                    \
    tmp_0 = _mm256_permute_ps(tmp_0, 0xD8);                                    \
    tmp_1 = _mm256_permute_ps(tmp_1, 0xD8);                                    \
    const __m256 lo_1 = _mm256_unpacklo_ps(tmp_0, tmp_1);                      \
    const __m256 hi_1 = _mm256_unpackhi_ps(tmp_0, tmp_1);                      \
    __m256 _result = _mm256_addsub_ps(lo_1, hi_1);                             \
    SCATTER4_256_S((sbase) + starr[stidx], offset, _result, is_contiguous);    \
}

#define ITW_SCATTER_256_S(sbase, starr, stidx, offset, ssrc, twbuf, lmc,       \
                          is_contiguous)                                       \
{                                                                              \
    __m256 twv;                                                                \
    if ((lmc))                                                                 \
    {                                                                          \
        const FFTZ_UINTP addr =                                                \
            DATA_STRIDE * (stidx - 1) * NUM_SETS_256_S;                        \
        twv = _mm256_loadu_ps((twbuf) + addr);                                 \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        const FFTZ_UINTP addr = DATA_STRIDE * (stidx - 1);                     \
        twv = (__m256)(_mm256_broadcast_sd((FFTZ_DOUBLE *)((twbuf) + addr)));  \
    }                                                                          \
    __m256 tmp_0 = _mm256_mul_ps(ssrc, twv);                                   \
    __m256 tmp_1 = _mm256_mul_ps(SWAP_RI_256_S(ssrc), twv);                    \
    tmp_0 = _mm256_permute_ps(tmp_0, 0xD8);                                    \
    tmp_1 = _mm256_permute_ps(tmp_1, 0xD8);                                    \
    const __m256 lo_1 = _mm256_unpacklo_ps(tmp_1, tmp_0);                      \
    const __m256 hi_1 = _mm256_unpackhi_ps(tmp_1, tmp_0);                      \
    __m256 _result = SWAP_RI_256_S(_mm256_addsub_ps(lo_1, hi_1));              \
    SCATTER4_256_S((sbase) + starr[stidx], offset, _result, is_contiguous);    \
}

#define TW_SCATTER_128_S(sbase, starr, stidx, offset, ssrc, twbuf, lmc,        \
                         is_contiguous)                                        \
{                                                                              \
    __m128 twv;                                                                \
    if ((lmc))                                                                 \
    {                                                                          \
        const FFTZ_UINTP addr =                                                \
            DATA_STRIDE * (stidx - 1) * NUM_SETS_128_S;                        \
        twv = _mm_loadu_ps((twbuf) + addr);                                    \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        const FFTZ_UINTP addr = DATA_STRIDE * (stidx - 1);                     \
        twv = (__m128)(_mm_loaddup_pd((FFTZ_DOUBLE *)((twbuf) + addr)));       \
    }                                                                          \
    __m128 tmp_0 = _mm_mul_ps(ssrc, twv);                                      \
    __m128 tmp_1 = _mm_mul_ps(SWAP_RI_128_S(ssrc), twv);                       \
    tmp_0 = _mm_permute_ps(tmp_0, 0xD8);                                       \
    tmp_1 = _mm_permute_ps(tmp_1, 0xD8);                                       \
    const __m128 lo_1 = _mm_unpacklo_ps(tmp_0, tmp_1);                         \
    const __m128 hi_1 = _mm_unpackhi_ps(tmp_0, tmp_1);                         \
    __m128 _result = _mm_addsub_ps(lo_1, hi_1);                                \
    SCATTER2_128_S((sbase) + starr[stidx], offset, _result, is_contiguous);    \
}

#define ITW_SCATTER_128_S(sbase, starr, stidx, offset, ssrc, twbuf, lmc,       \
                          is_contiguous)                                       \
{                                                                              \
    __m128 twv;                                                                \
    if ((lmc))                                                                 \
    {                                                                          \
        const FFTZ_UINTP addr =                                                \
            DATA_STRIDE * (stidx - 1) * NUM_SETS_128_S;                        \
        twv = _mm_loadu_ps((twbuf) + addr);                                    \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        const FFTZ_UINTP addr = DATA_STRIDE * (stidx - 1);                     \
        twv = (__m128)(_mm_loaddup_pd((FFTZ_DOUBLE *)((twbuf) + addr)));       \
    }                                                                          \
    __m128 tmp_0 = _mm_mul_ps(ssrc, twv);                                      \
    __m128 tmp_1 = _mm_mul_ps(SWAP_RI_128_S(ssrc), twv);                       \
    tmp_0 = _mm_permute_ps(tmp_0, 0xD8);                                       \
    tmp_1 = _mm_permute_ps(tmp_1, 0xD8);                                       \
    const __m128 lo_1 = _mm_unpacklo_ps(tmp_1, tmp_0);                         \
    const __m128 hi_1 = _mm_unpackhi_ps(tmp_1, tmp_0);                         \
    __m128 _result = SWAP_RI_128_S(_mm_addsub_ps(lo_1, hi_1));                 \
    SCATTER2_128_S((sbase) + starr[stidx], offset, _result, is_contiguous);    \
}

#define TW_SCATTER_LOW_128_S(sbase, starr, stidx, ssrc, twbuf, lmc,            \
                             is_contiguous)                                    \
{                                                                              \
    const FFTZ_UINTP addr = DATA_STRIDE * (stidx - 1);                         \
    __m128 twv;                                                                \
    LD_LOW_128_S((twbuf) + addr, twv);                                         \
    __m128 tmp_0 = _mm_mul_ps(ssrc, twv);                                      \
    __m128 tmp_1 = _mm_mul_ps(SWAP_RI_128_S(ssrc), twv);                       \
    tmp_0 = _mm_permute_ps(tmp_0, 0xD8);                                       \
    tmp_1 = _mm_permute_ps(tmp_1, 0xD8);                                       \
    const __m128 lo_1 = _mm_unpacklo_ps(tmp_0, tmp_1);                         \
    const __m128 hi_1 = _mm_unpackhi_ps(tmp_0, tmp_1);                         \
    __m128 _result = _mm_addsub_ps(lo_1, hi_1);                                \
    ST_LOW_128_S((sbase) + starr[stidx], _result);                             \
}

#define ITW_SCATTER_LOW_128_S(sbase, starr, stidx, ssrc, twbuf, lmc,           \
                              is_contiguous)                                   \
{                                                                              \
    const FFTZ_UINTP addr = DATA_STRIDE * (stidx - 1);                         \
    __m128 twv;                                                                \
    LD_LOW_128_S((twbuf) + addr, twv);                                         \
    __m128 tmp_0 = _mm_mul_ps(ssrc, twv);                                      \
    __m128 tmp_1 = _mm_mul_ps(SWAP_RI_128_S(ssrc), twv);                       \
    tmp_0 = _mm_permute_ps(tmp_0, 0xD8);                                       \
    tmp_1 = _mm_permute_ps(tmp_1, 0xD8);                                       \
    const __m128 lo_1 = _mm_unpacklo_ps(tmp_1, tmp_0);                         \
    const __m128 hi_1 = _mm_unpackhi_ps(tmp_1, tmp_0);                         \
    __m128 _result = SWAP_RI_128_S(_mm_addsub_ps(lo_1, hi_1));                 \
    ST_LOW_128_S((sbase) + starr[(stidx)], _result);                           \
}

#endif // AOCLFFTZ_SIMD_COMMON_H
