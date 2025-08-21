/**
 * Copyright (C) 2024, Advanced Micro Devices. All rights reserved.
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
#define GATHER2_128_S(base, offset, dest)                                      \
{                                                                              \
    if (offset == 2)                                                           \
    {                                                                          \
        dest = _mm_loadu_ps(base);                                             \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        dest = _mm_loadu_ps(base);                                             \
        dest = _mm_loadh_pi(dest, (__m64 *)((base) + (offset)));               \
    }                                                                          \
}

/**
 * @brief store two complex numbers(real,imaginary) of 32 bit single precision
 * floating point number from 128 bit register into memory addresses
 * specified by base address and offset.
 * Operations : 2 MOV(store)
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 2, perm: 0, other: 0}
#define SCATTER2_128_S(base, offset, src)                                      \
{                                                                              \
    if (offset == 2)                                                           \
    {                                                                          \
        _mm_storeu_ps(base, src);                                              \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        _mm_storel_pi((__m64 *)(base), src);                                   \
        _mm_storeh_pi((__m64 *)((base) + (offset)), src);                      \
    }                                                                          \
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
#define LD_128_OFFSET_D(base, offset, dest)                                    \
{                                                                              \
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
#define ST_128_OFFSET_D(base, offset, src)                                     \
{                                                                              \
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
#define GATHER4_256_S(base, offset, dest)                                      \
{                                                                              \
    if (offset == 2)                                                           \
    {                                                                          \
        dest = _mm256_loadu_ps(base);                                          \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        __m128 _low, _high, _tmp;                                              \
        _low = _mm_loadu_ps(base);                                             \
        _tmp = _mm_loadu_ps((base) + (offset));                                \
        _low = _mm_shuffle_ps(_low, _tmp, 68);                                 \
        _high = _mm_loadu_ps((base) + 2 * (offset));                           \
        _high = _mm_loadh_pi(_high, (__m64 *)((base) + 3 * (offset)));         \
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
#define SCATTER4_256_S(base, offset, src)                                      \
{                                                                              \
    if (offset == 2)                                                           \
    {                                                                          \
        _mm256_storeu_ps(base, src);                                           \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        __m128 _high = _mm256_extractf128_ps(src, 1);                          \
        __m128 _low = _mm256_castps256_ps128(src);                             \
        _mm_storel_pi((__m64 *)(base), _low);                                  \
        _mm_storeh_pi((__m64 *)((base) + (offset)), _low);                     \
        _mm_storel_pi((__m64 *)((base) + 2 * (offset)), _high);                \
        _mm_storeh_pi((__m64 *)((base) + 3 * (offset)), _high);                \
    }                                                                          \
}

/**
 * @brief load two complex numbers(real,imaginary) of 64 bit double precision
 * floating point number from memory addresses specified by base address
 * and offset into 256 bit register.
 * Operations : 2 MOV(load), 1 OTHERS(insert). Cast is excluded as it
 * will be a compile time operation.
 */
// Cost: {fma: 0, mul: 0, add: 0, move: 2, perm: 0, other: 1}
#define GATHER2_256_D(base, offset, dest)                                      \
{                                                                              \
    if (offset == 2)                                                           \
    {                                                                          \
        dest = _mm256_loadu_pd(base);                                          \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        __m128d _low, _high;                                                   \
        _low = _mm_loadu_pd(base);                                             \
        _high = _mm_loadu_pd((base) + (offset));                               \
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
#define SCATTER2_256_D(base, offset, src)                                      \
{                                                                              \
    if (offset == 2)                                                           \
    {                                                                          \
        _mm256_storeu_pd(base, src);                                           \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        __m128d _high = _mm256_extractf128_pd(src, 1);                         \
        __m128d _low = _mm256_castpd256_pd128(src);                            \
        _mm_storeu_pd(base, _low);                                             \
        _mm_storeu_pd((base) + offset, _high);                                 \
    }                                                                          \
}

// Cost: {fma: 0, mul: 2, add: 1, move: 3, perm: 2, other: 0}
#define ITW_GATHER_128_D(gbase, starr, stidx, offset, gdest, twbuf, n, col)    \
{                                                                              \
    const UINTP addr = DATA_STRIDE * ((stidx) * (n) + (col));                  \
    const __m128d cd = _mm_loadu_pd((twbuf) + addr);                           \
    const __m128d bb = _mm_loaddup_pd((gbase) + starr[(stidx)]);               \
    const __m128d aa = _mm_loaddup_pd((gbase) + starr[(stidx)] + 1);           \
    const __m128d ca_da = _mm_mul_pd(cd, aa);                                  \
    const __m128d cb_db = _mm_mul_pd(cd, bb);                                  \
    const __m128d db_cb = SWAP_RI_128_D(cb_db);                                \
    gdest = SWAP_RI_128_D(_mm_addsub_pd(ca_da, db_cb));                        \
}

// Cost: {fma: 0, mul: 2, add: 1, move: 3, perm: 1, other: 0}
#define TW_GATHER_128_D(gbase, starr, stidx, offset, gdest, twbuf, n, col)     \
{                                                                              \
    const UINTP addr = DATA_STRIDE * ((stidx) * (n) + (col));                  \
    const __m128d cd = _mm_loadu_pd((twbuf) + addr);                           \
    const __m128d aa = _mm_loaddup_pd((gbase) + starr[(stidx)]);               \
    const __m128d bb = _mm_loaddup_pd((gbase) + starr[(stidx)] + 1);           \
    const __m128d ca_da = _mm_mul_pd(cd, aa);                                  \
    const __m128d cb_db = _mm_mul_pd(cd, bb);                                  \
    const __m128d db_cb = SWAP_RI_128_D(cb_db);                                \
    gdest = _mm_addsub_pd(ca_da, db_cb);                                       \
}

// Cost: {fma: 0, mul: 2, add: 1, move: 3, perm: 4, other: 1}
#define ITW_GATHER_256_D(gbase, starr, stidx, offset, gdest, twbuf, n, col)    \
{                                                                              \
    const UINTP addr = DATA_STRIDE * ((stidx) * (n) + (col));                  \
    const __m256d twv = _mm256_loadu_pd((twbuf) + addr);                       \
    __m256d tmp_in;                                                            \
    GATHER2_256_D((gbase) + starr[(stidx)], (offset), tmp_in);                 \
    const __m256d tmp_0 = _mm256_mul_pd(tmp_in, twv);                          \
    const __m256d tmp_1 = _mm256_mul_pd(SWAP_RI_256_D(tmp_in), twv);           \
    const __m256d lo_1 = _mm256_unpacklo_pd(tmp_1, tmp_0);                     \
    const __m256d hi_1 = _mm256_unpackhi_pd(tmp_1, tmp_0);                     \
    gdest = SWAP_RI_256_D(_mm256_addsub_pd(lo_1, hi_1));                       \
}

// Cost: {fma: 0, mul: 2, add: 1, move: 3, perm: 3, other: 1}
#define TW_GATHER_256_D(gbase, starr, stidx, offset, gdest, twbuf, n, col)     \
{                                                                              \
    const UINTP addr = DATA_STRIDE * ((stidx) * (n) + (col));                  \
    const __m256d twv = _mm256_loadu_pd((twbuf) + addr);                       \
    __m256d tmp_in;                                                            \
    GATHER2_256_D((gbase) + starr[(stidx)], (offset), tmp_in);                 \
    const __m256d tmp_0 = _mm256_mul_pd(tmp_in, twv);                          \
    const __m256d tmp_1 = _mm256_mul_pd(SWAP_RI_256_D(tmp_in), twv);           \
    const __m256d lo_1 = _mm256_unpacklo_pd(tmp_0, tmp_1);                     \
    const __m256d hi_1 = _mm256_unpackhi_pd(tmp_0, tmp_1);                     \
    gdest = _mm256_addsub_pd(lo_1, hi_1);                                      \
}

// Cost: {fma: 0, mul: 2, add: 1, move: 5, perm: 7, other: 1}
#define ITW_GATHER_256_S(gbase, starr, stidx, offset, gdest, twbuf, n, col)    \
{                                                                              \
    const UINTP addr = DATA_STRIDE * ((stidx) * (n) + (col));                  \
    const __m256 twv = _mm256_loadu_ps((twbuf) + addr);                        \
    __m256 tmp_in;                                                             \
    GATHER4_256_S((gbase) + starr[(stidx)], (offset), tmp_in);                 \
    __m256 tmp_0 = _mm256_mul_ps(tmp_in, twv);                                 \
    __m256 tmp_1 = _mm256_mul_ps(SWAP_RI_256_S(tmp_in), twv);                  \
    tmp_0 = _mm256_permute_ps(tmp_0, 0xD8 /*0b11011000*/);                     \
    tmp_1 = _mm256_permute_ps(tmp_1, 0xD8 /*0b11011000*/);                     \
    const __m256 lo_1 = _mm256_unpacklo_ps(tmp_1, tmp_0);                      \
    const __m256 hi_1 = _mm256_unpackhi_ps(tmp_1, tmp_0);                      \
    gdest = SWAP_RI_256_S(_mm256_addsub_ps(lo_1, hi_1));                       \
}

// Cost: {fma: 0, mul: 2, add: 1, move: 5, perm: 6, other: 1}
#define TW_GATHER_256_S(gbase, starr, stidx, offset, gdest, twbuf, n, col)     \
{                                                                              \
    const UINTP addr = DATA_STRIDE * ((stidx) * (n) + (col));                  \
    const __m256 twv = _mm256_loadu_ps((twbuf) + addr);                        \
    __m256 tmp_in;                                                             \
    GATHER4_256_S((gbase) + starr[(stidx)], (offset), tmp_in);                 \
    __m256 tmp_0 = _mm256_mul_ps(tmp_in, twv);                                 \
    __m256 tmp_1 = _mm256_mul_ps(SWAP_RI_256_S(tmp_in), twv);                  \
    tmp_0 = _mm256_permute_ps(tmp_0, 0xD8 /*0b11011000*/);                     \
    tmp_1 = _mm256_permute_ps(tmp_1, 0xD8 /*0b11011000*/);                     \
    const __m256 lo_1 = _mm256_unpacklo_ps(tmp_0, tmp_1);                      \
    const __m256 hi_1 = _mm256_unpackhi_ps(tmp_0, tmp_1);                      \
    gdest = _mm256_addsub_ps(lo_1, hi_1);                                      \
}

// Cost: {fma: 0, mul: 2, add: 1, move: 3, perm: 6, other: 0}
#define ITW_GATHER_128_S(gbase, starr, stidx, offset, gdest, twbuf, n, col)    \
{                                                                              \
    const UINTP addr = DATA_STRIDE * ((stidx) * (n) + (col));                  \
    const __m128 twv = _mm_loadu_ps((twbuf) + addr);                           \
    __m128 tmp_in;                                                             \
    GATHER2_128_S((gbase) + starr[(stidx)], (offset), tmp_in);                 \
    __m128 tmp_0 = _mm_mul_ps(tmp_in, twv);                                    \
    __m128 tmp_1 = _mm_mul_ps(SWAP_RI_128_S(tmp_in), twv);                     \
    tmp_0 = _mm_permute_ps(tmp_0, 0xD8 /*0b11011000*/);                        \
    tmp_1 = _mm_permute_ps(tmp_1, 0xD8 /*0b11011000*/);                        \
    const __m128 lo_1 = _mm_unpacklo_ps(tmp_1, tmp_0);                         \
    const __m128 hi_1 = _mm_unpackhi_ps(tmp_1, tmp_0);                         \
    gdest = SWAP_RI_128_S(_mm_addsub_ps(lo_1, hi_1));                          \
}

// Cost: {fma: 0, mul: 2, add: 1, move: 3, perm: 5, other: 0}
#define TW_GATHER_128_S(gbase, starr, stidx, offset, gdest, twbuf, n, col)     \
{                                                                              \
    const UINTP addr = DATA_STRIDE * ((stidx) * (n) + (col));                  \
    const __m128 twv = _mm_loadu_ps((twbuf) + addr);                           \
    __m128 tmp_in;                                                             \
    GATHER2_128_S((gbase) + starr[(stidx)], (offset), tmp_in);                 \
    __m128 tmp_0 = _mm_mul_ps(tmp_in, twv);                                    \
    __m128 tmp_1 = _mm_mul_ps(SWAP_RI_128_S(tmp_in), twv);                     \
    tmp_0 = _mm_permute_ps(tmp_0, 0xD8 /*0b11011000*/);                        \
    tmp_1 = _mm_permute_ps(tmp_1, 0xD8 /*0b11011000*/);                        \
    const __m128 lo_1 = _mm_unpacklo_ps(tmp_0, tmp_1);                         \
    const __m128 hi_1 = _mm_unpackhi_ps(tmp_0, tmp_1);                         \
    gdest = _mm_addsub_ps(lo_1, hi_1);                                         \
}

// Cost: {fma: 0, mul: 2, add: 1, move: 2, perm: 6, other: 0}
#define ITW_GATHER_LOW_128_S(gbase, starr, stidx, gdest, twbuf, n, col)        \
{                                                                              \
    const UINTP addr = DATA_STRIDE * ((stidx) * (n) + (col));                  \
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
#define TW_GATHER_LOW_128_S(gbase, starr, stidx, gdest, twbuf, n, col)         \
{                                                                              \
    const UINTP addr = DATA_STRIDE * ((stidx) * (n) + (col));                  \
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

static const union data_union_128
            _conj_128_f = {{ 0x00000000, 0x80000000, 0x00000000, 0x80000000 }};
static const union data_union_128
            _conj_128_d = {{ 0x00000000, 0x00000000, 0x00000000, 0x80000000 }};

static const union data_union_128
    _neg_128_f[2] = {
        {.u = { 0x00000000, 0x00000000, 0x00000000, 0x00000000 }},
        {.u = { 0x80000000, 0x80000000, 0x80000000, 0x80000000 }}
    };
static const union data_union_128
    _neg_128_d[2] = {
        {.u = { 0x00000000, 0x00000000, 0x00000000, 0x00000000 }},
        {.u = { 0x00000000, 0x80000000, 0x00000000, 0x80000000 }}
    };

static const union data_union_256
            _conj_256_f = {{ 0x00000000, 0x80000000, 0x00000000, 0x80000000,
                             0x00000000, 0x80000000, 0x00000000, 0x80000000 }};
static const union data_union_256
            _conj_256_d = {{ 0x00000000, 0x00000000, 0x00000000, 0x80000000,
                             0x00000000, 0x00000000, 0x00000000, 0x80000000 }};

static const union data_union_256
    _neg_256_f[2] = {
        {.u = { 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 }},
        {.u = { 0x80000000, 0x80000000, 0x80000000, 0x80000000,
                0x80000000, 0x80000000, 0x80000000, 0x80000000 }}
    };
static const union data_union_256
    _neg_256_d[2] = {
        {.u = { 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                0x00000000, 0x00000000, 0x00000000, 0x00000000 }},
        {.u = { 0x00000000, 0x80000000, 0x00000000, 0x80000000,
                0x00000000, 0x80000000, 0x00000000, 0x80000000 }}
    };

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
#endif // AOCLFFTZ_SIMD_COMMON_H
