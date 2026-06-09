// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file r2hc_simd_avx128.h
 *
 *  @brief List of common operations and macros used for R2HC and R2HCF AVX128
 *  kernels.
 *
 *  This file contains the category-wise macros and common functions used
 *  in the AVX128 real kernel variants
 *
 *  @author Jeya R
 */

#ifndef AOCLFFTZ_R2HC_SIMD_AVX128_H
#define AOCLFFTZ_R2HC_SIMD_AVX128_H

#include <immintrin.h>
#include "api/types.h"

#define NUM_SETS_REAL_128_S 4
#define NUM_SETS_REAL_128_D 2

/************************ MACRO FUNCTIONS ************************/

/**
 * @brief load 2 64 bit double precision floating point number from memory
 * addresses specified by base address into 128 bit register.
 * Operation : 2 MOV(load)
 * Sample load of 2 real floating point with
 * Ar residing in base address and
 * Br residing in base + offset address.
 * dest = |-------------64-bit------------|-------------64-bit------------|
 * dest = |-------------Ar----------------|-------------Br----------------|
 */
#define LDR_128_D(base, offset, dest)                                          \
    {                                                                          \
        dest = _mm_load_sd(base);                                              \
        base += offset;                                                        \
        dest = _mm_loadh_pd(dest, base);                                       \
    }

/**
 * @brief Load 2 complex numbers (each with 64-bit double precision real and
 * imaginary parts) from memory into two 128-bit registers. The real parts are
 * loaded into one register and the imaginary parts into another.
 * Operation : 2 MOV(load), 2 PERM
 * Sample load of 2 complex floating point with
 * A(Ar,Ai) residing in base address and
 * B(Br,Bi) residing in base + offset address.
 * dest  = |-------------64-bit------------|-------------64-bit------------|
 * dest1 = |-------------Ar----------------|-------------Br----------------|
 * dest2 = |-------------Ai----------------|-------------Bi----------------|
*/
#define LDRI_2x128_D(base, offset, dest1, dest2)                               \
    {                                                                          \
        __m128d _r1, _r2;                                                      \
        _r1 = _mm_loadu_pd(base);                                              \
        base += offset;                                                        \
        _r2 = _mm_loadu_pd(base);                                              \
        dest1 = _mm_shuffle_pd(_r1, _r2, 0x0);                                 \
        dest2 = _mm_shuffle_pd(_r1, _r2, 0x3);                                 \
    }

/**
 * @brief load 4 32 bit single precision floating point number from memory
 * addresses specified by base address into 128 bit register.
 * Operation : 4 MOV(load), 3 PERM
 * Sample load of 4 real floating point with
 * Ar residing in base address
 * Br residing in base + offset address
 * Cr residing in base + (2 * offset) address and
 * Dr residing in base + (3 * offset) address.
 * dest = |-----32-bit-----|-----32-bit-----|-----32-bit-----|-----32-bit-----|
 * dest = |-------Ar-------|-------Br-------|-------Cr-------|-------Dr-------|
 */
#define LDR_128_S(base, offset, dest)                                          \
    {                                                                          \
        __m128 _low, _high, _tmp;                                              \
        _low = _mm_load_ss(base);                                              \
        _high = _mm_load_ss(base + (offset));                                  \
        _low = _mm_unpacklo_ps(_low, _high);                                   \
        _tmp = _mm_load_ss(base + (offset << 1));                              \
        _high = _mm_load_ss(base + (offset * 3));                              \
        _tmp = _mm_unpacklo_ps(_tmp, _high);                                   \
        dest = _mm_shuffle_ps(_low, _tmp, 0x44);                               \
    }

/**
 * @brief Load 4 complex numbers (each with 32-bit double precision real and
 * imaginary parts) from memory into two 128-bit registers. The real parts are
 * loaded into one register and the imaginary parts into another.
 * Operation : 4 MOV(load), 3 PERM
 * Sample load of 4 complex floating point with
 * A(Ar,Ai) residing in base address
 * B(Br,Bi) residing in base + offset address
 * C(Cr,Ci) residing in base + (2 * offset) address and
 * D(Dr,Di) residing in base + (3 * offset) address.
 * dest  = |-----32-bit-----|-----32-bit-----|-----32-bit-----|-----32-bit-----|
 * dest1 = |-------Ar-------|-------Br-------|-------Cr-------|-------Dr-------|
 * dest2 = |-------Ai-------|-------Bi-------|-------Ci-------|-------Di-------|
*/
#define LDRI_2x128_S(base, offset, dest1, dest2)                               \
    {                                                                          \
        __m128 _tmp, _tmp1;                                                    \
        _tmp = _mm_loadu_ps(base);                                             \
        base += offset;                                                        \
        _tmp1 = _mm_loadu_ps(base);                                            \
        _tmp = _mm_shuffle_ps(_tmp, _tmp1, 0x44);                              \
        base += offset;                                                        \
        _tmp1 = _mm_loadu_ps(base);                                            \
        base += offset;                                                        \
        _tmp1 = _mm_loadh_pi(_tmp1, (__m64 *)base);                            \
        dest1 = _mm_shuffle_ps(_tmp, _tmp1, 0x88);                             \
        dest2 = _mm_shuffle_ps(_tmp, _tmp1, 0xDD);                             \
    }

/**
 * @brief load 2 32 bit single precision floating point number from memory
 * addresses specified by base address into first half of the 128 bit register.
 * Operation : 2 MOV(load), 1 PERM
 * Sample load of 2 real floating point with
 * Ar residing in base address and
 * Br residing in base + offset address.
 * dest = |-----32-bit-----|-----32-bit-----|-----32-bit-----|-----32-bit-----|
 * dest = |-------Ar-------|-------Br-------|----------------|----------------|
 */
#define LDHR_128_S(base, offset, dest)                                         \
    {                                                                          \
        __m128 _low, _high;                                                    \
        _low = _mm_load_ss(base);                                              \
        _high = _mm_load_ss(base + (offset));                                  \
        dest = _mm_unpacklo_ps(_low, _high);                                   \
    }

/**
 * @brief @brief Load 2 complex numbers (each with 64-bit double precision real and
 * imaginary parts) from memory into first half of 2 128 bit registers. The real
 * parts are loaded into one register and the imaginary parts into another.
 * Operation : 2 MOV(load), 2 PERM
 * Sample load of 2 complex floating point with
 * A(Ar,Ai) residing in base address and
 * B(Br,Bi) residing in base + offset addres
 * dest  = |-----32-bit-----|-----32-bit-----|-----32-bit-----|-----32-bit-----|
 * dest1 = |-------Ar-------|-------Br-------|----------------|----------------|
 * dest2 = |-------Ai-------|-------Bi-------|----------------|----------------|
*/
#define LDHRI_2x128_S(base, offset, dest1, dest2)                              \
    {                                                                          \
        __m128 _tmp;                                                           \
        _tmp = _mm_loadu_ps(base);                                             \
        base += offset;                                                        \
        _tmp = _mm_loadh_pi(_tmp, (__m64 *)base);                              \
        dest1 = _mm_shuffle_ps(_tmp, _tmp, 0x8);                               \
        dest2 = _mm_shuffle_ps(_tmp, _tmp, 0xD);                               \
    }

/**
 * @brief store 2 64 bit double precision floating point number from 128 bit
 * register into memory addresses specified by base address
 * Operation : 2 MOV(store)
 * Sample store of 2 real floating point with
 * Ar stored in base address and
 * Br stored in base + offset address.
 * src = |-------------64-bit------------|-------------64-bit------------|
 * src = |-------------Ar----------------|-------------Br----------------|
 */
#define STR_128_D(base, offset, src)                                           \
    {                                                                          \
        _mm_storel_pd(base, src);                                              \
        base += offset;                                                        \
        _mm_storeh_pd(base, src);                                              \
    }

/**
 * @brief Store 2 complex numbers (each with 64-bit double precision real and
 * imaginary parts) from 2 128-bit registers into memory. The real parts are
 * taken from one register, and the imaginary parts from another, and stored
 * into the memory addresses specified by the base address.
 * Operation : 2 MOV(store), 2 PERM
 * Sample store of 2 complex floating point with
 * A(Ar,Ai) stored in base address and
 * B(Br,Bi) stored in base + offset addres.
 * src  = |-------------64-bit------------|-------------64-bit------------|
 * src1 = |-------------Ar----------------|-------------Br----------------|
 * src2 = |-------------Ai----------------|-------------Bi----------------|
 */
#define STRI_2x128_D(base, offset, src1, src2)                                 \
    {                                                                          \
        __m128d _s1c, _s2c;                                                    \
        _s1c = _mm_shuffle_pd(src1, src2, 0x0);                                \
        _mm_storeu_pd(base, _s1c);                                             \
        base += offset;                                                        \
        _s2c = _mm_shuffle_pd(src1, src2, 0x3);                                \
        _mm_storeu_pd(base, _s2c);                                             \
    }

/**
 * @brief store 4 32 bit single precision floating point number from 128 bit
 * register into memory addresses specified by base address
 * Operation : 4 MOV(store), 3 PERM
 * Sample store of 4 real floating point with
 * Ar stored in base address
 * Br stored in base + offset address
 * Cr stored in base + (2 * offset) address and
 * Dr stored in base + (3 * offset) address.
 * src = |-----32-bit-----|-----32-bit-----|-----32-bit-----|-----32-bit-----|
 * src = |-------Ar-------|-------Br-------|-------Cr-------|-------Dr-------|
 */

#define STR_128_S(base, offset, src)                                           \
    {                                                                          \
        _mm_store_ss(base, src);                                               \
        base += offset;                                                        \
        __m128 _tmp = _mm_permute_ps(src, 0xE1);                               \
        _mm_store_ss(base, _tmp);                                              \
        base += offset;                                                        \
        _tmp = _mm_permute_ps(src, 0xD2);                                      \
        _mm_store_ss(base, _tmp);                                              \
        base += offset;                                                        \
        _tmp = _mm_permute_ps(src, 0x93);                                      \
        _mm_store_ss(base, _tmp);                                              \
    }

/**
 * @brief Store 4 complex numbers (each with 32-bit double precision real and
 * imaginary parts) from 2 128-bit registers into memory. The real parts are
 * taken from one register, and the imaginary parts from another, and stored
 * into the memory addresses specified by the base address.
 * Operation : 4 MOV(store), 2 PERM
 * Sample store of 4 complex floating point with
 * A(Ar,Ai) stored in base address
 * B(Br,Bi) stored in base + offset address
 * C(Cr,Ci) stored in base + (2 * offset) address and
 * D(Dr,Di) stored in base + (3 * offset) address.
 * src  = |-----32-bit-----|-----32-bit-----|-----32-bit-----|-----32-bit-----|
 * src1 = |-------Ar-------|-------Br-------|-------Cr-------|-------Dr-------|
 * src2 = |-------Ai-------|-------Bi-------|-------Ci-------|-------Di-------|
 */
#define STRI_2x128_S(base, offset, src1, src2)                                 \
{                                                                              \
    __m128 _low, _high;                                                        \
    _low = _mm_unpacklo_ps(src1, src2);                                        \
    _high = _mm_unpackhi_ps(src1, src2);                                       \
    _mm_storel_pi((__m64 *)base, _low);                                        \
    base += offset;                                                            \
    _mm_storeh_pi((__m64 *)base, _low);                                        \
    base += offset;                                                            \
    _mm_storel_pi((__m64 *)base, _high);                                       \
    base += offset;                                                            \
    _mm_storeh_pi((__m64 *)base, _high);                                       \
}

/**
 * @brief store 2 32 bit single precision floating point number from 128 bit
 * register into memory addresses specified by base address
 * Operation : 2 MOV(store), 1 PERM
 * Sample store of 2 real floating point with
 * Ar stored in base address and
 * Br stored in base + offset address.
 * src = |-----32-bit-----|-----32-bit-----|-----32-bit-----|-----32-bit-----|
 * src = |-------Ar-------|-------Br-------|----------------|----------------|
 */

#define STHR_128_S(base, offset, src)                                          \
    {                                                                          \
        _mm_store_ss(base, src);                                               \
        base += offset;                                                        \
        __m128 _tmp = _mm_shuffle_ps(src, src, 0xE1);                          \
        _mm_store_ss(base, _tmp);                                              \
    }

/**
 * @brief Store 2 complex numbers (each with 32-bit double precision real and
 * imaginary parts) from fist half of 2 128-bit registers into memory. The real
 * parts are taken from one register, and the imaginary parts from another, and
 * stored into the memory addresses specified by the base address.
 * Operation : 2 MOV(store), 1 PERM
 * Sample store of 2 complex floating point with
 * A(Ar,Ai) stored in base address and
 * B(Br,Bi) stored in base + offset addres
 * src  = |-----32-bit-----|-----32-bit-----|-----32-bit-----|-----32-bit-----|
 * src1 = |-------Ar-------|-------Br-------|----------------|----------------|
 * src2 = |-------Ai-------|-------Bi-------|----------------|----------------|
 */
#define STHRI_2x128_S(base, offset, src1, src2)                                \
{                                                                              \
    __m128 _low;                                                               \
    _low = _mm_unpacklo_ps(src1, src2);                                        \
    _mm_storel_pi((__m64 *)base, _low);                                        \
    base += offset;                                                            \
    _mm_storeh_pi((__m64 *)base, _low);                                        \
}

/**
 * @brief prepare -0.0 for negation.
 */
union zero_negate_128
{
    unsigned u[4];
    __m128 s;
    __m128d d;
};

static const union zero_negate_128
        _negate_128_f = {{ 0x80000000, 0x80000000, 0x80000000, 0x80000000 }};
static const union zero_negate_128
        _negate_128_d = {{ 0x00000000, 0x80000000, 0x00000000, 0x80000000 }};

/**
 * @brief take negation of a 128-bit vector __m128d A => -A
 * Operation : 1 OTHERS(xor)
 */
#define NEGATE_128_S(x) _mm_xor_ps(_negate_128_f.s, x)
#define NEGATE_128_D(x) _mm_xor_pd(_negate_128_d.d, x)

#endif // AOCLFFTZ_R2HC_SIMD_AVX128_H
