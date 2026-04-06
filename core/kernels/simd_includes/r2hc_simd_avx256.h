// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file r2hc_simd_avx256.h
 *
 *  @brief List of common operations and macros used for R2HC and R2HCF AVX256
 *  kernels.
 *
 *  This file contains the category-wise macros and common functions used
 *  in the AVX256 real kernel variants
 *
 *  @author Jeya R
 */

#ifndef AOCLFFTZ_R2HC_SIMD_AVX256_H
#define AOCLFFTZ_R2HC_SIMD_AVX256_H

#include "core/kernels/simd_includes/r2hc_simd_avx128.h"

#define NUM_SETS_REAL_256_S 8
#define NUM_SETS_REAL_256_D 4

/************************ MACRO FUNCTIONS ************************/

/**
 * @brief load 4 64 bit double precision floating point number from memory
 * addresses specified by base address into 256 bit register.
 * Operation : 4 MOV, 1 OTHERS(insert)
 * Sample load of 4 real floating point with
 * Ar residing in base address
 * Br residing in base + offset address
 * Cr residing in base + (2 * offset) address and
 * Dr residing in base + (3 * offset) address.
 * dest = |-----64-bit-----|-----64-bit-----|-----64-bit-----|-----64-bit-----|
 * dest = |-------Ar-------|-------Br-------|-------Cr-------|-------Dr-------|
 */
#define LDR_256_D(base, offset, dest)                                          \
    {                                                                          \
        __m128d _low, _high;                                                   \
        _low = _mm_load_sd(base);                                              \
        base += offset;                                                        \
        _low = _mm_loadh_pd(_low, base);                                       \
        base += offset;                                                        \
        _high = _mm_load_sd(base);                                             \
        base += offset;                                                        \
        _high = _mm_loadh_pd(_high, base);                                     \
        dest = _mm256_insertf128_pd(_mm256_castpd128_pd256(_low),_high,1);     \
    }

/**
 * @brief Load 4 complex numbers (each with 64-bit double precision real and
 * imaginary parts) from memory into 2 256-bit registers. The real parts are
 * loaded into one register and the imaginary parts into another.
 * Operation : 4 MOV, 2 OTHERS, 2 PERM
 * Sample load of 4 complex floating point with
 * A(Ar,Ai) residing in base address
 * B(Br,Bi) residing in base + offset address
 * C(Cr,Ci) residing in base + (2 * offset) address and
 * D(Dr,Di) residing in base + (3 * offset) address.
 * dest  = |-----64-bit-----|-----64-bit-----|-----64-bit-----|-----64-bit-----|
 * dest1 = |-------Ar-------|-------Br-------|-------Cr-------|-------Dr-------|
 * dest2 = |-------Ai-------|-------Bi-------|-------Ci-------|-------Di-------|
 */
#define LDRI_2x256_D(base, offset, dest1, dest2)                               \
    {                                                                          \
        __m128d _s1c, _s2c, _s3c, _s4c;                                        \
        __m256d _r1, _r2;                                                      \
        _s1c = _mm_loadu_pd(base);                                             \
        base += offset;                                                        \
        _s2c = _mm_loadu_pd(base);                                             \
        base += offset;                                                        \
        _s3c = _mm_loadu_pd(base);                                             \
        _r1 = _mm256_insertf128_pd(_mm256_castpd128_pd256(_s1c),_s3c,1);       \
        base += offset;                                                        \
        _s4c = _mm_loadu_pd(base);                                             \
        _r2 = _mm256_insertf128_pd(_mm256_castpd128_pd256(_s2c),_s4c,1);       \
        dest1 = _mm256_shuffle_pd(_r1, _r2, 0x0);                              \
        dest2 = _mm256_shuffle_pd(_r1, _r2, 0xF);                              \
    }

/**
 * @brief load 8 32 bit single precision floating point number from memory
 * addresses specified by base address into given 256 bit register.
 * Operation : 8 MOV, 1 OTHERS, 6 PERM
 * Sample load of 8 real floating point with
 * Ar residing in base address
 * Br residing in base + offset address
 * Cr residing in base + (2 * offset) address
 * Dr residing in base + (3 * offset) address
 * Er residing in base + (4 * offset) address
 * Fr residing in base + (5 * offset) address
 * Gr residing in base + (6 * offset) address
 * Hr residing in base + (7 * offset) address.
 * dest = |32-bit-|32-bit-|32-bit-|32-bit-|32-bit-|32-bit-|32-bit-|32-bit-|
 * dest = |--Ar---|--Br---|--Cr---|--Dr---|--Er---|--Fr---|--Gr---|--Hr---|
 */
#define LDR_256_S(base, offset, dest)                                          \
    {                                                                          \
        __m128 _low, _high, _tmp, _tmp2;                                       \
        _low = _mm_load_ss(base);                                              \
        _high = _mm_load_ss(base + (offset));                                  \
        _low = _mm_unpacklo_ps(_low, _high);                                   \
        _tmp = _mm_load_ss(base + (offset << 1));                              \
        _high = _mm_load_ss(base + (offset * 3));                              \
        _tmp = _mm_unpacklo_ps(_tmp, _high);                                   \
        _low = _mm_shuffle_ps(_low, _tmp, 0x44);                               \
        _high = _mm_load_ss(base + (offset << 2));                             \
        _tmp = _mm_load_ss(base + (offset * 5));                               \
        _high = _mm_unpacklo_ps(_high, _tmp);                                  \
        _tmp2 = _mm_load_ss(base + (offset * 6));                              \
        _tmp = _mm_load_ss(base + (offset * 7));                               \
        _tmp2 = _mm_unpacklo_ps(_tmp2, _tmp);                                  \
        _tmp = _mm_shuffle_ps(_high, _tmp2, 0x44);                             \
        dest = _mm256_insertf128_ps(_mm256_castps128_ps256(_low), _tmp, 1);    \
    }

/**
 * @brief Load 8 complex numbers (each with 32-bit double precision real and
 * imaginary parts) from memory into 2 256-bit registers. The real parts are
 * loaded into one register and the imaginary parts into another.
 * Operation : 8 MOV, 2 OTHERS, 5 PERM
 * Sample load of 8 complex floating point with
 * A(Ar, Ai) residing in base address
 * B(Br, Bi) residing in base + offset address
 * C(Cr, Ci) residing in base + (2 * offset) address
 * D(Dr, Di) residing in base + (3 * offset) address
 * E(Er, Ei) residing in base + (4 * offset) address
 * F(Fr, Fi) residing in base + (5 * offset) address
 * G(Gr, Gi) residing in base + (6 * offset) address
 * H(Hr, Hi) residing in base + (7 * offset) address.
 * dest  = |32-bit-|32-bit-|32-bit-|32-bit-|32-bit-|32-bit-|32-bit-|32-bit-|
 * dest1 = |--Ar---|--Br---|--Cr---|--Dr---|--Er---|--Fr---|--Gr---|--Hr---|
 * dest2 = |--Ai---|--Bi---|--Ci---|--Di---|--Ei---|--Fi---|--Gi---|--Hi---|
 */
#define LDRI_2x256_S(base, offset, dest1, dest2)                               \
    {                                                                          \
        __m128 _tmp, _tmp1, _tmp2, _tmp3;                                      \
        __m256 _r1, _r2;                                                       \
        _tmp = _mm_loadu_ps(base);                                             \
        base += offset;                                                        \
        _tmp1 = _mm_loadu_ps(base);                                            \
        _tmp = _mm_shuffle_ps(_tmp, _tmp1, 0x44);                              \
        base += offset;                                                        \
        _tmp1 = _mm_loadu_ps(base);                                            \
        base += offset;                                                        \
        _tmp2 = _mm_loadu_ps(base);                                            \
        _tmp1 = _mm_shuffle_ps(_tmp1, _tmp2, 0x44);                            \
        base += offset;                                                        \
        _tmp2 = _mm_loadu_ps(base);                                            \
        base += offset;                                                        \
        _tmp3 = _mm_loadu_ps(base);                                            \
        _tmp2 = _mm_shuffle_ps(_tmp2, _tmp3, 0x44);                            \
        base += offset;                                                        \
        _tmp3 = _mm_loadu_ps(base);                                            \
        base += offset;                                                        \
        _tmp3 = _mm_loadh_pi(_tmp3, (__m64 *)base);                            \
        _r1 = _mm256_insertf128_ps(_mm256_castps128_ps256(_tmp), _tmp2, 1);    \
        _r2 = _mm256_insertf128_ps(_mm256_castps128_ps256(_tmp1), _tmp3, 1);   \
        dest1 = _mm256_shuffle_ps(_r1, _r2, 0x88);                             \
        dest2 = _mm256_shuffle_ps(_r1, _r2, 0xDD);                             \
    }

/**
 * @brief store 4 64 bit double precision floating point number from 256 bit
 * register into memory addresses specified by base address
 * Operation : 4 MOV(store), 1 OTHERS
 * Sample store of 4 real floating point with
 * Ar stored in base address
 * Br stored in base + offset address
 * Cr stored in base + (2 * offset) address and
 * Dr stored in base + (3 * offset) address.
 * src = |-----64-bit-----|-----64-bit-----|-----64-bit-----|-----64-bit-----|
 * src = |-------Ar-------|-------Br-------|-------Cr-------|-------Dr-------|
 */
#define STR_256_D(base, offset, src)                                           \
    {                                                                          \
        __m128d _high = _mm256_extractf128_pd(src, 1);                         \
        __m128d _low = _mm256_castpd256_pd128(src);                            \
        _mm_storel_pd(base, _low);                                             \
        base += offset;                                                        \
        _mm_storeh_pd(base, _low);                                             \
        base += offset;                                                        \
        _mm_storel_pd(base, _high);                                            \
        base += offset;                                                        \
        _mm_storeh_pd(base, _high);                                            \
    }

/**
 * @brief Store 4 complex numbers (each with 64-bit double precision real and
 * imaginary parts) from 2 256-bit registers into memory. The real parts are
 * taken from one register, and the imaginary parts from another, and stored
 * into the memory addresses specified by the base address.
 * Operation : 4 MOV(store), 2 OTHERS, 2 PERM
 * Sample store of 4 complex floating point with
 * A(Ar,Ai) stored in base address
 * B(Br,Bi) stored in base + offset address
 * C(Cr,Ci) stored in base + (2 * offset) address and
 * D(Dr,Di) stored in base + (3 * offset) address.
 * src  = |-----64-bit-----|-----64-bit-----|-----64-bit-----|-----64-bit-----|
 * src1 = |-------Ar-------|-------Br-------|-------Cr-------|-------Dr-------|
 * src2 = |-------Ai-------|-------Bi-------|-------Ci-------|-------Di-------|
 */
#define STRI_2x256_D(base, offset, src1, src2)                                 \
    {                                                                          \
        __m256d _s1_s3c, _s2_s4c;                                              \
        _s1_s3c = _mm256_shuffle_pd(src1, src2, 0x0);                          \
        __m128d _s3c = _mm256_extractf128_pd(_s1_s3c, 1);                      \
        __m128d _s1c = _mm256_castpd256_pd128(_s1_s3c);                        \
        _mm_storeu_pd(base, _s1c);                                             \
        _s2_s4c = _mm256_shuffle_pd(src1, src2, 0xF);                          \
        __m128d _s4c = _mm256_extractf128_pd(_s2_s4c, 1);                      \
        __m128d _s2c = _mm256_castpd256_pd128(_s2_s4c);                        \
        base += offset;                                                        \
        _mm_storeu_pd(base, _s2c);                                             \
        base += offset;                                                        \
        _mm_storeu_pd(base, _s3c);                                             \
        base += offset;                                                        \
        _mm_storeu_pd(base, _s4c);                                             \
    }

/**
 * @brief store 8 32 bit single precision floating point number from 256 bit
 * register into memory addresses specified by base address into
 * Operation : 8 MOV(store), 1 OTHERS, 6 PERM
 * Sample store of 8 real floating point with
 * Ar stored in base address
 * Br stored in base + offset address
 * Cr stored in base + (2 * offset) address
 * Dr stored in base + (3 * offset) address
 * Er stored in base + (4 * offset) address
 * Fr stored in base + (5 * offset) address
 * Gr stored in base + (6 * offset) address
 * Hr stored in base + (7 * offset) address.
 * src = |32-bit-|32-bit-|32-bit-|32-bit-|32-bit-|32-bit-|32-bit-|32-bit-|
 * src = |--Ar---|--Br---|--Cr---|--Dr---|--Er---|--Fr---|--Gr---|--Hr---|
 */
#define STR_256_S(base, offset, src)                                           \
    {                                                                          \
        __m128 _high = _mm256_extractf128_ps(src, 1);                          \
        __m128 _low = _mm256_castps256_ps128(src);                             \
        _mm_store_ss(base, _low);                                              \
        base += offset;                                                        \
        __m128 _tmp = _mm_permute_ps(_low, 0xE1);                              \
        _mm_store_ss(base, _tmp);                                              \
        base += offset;                                                        \
        _tmp = _mm_permute_ps(_low, 0xD2);                                     \
        _mm_store_ss(base, _tmp);                                              \
        base += offset;                                                        \
        _tmp = _mm_permute_ps(_low, 0x93);                                     \
        _mm_store_ss(base, _tmp);                                              \
        base += offset;                                                        \
        _mm_store_ss(base, _high);                                             \
        base += offset;                                                        \
        _tmp = _mm_permute_ps(_high, 0xE1);                                    \
        _mm_store_ss(base, _tmp);                                              \
        base += offset;                                                        \
        _tmp = _mm_permute_ps(_high, 0XD2);                                    \
        _mm_store_ss(base, _tmp);                                              \
        base += offset;                                                        \
        _tmp = _mm_permute_ps(_high, 0X93);                                    \
        _mm_store_ss(base, _tmp);                                              \
    }

/**
 * @brief Store 8 complex numbers (each with 32-bit double precision real and
 * imaginary parts) from 2 256-bit registers into memory. The real parts are
 * taken from one register, and the imaginary parts from another, and stored
 * into the memory addresses specified by the base address.
 * Operation : 8 MOV(store), 2 OTHERS, 2 PERM
 * Sample store of 8 complex floating point with
 * A(Ar, Ai) stored in base address
 * B(Br, Bi) stored in base + offset address
 * C(Cr, Ci) stored in base + (2 * offset) address
 * D(Dr, Di) stored in base + (3 * offset) address
 * E(Er, Ei) stored in base + (4 * offset) address
 * F(Fr, Fi) stored in base + (5 * offset) address
 * G(Gr, Gi) stored in base + (6 * offset) address
 * H(Hr, Hi) stored in base + (7 * offset) address.
 * src  = |32-bit-|32-bit-|32-bit-|32-bit-|32-bit-|32-bit-|32-bit-|32-bit-|
 * src1 = |--Ar---|--Br---|--Cr---|--Dr---|--Er---|--Fr---|--Gr---|--Hr---|
 * src2 = |--Ai---|--Bi---|--Ci---|--Di---|--Ei---|--Fi---|--Gi---|--Hi---|
 */
#define STRI_2x256_S(base, offset, src1, src2)                                 \
    {                                                                          \
        __m256 _s1256c, _s3478c;                                               \
        _s1256c = _mm256_unpacklo_ps(src1, src2);                              \
        _s3478c = _mm256_unpackhi_ps(src1, src2);                              \
        __m128 _s56c = _mm256_extractf128_ps(_s1256c, 1);                      \
        __m128 _s12c = _mm256_castps256_ps128(_s1256c);                        \
        _mm_storel_pi((__m64 *)base, _s12c);                                   \
        base += offset;                                                        \
        _mm_storeh_pi((__m64 *)base, _s12c);                                   \
        __m128 _s78c = _mm256_extractf128_ps(_s3478c, 1);                      \
        __m128 _s34c = _mm256_castps256_ps128(_s3478c);                        \
        base += offset;                                                        \
        _mm_storel_pi((__m64 *)base, _s34c);                                   \
        base += offset;                                                        \
        _mm_storeh_pi((__m64 *)base, _s34c);                                   \
        base += offset;                                                        \
        _mm_storel_pi((__m64 *)base, _s56c);                                   \
        base += offset;                                                        \
        _mm_storeh_pi((__m64 *)base, _s56c);                                   \
        base += offset;                                                        \
        _mm_storel_pi((__m64 *)base, _s78c);                                   \
        base += offset;                                                        \
        _mm_storeh_pi((__m64 *)base, _s78c);                                   \
    }

/**
 * @brief prepare -0.0 for negation.
 */
union zero_negate_256
{
    unsigned u[8];
    __m256 s;
    __m256d d;
};

static const union zero_negate_256
        _negate_256_f = {{ 0x80000000, 0x80000000, 0x80000000, 0x80000000,
                           0x80000000, 0x80000000, 0x80000000, 0x80000000 }};
static const union zero_negate_256
        _negate_256_d = {{ 0x00000000, 0x80000000, 0x00000000, 0x80000000,
                           0x00000000, 0x80000000, 0x00000000, 0x80000000 }};

/**
 * @brief take negation of a 256-bit vector __m256d A => -A
 * Operation : 1 OTHERS(xor)
 */
#define NEGATE_256_S(x) _mm256_xor_ps(_negate_256_f.s, x)
#define NEGATE_256_D(x) _mm256_xor_pd(_negate_256_d.d, x)

#endif // AOCLFFTZ_R2HC_SIMD_AVX256_H
