// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file r2hc_simd_avx512.h
 *
 *  @brief List of common operations and macros used for R2HC AVX512 simd
 *  kernels.
 *
 *  This file contains the category-wise macros and common functions used
 *  in the AVX512 R2HC kernel variants
 *
 *  @author Jeya R
 */

#ifndef AOCLFFTZ_R2HC_SIMD_AVX512_H
#define AOCLFFTZ_R2HC_SIMD_AVX512_H

#include "core/kernels/simd_includes/r2hc_simd_avx256.h"

#define NUM_SETS_REAL_512_S 16
#define NUM_SETS_REAL_512_D 8

/************************ MACRO FUNCTIONS ************************/

/**
 * @brief load 8 64 bit double precision floating point number from memory
 * addresses specified by base address into 512 bit register.
 * Sample load of 8 real floating point with
 * Ar residing in base address
 * Br residing in base + offset address
 * Cr residing in base + (2 * offset) address
 * Dr residing in base + (3 * offset) address
 * Er residing in base + (4 * offset) address
 * Fr residing in base + (5 * offset) address
 * Gr residing in base + (6 * offset) address
 * Hr residing in base + (7 * offset) address.
 * dest = |64-bit-|64-bit-|64-bit-|64-bit-|64-bit-|64-bit-|64-bit-|64-bit-|
 * dest = |--Ar---|--Br---|--Cr---|--Dr---|--Er---|--Fr---|--Gr---|--Hr---|
 * Operation : 8 MOV(load), 3 OTHERS
 */
#define LDR_512_D(base, offset, dest)                                          \
    {                                                                          \
        __m128d _low, _high;                                                   \
        __m256d _low256, _high256;                                             \
        _low = _mm_load_sd(base);                                              \
        base += offset;                                                        \
        _low = _mm_loadh_pd(_low, base);                                       \
        base += offset;                                                        \
        _high = _mm_load_sd(base);                                             \
        base += offset;                                                        \
        _high = _mm_loadh_pd(_high, base);                                     \
        _low256 = _mm256_insertf128_pd(_mm256_castpd128_pd256(_low), _high, 1);\
        base += offset;                                                        \
        _low = _mm_load_sd(base);                                              \
        base += offset;                                                        \
        _low = _mm_loadh_pd(_low, base);                                       \
        base += offset;                                                        \
        _high = _mm_load_sd(base);                                             \
        base += offset;                                                        \
        _high = _mm_loadh_pd(_high, base);                                     \
        _high256 = _mm256_insertf128_pd(_mm256_castpd128_pd256(_low) ,_high,1);\
        dest = _mm512_insertf64x4(_mm512_castpd256_pd512(_low256), _high256,1);\
    }

/**
 * @brief load 8 complex point of 64 bit double precision floating point number
 * from memory addresses specified by base address into given 2 512 bit register
 * Sample load of 8 complex floating point with
 * A(Ar, Ai) residing in base address
 * B(Br, Bi) residing in base + offset address
 * C(Cr, Ci) residing in base + (2 * offset) address
 * D(Dr, Di) residing in base + (3 * offset) address
 * E(Er, Ei) residing in base + (4 * offset) address
 * F(Fr, Fi) residing in base + (5 * offset) address
 * G(Gr, Gi) residing in base + (6 * offset) address
 * H(Hr, Hi) residing in base + (7 * offset) address.
 * dest  = |64-bit-|64-bit-|64-bit-|64-bit-|64-bit-|64-bit-|64-bit-|64-bit-|
 * dest1 = |--Ar---|--Br---|--Cr---|--Dr---|--Er---|--Fr---|--Gr---|--Hr---|
 * dest2 = |--Ai---|--Bi---|--Ci---|--Di---|--Ei---|--Fi---|--Gi---|--Hi---|
 * Operation : 8 MOV(load), 6 OTHERS, 2 PERM
 */
#define LDRI_2x512_D(base, offset, dest1, dest2)                               \
    {                                                                          \
        __m128d _s1c, _s2c, _s3c, _s4c;                                        \
        __m256d _r1, _r2;                                                      \
        __m512d _c1, _c2;                                                      \
        _s1c = _mm_loadu_pd(base);                                             \
        base += offset;                                                        \
        _s2c = _mm_loadu_pd(base);                                             \
        base += offset;                                                        \
        _s3c = _mm_loadu_pd(base);                                             \
        _r1 = _mm256_insertf128_pd(_mm256_castpd128_pd256(_s1c), _s3c, 1);     \
        base += offset;                                                        \
        _s4c = _mm_loadu_pd(base);                                             \
        base += offset;                                                        \
        _r2 = _mm256_insertf128_pd(_mm256_castpd128_pd256(_s2c), _s4c, 1);     \
        __m128d _s5c, _s6c, _s7c, _s8c;                                        \
        __m256d _r3, _r4;                                                      \
        _s5c = _mm_loadu_pd(base);                                             \
        base += offset;                                                        \
        _s6c = _mm_loadu_pd(base);                                             \
        base += offset;                                                        \
        _s7c = _mm_loadu_pd(base);                                             \
        _r3 = _mm256_insertf128_pd(_mm256_castpd128_pd256(_s5c), _s7c, 1);     \
        base += offset;                                                        \
        _s8c = _mm_loadu_pd(base);                                             \
        _r4 = _mm256_insertf128_pd(_mm256_castpd128_pd256(_s6c), _s8c, 1);     \
        _c1 = _mm512_insertf64x4(_mm512_castpd256_pd512(_r1), _r3, 1);         \
        _c2 = _mm512_insertf64x4(_mm512_castpd256_pd512(_r2), _r4, 1);         \
        dest1 = _mm512_shuffle_pd(_c1, _c2, 0x0);                              \
        dest2 = _mm512_shuffle_pd(_c1, _c2, 0xFF);                             \
    }

/**
 * @brief load 16 32 bit single precision floating point number from memory
 * addresses specified by base address into given 512 bit register.
 * Sample load of 16 real floating point with
 * Ar residing in base address
 * Br residing in base + offset address
 * Cr residing in base + (2 * offset) address
 * Dr residing in base + (3 * offset) address
 * Er residing in base + (4 * offset) address
 * Fr residing in base + (5 * offset) address
 * Gr residing in base + (6 * offset) address
 * Hr residing in base + (7 * offset) address.
 * Ir residing in base + (8 * offset)address
 * Jr residing in base + (9 * offset) address
 * Kr residing in base + (10 * offset) address
 * Lr residing in base + (11 * offset) address
 * Mr residing in base + (12 * offset) address
 * Nr residing in base + (13 * offset) address
 * Or residing in base + (14 * offset) address
 * Pr residing in base + (15 * offset) address.
 * dest = |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
 * dest = |Ar-|Br-|Cr-|Dr-|Er-|Fr-|Gr-|Hr-|Ir-|Jr-|Kr-|Lr-|Mr-|Nr-|Or-|Pr-|
 * Operation : 16 MOV(load), 3 OTHERS, 12 PERM
 */
#define LDR_512_S(base, offset, dest)                                          \
    {                                                                          \
        __m128 _low, _high, _tmp, _tmp2;                                       \
        __m256 _low256, _high256;                                              \
        _low = _mm_load_ss(base);                                              \
        _high = _mm_load_ss(base + (offset));                                  \
        _low = _mm_shuffle_ps(_low, _high, 0x44);                              \
        _tmp = _mm_load_ss(base + (offset << 1));                              \
        _high = _mm_load_ss(base + (offset * 3));                              \
        _tmp = _mm_shuffle_ps(_tmp, _high, 0x44);                              \
        _low = _mm_shuffle_ps(_low, _tmp, 0x88);                               \
        _high = _mm_load_ss(base + (offset << 2));                             \
        _tmp = _mm_load_ss(base + (offset * 5));                               \
        _high = _mm_shuffle_ps(_high, _tmp, 0x44);                             \
        _tmp2 = _mm_load_ss(base + (offset * 6));                              \
        _tmp = _mm_load_ss(base + (offset * 7));                               \
        _tmp2 = _mm_shuffle_ps(_tmp2, _tmp, 0x44);                             \
        _tmp = _mm_shuffle_ps(_high, _tmp2, 0x88);                             \
        _low256 = _mm256_insertf128_ps(_mm256_castps128_ps256(_low), _tmp, 1); \
        base = base + (offset << 3);                                           \
        _low = _mm_load_ss(base);                                              \
        _high = _mm_load_ss(base + (offset));                                  \
        _low = _mm_shuffle_ps(_low, _high, 0x44);                              \
        _tmp = _mm_load_ss(base + (offset << 1));                              \
        _high = _mm_load_ss(base + (offset * 3));                              \
        _tmp = _mm_shuffle_ps(_tmp, _high, 0x44);                              \
        _low = _mm_shuffle_ps(_low, _tmp, 0x88);                               \
        _high = _mm_load_ss(base + (offset << 2));                             \
        _tmp = _mm_load_ss(base + (offset * 5));                               \
        _high = _mm_shuffle_ps(_high, _tmp, 0x44);                             \
        _tmp2 = _mm_load_ss(base + (offset * 6));                              \
        _tmp = _mm_load_ss(base + (offset * 7));                               \
        _tmp2 = _mm_shuffle_ps(_tmp2, _tmp, 0x44);                             \
        _tmp = _mm_shuffle_ps(_high, _tmp2, 0x88);                             \
        _high256 = _mm256_insertf128_ps(_mm256_castps128_ps256(_low), _tmp, 1);\
        dest = _mm512_insertf32x8(_mm512_castps256_ps512(_low256),_high256, 1);\
    }

/**
 * @brief load 16 complex point of 32 bit single precision floating point number
 * from memory addresses specified by base address into given 2 512 bit register
 * Sample load of 16 complex floating point with
 * A(Ar, Ai) residing in base address
 * B(Br, Bi) residing in base + offset address
 * C(Cr, Ci) residing in base + (2 * offset) address
 * D(Dr, Di) residing in base + (3 * offset) address
 * E(Er, Ei) residing in base + (4 * offset) address
 * F(Fr, Fi) residing in base + (5 * offset) address
 * G(Gr, Gi) residing in base + (6 * offset) address
 * H(Hr, Hi) residing in base + (7 * offset) address.
 * I(Ir, Ii) residing in base + (8 * offset)address
 * J(Jr, Ji) residing in base + (9 * offset) address
 * K(Kr, Ki) residing in base + (10 * offset) address
 * L(Lr, Li) residing in base + (11 * offset) address
 * M(Mr, Mi) residing in base + (12 * offset) address
 * N(Nr, Ni) residing in base + (13 * offset) address
 * O(Or, Oi) residing in base + (14 * offset) address
 * P(Pr, Pi) residing in base + (15 * offset) address.
 * dest  = |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
 * dest1 = |Ar-|Br-|Cr-|Dr-|Er-|Fr-|Gr-|Hr-|Ir-|Jr-|Kr-|Lr-|Mr-|Nr-|Or-|Pr-|
 * dest2 = |Ai-|Bi-|Ci-|Di-|Ei-|Fi-|Gi-|Hi-|Ii-|Ji-|Ki-|Li-|Mi-|Ni-|Oi-|Pi-|
 * Operation : 16 MOV(load), 6 OTHERS, 10 PERM
 */
#define LDRI_2x512_S(base, offset, dest1, dest2)                               \
    {                                                                          \
        __m128 _tmp, _tmp1, _tmp2, _tmp3, _tmp4, _tmp5, _tmp6, _tmp7;          \
        __m256 _r1, _r2, _r3, _r4;                                             \
        __m512 _c1, _c2;                                                       \
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
        _tmp4 = _mm_loadu_ps(base);                                            \
        _tmp3 = _mm_shuffle_ps(_tmp3, _tmp4, 0x44);                            \
        base += offset;                                                        \
         _r1 = _mm256_insertf128_ps(_mm256_castps128_ps256(_tmp), _tmp2, 1);   \
         _r2 = _mm256_insertf128_ps(_mm256_castps128_ps256(_tmp1), _tmp3, 1);  \
         _tmp4 = _mm_loadu_ps(base);                                           \
         base += offset;                                                       \
         _tmp5 = _mm_loadu_ps(base);                                           \
         _tmp4 = _mm_shuffle_ps(_tmp4, _tmp5, 0x44);                           \
         base += offset;                                                       \
         _tmp5 = _mm_loadu_ps(base);                                           \
         base += offset;                                                       \
         _tmp6 = _mm_loadu_ps(base);                                           \
         _tmp5 = _mm_shuffle_ps(_tmp5, _tmp6, 0x44);                           \
         base += offset;                                                       \
         _tmp6 = _mm_loadu_ps(base);                                           \
         base += offset;                                                       \
         _tmp7 = _mm_loadu_ps(base);                                           \
         _tmp6 = _mm_shuffle_ps(_tmp6, _tmp7, 0x44);                           \
         base += offset;                                                       \
         _tmp7 = _mm_loadu_ps(base);                                           \
         base += offset;                                                       \
         _tmp7 = _mm_loadh_pi(_tmp7, (__m64 *)base);                           \
         _r3 = _mm256_insertf128_ps(_mm256_castps128_ps256(_tmp4), _tmp6, 1);  \
         _r4 = _mm256_insertf128_ps(_mm256_castps128_ps256(_tmp5), _tmp7, 1);  \
         _c1 = _mm512_insertf32x8(_mm512_castps256_ps512(_r1), _r3, 1);        \
         _c2 = _mm512_insertf32x8(_mm512_castps256_ps512(_r2), _r4, 1);        \
         dest1 = _mm512_shuffle_ps(_c1, _c2, 0x88);                            \
         dest2 = _mm512_shuffle_ps(_c1, _c2, 0xDD);                            \
    }

/**
 * @brief store 8 64 bit double precision floating point number from 512 bit
 * register into memory addresses specified by base address
 * Sample store of 8 real floating point with
 * Ar stored in base address
 * Br stored in base + offset address
 * Cr stored in base + (2 * offset) address
 * Dr stored in base + (3 * offset) address
 * Er stored in base + (4 * offset) address
 * Fr stored in base + (5 * offset) address
 * Gr stored in base + (6 * offset) address
 * Hr stored in base + (7 * offset) address.
 * src = |64-bit-|64-bit-|64-bit-|64-bit-|64-bit-|64-bit-|64-bit-|64-bit-|
 * src = |--Ar---|--Br---|--Cr---|--Dr---|--Er---|--Fr---|--Gr---|--Hr---|
 * Operation : 8 MOV(store), 3 OTHERS
 */
#define STR_512_D(base, offset, src)                                           \
    {                                                                          \
        __m256d _256high = _mm512_extractf64x4_pd(src, 1);                     \
        __m256d _256low = _mm512_castpd512_pd256(src);                         \
        __m128d _low, _high;                                                   \
        _high = _mm256_extractf128_pd(_256low, 1);                             \
        _low = _mm256_castpd256_pd128(_256low);                                \
        _mm_storel_pd(base, _low);                                             \
        base += offset;                                                        \
        _mm_storeh_pd(base, _low);                                             \
        base += offset;                                                        \
        _mm_storel_pd(base, _high);                                            \
        base += offset;                                                        \
        _mm_storeh_pd(base, _high);                                            \
        base += offset;                                                        \
        _high = _mm256_extractf128_pd(_256high, 1);                            \
        _low = _mm256_castpd256_pd128(_256high);                               \
        _mm_storel_pd(base, _low);                                             \
        base += offset;                                                        \
        _mm_storeh_pd(base, _low);                                             \
        base += offset;                                                        \
        _mm_storel_pd(base, _high);                                            \
        base += offset;                                                        \
        _mm_storeh_pd(base, _high);                                            \
    }

/**
 * @brief store 8 complex point of 64 bit double precision floating point number
 * from 2 512 bit register into memory addresses specified by base address
 * Sample store of 8 complex floating point with
 * A(Ar, Ai) stored in base address
 * B(Br, Bi) stored in base + offset address
 * C(Cr, Ci) stored in base + (2 * offset) address
 * D(Dr, Di) stored in base + (3 * offset) address
 * E(Er, Ei) stored in base + (4 * offset) address
 * F(Fr, Fi) stored in base + (5 * offset) address
 * G(Gr, Gi) stored in base + (6 * offset) address
 * H(Hr, Hi) stored in base + (7 * offset) address.
 * src  = |64-bit-|64-bit-|64-bit-|64-bit-|64-bit-|64-bit-|64-bit-|64-bit-|
 * src1 = |--Ar---|--Br---|--Cr---|--Dr---|--Er---|--Fr---|--Gr---|--Hr---|
 * src2 = |--Ai---|--Bi---|--Ci---|--Di---|--Ei---|--Fi---|--Gi---|--Hi---|
 * Operation : 8 MOV(store), 6 OTHERS, 2 PERM
 */
#define STRI_2x512_D(base, offset, src1, src2)                                 \
    {                                                                          \
        __m512d _s1_s3__s5_s7c, _s2_s4__s6_s8c;                                \
        _s1_s3__s5_s7c = _mm512_shuffle_pd(src1, src2, 0x0);                   \
        __m256d _s1_s3c, _s2_s4c, _s5_s7c, _s6_s8c;                            \
        _s5_s7c = _mm512_extractf64x4_pd(_s1_s3__s5_s7c, 0x1);                 \
        _s1_s3c = _mm512_castpd512_pd256(_s1_s3__s5_s7c);                      \
        __m128d _s7c = _mm256_extractf128_pd(_s5_s7c, 1);                      \
        __m128d _s5c = _mm256_castpd256_pd128(_s5_s7c);                        \
        __m128d _s3c = _mm256_extractf128_pd(_s1_s3c, 1);                      \
        __m128d _s1c = _mm256_castpd256_pd128(_s1_s3c);                        \
        _mm_storeu_pd(base, _s1c);                                             \
        _s2_s4__s6_s8c = _mm512_shuffle_pd(src1, src2, 0xFF);                  \
        _s6_s8c = _mm512_extractf64x4_pd(_s2_s4__s6_s8c, 1);                   \
        _s2_s4c = _mm512_castpd512_pd256(_s2_s4__s6_s8c);                      \
        __m128d _s8c = _mm256_extractf128_pd(_s6_s8c, 1);                      \
        __m128d _s6c = _mm256_castpd256_pd128(_s6_s8c);                        \
        __m128d _s4c = _mm256_extractf128_pd(_s2_s4c, 1);                      \
        __m128d _s2c = _mm256_castpd256_pd128(_s2_s4c);                        \
        base += offset;                                                        \
        _mm_storeu_pd(base, _s2c);                                             \
        base += offset;                                                        \
        _mm_storeu_pd(base, _s3c);                                             \
        base += offset;                                                        \
        _mm_storeu_pd(base, _s4c);                                             \
        base += offset;                                                        \
        _mm_storeu_pd(base, _s5c);                                             \
        base += offset;                                                        \
        _mm_storeu_pd(base, _s6c);                                             \
        base += offset;                                                        \
        _mm_storeu_pd(base, _s7c);                                             \
        base += offset;                                                        \
        _mm_storeu_pd(base, _s8c);                                             \
    }

/**
 * @brief store 16 32 bit single precision floating point number from 512 bit
 * register into memory addresses specified by base address
 * Sample store of 16 real floating point with
 * Ar stored in base address
 * Br stored in base + offset address
 * Cr stored in base + (2 * offset) address
 * Dr stored in base + (3 * offset) address
 * Er stored in base + (4 * offset) address
 * Fr stored in base + (5 * offset) address
 * Gr stored in base + (6 * offset) address
 * Hr stored in base + (7 * offset) address.
 * Ir stored in base + (8 * offset)address
 * Jr stored in base + (9 * offset) address
 * Kr stored in base + (10 * offset) address
 * Lr stored in base + (11 * offset) address
 * Mr stored in base + (12 * offset) address
 * Nr stored in base + (13 * offset) address
 * Or stored in base + (14 * offset) address
 * Pr stored in base + (15 * offset) address.
 * src = |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
 * src = |Ar-|Br-|Cr-|Dr-|Er-|Fr-|Gr-|Hr-|Ir-|Jr-|Kr-|Lr-|Mr-|Nr-|Or-|Pr-|
 * Operation : 16 MOV(store), 3 OTHERS, 12 PERM
 */
#define STR_512_S(base, offset, src)                                           \
    {                                                                          \
        __m256 _high256 = _mm512_extractf32x8_ps(src, 1);                      \
        __m256 _low256= _mm512_castps512_ps256(src);                           \
        __m128 _high, _low;                                                    \
        _high = _mm256_extractf128_ps(_low256, 1);                             \
        _low = _mm256_castps256_ps128(_low256);                                \
        _mm_store_ss(base, _low);                                              \
        base += offset;                                                        \
        __m128 _tmp = _mm_shuffle_ps(_low,_low, 0xE1);                         \
        _mm_store_ss(base, _tmp);                                              \
        base += offset;                                                        \
        _tmp = _mm_shuffle_ps(_low,_low, 0xD2);                                \
        _mm_store_ss(base, _tmp);                                              \
        base += offset;                                                        \
        _tmp = _mm_shuffle_ps(_low,_low, 0x93);                                \
        _mm_store_ss(base, _tmp);                                              \
        base += offset;                                                        \
        _mm_store_ss(base, _high);                                             \
        base += offset;                                                        \
        _tmp = _mm_shuffle_ps(_high,_high, 0xE1);                              \
        _mm_store_ss(base, _tmp);                                              \
        base += offset;                                                        \
        _tmp = _mm_shuffle_ps(_high,_high, 0XD2);                              \
        _mm_store_ss(base, _tmp);                                              \
        base += offset;                                                        \
        _tmp = _mm_shuffle_ps(_high, _high,0X93);                              \
        _mm_store_ss(base, _tmp);                                              \
        base += offset;                                                        \
        _high = _mm256_extractf128_ps(_high256, 1);                            \
        _low = _mm256_castps256_ps128(_high256);                               \
        _mm_store_ss(base, _low);                                              \
        base += offset;                                                        \
        _tmp = _mm_shuffle_ps(_low,_low, 0xE1);                                \
        _mm_store_ss(base, _tmp);                                              \
        base += offset;                                                        \
        _tmp = _mm_shuffle_ps(_low,_low, 0xD2);                                \
        _mm_store_ss(base, _tmp);                                              \
        base += offset;                                                        \
        _tmp = _mm_shuffle_ps(_low,_low, 0x93);                                \
        _mm_store_ss(base, _tmp);                                              \
        base += offset;                                                        \
        _mm_store_ss(base, _high);                                             \
        base += offset;                                                        \
        _tmp = _mm_shuffle_ps(_high,_high, 0xE1);                              \
        _mm_store_ss(base, _tmp);                                              \
        base += offset;                                                        \
        _tmp = _mm_shuffle_ps(_high,_high, 0XD2);                              \
        _mm_store_ss(base, _tmp);                                              \
        base += offset;                                                        \
        _tmp = _mm_shuffle_ps(_high,_high, 0X93);                              \
        _mm_store_ss(base, _tmp);                                              \
    }

/**
 * @brief store 16 complex point of 32 bit single precision floating point number
 * from 2 512 bit register into memory addresses specified by base address
 * Sample store of 16 complex floating point with
 * A(Ar, Ai) stored in base address
 * B(Br, Bi) stored in base + offset address
 * C(Cr, Ci) stored in base + (2 * offset) address
 * D(Dr, Di) stored in base + (3 * offset) address
 * E(Er, Ei) stored in base + (4 * offset) address
 * F(Fr, Fi) stored in base + (5 * offset) address
 * G(Gr, Gi) stored in base + (6 * offset) address
 * H(Hr, Hi) stored in base + (7 * offset) address.
 * I(Ir, Ii) stored in base + (8 * offset)address
 * J(Jr, Ji) stored in base + (9 * offset) address
 * K(Kr, Ki) stored in base + (10 * offset) address
 * L(Lr, Li) stored in base + (11 * offset) address
 * M(Mr, Mi) stored in base + (12 * offset) address
 * N(Nr, Ni) stored in base + (13 * offset) address
 * O(Or, Oi) stored in base + (14 * offset) address
 * P(Pr, Pi) stored in base + (15 * offset) address.
 * src  = |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
 * src1 = |Ar-|Br-|Cr-|Dr-|Er-|Fr-|Gr-|Hr-|Ir-|Jr-|Kr-|Lr-|Mr-|Nr-|Or-|Pr-|
 * src2 = |Ai-|Bi-|Ci-|Di-|Ei-|Fi-|Gi-|Hi-|Ii-|Ji-|Ki-|Li-|Mi-|Ni-|Oi-|Pi-|
 * Operation : 16 MOV(store), 6 OTHERS, 2 PERM
 */
#define STRI_2x512_S(base, offset, src1, src2)                                 \
    {                                                                          \
        __m512 _s12569101314c, _s347811121516c;                                \
        __m256 _s1256c, _s3478c, _s9101314c, _s11121516c;                      \
        _s12569101314c = _mm512_unpacklo_ps(src1, src2);                       \
        _s347811121516c = _mm512_unpackhi_ps(src1, src2);                      \
        _s9101314c = _mm512_extractf32x8_ps(_s12569101314c, 1);                \
        _s1256c = _mm512_castps512_ps256(_s12569101314c);                      \
        _s11121516c = _mm512_extractf32x8_ps(_s347811121516c, 1);              \
        _s3478c = _mm512_castps512_ps256(_s347811121516c);                     \
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
        base += offset;                                                        \
        __m128 _s1314c = _mm256_extractf128_ps(_s9101314c, 1);                 \
        __m128 _s910c = _mm256_castps256_ps128(_s9101314c);                    \
        _mm_storel_pi((__m64 *)base, _s910c);                                  \
        base += offset;                                                        \
        _mm_storeh_pi((__m64 *)base, _s910c);                                  \
        __m128 _s1516c = _mm256_extractf128_ps(_s11121516c, 1);                \
        __m128 _s1112c = _mm256_castps256_ps128(_s11121516c);                  \
        base += offset;                                                        \
        _mm_storel_pi((__m64 *)base, _s1112c);                                 \
        base += offset;                                                        \
        _mm_storeh_pi((__m64 *)base, _s1112c);                                 \
        base += offset;                                                        \
        _mm_storel_pi((__m64 *)base, _s1314c);                                 \
        base += offset;                                                        \
        _mm_storeh_pi((__m64 *)base, _s1314c);                                 \
        base += offset;                                                        \
        _mm_storel_pi((__m64 *)base, _s1516c);                                 \
        base += offset;                                                        \
        _mm_storeh_pi((__m64 *)base, _s1516c);                                 \
    }

/**
 * @brief prepare -0.0 for negation.
 */
union zero_negate_512
{
    unsigned u[16];
    __m512 s;
    __m512d d;
};

static const union zero_negate_512
        _negate_512_f = {{ 0x80000000, 0x80000000, 0x80000000, 0x80000000,
                           0x80000000, 0x80000000, 0x80000000, 0x80000000,
                           0x80000000, 0x80000000, 0x80000000, 0x80000000,
                           0x80000000, 0x80000000, 0x80000000, 0x80000000 }};
static const union zero_negate_512
        _negate_512_d = {{ 0x00000000, 0x80000000, 0x00000000, 0x80000000,
                           0x00000000, 0x80000000, 0x00000000, 0x80000000,
                           0x00000000, 0x80000000, 0x00000000, 0x80000000,
                           0x00000000, 0x80000000, 0x00000000, 0x80000000 }};

/**
 * @brief take negation of a 512-bit vector __m512d A => -A
 * Operation : 1 OTHERS(xor)
 */
#define NEGATE_512_S(x) _mm512_xor_ps(_negate_512_f.s, x)
#define NEGATE_512_D(x) _mm512_xor_pd(_negate_512_d.d, x)

#endif // AOCLFFTZ_R2HC_SIMD_AVX512_H
