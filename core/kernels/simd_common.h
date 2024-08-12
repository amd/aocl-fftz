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
#define GATHER2_128_S(base, offset, dest)                                      \
{                                                                              \
    dest = _mm_loadu_ps(base);                                                 \
    dest = _mm_loadh_pi(dest, (__m64 *)(base + offset));                       \
}

/**
 * @brief store two complex numbers(real,imaginary) of 32 bit single precision
 * floating point number from 128 bit register into memory addresses
 * specified by base address and offset.
 * Operations : 2 MOV(store)
 */
#define SCATTER2_128_S(base, offset, src)                                      \
{                                                                              \
    _mm_storel_pi((__m64 *)base, src);                                         \
    base += offset;                                                            \
    _mm_storeh_pi((__m64 *)base, src);                                         \
}

/**
 * @brief load a complex number(real,imaginary) of 32 bit single precision
 * floating point number from memory addresses specified by base address
 * into 128 bit register.
 * Operation : 1 MOV(load)
 */
#define LD_LOW_128_S(base, dest)                                               \
{                                                                              \
    dest = _mm_loadl_pi(_mm_setzero_ps(), (__m64 *)base);                      \
}

/**
 * @brief store a complex number(real,imaginary) of 32 bit single precision
 * floating point number from 128 bit register into memory addresses
 * specified by base address.
 * Operation : 1 MOV(store)
 */
#define ST_LOW_128_S(base, src)                                                \
{                                                                              \
    _mm_storel_pi((__m64 *)base, src);                                         \
}

/**
 * @brief load a complex number(real,imaginary) of 64 bit double precision
 * floating point number from memory addresses specified by base address
 * into 128 bit register.
 * Operation : 1 MOV(load)
 */
#define LD_128_D(base, dest)                                                   \
{                                                                              \
    dest = _mm_loadu_pd(base);                                                 \
}

/**
 * @brief store a complex number(real,imaginary) of 64 bit double precision
 * floating point number from 128 bit register into memory addresses
 * specified by base address.
 * Operation : 1 MOV(store)
 */
#define ST_128_D(base, src)                                                    \
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
#define GATHER4_256_S(base, offset, dest)                                      \
{                                                                              \
    __m128 _low, _high, _tmp;                                                  \
    _low = _mm_loadu_ps(base);                                                 \
    base += offset;                                                            \
    _tmp = _mm_loadu_ps(base);                                                 \
    _low = _mm_shuffle_ps(_low, _tmp, 68);                                     \
    base += offset;                                                            \
    _high = _mm_loadu_ps(base);                                                \
    base += offset;                                                            \
    _high = _mm_loadh_pi(_high, (__m64 *)base);                                \
    dest = _mm256_insertf128_ps(_mm256_castps128_ps256(_low), _high, 1);       \
}

/**
 * @brief store four complex numbers(real,imaginary) of 32 bit single precision
 * floating point number from 256 bit register into memory addresses
 * specified by base address and offset.
 * Operations : 4 MOV(store), 1 OTHERS(extract). Cast is excluded as it
 * will be a compile time operation.
 */
#define SCATTER4_256_S(base, offset, src)                                      \
{                                                                              \
    __m128 _high = _mm256_extractf128_ps(src, 1);                              \
    __m128 _low = _mm256_castps256_ps128(src);                                 \
    _mm_storel_pi((__m64 *)base, _low);                                        \
    base += offset;                                                            \
    _mm_storeh_pi((__m64 *)base, _low);                                        \
    base += offset;                                                            \
    _mm_storel_pi((__m64 *)base, _high);                                       \
    base += offset;                                                            \
    _mm_storeh_pi((__m64 *)base, _high);                                       \
}

/**
 * @brief load two complex numbers(real,imaginary) of 64 bit double precision
 * floating point number from memory addresses specified by base address
 * and offset into 256 bit register.
 * Operations : 2 MOV(load), 1 OTHERS(insert). Cast is excluded as it
 * will be a compile time operation.
 */
#define GATHER2_256_D(base, offset, dest)                                      \
{                                                                              \
    __m128d _low, _high;                                                       \
    _low = _mm_loadu_pd(base);                                                 \
    base += offset;                                                            \
    _high = _mm_loadu_pd(base);                                                \
    dest = _mm256_insertf128_pd(_mm256_castpd128_pd256(_low), _high, 1);       \
}

/**
 * @brief store two complex numbers(real,imaginary) of 64 bit double precision
 * floating point number from 256 bit register into memory addresses
 * specified by base address and offset.
 * Operations : 2 MOV(store), 1 OTHERS(extract). Cast is excluded as it
 * will be a compile time operation.
 */
#define SCATTER2_256_D(base, offset, src)                                      \
{                                                                              \
    __m128d _high = _mm256_extractf128_pd(src, 1);                             \
    __m128d _low = _mm256_castpd256_pd128(src);                                \
    _mm_storeu_pd(base, _low);                                                 \
    base += offset;                                                            \
    _mm_storeu_pd(base, _high);                                                \
}

/**
 * @brief load eight complex numbers(real,imaginary) of 32 bit single precision
 * floating point number from memory addresses specified by base address
 * and offset, into 512 bit register.
 * Operations : 8 MOV(load), 3 PERM(shuffle), 3 OTHERS(insert).
 * Cast is excluded as it will be a compile time operation.
 */
#define GATHER8_512_S(base, offset, dest)                                      \
{                                                                              \
    __m128 _low, _high, _tmp;                                                  \
    __m256 _256low, _256high;                                                  \
    _low = _mm_loadu_ps(base);                                                 \
    base += offset;                                                            \
    _tmp = _mm_loadu_ps(base);                                                 \
    _low = _mm_shuffle_ps(_low, _tmp, 68);                                     \
    base += offset;                                                            \
    _high = _mm_loadu_ps(base);                                                \
    base += offset;                                                            \
    _tmp = _mm_loadu_ps(base);                                                 \
    _high = _mm_shuffle_ps(_high, _tmp, 68);                                   \
    _256low = _mm256_insertf128_ps(_mm256_castps128_ps256(_low), _high, 1);    \
    base += offset;                                                            \
    _low = _mm_loadu_ps(base);                                                 \
    base += offset;                                                            \
    _tmp = _mm_loadu_ps(base);                                                 \
    _low = _mm_shuffle_ps(_low, _tmp, 68);                                     \
    base += offset;                                                            \
    _high = _mm_loadu_ps(base);                                                \
    base += offset;                                                            \
    _high = _mm_loadh_pi(_high, (__m64 *)base);                                \
    _256high = _mm256_insertf128_ps(_mm256_castps128_ps256(_low), _high, 1);   \
    dest = _mm512_insertf32x8(_mm512_castps256_ps512(_256low), _256high, 1);   \
}

/**
 * @brief store eight complex numbers(real,imaginary) of 32 bit single precision
 * floating point number from 512 bit register into memory addresses
 * specified by base address and offset.
 * Operations : 8 MOV(store), 3 OTHERS(extract).
 * Cast is excluded as it will be a compile time operation.
 */
#define SCATTER8_512_S(base, offset, src)                                      \
{                                                                              \
    __m256 _256high = _mm512_extractf32x8_ps(src, 1);                          \
    __m256 _256low = _mm512_castps512_ps256(src);                              \
    __m128 _high, _low;                                                        \
    _high = _mm256_extractf128_ps(_256low, 1);                                 \
    _low = _mm256_castps256_ps128(_256low);                                    \
    _mm_storel_pi((__m64 *)base, _low);                                        \
    base += offset;                                                            \
    _mm_storeh_pi((__m64 *)base, _low);                                        \
    base += offset;                                                            \
    _mm_storel_pi((__m64 *)base, _high);                                       \
    base += offset;                                                            \
    _mm_storeh_pi((__m64 *)base, _high);                                       \
    _high = _mm256_extractf128_ps(_256high, 1);                                \
    _low = _mm256_castps256_ps128(_256high);                                   \
    base += offset;                                                            \
    _mm_storel_pi((__m64 *)base, _low);                                        \
    base += offset;                                                            \
    _mm_storeh_pi((__m64 *)base, _low);                                        \
    base += offset;                                                            \
    _mm_storel_pi((__m64 *)base, _high);                                       \
    base += offset;                                                            \
    _mm_storeh_pi((__m64 *)base, _high);                                       \
}

/**
 * @brief load four complex numbers(real,imaginary) of 64 bit double precision
 * floating point number from memory addresses specified by base address
 * and offset into 512 bit register.
 * Operations : 4 MOV(load), 3 OTHERS(insert).
 * Cast is excluded as it will be a compile time operation
 */
#define GATHER4_512_D(base, offset, dest)                                      \
{                                                                              \
    __m128d _low, _high;                                                       \
    __m256d _256low, _256high;                                                 \
    _low = _mm_loadu_pd(base);                                                 \
    base += offset;                                                            \
    _high = _mm_loadu_pd(base);                                                \
    _256low = _mm256_insertf128_pd(_mm256_castpd128_pd256(_low), _high, 1);    \
    base += offset;                                                            \
    _low = _mm_loadu_pd(base);                                                 \
    base += offset;                                                            \
    _high = _mm_loadu_pd(base);                                                \
    _256high = _mm256_insertf128_pd(_mm256_castpd128_pd256(_low), _high, 1);   \
    dest = _mm512_insertf64x4(_mm512_castpd256_pd512(_256low), _256high, 1);   \
}

/**
 * @brief store four complex numbers(real,imaginary) of 64 bit double precision
 * floating point number from 512 bit register into memory addresses
 * specified by base address and offset.
 * Operations : 4 MOV(store), 3 OTHERS(extract).
 * Cast is excluded as it will be a compile time operation.
 */
#define SCATTER4_512_D(base, offset, src)                                      \
{                                                                              \
    __m256d _m256high = _mm512_extractf64x4_pd(src, 1);                        \
    __m256d _m256low = _mm512_castpd512_pd256(src);                            \
    __m128d _low, _high;                                                       \
    _high = _mm256_extractf128_pd(_m256low, 1);                                \
    _low = _mm256_castpd256_pd128(_m256low);                                   \
    _mm_storeu_pd(base, _low);                                                 \
    base += offset;                                                            \
    _mm_storeu_pd(base, _high);                                                \
    base += offset;                                                            \
    _high = _mm256_extractf128_pd(_m256high, 1);                               \
    _low = _mm256_castpd256_pd128(_m256high);                                  \
    _mm_storeu_pd(base, _low);                                                 \
    base += offset;                                                            \
    _mm_storeu_pd(base, _high);                                                \
}

/**
 * @brief interchanges the real and imaginary values in a 128 bit register
 * for single precision floating point.
 * Operation : 1 PERM(shuffle)
 */
#define SWAP_RI_128_S(val) _mm_shuffle_ps(val, val, 177)

/**
 * @brief interchanges the real and imaginary values in a 128 bit register
 * for double precision floating point.
 * Operation : 1 PERM(shuffle)
 */
#define SWAP_RI_128_D(val) _mm_shuffle_pd(val, val, 1)

/**
 * @brief interchanges the real and imaginary values in a 256 bit register
 * for single precision floating point using the control value in the last
 * integer argument(b 10 11 00 01)
 * Operation : 1 PERM
 */
#define SWAP_RI_256_S(val) _mm256_permute_ps(val, 177)

/**
 * @brief interchanges the real and imaginary values in a 256 bit register
 * for double precision floating point using the control value in the last
 * integer argument(b 01 01)
 * Operation : 1 PERM
 */
#define SWAP_RI_256_D(val) _mm256_permute_pd(val, 5)

/**
 * @brief interchanges the real and imaginary values in a 512 bit register
 * for single precision floating point using the control value in the last
 * integer argument(b 10 11 00 01)
 */
#define SWAP_RI_512_S(val) _mm512_permute_ps(val, 177)

/**
 * @brief interchanges the real and imaginary values in a 512 bit register
 * for double precision floating point using the control value in the last
 * integer argument(b 01 01 01 01)
 */
#define SWAP_RI_512_D(val) _mm512_permute_pd(val, 85)

/**
 * @brief implies the number of sets that can be processed in parallel.
 * Computed using Register width /(2* sizeof(floating point)
 */
#define NUM_SETS_128_S 2
#define NUM_SETS_128_D 1
#define NUM_SETS_256_S 4
#define NUM_SETS_256_D 2
#define NUM_SETS_512_S 8
#define NUM_SETS_512_D 4

/**
 * @brief prepare -0.0 for complex conjugate.
 */
union zero_conj_128
{
    unsigned u[4];
    __m128 s;
    __m128d d;
};
union zero_conj_256
{
    unsigned u[8];
    __m256 s;
    __m256d d;
};

static const union zero_conj_128
            _conj_128_f = {{ 0x00000000, 0x80000000, 0x00000000, 0x80000000 }};
static const union zero_conj_128
            _conj_128_d = {{ 0x00000000, 0x00000000, 0x00000000, 0x80000000 }};

static const union zero_conj_128
        _neg_zero_128_f = {{ 0x80000000, 0x80000000, 0x80000000, 0x80000000 }};
static const union zero_conj_128
        _neg_zero_128_d = {{ 0x00000000, 0x80000000, 0x00000000, 0x80000000 }};

static const union zero_conj_256
            _conj_256_f = {{ 0x00000000, 0x80000000, 0x00000000, 0x80000000,
                             0x00000000, 0x80000000, 0x00000000, 0x80000000 }};
static const union zero_conj_256
            _conj_256_d = {{ 0x00000000, 0x00000000, 0x00000000, 0x80000000,
                             0x00000000, 0x00000000, 0x00000000, 0x80000000 }};

#if defined (__unix__) || (defined (_WINDOWS) && defined (__AVX512F__))

union zero_conj_512
{
    unsigned u[16];
    __m512 s;
    __m512d d;
};

static const union zero_conj_512
            _conj_512_f = {{ 0x00000000, 0x80000000, 0x00000000, 0x80000000,
                             0x00000000, 0x80000000, 0x00000000, 0x80000000,
                             0x00000000, 0x80000000, 0x00000000, 0x80000000,
                             0x00000000, 0x80000000, 0x00000000, 0x80000000 }};
static const union zero_conj_512
            _conj_512_d = {{ 0x00000000, 0x00000000, 0x00000000, 0x80000000,
                             0x00000000, 0x00000000, 0x00000000, 0x80000000,
                             0x00000000, 0x00000000, 0x00000000, 0x80000000,
                             0x00000000, 0x00000000, 0x00000000, 0x80000000 }};
#endif

/**
 * @brief take conjugate of the complex number A+Bi => A-Bi
 * Operation : 1 OTHERS(xor)
 */
#define CONJ_128_S(x) _mm_xor_ps(_conj_128_f.s, x)
#define CONJ_128_D(x) _mm_xor_pd(_conj_128_d.d, x)
#define CONJ_256_S(x) _mm256_xor_ps(_conj_256_f.s, x)
#define CONJ_256_D(x) _mm256_xor_pd(_conj_256_d.d, x)
#define CONJ_512_S(x) _mm512_xor_ps(_conj_512_f.s, x)
#define CONJ_512_D(x) _mm512_xor_pd(_conj_512_d.d, x)
/**
 * @brief alternatively performs addition & subtraction in a 128 bit register
 * for single precision floating point.
 * Operation : 2 PERM, 1 ADD
 */
#define SUBADD_128_S(a, b, c)                                                  \
{                                                                              \
    a = SWAP_RI_128_S(a);                                                      \
    c = SWAP_RI_128_S(_mm_addsub_ps(a, b));                                    \
}

/**
 * @brief alternatively performs addition & subtraction in a 128 bit register
 * for double precision floating point.
 * Operation : 2 PERM, 1 ADD
 */
#define SUBADD_128_D(a, b, c)                                                  \
{                                                                              \
    a = SWAP_RI_128_D(a);                                                      \
    c = SWAP_RI_128_D(_mm_addsub_pd(a, b));                                    \
}
/**
 * @brief alternatively performs addition & subtraction in a 256 bit register
 * for single precision floating point.
 * Operation : 2 PERM, 1 ADD
 */
#define SUBADD_256_S(a, b, c)                                                  \
{                                                                              \
    a = SWAP_RI_256_S(a);                                                      \
    c = SWAP_RI_256_S(_mm256_addsub_ps(a, b));                                 \
}

/**
 * @brief alternatively performs addition & subtraction in a 256 bit register
 * for double precision floating point.
 * Operation : 2 PERM, 1 ADD
 */
#define SUBADD_256_D(a, b, c)                                                  \
{                                                                              \
    a = SWAP_RI_256_D(a);                                                      \
    c = SWAP_RI_256_D(_mm256_addsub_pd(a, b));                                 \
}

#endif // AOCLFFTZ_SIMD_COMMON_H
