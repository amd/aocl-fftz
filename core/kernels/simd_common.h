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
 * and offset(v_in_stride) into 128 bit register.
 */
#define GATHER2_128_S(base, offset, dest)                                      \
    {                                                                          \
        dest = _mm_loadu_ps(base);                                             \
        dest = _mm_loadh_pi(dest, (__m64 *)(base + offset));                   \
    }

/**
 * @brief store two complex numbers(real,imaginary) of 32 bit single precision
 * floating point number from 128 bit register into memory addresses
 * specified by base address and offset(v_out_stride).
 */
#define SCATTER2_128_S(base, offset, src)                                      \
    {                                                                          \
        _mm_storel_pi((__m64 *)base, src);                                     \
        base += offset;                                                        \
        _mm_storeh_pi((__m64 *)base, src);                                     \
    }

/**
 * @brief load a complex number(real,imaginary) of 32 bit single precision
 * floating point number from memory addresses specified by base address
 * into 128 bit register.
 */
#define LD_LOW_128_S(base, dest)                                               \
    {                                                                          \
        dest = _mm_loadl_pi(_mm_setzero_ps(), (__m64 *)base);                  \
    }

/**
 * @brief store a complex number(real,imaginary) of 32 bit single precision
 * floating point number from 128 bit register into memory addresses
 * specified by base address.
 */
#define ST_LOW_128_S(base, src)                                                \
    {                                                                          \
        _mm_storel_pi((__m64 *)base, src);                                     \
    }

/**
 * @brief load a complex number(real,imaginary) of 64 bit double precision
 * floating point number from memory addresses specified by base address
 * into 128 bit register.
 */
#define LD_128_D(base, dest)                                                   \
    {                                                                          \
        dest = _mm_loadu_pd(base);                                             \
    }

/**
 * @brief store a complex number(real,imaginary) of 64 bit double precision
 * floating point number from 128 bit register into memory addresses
 * specified by base address.
 */
#define ST_128_D(base, src)                                                    \
    {                                                                          \
        _mm_storeu_pd(base, src);                                              \
    }


/**
 * @brief interchanges the real and imaginary values in a 128 bit register
 * for single precision floating point.
 */
#define SWAP_RI_128_S(val) _mm_shuffle_ps(val, val, 177)

/**
 * @brief interchanges the real and imaginary values in a 128 bit register
 * for double precision floating point.
 */
#define SWAP_RI_128_D(val) _mm_shuffle_pd(val, val, 1)

/**
 * @brief implies the number of sets that can be processed parallely.
 * Computed using Register width /(2* sizeof(floating point)
 */
#define NUM_SETS_128_S 2
#define NUM_SETS_128_D 1

/**
 * @brief prepare -0.0 for complex conjucate.
 */
union zero_conj_128 {
    unsigned u[4];
    __m128 s;
    __m128d d;
};

static const union zero_conj_128
            _conj_128_f = {{ 0x00000000, 0x80000000, 0x00000000, 0x80000000 }};
static const union zero_conj_128
            _conj_128_d = {{ 0x00000000, 0x00000000, 0x00000000, 0x80000000 }};

/**
 * @brief take conjucate of the complex number A+Bi => A-Bi
 */
#define CONJ_128_S(x) _mm_xor_ps(_conj_128_f.s, x)
#define CONJ_128_D(x) _mm_xor_pd(_conj_128_d.d, x)

#endif // AOCLFFTZ_SIMD_COMMON_H
