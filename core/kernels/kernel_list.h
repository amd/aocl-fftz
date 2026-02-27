/**
 * Copyright (C) 2023-2025, Advanced Micro Devices. All rights reserved.
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

/** @file kernel_list.h
 *
 *  @brief List of kernels for computing DFT.
 *
 *  This file contains the category-wise list of C based kernels,
 *  AVX128 kernels, AVX256 kernels and AVX512 kernels for both single and
 *  double precisions.
 *
 *  @author S. Biplab Raut
 *  @author Ashwin K. Godbole
 */

// WARNING: This file is intended to be included only by selector/selector.c

#ifndef AOCLFFTZ_KERNEL_LIST_H
#define AOCLFFTZ_KERNEL_LIST_H

#include "core/kernels/kernel.h"

// Kernel registration simplification macros
// These are included directly into this file because they are not used anywhere
// outside the scope of this file.
#define KREG_C_(pref, rad)                                                     \
{                                                                              \
    register_kernel_##pref##rad##c, get_ops_cnt_##pref##rad##c, rad            \
},
#define KREG_C(prefix, radix) KREG_C_(prefix, radix)

#ifdef ENABLE_AVX128
#define KREG_128_(pref, rad)                                                   \
{                                                                              \
    register_kernel_##pref##rad##avx128, get_ops_cnt_##pref##rad##avx128, rad  \
},
#define KREG_128(prefix, radix) KREG_128_(prefix, radix)
#else
#define KREG_128(...)
#endif

#ifdef ENABLE_AVX256
#define KREG_256_(pref, rad)                                                   \
{                                                                              \
    register_kernel_##pref##rad##avx256, get_ops_cnt_##pref##rad##avx256, rad  \
},
#define KREG_256(prefix, radix) KREG_256_(prefix, radix)
#else
#define KREG_256(...)
#endif

#ifdef ENABLE_AVX512
#define KREG_512_(pref, rad)                                                   \
{                                                                              \
    register_kernel_##pref##rad##avx512, get_ops_cnt_##pref##rad##avx512, rad  \
},
#define KREG_512(prefix, radix) KREG_512_(prefix, radix)
#else
#define KREG_512(...)
#endif

#define KREG(prefix, radix)                                                    \
    KREG_C(prefix, radix)                                                      \
    KREG_128(prefix, radix)                                                    \
    KREG_256(prefix, radix)                                                    \
    KREG_512(prefix, radix)

// Standard C2C kernel table
// Since this table is used only within the scope of this file, we prefer
// creating it here. There is no reason for this table to be defined in a
// header.
static kernel_fp_list_t kernels_c2c[NUM_KERNELS_IN_EACH_CATEGORY]
                                   [NUM_KERNEL_CATEGORIES] =
{
    {KREG(fft, 2)},  // radix  2
    {KREG(fft, 3)},  // radix  3
    {KREG(fft, 4)},  // radix  4
    {KREG(fft, 5)},  // radix  5
    {KREG(fft, 6)},  // radix  6
    {KREG(fft, 7)},  // radix  7
    {KREG(fft, 8)},  // radix  8
    {KREG(fft, 9)},  // radix  9
    {KREG(fft, 10)}, // radix 10
    {KREG(fft, 11)}, // radix 11
    {KREG(fft, 12)}, // radix 12
    {KREG(fft, 13)}, // radix 13
    {KREG(fft, 14)}, // radix 14
    {KREG(fft, 15)}, // radix 15
    {KREG(fft, 16)}, // radix 16
    {KREG(fft, 20)}, // radix 20
    {KREG(fft, 48)}, // radix 48
};

// Twiddle C2C kernel table
// Since this table is used only within the scope of this file, we prefer
// creating it here. There is no reason for this table to be defined in a
// header.
static kernel_fp_list_t kernels_twid_c2c[NUM_KERNELS_IN_EACH_CATEGORY]
                                        [NUM_KERNEL_CATEGORIES] =
{
    {KREG(twid_fft, 2)},  // radix  2
    {KREG(twid_fft, 3)},  // radix  3
    {KREG(twid_fft, 4)},  // radix  4
    {KREG(twid_fft, 5)},  // radix  5
    {KREG(twid_fft, 6)},  // radix  6
    {KREG(twid_fft, 7)},  // radix  7
    {KREG(twid_fft, 8)},  // radix  8
    {KREG(twid_fft, 9)},  // radix  9
    {KREG(twid_fft, 10)}, // radix 10
    {KREG(twid_fft, 11)}, // radix 11
    {KREG(twid_fft, 12)}, // radix 12
    {KREG(twid_fft, 13)}, // radix 13
    {KREG(twid_fft, 14)}, // radix 14
    {KREG(twid_fft, 15)}, // radix 15
    {KREG(twid_fft, 16)}, // radix 16
};

// Real FFT kernels + C2C kernels table
// Since this table is used only within the scope of this file, we prefer
// creating it here. There is no reason for this table to be defined in a
// header.
static kernel_fp_list_t kernels_real[NUM_REAL_KERNELS_VARIANTS]
                                    [NUM_KERNELS_IN_EACH_CATEGORY]
                                    [NUM_KERNEL_CATEGORIES] =
{
    {
        {KREG(r2hc_rfft, 2)},  // radix  2
        {KREG(r2hc_rfft, 3)},  // radix  3
        {KREG(r2hc_rfft, 4)},  // radix  4
        {KREG(r2hc_rfft, 5)},  // radix  5
        {KREG(r2hc_rfft, 6)},  // radix  6
        {KREG(r2hc_rfft, 7)},  // radix  7
        {KREG(r2hc_rfft, 8)},  // radix  8
        //{KREG(r2hc_rfft, 9)},  // radix  9
        {KREG(r2hc_rfft, 10)}, // radix 10
        //{KREG(r2hc_rfft, 11)}, // radix 11
        {KREG(r2hc_rfft, 12)}, // radix 12
        //{KREG(r2hc_rfft, 13)}, // radix 13
        {KREG(r2hc_rfft, 14)}, // radix 14
        {KREG(r2hc_rfft, 15)}, // radix 15
        {KREG(r2hc_rfft, 16)}, // radix 16
    },
    {
        {KREG(r2hcf_rfft, 2)},  // radix  2
        {KREG(r2hcf_rfft, 3)},  // radix  3
        {KREG(r2hcf_rfft, 4)},  // radix  4
        {KREG(r2hcf_rfft, 5)},  // radix  5
        {KREG(r2hcf_rfft, 6)},  // radix  6
        {KREG(r2hcf_rfft, 7)},  // radix  7
        {KREG(r2hcf_rfft, 8)},  // radix  8
        //{KREG(r2hcf_rfft, 9)},  // radix  9
        {KREG(r2hcf_rfft, 10)}, // radix 10
        //{KREG(r2hcf_rfft, 11)}, // radix 11
        {KREG(r2hcf_rfft, 12)}, // radix 12
        //{KREG(r2hcf_rfft, 13)}, // radix 13
        {KREG(r2hcf_rfft, 14)}, // radix 14
        {KREG(r2hcf_rfft, 15)}, // radix 15
        {KREG(r2hcf_rfft, 16)}, // radix 16
    },
    {
        {KREG(fft, 2)},  // radix  2
        {KREG(fft, 3)},  // radix  3
        {KREG(fft, 4)},  // radix  4
        {KREG(fft, 5)},  // radix  5
        {KREG(fft, 6)},  // radix  6
        {KREG(fft, 7)},  // radix  7
        {KREG(fft, 8)},  // radix  8
        //{KREG(fft, 9)},  // radix  9
        {KREG(fft, 10)}, // radix 10
        //{KREG(fft, 11)}, // radix 11
        {KREG(fft, 12)}, // radix 12
        //{KREG(fft, 13)}, // radix 13
        {KREG(fft, 14)}, // radix 14
        {KREG(fft, 15)}, // radix 15
        {KREG(fft, 16)}, // radix 16
    }
};

// Real FFT kernels + Twiddle C2C kernels table
// Since this table is used only within the scope of this file, we prefer
// creating it here. There is no reason for this table to be defined in a
// header.
static kernel_fp_list_t kernels_twid_real[NUM_REAL_KERNELS_VARIANTS]
                                         [NUM_KERNELS_IN_EACH_CATEGORY]
                                         [NUM_KERNEL_CATEGORIES] =
{
    {
        {KREG(r2hc_rfft, 2)},  // radix  2
        {KREG(r2hc_rfft, 3)},  // radix  3
        {KREG(r2hc_rfft, 4)},  // radix  4
        {KREG(r2hc_rfft, 5)},  // radix  5
        {KREG(r2hc_rfft, 6)},  // radix  6
        {KREG(r2hc_rfft, 7)},  // radix  7
        {KREG(r2hc_rfft, 8)},  // radix  8
        //{KREG(r2hc_rfft, 9)},  // radix  9
        {KREG(r2hc_rfft, 10)}, // radix 10
        //{KREG(r2hc_rfft, 11)}, // radix 11
        {KREG(r2hc_rfft, 12)}, // radix 12
        //{KREG(r2hc_rfft, 13)}, // radix 13
        {KREG(r2hc_rfft, 14)}, // radix 14
        {KREG(r2hc_rfft, 15)}, // radix 15
        {KREG(r2hc_rfft, 16)}, // radix 16
    },
    {
        {KREG(r2hcf_rfft, 2)},  // radix  2
        {KREG(r2hcf_rfft, 3)},  // radix  3
        {KREG(r2hcf_rfft, 4)},  // radix  4
        {KREG(r2hcf_rfft, 5)},  // radix  5
        {KREG(r2hcf_rfft, 6)},  // radix  6
        {KREG(r2hcf_rfft, 7)},  // radix  7
        {KREG(r2hcf_rfft, 8)},  // radix  8
        //{KREG(r2hcf_rfft, 9)},  // radix  9
        {KREG(r2hcf_rfft, 10)}, // radix 10
        //{KREG(r2hcf_rfft, 11)}, // radix 11
        {KREG(r2hcf_rfft, 12)}, // radix 12
        //{KREG(r2hcf_rfft, 13)}, // radix 13
        {KREG(r2hcf_rfft, 14)}, // radix 14
        {KREG(r2hcf_rfft, 15)}, // radix 15
        {KREG(r2hcf_rfft, 16)}, // radix 16
    },
    {
        {KREG(twid_fft, 2)},  // radix  2
        {KREG(twid_fft, 3)},  // radix  3
        {KREG(twid_fft, 4)},  // radix  4
        {KREG(twid_fft, 5)},  // radix  5
        {KREG(twid_fft, 6)},  // radix  6
        {KREG(twid_fft, 7)},  // radix  7
        {KREG(twid_fft, 8)},  // radix  8
        //{KREG(twid_fft, 9)},  // radix  9
        {KREG(twid_fft, 10)}, // radix 10
        //{KREG(twid_fft, 11)}, // radix 11
        {KREG(twid_fft, 12)}, // radix 12
        //{KREG(twid_fft, 13)}, // radix 13
        {KREG(twid_fft, 14)}, // radix 14
        {KREG(twid_fft, 15)}, // radix 15
        {KREG(twid_fft, 16)}, // radix 16
    }
};

#endif // AOCLFFTZ_KERNEL_LIST_H
