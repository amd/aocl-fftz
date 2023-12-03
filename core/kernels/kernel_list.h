/**
 * Copyright (C) 2023, Advanced Micro Devices. All rights reserved.
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
 */

#ifndef AOCLFFTZ_KERNEL_LIST_H
#define AOCLFFTZ_KERNEL_LIST_H

#include "core/kernels/kernel.h"

// Data structure containing kernel function pointers corresponding to the
// registration, and operation count of the kernel
typedef struct kernel_fp_list
{
    k_register_kernel_ k_register_kernel;
    k_ops_cnt_ k_ops_cnt;
    UINT32 radix;
} kernel_fp_list_t;

kernel_fp_list_t kernels_c[NUM_KERNELS_IN_EACH_CATEGORY] =
{
    {register_kernel_fft2c, get_ops_cnt_fft2c, 2},    // radix-2
    {register_kernel_fft3c, get_ops_cnt_fft3c, 3},    // radix-3
    {register_kernel_fft4c, get_ops_cnt_fft4c, 4},    // radix-4
    {register_kernel_fft5c, get_ops_cnt_fft5c, 5},    // radix-5
    {register_kernel_fft6c, get_ops_cnt_fft6c, 6},    // radix-6
    {register_kernel_fft7c, get_ops_cnt_fft7c, 7},    // radix-7
    {register_kernel_fft8c, get_ops_cnt_fft8c, 8},    // radix-8
    {register_kernel_fft9c, get_ops_cnt_fft9c, 9},    // radix-9
    {register_kernel_fft10c, get_ops_cnt_fft10c, 10}, // radix-10
    {register_kernel_fft11c, get_ops_cnt_fft11c, 11}, // radix-11
    {register_kernel_fft12c, get_ops_cnt_fft12c, 12}, // radix-12
    {register_kernel_fft13c, get_ops_cnt_fft13c, 13}, // radix-13
    {register_kernel_fft14c, get_ops_cnt_fft14c, 14}, // radix-14
    {register_kernel_fft15c, get_ops_cnt_fft15c, 15}, // radix-15
    {register_kernel_fft16c, get_ops_cnt_fft16c, 16}, // radix-16
    {NULL, NULL, 20},                                 // radix-20
    {NULL, NULL, 25},                                 // radix-25
    {NULL, NULL, 32},                                 // radix-32
    {NULL, NULL, 64}                                  // radix-64
};

#ifdef ENABLE_AVX128
kernel_fp_list_t kernels_avx128[NUM_KERNELS_IN_EACH_CATEGORY] =
{
    {register_kernel_fft2avx128, get_ops_cnt_fft2avx128, 2}, // radix-2
    {register_kernel_fft3avx128, get_ops_cnt_fft3avx128, 3}, // radix-3
    {register_kernel_fft4avx128, get_ops_cnt_fft4avx128, 4}, // radix-4
    {register_kernel_fft5avx128, get_ops_cnt_fft5avx128, 5}, // radix-5
    {register_kernel_fft6avx128, get_ops_cnt_fft6avx128, 6}, // radix-6
    {register_kernel_fft7avx128, get_ops_cnt_fft7avx128, 7}, // radix-7
    {register_kernel_fft8avx128, get_ops_cnt_fft8avx128, 8}, // radix-8
    {register_kernel_fft9avx128, get_ops_cnt_fft9avx128, 9}, // radix-9
    {register_kernel_fft10avx128, get_ops_cnt_fft10avx128, 10},// radix-10
    {NULL, NULL, 11},                                        // radix-11
    {NULL, NULL, 12},                                        // radix-12
    {NULL, NULL, 13},                                        // radix-13
    {NULL, NULL, 14},                                        // radix-14
    {NULL, NULL, 15},                                        // radix-15
    {NULL, NULL, 16},                                        // radix-16
    {NULL, NULL, 20},                                        // radix-20
    {NULL, NULL, 25},                                        // radix-25
    {NULL, NULL, 32},                                        // radix-32
    {NULL, NULL, 64}                                         // radix-64
};
#endif

#ifdef ENABLE_AVX256
kernel_fp_list_t kernels_avx256[NUM_KERNELS_IN_EACH_CATEGORY] =
{
    {register_kernel_fft2avx256, get_ops_cnt_fft2avx256, 2},   // radix-2
    {register_kernel_fft5avx256, get_ops_cnt_fft5avx256, 5},   // radix-5
    {NULL, NULL, 3},   // radix-3
    {NULL, NULL, 4},   // radix-4
    {NULL, NULL, 6},   // radix-6
    {NULL, NULL, 7},   // radix-7
    {NULL, NULL, 8},   // radix-8
    {NULL, NULL, 9},   // radix-9
    {NULL, NULL, 10},  // radix-10
    {NULL, NULL, 11},  // radix-11
    {NULL, NULL, 12},  // radix-12
    {NULL, NULL, 13},  // radix-13
    {NULL, NULL, 14},  // radix-14
    {NULL, NULL, 15},  // radix-15
    {NULL, NULL, 16},  // radix-16
    {NULL, NULL, 20},  // radix-20
    {NULL, NULL, 25},  // radix-25
    {NULL, NULL, 32},  // radix-32
    {NULL, NULL, 64}   // radix-64
};
#endif

#ifdef ENABLE_AVX512
kernel_fp_list_t kernels_avx512[NUM_KERNELS_IN_EACH_CATEGORY] =
{
    {NULL, NULL, 2 }, //radix-2
    {NULL, NULL, 3 }, //radix-3
    {NULL, NULL, 4 }, //radix-4
    {NULL, NULL, 5 }, //radix-5
    {NULL, NULL, 6 }, //radix-6
    {NULL, NULL, 7 }, //radix-7
    {NULL, NULL, 8 }, //radix-8
    {NULL, NULL, 9 }, //radix-9
    {NULL, NULL, 10}, //radix-10
    {NULL, NULL, 11}, //radix-11
    {NULL, NULL, 12}, //radix-12
    {NULL, NULL, 13}, //radix-13
    {NULL, NULL, 14}, //radix-14
    {NULL, NULL, 15}, //radix-15
    {NULL, NULL, 16}, //radix-16
    {NULL, NULL, 20}, //radix-20
    {NULL, NULL, 25}, //radix-25
    {NULL, NULL, 32}, //radix-32
    {NULL, NULL, 64}  //radix-64
};
#endif

#endif // AOCLFFTZ_KERNEL_LIST_H
