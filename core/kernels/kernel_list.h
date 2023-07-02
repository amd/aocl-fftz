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
 *  This file contains the category-wise list of standard(c based) kernels,
 *  AVX128 kernels, AVX256 kernels and AVX512 kernels for both single and 
 *  double precisions.
 *
 *  @author S. Biplab Raut
 */

#ifndef AOCLFFTZ_KERNEL_LIST_H
#define AOCLFFTZ_KERNEL_LIST_H

#include "api/aoclfftz_internal.h"

#define NUM_KERNELS_IN_CATEGORY 64
UCHAR kernels_standard_c[NUM_KERNELS_IN_CATEGORY] = 
 { 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 25, 32, 64 };
UCHAR kernels_permuted_c[NUM_KERNELS_IN_CATEGORY] =
{ 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 25, 32, 64 };
UCHAR kernels_standard_avx128[NUM_KERNELS_IN_CATEGORY] =
{ 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 25, 32, 64 };
UCHAR kernels_permuted_avx128[NUM_KERNELS_IN_CATEGORY] =
{ 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 25, 32, 64 };
UCHAR kernels_standard_avx256[NUM_KERNELS_IN_CATEGORY] =
{ 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 25, 32, 64 };
UCHAR kernels_permuted_avx256[NUM_KERNELS_IN_CATEGORY] =
{ 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 25, 32, 64 };
UCHAR kernels_standard_avx512[NUM_KERNELS_IN_CATEGORY] =
{ 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 25, 32, 64 };
UCHAR kernels_permuted_avx512[NUM_KERNELS_IN_CATEGORY] =
{ 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 25, 32, 64 };

#endif //AOCLFFTZ_KERNEL_LIST_H