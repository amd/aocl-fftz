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

/** @file kernel.h
 *
 *  @brief Kernel template and related routines for computing DFT computations.
 *
 *  This file defines the kernel template that is used to statically derive the
 *  kernels of different precisions (float and double) related to all the
 *  compute types (C, AVX128, AVX256, AVX512).
 *
 *  @note Different variants of data structures are defined to
 *  support float and double precision types by default in ILP64 data model.
 *
 *  @author S. Biplab Raut
 *  @author Srirammaswamy Srinivasan
 */

#ifndef AOCLFFTZ_KERNEL_H
#define AOCLFFTZ_KERNEL_H

#include "api/aoclfftz_internal.h"
#ifdef AOCL_ENABLE_LOG
#include "utils/utils.h"
#endif

// macro to make in/out stride array volatile
// hack to prevent compiler optimization to prevent register to stack movements
// on the variable
#define VOLATILE_STRIDE_ARRAY

// Constants related to radix sizes
#define RADIX_2 2
#define RADIX_3 3
#define RADIX_4 4
#define RADIX_5 5
#define RADIX_6 6
#define RADIX_7 7
#define RADIX_8 8
#define RADIX_9 9
#define RADIX_10 10
#define RADIX_11 11
#define RADIX_12 12
#define RADIX_13 13
#define RADIX_14 14
#define RADIX_15 15
#define RADIX_16 16

// Implies the number of sets that can be processed in parallel.
// Computed using - (register_width / (2 * 8 * sizeof(floating point)))
#define NUM_SETS_C_S 1
#define NUM_SETS_C_D 1
#define NUM_SETS_128_S 2
#define NUM_SETS_128_D 1
#define NUM_SETS_256_S 4
#define NUM_SETS_256_D 2
#define NUM_SETS_512_S 8
#define NUM_SETS_512_D 4

// Error return codes related to Kernel
// Add more codes at the top
typedef enum
{
    KERNEL_FAILURE = -1,
    KERNEL_SUCCESS // Successful operation
} aoclfftz_kernel_status;

// Kernel types
typedef enum
{
    C2C_KERNEL = 0,
    R2HC_KERNEL,
    R2HCF_KERNEL // TODO: Fix naming R2HCF or R2HC_FUSED
} aoclfftz_kernel_type;

// Holds the kernel level operational complexity in terms approximate cycles
typedef struct ops_cycles
{
    USHORT fma;
    USHORT mul;
    USHORT add;
    USHORT move;
    USHORT perm;
    USHORT other;
} ops_cycles_t;

// Function pointer to get kernel compute operations in terms of approx cycles
typedef ops_cycles_t (*k_ops_cnt_)(UINT8, UINT8);
typedef kfft_ (*k_register_kernel_)(UINT8, UINT8);

// Kernel data structure that holds kernel function pointers and other
// associated parameters related to radix and compute operations
typedef struct kernel
{
    kfft_ kfft;
    k_ops_cnt_ k_ops_cnt;
    UINT32 radix;
    aoclfftz_kernel_type kernel_type;
    UINT8 sets[NUM_PRECISIONS];
} kernel_t;

// Function declarations for the common routines
INT32 register_kernels(kernel_t kertab[NUM_KERNELS_IN_TABLE], INT32 dt,
                       INT32 dir, INT32 is_real, INT32 cpu_flags);

// Kernel function declarations for different floating point precision types
// supported in scalar and vector compute variants

// Get Operations Count Functions

// C2C Kernels
ops_cycles_t get_ops_cnt_fft2c(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft3c(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft4c(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft5c(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft6c(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft7c(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft8c(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft9c(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft10c(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft11c(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft12c(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft13c(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft14c(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft15c(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft16c(UINT8 precision, UINT8 direction);

#ifdef ENABLE_AVX128
ops_cycles_t get_ops_cnt_fft2avx128(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft3avx128(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft4avx128(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft5avx128(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft6avx128(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft7avx128(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft8avx128(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft9avx128(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft10avx128(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft11avx128(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft12avx128(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft13avx128(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft14avx128(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft15avx128(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft16avx128(UINT8 precision, UINT8 direction);
#endif

#ifdef ENABLE_AVX256
ops_cycles_t get_ops_cnt_fft2avx256(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft3avx256(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft4avx256(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft5avx256(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft6avx256(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft7avx256(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft8avx256(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft9avx256(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft10avx256(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft11avx256(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft12avx256(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft13avx256(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft14avx256(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft15avx256(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft16avx256(UINT8 precision, UINT8 direction);
#endif

#ifdef ENABLE_AVX512
ops_cycles_t get_ops_cnt_fft2avx512(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft3avx512(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft4avx512(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft5avx512(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft6avx512(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft7avx512(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft8avx512(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft9avx512(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft10avx512(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft11avx512(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft12avx512(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft13avx512(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft14avx512(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft15avx512(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_fft16avx512(UINT8 precision, UINT8 direction);
#endif

// R2HC Kernels
ops_cycles_t get_ops_cnt_r2hc_rfft2c(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft3c(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft4c(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft5c(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft6c(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft7c(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft8c(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft10c(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft12c(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft14c(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft15c(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft16c(UINT8 precision, UINT8 direction);

// R2HC-Fused Kernels
ops_cycles_t get_ops_cnt_r2hcf_rfft2c(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft3c(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft4c(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft5c(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft6c(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft7c(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft8c(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft10c(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft12c(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft14c(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft15c(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft16c(UINT8 precision, UINT8 direction);

#ifdef ENABLE_AVX128
// R2HC AVX128 Kernels
ops_cycles_t get_ops_cnt_r2hc_rfft2avx128(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft3avx128(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft4avx128(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft5avx128(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft6avx128(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft7avx128(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft8avx128(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft10avx128(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft14avx128(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft15avx128(UINT8 precision, UINT8 direction);

// R2HC-Fused AVX128 Kernels
ops_cycles_t get_ops_cnt_r2hcf_rfft2avx128(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft3avx128(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft4avx128(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft5avx128(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft6avx128(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft7avx128(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft8avx128(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft10avx128(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft14avx128(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft15avx128(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft16avx128(UINT8 precision, UINT8 direction);
#endif

#ifdef ENABLE_AVX256
// R2HC AVX256 Kernels
ops_cycles_t get_ops_cnt_r2hc_rfft2avx256(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft3avx256(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft4avx256(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft5avx256(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft6avx256(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft7avx256(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft8avx256(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft10avx256(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft14avx256(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft15avx256(UINT8 precision, UINT8 direction);

// R2HC-Fused AVX256 Kernels
ops_cycles_t get_ops_cnt_r2hcf_rfft2avx256(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft3avx256(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft4avx256(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft5avx256(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft6avx256(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft8avx256(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft10avx256(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft14avx256(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft15avx256(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft16avx256(UINT8 precision, UINT8 direction);
#endif

#ifdef ENABLE_AVX512
// R2HC AVX512 Kernels
ops_cycles_t get_ops_cnt_r2hc_rfft2avx512(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft3avx512(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft4avx512(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft5avx512(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft6avx512(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft7avx512(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft8avx512(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft10avx512(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft14avx512(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hc_rfft15avx512(UINT8 precision, UINT8 direction);

// R2HC-Fused AVX512 Kernels
ops_cycles_t get_ops_cnt_r2hcf_rfft2avx512(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft3avx512(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft4avx512(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft5avx512(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft6avx512(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft8avx512(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft10avx512(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft14avx512(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft15avx512(UINT8 precision, UINT8 direction);
ops_cycles_t get_ops_cnt_r2hcf_rfft16avx512(UINT8 precision, UINT8 direction);
#endif

// Register kernels

// C2C Kernels
kfft_ register_kernel_fft2c(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft3c(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft4c(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft5c(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft6c(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft7c(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft8c(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft9c(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft10c(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft11c(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft12c(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft13c(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft14c(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft15c(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft16c(UINT8 precision, UINT8 direction);

#ifdef ENABLE_AVX128
kfft_ register_kernel_fft2avx128(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft3avx128(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft4avx128(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft5avx128(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft6avx128(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft7avx128(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft8avx128(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft9avx128(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft10avx128(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft11avx128(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft12avx128(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft13avx128(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft14avx128(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft15avx128(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft16avx128(UINT8 precision, UINT8 direction);
#endif

#ifdef ENABLE_AVX256
kfft_ register_kernel_fft2avx256(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft3avx256(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft4avx256(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft5avx256(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft6avx256(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft7avx256(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft8avx256(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft9avx256(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft10avx256(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft11avx256(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft12avx256(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft13avx256(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft14avx256(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft15avx256(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft16avx256(UINT8 precision, UINT8 direction);
#endif

#ifdef ENABLE_AVX512
kfft_ register_kernel_fft2avx512(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft3avx512(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft4avx512(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft5avx512(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft6avx512(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft7avx512(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft8avx512(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft9avx512(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft10avx512(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft11avx512(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft12avx512(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft13avx512(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft14avx512(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft15avx512(UINT8 precision, UINT8 direction);
kfft_ register_kernel_fft16avx512(UINT8 precision, UINT8 direction);
#endif

// R2HC Kernels
kfft_ register_kernel_r2hc_rfft2c(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft3c(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft4c(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft5c(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft6c(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft7c(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft8c(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft10c(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft12c(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft14c(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft15c(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft16c(UINT8 precision, UINT8 direction);

// R2HC-Fused Kernels
kfft_ register_kernel_r2hcf_rfft2c(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft3c(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft4c(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft5c(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft6c(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft7c(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft8c(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft10c(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft12c(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft14c(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft15c(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft16c(UINT8 precision, UINT8 direction);

#ifdef ENABLE_AVX128
// R2HC AVX128 Kernels
kfft_ register_kernel_r2hc_rfft2avx128(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft3avx128(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft4avx128(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft5avx128(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft6avx128(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft7avx128(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft8avx128(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft10avx128(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft14avx128(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft15avx128(UINT8 precision, UINT8 direction);

// R2HC-Fused AVX128 Kernels
kfft_ register_kernel_r2hcf_rfft2avx128(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft3avx128(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft4avx128(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft5avx128(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft6avx128(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft7avx128(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft8avx128(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft10avx128(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft14avx128(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft15avx128(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft16avx128(UINT8 precision, UINT8 direction);
#endif

#ifdef ENABLE_AVX256
// R2HC AVX256 Kernels
kfft_ register_kernel_r2hc_rfft2avx256(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft3avx256(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft4avx256(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft5avx256(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft6avx256(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft7avx256(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft8avx256(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft10avx256(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft14avx256(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft15avx256(UINT8 precision, UINT8 direction);

// R2HC-Fused AVX256 Kernels
kfft_ register_kernel_r2hcf_rfft2avx256(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft3avx256(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft4avx256(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft5avx256(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft6avx256(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft8avx256(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft10avx256(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft14avx256(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft15avx256(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft16avx256(UINT8 precision, UINT8 direction);
#endif


#ifdef ENABLE_AVX512
// R2HC AVX512 Kernels
kfft_ register_kernel_r2hc_rfft2avx512(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft3avx512(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft4avx512(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft5avx512(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft6avx512(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft7avx512(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft8avx512(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft10avx512(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft14avx512(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hc_rfft15avx512(UINT8 precision, UINT8 direction);

// R2HC-Fused AVX512 Kernels
kfft_ register_kernel_r2hcf_rfft2avx512(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft3avx512(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft4avx512(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft5avx512(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft6avx512(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft8avx512(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft10avx512(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft14avx512(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft15avx512(UINT8 precision, UINT8 direction);
kfft_ register_kernel_r2hcf_rfft16avx512(UINT8 precision, UINT8 direction);
#endif

// Permuted Copy Kernels
VOID permuted_copy_c_fp32(VOID *in, VOID *out, INTP n, INTP size,
                          aoclfftz_strides_t *strides, UINT8 data_stride);
VOID permuted_copy_c_fp64(VOID *in, VOID *out, INTP n, INTP size,
                          aoclfftz_strides_t *strides, UINT8 data_stride);

#endif // AOCLFFTZ_KERNEL_H
