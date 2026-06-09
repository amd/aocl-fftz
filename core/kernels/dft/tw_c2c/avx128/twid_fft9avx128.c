// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_fft9avx128.c
 *
 *  @brief Twiddle Radix-9 FFT kernel with AVX-128 operations using x86 SIMD
 *  intrinsics
 *
 *  This file contains the DIT twiddle radix-9 FFT implementations using AVX-128
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 *
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 36, 48, 44, 62, 6},
                                                     {0, 36, 48, 34, 30, 6}};


#define TWID_KNAME_FP32 twid_fft9avx128fp32
#define TWID_KNAME_FP64 twid_fft9avx128fp64
#define GET_OPS_COUNT get_ops_cnt_twid_fft9avx128
#define REGISTER_KERNEL register_kernel_twid_fft9avx128
#define KERNEL_USE_AVX128

#include "core/kernels/dft/tw_c2c/common/twid_fft9.h"

#undef KERNEL_USE_AVX128
#undef REGISTER_KERNEL
#undef GET_OPS_COUNT
#undef TWID_KNAME_FP32
#undef TWID_KNAME_FP64
