// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_fft14avx128.c
 *
 *  @brief Twiddle Radix-14 FFT kernel with avx128 operations using x86 SIMD
 *  intrinsics
 *
 *  This file contains the DIT twiddle radix-14 FFT implementations using
 *  AVX-128 operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 *
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 62, 87, 69, 84, 6},
                                                     {0, 62, 87, 54, 32, 6}};


#define TWID_KNAME_FP32 twid_fft14avx128fp32
#define TWID_KNAME_FP64 twid_fft14avx128fp64
#define GET_OPS_COUNT get_ops_cnt_twid_fft14avx128
#define REGISTER_KERNEL register_kernel_twid_fft14avx128
#define KERNEL_USE_AVX128

#include "core/kernels/tw_c2c/common/twid_fft14.h"

#undef KERNEL_USE_AVX128
#undef REGISTER_KERNEL
#undef GET_OPS_COUNT
#undef TWID_KNAME_FP32
#undef TWID_KNAME_FP64
