// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_fft2avx128.c
 *
 *  @brief Twiddle Radix-2 FFT kernel with avx128 operations using x86 SIMD
 *  intrinsics
 *
 *  This file contains the DIT twiddle radix-2 FFT implementations using avx128
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 *
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 2, 3, 17, 8, 8},
                                                     {0, 2, 3, 9, 4, 8}};

#define TWID_KNAME_FP32 twid_fft2avx256fp32
#define TWID_KNAME_FP64 twid_fft2avx256fp64
#define GET_OPS_COUNT get_ops_cnt_twid_fft2avx256
#define REGISTER_KERNEL register_kernel_twid_fft2avx256
#define KERNEL_USE_AVX256

#include "core/kernels/dft/tw_c2c/common/twid_fft2.h"

#undef KERNEL_USE_AVX256
#undef REGISTER_KERNEL
#undef GET_OPS_COUNT
#undef TWID_KNAME_FP32
#undef TWID_KNAME_FP64
