// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_fft7avx256.c
 *
 *  @brief Twiddle Radix-7 FFT kernel with AVX-256 operations using x86 SIMD
 *  intrinsics
 *
 *  This file contains the DIT twiddle radix-7 FFT implementations using AVX-256
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 *
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 30, 39, 62, 46, 17},
                                                     {0, 30, 39, 34, 27, 17}};


#define TWID_KNAME_FP32 twid_fft7avx256fp32
#define TWID_KNAME_FP64 twid_fft7avx256fp64
#define GET_OPS_COUNT get_ops_cnt_twid_fft7avx256
#define REGISTER_KERNEL register_kernel_twid_fft7avx256
#define KERNEL_USE_AVX256

#include "core/kernels/dft/tw_c2c/common/twid_fft7.h"

#undef KERNEL_USE_AVX256
#undef REGISTER_KERNEL
#undef GET_OPS_COUNT
#undef TWID_KNAME_FP32
#undef TWID_KNAME_FP64
