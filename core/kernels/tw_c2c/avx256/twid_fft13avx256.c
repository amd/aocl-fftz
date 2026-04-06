// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_fft13avx256.c
 *
 *  @brief Twiddle Radix-13 FFT kernel with avx256 operations using x86 SIMD
 *  intrinsics
 *
 *  This file contains the DIT twiddle radix-13 FFT implementations using avx256
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 *
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 62, 102, 116, 91, 35},
                                                     {0, 62, 102, 64, 54, 35}};


#define TWID_KNAME_FP32 twid_fft13avx256fp32
#define TWID_KNAME_FP64 twid_fft13avx256fp64
#define GET_OPS_COUNT get_ops_cnt_twid_fft13avx256
#define REGISTER_KERNEL register_kernel_twid_fft13avx256
#define KERNEL_USE_AVX256

#include "core/kernels/tw_c2c/common/twid_fft13.h"

#undef KERNEL_USE_AVX256
#undef REGISTER_KERNEL
#undef GET_OPS_COUNT
#undef TWID_KNAME_FP32
#undef TWID_KNAME_FP64
