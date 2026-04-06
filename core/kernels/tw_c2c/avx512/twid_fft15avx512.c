// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_fft15avx512.c
 *
 *  @brief Twiddle Radix-15 FFT kernel with AVX-512 operations using x86 SIMD
 *  intrinsics
 *
 *  This file contains the DIT twiddle radix-15 FFT implementations using
 *  AVX-512 operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 *
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common_avx512.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{14, 56, 78, 268, 136, 97},
                                                     {14, 56, 78, 148, 63, 97}};


#define TWID_KNAME_FP32 twid_fft15avx512fp32
#define TWID_KNAME_FP64 twid_fft15avx512fp64
#define GET_OPS_COUNT get_ops_cnt_twid_fft15avx512
#define REGISTER_KERNEL register_kernel_twid_fft15avx512
#define KERNEL_USE_AVX512

#include "core/kernels/tw_c2c/common/twid_fft15.h"

#undef KERNEL_USE_AVX512
#undef REGISTER_KERNEL
#undef GET_OPS_COUNT
#undef TWID_KNAME_FP32
#undef TWID_KNAME_FP64
