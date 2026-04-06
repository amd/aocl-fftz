// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_fft10avx512.c
 *
 *  @brief Twiddle Radix-10 FFT kernel with AVX-512 operations using x86 SIMD
 *  intrinsics
 *
 *  This file contains the DIT twiddle radix-10 FFT implementations using
 *  AVX-512 operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 *
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common_avx512.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{9, 30, 42, 178, 88, 64},
                                                     {9, 30, 42, 98, 40, 64}};


#define TWID_KNAME_FP32 twid_fft10avx512fp32
#define TWID_KNAME_FP64 twid_fft10avx512fp64
#define GET_OPS_COUNT get_ops_cnt_twid_fft10avx512
#define REGISTER_KERNEL register_kernel_twid_fft10avx512
#define KERNEL_USE_AVX512

#include "core/kernels/tw_c2c/common/twid_fft10.h"

#undef KERNEL_USE_AVX512
#undef REGISTER_KERNEL
#undef GET_OPS_COUNT
#undef TWID_KNAME_FP32
#undef TWID_KNAME_FP64
