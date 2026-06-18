// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_c2r_fft7avx128.c
 *
 *  @brief C2R fused twiddle (conjugate input + twiddle output) Radix-7 FFT
 *  kernel with avx128 operations using x86 SIMD intrinsics
 *
 *  This file contains the DIT twiddle radix-7 FFT implementations using AVX-128
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 *  @author Srirammaswamy Srinivasan
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 30, 39, 34, 33, 6},
                                                     {0, 30, 39, 20, 21, 6}};

#define TWID_KNAME_FP32 twid_c2r_fft7avx128fp32
#define TWID_KNAME_FP64 twid_c2r_fft7avx128fp64
#define GET_OPS_COUNT get_ops_cnt_twid_c2r_fft7avx128
#define REGISTER_KERNEL register_kernel_twid_c2r_fft7avx128
#define KERNEL_USE_AVX128

#define LOAD_IN_S GATHER_NOTW_S
#define LOAD_IN_128_S GATHER_NOTW_128_S
#define LOAD_IN_64_S GATHER_NOTW_LOW_128_S
#define LOAD_IN_D GATHER_NOTW_D
#define LOAD_IN_128_D GATHER_NOTW_128_D

#define STORE_OUT_S TW_SCATTER_S
#define STORE_OUT_128_S TW_SCATTER_128_S
#define STORE_OUT_64_S TW_SCATTER_LOW_128_S
#define STORE_OUT_D TW_SCATTER_D
#define STORE_OUT_128_D TW_SCATTER_128_D

#define IN_H2_S(val) CONJ_S(val)
#define IN_H2_128_S(val) CONJ_128_S(val)
#define IN_H2_D(val) CONJ_D(val)
#define IN_H2_128_D(val) CONJ_128_D(val)

#define KERNEL_VARIANT_C2R
#define KERNEL_DIRECTION_BWD

#include "core/kernels/dft/twid_c2c_common/twid_fft7.h"

#undef KERNEL_DIRECTION_BWD
#undef KERNEL_VARIANT_C2R

#undef STORE_OUT_128_D
#undef STORE_OUT_D
#undef STORE_OUT_64_S
#undef STORE_OUT_128_S
#undef STORE_OUT_S

#undef LOAD_IN_128_D
#undef LOAD_IN_D
#undef LOAD_IN_64_S
#undef LOAD_IN_128_S
#undef LOAD_IN_S

#undef IN_H2_128_D
#undef IN_H2_D
#undef IN_H2_128_S
#undef IN_H2_S

#undef KERNEL_USE_AVX128
#undef REGISTER_KERNEL
#undef GET_OPS_COUNT
#undef TWID_KNAME_FP64
#undef TWID_KNAME_FP32
