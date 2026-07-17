// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_r2c_fft7avx512.c
 *
 *  @brief R2C fused twiddle (forward twiddle + conjugate output) Radix-7 FFT
 *  kernel with avx512 operations using x86 SIMD intrinsics
 *
 *  This file contains the DIT twiddle radix-7 FFT implementations using AVX-512
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 *  @author Srirammaswamy Srinivasan
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common_avx512.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{6, 30, 33, 124, 54, 48},
                                                     {6, 30, 33, 68, 21, 48}};

#define TWID_KNAME_FP32 twid_r2c_fft7avx512fp32
#define TWID_KNAME_FP64 twid_r2c_fft7avx512fp64
#define GET_OPS_COUNT get_ops_cnt_twid_r2c_fft7avx512
#define REGISTER_KERNEL register_kernel_twid_r2c_fft7avx512
#define KERNEL_USE_AVX512

#define LOAD_IN_S TW_GATHER_S
#define LOAD_IN_256_S TW_GATHER_256_S
#define LOAD_IN_128_S TW_GATHER_128_S
#define LOAD_IN_64_S TW_GATHER_LOW_128_S
#define LOAD_IN_D TW_GATHER_D
#define LOAD_IN_256_D TW_GATHER_256_D
#define LOAD_IN_128_D TW_GATHER_128_D

#define STORE_OUT_S SCATTER_NOTW_S
#define STORE_OUT_256_S SCATTER_NOTW_256_S
#define STORE_OUT_128_S SCATTER_NOTW_128_S
#define STORE_OUT_64_S SCATTER_NOTW_LOW_128_S
#define STORE_OUT_D SCATTER_NOTW_D
#define STORE_OUT_256_D SCATTER_NOTW_256_D
#define STORE_OUT_128_D SCATTER_NOTW_128_D

#define STORE_OUT_H2_S SCATTER_NOTW_H2_S
#define STORE_OUT_H2_256_S SCATTER_NOTW_H2_256_S
#define STORE_OUT_H2_128_S SCATTER_NOTW_H2_128_S
#define STORE_OUT_H2_64_S SCATTER_NOTW_H2_LOW_128_S
#define STORE_OUT_H2_D SCATTER_NOTW_H2_D
#define STORE_OUT_H2_256_D SCATTER_NOTW_H2_256_D
#define STORE_OUT_H2_128_D SCATTER_NOTW_H2_128_D

#define OUT_H2_S(val) CONJ_S(val)
#define OUT_H2_256_S(val) CONJ_256_S(val)
#define OUT_H2_128_S(val) CONJ_128_S(val)
#define OUT_H2_D(val) CONJ_D(val)
#define OUT_H2_256_D(val) CONJ_256_D(val)
#define OUT_H2_128_D(val) CONJ_128_D(val)

#define KERNEL_VARIANT_R2C
#define KERNEL_DIRECTION_FWD

#include "core/kernels/dft/twid_c2c_common/twid_fft7.h"

#undef KERNEL_DIRECTION_FWD
#undef KERNEL_VARIANT_R2C

#undef STORE_OUT_H2_128_D
#undef STORE_OUT_H2_256_D
#undef STORE_OUT_H2_D
#undef STORE_OUT_H2_64_S
#undef STORE_OUT_H2_128_S
#undef STORE_OUT_H2_256_S
#undef STORE_OUT_H2_S

#undef STORE_OUT_128_D
#undef STORE_OUT_256_D
#undef STORE_OUT_D
#undef STORE_OUT_64_S
#undef STORE_OUT_128_S
#undef STORE_OUT_256_S
#undef STORE_OUT_S

#undef LOAD_IN_128_D
#undef LOAD_IN_256_D
#undef LOAD_IN_D
#undef LOAD_IN_64_S
#undef LOAD_IN_128_S
#undef LOAD_IN_256_S
#undef LOAD_IN_S

#undef OUT_H2_128_D
#undef OUT_H2_256_D
#undef OUT_H2_D
#undef OUT_H2_128_S
#undef OUT_H2_256_S
#undef OUT_H2_S

#undef KERNEL_USE_AVX512
#undef REGISTER_KERNEL
#undef GET_OPS_COUNT
#undef TWID_KNAME_FP64
#undef TWID_KNAME_FP32
