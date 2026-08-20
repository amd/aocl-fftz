// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_bwd_fft11avx128.c
 *
 *  @brief Backward-only twiddle Radix-11 FFT kernel with avx128 operations
 *  using x86 SIMD intrinsics
 *
 *  This file contains the DIT twiddle radix-11 FFT implementations using
 *  AVX-128 operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 *  @author Srirammaswamy Srinivasan
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 70, 80, 54, 65, 5},
                                                     {0, 70, 80, 42, 25, 5}};

#define TWID_KNAME_FP32 twid_bwd_fft11avx128fp32
#define TWID_KNAME_FP64 twid_bwd_fft11avx128fp64
#define GET_OPS_COUNT get_ops_cnt_twid_bwd_fft11avx128
#define REGISTER_KERNEL register_kernel_twid_bwd_fft11avx128
#define KERNEL_USE_AVX128

#define LOAD_IN_S ITW_GATHER_S
#define LOAD_IN_128_S ITW_GATHER_128_S
#define LOAD_IN_64_S ITW_GATHER_LOW_128_S
#define LOAD_IN_D ITW_GATHER_D
#define LOAD_IN_128_D ITW_GATHER_128_D

#define STORE_OUT_S SCATTER_NOTW_S
#define STORE_OUT_128_S SCATTER_NOTW_128_S
#define STORE_OUT_64_S SCATTER_NOTW_LOW_128_S
#define STORE_OUT_D SCATTER_NOTW_D
#define STORE_OUT_128_D SCATTER_NOTW_128_D

#define KERNEL_VARIANT_C2C
#define KERNEL_DIRECTION_BWD

#include "core/kernels/dft/twid_c2c_common/twid_fft11.h"

#undef KERNEL_DIRECTION_BWD
#undef KERNEL_VARIANT_C2C

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

#undef KERNEL_USE_AVX128
#undef REGISTER_KERNEL
#undef GET_OPS_COUNT
#undef TWID_KNAME_FP64
#undef TWID_KNAME_FP32
