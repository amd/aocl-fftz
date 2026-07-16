// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_bwd_fft3avx256.c
 *
 *  @brief Backward-only twiddle Radix-3 FFT kernel with avx256 operations using
 *  x86 SIMD intrinsics
 *
 *  This file contains the DIT twiddle radix-3 FFT implementations using AVX-256
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 *  @author Srirammaswamy Srinivasan
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 6, 8, 26, 16, 7},
                                                     {0, 6, 8, 14, 9, 7}};

#define TWID_KNAME_FP32 twid_bwd_fft3avx256fp32
#define TWID_KNAME_FP64 twid_bwd_fft3avx256fp64
#define GET_OPS_COUNT get_ops_cnt_twid_bwd_fft3avx256
#define REGISTER_KERNEL register_kernel_twid_bwd_fft3avx256
#define KERNEL_USE_AVX256

#define LOAD_IN_S ITW_GATHER_S
#define LOAD_IN_256_S ITW_GATHER_256_S
#define LOAD_IN_128_S ITW_GATHER_128_S
#define LOAD_IN_64_S ITW_GATHER_LOW_128_S
#define LOAD_IN_D ITW_GATHER_D
#define LOAD_IN_256_D ITW_GATHER_256_D
#define LOAD_IN_128_D ITW_GATHER_128_D

#define STORE_OUT_S SCATTER_NOTW_S
#define STORE_OUT_256_S SCATTER_NOTW_256_S
#define STORE_OUT_128_S SCATTER_NOTW_128_S
#define STORE_OUT_64_S SCATTER_NOTW_LOW_128_S
#define STORE_OUT_D SCATTER_NOTW_D
#define STORE_OUT_256_D SCATTER_NOTW_256_D
#define STORE_OUT_128_D SCATTER_NOTW_128_D

#define KERNEL_VARIANT_C2C
#define KERNEL_DIRECTION_BWD

#include "core/kernels/dft/twid_c2c_common/twid_fft3.h"

#undef KERNEL_DIRECTION_BWD
#undef KERNEL_VARIANT_C2C

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

#undef KERNEL_USE_AVX256
#undef REGISTER_KERNEL
#undef GET_OPS_COUNT
#undef TWID_KNAME_FP64
#undef TWID_KNAME_FP32
