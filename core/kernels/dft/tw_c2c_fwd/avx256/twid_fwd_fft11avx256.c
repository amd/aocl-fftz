// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_fwd_fft11avx256.c
 *
 *  @brief Forward-only twiddle Radix-11 FFT kernel with avx256 operations using
 *  x86 SIMD intrinsics
 *
 *  This file contains the DIT twiddle radix-11 FFT implementations using
 *  AVX-256 operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 *  @author Srirammaswamy Srinivasan
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 70, 80, 98, 76, 27},
                                                     {0, 70, 80, 54, 45, 27}};

#define TWID_KNAME_FP32 twid_fwd_fft11avx256fp32
#define TWID_KNAME_FP64 twid_fwd_fft11avx256fp64
#define GET_OPS_COUNT get_ops_cnt_twid_fwd_fft11avx256
#define REGISTER_KERNEL register_kernel_twid_fwd_fft11avx256
#define KERNEL_USE_AVX256

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

#define KERNEL_VARIANT_C2C
#define KERNEL_DIRECTION_FWD

#include "core/kernels/dft/twid_c2c_common/twid_fft11.h"

#undef KERNEL_DIRECTION_FWD
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
