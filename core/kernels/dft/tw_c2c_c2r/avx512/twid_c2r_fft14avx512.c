// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_c2r_fft14avx512.c
 *
 *  @brief C2R fused twiddle (conjugate input + twiddle output) Radix-14 FFT
 *  kernel with avx512 operations using x86 SIMD intrinsics
 *
 *  This file contains the DIT twiddle radix-14 FFT implementations using
 *  AVX-512 operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 *  @author Srirammaswamy Srinivasan
 */

#include "core/kernels/kernel.h"
#include "core/kernels/simd_includes/simd_common_avx512.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{13, 62, 74, 250, 113, 97},
                                                     {13, 62, 74, 138, 45, 97}};

#define TWID_KNAME_FP32 twid_c2r_fft14avx512fp32
#define TWID_KNAME_FP64 twid_c2r_fft14avx512fp64
#define GET_OPS_COUNT get_ops_cnt_twid_c2r_fft14avx512
#define REGISTER_KERNEL register_kernel_twid_c2r_fft14avx512
#define KERNEL_USE_AVX512

#define LOAD_IN_S GATHER_NOTW_S
#define LOAD_IN_256_S GATHER_NOTW_256_S
#define LOAD_IN_128_S GATHER_NOTW_128_S
#define LOAD_IN_64_S GATHER_NOTW_LOW_128_S
#define LOAD_IN_D GATHER_NOTW_D
#define LOAD_IN_256_D GATHER_NOTW_256_D
#define LOAD_IN_128_D GATHER_NOTW_128_D

#define LOAD_IN_H2_S GATHER_NOTW_H2_S
#define LOAD_IN_H2_256_S GATHER_NOTW_H2_256_S
#define LOAD_IN_H2_128_S GATHER_NOTW_H2_128_S
#define LOAD_IN_H2_64_S GATHER_NOTW_H2_LOW_128_S
#define LOAD_IN_H2_D GATHER_NOTW_H2_D
#define LOAD_IN_H2_256_D GATHER_NOTW_H2_256_D
#define LOAD_IN_H2_128_D GATHER_NOTW_H2_128_D

#define STORE_OUT_S TW_SCATTER_S
#define STORE_OUT_256_S TW_SCATTER_256_S
#define STORE_OUT_128_S TW_SCATTER_128_S
#define STORE_OUT_64_S TW_SCATTER_LOW_128_S
#define STORE_OUT_D TW_SCATTER_D
#define STORE_OUT_256_D TW_SCATTER_256_D
#define STORE_OUT_128_D TW_SCATTER_128_D

#define IN_H2_S(val) CONJ_S(val)
#define IN_H2_256_S(val) CONJ_256_S(val)
#define IN_H2_128_S(val) CONJ_128_S(val)
#define IN_H2_D(val) CONJ_D(val)
#define IN_H2_256_D(val) CONJ_256_D(val)
#define IN_H2_128_D(val) CONJ_128_D(val)

#define KERNEL_VARIANT_C2R
#define KERNEL_DIRECTION_BWD

#include "core/kernels/dft/twid_c2c_common/twid_fft14.h"

#undef KERNEL_DIRECTION_BWD
#undef KERNEL_VARIANT_C2R

#undef STORE_OUT_128_D
#undef STORE_OUT_256_D
#undef STORE_OUT_D
#undef STORE_OUT_64_S
#undef STORE_OUT_128_S
#undef STORE_OUT_256_S
#undef STORE_OUT_S

#undef LOAD_IN_H2_128_D
#undef LOAD_IN_H2_256_D
#undef LOAD_IN_H2_D
#undef LOAD_IN_H2_64_S
#undef LOAD_IN_H2_128_S
#undef LOAD_IN_H2_256_S
#undef LOAD_IN_H2_S

#undef LOAD_IN_128_D
#undef LOAD_IN_256_D
#undef LOAD_IN_D
#undef LOAD_IN_64_S
#undef LOAD_IN_128_S
#undef LOAD_IN_256_S
#undef LOAD_IN_S

#undef IN_H2_128_D
#undef IN_H2_256_D
#undef IN_H2_D
#undef IN_H2_128_S
#undef IN_H2_256_S
#undef IN_H2_S

#undef KERNEL_USE_AVX512
#undef REGISTER_KERNEL
#undef GET_OPS_COUNT
#undef TWID_KNAME_FP64
#undef TWID_KNAME_FP32
