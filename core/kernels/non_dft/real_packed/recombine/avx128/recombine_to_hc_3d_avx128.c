// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file recombine_to_hc_3d_avx128.c
 *
 * @brief Packed 3D R2C recombine kernel (DIT last stage) with AVX-128
 * intrinsics.
 *
 * @author Srirammaswamy Srinivasan
 */

#include "core/kernels/kernel.h"
#include "core/kernels/non_dft/real_packed/packed_rdft_avx128_common.h"

#define PACK_RDFT_KNAME_FP32 recombine_to_hc_3d_fp32_avx128
#define PACK_RDFT_KNAME_FP64 recombine_to_hc_3d_fp64_avx128
#define REGISTER_KERNEL register_recombine_to_hc_3d_avx128
#define ROW_CONJ_PAIR_FP32 recombine_to_hc_row_pair_fp32_avx128
#define ROW_CONJ_PAIR_FP64 recombine_to_hc_row_pair_fp64_avx128
#define PACKED_KERNEL_VARIANT_R2C

#include "core/kernels/non_dft/real_packed/packed_rdft_3d_common.h"

#undef PACKED_KERNEL_VARIANT_R2C
#undef ROW_CONJ_PAIR_FP64
#undef ROW_CONJ_PAIR_FP32
#undef REGISTER_KERNEL
#undef PACK_RDFT_KNAME_FP64
#undef PACK_RDFT_KNAME_FP32
