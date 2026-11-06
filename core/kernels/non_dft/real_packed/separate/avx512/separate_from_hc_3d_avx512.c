// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file separate_from_hc_3d_avx512.c
 *
 * @brief Plane-conj-pair packed 3D C2R traversal with AVX-512 row-conj-pairs.
 */

#include "core/kernels/kernel.h"
#include "core/kernels/non_dft/real_packed/packed_rdft_avx512_common.h"

#define PACK_RDFT_KNAME_FP32 separate_from_hc_3d_fp32_avx512
#define PACK_RDFT_KNAME_FP64 separate_from_hc_3d_fp64_avx512
#define REGISTER_KERNEL register_separate_from_hc_3d_avx512
#define ROW_CONJ_PAIR_FP32 separate_from_hc_row_pair_fp32_avx512
#define ROW_CONJ_PAIR_FP64 separate_from_hc_row_pair_fp64_avx512
#define PACKED_KERNEL_VARIANT_C2R

#include "core/kernels/non_dft/real_packed/packed_rdft_3d_common.h"

#undef PACKED_KERNEL_VARIANT_C2R
#undef ROW_CONJ_PAIR_FP64
#undef ROW_CONJ_PAIR_FP32
#undef REGISTER_KERNEL
#undef PACK_RDFT_KNAME_FP64
#undef PACK_RDFT_KNAME_FP32
