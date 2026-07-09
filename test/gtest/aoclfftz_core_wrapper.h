// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file aoclfftz_core_wrapper.h
 *
 *  @brief Contains wrapper function declarations for core funtions
 *  with dllexport attribute.
 *
 *  This file contains the wrapper function declarations for core functions
 *  with `__declspec(dllexport)` attribute for Windows compatibility.
 *
 *  @author Srirammaswamy Srinivasan
 */

#ifndef AOCLFFTZ_CORE_WRAPPER_H
#define AOCLFFTZ_CORE_WRAPPER_H

#include "core/common/memory_manager.h"
#include "core/common/strides.h"
#include "core/kernels/kernel.h"
#include "core/solvers/solver.h"
#include "selector/selector.h"
#include "core/kernels/non_dft/transpose/transpose_utils.h"
#include "core/kernels/non_dft/transpose/transpose_kernels.h"
#include "core/common/twiddle.h"

// Re-delcaring this struct to avoid using core/kernels/kernel_list.h file
typedef struct wrapper_kernel_fp_list
{
    k_register_kernel_ k_register_kernel;
    k_ops_cnt_ k_ops_cnt;
    FFTZ_UINT32 radix;
} wrapper_kernel_fp_list_t;

/* ---------------- kernels : get_opt_cnt_fft* ---------------- */

/* ---------------- core wrapper macros ---------------- */
#define GET_OPS_CNT_C2C_WRAPPER_DECL(radix, isa)                               \
    EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft##radix##isa##_wrapper(         \
        FFTZ_UINT8 prec, FFTZ_UINT8 dir);

#define GET_OPS_CNT_REAL_WRAPPER_DECL(kind, radix, isa)                        \
    EXPORT_SYM_DYN ops_cycles_t                                                \
        get_ops_cnt_##kind##_rfft##radix##isa##_wrapper(FFTZ_UINT8 prec,       \
                                                        FFTZ_UINT8 dir);

#define REGISTER_KERNEL_C2C_WRAPPER_DECL(radix, isa)                           \
    EXPORT_SYM_DYN kfft_ register_kernel_fft##radix##isa##_wrapper(            \
        FFTZ_UINT8 prec, FFTZ_UINT8 dir);

#define REGISTER_KERNEL_REAL_WRAPPER_DECL(kind, radix, isa)                    \
    EXPORT_SYM_DYN kfft_ register_kernel_##kind##_rfft##radix##isa##_wrapper(  \
        FFTZ_UINT8 prec, FFTZ_UINT8 dir);

#define WRAPPER_KERNEL_C2C_TABLE_ENTRY(radix, isa)                             \
    {register_kernel_fft##radix##isa##_wrapper,                                \
     get_ops_cnt_fft##radix##isa##_wrapper, radix},

#define WRAPPER_KERNEL_REAL_TABLE_ENTRY(kind, radix, isa)                      \
    {register_kernel_##kind##_rfft##radix##isa##_wrapper,                      \
     get_ops_cnt_##kind##_rfft##radix##isa##_wrapper, radix},

#define GET_OPS_CNT_C2C_WRAPPER_ALL_RADICES_DECL(isa)                          \
    GET_OPS_CNT_C2C_WRAPPER_DECL(2, isa)                                       \
    GET_OPS_CNT_C2C_WRAPPER_DECL(3, isa)                                       \
    GET_OPS_CNT_C2C_WRAPPER_DECL(4, isa)                                       \
    GET_OPS_CNT_C2C_WRAPPER_DECL(5, isa)                                       \
    GET_OPS_CNT_C2C_WRAPPER_DECL(6, isa)                                       \
    GET_OPS_CNT_C2C_WRAPPER_DECL(7, isa)                                       \
    GET_OPS_CNT_C2C_WRAPPER_DECL(8, isa)                                       \
    GET_OPS_CNT_C2C_WRAPPER_DECL(9, isa)                                       \
    GET_OPS_CNT_C2C_WRAPPER_DECL(10, isa)                                      \
    GET_OPS_CNT_C2C_WRAPPER_DECL(11, isa)                                      \
    GET_OPS_CNT_C2C_WRAPPER_DECL(12, isa)                                      \
    GET_OPS_CNT_C2C_WRAPPER_DECL(13, isa)                                      \
    GET_OPS_CNT_C2C_WRAPPER_DECL(14, isa)                                      \
    GET_OPS_CNT_C2C_WRAPPER_DECL(15, isa)                                      \
    GET_OPS_CNT_C2C_WRAPPER_DECL(16, isa)                                      \
    GET_OPS_CNT_C2C_WRAPPER_DECL(20, isa)                                      \
    GET_OPS_CNT_C2C_WRAPPER_DECL(48, isa)

#define REGISTER_KERNEL_C2C_WRAPPER_ALL_RADICES_DECL(isa)                      \
    REGISTER_KERNEL_C2C_WRAPPER_DECL(2, isa)                                   \
    REGISTER_KERNEL_C2C_WRAPPER_DECL(3, isa)                                   \
    REGISTER_KERNEL_C2C_WRAPPER_DECL(4, isa)                                   \
    REGISTER_KERNEL_C2C_WRAPPER_DECL(5, isa)                                   \
    REGISTER_KERNEL_C2C_WRAPPER_DECL(6, isa)                                   \
    REGISTER_KERNEL_C2C_WRAPPER_DECL(7, isa)                                   \
    REGISTER_KERNEL_C2C_WRAPPER_DECL(8, isa)                                   \
    REGISTER_KERNEL_C2C_WRAPPER_DECL(9, isa)                                   \
    REGISTER_KERNEL_C2C_WRAPPER_DECL(10, isa)                                  \
    REGISTER_KERNEL_C2C_WRAPPER_DECL(11, isa)                                  \
    REGISTER_KERNEL_C2C_WRAPPER_DECL(12, isa)                                  \
    REGISTER_KERNEL_C2C_WRAPPER_DECL(13, isa)                                  \
    REGISTER_KERNEL_C2C_WRAPPER_DECL(14, isa)                                  \
    REGISTER_KERNEL_C2C_WRAPPER_DECL(15, isa)                                  \
    REGISTER_KERNEL_C2C_WRAPPER_DECL(16, isa)                                  \
    REGISTER_KERNEL_C2C_WRAPPER_DECL(20, isa)                                  \
    REGISTER_KERNEL_C2C_WRAPPER_DECL(48, isa)

#define WRAPPER_KERNEL_C2C_TABLE_ALL_RADICES(isa)                              \
    WRAPPER_KERNEL_C2C_TABLE_ENTRY(2, isa)                                     \
    WRAPPER_KERNEL_C2C_TABLE_ENTRY(3, isa)                                     \
    WRAPPER_KERNEL_C2C_TABLE_ENTRY(4, isa)                                     \
    WRAPPER_KERNEL_C2C_TABLE_ENTRY(5, isa)                                     \
    WRAPPER_KERNEL_C2C_TABLE_ENTRY(6, isa)                                     \
    WRAPPER_KERNEL_C2C_TABLE_ENTRY(7, isa)                                     \
    WRAPPER_KERNEL_C2C_TABLE_ENTRY(8, isa)                                     \
    WRAPPER_KERNEL_C2C_TABLE_ENTRY(9, isa)                                     \
    WRAPPER_KERNEL_C2C_TABLE_ENTRY(10, isa)                                    \
    WRAPPER_KERNEL_C2C_TABLE_ENTRY(11, isa)                                    \
    WRAPPER_KERNEL_C2C_TABLE_ENTRY(12, isa)                                    \
    WRAPPER_KERNEL_C2C_TABLE_ENTRY(13, isa)                                    \
    WRAPPER_KERNEL_C2C_TABLE_ENTRY(14, isa)                                    \
    WRAPPER_KERNEL_C2C_TABLE_ENTRY(15, isa)                                    \
    WRAPPER_KERNEL_C2C_TABLE_ENTRY(16, isa)                                    \
    WRAPPER_KERNEL_C2C_TABLE_ENTRY(20, isa)                                    \
    WRAPPER_KERNEL_C2C_TABLE_ENTRY(48, isa)

#define GET_OPS_CNT_REAL_WRAPPER_ALL_RADICES_DECL(kind, isa)                   \
    GET_OPS_CNT_REAL_WRAPPER_DECL(kind, 2, isa)                                \
    GET_OPS_CNT_REAL_WRAPPER_DECL(kind, 3, isa)                                \
    GET_OPS_CNT_REAL_WRAPPER_DECL(kind, 4, isa)                                \
    GET_OPS_CNT_REAL_WRAPPER_DECL(kind, 5, isa)                                \
    GET_OPS_CNT_REAL_WRAPPER_DECL(kind, 6, isa)                                \
    GET_OPS_CNT_REAL_WRAPPER_DECL(kind, 7, isa)                                \
    GET_OPS_CNT_REAL_WRAPPER_DECL(kind, 8, isa)                                \
    GET_OPS_CNT_REAL_WRAPPER_DECL(kind, 9, isa)                                \
    GET_OPS_CNT_REAL_WRAPPER_DECL(kind, 10, isa)                               \
    GET_OPS_CNT_REAL_WRAPPER_DECL(kind, 11, isa)                               \
    GET_OPS_CNT_REAL_WRAPPER_DECL(kind, 12, isa)                               \
    GET_OPS_CNT_REAL_WRAPPER_DECL(kind, 14, isa)                               \
    GET_OPS_CNT_REAL_WRAPPER_DECL(kind, 15, isa)                               \
    GET_OPS_CNT_REAL_WRAPPER_DECL(kind, 16, isa)

#define REGISTER_KERNEL_REAL_WRAPPER_ALL_RADICES_DECL(kind, isa)               \
    REGISTER_KERNEL_REAL_WRAPPER_DECL(kind, 2, isa)                            \
    REGISTER_KERNEL_REAL_WRAPPER_DECL(kind, 3, isa)                            \
    REGISTER_KERNEL_REAL_WRAPPER_DECL(kind, 4, isa)                            \
    REGISTER_KERNEL_REAL_WRAPPER_DECL(kind, 5, isa)                            \
    REGISTER_KERNEL_REAL_WRAPPER_DECL(kind, 6, isa)                            \
    REGISTER_KERNEL_REAL_WRAPPER_DECL(kind, 7, isa)                            \
    REGISTER_KERNEL_REAL_WRAPPER_DECL(kind, 8, isa)                            \
    REGISTER_KERNEL_REAL_WRAPPER_DECL(kind, 9, isa)                            \
    REGISTER_KERNEL_REAL_WRAPPER_DECL(kind, 10, isa)                           \
    REGISTER_KERNEL_REAL_WRAPPER_DECL(kind, 11, isa)                           \
    REGISTER_KERNEL_REAL_WRAPPER_DECL(kind, 12, isa)                           \
    REGISTER_KERNEL_REAL_WRAPPER_DECL(kind, 14, isa)                           \
    REGISTER_KERNEL_REAL_WRAPPER_DECL(kind, 15, isa)                           \
    REGISTER_KERNEL_REAL_WRAPPER_DECL(kind, 16, isa)

#define WRAPPER_KERNEL_REAL_TABLE_ALL_RADICES(kind, isa)                       \
    WRAPPER_KERNEL_REAL_TABLE_ENTRY(kind, 2, isa)                              \
    WRAPPER_KERNEL_REAL_TABLE_ENTRY(kind, 3, isa)                              \
    WRAPPER_KERNEL_REAL_TABLE_ENTRY(kind, 4, isa)                              \
    WRAPPER_KERNEL_REAL_TABLE_ENTRY(kind, 5, isa)                              \
    WRAPPER_KERNEL_REAL_TABLE_ENTRY(kind, 6, isa)                              \
    WRAPPER_KERNEL_REAL_TABLE_ENTRY(kind, 7, isa)                              \
    WRAPPER_KERNEL_REAL_TABLE_ENTRY(kind, 8, isa)                              \
    WRAPPER_KERNEL_REAL_TABLE_ENTRY(kind, 9, isa)                              \
    WRAPPER_KERNEL_REAL_TABLE_ENTRY(kind, 10, isa)                             \
    WRAPPER_KERNEL_REAL_TABLE_ENTRY(kind, 11, isa)                             \
    WRAPPER_KERNEL_REAL_TABLE_ENTRY(kind, 12, isa)                             \
    WRAPPER_KERNEL_REAL_TABLE_ENTRY(kind, 14, isa)                             \
    WRAPPER_KERNEL_REAL_TABLE_ENTRY(kind, 15, isa)                             \
    WRAPPER_KERNEL_REAL_TABLE_ENTRY(kind, 16, isa)

// C2C Kernels
GET_OPS_CNT_C2C_WRAPPER_ALL_RADICES_DECL(c)

/* ---------------- kernels : register_kernel_fft* ---------------- */

// C2C Kernels
REGISTER_KERNEL_C2C_WRAPPER_ALL_RADICES_DECL(c)

/* ---------------- twiddle wrapper declarations ---------------- */
#define GET_OPS_CNT_TWID_WRAPPER_DECL(kind, radix, isa)                        \
    EXPORT_SYM_DYN ops_cycles_t                                                \
        get_ops_cnt_twid_##kind##_fft##radix##isa##_wrapper(FFTZ_UINT8 prec,   \
                                                            FFTZ_UINT8 dir);

#define GET_OPS_CNT_TWID_WRAPPER_ALL_RADICES_DECL(kind, isa)                   \
    GET_OPS_CNT_TWID_WRAPPER_DECL(kind, 2, isa)                                \
    GET_OPS_CNT_TWID_WRAPPER_DECL(kind, 3, isa)                                \
    GET_OPS_CNT_TWID_WRAPPER_DECL(kind, 4, isa)                                \
    GET_OPS_CNT_TWID_WRAPPER_DECL(kind, 5, isa)                                \
    GET_OPS_CNT_TWID_WRAPPER_DECL(kind, 6, isa)                                \
    GET_OPS_CNT_TWID_WRAPPER_DECL(kind, 7, isa)                                \
    GET_OPS_CNT_TWID_WRAPPER_DECL(kind, 8, isa)                                \
    GET_OPS_CNT_TWID_WRAPPER_DECL(kind, 9, isa)                                \
    GET_OPS_CNT_TWID_WRAPPER_DECL(kind, 10, isa)                               \
    GET_OPS_CNT_TWID_WRAPPER_DECL(kind, 11, isa)                               \
    GET_OPS_CNT_TWID_WRAPPER_DECL(kind, 12, isa)                               \
    GET_OPS_CNT_TWID_WRAPPER_DECL(kind, 13, isa)                               \
    GET_OPS_CNT_TWID_WRAPPER_DECL(kind, 14, isa)                               \
    GET_OPS_CNT_TWID_WRAPPER_DECL(kind, 15, isa)                               \
    GET_OPS_CNT_TWID_WRAPPER_DECL(kind, 16, isa)

#define REGISTER_KERNEL_TWID_WRAPPER_DECL(kind, radix, isa)                    \
    EXPORT_SYM_DYN kfft_                                                       \
        register_kernel_twid_##kind##_fft##radix##isa##_wrapper(               \
            FFTZ_UINT8 prec, FFTZ_UINT8 dir);

#define REGISTER_KERNEL_TWID_WRAPPER_ALL_RADICES_DECL(kind, isa)               \
    REGISTER_KERNEL_TWID_WRAPPER_DECL(kind, 2, isa)                            \
    REGISTER_KERNEL_TWID_WRAPPER_DECL(kind, 3, isa)                            \
    REGISTER_KERNEL_TWID_WRAPPER_DECL(kind, 4, isa)                            \
    REGISTER_KERNEL_TWID_WRAPPER_DECL(kind, 5, isa)                            \
    REGISTER_KERNEL_TWID_WRAPPER_DECL(kind, 6, isa)                            \
    REGISTER_KERNEL_TWID_WRAPPER_DECL(kind, 7, isa)                            \
    REGISTER_KERNEL_TWID_WRAPPER_DECL(kind, 8, isa)                            \
    REGISTER_KERNEL_TWID_WRAPPER_DECL(kind, 9, isa)                            \
    REGISTER_KERNEL_TWID_WRAPPER_DECL(kind, 10, isa)                           \
    REGISTER_KERNEL_TWID_WRAPPER_DECL(kind, 11, isa)                           \
    REGISTER_KERNEL_TWID_WRAPPER_DECL(kind, 12, isa)                           \
    REGISTER_KERNEL_TWID_WRAPPER_DECL(kind, 13, isa)                           \
    REGISTER_KERNEL_TWID_WRAPPER_DECL(kind, 14, isa)                           \
    REGISTER_KERNEL_TWID_WRAPPER_DECL(kind, 15, isa)                           \
    REGISTER_KERNEL_TWID_WRAPPER_DECL(kind, 16, isa)

// C2C Twiddle Kernels
GET_OPS_CNT_TWID_WRAPPER_ALL_RADICES_DECL(fwd, c)
GET_OPS_CNT_TWID_WRAPPER_ALL_RADICES_DECL(bwd, c)
GET_OPS_CNT_TWID_WRAPPER_ALL_RADICES_DECL(r2c, c)
GET_OPS_CNT_TWID_WRAPPER_ALL_RADICES_DECL(c2r, c)

// C2C Twiddle Kernels
REGISTER_KERNEL_TWID_WRAPPER_ALL_RADICES_DECL(fwd, c)
REGISTER_KERNEL_TWID_WRAPPER_ALL_RADICES_DECL(bwd, c)
REGISTER_KERNEL_TWID_WRAPPER_ALL_RADICES_DECL(r2c, c)
REGISTER_KERNEL_TWID_WRAPPER_ALL_RADICES_DECL(c2r, c)

// R2HC Kernels
GET_OPS_CNT_REAL_WRAPPER_ALL_RADICES_DECL(r2hc, c)
REGISTER_KERNEL_REAL_WRAPPER_ALL_RADICES_DECL(r2hc, c)

// R2HC-Fused Kernels
GET_OPS_CNT_REAL_WRAPPER_ALL_RADICES_DECL(r2hcf, c)
REGISTER_KERNEL_REAL_WRAPPER_ALL_RADICES_DECL(r2hcf, c)
/* ---------------- kernels : permuted_copy_* ---------------- */

EXPORT_SYM_DYN FFTZ_VOID permuted_copy_c_fp32_wrapper(
    FFTZ_VOID *in, FFTZ_VOID *out, FFTZ_INTP n, FFTZ_INTP size,
    FFTZ_INTP in_stride, FFTZ_INTP out_stride, FFTZ_INTP v_in_stride,
    FFTZ_INTP v_out_stride);
EXPORT_SYM_DYN FFTZ_VOID permuted_copy_c_fp64_wrapper(
    FFTZ_VOID *in, FFTZ_VOID *out, FFTZ_INTP n, FFTZ_INTP size,
    FFTZ_INTP in_stride, FFTZ_INTP out_stride, FFTZ_INTP v_in_stride,
    FFTZ_INTP v_out_stride);

/* ---------------- memory allocators/destroys ---------------- */

EXPORT_SYM_DYN
aoclfftz_decomp_scheme_t *alloc_decomp_scheme_wrapper(FFTZ_INT32 vec_rank,
                                                      FFTZ_INT32 dim_rank);
EXPORT_SYM_DYN aoclfftz_solution_t *alloc_solution_wrapper(FFTZ_INT32 vec_rank,
                                                           FFTZ_INT32 dim_rank);
EXPORT_SYM_DYN
aoclfftz_selector_t *alloc_selector_wrapper(FFTZ_INT32 vec_rank,
                                            FFTZ_INT32 dim_rank,
                                            kernel_tables_t *kernel_tables);
EXPORT_SYM_DYN FFTZ_VOID *
alloc_twiddle_for_solution_wrapper(FFTZ_UINT8 rad_size, FFTZ_UINT8 dt_prec);
EXPORT_SYM_DYN FFTZ_VOID destroy_selector_wrapper(aoclfftz_selector_t *sel);
EXPORT_SYM_DYN FFTZ_VOID destroy_solution_wrapper(aoclfftz_solution_t *sol);
EXPORT_SYM_DYN
FFTZ_VOID
destroy_decomp_scheme_wrapper(aoclfftz_decomp_scheme_t *decomp_scheme);
EXPORT_SYM_DYN FFTZ_VOID destroy_handle_wrapper(FFTZ_VOID *handle);

/* ---------------- strides wrapper ---------------- */
EXPORT_SYM_DYN FFTZ_VOID populate_stride_array_wrapper(
    FFTZ_INTP *strides, FFTZ_INTP stride_val, FFTZ_INTP n,
    FFTZ_UINT8 compute_half_complex, FFTZ_UINT8 adjust_to_full_complex);

/* ---------------- fused strides wrapper ---------------- */
EXPORT_SYM_DYN FFTZ_VOID prepare_fused_kernel_strides_wrapper(
    FFTZ_INTP *strides, FFTZ_INTP radix, FFTZ_INTP offset);

/* ---------------- wrapper kernel tables ---------------- */

// C2C Kernels
static wrapper_kernel_fp_list_t
    wrapper_kernels_c2c_c[NUM_KERNELS_IN_EACH_CATEGORY] = {
        WRAPPER_KERNEL_C2C_TABLE_ALL_RADICES(c)};

// R2HC Kernels
static wrapper_kernel_fp_list_t
    wrapper_kernels_r2hc_c[NUM_KERNELS_IN_EACH_CATEGORY] = {
        WRAPPER_KERNEL_REAL_TABLE_ALL_RADICES(r2hc, c)};

// R2HC-Fused Kernels
static wrapper_kernel_fp_list_t
    wrapper_kernels_r2hcf_c[NUM_KERNELS_IN_EACH_CATEGORY] = {
        WRAPPER_KERNEL_REAL_TABLE_ALL_RADICES(r2hcf, c)};

#ifdef ENABLE_AVX128
// C2C AVX128 Kernels
GET_OPS_CNT_C2C_WRAPPER_ALL_RADICES_DECL(avx128)
REGISTER_KERNEL_C2C_WRAPPER_ALL_RADICES_DECL(avx128)

// C2C Twiddle AVX128 Kernels
GET_OPS_CNT_TWID_WRAPPER_ALL_RADICES_DECL(fwd, avx128)
GET_OPS_CNT_TWID_WRAPPER_ALL_RADICES_DECL(bwd, avx128)
GET_OPS_CNT_TWID_WRAPPER_ALL_RADICES_DECL(r2c, avx128)
GET_OPS_CNT_TWID_WRAPPER_ALL_RADICES_DECL(c2r, avx128)
REGISTER_KERNEL_TWID_WRAPPER_ALL_RADICES_DECL(fwd, avx128)
REGISTER_KERNEL_TWID_WRAPPER_ALL_RADICES_DECL(bwd, avx128)
REGISTER_KERNEL_TWID_WRAPPER_ALL_RADICES_DECL(r2c, avx128)
REGISTER_KERNEL_TWID_WRAPPER_ALL_RADICES_DECL(c2r, avx128)

// R2HC AVX128 Kernels
GET_OPS_CNT_REAL_WRAPPER_ALL_RADICES_DECL(r2hc, avx128)
REGISTER_KERNEL_REAL_WRAPPER_ALL_RADICES_DECL(r2hc, avx128)

// R2HC-Fused AVX128 Kernels
GET_OPS_CNT_REAL_WRAPPER_ALL_RADICES_DECL(r2hcf, avx128)
REGISTER_KERNEL_REAL_WRAPPER_ALL_RADICES_DECL(r2hcf, avx128)

static wrapper_kernel_fp_list_t
    wrapper_kernels_c2c_avx128[NUM_KERNELS_IN_EACH_CATEGORY] = {
        WRAPPER_KERNEL_C2C_TABLE_ALL_RADICES(avx128)};

// R2HC AVX128 Kernels
static wrapper_kernel_fp_list_t
    wrapper_kernels_r2hc_avx128[NUM_KERNELS_IN_EACH_CATEGORY] = {
        WRAPPER_KERNEL_REAL_TABLE_ALL_RADICES(r2hc, avx128)};

// R2HC-Fused AVX128 Kernels
static wrapper_kernel_fp_list_t
    wrapper_kernels_r2hcf_avx128[NUM_KERNELS_IN_EACH_CATEGORY] = {
        WRAPPER_KERNEL_REAL_TABLE_ALL_RADICES(r2hcf, avx128)};
#endif

#ifdef ENABLE_AVX256
// C2C AVX256 Kernels
GET_OPS_CNT_C2C_WRAPPER_ALL_RADICES_DECL(avx256)
REGISTER_KERNEL_C2C_WRAPPER_ALL_RADICES_DECL(avx256)

// C2C Twiddle AVX256 Kernels
GET_OPS_CNT_TWID_WRAPPER_ALL_RADICES_DECL(fwd, avx256)
GET_OPS_CNT_TWID_WRAPPER_ALL_RADICES_DECL(bwd, avx256)
GET_OPS_CNT_TWID_WRAPPER_ALL_RADICES_DECL(r2c, avx256)
GET_OPS_CNT_TWID_WRAPPER_ALL_RADICES_DECL(c2r, avx256)
REGISTER_KERNEL_TWID_WRAPPER_ALL_RADICES_DECL(fwd, avx256)
REGISTER_KERNEL_TWID_WRAPPER_ALL_RADICES_DECL(bwd, avx256)
REGISTER_KERNEL_TWID_WRAPPER_ALL_RADICES_DECL(r2c, avx256)
REGISTER_KERNEL_TWID_WRAPPER_ALL_RADICES_DECL(c2r, avx256)

// R2HC AVX256 Kernels
GET_OPS_CNT_REAL_WRAPPER_ALL_RADICES_DECL(r2hc, avx256)
REGISTER_KERNEL_REAL_WRAPPER_ALL_RADICES_DECL(r2hc, avx256)

// R2HC-Fused AVX256 Kernels
GET_OPS_CNT_REAL_WRAPPER_ALL_RADICES_DECL(r2hcf, avx256)
REGISTER_KERNEL_REAL_WRAPPER_ALL_RADICES_DECL(r2hcf, avx256)

static wrapper_kernel_fp_list_t
    wrapper_kernels_c2c_avx256[NUM_KERNELS_IN_EACH_CATEGORY] = {
        WRAPPER_KERNEL_C2C_TABLE_ALL_RADICES(avx256)};

// R2HC AVX256 Kernels
static wrapper_kernel_fp_list_t
    wrapper_kernels_r2hc_avx256[NUM_KERNELS_IN_EACH_CATEGORY] = {
        WRAPPER_KERNEL_REAL_TABLE_ALL_RADICES(r2hc, avx256)};

// R2HC-Fused AVX256 Kernels
static wrapper_kernel_fp_list_t
    wrapper_kernels_r2hcf_avx256[NUM_KERNELS_IN_EACH_CATEGORY] = {
        WRAPPER_KERNEL_REAL_TABLE_ALL_RADICES(r2hcf, avx256)};
#endif

#ifdef ENABLE_AVX512
// C2C AVX512 Kernels
GET_OPS_CNT_C2C_WRAPPER_ALL_RADICES_DECL(avx512)
REGISTER_KERNEL_C2C_WRAPPER_ALL_RADICES_DECL(avx512)

// C2C Twiddle AVX512 Kernels
GET_OPS_CNT_TWID_WRAPPER_ALL_RADICES_DECL(fwd, avx512)
GET_OPS_CNT_TWID_WRAPPER_ALL_RADICES_DECL(bwd, avx512)
GET_OPS_CNT_TWID_WRAPPER_ALL_RADICES_DECL(r2c, avx512)
GET_OPS_CNT_TWID_WRAPPER_ALL_RADICES_DECL(c2r, avx512)
REGISTER_KERNEL_TWID_WRAPPER_ALL_RADICES_DECL(fwd, avx512)
REGISTER_KERNEL_TWID_WRAPPER_ALL_RADICES_DECL(bwd, avx512)
REGISTER_KERNEL_TWID_WRAPPER_ALL_RADICES_DECL(r2c, avx512)
REGISTER_KERNEL_TWID_WRAPPER_ALL_RADICES_DECL(c2r, avx512)

// R2HC AVX512 Kernels
GET_OPS_CNT_REAL_WRAPPER_ALL_RADICES_DECL(r2hc, avx512)
REGISTER_KERNEL_REAL_WRAPPER_ALL_RADICES_DECL(r2hc, avx512)

// R2HC-Fused AVX512 Kernels
GET_OPS_CNT_REAL_WRAPPER_ALL_RADICES_DECL(r2hcf, avx512)
REGISTER_KERNEL_REAL_WRAPPER_ALL_RADICES_DECL(r2hcf, avx512)

static wrapper_kernel_fp_list_t
    wrapper_kernels_c2c_avx512[NUM_KERNELS_IN_EACH_CATEGORY] = {
        WRAPPER_KERNEL_C2C_TABLE_ALL_RADICES(avx512)};

// R2HC AVX512 Kernels
static wrapper_kernel_fp_list_t
    wrapper_kernels_r2hc_avx512[NUM_KERNELS_IN_EACH_CATEGORY] = {
        WRAPPER_KERNEL_REAL_TABLE_ALL_RADICES(r2hc, avx512)};

// R2HC-Fused AVX512 Kernels
static wrapper_kernel_fp_list_t
    wrapper_kernels_r2hcf_avx512[NUM_KERNELS_IN_EACH_CATEGORY] = {
        WRAPPER_KERNEL_REAL_TABLE_ALL_RADICES(r2hcf, avx512)};
#endif

#undef GET_OPS_CNT_TWID_WRAPPER_DECL
#undef GET_OPS_CNT_TWID_WRAPPER_ALL_RADICES_DECL
#undef REGISTER_KERNEL_TWID_WRAPPER_DECL
#undef REGISTER_KERNEL_TWID_WRAPPER_ALL_RADICES_DECL
#undef GET_OPS_CNT_C2C_WRAPPER_DECL
#undef GET_OPS_CNT_REAL_WRAPPER_DECL
#undef REGISTER_KERNEL_C2C_WRAPPER_DECL
#undef REGISTER_KERNEL_REAL_WRAPPER_DECL
#undef WRAPPER_KERNEL_C2C_TABLE_ENTRY
#undef WRAPPER_KERNEL_REAL_TABLE_ENTRY
#undef GET_OPS_CNT_C2C_WRAPPER_ALL_RADICES_DECL
#undef REGISTER_KERNEL_C2C_WRAPPER_ALL_RADICES_DECL
#undef WRAPPER_KERNEL_C2C_TABLE_ALL_RADICES
#undef GET_OPS_CNT_REAL_WRAPPER_ALL_RADICES_DECL
#undef REGISTER_KERNEL_REAL_WRAPPER_ALL_RADICES_DECL
#undef WRAPPER_KERNEL_REAL_TABLE_ALL_RADICES
// Transpose wrappers
#define TRANSPOSE_WRAPPER_DECL(kernel_name, TYPE, isa)                         \
    EXPORT_SYM_DYN FFTZ_VOID CONCAT(FUNC(kernel_name, TYPE, isa),              \
                                    _wrapper)(TRANSPOSE_KERNEL_ARGS)

#define TRANSPOSE_WRAPPER_ALL_TYPES_DECL(kernel_name, isa)                     \
    TRANSPOSE_WRAPPER_DECL(kernel_name, FFTZ_FLOAT, isa);                      \
    TRANSPOSE_WRAPPER_DECL(kernel_name, FFTZ_DOUBLE, isa);                     \
    TRANSPOSE_WRAPPER_DECL(kernel_name, aoclfftz_complex_f_t, isa);            \
    TRANSPOSE_WRAPPER_DECL(kernel_name, aoclfftz_complex_d_t, isa);

TRANSPOSE_WRAPPER_ALL_TYPES_DECL(tiq_iterative, c)
TRANSPOSE_WRAPPER_ALL_TYPES_DECL(tisq_iterative, c)
TRANSPOSE_WRAPPER_ALL_TYPES_DECL(tiq_recursive_buf, c)
TRANSPOSE_WRAPPER_ALL_TYPES_DECL(tir_cycles, c)
TRANSPOSE_WRAPPER_ALL_TYPES_DECL(tisr_cycles, c)
TRANSPOSE_WRAPPER_ALL_TYPES_DECL(tos_iterative, c)
TRANSPOSE_WRAPPER_ALL_TYPES_DECL(tos_blocked, c)

// twiddle buffer setup wrappers
EXPORT_SYM_DYN FFTZ_VOID compute_twiddle_buffer_float_wrapper(
    FFTZ_VOID *twiddle_buffer, FFTZ_INTP r, FFTZ_INTP m);
EXPORT_SYM_DYN FFTZ_VOID compute_twiddle_buffer_double_wrapper(
    FFTZ_VOID *twiddle_buffer, FFTZ_INTP r, FFTZ_INTP m);

#endif // AOCLFFTZ_CORE_WRAPPER_H
