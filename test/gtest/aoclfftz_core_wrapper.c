// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file aoclfftz_core_wrapper.c
 *
 *  @brief Contains wrapper function definitions for core funtions
 *  with dllexport attribute.
 *
 *  This file contains the wrapper function definitions for core functions
 *  with `__declspec(dllexport)` attribute for Windows compatibility.
 *
 *  @author Srirammaswamy Srinivasan
 *  @author Ashwin K. Godbole
 */

#include "aoclfftz_core_wrapper.h"

/* ---------------- kernels : get_opt_cnt_fft* ---------------- */

/* ---------------- core wrapper macros ---------------- */
#define GET_OPS_CNT_C2C_WRAPPER_DEFN(radix, isa)                               \
    ops_cycles_t get_ops_cnt_fft##radix##isa##_wrapper(FFTZ_UINT8 precision,   \
                                                       FFTZ_UINT8 direction)   \
    {                                                                          \
        return get_ops_cnt_fft##radix##isa(precision, direction);              \
    }

#define GET_OPS_CNT_REAL_WRAPPER_DEFN(kind, radix, isa)                        \
    ops_cycles_t get_ops_cnt_##kind##_rfft##radix##isa##_wrapper(              \
        FFTZ_UINT8 precision, FFTZ_UINT8 direction)                            \
    {                                                                          \
        return get_ops_cnt_##kind##_rfft##radix##isa(precision, direction);    \
    }

#define REGISTER_KERNEL_C2C_WRAPPER_DEFN(radix, isa)                           \
    kfft_ register_kernel_fft##radix##isa##_wrapper(FFTZ_UINT8 precision,      \
                                                    FFTZ_UINT8 direction)      \
    {                                                                          \
        return register_kernel_fft##radix##isa(precision, direction);          \
    }

#define REGISTER_KERNEL_REAL_WRAPPER_DEFN(kind, radix, isa)                    \
    kfft_ register_kernel_##kind##_rfft##radix##isa##_wrapper(                 \
        FFTZ_UINT8 precision, FFTZ_UINT8 direction)                            \
    {                                                                          \
        return register_kernel_##kind##_rfft##radix##isa(precision,            \
                                                         direction);           \
    }

#define GET_OPS_CNT_C2C_WRAPPER_ALL_RADICES_DEFN(isa)                          \
    GET_OPS_CNT_C2C_WRAPPER_DEFN(2, isa)                                       \
    GET_OPS_CNT_C2C_WRAPPER_DEFN(3, isa)                                       \
    GET_OPS_CNT_C2C_WRAPPER_DEFN(4, isa)                                       \
    GET_OPS_CNT_C2C_WRAPPER_DEFN(5, isa)                                       \
    GET_OPS_CNT_C2C_WRAPPER_DEFN(6, isa)                                       \
    GET_OPS_CNT_C2C_WRAPPER_DEFN(7, isa)                                       \
    GET_OPS_CNT_C2C_WRAPPER_DEFN(8, isa)                                       \
    GET_OPS_CNT_C2C_WRAPPER_DEFN(9, isa)                                       \
    GET_OPS_CNT_C2C_WRAPPER_DEFN(10, isa)                                      \
    GET_OPS_CNT_C2C_WRAPPER_DEFN(11, isa)                                      \
    GET_OPS_CNT_C2C_WRAPPER_DEFN(12, isa)                                      \
    GET_OPS_CNT_C2C_WRAPPER_DEFN(13, isa)                                      \
    GET_OPS_CNT_C2C_WRAPPER_DEFN(14, isa)                                      \
    GET_OPS_CNT_C2C_WRAPPER_DEFN(15, isa)                                      \
    GET_OPS_CNT_C2C_WRAPPER_DEFN(16, isa)                                      \
    GET_OPS_CNT_C2C_WRAPPER_DEFN(20, isa)                                      \
    GET_OPS_CNT_C2C_WRAPPER_DEFN(48, isa)

#define REGISTER_KERNEL_C2C_WRAPPER_ALL_RADICES_DEFN(isa)                      \
    REGISTER_KERNEL_C2C_WRAPPER_DEFN(2, isa)                                   \
    REGISTER_KERNEL_C2C_WRAPPER_DEFN(3, isa)                                   \
    REGISTER_KERNEL_C2C_WRAPPER_DEFN(4, isa)                                   \
    REGISTER_KERNEL_C2C_WRAPPER_DEFN(5, isa)                                   \
    REGISTER_KERNEL_C2C_WRAPPER_DEFN(6, isa)                                   \
    REGISTER_KERNEL_C2C_WRAPPER_DEFN(7, isa)                                   \
    REGISTER_KERNEL_C2C_WRAPPER_DEFN(8, isa)                                   \
    REGISTER_KERNEL_C2C_WRAPPER_DEFN(9, isa)                                   \
    REGISTER_KERNEL_C2C_WRAPPER_DEFN(10, isa)                                  \
    REGISTER_KERNEL_C2C_WRAPPER_DEFN(11, isa)                                  \
    REGISTER_KERNEL_C2C_WRAPPER_DEFN(12, isa)                                  \
    REGISTER_KERNEL_C2C_WRAPPER_DEFN(13, isa)                                  \
    REGISTER_KERNEL_C2C_WRAPPER_DEFN(14, isa)                                  \
    REGISTER_KERNEL_C2C_WRAPPER_DEFN(15, isa)                                  \
    REGISTER_KERNEL_C2C_WRAPPER_DEFN(16, isa)                                  \
    REGISTER_KERNEL_C2C_WRAPPER_DEFN(20, isa)                                  \
    REGISTER_KERNEL_C2C_WRAPPER_DEFN(48, isa)

#define GET_OPS_CNT_REAL_WRAPPER_ALL_RADICES_DEFN(kind, isa)                   \
    GET_OPS_CNT_REAL_WRAPPER_DEFN(kind, 2, isa)                                \
    GET_OPS_CNT_REAL_WRAPPER_DEFN(kind, 3, isa)                                \
    GET_OPS_CNT_REAL_WRAPPER_DEFN(kind, 4, isa)                                \
    GET_OPS_CNT_REAL_WRAPPER_DEFN(kind, 5, isa)                                \
    GET_OPS_CNT_REAL_WRAPPER_DEFN(kind, 6, isa)                                \
    GET_OPS_CNT_REAL_WRAPPER_DEFN(kind, 7, isa)                                \
    GET_OPS_CNT_REAL_WRAPPER_DEFN(kind, 8, isa)                                \
    GET_OPS_CNT_REAL_WRAPPER_DEFN(kind, 9, isa)                                \
    GET_OPS_CNT_REAL_WRAPPER_DEFN(kind, 10, isa)                               \
    GET_OPS_CNT_REAL_WRAPPER_DEFN(kind, 11, isa)                               \
    GET_OPS_CNT_REAL_WRAPPER_DEFN(kind, 12, isa)                               \
    GET_OPS_CNT_REAL_WRAPPER_DEFN(kind, 13, isa)                               \
    GET_OPS_CNT_REAL_WRAPPER_DEFN(kind, 14, isa)                               \
    GET_OPS_CNT_REAL_WRAPPER_DEFN(kind, 15, isa)                               \
    GET_OPS_CNT_REAL_WRAPPER_DEFN(kind, 16, isa)

#define REGISTER_KERNEL_REAL_WRAPPER_ALL_RADICES_DEFN(kind, isa)               \
    REGISTER_KERNEL_REAL_WRAPPER_DEFN(kind, 2, isa)                            \
    REGISTER_KERNEL_REAL_WRAPPER_DEFN(kind, 3, isa)                            \
    REGISTER_KERNEL_REAL_WRAPPER_DEFN(kind, 4, isa)                            \
    REGISTER_KERNEL_REAL_WRAPPER_DEFN(kind, 5, isa)                            \
    REGISTER_KERNEL_REAL_WRAPPER_DEFN(kind, 6, isa)                            \
    REGISTER_KERNEL_REAL_WRAPPER_DEFN(kind, 7, isa)                            \
    REGISTER_KERNEL_REAL_WRAPPER_DEFN(kind, 8, isa)                            \
    REGISTER_KERNEL_REAL_WRAPPER_DEFN(kind, 9, isa)                            \
    REGISTER_KERNEL_REAL_WRAPPER_DEFN(kind, 10, isa)                           \
    REGISTER_KERNEL_REAL_WRAPPER_DEFN(kind, 11, isa)                           \
    REGISTER_KERNEL_REAL_WRAPPER_DEFN(kind, 12, isa)                           \
    REGISTER_KERNEL_REAL_WRAPPER_DEFN(kind, 13, isa)                           \
    REGISTER_KERNEL_REAL_WRAPPER_DEFN(kind, 14, isa)                           \
    REGISTER_KERNEL_REAL_WRAPPER_DEFN(kind, 15, isa)                           \
    REGISTER_KERNEL_REAL_WRAPPER_DEFN(kind, 16, isa)

// C2C Kernels
GET_OPS_CNT_C2C_WRAPPER_ALL_RADICES_DEFN(c)

// C2C Kernels
REGISTER_KERNEL_C2C_WRAPPER_ALL_RADICES_DEFN(c)

/* ---------------- twiddle wrapper macros ---------------- */
#define GET_OPS_CNT_TWID_WRAPPER_DEFN(kind, radix, isa)                        \
    ops_cycles_t get_ops_cnt_twid_##kind##_fft##radix##isa##_wrapper(          \
        FFTZ_UINT8 prec, FFTZ_UINT8 dir)                                       \
    {                                                                          \
        return get_ops_cnt_twid_##kind##_fft##radix##isa(prec, dir);           \
    }

#define GET_OPS_CNT_TWID_WRAPPER_ALL_RADICES_DEFN(kind, isa)                   \
    GET_OPS_CNT_TWID_WRAPPER_DEFN(kind, 2, isa)                                \
    GET_OPS_CNT_TWID_WRAPPER_DEFN(kind, 3, isa)                                \
    GET_OPS_CNT_TWID_WRAPPER_DEFN(kind, 4, isa)                                \
    GET_OPS_CNT_TWID_WRAPPER_DEFN(kind, 5, isa)                                \
    GET_OPS_CNT_TWID_WRAPPER_DEFN(kind, 6, isa)                                \
    GET_OPS_CNT_TWID_WRAPPER_DEFN(kind, 7, isa)                                \
    GET_OPS_CNT_TWID_WRAPPER_DEFN(kind, 8, isa)                                \
    GET_OPS_CNT_TWID_WRAPPER_DEFN(kind, 9, isa)                                \
    GET_OPS_CNT_TWID_WRAPPER_DEFN(kind, 10, isa)                               \
    GET_OPS_CNT_TWID_WRAPPER_DEFN(kind, 11, isa)                               \
    GET_OPS_CNT_TWID_WRAPPER_DEFN(kind, 12, isa)                               \
    GET_OPS_CNT_TWID_WRAPPER_DEFN(kind, 13, isa)                               \
    GET_OPS_CNT_TWID_WRAPPER_DEFN(kind, 14, isa)                               \
    GET_OPS_CNT_TWID_WRAPPER_DEFN(kind, 15, isa)                               \
    GET_OPS_CNT_TWID_WRAPPER_DEFN(kind, 16, isa)

#define REGISTER_KERNEL_TWID_WRAPPER_DEFN(kind, radix, isa)                    \
    kfft_ register_kernel_twid_##kind##_fft##radix##isa##_wrapper(             \
        FFTZ_UINT8 prec, FFTZ_UINT8 dir)                                       \
    {                                                                          \
        return register_kernel_twid_##kind##_fft##radix##isa(prec, dir);       \
    }

#define REGISTER_KERNEL_TWID_WRAPPER_ALL_RADICES_DEFN(kind, isa)               \
    REGISTER_KERNEL_TWID_WRAPPER_DEFN(kind, 2, isa)                            \
    REGISTER_KERNEL_TWID_WRAPPER_DEFN(kind, 3, isa)                            \
    REGISTER_KERNEL_TWID_WRAPPER_DEFN(kind, 4, isa)                            \
    REGISTER_KERNEL_TWID_WRAPPER_DEFN(kind, 5, isa)                            \
    REGISTER_KERNEL_TWID_WRAPPER_DEFN(kind, 6, isa)                            \
    REGISTER_KERNEL_TWID_WRAPPER_DEFN(kind, 7, isa)                            \
    REGISTER_KERNEL_TWID_WRAPPER_DEFN(kind, 8, isa)                            \
    REGISTER_KERNEL_TWID_WRAPPER_DEFN(kind, 9, isa)                            \
    REGISTER_KERNEL_TWID_WRAPPER_DEFN(kind, 10, isa)                           \
    REGISTER_KERNEL_TWID_WRAPPER_DEFN(kind, 11, isa)                           \
    REGISTER_KERNEL_TWID_WRAPPER_DEFN(kind, 12, isa)                           \
    REGISTER_KERNEL_TWID_WRAPPER_DEFN(kind, 13, isa)                           \
    REGISTER_KERNEL_TWID_WRAPPER_DEFN(kind, 14, isa)                           \
    REGISTER_KERNEL_TWID_WRAPPER_DEFN(kind, 15, isa)                           \
    REGISTER_KERNEL_TWID_WRAPPER_DEFN(kind, 16, isa)

// C2C Twiddle Forward Kernels
GET_OPS_CNT_TWID_WRAPPER_ALL_RADICES_DEFN(fwd, c)

// C2C Twiddle Backward Kernels
GET_OPS_CNT_TWID_WRAPPER_ALL_RADICES_DEFN(bwd, c)

// R2C Twiddle Kernels
GET_OPS_CNT_TWID_WRAPPER_ALL_RADICES_DEFN(r2c, c)

// C2R Twiddle Kernels
GET_OPS_CNT_TWID_WRAPPER_ALL_RADICES_DEFN(c2r, c)

// C2C Twiddle Forward Kernels
REGISTER_KERNEL_TWID_WRAPPER_ALL_RADICES_DEFN(fwd, c)

// C2C Twiddle Backward Kernels
REGISTER_KERNEL_TWID_WRAPPER_ALL_RADICES_DEFN(bwd, c)

// R2C Twiddle Kernels
REGISTER_KERNEL_TWID_WRAPPER_ALL_RADICES_DEFN(r2c, c)

// C2R Twiddle Kernels
REGISTER_KERNEL_TWID_WRAPPER_ALL_RADICES_DEFN(c2r, c)

// R2HC Kernels
GET_OPS_CNT_REAL_WRAPPER_ALL_RADICES_DEFN(r2hc, c)
REGISTER_KERNEL_REAL_WRAPPER_ALL_RADICES_DEFN(r2hc, c)

// R2HC-Fused Kernels
GET_OPS_CNT_REAL_WRAPPER_ALL_RADICES_DEFN(r2hcf, c)
REGISTER_KERNEL_REAL_WRAPPER_ALL_RADICES_DEFN(r2hcf, c)

#ifdef ENABLE_AVX128
// C2C AVX128 Kernels
GET_OPS_CNT_C2C_WRAPPER_ALL_RADICES_DEFN(avx128)
REGISTER_KERNEL_C2C_WRAPPER_ALL_RADICES_DEFN(avx128)

// C2C Twiddle AVX128 Kernels
GET_OPS_CNT_TWID_WRAPPER_ALL_RADICES_DEFN(fwd, avx128)
GET_OPS_CNT_TWID_WRAPPER_ALL_RADICES_DEFN(bwd, avx128)
GET_OPS_CNT_TWID_WRAPPER_ALL_RADICES_DEFN(r2c, avx128)
GET_OPS_CNT_TWID_WRAPPER_ALL_RADICES_DEFN(c2r, avx128)
REGISTER_KERNEL_TWID_WRAPPER_ALL_RADICES_DEFN(fwd, avx128)
REGISTER_KERNEL_TWID_WRAPPER_ALL_RADICES_DEFN(bwd, avx128)
REGISTER_KERNEL_TWID_WRAPPER_ALL_RADICES_DEFN(r2c, avx128)
REGISTER_KERNEL_TWID_WRAPPER_ALL_RADICES_DEFN(c2r, avx128)

// R2HC AVX128 Kernels
GET_OPS_CNT_REAL_WRAPPER_ALL_RADICES_DEFN(r2hc, avx128)
REGISTER_KERNEL_REAL_WRAPPER_ALL_RADICES_DEFN(r2hc, avx128)

// R2HC-Fused AVX128 Kernels
GET_OPS_CNT_REAL_WRAPPER_ALL_RADICES_DEFN(r2hcf, avx128)
REGISTER_KERNEL_REAL_WRAPPER_ALL_RADICES_DEFN(r2hcf, avx128)
#endif

#ifdef ENABLE_AVX256
// C2C AVX256 Kernels
GET_OPS_CNT_C2C_WRAPPER_ALL_RADICES_DEFN(avx256)
REGISTER_KERNEL_C2C_WRAPPER_ALL_RADICES_DEFN(avx256)

// C2C Twiddle AVX256 Kernels
GET_OPS_CNT_TWID_WRAPPER_ALL_RADICES_DEFN(fwd, avx256)
GET_OPS_CNT_TWID_WRAPPER_ALL_RADICES_DEFN(bwd, avx256)
GET_OPS_CNT_TWID_WRAPPER_ALL_RADICES_DEFN(r2c, avx256)
GET_OPS_CNT_TWID_WRAPPER_ALL_RADICES_DEFN(c2r, avx256)
REGISTER_KERNEL_TWID_WRAPPER_ALL_RADICES_DEFN(fwd, avx256)
REGISTER_KERNEL_TWID_WRAPPER_ALL_RADICES_DEFN(bwd, avx256)
REGISTER_KERNEL_TWID_WRAPPER_ALL_RADICES_DEFN(r2c, avx256)
REGISTER_KERNEL_TWID_WRAPPER_ALL_RADICES_DEFN(c2r, avx256)

// R2HC AVX256 Kernels
GET_OPS_CNT_REAL_WRAPPER_ALL_RADICES_DEFN(r2hc, avx256)
REGISTER_KERNEL_REAL_WRAPPER_ALL_RADICES_DEFN(r2hc, avx256)

// R2HC-Fused AVX256 Kernels
GET_OPS_CNT_REAL_WRAPPER_ALL_RADICES_DEFN(r2hcf, avx256)
REGISTER_KERNEL_REAL_WRAPPER_ALL_RADICES_DEFN(r2hcf, avx256)
#endif

#ifdef ENABLE_AVX512
// C2C AVX512 Kernels
GET_OPS_CNT_C2C_WRAPPER_ALL_RADICES_DEFN(avx512)
REGISTER_KERNEL_C2C_WRAPPER_ALL_RADICES_DEFN(avx512)

// C2C Twiddle AVX512 Kernels
GET_OPS_CNT_TWID_WRAPPER_ALL_RADICES_DEFN(fwd, avx512)
GET_OPS_CNT_TWID_WRAPPER_ALL_RADICES_DEFN(bwd, avx512)
GET_OPS_CNT_TWID_WRAPPER_ALL_RADICES_DEFN(r2c, avx512)
GET_OPS_CNT_TWID_WRAPPER_ALL_RADICES_DEFN(c2r, avx512)
REGISTER_KERNEL_TWID_WRAPPER_ALL_RADICES_DEFN(fwd, avx512)
REGISTER_KERNEL_TWID_WRAPPER_ALL_RADICES_DEFN(bwd, avx512)
REGISTER_KERNEL_TWID_WRAPPER_ALL_RADICES_DEFN(r2c, avx512)
REGISTER_KERNEL_TWID_WRAPPER_ALL_RADICES_DEFN(c2r, avx512)

// R2HC AVX512 Kernels
GET_OPS_CNT_REAL_WRAPPER_ALL_RADICES_DEFN(r2hc, avx512)
REGISTER_KERNEL_REAL_WRAPPER_ALL_RADICES_DEFN(r2hc, avx512)

// R2HC-Fused AVX512 Kernels
GET_OPS_CNT_REAL_WRAPPER_ALL_RADICES_DEFN(r2hcf, avx512)
REGISTER_KERNEL_REAL_WRAPPER_ALL_RADICES_DEFN(r2hcf, avx512)
#endif

#undef REGISTER_KERNEL_TWID_WRAPPER_ALL_RADICES_DEFN
#undef REGISTER_KERNEL_TWID_WRAPPER_DEFN
#undef GET_OPS_CNT_TWID_WRAPPER_ALL_RADICES_DEFN
#undef GET_OPS_CNT_TWID_WRAPPER_DEFN
#undef REGISTER_KERNEL_C2C_WRAPPER_ALL_RADICES_DEFN
#undef REGISTER_KERNEL_C2C_WRAPPER_DEFN
#undef REGISTER_KERNEL_REAL_WRAPPER_ALL_RADICES_DEFN
#undef REGISTER_KERNEL_REAL_WRAPPER_DEFN
#undef GET_OPS_CNT_C2C_WRAPPER_ALL_RADICES_DEFN
#undef GET_OPS_CNT_C2C_WRAPPER_DEFN
#undef GET_OPS_CNT_REAL_WRAPPER_ALL_RADICES_DEFN
#undef GET_OPS_CNT_REAL_WRAPPER_DEFN
/* ---------------- kernels : permuted_copy_* ---------------- */

FFTZ_VOID permuted_copy_c_fp32_wrapper(FFTZ_VOID *in, FFTZ_VOID *out,
                                       FFTZ_INTP n, FFTZ_INTP size,
                                       FFTZ_INTP in_stride,
                                       FFTZ_INTP out_stride,
                                       FFTZ_INTP v_in_stride,
                                       FFTZ_INTP v_out_stride)
{
    permuted_copy_c_fp32(in, out, n, size, in_stride, out_stride, v_in_stride,
                         v_out_stride);
}
FFTZ_VOID permuted_copy_c_fp64_wrapper(FFTZ_VOID *in, FFTZ_VOID *out,
                                       FFTZ_INTP n, FFTZ_INTP size,
                                       FFTZ_INTP in_stride,
                                       FFTZ_INTP out_stride,
                                       FFTZ_INTP v_in_stride,
                                       FFTZ_INTP v_out_stride)
{
    permuted_copy_c_fp64(in, out, n, size, in_stride, out_stride, v_in_stride,
                         v_out_stride);
}

/* ---------------- memory allocators/destroys ---------------- */

aoclfftz_solution_t *alloc_solution_wrapper(FFTZ_INT32 vec_rank,
                                            FFTZ_INT32 dim_rank)
{
    return alloc_solution(vec_rank, dim_rank);
}
aoclfftz_selector_t *alloc_selector_wrapper(FFTZ_INT32 vec_rank,
                                            FFTZ_INT32 dim_rank,
                                            kernel_tables_t *kernel_tables,
                                            FFTZ_UINT8 *has_nested)
{
    return alloc_selector(vec_rank, dim_rank, kernel_tables,
                          has_nested);
}
FFTZ_VOID *alloc_twiddle_buffer_wrapper(FFTZ_INTP size, FFTZ_UINT32 dt_prec)
{
    return alloc_twiddle_buffer(size, dt_prec);
}

FFTZ_VOID destroy_selector_wrapper(aoclfftz_selector_t *sel)
{
    destroy_selector(sel);
}
FFTZ_VOID destroy_solution_wrapper(aoclfftz_solution_t *sol)
{
    destroy_solution(sol);
}
FFTZ_VOID destroy_decomp_scheme_wrapper(aoclfftz_decomp_scheme_t *decomp_scheme)
{
    destroy_decomp_scheme(decomp_scheme);
}
FFTZ_VOID destroy_handle_wrapper(FFTZ_VOID *handle)
{
    destroy_handle(handle);
}

/* ---------------- strides wrapper ---------------- */
FFTZ_VOID populate_stride_array_wrapper(FFTZ_INTP *strides,
                                        FFTZ_INTP stride_val, FFTZ_INTP n,
                                        FFTZ_UINT8 compute_half_complex,
                                        FFTZ_UINT8 adjust_to_full_complex)
{
    populate_stride_array(strides, stride_val, n, compute_half_complex,
                          adjust_to_full_complex);
}

/* ---------------- fused strides wrapper ---------------- */
FFTZ_VOID prepare_fused_kernel_strides_wrapper(FFTZ_INTP *strides,
                                               FFTZ_INTP radix,
                                               FFTZ_INTP offset)
{
    prepare_fused_kernel_strides(strides, radix, offset);
}

// Transpose wrappers
#define TRANSPOSE_WRAPPER_DEFN(kernel_name, TYPE, isa)                         \
    FFTZ_VOID CONCAT(FUNC(kernel_name, TYPE, isa),                             \
                     _wrapper)(TRANSPOSE_KERNEL_ARGS)                          \
    {                                                                          \
        FUNC(kernel_name, TYPE, c)                                             \
        (in_ptr, out_ptr, row_metadata, column_metadata, aux_mem);             \
    }

#define TRANSPOSE_WRAPPER_ALL_TYPES_DEFN(kernel_name, isa)                     \
    TRANSPOSE_WRAPPER_DEFN(kernel_name, FFTZ_FLOAT, isa)                       \
    TRANSPOSE_WRAPPER_DEFN(kernel_name, FFTZ_DOUBLE, isa)                      \
    TRANSPOSE_WRAPPER_DEFN(kernel_name, aoclfftz_complex_f_t, isa)             \
    TRANSPOSE_WRAPPER_DEFN(kernel_name, aoclfftz_complex_d_t, isa)

TRANSPOSE_WRAPPER_ALL_TYPES_DEFN(tiq_iterative, c)
TRANSPOSE_WRAPPER_ALL_TYPES_DEFN(tisq_iterative, c)
TRANSPOSE_WRAPPER_ALL_TYPES_DEFN(tiq_recursive_buf, c)
TRANSPOSE_WRAPPER_ALL_TYPES_DEFN(tir_cycles, c)
TRANSPOSE_WRAPPER_ALL_TYPES_DEFN(tisr_cycles, c)
TRANSPOSE_WRAPPER_ALL_TYPES_DEFN(tos_iterative, c)
TRANSPOSE_WRAPPER_ALL_TYPES_DEFN(tos_blocked, c)

// Fused four-step twiddle + transpose register wrappers.
fused_twiddle_transpose_
register_fused_twiddle_transpose_c_wrapper(FFTZ_UINT8 prec, FFTZ_UINT8 dir)
{
    return register_fused_twiddle_transpose_c(prec, dir);
}
#ifdef ENABLE_AVX128
fused_twiddle_transpose_
register_fused_twiddle_transpose_avx128_wrapper(FFTZ_UINT8 prec, FFTZ_UINT8 dir)
{
    return register_fused_twiddle_transpose_avx128(prec, dir);
}
#endif
#ifdef ENABLE_AVX256
fused_twiddle_transpose_
register_fused_twiddle_transpose_avx256_wrapper(FFTZ_UINT8 prec, FFTZ_UINT8 dir)
{
    return register_fused_twiddle_transpose_avx256(prec, dir);
}
#endif
#ifdef ENABLE_AVX512
fused_twiddle_transpose_
register_fused_twiddle_transpose_avx512_wrapper(FFTZ_UINT8 prec, FFTZ_UINT8 dir)
{
    return register_fused_twiddle_transpose_avx512(prec, dir);
}
#endif

// for the gtests, we want to use the in-memory twiddle factors
// so we define IN_MEMORY_TWIDDLE_FACTORS to 1 if not explicitly set/defined
#if !defined(IN_MEMORY_TWIDDLE_FACTORS)
#define IN_MEMORY_TWIDDLE_FACTORS 1
#elif IN_MEMORY_TWIDDLE_FACTORS == 0
#undef IN_MEMORY_TWIDDLE_FACTORS
#define IN_MEMORY_TWIDDLE_FACTORS 1
#endif

// twiddle buffer setup wrappers
EXPORT_SYM_DYN FFTZ_VOID compute_twiddle_buffer_float_wrapper(
    FFTZ_VOID *twiddle_buffer, FFTZ_INTP r, FFTZ_INTP m,
    FFTZ_INTP register_width, FFTZ_INTP load_multi_cols)
{
    compute_twiddle_buffer(twiddle_buffer, r, m, register_width,
                           load_multi_cols, DT_FLOAT);
}

EXPORT_SYM_DYN FFTZ_VOID compute_twiddle_buffer_double_wrapper(
    FFTZ_VOID *twiddle_buffer, FFTZ_INTP r, FFTZ_INTP m,
    FFTZ_INTP register_width, FFTZ_INTP load_multi_cols)
{
    compute_twiddle_buffer(twiddle_buffer, r, m, register_width,
                           load_multi_cols, DT_DOUBLE);
}
