/**
 * Copyright (C) 2023, Advanced Micro Devices. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/** @file aoclfftz_core_wrapper.c
 *
 *  @brief Contains wrapper function definitions for core funtions
 *  with dllexport attribute.
 *
 *  This file contains the wrapper function definitions for core functions
 *  with `__declspec(dllexport)` attribute for Windows compatibility.
 *
 *  @author Srirammaswamy Srinivasan
 */

#include "gtest/aoclfftz_core_wrapper.h"

/* ---------------- kernels : get_opt_cnt_fft* ---------------- */

ops_cycles_t get_ops_cnt_fft2c_wrapper(INT32 precision)
{
    return get_ops_cnt_fft2c(precision);
}
ops_cycles_t get_ops_cnt_fft3c_wrapper(INT32 precision)
{
    return get_ops_cnt_fft3c(precision);
}
ops_cycles_t get_ops_cnt_fft4c_wrapper(INT32 precision)
{
    return get_ops_cnt_fft4c(precision);
}
ops_cycles_t get_ops_cnt_fft5c_wrapper(INT32 precision)
{
    return get_ops_cnt_fft5c(precision);
}
ops_cycles_t get_ops_cnt_fft6c_wrapper(INT32 precision)
{
    return get_ops_cnt_fft6c(precision);
}
ops_cycles_t get_ops_cnt_fft7c_wrapper(INT32 precision)
{
    return get_ops_cnt_fft7c(precision);
}
ops_cycles_t get_ops_cnt_fft8c_wrapper(INT32 precision)
{
    return get_ops_cnt_fft8c(precision);
}
ops_cycles_t get_ops_cnt_fft9c_wrapper(INT32 precision)
{
    return get_ops_cnt_fft9c(precision);
}
ops_cycles_t get_ops_cnt_fft10c_wrapper(INT32 precision)
{
    return get_ops_cnt_fft10c(precision);
}
ops_cycles_t get_ops_cnt_fft11c_wrapper(INT32 precision)
{
    return get_ops_cnt_fft11c(precision);
}
ops_cycles_t get_ops_cnt_fft12c_wrapper(INT32 precision)
{
    return get_ops_cnt_fft12c(precision);
}
ops_cycles_t get_ops_cnt_fft13c_wrapper(INT32 precision)
{
    return get_ops_cnt_fft13c(precision);
}
ops_cycles_t get_ops_cnt_fft14c_wrapper(INT32 precision)
{
    return get_ops_cnt_fft14c(precision);
}
ops_cycles_t get_ops_cnt_fft15c_wrapper(INT32 precision)
{
    return get_ops_cnt_fft15c(precision);
}
ops_cycles_t get_ops_cnt_fft16c_wrapper(INT32 precision)
{
    return get_ops_cnt_fft16c(precision);
}

#ifdef ENABLE_AVX128
ops_cycles_t get_ops_cnt_fft2avx128_wrapper(INT32 precision)
{
    return get_ops_cnt_fft2avx128(precision);
}
ops_cycles_t get_ops_cnt_fft3avx128_wrapper(INT32 precision)
{
    return get_ops_cnt_fft3avx128(precision);
}
ops_cycles_t get_ops_cnt_fft4avx128_wrapper(INT32 precision)
{
    return get_ops_cnt_fft4avx128(precision);
}
ops_cycles_t get_ops_cnt_fft5avx128_wrapper(INT32 precision)
{
    return get_ops_cnt_fft5avx128(precision);
}
ops_cycles_t get_ops_cnt_fft6avx128_wrapper(INT32 precision)
{
    return get_ops_cnt_fft6avx128(precision);
}
ops_cycles_t get_ops_cnt_fft7avx128_wrapper(INT32 precision)
{
    return get_ops_cnt_fft7avx128(precision);
}
ops_cycles_t get_ops_cnt_fft8avx128_wrapper(INT32 precision)
{
    return get_ops_cnt_fft8avx128(precision);
}
ops_cycles_t get_ops_cnt_fft9avx128_wrapper(INT32 precision)
{
    return get_ops_cnt_fft9avx128(precision);
}
ops_cycles_t get_ops_cnt_fft10avx128_wrapper(INT32 precision)
{
    return get_ops_cnt_fft10avx128(precision);
}
ops_cycles_t get_ops_cnt_fft11avx128_wrapper(INT32 precision)
{
    return get_ops_cnt_fft11avx128(precision);
}
ops_cycles_t get_ops_cnt_fft12avx128_wrapper(INT32 precision)
{
    return get_ops_cnt_fft12avx128(precision);
}
ops_cycles_t get_ops_cnt_fft14avx128_wrapper(INT32 precision)
{
    return get_ops_cnt_fft14avx128(precision);
}
ops_cycles_t get_ops_cnt_fft15avx128_wrapper(INT32 precision)
{
    return get_ops_cnt_fft15avx128(precision);
}
ops_cycles_t get_ops_cnt_fft16avx128_wrapper(INT32 precision)
{
    return get_ops_cnt_fft16avx128(precision);
}
#endif

#ifdef ENABLE_AVX256
ops_cycles_t get_ops_cnt_fft2avx256_wrapper(INT32 precision)
{
    return get_ops_cnt_fft2avx256(precision);
}
ops_cycles_t get_ops_cnt_fft3avx256_wrapper(INT32 precision)
{
    return get_ops_cnt_fft3avx256(precision);
}
ops_cycles_t get_ops_cnt_fft4avx256_wrapper(INT32 precision)
{
    return get_ops_cnt_fft4avx256(precision);
}
ops_cycles_t get_ops_cnt_fft5avx256_wrapper(INT32 precision)
{
    return get_ops_cnt_fft5avx256(precision);
}
ops_cycles_t get_ops_cnt_fft6avx256_wrapper(INT32 precision)
{
    return get_ops_cnt_fft6avx256(precision);
}
ops_cycles_t get_ops_cnt_fft7avx256_wrapper(INT32 precision)
{
    return get_ops_cnt_fft7avx256(precision);
}
ops_cycles_t get_ops_cnt_fft8avx256_wrapper(INT32 precision)
{
    return get_ops_cnt_fft8avx256(precision);
}
ops_cycles_t get_ops_cnt_fft9avx256_wrapper(INT32 precision)
{
    return get_ops_cnt_fft9avx256(precision);
}
ops_cycles_t get_ops_cnt_fft10avx256_wrapper(INT32 precision)
{
    return get_ops_cnt_fft10avx256(precision);
}
ops_cycles_t get_ops_cnt_fft11avx256_wrapper(INT32 precision)
{
    return get_ops_cnt_fft11avx256(precision);
}
ops_cycles_t get_ops_cnt_fft12avx256_wrapper(INT32 precision)
{
    return get_ops_cnt_fft12avx256(precision);
}
ops_cycles_t get_ops_cnt_fft14avx256_wrapper(INT32 precision)
{
    return get_ops_cnt_fft14avx256(precision);
}
ops_cycles_t get_ops_cnt_fft15avx256_wrapper(INT32 precision)
{
    return get_ops_cnt_fft15avx256(precision);
}
ops_cycles_t get_ops_cnt_fft16avx256_wrapper(INT32 precision)
{
    return get_ops_cnt_fft16avx256(precision);
}
#endif

#ifdef ENABLE_AVX512
ops_cycles_t get_ops_cnt_fft2avx512_wrapper(INT32 precision)
{
    return get_ops_cnt_fft2avx512(precision);
}
#endif

/* ---------------- kernels : register_kernel_fft* ---------------- */

kfft_ register_kernel_fft2c_wrapper(INT32 precision)
{
    return register_kernel_fft2c(precision);
}
kfft_ register_kernel_fft3c_wrapper(INT32 precision)
{
    return register_kernel_fft3c(precision);
}
kfft_ register_kernel_fft4c_wrapper(INT32 precision)
{
    return register_kernel_fft4c(precision);
}
kfft_ register_kernel_fft5c_wrapper(INT32 precision)
{
    return register_kernel_fft5c(precision);
}
kfft_ register_kernel_fft6c_wrapper(INT32 precision)
{
    return register_kernel_fft6c(precision);
}
kfft_ register_kernel_fft7c_wrapper(INT32 precision)
{
    return register_kernel_fft7c(precision);
}
kfft_ register_kernel_fft8c_wrapper(INT32 precision)
{
    return register_kernel_fft8c(precision);
}
kfft_ register_kernel_fft9c_wrapper(INT32 precision)
{
    return register_kernel_fft9c(precision);
}
kfft_ register_kernel_fft10c_wrapper(INT32 precision)
{
    return register_kernel_fft10c(precision);
}
kfft_ register_kernel_fft11c_wrapper(INT32 precision)
{
    return register_kernel_fft11c(precision);
}
kfft_ register_kernel_fft12c_wrapper(INT32 precision)
{
    return register_kernel_fft12c(precision);
}
kfft_ register_kernel_fft13c_wrapper(INT32 precision)
{
    return register_kernel_fft13c(precision);
}
kfft_ register_kernel_fft14c_wrapper(INT32 precision)
{
    return register_kernel_fft14c(precision);
}
kfft_ register_kernel_fft15c_wrapper(INT32 precision)
{
    return register_kernel_fft15c(precision);
}
kfft_ register_kernel_fft16c_wrapper(INT32 precision)
{
    return register_kernel_fft16c(precision);
}

#ifdef ENABLE_AVX128
kfft_ register_kernel_fft2avx128_wrapper(INT32 precision)
{
    return register_kernel_fft2avx128(precision);
}
kfft_ register_kernel_fft3avx128_wrapper(INT32 precision)
{
    return register_kernel_fft3avx128(precision);
}
kfft_ register_kernel_fft4avx128_wrapper(INT32 precision)
{
    return register_kernel_fft4avx128(precision);
}
kfft_ register_kernel_fft5avx128_wrapper(INT32 precision)
{
    return register_kernel_fft5avx128(precision);
}
kfft_ register_kernel_fft6avx128_wrapper(INT32 precision)
{
    return register_kernel_fft6avx128(precision);
}
kfft_ register_kernel_fft7avx128_wrapper(INT32 precision)
{
    return register_kernel_fft7avx128(precision);
}
kfft_ register_kernel_fft8avx128_wrapper(INT32 precision)
{
    return register_kernel_fft8avx128(precision);
}
kfft_ register_kernel_fft9avx128_wrapper(INT32 precision)
{
    return register_kernel_fft9avx128(precision);
}
kfft_ register_kernel_fft10avx128_wrapper(INT32 precision)
{
    return register_kernel_fft10avx128(precision);
}
kfft_ register_kernel_fft11avx128_wrapper(INT32 precision)
{
    return register_kernel_fft11avx128(precision);
}
kfft_ register_kernel_fft12avx128_wrapper(INT32 precision)
{
    return register_kernel_fft12avx128(precision);
}
kfft_ register_kernel_fft14avx128_wrapper(INT32 precision)
{
    return register_kernel_fft14avx128(precision);
}
kfft_ register_kernel_fft15avx128_wrapper(INT32 precision)
{
    return register_kernel_fft15avx128(precision);
}
kfft_ register_kernel_fft16avx128_wrapper(INT32 precision)
{
    return register_kernel_fft16avx128(precision);
}
#endif

#ifdef ENABLE_AVX256
kfft_ register_kernel_fft2avx256_wrapper(INT32 precision)
{
    return register_kernel_fft2avx256(precision);
}
kfft_ register_kernel_fft3avx256_wrapper(INT32 precision)
{
    return register_kernel_fft3avx256(precision);
}
kfft_ register_kernel_fft4avx256_wrapper(INT32 precision)
{
    return register_kernel_fft4avx256(precision);
}
kfft_ register_kernel_fft5avx256_wrapper(INT32 precision)
{
    return register_kernel_fft5avx256(precision);
}
kfft_ register_kernel_fft6avx256_wrapper(INT32 precision)
{
    return register_kernel_fft6avx256(precision);
}
kfft_ register_kernel_fft7avx256_wrapper(INT32 precision)
{
    return register_kernel_fft7avx256(precision);
}
kfft_ register_kernel_fft8avx256_wrapper(INT32 precision)
{
    return register_kernel_fft8avx256(precision);
}
kfft_ register_kernel_fft9avx256_wrapper(INT32 precision)
{
    return register_kernel_fft9avx256(precision);
}
kfft_ register_kernel_fft10avx256_wrapper(INT32 precision)
{
    return register_kernel_fft10avx256(precision);
}
kfft_ register_kernel_fft11avx256_wrapper(INT32 precision)
{
    return register_kernel_fft11avx256(precision);
}
kfft_ register_kernel_fft12avx256_wrapper(INT32 precision)
{
    return register_kernel_fft12avx256(precision);
}
kfft_ register_kernel_fft14avx256_wrapper(INT32 precision)
{
    return register_kernel_fft14avx256(precision);
}
kfft_ register_kernel_fft15avx256_wrapper(INT32 precision)
{
    return register_kernel_fft15avx256(precision);
}
kfft_ register_kernel_fft16avx256_wrapper(INT32 precision)
{
    return register_kernel_fft16avx256(precision);
}
#endif

#ifdef ENABLE_AVX512
kfft_ register_kernel_fft2avx512_wrapper(INT32 precision)
{
    return register_kernel_fft2avx512(precision);
}
#endif

/* ---------------- kernels : permuted_copy_* ---------------- */

VOID permuted_copy_c_fp32_wrapper(VOID *in, VOID *out, INTP n, INTP radix,
                                  aoclfftz_strides_t *strides)
{
    permuted_copy_c_fp32(in, out, n, radix, strides);
}
VOID permuted_copy_c_fp64_wrapper(VOID *in, VOID *out, INTP n, INTP radix,
                                  aoclfftz_strides_t *strides)
{
    permuted_copy_c_fp64(in, out, n, radix, strides);
}

/* ---------------- memory allocators/destroys ---------------- */

aoclfftz_decomp_scheme_t *alloc_decomp_scheme_wrapper(INT32 vec_rank,
                                                      INT32 dim_rank)
{
    return alloc_decomp_scheme(vec_rank, dim_rank);
}
aoclfftz_solution_t *alloc_solution_wrapper(INT32 vec_rank, INT32 dim_rank)
{
    return alloc_solution(vec_rank, dim_rank);
}
aoclfftz_selector_t *alloc_selector_wrapper(INT32 vec_rank, INT32 dim_rank)
{
    return alloc_selector(vec_rank, dim_rank);
}
#if IN_MEMORY_TWIDDLE_FACTORS == 1
VOID *alloc_twiddle_for_solution_wrapper(UINT32 rad_size, UINT32 dt_prec)
{
    return alloc_twiddle_for_solution(rad_size, dt_prec);
}
#endif
VOID destroy_selector_wrapper(aoclfftz_selector_t *sel)
{
    destroy_selector(sel);
}
VOID destroy_solution_wrapper(aoclfftz_solution_t *sol)
{
    destroy_solution(sol);
}
VOID destroy_decomp_scheme_wrapper(aoclfftz_decomp_scheme_t *decomp_scheme)
{
    destroy_decomp_scheme(decomp_scheme);
}
VOID destroy_handle_wrapper(VOID *handle)
{
    destroy_handle(handle);
}

// Transpose wrappers
#define TRANSPOSE_WRAPPER_DEFN(kernel_name, TYPE, isa)                         \
    VOID CONCAT(FUNC(kernel_name, TYPE, isa), _wrapper)(TRANSPOSE_KERNEL_ARGS) \
    {                                                                          \
        FUNC(kernel_name, TYPE, c)(in_ptr, out_ptr, row_metadata,              \
                                   column_metadata, aux_mem);                  \
    }

#define TRANSPOSE_WRAPPER_ALL_TYPES_DEFN(kernel_name, isa)                     \
    TRANSPOSE_WRAPPER_DEFN(kernel_name, FLOAT, isa);                           \
    TRANSPOSE_WRAPPER_DEFN(kernel_name, DOUBLE, isa);                          \
    TRANSPOSE_WRAPPER_DEFN(kernel_name, aoclfftz_complex_f_t, isa);            \
    TRANSPOSE_WRAPPER_DEFN(kernel_name, aoclfftz_complex_d_t, isa);

TRANSPOSE_WRAPPER_ALL_TYPES_DEFN(tiq_iterative, c);
TRANSPOSE_WRAPPER_ALL_TYPES_DEFN(tisq_iterative, c);
