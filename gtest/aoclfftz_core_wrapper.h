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
#include "core/kernels/kernel.h"
#include "core/solvers/solver.h"
#include "selector/selector.h"

// Re-delcaring this struct to avoid using core/kernels/kernel_list.h file
typedef struct
{
    k_register_kernel_ k_register_kernel;
    k_ops_cnt_ k_ops_cnt;
    UINT32 radix;
} wrapper_kernel_fp_list;

/* ---------------- kernels : get_opt_cnt_fft* ---------------- */

EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft2c_wrapper(INT32 precision);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft3c_wrapper(INT32 precision);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft4c_wrapper(INT32 precision);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft5c_wrapper(INT32 precision);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft6c_wrapper(INT32 precision);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft7c_wrapper(INT32 precision);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft8c_wrapper(INT32 precision);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft9c_wrapper(INT32 precision);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft10c_wrapper(INT32 precision);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft11c_wrapper(INT32 precision);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft12c_wrapper(INT32 precision);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft13c_wrapper(INT32 precision);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft14c_wrapper(INT32 precision);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft15c_wrapper(INT32 precision);
EXPORT_SYM_DYN ops_cycles_t get_ops_cnt_fft16c_wrapper(INT32 precision);

/* ---------------- kernels : register_kernel_fft* ---------------- */

EXPORT_SYM_DYN kfft_ register_kernel_fft2c_wrapper(INT32 precision);
EXPORT_SYM_DYN kfft_ register_kernel_fft3c_wrapper(INT32 precision);
EXPORT_SYM_DYN kfft_ register_kernel_fft4c_wrapper(INT32 precision);
EXPORT_SYM_DYN kfft_ register_kernel_fft5c_wrapper(INT32 precision);
EXPORT_SYM_DYN kfft_ register_kernel_fft6c_wrapper(INT32 precision);
EXPORT_SYM_DYN kfft_ register_kernel_fft7c_wrapper(INT32 precision);
EXPORT_SYM_DYN kfft_ register_kernel_fft8c_wrapper(INT32 precision);
EXPORT_SYM_DYN kfft_ register_kernel_fft9c_wrapper(INT32 precision);
EXPORT_SYM_DYN kfft_ register_kernel_fft10c_wrapper(INT32 precision);
EXPORT_SYM_DYN kfft_ register_kernel_fft11c_wrapper(INT32 precision);
EXPORT_SYM_DYN kfft_ register_kernel_fft12c_wrapper(INT32 precision);
EXPORT_SYM_DYN kfft_ register_kernel_fft13c_wrapper(INT32 precision);
EXPORT_SYM_DYN kfft_ register_kernel_fft14c_wrapper(INT32 precision);
EXPORT_SYM_DYN kfft_ register_kernel_fft15c_wrapper(INT32 precision);
EXPORT_SYM_DYN kfft_ register_kernel_fft16c_wrapper(INT32 precision);

/* ---------------- kernels : permuted_copy_* ---------------- */

EXPORT_SYM_DYN VOID permuted_copy_c_fp32_wrapper(VOID *in, VOID *out, INTP n,
                                                 INTP radix,
                                                 aoclfftz_strides_t *strides);
EXPORT_SYM_DYN VOID permuted_copy_c_fp64_wrapper(VOID *in, VOID *out, INTP n,
                                                 INTP radix,
                                                 aoclfftz_strides_t *strides);

/* ---------------- memory allocators/destroys ---------------- */

EXPORT_SYM_DYN
aoclfftz_decomp_scheme_t *alloc_decomp_scheme_wrapper(INT32 vec_rank,
                                                      INT32 dim_rank);
EXPORT_SYM_DYN aoclfftz_solution_t *alloc_solution_wrapper(INT32 vec_rank,
                                                           INT32 dim_rank);
EXPORT_SYM_DYN aoclfftz_selector_t *alloc_selector_wrapper(INT32 vec_rank,
                                                           INT32 dim_rank);
EXPORT_SYM_DYN VOID *alloc_twiddle_for_solution_wrapper(UINT32 rad_size,
                                                        UINT32 dt_prec);
EXPORT_SYM_DYN VOID destroy_selector_wrapper(aoclfftz_selector_t *sel);
EXPORT_SYM_DYN VOID destroy_solution_wrapper(aoclfftz_solution_t *sol);
EXPORT_SYM_DYN
VOID destroy_decomp_scheme_wrapper(aoclfftz_decomp_scheme_t *decomp_scheme);
EXPORT_SYM_DYN VOID destroy_handle_wrapper(VOID *handle);

/* ---------------- wrapper kernel tables ---------------- */

static wrapper_kernel_fp_list
    wrapper_kernels_c[NUM_KERNELS_IN_EACH_CATEGORY] =
{
    {register_kernel_fft2c_wrapper, get_ops_cnt_fft2c_wrapper, 2},    // radix-2
    {register_kernel_fft3c_wrapper, get_ops_cnt_fft3c_wrapper, 3},    // radix-3
    {register_kernel_fft4c_wrapper, get_ops_cnt_fft4c_wrapper, 4},    // radix-4
    {register_kernel_fft5c_wrapper, get_ops_cnt_fft5c_wrapper, 5},    // radix-5
    {register_kernel_fft6c_wrapper, get_ops_cnt_fft6c_wrapper, 6},    // radix-6
    {register_kernel_fft7c_wrapper, get_ops_cnt_fft7c_wrapper, 7},    // radix-7
    {register_kernel_fft8c_wrapper, get_ops_cnt_fft8c_wrapper, 8},    // radix-8
    {register_kernel_fft9c_wrapper, get_ops_cnt_fft9c_wrapper, 9},    // radix-9
    {register_kernel_fft10c_wrapper, get_ops_cnt_fft10c_wrapper, 10}, // radix-10
    {register_kernel_fft11c_wrapper, get_ops_cnt_fft11c_wrapper, 11}, // radix-11
    {register_kernel_fft12c_wrapper, get_ops_cnt_fft12c_wrapper, 12}, // radix-12
    {register_kernel_fft13c_wrapper, get_ops_cnt_fft13c_wrapper, 13}, // radix-13
    {register_kernel_fft14c_wrapper, get_ops_cnt_fft14c_wrapper, 14}, // radix-14
    {register_kernel_fft15c_wrapper, get_ops_cnt_fft15c_wrapper, 15}, // radix-15
    {register_kernel_fft16c_wrapper, get_ops_cnt_fft16c_wrapper, 16}, // radix-16
    {NULL, NULL, 20},                                                 // radix-20
    {NULL, NULL, 25},                                                 // radix-25
    {NULL, NULL, 32},                                                 // radix-32
    {NULL, NULL, 64}                                                  // radix-64
};
#ifdef ENABLE_AVX128
static wrapper_kernel_fp_list
    wrapper_kernels_avx128[NUM_KERNELS_IN_EACH_CATEGORY] =
{
    {NULL, NULL, 2},  // radix-2
    {NULL, NULL, 3},  // radix-3
    {NULL, NULL, 4},  // radix-4
    {NULL, NULL, 5},  // radix-5
    {NULL, NULL, 6},  // radix-6
    {NULL, NULL, 7},  // radix-7
    {NULL, NULL, 8},  // radix-8
    {NULL, NULL, 9},  // radix-9
    {NULL, NULL, 10}, // radix-10
    {NULL, NULL, 11}, // radix-11
    {NULL, NULL, 12}, // radix-12
    {NULL, NULL, 13}, // radix-13
    {NULL, NULL, 14}, // radix-14
    {NULL, NULL, 15}, // radix-15
    {NULL, NULL, 16}, // radix-16
    {NULL, NULL, 20}, // radix-20
    {NULL, NULL, 25}, // radix-25
    {NULL, NULL, 32}, // radix-32
    {NULL, NULL, 64}  // radix-64
};
#endif
#ifdef ENABLE_AVX256
static wrapper_kernel_fp_list
    wrapper_kernels_avx256[NUM_KERNELS_IN_EACH_CATEGORY] =
{
    {NULL, NULL, 2},  // radix-2
    {NULL, NULL, 3},  // radix-3
    {NULL, NULL, 4},  // radix-4
    {NULL, NULL, 5},  // radix-5
    {NULL, NULL, 6},  // radix-6
    {NULL, NULL, 7},  // radix-7
    {NULL, NULL, 8},  // radix-8
    {NULL, NULL, 9},  // radix-9
    {NULL, NULL, 10}, // radix-10
    {NULL, NULL, 11}, // radix-11
    {NULL, NULL, 12}, // radix-12
    {NULL, NULL, 13}, // radix-13
    {NULL, NULL, 14}, // radix-14
    {NULL, NULL, 15}, // radix-15
    {NULL, NULL, 16}, // radix-16
    {NULL, NULL, 20}, // radix-20
    {NULL, NULL, 25}, // radix-25
    {NULL, NULL, 32}, // radix-32
    {NULL, NULL, 64}  // radix-64
};
#endif
#ifdef ENABLE_AVX512
static wrapper_kernel_fp_list
    wrapper_kernels_avx512[NUM_KERNELS_IN_EACH_CATEGORY] =
{
    {NULL, NULL, 2},  // radix-2
    {NULL, NULL, 3},  // radix-3
    {NULL, NULL, 4},  // radix-4
    {NULL, NULL, 5},  // radix-5
    {NULL, NULL, 6},  // radix-6
    {NULL, NULL, 7},  // radix-7
    {NULL, NULL, 8},  // radix-8
    {NULL, NULL, 9},  // radix-9
    {NULL, NULL, 10}, // radix-10
    {NULL, NULL, 11}, // radix-11
    {NULL, NULL, 12}, // radix-12
    {NULL, NULL, 13}, // radix-13
    {NULL, NULL, 14}, // radix-14
    {NULL, NULL, 15}, // radix-15
    {NULL, NULL, 16}, // radix-16
    {NULL, NULL, 20}, // radix-20
    {NULL, NULL, 25}, // radix-25
    {NULL, NULL, 32}, // radix-32
    {NULL, NULL, 64}  // radix-64
};
#endif

#endif // AOCLFFTZ_CORE_WRAPPER_H