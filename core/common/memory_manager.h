// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file memory_manager.h
 *
 *  @brief Declares memory allocation and management functions of AOCL-FFTZ.
 *
 *  This file contains the function declarations for performing memory
 *  allocation management of different modules of the AOCL-FFTZ library.
 *
 *  @author S. Biplab Raut
 */

#ifndef MEMORY_MANAGER_H
#define MEMORY_MANAGER_H

#include "utils/allocator.h"
#include "api/aoclfftz_internal.h"
#include "selector/selector.h"


aoclfftz_decomp_scheme_t *alloc_decomp_scheme(INT32 vec_rank, INT32 dim_rank);
INT32 alloc_bluestein_buffers(aoclfftz_bluestein_t *bluestein, INTP size);
aoclfftz_solution_t *alloc_solution(INT32 vec_rank, INT32 dim_rank);
aoclfftz_solution_t **alloc_sol_array(INT32 n);
VOID alloc_stride_arrays(aoclfftz_strides_t *strides, INTP radix);
INT32 alloc_and_fill_stride_arrays(aoclfftz_strides_t *strides, INTP radix,
                                   INTP in_stride, INTP out_stride);

aoclfftz_selector_t *alloc_selector(INT32 vec_rank, INT32 dim_rank,
                                    kernel_tables_t *kernel_tables);

VOID *alloc_twiddle_buffer(UINTP size, UINT32 dt_prec);
VOID alloc_ndim_buffer(aoclfftz_solution_t *solution, VOID **buffer_ptr);

VOID destroy_selector(aoclfftz_selector_t *sel);
VOID destroy_selector_without_solution(aoclfftz_selector_t *sel);
VOID destroy_strides_grp(aoclfftz_strides_grp_t *strides_grp);

VOID destroy_solution(aoclfftz_solution_t *sol);
VOID destroy_solutions(aoclfftz_solution_t **sol, INT32 n);
VOID destroy_decomp_scheme(aoclfftz_decomp_scheme_t *decomp_scheme);
VOID destroy_bluestein(aoclfftz_bluestein_t *bluestein);
UINTP calculate_max_buffer_size(aoclfftz_solution_t *sol);
#endif // MEMORY_MANAGER_H
