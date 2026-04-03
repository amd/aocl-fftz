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
/**
 * Allocates the shared chirp buffers (B, B_out) and the per-thread in/out
 * pool for a Bluestein node.
 *
 * @param bluestein    Bluestein metadata struct (must be non-NULL).
 * @param bs_buf_size  Bytes consumed by a single thread's in (or out) buffer,
 *                     i.e. m * DATA_STRIDE * dt_bytes for extended length m,
 *                     padded to MIN_ALIGNMENT (64 B) so each per-thread slot
 *                     base stays 64-byte aligned for aligned SIMD load/store.
 * @param num_bs_buf   Number of concurrent threads that may invoke this
 *                     Bluestein node (snapshot of dft_bufs->num_ct_buf at
 *                     setup time; minimum 1). in and out are each allocated
 *                     as num_bs_buf * bs_buf_size contiguous bytes.
 *
 * On success bs_buf_allocated is set to 1 on this struct only; deep copies
 * that re-point to the same B/B_out/in/out keep bs_buf_allocated = 0 to
 * prevent double-free in destroy_bluestein.
 */
INT32 alloc_bluestein_buffers(aoclfftz_bluestein_t *bluestein,
                              INTP bs_buf_size, INT32 num_bs_buf);
aoclfftz_solution_t *alloc_solution(INT32 vec_rank, INT32 dim_rank);
aoclfftz_solution_t **alloc_sol_array(INT32 n);
INT32 alloc_stride_arrays(aoclfftz_strides_t *strides, INTP radix);
INT32 alloc_and_fill_stride_arrays(aoclfftz_strides_t *strides, INTP radix,
                                   INTP in_stride, INTP out_stride);

aoclfftz_selector_t *alloc_selector(INT32 vec_rank, INT32 dim_rank,
                                    kernel_tables_t *kernel_tables);

VOID *alloc_twiddle_buffer(UINTP size, UINT32 dt_prec);
INT32 alloc_ndim_buffer(aoclfftz_solution_t *solution, VOID **buffer_ptr);

VOID destroy_selector(aoclfftz_selector_t *sel);
VOID destroy_selector_without_solution(aoclfftz_selector_t *sel);
VOID destroy_strides_grp(aoclfftz_strides_grp_t *strides_grp);

VOID destroy_solution(aoclfftz_solution_t *sol);
VOID destroy_solutions(aoclfftz_solution_t **sol, INT32 n);
VOID destroy_decomp_scheme(aoclfftz_decomp_scheme_t *decomp_scheme);
VOID destroy_bluestein(aoclfftz_bluestein_t *bluestein);
UINTP calculate_max_buffer_size(aoclfftz_solution_t *sol);
#endif // MEMORY_MANAGER_H
