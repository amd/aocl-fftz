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
 *                     Bluestein node (taken from dft_bufs->active_threads_at_level at
 *                     setup time; minimum 1). bs_[in/out]_base are each allocated
 *                     as num_bs_buf * bs_buf_size bytes.
 */
FFTZ_INT32 alloc_bluestein_buffers(aoclfftz_bluestein_t *bluestein,
                                   FFTZ_INTP bs_buf_size);
aoclfftz_solution_t *alloc_solution(FFTZ_INT32 vec_rank, FFTZ_INT32 dim_rank);
aoclfftz_solution_t **alloc_sol_array(FFTZ_INT32 n);
FFTZ_INT32 alloc_stride_arrays(aoclfftz_strides_t *strides, FFTZ_INTP radix);
FFTZ_INT32 alloc_and_fill_stride_arrays(aoclfftz_strides_t *strides,
                                        FFTZ_INTP radix, FFTZ_INTP in_stride,
                                        FFTZ_INTP out_stride);
aoclfftz_selector_t *alloc_selector(FFTZ_INT32 vec_rank, FFTZ_INT32 dim_rank,
                                    kernel_tables_t *kernel_tables,
                                    FFTZ_UINT8 *has_nested);

FFTZ_VOID *alloc_twiddle_buffer(FFTZ_UINTP size, FFTZ_UINT32 dt_prec);
FFTZ_INT32 alloc_ndim_buffer(aoclfftz_solution_t *solution,
                             FFTZ_VOID **buffer_ptr);

// Allocate one aligned slab for the per-call scratch regions (CT, BS, SR) and
// point ctx at the carved sub-regions. On success, *scratch_slab holds the slab
// (or NULL if no scratch is needed). Returns the allocation status.
static inline aoclfftz_error_type alloc_per_call_scratch(
    aoclfftz_immutable_metadata_t *exec_meta, aoclfftz_mutable_ctx_t *ctx,
    FFTZ_VOID **scratch_slab)
{
    *scratch_slab = NULL;

    // Pack regions 64-byte aligned so AVX aligned loads/stores stay valid and
    // fast. CT & BS sizes are already padded at setup; pad SR here.
    FFTZ_UINTP ct_buffer_size     = exec_meta->ct_buffer_total_size;
    FFTZ_UINTP bs_buffer_size     = exec_meta->bs_buffer_size;
    FFTZ_UINTP sr_input_copy_size =
                                GET_PADDED_SIZE(exec_meta->sr_input_copy_size);

    // Two Bluestein regions of equal size: bs_in_base, bs_out_base
    FFTZ_UINTP total = ct_buffer_size
                       + (bs_buffer_size * 2)
                       + sr_input_copy_size;
    if (total == 0)
    {
        return AOCLFFTZ_SUCCESS;
    }

    FFTZ_VOID *slab = NULL;
    ALLOC_ALIGN_UNINIT(slab, FFTZ_VOID, total);
    if (slab == NULL)
    {
        return AOCLFFTZ_MEMORY_FAILURE;
    }

    FFTZ_UINTP offset = 0;
    if (exec_meta->ct_buffer_total_size > 0)
    {
        ctx->ct_buf_base = MOVE_ADDR(slab, offset);
        offset += ct_buffer_size;
    }
    if (exec_meta->bs_buffer_size > 0)
    {
        ctx->bs_in_base = MOVE_ADDR(slab, offset);
        offset += bs_buffer_size;
        ctx->bs_out_base = MOVE_ADDR(slab, offset);
        offset += bs_buffer_size;
    }
    if (exec_meta->sr_input_copy_size > 0)
    {
        ctx->sr_input_copy_base = MOVE_ADDR(slab, offset);
    }
    *scratch_slab = slab;
    return AOCLFFTZ_SUCCESS;
}

FFTZ_VOID destroy_selector(aoclfftz_selector_t *sel);
FFTZ_VOID destroy_selector_without_solution(aoclfftz_selector_t *sel);
FFTZ_VOID destroy_strides_grp(aoclfftz_strides_grp_t *strides_grp);

FFTZ_VOID destroy_solution(aoclfftz_solution_t *sol);
FFTZ_VOID destroy_solutions(aoclfftz_solution_t **sol, FFTZ_INT32 n);
FFTZ_VOID destroy_decomp_scheme(aoclfftz_decomp_scheme_t *decomp_scheme);
FFTZ_VOID destroy_bluestein(aoclfftz_bluestein_t *bluestein);
FFTZ_UINTP calculate_max_buffer_size(aoclfftz_solution_t *sol);
#endif // MEMORY_MANAGER_H
