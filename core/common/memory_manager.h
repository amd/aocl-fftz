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
 * Allocates shared, 64-byte-aligned chirp buffers B and B_out for one
 * Bluestein node. Also stores @p bs_buf_size as the per-thread slot size
 * in the bs_in/bs_out scratch pool.
 *
 * @param bluestein   Bluestein metadata struct (must be non-NULL).
 * @param bs_buf_size Padded byte size of one m-length complex buffer (B,
 *                    B_out, and each per-thread scratch slot).
 */
FFTZ_INT32 alloc_bluestein_buffers(aoclfftz_bluestein_t *bluestein,
                                   FFTZ_INTP bs_buf_size);
aoclfftz_solution_t *alloc_solution(FFTZ_INT32 vec_rank, FFTZ_INT32 dim_rank);
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

// Allocate one aligned slab for the per-call scratch regions (CT, BS, SR, the
// two REAL_BUFFERED aux ping-pong pools, the REAL_NDIM aux pool and the
// transpose bitmap) and point ctx at the carved sub-regions. On success,
// *scratch_slab holds the slab (or NULL if no scratch is needed).
// Returns the allocation status.
static inline aoclfftz_error_type alloc_per_call_scratch(
    aoclfftz_immutable_metadata_t *exec_meta, aoclfftz_mutable_ctx_t *ctx,
    FFTZ_VOID **scratch_slab)
{
    *scratch_slab = NULL;

    // Pack regions 64-byte aligned so AVX aligned loads/stores stay valid and
    // fast. CT, BS, pow2 and aux pools are padded at setup; pad SR and the
    // transpose bitmap here.
    FFTZ_UINTP ct_buffer_size     = exec_meta->ct_buffer_total_size;
    FFTZ_UINTP bs_buffer_size     = exec_meta->bs_buffer_size;
    FFTZ_UINTP aux_buffered_size  = exec_meta->aux_buffered_pool_size;
    FFTZ_UINTP aux_ndim_size      = exec_meta->aux_ndim_pool_size;
    FFTZ_UINTP c2c_strides_size   = exec_meta->c2c_strides_pool_size;
    FFTZ_UINTP sr_input_copy_size =
                                GET_PADDED_SIZE(exec_meta->sr_input_copy_size);
    FFTZ_UINTP pow2_buf_size      = exec_meta->pow2_buf_size;
    FFTZ_UINTP transpose_aux_size =
                                GET_PADDED_SIZE(exec_meta->transpose_aux_size);

    // Two Bluestein regions (bs_in_base, bs_out_base), two REAL_BUFFERED aux
    // ping-pong pools (aux_pool_base_1, aux_pool_base_2, each of
    // aux_buffered_size), one REAL_NDIM aux pool (aux_pool_base_ndim), one
    // real-direct C2C stride pool (c2c_strides_base), the SR input copy and one
    // transpose bitmap.
    FFTZ_UINTP aux_slab_total = 0;
    aux_slab_total = aux_buffered_size;
    // REAL_BUFFERED pre-seeds aux_pool_base_2 in base_ctx; L1 CT leaves it
    // NULL.
    if (ctx->aux_pool_base_2 != NULL)
    {
        aux_slab_total += aux_buffered_size;
    }
    FFTZ_UINTP total = ct_buffer_size
                       + (bs_buffer_size * 2)
                       + aux_slab_total
                       + aux_ndim_size
                       + c2c_strides_size
                       + sr_input_copy_size
                       + pow2_buf_size
                       + transpose_aux_size;
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
    if (exec_meta->aux_buffered_pool_size > 0)
    {
        // Real batched CT L1 direct solver owns one inter-stage aux pool. It'd
        // use aux_pool_base_1 and Real BUFFERED uses both aux_pool_base_1 and
        // aux_pool_base_2.
        ctx->aux_pool_base_1 = MOVE_ADDR(slab, offset);
        offset += aux_buffered_size;
        // REAL_BUFFERED ping-pong: base_ctx has aux_pool_base_2 preset.
        if (ctx->aux_pool_base_2 != NULL)
        {
            ctx->aux_pool_base_2 = MOVE_ADDR(slab, offset);
            offset += aux_buffered_size;
        }
    }
    if (exec_meta->aux_ndim_pool_size > 0)
    {
        ctx->aux_pool_base_ndim = MOVE_ADDR(slab, offset);
        offset += aux_ndim_size;
    }
    if (exec_meta->c2c_strides_pool_size > 0)
    {
        ctx->c2c_strides_base = MOVE_ADDR(slab, offset);
        offset += c2c_strides_size;
    }
    if (exec_meta->sr_input_copy_size > 0)
    {
        ctx->sr_input_copy_base = MOVE_ADDR(slab, offset);
        offset += sr_input_copy_size;
    }
    if (exec_meta->pow2_buf_size > 0)
    {
        ctx->pow2_buf_base = MOVE_ADDR(slab, offset);
        offset += pow2_buf_size;
    }
    if (exec_meta->transpose_aux_size > 0)
    {
        ctx->transpose_aux_base = MOVE_ADDR(slab, offset);
        offset += transpose_aux_size;
    }
    *scratch_slab = slab;
    return AOCLFFTZ_SUCCESS;
}

FFTZ_VOID destroy_selector(aoclfftz_selector_t *sel);
FFTZ_VOID destroy_selector_without_solution(aoclfftz_selector_t *sel);
FFTZ_VOID destroy_strides_grp(aoclfftz_strides_grp_t *strides_grp);

FFTZ_VOID release_owned_real_buffered_aux(aoclfftz_solution_t *sol);

FFTZ_VOID destroy_solution(aoclfftz_solution_t *sol);
FFTZ_VOID destroy_decomp_scheme(aoclfftz_decomp_scheme_t *decomp_scheme);
FFTZ_VOID destroy_bluestein(aoclfftz_bluestein_t *bluestein);
FFTZ_UINTP calculate_max_buffer_size(aoclfftz_solution_t *sol);
FFTZ_VOID destroy_pow2_iterative(aoclfftz_pow2_iterative_t *pow2_iterative);
#endif // MEMORY_MANAGER_H
