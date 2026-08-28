// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file ndim_solver_rdft.c
 *
 *  @brief N-Dimensional solver that solves an ND real problem
 *
 *  This file contains the functions that setup, execute and destroy
 *  the solver.
 *
 *  @author Prasandh Sankarankutty
 *  @author Srirammaswamy Srinivasan
 *  @author Jeevanantham N
 */

#include "core/common/memory_manager.h"
#include "utils/utils.h"

FFTZ_INT32 setup_real_ndim_solver(aoclfftz_solution_t *sol,
                             aoclfftz_solution_t *real_dim_sol,
                             aoclfftz_solution_t *complex_dims_sol,
                             aoclfftz_realhelper_t *realhelper)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    FFTZ_INT32 dt_prec, dt_bytes;
    dt_prec = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    dt_bytes = DT_PRECISION_BYTES(dt_prec);

    copy_solution_obj_wo_dims(complex_dims_sol, sol);
    copy_solution_obj_wo_dims(real_dim_sol, sol);

    // Only C2R needs an auxiliary buffer, to hold the intermediate output of
    // the (N-1)D complex stage, for both in-place and out-of-place problems.
    if (FFT_DIR(sol->decomp_scheme->flags) == BACKWARD_FFT_DIR)
    {
        if (sol->dft_bufs->buffered == NULL)
        {
            ALLOC_ALIGN_INIT(sol->dft_bufs->buffered, aoclfftz_buffered_t,
                             sizeof(aoclfftz_buffered_t));
            if (sol->dft_bufs->buffered == NULL)
            {
                AOCLFFTZ_ERROR("allocate buffered struct failed: %s",
                               get_status_string(AOCLFFTZ_MEMORY_FAILURE));
                return AOCLFFTZ_MEMORY_FAILURE;
            }
        }
        // Logical bytes per C2R aux slab; round up so thread slabs stay
        // aligned.
        FFTZ_INTP logical_aux_buf_size =
            calculate_c2r_aux_buffer_size(sol) * DATA_STRIDE * dt_bytes;
        FFTZ_INTP padded_aux_buf_size = GET_PADDED_SIZE(logical_aux_buf_size);

        FREE_ALIGN_ALLOCATED_MEM(sol->dft_bufs->buffered->aux_buffer_1);
        ALLOC_ALIGN_INIT(sol->dft_bufs->buffered->aux_buffer_1, FFTZ_VOID,
            sol->decomp_scheme->thread_info->active_threads *
            padded_aux_buf_size);
        if (sol->dft_bufs->buffered->aux_buffer_1 == NULL)
        {
            AOCLFFTZ_ERROR("allocate aux buffer failed: %s",
                           get_status_string(AOCLFFTZ_MEMORY_FAILURE));
            return AOCLFFTZ_MEMORY_FAILURE;
        }
        sol->dft_bufs->buffered->is_aux_buffer_allocated = 1;
        sol->dft_bufs->buffered->aux_buf_size_per_thread = padded_aux_buf_size;
    }
    else
    {
        if (sol->dft_bufs->buffered != NULL)
        {
            sol->dft_bufs->buffered->aux_buf_size_per_thread = 0;
        }
    }

    FFTZ_INT32 dim_rank = sol->decomp_scheme->dim_rank;
    FFTZ_UINT8 is_forward = (
        FFT_DIR(sol->decomp_scheme->flags) == FORWARD_FFT_DIR);

    // Assign buffer pointers
    if (is_forward)
    {
        complex_dims_sol->decomp_scheme->in_real =
            real_dim_sol->decomp_scheme->out_real;
        complex_dims_sol->decomp_scheme->in_imag =
            real_dim_sol->decomp_scheme->out_imag;
        complex_dims_sol->decomp_scheme->out_real =
            sol->decomp_scheme->out_real;
        complex_dims_sol->decomp_scheme->out_imag =
            sol->decomp_scheme->out_imag;
        // The R2C result is already in the output buffer, so reuse it for the
        // remaining dimensions.
        SET_INPLACE(complex_dims_sol->decomp_scheme->flags);
    }
    else
    {
        complex_dims_sol->decomp_scheme->in_real = sol->decomp_scheme->in_real;
        complex_dims_sol->decomp_scheme->in_imag = sol->decomp_scheme->in_imag;
        complex_dims_sol->decomp_scheme->out_real =
            sol->dft_bufs->buffered->aux_buffer_1;
        complex_dims_sol->decomp_scheme->out_imag =
            MOVE_ADDR(sol->dft_bufs->buffered->aux_buffer_1, dt_bytes);
        // Backward uses out-of-place C2C because its packed intermediate has
        // different strides from the input; aux holds that result for C2R.
        SET_OUTOFPLACE(complex_dims_sol->decomp_scheme->flags);
    }


    // setup (N-1)D solution
    complex_dims_sol->decomp_scheme->dim_rank = dim_rank - 1;
    complex_dims_sol->decomp_scheme->vec_rank = 1;

    // setup 1D solution
    real_dim_sol->decomp_scheme->dim_rank = 1;
    real_dim_sol->decomp_scheme->dims[0].n = sol->decomp_scheme->dims[0].n;
    real_dim_sol->decomp_scheme->vec_rank = dim_rank - 1;

    // Setup (N-1)D complex_dims_sol dimensions
    // R2C (forward): complex_dims_sol reads real_dim_sol's own output and
    //   writes straight into the caller's output buffer (no aux buffer
    //   involved), so both in_stride and out_stride must mirror the
    //   caller's real out_stride.
    // C2R (backward): complex_dims_sol reads the caller's real in_stride,
    //   but its own output only ever lands in the private aux_buffer_1
    //   (see above) that nothing outside this pair addresses. So its
    //   out_stride is free to be densely packed instead of reproducing the
    //   caller's (possibly padded/strided) layout; real_dim_sol's matching
    //   in_stride/dims[0].in_stride below are set to that same packed
    //   spacing so the two stages agree on where each element in
    //   aux_buffer_1 lives. This keeps aux_buffer_1's size (see
    //   calculate_c2r_aux_buffer_size) independent of the caller's strides.
    // FIXME : memcpy instead ?
    FFTZ_INTP packed_stride = 1;
    for (FFTZ_INT32 i = 0; i < dim_rank - 1; i++)
    {
        complex_dims_sol->decomp_scheme->dims[i].n =
            sol->decomp_scheme->dims[i + 1].n;
        if (is_forward)
        {
            complex_dims_sol->decomp_scheme->dims[i].in_stride =
                sol->decomp_scheme->dims[i + 1].out_stride;
            complex_dims_sol->decomp_scheme->dims[i].out_stride =
                sol->decomp_scheme->dims[i + 1].out_stride;
        }
        else
        {
            complex_dims_sol->decomp_scheme->dims[i].in_stride =
                sol->decomp_scheme->dims[i + 1].in_stride;
            complex_dims_sol->decomp_scheme->dims[i].out_stride =
                packed_stride;
            real_dim_sol->decomp_scheme->vecs[i].in_stride = packed_stride;
            packed_stride *= complex_dims_sol->decomp_scheme->dims[i].n;
        }
    }

    // Setup batch vector for complex_dims_sol
    // Only process n/2 + 1 batches since:
    // - R2C (forward): output is half-complex with n/2 + 1 valid points
    // - C2R (backward): input is half-complex with n/2 + 1 valid points
    complex_dims_sol->decomp_scheme->vecs[0].n =
        sol->decomp_scheme->dims[0].n / 2 + 1;
    if (is_forward)
    {
        FFTZ_INTP vec_stride = sol->decomp_scheme->dims[0].out_stride;
        complex_dims_sol->decomp_scheme->vecs[0].in_stride = vec_stride;
        complex_dims_sol->decomp_scheme->vecs[0].out_stride = vec_stride;
    }
    else
    {
        complex_dims_sol->decomp_scheme->vecs[0].in_stride =
            sol->decomp_scheme->dims[0].in_stride;
        // packed_stride is now the product of all (N-1)D dims: the spacing
        // between successive half-spectrum bins in the packed aux buffer.
        complex_dims_sol->decomp_scheme->vecs[0].out_stride = packed_stride;
        real_dim_sol->decomp_scheme->dims[0].in_stride = packed_stride;
    }

    // Setup 1D real_dim_sol
    if (is_forward)
    {
        real_dim_sol->decomp_scheme->dims[0].in_stride =
            sol->decomp_scheme->dims[0].in_stride;
    }
    real_dim_sol->decomp_scheme->dims[0].out_stride =
        sol->decomp_scheme->dims[0].out_stride;

    // Setup batch vectors for real_dim_sol
    for (FFTZ_INT32 i = 0; i < dim_rank - 1; i++)
    {
        real_dim_sol->decomp_scheme->vecs[i].n =
            sol->decomp_scheme->dims[i + 1].n;
        // C2R vec input strides were set above while packing the complex
        // stage's output into aux_buffer_1.
        if (is_forward)
        {
            real_dim_sol->decomp_scheme->vecs[i].in_stride =
                sol->decomp_scheme->dims[i + 1].in_stride;
        }
        real_dim_sol->decomp_scheme->vecs[i].out_stride =
            sol->decomp_scheme->dims[i + 1].out_stride;
    }

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return SOLVER_SUCCESS;
}

static FFTZ_INT32 execute_real_ndim_solver(aoclfftz_solution_t *sol,
                                           aoclfftz_mutable_ctx_t *ctx)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    aoclfftz_solution_t *complex_dims_sol = sol->dft_bufs->complex_sol;
    aoclfftz_solution_t *real_dim_sol = sol->next_sol;

    FFTZ_INT32 dt_prec, dt_bytes;
    dt_prec = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    dt_bytes = DT_PRECISION_BYTES(dt_prec);
    FFTZ_UINT8 is_forward = (
        FFT_DIR(sol->decomp_scheme->flags) == FORWARD_FFT_DIR);

    // A private ctx copy per sub-solver. Each branch below fills in the in/out
    // pointers its sub-solver should run on, leaving the caller's ctx untouched.
    aoclfftz_mutable_ctx_t c2c_child_ctx = *ctx;
    aoclfftz_mutable_ctx_t real_child_ctx = *ctx;

    if (is_forward)
    {
        // R2C (forward) Execution Flow:
        // 1. real_dim_sol (1D R2C):
        //    in: input buffer
        //    out: output buffer
        // 2. complex_dims_sol ((N-1)D C2C):
        //    in: output buffer
        //    out: output buffer

        // The 1D R2C runs on the plan's own in/out, i.e. an unmodified copy.
        real_dim_sol->solver->execute_solver(real_dim_sol, &real_child_ctx);

        c2c_child_ctx.in_real  = ctx->out_real;
        c2c_child_ctx.in_imag  = ctx->out_imag;
        c2c_child_ctx.out_real = ctx->out_real;
        c2c_child_ctx.out_imag = ctx->out_imag;
        c2c_child_ctx.flags    = complex_dims_sol->decomp_scheme->flags;
        // execute (n-1)d sub-problem
        complex_dims_sol->solver->execute_solver(complex_dims_sol,
                                                 &c2c_child_ctx);
    }
    else
    {
        // C2R (backward) Execution Flow:
        // 1. complex_dims_sol ((N-1)D C2C):
        //    in: input buffer, out: aux (per-thread slice of the ndim pool)
        // 2. real_dim_sol (1D C2R):
        //    in: aux, out: output buffer

        // `slot_idx` hands each concurrent caller a disjoint slot of the ndim aux
        // pool, so no two threads share the same C2R intermediate.
        FFTZ_INTP aux_size = sol->dft_bufs->buffered->aux_buf_size_per_thread;
        FFTZ_INTP aux_off = (FFTZ_INTP)ctx->slot_idx * aux_size;
        FFTZ_VOID *aux_real = MOVE_ADDR(ctx->aux_pool_base_ndim, aux_off);
        FFTZ_VOID *aux_imag = MOVE_ADDR(aux_real, dt_bytes);

        c2c_child_ctx.in_real  = ctx->in_real;
        c2c_child_ctx.in_imag  = ctx->in_imag;
        c2c_child_ctx.out_real = aux_real;
        c2c_child_ctx.out_imag = aux_imag;
        c2c_child_ctx.flags    = complex_dims_sol->decomp_scheme->flags;

        real_child_ctx.in_real  = aux_real;
        real_child_ctx.in_imag  = aux_imag;
        real_child_ctx.out_real = ctx->out_real;
        real_child_ctx.out_imag = ctx->out_imag;
        // execute (n-1)d sub-problem
        complex_dims_sol->solver->execute_solver(complex_dims_sol,
                                                 &c2c_child_ctx);
        // execute 1d sub-problem
        real_dim_sol->solver->execute_solver(real_dim_sol, &real_child_ctx);
    }

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return SOLVER_SUCCESS;
}

dft_solver_ register_execute_real_ndim_solver(FFTZ_VOID)
{
    return execute_real_ndim_solver;
}
