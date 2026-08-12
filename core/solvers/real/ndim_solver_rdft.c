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

    // Logical bytes per REAL_NDIM aux slab; round up so thread slabs stay
    // aligned.
    FFTZ_INTP logical_aux_buf_size =
        calculate_max_buffer_size(sol) * DATA_STRIDE * dt_bytes;
    FFTZ_INTP padded_aux_buf_size = GET_PADDED_SIZE(logical_aux_buf_size);

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
        sol->dft_bufs->buffered->aux_buf_size_per_thread = padded_aux_buf_size;
        sol->dft_bufs->buffered->is_aux_buffer_allocated = 1;
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
    }
    else
    {
        complex_dims_sol->decomp_scheme->in_real = sol->decomp_scheme->in_real;
        complex_dims_sol->decomp_scheme->in_imag = sol->decomp_scheme->in_imag;
        complex_dims_sol->decomp_scheme->out_real =
            sol->dft_bufs->buffered->aux_buffer_1;
        complex_dims_sol->decomp_scheme->out_imag =
            MOVE_ADDR(sol->dft_bufs->buffered->aux_buffer_1, dt_bytes);
    }


    // setup (N-1)D solution
    complex_dims_sol->decomp_scheme->dim_rank = dim_rank - 1;
    complex_dims_sol->decomp_scheme->vec_rank = 1;

    // setup 1D solution
    real_dim_sol->decomp_scheme->dim_rank = 1;
    real_dim_sol->decomp_scheme->dims[0].n = sol->decomp_scheme->dims[0].n;
    real_dim_sol->decomp_scheme->vec_rank = dim_rank - 1;

    // Setup (N-1)D complex_dims_sol dimensions
    // R2C (forward): half-complex data is in output,
    //   use out_stride for complex_dims_sol
    // C2R (backward): half-complex data is in input,
    //   use in_stride for complex_dims_sol
    // FIXME : memcpy instead ?
    for (FFTZ_INT32 i = 0; i < dim_rank - 1; i++)
    {
        complex_dims_sol->decomp_scheme->dims[i].n =
            sol->decomp_scheme->dims[i + 1].n;
        FFTZ_INTP dim_stride = is_forward
                            ? sol->decomp_scheme->dims[i + 1].out_stride
                            : sol->decomp_scheme->dims[i + 1].in_stride;
        complex_dims_sol->decomp_scheme->dims[i].in_stride = dim_stride;
        complex_dims_sol->decomp_scheme->dims[i].out_stride = dim_stride;
    }

    // Setup batch vector for complex_dims_sol
    // Only process n/2 + 1 batches since:
    // - R2C (forward): output is half-complex with n/2 + 1 valid points
    // - C2R (backward): input is half-complex with n/2 + 1 valid points
    complex_dims_sol->decomp_scheme->vecs[0].n =
        sol->decomp_scheme->dims[0].n / 2 + 1;
    FFTZ_INTP vec_stride = is_forward ? sol->decomp_scheme->dims[0].out_stride
                                 : sol->decomp_scheme->dims[0].in_stride;
    complex_dims_sol->decomp_scheme->vecs[0].in_stride = vec_stride;
    complex_dims_sol->decomp_scheme->vecs[0].out_stride = vec_stride;

    // Setup 1D real_dim_sol
    real_dim_sol->decomp_scheme->dims[0].in_stride =
        sol->decomp_scheme->dims[0].in_stride;
    real_dim_sol->decomp_scheme->dims[0].out_stride =
        sol->decomp_scheme->dims[0].out_stride;

    // Setup batch vectors for real_dim_sol
    for (FFTZ_INT32 i = 0; i < dim_rank - 1; i++)
    {
        real_dim_sol->decomp_scheme->vecs[i].n =
            sol->decomp_scheme->dims[i + 1].n;
        real_dim_sol->decomp_scheme->vecs[i].in_stride =
            sol->decomp_scheme->dims[i + 1].in_stride;
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

    aoclfftz_solution_t *complex_dims_sol = sol->dft_bufs->nd_sol;
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
