// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file mt_batched_solver_dft.c
 *
 *  @brief Multi threaded batched solver that sets up and solves a vector
 *  problem by parallelizing across multiple threads
 *
 *  This file contains the functions that setup, execute and destroy
 *  the solver.
 *
 *  @author Murugan Vairavel
 */

#include "core/solvers/solver.h"
#include "utils/thread_control.h"

FFTZ_INT32 setup_mt_batched_solver(aoclfftz_solution_t *sol,
                                   FFTZ_INT32 num_threads_used,
                                   FFTZ_UINT8 *has_nested)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    thread_info_t *thread_info = sol->decomp_scheme->thread_info;
    if (thread_info->active_threads != 1)
    {
        *has_nested = 1;
    }

    // Turn the vector problem into a single set/unit problem to find its
    // solution
    sol->decomp_scheme->vec_rank = 1;
    sol->decomp_scheme->vecs[0].n = 1;

    // Since we are using num_threads_used in current batched solver, we need to
    // update the available threads accordingly so that the remaining threads
    // can be used by the child threads in the next level
    sol->decomp_scheme->thread_info->avl_threads /= num_threads_used;
    sol->decomp_scheme->thread_info->active_threads *= num_threads_used;

    // The quotient above is only what the batch loop could not absorb. Trim it
    // to what the transform under the batch loop can actually feed.
    sol->decomp_scheme->thread_info->avl_threads = cap_nested_thread_budget(
        sol->decomp_scheme, sol->decomp_scheme->thread_info->avl_threads);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return SOLVER_SUCCESS;
}

/*  Recursively solves batched FFT by handling the innermost dimension first.
 *
 *  MT_BATCHED composes two variables per-thread :
 *      ▪ ct_offset (consumed by buffered/ctl1d solver)
 *          - (descendant)ct_offset = (ancestor)ct_offset + tid * next_ct_size
 *      ▪ bs_slot_idx (consumed by Bluestein/MT_Bluestein)
 *          - (descendant)bs_slot_idx = (ancestor)bs_slot_idx * n_threads + tid
 *
 *  How ct_offset grows (example: "3v180x200 -o 3 -r i -n 6")
 *
 *  next_ct_size = next_sol->dft_bufs->ct_buf_size (the child's ct_buf_size).
 *  Each thread advances its own slice by tid * next_ct_size on top of the
 *  offset it inherited from its ancestor, so slices never overlap in the
 *  shared ct pool. Consumers reach their slice via ct_buf_base + ct_offset.
 *
 *                  mt_batched (root solver)  ct_offset = 0
 *                  │  (n_threads = 3, next_ct_size = 180*200)
 *                  │  (descendant)ct_offset = 0 + tid * 180*200,  tid ∈ {0,1,2}
 *        ┌─────────┴───────────────┬───────────────────────────┐
 *        ▼                         ▼                           ▼
 *   ct_offset = 0            ct_offset = 180*200       ct_offset = 2*180*200
 *        │                         │                            │
 *   NDIM 180x200              NDIM 180x200                 NDIM 180x200
 *        ├──────► n_minus1         ├──────► n_minus1            ├──────► n_minus1
 *        └─► outer_dim   │         └─► outer_dim   │            └─► outer_dim   │
 *             │          │              │          │                 │          │
 *             ▼          │              ▼          │                 ▼          │
 *            ct          │             ct          │                ct          │
 *             ├──────► m │              ├──────► m │                 ├──────► m │  (m = buffered_solver)
 *             └─► r    │ │              └─► r    │ │                 └─► r    │ │  (r = mt_direct_batched)
 *               ┌──────┘ │                  ┌────┘ │                     ┌────┘ │
 *               ▼        │                  ▼      │                     ▼      │
 *      buffered_solver   │       buffered_solver   │          buffered_solver   │
 * ct_buf_base =          |   ct_buf_base =         |  ct_buf_base =             |
 * ct_buf_base + 0        |   ct_buf_base + 180*200 |  ct_buf_base + 2*180*200   |
 *                        ▼                         ▼                            ▼
 *                  mt_batched                mt_batched                   mt_batched
 *                        │                         │                            │
 *       (ancestor)ct_offset = 0    (ancestor)ct_offset = 180*200    (ancestor)ct_offset = 2*180*200
 *          next_ct_size = 200             next_ct_size = 200            next_ct_size = 200
 *                        ↓                         ↓                            ↓
 *                  ┌────────────┐              ┌────────────┐                 ┌────────────┐
 *                  0            0           180*200      180*200          2*180*200     2*180*200
 *                 +0*200       +1*200       +0*200       +1*200           +0*200        +1*200
 *    ct_offset    =0           =200         =36000       =36200           =72000        =72200
 *                  |            |             |             |                |             |
 *                  ▼            ▼             ▼             ▼                ▼             ▼
 *                CTL1D        CTL1D         CTL1D         CTL1D            CTL1D         CTL1D
 * ct_buf_real=  ct_buf_base  ct_buf_base   ct_buf_base   ct_buf_base      ct_buf_base   ct_buf_base
 *                +0           +200          +36000        +36200           +72000        +72200
 *
 */
static FFTZ_INT32 execute_mt_batched_solver_internal(aoclfftz_solution_t *sol,
                                                aoclfftz_solution_t *next_sol,
                                                FFTZ_INTP vec_rank,
                                                aoclfftz_mutable_ctx_t *ctx)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");


    FFTZ_INT32 status = SOLVER_SUCCESS;
    FFTZ_INTP rnk_offset, v_in_stride, v_out_stride;

    FFTZ_UINT32 dt_bytes = CTX_DT_SIZE(ctx);

    v_in_stride = sol->decomp_scheme->vecs[vec_rank - 1].in_stride *
                  DATA_STRIDE * dt_bytes;
    v_out_stride = sol->decomp_scheme->vecs[vec_rank - 1].out_stride *
                   DATA_STRIDE * dt_bytes;

    if (vec_rank == 1)
    {
        // For innermost vector rank, execute the solver
        FFTZ_INTP batches = sol->decomp_scheme->vecs[0].n;
        FFTZ_INT32 n_threads = sol->decomp_scheme->thread_info->n_threads;

        #pragma omp parallel for num_threads(n_threads)
        for (FFTZ_INTP b = 0; b < batches; b++)
        {
            FFTZ_INT32 tid = omp_get_thread_num();
            FFTZ_INT32 local_status = SOLVER_SUCCESS;

            aoclfftz_mutable_ctx_t thr_ctx = *ctx;
            thr_ctx.in_real  = MOVE_ADDR(ctx->in_real,  b * v_in_stride);
            thr_ctx.in_imag  = MOVE_ADDR(ctx->in_imag,  b * v_in_stride);
            thr_ctx.out_real = MOVE_ADDR(ctx->out_real, b * v_out_stride);
            thr_ctx.out_imag = MOVE_ADDR(ctx->out_imag, b * v_out_stride);

            // Keep ct_buf_base aliased to out per batch so the inner CTL1D
            // takes its in-place path: (m) in -> out, then (r) out -> out.
            if (ctx->ct_buf_base == ctx->out_real)
            {
                thr_ctx.ct_buf_base = thr_ctx.out_real;
            }

            // To let threads share one ct pool safely, ct_offset points each thread to
            // its own slice: parent offset + tid * child ct_buf_size. Consumers reach
            // their own slice by advancing their ct_buf_base by ct_offset.
            thr_ctx.ct_offset = ctx->ct_offset +
                (FFTZ_INTP)tid * (FFTZ_INTP)next_sol->dft_bufs->ct_buf_size;

            // Index used to slice the bs_[in/out]_base for Bluestein/MT_Bluestein.
            thr_ctx.bs_slot_idx = ctx->bs_slot_idx * n_threads + tid;

            local_status = next_sol->solver->execute_solver(next_sol, &thr_ctx);
            if (local_status != SOLVER_SUCCESS)
            {
                #pragma omp atomic write
                status = local_status;
            }
        }
    }
    else
    {
        aoclfftz_mutable_ctx_t batch_ctx = *ctx;

        for (rnk_offset = 0;
             rnk_offset < sol->decomp_scheme->vecs[vec_rank - 1].n;
             rnk_offset++)
        {
            status = execute_mt_batched_solver_internal(sol, next_sol,
                                                        vec_rank - 1,
                                                        &batch_ctx);
            if (status != SOLVER_SUCCESS)
            {
                return status;
            }

            batch_ctx.in_real  = MOVE_ADDR(batch_ctx.in_real,  v_in_stride);
            batch_ctx.in_imag  = MOVE_ADDR(batch_ctx.in_imag,  v_in_stride);
            batch_ctx.out_real = MOVE_ADDR(batch_ctx.out_real, v_out_stride);
            batch_ctx.out_imag = MOVE_ADDR(batch_ctx.out_imag, v_out_stride);
        }
    }
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return status;
}

/*
 * Considerations and assumptions for execute_mt_batched_solver():
 * For a multi-dimensional vector array of the DFT transforms,
 * sol->decomp_scheme->vecs[rnk].in_stride gives the offset at which
 * input buffer starts for the current rank/position in the vector array,
 * sol->decomp_scheme->vecs[rnk].out_stride gives the offset at which
 * output buffer starts for the current rank/position in the vector array.
 */
static FFTZ_INT32 execute_mt_batched_solver(aoclfftz_solution_t *sol,
                                            aoclfftz_mutable_ctx_t *ctx)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");


    FFTZ_INT32 status = SOLVER_SUCCESS;
    aoclfftz_solution_t *next_sol = sol->next_sol[0];

    status = execute_mt_batched_solver_internal(sol, next_sol,
                                                sol->decomp_scheme->vec_rank,
                                                ctx);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return status;
}

dft_solver_ register_execute_mt_batched_solver(FFTZ_VOID)
{
    return execute_mt_batched_solver;
}
