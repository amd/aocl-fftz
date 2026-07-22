// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file mt_batched_solver_rdft.c
 *
 *  @brief Multithreaded batched solver that sets up and solves a vector
 *  problem by parallelizing across multiple threads
 *
 *  This file contains the functions that setup and execute the solver.
 *
 *  @author Partiksha
 */

#include "core/common/memory_manager.h"

FFTZ_INT32 setup_real_mt_batched_solver(aoclfftz_solution_t *sol,
                                        aoclfftz_solution_t *next_sol,
                                        aoclfftz_realhelper_t *realhelper,
                                        FFTZ_UINT8 *has_nested)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    thread_info_t *thread_info = sol->decomp_scheme->thread_info;
    if (thread_info->active_threads != 1)
    {
        *has_nested = 1;
    }

    // Turn the vector problem into a single set/unit problem to find its
    // solution if it is not a direct problem
    next_sol->decomp_scheme->vec_rank = 1;
    next_sol->decomp_scheme->vecs[0].n = 1;

    // Since we are using n_threads in current batched solver, we need to update
    // the available threads accordingly so that the remaining threads can be
    // used by the child threads in the next level
    FFTZ_INT32 num_threads_used = sol->decomp_scheme->thread_info->n_threads;
    next_sol->decomp_scheme->thread_info->avl_threads /= num_threads_used;
    next_sol->decomp_scheme->thread_info->active_threads *= num_threads_used;
    // Strides are prepared based on real points, so adjust them (scale by 2)
    // for complex points (i.e. R2C output and C2R input)
    // Scale ALL vector strides, not just vecs[0], to handle vec_rank > 1 cases
    for (FFTZ_INTP i = 0; i < sol->decomp_scheme->vec_rank; i++)
    {
        if (FFT_DIR(sol->decomp_scheme->flags) == FORWARD_FFT_DIR)
        {
            sol->decomp_scheme->vecs[i].out_stride *= 2;
        }
        else
        {
            sol->decomp_scheme->vecs[i].in_stride *= 2;
        }
    }

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SOLVER_SUCCESS;
}

// Recursively solves batched RFFT by handling the innermost dimension first.
FFTZ_INT32 execute_real_mt_batched_solver_internal(aoclfftz_solution_t *sol,
                                              aoclfftz_solution_t **next_sol,
                                              FFTZ_INTP vec_rank,
                                              aoclfftz_mutable_ctx_t *ctx)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");


    FFTZ_INT32 status = SOLVER_SUCCESS;
    FFTZ_INTP rnk_offset;
    FFTZ_INTP v_in_stride;
    FFTZ_INTP v_out_stride;

    FFTZ_UINT32 dt_bytes = SOL_DT_SIZE(sol);

    v_in_stride = sol->decomp_scheme->vecs[vec_rank - 1].in_stride * dt_bytes;
    v_out_stride = sol->decomp_scheme->vecs[vec_rank - 1].out_stride * dt_bytes;

    if (vec_rank == 1)
    {
        // For innermost vector rank,
        // re-arrange the input if needed and then execute the solver
        FFTZ_INTP batches = sol->decomp_scheme->vecs[0].n;
        FFTZ_VOID *in_real = next_sol[0]->decomp_scheme->in_real;
        FFTZ_VOID *in_imag = next_sol[0]->decomp_scheme->in_imag;
        FFTZ_VOID *out_real = next_sol[0]->decomp_scheme->out_real;
        FFTZ_VOID *out_imag = next_sol[0]->decomp_scheme->out_imag;

        FFTZ_INT32 n_threads = sol->decomp_scheme->thread_info->n_threads;

        #pragma omp parallel for num_threads(n_threads)
        for (FFTZ_INTP b = 0; b < batches; b++)
        {
            FFTZ_INT32 tid = omp_get_thread_num();
            FFTZ_INT32 local_status = SOLVER_SUCCESS;

            // Update batch-specific input/output pointers
            next_sol[tid]->decomp_scheme->in_real =
                                MOVE_ADDR(in_real, b * v_in_stride);
            next_sol[tid]->decomp_scheme->in_imag =
                                MOVE_ADDR(in_imag, b * v_in_stride);
            next_sol[tid]->decomp_scheme->out_real =
                                MOVE_ADDR(out_real, b * v_out_stride);
            next_sol[tid]->decomp_scheme->out_imag =
                                MOVE_ADDR(out_imag, b * v_out_stride);

            aoclfftz_mutable_ctx_t thr_ctx = *ctx;

            // Relevant only when the child is an ndim complex sub-solver (nd_sol),
            // give each thread its own ct pool slice.
            thr_ctx.ct_offset = ctx->ct_offset +
               (FFTZ_INTP)tid * (FFTZ_INTP)next_sol[tid]->dft_bufs->ct_buf_size;

            // Index used to slice the bs_[in/out]_base for Bluestein/MT_Bluestein.
            thr_ctx.bs_slot_idx = ctx->bs_slot_idx * n_threads + tid;

            local_status = next_sol[tid]->solver->execute_solver(
                                                    next_sol[tid], &thr_ctx);
            if (local_status != SOLVER_SUCCESS)
            {
                #pragma omp atomic write
                status = local_status;
            }
        }
    }
    else
    {
        for (rnk_offset = 0;
             rnk_offset < sol->decomp_scheme->vecs[vec_rank - 1].n;
             rnk_offset++)
        {
            // save pointer to restore it below since
            // they will be moved while execution
            FFTZ_VOID *in_real = next_sol[0]->decomp_scheme->in_real;
            FFTZ_VOID *in_imag = next_sol[0]->decomp_scheme->in_imag;
            FFTZ_VOID *out_real = next_sol[0]->decomp_scheme->out_real;
            FFTZ_VOID *out_imag = next_sol[0]->decomp_scheme->out_imag;

            // recursive call to solve the inner batches
            status = execute_real_mt_batched_solver_internal(sol, next_sol,
                                                            vec_rank - 1, ctx);
            if (status != SOLVER_SUCCESS)
            {
                return status;
            }

            // Adjust pointers for the next iteration
            for (FFTZ_INT32 i = 0;
                 i < sol->decomp_scheme->thread_info->n_threads; i++)
            {
                next_sol[i]->decomp_scheme->in_real =
                    (FFTZ_VOID *)((FFTZ_CHAR *)in_real + v_in_stride);
                next_sol[i]->decomp_scheme->in_imag =
                    (FFTZ_VOID *)((FFTZ_CHAR *)in_imag + v_in_stride);
                next_sol[i]->decomp_scheme->out_real =
                    (FFTZ_VOID *)((FFTZ_CHAR *)out_real + v_out_stride);
                next_sol[i]->decomp_scheme->out_imag =
                    (FFTZ_VOID *)((FFTZ_CHAR *)out_imag + v_out_stride);
            }
        }
    }
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return status;
}

/*
 * TODO: Support ND batches for real problems
 */
static FFTZ_INT32 execute_real_mt_batched_solver(aoclfftz_solution_t *sol,
                                                 aoclfftz_mutable_ctx_t *ctx)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");


    FFTZ_INT32 status = SOLVER_SUCCESS;
    aoclfftz_solution_t **next_sol = sol->next_sol;
    for (FFTZ_INT32 i = 0; i < sol->decomp_scheme->thread_info->n_threads; i++)
    {
        next_sol[i]->decomp_scheme->in_real = sol->decomp_scheme->in_real;
        next_sol[i]->decomp_scheme->in_imag = sol->decomp_scheme->in_imag;
        next_sol[i]->decomp_scheme->out_real = sol->decomp_scheme->out_real;
        next_sol[i]->decomp_scheme->out_imag = sol->decomp_scheme->out_imag;
        next_sol[i]->decomp_scheme->flags = sol->decomp_scheme->flags;
    }

    status = execute_real_mt_batched_solver_internal(sol, next_sol,
                                                  sol->decomp_scheme->vec_rank,
                                                  ctx);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return status;
}

dft_solver_ register_execute_real_mt_batched_solver(FFTZ_VOID)
{
    return execute_real_mt_batched_solver;
}
