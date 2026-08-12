// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file batched_solver_dft.c
 *
 *  @brief Batched Solver that sets up and solves a vector problem
 *
 *  This file contains the functions that setup, execute and destroy
 *  the solver.
 *
 *  @author S. Biplab Raut
 *  @author Jeya R
 */

#include "core/solvers/solver.h"

FFTZ_INT32 setup_batched_solver(aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");


    // Turn the vector problem into a single set/unit problem to find its
    // solution
    sol->decomp_scheme->vec_rank = 1;
    sol->decomp_scheme->vecs[0].n = 1;

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return SOLVER_SUCCESS;
}

// Recursively solves batched FFT by handling the innermost dimension first.
static FFTZ_INT32 execute_batched_solver_internal(aoclfftz_solution_t *sol,
                                                  aoclfftz_solution_t *next_sol,
                                                  FFTZ_INTP vec_rank,
                                                  aoclfftz_mutable_ctx_t *ctx)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    FFTZ_INT32 status = SOLVER_SUCCESS;
    FFTZ_INTP rnk_offset;
    FFTZ_INTP v_in_stride;
    FFTZ_INTP v_out_stride;
    aoclfftz_mutable_ctx_t batch_ctx = *ctx;
    FFTZ_UINT32 dt_bytes = CTX_DT_SIZE(ctx);

    v_in_stride = sol->decomp_scheme->vecs[vec_rank - 1].in_stride *
                  DATA_STRIDE * dt_bytes;
    v_out_stride = sol->decomp_scheme->vecs[vec_rank - 1].out_stride *
                   DATA_STRIDE * dt_bytes;

    if (vec_rank == 1)
    {
        // For innermost vector rank, execute the solver
        FFTZ_INTP batches = sol->decomp_scheme->vecs[0].n;
        for (FFTZ_INTP b = 0; b < batches; b++)
        {
            status = next_sol->solver->execute_solver(next_sol, &batch_ctx);
            if (status != SOLVER_SUCCESS)
            {
                return status;
            }

            batch_ctx.in_real  = MOVE_ADDR(batch_ctx.in_real,  v_in_stride);
            batch_ctx.in_imag  = MOVE_ADDR(batch_ctx.in_imag,  v_in_stride);
            batch_ctx.out_real = MOVE_ADDR(batch_ctx.out_real, v_out_stride);
            batch_ctx.out_imag = MOVE_ADDR(batch_ctx.out_imag, v_out_stride);

            // Keep ct_buf_base aliased to out_real per batch so the inner CTL1D
            // takes its in-place path: (m) in -> out, then (r) out -> out.
            if (ctx->ct_buf_base == ctx->out_real)
            {
                batch_ctx.ct_buf_base = batch_ctx.out_real;
            }
        }
    }
    else
    {
        for (rnk_offset = 0;
             rnk_offset < sol->decomp_scheme->vecs[vec_rank - 1].n;
             rnk_offset++)
        {
            aoclfftz_mutable_ctx_t inner_ctx = batch_ctx;
            //recursive call to solve the inner batches
            status = execute_batched_solver_internal(sol, next_sol,
                                                     vec_rank - 1, &inner_ctx);
            if (status != SOLVER_SUCCESS)
            {
                return status;
            }

            // Adjust pointers for the next iteration
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
 * Considerations and assumptions for execute_batched_solver():
 * For a multi-dimensional vector array of the DFT transforms,
 * sol->decomp_scheme->vecs[rnk].in_stride gives the offset at which
 * input buffer starts for the current rank/position in the vector array,
 * sol->decomp_scheme->vecs[rnk].out_stride gives the offset at which
 * output buffer starts for the current rank/position in the vector array.
 */
static FFTZ_INT32 execute_batched_solver(aoclfftz_solution_t *sol,
                                         aoclfftz_mutable_ctx_t *ctx)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");


    FFTZ_INT32 status = SOLVER_SUCCESS;
    aoclfftz_solution_t *next_sol = sol->next_sol;

    status = execute_batched_solver_internal(sol, next_sol,
                                             sol->decomp_scheme->vec_rank, ctx);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return status;
}

dft_solver_ register_execute_batched_solver(FFTZ_VOID)
{
    return execute_batched_solver;
}
