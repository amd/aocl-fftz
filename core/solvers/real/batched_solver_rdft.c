// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file batched_solver_rdft.c
 *
 *  @brief Batched Solver that sets up and solves a vector problem
 *
 *  This file contains the functions that setup, execute and destroy
 *  the solver.
 *
 *  @author Srirammaswamy Srinivasan
 *  @author Jeya R
 */

#include "core/solvers/solver.h"

FFTZ_INT32 setup_real_batched_solver(aoclfftz_solution_t *sol,
                                aoclfftz_solution_t *next_sol,
                                aoclfftz_realhelper_t *realhelper)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");


    // Turn the vector problem into a single set/unit problem to find its
    // solution if it is not a direct problem
    next_sol->decomp_scheme->vec_rank = 1;
    next_sol->decomp_scheme->vecs[0].n = 1;

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
FFTZ_INT32 execute_real_batched_solver_internal(aoclfftz_solution_t *sol,
                                           aoclfftz_solution_t *next_sol,
                                           FFTZ_INTP vec_rank)
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

        for (FFTZ_INTP b = 0; b < batches; b++)
        {
            status = next_sol->solver->execute_solver(next_sol);
            if (status != SOLVER_SUCCESS)
            {
                return status;
            }

            next_sol->decomp_scheme->in_real =
                MOVE_ADDR(next_sol->decomp_scheme->in_real, v_in_stride);
            next_sol->decomp_scheme->in_imag =
                MOVE_ADDR(next_sol->decomp_scheme->in_imag, v_in_stride);
            next_sol->decomp_scheme->out_real =
                MOVE_ADDR(next_sol->decomp_scheme->out_real, v_out_stride);
            next_sol->decomp_scheme->out_imag =
                MOVE_ADDR(next_sol->decomp_scheme->out_imag, v_out_stride);
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
            FFTZ_VOID *in_real = next_sol->decomp_scheme->in_real;
            FFTZ_VOID *in_imag = next_sol->decomp_scheme->in_imag;
            FFTZ_VOID *out_real = next_sol->decomp_scheme->out_real;
            FFTZ_VOID *out_imag = next_sol->decomp_scheme->out_imag;

            // recursive call to solve the inner batches
            status = execute_real_batched_solver_internal(sol, next_sol,
                                                          vec_rank - 1);
            if (status != SOLVER_SUCCESS)
            {
                return status;
            }

            // Adjust pointers for the next iteration
            next_sol->decomp_scheme->in_real = MOVE_ADDR(in_real, v_in_stride);
            next_sol->decomp_scheme->in_imag = MOVE_ADDR(in_imag, v_in_stride);
            next_sol->decomp_scheme->out_real =
                MOVE_ADDR(out_real, v_out_stride);
            next_sol->decomp_scheme->out_imag =
                MOVE_ADDR(out_imag, v_out_stride);
        }
    }
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return status;
}

static FFTZ_INT32 execute_real_batched_solver(aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");


    FFTZ_INT32 status = SOLVER_SUCCESS;
    aoclfftz_solution_t *next_sol = sol->next_sol[0];

    next_sol->decomp_scheme->in_real = sol->decomp_scheme->in_real;
    next_sol->decomp_scheme->in_imag = sol->decomp_scheme->in_imag;
    next_sol->decomp_scheme->out_real = sol->decomp_scheme->out_real;
    next_sol->decomp_scheme->out_imag = sol->decomp_scheme->out_imag;

    status = execute_real_batched_solver_internal(sol, next_sol,
                                                  sol->decomp_scheme->vec_rank);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return status;
}

dft_solver_ register_execute_real_batched_solver(FFTZ_VOID)
{
    return execute_real_batched_solver;
}
