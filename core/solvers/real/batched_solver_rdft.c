/**
 * Copyright (C) 2025, Advanced Micro Devices. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

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

INT32 setup_real_batched_solver(aoclfftz_solution_t *sol,
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
    if (FFT_DIR(sol->decomp_scheme->flags) == FORWARD_FFT_DIR)
    {
        sol->decomp_scheme->vecs[0].out_stride *= 2;
    }
    else
    {
        sol->decomp_scheme->vecs[0].in_stride *= 2;
    }

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SOLVER_SUCCESS;
}

// Recursively solves batched RFFT by handling the innermost dimension first.
INT32 execute_real_batched_solver_internal(aoclfftz_solution_t *sol,
                                           aoclfftz_solution_t *next_sol,
                                           INTP vec_rank)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");


    INT32 status = SOLVER_SUCCESS;
    INTP rnk_offset;
    INTP v_in_stride;
    INTP v_out_stride;

    UINT32 dt_bytes = SOL_DT_SIZE(sol);

    v_in_stride = sol->decomp_scheme->vecs[vec_rank - 1].in_stride * dt_bytes;
    v_out_stride = sol->decomp_scheme->vecs[vec_rank - 1].out_stride * dt_bytes;

    if (vec_rank == 1)
    {
        // For innermost vector rank,
        // re-arrange the input if needed and then execute the solver
        INTP batches = sol->decomp_scheme->vecs[0].n;

        for (INTP b = 0; b < batches; b++)
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
            VOID *in_real = next_sol->decomp_scheme->in_real;
            VOID *in_imag = next_sol->decomp_scheme->in_imag;
            VOID *out_real = next_sol->decomp_scheme->out_real;
            VOID *out_imag = next_sol->decomp_scheme->out_imag;

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

/*
 * TODO: Check and fix ND batches for real problems
 */
static INT32 execute_real_batched_solver(aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");


    INT32 status = SOLVER_SUCCESS;
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

dft_solver_ register_execute_real_batched_solver(VOID)
{
    return execute_real_batched_solver;
}
