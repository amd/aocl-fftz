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

/** @file batched_solver.c
 *
 *  @brief Batched Solver that sets up and solves a vector problem
 *
 *  This file contains the functions that setup, execute and destroy
 *  the solver.
 *
 *  @author Srirammaswamy Srinivasan
 *  @author Jeya R
 */

#include "core/common/memory_manager.h"
#include "utils/utils.h"

INT32 setup_real_batched_solver(aoclfftz_solution_t *sol,
                                aoclfftz_realhelper_t *realhelper)
{
#ifdef AOCL_ENABLE_LOG
    INT32 logger_mode = sol->decomp_scheme->cntrl_params->logger_mode;
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Enter");
#endif

    // Turn the vector problem into a single set/unit problem to find its
    // solution
    sol->decomp_scheme->vec_rank = 1;
    sol->decomp_scheme->vecs[0].n = 1;

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit");
#endif
    return SOLVER_SUCCESS;
}

// Recursively solves batched RFFT by handling the innermost dimension first.
INT32 execute_real_batched_solver_internal(aoclfftz_solution_t *sol,
                                           aoclfftz_solution_t *next_sol,
                                           INTP vec_rank)
{
#ifdef AOCL_ENABLE_LOG
    INT32 logger_mode = sol->decomp_scheme->cntrl_params->logger_mode;
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Enter");
#endif

    INT32 status = SOLVER_SUCCESS;
    UINT32 dt_prec, dt_bytes;
    INTP rnk_offset;
    INTP v_in_stride;
    INTP v_out_stride;

    dt_prec = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    dt_bytes = DT_PRECISION_BYTES(dt_prec);

    v_in_stride = sol->decomp_scheme->vecs[vec_rank - 1].in_stride *
                  DATA_STRIDE * dt_bytes;
    v_out_stride = sol->decomp_scheme->vecs[vec_rank - 1].out_stride *
                   DATA_STRIDE * dt_bytes;

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
        // FIXIT: ND batched and ND sized in-place real forward problems
        //        might fail.
        //
        // TODO: Perform input data re-ordering of in-place forward problems for
        //       ND batched problems and ND sized problems

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
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit");
#endif
    return status;
}

/*
 * TODO: Check and fix ND batches for real problems
 */
static INT32 execute_real_batched_solver(aoclfftz_solution_t *sol)
{
#ifdef AOCL_ENABLE_LOG
    INT32 logger_mode = sol->decomp_scheme->cntrl_params->logger_mode;
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Enter");
#endif

    INT32 status = SOLVER_SUCCESS;
    aoclfftz_solution_t *next_sol = sol->next_sol;

    next_sol->decomp_scheme->in_real = sol->decomp_scheme->in_real;
    next_sol->decomp_scheme->in_imag = sol->decomp_scheme->in_imag;
    next_sol->decomp_scheme->out_real = sol->decomp_scheme->out_real;
    next_sol->decomp_scheme->out_imag = sol->decomp_scheme->out_imag;

    status = execute_real_batched_solver_internal(sol, next_sol,
                                                  sol->decomp_scheme->vec_rank);

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit");
#endif
    return status;
}

dft_solver_ register_execute_real_batched_solver(VOID)
{
    return execute_real_batched_solver;
}
