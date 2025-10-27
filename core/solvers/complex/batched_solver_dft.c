/**
 * Copyright (C) 2023-2025, Advanced Micro Devices. All rights reserved.
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

INT32 setup_batched_solver(aoclfftz_solution_t *sol)
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
INT32 execute_batched_solver_internal(aoclfftz_solution_t *sol,
                              aoclfftz_solution_t *next_sol, INTP vec_rank)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");


    INT32 status = SOLVER_SUCCESS;
    INTP rnk_offset;
    INTP v_in_stride;
    INTP v_out_stride;

    UINT32 dt_bytes = SOL_DT_SIZE(sol);

    v_in_stride = sol->decomp_scheme->vecs[vec_rank - 1].in_stride *
                  DATA_STRIDE * dt_bytes;
    v_out_stride = sol->decomp_scheme->vecs[vec_rank - 1].out_stride *
                   DATA_STRIDE * dt_bytes;

    if (vec_rank == 1)
    {
        // save pointer to reset it after all executions
        VOID *in_real = next_sol->decomp_scheme->in_real;
        VOID *in_imag = next_sol->decomp_scheme->in_imag;
        VOID *out_real = next_sol->decomp_scheme->out_real;
        VOID *out_imag = next_sol->decomp_scheme->out_imag;
    #if !defined (PERFORM_INTER_STAGE_PERMUTE)
        VOID *ct_buf_real = next_sol->dft_bufs->ct_buf_real;
        VOID *ct_buf_imag = next_sol->dft_bufs->ct_buf_imag;
    #endif

        INTP ct_buf_offset = 0;
        if (!sol->dft_bufs->reset_ct_buf_offset)
        {
            ct_buf_offset = v_out_stride;
        }

        // For innermost vector rank, execute the solver
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
    #if !defined (PERFORM_INTER_STAGE_PERMUTE)
            next_sol->dft_bufs->ct_buf_real =
                MOVE_ADDR(next_sol->dft_bufs->ct_buf_real, ct_buf_offset);
            next_sol->dft_bufs->ct_buf_imag =
                MOVE_ADDR(next_sol->dft_bufs->ct_buf_imag, ct_buf_offset);
    #endif
        }

        // reset pointers to enable multiple executions
        next_sol->decomp_scheme->in_real = in_real;
        next_sol->decomp_scheme->in_imag = in_imag;
        next_sol->decomp_scheme->out_real = out_real;
        next_sol->decomp_scheme->out_imag = out_imag;
    #if !defined (PERFORM_INTER_STAGE_PERMUTE)
        next_sol->dft_bufs->ct_buf_real = ct_buf_real;
        next_sol->dft_bufs->ct_buf_imag = ct_buf_imag;
    #endif
    }
    else
    {
        // save pointer to reset it after all executions
        VOID *in_real = next_sol->decomp_scheme->in_real;
        VOID *in_imag = next_sol->decomp_scheme->in_imag;
        VOID *out_real = next_sol->decomp_scheme->out_real;
        VOID *out_imag = next_sol->decomp_scheme->out_imag;
    #if !defined (PERFORM_INTER_STAGE_PERMUTE)
        VOID *ct_buf_real = next_sol->dft_bufs->ct_buf_real;
        VOID *ct_buf_imag = next_sol->dft_bufs->ct_buf_imag;
    #endif

        INTP ct_buf_offset = 0;
        if (!sol->dft_bufs->reset_ct_buf_offset)
        {
            ct_buf_offset = v_out_stride;
        }

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
    #if !defined (PERFORM_INTER_STAGE_PERMUTE)
            VOID *ct_buf_real = next_sol->dft_bufs->ct_buf_real;
            VOID *ct_buf_imag = next_sol->dft_bufs->ct_buf_imag;
    #endif
            //recursive call to solve the inner batches
            status = execute_batched_solver_internal(sol, next_sol,
                                                     vec_rank - 1);
            if (status != SOLVER_SUCCESS)
            {
                return status;
            }

            // Adjust pointers for the next iteration
            next_sol->decomp_scheme->in_real =
                (VOID *)((CHAR *)in_real + v_in_stride);
            next_sol->decomp_scheme->in_imag =
                (VOID *)((CHAR *)in_imag + v_in_stride);
            next_sol->decomp_scheme->out_real =
                (VOID *)((CHAR *)out_real + v_out_stride);
            next_sol->decomp_scheme->out_imag =
                (VOID *)((CHAR *)out_imag + v_out_stride);
    #if !defined (PERFORM_INTER_STAGE_PERMUTE)
            next_sol->dft_bufs->ct_buf_real =
                (VOID *)((CHAR *)ct_buf_real + ct_buf_offset);
            next_sol->dft_bufs->ct_buf_imag =
                (VOID *)((CHAR *)ct_buf_imag + ct_buf_offset);
    #endif
        }

        // reset pointers to enable multiple executions
        next_sol->decomp_scheme->in_real = in_real;
        next_sol->decomp_scheme->in_imag = in_imag;
        next_sol->decomp_scheme->out_real = out_real;
        next_sol->decomp_scheme->out_imag = out_imag;
    #if !defined (PERFORM_INTER_STAGE_PERMUTE)
        next_sol->dft_bufs->ct_buf_real = ct_buf_real;
        next_sol->dft_bufs->ct_buf_imag = ct_buf_imag;
    #endif
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
static INT32 execute_batched_solver(aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");


    INT32 status = SOLVER_SUCCESS;
    aoclfftz_solution_t *next_sol = sol->next_sol[0];

    next_sol->decomp_scheme->in_real = sol->decomp_scheme->in_real;
    next_sol->decomp_scheme->in_imag = sol->decomp_scheme->in_imag;
    next_sol->decomp_scheme->out_real = sol->decomp_scheme->out_real;
    next_sol->decomp_scheme->out_imag = sol->decomp_scheme->out_imag;
#if !defined (PERFORM_INTER_STAGE_PERMUTE)
    next_sol->dft_bufs->ct_buf_real = sol->dft_bufs->ct_buf_real;
    next_sol->dft_bufs->ct_buf_imag = sol->dft_bufs->ct_buf_imag;
#endif

    next_sol->decomp_scheme->flags = sol->decomp_scheme->flags;

    status = execute_batched_solver_internal(sol, next_sol,
                                             sol->decomp_scheme->vec_rank);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return status;
}

dft_solver_ register_execute_batched_solver(VOID)
{
    return execute_batched_solver;
}
