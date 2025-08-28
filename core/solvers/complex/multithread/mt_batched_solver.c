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

/** @file mt_batched_solver.c
 *
 *  @brief Multi threaded batched solver that sets up and solves a vector
 *  problem by parallelizing across multiple threads
 *
 *  This file contains the functions that setup, execute and destroy
 *  the solver.
 *
 *  @author Murugan Vairavel
 */

#include "core/common/memory_manager.h"

INT32 setup_mt_batched_solver(aoclfftz_solution_t *sol, UINT32 num_threads_used)
{
#ifdef AOCL_ENABLE_LOG
    INT32 logger_mode = sol->decomp_scheme->cntrl_params->logger_mode;
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Enter");
#endif

    // Turn the vector problem into a single set/unit problem to find its
    // solution
    sol->decomp_scheme->vec_rank = 1;
    sol->decomp_scheme->vecs[0].n = 1;

    // Since we are using num_threads_used in current batched solver, we need to
    // update the available threads accordingly so that the remaining threads
    // can be used by the child threads in the next level
    sol->decomp_scheme->thread_info->avl_threads /= num_threads_used;

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit");
#endif
    return SOLVER_SUCCESS;
}

// Recursively solves batched FFT by handling the innermost dimension first.
INT32 execute_mt_batched_solver_internal(aoclfftz_solution_t *sol,
                                aoclfftz_solution_t **next_sol, INTP vec_rank)
{
#ifdef AOCL_ENABLE_LOG
    INT32 logger_mode = sol->decomp_scheme->cntrl_params->logger_mode;
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Enter");
#endif

    INT32 status = SOLVER_SUCCESS;
    UINT32 dt_prec, dt_bytes;
    INTP rnk_offset, v_in_stride, v_out_stride;

    dt_prec = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    dt_bytes = DT_PRECISION_BYTES(dt_prec);

    v_in_stride = sol->decomp_scheme->vecs[vec_rank - 1].in_stride *
                  DATA_STRIDE * dt_bytes;
    v_out_stride = sol->decomp_scheme->vecs[vec_rank - 1].out_stride *
                   DATA_STRIDE * dt_bytes;

    if (vec_rank == 1)
    {
        // For innermost vector rank, execute the solver
        INTP batches = sol->decomp_scheme->vecs[0].n;
        VOID *in_real = next_sol[0]->decomp_scheme->in_real;
        VOID *in_imag = next_sol[0]->decomp_scheme->in_imag;
        VOID *out_real = next_sol[0]->decomp_scheme->out_real;
        VOID *out_imag = next_sol[0]->decomp_scheme->out_imag;
        VOID *nd_sol_out_real = next_sol[0]->dft_bufs->nd_sol_out_real;
        VOID *nd_sol_out_imag = next_sol[0]->dft_bufs->nd_sol_out_imag;

        omp_set_num_threads(sol->decomp_scheme->thread_info->n_threads);

        #pragma omp parallel for
        for (INTP b = 0; b < batches; b++)
        {
            UINT32 tid = omp_get_thread_num();

            next_sol[tid]->decomp_scheme->in_real =
                                (VOID *)((CHAR *)in_real + b * v_in_stride);
            next_sol[tid]->decomp_scheme->in_imag =
                                (VOID *)((CHAR *)in_imag + b * v_in_stride);
            next_sol[tid]->decomp_scheme->out_real =
                                (VOID *)((CHAR *)out_real + b * v_out_stride);
            next_sol[tid]->decomp_scheme->out_imag =
                                (VOID *)((CHAR *)out_imag + b * v_out_stride);

        #if !defined (PERFORM_INTER_STAGE_PERMUTE)
            if (next_sol[0]->dft_bufs->nd_sol_out_real != NULL)
            {
                // If the next solution has nd_sol_out buffers, move them too
                next_sol[tid]->dft_bufs->nd_sol_out_real =
                (VOID *)((CHAR *)nd_sol_out_real + b * v_out_stride);
                next_sol[tid]->dft_bufs->nd_sol_out_imag =
                (VOID *)((CHAR *)nd_sol_out_imag + b * v_out_stride);
            }
        #endif

            status = next_sol[tid]->solver->execute_solver(next_sol[tid]);
        }

        // reset pointers to enable multiple executions
        next_sol[0]->decomp_scheme->in_real = in_real;
        next_sol[0]->decomp_scheme->in_imag = in_imag;
        next_sol[0]->decomp_scheme->out_real = out_real;
        next_sol[0]->decomp_scheme->out_imag = out_imag;
        next_sol[0]->dft_bufs->nd_sol_out_real = nd_sol_out_real;
        next_sol[0]->dft_bufs->nd_sol_out_imag = nd_sol_out_imag;
    }
    else
    {
        // save pointers to reset after all executions
        VOID *in_real = next_sol[0]->decomp_scheme->in_real;
        VOID *in_imag = next_sol[0]->decomp_scheme->in_imag;
        VOID *out_real = next_sol[0]->decomp_scheme->out_real;
        VOID *out_imag = next_sol[0]->decomp_scheme->out_imag;
        VOID *nd_sol_out_real = next_sol[0]->dft_bufs->nd_sol_out_real;
        VOID *nd_sol_out_imag = next_sol[0]->dft_bufs->nd_sol_out_imag;

        for (rnk_offset = 0;
             rnk_offset < sol->decomp_scheme->vecs[vec_rank - 1].n; rnk_offset++)
        {
            // save pointer to restore it below since
            // they will be moved while execution
            VOID *in_real = next_sol[0]->decomp_scheme->in_real;
            VOID *in_imag = next_sol[0]->decomp_scheme->in_imag;
            VOID *out_real = next_sol[0]->decomp_scheme->out_real;
            VOID *out_imag = next_sol[0]->decomp_scheme->out_imag;
        #if !defined (PERFORM_INTER_STAGE_PERMUTE)
            VOID *nd_sol_out_real = next_sol[0]->dft_bufs->nd_sol_out_real;
            VOID *nd_sol_out_imag = next_sol[0]->dft_bufs->nd_sol_out_imag;
        #endif

            //recursive call to solve the inner batches
            status = execute_mt_batched_solver_internal(sol, next_sol,
                                                        vec_rank - 1);
            if (status != SOLVER_SUCCESS)
            {
                return status;
            }

            // Adjust pointers for the next iteration
            UINT32 n_threads = sol->decomp_scheme->thread_info->n_threads;
            for (UINT32 i = 0; i < n_threads; i++)
            {
                next_sol[i]->decomp_scheme->in_real =
                    (VOID *)((CHAR *)in_real + v_in_stride);
                next_sol[i]->decomp_scheme->in_imag =
                    (VOID *)((CHAR *)in_imag + v_in_stride);
                next_sol[i]->decomp_scheme->out_real =
                    (VOID *)((CHAR *)out_real + v_out_stride);
                next_sol[i]->decomp_scheme->out_imag =
                    (VOID *)((CHAR *)out_imag + v_out_stride);
             #if !defined (PERFORM_INTER_STAGE_PERMUTE)
                if (next_sol[0]->dft_bufs->nd_sol_out_real != NULL)
                {
                    // If the next solution has nd_sol_out buffers, move them too
                    next_sol[i]->dft_bufs->nd_sol_out_real =
                    (VOID *)((CHAR *)nd_sol_out_real + v_out_stride);
                    next_sol[i]->dft_bufs->nd_sol_out_imag =
                    (VOID *)((CHAR *)nd_sol_out_imag + v_out_stride);
                }
            #endif
            }
        }

        // reset pointers
        next_sol[0]->decomp_scheme->in_real = in_real;
        next_sol[0]->decomp_scheme->in_imag = in_imag;
        next_sol[0]->decomp_scheme->out_real = out_real;
        next_sol[0]->decomp_scheme->out_imag = out_imag;
        next_sol[0]->dft_bufs->nd_sol_out_real = nd_sol_out_real;
        next_sol[0]->dft_bufs->nd_sol_out_imag = nd_sol_out_imag;
    }
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit");
#endif
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
static INT32 execute_mt_batched_solver(aoclfftz_solution_t *sol)
{
#ifdef AOCL_ENABLE_LOG
    INT32 logger_mode = sol->decomp_scheme->cntrl_params->logger_mode;
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Enter");
#endif

    INT32 status = SOLVER_SUCCESS;
    aoclfftz_solution_t **next_sol = sol->next_sol;

    for (UINT32 i = 0; i < sol->decomp_scheme->thread_info->n_threads; i++)
    {
        next_sol[i]->decomp_scheme->in_real = sol->decomp_scheme->in_real;
        next_sol[i]->decomp_scheme->in_imag = sol->decomp_scheme->in_imag;
        next_sol[i]->decomp_scheme->out_real = sol->decomp_scheme->out_real;
        next_sol[i]->decomp_scheme->out_imag = sol->decomp_scheme->out_imag;
        next_sol[i]->decomp_scheme->flags = sol->decomp_scheme->flags;

    #if !defined (PERFORM_INTER_STAGE_PERMUTE)
        if (sol->dft_bufs->nd_sol_out_real != NULL)
        {
            next_sol[i]->dft_bufs->nd_sol_out_real = sol->dft_bufs->nd_sol_out_real;
            next_sol[i]->dft_bufs->nd_sol_out_imag = sol->dft_bufs->nd_sol_out_imag;
        }
    #endif
    }
    status = execute_mt_batched_solver_internal(sol, next_sol,
                                                sol->decomp_scheme->vec_rank);

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit");
#endif
    return status;
}

dft_solver_ register_execute_mt_batched_solver(VOID)
{
    return execute_mt_batched_solver;
}
