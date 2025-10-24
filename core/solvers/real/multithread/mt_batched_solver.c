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
 *  @brief Multithreaded batched solver that sets up and solves a vector
 *  problem by parallelizing across multiple threads
 *
 *  This file contains the functions that setup and execute the solver.
 *
 *  @author Partiksha
 */

#include "core/common/memory_manager.h"

INT32 setup_real_mt_batched_solver(aoclfftz_solution_t *sol,
                                   aoclfftz_solution_t *next_sol,
                                   aoclfftz_realhelper_t *realhelper)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");


    // Turn the vector problem into a single set/unit problem to find its
    // solution if it is not a direct problem
    next_sol->decomp_scheme->vec_rank = 1;
    next_sol->decomp_scheme->vecs[0].n = 1;

    // Since we are using n_threads in current batched solver, we need to update
    // the available threads accordingly so that the remaining threads can be
    // used by the child threads in the next level
    INT32 num_threads_used = sol->decomp_scheme->thread_info->n_threads;
    next_sol->decomp_scheme->thread_info->avl_threads /= num_threads_used;
    realhelper->num_aux_buf = num_threads_used;

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

INT32 update_pointers_real_buffered_solution(aoclfftz_solution_t *sol, INTP tid)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    INTP n = sol->decomp_scheme->dims->n;
    INT32 dt_bytes = SOL_DT_SIZE(sol);

    // problem_out points to the start addresses of out buffers
    VOID *problem_out = sol->decomp_scheme->out_real;

    // solution from of CT problem after buffered sol
    // ... -> buffered -> direct -> CT -> direct -> ... -> CT -> direct
    //
    // Here, the buffered will have in & out of the current batch
    //
    // Buffered solver will change the input/output buffers of direct & CT
    // solution in the following way:
    //
    // buffered    [in -> out]
    // |--> direct   [in -> aux1]
    // |----> CT       [aux1 -> aux2]
    // |----> direct   [aux1 -> aux2]
    // |------> CT       [aux2 -> aux1]
    // |------> direct   [aux2 -> aux1]
    // |--------> CT       [aux1 -> out]
    // |--------> direct   [aux1 -> out]
    // this example is for a 3 level CT problem

    // the input sol points to the first solution
    // move the sol pointer to buffered_solver and modify the input & output
    // addresses of buffered struct to point the updated problem input & output
    while (sol != NULL && (sol->solver->solver_type != SOLVER_REAL_BUFFERED))
    {
        sol = sol->next_sol[0];
    }

    if (sol == NULL)
    {
        return SOLVER_FAILURE;
    }

    // storing aux buffers in a temp variable
    aoclfftz_solution_t *buffered_sol = sol;
    VOID *aux_in = MOVE_ADDR(sol->dft_bufs->buffered->aux_buffer_1,
                             n * tid * dt_bytes);
    VOID *aux_out = MOVE_ADDR(sol->dft_bufs->buffered->aux_buffer_2,
                             n * tid * dt_bytes);

    // update `ct_buf_real_in` pointers used for C2R out-of-place problems
    sol->dft_bufs->ct_buf_real_in = aux_in;

    // move to the first direct solution of CT problem
    sol = sol->next_sol[0];

    // update first direct solution's in/out
    sol->decomp_scheme->out_real = aux_out;
    sol->decomp_scheme->out_imag = MOVE_ADDR(aux_out, dt_bytes);
    // swap aux buffers so that the current output should be the next input
    SWAP_BUFFERS(aux_in, aux_out);
    sol = sol->next_sol[0];

    // update all the CT + direct solutions' in/out
    // (except first direct and last CT + direct)
    while (sol && sol->next_sol[0] && sol->next_sol[0]->next_sol)
    {
        sol->decomp_scheme->in_real = aux_in;
        sol->decomp_scheme->in_imag = MOVE_ADDR(aux_in, dt_bytes);
        sol->decomp_scheme->out_real = aux_out;
        sol->decomp_scheme->out_imag = MOVE_ADDR(aux_out, dt_bytes);
        // swap aux buffers after every direct solution
        if (sol->solver->solver_type == SOLVER_REAL_DIRECT ||
            sol->solver->solver_type == SOLVER_REAL_DIRECT_TWIDDLE ||
            sol->solver->solver_type == SOLVER_REAL_MT_DIRECT ||
            sol->solver->solver_type == SOLVER_REAL_MT_DIRECT_TWIDDLE)
        {
            SWAP_BUFFERS(aux_in, aux_out);
        }
        sol = sol->next_sol[0];
    }
    if (sol == NULL)
    {
        return SOLVER_FAILURE;
    }
    // update last CT solution's in/out
    sol->decomp_scheme->in_real = aux_in;
    sol->decomp_scheme->in_imag = MOVE_ADDR(aux_in, dt_bytes);
    sol->decomp_scheme->out_real = problem_out;
    sol->decomp_scheme->out_imag = MOVE_ADDR(problem_out, dt_bytes);
    // update last direct solution's in/out
    sol = sol->next_sol[0];
    sol->decomp_scheme->in_real = aux_in;
    sol->decomp_scheme->in_imag = MOVE_ADDR(aux_in, dt_bytes);

    // store the address of last direct sol output as buffered->out_ptr
    // this will be modified by buffered executor
    buffered_sol->dft_bufs->buffered->out_ptr = &sol->decomp_scheme->out_real;

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SOLVER_SUCCESS;
}

// Recursively solves batched RFFT by handling the innermost dimension first.
INT32 execute_real_mt_batched_solver_internal(aoclfftz_solution_t *sol,
                                              aoclfftz_solution_t **next_sol,
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
        VOID *in_real = next_sol[0]->decomp_scheme->in_real;
        VOID *in_imag = next_sol[0]->decomp_scheme->in_imag;
        VOID *out_real = next_sol[0]->decomp_scheme->out_real;
        VOID *out_imag = next_sol[0]->decomp_scheme->out_imag;

        omp_set_num_threads(sol->decomp_scheme->thread_info->n_threads);
        #pragma omp parallel for
        for (INTP b = 0; b < batches; b++)
        {
            INT32 tid = omp_get_thread_num();
            if (sol->decomp_scheme->thread_info->n_threads <= 1)
            {
                tid = 0;
            }
            // process real buffered solver to change the input/output
            // pointer correctly for each thread
            if (next_sol[tid]->solver->solver_type == SOLVER_REAL_BUFFERED)
            {
                update_pointers_real_buffered_solution(next_sol[tid], tid);
            }

            next_sol[tid]->decomp_scheme->in_real =
                                (VOID *)((CHAR *)in_real + b * v_in_stride);
            next_sol[tid]->decomp_scheme->in_imag =
                                (VOID *)((CHAR *)in_imag + b * v_in_stride);
            next_sol[tid]->decomp_scheme->out_real =
                                (VOID *)((CHAR *)out_real + b * v_out_stride);
            next_sol[tid]->decomp_scheme->out_imag =
                                (VOID *)((CHAR *)out_imag + b * v_out_stride);
            status = next_sol[tid]->solver->execute_solver(next_sol[tid]);
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
            VOID *in_real = next_sol[0]->decomp_scheme->in_real;
            VOID *in_imag = next_sol[0]->decomp_scheme->in_imag;
            VOID *out_real = next_sol[0]->decomp_scheme->out_real;
            VOID *out_imag = next_sol[0]->decomp_scheme->out_imag;

            // recursive call to solve the inner batches
            status = execute_real_mt_batched_solver_internal(sol, next_sol,
                                                          vec_rank - 1);
            if (status != SOLVER_SUCCESS)
            {
                return status;
            }

            // Adjust pointers for the next iteration
            for (INT32 i = 0; i < sol->decomp_scheme->thread_info->n_threads;
                 i++)
            {
                next_sol[i]->decomp_scheme->in_real =
                    (VOID *)((CHAR *)in_real + v_in_stride);
                next_sol[i]->decomp_scheme->in_imag =
                    (VOID *)((CHAR *)in_imag + v_in_stride);
                next_sol[i]->decomp_scheme->out_real =
                    (VOID *)((CHAR *)out_real + v_out_stride);
                next_sol[i]->decomp_scheme->out_imag =
                    (VOID *)((CHAR *)out_imag + v_out_stride);
            }
        }
    }
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return status;
}

/*
 * TODO: Check and fix ND batches for real problems
 */
static INT32 execute_real_mt_batched_solver(aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");


    INT32 status = SOLVER_SUCCESS;
    aoclfftz_solution_t **next_sol = sol->next_sol;
    for (INT32 i = 0; i < sol->decomp_scheme->thread_info->n_threads; i++)
    {
        next_sol[i]->decomp_scheme->in_real = sol->decomp_scheme->in_real;
        next_sol[i]->decomp_scheme->in_imag = sol->decomp_scheme->in_imag;
        next_sol[i]->decomp_scheme->out_real = sol->decomp_scheme->out_real;
        next_sol[i]->decomp_scheme->out_imag = sol->decomp_scheme->out_imag;
        next_sol[i]->decomp_scheme->flags = sol->decomp_scheme->flags;
    }

    status = execute_real_mt_batched_solver_internal(sol, next_sol,
                                                  sol->decomp_scheme->vec_rank);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return status;
}

dft_solver_ register_execute_real_mt_batched_solver(VOID)
{
    return execute_real_mt_batched_solver;
}
