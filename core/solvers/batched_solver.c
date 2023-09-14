/**
 * Copyright (C) 2023, Advanced Micro Devices. All rights reserved.
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
 *  @author S. Biplab Raut
 */

#include "core/solvers/ct_solver.h"
#include "core/common/twiddle.h"
#include "core/common/memory_manager.h"
#include "utils/utils.h"

INT32 setup_batched_solver(aoclfftz_solution_t *sol)
{

    //Turn the vector problem into a single set/unit problem to find its solution
    sol->decomp_scheme->vec_rank = 1;
    sol->decomp_scheme->vecs[0].n = 1;
    sol->decomp_scheme->vecs[0].in_stride = 1;
    sol->decomp_scheme->vecs[0].out_stride = 1;

    return SOLVER_SUCCESS;
}

/*
* Considerations and assumptions for execute_batched_solver():
* For a multi-dimentional vector array (up to rank 3) of the DFT tranforms,
* sol->decomp_scheme->vecs[rnk].in_stride gives the offset at which 
* input buffer starts for the current rank/position in the vector array,
* sol->decomp_scheme->vecs[rnk].out_stride gives the offset at which 
* output buffer starts for the current rank/position in the vector array.
*/
INT32 execute_batched_solver(aoclfftz_solution_t *sol)
{
    aoclfftz_solution_t *next_sol = sol->next_sol;
    INT32 batch_size;
    INT32 rnk_offset;
    INT32 rnk;
    INT32 status = SOLVER_SUCCESS;

    switch (sol->decomp_scheme->vec_rank)
    {
    case 1:
        next_sol->decomp_scheme->in_real = sol->decomp_scheme->in_real;
        next_sol->decomp_scheme->in_imag = sol->decomp_scheme->in_imag;
        next_sol->decomp_scheme->out_real = sol->decomp_scheme->out_real;
        next_sol->decomp_scheme->out_imag = sol->decomp_scheme->out_imag;
        next_sol->strides->v_in_stride =
            sol->decomp_scheme->vecs[0].in_stride;
        next_sol->strides->v_out_stride =
            sol->decomp_scheme->vecs[0].out_stride;
        for (batch_size = 0; batch_size < sol->decomp_scheme->vecs[0].n;
            batch_size++)
        {
            next_sol->decomp_scheme->in_real +=
                sol->decomp_scheme->vecs[0].in_stride;
            next_sol->decomp_scheme->in_imag +=
                sol->decomp_scheme->vecs[0].in_stride;
            next_sol->decomp_scheme->out_real +=
                sol->decomp_scheme->vecs[0].out_stride;
            next_sol->decomp_scheme->out_imag +=
                sol->decomp_scheme->vecs[0].out_stride;

            status = next_sol->solver->execute_solver(next_sol);
            if (status != SOLVER_SUCCESS)
                return status;
        }
        break;
    case 2:
        for (rnk_offset = 0; rnk_offset < sol->decomp_scheme->vecs[1].n;
             rnk_offset++)
        {
            next_sol->decomp_scheme->in_real = sol->decomp_scheme->in_real +
                (rnk_offset * sol->decomp_scheme->vecs[1].in_stride);
            next_sol->decomp_scheme->in_imag = sol->decomp_scheme->in_imag +
                (rnk_offset * sol->decomp_scheme->vecs[1].in_stride);
            next_sol->decomp_scheme->out_real = sol->decomp_scheme->out_real +
                (rnk_offset * sol->decomp_scheme->vecs[1].out_stride);
            next_sol->decomp_scheme->out_imag = sol->decomp_scheme->out_imag +
                (rnk_offset * sol->decomp_scheme->vecs[1].out_stride);

            for (batch_size = 0; batch_size < sol->decomp_scheme->vecs[0].n;
                batch_size++)
            {
                next_sol->decomp_scheme->in_real +=
                    sol->decomp_scheme->vecs[0].in_stride;
                next_sol->decomp_scheme->in_imag +=
                    sol->decomp_scheme->vecs[0].in_stride;
                next_sol->decomp_scheme->out_real +=
                    sol->decomp_scheme->vecs[0].out_stride;
                next_sol->decomp_scheme->out_imag +=
                    sol->decomp_scheme->vecs[0].out_stride;

                status = next_sol->solver->execute_solver(next_sol);
                if (status != SOLVER_SUCCESS)
                    return status;
            }
        }
        break;
    case 3:
        for (rnk = 0; rnk < sol->decomp_scheme->vecs[2].n; rnk++)
        {
            next_sol->decomp_scheme->in_real = sol->decomp_scheme->in_real +
                (rnk * sol->decomp_scheme->vecs[2].in_stride);
            next_sol->decomp_scheme->in_imag = sol->decomp_scheme->in_imag +
                (rnk * sol->decomp_scheme->vecs[2].in_stride);
            next_sol->decomp_scheme->out_real = sol->decomp_scheme->out_real +
                (rnk * sol->decomp_scheme->vecs[2].out_stride);
            next_sol->decomp_scheme->out_imag = sol->decomp_scheme->out_imag +
                (rnk * sol->decomp_scheme->vecs[2].out_stride);

            for (rnk_offset = 0; rnk_offset < sol->decomp_scheme->vecs[1].n;
                rnk_offset++)
            {
                next_sol->decomp_scheme->in_real +=
                    (rnk_offset * sol->decomp_scheme->vecs[1].in_stride);
                next_sol->decomp_scheme->in_imag +=
                    (rnk_offset * sol->decomp_scheme->vecs[1].in_stride);
                next_sol->decomp_scheme->out_real +=
                    (rnk_offset * sol->decomp_scheme->vecs[1].out_stride);
                next_sol->decomp_scheme->out_imag +=
                    (rnk_offset * sol->decomp_scheme->vecs[1].out_stride);

                for (batch_size = 0; batch_size < sol->decomp_scheme->vecs[0].n;
                    batch_size++)
                {
                    next_sol->decomp_scheme->in_real +=
                        sol->decomp_scheme->vecs[0].in_stride;
                    next_sol->decomp_scheme->in_imag +=
                        sol->decomp_scheme->vecs[0].in_stride;
                    next_sol->decomp_scheme->out_real +=
                        sol->decomp_scheme->vecs[0].out_stride;
                    next_sol->decomp_scheme->out_imag +=
                        sol->decomp_scheme->vecs[0].out_stride;

                    status = next_sol->solver->execute_solver(next_sol);
                    if (status != SOLVER_SUCCESS)
                        return status;
                }
            }
        }
        break;
    default:
        return SOLVER_FAILURE;
    }

    return status;
}