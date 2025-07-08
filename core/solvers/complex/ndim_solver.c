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

/** @file ndim_solver.c
 *
 *  @brief N-Dimensional solver that solves an ND problem
 *
 *  This file contains the functions that setup, execute and destroy
 *  the solver.
 *
 *  @author Prasandh Sankarankutty
 */

#include "core/solvers/complex/ndim_solver.h"
#include "selector/selector.h"

INT32 setup_ndim_solver(aoclfftz_solution_t *sol,
                        aoclfftz_solution_t *n_minus1_sol,
                        aoclfftz_solution_t *outer_dim_sol)
{
#ifdef AOCL_ENABLE_LOG
    INT32 logger_mode = sol->decomp_scheme->cntrl_params->logger_mode;
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Enter");
#endif

    COPY_SOLUTION_OBJ_WO_DIMS(n_minus1_sol, sol);
    INT32 dim_rank = sol->decomp_scheme->dim_rank;

    // setup ND - 1 solution
    n_minus1_sol->decomp_scheme->dim_rank = dim_rank - 1;

    // FIXME : memcpy instead ?
    for (INT32 i = 0; i < dim_rank - 1; i++)
    {
        n_minus1_sol->decomp_scheme->dims[i].n = sol->decomp_scheme->dims[i].n;
        n_minus1_sol->decomp_scheme->dims[i].in_stride =
                sol->decomp_scheme->dims[i].in_stride;
        n_minus1_sol->decomp_scheme->dims[i].out_stride =
                sol->decomp_scheme->dims[i].out_stride;
    }

    n_minus1_sol->decomp_scheme->vec_rank = 1;
    n_minus1_sol->decomp_scheme->vecs[0].n =
                    sol->decomp_scheme->dims[dim_rank - 1].n;
    n_minus1_sol->decomp_scheme->vecs[0].in_stride =
                    sol->decomp_scheme->dims[dim_rank - 1].in_stride;
    n_minus1_sol->decomp_scheme->vecs[0].out_stride =
                    sol->decomp_scheme->dims[dim_rank - 1].out_stride;

    COPY_SOLUTION_OBJ_WO_DIMS(outer_dim_sol, sol);

    // In the context of an out-of-place problem, outer_dim_sol has to operate
    // on the output buffer (populated by n_minus1_sol's result), perform
    // computation and store the result back in same buffer, typically like an
    // inplace problem. So, it is necessary to convert the the out-to-place
    // problem to in-place, for the in & out strides to be properly set.
    SET_INPLACE(outer_dim_sol->decomp_scheme->flags);

    outer_dim_sol->decomp_scheme->dim_rank = 1;
    outer_dim_sol->decomp_scheme->dims[0].n =
                    sol->decomp_scheme->dims[dim_rank - 1].n;

    // since in-place, both in_stride & out_stride map to out_stride
    outer_dim_sol->decomp_scheme->dims[0].in_stride =
        sol->decomp_scheme->dims[dim_rank - 1].out_stride;
    outer_dim_sol->decomp_scheme->dims[0].out_stride =
        sol->decomp_scheme->dims[dim_rank - 1].out_stride;

    outer_dim_sol->decomp_scheme->vec_rank = dim_rank - 1;

    for (INT32 i = 0; i < dim_rank - 1; i++)
    {
        outer_dim_sol->decomp_scheme->vecs[i].n =
                sol->decomp_scheme->dims[i].n;
        outer_dim_sol->decomp_scheme->vecs[i].in_stride =
                sol->decomp_scheme->dims[i].out_stride;
        outer_dim_sol->decomp_scheme->vecs[i].out_stride =
                sol->decomp_scheme->dims[i].out_stride;
    }

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit");
#endif
    return SOLVER_SUCCESS;
}

static INT32 execute_ndim_solver(aoclfftz_solution_t *sol)
{
#ifdef AOCL_ENABLE_LOG
    INT32 logger_mode = sol->decomp_scheme->cntrl_params->logger_mode;
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Enter");
#endif

    aoclfftz_solution_t *n_minus1_sol = sol->dft_bufs->nd_sol;
    aoclfftz_solution_t *outer_dim_sol = sol->next_sol[0];

    // update solution data pointers
    n_minus1_sol->decomp_scheme->in_real  = sol->decomp_scheme->in_real;
    n_minus1_sol->decomp_scheme->in_imag  = sol->decomp_scheme->in_imag;
    n_minus1_sol->decomp_scheme->out_real = sol->decomp_scheme->out_real;
    n_minus1_sol->decomp_scheme->out_imag = sol->decomp_scheme->out_imag;

    outer_dim_sol->decomp_scheme->in_real  = sol->decomp_scheme->out_real;
    outer_dim_sol->decomp_scheme->in_imag  = sol->decomp_scheme->out_imag;
    outer_dim_sol->decomp_scheme->out_real = sol->decomp_scheme->out_real;
    outer_dim_sol->decomp_scheme->out_imag = sol->decomp_scheme->out_imag;

    // execute nd sub-problem
    n_minus1_sol->solver->execute_solver(n_minus1_sol);

    // execute 1d sub-problem
    outer_dim_sol->solver->execute_solver(outer_dim_sol);

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit");
#endif
    return SOLVER_SUCCESS;
}

dft_solver_ register_execute_ndim_solver(VOID)
{
    return execute_ndim_solver;
}
