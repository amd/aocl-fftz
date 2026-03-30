// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file ndim_solver_dft.c
 *
 *  @brief N-Dimensional solver that solves an ND problem
 *
 *  This file contains the functions that setup, execute and destroy
 *  the solver.
 *
 *  @author Prasandh Sankarankutty
 */

#include "selector/selector.h"
#include "core/common/memory_manager.h"

INT32 setup_ndim_solver(aoclfftz_solution_t *sol,
                        aoclfftz_solution_t *n_minus1_sol,
                        aoclfftz_solution_t *outer_dim_sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    alloc_ndim_buffer(sol, &sol->dft_bufs->ct_buffer);
    if (sol->dft_bufs->ct_buffer == NULL)
    {
        AOCLFFTZ_ERROR("Failed to allocate buffer for ND solver");
        return SOLVER_FAILURE;
    }

    COPY_SOLUTION_OBJ_WO_DIMS(n_minus1_sol, sol);
    INT32 dim_rank = sol->decomp_scheme->dim_rank;

    // NOTE: since "innermost" or "not-innermost" apply only to the leaf nodes
    // of the ND problem, we don't need to set the flags for the n-1 dimension.

    // setup ND - 1 solution
    n_minus1_sol->decomp_scheme->dim_rank = dim_rank - 1;

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

    SET_NOT_INNERMOST_DIM(outer_dim_sol->decomp_scheme->flags);

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

    outer_dim_sol->decomp_scheme->in_real  =
        n_minus1_sol->decomp_scheme->out_real;
    outer_dim_sol->decomp_scheme->in_imag  =
        n_minus1_sol->decomp_scheme->out_imag;
    outer_dim_sol->decomp_scheme->out_real = sol->decomp_scheme->out_real;
    outer_dim_sol->decomp_scheme->out_imag = sol->decomp_scheme->out_imag;

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return SOLVER_SUCCESS;
}

static INT32 execute_ndim_solver(aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    aoclfftz_solution_t *n_minus1_sol = sol->dft_bufs->nd_sol;
    aoclfftz_solution_t *outer_dim_sol = sol->next_sol[0];

    // update solution data pointers
    n_minus1_sol->decomp_scheme->in_real  = sol->decomp_scheme->in_real;
    n_minus1_sol->decomp_scheme->in_imag  = sol->decomp_scheme->in_imag;
    n_minus1_sol->decomp_scheme->out_real = sol->decomp_scheme->out_real;
    n_minus1_sol->decomp_scheme->out_imag = sol->decomp_scheme->out_imag;

    // propagate the pointers to next solution for it to set to the solution
    // after it ie., n_minus1_sol->next_sol->next_sol
    // only required for n_minus1_sol sub-problem since outer_dim_sol
    // will not have an NDim sub-problem
    n_minus1_sol->dft_bufs->ct_buf_real = sol->dft_bufs->ct_buf_real;
    n_minus1_sol->dft_bufs->ct_buf_imag = sol->dft_bufs->ct_buf_imag;

    // outer_dim_sol pointer updates
    outer_dim_sol->decomp_scheme->in_real  =
        n_minus1_sol->decomp_scheme->out_real;
    outer_dim_sol->decomp_scheme->in_imag  =
        n_minus1_sol->decomp_scheme->out_imag;
    outer_dim_sol->decomp_scheme->out_real = sol->decomp_scheme->out_real;
    outer_dim_sol->decomp_scheme->out_imag = sol->decomp_scheme->out_imag;
    outer_dim_sol->dft_bufs->ct_buf_real = sol->dft_bufs->ct_buf_real;
    outer_dim_sol->dft_bufs->ct_buf_imag = sol->dft_bufs->ct_buf_imag;

    // execute nd sub-problem
    n_minus1_sol->solver->execute_solver(n_minus1_sol);

    // execute 1d sub-problem
    outer_dim_sol->solver->execute_solver(outer_dim_sol);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return SOLVER_SUCCESS;
}

dft_solver_ register_execute_ndim_solver(VOID)
{
    return execute_ndim_solver;
}
