// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file ct_solver_dft.c
 *
 *  @brief Cooley Tukey Solver that decomposes and solves an input problem
 *
 *  This file contains the functions that setup, execute and destroy
 *  the solver.
 *
 *  @author S. Biplab Raut
 */

#include "core/common/twiddle.h"
#include "core/common/memory_manager.h"


INT32 setup_ct_solver(aoclfftz_solution_t *sol, aoclfftz_solution_t *sol_r,
                      aoclfftz_solution_t *sol_m, UINT32 radix_r,
                      UINT32 radix_m)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    // Setup radix-m sub-problem
    // out-of-order -> in-order for out-of-place problems
    // out-of-order -> out-of-order for inplace problems
    COPY_SOLUTION_OBJ(sol_m, sol);
    // Set buffered flag for radix-m sub-problem if the original problem is
    // in-place.
    if (sol->decomp_scheme->in_real == sol->decomp_scheme->out_real)
    {
        SET_BUFFERED(sol_m->decomp_scheme->flags);
    }
    sol_m->decomp_scheme->decomp_level = sol->decomp_scheme->decomp_level + 1;
    sol_m->decomp_scheme->dims[0].n = radix_m;
    sol_m->decomp_scheme->dims[0].in_stride =
        radix_r * sol->decomp_scheme->dims[0].in_stride;
    sol_m->decomp_scheme->dims[0].out_stride =
        sol->decomp_scheme->dims[0].out_stride;

    sol_m->decomp_scheme->vecs[0].n = radix_r;
    sol_m->decomp_scheme->vecs[0].in_stride =
        sol->decomp_scheme->dims[0].in_stride;
    sol_m->decomp_scheme->vecs[0].out_stride =
        radix_m * sol->decomp_scheme->dims[0].out_stride;

    // Setup radix-r sub-problem
    // out-of-order -> out-of-order for inplace & out-of-place problems
    COPY_SOLUTION_OBJ_OUT_P(sol_r, sol);
    sol_r->decomp_scheme->dims[0].n = radix_r;
    sol_r->decomp_scheme->dims[0].in_stride =
        radix_m * sol->decomp_scheme->dims[0].out_stride;
    sol_r->decomp_scheme->dims[0].out_stride =
        radix_m * sol->decomp_scheme->dims[0].out_stride;
    sol_r->decomp_scheme->vecs[0].n = radix_m;
    sol_r->decomp_scheme->vecs[0].in_stride =
        sol->decomp_scheme->dims[0].out_stride;
    sol_r->decomp_scheme->vecs[0].out_stride =
        sol->decomp_scheme->dims[0].out_stride;

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

	return SOLVER_SUCCESS;
}


static INT32 execute_ct_solver(aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    INT32 status = SOLVER_SUCCESS;
    aoclfftz_solution_t *radix_r_sol = sol->next_sol[0];
    aoclfftz_solution_t *radix_m_sol = radix_r_sol->next_sol[0];
    // update radix-m & radix-r solution data pointers
    radix_m_sol->decomp_scheme->in_real  = sol->decomp_scheme->in_real;
    radix_m_sol->decomp_scheme->in_imag  = sol->decomp_scheme->in_imag;
    radix_m_sol->decomp_scheme->out_real = sol->decomp_scheme->out_real;
    radix_m_sol->decomp_scheme->out_imag = sol->decomp_scheme->out_imag;
    radix_m_sol->decomp_scheme->flags = sol->decomp_scheme->flags;

    radix_r_sol->decomp_scheme->in_real  = sol->decomp_scheme->out_real;
    radix_r_sol->decomp_scheme->in_imag  = sol->decomp_scheme->out_imag;
    radix_r_sol->decomp_scheme->out_real = sol->decomp_scheme->out_real;
    radix_r_sol->decomp_scheme->out_imag = sol->decomp_scheme->out_imag;
    radix_r_sol->decomp_scheme->flags = sol->decomp_scheme->flags;

    // execute radix-m sub-problem
    radix_m_sol->solver->execute_solver(radix_m_sol);

    if (IS_OUT_OF_PLACE(sol->decomp_scheme->flags))
    {
        status = twiddle_multiplier(radix_r_sol);
    }
    else
    {
        status = twiddle_multiplier_inplace(radix_r_sol);
    }

    if (status != SOLVER_SUCCESS)
    {
        return SOLVER_FAILURE;
    }

    // execute radix-r DFT
    radix_r_sol->solver->execute_solver(radix_r_sol);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return status;
}


static INT32 execute_ct_twiddle_solver(aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    INT32 status = SOLVER_SUCCESS;
    aoclfftz_solution_t *radix_r_sol = sol->next_sol[0];
    aoclfftz_solution_t *radix_m_sol = radix_r_sol->next_sol[0];

    // update radix-m & radix-r solution data pointers
    radix_m_sol->decomp_scheme->in_real  = sol->decomp_scheme->in_real;
    radix_m_sol->decomp_scheme->in_imag  = sol->decomp_scheme->in_imag;
    radix_m_sol->decomp_scheme->out_real = sol->decomp_scheme->out_real;
    radix_m_sol->decomp_scheme->out_imag = sol->decomp_scheme->out_imag;
    radix_m_sol->decomp_scheme->flags = sol->decomp_scheme->flags;

    // execute radix-m sub-problem
    radix_m_sol->solver->execute_solver(radix_m_sol);


    // execute radix-r DFT
    // Note: radix-r input strides are precomputed at setup time in selector_ct_dft
    radix_r_sol->decomp_scheme->in_real  = radix_m_sol->decomp_scheme->out_real;
    radix_r_sol->decomp_scheme->in_imag  = radix_m_sol->decomp_scheme->out_imag;
    radix_r_sol->decomp_scheme->out_real = sol->decomp_scheme->out_real;
    radix_r_sol->decomp_scheme->out_imag = sol->decomp_scheme->out_imag;
    radix_r_sol->decomp_scheme->flags = sol->decomp_scheme->flags;

    radix_r_sol->solver->execute_solver(radix_r_sol);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return status;
}

dft_solver_ register_execute_ct_solver(VOID)
{
    return execute_ct_solver;
}

dft_solver_ register_execute_ct_twiddle_solver(VOID)
{
    return execute_ct_twiddle_solver;
}
