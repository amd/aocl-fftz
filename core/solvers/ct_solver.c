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

 /** @file ct_solver.c
 *
 *  @brief Cooley Tukey Solver that decomposes and solves an input problem
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

INT32 setup_ct_solver(aoclfftz_solution_t *sol,
                      aoclfftz_solution_t *sol_r,
                      aoclfftz_solution_t *sol_m,
                      UINT32 radix_r,
                      UINT32 radix_m)
{
    // Setup radix-m sub-problem
    // out-of-order -> in-order for out-of-place problems
    // out-of-order -> out-of-order for inplace problems
    COPY_SOLUTION_OBJ(sol_m, sol);
    sol_m->decomp_scheme->dims[0].n = radix_m;
    sol_m->decomp_scheme->dims[0].in_stride =
        radix_r * sol->decomp_scheme->dims[0].in_stride;
    sol_m->decomp_scheme->dims[0].out_stride =
        (IS_OUT_OF_PLACE(sol->decomp_scheme->flags)) ?
        sol->decomp_scheme->dims[0].out_stride :
        radix_r * sol->decomp_scheme->dims[0].out_stride;
    sol_m->decomp_scheme->vecs[0].n = radix_r;
    sol_m->decomp_scheme->vecs[0].in_stride =
        sol->decomp_scheme->dims[0].in_stride;
    sol_m->decomp_scheme->vecs[0].out_stride =
        (IS_OUT_OF_PLACE(sol->decomp_scheme->flags)) ?
        radix_m * sol->decomp_scheme->dims[0].out_stride :
        sol->decomp_scheme->dims[0].out_stride;

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

	return SOLVER_SUCCESS;
}

INT32 execute_ct_solver(aoclfftz_solution_t* sol)
{
    INT32 logger_mode = sol->decomp_scheme->cntrl_params->logger_mode;
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Enter");
    INT32 status = SOLVER_SUCCESS;
    aoclfftz_solution_t *radix_r_sol = sol->next_sol;
    aoclfftz_solution_t *radix_m_sol = radix_r_sol->next_sol;

    // update radix-m & radix-r solution data pointers
    radix_m_sol->decomp_scheme->in_real  = sol->decomp_scheme->in_real;
    radix_m_sol->decomp_scheme->in_imag  = sol->decomp_scheme->in_imag;
    radix_m_sol->decomp_scheme->out_real = sol->decomp_scheme->out_real;
    radix_m_sol->decomp_scheme->out_imag = sol->decomp_scheme->out_imag;

    radix_r_sol->decomp_scheme->in_real  = sol->decomp_scheme->out_real;
    radix_r_sol->decomp_scheme->in_imag  = sol->decomp_scheme->out_imag;
    radix_r_sol->decomp_scheme->out_real = sol->decomp_scheme->out_real;
    radix_r_sol->decomp_scheme->out_imag = sol->decomp_scheme->out_imag;

    // execute radix-m sub-problem
    if(radix_m_sol->solver->execute_solver(radix_m_sol) != SOLVER_SUCCESS)
    {
        return SOLVER_FAILURE;
    }

    if(IS_OUT_OF_PLACE(sol->decomp_scheme->flags))
    {
        status = twiddle_multiplier(radix_r_sol);
    }
    else
    {
        status = twiddle_multiplier_inplace(radix_r_sol);
    }

    if(status != SOLVER_SUCCESS)
    {
        return SOLVER_FAILURE;
    }

    // execute radix-r DFT
    if(radix_r_sol->solver->execute_solver(radix_r_sol) != SOLVER_SUCCESS)
    {
        return SOLVER_FAILURE;
    }

    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit");

    return status;
}
