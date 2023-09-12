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

    //Setup radix-m sub-problem
    COPY_SOLUTION_OBJ(sol_m, sol);
    sol_m->decomp_scheme->dims[0].n = radix_m;
    sol_m->decomp_scheme->dims[0].in_stride =
        (sol->decomp_scheme->vecs[0].n == 1) ? //First recursion level
        radix_r * sol->decomp_scheme->dims[0].in_stride :
        sol->decomp_scheme->vecs[0].in_stride; //Next recursion level
    sol_m->decomp_scheme->dims[0].out_stride =
        (sol->decomp_scheme->vecs[0].n == 1) ? //First recursion level
        radix_r * sol->decomp_scheme->dims[0].out_stride :
        sol->decomp_scheme->vecs[0].out_stride; //Next recursion level
    sol_m->decomp_scheme->vecs[0].n = radix_r;
    sol_m->decomp_scheme->vecs[0].in_stride =
        (sol->decomp_scheme->vecs[0].n == 1) ? //First recursion level
        radix_m :
        radix_r * sol->decomp_scheme->vecs[0].in_stride; //Next recursion level;
    sol_m->decomp_scheme->vecs[0].out_stride =
        (sol->decomp_scheme->vecs[0].n == 1) ? //First recursion level
        radix_m :
        radix_r * sol->decomp_scheme->vecs[0].out_stride; //Next recursion level;

    //Setup radix-r sub-problem
    COPY_SOLUTION_OBJ(sol_r, sol);
    sol_r->decomp_scheme->dims[0].n = radix_r;
    sol_r->decomp_scheme->dims[0].in_stride =
        (sol->decomp_scheme->vecs[0].n == 1) ? //First recursion level
        radix_m * sol->decomp_scheme->dims[0].in_stride :
        sol->decomp_scheme->vecs[0].in_stride; //Next recursion level
    sol_r->decomp_scheme->dims[0].out_stride =
        (sol->decomp_scheme->vecs[0].n == 1) ? //First recursion level
        sol->decomp_scheme->dims[0].out_stride ://For in-order output
        sol->decomp_scheme->vecs[0].out_stride; //Next recursion level
    sol_r->decomp_scheme->vecs[0].n = radix_m;
    sol_r->decomp_scheme->vecs[0].in_stride =
        (sol->decomp_scheme->vecs[0].n == 1) ? //First recursion level
        sol->decomp_scheme->vecs[0].in_stride :
        radix_r * sol->decomp_scheme->vecs[0].in_stride; //Next recursion level;
    sol_r->decomp_scheme->vecs[0].out_stride =
        (sol->decomp_scheme->vecs[0].n == 1) ? //First recursion level
        sol->decomp_scheme->vecs[0].out_stride :
        radix_r * sol->decomp_scheme->vecs[0].out_stride; //Next recursion level;
    //Swap pointers in case of out-of-place problem
    if (IS_OUT_OF_PLACE(sol->decomp_scheme->flags))
    {
        sol_r->decomp_scheme->in_real = sol->decomp_scheme->out_real;
        sol_r->decomp_scheme->in_imag = sol->decomp_scheme->out_imag;
        sol_r->decomp_scheme->out_real = sol->decomp_scheme->in_real;
        sol_r->decomp_scheme->out_imag = sol->decomp_scheme->in_imag;
    }

	return SOLVER_SUCCESS;
}

INT32 execute_ct_solver(aoclfftz_solution_t* sol)
{
    aoclfftz_generic_solver_t* solver_obj = sol->solver;
    aoclfftz_strides_t* strides = sol->strides;
    INT32 status = SOLVER_SUCCESS;

    //Call CT solver executor recursively for factors/sub-problems in a
    //depth-first way
    if (sol->next_sol != NULL)
    {
        //Depth-first recursive solving of the radix-m DFT sub-problem
        if (execute_ct_solver(sol->next_sol) != SOLVER_SUCCESS)
            return SOLVER_FAILURE;

        //Perform inter-stage twiddle factor multiplications
        if (twiddle_multiplier(sol) != SOLVER_SUCCESS)
            return SOLVER_FAILURE;

        //Solve radix-r DFT sub-problem
        if (solver_obj->kernel_r)
        {
            strides->in_stride = sol->decomp_scheme->dims[0].in_stride;
            strides->out_stride = sol->decomp_scheme->dims[0].out_stride;
            strides->v_in_stride = sol->decomp_scheme->vecs[0].in_stride;
            strides->v_out_stride = sol->decomp_scheme->vecs[0].out_stride;

            solver_obj->kernel_r(sol->decomp_scheme->in_real,
                sol->decomp_scheme->in_imag,
                sol->decomp_scheme->out_real,
                sol->decomp_scheme->out_imag,
                sol->decomp_scheme->vecs[0].n,
                strides);
        }
    }
    //Solve DFT computations for the leaf-level sub-problems
    else
    {
        //Solve radix-r DFT sub-problem
        if (solver_obj->kernel_r)
        {
            strides->in_stride = sol->decomp_scheme->dims[0].in_stride;
            strides->out_stride = sol->decomp_scheme->dims[0].out_stride;
            strides->v_in_stride = sol->decomp_scheme->vecs[0].in_stride;
            strides->v_out_stride = sol->decomp_scheme->vecs[0].out_stride;

            solver_obj->kernel_r(sol->decomp_scheme->in_real,
                sol->decomp_scheme->in_imag,
                sol->decomp_scheme->out_real,
                sol->decomp_scheme->out_imag,
                sol->decomp_scheme->vecs[0].n,
                strides);
        }
    }

    return status;
}