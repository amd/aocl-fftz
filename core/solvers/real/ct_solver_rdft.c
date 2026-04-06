// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file ct_solver_rdft.c
 *
 *  @brief Cooley Tukey Solver that decomposes and solves an input problem
 *
 *  This file contains the functions that setup, execute and destroy
 *  the solver.
 *
 *  @author Srirammaswamy Srinivasan
 */

#include "core/common/memory_manager.h"

INT32 setup_real_ct_solver(aoclfftz_solution_t *sol, aoclfftz_solution_t *sol_r,
                           aoclfftz_solution_t *sol_m, UINT32 radix_r,
                           UINT32 radix_m, aoclfftz_realhelper_t *realhelper)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    // Setup radix-m sub-problem
    COPY_SOLUTION_OBJ(sol_m, sol);
    sol_m->decomp_scheme->dims[0].n = radix_m;
    sol_m->decomp_scheme->vecs[0].n = realhelper->problem_size / radix_m;

    // Setup radix-r sub-problem
    COPY_SOLUTION_OBJ(sol_r, sol);
    sol_r->decomp_scheme->dims[0].n = radix_r;
    sol_r->decomp_scheme->vecs[0].n = realhelper->problem_size / radix_r;

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SOLVER_SUCCESS;
}

static INT32 execute_real_ct_solver(aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    INT32 ret = SOLVER_SUCCESS;

    // Execute the next direct solution
    // NOTE: The CT problem is executed in the execute_real_direct_solver,
    // along with direct problems
    ret = sol->next_sol[0]->solver->execute_solver(sol->next_sol[0]);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return ret;
}

dft_solver_ register_execute_real_ct_solver(VOID)
{
    return execute_real_ct_solver;
}
