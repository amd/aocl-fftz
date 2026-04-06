// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file executor.c
 *
 *  @brief Executes the DFT problem based on the solution found by selector.
 *
 *  This file contains the functions to execute a solution of kernels for the
 *  given input problem description.
 *
 *  @author S. Biplab Raut
 *  @author Prasandh Sankarankutty
 */

#include "core/executor.h"

static INT32 execute_dft(aoclfftz_executor_t *executor_obj)
{
    aoclfftz_solution_t *sol = executor_obj->solution;
#ifdef MULTI_THREADING
    // Retrive max nested levels from master application
    UINT32 cur_max_levels = omp_get_max_active_levels();

    // Set maximum nested levels to 3 as it is needed by real solutions
    omp_set_max_active_levels(3);
    INT32 ret = sol->solver->execute_solver(sol);

    // Restore max nested levels to original state
    omp_set_max_active_levels(cur_max_levels);
    return ret;
#else
    return sol->solver->execute_solver(sol);
#endif
}

execute_ register_execute_dft(VOID)
{
    return execute_dft;
}
