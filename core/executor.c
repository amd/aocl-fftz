/**
 * Copyright (C) 2023-2025, Advanced Micro Devices. All rights reserved.
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
