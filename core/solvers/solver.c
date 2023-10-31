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

 /** @file solver.h
 *
 *  @brief Functions of the Solver module.
 *
 *  This file contains Solver functions that are needed to solve a given problem.
 *
 *  @author S. Biplab Raut
 */

#include "core/solvers/solver.h"

//Table of solvers that is populated with applicable solvers at setup time
//ct, direct, nDim, buf, permKer, batched, bluestein, PFA, rader, permCopy,
//trans
dft_solver_ solvers_table[NUM_SOLVERS] = { 0x0, };

INT32 register_solvers(INT32 dt, INT32 cpu_flags)
{
	aoclfftz_solver_type solv_idx;

	for (solv_idx = SOLVER_DIRECT; solv_idx < NUM_SOLVERS_END; solv_idx++)
	{
		solvers_table[solv_idx] = NULL;
	}

	//Add all the available solvers
	solvers_table[SOLVER_DIRECT] = executor_direct_dft;
	solvers_table[SOLVER_CT] = executor_ct_dft;
	solvers_table[SOLVER_BATCHED] = executor_batched_dft;
	solvers_table[SOLVER_BLUESTEIN] = executor_bluestein_dft;

	return SOLVER_SUCCESS;
}

dft_solver_ get_solver_fp(aoclfftz_solution_t *sol)
{
	return sol->solver->execute_solver;
}

INT32 set_solver_fp(aoclfftz_generic_solver_t *solver_obj)
{
	if (solvers_table[solver_obj->solver_type] == NULL)
		return SOLVER_FAILURE;

	solver_obj->execute_solver = solvers_table[solver_obj->solver_type];
	return SOLVER_SUCCESS;
}

INT32 executor_direct_dft(aoclfftz_solution_t* solution)
{
	return execute_direct_solver(solution);
}

INT32 executor_ct_dft(aoclfftz_solution_t* solution)
{
	return execute_ct_solver(solution);
}

INT32 executor_batched_dft(aoclfftz_solution_t* solution)
{
	return execute_batched_solver(solution);
}

INT32 executor_bluestein_dft(aoclfftz_solution_t* solution)
{
	return execute_bluestein_solver(solution);
}
