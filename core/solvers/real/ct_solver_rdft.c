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
 *  @author S. Biplab Raut
 */

#include "core/common/memory_manager.h"

FFTZ_INT32 setup_real_ct_solver(aoclfftz_solution_t *sol,
                                aoclfftz_solution_t *sol_r,
                                aoclfftz_solution_t *sol_m, FFTZ_UINT32 radix_r,
                                FFTZ_UINT32 radix_m,
                                aoclfftz_realhelper_t *realhelper)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    // Setup radix-m sub-problem
    FFTZ_INT32 ret = copy_solution_obj(sol_m, sol);
    if (ret != AOCLFFTZ_SUCCESS)
    {
        AOCLFFTZ_ERROR("copy_solution_obj failed: %s", get_status_string(ret));
        return ret;
    }
    sol_m->decomp_scheme->dims[0].n = radix_m;
    sol_m->decomp_scheme->vecs[0].n = realhelper->problem_size / radix_m;

    // Setup radix-r sub-problem
    ret = copy_solution_obj(sol_r, sol);
    if (ret != AOCLFFTZ_SUCCESS)
    {
        AOCLFFTZ_ERROR("copy_solution_obj failed: %s", get_status_string(ret));
        return ret;
    }
    sol_r->decomp_scheme->dims[0].n = radix_r;
    sol_r->decomp_scheme->vecs[0].n = realhelper->problem_size / radix_r;

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SOLVER_SUCCESS;
}

/**
 * Recursive Real FFT solution tree (no SWAP):
 *   CT -> next_sol = radix_r (Direct, stage 0, R2HC/R2HCF real stage)
 *      -> radix_r->next_sol = radix_m (Direct C2C combine, or nested CT)
 *
 * PARTIAL_RECURSION: CT delegates to next_sol = radix_r, which then chains to
 *   radix_m via HAS_NEXT inside the Direct solver. Each Direct solver handles
 *   its own R2HC/R2HCF/C2C kernels and the twiddle multiplication internally.
 *
 * TRUE_RECURSION (default, SELECT_REAL_FFT_EXECUTION_ORDER=TRUE_RECURSION): the CT solver
 *   explicitly orchestrates its sub-problems, mirroring the Complex CT
 *   solver (core/solvers/complex/ct_solver_dft.c). The Direct nodes become
 *   pure leaves (they no longer chain via HAS_NEXT); the CT node drives the
 *   traversal:
 *       radix_r (recurse: real -> half-complex stage) -> radix_m (combine).
 *   The twiddle stays fused inside radix_m's C2C kernel, exactly as the
 *   Complex CT keeps the twiddle fused inside radix_r's C2C. The order is
 *   the real-FFT mirror of the complex order: the real-reading (R2HC) stage
 *   is radix_r (the CT child) and must execute before the C2C combine stage,
 *   whereas the complex CT executes radix_m (grandchild) before radix_r.
 *
 * The real leaf solvers resolve their buffers from the per-call ctx and their
 * setup-time roles, so ctx is forwarded unchanged. radix_m is the stage right
 * after radix_r, and every Direct CT stage ping-pongs the aux pools in ctx once
 * its kernels are done, so radix_m already sees the swapped aux pools.
 */
static FFTZ_INT32 execute_real_ct_solver(aoclfftz_solution_t *sol,
                                         aoclfftz_mutable_ctx_t *ctx)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    FFTZ_INT32 ret = SOLVER_SUCCESS;

    aoclfftz_solution_t *radix_r_sol = sol->next_sol;

#if REAL_FFT_EXECUTION_ORDER == REAL_FFT_ORDER_TRUE_RECURSION
    // Recurse into the radix-r (real-reading) sub-problem first.
    ret = radix_r_sol->solver->execute_solver(radix_r_sol, ctx);
    if (ret != SOLVER_SUCCESS)
    {
        return ret;
    }

    // Combine via the radix-m sub-problem (nested CT or Direct C2C leaf).
    // The twiddle multiplication is fused inside radix_m's C2C kernel. A CT node
    // is an r*m decomposition, so radix_m is always present (as in the complex
    // CT solver, which likewise executes it unconditionally).
    aoclfftz_solution_t *radix_m_sol = radix_r_sol->next_sol;
    ret = radix_m_sol->solver->execute_solver(radix_m_sol, ctx);
#else
    ret = radix_r_sol->solver->execute_solver(radix_r_sol, ctx);
#endif

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return ret;
}

dft_solver_ register_execute_real_ct_solver(FFTZ_VOID)
{
    return execute_real_ct_solver;
}
