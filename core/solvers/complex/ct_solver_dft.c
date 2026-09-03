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

#include "core/common/memory_manager.h"


FFTZ_INT32 setup_ct_solver(aoclfftz_solution_t *sol, aoclfftz_solution_t *sol_r,
                      aoclfftz_solution_t *sol_m, FFTZ_UINT32 radix_r,
                      FFTZ_UINT32 radix_m)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    // Setup radix-m sub-problem
    // out-of-order -> in-order for out-of-place problems
    // out-of-order -> out-of-order for inplace problems
    FFTZ_INT32 ret = copy_solution_obj(sol_m, sol);
    if (ret != AOCLFFTZ_SUCCESS)
    {
        AOCLFFTZ_ERROR("copy_solution_obj failed: %s", get_status_string(ret));
        return ret;
    }
    // Set buffered flag for radix-m sub-problem if the original problem is
    // in-place.
    if (sol->decomp_scheme->in_real != NULL &&
        sol->decomp_scheme->in_real == sol->decomp_scheme->out_real)
    {
        SET_BUFFERED(sol_m->decomp_scheme->flags);
    }
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
    ret = copy_solution_obj_out_p(sol_r, sol);
    if (ret != AOCLFFTZ_SUCCESS)
    {
        AOCLFFTZ_ERROR("copy_solution_obj_out_p failed: %s",
                       get_status_string(ret));
        return ret;
    }
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


static FFTZ_INT32 execute_ct_solver(aoclfftz_solution_t *sol,
                                    aoclfftz_mutable_ctx_t *ctx)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    FFTZ_INT32 status = SOLVER_SUCCESS;
    aoclfftz_solution_t *radix_r_sol = sol->next_sol;
    aoclfftz_solution_t *radix_m_sol = radix_r_sol->next_sol;

    // Build child ctx for radix-m: same in/out as parent
    aoclfftz_mutable_ctx_t m_ctx = *ctx;

    // execute radix-m sub-problem
    radix_m_sol->solver->execute_solver(radix_m_sol, &m_ctx);

    // Build child ctx for radix-r: input is radix-m's (modified) output.
    aoclfftz_mutable_ctx_t r_ctx = *ctx;
    // Depending on radix-m being a buffered solver or not, m_ctx.out_real/out_imag
    // may point to a ct_buf slice or remain unchanged from the parent ctx.
    r_ctx.in_real = m_ctx.out_real;
    r_ctx.in_imag = m_ctx.out_imag;

    radix_r_sol->solver->execute_solver(radix_r_sol, &r_ctx);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return status;
}

dft_solver_ register_execute_ct_solver(FFTZ_VOID)
{
    return execute_ct_solver;
}
