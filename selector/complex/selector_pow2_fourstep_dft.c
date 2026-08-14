// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file selector_pow2_fourstep_dft.c
 *
 *  @brief Selector entry for the power-of-2 four-step solver
 *  (SOLVER_POW2_FOURSTEP).
 *
 *  Reached from the dispatcher once is_pow2_solvable accepts the
 *  problem (power-of-two whose working set spills L1 but whose ~sqrt(N)
 *  sub-FFTs stay L1-resident). Delegates to setup_pow2_fourstep_solver and
 *  records the cost estimate; setup declines (caller falls through to CT) when
 *  no fused kernel is available.
 *
 *  @author Ashwin K. Godbole
 */

#include "selector/selector.h"

FFTZ_INT32 selector_pow2_fourstep_dft(aoclfftz_selector_t *sel)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    if (sel == NULL || sel->solution == NULL ||
        sel->solution->decomp_scheme == NULL || sel->kernel_tables == NULL ||
        sel->kernel_tables->kt_twid_dft == NULL ||
        sel->kernel_tables->kt_dft == NULL)
    {
        AOCLFFTZ_LOG(INFO, global_logger_mode,
                     "Invalid selector or solution passed to "
                     "selector_pow2_fourstep_dft");
        return SELECTOR_FAILURE;
    }

    aoclfftz_solution_t *sol = sel->solution;
    FFTZ_INTP batch = sol->decomp_scheme->vecs[0].n;

    FFTZ_INT64 solver_ops = 0;
    FFTZ_INT32 ret = setup_pow2_fourstep_solver(sol, sel->kernel_tables->kt_dft,
                                           sel->kernel_tables->kt_twid_dft,
                                           &solver_ops);
    if (ret != SOLVER_SUCCESS)
    {
        AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit (fail)");
        return SELECTOR_FAILURE;
    }

    // Report cost in the shared cycle-based units: the per-FFT estimate scaled
    // by the batch count, so selector_driver_dft_ compares models on one scale.
    sel->cost_analysis->ops = solver_ops * batch;
    sel->cost_analysis->time = 0;

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SELECTOR_SUCCESS;
}

