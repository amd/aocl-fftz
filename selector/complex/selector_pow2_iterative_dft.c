// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file selector_pow2_iterative_dft.c
 *
 *  @brief Selector entry for the power-of-2 iterative solver
 *
 *  Reached from the dispatcher once is_pow2_iterative_applicable accepts the
 *  problem. Delegates solver setup (radix-stage decomposition + ping-pong
 *  buffers) to setup_pow2_iterative_solver and records the cost estimate.
 *
 *  @author Ashwin K. Godbole
 */

#include "selector/selector.h"

FFTZ_INT32 selector_pow2_iterative_dft(aoclfftz_selector_t *sel)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    if (sel == NULL || sel->solution == NULL ||
        sel->solution->decomp_scheme == NULL || sel->kernel_tables == NULL ||
        sel->kernel_tables->kt_twid_dft == NULL ||
        sel->kernel_tables->kt_dft == NULL)
    {
        AOCLFFTZ_LOG(INFO, global_logger_mode,
                     "Invalid selector or solution passed to "
                     "selector_pow2_iterative_dft");
        return SELECTOR_FAILURE;
    }

    aoclfftz_solution_t *sol = sel->solution;
    FFTZ_INTP batch = sol->decomp_scheme->vecs[0].n;

    FFTZ_INT64 solver_cost = 0;
    FFTZ_INT32 ret = setup_pow2_iterative_solver(sol, sel->kernel_tables->kt_dft,
                                            sel->kernel_tables->kt_twid_dft,
                                            &solver_cost);
    if (ret != SOLVER_SUCCESS)
    {
        AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit (fail)");
        return SELECTOR_FAILURE;
    }

    // Report the cost in the same cycle-based units as the other selectors
    // (e.g. selector_batched_ct_l1_direct): the per-FFT stage-kernel cost
    // estimate scaled by the number of batched transforms. selector_driver_dft_
    // compares models by ops, so this keeps pow2-iterative on the same scale.
    sel->cost_analysis->ops = solver_cost * batch;
    sel->cost_analysis->time = 0;

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SELECTOR_SUCCESS;
}
