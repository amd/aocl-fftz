// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file selector_ndim_dft.c
 *
 *  @brief Wrapper that invokes the ND solver as guided by the selector.
 *
 *  This file contains the implementation of functions that are used to
 *  setup, split and evaluate the sub-problems of ND as applicable.
 *
 *  @author Prasandh Sankarankutty
 *  @author S. Biplab Raut
 */

#include "selector/selector.h"
#include "core/common/memory_manager.h"

FFTZ_INT32 selector_ndim_dft(aoclfftz_selector_t *sel, kernel_t *kertab)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    if (sel == NULL || sel->solution == NULL ||
        sel->solution->decomp_scheme == NULL)
    {
        AOCLFFTZ_LOG(INFO, global_logger_mode,
                     "Invalid selector or solution passed to "
                     "selector_ndim_dft");
        return SELECTOR_FAILURE;
    }

    aoclfftz_selector_t *n_minus1_sel = NULL;
    aoclfftz_selector_t *outer_dim_sel = NULL;

    FFTZ_INT32 dim_rank = sel->solution->decomp_scheme->dim_rank;
    FFTZ_INT32 stats_mode = sel->solution->decomp_scheme->cntrl_params->
                       measure_stats;
    FFTZ_INT32 ret = SELECTOR_FAILURE;

    n_minus1_sel = alloc_selector(1, dim_rank - 1, sel->kernel_tables,
                                  sel->has_nested);
    outer_dim_sel = alloc_selector(dim_rank - 1, 1, sel->kernel_tables,
                                   sel->has_nested);

    if (n_minus1_sel == NULL || outer_dim_sel == NULL)
    {
        ret = AOCLFFTZ_MEMORY_FAILURE;
        goto exit_nd_dft;
    }

    ret = setup_ndim_solver(sel->solution, n_minus1_sel->solution,
                            outer_dim_sel->solution);
    if (ret != SELECTOR_SUCCESS)
    {
        goto exit_nd_dft;
    }

    // Invoke selector for solving ND-1 sub-problem
    ret = selector_model_dft_(n_minus1_sel);
    if (ret != SELECTOR_SUCCESS)
    {
        goto exit_nd_dft;
    }

    // Invoke selector for solving 1D sub-problem
    ret = selector_model_dft_(outer_dim_sel);
    if (ret != SELECTOR_SUCCESS)
    {
        goto exit_nd_dft;
    }

    // Propagate ndim_concurrency upward from n_minus1_sel only: the outer-dim child
    // is always 1D, so a nested ndim node can arise solely in the (N-1)D subtree.
    sel->solution->decomp_scheme->thread_info->ndim_concurrency =
        n_minus1_sel->solution->decomp_scheme->thread_info->ndim_concurrency;

    sel->cost_analysis->ops = n_minus1_sel->cost_analysis->ops +
                              outer_dim_sel->cost_analysis->ops;
    sel->cost_analysis->time = n_minus1_sel->cost_analysis->time +
                               outer_dim_sel->cost_analysis->time;

    if (stats_mode)
    {
        // capture stats
    }
    sel->solution->next_sol = outer_dim_sel->solution;
    sel->solution->dft_bufs->nd_sol = n_minus1_sel->solution;

    destroy_selector_without_solution(n_minus1_sel);
    destroy_selector_without_solution(outer_dim_sel);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return SELECTOR_SUCCESS;

exit_nd_dft:
    destroy_selector(n_minus1_sel);
    destroy_selector(outer_dim_sel);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return ret;
}
