// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file selector_buffered_dft.c
 *
 *  @brief Wrapper that acts on the buffered solver as guided by the selector.
 *
 *  This file contains the implementation of functions that are used to
 *  setup, factorize and evaluate sub-problems and kernels as applicable.
 *
 *  @author S. Biplab Raut
 */

#include "selector/selector.h"
#include "core/common/memory_manager.h"
#include "utils/utils.h"

FFTZ_INT32 selector_buffered_dft(aoclfftz_selector_t *sel, kernel_t *kertab)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    if (sel == NULL || sel->solution == NULL ||
        sel->solution->decomp_scheme == NULL)
    {
        AOCLFFTZ_LOG(INFO, global_logger_mode,
                     "Invalid selector or solution passed to "
                     "selector_buffered_dft");
        return SELECTOR_FAILURE;
    }

    aoclfftz_selector_t *cur_sel = NULL;
    FFTZ_INT32 vec_rank = sel->solution->decomp_scheme->vec_rank;
    FFTZ_INT32 dim_rank = sel->solution->decomp_scheme->dim_rank;
    FFTZ_INT32 ret = SELECTOR_FAILURE;

    cur_sel = alloc_selector(vec_rank, dim_rank, sel->kernel_tables,
                             sel->has_nested);
    if (cur_sel == NULL)
    {
        ret = AOCLFFTZ_MEMORY_FAILURE;
        goto exit_buffered_dft;
    }

    // copy solution object from sel to cur_sel
    ret = copy_solution_obj(cur_sel->solution, sel->solution);
    if (ret != AOCLFFTZ_SUCCESS)
    {
        AOCLFFTZ_ERROR("copy_solution_obj failed: %s", get_status_string(ret));
        goto exit_buffered_dft;
    }
    // Clear buffered flag inherited from sel. The flag applies only to
    // the current sel node; cur_sel's buffering is determined by selector logic
    UNSET_BUFFERED(cur_sel->solution->decomp_scheme->flags);

    // Setup buffered solver
    ret = setup_buffered_solver(sel->solution, cur_sel->solution);

    if (ret != SELECTOR_SUCCESS)
    {
        goto exit_buffered_dft;
    }

    // Call selector for solving the problem
    ret = selector_model_dft_(cur_sel);
    if (ret != SELECTOR_SUCCESS)
    {
        goto exit_buffered_dft;
    }

    // Copy cost from cur_sel back to sel
    sel->cost_analysis->ops = cur_sel->cost_analysis->ops;
    sel->cost_analysis->time = cur_sel->cost_analysis->time;

    sel->solution->next_sol = alloc_sol_array(1 /*n_threads*/);
    if (sel->solution->next_sol == NULL)
    {
        ret = AOCLFFTZ_MEMORY_FAILURE;
        goto exit_buffered_dft;
    }
    sel->solution->next_sol[0] = cur_sel->solution;

    // destroy only the selector not the solution within it
    destroy_selector_without_solution(cur_sel);
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return SELECTOR_SUCCESS;

exit_buffered_dft:
    destroy_selector(cur_sel);
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return ret;
}
