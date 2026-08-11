// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file selector_batched_rdft.c
 *
 *  @brief Wrapper that acts on the batched solver as guided by the selector.
 *
 *  This file contains the implementation of functions that are used to
 *  handle the batches of multi-batched problems.
 *
 *  @author Srirammaswamy Srinivsan
 *  @author S. Biplab Raut
 */

#include "selector/selector.h"
#include "core/common/memory_manager.h"
#ifdef MULTI_THREADING
#include "utils/thread_control.h"
#endif

FFTZ_INT32 selector_batched_rdft(aoclfftz_selector_t *sel, kernel_t *kertab,
                            aoclfftz_realhelper_t *realhelper)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    if (sel == NULL || sel->solution == NULL ||
        sel->solution->decomp_scheme == NULL)
    {
        AOCLFFTZ_LOG(INFO, global_logger_mode,
                     "Invalid selector or solution passed to "
                     "selector_batched_rdft");
        return SELECTOR_FAILURE;
    }

    aoclfftz_selector_t *cur_sel = NULL;
    FFTZ_INT32 vec_rank = sel->solution->decomp_scheme->vec_rank;
    FFTZ_INT32 dim_rank = sel->solution->decomp_scheme->dim_rank;
    FFTZ_INT32 stats_mode =
        sel->solution->decomp_scheme->cntrl_params->measure_stats;
    FFTZ_INT32 rnk = 0;
    FFTZ_INTP batch_size = 1;
    FFTZ_INT32 ret = SELECTOR_FAILURE;
    FFTZ_INT32 setup_ret = AOCLFFTZ_SETUP_FAILURE;

    cur_sel = alloc_selector(vec_rank, dim_rank, sel->kernel_tables,
                             sel->has_nested);
    if (cur_sel == NULL)
    {
        AOCLFFTZ_ERROR("Sub-problem selector allocation failed");
        ret = SELECTOR_FAILURE;
        goto exit_batched_dft;
    }

    // copy solution object from sel to cur_sel
    setup_ret = copy_solution_obj(cur_sel->solution, sel->solution);
    if (setup_ret != AOCLFFTZ_SUCCESS)
    {
        AOCLFFTZ_ERROR("Copying parent solution to sub-problem failed: %s",
                       get_status_string(setup_ret));
        ret = SELECTOR_FAILURE;
        goto exit_batched_dft;
    }

    FFTZ_INT32 n_threads = 1;
#ifdef MULTI_THREADING
    FFTZ_INT32 avl_threads =
        sel->solution->decomp_scheme->thread_info->avl_threads;
    FFTZ_INT32 inner_batch = sel->solution->decomp_scheme->vecs[0].n;
    n_threads = (inner_batch < avl_threads) ? inner_batch : avl_threads;
    n_threads = cap_batch_loop_threads(sel->solution->decomp_scheme, n_threads);
    sel->solution->decomp_scheme->thread_info->n_threads = n_threads;
#endif

    // ST vs MT batched solver selection, based on the effective thread count.
    // Only the decision (solver_type) is recorded here; binding the execute
    // function pointer is the parent's responsibility (done at the selector
    // level once this selector returns).
    aoclfftz_generic_solver_t *solver_obj = sel->solution->solver;
    if (n_threads == 1)
    {
        solver_obj->solver_type = SOLVER_REAL_BATCHED;
    }
    else
    {
        solver_obj->solver_type = SOLVER_REAL_MT_BATCHED;
    }

    // Setup batched solver to find the next solution for a single set/unit
    // of the vector problem
    if (n_threads == 1)
    {
        setup_ret = setup_real_batched_solver(sel->solution, cur_sel->solution,
                                              realhelper);
    }
#ifdef MULTI_THREADING
    else
    {
        setup_ret = setup_real_mt_batched_solver(sel->solution,
                                                 cur_sel->solution, realhelper,
                                                 sel->has_nested);
    }
#endif
    if (setup_ret != SOLVER_SUCCESS)
    {
        AOCLFFTZ_ERROR("%s batched solver setup failed (status %d)",
                       (n_threads == 1) ? "ST" : "MT", setup_ret);
        ret = SELECTOR_FAILURE;
        goto exit_batched_dft;
    }

    // Call selector for solving a single set/unit of the vector problem
    ret = selector_model_rdft_(cur_sel, realhelper);
    if (ret != SELECTOR_SUCCESS)
    {
        AOCLFFTZ_ERROR("Failed to select a proper solution for a single real "
                       "FFT batch");
        goto exit_batched_dft;
    }

    // Calculate the batch size of all the sub-problems in the vector problem
    for (rnk = 0; rnk < vec_rank; rnk++)
    {
        batch_size *= sel->solution->decomp_scheme->vecs[rnk].n;
    }

    sel->cost_analysis->ops = batch_size * cur_sel->cost_analysis->ops;
    sel->cost_analysis->time = batch_size * cur_sel->cost_analysis->time;

    if (stats_mode)
    {
        // capture stats
    }

    sel->solution->next_sol = alloc_sol_array(n_threads);
    if (sel->solution->next_sol == NULL)
    {
        setup_ret = AOCLFFTZ_MEMORY_FAILURE;
        AOCLFFTZ_ERROR("alloc_sol_array failed: %s",
                       get_status_string(setup_ret));
        ret = SELECTOR_FAILURE;
        goto exit_batched_dft;
    }
    sel->solution->next_sol[0] = cur_sel->solution;

    // destroy only the selector not the solution within it
    destroy_selector_without_solution(cur_sel);
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SELECTOR_SUCCESS;

exit_batched_dft:
    destroy_selector(cur_sel);
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return ret;
}
