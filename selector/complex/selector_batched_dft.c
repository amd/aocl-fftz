// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file selector_batched_dft.c
 *
 *  @brief Wrapper that acts on the batched solver as guided by the selector.
 *
 *  This file contains the implementation of functions that are used to
 *  handle the batches of multi-batched problems.
 *
 *  @author S. Biplab Raut
 */

#include "selector/selector.h"
#include "core/common/memory_manager.h"
#include "utils/utils.h"

FFTZ_INT32 selector_batched_dft(aoclfftz_selector_t *sel, kernel_t *kertab)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    if (sel == NULL || sel->solution == NULL ||
        sel->solution->decomp_scheme == NULL)
    {
        AOCLFFTZ_LOG(INFO, global_logger_mode,
                     "Invalid selector or solution passed to "
                     "selector_batched_dft");
        return SELECTOR_FAILURE;
    }

    aoclfftz_selector_t *cur_sel = NULL;
    FFTZ_INT32 vec_rank = sel->solution->decomp_scheme->vec_rank;
    FFTZ_INT32 dim_rank = sel->solution->decomp_scheme->dim_rank;
    FFTZ_INT32 stats_mode = sel->solution->decomp_scheme->cntrl_params->
                       measure_stats;
    aoclfftz_solution_t *sol = sel->solution;
    FFTZ_INT32 rnk = 0;
    FFTZ_INTP batch_size = 1;
    FFTZ_INT32 ret = SELECTOR_FAILURE;

    cur_sel = alloc_selector(vec_rank, dim_rank, sel->kernel_tables);
    if (cur_sel == NULL)
    {
        ret = AOCLFFTZ_MEMORY_FAILURE;
        goto exit_batched_dft;
    }

    FFTZ_UINT8 is_col_major = check_col_major(sol->decomp_scheme);
    // Bluestein problems are excluded from the batched direct optimization
    // because the memory layout of bluestein buffer allocated is always row
    // major
    if (is_col_major && !check_bluestein_problem(sol->decomp_scheme) &&
        dim_rank == 1)
    {
        // allocate a new struct in the direct solver to hold the
        // vecs[0] from batched solver
        ALLOC_ALIGN_UNINIT(sol->decomp_scheme->batched_vecs, aoclfftz_dim_t_64_,
                           sizeof(aoclfftz_dim_t_64_));
        if (sol->decomp_scheme->batched_vecs != NULL)
        {
            sol->decomp_scheme->batched_vecs[0].n =
                sol->decomp_scheme->vecs[0].n;
            sol->decomp_scheme->batched_vecs[0].in_stride =
                sol->decomp_scheme->vecs[0].in_stride;
            sol->decomp_scheme->batched_vecs[0].out_stride =
                sol->decomp_scheme->vecs[0].out_stride;

            aoclfftz_dim_t_64_ *vecs = sol->decomp_scheme->vecs;
            if (vec_rank == 1)
            {
                // vec_rank must be at least 1, so set vecs[0].n = 1 to make it
                // non-batched
                sol->decomp_scheme->vecs[0].n = 1;
                sol->decomp_scheme->vecs[0].in_stride = 1;
                sol->decomp_scheme->vecs[0].out_stride = 1;
            }
            else
            {
                // remove vecs[0] from batched solver
                sol->decomp_scheme->vec_rank -= 1;
                for (FFTZ_INT32 i = 0; i < sol->decomp_scheme->vec_rank; i++)
                {
                    vecs[i].n = vecs[i + 1].n;
                    vecs[i].in_stride = vecs[i + 1].in_stride;
                    vecs[i].out_stride = vecs[i + 1].out_stride;
                }
            }
        }
    }

    // copy solution object from sel to cur_sel
    ret = copy_solution_obj(cur_sel->solution, sel->solution);
    if (ret != AOCLFFTZ_SUCCESS)
    {
        AOCLFFTZ_ERROR("copy_solution_obj failed: %s", get_status_string(ret));
        goto exit_batched_dft;
    }

    FFTZ_INT32 n_threads = 1;
#ifdef MULTI_THREADING
    // TODO: Multi-threaded parallelism has been applied only to the inner most
    // dimension of a Multi-Dimensional batched problem, need to support other
    // dimensions later with CPUPL-6843
    FFTZ_INT32 avl_threads =
        sel->solution->decomp_scheme->thread_info->avl_threads;
    FFTZ_INT32 inner_batch = sel->solution->decomp_scheme->vecs[0].n;
    if (sol->decomp_scheme->batched_vecs)
    {
        // Avoid nested parallelism: use 1 thread if inner_batch is small,
        // otherwise all threads
        n_threads = (inner_batch < avl_threads) ? 1 : avl_threads;
    }
    else
    {
        // Standard case: use minimum of inner_batch and available threads
        n_threads = (inner_batch < avl_threads) ? inner_batch : avl_threads;
    }
    sel->solution->decomp_scheme->thread_info->n_threads = n_threads;
#endif

    if (n_threads == 1)
    {
        // Setup batched solver to find the next solution for a single set/unit
        // of the vector problem
        sel->solution->solver->solver_type = SOLVER_BATCHED;
        sel->solution->solver->execute_solver =
            register_execute_batched_solver();
        ret = setup_batched_solver(cur_sel->solution);
    }
#ifdef MULTI_THREADING
    else
    {
        // Setup multi threaded batched solver to find solution for a
        // vector problem
        sel->solution->solver->solver_type = SOLVER_MT_BATCHED;
        sel->solution->solver->execute_solver =
            register_execute_mt_batched_solver();
        ret = setup_mt_batched_solver(cur_sel->solution, n_threads);
    }
#endif

    if (ret != SELECTOR_SUCCESS)
    {
        goto exit_batched_dft;
    }

    // Call selector for solving a single set/unit of the vector problem
    ret = selector_model_dft_(cur_sel);
    if (ret != SELECTOR_SUCCESS)
    {
        goto exit_batched_dft;
    }

    /*
    * ndim_concurrency rules:
    *     - This variable is used when the problem type is NDIM.
    *     - It first takes effect at an MT_BATCHED -> NDIM boundary, where it is
    *       initialised to the MT_BATCHED node's n_threads.
    *     - At every subsequent MT_BATCHED -> NDIM boundary, it is scaled up by that
    *       MT_BATCHED node's n_threads.
    *
    * ct_buf_size rules:
    *     Setup time can be broken down into two phases:
    *         - Phase 1: Solver tree construction phase
    *             Base setters (allocate ct_buffer, set ct_buf_size to a per-slice base size):
    *                 - BUFFERED solver
    *                 - CTL1D solver
    *                 - NDIM solver
    *                 Any other solver which is a child of these solvers
    *                 simply copies its parent's ct_buf_size down to its child.
    *
    *         - Phase 2: Unwinding phase, here two selectors propagate ct_buf_size
    *             from the child to the parent:
    *             - CT selector: copies the radix-m child's ct_buf_size as-is (the
    *               radix-r child never touches ct_buffer).
    *             - Batched selector: Scales up ct_buf_size based on the need for concurrent slots
    *                 1. NDIM child: scale the child's ct_buf_size by ndim_concurrency,
    *                    and set this [mt_]batched node's own ct_buf_size to
    *                    n_threads * child_ct_buf_size. This propagated value is
    *                    consumed only by a real NDIM solver sitting above this
    *                    [mt_]batched node.
    *                 2. non-NDIM child: multiply ct_buf_size by n_threads.
    *
    *     Execution time responsibilities:
    *         - MT_BATCHED advances each thread's ct_offset by tid * next_sol_ct_buf_size,
    *           then handsover ct_buf_size to the next solver.
    */
    if (cur_sel->solution->solver->solver_type == SOLVER_NDIM)
    {
        FFTZ_INT32 child_ndim_conc =
            cur_sel->solution->decomp_scheme->thread_info->ndim_concurrency;
        if(child_ndim_conc)
        {
            // This value is used by MT_BATCHED solver to advance the
            // ct_offset by `tid * ct_buf_size`
            cur_sel->solution->dft_bufs->ct_buf_size =
                child_ndim_conc * cur_sel->solution->dft_bufs->ct_buf_size;
            // This propagated ct_buf_size is consumed only when a real NDIM
            // solver sits above this batched node.
            sel->solution->dft_bufs->ct_buf_size =
                (FFTZ_INTP)n_threads * cur_sel->solution->dft_bufs->ct_buf_size;
            sel->solution->decomp_scheme->thread_info->ndim_concurrency =
                child_ndim_conc * n_threads;
        }
        else
        {
            // This propagated ct_buf_size is consumed only when a real NDIM
            // solver sits above this batched node.
            sel->solution->dft_bufs->ct_buf_size =
                (FFTZ_INTP)n_threads * cur_sel->solution->dft_bufs->ct_buf_size;
            sel->solution->decomp_scheme->thread_info->ndim_concurrency =
                n_threads;
        }
    }
    else
    {
        // Handles the case of multi-level CT problem and also real NDIM problem
        sel->solution->dft_bufs->ct_buf_size =
            (FFTZ_INTP)n_threads * cur_sel->solution->dft_bufs->ct_buf_size;
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
    // TODO: Instead of allocating an array of solutions (one sized), just allocate a single solution
    sel->solution->next_sol = alloc_sol_array(1);
    if (sel->solution->next_sol == NULL)
    {
        ret = AOCLFFTZ_MEMORY_FAILURE;
        AOCLFFTZ_ERROR("alloc_sol_array failed: %s", get_status_string(ret));
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
