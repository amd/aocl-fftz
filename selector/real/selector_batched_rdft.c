/**
 * Copyright (C) 2025, Advanced Micro Devices. All rights reserved.
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

#include "core/common/memory_manager.h"

INT32 selector_batched_rdft(aoclfftz_selector_t *sel, kernel_t *kertab,
                            aoclfftz_realhelper_t *realhelper)
{
#ifdef AOCL_ENABLE_LOG
    INT32 logger_mode = sel->solution->decomp_scheme->cntrl_params->logger_mode;
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Enter");
#endif
    aoclfftz_selector_t *cur_sel = NULL;
    INT32 vec_rank = sel->solution->decomp_scheme->vec_rank;
    INT32 dim_rank = sel->solution->decomp_scheme->dim_rank;
    INT32 stats_mode =
        sel->solution->decomp_scheme->cntrl_params->measure_stats;
    INT32 rnk = 0;
    INTP batch_size = 1;
    INT32 ret = SELECTOR_FAILURE;

    cur_sel = alloc_selector(vec_rank, dim_rank, sel->scratch_space,
                             0 /*unused*/);
    if (cur_sel == NULL)
    {
        goto exit_batched_dft;
    }

    // copy solution object from sel to cur_sel
    COPY_SOLUTION_OBJ(cur_sel->solution, sel->solution);

    UINT32 n_threads = 1;
#ifdef MULTI_THREADING
    UINT32 avl_threads = sel->solution->decomp_scheme->thread_info->avl_threads;
    UINT32 inner_batch = sel->solution->decomp_scheme->vecs[0].n;
    n_threads = (inner_batch < avl_threads) ? inner_batch : avl_threads;
    sel->solution->decomp_scheme->thread_info->n_threads = n_threads;
#endif


    // Setup batched solver to find the next solution for a single set/unit
    // of the vector problem
    if (n_threads <= 1)
    {
        ret = setup_real_batched_solver(sel->solution, cur_sel->solution,
                                        realhelper);
    }
#ifdef MULTI_THREADING
    else
    {
        ret = setup_real_mt_batched_solver(sel->solution, cur_sel->solution,
                                           realhelper);
    }
#endif
    if (ret != SELECTOR_SUCCESS)
    {
        goto exit_batched_dft;
    }

    // Call selector for solving a single set/unit of the vector problem
    ret = setup_rdft_(cur_sel, realhelper);
    if (ret != SELECTOR_SUCCESS)
    {
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
    sel->solution->next_sol[0] = cur_sel->solution;

    // destroy only the selector not the solution within it
    destroy_selector_without_solution(cur_sel);
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit");
#endif
    return SELECTOR_SUCCESS;

exit_batched_dft:
    destroy_selector_without_scratch_space(cur_sel);
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit");
#endif

    return ret;
}
