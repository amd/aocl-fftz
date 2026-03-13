/**
 * Copyright (C) 2023-2026, Advanced Micro Devices. All rights reserved.
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

INT32 selector_buffered_dft(aoclfftz_selector_t *sel, kernel_t *kertab)
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
    INT32 vec_rank = sel->solution->decomp_scheme->vec_rank;
    INT32 dim_rank = sel->solution->decomp_scheme->dim_rank;
    INT32 ret = SELECTOR_FAILURE;

    cur_sel = alloc_selector(vec_rank, dim_rank, sel->scratch_space,
                             sel->kernel_tables, 0 /*unused*/);
    if (cur_sel == NULL)
    {
        ret = AOCLFFTZ_MEMORY_FAILURE;
        goto exit_buffered_dft;
    }

    // copy solution object from sel to cur_sel
    COPY_SOLUTION_OBJ(cur_sel->solution, sel->solution);
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
    destroy_selector_without_scratch_space(cur_sel);
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return ret;
}
