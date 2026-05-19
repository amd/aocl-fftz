/**
 * Copyright (C) 2023-2025, Advanced Micro Devices. All rights reserved.
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

// In a single-threaded scenario, for the outer_dim_sol,
// if the dims are regular strided (where strides are proportional to the prev
// dim size), it is optimal to fuse those dims together and execute them as a
// single dim, as opposed to recursive calls for each dim.
// This function checks for such regular strided cases and returns the number of
// dims that can be fused.
INT32 get_fusable_dims(aoclfftz_solution_t *sol, INT32 dim_rank)
{
    INT32 fusable_dims = 1;

    // do not club cases where in_stride != out_stride
    if (sol->decomp_scheme->dims[0].in_stride !=
        sol->decomp_scheme->dims[0].out_stride)
    {
        return fusable_dims;
    }

    // expected stride is the regular stride we obtain by n * stride of prev dim
    INTP expected_stride = sol->decomp_scheme->dims[0].n *
                           sol->decomp_scheme->dims[0].in_stride;
    for (INT32 i = 1; i < dim_rank; i++)
    {
        if (sol->decomp_scheme->dims[i].in_stride !=
            sol->decomp_scheme->dims[i].out_stride)
        {
            break;
        }

        INTP actual_stride = sol->decomp_scheme->dims[i].in_stride;
        // we can no longer club
        if (expected_stride != actual_stride)
        {
            break;
        }
        fusable_dims += 1;
        expected_stride = expected_stride * sol->decomp_scheme->dims[i].n;
    }

    return fusable_dims;
}

INT32 selector_ndim_dft(aoclfftz_selector_t *sel, kernel_t *kertab)
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

    INT32 dim_rank = sel->solution->decomp_scheme->dim_rank;
    INT32 stats_mode = sel->solution->decomp_scheme->cntrl_params->
                       measure_stats;
    INT32 ret = SELECTOR_FAILURE;

    n_minus1_sel = alloc_selector(1, dim_rank - 1, sel->scratch_space,
                                  sel->kernel_tables, 0 /*unused*/);
    outer_dim_sel = alloc_selector(dim_rank - 1, 1, sel->scratch_space,
                                   sel->kernel_tables, 0 /*unused*/);

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

    sel->cost_analysis->ops = n_minus1_sel->cost_analysis->ops +
                              outer_dim_sel->cost_analysis->ops;
    sel->cost_analysis->time = n_minus1_sel->cost_analysis->time +
                               outer_dim_sel->cost_analysis->time;

    if (stats_mode)
    {
        // capture stats
    }
    sel->solution->next_sol = alloc_sol_array(1 /*n_threads*/);
    sel->solution->next_sol[0] = outer_dim_sel->solution;
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
