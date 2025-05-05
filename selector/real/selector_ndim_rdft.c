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

/** @file selector_ndim_rdft.c
 *
 *  @brief Wrapper that invokes the real ND solver as guided by the selector.
 *
 *  This file contains the implementation of functions that are used to
 *  setup, split and evaluate the sub-problems of real ND as applicable.
 *
 *  @author Srirammaswamy Srinivasan
 *  @author Prasandh Sankarankutty
 *  @author Jeevanantham N
 */

#include "selector/selector.h"
#include "core/common/memory_manager.h"
#include "utils/utils.h"

INT32 selector_ndim_rdft(aoclfftz_selector_t *sel, kernel_t *kertab,
                         aoclfftz_realhelper_t *realhelper)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    aoclfftz_selector_t *complex_dims_sol = NULL;
    aoclfftz_selector_t *real_dim_sol = NULL;

    INT32 dim_rank = sel->solution->decomp_scheme->dim_rank;
    INT32 stats_mode =
        sel->solution->decomp_scheme->cntrl_params->measure_stats;
    INT32 ret = SELECTOR_FAILURE;

    real_dim_sol = alloc_selector(dim_rank - 1, 1, sel->scratch_space,
                                  sel->kernel_tables, 0 /*unused*/);

    complex_dims_sol = alloc_selector(1, dim_rank - 1, sel->scratch_space,
                                      sel->kernel_tables, 0 /*unused*/);

    if (complex_dims_sol == NULL || real_dim_sol == NULL)
    {
        goto exit_nd_dft;
    }

    ret = setup_real_ndim_solver(sel->solution, real_dim_sol->solution,
                                 complex_dims_sol->solution, realhelper);
    if (ret != SELECTOR_SUCCESS)
    {
        goto exit_nd_dft;
    }

    // Invoke selector for solving 1D real sub-problem
    ret = selector_model_rdft_(real_dim_sol, realhelper);
    if (ret != SELECTOR_SUCCESS)
    {
        goto exit_nd_dft;
    }

    // Invoke selector for solving (N-1)D complex sub-problem
    ret = selector_model_dft_(complex_dims_sol);
    if (ret != SELECTOR_SUCCESS)
    {
        goto exit_nd_dft;
    }

    // Setup twiddle factors for the complex dimensions sub-problem
    setup_twiddle_buffer_complex(complex_dims_sol->solution);

    sel->cost_analysis->ops =
        complex_dims_sol->cost_analysis->ops + real_dim_sol->cost_analysis->ops;
    sel->cost_analysis->time = complex_dims_sol->cost_analysis->time +
                               real_dim_sol->cost_analysis->time;

    if (stats_mode)
    {
        // capture stats
    }
    sel->solution->next_sol = alloc_sol_array(1 /*n_threads*/);
    sel->solution->next_sol[0] = real_dim_sol->solution;
    sel->solution->dft_bufs->nd_sol = complex_dims_sol->solution;

    destroy_selector_without_solution(real_dim_sol);
    destroy_selector_without_solution(complex_dims_sol);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SELECTOR_SUCCESS;

exit_nd_dft:
    destroy_selector_without_scratch_space(real_dim_sol);
    destroy_selector_without_scratch_space(complex_dims_sol);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return ret;
}
