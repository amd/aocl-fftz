// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

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

FFTZ_INT32 selector_ndim_rdft(aoclfftz_selector_t *sel, kernel_t *kertab,
                         aoclfftz_realhelper_t *realhelper)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    if (sel == NULL || sel->solution == NULL ||
        sel->solution->decomp_scheme == NULL)
    {
        AOCLFFTZ_LOG(INFO, global_logger_mode,
                     "Invalid selector or solution passed to "
                     "selector_ndim_rdft");
        return SELECTOR_FAILURE;
    }

    aoclfftz_selector_t *complex_dims_sol = NULL;
    aoclfftz_selector_t *real_dim_sol = NULL;

    FFTZ_INT32 dim_rank = sel->solution->decomp_scheme->dim_rank;
    FFTZ_INT32 stats_mode =
        sel->solution->decomp_scheme->cntrl_params->measure_stats;
    FFTZ_INT32 ret = SELECTOR_FAILURE;

    real_dim_sol = alloc_selector(dim_rank - 1, 1, sel->kernel_tables,
                                  sel->has_nested);

    complex_dims_sol = alloc_selector(1, dim_rank - 1, sel->kernel_tables,
                                      sel->has_nested);

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

    // Propagate the (N-1)D complex child's ct_buf_size up so a parent MT_BATCHED
    // can stride ct_offset by it (tid * ct_buf_size) and give each concurrent
    // NDIM complex child its own non-overlapping ct pool slice.
    sel->solution->dft_bufs->ct_buf_size =
        complex_dims_sol->solution->dft_bufs->ct_buf_size;

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
    if (sel->solution->next_sol == NULL)
    {
        ret = AOCLFFTZ_MEMORY_FAILURE;
        AOCLFFTZ_ERROR("alloc_sol_array failed: %s", get_status_string(ret));
        goto exit_nd_dft;
    }
    sel->solution->next_sol[0] = real_dim_sol->solution;
    sel->solution->dft_bufs->nd_sol = complex_dims_sol->solution;

    destroy_selector_without_solution(real_dim_sol);
    destroy_selector_without_solution(complex_dims_sol);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SELECTOR_SUCCESS;

exit_nd_dft:
    destroy_selector(real_dim_sol);
    destroy_selector(complex_dims_sol);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return ret;
}
