// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file selector_bluestein_dft.c
 *
 *  @brief Wrapper that acts on the bluestein solver as guided by the selector.
 *
 *  This file contains the implementation of functions that are used to
 *  setup, factorize and evaluate sub-problems and kernels as applicable.
 *
 *  @author S. Biplab Raut
 *  @author Srirammaswamy Srinivasan
 */

#include "selector/selector.h"
#include "core/common/memory_manager.h"
#include "core/common/bluestein_utils.h"
#include "utils/utils.h"

INT32 selector_bluestein_dft(aoclfftz_selector_t *sel, kernel_t *kertab)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    if (sel == NULL || sel->solution == NULL ||
        sel->solution->decomp_scheme == NULL)
    {
        AOCLFFTZ_LOG(INFO, global_logger_mode,
                     "Invalid selector or solution passed to "
                     "selector_bluestein_dft");
        return SELECTOR_FAILURE;
    }

    INT32 vec_rank = sel->solution->decomp_scheme->vec_rank;
    INT32 dim_rank = sel->solution->decomp_scheme->dim_rank;
    INTP n = sel->solution->decomp_scheme->dims[0].n;
    INT32 ret = SELECTOR_FAILURE;

    // Get the extended length
    INTP m = get_extended_length(n);
    AOCLFFTZ_LOG(INFO, global_logger_mode,
                           "Problem length %td, extended Bluestein length %td",
                           n, m);


    // To hold the selector to perform FFT with extended length m
    aoclfftz_selector_t *next_sel = NULL;
    next_sel = alloc_selector(vec_rank, dim_rank, sel->scratch_space,
                              sel->kernel_tables, 0 /*unused*/);
    if (next_sel == NULL)
    {
        ret = AOCLFFTZ_MEMORY_FAILURE;
        goto exit_bluestein_dft;
    }

    // Allocate in, out buffers for next sol and
    // Bluestein sequence B buffers for cur sol
    ret = setup_bluestein_solver(sel->solution, next_sel->solution, m);
    if (ret != SELECTOR_SUCCESS)
    {
        goto exit_bluestein_dft;
    }

    // Initialize Bluestein sequence B
    ret = prepare_bluestein_sequence(sel->solution, m);
    if (ret != BLUESTEIN_SUCCESS)
    {
        goto exit_bluestein_dft;
    }

    // Invoke CT/direct selectors of extended length `m`
    ret = selector_model_dft_(next_sel);
    if (ret != SELECTOR_SUCCESS)
    {
        goto exit_bluestein_dft;
    }

    sel->solution->next_sol = alloc_sol_array(1 /*n_threads*/);
    sel->solution->next_sol[0] = next_sel->solution;

    // destroy only the selector not the solution within it
    destroy_selector_without_solution(next_sel);
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return SELECTOR_SUCCESS;

exit_bluestein_dft:
    destroy_selector_without_scratch_space(next_sel);
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit with failure");

    return ret;
}
