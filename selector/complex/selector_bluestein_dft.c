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
#include "core/solvers/solver.h"
#include "utils/utils.h"

INT32 selector_bluestein_dft(aoclfftz_selector_t *sel, kernel_t *kertab)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    if (sel == NULL || sel->solution == NULL ||
        sel->solution->decomp_scheme == NULL || sel->kernel_tables == NULL ||
        sel->kernel_tables->ele_mul[FORWARD_FFT_DIR] == NULL ||
        sel->kernel_tables->ele_mul[BACKWARD_FFT_DIR] == NULL ||
        sel->kernel_tables->normalize == NULL)
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
    next_sel = alloc_selector(vec_rank, dim_rank, sel->kernel_tables);
    if (next_sel == NULL)
    {
        ret = AOCLFFTZ_MEMORY_FAILURE;
        goto exit_bluestein_dft;
    }

    // Allocate in, out buffers for next sol and Bluestein chirp buffers for
    // cur sol. The MT and ST variants each carry their own setup function
    // (mirroring the ST/MT pairing of other solver families in the library).
#ifdef MULTI_THREADING
    if (sel->solution->solver->solver_type == SOLVER_MT_BLUESTEIN)
    {
        ret = setup_mt_bluestein_solver(sel->solution, next_sel->solution, m);
    }
    else
#endif
    {
        ret = setup_bluestein_solver(sel->solution, next_sel->solution, m);
    }
    if (ret != SELECTOR_SUCCESS)
    {
        goto exit_bluestein_dft;
    }

    // Bind the elementwise multiplication and normalization kernels registered
    // for the plan onto this Bluestein solution
    sel->solution->dft_bufs->bluestein->ele_mul[FORWARD_FFT_DIR] =
        sel->kernel_tables->ele_mul[FORWARD_FFT_DIR];
    sel->solution->dft_bufs->bluestein->ele_mul[BACKWARD_FFT_DIR] =
        sel->kernel_tables->ele_mul[BACKWARD_FFT_DIR];
    sel->solution->dft_bufs->bluestein->normalize =
        sel->kernel_tables->normalize;

    // Initialize Bluestein sequence B
    ret = compute_chirp_sequence(sel->solution, m);
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
    destroy_selector(next_sel);
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit with failure");

    return ret;
}
