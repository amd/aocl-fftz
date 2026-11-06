// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file selector_buffered_rdft.c
 *
 *  @brief Wrapper that acts on the buffered solver as guided by the selector.
 *
 *  This file contains the implementation of functions that are used to
 *  setup, factorize and evaluate sub-problems and kernels as applicable.
 *
 *  @author Srirammaswamy Srinivsan
 */

#include "api/aoclfftz_internal.h"
#include "selector/selector.h"
#include "core/common/memory_manager.h"
#include "utils/utils.h"

FFTZ_INT32 selector_buffered_rdft(aoclfftz_selector_t *sel, kernel_t *kertab,
                             aoclfftz_realhelper_t *realhelper)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    if (sel == NULL || sel->solution == NULL ||
        sel->solution->decomp_scheme == NULL)
    {
        AOCLFFTZ_LOG(INFO, global_logger_mode,
                     "Invalid selector or solution passed to "
                     "selector_buffered_rdft");
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
        goto exit_batched_dft;
    }

    // copy solution object from sel to cur_sel
    ret = copy_solution_obj(cur_sel->solution, sel->solution);
    if (ret != AOCLFFTZ_SUCCESS)
    {
        AOCLFFTZ_ERROR("copy_solution_obj failed: %s", get_status_string(ret));
        goto exit_batched_dft;
    }

    // Setup buffered solver
    ret = setup_real_buffered_solver(sel->solution, realhelper);
    if (ret != SELECTOR_SUCCESS)
    {
        goto exit_batched_dft;
    }

    // Copy the auxiliary buffers (address) from sel to cur_sel (i.e. next_sol)
    cur_sel->solution->dft_bufs->buffered->aux_buffer_1 =
        sel->solution->dft_bufs->buffered->aux_buffer_1;
    cur_sel->solution->dft_bufs->buffered->aux_buffer_2 =
        sel->solution->dft_bufs->buffered->aux_buffer_2;
    cur_sel->solution->dft_bufs->buffered->is_aux_buffer_allocated = 0;
    cur_sel->solution->dft_bufs->buffered->aux_buf_size_per_thread =
        sel->solution->dft_bufs->buffered->aux_buf_size_per_thread;

    // Call selector for solving it as a non-buffered problem
    ret = selector_model_rdft_(cur_sel, realhelper);
    if (ret != SELECTOR_SUCCESS)
    {
        goto exit_batched_dft;
    }

    sel->solution->next_sol = cur_sel->solution;

    // destroy only the selector not the solution within it
    destroy_selector_without_solution(cur_sel);
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SELECTOR_SUCCESS;

exit_batched_dft:
    destroy_selector(cur_sel);
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return ret;
}
