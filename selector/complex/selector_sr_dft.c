// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file selector_sr_dft.c
 *
 *  @brief Wrapper that invokes the SR solver as guided by the selector.
 *
 * This file contains the implementation of functions that are used to
 *  setup, split and evaluate the sub-problems of Split Radix as applicable.
 *
 *  @author Varaprasad, Malothu
 */

#include "selector/selector.h"
#include "core/common/memory_manager.h"
#include "utils/utils.h"
#include "core/common/twiddle.h"

FFTZ_INT32 selector_sr_dft(aoclfftz_selector_t *sel, kernel_t *kertab)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    if (sel == NULL || sel->solution == NULL ||
        sel->solution->decomp_scheme == NULL)
    {
        AOCLFFTZ_LOG(INFO, global_logger_mode,
                     "Invalid selector or solution passed to selector_sr_dft");
        return SELECTOR_FAILURE;
    }

    aoclfftz_selector_t *cur_sel_even = NULL;
    aoclfftz_selector_t *cur_sel_odd1 = NULL;
    aoclfftz_selector_t *cur_sel_odd3 = NULL;

    aoclfftz_solution_t *org_sol = NULL;

    FFTZ_INTP n = sel->solution->decomp_scheme->dims[0].n;
    FFTZ_INT32 vec_rank = sel->solution->decomp_scheme->vec_rank;
    FFTZ_INT32 dim_rank = sel->solution->decomp_scheme->dim_rank;
    FFTZ_INT32 ret = SELECTOR_FAILURE;

    FFTZ_INTP n_even = n / 2;
    FFTZ_INTP n_odd = n / 4;

    org_sol = alloc_solution(vec_rank, dim_rank);
    if (org_sol == NULL)
    {
        AOCLFFTZ_LOG(INFO, global_logger_mode,
                     "Failed to allocate org_sol for N=%td", n);
        ret = AOCLFFTZ_MEMORY_FAILURE;
        goto exit_sr_dft;
    }

    /* Allocate 3 sub-selectors for even, odd1, and odd3 parts */
    /* nthreads is set to 0 for all recursive SR sub-problems. */
    cur_sel_even = alloc_selector(vec_rank, dim_rank, sel->kernel_tables);
    cur_sel_odd1 = alloc_selector(vec_rank, dim_rank, sel->kernel_tables);
    cur_sel_odd3 = alloc_selector(vec_rank, dim_rank, sel->kernel_tables);

    if (cur_sel_even == NULL || cur_sel_odd1 == NULL || cur_sel_odd3 == NULL)
    {
        ret = AOCLFFTZ_MEMORY_FAILURE;
        goto exit_sr_dft;
    }

    ret = copy_solution_obj(org_sol, sel->solution);
    if (ret != AOCLFFTZ_SUCCESS)
    {
        AOCLFFTZ_ERROR("copy_solution_obj failed: %s", get_status_string(ret));
        goto exit_sr_dft;
    }
    org_sol->next_sol = NULL;

    /* Setup SR solver using sub-selector solutions */
    ret = setup_sr_solver(org_sol, cur_sel_even->solution,
                          cur_sel_odd1->solution, cur_sel_odd3->solution,
                          n_even, n_odd);
    if (ret != SELECTOR_SUCCESS)
    {
        goto exit_sr_dft;
    }

    /* Allocate input copy buffer for in-place transforms */
    if (!IS_OUT_OF_PLACE(org_sol->decomp_scheme->flags))
    {
        FFTZ_INTP in_stride = org_sol->decomp_scheme->dims[0].in_stride;
        FFTZ_UINT32 precision = DT_PRECISION_FLAG(
            org_sol->decomp_scheme->flags);
        FFTZ_UINT32 dt_bytes = DT_PRECISION_BYTES(precision);
        /* Complex: 2 * sizeof(type) */
        FFTZ_INTP elem_size = DATA_STRIDE * dt_bytes;

        FFTZ_INTP sr_input_copy_size = strided_buffer_size(n, in_stride,
                                                           elem_size);
        sel->solution->dft_bufs->sr->input_copy_size = sr_input_copy_size;

        ALLOC_ALIGN_UNINIT(sel->solution->dft_bufs->sr->input_copy, FFTZ_VOID,
                           sr_input_copy_size);
        if (sel->solution->dft_bufs->sr->input_copy == NULL)
        {
            ret = AOCLFFTZ_MEMORY_FAILURE;
            goto exit_sr_dft;
        }

        AOCLFFTZ_LOG(DEBUG, global_logger_mode,
                     "Allocated SR input copy buffer for N=%td (%td bytes)",
                     n, sr_input_copy_size);
    }

    /* Mark sub-problems as out-of-place so child SR levels
     * don't allocate their own input_copy buffers. */
    SET_OUTOFPLACE(cur_sel_even->solution->decomp_scheme->flags);
    SET_OUTOFPLACE(cur_sel_odd1->solution->decomp_scheme->flags);
    SET_OUTOFPLACE(cur_sel_odd3->solution->decomp_scheme->flags);

    /* Recursively solve each sub-problem */
    ret = selector_model_dft_(cur_sel_even);
    if (ret != SELECTOR_SUCCESS)
    {
        goto exit_sr_dft;
    }

    ret = selector_model_dft_(cur_sel_odd1);
    if (ret != SELECTOR_SUCCESS)
    {
        goto exit_sr_dft;
    }

    ret = selector_model_dft_(cur_sel_odd3);
    if (ret != SELECTOR_SUCCESS)
    {
        goto exit_sr_dft;
    }

    /* Link sub-solutions to parent */
    sel->solution->next_sol = alloc_sol_array(1);
    if (!sel->solution->next_sol)
    {
        ret = SELECTOR_FAILURE;
        goto exit_sr_dft;
    }

    sel->solution->next_sol[0] = cur_sel_even->solution;
    sel->solution->dft_bufs->sr->odd1_sol = cur_sel_odd1->solution;
    sel->solution->dft_bufs->sr->odd3_sol = cur_sel_odd3->solution;

    /*
     * Destroy selector wrappers but keep sub-solutions
     * and owned by parent
     */
    destroy_selector_without_solution(cur_sel_even);
    destroy_selector_without_solution(cur_sel_odd1);
    destroy_selector_without_solution(cur_sel_odd3);
    cur_sel_even = NULL;
    cur_sel_odd1 = NULL;
    cur_sel_odd3 = NULL;
    destroy_solution(org_sol);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SELECTOR_SUCCESS;

exit_sr_dft:
    /* Clean up SR input copy buffer if allocated */
    if (sel->solution->dft_bufs)
    {
        FREE_ALIGN_ALLOCATED_MEM(sel->solution->dft_bufs->sr->input_copy);
        sel->solution->dft_bufs->sr->input_copy = NULL;
        sel->solution->dft_bufs->sr->input_copy_size = 0;
    }

    destroy_selector(cur_sel_even);
    destroy_selector(cur_sel_odd1);
    destroy_selector(cur_sel_odd3);
    destroy_solution(org_sol);

    return ret;
}
