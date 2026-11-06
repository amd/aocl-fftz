// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file selector_packed_rdft.c
 *
 *  @brief Wrapper that invokes the real packed solver as guided by the
 * selector.
 *
 *  This file contains the implementation of functions that are used to
 *  setup child complex FFT and the required separate/recombine kernels.
 *  Rank 1 uses the packed 1D recombine/separate kernel. Rank 3 uses the
 *  dedicated 3D grid kernel over a child CFFT of size M x N1 x N2.
 *
 *  @author Srirammaswamy Srinivasan
 */

#include "core/common/memory_manager.h"
#include "selector/selector.h"
#include "utils/utils.h"

FFTZ_INT32 selector_real_packed_rdft(aoclfftz_selector_t *sel, kernel_t *kertab,
                                     aoclfftz_realhelper_t *realhelper)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    if (sel == NULL || sel->solution == NULL ||
        sel->solution->decomp_scheme == NULL)
    {
        AOCLFFTZ_LOG(INFO, global_logger_mode,
                     "Invalid selector or solution passed to "
                     "selector_real_packed_rdft");
        return SELECTOR_FAILURE;
    }

    aoclfftz_selector_t *child_sel = NULL;
    FFTZ_INT32 ret = SELECTOR_FAILURE;

    aoclfftz_decomp_scheme_t *parent_decomp = sel->solution->decomp_scheme;
    FFTZ_INTP n0_by2 = parent_decomp->dims[0].n / 2; // child complex FFT length

    // Child complex selector keeps the parent rank (1 or 3), single vector.
    child_sel = alloc_selector(1, parent_decomp->dim_rank, sel->kernel_tables,
                               sel->has_nested);
    if (child_sel == NULL)
    {
        ret = AOCLFFTZ_MEMORY_FAILURE;
        goto exit_real_packed;
    }

    ret = setup_real_packed_solver(sel->solution, child_sel->solution,
                                   sel->kernel_tables);
    if (ret != SOLVER_SUCCESS)
    {
        goto exit_real_packed;
    }

    // Model the child complex plan with the complex selector model initialized
    // by the root real selector before entering this packed path.
    ret = selector_model_dft_(child_sel);
    if (ret != SELECTOR_SUCCESS)
    {
        goto exit_real_packed;
    }

    // Child CFFT cost plus one packed recombine/separate pass over every row.
    FFTZ_INT64 packed_cost = n0_by2;
    for (FFTZ_INT32 dim = 1; dim < parent_decomp->dim_rank; dim++)
    {
        packed_cost *= parent_decomp->dims[dim].n;
    }
    sel->cost_analysis->ops = child_sel->cost_analysis->ops + packed_cost;
    sel->cost_analysis->time = child_sel->cost_analysis->time + packed_cost;

    // Propagate the complex child's CT slice size so an enclosing MT batch can
    // advance ct_offset before executing this packed node concurrently.
    sel->solution->dft_bufs->ct_buf_size =
        child_sel->solution->dft_bufs->ct_buf_size;

    // Packed node owns the complex child via next_sol. Iterative real CT/Direct
    // swap skips this node.
    sel->solution->next_sol = child_sel->solution;

    // Ownership of child_sel->solution transferred to sel->solution->next_sol.
    destroy_selector_without_solution(child_sel);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SELECTOR_SUCCESS;

exit_real_packed:
    destroy_selector(child_sel);
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return ret;
}
