// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file selector_bluestein_rdft.c
 *
 *  @brief Sets up the real (R2C/C2R) Bluestein solver and its complex child.
 *
 *  A real prime-length transform is solved by delegating to the complex
 *  Bluestein solver. The complex selector builds a size-n complex sub-problem,
 *  which is then linked as the child (complex_sol) of the real Bluestein node.
 *  The real node only converts the data on entry and on exit (see
 *  core/solvers/real/bluestein_solver_rdft.c).
 *
 *  @author Jeevanantham N
 */

#include "selector/selector.h"
#include "core/common/memory_manager.h"
#include "core/solvers/solver.h"
#include "utils/utils.h"

FFTZ_INT32 selector_bluestein_rdft(aoclfftz_selector_t *sel, kernel_t *kertab,
                                   aoclfftz_realhelper_t *realhelper)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    if (sel == NULL || sel->solution == NULL ||
        sel->solution->decomp_scheme == NULL || sel->kernel_tables == NULL)
    {
        AOCLFFTZ_LOG(INFO, global_logger_mode,
                     "Invalid selector or solution passed to "
                     "selector_bluestein_rdft");
        return SELECTOR_FAILURE;
    }

    FFTZ_INT32 ret = SELECTOR_FAILURE;

    // Allocate the empty child up front. setup_real_bluestein_solver copies
    // this node into it and reshapes it into the size-n complex sub-problem,
    // then selector_model_dft_ builds the complex subtree beneath it. The copy
    // inside setup runs while this node's Bluestein buffer pointers are still
    // NULL, so the child never double-frees a shared buffer.
    aoclfftz_selector_t *next_sel = alloc_selector(1, 1, sel->kernel_tables,
                                                   sel->has_nested);
    if (next_sel == NULL)
    {
        ret = AOCLFFTZ_MEMORY_FAILURE;
        goto exit_bluestein_rdft;
    }

    // Reshape the complex child, claim scratch, bind the conversion kernels,
    // and (for a CT stage) build the strides and buffer roles this node takes
    // over from the Direct it replaces.
    ret = setup_real_bluestein_solver(sel->solution, next_sel->solution,
                                      sel->kernel_tables, realhelper);
    if (ret != SOLVER_SUCCESS)
    {
        goto exit_bluestein_rdft;
    }

    // Invoke the complex selector to build the size-n complex subtree: a
    // Bluestein(n) node above the extended-length-m FFT it convolves with.
    ret = selector_model_dft_(next_sel);
    if (ret != SELECTOR_SUCCESS)
    {
        goto exit_bluestein_rdft;
    }

    // The complex FFT(n) child always lives on complex_sol, never next_sol.
    // That keeps next_sol free for the chain: a forward head links the real
    // combine there, while a backward tail and a standalone prime have nothing
    // following them.
    sel->solution->dft_bufs->complex_sol = next_sel->solution;

    // Propagate the child's thread demand up to the real node.
    sel->solution->decomp_scheme->thread_info->avl_threads =
        next_sel->solution->decomp_scheme->thread_info->avl_threads;

    // Destroy only the selector wrapper, keeping its solution linked above.
    destroy_selector_without_solution(next_sel);
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SELECTOR_SUCCESS;

exit_bluestein_rdft:
    destroy_selector(next_sel);
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit with failure");
    return ret;
}
