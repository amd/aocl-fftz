// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file selector_bluestein_rdft.c
 *
 *  @brief Sets up the real (R2C/C2R) Bluestein solver and its complex child.
 *
 *  A real prime-length transform is solved by delegating to the complex
 *  Bluestein solver. The complex selector builds a size-n complex sub-problem,
 *  which is then linked as the child (next_sol) of the real Bluestein node.
 *  The real node only converts the data on entry and on exit (see
 *  core/solvers/real/bluestein_solver_rdft.c).
 *
 *  @author Jeevanantham N
 */

#include "selector/selector.h"
#include "core/common/memory_manager.h"
#include "core/common/bluestein_utils.h"
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

    FFTZ_INTP n = sel->solution->decomp_scheme->dims[0].n;
    FFTZ_INT32 ret = SELECTOR_FAILURE;

    // Build the size-n complex child before this node claims any scratch.
    // copy_solution_obj carries buffer pointers into the copies, and
    // destroy_bluestein relies on copied nodes holding NULL bluestein pointers
    // so that each buffer is freed exactly once. Leaving this node's pointers
    // NULL until the copies complete preserves that invariant.
    aoclfftz_selector_t *next_sel = alloc_selector(1, 1, sel->kernel_tables,
                                                   sel->has_nested);
    if (next_sel == NULL)
    {
        ret = AOCLFFTZ_MEMORY_FAILURE;
        goto exit_bluestein_rdft;
    }

    ret = copy_solution_obj(next_sel->solution, sel->solution);
    if (ret != AOCLFFTZ_SUCCESS)
    {
        goto exit_bluestein_rdft;
    }

    // Reconfigure the child as a complex, 1D, single-batch, unit-stride size-n
    // problem. Its I/O pointers are irrelevant, as the real node drives the
    // child through a per-call ctx at execute time.
    SET_COMPLEX(next_sel->solution->decomp_scheme->flags);
    // The child always receives distinct in and out scratch, regardless of
    // whether the enclosing real problem is in-place. Complex setup reads
    // that from the pointers rather than the flag, and an in-place parent
    // leaves the same address in both, so clear them here.
    SET_OUTOFPLACE(next_sel->solution->decomp_scheme->flags);
    next_sel->solution->decomp_scheme->in_real  = NULL;
    next_sel->solution->decomp_scheme->in_imag  = NULL;
    next_sel->solution->decomp_scheme->out_real = NULL;
    next_sel->solution->decomp_scheme->out_imag = NULL;
    // Clear the parent's strides brought in by the copy.
    next_sel->solution->decomp_scheme->dims[0].in_stride = 1;
    next_sel->solution->decomp_scheme->dims[0].out_stride = 1;
    next_sel->solution->decomp_scheme->vecs[0].n = 1;
    next_sel->solution->decomp_scheme->vecs[0].in_stride = 1;
    next_sel->solution->decomp_scheme->vecs[0].out_stride = 1;
    next_sel->solution->next_sol = NULL;

    // Invoke the complex selector to build the size-n complex subtree: a
    // Bluestein(n) node above the extended-length-m FFT it convolves with.
    ret = selector_model_dft_(next_sel);
    if (ret != SELECTOR_SUCCESS)
    {
        goto exit_bluestein_rdft;
    }

    // The copies are complete, so this node may now claim its scratch.
    ret = setup_real_bluestein_solver(sel->solution, n);
    if (ret != SOLVER_SUCCESS)
    {
        goto exit_bluestein_rdft;
    }

    // Bind the pair of kernels this node will use. The forward direction reads
    // reals and writes a spectrum; the backward direction does the inverse.
    kernel_tables_t *kt = sel->kernel_tables;
    aoclfftz_bluestein_t *bluestein = sel->solution->dft_bufs->bluestein;

    if (FFT_DIR(sel->solution->decomp_scheme->flags) == FORWARD_FFT_DIR)
    {
        bluestein->cast_to_complex = kt->type_convert_r2c;
        bluestein->cast_from_complex = kt->type_convert_c2hc;
    }
    else
    {
        bluestein->cast_to_complex = kt->type_convert_hc2c;
        bluestein->cast_from_complex = kt->type_convert_c2r;
    }

    sel->solution->next_sol = next_sel->solution;

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
