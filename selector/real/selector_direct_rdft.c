// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file selector_direct_rdft.c
 *
 *  @brief Wrapper that acts on the real direct solver as guided by the
 *         real selector.
 *
 *  This file contains the implementation of functions that are used to
 *  setup and evaluate the kernels as applicable.
 *
 *  @author Srirammaswamy Srinivasan
 *  @author Ashwin K. Godbole
 */

#include "selector/selector.h"
#include "core/common/memory_manager.h"
#include "core/solvers/real/direct_solver_rdft_utils.h"
#include "utils/utils.h"

// Picks the ST or MT direct solver variant. MT is chosen only when multiple
// threads are available and enough parallel work exists. This only records the
// decision (solver_type + kernel counts); binding the execute function pointer
// is the caller's/parent's responsibility once this selector returns.
static FFTZ_INT32 select_real_direct_solver_type(aoclfftz_solution_t *solution,
                                                 aoclfftz_realhelper_t *realhelper,
                                                 FFTZ_INT32 *num_threads)
{
    aoclfftz_generic_solver_t *solver = solution->solver;

    // Populate kernel counts first, since the parallel-width check reads them.
    set_kernel_count_in_each_group(solution, realhelper);

#ifdef MULTI_THREADING
    // Only one of r2hc/r2hcf carries the groups, and both share the same set
    // width, so the group total covers either case.
    FFTZ_INTP num_groups = NUM_RFFT_GROUPS(solver);
    FFTZ_INTP r2c_iters = num_groups / solver->kernel_r2hc->sets;

    FFTZ_INTP c2c_iters = 0;
    if (solver->kernel_c2c->count != 0)
    {
        FFTZ_INTP num_c2c_per_group = solver->kernel_c2c->count / num_groups;
        c2c_iters = (num_c2c_per_group >= num_groups) ? num_groups
                                                      : num_c2c_per_group;
    }

    FFTZ_INTP max_parallel_iters =
        (r2c_iters > c2c_iters) ? r2c_iters : c2c_iters;

    if (*num_threads > 1 && max_parallel_iters > 1)
    {
        solver->solver_type =
            select_real_mt_direct_solver_type(solution, realhelper->is_CT);
    }
    else
#endif
    {
        *num_threads = 1;
        solver->solver_type =
            select_real_st_direct_solver_type(solution, realhelper->is_CT);
    }

    return SOLVER_SUCCESS;
}

FFTZ_INT32 selector_direct_rdft(aoclfftz_selector_t *sel, kernel_t *kertab,
                           aoclfftz_realhelper_t *realhelper)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    if (sel == NULL || sel->solution == NULL ||
        sel->solution->decomp_scheme == NULL)
    {
        AOCLFFTZ_LOG(INFO, global_logger_mode,
                     "Invalid selector or solution passed to "
                     "selector_direct_rdft");
        return SELECTOR_FAILURE;
    }

    aoclfftz_decomp_scheme_t *decomp_scheme = sel->solution->decomp_scheme;

    aoclfftz_selector_t *cur_sel = NULL;
    FFTZ_INTP n = decomp_scheme->dims[0].n;
    FFTZ_INTP n_batch = decomp_scheme->vecs[0].n;
    FFTZ_INT32 vec_rank = decomp_scheme->vec_rank;
    FFTZ_INT32 dim_rank = decomp_scheme->dim_rank;
    FFTZ_INT32 stats_mode = decomp_scheme->cntrl_params->measure_stats;
    FFTZ_INT32 avl_threads = decomp_scheme->thread_info->avl_threads;
    FFTZ_UINT32 precision = DT_PRECISION_FLAG(decomp_scheme->flags);
    FFTZ_UINT32 selector_mode = GET_SELECTOR_MODE(decomp_scheme->flags);
    FFTZ_UINT8 dir = FFT_DIR(decomp_scheme->flags);
    FFTZ_INT32 ret = SELECTOR_FAILURE;
    FFTZ_INT32 setup_ret = AOCLFFTZ_SETUP_FAILURE;

    kernel_t *kernel_c2c = NULL;
    kernel_t *kernel_r2hc = NULL;
    kernel_t *kernel_r2hcf = NULL;

    cur_sel = alloc_selector(vec_rank, dim_rank, sel->kernel_tables,
                             sel->has_nested);
    if (cur_sel == NULL)
    {
        ret = AOCLFFTZ_MEMORY_FAILURE;
        goto exit_direct_rdft;
    }

    FFTZ_INT32 num_threads = (n_batch < avl_threads) ? n_batch : avl_threads;
    decomp_scheme->thread_info->n_threads = num_threads;

    // copy solution object from sel to cur_sel
    ret = copy_solution_obj(cur_sel->solution, sel->solution);
    if (ret != AOCLFFTZ_SUCCESS)
    {
        AOCLFFTZ_ERROR("copy_solution_obj failed: %s", get_status_string(ret));
        goto exit_direct_rdft;
    }

    // find a suitable kernel within the list of C kernels, and if one is found,
    // check for the existance of other implementations for the same radix
    for (FFTZ_INTP i = 0; i < NUM_KERNELS_IN_EACH_CATEGORY; i++)
    {
        FFTZ_UINT32 radix = kertab[i].radix;

        if (radix == 0) // End of search for suitable kernels in the list
        {
            break;
        }

        if ((FFTZ_INTP)radix == n)
        {
            AOCLFFTZ_LOG(TRACE, global_logger_mode,
                                   "Evaluating Radix-%td kernel", n);

            for (FFTZ_INTP kcat = 0; kcat < NUM_KERNEL_CATEGORIES; kcat++)
            {
                FFTZ_INTP kloc = (kcat * NUM_KERNELS_IN_EACH_CATEGORY) + i;

                FFTZ_INTP kloc_r2hc =
                    0 * NUM_KERNELS_IN_EACH_DFT_VARIANT + kloc;
                FFTZ_INTP kloc_r2hcf =
                    1 * NUM_KERNELS_IN_EACH_DFT_VARIANT + kloc;
                FFTZ_INTP kloc_c2c = 2 * NUM_KERNELS_IN_EACH_DFT_VARIANT + kloc;

                if (kertab[kloc_r2hc].kfft[dir] == NULL ||
                    kertab[kloc_r2hcf].kfft[dir] == NULL ||
                    kertab[kloc_c2c].kfft[dir] == NULL)
                {
                    continue;
                }

                kernel_c2c = &kertab[kloc_c2c];
                kernel_r2hc = &kertab[kloc_r2hc];
                kernel_r2hcf = &kertab[kloc_r2hcf];

                cur_sel->solution->solver->kernel_c2c->kfft[FORWARD_FFT_DIR] =
                    kernel_c2c->kfft[FORWARD_FFT_DIR];
                cur_sel->solution->solver->kernel_c2c->kfft[BACKWARD_FFT_DIR] =
                    kernel_c2c->kfft[BACKWARD_FFT_DIR];
                cur_sel->solution->solver->kernel_r2hc->kfft[FORWARD_FFT_DIR] =
                    kernel_r2hc->kfft[FORWARD_FFT_DIR];
                cur_sel->solution->solver->kernel_r2hc->kfft[BACKWARD_FFT_DIR] =
                    kernel_r2hc->kfft[BACKWARD_FFT_DIR];
                cur_sel->solution->solver->kernel_r2hcf->kfft[FORWARD_FFT_DIR] =
                    kernel_r2hcf->kfft[FORWARD_FFT_DIR];
                cur_sel->solution->solver->kernel_r2hcf
                    ->kfft[BACKWARD_FFT_DIR] =
                    kernel_r2hcf->kfft[BACKWARD_FFT_DIR];

                cur_sel->solution->solver->kernel_c2c->sets =
                    kernel_c2c->sets[precision - 2];
                cur_sel->solution->solver->kernel_r2hc->sets =
                    kernel_r2hc->sets[precision - 2];
                cur_sel->solution->solver->kernel_r2hcf->sets =
                    kernel_r2hcf->sets[precision - 2];

                // Choose the direct solver variant and thread count for this
                // problem. The execute function pointer is bound by the parent
                // (selector level) after this selector returns.
                setup_ret = select_real_direct_solver_type(
                    cur_sel->solution, realhelper, &num_threads);
                if (setup_ret != SOLVER_SUCCESS)
                {
                    AOCLFFTZ_ERROR("Direct solver variant selection failed for "
                                   "solver_type %d",
                                   cur_sel->solution->solver->solver_type);
                    ret = SELECTOR_FAILURE;
                    goto exit_direct_rdft;
                }

#ifdef MULTI_THREADING
                if (num_threads > 1)
                {
                    setup_ret = setup_real_mt_direct_solver(
                        cur_sel->solution, cur_sel->cost_analysis, kernel_c2c,
                        kernel_r2hc, kernel_r2hcf, realhelper, sel->has_nested);
                    if (setup_ret != SOLVER_SUCCESS)
                    {
                        AOCLFFTZ_ERROR("setup_real_mt_direct_solver failed: %d",
                                       setup_ret);
                        ret = SELECTOR_FAILURE;
                        goto exit_direct_rdft;
                    }
                }
                else
#endif
                {
                    setup_ret = setup_real_direct_solver(
                        cur_sel->solution, cur_sel->cost_analysis, kernel_c2c,
                        kernel_r2hc, kernel_r2hcf, realhelper);
                    if (setup_ret != SOLVER_SUCCESS)
                    {
                        AOCLFFTZ_ERROR("setup_real_direct_solver failed: %d",
                                       setup_ret);
                        ret = SELECTOR_FAILURE;
                        goto exit_direct_rdft;
                    }
                }

                if (selector_mode == AOCLFFTZ_FIXED_SELECTOR)
                {
                    if (!sel->cost_analysis->ops
                        || (cur_sel->cost_analysis->ops
                            < sel->cost_analysis->ops))
                    {
                        sel->cost_analysis->ops =
                            cur_sel->cost_analysis->ops;
                        sel->cost_analysis->time =
                            cur_sel->cost_analysis->time;
                        // copy solution object from cur_sel to sel
                        setup_ret = copy_solution_obj(
                            sel->solution, cur_sel->solution);
                        if (setup_ret != AOCLFFTZ_SUCCESS)
                        {
                            AOCLFFTZ_ERROR("copy_solution_obj failed: %s",
                                            get_status_string(setup_ret));
                            ret = SELECTOR_FAILURE;
                            goto exit_direct_rdft;
                        }
                        setup_ret = copy_strides(
                            sel->solution, cur_sel->solution);
                        if (setup_ret != AOCLFFTZ_SUCCESS)
                        {
                            AOCLFFTZ_ERROR("copy_strides failed: %s",
                                            get_status_string(setup_ret));
                            ret = SELECTOR_FAILURE;
                            goto exit_direct_rdft;
                        }
                        ret = SELECTOR_SUCCESS;
                    }
                }
#ifdef AOCLFFTZ_AUTO_SELECTOR_MODE
                else
                {
                    if (sel->cost_analysis->time == 0 ||
                        cur_sel->cost_analysis->time < sel->cost_analysis->time)
                    {
                        sel->cost_analysis->ops =
                            cur_sel->cost_analysis->ops;
                        sel->cost_analysis->time =
                            cur_sel->cost_analysis->time;
                        // copy solution object from cur_sel to sel
                        setup_ret = copy_solution_obj(
                            sel->solution, cur_sel->solution);
                        if (setup_ret != AOCLFFTZ_SUCCESS)
                        {
                            AOCLFFTZ_ERROR("copy_solution_obj failed: %s",
                                            get_status_string(setup_ret));
                            ret = SELECTOR_FAILURE;
                            goto exit_direct_rdft;
                        }
                        setup_ret = copy_strides(
                            sel->solution, cur_sel->solution);
                        if (setup_ret != AOCLFFTZ_SUCCESS)
                        {
                            AOCLFFTZ_ERROR("copy_strides failed: %s",
                                            get_status_string(setup_ret));
                            ret = SELECTOR_FAILURE;
                            goto exit_direct_rdft;
                        }
                        ret = SELECTOR_SUCCESS;
                    }
                }
#endif // AOCLFFTZ_AUTO_SELECTOR_MODE
                if (stats_mode)
                {
                    // capture stats
                }
            } // End of FOR loop
            break;
        } // if (radix == n)
    } // End of FOR loop

    if (ret != SELECTOR_SUCCESS)
    {
        AOCLFFTZ_ERROR("No suitable kernel found");
    }

exit_direct_rdft:
    destroy_selector(cur_sel);
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return ret;
}
