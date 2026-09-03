// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file selector_direct_dft.c
 *
 *  @brief Wrapper that acts on the direct solver as guided by the selector.
 *
 *  This file contains the implementation of functions that are used to
 *  setup and evaluate the kernels as applicable.
 *
 *  @author S. Biplab Raut
 *  @author Ashwin K. Godbole
 */

#include "selector/selector.h"
#include "core/common/memory_manager.h"
#include "utils/utils.h"

// Resolve the execution strategy for a direct FFT: which solver variant
// to use and how many threads to give it, based on the problem's
// batch layout and its available parallelism. This only records the
// decision (solver_type + n_threads); binding the execute function
// pointer is the caller's/parent's responsibility (see set_solver_fp
// at the selector level once this selector returns).
static FFTZ_INT32 select_direct_solver_type(aoclfftz_solution_t *solution,
                                            kernel_t *kertab,
                                            FFTZ_INT32 avl_threads)
{
    aoclfftz_generic_solver_t *solver = solution->solver;
    aoclfftz_decomp_scheme_t *decomp_scheme = solution->decomp_scheme;
    FFTZ_INTP batch = decomp_scheme->vecs[0].n;
    FFTZ_INT32 sets = solver->kernel_c2c->sets;
    FFTZ_INT32 n_threads = 1;
#ifndef MULTI_THREADING
    (void)kertab;
#endif

    if (decomp_scheme->batched_vecs == NULL)
    {
#ifdef MULTI_THREADING
        // iteration count that gets divided among threads
        // (upper bound on useful threads).
        FFTZ_INTP max_useful_threads = batch / sets;

        n_threads = max_useful_threads < avl_threads ?
                    max_useful_threads : avl_threads;
        if (n_threads > 1)
        {
            solver->solver_type = SOLVER_MT_DIRECT;
        }
        else
#endif
        {
            n_threads = 1;
            solver->solver_type = SOLVER_DIRECT;
        }
    }
    else
    {
        // batch iteration count that gets divided among threads
        // (upper bound on useful threads for row-major layout).
        FFTZ_INTP max_useful_threads = decomp_scheme->batched_vecs[0].n / sets;
        if(decomp_scheme->batched_vecs[0].n > batch)
        {
            batch = decomp_scheme->batched_vecs[0].n;
        }

        n_threads = max_useful_threads < avl_threads ?
                    max_useful_threads : avl_threads;
#ifdef MULTI_THREADING
        if (avl_threads > 1 &&
            should_use_colmajor_batched_solver(solution, kertab, avl_threads))
        {
            solver->solver_type = SOLVER_MT_DIRECT_BATCHED_COLMAJOR;
            n_threads = avl_threads;
        }
        else if (n_threads > 1)
        {
            solver->solver_type = SOLVER_MT_DIRECT_BATCHED_ROWMAJOR;
        }
        else
#endif
        {
            n_threads = 1;
            solver->solver_type = SOLVER_DIRECT_BATCHED_COLMAJOR;
        }
    }

    if(n_threads > 1)
    {
        n_threads = (batch < avl_threads) ? batch : avl_threads;
    }

    decomp_scheme->thread_info->n_threads = n_threads;

    return SOLVER_SUCCESS;
}

FFTZ_INT32 selector_direct_dft(aoclfftz_selector_t *sel, kernel_t *kertab)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    if (sel == NULL || sel->solution == NULL ||
        sel->solution->decomp_scheme == NULL)
    {
        AOCLFFTZ_LOG(INFO, global_logger_mode,
                     "Invalid selector or solution passed to "
                     "selector_direct_dft");
        return SELECTOR_FAILURE;
    }

    aoclfftz_decomp_scheme_t *decomp_scheme = sel->solution->decomp_scheme;

    aoclfftz_selector_t *cur_sel = NULL;
    FFTZ_INTP n = decomp_scheme->dims[0].n;
    FFTZ_INT32 vec_rank = decomp_scheme->vec_rank;
    FFTZ_INT32 dim_rank = decomp_scheme->dim_rank;
    FFTZ_INT32 stats_mode = decomp_scheme->cntrl_params->measure_stats;
    FFTZ_INT32 avl_threads = decomp_scheme->thread_info->avl_threads;
    FFTZ_UINT32 precision = DT_PRECISION_FLAG(decomp_scheme->flags);
    FFTZ_UINT32 selector_mode = GET_SELECTOR_MODE(decomp_scheme->flags);
    FFTZ_INT32 ret = SELECTOR_FAILURE;
    FFTZ_INT32 setup_ret = AOCLFFTZ_SETUP_FAILURE;

    cur_sel = alloc_selector(vec_rank, dim_rank, sel->kernel_tables,
                             sel->has_nested);
    if (cur_sel == NULL)
    {
        return SELECTOR_FAILURE;
    }

    // copy solution object from sel to cur_sel
    setup_ret = copy_solution_obj(cur_sel->solution, sel->solution);
    if (setup_ret != AOCLFFTZ_SUCCESS)
    {
        AOCLFFTZ_ERROR("copy_solution_obj failed: %s",
                        get_status_string(setup_ret));
        goto exit_direct_dft;
    }

    // find a suitable kernel within the list of C kernels, and if one is found,
    // check for the existance of other implementations for the same radix
    aoclfftz_generic_solver_t *cur_solver = cur_sel->solution->solver;
    for (FFTZ_INTP i = 0; i < NUM_KERNELS_IN_EACH_CATEGORY; i++)
    {
        FFTZ_UINT32 radix = kertab[i].radix;
        FFTZ_UINT8 direction = FFT_DIR(decomp_scheme->flags);
        if (radix == 0) // End of search for suitable kernels in the list
        {
            break;
        }

        if ((FFTZ_INTP)radix == n)
        {
            for (FFTZ_INTP kcat = 0; kcat < NUM_KERNEL_CATEGORIES; kcat++)
            {
                FFTZ_INTP kloc = (kcat * NUM_KERNELS_IN_EACH_CATEGORY) + i;

                if (kertab[kloc].kfft[direction] == NULL)
                {
                    continue;
                }

                cur_solver->kernel_c2c->kfft[FORWARD_FFT_DIR] =
                    kertab[kloc].kfft[FORWARD_FFT_DIR];
                cur_solver->kernel_c2c->kfft[BACKWARD_FFT_DIR] =
                    kertab[kloc].kfft[BACKWARD_FFT_DIR];
                cur_solver->kernel_c2c->sets = kertab[kloc].sets[precision - 2];

                // Choose the direct solver variant and thread count for this
                // problem. The execute function pointer is bound by the parent
                // (selector level) after this selector returns.
                setup_ret = select_direct_solver_type(cur_sel->solution, kertab,
                                                      avl_threads);
                if (setup_ret != SOLVER_SUCCESS)
                {
                    AOCLFFTZ_ERROR("Direct solver variant selection failed for "
                                   "solver_type %d", cur_solver->solver_type);
                    ret = SELECTOR_FAILURE;
                    goto exit_direct_dft;
                }
#ifdef MULTI_THREADING
                if (cur_sel->solution->solver->solver_type == SOLVER_MT_DIRECT)
                {
                    // Call multi threaded direct solver
                    setup_ret = setup_mt_direct_solver(cur_sel->solution,
                                                 cur_sel->cost_analysis,
                                                 &kertab[kloc],
                                                 sel->has_nested);
                    if (setup_ret != SOLVER_SUCCESS)
                    {
                        AOCLFFTZ_ERROR("setup_mt_direct_solver failed: %d",
                                       setup_ret);
                        ret = SELECTOR_FAILURE;
                        goto exit_direct_dft;
                    }
                }
                else
#endif
                {
                    // Call single threaded direct solver
                    setup_ret = setup_direct_solver(cur_sel->solution,
                                              cur_sel->cost_analysis,
                                              &kertab[kloc]);
                    if (setup_ret != SOLVER_SUCCESS)
                    {
                        AOCLFFTZ_ERROR("setup_direct_solver failed: %d",
                                       setup_ret);
                        ret = SELECTOR_FAILURE;
                        goto exit_direct_dft;
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
                            goto exit_direct_dft;
                        }
                        setup_ret = copy_strides(
                            sel->solution, cur_sel->solution);
                        if (setup_ret != AOCLFFTZ_SUCCESS)
                        {
                            AOCLFFTZ_ERROR("copy_strides failed: %s",
                                            get_status_string(setup_ret));
                            ret = SELECTOR_FAILURE;
                            goto exit_direct_dft;
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
                            goto exit_direct_dft;
                        }
                        setup_ret = copy_strides(
                            sel->solution, cur_sel->solution);
                        if (setup_ret != AOCLFFTZ_SUCCESS)
                        {
                            AOCLFFTZ_ERROR("copy_strides failed: %s",
                                            get_status_string(setup_ret));
                            ret = SELECTOR_FAILURE;
                            goto exit_direct_dft;
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

exit_direct_dft:
    destroy_selector(cur_sel);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return ret;
}
