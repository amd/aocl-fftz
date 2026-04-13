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

INT32 selector_direct_dft(aoclfftz_selector_t *sel, kernel_t *kertab)
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
    INTP n = decomp_scheme->dims[0].n;
    INTP batch = decomp_scheme->vecs[0].n;
    if (decomp_scheme->batched_vecs != NULL &&
        decomp_scheme->batched_vecs[0].n > batch)
    {
        batch = decomp_scheme->batched_vecs[0].n;
    }
    INT32 vec_rank = decomp_scheme->vec_rank;
    INT32 dim_rank = decomp_scheme->dim_rank;
    INT32 stats_mode = decomp_scheme->cntrl_params->measure_stats;
    INT32 avl_threads = decomp_scheme->thread_info->avl_threads;
    UINT32 precision = DT_PRECISION_FLAG(decomp_scheme->flags);
    UINT32 selector_mode = GET_SELECTOR_MODE(decomp_scheme->flags);
    INT32 ret = SELECTOR_FAILURE;

    cur_sel = alloc_selector(vec_rank, dim_rank, sel->kernel_tables);
    if (cur_sel == NULL)
    {
        return SELECTOR_FAILURE;
    }

    // set number of threads for execution to no. of batches
    INT32 n_threads = (batch < avl_threads) ? batch : avl_threads;
    decomp_scheme->thread_info->n_threads = n_threads;

    // copy solution object from sel to cur_sel
    ret = copy_solution_obj(cur_sel->solution, sel->solution);
    if (ret != AOCLFFTZ_SUCCESS)
    {
        AOCLFFTZ_ERROR("copy_solution_obj failed: %s", get_status_string(ret));
        goto exit_direct_dft;
    }

    // find a suitable kernel within the list of C kernels, and if one is found,
    // check for the existance of other implementations for the same radix
    for (INTP i = 0; i < NUM_KERNELS_IN_EACH_CATEGORY; i++)
    {
        UINT32 radix = kertab[i].radix;

        if (radix == 0) // End of search for suitable kernels in the list
        {
            break;
        }

        if ((INTP)radix == n)
        {
            for (INTP kcat = 0; kcat < NUM_KERNEL_CATEGORIES; kcat++)
            {
                INTP kloc = (kcat * NUM_KERNELS_IN_EACH_CATEGORY) + i;

                if (kertab[kloc].kfft == NULL)
                {
                    continue;
                }

                cur_sel->solution->solver->kernel_c2c->kfft = kertab[kloc].kfft;
                cur_sel->solution->solver->kernel_c2c->sets =
                    kertab[kloc].sets[precision - 2];

#ifdef MULTI_THREADING
                if (n_threads > 1)
                {
                    // Call multi threaded direct solver
                    ret = setup_mt_direct_solver(cur_sel->solution,
                                                 cur_sel->cost_analysis,
                                                 &kertab[kloc]);
                }
                else
#endif
                {
                    // Call single threaded direct solver
                    ret = setup_direct_solver(cur_sel->solution,
                                              cur_sel->cost_analysis,
                                              &kertab[kloc]);
                }

                if (SELECTOR_SUCCESS == ret)
                {
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
                            ret = copy_solution_obj(
                                sel->solution, cur_sel->solution);
                            if (ret != AOCLFFTZ_SUCCESS)
                            {
                                AOCLFFTZ_ERROR("copy_solution_obj failed: %s",
                                               get_status_string(ret));
                                goto exit_direct_dft;
                            }
                            ret = copy_strides(
                                sel->solution, cur_sel->solution);
                            if (ret != AOCLFFTZ_SUCCESS)
                            {
                                AOCLFFTZ_ERROR("copy_strides failed: %s",
                                               get_status_string(ret));
                                goto exit_direct_dft;
                            }
                        }
                    }
#ifdef AOCLFFTZ_AUTO_SELECTOR_MODE
                    else
                    {
                        if (cur_sel->cost_analysis->time <
                            sel->cost_analysis->time)
                        {
                            sel->cost_analysis->ops =
                                cur_sel->cost_analysis->ops;
                            sel->cost_analysis->time =
                                cur_sel->cost_analysis->time;
                            // copy solution object from cur_sel to sel
                            ret = copy_solution_obj(
                                sel->solution, cur_sel->solution);
                            if (ret != AOCLFFTZ_SUCCESS)
                            {
                                AOCLFFTZ_ERROR("copy_solution_obj failed: %s",
                                               get_status_string(ret));
                                goto exit_direct_dft;
                            }
                            ret = copy_strides(
                                sel->solution, cur_sel->solution);
                            if (ret != AOCLFFTZ_SUCCESS)
                            {
                                AOCLFFTZ_ERROR("copy_strides failed: %s",
                                               get_status_string(ret));
                                goto exit_direct_dft;
                            }
                        }
                    }
#endif // AOCLFFTZ_AUTO_SELECTOR_MODE
                    if (stats_mode)
                    {
                        // capture stats
                    }
                } // if (SELECTOR_SUCCESS == ret)
            } // End of FOR loop
            break;
        } // if (radix == n)
    } // End of FOR loop

    destroy_selector(cur_sel);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return ret;

exit_direct_dft:
    destroy_selector(cur_sel);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return ret;
}
