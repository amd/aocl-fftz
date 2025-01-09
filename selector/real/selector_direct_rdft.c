/**
 * Copyright (C) 2025, Advanced Micro Devices. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

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
#include "utils/utils.h"

INT32 selector_direct_rdft(aoclfftz_selector_t *sel, kernel_t *kertab,
                           aoclfftz_realhelper_t *realhelper)
{
    aoclfftz_decomp_scheme_t *decomp_scheme = sel->solution->decomp_scheme;
#ifdef AOCL_ENABLE_LOG
    INT32 logger_mode = decomp_scheme->cntrl_params->logger_mode;
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Enter");
#endif
    aoclfftz_selector_t *cur_sel = NULL;
    INTP n = decomp_scheme->dims[0].n;
    INTP n_batch = decomp_scheme->vecs[0].n;
    INT32 vec_rank = decomp_scheme->vec_rank;
    INT32 dim_rank = decomp_scheme->dim_rank;
    INT32 stats_mode = decomp_scheme->cntrl_params->measure_stats;
    UINT32 avl_threads = decomp_scheme->thread_info->avl_threads;
    UINT32 precision = DT_PRECISION_FLAG(decomp_scheme->flags);
    UINT32 selector_mode = GET_SELECTOR_MODE(decomp_scheme->flags);
    INT32 ret = SELECTOR_FAILURE;

    kernel_t *kernel_c2c = NULL;
    kernel_t *kernel_r2hc = NULL;
    kernel_t *kernel_r2hcf = NULL;

    cur_sel = alloc_selector(vec_rank, dim_rank, sel->scratch_space,
                             0 /*unused*/);
    if (cur_sel == NULL)
    {
        return SELECTOR_FAILURE;
    }

    INT32 num_threads = (n_batch < avl_threads) ? n_batch : avl_threads;
    decomp_scheme->thread_info->n_threads = num_threads;

    // copy solution object from sel to cur_sel
    COPY_SOLUTION_OBJ(cur_sel->solution, sel->solution);

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
#ifdef AOCL_ENABLE_LOG
            AOCLFFTZ_LOG_FORMATTED(TRACE, logger_mode,
                                   "Evaluating Radix-%td kernel", n);
#endif
            for (INTP kcat = 0; kcat < NUM_KERNEL_CATEGORIES; kcat++)
            {
                INTP kloc = (kcat * NUM_KERNELS_IN_EACH_CATEGORY) + i;

                INTP kloc_r2hc  = 0 * NUM_KERNELS_IN_EACH_DFT_VARIANT + kloc;
                INTP kloc_r2hcf = 1 * NUM_KERNELS_IN_EACH_DFT_VARIANT + kloc;
                INTP kloc_c2c   = 2 * NUM_KERNELS_IN_EACH_DFT_VARIANT + kloc;

                if (kertab[kloc_r2hc].kfft == NULL ||
                    kertab[kloc_r2hcf].kfft == NULL ||
                    kertab[kloc_c2c].kfft == NULL)
                {
                    continue;
                }

                kernel_c2c = &kertab[kloc_c2c];
                kernel_r2hc = &kertab[kloc_r2hc];
                kernel_r2hcf = &kertab[kloc_r2hcf];

                cur_sel->solution->solver->kernel_c2c->kfft =
                    kernel_c2c->kfft;
                cur_sel->solution->solver->kernel_r2hc->kfft =
                    kernel_r2hc->kfft;
                cur_sel->solution->solver->kernel_r2hcf->kfft =
                    kernel_r2hcf->kfft;

                cur_sel->solution->solver->kernel_c2c->sets =
                    kernel_c2c->sets[precision - 2];
                cur_sel->solution->solver->kernel_r2hc->sets =
                    kernel_r2hc->sets[precision - 2];
                cur_sel->solution->solver->kernel_r2hcf->sets =
                    kernel_r2hcf->sets[precision - 2];

#ifdef MULTI_THREADING
                if (num_threads <= 1 || realhelper->is_CT)
                {
#endif
                    // call direct solver
                    ret = setup_real_direct_solver(
                        cur_sel->solution, cur_sel->cost_analysis, kernel_c2c,
                        kernel_r2hc, kernel_r2hcf, realhelper);
#ifdef MULTI_THREADING
                }
                else
                {
                    ret = setup_real_mt_direct_solver(
                        cur_sel->solution, cur_sel->cost_analysis, kernel_c2c,
                        kernel_r2hc, kernel_r2hcf, realhelper);
                }
#endif
                if (SELECTOR_SUCCESS == ret)
                {
                    if (selector_mode == AOCLFFTZ_FIXED_SELECTOR)
                    {
                        if (!sel->cost_analysis->ops)
                        {
                            sel->cost_analysis->ops =
                                cur_sel->cost_analysis->ops;
                            sel->cost_analysis->time =
                                cur_sel->cost_analysis->time;
                            // copy solution object from cur_sel to sel
                            COPY_SOLUTION_OBJ(sel->solution, cur_sel->solution);
                            COPY_STRIDES(sel->solution, cur_sel->solution);
                        }
                        if (cur_sel->cost_analysis->ops <
                            sel->cost_analysis->ops)
                        {
                            sel->cost_analysis->ops =
                                cur_sel->cost_analysis->ops;
                            sel->cost_analysis->time =
                                cur_sel->cost_analysis->time;
                            // copy solution object from cur_sel to sel
                            COPY_SOLUTION_OBJ(sel->solution, cur_sel->solution);
                            COPY_STRIDES(sel->solution, cur_sel->solution);
                        }
                    }
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
                            COPY_SOLUTION_OBJ(sel->solution, cur_sel->solution);
                            COPY_STRIDES(sel->solution, cur_sel->solution);
                        }
                    }
                    if (stats_mode)
                    {
                        // capture stats
                    }
                } // if (SELECTOR_SUCCESS == ret)
            } // End of FOR loop
            break;
        } // if (radix == n)
    } // End of FOR loop

    destroy_selector_without_scratch_space(cur_sel);

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit");
#endif
    return SELECTOR_SUCCESS;
}
