/**
 * Copyright (C) 2023, Advanced Micro Devices. All rights reserved.
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

 /** @file selector_direct_dft.c
 *
 *  @brief Wrapper that acts on the direct solver as guided by the selector.
 *
 *  This file contains the implementation of functions that are used to
 *  setup, factorize and evaluate sub-problems and kernels as applicable.
 *
 *  @author S. Biplab Raut
 */

#include "selector/selector.h"
#include "core/common/memory_manager.h"
#include "utils/utils.h"

INT32 selector_direct_dft(aoclfftz_selector_t *sel,
                          kernel_t *kertab)
{
    aoclfftz_selector_t *cur_sel = NULL;
    ptrdiff_t n = sel->solution->decomp_scheme->dims[0].n;
    INT32 vec_rank = sel->solution->decomp_scheme->vec_rank;
    INT32 dim_rank = sel->solution->decomp_scheme->dim_rank;
    INT32 logger_mode = sel->solution->decomp_scheme->cntrl_params->
                            logger_mode;
    INT32 stats_mode = sel->solution->decomp_scheme->cntrl_params->
                            measure_stats;
    UINT32 radix = 0;
    INT32 ker_cat = 0;
    UINT32 selector_mode = GET_SELECTOR_MODE(sel->solution->decomp_scheme->flags);
    INT32 ret = SELECTOR_FAILURE;

    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Enter");

    cur_sel = alloc_selector(vec_rank, dim_rank);
    if (cur_sel == NULL)
        return SELECTOR_FAILURE;

    //copy solution object from sel to cur_sel
    COPY_SOLUTION_OBJ(cur_sel->solution, sel->solution);

    for (ker_cat = 0; ker_cat < NUM_KERNELS_IN_TABLE; ker_cat++)
    {
        radix = kertab[ker_cat].radix;

        if (radix == 0) //End of search for suitable kernels in the list
            break;

        if (radix == n)
        {
            AOCLFFTZ_LOG_FORMATTED(TRACE, logger_mode,
                                "Evaluating Radix-%td kernel", n);
            cur_sel->solution->solver->kernel_r = kertab[ker_cat].kfft;
            cur_sel->solution->solver->kernel_m = NULL;

            //call direct solver
            ret = setup_direct_solver(cur_sel->solution,
                                      cur_sel->cost_analysis,
                                      &kertab[ker_cat]);

            if (SELECTOR_SUCCESS == ret)
            {
                if (selector_mode ==
                    AOCLFFTZ_FIXED_SELECTOR_MODE)
                {
                    if(!sel->cost_analysis->ops)
                    {
                        sel->cost_analysis->ops =
                            cur_sel->cost_analysis->ops;
                        sel->cost_analysis->time =
                            cur_sel->cost_analysis->time;
                        //copy solution object from cur_sel to sel
                        COPY_SOLUTION_OBJ(sel->solution, cur_sel->solution);
                    }
                    if (cur_sel->cost_analysis->ops <
                        sel->cost_analysis->ops)
                    {
                        sel->cost_analysis->ops =
                            cur_sel->cost_analysis->ops;
                        sel->cost_analysis->time =
                            cur_sel->cost_analysis->time;
                        //copy solution object from cur_sel to sel
                        COPY_SOLUTION_OBJ(sel->solution, cur_sel->solution);
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
                        //copy solution object from cur_sel to sel
                        COPY_SOLUTION_OBJ(sel->solution, cur_sel->solution);
                    }
                }
                if (stats_mode)
                {
                    //capture stats
                }
            } //if (SELECTOR_SUCCESS == ret)
        } //if (radix == n)
    } //End of FOR loop

    destroy_selector(cur_sel);

    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit");

    return SELECTOR_SUCCESS;
}
