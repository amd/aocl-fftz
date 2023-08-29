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

 /** @file selector_ct_dft.c
 *
 *  @brief Wrapper that invokes the CT solver as guided by the selector.
 *
 *  This file contains the implementation of functions that are used to
 *  setup, factorize and evaluate sub-problems and kernels as applicable.
 *
 *  @author S. Biplab Raut
 */

#include "selector/selector.h"
#include "core/common/memory_manager.h"
#include "utils/utils.h"
#include "core/common/twiddle.h"

UINT32 check_radix_applicable(ptrdiff_t n, UINT32 r)
{
    if (r > 0)
    {
        if ((n % r) == 0)
            return 1;
        else
            return 0;
    }
    else
    {
        return 0;
    }
}

INT32 selector_ct_dft(aoclfftz_selector_t *sel,
                      kernel_t *kertab)
{
    aoclfftz_selector_t *cur_sel = NULL;
    aoclfftz_selector_t *cur_sel_m = NULL;
    aoclfftz_solution_t *next_sol = NULL;
#if IN_MEMORY_TWIDDLE_FACTORS==1
    VOID* TW = NULL;
    UINT32 dt_prec = 0;
#endif
    ptrdiff_t n = sel->solution->decomp_scheme->dims[0].n;
    INT32 vec_rank = sel->solution->decomp_scheme->vec_rank;
    INT32 dim_rank = sel->solution->decomp_scheme->dim_rank;
    INT32 logger_mode = sel->solution->decomp_scheme->cntrl_params->
        logger_mode;
    INT32 stats_mode = sel->solution->decomp_scheme->cntrl_params->
        measure_stats;
    UINT32 radix_r = 0;
    UINT32 radix_m = 0;
    INT32 ker_cat = 0;
    INT32 ret = SELECTOR_FAILURE;

    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Enter");

    if (vec_rank != 1 || dim_rank != 1)
        return ret;

#if IN_MEMORY_TWIDDLE_FACTORS==1
    dt_prec = DT_PRECISION_FLAG(sel->solution->decomp_scheme->flags);
#endif
    cur_sel = alloc_selector(vec_rank, dim_rank);
    cur_sel_m = alloc_selector(vec_rank, dim_rank);
    if (cur_sel == NULL || cur_sel_m == NULL)
        goto exit_ct_dft;
    
    next_sol = alloc_solution(vec_rank, dim_rank);
    if (next_sol == NULL)
        goto exit_ct_dft;

#if IN_MEMORY_TWIDDLE_FACTORS==1
    TW = alloc_twiddle_for_solution(n, dt_prec);
    if (TW == NULL)
        goto exit_ct_dft;
#endif

    for (ker_cat = 0; ker_cat < NUM_KERNELS_IN_TABLE; ker_cat++)
    {
        radix_r = kertab[ker_cat].radix;

        if (radix_r == 0) //End of suitable kernels in the list
            break;

        //Check if this radix can factorize the problem
        if (check_radix_applicable(n, radix_r) == 0)
            continue;

        //choose the other radix m
        radix_m = n / radix_r;

        ret = setup_ct_solver(sel->solution,
            cur_sel->solution,
            cur_sel_m->solution,
            radix_r,
            radix_m);
        if (ret != SELECTOR_SUCCESS)
            goto exit_ct_dft;

        //Compute twiddle factors in a separate buffer : ToDo for IN_MEMORY_TWIDDLE_FACTORS

        //Call selector for applying CT on the m set of sub-problems (radix-m)
        ret = setup_dft_(cur_sel_m, kertab);
        if (ret != SELECTOR_SUCCESS)
            goto exit_ct_dft;

        if (GET_SELECTOR_MODE(sel->solution->decomp_scheme->flags) ==
            AOCLFFTZ_AUTO_SELECTOR_MODE)
        {
            //Call twiddle multiplier
        }
        
        //Call selector for the radix-r sub-problem
        ret = setup_dft_(cur_sel, kertab);
        if (ret != SELECTOR_SUCCESS)
            goto exit_ct_dft;

        if (GET_SELECTOR_MODE(sel->solution->decomp_scheme->flags) ==
            AOCLFFTZ_FIXED_SELECTOR_MODE)
        {
            if ((cur_sel->cost_analysis->ops +
                cur_sel_m->cost_analysis->ops) <
                sel->cost_analysis->ops)
            {
                sel->cost_analysis->ops = cur_sel->cost_analysis->ops +
                    cur_sel_m->cost_analysis->ops;
                sel->cost_analysis->time = cur_sel->cost_analysis->time +
                    cur_sel_m->cost_analysis->time;
                //copy cur_sel to sel and cur_sel_m to next_sol
                COPY_SOLUTION_OBJ(sel->solution, cur_sel->solution);
                COPY_SOLUTION_OBJ(next_sol, cur_sel_m->solution);
#if IN_MEMORY_TWIDDLE_FACTORS==1
                sel->solution->twiddle->TW = TW;
#endif
            }
        }
        else
        {
            if ((cur_sel->cost_analysis->time +
                cur_sel_m->cost_analysis->time) <
                sel->cost_analysis->time)
            {
                sel->cost_analysis->ops = cur_sel->cost_analysis->ops +
                cur_sel_m->cost_analysis->ops;
                sel->cost_analysis->time = cur_sel->cost_analysis->time +
                    cur_sel_m->cost_analysis->time;
                //copy cur_sel to sel and cur_sel_m to next_sol
                COPY_SOLUTION_OBJ(sel->solution, cur_sel->solution);
                COPY_SOLUTION_OBJ(next_sol, cur_sel_m->solution);
#if IN_MEMORY_TWIDDLE_FACTORS==1
                sel->solution->twiddle->TW = TW;
#endif
            }
        }
        if (stats_mode)
        {
            //capture stats
        }
    }

    sel->solution->next_sol = next_sol;

    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit");
    destroy_selector(cur_sel);
    destroy_selector(cur_sel_m);

    return ret;

exit_ct_dft:
    destroy_selector(cur_sel);
    destroy_selector(cur_sel_m);
    destroy_solution(next_sol);
#if IN_MEMORY_TWIDDLE_FACTORS==1
    FREE_ALLOCATED_MEM(TW);
#endif

    return ret;
}