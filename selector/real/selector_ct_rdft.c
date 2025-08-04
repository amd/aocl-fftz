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

/** @file selector_ct_rdft.c
 *
 *  @brief Wrapper that invokes the Real CT Solver as guided by the Selector.
 *
 *  This file contains the implementation of functions that are used to
 *  setup, factorize and evaluate sub-problems and kernels as applicable.
 *
 *  @author Srirammaswamy Srinivasan
 */

#include "api/aoclfftz_internal.h"
#include "selector/selector.h"
#include "core/common/memory_manager.h"
#include "utils/utils.h"
#include "core/common/twiddle.h"

INT32 selector_ct_rdft(aoclfftz_selector_t *sel, kernel_t *kertab,
                       aoclfftz_realhelper_t *realhelper)
{
#ifdef AOCL_ENABLE_LOG
    INT32 logger_mode = sel->solution->decomp_scheme->cntrl_params->logger_mode;
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Enter");
#endif
    aoclfftz_selector_t *cur_sel = NULL;
    aoclfftz_selector_t *cur_sel_m = NULL;

    // holds the original sub-problem. 'sel' would get overwritten while
    // updating cost
    aoclfftz_solution_t *org_sol = NULL;

    realhelper->is_CT = 1;

    INTP n = sel->solution->decomp_scheme->dims[0].n;
    INT32 vec_rank = sel->solution->decomp_scheme->vec_rank;
    INT32 dim_rank = sel->solution->decomp_scheme->dim_rank;
    INT32 stats_mode =
        sel->solution->decomp_scheme->cntrl_params->measure_stats;
    UINT32 radix_r = 0;
    UINT32 radix_m = 0;
    INT32 ker_cat = 0;
    aoclfftz_kernel_type kernel_type;
    UINT32 is_backward =
        FFT_DIR(sel->solution->decomp_scheme->flags) == BACKWARD_FFT_DIR;
    INT32 ret = SELECTOR_FAILURE;

    if (vec_rank != 1 || dim_rank != 1)
    {
        return ret;
    }

    org_sol = alloc_solution(vec_rank, dim_rank);
    if (org_sol == NULL)
    {
        goto exit_ct_dft;
    }

    cur_sel = alloc_selector(vec_rank, dim_rank, sel->scratch_space,
                             0 /*unused*/);
    cur_sel_m = alloc_selector(vec_rank, dim_rank, sel->scratch_space,
                               0 /*unused*/);
    if (cur_sel == NULL || cur_sel_m == NULL)
    {
        goto exit_ct_dft;
    }

    // Create empty solutions to copy cur_sel & cur_sel_m
    sel->solution->next_sol = alloc_sol_array(1 /*n_threads*/);
    sel->solution->next_sol[0] = alloc_solution(vec_rank, dim_rank);
    aoclfftz_solution_t *next_sol = sel->solution->next_sol[0];
    next_sol->next_sol = alloc_sol_array(1 /*n_threads*/);
    next_sol->next_sol[0] = alloc_solution(vec_rank, dim_rank);

    COPY_SOLUTION_OBJ(org_sol, sel->solution);
    org_sol->next_sol = NULL;

    // Flag to store whether the previous solution is selected
    // based on minimum ops cost
    UINT8 is_previous_solution_selected = 0;

    for (ker_cat = 0; ker_cat < NUM_KERNELS_IN_TABLE; ker_cat++)
    {
        radix_r = kertab[ker_cat].radix;
        kernel_type = kertab[ker_cat].kernel_type;

        if (radix_r == 0) // End of suitable kernels in the list
        {
            break;
        }

        // Check if this radix can factorize the problem
        if ((n % radix_r) != 0)
        {
            continue;
        }

        // FIXIT
        if (kernel_type != C2C_KERNEL)
        {
            continue;
        }

        // choose the other radix m
        radix_m = n / radix_r;

        // Create a new cur_sel & cur_sel_m selectors
        // if previous solutions is selected
        if (is_previous_solution_selected)
        {
            destroy_selector_without_scratch_space(cur_sel);
            destroy_selector_without_scratch_space(cur_sel_m);
            cur_sel = alloc_selector(vec_rank, dim_rank, sel->scratch_space,
                                     0 /*unused*/);
            cur_sel_m = alloc_selector(vec_rank, dim_rank, sel->scratch_space,
                                       0 /*unused*/);
            if (cur_sel == NULL || cur_sel_m == NULL)
            {
                goto exit_ct_dft;
            }
            is_previous_solution_selected = 0;
        }

        ret = setup_real_ct_solver(org_sol, cur_sel->solution,
                                   cur_sel_m->solution, radix_r, radix_m,
                                   realhelper);
        if (ret != SELECTOR_SUCCESS)
        {
            goto exit_ct_dft;
        }

        realhelper->is_last_stage = 1;
        realhelper->stage++;
        if (is_backward)
        {
            realhelper->p /= radix_r;
            realhelper->q *= radix_r;
        }
        else
        {
            realhelper->p *= radix_r;
            realhelper->q /= radix_r;
        }

        // Call selector for applying CT on the m set of sub-problems (radix-m)
        ret = selector_model_rdft_(cur_sel_m, realhelper);
        if (ret != SELECTOR_SUCCESS)
        {
            goto exit_ct_dft;
        }

        if (is_backward)
        {
            realhelper->p *= radix_r;
            realhelper->q /= radix_r;
        }
        else
        {
            realhelper->p /= radix_r;
            realhelper->q *= radix_r;
        }
        realhelper->stage--;
        realhelper->is_last_stage = 0;

        // Call selector for the radix-r sub-problem
        ret = selector_model_rdft_(cur_sel, realhelper);
        if (ret != SELECTOR_SUCCESS)
        {
            goto exit_ct_dft;
        }

        if (GET_SELECTOR_MODE(sel->solution->decomp_scheme->flags) ==
            AOCLFFTZ_AUTO_SELECTOR)
        {
            // Call twiddle multiplier
        }

        if (GET_SELECTOR_MODE(sel->solution->decomp_scheme->flags) ==
            AOCLFFTZ_FIXED_SELECTOR)
        {
            if (sel->cost_analysis->ops == 0 ||
                ((cur_sel->cost_analysis->ops + cur_sel_m->cost_analysis->ops) <
                 sel->cost_analysis->ops))
            {
                sel->cost_analysis->ops =
                    cur_sel->cost_analysis->ops + cur_sel_m->cost_analysis->ops;
                sel->cost_analysis->time = cur_sel->cost_analysis->time +
                                           cur_sel_m->cost_analysis->time;

                // Destroy the solutions except the first 3 objects
                // since it points to current CT, CT-R, CT-M respectively
                if (sel->solution != NULL && next_sol != NULL &&
                                             next_sol->next_sol[0] != NULL)
                {
                    destroy_solutions(next_sol->next_sol[0]->next_sol, 1);
                }
                aoclfftz_solution_t **sel_next_sol = next_sol->next_sol;
                COPY_SOLUTION_OBJ(next_sol, cur_sel->solution);
                COPY_STRIDES(next_sol, cur_sel->solution);
                // Restore the original next_sol after copy
                next_sol->next_sol = sel_next_sol;
                COPY_SOLUTION_OBJ(next_sol->next_sol[0], cur_sel_m->solution);
                COPY_STRIDES(next_sol->next_sol[0], cur_sel_m->solution);

                // Break the link from cur_sel and cur_sel_m
                // it can be still accessed through sel object
                cur_sel->solution->next_sol = NULL;
                cur_sel_m->solution->next_sol = NULL;
                is_previous_solution_selected = 1;
            }
            else
            {
                // Destroy the solutions of cur_sel and cur_sel_m
                // except first solution
                destroy_solutions(cur_sel->solution->next_sol, 1);
                cur_sel->solution->next_sol = NULL;
                destroy_solutions(cur_sel_m->solution->next_sol, 1);
                cur_sel_m->solution->next_sol = NULL;
                is_previous_solution_selected = 0;
                RESET_COST(cur_sel);
                RESET_COST(cur_sel_m);
            }
        }
        else
        {
            // FIXME: Update this logic
            if ((cur_sel->cost_analysis->time +
                 cur_sel_m->cost_analysis->time) < sel->cost_analysis->time)
            {
                sel->cost_analysis->ops =
                    cur_sel->cost_analysis->ops + cur_sel_m->cost_analysis->ops;
                sel->cost_analysis->time = cur_sel->cost_analysis->time +
                                           cur_sel_m->cost_analysis->time;
                next_sol = cur_sel->solution;
                next_sol->next_sol[0] = cur_sel_m->solution;
            }
        }
        if (stats_mode)
        {
            // capture stats
        }
    }

exit_ct_dft:
    destroy_selector_without_scratch_space(cur_sel);
    destroy_selector_without_scratch_space(cur_sel_m);
    destroy_solution(org_sol, 0);
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit");
#endif

    return ret;
}
