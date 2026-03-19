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
 *  @author Ashwin K. Godbole
 */

#include "api/aoclfftz_internal.h"
#include "selector/selector.h"
#include "core/common/memory_manager.h"
#include "utils/utils.h"
#include "core/common/twiddle.h"

INT32 selector_ct_rdft(aoclfftz_selector_t *sel, kernel_t *kertab,
                       aoclfftz_realhelper_t *realhelper)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    if (sel == NULL || sel->solution == NULL ||
        sel->solution->decomp_scheme == NULL)
    {
        AOCLFFTZ_LOG(INFO, global_logger_mode,
                     "Invalid selector or solution passed to selector_ct_rdft");
        return SELECTOR_FAILURE;
    }

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
        ret = AOCLFFTZ_MEMORY_FAILURE;
        goto exit_ct_dft;
    }

    cur_sel = alloc_selector(vec_rank, dim_rank, sel->scratch_space,
                             sel->kernel_tables, 0 /*unused*/);
    cur_sel_m = alloc_selector(vec_rank, dim_rank, sel->scratch_space,
                               sel->kernel_tables, 0 /*unused*/);
    if (cur_sel == NULL || cur_sel_m == NULL)
    {
        ret = AOCLFFTZ_MEMORY_FAILURE;
        goto exit_ct_dft;
    }

    // Create empty solutions to copy cur_sel & cur_sel_m
    sel->solution->next_sol = alloc_sol_array(1 /*n_threads*/);
    if (sel->solution->next_sol == NULL)
    {
        ret = AOCLFFTZ_MEMORY_FAILURE;
        goto exit_ct_dft;
    }
    sel->solution->next_sol[0] = alloc_solution(vec_rank, dim_rank);
    if (sel->solution->next_sol[0] == NULL)
    {
        ret = AOCLFFTZ_MEMORY_FAILURE;
        goto exit_ct_dft;
    }

    aoclfftz_solution_t *next_sol = sel->solution->next_sol[0];
    next_sol->next_sol = alloc_sol_array(1 /*n_threads*/);
    if (next_sol->next_sol == NULL)
    {
        ret = AOCLFFTZ_MEMORY_FAILURE;
        goto exit_ct_dft;
    }
    next_sol->next_sol[0] = alloc_solution(vec_rank, dim_rank);
    if (next_sol->next_sol[0] == NULL)
    {
        ret = AOCLFFTZ_MEMORY_FAILURE;
        goto exit_ct_dft;
    }

    COPY_SOLUTION_OBJ(org_sol, sel->solution);
    org_sol->next_sol = NULL;

    // Flag to store whether the previous solution is selected
    // based on minimum ops cost
    UINT8 is_previous_solution_selected = 0;

    for (INTP i = 0; i < NUM_KERNELS_IN_EACH_CATEGORY; i++)
    {
        radix_r = (INTP)kertab[i].radix;

        if (radix_r == 0) // End of suitable kernels in the list
        {
            break;
        }

        // Check if this radix can factorize the problem
        if ((n % radix_r) != 0)
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
                                     sel->kernel_tables, 0 /*unused*/);
            cur_sel_m = alloc_selector(vec_rank, dim_rank, sel->scratch_space,
                                       sel->kernel_tables, 0 /*unused*/);
            if (cur_sel == NULL || cur_sel_m == NULL)
            {
                ret = AOCLFFTZ_MEMORY_FAILURE;
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
            realhelper->freq_factor /= radix_r;
        }
        else
        {
            realhelper->freq_factor *= radix_r;
        }

        // Call selector for applying CT on the m set of sub-problems (radix-m)
        ret = selector_model_rdft_(cur_sel_m, realhelper);
        if (ret != SELECTOR_SUCCESS)
        {
            goto exit_ct_dft;
        }

        if (is_backward)
        {
            realhelper->freq_factor *= radix_r;
        }
        else
        {
            realhelper->freq_factor /= radix_r;
        }
        realhelper->stage--;
        realhelper->is_last_stage = 0;

        // Call selector for the radix-r sub-problem
        ret = selector_model_rdft_(cur_sel, realhelper);
        if (ret != SELECTOR_SUCCESS)
        {
            goto exit_ct_dft;
        }

        // TODO: if selector mode is AOCLFFTZ_AUTO_SELECTOR call twiddle multiplier

        if (GET_SELECTOR_MODE(sel->solution->decomp_scheme->flags) ==
            AOCLFFTZ_FIXED_SELECTOR)
        {
            if (sel->cost_analysis->ops == 0 ||
                ((cur_sel->cost_analysis->ops + cur_sel_m->cost_analysis->ops) <
                 sel->cost_analysis->ops))
            {
                INTP twiddle_ops_cnt = 0;
                // For backward FFT, twiddles are computed separately via
                // twiddle_multiplier_for_real() and not fused into kernels.
                // Forward FFT uses fused twiddle kernels, so ops are already counted.
                if (is_backward && 
                    (cur_sel_m->solution->solver->solver_type != 
                        SOLVER_REAL_DIRECT_TWIDDLE &&
                        cur_sel_m->solution->solver->solver_type !=
                        SOLVER_REAL_MT_DIRECT_TWIDDLE))
                {
                    twiddle_ops_cnt = (radix_r - 1) * (radix_m - 1) * 4 
                                        * AMD_ZEN_FP_MUL_CYCLES +
                                      (radix_r - 1) * (radix_m - 1) * 2 
                                        * AMD_ZEN_FP_ADD_CYCLES;
                }
                sel->cost_analysis->ops =
                    cur_sel->cost_analysis->ops + cur_sel_m->cost_analysis->ops
                                            + twiddle_ops_cnt;
                sel->cost_analysis->time = cur_sel->cost_analysis->time +
                                           cur_sel_m->cost_analysis->time;

                if (next_sol == NULL || next_sol->next_sol == NULL)
                {
                    goto exit_ct_dft;
                }
                // Destroy the solutions except the first 3 objects
                // since it points to current CT, CT-R, CT-M respectively
                if (next_sol->next_sol[0] != NULL)
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
                // The solution is being discarded
                // hence its strides are no longer needed.
                destroy_strides_grp(cur_sel->solution->strides_grp);
                destroy_strides_grp(cur_sel_m->solution->strides_grp);

                RESET_COST(cur_sel);
                RESET_COST(cur_sel_m);
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
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return ret;
}
