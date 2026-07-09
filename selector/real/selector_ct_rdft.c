// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

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

FFTZ_INT32 selector_ct_rdft(aoclfftz_selector_t *sel, kernel_t *kertab,
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

    FFTZ_INTP n = sel->solution->decomp_scheme->dims[0].n;
    FFTZ_INT32 vec_rank = sel->solution->decomp_scheme->vec_rank;
    FFTZ_INT32 dim_rank = sel->solution->decomp_scheme->dim_rank;
    FFTZ_INT32 stats_mode =
        sel->solution->decomp_scheme->cntrl_params->measure_stats;
    FFTZ_UINT32 radix_r = 0;
    FFTZ_UINT32 radix_m = 0;
    FFTZ_UINT32 is_backward =
        FFT_DIR(sel->solution->decomp_scheme->flags) == BACKWARD_FFT_DIR;
    FFTZ_INT32 ret = SELECTOR_FAILURE;

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

    cur_sel = alloc_selector(vec_rank, dim_rank, sel->kernel_tables);
    cur_sel_m = alloc_selector(vec_rank, dim_rank, sel->kernel_tables);
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

    ret = copy_solution_obj(org_sol, sel->solution);
    if (ret != AOCLFFTZ_SUCCESS)
    {
        AOCLFFTZ_ERROR("copy_solution_obj failed: %s", get_status_string(ret));
        goto exit_ct_dft;
    }
    org_sol->next_sol = NULL;

    // Flag to store whether the previous solution is selected
    // based on minimum ops cost
    FFTZ_UINT8 is_previous_solution_selected = 0;

    for (FFTZ_INTP i = 0; i < NUM_KERNELS_IN_EACH_CATEGORY; i++)
    {
        radix_r = (FFTZ_INTP)kertab[i].radix;

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
            destroy_selector(cur_sel);
            destroy_selector(cur_sel_m);
            cur_sel = alloc_selector(vec_rank, dim_rank, sel->kernel_tables);
            cur_sel_m = alloc_selector(vec_rank, dim_rank, sel->kernel_tables);
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

        // TODO: if selector mode is AOCLFFTZ_AUTO_SELECTOR
        // call twiddle multiplier

        if (GET_SELECTOR_MODE(sel->solution->decomp_scheme->flags)
            == AOCLFFTZ_FIXED_SELECTOR)
        {
            if (sel->cost_analysis->ops == 0 ||
                ((cur_sel->cost_analysis->ops + cur_sel_m->cost_analysis->ops) <
                 sel->cost_analysis->ops))
            {
                sel->cost_analysis->ops =
                    cur_sel->cost_analysis->ops + cur_sel_m->cost_analysis->ops;
                sel->cost_analysis->time = cur_sel->cost_analysis->time +
                                           cur_sel_m->cost_analysis->time;

                if (next_sol->next_sol == NULL)
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
                ret = copy_solution_obj(next_sol, cur_sel->solution);
                if (ret != AOCLFFTZ_SUCCESS)
                {
                    AOCLFFTZ_ERROR("copy_solution_obj failed: %s",
                                   get_status_string(ret));
                    goto exit_ct_dft;
                }
                ret = copy_strides(next_sol, cur_sel->solution);
                if (ret != AOCLFFTZ_SUCCESS)
                {
                    AOCLFFTZ_ERROR("copy_strides failed: %s",
                                   get_status_string(ret));
                    goto exit_ct_dft;
                }
                // Restore the original next_sol after copy
                next_sol->next_sol = sel_next_sol;
                ret = copy_solution_obj(
                    next_sol->next_sol[0], cur_sel_m->solution);
                if (ret != AOCLFFTZ_SUCCESS)
                {
                    AOCLFFTZ_ERROR("copy_solution_obj failed: %s",
                                   get_status_string(ret));
                    goto exit_ct_dft;
                }
                ret = copy_strides(
                    next_sol->next_sol[0], cur_sel_m->solution);
                if (ret != AOCLFFTZ_SUCCESS)
                {
                    AOCLFFTZ_ERROR("copy_strides failed: %s",
                                   get_status_string(ret));
                    goto exit_ct_dft;
                }

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
    destroy_selector(cur_sel);
    destroy_selector(cur_sel_m);
    destroy_solution(org_sol);
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return ret;
}
