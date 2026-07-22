// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file selector_ct_dft.c
 *
 *  @brief Wrapper that invokes the CT solver as guided by the selector.
 *
 *  This file contains the implementation of functions that are used to
 *  setup, factorize and evaluate sub-problems and kernels as applicable.
 *
 *  @author S. Biplab Raut
 *  @author Ashwin K. Godbole
 */

#include "selector/selector.h"
#include "core/common/memory_manager.h"
#include "utils/utils.h"
#include "core/common/twiddle.h"

FFTZ_INT32 selector_ct_dft(aoclfftz_selector_t *sel, kernel_t *kertab)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    if (sel == NULL || sel->solution == NULL ||
        sel->solution->decomp_scheme == NULL)
    {
        AOCLFFTZ_LOG(INFO, global_logger_mode,
                     "Invalid selector or solution passed to selector_ct_dft");
        return SELECTOR_FAILURE;
    }

    aoclfftz_selector_t *cur_sel = NULL;
    aoclfftz_selector_t *cur_sel_m = NULL;

    // holds the original sub-problem. 'sel' would get overwritten while
    // updating cost
    aoclfftz_solution_t *org_sol = NULL;

    FFTZ_INTP n = sel->solution->decomp_scheme->dims[0].n;
    FFTZ_INT32 vec_rank = sel->solution->decomp_scheme->vec_rank;
    FFTZ_INT32 dim_rank = sel->solution->decomp_scheme->dim_rank;
    FFTZ_INT32 stats_mode =
        sel->solution->decomp_scheme->cntrl_params->measure_stats;
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

    cur_sel = alloc_selector(vec_rank, dim_rank, sel->kernel_tables,
                             sel->has_nested);
    cur_sel_m = alloc_selector(vec_rank, dim_rank, sel->kernel_tables,
                               sel->has_nested);

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

    // Radix candidates below are speculative: the losing ones are thrown away,
    // so their sub-solvers must not leave the plan-wide nested-parallel flag
    // set. Every candidate starts from the same baseline and only the winning
    // candidate's contribution survives the loop.
    FFTZ_UINT8 nested_on_entry = *(sel->has_nested);
    FFTZ_UINT8 nested_selected = nested_on_entry;

    for (FFTZ_INTP i = 0; i < NUM_KERNELS_IN_EACH_CATEGORY; i++)
    {
        FFTZ_INTP radix_r = (FFTZ_INTP)kertab[i].radix;

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
        FFTZ_INTP radix_m = n / radix_r;

        // Create a new cur_sel & cur_sel_m selectors
        // if previous solutions is selected
        if (is_previous_solution_selected)
        {
            destroy_selector(cur_sel);
            destroy_selector(cur_sel_m);
            cur_sel = alloc_selector(vec_rank, dim_rank, sel->kernel_tables,
                                     sel->has_nested);
            cur_sel_m = alloc_selector(vec_rank, dim_rank, sel->kernel_tables,
                                       sel->has_nested);
            if (cur_sel == NULL || cur_sel_m == NULL)
            {
                ret = AOCLFFTZ_MEMORY_FAILURE;
                goto exit_ct_dft;
            }
            is_previous_solution_selected = 0;
        }

        *(sel->has_nested) = nested_on_entry;

        ret = setup_ct_solver(org_sol, cur_sel->solution, cur_sel_m->solution,
                              radix_r, radix_m);
        if (ret != SELECTOR_SUCCESS)
        {
            goto exit_ct_dft;
        }

        // Call selector for applying CT on the m set of sub-problems (radix-m)
        ret = selector_model_dft_(cur_sel_m);
        if (ret != SELECTOR_SUCCESS)
        {
            goto exit_ct_dft;
        }

        // TODO: if selector mode is AOCLFFTZ_AUTO_SELECTOR call twiddle
        // multiplier

        // Call selector for the radix-r sub-problem
#if 0
        ret = selector_model_dft_(cur_sel);//Call direct solver instead
#else
        aoclfftz_solution_t *radix_r_sol = cur_sel->solution;
        aoclfftz_solution_t *radix_m_sol = cur_sel_m->solution;

        radix_r_sol->decomp_scheme->dims[0].in_stride =
            radix_m_sol->decomp_scheme->vecs[0].out_stride;
        radix_r_sol->decomp_scheme->vecs[0].in_stride =
            radix_m_sol->decomp_scheme->dims[0].out_stride;

        aoclfftz_generic_solver_t* solver_obj = cur_sel->solution->solver;
        FFTZ_INT32 avl_threads =
            cur_sel->solution->decomp_scheme->thread_info->avl_threads;
        if (avl_threads == 1)
        {
            if (sel->solution->decomp_scheme->batched_vecs == NULL)
            {
                solver_obj->solver_type = SOLVER_DIRECT;
            }
            else
            {
                solver_obj->solver_type = SOLVER_DIRECT_BATCHED_COLMAJOR;
            }
        }
        else
        {
            if (sel->solution->decomp_scheme->batched_vecs == NULL)
            {
                solver_obj->solver_type = SOLVER_MT_DIRECT;
            }
            else
            {
                if (should_use_colmajor_batched_solver(cur_sel->solution,
                                                       kertab, avl_threads))
                {
                    solver_obj->solver_type = SOLVER_MT_DIRECT_BATCHED_COLMAJOR;
                }
                else
                {
                    solver_obj->solver_type = SOLVER_MT_DIRECT_BATCHED_ROWMAJOR;
                }
            }
        }
        if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
        {
            goto exit_ct_dft;
        }
        ret = selector_direct_dft(cur_sel, kertab);
        if (ret != SELECTOR_SUCCESS)
        {
            goto exit_ct_dft;
        }
#endif

        if (GET_SELECTOR_MODE(sel->solution->decomp_scheme->flags) ==
            AOCLFFTZ_FIXED_SELECTOR)
        {
            if (sel->cost_analysis->ops == 0 ||
                ((cur_sel->cost_analysis->ops + cur_sel_m->cost_analysis->ops) <
                 sel->cost_analysis->ops))
            {
                sel->cost_analysis->ops = cur_sel->cost_analysis->ops +
                                          cur_sel_m->cost_analysis->ops;
                sel->cost_analysis->time = cur_sel->cost_analysis->time +
                                           cur_sel_m->cost_analysis->time;

                // Verify all pointers are valid before proceeding
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

                // Free the ct_buffer if next_sol->next_sol[0] owns it
                if (next_sol->next_sol[0]->dft_bufs->ct_buf_allocated)
                {
                    FREE_ALIGN_ALLOCATED_MEM(
                        next_sol->next_sol[0]->dft_bufs->ct_buffer);
                    next_sol->next_sol[0]->dft_bufs->ct_buf_allocated = 0;
                }

                ret = copy_solution_obj(
                    next_sol->next_sol[0], cur_sel_m->solution);
                if (ret != AOCLFFTZ_SUCCESS)
                {
                    AOCLFFTZ_ERROR("copy_solution_obj failed: %s",
                                   get_status_string(ret));
                    goto exit_ct_dft;
                }
                if (cur_sel_m->solution->solver->solver_type ==
                    SOLVER_BATCHED_CT_L1_DIRECT)
                {
                    ret = copy_strides_batched_ct_l1_direct(
                        next_sol->next_sol[0],
                        cur_sel_m->solution);
                    if (ret != AOCLFFTZ_SUCCESS)
                    {
                        AOCLFFTZ_ERROR(
                            "copy_strides_batched_ct_l1_direct failed: %s",
                            get_status_string(ret));
                        goto exit_ct_dft;
                    }
                }
                else
                {
                    ret = copy_strides(
                        next_sol->next_sol[0], cur_sel_m->solution);
                    if (ret != AOCLFFTZ_SUCCESS)
                    {
                        AOCLFFTZ_ERROR("copy_strides failed: %s",
                                       get_status_string(ret));
                        goto exit_ct_dft;
                    }
                }

                if (cur_sel_m->solution->dft_bufs->ct_buf_allocated)
                {
                    next_sol->next_sol[0]->dft_bufs->ct_buf_allocated = 1;
                }
                cur_sel_m->solution->dft_bufs->ct_buf_allocated = 0;

                // Only the radix-m sub-solver uses the ct_buffer, so take the CT
                // node's ct_buf_size from it (ignore radix-r solver).
                sel->solution->dft_bufs->ct_buf_size =
                    cur_sel_m->solution->dft_bufs->ct_buf_size;

                // Break the link from cur_sel and cur_sel_m
                // it can be still accessed through sel object
                cur_sel->solution->next_sol = NULL;
                cur_sel_m->solution->next_sol = NULL;
                is_previous_solution_selected = 1;
                nested_selected = *(sel->has_nested);
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
                if (cur_sel_m->solution->dft_bufs->ct_buf_allocated)
                {
                    FREE_ALIGN_ALLOCATED_MEM(
                        cur_sel_m->solution->dft_bufs->ct_buffer);
                    cur_sel_m->solution->dft_bufs->ct_buf_allocated = 0;
                }
            }
        }
        if (stats_mode)
        {
            // capture stats
        }
    }

    *(sel->has_nested) = nested_selected;

exit_ct_dft:
    destroy_selector(cur_sel);
    destroy_selector(cur_sel_m);
    destroy_solution(org_sol);
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");


    return ret;
}
