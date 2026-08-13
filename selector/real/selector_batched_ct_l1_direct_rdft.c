// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file selector_batched_ct_l1_direct_rdft.c
 *
 *  @brief Selector for the fused batched + 1-level CT real solver (R2C/C2R).
 *
 *  For a batched 1D real FFT (R2C/C2R), picks one CT split n = radix_r ×
 *  radix_m and wires stage_r and stage_m as two direct stages in the fused
 *  executor, instead of going through batched → buffered → CT dispatch.
 *
 *  How this selector works:
 *    1. Try each valid factorization n = radix_r × radix_m.
 *    2. For each pair, build trial stage-r and stage-m plans the same way
 *       selector_ct_rdft does (setup_real_ct_solver, then selector_model_rdft_
 *       on m and r).
 *    3. Keep only pairs where both stages are flat REAL_DIRECT leaves (no
 *       nested CT under stage-m); skip the rest.
 *    4. Among those, pick the pair with the lowest combined op count.
 *    5. Copy the winning stage templates onto sol->next_sol once, then call
 *       setup_batched_ct_l1_direct_real_solver for aux, strides, and twiddles.
 *
 *  @author Amrin Fathima
 */

#include "api/aoclfftz_internal.h"
#include "selector/selector.h"
#include "core/common/memory_manager.h"
#include "core/solvers/solver.h"

/*
 * Return 1 if SOLVER_REAL_BATCHED_CT_L1_DIRECT may be tried at this node.
 *
 * Called from selector_fixed_mode_rdft_() after fuse_vecs(), so vec_rank
 * already reflects fused batch layout.
 *
 * All of the following must hold (returns 0 on first failure):
 *   - n has no single direct real kernel (non-direct size, like complex fused)
 *   - single-threaded only (n_threads, avl_threads, active_threads == 1)
 *   - 1D problem, one batch dimension (dim_rank == 1, vec_rank == 1)
 *   - row-major layout, innermost dimension, no ND batched_vecs
 *   - top-level node only: !realhelper->is_CT and n == realhelper->problem_size
 *   - some CT split n = radix_r * radix_m with kernel support on both stages
 *
 * CT-root guard: realhelper tracks the original user length N. While
 * selector_ct_rdft recurses into stage-r/stage-m, is_CT is set and n drops
 * below N. Fused L1 must not run on those inner nodes — it replaces one full
 * CT level at the root, not a subtree the CT selector already owns.
 * Example: N = 384, radix_r = 8 gives a stage-m node at n = 48; skip fused there.
 */
FFTZ_INT32
check_batched_ct_l1_direct_rdft_solvability(
    aoclfftz_decomp_scheme_t *decomp_scheme, aoclfftz_realhelper_t *realhelper,
    kernel_t *kertab_rdft)
{
    FFTZ_INTP n = decomp_scheme->dims[0].n;
    FFTZ_INT32 is_FFT_ker_supported = check_FFT_kernel_support(
        n, kertab_rdft, !IS_NOT_INNERMOST_DIM(decomp_scheme->flags));

    // Non-direct sizes only (same gate as complex SOLVER_BATCHED_CT_L1_DIRECT).
    if (is_FFT_ker_supported)
    {
        return 0;
    }

    // ST fused only (batch=1 allowed, like complex). MT child fused disabled.
    if (decomp_scheme->thread_info->n_threads != 1 ||
        decomp_scheme->thread_info->avl_threads != 1 ||
        decomp_scheme->thread_info->active_threads != 1)
    {
        return 0;
    }

    if (decomp_scheme->dim_rank != 1 || decomp_scheme->vec_rank != 1)
    {
        return 0;
    }

    if (decomp_scheme->batched_vecs != NULL)
    {
        return 0;
    }

    if (check_col_major(decomp_scheme))
    {
        return 0;
    }

    if (IS_NOT_INNERMOST_DIM(decomp_scheme->flags))
    {
        return 0;
    }

    if (realhelper->is_CT || n != realhelper->problem_size)
    {
        return 0;
    }

    for (FFTZ_INTP kr_base = 0; kr_base < NUM_KERNELS_IN_EACH_CATEGORY;
         kr_base++)
    {
        FFTZ_INTP radix_r = (FFTZ_INTP)kertab_rdft[kr_base].radix;

        if (radix_r == 0)
        {
            break;
        }
        if ((n % radix_r) != 0)
        {
            continue;
        }

        // radix_r is registered by construction. The r2hc, r2hcf and c2c
        // variant tables span an identical radix set, so a single lookup
        // answers for all three kernels the fused stages need, in either
        // direction.
        if (check_FFT_kernel_support(n / radix_r, kertab_rdft, 1))
        {
            return 1;
        }
    }
    return 0;
}

// True when a CT stage resolved to a leaf REAL_DIRECT solver (fusable).
static FFTZ_INT32
is_fusable_one_level_direct_stage(aoclfftz_solution_t *stage_sol)
{
    return stage_sol != NULL &&
           is_solver_real_direct_family(stage_sol->solver->solver_type) &&
           stage_sol->next_sol == NULL;
}

// Link winning CT stage templates onto sol->next_sol -> next_sol.
static FFTZ_INT32 setup_batched_ct_l1_rdft_stage_solutions(
    aoclfftz_solution_t *sol, aoclfftz_solution_t *template_r,
    aoclfftz_solution_t *template_m)
{
    if (sol->next_sol == NULL)
    {
        sol->next_sol = alloc_solution(sol->decomp_scheme->vec_rank,
                                       sol->decomp_scheme->dim_rank);
        if (sol->next_sol == NULL)
        {
            return AOCLFFTZ_MEMORY_FAILURE;
        }
    }

    aoclfftz_solution_t *stage_r = sol->next_sol;
    if (stage_r->next_sol == NULL)
    {
        stage_r->next_sol = alloc_solution(sol->decomp_scheme->vec_rank,
                                           sol->decomp_scheme->dim_rank);
        if (stage_r->next_sol == NULL)
        {
            return AOCLFFTZ_MEMORY_FAILURE;
        }
    }

    aoclfftz_solution_t *stage_m = stage_r->next_sol;
    aoclfftz_solution_t *saved_stage_r_next = stage_r->next_sol;
    FFTZ_INT32 ret = copy_solution_obj(stage_r, template_r);
    if (ret != AOCLFFTZ_SUCCESS)
    {
        return ret;
    }
    stage_r->next_sol = saved_stage_r_next;
    ret = copy_strides(stage_r, template_r);
    if (ret != AOCLFFTZ_SUCCESS)
    {
        return ret;
    }
    ret = copy_solution_obj(stage_m, template_m);
    if (ret != AOCLFFTZ_SUCCESS)
    {
        return ret;
    }
    stage_m->next_sol = NULL;
    ret = copy_strides(stage_m, template_m);
    if (ret != AOCLFFTZ_SUCCESS)
    {
        return ret;
    }

    return AOCLFFTZ_SUCCESS;
}

// Free aux and stage chain left on sol by a failed stage setup so fallback
// selectors start clean.
static FFTZ_VOID release_partial_batched_ct_l1_rdft_sol(
    aoclfftz_solution_t *sol)
{
    if (sol == NULL)
    {
        return;
    }

    release_owned_real_buffered_aux(sol);
    if (sol->next_sol != NULL)
    {
        destroy_solution(sol->next_sol);
        sol->next_sol = NULL;
    }
}

// Tear down a rejected (radix_r, radix_m) trial and reset its cost.
static FFTZ_VOID discard_ct_candidate(aoclfftz_selector_t *cur_sel,
                                      aoclfftz_selector_t *cur_sel_m)
{
    if (cur_sel != NULL && cur_sel->solution != NULL)
    {
        destroy_solution(cur_sel->solution->next_sol);
        cur_sel->solution->next_sol = NULL;
        destroy_strides_grp(cur_sel->solution->strides_grp);
        RESET_COST(cur_sel);
    }
    if (cur_sel_m != NULL && cur_sel_m->solution != NULL)
    {
        destroy_solution(cur_sel_m->solution->next_sol);
        cur_sel_m->solution->next_sol = NULL;
        destroy_strides_grp(cur_sel_m->solution->strides_grp);
        RESET_COST(cur_sel_m);
    }
}

/*
 * Selector for SOLVER_REAL_BATCHED_CT_L1_DIRECT.
 *
 * Search factorizations n = radix_r * radix_m. For each candidate, build trial
 * stage-r and stage-m trees through the same CT selection path as
 * selector_ct_rdft, keep pairs where both stages are fusable REAL_DIRECT
 * leaves, and choose the lowest combined op count. On success, set up the
 * winning stage templates on sol and finish setup via
 * setup_batched_ct_l1_direct_real_solver.
 */
FFTZ_INT32
selector_batched_ct_l1_direct_rdft(aoclfftz_selector_t *sel, kernel_t *kertab,
                                   aoclfftz_realhelper_t *realhelper)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    if (sel == NULL || sel->solution == NULL ||
        sel->solution->decomp_scheme == NULL || sel->kernel_tables == NULL)
    {
        AOCLFFTZ_LOG(INFO, global_logger_mode,
                     "Invalid selector passed to "
                     "selector_batched_ct_l1_direct_rdft");
        return SELECTOR_FAILURE;
    }

    aoclfftz_selector_t *cur_sel = NULL;
    aoclfftz_selector_t *cur_sel_m = NULL;
    aoclfftz_solution_t *org_sol = NULL;
    aoclfftz_solution_t *best_stage_r = NULL;
    aoclfftz_solution_t *best_stage_m = NULL;

    aoclfftz_solution_t *sol = sel->solution;
    FFTZ_INTP n = sol->decomp_scheme->dims[0].n;
    FFTZ_INTP batch = sol->decomp_scheme->vecs[0].n;
    FFTZ_INT32 vec_rank = sol->decomp_scheme->vec_rank;
    FFTZ_INT32 dim_rank = sol->decomp_scheme->dim_rank;
    FFTZ_UINT32 is_backward =
        FFT_DIR(sol->decomp_scheme->flags) == BACKWARD_FFT_DIR;
    FFTZ_INT32 ret = SELECTOR_FAILURE;

    FFTZ_UINT32 radix_r = 0;
    FFTZ_UINT32 radix_m = 0;
    FFTZ_INT64 best_ops = INT64_MAX;
    FFTZ_UINT8 have_best = 0;
    FFTZ_UINT8 is_previous_solution_selected = 0;

    aoclfftz_realhelper_t ct_helper;
    FFTZ_UINT8 nested_on_entry = *(sel->has_nested);
    FFTZ_UINT8 nested_selected = nested_on_entry;

    if (vec_rank != 1 || dim_rank != 1)
    {
        goto exit_batched_ct_l1_rdft;
    }

    org_sol = alloc_solution(vec_rank, dim_rank);
    cur_sel = alloc_selector(vec_rank, dim_rank, sel->kernel_tables,
                             sel->has_nested);
    cur_sel_m = alloc_selector(vec_rank, dim_rank, sel->kernel_tables,
                               sel->has_nested);
    if (org_sol == NULL || cur_sel == NULL || cur_sel_m == NULL)
    {
        ret = AOCLFFTZ_MEMORY_FAILURE;
        goto exit_batched_ct_l1_rdft;
    }

    ret = copy_solution_obj(org_sol, sol);
    if (ret != AOCLFFTZ_SUCCESS)
    {
        goto exit_batched_ct_l1_rdft;
    }
    org_sol->next_sol = NULL;

    for (FFTZ_INTP i = 0; i < NUM_KERNELS_IN_EACH_CATEGORY; i++)
    {
        radix_r = (FFTZ_INTP)kertab[i].radix;
        if (radix_r == 0)
        {
            break;
        }
        if ((n % radix_r) != 0)
        {
            continue;
        }
        radix_m = n / radix_r;

        // ct_helper reset each radix trial (assignments below). Failed trials
        // need no undo before continue — next iteration reruns this block.
        ct_helper = *realhelper;
        ct_helper.is_CT = 1;
        ct_helper.is_last_stage = 0;
        ct_helper.stage = 0;

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
                goto exit_batched_ct_l1_rdft;
            }
            is_previous_solution_selected = 0;
        }

        *(sel->has_nested) = nested_on_entry;

        ret = setup_real_ct_solver(org_sol, cur_sel->solution,
                                   cur_sel_m->solution, radix_r, radix_m,
                                   &ct_helper);
        if (ret != SELECTOR_SUCCESS)
        {
            discard_ct_candidate(cur_sel, cur_sel_m);
            continue;
        }

        ct_helper.is_last_stage = 1;
        ct_helper.is_buffered_invoked = 1;
        ct_helper.stage++;
        if (is_backward)
        {
            ct_helper.freq_factor /= radix_r;
        }
        else
        {
            ct_helper.freq_factor *= radix_r;
        }

        ret = selector_model_rdft_(cur_sel_m, &ct_helper);
        if (ret != SELECTOR_SUCCESS)
        {
            discard_ct_candidate(cur_sel, cur_sel_m);
            continue;
        }

        if (is_backward)
        {
            ct_helper.freq_factor *= radix_r;
        }
        else
        {
            ct_helper.freq_factor /= radix_r;
        }
        ct_helper.stage--;
        ct_helper.is_last_stage = 0;

        ret = selector_model_rdft_(cur_sel, &ct_helper);
        if (ret != SELECTOR_SUCCESS)
        {
            discard_ct_candidate(cur_sel, cur_sel_m);
            continue;
        }

        if (!is_fusable_one_level_direct_stage(cur_sel->solution) ||
            !is_fusable_one_level_direct_stage(cur_sel_m->solution))
        {
            discard_ct_candidate(cur_sel, cur_sel_m);
            continue;
        }

        FFTZ_INT64 cur_ops =
            cur_sel->cost_analysis->ops + cur_sel_m->cost_analysis->ops;
        if (cur_ops < best_ops)
        {
            best_ops = cur_ops;

            if (best_stage_r != NULL)
            {
                destroy_solution(best_stage_r);
                best_stage_r = NULL;
            }
            if (best_stage_m != NULL)
            {
                destroy_solution(best_stage_m);
                best_stage_m = NULL;
            }

            best_stage_r = alloc_solution(vec_rank, dim_rank);
            best_stage_m = alloc_solution(vec_rank, dim_rank);
            if (best_stage_r == NULL || best_stage_m == NULL)
            {
                ret = AOCLFFTZ_MEMORY_FAILURE;
                goto exit_batched_ct_l1_rdft;
            }

            ret = copy_solution_obj(best_stage_r, cur_sel->solution);
            if (ret != AOCLFFTZ_SUCCESS)
            {
                goto exit_batched_ct_l1_rdft;
            }
            ret = copy_strides(best_stage_r, cur_sel->solution);
            if (ret != AOCLFFTZ_SUCCESS)
            {
                goto exit_batched_ct_l1_rdft;
            }

            ret = copy_solution_obj(best_stage_m, cur_sel_m->solution);
            if (ret != AOCLFFTZ_SUCCESS)
            {
                goto exit_batched_ct_l1_rdft;
            }
            ret = copy_strides(best_stage_m, cur_sel_m->solution);
            if (ret != AOCLFFTZ_SUCCESS)
            {
                goto exit_batched_ct_l1_rdft;
            }

            have_best = 1;
            is_previous_solution_selected = 1;
            nested_selected = *(sel->has_nested);
        }
        else
        {
            discard_ct_candidate(cur_sel, cur_sel_m);
        }
    }

    if (!have_best)
    {
        ret = SELECTOR_FAILURE;
        goto exit_batched_ct_l1_rdft;
    }

    ret = setup_batched_ct_l1_rdft_stage_solutions(sol, best_stage_r,
                                                     best_stage_m);
    if (ret != AOCLFFTZ_SUCCESS)
    {
        release_partial_batched_ct_l1_rdft_sol(sol);
        goto exit_batched_ct_l1_rdft;
    }

    ret = setup_batched_ct_l1_direct_real_solver(sol);
    if (ret != SOLVER_SUCCESS)
    {
        release_partial_batched_ct_l1_rdft_sol(sol);
        goto exit_batched_ct_l1_rdft;
    }

    sel->cost_analysis->ops = best_ops * batch;
    sel->cost_analysis->time = 0;
    *(sel->has_nested) = nested_selected;
    ret = SELECTOR_SUCCESS;

exit_batched_ct_l1_rdft:
    if (ret != SELECTOR_SUCCESS)
    {
        *(sel->has_nested) = nested_on_entry;
    }
    destroy_selector(cur_sel);
    destroy_selector(cur_sel_m);
    destroy_solution(org_sol);
    if (best_stage_r != NULL)
    {
        destroy_solution(best_stage_r);
    }
    if (best_stage_m != NULL)
    {
        destroy_solution(best_stage_m);
    }
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return ret;
}
