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

// Divides out every factor of n that a kernel radix can handle, using the same
// radix table as check_CT_solvability, and returns the remaining residue. The
// residue determines how n must be solved:
//   1         : the radices cover n completely, so CT or direct handles it
//   n         : no radix divides n, so n is a prime or prime power with no
//               kernel and the Bluestein solver handles it standalone
//   otherwise : radices combined with such a prime, e.g. 34 = 2 x 17 or
//               153 = 9 x 17. The prime runs on a Bluestein stage inside the
//               Cooley-Tukey plan.
static FFTZ_INTP kernel_radix_residue(FFTZ_INTP n, kernel_t *kertab)
{
    FFTZ_INTP residue = n;

    // Strip one supported radix at a time until no kernel radix divides the
    // residue
    while (check_CT_solvability(residue, kertab))
    {
        for (FFTZ_INTP i = 0; i < NUM_KERNELS_IN_EACH_CATEGORY; i++)
        {
            FFTZ_UINT32 radix = kertab[i].radix;
            if (radix == 0) // End of suitable kernels in the list
            {
                break;
            }
            // Divide out this radix if it factorizes the residue
            if (radix > 1 && (residue % (FFTZ_INTP)radix) == 0)
            {
                residue /= (FFTZ_INTP)radix;
                break;
            }
        }
    }

    return residue;
}

// Returns the kernel-unsupported factor of n that must run on a Bluestein
// stage inside a CT plan, or 0 if n has no such factor. The factor may be a
// prime or a product of no-kernel primes (e.g. 323 = 17 x 19). The rest
// (n / residue) is kernel-factorable and is what the CT radix search peels off.
static FFTZ_INTP ct_bluestein_residue(FFTZ_INTP n, kernel_t *kertab)
{
    FFTZ_INTP residue = kernel_radix_residue(n, kertab);

    return (residue != 1 && residue != n) ? residue : 0;
}

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
    sel->solution->next_sol = alloc_solution(vec_rank, dim_rank);
    if (sel->solution->next_sol == NULL)
    {
        ret = AOCLFFTZ_MEMORY_FAILURE;
        goto exit_ct_dft;
    }

    aoclfftz_solution_t *stage_r = sel->solution->next_sol;
    stage_r->next_sol = alloc_solution(vec_rank, dim_rank);
    if (stage_r->next_sol == NULL)
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
    ret = SELECTOR_FAILURE;

    // Flag to store whether the previous solution is selected
    // based on minimum ops cost
    FFTZ_UINT8 is_previous_solution_selected = 0;

    // Radix candidates below are speculative: the losing ones are thrown away,
    // so their sub-solvers must not leave the plan-wide nested-parallel flag
    // set. Every candidate starts from the same baseline and only the winning
    // candidate's contribution survives the loop.
    FFTZ_UINT8 nested_on_entry = *(sel->has_nested);
    FFTZ_UINT8 nested_selected = nested_on_entry;

    // N = radix_r * radix_m. Cooley-Tukey puts a twiddle W_N between the two
    // stages, and one of them must apply it. The big prime runs on a Bluestein,
    // which can't fold in a twiddle -- so we put the prime on the real-signal
    // stage and let the small cofactor apply the twiddle for free in its fused
    // C2C kernel. The Bluestein then stays twiddle-free and runs on real data
    // (half the cost). Doing it the other way would need a separate O(N)
    // twiddle pass on complex data -- far worse. Forward reads the signal at
    // radix-r (head), backward writes it at radix-m (tail); the prime goes
    // there and the search factors the rest into the other stage.
    FFTZ_INTP bs_residue = ct_bluestein_residue(n, kertab);
    FFTZ_INTP cofactor = bs_residue ? (n / bs_residue) : n;
    FFTZ_UINT8 is_fwd_composite = !is_backward && (bs_residue != 0);

    for (FFTZ_INTP i = 0; i < NUM_KERNELS_IN_EACH_CATEGORY; i++)
    {
        radix_r = (FFTZ_INTP)kertab[i].radix;

        if (radix_r == 0) // End of suitable kernels in the list
        {
            break;
        }

        // A radix need only divide the cofactor, not peel it whole: a partial
        // peel leaves a combine that still carries the prime, which this
        // routine decomposes one level further down (a nested CT). The cost
        // comparison then picks between the whole-peel and nested shapes.
        if ((cofactor % radix_r) != 0)
        {
            continue;
        }

        // choose the other radix m
        radix_m = n / radix_r;

        // Forward composite forces a single split: prime at the head (radix-r),
        // whole cofactor as the combine (radix-m). The loop breaks after
        // building it (the radix check above only confirmed the cofactor is
        // factorable).
        if (is_fwd_composite)
        {
            radix_r = bs_residue;
            radix_m = n / bs_residue;
        }

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

        ret = setup_real_ct_solver(org_sol, cur_sel->solution,
                                   cur_sel_m->solution, radix_r, radix_m,
                                   realhelper);
        if (ret != SELECTOR_SUCCESS)
        {
            goto exit_ct_dft;
        }

        // Mark the bare prime stage (radix-m tail for backward, radix-r head
        // for forward) as packed, so it exchanges the CT stage's packed
        // sub-spectra and picks up the replaced Direct's strides. A
        // still-composite backward combine is a nested CT instead.
        if (is_backward && (FFTZ_INTP)radix_m == bs_residue)
        {
            SET_BLUESTEIN_CT_STAGE(cur_sel_m->solution->decomp_scheme->flags);
        }
        else if (is_fwd_composite)
        {
            SET_BLUESTEIN_CT_STAGE(cur_sel->solution->decomp_scheme->flags);
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

                aoclfftz_solution_t *stage_m = get_next_real_stage(stage_r);
                if (stage_m == NULL)
                {
                    goto exit_ct_dft;
                }
                // Destroy the solutions except the first 3 objects
                // since it points to current CT, CT-R, CT-M respectively
                if (stage_m->next_sol != NULL)
                {
                    destroy_solution(stage_m->next_sol);
                    stage_m->next_sol = NULL;
                }
                ret = copy_solution_obj(stage_r, cur_sel->solution);
                if (ret != AOCLFFTZ_SUCCESS)
                {
                    AOCLFFTZ_ERROR("copy_solution_obj failed: %s",
                                   get_status_string(ret));
                    goto exit_ct_dft;
                }
                ret = copy_strides(stage_r, cur_sel->solution);
                if (ret != AOCLFFTZ_SUCCESS)
                {
                    AOCLFFTZ_ERROR("copy_strides failed: %s",
                                   get_status_string(ret));
                    goto exit_ct_dft;
                }
                // Restore the CT continuation after the copied radix-r stage.
                // A packed prime is Batched -> Bluestein, so its continuation
                // belongs after the worker rather than directly on the wrapper.
                set_next_real_stage(stage_r, stage_m);
                // copy_solution_obj does not copy complex_sol, so move it by
                // hand. Free the old child first (else re-selection leaks it).
                destroy_solution(stage_r->dft_bufs->complex_sol);
                stage_r->dft_bufs->complex_sol =
                    cur_sel->solution->dft_bufs->complex_sol;
                cur_sel->solution->dft_bufs->complex_sol = NULL;
                ret = copy_solution_obj(stage_m, cur_sel_m->solution);
                if (ret != AOCLFFTZ_SUCCESS)
                {
                    AOCLFFTZ_ERROR("copy_solution_obj failed: %s",
                                   get_status_string(ret));
                    goto exit_ct_dft;
                }
                ret = copy_strides(stage_m, cur_sel_m->solution);
                if (ret != AOCLFFTZ_SUCCESS)
                {
                    AOCLFFTZ_ERROR("copy_strides failed: %s",
                                   get_status_string(ret));
                    goto exit_ct_dft;
                }
                // Same for radix-m: free the old child, then move the new one.
                destroy_solution(stage_m->dft_bufs->complex_sol);
                stage_m->dft_bufs->complex_sol =
                    cur_sel_m->solution->dft_bufs->complex_sol;
                cur_sel_m->solution->dft_bufs->complex_sol = NULL;

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
                destroy_solution(cur_sel->solution->next_sol);
                cur_sel->solution->next_sol = NULL;
                destroy_solution(cur_sel_m->solution->next_sol);
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

        // Forward composite has a single legal split (prime at the head, whole
        // cofactor combining after), already built above, so stop rather than
        // rebuilding the same shape for every remaining kernel radix.
        if (is_fwd_composite)
        {
            break;
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
