// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file selector_batched_ct_l1_direct_dft.c
 *
 *  @brief Selector for the fused batched + 1-level CT solver.
 *
 *  For batched problems whose inner FFT decomposes into exactly one CT
 *  level (n = radix_r * radix_m, radix_r from twiddle kernel table, radix_m from
 *  standard FFT kernel table), this selector:
 *
 *    1. Iterates over candidate factorizations n = radix_r * radix_m
 *    2. Evaluates kernel cost across all ISA categories for each
 *       candidate factorization and picks the cheapest pair
 *    3. Delegates buffer/stride/kernel setup to
 *       setup_batched_ct_l1_direct_solver (solver.h)
 *
 *  The twiddle post-processing pass (setup_twiddle_buffer_complex) reads
 *  radix_r = kernel_c2c->count and radix_m = kernel_c2c_r->count and
 *  writes twiddle factors into sol->twiddle.
 *
 *  @author Niranjan Reddy
 */

#include "selector/selector.h"

/* Tracks a candidate kernel's table index and its estimated cost. */
typedef struct
{
    INTP  idx;
    INT64 cost;
} kernel_choice_t;

/* Return the base-category slot index whose radix matches, or -1 if none. */
static INTP find_radix_base_idx(kernel_t *kertab, INTP radix)
{
    for (INTP base_idx = 0; base_idx < NUM_KERNELS_IN_EACH_CATEGORY; base_idx++)
    {
        if (kertab[base_idx].radix == 0) // End of suitable kernels in the list
        {
            break;
        }

        if ((INTP)kertab[base_idx].radix == radix)
        {
            return base_idx;
        }
    }
    return -1;
}

/* Pick the optimal kernel across all ISA categories for a given radix & batch size. */
static kernel_choice_t find_best_kernel(kernel_t *kertab, INTP base_idx,
                                      UINT8 precision, UINT8 direction,
                                      INTP batch)
{
    kernel_choice_t optimal = {-1, INT64_MAX};

    for (INTP kcat = 0; kcat < NUM_KERNEL_CATEGORIES; kcat++)
    {
        INTP kloc = kcat * NUM_KERNELS_IN_EACH_CATEGORY + base_idx;
        if (kertab[kloc].kfft[direction] == NULL)
        {
            continue;
        }

        INT64 cost = compute_kernel_cost(&kertab[kloc], precision,
                                         direction, batch);
        if (cost < optimal.cost)
        {
            optimal.idx  = kloc;
            optimal.cost = cost;
        }
    }
    return optimal;
}

INT32 selector_batched_ct_l1_direct_dft(aoclfftz_selector_t *sel)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    if (sel == NULL || sel->solution == NULL ||
        sel->solution->decomp_scheme == NULL || sel->kernel_tables == NULL ||
        sel->kernel_tables->kt_twid_dft == NULL ||
        sel->kernel_tables->kt_dft == NULL)
    {
        AOCLFFTZ_LOG(INFO, global_logger_mode,
                     "Invalid selector or solution passed to "
                     "selector_batched_ct_l1_direct_dft");
        return SELECTOR_FAILURE;
    }

    aoclfftz_solution_t *sol = sel->solution;
    INTP n = sol->decomp_scheme->dims[0].n;
    INTP batch = sol->decomp_scheme->vecs[0].n;
    UINT8 precision = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    UINT8 direction = FFT_DIR(sol->decomp_scheme->flags);
    INT32 ret = SELECTOR_FAILURE;

    kernel_t *kertab_twid = sel->kernel_tables->kt_twid_dft;
    kernel_t *kertab_dft  = sel->kernel_tables->kt_dft;

    INT64 best_ops = INT64_MAX;
    kernel_choice_t best_kr = {-1, INT64_MAX};
    kernel_choice_t best_km = {-1, INT64_MAX};
    INTP best_radix_r = 0;
    INTP best_radix_m = 0;

    for (INTP kr_base = 0; kr_base < NUM_KERNELS_IN_EACH_CATEGORY; kr_base++)
    {
        INTP radix_r = (INTP)kertab_twid[kr_base].radix;

        if (radix_r == 0) // End of suitable kernels in the list
        {
            break;
        }

        // Check if this radix can factorize the problem
        if ((n % radix_r) != 0)
        {
            continue;
        }

        INTP radix_m = n / radix_r;

        INTP km_base = find_radix_base_idx(kertab_dft, radix_m);
        if (km_base < 0)
        {
            continue;
        }

        kernel_choice_t kr = find_best_kernel(kertab_twid, kr_base, precision,
                                            direction, radix_m);
        kernel_choice_t km = find_best_kernel(kertab_dft, km_base, precision,
                                            direction, radix_r);
        if (kr.idx < 0 || km.idx < 0)
        {
            continue;
        }

        INT64 cur_ops = kr.cost + km.cost;
        if (cur_ops < best_ops)
        {
            best_ops     = cur_ops;
            best_kr      = kr;
            best_km      = km;
            best_radix_r = radix_r;
            best_radix_m = radix_m;
        }
    }

    if (best_kr.idx < 0)
    {
        goto exit_batched_ct_l1;
    }

    ret = setup_batched_ct_l1_direct_solver(sol,
                                             &kertab_dft[best_km.idx],
                                             &kertab_twid[best_kr.idx],
                                             best_radix_r, best_radix_m);
    if (ret != SOLVER_SUCCESS)
    {
        goto exit_batched_ct_l1;
    }

    sel->cost_analysis->ops = best_ops * batch;
    sel->cost_analysis->time = 0;
    ret = SELECTOR_SUCCESS;

exit_batched_ct_l1:
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return ret;
}
