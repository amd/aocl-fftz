// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file direct_solver_rdft_utils.c
 *
 *  @brief Direct Solver helper functions required for the setup of direct
 * solver
 *
 *  @author Partiksha
 */

#include "core/common/memory_manager.h"
#include "core/common/twiddle.h"
#include "core/solvers/real/strides_rdft.h"
#include "core/solvers/real/direct_solver_rdft_utils.h"

FFTZ_VOID setup_rdft_dc_nyquist_offsets_ds(aoclfftz_decomp_scheme_t *ds)
{
    if (FFT_DIR(ds->flags) != FORWARD_FFT_DIR)
    {
        return;
    }

    FFTZ_INTP transform_len = ds->dims[0].n;
    FFTZ_INTP out_stride = ds->dims[0].out_stride;
    FFTZ_INTP batch = ds->vecs[0].n;
    FFTZ_INTP transform_len_total = transform_len * batch;

    // R2C half-complex output stores Nyquist imag at index n*out_stride+1
    // (interleaved real/imag) when n is even; for odd n it folds onto DC
    // (offset 1). nyquist_im_offset_direct is per transform (direct R2C zero);
    // nyquist_im_offset_ct spans the full batch (CT last-stage zero).
    ds->nyquist_im_offset_direct =
        transform_len % 2 == 0 ? transform_len * out_stride + 1 : 1;
    ds->nyquist_im_offset_ct =
        transform_len_total % 2 == 0 ? transform_len_total * out_stride + 1
                                     : 1;
}

// Configure and sets the count values for different kernel types
// (c2c, r2hc, r2hcf) based on the problem parameters.
FFTZ_VOID set_kernel_count_in_each_group(aoclfftz_solution_t *sol,
                                         aoclfftz_realhelper_t *realhelper)
{
    FFTZ_UINT32 is_backward =
        FFT_DIR(sol->decomp_scheme->flags) == BACKWARD_FFT_DIR;
    FFTZ_INTP radix = sol->decomp_scheme->dims[0].n;
    FFTZ_INTP batch = sol->decomp_scheme->vecs[0].n;

    FFTZ_INTP num_groups = realhelper->problem_size / realhelper->freq_factor;
    num_groups = is_backward ? num_groups : num_groups / radix;
    FFTZ_INTP prev_freq_factor= is_backward
                           ? realhelper->freq_factor / radix
                           : realhelper->freq_factor;
    FFTZ_UINT32 is_even = prev_freq_factor% 2 == 0;

    if (realhelper->is_CT)
    {
        sol->solver->kernel_c2c->count = ((prev_freq_factor / 2) - is_even)
                                          * num_groups;
        sol->solver->kernel_r2hc->count = is_even ? 0 : num_groups;
        sol->solver->kernel_r2hcf->count = is_even ? num_groups : 0;
    }
    else // direct problem
    {
        sol->solver->kernel_c2c->count = 0;
        sol->solver->kernel_r2hc->count = batch;
        sol->solver->kernel_r2hcf->count = 0;
    }
}

// Allocate and set up stride arrays for different kernel types
FFTZ_INT32 allocate_and_setup_stride(aoclfftz_solution_t *sol,
                               aoclfftz_realhelper_t realhelper)
{
    FFTZ_INTP radix = sol->decomp_scheme->dims[0].n;
    FFTZ_INTP num_groups = NUM_RFFT_GROUPS(sol->solver);
    FFTZ_INTP num_c2c_per_group = sol->solver->kernel_c2c->count / num_groups;
    FFTZ_UINT8 use_asymmetric_kernel = num_c2c_per_group >= num_groups;
    base_strides_t element_strides = {1, 1};      // Individual element access
    base_strides_t vector_strides = {1, 1};       // Vector/batch traversal
    base_strides_t c2c_strides = {1, 1};          // C2C kernel batch stepping

    // Set Element strides
    if (realhelper.is_CT)
    {
        set_ct_base_strides(sol, realhelper, &element_strides, &vector_strides,
                            &c2c_strides);
    }
    else
    {
        set_base_strides(sol, &element_strides, &vector_strides);
    }

    FFTZ_INT32 ret = alloc_stride_arrays(sol->strides_grp->strides, radix);
    if (ret != SOLVER_SUCCESS)
    {
        return ret;
    }
    // Setup for C2C kernels if needed
    if (sol->solver->kernel_c2c->count != 0)
    {
        ret = setup_c2c_stride_arrays(sol, realhelper, element_strides,
                                      c2c_strides, num_groups,
                                      sol->solver->kernel_r2hcf->count != 0,
                                      num_c2c_per_group);
        if (ret != SOLVER_SUCCESS)
        {
            return ret;
        }
    }

    // Setup for R2HC kernels if needed
    if (sol->solver->kernel_r2hc->count != 0)
    {
        ret = setup_r2hc_stride_arrays(sol, realhelper, element_strides,
                                       num_groups, num_c2c_per_group);
        if (ret != SOLVER_SUCCESS)
        {
            return ret;
        }
    }

    // Setup for R2HCF kernels if needed
    if (sol->solver->kernel_r2hcf->count != 0)
    {
        ret = setup_r2hcf_stride_arrays(sol, realhelper, element_strides);
        if (ret != SOLVER_SUCCESS)
        {
            return ret;
        }
    }

    set_vector_strides_for_kernels(sol, vector_strides, c2c_strides,
                                   use_asymmetric_kernel, realhelper,
                                   sol->solver->kernel_r2hcf->count != 0);
    setup_rdft_dc_nyquist_offsets_ds(sol->decomp_scheme);
    return SOLVER_SUCCESS;
}

/** Record the aux I/O roles of a Direct node in a CT problem
 *
 * The roles come from the node's position in the CT decomposition, independent
 * of the node ordering in the chain. No buffer is bound here: the tree stays
 * read-only and the pointers are resolved per call from the roles. The
 * alternation between the two aux pools is added during execution, where every
 * stage swaps them in ctx once its kernels are done. In recursive mode the
 * chain mirrors the complex CT tree:
 * ... -> buffered -> CT -> direct(r) -> [CT -> direct]* -> direct(last)
 *
 * Here, the buffered holds the in & out of the current batch. The roles and the
 * per-stage swap together yield the following data flow (3-level CT example):
 *
 * buffered           [in -> out]
 * |--> direct(r)       [in   -> aux2]   (stage 0: in = problem input)
 * |--> direct(mid)     [aux2 -> aux1]   (stage 1)
 * |--> direct(last)    [aux1 -> out]    (last stage: out = problem output)
 */
FFTZ_VOID update_ct_buffers(aoclfftz_solution_t *sol,
                       aoclfftz_realhelper_t *realhelper)
{
    if (!realhelper->is_CT)
        return;

    // Record per-node I/O roles so the tree stays read-only. Every stage but the
    // first reads from aux, and every stage but the last writes to aux; the two
    // ends keep REAL_USE_IO_BUF so they use the plan's own in/out buffer.
    sol->decomp_scheme->real_in_role  = (realhelper->stage) > 0 ?
                                        REAL_USE_AUX_AND_SWAP :
                                        REAL_USE_IO_BUF;
    sol->decomp_scheme->real_out_role = realhelper->is_last_stage ?
                                        REAL_USE_IO_BUF :
                                        REAL_USE_AUX_AND_SWAP;
}

/**
 * Calculates the computational cost associated with executing different
 * kernel types (C2C, R2HC, R2HCF) based on their operation counts in
 * fixed selector mode.
 * Cost is measured in CPU cycles and considers the number of FMA, multiply,
 * add, move, permute and other operations.
 */
FFTZ_VOID compute_cost(aoclfftz_solution_t *sol, cost_analysis_t *cost,
                  const kernel_t *kernel_c2c, const kernel_t *kernel_r2hc,
                  const kernel_t *kernel_r2hcf)
{
    FFTZ_UINT32 is_backward =
        FFT_DIR(sol->decomp_scheme->flags) == BACKWARD_FFT_DIR;
    FFTZ_UINT32 precision = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    if (GET_SELECTOR_MODE(sol->decomp_scheme->flags) != AOCLFFTZ_FIXED_SELECTOR)
    {
        return;
    }

    cost->time = 0;
    FFTZ_INT64 c2c_cost = 0;
    FFTZ_INT64 r2hc_cost = 0;
    FFTZ_INT64 r2hcf_cost = 0;

    // Calculate C2C kernel cost
    if (sol->solver->kernel_c2c->count != 0)
    {
        c2c_cost =
            compute_kernel_cost(kernel_c2c, precision, is_backward,
                                (FFTZ_INTP)sol->solver->kernel_c2c->count);
    }

    // Calculate R2HC kernel cost
    if (sol->solver->kernel_r2hc->count != 0)
    {
        r2hc_cost =
            compute_kernel_cost(kernel_r2hc, precision, is_backward,
                                (FFTZ_INTP)sol->solver->kernel_r2hc->count);
    }

    // Calculate R2HCF kernel cost
    if (sol->solver->kernel_r2hcf->count != 0)
    {
        r2hcf_cost =
            compute_kernel_cost(kernel_r2hcf, precision, is_backward,
                                (FFTZ_INTP)sol->solver->kernel_r2hcf->count);
    }

    cost->ops = c2c_cost + r2hc_cost + r2hcf_cost;
}

#ifdef MULTI_THREADING
// Multiple stride copies are required for c2c parallel execution
// This function points the calling thread at its own slot of the C2C stride
FFTZ_VOID real_mt_c2c_thread_stride_slot(FFTZ_VOID *stride_slab,
                                    FFTZ_INTP stride_slot_bytes,
                                    aoclfftz_strides_t **strides_c2c_per_thread,
                                    FFTZ_INTP **local_strides)
{
    FFTZ_CHAR *slot = (FFTZ_CHAR *)stride_slab +
                      (FFTZ_INTP)omp_get_thread_num() * stride_slot_bytes;
    FFTZ_INTP stride_array_offset = (FFTZ_INTP)sizeof(aoclfftz_strides_t);

    *strides_c2c_per_thread = (aoclfftz_strides_t *)slot;
    *local_strides = (FFTZ_INTP *)(slot + stride_array_offset);
}

#endif
