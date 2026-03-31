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
#include "core/common/strides.h"
#include "core/solvers/real/direct_solver_rdft_utils.h"

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

// Calculates and configures the various element strides needed for CT sizes.
static inline FFTZ_VOID set_ct_base_strides(aoclfftz_solution_t *sol,
            aoclfftz_realhelper_t realhelper, base_strides_t *element_strides,
            base_strides_t *vector_strides, base_strides_t *c2c_stride)
{
    FFTZ_INTP batch = sol->decomp_scheme->vecs[0].n;
    FFTZ_INTP radix = sol->decomp_scheme->dims[0].n;
    FFTZ_UINT32 is_backward =
        FFT_DIR(sol->decomp_scheme->flags) == BACKWARD_FFT_DIR;
    FFTZ_UINT32 is_first_stage = realhelper.stage == 0;
    FFTZ_UINT32 is_last_stage = realhelper.is_last_stage;
    FFTZ_INTP is_input_problem_buffer = is_backward && is_first_stage;
    FFTZ_INTP is_output_problem_buffer = !is_backward && is_last_stage;

    FFTZ_INTP freq_factor       = is_backward
                           ? realhelper.freq_factor
                           : realhelper.freq_factor * radix;
    FFTZ_INTP prev_freq_factor = freq_factor / radix;

    base_strides_t  org_stride = {1, 1};
    org_stride.in_stride = sol->decomp_scheme->dims[0].in_stride;
    org_stride.out_stride = sol->decomp_scheme->dims[0].out_stride;

    element_strides->in_stride = is_backward ? prev_freq_factor : batch;
    element_strides->out_stride = is_backward ? batch : prev_freq_factor;

    vector_strides->in_stride = is_backward ? freq_factor : prev_freq_factor;
    vector_strides->out_stride = is_backward ? prev_freq_factor : freq_factor;

    if (is_first_stage)
    {
        element_strides->in_stride *= org_stride.in_stride;
        vector_strides->in_stride  *= org_stride.in_stride;
    }
    else if (is_last_stage)
    {
        element_strides->out_stride *= org_stride.out_stride;
        vector_strides->out_stride  *= org_stride.out_stride;
    }

    c2c_stride->in_stride  = is_input_problem_buffer ? org_stride.in_stride : 1;
    c2c_stride->out_stride = is_output_problem_buffer
                           ? org_stride.out_stride : 1;
}

// Set base strides for direct problem size
static inline FFTZ_VOID set_base_strides(aoclfftz_solution_t *sol,
                                    base_strides_t *element_strides,
                                    base_strides_t *vector_strides)
{
    FFTZ_UINT32 is_backward =
        FFT_DIR(sol->decomp_scheme->flags) == BACKWARD_FFT_DIR;
    element_strides->in_stride      = sol->decomp_scheme->dims[0].in_stride;
    element_strides->out_stride     = sol->decomp_scheme->dims[0].out_stride;
    vector_strides->in_stride  = is_backward ?
                            sol->decomp_scheme->vecs[0].in_stride * 2 :
                            sol->decomp_scheme->vecs[0].in_stride;
    vector_strides->out_stride = is_backward ?
                            sol->decomp_scheme->vecs[0].out_stride :
                            sol->decomp_scheme->vecs[0].out_stride * 2;
}

// Set vector strides for different kernel types (c2c, r2hc, r2hcf)
static inline FFTZ_VOID set_vector_strides_for_kernels(
    aoclfftz_solution_t *sol, base_strides_t vector_strides,
    base_strides_t c2c_strides, FFTZ_UINT8 use_asymmetric_kernel)
{
    aoclfftz_strides_grp_t *strides_grp = sol->strides_grp;
    FFTZ_UINT32 is_backward =
        FFT_DIR(sol->decomp_scheme->flags) == BACKWARD_FFT_DIR;

    strides_grp->strides->v_in_stride =
        strides_grp->strides->v_in_h2_stride = vector_strides.in_stride;
    strides_grp->strides->v_out_stride =
        strides_grp->strides->v_out_h2_stride = vector_strides.out_stride;
    if (!use_asymmetric_kernel)
    {
        // The strides within kernel will be symmetric strides
        strides_grp->strides_c2c->v_in_stride =
            strides_grp->strides_c2c->v_in_h2_stride =
                strides_grp->strides->v_in_stride;
        strides_grp->strides_c2c->v_out_stride =
            strides_grp->strides_c2c->v_out_h2_stride =
                strides_grp->strides->v_out_stride;
    }
    else
    {
        // The strides within kernel will be asymmetric strides
        strides_grp->strides_c2c->v_in_stride = 2 * c2c_strides.in_stride;
        strides_grp->strides_c2c->v_out_stride = 2 * c2c_strides.out_stride;
        strides_grp->strides_c2c->v_in_h2_stride =
            strides_grp->strides_c2c->v_in_stride * (is_backward ? -1 : 1);
        strides_grp->strides_c2c->v_out_h2_stride =
            strides_grp->strides_c2c->v_out_stride *
            (is_backward ? 1 : -1);
    }

    strides_grp->strides_r2hc->v_in_stride =
        strides_grp->strides_r2hc->v_in_h2_stride = vector_strides.in_stride;
    strides_grp->strides_r2hc->v_out_stride =
        strides_grp->strides_r2hc->v_out_h2_stride = vector_strides.out_stride;
    strides_grp->strides_r2hcf->v_in_stride =
        strides_grp->strides_r2hcf->v_in_h2_stride = vector_strides.in_stride;
    strides_grp->strides_r2hcf->v_out_stride =
        strides_grp->strides_r2hcf->v_out_h2_stride = vector_strides.out_stride;

}

// Determine complex and half-complex flags for kernel strides
static inline FFTZ_VOID
set_complex_format(aoclfftz_realhelper_t realhelper, FFTZ_UINT32 is_backward,
                   FFTZ_UINT32 *input_adjust_to_full_complex,
                   FFTZ_UINT32 *output_adjust_to_full_complex,
                   FFTZ_UINT32 *compute_half_complex_input,
                   FFTZ_UINT32 *compute_half_complex_output)
{
    FFTZ_UINT32 is_first_stage = realhelper.stage == 0;
    FFTZ_UINT32 is_last_stage = realhelper.is_last_stage;
    if (realhelper.is_CT)
    {
        *input_adjust_to_full_complex = is_backward ? is_first_stage : 0;
        *output_adjust_to_full_complex = is_backward ? 0 : is_last_stage;
    }
    else
    {
        *input_adjust_to_full_complex = is_backward;
        *output_adjust_to_full_complex = !is_backward;
    }

    // forward  : r2hc : real input -> half-complex output
    // backward : hc2r : half-complex input -> real output
    *compute_half_complex_input = is_backward ? 1 : 0;
    *compute_half_complex_output = is_backward ? 0 : 1;
}

// Configure stride arrays for R2HC kernels
static inline FFTZ_INT32 setup_r2hc_stride_arrays(aoclfftz_solution_t *sol,
                                            aoclfftz_realhelper_t realhelper,
                                            base_strides_t element_strides)
{
    FFTZ_INTP radix = sol->decomp_scheme->dims[0].n;
    FFTZ_UINT32 is_backward =
        FFT_DIR(sol->decomp_scheme->flags) == BACKWARD_FFT_DIR;

    // 1 if the stride values needs to be adjusted to full complex format
    FFTZ_UINT32 input_adjust_to_full_complex = 0;
    FFTZ_UINT32 output_adjust_to_full_complex = 0;

    // 1 for half-complex, 0 for real & complex data
    FFTZ_UINT32 compute_half_complex_input, compute_half_complex_output;

    set_complex_format(realhelper, is_backward,
                &input_adjust_to_full_complex, &output_adjust_to_full_complex,
                &compute_half_complex_input, &compute_half_complex_output);

    FFTZ_INT32 ret = alloc_stride_arrays(sol->strides_grp->strides_r2hc, radix);
    if (ret != SOLVER_SUCCESS)
    {
        return ret;
    }
    populate_stride_array(sol->strides_grp->strides_r2hc->in_strides,
                element_strides.in_stride, radix, compute_half_complex_input,
                input_adjust_to_full_complex);
    populate_stride_array(sol->strides_grp->strides_r2hc->out_strides,
                element_strides.out_stride, radix, compute_half_complex_output,
                output_adjust_to_full_complex);
    return SOLVER_SUCCESS;
}

// Configure stride arrays for R2HCF kernels
static inline FFTZ_INT32 setup_r2hcf_stride_arrays(aoclfftz_solution_t *sol,
                                      aoclfftz_realhelper_t realhelper,
                                      base_strides_t element_strides)
{
    FFTZ_INTP radix = sol->decomp_scheme->dims[0].n;
    FFTZ_UINT32 is_backward =
        FFT_DIR(sol->decomp_scheme->flags) == BACKWARD_FFT_DIR;

    // number of groups and no. of c2c kernel calls per group
    FFTZ_INTP num_groups = NUM_RFFT_GROUPS(sol->solver);
    FFTZ_INTP num_c2c_per_group = sol->solver->kernel_c2c->count / num_groups;

    // 1 if the stride values needs to be adjusted to full complex format
    FFTZ_UINT32 input_adjust_to_full_complex = 0;
    FFTZ_UINT32 output_adjust_to_full_complex = 0;

    // 1 for half-complex, 0 for real & complex data
    FFTZ_UINT32 compute_half_complex_input, compute_half_complex_output;

    set_complex_format(realhelper, is_backward,
                &input_adjust_to_full_complex, &output_adjust_to_full_complex,
                &compute_half_complex_input, &compute_half_complex_output);

    FFTZ_INT32 ret =
        alloc_stride_arrays(sol->strides_grp->strides_r2hcf, radix * 2);
    if (ret != SOLVER_SUCCESS)
    {
        return ret;
    }
    populate_stride_array(sol->strides_grp->strides_r2hcf->in_strides,
                is_backward ? element_strides.in_stride / 2
                            : element_strides.in_stride,
                is_backward ? radix * 2 : radix,
                compute_half_complex_input, input_adjust_to_full_complex);
    populate_stride_array(sol->strides_grp->strides_r2hcf->out_strides,
                is_backward ? element_strides.out_stride
                            : element_strides.out_stride / 2,
                is_backward ? radix : radix * 2,
                compute_half_complex_output, output_adjust_to_full_complex);

    prepare_fused_kernel_strides(is_backward ?
                sol->strides_grp->strides_r2hcf->out_strides :
                sol->strides_grp->strides_r2hcf->in_strides,
                radix, num_c2c_per_group * 2 + 1);
    return SOLVER_SUCCESS;
}

// Configure stride arrays for complex-to-complex (C2C) kernels
static inline FFTZ_INT32 setup_c2c_stride_arrays(aoclfftz_solution_t *sol,
                                           aoclfftz_realhelper_t realhelper,
                                           base_strides_t element_strides,
                                           base_strides_t c2c_stride)
{
    FFTZ_UINT32 is_backward =
        FFT_DIR(sol->decomp_scheme->flags) == BACKWARD_FFT_DIR;
    FFTZ_INTP radix = sol->decomp_scheme->dims[0].n;
    FFTZ_INTP freq_factor = is_backward
                      ? realhelper.freq_factor
                      : realhelper.freq_factor * radix;

    populate_stride_array(sol->strides_grp->strides->in_strides,
                          is_backward ? element_strides.in_stride * 2
                                      : element_strides.in_stride,
                          radix, 0, 0); /* half-complex flags are false */
    populate_stride_array(sol->strides_grp->strides->out_strides,
                          is_backward ? element_strides.out_stride
                                      : element_strides.out_stride * 2,
                          radix, 0, 0); /* half-complex flags are false */

    FFTZ_INT32 ret = alloc_stride_arrays(sol->strides_grp->strides_c2c, radix);
    if (ret != SOLVER_SUCCESS)
    {
        return ret;
    }
    memcpy(sol->strides_grp->strides_c2c->in_strides,
           sol->strides_grp->strides->in_strides, radix * sizeof(FFTZ_INTP));
    memcpy(sol->strides_grp->strides_c2c->out_strides,
           sol->strides_grp->strides->out_strides, radix * sizeof(FFTZ_INTP));

    // set frequency strides and call to prepare strides once
    FFTZ_INTP *target_stride_array = is_backward
                              ? sol->strides_grp->strides->in_strides
                              : sol->strides_grp->strides->out_strides;
    FFTZ_INTP c2c_batch_stride   = is_backward
                              ? c2c_stride.in_stride
                              : c2c_stride.out_stride;
    prepare_real_c2c_kernel_strides(target_stride_array, target_stride_array,
                                    radix, freq_factor, c2c_batch_stride);
    return SOLVER_SUCCESS;
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
        ret = setup_c2c_stride_arrays(sol, realhelper, element_strides, c2c_strides);
        if (ret != SOLVER_SUCCESS)
        {
            return ret;
        }
    }

    // Setup for R2HC kernels if needed
    if (sol->solver->kernel_r2hc->count != 0)
    {
        ret = setup_r2hc_stride_arrays(sol, realhelper, element_strides);
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
                                   use_asymmetric_kernel);
    return SOLVER_SUCCESS;
}

/** Update the in/out buffers of direct solution for CT problem
 *
 * This routes the single-thread aux_buffer_1/2 for each Direct node based on
 * its CT decomposition stage (parity), independent of the node ordering in the
 * chain. In recursive mode the chain mirrors the complex CT tree:
 * ... -> buffered -> CT -> direct(r) -> [CT -> direct]* -> direct(last)
 *
 * Here, the buffered holds the in & out of the current batch. The stage-parity
 * assignment yields the following Direct data flow (3-level CT example):
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

    FFTZ_UINT32 dt_bytes = SOL_DT_SIZE(sol);

    FFTZ_VOID *in_real = NULL;
    FFTZ_VOID *out_real = NULL;

    if (realhelper->stage & 0x1)
    {
        in_real = sol->dft_bufs->buffered->aux_buffer_2;
        out_real = sol->dft_bufs->buffered->aux_buffer_1;
    }
    else
    {
        in_real = sol->dft_bufs->buffered->aux_buffer_1;
        out_real = sol->dft_bufs->buffered->aux_buffer_2;
    }

    if (realhelper->stage > 0)
    {
        sol->decomp_scheme->in_real = in_real;
        sol->decomp_scheme->in_imag = MOVE_ADDR(in_real, dt_bytes);
    }
    if (!realhelper->is_last_stage)
    {
        sol->decomp_scheme->out_real = out_real;
        sol->decomp_scheme->out_imag = MOVE_ADDR(out_real, dt_bytes);
    }
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
