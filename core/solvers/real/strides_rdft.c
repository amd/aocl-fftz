// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file strides_rdft.c
 *
 *  @brief Stride utility functions for the real FFT solvers.
 *
 *  This file contains the function definitions that compute and manipulate
 *  stride array values for real, complex and half-complex data, and that set
 *  up the strides a real solution hands to its R2HC, R2HCF and C2C kernels,
 *  for the user buffer as well as the regrouped aux buffer layout.
 *
 *  @author Srirammaswamy Srinivasan
 *  @author Partiksha
 */

#include <string.h>
#include "core/common/memory_manager.h"
#include "core/solvers/real/strides_rdft.h"

// Element stride for standard points of R2HC fused kernel as they are separated
// by 2 complex points(real and imaginary parts) of c2c kernels, hence multiplying by 4.
#define STANDARD_ELEMENT_STRIDE(num_c2c_per_group) (num_c2c_per_group * 4)

/**
 * Prepare stride array values for real/complex and half-complex data
 *
 * Example with stride_val = 2 and n = 6
 *
 * Case 1: With compute_half_complex = 0 (adjust_to_full_complex doesn't
 * matter):
 *
 * strides array values: 0, 2, 4, 6, 8, 10
 * which represents: 0, x, 2, x, 4, x, 6, x, 8, x, 10
 *
 * Case 2: With compute_half_complex = 1 and adjust_to_full_complex = 0:
 *
 * strides array values: 0, 3, 4, 7, 8, 11
 * which represents: 0*, (x, x), (3, 4), (x, x), (7, 8), (x, x), 11*
 *
 * *here the buffer contains real and complex data,
 *  where the first and last points are real, remaining are complex
 *
 * Case 3: With compute_half_complex = 1  and adjust_to_full_complex = 1:
 *
 * strides array values: 0, 3, 4, 7, 8, 11
 * which represents: (0, -*), (x, x), (4, 5), (x, x), (8, 9), (x, x), (12, -*)
 *
 *  *here the buffer contains only complex data, so the first and last real
 *   numbers are stored in complex form, hence the indices are adjusted
 *   accordingly
 *
 * @param strides stride array to the prepared
 * @param stride_val stride value to be populated
 * @param n stride array size
 * @param compute_half_complex 1 for half-complex, 0 for real & complex data
 * @param adjust_to_full_complex 1 if the stride values needs to be adjusted to
 *                               full complex format, 0 otherwise
 *                               adjust_to_full_complex needs to be set for the
 *                               direct sizes and final level of CT
 * @return FFTZ_VOID
 */
FFTZ_VOID populate_stride_array(FFTZ_INTP *strides, FFTZ_INTP stride_val,
                                FFTZ_INTP n, FFTZ_UINT8 compute_half_complex,
                                FFTZ_UINT8 adjust_to_full_complex)
{
    if (compute_half_complex)
    {
        FFTZ_INTP offset = adjust_to_full_complex ? 1 : 0;
        FFTZ_INTP nby2_ceil = (n + 1) / 2;

        // first stride value
        strides[0] = 0;

        // inbetween stride values
        for (FFTZ_INTP i = 1; i < nby2_ceil; i++)
        {
            FFTZ_INTP cur_stride = i * stride_val * 2 + offset;
            strides[i * 2 - 1] = cur_stride - 1;
            strides[i * 2] = cur_stride;
        }

        // last stride value
        if (n % 2 == 0)
        {
            strides[n - 1] = stride_val * n - 1 + offset;
        }
    }
    else
    {
        for (FFTZ_INTP i = 0; i < n; i++)
        {
            strides[i] = i * stride_val;
        }
    }
}

/**
 * Prepare stride array values for R2HC kernels in real FFT problems.
 *
 * Used when setting up per-element in/out strides for R2HC kernels. At problem
 * boundaries the layout matches the user buffer; in intermediate CT stages the
 * aux buffer uses a regrouped layout with separate DC/Nyquist slots.
 *
 * Case 1: is_user_buffer == 1
 *    user/problem-buffer layout unchanged. Internally calls populate_stride_array().
 *
 * Case 2: is_user_buffer == 0, compute_half_complex == 0 (regrouped real input)
 *    Example with stride_val = 2 and n = 4:
 *    strides array values: 0, 2, 4, 6
 *
 * Case 3: is_user_buffer == 0, compute_half_complex == 1 (regrouped half-complex)
 *    Example with n = 6 (even radix), num_groups = 2, stride_val = 1:
 *    offset = num_groups * 2 = 4, start = offset, pair_stride = 2
 *    strides array values: 0, 4, 5, 6, 7, 2
 *    which represents: 0*, [--], (4, 5), (6, 7), [--], 2*
 *    *DC at 0 and Nyquist at num_groups; interior conjugate pairs are contiguous
 *    in the interior band starting at offset (even radix) or num_groups (odd).
 *
 *    Example with n = 6, num_groups = 2, stride_val = 4 (later regrouped sub-stage):
 *    start = offset + stride_val = 8, pair_stride = stride_val + 2 = 6
 *    strides array values: 0, 8, 9, 14, 15, 2
 *
 *    When stride_val == 1, interior starts at offset with no extra skip; when
 *    stride_val > 1, one additional stride_val chunk is skipped inside the band.
 *
 * @param strides stride array to be prepared
 * @param stride_val stride value to be populated
 * @param n stride array size (kernel radix)
 * @param compute_half_complex 1 for half-complex, 0 for real data
 * @param adjust_to_full_complex passed through when is_user_buffer == 1
 * @param is_user_buffer 1 for user/problem buffer, 0 for re-group aux buffer
 * @param num_groups number of R2HC/R2HCF groups
 * @return FFTZ_VOID
 */
FFTZ_VOID populate_stride_array_r2hc(FFTZ_INTP *strides, FFTZ_INTP stride_val,
                                     FFTZ_INTP n,
                                     FFTZ_UINT8 compute_half_complex,
                                     FFTZ_UINT8 adjust_to_full_complex,
                                     FFTZ_INT8 is_user_buffer,
                                     FFTZ_INTP num_groups)
{
    if (is_user_buffer)
    {
        populate_stride_array(strides, stride_val, n, compute_half_complex,
                              adjust_to_full_complex);
    }
    else
    {
        if (compute_half_complex)
        {
            FFTZ_INTP nby2_ceil = (n + 1) / 2;
            // dc point
            strides[0] = 0;

            // inbetween stride values: contiguous pairs in the interior band
            FFTZ_INT32 is_even_radix = n % 2 == 0;
            FFTZ_INTP offset = is_even_radix ? num_groups * 2 : num_groups;
            // stage 0 (stride_val == 1): interior starts at offset;
            // later stages skip one num_groups chunk inside the interior band
            FFTZ_INTP start = offset + (stride_val > 1 ? stride_val : 0);
            FFTZ_INTP pair_stride = stride_val + (stride_val > 1 ? 2 : 1);
            if (1 < nby2_ceil)
            {
                strides[1] = start;
                if (2 < n)
                {
                    strides[2] = start + 1;
                }
            }
            for (FFTZ_INTP i = 2; i < nby2_ceil; i++)
            {
                strides[2 * i - 1] = strides[2 * (i - 1) - 1] + pair_stride;
                strides[2 * i] = strides[2 * (i - 1)] + pair_stride;
            }

            // nyquist point
            if (n % 2 == 0)
            {
                strides[n - 1] = num_groups;
            }
        }
        else
        {
            for (FFTZ_INTP i = 0; i < n; i++)
            {
                strides[i] = i * stride_val;
            }
        }
    }
}

/**
 * Prepare stride array values for R2HCF (fused) kernels in real FFT problems.
 *
 * Used when setting up per-element in/out strides for R2HCF kernels. At problem
 * boundaries the layout matches the user buffer; in intermediate CT stages the
 * aux buffer uses a regrouped layout. Unlike R2HC, fused paths may use
 * an interleaved standard/shifted stride order when compute_half_complex == 0.
 *
 * Case 1: is_user_buffer == 1
 *    user/problem-buffer layout unchanged. Internally calls populate_stride_array().
 *
 * Case 2: is_user_buffer == 0, compute_half_complex == 1 (regrouped half-complex)
 *    Example with n = 6 (even radix), num_groups = 2, stride_val = 4:
 *    offset = num_groups * 2 = 4; first pair at offset + stride_val
 *    strides array values: 0, 8, 9, 14, 15, 2
 *    which represents: 0*, [--], (8, 9), (14, 15), [--], 2*
 *    *DC at 0 and Nyquist at num_groups; interior pairs start at offset + stride_val
 *    and advance by (stride_val + 2) after the first pair (stride_val is bumped by 2
 *    before the remaining pairs are filled).
 *
 * Case 3: is_user_buffer == 0, compute_half_complex == 0 (regrouped fused layout)
 *    Example with n = 4, num_groups = 2, stride_val = 2:
 *    even indices (standard kernel):  0,  2,  4,  6
 *    odd indices (shifted kernel):    8, 10, 12, 14
 *    strides array values: 0, 8, 2, 10, 4, 12, 6, 14
 *
 *    Even and odd stride slots hold standard and shifted DFT inputs in interleaved
 *    fused order, which is what the fused R2HCF kernels read directly.
 *
 * @param strides stride array to be prepared
 * @param stride_val stride value to be populated
 * @param n stride array size (kernel radix, or radix * 2 for fused backward paths)
 * @param compute_half_complex 1 for half-complex, 0 for fused full-complex data
 * @param adjust_to_full_complex passed through when is_user_buffer == 1
 * @param is_user_buffer 1 for user/problem buffer, 0 for re-group aux buffer
 * @param num_groups number of R2HC/R2HCF groups
 * @return FFTZ_VOID
 */
FFTZ_VOID populate_stride_array_r2hcf(FFTZ_INTP *strides, FFTZ_INTP stride_val,
                                      FFTZ_INTP n,
                                      FFTZ_UINT8 compute_half_complex,
                                      FFTZ_UINT8 adjust_to_full_complex,
                                      FFTZ_INT8 is_user_buffer,
                                      FFTZ_INTP num_groups)
{
    if (is_user_buffer)
    {
        populate_stride_array(strides, stride_val, n, compute_half_complex,
                              adjust_to_full_complex);
    }
    else
    {
        if (compute_half_complex)
        {
            FFTZ_INTP nby2_ceil = (n + 1) / 2;

            // dc point
            strides[0] = 0;
            // inbetween stride values
            FFTZ_INTP offset = num_groups * 2;
            if (1 < nby2_ceil)
            {
                strides[1] = offset + stride_val;
                if (2 < n)
                {
                    strides[2] = offset + stride_val + 1;
                }
            }
            // stride val doesn't count for itself as offset
            stride_val += 2;
            for (FFTZ_INTP i = 2; i < nby2_ceil; i++)
            {
                strides[2 * i - 1] = strides[2 * (i-1) - 1] + stride_val;
                strides[2 * i] = strides[2 * (i-1)] + stride_val;
            }

            // nyquist point
            if (n % 2 == 0)
            {
                strides[n - 1] = num_groups;
            }
        }
        else
        {
            // dc point
            strides[0] = 0;
            for (FFTZ_INTP i = 1; i < n; i++)
            {
                strides[2 * i] = i * stride_val;
            }
            strides[1] = n * num_groups;
            for (FFTZ_INTP i = 1; i < n; i++)
            {
                strides[2 * i + 1] = n * num_groups + i * stride_val;
            }
        }
    }
}

/**
 * @brief Prepare the stride array for C2C Kernel in Real Problem
 *
 * The stride array of C2C kernel in real problem needs to be rearranged in
 * such a way that the second half of the strides will be reversed to its
 * complex conjugate point.
 *
 * This function will prepare rearranged strides for the first iteration in a
 * CT stage and for the remaining iterations, the strides will be adjusted by
 * subtracting a constant value during execution.
 *
 * Example for radix 6 C2C kernel with stride complex stride 4 (i.e. stride in
 * data = 8):
 *
 * full complex data    : x1------x2------x3------x4------x5------x6
 * half complex buffer  : |----------------------|
 * half complex data    : x1--y6--x2--y4--x3--y3-- (here y refers to complex
 * conjugate of x)
 *
 * full complex strides : 0, 8, 16, 24, 32, 40
 * half complex strides : 0, 8, 16, 20, 12, 4
 *
 * @param strides stride array
 * @param radix radix value i.e. length of the stride array
 * @param n length of the data buffer
 * @param stride stride value for the given buffer
 * @return FFTZ_VOID
 */
FFTZ_VOID prepare_real_c2c_kernel_strides(FFTZ_INTP *in, FFTZ_INTP *out,
                                          FFTZ_INTP radix, FFTZ_INTP n,
                                          FFTZ_INTP stride)
{
    // align stride to complex points
    stride *= 2;
    for (FFTZ_INTP i = 0; i < radix; i++)
    {
        FFTZ_INTP a = in[i] / stride + 1;
        if (a > n / 2)
        {
            a = n - a;
            out[i] = (a - 1) * stride;
        }
        else
        {
            out[i] = in[i];
        }
    }
}

// Calculates and configures the various element strides needed for CT sizes.
FFTZ_VOID set_ct_base_strides(aoclfftz_solution_t *sol,
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
FFTZ_VOID set_base_strides(aoclfftz_solution_t *sol,
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
FFTZ_VOID
set_vector_strides_for_kernels(aoclfftz_solution_t *sol,
                               base_strides_t vector_strides,
                               base_strides_t c2c_strides,
                               FFTZ_UINT8 use_asymmetric_kernel,
                               aoclfftz_realhelper_t realhelper,
                               FFTZ_INTP with_r2hcf)
{
    FFTZ_INTP radix = sol->decomp_scheme->dims[0].n;
    aoclfftz_strides_grp_t *strides_grp = sol->strides_grp;
    FFTZ_UINT32 is_backward =
        FFT_DIR(sol->decomp_scheme->flags) == BACKWARD_FFT_DIR;
    FFTZ_INTP num_groups = NUM_RFFT_GROUPS(sol->solver);

    strides_grp->strides->v_in_stride =
        strides_grp->strides->v_in_sym_stride = vector_strides.in_stride;
    strides_grp->strides->v_out_stride =
        strides_grp->strides->v_out_sym_stride = vector_strides.out_stride;
    if (!use_asymmetric_kernel)
    {
        // Symmetric C2C: kernel loops over NUM_RFFT_GROUPS batches with v_*_stride.
        FFTZ_INTP subtract_real_in_points = (with_r2hcf ? 2 : 1);
        FFTZ_INTP subtract_real_out_points = with_r2hcf ? 2 : 1;
        FFTZ_INTP subtract_c2c_in_points =
            (is_backward && num_groups > 1 && (radix % 2 == 0))
                ? 2
                : subtract_real_in_points;
        strides_grp->strides_c2c->v_in_stride =
        strides_grp->strides_c2c->v_in_sym_stride =
                strides_grp->strides->v_in_stride - subtract_c2c_in_points;
        FFTZ_INTP subtract_c2c_out_points =
            (!is_backward && !realhelper.is_last_stage
             && num_groups > 1 && (radix % 2 == 0))
                ? 2
                : subtract_real_out_points;
        strides_grp->strides_c2c->v_out_stride =
            strides_grp->strides_c2c->v_out_sym_stride =
                strides_grp->strides->v_out_stride - subtract_c2c_out_points;
    }
    else
    {
        // The strides within kernel will be asymmetric strides
        strides_grp->strides_c2c->v_in_stride = 2 * c2c_strides.in_stride;
        strides_grp->strides_c2c->v_out_stride = 2 * c2c_strides.out_stride;
        strides_grp->strides_c2c->v_in_sym_stride =
            strides_grp->strides_c2c->v_in_stride * (is_backward ? -1 : 1);
        strides_grp->strides_c2c->v_out_sym_stride =
            strides_grp->strides_c2c->v_out_stride *
            (is_backward ? 1 : -1);

        // Asymmetric C2C steps group by group over the regrouped aux buffer,
        // where each group holds the endpoint points of its real kernels ahead
        // of the interior band. Fold those points out of the group step, on
        // whichever side is an aux buffer.
        FFTZ_UINT8 out_is_prob_buffer =
            !is_backward && realhelper.is_last_stage;
        FFTZ_INTP real_band_offset = with_r2hcf ? 2 : 1;
        FFTZ_INTP hc_band_offset = !with_r2hcf && (radix % 2) ? 1 : 2;
        if (!is_input_prob_buffer(sol))
        {
            strides_grp->strides->v_in_stride -=
                is_backward ? hc_band_offset : real_band_offset;
        }
        if (!out_is_prob_buffer)
        {
            strides_grp->strides->v_out_stride -=
                is_backward ? real_band_offset : hc_band_offset;
        }
    }

    strides_grp->strides_r2hc->v_in_stride =
        strides_grp->strides_r2hc->v_in_sym_stride = vector_strides.in_stride;
    strides_grp->strides_r2hc->v_out_stride =
        strides_grp->strides_r2hc->v_out_sym_stride =
            vector_strides.out_stride;
    strides_grp->strides_r2hcf->v_in_stride =
        strides_grp->strides_r2hcf->v_in_sym_stride = vector_strides.in_stride;
    strides_grp->strides_r2hcf->v_out_stride =
        strides_grp->strides_r2hcf->v_out_sym_stride =
            vector_strides.out_stride;

    if (!realhelper.is_last_stage)
    {
        if (is_backward)
        {
            strides_grp->strides_r2hc->v_out_stride = 1;
            strides_grp->strides_r2hc->v_out_sym_stride = 1;
            strides_grp->strides_r2hcf->v_out_stride = 1;
            strides_grp->strides_r2hcf->v_out_sym_stride = 1;
        }
        else
        {
            strides_grp->strides_r2hc->v_out_stride =
                radix % 2 == 0 ? vector_strides.out_stride - 2
                               : vector_strides.out_stride - 1;
            strides_grp->strides_r2hc->v_out_sym_stride = 1;
            strides_grp->strides_r2hcf->v_out_stride =
                                vector_strides.out_stride - 2;
            strides_grp->strides_r2hcf->v_out_sym_stride = 1;
        }
    }
    if (realhelper.stage != 0)
    {
        if (is_backward)
        {
            strides_grp->strides_r2hc->v_in_stride =
            radix % 2 == 0 ? vector_strides.in_stride - 2
                           : vector_strides.in_stride - 1;
            strides_grp->strides_r2hc->v_in_sym_stride = 1;
            strides_grp->strides_r2hcf->v_in_stride =
                                vector_strides.in_stride - 2;
            strides_grp->strides_r2hcf->v_in_sym_stride = 1;
        }
        else
        {
            strides_grp->strides_r2hc->v_in_stride = 1;
            strides_grp->strides_r2hc->v_in_sym_stride = 1;
            strides_grp->strides_r2hcf->v_in_stride = 1;
            strides_grp->strides_r2hcf->v_in_sym_stride = 1;
        }
    }
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

static inline FFTZ_INTP regroup_c2c_batch_stride(FFTZ_INTP num_groups,
                                                FFTZ_INTP num_c2c_per_group)
{
    return num_groups * num_c2c_per_group * 2;
}

static inline FFTZ_INTP regroup_c2c_group_stride(FFTZ_INTP num_c2c_per_group,
                                        FFTZ_INTP radix, FFTZ_INTP with_r2hcf)
{
    FFTZ_INTP forward_stride = num_c2c_per_group * 2;
    FFTZ_INTP reverse_stride = num_c2c_per_group * 2;
    // dc/nyquist points from r2hc/r2hcf kernels
    FFTZ_INTP real_points = with_r2hcf ? 4 : 2;
    if (radix == 2)
    {
        reverse_stride -= 2; // subtract own reverse stride
        real_points = with_r2hcf ? 2 : 0;
    }
    return forward_stride + reverse_stride + real_points;
}

// Configure stride arrays for R2HC kernels
FFTZ_INT32 setup_r2hc_stride_arrays(aoclfftz_solution_t *sol,
                                            aoclfftz_realhelper_t realhelper,
                                            base_strides_t element_strides,
                                            FFTZ_INTP num_groups,
                                            FFTZ_INTP num_c2c_per_group)
{
    FFTZ_INTP radix = sol->decomp_scheme->dims[0].n;

    FFTZ_UINT32 is_user_buffer = 0;
    FFTZ_INTP element_stride_offset = 0;
    FFTZ_INTP stride_offset = 0;
    FFTZ_UINT32 is_backward =
        FFT_DIR(sol->decomp_scheme->flags) == BACKWARD_FFT_DIR;
    FFTZ_INTP num_c2c_per_group_stride =
        (sol->solver->kernel_c2c->count == 0) ? 1 : num_c2c_per_group * 4;

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

    // Input strides
    is_user_buffer        = realhelper.stage == 0 ? 1 : 0;
    element_stride_offset = compute_half_complex_input
                          ? num_c2c_per_group_stride : num_groups;
    stride_offset         = is_user_buffer
                          ? element_strides.in_stride
                          : element_stride_offset;
    populate_stride_array_r2hc(sol->strides_grp->strides_r2hc->in_strides,
                stride_offset, radix, compute_half_complex_input,
                input_adjust_to_full_complex, is_user_buffer, num_groups);

    // Output strides
    is_user_buffer        = !realhelper.is_CT || realhelper.is_last_stage;
    element_stride_offset = compute_half_complex_output
                          ? num_c2c_per_group_stride : num_groups;
    stride_offset         = is_user_buffer
                          ? element_strides.out_stride : element_stride_offset;
    populate_stride_array_r2hc(sol->strides_grp->strides_r2hc->out_strides,
                stride_offset, radix, compute_half_complex_output,
                output_adjust_to_full_complex, is_user_buffer, num_groups);

    return SOLVER_SUCCESS;
}

// Configure stride arrays for R2HCF kernels
FFTZ_INT32 setup_r2hcf_stride_arrays(aoclfftz_solution_t *sol,
                                      aoclfftz_realhelper_t realhelper,
                                      base_strides_t element_strides)
{
    FFTZ_INTP radix = sol->decomp_scheme->dims[0].n;
    FFTZ_UINT32 is_backward =
        FFT_DIR(sol->decomp_scheme->flags) == BACKWARD_FFT_DIR;

    FFTZ_UINT8 is_user_buffer = 0;
    FFTZ_INTP element_stride_offset = 0;
    FFTZ_INTP stride_offset = 0;

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

    // Input strides
    is_user_buffer        = realhelper.stage == 0;
    element_stride_offset = compute_half_complex_input
                          ? STANDARD_ELEMENT_STRIDE(num_c2c_per_group)
                          : num_groups;
    stride_offset         = is_user_buffer
                          ? element_strides.in_stride : element_stride_offset;
    populate_stride_array_r2hcf(sol->strides_grp->strides_r2hcf->in_strides,
                is_backward ? stride_offset / 2 : stride_offset,
                is_backward ? radix * 2 : radix,
                compute_half_complex_input, input_adjust_to_full_complex,
                is_user_buffer, num_groups);

    // Output strides
    is_user_buffer        = realhelper.is_last_stage;
    element_stride_offset = compute_half_complex_output
                          ? STANDARD_ELEMENT_STRIDE(num_c2c_per_group)
                          : num_groups;
    stride_offset         = is_user_buffer
                          ? element_strides.out_stride : element_stride_offset;
    populate_stride_array_r2hcf(sol->strides_grp->strides_r2hcf->out_strides,
                is_backward ? stride_offset : stride_offset / 2,
                is_backward ? radix : radix * 2,
                compute_half_complex_output, output_adjust_to_full_complex,
                is_user_buffer, num_groups);

    return SOLVER_SUCCESS;
}

// Configure stride arrays for complex-to-complex (C2C) kernels
FFTZ_INT32 setup_c2c_stride_arrays(aoclfftz_solution_t *sol,
                                           aoclfftz_realhelper_t realhelper,
                                           base_strides_t element_strides,
                                           base_strides_t c2c_stride,
                                           FFTZ_INTP num_groups,
                                           FFTZ_INTP with_r2hcf,
                                           FFTZ_INTP num_c2c_per_group)
{
    FFTZ_UINT32 is_backward =
        FFT_DIR(sol->decomp_scheme->flags) == BACKWARD_FFT_DIR;
    FFTZ_INTP radix = sol->decomp_scheme->dims[0].n;
    FFTZ_INTP freq_factor = is_backward
                      ? realhelper.freq_factor
                      : realhelper.freq_factor * radix;

    FFTZ_INTP regroup_batch_stride =
        regroup_c2c_batch_stride(num_groups, num_c2c_per_group);
    FFTZ_INTP regroup_group_stride =
        regroup_c2c_group_stride(num_c2c_per_group, radix, with_r2hcf);

    FFTZ_INT8 is_user_buffer = 0;
    FFTZ_INTP stride_offset = 0;

    // Input strides
    is_user_buffer = realhelper.stage == 0;
    if (is_backward)
    {
        stride_offset = is_user_buffer
                      ? element_strides.in_stride * 2
                      : regroup_group_stride;
    }
    else
    {
        stride_offset = is_user_buffer
                      ? element_strides.in_stride
                      : regroup_batch_stride;
    }
    populate_stride_array(sol->strides_grp->strides->in_strides,
                          stride_offset, radix, 0, 0);

    // Output strides
    is_user_buffer = realhelper.is_last_stage;
    if (is_backward)
    {
        stride_offset = is_user_buffer
                      ? element_strides.out_stride : regroup_batch_stride;
    }
    else
    {
        stride_offset = is_user_buffer
                      ? element_strides.out_stride * 2 : regroup_group_stride;
    }
    populate_stride_array(sol->strides_grp->strides->out_strides,
                          stride_offset, radix, 0, 0);

    // Allocate stride arrays for C2C kernels
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

/*
 * Setup stride tables before execute for one fused CT stage (stage_r or
 * stage_m in REAL_BATCHED_CT_L1_DIRECT).
 *
 * Each direct stage may run R2HC, R2HCF, and C2C kernels.Real and C2C kernels
 * use separate tables: strides (real) and strides_c2c (complex).
 *
 * Some C2C work only touches the upper half of the real spectrum (from
 * Nyquist index (radix+1)/2 through radix-1). For that half, C2C must use
 * the same memory offsets as the real kernels, not default c2c layout.
 *
 * This function copies those upper-half entries once at setup:
 *   forward  (R2C): real out_strides  -> c2c out_strides
 *   backward (C2R): real in_strides   -> c2c in_strides
 *
 * Called from setup_batched_ct_l1_direct_real_solver for both stages.
 * Returns immediately if kernel_c2c->count == 0 (stage has no C2C pass).
 */
FFTZ_VOID prepare_fused_c2c_asymmetric_strides(aoclfftz_solution_t *sol)
{
    if (sol->solver->kernel_c2c->count == 0)
    {
        return;
    }

    FFTZ_INTP radix = sol->decomp_scheme->dims[0].n;
    FFTZ_INTP half_stride_start = (radix + 1) >> 1;
    FFTZ_INTP half_stride_n = radix - half_stride_start;
    FFTZ_UINT8 direction = FFT_DIR(sol->decomp_scheme->flags);

    if (direction == FORWARD_FFT_DIR)
    {
        memcpy(sol->strides_grp->strides_c2c->out_strides + half_stride_start,
               sol->strides_grp->strides->out_strides + half_stride_start,
               half_stride_n * sizeof(FFTZ_INTP));
    }
    else
    {
        memcpy(sol->strides_grp->strides_c2c->in_strides + half_stride_start,
               sol->strides_grp->strides->in_strides + half_stride_start,
               half_stride_n * sizeof(FFTZ_INTP));
    }
}
