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

/** @file direct_solver_utils.c
 *
 *  @brief Direct Solver helper functions required for the setup of direct solver
 *
 *  @author Partiksha
 */

#include "core/solvers/real/direct_solver.h"
#include "api/aoclfftz_internal.h"
#include "core/common/memory_manager.h"
#include "core/common/strides.h"
#include "core/solvers/real/direct_solver_utils.h"

/**
 * @brief Compute complex conjugates for a set of selective points.
 *        Conjugate the second half of the complex numbers in the input buffer.
 *
 * Example:
 * Given an input for radix 4 (4 complex numbers in interleaved format)
 * input  -> (1,  2), (3,  4), (5,  6), (7,  8)
 * output -> (1,  2), (3,  4), (5, -6), (7, -8)
 *
 * @param data in/out data buffer
 * @param radix radix of the C2C kernel
 * @param n batch of the C2C kernels
 * @param strides strides array for the buffer
 * @param vec_stride vector stride for the buffer
 * @param prec precision flag (DT_FLOAT or DT_DOUBLE)
 * @return VOID
 */
VOID compute_conjugates(VOID *data, INTP radix, INTP n, INTP *strides,
                        INTP vec_stride, UINT32 prec)
{
    INTP points = (radix + 1) >> 1; // ceil div
    if (prec == DT_FLOAT)
    {
        FLOAT *data_i = (FLOAT *)data + 1;
        for (INTP i = 0; i < n; i++)
        {
            for (INTP j = points; j < radix; j++)
            {
                data_i[strides[j]] = -data_i[strides[j]];
            }
            data_i += vec_stride;
        }
    }
    else
    {
        DOUBLE *data_i = (DOUBLE *)data + 1;
        for (INTP i = 0; i < n; i++)
        {
            for (INTP j = points; j < radix; j++)
            {
                data_i[strides[j]] = -data_i[strides[j]];
            }
            data_i += vec_stride;
        }
    }
}

/**
 * @brief Compute complex conjugates for a set of selective points.
 *        Conjugate the second half of the complex numbers in the input buffer.
 *
 * Example:
 * Given an input for radix 4 (4 complex numbers in interleaved format)
 * input  -> (1,  2), (3,  4), (5,  6), (7,  8)
 * output -> (1,  2), (3,  4), (5, -6), (7, -8)
 *
 * This is the out-of-place version of the compute_conjugates function.
 *
 * @param out out data buffer
 * @param in in data buffer
 * @param radix radix of the C2C kernel
 * @param n batch of the C2C kernels
 * @param strides strides array for the buffer
 * @param vec_stride vector stride for the buffer
 * @param prec precision flag (DT_FLOAT or DT_DOUBLE)
 * @return VOID
 */
VOID compute_conjugates_outplace(VOID *out, VOID *in, INTP radix, INTP n,
                                 INTP *strides, INTP vec_stride, UINT32 prec)
{
    INTP points = (radix + 1) >> 1; // ceil div
    if (prec == DT_FLOAT)
    {
        FLOAT *in_f = (FLOAT *)in;
        FLOAT *out_f = (FLOAT *)out;
        for (INTP i = 0; i < n; i++)
        {
            for (INTP j = 0; j < points; j++)
            {
                out_f[j * DATA_STRIDE] = in_f[strides[j]];
                out_f[j * DATA_STRIDE + 1] = in_f[strides[j] + 1];
            }
            for (INTP j = points; j < radix; j++)
            {
                out_f[j * DATA_STRIDE] = in_f[strides[j]];
                out_f[j * DATA_STRIDE + 1] = -in_f[strides[j] + 1];
            }
            in_f += vec_stride;
            out_f += (radix * DATA_STRIDE);
        }
    }
    else
    {
        DOUBLE *in_d = (DOUBLE *)in;
        DOUBLE *out_d = (DOUBLE *)out;
        for (INTP i = 0; i < n; i++)
        {
            for (INTP j = 0; j < points; j++)
            {
                out_d[j * DATA_STRIDE] = in_d[strides[j]];
                out_d[j * DATA_STRIDE + 1] = in_d[strides[j] + 1];
            }
            for (INTP j = points; j < radix; j++)
            {
                out_d[j * DATA_STRIDE] = in_d[strides[j]];
                out_d[j * DATA_STRIDE + 1] = -in_d[strides[j] + 1];
            }
            in_d += vec_stride;
            out_d += (radix * DATA_STRIDE);
        }
    }
}

/**
 * @brief Sets imaginary parts of DC and Nyquist frequencies to zero for
 * batched R2C transforms.
 *
 * For R2C (real forward) problems, this function sets the imaginary part of
 * the first complex number (DC component) and the complex number
 * corresponding to the Nyquist frequency to zero. This is a property of the
 * discrete Fourier transform of a real-valued signal. This function handles
 * batched transforms.
 *
 * @param sol [in, out] The solution object containing problem details,
 *            including the output buffer to be modified.
 */
VOID set_zero_for_dc_and_nyquist_batched(aoclfftz_solution_t *sol)
{
    UINTP transform_len = sol->decomp_scheme->dims[0].n;
    UINTP num_batches = sol->decomp_scheme->vecs[0].n;
    UINTP out_stride = sol->decomp_scheme->dims[0].out_stride;
    UINTP v_out_stride = sol->decomp_scheme->vecs[0].out_stride * 2;
    UINTP nyquist_im_offset =
        transform_len % 2 == 0 ? transform_len * out_stride + 1 : 1;
    if (DT_PRECISION_FLAG(sol->decomp_scheme->flags) == DT_FLOAT)
    {
        FLOAT *out = (FLOAT *)sol->decomp_scheme->out_real;
        for (UINTP b = 0; b < num_batches; b++)
        {
            out[1] = 0.0f;
            out[nyquist_im_offset] = 0.0f;
            out += v_out_stride;
        }
    }
    else
    {
        DOUBLE *out = (DOUBLE *)sol->decomp_scheme->out_real;
        for (UINTP b = 0; b < num_batches; b++)
        {
            out[1] = 0.0;
            out[nyquist_im_offset] = 0.0;
            out += v_out_stride;
        }
    }
}

/**
 * @brief Sets imaginary parts of DC and Nyquist frequencies to zero for R2C
 * transforms.
 *
 * For R2C (real forward) problems, this function sets the imaginary part of
 * the first complex number (DC component) and the complex number
 * corresponding to the Nyquist frequency to zero. This is a property of the
 * discrete Fourier transform of a real-valued signal.
 *
 * @param sol [in, out] The solution object containing problem details,
 *            including the output buffer to be modified.
 */
VOID set_zero_for_dc_and_nyquist(aoclfftz_solution_t *sol)
{
    // For R2C (real forward) problems, set the imaginary part of first and
    // last points in half-complex buffer to 0.
    INTP transform_len =
        sol->decomp_scheme->dims[0].n * sol->decomp_scheme->vecs[0].n;
    INTP out_stride = sol->decomp_scheme->dims[0].out_stride;
    INTP nyquist_im_offset =
        transform_len % 2 == 0 ? transform_len * out_stride + 1 : 1;

    if (DT_PRECISION_FLAG(sol->decomp_scheme->flags) == DT_FLOAT)
    {
        FLOAT *out = (FLOAT *)sol->decomp_scheme->out_real;
        out[1] = 0.0f;
        out[nyquist_im_offset] = 0.0f;
    }
    else
    {
        DOUBLE *out = (DOUBLE *)sol->decomp_scheme->out_real;
        out[1] = 0.0;
        out[nyquist_im_offset] = 0.0;
    }
}

// Configure and sets the count values for different kernel types
// (c2c, r2hc, r2hcf) based on the problem parameters.
VOID set_kernel_count_in_each_group(aoclfftz_solution_t *sol,
                    aoclfftz_realhelper_t *realhelper)
{
    UINT32 is_backward = FFT_DIR(sol->decomp_scheme->flags) == BACKWARD_FFT_DIR;
    INTP radix = sol->decomp_scheme->dims[0].n;
    INTP batch = sol->decomp_scheme->vecs[0].n;

    INTP num_groups = realhelper->problem_size / realhelper->freq_factor;
    num_groups = is_backward ? num_groups : num_groups / radix;
    INTP prev_freq_factor= is_backward
                           ? realhelper->freq_factor / radix
                           : realhelper->freq_factor;
    UINT32 is_even = prev_freq_factor% 2 == 0;

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
static inline VOID set_ct_base_strides(aoclfftz_solution_t *sol,
            aoclfftz_realhelper_t realhelper, base_strides_t *element_strides,
            base_strides_t *vector_strides, base_strides_t *c2c_stride)
{
    INTP batch = sol->decomp_scheme->vecs[0].n;
    INTP radix = sol->decomp_scheme->dims[0].n;
    UINT32 is_backward = FFT_DIR(sol->decomp_scheme->flags) == BACKWARD_FFT_DIR;
    UINT32 is_first_stage = realhelper.stage == 0;
    UINT32 is_last_stage = realhelper.is_last_stage;
    INTP is_input_problem_buffer = is_backward && is_first_stage;
    INTP is_output_problem_buffer = !is_backward && is_last_stage;

    INTP freq_factor       = is_backward
                           ? realhelper.freq_factor
                           : realhelper.freq_factor * radix;
    INTP prev_freq_factor = freq_factor / radix;

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
static inline VOID set_base_strides(aoclfftz_solution_t *sol,
                                    base_strides_t *element_strides,
                                    base_strides_t *vector_strides)
{
    UINT32 is_backward = FFT_DIR(sol->decomp_scheme->flags) == BACKWARD_FFT_DIR;
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
static inline VOID set_vector_strides_for_kernels(aoclfftz_solution_t *sol,
                                                  base_strides_t vector_strides)
{
    aoclfftz_strides_grp_t *strides_grp = sol->strides_grp;

    strides_grp->strides->v_in_stride = vector_strides.in_stride;
    strides_grp->strides->v_out_stride = vector_strides.out_stride;
    strides_grp->strides_c2c->v_in_stride = vector_strides.in_stride;
    strides_grp->strides_c2c->v_out_stride = vector_strides.out_stride;
    strides_grp->strides_r2hc->v_in_stride = vector_strides.in_stride;
    strides_grp->strides_r2hc->v_out_stride = vector_strides.out_stride;
    strides_grp->strides_r2hcf->v_in_stride = vector_strides.in_stride;
    strides_grp->strides_r2hcf->v_out_stride = vector_strides.out_stride;
}

// Determine complex and half-complex flags for kernel strides
static inline VOID set_complex_format(aoclfftz_realhelper_t realhelper,
                                      UINT32 is_backward,
                                      UINT32 *input_adjust_to_full_complex,
                                      UINT32 *output_adjust_to_full_complex,
                                      UINT32 *compute_half_complex_input,
                                      UINT32 *compute_half_complex_output)
{
    UINT32 is_first_stage = realhelper.stage == 0;
    UINT32 is_last_stage = realhelper.is_last_stage;
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
static inline VOID setup_r2hc_stride_arrays(aoclfftz_solution_t *sol,
                                            aoclfftz_realhelper_t realhelper,
                                            base_strides_t element_strides)
{
    INTP radix = sol->decomp_scheme->dims[0].n;
    UINT32 is_backward = FFT_DIR(sol->decomp_scheme->flags) == BACKWARD_FFT_DIR;

    // 1 if the stride values needs to be adjusted to full complex format
    UINT32 input_adjust_to_full_complex = 0;
    UINT32 output_adjust_to_full_complex = 0;

    // 1 for half-complex, 0 for real & complex data
    UINT32 compute_half_complex_input, compute_half_complex_output;

    set_complex_format(realhelper, is_backward,
                &input_adjust_to_full_complex, &output_adjust_to_full_complex,
                &compute_half_complex_input, &compute_half_complex_output);

    alloc_stride_arrays(sol->strides_grp->strides_r2hc, radix);
    populate_stride_array(sol->strides_grp->strides_r2hc->in_strides,
                element_strides.in_stride, radix, compute_half_complex_input,
                input_adjust_to_full_complex);
    populate_stride_array(sol->strides_grp->strides_r2hc->out_strides,
                element_strides.out_stride, radix, compute_half_complex_output,
                output_adjust_to_full_complex);
}

// Configure stride arrays for R2HCF kernels
static inline VOID setup_r2hcf_stride_arrays(aoclfftz_solution_t *sol,
                                      aoclfftz_realhelper_t realhelper,
                                      base_strides_t element_strides)
{
    INTP radix = sol->decomp_scheme->dims[0].n;
    UINT32 is_backward = FFT_DIR(sol->decomp_scheme->flags) == BACKWARD_FFT_DIR;

    // number of groups and no. of c2c kernel calls per group
    INTP num_groups = NUM_RFFT_GROUPS(sol->solver);
    INTP num_c2c_per_group = sol->solver->kernel_c2c->count / num_groups;

    // 1 if the stride values needs to be adjusted to full complex format
    UINT32 input_adjust_to_full_complex = 0;
    UINT32 output_adjust_to_full_complex = 0;

    // 1 for half-complex, 0 for real & complex data
    UINT32 compute_half_complex_input, compute_half_complex_output;

    set_complex_format(realhelper, is_backward,
                &input_adjust_to_full_complex, &output_adjust_to_full_complex,
                &compute_half_complex_input, &compute_half_complex_output);

    alloc_stride_arrays(sol->strides_grp->strides_r2hcf, radix * 2);
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
}

// Configure stride arrays for complex-to-complex (C2C) kernels
static inline VOID setup_c2c_stride_arrays(aoclfftz_solution_t *sol,
                                           aoclfftz_realhelper_t realhelper,
                                           base_strides_t element_strides,
                                           base_strides_t c2c_stride)
{
    UINT32 is_backward = FFT_DIR(sol->decomp_scheme->flags) == BACKWARD_FFT_DIR;
    INTP radix = sol->decomp_scheme->dims[0].n;
    INTP freq_factor = is_backward
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

    alloc_stride_arrays(sol->strides_grp->strides_c2c, radix);
    memcpy(sol->strides_grp->strides_c2c->in_strides,
           sol->strides_grp->strides->in_strides, radix * sizeof(INTP));
    memcpy(sol->strides_grp->strides_c2c->out_strides,
           sol->strides_grp->strides->out_strides, radix * sizeof(INTP));

    // set frequency strides and call to prepare strides once
    INTP *target_stride_array = is_backward
                              ? sol->strides_grp->strides->in_strides
                              : sol->strides_grp->strides->out_strides;
    INTP c2c_batch_stride   = is_backward
                              ? c2c_stride.in_stride
                              : c2c_stride.out_stride;
    prepare_real_c2c_kernel_strides(target_stride_array, target_stride_array,
                                    radix, freq_factor, c2c_batch_stride);
}

// Allocate and set up stride arrays for different kernel types
VOID allocate_and_setup_stride(aoclfftz_solution_t *sol,
                               aoclfftz_realhelper_t realhelper)
{
    INTP radix = sol->decomp_scheme->dims[0].n;
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

    alloc_stride_arrays(sol->strides_grp->strides, radix);
    // Setup for C2C kernels if needed
    if (sol->solver->kernel_c2c->count != 0)
    {
        setup_c2c_stride_arrays(sol, realhelper, element_strides, c2c_strides);
    }

    // Setup for R2HC kernels if needed
    if (sol->solver->kernel_r2hc->count != 0)
    {
        setup_r2hc_stride_arrays(sol, realhelper, element_strides);
    }

    // Setup for R2HCF kernels if needed
    if (sol->solver->kernel_r2hcf->count != 0)
    {
        setup_r2hcf_stride_arrays(sol, realhelper, element_strides);
    }

    set_vector_strides_for_kernels(sol, vector_strides);

    // for C2R out-of-place CT problems, allocate and prepare
    // `strides_c2r_ct_op` for the CT stage that accesses the input buffer
    if (IS_OUT_OF_PLACE(sol->decomp_scheme->flags) && realhelper.is_CT &&
        is_input_prob_buffer(sol))
    {
        if (sol->strides_grp->strides_c2r_ct_op == NULL)
        {
            ALLOC_ALIGN_INIT(sol->strides_grp->strides_c2r_ct_op,
                             aoclfftz_strides_t, sizeof(aoclfftz_strides_t));
            FREE_ALIGN_ALLOCATED_MEM(
                sol->strides_grp->strides_c2r_ct_op->in_strides);
            FREE_ALIGN_ALLOCATED_MEM(
                sol->strides_grp->strides_c2r_ct_op->out_strides);
            ALLOC_ALIGN_INIT(sol->strides_grp->strides_c2r_ct_op->in_strides,
                             INTP, radix * sizeof(INTP));
            ALLOC_ALIGN_INIT(sol->strides_grp->strides_c2r_ct_op->out_strides,
                             INTP, radix * sizeof(INTP));
        }
        memcpy(sol->strides_grp->strides_c2r_ct_op->out_strides,
               sol->strides_grp->strides->out_strides, radix * sizeof(INTP));
        sol->strides_grp->strides_c2r_ct_op->v_in_stride = radix * DATA_STRIDE;
        sol->strides_grp->strides_c2r_ct_op->v_out_stride =
            sol->strides_grp->strides_c2c->v_out_stride;
        for (INT32 r = 0; r < radix; r++)
        {
            sol->strides_grp->strides_c2r_ct_op->in_strides[r] =
                r * DATA_STRIDE;
        }
    }
}

/** Update the in/out buffers of direct solution for CT problem
 *
 * solution from of CT problem after buffered sol
 * ... -> buffered -> direct -> CT -> direct -> ... -> CT -> direct
 *
 * Here, the buffered will have in & out of the current batch
 *
 * Buffered solver will change the input/output buffers of direct solution
 * in the following way:
 *
 * buffered    [in -> out]
 * |--> direct   [in -> aux1]
 * |----> CT
 * |----> direct   [aux1 -> aux2]
 * |------> CT
 * |------> direct   [aux2 -> aux1]
 * |--------> CT
 * |--------> direct   [aux1 -> out]
 * this example is for a 3 level CT problem
 */
VOID update_ct_buffers(aoclfftz_solution_t *sol,
                       aoclfftz_realhelper_t *realhelper)
{
    if (!realhelper->is_CT)
        return;

    UINT32 dt_bytes = SOL_DT_SIZE(sol);

    VOID *in_real = NULL;
    VOID *out_real = NULL;

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
VOID compute_cost(aoclfftz_solution_t *sol, cost_analysis_t *cost,
                  const kernel_t *kernel_c2c, const kernel_t *kernel_r2hc,
                  const kernel_t *kernel_r2hcf)
{
    UINT32 is_backward = FFT_DIR(sol->decomp_scheme->flags) == BACKWARD_FFT_DIR;
    UINT32 precision = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    if (GET_SELECTOR_MODE(sol->decomp_scheme->flags) != AOCLFFTZ_FIXED_SELECTOR)
        return;

    UINT8 sets_c2c = kernel_c2c->sets[precision - 2];
    UINT8 sets_r2hc = kernel_r2hc->sets[precision - 2];
    UINT8 sets_r2hcf = kernel_r2hcf->sets[precision - 2];

    ops_cycles_t ops_cycles_c2c, ops_cycles_r2hc, ops_cycles_r2hcf;
    cost->time = 0;
    INT64 c2c_cost = 0;
    INT64 r2hc_cost = 0;
    INT64 r2hcf_cost = 0;

    // Calculate C2C kernel cost
    if (sol->solver->kernel_c2c->count != 0)
    {
        ops_cycles_c2c = kernel_c2c->k_ops_cnt(precision, is_backward);
        c2c_cost = ((ops_cycles_c2c.fma * AMD_ZEN_FP_FMA_CYCLES) +
                    (ops_cycles_c2c.mul * AMD_ZEN_FP_MUL_CYCLES) +
                    (ops_cycles_c2c.add * AMD_ZEN_FP_ADD_CYCLES) +
                    (ops_cycles_c2c.move * AMD_ZEN_FP_MOVE_CYCLES) +
                    (ops_cycles_c2c.perm * AMD_ZEN_FP_PERM_CYCLES) +
                    (ops_cycles_c2c.other * AMD_ZEN_FP_OTHER_CYCLES));

        if (sol->solver->kernel_c2c->count >= sets_c2c)
        {
            c2c_cost = (c2c_cost + sets_c2c - 1) / sets_c2c;
        }
        c2c_cost = c2c_cost * sol->solver->kernel_c2c->count;
    }

    // Calculate R2HC kernel cost
    if (sol->solver->kernel_r2hc->count != 0)
    {
        ops_cycles_r2hc = kernel_r2hc->k_ops_cnt(precision, is_backward);
        r2hc_cost = ((ops_cycles_r2hc.fma * AMD_ZEN_FP_FMA_CYCLES) +
                     (ops_cycles_r2hc.mul * AMD_ZEN_FP_MUL_CYCLES) +
                     (ops_cycles_r2hc.add * AMD_ZEN_FP_ADD_CYCLES) +
                     (ops_cycles_r2hc.move * AMD_ZEN_FP_MOVE_CYCLES) +
                     (ops_cycles_r2hc.perm * AMD_ZEN_FP_PERM_CYCLES) +
                     (ops_cycles_r2hc.other * AMD_ZEN_FP_OTHER_CYCLES));

        if (sol->solver->kernel_r2hc->count >= sets_r2hc)
        {
            r2hc_cost = (r2hc_cost + sets_r2hc - 1) / sets_r2hc;
        }
        r2hc_cost = r2hc_cost * sol->solver->kernel_r2hc->count;
    }

    // Calculate R2HCF kernel cost
    if (sol->solver->kernel_r2hcf->count != 0)
    {
        ops_cycles_r2hcf = kernel_r2hcf->k_ops_cnt(precision, is_backward);
        r2hcf_cost = ((ops_cycles_r2hcf.fma * AMD_ZEN_FP_FMA_CYCLES) +
                      (ops_cycles_r2hcf.mul * AMD_ZEN_FP_MUL_CYCLES) +
                      (ops_cycles_r2hcf.add * AMD_ZEN_FP_ADD_CYCLES) +
                      (ops_cycles_r2hcf.move * AMD_ZEN_FP_MOVE_CYCLES) +
                      (ops_cycles_r2hcf.perm * AMD_ZEN_FP_PERM_CYCLES) +
                      (ops_cycles_r2hcf.other * AMD_ZEN_FP_OTHER_CYCLES));

        if (sol->solver->kernel_r2hcf->count >= sets_r2hcf)
        {
            r2hcf_cost = (r2hcf_cost + sets_r2hcf - 1) / sets_r2hcf;
        }
        r2hcf_cost = r2hcf_cost * sol->solver->kernel_r2hcf->count;
    }

    cost->ops = c2c_cost + r2hc_cost + r2hcf_cost;
}
