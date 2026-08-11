// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file direct_solver_rdft_utils.h
 *
 *  @brief Direct Solver helper function's declarations required for
 *         the setup of direct solver
 *
 *  @author Partiksha
 */

#ifndef DIRECT_SOLVER_UTILS_H
#define DIRECT_SOLVER_UTILS_H

#include "api/aoclfftz_internal.h"
#include "core/kernels/kernel.h"
#include "core/solvers/solver.h"

/**
 * @brief Paired input and output stride values used for element-level access,
 * vector/batch traversal, and C2C kernel batch stepping in both direct
 * and CT decomposition scenarios.
 * Note: Stride arrays cannot be stored here.
 */
typedef struct base_strides
{
    FFTZ_INTP in_stride;  /**< Stride for input buffer access */
    FFTZ_INTP out_stride; /**< Stride for output buffer access */
} base_strides_t;

/**
 * @brief In RFFT, output can be the real problem input buffer or a temp buffer.
 * This function checks if the output is the real problem output buffer.
 */
static inline FFTZ_UINT8 is_output_prob_buffer(aoclfftz_solution_t *sol)
{
    FFTZ_UINT32 is_fwd = FFT_DIR(sol->decomp_scheme->flags) == FORWARD_FFT_DIR;
    FFTZ_UINT32 is_last_stage = sol->next_sol == NULL;
    return (is_fwd && is_last_stage);
}

/**
 * @brief In RFFT, input can be the real problem input buffer or a temp buffer.
 * This function checks if the input is the real problem input buffer.
 */
static inline FFTZ_UINT8 is_input_prob_buffer(aoclfftz_solution_t *sol)
{
    return (FFT_DIR(sol->decomp_scheme->flags) == BACKWARD_FFT_DIR &&
            NUM_RFFT_GROUPS(sol->solver) == 1);
}

/**
 * Updates the strides for the second half of the batch.
 * It subtracts the stride_offset from each stride value to properly address
 * memory locations in the next iteration of C2C kernel execution.
 */
static inline FFTZ_VOID update_asymmetric_strides(FFTZ_INTP *strides,
                                                  FFTZ_INTP radix,
                                                  FFTZ_INTP batch_stride)
{
    // Since the, the second half of the batch wraps around at nyquist point,
    // the stride for second half is `-original_stride*2`
    FFTZ_INTP h2_stride = batch_stride * 2;
    FFTZ_INTP half_stride_start = (radix + 1) >> 1;
    for (FFTZ_INTP i = half_stride_start; i < radix; i++)
    {
        strides[i] -= h2_stride;
    }
}

static inline FFTZ_VOID
zero_rdft_dc_and_nyquist_fp32(FFTZ_FLOAT *out, FFTZ_INTP nyquist_im_offset)
{
    out[1] = 0.0f;
    out[nyquist_im_offset] = 0.0f;
}

static inline FFTZ_VOID
zero_rdft_dc_and_nyquist_fp64(FFTZ_DOUBLE *out, FFTZ_INTP nyquist_im_offset)
{
    out[1] = 0.0;
    out[nyquist_im_offset] = 0.0;
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
static inline FFTZ_VOID set_zero_for_dc_and_nyquist_batched(aoclfftz_solution_t *sol)
{
    FFTZ_INTP nyquist_im_offset = sol->decomp_scheme->nyquist_im_offset_direct;
    FFTZ_INTP v_out_stride = sol->strides_grp->strides->v_out_stride;
    FFTZ_UINTP num_batches = sol->decomp_scheme->vecs[0].n;

    if (DT_PRECISION_FLAG(sol->decomp_scheme->flags) == DT_FLOAT)
    {
        FFTZ_FLOAT *out = (FFTZ_FLOAT *)sol->decomp_scheme->out_real;
        for (FFTZ_UINTP b = 0; b < num_batches; b++)
        {
            zero_rdft_dc_and_nyquist_fp32(out, nyquist_im_offset);
            out += v_out_stride;
        }
    }
    else
    {
        FFTZ_DOUBLE *out = (FFTZ_DOUBLE *)sol->decomp_scheme->out_real;
        for (FFTZ_UINTP b = 0; b < num_batches; b++)
        {
            zero_rdft_dc_and_nyquist_fp64(out, nyquist_im_offset);
            out += v_out_stride;
        }
    }
}

/**
 * @brief Sets imaginary parts of DC and Nyquist frequencies to zero for direct
 * forward R2C transforms (batch == 1).
 *
 * For R2C (real forward) problems, this function sets the imaginary part of
 * the first complex number (DC component) and the complex number
 * corresponding to the Nyquist frequency to zero. This is a property of the
 * discrete Fourier transform of a real-valued signal.
 *
 * @param sol [in, out] The solution object containing problem details,
 *            including the output buffer to be modified.
 */
static inline FFTZ_VOID set_zero_for_dc_and_nyquist(aoclfftz_solution_t *sol)
{
    FFTZ_INTP nyquist_im_offset = sol->decomp_scheme->nyquist_im_offset_direct;

    if (DT_PRECISION_FLAG(sol->decomp_scheme->flags) == DT_FLOAT)
    {
        zero_rdft_dc_and_nyquist_fp32((FFTZ_FLOAT *)sol->decomp_scheme->out_real,
                                      nyquist_im_offset);
    }
    else
    {
        zero_rdft_dc_and_nyquist_fp64((FFTZ_DOUBLE *)sol->decomp_scheme->out_real,
                                      nyquist_im_offset);
    }
}

/**
 * @brief Sets imaginary parts of DC and Nyquist frequencies to zero for CT
 * forward R2C on the last stage.
 *
 * For R2C (real forward) problems, this function sets the imaginary part of
 * the first complex number (DC component) and the complex number
 * corresponding to the Nyquist frequency to zero. This is a property of the
 * discrete Fourier transform of a real-valued signal.
 *
 * @param sol [in, out] The solution object containing problem details,
 *            including the output buffer to be modified.
 */
static inline FFTZ_VOID set_zero_for_dc_and_nyquist_ct(aoclfftz_solution_t *sol)
{
    FFTZ_INTP nyquist_im_offset = sol->decomp_scheme->nyquist_im_offset_ct;

    if (DT_PRECISION_FLAG(sol->decomp_scheme->flags) == DT_FLOAT)
    {
        zero_rdft_dc_and_nyquist_fp32((FFTZ_FLOAT *)sol->decomp_scheme->out_real,
                                      nyquist_im_offset);
    }
    else
    {
        zero_rdft_dc_and_nyquist_fp64((FFTZ_DOUBLE *)sol->decomp_scheme->out_real,
                                      nyquist_im_offset);
    }
}

static inline aoclfftz_solver_type
select_real_st_direct_solver_type(aoclfftz_solution_t *sol, FFTZ_UINT8 is_ct)
{
    FFTZ_UINT32 is_forward =
        FFT_DIR(sol->decomp_scheme->flags) == FORWARD_FFT_DIR;

    if (!is_ct)
    {
        if (is_forward)
        {
            if (sol->decomp_scheme->vecs[0].n > 1)
            {
                return SOLVER_REAL_DIRECT_R2C_BATCHED;
            }
            return SOLVER_REAL_DIRECT_R2C;
        }
        return SOLVER_REAL_DIRECT_C2R;
    }

    if (is_forward)
    {
        return SOLVER_REAL_DIRECT_CT_R2C;
    }
    return SOLVER_REAL_DIRECT_CT_C2R;
}

#ifdef MULTI_THREADING
static inline aoclfftz_solver_type
select_real_mt_direct_solver_type(aoclfftz_solution_t *sol, FFTZ_UINT8 is_ct)
{
    FFTZ_UINT32 is_forward =
        FFT_DIR(sol->decomp_scheme->flags) == FORWARD_FFT_DIR;

    if (!is_ct)
    {
        if (is_forward)
        {
            if (sol->decomp_scheme->vecs[0].n > 1)
            {
                return SOLVER_REAL_MT_DIRECT_R2C_BATCHED;
            }
            return SOLVER_REAL_MT_DIRECT_R2C;
        }
        return SOLVER_REAL_MT_DIRECT_C2R;
    }

    if (is_forward)
    {
        return SOLVER_REAL_MT_DIRECT_CT_R2C;
    }
    return SOLVER_REAL_MT_DIRECT_CT_C2R;
}
#endif

FFTZ_VOID setup_rdft_dc_nyquist_offsets_ds(aoclfftz_decomp_scheme_t *ds);
FFTZ_VOID set_kernel_count_in_each_group(aoclfftz_solution_t *sol,
                             aoclfftz_realhelper_t *realhelper);
FFTZ_INT32 allocate_and_setup_stride(aoclfftz_solution_t *sol,
                               aoclfftz_realhelper_t realhelper);
FFTZ_VOID update_ct_buffers(aoclfftz_solution_t *sol,
                       aoclfftz_realhelper_t *realhelper);
FFTZ_VOID compute_cost(aoclfftz_solution_t *sol, cost_analysis_t *cost,
                  const kernel_t *kernel_c2c, const kernel_t *kernel_r2hc,
                  const kernel_t *kernel_r2hcf);

#endif // DIRECT_SOLVER_UTILS_H
