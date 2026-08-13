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

#include <string.h>
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
 * @brief Initializes the C2C kernel stride array `dst`: first half from `strides_c2c`,
 * second half, which wraps around at Nyquist, from `strides`.
 */
static inline FFTZ_VOID init_real_c2c_strides(FFTZ_INTP *dst,
                                              const FFTZ_INTP *strides_c2c,
                                              const FFTZ_INTP *strides,
                                              FFTZ_INTP radix)
{
    FFTZ_INTP half_stride_start = (radix + 1) >> 1;
    memcpy(dst, strides_c2c, radix * sizeof(FFTZ_INTP));
    memcpy(dst + half_stride_start, strides + half_stride_start,
           (radix - half_stride_start) * sizeof(FFTZ_INTP));
}

/**
 * @brief Updates the strides for the second half of the batch.
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
 * @brief Resolve a node's I/O pointers from its roles and the per-call ctx. A SWAP
 * input reads aux_pool_base_1, a SWAP output writes aux_pool_base_2, and each
 * stage swaps the two so its output feeds the next stage.
 */
static inline FFTZ_VOID aoclfftz_resolve_real_io(
    const aoclfftz_mutable_ctx_t *ctx, FFTZ_UINT8 in_role, FFTZ_UINT8 out_role,
    FFTZ_VOID **in_real, FFTZ_VOID **out_real)
{
    *in_real  = (in_role == REAL_USE_AUX_AND_SWAP) ?
                ctx->aux_pool_base_1 : ctx->in_real;
    *out_real = (out_role == REAL_USE_AUX_AND_SWAP) ?
                ctx->aux_pool_base_2 : ctx->out_real;
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
 * @param sol [in] The solution object containing problem details.
 * @param out_real [in, out] The output buffer to be modified.
 */
static inline FFTZ_VOID set_zero_for_dc_and_nyquist_batched(aoclfftz_solution_t *sol,
                                                            FFTZ_VOID *out_real)
{
    FFTZ_INTP nyquist_im_offset = sol->decomp_scheme->nyquist_im_offset_direct;
    FFTZ_INTP v_out_stride = sol->strides_grp->strides->v_out_stride;
    FFTZ_UINTP num_batches = sol->decomp_scheme->vecs[0].n;

    if (DT_PRECISION_FLAG(sol->decomp_scheme->flags) == DT_FLOAT)
    {
        FFTZ_FLOAT *out = (FFTZ_FLOAT *)out_real;
        for (FFTZ_UINTP b = 0; b < num_batches; b++)
        {
            zero_rdft_dc_and_nyquist_fp32(out, nyquist_im_offset);
            out += v_out_stride;
        }
    }
    else
    {
        FFTZ_DOUBLE *out = (FFTZ_DOUBLE *)out_real;
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
 * @param sol [in] The solution object containing problem details.
 * @param out_real [in, out] The output buffer to be modified.
 */
static inline FFTZ_VOID set_zero_for_dc_and_nyquist(aoclfftz_solution_t *sol,
                                                    FFTZ_VOID *out_real)
{
    FFTZ_INTP nyquist_im_offset = sol->decomp_scheme->nyquist_im_offset_direct;

    if (DT_PRECISION_FLAG(sol->decomp_scheme->flags) == DT_FLOAT)
    {
        zero_rdft_dc_and_nyquist_fp32((FFTZ_FLOAT *)out_real,
                                      nyquist_im_offset);
    }
    else
    {
        zero_rdft_dc_and_nyquist_fp64((FFTZ_DOUBLE *)out_real,
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
 * @param sol [in] The solution object containing problem details.
 * @param out_real [in, out] The output buffer to be modified.
 */
static inline FFTZ_VOID set_zero_for_dc_and_nyquist_ct(aoclfftz_solution_t *sol,
                                                       FFTZ_VOID *out_real)
{
    FFTZ_INTP nyquist_im_offset = sol->decomp_scheme->nyquist_im_offset_ct;

    if (DT_PRECISION_FLAG(sol->decomp_scheme->flags) == DT_FLOAT)
    {
        zero_rdft_dc_and_nyquist_fp32((FFTZ_FLOAT *)out_real,
                                      nyquist_im_offset);
    }
    else
    {
        zero_rdft_dc_and_nyquist_fp64((FFTZ_DOUBLE *)out_real,
                                      nyquist_im_offset);
    }
}

/**
 * @brief Run C2C kernel batches for a real direct / CT stage node.
 *
 * @param ctx Per-call mutable context for thread-safe C2C stride slots.
 * @param strides_prepped 1 when prepare_fused_c2c_asymmetric_strides already
 *        copied Nyquist-half strides at setup; 0 for the normal direct path.
 */
static inline FFTZ_VOID
execute_c2c_kernels_rdft(aoclfftz_solution_t *sol,
                         const aoclfftz_mutable_ctx_t *ctx, FFTZ_VOID *in,
                         FFTZ_VOID *out, FFTZ_UINT8 strides_prepped)
{
    if (sol->solver->kernel_c2c->count == 0)
    {
        return;
    }

    FFTZ_UINT32 dt_bytes = SOL_DT_SIZE(sol);
    FFTZ_INTP radix = sol->decomp_scheme->dims[0].n;
    FFTZ_UINT8 direction = FFT_DIR(sol->decomp_scheme->flags);
    kfft_ kernel_c2c = sol->solver->kernel_c2c->kfft[direction];
    FFTZ_UINT32 is_fwd = (direction == FORWARD_FFT_DIR);

    FFTZ_INTP num_groups = NUM_RFFT_GROUPS(sol->solver);
    FFTZ_INTP num_c2c_per_group = sol->solver->kernel_c2c->count / num_groups;
    FFTZ_UINT8 use_asymmetric_kernel = num_c2c_per_group >= num_groups;

    FFTZ_INTP half_stride_start = (radix + 1) >> 1;
    FFTZ_INTP half_stride_n = radix - half_stride_start;

    FFTZ_INTP *stride_mut = NULL;
    aoclfftz_strides_t strides_local = *(sol->strides_grp->strides_c2c);

    if (ctx != NULL)
    {
        FFTZ_INTP strides_off = (FFTZ_INTP)ctx->slot_idx * MAX_REAL_KERNEL_RADIX *
                                (FFTZ_INTP)sizeof(FFTZ_INTP);
        stride_mut = MOVE_ADDR(ctx->c2c_strides_base, strides_off);

        if (is_fwd)
        {
            init_real_c2c_strides(stride_mut,
                                  sol->strides_grp->strides_c2c->out_strides,
                                  sol->strides_grp->strides->out_strides, radix);
            strides_local.out_strides = stride_mut;
        }
        else
        {
            init_real_c2c_strides(stride_mut,
                                  sol->strides_grp->strides_c2c->in_strides,
                                  sol->strides_grp->strides->in_strides, radix);
            strides_local.in_strides = stride_mut;
        }
    }
    else if (is_fwd)
    {
        stride_mut = sol->strides_grp->strides_c2c->out_strides;
        if (!use_asymmetric_kernel || !strides_prepped)
        {
            init_real_c2c_strides(stride_mut,
                                  sol->strides_grp->strides_c2c->out_strides,
                                  sol->strides_grp->strides->out_strides, radix);
        }
    }
    else
    {
        stride_mut = sol->strides_grp->strides_c2c->in_strides;
        if (!use_asymmetric_kernel || !strides_prepped)
        {
            memcpy(stride_mut + half_stride_start,
                   sol->strides_grp->strides->in_strides + half_stride_start,
                   half_stride_n * sizeof(FFTZ_INTP));
        }
    }

    FFTZ_INTP batch_in_stride =
        is_input_prob_buffer(sol)
            ? sol->decomp_scheme->dims[0].in_stride * DATA_STRIDE
            : DATA_STRIDE;
    FFTZ_INTP batch_out_stride =
        is_output_prob_buffer(sol)
            ? sol->decomp_scheme->dims[0].out_stride * DATA_STRIDE
            : DATA_STRIDE;

    if (is_fwd)
    {
        aoclfftz_twiddle_t tw_local = *(sol->twiddle);
        tw_local.load_multi_cols = 0; // use same twiddle values across batches
        if (!use_asymmetric_kernel)
        {
            for (FFTZ_INTP group_id = 0; group_id < num_c2c_per_group;
                 group_id++)
            {
                kernel_c2c(in, MOVE_ADDR(in, dt_bytes), out,
                           MOVE_ADDR(out, dt_bytes), num_groups, &strides_local,
                           &tw_local, direction);

                update_asymmetric_strides(stride_mut, radix, batch_out_stride);

                tw_local.TW = MOVE_ADDR(tw_local.TW, (FFTZ_INTP)(radix - 1) *
                                        DATA_STRIDE * dt_bytes);
                in = MOVE_ADDR(in, batch_in_stride * dt_bytes);
                out = MOVE_ADDR(out, batch_out_stride * dt_bytes);
            }
        }
        else
        {
            FFTZ_INTP v_in_stride = sol->strides_grp->strides->v_in_stride;
            FFTZ_INTP v_out_stride = sol->strides_grp->strides->v_out_stride;
            for (FFTZ_INTP group_id = 0; group_id < num_groups; group_id++)
            {
                kernel_c2c(in, MOVE_ADDR(in, dt_bytes), out,
                           MOVE_ADDR(out, dt_bytes), num_c2c_per_group,
                           &strides_local, sol->twiddle, direction);

                in = MOVE_ADDR(in, v_in_stride * dt_bytes);
                out = MOVE_ADDR(out, v_out_stride * dt_bytes);
            }
        }
    }
    else
    {
        aoclfftz_twiddle_t tw_local = *(sol->twiddle);
        tw_local.load_multi_cols = 0; // use same twiddle values across batches
        if (!use_asymmetric_kernel)
        {
            for (FFTZ_INTP group_id = 0; group_id < num_c2c_per_group;
                 group_id++)
            {
                kernel_c2c(in, MOVE_ADDR(in, dt_bytes), out,
                           MOVE_ADDR(out, dt_bytes), num_groups, &strides_local,
                           &tw_local, direction);
                update_asymmetric_strides(stride_mut, radix, batch_in_stride);
                tw_local.TW = MOVE_ADDR(tw_local.TW, (FFTZ_INTP)(radix - 1) *
                                        DATA_STRIDE * dt_bytes);
                in = MOVE_ADDR(in, batch_in_stride * dt_bytes);
                out = MOVE_ADDR(out, batch_out_stride * dt_bytes);
            }
        }
        else
        {
            FFTZ_INTP v_in_stride = sol->strides_grp->strides->v_in_stride;
            FFTZ_INTP v_out_stride = sol->strides_grp->strides->v_out_stride;
            for (FFTZ_INTP group_id = 0; group_id < num_groups; group_id++)
            {
                kernel_c2c(in, MOVE_ADDR(in, dt_bytes), out,
                           MOVE_ADDR(out, dt_bytes), num_c2c_per_group,
                           &strides_local, sol->twiddle, direction);
                in = MOVE_ADDR(in, v_in_stride * dt_bytes);
                out = MOVE_ADDR(out, v_out_stride * dt_bytes);
            }
        }
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
FFTZ_VOID prepare_fused_c2c_asymmetric_strides(aoclfftz_solution_t *sol);

#endif // DIRECT_SOLVER_UTILS_H
