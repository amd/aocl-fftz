// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file direct_solver_rdft.c
 *
 *  @brief Direct Solver that applies an available kernel to the input problem
 *
 *  This file contains the functions that setup, execute and destroy
 *  the solver.
 *
 *  @author Srirammaswamy Srinivasan
 */

#include <assert.h>
#include "core/common/memory_manager.h"
#include "core/solvers/real/direct_solver_rdft_utils.h"

static FFTZ_INT32 execute_real_direct_r2c(aoclfftz_solution_t *sol,
                                          aoclfftz_mutable_ctx_t *ctx);
static FFTZ_INT32 execute_real_direct_r2c_batched(aoclfftz_solution_t *sol,
                                                  aoclfftz_mutable_ctx_t *ctx);
static FFTZ_INT32 execute_real_direct_c2r(aoclfftz_solution_t *sol,
                                          aoclfftz_mutable_ctx_t *ctx);
static FFTZ_INT32 execute_real_direct_ct_r2c(aoclfftz_solution_t *sol,
                                             aoclfftz_mutable_ctx_t *ctx);
static FFTZ_INT32 execute_real_direct_ct_c2r(aoclfftz_solution_t *sol,
                                             aoclfftz_mutable_ctx_t *ctx);

/** This function will setup the direct solution with the required information
 *  to execute both direct and CT problems.
 *  Even for a CT problem, most of the kernel execution information is required
 *  by a direct solution.
 *  Setup includes the following steps:
 *    1. Set the strides for different kernel variants (C2C, R2HC, R2HCF)
 *    2. Setting up the no. of batch for each kernel variant
 *    3. Updating the input & output buffers for CT problem/sub-problem
 *    4. Cost computation
 *  NOTE: This direct solver will handle both direct and CT problems.
 *  TODO: Separate solver responsibilities for better maintainability:
 *        - Direct solver should handle only direct FFT problems (R2HC kernels)
 *        - CT solver should handle only Cooley-Tukey decomposition problems
 *         (C2C + R2HC/R2HCF kernels)
 */
FFTZ_INT32 setup_real_direct_solver(aoclfftz_solution_t *sol,
                                    cost_analysis_t *cost,
                                    const kernel_t *kernel_c2c,
                                    const kernel_t *kernel_r2hc,
                                    const kernel_t *kernel_r2hcf,
                                    aoclfftz_realhelper_t *realhelper)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");
    FFTZ_INT32 status = SOLVER_SUCCESS;

    status = allocate_and_setup_stride(sol, *realhelper);
    if (status != SOLVER_SUCCESS)
    {
        return status;
    }

    update_ct_buffers(sol, realhelper);

    compute_cost(sol, cost, kernel_c2c, kernel_r2hc, kernel_r2hcf);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return status;
}

/**
 * Executes R2HC (Real to Half-Complex) kernels and processes direct-only
 * FFT problems.
 */
static inline FFTZ_VOID execute_r2hc_kernels(aoclfftz_solution_t *sol,
                                             FFTZ_VOID *in, FFTZ_VOID *out)
{
    if (sol->solver->kernel_r2hc->count == 0)
    {
        return;
    }

    // R2HC kernels are bidirectional
    // so both kfft[FORWARD_FFT_DIR] and kfft[BACKWARD_FFT_DIR] point to the
    // same kernel
    kfft_ kernel_r2hc = sol->solver->kernel_r2hc->kfft[FORWARD_FFT_DIR];
    kernel_r2hc(in, in, out, out, sol->solver->kernel_r2hc->count,
                sol->strides_grp->strides_r2hc, sol->twiddle,
                FFT_DIR(sol->decomp_scheme->flags));
}

/**
 * Executes Real-to-Half-Complex Fused (R2HCF) kernels
 */
static inline FFTZ_VOID execute_r2hcf_kernels(aoclfftz_solution_t *sol,
                                              FFTZ_VOID *in, FFTZ_VOID *out)
{
    // Execute R2HCF kernels (for CT problems)
    if (sol->solver->kernel_r2hcf->count == 0)
    {
        return;
    }

    // R2HCF kernels are bidirectional
    // so both kfft[FORWARD_FFT_DIR] and kfft[BACKWARD_FFT_DIR] point to the
    // same kernel
    kfft_ kernel_r2hcf = sol->solver->kernel_r2hcf->kfft[FORWARD_FFT_DIR];
    kernel_r2hcf(in, in, out, out, sol->solver->kernel_r2hcf->count,
                 sol->strides_grp->strides_r2hcf, sol->twiddle,
                 FFT_DIR(sol->decomp_scheme->flags));
}

static inline FFTZ_VOID execute_c2c_kernels(aoclfftz_solution_t *sol,
                                            const aoclfftz_mutable_ctx_t *ctx,
                                            FFTZ_VOID *in, FFTZ_VOID *out)
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

    // Select this invocation's private stride slot from the shared per-call pool.
    // This keeps asymmetric stride updates isolated from concurrent executions.
    FFTZ_INTP strides_off = (FFTZ_INTP)ctx->slot_idx * MAX_REAL_KERNEL_RADIX *
                            (FFTZ_INTP)sizeof(FFTZ_INTP);
    FFTZ_INTP *c2c_stride = MOVE_ADDR(ctx->c2c_strides_base, strides_off);

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
        init_real_c2c_strides(c2c_stride,
                              sol->strides_grp->strides_c2c->out_strides,
                              sol->strides_grp->strides->out_strides, radix);
        aoclfftz_strides_t strides_local = *(sol->strides_grp->strides_c2c);
        strides_local.out_strides = c2c_stride;

        aoclfftz_twiddle_t tw_local = *(sol->twiddle);
        tw_local.load_multi_cols = 0; // use same twiddle values across batches
        if (!use_asymmetric_kernel)
        {
            for (FFTZ_INTP group_id = 0; group_id < num_c2c_per_group;
                 group_id++)
            {
                // This for loop computes C2C batches within the groups,
                // while the kernel does across multiple groups
                kernel_c2c(in, MOVE_ADDR(in, dt_bytes), out,
                           MOVE_ADDR(out, dt_bytes), num_groups,
                           &strides_local, &tw_local, direction);

                update_asymmetric_strides(c2c_stride, radix,
                                          batch_out_stride);

                // move twiddle buffer to next batch
                tw_local.TW = MOVE_ADDR(tw_local.TW, (FFTZ_INTP)(radix - 1) *
                                        DATA_STRIDE * dt_bytes);
                // Move the in & out buffers to point the next batch
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
                           &strides_local, sol->twiddle,
                           direction);

                // Move the in & out buffers to point the next valid data
                in = MOVE_ADDR(in, v_in_stride * dt_bytes);
                out = MOVE_ADDR(out, v_out_stride * dt_bytes);
            }
        }
    }
    else
    {
        init_real_c2c_strides(c2c_stride,
                              sol->strides_grp->strides_c2c->in_strides,
                              sol->strides_grp->strides->in_strides, radix);
        aoclfftz_strides_t strides_local = *(sol->strides_grp->strides_c2c);
        strides_local.in_strides = c2c_stride;

        aoclfftz_twiddle_t tw_local = *(sol->twiddle);
        tw_local.load_multi_cols = 0; // use same twiddle values across batches
        if (!use_asymmetric_kernel)
        {
            for (FFTZ_INTP group_id = 0; group_id < num_c2c_per_group;
                 group_id++)
            {
                kernel_c2c(in, MOVE_ADDR(in, dt_bytes), out,
                           MOVE_ADDR(out, dt_bytes), num_groups,
                           &strides_local, &tw_local, direction);
                update_asymmetric_strides(c2c_stride, radix,
                                          batch_in_stride);
                // move twiddle buffer to next batch
                tw_local.TW = MOVE_ADDR(tw_local.TW, (FFTZ_INTP)(radix - 1) *
                                        DATA_STRIDE * dt_bytes);
                // Move the in & out buffers to point the next batch
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
                           &strides_local, sol->twiddle,
                           direction);
                in = MOVE_ADDR(in, v_in_stride * dt_bytes);
                out = MOVE_ADDR(out, v_out_stride * dt_bytes);
            }
        }
    }
}

// Runs one CT stage's kernels, then swaps the aux ping-pong pools in ctx.
// ITERATIVE and PARTIAL_RECURSION hand that ctx straight to next_sol; under
// TRUE_RECURSION ctx is by reference, so radix_r's swap is seen by radix_m.
static inline
FFTZ_VOID execute_ct_intra_stage_kernels(aoclfftz_solution_t *sol,
                                         aoclfftz_mutable_ctx_t *ctx,
                                         FFTZ_VOID *in,
                                         FFTZ_VOID *out)
{
    execute_r2hc_kernels(sol, in, out);
    execute_r2hcf_kernels(sol, in, out);

    FFTZ_UINT32 dt_bytes = SOL_DT_SIZE(sol);
    FFTZ_INTP in_offset =
        is_input_prob_buffer(sol)
            ? sol->decomp_scheme->dims[0].in_stride * DATA_STRIDE
            : 1;
    FFTZ_INTP out_offset =
        is_output_prob_buffer(sol)
            ? sol->decomp_scheme->dims[0].out_stride * DATA_STRIDE
            : 1;
    // move in,out pointers to the start of C2C points, by skipping R2HC points
    in = MOVE_ADDR(in, in_offset * dt_bytes);
    out = MOVE_ADDR(out, out_offset * dt_bytes);

    execute_c2c_kernels(sol, ctx, in, out);

    // Alternate the pools so the next stage reads what this one wrote.
    SWAP_BUFFERS(ctx->aux_pool_base_1, ctx->aux_pool_base_2);
}

// Direct-only forward R2C for non-batched problems (vecs[0].n == 1).
static FFTZ_INT32 execute_real_direct_r2c(aoclfftz_solution_t *sol,
                                          aoclfftz_mutable_ctx_t *ctx)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    FFTZ_VOID *in = ctx->in_real;
    FFTZ_VOID *out = ctx->out_real;

    execute_r2hc_kernels(sol, in, out);
    set_zero_for_dc_and_nyquist(sol, out);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SOLVER_SUCCESS;
}

// Direct-only forward R2C for batched problems (vecs[0].n > 1). Batched
// variant is required as DC / Nyquist zeroing needs batch-aware handling.
static FFTZ_INT32 execute_real_direct_r2c_batched(aoclfftz_solution_t *sol,
                                                  aoclfftz_mutable_ctx_t *ctx)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    FFTZ_VOID *in = ctx->in_real;
    FFTZ_VOID *out = ctx->out_real;

    execute_r2hc_kernels(sol, in, out);
    set_zero_for_dc_and_nyquist_batched(sol, out);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SOLVER_SUCCESS;
}

// Direct-only backward C2R
static FFTZ_INT32 execute_real_direct_c2r(aoclfftz_solution_t *sol,
                                          aoclfftz_mutable_ctx_t *ctx)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    FFTZ_VOID *in = ctx->in_real;
    FFTZ_VOID *out = ctx->out_real;

    execute_r2hc_kernels(sol, in, out);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SOLVER_SUCCESS;
}

// CT forward R2C stage (R2HC + R2HCF + C2C); zero DC/Nyquist on last stage
// only.
static FFTZ_INT32 execute_real_direct_ct_r2c(aoclfftz_solution_t *sol,
                                             aoclfftz_mutable_ctx_t *ctx)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    FFTZ_INT32 ret = SOLVER_SUCCESS;
    FFTZ_VOID *in = NULL;
    FFTZ_VOID *out = NULL;
    aoclfftz_resolve_real_io(ctx, sol->decomp_scheme->real_in_role,
                             sol->decomp_scheme->real_out_role, &in, &out);

    execute_ct_intra_stage_kernels(sol, ctx, in, out);

#if REAL_FFT_EXECUTION_ORDER == REAL_FFT_ORDER_TRUE_RECURSION
    // Recurse-then-combine mode: the CT solver owns tree traversal, so the
    // Direct node behaves as a pure leaf (like the Complex Direct solver) and
    // never chains to next_sol. Only the terminal leaf emits DC/Nyquist zeroing.
    if (!HAS_NEXT(sol) &&
        FFT_DIR(sol->decomp_scheme->flags) == FORWARD_FFT_DIR)
    {
        set_zero_for_dc_and_nyquist_ct(sol, out);
    }
#else
    if (HAS_NEXT(sol))
    {
        ret = sol->next_sol->solver->execute_solver(sol->next_sol, ctx);
    }
    else
    {
        set_zero_for_dc_and_nyquist_ct(sol, out);
    }
#endif

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return ret;
}

// CT backward C2R stage (R2HC + R2HCF + C2C).
static FFTZ_INT32 execute_real_direct_ct_c2r(aoclfftz_solution_t *sol,
                                             aoclfftz_mutable_ctx_t *ctx)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    FFTZ_INT32 ret = SOLVER_SUCCESS;
    FFTZ_VOID *in = NULL;
    FFTZ_VOID *out = NULL;
    aoclfftz_resolve_real_io(ctx, sol->decomp_scheme->real_in_role,
                             sol->decomp_scheme->real_out_role, &in, &out);

    execute_ct_intra_stage_kernels(sol, ctx, in, out);

#if REAL_FFT_EXECUTION_ORDER != REAL_FFT_ORDER_TRUE_RECURSION
    if (HAS_NEXT(sol))
    {
        ret = sol->next_sol->solver->execute_solver(sol->next_sol, ctx);
    }
#endif

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return ret;
}

dft_solver_ register_execute_real_direct_r2c(FFTZ_VOID)
{
    return execute_real_direct_r2c;
}

dft_solver_ register_execute_real_direct_r2c_batched(FFTZ_VOID)
{
    return execute_real_direct_r2c_batched;
}

dft_solver_ register_execute_real_direct_c2r(FFTZ_VOID)
{
    return execute_real_direct_c2r;
}

dft_solver_ register_execute_real_direct_ct_r2c(FFTZ_VOID)
{
    return execute_real_direct_ct_r2c;
}

dft_solver_ register_execute_real_direct_ct_c2r(FFTZ_VOID)
{
    return execute_real_direct_ct_c2r;
}
