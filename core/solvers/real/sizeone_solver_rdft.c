// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file sizeone_solver_rdft.c
 *
 *  @brief Size One solver that solves input problems with size as 1.
 *
 *  This file contains the function that execute the solver.
 *
 *  @author D. Vijay Krishna
 *  @author Jeya R
 */

#include "core/solvers/solver.h"

/**
 * @brief Execute single-precision float kernel for size-one real FFT.
 *
 * Performs R2C (Real-to-Complex) forward transform or C2R
 * (Complex-to-Real) backward transform for single-precision float data with
 * transform size of 1. For R2C, copies the real input and sets imaginary
 * part to zero. For C2R, copies only the real part from complex input.
 *
 * @param in_real [in] Pointer to input real data buffer
 * @param in_imag [in] Pointer to input imaginary data buffer (unused)
 * @param out_real [in,out] Pointer to output real data buffer
 * @param out_imag [in,out] Pointer to output imaginary data buffer (unused)
 * @param batch [in] Number of batched transforms to execute
 * @param strides [in] Pointer to stride configuration for input/output access
 * @param UNUSED [in] Unused parameter (twiddle factors not needed for size 1)
 * @param flag [in] Transform direction flag (FORWARD_FFT_DIR or backward)
 */
static FFTZ_VOID execute_real_float_kernel(FFTZ_VOID *in_real,
                                           FFTZ_VOID *in_imag,
                                           FFTZ_VOID *out_real,
                                           FFTZ_VOID *out_imag, FFTZ_INTP batch,
                                           aoclfftz_strides_t *strides,
                                           FFTZ_VOID *UNUSED, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");
    FFTZ_UINT32 dt_bytes = sizeof(FFTZ_FLOAT);
    FFTZ_FLOAT *in_r = (FFTZ_FLOAT *)in_real;
    FFTZ_FLOAT *out_r = (FFTZ_FLOAT *)out_real;

    if (flag == FORWARD_FFT_DIR)
    {
        FFTZ_INTP v_in_stride = strides->v_in_stride * dt_bytes;
        FFTZ_INTP v_out_stride = strides->v_out_stride * dt_bytes * DATA_STRIDE;
        // R2C: retain the real part and set imaginary to zero
        for (FFTZ_INTP i = 0; i < batch; i++)
        {
            out_r[0] = in_r[0];
            out_r[1] = 0.0f;
            in_r = MOVE_ADDR(in_r, v_in_stride);
            out_r = MOVE_ADDR(out_r, v_out_stride);
        }
    }
    else
    {
        FFTZ_INTP v_in_stride = strides->v_in_stride * dt_bytes * DATA_STRIDE;
        FFTZ_INTP v_out_stride = strides->v_out_stride * dt_bytes;
        // C2R: retain the real part
        for (FFTZ_INTP i = 0; i < batch; i++)
        {
            out_r[0] = in_r[0];
            in_r = MOVE_ADDR(in_r, v_in_stride);
            out_r = MOVE_ADDR(out_r, v_out_stride);
        }
    }
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
}

/**
 * @brief Execute double-precision kernel for size-one real FFT.
 *
 * Performs R2C (Real-to-Complex) forward transform or C2R
 * (Complex-to-Real) backward transform for double-precision data with
 * transform size of 1. For R2C, copies the real input and sets imaginary
 * part to zero. For C2R, copies only the real part from complex input.
 *
 * @param in_real [in] Pointer to input real data buffer
 * @param in_imag [in] Pointer to input imaginary data buffer (unused)
 * @param out_real [in,out] Pointer to output real data buffer
 * @param out_imag [in,out] Pointer to output imaginary data buffer (unused)
 * @param batch [in] Number of batched transforms to execute
 * @param strides [in] Pointer to stride configuration for input/output access
 * @param UNUSED [in] Unused parameter (twiddle factors not needed for size 1)
 * @param flag [in] Transform direction flag (FORWARD_FFT_DIR or backward)
 */
static FFTZ_VOID
execute_real_double_kernel(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                           FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                           FFTZ_INTP batch, aoclfftz_strides_t *strides,
                           FFTZ_VOID *UNUSED, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");
    FFTZ_UINT32 dt_bytes = sizeof(FFTZ_DOUBLE);
    FFTZ_DOUBLE *in_r = (FFTZ_DOUBLE *)in_real;
    FFTZ_DOUBLE *out_r = (FFTZ_DOUBLE *)out_real;

    if (flag == FORWARD_FFT_DIR)
    {
        FFTZ_INTP v_in_stride = strides->v_in_stride * dt_bytes;
        FFTZ_INTP v_out_stride = strides->v_out_stride * dt_bytes *
                            DATA_STRIDE; /* R2C: retain the real part and set
                                            imaginary to zero*/
        for (FFTZ_INTP i = 0; i < batch; i++)
        {
            out_r[0] = in_r[0];
            out_r[1] = 0.0;
            in_r = MOVE_ADDR(in_r, v_in_stride);
            out_r = MOVE_ADDR(out_r, v_out_stride);
        }
    }
    else
    {
        FFTZ_INTP v_in_stride = strides->v_in_stride * dt_bytes * DATA_STRIDE;
        FFTZ_INTP v_out_stride =
            strides->v_out_stride * dt_bytes; /*C2R: retain the real part*/
        for (FFTZ_INTP i = 0; i < batch; i++)
        {
            out_r[0] = in_r[0];
            in_r = MOVE_ADDR(in_r, v_in_stride);
            out_r = MOVE_ADDR(out_r, v_out_stride);
        }
    }
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
}

/**
 * @brief Setup and configure the real size-one solver for R2C and C2R
 * transforms.
 *
 * @param sol [in,out] Pointer to the solution structure to be configured
 *
 * @return SOLVER_SUCCESS on successful setup, error code otherwise
 */
FFTZ_INT32 setup_real_sizeone_solver(aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    aoclfftz_strides_t *strides = sol->strides_grp->strides;
    strides->v_in_sym_stride = strides->v_in_stride =
        sol->decomp_scheme->vecs[0].in_stride;
    strides->v_out_sym_stride = strides->v_out_stride =
        sol->decomp_scheme->vecs[0].out_stride;

    FFTZ_UINT8 dt_prec = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    kfft_ kernel = (dt_prec == DT_FLOAT) ? execute_real_float_kernel
                                         : execute_real_double_kernel;
    sol->solver->kernel_c2c->kfft[FORWARD_FFT_DIR] = kernel;
    sol->solver->kernel_c2c->kfft[BACKWARD_FFT_DIR] = kernel;

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SOLVER_SUCCESS;
}

/**
 * @brief Internal recursive function to execute real size-one solver for
 * multi-dimensional problems.
 *
 * @param sol [in,out] Pointer to the solution structure containing problem
 * configuration
 * @param vec_rank [in] Current vector rank being processed (recursion depth)
 *
 * @return SOLVER_SUCCESS on successful execution, error code otherwise
 */
static FFTZ_INT32 execute_real_sizeone_solver_internal(aoclfftz_solution_t *sol,
                                                  FFTZ_INTP vec_rank,
                                                  aoclfftz_mutable_ctx_t *ctx)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");
    FFTZ_UINT32 dt_bytes = CTX_DT_SIZE(ctx);

    if (vec_rank == 1)
    {
        // size-one kernel is bidirectional,
        // so both kfft[FORWARD_FFT_DIR] and kfft[BACKWARD_FFT_DIR] point to the
        // same kernel
        kfft_ execute_innermost_batch =
            sol->solver->kernel_c2c->kfft[FORWARD_FFT_DIR];
        aoclfftz_strides_t *strides = sol->strides_grp->strides;
        execute_innermost_batch(
            ctx->in_real, ctx->in_imag,
            ctx->out_real, ctx->out_imag,
            sol->decomp_scheme->vecs[0].n, strides, sol->twiddle,
            FFT_DIR(ctx->flags));
        return SOLVER_SUCCESS;
    }

    FFTZ_INTP batch;
    FFTZ_INTP v_in_stride;
    FFTZ_INTP v_out_stride;
    FFTZ_INT32 status = SOLVER_SUCCESS;
    aoclfftz_mutable_ctx_t batch_ctx = *ctx;

    v_in_stride = sol->decomp_scheme->vecs[vec_rank - 1].in_stride * dt_bytes;
    v_out_stride = sol->decomp_scheme->vecs[vec_rank - 1].out_stride * dt_bytes;
    for (batch = 0; batch < sol->decomp_scheme->vecs[vec_rank - 1].n; batch++)
    {
        aoclfftz_mutable_ctx_t inner_ctx = batch_ctx;

        // recursive call to solve the inner batches
        status = execute_real_sizeone_solver_internal(sol, vec_rank - 1,
                                                      &inner_ctx);
        if (status != SOLVER_SUCCESS)
        {
            return status;
        }

        // Adjust pointers for the next iteration.
        batch_ctx.in_real  = MOVE_ADDR(batch_ctx.in_real,  v_in_stride);
        batch_ctx.out_real = MOVE_ADDR(batch_ctx.out_real, v_out_stride);
        batch_ctx.in_imag  = MOVE_ADDR(batch_ctx.in_imag,  v_in_stride);
        batch_ctx.out_imag = MOVE_ADDR(batch_ctx.out_imag, v_out_stride);
    }

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SOLVER_SUCCESS;
}

/**
 * @brief Main entry point for executing real size-one solver.
 *
 * The function handles both R2C and C2R transforms for problems where the
 * transform size is 1, supporting multi-dimensional batched operations.
 *
 * @param sol [in,out] Pointer to the solution structure containing problem
 * configuration
 *
 * @return SOLVER_SUCCESS on successful execution, error code otherwise
 */
static FFTZ_INT32 execute_real_sizeone_solver(aoclfftz_solution_t *sol,
                                              aoclfftz_mutable_ctx_t *ctx)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    // inplace check
    if (!IS_OUT_OF_PLACE(ctx->flags) && FFT_DIR(ctx->flags))
    {
        return SOLVER_SUCCESS;
    }

    FFTZ_INTP vec_rank = sol->decomp_scheme->vec_rank;
    execute_real_sizeone_solver_internal(sol, vec_rank, ctx);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SOLVER_SUCCESS;
}

/**
 * @brief Register the real size-one solver execution function.
 *
 * This function returns a function pointer to the real size-one solver
 * execution function.
 *
 * @return Function pointer to execute_real_sizeone_solver
 */
dft_solver_ register_execute_real_sizeone_solver(FFTZ_VOID)
{
    return execute_real_sizeone_solver;
}
