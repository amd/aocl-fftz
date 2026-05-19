/**
 * Copyright (C) 2026, Advanced Micro Devices. All rights reserved.
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
static VOID execute_real_float_kernel(VOID *in_real, VOID *in_imag,
                                      VOID *out_real, VOID *out_imag,
                                      INTP batch, aoclfftz_strides_t *strides,
                                      VOID *UNUSED, UINT8 flag)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");
    UINT32 dt_bytes = sizeof(FLOAT);
    FLOAT *in_r = (FLOAT *)in_real;
    FLOAT *out_r = (FLOAT *)out_real;

    if (flag == FORWARD_FFT_DIR)
    {
        INTP v_in_stride = strides->v_in_stride * dt_bytes;
        INTP v_out_stride = strides->v_out_stride * dt_bytes * DATA_STRIDE;
        // R2C: retain the real part and set imaginary to zero
        for (INTP i = 0; i < batch; i++)
        {
            out_r[0] = in_r[0];
            out_r[1] = 0.0f;
            in_r = MOVE_ADDR(in_r, v_in_stride);
            out_r = MOVE_ADDR(out_r, v_out_stride);
        }
    }
    else
    {
        INTP v_in_stride = strides->v_in_stride * dt_bytes * DATA_STRIDE;
        INTP v_out_stride = strides->v_out_stride * dt_bytes;
        // C2R: retain the real part
        for (INTP i = 0; i < batch; i++)
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
static VOID execute_real_double_kernel(VOID *in_real, VOID *in_imag,
                                      VOID *out_real, VOID *out_imag,
                                      INTP batch, aoclfftz_strides_t *strides,
                                      VOID *UNUSED, UINT8 flag)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");
    UINT32 dt_bytes = sizeof(DOUBLE);
    DOUBLE *in_r = (DOUBLE *)in_real;
    DOUBLE *out_r = (DOUBLE *)out_real;

    if (flag == FORWARD_FFT_DIR)
    {
        INTP v_in_stride = strides->v_in_stride * dt_bytes;
        INTP v_out_stride = strides->v_out_stride * dt_bytes *
                            DATA_STRIDE; /* R2C: retain the real part and set
                                            imaginary to zero*/
        for (INTP i = 0; i < batch; i++)
        {
            out_r[0] = in_r[0];
            out_r[1] = 0.0;
            in_r = MOVE_ADDR(in_r, v_in_stride);
            out_r = MOVE_ADDR(out_r, v_out_stride);
        }
    }
    else
    {
        INTP v_in_stride = strides->v_in_stride * dt_bytes * DATA_STRIDE;
        INTP v_out_stride =
            strides->v_out_stride * dt_bytes; /*C2R: retain the real part*/
        for (INTP i = 0; i < batch; i++)
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
INT32 setup_real_sizeone_solver(aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    aoclfftz_strides_t *strides = sol->strides_grp->strides;
    strides->v_in_stride = sol->decomp_scheme->vecs[0].in_stride;
    strides->v_out_stride = sol->decomp_scheme->vecs[0].out_stride;

    UINT8 dt_prec = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    sol->solver->kernel_c2c->kfft = (dt_prec == DT_FLOAT)
                                        ? execute_real_float_kernel
                                        : execute_real_double_kernel;

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
static INT32 execute_real_sizeone_solver_internal(aoclfftz_solution_t *sol,
                                                  INTP vec_rank)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    if (vec_rank == 1)
    {
        kfft_ execute_innermost_batch = sol->solver->kernel_c2c->kfft;
        aoclfftz_strides_t *strides = sol->strides_grp->strides;
        execute_innermost_batch(
            sol->decomp_scheme->in_real, sol->decomp_scheme->in_imag,
            sol->decomp_scheme->out_real, sol->decomp_scheme->out_imag,
            sol->decomp_scheme->vecs[0].n, strides, sol->twiddle,
            FFT_DIR(sol->decomp_scheme->flags));
        return SOLVER_SUCCESS;
    }

    INTP batch;
    INTP v_in_stride;
    INTP v_out_stride;
    INT32 status = SOLVER_SUCCESS;
    UINT32 dt_bytes = SOL_DT_SIZE(sol);

    v_in_stride = sol->decomp_scheme->vecs[vec_rank - 1].in_stride * dt_bytes;
    v_out_stride = sol->decomp_scheme->vecs[vec_rank - 1].out_stride * dt_bytes;
    for (batch = 0; batch < sol->decomp_scheme->vecs[vec_rank - 1].n; batch++)
    {
        // save pointer to restore it below since
        // they will be moved while execution
        VOID *in_real = sol->decomp_scheme->in_real;
        VOID *out_real = sol->decomp_scheme->out_real;

        // recursive call to solve the inner batches
        status = execute_real_sizeone_solver_internal(sol, vec_rank - 1);
        if (status != SOLVER_SUCCESS)
        {
            return status;
        }

        // Adjust pointers for the next iteration
        sol->decomp_scheme->in_real = MOVE_ADDR(in_real, v_in_stride);
        sol->decomp_scheme->out_real = MOVE_ADDR(out_real, v_out_stride);
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
static INT32 execute_real_sizeone_solver(aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    // inplace check
    if (!IS_OUT_OF_PLACE(sol->decomp_scheme->flags) &&
        FFT_DIR(sol->decomp_scheme->flags))
    {
        return SOLVER_SUCCESS;
    }

    INTP vec_rank = sol->decomp_scheme->vec_rank;
    execute_real_sizeone_solver_internal(sol, vec_rank);

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
dft_solver_ register_execute_real_sizeone_solver(VOID)
{
    return execute_real_sizeone_solver;
}
