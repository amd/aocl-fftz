// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file sizeone_solver_dft.c
 *
 *  @brief Size One solver that solves input problems with size as 1.
 *
 *  This file contains the function that execute the solver.
 *
 *  @author D. Vijay Krishna
 */

#include <string.h> // for memcpy
#include "core/solvers/solver.h"

static FFTZ_VOID execute_float_kernel(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                      FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                      FFTZ_INTP n, aoclfftz_strides_t *strides,
                                      FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");
    FFTZ_UINT32 dt_bytes = sizeof(FFTZ_FLOAT);
    FFTZ_INTP v_in_stride = strides->v_in_stride * dt_bytes * DATA_STRIDE;
    FFTZ_INTP v_out_stride = strides->v_out_stride * dt_bytes * DATA_STRIDE;

    for (FFTZ_INTP i = 0; i < n; i++)
    {
        memcpy(out_real, in_real, dt_bytes * DATA_STRIDE);
        in_real = (FFTZ_VOID *)((FFTZ_CHAR *)in_real + v_in_stride);
        out_real = (FFTZ_VOID *)((FFTZ_CHAR *)out_real + v_out_stride);
    }

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
}

static FFTZ_VOID execute_double_kernel(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                       FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                       FFTZ_INTP n, aoclfftz_strides_t *strides,
                                       FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");
    FFTZ_UINT32 dt_bytes = sizeof(FFTZ_DOUBLE);
    FFTZ_INTP v_in_stride = strides->v_in_stride * dt_bytes * DATA_STRIDE;
    FFTZ_INTP v_out_stride = strides->v_out_stride * dt_bytes * DATA_STRIDE;

    for (FFTZ_INTP i = 0; i < n; i++)
    {
        memcpy(out_real, in_real, dt_bytes * DATA_STRIDE);
        in_real = (FFTZ_VOID *)((FFTZ_CHAR *)in_real + v_in_stride);
        out_real = (FFTZ_VOID *)((FFTZ_CHAR *)out_real + v_out_stride);
    }

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
}

FFTZ_INT32 setup_sizeone_solver(aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    aoclfftz_strides_t *strides = sol->strides_grp->strides;
    strides->v_in_h2_stride = strides->v_in_stride =
        sol->decomp_scheme->vecs[0].in_stride;
    strides->v_out_h2_stride = strides->v_out_stride =
        sol->decomp_scheme->vecs[0].out_stride;

    FFTZ_UINT8 dt_prec = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    kfft_ kernel = (dt_prec == DT_FLOAT) ? execute_float_kernel
                                         : execute_double_kernel;
    sol->solver->kernel_c2c->kfft[FORWARD_FFT_DIR]  = kernel;
    sol->solver->kernel_c2c->kfft[BACKWARD_FFT_DIR] = kernel;

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SOLVER_SUCCESS;
}

static FFTZ_INT32 execute_sizeone_solver_internal(aoclfftz_solution_t *sol,
                                                  FFTZ_INTP vec_rank)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");


    if (vec_rank == 1)
    {
        // size-one kernel is bidirectional,
        // so both kfft[FORWARD_FFT_DIR] and kfft[BACKWARD_FFT_DIR] point to the
        // same kernel
        kfft_ execute_innermost_batch =
            sol->solver->kernel_c2c->kfft[FORWARD_FFT_DIR];
        aoclfftz_strides_t *strides = sol->strides_grp->strides;

        execute_innermost_batch(
            sol->decomp_scheme->in_real, sol->decomp_scheme->in_imag,
            sol->decomp_scheme->out_real, sol->decomp_scheme->out_imag,
            sol->decomp_scheme->vecs[0].n, strides,
            sol->twiddle,
            FFT_DIR(sol->decomp_scheme->flags));
        return SOLVER_SUCCESS;
    }

    FFTZ_INTP batch;
    FFTZ_INTP v_in_stride;
    FFTZ_INTP v_out_stride;
    FFTZ_INT32 status = SOLVER_SUCCESS;
    FFTZ_UINT32 dt_bytes = SOL_DT_SIZE(sol);

    v_in_stride = sol->decomp_scheme->vecs[vec_rank - 1].in_stride *
                  DATA_STRIDE * dt_bytes;
    v_out_stride = sol->decomp_scheme->vecs[vec_rank - 1].out_stride *
                   DATA_STRIDE * dt_bytes;
    for (batch = 0; batch < sol->decomp_scheme->vecs[vec_rank - 1].n; batch++)
    {
        // save pointer to restore it below since
        // they will be moved while execution
        FFTZ_VOID *in_real = sol->decomp_scheme->in_real;
        FFTZ_VOID *out_real = sol->decomp_scheme->out_real;

        //recursive call to solve the inner batches
        status = execute_sizeone_solver_internal(sol, vec_rank - 1);
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

static FFTZ_INT32 execute_sizeone_solver(aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    // inplace check
    if (!IS_OUT_OF_PLACE(sol->decomp_scheme->flags))
    {
        return SOLVER_SUCCESS;
    }

    FFTZ_INTP vec_rank = sol->decomp_scheme->vec_rank;
    // save pointer to restore it below since
    // they will be moved while execution
    FFTZ_VOID *in_real = sol->decomp_scheme->in_real;
    FFTZ_VOID *out_real = sol->decomp_scheme->out_real;

    execute_sizeone_solver_internal(sol, vec_rank);

    sol->decomp_scheme->in_real = in_real;
    sol->decomp_scheme->out_real = out_real;
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SOLVER_SUCCESS;
}

dft_solver_ register_execute_sizeone_solver(FFTZ_VOID)
{
    return execute_sizeone_solver;
}
