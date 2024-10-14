/**
 * Copyright (C) 2024, Advanced Micro Devices. All rights reserved.
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

/** @file sizeone_solver.c
 *
 *  @brief Size One solver that solves input problems with size as 1.
 *
 *  This file contains the function that execute the solver.
 *
 *  @author Varun Sanjay
 */

#include "core/solvers/sizeone_solver.h"
#include "core/common/memory_manager.h"
#include "utils/utils.h"

static VOID execute_float_kernel(VOID *in_real, VOID *in_imag, VOID *out_real,
                          VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                          UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    UINT32 dt_bytes = sizeof(FLOAT);
    INTP v_in_stride = strides->v_in_stride * dt_bytes * DATA_STRIDE;
    INTP v_out_stride = strides->v_out_stride * dt_bytes * DATA_STRIDE;

    for (INTP i = 0; i < n; i++)
    {
        memcpy(out_real, in_real, dt_bytes * DATA_STRIDE);
        in_real = (VOID *)((CHAR *)in_real + v_in_stride);
        out_real = (VOID *)((CHAR *)out_real + v_out_stride);
    }

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

static VOID execute_double_kernel(VOID *in_real, VOID *in_imag, VOID *out_real,
                           VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                           UINT8 flag)
{
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Enter");
#endif
    UINT32 dt_bytes = sizeof(DOUBLE);
    INTP v_in_stride = strides->v_in_stride * dt_bytes * DATA_STRIDE;
    INTP v_out_stride = strides->v_out_stride * dt_bytes * DATA_STRIDE;

    for (INTP i = 0; i < n; i++)
    {
        memcpy(out_real, in_real, dt_bytes * DATA_STRIDE);
        in_real = (VOID *)((CHAR *)in_real + v_in_stride);
        out_real = (VOID *)((CHAR *)out_real + v_out_stride);
    }

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, TRACE, "Exit");
#endif
}

INT32 setup_sizeone_solver(aoclfftz_solution_t *sol)
{
#ifdef AOCL_ENABLE_LOG
    INT32 logger_mode = sol->decomp_scheme->cntrl_params->logger_mode;
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Enter");
#endif
    aoclfftz_strides_t *strides = sol->strides;
    strides->v_in_stride = sol->decomp_scheme->vecs[0].in_stride;
    strides->v_out_stride = sol->decomp_scheme->vecs[0].out_stride;

    VOID *in = (FFT_DIR(sol->decomp_scheme->flags)) ?
                sol->decomp_scheme->in_imag :
                sol->decomp_scheme->in_real;
    VOID *out = (FFT_DIR(sol->decomp_scheme->flags)) ?
                sol->decomp_scheme->out_imag :
                sol->decomp_scheme->out_real;
    sol->decomp_scheme->in_real = in;
    sol->decomp_scheme->out_real = out;

    UINT32 dt_prec;
    dt_prec = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    sol->solver->kernel_r =
        (dt_prec == DT_FLOAT) ? execute_float_kernel : execute_double_kernel;

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit");
#endif
    return SOLVER_SUCCESS;
}

static INT32 execute_sizeone_solver_internal(aoclfftz_solution_t *sol, INTP vec_rank)
{
#ifdef AOCL_ENABLE_LOG
    INT32 logger_mode = sol->decomp_scheme->cntrl_params->logger_mode;
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Enter");
#endif

    if (vec_rank == 1)
    {
        kfft_ execute_innermost_batch = sol->solver->kernel_r;
        aoclfftz_strides_t *strides = sol->strides;

        execute_innermost_batch(
            sol->decomp_scheme->in_real, sol->decomp_scheme->in_imag,
            sol->decomp_scheme->out_real, sol->decomp_scheme->out_imag,
            sol->decomp_scheme->vecs[0].n, strides,
            FFT_DIR(sol->decomp_scheme->flags));
        return SOLVER_SUCCESS;
    }

    UINT32 dt_prec, dt_bytes;
    INTP batch;
    INTP v_in_stride;
    INTP v_out_stride;
    INT32 status = SOLVER_SUCCESS;
    dt_prec = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    dt_bytes = DT_PRECISION_BYTES(dt_prec);

    v_in_stride = sol->decomp_scheme->vecs[vec_rank - 1].in_stride *
                  DATA_STRIDE * dt_bytes;
    v_out_stride = sol->decomp_scheme->vecs[vec_rank - 1].out_stride *
                   DATA_STRIDE * dt_bytes;
    for (batch = 0; batch < sol->decomp_scheme->vecs[vec_rank - 1].n; batch++)
    {
        // save pointer to restore it below since
        // they will be moved while execution
        VOID *in_real = sol->decomp_scheme->in_real;
        VOID *out_real = sol->decomp_scheme->out_real;

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

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit");
#endif
    return SOLVER_SUCCESS;
}

static INT32 execute_sizeone_solver(aoclfftz_solution_t *sol)
{
#ifdef AOCL_ENABLE_LOG
    INT32 logger_mode = sol->decomp_scheme->cntrl_params->logger_mode;
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Enter");
#endif
    // inplace check
    if (!IS_OUT_OF_PLACE(sol->decomp_scheme->flags))
    {
        return SOLVER_SUCCESS;
    }

    INTP vec_rank = sol->decomp_scheme->vec_rank;
    // save pointer to restore it below since
    // they will be moved while execution
    VOID *in_real = sol->decomp_scheme->in_real;
    VOID *out_real = sol->decomp_scheme->out_real;

    execute_sizeone_solver_internal(sol, vec_rank);

    sol->decomp_scheme->in_real = in_real;
    sol->decomp_scheme->out_real = out_real;
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit");
#endif
    return SOLVER_SUCCESS;
}

dft_solver_ register_execute_sizeone_solver(VOID)
{
    return execute_sizeone_solver;
}
