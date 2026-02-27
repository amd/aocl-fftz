/**
 * Copyright (C) 2023-2025, Advanced Micro Devices. All rights reserved.
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

/** @file direct_solver_dft.c
 *
 *  @brief Direct Solver that applies an available kernel to the input problem
 *
 *  This file contains the functions that setup, execute and destroy
 *  the solver.
 *
 *  @author S. Biplab Raut
 *  @author Srirammaswamy Srinivasan
 *  @author Prasandh Sankarankutty
 */

#include "core/common/memory_manager.h"

INT32 setup_direct_solver(aoclfftz_solution_t *sol, cost_analysis_t *cost,
                          kernel_t *kernel)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");


    aoclfftz_strides_t *strides = sol->strides_grp->strides;
    INTP batch = sol->decomp_scheme->vecs[0].n;
    INTP radix = sol->decomp_scheme->dims[0].n;
    UINT8 precision = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    UINT8 direction = FFT_DIR(sol->decomp_scheme->flags);
    INT32 status = SOLVER_SUCCESS;

    if (strides->in_strides == NULL)
    {
        INT32 ret = alloc_and_fill_stride_arrays(strides, radix,
                        sol->decomp_scheme->dims[0].in_stride,
                        sol->decomp_scheme->dims[0].out_stride);
        if (ret != SOLVER_SUCCESS)
        {
            return ret;
        }
    }

    if (sol->decomp_scheme->batched_vecs != NULL)
    {
        strides->v_in_stride = sol->decomp_scheme->batched_vecs[0].in_stride * DATA_STRIDE;
        strides->v_out_stride =
            sol->decomp_scheme->batched_vecs[0].out_stride * DATA_STRIDE;
    }
    else
    {
        strides->v_in_stride = sol->decomp_scheme->vecs[0].in_stride * DATA_STRIDE;
        strides->v_out_stride =
            sol->decomp_scheme->vecs[0].out_stride * DATA_STRIDE;
    }

    if (GET_SELECTOR_MODE(sol->decomp_scheme->flags) ==
        AOCLFFTZ_FIXED_SELECTOR)
    {
        cost->time = 0;
        cost->ops = compute_kernel_cost(kernel, precision, direction, batch);
    }
    else
    {
        /** Auto tuner mode **/
#ifdef WIN32
        timer clkTick;
#endif
        timeVal startTime, endTime;
        initTimer(clkTick);
        getTime(startTime);

        // execute the direct kernel
        kernel->kfft(sol->decomp_scheme->in_real, sol->decomp_scheme->in_imag,
                     sol->decomp_scheme->out_real, sol->decomp_scheme->out_imag,
                     batch, strides, sol->twiddle, direction);

        getTime(endTime);
        cost->time = diffTime(clkTick, startTime, endTime);
        cost->ops = compute_kernel_cost(kernel, precision, direction, batch);
    }

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return status;
}

static INT32 execute_direct_solver(aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");


    kfft_ kernel = sol->solver->kernel_c2c->kfft;
    aoclfftz_strides_t *strides = sol->strides_grp->strides;
    UINT8 direction = FFT_DIR(sol->decomp_scheme->flags);

    // execute the direct kernel
    kernel(sol->decomp_scheme->in_real, sol->decomp_scheme->in_imag,
           sol->decomp_scheme->out_real, sol->decomp_scheme->out_imag,
           sol->decomp_scheme->vecs[0].n, strides, sol->twiddle, direction);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return SOLVER_SUCCESS;
}

/**
 * @brief Direct solver variant that handles two levels of batches in
 * column-major fashion.
 *
 * Unlike the standard direct solver which handles a single level of batches,
 * this variant processes two levels:
 *   - 1st level (outer loop): CT decomposition batches stored in vecs,
 *     representing DFT butterflies within a single problem
 *   - 2nd level (inner loop): Problem batches stored in batched_vecs,
 *     representing multiple independent FFT problems
 *
 * Column-major processing: Iterates over CT batches (vecs) in the outer loop,
 * executing all problem batches (batched_vecs) for each CT batch. This access
 * pattern is optimal when problem batch stride < elemental stride.
 */
static INT32 execute_direct_batched_colmajor_solver(aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");


    kfft_ kernel = sol->solver->kernel_c2c->kfft;
    aoclfftz_strides_t *strides = sol->strides_grp->strides;
    UINT8 direction = FFT_DIR(sol->decomp_scheme->flags);

    VOID *in_real = sol->decomp_scheme->in_real;
    VOID *in_imag = sol->decomp_scheme->in_imag;
    VOID *out_real = sol->decomp_scheme->out_real;
    VOID *out_imag = sol->decomp_scheme->out_imag;

    UINT32 dt_bytes = SOL_DT_SIZE(sol);

    // vec-strides across DFT butterflies of the same CT problem
    INTP ct_in_stride =
        sol->decomp_scheme->vecs[0].in_stride * DATA_STRIDE * dt_bytes;
    INTP ct_out_stride =
        sol->decomp_scheme->vecs[0].out_stride * DATA_STRIDE * dt_bytes;

    // execute the direct kernel
    for (INTP i = 0; i < sol->decomp_scheme->vecs[0].n; i++)
    {
        aoclfftz_twiddle_t tw_local = {
            .twiddle_buf_ptr = sol->twiddle->twiddle_buf_ptr,
            .TW = MOVE_ADDR(sol->twiddle->TW, i * DATA_STRIDE * dt_bytes),
            .cols = sol->twiddle->cols,
            .load_multi_cols = 0, // use same twiddle values across batches
        };                        // since different batches solve the same
                                  // DFT butterfly of different problems
        kernel(in_real, in_imag, out_real, out_imag,
               sol->decomp_scheme->batched_vecs[0].n, strides, &tw_local,
               direction);
        in_real = MOVE_ADDR(in_real, ct_in_stride);
        in_imag = MOVE_ADDR(in_imag, ct_in_stride);
        out_real = MOVE_ADDR(out_real, ct_out_stride);
        out_imag = MOVE_ADDR(out_imag, ct_out_stride);
    }

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return SOLVER_SUCCESS;
}

dft_solver_ register_execute_direct_solver(VOID)
{
    return execute_direct_solver;
}

dft_solver_ register_execute_direct_batched_colmajor_solver(VOID)
{
    return execute_direct_batched_colmajor_solver;
}
