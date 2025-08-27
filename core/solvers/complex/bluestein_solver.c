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

/** @file bluestein_solver.c
 *
 *  @brief Bluestein Solver that solves an input problem
 *
 *  This file contains the functions that setup, execute and destroy
 *  the solver.
 *
 *  @author Srirammaswamy Srininvasan
 */

#include <string.h> /* for memset, memcpy */
#include "core/common/bluestein_utils.h"
#include "core/common/memory_manager.h"

INT32 setup_bluestein_solver(aoclfftz_solution_t *sol,
                             aoclfftz_solution_t *next_sol, INTP m)
{
#ifdef AOCL_ENABLE_LOG
    INT32 logger_mode = sol->decomp_scheme->cntrl_params->logger_mode;
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Enter");
#endif

    // Setup next_sol with extended length m
    COPY_SOLUTION_OBJ(next_sol, sol);
    next_sol->decomp_scheme->dims[0].n = m;
    next_sol->decomp_scheme->dims[0].in_stride = 1;
    next_sol->decomp_scheme->dims[0].out_stride = 1;

    UINT8 dt_prec = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    UINT32 dt_bytes = DT_PRECISION_BYTES(dt_prec);

    if (alloc_bluestein_buffers(sol->dft_bufs->bluestein, m * DATA_STRIDE * dt_bytes) !=
        AOCLFFTZ_SUCCESS)
    {
        return AOCLFFTZ_MEMORY_FAILURE;
    }

    // Map the internal in, out buffers to the next solution
    next_sol->decomp_scheme->in_real = sol->dft_bufs->bluestein->in;
    next_sol->decomp_scheme->in_imag = MOVE_ADDR(sol->dft_bufs->bluestein->in, dt_bytes);
    next_sol->decomp_scheme->out_real = sol->dft_bufs->bluestein->out;
    next_sol->decomp_scheme->out_imag =
        MOVE_ADDR(sol->dft_bufs->bluestein->out, dt_bytes);
    next_sol->dft_bufs->interim_buf_ptr = sol->dft_bufs->bluestein->out;

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit");
#endif
    return SOLVER_SUCCESS;
}

static INT32 execute_bluestein_solver(aoclfftz_solution_t *sol)
{
#ifdef AOCL_ENABLE_LOG
    INT32 logger_mode = sol->decomp_scheme->cntrl_params->logger_mode;
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Enter");
#endif

    aoclfftz_solution_t *next_sol = sol->next_sol[0];
    UINT8 dt_prec = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    UINT32 dt_bytes = DT_PRECISION_BYTES(dt_prec);
    UINT32 dir = FFT_DIR(sol->decomp_scheme->flags);
    UINT32 mask = 1 << 2;
    UINT32 initial_flags = next_sol->decomp_scheme->flags;

    // Switch bwd flag to fwd for AVX kernels

    // As per the design, in bwd scenario, the AVX kernels expect swapped
    // in_real & in_imag pointers so that it can re-swap again within the kernel
    // to pick the correct in_real for processing. But in Bluestein solver,
    // the in_real & in_imag pointers are not swapped and hence there is a need
    // to prevent the re-swapping within the kernel, which can only be achieved
    // by setting the direction flag to Forward.
    if (dir == BACKWARD_FFT_DIR)
    {
        next_sol->decomp_scheme->flags ^= mask;
    }
    INTP n = sol->decomp_scheme->dims[0].n;           // original length
    INTP m = next_sol->decomp_scheme->dims[0].n; // extended length
    INT32 status = SOLVER_SUCCESS;

    INTP in_stride = sol->decomp_scheme->dims[0].in_stride;
    INTP out_stride = sol->decomp_scheme->dims[0].out_stride;

    INTP unit_val = 1;

    // Using sol->strides_grp->strides for permuted copy
    // FIX: A variable is referenced to stride pointer instead of using array
    // TODO: Revamp the permuted copy function to avoid this fix
    sol->strides_grp->strides->in_strides = &unit_val;
    sol->strides_grp->strides->out_strides = &unit_val;
    sol->strides_grp->strides->v_in_stride = 1;
    sol->strides_grp->strides->v_out_stride = 1;

    // Store the default in, out buffer addresses in separate pointers
    // next_sol->bluestein->in
    VOID *in_real = next_sol->decomp_scheme->in_real;
    VOID *in_imag = next_sol->decomp_scheme->in_imag;
    // next_sol->bluestein->out
    VOID *out_real = next_sol->decomp_scheme->out_real;
    VOID *out_imag = next_sol->decomp_scheme->out_imag;
    VOID *interim_buf_ptr = next_sol->dft_bufs->interim_buf_ptr;

    // Copy input from current sol to next sol
    VOID *cur_in  = sol->decomp_scheme->in_real;
    VOID *cur_out = sol->decomp_scheme->out_real;
    // Elementwise multiplication sign
    UINT8 mul_sign = (dir == FORWARD_FFT_DIR) ? 1 : 0;

    // Copy problem input to the next sol input buffer.
    // in/out strides of next sol will be 1.
    // If input stride of cur sol is more than 1, then do permuted copy to read
    // values with input stride from cur sol and write the values to next sol
    // with stride 1.
    if (in_stride > 1)
    {
        sol->strides_grp->strides->in_strides = &in_stride;
        sol->strides_grp->strides->out_strides = &unit_val;
        // TODO: Use registered function to avoid precision based condition
        if (dt_prec == DT_FLOAT)
        {
            permuted_copy_c_fp32(cur_in, in_real, 1, n, sol->strides_grp->strides,
                                 DATA_STRIDE);
        }
        else
        {
            permuted_copy_c_fp64(cur_in, in_real, 1, n, sol->strides_grp->strides,
                                 DATA_STRIDE);
        }
    }
    else
    {
        memcpy(in_real, cur_in, n * DATA_STRIDE * dt_bytes);
    }
    // Padding zeros in range n to m-1
    memset(MOVE_ADDR(in_real, n * DATA_STRIDE * dt_bytes), 0,
           (m - n) * DATA_STRIDE * dt_bytes);

    // Prepare bluestein sequence A
    // A = input * B_inv
    status = elementwise_multiplication(in_real, in_real, sol->dft_bufs->bluestein->B, n,
                                        mul_sign, dt_prec);
    if (status != BLUESTEIN_SUCCESS)
    {
        return SOLVER_FAILURE;
    }

    /****** convolution starts here ******/
    // 1. Perform FFT forward for bluestein sequence A

    // input  : in_real
    // output : out_real
    status = next_sol->solver->execute_solver(next_sol);
    if (status != SOLVER_SUCCESS)
    {
        return SOLVER_FAILURE;
    }

    // 2. Perform FFT forward for bluestein sequence B
    // this will be triggered only during the first execution of a bluestein
    // solution, further executions with same solution will re-use the computed
    // B_out data
    if (!sol->dft_bufs->bluestein->is_B_out_valid)
    {
        next_sol->decomp_scheme->in_real = sol->dft_bufs->bluestein->B;
        next_sol->decomp_scheme->in_imag =
            MOVE_ADDR(sol->dft_bufs->bluestein->B, dt_bytes);
        next_sol->decomp_scheme->out_real = sol->dft_bufs->bluestein->B_out;
        next_sol->decomp_scheme->out_imag =
            MOVE_ADDR(sol->dft_bufs->bluestein->B_out, dt_bytes);
        next_sol->dft_bufs->interim_buf_ptr = sol->dft_bufs->bluestein->B_out;

        // input  : sol->dft_bufs->bluestein->B
        // output : sol->dft_bufs->bluestein->B_out
        status = next_sol->solver->execute_solver(next_sol);
        if (status != SOLVER_SUCCESS)
        {
            return SOLVER_FAILURE;
        }

        sol->dft_bufs->bluestein->is_B_out_valid = 1;
    }

    // 3. Perform FFT backward for AB_out
    // AB_out = A_out * B_out (stored in A_out)
    status = elementwise_multiplication(
        out_real, out_real, sol->dft_bufs->bluestein->B_out, m, !mul_sign, dt_prec);
    if (status != BLUESTEIN_SUCCESS)
    {
        return SOLVER_FAILURE;
    }

    next_sol->decomp_scheme->in_real = out_real;
    next_sol->decomp_scheme->in_imag = out_imag;
    next_sol->decomp_scheme->out_real = in_real;
    next_sol->decomp_scheme->out_imag = in_imag;
    next_sol->dft_bufs->interim_buf_ptr = in_real;

    next_sol->decomp_scheme->flags ^= mask;

    // input  : out_imag
    // output : in_imag
    status = next_sol->solver->execute_solver(next_sol);
    if (status != SOLVER_SUCCESS)
    {
        return SOLVER_FAILURE;
    }

    // Normalize the AB_out with data length `m`
    normalize_data(in_real, n, (1.0 / m), dt_prec);

    /****** convolution ends here ******/

    // output = out_AB * B_inv
    // then, copy the output to the problem output
    // If ouput stride of cur sol is more than 1, then do permuted copy to read
    // values with stride 1 from next sol and write the values to cur sol with
    // output stride.
    if (out_stride > 1)
    {
        sol->strides_grp->strides->in_strides = &unit_val;
        sol->strides_grp->strides->out_strides = &out_stride;
        status = elementwise_multiplication(in_real, in_real, sol->dft_bufs->bluestein->B,
                                            n, mul_sign, dt_prec);
        // TODO: Use registered function to avoid precision based condition
        if (dt_prec == DT_FLOAT)
        {
            permuted_copy_c_fp32(in_real, cur_out, 1, n, sol->strides_grp->strides,
                                 DATA_STRIDE);
        }
        else
        {
            permuted_copy_c_fp64(in_real, cur_out, 1, n, sol->strides_grp->strides,
                                 DATA_STRIDE);
        }
    }
    else
    {
        status = elementwise_multiplication(cur_out, in_real, sol->dft_bufs->bluestein->B,
                                            n, mul_sign, dt_prec);
    }

    // Reset the stride pointers to NULL
    sol->strides_grp->strides->in_strides = NULL;
    sol->strides_grp->strides->out_strides = NULL;

    // Reset the in, out buffers of next solution to the original state
    next_sol->decomp_scheme->in_real = in_real;
    next_sol->decomp_scheme->in_imag = in_imag;
    next_sol->decomp_scheme->out_real = out_real;
    next_sol->decomp_scheme->out_imag = out_imag;
    next_sol->dft_bufs->interim_buf_ptr = interim_buf_ptr;
    next_sol->decomp_scheme->flags = initial_flags;

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit");
#endif
    return status;
}

dft_solver_ register_execute_bluestein_solver(VOID)
{
    return execute_bluestein_solver;
}
