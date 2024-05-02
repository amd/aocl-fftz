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

INT32 execute_sizeone_solver(aoclfftz_solution_t *sol)
{
    // inplace check
    if (!IS_OUT_OF_PLACE(sol->decomp_scheme->flags))
    {
        return SOLVER_SUCCESS;
    }
    INT32 logger_mode = sol->decomp_scheme->cntrl_params->logger_mode;
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Enter");
    UINT32 dt_prec, dt_bytes;
    dt_prec = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    DT_PRECISION_BYTES(dt_prec);
    dt_bytes = (dt_prec == DT_FLOAT) ? sizeof(FLOAT) : sizeof(DOUBLE);
    INTP out_buffer_size = sol->decomp_scheme->vecs[0].n;

    VOID *in  = (FFT_DIR(sol->decomp_scheme->flags)) ?
                sol->decomp_scheme->in_imag : sol->decomp_scheme->in_real;
    VOID *out = (FFT_DIR(sol->decomp_scheme->flags)) ?
                sol->decomp_scheme->out_imag : sol->decomp_scheme->out_real;

    INTP v_in_stride =
            sol->decomp_scheme->vecs[0].in_stride * DATA_STRIDE * dt_bytes;
    INTP v_out_stride =
            sol->decomp_scheme->vecs[0].out_stride * DATA_STRIDE * dt_bytes;

    for (INTP i = 0; i < out_buffer_size; i++)
    {
        // copy in buffer to out buffer
        memcpy(out, in, dt_bytes * DATA_STRIDE);

        in = (VOID *)((CHAR *)in + v_in_stride);
        out = (VOID *)((CHAR *)out + v_out_stride);
    }

    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit");
    return SOLVER_SUCCESS;
}
