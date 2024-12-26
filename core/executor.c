/**
 * Copyright (C) 2023, Advanced Micro Devices. All rights reserved.
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

/** @file executor.c
 *
 *  @brief Executes the DFT problem based on the solution found by selector.
 *
 *  This file contains the functions to execute a solution of kernels for the
 *  given input problem description.
 *
 *  @author S. Biplab Raut
 *  @author Prasandh Sankarankutty
 */

#include "core/executor.h"
#include "core/common/realfft_utils.h"

static INT32 execute_dft(aoclfftz_executor_t *executor_obj)
{
    aoclfftz_solution_t *sol = executor_obj->solution;
    // Re-order input buffer for batched in-place real forward (R2C) problems
    UINT32 flags = sol->decomp_scheme->flags;
    UINT8 is_batched_inplace_real_fwd =
        (sol->decomp_scheme->vecs[0].n > 1) && !IS_OUT_OF_PLACE(flags) &&
        IS_REAL(flags) && (FFT_DIR(flags) == FORWARD_FFT_DIR);
    if (is_batched_inplace_real_fwd)
    {
        REORDER_INPUT(sol);
    }
    return sol->solver->execute_solver(sol);
}

execute_ register_execute_dft(VOID)
{
    return execute_dft;
}
