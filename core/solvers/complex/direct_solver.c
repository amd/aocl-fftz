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

/** @file direct_solver.c
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

#include "core/solvers/complex/direct_solver.h"
#include "core/common/memory_manager.h"
#include "utils/utils.h"

INT32 setup_direct_solver(aoclfftz_solution_t *sol, cost_analysis_t *cost,
                          kernel_t *kernel)
{
#ifdef AOCL_ENABLE_LOG
    INT32 logger_mode = sol->decomp_scheme->cntrl_params->logger_mode;
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Enter");
#endif

    aoclfftz_strides_t *strides = sol->strides;
    INTP n = sol->decomp_scheme->vecs[0].n;
    INTP radix = sol->decomp_scheme->dims[0].n;
    ops_cycles_t ops_cycles;
    UINT8 precision = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    UINT8 direction = FFT_DIR(sol->decomp_scheme->flags);
    INT32 status = SOLVER_SUCCESS;
    UINT8 sets = kernel->sets[precision - 2];

    if (strides->in_strides == NULL)
    {
        ALLOC_ALIGN_UNINIT(strides->in_strides, INTP, radix * sizeof(INTP));
        ALLOC_ALIGN_UNINIT(strides->out_strides, INTP, radix * sizeof(INTP));
        INTP in_stride = sol->decomp_scheme->dims[0].in_stride;
        INTP out_stride = sol->decomp_scheme->dims[0].out_stride;
        for (INTP i = 0; i < radix; i++)
        {
            strides->in_strides[i] = i * in_stride * DATA_STRIDE;
            strides->out_strides[i] = i * out_stride * DATA_STRIDE;
        }
    }

    strides->v_in_stride = sol->decomp_scheme->vecs[0].in_stride * DATA_STRIDE;
    strides->v_out_stride =
        sol->decomp_scheme->vecs[0].out_stride * DATA_STRIDE;

    if (GET_SELECTOR_MODE(sol->decomp_scheme->flags) ==
        AOCLFFTZ_FIXED_SELECTOR_MODE)
    {
        /** Fixed mode **/
        cost->time = 0;
        ops_cycles = kernel->k_ops_cnt(precision, direction);
        cost->ops = ((ops_cycles.fma * AMD_ZEN_FP_FMA_CYCLES) +
                     (ops_cycles.mul * AMD_ZEN_FP_MUL_CYCLES) +
                     (ops_cycles.add * AMD_ZEN_FP_ADD_CYCLES) +
                     (ops_cycles.move * AMD_ZEN_FP_MOVE_CYCLES) +
                     (ops_cycles.perm * AMD_ZEN_FP_PERM_CYCLES) +
                     (ops_cycles.other * AMD_ZEN_FP_OTHER_CYCLES));
        if (n >= sets)
        {
            cost->ops = (cost->ops + sets - 1) / sets; // ceil div
        }
        cost->ops = cost->ops * n;
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
                     n, strides, direction);

        getTime(endTime);
        cost->time = diffTime(clkTick, startTime, endTime);
        ops_cycles = kernel->k_ops_cnt(precision, direction);
        cost->ops = ((ops_cycles.fma * AMD_ZEN_FP_FMA_CYCLES) +
                     (ops_cycles.mul * AMD_ZEN_FP_MUL_CYCLES) +
                     (ops_cycles.add * AMD_ZEN_FP_ADD_CYCLES) +
                     (ops_cycles.move * AMD_ZEN_FP_MOVE_CYCLES) +
                     (ops_cycles.perm * AMD_ZEN_FP_PERM_CYCLES) +
                     (ops_cycles.other * AMD_ZEN_FP_OTHER_CYCLES));
        if (n >= sets)
        {
            cost->ops = (cost->ops + sets - 1) / sets; // ceil div
        }
        cost->ops = cost->ops * n;
    }

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit");
#endif
    return status;
}

static INT32 execute_direct_solver(aoclfftz_solution_t *sol)
{
#ifdef AOCL_ENABLE_LOG
    INT32 logger_mode = sol->decomp_scheme->cntrl_params->logger_mode;
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Enter");
#endif

    kfft_ kernel = sol->solver->kernel_r;
    aoclfftz_strides_t *strides = sol->strides;
    UINT8 direction = FFT_DIR(sol->decomp_scheme->flags);

    // execute the direct kernel
    kernel(sol->decomp_scheme->in_real, sol->decomp_scheme->in_imag,
           sol->decomp_scheme->out_real, sol->decomp_scheme->out_imag,
           sol->decomp_scheme->vecs[0].n, strides, direction);

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit");
#endif
    return SOLVER_SUCCESS;
}

dft_solver_ register_execute_direct_solver(VOID)
{
    return execute_direct_solver;
}
