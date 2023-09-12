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

#include "core/solvers/direct_solver.h"
#include "core/common/memory_manager.h"
#include "utils/utils.h"

INT32 setup_direct_solver(aoclfftz_solution_t *sol,
                          cost_analysis_t *cost,
                          kernel_t *kernel)
{
    INT32 logger_mode = sol->decomp_scheme->cntrl_params->logger_mode;
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Enter");

    aoclfftz_strides_t *strides = sol->strides;
    ptrdiff_t n = sol->decomp_scheme->vecs[0].n;
    ops_cycles_t ops_cycles;
    UINT32 precision = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    INT32 status = SOLVER_SUCCESS;

    strides->in_stride = sol->decomp_scheme->dims[0].in_stride;
    strides->out_stride = sol->decomp_scheme->dims[0].out_stride;
    strides->v_in_stride = sol->decomp_scheme->vecs[0].in_stride;
    strides->v_out_stride = sol->decomp_scheme->vecs[0].out_stride;

    if (GET_SELECTOR_MODE(sol->decomp_scheme->flags) == AOCLFFTZ_FIXED_SELECTOR_MODE)
    {
        /** Fixed mode **/
        cost->time = 0;
        ops_cycles = kernel->k_ops_cnt(precision);
        cost->ops = n * ((ops_cycles.fma * AMD_ZEN_FP_FMA_CYCLES) +
                         (ops_cycles.mul * AMD_ZEN_FP_FMA_CYCLES) +
                         (ops_cycles.add * AMD_ZEN_FP_FMA_CYCLES) +
                         (ops_cycles.move * AMD_ZEN_FP_FMA_CYCLES) +
                         (ops_cycles.perm * AMD_ZEN_FP_FMA_CYCLES) +
                         (ops_cycles.other * AMD_ZEN_FP_FMA_CYCLES));
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
        kernel->kfft(sol->decomp_scheme->in_real,
                     sol->decomp_scheme->in_imag,
                     sol->decomp_scheme->out_real,
                     sol->decomp_scheme->out_imag,
                     n,
                     strides);

        getTime(endTime);
        cost->time = diffTime(clkTick, startTime, endTime);
        ops_cycles = kernel->k_ops_cnt(precision);
        cost->ops = n * ((ops_cycles.fma * AMD_ZEN_FP_FMA_CYCLES) +
                         (ops_cycles.mul * AMD_ZEN_FP_FMA_CYCLES) +
                         (ops_cycles.add * AMD_ZEN_FP_FMA_CYCLES) +
                         (ops_cycles.move * AMD_ZEN_FP_FMA_CYCLES) +
                         (ops_cycles.perm * AMD_ZEN_FP_FMA_CYCLES) +
                         (ops_cycles.other * AMD_ZEN_FP_FMA_CYCLES));
    }

    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit");
    return status;
}

INT32 execute_direct_solver(aoclfftz_solution_t *sol)
{
    INT32 logger_mode = sol->decomp_scheme->cntrl_params->logger_mode;
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Enter");

    kfft_ kernel = sol->solver->kernel_r;
    aoclfftz_strides_t *strides = sol->strides;
    if (strides == NULL)
    {
        AOCLFFTZ_LOG_UNFORMATTED(ERR, logger_mode, "Invalid Strides");
        return SOLVER_FAILURE;
    }

    AOCLFFTZ_LOG_FORMATTED(TRACE, logger_mode, "Executing Radix-%ld kernel",
                           sol->decomp_scheme->dims[0].n);
    // execute the direct kernel
    kernel(sol->decomp_scheme->in_real,
           sol->decomp_scheme->in_imag,
           sol->decomp_scheme->out_real,
           sol->decomp_scheme->out_imag,
           sol->decomp_scheme->vecs[0].n,
           strides);

    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit");
    return SOLVER_SUCCESS;
}