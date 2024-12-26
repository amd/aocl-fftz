/**
 * Copyright (C) 2025, Advanced Micro Devices. All rights reserved.
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
 *  @author Srirammaswamy Srinivasan
 */

#include "core/solvers/real/direct_solver.h"
#include "core/common/memory_manager.h"
#include "core/common/strides.h"
#include "utils/utils.h"

INT32 setup_real_direct_solver(aoclfftz_solution_t *sol, cost_analysis_t *cost,
                               kernel_t *kernel_c2c, kernel_t *kernel_r2hc,
                               kernel_t *kernel_r2hcf,
                               aoclfftz_realhelper_t *realhelper)
{
#ifdef AOCL_ENABLE_LOG
    INT32 logger_mode = sol->decomp_scheme->cntrl_params->logger_mode;
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Enter");
#endif

    INTP n = sol->decomp_scheme->vecs[0].n;
    INTP radix = sol->decomp_scheme->dims[0].n;
    UINT32 precision = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    UINT32 is_backward = FFT_DIR(sol->decomp_scheme->flags) == BACKWARD_FFT_DIR;
    INT32 status = SOLVER_SUCCESS;
    UINT8 sets_c2c = 0;
    UINT8 sets_r2hc = 0;
    UINT8 sets_r2hcf = 0;
    if (kernel_c2c != NULL)
    {
        sets_c2c = kernel_c2c->sets[precision - 2];
    }
    if (kernel_r2hc != NULL)
    {
        sets_r2hc = kernel_r2hc->sets[precision - 2];
    }
    if (kernel_r2hcf != NULL)
    {
        sets_r2hcf = kernel_r2hcf->sets[precision - 2];
    }

    if (realhelper->is_direct)
    {
        sol->solver->batches[C2C_KERNEL] = 0;
        sol->solver->batches[R2HC_KERNEL] = n;
        sol->solver->batches[R2HCF_KERNEL] = 0;
    }
    else // CT problem
    {
        // TODO: To be implemented
        AOCLFFTZ_LOG_UNFORMATTED(ERR, ERR, "Not implemented");
        return SOLVER_FAILURE;
    }

    // adjust the vec in-strides to complex adjusted values
    // for forward in-place real problems
    // TODO: handle this case for CT problems
    if (realhelper->is_direct && !is_backward &&
        !IS_OUT_OF_PLACE(sol->decomp_scheme->flags))
    {
        sol->decomp_scheme->vecs[0].in_stride *= 2;
    }

    INTP in_stride = sol->decomp_scheme->dims[0].in_stride;
    INTP out_stride = sol->decomp_scheme->dims[0].out_stride;
    INTP v_in_stride = sol->decomp_scheme->vecs[0].in_stride;
    INTP v_out_stride = sol->decomp_scheme->vecs[0].out_stride;

    // forward  : r2hc : real input -> half-complex output
    // backward : hc2r : half-complex input -> real output
    UINT32 is_half_complex_input = is_backward ? 1 : 0;
    UINT32 is_half_complex_output = is_backward ? 0 : 1;

    // true for direct solver and last level of CT solver
    // false for intermediate CT levels
    UINT32 adjust_to_full_complex = 1;

    if (sol->solver->batches[C2C_KERNEL] != 0)
    {
        if (sol->strides->in_strides == NULL)
        {
            ALLOC_ALIGN_UNINIT(sol->strides->in_strides, INTP,
                radix * sizeof(INTP));
            ALLOC_ALIGN_UNINIT(sol->strides->out_strides, INTP,
                radix * sizeof(INTP));
        }
        // TODO: To be implemented
    }
    if (sol->solver->batches[R2HC_KERNEL] != 0)
    {
        if (sol->strides_r2hc->in_strides == NULL)
        {
            ALLOC_ALIGN_UNINIT(sol->strides_r2hc->in_strides, INTP,
                radix * sizeof(INTP));
            ALLOC_ALIGN_UNINIT(sol->strides_r2hc->out_strides, INTP,
                radix * sizeof(INTP));
        }
        populate_stride_array(sol->strides_r2hc->in_strides, in_stride, radix,
                              is_half_complex_input, adjust_to_full_complex);
        populate_stride_array(sol->strides_r2hc->out_strides, out_stride, radix,
                              is_half_complex_output, adjust_to_full_complex);
        sol->strides_r2hc->v_in_stride = v_in_stride;
        sol->strides_r2hc->v_out_stride = v_out_stride;
        if (is_backward)
        {
            sol->strides_r2hc->v_in_stride *= 2;
        }
        else
        {
            sol->strides_r2hc->v_out_stride *= 2;
        }
    }
    if (sol->solver->batches[R2HCF_KERNEL] != 0)
    {
        if (sol->strides_r2hcf->in_strides == NULL)
        {
            ALLOC_ALIGN_UNINIT(sol->strides_r2hcf->in_strides, INTP,
                radix * sizeof(INTP));
            ALLOC_ALIGN_UNINIT(sol->strides_r2hcf->out_strides, INTP,
                radix * sizeof(INTP));
        }
        // TODO: To be implemented
    }

    if (GET_SELECTOR_MODE(sol->decomp_scheme->flags) ==
        AOCLFFTZ_FIXED_SELECTOR_MODE)
    {
        /** Fixed mode **/
        ops_cycles_t ops_cycles_c2c, ops_cycles_r2hc, ops_cycles_r2hcf;
        cost->time = 0;
        INT64 c2c_cost = 0;
        INT64 r2hc_cost = 0;
        INT64 r2hcf_cost = 0;
        if (sol->solver->batches[C2C_KERNEL] != 0)
        {
            ops_cycles_c2c = kernel_c2c->k_ops_cnt(precision, is_backward);
            c2c_cost = ((ops_cycles_c2c.fma * AMD_ZEN_FP_FMA_CYCLES) +
                        (ops_cycles_c2c.mul * AMD_ZEN_FP_MUL_CYCLES) +
                        (ops_cycles_c2c.add * AMD_ZEN_FP_ADD_CYCLES) +
                        (ops_cycles_c2c.move * AMD_ZEN_FP_MOVE_CYCLES) +
                        (ops_cycles_c2c.perm * AMD_ZEN_FP_PERM_CYCLES) +
                        (ops_cycles_c2c.other * AMD_ZEN_FP_OTHER_CYCLES));
            if (sol->solver->batches[C2C_KERNEL] >= sets_c2c)
            {
                c2c_cost = (c2c_cost + sets_c2c - 1) / sets_c2c;
            }
            c2c_cost = c2c_cost * sol->solver->batches[C2C_KERNEL];
        }
        if (sol->solver->batches[R2HC_KERNEL] != 0)
        {
            ops_cycles_r2hc = kernel_r2hc->k_ops_cnt(precision, is_backward);
            r2hc_cost = ((ops_cycles_r2hc.fma * AMD_ZEN_FP_FMA_CYCLES) +
                         (ops_cycles_r2hc.mul * AMD_ZEN_FP_MUL_CYCLES) +
                         (ops_cycles_r2hc.add * AMD_ZEN_FP_ADD_CYCLES) +
                         (ops_cycles_r2hc.move * AMD_ZEN_FP_MOVE_CYCLES) +
                         (ops_cycles_r2hc.perm * AMD_ZEN_FP_PERM_CYCLES) +
                         (ops_cycles_r2hc.other * AMD_ZEN_FP_OTHER_CYCLES));
            if (sol->solver->batches[R2HC_KERNEL] >= sets_r2hc)
            {
                r2hc_cost = (r2hc_cost + sets_r2hc - 1) / sets_r2hc;
            }
            r2hc_cost = r2hc_cost * sol->solver->batches[R2HC_KERNEL];
        }
        if (sol->solver->batches[R2HCF_KERNEL] != 0)
        {
            ops_cycles_r2hcf = kernel_r2hcf->k_ops_cnt(precision, is_backward);
            r2hcf_cost = ((ops_cycles_r2hcf.fma * AMD_ZEN_FP_FMA_CYCLES) +
                          (ops_cycles_r2hcf.mul * AMD_ZEN_FP_MUL_CYCLES) +
                          (ops_cycles_r2hcf.add * AMD_ZEN_FP_ADD_CYCLES) +
                          (ops_cycles_r2hcf.move * AMD_ZEN_FP_MOVE_CYCLES) +
                          (ops_cycles_r2hcf.perm * AMD_ZEN_FP_PERM_CYCLES) +
                          (ops_cycles_r2hcf.other * AMD_ZEN_FP_OTHER_CYCLES));
            if (sol->solver->batches[R2HCF_KERNEL] >= sets_r2hcf)
            {
                r2hcf_cost = (r2hcf_cost + sets_r2hcf - 1) / sets_r2hcf;
            }
            r2hcf_cost = r2hcf_cost * sol->solver->batches[R2HCF_KERNEL];
        }
        cost->ops = c2c_cost + r2hc_cost + r2hcf_cost;
    }

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit");
#endif
    return status;
}

static INT32 execute_real_direct_solver(aoclfftz_solution_t *sol)
{
#ifdef AOCL_ENABLE_LOG
    INT32 logger_mode = sol->decomp_scheme->cntrl_params->logger_mode;
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Enter");
#endif

    INT32 ret = SOLVER_SUCCESS;

    kfft_ kernel_c2c = sol->solver->kernel_c2c;
    kfft_ kernel_r2hc = sol->solver->kernel_r2hc;
    kfft_ kernel_r2hcf = sol->solver->kernel_r2hcf;

    // TODO: Fix execution order (r2hc -> r2hcf -> c2c)
    if (sol->solver->batches[R2HC_KERNEL] != 0)
    {
        kernel_r2hc(sol->decomp_scheme->in_real, sol->decomp_scheme->in_imag,
                    sol->decomp_scheme->out_real, sol->decomp_scheme->out_imag,
                    sol->solver->batches[R2HC_KERNEL], sol->strides_r2hc,
                    FFT_DIR(sol->decomp_scheme->flags));
    }
    if (sol->solver->batches[R2HCF_KERNEL] != 0)
    {
        kernel_r2hcf(sol->decomp_scheme->in_real, sol->decomp_scheme->in_imag,
                     sol->decomp_scheme->out_real, sol->decomp_scheme->out_imag,
                     sol->solver->batches[R2HCF_KERNEL], sol->strides_r2hcf,
                     FFT_DIR(sol->decomp_scheme->flags));
    }
    if (sol->solver->batches[C2C_KERNEL] != 0)
    {
        kernel_c2c(sol->decomp_scheme->in_real, sol->decomp_scheme->in_imag,
                   sol->decomp_scheme->out_real, sol->decomp_scheme->out_imag,
                   sol->solver->batches[C2C_KERNEL], sol->strides,
                   FFT_DIR(sol->decomp_scheme->flags));
    }

    // Required for iterative executor
    if (sol->next_sol != NULL)
    {
        ret = sol->next_sol->solver->execute_solver(sol->next_sol);
    }
    else if (FFT_DIR(sol->decomp_scheme->flags) == FORWARD_FFT_DIR)
    {
        // For R2C (real forward) problems, set the imaginary part of first and
        // last points in half-complex buffer to 0.
        // TODO: This should be moved to a separate solver
        INTP n = sol->decomp_scheme->dims[0].n * sol->decomp_scheme->vecs[0].n;
        INTP out_stride = sol->decomp_scheme->dims[0].out_stride;
        if (DT_PRECISION_FLAG(sol->decomp_scheme->flags) == DT_FLOAT)
        {
            FLOAT *out = (FLOAT *)sol->decomp_scheme->out_real;
            out[1] = 0.0f;
            out[n * out_stride + 1] = 0.0f;
        }
        else
        {
            DOUBLE *out = (DOUBLE *)sol->decomp_scheme->out_real;
            out[1] = 0.0;
            out[n * out_stride + 1] = 0.0;
        }
    }

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit");
#endif
    return ret;
}

dft_solver_ register_execute_real_direct_solver(VOID)
{
    return execute_real_direct_solver;
}
