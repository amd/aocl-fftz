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

#include <assert.h>
#include "core/solvers/real/direct_solver.h"
#include "api/aoclfftz_internal.h"
#include "core/common/memory_manager.h"
#include "core/common/strides.h"
#include "core/common/twiddle.h"
#include "selector/selector.h"
#include "core/solvers/real/direct_solver_utils.h"

/** This function will setup the direct solution with the required information
 *  to execute both direct and CT problems.
 *  Even for a CT problem, most of the kernel execution information is required
 *  by a direct solution.
 *  Setup includes the following steps:
 *    1. Set the strides for different kernel variants (C2C, R2HC, R2HCF)
 *    2. Setting up the no. of batch for each kernel variant
 *    3. Updating the input & output buffers for CT problem/sub-problem
 *    4. Cost computation
 *  NOTE: This direct solver will handle both direct and CT problems.
 *  TODO: Separate solver responsibilities for better maintainability:
 *        - Direct solver should handle only direct FFT problems (R2HC kernels)
 *        - CT solver should handle only Cooley-Tukey decomposition problems
 *         (C2C + R2HC/R2HCF kernels)
 */
INT32 setup_real_direct_solver(aoclfftz_solution_t *sol, cost_analysis_t *cost,
                               const kernel_t *kernel_c2c,
                               const kernel_t *kernel_r2hc,
                               const kernel_t *kernel_r2hcf,
                               aoclfftz_realhelper_t *realhelper)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");
    INT32 status = SOLVER_SUCCESS;

    set_kernel_count_in_each_group(sol, realhelper);

    allocate_and_setup_stride(sol, *realhelper);

    update_ct_buffers(sol, realhelper);

    compute_cost(sol, cost, kernel_c2c, kernel_r2hc, kernel_r2hcf);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return status;
}

/**
 * Executes R2HC (Real to Half-Complex) kernels and processes direct-only
 * FFT problems.
 */
static inline VOID execute_r2hc_kernels(aoclfftz_solution_t *sol, VOID *in, VOID *out)
{
    if (sol->solver->kernel_r2hc->count == 0)
    {
        return;
    }

    kfft_ kernel_r2hc = sol->solver->kernel_r2hc->kfft;
    kernel_r2hc(in, in, out, out, sol->solver->kernel_r2hc->count,
                sol->strides_grp->strides_r2hc, sol->twiddle,
                FFT_DIR(sol->decomp_scheme->flags));
}

/**
 * Executes Real-to-Half-Complex Fused (R2HCF) kernels
 */
static inline VOID execute_r2hcf_kernels(aoclfftz_solution_t *sol, VOID *in, VOID *out)
{
    // Execute R2HCF kernels (for CT problems)
    if (sol->solver->kernel_r2hcf->count == 0)
    {
        return;
    }

    kfft_ kernel_r2hcf = sol->solver->kernel_r2hcf->kfft;
    kernel_r2hcf(in, in, out, out, sol->solver->kernel_r2hcf->count,
                 sol->strides_grp->strides_r2hcf, sol->twiddle,
                 FFT_DIR(sol->decomp_scheme->flags));
}

static inline VOID execute_c2c_kernels(aoclfftz_solution_t *sol, VOID *in, VOID *out)
{
    if (sol->solver->kernel_c2c->count == 0)
    {
        return;
    }

    UINT32 dt_bytes = SOL_DT_SIZE(sol);
    INTP radix = sol->decomp_scheme->dims[0].n;
    kfft_ kernel_c2c = sol->solver->kernel_c2c->kfft;
    UINT32 is_fwd = FFT_DIR(sol->decomp_scheme->flags) == FORWARD_FFT_DIR;

    INTP num_groups = NUM_RFFT_GROUPS(sol->solver);
    INTP num_c2c_per_group = sol->solver->kernel_c2c->count / num_groups;
    INTP freq_factor =
        (sol->decomp_scheme->vecs[0].n * radix) / (num_groups);

    INTP half_stride_start = (radix + 1) >> 1;
    INTP half_stride_n = radix - half_stride_start;

    INTP batch_in_stride =
        is_input_prob_buffer(sol)
            ? sol->decomp_scheme->dims[0].in_stride * DATA_STRIDE
            : DATA_STRIDE;
    INTP batch_out_stride =
        is_output_prob_buffer(sol)
            ? sol->decomp_scheme->dims[0].out_stride * DATA_STRIDE
            : DATA_STRIDE;
    if (is_fwd)
    {
        if (sol->solver->solver_type != SOLVER_REAL_DIRECT_TWIDDLE)
        {
            twiddle_multiplier_for_real(sol, freq_factor);
        }
        memcpy(sol->strides_grp->strides_c2c->out_strides + half_stride_start,
               sol->strides_grp->strides->out_strides + half_stride_start,
               half_stride_n * sizeof(INTP));

        aoclfftz_twiddle_t tw_local = *(sol->twiddle);
        tw_local.load_multi_cols = 0; // use same twiddle values across batches
        for (INTP group_id = 0; group_id < num_c2c_per_group; group_id++)
        {
            // This for loop computes C2C batches within the groups,
            // while the kernel does across multiple groups

            kernel_c2c(in, MOVE_ADDR(in, dt_bytes), out,
                       MOVE_ADDR(out, dt_bytes), num_groups,
                       sol->strides_grp->strides_c2c, &tw_local,
                       FFT_DIR(sol->decomp_scheme->flags));

            compute_conjugates(out, radix, num_groups,
                               sol->strides_grp->strides_c2c->out_strides,
                               sol->strides_grp->strides_c2c->v_out_stride,
                               DT_PRECISION_FLAG(sol->decomp_scheme->flags));

            update_asymmetric_strides(sol->strides_grp->strides_c2c->out_strides,
                                     radix, batch_out_stride);

            // move twiddle buffer to next batch
            tw_local.TW =
                MOVE_ADDR(tw_local.TW, DATA_STRIDE * dt_bytes);
            // Move the in & out buffers to point the next batch
            in = MOVE_ADDR(in, batch_in_stride * dt_bytes);
            out = MOVE_ADDR(out, batch_out_stride * dt_bytes);
        }
    }
    else
    {
        assert(sol->solver->solver_type != SOLVER_REAL_DIRECT_TWIDDLE);

        memcpy(sol->strides_grp->strides_c2c->in_strides + half_stride_start,
               sol->strides_grp->strides->in_strides + half_stride_start,
               half_stride_n * sizeof(INTP));
        for (INTP group_id = 0; group_id < num_c2c_per_group; group_id++)
        {
            VOID *kernel_in = NULL;
            VOID *kernel_strides = NULL;
            if (IS_OUT_OF_PLACE(sol->decomp_scheme->flags) &&
                is_input_prob_buffer(sol))
            {
                kernel_in = sol->dft_bufs->ct_buf_real_in;
                kernel_strides = sol->strides_grp->strides_c2r_ct_op;
                compute_conjugates_outplace(
                    kernel_in, in, radix, num_groups,
                    sol->strides_grp->strides_c2c->in_strides,
                    sol->strides_grp->strides_c2c->v_in_stride,
                    DT_PRECISION_FLAG(sol->decomp_scheme->flags));
            }
            else
            {
                kernel_in = in;
                kernel_strides = sol->strides_grp->strides_c2c;
                compute_conjugates(
                    in, radix, num_groups,
                    sol->strides_grp->strides_c2c->in_strides,
                    sol->strides_grp->strides_c2c->v_in_stride,
                    DT_PRECISION_FLAG(sol->decomp_scheme->flags));
            }

            kernel_c2c(kernel_in, MOVE_ADDR(kernel_in, dt_bytes), out,
                       MOVE_ADDR(out, dt_bytes), num_groups, kernel_strides,
                       NULL, FFT_DIR(sol->decomp_scheme->flags));

            update_asymmetric_strides(sol->strides_grp->strides_c2c->in_strides,
                                     radix, batch_in_stride);

            in = MOVE_ADDR(in, batch_in_stride * dt_bytes);
            out = MOVE_ADDR(out, batch_out_stride * dt_bytes);
        }
        twiddle_multiplier_for_real(sol, freq_factor);
    }
}

/**
 * @brief Executes the real direct solver on the given solution
 *
 * A RFFT stage may have:
 * - 0 or 1 R2HC kernels
 * - 0 or more C2C kernels
 * - 0 or 1 R2HCF kernels for CT stages
 * This function computes all the kernels and calls the next solution if available.
 *
 * @param sol Pointer to the solution structure containing solver configuration
 * @return SOLVER_SUCCESS on successful execution, error code otherwise
 */
static INT32 execute_real_direct_solver(aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");


    INT32 ret = SOLVER_SUCCESS;
    VOID *in = sol->decomp_scheme->in_real;
    VOID *out = sol->decomp_scheme->out_real;
    UINT32 is_forward = FFT_DIR(sol->decomp_scheme->flags) == FORWARD_FFT_DIR;

    execute_r2hc_kernels(sol, in, out);
    if (IS_DIRECT_ONLY_PROBLEM(sol))
    {
        assert(sol->solver->kernel_r2hc->count > 0 &&
               sol->solver->kernel_c2c->count == 0 &&
               sol->solver->kernel_r2hcf->count == 0);

        //NOTE: DIRECT problems don't have C2C, R2HCF kernels
        if (is_forward)
            set_zero_for_dc_and_nyquist_batched(sol);

        return ret;
    }

    execute_r2hcf_kernels(sol, in, out);

    UINT32 dt_bytes = SOL_DT_SIZE(sol);
    INTP in_offset = is_input_prob_buffer(sol)
                         ? sol->decomp_scheme->dims[0].in_stride * DATA_STRIDE
                         : 1;
    INTP out_offset = is_output_prob_buffer(sol)
                          ? sol->decomp_scheme->dims[0].out_stride * DATA_STRIDE
                          : 1;
    // move in,out pointers to the start of C2C points, by skipping R2HC points
    in = MOVE_ADDR(in, in_offset * dt_bytes);
    out = MOVE_ADDR(out, out_offset * dt_bytes);

    execute_c2c_kernels(sol, in, out);

    if (HAS_NEXT(sol))
    {
        ret = sol->next_sol[0]->solver->execute_solver(sol->next_sol[0]);
    }
    else if (is_forward)
    {
        set_zero_for_dc_and_nyquist(sol);
    }

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return ret;
}

dft_solver_ register_execute_real_direct_solver(VOID)
{
    return execute_real_direct_solver;
}
