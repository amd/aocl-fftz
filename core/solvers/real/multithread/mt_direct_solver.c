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

/** @file mt_direct_solver.c
 *
 *  @brief Multi threaded direct real fft solver that enables multi threading
 *  for the available real fft direct kernels
 *
 *  This file contains the functions that setup and execute the solver.
 *
 *  @author Partiksha
 */

#include "core/solvers/real/direct_solver.h"
#include "api/aoclfftz_internal.h"
#include "core/common/memory_manager.h"
#include "core/common/realfft_utils.h"
#include "core/common/strides.h"
#include "core/common/twiddle.h"
#include "selector/selector.h"

/* This function will setup the direct solution with the required information
   to execute both direct problem and CT r subproblem.

   Even for a CT problem, most of the kernel execution information is required
   by a direct solution.

   Setup includes the following steps:
     1. Set the strides for different kernel variants (C2C, R2HC, R2HCF)
     2. Setting up the no. of batch for each kernel variant
     3. Updating the input & output buffers for CT problem/sub-problem
     4. Cost computation

   NOTE: This direct solver will handle both direct and CT problems.
   TODO: Introduce two different solvers one to setup and execute real direct
         problems and another one for CT problems.
 */
INT32 setup_real_mt_direct_solver(aoclfftz_solution_t *sol,
                                  cost_analysis_t *cost,
                                  const kernel_t *kernel_c2c,
                                  const kernel_t *kernel_r2hc,
                                  const kernel_t *kernel_r2hcf,
                                  aoclfftz_realhelper_t *realhelper)
{
#ifdef AOCL_ENABLE_LOG
    INT32 logger_mode = sol->decomp_scheme->cntrl_params->logger_mode;
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Enter");
#endif

    INTP batch = sol->decomp_scheme->vecs[0].n;
    INTP radix = sol->decomp_scheme->dims[0].n;
    UINT32 precision = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    UINT32 is_backward = FFT_DIR(sol->decomp_scheme->flags) == BACKWARD_FFT_DIR;
    INT32 status = SOLVER_SUCCESS;
    UINT8 sets_c2c = kernel_c2c->sets[precision - 2];
    UINT8 sets_r2hc = kernel_r2hc->sets[precision - 2];
    UINT8 sets_r2hcf = kernel_r2hcf->sets[precision - 2];
    INT32 avl_threads = sol->decomp_scheme->thread_info->avl_threads;
    INTP total_batches = 0;

    INTP p = 0;
    INTP p_last = 0;
    INTP q = 0;
    if (is_backward)
    {
        q = realhelper->q;
        p = realhelper->p;
        p_last = realhelper->p / radix;
    }
    else
    {
        p_last = realhelper->p;
        q = realhelper->q / radix;
        p = realhelper->p * radix;
    }
    UINT32 is_even = p_last % 2 == 0;
    UINT32 is_first_stage = realhelper->stage == 0;
    UINT32 is_last_stage = realhelper->is_last_stage;

    // Compute batches and strides for different kernel types
    // Original strides for the given problem
    INTP org_in_stride = sol->decomp_scheme->dims[0].in_stride;
    INTP org_out_stride = sol->decomp_scheme->dims[0].out_stride;
    INTP org_v_in_stride = sol->decomp_scheme->vecs[0].in_stride;
    INTP org_v_out_stride = sol->decomp_scheme->vecs[0].out_stride;
    // Newly computed strides
    INTP in_stride = 1;
    INTP out_stride = 1;
    INTP v_in_stride = 1;
    INTP v_out_stride = 1;
    INTP c2c_in_stride = 1;
    INTP c2c_out_stride = 1;

    if (realhelper->is_CT)
    {
        sol->solver->batches[C2C_KERNEL] = (p_last / 2 - is_even) * q;
        sol->solver->batches[R2HC_KERNEL] = is_even ? 0 : q;
        sol->solver->batches[R2HCF_KERNEL] = is_even ? q : 0;
        // TODO: Reduce redundant assignments
        if (is_backward)
        {
            if (is_first_stage)
            {
                in_stride = p_last * org_in_stride;
                out_stride = batch;
                v_in_stride = p * org_in_stride;
                v_out_stride = p_last;
                c2c_in_stride = org_in_stride;
            }
            else if (is_last_stage)
            {
                in_stride = p_last;
                out_stride = batch * org_out_stride;
                v_in_stride = p;
                v_out_stride = p_last * org_out_stride;
            }
            else
            {
                in_stride = p_last;
                out_stride = batch;
                v_in_stride = p;
                v_out_stride = p_last;
            }
        }
        else
        {
            if (is_first_stage)
            {
                in_stride = batch * org_in_stride;
                out_stride = p_last;
                v_in_stride = p_last * org_in_stride;
                v_out_stride = p;
            }
            else if (is_last_stage)
            {
                in_stride = batch;
                out_stride = p_last * org_out_stride;
                v_in_stride = p_last;
                v_out_stride = p * org_out_stride;
                c2c_out_stride = org_out_stride;
            }
            else
            {
                in_stride = batch;
                out_stride = p_last;
                v_in_stride = p_last;
                v_out_stride = p;
            }
        }
    }
    else // direct problem
    {
        sol->solver->batches[C2C_KERNEL] = 0;
        sol->solver->batches[R2HC_KERNEL] = batch;
        sol->solver->batches[R2HCF_KERNEL] = 0;
        in_stride = org_in_stride;
        out_stride = org_out_stride;
        v_in_stride = is_backward ? org_v_in_stride * 2 : org_v_in_stride;
        v_out_stride = is_backward ? org_v_out_stride : org_v_out_stride * 2;
    }

    // Setting the number of threads based on the batches
    if (sol->solver->batches[C2C_KERNEL])
    {
        total_batches = sol->solver->batches[C2C_KERNEL] +
                        (sol->solver->batches[R2HC_KERNEL] + 1) / 2 +
                        sol->solver->batches[R2HCF_KERNEL];
    }
    else
    {
        total_batches = sol->solver->batches[C2C_KERNEL] +
                        sol->solver->batches[R2HC_KERNEL] +
                        sol->solver->batches[R2HCF_KERNEL];
    }
    UINT32 n_threads = (total_batches < avl_threads) ?
                       total_batches : avl_threads;
    sol->decomp_scheme->thread_info->n_threads = n_threads;
    INTP no_of_groups = sol->solver->batches[R2HCF_KERNEL] > 0
                            ? sol->solver->batches[R2HCF_KERNEL]
                            : sol->solver->batches[R2HC_KERNEL];
    INTP group_size = sol->solver->batches[C2C_KERNEL] / no_of_groups;

    // Compute strides for different kernels from the base stride values
    UINT32 input_in_full_complex = 0;
    UINT32 output_in_full_complex = 0;
    if (realhelper->is_CT)
    {
        if (is_backward) // CT backward
        {
            input_in_full_complex = is_first_stage;
            output_in_full_complex = 0;
        }
        else // CT forward
        {
            input_in_full_complex = 0;
            output_in_full_complex = is_last_stage;
        }
    }
    else
    {
        input_in_full_complex = is_backward;
        output_in_full_complex = !is_backward;
    }

    // forward  : r2hc : real input -> half-complex output
    // backward : hc2r : half-complex input -> real output
    UINT32 is_half_complex_input = is_backward ? 1 : 0;
    UINT32 is_half_complex_output = is_backward ? 0 : 1;

    // FIXME: Moving this inside the if condition below causes issue with
    // the C2R CT problem
    if (sol->strides_grp->strides->in_strides == NULL)
    {
        ALLOC_ALIGN_UNINIT(sol->strides_grp->strides->in_strides, INTP,
                           radix * sizeof(INTP));
        ALLOC_ALIGN_UNINIT(sol->strides_grp->strides->out_strides, INTP,
                           radix * sizeof(INTP));
    }
    if (sol->solver->batches[C2C_KERNEL] != 0)
    {
        if (sol->strides_grp->strides->in_strides == NULL)
        {
            ALLOC_ALIGN_UNINIT(sol->strides_grp->strides->in_strides, INTP,
                               radix * sizeof(INTP));
            ALLOC_ALIGN_UNINIT(sol->strides_grp->strides->out_strides, INTP,
                               radix * sizeof(INTP));
        }
        populate_stride_array(sol->strides_grp->strides->in_strides,
                              is_backward ? in_stride * 2 : in_stride, radix,
                              0, 0); /* half-complex flags are false */
        populate_stride_array(sol->strides_grp->strides->out_strides,
                              is_backward ? out_stride : out_stride * 2, radix,
                              0, 0); /* half-complex flags are false */
        if (sol->strides_grp->strides_c2c->in_strides == NULL)
        {
            ALLOC_ALIGN_UNINIT(sol->strides_grp->strides_c2c->in_strides, INTP,
                               radix * sizeof(INTP));
            ALLOC_ALIGN_UNINIT(sol->strides_grp->strides_c2c->out_strides, INTP,
                               radix * sizeof(INTP));
        }
        memcpy(sol->strides_grp->strides_c2c->in_strides, sol->strides_grp->strides->in_strides,
               radix * sizeof(INTP));
        memcpy(sol->strides_grp->strides_c2c->out_strides, sol->strides_grp->strides->out_strides,
               radix * sizeof(INTP));

        if (is_backward)
        {
            prepare_real_c2c_kernel_strides(
                sol->strides_grp->strides->in_strides, sol->strides_grp->strides->in_strides,
                radix, p, c2c_in_stride);
        }
        else
        {
            prepare_real_c2c_kernel_strides(
                sol->strides_grp->strides->out_strides, sol->strides_grp->strides->out_strides,
                radix, p, c2c_out_stride);
        }
    }
    if (sol->solver->batches[R2HC_KERNEL] != 0)
    {
        if (sol->strides_grp->strides_r2hc->in_strides == NULL)
        {
            ALLOC_ALIGN_UNINIT(sol->strides_grp->strides_r2hc->in_strides, INTP,
                               radix * sizeof(INTP));
            ALLOC_ALIGN_UNINIT(sol->strides_grp->strides_r2hc->out_strides, INTP,
                               radix * sizeof(INTP));
        }
        populate_stride_array(sol->strides_grp->strides_r2hc->in_strides, in_stride, radix,
                              is_half_complex_input, input_in_full_complex);
        populate_stride_array(sol->strides_grp->strides_r2hc->out_strides, out_stride, radix,
                              is_half_complex_output, output_in_full_complex);
    }
    if (sol->solver->batches[R2HCF_KERNEL] != 0)
    {
        if (sol->strides_grp->strides_r2hcf->in_strides == NULL)
        {
            ALLOC_ALIGN_UNINIT(sol->strides_grp->strides_r2hcf->in_strides, INTP,
                               radix * 2 * sizeof(INTP));
            ALLOC_ALIGN_UNINIT(sol->strides_grp->strides_r2hcf->out_strides, INTP,
                               radix * 2 * sizeof(INTP));
        }
        populate_stride_array(sol->strides_grp->strides_r2hcf->in_strides,
                              is_backward ? in_stride / 2 : in_stride,
                              is_backward ? radix * 2 : radix,
                              is_half_complex_input, input_in_full_complex);
        populate_stride_array(sol->strides_grp->strides_r2hcf->out_strides,
                              is_backward ? out_stride : out_stride / 2,
                              is_backward ? radix : radix * 2,
                              is_half_complex_output, output_in_full_complex);
        prepare_fused_kernel_strides(is_backward
                                         ? sol->strides_grp->strides_r2hcf->out_strides
                                         : sol->strides_grp->strides_r2hcf->in_strides,
                                     radix, group_size * 2 + 1);
    }

    sol->strides_grp->strides->v_in_stride = v_in_stride;
    sol->strides_grp->strides->v_out_stride = v_out_stride;
    sol->strides_grp->strides_c2c->v_in_stride = v_in_stride;
    sol->strides_grp->strides_c2c->v_out_stride = v_out_stride;
    sol->strides_grp->strides_r2hc->v_in_stride = v_in_stride;
    sol->strides_grp->strides_r2hc->v_out_stride = v_out_stride;
    sol->strides_grp->strides_r2hcf->v_in_stride = v_in_stride;
    sol->strides_grp->strides_r2hcf->v_out_stride = v_out_stride;

    UINT8 dt_prec = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    UINT32 dt_bytes = DT_PRECISION_BYTES(dt_prec);

    // Update the in/out buffers of direct solution for CT problem
    //
    // solution from of CT problem after buffered sol
    // ... -> buffered -> direct -> CT -> direct -> ... -> CT -> direct
    //
    // Here, the buffered will have in & out of the current batch
    //
    // Buffered solver will change the input/output buffers of direct solution
    // in the following way:
    //
    // buffered    [in -> out]
    // |--> direct   [in -> aux1]
    // |----> CT
    // |----> direct   [aux1 -> aux2]
    // |------> CT
    // |------> direct   [aux2 -> aux1]
    // |--------> CT
    // |--------> direct   [aux1 -> out]
    // this example is for a 3 level CT problem

    if (realhelper->is_CT)
    {
        VOID *in_real = NULL;
        VOID *out_real = NULL;
        if (realhelper->stage & 0x1) // odd stage
        {
            in_real = sol->dft_bufs->buffered->aux_buffer_2;
            out_real = sol->dft_bufs->buffered->aux_buffer_1;
        }
        else // even stage
        {
            in_real = sol->dft_bufs->buffered->aux_buffer_1;
            out_real = sol->dft_bufs->buffered->aux_buffer_2;
        }
        if (realhelper->stage == 0)
        {
            sol->decomp_scheme->out_real = out_real;
            sol->decomp_scheme->out_imag = MOVE_ADDR(out_real, dt_bytes);
        }
        else if (realhelper->is_last_stage)
        {
            sol->decomp_scheme->in_real = in_real;
            sol->decomp_scheme->in_imag = MOVE_ADDR(in_real, dt_bytes);
        }
        else
        {
            sol->decomp_scheme->in_real = in_real;
            sol->decomp_scheme->in_imag = MOVE_ADDR(in_real, dt_bytes);
            sol->decomp_scheme->out_real = out_real;
            sol->decomp_scheme->out_imag = MOVE_ADDR(out_real, dt_bytes);
        }
    }

    // Compute cost
    if (GET_SELECTOR_MODE(sol->decomp_scheme->flags) ==
        AOCLFFTZ_FIXED_SELECTOR)
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

static VOID execute_real_kernel(aoclfftz_solution_t *sol, UINT32 n_threads_real)
{

    UINT32 dt_prec = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    UINT32 dt_bytes = DT_PRECISION_BYTES(dt_prec);

    VOID *in = sol->decomp_scheme->in_real;
    VOID *out = sol->decomp_scheme->out_real;

    omp_set_num_threads(n_threads_real);
    /* Execute R2HC Kernels */
    if (sol->solver->batches[R2HC_KERNEL] != 0)
    {
        kfft_ kernel_r2hc = sol->solver->kernel_r2hc->kfft;
        UINT8 num_sets_r2hc = sol->solver->kernel_r2hc->sets;

        // vector stride prep for R2HC kernels
        INTP v_in_stride_r2hc, v_out_stride_r2hc;
        v_in_stride_r2hc  = sol->strides_grp->strides_r2hc->v_in_stride
                            * dt_bytes;
        v_out_stride_r2hc = sol->strides_grp->strides_r2hc->v_out_stride
                            * dt_bytes;
        INTP num_iters_r2hc = sol->solver->batches[R2HC_KERNEL] / num_sets_r2hc;
        INTP rem_iters_r2hc = sol->solver->batches[R2HC_KERNEL] -
                              (num_iters_r2hc * num_sets_r2hc);

        #pragma omp parallel for
        for (INTP batch = 0; batch < num_iters_r2hc; batch++)
        {
            INTP v_istride = batch * v_in_stride_r2hc;
            INTP v_ostride = batch * v_out_stride_r2hc;
            kernel_r2hc((VOID *)((CHAR *)in + v_istride),
                        (VOID *)((CHAR *)in + v_istride),
                        (VOID *)((CHAR *)out + v_ostride),
                        (VOID *)((CHAR *)out + v_ostride),
                        num_sets_r2hc, sol->strides_grp->strides_r2hc,
                        sol->twiddle->TW, FFT_DIR(sol->decomp_scheme->flags));
        }
        /* Process the tail cases of the kernel */
        if(rem_iters_r2hc)
        {
            INTP v_istride = num_iters_r2hc * v_in_stride_r2hc;
            INTP v_ostride = num_iters_r2hc * v_out_stride_r2hc;
            kernel_r2hc((VOID *)((CHAR *)in + v_istride),
                        (VOID *)((CHAR *)in + v_istride),
                        (VOID *)((CHAR *)out + v_ostride),
                        (VOID *)((CHAR *)out + v_ostride),
                        rem_iters_r2hc, sol->strides_grp->strides_r2hc,
                        sol->twiddle->TW, FFT_DIR(sol->decomp_scheme->flags));
        }
    }
    /* Execute R2HCF Kernels */
    else if (sol->solver->batches[R2HCF_KERNEL] != 0)
    {
        kfft_ kernel_r2hcf = sol->solver->kernel_r2hcf->kfft;
        UINT8 num_sets_r2hcf = sol->solver->kernel_r2hcf->sets;

        // vector stride prep for R2HCF kernels
        INTP v_in_stride_r2hcf, v_out_stride_r2hcf;
        v_in_stride_r2hcf  = sol->strides_grp->strides_r2hcf->v_in_stride
                             * dt_bytes;
        v_out_stride_r2hcf = sol->strides_grp->strides_r2hcf->v_out_stride
                             * dt_bytes;
        INTP num_iters_r2hcf = sol->solver->batches[R2HCF_KERNEL]
                               / num_sets_r2hcf;
        INTP rem_iters_r2hcf = sol->solver->batches[R2HCF_KERNEL] -
                               (num_iters_r2hcf * num_sets_r2hcf);

        #pragma omp parallel for
        for (INTP batch = 0; batch < num_iters_r2hcf; batch++)
        {
            INTP v_istride = batch * v_in_stride_r2hcf;
            INTP v_ostride = batch * v_out_stride_r2hcf;
            kernel_r2hcf((VOID *)((CHAR *)in + v_istride),
                         (VOID *)((CHAR *)in + v_istride),
                         (VOID *)((CHAR *)out + v_ostride),
                         (VOID *)((CHAR *)out + v_ostride),
                         num_sets_r2hcf, sol->strides_grp->strides_r2hcf,
                         sol->twiddle->TW, FFT_DIR(sol->decomp_scheme->flags));
        }
        /* Process the tail cases of the kernel */
        if(rem_iters_r2hcf)
        {
            INTP v_istride = num_iters_r2hcf * v_in_stride_r2hcf;
            INTP v_ostride = num_iters_r2hcf * v_out_stride_r2hcf;
            kernel_r2hcf((VOID *)((CHAR *)in + v_istride),
                         (VOID *)((CHAR *)in + v_istride),
                         (VOID *)((CHAR *)out + v_ostride),
                         (VOID *)((CHAR *)out + v_ostride),
                         rem_iters_r2hcf, sol->strides_grp->strides_r2hcf,
                         sol->twiddle->TW, FFT_DIR(sol->decomp_scheme->flags));
        }
    }
}

/* This function will execute the kernels for both real direct and CT problems.

   For real direct problem, it will execute R2HC kernels.
   For real CT problems, following steps will be performed:
     1. Call R2HC/R2HCF kernels
     2. Perform twiddle multiplication for the C2C kernel points
     3. Get the no. of groups and group size for C2C kernels
     4. Update C2C kernel strides for each kernel within a group
     5. Execute C2C kernels
     6. Get conjugates for the required C2C points
        TODO: Move conjugates functionality into C2C kernels
 */
static INT32 execute_real_mt_direct_solver(aoclfftz_solution_t *sol)
{
#ifdef AOCL_ENABLE_LOG
    INT32 logger_mode = sol->decomp_scheme->cntrl_params->logger_mode;
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Enter");
#endif

    INT32 ret = SOLVER_SUCCESS;

    VOID *in = sol->decomp_scheme->in_real;
    VOID *out = sol->decomp_scheme->out_real;

    // Kernel execution order : R2HC / R2HCF -> C2C
    // R2HC and R2HCF will not come together

    // Execute R2HC Kernels
    UINT8 is_direct_only_problem = IS_DIRECT_ONLY_PROBLEM(sol);
    if (is_direct_only_problem)
    {
        execute_real_kernel(sol, sol->decomp_scheme->thread_info->n_threads);
        if (FFT_DIR(sol->decomp_scheme->flags) == FORWARD_FFT_DIR)
        {
            set_zero_for_dc_and_nyquist_batched(sol);
        }
        return ret;
    }

    UINT8 dt_prec = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    UINT32 dt_bytes = DT_PRECISION_BYTES(dt_prec);
    UINT32 is_backward = FFT_DIR(sol->decomp_scheme->flags) == BACKWARD_FFT_DIR;
    UINT32 n_threads = sol->decomp_scheme->thread_info->n_threads;
    kfft_ kernel_c2c = sol->solver->kernel_c2c->kfft;

    INTP radix = sol->decomp_scheme->dims[0].n;
    INTP no_of_groups = sol->solver->batches[R2HCF_KERNEL] > 0
                            ? sol->solver->batches[R2HCF_KERNEL]
                            : sol->solver->batches[R2HC_KERNEL];
    INTP group_size = sol->solver->batches[C2C_KERNEL] / no_of_groups;
    INTP p = (sol->decomp_scheme->vecs[0].n * radix) / no_of_groups;

    // strides between C2C kernels
    INTP c2c_in_stride = 1;
    INTP c2c_out_stride = 1;
    // strides between R2HC/R2HCF and C2C kernel
    INTP r2hc_in_stride = 1;
    INTP r2hc_out_stride = 1;
    // Set strides based on the CT stages
    // TODO: Fix the design
    // Current version of CT uses strides only in first and last CT stages
    // i.e. strided input points will be read and the intermediate outputs
    // will be stored without strides and the final output will be written to
    // output buffer with strides
    UINT32 is_last_stage = sol->next_sol == NULL;
    UINT32 is_last_forward_stage = !is_backward && is_last_stage;
    UINT32 is_first_backward_stage = is_backward && no_of_groups == 1;
    if (is_first_backward_stage)
    {
        c2c_in_stride = sol->decomp_scheme->dims[0].in_stride;
        r2hc_in_stride = (c2c_in_stride - 1) * 2 + 1;
    }
    else if (is_last_forward_stage)
    {
        c2c_out_stride = sol->decomp_scheme->dims[0].out_stride;
        r2hc_out_stride = (c2c_out_stride - 1) * 2 + 1;
    }

    // vector stride prep for C2C kernels
    INTP v_in_stride_c2c, v_out_stride_c2c;
    v_in_stride_c2c = sol->strides_grp->strides_c2c->v_in_stride * dt_bytes;
    v_out_stride_c2c = sol->strides_grp->strides_c2c->v_out_stride * dt_bytes;
    INTP batch_real = 0;
    INTP total_batches = 0;
    // Execute C2C Kernels
    if (sol->solver->batches[C2C_KERNEL])
    {
        batch_real = (sol->solver->batches[R2HC_KERNEL] > 0) ?
                      (sol->solver->batches[R2HC_KERNEL] + 1) / 2 :
                      sol->solver->batches[R2HCF_KERNEL];
        // Calculate the number of threads for real solver and complex kernels
        // based on their respective number of batches
        total_batches = sol->solver->batches[C2C_KERNEL] +
                        (sol->solver->batches[R2HC_KERNEL] + 1) / 2  +
                        sol->solver->batches[R2HCF_KERNEL];
    }
    else
    {
        batch_real = (sol->solver->batches[R2HC_KERNEL] > 0) ?
                      sol->solver->batches[R2HC_KERNEL] :
                      sol->solver->batches[R2HCF_KERNEL];
        // Calculate the number of threads for real solver and complex kernels
        // based on their respective number of batches
        total_batches = sol->solver->batches[C2C_KERNEL] +
                        sol->solver->batches[R2HC_KERNEL] +
                        sol->solver->batches[R2HCF_KERNEL];
    }

    UINT32 n_threads_real = (n_threads * batch_real) / total_batches;
    UINT32 n_threads_c2c = n_threads - n_threads_real;

    /*
     * Parallelization Strategy:
     * Parallelize the execution blocks of both real and complex kernels only
     * when there are batches[R2HC/R2HCF] > 0 and batches[C2C] > 0.
     */
    int enable_parallelism = (sol->solver->batches[C2C_KERNEL] > 0);

    // Kernel execution order : R2HC / R2HCF -> C2C
    // R2HC and R2HCF will not come together
    if (enable_parallelism)
    {
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                // this function will execute the real kernels r2hc/r2hcf and
                // the reason behind to move the code to a separate function
                // is to avoid the code duplication in the case of CT problem
                // where the real kernels are executed in parallel with C2C
                // kernels.
                execute_real_kernel(sol, n_threads_real);
            }
            // Execute C2C Kernels
            #pragma omp section
            {
                VOID *in_c2c = MOVE_ADDR(in, r2hc_in_stride * dt_bytes);
                VOID *out_c2c = MOVE_ADDR(out, r2hc_out_stride * dt_bytes);
                if (is_last_stage)
                {
                    out_c2c = MOVE_ADDR(out_c2c, dt_bytes);
                }
                if (sol->solver->batches[C2C_KERNEL] != 0)
                {
                    // setting the number of threads for inner C2C kernel
                    // execution
                    UINT32 n_threads_c2c_outer = n_threads_c2c > group_size ?
                                                 group_size : n_threads_c2c;
                    UINT32 n_threads_c2c_inner = n_threads_c2c /
                                                 n_threads_c2c_outer;
                    INTP half_stride_start = (radix + 1) >> 1;
                    INTP half_stride_n = radix - half_stride_start;
                    if (is_backward)
                    {
                        // Move in buffer by one element only for the first CT
                        // stage where the no. of group of C2C kernel is 1
                        if (no_of_groups == 1)
                        {
                            in_c2c = MOVE_ADDR(in_c2c, dt_bytes);
                        }

                        // Copy unmodified in-stride values from strides to
                        // strides_c2c and then modify strides_c2c values 
                        // within loop iterations
                        memcpy(sol->strides_grp->strides_c2c->in_strides
                               + half_stride_start,
                               sol->strides_grp->strides->in_strides
                                + half_stride_start,
                               half_stride_n * sizeof(INTP));

                        INTP stride_offset = c2c_in_stride * 4;
                        #pragma omp parallel for \
                                num_threads(n_threads_c2c_outer)
                        for (INTP group_id = 0; group_id < group_size;
                             group_id++)
                        {
                            VOID *in_local = MOVE_ADDR(in_c2c,
                                    group_id * c2c_in_stride * 2 * dt_bytes);
                            VOID *out_local = MOVE_ADDR(out_c2c,
                                    group_id * c2c_out_stride * 2 * dt_bytes);

                            aoclfftz_strides_t *strides_c2c_per_thread;
                            ALLOC_ALIGN_UNINIT(strides_c2c_per_thread,
                                aoclfftz_strides_t, sizeof(aoclfftz_strides_t));
                            memcpy(strides_c2c_per_thread,
                                   sol->strides_grp->strides_c2c,
                                   sizeof(aoclfftz_strides_t));

                            // Allocate and copy a private in_strides array for
                            // this thread
                            INTP *local_in_strides;
                            ALLOC_ALIGN_UNINIT(local_in_strides, INTP,
                                sizeof(INTP) * radix);
                            memcpy(local_in_strides,
                                   sol->strides_grp->strides_c2c->in_strides,
                                   sizeof(INTP) * radix);
                            strides_c2c_per_thread->in_strides =
                                                    local_in_strides;

                            // Update the C2C in-strides for next iteration by
                            // subtracting it by stride_offset
                            for (INTP i = 0; i < half_stride_n; i++) {
                                local_in_strides[half_stride_start + i] -=
                                        group_id * stride_offset;
                            }
                            // Take complex conjucate for the required points
                            // TODO: this should be moved into C2C kernel
                            compute_conjugates(
                                in_local, radix, no_of_groups,
                                strides_c2c_per_thread->in_strides,
                                strides_c2c_per_thread->v_in_stride,
                                DT_PRECISION_FLAG(sol->decomp_scheme->flags));

                            #pragma omp parallel for \
                                    num_threads(n_threads_c2c_inner)
                            for (INTP group_num = 0; group_num < no_of_groups;
                                 group_num++)
                            {
                                INTP v_istride = group_num * v_in_stride_c2c;
                                INTP v_ostride = group_num * v_out_stride_c2c;
                                kernel_c2c((VOID *)((CHAR *)in_local +
                                            v_istride + dt_bytes),
                                           (VOID *)((CHAR *)in_local +
                                            v_istride),
                                           (VOID *)((CHAR *)out_local +
                                            v_ostride + dt_bytes),
                                           (VOID *)((CHAR *)out_local +
                                            v_ostride),
                                           1 /* num_sets_c2c */,
                                           strides_c2c_per_thread,
                                           sol->twiddle->TW,
                                           FFT_DIR(sol->decomp_scheme->flags));
                            }
                            FREE_ALIGN_ALLOCATED_MEM(local_in_strides);
                            FREE_ALIGN_ALLOCATED_MEM(strides_c2c_per_thread);
                        }

                        /* FIXIT: */
                        // Real CT executor uses Direct-R -> CT -> Direct-M
                        // flow, so in this approach it is not possible to get
                        // the Direct-R info for the next CT executor to perform
                        // twiddle multiplication. Hence the twiddle
                        // multiplication is performed here itself.

                        // Selective twiddle multiplication on C2C kernel output
                        // points only
                        twiddle_multiplier_for_real(sol, p);
                    }
                    else
                    {
                        // Selective twiddle multiplication on C2C kernel input
                        // points only
                        twiddle_multiplier_for_real(sol, p);

                        // Copy unmodified out-stride values from strides to
                        // strides_c2c and then modify strides_c2c values within
                        // loop iterations
                        memcpy(sol->strides_grp->strides_c2c->out_strides
                               + half_stride_start,
                               sol->strides_grp->strides->out_strides
                               + half_stride_start,
                               half_stride_n * sizeof(INTP));

                        INTP stride_offset = c2c_out_stride * 4;
                        #pragma omp parallel for \
                                num_threads(n_threads_c2c_outer)
                        for (INTP group_id = 0; group_id < group_size;
                             group_id++)
                        {
                            VOID *in_local = MOVE_ADDR(in_c2c,
                                    group_id * c2c_in_stride * 2 * dt_bytes);
                            VOID *out_local = MOVE_ADDR(out_c2c,
                                    group_id * c2c_out_stride * 2 * dt_bytes);

                            aoclfftz_strides_t *strides_c2c_per_thread;
                            ALLOC_ALIGN_UNINIT(strides_c2c_per_thread,
                                aoclfftz_strides_t, sizeof(aoclfftz_strides_t));
                            memcpy(strides_c2c_per_thread,
                                   sol->strides_grp->strides_c2c,
                                   sizeof(aoclfftz_strides_t));

                            // Allocate and copy a private in_strides array for
                            // this thread
                            INTP *local_out_strides;
                            ALLOC_ALIGN_UNINIT(local_out_strides, INTP,
                                sizeof(INTP) * radix);
                            memcpy(local_out_strides,
                                   sol->strides_grp->strides_c2c->out_strides,
                                   sizeof(INTP) * radix);
                            strides_c2c_per_thread->out_strides =
                                                    local_out_strides;

                            // Update the C2C in-strides for next iteration by
                            // subtracting it by stride_offset
                            for (INTP i = 0; i < half_stride_n; i++) {
                                local_out_strides[half_stride_start + i] -=
                                                group_id * stride_offset;
                            }

                            #pragma omp parallel for \
                                    num_threads(n_threads_c2c_inner)
                            for (INTP group_num = 0; group_num < no_of_groups;
                                 group_num++)
                            {
                                INTP v_istride = group_num * v_in_stride_c2c;
                                INTP v_ostride = group_num * v_out_stride_c2c;
                                kernel_c2c((VOID *)((CHAR *)in_local
                                            + v_istride),
                                           (VOID *)((CHAR *)in_local
                                            + v_istride + dt_bytes),
                                           (VOID *)((CHAR *)out_local
                                            + v_ostride),
                                           (VOID *)((CHAR *)out_local
                                            + v_ostride + dt_bytes),
                                           1 /* num_sets_c2c */,
                                           strides_c2c_per_thread,
                                           sol->twiddle->TW,
                                           FFT_DIR(sol->decomp_scheme->flags));
                            }
                            // Take complex conjucate for the required points
                            // TODO: this should be moved into C2C kernel
                            compute_conjugates(
                                out_local, radix, no_of_groups,
                                strides_c2c_per_thread->out_strides,
                                strides_c2c_per_thread->v_out_stride,
                                DT_PRECISION_FLAG(sol->decomp_scheme->flags));

                            FREE_ALIGN_ALLOCATED_MEM(local_out_strides);
                            FREE_ALIGN_ALLOCATED_MEM(strides_c2c_per_thread);
                        }
                    }
                }
            }
        }
    }
    else
    {
        // Execute when problem is not direct and batches[R2HC/R2HCF] > 0
        execute_real_kernel(sol, n_threads_real);
    }

    // Iteratively execute the next solution
    if (sol->next_sol != NULL && sol->next_sol[0] != NULL)
    {
        ret = sol->next_sol[0]->solver->execute_solver(sol->next_sol[0]);
    }
    else if (FFT_DIR(sol->decomp_scheme->flags) == FORWARD_FFT_DIR)
    {
        set_zero_for_dc_and_nyquist(sol);
    }

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit");
#endif
    return ret;
}

dft_solver_ register_execute_real_mt_direct_solver(VOID)
{
    return execute_real_mt_direct_solver;
}
