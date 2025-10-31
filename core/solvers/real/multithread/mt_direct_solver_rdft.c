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

/** @file mt_direct_solver_rdft.c
 *
 *  @brief Multi threaded direct real fft solver that enables multi threading
 *  for the available real fft direct kernels
 *
 *  This file contains the functions that setup and execute the solver.
 *
 *  @author Partiksha
 */

#include <assert.h>
#include "core/common/memory_manager.h"
#include "core/common/twiddle.h"
#include "core/solvers/real/direct_solver_rdft_utils.h"

/* This function will setup the direct solution with the required information
 *  to execute both direct problem and CT r subproblem for MT.
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
INT32 setup_real_mt_direct_solver(aoclfftz_solution_t *sol,
                                  cost_analysis_t *cost,
                                  const kernel_t *kernel_c2c,
                                  const kernel_t *kernel_r2hc,
                                  const kernel_t *kernel_r2hcf,
                                  aoclfftz_realhelper_t *realhelper)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");
    INT32 status = SOLVER_SUCCESS;

    INT32 avl_threads = sol->decomp_scheme->thread_info->avl_threads;

    set_kernel_count_in_each_group(sol, realhelper);

    // Setting the number of threads based on the batches
    UINTP c2c_batches = sol->solver->kernel_c2c->count;
    UINTP r2hc_batches = sol->solver->kernel_r2hc->count;
    UINTP r2hcf_batches = sol->solver->kernel_r2hcf->count;
    if (c2c_batches)
    {
        r2hc_batches = (r2hc_batches + 1) / DATA_STRIDE;
    }
    INTP total_batches = r2hc_batches + r2hcf_batches + c2c_batches;
    sol->decomp_scheme->thread_info->n_threads = (avl_threads < total_batches)
                                               ? avl_threads : total_batches;

    allocate_and_setup_stride(sol, *realhelper);

    update_ct_buffers(sol, realhelper);

    compute_cost(sol, cost, kernel_c2c, kernel_r2hc, kernel_r2hcf);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return status;
}

static inline VOID execute_real_mt_r2c_kernels(aoclfftz_solution_t *sol, INT32 n_threads_real)
{

    UINT32 dt_bytes = SOL_DT_SIZE(sol);

    VOID *in = sol->decomp_scheme->in_real;
    VOID *out = sol->decomp_scheme->out_real;

    /* Execute R2HC Kernels */
    if (sol->solver->kernel_r2hc->count != 0)
    {
        kfft_ kernel_r2hc = sol->solver->kernel_r2hc->kfft;
        UINT8 num_sets_r2hc = sol->solver->kernel_r2hc->sets;

        // vector stride prep for R2HC kernels
        INTP v_in_stride_r2hc, v_out_stride_r2hc;
        v_in_stride_r2hc  = sol->strides_grp->strides_r2hc->v_in_stride
                            * dt_bytes * num_sets_r2hc;
        v_out_stride_r2hc = sol->strides_grp->strides_r2hc->v_out_stride
                            * dt_bytes * num_sets_r2hc;
        UINTP num_iters_r2hc = sol->solver->kernel_r2hc->count /
                               num_sets_r2hc;
        UINTP rem_iters_r2hc = sol->solver->kernel_r2hc->count -
                              (num_iters_r2hc * num_sets_r2hc);

        #pragma omp parallel for num_threads(n_threads_real)
        for (UINTP batch = 0; batch < num_iters_r2hc; batch++)
        {
            INTP v_istride = batch * v_in_stride_r2hc;
            INTP v_ostride = batch * v_out_stride_r2hc;
            kernel_r2hc(MOVE_ADDR(in, v_istride), MOVE_ADDR(in, v_istride),
                        MOVE_ADDR(out, v_ostride), MOVE_ADDR(out, v_ostride),
                        num_sets_r2hc, sol->strides_grp->strides_r2hc,
                        sol->twiddle, FFT_DIR(sol->decomp_scheme->flags));
        }
        /* Process the tail cases of the kernel */
        if(rem_iters_r2hc)
        {
            INTP v_istride = num_iters_r2hc * v_in_stride_r2hc;
            INTP v_ostride = num_iters_r2hc * v_out_stride_r2hc;
            kernel_r2hc(MOVE_ADDR(in, v_istride), MOVE_ADDR(in, v_istride),
                        MOVE_ADDR(out, v_ostride), MOVE_ADDR(out, v_ostride),
                        rem_iters_r2hc, sol->strides_grp->strides_r2hc,
                        sol->twiddle, FFT_DIR(sol->decomp_scheme->flags));
        }
    }
    /* Execute R2HCF Kernels */
    else if (sol->solver->kernel_r2hcf->count != 0)
    {
        kfft_ kernel_r2hcf = sol->solver->kernel_r2hcf->kfft;
        UINT8 num_sets_r2hcf = sol->solver->kernel_r2hcf->sets;

        // vector stride prep for R2HCF kernels
        INTP v_in_stride_r2hcf = sol->strides_grp->strides_r2hcf->v_in_stride *
                                 dt_bytes * num_sets_r2hcf;
        INTP v_out_stride_r2hcf =
            sol->strides_grp->strides_r2hcf->v_out_stride * dt_bytes *
            num_sets_r2hcf;
        UINTP num_iters_r2hcf =
            sol->solver->kernel_r2hcf->count / num_sets_r2hcf;
        UINTP rem_iters_r2hcf = sol->solver->kernel_r2hcf->count -
                                (num_iters_r2hcf * num_sets_r2hcf);

        #pragma omp parallel for num_threads(n_threads_real)
        for (UINTP batch = 0; batch < num_iters_r2hcf; batch++)
        {
            INTP v_istride = batch * v_in_stride_r2hcf;
            INTP v_ostride = batch * v_out_stride_r2hcf;
            kernel_r2hcf(MOVE_ADDR(in, v_istride), MOVE_ADDR(in, v_istride),
                         MOVE_ADDR(out, v_ostride), MOVE_ADDR(out, v_ostride),
                         num_sets_r2hcf, sol->strides_grp->strides_r2hcf,
                         sol->twiddle, FFT_DIR(sol->decomp_scheme->flags));
        }
        /* Process the tail cases of the kernel */
        if(rem_iters_r2hcf)
        {
            INTP in_offset = num_iters_r2hcf * v_in_stride_r2hcf;
            INTP out_offset = num_iters_r2hcf * v_out_stride_r2hcf;
            kernel_r2hcf(MOVE_ADDR(in, in_offset), MOVE_ADDR(in, in_offset),
                         MOVE_ADDR(out, out_offset), MOVE_ADDR(out, out_offset),
                         rem_iters_r2hcf, sol->strides_grp->strides_r2hcf,
                         sol->twiddle, FFT_DIR(sol->decomp_scheme->flags));
        }
    }
}

static inline VOID execute_real_mt_c2c_kernels(aoclfftz_solution_t *sol, INT32 n_threads_c2c)
{
    VOID *in = sol->decomp_scheme->in_real;
    VOID *out = sol->decomp_scheme->out_real;

    UINT32 dt_bytes = SOL_DT_SIZE(sol);
    UINT8 fft_dir = FFT_DIR(sol->decomp_scheme->flags);
    kfft_ kernel_c2c = sol->solver->kernel_c2c->kfft;
    UINT8 num_sets_c2c = sol->solver->kernel_c2c->sets;

    INTP radix = sol->decomp_scheme->dims[0].n;
    INTP num_groups = NUM_RFFT_GROUPS(sol->solver);
    INTP num_c2c_per_group = sol->solver->kernel_c2c->count / num_groups;
    INTP freq_factor = (sol->decomp_scheme->vecs[0].n * radix) / num_groups;

    INTP batch_in_stride = 1;
    INTP batch_out_stride = 1;
    INTP c2c_in_offset = 1;
    INTP c2c_out_offset = 1;
    if (is_input_prob_buffer(sol))
    {
        batch_in_stride = sol->decomp_scheme->dims[0].in_stride;
        c2c_in_offset = sol->decomp_scheme->dims[0].in_stride * DATA_STRIDE;
    }
    else if (is_output_prob_buffer(sol))
    {
        batch_out_stride = sol->decomp_scheme->dims[0].out_stride;
        c2c_out_offset = sol->decomp_scheme->dims[0].out_stride * DATA_STRIDE;
    }

    VOID *in_c2c = MOVE_ADDR(in, c2c_in_offset * dt_bytes);
    VOID *out_c2c = MOVE_ADDR(out, c2c_out_offset * dt_bytes);

    INTP v_in_stride_c2c =
        sol->strides_grp->strides_c2c->v_in_stride * dt_bytes * num_sets_c2c;
    INTP v_out_stride_c2c =
        sol->strides_grp->strides_c2c->v_out_stride * dt_bytes * num_sets_c2c;

    UINTP num_iters_c2c = num_groups / num_sets_c2c;
    UINTP rem_iters_c2c = num_groups - (num_iters_c2c * num_sets_c2c);

    // setting the number of threads for inner C2C kernel
    // execution
    INT32 n_threads_c2c_outer =
        n_threads_c2c > num_c2c_per_group ? num_c2c_per_group : n_threads_c2c;
    INT32 n_threads_c2c_inner = n_threads_c2c / n_threads_c2c_outer;

    INTP half_stride_start = (radix + 1) >> 1;
    INTP half_stride_n = radix - half_stride_start;
    if (fft_dir == BACKWARD_FFT_DIR)
    {

        memcpy(sol->strides_grp->strides_c2c->in_strides + half_stride_start,
               sol->strides_grp->strides->in_strides + half_stride_start,
               half_stride_n * sizeof(INTP));

        INTP batch_stride = batch_in_stride * DATA_STRIDE;
        #pragma omp parallel for num_threads(n_threads_c2c_outer)
        for (INTP group_id = 0; group_id < num_c2c_per_group; group_id++)
        {
            VOID *in_local = MOVE_ADDR(in_c2c, group_id * batch_in_stride *
                                                   DATA_STRIDE * dt_bytes);
            VOID *out_local = MOVE_ADDR(out_c2c, group_id * batch_out_stride *
                                                     DATA_STRIDE * dt_bytes);

            aoclfftz_strides_t *strides_c2c_per_thread;
            // TODO: allocation happens per loop iteration > num_threads
            ALLOC_ALIGN_UNINIT(strides_c2c_per_thread, aoclfftz_strides_t,
                               sizeof(aoclfftz_strides_t));
            memcpy(strides_c2c_per_thread, sol->strides_grp->strides_c2c,
                   sizeof(aoclfftz_strides_t));

            INTP *local_in_strides;
            ALLOC_ALIGN_UNINIT(local_in_strides, INTP, sizeof(INTP) * radix);
            memcpy(local_in_strides, sol->strides_grp->strides_c2c->in_strides,
                   sizeof(INTP) * radix);
            strides_c2c_per_thread->in_strides = local_in_strides;
            update_asymmetric_strides(local_in_strides, radix, group_id * batch_stride);

            VOID *kernel_in = NULL;
            VOID *kernel_strides = NULL;
            if (IS_OUT_OF_PLACE(sol->decomp_scheme->flags) &&
                is_input_prob_buffer(sol))
            {
                kernel_in = MOVE_ADDR(sol->dft_bufs->ct_buf_real_in,
                                      group_id * radix *
                                          (num_iters_c2c + rem_iters_c2c) *
                                          DATA_STRIDE * dt_bytes);
                kernel_strides = sol->strides_grp->strides_c2r_ct_op;
                compute_conjugates_outplace(
                    kernel_in, in_local, radix, num_groups,
                    strides_c2c_per_thread->in_strides,
                    strides_c2c_per_thread->v_in_stride,
                    DT_PRECISION_FLAG(sol->decomp_scheme->flags));
            }
            else
            {
                kernel_in = in_local;
                kernel_strides = strides_c2c_per_thread;
                compute_conjugates(
                    in_local, radix, num_groups,
                    strides_c2c_per_thread->in_strides,
                    strides_c2c_per_thread->v_in_stride,
                    DT_PRECISION_FLAG(sol->decomp_scheme->flags));
            }

            #pragma omp parallel for num_threads(n_threads_c2c_inner)
            for (INTP group_num = 0; group_num < num_iters_c2c; group_num++)
            {
                VOID *in_real_c2c =
                    MOVE_ADDR(kernel_in, group_num * v_in_stride_c2c);
                VOID *out_real_c2c =
                    MOVE_ADDR(out_local, group_num * v_out_stride_c2c);
                kernel_c2c(in_real_c2c, MOVE_ADDR(in_real_c2c, dt_bytes),
                           out_real_c2c, MOVE_ADDR(out_real_c2c, dt_bytes),
                           num_sets_c2c, kernel_strides, NULL, fft_dir);
            }
            if (rem_iters_c2c)
            {
                VOID *in_real_c2c =
                    MOVE_ADDR(kernel_in, num_iters_c2c * v_in_stride_c2c);
                VOID *out_real_c2c =
                    MOVE_ADDR(out_local, num_iters_c2c * v_out_stride_c2c);
                kernel_c2c(in_real_c2c, MOVE_ADDR(in_real_c2c, dt_bytes),
                           out_real_c2c, MOVE_ADDR(out_real_c2c, dt_bytes),
                           rem_iters_c2c, kernel_strides, NULL,
                           fft_dir);
            }
            FREE_ALIGN_ALLOCATED_MEM(local_in_strides);
            FREE_ALIGN_ALLOCATED_MEM(strides_c2c_per_thread);
        }

        assert(sol->solver->solver_type != SOLVER_REAL_MT_DIRECT_TWIDDLE);
        twiddle_multiplier_mt_for_real(sol, freq_factor, n_threads_c2c_outer,
                                       n_threads_c2c_inner);
    }
    else
    {
        // do twiddle multiplication when twiddle kernels are not used
        if (sol->solver->solver_type != SOLVER_REAL_MT_DIRECT_TWIDDLE)
        {
            twiddle_multiplier_mt_for_real(
                sol, freq_factor, n_threads_c2c_outer, n_threads_c2c_inner);
        }
        memcpy(sol->strides_grp->strides_c2c->out_strides + half_stride_start,
               sol->strides_grp->strides->out_strides + half_stride_start,
               half_stride_n * sizeof(INTP));

        INTP batch_stride = batch_out_stride * DATA_STRIDE;
        #pragma omp parallel for num_threads(n_threads_c2c_outer)
        for (INTP group_id = 0; group_id < num_c2c_per_group; group_id++)
        {
            VOID *in_local =
                MOVE_ADDR(in_c2c, group_id * batch_in_stride * DATA_STRIDE * dt_bytes);
            VOID *out_local =
                MOVE_ADDR(out_c2c, group_id * batch_out_stride * DATA_STRIDE * dt_bytes);

            aoclfftz_strides_t *strides_c2c_per_thread;
            ALLOC_ALIGN_UNINIT(strides_c2c_per_thread, aoclfftz_strides_t,
                               sizeof(aoclfftz_strides_t));
            memcpy(strides_c2c_per_thread, sol->strides_grp->strides_c2c,
                   sizeof(aoclfftz_strides_t));

            INTP *local_out_strides;
            ALLOC_ALIGN_UNINIT(local_out_strides, INTP, sizeof(INTP) * radix);
            memcpy(local_out_strides,
                   sol->strides_grp->strides_c2c->out_strides,
                   sizeof(INTP) * radix);
            strides_c2c_per_thread->out_strides = local_out_strides;
            update_asymmetric_strides(local_out_strides, radix, group_id * batch_stride);

            UINTP tw_offset = DATA_STRIDE * dt_bytes * group_id;
            aoclfftz_twiddle_t tw_local = *(sol->twiddle);
            tw_local.TW = MOVE_ADDR(tw_local.TW, tw_offset);
            // use same twiddle values across batches
            tw_local.load_multi_cols = 0;
            #pragma omp parallel for num_threads(n_threads_c2c_inner)
            for (INTP group_num = 0; group_num < num_iters_c2c; group_num++)
            {
                VOID *in_real_c2c =
                    MOVE_ADDR(in_local, group_num * v_in_stride_c2c);
                VOID *out_real_c2c =
                    MOVE_ADDR(out_local, group_num * v_out_stride_c2c);
                kernel_c2c(in_real_c2c, MOVE_ADDR(in_real_c2c, dt_bytes),
                           out_real_c2c, MOVE_ADDR(out_real_c2c, dt_bytes),
                           num_sets_c2c, strides_c2c_per_thread, &tw_local,
                           FFT_DIR(sol->decomp_scheme->flags));
            }
            if (rem_iters_c2c)
            {
                VOID *in_real_c2c =
                    MOVE_ADDR(in_local, num_iters_c2c * v_in_stride_c2c);
                VOID *out_real_c2c =
                    MOVE_ADDR(out_local, num_iters_c2c * v_out_stride_c2c);
                kernel_c2c(in_real_c2c, MOVE_ADDR(in_real_c2c, dt_bytes),
                           out_real_c2c, MOVE_ADDR(out_real_c2c, dt_bytes),
                           rem_iters_c2c, strides_c2c_per_thread, &tw_local,
                           FFT_DIR(sol->decomp_scheme->flags));
            }
            compute_conjugates(out_local, radix, num_groups,
                               strides_c2c_per_thread->out_strides,
                               strides_c2c_per_thread->v_out_stride,
                               DT_PRECISION_FLAG(sol->decomp_scheme->flags));

            FREE_ALIGN_ALLOCATED_MEM(local_out_strides);
            FREE_ALIGN_ALLOCATED_MEM(strides_c2c_per_thread);
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
 */
static INT32 execute_real_mt_direct_solver(aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");


    INT32 ret = SOLVER_SUCCESS;

    UINT8 is_direct_only_problem = IS_DIRECT_ONLY_PROBLEM(sol);
    if (is_direct_only_problem)
    {
        execute_real_mt_r2c_kernels(sol, sol->decomp_scheme->thread_info->n_threads);
        if (FFT_DIR(sol->decomp_scheme->flags) == FORWARD_FFT_DIR)
        {
            set_zero_for_dc_and_nyquist_batched(sol);
        }
        return ret;
    }

    INT32 n_threads = sol->decomp_scheme->thread_info->n_threads;

    // since r2hc perform computation on half points, their weightage in thread
    // distributions is half
    INTP batch_wt_sum = sol->solver->kernel_c2c->count +
                         (sol->solver->kernel_r2hc->count + 1) / DATA_STRIDE +
                         sol->solver->kernel_r2hcf->count;
    INT32 n_threads_c2c = (n_threads * sol->solver->kernel_c2c->count) / batch_wt_sum;
    INT32 n_threads_real = n_threads - n_threads_c2c;

    /*
     * Parallelization Strategy:
     * Parallelize the execution blocks of both real and complex kernels only
     * when count of R2HC/R2HCF > 0 and count of C2C > 0.
     */
    int enable_parallelism = n_threads_c2c > 0;
    if (enable_parallelism)
    {
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                execute_real_mt_r2c_kernels(sol, n_threads_real);
            }
            #pragma omp section
            {
                execute_real_mt_c2c_kernels(sol, n_threads_c2c);
            }
        }
    }
    else
    {
        execute_real_mt_r2c_kernels(sol, n_threads_real);
    }

    if (HAS_NEXT(sol))
    {
        ret = sol->next_sol[0]->solver->execute_solver(sol->next_sol[0]);
    }
    else if (FFT_DIR(sol->decomp_scheme->flags) == FORWARD_FFT_DIR)
    {
        set_zero_for_dc_and_nyquist(sol);
    }

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return ret;
}

dft_solver_ register_execute_real_mt_direct_solver(VOID)
{
    return execute_real_mt_direct_solver;
}
