// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

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
FFTZ_INT32 setup_real_mt_direct_solver(aoclfftz_solution_t *sol,
                                  cost_analysis_t *cost,
                                  const kernel_t *kernel_c2c,
                                  const kernel_t *kernel_r2hc,
                                  const kernel_t *kernel_r2hcf,
                                  aoclfftz_realhelper_t *realhelper)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");
    FFTZ_INT32 status = SOLVER_SUCCESS;

    FFTZ_INT32 avl_threads = sol->decomp_scheme->thread_info->avl_threads;

    set_kernel_count_in_each_group(sol, realhelper);

    // Setting the number of threads based on the batches
    FFTZ_UINTP c2c_batches = sol->solver->kernel_c2c->count;
    FFTZ_UINTP r2hc_batches = sol->solver->kernel_r2hc->count;
    FFTZ_UINTP r2hcf_batches = sol->solver->kernel_r2hcf->count;
    if (c2c_batches)
    {
        r2hc_batches = (r2hc_batches + 1) / DATA_STRIDE;
    }
    FFTZ_INTP total_batches = r2hc_batches + r2hcf_batches + c2c_batches;
    sol->decomp_scheme->thread_info->n_threads = (avl_threads < total_batches)
                                               ? avl_threads : total_batches;

    allocate_and_setup_stride(sol, *realhelper);

    update_ct_buffers(sol, realhelper);

    compute_cost(sol, cost, kernel_c2c, kernel_r2hc, kernel_r2hcf);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return status;
}

static inline FFTZ_VOID execute_real_mt_r2c_kernels(aoclfftz_solution_t *sol,
                                                    FFTZ_INT32 n_threads_real)
{
    FFTZ_UINT32 dt_bytes = SOL_DT_SIZE(sol);

    FFTZ_VOID *in = sol->decomp_scheme->in_real;
    FFTZ_VOID *out = sol->decomp_scheme->out_real;

    /* Execute R2HC Kernels */
    if (sol->solver->kernel_r2hc->count != 0)
    {
        // R2HC kernels are bidirectional
        // so both kfft[FORWARD_FFT_DIR] and kfft[BACKWARD_FFT_DIR] point to the
        // same kernel
        kfft_ kernel_r2hc = sol->solver->kernel_r2hc->kfft[FORWARD_FFT_DIR];
        FFTZ_UINT8 num_sets_r2hc = sol->solver->kernel_r2hc->sets;

        // vector stride prep for R2HC kernels
        FFTZ_INTP v_in_stride_r2hc, v_out_stride_r2hc;
        v_in_stride_r2hc  = sol->strides_grp->strides_r2hc->v_in_stride
                            * dt_bytes * num_sets_r2hc;
        v_out_stride_r2hc = sol->strides_grp->strides_r2hc->v_out_stride
                            * dt_bytes * num_sets_r2hc;
        FFTZ_UINTP num_iters_r2hc = sol->solver->kernel_r2hc->count /
                               num_sets_r2hc;
        FFTZ_UINTP rem_iters_r2hc = sol->solver->kernel_r2hc->count -
                              (num_iters_r2hc * num_sets_r2hc);

        #pragma omp parallel for num_threads(n_threads_real)
        for (FFTZ_UINTP batch = 0; batch < num_iters_r2hc; batch++)
        {
            FFTZ_INTP v_istride = batch * v_in_stride_r2hc;
            FFTZ_INTP v_ostride = batch * v_out_stride_r2hc;
            kernel_r2hc(MOVE_ADDR(in, v_istride), MOVE_ADDR(in, v_istride),
                        MOVE_ADDR(out, v_ostride), MOVE_ADDR(out, v_ostride),
                        num_sets_r2hc, sol->strides_grp->strides_r2hc,
                        sol->twiddle, FFT_DIR(sol->decomp_scheme->flags));
        }
        /* Process the tail cases of the kernel */
        if (rem_iters_r2hc)
        {
            FFTZ_INTP v_istride = num_iters_r2hc * v_in_stride_r2hc;
            FFTZ_INTP v_ostride = num_iters_r2hc * v_out_stride_r2hc;
            kernel_r2hc(MOVE_ADDR(in, v_istride), MOVE_ADDR(in, v_istride),
                        MOVE_ADDR(out, v_ostride), MOVE_ADDR(out, v_ostride),
                        rem_iters_r2hc, sol->strides_grp->strides_r2hc,
                        sol->twiddle, FFT_DIR(sol->decomp_scheme->flags));
        }
    }
    /* Execute R2HCF Kernels */
    else if (sol->solver->kernel_r2hcf->count != 0)
    {
        // R2HCF kernels are bidirectional
        // so both kfft[FORWARD_FFT_DIR] and kfft[BACKWARD_FFT_DIR] point to the
        // same kernel
        kfft_ kernel_r2hcf = sol->solver->kernel_r2hcf->kfft[FORWARD_FFT_DIR];
        FFTZ_UINT8 num_sets_r2hcf = sol->solver->kernel_r2hcf->sets;

        // vector stride prep for R2HCF kernels
        FFTZ_INTP v_in_stride_r2hcf =
            sol->strides_grp->strides_r2hcf->v_in_stride * dt_bytes *
            num_sets_r2hcf;
        FFTZ_INTP v_out_stride_r2hcf =
            sol->strides_grp->strides_r2hcf->v_out_stride * dt_bytes *
            num_sets_r2hcf;
        FFTZ_UINTP num_iters_r2hcf =
            sol->solver->kernel_r2hcf->count / num_sets_r2hcf;
        FFTZ_UINTP rem_iters_r2hcf = sol->solver->kernel_r2hcf->count -
                                (num_iters_r2hcf * num_sets_r2hcf);

        #pragma omp parallel for num_threads(n_threads_real)
        for (FFTZ_UINTP batch = 0; batch < num_iters_r2hcf; batch++)
        {
            FFTZ_INTP v_istride = batch * v_in_stride_r2hcf;
            FFTZ_INTP v_ostride = batch * v_out_stride_r2hcf;
            kernel_r2hcf(MOVE_ADDR(in, v_istride), MOVE_ADDR(in, v_istride),
                         MOVE_ADDR(out, v_ostride), MOVE_ADDR(out, v_ostride),
                         num_sets_r2hcf, sol->strides_grp->strides_r2hcf,
                         sol->twiddle, FFT_DIR(sol->decomp_scheme->flags));
        }
        /* Process the tail cases of the kernel */
        if (rem_iters_r2hcf)
        {
            FFTZ_INTP in_offset = num_iters_r2hcf * v_in_stride_r2hcf;
            FFTZ_INTP out_offset = num_iters_r2hcf * v_out_stride_r2hcf;
            kernel_r2hcf(MOVE_ADDR(in, in_offset), MOVE_ADDR(in, in_offset),
                         MOVE_ADDR(out, out_offset), MOVE_ADDR(out, out_offset),
                         rem_iters_r2hcf, sol->strides_grp->strides_r2hcf,
                         sol->twiddle, FFT_DIR(sol->decomp_scheme->flags));
        }
    }
}

static inline FFTZ_VOID execute_real_mt_c2c_kernels(aoclfftz_solution_t *sol,
                                                    FFTZ_INT32 n_threads_c2c)
{
    FFTZ_VOID *in = sol->decomp_scheme->in_real;
    FFTZ_VOID *out = sol->decomp_scheme->out_real;

    FFTZ_UINT32 dt_bytes = SOL_DT_SIZE(sol);
    FFTZ_UINT8 direction = FFT_DIR(sol->decomp_scheme->flags);
    kfft_ kernel_c2c = sol->solver->kernel_c2c->kfft[direction];
    FFTZ_UINT8 num_sets_c2c = sol->solver->kernel_c2c->sets;

    FFTZ_INTP radix = sol->decomp_scheme->dims[0].n;
    FFTZ_INTP num_groups = NUM_RFFT_GROUPS(sol->solver);
    FFTZ_INTP num_c2c_per_group = sol->solver->kernel_c2c->count / num_groups;
    FFTZ_UINT8 use_asymmetric_kernel = (num_c2c_per_group >= num_groups);

    FFTZ_INTP batch_in_stride = 1;
    FFTZ_INTP batch_out_stride = 1;
    FFTZ_INTP c2c_in_offset = 1;
    FFTZ_INTP c2c_out_offset = 1;
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

    FFTZ_VOID *in_c2c = MOVE_ADDR(in, c2c_in_offset * dt_bytes);
    FFTZ_VOID *out_c2c = MOVE_ADDR(out, c2c_out_offset * dt_bytes);

    FFTZ_INTP v_in_stride_c2c =
        sol->strides_grp->strides_c2c->v_in_stride * dt_bytes * num_sets_c2c;
    FFTZ_INTP v_out_stride_c2c =
        sol->strides_grp->strides_c2c->v_out_stride * dt_bytes * num_sets_c2c;

    FFTZ_UINTP num_iters_c2c = num_groups / num_sets_c2c;
    FFTZ_UINTP rem_iters_c2c = num_groups - (num_iters_c2c * num_sets_c2c);

    // setting the number of threads for inner C2C kernel
    // execution
    FFTZ_INT32 n_threads_c2c_outer = 1;
    FFTZ_INT32 n_threads_c2c_inner = 1;
    if (!use_asymmetric_kernel)
    {
        // Symmetric case: parallelize over num_c2c_per_group
        n_threads_c2c_outer = n_threads_c2c > num_c2c_per_group
                                  ? num_c2c_per_group
                                  : n_threads_c2c;
        n_threads_c2c_inner = n_threads_c2c / n_threads_c2c_outer;
    }
    else
    {
        // Asymmetric case: parallelize over num_groups
        n_threads_c2c_outer = n_threads_c2c > num_groups
                                  ? num_groups
                                  : n_threads_c2c;
        n_threads_c2c_inner = 1; // No inner parallelization for asymmetric
    }

    FFTZ_INTP half_stride_start = (radix + 1) >> 1;
    FFTZ_INTP half_stride_n = radix - half_stride_start;
    if (direction == BACKWARD_FFT_DIR)
    {
        memcpy(sol->strides_grp->strides_c2c->in_strides + half_stride_start,
               sol->strides_grp->strides->in_strides + half_stride_start,
               half_stride_n * sizeof(FFTZ_INTP));

        if (!use_asymmetric_kernel)
        {
            FFTZ_INTP batch_stride = batch_in_stride * DATA_STRIDE;
            #pragma omp parallel for num_threads(n_threads_c2c_outer)
            for (FFTZ_INTP group_id = 0; group_id < num_c2c_per_group;
                 group_id++)
            {
                FFTZ_VOID *in_local =
                    MOVE_ADDR(in_c2c, group_id * batch_in_stride * DATA_STRIDE *
                                          dt_bytes);
                FFTZ_VOID *out_local =
                    MOVE_ADDR(out_c2c, group_id * batch_out_stride *
                                           DATA_STRIDE * dt_bytes);

                aoclfftz_strides_t *strides_c2c_per_thread;
                // TODO: allocation happens per loop iteration > num_threads
                // TODO: pre-allocate a single strides buffer for all threads outside
                // the loop and give each thread its own chunk, so the allocation and
                // its failure check happen once instead of per iteration
                ALLOC_ALIGN_UNINIT(strides_c2c_per_thread, aoclfftz_strides_t,
                                   sizeof(aoclfftz_strides_t));
                memcpy(strides_c2c_per_thread, sol->strides_grp->strides_c2c,
                       sizeof(aoclfftz_strides_t));

                FFTZ_INTP *local_in_strides;
                ALLOC_ALIGN_UNINIT(local_in_strides, FFTZ_INTP,
                                   sizeof(FFTZ_INTP) * radix);
                memcpy(local_in_strides,
                       sol->strides_grp->strides_c2c->in_strides,
                       sizeof(FFTZ_INTP) * radix);
                strides_c2c_per_thread->in_strides = local_in_strides;
                update_asymmetric_strides(local_in_strides, radix,
                                          group_id * batch_stride);

                // Slide TW per group so distinct C2C-points read distinct
                // twiddle columns; load_multi_cols=0 mirrors the ST path so
                // each butterfly reuses the same column.
                aoclfftz_twiddle_t tw_local = *(sol->twiddle);
                FFTZ_UINTP tw_offset =
                    (FFTZ_UINTP)(radix - 1) * DATA_STRIDE * dt_bytes * group_id;
                tw_local.TW = MOVE_ADDR(tw_local.TW, tw_offset);
                tw_local.load_multi_cols = 0;

                for (FFTZ_INTP group_num = 0; group_num < num_iters_c2c;
                     group_num++)
                {
                    FFTZ_VOID *in_real_c2c =
                        MOVE_ADDR(in_local, group_num * v_in_stride_c2c);
                    FFTZ_VOID *out_real_c2c =
                        MOVE_ADDR(out_local, group_num * v_out_stride_c2c);
                    kernel_c2c(in_real_c2c, MOVE_ADDR(in_real_c2c, dt_bytes),
                               out_real_c2c, MOVE_ADDR(out_real_c2c, dt_bytes),
                               num_sets_c2c, strides_c2c_per_thread, &tw_local,
                               direction);
                }

                if (rem_iters_c2c)
                {
                    FFTZ_VOID *in_real_c2c =
                        MOVE_ADDR(in_local, num_iters_c2c * v_in_stride_c2c);
                    FFTZ_VOID *out_real_c2c =
                        MOVE_ADDR(out_local, num_iters_c2c * v_out_stride_c2c);
                    kernel_c2c(in_real_c2c, MOVE_ADDR(in_real_c2c, dt_bytes),
                               out_real_c2c, MOVE_ADDR(out_real_c2c, dt_bytes),
                               rem_iters_c2c, strides_c2c_per_thread, &tw_local,
                               FFT_DIR(sol->decomp_scheme->flags));
                }
                FREE_ALIGN_ALLOCATED_MEM(local_in_strides);
                FREE_ALIGN_ALLOCATED_MEM(strides_c2c_per_thread);
            }
        }
        else
        {
            FFTZ_INTP v_in_stride  = sol->strides_grp->strides->v_in_stride;
            FFTZ_INTP v_out_stride = sol->strides_grp->strides->v_out_stride;

            #pragma omp parallel for num_threads(n_threads_c2c_outer)
            for (FFTZ_INTP group_id = 0; group_id < num_groups; group_id++)
            {
                FFTZ_VOID *in_local  = MOVE_ADDR(in_c2c,
                                 group_id * v_in_stride * dt_bytes);
                FFTZ_VOID *out_local = MOVE_ADDR(out_c2c,
                                  group_id * v_out_stride * dt_bytes);

                aoclfftz_strides_t *strides_c2c_per_thread;
                ALLOC_ALIGN_UNINIT(strides_c2c_per_thread, aoclfftz_strides_t,
                                   sizeof(aoclfftz_strides_t));
                memcpy(strides_c2c_per_thread, sol->strides_grp->strides_c2c,
                       sizeof(aoclfftz_strides_t));

                // Allocate and copy a private in_strides array
                // for this thread
                FFTZ_INTP *local_in_strides;
                ALLOC_ALIGN_UNINIT(local_in_strides, FFTZ_INTP,
                                   sizeof(FFTZ_INTP) * radix);

                memcpy(local_in_strides,
                       sol->strides_grp->strides_c2c->in_strides,
                       sizeof(FFTZ_INTP) * radix);
                strides_c2c_per_thread->in_strides = local_in_strides;

                // Kernel execution - using num_c2c_per_group for
                // asymmetric case swapping real & imag points
                // for backward kernel
                kernel_c2c(in_local, MOVE_ADDR(in_local, dt_bytes), out_local,
                           MOVE_ADDR(out_local, dt_bytes), num_c2c_per_group,
                           strides_c2c_per_thread, sol->twiddle,
                           FFT_DIR(sol->decomp_scheme->flags));

                FREE_ALIGN_ALLOCATED_MEM(local_in_strides);
                FREE_ALIGN_ALLOCATED_MEM(strides_c2c_per_thread);
            }
        }
    }
    else
    {
        memcpy(sol->strides_grp->strides_c2c->out_strides + half_stride_start,
               sol->strides_grp->strides->out_strides + half_stride_start,
               half_stride_n * sizeof(FFTZ_INTP));

        if (!use_asymmetric_kernel)
        {
            FFTZ_INTP batch_stride = batch_out_stride * DATA_STRIDE;
            #pragma omp parallel for num_threads(n_threads_c2c_outer)
            for (FFTZ_INTP group_id = 0; group_id < num_c2c_per_group;
                 group_id++)
            {
                // This for loop computes C2C batches within the groups,
                // while the kernel does across multiple groups
                FFTZ_VOID *in_local =
                    MOVE_ADDR(in_c2c, group_id * batch_in_stride * DATA_STRIDE *
                                          dt_bytes);
                FFTZ_VOID *out_local =
                    MOVE_ADDR(out_c2c, group_id * batch_out_stride *
                                           DATA_STRIDE * dt_bytes);

                aoclfftz_strides_t *strides_c2c_per_thread;
                ALLOC_ALIGN_UNINIT(strides_c2c_per_thread, aoclfftz_strides_t,
                                   sizeof(aoclfftz_strides_t));
                memcpy(strides_c2c_per_thread, sol->strides_grp->strides_c2c,
                       sizeof(aoclfftz_strides_t));

                FFTZ_INTP *local_out_strides;
                ALLOC_ALIGN_UNINIT(local_out_strides, FFTZ_INTP,
                                   sizeof(FFTZ_INTP) * radix);
                memcpy(local_out_strides,
                       sol->strides_grp->strides_c2c->out_strides,
                       sizeof(FFTZ_INTP) * radix);
                strides_c2c_per_thread->out_strides = local_out_strides;
                update_asymmetric_strides(local_out_strides, radix,
                                          group_id * batch_stride);

                FFTZ_UINTP tw_offset =
                    (FFTZ_UINTP)(radix - 1) * DATA_STRIDE * dt_bytes * group_id;
                aoclfftz_twiddle_t tw_local = *(sol->twiddle);
                tw_local.TW = MOVE_ADDR(tw_local.TW, tw_offset);
                tw_local.load_multi_cols = 0;

                #pragma omp parallel for num_threads(n_threads_c2c_inner)
                for (FFTZ_INTP group_num = 0; group_num < num_iters_c2c;
                     group_num++)
                {
                    FFTZ_VOID *in_real_c2c =
                        MOVE_ADDR(in_local, group_num * v_in_stride_c2c);
                    FFTZ_VOID *out_real_c2c =
                        MOVE_ADDR(out_local, group_num * v_out_stride_c2c);
                    kernel_c2c(in_real_c2c, MOVE_ADDR(in_real_c2c, dt_bytes),
                               out_real_c2c, MOVE_ADDR(out_real_c2c, dt_bytes),
                               num_sets_c2c, strides_c2c_per_thread, &tw_local,
                               FFT_DIR(sol->decomp_scheme->flags));
                }
                if (rem_iters_c2c)
                {
                    FFTZ_VOID *in_real_c2c =
                        MOVE_ADDR(in_local, num_iters_c2c * v_in_stride_c2c);
                    FFTZ_VOID *out_real_c2c =
                        MOVE_ADDR(out_local, num_iters_c2c * v_out_stride_c2c);
                    kernel_c2c(in_real_c2c, MOVE_ADDR(in_real_c2c, dt_bytes),
                               out_real_c2c, MOVE_ADDR(out_real_c2c, dt_bytes),
                               rem_iters_c2c, strides_c2c_per_thread, &tw_local,
                               FFT_DIR(sol->decomp_scheme->flags));
                }

                FREE_ALIGN_ALLOCATED_MEM(local_out_strides);
                FREE_ALIGN_ALLOCATED_MEM(strides_c2c_per_thread);
            }
        }
        else
        {
            // Asymmetric stride execution path for forward
            FFTZ_INTP v_in_stride  = sol->strides_grp->strides->v_in_stride;
            FFTZ_INTP v_out_stride = sol->strides_grp->strides->v_out_stride;

            #pragma omp parallel for num_threads(n_threads_c2c_outer)
            for (FFTZ_INTP group_id = 0; group_id < num_groups; group_id++)
            {
                FFTZ_VOID *in_local  = MOVE_ADDR(in_c2c,
                                  group_id * v_in_stride * dt_bytes);
                FFTZ_VOID *out_local = MOVE_ADDR(out_c2c,
                                  group_id * v_out_stride * dt_bytes);
                aoclfftz_strides_t *strides_c2c_per_thread;
                ALLOC_ALIGN_UNINIT(strides_c2c_per_thread,
                                    aoclfftz_strides_t,
                                    sizeof(aoclfftz_strides_t));
                memcpy(strides_c2c_per_thread, sol->strides_grp->strides_c2c,
                       sizeof(aoclfftz_strides_t));

                // Allocate and copy a private out_strides array
                // for this thread
                FFTZ_INTP *local_out_strides;
                ALLOC_ALIGN_UNINIT(local_out_strides, FFTZ_INTP,
                                   sizeof(FFTZ_INTP) * radix);
                memcpy(local_out_strides,
                       sol->strides_grp->strides_c2c->out_strides,
                       sizeof(FFTZ_INTP) * radix);
                strides_c2c_per_thread->out_strides = local_out_strides;

                // Kernel execution - using group_size for asymmetric case
                kernel_c2c(in_local, MOVE_ADDR(in_local, dt_bytes),
                           out_local, MOVE_ADDR(out_local, dt_bytes),
                           num_c2c_per_group, strides_c2c_per_thread,
                           sol->twiddle, FFT_DIR(sol->decomp_scheme->flags));

                FREE_ALIGN_ALLOCATED_MEM(local_out_strides);
                FREE_ALIGN_ALLOCATED_MEM(strides_c2c_per_thread);
            }
        }
    }
}

/* This function will execute the kernels for both real direct and CT problems.

   For real direct problem, it will execute R2HC kernels.
   For real CT problems, following steps will be performed:
     1. Call R2HC/R2HCF kernels
     2. Get the no. of groups and group size for C2C kernels
     3. Update C2C kernel strides for each kernel within a group
     4. Execute the fused C2C kernels (twiddle multiplication and the
        half-point conjugation are handled internally by the kernel).
*/
static FFTZ_INT32 execute_real_mt_direct_solver(aoclfftz_solution_t *sol,
                                                aoclfftz_mutable_ctx_t *ctx)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    FFTZ_INT32 ret = SOLVER_SUCCESS;

    FFTZ_UINT8 is_direct_only_problem = IS_DIRECT_ONLY_PROBLEM(sol);
    if (is_direct_only_problem)
    {
        execute_real_mt_r2c_kernels(sol,
                                    sol->decomp_scheme->thread_info->n_threads);
        if (FFT_DIR(sol->decomp_scheme->flags) == FORWARD_FFT_DIR)
        {
            set_zero_for_dc_and_nyquist_batched(sol);
        }
        return ret;
    }

    FFTZ_INT32 n_threads = sol->decomp_scheme->thread_info->n_threads;

    // since r2hc perform computation on half points, their weightage in thread
    // distributions is half
    FFTZ_INTP batch_wt_sum = sol->solver->kernel_c2c->count +
                         (sol->solver->kernel_r2hc->count + 1) / DATA_STRIDE +
                         sol->solver->kernel_r2hcf->count;
    FFTZ_INT32 n_threads_c2c =
        (n_threads * sol->solver->kernel_c2c->count) / batch_wt_sum;
    FFTZ_INT32 n_threads_real = n_threads - n_threads_c2c;

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

#if REAL_FFT_EXECUTION_ORDER == REAL_FFT_ORDER_TRUE_RECURSION
    // Recurse-then-combine mode: the CT solver owns tree traversal, so the
    // Direct node behaves as a pure leaf and never chains to next_sol. Only
    // the terminal leaf emits DC/Nyquist zeroing.
    if (!HAS_NEXT(sol) &&
        FFT_DIR(sol->decomp_scheme->flags) == FORWARD_FFT_DIR)
    {
        set_zero_for_dc_and_nyquist(sol);
    }
#else
    if (HAS_NEXT(sol))
    {
        ret = sol->next_sol[0]->solver->execute_solver(sol->next_sol[0], ctx);
    }
    else if (FFT_DIR(sol->decomp_scheme->flags) == FORWARD_FFT_DIR)
    {
        set_zero_for_dc_and_nyquist(sol);
    }
#endif

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return ret;
}

dft_solver_ register_execute_real_mt_direct_solver(FFTZ_VOID)
{
    return execute_real_mt_direct_solver;
}
