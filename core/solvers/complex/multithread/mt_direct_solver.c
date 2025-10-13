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
 *  @brief Multi threaded direct solver that enables multi threading for the
 *  available direct kernels
 *
 *  This file contains the functions that setup, execute and destroy
 *  the solver.
 *
 *  @author Murugan Vairavel
 *  @author Srirammaswamy Srinivasan
 */

#include "api/aoclfftz_internal.h"
#include "core/common/memory_manager.h"
#include "utils/utils.h"

INT32 setup_mt_direct_solver(aoclfftz_solution_t *sol, cost_analysis_t *cost,
                          kernel_t *kernel)
{
    aoclfftz_decomp_scheme_t *decomp_scheme = sol->decomp_scheme;
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");


    aoclfftz_strides_t *strides = sol->strides_grp->strides;
    ops_cycles_t ops_cycles;
    INTP n = decomp_scheme->vecs[0].n;
    INTP radix = decomp_scheme->dims[0].n;
    UINT8 precision = DT_PRECISION_FLAG(decomp_scheme->flags);
    UINT8 direction = FFT_DIR(decomp_scheme->flags);
    INT32 status = SOLVER_SUCCESS;
    UINT8 num_sets = kernel->sets[precision - 2];

    if (strides->in_strides == NULL)
    {
        ALLOC_ALIGN_UNINIT(strides->in_strides, INTP, radix * sizeof(INTP));
        ALLOC_ALIGN_UNINIT(strides->out_strides, INTP, radix * sizeof(INTP));
        INTP in_stride = decomp_scheme->dims[0].in_stride;
        INTP out_stride = decomp_scheme->dims[0].out_stride;
        for (INTP i = 0; i < radix; i++)
        {
            strides->in_strides[i] = i * in_stride * DATA_STRIDE;
            strides->out_strides[i] = i * out_stride * DATA_STRIDE;
        }
    }

    strides->v_in_stride = decomp_scheme->vecs[0].in_stride * DATA_STRIDE;
    strides->v_out_stride = decomp_scheme->vecs[0].out_stride * DATA_STRIDE;

    if (GET_SELECTOR_MODE(decomp_scheme->flags) == AOCLFFTZ_FIXED_SELECTOR)
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
        if (n >= num_sets)
        {
            cost->ops = (cost->ops + num_sets - 1) / num_sets; // ceil div
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
        INTP v_in_stride, v_out_stride, data_offset;
        UINT32 dt_bytes = SOL_DT_SIZE(sol);
        data_offset = DATA_STRIDE * dt_bytes * num_sets;
        v_in_stride = decomp_scheme->vecs[0].in_stride * data_offset;
        v_out_stride = decomp_scheme->vecs[0].out_stride * data_offset;

        INTP num_iters = decomp_scheme->vecs[0].n / num_sets;
        INTP rem_iters = decomp_scheme->vecs[0].n - (num_iters * num_sets);

        // Set threads for parallel execution
        omp_set_num_threads(decomp_scheme->thread_info->n_threads);
        #pragma omp parallel for
        for (INTP batch = 0; batch < num_iters; batch++)
        {
            INTP v_istride = batch * v_in_stride;
            INTP v_ostride = batch * v_out_stride;
            kernel->kfft(MOVE_ADDR(decomp_scheme->in_real, v_istride),
                         MOVE_ADDR(decomp_scheme->in_imag, v_istride),
                         MOVE_ADDR(decomp_scheme->out_real, v_ostride),
                         MOVE_ADDR(decomp_scheme->out_imag, v_ostride),
                         num_sets, strides, sol->twiddle->TW, FFT_DIR(decomp_scheme->flags));
        }

        // Process the tail cases of the kernel
        if(rem_iters)
        {
            INTP v_istride = num_iters * v_in_stride;
            INTP v_ostride = num_iters * v_out_stride;
            kernel->kfft(MOVE_ADDR(decomp_scheme->in_real, v_istride),
                         MOVE_ADDR(decomp_scheme->in_imag, v_istride),
                         MOVE_ADDR(decomp_scheme->out_real, v_ostride),
                         MOVE_ADDR(decomp_scheme->out_imag, v_ostride),
                         rem_iters, strides, sol->twiddle->TW, FFT_DIR(decomp_scheme->flags));
        }

        getTime(endTime);
        cost->time = diffTime(clkTick, startTime, endTime);
        ops_cycles = kernel->k_ops_cnt(precision, direction);
        cost->ops = ((ops_cycles.fma * AMD_ZEN_FP_FMA_CYCLES) +
                     (ops_cycles.mul * AMD_ZEN_FP_MUL_CYCLES) +
                     (ops_cycles.add * AMD_ZEN_FP_ADD_CYCLES) +
                     (ops_cycles.move * AMD_ZEN_FP_MOVE_CYCLES) +
                     (ops_cycles.perm * AMD_ZEN_FP_PERM_CYCLES) +
                     (ops_cycles.other * AMD_ZEN_FP_OTHER_CYCLES));
        if (n >= num_sets)
        {
            cost->ops = (cost->ops + num_sets - 1) / num_sets; // ceil div
        }
        cost->ops = cost->ops * n;
    }

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return status;
}

static INT32 execute_mt_direct_solver(aoclfftz_solution_t *sol)
{
    aoclfftz_decomp_scheme_t *decomp_scheme = sol->decomp_scheme;
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");


    kfft_ kernel = sol->solver->kernel_c2c->kfft;
    UINT8 num_sets = sol->solver->kernel_c2c->sets;
    aoclfftz_strides_t *strides = sol->strides_grp->strides;

    INTP v_in_stride, v_out_stride, data_offset;
    UINT32 dt_bytes = SOL_DT_SIZE(sol);
    data_offset = DATA_STRIDE * dt_bytes * num_sets;
    v_in_stride = decomp_scheme->vecs[0].in_stride * data_offset;
    v_out_stride = decomp_scheme->vecs[0].out_stride * data_offset;

    INTP num_iters = decomp_scheme->vecs[0].n / num_sets;
    INTP rem_iters = decomp_scheme->vecs[0].n - (num_iters * num_sets);

    // Set threads for parallel execution
    omp_set_num_threads(decomp_scheme->thread_info->n_threads);
    #pragma omp parallel for
    for (INTP batch = 0; batch < num_iters; batch++)
    {
        aoclfftz_twiddle_t tw_local = {
            .TW = MOVE_ADDR(sol->twiddle->TW,
                            DATA_STRIDE * dt_bytes * batch * num_sets),
            .twiddle_buf_ptr = sol->twiddle->twiddle_buf_ptr,
            .cols = sol->twiddle->cols,
            .load_multi_cols = 1, // use different twiddle values across batches
        };

        INTP v_istride = batch * v_in_stride;
        INTP v_ostride = batch * v_out_stride;
        kernel(MOVE_ADDR(decomp_scheme->in_real, v_istride),
               MOVE_ADDR(decomp_scheme->in_imag, v_istride),
               MOVE_ADDR(decomp_scheme->out_real, v_ostride),
               MOVE_ADDR(decomp_scheme->out_imag, v_ostride),
               num_sets, strides, &tw_local, FFT_DIR(decomp_scheme->flags));
    }

    // Process the tail cases of the kernel
    aoclfftz_twiddle_t tw_local = {
        .TW = MOVE_ADDR(sol->twiddle->TW,
                        DATA_STRIDE * dt_bytes * num_iters * num_sets),
        .twiddle_buf_ptr = sol->twiddle->twiddle_buf_ptr,
        .cols = sol->twiddle->cols,
        .load_multi_cols = 1, // use different twiddle values across batches
    };
    if (rem_iters)
    {
        INTP v_istride = num_iters * v_in_stride;
        INTP v_ostride = num_iters * v_out_stride;
        kernel(MOVE_ADDR(decomp_scheme->in_real, v_istride),
               MOVE_ADDR(decomp_scheme->in_imag, v_istride),
               MOVE_ADDR(decomp_scheme->out_real, v_ostride),
               MOVE_ADDR(decomp_scheme->out_imag, v_ostride),
               rem_iters, strides, &tw_local, FFT_DIR(decomp_scheme->flags));
    }

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return SOLVER_SUCCESS;
}

/**
 * Parallelizes over problem batches.
 *
 * Example:
 *   problem size: 100:1:1v48:100:100 (column-major CT problem)
 *   num_threads: 48
 *   batches: 100
 *   problem size: 48 (4 x 12) -> [12v4 (12 butterflies of radix-4), 4v12 (4 butterflies of radix-12)]
 *
 *   for radix-4 kernel:
 *      #omp parallel for
 *      for (1..100) // batches
 *          for (1..12) // butterflies
 *              kernel(radix-4)
 *
 *   for radix-12 kernel:
 *      #omp parallel for
 *      for (1..100) // batches
 *          for (1..4) // butterflies
 *              kernel(radix-12)
 */
static INT32 execute_mt_direct_batched_rowmajor_solver(aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");


    kfft_ kernel = sol->solver->kernel_c2c->kfft;
    aoclfftz_strides_t *strides = sol->strides_grp->strides;
    UINT8 direction = FFT_DIR(sol->decomp_scheme->flags);

    VOID *in_real = sol->decomp_scheme->in_real;
    VOID *in_imag = sol->decomp_scheme->in_imag;
    VOID *out_real = sol->decomp_scheme->out_real;
    VOID *out_imag = sol->decomp_scheme->out_imag;

    UINT32 dt_bytes = SOL_DT_SIZE(sol);

    // vec-strides across DFT butterflies of the same CT problem
    INTP ct_in_stride =
        sol->decomp_scheme->vecs[0].in_stride * DATA_STRIDE * dt_bytes;
    INTP ct_out_stride =
        sol->decomp_scheme->vecs[0].out_stride * DATA_STRIDE * dt_bytes;

    UINT8 num_sets = sol->solver->kernel_c2c->sets;
    INTP num_iters = sol->decomp_scheme->batched_vecs[0].n / num_sets;
    INTP rem_iters =
        sol->decomp_scheme->batched_vecs[0].n - (num_iters * num_sets);

    // Set threads for parallel execution
    omp_set_num_threads(sol->decomp_scheme->thread_info->n_threads);
    // **NOTE**:
    // This `direct_batched` solver now parallelizes over batches instead of
    // DFT butterflies. This provides better scalability for large batched
    // problems with more number of threads.
    #pragma omp parallel for
    for (INTP batch = 0; batch < num_iters; batch++)
    {
        // Calculate batch offsets for input/output arrays
        INTP batch_in_offset =
            batch * strides->v_in_stride * dt_bytes * num_sets;
        INTP batch_out_offset =
            batch * strides->v_out_stride * dt_bytes * num_sets;

        // Sequential loop over DFT butterflies
        for (INTP i = 0; i < sol->decomp_scheme->vecs[0].n; i++)
        {
            aoclfftz_twiddle_t tw_local = {
                .TW = MOVE_ADDR(sol->twiddle->TW, i * DATA_STRIDE * dt_bytes),
                .cols = sol->twiddle->cols,
                .twiddle_buf_ptr = sol->twiddle->twiddle_buf_ptr,
                .load_multi_cols = 0, // use same twiddle values across batches
            };                        // since different batches solves the same
                                      // DFT butterfly for different problems
            kernel(MOVE_ADDR(in_real, batch_in_offset + i * ct_in_stride),
                   MOVE_ADDR(in_imag, batch_in_offset + i * ct_in_stride),
                   MOVE_ADDR(out_real, batch_out_offset + i * ct_out_stride),
                   MOVE_ADDR(out_imag, batch_out_offset + i * ct_out_stride),
                   num_sets, strides, &tw_local, direction);
        }
    }

    if (rem_iters)
    {
        INTP batch_in_offset =
            num_iters * strides->v_in_stride * dt_bytes * num_sets;
        INTP batch_out_offset =
            num_iters * strides->v_out_stride * dt_bytes * num_sets;

        // Sequential loop over DFT butterflies
        for (INTP i = 0; i < sol->decomp_scheme->vecs[0].n; i++)
        {
            aoclfftz_twiddle_t tw_local = {
                .TW = MOVE_ADDR(sol->twiddle->TW, i * DATA_STRIDE * dt_bytes),
                .cols = sol->twiddle->cols,
                .twiddle_buf_ptr = sol->twiddle->twiddle_buf_ptr,
                .load_multi_cols = 0, // use same twiddle values across batches
            };                        // since different batches solves the same
                                      // DFT butterfly for different problems
            kernel(MOVE_ADDR(in_real, batch_in_offset + i * ct_in_stride),
                   MOVE_ADDR(in_imag, batch_in_offset + i * ct_in_stride),
                   MOVE_ADDR(out_real, batch_out_offset + i * ct_out_stride),
                   MOVE_ADDR(out_imag, batch_out_offset + i * ct_out_stride),
                   rem_iters, strides, &tw_local, direction);
        }
    }

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return SOLVER_SUCCESS;
}

/**
 * Parallelizes over DFT butterflies within a CT stage.
 *
 * Example:
 *   problem size: 100:1:1v48:100:100 (column-major CT problem)
 *   num_threads: 48
 *   batches: 100
 *   problem size: 48 (4 x 12) -> [12v4 (12 butterflies of radix-4), 4v12 (4 butterflies of radix-12)]
 *
 *   for radix-4 kernel:
 *      #omp parallel for
 *      for (1..12) // butterflies
 *          for (1..100) // batches
 *              kernel(radix-4)
 *
 *   for radix-12 kernel:
 *      #omp parallel for
 *      for (1..4) // butterflies
 *          for (1..100) // batches
 *              kernel(radix-12)
 */
static INT32 execute_mt_direct_batched_colmajor_solver(aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");


    kfft_ kernel = sol->solver->kernel_c2c->kfft;
    aoclfftz_strides_t *strides = sol->strides_grp->strides;
    UINT8 direction = FFT_DIR(sol->decomp_scheme->flags);

    VOID *in_real = sol->decomp_scheme->in_real;
    VOID *in_imag = sol->decomp_scheme->in_imag;
    VOID *out_real = sol->decomp_scheme->out_real;
    VOID *out_imag = sol->decomp_scheme->out_imag;

    UINT32 dt_bytes = SOL_DT_SIZE(sol);

    // vec-strides across DFT butterflies of the same CT problem
    INTP ct_in_stride =
        sol->decomp_scheme->vecs[0].in_stride * DATA_STRIDE * dt_bytes;
    INTP ct_out_stride =
        sol->decomp_scheme->vecs[0].out_stride * DATA_STRIDE * dt_bytes;
    // INTP outer_iters = sol->decomp_scheme->vecs[0].n;
    // INTP inner_iters = sol->decomp_scheme->batched_vecs[0].n;
    INT32 num_sets = sol->solver->kernel_c2c->sets;
    INTP num_iters = sol->decomp_scheme->batched_vecs[0].n / num_sets;
    INTP rem_iters =
        sol->decomp_scheme->batched_vecs[0].n - (num_iters * num_sets);

    INTP num_threads = sol->decomp_scheme->thread_info->n_threads;

    omp_set_num_threads(sol->decomp_scheme->thread_info->n_threads);

    #pragma omp parallel for collapse(2) schedule(static)
    for (INTP i = 0; i < sol->decomp_scheme->vecs[0].n; i++)
    {
        for( INTP block = 0; block < num_threads; block++)
        {
            UINT8 num_sets = sol->solver->kernel_c2c->sets;
            // INTP num_iters = sol->decomp_scheme->batched_vecs[0].n / num_sets;
            // INTP rem_iters =
            //     sol->decomp_scheme->batched_vecs[0].n - (num_iters * num_sets);
            INTP block_sz = num_iters / num_threads;
            INTP rem_blocks = num_iters % num_threads; // Remaining blocks to distribute

            // Calculate this thread's work chunk
            INTP start_iter = block * block_sz + (block < rem_blocks ? block : rem_blocks);
            INTP end_iter = start_iter + block_sz + (block < rem_blocks ? 1 : 0);
            INTP thread_iters = end_iter - start_iter;
            aoclfftz_twiddle_t tw_thr_local = {
                .TW = MOVE_ADDR(sol->twiddle->TW, i * DATA_STRIDE * dt_bytes),
                .cols = sol->twiddle->cols,
                .twiddle_buf_ptr = sol->twiddle->twiddle_buf_ptr,
                .load_multi_cols = 0, // use same twiddle values across batches
            };                        // since different batches solves the same
                                      // DFT butterfly for different problems

            // Calculate input/output offsets for this thread's chunk
            INTP thread_in_offset = start_iter * strides->v_in_stride * dt_bytes * num_sets;
            INTP thread_out_offset = start_iter * strides->v_out_stride * dt_bytes * num_sets;

            kernel(MOVE_ADDR(in_real, (i * ct_in_stride) + thread_in_offset),
                    MOVE_ADDR(in_imag, (i * ct_in_stride) + thread_in_offset),
                    MOVE_ADDR(out_real, (i * ct_out_stride) + thread_out_offset),
                    MOVE_ADDR(out_imag, (i * ct_out_stride) + thread_out_offset),
                    thread_iters * num_sets, strides, &tw_thr_local, direction);
        }
    }

    // Handle remaining iterations (rem_iters) after the parallel section
    if (rem_iters > 0)
    {
        for (INTP i = 0; i < sol->decomp_scheme->vecs[0].n; i++)
        {
            aoclfftz_twiddle_t tw_local = {
                .TW = MOVE_ADDR(sol->twiddle->TW, i * DATA_STRIDE * dt_bytes),
                .cols = sol->twiddle->cols,
                .twiddle_buf_ptr = sol->twiddle->twiddle_buf_ptr,
                .load_multi_cols = 0, // use same twiddle values across batches
            };                        // since different batches solves the same
                                      // DFT butterfly for different problems

            INTP rem_in_offset = num_iters * strides->v_in_stride * dt_bytes * num_sets;
            INTP rem_out_offset = num_iters * strides->v_out_stride * dt_bytes * num_sets;

            kernel(MOVE_ADDR(in_real, (i * ct_in_stride) + rem_in_offset),
                   MOVE_ADDR(in_imag, (i * ct_in_stride) + rem_in_offset),
                   MOVE_ADDR(out_real, (i * ct_out_stride) + rem_out_offset),
                   MOVE_ADDR(out_imag, (i * ct_out_stride) + rem_out_offset),
                   rem_iters, strides, &tw_local, direction);
        }
    }
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return SOLVER_SUCCESS;
}

dft_solver_ register_execute_mt_direct_solver(VOID)
{
    return execute_mt_direct_solver;
}

dft_solver_ register_execute_mt_direct_batched_rowmajor_solver(VOID)
{
    return execute_mt_direct_batched_rowmajor_solver;
}

dft_solver_ register_execute_mt_direct_batched_colmajor_solver(VOID)
{
    return execute_mt_direct_batched_colmajor_solver;
}
