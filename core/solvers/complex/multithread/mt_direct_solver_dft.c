// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file mt_direct_solver_dft.c
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

#include "core/common/memory_manager.h"

FFTZ_INT32 setup_mt_direct_solver(aoclfftz_solution_t *sol,
                                  cost_analysis_t *cost, kernel_t *kernel,
                                  FFTZ_UINT8 *has_nested)
{
    aoclfftz_decomp_scheme_t *decomp_scheme = sol->decomp_scheme;
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    if (decomp_scheme->thread_info->active_threads != 1)
    {
        *has_nested = 1;
    }

    aoclfftz_strides_t *strides = sol->strides_grp->strides;
    FFTZ_INTP n = decomp_scheme->vecs[0].n;
    FFTZ_INTP radix = decomp_scheme->dims[0].n;
    FFTZ_UINT8 precision = DT_PRECISION_FLAG(decomp_scheme->flags);
    FFTZ_UINT8 direction = FFT_DIR(decomp_scheme->flags);
    FFTZ_INT32 status = SOLVER_SUCCESS;

    if (strides->in_strides == NULL)
    {
        FFTZ_INT32 ret = alloc_and_fill_stride_arrays(strides, radix,
                        decomp_scheme->dims[0].in_stride,
                        decomp_scheme->dims[0].out_stride);
        if (ret != SOLVER_SUCCESS)
        {
            return ret;
        }
    }

    if (sol->decomp_scheme->batched_vecs != NULL)
    {
        strides->v_in_h2_stride = strides->v_in_stride =
            sol->decomp_scheme->batched_vecs[0].in_stride * DATA_STRIDE;
        strides->v_out_h2_stride = strides->v_out_stride =
            sol->decomp_scheme->batched_vecs[0].out_stride * DATA_STRIDE;
    }
    else
    {
        strides->v_in_h2_stride = strides->v_in_stride =
            sol->decomp_scheme->vecs[0].in_stride * DATA_STRIDE;
        strides->v_out_h2_stride = strides->v_out_stride =
            sol->decomp_scheme->vecs[0].out_stride * DATA_STRIDE;
    }

    if (GET_SELECTOR_MODE(decomp_scheme->flags) == AOCLFFTZ_FIXED_SELECTOR)
    {
        cost->time = 0;
        cost->ops = compute_kernel_cost(kernel, precision, direction, n);
    }
#ifdef AOCLFFTZ_AUTO_SELECTOR_MODE
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
        FFTZ_INTP v_in_stride, v_out_stride, data_offset;
        FFTZ_UINT32 dt_bytes = SOL_DT_SIZE(sol);
        FFTZ_UINT8 num_sets = kernel->sets[precision - 2];
        data_offset = DATA_STRIDE * dt_bytes * num_sets;
        v_in_stride = decomp_scheme->vecs[0].in_stride * data_offset;
        v_out_stride = decomp_scheme->vecs[0].out_stride * data_offset;

        FFTZ_INTP num_iters = decomp_scheme->vecs[0].n / num_sets;
        FFTZ_INTP rem_iters = decomp_scheme->vecs[0].n - (num_iters * num_sets);

        kfft_ kfft = kernel->kfft[direction];

        // Set threads for parallel execution
        FFTZ_INT32 n_threads = decomp_scheme->thread_info->n_threads;
        #pragma omp parallel for num_threads(n_threads)
        for (FFTZ_INTP batch = 0; batch < num_iters; batch++)
        {
            FFTZ_INTP v_istride = batch * v_in_stride;
            FFTZ_INTP v_ostride = batch * v_out_stride;
            kfft(MOVE_ADDR(decomp_scheme->in_real, v_istride),
                 MOVE_ADDR(decomp_scheme->in_imag, v_istride),
                 MOVE_ADDR(decomp_scheme->out_real, v_ostride),
                 MOVE_ADDR(decomp_scheme->out_imag, v_ostride),
                 num_sets, strides, sol->twiddle->TW, direction);
        }

        // Process the tail cases of the kernel
        if (rem_iters)
        {
            FFTZ_INTP v_istride = num_iters * v_in_stride;
            FFTZ_INTP v_ostride = num_iters * v_out_stride;
            kfft(MOVE_ADDR(decomp_scheme->in_real, v_istride),
                 MOVE_ADDR(decomp_scheme->in_imag, v_istride),
                 MOVE_ADDR(decomp_scheme->out_real, v_ostride),
                 MOVE_ADDR(decomp_scheme->out_imag, v_ostride),
                 rem_iters, strides, sol->twiddle->TW, direction);
        }

        getTime(endTime);
        cost->time = diffTime(clkTick, startTime, endTime);
        cost->ops = compute_kernel_cost(kernel, precision, direction, n);
    }
#endif // AOCLFFTZ_AUTO_SELECTOR_MODE
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return status;
}

static FFTZ_INT32 execute_mt_direct_solver(aoclfftz_solution_t *sol,
                                           aoclfftz_mutable_ctx_t *ctx)
{
    aoclfftz_decomp_scheme_t *decomp_scheme = sol->decomp_scheme;
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");


    FFTZ_UINT8 direction = FFT_DIR(ctx->flags);
    kfft_ kernel = sol->solver->kernel_c2c->kfft[direction];
    FFTZ_UINT8 num_sets = sol->solver->kernel_c2c->sets;
    aoclfftz_strides_t *strides = sol->strides_grp->strides;

    FFTZ_INTP v_in_stride, v_out_stride, data_offset;
    FFTZ_UINT32 dt_bytes = CTX_DT_SIZE(ctx);
    data_offset = DATA_STRIDE * dt_bytes * num_sets;
    v_in_stride = decomp_scheme->vecs[0].in_stride * data_offset;
    v_out_stride = decomp_scheme->vecs[0].out_stride * data_offset;

    FFTZ_VOID *in_real  = ctx->in_real;
    FFTZ_VOID *in_imag  = ctx->in_imag;
    FFTZ_VOID *out_real = ctx->out_real;
    FFTZ_VOID *out_imag = ctx->out_imag;

    FFTZ_INTP num_iters = decomp_scheme->vecs[0].n / num_sets;
    FFTZ_INTP rem_iters = decomp_scheme->vecs[0].n - (num_iters * num_sets);

    FFTZ_INTP radix = decomp_scheme->dims[0].n;
    FFTZ_INTP tw_per_butterfly = (radix - 1) * (FFTZ_INTP)num_sets;

    // Set threads for parallel execution
    #pragma omp parallel for num_threads(decomp_scheme->thread_info->n_threads)
    for (FFTZ_INTP batch = 0; batch < num_iters; batch++)
    {
        aoclfftz_twiddle_t tw_local = {
            .twiddle_buf_ptr = sol->twiddle->twiddle_buf_ptr,
            .TW = MOVE_ADDR(sol->twiddle->TW,
                            DATA_STRIDE * dt_bytes * batch * tw_per_butterfly),
            .load_multi_cols = 1, // use different twiddle values across batches
        };

        FFTZ_INTP v_istride = batch * v_in_stride;
        FFTZ_INTP v_ostride = batch * v_out_stride;
        kernel(MOVE_ADDR(in_real, v_istride),
               MOVE_ADDR(in_imag, v_istride),
               MOVE_ADDR(out_real, v_ostride),
               MOVE_ADDR(out_imag, v_ostride),
               num_sets, strides, &tw_local, direction);
    }

    // Process the tail cases of the kernel
    // Process the tail cases of the kernel
    aoclfftz_twiddle_t tw_local = {
        .twiddle_buf_ptr = sol->twiddle->twiddle_buf_ptr,
        .TW = MOVE_ADDR(sol->twiddle->TW,
                        DATA_STRIDE * dt_bytes * num_iters * tw_per_butterfly),
        .load_multi_cols = 1, // use different twiddle values across batches
    };
    if (rem_iters)
    {
        FFTZ_INTP v_istride = num_iters * v_in_stride;
        FFTZ_INTP v_ostride = num_iters * v_out_stride;
        kernel(MOVE_ADDR(in_real, v_istride),
               MOVE_ADDR(in_imag, v_istride),
               MOVE_ADDR(out_real, v_ostride),
               MOVE_ADDR(out_imag, v_ostride),
               rem_iters, strides, &tw_local, direction);
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
 *   problem size: 48 (4 x 12) -> [12v4 (12 butterflies of radix-4), 4v12 (4
 * butterflies of radix-12)]
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
static FFTZ_INT32 execute_mt_direct_batched_rowmajor_solver(
                                                    aoclfftz_solution_t *sol,
                                                    aoclfftz_mutable_ctx_t *ctx)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");


    aoclfftz_strides_t *strides = sol->strides_grp->strides;
    FFTZ_UINT8 direction = FFT_DIR(ctx->flags);
    kfft_ kernel = sol->solver->kernel_c2c->kfft[direction];

    FFTZ_UINT32 dt_bytes = CTX_DT_SIZE(ctx);
    FFTZ_VOID *in_real  = ctx->in_real;
    FFTZ_VOID *in_imag  = ctx->in_imag;
    FFTZ_VOID *out_real = ctx->out_real;
    FFTZ_VOID *out_imag = ctx->out_imag;

    // vec-strides across DFT butterflies of the same CT problem
    FFTZ_INTP ct_in_stride =
        sol->decomp_scheme->vecs[0].in_stride * DATA_STRIDE * dt_bytes;
    FFTZ_INTP ct_out_stride =
        sol->decomp_scheme->vecs[0].out_stride * DATA_STRIDE * dt_bytes;

    FFTZ_INTP radix = sol->decomp_scheme->dims[0].n;
    FFTZ_INTP tw_per_butterfly = (radix - 1) * DATA_STRIDE * (FFTZ_INTP)dt_bytes;

    FFTZ_UINT8 num_sets = sol->solver->kernel_c2c->sets;
    FFTZ_INTP num_iters = sol->decomp_scheme->batched_vecs[0].n / num_sets;
    FFTZ_INTP rem_iters =
        sol->decomp_scheme->batched_vecs[0].n - (num_iters * num_sets);


    // Set threads for parallel execution
    // **NOTE**:
    // This `direct_batched` solver now parallelizes over batches instead of
    // DFT butterflies. This provides better scalability for large batched
    // problems with more number of threads.
    FFTZ_INT32 n_threads = sol->decomp_scheme->thread_info->n_threads;
    #pragma omp parallel for num_threads(n_threads)
    for (FFTZ_INTP batch = 0; batch < num_iters; batch++)
    {
        // Calculate batch offsets for input/output arrays
        FFTZ_INTP batch_in_offset =
            batch * strides->v_in_stride * dt_bytes * num_sets;
        FFTZ_INTP batch_out_offset =
            batch * strides->v_out_stride * dt_bytes * num_sets;

        // Sequential loop over DFT butterflies
        for (FFTZ_INTP i = 0; i < sol->decomp_scheme->vecs[0].n; i++)
        {
            aoclfftz_twiddle_t tw_local = {
                .twiddle_buf_ptr = sol->twiddle->twiddle_buf_ptr,
                .TW = MOVE_ADDR(sol->twiddle->TW, i * tw_per_butterfly),
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
        FFTZ_INTP batch_in_offset =
            num_iters * strides->v_in_stride * dt_bytes * num_sets;
        FFTZ_INTP batch_out_offset =
            num_iters * strides->v_out_stride * dt_bytes * num_sets;

        // Sequential loop over DFT butterflies
        for (FFTZ_INTP i = 0; i < sol->decomp_scheme->vecs[0].n; i++)
        {
            aoclfftz_twiddle_t tw_local = {
                .twiddle_buf_ptr = sol->twiddle->twiddle_buf_ptr,
                .TW = MOVE_ADDR(sol->twiddle->TW, i * tw_per_butterfly),
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
 *   problem size: 48 (4 x 12) -> [12v4 (12 butterflies of radix-4), 4v12 (4
 * butterflies of radix-12)]
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
static FFTZ_INT32 execute_mt_direct_batched_colmajor_solver(
                                                    aoclfftz_solution_t *sol,
                                                    aoclfftz_mutable_ctx_t *ctx)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");


    aoclfftz_strides_t *strides = sol->strides_grp->strides;
    FFTZ_UINT8 direction = FFT_DIR(ctx->flags);
    kfft_ kernel = sol->solver->kernel_c2c->kfft[direction];

    FFTZ_UINT32 dt_bytes = CTX_DT_SIZE(ctx);
    FFTZ_VOID *in_real  = ctx->in_real;
    FFTZ_VOID *in_imag  = ctx->in_imag;
    FFTZ_VOID *out_real = ctx->out_real;
    FFTZ_VOID *out_imag = ctx->out_imag;

    // vec-strides across DFT butterflies of the same CT problem
    FFTZ_INTP ct_in_stride =
        sol->decomp_scheme->vecs[0].in_stride * DATA_STRIDE * dt_bytes;
    FFTZ_INTP ct_out_stride =
        sol->decomp_scheme->vecs[0].out_stride * DATA_STRIDE * dt_bytes;

    FFTZ_INTP radix = sol->decomp_scheme->dims[0].n;
    FFTZ_INTP tw_per_butterfly = (radix - 1) * DATA_STRIDE * (FFTZ_INTP)dt_bytes;
    // FFTZ_INTP outer_iters = sol->decomp_scheme->vecs[0].n;
    // FFTZ_INTP inner_iters = sol->decomp_scheme->batched_vecs[0].n;
    FFTZ_INT32 num_sets = sol->solver->kernel_c2c->sets;
    FFTZ_INTP num_iters = sol->decomp_scheme->batched_vecs[0].n / num_sets;
    FFTZ_INTP rem_iters =
        sol->decomp_scheme->batched_vecs[0].n - (num_iters * num_sets);

    FFTZ_INT32 n_threads = sol->decomp_scheme->thread_info->n_threads;

    #pragma omp parallel for num_threads(n_threads) collapse(2) schedule(static)
    for (FFTZ_INTP i = 0; i < sol->decomp_scheme->vecs[0].n; i++)
    {
        for (FFTZ_INT32 block = 0; block < n_threads; block++)
        {
            FFTZ_UINT8 num_sets = sol->solver->kernel_c2c->sets;
            // FFTZ_INTP num_iters = sol->decomp_scheme->batched_vecs[0].n /
            // num_sets; FFTZ_INTP rem_iters =
            //     sol->decomp_scheme->batched_vecs[0].n - (num_iters *
            //     num_sets);
            FFTZ_INTP block_sz = num_iters / n_threads;
            FFTZ_INTP rem_blocks =
                num_iters % n_threads; // Remaining blocks to distribute

            // Calculate this thread's work chunk
            FFTZ_INTP start_iter =
                block * block_sz + (block < rem_blocks ? block : rem_blocks);
            FFTZ_INTP end_iter =
                start_iter + block_sz + (block < rem_blocks ? 1 : 0);
            FFTZ_INTP thread_iters = end_iter - start_iter;
            aoclfftz_twiddle_t tw_thr_local = {
                .twiddle_buf_ptr = sol->twiddle->twiddle_buf_ptr,
                .TW = MOVE_ADDR(sol->twiddle->TW, i * tw_per_butterfly),
                .load_multi_cols = 0, // use same twiddle values across batches
            };                        // since different batches solves the same
                                      // DFT butterfly for different problems

            // Calculate input/output offsets for this thread's chunk
            FFTZ_INTP thread_in_offset =
                start_iter * strides->v_in_stride * dt_bytes * num_sets;
            FFTZ_INTP thread_out_offset =
                start_iter * strides->v_out_stride * dt_bytes * num_sets;

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
        for (FFTZ_INTP i = 0; i < sol->decomp_scheme->vecs[0].n; i++)
        {
            aoclfftz_twiddle_t tw_local = {
                .twiddle_buf_ptr = sol->twiddle->twiddle_buf_ptr,
                .TW = MOVE_ADDR(sol->twiddle->TW, i * tw_per_butterfly),
                .load_multi_cols = 0, // use same twiddle values across batches
            };                        // since different batches solves the same
                                      // DFT butterfly for different problems

            FFTZ_INTP rem_in_offset =
                num_iters * strides->v_in_stride * dt_bytes * num_sets;
            FFTZ_INTP rem_out_offset =
                num_iters * strides->v_out_stride * dt_bytes * num_sets;

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

dft_solver_ register_execute_mt_direct_solver(FFTZ_VOID)
{
    return execute_mt_direct_solver;
}

dft_solver_ register_execute_mt_direct_batched_rowmajor_solver(FFTZ_VOID)
{
    return execute_mt_direct_batched_rowmajor_solver;
}

dft_solver_ register_execute_mt_direct_batched_colmajor_solver(FFTZ_VOID)
{
    return execute_mt_direct_batched_colmajor_solver;
}
