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
 */

#include "core/common/memory_manager.h"
#include "utils/utils.h"

INT32 setup_mt_direct_solver(aoclfftz_solution_t *sol, cost_analysis_t *cost,
                          kernel_t *kernel)
{
    aoclfftz_decomp_scheme_t *decomp_scheme = sol->decomp_scheme;
#ifdef AOCL_ENABLE_LOG
    INT32 logger_mode = decomp_scheme->cntrl_params->logger_mode;
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Enter");
#endif

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
        UINT32 dt_prec, dt_bytes;
        INTP v_in_stride, v_out_stride, data_offset;
        dt_prec = DT_PRECISION_FLAG(decomp_scheme->flags);
        dt_bytes = DT_PRECISION_BYTES(dt_prec);
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
            kernel->kfft((VOID *)((CHAR *)decomp_scheme->in_real  + v_istride),
                         (VOID *)((CHAR *)decomp_scheme->in_imag  + v_istride),
                         (VOID *)((CHAR *)decomp_scheme->out_real + v_ostride),
                         (VOID *)((CHAR *)decomp_scheme->out_imag + v_ostride),
                         num_sets, strides, sol->twiddle->TW, FFT_DIR(decomp_scheme->flags));
        }

        // Process the tail cases of the kernel
        if(rem_iters)
        {
            INTP v_istride = num_iters * v_in_stride;
            INTP v_ostride = num_iters * v_out_stride;
            kernel->kfft((VOID *)((CHAR *)decomp_scheme->in_real  + v_istride),
                         (VOID *)((CHAR *)decomp_scheme->in_imag  + v_istride),
                         (VOID *)((CHAR *)decomp_scheme->out_real + v_ostride),
                         (VOID *)((CHAR *)decomp_scheme->out_imag + v_ostride),
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

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit");
#endif
    return status;
}

static INT32 execute_mt_direct_solver(aoclfftz_solution_t *sol)
{
    aoclfftz_decomp_scheme_t *decomp_scheme = sol->decomp_scheme;
#ifdef AOCL_ENABLE_LOG
    INT32 logger_mode = decomp_scheme->cntrl_params->logger_mode;
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Enter");
#endif

    kfft_ kernel = sol->solver->kernel_c2c->kfft;
    UINT8 num_sets = sol->solver->kernel_c2c->sets;
    aoclfftz_strides_t *strides = sol->strides_grp->strides;

    UINT32 dt_prec, dt_bytes;
    INTP v_in_stride, v_out_stride, data_offset;
    dt_prec = DT_PRECISION_FLAG(decomp_scheme->flags);
    dt_bytes = DT_PRECISION_BYTES(dt_prec);
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
            .TW = MOVE_ADDR(sol->twiddle->TW, DATA_STRIDE * dt_bytes * batch),
            .twiddle_buf_ptr = sol->twiddle->twiddle_buf_ptr,
            .cols = sol->twiddle->cols,
        };

        INTP v_istride = batch * v_in_stride;
        INTP v_ostride = batch * v_out_stride;
        kernel((VOID *)((CHAR *)decomp_scheme->in_real  + v_istride),
               (VOID *)((CHAR *)decomp_scheme->in_imag  + v_istride),
               (VOID *)((CHAR *)decomp_scheme->out_real + v_ostride),
               (VOID *)((CHAR *)decomp_scheme->out_imag + v_ostride),
               num_sets, strides, &tw_local, FFT_DIR(decomp_scheme->flags));
    }

    // Process the tail cases of the kernel
    aoclfftz_twiddle_t tw_local = {
        .TW = MOVE_ADDR(sol->twiddle->TW, DATA_STRIDE * dt_bytes * num_iters),
        .twiddle_buf_ptr = sol->twiddle->twiddle_buf_ptr,
        .cols = sol->twiddle->cols,
    };
    if (rem_iters)
    {
        INTP v_istride = num_iters * v_in_stride;
        INTP v_ostride = num_iters * v_out_stride;
        kernel((VOID *)((CHAR *)decomp_scheme->in_real  + v_istride),
               (VOID *)((CHAR *)decomp_scheme->in_imag  + v_istride),
               (VOID *)((CHAR *)decomp_scheme->out_real + v_ostride),
               (VOID *)((CHAR *)decomp_scheme->out_imag + v_ostride),
               rem_iters, strides, &tw_local, FFT_DIR(decomp_scheme->flags));
    }

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit");
#endif
    return SOLVER_SUCCESS;
}

dft_solver_ register_execute_mt_direct_solver(VOID)
{
    return execute_mt_direct_solver;
}
