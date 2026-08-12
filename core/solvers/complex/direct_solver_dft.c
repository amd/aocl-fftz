// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file direct_solver_dft.c
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

#include "core/common/memory_manager.h"

FFTZ_INT32 setup_direct_solver(aoclfftz_solution_t *sol, cost_analysis_t *cost,
                          kernel_t *kernel)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");


    aoclfftz_strides_t *strides = sol->strides_grp->strides;
    // TODO: Update the batch to batched_vecs[0].n if batched_vecs is not NULL
    FFTZ_INTP batch = sol->decomp_scheme->vecs[0].n;
    FFTZ_INTP radix = sol->decomp_scheme->dims[0].n;
    FFTZ_UINT8 precision = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    FFTZ_UINT8 direction = FFT_DIR(sol->decomp_scheme->flags);
    FFTZ_INT32 status = SOLVER_SUCCESS;

    if (strides->in_strides == NULL)
    {
        FFTZ_INT32 ret = alloc_and_fill_stride_arrays(strides, radix,
                        sol->decomp_scheme->dims[0].in_stride,
                        sol->decomp_scheme->dims[0].out_stride);
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

    if (GET_SELECTOR_MODE(sol->decomp_scheme->flags) ==
        AOCLFFTZ_FIXED_SELECTOR)
    {
        cost->time = 0;
        cost->ops = compute_kernel_cost(kernel, precision, direction, batch);
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
        kfft_ kfft = sol->solver->kernel_c2c->kfft[direction];
        kfft(sol->decomp_scheme->in_real, sol->decomp_scheme->in_imag,
             sol->decomp_scheme->out_real, sol->decomp_scheme->out_imag, batch,
             strides, sol->twiddle, direction);

        getTime(endTime);
        cost->time = diffTime(clkTick, startTime, endTime);
        cost->ops = compute_kernel_cost(kernel, precision, direction, batch);
    }
#endif // AOCLFFTZ_AUTO_SELECTOR_MODE

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return status;
}

static FFTZ_INT32 execute_direct_solver(aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");


    aoclfftz_strides_t *strides = sol->strides_grp->strides;
    FFTZ_UINT8 direction = FFT_DIR(sol->decomp_scheme->flags);

    kfft_ kfft = sol->solver->kernel_c2c->kfft[direction];
    kfft(sol->decomp_scheme->in_real, sol->decomp_scheme->in_imag,
         sol->decomp_scheme->out_real, sol->decomp_scheme->out_imag,
         sol->decomp_scheme->vecs[0].n, strides, sol->twiddle, direction);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return SOLVER_SUCCESS;
}

/**
 * @brief Direct solver variant that handles two levels of batches in
 * column-major fashion.
 *
 * Unlike the standard direct solver which handles a single level of batches,
 * this variant processes two levels:
 *   - 1st level (outer loop): CT decomposition batches stored in vecs,
 *     representing DFT butterflies within a single problem
 *   - 2nd level (inner loop): Problem batches stored in batched_vecs,
 *     representing multiple independent FFT problems
 *
 * Column-major processing: Iterates over CT batches (vecs) in the outer loop,
 * executing all problem batches (batched_vecs) for each CT batch. This access
 * pattern is optimal when problem batch stride < elemental stride.
 */
static FFTZ_INT32
execute_direct_batched_colmajor_solver(aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");


    aoclfftz_strides_t *strides = sol->strides_grp->strides;
    FFTZ_UINT8 direction = FFT_DIR(sol->decomp_scheme->flags);
    kfft_ kernel = sol->solver->kernel_c2c->kfft[direction];

    FFTZ_VOID *in_real = sol->decomp_scheme->in_real;
    FFTZ_VOID *in_imag = sol->decomp_scheme->in_imag;
    FFTZ_VOID *out_real = sol->decomp_scheme->out_real;
    FFTZ_VOID *out_imag = sol->decomp_scheme->out_imag;

    FFTZ_UINT32 dt_bytes = SOL_DT_SIZE(sol);

    // vec-strides across DFT butterflies of the same CT problem
    FFTZ_INTP ct_in_stride =
        sol->decomp_scheme->vecs[0].in_stride * DATA_STRIDE * dt_bytes;
    FFTZ_INTP ct_out_stride =
        sol->decomp_scheme->vecs[0].out_stride * DATA_STRIDE * dt_bytes;

    // execute the direct kernel
    for (FFTZ_INTP i = 0; i < sol->decomp_scheme->vecs[0].n; i++)
    {
        aoclfftz_twiddle_t tw_local = {
            .twiddle_buf_ptr = sol->twiddle->twiddle_buf_ptr,
            .TW = MOVE_ADDR(sol->twiddle->TW, i * DATA_STRIDE * dt_bytes),
            .cols = sol->twiddle->cols,
            .load_multi_cols = 0, // use same twiddle values across batches
        };                        // since different batches solve the same
                                  // DFT butterfly of different problems
        kernel(in_real, in_imag, out_real, out_imag,
               sol->decomp_scheme->batched_vecs[0].n, strides, &tw_local,
               direction);
        in_real = MOVE_ADDR(in_real, ct_in_stride);
        in_imag = MOVE_ADDR(in_imag, ct_in_stride);
        out_real = MOVE_ADDR(out_real, ct_out_stride);
        out_imag = MOVE_ADDR(out_imag, ct_out_stride);
    }

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return SOLVER_SUCCESS;
}

dft_solver_ register_execute_direct_solver(FFTZ_VOID)
{
    return execute_direct_solver;
}

dft_solver_ register_execute_direct_batched_colmajor_solver(FFTZ_VOID)
{
    return execute_direct_batched_colmajor_solver;
}
