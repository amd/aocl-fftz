// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file batched_ct_l1_direct_solver_dft.c
 *
 *  @brief Fused batched + 1-level CT solver.
 *
 *  Handles problems where the inner FFT decomposes into exactly one CT
 *  level: n = radix_r * radix_m, both directly supported by kernels.
 *  Calls the radix_m (standard) and radix_r (twiddle) kernel function
 *  pointers directly in a tight batch loop -- no solver dispatch, no
 *  recursion.
 *
 *  Data layout in sol->:
 *    kfft_m    = sol->solver->kernel_c2c->kfft
 *    vecs_m    = sol->solver->kernel_c2c->count
 *    strides_m = sol->strides_grp->strides
 *    kfft_r    = sol->solver->kernel_c2c_r->kfft
 *    vecs_r    = sol->solver->kernel_c2c_r->count
 *    strides_r = sol->strides_grp->strides_c2c
 *    twiddle_r = sol->twiddle
 *
 *  @author Niranjan Reddy
 */

#include "core/solvers/solver.h"
#include "core/common/memory_manager.h"

static INT32 fill_direct_strides(aoclfftz_strides_t *strides, INTP radix,
                          INTP dim_in_stride, INTP dim_out_stride,
                          INTP vec_in_stride, INTP vec_out_stride)
{
    // Free any pre-existing arrays: solution objects can be reused across
    // different ND dimensions, so the radix (and thus array size) may differ.
    FREE_ALIGN_ALLOCATED_MEM(strides->in_strides);
    FREE_ALIGN_ALLOCATED_MEM(strides->out_strides);
    strides->in_strides = NULL;
    strides->out_strides = NULL;

    INT32 ret = alloc_and_fill_stride_arrays(strides, radix,
                                             dim_in_stride, dim_out_stride);
    if (ret != SOLVER_SUCCESS)
    {
        return ret;
    }

    strides->v_in_stride  = vec_in_stride  * DATA_STRIDE;
    strides->v_out_stride = vec_out_stride * DATA_STRIDE;
    return SOLVER_SUCCESS;
}

INT32 setup_batched_ct_l1_direct_solver(aoclfftz_solution_t *sol,
                                         kernel_t *ker_m, kernel_t *ker_r,
                                         INTP radix_r, INTP radix_m)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    INTP n = sol->decomp_scheme->dims[0].n;
    INTP in_stride  = sol->decomp_scheme->dims[0].in_stride;
    INTP out_stride = sol->decomp_scheme->dims[0].out_stride;
    INT32 ret = SOLVER_SUCCESS;

    UINT32 dt_bytes = SOL_DT_SIZE(sol);
    INTP buffer_out_stride = 1;

    // Allocate a private scratch buffer when no parent buffer exists or
    // when the parent's is BUFFERED.
    // Otherwise reuse the parent ND buffer already pointed to by ct_buf_real.
    // TODO: In all cases, we should use the parent ND buffer instead of allocating a new one.
    if (sol->dft_bufs->ct_buf_real == NULL ||
        sol->decomp_scheme->out_real == sol->dft_bufs->ct_buf_real)
    {
        UINTP buffer_length = (UINTP)n;
        UINTP ct_buf_size = buffer_length * DATA_STRIDE * dt_bytes;
        ALLOC_ALIGN_UNINIT(sol->dft_bufs->ct_buffer, VOID, ct_buf_size);
        if (sol->dft_bufs->ct_buffer == NULL)
        {
            ret = AOCLFFTZ_MEMORY_FAILURE;
            goto exit_setup;
        }
        sol->dft_bufs->ct_buf_allocated = 1;
        sol->dft_bufs->ct_buf_size = ct_buf_size;
        sol->dft_bufs->ct_buf_real = sol->dft_bufs->ct_buffer;
        sol->dft_bufs->ct_buf_imag =
            MOVE_ADDR(sol->dft_bufs->ct_buffer, dt_bytes);
    }

    sol->solver->kernel_c2c->kfft  = ker_m->kfft;
    sol->solver->kernel_c2c->count = (UINTP)radix_r;

    {
        aoclfftz_strides_t *strides_m = sol->strides_grp->strides;
        ret = fill_direct_strides(strides_m, radix_m,
                                  radix_r * in_stride, buffer_out_stride,
                                  in_stride, radix_m * buffer_out_stride);
        if (ret != SOLVER_SUCCESS)
        {
            ret = AOCLFFTZ_MEMORY_FAILURE;
            goto exit_setup;
        }
    }

    sol->solver->kernel_c2c_r->kfft  = ker_r->kfft;
    sol->solver->kernel_c2c_r->count = (UINTP)radix_m;

    {
        aoclfftz_strides_t *strides_r = sol->strides_grp->strides_c2c;
        ret = fill_direct_strides(strides_r, radix_r,
                                  radix_m * buffer_out_stride,
                                  radix_m * out_stride,
                                  buffer_out_stride, out_stride);
        if (ret != SOLVER_SUCCESS)
        {
            ret = AOCLFFTZ_MEMORY_FAILURE;
            goto exit_setup;
        }
    }

    sol->twiddle->twiddle_buf_ptr = NULL;
    sol->twiddle->TW = NULL;
    sol->twiddle->cols = 0;
    sol->twiddle->load_multi_cols = 1;

    sol->next_sol = NULL;

exit_setup:
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return ret;
}

static INT32 execute_batched_ct_l1_direct_solver(aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    kfft_ kfft_m = sol->solver->kernel_c2c->kfft;
    aoclfftz_strides_t *strides_m = sol->strides_grp->strides;
    INTP vecs_m = (INTP)sol->solver->kernel_c2c->count;

    kfft_ kfft_r = sol->solver->kernel_c2c_r->kfft;
    aoclfftz_strides_t *strides_r = sol->strides_grp->strides_c2c;
    aoclfftz_twiddle_t *twiddle_r = sol->twiddle;
    INTP vecs_r = (INTP)sol->solver->kernel_c2c_r->count;

    UINT32 dt_bytes = SOL_DT_SIZE(sol);
    UINT8 direction = FFT_DIR(sol->decomp_scheme->flags);

    INTP v_in_stride = sol->decomp_scheme->vecs[0].in_stride *
                       DATA_STRIDE * dt_bytes;
    INTP v_out_stride = sol->decomp_scheme->vecs[0].out_stride *
                        DATA_STRIDE * dt_bytes;

    VOID *in_real  = sol->decomp_scheme->in_real;
    VOID *in_imag  = sol->decomp_scheme->in_imag;
    VOID *out_real = sol->decomp_scheme->out_real;
    VOID *out_imag = sol->decomp_scheme->out_imag;
    VOID *ct_buf_real = sol->dft_bufs->ct_buf_real;
    VOID *ct_buf_imag = sol->dft_bufs->ct_buf_imag;

    INTP batches = sol->decomp_scheme->vecs[0].n;

    for (INTP b = 0; b < batches; b++)
    {
        kfft_m(in_real, in_imag, ct_buf_real, ct_buf_imag,
               vecs_m, strides_m, NULL, direction);
        kfft_r(ct_buf_real, ct_buf_imag, out_real, out_imag,
               vecs_r, strides_r, twiddle_r, direction);

        in_real  = MOVE_ADDR(in_real,  v_in_stride);
        in_imag  = MOVE_ADDR(in_imag,  v_in_stride);
        out_real = MOVE_ADDR(out_real, v_out_stride);
        out_imag = MOVE_ADDR(out_imag, v_out_stride);
    }

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return SOLVER_SUCCESS;
}

dft_solver_ register_execute_batched_ct_l1_direct_solver(VOID)
{
    return execute_batched_ct_l1_direct_solver;
}
