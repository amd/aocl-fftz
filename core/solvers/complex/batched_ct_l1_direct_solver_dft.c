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
 *    kfft_m    = sol->solver->kernel_c2c->kfft[dir]
 *    vecs_m    = sol->solver->kernel_c2c->count
 *    strides_m = sol->strides_grp->strides
 *    kfft_r    = sol->solver->kernel_c2c_r->kfft[dir]
 *    vecs_r    = sol->solver->kernel_c2c_r->count
 *    strides_r = sol->strides_grp->strides_c2c
 *    twiddle_r = sol->twiddle
 *
 *  @author Niranjan Reddy
 */

#include "core/solvers/solver.h"
#include "core/common/memory_manager.h"

static FFTZ_INT32 fill_direct_strides(aoclfftz_strides_t *strides,
                                      FFTZ_INTP radix, FFTZ_INTP dim_in_stride,
                                      FFTZ_INTP dim_out_stride,
                                      FFTZ_INTP vec_in_stride,
                                      FFTZ_INTP vec_out_stride)
{
    // Free any pre-existing arrays: solution objects can be reused across
    // different ND dimensions, so the radix (and thus array size) may differ.
    FREE_ALIGN_ALLOCATED_MEM(strides->in_strides);
    FREE_ALIGN_ALLOCATED_MEM(strides->out_strides);
    strides->in_strides = NULL;
    strides->out_strides = NULL;

    FFTZ_INT32 ret = alloc_and_fill_stride_arrays(strides, radix,
                                             dim_in_stride, dim_out_stride);
    if (ret != SOLVER_SUCCESS)
    {
        return ret;
    }

    strides->v_in_sym_stride = strides->v_in_stride =
        vec_in_stride * DATA_STRIDE;
    strides->v_out_sym_stride = strides->v_out_stride =
        vec_out_stride * DATA_STRIDE;
    return SOLVER_SUCCESS;
}

FFTZ_INT32 setup_batched_ct_l1_direct_solver(aoclfftz_solution_t *sol,
                                         kernel_t *ker_m, kernel_t *ker_r,
                                         FFTZ_INTP radix_r, FFTZ_INTP radix_m)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    FFTZ_INTP n = sol->decomp_scheme->dims[0].n;
    FFTZ_INTP in_stride  = sol->decomp_scheme->dims[0].in_stride;
    FFTZ_INTP out_stride = sol->decomp_scheme->dims[0].out_stride;
    FFTZ_INT32 ret = SOLVER_SUCCESS;

    FFTZ_UINT32 dt_bytes = SOL_DT_SIZE(sol);
    FFTZ_INTP buffer_out_stride = 1;

    // Unpadded: the inherited NDIM pool has no per-slice padding, if parent is mt_batched
    // the padded ct_buf_size might cross the assigned memory slot, hence avoid padding.
    FFTZ_INTP ct_buf_size = n * DATA_STRIDE * dt_bytes;

    // Allocate a private scratch buffer when no parent buffer exists
    if (sol->dft_bufs->ct_buf_real == NULL)
    {
        ct_buf_size = GET_PADDED_SIZE(n * DATA_STRIDE * dt_bytes);
        FFTZ_INT32 n_bufs = sol->decomp_scheme->thread_info->active_threads;
        ALLOC_ALIGN_UNINIT(sol->dft_bufs->ct_buffer, FFTZ_VOID, ct_buf_size
                                                                * n_bufs);
        if (sol->dft_bufs->ct_buffer == NULL)
        {
            ret = AOCLFFTZ_MEMORY_FAILURE;
            goto exit_setup;
        }
        sol->dft_bufs->ct_buf_allocated = 1;
        sol->dft_bufs->ct_buf_real = sol->dft_bufs->ct_buffer;
        sol->dft_bufs->ct_buf_imag =
            MOVE_ADDR(sol->dft_bufs->ct_buffer, dt_bytes);
    }
    sol->dft_bufs->ct_buf_size = ct_buf_size;

    sol->solver->kernel_c2c->kfft[FORWARD_FFT_DIR] =
        ker_m->kfft[FORWARD_FFT_DIR];
    sol->solver->kernel_c2c->kfft[BACKWARD_FFT_DIR] =
        ker_m->kfft[BACKWARD_FFT_DIR];
    sol->solver->kernel_c2c->count = (FFTZ_UINTP)radix_r;
    sol->solver->kernel_c2c->sets =
        ker_m->sets[DT_PRECISION_FLAG(sol->decomp_scheme->flags) - 2];

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

    sol->solver->kernel_c2c_r->kfft[FORWARD_FFT_DIR] =
        ker_r->kfft[FORWARD_FFT_DIR];
    sol->solver->kernel_c2c_r->kfft[BACKWARD_FFT_DIR] =
        ker_r->kfft[BACKWARD_FFT_DIR];
    sol->solver->kernel_c2c_r->count = (FFTZ_UINTP)radix_m;
    sol->solver->kernel_c2c_r->sets =
        ker_r->sets[DT_PRECISION_FLAG(sol->decomp_scheme->flags) - 2];

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
    sol->twiddle->load_multi_cols = 1;

    sol->next_sol = NULL;

exit_setup:
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return ret;
}

static FFTZ_INT32 execute_batched_ct_l1_direct_solver(aoclfftz_solution_t *sol,
                                                    aoclfftz_mutable_ctx_t *ctx)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    FFTZ_UINT32 dt_bytes = CTX_DT_SIZE(ctx);
    FFTZ_UINT8 direction = FFT_DIR(ctx->flags);

    kfft_ kfft_m = sol->solver->kernel_c2c->kfft[direction];
    aoclfftz_strides_t *strides_m = sol->strides_grp->strides;
    FFTZ_INTP vecs_m = (FFTZ_INTP)sol->solver->kernel_c2c->count;

    kfft_ kfft_r = sol->solver->kernel_c2c_r->kfft[direction];
    aoclfftz_strides_t *strides_r = sol->strides_grp->strides_c2c;
    aoclfftz_twiddle_t *twiddle_r = sol->twiddle;
    FFTZ_INTP vecs_r = (FFTZ_INTP)sol->solver->kernel_c2c_r->count;

    FFTZ_INTP v_in_stride = sol->decomp_scheme->vecs[0].in_stride *
                            DATA_STRIDE * dt_bytes;
    FFTZ_INTP v_out_stride = sol->decomp_scheme->vecs[0].out_stride *
                             DATA_STRIDE * dt_bytes;

    FFTZ_VOID *in_real  = ctx->in_real;
    FFTZ_VOID *in_imag  = ctx->in_imag;
    FFTZ_VOID *out_real = ctx->out_real;
    FFTZ_VOID *out_imag = ctx->out_imag;

    FFTZ_INTP batches = sol->decomp_scheme->vecs[0].n;

    if (ctx->ct_buf_base != out_real)
    {
        // Out-of-place path: radix-m writes into this thread's private CT slot,
        // radix-r then reads that slot into out. (m) in -> ct, (r) ct -> out.
        FFTZ_VOID *ct_buf_real;
        FFTZ_VOID *ct_buf_imag;

        // Pick this thread's slot within the shared ct_buffer.
        ct_buf_real = MOVE_ADDR(ctx->ct_buf_base, ctx->ct_offset);
        ct_buf_imag = MOVE_ADDR(ct_buf_real, dt_bytes);

        for (FFTZ_INTP b = 0; b < batches; b++)
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
    }
    else
    {
        // In-place path (ct_buf_base aliased to out): (m) in -> out, (r) out -> out.
        for (FFTZ_INTP b = 0; b < batches; b++)
        {
            kfft_m(in_real, in_imag, out_real, out_imag,
                vecs_m, strides_m, NULL, direction);
            kfft_r(out_real, out_imag, out_real, out_imag,
                vecs_r, strides_r, twiddle_r, direction);

            in_real  = MOVE_ADDR(in_real,  v_in_stride);
            in_imag  = MOVE_ADDR(in_imag,  v_in_stride);
            out_real = MOVE_ADDR(out_real, v_out_stride);
            out_imag = MOVE_ADDR(out_imag, v_out_stride);
        }
    }

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return SOLVER_SUCCESS;
}

dft_solver_ register_execute_batched_ct_l1_direct_solver(FFTZ_VOID)
{
    return execute_batched_ct_l1_direct_solver;
}
