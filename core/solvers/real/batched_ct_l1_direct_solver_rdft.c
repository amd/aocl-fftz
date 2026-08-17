// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file batched_ct_l1_direct_solver_rdft.c
 *
 *  @brief Fused batched + 1-level CT solver for real FFT (R2C/C2R).
 *
 *  Runs many independent real transforms in one tight loop. Each transform is
 *  one CT level (n = radix_r × radix_m) executed as two direct kernel stages
 *  (stage_r, then stage_m) without going through BATCHED → BUFFERED → CT
 *  solver dispatch.
 *
 *  Selector sets up winning stage templates on sol->next_sol before setup.
 *  Setup allocates aux buffer and prepares strides/twiddles. Execute calls
 *  R2HC / R2HCF / C2C directly.
 *
 *  @author Amrin Fathima
 */

#include "core/solvers/solver.h"
#include "core/common/memory_manager.h"
#include "core/solvers/real/direct_solver_rdft_utils.h"

// Allocate stage_r → stage_m inter-stage scratch (single slot; ST-only today).
static FFTZ_INT32 setup_batched_ct_l1_rdft_aux_buffer(aoclfftz_solution_t *sol,
                                                      FFTZ_INTP n,
                                                      FFTZ_UINT32 dt_bytes)
{
    if (sol->dft_bufs->buffered->is_aux_buffer_allocated &&
        sol->dft_bufs->buffered->aux_buffer_1 != NULL)
    {
        return SOLVER_SUCCESS;
    }

    FFTZ_INTP aux_buf_size = GET_PADDED_SIZE(n * dt_bytes);
    // Single-threaded only; execute uses one aux slot with no per-thread
    // offset. Scale with active_threads when MT fused path is added.
    FFTZ_INTP num_slots = 1;
    FREE_ALIGN_ALLOCATED_MEM(sol->dft_bufs->buffered->aux_buffer_1);
    ALLOC_ALIGN_UNINIT(sol->dft_bufs->buffered->aux_buffer_1, FFTZ_VOID,
                       aux_buf_size * num_slots + dt_bytes);
    if (sol->dft_bufs->buffered->aux_buffer_1 == NULL)
    {
        AOCLFFTZ_ERROR("setup_batched_ct_l1_rdft_aux_buffer failed: %s",
                       get_status_string(AOCLFFTZ_MEMORY_FAILURE));
        return AOCLFFTZ_MEMORY_FAILURE;
    }

    sol->dft_bufs->buffered->aux_buf_size_per_thread = aux_buf_size;
    sol->dft_bufs->buffered->is_aux_buffer_allocated = 1;
    return SOLVER_SUCCESS;
}

// Setup entry: aux buffer and stride/twiddle prep for
// SOLVER_REAL_BATCHED_CT_L1_DIRECT. Stage templates are set up on
// sol->next_sol by selector_batched_ct_l1_direct_rdft before this runs.
FFTZ_INT32 setup_batched_ct_l1_direct_real_solver(aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    FFTZ_INTP n = sol->decomp_scheme->dims[0].n;
    FFTZ_UINT32 dt_bytes = SOL_DT_SIZE(sol);
    FFTZ_UINT8 is_bwd =
        FFT_DIR(sol->decomp_scheme->flags) == BACKWARD_FFT_DIR;
    FFTZ_INT32 ret = SOLVER_SUCCESS;

    ret = setup_batched_ct_l1_rdft_aux_buffer(sol, n, dt_bytes);
    if (ret != SOLVER_SUCCESS)
    {
        goto exit_setup;
    }

    if (sol->next_sol == NULL || sol->next_sol->next_sol == NULL)
    {
        ret = AOCLFFTZ_SETUP_FAILURE;
        goto exit_setup;
    }

    sol->twiddle->twiddle_buf_ptr = NULL;
    sol->twiddle->TW = NULL;
    sol->twiddle->load_multi_cols = 1;

    setup_rdft_dc_nyquist_offsets_ds(sol->decomp_scheme);

    // Top-level batched output stride for set_zero_for_dc_and_nyquist_batched.
    sol->strides_grp->strides->v_out_stride =
        is_bwd ? sol->decomp_scheme->vecs[0].out_stride
               : sol->decomp_scheme->vecs[0].out_stride * 2;
    sol->strides_grp->strides->v_in_stride =
        is_bwd ? sol->decomp_scheme->vecs[0].in_stride * 2
               : sol->decomp_scheme->vecs[0].in_stride;

exit_setup:
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return ret;
}

// Run one CT stage (R2HC, then R2HCF, then C2C) with direct kernel calls.
static FFTZ_VOID
execute_batched_ct_l1_rdft_stage(aoclfftz_solution_t *stage_sol,
                                 aoclfftz_mutable_ctx_t *ctx,
                                 FFTZ_VOID *in, FFTZ_VOID *out)
{
    FFTZ_UINT8 direction = FFT_DIR(stage_sol->decomp_scheme->flags);
    kfft_ kfft_r2hc = stage_sol->solver->kernel_r2hc->kfft[FORWARD_FFT_DIR];
    kfft_ kfft_r2hcf = stage_sol->solver->kernel_r2hcf->kfft[FORWARD_FFT_DIR];

    if (stage_sol->solver->kernel_r2hc->count > 0)
    {
        kfft_r2hc(in, in, out, out, stage_sol->solver->kernel_r2hc->count,
                  stage_sol->strides_grp->strides_r2hc, stage_sol->twiddle,
                  direction);
    }
    if (stage_sol->solver->kernel_r2hcf->count > 0)
    {
        kfft_r2hcf(in, in, out, out, stage_sol->solver->kernel_r2hcf->count,
                   stage_sol->strides_grp->strides_r2hcf, stage_sol->twiddle,
                   direction);
    }

    if (stage_sol->solver->kernel_c2c->count == 0)
    {
        return;
    }

    // execute_c2c_kernels_rdft() skips the R2HC/R2HCF bands itself, as the
    // regrouped aux layout makes that offset depend on the stage's kernel mix.
    execute_c2c_kernels_rdft(stage_sol, ctx, in, out);
}

// Execute all batches: stage_r → aux → stage_m → out, then zero DC/Nyquist
// (fwd).
static FFTZ_INT32 execute_real_batched_ct_l1_direct_solver(
    aoclfftz_solution_t *sol, aoclfftz_mutable_ctx_t *ctx)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    aoclfftz_solution_t *stage_r = sol->next_sol;
    aoclfftz_solution_t *stage_m = stage_r->next_sol;

    FFTZ_UINT32 dt_bytes = SOL_DT_SIZE(sol);
    FFTZ_UINT8 is_fwd = (FFT_DIR(sol->decomp_scheme->flags) == FORWARD_FFT_DIR);
    FFTZ_VOID *aux = ctx->aux_pool_base_1;

    FFTZ_INTP batches = sol->decomp_scheme->vecs[0].n;
    FFTZ_INTP v_in_stride =
       is_fwd ? sol->decomp_scheme->vecs[0].in_stride * dt_bytes
              : sol->decomp_scheme->vecs[0].in_stride * DATA_STRIDE * dt_bytes;
    FFTZ_INTP v_out_stride =
       is_fwd ? sol->decomp_scheme->vecs[0].out_stride * DATA_STRIDE * dt_bytes
               : sol->decomp_scheme->vecs[0].out_stride * dt_bytes;

    FFTZ_VOID *in = ctx->in_real;
    FFTZ_VOID *out = ctx->out_real;

    for (FFTZ_INTP b = 0; b < batches; b++)
    {
        execute_batched_ct_l1_rdft_stage(stage_r, ctx, in, aux);
        execute_batched_ct_l1_rdft_stage(stage_m, ctx, aux, out);

        in = MOVE_ADDR(in, v_in_stride);
        out = MOVE_ADDR(out, v_out_stride);
    }

    if (is_fwd)
    {
        set_zero_for_dc_and_nyquist_batched(sol, ctx->out_real);
    }

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SOLVER_SUCCESS;
}

dft_solver_ register_execute_real_batched_ct_l1_direct_solver(FFTZ_VOID)
{
    return execute_real_batched_ct_l1_direct_solver;
}
