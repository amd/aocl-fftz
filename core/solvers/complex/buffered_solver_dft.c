// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file buffered_solver_dft.c
 *
 *  @brief Buffered Solver that sets up buffer and its stride for the given
 *  problem
 *
 *  This file contains the functions that setup, execute and destroy
 *  the solver.
 *
 *  @author Jeya R
 */

#include "core/solvers/solver.h"
#include "core/common/memory_manager.h"

/**
 * @brief Configures output buffer strides and pointers for buffered solver
 *
 * Sets up the output stride configuration for both the current solution
 * and next solver.
 *
 * @param sol      Current solution object
 * @param next_sol Next solver's solution object
 */
static FFTZ_VOID setup_buffered_output_strides(aoclfftz_solution_t *sol,
                                          aoclfftz_solution_t *next_sol)
{
    sol->decomp_scheme->out_real = sol->dft_bufs->ct_buf_real;
    sol->decomp_scheme->out_imag = sol->dft_bufs->ct_buf_imag;

    // Calculate stride factor from batched vectors
    FFTZ_INTP batched_stride_factor = 1;
    if (sol->decomp_scheme->batched_vecs)
    {
        batched_stride_factor = sol->decomp_scheme->batched_vecs[0].n *
                                sol->decomp_scheme->batched_vecs[0].out_stride;
    }
    sol->decomp_scheme->dims[0].out_stride = batched_stride_factor;
    sol->decomp_scheme->vecs[0].out_stride =
        sol->decomp_scheme->dims[0].n * batched_stride_factor;

    // Propagate output configuration to next solver
    next_sol->decomp_scheme->out_real = sol->decomp_scheme->out_real;
    next_sol->decomp_scheme->out_imag = sol->decomp_scheme->out_imag;
    next_sol->decomp_scheme->dims[0].out_stride =
        sol->decomp_scheme->dims[0].out_stride;
    next_sol->decomp_scheme->vecs[0].out_stride =
        sol->decomp_scheme->vecs[0].out_stride;
}

FFTZ_INT32 setup_buffered_solver(aoclfftz_solution_t *sol,
                            aoclfftz_solution_t *next_sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    // Buffer allocated at NDIM level can be reused by any solution subtree
    // under the NDIM node (both next_sol and nd_sol paths), avoiding
    // reallocation.
    if (sol->dft_bufs->ct_buffer != NULL)
    {
        // Buffer already allocated, reuse it
        AOCLFFTZ_LOG(TRACE, global_logger_mode,
                     "Buffer already allocated, reusing it");
        setup_buffered_output_strides(sol, next_sol);
        AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
        return SOLVER_SUCCESS;
    }

    FFTZ_INT32 dim_rank = sol->decomp_scheme->dim_rank;
    FFTZ_INT32 vec_rank = sol->decomp_scheme->vec_rank;
    aoclfftz_dim_t_64_ *dims = sol->decomp_scheme->dims;
    aoclfftz_dim_t_64_ *vecs = sol->decomp_scheme->vecs;

    FFTZ_UINTP buffer_length = 1;
    FFTZ_UINTP buffer_size = 0;

    FFTZ_UINT32 dt_bytes = SOL_DT_SIZE(sol);

    for (FFTZ_INT32 i = 0; i < dim_rank; i++)
    {
        buffer_length *= (dims[i].n);
    }
    for (FFTZ_INT32 i = 0; i < vec_rank; i++)
    {
        buffer_length *= (vecs[i].n);
    }
    if (sol->decomp_scheme->batched_vecs)
    {
        // Multiply by both n and out_stride to account for strided access
        buffer_length *= (sol->decomp_scheme->batched_vecs[0].n) *
                         (sol->decomp_scheme->batched_vecs[0].out_stride);
    }

    buffer_size = GET_PADDED_SIZE(buffer_length * DATA_STRIDE * dt_bytes);
    FFTZ_INT32 active_threads = sol->decomp_scheme->thread_info->active_threads;
    ALLOC_ALIGN_UNINIT(sol->dft_bufs->ct_buffer, FFTZ_VOID,
        buffer_size * active_threads);
    if (sol->dft_bufs->ct_buffer == NULL)
    {
        AOCLFFTZ_ERROR("Failed to allocate ct_buffer of size %ld",
                       (long)(buffer_size * active_threads));
        return SOLVER_FAILURE;
    }

    sol->dft_bufs->ct_buf_allocated = 1;
    sol->dft_bufs->ct_buf_size = buffer_size;
    sol->dft_bufs->ct_buf_real = sol->dft_bufs->ct_buffer;
    sol->dft_bufs->ct_buf_imag =
        MOVE_ADDR(sol->dft_bufs->ct_buffer, SOL_DT_SIZE(sol));
    setup_buffered_output_strides(sol, next_sol);

    next_sol->dft_bufs->ct_buffer = sol->dft_bufs->ct_buffer;
    next_sol->dft_bufs->ct_buf_allocated = 0;
    next_sol->dft_bufs->ct_buf_size = sol->dft_bufs->ct_buf_size;
    next_sol->dft_bufs->ct_buf_real = sol->dft_bufs->ct_buffer;
    next_sol->dft_bufs->ct_buf_imag = sol->dft_bufs->ct_buf_imag;

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return SOLVER_SUCCESS;
}

static FFTZ_INT32 execute_buffered_solver(aoclfftz_solution_t *sol,
                                          aoclfftz_mutable_ctx_t *ctx)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    FFTZ_UINT32 dt_bytes = CTX_DT_SIZE(ctx);
    aoclfftz_solution_t *next_sol = sol->next_sol[0];
    aoclfftz_mutable_ctx_t child_ctx = *ctx;
    // Reroute the child's output into a private CT-buffer slot.
#ifdef MULTI_THREADING
    // Pick this thread's slot within the shared ct_buffer.
    child_ctx.out_real = MOVE_ADDR(ctx->ct_buf_base, ctx->ct_offset);
    child_ctx.out_imag = MOVE_ADDR(child_ctx.out_real, dt_bytes);
#else
    child_ctx.out_real = ctx->ct_buf_base;
    child_ctx.out_imag = MOVE_ADDR(ctx->ct_buf_base, dt_bytes);
#endif
    ctx->out_real = child_ctx.out_real;
    ctx->out_imag = child_ctx.out_imag;
    // Alias ct_buf_base to out so the inner CTL1D takes its in-place path:
    // (m) in -> out, then (r) out -> out.
    child_ctx.ct_buf_base = child_ctx.out_real;

    FFTZ_INT32 status = next_sol->solver->execute_solver(next_sol, &child_ctx);
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return status;
}

dft_solver_ register_execute_buffered_solver(FFTZ_VOID)
{
    return execute_buffered_solver;
}
