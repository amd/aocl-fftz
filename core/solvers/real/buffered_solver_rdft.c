// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file buffered_solver_rdft.c
 *
 *  @brief Buffered Solver that sets up auxiliary buffers for real CT problems
 *
 *  This file contains the functions that setup, execute and destroy
 *  the solver.
 *
 *  @author Srirammaswamy Srinivasan
 */

#include "core/common/memory_manager.h"

FFTZ_INT32 setup_real_buffered_solver(aoclfftz_solution_t *sol,
                                 aoclfftz_realhelper_t *realhelper)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    FFTZ_INT32 status = SOLVER_SUCCESS;
    realhelper->is_buffered_invoked = 1;

    FFTZ_INT32 dt_bytes = SOL_DT_SIZE(sol);
    FFTZ_INTP n = sol->decomp_scheme->dims[0].n;

    // FUTURE: if buffered MT is added, update active_threads = active_threads * n_threads.
    FFTZ_INTP num_slots =
        (FFTZ_INTP)sol->decomp_scheme->thread_info->active_threads;

    FFTZ_INTP aux_buf_size = GET_PADDED_SIZE(n * dt_bytes);

    FREE_ALIGN_ALLOCATED_MEM(sol->dft_bufs->buffered->aux_buffer_1);
    FREE_ALIGN_ALLOCATED_MEM(sol->dft_bufs->buffered->aux_buffer_2);
    ALLOC_ALIGN_UNINIT(sol->dft_bufs->buffered->aux_buffer_1, FFTZ_VOID,
                       aux_buf_size * num_slots + 2 * dt_bytes); // FIXME: remove the "+ 2*dt_bytes" padding
    ALLOC_ALIGN_UNINIT(sol->dft_bufs->buffered->aux_buffer_2, FFTZ_VOID,
                       aux_buf_size * num_slots + 2 * dt_bytes); // FIXME: remove the "+ 2*dt_bytes" padding
    if (sol->dft_bufs->buffered->aux_buffer_1 == NULL ||
        sol->dft_bufs->buffered->aux_buffer_2 == NULL)
    {
        FREE_ALIGN_ALLOCATED_MEM(sol->dft_bufs->buffered->aux_buffer_1);
        FREE_ALIGN_ALLOCATED_MEM(sol->dft_bufs->buffered->aux_buffer_2);
        AOCLFFTZ_ERROR("setup_real_buffered_solver failed: %s",
                       get_status_string(AOCLFFTZ_MEMORY_FAILURE));
        return AOCLFFTZ_MEMORY_FAILURE;
    }

    sol->dft_bufs->buffered->aux_buf_size_per_thread = aux_buf_size;
    sol->dft_bufs->buffered->is_aux_buffer_allocated = 1;

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
#endif
    return status;
}

static FFTZ_INT32 execute_real_buffered_solver(aoclfftz_solution_t *sol,
                                               aoclfftz_mutable_ctx_t *ctx)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    FFTZ_INT32 ret = SOLVER_SUCCESS;

    // Select the aux region belonging to this thread and hand it down. This is
    // the only place the per-thread offset is applied; the nodes below just use
    // these pointers, and every Direct CT stage swaps aux_pool_base_1/_2 once
    // its kernels are done.
    aoclfftz_mutable_ctx_t child_ctx = *ctx;
    FFTZ_INTP aux_size = sol->dft_bufs->buffered->aux_buf_size_per_thread;
    FFTZ_INTP aux_off = (FFTZ_INTP)ctx->slot_idx * aux_size;
    child_ctx.aux_pool_base_1 = MOVE_ADDR(ctx->aux_pool_base_1, aux_off);
    child_ctx.aux_pool_base_2 = MOVE_ADDR(ctx->aux_pool_base_2, aux_off);

    ret = sol->next_sol->solver->execute_solver(sol->next_sol, &child_ctx);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return ret;
}

dft_solver_ register_execute_real_buffered_solver(FFTZ_VOID)
{
    return execute_real_buffered_solver;
}
