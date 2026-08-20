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

#include <assert.h>

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
                       aux_buf_size * num_slots + dt_bytes); // FIXME: remove the "+ dt_bytes" padding
    ALLOC_ALIGN_UNINIT(sol->dft_bufs->buffered->aux_buffer_2, FFTZ_VOID,
                       aux_buf_size * num_slots + dt_bytes); // FIXME: remove the "+ dt_bytes" padding
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

    // composite problem input
    sol->next_sol[0]->decomp_scheme->in_real = sol->decomp_scheme->in_real;
    // composite problem output
    *sol->dft_bufs->buffered->out_ptr = sol->decomp_scheme->out_real;

#if REAL_FFT_EXECUTION_ORDER != REAL_FFT_ORDER_ITERATIVE
    // In recursive mode the tree is Buffered → CT → Direct(r, stage=0).
    // The batched solver advances Buffered's in_real per batch, which is
    // propagated above to CT. Direct(r, stage=0) also reads from the problem
    // input, so forward the per-batch pointer to it. Inner Direct nodes
    // (stage > 0) read from auxiliary buffers set at setup time and must
    // NOT be overwritten, so only the CT's immediate child is updated.
    // Invariant: the real buffered solver always wraps a CT problem
    // (Buffered -> CT -> Direct(r, stage=0)); the selector only chooses
    // REAL_BUFFERED when the kernel is not directly supported, so next_sol[0]
    // is always a REAL_CT node. Asserted in debug builds; the NULL guards below
    // keep release builds safe against a malformed chain.
    assert(sol->next_sol[0]->solver->solver_type == SOLVER_REAL_CT);
    if (sol->next_sol[0]->next_sol != NULL &&
        sol->next_sol[0]->next_sol[0] != NULL)
    {
        sol->next_sol[0]->next_sol[0]->decomp_scheme->in_real =
            sol->decomp_scheme->in_real;
    }
#endif

    ret = sol->next_sol[0]->solver->execute_solver(sol->next_sol[0], ctx);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return ret;
}

dft_solver_ register_execute_real_buffered_solver(FFTZ_VOID)
{
    return execute_real_buffered_solver;
}
