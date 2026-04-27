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
static VOID setup_buffered_output_strides(aoclfftz_solution_t *sol,
                                          aoclfftz_solution_t *next_sol)
{
    sol->decomp_scheme->out_real = sol->dft_bufs->ct_buf_real;
    sol->decomp_scheme->out_imag = sol->dft_bufs->ct_buf_imag;

    // Calculate stride factor from batched vectors
    INTP batched_stride_factor = 1;
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

INT32 setup_buffered_solver(aoclfftz_solution_t *sol,
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

    INT32 dim_rank = sol->decomp_scheme->dim_rank;
    INT32 vec_rank = sol->decomp_scheme->vec_rank;
    aoclfftz_dim_t_64_ *dims = sol->decomp_scheme->dims;
    aoclfftz_dim_t_64_ *vecs = sol->decomp_scheme->vecs;

    UINTP buffer_length = 1;
    UINTP buffer_size = 0;

    UINT32 dt_bytes = SOL_DT_SIZE(sol);

    for (INT32 i = 0; i < dim_rank; i++)
    {
        buffer_length *= (dims[i].n);
    }
    for (INT32 i = 0; i < vec_rank; i++)
    {
        buffer_length *= (vecs[i].n);
    }
    if (sol->decomp_scheme->batched_vecs)
    {
        // Multiply by both n and out_stride to account for strided access
        buffer_length *= (sol->decomp_scheme->batched_vecs[0].n) *
                         (sol->decomp_scheme->batched_vecs[0].out_stride);
    }

    // Mirrors alloc_ndim_buffer (see that function for rationale) so both
    // allocation sites agree on layout.
    buffer_size =  buffer_length * DATA_STRIDE * dt_bytes;
    ALLOC_ALIGN_UNINIT(sol->dft_bufs->ct_buffer, VOID,
                       buffer_size * sol->dft_bufs->num_ct_buf);
    if (sol->dft_bufs->ct_buffer == NULL)
    {
        AOCLFFTZ_ERROR("Failed to allocate ct_buffer of size %ld",
                       (long)buffer_size);
        return SOLVER_FAILURE;
    }

    sol->dft_bufs->ct_buf_allocated = 1;
    sol->dft_bufs->ct_buf_size = buffer_size;
    sol->dft_bufs->ct_buf_real = sol->dft_bufs->ct_buffer;
    sol->dft_bufs->ct_buf_imag =
        MOVE_ADDR(sol->dft_bufs->ct_buffer, SOL_DT_SIZE(sol));
    setup_buffered_output_strides(sol, next_sol);
    next_sol->dft_bufs->ct_buffer = sol->dft_bufs->ct_buffer;

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return SOLVER_SUCCESS;
}

static INT32 execute_buffered_solver(aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    INT32 status = SOLVER_SUCCESS;
    aoclfftz_solution_t *next_sol = sol->next_sol[0];

    // Update sol and next_sol's output pointers to the buffer.
    next_sol->decomp_scheme->in_real = sol->decomp_scheme->in_real;
    next_sol->decomp_scheme->in_imag = sol->decomp_scheme->in_imag;
    // For inplace CT, reroute buf node output to ct_buf; Radix-m writes here,
    // Radix-r reads.
    sol->decomp_scheme->out_real = sol->dft_bufs->ct_buf_real;
    sol->decomp_scheme->out_imag = sol->dft_bufs->ct_buf_imag;

    next_sol->decomp_scheme->out_real = sol->dft_bufs->ct_buf_real;
    next_sol->decomp_scheme->out_imag = sol->dft_bufs->ct_buf_imag;
    next_sol->decomp_scheme->flags = sol->decomp_scheme->flags;
    next_sol->solver->execute_solver(next_sol);
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return status;
}

dft_solver_ register_execute_buffered_solver(VOID)
{
    return execute_buffered_solver;
}
