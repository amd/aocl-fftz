/**
 * Copyright (C) 2023, Advanced Micro Devices. All rights reserved.
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

/** @file batched_solver.c
 *
 *  @brief Batched Solver that sets up and solves a vector problem
 *
 *  This file contains the functions that setup, execute and destroy
 *  the solver.
 *
 *  @author S. Biplab Raut
 */

#include "core/solvers/ct_solver.h"
#include "core/common/twiddle.h"
#include "core/common/memory_manager.h"
#include "utils/utils.h"

INT32 setup_batched_solver(aoclfftz_solution_t *sol)
{
#ifdef AOCL_ENABLE_LOG
    INT32 logger_mode = sol->decomp_scheme->cntrl_params->logger_mode;
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Enter");
#endif

    // Turn the vector problem into a single set/unit problem to find its
    // solution
    sol->decomp_scheme->vec_rank = 1;
    sol->decomp_scheme->vecs[0].n = 1;

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit");
#endif
    return SOLVER_SUCCESS;
}

/*
 * Considerations and assumptions for execute_batched_solver():
 * For a multi-dimentional vector array (up to rank 3) of the DFT tranforms,
 * sol->decomp_scheme->vecs[rnk].in_stride gives the offset at which
 * input buffer starts for the current rank/position in the vector array,
 * sol->decomp_scheme->vecs[rnk].out_stride gives the offset at which
 * output buffer starts for the current rank/position in the vector array.
 */
static INT32 execute_batched_solver(aoclfftz_solution_t *sol)
{
#ifdef AOCL_ENABLE_LOG
    INT32 logger_mode = sol->decomp_scheme->cntrl_params->logger_mode;
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Enter");
#endif

    aoclfftz_solution_t *next_sol = sol->next_sol;
    INTP batch_size;
    INTP rnk_offset;
    INTP rnk;
    INT32 status = SOLVER_SUCCESS;
    UINT32 dt, dt_bytes;
    INTP v_in_stride_0;
    INTP v_out_stride_0;
    INTP v_in_stride_1;
    INTP v_out_stride_1;
    INTP v_in_stride_2;
    INTP v_out_stride_2;

    dt = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    DT_PRECISION_BYTES(dt);

    switch (sol->decomp_scheme->vec_rank)
    {
    case 1:
        v_in_stride_0 =
            sol->decomp_scheme->vecs[0].in_stride * DATA_STRIDE * dt_bytes;
        v_out_stride_0 =
            sol->decomp_scheme->vecs[0].out_stride * DATA_STRIDE * dt_bytes;

        next_sol->decomp_scheme->in_real = sol->decomp_scheme->in_real;
        next_sol->decomp_scheme->in_imag = sol->decomp_scheme->in_imag;
        next_sol->decomp_scheme->out_real = sol->decomp_scheme->out_real;
        next_sol->decomp_scheme->out_imag = sol->decomp_scheme->out_imag;

        for (batch_size = 0; batch_size < sol->decomp_scheme->vecs[0].n;
            batch_size++)
        {
            status = next_sol->solver->execute_solver(next_sol);
            if (status != SOLVER_SUCCESS)
                return status;

            next_sol->decomp_scheme->in_real =
                (VOID *)((CHAR *)next_sol->decomp_scheme->in_real +
                         v_in_stride_0);
            next_sol->decomp_scheme->in_imag =
                (VOID *)((CHAR *)next_sol->decomp_scheme->in_imag +
                         v_in_stride_0);
            next_sol->decomp_scheme->out_real =
                (VOID *)((CHAR *)next_sol->decomp_scheme->out_real +
                         v_out_stride_0);
            next_sol->decomp_scheme->out_imag =
                (VOID *)((CHAR *)next_sol->decomp_scheme->out_imag +
                         v_out_stride_0);
        }
        break;
    case 2:
        v_in_stride_0 =
            sol->decomp_scheme->vecs[0].in_stride * DATA_STRIDE * dt_bytes;
        v_out_stride_0 =
            sol->decomp_scheme->vecs[0].out_stride * DATA_STRIDE * dt_bytes;

        for (rnk_offset = 0; rnk_offset < sol->decomp_scheme->vecs[1].n;
             rnk_offset++)
        {
            v_in_stride_1 =
                rnk_offset * sol->decomp_scheme->vecs[1].in_stride *
                DATA_STRIDE * dt_bytes;
            v_out_stride_1 =
                rnk_offset * sol->decomp_scheme->vecs[1].out_stride *
                DATA_STRIDE * dt_bytes;

            next_sol->decomp_scheme->in_real =
                (VOID *)((CHAR *)sol->decomp_scheme->in_real + v_in_stride_1);
            next_sol->decomp_scheme->in_imag =
                (VOID *)((CHAR *)sol->decomp_scheme->in_imag + v_in_stride_1);
            next_sol->decomp_scheme->out_real =
                (VOID *)((CHAR *)sol->decomp_scheme->out_real + v_out_stride_1);
            next_sol->decomp_scheme->out_imag =
                (VOID *)((CHAR *)sol->decomp_scheme->out_imag + v_out_stride_1);

            for (batch_size = 0; batch_size < sol->decomp_scheme->vecs[0].n;
                 batch_size++)
            {
                status = next_sol->solver->execute_solver(next_sol);
                if (status != SOLVER_SUCCESS)
                    return status;

                next_sol->decomp_scheme->in_real =
                    (VOID *)((CHAR *)next_sol->decomp_scheme->in_real +
                             v_in_stride_0);
                next_sol->decomp_scheme->in_imag =
                    (VOID *)((CHAR *)next_sol->decomp_scheme->in_imag +
                             v_in_stride_0);
                next_sol->decomp_scheme->out_real =
                    (VOID *)((CHAR *)next_sol->decomp_scheme->out_real +
                             v_out_stride_0);
                next_sol->decomp_scheme->out_imag =
                    (VOID *)((CHAR *)next_sol->decomp_scheme->out_imag +
                             v_out_stride_0);
            }
        }
        break;
    case 3:
        v_in_stride_0 =
            sol->decomp_scheme->vecs[0].in_stride * DATA_STRIDE * dt_bytes;
        v_out_stride_0 =
            sol->decomp_scheme->vecs[0].out_stride * DATA_STRIDE * dt_bytes;

        for (rnk = 0; rnk < sol->decomp_scheme->vecs[2].n; rnk++)
        {
            v_in_stride_2 =
                rnk * sol->decomp_scheme->vecs[2].in_stride *
                DATA_STRIDE * dt_bytes;
            v_out_stride_2 =
                rnk * sol->decomp_scheme->vecs[2].out_stride *
                DATA_STRIDE * dt_bytes;

            next_sol->decomp_scheme->in_real =
                (VOID *)((CHAR *)sol->decomp_scheme->in_real + v_in_stride_2);
            next_sol->decomp_scheme->in_imag =
                (VOID *)((CHAR *)sol->decomp_scheme->in_imag + v_in_stride_2);
            next_sol->decomp_scheme->out_real =
                (VOID *)((CHAR *)sol->decomp_scheme->out_real + v_out_stride_2);
            next_sol->decomp_scheme->out_imag =
                (VOID *)((CHAR *)sol->decomp_scheme->out_imag + v_out_stride_2);

            // save pointer to restore it below since they will be moved while execution
            VOID *in_real = next_sol->decomp_scheme->in_real;
            VOID *in_imag = next_sol->decomp_scheme->in_imag;
            VOID *out_real = next_sol->decomp_scheme->out_real;
            VOID *out_imag = next_sol->decomp_scheme->out_imag;

            for (rnk_offset = 0; rnk_offset < sol->decomp_scheme->vecs[1].n;
                 rnk_offset++)
            {
                v_in_stride_1 =
                    rnk_offset * sol->decomp_scheme->vecs[1].in_stride *
                    DATA_STRIDE * dt_bytes;
                v_out_stride_1 =
                    rnk_offset * sol->decomp_scheme->vecs[1].out_stride *
                    DATA_STRIDE * dt_bytes;

                // restore to start location
                next_sol->decomp_scheme->in_real = in_real;
                next_sol->decomp_scheme->in_imag = in_imag;
                next_sol->decomp_scheme->out_real = out_real;
                next_sol->decomp_scheme->out_imag = out_imag;

                next_sol->decomp_scheme->in_real =
                    (VOID *)((CHAR *)next_sol->decomp_scheme->in_real +
                             v_in_stride_1);
                next_sol->decomp_scheme->in_imag =
                    (VOID *)((CHAR *)next_sol->decomp_scheme->in_imag +
                             v_in_stride_1);
                next_sol->decomp_scheme->out_real =
                    (VOID *)((CHAR *)next_sol->decomp_scheme->out_real +
                             v_out_stride_1);
                next_sol->decomp_scheme->out_imag =
                    (VOID *)((CHAR *)next_sol->decomp_scheme->out_imag +
                             v_out_stride_1);

                for (batch_size = 0; batch_size < sol->decomp_scheme->vecs[0].n;
                     batch_size++)
                {
                    status = next_sol->solver->execute_solver(next_sol);
                    if (status != SOLVER_SUCCESS)
                        return status;

                    next_sol->decomp_scheme->in_real =
                        (VOID *)((CHAR *)next_sol->decomp_scheme->in_real +
                                 v_in_stride_0);
                    next_sol->decomp_scheme->in_imag =
                        (VOID *)((CHAR *)next_sol->decomp_scheme->in_imag +
                                 v_in_stride_0);
                    next_sol->decomp_scheme->out_real =
                        (VOID *)((CHAR *)next_sol->decomp_scheme->out_real +
                                 v_out_stride_0);
                    next_sol->decomp_scheme->out_imag =
                        (VOID *)((CHAR *)next_sol->decomp_scheme->out_imag +
                                 v_out_stride_0);

                }
            }
        }
        break;
    default:
        return SOLVER_FAILURE;
    }

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit");
#endif
    return status;
}

dft_solver_ register_execute_batched_solver()
{
    return execute_batched_solver;
}
