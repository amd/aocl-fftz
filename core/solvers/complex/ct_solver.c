/**
 * Copyright (C) 2023-2025, Advanced Micro Devices. All rights reserved.
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

/** @file ct_solver.c
 *
 *  @brief Cooley Tukey Solver that decomposes and solves an input problem
 *
 *  This file contains the functions that setup, execute and destroy
 *  the solver.
 *
 *  @author S. Biplab Raut
 */

#include "core/solvers/complex/ct_solver.h"
#include "api/aoclfftz_internal.h"
#include "core/common/twiddle.h"
#include "core/common/memory_manager.h"
#include "core/kernels/transpose/transpose_utils.h"

INT32 setup_ct_solver(aoclfftz_solution_t *sol, aoclfftz_solution_t *sol_r,
                      aoclfftz_solution_t *sol_m, UINT32 radix_r,
                      UINT32 radix_m)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");


    // Setup radix-m sub-problem
    // out-of-order -> in-order for out-of-place problems
    // out-of-order -> out-of-order for inplace problems
    COPY_SOLUTION_OBJ(sol_m, sol);
    sol_m->decomp_scheme->decomp_level = sol->decomp_scheme->decomp_level + 1;
    sol_m->decomp_scheme->dims[0].n = radix_m;
    sol_m->decomp_scheme->dims[0].in_stride =
        radix_r * sol->decomp_scheme->dims[0].in_stride;
    sol_m->decomp_scheme->dims[0].out_stride =
    #if defined PERFORM_INTER_STAGE_PERMUTE
        (IS_OUT_OF_PLACE(sol->decomp_scheme->flags)) ?
        sol->decomp_scheme->dims[0].out_stride :
        radix_r * sol->decomp_scheme->dims[0].out_stride;
    #else
        sol->decomp_scheme->dims[0].out_stride;
    #endif

    sol_m->decomp_scheme->vecs[0].n = radix_r;
    sol_m->decomp_scheme->vecs[0].in_stride =
        sol->decomp_scheme->dims[0].in_stride;
    sol_m->decomp_scheme->vecs[0].out_stride =
    #if defined PERFORM_INTER_STAGE_PERMUTE
        (IS_OUT_OF_PLACE(sol->decomp_scheme->flags)) ?
        radix_m * sol->decomp_scheme->dims[0].out_stride :
        sol->decomp_scheme->dims[0].out_stride;
    #else
        radix_m * sol->decomp_scheme->dims[0].out_stride;
    #endif

    // Setup radix-r sub-problem
    // out-of-order -> out-of-order for inplace & out-of-place problems
    COPY_SOLUTION_OBJ_OUT_P(sol_r, sol);
    sol_r->decomp_scheme->dims[0].n = radix_r;
    sol_r->decomp_scheme->dims[0].in_stride =
        radix_m * sol->decomp_scheme->dims[0].out_stride;
    sol_r->decomp_scheme->dims[0].out_stride =
        radix_m * sol->decomp_scheme->dims[0].out_stride;
    sol_r->decomp_scheme->vecs[0].n = radix_m;
    sol_r->decomp_scheme->vecs[0].in_stride =
        sol->decomp_scheme->dims[0].out_stride;
    sol_r->decomp_scheme->vecs[0].out_stride =
        sol->decomp_scheme->dims[0].out_stride;

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

	return SOLVER_SUCCESS;
}

#if defined PERFORM_INTER_STAGE_PERMUTE

static VOID transpose_buffer_float(INTP rows, INTP cols, INTP cstride, VOID *in)
{
    // rows = sets
    // cols = radix

    // buffer to mark processed index
    aoclfftz_complex_f_t *in_complex = (aoclfftz_complex_f_t *)in;

    INTP size = rows * cols - 1;
    UINT8 *visited = NULL;
    ALLOC_ALIGN_INIT(visited, UINT8, (size + 1) * sizeof(UINT8));

    visited[0] = 1;
    visited[size] = 1;
    INTP rstride = cstride * cols;

    aoclfftz_complex_f_t hold;
    INTP next;
    INTP start;
    INTP iter = 1;

    while (iter < size)
    {
        start = iter;
        INTP true_index = NTH_ELEMS_LINEAR_IDX_2D(iter, cols, cstride, rstride);
        hold = in_complex[true_index];

        do
        {
            next = (iter * rows) % size;
            true_index = NTH_ELEMS_LINEAR_IDX_2D(next, cols, cstride, rstride);
            aoclfftz_complex_f_t temp = hold;
            hold = in_complex[true_index];
            in_complex[true_index] = temp;
            visited[iter] = 1;
            iter = next;
        } while (iter != start);

        for (iter = start + 1; iter < size && visited[iter]; ++iter);
    }

    FREE_ALIGN_ALLOCATED_MEM(visited);
}

static VOID transpose_buffer_double(INTP rows, INTP cols, INTP cstride,
                                    VOID *in)
{
    // rows = sets
    // cols = radix

    // buffer to mark processed index
    aoclfftz_complex_d_t *in_complex = (aoclfftz_complex_d_t *)in;

    INTP size = rows * cols - 1;
    UINT8 *visited = NULL;
    ALLOC_ALIGN_INIT(visited, UINT8, (size + 1) * sizeof(UINT8));

    visited[0] = 1;
    visited[size] = 1;
    INTP rstride = cstride * cols;

    aoclfftz_complex_d_t hold;
    INTP next;
    INTP start;
    INTP iter = 1;

    while (iter < size)
    {
        start = iter;
        INTP true_index = NTH_ELEMS_LINEAR_IDX_2D(iter, cols, cstride, rstride);
        hold = in_complex[true_index];

        do
        {
            next = (iter * rows) % size;
            true_index = NTH_ELEMS_LINEAR_IDX_2D(next, cols, cstride, rstride);
            aoclfftz_complex_d_t temp = hold;
            hold = in_complex[true_index];
            in_complex[true_index] = temp;
            visited[iter] = 1;
            iter = next;
        } while (iter != start);

        for (iter = start + 1; iter < size && visited[iter]; ++iter);
    }

    FREE_ALIGN_ALLOCATED_MEM(visited);
}

// TODO: This code is implemented specifically for use in the ct solver, but it
// can be generalized as a dedicated out-of-place-in-place transpose kernel.
static VOID transpose_using_scratch_buffer_double(INTP rows, INTP cols,
                                                  INTP cstride, VOID *in,
                                                  VOID *scratch_space)
{
    // rows = sets
    // cols = radix
    INTP rstride = cstride * cols;

    aoclfftz_complex_d_t *in_complex = (aoclfftz_complex_d_t *)in;
    aoclfftz_complex_d_t *sc_complex = (aoclfftz_complex_d_t *)scratch_space;

    INTP sc_rows = cols;
    INTP sc_cols = rows;
    INTP sc_cs = 1;
    INTP sc_rs = rows;

    INTP in_cs = cstride;
    INTP in_rs = rstride;

    for (INTP i = 0; i < rows; i++)
    {
        for (INTP j = 0; j < cols; j++)
        {
            sc_complex[LINEAR_IDX_2D(j, i, sc_cs, sc_rs)] =
                in_complex[LINEAR_IDX_2D(i, j, in_cs, in_rs)];
        }
    }

    INTP final_in_cs = in_cs;
    INTP final_in_rs = in_cs * rows;
    INTP final_sc_cs = sc_cs;
    INTP final_sc_rs = sc_cs * rows;

    for (INTP i = 0; i < sc_rows; i++)
    {
        for (INTP j = 0; j < sc_cols; j++)
        {
            in_complex[LINEAR_IDX_2D(i, j, final_in_cs, final_in_rs)] =
                sc_complex[LINEAR_IDX_2D(i, j, final_sc_cs, final_sc_rs)];
        }
    }
}

static VOID transpose_using_scratch_buffer_float(INTP rows, INTP cols,
                                                 INTP cstride, VOID *in,
                                                 VOID *scratch_space)
{
    // rows = sets
    // cols = radix
    INTP rstride = cstride * cols;

    aoclfftz_complex_f_t *in_complex = (aoclfftz_complex_f_t *)in;
    aoclfftz_complex_f_t *sc_complex = (aoclfftz_complex_f_t *)scratch_space;

    INTP sc_rows = cols;
    INTP sc_cols = rows;
    INTP sc_cs = 1;
    INTP sc_rs = rows;

    INTP in_cs = cstride;
    INTP in_rs = rstride;

    for (INTP i = 0; i < rows; i++)
    {
        for (INTP j = 0; j < cols; j++)
        {
          sc_complex[LINEAR_IDX_2D(j, i, sc_cs, sc_rs)] =
              in_complex[LINEAR_IDX_2D(i, j, in_cs, in_rs)];
        }
    }

    INTP final_in_cs = in_cs;
    INTP final_in_rs = in_cs * rows;
    INTP final_sc_cs = sc_cs;
    INTP final_sc_rs = sc_cs * rows;

    for (INTP i = 0; i < sc_rows; i++)
    {
        for (INTP j = 0; j < sc_cols; j++)
        {
            in_complex[LINEAR_IDX_2D(i, j, final_in_cs, final_in_rs)] =
                sc_complex[LINEAR_IDX_2D(i, j, final_sc_cs, final_sc_rs)];
        }
    }
}

static VOID transpose_buffer(aoclfftz_solution_t *sol,
                              aoclfftz_solution_t *radix_r_sol )
{
    // for in place problems, we must first transpose the output of the "m"
    // subproblem. this is done using 2 different algorithms based on how large
    // the problem size is. if the problem size is small enough to fit inside
    // the scratch space, we do an out of place transpose into the scratch
    // space, and a copy into the original buffer. Otherwise, we resort to using
    // the cycles based transpose algorithm.

    if (!IS_OUT_OF_PLACE(sol->decomp_scheme->flags))
    {
        INTP sets = radix_r_sol->decomp_scheme->vecs[0].n;  // rows
        INTP radix = radix_r_sol->decomp_scheme->dims[0].n; // cols
        INTP total_size = radix * sets * DATA_STRIDE;
        UINT32 precision = DT_PRECISION_FLAG(sol->decomp_scheme->flags);

        INTP available = scratch_space_capacity / (2 * (1 << precision));

        VOID *in = sol->decomp_scheme->out_real;

        if (total_size > available || sol->dft_bufs->scratch_space == NULL)
        {
            INTP cstride = sol->decomp_scheme->dims[0].in_stride;

            if (DT_FLOAT == precision)
            {
                transpose_buffer_float(sets, radix, cstride, in);
            }
            else
            {
                transpose_buffer_double(sets, radix, cstride, in);
            }
        }
        else
        {
            INTP cstride = sol->decomp_scheme->dims[0].in_stride;

            if (DT_FLOAT == precision)
            {
                transpose_using_scratch_buffer_float(
                    sets, radix, cstride, in, sol->dft_bufs->scratch_space);
            }
            else
            {
                transpose_using_scratch_buffer_double(
                    sets, radix, cstride, in, sol->dft_bufs->scratch_space);
            }
        }
    }

    return;
}

#endif

static INT32 execute_ct_solver(aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    INT32 status = SOLVER_SUCCESS;
    aoclfftz_solution_t *radix_r_sol = sol->next_sol[0];
    aoclfftz_solution_t *radix_m_sol = radix_r_sol->next_sol[0];
    // update radix-m & radix-r solution data pointers
    radix_m_sol->decomp_scheme->in_real  = sol->decomp_scheme->in_real;
    radix_m_sol->decomp_scheme->in_imag  = sol->decomp_scheme->in_imag;
    radix_m_sol->decomp_scheme->out_real = sol->decomp_scheme->out_real;
    radix_m_sol->decomp_scheme->out_imag = sol->decomp_scheme->out_imag;
    radix_m_sol->decomp_scheme->flags = sol->decomp_scheme->flags;

    radix_r_sol->decomp_scheme->in_real  = sol->decomp_scheme->out_real;
    radix_r_sol->decomp_scheme->in_imag  = sol->decomp_scheme->out_imag;
    radix_r_sol->decomp_scheme->out_real = sol->decomp_scheme->out_real;
    radix_r_sol->decomp_scheme->out_imag = sol->decomp_scheme->out_imag;
    radix_r_sol->decomp_scheme->flags = sol->decomp_scheme->flags;

    // execute radix-m sub-problem
    radix_m_sol->solver->execute_solver(radix_m_sol);

    if (IS_OUT_OF_PLACE(sol->decomp_scheme->flags))
    {
        status = twiddle_multiplier(radix_r_sol);
    }
    else
    {
        status = twiddle_multiplier_inplace(radix_r_sol);
    }

    if (status != SOLVER_SUCCESS)
    {
        return SOLVER_FAILURE;
    }

    // execute radix-r DFT
    radix_r_sol->solver->execute_solver(radix_r_sol);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return status;
}


static INT32 execute_ct_twiddle_solver(aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    INT32 status = SOLVER_SUCCESS;
    aoclfftz_solution_t *radix_r_sol = sol->next_sol[0];
    aoclfftz_solution_t *radix_m_sol = radix_r_sol->next_sol[0];

    // update radix-m & radix-r solution data pointers
    radix_m_sol->decomp_scheme->in_real  = sol->decomp_scheme->in_real;
    radix_m_sol->decomp_scheme->in_imag  = sol->decomp_scheme->in_imag;
    radix_m_sol->decomp_scheme->out_real = sol->dft_bufs->ct_buf_real;
    radix_m_sol->decomp_scheme->out_imag = sol->dft_bufs->ct_buf_imag;
    radix_m_sol->dft_bufs->ct_buf_real = sol->dft_bufs->ct_buf_real;
    radix_m_sol->dft_bufs->ct_buf_imag = sol->dft_bufs->ct_buf_imag;
    radix_m_sol->decomp_scheme->flags = sol->decomp_scheme->flags;

    radix_r_sol->decomp_scheme->in_real  = radix_m_sol->decomp_scheme->out_real;
    radix_r_sol->decomp_scheme->in_imag  = radix_m_sol->decomp_scheme->out_imag;
    radix_r_sol->decomp_scheme->out_real = sol->decomp_scheme->out_real;
    radix_r_sol->decomp_scheme->out_imag = sol->decomp_scheme->out_imag;
    radix_r_sol->decomp_scheme->flags = sol->decomp_scheme->flags;

    // execute radix-m sub-problem
    radix_m_sol->solver->execute_solver(radix_m_sol);

#if defined PERFORM_INTER_STAGE_PERMUTE
    transpose_buffer(sol, radix_r_sol);
#endif

    // execute radix-r DFT
    radix_r_sol->solver->execute_solver(radix_r_sol);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return status;
}

dft_solver_ register_execute_ct_solver(VOID)
{
    return execute_ct_solver;
}

dft_solver_ register_execute_ct_twiddle_solver(VOID)
{
    return execute_ct_twiddle_solver;
}
