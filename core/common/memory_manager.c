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

 /** @file memory_manager.h
 *
 *  @brief Declares memory allocation and management functions of AOCL-FFTZ.
 *
 *  This file contains the function declarations for performing memory
 *  allocation management of different modules of the AOCL-FFTZ library.
 *
 *  @author S. Biplab Raut
 */

#include "core/common/memory_manager.h"

aoclfftz_decomp_scheme_t *alloc_decomp_scheme(INT32 vec_rank, INT32 dim_rank)
{
    aoclfftz_decomp_scheme_t* decomp_scheme = NULL;

    ALLOC_ALIGN_UNINIT(decomp_scheme, aoclfftz_decomp_scheme_t,
                         sizeof(aoclfftz_decomp_scheme_t));
    if (decomp_scheme)
    {
        ALLOC_ALIGN_UNINIT(decomp_scheme->dims, aoclfftz_dim_t_64_,
                             dim_rank * sizeof(aoclfftz_dim_t_64_));
        ALLOC_ALIGN_UNINIT(decomp_scheme->vecs, aoclfftz_dim_t_64_,
                             vec_rank * sizeof(aoclfftz_dim_t_64_));
        if (decomp_scheme->dims == NULL || decomp_scheme->vecs == NULL)
        {
            destroy_decomp_scheme(decomp_scheme);
            return NULL;
        }
        return decomp_scheme;
    }
    else
    {
        return NULL;
    }
}

aoclfftz_solution_t *alloc_solution(INT32 vec_rank, INT32 dim_rank)
{
    aoclfftz_solution_t *sol = NULL;

    ALLOC_ALIGN_UNINIT(sol, aoclfftz_solution_t, sizeof(aoclfftz_solution_t));
    if (sol)
    {
        ALLOC_ALIGN_UNINIT(sol->solver, aoclfftz_generic_solver_t,
                            sizeof(aoclfftz_generic_solver_t));
        sol->solver->execute_solver = NULL;
        sol->solver->kernel_r = NULL;
        sol->solver->kernel_m = NULL;
        sol->solver->destroy_solver = NULL;
        sol->decomp_scheme = alloc_decomp_scheme(vec_rank, dim_rank);
        ALLOC_ALIGN_INIT(sol->strides, aoclfftz_strides_t,
                         sizeof(aoclfftz_strides_t));
        ALLOC_ALIGN_UNINIT(sol->twiddle, aoclfftz_twiddle_t,
                                sizeof(aoclfftz_twiddle_t));
        ALLOC_ALIGN_UNINIT(sol->bluestein, aoclfftz_bluestein_t,
                             sizeof(aoclfftz_bluestein_t));
        sol->next_sol = NULL;
        if (sol->solver == NULL || sol->decomp_scheme == NULL ||
            sol->strides == NULL || sol->bluestein == NULL || sol->twiddle == NULL)
        {
            FREE_ALIGN_ALLOCATED_MEM(sol->solver);
            destroy_decomp_scheme(sol->decomp_scheme);
            FREE_ALIGN_ALLOCATED_MEM(sol->strides);
            destroy_bluestein(sol->bluestein);
            FREE_ALIGN_ALLOCATED_MEM(sol->twiddle);
            FREE_ALIGN_ALLOCATED_MEM(sol);
            return NULL;
        }
        sol->twiddle->TW = NULL;
        sol->bluestein->B = NULL;
        sol->bluestein->B_out = NULL;
        sol->bluestein->in = NULL;
        sol->bluestein->out = NULL;
        sol->bluestein->is_B_out_valid = 0;
        return sol;
    }
    else
    {
        return NULL;
    }
}

aoclfftz_selector_t *alloc_selector(INT32 vec_rank, INT32 dim_rank)
{
    aoclfftz_selector_t *selector = NULL;

    ALLOC_ALIGN_UNINIT(selector, aoclfftz_selector_t,
                         sizeof(aoclfftz_selector_t));
    if (selector)
    {
        selector->solution = alloc_solution(vec_rank, dim_rank);
        ALLOC_ALIGN_UNINIT(selector->cost_analysis, cost_analysis_t,
                             sizeof(cost_analysis_t));
        if (selector->solution == NULL || selector->cost_analysis == NULL)
        {
            destroy_selector(selector);
            return NULL;
        }
        selector->cost_analysis->ops = 0;
        selector->cost_analysis->time = 0;
        return selector;
    }
    else
    {
        return NULL;
    }
}

VOID *alloc_bluestein_sequence(INTP n, UINT32 dt_prec)
{
    UINT32 dt_bytes;
    DT_PRECISION_BYTES(dt_prec);
    VOID *buffer = NULL;
    ALLOC_ALIGN_UNINIT(buffer, VOID, n * dt_bytes);
    return buffer;
}

VOID *alloc_twiddle_for_solution(UINT32 rad_size, UINT32 dt_prec)
{
    UINT32 dt_bytes;
    DT_PRECISION_BYTES(dt_prec);
    VOID *buffer = NULL;
    ALLOC_ALIGN_UNINIT(buffer, VOID, rad_size * dt_bytes);
    return buffer;
}

VOID destroy_decomp_scheme(aoclfftz_decomp_scheme_t *decomp_scheme)
{
    if (decomp_scheme != NULL)
    {
        FREE_ALIGN_ALLOCATED_MEM(decomp_scheme->dims);
        FREE_ALIGN_ALLOCATED_MEM(decomp_scheme->vecs);
        FREE_ALIGN_ALLOCATED_MEM(decomp_scheme);
    }
    return;
}

VOID destroy_solution(aoclfftz_solution_t *sol)
{
    aoclfftz_solution_t *cur_sol = NULL;
    while (sol != NULL)
    {
        cur_sol = sol;
        sol = sol->next_sol;
        FREE_ALIGN_ALLOCATED_MEM(cur_sol->solver);
        destroy_decomp_scheme(cur_sol->decomp_scheme);
        FREE_ALIGN_ALLOCATED_MEM(cur_sol->strides);
        destroy_bluestein(cur_sol->bluestein);
        FREE_ALIGN_ALLOCATED_MEM(cur_sol->twiddle->TW);
        FREE_ALIGN_ALLOCATED_MEM(cur_sol->twiddle);
        FREE_ALIGN_ALLOCATED_MEM(cur_sol);
    }
    return;
}

VOID destroy_selector_without_solution(aoclfftz_selector_t *sel)
{
    if (sel != NULL)
    {
        FREE_ALIGN_ALLOCATED_MEM(sel->cost_analysis);
        FREE_ALIGN_ALLOCATED_MEM(sel);
    }
    return;
}

VOID destroy_selector(aoclfftz_selector_t *sel)
{
    if (sel != NULL)
    {
        destroy_solution(sel->solution);
        destroy_selector_without_solution(sel);
    }
    return;
}

VOID destroy_bluestein(aoclfftz_bluestein_t *bluestein)
{
    if (bluestein != NULL)
    {
        FREE_ALIGN_ALLOCATED_MEM(bluestein->B);
        FREE_ALIGN_ALLOCATED_MEM(bluestein->B_out);
        FREE_ALIGN_ALLOCATED_MEM(bluestein->in);
        FREE_ALIGN_ALLOCATED_MEM(bluestein->out);
        FREE_ALIGN_ALLOCATED_MEM(bluestein);
    }
}
