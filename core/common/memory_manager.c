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

    decomp_scheme = ALLOC_UNALIGN_UNINIT(sizeof(aoclfftz_decomp_scheme_t));
    if (decomp_scheme)
    {
        decomp_scheme->dims = 
            ALLOC_UNALIGN_UNINIT(vec_rank * sizeof(aoclfftz_dim_t_64_));
        decomp_scheme->vecs = 
            ALLOC_UNALIGN_UNINIT(dim_rank * sizeof(aoclfftz_dim_t_64_));
        if (decomp_scheme->dims == NULL || decomp_scheme->vecs == NULL)
        {
            FREE_ALLOCATED_MEM(decomp_scheme->dims);
            FREE_ALLOCATED_MEM(decomp_scheme->vecs);
            FREE_ALLOCATED_MEM(decomp_scheme);
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
    aoclfftz_solution_t* sol = NULL;

    sol = ALLOC_UNALIGN_UNINIT(sizeof(aoclfftz_solution_t));
    if (sol)
    {
        sol->solver = ALLOC_UNALIGN_UNINIT(sizeof(aoclfftz_generic_solver_t));
        sol->decomp_scheme = alloc_decomp_scheme(vec_rank, dim_rank);
        sol->strides = ALLOC_UNALIGN_UNINIT(sizeof(aoclfftz_strides_t));
        sol->twiddle = ALLOC_UNALIGN_UNINIT(sizeof(aoclfftz_twiddle_t));
        sol->next_sol = NULL;
        if (sol->solver == NULL || sol->decomp_scheme == NULL ||
            sol->strides == NULL || sol->twiddle == NULL)
        {
            FREE_ALLOCATED_MEM(sol->solver);
            FREE_ALLOCATED_MEM(sol->decomp_scheme);
            FREE_ALLOCATED_MEM(sol->strides);
            FREE_ALLOCATED_MEM(sol->twiddle);
            FREE_ALLOCATED_MEM(sol);
            return NULL;
        }
        return sol;
    }
    else
    {
        return NULL;
    }
}

aoclfftz_selector_t *alloc_selector(INT32 vec_rank, INT32 dim_rank)
{
    aoclfftz_selector_t* selector = NULL;
    selector = ALLOC_UNALIGN_UNINIT(sizeof(aoclfftz_selector_t));
    if (selector)
    {
        selector->solution = alloc_solution(vec_rank, dim_rank);
        selector->cost_analysis = 
            ALLOC_UNALIGN_UNINIT(sizeof(cost_analysis_t));
        if (selector->solution == NULL || selector->cost_analysis == NULL)
        {
            FREE_ALLOCATED_MEM(selector->solution);
            FREE_ALLOCATED_MEM(selector->cost_analysis);
            FREE_ALLOCATED_MEM(selector);
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

VOID *alloc_twiddle_for_solution(UINT32 rad_size, UINT32 dt_prec)
{
    UINT32 dt_bytes;
    DT_PRECISION_BYTES(dt_prec);
    return ALLOC_UNALIGN_UNINIT(rad_size * dt_bytes);
}

VOID destroy_solution(aoclfftz_solution_t *sol)
{
    while (sol != NULL)
    {
        FREE_ALLOCATED_MEM(sol->solver);
        FREE_ALLOCATED_MEM(sol->decomp_scheme);
        FREE_ALLOCATED_MEM(sol->strides);
        FREE_ALLOCATED_MEM(sol->twiddle->TW);
        FREE_ALLOCATED_MEM(sol->twiddle);
        sol = sol->next_sol;
    }
    return;
}

VOID destroy_selector(aoclfftz_selector_t *sel)
{
    destroy_solution(sel->solution);
    FREE_ALLOCATED_MEM(sel->cost_analysis);
    return;
}
