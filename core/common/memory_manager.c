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
    aoclfftz_decomp_scheme_t *decomp_scheme = NULL;

    ALLOC_ALIGN_UNINIT(decomp_scheme, aoclfftz_decomp_scheme_t,
                       sizeof(aoclfftz_decomp_scheme_t));
    if (decomp_scheme)
    {
        ALLOC_ALIGN_UNINIT(decomp_scheme->dims, aoclfftz_dim_t_64_,
                           dim_rank * sizeof(aoclfftz_dim_t_64_));
        ALLOC_ALIGN_UNINIT(decomp_scheme->vecs, aoclfftz_dim_t_64_,
                           vec_rank * sizeof(aoclfftz_dim_t_64_));
        ALLOC_ALIGN_UNINIT(decomp_scheme->cntrl_params, aoclfftz_cntrl_params_t,
                           sizeof(aoclfftz_cntrl_params_t));
        ALLOC_ALIGN_UNINIT(decomp_scheme->pthr_fft, aoclfftz_smp_pfft_t,
                           sizeof(aoclfftz_smp_pfft_t));
        if (decomp_scheme->dims == NULL || decomp_scheme->vecs == NULL ||
            decomp_scheme->cntrl_params == NULL ||
            decomp_scheme->pthr_fft == NULL)
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
        sol->solver->kernel_c2c = NULL;
        sol->solver->kernel_r2hc = NULL;
        sol->solver->kernel_r2hcf = NULL;
        sol->solver->destroy_solver = NULL;
        sol->decomp_scheme = alloc_decomp_scheme(vec_rank, dim_rank);
        ALLOC_ALIGN_INIT(sol->strides, aoclfftz_strides_t,
                         sizeof(aoclfftz_strides_t));
        ALLOC_ALIGN_INIT(sol->strides_c2c, aoclfftz_strides_t,
                         sizeof(aoclfftz_strides_t));
        ALLOC_ALIGN_INIT(sol->strides_r2hc, aoclfftz_strides_t,
                         sizeof(aoclfftz_strides_t));
        ALLOC_ALIGN_INIT(sol->strides_r2hcf, aoclfftz_strides_t,
                         sizeof(aoclfftz_strides_t));
        ALLOC_ALIGN_UNINIT(sol->twiddle, aoclfftz_twiddle_t,
                           sizeof(aoclfftz_twiddle_t));
        ALLOC_ALIGN_UNINIT(sol->bluestein, aoclfftz_bluestein_t,
                           sizeof(aoclfftz_bluestein_t));
        ALLOC_ALIGN_UNINIT(sol->buffered, aoclfftz_buffered_t,
                           sizeof(aoclfftz_buffered_t));
        ALLOC_ALIGN_INIT(sol->transpose, aoclfftz_transpose_t,
                         sizeof(aoclfftz_transpose_t));
        ALLOC_ALIGN_INIT(sol->transpose->aux_mem, aoclfftz_transpose_aux_mem_t,
                         sizeof(aoclfftz_transpose_aux_mem_t));
        sol->next_sol = NULL;
        sol->nd_sol = NULL;
        if (sol->solver == NULL || sol->decomp_scheme == NULL ||
            sol->strides == NULL || sol->strides_c2c == NULL ||
            sol->strides_r2hc == NULL || sol->strides_r2hcf == NULL ||
            sol->bluestein == NULL || sol->buffered == NULL ||
            sol->twiddle == NULL)
        {
            FREE_ALIGN_ALLOCATED_MEM(sol->solver);
            destroy_decomp_scheme(sol->decomp_scheme);
            FREE_ALIGN_ALLOCATED_MEM(sol->strides);
            FREE_ALIGN_ALLOCATED_MEM(sol->strides_c2c);
            FREE_ALIGN_ALLOCATED_MEM(sol->strides_r2hc);
            FREE_ALIGN_ALLOCATED_MEM(sol->strides_r2hcf);
            destroy_bluestein(sol->bluestein);
            FREE_ALIGN_ALLOCATED_MEM(sol->buffered);
            FREE_ALIGN_ALLOCATED_MEM(sol->twiddle);
            FREE_ALIGN_ALLOCATED_MEM(sol);
            return NULL;
        }
        sol->strides->in_strides = NULL;
        sol->strides->out_strides = NULL;
        sol->strides_c2c->in_strides = NULL;
        sol->strides_c2c->out_strides = NULL;
        sol->strides_r2hc->in_strides = NULL;
        sol->strides_r2hc->out_strides = NULL;
        sol->strides_r2hcf->in_strides = NULL;
        sol->strides_r2hcf->out_strides = NULL;
        sol->twiddle->TW = NULL;
        sol->bluestein->B = NULL;
        sol->bluestein->B_out = NULL;
        sol->bluestein->in = NULL;
        sol->bluestein->out = NULL;
        sol->bluestein->is_B_out_valid = 0;
        sol->buffered->aux_buffer_1 = NULL;
        sol->buffered->aux_buffer_2 = NULL;
        sol->buffered->out_ptr = NULL;
        sol->transpose->row_info = (aoclfftz_dim_t_64_){0};
        sol->transpose->col_info = (aoclfftz_dim_t_64_){0};
        sol->transpose->aux_mem->size = 0;
        sol->transpose->aux_mem->data = NULL;
        sol->scratch_space = NULL;
        sol->solver->batches[C2C_KERNEL] = 0;
        sol->solver->batches[R2HC_KERNEL] = 0;
        sol->solver->batches[R2HCF_KERNEL] = 0;
        return sol;
    }
    else
    {
        return NULL;
    }
}

// Allocates a new scratch_space iff the argument passed is NULL.
// Otherwise sets the selector's scratch_space to the passed argument.
aoclfftz_selector_t *alloc_selector(INT32 vec_rank, INT32 dim_rank,
                                    VOID *scratch_space)
{
    aoclfftz_selector_t *selector = NULL;

    ALLOC_ALIGN_UNINIT(selector, aoclfftz_selector_t,
                       sizeof(aoclfftz_selector_t));
    if (selector)
    {
        selector->scratch_space = NULL;

        if (scratch_space == NULL)
        {
            // Note: this allocation could fail, but that is ok. All functions
            //       that use the scratch buffer are expected to check if the
            //       buffer is valid (not-null)
            ALLOC_ALIGN_UNINIT(selector->scratch_space, UINT8,
                               scratch_space_capacity);

        }
        else
        {
            selector->scratch_space = scratch_space;
        }
        selector->solution = alloc_solution(vec_rank, dim_rank);
        ALLOC_ALIGN_UNINIT(selector->cost_analysis, cost_analysis_t,
                           sizeof(cost_analysis_t));
        if (selector->solution == NULL || selector->cost_analysis == NULL)
        {
            destroy_selector(selector);
            return NULL;
        }
        selector->solution->scratch_space = selector->scratch_space;
        selector->cost_analysis->ops = 0;
        selector->cost_analysis->time = 0;
        return selector;
    }
    else
    {
        return NULL;
    }
}

INT32 alloc_bluestein_buffers(aoclfftz_bluestein_t *bluestein, INTP size)
{
    // Allocate bluestein sequence buffers
    ALLOC_ALIGN_UNINIT(bluestein->B, VOID, size);
    if (bluestein->B == NULL)
    {
        return AOCLFFTZ_MEMORY_FAILURE;
    }
    ALLOC_ALIGN_INIT(bluestein->B_out, VOID, size);
    if (bluestein->B_out == NULL)
    {
        return AOCLFFTZ_MEMORY_FAILURE;
    }

    // Allocate bluestein in and out buffers
    ALLOC_ALIGN_UNINIT(bluestein->in, VOID, size);
    if (bluestein->in == NULL)
    {
        return AOCLFFTZ_MEMORY_FAILURE;
    }
    ALLOC_ALIGN_UNINIT(bluestein->out, VOID, size);
    if (bluestein->out == NULL)
    {
        return AOCLFFTZ_MEMORY_FAILURE;
    }

    return AOCLFFTZ_SUCCESS;
}

#if IN_MEMORY_TWIDDLE_FACTORS == 1
VOID *alloc_twiddle_for_solution(UINT32 rad_size, UINT8 dt_prec)
{
    UINT32 dt_bytes = DT_PRECISION_BYTES(dt_prec);
    VOID *buffer = NULL;
    ALLOC_ALIGN_UNINIT(buffer, VOID, rad_size * dt_bytes);
    return buffer;
}
#endif

VOID destroy_decomp_scheme(aoclfftz_decomp_scheme_t *decomp_scheme)
{
    if (decomp_scheme != NULL)
    {
        FREE_ALIGN_ALLOCATED_MEM(decomp_scheme->dims);
        FREE_ALIGN_ALLOCATED_MEM(decomp_scheme->vecs);
        FREE_ALIGN_ALLOCATED_MEM(decomp_scheme->cntrl_params);
        FREE_ALIGN_ALLOCATED_MEM(decomp_scheme->pthr_fft);
        FREE_ALIGN_ALLOCATED_MEM(decomp_scheme);
    }
    return;
}

VOID destroy_transpose(aoclfftz_transpose_t *transpose)
{
    if (transpose)
    {
        if (transpose->aux_mem)
        {
            FREE_ALIGN_ALLOCATED_MEM(transpose->aux_mem->data);
            FREE_ALIGN_ALLOCATED_MEM(transpose->aux_mem);
        }
        FREE_ALIGN_ALLOCATED_MEM(transpose);
    }
}

VOID destroy_solution(aoclfftz_solution_t *sol)
{
    aoclfftz_solution_t *cur_sol = NULL;
    while (sol != NULL)
    {
        cur_sol = sol;
        sol = sol->next_sol;
        INT32 solver_type = cur_sol->solver->solver_type;
        FREE_ALIGN_ALLOCATED_MEM(cur_sol->solver);
        destroy_decomp_scheme(cur_sol->decomp_scheme);
        FREE_ALIGN_ALLOCATED_MEM(cur_sol->strides->in_strides);
        FREE_ALIGN_ALLOCATED_MEM(cur_sol->strides->out_strides);
        FREE_ALIGN_ALLOCATED_MEM(cur_sol->strides);
        FREE_ALIGN_ALLOCATED_MEM(cur_sol->strides_c2c->in_strides);
        FREE_ALIGN_ALLOCATED_MEM(cur_sol->strides_c2c->out_strides);
        FREE_ALIGN_ALLOCATED_MEM(cur_sol->strides_c2c);
        FREE_ALIGN_ALLOCATED_MEM(cur_sol->strides_r2hc->in_strides);
        FREE_ALIGN_ALLOCATED_MEM(cur_sol->strides_r2hc->out_strides);
        FREE_ALIGN_ALLOCATED_MEM(cur_sol->strides_r2hc);
        FREE_ALIGN_ALLOCATED_MEM(cur_sol->strides_r2hcf->in_strides);
        FREE_ALIGN_ALLOCATED_MEM(cur_sol->strides_r2hcf->out_strides);
        FREE_ALIGN_ALLOCATED_MEM(cur_sol->strides_r2hcf);
        FREE_ALIGN_ALLOCATED_MEM(cur_sol->twiddle->TW);
        FREE_ALIGN_ALLOCATED_MEM(cur_sol->twiddle);
        destroy_bluestein(cur_sol->bluestein);
        destroy_transpose(cur_sol->transpose);
        destroy_solution(cur_sol->nd_sol);
        // Buffered solver will create aux_buffers and the same address will be
        // used in other solvers.
        // So free the aux_buffers only for buffered solver.
        //
        // Clearing these buffers will happen only once (which will be from
        // destroy_handle).
        if (solver_type == SOLVER_BUFFERED)
        {
            FREE_ALIGN_ALLOCATED_MEM(cur_sol->buffered->aux_buffer_1);
            FREE_ALIGN_ALLOCATED_MEM(cur_sol->buffered->aux_buffer_2);
        }
        FREE_ALIGN_ALLOCATED_MEM(cur_sol->buffered);
        FREE_ALIGN_ALLOCATED_MEM(cur_sol);
    }
    return;
}

// Note: Since the use case for this function is primarily to free "temporary"
//       selectors, and since temporary selectors are expected to be allocated
//       using `alloc_selector_without_scratch_space`, this function does not
//       free the scratch space.
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
        FREE_ALIGN_ALLOCATED_MEM(sel->scratch_space);
        destroy_solution(sel->solution);
        destroy_selector_without_solution(sel);
    }
    return;
}

VOID destroy_selector_without_scratch_space(aoclfftz_selector_t *sel)
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
