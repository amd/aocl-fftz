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

/** @file selector.h
 *
 *  @brief Functions and data structures of the selector module.
 *
 *  This file contains the functions and data structures that are used to
 *  select a solution of kernels for the given input problem description.
 *
 *  @author S. Biplab Raut
 */

#ifndef AOCLFFTZ_SELECTOR_H
#define AOCLFFTZ_SELECTOR_H

#include "core/solvers/solver.h"
#include "core/executor.h"

#define AOCLFFTZ_FIXED_SELECTOR_MODE 0 // Fixed decision logic
#define AOCLFFTZ_AUTO_SELECTOR_MODE 1  // Auto tuner

// Error return codes related to selector
// Add more codes at the top
typedef enum
{
    SELECTOR_FAILURE = -1,
    SELECTOR_SUCCESS // Successful operation
} aoclfftz_selector_status;

// Note: The choice of size (16 KB) is completely arbitrary and can be
//       experimented with.
static const INTP scratch_space_capacity = 16 * 1024; // 16 KB

// Selector data structure that is used to hold the solution and cost analysis
// at each decomposition level for the associated sub-problem
typedef struct aoclfftz_selector
{
    aoclfftz_solution_t *solution;
    execute_ execute;
    cost_analysis_t *cost_analysis;

    // A global buffer to help with transposition of twiddle multiplied elements
    void* scratch_space;
} aoclfftz_selector_t;

// macro functions
#ifdef MULTI_THREADING
// if dynamic_load_model is set we are allowing the library to take all the
// available threads in the system and use them efficiently
#define INIT_THREADS(sel_obj, problem)                                         \
{                                                                              \
    if (problem->pthr_fft.dynamic_load_model)                                  \
    {                                                                          \
        UINT32 procs = omp_get_num_procs();                                    \
        sel_obj->solution->decomp_scheme->thread_info->pthr_fft->num_threads = \
            procs;                                                             \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        sel_obj->solution->decomp_scheme->thread_info->pthr_fft->num_threads = \
            problem->pthr_fft.num_threads;                                     \
    }                                                                          \
    /*Enable nested threading in openmp to support multi-level threads*/       \
    omp_set_nested(1);                                                         \
}
#else
#define INIT_THREADS(sel_obj, problem)                                         \
{                                                                              \
    sel_obj->solution->decomp_scheme->thread_info->pthr_fft->num_threads =     \
        problem->pthr_fft.num_threads;                                         \
}
#endif

#define INIT_DECOMP_SCHEME(sel_obj, problem, dim_rank)                         \
{                                                                              \
    sel_obj->solution->decomp_scheme->vec_rank = problem->vec_rank;            \
    sel_obj->solution->decomp_scheme->dim_rank = dim_rank;                     \
    UINT32 cnt, idx = 0;                                                       \
    for (cnt = 0; cnt < problem->dim_rank; cnt++)                              \
    {                                                                          \
        if (problem->dims[cnt].n != 1)                                         \
        {                                                                      \
            sel_obj->solution->decomp_scheme->dims[idx].n =                    \
                problem->dims[cnt].n;                                          \
            sel_obj->solution->decomp_scheme->dims[idx].in_stride =            \
                problem->dims[cnt].in_stride;                                  \
            sel_obj->solution->decomp_scheme->dims[idx].out_stride =           \
                problem->dims[cnt].out_stride;                                 \
            idx++;                                                             \
        }                                                                      \
    }                                                                          \
    /* Gets Executed in scenario where the shrinked dim_rank is one and        \
       the problem size is also one.                                           \
       Example: 1x1x1 or 1 */                                                  \
    if (idx == 0)                                                              \
    {                                                                          \
        sel_obj->solution->decomp_scheme->dims[0].n = problem->dims[0].n;      \
        sel_obj->solution->decomp_scheme->dims[0].in_stride =                  \
            problem->dims[0].in_stride;                                        \
        sel_obj->solution->decomp_scheme->dims[0].out_stride =                 \
            problem->dims[0].out_stride;                                       \
    }                                                                          \
    for (cnt = 0; cnt < problem->vec_rank; cnt++)                              \
    {                                                                          \
        sel_obj->solution->decomp_scheme->vecs[cnt].n =                        \
            problem->vecs[cnt].n;                                              \
        sel_obj->solution->decomp_scheme->vecs[cnt].in_stride =                \
            problem->vecs[cnt].in_stride;                                      \
        sel_obj->solution->decomp_scheme->vecs[cnt].out_stride =               \
            problem->vecs[cnt].out_stride;                                     \
    }                                                                          \
    if (FFT_DIR(problem->flags) == FORWARD_FFT_DIR || IS_REAL(problem->flags)) \
    {                                                                          \
        sel_obj->solution->decomp_scheme->in_real = problem->in;               \
        sel_obj->solution->decomp_scheme->in_imag = problem->in + 1;           \
        sel_obj->solution->decomp_scheme->out_real = problem->out;             \
        sel_obj->solution->decomp_scheme->out_imag = problem->out + 1;         \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        sel_obj->solution->decomp_scheme->in_real = problem->in + 1;           \
        sel_obj->solution->decomp_scheme->in_imag = problem->in;               \
        sel_obj->solution->decomp_scheme->out_real = problem->out + 1;         \
        sel_obj->solution->decomp_scheme->out_imag = problem->out;             \
    }                                                                          \
    sel_obj->solution->decomp_scheme->cntrl_params->opt_level =                \
        problem->cntrl_params.opt_level;                                       \
    sel_obj->solution->decomp_scheme->cntrl_params->opt_off =                  \
        problem->cntrl_params.opt_off;                                         \
    sel_obj->solution->decomp_scheme->cntrl_params->logger_mode =              \
        problem->cntrl_params.logger_mode;                                     \
    sel_obj->solution->decomp_scheme->cntrl_params->measure_stats =            \
        problem->cntrl_params.measure_stats;                                   \
    sel_obj->solution->decomp_scheme->thread_info->pthr_fft->                  \
        dynamic_load_model = problem->pthr_fft.dynamic_load_model;             \
    INIT_THREADS(sel_obj, problem)                                             \
    sel_obj->solution->decomp_scheme->thread_info->avl_threads =               \
    sel_obj->solution->decomp_scheme->thread_info->pthr_fft->num_threads;      \
    sel_obj->solution->decomp_scheme->thread_info->n_threads = 1;              \
    sel_obj->solution->decomp_scheme->flags = problem->flags;                  \
}

/**
 * @brief Swap the CT and direct solution nodes for the iterative execution
 *
 * Before swap: CT -> Direct -> CT -> Direct -> CT -> Direct -> Direct
 * After swap : Direct -> CT -> Direct -> CT -> Direct -> CT -> Direct
 *
 */
#define SWAP_CT_SOLUTIONS(sel)                                                 \
{                                                                              \
    aoclfftz_solution_t *prev = NULL;                                          \
    aoclfftz_solution_t *curr = sel->solution;                                 \
    aoclfftz_solution_t *next = curr->next_sol;                                \
    if (sel->solution->next_sol != NULL) {                                     \
      /* swap first CT node */                                                 \
      if (sel->solution->solver->solver_type == SOLVER_CT &&                   \
          sel->solution->next_sol->solver->solver_type == SOLVER_DIRECT) {     \
        sel->solution = curr->next_sol;                                        \
        curr->next_sol = sel->solution->next_sol;                              \
        sel->solution->next_sol = curr;                                        \
      }                                                                        \
      /* swap remaining CT nodes */                                            \
      prev = curr;                                                             \
      curr = curr->next_sol;                                                   \
      while (curr && curr->next_sol) {                                         \
        next = curr->next_sol;                                                 \
        if (curr->solver->solver_type == SOLVER_CT &&                          \
            next->solver->solver_type == SOLVER_DIRECT) {                      \
          prev->next_sol = next;                                               \
          curr->next_sol = next->next_sol;                                     \
          next->next_sol = curr;                                               \
        }                                                                      \
        prev = curr;                                                           \
        curr = curr->next_sol;                                                 \
      }                                                                        \
    }                                                                          \
}

// Few additional steps are required for RealFFT problems before and after
// the setup stages.
// FIXIT: These additional initialization steps will only work for 1D problems.
#define PREPARE_AND_SETUP_DFT(sel_obj, kernels_table, ret)                     \
{                                                                              \
    sel_obj->execute = register_execute_dft();                                 \
    if (IS_REAL(sel_obj->solution->decomp_scheme->flags))                      \
    {                                                                          \
        aoclfftz_realhelper_t *realhelper;                                     \
        ALLOC_ALIGN_UNINIT(realhelper, aoclfftz_realhelper_t,                  \
            sizeof(aoclfftz_realhelper_t));                                    \
        realhelper->stage = 0;                                                 \
        realhelper->is_CT = 0;                                                 \
        realhelper->is_buffered_invoked = 0;                                   \
        realhelper->problem_size = sel_obj->solution->decomp_scheme->dims[0].n;\
        if (FFT_DIR(sel_obj->solution->decomp_scheme->flags) ==                \
            FORWARD_FFT_DIR)                                                   \
        {                                                                      \
            realhelper->p = 1;                                                 \
            realhelper->q = realhelper->problem_size;                          \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            realhelper->p = realhelper->problem_size;                          \
            realhelper->q = 1;                                                 \
        }                                                                      \
        ret = setup_rdft_(sel_obj, (kernel_t *)kernels_table, realhelper);     \
        SWAP_CT_SOLUTIONS(sel_obj);                                            \
        setup_twiddle_buffer_real(sel_obj->solution);                          \
        FREE_ALIGN_ALLOCATED_MEM(realhelper);                                  \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        ret = setup_dft_(sel_obj, (kernel_t *)kernels_table);                  \
        setup_twiddle_buffer_complex(sel_obj->solution);                       \
    }                                                                          \
}

#define COPY_SOLUTION_OBJ(to_sol_obj, from_sol_obj)                            \
{                                                                              \
    to_sol_obj->solver->solver_type = from_sol_obj->solver->solver_type;       \
    to_sol_obj->solver->execute_solver = from_sol_obj->solver->execute_solver; \
    to_sol_obj->solver->destroy_solver = from_sol_obj->solver->destroy_solver; \
    to_sol_obj->solver->kernel_c2c->kfft =                                     \
        from_sol_obj->solver->kernel_c2c->kfft;                                \
    to_sol_obj->solver->kernel_c2c->sets =                                     \
        from_sol_obj->solver->kernel_c2c->sets;                                \
    to_sol_obj->solver->kernel_r2hc->kfft =                                    \
        from_sol_obj->solver->kernel_r2hc->kfft;                               \
    to_sol_obj->solver->kernel_r2hc->sets =                                    \
        from_sol_obj->solver->kernel_r2hc->sets;                               \
    to_sol_obj->solver->kernel_r2hcf->kfft =                                   \
        from_sol_obj->solver->kernel_r2hcf->kfft;                              \
    to_sol_obj->solver->kernel_r2hcf->sets =                                   \
        from_sol_obj->solver->kernel_r2hcf->sets;                              \
    to_sol_obj->solver->batches[C2C_KERNEL] =                                  \
        from_sol_obj->solver->batches[C2C_KERNEL];                             \
    to_sol_obj->solver->batches[R2HC_KERNEL] =                                 \
        from_sol_obj->solver->batches[R2HC_KERNEL];                            \
    to_sol_obj->solver->batches[R2HCF_KERNEL] =                                \
        from_sol_obj->solver->batches[R2HCF_KERNEL];                           \
    to_sol_obj->decomp_scheme->vec_rank =                                      \
        from_sol_obj->decomp_scheme->vec_rank;                                 \
    to_sol_obj->decomp_scheme->dim_rank =                                      \
        from_sol_obj->decomp_scheme->dim_rank;                                 \
    INT32 cnt;                                                                 \
    for (cnt = 0; cnt < to_sol_obj->decomp_scheme->dim_rank; cnt++)            \
    {                                                                          \
        to_sol_obj->decomp_scheme->dims[cnt].n =                               \
            from_sol_obj->decomp_scheme->dims[cnt].n;                          \
        to_sol_obj->decomp_scheme->dims[cnt].in_stride =                       \
            from_sol_obj->decomp_scheme->dims[cnt].in_stride;                  \
        to_sol_obj->decomp_scheme->dims[cnt].out_stride =                      \
            from_sol_obj->decomp_scheme->dims[cnt].out_stride;                 \
    }                                                                          \
    for (cnt = 0; cnt < to_sol_obj->decomp_scheme->vec_rank; cnt++)            \
    {                                                                          \
        to_sol_obj->decomp_scheme->vecs[cnt].n =                               \
            from_sol_obj->decomp_scheme->vecs[cnt].n;                          \
        to_sol_obj->decomp_scheme->vecs[cnt].in_stride =                       \
            from_sol_obj->decomp_scheme->vecs[cnt].in_stride;                  \
        to_sol_obj->decomp_scheme->vecs[cnt].out_stride =                      \
            from_sol_obj->decomp_scheme->vecs[cnt].out_stride;                 \
    }                                                                          \
    to_sol_obj->decomp_scheme->in_real = from_sol_obj->decomp_scheme->in_real; \
    to_sol_obj->decomp_scheme->in_imag = from_sol_obj->decomp_scheme->in_imag; \
    to_sol_obj->decomp_scheme->out_real =                                      \
        from_sol_obj->decomp_scheme->out_real;                                 \
    to_sol_obj->decomp_scheme->out_imag =                                      \
        from_sol_obj->decomp_scheme->out_imag;                                 \
    to_sol_obj->decomp_scheme->cntrl_params->opt_level =                       \
        from_sol_obj->decomp_scheme->cntrl_params->opt_level;                  \
    to_sol_obj->decomp_scheme->cntrl_params->opt_off =                         \
        from_sol_obj->decomp_scheme->cntrl_params->opt_off;                    \
    to_sol_obj->decomp_scheme->cntrl_params->logger_mode =                     \
        from_sol_obj->decomp_scheme->cntrl_params->logger_mode;                \
    to_sol_obj->decomp_scheme->cntrl_params->measure_stats =                   \
        from_sol_obj->decomp_scheme->cntrl_params->measure_stats;              \
    to_sol_obj->decomp_scheme->thread_info->pthr_fft->num_threads =            \
        from_sol_obj->decomp_scheme->thread_info->pthr_fft->num_threads;       \
    to_sol_obj->decomp_scheme->thread_info->pthr_fft->dynamic_load_model =     \
        from_sol_obj->decomp_scheme->thread_info->pthr_fft->dynamic_load_model;\
    to_sol_obj->decomp_scheme->thread_info->avl_threads =                      \
        from_sol_obj->decomp_scheme->thread_info->avl_threads;                 \
    to_sol_obj->decomp_scheme->thread_info->n_threads =                        \
        from_sol_obj->decomp_scheme->thread_info->n_threads;                   \
    to_sol_obj->decomp_scheme->flags = from_sol_obj->decomp_scheme->flags;     \
    to_sol_obj->twiddle->TW = from_sol_obj->twiddle->TW;                       \
    to_sol_obj->bluestein->B = from_sol_obj->bluestein->B;                     \
    to_sol_obj->bluestein->B_out = from_sol_obj->bluestein->B_out;             \
    to_sol_obj->bluestein->in = from_sol_obj->bluestein->in;                   \
    to_sol_obj->bluestein->out = from_sol_obj->bluestein->out;                 \
    to_sol_obj->bluestein->is_B_out_valid =                                    \
        from_sol_obj->bluestein->is_B_out_valid;                               \
    to_sol_obj->buffered->aux_buffer_1 = from_sol_obj->buffered->aux_buffer_1; \
    to_sol_obj->buffered->aux_buffer_2 = from_sol_obj->buffered->aux_buffer_2; \
    to_sol_obj->buffered->out_ptr = from_sol_obj->buffered->out_ptr;           \
    if (from_sol_obj->transpose && to_sol_obj->transpose)                      \
    {                                                                          \
        to_sol_obj->transpose->row_info = from_sol_obj->transpose->row_info;   \
        to_sol_obj->transpose->col_info = from_sol_obj->transpose->col_info;   \
        to_sol_obj->transpose->kernel = from_sol_obj->transpose->kernel;       \
        if (from_sol_obj->transpose->aux_mem &&                                \
            from_sol_obj->transpose->aux_mem->data &&                          \
            from_sol_obj->transpose->aux_mem->size > 0)                        \
        {                                                                      \
            if (!to_sol_obj->transpose->aux_mem->data)                         \
            {                                                                  \
                ALLOC_ALIGN_INIT(to_sol_obj->transpose->aux_mem->data, UINT8,  \
                                 from_sol_obj->transpose->aux_mem->size);      \
            }                                                                  \
            else                                                               \
            {                                                                  \
                FREE_ALIGN_ALLOCATED_MEM(to_sol_obj->transpose->aux_mem->data) \
                ALLOC_ALIGN_INIT(to_sol_obj->transpose->aux_mem->data, UINT8,  \
                                 from_sol_obj->transpose->aux_mem->size);      \
            }                                                                  \
            memcpy(to_sol_obj->transpose->aux_mem->data,                       \
                   from_sol_obj->transpose->aux_mem->data,                     \
                   from_sol_obj->transpose->aux_mem->size);                    \
        }                                                                      \
        to_sol_obj->transpose->aux_mem->size =                                 \
            from_sol_obj->transpose->aux_mem->size;                            \
    }                                                                          \
    to_sol_obj->next_sol = from_sol_obj->next_sol;                             \
}

// maps both in & out pointers to out pointer
// incase of out-of-place problems, except the first DFT, other DFTs happen
// in-place ie., in the output buffer.
#define COPY_SOLUTION_OBJ_OUT_P(to_sol_obj, from_sol_obj)                      \
{                                                                              \
    COPY_SOLUTION_OBJ(to_sol_obj, from_sol_obj)                                \
    UINT32 cnt;                                                                \
    for (cnt = 0; cnt < to_sol_obj->decomp_scheme->dim_rank; cnt++)            \
    {                                                                          \
        to_sol_obj->decomp_scheme->dims[cnt].n =                               \
            from_sol_obj->decomp_scheme->dims[cnt].n;                          \
        to_sol_obj->decomp_scheme->dims[cnt].in_stride =                       \
            from_sol_obj->decomp_scheme->dims[cnt].out_stride;                 \
        to_sol_obj->decomp_scheme->dims[cnt].out_stride =                      \
            from_sol_obj->decomp_scheme->dims[cnt].out_stride;                 \
    }                                                                          \
    for (cnt = 0; cnt < to_sol_obj->decomp_scheme->vec_rank; cnt++)            \
    {                                                                          \
        to_sol_obj->decomp_scheme->vecs[cnt].n =                               \
            from_sol_obj->decomp_scheme->vecs[cnt].n;                          \
        to_sol_obj->decomp_scheme->vecs[cnt].in_stride =                       \
            from_sol_obj->decomp_scheme->vecs[cnt].out_stride;                 \
        to_sol_obj->decomp_scheme->vecs[cnt].out_stride =                      \
            from_sol_obj->decomp_scheme->vecs[cnt].out_stride;                 \
    }                                                                          \
    to_sol_obj->decomp_scheme->in_real =                                       \
        from_sol_obj->decomp_scheme->out_real;                                 \
    to_sol_obj->decomp_scheme->in_imag =                                       \
        from_sol_obj->decomp_scheme->out_imag;                                 \
    to_sol_obj->decomp_scheme->out_real =                                      \
        from_sol_obj->decomp_scheme->out_real;                                 \
    to_sol_obj->decomp_scheme->out_imag =                                      \
        from_sol_obj->decomp_scheme->out_imag;                                 \
}

// copy all contents except dims & vecs
// necessary in ND setup where dim_rank & vec_rank will differ for the
// sub-problem
#define COPY_SOLUTION_OBJ_WO_DIMS(to_sol_obj, from_sol_obj)                    \
{                                                                              \
    to_sol_obj->solver->solver_type = from_sol_obj->solver->solver_type;       \
    to_sol_obj->solver->execute_solver =                                       \
        from_sol_obj->solver->execute_solver;                                  \
    to_sol_obj->solver->destroy_solver =                                       \
        from_sol_obj->solver->destroy_solver;                                  \
    to_sol_obj->solver->kernel_c2c->kfft =                                     \
        from_sol_obj->solver->kernel_c2c->kfft;                                \
    to_sol_obj->solver->kernel_c2c->sets =                                     \
        from_sol_obj->solver->kernel_c2c->sets;                                \
    to_sol_obj->solver->kernel_r2hc->kfft =                                    \
        from_sol_obj->solver->kernel_r2hc->kfft;                               \
    to_sol_obj->solver->kernel_r2hc->sets =                                    \
        from_sol_obj->solver->kernel_r2hc->sets;                               \
    to_sol_obj->solver->kernel_r2hcf->kfft =                                   \
        from_sol_obj->solver->kernel_r2hcf->kfft;                              \
    to_sol_obj->solver->kernel_r2hcf->sets =                                   \
        from_sol_obj->solver->kernel_r2hcf->sets;                              \
    to_sol_obj->solver->batches[C2C_KERNEL] =                                  \
        from_sol_obj->solver->batches[C2C_KERNEL];                             \
    to_sol_obj->solver->batches[R2HC_KERNEL] =                                 \
        from_sol_obj->solver->batches[R2HC_KERNEL];                            \
    to_sol_obj->solver->batches[R2HCF_KERNEL] =                                \
        from_sol_obj->solver->batches[R2HCF_KERNEL];                           \
    to_sol_obj->decomp_scheme->in_real =                                       \
        from_sol_obj->decomp_scheme->in_real;                                  \
    to_sol_obj->decomp_scheme->in_imag =                                       \
        from_sol_obj->decomp_scheme->in_imag;                                  \
    to_sol_obj->decomp_scheme->out_real =                                      \
        from_sol_obj->decomp_scheme->out_real;                                 \
    to_sol_obj->decomp_scheme->out_imag =                                      \
        from_sol_obj->decomp_scheme->out_imag;                                 \
    to_sol_obj->decomp_scheme->cntrl_params->opt_level =                       \
        from_sol_obj->decomp_scheme->cntrl_params->opt_level;                  \
    to_sol_obj->decomp_scheme->cntrl_params->opt_off =                         \
        from_sol_obj->decomp_scheme->cntrl_params->opt_off;                    \
    to_sol_obj->decomp_scheme->cntrl_params->logger_mode =                     \
        from_sol_obj->decomp_scheme->cntrl_params->logger_mode;                \
    to_sol_obj->decomp_scheme->cntrl_params->measure_stats =                   \
        from_sol_obj->decomp_scheme->cntrl_params->measure_stats;              \
    to_sol_obj->decomp_scheme->thread_info->pthr_fft->num_threads =            \
        from_sol_obj->decomp_scheme->thread_info->pthr_fft->num_threads;       \
    to_sol_obj->decomp_scheme->thread_info->pthr_fft->dynamic_load_model =     \
        from_sol_obj->decomp_scheme->thread_info->pthr_fft->dynamic_load_model;\
    to_sol_obj->decomp_scheme->thread_info->avl_threads =                      \
        from_sol_obj->decomp_scheme->thread_info->avl_threads;                 \
    to_sol_obj->decomp_scheme->thread_info->n_threads =                        \
        from_sol_obj->decomp_scheme->thread_info->n_threads;                   \
    to_sol_obj->decomp_scheme->flags = from_sol_obj->decomp_scheme->flags;     \
    to_sol_obj->twiddle->TW = from_sol_obj->twiddle->TW;                       \
    to_sol_obj->bluestein->B = from_sol_obj->bluestein->B;                     \
    to_sol_obj->bluestein->B_out = from_sol_obj->bluestein->B_out;             \
    to_sol_obj->bluestein->in = from_sol_obj->bluestein->in;                   \
    to_sol_obj->bluestein->out = from_sol_obj->bluestein->out;                 \
    to_sol_obj->bluestein->is_B_out_valid =                                    \
        from_sol_obj->bluestein->is_B_out_valid;                               \
    to_sol_obj->buffered->aux_buffer_1 = from_sol_obj->buffered->aux_buffer_1; \
    to_sol_obj->buffered->aux_buffer_2 = from_sol_obj->buffered->aux_buffer_2; \
    to_sol_obj->buffered->out_ptr = from_sol_obj->buffered->out_ptr;           \
    to_sol_obj->next_sol = from_sol_obj->next_sol;                             \
}

// Copy strides from one solution to another
#define COPY_STRIDES(to_sol_obj, from_sol_obj)                                 \
{                                                                              \
    if (from_sol_obj->strides->in_strides != NULL)                             \
    {                                                                          \
        FREE_ALIGN_ALLOCATED_MEM(to_sol_obj->strides->in_strides);             \
        ALLOC_ALIGN_UNINIT(to_sol_obj->strides->in_strides, INTP,              \
                            from_sol_obj->decomp_scheme->dims[0].n *           \
                                sizeof(INTP));                                 \
        memcpy(to_sol_obj->strides->in_strides,                                \
                from_sol_obj->strides->in_strides,                             \
                from_sol_obj->decomp_scheme->dims[0].n * sizeof(INTP));        \
    }                                                                          \
    if (from_sol_obj->strides->out_strides != NULL)                            \
    {                                                                          \
        FREE_ALIGN_ALLOCATED_MEM(to_sol_obj->strides->out_strides);            \
        ALLOC_ALIGN_UNINIT(to_sol_obj->strides->out_strides, INTP,             \
                            from_sol_obj->decomp_scheme->dims[0].n *           \
                                sizeof(INTP));                                 \
        memcpy(to_sol_obj->strides->out_strides,                               \
                from_sol_obj->strides->out_strides,                            \
                from_sol_obj->decomp_scheme->dims[0].n * sizeof(INTP));        \
    }                                                                          \
    to_sol_obj->strides->v_in_stride = from_sol_obj->strides->v_in_stride;     \
    to_sol_obj->strides->v_out_stride =                                        \
        from_sol_obj->strides->v_out_stride;                                   \
                                                                               \
    if (from_sol_obj->solver->batches[C2C_KERNEL] != 0)                        \
    {                                                                          \
        if (from_sol_obj->strides_c2c->in_strides != NULL)                     \
        {                                                                      \
            FREE_ALIGN_ALLOCATED_MEM(to_sol_obj->strides_c2c->in_strides);     \
            ALLOC_ALIGN_UNINIT(to_sol_obj->strides_c2c->in_strides, INTP,      \
                               from_sol_obj->decomp_scheme->dims[0].n *        \
                                   sizeof(INTP));                              \
            memcpy(to_sol_obj->strides_c2c->in_strides,                        \
                   from_sol_obj->strides_c2c->in_strides,                      \
                   from_sol_obj->decomp_scheme->dims[0].n * sizeof(INTP));     \
        }                                                                      \
        if (from_sol_obj->strides_c2c->out_strides != NULL)                    \
        {                                                                      \
            FREE_ALIGN_ALLOCATED_MEM(to_sol_obj->strides_c2c->out_strides);    \
            ALLOC_ALIGN_UNINIT(to_sol_obj->strides_c2c->out_strides, INTP,     \
                               from_sol_obj->decomp_scheme->dims[0].n *        \
                                   sizeof(INTP));                              \
            memcpy(to_sol_obj->strides_c2c->out_strides,                       \
                   from_sol_obj->strides_c2c->out_strides,                     \
                   from_sol_obj->decomp_scheme->dims[0].n * sizeof(INTP));     \
        }                                                                      \
        to_sol_obj->strides_c2c->v_in_stride =                                 \
            from_sol_obj->strides_c2c->v_in_stride;                            \
        to_sol_obj->strides_c2c->v_out_stride =                                \
            from_sol_obj->strides_c2c->v_out_stride;                           \
    }                                                                          \
                                                                               \
    if (from_sol_obj->solver->batches[R2HC_KERNEL] != 0)                       \
    {                                                                          \
        if (from_sol_obj->strides_r2hc->in_strides != NULL)                    \
        {                                                                      \
            FREE_ALIGN_ALLOCATED_MEM(to_sol_obj->strides_r2hc->in_strides);    \
            ALLOC_ALIGN_UNINIT(to_sol_obj->strides_r2hc->in_strides, INTP,     \
                                from_sol_obj->decomp_scheme->dims[0].n *       \
                                    sizeof(INTP));                             \
            memcpy(to_sol_obj->strides_r2hc->in_strides,                       \
                from_sol_obj->strides_r2hc->in_strides,                        \
                from_sol_obj->decomp_scheme->dims[0].n * sizeof(INTP));        \
        }                                                                      \
        if (from_sol_obj->strides_r2hc->out_strides != NULL)                   \
        {                                                                      \
            FREE_ALIGN_ALLOCATED_MEM(to_sol_obj->strides_r2hc->out_strides);   \
            ALLOC_ALIGN_UNINIT(to_sol_obj->strides_r2hc->out_strides, INTP,    \
                                from_sol_obj->decomp_scheme->dims[0].n *       \
                                    sizeof(INTP));                             \
            memcpy(to_sol_obj->strides_r2hc->out_strides,                      \
                from_sol_obj->strides_r2hc->out_strides,                       \
                from_sol_obj->decomp_scheme->dims[0].n * sizeof(INTP));        \
        }                                                                      \
        to_sol_obj->strides_r2hc->v_in_stride =                                \
            from_sol_obj->strides_r2hc->v_in_stride;                           \
        to_sol_obj->strides_r2hc->v_out_stride =                               \
            from_sol_obj->strides_r2hc->v_out_stride;                          \
    }                                                                          \
                                                                               \
    if (from_sol_obj->solver->batches[R2HCF_KERNEL] != 0)                      \
    {                                                                          \
        if (from_sol_obj->strides_r2hcf->in_strides != NULL)                   \
        {                                                                      \
            FREE_ALIGN_ALLOCATED_MEM(to_sol_obj->strides_r2hcf->in_strides);   \
            ALLOC_ALIGN_UNINIT(to_sol_obj->strides_r2hcf->in_strides, INTP,    \
                                from_sol_obj->decomp_scheme->dims[0].n *       \
                                    2 * sizeof(INTP));                         \
            memcpy(to_sol_obj->strides_r2hcf->in_strides,                      \
                from_sol_obj->strides_r2hcf->in_strides,                       \
                from_sol_obj->decomp_scheme->dims[0].n * 2 * sizeof(INTP));    \
        }                                                                      \
        if (from_sol_obj->strides_r2hcf->out_strides != NULL)                  \
        {                                                                      \
            FREE_ALIGN_ALLOCATED_MEM(to_sol_obj->strides_r2hcf->out_strides);  \
            ALLOC_ALIGN_UNINIT(to_sol_obj->strides_r2hcf->out_strides, INTP,   \
                                from_sol_obj->decomp_scheme->dims[0].n *       \
                                    2 * sizeof(INTP));                         \
            memcpy(to_sol_obj->strides_r2hcf->out_strides,                     \
                from_sol_obj->strides_r2hcf->out_strides,                      \
                from_sol_obj->decomp_scheme->dims[0].n * 2 * sizeof(INTP));    \
        }                                                                      \
        to_sol_obj->strides_r2hcf->v_in_stride =                               \
            from_sol_obj->strides_r2hcf->v_in_stride;                          \
        to_sol_obj->strides_r2hcf->v_out_stride =                              \
            from_sol_obj->strides_r2hcf->v_out_stride;                         \
    }                                                                          \
}

#define RESET_COST(sol)                                                        \
{                                                                              \
    sol->cost_analysis->ops = 0;                                               \
    sol->cost_analysis->time = 0;                                              \
}

// Shrink_dim_rank : returns the new dim rank by adding the no of dimentions
// those size is not equal to one.
// Ex:- 2x1x3x1, returns 2
#define SHRINK_DIM_RANK(dims, dim_rank, ret)                                   \
{                                                                              \
    if (dim_rank == 1)                                                         \
    {                                                                          \
        ret = 1;                                                               \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        INT32 dim_rank_counter = 0;                                            \
        for (INT32 i = 0; i < dim_rank; i++)                                   \
        {                                                                      \
            if (dims[i].n != 1)                                                \
            {                                                                  \
                dim_rank_counter++;                                            \
            }                                                                  \
        }                                                                      \
        ret = dim_rank_counter > 0 ? dim_rank_counter : 1;                     \
    }                                                                          \
}

// Function declarations
INT32 register_solvers_kernels(kernel_t[NUM_KERNELS_IN_TABLE], INT32 dt,
                               INT32 dir, INT32 is_real, INT32 cpu_flags);
INT32 setup_dft_(aoclfftz_selector_t *sel, kernel_t *kertab);
INT32 setup_rdft_(aoclfftz_selector_t *sel, kernel_t *kertab,
                  aoclfftz_realhelper_t *realhelper);
VOID setup_twiddle_buffer_complex(aoclfftz_solution_t *sel);
VOID setup_twiddle_buffer_real(aoclfftz_solution_t *sel);
INT32 setup_dft_f_(aoclfftz_selector_t *sel, kernel_t *kertab);
INT32 setup_dft_d_(aoclfftz_selector_t *sel, kernel_t *kertab);
VOID *setup_dft_f(aoclfftz_prob_desc_f *problem);
VOID *setup_dft_d(aoclfftz_prob_desc_d *problem);
VOID *setup_dft_f_64_(aoclfftz_prob_desc_f_64_ *problem);
VOID *setup_dft_d_64_(aoclfftz_prob_desc_d_64_ *problem);
INT32 selector_batched_dft(aoclfftz_selector_t *sel, kernel_t *kertab);
INT32 selector_ndim_dft(aoclfftz_selector_t *sel, kernel_t *kertab);
INT32 selector_bluestein_dft(aoclfftz_selector_t *sel, kernel_t *kertab);
INT32 selector_buffered_dft(aoclfftz_selector_t *sel, kernel_t *kertab);
INT32 selector_permuted_dft(aoclfftz_selector_t *sel, kernel_t *kertab);
INT32 selector_direct_dft(aoclfftz_selector_t *sel, kernel_t *kertab);
INT32 selector_ct_dft(aoclfftz_selector_t *sel, kernel_t *kertab);
INT32 selector_sizeone_dft(aoclfftz_selector_t *sel, kernel_t *kertab);
INT32 selector_transpose(aoclfftz_selector_t *sel);

INT32 selector_direct_rdft(aoclfftz_selector_t *sel, kernel_t *kertab,
                           aoclfftz_realhelper_t *realhelper);
INT32 selector_batched_rdft(aoclfftz_selector_t *sel, kernel_t *kertab,
                             aoclfftz_realhelper_t *realhelper);
INT32 selector_buffered_rdft(aoclfftz_selector_t *sel, kernel_t *kertab,
                             aoclfftz_realhelper_t *realhelper);
INT32 selector_ct_rdft(aoclfftz_selector_t *sel, kernel_t *kertab,
                       aoclfftz_realhelper_t *realhelper);
VOID destroy_handle(VOID *handle);
VOID fuse_vecs(aoclfftz_solution_t *sol);

#endif // AOCLFFTZ_SELECTOR_H
