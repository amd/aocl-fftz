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

#define AOCLFFTZ_FIXED_SELECTOR_MODE 0    //Fixed decision logic
#define AOCLFFTZ_AUTO_SELECTOR_MODE  1    //Auto tuner

//Error return codes related to selector
//Add more codes at the top
typedef enum
{
    SELECTOR_FAILURE = -1,
    SELECTOR_SUCCESS         //Successful operation
} aoclfftz_selector_status;

//Selector data structure that is used to hold the solution and cost analysis
// at each decomposition level for the associated sub-problem
typedef struct
{
    aoclfftz_solution_t *solution;
    cost_analysis_t *cost_analysis;
 } aoclfftz_selector_t;

//macro functions
#define INIT_DECOMP_SCHEME(sel_obj, problem) { \
    sel_obj->solution->decomp_scheme->vec_rank = problem->vec_rank; \
    sel_obj->solution->decomp_scheme->dim_rank = problem->dim_rank; \
    UINT32 cnt; \
    for (cnt = 0; cnt < problem->dim_rank; cnt++) \
    { \
        sel_obj->solution->decomp_scheme->dims[cnt].n = \
                                            problem->dims[cnt].n; \
        sel_obj->solution->decomp_scheme->dims[cnt].in_stride = \
                                            problem->dims[cnt].in_stride; \
        sel_obj->solution->decomp_scheme->dims[cnt].out_stride = \
                                            problem->dims[cnt].out_stride; \
    } \
    for (cnt = 0; cnt < problem->vec_rank; cnt++) \
    { \
        sel_obj->solution->decomp_scheme->vecs[cnt].n = \
                                            problem->vecs[cnt].n; \
        sel_obj->solution->decomp_scheme->vecs[cnt].in_stride = \
                                            problem->vecs[cnt].in_stride; \
        sel_obj->solution->decomp_scheme->vecs[cnt].out_stride = \
                                            problem->vecs[cnt].out_stride; \
    } \
    if (FFT_DIR(problem->flags) == FORWARD_FFT_DIR) \
    { \
        sel_obj->solution->decomp_scheme->in_real = problem->in; \
        sel_obj->solution->decomp_scheme->in_imag = problem->in+1; \
        sel_obj->solution->decomp_scheme->out_real = problem->out; \
        sel_obj->solution->decomp_scheme->out_imag = problem->out+1; \
    } \
    else \
    { \
        sel_obj->solution->decomp_scheme->in_real = problem->in+1; \
        sel_obj->solution->decomp_scheme->in_imag = problem->in; \
        sel_obj->solution->decomp_scheme->out_real = problem->out+1; \
        sel_obj->solution->decomp_scheme->out_imag = problem->out; \
    } \
    sel_obj->solution->decomp_scheme->cntrl_params = \
                                            &(problem->cntrl_params); \
    sel_obj->solution->decomp_scheme->pthr_fft = &(problem->pthr_fft); \
    sel_obj->solution->decomp_scheme->flags = problem->flags; \
}

#define COPY_SOLUTION_OBJ(to_sol_obj, from_sol_obj) { \
    to_sol_obj->solver->solver_type = \
            from_sol_obj->solver->solver_type; \
    to_sol_obj->solver->execute_solver = \
            from_sol_obj->solver->execute_solver; \
    to_sol_obj->solver->destroy_solver = \
            from_sol_obj->solver->destroy_solver; \
    to_sol_obj->solver->kernel_r = \
            from_sol_obj->solver->kernel_r; \
    to_sol_obj->solver->kernel_m = \
            from_sol_obj->solver->kernel_m; \
    to_sol_obj->decomp_scheme->vec_rank = \
        from_sol_obj->decomp_scheme->vec_rank; \
    to_sol_obj->decomp_scheme->dim_rank = \
        from_sol_obj->decomp_scheme->dim_rank; \
    UINT32 cnt; \
    for (cnt = 0; cnt < to_sol_obj->decomp_scheme->dim_rank; cnt++) \
    { \
        to_sol_obj->decomp_scheme->dims[cnt].n = \
            from_sol_obj->decomp_scheme->dims[cnt].n; \
        to_sol_obj->decomp_scheme->dims[cnt].in_stride = \
            from_sol_obj->decomp_scheme->dims[cnt].in_stride; \
        to_sol_obj->decomp_scheme->dims[cnt].out_stride = \
            from_sol_obj->decomp_scheme->dims[cnt].out_stride; \
    } \
    for (cnt = 0; cnt < to_sol_obj->decomp_scheme->vec_rank; cnt++) \
    { \
        to_sol_obj->decomp_scheme->vecs[cnt].n = \
            from_sol_obj->decomp_scheme->vecs[cnt].n; \
        to_sol_obj->decomp_scheme->vecs[cnt].in_stride = \
            from_sol_obj->decomp_scheme->vecs[cnt].in_stride; \
        to_sol_obj->decomp_scheme->vecs[cnt].out_stride = \
            from_sol_obj->decomp_scheme->vecs[cnt].out_stride; \
    } \
    to_sol_obj->decomp_scheme->in_real = \
        from_sol_obj->decomp_scheme->in_real; \
    to_sol_obj->decomp_scheme->in_imag = \
        from_sol_obj->decomp_scheme->in_imag; \
    to_sol_obj->decomp_scheme->out_real = \
        from_sol_obj->decomp_scheme->out_real; \
    to_sol_obj->decomp_scheme->out_imag = \
        from_sol_obj->decomp_scheme->out_imag; \
    to_sol_obj->decomp_scheme->cntrl_params = \
        from_sol_obj->decomp_scheme->cntrl_params; \
    to_sol_obj->decomp_scheme->pthr_fft = \
        from_sol_obj->decomp_scheme->pthr_fft; \
    to_sol_obj->decomp_scheme->flags = \
        from_sol_obj->decomp_scheme->flags; \
    to_sol_obj->strides->in_stride = \
        from_sol_obj->strides->in_stride; \
    to_sol_obj->strides->out_stride = \
        from_sol_obj->strides->out_stride; \
    to_sol_obj->strides->v_in_stride = \
        from_sol_obj->strides->v_in_stride; \
    to_sol_obj->strides->v_out_stride = \
        from_sol_obj->strides->v_out_stride; \
    to_sol_obj->twiddle->TW = \
        from_sol_obj->twiddle->TW; \
    to_sol_obj->next_sol = \
        from_sol_obj->next_sol; \
}

//Function declarations
INT32 register_solvers_kernels(
                            kernel_t [NUM_KERNELS_IN_TABLE],
                            INT32 dt, INT32 cpu_flags);
INT32 setup_dft_(aoclfftz_selector_t *sel, kernel_t *kertab);
INT32 setup_dft_f_(aoclfftz_selector_t *, kernel_t *);
INT32 setup_dft_d_(aoclfftz_selector_t *, kernel_t *);
VOID *setup_dft_f(aoclfftz_prob_desc_f *);
VOID *setup_dft_d(aoclfftz_prob_desc_d *);
VOID *setup_dft_f_64_(aoclfftz_prob_desc_f_64_ *);
VOID *setup_dft_d_64_(aoclfftz_prob_desc_d_64_ *);
INT32 selector_batched_dft(aoclfftz_selector_t *sel, kernel_t *kertab);
INT32 selector_ndim_dft(aoclfftz_selector_t *sel, kernel_t *kertab);
INT32 selector_bluestein_dft(aoclfftz_selector_t *sel, kernel_t *kertab);
INT32 selector_buffered_dft(aoclfftz_selector_t *sel, kernel_t *kertab);
INT32 selector_permuted_dft(aoclfftz_selector_t *sel, kernel_t *kertab);
INT32 selector_direct_dft(aoclfftz_selector_t *sel, kernel_t *kertab);
INT32 selector_ct_dft(aoclfftz_selector_t *sel, kernel_t *kertab);
VOID destroy_handle(VOID *handle);

#endif //AOCLFFTZ_SELECTOR_H