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

/** @file solver.h
 *
 *  @brief Solver data strcture and types.
 *
 *  This file contains the list of different solvers and the data structure to
 *  hold a specific solver for solving a given input problem or sub-problem.
 *
 *  @author S. Biplab Raut
 */

#ifndef AOCLFFTZ_SOLVER_H
#define AOCLFFTZ_SOLVER_H

#include "api/aoclfftz_internal.h"
#include "core/kernels/kernel.h"

// Error return codes related to solver
// Add more codes at the top
typedef enum
{
    SOLVER_FAILURE = -1,
    SOLVER_SUCCESS         // Successful operation
} aoclfftz_solver_status;

// Solver types implemented in the library for executing a given DFT problem
typedef enum
{
    SOLVER_DIRECT = 1,
    SOLVER_CT,
    SOLVER_CT_TWIDDLE,
    SOLVER_NDIM,
    SOLVER_BUFFERED,
    SOLVER_PERM_KER,
    SOLVER_BATCHED,
    SOLVER_BLUESTEIN,
    SOLVER_PFA,
    SOLVER_RADER,
    SOLVER_PERM_COPY,
    SOLVER_TRANSPOSE,
    SOLVER_SIZEONE,
    SOLVER_MT_DIRECT,
    SOLVER_MT_BATCHED,
    SOLVER_REAL_DIRECT,
    SOLVER_REAL_DIRECT_TWIDDLE,
    SOLVER_REAL_CT,
    SOLVER_REAL_NDIM,
    SOLVER_REAL_BUFFERED,
    SOLVER_REAL_BATCHED,
    SOLVER_REAL_PERM_KER,
    SOLVER_REAL_MT_DIRECT,
    SOLVER_REAL_MT_DIRECT_TWIDDLE,
    SOLVER_REAL_MT_BATCHED,
    NUM_SOLVERS_END
} aoclfftz_solver_type;

// Solver data structure that holds solver object/pointer and its type
typedef struct solver
{
    aoclfftz_generic_solver_t *solver;
    // aoclfftz_solver_type solv_type;
} solver_t;

INT32 register_solvers(INT32 dt, INT32 is_real, INT32 cpu_flags);
dft_solver_ get_solver_fp(aoclfftz_solution_t *sol);
INT32 set_solver_fp(aoclfftz_generic_solver_t *solver_obj);

// Function declarations of all the supported solvers
// (called by selector and executor)
INT32 setup_direct_solver(aoclfftz_solution_t *sol, cost_analysis_t *cost,
                          kernel_t *kernel);
INT32 setup_ct_solver(aoclfftz_solution_t *sol, aoclfftz_solution_t *sol_r,
                      aoclfftz_solution_t *sol_m, UINT32 radix_r,
                      UINT32 radix_m);
INT32 setup_batched_solver(aoclfftz_solution_t *sol);
INT32 setup_bluestein_solver(aoclfftz_solution_t *sol,
                             aoclfftz_solution_t *next_sol, INTP m);
INT32 setup_ndim_solver(aoclfftz_solution_t *sol,
                        aoclfftz_solution_t *n_minus1_sol,
                        aoclfftz_solution_t *outer_dim_sol);
INT32 setup_sizeone_solver(aoclfftz_solution_t *sol);
INT32 setup_transpose_solver(aoclfftz_solution_t *sol, INT32 cpu_flags);
#if 0
INT32 setup_permuted_solver(aoclfftz_solution_t *sol, cost_analysis_t *cost,
                            kernel_t *kernel);
INT32 setup_buffered_solver(aoclfftz_solution_t *sol, cost_analysis_t *cost,
                            kernel_t *kernel);
#endif
#ifdef MULTI_THREADING
INT32 setup_mt_direct_solver(aoclfftz_solution_t *sol, cost_analysis_t *cost,
                             kernel_t *kernel);
INT32 setup_mt_batched_solver(aoclfftz_solution_t *sol,
                              UINT32 num_threads_used);
#endif

// RealFFT-Solvers
INT32 setup_real_direct_solver(aoclfftz_solution_t *sol, cost_analysis_t *cost,
                               const kernel_t *kernel_c2c,
                               const kernel_t *kernel_r2hc,
                               const kernel_t *kernel_r2hcf,
                               aoclfftz_realhelper_t *realhelper);
INT32 setup_real_batched_solver(aoclfftz_solution_t *sol,
                                aoclfftz_solution_t *next_sol,
                                aoclfftz_realhelper_t *realhelper);
INT32 setup_real_buffered_solver(aoclfftz_solution_t *sol,
                                aoclfftz_realhelper_t *realhelper);
INT32 setup_real_ct_solver(aoclfftz_solution_t *sol,
                           aoclfftz_solution_t *sol_r,
                           aoclfftz_solution_t *sol_m,
                           UINT32 radix_r,
                           UINT32 radix_m,
                           aoclfftz_realhelper_t * realhelper);
INT32 setup_real_buffered_solver(aoclfftz_solution_t *sol,
                                aoclfftz_realhelper_t *realhelper);
INT32 setup_real_ct_solver(aoclfftz_solution_t *sol, aoclfftz_solution_t *sol_r,
                           aoclfftz_solution_t *sol_m, UINT32 radix_r,
                           UINT32 radix_m, aoclfftz_realhelper_t * realhelper);
#ifdef MULTI_THREADING
INT32 setup_real_mt_direct_solver(aoclfftz_solution_t *sol,
                                  cost_analysis_t *cost,
                                  const kernel_t *kernel_c2c,
                                  const kernel_t *kernel_r2hc,
                                  const kernel_t *kernel_r2hcf,
                                  aoclfftz_realhelper_t *realhelper);
INT32 setup_real_mt_batched_solver(aoclfftz_solution_t *sol,
                                   aoclfftz_solution_t *next_sol,
                                   aoclfftz_realhelper_t *realhelper);
#endif

dft_solver_ register_execute_direct_solver(VOID);
dft_solver_ register_execute_ct_solver(VOID);
dft_solver_ register_execute_ct_twiddle_solver(VOID);
dft_solver_ register_execute_last_stage_ip_ct_solver(VOID);
dft_solver_ register_execute_batched_solver(VOID);
dft_solver_ register_execute_bluestein_solver(VOID);
dft_solver_ register_execute_ndim_solver(VOID);
dft_solver_ register_execute_last_stage_ip_ndim_solver(VOID);
dft_solver_ register_execute_sizeone_solver(VOID);
dft_solver_ register_execute_transpose_solver(VOID);
#ifdef MULTI_THREADING
dft_solver_ register_execute_mt_direct_solver(VOID);
dft_solver_ register_execute_mt_batched_solver(VOID);
#endif

dft_solver_ register_execute_real_direct_solver(VOID);
dft_solver_ register_execute_real_batched_solver(VOID);
dft_solver_ register_execute_real_buffered_solver(VOID);
dft_solver_ register_execute_real_ct_solver(VOID);

#ifdef MULTI_THREADING
dft_solver_ register_execute_real_mt_direct_solver(VOID);
dft_solver_ register_execute_real_mt_batched_solver(VOID);
#endif
#endif // AOCLFFTZ_SOLVER_H
