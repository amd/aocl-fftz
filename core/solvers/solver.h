// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

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
    SOLVER_NULL = 0,
    SOLVER_DIRECT = 1,
    SOLVER_DIRECT_BATCHED_COLMAJOR,
    SOLVER_CT,
    SOLVER_BATCHED_CT_L1_DIRECT,
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
    SOLVER_SR,
    SOLVER_MT_DIRECT,
    SOLVER_MT_DIRECT_BATCHED_COLMAJOR,
    SOLVER_MT_DIRECT_BATCHED_ROWMAJOR,
    SOLVER_MT_BATCHED,
    SOLVER_MT_BLUESTEIN,
    SOLVER_REAL_DIRECT_R2C,
    SOLVER_REAL_DIRECT_R2C_BATCHED,
    SOLVER_REAL_DIRECT_C2R,
    SOLVER_REAL_DIRECT_CT_R2C,
    SOLVER_REAL_DIRECT_CT_C2R,
    SOLVER_REAL_CT,
    SOLVER_REAL_NDIM,
    SOLVER_REAL_BUFFERED,
    SOLVER_REAL_BATCHED,
    SOLVER_REAL_PERM_KER,
    SOLVER_REAL_SIZEONE,
    SOLVER_REAL_MT_DIRECT_R2C,
    SOLVER_REAL_MT_DIRECT_R2C_BATCHED,
    SOLVER_REAL_MT_DIRECT_C2R,
    SOLVER_REAL_MT_DIRECT_CT_R2C,
    SOLVER_REAL_MT_DIRECT_CT_C2R,
    SOLVER_REAL_MT_BATCHED,
    NUM_SOLVERS_END
} aoclfftz_solver_type;

static inline FFTZ_UINT8
is_solver_real_direct_family(aoclfftz_solver_type solver_type)
{
    return (solver_type >= SOLVER_REAL_DIRECT_R2C &&
            solver_type <= SOLVER_REAL_DIRECT_CT_C2R) ||
           (solver_type >= SOLVER_REAL_MT_DIRECT_R2C &&
            solver_type <= SOLVER_REAL_MT_DIRECT_CT_C2R);
}

FFTZ_INT32 register_solvers(FFTZ_VOID);
FFTZ_INT32 set_solver_fp(aoclfftz_generic_solver_t *solver_obj);
FFTZ_INT32 is_solver_registered(aoclfftz_solver_type solver_type);

// Function declarations of all the supported solvers
// (called by selector and executor)
FFTZ_INT32 setup_direct_solver(aoclfftz_solution_t *sol, cost_analysis_t *cost,
                          kernel_t *kernel);
FFTZ_INT32 setup_ct_solver(aoclfftz_solution_t *sol, aoclfftz_solution_t *sol_r,
                      aoclfftz_solution_t *sol_m, FFTZ_UINT32 radix_r,
                      FFTZ_UINT32 radix_m);
FFTZ_INT32 setup_batched_ct_l1_direct_solver(aoclfftz_solution_t *sol,
                                        kernel_t *ker_m, kernel_t *ker_r,
                                        FFTZ_INTP radix_r, FFTZ_INTP radix_m);
FFTZ_INT32 setup_buffered_solver(aoclfftz_solution_t *sol,
                            aoclfftz_solution_t *next_sol);
FFTZ_INT32 setup_batched_solver(aoclfftz_solution_t *sol);
FFTZ_INT32 setup_bluestein_solver(aoclfftz_solution_t *sol,
                                  aoclfftz_solution_t *next_sol, FFTZ_INTP m);
FFTZ_INT32 compute_chirp_fft(aoclfftz_solution_t *sol,
                             aoclfftz_solution_t *next_sol,
                             aoclfftz_mutable_ctx_t *ctx);
FFTZ_INT32 setup_ndim_solver(aoclfftz_solution_t *sol,
                             aoclfftz_solution_t *n_minus1_sol,
                             aoclfftz_solution_t *outer_dim_sol);
FFTZ_INT32 setup_sizeone_solver(aoclfftz_solution_t *sol);
FFTZ_INT32 setup_transpose_solver(aoclfftz_solution_t *sol);
FFTZ_INT32 setup_sr_solver(aoclfftz_solution_t *sol,
                           aoclfftz_solution_t *sol_even,
                           aoclfftz_solution_t *sol_odd1,
                           aoclfftz_solution_t *sol_odd3, FFTZ_INTP n_even,
                           FFTZ_INTP n_odd);
#ifdef MULTI_THREADING
FFTZ_INT32 setup_mt_direct_solver(aoclfftz_solution_t *sol,
                                  cost_analysis_t *cost, kernel_t *kernel,
                                  FFTZ_UINT8 *has_nested);
FFTZ_INT32 setup_mt_batched_solver(aoclfftz_solution_t *sol,
                                   FFTZ_INT32 num_threads_used,
                                   FFTZ_UINT8 *has_nested);
FFTZ_INT32 setup_mt_bluestein_solver(aoclfftz_solution_t *sol,
                                     aoclfftz_solution_t *next_sol,
                                     FFTZ_INTP m, FFTZ_UINT8 *has_nested);
#endif

// RealFFT-Solvers
FFTZ_INT32 setup_real_direct_solver(aoclfftz_solution_t *sol,
                                    cost_analysis_t *cost,
                                    const kernel_t *kernel_c2c,
                                    const kernel_t *kernel_r2hc,
                                    const kernel_t *kernel_r2hcf,
                                    aoclfftz_realhelper_t *realhelper);
FFTZ_INT32 setup_real_batched_solver(aoclfftz_solution_t *sol,
                                aoclfftz_solution_t *next_sol,
                                aoclfftz_realhelper_t *realhelper);
FFTZ_INT32 setup_real_buffered_solver(aoclfftz_solution_t *sol,
                                 aoclfftz_realhelper_t *realhelper);
FFTZ_INT32 setup_real_ct_solver(aoclfftz_solution_t *sol,
                                aoclfftz_solution_t *sol_r,
                                aoclfftz_solution_t *sol_m, FFTZ_UINT32 radix_r,
                                FFTZ_UINT32 radix_m,
                                aoclfftz_realhelper_t *realhelper);
FFTZ_INT32 setup_real_ndim_solver(aoclfftz_solution_t *sol,
                             aoclfftz_solution_t *real_dim_sol,
                             aoclfftz_solution_t *complex_dims_sol,
                             aoclfftz_realhelper_t *realhelper);
FFTZ_INT32 setup_real_sizeone_solver(aoclfftz_solution_t *sol);
#ifdef MULTI_THREADING
FFTZ_INT32 setup_real_mt_direct_solver(aoclfftz_solution_t *sol,
                                  cost_analysis_t *cost,
                                  const kernel_t *kernel_c2c,
                                  const kernel_t *kernel_r2hc,
                                  const kernel_t *kernel_r2hcf,
                                  aoclfftz_realhelper_t *realhelper,
                                  FFTZ_UINT8 *has_nested);
FFTZ_INT32 setup_real_mt_batched_solver(aoclfftz_solution_t *sol,
                                   aoclfftz_solution_t *next_sol,
                                   aoclfftz_realhelper_t *realhelper,
                                   FFTZ_UINT8 *has_nested);
#endif

dft_solver_ register_execute_direct_solver(FFTZ_VOID);
dft_solver_ register_execute_direct_batched_colmajor_solver(FFTZ_VOID);
dft_solver_ register_execute_ct_solver(FFTZ_VOID);
dft_solver_ register_execute_batched_ct_l1_direct_solver(FFTZ_VOID);
dft_solver_ register_execute_buffered_solver(FFTZ_VOID);
dft_solver_ register_execute_batched_solver(FFTZ_VOID);
dft_solver_ register_execute_bluestein_solver(FFTZ_VOID);
dft_solver_ register_execute_ndim_solver(FFTZ_VOID);
dft_solver_ register_execute_sizeone_solver(FFTZ_VOID);
dft_solver_ register_execute_transpose_solver(FFTZ_VOID);
dft_solver_ register_execute_sr_solver(FFTZ_VOID);
#ifdef MULTI_THREADING
dft_solver_ register_execute_mt_direct_solver(FFTZ_VOID);
dft_solver_ register_execute_mt_direct_batched_rowmajor_solver(FFTZ_VOID);
dft_solver_ register_execute_mt_direct_batched_colmajor_solver(FFTZ_VOID);
dft_solver_ register_execute_mt_batched_solver(FFTZ_VOID);
dft_solver_ register_execute_mt_bluestein_solver(FFTZ_VOID);
#endif

dft_solver_ register_execute_real_direct_r2c(FFTZ_VOID);
dft_solver_ register_execute_real_direct_r2c_batched(FFTZ_VOID);
dft_solver_ register_execute_real_direct_c2r(FFTZ_VOID);
dft_solver_ register_execute_real_direct_ct_r2c(FFTZ_VOID);
dft_solver_ register_execute_real_direct_ct_c2r(FFTZ_VOID);
dft_solver_ register_execute_real_batched_solver(FFTZ_VOID);
dft_solver_ register_execute_real_buffered_solver(FFTZ_VOID);
dft_solver_ register_execute_real_ct_solver(FFTZ_VOID);
dft_solver_ register_execute_real_ndim_solver(FFTZ_VOID);
dft_solver_ register_execute_real_sizeone_solver(FFTZ_VOID);

#ifdef MULTI_THREADING
dft_solver_ register_execute_real_mt_direct_r2c(FFTZ_VOID);
dft_solver_ register_execute_real_mt_direct_r2c_batched(FFTZ_VOID);
dft_solver_ register_execute_real_mt_direct_c2r(FFTZ_VOID);
dft_solver_ register_execute_real_mt_direct_ct_r2c(FFTZ_VOID);
dft_solver_ register_execute_real_mt_direct_ct_c2r(FFTZ_VOID);
dft_solver_ register_execute_real_mt_batched_solver(FFTZ_VOID);
#endif
FFTZ_INT64 compute_kernel_cost(const kernel_t *ker, FFTZ_UINT8 precision,
                          FFTZ_UINT8 direction, FFTZ_INTP batch);

#endif // AOCLFFTZ_SOLVER_H
