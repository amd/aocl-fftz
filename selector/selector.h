// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

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

#define AOCLFFTZ_FIXED_SELECTOR_MODE
//#define AOCLFFTZ_AUTO_SELECTOR_MODE

typedef enum
{
    AOCLFFTZ_FIXED_SELECTOR, // Fixed decision logic + Fused Twiddle and DFT
                             // kernels
    AOCLFFTZ_AUTO_SELECTOR,  // Auto tuner mode
    AOCLFFTZ_SELECTOR_MODELS // Total selector models
} selector_model_t;

// Error return codes related to selector
// Add more codes at the top
typedef enum
{
    SELECTOR_FAILURE = -1,
    SELECTOR_SUCCESS // Successful operation
} aoclfftz_selector_status;

// Selector data structure that is used to hold the solution and cost analysis
// at each decomposition level for the associated sub-problem
typedef struct aoclfftz_selector
{
    aoclfftz_solution_t *solution;
    execute_ execute;
    cost_analysis_t *cost_analysis;
    kernel_tables_t *kernel_tables;
} aoclfftz_selector_t;

// Define thread workload threshold to choose between batched direct approaches
// in MT and these parameters are set based on performance experiments.
#define MIN_OPCNT_PER_THREAD 5000
#define MIN_DIM_STRIDE_FOR_COLMAJOR 50000
// Scaling factor for kernel weightage calculation to normalize cycle counts
#define KERNEL_WEIGHTAGE_SCALE_FACTOR 100.0
/*
 * @brief Helper function to iterate over the solutions
 */
#define FOR_EACH_SOLUTION(var, start_solution)                                 \
    for (aoclfftz_solution_t *var = start_solution; var != NULL;               \
         var = (var != NULL && var->next_sol) ? *(var->next_sol) : NULL)

/*
 * @brief Macro to check if there is a next solution
 */
#define HAS_NEXT(sol) (sol->next_sol != NULL && sol->next_sol[0] != NULL)

// macro functions
#define INIT_DECOMP_SCHEME(sel_obj, problem, dim_rank)                         \
{                                                                              \
    sel_obj->solution->decomp_scheme->vec_rank = problem->vec_rank;            \
    sel_obj->solution->decomp_scheme->dim_rank = dim_rank;                     \
    FFTZ_INT32 cnt, idx = 0; \
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
    sel_obj->solution->decomp_scheme->in_real = problem->in;                   \
    sel_obj->solution->decomp_scheme->in_imag = problem->in + 1;               \
    sel_obj->solution->decomp_scheme->out_real = problem->out;                 \
    sel_obj->solution->decomp_scheme->out_imag = problem->out + 1;             \
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
    sel_obj->solution->decomp_scheme->thread_info->pthr_fft->num_threads =     \
        problem->pthr_fft.num_threads;                                         \
    sel_obj->solution->decomp_scheme->thread_info->avl_threads =               \
    sel_obj->solution->decomp_scheme->thread_info->pthr_fft->num_threads;      \
    sel_obj->solution->decomp_scheme->thread_info->n_threads = 1;              \
    sel_obj->solution->decomp_scheme->outer_buf_cnt = 1;                       \
    sel_obj->solution->decomp_scheme->flags =                                  \
        (problem->flags.fft_placement       << 0) |                            \
        (problem->flags.storage_order       << 1) |                            \
        (problem->flags.fft_direction       << 2) |                            \
        (problem->flags.fft_type            << 3) |                            \
        (problem->flags.bit_reproducibility << 4) |                            \
        (problem->flags.transpose_mode      << 8);                             \
}

/*
 * @brief Overwrite the solution-side `opt_level` with the dispatcher-resolved
 * level (cpu_flags = min(user opt_level, hw/build ISA level), or scalar when
 * opt_off is set or the level is non-positive).
 *
 * Conceptually:
 *   - `problem->cntrl_params.opt_level` holds the user request (unchanged).
 *   - `sel_obj->solution->decomp_scheme->cntrl_params->opt_level` holds the
 *     effective level the library will actually run at.
 * The struct/field name is shared between the two; only the meaning of the
 * value differs based on which side it lives on.
 *
 * Must be called after `INIT_DECOMP_SCHEME` (which initially copies the user
 * value) and after `setup_dynamic_dispatcher` has produced `cpu_flags`.
 */
#define SET_EFFECTIVE_OPT_LEVEL(sel_obj, cpu_flags)                            \
{                                                                              \
    sel_obj->solution->decomp_scheme->cntrl_params->opt_level = (cpu_flags);   \
}

/*
 * @brief Check if the Root problem is a Direct Problem or not.
 * If a problem is not direct, it will be a multi stage with atleast one CT
 * Problem.
 *
 * `sol` can be any solution in the hierarchy of solutions.
 *
 * NOTE:
 * Reasoning:
 * * In a generic solution plan, `TW` is `NULL` for all solutions before first
 * CT. Hence `TW == NULL` ensures current solution isn't after a CT.
 * * `sol->next_sol == NULL` checks if the solution is the last one in the
 * hierarchy.
 * * When both are true, it checks that we have walked the entire solution
 * hierarchy and found no CT solution.
 * * This is only possible for a Direct only porblem.
 *
 * @param sol Pointer to the solution structure.
 *
 */
#define IS_DIRECT_ONLY_PROBLEM(sol)                                            \
    (sol->twiddle->TW == NULL && sol->next_sol == NULL)

/**
 * @brief Swap the buffers of two pointers
 *
 */
#define SWAP_BUFFERS(buf1, buf2)                                               \
{                                                                              \
    FFTZ_VOID *temp_buffer_for_swap = buf1; \
    buf1 = buf2;                                                               \
    buf2 = temp_buffer_for_swap;                                               \
}

#define RESET_COST(sol)                                                        \
{                                                                              \
    sol->cost_analysis->ops = 0;                                               \
    sol->cost_analysis->time = 0;                                              \
}

// Shrink_dim_rank : returns the new dim rank by adding the number of dimensions
// whose size is not equal to one.
// Ex:- 2x1x3x1, returns 2
#define SHRINK_DIM_RANK(dims, dim_rank, ret)                                   \
{                                                                              \
    if (dim_rank == 1)                                                         \
    {                                                                          \
        ret = 1;                                                               \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        FFTZ_INT32 dim_rank_counter = 0; \
        for (FFTZ_INT32 i = 0; i < dim_rank; i++) \
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
FFTZ_INT32 copy_decomp_scheme(
    aoclfftz_decomp_scheme_t *to_ds,
    aoclfftz_decomp_scheme_t *from_ds);
FFTZ_INT32 copy_solution_obj(aoclfftz_solution_t *to_sol_obj,
                        aoclfftz_solution_t *from_sol_obj);
// maps both in & out pointers to out pointer
// incase of out-of-place problems, except the first DFT, other DFTs happen
// in-place ie., in the output buffer.
FFTZ_INT32 copy_solution_obj_out_p(aoclfftz_solution_t *to_sol_obj,
                             aoclfftz_solution_t *from_sol_obj);
// Copy strides from one solution to another
// except for the BATCHED_CT_L1_DIRECT solver
FFTZ_INT32 copy_strides(aoclfftz_solution_t *to_sol_obj,
                   aoclfftz_solution_t *from_sol_obj);
// Copy strides from one solution to another for the BATCHED_CT_L1_DIRECT solver
FFTZ_INT32 copy_strides_batched_ct_l1_direct(
    aoclfftz_solution_t *to_sol_obj,
    aoclfftz_solution_t *from_sol_obj);
// copy all contents except dims & vecs
// necessary in ND setup where dim_rank & vec_rank will differ for the
// sub-problem
FFTZ_VOID copy_solution_obj_wo_dims(aoclfftz_solution_t *to_sol_obj,
                               aoclfftz_solution_t *from_sol_obj);
FFTZ_VOID swap_real_ct_solutions(aoclfftz_selector_t *sel);
FFTZ_INT32 register_solvers_kernels(kernel_tables_t *kernel_tables,
                                    FFTZ_INT32 dt, FFTZ_INT32 dir,
                                    FFTZ_INT32 is_real, FFTZ_INT32 cpu_flags);
FFTZ_INT32 selector_driver_dft_(aoclfftz_selector_t *sel);
FFTZ_INT32 selector_driver_rdft_(aoclfftz_selector_t *sel,
                            aoclfftz_realhelper_t *realhelper);
FFTZ_INT32 selector_model_dft_(aoclfftz_selector_t *sel);
FFTZ_INT32 selector_model_rdft_(aoclfftz_selector_t *sel,
                           aoclfftz_realhelper_t *realhelper);
FFTZ_VOID setup_twiddle_buffer_complex(aoclfftz_solution_t *sol);
FFTZ_VOID setup_twiddle_buffer_real(aoclfftz_solution_t *sol);
FFTZ_VOID *setup_dft_f(aoclfftz_prob_desc_f *problem);
FFTZ_VOID *setup_dft_d(aoclfftz_prob_desc_d *problem);
FFTZ_VOID *setup_dft_f_64_(aoclfftz_prob_desc_f_64_ *problem);
FFTZ_VOID *setup_dft_d_64_(aoclfftz_prob_desc_d_64_ *problem);
FFTZ_INT32 selector_batched_dft(aoclfftz_selector_t *sel, kernel_t *kertab);
FFTZ_INT32 selector_ndim_dft(aoclfftz_selector_t *sel, kernel_t *kertab);
FFTZ_INT32 selector_bluestein_dft(aoclfftz_selector_t *sel, kernel_t *kertab);
FFTZ_INT32 selector_buffered_dft(aoclfftz_selector_t *sel, kernel_t *kertab);
FFTZ_INT32 selector_permuted_dft(aoclfftz_selector_t *sel, kernel_t *kertab);
FFTZ_INT32 selector_direct_dft(aoclfftz_selector_t *sel, kernel_t *kertab);
FFTZ_INT32 selector_ct_dft(aoclfftz_selector_t *sel, kernel_t *kertab);
FFTZ_INT32 selector_batched_ct_l1_direct_dft(aoclfftz_selector_t *sel);
FFTZ_INT32 selector_sizeone_dft(aoclfftz_selector_t *sel, kernel_t *kertab);
FFTZ_INT32 selector_transpose(aoclfftz_selector_t *sel);
FFTZ_INT32 selector_sr_dft(aoclfftz_selector_t *sel, kernel_t *kertab);

FFTZ_INT32 selector_direct_rdft(aoclfftz_selector_t *sel, kernel_t *kertab,
                           aoclfftz_realhelper_t *realhelper);
FFTZ_INT32 selector_batched_rdft(aoclfftz_selector_t *sel, kernel_t *kertab,
                             aoclfftz_realhelper_t *realhelper);
FFTZ_INT32 selector_buffered_rdft(aoclfftz_selector_t *sel, kernel_t *kertab,
                             aoclfftz_realhelper_t *realhelper);
FFTZ_INT32 selector_ct_rdft(aoclfftz_selector_t *sel, kernel_t *kertab,
                       aoclfftz_realhelper_t *realhelper);
FFTZ_INT32 selector_ndim_rdft(aoclfftz_selector_t *sel, kernel_t *kertab,
                         aoclfftz_realhelper_t *realhelper);
FFTZ_VOID destroy_handle(FFTZ_VOID *handle);
FFTZ_VOID fuse_vecs(aoclfftz_solution_t *sol, FFTZ_INT32 is_FFT_ker_supported);
FFTZ_INT32 check_bluestein_problem(aoclfftz_decomp_scheme_t *decomp_scheme);
FFTZ_INT32 check_FFT_kernel_support(FFTZ_INTP n, kernel_t *kernels_table,
                               FFTZ_INT32 is_innermost_dim);
FFTZ_DOUBLE get_kernel_weightage(FFTZ_INTP radix, kernel_t *kertab,
                            aoclfftz_solution_t *sol);
FFTZ_UINT8 should_use_colmajor_batched_solver(aoclfftz_solution_t *solution,
                                              kernel_t *kertab,
                                              FFTZ_INT32 avl_threads);
FFTZ_UINT8 check_col_major(aoclfftz_decomp_scheme_t *decomp_scheme);

#endif // AOCLFFTZ_SELECTOR_H
