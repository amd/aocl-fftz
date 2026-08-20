// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file bench_problem.h
 *
 *  @brief Test bench problem descriptor functions and macros
 *
 *  This file contains the functions and macros related to test bench params
 *  and problem descriptor.
 *
 *  @author V. Murugan
 *  @author Srirammaswamy Srinivasan
 *  @author Jeya R
 */

#ifndef BENCH_PROBLEM_H
#define BENCH_PROBLEM_H


#include <math.h>
#include <stdio.h>
#include "test/bench/aoclfftz_bench.h"
#include "test/bench/utils/register_functions.h"
#include "test/bench/utils/bench_utils.h"
#include "test/utils/dims_vecs_helper.h"

/**
 * @brief Initialize problem descriptor with the bench params
 *
 */
#define INIT_PD(params, p_desc, dt_t, dim_t)                                   \
    {                                                                          \
        p_desc->dim_rank = params->dim_rank;                                   \
        p_desc->vec_rank = params->vec_rank;                                   \
        FFTZ_UINT32 is_align = params->aligned_alloc; \
        ALLOC_UNINIT(p_desc->dims, dim_t, sizeof(dim_t) * p_desc->dim_rank,    \
                        is_align);                                             \
        for (FFTZ_INT32 i = 0; i < p_desc->dim_rank; i++) \
        {                                                                      \
            p_desc->dims[i].n = (dt_t)params->dims[i].n;                       \
            p_desc->dims[i].in_stride = (dt_t)params->dims[i].in_stride;       \
            p_desc->dims[i].out_stride = (dt_t)params->dims[i].out_stride;     \
        }                                                                      \
        ALLOC_UNINIT(p_desc->vecs, dim_t, sizeof(dim_t) * p_desc->vec_rank,    \
                        is_align);                                             \
        for (FFTZ_INT32 i = 0; i < p_desc->vec_rank; i++) \
        {                                                                      \
            p_desc->vecs[i].n = (dt_t)params->vecs[i].n;                       \
            p_desc->vecs[i].in_stride = (dt_t)params->vecs[i].in_stride;       \
            p_desc->vecs[i].out_stride = (dt_t)params->vecs[i].out_stride;     \
        }                                                                      \
        if (params->num_threads > 0)                                           \
        {                                                                      \
            p_desc->pthr_fft.num_threads = params->num_threads;                \
            p_desc->pthr_fft.dynamic_load_model = 0;                           \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            p_desc->pthr_fft.num_threads = 1;                                  \
            p_desc->pthr_fft.dynamic_load_model = 0;                           \
        }                                                                      \
        if (params->dynamic_load_model >= 1)                                   \
        {                                                                      \
            p_desc->pthr_fft.dynamic_load_model = 1;                           \
        }                                                                      \
        p_desc->cntrl_params.opt_level = params->opt_level;                    \
        p_desc->flags = set_flag(params);                                      \
        p_desc->cntrl_params.opt_off = 0;                                      \
        if (params->opt_level == -1)                                           \
        {                                                                      \
            p_desc->cntrl_params.opt_off = 1;                                  \
        }                                                                      \
        p_desc->cntrl_params.logger_mode = params->logger_mode;                \
        p_desc->cntrl_params.measure_stats = 0;                                \
    }

/**
 * @brief Destroy the problem descriptor
 *
 */
#define DESTROY_PD(p_desc, is_align)                                           \
    {                                                                          \
        if (p_desc != NULL)                                                    \
        {                                                                      \
            FREE_ALLOCATED_MEM(p_desc->dims, is_align);                        \
            FREE_ALLOCATED_MEM(p_desc->vecs, is_align);                        \
            FREE_ALLOCATED_MEM(p_desc, is_align);                              \
        }                                                                      \
    }

FFTZ_INT32 prepare_bench_params(FFTZ_INT32 argc, FFTZ_CHAR **argv,
                           aoclfftz_bench_params_t *bench_params);
aoclfftz_flags_t set_flag(aoclfftz_bench_params_t *params);
FFTZ_INT32 get_option(FFTZ_CHAR **argv, FFTZ_INT32 arg_idx);
FFTZ_VOID *setup_problem_f(aoclfftz_bench_params_t *params);
FFTZ_VOID *setup_problem_d(aoclfftz_bench_params_t *params);
FFTZ_VOID *setup_problem_f_64_(aoclfftz_bench_params_t *params);
FFTZ_VOID *setup_problem_d_64_(aoclfftz_bench_params_t *params);
FFTZ_VOID destroy_bench_param(aoclfftz_bench_params_t *params);
FFTZ_VOID show_help_menu(FFTZ_VOID);

#endif // BENCH_PROBLEM_H
