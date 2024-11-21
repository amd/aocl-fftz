/**
 * Copyright (C) 2024, Advanced Micro Devices. All rights reserved.
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
        UINT32 is_align = params->aligned_alloc;                               \
        ALLOC_UNINIT(p_desc->dims, dim_t, sizeof(dim_t) * p_desc->dim_rank,    \
                        is_align);                                             \
        for (INT32 i = 0; i < p_desc->dim_rank; i++)                           \
        {                                                                      \
            p_desc->dims[i].n = (dt_t)params->dims[i].n;                       \
            p_desc->dims[i].in_stride = (dt_t)params->dims[i].in_stride;       \
            p_desc->dims[i].out_stride = (dt_t)params->dims[i].out_stride;     \
        }                                                                      \
        ALLOC_UNINIT(p_desc->vecs, dim_t, sizeof(dim_t) * p_desc->vec_rank,    \
                        is_align);                                             \
        for (INT32 i = 0; i < p_desc->vec_rank; i++)                           \
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
            p_desc->pthr_fft.num_threads = 0;                                  \
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

INT32 prepare_bench_params(INT32 argc, CHAR **argv,
                           aoclfftz_bench_params_t *bench_params);
UINT32 set_flag(aoclfftz_bench_params_t *params);
INT32 get_option(CHAR **argv, INT32 arg_idx);
VOID *setup_problem_f(aoclfftz_bench_params_t *params);
VOID *setup_problem_d(aoclfftz_bench_params_t *params);
VOID *setup_problem_f_64_(aoclfftz_bench_params_t *params);
VOID *setup_problem_d_64_(aoclfftz_bench_params_t *params);
VOID destroy_bench_param(aoclfftz_bench_params_t *params);
VOID show_help_menu(VOID);

#endif // BENCH_PROBLEM_H
