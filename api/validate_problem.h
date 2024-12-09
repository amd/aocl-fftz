/**
 * Copyright (C) 2025, Advanced Micro Devices. All rights reserved.
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

 /** @file validate_problem.h
 *
 *  @brief Utility macros and functions for validation of input problem
 *  descriptor.
 *
 *  This file contains the utility macros and functions related to validation
 *  of problem descriptor.
 *
 *  @author Jeya R
 */

#ifndef VALIDATE_PROBLEM_H
#define VALIDATE_PROBLEM_H

#include <stdio.h>
#include "aoclfftz.h"
#include "selector/selector.h"
#include "utils/utils.h"

// checks if input & output strides are the same for of an inplace problem
#define VALIDATE_INPLACE_STRIDES(dims, vecs, dim_rank, vec_rank, errno)        \
{                                                                              \
    for (INT32 i = 0; i < dim_rank; i++)                                       \
    {                                                                          \
        if (dims[i].in_stride != dims[i].out_stride)                           \
        {                                                                      \
            errno = AOCLFFTZ_INVALID_INPUT;                                    \
        }                                                                      \
    }                                                                          \
    for (INT32 i = 0; i < vec_rank; i++)                                       \
    {                                                                          \
        if (vecs[i].in_stride != vecs[i].out_stride)                           \
        {                                                                      \
            errno = AOCLFFTZ_INVALID_INPUT;                                    \
        }                                                                      \
    }                                                                          \
}

static inline INT32 validate_flags(UINT32 flags)
{
    INT32 is_real = IS_REAL(flags);
    INT32 is_out_of_order = IS_OUT_OF_ORDER(flags);

    // TODO: Remove validation once support for real FFTs is added
    if (is_real)
    {
        AOCLFFTZ_LOG_UNFORMATTED(ERR, ERR, "Library does not support "
                                           "real inputs");
        return AOCLFFTZ_INVALID_INPUT;
    }

    // TODO: Remove validation once support for out-of-order output is added
    if (is_out_of_order)
    {
        AOCLFFTZ_LOG_UNFORMATTED(ERR, ERR, "Library does not support "
                                            "out-of-order outputs");
        return AOCLFFTZ_INVALID_INPUT;
    }
    return AOCLFFTZ_SUCCESS;
}

#define VALIDATE_DIMS(dims, dim_rank, errno)                                   \
{                                                                              \
    if (dims == NULL)                                                          \
    {                                                                          \
        AOCLFFTZ_LOG_UNFORMATTED(ERR, ERR, "dims cannot be NULL\n");           \
        errno = AOCLFFTZ_INVALID_INPUT;                                        \
        goto validation_exit;                                                  \
    }                                                                          \
    for (INT32 i = 0; i < dim_rank; i++)                                       \
    {                                                                          \
        if (dims[i].n <= 0)                                                    \
        {                                                                      \
            AOCLFFTZ_LOG_FORMATTED(ERR, ERR, "dimension[%d]: size "            \
                                   "must be at least 1", i);                   \
            errno = AOCLFFTZ_INVALID_INPUT;                                    \
            goto validation_exit;                                              \
        }                                                                      \
        if (dims[i].in_stride <= 0)                                            \
        {                                                                      \
            AOCLFFTZ_LOG_FORMATTED(ERR, ERR, "dimension[%d]: in_stride "       \
                                   "must be greater than zero\n", i);          \
            errno = AOCLFFTZ_INVALID_INPUT;                                    \
            goto validation_exit;                                              \
        }                                                                      \
        if (dims[i].out_stride <= 0)                                           \
        {                                                                      \
            AOCLFFTZ_LOG_FORMATTED(ERR, ERR, "dimension[%d]: out_stride "      \
                                   "must be greater than zero\n", i);          \
            errno = AOCLFFTZ_INVALID_INPUT;                                    \
            goto validation_exit;                                              \
        }                                                                      \
    }                                                                          \
}

#define VALIDATE_VECS(dims, dim_rank, vecs, vec_rank, errno)                   \
{                                                                              \
    if (vecs == NULL)                                                          \
    {                                                                          \
        AOCLFFTZ_LOG_UNFORMATTED(ERR, ERR, "vecs cannot be NULL");             \
        errno = AOCLFFTZ_INVALID_INPUT;                                        \
        goto validation_exit;                                                  \
    }                                                                          \
    for (INT32 i = 0; i < vec_rank; i++)                                       \
    {                                                                          \
        if (vecs[i].n <= 0)                                                    \
        {                                                                      \
            AOCLFFTZ_LOG_FORMATTED(ERR, ERR, "vector[%d]: size "               \
                                   "must be at least 1", i);                   \
            errno = AOCLFFTZ_INVALID_INPUT;                                    \
            goto validation_exit;                                              \
        }                                                                      \
        if (vecs[i].in_stride <= 0)                                            \
        {                                                                      \
            AOCLFFTZ_LOG_FORMATTED(ERR, ERR, "vector[%d]: in_stride "          \
                                   "must be greater than zero\n", i);          \
            errno = AOCLFFTZ_INVALID_INPUT;                                    \
            goto validation_exit;                                              \
        }                                                                      \
        if (vecs[i].out_stride <= 0)                                           \
        {                                                                      \
            AOCLFFTZ_LOG_FORMATTED(ERR, ERR, "vector[%d]: out_stride "         \
                                   "must be greater than zero\n", i);          \
            errno = AOCLFFTZ_INVALID_INPUT;                                    \
            goto validation_exit;                                              \
        }                                                                      \
    }                                                                          \
}

static inline INT32 validate_control_params(aoclfftz_cntrl_params_t *cntrl_p)
{
    if (cntrl_p->logger_mode < 0 || cntrl_p->logger_mode > 4)
    {
        AOCLFFTZ_LOG_UNFORMATTED(INFO, INFO, "Invalid logger mode, "
                                 "running with default logger mode (0)");
        // set to default
        cntrl_p->logger_mode = 0;
    }
    if (cntrl_p->measure_stats != 0)
    {
        AOCLFFTZ_LOG_UNFORMATTED(INFO, INFO,
                               "measure-stats opt is currently not supported, "
                               "running with measure-stats disabled");
        // set to default
        cntrl_p->measure_stats = 0;
    }
    if (!cntrl_p->opt_off)
    {
        if (cntrl_p->opt_level < 0 || cntrl_p->opt_level > 4)
        {
            AOCLFFTZ_LOG_UNFORMATTED(INFO, INFO, "only opt-level 0-3 are "
                                    "currently supported; disabling "
                                    "optimization");
            // disabling optimization
            cntrl_p->opt_off = 1;
        }
    }
    return AOCLFFTZ_SUCCESS;
}

#define VALIDATE_PROBLEM_DESCRIPTOR(problem)                                   \
{                                                                              \
    INT32 ret = AOCLFFTZ_SUCCESS;                                              \
    if (problem == NULL)                                                       \
    {                                                                          \
        ret = AOCLFFTZ_INVALID_INPUT;                                          \
        goto validation_exit;                                                  \
    }                                                                          \
    if (problem->out == NULL)                                                  \
    {                                                                          \
        AOCLFFTZ_LOG_UNFORMATTED(ERR, ERR, "output buffer cannot be NULL");    \
        ret = AOCLFFTZ_INVALID_INPUT;                                          \
        goto validation_exit;                                                  \
    }                                                                          \
    if (problem->in == NULL)                                                   \
    {                                                                          \
        AOCLFFTZ_LOG_UNFORMATTED(ERR, ERR, "input buffer cannot be NULL");     \
        ret = AOCLFFTZ_INVALID_INPUT;                                          \
        goto validation_exit;                                                  \
    }                                                                          \
    if (problem->dim_rank <= 0)                                                \
    {                                                                          \
        AOCLFFTZ_LOG_UNFORMATTED(ERR, ERR, "dim rank must be atleast 1");      \
        ret = AOCLFFTZ_INVALID_INPUT;                                          \
        goto validation_exit;                                                  \
    }                                                                          \
    if (problem->vec_rank <= 0)                                                \
    {                                                                          \
        AOCLFFTZ_LOG_UNFORMATTED(ERR, ERR, "vec rank must be atleast 1");      \
        ret = AOCLFFTZ_INVALID_INPUT;                                          \
        goto validation_exit;                                                  \
    }                                                                          \
    VALIDATE_DIMS(problem->dims, problem->dim_rank, ret)                       \
    if (ret)                                                                   \
    {                                                                          \
        goto validation_exit;                                                  \
    }                                                                          \
    VALIDATE_VECS(problem->dims, problem->dim_rank,                            \
                problem->vecs, problem->vec_rank, ret)                         \
    if (ret)                                                                   \
    {                                                                          \
        goto validation_exit;                                                  \
    }                                                                          \
    if ((ret = validate_control_params(&(problem->cntrl_params))) != 0)        \
    {                                                                          \
        goto validation_exit;                                                  \
    }                                                                          \
    if ((ret = validate_flags(problem->flags)) != 0)                           \
    {                                                                          \
        goto validation_exit;                                                  \
    }                                                                          \
    if (!IS_OUT_OF_PLACE(problem->flags))                                      \
    {                                                                          \
        VALIDATE_INPLACE_STRIDES(problem->dims,problem->vecs,                  \
                                    problem->dim_rank, problem->vec_rank, ret) \
        if (ret)                                                               \
        {                                                                      \
            AOCLFFTZ_LOG_UNFORMATTED(ERR, ERR, "Input & Output strides"        \
                                     " in Inplace problem must be equal");     \
            goto validation_exit;                                              \
        }                                                                      \
    }                                                                          \
    validation_exit:                                                           \
    if (ret)                                                                   \
    {                                                                          \
        AOCLFFTZ_LOG_FORMATTED(ERR, ERR, "Problem descriptor validation "      \
                                        "failed with error code : %d", ret);   \
        return NULL;                                                           \
    }                                                                          \
}

#endif // VALIDATE_PROBLEM_H
