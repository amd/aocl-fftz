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

#include "aoclfftz.h"
#include "selector/selector.h"
#include "utils/utils.h"

// checks if input & output strides are the same for of an inplace problem
// TODO: Fix for R2C/C2R ND problems
#define VALIDATE_INPLACE_STRIDES(dims, vecs, dim_rank, vec_rank, flags, errno) \
{                                                                              \
    if (!flags.fft_type)                                                       \
    {   /* Validate C2C dims */                                                \
        for (INT32 i = 0; i < dim_rank; i++)                                   \
        {                                                                      \
            if (dims[i].in_stride != dims[i].out_stride)                       \
            {                                                                  \
                AOCLFFTZ_ERROR("For dimension[%d]:  "                          \
                    "in_stride (%d) != out_stride (%d)",                       \
                    i, (INT32)dims[i].in_stride, (INT32)dims[i].out_stride);   \
                errno = AOCLFFTZ_INVALID_INPUT;                                \
                goto validation_exit;                                          \
            }                                                                  \
        }                                                                      \
        /* Validate C2C vecs */                                                \
        for (INT32 i = 0; i < vec_rank; i++)                                   \
        {                                                                      \
            if (vecs[i].in_stride != vecs[i].out_stride)                       \
            {                                                                  \
                AOCLFFTZ_ERROR("For vecs[%d]:  "                               \
                    "in_stride (%d) != out_stride (%d)",                       \
                    i, (INT32)vecs[i].in_stride, (INT32)vecs[i].out_stride);   \
                errno = AOCLFFTZ_INVALID_INPUT;                                \
                goto validation_exit;                                          \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        INTP in_scale = (flags.fft_direction == FORWARD_FFT_DIR) ? 2 : 1;      \
        INTP out_scale = (flags.fft_direction == BACKWARD_FFT_DIR) ? 2 : 1;    \
        /* Validate R2C/C2R dims */                                            \
        /* For R2C, in_stride should be the double of out_stride */            \
        /* (or vice versa for C2R) except for dims[0] for which in_stride */   \
        /* and out_stride can either be equal or in_stride can be double of */ \
        /* out_stride (or vice versa  for C2R)*/                               \
        for (INT32 i = 0; i < dim_rank; i++)                                   \
        {                                                                      \
            if ((dims[i].in_stride * out_scale !=                              \
                      dims[i].out_stride * in_scale))                          \
            {                                                                  \
                if (i == 0)                                                    \
                {                                                              \
                    if (dims[i].in_stride != dims[i].out_stride)               \
                    {                                                          \
                        if (flags.fft_direction == FORWARD_FFT_DIR)            \
                        {                                                      \
                            AOCLFFTZ_ERROR("For dimension[%d]: "               \
                                "Strides should either be equal or"            \
                                "in_stride should be equal to out_stride * %d",\
                                i, (INT32)in_scale);                           \
                        }                                                      \
                        else /* flags.fft_direction == BACKWARD_FFT_DIR */     \
                        {                                                      \
                            AOCLFFTZ_ERROR("For dimension[%d]: "               \
                                "Strides should either be equal or"            \
                                "out_stride should be equal to in_stride * %d",\
                                i, (INT32)in_scale);                           \
                        }                                                      \
                        errno = AOCLFFTZ_INVALID_INPUT;                        \
                        goto validation_exit;                                  \
                    }                                                          \
                }                                                              \
                else if (flags.fft_direction == FORWARD_FFT_DIR)               \
                {                                                              \
                    AOCLFFTZ_ERROR("For dimension[%d]: "                       \
                        "in_stride != out_stride * %d\n"                       \
                        "        Possible dims in_stride "                     \
                        "and dims out_stride values are %d, %d",               \
                        i, (INT32)in_scale,                                    \
                        (INT32)(dims[i].out_stride * in_scale),                \
                        (INT32)(dims[i].out_stride));                          \
                    errno = AOCLFFTZ_INVALID_INPUT;                            \
                    goto validation_exit;                                      \
                }                                                              \
                else                                                           \
                {                                                              \
                    AOCLFFTZ_ERROR("For dimension[%d]: "                       \
                        "in_stride * %d != out_stride\n"                       \
                        "        Possible dims in_stride "                     \
                        "and dims out_stride values are %d, %d",               \
                        i, (INT32)out_scale,                                   \
                        (INT32)(dims[i].in_stride),                            \
                        (INT32)(dims[i].in_stride * out_scale));               \
                    errno = AOCLFFTZ_INVALID_INPUT;                            \
                    goto validation_exit;                                      \
                }                                                              \
            }                                                                  \
        }                                                                      \
        /* Validate R2C/C2R vecs */                                            \
        for (INT32 i = 0; i < vec_rank; i++)                                   \
        {                                                                      \
            /* For R2C, in_stride should be the double of out_stride */        \
            /* (or vice versa for C2R) except when vec_rank is 1 and */        \
            /* vecs[0].n = 1 i.e non-batched problem*/                         \
            if ((vecs[i].in_stride * out_scale !=                              \
                 vecs[i].out_stride * in_scale) &&                             \
                 !((i == 0) && (vecs[i].n == 1)))                              \
            {                                                                  \
                if (flags.fft_direction == FORWARD_FFT_DIR)                    \
                {                                                              \
                    AOCLFFTZ_ERROR("For vecs[%d]: "                            \
                        "in_stride != out_stride * %d\n"                       \
                        "        Possible vecs in_stride "                     \
                        "and vecs out_stride values are %d, %d",               \
                        i, (INT32)in_scale,                                    \
                        (INT32)(vecs[i].out_stride * in_scale),                \
                        (INT32)(vecs[i].out_stride));                          \
                }                                                              \
                else                                                           \
                {                                                              \
                    AOCLFFTZ_ERROR("For vecs[%d]: "                            \
                        "in_stride * %d != out_stride\n"                       \
                        "        Possible vecs in_stride "                     \
                        "and vecs out_stride values are %d, %d",               \
                        i, (INT32)out_scale,                                   \
                        (INT32)(vecs[i].in_stride),                            \
                        (INT32)(vecs[i].in_stride * out_scale));               \
                }                                                              \
                errno = AOCLFFTZ_INVALID_INPUT;                                \
                goto validation_exit;                                          \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

#define VALIDATE_BUFFERS(in, out, out_of_place, errno)                         \
{                                                                              \
    if (!out_of_place &&  (in != out))                                         \
    {                                                                          \
        AOCLFFTZ_ERROR("Input and output buffer must be the "                  \
                                   "same for in-place problems");              \
        errno = AOCLFFTZ_INVALID_INPUT;                                        \
        goto validation_exit;                                                  \
    }                                                                          \
    else if (out_of_place && (in == out))                                      \
    {                                                                          \
        AOCLFFTZ_ERROR("Input and output buffer cannot be the "                \
                                   "same for out-of-place problems");          \
        errno = AOCLFFTZ_INVALID_INPUT;                                        \
        goto validation_exit;                                                  \
    }                                                                          \
}

static inline INT32 validate_flags(aoclfftz_flags_t *flags)
{
    if (flags->fft_type > 1)
    {
        AOCLFFTZ_ERROR("fft_type can be Complex(0) or Real(1)");
        return AOCLFFTZ_INVALID_INPUT;
    }
    if (flags->fft_direction > 1)
    {
        AOCLFFTZ_ERROR("fft_direction can be Forward(0) or Backward(1)");
        return AOCLFFTZ_INVALID_INPUT;
    }
    // TODO: Remove validation once support for out-of-order output is added
    if (flags->storage_order)
    {
        AOCLFFTZ_ERROR("Library does not support "
                                   "out-of-order outputs");
        return AOCLFFTZ_INVALID_INPUT;
    }
    if (flags->fft_placement > 1)
    {
        AOCLFFTZ_ERROR("fft_placement can be In-place(0) or Out-of-place(1)");
        return AOCLFFTZ_INVALID_INPUT;
    }
    // TODO: Remove validation once support for standalone transpose is added
    if (flags->transpose_mode)
    {
        AOCLFFTZ_ERROR("Library does not support "
                                   "standalone transpose");
        return AOCLFFTZ_INVALID_INPUT;
    }
    if (flags->bit_reproducibility > 1)
    {
        AOCLFFTZ_ERROR("bit_reproducibility can be Disable(0) or Enable(1)");
        return AOCLFFTZ_INVALID_INPUT;
    }

    return AOCLFFTZ_SUCCESS;
}

#define VALIDATE_DIMS(dims, dim_rank, errno)                                   \
{                                                                              \
    if (dims == NULL)                                                          \
    {                                                                          \
        AOCLFFTZ_ERROR("Dims cannot be NULL\n");                               \
        errno = AOCLFFTZ_INVALID_INPUT;                                        \
        goto validation_exit;                                                  \
    }                                                                          \
    for (INT32 i = 0; i < dim_rank; i++)                                       \
    {                                                                          \
        if (dims[i].n <= 0)                                                    \
        {                                                                      \
            AOCLFFTZ_ERROR("Dimension[%d]: size "                              \
                                     "must be atleast 1", i);                  \
            errno = AOCLFFTZ_INVALID_INPUT;                                    \
            goto validation_exit;                                              \
        }                                                                      \
        if (dims[i].in_stride <= 0)                                            \
        {                                                                      \
            AOCLFFTZ_ERROR("Dimension[%d]: in_stride must be "                 \
                                     "greater than zero\n", i);                \
            errno = AOCLFFTZ_INVALID_INPUT;                                    \
            goto validation_exit;                                              \
        }                                                                      \
        if (dims[i].out_stride <= 0)                                           \
        {                                                                      \
            AOCLFFTZ_ERROR("Dimension[%d]: out_stride must be "                \
                                     "greater than zero\n", i);                \
            errno = AOCLFFTZ_INVALID_INPUT;                                    \
            goto validation_exit;                                              \
        }                                                                      \
    }                                                                          \
}

#define VALIDATE_VECS(dims, dim_rank, vecs, vec_rank, errno)                   \
{                                                                              \
    if (vecs == NULL)                                                          \
    {                                                                          \
        AOCLFFTZ_ERROR("Vecs cannot be NULL");                                 \
        errno = AOCLFFTZ_INVALID_INPUT;                                        \
        goto validation_exit;                                                  \
    }                                                                          \
    for (INT32 i = 0; i < vec_rank; i++)                                       \
    {                                                                          \
        if (vecs[i].n <= 0)                                                    \
        {                                                                      \
            AOCLFFTZ_ERROR("Vector[%d]: size must be at least 1", i);          \
            errno = AOCLFFTZ_INVALID_INPUT;                                    \
            goto validation_exit;                                              \
        }                                                                      \
        if (vecs[i].in_stride <= 0)                                            \
        {                                                                      \
            AOCLFFTZ_ERROR("Vector[%d]: in_stride must be greater "            \
                                     "than zero\n", i);                        \
            errno = AOCLFFTZ_INVALID_INPUT;                                    \
            goto validation_exit;                                              \
        }                                                                      \
        if (vecs[i].out_stride <= 0)                                           \
        {                                                                      \
            AOCLFFTZ_ERROR("Vector[%d]: out_stride must be greater "           \
                                     "than zero\n", i);                        \
            errno = AOCLFFTZ_INVALID_INPUT;                                    \
            goto validation_exit;                                              \
        }                                                                      \
    }                                                                          \
}

static inline INT32 sanitize_threads(INT32 num_threads)
{
#ifdef MULTI_THREADING
    INT32 max_threads = omp_get_num_procs();
    if (num_threads < 1)
    {
        AOCLFFTZ_LOG(INFO, global_logger_mode, "Requested num_threads value "
             "(%d) is less than minimum required value (1), defaulting to "
             "single threaded execution", num_threads);
        return 1;
    }
    else if (num_threads > max_threads)
    {
        AOCLFFTZ_LOG(INFO, global_logger_mode, "Requested num_threads "
             "(%d) exceeds available logical CPUs (%d), using %d\n as "
             "num_threads", num_threads, max_threads, max_threads);
        return max_threads;
    }
#else
    if (num_threads != 1)
    {
        AOCLFFTZ_LOG(INFO, global_logger_mode, "Multi-Threading Disabled !! "
             "Running in single threaded mode, to use multi-threaded FFT "
             "Please enable Multi-threading at compile time");
        return 1;
    }
#endif
    return num_threads;
}

static inline INT32 validate_control_params(aoclfftz_cntrl_params_t *cntrl_p)
{
    if (cntrl_p->logger_mode < 0 || cntrl_p->logger_mode > 3)
    {
        AOCLFFTZ_LOG(INFO, global_logger_mode, "Invalid logger mode, "
                                 "running with default logger mode (0)");
        // set to default
        cntrl_p->logger_mode = 0;
    }
    if (cntrl_p->measure_stats != 0)
    {
        AOCLFFTZ_LOG(INFO, global_logger_mode,
                     "measure-stats opt is currently not supported, "
                     "running with measure-stats disabled");
        // set to default
        cntrl_p->measure_stats = 0;
    }
    if (!cntrl_p->opt_off)
    {
        if (cntrl_p->opt_level < 0 || cntrl_p->opt_level > 3)
        {
            AOCLFFTZ_LOG(INFO, global_logger_mode,
                         "Either the opt-level is not set or out of range "
                         "[0-3]. Running with highest opt level");
            // Setting to highest opt level
            cntrl_p->opt_level = 3;
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
        AOCLFFTZ_ERROR("Output buffer cannot be NULL");                        \
        ret = AOCLFFTZ_INVALID_INPUT;                                          \
        goto validation_exit;                                                  \
    }                                                                          \
    if (problem->in == NULL)                                                   \
    {                                                                          \
        AOCLFFTZ_ERROR("Input buffer cannot be NULL");                         \
        ret = AOCLFFTZ_INVALID_INPUT;                                          \
        goto validation_exit;                                                  \
    }                                                                          \
    if (problem->dim_rank <= 0)                                                \
    {                                                                          \
        AOCLFFTZ_ERROR("Dim rank must be atleast 1");                          \
        ret = AOCLFFTZ_INVALID_INPUT;                                          \
        goto validation_exit;                                                  \
    }                                                                          \
    if (problem->vec_rank <= 0)                                                \
    {                                                                          \
        AOCLFFTZ_ERROR("Vec rank must be atleast 1");                          \
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
    if ((ret = validate_flags(&(problem->flags))) != 0)                        \
    {                                                                          \
        goto validation_exit;                                                  \
    }                                                                          \
    if (!problem->flags.fft_placement)                                         \
    {                                                                          \
        VALIDATE_BUFFERS(problem->in, problem->out, 0 /* in-place */, ret)     \
        VALIDATE_INPLACE_STRIDES(problem->dims,problem->vecs,                  \
                                 problem->dim_rank, problem->vec_rank,         \
                                 problem->flags, ret)                          \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        VALIDATE_BUFFERS(problem->in, problem->out, 1 /* out_of_place */, ret) \
    }                                                                          \
    if (problem->pthr_fft.dynamic_load_model != 0)                             \
    {                                                                          \
        AOCLFFTZ_LOG(INFO, global_logger_mode,                                 \
                "dynamic_load_model is currently unsupported, "                \
                "disabling it");                                               \
        problem->pthr_fft.dynamic_load_model = 0;                              \
    }                                                                          \
    if ((problem->pthr_fft.num_threads != 1))                                  \
    {                                                                          \
        problem->pthr_fft.num_threads =                                        \
            sanitize_threads(problem->pthr_fft.num_threads);                   \
    }                                                                          \
    validation_exit:                                                           \
    if (ret)                                                                   \
    {                                                                          \
        AOCLFFTZ_ERROR("Problem descriptor validation failed "                 \
                                 "with %s", get_status_string(ret));           \
        return NULL;                                                           \
    }                                                                          \
}

#endif // VALIDATE_PROBLEM_H
