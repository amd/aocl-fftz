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

/** @file realfft_utils.h
 *
 *  @brief Declarations for RealFFT utility functions.
 *
 *  This file contains the function declarations and function macros of utility
 *  functions related to RealFFT problems.
 *
 *  @author Srirammaswamy Srinivasan
 */

#ifndef REALFFT_UTILS_H
#define REALFFT_UTILS_H

#include "api/aoclfftz_internal.h"

/* Re-order input buffer for batched in-place real forward (R2C) problems

   Move each batch of data by two times to make enough space for
   complex output in each batch

    example: 2v3 [batch-2, size-3]

    input size = 12 (6 for valid data + 6 for extra padding for output)

    input           : | r1  r2  r3| r4  r5  r6|  x   x   x   x   x   x|
    reordered input : | r1  r2  r3  x   x   x | r4  r5  r6   x   x   x|
    output          : | R1  I1  R2  I2  R3  I3| R4  I4  R5  I5  R6  I6|
 */
#define REORDER_INPUT(sol)                                                     \
{                                                                              \
    INT8 dt_prec = DT_PRECISION_FLAG(sol->decomp_scheme->flags);               \
    INT32 dt_bytes = DT_PRECISION_BYTES(dt_prec);                              \
    INTP n = sol->decomp_scheme->dims[0].n;                                    \
    INTP batches = sol->decomp_scheme->vecs[0].n;                              \
    INTP in_stride = sol->decomp_scheme->dims[0].in_stride;                    \
    INTP n_with_strides = (n + (n - 1) * (in_stride - 1));                     \
    INTP one_batch_size = sol->decomp_scheme->vecs[0].in_stride;               \
    VOID *data = sol->decomp_scheme->in_real;                                  \
                                                                               \
    if (in_stride > 1)                                                         \
    {                                                                          \
        if (dt_prec == DT_FLOAT)                                               \
        {                                                                      \
            FLOAT *in = (FLOAT *)data + batches * one_batch_size / 2;          \
            FLOAT *out = (FLOAT *)data + batches * one_batch_size;             \
            for (INTP b = batches - 1; b > 0; b--)                             \
            {                                                                  \
                in -= one_batch_size / 2;                                      \
                out -= one_batch_size;                                         \
                for (INTP i = 0; i < n_with_strides; i += in_stride)           \
                {                                                              \
                    out[i] = in[i];                                            \
                }                                                              \
            }                                                                  \
        }                                                                      \
        else                                                                   \
        {                                                                      \
            DOUBLE *in = (DOUBLE *)data + batches * one_batch_size / 2;        \
            DOUBLE *out = (DOUBLE *)data + batches * one_batch_size;           \
            for (INTP b = batches - 1; b > 0; b--)                             \
            {                                                                  \
                in -= one_batch_size / 2;                                      \
                out -= one_batch_size;                                         \
                for (INTP i = 0; i < n_with_strides; i += in_stride)           \
                {                                                              \
                    out[i] = in[i];                                            \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        for (INTP b = batches - 1; b > 0; b--)                                 \
        {                                                                      \
            memcpy(MOVE_ADDR(data, b * one_batch_size * dt_bytes),             \
                   MOVE_ADDR(data, b * one_batch_size * dt_bytes / 2),         \
                   n_with_strides * dt_bytes);                                 \
        }                                                                      \
    }                                                                          \
}

VOID compute_conjugates(VOID *data, INTP radix, INTP n, INTP *strides,
                        INTP vec_stride, UINT32 prec);

#endif // REALFFT_UTILS_H
