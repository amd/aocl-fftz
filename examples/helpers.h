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

/** @file helpers.h
 *
 *  @brief Helper functions definitions and macros for FFT example to prepare
 *         input data, set default strides for dims and vecs, calculate input
 *         and output buffer sizes.
 *
 *  @author Partiksha
 */

#ifndef HELPERS_H
#define HELPERS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "aoclfftz.h"

// 1 - for real problems, 2 - for complex problems
#define DATA_STRIDE(fft_type) (((fft_type) == 1) ? 1 : 2)
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define ALLOC(ptr, type, n_ele)                                                \
{                                                                              \
    ptr = (type *)malloc(sizeof(type) * n_ele);                                \
    if(ptr == NULL)                                                            \
    {                                                                          \
        printf("\nMemory allocation failed\n");                                \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
    else                                                                       \
    {                                                                          \
        memset(ptr, 0, sizeof(type) * n_ele);                                  \
    }                                                                          \
}

#define PREPARE_RANDOM_INPUT(in, input_size, fft_type, data_type)              \
{                                                                              \
    UINTP data_stride = DATA_STRIDE(fft_type);                                 \
    INTP idx = 0;                                                              \
    for (idx = 0; idx < input_size * data_stride; ++idx)                       \
    {                                                                          \
        in[idx] = (data_type)(20.0 / RAND_MAX) * rand() - 10.0;                \
    }                                                                          \
}

/**
 * @brief Set default stride values for dims and vecs.
 *
 * This function calculates appropriate default stride values for dims and vecs
 * The stride calculations vary based on the FFT type and whether the operation
 * is in-place or out-of-place.
 *
 * Stride Calculation Rules:
 *
 * For dims:
 * - dims[0]: Unit stride (1) for both input and output
 * - dims[1]: Stride calculations based on FFT type:
 *   * consider, n = n, is = in_stride, os = out_stride of dims[0]
 *   ------------------|--------------------|--------------------
 *    Type             | in_stride          | out_stride
 *   ------------------|--------------------|--------------------
 *    C2C              | n * is             | n * os
 *    R2C out-of-place | n * is             | (n/2 + 1) * os
 *    R2C in-place     | (n/2 + 1) * is * 2 | (n/2 + 1) * os     (where is = os)
 *    C2R out-of-place | (n/2 + 1) * is     | n * os
 *    C2R in-place     | (n/2 + 1) * is     | (n/2 + 1) * os * 2 (where is = os)
 *   ------------------|--------------------|--------------------
 * - dims[2] or above: dims[i-1].n * dims[i-1].stride
 *                     (where stride = in_stride or out_stride)
 *
 * For vecs:
 *
 * - vecs[0]: Use n, is, os from the last dim (dims[dim_rank-1]).
 *   - Real, dim_rank > 1: Use full-length strides (n*is, n*os). The half-complex
 *     layout only shortens the fastest changing dimension; the batch stride still 
 *     needs full n or else batches may overlap.
 *   - Real, dim_rank == 1: Stride calculations are as follows:
 *   ------------------|--------------------|--------------------
 *    Type             | in_stride          | out_stride
 *   ------------------|--------------------|--------------------
 *    C2C              | n * is             | n * os
 *    R2C out-of-place | n * is             | (n/2 + 1) * os
 *    R2C in-place     | (n/2 + 1) * is * 2 | (n/2 + 1) * os     (where is = os)
 *    C2R out-of-place | (n/2 + 1) * is     | n * os
 *    C2R in-place     | (n/2 + 1) * is     | (n/2 + 1) * os * 2 (where is = os)
 *   ------------------|--------------------|--------------------
 * - vecs[1] or above: vecs[i-1].n * vecs[i-1].stride
 *                     (where stride = in_stride or out_stride)
 *
 * @param dim_rank rank of the dimensions
 * @param vec_rank rank of the vectors
 * @param dims dims structure to set default strides
 * @param vecs vecs structure to set default strides
 * @param flags fft configuration flags
 * @return VOID
 */
VOID set_default_dims_vecs(aoclfftz_dim_t_64_ *dims, INT32 dim_rank,
                           aoclfftz_dim_t_64_ *vecs, INT32 vec_rank,
                           aoclfftz_flags_t flags)
{
    // Set default strides for dims if not explicitly provided
    UINT8 is_in_place = !flags.fft_placement;
    UINT8 is_forward = !flags.fft_direction;
    for (INT32 i = 0; i < dim_rank; i++)
    {
        INTP def_in_stride = 0;
        INTP def_out_stride = 0;
        if (i == 0)
        {
            def_in_stride = 1;
            def_out_stride = 1;
        }
        else if (i == 1)
        {
            if (flags.fft_type == 1 /* Real */)
            {
                if (is_forward)
                {
                    if (is_in_place)
                    {
                        def_in_stride = (dims[0].n / 2 + 1) * dims[0].in_stride * 2;
                        def_out_stride = (dims[0].n / 2 + 1) * dims[0].out_stride;
                    }
                    else 
                    {
                        def_in_stride = dims[0].n * dims[0].in_stride;
                        def_out_stride = (dims[0].n / 2 + 1) * dims[0].out_stride;
                    }
                }
                else
                {   
                    if (is_in_place)
                    {
                        def_in_stride = (dims[0].n / 2 + 1) * dims[0].in_stride;
                        def_out_stride = (dims[0].n / 2 + 1) * dims[0].out_stride * 2;
                    }
                    else
                    {
                        def_in_stride = (dims[0].n / 2 + 1) * dims[0].in_stride;
                        def_out_stride = dims[0].n * dims[0].out_stride;
                    }
                }
            }
            else /* Complex */
            {
                def_in_stride = dims[0].n * dims[0].in_stride;
                def_out_stride = dims[0].n * dims[0].out_stride;
            }
        }
        else /* i > 1 */
        {
            def_in_stride = dims[i - 1].n * dims[i - 1].in_stride;
            def_out_stride = dims[i - 1].n * dims[i - 1].out_stride;
        }
        if (dims[i].in_stride == 0)
        {
            /* in stride for dims[i] is set to default value */
            dims[i].in_stride = def_in_stride;
        }
        if (dims[i].out_stride == 0)
        {
            /* out stride for dim[i] is set to default value */
            dims[i].out_stride = def_out_stride;
        }
    }

    // set strides for vecs if not provided
    for (INT32 i = 0; i < vec_rank; i++)
    {
        INTP def_in_stride = 0;
        INTP def_out_stride = 0;
        if (i == 0)
        {
            aoclfftz_dim_t_64_ last_dim = dims[dim_rank - 1];
            if (flags.fft_type == 1 /* Real */)
            {
                // For dim_rank == 1, apply half-complex modification to vecs.
                // For dim_rank > 1, use last_dim.n so batches don't overlap.
                if (dim_rank == 1)
                {
                    if (is_forward) 
                    {
                        if (is_in_place) /* R2C in-place */
                        {
                            def_in_stride =
                                (last_dim.n / 2 + 1) * last_dim.in_stride * 2;
                            def_out_stride =
                                (last_dim.n / 2 + 1) * last_dim.out_stride;
                        }
                        else /* R2C out-of-place */
                        {
                            def_in_stride = last_dim.n * last_dim.in_stride;
                            def_out_stride = (last_dim.n / 2 + 1) * last_dim.out_stride;
                        }
                    }
                    else
                    {
                        if (is_in_place) /* C2R in-place */
                        {
                            def_in_stride = (last_dim.n / 2 + 1) * last_dim.in_stride;
                            def_out_stride = (last_dim.n / 2 + 1) * last_dim.out_stride * 2;
                        }
                        else /* C2R out-of-place */
                        {
                            def_in_stride = (last_dim.n / 2 + 1) * last_dim.in_stride;
                            def_out_stride = last_dim.n * last_dim.out_stride;
                        }
                    }
                }
                else /* dim_rank > 1 */
                {
                    def_in_stride = last_dim.n * last_dim.in_stride;
                    def_out_stride = last_dim.n * last_dim.out_stride;
                }
            }
            else /* Complex */
            {
                def_in_stride = last_dim.n * last_dim.in_stride;
                def_out_stride = last_dim.n * last_dim.out_stride;
            }
        }
        else /* i > 0 */
        {
            def_in_stride = vecs[i - 1].n * vecs[i - 1].in_stride;
            def_out_stride = vecs[i - 1].n * vecs[i - 1].out_stride;
        }
        if (vecs[i].in_stride == 0)
        {
            /* in stride for vec[i] is set to default value */
            vecs[i].in_stride = def_in_stride;
        }
        if (vecs[i].out_stride == 0)
        {
            /* out stride for vec[i] is set to default value */
            vecs[i].out_stride = def_out_stride;
        }
    }
}

/**
 * @brief calculate the buffer sizes for input and output
 *
 * @param dims dims structure to set default strides
 * @param dim_rank rank of the dimensions
 * @param vecs vecs structure to set default strides
 * @param vec_rank rank of the vectors
 * @param in_buffer_size calculated size of input
 * @param out_buffer_size calculated size of output
 * @return VOID
 */
VOID calculate_buffer_sizes(aoclfftz_dim_t_64_ *dims, INT32 dim_rank,
                            aoclfftz_dim_t_64_ *vecs, INT32 vec_rank,
                            UINTP *in_buffer_size, UINTP *out_buffer_size)
{
    // Example: for an 1D problem with 1D batch
    // Problem size : 3:6:6v4:1:1
    // Data arrangement considered :
    // [1, 2, 3, 4]<0, 0>[5, 6, 7, 8]<0, 0>[9, 10, 11, 12]
    // <---vec stride--->
    // <-------------(Batches -1)---------><--- Problem size * dim_stride --->
    // ((Batches -1) * (vec_stride)) + (Problem size * dim stride)
    UINTP in_size = 1;
    UINTP out_size = 1;
    for (INT32 i = 0; i < dim_rank; i++)
    {
        in_size += ((dims[i].n - 1) * (dims[i].in_stride));
        out_size += ((dims[i].n - 1) * (dims[i].out_stride));
    }
    for (INT32 i = 0; i < vec_rank; i++)
    {
        in_size += ((vecs[i].n - 1) * (vecs[i].in_stride));
        out_size += ((vecs[i].n - 1) * (vecs[i].out_stride));
    }
    *in_buffer_size = in_size;
    *out_buffer_size = out_size;
}

#endif // HELPERS_H
