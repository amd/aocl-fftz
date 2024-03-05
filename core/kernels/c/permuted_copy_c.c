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

/** @file permuted_copy.c
 *
 *  @brief Permuted copy with scalar operations in C
 *
 *  This file contains the permuted copy implementation using scalar operations
 *  for single-precision and double-precision data.
 *
 *  @author Srirammaswamy Srinivasan
 */

#include "core/kernels/kernel.h"
#include "utils/utils.h"

// TODO: Make it common for transpose kernels and move this to
// aoclfftz_internal.h file
typedef VOID (*kdata_)(VOID *in, VOID *out, INTP n, INTP size,
                       aoclfftz_strides_t *strides);

kdata_ register_kernel_permuted_copy_c(INT32 precision)
{
    if (precision == DT_FLOAT)
        return permuted_copy_c_fp32;
    else if (precision == DT_DOUBLE)
        return permuted_copy_c_fp64;
    else
        return NULL;
}

VOID permuted_copy_c_fp64(VOID *in, VOID *out, INTP n, INTP size,
                          aoclfftz_strides_t *strides)
{
    DOUBLE *in_d = (DOUBLE *)in;
    DOUBLE *out_d = (DOUBLE *)out;
    //FIXME: remove the condition to pick the strides, will be revamped in
    // permute solver patch
    INTP in_stride = strides->in_strides[0] ?
             strides->in_strides[0] * DATA_STRIDE : strides->in_strides[1];
    INTP out_stride = strides->out_strides[0] ?
             strides->out_strides[0] * DATA_STRIDE : strides->out_strides[1];
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;
    // iterates over the number of offsets (n)
    for (INTP i = 0; i < n; i++)
    {
        // iterates over the size of each offset (size)
        for (INTP j = 0; j < size; j++)
        {
            out_d[j * out_stride] = in_d[j * in_stride];
            out_d[j * out_stride + 1] = in_d[j * in_stride + 1];
        }
        in_d += v_in_stride;
        out_d += v_out_stride;
    }
}

VOID permuted_copy_c_fp32(VOID *in, VOID *out, INTP n, INTP size,
                          aoclfftz_strides_t *strides)
{
    FLOAT *in_f = (FLOAT *)in;
    FLOAT *out_f = (FLOAT *)out;
    //FIXME: remove the condition to pick the strides, will be revamped in
    // permute solver patch
    INTP in_stride = strides->in_strides[0] ?
             strides->in_strides[0] * DATA_STRIDE : strides->in_strides[1];
    INTP out_stride = strides->out_strides[0] ?
             strides->out_strides[0] * DATA_STRIDE : strides->out_strides[1];
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;
    // iterates over the number of offsets (n)
    for (INTP i = 0; i < n; i++)
    {
        // iterates over the size of each offset (size)
        for (INTP j = 0; j < size; j++)
        {
            out_f[j * out_stride] = in_f[j * in_stride];
            out_f[j * out_stride + 1] = in_f[j * in_stride + 1];
        }
        in_f += v_in_stride;
        out_f += v_out_stride;
    }
}
