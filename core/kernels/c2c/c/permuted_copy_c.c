// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

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
                       aoclfftz_strides_t *strides, UINT8 data_stride);

kdata_ register_kernel_permuted_copy_c(INT32 precision)
{
    if (precision == DT_FLOAT)
    {
        return permuted_copy_c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return permuted_copy_c_fp64;
    }
    else
    {
        return NULL;
    }
}

VOID permuted_copy_c_fp64(VOID *in, VOID *out, INTP n, INTP size,
                          aoclfftz_strides_t *strides, UINT8 data_stride)
{
    DOUBLE *in_d = (DOUBLE *)in;
    DOUBLE *out_d = (DOUBLE *)out;
    // FIXME: remove the condition to pick the strides, will be revamped in
    // permute solver patch
    INTP in_stride = strides->in_strides[0] ?
             strides->in_strides[0] * data_stride : strides->in_strides[1];
    INTP out_stride = strides->out_strides[0] ?
             strides->out_strides[0] * data_stride : strides->out_strides[1];
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;
    // iterates over the number of offsets (n)
    for (INTP i = 0; i < n; i++)
    {
        // iterates over the size of each offset (size)
        for (INTP j = 0; j < size; j++)
        {
            for (INTP k = 0; k < data_stride; k++)
            {
                out_d[j * out_stride + k] = in_d[j * in_stride + k];
            }
        }
        in_d += v_in_stride;
        out_d += v_out_stride;
    }
}

VOID permuted_copy_c_fp32(VOID *in, VOID *out, INTP n, INTP size,
                          aoclfftz_strides_t *strides, UINT8 data_stride)
{
    FLOAT *in_f = (FLOAT *)in;
    FLOAT *out_f = (FLOAT *)out;
    // FIXME: remove the condition to pick the strides, will be revamped in
    // permute solver patch
    INTP in_stride = strides->in_strides[0] ?
             strides->in_strides[0] * data_stride : strides->in_strides[1];
    INTP out_stride = strides->out_strides[0] ?
             strides->out_strides[0] * data_stride : strides->out_strides[1];
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;
    // iterates over the number of offsets (n)
    for (INTP i = 0; i < n; i++)
    {
        // iterates over the size of each offset (size)
        for (INTP j = 0; j < size; j++)
        {
            for (INTP k = 0; k < data_stride; k++)
            {
                out_f[j * out_stride + k] = in_f[j * in_stride + k];
            }
        }
        in_f += v_in_stride;
        out_f += v_out_stride;
    }
}
