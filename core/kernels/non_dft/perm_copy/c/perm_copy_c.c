// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file perm_copy_c.c
 *
 * @brief Strided permuted copy operations using scalar C code.
 *
 * Provides functions to copy complex data with arbitrary input and output
 * strides. Used for data rearrangement in FFT solvers when non-unit strides
 * are required.
 *
 * The copy operation supports:
 * - Multiple offset groups (n parameter)
 * - Multiple elements per group (size parameter)
 * - Independent input and output strides
 * - Both single and double precision complex data
 *
 * @author Srirammaswamy Srinivasan
 */

#include "core/kernels/kernel.h"
#include "utils/utils.h"

/**
 * @brief Double-precision strided permuted copy.
 *
 * Copies complex double-precision data with arbitrary strides. The operation
 * processes 'n' offset groups, each containing 'size' elements.
 *
 * @param[in]  in           Input buffer
 * @param[out] out          Output buffer
 * @param[in]  n            Number of offset groups to process
 * @param[in]  size         Number of elements per offset group
 * @param[in]  in_stride    Input stride between consecutive elements
 * @param[in]  out_stride   Output stride between consecutive elements
 * @param[in]  v_in_stride  Input stride between offset groups
 * @param[in]  v_out_stride Output stride between offset groups
 */
VOID permuted_copy_c_fp64(VOID *in, VOID *out, INTP n, INTP size,
                          INTP in_stride, INTP out_stride, INTP v_in_stride,
                          INTP v_out_stride)
{
    DOUBLE *p_in = (DOUBLE *)in;
    DOUBLE *p_out = (DOUBLE *)out;

    /* Iterate over offset groups */
    for (INTP i = 0; i < n; i++)
    {
        /* Copy each element within the offset group */
        for (INTP j = 0; j < size; j++)
        {
            INTP in_idx = j * in_stride;
            INTP out_idx = j * out_stride;

            /* Copy real and imaginary parts */
            p_out[out_idx] = p_in[in_idx];
            p_out[out_idx + 1] = p_in[in_idx + 1];
        }

        /* Advance to next offset group */
        p_in += v_in_stride;
        p_out += v_out_stride;
    }
}

/**
 * @brief Single-precision strided permuted copy.
 *
 * Copies complex single-precision data with arbitrary strides.
 * See permuted_copy_c_fp64 for detailed documentation.
 *
 * @param[in]  in           Input buffer
 * @param[out] out          Output buffer
 * @param[in]  n            Number of offset groups to process
 * @param[in]  size         Number of elements per offset group
 * @param[in]  in_stride    Input stride between consecutive elements
 * @param[in]  out_stride   Output stride between consecutive elements
 * @param[in]  v_in_stride  Input stride between offset groups
 * @param[in]  v_out_stride Output stride between offset groups
 */
VOID permuted_copy_c_fp32(VOID *in, VOID *out, INTP n, INTP size,
                          INTP in_stride, INTP out_stride, INTP v_in_stride,
                          INTP v_out_stride)
{
    FLOAT *p_in = (FLOAT *)in;
    FLOAT *p_out = (FLOAT *)out;

    /* Iterate over offset groups */
    for (INTP i = 0; i < n; i++)
    {
        /* Copy each element within the offset group */
        for (INTP j = 0; j < size; j++)
        {
            INTP in_idx = j * in_stride;
            INTP out_idx = j * out_stride;

            /* Copy real and imaginary parts */
            p_out[out_idx] = p_in[in_idx];
            p_out[out_idx + 1] = p_in[in_idx + 1];
        }

        /* Advance to next offset group */
        p_in += v_in_stride;
        p_out += v_out_stride;
    }
}
