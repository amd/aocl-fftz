// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file strides.c
 *
 *  @brief Stride utility functions.
 *
 *  This file contains the function definitions related to strided-memcpy,
 *  computing and manipulating stride array values for real, complex and
 *  half-complex data.
 *
 *  @author Srirammaswamy Srinivasan
 */

#include "core/common/strides.h"

/**
 * Prepare stride array values for real/complex and half-complex data
 *
 * Example with stride_val = 2 and n = 6
 *
 * Case 1: With compute_half_complex = 0 (adjust_to_full_complex doesn't
 * matter):
 *
 * strides array values: 0, 2, 4, 6, 8, 10
 * which represents: 0, x, 2, x, 4, x, 6, x, 8, x, 10
 *
 * Case 2: With compute_half_complex = 1 and adjust_to_full_complex = 0:
 *
 * strides array values: 0, 3, 4, 7, 8, 11
 * which represents: 0*, (x, x), (3, 4), (x, x), (7, 8), (x, x), 11*
 *
 * *here the buffer contains real and complex data,
 *  where the first and last points are real, remaining are complex
 *
 * Case 3: With compute_half_complex = 1  and adjust_to_full_complex = 1:
 *
 * strides array values: 0, 3, 4, 7, 8, 11
 * which represents: (0, -*), (x, x), (4, 5), (x, x), (8, 9), (x, x), (12, -*)
 *
 *  *here the buffer contains only complex data, so the first and last real
 *   numbers are stored in complex form, hence the indices are adjusted
 *   accordingly
 *
 * @param strides stride array to the prepared
 * @param stride_val stride value to be populated
 * @param n stride array size
 * @param compute_half_complex 1 for half-complex, 0 for real & complex data
 * @param adjust_to_full_complex 1 if the stride values needs to be adjusted to
 *                               full complex format, 0 otherwise
 *                               adjust_to_full_complex needs to be set for the
 *                               direct sizes and final level of CT
 * @return FFTZ_VOID
 */
FFTZ_VOID populate_stride_array(FFTZ_INTP *strides, FFTZ_INTP stride_val,
                                FFTZ_INTP n, FFTZ_UINT8 compute_half_complex,
                                FFTZ_UINT8 adjust_to_full_complex)
{
    if (compute_half_complex)
    {
        FFTZ_INTP offset = adjust_to_full_complex ? 1 : 0;
        FFTZ_INTP nby2_ceil = (n + 1) / 2;

        // first stride value
        strides[0] = 0;

        // inbetween stride values
        for (FFTZ_INTP i = 1; i < nby2_ceil; i++)
        {
            FFTZ_INTP cur_stride = i * stride_val * 2 + offset;
            strides[i * 2 - 1] = cur_stride - 1;
            strides[i * 2] = cur_stride;
        }

        // last stride value
        if (n % 2 == 0)
        {
            strides[n - 1] = stride_val * n - 1 + offset;
        }
    }
    else
    {
        for (FFTZ_INTP i = 0; i < n; i++)
        {
            strides[i] = i * stride_val;
        }
    }
}

/**
 * @brief Prepare the stride array for C2C Kernel in Real Problem
 *
 * The stride array of C2C kernel in real problem needs to be rearranged in
 * such a way that the second half of the strides will be reversed to its
 * complex conjugate point.
 *
 * This function will prepare rearranged strides for the first iteration in a
 * CT stage and for the remaining iterations, the strides will be adjusted by
 * subtracting a constant value during execution.
 *
 * Example for radix 6 C2C kernel with stride complex stride 4 (i.e. stride in
 * data = 8):
 *
 * full complex data    : x1------x2------x3------x4------x5------x6
 * half complex buffer  : |----------------------|
 * half complex data    : x1--y6--x2--y4--x3--y3-- (here y refers to complex
 * conjugate of x)
 *
 * full complex strides : 0, 8, 16, 24, 32, 40
 * half complex strides : 0, 8, 16, 20, 12, 4
 *
 * @param strides stride array
 * @param radix radix value i.e. length of the stride array
 * @param n length of the data buffer
 * @param stride stride value for the given buffer
 * @return FFTZ_VOID
 */
FFTZ_VOID prepare_real_c2c_kernel_strides(FFTZ_INTP *in, FFTZ_INTP *out,
                                          FFTZ_INTP radix, FFTZ_INTP n,
                                          FFTZ_INTP stride)
{
    // align stride to complex points
    stride *= 2;
    for (FFTZ_INTP i = 0; i < radix; i++)
    {
        FFTZ_INTP a = in[i] / stride + 1;
        if (a > n / 2)
        {
            a = n - a;
            out[i] = (a - 1) * stride;
        }
        else
        {
            out[i] = in[i];
        }
    }
}

/**
 * @brief Prepare the strides for R2HCF kernels by fusing strides of standard
 *        and shifted kernels in interleaved order
 *
 * Example:
 *  input: 0, 3, 6, 9 (which are standard kernel strides)
 *  radix: 4 (length of the strides for one kernel variant)
 *  if offset = 2, then the shifted kernel strides will be: 2, 5, 8, 11
 *  fused strides: 0, 2, 3, 5, 6, 8, 9, 11 (output)
 *
 * @param strides strides data to be reordered for R2HCF kernels
 * @param radix radix of the kernel
 * @param offset distance between the data point of standard and shifted kernels
 *               within one R2HCF kernel
 * @return FFTZ_VOID
 */

// TODO: change the order of fused strides from interleaved to split order
//       i.e. 0, 2, 3, 5, 6, 8, 9, 11 ---> 0, 3, 6, 9, 2, 5, 8, 11
//       it is required since the fused kernels work in this format
FFTZ_VOID prepare_fused_kernel_strides(FFTZ_INTP *strides, FFTZ_INTP radix,
                                       FFTZ_INTP offset)
{
    for (FFTZ_INTP i = radix - 1; i >= 0; i--)
    {
        FFTZ_INTP stride = strides[i];
        strides[i * 2] = stride;
        strides[i * 2 + 1] = stride + offset;
    }
}
