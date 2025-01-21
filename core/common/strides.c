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
 * Case 1: With compute_half_complex = 0 (adjust_to_full_complex doesn't matter):
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
 * @return VOID
 */
VOID populate_stride_array(INTP *strides, INTP stride_val, INTP n,
                           UINT8 compute_half_complex,
                           UINT8 adjust_to_full_complex)
{
    if (compute_half_complex)
    {
        INTP offset = adjust_to_full_complex ? 1 : 0;
        INTP nby2_ceil = (n + 1) / 2;

        // first stride value
        strides[0] = 0;

        // inbetween stride values
        for (INTP i = 1; i < nby2_ceil; i++)
        {
            INTP cur_stride = i * stride_val * 2 + offset;
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
        for (INTP i = 0; i < n; i++)
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
 * Example for radix 6 C2C kernel with stride complex stride 4 (i.e. stride in data = 8):
 *
 * full complex data    : x1------x2------x3------x4------x5------x6
 * half complex buffer  : |----------------------|
 * half complex data    : x1--y6--x2--y4--x3--y3-- (here y refers to complex conjugate of x)
 *
 * full complex strides : 0, 8, 16, 24, 32, 40
 * half complex strides : 0, 8, 16, 20, 12, 4
 *
 * @param strides stride array
 * @param radix radix value i.e. length of the stride array
 * @param n length of the data buffer
 * @param stride stride value for the given buffer
 * @return VOID
 */
VOID prepare_real_c2c_kernel_strides(INTP *in, INTP *out, INTP radix,
                                     INTP n, INTP stride)
{
    // align stride to complex points
    stride *= 2;
    for (INTP i = 0; i < radix; i++)
    {
        INTP a = in[i] / stride + 1;
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
 * @return VOID
 */

// TODO: change the order of fused strides from interleaved to split order
//       i.e. 0, 2, 3, 5, 6, 8, 9, 11 ---> 0, 3, 6, 9, 2, 5, 8, 11
//       it is required since the fused kernels work in this format
VOID prepare_fused_kernel_strides(INTP *strides, INTP radix, INTP offset)
{
    for (INTP i = radix - 1; i >= 0; i--)
    {
        INTP stride = strides[i];
        strides[i * 2] = stride;
        strides[i * 2 + 1] = stride + offset;
    }
}
