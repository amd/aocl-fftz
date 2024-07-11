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
 *  This file contains the function definitions related to computing and
 *  manipulating stride array values for real, complex and half-complex data.
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
 * @brief Rearrange the stride array for C2C Kernel in Real Problem
 *
 * The stride array of C2C kernel in real problem needs to be rearranged in
 * such a way that the second half of the strides will be reversed.
 *
 * Example for radix = 6 strides:
 * data buffer size   : |-----------------|
 * generated strides  : 1-----2-----3-----4-----5-----6
 * rearranged strides : 1--6--2--5--3--4--
 *
 * @param strides stride array
 * @param radix radix value i.e. length of the stride array
 * @param n length of the data buffer
 * @return VOID
 */
VOID rearrange_stride_array(INTP *strides, INTP radix, INTP n)
{
    INTP offset = strides[1];
    for (INTP i = 2; i < radix; i++)
    {
        if (strides[i] >= n)
        {
            strides[i] = (strides[i] - n) + offset;
        }
    }
}
