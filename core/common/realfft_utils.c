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

/** @file realfft_utils.c
 *
 *  @brief RealFFT utility functions.
 *
 *  This file contains the function definitions of utility functions related to
 *  RealFFT problems.
 *
 *  @author Srirammaswamy Srinivasan
 */

#include "core/common/realfft_utils.h"
#include "core/common/strides.h"
#include "core/kernels/kernel.h"

/**
 * @brief Compute complex conjugates for a set of selective points.
 *        Conjugate the second half of the complex numbers in the input buffer.
 *
 * Example:
 * Given an input for radix 4 (4 complex numbers in interleaved format)
 * input  -> (1,  2), (3,  4), (5,  6), (7,  8)
 * output -> (1,  2), (3,  4), (5, -6), (7, -8)
 *
 * @param data in/out data buffer
 * @param radix radix of the C2C kernel
 * @param n batch of the C2C kernels
 * @param strides strides array for the buffer
 * @param vec_stride vector stride for the buffer
 * @param prec precision flag (DT_FLOAT or DT_DOUBLE)
 * @return VOID
 */
VOID compute_conjugates(VOID *data, INTP radix, INTP n, INTP *strides,
                        INTP vec_stride, UINT32 prec)
{
    INTP points = (radix + 1) >> 1; // ceil div
    if (prec == DT_FLOAT)
    {
        FLOAT *data_i = (FLOAT *)data + 1;
        for (INTP i = 0; i < n; i++)
        {
            for (INTP j = points; j < radix; j++)
            {
                data_i[strides[j]] = -data_i[strides[j]];
            }
            data_i += vec_stride;
        }
    }
    else
    {
        DOUBLE *data_i = (DOUBLE *)data + 1;
        for (INTP i = 0; i < n; i++)
        {
            for (INTP j = points; j < radix; j++)
            {
                data_i[strides[j]] = -data_i[strides[j]];
            }
            data_i += vec_stride;
        }
    }
}
