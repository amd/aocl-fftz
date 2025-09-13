/**
 * Copyright (C) 2024-2025, Advanced Micro Devices. All rights reserved.
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

/** @file dft_reference.h
 *
 *  @brief DFT reference implementation.
 *
 *  This file contains the implementation of DFT reference and its helper
 *  functions.
 *
 *  @author V. Murugan
 *  @author Srirammaswamy Srinivasan
 *  @author Jeya R
 */

#ifndef DFT_REFERENCE_H
#define DFT_REFERENCE_H

#include <math.h>
#include "test/bench/aoclfftz_bench.h"

#ifdef ENABLE_DFT_REFERENCE
/**
 * @brief angle = angle * [(i0+k0)/n0 + (i1+k1)/n1 + ... + (iR+kR)/nR]
 * where R = dim rank
 *
 */
#define UPDATE_ANGLE(angle, in_arr_idx, out_arr_idx, inv_dims, rank)           \
    {                                                                          \
        DOUBLE x = 0.0;                                                        \
        for (INTP i = 0; i < rank; i++)                                        \
        {                                                                      \
            x += (((DOUBLE)in_arr_idx[i] * out_arr_idx[i]) * inv_dims[i]);     \
        }                                                                      \
        angle = angle * x;                                                     \
    }

// Buffer management structure
typedef struct
{
    VOID *complex_in;  // For R2C/C2R: real->complex conversion
    VOID *complex_out; // For R2C/C2R: half-complex->complex conversion
    VOID *in_ref;      // DFT reference input buffer
    VOID *out_ref;     // DFT reference output buffer
} dft_ref_buffers_t;

INT32 run_dft_reference_test(aoclfftz_bench_params_t *params, INTP *in_idx_map,
                             INTP *out_idx_map, VOID *handle,
                             VOID *input_buffer);
VOID dft_ref_f(aoclfftz_bench_params_t *params, VOID *in, VOID *out,
               INTP *in_idx_map, INTP *out_idx_map);
VOID dft_ref_d(aoclfftz_bench_params_t *params, VOID *in, VOID *out,
               INTP *in_idx_map, INTP *out_idx_map);
INT32 allocate_dftref_buffers(aoclfftz_bench_params_t *params,
                              dft_ref_buffers_t *buffers, UINT32 is_align);
INT32 execute_fft_and_postprocess(aoclfftz_bench_params_t *params,
                                  dft_ref_buffers_t *buffers, VOID *handle,
                                  INTP *in_idx_map, INTP *out_idx_map);
VOID cleanup_buffers(aoclfftz_bench_params_t *params,
                     dft_ref_buffers_t *buffers, UINT32 is_align);

#endif
#endif // DFT_REFERENCE_H
