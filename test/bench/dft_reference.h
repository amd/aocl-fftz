// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

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
