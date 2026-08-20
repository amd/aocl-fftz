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
        FFTZ_DOUBLE x = 0.0; \
        for (FFTZ_INTP i = 0; i < rank; i++) \
        {                                                                      \
            x += (((FFTZ_DOUBLE)in_arr_idx[i] *                                \
                   out_arr_idx[i]) * inv_dims[i]);                             \
        }                                                                      \
        angle = angle * x;                                                     \
    }

// Buffer management structure
typedef struct
{
    FFTZ_VOID *complex_in;  // For R2C/C2R: real->complex conversion
    FFTZ_VOID *complex_out; // For R2C/C2R: half-complex->complex conversion
    FFTZ_VOID *in_ref;      // DFT reference input buffer
    FFTZ_VOID *out_ref;     // DFT reference output buffer
} dft_ref_buffers_t;

FFTZ_INT32 run_dft_reference_test(aoclfftz_bench_params_t *params,
                                  FFTZ_INTP *in_idx_map, FFTZ_INTP *out_idx_map,
                                  FFTZ_VOID *handle, FFTZ_VOID *input_buffer);
FFTZ_VOID dft_ref_f(aoclfftz_bench_params_t *params, FFTZ_VOID *in,
                    FFTZ_VOID *out, FFTZ_INTP *in_idx_map,
                    FFTZ_INTP *out_idx_map);
FFTZ_VOID dft_ref_d(aoclfftz_bench_params_t *params, FFTZ_VOID *in,
                    FFTZ_VOID *out, FFTZ_INTP *in_idx_map,
                    FFTZ_INTP *out_idx_map);
FFTZ_INT32 allocate_dftref_buffers(aoclfftz_bench_params_t *params,
                              dft_ref_buffers_t *buffers, FFTZ_UINT32 is_align);
FFTZ_INT32 execute_fft_and_postprocess(aoclfftz_bench_params_t *params,
                                       dft_ref_buffers_t *buffers,
                                       FFTZ_VOID *handle, FFTZ_INTP *in_idx_map,
                                       FFTZ_INTP *out_idx_map);
FFTZ_VOID cleanup_buffers(aoclfftz_bench_params_t *params,
                     dft_ref_buffers_t *buffers, FFTZ_UINT32 is_align);

#endif
#endif // DFT_REFERENCE_H
