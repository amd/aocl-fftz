// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file bluestein_utils.h
 *
 * @brief Declarations for utility functions related to Bluestein solver.
 *
 * This file contains the functions for computing the extended length and
 * the Bluestein chirp sequence.
 *
 * @author Srirammaswamy Srinivasan
 */

#ifndef BLUESTEIN_UTILS_H
#define BLUESTEIN_UTILS_H

#include "api/aoclfftz_internal.h"
#include "core/kernels/kernel.h"

// Error return codes related to Bluestein sequence
typedef enum
{
    BLUESTEIN_FAILURE = -1,
    BLUESTEIN_SUCCESS
} bluestein_status;

// Core utility functions
FFTZ_INTP get_extended_length(FFTZ_INTP n);
FFTZ_INT32 compute_chirp_sequence(aoclfftz_solution_t *sol, FFTZ_INTP m);

/**
 * Copies `n` complex elements between Bluestein I/O buffers.
 *
 * For strided cases (src_stride > 1 or dst_stride > 1) dispatches to the
 * stride-aware permuted_copy_c_* kernel; otherwise falls back to a single
 * memcpy. Shared by both the ST and MT Bluestein execute paths.
 */
FFTZ_VOID bluestein_copy_data(FFTZ_VOID *src, FFTZ_VOID *dst, FFTZ_INTP n,
                              FFTZ_INTP src_stride, FFTZ_INTP dst_stride,
                              FFTZ_UINT8 dt_prec, FFTZ_UINT32 dt_bytes);

#endif // BLUESTEIN_UTILS_H
