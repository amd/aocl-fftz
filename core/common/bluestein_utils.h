// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file bluestein_utils.h
 *
 *  @brief Declarations for utility functions related to Bluestein solver.
 *
 *  This file contains the functions for computing extended length, computing
 *  bluestein sequence, elementwise multiplication and normalize data.
 *
 *  @author Srirammaswamy Srinivasan
 */

#ifndef BLUESTEIN_UTILS_H
#define BLUESTEIN_UTILS_H

#include "api/aoclfftz_internal.h"
#include "core/kernels/kernel.h"

// Error return codes related to Bluestein sequence and elementwise
// multiplication
typedef enum
{
    BLUESTEIN_FAILURE = -1,
    BLUESTEIN_SUCCESS // Successful operation
} bluestein_status;

INTP get_extended_length(INTP n);
INT32 prepare_bluestein_sequence(aoclfftz_solution_t *sol, INTP m);
INT32 elementwise_multiplication(VOID *out, VOID *a, VOID *b, INTP n,
                                 UINT8 sign, UINT8 precision);
INT32 normalize_data(VOID *data, INTP n, DOUBLE normalize_factor,
                     UINT8 precision);

#endif // BLUESTEIN_UTILS_H
