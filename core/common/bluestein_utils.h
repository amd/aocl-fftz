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
INTP get_extended_length(INTP n);
INT32 compute_chirp_sequence(aoclfftz_solution_t *sol, INTP m);

#endif // BLUESTEIN_UTILS_H
