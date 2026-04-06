// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file executor.h
 *
 *  @brief Functions and data structures for executor module.
 *
 *  This file contains the functions and data structures that are used to
 *  execute a solution of kernels for the given input problem description.
 *
 *  @author S. Biplab Raut
 */

#ifndef AOCLFFTZ_EXECUTOR_H
#define AOCLFFTZ_EXECUTOR_H

#include <string.h> /* for memcpy */
#include "core/solvers/solver.h"

// Error return codes related to executor
// Add more codes at the top
typedef enum
{
    EXECUTOR_FAILURE = -1,
    EXECUTOR_SUCCESS         // Successful operation
} aoclfftz_executor_status;

// Executor data structure that is used to hold the solution and cost analysis
// at each decomposition level for the associated sub-problem
typedef struct aoclfftz_executor
{
    aoclfftz_solution_t *solution;
    execute_ execute;
    // cost_analysis_t *cost_analysis;
} aoclfftz_executor_t;

#endif // AOCLFFTZ_EXECUTOR_H
