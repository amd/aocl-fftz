// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file nop_solver.c
 *
 *  @brief No-op solver for zero-length batch problems.
 *
 *  @author Jeya R
 */

#include "core/solvers/solver.h"

static FFTZ_INT32 execute_nop_solver(aoclfftz_solution_t *sol,
                                     aoclfftz_mutable_ctx_t *ctx)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SOLVER_SUCCESS;
}

dft_solver_ register_execute_nop_solver(FFTZ_VOID)
{
    return execute_nop_solver;
}
