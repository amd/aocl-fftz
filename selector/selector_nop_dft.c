// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file selector_nop_dft.c
 *
 *  @brief Selector hook for the no-op solver.
 *
 *  @author Jeya R
 */

#include "selector/selector.h"

FFTZ_INT32 selector_nop_dft(aoclfftz_selector_t *sel)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    aoclfftz_generic_solver_t *solver_obj = sel->solution->solver;
    solver_obj->solver_type = SOLVER_NOP;
    if (set_solver_fp(solver_obj) != SOLVER_SUCCESS)
    {
        AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit with failure");
        return SELECTOR_FAILURE;
    }

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SELECTOR_SUCCESS;
}
