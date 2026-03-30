// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file selector_transpose.c
 *
 *  @brief Wrapper that acts on the transpose solver as guided by the selector.
 *
 *  This file contains the implementation of functions that are used to
 *  setup a standalone transpose problem.
 *
 *  @author Ashwin K. Godbole
 */

#include "selector/selector.h"
#include "utils/utils.h"

INT32 selector_transpose(aoclfftz_selector_t *sel)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    if (sel == NULL || sel->solution == NULL ||
        sel->solution->decomp_scheme == NULL)
    {
        AOCLFFTZ_LOG(INFO, global_logger_mode,
                     "Invalid selector or solution passed to "
                     "selector_transpose");
        return SELECTOR_FAILURE;
    }

    INT32 ret = SELECTOR_FAILURE;

    // Support only 2D transpose problems.
    if (sel->solution->decomp_scheme->dim_rank != 2)
    {
        goto exit_transpose;
    }

    ret = setup_transpose_solver(sel->solution);
    if (ret != SELECTOR_SUCCESS)
    {
        goto exit_transpose;
    }

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");

    return SELECTOR_SUCCESS;

exit_transpose:

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit with failure");

    return ret;
}
