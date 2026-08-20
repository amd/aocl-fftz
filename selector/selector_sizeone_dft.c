// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file selector_sizeone_dft.c
 *
 *  @brief Wrapper that acts on the sizeone solver as guided by the selector.
 *
 *  This file includes a function call which sets up the kernel function
 *  pointers based on data presicion.
 *
 *  @author D. Vijay Krishna
 */

#include "selector/selector.h"

FFTZ_INT32 selector_sizeone_dft(aoclfftz_selector_t *sel, kernel_t *kertab)
{
    if (sel == NULL || sel->solution == NULL ||
        sel->solution->decomp_scheme == NULL)
    {
        return SELECTOR_FAILURE;
    }

    FFTZ_UINT8 is_real = IS_REAL(sel->solution->decomp_scheme->flags);
    FFTZ_INT32 ret = is_real ? setup_real_sizeone_solver(sel->solution)
                        : setup_sizeone_solver(sel->solution);
    return ret;
}
