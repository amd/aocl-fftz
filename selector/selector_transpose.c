/**
 * Copyright (C) 2025, Advanced Micro Devices. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

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
#ifdef AOCL_ENABLE_LOG
    INT32 logger_mode = sel->solution->decomp_scheme->cntrl_params->logger_mode;
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Enter");
#endif

    INT32 ret = SELECTOR_FAILURE;

    // Support only 2D transpose problems.
    if (sel->solution->decomp_scheme->dim_rank != 2)
    {
        goto exit_transpose;
    }

    // Find CPU feature flags that will be used by dynamic dispatcher
    INT32 cpu_flags = setup_dynamic_dispatcher(
            sel->solution->decomp_scheme->cntrl_params->opt_off,
            sel->solution->decomp_scheme->cntrl_params->opt_level,
            sel->solution->decomp_scheme->cntrl_params->logger_mode);

    ret = setup_transpose_solver(sel->solution, cpu_flags);
    if (ret != SELECTOR_SUCCESS)
    {
        goto exit_transpose;
    }

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit");
#endif

    return SELECTOR_SUCCESS;

exit_transpose:

#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit with failure");
#endif

    return ret;
}
