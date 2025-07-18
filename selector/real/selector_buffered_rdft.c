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

/** @file selector_buffered_rdft.c
 *
 *  @brief Wrapper that acts on the buffered solver as guided by the selector.
 *
 *  This file contains the implementation of functions that are used to
 *  setup, factorize and evaluate sub-problems and kernels as applicable.
 *
 *  @author Srirammaswamy Srinivsan
 */

#include "selector/selector.h"
#include "core/common/memory_manager.h"
#include "utils/utils.h"

INT32 selector_buffered_rdft(aoclfftz_selector_t *sel, kernel_t *kertab,
                             aoclfftz_realhelper_t *realhelper)
{
#ifdef AOCL_ENABLE_LOG
    INT32 logger_mode = sel->solution->decomp_scheme->cntrl_params->logger_mode;
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Enter");
#endif
    aoclfftz_selector_t *cur_sel = NULL;
    INT32 vec_rank = sel->solution->decomp_scheme->vec_rank;
    INT32 dim_rank = sel->solution->decomp_scheme->dim_rank;
    INT32 ret = SELECTOR_FAILURE;

    cur_sel = alloc_selector(vec_rank, dim_rank, sel->scratch_space);
    if (cur_sel == NULL)
    {
        goto exit_batched_dft;
    }

    // copy solution object from sel to cur_sel
    COPY_SOLUTION_OBJ(cur_sel->solution, sel->solution);

    // Setup buffered solver
    ret = setup_real_buffered_solver(sel->solution, realhelper);
    if (ret != SELECTOR_SUCCESS)
    {
        goto exit_batched_dft;
    }

    // Copy the auxiliary buffers (address) from sel to cur_sel (i.e. next_sol)
    cur_sel->solution->dft_bufs->buffered->aux_buffer_1 =
        sel->solution->dft_bufs->buffered->aux_buffer_1;
    cur_sel->solution->dft_bufs->buffered->aux_buffer_2 =
        sel->solution->dft_bufs->buffered->aux_buffer_2;

    // Call selector for solving it as a non-buffered problem
    ret = setup_rdft_(cur_sel, kertab, realhelper);
    if (ret != SELECTOR_SUCCESS)
    {
        goto exit_batched_dft;
    }

    sel->solution->next_sol = cur_sel->solution;

    // Set the out_ptr to last direct solution's output
    aoclfftz_solution_t *temp_sol = sel->solution->next_sol;
    while (temp_sol->next_sol != NULL)
    {
        temp_sol = temp_sol->next_sol;
    }
    sel->solution->dft_bufs->buffered->out_ptr = &temp_sol->decomp_scheme->out_real;

    // destroy only the selector not the solution within it
    destroy_selector_without_solution(cur_sel);
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit");
#endif
    return SELECTOR_SUCCESS;

exit_batched_dft:
    destroy_selector_without_scratch_space(cur_sel);
#ifdef AOCL_ENABLE_LOG
    AOCLFFTZ_LOG_UNFORMATTED(TRACE, logger_mode, "Exit");
#endif

    return ret;
}
