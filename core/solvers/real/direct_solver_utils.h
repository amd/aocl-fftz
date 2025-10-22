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

/** @file direct_solver_utils.h
 *
 *  @brief Direct Solver helper function's declarations required for
 *         the setup of direct solver
 *
 *  @author Partiksha
 */

#ifndef DIRECT_SOLVER_UTILS_H
#define DIRECT_SOLVER_UTILS_H

#include "api/aoclfftz_internal.h"
#include "core/kernels/kernel.h"

/**
 * @brief Paired input and output stride values used for element-level access,
 * vector/batch traversal, and C2C kernel batch stepping in both direct
 * and CT decomposition scenarios.
 * Note: Stride arrays cannot be stored here.
 */
typedef struct base_strides
{
    INTP in_stride;  /**< Stride for input buffer access */
    INTP out_stride; /**< Stride for output buffer access */
}base_strides_t;

/**
 * @brief In RFFT, output can be the real problem input buffer or a temp buffer.
 * This function checks if the output is the real problem output buffer.
 */
static inline UINT8 is_output_prob_buffer(aoclfftz_solution_t *sol)
{
    UINT32 is_fwd = FFT_DIR(sol->decomp_scheme->flags) == FORWARD_FFT_DIR;
    UINT32 is_last_stage = sol->next_sol == NULL;
    return (is_fwd && is_last_stage);
}

/**
 * @brief In RFFT, input can be the real problem input buffer or a temp buffer.
 * This function checks if the input is the real problem input buffer.
 */
static inline UINT8 is_input_prob_buffer(aoclfftz_solution_t *sol)
{
    return (FFT_DIR(sol->decomp_scheme->flags) == BACKWARD_FFT_DIR &&
            NUM_RFFT_GROUPS(sol->solver) == 1);
}

/**
 * Updates the strides for the second half of the batch.
 * It subtracts the stride_offset from each stride value to properly address
 * memory locations in the next iteration of C2C kernel execution.
 */
static inline VOID update_asymmetric_strides(INTP *strides, INTP radix,
                                     INTP batch_stride)
{
    // Since the, the second half of the batch wraps around at nyquist point,
    // the stride for second half is `-original_stride*2`
    INTP h2_stride = batch_stride * 2;
    INTP half_stride_start = (radix + 1) >> 1;
    for (INTP i = half_stride_start; i < radix; i++)
    {
        strides[i] -= h2_stride;
    }
}

VOID compute_conjugates(VOID *data, INTP radix, INTP n, INTP *strides,
                        INTP vec_stride, UINT32 prec);

VOID set_zero_for_dc_and_nyquist_batched(aoclfftz_solution_t *sol);
VOID set_zero_for_dc_and_nyquist(aoclfftz_solution_t *sol);

VOID set_kernel_count_in_each_group(aoclfftz_solution_t *sol,
                             aoclfftz_realhelper_t *realhelper);
VOID allocate_and_setup_stride(aoclfftz_solution_t *sol,
                               aoclfftz_realhelper_t realhelper);
VOID update_ct_buffers(aoclfftz_solution_t *sol,
                       aoclfftz_realhelper_t *realhelper);
VOID compute_cost(aoclfftz_solution_t *sol, cost_analysis_t *cost,
                  const kernel_t *kernel_c2c, const kernel_t *kernel_r2hc,
                  const kernel_t *kernel_r2hcf);

#endif // DIRECT_SOLVER_UTILS_H
