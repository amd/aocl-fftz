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

/** @file post_process_utils.c
 *
 *  @brief Post-selection utilities that enables additional optimizations.
 *
 *  This module walks the selected solution graph and, when eligible, modifies
 * the solution graph to enable additional optimization.
 *
 *  Optimization : 2D-buffering for 3D problems:
 *  - Allocate 2D size ct_buffer for 3D problems instead of the original 3D size
 * ct_buffer
 *  - Batch partitioning of the outermost dim's batches between batched solver
 * and direct solver
 *  - For the outermost dim, reduce the strides of ct_buffer to match the
 * smaller ct_buffer size
 *
 *  Known limitations:
 *  - Bluestein problems are excluded
 *  - Real variants (R2C/C2R) are not supported
 *  - Only column-major batched complex CT problems are supported
 *
 *  @author Srirammaswamy Srinivasan
 */

#include "core/common/post_process_utils.h"
#include "api/aoclfftz_internal.h"
#include "core/solvers/solver.h"
#include "selector/selector.h"
#include "utils/allocator.h"
#include "utils/utils.h"

#ifndef DISABLE_OPTIMAL_BUFFERING_BATCHING

/**
 * @brief Check whether the current position is eligible for batch partitioning.
 *
 * Evaluates structural and layout constraints around the current solution node
 * to determine whether a batch partition conversion can be applied. The checks
 * include bluestein detection, solver ordering (prev/curr/next), and whether
 * the problem has a column-major layout.
 *
 * @param sol      Current solution node (expected to be a Batched solver node).
 * @param prev_sol Previous solution node, or NULL if at the root.
 * @return INT32   1 if eligible, 0 otherwise.
 */
static INT32 is_eligible_for_batch_partition(aoclfftz_solution_t *sol,
                                             aoclfftz_solution_t *prev_sol)
{
    // batched-direct approach is not supported for bluestein problems
    if (check_bluestein_problem(sol->decomp_scheme))
    {
        return 0; // return false
    }

    // Step 1: Check for eligible solver orders in the solution list
    aoclfftz_solver_type prev_solver =
        prev_sol ? prev_sol->solver->solver_type
                 : SOLVER_NULL; // 0 means no solver (i.e. sol is the root node)
    aoclfftz_solver_type curr_solver = sol->solver->solver_type;
    aoclfftz_solver_type next_solver =
        sol->next_sol ? sol->next_sol[0]->solver->solver_type
                      : SOLVER_NULL; // 0 means no solver (i.e. sol is the leaf node)

    // previous solver should be NULL or an NDim variant
    INT32 valid_prev_sol = prev_solver == SOLVER_NDIM;
    // current solver should be a Batched variant
    INT32 valid_curr_sol =
        (curr_solver == SOLVER_BATCHED) || (curr_solver == SOLVER_MT_BATCHED);
    // next solver should be a CT variant
    INT32 valid_next_sol =
        (next_solver == SOLVER_CT) || (next_solver == SOLVER_CT_TWIDDLE);

    // return false if any of the conditions are not met
    if (!valid_prev_sol || !valid_curr_sol || !valid_next_sol)
    {
        return 0;
    }

    // Step 2: Check for batched_vecs existence
    aoclfftz_dim_t_64_ *batched_vecs = sol->decomp_scheme->batched_vecs;

    if (batched_vecs != NULL)
    {
        AOCLFFTZ_LOG(
            TRACE, global_logger_mode,
            "using batch partitioning for problem: "
            "%td:%td:%td v [%td:%td:%td v %td:%td:%td]",
            sol->decomp_scheme->vecs[0].n,
            sol->decomp_scheme->vecs[0].in_stride,
            sol->decomp_scheme->vecs[0].out_stride,
            batched_vecs[0].n, batched_vecs[0].in_stride,
            batched_vecs[0].out_stride,
            sol->decomp_scheme->dims[0].n,
            sol->decomp_scheme->dims[0].in_stride,
            sol->decomp_scheme->dims[0].out_stride);

        return 1; // return true
    }

    return 0; // return false
}

/**
 * @brief Traverse and rewrite the solution list for eligible optimizations.
 *
 * Optimization: 2D-buffering for 3D problems:
 * ------------ -- ------------ --- -- ---------
 * Use smaller ct_buffer (of 2D size) for 3D problems instead of the original
 * ct_buffer (of 3D size)
 *
 * Approach:
 * - Allocate 2D size ct_buffer for 3D problems instead of the original 3D size
 * ct_buffer
 * - For the outermost dim, reduce the strides of ct_buffer to match the smaller
 * ct_buffer size
 *
 * @param curr_sol Current solution being inspected.
 * @param prev_sol Previous solution context for eligibility checks.
 * @param dim_rank Dimension rank of the problem.
 */
static VOID
post_process_for_optimal_buffering_batching_(aoclfftz_solution_t *curr_sol,
                                             aoclfftz_solution_t *prev_sol,
                                             INT32 dim_rank)
{
    // iterate through the solution list
    while (curr_sol)
    {
        // check if the current solution is a batched solver with its dim_rank
        // is 2 or the current dim_rank is 3
        // this is required to handle the outermost dim's batched solver
        UINT8 is_3rd_dim_batched =
            curr_sol &&
            (curr_sol->solver->solver_type == SOLVER_BATCHED ||
             curr_sol->solver->solver_type == SOLVER_MT_BATCHED) &&
            (curr_sol->decomp_scheme->dim_rank == 2 || dim_rank == 3);

        // traverse nd_sol first so inner batched nodes after NDim are handled
        if (curr_sol->solver->solver_type == SOLVER_NDIM &&
            curr_sol->dft_bufs->nd_sol)
        {
            post_process_for_optimal_buffering_batching_(
                curr_sol->dft_bufs->nd_sol, curr_sol, dim_rank - 1);
        }

        // Compact buffer batch partition can be used if the following
        // conditions are met:
        // 1. Not bluestein solvable
        // 2. batched_vecs exist
        if (is_eligible_for_batch_partition(curr_sol, prev_sol) &&
            is_3rd_dim_batched)
        {
            // if eligible, then the current solution is a batched solver type
            // so using batched_sol & batched_vecs to refer to that solution
            aoclfftz_solution_t *batched_sol = curr_sol;

            // get the vecs[0] values before removing them in the batched_sol
            INTP batched_vec_n = 0;
            INTP batched_vec_in_stride = 0;
            INTP batched_vec_out_stride = 0;

            if (is_3rd_dim_batched)
            {
                // For 3D unit-strided C2C problems, when compact buffer
                // approach is used preserve few batches in batched solver and
                // move the rest to direct solver instead of moving all the
                // batches to direct solver Example: problem size of 30x40x50 ->
                // ct_buffer size is 40x50 Consider the outermost dim which is
                // 30 point FFT of 40x50 batches where these batches will be
                // fused to 2000 (40*50) batches
                //
                // properties of `batched_sol`:
                // vec_rank = 1
                //   vecs[0].n = 2000
                // dim_rank = 3
                //   dims[0].n = 30, dims[1].n = 40, dims[2].n = 50
                //
                // properties of `prev_sol` (ndim solver):
                // vec_rank = 1
                //   vecs[0].n = 30
                // dim_rank = 2
                //   dims[0].n = 40 (stored as `prev_dim`), dims[1].n = 50
                //
                // batches before this change:
                // vecs[0].n of batched-solver = 2000
                // vecs[0].n of direct-solver = 1 (1 CT problem at a time)
                //
                // batches after this change:
                // vecs[0].n of batched-solver = 40
                // vecs[0].n of direct-solver = 50
                //
                // do the following changes:
                // - set direct-solver batched_vecs[0].n to prev_dim (i.e. 40)
                // - update batched-solver vecs[0].n to vecs[0].n / prev_dim
                // (i.e. 2000 / 40 = 50)
                // - update batched-solver vecs[0].in_stride and
                // vecs[0].out_stride to prev_dim (i.e. 40)
                INTP prev_dim = prev_sol->decomp_scheme->dims[0].n;
                batched_vec_n = prev_dim;
                batched_vec_in_stride =
                    batched_sol->decomp_scheme->vecs[0].in_stride;
                batched_vec_out_stride =
                    batched_sol->decomp_scheme->vecs[0].out_stride;

                batched_sol->decomp_scheme->vecs[0].n =
                    batched_sol->decomp_scheme->batched_vecs[0].n / prev_dim;
                batched_sol->decomp_scheme->vecs[0].in_stride = prev_dim;
                batched_sol->decomp_scheme->vecs[0].out_stride = prev_dim;
            }

            // reduce the strides of ct_buffer to match the smaller ct_buffer
            // size
            INTP ct_buffer_stride_factor =
                batched_sol->decomp_scheme->dims[0].out_stride / batched_vec_n;

            // iterate through the next soultions of the batched solver to find
            // the direct solver of CT problems (i.e. CT-r direct and last CT-m
            // direct) and perform the following changes to CT-r / CT-m direct
            // solvers:
            // - change it to batched-direct solver (updates solver_type and
            // execute_solver ptr)
            // - use `batched_vecs` to hold the vecs[0] from batched solver
            // - update thread_info from batched solver
            // - update vec-strides in strides struct to use problem vec-strides
            while (curr_sol)
            {
                if (curr_sol->solver->solver_type ==
                        SOLVER_DIRECT_BATCHED_COLMAJOR ||
                    curr_sol->solver->solver_type ==
                        SOLVER_MT_DIRECT_BATCHED_COLMAJOR ||
                    curr_sol->solver->solver_type ==
                        SOLVER_MT_DIRECT_BATCHED_ROWMAJOR)
                {
                    // allocate a new struct in the direct solver to hold the
                    // vecs[0] from batched solver
                    if (curr_sol->decomp_scheme->batched_vecs == NULL)
                    {
                        ALLOC_ALIGN_UNINIT(
                            curr_sol->decomp_scheme->batched_vecs,
                            aoclfftz_dim_t_64_, sizeof(aoclfftz_dim_t_64_));
                    }
                    curr_sol->decomp_scheme->batched_vecs[0].n = batched_vec_n;
                    curr_sol->decomp_scheme->batched_vecs[0].in_stride =
                        batched_vec_in_stride;
                    curr_sol->decomp_scheme->batched_vecs[0].out_stride =
                        batched_vec_out_stride;
                    AOCLFFTZ_LOG(
                        TRACE, global_logger_mode,
                        " -> sub-problem: "
                        "%td:%td:%td v [%td:%td:%td v %td:%td:%td]",
                        curr_sol->decomp_scheme->vecs[0].n,
                        curr_sol->decomp_scheme->vecs[0].in_stride,
                        curr_sol->decomp_scheme->vecs[0].out_stride,
                        curr_sol->decomp_scheme->batched_vecs[0].n,
                        curr_sol->decomp_scheme->batched_vecs[0].in_stride,
                        curr_sol->decomp_scheme->batched_vecs[0].out_stride,
                        curr_sol->decomp_scheme->dims[0].n,
                        curr_sol->decomp_scheme->dims[0].in_stride,
                        curr_sol->decomp_scheme->dims[0].out_stride);

                    // For 3D unit-strided C2C problems, when compact buffer
                    // approach is used reduce the strides of ct_buffer to match
                    // the smaller ct_buffer size
                    INTP radix = curr_sol->decomp_scheme->dims[0].n;
                    if (curr_sol->next_sol) // twiddle kernel
                    {
                        // reduce the out-strides of outermost CT-r twiddle
                        // kernel
                        if (curr_sol->decomp_scheme->decomp_level == 0)
                        {
                            curr_sol->decomp_scheme->dims[0].in_stride /=
                                ct_buffer_stride_factor;
                            curr_sol->decomp_scheme->vecs[0].in_stride /=
                                ct_buffer_stride_factor;
                            for (INTP i = 0; i < radix; i++)
                            {
                                curr_sol->strides_grp->strides->in_strides[i] /=
                                    ct_buffer_stride_factor;
                            }
                        }
                        // reduce both in and out-strides of remaining CT-r
                        // twiddle kernels
                        else
                        {
                            curr_sol->decomp_scheme->dims[0].in_stride /=
                                ct_buffer_stride_factor;
                            curr_sol->decomp_scheme->vecs[0].in_stride /=
                                ct_buffer_stride_factor;
                            curr_sol->decomp_scheme->dims[0].out_stride /=
                                ct_buffer_stride_factor;
                            curr_sol->decomp_scheme->vecs[0].out_stride /=
                                ct_buffer_stride_factor;
                            for (INTP i = 0; i < radix; i++)
                            {
                                curr_sol->strides_grp->strides->in_strides[i] /=
                                    ct_buffer_stride_factor;
                                curr_sol->strides_grp->strides
                                    ->out_strides[i] /= ct_buffer_stride_factor;
                            }
                        }
                    }
                    else
                    {
                        // reduce the in-strides of CT-m standard kernel
                        curr_sol->decomp_scheme->dims[0].out_stride /=
                            ct_buffer_stride_factor;
                        curr_sol->decomp_scheme->vecs[0].out_stride /=
                            ct_buffer_stride_factor;
                        for (INTP i = 0; i < radix; i++)
                        {
                            curr_sol->strides_grp->strides->out_strides[i] /=
                                ct_buffer_stride_factor;
                        }
                    }
                    // update vec strides
                    curr_sol->strides_grp->strides->v_in_stride =
                        batched_vec_in_stride * DATA_STRIDE;
                    curr_sol->strides_grp->strides->v_out_stride =
                        batched_vec_out_stride * DATA_STRIDE;
                }
                else // other than direct solvers
                {
                    if (((curr_sol->solver->solver_type == SOLVER_BATCHED ||
                          curr_sol->solver->solver_type ==
                              SOLVER_MT_BATCHED)) &&
                        (curr_sol->decomp_scheme->decomp_level == 0) &&
                        is_3rd_dim_batched)
                    {
                        // in a 3D problem (when 2D_buffering is used), the
                        // outermost dim will reuse the smaller 2D size
                        // ct_buffer for each batched solver iteration,
                        // hence the ct_buffer offset should not be moved
                        // across batches,
                        // so setting reset_ct_buf_offset to 1 to prevent
                        // ct_buffer movement
                        curr_sol->dft_bufs->reset_ct_buf_offset = 1;
                    }
                    else
                    {
                        // ct_buffer offset should be moved across batches
                        // so setting reset_ct_buf_offset to 0
                        curr_sol->dft_bufs->reset_ct_buf_offset = 0;
                        curr_sol->decomp_scheme->dims[0].out_stride /=
                            ct_buffer_stride_factor;
                        curr_sol->decomp_scheme->vecs[0].out_stride /=
                            ct_buffer_stride_factor;
                    }
                }

                curr_sol = curr_sol->next_sol ? curr_sol->next_sol[0] : NULL;
            }
        }
        else
        {
            if (((curr_sol->solver->solver_type == SOLVER_BATCHED ||
                  curr_sol->solver->solver_type == SOLVER_MT_BATCHED)) &&
                (curr_sol->decomp_scheme->decomp_level == 0) &&
                is_3rd_dim_batched)
            {
                // in a 3D problem (when 2D_buffering is used), the
                // outermost dim will reuse the smaller 2D size ct_buffer
                // for each batched solver iteration,
                // hence the ct_buffer offset should not be moved across
                // batches,
                // so setting reset_ct_buf_offset to 1 to prevent
                // ct_buffer movement
                curr_sol->dft_bufs->reset_ct_buf_offset = 1;
            }
            else
            {
                // ct_buffer offset should be moved across batches
                // so setting reset_ct_buf_offset to 0
                curr_sol->dft_bufs->reset_ct_buf_offset = 0;
                curr_sol->dft_bufs->use_2D_buffering = 0;
            }
        }
        if (!curr_sol)
        {
            break;
        }
        prev_sol = curr_sol;
        curr_sol = curr_sol->next_sol ? curr_sol->next_sol[0] : NULL;
    }
}
#endif

/**
 * @brief High-level function to apply the optimizations to the solution list.
 *
 * @param sol Solution list to apply the optimizations to.
 */
VOID post_process_for_optimal_buffering_batching(aoclfftz_solution_t *sol)
{
#ifndef DISABLE_OPTIMAL_BUFFERING_BATCHING
    if (sol && sol->dft_bufs && sol->dft_bufs->use_2D_buffering)
    {
        post_process_for_optimal_buffering_batching_(sol, NULL,
                                                 sol->decomp_scheme->dim_rank);
    }
#endif
}
