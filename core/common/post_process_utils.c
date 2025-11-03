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
 *  This module walks the selected solution graph and, when eligible, modifies the
 *  solution graph to enable additional optimizations.
 *
 *  Optimization 1: Batched-Direct solver:
 *  Direct CT sub-problem into a Batched-Direct sub-problem by:
 *  - moving one level of vectorization (the innermost `vecs[0]`) from the
 *    preceding Batched node into the CT Direct node as `batched_vecs[0]`
 *  - changing the solver type and `execute_solver` function pointer
 *
 *  Optimization 2: 2D-buffering for 3D problems:
 *  - Allocate 2D size ct_buffer for 3D problems instead of the original 3D size ct_buffer
 *  - For the outermost dim, reduce the strides of ct_buffer to match the smaller ct_buffer size
 *
 *  Known limitations:
 *  - Bluestein problems are excluded
 *  - Real variants (R2C/C2R) are not supported
 *  - Only column-major batched complex CT problems are supported
 *  - Complex multi-threaded problems have scaling issues
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
 * @brief Check whether the current position is eligible for Batched-Direct.
 *
 * Evaluates structural and layout constraints around the current solution node
 * to determine whether a Batched-Direct conversion can be applied. The checks
 * include bluestein detection, solver ordering (prev/curr/next), and whether
 * the problem has a column-major layout.
 *
 * TODO: support bluestein and r2c/c2r problems
 *
 * @param sol      Current solution node (expected to be a Batched solver node).
 * @param prev_sol Previous solution node, or NULL if at the root.
 * @return INT32   1 if eligible, 0 otherwise.
 */
static INT32 is_eligible_for_batched_direct(aoclfftz_solution_t *sol,
                                            aoclfftz_solution_t *prev_sol)
{
    // batched-direct approach is not supported for bluestein problems
    if (check_bluestein_problem(sol->decomp_scheme))
    {
        return 0; // return false
    }

    // Step 1: Check for eligible solver orders in the solution list
    aoclfftz_solver_type prev_solver =
        prev_sol ? prev_sol->solver->solver_type : 0; // 0 means no solver (i.e. sol is the root node)
    aoclfftz_solver_type curr_solver = sol->solver->solver_type;
    aoclfftz_solver_type next_solver =
        sol->next_sol ? sol->next_sol[0]->solver->solver_type : 0; // 0 means no solver (i.e. sol is the leaf node)

    // previous solver should be NULL or an NDim variant
    INT32 valid_prev_sol = (prev_solver == 0) ||
                           (prev_solver == SOLVER_NDIM);
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

    // Step 2: Check for eligible vec-strided batched problems
    aoclfftz_dim_t_64_ *dims = sol->decomp_scheme->dims;
    aoclfftz_dim_t_64_ *vecs = sol->decomp_scheme->vecs;

    if (vecs[0].n > 1 && /* batched problem */
        vecs[0].in_stride < dims[0].in_stride && /* column-major order input */
        vecs[0].out_stride < dims[0].out_stride && /* column-major order output */
        dims[0].in_stride == dims[0].out_stride && /* TODO: support different in/out strides */
        vecs[0].in_stride == vecs[0].out_stride)
    {
        AOCLFFTZ_LOG(
            TRACE, global_logger_mode,
            "using batched-direct for problem: "
            "%td:%td:%td v %td:%td:%td",
            vecs[0].n, vecs[0].in_stride, vecs[0].out_stride, dims[0].n,
            dims[0].in_stride, dims[0].out_stride);

        return 1; // return true
    }

    return 0; // return false
}

/**
 * @brief Traverse and rewrite the solution list for eligible optimizations.
 *
 * Optimization 1: Batched-Direct solver:
 * ------------ -- -------------- -------
 * Rewrite eligible Batched -> CT sub-paths in place and
 * remove the innermost batches from batched solver and add it to direct solver.
 *
 * Approach:
 * - Recursively walks the solution list (including N-D branches).
 * - For each eligible Batched -> CT path:
 *   - Moves the innermost vectorization from batched solver to the direct
 * solver as `batched_vecs[0]`.
 *   - Reduces the batches by 1 rank at the batched solver node.
 *   - For multi-threaded batched solver, moves the thread_info from batched
 * solver to the direct solver, to enable parallel execution within the direct
 * solver.
 *
 * Limitations:
 * - This is performed as a post-process step; ideally it should be a part of
 * selectors & selector_model_dft_
 * - It will not work when bluestein solver is used (large primes or multiples
 * of large primes)
 * - It will not work for R2C and C2R problems
 *
 * Optimization 2: 2D-buffering for 3D problems:
 * ------------ -- ------------ --- -- ---------
 * Use smaller ct_buffer (of 2D size) for 3D problems instead of the original ct_buffer (of 3D size)
 *
 * Approach:
 * - Allocate 2D size ct_buffer for 3D problems instead of the original 3D size ct_buffer
 * - For the outermost dim, reduce the strides of ct_buffer to match the smaller ct_buffer size
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
    // iterate through the solution list on ndim branches first
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

        // batched-direct solver can be used if the following conditions are met:
        // 1. The previous solver is a NDim solver or NULL
        // 2. The current solver is a batched solver
        // 3. The next solver is a CT solver
        if (is_eligible_for_batched_direct(curr_sol, prev_sol))
        {
            // if eligible, then the current solution is a batched solver type
            // so using batched_sol & batched_vecs to refer to that solution
            aoclfftz_solution_t *batched_sol = curr_sol;
            aoclfftz_dim_t_64_ *vecs = batched_sol->decomp_scheme->vecs;
            // check if the problem is col-major or row-major
            // col-major: vec-strides < elemental-strides
            // row-major: elemental-strides < vec-strides
            UINT8 is_col_major =
                (batched_sol->decomp_scheme->vecs[0].in_stride <
                 batched_sol->decomp_scheme->dims[0].in_stride) &&
                (batched_sol->decomp_scheme->vecs[0].out_stride <
                 batched_sol->decomp_scheme->dims[0].out_stride);

            // get the vecs[0] values before removing them in the batched_sol
            INTP batched_vec_n;
            INTP batched_vec_in_stride;
            INTP batched_vec_out_stride;

            if (is_3rd_dim_batched && curr_sol->dft_bufs->use_2D_buffering)
            {
                // For 3D unit-strided C2C problems, when compact buffer approach is used
                // preserve few batches in batched solver and move the rest to direct solver
                // instead of moving all the batches to direct solver
                // Example: problem size of 30x40x50 -> ct_buffer size is 40x50
                // Consider the outermost dim which is 30 point FFT of 40x50 batches
                // where these batches will be fused to 2000 (40*50) batches
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
                // - update batched-solver vecs[0].n to vecs[0].n / prev_dim (i.e. 2000 / 40 = 50)
                // - update batched-solver vecs[0].in_stride and vecs[0].out_stride to prev_dim (i.e. 40)
                INTP prev_dim = prev_sol->decomp_scheme->dims[0].n;
                batched_vec_n = prev_dim;
                batched_vec_in_stride =
                    batched_sol->decomp_scheme->vecs[0].in_stride;
                batched_vec_out_stride =
                    batched_sol->decomp_scheme->vecs[0].out_stride;

                batched_sol->decomp_scheme->vecs[0].n = vecs[0].n / prev_dim;
                batched_sol->decomp_scheme->vecs[0].in_stride = prev_dim;
                batched_sol->decomp_scheme->vecs[0].out_stride = prev_dim;
            }
            else
            {
                batched_vec_n = batched_sol->decomp_scheme->vecs[0].n;
                batched_vec_in_stride =
                    batched_sol->decomp_scheme->vecs[0].in_stride;
                batched_vec_out_stride =
                    batched_sol->decomp_scheme->vecs[0].out_stride;

                // do the following changes:
                // - set direct-solver batched_vecs[0].n to vecs[0].n
                // - set direct-solver batched_vecs[0].in_stride & out_stride to vecs[0].in_stride & out_stride
                // - remove vecs[0] from batched solver
                if (batched_sol->decomp_scheme->vec_rank == 1)
                {
                    // we can't make vec_rank 1 to 0 since vec_rank should be atleast 1
                    // so setting vecs[0].n as 1 which makes it as a non-batched problem
                    batched_sol->decomp_scheme->vecs[0].n = 1;
                    batched_sol->decomp_scheme->vecs[0].in_stride = 1;
                    batched_sol->decomp_scheme->vecs[0].out_stride = 1;
                }
                else
                {
                    // remove vecs[0] from batched solver
                    batched_sol->decomp_scheme->vec_rank -= 1;
                    for (INT32 i = 0; i < batched_sol->decomp_scheme->vec_rank;
                            i++)
                    {
                        vecs[i].n = vecs[i + 1].n;
                        vecs[i].in_stride = vecs[i + 1].in_stride;
                        vecs[i].out_stride = vecs[i + 1].out_stride;
                    }
                }
            }
#ifdef MULTI_THREADING
            // get the number of threads from the batched solver before making
            // it single threaded
            INT32 batched_n_threads =
                batched_sol->decomp_scheme->thread_info->n_threads;
            // make the multi-threaded batched_solver as single threaded
            if (batched_sol->solver->solver_type == SOLVER_MT_BATCHED &&
                !is_3rd_dim_batched)
            {
                batched_sol->decomp_scheme->thread_info->n_threads = 1;
                batched_sol->decomp_scheme->thread_info->avl_threads = 1;
                batched_sol->solver->solver_type = SOLVER_BATCHED;
                batched_sol->solver->execute_solver =
                    register_execute_batched_solver();
            }
#endif

            // For 3D unit-strided C2C problems, when compact buffer approach is used
            // reduce the strides of ct_buffer to match the smaller ct_buffer size
            INTP ct_buffer_stride_factor = 1;
            if (is_3rd_dim_batched && curr_sol->dft_bufs->use_2D_buffering)
            {
                ct_buffer_stride_factor =
                    batched_sol->decomp_scheme->dims[0].out_stride /
                    batched_vec_n;
            }

            // iterate through the next soultions of the batched solver to find
            // the direct solver of CT problems (i.e. CT-r direct and last CT-m direct)
            // and perform the following changes to CT-r / CT-m direct solvers:
            // - change it to batched-direct solver (updates solver_type and execute_solver ptr)
            // - use `batched_vecs` to hold the vecs[0] from batched solver
            // - update thread_info from batched solver
            // - update vec-strides in strides struct to use problem vec-strides
            while (curr_sol)
            {
                if (curr_sol->solver->solver_type == SOLVER_DIRECT ||
                    curr_sol->solver->solver_type == SOLVER_MT_DIRECT)
                {
                    // update solver type
#ifdef MULTI_THREADING
                    // Multi-threaded batched solver won't be used for 3D
                    // problems when compact buffer approach is used
                    if (batched_n_threads > 1 &&
                        (!is_3rd_dim_batched ||
                         !curr_sol->dft_bufs->use_2D_buffering))
                    {
                        if (is_col_major)
                        {
                            // FIXME: a temporary fix to use different MT solvers
                            // for 1D and ND problems; it is based on the
                            // performance study for 1D and 3D problems,
                            // remove this once we have an MT solver which works
                            // for all the cases
                            if (prev_sol &&
                                prev_sol->solver->solver_type == SOLVER_NDIM)
                            {
                                curr_sol->solver->solver_type =
                                    SOLVER_MT_DIRECT_BATCHED_COLMAJOR;
                                curr_sol->solver->execute_solver =
                                    register_execute_mt_direct_batched_colmajor_solver();
                            }
                            else
                            {
                                curr_sol->solver->solver_type =
                                    SOLVER_MT_DIRECT_BATCHED_ROWMAJOR;
                                curr_sol->solver->execute_solver =
                                    register_execute_mt_direct_batched_rowmajor_solver();
                            }
                        }
                        // TODO: use row-major variant in else case

                        curr_sol->decomp_scheme->thread_info->n_threads =
                            batched_n_threads;
                    }
                    else
                    {
#endif
                        if (is_col_major)
                        {
                            curr_sol->solver->solver_type =
                                SOLVER_DIRECT_BATCHED_COLMAJOR;
                            curr_sol->solver->execute_solver =
                                register_execute_direct_batched_colmajor_solver();
                        }
                        else
                        {
                            curr_sol->solver->solver_type =
                                SOLVER_DIRECT_BATCHED_ROWMAJOR;
                            curr_sol->solver->execute_solver =
                                register_execute_direct_batched_rowmajor_solver();
                        }
#ifdef MULTI_THREADING
                    }
#endif
                    // allocate a new struct in the direct solver to hold the
                    // vecs[0] from batched solver
                    ALLOC_ALIGN_UNINIT(curr_sol->decomp_scheme->batched_vecs,
                                    aoclfftz_dim_t_64_,
                                    sizeof(aoclfftz_dim_t_64_));
                    curr_sol->decomp_scheme->batched_vecs[0].n = batched_vec_n;
                    curr_sol->decomp_scheme->batched_vecs[0].in_stride =
                        batched_vec_in_stride;
                    curr_sol->decomp_scheme->batched_vecs[0].out_stride =
                        batched_vec_out_stride;
                    AOCLFFTZ_LOG(
                        TRACE,
                        global_logger_mode,
                        " -> sub-problem: "
                        "[%td:%td:%td] %td:%td:%td v %td:%td:%td",
                        curr_sol->decomp_scheme->vecs[0].n,
                        curr_sol->decomp_scheme->vecs[0].in_stride,
                        curr_sol->decomp_scheme->vecs[0].out_stride,
                        curr_sol->decomp_scheme->batched_vecs[0].n,
                        curr_sol->decomp_scheme->batched_vecs[0].in_stride,
                        curr_sol->decomp_scheme->batched_vecs[0].out_stride,
                        curr_sol->decomp_scheme->dims[0].n,
                        curr_sol->decomp_scheme->dims[0].in_stride,
                        curr_sol->decomp_scheme->dims[0].out_stride);

                    // For 3D unit-strided C2C problems, when compact buffer approach is used
                    // reduce the strides of ct_buffer to match the smaller ct_buffer size
                    INTP radix = curr_sol->decomp_scheme->dims[0].n;
                    if (curr_sol->next_sol) // twiddle kernel
                    {
                        // reduce the out-strides of outermost CT-r twiddle kernel
                        if (curr_sol->decomp_scheme->decomp_level == 0)
                        {
                            curr_sol->decomp_scheme->dims[0].in_stride /= ct_buffer_stride_factor;
                            curr_sol->decomp_scheme->vecs[0].in_stride /= ct_buffer_stride_factor;
                            for (INTP i = 0; i < radix; i++)
                            {
                                curr_sol->strides_grp->strides->in_strides[i] /= ct_buffer_stride_factor;
                            }
                        }
                        // reduce both in and out-strides of remaining CT-r twiddle kernels
                        else
                        {
                            curr_sol->decomp_scheme->dims[0].in_stride /= ct_buffer_stride_factor;
                            curr_sol->decomp_scheme->vecs[0].in_stride /= ct_buffer_stride_factor;
                            curr_sol->decomp_scheme->dims[0].out_stride /= ct_buffer_stride_factor;
                            curr_sol->decomp_scheme->vecs[0].out_stride /= ct_buffer_stride_factor;
                            for (INTP i = 0; i < radix; i++)
                            {
                                curr_sol->strides_grp->strides->in_strides[i] /= ct_buffer_stride_factor;
                                curr_sol->strides_grp->strides->out_strides[i] /= ct_buffer_stride_factor;
                            }
                        }
                    }
                    else
                    {
                        // reduce the in-strides of CT-m standard kernel
                        curr_sol->decomp_scheme->dims[0].out_stride /= ct_buffer_stride_factor;
                        curr_sol->decomp_scheme->vecs[0].out_stride /= ct_buffer_stride_factor;
                        for (INTP i = 0; i < radix; i++)
                        {
                            curr_sol->strides_grp->strides->out_strides[i] /= ct_buffer_stride_factor;
                        }
                    }
                    if (is_col_major)
                    {
                        // update vec strides
                        curr_sol->strides_grp->strides->v_in_stride =
                            batched_vec_in_stride * DATA_STRIDE;
                        curr_sol->strides_grp->strides->v_out_stride =
                            batched_vec_out_stride * DATA_STRIDE;
                    }
                }
                else // other than direct solvers
                {
                    if (curr_sol->dft_bufs->use_2D_buffering)
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
                            curr_sol->decomp_scheme->dims[0].out_stride /= ct_buffer_stride_factor;
                            curr_sol->decomp_scheme->vecs[0].out_stride /= ct_buffer_stride_factor;
                        }
                    }
                }

                curr_sol = curr_sol->next_sol ? curr_sol->next_sol[0] : NULL;
            }
        }
        else
        {
            if (curr_sol->dft_bufs->use_2D_buffering)
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
    post_process_for_optimal_buffering_batching_(sol, NULL,
                                                 sol->decomp_scheme->dim_rank);
#else
    return;
#endif
}
