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

/** @file batched_direct_utils.c
 *
 *  @brief Post-selection utilities that enable the Batched-Direct solver path.
 *
 *  This module walks the selected solution graph and, when eligible, converts a
 *  Direct CT sub-problem into a Batched-Direct sub-problem by:
 *  - moving one level of vectorization (the innermost `vecs[0]`) from the
 *    preceding Batched node into the CT Direct node as `batched_vecs[0]`
 *  - changing the solver type and `execute_solver` function pointer
 *
 *  Known limitations:
 *  - Bluestein problems are excluded
 *  - Real variants (R2C/C2R) are not supported
 *  - Only column-major batched complex CT problems are supported
 *  - Complex multi-threaded problems have scaling issues
 *
 *  This pass is currently applied as a post-selection rewrite of the solution.
 *
 *  @author Srirammaswamy Srinivasan
 */

#include "core/common/batched_direct_utils.h"
#include "api/aoclfftz_internal.h"
#include "core/solvers/solver.h"
#include "selector/selector.h"
#include "utils/allocator.h"
#include "utils/utils.h"

#ifndef DISABLE_BATCHED_DIRECT_SOLVER

/**
 * @brief Check whether the current position is eligible for Batched-Direct.
 *
 * Evaluates structural and layout constraints around the current solution node
 * to determine whether a Batched-Direct conversion can be applied. The checks
 * include bluestein detection, solver ordering (prev/curr/next), and whether
 * the problem has a column-major layout.
 *
 * @param sol      Current solution node (expected to be a Batched solver node).
 * @param prev_sol Previous solution node, or NULL if at the root.
 * @return INT32   1 if eligible, 0 otherwise.
 */
static INT32 is_eligible_for_batched_direct(aoclfftz_solution_t *sol,
                                            aoclfftz_solution_t *prev_sol)
{
    // TODO: support for bluestein solver
    // currently, we are not supporting batched-direct when bluestein problem is detected
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

    // TODO: Add MT & REAL variants
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
    INT32 dim_rank = sol->decomp_scheme->dim_rank;
    aoclfftz_dim_t_64_ *dims = sol->decomp_scheme->dims;
    aoclfftz_dim_t_64_ *vecs = sol->decomp_scheme->vecs;

#ifdef MULTI_THREADING
    if (dim_rank == 1 && /* 1D problem (FIXME: this should be always true, so remove this check) */
        vecs[0].n > 1 && /* batched problem */
        vecs[0].in_stride < dims[0].in_stride && /* column-major order input */
        vecs[0].out_stride < dims[0].out_stride && /* column-major order output */
        dims[0].in_stride == dims[0].out_stride && /* FIXME: support different in/out strides */
        vecs[0].in_stride == vecs[0].out_stride)
#else
    if (dim_rank == 1 && /* 1D problem (FIXME: this should be always true, so remove this check) */
        vecs[0].n > 1 && /* batched problem */
        dims[0].in_stride == dims[0].out_stride && /* FIXME: support different in/out strides */
        vecs[0].in_stride == vecs[0].out_stride)
#endif
    {
        AOCLFFTZ_LOG_FORMATTED(
            TRACE, sol->decomp_scheme->cntrl_params->logger_mode,
            "using batched-direct for problem: "
            "%td:%td:%td v %td:%td:%td",
            vecs[0].n, vecs[0].in_stride, vecs[0].out_stride, dims[0].n,
            dims[0].in_stride, dims[0].out_stride);

        return 1; // return true
    }

    return 0; // return false
}

/**
 * @brief Traverse and rewrite eligible Batched -> CT sub-paths in place and
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
 * @param curr_sol Current solution being inspected.
 * @param prev_sol Previous solution context for eligibility checks.
 */
static VOID apply_batched_direct_solver_internal(aoclfftz_solution_t *curr_sol,
                                                 aoclfftz_solution_t *prev_sol)
{
    // iterate through the solution list on ndim branches first
    while (curr_sol)
    {
        // traverse nd_sol first so inner batched nodes after NDim are handled
        if (curr_sol->solver->solver_type == SOLVER_NDIM &&
            curr_sol->dft_bufs->nd_sol)
        {
            apply_batched_direct_solver_internal(curr_sol->dft_bufs->nd_sol,
                                                 curr_sol);
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
            // get the vecs[0] values before removing them in the batched_sol
            INTP batched_vec_n = vecs[0].n;
            INTP batched_vec_in_stride = vecs[0].in_stride;
            INTP batched_vec_out_stride = vecs[0].out_stride;

            // check if the problem is col-major or row-major
            // col-major: vec-strides < elemental-strides
            // row-major: elemental-strides < vec-strides
            UINT8 is_col_major =
                (batched_sol->decomp_scheme->vecs[0].in_stride <
                 batched_sol->decomp_scheme->dims[0].in_stride) &&
                (batched_sol->decomp_scheme->vecs[0].out_stride <
                 batched_sol->decomp_scheme->dims[0].out_stride);

            // do the following changes to the batched solver:
            // - reduce the vec_rank by 1 (i.e. remove the vec[0])
            // - make the multi-threaded batched_solver as single threaded
            // - update solver_type, execute_solver ptr & thread_info
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
                // remove the innermost batches (i.e. remove the vec[0])
                batched_sol->decomp_scheme->vec_rank -= 1;
                for (INT32 i = 0; i < batched_sol->decomp_scheme->vec_rank;
                        i++)
                {
                    vecs[i].n = vecs[i + 1].n;
                    vecs[i].in_stride = vecs[i + 1].in_stride;
                    vecs[i].out_stride = vecs[i + 1].out_stride;
                }
            }
#ifdef MULTI_THREADING
            // get the number of threads from the batched solver before making
            // it single threaded
            UINT32 batched_n_threads =
                batched_sol->decomp_scheme->thread_info->n_threads;
            UINT32 batched_avl_threads =
                batched_sol->decomp_scheme->thread_info->avl_threads;
            // make the multi-threaded batched_solver as single threaded
            if (batched_sol->solver->solver_type == SOLVER_MT_BATCHED)
            {
                batched_sol->decomp_scheme->thread_info->n_threads = 1;
                batched_sol->decomp_scheme->thread_info->avl_threads = 1;
                batched_sol->solver->solver_type = SOLVER_BATCHED;
                batched_sol->solver->execute_solver =
                    register_execute_batched_solver();
            }
#endif

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
                    if (batched_n_threads > 1)
                    {
                        if (is_col_major)
                        {
                            // FIXME: a temporary fix to use different MT solvers
                            // for 1D and ND problems; it is based on the
                            // performance study for 1D and 3D problems
                            // TODO: remove this once we have an MT solver
                            // which works for all the cases
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
                        else
                        {
                            // TODO: use row-major variant for this case
                        }
                        // TODO: Verify whether assigning both n_threads and
                        //       avl_threads are required or not
                        curr_sol->decomp_scheme->thread_info->n_threads =
                            batched_n_threads;
                        curr_sol->decomp_scheme->thread_info->avl_threads =
                            batched_avl_threads;
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
                    // update vec-strides for kernels only for column-major problems
                    if (is_col_major)
                    {
                        // update vec strides
                        curr_sol->strides_grp->strides->v_in_stride =
                            batched_vec_in_stride * DATA_STRIDE;
                        curr_sol->strides_grp->strides->v_out_stride =
                            batched_vec_out_stride * DATA_STRIDE;
                    }
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
                    AOCLFFTZ_LOG_FORMATTED(
                        TRACE,
                        curr_sol->decomp_scheme->cntrl_params->logger_mode,
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
                }
                curr_sol = curr_sol->next_sol ? curr_sol->next_sol[0] : NULL;
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

VOID apply_batched_direct_solver(aoclfftz_solution_t *sol)
{
#ifndef DISABLE_BATCHED_DIRECT_SOLVER
    apply_batched_direct_solver_internal(sol, NULL);
#else
    return;
#endif
}
