// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file thread_control.h
 *
 *  @brief Threading policy for the solver tree.
 *
 *  This file exposes the helpers that decide how many threads a level of the
 *  solver tree may keep and whether a level may split its work further. The
 *  decisions are policy only: they read the decomposition and return or trim a
 *  thread count, and never create a team themselves.
 *
 *  The policy is only in effect under dynamic_load_model = 1. With the model
 *  off, every helper below returns the count it was given.
 *
 *  @author Ashwin K. Godbole
 */

#ifndef AOCLFFTZ_THREAD_CONTROL_H
#define AOCLFFTZ_THREAD_CONTROL_H

#include "api/aoclfftz_internal.h"

#ifdef MULTI_THREADING

/**
 * @brief Trim the share a batched solver's child inherits.
 *
 * The batch loop leaves the child whatever the batch could not absorb. That
 * surplus is only worth keeping while the transform can feed a nested split.
 *
 * @param decomp_scheme Decomposition of the transform under the batch loop
 * @param child_threads Budget the child would otherwise inherit
 * @return @p child_threads, or a smaller count if the transform is too narrow
 */
FFTZ_INT32 cap_nested_thread_budget(aoclfftz_decomp_scheme_t *decomp_scheme,
                                    FFTZ_INT32 child_threads);

/**
 * @brief Trim a batched solver's team to what the problem can feed.
 *
 * @param decomp_scheme Decomposition the batch loop runs over
 * @param n_threads     Team size the caller would otherwise use
 * @return @p n_threads, or a smaller count if the work does not justify it
 */
FFTZ_INT32 cap_batch_loop_threads(aoclfftz_decomp_scheme_t *decomp_scheme,
                                  FFTZ_INT32 n_threads);

/**
 * @brief Lower thread_info->avl_threads to what the problem can feed, before
 *        planning begins.
 *
 * @param decomp_scheme Decomposition of the top-level problem
 */
FFTZ_VOID cap_plan_thread_budget(aoclfftz_decomp_scheme_t *decomp_scheme);

/**
 * @brief Trim a Real-kernel OpenMP team to what the kernel loop can feed.
 *
 * @param decomp_scheme Decomposition of the transform being executed
 * @param kernel_count  Total r2hc + r2hcf kernel invocations
 * @param n_threads     Team size the caller would otherwise use
 * @return @p n_threads, or a smaller count if the work does not justify it
 */
FFTZ_INT32 cap_real_kernel_loop_threads(aoclfftz_decomp_scheme_t *decomp_scheme,
                                   FFTZ_INTP kernel_count, FFTZ_INT32 n_threads);

/**
 * @brief Trim a C2C outer-loop OpenMP team from iteration and inner work.
 *
 * @param decomp_scheme Decomposition of the transform being executed
 * @param outer_iters   Parallel loop trip count
 * @param exec_per_iter Kernel batch units each outer iteration runs
 * @param n_threads     Team size the caller would otherwise use
 * @return @p n_threads, or a smaller count if the work does not justify it
 */
FFTZ_INT32 cap_real_c2c_loop_threads(aoclfftz_decomp_scheme_t *decomp_scheme,
                                          FFTZ_INTP outer_iters,
                                          FFTZ_INTP exec_per_iter,
                                          FFTZ_INT32 n_threads);

/**
 * @brief Team size for one MT real direct stage, settled at setup time.
 *
 * Returning 1 means the stage takes the single-threaded solver instead. The
 * widest of the two loops decides whether a team forms at all; past that gate
 * the load model decides the width (or, with the model off, the budget goes
 * out whole). Under the model both kernel families run on the same team, so
 * the narrowest request wins.
 *
 * @param sol       Solution whose stage kernels are being sized
 * @param n_threads Budget the caller would otherwise use
 * @return Settled team size, or 1 to select the ST solver
 */
FFTZ_INT32 cap_real_mt_direct_threads(aoclfftz_solution_t *sol,
                                      FFTZ_INT32 n_threads);

#endif // MULTI_THREADING

#endif // AOCLFFTZ_THREAD_CONTROL_H

