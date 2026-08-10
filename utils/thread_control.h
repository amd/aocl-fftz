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

#endif // MULTI_THREADING

#endif // AOCLFFTZ_THREAD_CONTROL_H

