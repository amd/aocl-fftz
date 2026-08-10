// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file thread_control.c
 *
 *  @brief Threading policy for the solver tree.
 *
 *  This file contains the implementation of the helpers that bound the thread
 *  count each level of the solver tree may keep, along with the tuning knobs
 *  those decisions are based on.
 *
 *  @author Ashwin K. Godbole
 */

#include "utils/thread_control.h"

#ifdef MULTI_THREADING

// The thread budget flows down the solver tree, each level keeping a share:
//
//   avl_threads --> batched solver --> batch loop: one transform per thread
//                        |
//                        +-- quotient --> child: nested split *inside* each
//                                                transform
//
// The batch loop is the cheaper of the two axes, so a level takes what it can
// from the batch first and only passes the remainder down. Neither axis scales
// without bound, so the rules below bound what each level may keep.
//
// All of it is opt-in. Under dynamic_load_model = 0 the caller has asked for
// num_threads to be spent as given, so every cap below returns its input
// untouched and the budget reaches the solvers whole.

static FFTZ_INT32 dynamic_load_model_on(aoclfftz_decomp_scheme_t *decomp_scheme)
{
    return decomp_scheme->thread_info->pthr_fft->dynamic_load_model == 1;
}

static FFTZ_INTP count_dim_elems(aoclfftz_decomp_scheme_t *decomp_scheme)
{
    FFTZ_INTP n = 1;
    for (FFTZ_INT32 rnk = 0; rnk < decomp_scheme->dim_rank; rnk++)
    {
        n *= decomp_scheme->dims[rnk].n;
    }
    return n;
}

// The batch a level runs over is not vecs[] alone. A batched-direct solver
// parks the problem batch in batched_vecs and leaves only the CT batch in
// vecs, so counting vecs[] alone understates the work by that factor and
// starves the level of threads it could have used.
static FFTZ_INTP count_vec_elems(aoclfftz_decomp_scheme_t *decomp_scheme)
{
    FFTZ_INTP batch = 1;
    for (FFTZ_INT32 rnk = 0; rnk < decomp_scheme->vec_rank; rnk++)
    {
        batch *= decomp_scheme->vecs[rnk].n;
    }
    if (decomp_scheme->batched_vecs != NULL)
    {
        batch *= decomp_scheme->batched_vecs[0].n;
    }
    return batch;
}

// Threads that split one transform between them synchronise part-way through
// rather than once at the end, so against batch-loop threads they need more
// work each, and they stop paying past a team the decomposition cannot feed.
//
// Where they come from decides how far they go, so the ceiling is per rank:
//
//   dim_rank > 1  A dimension is peeled off and becomes a batch axis, so the
//                 threads are really batch-loop threads over planes or lines
//                 and only stop at the width of a memory domain.
//   dim_rank == 1 There is no axis to peel. Threads come from splitting the
//                 Cooley-Tukey stages of the one transform, which costs a
//                 barrier per stage and stops paying much sooner.
#define FFTZ_MIN_ELEMS_PER_INTRA_THREAD 2048
#define FFTZ_MAX_INTRA_TRANSFORM_THREADS 32
#define FFTZ_MAX_CT_SPLIT_THREADS 16

static FFTZ_INTP intra_transform_thread_limit(FFTZ_INTP n, FFTZ_INT32 dim_rank)
{
    FFTZ_INTP ceiling = (dim_rank > 1) ? FFTZ_MAX_INTRA_TRANSFORM_THREADS
                                       : FFTZ_MAX_CT_SPLIT_THREADS;

    FFTZ_INTP max_threads = n / FFTZ_MIN_ELEMS_PER_INTRA_THREAD;
    if (max_threads > ceiling)
    {
        max_threads = ceiling;
    }
    return (max_threads < 1) ? 1 : max_threads;
}

FFTZ_INT32 cap_nested_thread_budget(aoclfftz_decomp_scheme_t *decomp_scheme,
                                    FFTZ_INT32 child_threads)
{
    if (!dynamic_load_model_on(decomp_scheme))
    {
        return child_threads;
    }

    if (child_threads <= 1)
    {
        return 1;
    }

    FFTZ_INTP max_threads = intra_transform_thread_limit(
        count_dim_elems(decomp_scheme), decomp_scheme->dim_rank);

    return (max_threads < (FFTZ_INTP)child_threads) ? (FFTZ_INT32)max_threads
                                                    : child_threads;
}

// Below this, a batch-loop thread costs more in team startup and barrier
// traffic than the arithmetic it saves.
#define FFTZ_MIN_ELEMS_REQ_PER_THREAD 256

FFTZ_INT32 cap_batch_loop_threads(aoclfftz_decomp_scheme_t *decomp_scheme,
                                  FFTZ_INT32 n_threads)
{
    if (!dynamic_load_model_on(decomp_scheme))
    {
        return n_threads;
    }

    FFTZ_INTP elems =
        count_dim_elems(decomp_scheme) * count_vec_elems(decomp_scheme);

    FFTZ_INTP max_threads = elems / FFTZ_MIN_ELEMS_REQ_PER_THREAD;
    if (max_threads < 1)
    {
        max_threads = 1;
    }

    return (max_threads < (FFTZ_INTP)n_threads) ? (FFTZ_INT32)max_threads
                                                : n_threads;
}

// Runs before planning, for two reasons: the decomposition depends on the
// budget the planner sees, so a later trim would leave a plan shaped for a
// different count; and the solver type is chosen from avl_threads, so trimming
// further down would leave a multi-threaded solver driving a one-thread plan.
FFTZ_VOID cap_plan_thread_budget(aoclfftz_decomp_scheme_t *decomp_scheme)
{
    if (!dynamic_load_model_on(decomp_scheme))
    {
        return;
    }

    thread_info_t *thread_info = decomp_scheme->thread_info;

    // Two independent bounds, both of which have to hold.
    //
    // The first is the total work, and it is the same bound the batch loop
    // obeys: a batch of tiny transforms cannot feed a large team however many
    // members it has.
    FFTZ_INT32 max_threads =
        cap_batch_loop_threads(decomp_scheme, thread_info->avl_threads);

    // The second is the widest axis the plan can actually spread over. The
    // batch loop is that axis whenever the batch is wide, and it scales, so a
    // wide batch is left alone. A narrow batch cannot absorb the budget on its
    // own; the surplus has to nest inside the transforms, and nesting is what
    // stops scaling, so the whole plan is held to the intra-transform limit
    // rather than to that limit once per batch element.
    FFTZ_INTP widest_axis = intra_transform_thread_limit(
        count_dim_elems(decomp_scheme), decomp_scheme->dim_rank);
    FFTZ_INTP batch = count_vec_elems(decomp_scheme);
    if (batch > widest_axis)
    {
        widest_axis = batch;
    }
    if ((FFTZ_INTP)max_threads > widest_axis)
    {
        max_threads = (FFTZ_INT32)widest_axis;
    }

    thread_info->avl_threads = max_threads;
}

#endif // MULTI_THREADING

