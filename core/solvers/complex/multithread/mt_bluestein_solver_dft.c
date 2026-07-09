// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file mt_bluestein_solver_dft.c
 *
 *  @brief Multi threaded Bluestein FFT solver that parallelises the
 *  pre_mul, mul, and post_mul steps across the available kernel thread budget
 *
 *  This file contains the functions that setup and execute the solver.
 *
 *  @author Jeevanantham N
 */

#include "core/common/memory_manager.h"
#include "core/kernels/kernel.h"
#include "utils/utils.h"

/**
 * @brief Returns thread tid's contiguous [start, start + size) slice of n
 * complex elements, split as evenly as possible across n_threads workers.
 *
 * When n does not divide evenly, the first (n % n_threads) threads each take
 * one extra element (e.g. n=10, n_threads=3 -> chunks of 4, 3, and 3).
 *
 * @param[in]  tid        Thread index in [0, n_threads).
 * @param[in]  n          Total complex elements to split.
 * @param[in]  n_threads  Number of workers.
 * @param[out] start      First element index owned by tid.
 * @param[out] size       Element count owned by tid (0 if none).
 */
static inline FFTZ_VOID thread_elem_range(FFTZ_INT32 tid, FFTZ_INTP n,
                                          FFTZ_INT32 n_threads, FFTZ_INTP *start,
                                          FFTZ_INTP *size)
{
    FFTZ_INTP elems_per_thread = n / n_threads;
    FFTZ_INTP extra_elems = n % n_threads;

    *start = tid * elems_per_thread + (tid < extra_elems ? tid : extra_elems);
    *size = elems_per_thread + (tid < extra_elems ? 1 : 0);
}

/**
 * @brief Elementwise multiply (out = a .* b) over n complex elements, split
 * into static contiguous chunks across n_threads workers.
 *
 * @param[in]  kernel     Elementwise multiplication kernel.
 * @param[out] out        Destination buffer (out = a .* b).
 * @param[in]  a          First operand buffer.
 * @param[in]  b          Second operand buffer.
 * @param[in]  n          Number of complex elements to process.
 * @param[in]  start_idx  Logical element index offset (used by strided variants).
 * @param[in]  stride     Stride in complex elements (used by strided variants).
 * @param[in]  n_threads  Number of OpenMP workers to dispatch.
 * @param[in]  elem_bytes Byte stride per complex element
 *                        (DATA_STRIDE * dt_bytes).
 */
static inline FFTZ_VOID mt_ele_mul_dispatch(elementwise_mul_ kernel,
                                            FFTZ_VOID *out, FFTZ_VOID *a,
                                            FFTZ_VOID *b, FFTZ_INTP n,
                                            FFTZ_INTP start_idx,
                                            FFTZ_INTP stride,
                                            FFTZ_INT32 n_threads,
                                            FFTZ_INTP elem_bytes)
{
    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (FFTZ_INT32 tid = 0; tid < n_threads; tid++)
    {
        FFTZ_INTP start_elem, num_elems;
        thread_elem_range(tid, n, n_threads, &start_elem, &num_elems);
        if (num_elems <= 0)
        {
            continue;
        }
        FFTZ_INTP thread_offset = start_elem * elem_bytes;
        kernel(MOVE_ADDR(out, thread_offset), MOVE_ADDR(a, thread_offset),
               MOVE_ADDR(b, thread_offset), num_elems,
               start_idx + start_elem, stride);
    }
}

/**
 * @brief Elementwise multiply (out = a .* b) with strided first operand,
 * split into static contiguous chunks across n_threads workers.
 *
 * Used by pre_mul when caller input is strided (in_stride > 1). Each thread
 * gets a contiguous slice of @p out and @p b; @p a stays at the caller base
 * and the thread's element start index is passed into the strided-in kernel.
 *
 * @param[in]  kernel     Elementwise multiplication kernel 
 *                        (strided-in variant).
 * @param[out] out        Destination buffer (out = a .* b), chunked by thread.
 * @param[in]  a          Strided first operand, shared base across all
 *                        workers.
 * @param[in]  b          Second operand buffer, chunked by thread.
 * @param[in]  n          Number of complex elements to process.
 * @param[in]  in_stride  Stride in complex elements between successive @p a
 *                        samples.
 * @param[in]  n_threads  Number of OpenMP workers to dispatch.
 * @param[in]  elem_bytes Byte stride per complex element
 *                        (DATA_STRIDE * dt_bytes).
 */
static inline FFTZ_VOID
mt_ele_mul_strided_in_dispatch(elementwise_mul_ kernel, FFTZ_VOID *out,
                               FFTZ_VOID *a, FFTZ_VOID *b, FFTZ_INTP n,
                               FFTZ_INTP in_stride, FFTZ_INT32 n_threads,
                               FFTZ_INTP elem_bytes)
{
#pragma omp parallel for num_threads(n_threads) schedule(static)
    for (FFTZ_INT32 tid = 0; tid < n_threads; tid++)
    {
        FFTZ_INTP start_elem, num_elems;
        thread_elem_range(tid, n, n_threads, &start_elem, &num_elems);
        if (num_elems <= 0)
        {
            continue;
        }
        FFTZ_INTP thread_offset = start_elem * elem_bytes;
        kernel(MOVE_ADDR(out, thread_offset), a, MOVE_ADDR(b, thread_offset),
               num_elems, start_elem, in_stride);
    }
}

/**
 * @brief Parallel Bluestein pre_mul: routes to contiguous or strided-in
 * elementwise dispatch based on @p in_stride.
 *
 * @param[in]  kernel     pre_mul kernel (contiguous or strided-in).
 * @param[out] out        Bluestein workspace output (contiguous).
 * @param[in]  a          Caller input buffer.
 * @param[in]  b          Chirp sequence B.
 * @param[in]  n          Original problem length.
 * @param[in]  in_stride  Caller input stride in complex elements.
 * @param[in]  n_threads  OpenMP worker count.
 * @param[in]  elem_bytes Byte stride per complex element.
 */
static inline FFTZ_VOID
mt_pre_mul_dispatch(elementwise_mul_ kernel, FFTZ_VOID *out, FFTZ_VOID *a,
                    FFTZ_VOID *b, FFTZ_INTP n, FFTZ_INTP in_stride,
                    FFTZ_INT32 n_threads, FFTZ_INTP elem_bytes)
{
    if (in_stride == 1)
    {
        mt_ele_mul_dispatch(kernel, out, a, b, n, 0, 1, n_threads, elem_bytes);
    }
    else
    {
        mt_ele_mul_strided_in_dispatch(kernel, out, a, b, n, in_stride,
                                       n_threads, elem_bytes);
    }
}

/**
 * @brief Parallel Bluestein post_mul: normalize and apply chirp into caller
 * output, split into static contiguous chunks across n_threads workers.
 *
 * @param[in]  kernel     post_mul kernel.
 * @param[out] out        Caller output buffer (strided when out_stride > 1).
 * @param[in]  a          Bluestein workspace after inverse FFT.
 * @param[in]  b          chirp buffer (B).
 * @param[in]  n          Original problem length.
 * @param[in]  factor     Normalization factor (1/m).
 * @param[in]  out_stride Caller output stride in complex elements.
 * @param[in]  n_threads  OpenMP worker count.
 * @param[in]  elem_bytes Byte stride per complex element.
 */
static inline FFTZ_VOID
mt_post_mul_dispatch(elementwise_mul_fused_norm_ kernel, FFTZ_VOID *out,
                     FFTZ_VOID *a, FFTZ_VOID *b, FFTZ_INTP n,
                     FFTZ_DOUBLE factor, FFTZ_INTP out_stride,
                     FFTZ_INT32 n_threads, FFTZ_INTP elem_bytes)
{
#pragma omp parallel for num_threads(n_threads) schedule(static)
    for (FFTZ_INT32 tid = 0; tid < n_threads; tid++)
    {
        FFTZ_INTP start_elem, num_elems;
        thread_elem_range(tid, n, n_threads, &start_elem, &num_elems);
        if (num_elems <= 0)
        {
            continue;
        }
        FFTZ_INTP in_offset = start_elem * elem_bytes;
        FFTZ_INTP out_offset = start_elem * out_stride * elem_bytes;
        kernel(MOVE_ADDR(out, out_offset), MOVE_ADDR(a, in_offset),
               MOVE_ADDR(b, in_offset), num_elems, factor, out_stride);
    }
}

/**
 * @brief Sets up the MT Bluestein solver with extended length buffers.
 *
 * Initializes the next solution object with extended length m, allocates
 * the internal buffers (B and B_out), and snapshots avl_threads into
 * thread_info->n_threads as the per-invocation kernel thread budget for
 * ele_mul dispatch.
 *
 * @param[in,out] sol      Current solution object
 * @param[out]    next_sol Next solution to configure
 * @param[in]     m        Extended length (must be >= 2*n-1 and factorable)
 * @return FFTZ_INT32 SOLVER_SUCCESS on success, error code on failure
 */
FFTZ_INT32 setup_mt_bluestein_solver(aoclfftz_solution_t *sol,
                                aoclfftz_solution_t *next_sol, FFTZ_INTP m,
                                FFTZ_UINT8 *has_nested)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    thread_info_t *thread_info = sol->decomp_scheme->thread_info;
    if (thread_info->active_threads != 1)
    {
        *has_nested = 1;
    }

    // Setup next_sol with extended length m
    FFTZ_INT32 ret = copy_solution_obj(next_sol, sol);
    if (ret != AOCLFFTZ_SUCCESS)
    {
        AOCLFFTZ_ERROR("copy_solution_obj failed: %s", get_status_string(ret));
        return ret;
    }
    next_sol->decomp_scheme->dims[0].n = m;
    next_sol->decomp_scheme->dims[0].in_stride = 1;
    next_sol->decomp_scheme->dims[0].out_stride = 1;

    FFTZ_UINT32 dt_bytes = SOL_DT_SIZE(sol);

    // Byte size of m complex elements, padded to 64-byte alignment.
    FFTZ_INTP bs_buf_size = GET_PADDED_SIZE(
        (FFTZ_INTP)m * DATA_STRIDE * dt_bytes);
    ret = alloc_bluestein_buffers(sol->dft_bufs->bluestein, bs_buf_size);
    if (ret != AOCLFFTZ_SUCCESS)
    {
        return ret;
    }

    // Snapshot avl_threads as the per-invocation kernel thread budget.
    sol->decomp_scheme->thread_info->n_threads =
        sol->decomp_scheme->thread_info->avl_threads;

    next_sol->decomp_scheme->in_real = NULL;
    next_sol->decomp_scheme->in_imag = NULL;
    next_sol->decomp_scheme->out_real = NULL;
    next_sol->decomp_scheme->out_imag = NULL;

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SOLVER_SUCCESS;
}

/**
 * @brief Executes the Bluestein FFT algorithm with parallel kernel dispatch.
 *
 * Algorithm Overview:
 * 1. Multiply input by chirp sequence B (pre-processing)
 * 2. Zero-pad the multiplied input to extended length m
 * 3. Perform forward FFT on the padded sequence
 * 4. Multiply with pre-computed FFT of chirp sequence B
 * 5. Perform inverse FFT
 * 6. Post-process: fused normalize + chirp multiply with B
 *
 * The pre_mul, mul, and post_mul steps are chunked across thread_info->n_threads via
 * the mt_*_dispatch helpers; the inner FFT(m) subtree threads independently.
 *
 * @param[in,out] sol Solution object containing problem configuration
 * @param[in,out] ctx Per-call execution context
 * @return FFTZ_INT32 SOLVER_SUCCESS on success, SOLVER_FAILURE on error
 */
static FFTZ_INT32 execute_mt_bluestein_solver(aoclfftz_solution_t *sol,
                                              aoclfftz_mutable_ctx_t *ctx)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");


    aoclfftz_solution_t *next_sol = sol->next_sol[0];
    aoclfftz_bluestein_t *bluestein = sol->dft_bufs->bluestein;
    FFTZ_UINT8 dt_prec = DT_PRECISION_FLAG(ctx->flags);
    FFTZ_UINT32 dt_bytes = DT_PRECISION_BYTES(dt_prec);
    FFTZ_UINT32 dir = FFT_DIR(ctx->flags);
    FFTZ_INT32 n_threads = sol->decomp_scheme->thread_info->n_threads;
    FFTZ_INTP elem_bytes = (FFTZ_INTP)DATA_STRIDE * dt_bytes;
    aoclfftz_mutable_ctx_t bs_ctx = *ctx;

    // Bluestein convolution doesn't use ct_buffer, but reset
    // ct_offset anyway to avoid invalid values downstream.
    bs_ctx.ct_offset = 0;

    // Two-level split of the shared bs pool: bs_dim_offset selects this dim's slice,
    // then bs_buf_size * bs_slot_idx picks this thread's slot within it.
    FFTZ_INTP bs_buf_offset = bluestein->bs_dim_offset +
                              bluestein->bs_buf_size * ctx->bs_slot_idx;

    bs_ctx.in_real     = MOVE_ADDR(ctx->bs_in_base, bs_buf_offset);
    bs_ctx.in_imag     = MOVE_ADDR(bs_ctx.in_real, dt_bytes);
    bs_ctx.out_real    = MOVE_ADDR(ctx->bs_out_base, bs_buf_offset);
    bs_ctx.out_imag    = MOVE_ADDR(bs_ctx.out_real, dt_bytes);

    // next_sol inherits the requested direction, but the convolution always
    // runs a forward FFT at step 2a and an inverse FFT at step 2c. The kernels
    // pick forward/inverse from this flag, so force forward here for a backward
    // request; step 2c restores backward for the inverse transform.
    if (dir == BACKWARD_FFT_DIR)
    {
        SET_FFT_DIR(bs_ctx.flags, FORWARD_FFT_DIR);
    }

    FFTZ_INTP n = sol->decomp_scheme->dims[0].n;      // Original length
    FFTZ_INTP m = next_sol->decomp_scheme->dims[0].n; // Extended length
    FFTZ_INT32 status = SOLVER_SUCCESS;

    FFTZ_VOID *bs_in_real  = bs_ctx.in_real;
    FFTZ_VOID *bs_out_real = bs_ctx.out_real;

    // Current solution I/O buffers
    FFTZ_VOID *cur_in = ctx->in_real;
    FFTZ_VOID *cur_out = ctx->out_real;

    //=========================================================================
    // Step 1: Gather input and apply chirp pre-processing
    //=========================================================================
    mt_pre_mul_dispatch(bluestein->pre_mul[dir], bs_in_real, cur_in, bluestein->B,
                        n, sol->decomp_scheme->dims[0].in_stride, n_threads,
                        elem_bytes);

    // Zero-pad the input from index n to m-1
    memset(MOVE_ADDR(bs_in_real, n * elem_bytes), 0, (m - n) * elem_bytes);

    //=========================================================================
    // Step 2: Convolution via FFT
    //=========================================================================

    // 2a. Forward FFT of pre-processed input.
    //     FFT of chirp sequence B is pre-computed during plan setup
    //     ct_buf_base is set to this thread's private out slot
    bs_ctx.ct_buf_base = bs_out_real;

    status = next_sol->solver->execute_solver(next_sol, &bs_ctx);
    if (status != SOLVER_SUCCESS)
    {
        goto exit_mt_bluestein_solver;
    }

    // 2b. Pointwise multiplication: A_out × B_out (with conjugate for inverse)
    mt_ele_mul_dispatch(bluestein->mul[!dir], bs_out_real, bs_out_real,
                        bluestein->B_out, m, 0, 1, n_threads, elem_bytes);

    // 2c. Inverse FFT of the product.
    bs_ctx.in_real     = bs_out_real;
    bs_ctx.in_imag     = MOVE_ADDR(bs_out_real, dt_bytes);
    bs_ctx.out_real    = bs_in_real;
    bs_ctx.out_imag    = MOVE_ADDR(bs_in_real, dt_bytes);
    bs_ctx.ct_buf_base = bs_ctx.out_real;
    SET_FFT_DIR(bs_ctx.flags, BACKWARD_FFT_DIR);

    status = next_sol->solver->execute_solver(next_sol, &bs_ctx);
    if (status != SOLVER_SUCCESS)
    {
        goto exit_mt_bluestein_solver;
    }

    //=========================================================================
    // Step 3: fused normalize+chirp multiply.
    //=========================================================================
    mt_post_mul_dispatch(bluestein->post_mul[dir], cur_out, bs_in_real,
                         bluestein->B, n, 1.0 / (FFTZ_DOUBLE)m,
                         sol->decomp_scheme->dims[0].out_stride, n_threads,
                         elem_bytes);

exit_mt_bluestein_solver:
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return status;
}

dft_solver_ register_execute_mt_bluestein_solver(FFTZ_VOID)
{
    return execute_mt_bluestein_solver;
}
