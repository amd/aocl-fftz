// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file mt_bluestein_solver_dft.c
 *
 *  @brief Multi threaded Bluestein FFT solver that parallelises the
 *  ele_mul / normalize steps across the available kernel thread budget
 *
 *  This file contains the functions that setup and execute the solver.
 *
 *  @author Jeevanantham N
 */

#include "core/common/bluestein_utils.h"
#include "core/common/memory_manager.h"
#include "utils/utils.h"

/**
 * @brief Returns thread tid's contiguous [start, size) slice of num_items
 * items, split into blocks of `align` items shared evenly across the workers.
 *
 * align > 1 keeps every slice start on an `align`-item (64-byte) boundary for
 * aligned SIMD; align == 1 is a plain even split. The last working thread
 * absorbs the partial trailing block.
 *
 * @param[in]  tid                Thread index in [0, n_threads).
 * @param[in]  blocks_per_thread  Blocks per thread (num_blocks / n_threads).
 * @param[in]  extra_blocks       Leftover blocks (num_blocks % n_threads), one
 *                                per thread for the first extra_blocks threads.
 * @param[in]  align              Block size: 1 or MIN_ALIGNMENT/elem_bytes.
 * @param[in]  num_items          Total items to split (clamps trailing block).
 * @param[out] start              First item index owned by tid.
 * @param[out] size               Item count owned by tid (0 if none).
 */
static inline FFTZ_VOID chunk_range(FFTZ_INT32 tid, FFTZ_INTP blocks_per_thread,
                                    FFTZ_INTP extra_blocks, FFTZ_INTP align,
                                    FFTZ_INTP num_items, FFTZ_INTP *start,
                                    FFTZ_INTP *size)
{
    // Block index range owned by this thread.
    FFTZ_INTP block_begin =
        tid * blocks_per_thread + (tid < extra_blocks ? tid : extra_blocks);
    FFTZ_INTP block_end = (tid + 1) * blocks_per_thread
                     + (tid + 1 < extra_blocks ? tid + 1 : extra_blocks);

    // Convert block range to item range.
    FFTZ_INTP elem_begin = block_begin * align;
    FFTZ_INTP elem_end = block_end * align;

    // Clamp to num_items and emit [start, size).
    *start = elem_begin < num_items ? elem_begin : num_items;
    *size = (elem_end < num_items ? elem_end : num_items) - (*start);
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
 * @param[in]  n_threads  Number of OpenMP workers to dispatch.
 * @param[in]  elem_bytes Byte stride per complex element
 *                        (DATA_STRIDE * dt_bytes).
 */
static inline FFTZ_VOID mt_ele_mul_dispatch(elementwise_mul_ kernel,
                                            FFTZ_VOID *out, FFTZ_VOID *a,
                                            FFTZ_VOID *b, FFTZ_INTP n,
                                            FFTZ_INT32 n_threads,
                                            FFTZ_INTP elem_bytes)
{
    FFTZ_INTP elems_per_thread = n / n_threads;
    FFTZ_INTP extra_elems = n % n_threads;
    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (FFTZ_INT32 tid = 0; tid < n_threads; tid++)
    {
        FFTZ_INTP start_elem, num_elems;
        chunk_range(tid, elems_per_thread, extra_elems, 1, n, &start_elem,
                    &num_elems);
        if (num_elems <= 0)
        {
            continue;
        }
        FFTZ_INTP thread_offset = start_elem * elem_bytes;
        kernel(MOVE_ADDR(out, thread_offset), MOVE_ADDR(a, thread_offset),
               MOVE_ADDR(b, thread_offset), num_elems);
    }
}

/**
 * @brief Parallel in-place normalization split across n_threads workers.
 *
 * @param[in]     kernel     Scales data in place by a factor.
 * @param[in,out] data       Buffer to normalize.
 * @param[in]     n          Number of complex elements in data.
 * @param[in]     factor     Scaling factor (typically 1.0 / m).
 * @param[in]     n_threads  Number of OpenMP workers to dispatch.
 * @param[in]     elem_bytes Byte stride per complex element
 *                           (DATA_STRIDE * dt_bytes).
 */
static inline FFTZ_VOID mt_normalize_dispatch(normalize_ kernel,
                                              FFTZ_VOID *data, FFTZ_INTP n,
                                              FFTZ_DOUBLE factor,
                                              FFTZ_INT32 n_threads,
                                              FFTZ_INTP elem_bytes)
{
    // 64-byte-aligned block split so the kernel can use aligned load/store.
    FFTZ_INTP align = MIN_ALIGNMENT / elem_bytes;
    FFTZ_INTP num_blocks = (n + align - 1) / align;
    FFTZ_INTP blocks_per_thread = num_blocks / n_threads;
    FFTZ_INTP extra_blocks = num_blocks % n_threads;
    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (FFTZ_INT32 tid = 0; tid < n_threads; tid++)
    {
        FFTZ_INTP start_elem, num_elems;
        chunk_range(tid, blocks_per_thread, extra_blocks, align, n,
                    &start_elem, &num_elems);
        if (num_elems <= 0)
        {
            continue;
        }
        kernel(MOVE_ADDR(data, start_elem * elem_bytes), num_elems, factor);
    }
}

/**
 * @brief Sets up the MT Bluestein solver with extended length buffers.
 *
 * Initializes the next solution object with extended length m, allocates
 * the internal buffers, and snapshots avl_threads into thread_info->n_threads
 * as the per-invocation kernel thread budget for ele_mul / normalize dispatch.
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

    // in/out form a pool of active_threads per-thread slots (one
    // per concurrent Bluestein invocation), each padded to MIN_ALIGNMENT
    // (64 B) so every slot base is 64-byte aligned for aligned SIMD
    // load/store in normalize.
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
 * 1. Multiply input by chirp sequence B_inv (pre-processing)
 * 2. Zero-pad the multiplied input to extended length m
 * 3. Perform forward FFT on the padded sequence
 * 4. Multiply with pre-computed FFT of chirp sequence B
 * 5. Perform inverse FFT
 * 6. Normalize and multiply by B_inv (post-processing)
 *
 * The ele_mul / normalize steps are chunked across thread_info->n_threads via
 * the mt_*_dispatch helpers; the inner FFT(m) subtree threads independently.
 *
 * @param[in,out] sol Solution object containing problem configuration
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
    FFTZ_INTP in_stride = sol->decomp_scheme->dims[0].in_stride;
    FFTZ_INTP out_stride = sol->decomp_scheme->dims[0].out_stride;
    FFTZ_INT32 status = SOLVER_SUCCESS;

    FFTZ_VOID *bs_in_real  = bs_ctx.in_real;
    FFTZ_VOID *bs_out_real = bs_ctx.out_real;

    // Current solution I/O buffers
    FFTZ_VOID *cur_in = ctx->in_real;
    FFTZ_VOID *cur_out = ctx->out_real;

    //=========================================================================
    // Step 1: Copy input and apply chirp pre-processing
    //=========================================================================
    bluestein_copy_data(cur_in, bs_in_real, n, in_stride, 1, dt_prec, dt_bytes);

    // Zero-pad the input from index n to m-1
    memset(MOVE_ADDR(bs_in_real, n * elem_bytes), 0, (m - n) * elem_bytes);

    // Multiply input by chirp sequence B (or its conjugate)
    mt_ele_mul_dispatch(bluestein->ele_mul[dir], bs_in_real, bs_in_real,
                        bluestein->B, n, n_threads, elem_bytes);

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
    mt_ele_mul_dispatch(bluestein->ele_mul[!dir], bs_out_real, bs_out_real,
                        bluestein->B_out, m, n_threads, elem_bytes);

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
    // Step 3: Post-processing - 1/N scaling and apply final chirp
    // multiplication
    //=========================================================================
    mt_normalize_dispatch(bluestein->normalize, bs_in_real, n, (1.0 / m),
                          n_threads, elem_bytes);

    // Apply final chirp multiplication and copy with stride optimization
    if (out_stride == 1)
    {
        // Optimization: multiply directly to output for unit stride
        mt_ele_mul_dispatch(bluestein->ele_mul[dir], cur_out, bs_in_real,
                            bluestein->B, n, n_threads, elem_bytes);
    }
    else
    {
        // For non-unit stride: multiply in-place then copy with stride
        mt_ele_mul_dispatch(bluestein->ele_mul[dir], bs_in_real, bs_in_real,
                            bluestein->B, n, n_threads, elem_bytes);
        bluestein_copy_data(bs_in_real, cur_out, n, 1, out_stride,
                            dt_prec, dt_bytes);
    }

exit_mt_bluestein_solver:
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return status;
}

dft_solver_ register_execute_mt_bluestein_solver(FFTZ_VOID)
{
    return execute_mt_bluestein_solver;
}
