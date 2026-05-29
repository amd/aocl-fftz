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
static inline VOID chunk_range(INT32 tid, INTP blocks_per_thread,
                               INTP extra_blocks, INTP align, INTP num_items,
                               INTP *start, INTP *size)
{
    // Block index range owned by this thread.
    INTP block_begin =
        tid * blocks_per_thread + (tid < extra_blocks ? tid : extra_blocks);
    INTP block_end = (tid + 1) * blocks_per_thread
                     + (tid + 1 < extra_blocks ? tid + 1 : extra_blocks);

    // Convert block range to item range.
    INTP elem_begin = block_begin * align;
    INTP elem_end = block_end * align;

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
static inline VOID mt_ele_mul_dispatch(elementwise_mul_ kernel,
                                       VOID *out, VOID *a, VOID *b, INTP n,
                                       INT32 n_threads, INTP elem_bytes)
{
    INTP elems_per_thread = n / n_threads;
    INTP extra_elems = n % n_threads;
    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (INT32 tid = 0; tid < n_threads; tid++)
    {
        INTP start_elem, num_elems;
        chunk_range(tid, elems_per_thread, extra_elems, 1, n, &start_elem,
                    &num_elems);
        if (num_elems <= 0)
        {
            continue;
        }
        INTP thread_offset = start_elem * elem_bytes;
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
static inline VOID mt_normalize_dispatch(normalize_ kernel, VOID *data,
                                         INTP n, DOUBLE factor,
                                         INT32 n_threads, INTP elem_bytes)
{
    // 64-byte-aligned block split so the kernel can use aligned load/store.
    INTP align = MIN_ALIGNMENT / elem_bytes;
    INTP num_blocks = (n + align - 1) / align;
    INTP blocks_per_thread = num_blocks / n_threads;
    INTP extra_blocks = num_blocks % n_threads;
    #pragma omp parallel for num_threads(n_threads) schedule(static)
    for (INT32 tid = 0; tid < n_threads; tid++)
    {
        INTP start_elem, num_elems;
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
 * @return INT32 SOLVER_SUCCESS on success, error code on failure
 */
INT32 setup_mt_bluestein_solver(aoclfftz_solution_t *sol,
                                aoclfftz_solution_t *next_sol, INTP m)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");


    // Setup next_sol with extended length m
    INT32 ret = copy_solution_obj(next_sol, sol);
    if (ret != AOCLFFTZ_SUCCESS)
    {
        AOCLFFTZ_ERROR("copy_solution_obj failed: %s", get_status_string(ret));
        return ret;
    }
    next_sol->decomp_scheme->dims[0].n = m;
    next_sol->decomp_scheme->dims[0].in_stride = 1;
    next_sol->decomp_scheme->dims[0].out_stride = 1;

    UINT32 dt_bytes = SOL_DT_SIZE(sol);

    // in/out form a pool of num_ct_buf per-thread slots (one per concurrent
    // Bluestein invocation), each padded to MIN_ALIGNMENT (64 B) so every slot
    // base is 64-byte aligned for aligned SIMD load/store in normalize.
    INTP bs_buf_size = GET_PADDED_SIZE((INTP)m * DATA_STRIDE * dt_bytes);
    ret = alloc_bluestein_buffers(sol->dft_bufs->bluestein,
                                  bs_buf_size, sol->dft_bufs->num_ct_buf);
    if (ret != AOCLFFTZ_SUCCESS)
    {
        return ret;
    }

    // Snapshot avl_threads as the per-invocation kernel thread budget.
    sol->decomp_scheme->thread_info->n_threads =
        sol->decomp_scheme->thread_info->avl_threads;

    // Map slot 0 of the in/out pool to next_sol's I/O pointers.
    // deep_copy_solution_tree re-points these to slot t for thread t.
    next_sol->decomp_scheme->in_real = sol->dft_bufs->bluestein->in;
    next_sol->decomp_scheme->in_imag =
        MOVE_ADDR(sol->dft_bufs->bluestein->in, dt_bytes);
    next_sol->decomp_scheme->out_real = sol->dft_bufs->bluestein->out;
    next_sol->decomp_scheme->out_imag =
        MOVE_ADDR(sol->dft_bufs->bluestein->out, dt_bytes);

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
 * @return INT32 SOLVER_SUCCESS on success, SOLVER_FAILURE on error
 */
static INT32 execute_mt_bluestein_solver(aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");


    aoclfftz_solution_t *next_sol = sol->next_sol[0];
    aoclfftz_bluestein_t *bluestein = sol->dft_bufs->bluestein;
    UINT8 dt_prec = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    UINT32 dt_bytes = DT_PRECISION_BYTES(dt_prec);
    UINT32 dir = FFT_DIR(sol->decomp_scheme->flags);
    UINT32 initial_flags = next_sol->decomp_scheme->flags;
    INT32 n_threads = sol->decomp_scheme->thread_info->n_threads;
    INTP elem_bytes = (INTP)DATA_STRIDE * dt_bytes;

    // next_sol inherits the requested direction, but the convolution always
    // runs a forward FFT at step 2a and an inverse FFT at step 2c. The kernels
    // pick forward/inverse from this flag, so force forward here for a backward
    // request; step 2c restores backward for the inverse transform.
    if (dir == BACKWARD_FFT_DIR)
    {
        SET_FFT_DIR(next_sol->decomp_scheme->flags, FORWARD_FFT_DIR);
    }

    INTP n = sol->decomp_scheme->dims[0].n;      // Original length
    INTP m = next_sol->decomp_scheme->dims[0].n; // Extended length
    INTP in_stride = sol->decomp_scheme->dims[0].in_stride;
    INTP out_stride = sol->decomp_scheme->dims[0].out_stride;
    INT32 status = SOLVER_SUCCESS;

    // Save original buffer pointers for restoration after execution
    VOID *in_real = next_sol->decomp_scheme->in_real;
    VOID *in_imag = next_sol->decomp_scheme->in_imag;
    VOID *out_real = next_sol->decomp_scheme->out_real;
    VOID *out_imag = next_sol->decomp_scheme->out_imag;
    VOID *ct_buf_real = next_sol->dft_bufs->ct_buf_real;
    VOID *ct_buf_imag = next_sol->dft_bufs->ct_buf_imag;

    // Current solution I/O buffers
    VOID *cur_in = sol->decomp_scheme->in_real;
    VOID *cur_out = sol->decomp_scheme->out_real;

    //=========================================================================
    // Step 1: Copy input and apply chirp pre-processing
    //=========================================================================
    bluestein_copy_data(cur_in, in_real, n, in_stride, 1, dt_prec, dt_bytes);

    // Zero-pad the input from index n to m-1
    memset(MOVE_ADDR(in_real, n * elem_bytes), 0, (m - n) * elem_bytes);

    // Multiply input by chirp sequence B (or its conjugate)
    mt_ele_mul_dispatch(bluestein->ele_mul[dir], in_real, in_real, bluestein->B,
                        n, n_threads, elem_bytes);

    //=========================================================================
    // Step 2: Convolution via FFT
    //=========================================================================

    // 2a. Forward FFT of pre-processed input.
    //     FFT of chirp sequence B is pre-computed during plan setup
    next_sol->dft_bufs->ct_buf_real = next_sol->decomp_scheme->out_real;
    next_sol->dft_bufs->ct_buf_imag = next_sol->decomp_scheme->out_imag;
    status = next_sol->solver->execute_solver(next_sol);
    if (status != SOLVER_SUCCESS)
    {
        goto exit_mt_bluestein_solver;
    }

    // 2b. Pointwise multiplication: A_out × B_out (with conjugate for inverse)
    mt_ele_mul_dispatch(bluestein->ele_mul[!dir], out_real, out_real,
                        bluestein->B_out, m, n_threads, elem_bytes);

    // 2c. Inverse FFT of the product
    next_sol->decomp_scheme->in_real = out_real;
    next_sol->decomp_scheme->in_imag = out_imag;
    next_sol->decomp_scheme->out_real = in_real;
    next_sol->decomp_scheme->out_imag = in_imag;
    next_sol->dft_bufs->ct_buf_real = next_sol->decomp_scheme->out_real;
    next_sol->dft_bufs->ct_buf_imag = next_sol->decomp_scheme->out_imag;
    SET_FFT_DIR(next_sol->decomp_scheme->flags, BACKWARD_FFT_DIR);

    status = next_sol->solver->execute_solver(next_sol);
    if (status != SOLVER_SUCCESS)
    {
        goto exit_mt_bluestein_solver;
    }

    //=========================================================================
    // Step 3: Post-processing - 1/N scaling and apply final chirp
    // multiplication
    //=========================================================================
    mt_normalize_dispatch(bluestein->normalize, in_real, n, (1.0 / m),
                          n_threads, elem_bytes);

    // Apply final chirp multiplication and copy with stride optimization
    if (out_stride == 1)
    {
        // Optimization: multiply directly to output for unit stride
        mt_ele_mul_dispatch(bluestein->ele_mul[dir], cur_out, in_real,
                            bluestein->B, n, n_threads, elem_bytes);
    }
    else
    {
        // For non-unit stride: multiply in-place then copy with stride
        mt_ele_mul_dispatch(bluestein->ele_mul[dir], in_real, in_real,
                            bluestein->B, n, n_threads, elem_bytes);
        bluestein_copy_data(in_real, cur_out, n, 1, out_stride,
                            dt_prec, dt_bytes);
    }

    //=========================================================================
    // Cleanup: Restore original buffer pointers and flags. Reached on both the
    // success path and any error goto, so a failed execute never leaves
    // next_sol re-wired for a subsequent invocation.
    //=========================================================================
exit_mt_bluestein_solver:
    next_sol->decomp_scheme->in_real = in_real;
    next_sol->decomp_scheme->in_imag = in_imag;
    next_sol->decomp_scheme->out_real = out_real;
    next_sol->decomp_scheme->out_imag = out_imag;
    next_sol->dft_bufs->ct_buf_real = ct_buf_real;
    next_sol->dft_bufs->ct_buf_imag = ct_buf_imag;
    next_sol->decomp_scheme->flags = initial_flags;

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return status;
}

dft_solver_ register_execute_mt_bluestein_solver(VOID)
{
    return execute_mt_bluestein_solver;
}
