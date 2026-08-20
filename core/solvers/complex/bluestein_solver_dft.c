// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/**
 * @file bluestein_solver_dft.c
 *
 * @brief Bluestein FFT solver for arbitrary-length DFT computation.
 *
 * Implements Bluestein's algorithm, a special case of the chirp z-transform
 * (CZT), to compute the DFT for lengths that are not efficiently factorizable.
 * The algorithm converts the DFT into a convolution, which is computed via
 * FFT of a larger power-of-two (or smooth) size.
 *
 * @author Srirammaswamy Srininvasan
 */

#include "core/common/bluestein_utils.h"
#include "core/common/memory_manager.h"
#include "core/solvers/solver.h"
#include "utils/utils.h"

/**
 * @brief Sets up the Bluestein solver with extended length buffers.
 *
 * Initializes the next solution object with extended length m and allocates
 * the required internal buffers.
 *
 * @param[in,out] sol      Current solution object
 * @param[out]    next_sol Next solution to configure
 * @param[in]     m        Extended length (must be >= 2*n-1 and factorable)
 * @return FFTZ_INT32 SOLVER_SUCCESS on success, error code on failure
 */
FFTZ_INT32 setup_bluestein_solver(aoclfftz_solution_t *sol,
                             aoclfftz_solution_t *next_sol, FFTZ_INTP m)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

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

    // bs_buf_size is the padded byte size of one per-call bs_in/out_base slot;
    // Padded to MIN_ALIGNMENT (64 B) so every slot base is 64-byte aligned for
    // aligned SIMD load/store in normalize.
    FFTZ_INTP bs_buf_size =
        GET_PADDED_SIZE((FFTZ_INTP)m * DATA_STRIDE * dt_bytes);
    ret = alloc_bluestein_buffers(sol->dft_bufs->bluestein, bs_buf_size);
    if (ret != AOCLFFTZ_SUCCESS)
    {
        return ret;
    }

    next_sol->decomp_scheme->in_real = NULL;
    next_sol->decomp_scheme->in_imag = NULL;
    next_sol->decomp_scheme->out_real = NULL;
    next_sol->decomp_scheme->out_imag = NULL;

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SOLVER_SUCCESS;
}

/**
 * @brief Computes FFT of chirp sequence into the shared B_out buffer.
 *
 * Runs once during plan setup so the result is available (read-only) for
 * every subsequent execute call. This removes the first-execute overhead
 * and lets deep-copied solution trees share B_out safely under MT.
 *
 * @param[in,out] sol      Current solution containing bluestein buffers
 * @param[in,out] next_sol Next solution used for FFT computation
 * @param[in]     ctx      Per-call execution context
 * @return FFTZ_INT32 SOLVER_SUCCESS on success, SOLVER_FAILURE on error
 */
FFTZ_INT32 compute_chirp_fft(aoclfftz_solution_t *sol,
                             aoclfftz_solution_t *next_sol,
                             aoclfftz_mutable_ctx_t *ctx)
{
    FFTZ_UINT32 chirp_dt_bytes = CTX_DT_SIZE(ctx);
    aoclfftz_mutable_ctx_t chirp_ctx = *ctx;
    chirp_ctx.in_real     = sol->dft_bufs->bluestein->B;
    chirp_ctx.in_imag     = MOVE_ADDR(sol->dft_bufs->bluestein->B,
                                      chirp_dt_bytes);
    chirp_ctx.out_real    = sol->dft_bufs->bluestein->B_out;
    chirp_ctx.out_imag    = MOVE_ADDR(sol->dft_bufs->bluestein->B_out,
                                      chirp_dt_bytes);
    chirp_ctx.ct_buf_base = chirp_ctx.out_real;

    // Chirp FFT is always forward. Force it here to prevent AVX kernels
    // from swapping in_real/in_imag pointers under BACKWARD_FFT_DIR.
    SET_FFT_DIR(chirp_ctx.flags, FORWARD_FFT_DIR);

    FFTZ_INT32 status = next_sol->solver->execute_solver(next_sol, &chirp_ctx);
    if (status != SOLVER_SUCCESS)
    {
        return SOLVER_FAILURE;
    }

    return SOLVER_SUCCESS;
}

/**
 * @brief Executes the Bluestein FFT algorithm.
 *
 * Algorithm Overview:
 * 1. Multiply input by chirp sequence B_inv (pre-processing)
 * 2. Zero-pad the multiplied input to extended length m
 * 3. Perform forward FFT on the padded sequence
 * 4. Multiply with computed FFT of chirp sequence B
 * 5. Perform inverse FFT
 * 6. Normalize and multiply by B_inv (post-processing)
 *
 * @param[in,out] sol Solution object containing problem configuration
 * @param[in,out] ctx Per-call execution context
 * @return FFTZ_INT32 SOLVER_SUCCESS on success, SOLVER_FAILURE on error
 */
static FFTZ_INT32 execute_bluestein_solver(aoclfftz_solution_t *sol,
                                           aoclfftz_mutable_ctx_t *ctx)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    aoclfftz_solution_t *next_sol = sol->next_sol[0];
    FFTZ_UINT8 dt_prec = DT_PRECISION_FLAG(ctx->flags);
    FFTZ_UINT32 dt_bytes = DT_PRECISION_BYTES(dt_prec);
    FFTZ_UINT32 dir = FFT_DIR(ctx->flags);
    aoclfftz_bluestein_t *bluestein = sol->dft_bufs->bluestein;
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
    memset(MOVE_ADDR(bs_in_real, n * DATA_STRIDE * dt_bytes), 0,
           (m - n) * DATA_STRIDE * dt_bytes);

    // Multiply input by chirp sequence B (or its conjugate)
    bluestein->ele_mul[dir](bs_in_real, bs_in_real, bluestein->B, n);

    //=========================================================================
    // Step 2: Convolution via FFT
    //=========================================================================

    // 2a. Forward FFT of pre-processed input. ct_buf_base is this
    // thread's private out slot.
    bs_ctx.ct_buf_base = bs_ctx.out_real;

    status = next_sol->solver->execute_solver(next_sol, &bs_ctx);
    if (status != SOLVER_SUCCESS)
    {
        return SOLVER_FAILURE;
    }

    // FFT of chirp sequence B (B_out) is computed during plan setup
    // 2b. Pointwise multiplication: A_out × B_out (with conjugate for inverse)
    bluestein->ele_mul[!dir](bs_out_real, bs_out_real, bluestein->B_out, m);

    // 2c. Inverse FFT of the product
    bs_ctx.in_real     = bs_out_real;
    bs_ctx.in_imag     = MOVE_ADDR(bs_out_real, dt_bytes);
    bs_ctx.out_real    = bs_in_real;
    bs_ctx.out_imag    = MOVE_ADDR(bs_in_real, dt_bytes);
    bs_ctx.ct_buf_base = bs_ctx.out_real;
    SET_FFT_DIR(bs_ctx.flags, BACKWARD_FFT_DIR);

    status = next_sol->solver->execute_solver(next_sol, &bs_ctx);
    if (status != SOLVER_SUCCESS)
    {
        return SOLVER_FAILURE;
    }

    //=========================================================================
    // Step 3: Post-processing - 1/N scaling and apply final chirp
    // multiplication
    //=========================================================================
    bluestein->normalize(bs_in_real, n, (1.0 / m));

    // Apply final chirp multiplication and copy with stride optimization
    if (out_stride == 1)
    {
        // Optimization: multiply directly to output for unit stride
        bluestein->ele_mul[dir](cur_out, bs_in_real, bluestein->B, n);
    }
    else
    {
        // For non-unit stride: multiply in-place then copy with stride
        bluestein->ele_mul[dir](bs_in_real, bs_in_real, bluestein->B, n);
        bluestein_copy_data(bs_in_real, cur_out, n, 1, out_stride,
                            dt_prec, dt_bytes);
    }

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return status;
}

dft_solver_ register_execute_bluestein_solver(FFTZ_VOID)
{
    return execute_bluestein_solver;
}
