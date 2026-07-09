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

    // in/out form a pool of num_ct_buf per-thread slots (one per concurrent
    // Bluestein invocation), each padded to MIN_ALIGNMENT (64 B) so every slot
    // base is 64-byte aligned for aligned SIMD load/store in normalize.
    FFTZ_INTP bs_buf_size =
        GET_PADDED_SIZE((FFTZ_INTP)m * DATA_STRIDE * dt_bytes);
    ret = alloc_bluestein_buffers(sol->dft_bufs->bluestein,
                                  bs_buf_size, sol->dft_bufs->num_ct_buf);
    if (ret != AOCLFFTZ_SUCCESS)
    {
        return ret;
    }

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
 * @brief Computes FFT of chirp sequence into the shared B_out buffer.
 *
 * Runs once during plan setup so the result is available (read-only) for
 * every subsequent execute call. This removes the first-execute overhead
 * and lets deep-copied solution trees share B_out safely under MT.
 *
 * @param[in,out] sol      Current solution containing bluestein buffers
 * @param[in,out] next_sol Next solution used for FFT computation
 * @return FFTZ_INT32 SOLVER_SUCCESS on success, SOLVER_FAILURE on error
 */
FFTZ_INT32 compute_chirp_fft(aoclfftz_solution_t *sol,
                             aoclfftz_solution_t *next_sol)
{
    FFTZ_UINT8 dt_prec = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    FFTZ_UINT32 dt_bytes = DT_PRECISION_BYTES(dt_prec);

    // Save next_sol I/O and flag state for restoration after execution
    FFTZ_VOID *in_real = next_sol->decomp_scheme->in_real;
    FFTZ_VOID *in_imag = next_sol->decomp_scheme->in_imag;
    FFTZ_VOID *out_real = next_sol->decomp_scheme->out_real;
    FFTZ_VOID *out_imag = next_sol->decomp_scheme->out_imag;
    FFTZ_VOID *ct_buf_real = next_sol->dft_bufs->ct_buf_real;
    FFTZ_VOID *ct_buf_imag = next_sol->dft_bufs->ct_buf_imag;
    FFTZ_UINT32 initial_flags = next_sol->decomp_scheme->flags;

    // Chirp FFT is always forward. Force it here to prevent AVX kernels
    // from swapping in_real/in_imag pointers under BACKWARD_FFT_DIR.
    SET_FFT_DIR(next_sol->decomp_scheme->flags, FORWARD_FFT_DIR);

    // Set up input/output buffers for chirp sequence FFT
    next_sol->decomp_scheme->in_real = sol->dft_bufs->bluestein->B;
    next_sol->decomp_scheme->in_imag =
        MOVE_ADDR(sol->dft_bufs->bluestein->B, dt_bytes);
    next_sol->decomp_scheme->out_real = sol->dft_bufs->bluestein->B_out;
    next_sol->decomp_scheme->out_imag =
        MOVE_ADDR(sol->dft_bufs->bluestein->B_out, dt_bytes);
    next_sol->dft_bufs->ct_buf_real = next_sol->decomp_scheme->out_real;
    next_sol->dft_bufs->ct_buf_imag = next_sol->decomp_scheme->out_imag;

    // Execute forward FFT on chirp sequence
    FFTZ_INT32 status = next_sol->solver->execute_solver(next_sol);

    // Restore next_sol state so execute_bluestein_solver sees the original
    // pointers set by setup_bluestein_solver
    next_sol->decomp_scheme->in_real = in_real;
    next_sol->decomp_scheme->in_imag = in_imag;
    next_sol->decomp_scheme->out_real = out_real;
    next_sol->decomp_scheme->out_imag = out_imag;
    next_sol->dft_bufs->ct_buf_real = ct_buf_real;
    next_sol->dft_bufs->ct_buf_imag = ct_buf_imag;
    next_sol->decomp_scheme->flags = initial_flags;

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
 * @return FFTZ_INT32 SOLVER_SUCCESS on success, SOLVER_FAILURE on error
 */
static FFTZ_INT32 execute_bluestein_solver(aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    aoclfftz_solution_t *next_sol = sol->next_sol[0];
    FFTZ_UINT8 dt_prec = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    FFTZ_UINT32 dt_bytes = DT_PRECISION_BYTES(dt_prec);
    FFTZ_UINT32 dir = FFT_DIR(sol->decomp_scheme->flags);
    FFTZ_UINT32 initial_flags = next_sol->decomp_scheme->flags;

    // next_sol inherits the requested direction, but the convolution always
    // runs a forward FFT at step 2a and an inverse FFT at step 2c. The kernels
    // pick forward/inverse from this flag, so force forward here for a backward
    // request; step 2c restores backward for the inverse transform.
    if (dir == BACKWARD_FFT_DIR)
    {
        SET_FFT_DIR(next_sol->decomp_scheme->flags, FORWARD_FFT_DIR);
    }

    FFTZ_INTP n = sol->decomp_scheme->dims[0].n;      // Original length
    FFTZ_INTP m = next_sol->decomp_scheme->dims[0].n; // Extended length
    FFTZ_INTP in_stride = sol->decomp_scheme->dims[0].in_stride;
    FFTZ_INTP out_stride = sol->decomp_scheme->dims[0].out_stride;
    FFTZ_INT32 status = SOLVER_SUCCESS;

    // Save original buffer pointers for restoration after execution
    FFTZ_VOID *in_real = next_sol->decomp_scheme->in_real;
    FFTZ_VOID *in_imag = next_sol->decomp_scheme->in_imag;
    FFTZ_VOID *out_real = next_sol->decomp_scheme->out_real;
    FFTZ_VOID *out_imag = next_sol->decomp_scheme->out_imag;
    FFTZ_VOID *ct_buf_real = next_sol->dft_bufs->ct_buf_real;
    FFTZ_VOID *ct_buf_imag = next_sol->dft_bufs->ct_buf_imag;

    // Current solution I/O buffers
    FFTZ_VOID *cur_in = sol->decomp_scheme->in_real;
    FFTZ_VOID *cur_out = sol->decomp_scheme->out_real;

    //=========================================================================
    // Step 1: Copy input and apply chirp pre-processing
    //=========================================================================
    bluestein_copy_data(cur_in, in_real, n, in_stride, 1, dt_prec, dt_bytes);

    // Zero-pad the input from index n to m-1
    memset(MOVE_ADDR(in_real, n * DATA_STRIDE * dt_bytes), 0,
           (m - n) * DATA_STRIDE * dt_bytes);

    // Multiply input by chirp sequence B (or its conjugate)
    sol->dft_bufs->bluestein->ele_mul[dir](in_real, in_real,
                                           sol->dft_bufs->bluestein->B, n);

    //=========================================================================
    // Step 2: Convolution via FFT
    //=========================================================================

    // 2a. Forward FFT of pre-processed input
    next_sol->dft_bufs->ct_buf_real = next_sol->decomp_scheme->out_real;
    next_sol->dft_bufs->ct_buf_imag = next_sol->decomp_scheme->out_imag;
    status = next_sol->solver->execute_solver(next_sol);
    if (status != SOLVER_SUCCESS)
    {
        return SOLVER_FAILURE;
    }

    // FFT of chirp sequence B (B_out) is computed during plan setup
    // 2b. Pointwise multiplication: A_out × B_out (with conjugate for inverse)
    sol->dft_bufs->bluestein->ele_mul[!dir](out_real, out_real,
                                            sol->dft_bufs->bluestein->B_out, m);

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
        return SOLVER_FAILURE;
    }

    //=========================================================================
    // Step 3: Post-processing - 1/N scaling and apply final chirp
    // multiplication
    //=========================================================================
    sol->dft_bufs->bluestein->normalize(in_real, n, (1.0 / m));

    // Apply final chirp multiplication and copy with stride optimization
    if (out_stride == 1)
    {
        // Optimization: multiply directly to output for unit stride
        sol->dft_bufs->bluestein->ele_mul[dir](cur_out, in_real,
                                               sol->dft_bufs->bluestein->B, n);
    }
    else
    {
        // For non-unit stride: multiply in-place then copy with stride
        sol->dft_bufs->bluestein->ele_mul[dir](
            in_real, in_real, sol->dft_bufs->bluestein->B, n);
        bluestein_copy_data(in_real, cur_out, n, 1, out_stride,
                            dt_prec, dt_bytes);
    }

    //=========================================================================
    // Cleanup: Restore original buffer pointers and flags
    //=========================================================================
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

dft_solver_ register_execute_bluestein_solver(FFTZ_VOID)
{
    return execute_bluestein_solver;
}
