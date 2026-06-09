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
 * @return INT32 SOLVER_SUCCESS on success, error code on failure
 */
INT32 setup_bluestein_solver(aoclfftz_solution_t *sol,
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

    // Allocate internal buffers for Bluestein computation
    ret = alloc_bluestein_buffers(sol->dft_bufs->bluestein,
                                  m * DATA_STRIDE * dt_bytes);
    if (ret != AOCLFFTZ_SUCCESS)
    {
        return ret;
    }

    // Map internal buffers to next solution's I/O pointers
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
 * @brief Copies complex data between buffers with stride handling.
 *
 * Dispatches to permuted_copy when either source or destination stride is
 * non-unit, otherwise performs a direct memcpy.
 *
 * @param[out] dst        Destination buffer
 * @param[in]  src        Source buffer
 * @param[in]  n          Number of complex elements to copy
 * @param[in]  src_stride Source stride value
 * @param[in]  dst_stride Destination stride value
 * @param[in]  dt_prec    Data precision (DT_FLOAT or DT_DOUBLE)
 * @param[in]  dt_bytes   Size of data type in bytes
 */
static inline VOID copy_data(VOID *dst, VOID *src, INTP n, INTP src_stride,
                             INTP dst_stride, UINT8 dt_prec, UINT32 dt_bytes)
{
    if (src_stride > 1 || dst_stride > 1)
    {
        INTP scaled_src_stride = src_stride * DATA_STRIDE;
        INTP scaled_dst_stride = dst_stride * DATA_STRIDE;
        if (dt_prec == DT_FLOAT)
        {
            permuted_copy_c_fp32(src, dst, 1, n, scaled_src_stride,
                                 scaled_dst_stride, 1, 1);
        }
        else
        {
            permuted_copy_c_fp64(src, dst, 1, n, scaled_src_stride,
                                 scaled_dst_stride, 1, 1);
        }
    }
    else
    {
        memcpy(dst, src, n * DATA_STRIDE * dt_bytes);
    }
}

/**
 * @brief Computes FFT of chirp sequence.
 *
 * @param[in,out] sol      Current solution containing bluestein buffers
 * @param[in,out] next_sol Next solution used for FFT computation
 * @param[in]     dt_bytes Size of data type in bytes
 * @return INT32 SOLVER_SUCCESS on success, SOLVER_FAILURE on error
 */
static inline INT32 compute_chirp_fft(aoclfftz_solution_t *sol,
                                      aoclfftz_solution_t *next_sol,
                                      UINT32 dt_bytes)
{
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
    INT32 status = next_sol->solver->execute_solver(next_sol);
    if (status != SOLVER_SUCCESS)
    {
        return SOLVER_FAILURE;
    }

    // Mark as valid for future executions
    sol->dft_bufs->bluestein->is_chirp_fft_computed = 1;

    return SOLVER_SUCCESS;
}

/**
 * @brief Executes the Bluestein FFT algorithm.
 *
 * Algorithm Overview:
 * 1. Multiply input by chirp sequence B_inv (pre-processing)
 * 2. Zero-pad the multiplied input to extended length m
 * 3. Perform forward FFT on the padded sequence
 * 4. Multiply with pre-computed FFT of chirp sequence B
 * 5. Perform inverse FFT
 * 6. Normalize and multiply by B_inv (post-processing)
 *
 * @param[in,out] sol Solution object containing problem configuration
 * @return INT32 SOLVER_SUCCESS on success, SOLVER_FAILURE on error
 */
static INT32 execute_bluestein_solver(aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    aoclfftz_solution_t *next_sol = sol->next_sol[0];
    UINT8 dt_prec = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    UINT32 dt_bytes = DT_PRECISION_BYTES(dt_prec);
    UINT32 dir = FFT_DIR(sol->decomp_scheme->flags);
    UINT32 initial_flags = next_sol->decomp_scheme->flags;

    // Direction flag handling for AVX kernels:
    // In backward FFT, AVX kernels expect swapped in_real/in_imag pointers.
    // Since Bluestein doesn't swap these pointers, we temporarily set the
    // direction to forward to prevent incorrect re-swapping inside the kernel.
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
    copy_data(in_real, cur_in, n, in_stride, 1, dt_prec, dt_bytes);

    // Zero-pad the input from index n to m-1
    memset(MOVE_ADDR(in_real, n * DATA_STRIDE * dt_bytes), 0,
           (m - n) * DATA_STRIDE * dt_bytes);

    // Multiply input by chirp sequence B (or its conjugate)
    sol->dft_bufs->bluestein->ele_mul[dir](
        in_real, in_real, sol->dft_bufs->bluestein->B, n);

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

    // 2b. Compute FFT of chirp sequence B on the first execute; reused on
    //     subsequent executes since B depends only on the problem size.
    if (!sol->dft_bufs->bluestein->is_chirp_fft_computed)
    {
        status = compute_chirp_fft(sol, next_sol, dt_bytes);
        if (status != SOLVER_SUCCESS)
        {
            return SOLVER_FAILURE;
        }
    }

    // 2c. Pointwise multiplication: A_out × B_out (with conjugate for inverse)
    sol->dft_bufs->bluestein->ele_mul[!dir](
        out_real, out_real, sol->dft_bufs->bluestein->B_out, m);

    // 2d. Inverse FFT of the product
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
        sol->dft_bufs->bluestein->ele_mul[dir](
            cur_out, in_real, sol->dft_bufs->bluestein->B, n);
    }
    else
    {
        // For non-unit stride: multiply in-place then copy with stride
        sol->dft_bufs->bluestein->ele_mul[dir](
            in_real, in_real, sol->dft_bufs->bluestein->B, n);
        copy_data(cur_out, in_real, n, 1, out_stride, dt_prec, dt_bytes);
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

/**
 * @brief Registers the Bluestein solver execution function.
 *
 * @return Function pointer to execute_bluestein_solver
 */
dft_solver_ register_execute_bluestein_solver(VOID)
{
    return execute_bluestein_solver;
}
