// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file bluestein_solver_rdft.c
 *
 *  @brief Real (R2C/C2R) Bluestein FFT solver for arbitrary prime lengths.
 *
 *  Real prime-length transforms are computed by reusing the existing complex
 *  Bluestein solver:
 *    - R2C (forward): the n real inputs are expanded to n complex values
 *      (imaginary part = 0), a size-n complex forward FFT is computed via the
 *      child complex Bluestein solver, and only the first n/2+1 spectral
 *      points are stored (the rest follow from Hermitian symmetry).
 *    - C2R (backward): the n/2+1 half-complex inputs are expanded to a full
 *      size-n complex spectrum using Hermitian symmetry, a size-n complex
 *      backward FFT is computed, and the n real parts are stored.
 *
 *  The chirp pre/post multiply, the extended-length convolution FFT and the
 *  normalization are all performed by the child complex Bluestein node. Both
 *  directions therefore reduce to the same three steps: expand the input,
 *  execute the child, store the result. selector_bluestein_rdft determines
 *  which pair of conversion kernels this node uses.
 *
 *  @author Jeevanantham N
 */

#include "core/common/memory_manager.h"
#include "core/solvers/solver.h"
#include "utils/utils.h"

/**
 * @brief Sets up the real Bluestein solver.
 *
 * Records the per-thread size of the complex scratch required here: space for
 * the real->complex expanded input and for the complex FFT output, each of
 * size n.
 *
 * The scratch itself is drawn from the Bluestein pool shared across a call.
 * compute_exec_metadata adds this node's slice to bs_buffer_size and records
 * its start in bs_dim_offset; execute then reaches it through ctx->bs_in_base
 * and ctx->bs_out_base. This node requires no chirp buffers (B/B_out) of its
 * own, as the child complex Bluestein node owns those. The child solution is
 * configured by selector_bluestein_rdft.
 *
 * @param[in,out] sol Current (real) solution object
 * @param[in]     n   Original real transform length
 * @return FFTZ_INT32 SOLVER_SUCCESS on success, error code on failure
 */
FFTZ_INT32 setup_real_bluestein_solver(aoclfftz_solution_t *sol, FFTZ_INTP n)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    FFTZ_UINT32 dt_bytes = SOL_DT_SIZE(sol);

    // One size-n complex slot (interleaved re/im) in each of bs_in / bs_out,
    // padded to MIN_ALIGNMENT (64 B) so every slot base stays 64-byte aligned
    // for the child complex solver's aligned SIMD load/stores.
    sol->dft_bufs->bluestein->bs_buf_size =
        GET_PADDED_SIZE((FFTZ_INTP)n * DATA_STRIDE * dt_bytes);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SOLVER_SUCCESS;
}

/**
 * @brief Executes the real Bluestein FFT by delegating to the complex solver.
 *
 * The real input and output come from ctx, as in the rest of the real solver
 * family, so the solution tree stays read-only. The size-n complex scratch is
 * drawn from the per-call Bluestein pool reached through ctx, so that
 * concurrent callers sharing a handle never occupy the same slot. The child
 * complex Bluestein node executes on a copy of ctx whose in/out point at that
 * scratch, and retains the pool bases and slot index it needs for its own
 * extended-length-m scratch.
 *
 * @param[in,out] sol Real Bluestein solution object
 * @param[in,out] ctx Per-call execution context
 * @return FFTZ_INT32 SOLVER_SUCCESS on success, SOLVER_FAILURE on error
 */
static FFTZ_INT32 execute_real_bluestein_solver(aoclfftz_solution_t *sol,
                                                aoclfftz_mutable_ctx_t *ctx)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    aoclfftz_solution_t *next_sol = sol->next_sol; // complex Bluestein(n)
    aoclfftz_bluestein_t *bluestein = sol->dft_bufs->bluestein;
    FFTZ_UINT8 dt_prec = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    FFTZ_UINT32 dt_bytes = DT_PRECISION_BYTES(dt_prec);

    FFTZ_INTP n = sol->decomp_scheme->dims[0].n;
    FFTZ_INTP in_stride = sol->decomp_scheme->dims[0].in_stride;
    FFTZ_INTP out_stride = sol->decomp_scheme->dims[0].out_stride;

    FFTZ_VOID *real_in = ctx->in_real;
    FFTZ_VOID *real_out = ctx->out_real;

    // Two-level split of the shared pool: bs_dim_offset selects this node's
    // slice, then bs_buf_size * slot_idx selects this thread's slot in it.
    FFTZ_INTP bs_buf_offset = bluestein->bs_dim_offset +
                              bluestein->bs_buf_size * ctx->slot_idx;

    FFTZ_VOID *in_c = MOVE_ADDR(ctx->bs_in_base, bs_buf_offset);
    FFTZ_VOID *out_c = MOVE_ADDR(ctx->bs_out_base, bs_buf_offset);

    // The child complex Bluestein solver reads its input from ctx->in_real and
    // writes its result to ctx->out_real; point both at our complex scratch
    // (interleaved re/im) and adopt the child's own (complex) plan flags.
    aoclfftz_mutable_ctx_t c2c_child_ctx = *ctx;
    c2c_child_ctx.in_real = in_c;
    c2c_child_ctx.in_imag = MOVE_ADDR(in_c, dt_bytes);
    c2c_child_ctx.out_real = out_c;
    c2c_child_ctx.out_imag = MOVE_ADDR(out_c, dt_bytes);
    c2c_child_ctx.flags = next_sol->decomp_scheme->flags;

    // The forward pair expands n reals to n complex and retains n/2+1 of the
    // results; the backward pair reconstructs n complex from n/2+1 points and
    // retains the n real parts. The selector has already bound the pair here.
    bluestein->cast_to_complex(in_c, real_in, n, in_stride);

    FFTZ_INT32 status = next_sol->solver->execute_solver(next_sol,
                                                         &c2c_child_ctx);
    if (status == SOLVER_SUCCESS)
    {
        bluestein->cast_from_complex(real_out, out_c, n, out_stride);
    }

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return status;
}

dft_solver_ register_execute_real_bluestein_solver(FFTZ_VOID)
{
    return execute_real_bluestein_solver;
}
