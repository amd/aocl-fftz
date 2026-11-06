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
 *  execute the child, store the result. setup_real_bluestein_solver binds
 *  which pair of conversion kernels this node uses.
 *
 *  @author Jeevanantham N
 */

#include "core/common/memory_manager.h"
#include "core/solvers/real/direct_solver_rdft_utils.h"
#include "core/solvers/solver.h"
#include "utils/utils.h"

/**
 * @brief Sets up the real Bluestein node and configures its complex child.
 *
 * Mirrors the real ND setup: the caller allocates an empty child solution and
 * passes it in, this routine copies the parent into it and reshapes it, and
 * the caller then hands the child to the complex selector to build the size-n
 * complex subtree. The copy must precede any scratch claim so the child, whose
 * copy carries the parent's (still NULL) Bluestein buffer pointers, never
 * double-frees them.
 *
 * This node records only the per-thread size of the complex scratch it needs:
 * space for the real->complex expanded input and the complex FFT output, each
 * of size n. The scratch itself is drawn from the Bluestein pool shared across
 * a call; compute_exec_metadata adds this node's slice to bs_buffer_size and
 * records its start in bs_dim_offset, and execute reaches it through
 * ctx->bs_in_base / ctx->bs_out_base. The chirp buffers (B/B_out) belong to
 * the child complex Bluestein node, not here.
 *
 * A CT-stage Bluestein also takes over the strides and buffer roles of the
 * Direct stage it replaces; a standalone prime keeps the plan's own I/O.
 *
 * @param[in,out] sol         Current (real) Bluestein solution object
 * @param[out]    complex_sol Empty child solution to reshape into the size-n
 *                            complex sub-problem
 * @param[in]     kt          Kernel tables for binding the conversion kernels
 * @param[in]     realhelper  Setup-time helper carrying the CT stage state
 * @return FFTZ_INT32 SOLVER_SUCCESS on success, error code on failure
 */
FFTZ_INT32 setup_real_bluestein_solver(aoclfftz_solution_t *sol,
                                       aoclfftz_solution_t *complex_sol,
                                       kernel_tables_t *kt,
                                       aoclfftz_realhelper_t *realhelper)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    FFTZ_INT32 ret = copy_solution_obj(complex_sol, sol);
    if (ret != AOCLFFTZ_SUCCESS)
    {
        return ret;
    }

    // Reshape the child into a complex, 1D, single-batch, unit-stride size-n
    // problem; the real node drives its I/O through a per-call ctx.
    SET_COMPLEX(complex_sol->decomp_scheme->flags);
    // The child always uses distinct in/out scratch; clear the parent's
    // pointers so an in-place parent doesn't alias them.
    SET_OUTOFPLACE(complex_sol->decomp_scheme->flags);
    complex_sol->decomp_scheme->in_real  = NULL;
    complex_sol->decomp_scheme->in_imag  = NULL;
    complex_sol->decomp_scheme->out_real = NULL;
    complex_sol->decomp_scheme->out_imag = NULL;
    // Clear the parent's strides brought in by the copy.
    complex_sol->decomp_scheme->dims[0].in_stride = 1;
    complex_sol->decomp_scheme->dims[0].out_stride = 1;
    complex_sol->decomp_scheme->vecs[0].n = 1;
    complex_sol->decomp_scheme->vecs[0].in_stride = 1;
    complex_sol->decomp_scheme->vecs[0].out_stride = 1;
    complex_sol->next_sol = NULL;

    FFTZ_INTP n = sol->decomp_scheme->dims[0].n;
    FFTZ_UINT32 dt_bytes = SOL_DT_SIZE(sol);

    // One size-n complex slot (interleaved re/im) per bs_in/bs_out, padded to
    // MIN_ALIGNMENT (64 B) to keep each slot aligned for the child's SIMD.
    aoclfftz_bluestein_t *bluestein = sol->dft_bufs->bluestein;
    bluestein->bs_buf_size =
        GET_PADDED_SIZE((FFTZ_INTP)n * DATA_STRIDE * dt_bytes);

    // Bind the node's conversion pair. Only the spectrum side differs: a CT
    // stage packs it (n compact reals), a standalone node uses n/2+1 complex.
    FFTZ_UINT8 is_ct_stage = IS_BLUESTEIN_CT_STAGE(sol->decomp_scheme->flags);
    FFTZ_UINT8 is_forward =
        FFT_DIR(sol->decomp_scheme->flags) == FORWARD_FFT_DIR;

    if (is_forward)
    {
        // Read reals, write the spectrum. A CT head packs it (c2hc_packed);
        // a standalone node writes n/2+1 complex points (c2hc).
        bluestein->cast_to_complex = kt->bs.convert_r2c;
        bluestein->cast_from_complex = (is_ct_stage)
                                           ? kt->bs.convert_c2hc_packed
                                           : kt->bs.convert_c2hc;
    }
    else
    {
        // Read the spectrum, write reals. A CT tail unpacks it (hc2c_packed);
        // a standalone node reads n/2+1 complex points (hc2c).
        bluestein->cast_to_complex = (is_ct_stage)
                                         ? kt->bs.convert_hc2c_packed
                                         : kt->bs.convert_hc2c;
        bluestein->cast_from_complex = kt->bs.convert_c2r;
    }

    // A CT stage takes over the strides and buffer roles of the Direct it
    // replaces; a standalone prime uses the plan's own I/O.
    if (is_ct_stage)
    {
        // Build the replaced Direct's R2HC stride table so the packed
        // converters address the regrouped aux like the neighbour. Order
        // matters: counts, strides, then set_ct_stage_strides rewrites dims.
        set_kernel_count_in_each_group(sol, realhelper);
        ret = allocate_and_setup_stride(sol, *realhelper);
        if (ret != SOLVER_SUCCESS)
        {
            return ret;
        }
        set_ct_stage_strides(sol, realhelper);
        update_ct_buffers(sol, realhelper);
    }
    else
    {
        sol->decomp_scheme->real_in_role = REAL_USE_IO_BUF;
        sol->decomp_scheme->real_out_role = REAL_USE_IO_BUF;
    }

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SOLVER_SUCCESS;
}

/**
 * @brief Executes one real Bluestein transform through its complex child.
 *
 * A Batched parent invokes this single-transform worker once per group. The
 * group index in ctx selects the packed CT bands; IO-backed pointers have
 * already been advanced by the parent, while aux-backed plain sides are
 * offset here. A standalone prime enters once with batch_idx zero.
 *
 * @param[in,out] sol Real Bluestein solution object
 * @param[in,out] ctx Per-call execution context (batch index in ctx->batch_idx)
 * @return FFTZ_INT32 SOLVER_SUCCESS on success, SOLVER_FAILURE on error
 */
static FFTZ_INT32 execute_real_bluestein_solver(aoclfftz_solution_t *sol,
                                                aoclfftz_mutable_ctx_t *ctx)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    // The complex FFT(n) child always lives on complex_sol.
    aoclfftz_solution_t *child_sol = sol->dft_bufs->complex_sol;
    aoclfftz_bluestein_t *bluestein = sol->dft_bufs->bluestein;
    FFTZ_UINT8 dt_prec = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    FFTZ_UINT32 dt_bytes = DT_PRECISION_BYTES(dt_prec);

    FFTZ_INTP n = sol->decomp_scheme->dims[0].n;
    FFTZ_INTP in_stride = sol->decomp_scheme->dims[0].in_stride;
    FFTZ_INTP out_stride = sol->decomp_scheme->dims[0].out_stride;
    FFTZ_INTP b = ctx->batch_idx;

    // A CT stage reads/writes the aux ping-pong pools; a standalone prime uses
    // the handle I/O. The per-node roles resolve either case.
    FFTZ_UINT8 is_ct_stage = IS_BLUESTEIN_CT_STAGE(sol->decomp_scheme->flags);
    FFTZ_UINT8 is_forward =
        FFT_DIR(sol->decomp_scheme->flags) == FORWARD_FFT_DIR;
    FFTZ_VOID *real_in = NULL;
    FFTZ_VOID *real_out = NULL;
    aoclfftz_resolve_real_io(ctx, sol->decomp_scheme->real_in_role,
                             sol->decomp_scheme->real_out_role, &real_in,
                             &real_out);

    // A CT stage's spectrum side uses the packed converter (scatter/gather via
    // the R2HC stride table); the real-signal side keeps the plain strided one.
    aoclfftz_strides_t *r2hc = sol->strides_grp->strides_r2hc;
    FFTZ_UINT8 packed_in = is_ct_stage && !is_forward;
    FFTZ_UINT8 packed_out = is_ct_stage && is_forward;

    // IO-buf sides are already stepped per batch by the driver; aux sides are
    // not, so offset them by the batch index. The packed side uses group b.
    FFTZ_INTP v_in_bytes = sol->decomp_scheme->vecs[0].in_stride * dt_bytes;
    FFTZ_INTP v_out_bytes = sol->decomp_scheme->vecs[0].out_stride * dt_bytes;
    if (!packed_in &&
        sol->decomp_scheme->real_in_role == REAL_USE_AUX_AND_SWAP)
    {
        real_in = MOVE_ADDR(real_in, b * v_in_bytes);
    }
    if (!packed_out &&
        sol->decomp_scheme->real_out_role == REAL_USE_AUX_AND_SWAP)
    {
        real_out = MOVE_ADDR(real_out, b * v_out_bytes);
    }

    // Two-level split of the shared pool: bs_dim_offset selects this node's
    // slice, then bs_buf_size * slot_idx selects this thread's slot in it.
    FFTZ_INTP bs_buf_offset = bluestein->bs_dim_offset +
                              bluestein->bs_buf_size * ctx->slot_idx;

    FFTZ_VOID *in_c = MOVE_ADDR(ctx->bs_in_base, bs_buf_offset);
    FFTZ_VOID *out_c = MOVE_ADDR(ctx->bs_out_base, bs_buf_offset);

    // Point the child's ctx I/O at our complex scratch (interleaved re/im) and
    // adopt its complex plan flags.
    aoclfftz_mutable_ctx_t c2c_child_ctx = *ctx;
    c2c_child_ctx.in_real = in_c;
    c2c_child_ctx.in_imag = MOVE_ADDR(in_c, dt_bytes);
    c2c_child_ctx.out_real = out_c;
    c2c_child_ctx.out_imag = MOVE_ADDR(out_c, dt_bytes);
    c2c_child_ctx.flags = child_sol->decomp_scheme->flags;

    // Each converter reads only the args its layout needs: the plain side the
    // scalar stride, the packed side the R2HC stride table and group index b.
    bluestein->cast_to_complex(in_c, real_in, n, in_stride, r2hc, b);

    FFTZ_INT32 status =
        child_sol->solver->execute_solver(child_sol, &c2c_child_ctx);
    if (status != SOLVER_SUCCESS)
    {
        AOCLFFTZ_LOG(DEBUG, global_logger_mode, "child complex FFT failed");
        return status;
    }

    bluestein->cast_from_complex(real_out, out_c, n, out_stride, r2hc, b);
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SOLVER_SUCCESS;
}

dft_solver_ register_execute_real_bluestein_solver(FFTZ_VOID)
{
    return execute_real_bluestein_solver;
}
