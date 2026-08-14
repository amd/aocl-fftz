// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file pow2_fourstep_solver_dft.c
 *
 *  @brief Power-of-2 four-step solver (SOLVER_POW2_FOURSTEP).
 *
 *  N = n1 * n2 runs as two batches of short, cache-resident sub-FFTs separated
 *  by a fused twiddle + transpose pass. Plan layout: aoclfftz_pow2_fourstep_t
 *  in api/aoclfftz_internal.h; rationale: pow2 fast-path doc, Regime 2.
 *
 *  Pipeline (input read as an n1 x n2 row-major matrix A[i][j] = x[i*n2 + j]):
 *    sub1 : n1-point FFT down each of n2 contiguous columns, written already
 *           transposed, so no separate transpose is needed;
 *    fused: twiddle multiply + transpose (n2 x n1) in a single pass;
 *    sub2 : n2-point FFT down each of n1 contiguous columns, output in natural
 *           order in the caller buffer.
 *
 *  SIMD: the points of one column FFT are row-stride separated while adjacent
 *  columns are row-contiguous, so fourstep_batched_colfft batches columns as
 *  the kernels' SIMD dimension.
 *
 *  Sizes: the gate (is_pow2_solvable) takes N past the iterative solver's 2x
 *  L1D budget while n1 still fits L2. Setup declines (SOLVER_FAILURE, caller falls
 *  through to CT) when no fused kernel exists for the configuration or n1/n2
 *  are not micro-tile multiples: fused-only, with no standalone twiddle or
 *  transpose path.
 *
 *  Memory above the caller buffers, in complex elements:
 *    ping-pong pool : 2 * slots * max(n1 * sub1_row_stride,
 *                                     n2 * sub2_row_stride)
 *    step twiddles  : n1 * n2
 *    sub-FFT tables : ~N/n1 + ~N/n2
 *  slots is active_threads, which the gate's single-thread requirement pins to
 *  1 (a concurrent execute_io gets its own copy of the pool, not a slot), and a
 *  row stride is its dense width (n2 for sub1, n1 for sub2) padded to an odd
 *  cache-line count, so each buffer is barely over N and the total is ~3N.
 *
 *  @author Ashwin K. Godbole
 */

#include "core/solvers/solver.h"
#include "core/common/memory_manager.h"
#include "core/common/twiddle.h"
#include "core/common/pow2_radix_decompose.h"
#include "core/kernels/non_dft/fused_twiddle_transpose/fused_twiddle_transpose.h"
#include "utils/cpu_features.h"
#include <math.h>

// Factor sub_n into radix stages, filling radixes[] and kffts[] (stage s at
// kffts[s * NUM_FFT_DIRS + dir]). Returns the stage count, or 0 on failure.
static FFTZ_INT32 fourstep_decompose_subfft(FFTZ_INTP sub_n, kernel_t *kt_dft,
                                            kernel_t *kt_twid,
                                            FFTZ_UINT8 precision,
                                            FFTZ_UINT8 direction,
                                            FFTZ_INTP batch,
                                            FFTZ_INTP *radixes, kfft_ *kffts)
{
    FFTZ_INT32 num_stages = pow2_radix_decompose(sub_n, kt_dft, kt_twid,
                                                 radixes);
    if (num_stages <= 0)
    {
        return 0;
    }

    // Stage 0 selects from the c2c kernel table, the rest from the twiddle one.
    for (FFTZ_INT32 stage = 0; stage < num_stages; stage++)
    {
        kernel_t *kt = (stage == 0) ? kt_dft : kt_twid;
        if (!pow2_radix_pick_kernel(kt, radixes[stage], precision, direction,
                                    batch, &kffts[stage * NUM_FFT_DIRS], NULL,
                                    NULL))
        {
            return 0;
        }
    }
    return num_stages;
}

// Padded row stride (in complex elements) for a `row_width` wide row.
// Returns the dense width rounded up to an odd number of cache lines.
static FFTZ_INTP fourstep_padded_row_stride(FFTZ_INTP row_width,
                                            FFTZ_UINT32 dt_bytes,
                                            FFTZ_INTP cache_line_size)
{
    FFTZ_INTP elem_size = (FFTZ_INTP)DATA_STRIDE * (FFTZ_INTP)dt_bytes;
    FFTZ_INTP row_bytes = row_width * elem_size;

    FFTZ_INTP cache_lines = (row_bytes + cache_line_size - 1) / cache_line_size;
    cache_lines = cache_lines < 1 ? 1 : cache_lines;

    FFTZ_INTP elems_per_line = cache_line_size / elem_size;
    elems_per_line = elems_per_line < 1 ? 1 : elems_per_line;

    cache_lines = (cache_lines & 1) ? cache_lines : cache_lines + 1;

    return cache_lines * elems_per_line;
}

// Store the twiddle pair at `idx` in the plan's precision.
static inline void fourstep_store_tw_pair(FFTZ_VOID *tw, FFTZ_INTP idx,
                                          FFTZ_DOUBLE re, FFTZ_DOUBLE im,
                                          FFTZ_UINT8 precision)
{
    if (precision == DT_FLOAT)
    {
        FFTZ_FLOAT *tw_fp32 = (FFTZ_FLOAT *)tw;
        tw_fp32[idx * 2] = (FFTZ_FLOAT)re;
        tw_fp32[idx * 2 + 1] = (FFTZ_FLOAT)im;
    }
    else
    {
        FFTZ_DOUBLE *tw_fp64 = (FFTZ_DOUBLE *)tw;
        tw_fp64[idx * 2] = re;
        tw_fp64[idx * 2 + 1] = im;
    }
}

// Fill the inter-step twiddles in the blocked order the fused kernels consume:
// one pointer walking dense tile_size x tile_size tiles (see
// fused_twiddle_transpose.h). The phase is formed in double for both
// precisions, since col * row reaches n_total, past float's 24-bit integer
// range.
static void fourstep_fill_blocked_step_twiddles(FFTZ_VOID *twiddles,
                                                FFTZ_INTP n1, FFTZ_INTP n2,
                                                FFTZ_INTP block_size,
                                                FFTZ_INTP tile_size,
                                                FFTZ_UINT8 precision)
{
    FFTZ_INTP n_total = n1 * n2;
    FFTZ_INTP out_idx = 0;
    // Blocks step column-first so the transposed output is written in row
    // order; inside a block, tiles and elements step row-first to read the
    // source in order.
    for (FFTZ_INTP bcol = 0; bcol < n2; bcol += block_size)
    {
        FFTZ_INTP bcol_end = (bcol + block_size < n2) ? bcol + block_size : n2;
        for (FFTZ_INTP brow = 0; brow < n1; brow += block_size)
        {
            FFTZ_INTP brow_end =
                (brow + block_size < n1) ? brow + block_size : n1;
            for (FFTZ_INTP trow = brow; trow < brow_end; trow += tile_size)
            {
                for (FFTZ_INTP tcol = bcol; tcol < bcol_end; tcol += tile_size)
                {
                    for (FFTZ_INTP row = trow; row < trow + tile_size; row++)
                    {
                        for (FFTZ_INTP col = tcol; col < tcol + tile_size;
                             col++)
                        {
                            FFTZ_DOUBLE re = 1.0;
                            FFTZ_DOUBLE im = 0.0;
                            if (row != 0 && col != 0)
                            {
                                FFTZ_DOUBLE angle = -AOCLFFTZ_2_PI
                                                    * (FFTZ_DOUBLE)col
                                                    * (FFTZ_DOUBLE)row
                                                    / (FFTZ_DOUBLE)n_total;
                                re = cos(angle);
                                im = sin(angle);
                            }
                            fourstep_store_tw_pair(twiddles, out_idx, re, im,
                                                   precision);
                            out_idx++;
                        }
                    }
                }
            }
        }
    }
}

// Inter-stage twiddle pairs a sub-FFT needs: sum over non-leaf stages of
// radix * sub_len (sub_len = product of earlier radixes).
static FFTZ_INTP fourstep_subfft_tw_pairs(const FFTZ_INTP *radixes,
                                          FFTZ_INT32 num_stages)
{
    FFTZ_INTP total = 0;
    FFTZ_INTP sub_len = radixes[0];
    for (FFTZ_INT32 stage = 1; stage < num_stages; stage++)
    {
        total += radixes[stage] * sub_len;
        sub_len *= radixes[stage];
    }
    return total;
}

// Precompute one stage's kernel strides (replayed verbatim at execute time).
// `sub_len` is the running sub-DFT length before this stage.
static FFTZ_INT32 fourstep_fill_stage_strides(
    aoclfftz_pow2_stage_t *st, FFTZ_INT32 stage, FFTZ_INT32 num_stages,
    FFTZ_INTP fft_len, FFTZ_INTP row_stride, FFTZ_INTP sub_len,
    FFTZ_INTP leaf_in_row_stride, FFTZ_INTP dst_elem_stride,
    FFTZ_INTP dst_row_stride)
{
    FFTZ_INTP radix = st->radix;
    st->strides.in_strides = NULL;
    st->strides.out_strides = NULL;
    ALLOC_ALIGN_UNINIT(st->strides.in_strides, FFTZ_INTP,
                       radix * sizeof(FFTZ_INTP));
    ALLOC_ALIGN_UNINIT(st->strides.out_strides, FFTZ_INTP,
                       radix * sizeof(FFTZ_INTP));
    if (st->strides.in_strides == NULL || st->strides.out_strides == NULL)
    {
        return SOLVER_FAILURE;
    }

    if (stage == 0)
    {
        // Leaf stage: input matrix -> first scratch (or caller buffer if the
        // only stage). In-stride uses the source row stride.
        FFTZ_INTP out_elem_stride = (num_stages == 1) ? dst_elem_stride : 1;
        FFTZ_INTP out_row_stride =
            (num_stages == 1) ? dst_row_stride : row_stride;
        FFTZ_INTP num_groups = fft_len / radix;
        for (FFTZ_INTP point = 0; point < radix; point++)
        {
            st->strides.in_strides[point] =
                point * num_groups * leaf_in_row_stride * DATA_STRIDE;
            st->strides.out_strides[point] = point * num_groups *
                                             out_row_stride *
                                             out_elem_stride * DATA_STRIDE;
        }
        st->strides.v_in_stride = DATA_STRIDE;
        st->strides.v_out_stride = out_elem_stride * DATA_STRIDE;
    }
    else
    {
        // Twiddle stage: radix pass over the running sub-DFT length. The last
        // stage writes the caller buffer (its element/row strides).
        FFTZ_INT32 is_last = (stage == num_stages - 1);
        FFTZ_INTP out_elem_stride = is_last ? dst_elem_stride : 1;
        FFTZ_INTP out_row_stride = is_last ? dst_row_stride : row_stride;
        FFTZ_INTP num_groups_next = fft_len / (sub_len * radix);
        for (FFTZ_INTP point = 0; point < radix; point++)
        {
            st->strides.in_strides[point] =
                point * num_groups_next * row_stride * DATA_STRIDE;
            st->strides.out_strides[point] = point * sub_len * num_groups_next *
                                             out_row_stride * out_elem_stride *
                                             DATA_STRIDE;
        }
        st->strides.v_in_stride = DATA_STRIDE;
        st->strides.v_out_stride = out_elem_stride * DATA_STRIDE;
    }

    // The c2c radix kernels step the second point-half by *_sym strides; this
    // dense layout shares the primary stride across both halves.
    st->strides.v_in_sym_stride = st->strides.v_in_stride;
    st->strides.v_out_sym_stride = st->strides.v_out_stride;
    return SOLVER_SUCCESS;
}

// Set up one sub-FFT: alloc stages, copy radixes/kffts, bind twiddle tables in
// `tw_base` (advancing `*tw_pair_off`), and precompute stage strides.
static FFTZ_INT32 fourstep_setup_subfft(aoclfftz_pow2_fourstep_subfft_t *sf,
                                        FFTZ_INTP fft_len, FFTZ_INTP num_cols,
                                        FFTZ_INTP row_stride,
                                        const FFTZ_INTP *radixes,
                                        kfft_ const *kffts,
                                        FFTZ_INT32 num_stages,
                                        FFTZ_VOID *tw_base,
                                        FFTZ_INTP *tw_pair_off,
                                        FFTZ_UINT8 precision,
                                        FFTZ_UINT32 dt_bytes,
                                        FFTZ_INTP leaf_in_row_stride,
                                        FFTZ_INTP dst_elem_stride,
                                        FFTZ_INTP dst_row_stride)
{
    sf->fft_len = fft_len;
    sf->num_cols = num_cols;
    sf->row_stride = row_stride;
    sf->num_stages = num_stages;
    sf->stages = NULL;

    ALLOC_ALIGN_INIT(
        sf->stages, aoclfftz_pow2_stage_t,
        (FFTZ_INTP)num_stages * sizeof(aoclfftz_pow2_stage_t));
    if (sf->stages == NULL)
    {
        return SOLVER_FAILURE;
    }

    FFTZ_INTP sub_len = radixes[0];
    for (FFTZ_INT32 stage = 0; stage < num_stages; stage++)
    {
        sf->stages[stage].radix = radixes[stage];
        sf->stages[stage].kfft[FORWARD_FFT_DIR] =
            kffts[stage * NUM_FFT_DIRS + FORWARD_FFT_DIR];
        sf->stages[stage].kfft[BACKWARD_FFT_DIR] =
            kffts[stage * NUM_FFT_DIRS + BACKWARD_FFT_DIR];
        sf->stages[stage].twiddle = NULL;

        // sub_len entering this iteration is the product of the earlier radixes
        // (the leaf span), which the stride math and twiddle table both need.
        if (fourstep_fill_stage_strides(&sf->stages[stage], stage, num_stages,
                                        fft_len, row_stride, sub_len,
                                        leaf_in_row_stride, dst_elem_stride,
                                        dst_row_stride) != SOLVER_SUCCESS)
        {
            return SOLVER_FAILURE;
        }

        if (stage >= 1)
        {
            FFTZ_VOID *stage_tw = MOVE_ADDR(
                tw_base, (*tw_pair_off) * DATA_STRIDE * (FFTZ_INTP)dt_bytes);
            // load_multi_cols = 0: one column per call, so the table is column-
            // major and the register width is unused (hence sets = 1).
            compute_twiddle_buffer(stage_tw, radixes[stage], sub_len, 1, 0,
                                   precision);
            sf->stages[stage].twiddle = stage_tw;
            *tw_pair_off += radixes[stage] * sub_len;
            sub_len *= radixes[stage];
        }
    }
    return SOLVER_SUCCESS;
}

// Set up the four-step solver: decompose N = n1*n2, set up both sub-FFTs, fill
// step twiddles, bind the fused kernel, alloc scratch. Declines if not fusable.
FFTZ_INT32 setup_pow2_fourstep_solver(aoclfftz_solution_t *sol,
                                      kernel_t *kt_dft, kernel_t *kt_twid,
                                      FFTZ_INT64 *out_ops)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    if (out_ops != NULL)
    {
        *out_ops = 0;
    }

    FFTZ_INTP n = sol->decomp_scheme->dims[0].n;
    FFTZ_UINT8 precision = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    FFTZ_UINT8 direction = FFT_DIR(sol->decomp_scheme->flags);
    FFTZ_UINT32 dt_bytes = SOL_DT_SIZE(sol);
    FFTZ_INT32 opt_level = sol->decomp_scheme->cntrl_params->opt_level;

    // Owned twiddle buffer; tracked here so the fail: path can release it.
    FFTZ_VOID *tw_base = NULL;

    // Balanced split of N = 2^k into near-square factors n1 >= n2. The selector
    // gate runs the same helper, so the two can never disagree.
    FFTZ_INTP n1, n2;
    if (!pow2_balanced_split(n, &n1, &n2))
    {
        return SOLVER_FAILURE;
    }

    // Fused-only gate: n1, n2 must be tile-edge multiples and a fused
    // fused_twiddle_transpose kernel must exist for this config.
    FFTZ_INTP tile_size = (precision == DT_FLOAT)
                              ? FUSED_TWIDDLE_TRANSPOSE_FP32_MICRO_TILE
                              : FUSED_TWIDDLE_TRANSPOSE_FP64_MICRO_TILE;
    FFTZ_INTP block_size = (precision == DT_FLOAT)
                               ? FUSED_TWIDDLE_TRANSPOSE_FP32_CACHE_BLOCK
                               : FUSED_TWIDDLE_TRANSPOSE_FP64_CACHE_BLOCK;
    if ((n1 % tile_size) != 0 || (n2 % tile_size) != 0)
    {
        return SOLVER_FAILURE;
    }
    // Bind both directions: this node may later be replayed the other way.
    fused_twiddle_transpose_ tt_kernel[NUM_FFT_DIRS];
    tt_kernel[FORWARD_FFT_DIR] =
        register_fused_twiddle_transpose_kernel(opt_level, precision,
                                          FORWARD_FFT_DIR);
    tt_kernel[BACKWARD_FFT_DIR] =
        register_fused_twiddle_transpose_kernel(opt_level, precision,
                                          BACKWARD_FFT_DIR);
    if (tt_kernel[FORWARD_FFT_DIR] == NULL
        || tt_kernel[BACKWARD_FFT_DIR] == NULL)
    {
        return SOLVER_FAILURE;
    }

    // Decompose both sub-FFTs (sub1: n1-pt over n2 cols, sub2: n2-pt over n1
    // cols); the per-call column count ranks the kernel variants.
    FFTZ_INTP sub1_radixes[POW2_MAX_DECOMP_STAGES];
    FFTZ_INTP sub2_radixes[POW2_MAX_DECOMP_STAGES];
    kfft_ sub1_kffts[POW2_MAX_DECOMP_STAGES * NUM_FFT_DIRS];
    kfft_ sub2_kffts[POW2_MAX_DECOMP_STAGES * NUM_FFT_DIRS];
    FFTZ_INT32 sub1_stages =
        fourstep_decompose_subfft(n1, kt_dft, kt_twid, precision, direction, n2,
                                  sub1_radixes, sub1_kffts);
    FFTZ_INT32 sub2_stages =
        fourstep_decompose_subfft(n2, kt_dft, kt_twid, precision, direction, n1,
                                  sub2_radixes, sub2_kffts);
    if (sub1_stages <= 0 || sub2_stages <= 0)
    {
        return SOLVER_FAILURE;
    }

    // Anti-aliasing padded row strides (sub1 >= n2, sub2 >= n1); scratch bufs
    // sized for the larger role and zero-filled so folded pad lanes stay 0.
    FFTZ_INTP cache_line_size = cpuid_cache_line_size();
    FFTZ_INTP sub1_row_stride = fourstep_padded_row_stride(n2, dt_bytes,
                                                           cache_line_size);
    FFTZ_INTP sub2_row_stride = fourstep_padded_row_stride(n1, dt_bytes,
                                                           cache_line_size);
    FFTZ_INTP buf_elems = (n1 * sub1_row_stride > n2 * sub2_row_stride)
                              ? (n1 * sub1_row_stride)
                              : (n2 * sub2_row_stride);
    FFTZ_INTP buf_bytes =
        GET_PADDED_SIZE(buf_elems * DATA_STRIDE * (FFTZ_INTP)dt_bytes);

    aoclfftz_pow2_fourstep_t *fs = NULL;
    ALLOC_ALIGN_INIT(fs, aoclfftz_pow2_fourstep_t,
                     sizeof(aoclfftz_pow2_fourstep_t));
    if (fs == NULL)
    {
        goto fail;
    }
    fs->n1 = n1;
    fs->n2 = n2;
    fs->sub1.stages = NULL;
    fs->sub2.stages = NULL;
    fs->step_twiddles = NULL;
    fs->fused_twiddle_transpose[FORWARD_FFT_DIR] = tt_kernel[FORWARD_FFT_DIR];
    fs->fused_twiddle_transpose[BACKWARD_FFT_DIR] = tt_kernel[BACKWARD_FFT_DIR];
    fs->scratch = NULL;
    fs->buf_bytes = 0;
    fs->pool_bytes = 0;

    // One combined twiddle buffer (owned by sol->twiddle->twiddle_buf_ptr).
    // Layout: [ step (n1*n2 pairs) | sub1 tables | sub2 tables ].
    FFTZ_INTP step_pairs = n1 * n2;
    FFTZ_INTP sub1_pairs = fourstep_subfft_tw_pairs(sub1_radixes, sub1_stages);
    FFTZ_INTP sub2_pairs = fourstep_subfft_tw_pairs(sub2_radixes, sub2_stages);
    FFTZ_INTP total_pairs = step_pairs + sub1_pairs + sub2_pairs;

    tw_base = alloc_twiddle_buffer((FFTZ_UINTP)total_pairs, precision);
    if (tw_base == NULL)
    {
        goto fail;
    }
    sol->twiddle->twiddle_buf_ptr = tw_base;
    sol->twiddle->TW = NULL;
    sol->twiddle->load_multi_cols = 0;

    fs->step_twiddles = tw_base;
    fourstep_fill_blocked_step_twiddles(fs->step_twiddles, n1, n2, block_size,
                                        tile_size, precision);

    // Per-sub-FFT source/dest strides are batch-invariant, so stage strides
    // are precomputed now: sub1 dense-in -> padded buffer; sub2 padded ->
    // dense out.
    FFTZ_INTP out_stride = sol->decomp_scheme->dims[0].out_stride;
    // The sub-FFT tables start after the step table.
    FFTZ_INTP tw_pair_off = step_pairs;
    if (fourstep_setup_subfft(&fs->sub1, n1, n2, sub1_row_stride, sub1_radixes,
                              sub1_kffts, sub1_stages, tw_base, &tw_pair_off,
                              precision, dt_bytes, n2, 1,
                              sub1_row_stride) != SOLVER_SUCCESS)
    {
        goto fail;
    }
    if (fourstep_setup_subfft(&fs->sub2, n2, n1, sub2_row_stride, sub2_radixes,
                              sub2_kffts, sub2_stages, tw_base, &tw_pair_off,
                              precision, dt_bytes, sub2_row_stride, out_stride,
                              n1) != SOLVER_SUCCESS)
    {
        goto fail;
    }

    // Ping-pong pool, one slot of two padded buffers per active thread (a
    // thread takes the slot at its ctx->slot_idx):
    // - the whole pipeline rotates between the two, so no third buffer;
    // - solver-owned: ancestors alias the CT scratch onto the caller's output;
    // - zero-initialised so the folded pad lanes start at 0.
    {
        FFTZ_INT32 slots = sol->decomp_scheme->thread_info->active_threads;
        if (slots < 1)
        {
            slots = 1;
        }
        fs->buf_bytes = buf_bytes;
        fs->pool_bytes = (FFTZ_UINTP)slots * 2u * (FFTZ_UINTP)buf_bytes;

        ALLOC_ALIGN_INIT(fs->scratch, FFTZ_VOID, fs->pool_bytes);
        if (fs->scratch == NULL)
        {
            goto fail;
        }
    }

    sol->next_sol = NULL;
    sol->dft_bufs->pow2_fourstep = fs;

    // Per-FFT cost (cycles): sum both sub-FFTs' stage-kernel costs plus
    // a streaming term for the fused pass; caller scales by the outer batch.
    FFTZ_INT64 solver_ops = 0;
    {
        FFTZ_INTP sub1_batch = n2, sub2_batch = n1;
        for (FFTZ_INT32 stage = 0; stage < sub1_stages; stage++)
        {
            kernel_t *kt = (stage == 0) ? kt_dft : kt_twid;
            FFTZ_INT64 stage_cost = 0;
            pow2_radix_pick_kernel(kt, sub1_radixes[stage], precision,
                                   direction, sub1_batch, NULL, &stage_cost,
                                   NULL);
            solver_ops += stage_cost;
        }
        for (FFTZ_INT32 stage = 0; stage < sub2_stages; stage++)
        {
            kernel_t *kt = (stage == 0) ? kt_dft : kt_twid;
            FFTZ_INT64 stage_cost = 0;
            pow2_radix_pick_kernel(kt, sub2_radixes[stage], precision,
                                   direction, sub2_batch, NULL, &stage_cost,
                                   NULL);
            solver_ops += stage_cost;
        }
        // Fused twiddle+transpose streaming pass.
        solver_ops += (FFTZ_INT64)n1 * n2;
    }
    if (out_ops != NULL)
    {
        *out_ops = solver_ops;
    }

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SOLVER_SUCCESS;

fail:
    // Release the twiddle buffer this setup allocated so a fall-through to
    // another solver neither leaks it nor reuses stale twiddle state.
    if (tw_base != NULL)
    {
        FREE_ALIGN_ALLOCATED_MEM(tw_base);
        if (sol->twiddle->twiddle_buf_ptr == tw_base)
        {
            sol->twiddle->twiddle_buf_ptr = NULL;
            sol->twiddle->TW = NULL;
            sol->twiddle->load_multi_cols = 0;
        }
    }
    destroy_pow2_fourstep(fs);
    sol->dft_bufs->pow2_fourstep = NULL;
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit (fail)");
    return SOLVER_FAILURE;
}

static inline void fourstep_colfft_leaf(
    const aoclfftz_pow2_stage_t *stage0, FFTZ_VOID *src_base,
    FFTZ_INTP leaf_in_row_stride, FFTZ_VOID *out_base,
    FFTZ_INTP out_elem_stride, FFTZ_INTP out_row_stride,
    FFTZ_INTP leaf_num_groups, FFTZ_INTP num_cols, FFTZ_UINT32 dt_bytes,
    FFTZ_UINT8 direction)
{
    aoclfftz_strides_t *strides = (aoclfftz_strides_t *)&stage0->strides;

    if (leaf_in_row_stride == out_row_stride)
    {
        stage0->kfft[direction](src_base,
                                MOVE_ADDR(src_base, (FFTZ_INTP)dt_bytes),
                                out_base,
                                MOVE_ADDR(out_base, (FFTZ_INTP)dt_bytes),
                                leaf_num_groups * out_row_stride, strides, NULL,
                                direction);
    }
    else
    {
        FFTZ_INTP in_row_bytes =
            leaf_in_row_stride * DATA_STRIDE * (FFTZ_INTP)dt_bytes;
        FFTZ_INTP out_row_bytes = out_row_stride * out_elem_stride
                                  * DATA_STRIDE * (FFTZ_INTP)dt_bytes;
        for (FFTZ_INTP group = 0; group < leaf_num_groups; group++)
        {
            FFTZ_VOID *in_row = MOVE_ADDR(src_base, group * in_row_bytes);
            FFTZ_VOID *out_row = MOVE_ADDR(out_base, group * out_row_bytes);
            stage0->kfft[direction](in_row,
                                    MOVE_ADDR(in_row, (FFTZ_INTP)dt_bytes),
                                    out_row,
                                    MOVE_ADDR(out_row, (FFTZ_INTP)dt_bytes),
                                    num_cols, strides, NULL, direction);
        }
    }
}

static inline void fourstep_colfft_stage(
    const aoclfftz_pow2_stage_t *stage, FFTZ_VOID *cur_buf,
    FFTZ_VOID *stage_out, FFTZ_INTP out_elem_stride, FFTZ_INTP out_row_stride,
    FFTZ_INTP row_stride, FFTZ_INTP fft_len, FFTZ_INTP sub_len,
    FFTZ_INTP num_cols, FFTZ_UINT32 dt_bytes, FFTZ_UINT8 direction)
{
    FFTZ_INTP radix = stage->radix;
    FFTZ_INTP num_groups = fft_len / sub_len;
    FFTZ_INTP num_groups_next = fft_len / (sub_len * radix);

    aoclfftz_strides_t *strides = (aoclfftz_strides_t *)&stage->strides;
    // load_multi_cols = 0 layout: each column owns radix-1 contiguous twiddle
    // pairs, so consecutive columns are that far apart.
    FFTZ_INTP tw_col_stride = (radix - 1) * DATA_STRIDE * (FFTZ_INTP)dt_bytes;

    // Per-column base pointers and twiddle match in both paths; only the
    // per-column group dispatch differs (fold path needs out_row==row stride).
    FFTZ_INTP in_row_bytes = row_stride * DATA_STRIDE * (FFTZ_INTP)dt_bytes;
    FFTZ_INTP out_row_bytes =
        out_row_stride * out_elem_stride * DATA_STRIDE * (FFTZ_INTP)dt_bytes;
    FFTZ_INT32 fold_groups = (out_row_stride == row_stride);

    for (FFTZ_INTP col = 0; col < sub_len; col++)
    {
        FFTZ_VOID *in_col =
            MOVE_ADDR(cur_buf, (col * num_groups) * in_row_bytes);
        FFTZ_VOID *out_col =
            MOVE_ADDR(stage_out, (col * num_groups_next) * out_row_bytes);

        aoclfftz_twiddle_t tw;
        tw.TW = MOVE_ADDR(stage->twiddle, col * tw_col_stride);
        tw.load_multi_cols = 0;
        tw.twiddle_buf_ptr = NULL;

        if (fold_groups)
        {
            // Matching row strides: one call sweeps all groups (with pad cols)
            // as a single contiguous batch.
            stage->kfft[direction](in_col,
                                   MOVE_ADDR(in_col, (FFTZ_INTP)dt_bytes),
                                   out_col,
                                   MOVE_ADDR(out_col, (FFTZ_INTP)dt_bytes),
                                   num_groups_next * row_stride, strides, &tw,
                                   direction);
        }
        else
        {
            // Differing row strides: dispatch each group over real cols only.
            for (FFTZ_INTP group = 0; group < num_groups_next; group++)
            {
                FFTZ_VOID *in_base = MOVE_ADDR(in_col, group * in_row_bytes);
                FFTZ_VOID *out_base = MOVE_ADDR(out_col, group * out_row_bytes);
                stage->kfft[direction](in_base,
                                       MOVE_ADDR(in_base, (FFTZ_INTP)dt_bytes),
                                       out_base,
                                       MOVE_ADDR(out_base, (FFTZ_INTP)dt_bytes),
                                       num_cols, strides, &tw, direction);
            }
        }
    }
}

static void fourstep_batched_colfft(
    const aoclfftz_pow2_fourstep_subfft_t *sub_ffts, FFTZ_VOID *src_base,
    FFTZ_INTP leaf_in_row_stride, FFTZ_VOID *dst_base,
    FFTZ_INTP dst_elem_stride, FFTZ_INTP dst_row_stride,
    FFTZ_VOID *scratch_ping, FFTZ_VOID *scratch_pong, FFTZ_UINT32 dt_bytes,
    FFTZ_UINT8 direction)
{
    FFTZ_INTP fft_len = sub_ffts->fft_len;
    FFTZ_INTP num_cols = sub_ffts->num_cols;
    FFTZ_INTP row_stride = sub_ffts->row_stride;
    FFTZ_INT32 num_stages = sub_ffts->num_stages;
    const aoclfftz_pow2_stage_t *stages = sub_ffts->stages;

    FFTZ_INTP leaf_radix = stages[0].radix;
    FFTZ_INTP leaf_num_groups = fft_len / leaf_radix;

    // Single-stage transforms write to the caller buffer; otherwise the
    // leaf writes the first padded scratch buffer.
    FFTZ_VOID *leaf_out = (num_stages == 1) ? dst_base : scratch_ping;
    FFTZ_INTP leaf_out_elem_stride = (num_stages == 1) ? dst_elem_stride : 1;
    FFTZ_INTP leaf_out_row_stride =
        (num_stages == 1) ? dst_row_stride : row_stride;

    fourstep_colfft_leaf(&stages[0], src_base, leaf_in_row_stride, leaf_out,
                         leaf_out_elem_stride, leaf_out_row_stride,
                         leaf_num_groups, num_cols, dt_bytes, direction);

    if (num_stages == 1)
    {
        return;
    }

    // sub_len is the running sub-DFT length (product of radixes done so far).
    FFTZ_VOID *cur_buf = scratch_ping;
    FFTZ_VOID *alt_buf = scratch_pong;
    FFTZ_INTP sub_len = leaf_radix;
    for (FFTZ_INT32 stage = 1; stage < num_stages; stage++)
    {
        FFTZ_INT32 is_last = (stage == num_stages - 1);
        FFTZ_VOID *stage_out = is_last ? dst_base : alt_buf;
        FFTZ_INTP out_elem_stride = is_last ? dst_elem_stride : 1;
        FFTZ_INTP out_row_stride = is_last ? dst_row_stride : row_stride;

        fourstep_colfft_stage(&stages[stage], cur_buf, stage_out,
                              out_elem_stride, out_row_stride, row_stride,
                              fft_len, sub_len, num_cols, dt_bytes, direction);

        sub_len *= stages[stage].radix;
        if (!is_last)
        {
            FFTZ_VOID *tmp = cur_buf;
            cur_buf = alt_buf;
            alt_buf = tmp;
        }
    }
}

// Four-step executor: per batch run sub1 (col FFTs -> transposed n1 x n2), the
// fused twiddle+transpose (-> n2 x n1), then sub2 (col FFTs -> natural order).
static FFTZ_INT32 execute_pow2_fourstep_solver(aoclfftz_solution_t *sol,
                                               aoclfftz_mutable_ctx_t *ctx)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    aoclfftz_pow2_fourstep_t *fs = sol->dft_bufs->pow2_fourstep;
    FFTZ_UINT32 dt_bytes = CTX_DT_SIZE(ctx);
    FFTZ_UINT8 direction = FFT_DIR(ctx->flags);
    FFTZ_INTP out_stride = sol->decomp_scheme->dims[0].out_stride;

    FFTZ_VOID *in_real = ctx->in_real;
    FFTZ_VOID *out_real = ctx->out_real;

    // This thread's slot of two padded buffers; slot_idx is the dense index in
    // [0, active_threads) that MT_BATCHED composes on the way down. The caller
    // buffers are touched only by the first read and the last write.
    FFTZ_VOID *buf_a = MOVE_ADDR(ctx->pow2_buf_base,
                                 (FFTZ_INTP)ctx->slot_idx * 2 * fs->buf_bytes);
    FFTZ_VOID *buf_b = MOVE_ADDR(buf_a, fs->buf_bytes);

    FFTZ_INTP n1 = fs->n1;
    FFTZ_INTP n2 = fs->n2;
    FFTZ_INTP sub1_row_stride = fs->sub1.row_stride;
    FFTZ_INTP sub2_row_stride = fs->sub2.row_stride;

    FFTZ_INTP batches = sol->decomp_scheme->vecs[0].n;
    FFTZ_INTP v_in_stride_bytes = sol->decomp_scheme->vecs[0].in_stride
                                  * DATA_STRIDE * (FFTZ_INTP)dt_bytes;
    FFTZ_INTP v_out_stride_bytes = sol->decomp_scheme->vecs[0].out_stride
                                   * DATA_STRIDE * (FFTZ_INTP)dt_bytes;

    // sub1 starts in buf_a, so an odd stage count leaves its result there and
    // an even one in buf_b; the fused pass transposes it into the other buffer.
    FFTZ_VOID *sub1_res = (fs->sub1.num_stages & 1) ? buf_a : buf_b;
    FFTZ_VOID *fused_dst = (sub1_res == buf_a) ? buf_b : buf_a;

    for (FFTZ_INTP batch = 0; batch < batches; batch++)
    {
        // sub1: n1-pt FFT over n2 dense-input cols -> transposed padded result.
        // Ping-pong buffers: buf_a (leaf target) and buf_b; result in sub1_res.
        fourstep_batched_colfft(&fs->sub1, in_real, n2, sub1_res, 1,
                                sub1_row_stride, buf_a, buf_b, dt_bytes,
                                direction);

        // Fused step2+step3: multiply sub1_res by step twiddles and write its
        // transpose into the other buffer (fused_dst) in a single pass.
        fs->fused_twiddle_transpose[direction](sub1_res, fused_dst,
                                              fs->step_twiddles, n1, n2,
                                              sub1_row_stride,
                                              sub2_row_stride);

        // sub2: n2-pt FFT over n1 fused_dst cols -> natural-order out_real. Its
        // leaf writes sub1_res (now free), then the two buffers ping-pong.
        fourstep_batched_colfft(&fs->sub2, fused_dst, sub2_row_stride, out_real,
                                out_stride, n1, sub1_res, fused_dst, dt_bytes,
                                direction);

        in_real = MOVE_ADDR(in_real, v_in_stride_bytes);
        out_real = MOVE_ADDR(out_real, v_out_stride_bytes);
    }

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SOLVER_SUCCESS;
}

dft_solver_ register_execute_pow2_fourstep_solver(FFTZ_VOID)
{
    return execute_pow2_fourstep_solver;
}

