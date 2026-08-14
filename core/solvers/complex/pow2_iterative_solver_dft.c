// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file pow2_iterative_solver_dft.c
 *
 *  @brief Power-of-2 iterative solver.
 *
 *  An iterative mixed-radix c2c FFT for a contiguous single power-of-two
 *  transform. N is factored into stages of radix {2,4,8,16}; the stage
 *  kernels run back-to-back over two ping-pong buffers.
 *
 *  @author Ashwin K. Godbole
 */

#include "core/solvers/solver.h"
#include "core/common/memory_manager.h"
#include "core/common/twiddle.h"
#include "core/common/pow2_radix_decompose.h"

// This solver plans its own kernels (the sibling families are planned selector-
// side), reusing the radix split shared with four-step in pow2_radix_decompose.

// Fill in one stage's counts and strides; `cols` is the product of the earlier
// radixes and `stage_data->stage_info.radix` must already be set. Stage 0 gathers the
// strided input into a dense buffer as one group; later stages walk num_groups
// groups over it, and the last writes out through the caller's output stride.
static FFTZ_VOID pow2_setup_stage_layout(aoclfftz_solution_t *sol,
                                    aoclfftz_pow2_iterative_stage_t *stage_data,
                                    FFTZ_INT32 stage_num, FFTZ_INT32 num_stages,
                                    FFTZ_INTP cols)
{
    FFTZ_INTP n = sol->decomp_scheme->dims[0].n;
    FFTZ_INTP in_stride = sol->decomp_scheme->dims[0].in_stride;
    FFTZ_INTP out_stride = sol->decomp_scheme->dims[0].out_stride;
    FFTZ_UINT32 dt_bytes = SOL_DT_SIZE(sol);
    FFTZ_INTP radix = stage_data->stage_info.radix;
    aoclfftz_strides_t *strides = &stage_data->stage_info.strides;

    if (stage_num == 0)
    {
        FFTZ_INTP vecs_0 = n / radix;
        stage_data->count = vecs_0;
        stage_data->num_groups = 1;
        for (FFTZ_INTP i = 0; i < radix; i++)
        {
            strides->in_strides[i] = i * vecs_0 * in_stride * DATA_STRIDE;
            strides->out_strides[i] = i * DATA_STRIDE;
        }
        strides->v_in_stride = in_stride * DATA_STRIDE;
        strides->v_out_stride = radix * DATA_STRIDE;
        stage_data->src_grp_stride = 0;
        stage_data->dst_grp_stride = 0;
    }
    else
    {
        FFTZ_INT32 is_last = (stage_num == num_stages - 1);
        FFTZ_INTP eff_out_stride = is_last ? out_stride : 1;
        FFTZ_INTP num_groups = n / (cols * radix);
        stage_data->count = cols;
        stage_data->num_groups = num_groups;
        for (FFTZ_INTP i = 0; i < radix; i++)
        {
            strides->in_strides[i] = i * num_groups * cols * DATA_STRIDE;

            strides->out_strides[i] = i * cols * eff_out_stride * DATA_STRIDE;
        }
        strides->v_in_stride = DATA_STRIDE;
        strides->v_out_stride = eff_out_stride * DATA_STRIDE;

        stage_data->src_grp_stride = cols * DATA_STRIDE * (FFTZ_INTP)dt_bytes;

        stage_data->dst_grp_stride = (cols * radix) * eff_out_stride *
                                     DATA_STRIDE * (FFTZ_INTP)dt_bytes;
    }

    strides->v_in_h2_stride = strides->v_in_stride;
    strides->v_out_h2_stride = strides->v_out_stride;
}

// Allocate one twiddle block for the plan and give each twiddle stage (>= 1)
// its slice, sized (radix - 1) * cols where cols is the product of earlier
// radixes. The block is owned by sol->twiddle->twiddle_buf_ptr and freed by the
// standard twiddle teardown; the per-stage pointers are slices into it.
static FFTZ_INT32 pow2_setup_twiddles(aoclfftz_solution_t *sol,
                                      aoclfftz_pow2_iterative_t *it)
{
    FFTZ_UINT32 dt_prec = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    FFTZ_UINT32 dt_bytes = DT_PRECISION_BYTES(dt_prec);

    FFTZ_INTP total_tw = 0;
    FFTZ_INTP cols = it->stages[0].stage_info.radix;
    for (FFTZ_INT32 stage = 1; stage < it->num_stages; stage++)
    {
        total_tw += (it->stages[stage].stage_info.radix - 1) * cols;
        cols *= it->stages[stage].stage_info.radix;
    }

    FFTZ_VOID *TW = alloc_twiddle_buffer((FFTZ_UINTP)total_tw, dt_prec);
    if (TW == NULL)
    {
        return SOLVER_FAILURE;
    }
    sol->twiddle->twiddle_buf_ptr = TW;

    FFTZ_INTP offset = 0;
    cols = it->stages[0].stage_info.radix;
    for (FFTZ_INT32 stage = 1; stage < it->num_stages; stage++)
    {
        FFTZ_INTP rs = it->stages[stage].stage_info.radix;
        FFTZ_VOID *tw_s = MOVE_ADDR(TW,
                                    offset * DATA_STRIDE * (FFTZ_INTP)dt_bytes);
        // Repacked (linear) layout must match the register width of the kernel
        // that will consume it. The solver runs with load_multi_cols == 1.
        compute_twiddle_buffer(tw_s, rs, cols, it->stages[stage].sets, 1,
                               dt_prec);
        it->stages[stage].stage_info.twiddle = tw_s;
        offset += (rs - 1) * cols;
        cols *= rs;
    }
    // Same address as stages[1].twiddle, but valid even for a single stage.
    sol->twiddle->TW = TW;
    return SOLVER_SUCCESS;
}

// Set up the solver: decompose N into radix stages, bind a kernel per stage and
// allocate the stage array, ping-pong pool and twiddles. Returns SOLVER_FAILURE
// when no radix split exists or any allocation fails.
FFTZ_INT32 setup_pow2_iterative_solver(aoclfftz_solution_t *sol,
                                       kernel_t *kt_dft, kernel_t *kt_twid,
                                       FFTZ_INT64 *out_cost)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    if (out_cost != NULL)
    {
        *out_cost = 0;
    }

    FFTZ_INTP n = sol->decomp_scheme->dims[0].n;
    FFTZ_UINT8 precision = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    FFTZ_UINT8 direction = FFT_DIR(sol->decomp_scheme->flags);
    FFTZ_UINT32 dt_bytes = SOL_DT_SIZE(sol);

    FFTZ_INTP radixes[POW2_MAX_DECOMP_STAGES];
    FFTZ_INT32 num_stages = pow2_radix_decompose(n, kt_dft, kt_twid, radixes);

    // The largest power-of-two radix is 16, so two stages reach only 256 and
    // POW2_ITERATIVE_MIN_N rules those sizes out. Only a kernel table with no
    // usable radix can get here.
    if (num_stages == 0)
    {
        return SOLVER_FAILURE;
    }

    FFTZ_INT64 solver_cost_ops = 0;

    aoclfftz_pow2_iterative_t *it = NULL;
    ALLOC_ALIGN_INIT(it, aoclfftz_pow2_iterative_t,
                     sizeof(aoclfftz_pow2_iterative_t));
    if (it == NULL)
    {
        goto exit_pow2_iterative;
    }
    it->num_stages = num_stages;
    it->stages = NULL;
    it->pingpong_buf = NULL;

    ALLOC_ALIGN_INIT(it->stages, aoclfftz_pow2_iterative_stage_t,
                     (FFTZ_INTP)num_stages * sizeof(aoclfftz_pow2_iterative_stage_t));
    if (it->stages == NULL)
    {
        goto exit_pow2_iterative;
    }

    // Precompute the per-stage kernel arguments.
    // `cols` tracks the product of the earlier radixes.
    FFTZ_INTP cols = 1;
    for (FFTZ_INT32 stage_num = 0; stage_num < num_stages; stage_num++)
    {
        aoclfftz_pow2_iterative_stage_t *stage_data = &it->stages[stage_num];
        FFTZ_INTP radix = radixes[stage_num];
        kernel_t *kt = (stage_num == 0) ? kt_dft : kt_twid;

        aoclfftz_strides_t *strides = &stage_data->stage_info.strides;

        stage_data->stage_info.radix = radix;
        stage_data->stage_info.kfft[FORWARD_FFT_DIR] = NULL;
        stage_data->stage_info.kfft[BACKWARD_FFT_DIR] = NULL;
        stage_data->stage_info.twiddle = NULL;
        stage_data->sets = 1;
        strides->in_strides = NULL;
        strides->out_strides = NULL;

        ALLOC_ALIGN_UNINIT(strides->in_strides, FFTZ_INTP, radix * sizeof(FFTZ_INTP));
        ALLOC_ALIGN_UNINIT(strides->out_strides, FFTZ_INTP, radix * sizeof(FFTZ_INTP));
        if (strides->in_strides == NULL || strides->out_strides == NULL)
        {
            goto exit_pow2_iterative;
        }

        pow2_setup_stage_layout(sol, stage_data, stage_num, num_stages, cols);

        FFTZ_INT64 stage_cost = 0;
        if (!pow2_radix_pick_kernel(kt, radix, precision, direction,
                                    stage_data->count, stage_data->stage_info.kfft,
                                    &stage_cost, &stage_data->sets))
        {
            goto exit_pow2_iterative;
        }
        solver_cost_ops += (FFTZ_INT64)stage_data->num_groups * stage_cost;

        cols *= radix;
    }

    // Ping-pong pool: buffers A and B, one pair per thread that can reach this
    // node concurrently. Solver-owned rather than carved from the shared CT
    // scratch because ancestors (BUFFERED, BLUESTEIN) alias ctx->ct_buf_base
    // onto the caller's output buffer, which only fits one of the two.
    //
    // Only the general execute path (strided or split-real/imag output) uses
    // buffer B -- the dense out-of-place case ping-pongs between the caller
    // output and buffer A -- but a later aoclfftz_execute_io swap can move any
    // plan onto the general path, so both are allocated once here at setup
    // rather than on the execute hot path.
    {
        FFTZ_INTP buf_bytes =
            GET_PADDED_SIZE(n * DATA_STRIDE * (FFTZ_INTP)dt_bytes);
        FFTZ_INT32 slots = sol->decomp_scheme->thread_info->active_threads;
        if (slots < 1)
        {
            slots = 1;
        }
        it->buf_bytes = buf_bytes;
        it->pool_bytes = (FFTZ_UINTP)slots * 2u * (FFTZ_UINTP)buf_bytes;

        ALLOC_ALIGN_UNINIT(it->pingpong_buf, FFTZ_VOID, it->pool_bytes);
        if (it->pingpong_buf == NULL)
        {
            goto exit_pow2_iterative;
        }
    }

    sol->twiddle->twiddle_buf_ptr = NULL;
    sol->twiddle->TW = NULL;
    sol->twiddle->load_multi_cols = 1;

    // Bound here rather than in the setup_twiddle_buffer_complex post-pass so
    // an allocation failure rejects the plan instead of surfacing at execute
    // time, letting the executor assume every twiddle stage has a buffer.
    if (pow2_setup_twiddles(sol, it) != SOLVER_SUCCESS)
    {
        goto exit_pow2_iterative;
    }

    sol->next_sol = NULL;

    sol->dft_bufs->pow2_iterative = it;

    if (out_cost != NULL)
    {
        *out_cost = solver_cost_ops;
    }

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SOLVER_SUCCESS;

exit_pow2_iterative:
    destroy_pow2_iterative(it);
    AOCLFFTZ_ERROR("SOLVER_FAILURE : pow2-iterative setup ran out of memory");
    return SOLVER_FAILURE;
}

// Fused multi-stage executor: stage 0 gathers strided inputs into a dense
// buffer; twiddle stages ping-pong until the last stage lands the result in `out`.
static FFTZ_INT32 execute_pow2_iterative_solver(aoclfftz_solution_t *sol,
                                               aoclfftz_mutable_ctx_t *ctx)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    aoclfftz_pow2_iterative_t *it = sol->dft_bufs->pow2_iterative;
    FFTZ_INT32 num_stages = it->num_stages;

    FFTZ_UINT32 dt_bytes = CTX_DT_SIZE(ctx);
    FFTZ_UINT8 direction = FFT_DIR(ctx->flags);
    FFTZ_INTP out_stride = sol->decomp_scheme->dims[0].out_stride;

    FFTZ_VOID *in_real = ctx->in_real;
    FFTZ_VOID *in_imag = ctx->in_imag;
    FFTZ_VOID *out_real = ctx->out_real;
    FFTZ_VOID *out_imag = ctx->out_imag;

    // This thread's ping-pong slot.
    FFTZ_VOID *buf_a = MOVE_ADDR(ctx->pow2_buf_base,
                                 (FFTZ_INTP)ctx->slot_idx * 2 *
                                 it->buf_bytes);
    FFTZ_VOID *buf_b = MOVE_ADDR(buf_a, it->buf_bytes);

    FFTZ_INTP batches = sol->decomp_scheme->vecs[0].n;
    FFTZ_INTP v_in_stride_bytes = sol->decomp_scheme->vecs[0].in_stride *
                              DATA_STRIDE * (FFTZ_INTP)dt_bytes;
    FFTZ_INTP v_out_stride_bytes = sol->decomp_scheme->vecs[0].out_stride *
                               DATA_STRIDE * (FFTZ_INTP)dt_bytes;

    // Out-of-place dense fast path: when the output is a distinct, unit-stride,
    // interleaved buffer, ping-pong between `out` and buf_a instead of buf_b.
    FFTZ_INT32 oop_dense = (in_real != out_real) && (out_stride == 1) &&
                      (out_imag == MOVE_ADDR(out_real, (FFTZ_INTP)dt_bytes));

    // oop_dense picks stage 0's destination by parity so the last twiddle stage
    // always lands on out_real, with dst_imag at real + dt_bytes. Otherwise the
    // stages ping-pong over buf_a/buf_b and the last writes out_real/out_imag.
    for (FFTZ_INTP batch_idx = 0; batch_idx < batches; batch_idx++)
    {
        FFTZ_VOID *stage0_dst;
        
        if (oop_dense)
        {
            stage0_dst = (((num_stages - 1) & 1) == 0) ? out_real : buf_a;
        }
        else
        {
            stage0_dst = buf_a;
        }

        it->stages[0].stage_info.kfft[direction](in_real, in_imag,
                           stage0_dst, MOVE_ADDR(stage0_dst, (FFTZ_INTP)dt_bytes),
                           it->stages[0].count, &it->stages[0].stage_info.strides,
                           NULL, direction);

        FFTZ_VOID *src_buf = stage0_dst;

        FFTZ_VOID *dst_buf = oop_dense
            ? (stage0_dst == out_real ? buf_a : out_real)
            : buf_b;

        for (FFTZ_INT32 stage_idx = 1; stage_idx < num_stages; stage_idx++)
        {
            FFTZ_INT32 is_last = (stage_idx == num_stages - 1);
            aoclfftz_pow2_iterative_stage_t *stage = &it->stages[stage_idx];

            aoclfftz_twiddle_t tw;
            tw.twiddle_buf_ptr = NULL;
            tw.TW = stage->stage_info.twiddle;
            tw.load_multi_cols = 1;

            FFTZ_VOID *dst_real = (!oop_dense && is_last) ? out_real : dst_buf;
            
            FFTZ_VOID *dst_imag = (!oop_dense && is_last)
                ? out_imag
                : MOVE_ADDR(dst_buf, (FFTZ_INTP)dt_bytes);

            for (FFTZ_INTP grp_idx = 0; grp_idx < stage->num_groups; grp_idx++)
            {
                FFTZ_INTP src_offset = grp_idx * stage->src_grp_stride;
                FFTZ_INTP dst_offset = grp_idx * stage->dst_grp_stride;
                stage->stage_info.kfft[direction](MOVE_ADDR(src_buf, src_offset),
                            MOVE_ADDR(src_buf, src_offset + (FFTZ_INTP)dt_bytes),
                            MOVE_ADDR(dst_real, dst_offset),
                            MOVE_ADDR(dst_imag, dst_offset),
                            stage->count, &stage->stage_info.strides, &tw,
                            direction);
            }

            if (!is_last)
            {
                FFTZ_VOID *tmp = src_buf;
                src_buf = dst_buf;
                dst_buf = tmp;
            }
        }

        in_real = MOVE_ADDR(in_real, v_in_stride_bytes);
        in_imag = MOVE_ADDR(in_imag, v_in_stride_bytes);
        out_real = MOVE_ADDR(out_real, v_out_stride_bytes);
        out_imag = MOVE_ADDR(out_imag, v_out_stride_bytes);
    }

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SOLVER_SUCCESS;
}

dft_solver_ register_execute_pow2_iterative_solver(FFTZ_VOID)
{
    return execute_pow2_iterative_solver;
}

