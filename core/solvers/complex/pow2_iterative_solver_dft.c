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

#if defined(__GNUC__) || defined(__clang__)
    #define LOG_BASE_2(n) (63 - __builtin_clzll((FFTZ_UINT64)(n)))
#else
    static inline FFTZ_INT32 fftz_log2_floor(FFTZ_UINT64 v)
    {
        FFTZ_INT32 r = 0;
        while (v >>= 1)
        {
            r++;
        }
        return r;
    }
    #define LOG_BASE_2(n) fftz_log2_floor((FFTZ_UINT64)(n))
#endif

// This solver plans its own kernels; the sibling complex families are planned
// selector-side instead.
//
// Lowest-cost kernel variant (across ISA categories) for the exact `radix`,
// ranked by cost for `direction`. Both directions come from the *same* table
// entry: a node can run either way (Bluestein drives its inner FFT forward then
// backward through one node), and one entry means one register width, so the
// twiddle layout repacked by the selector stays valid for both.
//
// Fills out_kfft[] and returns 1, or 0 if no usable entry exists. out_cost and
// out_sets, when non-NULL, receive the variant's estimated cycles for `batch`
// and its register width.
static FFTZ_INT32 pow2_pick_kernel(kernel_t *kertab, FFTZ_INTP radix,
                                   FFTZ_UINT8 precision, FFTZ_UINT8 direction,
                                   FFTZ_INTP batch, kfft_ *out_kfft,
                                   FFTZ_INT64 *out_cost, FFTZ_INTP *out_sets)
{
    if (out_cost != NULL)
    {
        *out_cost = 0;
    }
    if (out_sets != NULL)
    {
        *out_sets = 1;
    }

    FFTZ_INTP base_idx = find_radix_base_idx(kertab, radix);
    if (base_idx < 0)
    {
        return 0;
    }

    kernel_choice_t choice = find_best_kernel(kertab, base_idx,
                                              precision, direction,
                                              batch);
    if (choice.idx < 0)
    {
        return 0;
    }

    kfft_ fwd = kertab[choice.idx].kfft[FORWARD_FFT_DIR];
    kfft_ bwd = kertab[choice.idx].kfft[BACKWARD_FFT_DIR];
    if (fwd == NULL || bwd == NULL)
    {
        return 0;
    }

    out_kfft[FORWARD_FFT_DIR] = fwd;
    out_kfft[BACKWARD_FFT_DIR] = bwd;
    if (out_cost != NULL)
    {
        *out_cost = choice.cost;
    }
    if (out_sets != NULL)
    {
        *out_sets = (FFTZ_INTP)kertab[choice.idx].sets[precision - 2];
    }
    return 1;
}

// Whether `kertab` has an entry for `radix` that serves both FFT directions.
static FFTZ_INT32 pow2_radix_has_both_directions(kernel_t *kertab,
                                                 FFTZ_INTP radix)
{
    FFTZ_INTP base_idx = find_radix_base_idx(kertab, radix);
    if (base_idx < 0)
    {
        return 0;
    }
    for (FFTZ_INTP kcat = 0; kcat < NUM_KERNEL_CATEGORIES; kcat++)
    {
        FFTZ_INTP kloc = kcat * NUM_KERNELS_IN_EACH_CATEGORY + base_idx;
        if (kertab[kloc].kfft[FORWARD_FFT_DIR] != NULL
            && kertab[kloc].kfft[BACKWARD_FFT_DIR] != NULL)
        {
            return 1;
        }
    }
    return 0; // radix present but not usable in both directions
}

// Largest radix in `kertab` that divides `remaining` and serves both
// directions, or 0 if there is none.
static FFTZ_INTP pow2_pick_largest(kernel_t *kertab, FFTZ_INTP remaining)
{
    FFTZ_INTP best_radix = 0;
    for (FFTZ_INTP b = 0; b < NUM_KERNELS_IN_EACH_CATEGORY; b++)
    {
        FFTZ_INTP r = (FFTZ_INTP)kertab[b].radix;
        if (r == 0) // end of the kernel list
        {
            break;
        }
        if (r <= best_radix || r > remaining || (remaining % r) != 0)
        {
            continue;
        }
        if (pow2_radix_has_both_directions(kertab, r))
        {
            best_radix = r;
        }
    }
    return best_radix;
}

// Factor n = 2^log2(n) into the fewest radix stages from {2,4,8,16}, spreading
// the exponent evenly so the smallest radix is maximised: 1024 -> [16, 8, 8],
// not [16, 16, 4]. `n` must be a power of two, which
// is_pow2_iterative_applicable guarantees. Returns the stage count, or 0 if the
// split needs too many stages or a stage's radix has no bidirectional kernel.
static FFTZ_INT32 pow2_balanced_decompose(FFTZ_INTP n, kernel_t *kt_dft,
                                          kernel_t *kt_twid, FFTZ_INTP *radixes)
{
    FFTZ_INT32 log2n = LOG_BASE_2(n);

    // Each stage covers a radix of at most 16 = 2^4, so at least ceil(log2n/4)
    // stages are needed.
    FFTZ_INT32 num_stages = (log2n + 3) / 4;
    if (num_stages > POW2_ITERATIVE_MAX_STAGES)
    {
        return 0;
    }

    // Give every stage the same base exponent; hand the leftover units one each
    // to the first few stages.
    FFTZ_INT32 base_exponent = log2n / num_stages;
    FFTZ_INT32 stages_with_extra = log2n % num_stages;
    for (FFTZ_INT32 stage = 0; stage < num_stages; stage++)
    {
        FFTZ_INT32 exponent = base_exponent + (stage < stages_with_extra ? 1 : 0);
        FFTZ_INTP radix = (FFTZ_INTP)1 << exponent;
        kernel_t *kt = (stage == 0) ? kt_dft : kt_twid;
        if (!pow2_radix_has_both_directions(kt, radix))
        {
            return 0;
        }
        radixes[stage] = radix;
    }
    return num_stages;
}

// Factor n by taking the largest usable radix at each stage. Returns the stage
// count, or 0 if n cannot be consumed within POW2_ITERATIVE_MAX_STAGES stages.
static FFTZ_INT32 pow2_greedy_decompose(FFTZ_INTP n, kernel_t *kt_dft,
                                        kernel_t *kt_twid, FFTZ_INTP *radixes)
{
    FFTZ_INTP remaining = n;
    FFTZ_INT32 num_stages = 0;

    while (remaining > 1 && num_stages < POW2_ITERATIVE_MAX_STAGES)
    {
        kernel_t *kt = (num_stages == 0) ? kt_dft : kt_twid;
        FFTZ_INTP radix = pow2_pick_largest(kt, remaining);
        if (radix == 0)
        {
            return 0;
        }
        radixes[num_stages] = radix;
        remaining /= radix;
        num_stages++;
    }
    return (remaining == 1) ? num_stages : 0;
}

// Factor n into radix stages (stage 0 from kt_dft, the rest from kt_twid).
// Prefers the balanced split, falls back to greedy largest-first. Returns the
// stage count, or 0 on failure.
static FFTZ_INT32 pow2_decompose(FFTZ_INTP n, kernel_t *kt_dft, kernel_t *kt_twid,
                             FFTZ_INTP *radixes)
{
    FFTZ_INT32 num_stages = pow2_balanced_decompose(n, kt_dft, kt_twid, radixes);
    if (num_stages > 0)
    {
        return num_stages;
    }
    return pow2_greedy_decompose(n, kt_dft, kt_twid, radixes);
}

// Fill in one stage's counts and strides; `cols` is the product of the earlier
// radixes and `stage_data->radix` must already be set. Stage 0 gathers the
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
    FFTZ_INTP radix = stage_data->radix;

    if (stage_num == 0)
    {
        FFTZ_INTP vecs_0 = n / radix;
        stage_data->count = vecs_0;
        stage_data->num_groups = 1;
        for (FFTZ_INTP i = 0; i < radix; i++)
        {
            stage_data->strides.in_strides[i] = i * vecs_0 * in_stride * DATA_STRIDE;
            stage_data->strides.out_strides[i] = i * DATA_STRIDE;
        }
        stage_data->strides.v_in_stride = in_stride * DATA_STRIDE;
        stage_data->strides.v_out_stride = radix * DATA_STRIDE;
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
            stage_data->strides.in_strides[i] = i * num_groups * cols *
                                                DATA_STRIDE;

            stage_data->strides.out_strides[i] = i * cols *
                                                 eff_out_stride *
                                                 DATA_STRIDE;
        }
        stage_data->strides.v_in_stride = DATA_STRIDE;
        stage_data->strides.v_out_stride = eff_out_stride * DATA_STRIDE;

        stage_data->src_grp_stride = cols * DATA_STRIDE * (FFTZ_INTP)dt_bytes;

        stage_data->dst_grp_stride = (cols * radix) * eff_out_stride *
                                     DATA_STRIDE * (FFTZ_INTP)dt_bytes;
    }

    stage_data->strides.v_in_h2_stride = stage_data->strides.v_in_stride;
    stage_data->strides.v_out_h2_stride = stage_data->strides.v_out_stride;
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
    FFTZ_INTP cols = it->stages[0].radix;
    for (FFTZ_INT32 stage = 1; stage < it->num_stages; stage++)
    {
        total_tw += (it->stages[stage].radix - 1) * cols;
        cols *= it->stages[stage].radix;
    }

    FFTZ_VOID *TW = alloc_twiddle_buffer((FFTZ_UINTP)total_tw, dt_prec);
    if (TW == NULL)
    {
        return SOLVER_FAILURE;
    }
    sol->twiddle->twiddle_buf_ptr = TW;

    FFTZ_INTP offset = 0;
    cols = it->stages[0].radix;
    for (FFTZ_INT32 stage = 1; stage < it->num_stages; stage++)
    {
        FFTZ_INTP rs = it->stages[stage].radix;
        FFTZ_VOID *tw_s = MOVE_ADDR(TW,
                                    offset * DATA_STRIDE * (FFTZ_INTP)dt_bytes);
        // Repacked (linear) layout must match the register width of the kernel
        // that will consume it. The solver runs with load_multi_cols == 1.
        compute_twiddle_buffer(tw_s, rs, cols, it->stages[stage].sets, 1,
                               dt_prec);
        it->stages[stage].twiddle = tw_s;
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
FFTZ_INT32 setup_pow2_iterative_solver(aoclfftz_solution_t *sol, kernel_t *kt_dft,
                                  kernel_t *kt_twid, FFTZ_INT64 *out_cost)
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

    FFTZ_INTP radixes[POW2_ITERATIVE_MAX_STAGES];
    FFTZ_INT32 num_stages = pow2_decompose(n, kt_dft, kt_twid, radixes);

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

        stage_data->radix = radix;
        stage_data->kfft[FORWARD_FFT_DIR] = NULL;
        stage_data->kfft[BACKWARD_FFT_DIR] = NULL;
        stage_data->twiddle = NULL;
        stage_data->sets = 1;
        stage_data->strides.in_strides = NULL;
        stage_data->strides.out_strides = NULL;

        ALLOC_ALIGN_UNINIT(stage_data->strides.in_strides, FFTZ_INTP, radix * sizeof(FFTZ_INTP));
        ALLOC_ALIGN_UNINIT(stage_data->strides.out_strides, FFTZ_INTP, radix * sizeof(FFTZ_INTP));
        if (stage_data->strides.in_strides == NULL || stage_data->strides.out_strides == NULL)
        {
            goto exit_pow2_iterative;
        }

        pow2_setup_stage_layout(sol, stage_data, stage_num, num_stages, cols);

        FFTZ_INT64 stage_cost = 0;
        if (!pow2_pick_kernel(kt, radix, precision, direction,
                              stage_data->count, stage_data->kfft, &stage_cost,
                              &stage_data->sets))
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
                                 (FFTZ_INTP)ctx->thr_slot_idx * 2 *
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

        it->stages[0].kfft[direction](in_real, in_imag,
                           stage0_dst, MOVE_ADDR(stage0_dst, (FFTZ_INTP)dt_bytes),
                           it->stages[0].count, &it->stages[0].strides,
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
            tw.TW = stage->twiddle;
            tw.load_multi_cols = 1;

            FFTZ_VOID *dst_real = (!oop_dense && is_last) ? out_real : dst_buf;
            
            FFTZ_VOID *dst_imag = (!oop_dense && is_last)
                ? out_imag
                : MOVE_ADDR(dst_buf, (FFTZ_INTP)dt_bytes);

            for (FFTZ_INTP grp_idx = 0; grp_idx < stage->num_groups; grp_idx++)
            {
                FFTZ_INTP src_offset = grp_idx * stage->src_grp_stride;
                FFTZ_INTP dst_offset = grp_idx * stage->dst_grp_stride;
                stage->kfft[direction](MOVE_ADDR(src_buf, src_offset),
                            MOVE_ADDR(src_buf, src_offset + (FFTZ_INTP)dt_bytes),
                            MOVE_ADDR(dst_real, dst_offset),
                            MOVE_ADDR(dst_imag, dst_offset),
                            stage->count, &stage->strides, &tw, direction);
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
