// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file sr_solver_dft.c
 *
 *  @brief Split-Radix Solver implementation
 *
 *  This file contains the implementation of the Split-Radix FFT solver.
 *
 *  Split-radix algorithm decomposes N-point FFT as:
 *  - Even part: N/2-point FFT (indices 0, 2, 4, 6, ...)
 *  - Odd part 1: N/4-point FFT (indices 1, 5, 9, 13, ...)
 *  - Odd part 2: N/4-point FFT (indices 3, 7, 11, 15, ...)
 *
 *  @author Varaprasad, Malothu
 */

#include "api/aoclfftz_internal.h"
#include "core/common/memory_manager.h"
#include "core/kernels/kernel.h"
#include "selector/selector.h"

/**
 * @brief Combines even and odd sub-FFT results using split-radix butterfly
 *
 * @param[in,out] sol  Solution object with sub-problem results
 *
 * @return SOLVER_SUCCESS on success, SOLVER_FAILURE on error
 */
static FFTZ_INT32 split_radix_butterfly(aoclfftz_solution_t *sol);

/* Setup: configure the three SR sub-problems */

FFTZ_INT32 setup_sr_solver(aoclfftz_solution_t *sol,
                      aoclfftz_solution_t *sol_even,
                      aoclfftz_solution_t *sol_odd1,
                      aoclfftz_solution_t *sol_odd3,
                      FFTZ_INTP n_even, FFTZ_INTP n_odd)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    /* Setup even part (N/2 sub-problem): indices 0, 2, 4, 6, ... */
    FFTZ_INT32 ret = copy_solution_obj(sol_even, sol);
    if (ret != AOCLFFTZ_SUCCESS)
    {
        AOCLFFTZ_ERROR("copy_solution_obj failed: %s", get_status_string(ret));
        return ret;
    }
    sol_even->decomp_scheme->dims[0].n = n_even;
    sol_even->decomp_scheme->dims[0].in_stride =
        2 * sol->decomp_scheme->dims[0].in_stride;
    sol_even->decomp_scheme->dims[0].out_stride =
        sol->decomp_scheme->dims[0].out_stride;
        /*  vecs don't play a role in SR solver */
    sol_even->decomp_scheme->vecs[0].n = 1;

    /* Setup odd1 part (N/4 sub-problem): indices 1, 5, 9, 13, ... */
    ret = copy_solution_obj(sol_odd1, sol);
    if (ret != AOCLFFTZ_SUCCESS)
    {
        AOCLFFTZ_ERROR("copy_solution_obj failed: %s", get_status_string(ret));
        return ret;
    }
    sol_odd1->decomp_scheme->dims[0].n = n_odd;
    sol_odd1->decomp_scheme->dims[0].in_stride =
        4 * sol->decomp_scheme->dims[0].in_stride;
    sol_odd1->decomp_scheme->dims[0].out_stride =
        sol->decomp_scheme->dims[0].out_stride;
    sol_odd1->decomp_scheme->vecs[0].n = 1;

    /* Setup odd3 part (N/4 sub-problem): indices 3, 7, 11, 15, ... */
    ret = copy_solution_obj(sol_odd3, sol);
    if (ret != AOCLFFTZ_SUCCESS)
    {
        AOCLFFTZ_ERROR("copy_solution_obj failed: %s", get_status_string(ret));
        return ret;
    }
    sol_odd3->decomp_scheme->dims[0].n = n_odd;
    sol_odd3->decomp_scheme->dims[0].in_stride =
        4 * sol->decomp_scheme->dims[0].in_stride;
    sol_odd3->decomp_scheme->dims[0].out_stride =
        sol->decomp_scheme->dims[0].out_stride;
    sol_odd3->decomp_scheme->vecs[0].n = 1;

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return SOLVER_SUCCESS;
}

FFTZ_INT32 execute_sr_solver(aoclfftz_solution_t *sol)
{
    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Enter");

    /* Validate input pointers */
    if (!sol || !sol->decomp_scheme || !sol->next_sol || !sol->dft_bufs)
    {
        AOCLFFTZ_LOG(
            INFO, global_logger_mode,
            "SR execute: NULL sol, decomp_scheme, next_sol, or dft_bufs");
        return SOLVER_FAILURE;
    }

    /* Get sub-problems */
    aoclfftz_solution_t *even_sol = sol->next_sol[0];
    aoclfftz_solution_t *odd1_sol = sol->dft_bufs->sr->odd1_sol;
    aoclfftz_solution_t *odd3_sol = sol->dft_bufs->sr->odd3_sol;

    FFTZ_INTP n = sol->decomp_scheme->dims[0].n;
    FFTZ_INTP n2 = n >> 1;  /* n/2 */
    FFTZ_INTP n4 = n >> 2;  /* n/4 */

    FFTZ_UINT32 precision = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    FFTZ_UINT32 dt_bytes = DT_PRECISION_BYTES(precision);
    FFTZ_INTP elem_size = DATA_STRIDE * dt_bytes;
    FFTZ_INTP in_stride = sol->decomp_scheme->dims[0].in_stride;
    FFTZ_INTP out_stride = sol->decomp_scheme->dims[0].out_stride;

    /*
     * Check addresses (not flags) to decide in-place vs out-of-place.
     * NDIM solver may set same buffer even with OUT_OF_PLACE flag.
     */
    FFTZ_VOID *input_base_real = sol->decomp_scheme->in_real;
    FFTZ_VOID *input_base_imag = sol->decomp_scheme->in_imag;
    FFTZ_UINT8 need_input_copy =
        (sol->decomp_scheme->in_real == sol->decomp_scheme->out_real);

    if (need_input_copy)
    {
        /* Input and output share the same buffer. Copy the input
         * into a separate buffer(sr_input_copy) before sub-FFTs overwrite it.
         */
        if (sol->dft_bufs->sr->input_copy == NULL)
        {
            /* Buffer was not allocated because the plan was
             * created as out-of-place, but execute got same in/out. */
            AOCLFFTZ_LOG(
                INFO, global_logger_mode,
                "SR input copy buffer not allocated for in-place execution");
            return SOLVER_FAILURE;
        }
        FFTZ_INTP buffer_size = sol->dft_bufs->sr->input_copy_size;
        memcpy(sol->dft_bufs->sr->input_copy,
               sol->decomp_scheme->in_real, buffer_size);

        input_base_real = sol->dft_bufs->sr->input_copy;
        input_base_imag =
            MOVE_ADDR(sol->dft_bufs->sr->input_copy, dt_bytes);
    }

    /* Execute even sub-problem */
    even_sol->decomp_scheme->in_real  = input_base_real;
    even_sol->decomp_scheme->in_imag  = input_base_imag;
    even_sol->decomp_scheme->out_real = sol->decomp_scheme->out_real;
    even_sol->decomp_scheme->out_imag = sol->decomp_scheme->out_imag;
    even_sol->decomp_scheme->flags = sol->decomp_scheme->flags;
    SET_OUTOFPLACE(even_sol->decomp_scheme->flags);

    FFTZ_INT32 status = even_sol->solver->execute_solver(even_sol);
    if (status != SOLVER_SUCCESS)
    {
        return status;
    }

    /* Execute odd1 sub-problem */
    odd1_sol->decomp_scheme->in_real  =
        MOVE_ADDR(input_base_real, 1 * in_stride * elem_size);
    odd1_sol->decomp_scheme->in_imag  =
        MOVE_ADDR(input_base_imag, 1 * in_stride * elem_size);
    odd1_sol->decomp_scheme->out_real =
        MOVE_ADDR(sol->decomp_scheme->out_real, n2 * out_stride * elem_size);
    odd1_sol->decomp_scheme->out_imag =
        MOVE_ADDR(sol->decomp_scheme->out_imag, n2 * out_stride * elem_size);
    odd1_sol->decomp_scheme->flags = sol->decomp_scheme->flags;
    SET_OUTOFPLACE(odd1_sol->decomp_scheme->flags);

    status = odd1_sol->solver->execute_solver(odd1_sol);
    if (status != SOLVER_SUCCESS)
    {
        return status;
    }

    /* Execute odd3 sub-problem */
    odd3_sol->decomp_scheme->in_real  =
        MOVE_ADDR(input_base_real, 3 * in_stride * elem_size);
    odd3_sol->decomp_scheme->in_imag  =
        MOVE_ADDR(input_base_imag, 3 * in_stride * elem_size);
    odd3_sol->decomp_scheme->out_real =
        MOVE_ADDR(sol->decomp_scheme->out_real,
                  (n2 + n4) * out_stride * elem_size);
    odd3_sol->decomp_scheme->out_imag =
        MOVE_ADDR(sol->decomp_scheme->out_imag,
                  (n2 + n4) * out_stride * elem_size);
    odd3_sol->decomp_scheme->flags = sol->decomp_scheme->flags;
    SET_OUTOFPLACE(odd3_sol->decomp_scheme->flags);

    status = odd3_sol->solver->execute_solver(odd3_sol);
    if (status != SOLVER_SUCCESS)
    {
        return status;
    }

    /* Combine results with butterfly */
    status = split_radix_butterfly(sol);

    AOCLFFTZ_LOG(TRACE, global_logger_mode, "Exit");
    return status;
}

/**
 * @brief Split-radix butterfly combining E, O1, O3 into final output.
 *
 * Output buffer holds [Even(N/2) | Odd1(N/4) | Odd3(N/4)] after recursion.
 * Interleaved format: [real0, imag0, real1, imag1, ...]
 *
 * TODO: Vectorize the butterfly loop using SIMD intrinsics.
 */
static FFTZ_INT32 split_radix_butterfly(aoclfftz_solution_t *sol)
{
    FFTZ_INTP n = sol->decomp_scheme->dims[0].n;
    FFTZ_INTP n2 = n >> 1;  /* n/2 */
    FFTZ_INTP n4 = n >> 2;  /* n/4 */

    FFTZ_UINT32 precision = DT_PRECISION_FLAG(sol->decomp_scheme->flags);
    FFTZ_UINT8 direction = FFT_DIR(sol->decomp_scheme->flags);

    FFTZ_INTP out_stride = sol->decomp_scheme->dims[0].out_stride;

    /* Twiddles must be available */
    if (!sol->twiddle || !sol->twiddle->TW)
    {
        AOCLFFTZ_LOG(INFO, global_logger_mode,
                    "SR butterfly: twiddles not available");
        return SOLVER_FAILURE;
    }

    /* +1 for forward, -1 for backward (flips the j*v terms) */
    FFTZ_INTP dir_sign = (direction == FORWARD_FFT_DIR) ? 1 : -1;

    /* Stride in elements (interleaved format: real, imag pairs) */
    FFTZ_INTP stride_elems = out_stride * DATA_STRIDE;

    /*
     * Output buffer layout after the three recursive sub-FFTs.
     * Example for N=8 (stride_elems=2 for interleaved real/imag):
     *
     *   Index: 0  2      4  6 |  8  10    | 12 14
     *          E0 E1    E2 E3 | O1_0 O1_1 | O3_0 O3_1
     *          |        |       |            |
     *          0   even_mid  odd1_start   odd3_start
     *                (4)       (8)          (12)
     *
     * even_mid   – offset to E[k+N/4]  (= N/4 * stride)
     * odd1_start – offset to O1[0]     (= N/2 * stride)
     * odd3_start – offset to O3[0]     (= 3N/4 * stride)
     * k_pos      – base index of element k in  output buffer (= k * stride)
     */
    FFTZ_INTP even_mid    = n4 * stride_elems;
    FFTZ_INTP odd1_start  = n2 * stride_elems;
    FFTZ_INTP odd3_start  = odd1_start + even_mid;

    if (precision == DT_FLOAT)
    {
        FFTZ_FLOAT *data = (FFTZ_FLOAT *)sol->decomp_scheme->out_real;
        aoclfftz_complex_f_t *tw = (aoclfftz_complex_f_t *)sol->twiddle->TW;
        FFTZ_FLOAT fdir = (FFTZ_FLOAT)dir_sign;

        /*
         * k=0 special case: W^0 = 1, so twiddle multiply is identity.
         * t1 = O1[0], t3 = O3[0] (no multiplication needed)
         */
        {
            /* Load even sub-problem: E[0], E[N/4] */
            FFTZ_FLOAT e0_r = data[0];
            FFTZ_FLOAT e0_i = data[1];
            FFTZ_FLOAT en4_r = data[even_mid];
            FFTZ_FLOAT en4_i = data[even_mid + 1];

            /* Load odd sub-problems: O1[0], O3[0] */
            FFTZ_FLOAT o1_r = data[odd1_start];
            FFTZ_FLOAT o1_i = data[odd1_start + 1];
            FFTZ_FLOAT o3_r = data[odd3_start];
            FFTZ_FLOAT o3_i = data[odd3_start + 1];

            /* u = O1 + O3, v = O1 - O3 */
            FFTZ_FLOAT u_r = o1_r + o3_r;
            FFTZ_FLOAT u_i = o1_i + o3_i;
            FFTZ_FLOAT v_r = o1_r - o3_r;
            FFTZ_FLOAT v_i = o1_i - o3_i;

            /* X[0] = E[0] + u, X[N/2] = E[0] - u */
            data[0] = e0_r + u_r;
            data[1] = e0_i + u_i;
            data[odd1_start] = e0_r - u_r;
            data[odd1_start + 1] = e0_i - u_i;

            /* X[N/4] = E[N/4] + j*dir*v, X[3N/4] = E[N/4] - j*dir*v */
            data[even_mid] = en4_r + fdir * v_i;
            data[even_mid + 1] = en4_i - fdir * v_r;
            data[odd3_start] = en4_r - fdir * v_i;
            data[odd3_start + 1] = en4_i + fdir * v_r;
        }

        /* Main loop: k=1 to N/4-1 with twiddle factors */
        for (FFTZ_INTP k = 1; k < n4; k++)
        {
            FFTZ_INTP k_pos = k * stride_elems;

            /* Load even sub-problem: E[k], E[k+N/4] */
            FFTZ_FLOAT e_k_r = data[k_pos];
            FFTZ_FLOAT e_k_i = data[k_pos + 1];
            FFTZ_FLOAT e_kn4_r = data[k_pos + even_mid];
            FFTZ_FLOAT e_kn4_i = data[k_pos + even_mid + 1];

            /* Load odd sub-problems: O1[k], O3[k] */
            FFTZ_FLOAT o1_r = data[k_pos + odd1_start];
            FFTZ_FLOAT o1_i = data[k_pos + odd1_start + 1];
            FFTZ_FLOAT o3_r = data[k_pos + odd3_start];
            FFTZ_FLOAT o3_i = data[k_pos + odd3_start + 1];

            /* Get twiddles: W_N^k and W_N^(3k) stored as pairs */
            FFTZ_INTP tw_idx = k << 1;
            FFTZ_FLOAT w1_cos = tw[tw_idx].real;
            FFTZ_FLOAT w1_sin = tw[tw_idx].imag * fdir;
            FFTZ_FLOAT w3_cos = tw[tw_idx + 1].real;
            FFTZ_FLOAT w3_sin = tw[tw_idx + 1].imag * fdir;

            /* Twiddle multiply: t1 = W^k * O1[k], t3 = W^(3k) * O3[k] */
            FFTZ_FLOAT t1_r = w1_cos * o1_r - w1_sin * o1_i;
            FFTZ_FLOAT t1_i = w1_sin * o1_r + w1_cos * o1_i;
            FFTZ_FLOAT t3_r = w3_cos * o3_r - w3_sin * o3_i;
            FFTZ_FLOAT t3_i = w3_sin * o3_r + w3_cos * o3_i;

            /* u = t1 + t3, v = t1 - t3 */
            FFTZ_FLOAT u_r = t1_r + t3_r;
            FFTZ_FLOAT u_i = t1_i + t3_i;
            FFTZ_FLOAT v_r = t1_r - t3_r;
            FFTZ_FLOAT v_i = t1_i - t3_i;

            /* X[k] = E[k] + u, X[k+N/2] = E[k] - u */
            data[k_pos] = e_k_r + u_r;
            data[k_pos + 1] = e_k_i + u_i;
            data[k_pos + odd1_start] = e_k_r - u_r;
            data[k_pos + odd1_start + 1] = e_k_i - u_i;

            /* X[k+N/4] = E[k+N/4] + j*dir*v, X[k+3N/4] = E[k+N/4] - j*dir*v */
            data[k_pos + even_mid] = e_kn4_r + fdir * v_i;
            data[k_pos + even_mid + 1] = e_kn4_i - fdir * v_r;
            data[k_pos + odd3_start] = e_kn4_r - fdir * v_i;
            data[k_pos + odd3_start + 1] = e_kn4_i + fdir * v_r;
        }
    }
    else /* FFTZ_DOUBLE precision */
    {
        FFTZ_DOUBLE *data = (FFTZ_DOUBLE *)sol->decomp_scheme->out_real;
        aoclfftz_complex_d_t *tw = (aoclfftz_complex_d_t *)sol->twiddle->TW;
        FFTZ_DOUBLE ddir = (FFTZ_DOUBLE)dir_sign;

        /*
         * k=0 special case: W^0 = 1, so twiddle multiply is identity.
         */
        {
            /* Load even sub-problem: E[0], E[N/4] */
            FFTZ_DOUBLE e0_r = data[0];
            FFTZ_DOUBLE e0_i = data[1];
            FFTZ_DOUBLE en4_r = data[even_mid];
            FFTZ_DOUBLE en4_i = data[even_mid + 1];

            /* Load odd sub-problems: O1[0], O3[0] */
            FFTZ_DOUBLE o1_r = data[odd1_start];
            FFTZ_DOUBLE o1_i = data[odd1_start + 1];
            FFTZ_DOUBLE o3_r = data[odd3_start];
            FFTZ_DOUBLE o3_i = data[odd3_start + 1];

            /* u = O1 + O3, v = O1 - O3 */
            FFTZ_DOUBLE u_r = o1_r + o3_r;
            FFTZ_DOUBLE u_i = o1_i + o3_i;
            FFTZ_DOUBLE v_r = o1_r - o3_r;
            FFTZ_DOUBLE v_i = o1_i - o3_i;

            /* X[0] = E[0] + u, X[N/2] = E[0] - u */
            data[0] = e0_r + u_r;
            data[1] = e0_i + u_i;
            data[odd1_start] = e0_r - u_r;
            data[odd1_start + 1] = e0_i - u_i;

            /* X[N/4] = E[N/4] + j*dir*v, X[3N/4] = E[N/4] - j*dir*v */
            data[even_mid] = en4_r + ddir * v_i;
            data[even_mid + 1] = en4_i - ddir * v_r;
            data[odd3_start] = en4_r - ddir * v_i;
            data[odd3_start + 1] = en4_i + ddir * v_r;
        }

        /* Main loop: k=1 to N/4-1 with twiddle factors */
        for (FFTZ_INTP k = 1; k < n4; k++)
        {
            FFTZ_INTP k_pos = k * stride_elems;

            /* Load even sub-problem: E[k], E[k+N/4] */
            FFTZ_DOUBLE e_k_r = data[k_pos];
            FFTZ_DOUBLE e_k_i = data[k_pos + 1];
            FFTZ_DOUBLE e_kn4_r = data[k_pos + even_mid];
            FFTZ_DOUBLE e_kn4_i = data[k_pos + even_mid + 1];

            /* Load odd sub-problems: O1[k], O3[k] */
            FFTZ_DOUBLE o1_r = data[k_pos + odd1_start];
            FFTZ_DOUBLE o1_i = data[k_pos + odd1_start + 1];
            FFTZ_DOUBLE o3_r = data[k_pos + odd3_start];
            FFTZ_DOUBLE o3_i = data[k_pos + odd3_start + 1];

            /* Get twiddles: W_N^k and W_N^(3k) stored as pairs */
            FFTZ_INTP tw_idx = k << 1;
            FFTZ_DOUBLE w1_cos = tw[tw_idx].real;
            FFTZ_DOUBLE w1_sin = tw[tw_idx].imag * ddir;
            FFTZ_DOUBLE w3_cos = tw[tw_idx + 1].real;
            FFTZ_DOUBLE w3_sin = tw[tw_idx + 1].imag * ddir;

            /* Twiddle multiply: t1 = W^k * O1[k], t3 = W^(3k) * O3[k] */
            FFTZ_DOUBLE t1_r = w1_cos * o1_r - w1_sin * o1_i;
            FFTZ_DOUBLE t1_i = w1_sin * o1_r + w1_cos * o1_i;
            FFTZ_DOUBLE t3_r = w3_cos * o3_r - w3_sin * o3_i;
            FFTZ_DOUBLE t3_i = w3_sin * o3_r + w3_cos * o3_i;

            /* u = t1 + t3, v = t1 - t3 */
            FFTZ_DOUBLE u_r = t1_r + t3_r;
            FFTZ_DOUBLE u_i = t1_i + t3_i;
            FFTZ_DOUBLE v_r = t1_r - t3_r;
            FFTZ_DOUBLE v_i = t1_i - t3_i;

            /* X[k] = E[k] + u, X[k+N/2] = E[k] - u */
            data[k_pos] = e_k_r + u_r;
            data[k_pos + 1] = e_k_i + u_i;
            data[k_pos + odd1_start] = e_k_r - u_r;
            data[k_pos + odd1_start + 1] = e_k_i - u_i;

            /* X[k+N/4] = E[k+N/4] + j*dir*v, X[k+3N/4] = E[k+N/4] - j*dir*v */
            data[k_pos + even_mid] = e_kn4_r + ddir * v_i;
            data[k_pos + even_mid + 1] = e_kn4_i - ddir * v_r;
            data[k_pos + odd3_start] = e_kn4_r - ddir * v_i;
            data[k_pos + odd3_start + 1] = e_kn4_i + ddir * v_r;
        }
    }

    return SOLVER_SUCCESS;
}

dft_solver_ register_execute_sr_solver(FFTZ_VOID)
{
    return execute_sr_solver;
}
