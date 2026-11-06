// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft2c.c
 *
 *  @brief Radix-2 r2hc_fused Real-FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-2 real-to-halfcomplex fused of two
 *  different implementations (Standard DFT and Shifted DFT that differs in DFT
 *  weight matrix) using scalar operations for single-precision and
 *  double-precision inputs.
 *
 *  @author Srirammaswamy Srinivasan
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                    {{{0, 0, 2, 8, 0, 0},
                                                      {0, 2, 2, 8, 0, 0}},
                                                     {{0, 0, 2, 8, 0, 0},
                                                      {0, 2, 2, 8, 0, 0}}};

ops_cycles_t get_ops_cnt_r2hcf_rfft2c(FFTZ_UINT8 precision,
                                      FFTZ_UINT8 direction)
{
    if (precision == DT_FLOAT)
    {
        if (direction == FORWARD_FFT_DIR)
        {
            return ops_cnt[0][0];
        }
        else
        {
            return ops_cnt[0][1];
        }
    }
    else
    {
        if (direction == FORWARD_FFT_DIR)
        {
            return ops_cnt[1][0];
        }
        else
        {
            return ops_cnt[1][1];
        }
    }
}

FFTZ_VOID r2hcf_rfft2c_fp32_fwd(FFTZ_VOID *in_real, FFTZ_VOID *in_complex,
                                FFTZ_VOID *out_real, FFTZ_VOID *out_complex,
                                FFTZ_INTP n, aoclfftz_strides_t *strides,
                                FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FFTZ_FLOAT *in_r = (FFTZ_FLOAT *)in_real;
    FFTZ_FLOAT *out_r = (FFTZ_FLOAT *)out_real;
    FFTZ_FLOAT *out_cp = (FFTZ_FLOAT *)out_complex;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_INTP v_out_dc_nyq_stride = strides->v_out_sym_stride;
    FFTZ_INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        /* Standard DFT */
        FFTZ_FLOAT av0, av1;
        av0 = *in_r;                          // Input point 1: x(0)
        av1 = in_r[in_strides[2]];            // Input point 3: x(2)

        *out_r = av0 + av1;                   // Output point 1: X(0)
        out_r[out_strides[3]] = av0 - av1;    // Output point 4: X(3)

        /* Shifted DFT */
        FFTZ_FLOAT bv0, bv1;
        bv0 = in_r[in_strides[1]];            // Input point 2: x(1)
        bv1 = in_r[in_strides[3]];            // Input point 4: x(3)

        out_cp[out_strides[1]] = bv0;          // Output point 2: X(1)
        out_cp[out_strides[2]] = -bv1;         // Output point 3: X(2)

        in_r = in_r + v_in_stride;
        out_cp = out_cp + v_out_stride;
        out_r = out_r + v_out_dc_nyq_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

FFTZ_VOID r2hcf_rfft2c_fp32_bwd(FFTZ_VOID *in_real, FFTZ_VOID *in_complex,
                                FFTZ_VOID *out_real, FFTZ_VOID *out_complex,
                                FFTZ_INTP n, aoclfftz_strides_t *strides,
                                FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_2_1 = 2.0f;

    FFTZ_FLOAT *in_r = (FFTZ_FLOAT *)in_real;
    FFTZ_FLOAT *in_cp = (FFTZ_FLOAT *)in_complex;
    FFTZ_FLOAT *out_r = (FFTZ_FLOAT *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_INTP v_in_dc_nyq_stride = strides->v_in_sym_stride;
    FFTZ_INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        /* Standard DFT */
        FFTZ_FLOAT av0, av1;
        av0 = *in_r;                            // Input point 1: x(0)
        av1 = in_r[in_strides[3]];              // Input point 4: x(3)

        *out_r = av0 + av1;                     // Output point 1: X(0)
        out_r[out_strides[2]] = av0 - av1;      // Output point 3: X(2)

        /* Shifted DFT */
        FFTZ_FLOAT bv0, bv1;
        bv0 = in_cp[in_strides[1]];              // Input point 2: x(1)
        bv1 = in_cp[in_strides[2]];              // Input point 3: x(2)

        out_r[out_strides[1]] = bv0 * CRTM_2_1; // Output point 2: X(1)
        out_r[out_strides[3]] = -bv1 * CRTM_2_1;// Output point 4: X(3)

        in_cp = in_cp + v_in_stride;
        in_r = in_r + v_in_dc_nyq_stride;
        out_r = out_r + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

FFTZ_VOID r2hcf_rfft2c_fp64_fwd(FFTZ_VOID *in_real, FFTZ_VOID *in_complex,
                                FFTZ_VOID *out_real, FFTZ_VOID *out_complex,
                                FFTZ_INTP n, aoclfftz_strides_t *strides,
                                FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FFTZ_DOUBLE *in_r = (FFTZ_DOUBLE *)in_real;
    FFTZ_DOUBLE *out_r = (FFTZ_DOUBLE *)out_real;
    FFTZ_DOUBLE *out_cp = (FFTZ_DOUBLE *)out_complex;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_INTP v_out_dc_nyq_stride = strides->v_out_sym_stride;
    FFTZ_INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        /* Standard DFT */
        FFTZ_DOUBLE av0, av1;
        av0 = *in_r;                          // Input point 1: x(0)
        av1 = in_r[in_strides[2]];            // Input point 3: x(2)

        *out_r = av0 + av1;                   // Output point 1: X(0)
        out_r[out_strides[3]] = av0 - av1;    // Output point 4: X(3)

        // Shifted DFT
        FFTZ_DOUBLE bv0, bv1;
        bv0 = in_r[in_strides[1]];            // Input point 2: x(1)
        bv1 = in_r[in_strides[3]];            // Input point 4: x(3)

        out_cp[out_strides[1]] = bv0;          // Output point 2: X(1)
        out_cp[out_strides[2]] = -bv1;         // Output point 3: X(2)

        in_r = in_r + v_in_stride;
        out_cp = out_cp + v_out_stride;
        out_r = out_r + v_out_dc_nyq_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

FFTZ_VOID r2hcf_rfft2c_fp64_bwd(FFTZ_VOID *in_real, FFTZ_VOID *in_complex,
                                FFTZ_VOID *out_real, FFTZ_VOID *out_complex,
                                FFTZ_INTP n, aoclfftz_strides_t *strides,
                                FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_2_1 = 2.0;

    FFTZ_DOUBLE *in_r = (FFTZ_DOUBLE *)in_real;
    FFTZ_DOUBLE *in_cp = (FFTZ_DOUBLE *)in_complex;
    FFTZ_DOUBLE *out_r = (FFTZ_DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_INTP v_in_dc_nyq_stride = strides->v_in_sym_stride;
    FFTZ_INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        /* Standard DFT */
        FFTZ_DOUBLE av0, av1;
        av0 = *in_r;                            // Input point 1: x(0)
        av1 = in_r[in_strides[3]];              // Input point 4: x(3)

        *out_r = av0 + av1;                     // Output point 1: X(0)
        out_r[out_strides[2]] = av0 - av1;      // Output point 3: X(2)

        // Shifted DFT
        FFTZ_DOUBLE bv0, bv1;
        bv0 = in_cp[in_strides[1]];              // Input point 2: x(1)
        bv1 = in_cp[in_strides[2]];              // Input point 3: x(2)

        out_r[out_strides[1]] = bv0 * CRTM_2_1; // Output point 2: X(1)
        out_r[out_strides[3]] = -bv1 * CRTM_2_1;// Output point 4: X(3)

        in_cp = in_cp + v_in_stride;
        in_r = in_r + v_in_dc_nyq_stride;
        out_r = out_r + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hcf_rfft2c(FFTZ_UINT8 precision, FFTZ_UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hcf_rfft2c_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft2c_fp64_fwd;
        }
        else
        {
            return NULL;
        }
    }
    else
    {
        if (precision == DT_FLOAT)
        {
            return r2hcf_rfft2c_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hcf_rfft2c_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
