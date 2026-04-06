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

ops_cycles_t get_ops_cnt_r2hcf_rfft2c(UINT8 precision, UINT8 direction)
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

VOID r2hcf_rfft2c_fp32_fwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                           VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                           VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FLOAT *in = (FLOAT *)in_real;
    FLOAT *out = (FLOAT *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;
    INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        /* Standard DFT */
        FLOAT av0, av1;
        av0 = *in;                          // Input point 1: x(0)
        av1 = in[in_strides[2]];            // Input point 3: x(2)

        *out = av0 + av1;                   // Output point 1: X(0)
        out[out_strides[3]] = av0 - av1;    // Output point 4: X(3)

        /* Shifted DFT */
        FLOAT bv0, bv1;
        bv0 = in[in_strides[1]];            // Input point 2: x(1)
        bv1 = in[in_strides[3]];            // Input point 4: x(3)

        out[out_strides[1]] = bv0;          // Output point 2: X(1)
        out[out_strides[2]] = -bv1;         // Output point 3: X(2)

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

VOID r2hcf_rfft2c_fp32_bwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                           VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                           VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_2_1 = 2.0f;

    FLOAT *in = (FLOAT *)in_real;
    FLOAT *out = (FLOAT *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;
    INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        /* Standard DFT */
        FLOAT av0, av1;
        av0 = *in;                            // Input point 1: x(0)
        av1 = in[in_strides[3]];              // Input point 4: x(3)

        *out = av0 + av1;                     // Output point 1: X(0)
        out[out_strides[2]] = av0 - av1;      // Output point 3: X(2)

        /* Shifted DFT */
        FLOAT bv0, bv1;
        bv0 = in[in_strides[1]];              // Input point 2: x(1)
        bv1 = in[in_strides[2]];              // Input point 3: x(2)

        out[out_strides[1]] = bv0 * CRTM_2_1; // Output point 2: X(1)
        out[out_strides[3]] = -bv1 * CRTM_2_1;// Output point 4: X(3)

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

VOID r2hcf_rfft2c_fp64_fwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                           VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                           VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    DOUBLE *in = (DOUBLE *)in_real;
    DOUBLE *out = (DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;
    INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        /* Standard DFT */
        DOUBLE av0, av1;
        av0 = *in;                          // Input point 1: x(0)
        av1 = in[in_strides[2]];            // Input point 3: x(2)

        *out = av0 + av1;                   // Output point 1: X(0)
        out[out_strides[3]] = av0 - av1;    // Output point 4: X(3)

        // Shifted DFT
        DOUBLE bv0, bv1;
        bv0 = in[in_strides[1]];            // Input point 2: x(1)
        bv1 = in[in_strides[3]];            // Input point 4: x(3)

        out[out_strides[1]] = bv0;          // Output point 2: X(1)
        out[out_strides[2]] = -bv1;         // Output point 3: X(2)

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

VOID r2hcf_rfft2c_fp64_bwd(VOID *in_real, VOID *in_imag, VOID *out_real,
                           VOID *out_imag, INTP n, aoclfftz_strides_t *strides,
                           VOID *twd, UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_2_1 = 2.0;

    DOUBLE *in = (DOUBLE *)in_real;
    DOUBLE *out = (DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_out_stride = strides->v_out_stride;
    INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        /* Standard DFT */
        DOUBLE av0, av1;
        av0 = *in;                            // Input point 1: x(0)
        av1 = in[in_strides[3]];              // Input point 4: x(3)

        *out = av0 + av1;                     // Output point 1: X(0)
        out[out_strides[2]] = av0 - av1;      // Output point 3: X(2)

        // Shifted DFT
        DOUBLE bv0, bv1;
        bv0 = in[in_strides[1]];              // Input point 2: x(1)
        bv1 = in[in_strides[2]];              // Input point 3: x(2)

        out[out_strides[1]] = bv0 * CRTM_2_1; // Output point 2: X(1)
        out[out_strides[3]] = -bv1 * CRTM_2_1;// Output point 4: X(3)

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hcf_rfft2c(UINT8 precision, UINT8 direction)
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
