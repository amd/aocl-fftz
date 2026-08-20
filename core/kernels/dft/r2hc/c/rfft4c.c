// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft4c.c
 *
 *  @brief Radix-4 r2hc Real-FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-4 real-to-halfcomplex implementations
 *  using scalar operations for single-precision and double-precision inputs.
 *
 *  @author Srirammaswamy Srinivasan
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                    {{{0, 0, 6, 8, 0, 0},
                                                      {0, 2, 6, 8, 0, 0}},
                                                     {{0, 0, 6, 8, 0, 0},
                                                      {0, 2, 6, 8, 0, 0}}};

ops_cycles_t get_ops_cnt_r2hc_rfft4c(FFTZ_UINT8 precision, FFTZ_UINT8 direction)
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

static FFTZ_VOID r2hc_rfft4c_fp32_fwd(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                      FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                      FFTZ_INTP n, aoclfftz_strides_t *strides,
                                      FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FFTZ_FLOAT *in = (FFTZ_FLOAT *)in_real;
    FFTZ_FLOAT *out = (FFTZ_FLOAT *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        FFTZ_FLOAT v0, v1, v2, v3;
        FFTZ_FLOAT t0, t1;

        // Input point 1: x(0)
        v0 = *in;
        // Input point 2: x(1)
        v1 = in[in_strides[1]];
        // Input point 3: x(2)
        v2 = in[in_strides[2]];
        // Input point 4: x(3)
        v3 = in[in_strides[3]];

        t0 = v0 + v2;
        t1 = v1 + v3;

        // Output point 1: X(0)
        *out = t0 + t1;

        // Output point 2: X(1)
        out[out_strides[1]] = v0 - v2;

        // Output point 3: X(2)
        out[out_strides[2]] = v3 - v1;

        // Output point 4: X(3)
        out[out_strides[3]] = t0 - t1;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft4c_fp32_bwd(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                      FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                      FFTZ_INTP n, aoclfftz_strides_t *strides,
                                      FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_4_1 = 2.0f;

    FFTZ_FLOAT *in = (FFTZ_FLOAT *)in_real;
    FFTZ_FLOAT *out = (FFTZ_FLOAT *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        FFTZ_FLOAT v0, v1, v2, v3;
        FFTZ_FLOAT t0, t1, t2, t3;

        // Input point 1: x(0)
        v0 = *in;
        // Input point 2: x(1)
        v1 = in[in_strides[1]];
        // Input point 3: x(2)
        v2 = in[in_strides[2]];
        // Input point 4: x(3)
        v3 = in[in_strides[3]];

        t0 = v0 + v3;
        t1 = CRTM_4_1 * v1;
        t2 = v0 - v3;
        t3 = CRTM_4_1 * v2;

        // Output point 1: X(0)
        *out = t0 + t1;

        // Output point 2: X(1)
        out[out_strides[1]] = t2 - t3;

        // Output point 3: X(2)
        out[out_strides[2]] = t0 - t1;

        // Output point 4: X(3)
        out[out_strides[3]] = t2 + t3;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft4c_fp64_fwd(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                      FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                      FFTZ_INTP n, aoclfftz_strides_t *strides,
                                      FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FFTZ_DOUBLE *in = (FFTZ_DOUBLE *)in_real;
    FFTZ_DOUBLE *out = (FFTZ_DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        FFTZ_DOUBLE v0, v1, v2, v3;
        FFTZ_DOUBLE t0, t1;

        // Input point 1: x(0)
        v0 = *in;
        // Input point 2: x(1)
        v1 = in[in_strides[1]];
        // Input point 3: x(2)
        v2 = in[in_strides[2]];
        // Input point 4: x(3)
        v3 = in[in_strides[3]];

        t0 = v0 + v2;
        t1 = v1 + v3;

        // Output point 1: X(0)
        *out = t0 + t1;

        // Output point 2: X(1)
        out[out_strides[1]] = v0 - v2;

        // Output point 3: X(2)
        out[out_strides[2]] = v3 - v1;

        // Output point 4: X(3)
        out[out_strides[3]] = t0 - t1;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft4c_fp64_bwd(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                      FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                      FFTZ_INTP n, aoclfftz_strides_t *strides,
                                      FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_4_1 = 2.0;

    FFTZ_DOUBLE *in = (FFTZ_DOUBLE *)in_real;
    FFTZ_DOUBLE *out = (FFTZ_DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        FFTZ_DOUBLE v0, v1, v2, v3;
        FFTZ_DOUBLE t0, t1, t2, t3;

        // Input point 1: x(0)
        v0 = *in;
        // Input point 2: x(1)
        v1 = in[in_strides[1]];
        // Input point 3: x(2)
        v2 = in[in_strides[2]];
        // Input point 4: x(3)
        v3 = in[in_strides[3]];

        t0 = v0 + v3;
        t1 = CRTM_4_1 * v1;
        t2 = v0 - v3;
        t3 = CRTM_4_1 * v2;

        // Output point 1: X(0)
        *out = t0 + t1;

        // Output point 2: X(1)
        out[out_strides[1]] = t2 - t3;

        // Output point 3: X(2)
        out[out_strides[2]] = t0 - t1;

        // Output point 4: X(3)
        out[out_strides[3]] = t2 + t3;

        in = in + v_in_stride;
        out = out + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hc_rfft4c(FFTZ_UINT8 precision, FFTZ_UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hc_rfft4c_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft4c_fp64_fwd;
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
            return r2hc_rfft4c_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft4c_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
