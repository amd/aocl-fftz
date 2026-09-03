// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft3c.c
 *
 *  @brief Radix-3 r2hc Real-FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-3 real-to-halfcomplex implementations
 *  using scalar operations for single-precision and double-precision inputs.
 *
 *  @author Srirammaswamy Srinivasan
 */

#include "core/kernels/kernel.h"

// Forward and backward opscount are identical for float and double
static const ops_cycles_t ops_cnt[NUM_PRECISIONS][NUM_FFT_DIRS] =
                                                    {{{0, 2, 4, 6, 0, 0},
                                                      {0, 2, 4, 6, 0, 0}},
                                                     {{0, 2, 4, 6, 0, 0},
                                                      {0, 2, 4, 6, 0, 0}}};

ops_cycles_t get_ops_cnt_r2hc_rfft3c(FFTZ_UINT8 precision, FFTZ_UINT8 direction)
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

static FFTZ_VOID r2hc_rfft3c_fp32_fwd(FFTZ_VOID *in_real,
                                      FFTZ_VOID *in_complex,
                                      FFTZ_VOID *out_real,
                                      FFTZ_VOID *out_complex, FFTZ_INTP n,
                                      aoclfftz_strides_t *strides,
                                      FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_3_1 =
        0.500000000000000000000000000000000000000000000f;
    const FFTZ_FLOAT CRTM_3_2 =
        0.866025403784438646763723170752936183471402627f;

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
        FFTZ_FLOAT v0, v1, v2;
        FFTZ_FLOAT t0, t1, t2, t3;

        // Input point 1: x(0)
        v0 = *in_r;
        // Input point 2: x(1)
        v1 = in_r[in_strides[1]];
        // Input point 3: x(2)
        v2 = in_r[in_strides[2]];

        t0 = v2 + v1;
        t1 = v2 - v1;

        t2 = CRTM_3_1 * t0;
        t3 = CRTM_3_2 * t1;

        // Output point 1: X(0)
        *out_r = v0 + t0;

        // Output point 2: X(1)
        out_cp[out_strides[1]] = v0 - t2;

        // Output point 3: X(2)
        out_cp[out_strides[2]] = t3;

        in_r = in_r + v_in_stride;
        out_cp = out_cp + v_out_stride;
        out_r = out_r + v_out_dc_nyq_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft3c_fp32_bwd(FFTZ_VOID *in_real,
                                      FFTZ_VOID *in_complex,
                                      FFTZ_VOID *out_real,
                                      FFTZ_VOID *out_complex, FFTZ_INTP n,
                                      aoclfftz_strides_t *strides,
                                      FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_3_1 =
        1.732050807568877293527446341505872366942805254f;
    const FFTZ_FLOAT CRTM_3_2 = 2.0f;

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
        FFTZ_FLOAT v0, v1, v2;
        FFTZ_FLOAT t0, t1, t2;

        // Input point 1: x(0)
        v0 = *in_r;
        // Input point 2: x(1)
        v1 = in_cp[in_strides[1]];
        // Input point 3: x(2)
        v2 = in_cp[in_strides[2]];

        t0 = v0 - v1;
        t1 = CRTM_3_1 * v2;
        t2 = CRTM_3_2 * v1;

        // Output point 1: X(0)
        *out_r = v0 + t2;

        // Output point 2: X(1)
        out_r[out_strides[1]] = t0 - t1;

        // Output point 3: X(2)
        out_r[out_strides[2]] = t0 + t1;

        in_cp = in_cp + v_in_stride;
        in_r = in_r + v_in_dc_nyq_stride;
        out_r = out_r + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft3c_fp64_fwd(FFTZ_VOID *in_real,
                                      FFTZ_VOID *in_complex,
                                      FFTZ_VOID *out_real,
                                      FFTZ_VOID *out_complex, FFTZ_INTP n,
                                      aoclfftz_strides_t *strides,
                                      FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_3_1 =
        0.500000000000000000000000000000000000000000000;
    const FFTZ_DOUBLE CRTM_3_2 =
        0.866025403784438646763723170752936183471402627;

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
        FFTZ_DOUBLE v0, v1, v2;
        FFTZ_DOUBLE t0, t1, t2, t3;

        // Input point 1: x(0)
        v0 = *in_r;
        // Input point 2: x(1)
        v1 = in_r[in_strides[1]];
        // Input point 3: x(2)
        v2 = in_r[in_strides[2]];

        t0 = v2 + v1;
        t1 = v2 - v1;

        t2 = CRTM_3_1 * t0;
        t3 = CRTM_3_2 * t1;

        // Output point 1: X(0)
        *out_r = v0 + t0;

        // Output point 2: X(1)
        out_cp[out_strides[1]] = v0 - t2;

        // Output point 3: X(2)
        out_cp[out_strides[2]] = t3;

        in_r = in_r + v_in_stride;
        out_cp = out_cp + v_out_stride;
        out_r = out_r + v_out_dc_nyq_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID r2hc_rfft3c_fp64_bwd(FFTZ_VOID *in_real,
                                      FFTZ_VOID *in_complex,
                                      FFTZ_VOID *out_real,
                                      FFTZ_VOID *out_complex, FFTZ_INTP n,
                                      aoclfftz_strides_t *strides,
                                      FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_3_1 =
        1.732050807568877293527446341505872366942805254;
    const FFTZ_DOUBLE CRTM_3_2 = 2.0;

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
        FFTZ_DOUBLE v0, v1, v2;
        FFTZ_DOUBLE t0, t1, t2;

        // Input point 1: x(0)
        v0 = *in_r;
        // Input point 2: x(1)
        v1 = in_cp[in_strides[1]];
        // Input point 3: x(2)
        v2 = in_cp[in_strides[2]];

        t0 = v0 - v1;
        t1 = CRTM_3_1 * v2;
        t2 = CRTM_3_2 * v1;

        // Output point 1: X(0)
        *out_r = v0 + t2;

        // Output point 2: X(1)
        out_r[out_strides[1]] = t0 - t1;

        // Output point 3: X(2)
        out_r[out_strides[2]] = t0 + t1;

        in_cp = in_cp + v_in_stride;
        in_r = in_r + v_in_dc_nyq_stride;
        out_r = out_r + v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hc_rfft3c(FFTZ_UINT8 precision, FFTZ_UINT8 direction)
{
    if (direction == FORWARD_FFT_DIR)
    {
        if (precision == DT_FLOAT)
        {
            return r2hc_rfft3c_fp32_fwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft3c_fp64_fwd;
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
            return r2hc_rfft3c_fp32_bwd;
        }
        else if (precision == DT_DOUBLE)
        {
            return r2hc_rfft3c_fp64_bwd;
        }
        else
        {
            return NULL;
        }
    }
}
