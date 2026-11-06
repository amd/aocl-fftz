// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file rfft2c.c
 *
 *  @brief Radix-2 r2hc Real-FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-2 real-to-halfcomplex implementations
 *  using scalar operations for single-precision and double-precision inputs.
 *
 *  @author Srirammaswamy Srinivasan
 *  @author Prasandh Sankarankutty
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 0, 2, 4, 0, 0},
                                                     {0, 0, 2, 4, 0, 0}};

ops_cycles_t get_ops_cnt_r2hc_rfft2c(FFTZ_UINT8 precision, FFTZ_UINT8 direction)
{
    if (precision == DT_FLOAT)
    {
        return ops_cnt[0];
    }
    else
    {
        return ops_cnt[1];
    }
}

FFTZ_VOID r2hc_rfft2c_fp32(FFTZ_VOID *in_real, FFTZ_VOID *in_complex,
                           FFTZ_VOID *out_real, FFTZ_VOID *out_complex,
                           FFTZ_INTP n, aoclfftz_strides_t *strides,
                           FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FFTZ_FLOAT *in_r = (FFTZ_FLOAT *)in_real;
    FFTZ_FLOAT *out_r = (FFTZ_FLOAT *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_dc_nyq_stride = strides->v_in_sym_stride;
    FFTZ_INTP v_out_dc_nyq_stride = strides->v_out_sym_stride;

    FFTZ_INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        FFTZ_FLOAT v0, v1;
        // Input point 1: x(0)
        v0 = *in_r;
        // Input point 2: x(1)
        v1 = in_r[in_strides[1]];
        // Output point 1: X(0)
        *out_r = v0 + v1;
        // Output point 2: X(0)
        out_r[out_strides[1]] = v0 - v1;
        in_r = in_r + v_in_dc_nyq_stride;
        out_r = out_r + v_out_dc_nyq_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

FFTZ_VOID r2hc_rfft2c_fp64(FFTZ_VOID *in_real, FFTZ_VOID *in_complex,
                           FFTZ_VOID *out_real, FFTZ_VOID *out_complex,
                           FFTZ_INTP n, aoclfftz_strides_t *strides,
                           FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FFTZ_DOUBLE *in_r = (FFTZ_DOUBLE *)in_real;
    FFTZ_DOUBLE *out_r = (FFTZ_DOUBLE *)out_real;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_dc_nyq_stride = strides->v_in_sym_stride;
    FFTZ_INTP v_out_dc_nyq_stride = strides->v_out_sym_stride;

    FFTZ_INTP cnt;

    for (cnt = 0; cnt < n; cnt++)
    {
        FFTZ_DOUBLE v0, v1;
        // Input point 1: x(0)
        v0 = *in_r;
        // Input point 2: x(1)
        v1 = in_r[in_strides[1]];
        // Output point 1: X(0)
        *out_r = v0 + v1;
        // Output point 2: X(0)
        out_r[out_strides[1]] = v0 - v1;
        in_r = in_r + v_in_dc_nyq_stride;
        out_r = out_r + v_out_dc_nyq_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_r2hc_rfft2c(FFTZ_UINT8 precision, FFTZ_UINT8 direction)
{

    if (precision == DT_FLOAT)
    {
        return r2hc_rfft2c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return r2hc_rfft2c_fp64;
    }
    else
    {
        return NULL;
    }
}
