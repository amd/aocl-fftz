// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_fwd_fft2c.c
 *
 *  @brief Forward-only twiddle Radix-2 FFT kernel with scalar operations in C
 *
 *  This file contains the DIT twiddle radix-2 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 4, 6, 10, 0, 0},
                                                     {0, 4, 6, 10, 0, 0}};

ops_cycles_t get_ops_cnt_twid_fwd_fft2c(UINT8 precision, UINT8 direction)
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

static VOID twid_fwd_fft2c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                                VOID *out_imag, INTP n,
                                aoclfftz_strides_t *strides, VOID *twd,
                                UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FLOAT *in_h1_r, *in_h2_r, *in_h1_i, *in_h2_i, *out_h1_r, *out_h2_r,
        *out_h1_i, *out_h2_i;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_in_h2_stride = strides->v_in_h2_stride;
    INTP v_out_stride = strides->v_out_stride;
    INTP v_out_h2_stride = strides->v_out_h2_stride;
    INTP cnt;

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    FLOAT *tw = (FLOAT *)(tws->TW);
    UINTP cols = tws->cols;
    UINTP load_multi_cols = tws->load_multi_cols;
    FLOAT twr, twi;

    in_h1_r = (FLOAT *)in_real;
    in_h2_r = in_h1_r;
    in_h1_i = (FLOAT *)in_imag;
    in_h2_i = in_h1_i;
    out_h1_r = (FLOAT *)out_real;
    out_h2_r = out_h1_r;
    out_h1_i = (FLOAT *)out_imag;
    out_h2_i = out_h1_i;

    for (cnt = 0; cnt < n; cnt++)
    {
        // Input point 1: x(0)
        FLOAT v1r = *in_h1_r;
        FLOAT v1i = *in_h1_i;

        // Input point 2: x(1)
        FLOAT v2r_t = in_h2_r[in_strides[1]];
        FLOAT v2i_t = in_h2_i[in_strides[1]];

        UINTP twid_addr2 = DATA_STRIDE * (1 * cols + cnt * load_multi_cols);
        twr = tw[twid_addr2];
        twi = tw[1 + twid_addr2];

        FLOAT v2r = v2r_t * twr - v2i_t * twi;
        FLOAT v2i = v2r_t * twi + v2i_t * twr;

        // Output point 1: X(0)
        *out_h1_r = v1r + v2r;
        *out_h1_i = v1i + v2i;

        // Output point 2: X(1)
        out_h2_r[out_strides[1]] = v1r - v2r;
        out_h2_i[out_strides[1]] = v1i - v2i;

        in_h1_r = in_h1_r + v_in_stride;
        in_h2_r = in_h2_r + v_in_h2_stride;
        in_h1_i = in_h1_i + v_in_stride;
        in_h2_i = in_h2_i + v_in_h2_stride;
        out_h1_r = out_h1_r + v_out_stride;
        out_h2_r = out_h2_r + v_out_h2_stride;
        out_h1_i = out_h1_i + v_out_stride;
        out_h2_i = out_h2_i + v_out_h2_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static VOID twid_fwd_fft2c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                                VOID *out_imag, INTP n,
                                aoclfftz_strides_t *strides, VOID *twd,
                                UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    DOUBLE *in_h1_r, *in_h2_r, *in_h1_i, *in_h2_i, *out_h1_r, *out_h2_r,
        *out_h1_i, *out_h2_i;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile INTP *in_strides = strides->in_strides;
    volatile INTP *out_strides = strides->out_strides;
#else
    INTP *in_strides = strides->in_strides;
    INTP *out_strides = strides->out_strides;
#endif
    INTP v_in_stride = strides->v_in_stride;
    INTP v_in_h2_stride = strides->v_in_h2_stride;
    INTP v_out_stride = strides->v_out_stride;
    INTP v_out_h2_stride = strides->v_out_h2_stride;
    INTP cnt;

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    DOUBLE *tw = (DOUBLE *)(tws->TW);
    UINTP cols = tws->cols;
    UINTP load_multi_cols = tws->load_multi_cols;
    DOUBLE twr, twi;

    in_h1_r = (DOUBLE *)in_real;
    in_h2_r = in_h1_r;
    in_h1_i = (DOUBLE *)in_imag;
    in_h2_i = in_h1_i;
    out_h1_r = (DOUBLE *)out_real;
    out_h2_r = out_h1_r;
    out_h1_i = (DOUBLE *)out_imag;
    out_h2_i = out_h1_i;

    for (cnt = 0; cnt < n; cnt++)
    {
        // Input point 1: x(0)
        DOUBLE v1r = *in_h1_r;
        DOUBLE v1i = *in_h1_i;
        // Input point 2: x(1)
        DOUBLE v2r_t = in_h2_r[in_strides[1]];
        DOUBLE v2i_t = in_h2_i[in_strides[1]];
        UINTP twid_addr2 = DATA_STRIDE * (1 * cols + cnt * load_multi_cols);
        twr = tw[twid_addr2];
        twi = tw[1 + twid_addr2];
        DOUBLE v2r = v2r_t * twr - v2i_t * twi;
        DOUBLE v2i = v2r_t * twi + v2i_t * twr;

        // Output point 1: X(0)
        *out_h1_r = v1r + v2r;
        *out_h1_i = v1i + v2i;
        // Output point 2: X(0)
        out_h2_r[out_strides[1]] = v1r - v2r;
        out_h2_i[out_strides[1]] = v1i - v2i;
        in_h1_r = in_h1_r + v_in_stride;
        in_h2_r = in_h2_r + v_in_h2_stride;
        in_h1_i = in_h1_i + v_in_stride;
        in_h2_i = in_h2_i + v_in_h2_stride;
        out_h1_r = out_h1_r + v_out_stride;
        out_h2_r = out_h2_r + v_out_h2_stride;
        out_h1_i = out_h1_i + v_out_stride;
        out_h2_i = out_h2_i + v_out_h2_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_twid_fwd_fft2c(UINT8 precision,
                                     UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return twid_fwd_fft2c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return twid_fwd_fft2c_fp64;
    }
    else
    {
        return NULL;
    }
}
