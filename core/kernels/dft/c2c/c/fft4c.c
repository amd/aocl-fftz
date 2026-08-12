// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file fft4c.c
 *
 *  @brief Radix-4 FFT kernel with scalar operations in C
 *
 *  This file contains the DIT radix-4 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author S. Biplab Raut
 *  @author Varun Sanjay
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 0, 16, 16, 0, 0},
                                                     {0, 0, 16, 16, 0, 0}};

ops_cycles_t get_ops_cnt_fft4c(FFTZ_UINT8 precision, FFTZ_UINT8 direction)
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

static FFTZ_VOID fft4c_fp64(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                            FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                            FFTZ_INTP n, aoclfftz_strides_t *strides,
                            FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    FFTZ_DOUBLE *in_r, *in_i, *out_r, *out_i;
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

    if (flag) // non-zero flag indicates that the fft is inverse
    {
        in_r = (FFTZ_DOUBLE *)in_imag;
        in_i = (FFTZ_DOUBLE *)in_real;
        out_r = (FFTZ_DOUBLE *)out_imag;
        out_i = (FFTZ_DOUBLE *)out_real;
    }
    else
    {
        in_r = (FFTZ_DOUBLE *)in_real;
        in_i = (FFTZ_DOUBLE *)in_imag;
        out_r = (FFTZ_DOUBLE *)out_real;
        out_i = (FFTZ_DOUBLE *)out_imag;
    }

    for (cnt = 0; cnt < n; cnt++)
    {
        FFTZ_DOUBLE v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, tvri, tvir, tvii,
            tvrr, v13r, v24r, v13i, v24i;

        // Input point 1: x(0)
        v1r = *in_r;
        v1i = *in_i;

        // Input point 2: x(1)
        v2r = in_r[in_strides[1]];
        v2i = in_i[in_strides[1]];

        // Input point 3: x(2)
        v3r = in_r[in_strides[2]];
        v3i = in_i[in_strides[2]];

        // Input point 4: x(3)
        v4r = in_r[in_strides[3]];
        v4i = in_i[in_strides[3]];

        v13r = v1r + v3r;
        v24r = v2r + v4r;
        v13i = v1i + v3i;
        v24i = v2i + v4i;

        // Output point 1: X(0)
        *out_r = v13r + v24r;
        *out_i = v13i + v24i;

        // Output point 2: X(1)
        tvri = v4i - v2i;
        tvir = v4r - v2r;

        tvrr = v1r - v3r;
        tvii = v1i - v3i;

        out_r[out_strides[1]] = tvrr - tvri;
        out_i[out_strides[1]] = tvii + tvir;

        // Output point 4: X(3)
        out_r[out_strides[3]] = tvrr + tvri;
        out_i[out_strides[3]] = tvii - tvir;

        // Output point 3: X(2)
        tvrr = v13r - v24r;
        tvii = v13i - v24i;

        out_r[out_strides[2]] = tvrr;
        out_i[out_strides[2]] = tvii;

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID fft4c_fp32(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                            FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                            FFTZ_INTP n, aoclfftz_strides_t *strides,
                            FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    FFTZ_FLOAT *in_r, *in_i, *out_r, *out_i;
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

    if (flag) // non-zero flag indicates that the fft is inverse
    {
        in_r = (FFTZ_FLOAT *)in_imag;
        in_i = (FFTZ_FLOAT *)in_real;
        out_r = (FFTZ_FLOAT *)out_imag;
        out_i = (FFTZ_FLOAT *)out_real;
    }
    else
    {
        in_r = (FFTZ_FLOAT *)in_real;
        in_i = (FFTZ_FLOAT *)in_imag;
        out_r = (FFTZ_FLOAT *)out_real;
        out_i = (FFTZ_FLOAT *)out_imag;
    }

    for (cnt = 0; cnt < n; cnt++)
    {
        FFTZ_FLOAT v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, tvri, tvir, tvii,
            tvrr, v13r, v24r, v13i, v24i;

        // Input point 1: x(0)
        v1r = *in_r;
        v1i = *in_i;

        // Input point 2: x(1)
        v2r = in_r[in_strides[1]];
        v2i = in_i[in_strides[1]];

        // Input point 3: x(2)
        v3r = in_r[in_strides[2]];
        v3i = in_i[in_strides[2]];

        // Input point 4: x(3)
        v4r = in_r[in_strides[3]];
        v4i = in_i[in_strides[3]];

        v13r = v1r + v3r;
        v24r = v2r + v4r;
        v13i = v1i + v3i;
        v24i = v2i + v4i;

        // Output point 1: X(0)
        *out_r = v13r + v24r;
        *out_i = v13i + v24i;

        // Output point 2: X(1)
        tvri = v4i - v2i;
        tvir = v4r - v2r;

        tvrr = v1r - v3r;
        tvii = v1i - v3i;

        out_r[out_strides[1]] = tvrr - tvri;
        out_i[out_strides[1]] = tvii + tvir;

        // Output point 4: X(3)
        out_r[out_strides[3]] = tvrr + tvri;
        out_i[out_strides[3]] = tvii - tvir;

        // Output point 3: X(2)
        tvrr = v13r - v24r;
        tvii = v13i - v24i;

        out_r[out_strides[2]] = tvrr;
        out_i[out_strides[2]] = tvii;

        in_r += v_in_stride;
        in_i += v_in_stride;
        out_r += v_out_stride;
        out_i += v_out_stride;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_fft4c(FFTZ_UINT8 precision,
                            FFTZ_UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return fft4c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return fft4c_fp64;
    }
    else
    {
        return NULL;
    }
}
