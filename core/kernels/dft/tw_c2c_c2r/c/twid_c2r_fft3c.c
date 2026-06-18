// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_c2r_fft3c.c
 *
 *  @brief C2R fused twiddle (conjugate input + twiddle output) Radix-3 FFT
 * kernel with scalar operations in C
 *
 *  This file contains the DIT twiddle radix-3 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 */

#include "core/kernels/kernel.h"

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 12, 16, 16, 0, 0},
                                                     {0, 12, 16, 16, 0, 0}};

ops_cycles_t get_ops_cnt_twid_c2r_fft3c(UINT8 precision, UINT8 direction)
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

static VOID twid_c2r_fft3c_fp32(VOID *in_real, VOID *in_imag, VOID *out_real,
                                VOID *out_imag, INTP n,
                                aoclfftz_strides_t *strides, VOID *twd,
                                UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FLOAT CRTM_3_1 = +0.500000000000000000000000000000000000000000000f;
    const FLOAT CRTM_3_2 = +0.866025403784438646763723170752936183471402627f;

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

    in_h1_r = (FLOAT *)in_imag;
    in_h2_r = in_h1_r;
    in_h1_i = (FLOAT *)in_real;
    in_h2_i = in_h1_i;
    out_h1_r = (FLOAT *)out_imag;
    out_h2_r = out_h1_r;
    out_h1_i = (FLOAT *)out_real;
    out_h2_i = out_h1_i;

    for (cnt = 0; cnt < n; cnt++)
    {
        FLOAT tv1r, tv1i, tv2r, tv2i, tv3r, tv3i, avrr, avri, avir, avii;

        // Input point 1: x(0)
        FLOAT v1r = *in_h1_r;
        FLOAT v1i = *in_h1_i;
        // Input point 2: x(1)
        FLOAT v2r_t = in_h1_r[in_strides[1]];
        FLOAT v2i_t = in_h1_i[in_strides[1]];
        FLOAT v2r = v2r_t;
        FLOAT v2i = v2i_t;
        // Input point 2: x(1)
        FLOAT v3r_t = in_h2_r[in_strides[2]];
        FLOAT v3i_t = in_h2_i[in_strides[2]];
        v3r_t = -v3r_t;
        FLOAT v3r = v3r_t;
        FLOAT v3i = v3i_t;

        avrr = v2r + v3r;
        avri = v3i - v2i;
        avir = v3r - v2r;
        avii = v2i + v3i;

        tv1r = CRTM_3_1 * avrr;
        tv1i = CRTM_3_2 * avri;
        tv2r = CRTM_3_2 * avir;
        tv2i = CRTM_3_1 * avii;

        // Output point 1: X(0)
        *out_h1_r = v1r + avrr;
        *out_h1_i = v1i + avii;

        // Output point 2: X(1)
        tv3r = v1r - tv1r;
        tv3i = v1i - tv2i;
        {
            UINTP _twa = DATA_STRIDE * (1 * cols + cnt * load_multi_cols);
            FLOAT _twr = tw[_twa];
            FLOAT _twi = tw[1 + _twa];
            FLOAT _or_1 = tv3r - tv1i;
            FLOAT _oi_1 = tv3i + tv2r;
            out_h1_r[out_strides[1]] = _or_1 * _twr + _oi_1 * _twi;
            out_h1_i[out_strides[1]] = _oi_1 * _twr - _or_1 * _twi;
        }

        // Output point 3: X(2)
        FLOAT _or_2 = tv3r + tv1i;
        {
            UINTP _twa = DATA_STRIDE * (2 * cols + cnt * load_multi_cols);
            FLOAT _twr = tw[_twa];
            FLOAT _twi = tw[1 + _twa];
            FLOAT _oi = tv3i - tv2r;
            out_h2_r[out_strides[2]] = _or_2 * _twr + _oi * _twi;
            out_h2_i[out_strides[2]] = _oi * _twr - _or_2 * _twi;
        }

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

static VOID twid_c2r_fft3c_fp64(VOID *in_real, VOID *in_imag, VOID *out_real,
                                VOID *out_imag, INTP n,
                                aoclfftz_strides_t *strides, VOID *twd,
                                UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const DOUBLE CRTM_3_1 = +0.500000000000000000000000000000000000000000000;
    const DOUBLE CRTM_3_2 = +0.866025403784438646763723170752936183471402627;

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

    in_h1_r = (DOUBLE *)in_imag;
    in_h2_r = in_h1_r;
    in_h1_i = (DOUBLE *)in_real;
    in_h2_i = in_h1_i;
    out_h1_r = (DOUBLE *)out_imag;
    out_h2_r = out_h1_r;
    out_h1_i = (DOUBLE *)out_real;
    out_h2_i = out_h1_i;

    for (cnt = 0; cnt < n; cnt++)
    {
        DOUBLE tv1r, tv1i, tv2r, tv2i, tv3r, tv3i, avrr, avri, avir, avii;

        // Input point 1: x(0)
        DOUBLE v1r = *in_h1_r;
        DOUBLE v1i = *in_h1_i;
        // Input point 2: x(1)
        DOUBLE v2r_t = in_h1_r[in_strides[1]];
        DOUBLE v2i_t = in_h1_i[in_strides[1]];
        DOUBLE v2r = v2r_t;
        DOUBLE v2i = v2i_t;
        // Input point 2: x(1)
        DOUBLE v3r_t = in_h2_r[in_strides[2]];
        DOUBLE v3i_t = in_h2_i[in_strides[2]];
        v3r_t = -v3r_t;
        DOUBLE v3r = v3r_t;
        DOUBLE v3i = v3i_t;

        avrr = v2r + v3r;
        avri = v3i - v2i;
        avir = v3r - v2r;
        avii = v2i + v3i;

        tv1r = CRTM_3_1 * avrr;
        tv1i = CRTM_3_2 * avri;
        tv2r = CRTM_3_2 * avir;
        tv2i = CRTM_3_1 * avii;

        // Output point 1: X(0)
        *out_h1_r = v1r + avrr;
        *out_h1_i = v1i + avii;

        // Output point 2: X(1)
        tv3r = v1r - tv1r;
        tv3i = v1i - tv2i;
        {
            UINTP _twa = DATA_STRIDE * (1 * cols + cnt * load_multi_cols);
            DOUBLE _twr = tw[_twa];
            DOUBLE _twi = tw[1 + _twa];
            DOUBLE _or_1 = tv3r - tv1i;
            DOUBLE _oi_1 = tv3i + tv2r;
            out_h1_r[out_strides[1]] = _or_1 * _twr + _oi_1 * _twi;
            out_h1_i[out_strides[1]] = _oi_1 * _twr - _or_1 * _twi;
        }

        // Output point 3: X(2)
        DOUBLE _or_2 = tv3r + tv1i;
        {
            UINTP _twa = DATA_STRIDE * (2 * cols + cnt * load_multi_cols);
            DOUBLE _twr = tw[_twa];
            DOUBLE _twi = tw[1 + _twa];
            DOUBLE _oi = tv3i - tv2r;
            out_h2_r[out_strides[2]] = _or_2 * _twr + _oi * _twi;
            out_h2_i[out_strides[2]] = _oi * _twr - _or_2 * _twi;
        }

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

kfft_ register_kernel_twid_c2r_fft3c(UINT8 precision,
                                     UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return twid_c2r_fft3c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return twid_c2r_fft3c_fp64;
    }
    else
    {
        return NULL;
    }
}
