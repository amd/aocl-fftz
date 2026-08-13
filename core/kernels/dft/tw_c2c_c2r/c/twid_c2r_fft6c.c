// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: BSD-3-Clause

/** @file twid_c2r_fft6c.c
 *
 *  @brief C2R fused twiddle (conjugate input + twiddle output) Radix-6 FFT
 * kernel with scalar operations in C
 *
 *  This file contains the DIT twiddle radix-6 FFT implementations using scalar
 *  operations for single-precision and double-precision inputs.
 *
 *  @author Ashwin K. Godbole
 */

#include "core/kernels/kernel.h"

#define RADIX 6

static const ops_cycles_t ops_cnt[NUM_PRECISIONS] = {{0, 28, 46, 34, 0, 0},
                                                     {0, 28, 46, 34, 0, 0}};

ops_cycles_t get_ops_cnt_twid_c2r_fft6c(FFTZ_UINT8 precision,
                                        FFTZ_UINT8 direction)
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

static FFTZ_VOID twid_c2r_fft6c_fp64(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                     FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                     FFTZ_INTP n, aoclfftz_strides_t *strides,
                                     FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_DOUBLE CRTM_6_1 =
        +0.500000000000000000000000000000000000000000000;
    const FFTZ_DOUBLE CRTM_6_2 =
        +0.866025403784438646763723170752936183471402627;

    FFTZ_DOUBLE *in_h1_r, *in_h2_r, *in_h1_i, *in_h2_i, *out_h1_r, *out_h2_r,
        *out_h1_i, *out_h2_i;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_in_h2_stride = strides->v_in_h2_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_INTP v_out_h2_stride = strides->v_out_h2_stride;
    FFTZ_INTP cnt;

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    FFTZ_DOUBLE *tw = (FFTZ_DOUBLE *)(tws->TW);
    FFTZ_UINTP load_multi_cols = tws->load_multi_cols;

    in_h1_r = (FFTZ_DOUBLE *)in_real;
    in_h2_r = in_h1_r;
    in_h1_i = (FFTZ_DOUBLE *)in_imag;
    in_h2_i = in_h1_i;
    out_h1_r = (FFTZ_DOUBLE *)out_real;
    out_h2_r = out_h1_r;
    out_h1_i = (FFTZ_DOUBLE *)out_imag;
    out_h2_i = out_h1_i;

    FFTZ_DOUBLE *tw_ptr = tw;

    for (cnt = 0; cnt < n; cnt++)
    {
        FFTZ_DOUBLE v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v6r, v6i,
            av1rr, av2rr, av3rr, av4rr, av5rr, av6rr, av7rr, av8rr, tv1rr,
            tv2rr, av1ii, av2ii, av3ii, av4ii, av5ii, av6ii, av7ii, av8ii,
            tv2ii, tv1ii;
        // Input point 1: x(0)
        v1r = *in_h1_r;
        v1i = *in_h1_i;

        // Input point 2: x(1)
        FFTZ_DOUBLE v2r_t = in_h1_r[in_strides[1]];
        FFTZ_DOUBLE v2i_t = in_h1_i[in_strides[1]];
        v2r = v2r_t;
        v2i = v2i_t;

        // Input point 3: x(2)
        FFTZ_DOUBLE v3r_t = in_h1_r[in_strides[2]];
        FFTZ_DOUBLE v3i_t = in_h1_i[in_strides[2]];
        v3r = v3r_t;
        v3i = v3i_t;

        // Input point 4: x(3)
        FFTZ_DOUBLE v4r_t = in_h2_r[in_strides[3]];
        FFTZ_DOUBLE v4i_t = in_h2_i[in_strides[3]];
        v4i_t = -v4i_t;
        v4r = v4r_t;
        v4i = v4i_t;

        // Input point 5: x(4)
        FFTZ_DOUBLE v5r_t = in_h2_r[in_strides[4]];
        FFTZ_DOUBLE v5i_t = in_h2_i[in_strides[4]];
        v5i_t = -v5i_t;
        v5r = v5r_t;
        v5i = v5i_t;

        // Input point 6: x(5)
        FFTZ_DOUBLE v6r_t = in_h2_r[in_strides[5]];
        FFTZ_DOUBLE v6i_t = in_h2_i[in_strides[5]];
        v6i_t = -v6i_t;
        v6r = v6r_t;
        v6i = v6i_t;

        // Common calculations -> real
        av1rr = v1r + v4r;
        av2rr = v2r + v6r;
        av3rr = v3r + v5r;
        av4rr = v1r - v4r;
        av5rr = v2r - v6r;
        av6rr = v3r - v5r;
        av7rr = av3rr - av2rr;
        av8rr = av3rr + av2rr;

        // Common calculations -> imaginary
        av1ii = v1i + v4i;
        av2ii = v2i + v6i;
        av3ii = v3i + v5i;
        av4ii = v1i - v4i;
        av5ii = v2i - v6i;
        av6ii = v3i - v5i;
        av7ii = av3ii - av2ii;
        av8ii = av3ii + av2ii;

        // Output point 1: X(0)
        *out_h1_r = av1rr + av8rr;
        *out_h1_i = av1ii + av8ii;
        // Output point 4: X(3)
        FFTZ_DOUBLE _or_3 = av4rr + av7rr;
        {
            FFTZ_DOUBLE _twr = tw_ptr[2 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[2 * DATA_STRIDE + 1];
            FFTZ_DOUBLE _oi = av4ii + av7ii;
            out_h2_r[out_strides[3]] = _or_3 * _twr - _oi * _twi;
            out_h2_i[out_strides[3]] = _or_3 * _twi + _oi * _twr;
        }

        // Common values for X(1) && X(5)
        tv1rr = av4rr - av7rr * CRTM_6_1;
        tv1ii = (av5ii + av6ii) * CRTM_6_2;
        tv2ii = av4ii - av7ii * CRTM_6_1;
        tv2rr = (av6rr + av5rr) * CRTM_6_2;

        // Output point 2: X(1)
        {
            FFTZ_DOUBLE _twr = tw_ptr[0];
            FFTZ_DOUBLE _twi = tw_ptr[1];
            FFTZ_DOUBLE _or_1 = tv1rr - tv1ii;
            FFTZ_DOUBLE _oi_1 = tv2ii + tv2rr;
            out_h1_r[out_strides[1]] = _or_1 * _twr - _oi_1 * _twi;
            out_h1_i[out_strides[1]] = _or_1 * _twi + _oi_1 * _twr;
        }
        // Output point 6: X(5)
        FFTZ_DOUBLE _or_5 = tv1rr + tv1ii;
        {
            FFTZ_DOUBLE _twr = tw_ptr[4 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[4 * DATA_STRIDE + 1];
            FFTZ_DOUBLE _oi = tv2ii - tv2rr;
            out_h2_r[out_strides[5]] = _or_5 * _twr - _oi * _twi;
            out_h2_i[out_strides[5]] = _or_5 * _twi + _oi * _twr;
        }

        // Common values for X(2) && X(4)
        tv1rr = av1rr - av8rr * CRTM_6_1;
        tv1ii = (av5ii - av6ii) * CRTM_6_2;
        tv2ii = av1ii - av8ii * CRTM_6_1;
        tv2rr = (av6rr - av5rr) * CRTM_6_2;

        // Output point 3: X(2)
        {
            FFTZ_DOUBLE _twr = tw_ptr[DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[DATA_STRIDE + 1];
            FFTZ_DOUBLE _or_2 = tv1rr - tv1ii;
            FFTZ_DOUBLE _oi_2 = tv2ii - tv2rr;
            out_h1_r[out_strides[2]] = _or_2 * _twr - _oi_2 * _twi;
            out_h1_i[out_strides[2]] = _or_2 * _twi + _oi_2 * _twr;
        }
        // Output point 5: X(4)
        FFTZ_DOUBLE _or_4 = tv1rr + tv1ii;
        {
            FFTZ_DOUBLE _twr = tw_ptr[3 * DATA_STRIDE];
            FFTZ_DOUBLE _twi = tw_ptr[3 * DATA_STRIDE + 1];
            FFTZ_DOUBLE _oi = tv2ii + tv2rr;
            out_h2_r[out_strides[4]] = _or_4 * _twr - _oi * _twi;
            out_h2_i[out_strides[4]] = _or_4 * _twi + _oi * _twr;
        }

        in_h1_r += v_in_stride;
        in_h2_r += v_in_h2_stride;
        in_h1_i += v_in_stride;
        in_h2_i += v_in_h2_stride;
        out_h1_r += v_out_stride;
        out_h2_r += v_out_h2_stride;
        out_h1_i += v_out_stride;
        out_h2_i += v_out_h2_stride;

        tw_ptr += load_multi_cols * (RADIX - 1) * DATA_STRIDE;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

static FFTZ_VOID twid_c2r_fft6c_fp32(FFTZ_VOID *in_real, FFTZ_VOID *in_imag,
                                     FFTZ_VOID *out_real, FFTZ_VOID *out_imag,
                                     FFTZ_INTP n, aoclfftz_strides_t *strides,
                                     FFTZ_VOID *twd, FFTZ_UINT8 flag)
{
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Enter");
    const FFTZ_FLOAT CRTM_6_1 =
        +0.500000000000000000000000000000000000000000000f;
    const FFTZ_FLOAT CRTM_6_2 =
        +0.866025403784438646763723170752936183471402627f;

    FFTZ_FLOAT *in_h1_r, *in_h2_r, *in_h1_i, *in_h2_i, *out_h1_r, *out_h2_r,
        *out_h1_i, *out_h2_i;
#ifdef VOLATILE_STRIDE_ARRAY
    volatile FFTZ_INTP *in_strides = strides->in_strides;
    volatile FFTZ_INTP *out_strides = strides->out_strides;
#else
    FFTZ_INTP *in_strides = strides->in_strides;
    FFTZ_INTP *out_strides = strides->out_strides;
#endif
    FFTZ_INTP v_in_stride = strides->v_in_stride;
    FFTZ_INTP v_in_h2_stride = strides->v_in_h2_stride;
    FFTZ_INTP v_out_stride = strides->v_out_stride;
    FFTZ_INTP v_out_h2_stride = strides->v_out_h2_stride;
    FFTZ_INTP cnt;

    aoclfftz_twiddle_t *tws = (aoclfftz_twiddle_t *)twd;
    FFTZ_FLOAT *tw = (FFTZ_FLOAT *)(tws->TW);
    FFTZ_UINTP load_multi_cols = tws->load_multi_cols;

    in_h1_r = (FFTZ_FLOAT *)in_real;
    in_h2_r = in_h1_r;
    in_h1_i = (FFTZ_FLOAT *)in_imag;
    in_h2_i = in_h1_i;
    out_h1_r = (FFTZ_FLOAT *)out_real;
    out_h2_r = out_h1_r;
    out_h1_i = (FFTZ_FLOAT *)out_imag;
    out_h2_i = out_h1_i;

    FFTZ_FLOAT *tw_ptr = tw;

    for (cnt = 0; cnt < n; cnt++)
    {
        FFTZ_FLOAT v1r, v1i, v2r, v2i, v3r, v3i, v4r, v4i, v5r, v5i, v6r, v6i,
            av1rr, av2rr, av3rr, av4rr, av5rr, av6rr, av7rr, av8rr, tv1rr,
            tv2rr, av1ii, av2ii, av3ii, av4ii, av5ii, av6ii, av7ii, av8ii,
            tv2ii, tv1ii;
        // Input point 1: x(0)
        v1r = *in_h1_r;
        v1i = *in_h1_i;

        // Input point 2: x(1)
        FFTZ_FLOAT v2r_t = in_h1_r[in_strides[1]];
        FFTZ_FLOAT v2i_t = in_h1_i[in_strides[1]];
        v2r = v2r_t;
        v2i = v2i_t;

        // Input point 3: x(2)
        FFTZ_FLOAT v3r_t = in_h1_r[in_strides[2]];
        FFTZ_FLOAT v3i_t = in_h1_i[in_strides[2]];
        v3r = v3r_t;
        v3i = v3i_t;

        // Input point 4: x(3)
        FFTZ_FLOAT v4r_t = in_h2_r[in_strides[3]];
        FFTZ_FLOAT v4i_t = in_h2_i[in_strides[3]];
        v4i_t = -v4i_t;
        v4r = v4r_t;
        v4i = v4i_t;

        // Input point 5: x(4)
        FFTZ_FLOAT v5r_t = in_h2_r[in_strides[4]];
        FFTZ_FLOAT v5i_t = in_h2_i[in_strides[4]];
        v5i_t = -v5i_t;
        v5r = v5r_t;
        v5i = v5i_t;

        // Input point 6: x(5)
        FFTZ_FLOAT v6r_t = in_h2_r[in_strides[5]];
        FFTZ_FLOAT v6i_t = in_h2_i[in_strides[5]];
        v6i_t = -v6i_t;
        v6r = v6r_t;
        v6i = v6i_t;

        // Common calculations -> real
        av1rr = v1r + v4r;
        av2rr = v2r + v6r;
        av3rr = v3r + v5r;
        av4rr = v1r - v4r;
        av5rr = v2r - v6r;
        av6rr = v3r - v5r;
        av7rr = av3rr - av2rr;
        av8rr = av3rr + av2rr;

        // Common calculations -> imaginary
        av1ii = v1i + v4i;
        av2ii = v2i + v6i;
        av3ii = v3i + v5i;
        av4ii = v1i - v4i;
        av5ii = v2i - v6i;
        av6ii = v3i - v5i;
        av7ii = av3ii - av2ii;
        av8ii = av3ii + av2ii;

        // Output point 1: X(0)
        *out_h1_r = av1rr + av8rr;
        *out_h1_i = av1ii + av8ii;
        // Output point 4: X(3)
        FFTZ_FLOAT _or_3 = av4rr + av7rr;
        {
            FFTZ_FLOAT _twr = tw_ptr[2 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[2 * DATA_STRIDE + 1];
            FFTZ_FLOAT _oi = av4ii + av7ii;
            out_h2_r[out_strides[3]] = _or_3 * _twr - _oi * _twi;
            out_h2_i[out_strides[3]] = _or_3 * _twi + _oi * _twr;
        }

        // Common values for X(1) && X(5)
        tv1rr = av4rr - av7rr * CRTM_6_1;
        tv1ii = (av5ii + av6ii) * CRTM_6_2;
        tv2ii = av4ii - av7ii * CRTM_6_1;
        tv2rr = (av6rr + av5rr) * CRTM_6_2;

        // Output point 2: X(1)
        {
            FFTZ_FLOAT _twr = tw_ptr[0];
            FFTZ_FLOAT _twi = tw_ptr[1];
            FFTZ_FLOAT _or_1 = tv1rr - tv1ii;
            FFTZ_FLOAT _oi_1 = tv2ii + tv2rr;
            out_h1_r[out_strides[1]] = _or_1 * _twr - _oi_1 * _twi;
            out_h1_i[out_strides[1]] = _or_1 * _twi + _oi_1 * _twr;
        }
        // Output point 6: X(5)
        FFTZ_FLOAT _or_5 = tv1rr + tv1ii;
        {
            FFTZ_FLOAT _twr = tw_ptr[4 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[4 * DATA_STRIDE + 1];
            FFTZ_FLOAT _oi = tv2ii - tv2rr;
            out_h2_r[out_strides[5]] = _or_5 * _twr - _oi * _twi;
            out_h2_i[out_strides[5]] = _or_5 * _twi + _oi * _twr;
        }

        // Common values for X(2) && X(4)
        tv1rr = av1rr - av8rr * CRTM_6_1;
        tv1ii = (av5ii - av6ii) * CRTM_6_2;
        tv2ii = av1ii - av8ii * CRTM_6_1;
        tv2rr = (av6rr - av5rr) * CRTM_6_2;

        // Output point 3: X(2)
        {
            FFTZ_FLOAT _twr = tw_ptr[DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[DATA_STRIDE + 1];
            FFTZ_FLOAT _or_2 = tv1rr - tv1ii;
            FFTZ_FLOAT _oi_2 = tv2ii - tv2rr;
            out_h1_r[out_strides[2]] = _or_2 * _twr - _oi_2 * _twi;
            out_h1_i[out_strides[2]] = _or_2 * _twi + _oi_2 * _twr;
        }
        // Output point 5: X(4)
        FFTZ_FLOAT _or_4 = tv1rr + tv1ii;
        {
            FFTZ_FLOAT _twr = tw_ptr[3 * DATA_STRIDE];
            FFTZ_FLOAT _twi = tw_ptr[3 * DATA_STRIDE + 1];
            FFTZ_FLOAT _oi = tv2ii + tv2rr;
            out_h2_r[out_strides[4]] = _or_4 * _twr - _oi * _twi;
            out_h2_i[out_strides[4]] = _or_4 * _twi + _oi * _twr;
        }

        in_h1_r += v_in_stride;
        in_h2_r += v_in_h2_stride;
        in_h1_i += v_in_stride;
        in_h2_i += v_in_h2_stride;
        out_h1_r += v_out_stride;
        out_h2_r += v_out_h2_stride;
        out_h1_i += v_out_stride;
        out_h2_i += v_out_h2_stride;

        tw_ptr += load_multi_cols * (RADIX - 1) * DATA_STRIDE;
    }
    AOCLFFTZ_LOG(DEBUG, global_logger_mode, "Exit");
}

kfft_ register_kernel_twid_c2r_fft6c(FFTZ_UINT8 precision,
                                     FFTZ_UINT8 direction /* unused */)
{
    if (precision == DT_FLOAT)
    {
        return twid_c2r_fft6c_fp32;
    }
    else if (precision == DT_DOUBLE)
    {
        return twid_c2r_fft6c_fp64;
    }
    else
    {
        return NULL;
    }
}

#undef RADIX
